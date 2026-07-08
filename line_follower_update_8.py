#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32, Bool, Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
from enum import Enum
import time
import math
import signal
import sys

class RobotState(Enum):
    FOLLOW_LINE = 1
    EXECUTE_TURN = 2
    HANDLE_OBSTACLE = 3
    CROSS_INTERSECTION = 4
    COURSE_COMPLETE = 5
    SEEK_INTERSECTION = 6
    SEARCH_OBSTACLE = 7
    TURN_DELAY = 8

class TurtleBotBrain(Node):
    def __init__(self):
        super().__init__('turtlebot_brain')

        # =========================================================
        # THE TUNING DASHBOARD
        # Modify these values to tune the robot's physical behavior
        # =========================================================

        # --- 1. Line Following & Steering ---
        self.master_p_gain = 0.01         # Steering aggressiveness
        self.master_d_gain = 0.19         # Shock absorber for wiggles
        self.camera_center = 39.0         # Pixy camera center line

        # --- 2. Distances (in meters) ---
        self.turn_distance_threshold = 0.30  # Distance from barcode to 90-deg turn point
        self.turn_delay_distance = 0.15      # 15cm delay after crossing intersection before turning
        self.obstacle_approach_offset = 0.20 # 10cm gap to leave between robot and obstacle
        self.target_reverse_distance = 0.05  # How far to push forward on the final exit

        # --- 3. Speeds & Durations ---
        self.obstacle_angular_speed = 0.8    # Speed when rotating to face obstacle (rad/s)
        self.obstacle_stop_duration = 2.5    # Seconds to fully stop before awaiting lidar

        # --- 3.5. Cruise & Turn Speeds ---
        self.cruise_base_speed = 0.15        # Max forward speed on straightaways (m/s)
        self.cruise_min_speed = 0.05         # Minimum speed during sharp line-following turns
        self.speed_penalty_factor = 0.003    # How aggressively to brake when the line is off-center
        self.turn_90_speed = 1.2             # Angular speed for hard 90-degree intersection turns
        self.delay_p_gain = 0.04             # Gentle steering P-gain while waiting for turn distance
        self.delay_d_gain = 0.08             # Gentle steering D-gain while waiting for turn distance

        # --- 4. Failsafe Timers ---
        self.line_timeout = 0.5              # Seconds without seeing line before stopping
        self.maneuver_timeout = 4.0          # Max seconds allowed for Approach/Reverse to prevent wheel slip
        self.topple_wait_duration = 20.0      # Max seconds to wait for topple confirmation
        self.topple_settle_duration = 1.0    # Seconds to wait before trusting the lidar recheck
        self.topple_recheck_interval = 0.5   # Seconds between lidar recheck pings
        # =========================================================

        # --- System State Variables ---
        self.current_state = RobotState.FOLLOW_LINE
        self.last_barcode_seen = -1
        self.line_error = 0.0
        self.last_line_error = 0.0
        self.intersection_detected = False
        self.estopped = False

        # --- Dynamic Memory Variables ---
        self.intersection_memory = "NONE"
        self.cross_intersection_start_time = 0.0
        self.cross_intersection_duration = 0.8  # seconds to drive straight through a no-barcode intersection
        self.turn_start_time = 0.0

        # --- Timers ---
        self.last_line_seen_time = time.time()

        # --- Loop Timing Instrumentation ---
        # Used to measure how fast fsm_loop actually runs (vs the nominal
        # 30Hz from create_timer) and how long each line-tracing adjustment
        # takes to compute, so you can work out adjustments/sec.
        self.loop_count = 0
        self.timing_window_start = time.time()
        self.last_loop_start = time.time()
        self.total_compute_time = 0.0

        # --- Odometry Variables ---
        self.current_x = 0.0
        self.current_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.just_entered_reverse = False
        self.delay_start_x = 0.0
        self.delay_start_y = 0.0

        # --- Intersection Search Safety ---
        self.search_start_x = 0.0
        self.search_start_y = 0.0
        self.search_distance_limit = 0.6

        # --- Obstacle Search (Lidar) Settings ---
        self.obstacle_search_entry_time = 0.0
        self.obstacle_info_received = False
        self.obstacle_recommended_dir = 0.0   # +1.0 = LEFT, -1.0 = RIGHT (from lidar_locator)
        self.obstacle_target_angle_deg = 0.0  # degrees to rotate to face the obstacle
        self.obstacle_last_known_distance = 0.0  # meters - used as the reference band for the final recheck
        self.obstacle_rotate_start_time = 0.0
        self.obstacle_rotate_duration = 0.0

        self.approached_obstacle = False
        self.approach_start_x = 0.0
        self.approach_start_y = 0.0
        self.target_approach_distance = 0.0

        # --- Topple / Return-to-Front Settings ---
        self.topple_wait_start = 0.0
        self.returned_to_front = False
        self.return_rotate_start_time = 0.0

        # --- Topple Lidar-Confirmation Settings ---
        self.topple_last_recheck_time = 0.0
        self.topple_awaiting_response = False
        self.topple_confirmed_clear = False
        self.topple_done_received = False

        # --- Publishers & Subscribers ---
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.servo_pub = self.create_publisher(Int32, 'topple_trigger', 10)
        self.topple_recheck_pub = self.create_publisher(Float32, 'topple_recheck_trigger', 10)

        self.create_subscription(Float32, 'pixy_vector', self.line_callback, 10)
        self.create_subscription(Int32, 'pixy_barcode', self.barcode_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Bool, 'pixy_intersection', self.intersection_callback, 10)
        self.create_subscription(Float32MultiArray, 'obstacle_info', self.obstacle_info_callback, 10)
        self.create_subscription(Bool, 'obstacle_detected', self.obstacle_detected_callback, 10)
        self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Bool, 'topple_done', self.topple_done_callback, 10)

        self.timer = self.create_timer(0.033, self.fsm_loop)
        self.get_logger().info('Brain activated. Press Ctrl+C to Emergency Stop.')

    # ---------------------------------------------------------
    # SUPPORTING FUNCTIONS
    # ---------------------------------------------------------
    def get_stop_twist(self, reason="No reason provided"):
        """Returns a zero-velocity Twist and logs the reason for stopping."""
        self.get_logger().info(f'STOPPING: {reason}', throttle_duration_sec=1.0)

        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        return twist

    def get_distance_from(self, start_x, start_y, debug_msg=None):
        """Calculates 2D distance and optionally prints a debug message."""
        distance = math.sqrt((self.current_x - start_x)**2 + (self.current_y - start_y)**2)

        if debug_msg:
            self.get_logger().info(
                f'{debug_msg} dist={distance:.3f}m',
                throttle_duration_sec=0.5)

        return distance

    def calculate_line_follow_twist(self, p_gain, d_gain, state_name="FOLLOW_LINE"):
        """Calculates and returns a Twist message for line tracing based on provided gains."""
        twist = Twist()

        # 1. Steering
        error_derivative = self.line_error - self.last_line_error
        turn_speed = (-p_gain * self.line_error) + (-d_gain * error_derivative)
        self.last_line_error = self.line_error

        # 2. Dynamic Braking
        speed_penalty = abs(self.line_error) * self.speed_penalty_factor
        forward_speed = max(self.cruise_base_speed - speed_penalty, self.cruise_min_speed)

        twist.linear.x = forward_speed
        twist.angular.z = turn_speed

        # Debug Print (throttled to twice a second)
        self.get_logger().info(
            f'[{state_name}] Calc Line Follow: err={self.line_error:.2f}, '
            f'fwd_spd={forward_speed:.2f}, turn_spd={turn_speed:.2f}',
            throttle_duration_sec=1.0)

        return twist

    def process_directional_barcode(self, direction, barcode_id):
        """Handles state hijacking for Left and Right turns with original debug prints."""
        self.set_next_turn(direction)
        self.intersection_detected = False

        if self.current_state == RobotState.CROSS_INTERSECTION:
            self.current_state = RobotState.TURN_DELAY
            self.delay_start_x, self.delay_start_y = self.current_x, self.current_y
            self.cross_intersection_start_time = 0.0
            self.get_logger().info(f'BARCODE {barcode_id} seen while crossing! Starting 10cm delay before {direction} turn.')
        else:
            self.current_state = RobotState.SEEK_INTERSECTION
            self.search_start_x, self.search_start_y = self.current_x, self.current_y
            self.get_logger().info(f'BARCODE {barcode_id}: Set to {direction}, waiting for distance...')

    # ---------------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------------
    def line_callback(self, msg):
        self.line_error = msg.data - self.camera_center
        self.last_line_seen_time = time.time()

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def scan_callback(self, msg):
        pass # We ignore raw scan data here because lidar.py handles it

    def intersection_callback(self, msg):
        self.intersection_detected = msg.data

    def obstacle_detected_callback(self, msg):
        # Response from a lidar recheck ping (only meaningful while we're
        # actively waiting for one in HANDLE_OBSTACLE).
        if self.current_state != RobotState.HANDLE_OBSTACLE or not self.topple_awaiting_response:
            return

        self.topple_awaiting_response = False
        if not msg.data:
            self.topple_confirmed_clear = True
            self.get_logger().info('Lidar recheck: obstacle NO LONGER detected — topple confirmed.')
        else:
            self.get_logger().info('Lidar recheck: obstacle still detected — will check again shortly.')

    def obstacle_info_callback(self, msg):
        # Only pay attention while actively waiting for it in SEARCH_OBSTACLE,
        # and only take the first reading (avoids being overwritten mid-rotate).
        if self.current_state != RobotState.SEARCH_OBSTACLE or self.obstacle_info_received:
            return

        data = msg.data
        if len(data) >= 3:
            # lidar_locator publishes [distance_m, angle_deg, direction]
            distance_m, angle_deg, direction = data[0], data[1], data[2]

            self.obstacle_recommended_dir = direction
            self.obstacle_target_angle_deg = angle_deg
            self.obstacle_last_known_distance = distance_m
            self.obstacle_info_received = True
            self.get_logger().info(
                f'Lidar obstacle info: dist={distance_m:.2f}m angle={angle_deg:.1f}deg '
                f'dir={"LEFT" if direction > 0 else "RIGHT"}')

    def barcode_callback(self, msg):
       # Allow barcode reads in FOLLOW_LINE AND CROSS_INTERSECTION (in case vision sees the intersection early)
        if self.current_state not in (RobotState.FOLLOW_LINE, RobotState.CROSS_INTERSECTION):
            self.get_logger().info(
                f'[TROUBLESHOOT] Barcode {msg.data} ignored - currently in {self.current_state.name}',
                throttle_duration_sec=0.5)
            return

        barcode_id = msg.data
        if barcode_id == 0:
            self.process_directional_barcode("LEFT", 0)

        elif barcode_id == 1:
            self.process_directional_barcode("RIGHT", 1)

        elif barcode_id == 2:
            self.intersection_detected = False
            if self.current_state not in (RobotState.HANDLE_OBSTACLE, RobotState.SEARCH_OBSTACLE):
                # Hijack the robot if it already started crossing the intersection
                if self.current_state == RobotState.CROSS_INTERSECTION:
                    self.get_logger().info('BARCODE 2 seen while crossing! Executing obstacle search immediately.')
                    self.cross_intersection_start_time = 0.0
                else:
                    self.get_logger().info('BARCODE 2: Obstacle flagged. Stopping and arming lidar search...')

                self.current_state = RobotState.SEARCH_OBSTACLE
                self.obstacle_search_entry_time = time.time()
                self.obstacle_info_received = False
                self.obstacle_rotate_start_time = 0.0
                self.returned_to_front = False
                self.topple_wait_start = 0.0
                self.topple_last_recheck_time = 0.0
                self.topple_awaiting_response = False
                self.topple_confirmed_clear = False
                self.cross_intersection_start_time = 0.0 # Reset crossing timer just in case
                self.get_logger().info('BARCODE 2: Obstacle flagged. Stopping and arming lidar search...')

        elif barcode_id == 3:
            # Wipe the intersection from memory
            self.intersection_detected = False

            # Hijack the robot if it already started crossing the intersection
            if self.current_state == RobotState.CROSS_INTERSECTION:
                self.get_logger().info('BARCODE 3 seen while crossing! Stopping for course complete.')
                self.cross_intersection_start_time = 0.0
            self.current_state = RobotState.COURSE_COMPLETE

    def set_next_turn(self, direction):
        self.intersection_memory = direction
        self.get_logger().info(f'Next turn buffered: {direction}')

    def topple_done_callback(self, msg):
        if msg.data:
            self.topple_done_received = True
            self.get_logger().info('Actuator confirmed: Topple complete.')

    # ---------------------------------------------------------
    # MAIN FSM LOOP
    # ---------------------------------------------------------
    def fsm_loop(self):
        if self.estopped:
            return

        # --- Loop Timing Instrumentation ---
        loop_start = time.perf_counter()
        now = time.time()
        self.last_loop_start = now
        self.loop_count += 1

        # Print a summary once per second: actual achieved rate (Hz) and
        # average interval between adjustments (ms). This tells you how
        # many micro-adjustments per second you're really getting, vs the
        # 30Hz (0.033s) the timer is nominally set to.
        elapsed = now - self.timing_window_start
        if elapsed >= 1.0:
            achieved_hz = self.loop_count / elapsed
            avg_interval_ms = (elapsed / self.loop_count) * 1000.0
            avg_compute_ms = (self.total_compute_time / self.loop_count) * 1000.0
            print(f'[TIMING] {achieved_hz:.1f} adjustments/sec  '
                  f'(avg interval {avg_interval_ms:.1f} ms, '
                  f'avg compute {avg_compute_ms:.3f} ms, '
                  f'{self.loop_count} loops in {elapsed:.2f}s)')
            self.loop_count = 0
            self.timing_window_start = now
            self.total_compute_time = 0.0

        twist = Twist()
        time_since_last_line = time.time() - self.last_line_seen_time

        self.get_logger().info(f'Current Brain State: {self.current_state.name}', throttle_duration_sec=1.0)

        if self.current_state == RobotState.FOLLOW_LINE:

            if self.intersection_detected and self.intersection_memory == "NONE":
                self.get_logger().info(
                    '[TROUBLESHOOT] Intersection detected with no barcode set -> CROSS_INTERSECTION')
                self.intersection_detected = False
                self.current_state = RobotState.CROSS_INTERSECTION

            elif time_since_last_line > self.line_timeout:
                twist = self.get_stop_twist("No line found for 0.5s - stopping to avoid going off course.")
            else:
                twist = self.calculate_line_follow_twist(self.master_p_gain, self.master_d_gain, "FOLLOW_LINE")

        elif self.current_state == RobotState.SEEK_INTERSECTION:
            # Calculate distance since barcode scan
            dist_traveled = self.get_distance_from(self.search_start_x, self.search_start_y, "[SEEK_INTERSECTION]")

            self.get_logger().info(
                f'SEEK: intersection_detected={self.intersection_detected} '
                f'dist={dist_traveled:.2f} threshold={self.turn_distance_threshold:.2f}',
                throttle_duration_sec=0.5)

            # --- PRIMARY TRIGGER: vision (intersection_detected) ---
            if self.intersection_detected:
                self.get_logger().info(f'Vision trigger! (dist={dist_traveled:.2f}m) Starting 10cm delay before turn.')
                self.intersection_detected = False  # consume it so it can't latch

                # Send to TURN_DELAY and snapshot the current position
                self.current_state = RobotState.TURN_DELAY
                self.delay_start_x, self.delay_start_y = self.current_x, self.current_y

            # --- FAILSAFE TRIGGER: distance (in case vision misses it) ---
            elif dist_traveled >= self.turn_distance_threshold:
                self.get_logger().warn(f'FAILSAFE: no vision trigger by {dist_traveled:.2f}m. Triggering turn anyway.')
                self.current_state = RobotState.EXECUTE_TURN
                self.turn_start_time = time.time()
            else:
                # Still cruising to the turn point using PD control
                twist = self.calculate_line_follow_twist(self.master_p_gain, self.master_d_gain, "SEEK_INTERSECTION")

        elif self.current_state == RobotState.TURN_DELAY:
            # Calculate how far we've driven since seeing the intersection
            delay_dist_traveled = self.get_distance_from(self.delay_start_x, self.delay_start_y, "[TURN_DELAY]")

            if delay_dist_traveled >= self.turn_delay_distance:
                self.get_logger().info(f'10cm delay complete (dist={delay_dist_traveled:.2f}m). Executing turn.')
                self.current_state = RobotState.EXECUTE_TURN
                self.turn_start_time = time.time()
            else:
                # Keep following the line forward while we wait for the 10cm to finish
                twist = self.calculate_line_follow_twist(self.delay_p_gain, self.delay_d_gain, "TURN_DELAY")

        elif self.current_state == RobotState.EXECUTE_TURN:
            if self.intersection_memory == "LEFT":
                twist.angular.z = self.turn_90_speed
            elif self.intersection_memory == "RIGHT":
                twist.angular.z = -self.turn_90_speed

            self.get_logger().info(
                f'[TROUBLESHOOT] EXECUTE_TURN: elapsed={time.time() - self.turn_start_time:.2f}s/1.1s '
                f'time_since_last_line={time_since_last_line:.2f}s',
                throttle_duration_sec=0.3)

            if (time.time() - self.turn_start_time) > 1.1:
                if time_since_last_line < 0.2:
                    self.get_logger().info('90-Degree line acquired! Resuming cruise.')
                    self.intersection_memory = "NONE"
                    self.current_state = RobotState.FOLLOW_LINE

            self.total_compute_time += time.perf_counter() - loop_start
            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.SEARCH_OBSTACLE:
            elapsed_since_stop_trigger = time.time() - self.obstacle_search_entry_time

            if elapsed_since_stop_trigger < self.obstacle_stop_duration:
                # --- PHASE 1: Come to a full stop for 2-3s ---
                twist = self.get_stop_twist()
                self.get_logger().info(
                    f'SEARCH_OBSTACLE: stopping... ({elapsed_since_stop_trigger:.1f}s/{self.obstacle_stop_duration}s)',
                    throttle_duration_sec=0.5)

            elif not self.obstacle_info_received:
                # --- PHASE 2: Stopped, awaiting lidar result ---
                twist = self.get_stop_twist("SEARCH_OBSTACLE: stopped, awaiting lidar obstacle_info...")

            else:
                # --- PHASE 3: Rotate on the spot toward the obstacle ---
                if self.obstacle_rotate_start_time == 0.0:
                    self.obstacle_rotate_start_time = time.time()
                    angle_rad = abs(math.radians(self.obstacle_target_angle_deg))
                    if angle_rad > 0.0:
                        self.obstacle_rotate_duration = angle_rad / self.obstacle_angular_speed
                    else:
                        # No angle provided (3-field message) - fall back to a
                        # short fixed rotate just to face toward that side.
                        self.obstacle_rotate_duration = 0.5
                    self.get_logger().info(
                        f'Rotating {"LEFT" if self.obstacle_recommended_dir > 0 else "RIGHT"} '
                        f'for {self.obstacle_rotate_duration:.2f}s to face obstacle.')

                direction = 1.0 if self.obstacle_recommended_dir > 0 else -1.0
                twist.angular.z = self.obstacle_angular_speed * direction

                self.get_logger().info(
                    f'[TROUBLESHOOT] Facing obstacle: '
                    f'{time.time() - self.obstacle_rotate_start_time:.2f}s/{self.obstacle_rotate_duration:.2f}s',
                    throttle_duration_sec=0.3)

                if (time.time() - self.obstacle_rotate_start_time) >= self.obstacle_rotate_duration:
                    self.get_logger().info('Obstacle faced. Handing off to HANDLE_OBSTACLE.')
                    self.current_state = RobotState.HANDLE_OBSTACLE
                    # --- NEW: Setup for approach & reverse ---
                    self.approached_obstacle = False
                    self.approach_start_x = self.current_x
                    self.approach_start_y = self.current_y
                    self.target_approach_distance = self.obstacle_last_known_distance - self.obstacle_approach_offset
                    self.actual_approach_distance = 0.0  # Tracks exactly how far we went

                    self.approach_start_time = 0.0
                    self.reverse_start_time = 0.0
                    self.reversed_to_start = False
                    self.started_reverse_back = False
                    self.reverse_back_start_x = 0.0
                    self.reverse_back_start_y = 0.0

                    # We delay this trigger until the new reverse phase finishes
                    self.just_entered_reverse = False
                    # Reset search state so the next barcode 2 starts fresh
                    self.obstacle_info_received = False
                    self.obstacle_rotate_start_time = 0.0

            self.total_compute_time += time.perf_counter() - loop_start
            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.HANDLE_OBSTACLE:
            # --- PHASE 1: Approach the obstacle to exactly 10cm away ---
            if not self.approached_obstacle:
                if self.target_approach_distance <= 0.0:
                    self.actual_approach_distance = 0.0
                    self.approached_obstacle = True
                else:
                    # NEW: Initialize the approach timer
                    if not hasattr(self, 'approach_start_time') or self.approach_start_time == 0.0:
                        self.approach_start_time = time.time()

                    distance_moved = self.get_distance_from(self.approach_start_x, self.approach_start_y)
                    elapsed_approach = time.time() - self.approach_start_time

                    # NEW: Timeout Failsafe Check
                    if elapsed_approach > self.maneuver_timeout:
                        self.get_logger().warn(f'Approach timeout! Wheel slip? Aborting after {elapsed_approach:.1f}s.')
                        twist = self.get_stop_twist()
                        self.actual_approach_distance = distance_moved
                        self.approached_obstacle = True
                        self.topple_wait_start = 0.0
                    elif distance_moved < self.target_approach_distance:
                        twist.linear.x = 0.10  # Creep forward at 0.1 m/s
                        self.get_logger().info(
                            f'Approaching obstacle: {distance_moved:.2f}m / {self.target_approach_distance:.2f}m',
                            throttle_duration_sec=0.3)
                    else:
                        self.get_logger().info('Reached 10cm from obstacle. Stopping to topple.')
                        twist = self.get_stop_twist()
                        self.actual_approach_distance = distance_moved
                        self.approached_obstacle = True
                        self.topple_wait_start = 0.0

                self.total_compute_time += time.perf_counter() - loop_start
                self.vel_pub.publish(twist)
                return

            # --- PHASE 2: Topple & Confirm ---
            if not self.topple_confirmed_clear:
                if self.topple_wait_start == 0.0:
                    self.topple_wait_start = time.time()
                    self.topple_done_received = False  # Reset flag!
                    trigger = Int32()
                    trigger.data = 1
                    self.servo_pub.publish(trigger)
                    self.get_logger().info('Toppling obstacle. Waiting for actuator and lidar...')

                elapsed_topple = time.time() - self.topple_wait_start

                # WAIT CONDITION: If actuator is NOT done AND we haven't hit the 6s cap, stay here!
                actuator_finished = self.topple_done_received
                time_finished = elapsed_topple >= self.topple_wait_duration

                if not (actuator_finished or time_finished):
                    # GATE: Actuator is still moving. Stay stopped and return early.
                    twist = self.get_stop_twist("Waiting for actuator...")
                    self.vel_pub.publish(twist)
                    return

                # If we reached here, the actuator is done OR the timer expired.
                if time_finished and not actuator_finished:
                    self.get_logger().warn(
                        f'Topple safety cap ({self.topple_wait_duration}s) hit without '
                        f'lidar confirmation. Proceeding anyway.')

                # Settle phase before Lidar recheck
                if elapsed_topple < self.topple_settle_duration:
                    twist = self.get_stop_twist()
                    self.get_logger().info(
                        f'[TROUBLESHOOT] Settling before recheck: '
                        f'{elapsed_topple:.2f}s/{self.topple_settle_duration}s',
                        throttle_duration_sec=0.3)
                    self.vel_pub.publish(twist)
                    return

                # Perform lidar recheck pings
                elif not self.topple_confirmed_clear:
                    if not self.topple_awaiting_response and \
                       (time.time() - self.topple_last_recheck_time) >= self.topple_recheck_interval:
                        recheck_msg = Float32()
                        recheck_msg.data = self.obstacle_last_known_distance
                        self.topple_recheck_pub.publish(recheck_msg)
                        self.topple_awaiting_response = True
                        self.topple_last_recheck_time = time.time()
                        self.get_logger().info(
                            f'[TROUBLESHOOT] Pinging lidar to recheck obstacle '
                            f'(ref_dist={self.obstacle_last_known_distance:.2f}m, '
                            f'elapsed={elapsed_topple:.1f}s)...')

                self.total_compute_time += time.perf_counter() - loop_start
                self.vel_pub.publish(twist)
                return

            # --- PHASE 3: Reverse Back (while still facing the obstacle) ---
            if not self.reversed_to_start:
                if not self.started_reverse_back:
                    self.reverse_back_start_x = self.current_x
                    self.reverse_back_start_y = self.current_y
                    self.started_reverse_back = True
                    # NEW: Start the reverse timer
                    self.reverse_start_time = time.time()
                    self.get_logger().info(f'Starting reverse maneuver for {self.actual_approach_distance:.2f}m')

                distance_reversed = self.get_distance_from(self.reverse_back_start_x, self.reverse_back_start_y)
                elapsed_reverse = time.time() - getattr(self, 'reverse_start_time', time.time())

                # NEW: Timeout Failsafe Check
                if elapsed_reverse > self.maneuver_timeout:
                    self.get_logger().warn(f'Reverse timeout! Wheel slip? Aborting after {elapsed_reverse:.1f}s.')
                    twist = self.get_stop_twist()
                    self.reversed_to_start = True
                elif distance_reversed < self.actual_approach_distance:
                    twist.linear.x = -0.10  # Negative speed to reverse
                    self.get_logger().info(
                        f'Reversing back: {distance_reversed:.2f}m / {self.actual_approach_distance:.2f}m',
                        throttle_duration_sec=0.3)
                else:
                    self.get_logger().info('Reverse complete. Back at original track position.')
                    twist = self.get_stop_twist()
                    self.reversed_to_start = True

                self.total_compute_time += time.perf_counter() - loop_start
                self.vel_pub.publish(twist)
                return

            # --- PHASE 4: Rotate back to face front ---
            if not self.returned_to_front:
                if self.return_rotate_start_time == 0.0:
                    self.return_rotate_start_time = time.time()
                    self.get_logger().info('Rotating back to face front...')

                undo_direction = -1.0 if self.obstacle_recommended_dir > 0 else 1.0
                twist.angular.z = self.obstacle_angular_speed * undo_direction

                self.get_logger().info(
                    f'[TROUBLESHOOT] Rotating back to front: '
                    f'{time.time() - self.return_rotate_start_time:.2f}s/{self.obstacle_rotate_duration:.2f}s',
                    throttle_duration_sec=0.3)

                if (time.time() - self.return_rotate_start_time) >= self.obstacle_rotate_duration:
                    self.get_logger().info('Facing front again. Beginning forward-out.')
                    self.returned_to_front = True
                    self.return_rotate_start_time = 0.0

                    # Prime the final forward-out phase
                    self.just_entered_reverse = True

                self.total_compute_time += time.perf_counter() - loop_start
                self.vel_pub.publish(twist)
                return

            # --- PHASE 5: Forward-Out (Push past the obstacle zone) ---
            if self.just_entered_reverse:
                self.start_x = self.current_x
                self.start_y = self.current_y
                self.just_entered_reverse = False

            distance = self.get_distance_from(self.start_x, self.start_y)
            if distance < self.target_reverse_distance:
                twist.linear.x = 0.1
                self.get_logger().info(
                    f'[TROUBLESHOOT] Moving FORWARD: distance={distance:.3f}m / '
                    f'target={self.target_reverse_distance:.3f}m',
                    throttle_duration_sec=0.3)
            else:
                self.get_logger().info('Forward-out distance reached. Resuming line tracing immediately.')
                self.current_state = RobotState.FOLLOW_LINE

            self.total_compute_time += time.perf_counter() - loop_start
            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.CROSS_INTERSECTION:
            # No turn barcode was set before this intersection - per spec,
            # continue straight ahead (closest to straight), not stop.
            if self.cross_intersection_start_time == 0.0:
                self.cross_intersection_start_time = time.time()
                self.get_logger().info('No barcode set at intersection - crossing straight ahead.')

            twist = self.calculate_line_follow_twist(self.master_p_gain, self.master_d_gain, "CROSS_INTERSECTION")

            self.get_logger().info(
                f'[TROUBLESHOOT] Crossing intersection straight: '
                f'{time.time() - self.cross_intersection_start_time:.2f}s / {self.cross_intersection_duration}s',
                throttle_duration_sec=0.3)

            if (time.time() - self.cross_intersection_start_time) >= self.cross_intersection_duration:
                self.get_logger().info('Intersection crossed. Resuming normal cruise.')
                self.cross_intersection_start_time = 0.0
                self.current_state = RobotState.FOLLOW_LINE

            self.total_compute_time += time.perf_counter() - loop_start
            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.COURSE_COMPLETE:
            twist = self.get_stop_twist()

        self.total_compute_time += time.perf_counter() - loop_start
        self.vel_pub.publish(twist)

    def emergency_stop(self):
        self.get_logger().warn('EMERGENCY STOP TRIGGERED')
        self.estopped = True
        twist = self.get_stop_twist()
        twist.angular.z = 0.0
        # Publish repeatedly with small delays to maximize the chance the
        # zero-velocity command actually reaches the DDS layer / motor
        # driver before the process is torn down.
        for _ in range(10):
            self.vel_pub.publish(twist)
            time.sleep(0.02)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotBrain()

    # Register our own SIGINT (Ctrl+C) handler. rclpy's default handling of
    # SIGINT can tear down the context before your except-KeyboardInterrupt
    # block reliably runs, which sometimes means the "stop" Twist never
    # actually reaches the robot. Handling it directly guarantees the
    # zero-velocity command is published the instant Ctrl+C is pressed.
    def sigint_handler(sig, frame):
        node.emergency_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Fallback path in case SIGINT arrives in a way that still raises
        # here (e.g. between spin cycles) rather than hitting the handler.
        node.emergency_stop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
