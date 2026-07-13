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
    REVERSE_TO_INTERSECTION = 9
    T_JUNCTION_STOP = 10

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
        self.obstacle_approach_offset = 0.15 # 10cm gap to leave between robot and obstacle
        self.target_reverse_distance = 0.05  # How far to push forward on the final exit

        # --- 3. Speeds & Durations ---
        self.obstacle_angular_speed = 0.8    # Speed when rotating to face obstacle (rad/s)
        self.obstacle_square_turn_deg = 90.0  # ALWAYS turn exactly this many degrees toward the obstacle (direction from lidar), instead of the exact measured bearing - keeps the approach square-on to the board instead of diagonal.
        self.obstacle_turn_calibration = 1.05  # Correction multiplier for the OPEN-LOOP timed turn (duration = angle / obstacle_angular_speed). The robot's real angular velocity rarely matches obstacle_angular_speed exactly (wheel slip, odom calibration, control loop latency), so a small consistent over/under-rotation (e.g. ~1 deg) is normal. If it consistently turns SHORT of 90, increase this slightly (e.g. 1.01). If it consistently overshoots, decrease it (e.g. 0.99). Tune in small ~0.005 steps.
        self.obstacle_angle_tolerance_near_dist = 0.15   # meters - your closest sensing distance
        self.obstacle_angle_tolerance_near_deg = 40.0    # degrees - abeam tolerance needed at that close distance
        self.obstacle_angle_tolerance_far_dist = 0.50    # meters - distance by which tolerance has dropped to its tightest
        self.obstacle_angle_tolerance_far_deg = 5.0      # degrees - abeam tolerance needed at/beyond that distance (drops sharply from near to far - see get_angle_tolerance_for_distance(), which fits a power-law curve through these two points instead of a straight line, since the bearing sweeps roughly as 1/distance near the crossing point)
        self.obstacle_search_forward_speed = 0.05   # m/s - slow forward creep after barcode 2 while lidar scans left/right for the obstacle
        self.obstacle_search_timeout = 8.0          # seconds - safety stop if lidar never reports an obstacle while creeping forward

        # --- 3.5. Cruise & Turn Speeds ---
        self.cruise_base_speed = 0.15        # Max forward speed on straightaways (m/s)
        self.cruise_min_speed = 0.05         # Minimum speed during sharp line-following turns
        self.speed_penalty_factor = 0.003    # How aggressively to brake when the line is off-center
        self.turn_90_speed = 1.0             # Angular speed for hard 90-degree intersection turns
        self.delay_p_gain = 0.04             # (No longer used for TURN_DELAY steering - kept in case you want line-tracking back during the delay. See turn_delay_straight_speed below.)
        self.delay_d_gain = 0.08             # (No longer used for TURN_DELAY steering - see above.)
        self.turn_delay_straight_speed = 0.10  # m/s - constant straight-ahead speed during TURN_DELAY (zero steering, no line tracking). Prevents randomly veering onto whichever branch is under the sensor at a T-junction before the deliberate EXECUTE_TURN.

        # --- 4. Failsafe Timers ---
        self.line_timeout = 0.5              # Seconds without seeing line before stopping
        self.maneuver_timeout = 4.0          # Max seconds allowed for Approach/Reverse to prevent wheel slip
        self.topple_wait_duration = 20.0      # Max seconds to wait for topple confirmation
        self.topple_settle_duration = 1.0    # Seconds to wait before trusting the lidar recheck
        self.topple_recheck_interval = 0.5   # Seconds between lidar recheck pings
        self.near_turn_obstacle_threshold = 0.7  # (Currently UNUSED - retrace-to-intersection now always runs after any turn, per your request. Kept here in case you want to bring back the distance-based straightaway/spur classification later.)

        # --- 5. T-Junction Safety ---
        # True = Stop at un-barcoded T-Junctions. False = Attempt to cross straight.
        self.stop_at_unbarcoded_intersections = True
        self.t_junction_wait_start = 0.0
        self.t_junction_slowdown_threshold = 0.15  # seconds since line lost before easing off speed (must be < the 0.30s T-junction confirmation time below)
        self.t_junction_slowdown_speed = 0.05      # m/s - forward speed cap once slowdown_threshold is passed, so the robot eases down instead of cruising at full speed right up until the abrupt stop

        # --- 6. Competition Day Overrides ---
        # True = Robot reverses all the way back to the intersection after toppling an obstacle.
        # False = Robot just drops back onto the line and continues forward.
        self.enable_strict_retrace = False
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
        self.cross_intersection_duration = 0.3  # seconds to drive straight through a no-barcode intersection
        self.turn_start_time = 0.0

        # --- Timers ---
        self.last_line_seen_time = time.time()

        # --- Last-Turn Tracking (for deciding straightaway vs. spur-off-intersection) ---
        self.last_turn_x = 0.0
        self.last_turn_y = 0.0
        self.last_turn_direction = "NONE"  # persists after intersection_memory resets to NONE
        self.obstacle_needs_full_retrace = False
        self.obstacle_dist_from_last_turn = 0.0
        self.retrace_start_x = 0.0
        self.retrace_start_y = 0.0

        # --- Loop Timing Instrumentation ---
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
        self.obstacle_locked = False  # True once the abeam (~90 deg) reading is locked in and creeping stops
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

    def get_angle_tolerance_for_distance(self, distance_m):
        """Fit tolerance = a * D^(-p) exactly through the two calibration
        points (near_dist,near_deg) and (far_dist,far_deg), instead of a
        straight line. This matches the physics: near the abeam crossing,
        the bearing sweeps at a rate roughly proportional to 1/distance,
        so the angular buffer needed to reliably catch a scan within the
        window shrinks sharply at first and then flattens out - a power
        law, not linear. Clamped to the calibrated near/far range."""
        near_d, near_deg = self.obstacle_angle_tolerance_near_dist, self.obstacle_angle_tolerance_near_deg
        far_d, far_deg = self.obstacle_angle_tolerance_far_dist, self.obstacle_angle_tolerance_far_deg
        d = max(near_d, min(far_d, distance_m))

        if near_d <= 0 or far_d <= 0 or near_deg <= 0 or far_deg <= 0 or near_d == far_d:
            return near_deg  # degenerate calibration - fall back safely

        p = (math.log(near_deg) - math.log(far_deg)) / (math.log(far_d) - math.log(near_d))
        a = near_deg * (near_d ** p)
        return a * (d ** -p)

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
        """Handles state hijacking for Left and Right turns."""
        self.set_next_turn(direction)
        self.intersection_detected = False

        if self.current_state in (RobotState.CROSS_INTERSECTION, RobotState.T_JUNCTION_STOP):
            self.current_state = RobotState.EXECUTE_TURN  # Skip the delay, turn instantly!
            self.turn_start_time = time.time()
            self.cross_intersection_start_time = 0.0
            self.t_junction_wait_start = 0.0
            self.get_logger().info(f'BARCODE {barcode_id} seen at junction! Executing immediate {direction} turn.')
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
        # Response from a lidar recheck ping (only meaningful while we're actively waiting for one in HANDLE_OBSTACLE).
        if self.current_state != RobotState.HANDLE_OBSTACLE or not self.topple_awaiting_response:
            return

        self.topple_awaiting_response = False
        if not msg.data:
            self.topple_confirmed_clear = True
            self.get_logger().info('Lidar recheck: obstacle NO LONGER detected — topple confirmed.')
        else:
            self.get_logger().info('Lidar recheck: obstacle still detected — will check again shortly.')

    def obstacle_info_callback(self, msg):
        # Keep updating continuously while creeping forward in
        # SEARCH_OBSTACLE - the board is parallel to the line, so the FIRST
        # reading (board still ahead-and-to-the-side) way overstates the
        # true perpendicular distance. We only stop updating once the main
        # loop locks it in (bearing has swung around to ~90 deg, i.e. the
        # robot is now directly abeam the board).
        if self.current_state != RobotState.SEARCH_OBSTACLE or self.obstacle_locked:
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
                # NEW: Hijack the robot if it's in the intersection OR stopped at a T-Junction
                if self.current_state in (RobotState.CROSS_INTERSECTION, RobotState.T_JUNCTION_STOP):
                    self.get_logger().info('BARCODE 2 seen at junction! Executing obstacle search immediately.')
                    self.cross_intersection_start_time = 0.0
                    self.t_junction_wait_start = 0.0
                else:
                    self.get_logger().info('BARCODE 2: Obstacle flagged. Stopping and arming lidar search...')

                self.current_state = RobotState.SEARCH_OBSTACLE
                self.obstacle_search_entry_time = time.time()
                self.obstacle_info_received = False
                self.obstacle_locked = False
                self.obstacle_rotate_start_time = 0.0
                self.returned_to_front = False
                self.topple_wait_start = 0.0
                self.topple_last_recheck_time = 0.0
                self.topple_awaiting_response = False
                self.topple_confirmed_clear = False
                self.cross_intersection_start_time = 0.0 # Reset crossing timer just in case
                self.get_logger().info('BARCODE 2: Obstacle flagged. Creeping forward while lidar searches...')

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
                self.intersection_detected = False
                self.get_logger().info('Intersection flagged. Checking if line ends (T-Junction)...')
                self.current_state = RobotState.CROSS_INTERSECTION

            elif time_since_last_line > self.line_timeout:
                twist = self.get_stop_twist("No line found for 0.5s - stopping to avoid going off course.")
            else:
                twist = self.calculate_line_follow_twist(self.master_p_gain, self.master_d_gain, "FOLLOW_LINE")

        elif self.current_state == RobotState.REVERSE_TO_INTERSECTION:
            # Reverse the same distance we'd traveled since the last turn,
            # putting us back roughly at the intersection. Then hand off to
            # EXECUTE_TURN with the direction inverted, so the same turn
            # logic (which stops turning once the line is seen again) undoes the original turn.
            dist_reversed = self.get_distance_from(self.retrace_start_x, self.retrace_start_y)

            if dist_reversed < self.obstacle_dist_from_last_turn:
                twist.linear.x = -0.10
                self.get_logger().info(
                    f'Reversing to intersection: {dist_reversed:.2f}m / {self.obstacle_dist_from_last_turn:.2f}m',
                    throttle_duration_sec=0.3)
            else:
                self.get_logger().info('Back at intersection - executing reverse-turn to resume main line.')
                self.intersection_memory = "RIGHT" if self.last_turn_direction == "LEFT" else "LEFT"
                self.last_turn_direction = "NONE"  # consumed
                self.current_state = RobotState.EXECUTE_TURN
                self.turn_start_time = time.time()

            self.total_compute_time += time.perf_counter() - loop_start
            self.vel_pub.publish(twist)
            return

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

            # --- NEW T-JUNCTION TRIGGER: line ends ---
            elif time_since_last_line > 0.30:
                self.get_logger().info('Line ended while seeking! (T-Junction). Turning instantly.')
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

            # --- NEW T-JUNCTION TRIGGER: line ends ---
            elif time_since_last_line > 0.30:
                self.get_logger().info('Line ended in delay zone! (T-Junction). Turning instantly.')
                self.current_state = RobotState.EXECUTE_TURN
                self.turn_start_time = time.time()

            else:
                # Drive straight ahead, no line-tracking steering - we
                # already got the vision trigger, so blindly following
                # line_error here can pull us onto whichever branch is
                # under the sensor at a T-junction before the deliberate
                # EXECUTE_TURN gets a chance to run.
                twist = Twist()
                twist.linear.x = self.turn_delay_straight_speed
                twist.angular.z = 0.0
                self.get_logger().info(
                    f'[TURN_DELAY] Driving straight (no steering) at {self.turn_delay_straight_speed} m/s '
                    f'while waiting out the {self.turn_delay_distance}m delay...',
                    throttle_duration_sec=0.3)

        elif self.current_state == RobotState.EXECUTE_TURN:
            if self.intersection_memory == "LEFT":
                twist.angular.z = self.turn_90_speed
            elif self.intersection_memory == "RIGHT":
                twist.angular.z = -self.turn_90_speed

            self.get_logger().info(
                f'[TROUBLESHOOT] EXECUTE_TURN: elapsed={time.time() - self.turn_start_time:.2f}s/1.1s '
                f'time_since_last_line={time_since_last_line:.2f}s',
                throttle_duration_sec=0.3)

            if (time.time() - self.turn_start_time) > 1.4:
                if time_since_last_line < 0.2:
                    self.get_logger().info('90-Degree line acquired! Resuming cruise.')
                    # Remember where/which way we just turned, in case an
                    # obstacle shows up shortly after (spur-off-intersection
                    # case) and we need to retrace + reverse-turn later.
                    self.last_turn_direction = self.intersection_memory
                    self.last_turn_x = self.current_x
                    self.last_turn_y = self.current_y
                    self.intersection_memory = "NONE"
                    self.current_state = RobotState.FOLLOW_LINE

            self.total_compute_time += time.perf_counter() - loop_start
            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.SEARCH_OBSTACLE:
            elapsed_since_stop_trigger = time.time() - self.obstacle_search_entry_time
            current_angle_tolerance = self.get_angle_tolerance_for_distance(self.obstacle_last_known_distance)
            angle_is_abeam = (self.obstacle_info_received and
                               abs(abs(self.obstacle_target_angle_deg) - 90.0) <= current_angle_tolerance)

            if not angle_is_abeam:
                # --- PHASE 1+2: Creep forward while lidar scans left/right
                # for the obstacle. No stopping - lidar.py is already armed
                # and scanning the full 180 deg (both sides) continuously.
                # Since the board is parallel to the line, keep creeping
                # even after the first reading arrives - a reading taken
                # while the board is still ahead-and-to-the-side reports a
                # longer diagonal distance than the true perpendicular gap.
                # Only lock in once the bearing has swung around to ~90 deg
                # (robot directly abeam the board).
                if elapsed_since_stop_trigger > self.obstacle_search_timeout:
                    twist = self.get_stop_twist(
                        f'SEARCH_OBSTACLE: no obstacle found after {self.obstacle_search_timeout}s - stopping (safety).')
                else:
                    twist.linear.x = self.obstacle_search_forward_speed
                    self.get_logger().info(
                        f'SEARCH_OBSTACLE: creeping forward, scanning for obstacle... '
                        f'({elapsed_since_stop_trigger:.1f}s/{self.obstacle_search_timeout}s, '
                        f'last_angle={self.obstacle_target_angle_deg:.1f}deg, '
                        f'last_dist={self.obstacle_last_known_distance:.2f}m, '
                        f'tolerance={current_angle_tolerance:.1f}deg)',
                        throttle_duration_sec=0.5)

            else:
                # --- PHASE 3: Rotate on the spot toward the obstacle ---
                if self.obstacle_rotate_start_time == 0.0:
                    self.obstacle_locked = True  # freeze the reading - stop updating from new lidar messages
                    self.obstacle_rotate_start_time = time.time()
                    # Always turn a fixed 90 deg (direction from lidar's
                    # LEFT/RIGHT recommendation), not the exact measured
                    # bearing - keeps the approach square-on to the board's
                    # face instead of driving at it diagonally.
                    angle_rad = math.radians(self.obstacle_square_turn_deg)
                    self.obstacle_rotate_duration = (angle_rad / self.obstacle_angular_speed) * self.obstacle_turn_calibration
                    self.get_logger().info(
                        f'Abeam obstacle locked in at dist={self.obstacle_last_known_distance:.2f}m '
                        f'angle={self.obstacle_target_angle_deg:.1f}deg. Rotating '
                        f'{"LEFT" if self.obstacle_recommended_dir > 0 else "RIGHT"} '
                        f'{self.obstacle_square_turn_deg:.0f} deg ({self.obstacle_rotate_duration:.2f}s) to square up.')

                    # Classify straightaway vs. spur-off-intersection NOW,
                    # at the exact spot the robot stopped creeping forward -
                    # not back at the barcode-2 trigger point, since it may
                    # have crept some extra distance forward since then.
                    if self.last_turn_direction != "NONE":
                        self.obstacle_dist_from_last_turn = self.get_distance_from(self.last_turn_x, self.last_turn_y)
                        self.obstacle_needs_full_retrace = True
                    else:
                        self.obstacle_dist_from_last_turn = 0.0
                        self.obstacle_needs_full_retrace = False
                    self.get_logger().info(
                        f'Obstacle classification: dist_from_last_turn={self.obstacle_dist_from_last_turn:.2f}m, '
                        f'needs_full_retrace={self.obstacle_needs_full_retrace}')

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
                    # The board runs PARALLEL to the original line, and we
                    # now always turn a fixed 90 deg (not the exact measured
                    # bearing) - so the raw lidar distance is the diagonal
                    # to wherever the board was first detected, not the true
                    # gap remaining after squaring up. Use the perpendicular
                    # (lateral) component instead: D * sin(bearing) is the
                    # actual distance to close once facing the board square-on.
                    perpendicular_distance = self.obstacle_last_known_distance * math.sin(
                        math.radians(abs(self.obstacle_target_angle_deg)))
                    self.target_approach_distance = perpendicular_distance - self.obstacle_approach_offset
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
                    self.obstacle_locked = False
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
                if self.enable_strict_retrace and self.obstacle_needs_full_retrace:
                    self.get_logger().info(
                        f'Obstacle was near a turn ({self.obstacle_dist_from_last_turn:.2f}m) - '
                        f'reversing back to the intersection to undo the turn.')
                    self.current_state = RobotState.REVERSE_TO_INTERSECTION
                    self.retrace_start_x = self.current_x
                    self.retrace_start_y = self.current_y
                else:
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

            # --- TRUE T-JUNCTION DEFINITION: NO LINE AHEAD ---
            if time_since_last_line > 0.30:
                if self.stop_at_unbarcoded_intersections:
                    self.get_logger().info('Line ended! Confirmed T-Junction. Stopping safely.')
                    self.current_state = RobotState.T_JUNCTION_STOP
                    self.cross_intersection_start_time = 0.0
                    twist = self.get_stop_twist()
                    self.vel_pub.publish(twist)
                    return

            twist = self.calculate_line_follow_twist(self.master_p_gain, self.master_d_gain, "CROSS_INTERSECTION")

            # Ease off speed once the line's been missing for a bit, instead
            # of cruising at full speed right up until the abrupt stop at
            # the 0.30s T-junction confirmation above.
            if time_since_last_line > self.t_junction_slowdown_threshold:
                twist.linear.x = min(twist.linear.x, self.t_junction_slowdown_speed)
                self.get_logger().info(
                    f'[TROUBLESHOOT] Easing off speed, line missing {time_since_last_line:.2f}s '
                    f'(slowdown_speed={self.t_junction_slowdown_speed})',
                    throttle_duration_sec=0.3)

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

        elif self.current_state == RobotState.T_JUNCTION_STOP:
            if self.t_junction_wait_start == 0.0:
                self.t_junction_wait_start = time.time()

            elapsed = time.time() - self.t_junction_wait_start

            # Do not freeze here indefinitely and fail the "shortest time" rule.
            if elapsed > 2.0:
                self.get_logger().warn('T-Junction timeout (2s) reached. Forcing straight cross to comply with rules.')
                self.current_state = RobotState.CROSS_INTERSECTION
                self.cross_intersection_start_time = time.time()
                self.stop_at_unbarcoded_intersections = False # Temporarily bypass the stop flag to force the cross
                self.t_junction_wait_start = 0.0
                twist = self.get_stop_twist("Timeout reached, preparing to force cross.")
            else:
                twist = self.get_stop_twist(f"T-Junction, no barcode. Waiting... ({elapsed:.1f}s/2.0s)")

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
        for _ in range(10):
            self.vel_pub.publish(twist)
            time.sleep(0.02)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotBrain()
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
        node.emergency_stop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
