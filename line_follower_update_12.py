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
    APPROACH_INTERSECTION = 8
    REVERSE_TO_INTERSECTION = 9
    T_JUNCTION_STOP = 10
    APPROACH_T_JUNCTION = 11

class TurtleBotBrain(Node):
    def __init__(self):
        super().__init__('turtlebot_brain')

        # =========================================================
        # THE TUNING DASHBOARD
        # Modify these values to tune the robot's physical behavior
        # =========================================================

        # --- 1. Line Following & Steering ---
        self.master_p_gain = 0.02         # Steering aggressiveness
        self.master_d_gain = 0.19         # Shock absorber for wiggles
        self.camera_center = 39.0         # Pixy camera center line

        # --- 2. Distances (in meters) ---
        self.turn_distance_threshold = 0.5  # Distance from barcode to 90-deg turn point
        self.turn_delay_distance = 0.15      # 15cm delay after crossing intersection before turning
        self.obstacle_approach_offset = 0.12 # __cm gap to leave between robot and obstacle
        self.target_reverse_distance = 0.05  # How far to push forward on the final exit

        # --- 3. Obstacles related ---
        self.obstacle_align_offset = -0.05   # meters - Negative = stop earlier, Positive = drive further. Use this to physically center the actuator.
        self.obstacle_angular_speed = 0.8    # Speed when rotating to face obstacle (rad/s)
        self.obstacle_square_turn_deg = 90.0  # ALWAYS turn exactly this many degrees toward the obstacle (direction from lidar), instead of the exact measured bearing - keeps the approach square-on to the board instead of diagonal.
        self.obstacle_turn_calibration = 1.04  # Correction multiplier for the OPEN-LOOP timed turn (duration = angle / obstacle_angular_speed). The robot's real angular velocity rarely matches obstacle_angular_speed exactly (wheel slip, odom calibration, control loop latency), so a small consistent over/under-rotation (e.g. ~1 deg) is normal. If it consistently turns SHORT of 90, increase this slightly (e.g. 1.01). If it consistently overshoots, decrease it (e.g. 0.99). Tune in small ~0.005 steps.
        self.obstacle_valid_angle_min = 20.0    # degrees - reject readings closer to straight-ahead than this
        self.obstacle_valid_angle_max = 160.0   # degrees - reject readings closer to directly-behind than this
        self.obstacle_search_forward_speed = 0.1   # m/s - slow forward creep after barcode 2 while lidar scans left/right for the obstacle
        self.obstacle_search_timeout = 8.0          # seconds - safety stop if lidar never reports an obstacle while creeping forward
        self.obstacle_cooldown_time = 0.0

        # --- 3.5. Cruise & Turn Speeds ---
        self.cruise_base_speed = 0.2       # Max forward speed on straightaways (m/s)
        self.cruise_min_speed = 0.05         # Minimum speed during sharp line-following turns
        self.speed_penalty_factor = 0.003    # How aggressively to brake when the line is off-center
        self.turn_90_speed = 1.6             # Angular speed for hard 90-degree intersection turns
        self.delay_p_gain = 0.04             # Gentle steering P-gain used during TURN_DELAY when turn_delay_use_line_tracking is True
        self.delay_d_gain = 0.08             # Gentle steering D-gain used during TURN_DELAY when turn_delay_use_line_tracking is True
        self.turn_delay_use_line_tracking = True  # True = gently follow the line (delay_p_gain/delay_d_gain) during TURN_DELAY, fixing diagonal drift on curved sections. False = drive dead straight (turn_delay_straight_speed) instead. Safe either way - a genuine T-junction (line lost) is already caught by the "line ended" check above and diverted straight to EXECUTE_TURN before this branch ever runs.
        self.turn_delay_straight_speed = 0.10  # m/s - constant straight-ahead speed during TURN_DELAY, used only when turn_delay_use_line_tracking is False.

        # --- 4. Failsafe Timers ---
        self.line_timeout = 0.5              # Seconds without seeing line before stopping
        self.maneuver_timeout = 4.0          # Max seconds allowed for Approach/Reverse to prevent wheel slip
        self.topple_wait_duration = 12.0      # Max seconds to wait for topple confirmation
        self.topple_overall_timeout = 20.0  # seconds - absolute cap on the ENTIRE Phase 2 (actuator wait + settle + repeated lidar rechecks). topple_wait_duration only bounds the actuator wait; without this, a board that genuinely never clears the lidar recheck (didn't topple) would hang the FSM forever. If this fires, force-advance anyway with a warning - better to continue the course than get stuck.
        self.topple_settle_duration = 1.0    # Seconds to wait before trusting the lidar recheck
        self.topple_recheck_interval = 0.5   # Seconds between lidar recheck pings
        self.near_turn_obstacle_threshold = 0.7  # (Currently UNUSED - retrace-to-intersection now always runs after any turn, per your request. Kept here in case you want to bring back the distance-based straightaway/spur classification later.)

        # --- 5. T-Junction Safety ---
        # True = Stop at un-barcoded T-Junctions. False = Attempt to cross straight.
        self.stop_at_unbarcoded_intersections = True
        self.t_junction_wait_start = 0.0
        self.t_junction_slowdown_threshold = 0.15  # seconds since line lost before easing off speed (must be < the 0.30s T-junction confirmation time below)
        self.t_junction_slowdown_speed = 0.05      # m/s - forward speed cap once slowdown_threshold is passed, so the robot eases down instead of cruising at full speed right up until the abrupt stop
        self.turn_delay_force_straight = False     # (No longer set anywhere - T-junction turns now go through their own dedicated T_JUNCTION_TURN_DELAY state instead of hijacking TURN_DELAY. Kept here + the checks below in TURN_DELAY as inert dead code in case you want to merge them back.)
        self.t_junction_turn_delay_distance = 0.08  # meters - SEPARATE from turn_delay_distance. Distance driven straight (no line, nothing to track) after a barcode arrives at a stopped T-junction, before executing the 90 deg turn. Needs its own (typically shorter) value - using the normal intersection's turn_delay_distance here was overshooting and causing it to track the wrong line after turning. Tune this down/up based on your actual T-junction geometry.

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
        self.cross_intersection_duration = 0.3  # (No longer used - CROSS_INTERSECTION is now a one-shot decision made right when entered, since the turn_delay_distance gauge is already complete by then. Kept in case you want a timed cruise-through back.)
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
        self.t_junction_delay_start_x = 0.0
        self.t_junction_delay_start_y = 0.0

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

        # --- Odometry-Align Search (replaces the old near-90deg lidar wait) ---
        self.obstacle_reading_buffer = []       # list of (distance_m, angle_deg) tuples
        self.obstacle_reading_buffer_size = 5   # collect this many valid readings before locking
        self.obstacle_align_started = False
        self.obstacle_align_done = False
        self.obstacle_align_target = 0.0        # signed distance still needed to reach the true abeam point
        self.obstacle_align_start_x = 0.0
        self.obstacle_align_start_y = 0.0
        self.obstacle_perp_distance = 0.0       # true perpendicular offset to the board, from trig on the median reading

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

        # --- State Machine Dispatcher ---
        # Map each enum state to its dedicated handler method
        self.state_handlers = {
            RobotState.FOLLOW_LINE: self.handle_follow_line,
            RobotState.SEEK_INTERSECTION: self.handle_seek_intersection,
            RobotState.APPROACH_INTERSECTION: self.handle_approach_intersection,
            RobotState.EXECUTE_TURN: self.handle_execute_turn,
            RobotState.SEARCH_OBSTACLE: self.handle_search_obstacle,
            RobotState.HANDLE_OBSTACLE: self.handle_handle_obstacle,
            RobotState.CROSS_INTERSECTION: self.handle_cross_intersection,
            RobotState.T_JUNCTION_STOP: self.handle_t_junction_stop,
            RobotState.APPROACH_T_JUNCTION: self.handle_approach_t_junction,
            RobotState.REVERSE_TO_INTERSECTION: self.handle_reverse_to_intersection,
            RobotState.COURSE_COMPLETE: self.handle_course_complete,
        }

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
        """Handles state hijacking for Left and Right turns."""
        if self.current_state in (RobotState.APPROACH_INTERSECTION, RobotState.APPROACH_T_JUNCTION):
            # Already committed to a direction for this intersection.
            # Only the FIRST direction counts here - ignore any DIFFERENT direction
            # seen from here on. A repeat of the SAME direction is harmless.
            if self.intersection_memory not in ("NONE", direction):
                self.get_logger().warn(
                    f'BARCODE {barcode_id} ({direction}) IGNORED during {self.current_state.name} - '
                    f'already committed to {self.intersection_memory}.')
                return

            self.set_next_turn(direction)
            self.intersection_detected = False

            # Dynamically grab the correct distance for the log message
            delay_dist = self.turn_delay_distance if self.current_state == RobotState.APPROACH_INTERSECTION else self.t_junction_turn_delay_distance

            self.get_logger().info(
                f'BARCODE {barcode_id} seen mid-delay! Direction buffered as {direction} - '
                f'still finishing the {delay_dist}m approach before turning.')
            return

        self.set_next_turn(direction)
        self.intersection_detected = False

        if self.current_state in (RobotState.CROSS_INTERSECTION, RobotState.T_JUNCTION_STOP):
            # Dedicated state, dedicated (shorter) distance - using the
            # normal intersection's turn_delay_distance here was
            # overshooting and causing it to track the wrong line after
            # the 90 deg turn. There's no line here to gauge with anyway
            # (T_JUNCTION_STOP means the line's already gone), so this
            # state always just drives straight for t_junction_turn_delay_distance.
            self.current_state = RobotState.APPROACH_T_JUNCTION
            self.t_junction_delay_start_x, self.t_junction_delay_start_y = self.current_x, self.current_y
            self.cross_intersection_start_time = 0.0
            self.t_junction_wait_start = 0.0
            self.get_logger().info(
                f'BARCODE {barcode_id} seen at T-junction! Running dedicated '
                f'{self.t_junction_turn_delay_distance}m delay before {direction} turn.')
        else:
            self.current_state = RobotState.SEEK_INTERSECTION
            self.search_start_x, self.search_start_y = self.current_x, self.current_y
            self.get_logger().info(f'BARCODE {barcode_id}: Set to {direction}, waiting for distance...')

    def get_straight_twist(self, speed, log_msg=None):
        """Returns a Twist for driving straight and optionally logs a message."""
        if log_msg:
            self.get_logger().info(log_msg, throttle_duration_sec=0.3)

        twist = Twist()
        twist.linear.x = float(speed)
        twist.angular.z = 0.0
        return twist

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
        # Only pay attention while actively searching, and only until we
        # lock a reading in Phase 2 of handle_search_obstacle.
        if self.current_state != RobotState.SEARCH_OBSTACLE or self.obstacle_locked:
            return

        data = msg.data
        if len(data) < 3:
            return

        distance_m, angle_deg, direction = data[0], data[1], data[2]

        # Sanity filter FIRST: reject readings too close to straight-ahead
        # or straight-behind before they ever enter the buffer. This
        # rejects a wrong-target detection (front wall, second board
        # behind) as a wide band, instead of the old narrow near-90deg
        # window that created a close-range blind spot.
        abs_angle = abs(angle_deg)
        if not (self.obstacle_valid_angle_min <= abs_angle <= self.obstacle_valid_angle_max):
            self.get_logger().info(
                f'Lidar reading REJECTED (angle={angle_deg:.1f}deg outside '
                f'[{self.obstacle_valid_angle_min},{self.obstacle_valid_angle_max}] valid band)',
                throttle_duration_sec=0.3)
            return

        self.obstacle_recommended_dir = direction
        self.obstacle_target_angle_deg = angle_deg
        self.obstacle_last_known_distance = distance_m
        self.obstacle_info_received = True
        self.get_logger().info(
            f'Lidar obstacle info: dist={distance_m:.2f}m angle={angle_deg:.1f}deg '
            f'dir={"LEFT" if direction > 0 else "RIGHT"}')

        # Buffer for median filtering - a single sample can be noisy.
        self.obstacle_reading_buffer.append((distance_m, angle_deg))
        if len(self.obstacle_reading_buffer) > self.obstacle_reading_buffer_size:
            self.obstacle_reading_buffer.pop(0)

    def barcode_callback(self, msg):
       # Allow barcode reads in FOLLOW_LINE, CROSS_INTERSECTION (in case vision sees the intersection early), TURN_DELAY, and T_JUNCTION_TURN_DELAY (a barcode seen partway through either delay is buffered, not acted on early - see process_directional_barcode)
        if self.current_state not in (RobotState.FOLLOW_LINE, RobotState.CROSS_INTERSECTION, RobotState.APPROACH_INTERSECTION, RobotState.APPROACH_T_JUNCTION):
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
            if time.time() < self.obstacle_cooldown_time:
                self.get_logger().info('BARCODE 2 ignored (on cooldown after recent obstacle).', throttle_duration_sec=1.0)
                return

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
                self.obstacle_reading_buffer = []
                self.obstacle_align_started = False
                self.obstacle_align_done = False
                self.returned_to_front = False
                self.topple_wait_start = 0.0
                self.topple_last_recheck_time = 0.0
                self.topple_awaiting_response = False
                self.topple_confirmed_clear = False
                self.cross_intersection_start_time = 0.0 # Reset crossing timer just in case
                self.get_logger().info('BARCODE 2: Obstacle flagged. Stopping while lidar takes 5 samples...')

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
    # STATE HANDLERS
    # ---------------------------------------------------------
    def handle_follow_line(self, twist, time_since_last_line):
        if self.intersection_detected and self.intersection_memory == "NONE":
            self.intersection_detected = False
            self.get_logger().info(
                f'Intersection flagged, no barcode yet. Running the same {self.turn_delay_distance}m '
                f'gauge as a known turn - deciding stop/turn/cross once it completes.')
            self.current_state = RobotState.APPROACH_INTERSECTION
            self.delay_start_x, self.delay_start_y = self.current_x, self.current_y

        elif time_since_last_line > self.line_timeout:
            twist = self.get_stop_twist("No line found for 0.5s - stopping to avoid going off course.")
        else:
            twist = self.calculate_line_follow_twist(self.master_p_gain, self.master_d_gain, "FOLLOW_LINE")
        return twist

    def handle_reverse_to_intersection(self, twist, time_since_last_line):
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

        return twist

    def handle_seek_intersection(self, twist, time_since_last_line):
         # Calculate distance since barcode scan
        dist_traveled = self.get_distance_from(self.search_start_x, self.search_start_y, "[SEEK_INTERSECTION]")

        self.get_logger().info(
            f'SEEK: intersection_detected={self.intersection_detected} '
            f'dist={dist_traveled:.2f} threshold={self.turn_distance_threshold:.2f}',
            throttle_duration_sec=0.5)

        # --- PRIMARY TRIGGER: vision (intersection_detected) ---
        if self.intersection_detected:
            self.get_logger().info(f'Vision trigger! (dist={dist_traveled:.2f}m) Starting {self.turn_delay_distance*100:.0f}cm approach before deciding.')
            self.intersection_detected = False  # consume it so it can't latch

            # Send to TURN_DELAY and snapshot the current position
            self.current_state = RobotState.APPROACH_INTERSECTION
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
        return twist

    def handle_approach_intersection(self, twist, time_since_last_line):
        # Calculate how far we've driven since seeing the intersection
        delay_dist_traveled = self.get_distance_from(self.delay_start_x, self.delay_start_y, "[TURN_DELAY]")

        if delay_dist_traveled >= self.turn_delay_distance:
            if self.intersection_memory != "NONE":
                self.get_logger().info(f'{self.turn_delay_distance}m delay complete (dist={delay_dist_traveled:.2f}m). Executing turn.')
                self.current_state = RobotState.EXECUTE_TURN
                self.turn_start_time = time.time()
            else:
                self.get_logger().info(
                    f'{self.turn_delay_distance}m delay complete (dist={delay_dist_traveled:.2f}m), '
                    f'still no barcode. Deciding stop vs. cross-straight now.')
                self.current_state = RobotState.CROSS_INTERSECTION
            self.turn_delay_force_straight = False  # reset for next time

        # --- NEW T-JUNCTION TRIGGER: line ends ---
        # Only allowed to short-circuit early if we ALREADY know the
        # direction (intersection_memory != NONE) - safe to act on
        # early line-loss then. If we don't know the direction yet, a
        # barcode might still arrive mid-delay (see
        # process_directional_barcode's TURN_DELAY branch), so we must
        # finish the full 15cm regardless of line loss before deciding
        # stop vs turn vs cross-straight.
        elif ((not self.turn_delay_force_straight) and self.intersection_memory != "NONE"
                and time_since_last_line > 0.30):
            self.get_logger().info('Line ended in delay zone! (T-Junction). Turning instantly.')
            self.current_state = RobotState.EXECUTE_TURN
            self.turn_start_time = time.time()

        else:
            if self.turn_delay_force_straight:
                # No line exists here at all (T-junction) - nothing to
                # track, so just drive straight regardless of the
                # turn_delay_use_line_tracking toggle below.
                twist = self.get_straight_twist(
                    self.turn_delay_straight_speed,
                    f'[APPROACH_INTERSECTION] Driving straight at {self.turn_delay_straight_speed} m/s '
                    f'({delay_dist_traveled:.2f}m/{self.t_junction_turn_delay_distance}m)...'
                )
            elif self.turn_delay_use_line_tracking:
                # Gently follow the line during the delay (fixes
                # diagonal drift if the intersection sits on a curve).
                twist = self.calculate_line_follow_twist(self.delay_p_gain, self.delay_d_gain, "TURN_DELAY")
                twist.linear.x = self.turn_delay_straight_speed

                # --- NEW: Anti-Donut Clamp ---
                # The horizontal line of an intersection causes a massive error spike.
                # This clamps the steering to prevent violent spins, allowing only gentle curves.
                twist.angular.z = max(-0.4, min(0.4, twist.angular.z))

                self.get_logger().info(
                    f'[TURN_DELAY] Line-tracking clamped to {twist.angular.z:.2f} rad/s, '
                    f'speed={self.turn_delay_straight_speed} m/s '
                    f'while waiting out the {self.turn_delay_distance}m delay...',
                    throttle_duration_sec=0.3)
            else:
                # Drive straight ahead, no line-tracking steering - we
                # already got the vision trigger, so blindly following
                # line_error here can pull us onto whichever branch is
                # under the sensor at a T-junction before the deliberate
                # EXECUTE_TURN gets a chance to run.
                twist = self.get_straight_twist(
                    self.turn_delay_straight_speed,
                    f'[APPROACH_INTERSECTION] Driving straight at {self.turn_delay_straight_speed} m/s '
                    f'({delay_dist_traveled:.2f}m/{self.t_junction_turn_delay_distance}m)...'
                )
        return twist

    def handle_approach_t_junction(self, twist, time_since_last_line):
        # Dedicated delay for the T-junction case: no line exists here
        # at all, so always drive straight (nothing to line-track) for
        # its own separate, shorter distance - using the normal
        # intersection's turn_delay_distance here was overshooting and
        # causing the wrong line to be picked up after the 90 deg turn.
        t_junction_dist_traveled = self.get_distance_from(
            self.t_junction_delay_start_x, self.t_junction_delay_start_y, "[T_JUNCTION_TURN_DELAY]")

        if t_junction_dist_traveled >= self.t_junction_turn_delay_distance:
            self.get_logger().info(
                f'{self.t_junction_turn_delay_distance}m T-junction delay complete '
                f'(dist={t_junction_dist_traveled:.2f}m). Executing turn.')
            self.current_state = RobotState.EXECUTE_TURN
            self.turn_start_time = time.time()
        else:
            twist = self.get_straight_twist(
                self.turn_delay_straight_speed,
                f'[APPROACH_T_JUNCTION] Driving straight at {self.turn_delay_straight_speed} m/s '
                f'({t_junction_dist_traveled:.2f}m/{self.t_junction_turn_delay_distance}m)...'
            )
        return twist

    def handle_execute_turn(self, twist, time_since_last_line):
        if self.intersection_memory == "LEFT":
                twist.angular.z = self.turn_90_speed
        elif self.intersection_memory == "RIGHT":
            twist.angular.z = -self.turn_90_speed

        self.get_logger().info(
            f'[TROUBLESHOOT] EXECUTE_TURN: elapsed={time.time() - self.turn_start_time:.2f}s/1.1s '
            f'time_since_last_line={time_since_last_line:.2f}s',
            throttle_duration_sec=0.3)

        if (time.time() - self.turn_start_time) > 0.9:
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
        return twist

    def handle_search_obstacle(self, twist, time_since_last_line):
        elapsed_since_stop_trigger = time.time() - self.obstacle_search_entry_time

        if not self.obstacle_info_received:
            # --- PHASE 1: STOP and wait until we get a valid reading. ---
            if elapsed_since_stop_trigger > self.obstacle_search_timeout:
                self.get_logger().warn(f'SEARCH_OBSTACLE: No obstacle found after {self.obstacle_search_timeout}s. Giving up and resuming course.')
                self.obstacle_cooldown_time = time.time() + 4.0
                self.current_state = RobotState.FOLLOW_LINE
                return twist
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info(
                    f'SEARCH_OBSTACLE: Stopped at barcode, waiting for first obstacle scan... '
                    f'({elapsed_since_stop_trigger:.1f}s/{self.obstacle_search_timeout}s)',
                    throttle_duration_sec=0.5)

        elif not self.obstacle_align_done:
            # --- PHASE 2: Buffer a few readings, take the MEDIAN, compute
            # true geometry via trig ONCE, then close the remaining
            # distance via ODOMETRY - not another lidar sample. This
            # avoids waiting for a scan near exactly 90 deg, which can be
            # missed entirely at close range (perpendicular distance dips
            # below MIN_VALID_RANGE right at the crossing point).
            if not self.obstacle_align_started:
                if len(self.obstacle_reading_buffer) < self.obstacle_reading_buffer_size:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info(
                        f'Stopped. Buffering readings before locking: '
                        f'{len(self.obstacle_reading_buffer)}/{self.obstacle_reading_buffer_size}',
                        throttle_duration_sec=0.3)
                    return twist

                self.obstacle_locked = True  # freeze - stop accepting new lidar readings
                dists = sorted(d for d, a in self.obstacle_reading_buffer)
                angles = sorted(a for d, a in self.obstacle_reading_buffer)
                median_dist = dists[len(dists) // 2]
                median_angle = angles[len(angles) // 2]

                theta = math.radians(abs(median_angle))
                self.obstacle_perp_distance = median_dist * math.sin(theta)
                raw_target = median_dist * math.cos(theta)
                self.obstacle_align_target = raw_target + self.obstacle_align_offset
                self.obstacle_align_start_x = self.current_x
                self.obstacle_align_start_y = self.current_y
                self.obstacle_align_started = True
                self.get_logger().info(
                    f'Locked (median of {len(self.obstacle_reading_buffer)}): dist={median_dist:.2f}m '
                    f'angle={median_angle:.1f}deg -> perp={self.obstacle_perp_distance:.2f}m, '
                    f'align_target={self.obstacle_align_target:.2f}m')

            dist_moved = self.get_distance_from(self.obstacle_align_start_x, self.obstacle_align_start_y)
            if dist_moved >= abs(self.obstacle_align_target):
                self.obstacle_align_done = True
            else:
                twist.linear.x = (self.obstacle_search_forward_speed
                                   if self.obstacle_align_target >= 0
                                   else -self.obstacle_search_forward_speed)
                self.get_logger().info(
                    f'Aligning to true abeam point: {dist_moved:.2f}m / {abs(self.obstacle_align_target):.2f}m',
                    throttle_duration_sec=0.3)

        else:
            # --- PHASE 3: Rotate on the spot toward the obstacle ---
            if self.obstacle_rotate_start_time == 0.0:
                self.obstacle_rotate_start_time = time.time()
                angle_rad = math.radians(self.obstacle_square_turn_deg)
                self.obstacle_rotate_duration = (angle_rad / self.obstacle_angular_speed) * self.obstacle_turn_calibration
                self.get_logger().info(
                    f'Aligned. Rotating {"LEFT" if self.obstacle_recommended_dir > 0 else "RIGHT"} '
                    f'{self.obstacle_square_turn_deg:.0f} deg ({self.obstacle_rotate_duration:.2f}s) to square up.')

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
                self.approached_obstacle = False
                self.approach_start_x = self.current_x
                self.approach_start_y = self.current_y
                # Reuse the trig result already computed in Phase 2 -
                # don't recompute sin() from obstacle_last_known_distance
                # here, since that only reflects the LAST reading before
                # lock, not the median used for the actual geometry.
                self.target_approach_distance = self.obstacle_perp_distance - self.obstacle_approach_offset
                self.actual_approach_distance = 0.0

                self.approach_start_time = 0.0
                self.reverse_start_time = 0.0
                self.reversed_to_start = False
                self.started_reverse_back = False
                self.reverse_back_start_x = 0.0
                self.reverse_back_start_y = 0.0
                self.just_entered_reverse = False

                # Reset search state so the next barcode 2 starts fresh
                self.obstacle_info_received = False
                self.obstacle_locked = False
                self.obstacle_rotate_start_time = 0.0
                self.obstacle_reading_buffer = []
                self.obstacle_align_started = False
                self.obstacle_align_done = False
        return twist

    def handle_handle_obstacle(self, twist, time_since_last_line):
        # --- PHASE 1: Approach the obstacle to exactly __cm away ---
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
                    self.get_logger().info(f'Reached {self.obstacle_approach_offset*100:.0f}cm from obstacle. Stopping to topple.')
                    twist = self.get_stop_twist()
                    self.actual_approach_distance = distance_moved
                    self.approached_obstacle = True
                    self.topple_wait_start = 0.0
            return twist

        # --- PHASE 2: Topple & Confirm (concurrent with actuator motion) ---
        if not self.topple_confirmed_clear:
            if self.topple_wait_start == 0.0:
                self.topple_wait_start = time.time()
                self.topple_done_received = False
                trigger = Int32()
                trigger.data = 1
                self.servo_pub.publish(trigger)
                self.get_logger().info('Toppling obstacle. Starting concurrent lidar rechecks immediately...')

            elapsed_topple = time.time() - self.topple_wait_start

            # Overall safety cap covers actuator wait + settle + all rechecks combined
            if elapsed_topple > self.topple_overall_timeout:
                self.get_logger().warn(
                    f'Overall topple timeout ({self.topple_overall_timeout}s) reached - proceeding anyway.')
                self.topple_confirmed_clear = True
            else:
                # Short settle before the FIRST ping only - avoid pinging before
                # the board's even started moving. After that, keep pinging on
                # the normal interval REGARDLESS of whether the actuator has
                # reported done yet - no reason to wait for it.
                if elapsed_topple >= self.topple_settle_duration:
                    if not self.topple_awaiting_response and \
                        (time.time() - self.topple_last_recheck_time) >= self.topple_recheck_interval:
                        recheck_msg = Float32()
                        recheck_msg.data = self.obstacle_approach_offset  # true close-range distance, not the stale abeam reading
                        self.topple_recheck_pub.publish(recheck_msg)
                        self.topple_awaiting_response = True
                        self.topple_last_recheck_time = time.time()
                        self.get_logger().info(
                            f'[TROUBLESHOOT] Pinging lidar recheck (concurrent with actuator, '
                            f'actuator_done={self.topple_done_received})... elapsed={elapsed_topple:.1f}s')

                twist = self.get_stop_twist("Toppling / rechecking concurrently...")
                self.vel_pub.publish(twist)
                return twist

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
            return twist

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
            return twist

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
            self.obstacle_cooldown_time = time.time() + 4.0
            if self.enable_strict_retrace and self.obstacle_needs_full_retrace:
                self.get_logger().info(
                    f'Obstacle was near a turn ({self.obstacle_dist_from_last_turn:.2f}m) - '
                    f'reversing back to the intersection to undo the turn.')
                self.current_state = RobotState.REVERSE_TO_INTERSECTION
                self.retrace_start_x = self.current_x
                self.retrace_start_y = self.current_y
            else:
                self.current_state = RobotState.FOLLOW_LINE
        return twist

    def handle_cross_intersection(self, twist, time_since_last_line):
    # Entered only AFTER the full turn_delay_distance has already
        # been traveled with no barcode arriving (see TURN_DELAY
        # above) - so this is a one-shot decision now, not its own
        # separate timing/distance gauge. We already know exactly
        # where we are relative to the intersection; just check
        # whether a line is currently visible and act immediately.
        if time_since_last_line > 0.30:
            if self.stop_at_unbarcoded_intersections:
                self.get_logger().info('No line here after the gauge distance - confirmed T-Junction. Stopping.')
                self.current_state = RobotState.T_JUNCTION_STOP
            else:
                self.get_logger().info('No line here, but stop_at_unbarcoded_intersections is False - crossing anyway.')
                self.current_state = RobotState.FOLLOW_LINE
        else:
            self.get_logger().info('Line still visible after the gauge distance - real crossroad, continuing straight.')
            self.current_state = RobotState.FOLLOW_LINE
        return twist

    def handle_t_junction_stop(self, twist, time_since_last_line):
        if self.t_junction_wait_start == 0.0:
            self.t_junction_wait_start = time.time()

        elapsed = time.time() - self.t_junction_wait_start

        # Do not freeze here indefinitely and fail the "shortest time" rule.
        if elapsed > 1.2:
            self.get_logger().warn('T-Junction timeout (1.2s) reached. Forcing straight cross to comply with rules.')
            self.current_state = RobotState.FOLLOW_LINE
            self.t_junction_wait_start = 0.0
            twist = self.get_stop_twist("Timeout reached, forcing straight cross.")
        else:
            twist = self.get_stop_twist(f"T-Junction, no barcode. Waiting... ({elapsed:.1f}s/1.2s)")
        return twist

    def handle_course_complete(self, twist, time_since_last_line):
        return self.get_stop_twist()

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

        # ---------------------------------------------------------
        # THE DISPATCHER
        # Look up the handler method for the current state and run it
        # ---------------------------------------------------------
        handler_method = self.state_handlers.get(self.current_state)

        if handler_method:
            # Pass the twist object and time variable into the handler
            twist = handler_method(twist, time_since_last_line)
        else:
            self.get_logger().error(f'No handler defined for state {self.current_state}!')

        # ---------------------------------------------------------
        # UNIFIED PUBLISH
        # ---------------------------------------------------------
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
