#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Float32, Bool
from nav_msgs.msg import Odometry
from enum import Enum
import time
import math
import signal
import sys

class RobotState(Enum):
    FOLLOW_LINE = 1
    EXECUTE_TURN = 2
    HANDLE_OBSTACLE = 3
    ESCAPE_TURN = 4
    COURSE_COMPLETE = 5
    SEEK_INTERSECTION = 6

class TurtleBotBrain(Node):
    def __init__(self):
        super().__init__('turtlebot_brain')

        # --- System State Variables ---
        self.current_state = RobotState.FOLLOW_LINE
        self.last_barcode_seen = -1
        self.line_error = 0.0
        self.last_line_error = 0.0
        self.intersection_detected = False
        self.estopped = False

        # --- Dynamic Memory Variables ---
        self.intersection_memory = "NONE"
        self.escape_turn = "NONE"
        self.turn_start_time = 0.0

        # --- Calibration Parameters ---
        # LOCKED: Camera center is now fixed at 39.0
        self.camera_center = 39.0
        self.line_timeout = 0.5

        # --- Distance-Based Turn Settings ---
        self.turn_distance_threshold = 0.2 # SET THIS: Meters from barcode to turn point

        # --- Timers ---
        self.last_line_seen_time = time.time()

        # --- Odometry Variables ---
        self.current_x = 0.0
        self.current_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.target_reverse_distance = 0.2
        self.just_entered_reverse = False

        # --- Intersection Search Safety ---
        self.search_start_x = 0.0
        self.search_start_y = 0.0
        self.search_distance_limit = 0.6

        # --- Publishers & Subscribers ---
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.servo_pub = self.create_publisher(Int32, 'topple_trigger', 10)

        self.create_subscription(Float32, 'pixy_vector', self.line_callback, 10)
        self.create_subscription(Int32, 'pixy_barcode', self.barcode_callback, 10)
        self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.create_subscription(Bool, 'pixy_intersection', self.intersection_callback, 10)

        self.timer = self.create_timer(0.033, self.fsm_loop)
        self.get_logger().info('Brain activated. Press Ctrl+C to Emergency Stop.')

    # ---------------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------------
    def line_callback(self, msg):
        self.line_error = msg.data - self.camera_center
        self.last_line_seen_time = time.time()

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

    def intersection_callback(self, msg):
        self.intersection_detected = msg.data

    def barcode_callback(self, msg):
        # Ignore if we are already in the middle of a maneuver
        if self.current_state != RobotState.FOLLOW_LINE:
            return

        barcode_id = msg.data
        if barcode_id == 0:
            self.set_next_turn("LEFT")
            self.escape_turn = "RIGHT"
            self.current_state = RobotState.SEEK_INTERSECTION
            self.search_start_x, self.search_start_y = self.current_x, self.current_y
            self.get_logger().info('BARCODE 0: Set to Left, waiting for distance...')

        elif barcode_id == 1:
            self.set_next_turn("RIGHT")
            self.escape_turn = "LEFT"
            self.current_state = RobotState.SEEK_INTERSECTION
            self.search_start_x, self.search_start_y = self.current_x, self.current_y
            self.get_logger().info('BARCODE 1: Set to Right, waiting for distance...')

        elif barcode_id == 2:
            if self.current_state != RobotState.HANDLE_OBSTACLE:
                self.current_state = RobotState.HANDLE_OBSTACLE
                self.just_entered_reverse = True

        elif barcode_id == 3:
            self.current_state = RobotState.COURSE_COMPLETE

    # Add this to your class
    def set_next_turn(self, direction):
        self.intersection_memory = direction
        self.get_logger().info(f'Next turn buffered: {direction}')

    # ---------------------------------------------------------
    # MAIN FSM LOOP
    # ---------------------------------------------------------
    def fsm_loop(self):
        if self.estopped:
            return

        twist = Twist()
        time_since_last_line = time.time() - self.last_line_seen_time

        self.get_logger().info(f'Current Brain State: {self.current_state.name}', throttle_duration_sec=1.0)

        # =========================================================
        # MASTER TUNING DIALS
        # Change these two numbers to tune the entire robot's handling
        # =========================================================
        master_p_gain = 0.012  # Controls how aggressively it steers toward the line
        master_d_gain = 0.19  # Controls the "shock absorber" resistance to sudden turns
        # =========================================================

        if self.current_state == RobotState.FOLLOW_LINE:

            if time_since_last_line > self.line_timeout:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                # 1. Steering (PD-Controller)
                error_derivative = self.line_error - self.last_line_error
                turn_speed = (-master_p_gain * self.line_error) + (-master_d_gain * error_derivative)
                self.last_line_error = self.line_error

                # 2. Dynamic Braking
                base_speed = 0.15
                speed_penalty = abs(self.line_error) * 0.003
                forward_speed = max(base_speed - speed_penalty, 0.05)

                twist.linear.x = forward_speed
                twist.angular.z = turn_speed

        elif self.current_state == RobotState.SEEK_INTERSECTION:
            # Calculate distance since barcode scan
            dist_traveled = math.sqrt((self.current_x - self.search_start_x)**2 +
                                      (self.current_y - self.search_start_y)**2)

            # --- DISTANCE TRIGGER ---
            if dist_traveled >= self.turn_distance_threshold:
                self.get_logger().info(f'Traveled {dist_traveled:.2f}m. Triggering turn!')
                self.current_state = RobotState.EXECUTE_TURN
                self.turn_start_time = time.time()
            else:
                # Still cruising to the turn point using PD control
                error_derivative = self.line_error - self.last_line_error
                turn_speed = (-0.04 * self.line_error) + (-0.08 * error_derivative)
                self.last_line_error = self.line_error

                base_speed = 0.15
                speed_penalty = abs(self.line_error) * 0.003
                forward_speed = max(base_speed - speed_penalty, 0.05)

                twist.linear.x = forward_speed
                twist.angular.z = turn_speed

        elif self.current_state == RobotState.EXECUTE_TURN:
            if self.intersection_memory == "LEFT":
                twist.angular.z = 0.8
            elif self.intersection_memory == "RIGHT":
                twist.angular.z = -0.8

            if (time.time() - self.turn_start_time) > 1.7:
                if time_since_last_line < 0.2:
                    self.get_logger().info('90-Degree line acquired! Resuming cruise.')
                    self.intersection_memory = "NONE"
                    self.current_state = RobotState.FOLLOW_LINE

            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.HANDLE_OBSTACLE:
            if self.just_entered_reverse:
                trigger = Int32()
                trigger.data = 1
                self.servo_pub.publish(trigger)
                self.start_x = self.current_x
                self.start_y = self.current_y
                self.just_entered_reverse = False

            distance = math.sqrt((self.current_x - self.start_x)**2 + (self.current_y - self.start_y)**2)
            if distance < self.target_reverse_distance:
                twist.linear.x = -0.1
            else:
                self.current_state = RobotState.ESCAPE_TURN
                self.turn_start_time = time.time()

            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.ESCAPE_TURN:
            twist.angular.z = 0.8 if self.escape_turn == "LEFT" else -0.8
            if (time.time() - self.turn_start_time) > 1.7 and time_since_last_line < 0.2:
                self.get_logger().info('Escape complete. Resuming cruise.')
                self.escape_turn = "NONE"
                self.current_state = RobotState.FOLLOW_LINE

            self.vel_pub.publish(twist)
            return

        elif self.current_state == RobotState.COURSE_COMPLETE:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.vel_pub.publish(twist)

    def emergency_stop(self):
        self.get_logger().warn('EMERGENCY STOP TRIGGERED')
        self.estopped = True
        twist = Twist()
        twist.linear.x = 0.0
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
