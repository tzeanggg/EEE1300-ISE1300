#!/usr/bin/env python3

import time
from enum import Enum

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool

import lgpio


class ActuatorState(Enum):
    IDLE = 0
    EXTENDING = 1
    RETRACTING = 2


class PiHandleObstacleActuator(Node):
    def __init__(self):
        super().__init__('pi_handle_obstacle_actuator')

        # GPIO18 = physical pin 12 on Raspberry Pi
        self.gpio_pin = 18

        # L16-R actuator servo-style PWM values
        self.retract_us = 1000
        self.middle_us = 1500
        self.extend_us = 2000

        # Linear actuator is slow, so give it enough time
        self.extend_duration = 10.0
        self.retract_duration = 10.0

        # Prevent repeated triggers
        self.cooldown_sec = 3.0
        self.last_trigger_time = 0.0

        self.state = ActuatorState.IDLE
        self.state_start_time = 0.0

        # Open Raspberry Pi GPIO chip
        self.gpio_handle = lgpio.gpiochip_open(0)

        # Subscribe to the trigger from your main brain node
        self.subscription = self.create_subscription(
            Int32,
            'topple_trigger',
            self.topple_trigger_callback,
            10
        )

        self.done_pub = self.create_publisher(Bool, 'topple_done', 10)

        # Timer updates actuator state without blocking ROS2
        self.timer = self.create_timer(0.05, self.update_actuator)

        # Start retracted
        self.send_servo_pulse(self.retract_us)

        self.get_logger().info('Pi Handle Obstacle Actuator Node started.')
        self.get_logger().info('Listening to /topple_trigger')
        self.get_logger().info('GPIO18 will control the actuator signal.')

    def send_servo_pulse(self, pulse_us):
        """
        Sends servo-style PWM to the actuator.
        1000us = retract
        1500us = middle
        2000us = extend
        0us    = stop PWM
        """

        if pulse_us == 0:
            lgpio.tx_servo(self.gpio_handle, self.gpio_pin, 0)
            self.get_logger().info('PWM stopped.')
            return

        lgpio.tx_servo(self.gpio_handle, self.gpio_pin, pulse_us)
        self.get_logger().info(
            f'Sent actuator pulse: {pulse_us} us',
            throttle_duration_sec=0.5
        )

    def topple_trigger_callback(self, msg):
        """
        Expected commands:
        data = 1 → topple once
        data = 0 → manual retract
        data = 2 → manual extend
        data = 3 → manual middle
        """

        now = time.time()

        if msg.data == 1:
            if self.state != ActuatorState.IDLE:
                self.get_logger().warn('Ignored topple command: actuator already moving.')
                return

            if now - self.last_trigger_time < self.cooldown_sec:
                self.get_logger().warn('Ignored topple command: cooldown active.')
                return

            self.last_trigger_time = now
            self.state = ActuatorState.EXTENDING
            self.state_start_time = now

            self.get_logger().info('Topple trigger received. Extending actuator.')
            self.send_servo_pulse(self.extend_us)

        elif msg.data == 0:
            self.get_logger().info('Manual retract command received.')
            self.state = ActuatorState.IDLE
            self.send_servo_pulse(self.retract_us)

        elif msg.data == 2:
            self.get_logger().info('Manual extend command received.')
            self.state = ActuatorState.IDLE
            self.send_servo_pulse(self.extend_us)

        elif msg.data == 3:
            self.get_logger().info('Manual middle command received.')
            self.state = ActuatorState.IDLE
            self.send_servo_pulse(self.middle_us)

        else:
            self.get_logger().warn(f'Unknown topple_trigger value: {msg.data}')

    def update_actuator(self):
        now = time.time()

        if self.state == ActuatorState.EXTENDING:
            elapsed = now - self.state_start_time

            if elapsed >= self.extend_duration:
                self.get_logger().info('Extend complete. Retracting actuator.')
                self.state = ActuatorState.RETRACTING
                self.state_start_time = now
                self.send_servo_pulse(self.retract_us)

        elif self.state == ActuatorState.RETRACTING:
            elapsed = now - self.state_start_time

            if elapsed >= self.retract_duration:
                self.get_logger().info('Topple cycle complete. Stopping PWM.')
                self.state = ActuatorState.IDLE

                # Stop PWM to reduce vibration/jitter after movement
                self.send_servo_pulse(0)

                # NEW: Publish confirmation to Brain
                done_msg = Bool()
                done_msg.data = True
                self.done_pub.publish(done_msg)

    def destroy_node(self):
        self.get_logger().info('Shutting down actuator node.')
        self.send_servo_pulse(0)
        lgpio.gpiochip_close(self.gpio_handle)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PiHandleObstacleActuator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
