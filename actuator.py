#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Bool
from gpiozero import Servo
import time

class ActuatorNode(Node):
    def __init__(self):
        super().__init__('actuator_node')

        # 1. Setup the Subscriber (Listens to line_follower.py)
        self.subscription = self.create_subscription(
            Int32,
            'topple_trigger',
            self.trigger_callback,
            10)

        # 2. NEW: Setup the Publisher (Announces when finished)
        self.done_pub = self.create_publisher(Bool, 'topple_done', 10)

        # 3. Hardware Setup (GPIO 18)
        self.actuator_pin = 18
        self.actuator = Servo(
            self.actuator_pin, 
            min_pulse_width=0.001,  
            max_pulse_width=0.002   
        )

        self.is_active = False
        self.timer = None

        self.get_logger().info('Actuator node ready. Listening on /topple_trigger | Publishing on /topple_done')

    def trigger_callback(self, msg):
        if msg.data == 1 and not self.is_active:
            self.get_logger().info('Trigger received! Activating topple sequence.')
            self.is_active = True
            self.execute_topple()

    def execute_topple(self):
        self.get_logger().info('Extending actuator...')
        self.actuator.max()

        # Wait 2.0 seconds for the hardware to push the obstacle
        self.timer = self.create_timer(2.0, self.retract_actuator)

    def retract_actuator(self):
        self.get_logger().info('Retracting actuator...')
        self.actuator.min()

        # Clean up the timer
        self.timer.cancel()
        
        # --- NEW: Publish the "Done" signal ---
        # Give the physical servo a tiny fraction of a second to actually 
        # pull back before we broadcast that it is finished.
        time.sleep(0.5) 
        
        done_msg = Bool()
        done_msg.data = True
        self.done_pub.publish(done_msg)
        self.get_logger().info('Published [True] to /topple_done. Ready for next obstacle.')
        
        self.is_active = False

def main(args=None):
    rclpy.init(args=args)
    node = ActuatorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down actuator node.')
    finally:
        node.actuator.min()
        time.sleep(0.5) 
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
