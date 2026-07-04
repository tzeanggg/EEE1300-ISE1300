#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, Bool, Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
import math

class LidarObstacleLocator(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_locator')

        # =========================================================
        # TUNING DIALS
        # =========================================================
        # Full front hemisphere to search for the obstacle itself (this bot
        # needs to FACE and topple the obstacle, not avoid it), measured
        # from straight-ahead (0 deg). Positive = LEFT, negative = RIGHT.
        self.FRONT_SEARCH_DEG = 180.0    # search +/- this many degrees from center
        self.MAX_VALID_RANGE = 1.0      # meters, ignore anything farther than this
        self.MIN_VALID_RANGE = 0.1     # meters, ignore anything closer (sensor noise)
        self.ARM_BARCODE_ID = 2         # barcode that arms the search
        self.DISARM_TIMEOUT = 5.0       # seconds; auto-disarm if nothing found (safety)
        # =========================================================

        self.armed = False
        self.arm_time = 0.0

        # --- Subscribers ---
        self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Int32, 'pixy_barcode', self.barcode_callback, 10)

        # --- Publishers ---
        # [distance_m, angle_deg, direction]
        # angle_deg: bearing to the nearest obstacle, 0 = straight ahead,
        #            positive = LEFT, negative = RIGHT
        # direction: +1.0 = rotate LEFT to face it, -1.0 = rotate RIGHT
        self.obstacle_info_pub = self.create_publisher(Float32MultiArray, 'obstacle_info', 10)
        self.obstacle_detected_pub = self.create_publisher(Bool, 'obstacle_detected', 10)

        self.get_logger().info('Lidar Obstacle Locator ready. Waiting for barcode 2...')

    def barcode_callback(self, msg):
        if msg.data == self.ARM_BARCODE_ID and not self.armed:
            self.armed = True
            self.arm_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info('ARMED: searching for obstacle with lidar...')

    def disarm(self, reason=''):
        self.armed = False
        if reason:
            self.get_logger().info(f'DISARMED: {reason}')

    def scan_callback(self, msg: LaserScan):
        if not self.armed:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.arm_time > self.DISARM_TIMEOUT:
            self.disarm('timeout, no obstacle found in time')
            detected = Bool()
            detected.data = False
            self.obstacle_detected_pub.publish(detected)
            return

        best_range = None
        best_angle_deg = 0.0

        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r):
                continue
            if r < self.MIN_VALID_RANGE or r > self.MAX_VALID_RANGE:
                continue

            angle_rad = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle_rad)

            # Normalize to [-180, 180] so "front" (0 deg) is unambiguous
            # regardless of how the driver wraps the scan.
            if angle_deg > 180.0:
                angle_deg -= 360.0

            if abs(angle_deg) > self.FRONT_SEARCH_DEG:
                continue

            if best_range is None or r < best_range:
                best_range = r
                best_angle_deg = angle_deg

        found_anything = best_range is not None
        detected = Bool()
        detected.data = found_anything
        self.obstacle_detected_pub.publish(detected)

        if found_anything:
            direction = 1.0 if best_angle_deg > 0 else -1.0
            info = Float32MultiArray()
            info.data = [float(best_range), float(best_angle_deg), direction]
            self.obstacle_info_pub.publish(info)

            self.get_logger().info(
                f'Obstacle found: dist={best_range:.2f}m angle={best_angle_deg:.1f}deg '
                f'-> rotate {"LEFT" if direction > 0 else "RIGHT"} to face it',
                throttle_duration_sec=0.5)

            # Got a reading — disarm so we don't keep re-publishing forever.
            # Re-arms next time barcode 2 fires.
            self.disarm('obstacle located')


def main(args=None):
    rclpy.init(args=args)
    node = LidarObstacleLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
