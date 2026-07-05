#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32, Bool, Float32, Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
import math

class LidarObstacleLocator(Node):
    def __init__(self):
        super().__init__('lidar_obstacle_locator')

        # =========================================================
        # TUNING DIALS
        # =========================================================
        # Cone used for the INITIAL search (barcode 2 -> find & face the
        # obstacle), measured from straight-ahead (0 deg). Positive = LEFT,
        # negative = RIGHT.
        self.FRONT_SEARCH_DEG = 180.0    # search +/- this many degrees from center
        # Cone used for the FINAL CHECK (after toppling, confirming the
        # obstacle is actually gone) - full 360 degrees around the robot,
        # since the board (or something else) could end up anywhere.
        self.RECHECK_SEARCH_DEG = 30.0  # 30 = front cone
        self.MAX_VALID_RANGE = 1.0      # meters, ignore anything farther than this
        self.MIN_VALID_RANGE = 0.18     # meters, ignore anything closer (sensor noise)
        self.RECHECK_DISTANCE_TOLERANCE = 0.20  # +/- meters band around the recorded obstacle distance
        self.ARM_BARCODE_ID = 2         # barcode that arms the search
        self.DISARM_TIMEOUT = 5.0       # seconds; auto-disarm if nothing found (safety)
        # =========================================================

        self.armed = False
        self.arm_time = 0.0
        self.armed_for_recheck = False   # True = this arm session is the FINAL CHECK (360)
        self.recheck_ref_distance = 0.0  # reference distance to match against during recheck

        # --- Subscribers ---
        self.create_subscription(LaserScan, 'scan', self.scan_callback, qos_profile_sensor_data)
        self.create_subscription(Int32, 'pixy_barcode', self.barcode_callback, 10)
        self.create_subscription(Float32, 'topple_recheck_trigger', self.recheck_callback, 10)

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
            self.armed_for_recheck = False
            self.arm_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info('ARMED: searching for obstacle with lidar...')

    def recheck_callback(self, msg):
        # Lets the brain ask "is the obstacle still there?" on demand
        # (e.g. after triggering the topple actuator). This is the FINAL
        # CHECK, so it scans the full 360 degrees, and only counts a hit
        # if it's within +/- RECHECK_DISTANCE_TOLERANCE of where the
        # obstacle was originally found (msg.data = that reference distance).
        if not self.armed:
            self.recheck_ref_distance = msg.data
            self.armed = True
            self.armed_for_recheck = True
            self.arm_time = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().info(
                f'ARMED (recheck, 360): confirming whether obstacle is still near '
                f'{self.recheck_ref_distance:.2f}m +/- {self.RECHECK_DISTANCE_TOLERANCE:.2f}m...')

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
        search_cone_deg = self.RECHECK_SEARCH_DEG if self.armed_for_recheck else self.FRONT_SEARCH_DEG

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

            if abs(angle_deg) > search_cone_deg:
                continue

            # During the final recheck, only count a point if it's within
            # the tolerance band around where the obstacle was originally
            # found - ignores unrelated walls/track edges elsewhere in the
            # 360 sweep and specifically answers "is IT still there?"
            if self.armed_for_recheck:
                if abs(r - self.recheck_ref_distance) > self.RECHECK_DISTANCE_TOLERANCE:
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
                f'[{"RECHECK-360" if self.armed_for_recheck else "INITIAL"}] '
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
