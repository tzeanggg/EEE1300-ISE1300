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
        self.MAX_VALID_RANGE = 0.5      # meters, ignore anything farther than this
        self.MIN_VALID_RANGE = 0.12     # meters, ignore anything closer (sensor noise)
        self.RECHECK_DISTANCE_TOLERANCE = 0.20  # +/- meters band around the recorded obstacle distance
        self.ARM_BARCODE_ID = 2         # barcode that arms the search
        self.DISARM_TIMEOUT = 5.0       # seconds; auto-disarm if nothing found (safety)
        self.CLUSTER_IDX_GAP = 5        # max index gap between points to still count as same cluster (lower = stricter, terminates clusters closer to true edges)
        self.EDGE_PERP_TOLERANCE = 0.03   # meters - points in the best cluster whose PERPENDICULAR distance from the fitted board-line exceeds this are treated as edge/mixed-pixel noise and excluded. Unlike a raw range check, this correctly keeps genuine far-edge points on a tilted/angled board while still rejecting true noise.
        self.ANGLE_BIAS_DEG = 0.0         # degrees - fixed correction for lidar mounting/calibration skew. Recall: positive angle = LEFT, negative = RIGHT (see obstacle_info_pub comment below). If the robot consistently approaches too far LEFT of the true center, INCREASE this value. If consistently too far RIGHT, DECREASE it (go negative). Tune empirically in ~1 deg steps.
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

        search_cone_deg = self.RECHECK_SEARCH_DEG if self.armed_for_recheck else self.FRONT_SEARCH_DEG

       # --- IMPROVED: Filtered Cluster Logic ---
        valid_points = []
        for i, r in enumerate(msg.ranges):
            # 1. Standard filtering
            if math.isinf(r) or math.isnan(r) or r < self.MIN_VALID_RANGE or r > self.MAX_VALID_RANGE:
                continue

            angle_rad = msg.angle_min + i * msg.angle_increment
            angle_deg = math.degrees(angle_rad)
            if angle_deg > 180.0: angle_deg -= 360.0

            if abs(angle_deg) <= search_cone_deg:
                # During recheck, enforce distance tolerance band
                if self.armed_for_recheck and abs(r - self.recheck_ref_distance) > self.RECHECK_DISTANCE_TOLERANCE:
                    continue
                valid_points.append({'r': r, 'idx': i})

        clusters = []
        if valid_points:
            current_cluster = [valid_points[0]]
            for i in range(1, len(valid_points)):
                # If indices are adjacent (within CLUSTER_IDX_GAP steps), it's the same object
                dist_diff = abs(valid_points[i]['r'] - valid_points[i-1]['r'])
                idx_diff = valid_points[i]['idx'] - valid_points[i-1]['idx']

                if idx_diff < self.CLUSTER_IDX_GAP and dist_diff < 0.05:
                    current_cluster.append(valid_points[i])
                else:
                    if len(current_cluster) > 2: clusters.append(current_cluster)
                    current_cluster = [valid_points[i]]
            if len(current_cluster) > 2: clusters.append(current_cluster)

        # --- Calculate Midpoint of largest Cluster ---
        found_anything = len(clusters) > 0
        if found_anything:
            # 1. Pick the LARGEST cluster (most points = most surface area)
            best_cluster = max(clusters, key=lambda c: len(c))

            # 2. Fit a straight line through the cluster in Cartesian space.
            # A flat board still forms a straight line even when viewed at
            # an angle (e.g. one face of a hexagon relative to the robot) -
            # its range legitimately increases from the near edge to the
            # far edge. That gradient is real geometry, NOT noise, so we
            # can't filter by "how far is this point's range from the
            # median" anymore (that would wrongly discard the genuine far
            # edge). Instead: fit the best-fit line through the points,
            # then reject only points that deviate PERPENDICULAR to that
            # line - that's the actual signature of edge/mixed-pixel noise,
            # regardless of the board's tilt.
            pts_xy = []
            for p in best_cluster:
                a = msg.angle_min + p['idx'] * msg.angle_increment
                pts_xy.append((p['r'] * math.cos(a), p['r'] * math.sin(a)))

            cx = sum(x for x, y in pts_xy) / len(pts_xy)
            cy = sum(y for x, y in pts_xy) / len(pts_xy)
            sxx = sum((x - cx) ** 2 for x, y in pts_xy)
            syy = sum((y - cy) ** 2 for x, y in pts_xy)
            sxy = sum((x - cx) * (y - cy) for x, y in pts_xy)

            # Principal direction of the point spread (PCA / total-least-squares line fit)
            theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
            dx, dy = math.cos(theta), math.sin(theta)

            # 3. Reject points whose perpendicular distance from the fitted
            # line exceeds tolerance - genuine noise, not gradient.
            inliers = []
            for x, y in pts_xy:
                perp = abs(-dy * (x - cx) + dx * (y - cy))
                if perp <= self.EDGE_PERP_TOLERANCE:
                    inliers.append((x, y))
            if not inliers:
                inliers = pts_xy  # fallback safety net, shouldn't trigger

            # 4. Midpoint = center of the inliers' span ALONG the fitted
            # line (not a simple average, so a lopsided inlier count on one
            # side still gives the true geometric midpoint of the board).
            projections = [dx * (x - cx) + dy * (y - cy) for x, y in inliers]
            mid_t = (min(projections) + max(projections)) / 2.0
            mid_x = cx + dx * mid_t
            mid_y = cy + dy * mid_t

            best_range = math.hypot(mid_x, mid_y)

            # 5. Angle to that geometric midpoint (correct regardless of tilt)
            best_angle_deg = math.degrees(math.atan2(mid_y, mid_x))
            if best_angle_deg > 180.0: best_angle_deg -= 360.0
            if best_angle_deg < -180.0: best_angle_deg += 360.0

            # 4. Apply mounting/calibration bias correction (tune this, not
            # the geometry code, if the miss is a consistent same-direction offset)
            best_angle_deg = best_angle_deg - self.ANGLE_BIAS_DEG
            direction = 1.0 if best_angle_deg > 0 else -1.0

            # Publish findings
            info = Float32MultiArray()
            info.data = [float(best_range), float(best_angle_deg), direction]
            self.obstacle_info_pub.publish(info)

            self.get_logger().info(
                f'Midpoint found: dist={best_range:.2f}m angle={best_angle_deg:.1f}deg',
                throttle_duration_sec=0.5)

           # Throttled to 0.2s so you can clearly see the angle changing in real-time
            self.get_logger().info(
                f'[TRACKING] Obstacle: dist={best_range:.2f}m angle={best_angle_deg:.1f}deg',
                throttle_duration_sec=0.2)

            detected = Bool()
            detected.data = True
            self.obstacle_detected_pub.publish(detected)

            if not self.armed_for_recheck:
                # DO NOT DISARM.
                # Instead, reset the arm_time so the 5-second safety timeout
                # doesn't trigger while the robot is slowly creeping forward.
                self.arm_time = now
        else:
            detected = Bool()
            detected.data = False
            self.obstacle_detected_pub.publish(detected)


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
