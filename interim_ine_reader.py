#!/usr/bin/env python3
import sys
sys.path.append('/home/group08/ros2_ws/src/project/project')

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Bool
import pixy
from ctypes import *
import time

# 1. C++ Structure for the Vector (Line)
class Vector(Structure):
    _fields_ = [
        ("m_x0", c_uint),
        ("m_y0", c_uint),
        ("m_x1", c_uint),
        ("m_y1", c_uint),
        ("m_index", c_uint),
        ("m_flags", c_uint)
    ]

# 2. C++ Structure for the Barcode
class Barcode(Structure):
    _fields_ = [
        ("m_x", c_uint),
        ("m_y", c_uint),
        ("m_flags", c_uint),
        ("m_code", c_uint)
    ]

# 3. C++ Structure for the Intersection
class Intersection(Structure):
    _fields_ = [
        ("m_x", c_uint),
        ("m_y", c_uint),
        ("m_n", c_uint),
        ("m_reserved", c_uint)
    ]

class PixyPublisher(Node):
    def __init__(self):
        super().__init__('pixy_publisher')

        # Publishers
        self.vector_pub = self.create_publisher(Float32, 'pixy_vector', 10)
        self.barcode_pub = self.create_publisher(Int32, 'pixy_barcode', 10)
        self.intersection_pub = self.create_publisher(Bool, 'pixy_intersection', 10)

        # Initialize the Pixy2 camera
        pixy.init()
        pixy.change_prog("line")

        # Turn on the Pixy2's onboard white illumination LEDs (upper, lower)
        pixy.set_lamp(1, 1)  # 1 = on, 0 = off, for each of the two lamps

        # --- Latch & Debounce Variables ---
        self.intersection_latch_time = 0.0
        self.intersection_counter = 0        # NEW: Tracks consecutive frames
        self.INTERSECTION_THRESHOLD = 4      # NEW: Must see it 4 times to trigger

        # Read the camera 30 times a second
        self.timer = self.create_timer(0.016, self.timer_callback)
        self.get_logger().info('Pixy2 Camera Publisher Started! Looking for lines, barcodes, and intersections...')

        self.last_published_barcode = -1

    def timer_callback(self):
        vectors = pixy.VectorArray(100)
        barcodes = pixy.BarcodeArray(100)
        intersections = pixy.IntersectionArray(100)

        # Tell the camera to report ALL feature types (vectors, intersections,
        # barcodes) every frame. In this SWIG binding, line_get_main_features()
        # only returns the single "best" feature and takes no arguments to
        # change that — line_get_all_features() is the separate call needed
        # to reliably get intersections/barcodes too.
        pixy.line_get_all_features()

        # --- Extract Vectors ---
        v_count = pixy.line_get_vectors(100, vectors)
        if v_count > 0:
            line_x = float(vectors[0].m_x1)
            msg_vec = Float32()
            msg_vec.data = line_x
            self.vector_pub.publish(msg_vec)

        # --- Extract Intersections ---
        i_count = pixy.line_get_intersections(100, intersections)
        msg_int = Bool()

        self.get_logger().info(
            f'i_count={i_count} counter={self.intersection_counter}',
            throttle_duration_sec=0.5)

        # NEW DEBOUNCE LOGIC: Filter out false positives
        if i_count > 0:
            self.intersection_counter += 1
        else:
            if self.intersection_counter > 0:
                self.intersection_counter -= 1 # Decay slowly in case of dropped frames

        # Cap the counter so it doesn't grow infinitely
        if self.intersection_counter > self.INTERSECTION_THRESHOLD:
            self.intersection_counter = self.INTERSECTION_THRESHOLD

        # Trigger logic based on the filtered counter, not the raw i_count
        if self.intersection_counter >= self.INTERSECTION_THRESHOLD:
            self.intersection_latch_time = time.time()
            msg_int.data = True

        elif (time.time() - self.intersection_latch_time) < 0.5:
            # Keep publishing True to give the Brain Node time to react!
            msg_int.data = True

        else:
            msg_int.data = False

        # Constantly publish True or False so the Brain Node always knows the status
        self.intersection_pub.publish(msg_int)

        # --- Extract Barcodes ---
        b_count = pixy.line_get_barcodes(100, barcodes)
        if b_count > 0:
            code_val = int(barcodes[0].m_code)
            if code_val != self.last_published_barcode:
                msg_bar = Int32()
                msg_bar.data = code_val
                self.barcode_pub.publish(msg_bar)
                self.get_logger().info(f'>>> BARCODE {code_val} DETECTED! <<<')
                self.last_published_barcode = code_val
        else:
            self.last_published_barcode = -1  # reset once it's out of view

def main(args=None):
    rclpy.init(args=args)
    node = PixyPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
