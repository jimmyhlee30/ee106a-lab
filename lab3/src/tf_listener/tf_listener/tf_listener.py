#!/usr/bin/env python3
import sys

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node


class TfEchoNode(Node):
    def __init__(self, target_frame, source_frame):
        super().__init__("tf_echo_node")

        # TODO: Create a buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.target_frame = target_frame
        self.source_frame = source_frame

        # Timer to print transform repeatedly
        self.timer = self.create_timer(0.5, self.print_transform)

    def quaternion_to_rotation_matrix(self, x, y, z, w):
        """Convert quaternion (x, y, z, w) to a 3x3 rotation matrix."""
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        if norm > 0:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def print_transform(self):
        # YOUR CODE HERE
        try:
            transform = self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, rclpy.time.Time())
            translation = transform.transform.translation
            rotation = transform.transform.rotation

            rotation_matrix = self.quaternion_to_rotation_matrix(rotation.x, rotation.y, rotation.z, rotation.w)

            print("Translation")
            print(np.array([translation.x, translation.y, translation.z]))

            print("Rotation Matrix")
            print(rotation_matrix)

        except tf2_ros.TransformException as e:
            print("Could not transform")


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 3:
        print("Usage: ros2 run <package> tf_listener <target_frame> <source_frame>")
        return

    target_frame = sys.argv[1]
    source_frame = sys.argv[2]

    node = TfEchoNode(target_frame, source_frame)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
