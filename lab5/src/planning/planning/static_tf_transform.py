#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
# import tf_transformations
# from autolab_core import transformations
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from tf2_ros import TransformBroadcaster

class ConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('constant_tf_publisher')
        self.br = StaticTransformBroadcaster(self)

        # Homogeneous transform G_wrist_3_link->camera_depth_optical
        G = np.array([[1, 0, 0, -0.025],
                      [0, 1, 0, 0.13],
                      [0, 0, 1, 0.0],
                      [0, 0, 0, 1.0]
        ])

        # Create TransformStamped
        self.transform = TransformStamped()

        # ------------------------------------------
        # TODO: Fill out TransformStamped message
        # ------------------------------------------

        self.transform.header.frame_id = 'wrist_3_link'
        self.transform.child_frame_id = 'camera_depth_optical_frame'

        R_matrix = G[:3, :3]
        translation = G[:3, 3]

        r = R.from_matrix(R_matrix)
        q = r.as_quat()

        self.transform.transform.translation.x = float(translation[0])
        self.transform.transform.translation.y = float(translation[1])
        self.transform.transform.translation.z = float(translation[2])

        # Convert rotation matrix to quaternion (x, y, z, w)
        self.transform.transform.rotation.x = float(q[0])
        self.transform.transform.rotation.y = float(q[1])
        self.transform.transform.rotation.z = float(q[2])
        self.transform.transform.rotation.w = float(q[3])

        self.get_logger().info(f"Broadcasting transform:\n{G}\nQuaternion: {q}")


        self.timer = self.create_timer(0.05, self.broadcast_tf)

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.

    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True

    """
    q = np.empty((4,), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

if __name__ == '__main__':
    main()
