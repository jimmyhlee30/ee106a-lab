import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    aruco_params = os.path.join(
        get_package_share_directory('ros2_aruco'), 'config', 'aruco_parameters.yaml')
    robot = LaunchConfiguration('robot')
    robot_arg = DeclareLaunchArgument(
        'robot',
        description='Your TurtleBot name, e.g. apple. The camera publishes under it.')
    # base_link -> camera. The TurtleBot description has no camera link, so without
    # this every ar_marker_N frame is an orphan with no relationship to the map.
    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_link_to_camera',
        output='screen',
        arguments=[
            '--x', '0.115', '--y', '0.0', '--z', '0.0',
            '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
            '--frame-id', 'base_link', '--child-frame-id', 'camera',
        ],
    )
    # The camera topics are namespaced per robot, so they cannot live in the yaml.
    # These two entries override whatever the params file says.
    aruco = Node(
        package='ros2_aruco',
        executable='aruco_node',
        name='aruco_node',
        output='screen',
        parameters=[
            aruco_params,
            {
                'image_topic': ['/', robot, '/image_raw'],
                'camera_info_topic': ['/', robot, '/camera_info'],
            },
        ],
    )

    return LaunchDescription([
        robot_arg,
        camera_tf,
        aruco,
    ])