from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="image_object_detection",
                executable="image_object_detection_node",
                output="screen",
            )
        ],
    )
