from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python import get_package_share_directory
import os


def generate_launch_description():
    # Declare the launch argument
    declared_arguments = [
        DeclareLaunchArgument(
            "config_file",
            default_value=os.path.join(
                get_package_share_directory("image_object_detection"),
                "config/image_object_detection.yaml",
            ),
            description="Path to the configuration file",
        )
    ]

    config_file = LaunchConfiguration("config_file")

    image_object_detection_node = Node(
        package="image_object_detection",
        executable="image_object_detection_node",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription(declared_arguments + [image_object_detection_node])
