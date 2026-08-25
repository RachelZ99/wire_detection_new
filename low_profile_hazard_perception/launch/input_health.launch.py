from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory("low_profile_hazard_perception")
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(
                    package_share / "config" / "input_health.yaml"
                ),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            Node(
                package="low_profile_hazard_perception",
                executable="input_health_node",
                name="input_health",
                output="screen",
                parameters=[
                    LaunchConfiguration("config"),
                    {"use_sim_time": LaunchConfiguration("use_sim_time")},
                ],
            ),
        ]
    )
