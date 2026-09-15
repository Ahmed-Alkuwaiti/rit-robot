from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='rit_robocomp_2026', executable='navigation_node',
             name='navigation_node', output='screen'),
    ])
