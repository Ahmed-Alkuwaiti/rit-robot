from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('erc_bringup'),
                'launch',
                'simulation.launch.py',
            )
        )
    )

    perception = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            'mkdir -p /opt/erc_ws/runtime_logs && '
            'exec ros2 run rit_robocomp_2026 perception_node '
            '>> /opt/erc_ws/runtime_logs/perception.log 2>&1'
        ],
    )

    manipulation = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            'mkdir -p /opt/erc_ws/runtime_logs && '
            'exec ros2 run rit_robocomp_2026 manipulation_node '
            '>> /opt/erc_ws/runtime_logs/manipulation.log 2>&1'
        ],
    )

    navigation = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            'mkdir -p /opt/erc_ws/runtime_logs && '
            'exec ros2 run rit_robocomp_2026 navigation_node '
            '--ros-args '
            '-p shelf_standoff_distance:=0.75 '
            '-p bin_standoff_distance:=0.75 '
            '-p minimum_target_distance:=0.50 '
            '-p obstacle_stop_distance:=0.30 '
            '>> /opt/erc_ws/runtime_logs/navigation.log 2>&1'
        ],
    )

    task_manager = Node(
        package='rit_robocomp_2026',
        executable='task_manager',
        name='task_manager',
        output='screen',
    )

    return LaunchDescription([
        simulation,

        TimerAction(
            period=10.0,
            actions=[
                perception,
                manipulation,
                navigation,
            ],
        ),

        TimerAction(
            period=20.0,
            actions=[
                task_manager,
            ],
        ),
    ])
