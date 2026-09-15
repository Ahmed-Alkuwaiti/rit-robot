from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    book_colour = LaunchConfiguration('book_colour')
    shelf_column = LaunchConfiguration('shelf_column_number')
    auto_start = LaunchConfiguration('auto_start')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    shelf_standoff = LaunchConfiguration('shelf_standoff_distance')
    bin_standoff = LaunchConfiguration('bin_standoff_distance')
    minimum_target_distance = LaunchConfiguration('minimum_target_distance')
    obstacle_stop_distance = LaunchConfiguration('obstacle_stop_distance')
    book_bbox_stop_height = LaunchConfiguration('book_bbox_stop_height')
    search_angular_speed = LaunchConfiguration('search_angular_speed')
    return LaunchDescription([
        DeclareLaunchArgument('book_colour', default_value='red'),
        DeclareLaunchArgument('shelf_column_number', default_value='1'),
        DeclareLaunchArgument('auto_start', default_value='true'),
        DeclareLaunchArgument(
            'depth_topic', default_value=(
                '/head_front_camera/head_front_camera/'
                'depth/image_rect_raw')),
        DeclareLaunchArgument(
            'camera_info_topic', default_value=(
                '/head_front_camera/head_front_camera/depth/camera_info')),
        DeclareLaunchArgument('shelf_standoff_distance', default_value='0.65'),
        DeclareLaunchArgument('bin_standoff_distance', default_value='0.65'),
        DeclareLaunchArgument('minimum_target_distance', default_value='0.50'),
        DeclareLaunchArgument('obstacle_stop_distance', default_value='0.30'),
        DeclareLaunchArgument('book_bbox_stop_height', default_value='0'),
        DeclareLaunchArgument('search_angular_speed', default_value='-0.15'),
        ExecuteProcess(
            cmd=['bash', '-c',
                 'ros2 param get /robot_state_publisher robot_description '
                 '--hide-type > /tmp/tiago.urdf'],
            output='screen'),
        TimerAction(period=2.0, actions=[
            Node(package='rit_robocomp_2026', executable='perception_node',
                 name='perception_node', output='screen', parameters=[{
                     'book_color': book_colour,
                     'shelf_column_number': shelf_column,
                     'depth_topic': depth_topic,
                     'camera_info_topic': camera_info_topic,
                 }]),
            Node(package='rit_robocomp_2026', executable='manipulation_node',
                 name='manipulation_node', output='screen'),
            Node(package='rit_robocomp_2026', executable='navigation_node',
                 name='navigation_node', output='screen', parameters=[{
                     'shelf_standoff_distance': shelf_standoff,
                     'bin_standoff_distance': bin_standoff,
                     'minimum_target_distance': minimum_target_distance,
                     'obstacle_stop_distance': obstacle_stop_distance,
                     'book_bbox_stop_height': book_bbox_stop_height,
                     'search_angular_speed': search_angular_speed,
                 }]),
        ]),
        TimerAction(period=7.0, actions=[
            Node(package='rit_robocomp_2026', executable='task_manager',
                 name='task_manager', output='screen', parameters=[{
                     'auto_start': auto_start,
                 }]),
        ]),
    ])
