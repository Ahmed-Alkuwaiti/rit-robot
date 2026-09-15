from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    book_colour_arg = DeclareLaunchArgument(
        'book_colour',
        default_value='red',
        description='Requested book color (launch-facing arg name)'
    )

    shelf_column_arg = DeclareLaunchArgument(
        'shelf_column_number',
        default_value='1',
        description='Requested shelf marker number (1-5)'
    )

    perception_node = Node(
        package='rit_robocomp_2026',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=[{
            # launch-facing arg is "book_colour", perception node parameter is "book_color"
            'book_color': LaunchConfiguration('book_colour'),
            'shelf_column_number': LaunchConfiguration('shelf_column_number'),
        }]
    )

    navigation_node = Node(
        package='rit_robocomp_2026',
        executable='navigation_node',
        name='navigation_node',
        output='screen'
    )

    return LaunchDescription([
        book_colour_arg,
        shelf_column_arg,
        perception_node,
        navigation_node,
    ])
