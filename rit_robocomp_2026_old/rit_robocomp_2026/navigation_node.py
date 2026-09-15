import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan


class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(
            String,
            '/rit/navigation/status',
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.manipulation_sub = self.create_subscription(
            String,
            '/rit/manipulation/status',
            self.manipulation_status_callback,
            10
        )

        self.command_sub = self.create_subscription(
            String,
            '/rit/navigation/command',
            self.command_callback,
            10
        )

        self.front_scan_sub = self.create_subscription(
            LaserScan,
            '/scan_front_raw',
            self.front_scan_callback,
            10
        )

        self.rear_scan_sub = self.create_subscription(
            LaserScan,
            '/scan_rear_raw',
            self.rear_scan_callback,
            10
        )

        self.front_min_range = None
        self.rear_min_range = None
        self.obstacle_stop_distance = 0.30

        self.last_odom_time = None
        self.last_front_scan_time = None
        self.last_rear_scan_time = None
        self.sensor_timeout = 1.0

        self.front_scan_sub = self.create_subscription(
            LaserScan,
            '/scan_front_raw',
            self.front_scan_callback,
            10
        )

        self.rear_scan_sub = self.create_subscription(
            LaserScan,
            '/scan_rear_raw',
            self.rear_scan_callback,
            10
        )

        self.front_min_range = None
        self.rear_min_range = None
        self.obstacle_stop_distance = 0.30

        self.current_x = None
        self.current_y = None
        self.current_yaw = None

        self.start_x = None
        self.start_y = None
        self.start_yaw = None
        self.start_pose_recorded = False

        self.arms_tucked = False
        self.last_manipulation_status_time = None
        self.manipulation_status_timeout = 30.0

        self.rotation_active = False
        self.target_yaw = None

        self.drive_active = False
        self.drive_start_x = None
        self.drive_start_y = None
        self.drive_distance = None

        self.strafe_active = False
        self.strafe_start_x = None
        self.strafe_start_y = None
        self.strafe_distance = None

        self.return_active = False
        self.return_stage = None
        self.align_shelf_active = False
        self.align_bin_active = False
        self.approach_shelf_active = False
        self.search_shelf_active = False
        self.search_bin_active = False
        self.approach_shelf_active = False

        self.align_shelf_active = False
        self.mock_shelf_lateral_error = 0.0
        self.shelf_align_tolerance = 0.02
        self.kp_shelf_align = 0.8
        self.max_strafe_speed = 0.08

        self.align_bin_active = False
        self.mock_bin_lateral_error = 0.0
        self.bin_align_tolerance = 0.02
        self.kp_bin_align = 0.8

        self.approach_shelf_active = False
        self.mock_shelf_distance_error = 0.0
        self.shelf_distance_tolerance = 0.03
        self.kp_shelf_distance = 0.6
        self.max_approach_speed = 0.08

        self.search_shelf_active = False
        self.mock_shelf_found = False
        self.search_angular_speed = 0.12

        self.search_bin_active = False
        self.mock_bin_found = False

        self.approach_shelf_active = False
        self.mock_shelf_distance_error = 0.0
        self.shelf_distance_tolerance = 0.03
        self.kp_shelf_distance = 0.6
        self.max_approach_speed = 0.08

        self.active_command = None
        self.motion_start_time = None
        self.motion_timeout = 90.0

        self.kp_angular = 0.8
        self.max_angular_speed = 0.20
        self.angle_tolerance = 0.03

        self.kp_linear = 0.8
        self.max_linear_speed = 0.10
        self.distance_tolerance = 0.015

        self.create_timer(0.05, self.control_loop)
        self.create_timer(1.0, self.print_pose)

        self.get_logger().info('Navigation node started')
        self.publish_status('IDLE')
        self.stop_robot()

    def publish_status(self, status):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Status: {status}')
        self.get_logger().info(f'Status: {status}')

    def odom_callback(self, msg):
        self.last_odom_time = time.monotonic()

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        if not self.start_pose_recorded:
            self.start_x = self.current_x
            self.start_y = self.current_y
            self.start_yaw = self.current_yaw
            self.start_pose_recorded = True

    def manipulation_status_callback(self, msg):
        self.last_manipulation_status_time = time.monotonic()

        if msg.data == 'SUCCEEDED:TUCK_ARMS':
            self.arms_tucked = True
        else:
            self.arms_tucked = False
            if self.motion_active():
                self.fail_motion('ARMS_NOT_TUCKED')

    def valid_scan_minimum(self, msg):
        valid_ranges = [
            r for r in msg.ranges
            if math.isfinite(r)
            and msg.range_min <= r <= msg.range_max
        ]

        if not valid_ranges:
            return None

        return min(valid_ranges)

    def front_scan_callback(self, msg):
        self.last_front_scan_time = time.monotonic()
        self.front_min_range = self.valid_scan_minimum(msg)

    def rear_scan_callback(self, msg):
        self.last_rear_scan_time = time.monotonic()
        self.rear_min_range = self.valid_scan_minimum(msg)

    def command_callback(self, msg):
        command = msg.data.strip().upper()

        if command == 'STOP':
            self.cancel_motion()
            self.publish_status('STOPPED')

        elif command == 'TEST_ROTATE_10':
            self.start_relative_rotation(
                math.radians(10.0),
                command
            )

        elif command == 'TEST_FORWARD_10CM':
            self.start_drive_distance(
                0.10,
                command
            )

        elif command == 'TEST_STRAFE_LEFT_10CM':
            self.start_strafe_distance(
                0.10,
                command
            )

        elif command == 'TEST_STRAFE_RIGHT_10CM':
            self.start_strafe_distance(
                -0.10,
                command
            )

        elif command == 'RETURN_TO_START':
            self.start_return_to_start(command)

        elif command == 'SEARCH_SHELF':
            self.start_search_shelf(command)

        elif command == 'MOCK_SHELF_FOUND':
            self.mock_shelf_found = True

        elif command == 'MOCK_SHELF_NOT_FOUND':
            self.mock_shelf_found = False

        elif command == 'APPROACH_SHELF':
            self.start_approach_shelf(command)

        elif command == 'MOCK_SHELF_FAR':
            self.mock_shelf_distance_error = 0.20

        elif command == 'MOCK_SHELF_NEAR':
            self.mock_shelf_distance_error = 0.02

        elif command == 'ALIGN_SHELF':
            self.start_align_shelf(command)

        elif command == 'MOCK_SHELF_LEFT':
            self.mock_shelf_lateral_error = 0.10

        elif command == 'MOCK_SHELF_RIGHT':
            self.mock_shelf_lateral_error = -0.10

        elif command == 'MOCK_SHELF_CENTER':
            self.mock_shelf_lateral_error = 0.0

        elif command == 'SEARCH_BIN':
            self.start_search_bin(command)

        elif command == 'MOCK_BIN_FOUND':
            self.mock_bin_found = True

        elif command == 'MOCK_BIN_NOT_FOUND':
            self.mock_bin_found = False

        elif command == 'ALIGN_BIN':
            self.start_align_bin(command)

        elif command == 'MOCK_BIN_LEFT':
            self.mock_bin_lateral_error = 0.10

        elif command == 'MOCK_BIN_RIGHT':
            self.mock_bin_lateral_error = -0.10

        elif command == 'MOCK_BIN_CENTER':
            self.mock_bin_lateral_error = 0.0

        else:
            self.stop_robot()
            self.publish_status(
                f'FAILED:{command}:UNKNOWN_COMMAND'
            )

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi

        while angle < -math.pi:
            angle += 2.0 * math.pi

        return angle

    def motion_active(self):
        return (
            self.rotation_active
            or self.drive_active
            or self.strafe_active
            or self.return_active
            or self.align_shelf_active
            or self.align_bin_active
            or self.approach_shelf_active
            or self.search_shelf_active
            or self.search_bin_active
        )

    def cancel_motion(self):
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False
        self.return_active = False
        self.return_stage = None
        self.active_command = None
        self.motion_start_time = None
        self.stop_robot()

    def fail_motion(self, reason):
        command = self.active_command
        self.cancel_motion()

        if command is not None:
            self.publish_status(
                f'FAILED:{command}:{reason}'
            )

    def succeed_motion(self):
        command = self.active_command
        self.cancel_motion()

        if command is not None:
            self.publish_status(
                f'SUCCEEDED:{command}'
            )

    def start_motion(self, command):
        if not self.arms_tucked:
            self.publish_status(
                f'FAILED:{command}:ARMS_NOT_TUCKED'
            )
            self.stop_robot()
            return False

        if self.current_x is None or self.current_y is None:
            self.publish_status(
                f'FAILED:{command}:NO_ODOMETRY'
            )
            self.stop_robot()
            return False

        self.active_command = command
        self.motion_start_time = time.monotonic()
        self.publish_status(f'RUNNING:{command}')

        return True

    def start_relative_rotation(self, angle_radians, command):
        if self.current_yaw is None:
            self.publish_status(
                f'FAILED:{command}:NO_ODOMETRY'
            )
            return

        if not self.start_motion(command):
            return

        self.target_yaw = self.normalize_angle(
            self.current_yaw + angle_radians
        )

        self.rotation_active = True

    def start_drive_distance(self, distance, command):
        if not self.start_motion(command):
            return

        self.drive_start_x = self.current_x
        self.drive_start_y = self.current_y
        self.drive_distance = distance
        self.drive_active = True

    def start_strafe_distance(self, distance, command):
        if not self.start_motion(command):
            return

        self.strafe_start_x = self.current_x
        self.strafe_start_y = self.current_y
        self.strafe_distance = distance
        self.strafe_active = True

    def start_return_to_start(self, command):
        if not self.start_pose_recorded:
            self.publish_status(
                f'FAILED:{command}:NO_START_POSE'
            )
            return

        if not self.start_motion(command):
            return

        dx = self.start_x - self.current_x
        dy = self.start_y - self.current_y

        distance = math.sqrt(dx * dx + dy * dy)

        self.return_active = True

        if distance <= self.distance_tolerance:
            self.target_yaw = self.start_yaw
            self.return_stage = 'RESTORE_START_YAW'
            self.rotation_active = True
            return

        self.target_yaw = math.atan2(dy, dx)
        self.return_stage = 'ROTATE_TO_START'
        self.rotation_active = True

    def start_align_shelf(self, command):
        if not self.start_motion(command):
            return

        self.align_shelf_active = True
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False

    def shelf_align_control(self):
        error = self.mock_shelf_lateral_error

        if abs(error) <= self.shelf_align_tolerance:
            self.align_shelf_active = False
            self.succeed_motion()
            return

        command = self.kp_shelf_align * error

        command = max(
            -self.max_strafe_speed,
            min(self.max_strafe_speed, command)
        )

        msg = Twist()
        msg.linear.y = command
        self.cmd_vel_pub.publish(msg)

    def start_align_bin(self, command):
        if not self.start_motion(command):
            return

        self.align_bin_active = True
        self.align_shelf_active = False
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False

    def bin_align_control(self):
        error = self.mock_bin_lateral_error

        if abs(error) <= self.bin_align_tolerance:
            self.align_bin_active = False
            self.succeed_motion()
            return

        command = self.kp_bin_align * error

        command = max(
            -self.max_strafe_speed,
            min(self.max_strafe_speed, command)
        )

        msg = Twist()
        msg.linear.y = command
        self.cmd_vel_pub.publish(msg)

    def start_approach_shelf(self, command):
        if not self.start_motion(command):
            return

        self.approach_shelf_active = True
        self.align_shelf_active = False
        self.align_bin_active = False
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False

    def approach_shelf_control(self):
        error = self.mock_shelf_distance_error

        if abs(error) <= self.shelf_distance_tolerance:
            self.approach_shelf_active = False
            self.succeed_motion()
            return

        if (
            self.front_min_range is not None
            and self.front_min_range < self.obstacle_stop_distance
        ):
            self.approach_shelf_active = False
            self.fail_motion('FRONT_OBSTACLE')
            return

        command = self.kp_shelf_distance * error

        command = max(
            -self.max_approach_speed,
            min(self.max_approach_speed, command)
        )

        msg = Twist()
        msg.linear.x = command
        self.cmd_vel_pub.publish(msg)

    def start_approach_shelf(self, command):
        if not self.start_motion(command):
            return

        self.approach_shelf_active = True
        self.align_shelf_active = False
        self.align_bin_active = False
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False

    def approach_shelf_control(self):
        error = self.mock_shelf_distance_error

        if abs(error) <= self.shelf_distance_tolerance:
            self.approach_shelf_active = False
            self.succeed_motion()
            return

        if (
            self.front_min_range is not None
            and self.front_min_range < self.obstacle_stop_distance
        ):
            self.approach_shelf_active = False
            self.fail_motion('FRONT_OBSTACLE')
            return

        command = self.kp_shelf_distance * error

        command = max(
            -self.max_approach_speed,
            min(self.max_approach_speed, command)
        )

        msg = Twist()
        msg.linear.x = command
        self.cmd_vel_pub.publish(msg)

    def start_search_shelf(self, command):
        if not self.start_motion(command):
            return

        self.search_shelf_active = True
        self.approach_shelf_active = False
        self.align_shelf_active = False
        self.align_bin_active = False
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False

    def search_shelf_control(self):
        if self.mock_shelf_found:
            self.search_shelf_active = False
            self.stop_robot()
            self.succeed_motion()
            return

        msg = Twist()
        msg.angular.z = self.search_angular_speed
        self.cmd_vel_pub.publish(msg)

    def start_search_bin(self, command):
        if not self.start_motion(command):
            return

        self.search_bin_active = True
        self.search_shelf_active = False
        self.approach_shelf_active = False
        self.align_shelf_active = False
        self.align_bin_active = False
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False

    def search_bin_control(self):
        if self.mock_bin_found:
            self.search_bin_active = False
            self.stop_robot()
            self.succeed_motion()
            return

        msg = Twist()
        msg.angular.z = self.search_angular_speed
        self.cmd_vel_pub.publish(msg)

    def start_search_bin(self, command):
        if not self.start_motion(command):
            return

        self.search_bin_active = True
        self.search_shelf_active = False
        self.approach_shelf_active = False
        self.align_shelf_active = False
        self.align_bin_active = False
        self.rotation_active = False
        self.drive_active = False
        self.strafe_active = False

    def search_bin_control(self):
        if self.mock_bin_found:
            self.search_bin_active = False
            self.stop_robot()
            self.succeed_motion()
            return

        msg = Twist()
        msg.angular.z = self.search_angular_speed
        self.cmd_vel_pub.publish(msg)

    def control_loop(self):
        if not self.motion_active():
            return

        if self.motion_start_time is not None:
            if time.monotonic() - self.motion_start_time > self.motion_timeout:
                self.fail_motion('TIMEOUT')
                return

        if not self.arms_tucked:
            self.fail_motion('ARMS_NOT_TUCKED')
            return

        if (
            self.last_odom_time is None
            or time.monotonic() - self.last_odom_time > self.sensor_timeout
        ):
            self.fail_motion('ODOMETRY_TIMEOUT')
            return

        if (
            self.current_x is None
            or self.current_y is None
            or self.current_yaw is None
        ):
            self.fail_motion('NO_ODOMETRY')
            return

        if self.rotation_active:
            self.rotation_control()

        elif self.drive_active:
            self.drive_control()

        elif self.strafe_active:
            self.strafe_control()

        elif self.align_shelf_active:
            self.shelf_align_control()

        elif self.align_bin_active:
            self.bin_align_control()

        elif self.approach_shelf_active:
            self.approach_shelf_control()

        elif self.search_shelf_active:
            self.search_shelf_control()

        elif self.search_bin_active:
            self.search_bin_control()

    def rotation_control(self):
        error = self.normalize_angle(
            self.target_yaw - self.current_yaw
        )

        if abs(error) <= self.angle_tolerance:
            self.rotation_active = False
            self.stop_robot()

            if self.return_active:
                if self.return_stage == 'ROTATE_TO_START':
                    dx = self.start_x - self.current_x
                    dy = self.start_y - self.current_y

                    self.drive_start_x = self.current_x
                    self.drive_start_y = self.current_y
                    self.drive_distance = math.sqrt(
                        dx * dx + dy * dy
                    )

                    self.return_stage = 'DRIVE_TO_START'
                    self.drive_active = True
                    return

                if self.return_stage == 'RESTORE_START_YAW':
                    self.return_active = False
                    self.return_stage = None
                    self.succeed_motion()
                    return

            self.succeed_motion()
            return

        command = self.kp_angular * error

        command = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, command)
        )

        msg = Twist()
        msg.angular.z = command
        self.cmd_vel_pub.publish(msg)

    def drive_control(self):
        now = time.monotonic()

        if self.drive_distance > 0.0:
            if (
                self.last_front_scan_time is None
                or now - self.last_front_scan_time > self.sensor_timeout
            ):
                self.fail_motion('FRONT_LIDAR_TIMEOUT')
                return

            if (
                self.front_min_range is not None
                and self.front_min_range < self.obstacle_stop_distance
            ):
                self.fail_motion('FRONT_OBSTACLE')
                return

        elif self.drive_distance < 0.0:
            if (
                self.last_rear_scan_time is None
                or now - self.last_rear_scan_time > self.sensor_timeout
            ):
                self.fail_motion('REAR_LIDAR_TIMEOUT')
                return

            if (
                self.rear_min_range is not None
                and self.rear_min_range < self.obstacle_stop_distance
            ):
                self.fail_motion('REAR_OBSTACLE')
                return

        dx = self.current_x - self.drive_start_x
        dy = self.current_y - self.drive_start_y

        travelled = math.sqrt(dx * dx + dy * dy)
        error = abs(self.drive_distance) - travelled

        if error <= self.distance_tolerance:
            self.drive_active = False
            self.stop_robot()

            if (
                self.return_active
                and self.return_stage == 'DRIVE_TO_START'
            ):
                self.target_yaw = self.start_yaw
                self.return_stage = 'RESTORE_START_YAW'
                self.rotation_active = True
                return

            self.succeed_motion()
            return

        command = min(
            self.kp_linear * error,
            self.max_linear_speed
        )

        if self.drive_distance < 0.0:
            command = -command

        msg = Twist()
        msg.linear.x = command
        self.cmd_vel_pub.publish(msg)

    def strafe_control(self):
        dx = self.current_x - self.strafe_start_x
        dy = self.current_y - self.strafe_start_y

        travelled = math.sqrt(dx * dx + dy * dy)
        error = abs(self.strafe_distance) - travelled

        if error <= self.distance_tolerance:
            self.succeed_motion()
            return

        command = min(
            self.kp_linear * error,
            self.max_linear_speed
        )

        if self.strafe_distance < 0.0:
            command = -command

        msg = Twist()
        msg.linear.y = command
        self.cmd_vel_pub.publish(msg)

    def print_pose(self):
        if self.current_yaw is None:
            return

        self.get_logger().info(
            f'Pose: x={self.current_x:.3f}, '
            f'y={self.current_y:.3f}, '
            f'yaw={self.current_yaw:.3f}, '
            f'arms_tucked={self.arms_tucked}'
        )

    def stop_robot(self):
        if rclpy.ok():
            self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
