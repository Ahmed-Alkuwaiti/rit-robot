#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32

MAX_RETRIES = 2

TIMEOUTS = {
    'TUCK_ARMS': 120.0,
    'SEARCH_SHELF': 30.0,
    'IDENTIFY_SHELF_COLUMN': 20.0,
    'APPROACH_SHELF': 30.0,
    'IDENTIFY_BOOK_AND_ROW': 20.0,
    'ALIGN_SHELF': 30.0,
    'PREPARE_GRASP': 30.0,
    'GRASP': 30.0,
    'LIFT_AND_RETRACT': 30.0,
    'TRANSPORT_POSE': 30.0,
    'RETURN_TO_START': 30.0,
    'SEARCH_BIN': 30.0,
    'ALIGN_BIN': 30.0,
    'PREPARE_PLACE': 30.0,
    'PLACE': 30.0,
    'SAFE_POSE': 30.0,
}

MANIPULATION_STATES = {
    'TUCK_ARMS', 'PREPARE_GRASP', 'GRASP', 'LIFT_AND_RETRACT',
    'TRANSPORT_POSE', 'PREPARE_PLACE', 'PLACE', 'SAFE_POSE'
}

NAVIGATION_STATES = {
    'SEARCH_SHELF', 'APPROACH_SHELF', 'ALIGN_SHELF',
    'RETURN_TO_START', 'SEARCH_BIN', 'ALIGN_BIN'
}

STATE_ORDER = [
    'TUCK_ARMS', 'SEARCH_SHELF', 'IDENTIFY_SHELF_COLUMN', 'APPROACH_SHELF',
    'IDENTIFY_BOOK_AND_ROW', 'ALIGN_SHELF', 'PREPARE_GRASP', 'GRASP',
    'LIFT_AND_RETRACT', 'TRANSPORT_POSE', 'RETURN_TO_START', 'SEARCH_BIN',
    'ALIGN_BIN', 'PREPARE_PLACE', 'PLACE', 'SAFE_POSE', 'DONE'
]


class TaskManager(Node):
    def __init__(self):
        super().__init__('task_manager')
        self.manipulation_command_pub = self.create_publisher(String, '/rit/manipulation/command', 10)
        self.navigation_command_pub = self.create_publisher(String, '/rit/navigation/command', 10)
        self.task_state_pub = self.create_publisher(String, '/rit/task/state', 10)
        self.create_subscription(String, '/rit/manipulation/status', self.manipulation_status_callback, 10)
        self.create_subscription(String, '/rit/navigation/status', self.navigation_status_callback, 10)
        self.create_subscription(Int32, '/erc/shelf_column_identification', self.shelf_column_callback, 10)
        self.create_subscription(Int32, '/erc/shelf_row_identification', self.shelf_row_callback, 10)

        self.current_state = None
        self.state_start_time = None
        self.timeout_timer = None
        self.delay_timer = None
        self.retry_count = 0

        # Guards against race conditions / bad messages
        self.mission_failed = False   # True once stop_all() has fired permanently
        self.advancing = False        # True while a success is being processed, before next state starts

        self.shelf_column_result = None
        self.shelf_row_result = None

        self.enter_state('TUCK_ARMS')

    # ---------- generic state machinery ----------

    def enter_state(self, state_name, is_retry=False):
        self.current_state = state_name
        self.advancing = False
        if not is_retry:
            self.retry_count = 0

        self.task_state_pub.publish(String(data=state_name))
        self.get_logger().info(f'Entering state: {state_name} (attempt {self.retry_count + 1})')

        if state_name == 'DONE':
            self.get_logger().info('Mission complete: DONE.')
            return

        if state_name in MANIPULATION_STATES:
            msg = String()
            msg.data = state_name
            self.manipulation_command_pub.publish(msg)
            self.get_logger().info(f'Published {state_name} to manipulation. Waiting for SUCCEEDED:{state_name}...')
            self.start_timeout(state_name)

        elif state_name in NAVIGATION_STATES:
            msg = String()
            msg.data = state_name
            self.navigation_command_pub.publish(msg)
            self.get_logger().info(f'Published {state_name} to navigation. Waiting for SUCCEEDED:{state_name}...')
            self.start_timeout(state_name)

        elif state_name in ('IDENTIFY_SHELF_COLUMN', 'IDENTIFY_BOOK_AND_ROW'):
            self.get_logger().info(f'Waiting for perception data for {state_name}...')
            self.start_timeout(state_name)

        # Guards against race conditions / bad messages
        self.mission_failed = False   # True once stop_all() has fired permanently
        self.advancing = False        # True while a success is being processed, before next state starts

        self.shelf_column_result = None
        self.shelf_row_result = None

        self.enter_state('TUCK_ARMS')

    # ---------- generic state machinery ----------

    def enter_state(self, state_name, is_retry=False):
        self.current_state = state_name
        self.advancing = False
        if not is_retry:
            self.retry_count = 0

        self.task_state_pub.publish(String(data=state_name))
        self.get_logger().info(f'Entering state: {state_name} (attempt {self.retry_count + 1})')

        if state_name == 'DONE':
            self.get_logger().info('Mission complete: DONE.')
            return

        if state_name in MANIPULATION_STATES:
            msg = String()
            msg.data = state_name
            self.manipulation_command_pub.publish(msg)
            self.get_logger().info(f'Published {state_name} to manipulation. Waiting for SUCCEEDED:{state_name}...')
            self.start_timeout(state_name)

        elif state_name in NAVIGATION_STATES:
            msg = String()
            msg.data = state_name
            self.navigation_command_pub.publish(msg)
            self.get_logger().info(f'Published {state_name} to navigation. Waiting for SUCCEEDED:{state_name}...')
            self.start_timeout(state_name)

        elif state_name in ('IDENTIFY_SHELF_COLUMN', 'IDENTIFY_BOOK_AND_ROW'):
            self.get_logger().info(f'Waiting for perception data for {state_name}...')
            self.start_timeout(state_name)

    def start_timeout(self, state_name):
        self.state_start_time = time.monotonic()
        self.timeout_timer = self.create_timer(1.0, lambda: self.check_timeout(state_name))

    def cancel_timeout(self):
        if self.timeout_timer is not None:
            self.timeout_timer.cancel()
            self.timeout_timer = None

    def cancel_delay(self):
        if self.delay_timer is not None:
            self.delay_timer.cancel()
            self.delay_timer = None

    def check_timeout(self, state_name):
        if self.mission_failed:
            return
        if self.current_state != state_name:
            return
        elapsed = time.monotonic() - self.state_start_time
        if elapsed > TIMEOUTS[state_name]:
            self.cancel_timeout()
            self.handle_failure(state_name, f'TIMEOUT after {TIMEOUTS[state_name]}s')

    def handle_failure(self, state_name, reason):
        if self.mission_failed:
            return
        if self.retry_count < MAX_RETRIES:
            self.retry_count += 1
            self.get_logger().warn(
                f'{state_name} failed ({reason}). Retrying ({self.retry_count}/{MAX_RETRIES})...'
            )
            self.enter_state(state_name, is_retry=True)
        else:
            self.get_logger().error(
                f'{state_name} failed permanently after {MAX_RETRIES} retries ({reason}). Stopping all nodes.'
            )
            self.stop_all()

    def stop_all(self):
        self.mission_failed = True
        self.cancel_timeout()
        self.cancel_delay()
        stop_msg = String()
        stop_msg.data = 'STOP'
        self.manipulation_command_pub.publish(stop_msg)
        self.navigation_command_pub.publish(stop_msg)
        self.task_state_pub.publish(String(data='FAILED'))
        self.get_logger().warn('Published STOP to manipulation and navigation. Mission aborted.')

    def advance_after_delay(self, delay_seconds=1.0):
        if self.mission_failed or self.advancing:
            return
        self.advancing = True

        finished_state = self.current_state
        index = STATE_ORDER.index(finished_state)
        next_state = STATE_ORDER[index + 1]

        self.cancel_delay()

        def _fire():
            self.delay_timer.cancel()
            self.delay_timer = None
            self.enter_state(next_state)

        self.delay_timer = self.create_timer(delay_seconds, _fire)

    # ---------- status callbacks ----------

    def manipulation_status_callback(self, msg):
        if self.mission_failed or self.advancing:
            return
        status = msg.data
        state = self.current_state
        if state not in MANIPULATION_STATES:
            return

        if status == f'SUCCEEDED:{state}':
            self.cancel_timeout()
            self.get_logger().info(f'{state} succeeded.')
            self.advance_after_delay()
        elif status.startswith(f'FAILED:{state}'):
            self.cancel_timeout()
            self.handle_failure(state, status)
        else:
            self.get_logger().info(f'Ignoring unrelated/interim manipulation status in {state}: {status}')

    def navigation_status_callback(self, msg):
        if self.mission_failed or self.advancing:
            return
        status = msg.data
        state = self.current_state
        if state not in NAVIGATION_STATES:
            return

        if status == f'SUCCEEDED:{state}':
            self.cancel_timeout()
            self.get_logger().info(f'{state} succeeded.')
            self.advance_after_delay()
        elif status.startswith(f'FAILED:{state}'):
            self.cancel_timeout()
            self.handle_failure(state, status)
        else:
            self.get_logger().info(f'Ignoring unrelated/interim navigation status in {state}: {status}')

    def shelf_column_callback(self, msg):
        if self.mission_failed or self.advancing:
            return
        if self.current_state != 'IDENTIFY_SHELF_COLUMN':
            return
        self.shelf_column_result = msg.data
        self.cancel_timeout()
        self.get_logger().info(f'IDENTIFY_SHELF_COLUMN succeeded: column={msg.data}')
        self.advance_after_delay()

    def shelf_row_callback(self, msg):
        if self.mission_failed or self.advancing:
            return
        if self.current_state != 'IDENTIFY_BOOK_AND_ROW':
            return
        self.shelf_row_result = msg.data
        self.cancel_timeout()
        self.get_logger().info(f'IDENTIFY_BOOK_AND_ROW succeeded: row={msg.data}')
        self.advance_after_delay()


def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
