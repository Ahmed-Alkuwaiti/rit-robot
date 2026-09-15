import cv2
import numpy as np
import rclpy
import os
import time
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32, String
from tf2_ros import Buffer, TransformException, TransformListener
import tf2_geometry_msgs  
from ament_index_python.packages import get_package_share_directory
from datetime import datetime

from .target_geometry import median_depth_metres, pixel_to_optical_point, foreground_depth_metres

ERC_IMAGES_DIR = '/erc_images'
CAMERA_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
IMAGE_PATH = '/tmp/tiago_camera_frame.png'
SHELF_EDGES_PATH = '/tmp/shelf_edges.png'
SHELF_LINES_PATH = '/tmp/shelf_lines.png'
NUMBERS_PATH = '/tmp/number.png'
NUMBER_CROP_PATH = '/tmp/number_{}.png'
ORDERED_NUMBER_PATH = '/tmp/ordered_number_{}.png'

red_MASK_PATH = '/tmp/red_mask.png'
red_ANNOTATED_IMAGE_PATH = '/tmp/red_detection.png'
red_LOWER_1 = (0, 120, 70)
red_UPPER_1 = (10, 255, 255)
red_LOWER_2 = (170, 120, 70)
red_UPPER_2 = (179, 255, 255)
MIN_CONTOUR_AREA = 50

blue_MASK_PATH = '/tmp/blue_mask.png'
blue_ANNOTATED_IMAGE_PATH = '/tmp/blue_detection.png'
blue_LOWER = (100, 120, 70)
blue_UPPER = (130, 255, 255)

yllw_MASK_PATH = '/tmp/yllw_mask.png'
yllw_ANNOTATED_IMAGE_PATH = '/tmp/yllw_detection.png'
yllw_LOWER = (20, 120, 70)
yllw_UPPER = (35, 255, 255)

grn_MASK_PATH = '/tmp/grn_mask.png'
grn_ANNOTATED_IMAGE_PATH = '/tmp/grn_detection.png'
grn_LOWER = (40, 120, 70)
grn_UPPER = (85, 255, 255)

class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        self.target_physical_column = None
        os.makedirs(ERC_IMAGES_DIR, exist_ok=True)
        self.final_images_saved = False
        self.shelf_column_publisher = self.create_publisher(Int32,'/erc/shelf_column_identification',10)

        package_share = get_package_share_directory('rit_robocomp_2026')
        self.template_dir = os.path.join(package_share, 'templates')

        self.declare_parameter('book_color', 'red')
        self.declare_parameter('shelf_column_number', 1)
        self.declare_parameter(
            'depth_topic',
            '/head_front_camera/head_front_camera/depth/image_rect_raw')
        self.declare_parameter(
            'camera_info_topic',
            '/head_front_camera/head_front_camera/depth/camera_info')
        self.declare_parameter('target_frame', 'base_footprint')
        self.declare_parameter('depth_window_radius', 4)
        self.declare_parameter('depth_timeout', 0.5)
        self.declare_parameter('bin_min_area', 1500.0)
        self.declare_parameter('bin_min_width_height_ratio', 0.8)
        self.declare_parameter('bin_recess_depth', 0.06)
        self.declare_parameter('bin_opening_min_pixels', 60)
        self.declare_parameter('book_offset_x', 0.0)
        self.declare_parameter('book_offset_y', 0.0)
        self.declare_parameter('book_offset_z', 0.0)
        self.declare_parameter('bin_offset_x', 0.0)
        self.declare_parameter('bin_offset_y', 0.0)
        self.declare_parameter('bin_offset_z', 0.0)
        self.book_color = (self.get_parameter('book_color').value.lower())
        self.shelf_column_number = int(self.get_parameter('shelf_column_number').value)
        if not (1 <= self.shelf_column_number <= 5):
            raise ValueError('shelf_column_number must be between 1 and 5')

        self.get_logger().info(f'Requested book color: {self.book_color}') 

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_depth_time = None
        self.camera_info = None
        self.target_mode = 'BOOK'
        # Once the requested book has been identified from the full shelf
        # geometry, keep a lightweight image-space track of that SAME colored
        # rectangle.  During alignment/approach the camera can lose one of the
        # five number markers or a shelf line; that must not suppress
        # book_pixel_x/book_target while the selected book itself is still
        # plainly visible.
        self.tracked_book_center = None
        self.tracked_book_bbox = None
        self.tracked_book_last_time = None
        self.book_track_max_jump_px = 140.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.book_target_publisher = self.create_publisher(PoseStamped, '/rit/perception/book_target', 10)
        self.book_pixel_x_publisher = self.create_publisher(Int32,'/rit/perception/book_pixel_x',10)
        self.book_pixel_y_publisher = self.create_publisher(Int32,'/rit/perception/book_pixel_y',10)
        self.book_bbox_height_publisher = self.create_publisher(
            Int32, '/rit/perception/book_bbox_height', 10)
        self.bin_target_publisher = self.create_publisher(PoseStamped, '/rit/perception/bin_target', 10)
        self.create_subscription(Image, self.get_parameter('depth_topic').value,self.depth_callback, 10)
        self.create_subscription(CameraInfo, self.get_parameter('camera_info_topic').value,self.camera_info_callback, 10)
        self.create_subscription(
            String, '/rit/perception/command', self.perception_command_callback, 10)
        self.frame_count = 0
        self.camera_subscription = self.create_subscription(Image, CAMERA_TOPIC, self.camera_callback, 10,)
        self.get_logger().info(
            f'Waiting for camera images on {CAMERA_TOPIC}')        

        self.column_publisher = self.create_publisher(Int32,'/erc/shelf_column_identification',10)

        self.row_publisher = self.create_publisher(Int32,'/erc/shelf_row_identification',10)

    def depth_callback(self, message):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                message, desired_encoding='passthrough')
            self.latest_depth_stamp = message.header.stamp
            self.latest_depth_time = time.monotonic()
        except Exception as error:
            self.get_logger().warning(f'Could not decode depth image: {error}')

    def camera_info_callback(self, message):
        self.camera_info = message

    def perception_command_callback(self, message):
        mode = message.data.strip().upper()
        if mode in ('BOOK', 'BIN'):
            if mode != self.target_mode:
                # A genuine mode change starts a new visual target track.
                self.tracked_book_center = None
                self.tracked_book_bbox = None
                self.tracked_book_last_time = None
            self.target_mode = mode
            self.get_logger().info(f'Perception target mode: {mode}')

    def publish_book_detection(self, center_x, center_y, x, y, width, height):
        """Publish the selected book and refresh its lightweight 2D track."""
        pixel_msg = Int32()
        pixel_msg.data = int(center_x)
        self.book_pixel_x_publisher.publish(pixel_msg)

        pixel_y_msg = Int32()
        pixel_y_msg.data = int(center_y)
        self.book_pixel_y_publisher.publish(pixel_y_msg)

        height_msg = Int32()
        height_msg.data = int(height)
        self.book_bbox_height_publisher.publish(height_msg)

        self.publish_target(
            (int(center_x), int(center_y)),
            self.book_target_publisher,
            'BOOK',
            self.offsets('book'),
            bbox=(int(x), int(y), int(width), int(height)),
        )

        self.tracked_book_center = (float(center_x), float(center_y))
        self.tracked_book_bbox = (int(x), int(y), int(width), int(height))
        self.tracked_book_last_time = time.monotonic()
    def publish_target( self, pixel, publisher, label, offsets, bbox=None, depth_override=None, ):
        if self.latest_depth is None or self.camera_info is None:
            self.get_logger().warning(
                f'Cannot publish {label}: waiting for registered depth/camera_info')
            return
        if (
            self.latest_depth_time is None
            or time.monotonic() - self.latest_depth_time
            > float(self.get_parameter('depth_timeout').value)
        ):
            self.get_logger().warning(
                f'Cannot publish {label}: depth image is stale')
            return
        u, v = pixel
        radius = int(
            self.get_parameter('depth_window_radius').value
        )

        depth = depth_override

        if depth is None and bbox is not None:
            bx, by, bw, bh = bbox

            depth = foreground_depth_metres(
                self.latest_depth,
                bx,
                by,
                bw,
                bh,
            )

        if depth is None:
            depth = median_depth_metres(
                self.latest_depth,
                u,
                v,
                radius,
            )

        if depth is None and bbox is not None:
            bx, by, bw, bh = bbox
            candidates = [
                (0.35, 0.50), (0.65, 0.50),
                (0.50, 0.35), (0.50, 0.65),
                (0.35, 0.35), (0.65, 0.35),
                (0.35, 0.65), (0.65, 0.65),
            ]
            for fx, fy in candidates:
                cu = int(round(bx + fx * max(1, bw - 1)))
                cv = int(round(by + fy * max(1, bh - 1)))
                candidate_depth = median_depth_metres(
                    self.latest_depth, cu, cv, radius
                )
                if candidate_depth is not None:
                    u, v = cu, cv
                    depth = candidate_depth
                    break

        # Last depth-only fallback: median of the inner 70% of the detected
        # book box.  Keep the RGB centre for ray projection.
        if depth is None and bbox is not None:
            bx, by, bw, bh = bbox
            ix = max(1, int(round(0.15 * bw)))
            iy = max(1, int(round(0.15 * bh)))
            x0 = max(0, bx + ix)
            x1 = min(self.latest_depth.shape[1], bx + bw - ix)
            y0 = max(0, by + iy)
            y1 = min(self.latest_depth.shape[0], by + bh - iy)
            if x1 > x0 and y1 > y0:
                values = np.asarray(
                    self.latest_depth[y0:y1, x0:x1], dtype=float
                ).reshape(-1)
                values = values[np.isfinite(values) & (values > 0.0)]
                if values.size:
                    depth = float(np.median(values))
                    if depth > 20.0:
                        depth /= 1000.0

        info = self.camera_info
        point = pixel_to_optical_point(
            u, v, depth, info.k[0], info.k[4], info.k[2], info.k[5])
        if point is None:
            self.get_logger().warning(f'Cannot publish {label}: invalid depth')
            return
        source = PoseStamped()
        source.header.stamp = self.latest_depth_stamp
        source.header.frame_id = info.header.frame_id
        source.pose.position.x, source.pose.position.y, source.pose.position.z = point
        source.pose.orientation.w = 1.0
        target_frame = self.get_parameter('target_frame').value
        try:
            # Prefer the transform at the depth-image timestamp so the 3D point
            # and robot pose are temporally consistent.
            target = self.tf_buffer.transform(
                source, target_frame, timeout=Duration(seconds=0.15))
        except TransformException as exact_error:
            # Gazebo can publish the camera image slightly ahead of the newest
            # TF sample.  In that case an exact-time lookup reports
            # "extrapolation into the future" even though the camera->base TF
            # itself is available.  Retry with timestamp zero, which asks tf2
            # for the latest available transform instead of dropping the target.
            latest_source = PoseStamped()
            latest_source.header.frame_id = source.header.frame_id
            latest_source.header.stamp = Time().to_msg()
            latest_source.pose = source.pose
            try:
                target = self.tf_buffer.transform(
                    latest_source, target_frame,
                    timeout=Duration(seconds=0.15))
            except TransformException as latest_error:
                self.get_logger().warning(
                    f'Cannot transform {label}: exact={exact_error}; latest={latest_error}')
                return
        target.pose.position.x += offsets[0]
        target.pose.position.y += offsets[1]
        target.pose.position.z += offsets[2]
        publisher.publish(target)

    def offsets(self, prefix):
        return tuple(float(self.get_parameter(f'{prefix}_offset_{axis}').value)
                     for axis in ('x', 'y', 'z'))

    def camera_callback(self, message):
        self.frame_count += 1
        frame = self.bridge.imgmsg_to_cv2(message,desired_encoding='bgr8',)

        if self.frame_count % 3 == 0:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            contrast_enhancer = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))
            enhanced_gray_frame = contrast_enhancer.apply(gray_frame)
            shelf_edges = cv2.Canny(enhanced_gray_frame,25,80)
            lines = cv2.HoughLinesP(shelf_edges,1,3.14159/180,60,minLineLength=80,maxLineGap=25)

            shelf_lines_frame = frame.copy()
            detected_line_candidates = 0
            horizontal_line_y_positions = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    horizontal_distance = abs(x2 - x1)
                    vertical_distance = abs(y2 - y1)
                    if horizontal_distance == 0:
                        continue
                    if vertical_distance / horizontal_distance > 0.4:
                        continue

                    line_center_y = (y1 + y2) // 2
                    horizontal_line_y_positions.append(line_center_y)

                    cv2.line(shelf_lines_frame,(x1, y1),(x2, y2),(255, 0, 255),2)
                    detected_line_candidates += 1

            red_mask_1 = cv2.inRange(hsv_frame,red_LOWER_1,red_UPPER_1,)
            red_mask_2 = cv2.inRange(hsv_frame,red_LOWER_2,red_UPPER_2,)
            red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

            blue_mask = cv2.inRange(hsv_frame,blue_LOWER,blue_UPPER)

            yllw_mask = cv2.inRange(hsv_frame,yllw_LOWER,yllw_UPPER)

            grn_mask = cv2.inRange(hsv_frame,grn_LOWER,grn_UPPER)

            color_masks = {'red': red_mask,'blue': blue_mask,'yellow': yllw_mask,'green': grn_mask}

            if self.book_color not in color_masks:
                self.get_logger().error(f'Unsupported book color: {self.book_color}')
                return
            
            requested_mask = color_masks[self.book_color]
            color_contours = {}

            for color, mask in color_masks.items():
                contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)
                color_contours[color] = contours

            requested_contours = color_contours[self.book_color]

            requested_annotated_frame = frame.copy()
            requested_detected_regions = 0
            requested_centers = []
            book_published_this_frame = False

            for contour in requested_contours:
                area = cv2.contourArea(contour)

                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, width, height = cv2.boundingRect(contour)
                center_x = x + width// 2
                center_y = y + height // 2
                requested_centers.append((center_x, center_y, x, y, width, height))
                cv2.rectangle(requested_annotated_frame,(x, y),(x + width, y + height),(0, 255, 0),2)
                cv2.circle(requested_annotated_frame,(center_x, center_y),4,(255, 0, 255),-1)
                requested_detected_regions += 1

            if self.target_mode == 'BIN':
                min_bin_area = float(
                    self.get_parameter(
                        'bin_min_area'
                    ).value
                )

                min_bin_ratio = float(
                    self.get_parameter(
                        'bin_min_width_height_ratio'
                    ).value
                )

                recess_depth = float(
                    self.get_parameter(
                        'bin_recess_depth'
                    ).value
                )

                min_opening_pixels = int(
                    self.get_parameter(
                        'bin_opening_min_pixels'
                    ).value
                )

                bin_contours = []

                for contour in color_contours['red']:
                    area = cv2.contourArea(contour)

                    if area < min_bin_area:
                        continue

                    x, y, width, height = (
                        cv2.boundingRect(contour)
                    )

                    if height <= 0:
                        continue

                    if (
                        width / float(height)
                        < min_bin_ratio
                    ):
                        continue

                    bin_contours.append(contour)

                if not bin_contours:
                    self.get_logger().info(
                        'BIN search: no large red '
                        'candidate visible'
                    )
                    return

                bin_contour = max(
                    bin_contours,
                    key=cv2.contourArea,
                )

                x, y, width, height = (
                    cv2.boundingRect(bin_contour)
                )

                center_x = x + width // 2
                center_y = y + height // 2

                self.get_logger().info(
                    f'Large red BIN candidate: '
                    f'x={x}, y={y}, '
                    f'w={width}, h={height}, '
                    f'area={cv2.contourArea(bin_contour):.0f}'
                )

                # If depth is not ready yet, the large red
                # object is still enough for SEARCH_BIN.
                if (
                    self.latest_depth is None
                    or self.camera_info is None
                ):
                    return

                depth_height, depth_width = (
                    self.latest_depth.shape[:2]
                )

                x0 = max(0, x)
                y0 = max(0, y)

                x1 = min(
                    depth_width,
                    x + width,
                )

                y1 = min(
                    depth_height,
                    y + height,
                )

                if x1 <= x0 or y1 <= y0:
                    return

                depth_roi = np.asarray(
                    self.latest_depth[
                        y0:y1,
                        x0:x1
                    ],
                    dtype=float,
                )

                red_roi = (
                    red_mask[
                        y0:y1,
                        x0:x1
                    ] > 0
                )

                if depth_roi.shape != red_roi.shape:
                    return

                valid_depth = (
                    np.isfinite(depth_roi)
                    & (depth_roi > 0.0)
                )

                valid_values = depth_roi[
                    valid_depth
                ]

                if valid_values.size == 0:
                    return

                # Gazebo depth may arrive in millimetres
                # depending on image encoding.
                if np.median(valid_values) > 20.0:
                    depth_roi = depth_roi / 1000.0

                    valid_depth = (
                        np.isfinite(depth_roi)
                        & (depth_roi > 0.0)
                    )

                rim_values = depth_roi[
                    valid_depth
                    & red_roi
                ]

                # We need enough red depth pixels to
                # estimate the front/rim of the bin.
                if rim_values.size < 20:
                    self.publish_target(
                        (center_x, center_y),
                        self.bin_target_publisher,
                        'BIN',
                        self.offsets('bin'),
                        bbox=(
                            x,
                            y,
                            width,
                            height,
                        ),
                    )
                    return

                rim_depth = float(
                    np.median(rim_values)
                )

                # Ignore the outside edge of the bounding
                # box. The hollow must be inside the bin.
                interior = np.zeros(
                    depth_roi.shape,
                    dtype=bool,
                )

                margin_x = max(
                    2,
                    int(0.10 * width),
                )

                margin_y = max(
                    2,
                    int(0.10 * height),
                )

                interior[
                    margin_y:
                    max(
                        margin_y + 1,
                        depth_roi.shape[0]
                        - margin_y
                    ),
                    margin_x:
                    max(
                        margin_x + 1,
                        depth_roi.shape[1]
                        - margin_x
                    ),
                ] = True

                # The hollow is:
                #   - inside the red bin
                #   - not itself red
                #   - deeper than the red rim
                opening_mask = (
                    valid_depth
                    & interior
                    & (~red_roi)
                    & (
                        depth_roi
                        > rim_depth + recess_depth
                    )
                    & (
                        depth_roi
                        < rim_depth + 0.60
                    )
                )

                opening_u8 = (
                    opening_mask.astype(
                        np.uint8
                    )
                    * 255
                )

                (
                    number_of_labels,
                    labels,
                    stats,
                    centroids,
                ) = cv2.connectedComponentsWithStats(
                    opening_u8,
                    8,
                )

                best_label = None
                best_area = 0

                for label_index in range(
                    1,
                    number_of_labels,
                ):
                    area = stats[
                        label_index,
                        cv2.CC_STAT_AREA,
                    ]

                    if (
                        area >= min_opening_pixels
                        and area > best_area
                    ):
                        best_area = area
                        best_label = label_index

                if best_label is None:
                    # We can already see the large red bin.
                    # Publish its centre so SEARCH_BIN /
                    # ALIGN_BIN can move closer. At closer
                    # range the hollow becomes measurable.
                    self.get_logger().info(
                        'BIN visible but hollow depth '
                        'not resolved yet; approaching '
                        'red candidate'
                    )

                    self.publish_target(
                        (center_x, center_y),
                        self.bin_target_publisher,
                        'BIN',
                        self.offsets('bin'),
                        bbox=(
                            x,
                            y,
                            width,
                            height,
                        ),
                    )

                    return

                opening_x = int(
                    round(
                        centroids[
                            best_label,
                            0,
                        ]
                    )
                )

                opening_y = int(
                    round(
                        centroids[
                            best_label,
                            1,
                        ]
                    )
                )

                opening_pixel_x = (
                    x0 + opening_x
                )

                opening_pixel_y = (
                    y0 + opening_y
                )

                self.get_logger().info(
                    f'BIN hollow found: '
                    f'pixel=('
                    f'{opening_pixel_x},'
                    f'{opening_pixel_y}), '
                    f'rim_depth={rim_depth:.3f} m, '
                    f'recess_pixels={best_area}'
                )

                # Important:
                # use the opening pixel for X/Y direction,
                # but use RIM depth rather than bottom depth.
                # This gives manipulation the centre of the
                # opening at the top plane of the bin.
                self.publish_target(
                    (
                        opening_pixel_x,
                        opening_pixel_y,
                    ),
                    self.bin_target_publisher,
                    'BIN',
                    self.offsets('bin'),
                    depth_override=rim_depth,
                )

                return

            requested_mask_paths = {'red': red_MASK_PATH,'blue': blue_MASK_PATH,'yellow': yllw_MASK_PATH,'green': grn_MASK_PATH}
            requested_annotated_paths = {'red': red_ANNOTATED_IMAGE_PATH,'blue': blue_ANNOTATED_IMAGE_PATH,'yellow': yllw_ANNOTATED_IMAGE_PATH,'green': grn_ANNOTATED_IMAGE_PATH}

            numbers = self.find_numbers(frame,gray_frame)

            target_number_box = None

            for digit, x, y, width, height in numbers:
                if digit == self.shelf_column_number:
                    target_number_box = (x, y, width, height)
                    break

            if target_number_box is not None:
                x, y, width, height = target_number_box
                self.get_logger().info(f'Target shelf number {self.shelf_column_number} found at 'f'({x}, {y})')

            else: 
                self.get_logger().warning(f'Target shelf number {self.shelf_column_number} was not found')

            if len(numbers) == 5:
                numbers_left_to_right = sorted(numbers, key=lambda item: item[1])

                current_physical_column = None

                for index, number in enumerate(numbers_left_to_right):
                    digit = number[0]

                    if digit == self.shelf_column_number:
                        current_physical_column = index + 1
                        break

                if current_physical_column is not None:
                    self.target_physical_column = current_physical_column

                    column_msg = Int32()
                    column_msg.data = self.shelf_column_number
                    self.shelf_column_publisher.publish(column_msg)

                    self.get_logger().info(f'number searched for: {self.shelf_column_number}, ' f'found at physical cabinet ' f'{self.target_physical_column} from the left')
            else:
                self.get_logger().warning(f'Need 5 numbers to determine physical column, found {len(numbers)}')

            if self.target_physical_column is not None and len(requested_centers) > 0:
                line_y_positions = sorted(horizontal_line_y_positions)
                clustered_lines = []

                for y in line_y_positions:
                    if len(clustered_lines) == 0:
                        clustered_lines.append(y)

                    elif abs(y - clustered_lines[-1]) <= 6:
                        clustered_lines[-1] = int((clustered_lines[-1] + y) / 2)
                    else:
                        clustered_lines.append(y)


                if len(clustered_lines) >= 5 and len(numbers) >= 5:
                    top_y = clustered_lines[0]
                    bottom_y = clustered_lines[-1]

                    usable_height = bottom_y - top_y
                    row_height = usable_height / 6.0

                    numbers_left_to_right = sorted(numbers,key=lambda item: item[1])

                    marker_centers_x = [x + width // 2 for digit, x, y, width, height in numbers_left_to_right]

                    column_index = self.target_physical_column - 1

                    if column_index == 0:
                        left_boundary = 0
                    else:
                        left_boundary = (marker_centers_x[column_index - 1]+ marker_centers_x[column_index]) // 2

                    if column_index == 4:
                        right_boundary = frame.shape[1]
                    else:
                        right_boundary = (marker_centers_x[column_index] + marker_centers_x[column_index + 1]) // 2

                    column_annotated_frame = frame.copy()
                    cv2.rectangle(column_annotated_frame,(left_boundary, 0),(right_boundary, frame.shape[0]),(0, 0, 255),3)
                    for center_x, center_y, x, y, width, height in requested_centers:
                        if not (left_boundary <= center_x < right_boundary):
                            continue

                        row_index = int((center_y - top_y) / row_height)

                        if 1 <= row_index <= 4:
                            target_row = row_index

                            row_msg = Int32()
                            row_msg.data = target_row
                            self.row_publisher.publish(row_msg)
                            cv2.rectangle(requested_annotated_frame,(x, y),(x + width, y + height),(0, 0, 255),3)
                            cv2.putText(requested_annotated_frame,f'TARGET row {target_row}',(center_x - 40, center_y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),2)
                            if not self.final_images_saved:
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                target_book_path = os.path.join(ERC_IMAGES_DIR,f'target_book_{timestamp}.png')
                                target_column_path = os.path.join(ERC_IMAGES_DIR,f'target_column_{timestamp}.png')
                                cv2.imwrite(target_book_path,requested_annotated_frame)
                                cv2.imwrite(target_column_path,column_annotated_frame)
                                self.final_images_saved = True
                            self.get_logger().info(f'target {self.book_color} book in cabinet 'f' {self.target_physical_column} found in row {target_row}')

                            if self.target_mode == 'BOOK':
                                self.publish_book_detection(
                                    center_x, center_y, x, y, width, height
                                )
                                book_published_this_frame = True
                            break

            # After the target has been identified once, do NOT require all
            # five cabinet markers and all shelf lines on every subsequent
            # frame.  Alignment/approach changes the view and those landmarks
            # can leave the image even while the selected colored book remains
            # visible.  Track the nearest same-color rectangle from the
            # previous frame and keep publishing fresh pixel/depth targets.
            if (
                self.target_mode == 'BOOK'
                and not book_published_this_frame
                and self.tracked_book_center is not None
                and requested_centers
            ):
                previous_x, previous_y = self.tracked_book_center
                tracked = min(
                    requested_centers,
                    key=lambda item: (item[0] - previous_x) ** 2
                    + (item[1] - previous_y) ** 2,
                )
                center_x, center_y, x, y, width, height = tracked
                jump = ((center_x - previous_x) ** 2 +
                        (center_y - previous_y) ** 2) ** 0.5
                if jump <= self.book_track_max_jump_px:
                    self.publish_book_detection(
                        center_x, center_y, x, y, width, height
                    )
                    book_published_this_frame = True
                else:
                    self.get_logger().warning(f"BOOK TRACK REJECTED: jump={jump:.1f}px limit={self.book_track_max_jump_px:.1f}px prev=({previous_x:.1f},{previous_y:.1f}) new=({center_x},{center_y})")

            frame_success = cv2.imwrite(IMAGE_PATH, frame)
            requested_mask_success = cv2.imwrite(requested_mask_paths[self.book_color],requested_mask)
            requested_annotated_success = cv2.imwrite(requested_annotated_paths[self.book_color],requested_annotated_frame,)

            edges_success = cv2.imwrite(SHELF_EDGES_PATH,shelf_edges)
            lines_success = cv2.imwrite(SHELF_LINES_PATH,shelf_lines_frame)

            if frame_success and requested_mask_success and requested_annotated_success and edges_success and lines_success:
                self.get_logger().info(f'Saved requested {self.book_color} detection with {requested_detected_regions} regions and {detected_line_candidates} horizontal line candidates')
                self.get_logger().info(f'Horizontal line y positions: {sorted(horizontal_line_y_positions)}')
                self.get_logger().info(f'Found number boxes: {numbers}')
            else:
                self.get_logger().error(
                    'Could not save the requested color output images'
                )

            return
    def find_numbers(self, frame, gray_frame):
        image_height, image_width = gray_frame.shape

        top_limit = int(image_height * 0.45)
        number_region = gray_frame[:top_limit]

        _, dark_pixel_mask = cv2.threshold(
            number_region,
            90,
            255,
            cv2.THRESH_BINARY_INV
        )

        contours, _ = cv2.findContours(
            dark_pixel_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        numbers = []

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)

            if area < 40 or area > 1800:
                continue

            if width < 5 or width > 70:
                continue

            if height < 18 or height > 75:
                continue

            numbers.append(
                (x, y, width, height)
            )

        numbers.sort(
            key=lambda box: box[0]
        )

        numbers_frame = frame.copy()
        recognized_numbers = []

        for index, (x, y, width, height) in enumerate(numbers):
            cv2.rectangle(
                numbers_frame,
                (x, y),
                (x + width, y + height),
                (255, 0, 255),
                2
            )

            center_x = x + width // 2
            center_y = y + height // 2

            padding = 4

            crop_left = max(
                0,
                x - padding
            )

            crop_top = max(
                0,
                y - padding
            )

            crop_right = min(
                image_width,
                x + width + padding
            )

            crop_bottom = min(
                image_height,
                y + height + padding
            )

            number_image = gray_frame[
                crop_top:crop_bottom,
                crop_left:crop_right
            ]

            digit, difference = self.read_number(
                number_image
            )

            if difference > 575:
                continue

            recognized_numbers.append(
                (
                    digit,
                    x,
                    y,
                    width,
                    height
                )
            )

            self.get_logger().info(
                f'Detected digit {digit}, '
                f'difference {difference}, '
                f'position ({x},{y})'
            )

            cv2.circle(
                numbers_frame,
                (center_x, center_y),
                4,
                (0, 255, 0),
                -1
            )

            cv2.imwrite(
                NUMBER_CROP_PATH.format(index),
                number_image
            )

            cv2.imwrite(
                ORDERED_NUMBER_PATH.format(digit),
                number_image
            )

        recognized_numbers.sort(
            key=lambda information: information[0]
        )

        cv2.imwrite(
            NUMBERS_PATH,
            numbers_frame
        )

        self.get_logger().info(
            f'recognized numbers: {recognized_numbers}'
        )

        return recognized_numbers


    def read_number(self, number_image):
        number_mask = cv2.threshold(
            number_image,
            90,
            255,
            cv2.THRESH_BINARY_INV
        )[1]

        number_mask = cv2.resize(
            number_mask,
            (40, 60)
        )

        best_number = None
        best_difference = 99999999999999

        for digit in range(1, 6):
            template_path = os.path.join(
                self.template_dir,
                f'template_{digit}.png'
            )

            template = cv2.imread(
                template_path,
                cv2.IMREAD_GRAYSCALE
            )

            if template is None:
                self.get_logger().warning(
                    f'Could not load template {digit}: '
                    f'{template_path}'
                )
                continue

            template_mask = cv2.threshold(
                template,
                90,
                255,
                cv2.THRESH_BINARY_INV
            )[1]

            template_mask = cv2.resize(
                template_mask,
                (40, 60)
            )

            difference_image = cv2.absdiff(
                number_mask,
                template_mask
            )

            difference = cv2.countNonZero(
                difference_image
            )

            if difference < best_difference:
                best_difference = difference
                best_number = digit

        return best_number, best_difference
'''
            contours_red= color_contours["red"]
            contours_blue= color_contours["blue"]
            contours_yllw= color_contours["yellow"]
            contours_grn = color_contours["green"]


            red_annotated_frame = frame.copy()
            blue_annotated_frame = frame.copy()
            yllw_annotated_frame = frame.copy()
            grn_annotated_frame = frame.copy()

            red_detected_regions = 0
            blue_detected_regions = 0
            yllw_detected_regions = 0
            grn_detected_regions = 0

            for contour in contours_red:
                area = cv2.contourArea(contour)

                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, width, height = cv2.boundingRect(contour)

                cv2.rectangle(red_annotated_frame,(x, y),(x + width, y + height),(0, 255, 0),2,)
                red_detected_regions += 1

            red_mask_success = cv2.imwrite(red_MASK_PATH, red_mask)
            red_annotated_success = cv2.imwrite(red_ANNOTATED_IMAGE_PATH,red_annotated_frame,)

            for contour in contours_blue:
                area = cv2.contourArea(contour)

                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, width, height = cv2.boundingRect(contour)

                cv2.rectangle(blue_annotated_frame,(x, y),(x + width, y + height),(0, 255, 0),2,)
                blue_detected_regions += 1

            blue_mask_success = cv2.imwrite(blue_MASK_PATH, blue_mask)
            blue_annotated_success = cv2.imwrite(blue_ANNOTATED_IMAGE_PATH,blue_annotated_frame,)

            for contour in contours_yllw:
                area = cv2.contourArea(contour)

                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, width, height = cv2.boundingRect(contour)

                cv2.rectangle(yllw_annotated_frame,(x, y),(x + width, y + height),(0, 255, 0),2,)
                yllw_detected_regions += 1

            yllw_mask_success = cv2.imwrite(yllw_MASK_PATH, yllw_mask)
            yllw_annotated_success = cv2.imwrite(yllw_ANNOTATED_IMAGE_PATH,yllw_annotated_frame,)

            for contour in contours_grn:
                area = cv2.contourArea(contour)

                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, width, height = cv2.boundingRect(contour)

                cv2.rectangle(grn_annotated_frame,(x, y),(x + width, y + height),(0, 255, 0),2,)
                grn_detected_regions += 1

            grn_mask_success = cv2.imwrite(grn_MASK_PATH, grn_mask)
            grn_annotated_success = cv2.imwrite(grn_ANNOTATED_IMAGE_PATH,grn_annotated_frame,)
            
            
            frame_success = cv2.imwrite(IMAGE_PATH, frame)
            total_detected_regions = red_detected_regions + blue_detected_regions + yllw_detected_regions + grn_detected_regions
            all_images_saved = all((frame_success, red_mask_success,red_annotated_success,blue_mask_success,blue_annotated_success,yllw_mask_success,
                yllw_annotated_success,grn_mask_success,grn_annotated_success,))
            if all_images_saved:
                self.image_saved = True
                self.get_logger().info(f'All images saved total w
detected regions = {total_detected_regions}, 'f'red regions: {red_detected_regions}, 'f'blue regions: {blue_detected_regions}, 'f'yellow regions: {yllw_detected_regions}, 'f'green regions: {grn_detected_regions}')
            else:
                self.get_logger().error(
                    'Could not save one or more of the output images/pics'
                )'''

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

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
