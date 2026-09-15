import cv2
import rclpy
import os
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from ament_index_python.packages import get_package_share_directory
from datetime import datetime

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
        self.book_color = (self.get_parameter('book_color').value.lower())
        self.shelf_column_number = int(self.get_parameter('shelf_column_number').value)
        if not (1 <= self.shelf_column_number <= 5):
            raise ValueError('shelf_column_number must be between 1 and 5')

        self.get_logger().info(f'Requested book color: {self.book_color}') 

        self.bridge = CvBridge()
        self.frame_count = 0
        self.camera_subscription = self.create_subscription(Image, CAMERA_TOPIC, self.camera_callback, 10,)
        self.get_logger().info(
            f'Waiting for camera images on {CAMERA_TOPIC}')        

        self.column_publisher = self.create_publisher(Int32,'/erc/shelf_column_identification',10)

        self.row_publisher = self.create_publisher(Int32,'/erc/shelf_row_identification',10)
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

            if self.target_physical_column is None:
                if len(numbers) == 5:
                    numbers_sorted = sorted(numbers,key=lambda item: item[1])

                    for index, number in enumerate(numbers_sorted):
                        digit = number[0]
 
                        if digit == self.shelf_column_number:
                            self.target_physical_column = index + 1
                            msg = Int32()
                            msg.data = self.shelf_column_number
                            self.shelf_column_publisher.publish(msg)
                            self.get_logger().info(f'number searched for: {self.shelf_column_number},'f' found target marker {self.shelf_column_number} at 'f' cabinet number {self.target_physical_column} from the left')
                           
                            break
 
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


                if len(clustered_lines) >= 5:
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

                            break
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

        _, dark_pixel_mask = cv2.threshold(number_region,90,255,cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(dark_pixel_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
            numbers.append((x, y, width, height))
        numbers.sort(key=lambda box: box[0])

        numbers_frame = frame.copy()
        recognized_numbers = []

        for index, (x, y, width, height) in enumerate(numbers):
            cv2.rectangle(numbers_frame,(x, y), (x + width, y + height),(255, 0, 255), 2)

            center_x = x + width // 2
            center_y = y + height // 2

            padding = 4
            crop_left = max(0, x - padding)
            crop_top = max(0, y - padding)
            crop_right = min(image_width,x + width + padding)
            crop_bottom = min(image_height,y + height + padding)

            number_image = gray_frame[crop_top:crop_bottom,crop_left:crop_right]
            digit, difference = self.read_number(number_image)

            if difference > 600: 
                continue

            recognized_numbers.append((digit,x,y,width, height))
            self.get_logger().info(f'Detected digit {digit}, difference {difference}, position ({x},{y})')

            cv2.circle(numbers_frame,(center_x, center_y),4,(0, 255, 0),-1)

            cv2.imwrite(NUMBER_CROP_PATH.format(index),number_image)
            cv2.imwrite(ORDERED_NUMBER_PATH.format(digit),number_image)

        recognized_numbers.sort(key=lambda information: information[0])
        cv2.imwrite(NUMBERS_PATH, numbers_frame)

        self.get_logger().info(f'recognized numbers: {recognized_numbers}')

        return recognized_numbers


    def read_number(self, number_image):
        number_mask = cv2.threshold(number_image,90,255,cv2.THRESH_BINARY_INV)[1]

        number_mask = cv2.resize(number_mask,(40,60))

        best_number = None
        best_difference = 99999999999999

        for digit in range(1, 6):
            template_path = os.path.join(self.template_dir,f'template_{digit}.png')

            template = cv2.imread(template_path,cv2.IMREAD_GRAYSCALE)

            if template is None:
                self.get_logger().warning(f'Could not load template {digit}: {template_path}')
                continue

            template_mask = cv2.threshold(template,90,255,cv2.THRESH_BINARY_INV)[1]

            template_mask = cv2.resize(template_mask,(40, 60))

            difference_image = cv2.absdiff(number_mask,template_mask)

            difference = cv2.countNonZero(difference_image)

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
