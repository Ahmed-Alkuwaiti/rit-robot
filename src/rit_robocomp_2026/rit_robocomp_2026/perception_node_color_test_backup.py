import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


CAMERA_TOPIC = '/head_front_camera/head_front_camera/color/image_raw'
IMAGE_PATH = '/tmp/tiago_camera_frame.png'
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
        self.bridge = CvBridge()
        self.frame_count = 0
        self.image_saved = False
        self.camera_subscription = self.create_subscription(Image, CAMERA_TOPIC, self.camera_callback, 10,)
        self.get_logger().info(
            f'Waiting for camera images on {CAMERA_TOPIC}')

    def camera_callback(self, message):
        self.frame_count += 1
        frame = self.bridge.imgmsg_to_cv2(message,desired_encoding='bgr8',)

        if self.frame_count >= 30 and not self.image_saved:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            red_mask_1 = cv2.inRange(hsv_frame,red_LOWER_1,red_UPPER_1,)
            red_mask_2 = cv2.inRange(hsv_frame,red_LOWER_2,red_UPPER_2,)
            red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
            
            blue_mask = cv2.inRange(hsv_frame,blue_LOWER,blue_UPPER)
            
            yllw_mask = cv2.inRange(hsv_frame,yllw_LOWER,yllw_UPPER)
            
            grn_mask = cv2.inRange(hsv_frame,grn_LOWER,grn_UPPER)
            
            contours_red, _ = cv2.findContours(red_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)
            contours_blue, _ = cv2.findContours(blue_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)
            contours_yllw, _ = cv2.findContours(yllw_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)
            contours_grn, _ = cv2.findContours(grn_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE,)
            
            
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
            all_images_saved = all((frame_success,red_mask_success, red_annotated_success,  blue_mask_success, blue_annotated_success, yllw_mask_success, yllw_annotated_success, grn_mask_success, grn_annotated_success, ))
            if all_images_saved:
                self.image_saved = True
                self.get_logger().info(f'All images saved' f' total detected regions = {total_detected_regions}, red regions: {red_detected_regions}, blue regions: {blue_detected_regions}, yellow regions: {yllw_detected_regions}, green regions: {grn_detected_regions}')
            else:
                self.get_logger().error(
                    'Could not save one or more of the output images/pics'
                )

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
