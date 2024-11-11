import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
import os
from cv_bridge import CvBridge
from rclpy.timer import Timer


class ImagePublisher(Node):
    def __init__(self):
        super().__init__("image_publisher")

        self.publish_compressed = True
        self.publish_raw = False

        if self.publish_raw:
            self.get_logger().info("Publishing raw image")
            self.publisher_ = self.create_publisher(Image, "image", 10)

        if self.publish_compressed:
            self.get_logger().info("Publishing compressed image")
            self.publish_image_compressed_ = self.create_publisher(
                CompressedImage, "image/compressed", 10
            )

        self.cv_bridge = CvBridge()
        self.timer_ = self.create_timer(0.05, self.timer_callback)

    def publish_image(self, file_path):
        # Load the image from file
        image = cv2.imread(file_path)

        if not os.path.exists(file_path):
            self.get_logger().info(f"File not found: {file_path}")
            return

        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image to a ROS2 Image message
        ros_image = self.cv_bridge.cv2_to_imgmsg(image, "rgb8")
        # Publish the ROS2 Image message
        self.publisher_.publish(ros_image)

    def publish_image_compressed(self, file_path):
        if not os.path.exists(file_path):
            self.get_logger().info(f"File not found: {file_path}")
            return

        image_compressed_msg = CompressedImage()
        image_compressed_msg.header.stamp = self.get_clock().now().to_msg()
        image_compressed_msg.header.frame_id = "image"
        image_compressed_msg.format = "jpg"

        with open(file_path, "rb") as f:
            image_compressed_msg.data = f.read()

        self.publish_image_compressed_.publish(image_compressed_msg)

    def timer_callback(self):
        # Publish the image at file path 'image.jpg'
        filename = "/home/jgamero/Downloads/traffic.jpg"

        gauge_file = os.path.join(os.path.dirname(__file__), "gauge.jpg")
        # filename = gauge_file

        if self.publish_raw:
            self.publish_image(filename)

        if self.publish_compressed:
            self.publish_image_compressed(filename)


def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()

    rclpy.spin(image_publisher)

    image_publisher.timer_.cancel()
    image_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
