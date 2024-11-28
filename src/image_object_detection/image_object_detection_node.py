# Copyright 2023 Intelligent Behavior Robots, Inc.
#
# Licensed under the BSD 3-Clause License
#
# Author: Pablo Iñigo Blasco

import os
import sys
import numpy as np
from numpy import random

import cv2

import cv_bridge
from cv_bridge import CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import std_srvs.srv
from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from vision_msgs.msg import Detection2DArray, Detection2D
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "image_object_detection"


class ImageDetectObjectNode(Node):
    def __init__(self):
        super().__init__("image_object_detection_node")

        # Model parameters
        self.declare_parameter("model.image_size", 640)
        self.model_image_size = (
            self.get_parameter("model.image_size").get_parameter_value().integer_value
        )
        self.get_logger().info(f"model.image_size: {self.model_image_size}")

        self.declare_parameter("selected_detections", ["person", "car"])
        self.selected_detections = (
            self.get_parameter("selected_detections").get_parameter_value().string_array_value
        )
        self.get_logger().info(f"selected_detections: {self.selected_detections}")

        self.declare_parameter("model.confidence", 0.25)
        self.confidence = self.get_parameter("model.confidence").get_parameter_value().double_value
        self.get_logger().info(f"model.confidence: {self.confidence}")

        self.declare_parameter("model.iou_threshold", 0.45)
        self.iou_threshold = (
            self.get_parameter("model.iou_threshold").get_parameter_value().double_value
        )
        self.get_logger().info(f"model.iou_threshold: {self.iou_threshold}")

        self.declare_parameter(
            "model.weights_file",
            os.path.join(get_package_share_directory(PACKAGE_NAME), "yolov7-tiny.pt"),
        )
        self.model_weights_file = (
            self.get_parameter("model.weights_file").get_parameter_value().string_value
        )
        self.get_logger().info(f"model.weights_file: {self.model_weights_file}")

        if not os.path.isfile(self.model_weights_file):
            self.model_weights_file = os.path.join(
                get_package_share_directory(PACKAGE_NAME), self.model_weights_file
            )
            if not os.path.isfile(self.model_weights_file):
                raise Exception("model weights file not found")

        self.declare_parameter("model.device", "")
        self.device = self.get_parameter("model.device").get_parameter_value().string_value
        self.get_logger().info(f"model.device: {self.device}")

        self.declare_parameter("model.show_image", False)
        self.show_image = self.get_parameter("model.show_image").get_parameter_value().bool_value
        self.get_logger().info(f"model.show_image: {self.show_image}")

        self.declare_parameter("model.publish_debug_image", True)
        self.enable_publish_debug_image = (
            self.get_parameter("model.publish_debug_image").get_parameter_value().bool_value
        )
        self.get_logger().info(f"model.publish_debug_image: {self.enable_publish_debug_image}")

        self.declare_parameter("image_debug_publisher.qos_policy", "best_effort")
        self.qos_policy = (
            self.get_parameter("image_debug_publisher.qos_policy")
            .get_parameter_value()
            .string_value
        )

        self.declare_parameter("subscribers.qos_policy", "best_effort")
        self.subscribers_qos = (
            self.get_parameter("subscribers.qos_policy").get_parameter_value().string_value
        )

        self.debug_image_output_format = "mono8" # "mono8"  # "bgr8"

        self.processing_enabled = self.get_parameter_or("processing_enabled", True)

        self.service = self.create_service(
            std_srvs.srv.SetBool,
            "enable_processing",
            callback=self.set_processing_enabled_callback,
        )

        if self.subscribers_qos == "best_effort":
            self.get_logger().info("Using best effort qos policy for subscribers")
            self.qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
        else:
            self.get_logger().info("Using reliable qos policy for subscribers")
            self.qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )

        self.bridge = cv_bridge.CvBridge()

        # Get the list of camera topics from the config file
        self.declare_parameter("camera_topics", ["/cameras/frontleft_fisheye_image/image", "/cameras/frontright_fisheye_image/image", "/cameras/left_fisheye_image/image", "/cameras/right_fisheye_image/image"])
        self.camera_topics = self.get_parameter("camera_topics").get_parameter_value().string_array_value
        
        self.get_logger().info(f"Subscribed to topics: {self.camera_topics}")

        # Initialize subscribers and publishers for each camera topic
        self.subscribers = []
        self.detection_publishers = {}
        self.debug_image_publishers = {}

        for topic in self.camera_topics:
            # Create a subscriber for each camera topic
            self.subscribers.append(
                self.create_subscription(
                    Image,
                    topic,
                    callback=self.image_callback_factory(topic),
                    qos_profile=self.qos,
                )
            )

            # Create a detection publisher for each camera
            detection_topic = f"{topic}/detections"
            self.detection_publishers[topic] = self.create_publisher(
                Detection2DArray, detection_topic, self.qos
            )

            # Create a debug image publisher for each camera (if enabled)
            if self.enable_publish_debug_image:
                debug_image_topic = f"{topic}/debug_image"
                self.debug_image_publishers[topic] = self.create_publisher(
                    Image, debug_image_topic, self.qos
                )

        self.initialize_model()

    def initialize_model(self):
        with torch.no_grad():
            set_logging()
            self.device = select_device(self.device)
            self.half = self.device.type != "cpu"

            self.model = attempt_load(
                self.model_weights_file, map_location=self.device
            )
            self.stride = int(self.model.stride.max())

            self.imgsz = check_img_size(self.model_image_size, s=self.stride)

            if self.half:
                self.model.half()

            cudnn.benchmark = True

            if self.device.type != "cpu":
                self.model(
                    torch.zeros(1, 3, self.imgsz, self.imgsz)
                    .to(self.device)
                    .type_as(next(self.model.parameters()))
                )  # run once

            self.names = (
                self.model.module.names if hasattr(self.model, "module") else self.model.names
            )
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def set_processing_enabled_callback(self, request, response):
        self.processing_enabled = request.data
        response.success = True
        response.message = "OK"

        return response

    def accomodate_image_to_model(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return img

    def image_compressed_callback(self, msg):
        if not self.processing_enabled:
            return

        self.cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, self.debug_image_output_format)
        img = self.accomodate_image_to_model(self.cv_img)

        detections_msg, debugimg = self.predict(img, self.cv_img)

        self.detection_publisher.publish(detections_msg)

        if debugimg is not None:
            self.publish_debug_image(debugimg)

        if self.show_image:
            cv2.imshow("Compressed Image", debugimg)
            cv2.waitKey(1)

    def image_callback_factory(self, topic):
        def callback(msg):
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.image_queue[topic] = cv_img
            except CvBridgeError as e:
                self.get_logger().error(f"Error converting image from {topic}: {e}")

        return callback
    
    def image_callback_factory(self, topic):
        def callback(msg):
            if not self.processing_enabled:
                return

            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                img = self.accomodate_image_to_model(cv_img)

                detections_msg, debugimg = self.predict(img, cv_img)

                # Publish detections for the current camera
                self.detection_publishers[topic].publish(detections_msg)

                # Publish debug image for the current camera (if enabled)
                if self.enable_publish_debug_image and topic in self.debug_image_publishers:
                    self.publish_debug_image(debugimg, topic)

                if self.show_image:
                    cv2.imshow(f"Detection from {topic}", debugimg)
                    cv2.waitKey(1)
            except CvBridgeError as e:
                self.get_logger().error(f"Error converting image from {topic}: {e}")

        return callback

    def publish_debug_image(self, debugimg, topic):
        if self.debug_image_output_format == "mono8":
            debugimg = cv2.cvtColor(debugimg, cv2.COLOR_RGB2GRAY)
        elif self.debug_image_output_format == "rgb8":
            debugimg = cv2.cvtColor(debugimg, cv2.COLOR_BGR2RGB)
        elif self.debug_image_output_format == "rgba8":
            debugimg = cv2.cvtColor(debugimg, cv2.COLOR_BGR2RGBA)
        else:
            self.get_logger().error(
                f"Unsupported debug image output format: {self.debug_image_output_format}"
            )
            return

        # Publish the debug image for the current camera
        self.debug_image_publishers[topic].publish(
            self.bridge.cv2_to_imgmsg(debugimg, self.debug_image_output_format)
        )

    def predict(self, model_img, original_image):
        with torch.no_grad():
            model_img = torch.from_numpy(model_img).to(self.device)
            model_img = model_img.half() if self.half else model_img.float()  # uint8 to fp16/32
            model_img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if model_img.ndimension() == 3:
                model_img = model_img.unsqueeze(0)

            # Inference
            with torch.no_grad():
                pred = self.model(model_img, augment=False)[0]

            # NMS
            pred = non_max_suppression(pred, self.confidence, self.iou_threshold, agnostic=False)

            detections_msg = Detection2DArray()

            for i, det in enumerate(pred):
                gn = torch.tensor(original_image.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(
                        model_img.shape[2:], det[:, :4], original_image.shape
                    ).round()

                    for *xyxy, conf, cls in reversed(det):
                        if self.names[int(cls)] in self.selected_detections:
                            detection2D_msg = Detection2D()
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()

                            detection2D_msg.bbox.center.position.x = xywh[0]
                            detection2D_msg.bbox.center.position.y = xywh[1]
                            detection2D_msg.bbox.size_x = xywh[2]
                            detection2D_msg.bbox.size_y = xywh[3]

                            hypothesis = ObjectHypothesisWithPose()
                            hypothesis.hypothesis.score = float(conf.cpu().numpy())
                            classid = int(cls)
                            hypothesis.hypothesis.class_id = self.names[int(cls)]

                            detection2D_msg.results.append(hypothesis)
                            detections_msg.detections.append(detection2D_msg)

                            plot_one_box(
                                xyxy,
                                original_image,
                                label=self.names[classid],
                                line_thickness=2,
                                color=self.colors[classid],
                            )

            return detections_msg, original_image

def main(args=None):
    print(args)
    rclpy.init(args=sys.argv)

    minimal_publisher = ImageDetectObjectNode()
    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
