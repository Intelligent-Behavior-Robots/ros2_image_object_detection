
# Copyright 2023 Intelligent Behavior Robots, Inc.
#
# Licensed under the BSD 3-Clause License 
#
# Author: Pablo Iñigo Blasco

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import CompressedImage, Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

import os
import cv2
import cv_bridge

import numpy as np
from numpy import random

import torch
import torch.backends.cudnn as cudnn
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from vision_msgs.msg import Detection2DArray, Detection2D
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = 'image_object_detection'

class ImageDetectObjectNode(Node):
    def __init__(self):
        super().__init__(PACKAGE_NAME)

        self.model_image_size = self.get_parameter_or('model.image_size', 640)
        self.confidence = self.get_parameter_or('model.confidence', 0.25)
        self.iou_threshold = self.get_parameter_or('model.iou_threshold', 0.45)
        self.model_weights_file = self.get_parameter_or('model.weights_file', os.path.join(get_package_share_directory(PACKAGE_NAME), "yolov7-tiny.pt"))
        self.device = self.get_parameter_or('model.device', '')
        self.show_image = self.get_parameter_or('show_image', False)
        self.publish_debug_image = self.get_parameter_or('publish_debug_image', True)

        self.bridge = cv_bridge.CvBridge()

        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_sub = self.create_subscription(
            msg_type=Image,
            topic='image',
            callback=self.image_callback,
            qos_profile=self.qos
        )

        self.image_compressed_sub = self.create_subscription(
            msg_type=CompressedImage,
            topic='image/compressed',
            callback=self.image_compressed_callback,
            qos_profile=self.qos
        )

        self.detection_publisher = self.create_publisher(
            msg_type=Detection2DArray,
            topic='detections',
            qos_profile=self.qos
        )

        if self.publish_debug_image:
            self.debug_image_publisher = self.create_publisher(
                msg_type=Image,
                topic='debug_image',
                qos_profile=self.qos
            )

        self.initialize_model()

    def initialize_model(self):
        with torch.no_grad():
            # Initialize
            set_logging()
            self.device = select_device(self.device)
            self.half = self.device.type != 'cpu'  

            # Load model
            self.model = attempt_load(
                self.model_weights_file, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max()) 
             
            self.imgsz = check_img_size(
                self.model_image_size, s=self.stride)  

            if self.half:
                self.model.half()  # to FP16

            cudnn.benchmark = True

            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(
                    self.device).type_as(next(self.model.parameters())))  # run once

            self.names = self.model.module.names if hasattr(
                self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255)
                            for _ in range(3)] for _ in self.names]

    def accomodate_image_to_model(self, img0):
        img = letterbox(img0,  self.imgsz, stride=self.stride)[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        return img

    def image_compressed_callback(self, msg):
        self.cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        img = self.accomodate_image_to_model(self.cv_img)

        detections_msg, debugimg = self.predict(img, self.cv_img)

        self.detection_publisher.publish(detections_msg)

        if debugimg is not None:
            self.debug_image_publisher.publish(
                self.bridge.cv2_to_compressed_imgmsg(debugimg, "jpg"))

        if self.show_image:
            cv2.imshow('Compressed Image', debugimg)
            cv2.waitKey(1)

    def image_callback(self, msg):
        self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img = self.accomodate_image_to_model(self.cv_img)

        detections_msg, debugimg = self.predict(img, self.cv_img)

        self.detection_publisher.publish(detections_msg)

        if debugimg is not None:
            print("Publishing debug image")
            self.debug_image_publisher.publish(
                self.bridge.cv2_to_imgmsg(debugimg, "bgr8"))

        if self.show_image:
            cv2.imshow('Detection', debugimg)
            cv2.waitKey(1)

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
            pred = non_max_suppression(
                pred, self.confidence, self.iou_threshold, agnostic=False)

            detections_msg = Detection2DArray()

            for i, det in enumerate(pred):
                gn = torch.tensor(original_image.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(
                        model_img.shape[2:], det[:, :4], original_image.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        detection2D_msg = Detection2D()
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 

                        detection2D_msg.bbox.center.x = xywh[0]
                        detection2D_msg.bbox.center.y = xywh[1]
                        detection2D_msg.bbox.size_x = xywh[2]
                        detection2D_msg.bbox.size_y = xywh[3]

                        hypothesis = ObjectHypothesisWithPose()
                        hypothesis.hypothesis.score = float(conf.cpu().numpy())
                        classid = int(cls)
                        hypothesis.hypothesis.class_id = self.names[int(cls)]

                        detection2D_msg.results.append(hypothesis)
                        detections_msg.detections.append(detection2D_msg)

                        plot_one_box(
                            xyxy, original_image, label=self.names[classid], line_thickness=2, color=self.colors[classid])

            return detections_msg, original_image


def main(args=None):
    rclpy.init()

    minimal_publisher = ImageDetectObjectNode()
    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()