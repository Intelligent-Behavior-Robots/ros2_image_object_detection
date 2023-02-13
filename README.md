# ros2_image_object_detection
This is a ROS2 package that implements an object detection node. It takes in input images and uses a default YOLOv7-tiny model and PyTorch technology to perform object detection. The package is designed to be flexible and allow for customization by providing various parameters and topics for input and output.

<p align="center">
<img src="https://user-images.githubusercontent.com/13334595/218350166-af3e5fab-1844-4aa4-92ba-a8abe5bfaaff.png" width="500" style="align-text: center">
</p>

## Usage

```
ros2 launch image_object_detection image_object_detection_launch.py
```

## Topics

### Subscriptions
* *image*: A topic that publishes raw images. Subscribed to using the Image message type.
* *image/compressed*: A topic that publishes compressed images. Subscribed to using the CompressedImage message type.

### Publishers
* *detections*: A topic that publishes object detections. Published using the Detection2DArray message type.
* *debug_image*: A topic that publishes a debug image with detections drawn on it. Published using the Image message type if publish_debug_image is set to True.

## Parameters

* *model.image_size*: The size of the input image to the model. Default value is 640.
* *model.confidence*: The minimum confidence threshold for object detection. Default value is 0.25.
* *model.iou_threshold*: The IoU (Intersection over Union) threshold for overlapping detections. Default value is 0.45.
* *model.weights_file*: The path to the model weights file. Default value is yolov7-tiny.pt in the package share directory.
* *model.device*: The device to run the model on, e.g. CPU or GPU. Default value is an empty string.
* *show_image*: A flag to indicate whether to display the input image. Default value is False.
* *publish_debug_image*: A flag to indicate whether to publish a debug image with detections drawn on it. Default value is True.

## About
* Author: Pablo Iñigo Blasco (pablo@ibrobotics.com) - Intelligent Behavior Robots
* License: BSDv3

