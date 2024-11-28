import rclpy
from rclpy.node import Node
from flask import Flask, render_template, request, jsonify
import threading
from rclpy.parameter import Parameter

class WebInterfaceNode(Node):
    def __init__(self, detection_node):
        super().__init__('web_interface_node')
        self.detection_node = detection_node
        self.app = Flask(__name__)
        self.setup_routes()
        
        self.flask_thread = threading.Thread(target=self.run_flask)
        self.flask_thread.daemon = True
        self.flask_thread.start()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html',
                                 available_classes=self.detection_node.names,
                                 selected_classes=self.detection_node.selected_detections,
                                 confidence=self.detection_node.confidence,
                                 iou_threshold=self.detection_node.iou_threshold,
                                 camera_topics=self.detection_node.camera_topics,
                                 publish_debug_image=self.detection_node.enable_publish_debug_image,
                                 image_size=self.detection_node.model_image_size)

        @self.app.route('/update_params', methods=['POST'])
        def update_params():
            if request.method == 'POST':
                # Handle camera topics
                camera_topics = request.form.get('camera_topics', '').split('\n')
                camera_topics = [topic.strip() for topic in camera_topics if topic.strip()]
                self.detection_node.set_parameters([
                    Parameter('camera_topics', value=camera_topics)
                ])

                # Handle selected detections
                selected_classes = request.form.getlist('classes[]')
                self.detection_node.set_parameters([
                    Parameter('selected_detections', value=selected_classes)
                ])

                # Handle confidence threshold
                confidence = float(request.form.get('confidence', 0.25))
                self.detection_node.set_parameters([
                    Parameter('model.confidence', value=confidence)
                ])

                # Handle IoU threshold
                iou_threshold = float(request.form.get('iou_threshold', 0.45))
                self.detection_node.set_parameters([
                    Parameter('model.iou_threshold', value=iou_threshold)
                ])

                # Add to existing parameter updates
                publish_debug = request.form.get('publish_debug_image') == 'true'
                self.detection_node.set_parameters([
                    Parameter('model.publish_debug_image', value=publish_debug)
                ])

                image_size = int(request.form.get('image_size', 640))
                self.detection_node.set_parameters([
                    Parameter('model.image_size', value=image_size)
                ])

                return jsonify({'status': 'success'})
    def run_flask(self):
        self.app.run(host='0.0.0.0', port=5005)
