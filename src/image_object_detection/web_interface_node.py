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
        
        # Start Flask in a separate thread
        self.flask_thread = threading.Thread(target=self.run_flask)
        self.flask_thread.daemon = True
        self.flask_thread.start()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            available_classes = self.detection_node.names
            selected_classes = self.detection_node.selected_detections
            current_confidence = self.detection_node.confidence
            current_iou = self.detection_node.iou_threshold
            
            return render_template('index.html',
                                 available_classes=available_classes,
                                 selected_classes=selected_classes,
                                 confidence=current_confidence,
                                 iou_threshold=current_iou)

        @self.app.route('/update_params', methods=['POST'])
        def update_params():
            if request.method == 'POST':
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

                return jsonify({'status': 'success'})

    def run_flask(self):
        self.app.run(host='0.0.0.0', port=5005)
