from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Color map for drawing boxes
color_map = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

# Object detection function
def detect_objects(image, min_size=0.1):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416),
                                 mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detected_objects = []

    for i, out in enumerate(outs):
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                if w * h / (width * height) > min_size:
                    x = center_x - w // 2
                    y = center_y - h // 2
                    object_name = classes[class_id]
                    color = color_map[i % len(color_map)]

                    detected_objects.append({
                        "name": object_name,
                        "bounding_box": {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h
                        },
                        "confidence": float(confidence),
                        "color": color
                    })

    return detected_objects

# GET route (for browser access)
@app.route("/", methods=["GET"])
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Object Detection</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f5f5f5;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background-color: white;
                padding: 30px 40px;
                border-radius: 12px;
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
                text-align: center;
            }
            h2 {
                margin-bottom: 25px;
                color: #333;
            }
            input[type="file"] {
                margin-bottom: 15px;
            }
            input[type="submit"] {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 16px;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Upload Image for Object Detection</h2>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required><br>
                <input type="submit" value="Upload">
            </form>
        </div>
    </body>
    </html>
    '''

# POST route (for image detection)
@app.route("/", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image format"}), 400

    objects = detect_objects(image)

    return jsonify({"objects": objects})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

