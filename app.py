from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load the SHMHEHAI model
MODEL_PATH = "shmhehai_model.h5"  # Update if your model file has a different name
model = load_model(MODEL_PATH)

# Preprocessing function (for digits or short text)
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))  # adjust size to model input
    image = image / 255.0  # normalize
    image = np.expand_dims(image, axis=(0, -1))  # shape: (1, 28, 28, 1)
    return image

@app.route('/')
def index():
    return "SHMHEHAI API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("Available keys in request.files:", request.files.keys())
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        print("Image decoding failed!")  # 👈 debug line
        return jsonify({"error": "Invalid image format"}), 400

    try:
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        print("Prediction error:", str(e))  # 👈 catch any crash
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

