import os
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import numpy as np
import tensorflow as tf
from PIL import Image
import io

import cv2
from keras_facenet import FaceNet
from scipy.spatial.distance import euclidean, cosine

app = Flask(__name__)

# Set dataset directory
dataset_dir = 'face_features_88'
image_paths = [os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir) if fname.endswith('.jpg')]

# Load models
model = YOLO("yolov8s.pt")
embedder = FaceNet()

# Temporary image paths
temp_paths = image_paths[:2]


# Face detection function using YOLO
def detect_faces_yolo(model, image):
    results = model(image)
    if isinstance(results, list):
        detections = results[0].boxes.xyxy
    else:
        detections = results.boxes.xyxy
    return detections


# Image preprocessing function
def preprocess_image(image):
    image = Image.open(io.BytesIO(image))
    image = image.convert('RGB')
    image = np.array(image)
    return image


@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image'].read()
    image = preprocess_image(image)

    # Face detection
    detections = detect_faces_yolo(model, image)
    detections = detections.numpy().tolist()

    return jsonify({"detections": detections})


@app.route('/process_images', methods=['GET'])
def process_images():
    embeddings = []
    detected_faces_info = []

    for path in temp_paths:
        image = cv2.imread(path)  # Load image
        detections = detect_faces_yolo(model, image)  # Detect faces using YOLO
        detections = detections.numpy().tolist()

        # Draw bounding boxes and extract embeddings
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])  # Convert coordinates to integers
            face = image[y1:y2, x1:x2]  # Extract face region

            # Extract face embeddings using FaceNet
            embedding = embedder.extract(face, threshold=0.95)  # Use only the face region as input
            embeddings.append(embedding[0]['embedding'])  # Store embedding vector
            detected_faces_info.append({
                'detection': detection,
                'embedding': embedding[0]['embedding'].tolist()
            })

            # Draw bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red box

        # Output detected bounding boxes
        print(detections)

    # Output embedding vectors and calculate similarity between two images
    if len(embeddings) >= 2:
        embedding1, embedding2 = embeddings[:2]
        euclidean_distance = calculate_similarity(embedding1, embedding2, metric='euclidean')
        cosine_similarity = calculate_similarity(embedding1, embedding2, metric='cosine')

        similarity_response = {
            "Embeddings for the first image": embedding1.tolist(),
            "Embeddings for the second image": embedding2.tolist(),
            "Euclidean Distance": euclidean_distance,
            "Cosine Similarity": cosine_similarity
        }
    else:
        similarity_response = {"error": "Not enough embeddings found."}

    return jsonify(similarity_response)


@app.route('/show_images', methods=['GET'])
def show_images():
    images = []
    for path in temp_paths:
        img = cv2.imread(path)
        _, img_encoded = cv2.imencode('.jpg', img)
        images.append(img_encoded.tobytes())

    return jsonify({"images": images})


# Similarity calculation function
def calculate_similarity(embedding1, embedding2, metric='euclidean'):
    if metric == 'euclidean':
        return euclidean(embedding1, embedding2)
    elif metric == 'cosine':
        return cosine(embedding1, embedding2)
    else:
        raise ValueError("Unsupported similarity metric. Use 'euclidean' or 'cosine'.")


if __name__ == '__main__':
    app.run(debug=True)
