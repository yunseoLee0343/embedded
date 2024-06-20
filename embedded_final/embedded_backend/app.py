import os
from flask import Flask, request, jsonify, send_file
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import base64

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


@app.route('/process_images', methods=['POST'])
def process_images():
    embeddings = []
    detected_faces_info = []

    # Get JSON data from request body
    data = request.get_json()

    if 'image1' not in data or 'image2' not in data:
        return jsonify({"error": "Please provide both image1 and image2 as Base64 encoded strings."}), 400

    # Decode Base64 images
    image1_base64 = data['image1']
    image2_base64 = data['image2']

    image1 = base64.b64decode(image1_base64)
    image2 = base64.b64decode(image2_base64)

    # Preprocess and detect faces for image1
    image1 = preprocess_image(image1)
    detections1 = detect_faces_yolo(model, image1)
    detections1 = detections1.numpy().tolist()

    for detection in detections1:
        x1, y1, x2, y2 = map(int, detection[:4])
        face = image1[y1:y2, x1:x2]

        embedding = embedder.extract(face, threshold=0.95)
        embeddings.append(embedding[0]['embedding'])
        detected_faces_info.append({
            'detection': detection,
            'embedding': embedding[0]['embedding'].tolist()
        })

        cv2.rectangle(image1, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Preprocess and detect faces for image2
    image2 = preprocess_image(image2)
    detections2 = detect_faces_yolo(model, image2)
    detections2 = detections2.numpy().tolist()

    for detection in detections2:
        x1, y1, x2, y2 = map(int, detection[:4])
        face = image2[y1:y2, x1:x2]

        embedding = embedder.extract(face, threshold=0.95)
        embeddings.append(embedding[0]['embedding'])
        detected_faces_info.append({
            'detection': detection,
            'embedding': embedding[0]['embedding'].tolist()
        })

        cv2.rectangle(image2, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Output detected bounding boxes for image1 and image2 (you can remove print(detections1) and print(detections2))

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

    print(similarity_response)

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
