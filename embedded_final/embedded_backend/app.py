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

# 설정된 경로
dataset_dir = 'face_features_88'
image_paths = [os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir) if fname.endswith('.jpg')]

# 모델 로드
model = YOLO("yolov8s.pt")
embedder = FaceNet()

# 임시 이미지 경로
temp_paths = image_paths[:2]


# 얼굴 탐지 함수
def detect_faces_yolo(model, image):
    results = model(image)
    if isinstance(results, list):
        detections = results[0].boxes.xyxy
    else:
        detections = results.boxes.xyxy
    return detections


# 이미지 전처리 함수
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

    # 얼굴 탐지
    detections = detect_faces_yolo(model, image)
    detections = detections.numpy().tolist()

    return jsonify({"detections": detections})


@app.route('/process_images', methods=['GET'])
def process_images():
    embeddings = []
    detected_faces_info = []

    for path in temp_paths:
        image = cv2.imread(path)  # 이미지 로드
        detections = detect_faces_yolo(model, image)  # Yolo를 사용하여 얼굴 감지
        detections = detections.numpy().tolist()

        # 바운딩 박스 그리기 및 임베딩 추출
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection[:4])  # 좌표를 정수로 변환
            face = image[y1:y2, x1:x2]  # 얼굴 영역 추출

            # FaceNet 모델을 사용하여 얼굴 임베딩 추출
            embedding = embedder.extract(face, threshold=0.95)  # 얼굴 영역만을 입력으로 사용
            embeddings.append(embedding[0]['embedding'])  # 임베딩 벡터 저장
            detected_faces_info.append({
                'detection': detection,
                'embedding': embedding[0]['embedding'].tolist()
            })

            # 이미지에 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 빨간색 박스

        # 감지된 바운딩 박스 출력
        print(detections)

    # 두 이미지의 임베딩 벡터 출력 및 유사도 계산
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


# 유사도 계산 함수
def calculate_similarity(embedding1, embedding2, metric='euclidean'):
    if metric == 'euclidean':
        return euclidean(embedding1, embedding2)
    elif metric == 'cosine':
        return cosine(embedding1, embedding2)
    else:
        raise ValueError("지원하지 않는 유사도 측정 방식입니다. 'euclidean' 또는 'cosine'을 사용하세요.")


if __name__ == '__main__':
    app.run(debug=True)
