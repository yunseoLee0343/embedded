# Load datasets

import os

dataset_dir = '/content/drive/MyDrive/face features_88' # change to your dataset directory
image_paths = [os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir) if fname.endswith('.jpg')]

# Extract ROI of face segment by YOLO(face detection) and OpenCV(ROI)

import cv2
import torch
from ultralytics.yolov8 import YOLOv8
import pandas as pd

data = []

def extract_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    return None

def calculate_similarity(landmarks1, landmarks2):
    return distance.euclidean(np.array(landmarks1), np.array(landmarks2))


# ------
model = YOLOv8()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

for image_path in image_paths:
    landmarks = extract_landmarks(image_path)
    if landmarks:
        data.append({
            'image_path': image_path,
            'landmarks': landmarks
        })

for path in image_paths:
    image = cv2.imread(path) # load image
    results = model(image) # detect face using Yolo

    detections = results.pandas().xyxy[0]
    print(detections)

    for index, row in detections.iterrows():
        if row['name'] in ['face', 'eye', 'nose', 'mouth']:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            roi = image[y1:y2, x1:x2] # extract ROI of detected face using OpenCV
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # draw bounding box

        if row['name'] == 'face':
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            face_roi = image[y1:y2, x1:x2]
            face_landmarks = extract_landmarks(face_roi)
        elif row['name'] in ['eye', 'nose', 'mouth']:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            feature_roi = image[y1:y2, x1:x2]
            feature_landmarks = extract_landmarks(feature_roi)

    cv2.imshow("Detected Faces and Features", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# Extract landmark from specific face segment, by Mediapipe Face Mesh.

import mediapipe as mp

def extract_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    return None


# extract landmarks from ROI
for i in range(image_paths.len())
    for index, row in detections.iterrows():
        if row['name'] == 'face':
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            face_roi = image[y1:y2, x1:x2]
            face_landmarks = extract_landmarks(face_roi)
        elif row['name'] in ['eye', 'nose', 'mouth']:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            feature_roi = image[y1:y2, x1:x2]
            feature_landmarks = extract_landmarks(feature_roi)

# Normalizae landmarks.

import numpy as np
from scipy.spatial import distance

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    center = np.mean(landmarks, axis=0)
    landmarks -= center
    return landmarks

def calculate_similarity(landmarks1, landmarks2):
    return distance.euclidean(np.array(landmarks1), np.array(landmarks2))

def makePairs():
    i=0, j=1
    for i in range(image_paths.len())
        for j in range(image_paths.len() - 1)
            image_pairs[i][0] = image_paths[i]


image_pairs = [][2]

# 이미지 쌍에 대한 유사도 점수 계산
for img1_path, img2_path in pairs:
    landmarks1 = extract_landmarks(img1_path)
    landmarks2 = extract_landmarks(img2_path)
    if landmarks1 and landmarks2:
        left_eye_sim = calculate_similarity(select_landmarks(landmarks1, LEFT_EYE_INDEXES), select_landmarks(landmarks2, LEFT_EYE_INDEXES))
        right_eye_sim = calculate_similarity(select_landmarks(landmarks1, RIGHT_EYE_INDEXES), select_landmarks(landmarks2, RIGHT_EYE_INDEXES))
        nose_sim = calculate_similarity(select_landmarks(landmarks1, NOSE_INDEXES), select_landmarks(landmarks2, NOSE_INDEXES))
        mouth_sim = calculate_similarity(select_landmarks(landmarks1, MOUTH_INDEXES), select_landmarks(landmarks2, MOUTH_INDEXES))
        jaw_sim = calculate_similarity(select_landmarks(landmarks1, JAW_INDEXES), select_landmarks(landmarks2, JAW_INDEXES))
        eye_nose_sim = calculate_eye_nose_distance(landmarks1, landmarks2)
        nose_mouth_sim = calculate_nose_mouth_distance(landmarks1, landmarks2)

        data.append({
            'left_eye_sim': left_eye_sim,
            'right_eye_sim': right_eye_sim,
            'nose_sim': nose_sim,
            'mouth_sim': mouth_sim,
            'jaw_sim': jaw_sim,
            'eye_nose_sim': eye_nose_sim,
            'nose_mouth_sim': nose_mouth_sim,
            'label': label  # 유사도의 레이블 (예: 0: 비슷하지 않음, 1: 비슷함)
        })

df = pd.DataFrame(data)

# 데이터셋 저장
df.to_csv('similarity_dataset.csv', index=False)