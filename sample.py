data = []

def extract_landmarks(image):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Face Mesh
    result = face_mesh.process(rgb_image)

    # Extract landmarks if faces are detected
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

    return None

def calculate_similarity(landmarks1, landmarks2):
    return distance.euclidean(np.array(landmarks1), np.array(landmarks2))

# ------
# model = YOLOv8()
# model = YOLO("yolov8n.pt")
#model = YOLO("yolov8m.pt")
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

    # Check if results is a list, if so, convert it to a Results object
    #if isinstance(results, list):
    #  results = Results(results, None)

   # detections = results.pd().xyxy[0]
   # detections = results[0].xyxy
    if isinstance(results, list):
      detections = results[0].boxes.xyxy
    else:
      detections = results.boxes.xyxy

    print(detections)

    # for index, row in detections.iterrows():
    #     if row['name'] in ['face', 'eye', 'nose', 'mouth']:
    #         x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    #         roi = image[y1:y2, x1:x2] # extract ROI of detected face using OpenCV
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # draw bounding box

    #     if row['name'] == 'face':
    #         x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    #         face_roi = image[y1:y2, x1:x2]
    #         face_landmarks = extract_landmarks(face_roi)
    #     elif row['name'] in ['eye', 'nose', 'mouth']:
    #         x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    #         feature_roi = image[y1:y2, x1:x2]
    #         feature_landmarks = extract_landmarks(feature_roi)
    for index in range(detections.shape[0]):
      row = detections[index]
      class_index = int(row[-1])  # Access the correct class index from the tensor
      if class_index < len(model.names) and model.names[class_index] in ['face', 'eye', 'nose', 'mouth']: # Check if class index is valid
          x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
          roi = image[y1:y2, x1:x2]  # extract ROI of detected face using OpenCV
          cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # draw bounding box

      if class_index < len(model.names) and model.names[class_index] == 'face': # Check if class index is valid
          x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
          face_roi = image[y1:y2, x1:x2]
          face_landmarks = extract_landmarks(face_roi)
      elif class_index < len(model.names) and model.names[class_index] in ['eye', 'nose', 'mouth']: # Check if class index is valid
          x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
          feature_roi = image[y1:y2, x1:x2]
          feature_landmarks = extract_landmarks(feature_roi)

    #################

    from google.colab.patches import cv2_imshow

    cv2_imshow(image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

def extract_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    return None


# extract landmarks from ROI
for i in range(image_paths.len()):
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
    i=0
    j=1
    for i in range(image_paths.len()):
        for j in range(image_paths.len() - 1):
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