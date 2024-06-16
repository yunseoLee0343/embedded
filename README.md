
# Face Recognition Flask & Flutter App

---

## Project Summary
This project is a face recognition system with a Flask backend and a Flutter frontend. The Flask server uses YOLO for face detection and FaceNet for face embedding extraction. The Flutter app allows users to select an image from their gallery and upload it to the server. The server processes the image, detects faces, extracts embeddings, and calculates similarity. The similarity results are then displayed in the Flutter app.

### Similarity Extraction Logic
1. **Face Detection**: The Flask backend uses the YOLO model to detect faces in the uploaded image.
2. **Face Embedding Extraction**: Detected face regions are extracted and passed to the FaceNet model to generate embedding vectors.
3. **Similarity Calculation**: The similarity between embeddings is calculated using two metrics:
   - Euclidean Distance: Measures the straight-line distance between two embedding vectors.
   - Cosine Similarity: Measures the cosine of the angle between two embedding vectors.
4. **Results Handling**: The similarity results, including Euclidean distance and cosine similarity, are sent back to the Flutter app and displayed to the user.

## Requirements

### Backend (Flask)
- Python 3.6+
- Flask
- OpenCV
- TensorFlow
- Pillow
- Keras-FaceNet
- Ultralytics YOLO
- SciPy

### Frontend (Flutter)
- Flutter SDK
- Dart

## Installation

### Backend

1. Clone the repository:

    ```bash
    git clone https://github.com/yunseoLee0343/embedded.git
    cd embedded_final/embedded_backend
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Place your dataset of images in the `face_features_88` directory. Ensure that the images are in JPG format.

4. Start the Flask app:

    ```bash
    python app.py
    ```

5. The Flask app will be running at `http://127.0.0.1:5000`.

### Frontend

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/face-recognition-flask.git
    cd face-recognition-flutter/frontend
    ```

2. Ensure you have Flutter installed. If not, follow the instructions on the [Flutter official website](https://flutter.dev/docs/get-started/install).

3. Open the project in your preferred IDE (e.g., VSCode, Android Studio).

4. Run the Flutter app on an emulator or physical device:

    ```bash
    flutter run
    ```

## Usage

### Flask API Endpoints

#### 1. Detect Faces

**Endpoint:** `/detect_faces`

**Method:** `POST`

**Description:** Detects faces in an uploaded image.

**Request:**

- `image`: The image file to be uploaded.

**Response:**

    ```json
    {
      "detections": [[x1, y1, x2, y2], ...]
    }
    ```

#### 2. Process Images

**Endpoint:** `/process_images`

**Method:** `GET`

**Description:** Processes images in the dataset directory, detects faces, extracts embeddings, and calculates the similarity between the first two images.

**Response:**

    ```json
    {
      "Embeddings for the first image": [embedding vector],
      "Embeddings for the second image": [embedding vector],
      "Euclidean Distance": euclidean_distance_value,
      "Cosine Similarity": cosine_similarity_value
    }
    ```

#### 3. Show Images

**Endpoint:** `/show_images`

**Method:** `GET`

**Description:** Returns the processed images with detected faces and bounding boxes.

**Response:**

    ```json
    {
      "images": [image_bytes, ...]
    }
    ```

### Flutter App

1. The Flutter app allows users to pick an image from their gallery and upload it to the Flask backend for face detection and similarity calculation.

2. The similarity result is displayed in the Flutter app once the image is processed.


## Running the App

1. Ensure you have placed images in the `face_features_88` directory.
2. Start the Flask application using:

    ```bash
    python app.py
    ```

3. Run the Flutter application using:

    ```bash
    flutter run
    ```

4. Use the Flutter app to upload images and get similarity results from the Flask backend.