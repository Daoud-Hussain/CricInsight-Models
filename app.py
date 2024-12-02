from flask import Flask, request, jsonify
import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
import joblib
import pandas as pd
import datetime

# Ensure the current directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the utility functions
from utils import compute_features

app = Flask(__name__)

# Load models and configurations
MODEL_PATH = 'models/advanced_lstm_model.h5'
SCALER_PATH = 'models/scaler.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'

# Load the trained LSTM model
model = tf.keras.models.load_model(MODEL_PATH)

# Load the fitted scaler
scaler = joblib.load(SCALER_PATH)

# Load the label encoder
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_video_features(pose_landmarks):
    """
    Extract features from the pose landmarks.
    """
    keypoints = {
        'nose': (pose_landmarks[mp_pose.PoseLandmark.NOSE].x, pose_landmarks[mp_pose.PoseLandmark.NOSE].y),
        'left_wrist': (pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y),
        'right_wrist': (pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y),
        'left_elbow': (pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y),
        'right_elbow': (pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y),
        'left_shoulder': (pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y),
        'right_shoulder': (pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y),
        'left_hip': (pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP].y),
        'right_hip': (pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y),
        'left_knee': (pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y),
        'right_knee': (pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y),
        'left_ankle': (pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y),
        'right_ankle': (pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y)
    }
    features = np.array([compute_features(keypoints)])
    return features

@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        
        video_file = request.files['video']
        
        # Save the uploaded video
        video_path = '/tmp/uploaded_video.mp4'
        video_file.save(video_path)
        
        # Initialize shot tracking
        shot_count = {'cut': 0, 'pull': 0, 'cover_drive': 0, 'straight_drive': 0, 'flick': 0}
        prediction_window = deque(maxlen=5)

        # Open the video
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            # Process pose landmarks if detected
            if result.pose_landmarks:
                # Extract features from the video frame
                features = extract_video_features(result.pose_landmarks.landmark)

                # Scale the features using the fitted scaler
                features_scaled = scaler.transform(features)
                features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

                # Predict the shot type based on pose landmarks
                prediction = model.predict(features_scaled)
                predicted_class = np.argmax(prediction)

                # Smooth the prediction
                prediction_window.append(predicted_class)
                smoothed_prediction = max(set(prediction_window), key=prediction_window.count)

                # Convert prediction index to label
                shot_label = label_encoder.inverse_transform([smoothed_prediction])[0]
                shot_count[shot_label] += 1

        # Calculate total shots
        total_shots = sum(shot_count.values())

        # Calculate percentages of each shot
        shot_percentage = {shot: (count / total_shots) * 100 if total_shots > 0 else 0 
                           for shot, count in shot_count.items()}

        # Prepare results
        results = {
            'shot_counts': shot_count,
            'shot_percentages': shot_percentage,
            'total_shots': total_shots
        }

        # Clean up
        cap.release()
        os.remove(video_path)
        
        return jsonify(results), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)