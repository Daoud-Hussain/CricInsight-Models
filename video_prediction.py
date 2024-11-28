import argparse
import boto3
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
import joblib  # To load the scaler and label encoder
import pandas as pd  # To save results to CSV
import os
import datetime  # To track timestamps for appending test cases
from tensorflow.keras.models import load_model
from utils import compute_features  # Import feature extraction from utils

# S3 Video Download Function
def download_video(s3_url, local_path):
    s3 = boto3.client('s3')
    bucket_name, key = s3_url.replace("s3://", "").split("/", 1)
    s3.download_file(bucket_name, key, local_path)

# Load the trained LSTM model
model = load_model('models/advanced_lstm_model.h5')

# Load the fitted scaler
scaler = joblib.load('models/scaler.pkl')

# Load the label encoder
label_encoder = joblib.load('models/label_encoder.pkl')

# Initialize MediaPipe for pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Rolling window for smoothing predictions
prediction_window = deque(maxlen=5)

# Argument Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing Script")
    parser.add_argument("--video_url", required=True, help="S3 URL of the video to process")
    args = parser.parse_args()

    # Local path to save the video
    local_video_path = "/tmp/video.mp4"

    # Download video from S3
    print(f"Downloading video from {args.video_url}...")
    download_video(args.video_url, local_video_path)
    print(f"Video downloaded to {local_video_path}.")

    # Track shot count for real-time detection (adjusted for new shot types)
    shot_count = {'cut': 0, 'pull': 0, 'cover_drive': 0, 'straight_drive': 0, 'flick': 0}

    # Open the downloaded video
    cap = cv2.VideoCapture(local_video_path)

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        # Draw pose landmarks on the frame
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract features from the video frame
            features = extract_video_features(result.pose_landmarks.landmark)

            # Scale the features using the fitted scaler
            features_scaled = scaler.transform(features)
            features_scaled = features_scaled.reshape(1, 1, features_scaled.shape[1])

            # Predict the shot type based on pose landmarks
            prediction = model.predict(features_scaled)
            predicted_class = np.argmax(prediction)

            # Smooth the prediction using a rolling window
            prediction_window.append(predicted_class)
            smoothed_prediction = max(set(prediction_window), key=prediction_window.count)

            # Convert prediction index to label
            shot_label = label_encoder.inverse_transform([smoothed_prediction])[0]
            shot_count[shot_label] += 1

            # Display the predicted shot label in large red font on the frame
            cv2.putText(frame, f"Shot: {shot_label.upper()}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Video Shot Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate total shots
    total_shots = sum(shot_count.values())

    # Calculate percentages of each shot
    shot_percentage = {shot: (count / total_shots) * 100 if total_shots > 0 else 0 for shot, count in shot_count.items()}

    # Prepare data for CSV (including timestamp to append new test cases)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shot_data = {
        'Timestamp': [timestamp] * len(shot_count),  # Same timestamp for all rows
        'Shot Type': list(shot_count.keys()),
        'Shot Count': list(shot_count.values()),
        'Percentage': [round(shot_percentage[shot], 2) for shot in shot_count]
    }

    # Save shot count and percentages to CSV (append if file exists)
    output_path = 'output/shot_analysis.csv'
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Append new data to the file if it exists, else create it
    if os.path.exists(output_path):
        pd.DataFrame(shot_data).to_csv(output_path, mode='a', header=False, index=False)
    else:
        pd.DataFrame(shot_data).to_csv(output_path, index=False)

    print(f"Shot analysis saved to {output_path}")

    # Display shot counts and percentages
    print(f"Shot Counts: {shot_count}")
    print(f"Shot Percentages: {shot_percentage}")

