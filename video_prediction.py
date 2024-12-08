import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp
import joblib
import pandas as pd
import os
import datetime
from tensorflow.keras.models import load_model
from utils import compute_features

# Ensure OpenCV GUI dependencies are installed
import sys
print("Python version:", sys.version)
print("OpenCV version:", cv2.__version__)

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

# Path to the video
video_path = r"/home/octaloop/Model/practice-video.mp4"

# Track shot count for real-time detection
shot_count = {'cut': 0, 'pull': 0, 'cover_drive': 0, 'straight_drive': 0, 'flick': 0}

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

def main():
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame.")
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
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Video Shot Detection', frame)

        # Wait for 1ms and check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Calculate total shots
    total_shots = sum(shot_count.values())

    # Calculate percentages of each shot
    shot_percentage = {shot: (count / total_shots) * 100 if total_shots > 0 else 0 
                       for shot, count in shot_count.items()}


# shot_percentage = {shot: (count / total_shots) * 100 if total_shots > 0 else 0 for shot, count in shot_count.items()}

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
        pd.DataFrame(shot_data).to_csv(output_path, mode='w',  index=False)
    else:
        pd.DataFrame(shot_data).to_csv(output_path, index=False)

    print(f"Shot analysis saved to {output_path}")
        # Print results
    print("Shot Counts:", shot_count)
    print("Shot Percentages:", shot_percentage)

if __name__ == '__main__':
    main()

