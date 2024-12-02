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

# Disable GPU logging and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Explicitly disable GPU if not needed
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

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

# Path to the video (update the path or use a live feed)
video_path = r"/home/ubuntu/CricInsight-Models/video.mp4"

def process_video(video_path):
    # Track shot count for real-time detection (adjusted for new shot types)
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

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    # Prepare to save processed frames (optional)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output/processed_video.mp4', fourcc, 20.0, (640, 480))

    frames_processed = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to a standard size
        frame = cv2.resize(frame, (640, 480))

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

        # Write the frame to output video
        out.write(frame)
        frames_processed += 1

    # Release resources
    cap.release()
    out.release()

    # Calculate total shots
    total_shots = sum(shot_count.values())

    # Calculate percentages of each shot
    shot_percentage = {shot: (count / total_shots) * 100 if total_shots > 0 else 0 
                       for shot, count in shot_count.items()}

    # Prepare data for CSV (including timestamp to append new test cases)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shot_data = {
        'Timestamp': [timestamp] * len(shot_count),
        'Shot Type': list(shot_count.keys()),
        'Shot Count': list(shot_count.values()),
        'Percentage': [round(shot_percentage[shot], 2) for shot in shot_count]
    }

    # Ensure output directory exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Save shot count and percentages to CSV
    output_path = os.path.join(output_dir, 'shot_analysis.csv')
    pd.DataFrame(shot_data).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)

    # Prepare return dictionary
    return {
        'shot_counts': shot_count,
        'shot_percentages': shot_percentage,
        'frames_processed': frames_processed,
        'output_video': 'output/processed_video.mp4',
        'output_csv': output_path
    }

# If script is run directly
if __name__ == '__main__':
    # Process the video
    results = process_video(video_path)
    
    # Print results
    print("Shot Counts:", results['shot_counts'])
    print("Shot Percentages:", results['shot_percentages'])
    print("Frames Processed:", results['frames_processed'])
