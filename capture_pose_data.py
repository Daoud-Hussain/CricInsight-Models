import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directory to store the collected data
data_dir = 'data/collected_data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# List to store pose landmarks and shot labels
pose_data = []

def capture_pose_data(shot_label, num_takes=30, frames_per_take=100):
    """
    Capture pose landmarks from the webcam and store them with the given shot label.
    The user is expected to perform the shot (cut, pull, cover_drive, straight_drive, flick) multiple times.
    The script will automatically proceed through each take.
    """
    for take in range(1, num_takes + 1):
        print(f"Take {take}/{num_takes} for '{shot_label}' shot. Get ready!")

        # Brief delay before each take to allow the user to get ready (e.g., 3 seconds)
        time.sleep(3)

        cap = cv2.VideoCapture(0)  # Use the webcam
        frame_count = 0

        while cap.isOpened() and frame_count < frames_per_take:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video frame.")
                break

            # Convert the image to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            # Draw landmarks on the frame
            if result.pose_landmarks:
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Extract key landmarks and store them
                landmarks = result.pose_landmarks.landmark
                keypoints = [
                    landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
                    
                ]
                # Append pose keypoints with shot label
                pose_data.append(keypoints + [shot_label])

                frame_count += 1

            # Show shot type in red and large font inside the window
            cv2.putText(frame, f'Shot: {shot_label.upper()}', (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)

            # Show the frame with landmarks and shot label
            cv2.imshow(f'Capture Pose for Shot: {shot_label} (Take {take})', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    print(f"Completed {num_takes} takes for '{shot_label}' shot.")

# Capture pose data for different shots with multiple takes
def main():
    # Number of takes and frames per take can be adjusted based on your needs
    num_takes = 30
    frames_per_take = 100

    # Ask user to play different shots and capture landmarks
    print("Recording 'cut' shots")
    capture_pose_data('cut', num_takes=num_takes, frames_per_take=frames_per_take)

    print("Recording 'pull' shots")
    capture_pose_data('pull', num_takes=num_takes, frames_per_take=frames_per_take)

    print("Recording 'cover_drive' shots")
    capture_pose_data('cover_drive', num_takes=num_takes, frames_per_take=frames_per_take)

    print("Recording 'straight_drive' shots")
    capture_pose_data('straight_drive', num_takes=num_takes, frames_per_take=frames_per_take)

    print("Recording 'flick' shots")
    capture_pose_data('flick', num_takes=num_takes, frames_per_take=frames_per_take)

    # Convert the pose data to a DataFrame and save as CSV
    pose_df = pd.DataFrame(pose_data, columns=[
        'nose_x', 'nose_y',
        'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',
        'left_elbow_x', 'left_elbow_y', 'right_elbow_x', 'right_elbow_y',
        'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y',
        'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y',
        'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y',
        'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y',
        'shot_type'
    ])

    pose_data_path = os.path.join(data_dir, 'pose_data_shots.csv')
    pose_df.to_csv(pose_data_path, index=False)
    print(f"Pose data saved to {pose_data_path}")

if __name__ == '__main__':
    main()
