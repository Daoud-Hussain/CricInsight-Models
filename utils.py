import numpy as np

def compute_features(pose_landmarks):
    """
    Compute angles and distances between key pose landmarks.
    Includes head, shoulders, hips, knees, and ankles.
    """
    def compute_angle(a, b, c):
        # Compute angle between three points (a -> b -> c)
        ba = np.array([a[0] - b[0], a[1] - b[1]])
        bc = np.array([c[0] - b[0], c[1] - b[1]])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    # Extract keypoints (accessing tuples by index)
    nose = pose_landmarks['nose']
    left_wrist = pose_landmarks['left_wrist']
    right_wrist = pose_landmarks['right_wrist']
    left_elbow = pose_landmarks['left_elbow']
    right_elbow = pose_landmarks['right_elbow']
    left_shoulder = pose_landmarks['left_shoulder']
    right_shoulder = pose_landmarks['right_shoulder']
    left_hip = pose_landmarks['left_hip']
    right_hip = pose_landmarks['right_hip']
    left_knee = pose_landmarks['left_knee']
    right_knee = pose_landmarks['right_knee']
    left_ankle = pose_landmarks['left_ankle']
    right_ankle = pose_landmarks['right_ankle']

    # Compute angles (e.g., elbow angle, shoulder angle, hip angle, knee angle)
    left_elbow_angle = compute_angle(left_wrist, left_elbow, left_shoulder)
    right_elbow_angle = compute_angle(right_wrist, right_elbow, right_shoulder)
    left_shoulder_angle = compute_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = compute_angle(right_elbow, right_shoulder, right_hip)
    left_hip_angle = compute_angle(left_knee, left_hip, left_shoulder)
    right_hip_angle = compute_angle(right_knee, right_hip, right_shoulder)
    left_knee_angle = compute_angle(left_ankle, left_knee, left_hip)
    right_knee_angle = compute_angle(right_ankle, right_knee, right_hip)

    # Compute distances (e.g., distance between wrists, ankles, hips, shoulders)
    wrist_distance = np.linalg.norm(np.array(left_wrist) - np.array(right_wrist))
    shoulder_distance = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    hip_distance = np.linalg.norm(np.array(left_hip) - np.array(right_hip))
    ankle_distance = np.linalg.norm(np.array(left_ankle) - np.array(right_ankle))

    # Additional distances
    wrist_to_hip_distance = (np.linalg.norm(np.array(left_wrist) - np.array(left_hip)) +
                             np.linalg.norm(np.array(right_wrist) - np.array(right_hip))) / 2
    elbow_to_knee_distance = (np.linalg.norm(np.array(left_elbow) - np.array(left_knee)) +
                              np.linalg.norm(np.array(right_elbow) - np.array(right_knee))) / 2

    # Normalize distances by height (distance between nose and midpoint between ankles)
    midpoint_ankles = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
    height = np.linalg.norm(np.array(nose) - np.array(midpoint_ankles))
    
    wrist_distance /= height
    shoulder_distance /= height
    hip_distance /= height
    ankle_distance /= height
    wrist_to_hip_distance /= height
    elbow_to_knee_distance /= height

    # Combine all features into a single feature vector
    features = [
        left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
        left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle,
        wrist_distance, shoulder_distance, hip_distance, ankle_distance,
        wrist_to_hip_distance, elbow_to_knee_distance
    ]
    
    return features
