import pandas as pd
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.utils import to_categorical
from utils import compute_features

# Load and process the data
data_dir = 'data/collected_data/'
pose_data = pd.read_csv(os.path.join(data_dir, 'pose_data_shots.csv'))

# Ensure 'models/' directory exists
models_dir = 'models/'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Feature extraction (joint angles and distances)
def extract_features_and_labels(pose_data):
    X, y = [], []
    for i, row in pose_data.iterrows():
        keypoints = {
            'nose': (row['nose_x'], row['nose_y']),
            'left_wrist': (row['left_wrist_x'], row['left_wrist_y']),
            'right_wrist': (row['right_wrist_x'], row['right_wrist_y']),
            'left_elbow': (row['left_elbow_x'], row['left_elbow_y']),
            'right_elbow': (row['right_elbow_x'], row['right_elbow_y']),
            'left_shoulder': (row['left_shoulder_x'], row['left_shoulder_y']),
            'right_shoulder': (row['right_shoulder_x'], row['right_shoulder_y']),
            'left_hip': (row['left_hip_x'], row['left_hip_y']),
            'right_hip': (row['right_hip_x'], row['right_hip_y']),
            'left_knee': (row['left_knee_x'], row['left_knee_y']),
            'right_knee': (row['right_knee_x'], row['right_knee_y']),
            'left_ankle': (row['left_ankle_x'], row['left_ankle_y']),
            'right_ankle': (row['right_ankle_x'], row['right_ankle_y'])
        }
        features = compute_features(keypoints)
        X.append(features)
        y.append(row['shot_type'])
    return np.array(X), np.array(y)

# Extract features and labels
X, y = extract_features_and_labels(pose_data)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the fitted scaler
scaler_path = os.path.join(models_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print("Fitted scaler saved.")

# Encode the labels for the new shots
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save the label encoder
label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)
print("Label encoder saved.")

# Reshape data for LSTM (samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Advanced LSTM Model Architecture (adjusted for 5 output classes)
model = Sequential()

# Add a Bidirectional LSTM layer
model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
model.add(Dropout(0.3))
model.add(BatchNormalization())  # Batch Normalization to stabilize training

# Add another Bidirectional LSTM layer
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.3))
model.add(BatchNormalization())

# Add a final LSTM layer
model.add(LSTM(32))
model.add(Dropout(0.3))

# Dense layer with L2 regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

# Output layer for 5 shot classes (adjusted from 3 to 5)
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save(os.path.join(models_dir, 'advanced_lstm_model.h5'))
print("Advanced LSTM Model saved.")
