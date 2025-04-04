import os
import json
import cv2 #type: ignore
import numpy as np #type: ignore
import random
from tensorflow.keras.models import load_model #type: ignore
import pickle
from datetime import datetime

DATASET_PATH = "sign_dataset"
MODEL_PATH = "sign_language_model.h5"
ENCODER_PATH = "label_encoder.pkl"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_DELAY = 50

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # right arm
    (0, 4), (4, 5), (5, 6), (6, 8),  # left arm
    (9, 10),  # shoulders
    (11, 12), (12, 14), (14, 16),  # right leg
    (11, 13), (13, 15),  # left leg
    (11, 23), (12, 24),  # torso
    (23, 24), (23, 25), (24, 26),  # hips
    (25, 27), (27, 29), (26, 28), (28, 30)  # feet
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
]

model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=3, connections=None):
    # Draw circles
    for lm in landmarks:
        x = int(lm['x'] * FRAME_WIDTH)
        y = int(lm['y'] * FRAME_HEIGHT)
        cv2.circle(frame, (x, y), radius, color, -1)

    # Draw connections if provided
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                x1 = int(landmarks[start_idx]['x'] * FRAME_WIDTH)
                y1 = int(landmarks[start_idx]['y'] * FRAME_HEIGHT)
                x2 = int(landmarks[end_idx]['x'] * FRAME_WIDTH)
                y2 = int(landmarks[end_idx]['y'] * FRAME_HEIGHT)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)



def load_sequence_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["sequence"]

def predict_label_from_json(json_path):
    seq = load_sequence_from_json(json_path)

    features = []
    for frame in seq:
        frame_vec = []

        for group in ["pose", "left_hand", "right_hand"]:
            landmarks = frame.get(group, [])
            for pt in landmarks:
                frame_vec.extend([pt["x"], pt["y"], pt["z"]])

            max_points = 33 if group == "pose" else 21
            missing = max_points - len(landmarks)
            frame_vec.extend([0.0, 0.0, 0.0] * missing)

        features.append(frame_vec)

    max_frames = 60
    if len(features) < max_frames:
        features += [features[-1]] * (max_frames - len(features))
    features = features[:max_frames]

    input_array = np.array(features).reshape(1, max_frames, -1)
    pred = model.predict(input_array)
    label_idx = np.argmax(pred)
    return label_encoder.inverse_transform([label_idx])[0]



def find_json_for_label(label):
    label_path = os.path.join(DATASET_PATH, label)
    if not os.path.exists(label_path):
        return None
    files = [f for f in os.listdir(label_path) if f.endswith(".json")]
    return os.path.join(label_path, random.choice(files)) if files else None

def slugify(word):
    import re
    return re.sub(r'\s+', '_', word.strip().lower())

def text_to_animation(text_input, output_path: str = "output_video.mp4"):
    all_frames = []
    words = [slugify(word) for word in text_input.strip().split()]

    for word in words:
        json_path = find_json_for_label(word)

        #  Word exists in dataset
        if json_path:
            sequence = load_sequence_from_json(json_path)
            all_frames.extend(sequence)
            continue

        #  Fallback to letters
        if word.isalpha():
            print(f"[INFO] '{word}' not found, showing character-level signs...")
            for char in word:
                char_path = find_json_for_label(char)
                if char_path:
                    sequence = load_sequence_from_json(char_path)
                    all_frames.extend(sequence)
                else:
                    print(f"[WARNING] No data found for letter '{char}'")
            continue

        # Predict closest known sign
        print(f"[INFO] Predicting best match for unknown word '{word}'...")
        samples = []
        for lbl in os.listdir(DATASET_PATH):
            path = find_json_for_label(lbl)
            if path:
                samples.append(path)
        if samples:
            json_path = random.choice(samples)
            predicted_label = predict_label_from_json(json_path)
            json_path = find_json_for_label(predicted_label)
            print(f"[MODEL PREDICTED] Closest match: '{predicted_label}'")

            if json_path:
                sequence = load_sequence_from_json(json_path)
                all_frames.extend(sequence)
        else:
            print(f"[WARNING] Could not find or predict sign for '{word}'")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (FRAME_WIDTH, FRAME_HEIGHT))

    # 🖼 Render
    for frame_data in all_frames:
        frame = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255
        draw_landmarks(frame, frame_data.get("pose", []), (255, 0, 0), connections=POSE_CONNECTIONS)
        draw_landmarks(frame, frame_data.get("left_hand", []), (0, 255, 0), connections=HAND_CONNECTIONS)
        draw_landmarks(frame, frame_data.get("right_hand", []), (0, 0, 255), connections=HAND_CONNECTIONS)
        
        # Add a text label to the frame
        # cv2.putText(frame, text_input, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)

    out.release()
    print(f"[SUCCESS] Video generated at {output_path}")
    return output_path