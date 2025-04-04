import os
import json
import cv2 #type: ignore
import numpy as np #type: ignore
import random
from tensorflow.keras.models import load_model #type: ignore
import pickle
from datetime import datetime

DATASET_PATH = "sign_dataset"
MODEL_PATH = "sign_language_model.keras"
ENCODER_PATH = "label_encoder.pkl"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_DELAY = 50

model = load_model(MODEL_PATH)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


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




def load_data_info():
    """Load dataset information and return available signs"""
    if not os.path.exists(DATASET_PATH):
        return []
    
    available_signs = []
    for item in os.listdir(DATASET_PATH):
        if os.path.isdir(os.path.join(DATASET_PATH, item)):
            available_signs.append(item)
    
    return available_signs

