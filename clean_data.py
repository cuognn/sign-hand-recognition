import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize MediaPipe HandLandmarks model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Function to detect hand landmarks in an image
def detect_hand_landmarks(image):
    results = hands.process(image)
    if results.multi_hand_landmarks:
        return True
    else:
        return False

# Path to your data directory
DATA_DIR = "data"

# Iterate through directories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        # Iterate through image files in the directory
        for img_path in os.listdir(dir_path):
            img_path_full = os.path.join(dir_path, img_path)
            if img_path.endswith(".jpg") or img_path.endswith(".png"):
                img = cv2.imread(img_path_full)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Detect hand landmarks
                    results = detect_hand_landmarks(img_rgb)
                    if not results:
                        # If hand landmarks not detected, delete the image
                        os.remove(img_path_full)
                        print(f"Deleted {img_path_full} because hand landmarks were not detected.")
                    else:
                        print(f"Hand landmarks detected in {img_path_full}, keeping...")

# Release resources
hands.close()