"""
mphands.py
Created by: Flávia Pessoni Faleiros Macêdo & Paulo Roberto Pereira Santiago
date: 01/01/2025
updated: 11/02/2025

Description:
This script uses the MediaPipe Hand Landmarker in video mode to detect hand landmarks 
from a user-selected video. It processes the entire video offline, saves the processed 
video with drawn landmarks, and outputs the landmark data into a CSV file.

MediaPipe Hands for Vailá (Video Mode Offline Analysis)
---------------------------------------------------------

This script uses the MediaPipe Hand Landmarker in video mode to detect hand landmarks 
from a user-selected video. It processes the entire video offline, saves the processed 
video with drawn landmarks, and outputs the landmark data into a CSV file.

Requirements:
- Python 3.x
- OpenCV (pip install opencv-python)
- MediaPipe (pip install mediapipe)
- requests (pip install requests)
- Tkinter (usually bundled with Python)

The "hand_landmarker.task" model will be downloaded to the project's "models" folder,
following the standard used in other files.
"""

import os
import requests
import cv2
import mediapipe as mp
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import colorsys  # Added for color conversion

# Tenta importar os labels oficiais do MediaPipe Hands
try:
    from mediapipe.python.solutions.hands import HandLandmark
    # Garante que os landmarks estejam ordenados de acordo com o valor numérico
    LANDMARK_NAMES = [landmark.name for landmark in sorted(HandLandmark, key=lambda x: x.value)]
except ImportError:
    LANDMARK_NAMES = [
        "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
    ]

# Define the local directory to store models (same standard as used in other modules)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FILENAME = "hand_landmarker.task"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def download_model_if_needed():
    """
    Checks if the 'hand_landmarker.task' file exists in the 'models' folder.
    If it does not exist, downloads the model from the specified URL.
    """
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODELS_DIR, exist_ok=True)
        print("hand_landmarker.task model not found at:", MODEL_PATH)
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download the model: status code {response.status_code}")
        with open(MODEL_PATH, "wb") as model_file:
            model_file.write(response.content)
        print("Download completed! Model saved at:", MODEL_PATH)
    else:
        print("hand_landmarker.task model already exists at:", MODEL_PATH)

def get_landmark_color(hand_index, landmark_index):
    """
    Generates a unique color for a given hand and landmark index using HSV color space.
    The color is returned as a BGR tuple suitable for OpenCV.
    """
    # Map the landmark index to a hue value and use the hand index as an offset
    hue = ((landmark_index / 21.0) + (hand_index * 0.5)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    # OpenCV uses BGR order
    return (int(b * 255), int(g * 255), int(r * 255))

def draw_hand_landmarks(image, landmarks, hand_index=0):
    """
    Draws the landmarks and connections on the image with enhanced visualization.
    Each landmark is drawn as a colored circle (differentiated by landmark index and hand index)
    and connections are drawn using an average color between the connected landmarks.
    The input landmarks is expected to be a list of normalized landmark objects.
    """
    image_height, image_width = image.shape[:2]
    # Draw each landmark as a circle
    for lm_index, lm in enumerate(landmarks):
        x = int(lm.x * image_width)
        y = int(lm.y * image_height)
        color = get_landmark_color(hand_index, lm_index)
        cv2.circle(image, (x, y), 5, color, -1)  # using a slightly larger radius for visibility

    # Define hand connections as per MediaPipe Hands specification
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            x0 = int(landmarks[start_idx].x * image_width)
            y0 = int(landmarks[start_idx].y * image_height)
            x1 = int(landmarks[end_idx].x * image_width)
            y1 = int(landmarks[end_idx].y * image_height)
            color_start = get_landmark_color(hand_index, start_idx)
            color_end = get_landmark_color(hand_index, end_idx)
            # Calculate average color for the connection line
            connection_color = (
                (color_start[0] + color_end[0]) // 2,
                (color_start[1] + color_end[1]) // 2,
                (color_start[2] + color_end[2]) // 2,
            )
            cv2.line(image, (x0, y0), (x1, y1), connection_color, 2)

def select_video_file():
    """
    Opens a dialog so that the user can select the video file for analysis.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select the video for analysis",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*")]
    )
    return file_path

def run_mphands():
    """
    Processes a user-selected video offline using MediaPipe Hands.
    It saves a processed video (with drawn landmarks) and exports the landmark data into a CSV file.
    
    The CSV file is organized with one row per frame. The first column contains the frame index.
    Then, for up to 2 hands and 21 landmarks per hand, the following five columns are output:
         normalized x, normalized y, normalized z, pixel x, pixel y.
    If no detection is made for a frame (or for a hand slot), the corresponding fields are left empty.
    The CSV headers now include the actual landmark names for better identification.
    """
    print("Running Markerless Hands (Offline Video Mode)...")
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

    # Ensure the model is available
    download_model_if_needed()

    # Prompt the user to select a video file
    video_file = select_video_file()
    if not video_file:
        messagebox.showerror("Error", "No video selected. Aborting.")
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Error: Could not open the video.")
        return

    # Get video parameters
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define output file paths based on the input video's name
    video_path = Path(video_file)
    output_video_path = str(video_path.parent / f"{video_path.stem}_processed{video_path.suffix}")
    data_file_path = str(video_path.parent / f"{video_path.stem}_landmarks.csv")

    # Initialize video writer for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Open CSV file for saving landmark data
    csv_file = open(data_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    max_hands = 2
    # Build CSV header using os nomes extraídos do MediaPipe Hands
    header = ["frame"]
    for hand in range(max_hands):
        for landmark_name in LANDMARK_NAMES:
            header.extend([
                f"hand{hand}_{landmark_name}_norm_x",
                f"hand{hand}_{landmark_name}_norm_y",
                f"hand{hand}_{landmark_name}_norm_z",
                f"hand{hand}_{landmark_name}_pixel_x",
                f"hand{hand}_{landmark_name}_pixel_y"
            ])
    csv_writer.writerow(header)

    # Configure the Hand Landmarker for video mode
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # Compute a timestamp (in ms)
            timestamp = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp)

            # Build one CSV row per frame, consolidating data for up to max_hands hands.
            row = [frame_idx]
            if result.hand_landmarks:
                # For each hand slot (up to max_hands)
                for hand_slot in range(max_hands):
                    if hand_slot < len(result.hand_landmarks):
                        hand_landmarks = result.hand_landmarks[hand_slot]
                        # Draw landmarks on the frame (for display) with enhanced visualization
                        draw_hand_landmarks(frame, hand_landmarks, hand_index=hand_slot)
                        for lm_index in range(len(LANDMARK_NAMES)):
                            if lm_index < len(hand_landmarks):
                                lm = hand_landmarks[lm_index]
                                norm_x = lm.x
                                norm_y = lm.y
                                norm_z = lm.z if hasattr(lm, "z") else 0.0
                                pixel_x = int(norm_x * frame_width)
                                pixel_y = int(norm_y * frame_height)
                                row.extend([norm_x, norm_y, norm_z, pixel_x, pixel_y])
                            else:
                                row.extend(["", "", "", "", ""])
                    else:
                        row.extend([""] * (len(LANDMARK_NAMES) * 5))
            else:
                # If no hand is detected, fill with empty fields for all hand slots.
                for hand_slot in range(max_hands):
                    row.extend([""] * (len(LANDMARK_NAMES) * 5))
            csv_writer.writerow(row)
            # Write the processed frame to the output video file
            out_video.write(frame)

    cap.release()
    out_video.release()
    csv_file.close()

    print("Processing completed.")
    print("Output video saved at:", output_video_path)
    print("Landmark data saved at:", data_file_path)

if __name__ == "__main__":
    run_mphands() 