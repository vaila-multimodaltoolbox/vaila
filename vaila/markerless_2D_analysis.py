# markerless_2D_analysis.py
# Author: Paulo Santiago
# Version: 0.0.5
# Last Updated: August 9, 2024
# Description: This script performs batch processing of videos for 2D pose estimation using MediaPipe's Pose model.
# --------------------------------------------------
# Usage Instructions:
# - Select the input directory containing video files (.mp4, .avi, .mov).
# - Select the base output directory where the processed videos and landmarks will be saved.
# - Enter the minimum detection and tracking confidence values in the provided dialog boxes.
# - The script will process each video in the directory, save the processed video with pose landmarks overlaid, and output the landmark coordinates in both relative and pixel formats.
# --------------------------------------------------

import cv2
import mediapipe as mp
import os
import time
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# Mapping of landmark indices to their names
landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear",
    "mouth_left", "mouth_right", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index"
]

class ConfidenceInputDialog(tk.simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(row=0)
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(row=1)

        self.min_detection_confidence = tk.DoubleVar(value=0.1)
        self.min_tracking_confidence = tk.DoubleVar(value=0.1)

        self.min_detection_entry = tk.Entry(master, textvariable=self.min_detection_confidence)
        self.min_tracking_entry = tk.Entry(master, textvariable=self.min_tracking_confidence)

        self.min_detection_entry.grid(row=0, column=1)
        self.min_tracking_entry.grid(row=1, column=1)

        return self.min_detection_entry  # initial focus

    def apply(self):
        # Ensure the dialog's result captures the current value
        self.result = {
            'min_detection_confidence': float(self.min_detection_entry.get()),
            'min_tracking_confidence': float(self.min_tracking_entry.get())
        }

def get_pose_config():
    root = tk.Tk()
    root.withdraw()

    dialog = ConfidenceInputDialog(root, title="Pose Configuration")
    if dialog.result:
        return dialog.result
    else:
        messagebox.showerror("Error", "No values entered.")
        return None

def process_video(video_path, output_dir, pose_config):
    print(f"Processing video: {video_path}")
    start_time = time.time()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_name = os.path.splitext(os.path.basename(video_path))[0] + "_mp.mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    output_landmarks_name = os.path.splitext(os.path.basename(video_path))[0] + "_mp_landmarks.csv"
    output_file_path = os.path.join(output_dir, output_landmarks_name)
    output_pixel_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_mp_pixel.csv")

    codec = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Use the pose_config provided by the user
    pose = mp.solutions.pose.Pose(**pose_config)

    # Generate the headers for the files
    headers = ['frame_index'] + [f'{name}_x,{name}_y,{name}_z' for name in landmark_names]
    pixel_headers = ['frame_index'] + [f'{name}_x_pixel,{name}_y_pixel,{name}_z' for name in landmark_names]

    frame_count = 0
    with open(output_file_path, 'w') as f, open(output_pixel_file_path, 'w') as f_pixel:
        f.write(','.join(headers) + '\n')
        f_pixel.write(','.join(pixel_headers) + '\n')

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                landmarks = [f"{landmark.x:.6f},{landmark.y:.6f},{landmark.z:.6f}" for landmark in results.pose_landmarks.landmark]
                f.write(f"{frame_count}," + ','.join(landmarks) + '\n')

                # Calculate pixel coordinates
                pixel_landmarks = [f"{int(landmark.x * width)},{int(landmark.y * height)},{landmark.z:.6f}" for landmark in results.pose_landmarks.landmark]
                f_pixel.write(f"{frame_count}," + ','.join(pixel_landmarks) + '\n')

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()
    pose.close()

    end_time = time.time()
    execution_time = end_time - start_time

    log_info_path = os.path.join(output_dir, "log_info.txt")
    with open(log_info_path, 'w') as log_file:
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write(f"Codec: {codec}\n")
        log_file.write(f"Resolution: {width}x{height}\n")
        log_file.write(f"FPS: {fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")
        log_file.write(f"MediaPipe Pose Configuration: {pose_config}\n")

def process_videos_in_directory():
    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(title="Select the input directory containing videos")
    if not input_dir:
        messagebox.showerror("Error", "No input directory selected.")
        return

    output_base = filedialog.askdirectory(title="Select the base output directory")
    if not output_base:
        messagebox.showerror("Error", "No output directory selected.")
        return

    pose_config = get_pose_config()
    if not pose_config:
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(output_base, f"working_{timestamp}")
    os.makedirs(output_base, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        # Check if we are in the top level of the directory
        if root != input_dir:
            continue  # Skip subdirectories

        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                output_dir = os.path.join(output_base, os.path.splitext(file)[0])
                os.makedirs(output_dir, exist_ok=True)
                print(f"Processing video: {video_path}")
                process_video(video_path, output_dir, pose_config)

if __name__ == "__main__":
    process_videos_in_directory()
