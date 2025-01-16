"""
Script: markerless2d_mpyolo.py
Author: Prof. Dr. Paulo Santiago
Version: 1.0.0
Last Updated: January 16, 2025

Description:
This script integrates YOLOv8 and MediaPipe for 2D pose estimation with improved detection accuracy and multi-person tracking. YOLOv8 detects persons in video frames, and MediaPipe estimates the pose of each detected person. The output includes videos with pose landmarks overlaid and CSV files containing normalized and pixel-based coordinates.

New Features:
- YOLOv8 for detecting multiple persons in a frame.
- MediaPipe for pose estimation on detected persons.
- Outputs normalized and pixel-based coordinates to CSV.
- Logs detailed processing information.

Usage:
1. Install dependencies:
   - Install OpenCV: `pip install opencv-python`
   - Install MediaPipe: `pip install mediapipe`
   - Install Ultralytics: `pip install ultralytics`
   - Tkinter is usually bundled with Python installations.
2. Run the script:

   python markerless2d_mpyolo.py

3. Follow prompts to select input/output directories and configuration options.

Requirements:
- Python 3.11.9
- OpenCV
- MediaPipe
- Ultralytics (YOLOv8)
- Tkinter

Output:
- Processed videos with pose landmarks.
- CSV files with normalized and pixel-based coordinates.
- Log file with processing details.

License:
This program is distributed under the GNU General Public License v3.0.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from ultralytics import YOLO

class ConfidenceInputDialog(tk.simpledialog.Dialog):
    def body(self, master):
        # YOLO Configuration
        tk.Label(master, text="YOLO Configuration", font=('Arial', 10, 'bold')).grid(
            row=0, columnspan=2, pady=10
        )
        tk.Label(master, text="YOLO confidence threshold (0.0 - 1.0):").grid(row=1)
        self.yolo_conf_entry = tk.Entry(master)
        self.yolo_conf_entry.insert(0, "0.5")
        self.yolo_conf_entry.grid(row=1, column=1)

        # MediaPipe Configuration
        tk.Label(master, text="\nMediaPipe Configuration", font=('Arial', 10, 'bold')).grid(
            row=2, columnspan=2, pady=10
        )
        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(row=3)
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(row=4)
        tk.Label(master, text="Enter model complexity (0, 1, or 2):").grid(row=5)
        tk.Label(master, text="Enable segmentation? (True/False):").grid(row=6)
        tk.Label(master, text="Smooth segmentation? (True/False):").grid(row=7)
        tk.Label(master, text="Static image mode? (True/False):").grid(row=8)

        self.min_detection_entry = tk.Entry(master)
        self.min_detection_entry.insert(0, "0.1")
        self.min_tracking_entry = tk.Entry(master)
        self.min_tracking_entry.insert(0, "0.1")
        self.model_complexity_entry = tk.Entry(master)
        self.model_complexity_entry.insert(0, "2")
        self.enable_segmentation_entry = tk.Entry(master)
        self.enable_segmentation_entry.insert(0, "False")
        self.smooth_segmentation_entry = tk.Entry(master)
        self.smooth_segmentation_entry.insert(0, "False")
        self.static_image_mode_entry = tk.Entry(master)
        self.static_image_mode_entry.insert(0, "False")

        self.min_detection_entry.grid(row=3, column=1)
        self.min_tracking_entry.grid(row=4, column=1)
        self.model_complexity_entry.grid(row=5, column=1)
        self.enable_segmentation_entry.grid(row=6, column=1)
        self.smooth_segmentation_entry.grid(row=7, column=1)
        self.static_image_mode_entry.grid(row=8, column=1)

        return self.min_detection_entry

    def apply(self):
        self.result = {
            # YOLO config
            "yolo_conf": float(self.yolo_conf_entry.get()),
            
            # MediaPipe config
            "min_detection_confidence": float(self.min_detection_entry.get()),
            "min_tracking_confidence": float(self.min_tracking_entry.get()),
            "model_complexity": int(self.model_complexity_entry.get()),
            "enable_segmentation": False,
            "smooth_segmentation": False,
            "static_image_mode": self.static_image_mode_entry.get().lower() == "true",
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

def process_video(video_path, output_dir, config):
    # Create a specific directory for this video
    video_output_dir = output_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Update output paths to use video-specific directory
    output_video_path = video_output_dir / f"{video_path.stem}_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Use YOLO from vaila directory
    current_dir = Path(__file__).parent
    yolo_path = current_dir / 'yolo11n.pt'
    yolo = YOLO(str(yolo_path))
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=config['static_image_mode'],
        min_detection_confidence=config['min_detection_confidence'],
        min_tracking_confidence=config['min_tracking_confidence'],
        model_complexity=config['model_complexity'],
        enable_segmentation=config['enable_segmentation'],
        smooth_segmentation=config['smooth_segmentation'],
        smooth_landmarks=True
    )

    # Dictionary to store landmarks for each person
    person_landmarks = {}
    
    # Headers for CSV files
    headers = ["frame"] + [f"{landmark}_{axis}" for landmark in range(33) for axis in ["x", "y", "z"]]

    for frame_idx in range(total_frames):
        success, frame = cap.read()
        if not success:
            break

        results = yolo(frame, conf=config['yolo_conf'])
        
        for person_idx, detection in enumerate(results[0].boxes.data):
            x_min, y_min, x_max, y_max, conf, cls = detection.cpu().numpy()
            if int(cls) != 0:  # Only process persons (class 0)
                continue

            # Draw bounding box and ID
            x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Add person ID text with background
            text = f"ID: {person_idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background rectangle for text
            cv2.rectangle(frame, 
                        (x_min, y_min - text_size[1] - 10), 
                        (x_min + text_size[0] + 10, y_min), 
                        (0, 255, 0), 
                        -1)  # Filled rectangle
            
            # Draw text
            cv2.putText(frame, 
                       text, 
                       (x_min + 5, y_min - 5), 
                       font, 
                       font_scale, 
                       (0, 0, 0),  # Black text
                       thickness)

            person_frame = frame[y_min:y_max, x_min:x_max]
            if person_frame.size == 0:
                continue

            person_frame_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(person_frame_rgb)

            if pose_results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame[y_min:y_max, x_min:x_max],
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                # Store landmarks for this person
                if person_idx not in person_landmarks:
                    output_csv_path = video_output_dir / f"{video_path.stem}_landmarks_person_{person_idx}.csv"
                    person_landmarks[person_idx] = open(output_csv_path, 'w')
                    person_landmarks[person_idx].write(",".join(headers) + "\n")

                landmark_data = [frame_idx] + [
                    coord for landmark in pose_results.pose_landmarks.landmark
                    for coord in (landmark.x, landmark.y, landmark.z)
                ]
                person_landmarks[person_idx].write(",".join(map(str, landmark_data)) + "\n")

        out.write(frame)

    # Close all CSV files
    for file in person_landmarks.values():
        file.close()

    cap.release()
    out.release()

    # Show completion dialog with updated path
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Processing Complete",
        f"Video processing completed successfully!\n\n"
        f"Output directory: {video_output_dir}\n"
        f"Output video: {output_video_path.name}\n"
        f"Number of person tracks: {len(person_landmarks)}"
    )

def run_markerless2d_mpyolo():
    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(title="Select Input Directory")
    if not input_dir:
        messagebox.showerror("Error", "No input directory selected.")
        return

    output_base_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_base_dir:
        messagebox.showerror("Error", "No output directory selected.")
        return

    config = get_pose_config()
    if not config:
        return

    input_dir, output_base_dir = Path(input_dir), Path(output_base_dir)
    
    video_files = list(input_dir.glob("*.*"))
    video_files = [f for f in video_files if f.suffix.lower() in [".mp4", ".avi", ".mov"]]

    for video_file in video_files:
        process_video(video_file, output_base_dir, config)

if __name__ == "__main__":
    run_markerless2d_mpyolo()
