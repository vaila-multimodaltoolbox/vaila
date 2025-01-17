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
- Python 3.12.8
- OpenCV
- MediaPipe
- Ultralytics (YOLOv11)
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
from boxmot import StrongSort
import torch

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

        # Person Limit Configuration
        tk.Label(master, text="\nPerson Limit Configuration", font=('Arial', 10, 'bold')).grid(
            row=2, columnspan=2, pady=10
        )
        tk.Label(master, text="Maximum number of persons to track (0 = no limit):").grid(row=3)
        self.max_persons_entry = tk.Entry(master)
        self.max_persons_entry.insert(0, "0")
        self.max_persons_entry.grid(row=3, column=1)

        # MediaPipe Configuration
        tk.Label(master, text="\nMediaPipe Configuration", font=('Arial', 10, 'bold')).grid(
            row=4, columnspan=2, pady=10
        )
        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(row=5)
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(row=6)
        tk.Label(master, text="Enter model complexity (0, 1, or 2):").grid(row=7)
        tk.Label(master, text="Enable segmentation? (True/False):").grid(row=8)
        tk.Label(master, text="Smooth segmentation? (True/False):").grid(row=9)
        tk.Label(master, text="Static image mode? (True/False):").grid(row=10)

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

        self.min_detection_entry.grid(row=5, column=1)
        self.min_tracking_entry.grid(row=6, column=1)
        self.model_complexity_entry.grid(row=7, column=1)
        self.enable_segmentation_entry.grid(row=8, column=1)
        self.smooth_segmentation_entry.grid(row=9, column=1)
        self.static_image_mode_entry.grid(row=10, column=1)

        return self.min_detection_entry

    def apply(self):
        self.result = {
            # YOLO config
            "yolo_conf": float(self.yolo_conf_entry.get()),
            
            # Person Limit config
            "max_persons": int(self.max_persons_entry.get()),
            
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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = output_dir / f"{video_path.stem}_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Initialize YOLO
    current_dir = Path(__file__).parent
    yolo_path = current_dir / 'yolo11n.pt'
    yolo = YOLO(str(yolo_path))
    
    # Initialize StrongSort with minimal parameters
    tracker = StrongSort(
        reid_weights=Path(current_dir) / 'osnet_x0_25_msmt17.pt',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        half=False,
        max_age=70,
        n_init=3,
        nn_budget=100,
    )

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

        # Get YOLO detections
        results = yolo(frame, conf=config['yolo_conf'])
        
        # Convert YOLO detections to tracker format
        if len(results[0].boxes) > 0:
            dets = results[0].boxes.data.cpu().numpy()
            person_mask = dets[:, 5] == 0
            if person_mask.any():
                dets = dets[person_mask]
                
                # Limitar o número de detecções se max_persons > 0
                if config['max_persons'] > 0 and len(dets) > config['max_persons']:
                    # Ordenar por confiança e pegar os top-N
                    conf_sort_idx = np.argsort(dets[:, 4])[::-1]  # Ordenar por confiança (decrescente)
                    dets = dets[conf_sort_idx[:config['max_persons']]]
                
                dets_for_tracker = np.column_stack((
                    dets[:, :4],
                    dets[:, 4],
                    np.zeros(len(dets))
                ))
                
                tracks = tracker.update(dets_for_tracker, frame)

                # Process each tracked person
                for track in tracks:
                    # track format: [x1, y1, x2, y2, track_id, class_id, conf]
                    x_min, y_min, x_max, y_max = map(int, track[:4])
                    track_id = int(track[4])

                    # Draw bounding box with persistent ID
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Add person ID text with background
                    text = f"ID: {track_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, 
                                (x_min, y_min - text_size[1] - 10), 
                                (x_min + text_size[0] + 10, y_min), 
                                (0, 255, 0), 
                                -1)
                    
                    # Draw text
                    cv2.putText(frame, 
                               text, 
                               (x_min + 5, y_min - 5), 
                               font, 
                               font_scale, 
                               (0, 0, 0),
                               thickness)

                    # Process MediaPipe pose
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

                        # Store landmarks with persistent track_id
                        if track_id not in person_landmarks:
                            output_csv_path = output_dir / f"{video_path.stem}_landmarks_person_{track_id}.csv"
                            person_landmarks[track_id] = open(output_csv_path, 'w')
                            person_landmarks[track_id].write(",".join(headers) + "\n")

                        landmark_data = [frame_idx] + [
                            coord for landmark in pose_results.pose_landmarks.landmark
                            for coord in (landmark.x, landmark.y, landmark.z)
                        ]
                        person_landmarks[track_id].write(",".join(map(str, landmark_data)) + "\n")

        out.write(frame)

    # Close all CSV files
    for file in person_landmarks.values():
        file.close()

    cap.release()
    out.release()

    print(f"Completed processing: {video_path.name}")

def run_markerless2d_mpyolo():
    root = tk.Tk()
    root.withdraw()

    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

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

    # Create timestamp directory (main directory)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = Path(output_base_dir) / f"vaila_mpyolo_{timestamp}"
    main_output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    video_files = list(input_dir.glob("*.*"))
    video_files = [f for f in video_files if f.suffix.lower() in [".mp4", ".avi", ".mov"]]

    total_videos = len(video_files)
    print(f"\nFound {total_videos} videos to process")
    
    success_count = 0
    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{total_videos}: {video_file.name}")
        
        # Create video-specific directory inside main directory
        video_output_dir = main_output_dir / video_file.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_video(video_file, video_output_dir, config)
            success_count += 1
        except Exception as e:
            print(f"Error processing {video_file.name}: {str(e)}")

    # Final completion dialog
    messagebox.showinfo(
        "Batch Processing Complete",
        f"Processing completed!\n\n"
        f"Total videos processed: {success_count}/{total_videos}\n"
        f"Output directory: {main_output_dir}"
    )

if __name__ == "__main__":
    run_markerless2d_mpyolo()
