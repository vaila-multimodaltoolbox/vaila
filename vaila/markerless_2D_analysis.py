"""
Script: markerless_2D_analysis.py
Author: Prof. Dr. Paulo Santiago
Version: 0.2.0
Last Updated: September 28, 2024

Description:
This script performs batch processing of videos for 2D pose estimation using 
MediaPipe's Pose model. It processes videos from a specified input directory, 
overlays pose landmarks on each video frame, and exports both normalized and 
pixel-based landmark coordinates to CSV files. 

The user can configure key MediaPipe parameters via a graphical interface, 
including detection confidence, tracking confidence, model complexity, and 
whether to enable segmentation and smooth segmentation. The default settings 
prioritize the highest detection accuracy and tracking precision, which may 
increase computational cost.

New Features:
- Default values for MediaPipe parameters are set to maximize detection and 
  tracking accuracy:
    - `min_detection_confidence=1.0`
    - `min_tracking_confidence=1.0`
    - `model_complexity=2` (maximum complexity)
    - `enable_segmentation=True` (segmentation activated)
    - `smooth_segmentation=True` (smooth segmentation enabled)
- User input dialog allows fine-tuning these values if desired.

Usage:
- Run the script to open a graphical interface for selecting the input directory 
  containing video files (.mp4, .avi, .mov), the output directory, and for 
  specifying the MediaPipe configuration parameters.
- The script processes each video, generating an output video with overlaid pose 
  landmarks, and CSV files containing both normalized and pixel-based landmark 
  coordinates.

How to Execute:
1. Ensure you have all dependencies installed:
   - Install OpenCV: `pip install opencv-python`
   - Install MediaPipe: `pip install mediapipe`
   - Tkinter is usually bundled with Python installations.
2. Open a terminal and navigate to the directory where `markerless_2D_analysis.py` is located.
3. Run the script using Python:
   
   python markerless_2D_analysis.py
   
4. Follow the graphical interface prompts:
   - Select the input directory with videos (.mp4, .avi, .mov).
   - Select the base output directory for processed videos and CSVs.
   - Configure the MediaPipe parameters (or leave them as default for maximum accuracy).
5. The script will process the videos and save the outputs in the specified output directory.

Requirements:
- Python 3.11.9
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Tkinter (usually included with Python installations)
- Pillow (if using image manipulation: `pip install Pillow`)

Output:
The following files are generated for each processed video:
1. Processed Video (`*_mp.mp4`): 
   The video with the 2D pose landmarks overlaid on the original frames.
2. Normalized Landmark CSV (`*_mp_norm.csv`):
   A CSV file containing the landmark coordinates normalized to a scale between 0 and 1 
   for each frame. These coordinates represent the relative positions of landmarks in the video.
3. Pixel Landmark CSV (`*_mp_pixel.csv`):
   A CSV file containing the landmark coordinates in pixel format. The x and y coordinates 
   are scaled to the video’s resolution, representing the exact pixel positions of the landmarks.
4. Log File (`log_info.txt`):
   A log file containing video metadata and processing information, such as resolution, frame rate, 
   total number of frames, codec used, and the MediaPipe Pose configuration used in the processing.

License:
This program is free software: you can redistribute it and/or modify it under the terms of 
the GNU General Public License as published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU GPLv3 (General Public License Version 3) along with this program. 
If not, see <https://www.gnu.org/licenses/>.
"""


import cv2
import mediapipe as mp
import os
import time
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

class ConfidenceInputDialog(tk.simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(row=0)
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(row=1)
        tk.Label(master, text="Enter model complexity (0, 1, or 2):").grid(row=2)
        tk.Label(master, text="Enable segmentation? (True/False):").grid(row=3)
        tk.Label(master, text="Smooth segmentation? (True/False):").grid(row=4)
        tk.Label(master, text="Static image mode? (True/False):").grid(row=5)

        self.min_detection_entry = tk.Entry(master)
        self.min_detection_entry.insert(0, "0.1")
        self.min_tracking_entry = tk.Entry(master)
        self.min_tracking_entry.insert(0, "0.1")
        self.model_complexity_entry = tk.Entry(master)
        self.model_complexity_entry.insert(0, "2")
        self.enable_segmentation_entry = tk.Entry(master)
        self.enable_segmentation_entry.insert(0, "True")
        self.smooth_segmentation_entry = tk.Entry(master)
        self.smooth_segmentation_entry.insert(0, "True")
        self.static_image_mode_entry = tk.Entry(master)
        self.static_image_mode_entry.insert(0, "False")

        self.min_detection_entry.grid(row=0, column=1)
        self.min_tracking_entry.grid(row=1, column=1)
        self.model_complexity_entry.grid(row=2, column=1)
        self.enable_segmentation_entry.grid(row=3, column=1)
        self.smooth_segmentation_entry.grid(row=4, column=1)
        self.static_image_mode_entry.grid(row=5, column=1)

        return self.min_detection_entry

    def apply(self):
        self.result = {
            "min_detection_confidence": float(self.min_detection_entry.get()),
            "min_tracking_confidence": float(self.min_tracking_entry.get()),
            "model_complexity": int(self.model_complexity_entry.get()),
            "enable_segmentation": self.enable_segmentation_entry.get().lower() == "true",
            "smooth_segmentation": self.smooth_segmentation_entry.get().lower() == "true",
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

    # Garantir que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    output_landmarks_name = os.path.splitext(os.path.basename(video_path))[0] + "_mp_norm.csv"
    output_file_path = os.path.join(output_dir, output_landmarks_name)
    output_pixel_file_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_mp_pixel.csv"
    )

    codec = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    pose = mp.solutions.pose.Pose(
        static_image_mode=pose_config["static_image_mode"],
        min_detection_confidence=pose_config["min_detection_confidence"],
        min_tracking_confidence=pose_config["min_tracking_confidence"],
        model_complexity=pose_config["model_complexity"],
        enable_segmentation=pose_config["enable_segmentation"],
        smooth_segmentation=pose_config["smooth_segmentation"],
        smooth_landmarks=True,
    )

    headers = ["frame_index"] + [f"{name}_x,{name}_y,{name}_z" for name in landmark_names]
    pixel_headers = ["frame_index"] + [f"{name}_x,{name}_y,{name}_z" for name in landmark_names]

    frame_count = 0
    with open(output_file_path, "w") as f, open(output_pixel_file_path, "w") as f_pixel:
        f.write(",".join(headers) + "\n")
        f_pixel.write(",".join(pixel_headers) + "\n")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )
                landmarks = [f"{landmark.x:.6f},{landmark.y:.6f},{landmark.z:.6f}"
                             for landmark in results.pose_landmarks.landmark]
                f.write(f"{frame_count}," + ",".join(landmarks) + "\n")

                pixel_landmarks = [f"{int(landmark.x * width)},{int(landmark.y * height)},{landmark.z:.6f}"
                                   for landmark in results.pose_landmarks.landmark]
                f_pixel.write(f"{frame_count}," + ",".join(pixel_landmarks) + "\n")

            out.write(frame)
            frame_count += 1

    cap.release()
    out.release()
    pose.close()

    end_time = time.time()
    execution_time = end_time - start_time

    log_info_path = os.path.join(output_dir, "log_info.txt")
    with open(log_info_path, "w") as log_file:
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write(f"Codec: {codec}\n")
        log_file.write(f"Resolution: {width}x{height}\n")
        log_file.write(f"FPS: {fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")
        log_file.write(f"MediaPipe Pose Configuration: {pose_config}\n")

def process_videos_in_directory():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(
        title="Select the input directory containing videos"
    )
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
    output_base = os.path.join(output_base, f"mediapipe_{timestamp}")
    os.makedirs(output_base, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        if root != input_dir:
            continue  # Skip subdirectories

        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                video_path = os.path.join(root, file)
                output_dir = os.path.join(output_base, os.path.splitext(file)[0])
                os.makedirs(output_dir, exist_ok=True)
                print(f"Processing video: {video_path}")
                process_video(video_path, output_dir, pose_config)

if __name__ == "__main__":
    process_videos_in_directory()
