"""
Script: markerless_2D_analysis.py
Author: Prof. Dr. Paulo Santiago
Version: 0.2.1
Last Updated: January 15, 2025

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
    - `enable_segmentation=False` (segmentation activated)
    - `smooth_segmentation=False` (smooth segmentation enabled)
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
from pathlib import Path
import platform
import numpy as np  # Adicionado para trabalhar com NaN

landmark_names = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


class ConfidenceInputDialog(tk.simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(
            row=0
        )
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(
            row=1
        )
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
        self.enable_segmentation_entry.insert(0, "False")
        self.smooth_segmentation_entry = tk.Entry(master)
        self.smooth_segmentation_entry.insert(0, "False")
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
            "enable_segmentation": self.enable_segmentation_entry.get().lower()
            == "true",
            "smooth_segmentation": self.smooth_segmentation_entry.get().lower()
            == "true",
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
    """
    Process a video file using MediaPipe Pose estimation.

    Args:
        video_path (Path): Path to the input video file
        output_dir (Path): Directory to save output files
        pose_config (dict): MediaPipe Pose configuration parameters

    Returns:
        None

    Outputs:
        - Processed video with pose landmarks
        - CSV file with normalized coordinates
        - CSV file with pixel coordinates
        - Log file with processing information
    """
    if platform.system() == "Windows" and platform.version().startswith("10."):
        if len(str(video_path)) > 255 or len(str(output_dir)) > 255:
            messagebox.showerror(
                "Path Too Long",
                "The selected path is too long. Please choose a shorter path for both the input video and output directory.",
            )
            return

    print(f"Processing video: {video_path}")
    start_time = time.time()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_name = video_path.stem + "_mp.mp4"
    output_video_path = output_dir / output_video_name

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    output_landmarks_name = video_path.stem + "_mp_norm.csv"
    output_file_path = output_dir / output_landmarks_name
    output_pixel_file_path = output_dir / f"{video_path.stem}_mp_pixel.csv"

    codec = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    pose = mp.solutions.pose.Pose(
        static_image_mode=pose_config["static_image_mode"],
        min_detection_confidence=pose_config["min_detection_confidence"],
        min_tracking_confidence=pose_config["min_tracking_confidence"],
        model_complexity=pose_config["model_complexity"],
        enable_segmentation=pose_config["enable_segmentation"],
        smooth_segmentation=pose_config["smooth_segmentation"],
        smooth_landmarks=True,
    )

    headers = ["frame_index"] + [
        f"{name}_x,{name}_y,{name}_z" for name in landmark_names
    ]

    # Inicializa listas para armazenar os dados
    normalized_landmarks_list = []
    pixel_landmarks_list = []
    frames_with_missing_data = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nTotal frames to process: {total_frames}")

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Display progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(
                f"\rProcessing frame {frame_count}/{total_frames} ({progress:.1f}%)",
                end="",
            )

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )
            landmarks = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.pose_landmarks.landmark
            ]
            normalized_landmarks_list.append(landmarks)

            pixel_landmarks = [
                [int(landmark.x * width), int(landmark.y * height), landmark.z]
                for landmark in results.pose_landmarks.landmark
            ]
            pixel_landmarks_list.append(pixel_landmarks)
        else:
            # Insere NaN para os frames com dados ausentes
            num_landmarks = len(landmark_names)
            nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
            normalized_landmarks_list.append(nan_landmarks)
            pixel_landmarks_list.append(nan_landmarks)
            frames_with_missing_data.append(frame_count)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    pose.close()

    total_frames = len(normalized_landmarks_list)

    # Escreve os dados nos arquivos CSV
    with open(output_file_path, "w") as f_norm, open(
        output_pixel_file_path, "w"
    ) as f_pixel:
        f_norm.write(",".join(headers) + "\n")
        f_pixel.write(",".join(headers) + "\n")

        for frame_idx in range(total_frames):
            landmarks_norm = normalized_landmarks_list[frame_idx]
            landmarks_pixel = pixel_landmarks_list[frame_idx]

            flat_landmarks_norm = [
                coord for landmark in landmarks_norm for coord in landmark
            ]
            flat_landmarks_pixel = [
                coord for landmark in landmarks_pixel for coord in landmark
            ]

            # Converte valores NaN para a string 'NaN'
            landmarks_norm_str = ",".join(
                "NaN" if np.isnan(value) else f"{value:.6f}"
                for value in flat_landmarks_norm
            )
            landmarks_pixel_str = ",".join(
                "NaN" if np.isnan(value) else str(value)
                for value in flat_landmarks_pixel
            )

            f_norm.write(f"{frame_idx}," + landmarks_norm_str + "\n")
            f_pixel.write(f"{frame_idx}," + landmarks_pixel_str + "\n")

    end_time = time.time()
    execution_time = end_time - start_time

    log_info_path = output_dir / "log_info.txt"
    with open(log_info_path, "w") as log_file:
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write(f"Codec: {codec}\n")
        log_file.write(f"Resolution: {width}x{height}\n")
        log_file.write(f"FPS: {fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")
        log_file.write(f"MediaPipe Pose Configuration: {pose_config}\n")
        if frames_with_missing_data:
            log_file.write(
                f"Frames with missing data (NaN inserted): {frames_with_missing_data}\n"
            )
        else:
            log_file.write("No frames with missing data.\n")

    print(f"\nCompleted processing {video_path.name}")
    print(f"Output saved to: {output_dir}")
    print(f"Processing time: {time.time() - start_time:.2f} seconds\n")


def process_videos_in_directory():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

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
    output_base = Path(output_base) / f"mediapipe_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    video_files = list(input_dir.glob("*.*"))
    video_files = [
        f for f in video_files if f.suffix.lower() in [".mp4", ".avi", ".mov"]
    ]

    print(f"\nFound {len(video_files)} videos to process")

    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file.name}")
        output_dir = output_base / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        process_video(video_file, output_dir, pose_config)


if __name__ == "__main__":
    process_videos_in_directory()
