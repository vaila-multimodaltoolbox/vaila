"""
Script: markerless_3D_analysis.py
Author: Prof. Dr. Paulo Santiago
Version: 0.0.1
Creation Date: March 31, 2025
Last Updated: March 31, 2025

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
   are scaled to the video's resolution, representing the exact pixel positions of the landmarks.
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
from collections import deque
from scipy.signal import savgol_filter
import copy
from mediapipe.framework.formats import landmark_pb2

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
        tk.Label(master, text="Apply temporal filtering? (True/False):").grid(row=6)
        tk.Label(master, text="Estimate occluded landmarks? (True/False):").grid(row=7)

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
        self.apply_filtering_entry = tk.Entry(master)
        self.apply_filtering_entry.insert(0, "True")
        self.estimate_occluded_entry = tk.Entry(master)
        self.estimate_occluded_entry.insert(0, "True")

        self.min_detection_entry.grid(row=0, column=1)
        self.min_tracking_entry.grid(row=1, column=1)
        self.model_complexity_entry.grid(row=2, column=1)
        self.enable_segmentation_entry.grid(row=3, column=1)
        self.smooth_segmentation_entry.grid(row=4, column=1)
        self.static_image_mode_entry.grid(row=5, column=1)
        self.apply_filtering_entry.grid(row=6, column=1)
        self.estimate_occluded_entry.grid(row=7, column=1)

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
            "apply_filtering": self.apply_filtering_entry.get().lower() == "true",
            "estimate_occluded": self.estimate_occluded_entry.get().lower() == "true",
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


def apply_temporal_filter(landmarks_history, window=5):
    """Aplica filtro Savitzky-Golay para suavizar movimento dos landmarks"""
    if len(landmarks_history) < window:
        return landmarks_history[-1]

    # Certifique-se de que window é ímpar
    if window % 2 == 0:
        window -= 1

    filtered_landmarks = []
    for i in range(len(landmarks_history[0])):
        # Extrair histórico para este landmark
        landmark_data = [frame[i] for frame in landmarks_history]

        # Filtrar cada dimensão separadamente
        filtered_coords = []
        for dim in range(3):  # x, y, z
            values = [lm[dim] for lm in landmark_data if not np.isnan(lm[dim])]
            if len(values) >= window:
                try:
                    filtered = savgol_filter(values, window, 2)
                    filtered_coords.append(filtered[-1])
                except:
                    filtered_coords.append(landmark_data[-1][dim])
            else:
                filtered_coords.append(landmark_data[-1][dim])

        filtered_landmarks.append(filtered_coords)

    return filtered_landmarks


def estimate_occluded_landmarks(landmarks, landmarks_history=None):
    """Estima posições de landmarks ocultos com base em restrições anatômicas"""
    estimated = landmarks.copy()

    # Só prosseguir se temos alguns landmarks visíveis
    if all(np.isnan(lm[0]) for lm in landmarks):
        return landmarks

    # 1. Regras de simetria bilateral
    # Se um lado estiver visível mas o outro não, usar simetria
    pairs = [
        (11, 12),  # ombros
        (13, 14),  # cotovelos
        (15, 16),  # pulsos
        (23, 24),  # quadris
        (25, 26),  # joelhos
        (27, 28),  # tornozelos
    ]

    for left_idx, right_idx in pairs:
        left_visible = not np.isnan(landmarks[left_idx][0])
        right_visible = not np.isnan(landmarks[right_idx][0])

        if left_visible and not right_visible:
            # Simetria espelhada no eixo X (invertendo o centro)
            if not np.isnan(landmarks[0][0]):  # Se o nariz estiver visível
                center_x = landmarks[0][0]
                offset_x = landmarks[left_idx][0] - center_x
                estimated[right_idx][0] = center_x - offset_x
                estimated[right_idx][1] = landmarks[left_idx][1]
                estimated[right_idx][2] = landmarks[left_idx][2]

        elif right_visible and not left_visible:
            # Mesma lógica, mas para o outro lado
            if not np.isnan(landmarks[0][0]):
                center_x = landmarks[0][0]
                offset_x = landmarks[right_idx][0] - center_x
                estimated[left_idx][0] = center_x - offset_x
                estimated[left_idx][1] = landmarks[right_idx][1]
                estimated[left_idx][2] = landmarks[right_idx][2]

    # 2. Regras de continuidade de membros
    # Se ombro e pulso estiverem visíveis mas cotovelo não, estimar posição do cotovelo
    if (
        not np.isnan(landmarks[11][0])
        and not np.isnan(landmarks[15][0])
        and np.isnan(landmarks[13][0])
    ):
        # Cotovelo esquerdo: interpolação simples entre ombro e pulso
        estimated[13][0] = (landmarks[11][0] + landmarks[15][0]) / 2
        estimated[13][1] = (landmarks[11][1] + landmarks[15][1]) / 2
        estimated[13][2] = (landmarks[11][2] + landmarks[15][2]) / 2

    if (
        not np.isnan(landmarks[12][0])
        and not np.isnan(landmarks[16][0])
        and np.isnan(landmarks[14][0])
    ):
        # Cotovelo direito: interpolação simples
        estimated[14][0] = (landmarks[12][0] + landmarks[16][0]) / 2
        estimated[14][1] = (landmarks[12][1] + landmarks[16][1]) / 2
        estimated[14][2] = (landmarks[12][2] + landmarks[16][2]) / 2

    # 3. Usar histórico de landmarks se disponível
    if landmarks_history and len(landmarks_history) > 0:
        for i, landmark in enumerate(estimated):
            if np.isnan(landmark[0]):
                # Procurar o último valor válido no histórico
                for past_frame in reversed(landmarks_history):
                    if not np.isnan(past_frame[i][0]):
                        estimated[i] = past_frame[i]
                        break

    return estimated


def process_video(video_path, output_dir, pose_config):
    """
    Process a video file using MediaPipe Pose estimation.
    """
    print(f"Processing video: {video_path}")
    start_time = time.time()

    # Initial configuration
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output directory and files
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_path.stem}_mp.mp4"
    output_file_path = output_dir / f"{video_path.stem}_mp_norm.csv"

    # Initialize MediaPipe
    pose = mp.solutions.pose.Pose(
        static_image_mode=pose_config["static_image_mode"],
        min_detection_confidence=pose_config["min_detection_confidence"],
        min_tracking_confidence=pose_config["min_tracking_confidence"],
        model_complexity=pose_config["model_complexity"],
        enable_segmentation=pose_config["enable_segmentation"],
        smooth_segmentation=pose_config["smooth_segmentation"],
        smooth_landmarks=True,
    )

    # Prepare CSV headers
    headers = ["frame_index"] + [
        f"{name}_x,{name}_y,{name}_z" for name in landmark_names
    ]

    # Lists to store landmarks
    normalized_landmarks_list = []
    pixel_landmarks_list = []  # Keep for video visualization only
    frames_with_missing_data = []
    landmarks_history = deque(maxlen=10)

    print(f"\nStep 1/2: Processing landmarks (total frames: {total_frames})")

    # STEP 1: Process video and generate landmarks
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Show progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(
                f"\rProcessing frame {frame_count}/{total_frames} ({progress:.1f}%)",
                end="",
            )

        # Process frame with MediaPipe
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = [
                [landmark.x, landmark.y, landmark.z]
                for landmark in results.pose_landmarks.landmark
            ]

            # Estimate occluded landmarks
            if pose_config.get("estimate_occluded", False):
                landmarks = estimate_occluded_landmarks(
                    landmarks, list(landmarks_history)
                )

            # Add to history
            landmarks_history.append(landmarks)

            # Apply temporal filtering
            if pose_config.get("apply_filtering", False) and len(landmarks_history) > 3:
                landmarks = apply_temporal_filter(list(landmarks_history))

            # Store processed landmarks
            normalized_landmarks_list.append(landmarks)

            # Calculate pixel landmarks for video visualization only
            pixel_landmarks = [
                [int(landmark[0] * width), int(landmark[1] * height), landmark[2]]
                for landmark in landmarks
            ]
            pixel_landmarks_list.append(pixel_landmarks)
        else:
            # Insert NaN for frames without detection
            num_landmarks = len(landmark_names)
            nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
            normalized_landmarks_list.append(nan_landmarks)
            pixel_landmarks_list.append(nan_landmarks)
            frames_with_missing_data.append(frame_count)

        frame_count += 1

    # Close resources from first step
    cap.release()
    pose.close()

    # Save normalized landmarks CSV
    print("\nSaving normalized landmarks data...")
    with open(output_file_path, "w") as f_norm:
        f_norm.write(",".join(headers) + "\n")
        for frame_idx in range(len(normalized_landmarks_list)):
            landmarks_norm = normalized_landmarks_list[frame_idx]
            flat_landmarks_norm = [
                coord for landmark in landmarks_norm for coord in landmark
            ]
            landmarks_norm_str = ",".join(
                "NaN" if np.isnan(value) else f"{value:.6f}"
                for value in flat_landmarks_norm
            )
            f_norm.write(f"{frame_idx}," + landmarks_norm_str + "\n")

    print(f"\nStep 2/2: Generating video with landmarks overlay")

    # STEP 2: Generate video using processed landmarks
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    drawing_spec = mp_drawing.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=2
    )
    connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(
                f"\rGenerating video {frame_idx}/{total_frames} ({progress:.1f}%)",
                end="",
            )

        # Get processed landmarks for this frame
        if frame_idx < len(pixel_landmarks_list):
            landmarks_px = pixel_landmarks_list[frame_idx]

            # Draw landmarks using processed data
            if not all(np.isnan(lm[0]) for lm in landmarks_px):
                # Create PoseLandmarkList object for drawing
                landmark_proto = landmark_pb2.NormalizedLandmarkList()

                for i, lm in enumerate(landmarks_px):
                    landmark = landmark_proto.landmark.add()
                    landmark.x = lm[0] / width  # Normalize to 0-1
                    landmark.y = lm[1] / height  # Normalize to 0-1
                    landmark.z = lm[2] if not np.isnan(lm[2]) else 0
                    landmark.visibility = (
                        1.0  # Maximum visibility for all processed points
                    )

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    landmark_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=connection_spec,
                )

        out.write(frame)
        frame_idx += 1

    # Close resources
    cap.release()
    out.release()

    # Create log
    end_time = time.time()
    execution_time = end_time - start_time

    log_info_path = output_dir / "log_info.txt"
    with open(log_info_path, "w") as log_file:
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write(f"Normalized Landmarks CSV: {output_file_path}\n")
        log_file.write(f"Resolution: {width}x{height}\n")
        log_file.write(f"FPS: {fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")
        log_file.write(f"MediaPipe Pose Configuration: {pose_config}\n")
        if frames_with_missing_data:
            log_file.write(
                f"Frames with missing data: {len(frames_with_missing_data)}\n"
            )
        else:
            log_file.write("No frames with missing data.\n")

    print(f"\nCompleted processing {video_path.name}")
    print(f"Output saved to: {output_dir}")
    print(f"Processing time: {execution_time:.2f} seconds\n")


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
