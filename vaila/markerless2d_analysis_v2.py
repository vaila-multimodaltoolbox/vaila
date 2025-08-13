"""
Script: markerless_2D_analysis_v2.py
Author: Prof. Dr. Paulo Santiago
Version: 0.2.0
Created: 01 December 2024
Updated: 09 June 2025

Description:
Version 0.2.0 of the markerless_2D_analysis.py script that corrects problems with detection
using YOLO and MediaPipe, especially for use on CPU.

Main improvements:
- Correction of MediaPipe processing within bounding boxes
- Optimization for CPU with better performance
- Better tracking of people
- More efficient and robust processing

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the MediaPipe configuration parameters.

Requirements:
- Python 3.12.11
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Ultralytics (`pip install ultralytics`)
- Tkinter (usually included with Python installations)
"""

import cv2
import mediapipe as mp
import os
import time
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import platform
import numpy as np
from ultralytics import YOLO
import torch
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
from mediapipe.framework.formats import landmark_pb2

def get_hardware_info():
    """Get detailed hardware information"""
    info = []
    info.append(f"Python version: {platform.python_version()}")
    info.append(f"OpenCV version: {cv2.__version__}")
    info.append(f"MediaPipe version: {mp.__version__}")
    info.append(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        info.append(f"CUDA available: Yes")
        info.append(f"CUDA version: {torch.version.cuda}")
        info.append(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            info.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            info.append(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        info.append("CUDA available: No")
        info.append(f"CPU: {platform.processor()}")
        info.append(f"CPU cores: {os.cpu_count()}")
    
    return "\n".join(info)

# Check for GPU availability
if torch.cuda.is_available():
    # If multiple GPUs, use the first one by default
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"Multiple GPUs detected ({gpu_count}). Using GPU 0 by default.")
    
    device = 'cuda:0'
    torch.cuda.set_device(0)
    
    # Clear cache for optimal performance
    torch.cuda.empty_cache()
    
    print("=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    print(get_hardware_info())
    print("=" * 60)
else:
    device = 'cpu'
    print("=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    print(get_hardware_info())
    print("=" * 60)
    print("No GPU detected, using CPU")
    print("For better performance, consider using a CUDA-capable GPU")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Optimize CPU performance
    torch.set_num_threads(os.cpu_count())  # Use all available CPU cores
    cv2.setNumThreads(os.cpu_count())
    cv2.setUseOptimized(True)

landmark_names = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
    "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index",
    "right_foot_index"
]


class ConfidenceInputDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(row=0)
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(row=1)
        tk.Label(master, text="Enter model complexity (0, 1, or 2):").grid(row=2)
        tk.Label(master, text="Enable segmentation? (True/False):").grid(row=3)
        tk.Label(master, text="Smooth segmentation? (True/False):").grid(row=4)
        tk.Label(master, text="Static image mode? (True/False):").grid(row=5)
        tk.Label(master, text="Use YOLO detection? (True/False):").grid(row=6)
        tk.Label(master, text="YOLO confidence threshold (0.0 - 1.0):").grid(row=7)
        tk.Label(master, text="Apply filter? (none/kalman/savgol):").grid(row=8)

        self.min_detection_entry = tk.Entry(master)
        self.min_detection_entry.insert(0, "0.5")
        self.min_tracking_entry = tk.Entry(master)
        self.min_tracking_entry.insert(0, "0.5")
        self.model_complexity_entry = tk.Entry(master)
        self.model_complexity_entry.insert(0, "2")
        self.enable_segmentation_entry = tk.Entry(master)
        self.enable_segmentation_entry.insert(0, "True")
        self.smooth_segmentation_entry = tk.Entry(master)
        self.smooth_segmentation_entry.insert(0, "True")
        self.static_image_mode_entry = tk.Entry(master)
        self.static_image_mode_entry.insert(0, "False")
        self.use_yolo_entry = tk.Entry(master)
        self.use_yolo_entry.insert(0, "True")
        self.yolo_conf_entry = tk.Entry(master)
        self.yolo_conf_entry.insert(0, "0.5")
        self.filter_type_entry = tk.Entry(master)
        self.filter_type_entry.insert(0, "kalman")

        self.min_detection_entry.grid(row=0, column=1)
        self.min_tracking_entry.grid(row=1, column=1)
        self.model_complexity_entry.grid(row=2, column=1)
        self.enable_segmentation_entry.grid(row=3, column=1)
        self.smooth_segmentation_entry.grid(row=4, column=1)
        self.static_image_mode_entry.grid(row=5, column=1)
        self.use_yolo_entry.grid(row=6, column=1)
        self.yolo_conf_entry.grid(row=7, column=1)
        self.filter_type_entry.grid(row=8, column=1)

        return self.min_detection_entry

    def apply(self):
        self.result = {
            "min_detection_confidence": float(self.min_detection_entry.get()),
            "min_tracking_confidence": float(self.min_tracking_entry.get()),
            "model_complexity": int(self.model_complexity_entry.get()),
            "enable_segmentation": self.enable_segmentation_entry.get().lower() == "true",
            "smooth_segmentation": self.smooth_segmentation_entry.get().lower() == "true",
            "static_image_mode": self.static_image_mode_entry.get().lower() == "true",
            "use_yolo": self.use_yolo_entry.get().lower() == "true",
            "yolo_conf": float(self.yolo_conf_entry.get()),
            "filter_type": self.filter_type_entry.get().lower(),
        }


def get_pose_config(existing_root=None):
    if existing_root is not None:
        root = existing_root
    else:
        root = tk.Tk()
        root.withdraw()
    dialog = ConfidenceInputDialog(root, title="Pose Configuration")
    if dialog.result:
        return dialog.result
    else:
        messagebox.showerror("Error", "No values entered.")
        return None


def download_or_load_yolo_model():
    """Download or load the most accurate YOLO model for maximum quality"""
    # Use the largest and most accurate model
    model_name = "yolov12x.pt"  # Extra large model for maximum accuracy
    
    script_dir = Path(__file__).parent.resolve()
    models_dir = script_dir / "models"
    model_path = models_dir / model_name
    
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Loading YOLO model {model_name} for maximum accuracy...")
        # Check if model exists locally
        if model_path.exists():
            print(f"Found local model at {model_path}")
            model = YOLO(str(model_path))
        else:
            print(f"Downloading {model_name}... This may take a while for the first time.")
            model = YOLO(model_name)
            # Save to local directory for future use
            if hasattr(model, 'save'):
                model.save(str(model_path))
        
        # Configure for GPU or CPU
        model.to(device)
        print(f"YOLO model loaded on {device}")
        
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Try fallback to a lighter model
        try:
            print("Trying fallback to yolov12x.pt...")
            model = YOLO("yolov12x.pt")
            model.to(device)
            return model
        except:
            print("Failed to load any YOLO model")
            return None


def detect_persons_with_yolo(frame, model, conf_threshold=0.5):
    """Detect persons in a frame using YOLO with maximum accuracy"""
    # Use higher resolution for better detection quality
    h, w = frame.shape[:2]
    
    # Use 1280 for maximum quality (or original size if smaller)
    target_size = 1280
    scale = target_size / max(h, w)
    
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_frame = frame
        scale = 1
    
    # Detection with optimized settings for maximum accuracy
    results = model(resized_frame, 
                   conf=conf_threshold, 
                   classes=0,  # only persons
                   device=device,
                   imgsz=target_size,
                   verbose=False,
                   max_det=10,  # Maximum 10 detections per image
                   agnostic_nms=True,  # Class-agnostic NMS for better results
                   retina_masks=True)  # High quality masks if using segmentation
    
    persons = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            if cls == 0:  # person class
                # Scale back to original size
                persons.append({
                    "bbox": [int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)],
                    "conf": float(conf)
                })
    
    return persons


def process_frame_with_mediapipe(frame, pose, yolo_model=None, yolo_conf=0.4, use_yolo=True):
    """
    Processa um frame completo com MediaPipe, opcionalmente usando YOLO para melhorar a detecção
    """
    height, width = frame.shape[:2]
    
    # If using YOLO, detect persons first
    best_person_bbox = None
    if use_yolo and yolo_model is not None:
        persons = detect_persons_with_yolo(frame, yolo_model, yolo_conf)
        
        # Select the person with highest confidence or largest bbox
        if persons:
            # Sort by bbox area (largest first) and confidence
            persons.sort(key=lambda p: (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1]) * p["conf"], reverse=True)
            best_person_bbox = persons[0]["bbox"]
    
    # Process with MediaPipe on the full frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if not results.pose_landmarks:
        return None, None, best_person_bbox
    
    # If we have a YOLO bbox, check if the landmarks are inside it
    if best_person_bbox and use_yolo:
        x1, y1, x2, y2 = best_person_bbox
        
        # Check if most of the landmarks are inside the bbox
        inside_count = 0
        total_visible = 0
        
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility > 0.5:
                total_visible += 1
                lm_x = landmark.x * width
                lm_y = landmark.y * height
                
                # Add 10% margin to the bbox
                margin = 0.1
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                x1_margin = x1 - bbox_w * margin
                y1_margin = y1 - bbox_h * margin
                x2_margin = x2 + bbox_w * margin
                y2_margin = y2 + bbox_h * margin
                
                if x1_margin <= lm_x <= x2_margin and y1_margin <= lm_y <= y2_margin:
                    inside_count += 1
        
        # If less than 50% of the visible landmarks are inside the bbox, ignore
        if total_visible > 0 and inside_count / total_visible < 0.5:
            return None, None, best_person_bbox
    
    # Convert landmarks to the necessary formats
    landmarks_norm = []
    landmarks_px = []
    
    for landmark in results.pose_landmarks.landmark:
        visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
        landmarks_norm.append([landmark.x, landmark.y, landmark.z, visibility])
        landmarks_px.append([int(landmark.x * width), int(landmark.y * height), landmark.z, visibility])
    
    return landmarks_norm, landmarks_px, best_person_bbox


def apply_temporal_filter(landmarks_history, current_landmarks, filter_type='none'):
    """Apply temporal filter to landmarks"""
    if filter_type == 'none' or not landmarks_history:
        return current_landmarks
    
    if filter_type == 'kalman':
        return apply_kalman_filter(landmarks_history, current_landmarks)
    elif filter_type == 'savgol' and len(landmarks_history) >= 5:
        return apply_savgol_filter(landmarks_history, current_landmarks)
    
    return current_landmarks


def apply_kalman_filter(landmarks_history, current_landmarks):
    """Apply simplified Kalman filter"""
    if not landmarks_history or current_landmarks is None:
        return current_landmarks
    
    filtered = []
    for i in range(len(current_landmarks)):
        if i < len(landmarks_history[-1]):
            # Simple weighted average between the previous and current value
            alpha = 0.7  # Weight for the current value
            prev = landmarks_history[-1][i]
            curr = current_landmarks[i]
            
            if prev is not None and curr is not None:
                filtered_val = [
                    alpha * curr[0] + (1 - alpha) * prev[0],
                    alpha * curr[1] + (1 - alpha) * prev[1],
                    alpha * curr[2] + (1 - alpha) * prev[2],
                    curr[3] if len(curr) > 3 else 1.0
                ]
                filtered.append(filtered_val)
            else:
                filtered.append(curr)
        else:
            filtered.append(current_landmarks[i])
    
    return filtered


def apply_savgol_filter(landmarks_history, current_landmarks, window_length=5):
    """Apply simplified Savitzky-Golay filter"""
    if not landmarks_history or current_landmarks is None:
        return current_landmarks
    
    # Simplify: use only moving average
    filtered = []
    window = min(window_length, len(landmarks_history))
    
    for i in range(len(current_landmarks)):
        history_values = []
        for j in range(max(0, len(landmarks_history) - window), len(landmarks_history)):
            if j < len(landmarks_history) and i < len(landmarks_history[j]):
                history_values.append(landmarks_history[j][i])
        
        if history_values and current_landmarks[i] is not None:
            history_values.append(current_landmarks[i])
            
            # Calculate average
            avg = [
                sum(v[0] for v in history_values) / len(history_values),
                sum(v[1] for v in history_values) / len(history_values),
                sum(v[2] for v in history_values) / len(history_values),
                current_landmarks[i][3] if len(current_landmarks[i]) > 3 else 1.0
            ]
            filtered.append(avg)
        else:
            filtered.append(current_landmarks[i])
    
    return filtered


def process_video(video_path, output_dir, pose_config, yolo_model=None):
    """Process a video file with optimized YOLO + MediaPipe pipeline"""
    print(f"Processing video: {video_path}")
    start_time = time.time()
    
    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    # Get
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configure output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_path.stem}_mp.mp4"
    output_file_path = output_dir / f"{video_path.stem}_mp_norm.csv"
    output_pixel_file_path = output_dir / f"{video_path.stem}_mp_pixel.csv"
    
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
    
    # History for temporal filtering
    landmarks_history = []
    
    # Prepare headers for CSV
    headers = ["frame_index"] + [
        f"{name}_x,{name}_y,{name}_z" for name in landmark_names
    ]
    
    # Lists to store landmarks
    normalized_landmarks_list = []
    pixel_landmarks_list = []
    bbox_list = []
    frames_with_missing_data = []
    
    print(f"\nProcessando {total_frames} frames...")
    
    # Process video
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Show progress
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\rFrame {frame_count}/{total_frames} ({progress:.1f}%)", end="")
        
        # Process frame
        landmarks_norm, landmarks_px, bbox = process_frame_with_mediapipe(
            frame, pose, yolo_model, 
            pose_config["yolo_conf"], 
            pose_config["use_yolo"]
        )
        
        # Apply temporal filter if configured
        if landmarks_norm and pose_config["filter_type"] != "none":
            landmarks_norm = apply_temporal_filter(
                landmarks_history, landmarks_norm, pose_config["filter_type"]
            )
            # Update landmarks_px with filtered values
            landmarks_px = []
            for lm in landmarks_norm:
                landmarks_px.append([
                    int(lm[0] * width),
                    int(lm[1] * height),
                    lm[2],
                    lm[3] if len(lm) > 3 else 1.0
                ])
        
        # Add to history (keep only last N frames)
        if landmarks_norm:
            landmarks_history.append(landmarks_norm)
            if len(landmarks_history) > 30:
                landmarks_history.pop(0)
        
        # Store results
        if landmarks_norm:
            normalized_landmarks_list.append(landmarks_norm)
            pixel_landmarks_list.append(landmarks_px)
            bbox_list.append(bbox)
        else:
            num_landmarks = len(landmark_names)
            nan_landmarks = [[np.nan, np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
            normalized_landmarks_list.append(nan_landmarks)
            pixel_landmarks_list.append(nan_landmarks)
            bbox_list.append(bbox)
            frames_with_missing_data.append(frame_count)
        
        frame_count += 1
    
    # Close capture
    cap.release()
    pose.close()
    
    print(f"\n\nSaving CSV files...")
    
    # Save CSVs
    with open(output_file_path, "w") as f_norm, open(output_pixel_file_path, "w") as f_pixel:
        f_norm.write(",".join(headers) + "\n")
        f_pixel.write(",".join(headers) + "\n")
        
        for frame_idx in range(len(normalized_landmarks_list)):
            landmarks_norm = normalized_landmarks_list[frame_idx]
            landmarks_pixel = pixel_landmarks_list[frame_idx]
            
            # Flatten only the first 3 values (x, y, z)
            flat_landmarks_norm = []
            flat_landmarks_pixel = []
            
            for lm_norm, lm_px in zip(landmarks_norm, landmarks_pixel):
                flat_landmarks_norm.extend(lm_norm[:3])  # Only x, y, z
                flat_landmarks_pixel.extend(lm_px[:3])   # Only x, y, z
            
            landmarks_norm_str = ",".join(
                "NaN" if np.isnan(value) else f"{value:.6f}"
                for value in flat_landmarks_norm
            )
            landmarks_pixel_str = ",".join(
                "NaN" if np.isnan(value) else str(int(value)) if i % 3 != 2 else f"{value:.6f}"
                for i, value in enumerate(flat_landmarks_pixel)
            )
            
            f_norm.write(f"{frame_idx}," + landmarks_norm_str + "\n")
            f_pixel.write(f"{frame_idx}," + landmarks_pixel_str + "\n")
    
    print(f"Creating video with visualization...")
    
    # Create video with visualization
    cap = cv2.VideoCapture(str(video_path))
    codec = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    # Drawing styles
    landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
    connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"\rCriando vídeo {frame_idx}/{total_frames} ({progress:.1f}%)", end="")
        
        # Recover data
        if frame_idx < len(pixel_landmarks_list):
            landmarks_px = pixel_landmarks_list[frame_idx]
            bbox = bbox_list[frame_idx]
            
            # Draw bbox if available
            if bbox and pose_config["use_yolo"]:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Check if there are valid landmarks
            if not all(np.isnan(lm[0]) for lm in landmarks_px):
                # Create landmark object for drawing
                landmark_proto = landmark_pb2.NormalizedLandmarkList()
                
                for lm in landmarks_px:
                    if not np.isnan(lm[0]):
                        landmark = landmark_proto.landmark.add()
                        landmark.x = lm[0] / width
                        landmark.y = lm[1] / height
                        landmark.z = lm[2] if not np.isnan(lm[2]) else 0
                        landmark.visibility = lm[3] if len(lm) > 3 else 1.0
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    landmark_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )
        
        out.write(frame)
        frame_idx += 1
    
    # Close resources
    cap.release()
    out.release()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Create log
    log_info_path = output_dir / "log_info.txt"
    with open(log_info_path, "w") as log_file:
        log_file.write("=" * 60 + "\n")
        log_file.write("PROCESSING LOG\n")
        log_file.write("=" * 60 + "\n\n")
        log_file.write("HARDWARE INFORMATION:\n")
        log_file.write(get_hardware_info() + "\n\n")
        log_file.write("=" * 60 + "\n")
        log_file.write("VIDEO INFORMATION:\n")
        log_file.write("=" * 60 + "\n")
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write(f"Codec: {codec}\n")
        log_file.write(f"Resolution: {width}x{height}\n")
        log_file.write(f"FPS: {fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time:.2f} seconds\n")
        log_file.write(f"Average FPS: {frame_count/execution_time:.2f}\n")
        log_file.write(f"Processing device: {device}\n\n")
        log_file.write("=" * 60 + "\n")
        log_file.write("MEDIAPIPE CONFIGURATION:\n")
        log_file.write("=" * 60 + "\n")
        for key, value in pose_config.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")
        if frames_with_missing_data:
            log_file.write(f"Frames with missing data: {len(frames_with_missing_data)}\n")
            log_file.write(f"Missing data percentage: {len(frames_with_missing_data)/frame_count*100:.2f}%\n")
        else:
            log_file.write("No frames with missing data.\n")
        
        # Add memory usage if GPU
        if device != 'cpu':
            log_file.write(f"\nGPU Memory used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB\n")
    
    print(f"\n\nProcessing completed!")
    print(f"Total time: {execution_time:.2f} seconds")
    print(f"Average FPS: {frame_count/execution_time:.2f}")
    print(f"Files saved in: {output_dir}")


def process_videos_in_directory(existing_root=None):
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    
    if existing_root is not None:
        root = existing_root
    else:
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
    
    pose_config = get_pose_config(root)
    if not pose_config:
        return
    
    # Load YOLO model if necessary
    yolo_model = None
    if pose_config["use_yolo"]:
        yolo_model = download_or_load_yolo_model()
        if yolo_model is None:
            print("Warning: Could not load YOLO model, proceeding without it")
            pose_config["use_yolo"] = False
    
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
        print(f"\n\nProcessing video {i}/{len(video_files)}: {video_file.name}")
        output_dir = output_base / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            process_video(video_file, output_dir, pose_config, yolo_model)
        except Exception as e:
            print(f"\nError processing {video_file.name}: {e}")
            continue
    
    print("\n\nAll videos processed!")


if __name__ == "__main__":
    process_videos_in_directory() 