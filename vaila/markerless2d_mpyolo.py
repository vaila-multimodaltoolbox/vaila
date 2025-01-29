"""
Script: markerless2d_mpyolo.py
Author: Prof. Dr. Paulo Santiago
Version: 0.0.5
Last Updated: January 24, 2025

Description:
This script combines YOLOv11 for person detection/tracking with MediaPipe for pose estimation.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import datetime
import pandas as pd
import subprocess
from rich import print as rprint
import time
from tkinter import ttk

# Configurações para evitar conflitos
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

# COCO classes dictionary
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
    49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
    54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
    59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
    64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
    79: 'toothbrush'
}

def initialize_csv(output_dir, class_name, object_id, is_person=False):
    """Initialize a CSV file for a specific class."""
    csv_path = os.path.join(output_dir, f'{class_name}_{object_id}.csv')
    
    if is_person:
        # For persons, include MediaPipe pose landmarks
        columns = ['frame', 'person_id', 'yolo_bbox_x1', 'yolo_bbox_y1', 
                  'yolo_bbox_x2', 'yolo_bbox_y2', 'yolo_confidence']
        for idx in range(33):
            columns.extend([f'landmark_{idx}_x', f'landmark_{idx}_y', 
                          f'landmark_{idx}_z', f'landmark_{idx}_visibility'])
    else:
        # For other classes, save only bounding box coordinates
        columns = ['frame', 'object_id', 'x1', 'y1', 'x2', 'y2', 'confidence']
    
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, index=False)
    return csv_path

def save_detection_to_csv(csv_path, frame_idx, object_id, box, confidence):
    """Salva os dados da detecção no CSV."""
    x_min, y_min, x_max, y_max = map(float, box)
    row_data = {
        'frame': frame_idx,
        'object_id': object_id,
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'confidence': confidence
    }
    
    df = pd.DataFrame([row_data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def save_landmarks_to_csv(csv_path, frame_idx, person_id, landmarks):
    """Salva os dados dos landmarks no CSV."""
    row_data = {'frame': frame_idx, 'person_id': person_id}
    
    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            prefix = f'landmark_{idx}'
            row_data[f'{prefix}_x'] = landmark.x
            row_data[f'{prefix}_y'] = landmark.y
            row_data[f'{prefix}_z'] = landmark.z
            row_data[f'{prefix}_visibility'] = landmark.visibility
    else:
        # Preenche com NaN quando não há landmarks
        for idx in range(33):  # MediaPipe Pose tem 33 landmarks
            prefix = f'landmark_{idx}'
            row_data[f'{prefix}_x'] = np.nan
            row_data[f'{prefix}_y'] = np.nan
            row_data[f'{prefix}_z'] = np.nan
            row_data[f'{prefix}_visibility'] = np.nan
    
    # Append ao CSV existente
    df = pd.DataFrame([row_data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def get_parameters_dialog():
    """Create a dialog for MediaPipe and YOLO parameters."""
    dialog = tk.Tk()
    dialog.title("Detection Parameters")
    
    # Dictionary to store the results
    params = {}
    
    # Create frames
    yolo_frame = tk.LabelFrame(dialog, text="YOLO Parameters", padx=5, pady=5)
    yolo_frame.pack(padx=10, pady=5, fill="x")
    
    mp_frame = tk.LabelFrame(dialog, text="MediaPipe Parameters", padx=5, pady=5)
    mp_frame.pack(padx=10, pady=5, fill="x")
    
    # YOLO parameters with explanations
    tk.Label(yolo_frame, text="Confidence (0-1):").grid(row=0, column=0, sticky="e")
    yolo_conf = tk.Entry(yolo_frame)
    yolo_conf.insert(0, "0.15")
    yolo_conf.grid(row=0, column=1)
    tk.Label(yolo_frame, 
            text="Confidence threshold for detections (higher = more selective)",
            font=("Arial", 8, "italic")).grid(row=1, column=0, columnspan=2)
    
    tk.Label(yolo_frame, text="IOU (0-1):").grid(row=2, column=0, sticky="e")
    yolo_iou = tk.Entry(yolo_frame)
    yolo_iou.insert(0, "0.7")
    yolo_iou.grid(row=2, column=1)
    tk.Label(yolo_frame, 
            text="Intersection over Union threshold for overlap (higher = more overlapping allowed)",
            font=("Arial", 8, "italic")).grid(row=3, column=0, columnspan=2)
    
    # Classes display in grid
    classes_frame = tk.LabelFrame(dialog, text="Available Classes (Enter numbers separated by commas)", padx=5, pady=5)
    classes_frame.pack(padx=10, pady=5, fill="both", expand=True)
    
    # Display classes in a 5-column grid
    cols = 5
    for idx, name in COCO_CLASSES.items():
        row = idx // cols
        col = idx % cols
        tk.Label(classes_frame, 
                text=f"{idx}: {name}", 
                font=("Courier", 8)).grid(row=row, column=col, sticky="w", padx=5)
    
    # Class selection entry
    tk.Label(yolo_frame, text="Selected Classes:").grid(row=4, column=0, sticky="e")
    class_entry = tk.Entry(yolo_frame)
    class_entry.insert(0, "0")  # Default to person class
    class_entry.grid(row=4, column=1)
    tk.Label(yolo_frame, 
            text="Enter class numbers separated by commas (e.g., '0,1,2')",
            font=("Arial", 8, "italic")).grid(row=5, column=0, columnspan=2)
    
    # MediaPipe parameters
    # Adiciona combobox para static_image_mode
    tk.Label(mp_frame, text="Static Image Mode (True/False):").grid(row=0, column=0, sticky="e")
    mp_static_mode = tk.Entry(mp_frame)
    mp_static_mode.insert(0, "False")
    mp_static_mode.grid(row=0, column=1)

    # Outros parâmetros do MediaPipe
    tk.Label(mp_frame, text="Model Complexity (0-2):").grid(row=2, column=0, sticky="e")
    mp_complexity = tk.Entry(mp_frame)
    mp_complexity.insert(0, "2")
    mp_complexity.grid(row=2, column=1)
    
    tk.Label(mp_frame, text="Detection Confidence (0-1):").grid(row=3, column=0, sticky="e")
    mp_detection_conf = tk.Entry(mp_frame)
    mp_detection_conf.insert(0, "0.15")
    mp_detection_conf.grid(row=3, column=1)
    
    tk.Label(mp_frame, text="Tracking Confidence (0-1):").grid(row=4, column=0, sticky="e")
    mp_tracking_conf = tk.Entry(mp_frame)
    mp_tracking_conf.insert(0, "0.15")
    mp_tracking_conf.grid(row=4, column=1)
    
    def on_submit():
        try:
            params["yolo_conf"] = float(yolo_conf.get())
            params["yolo_iou"] = float(yolo_iou.get())
            
            # Parse class numbers
            class_str = class_entry.get().strip()
            if not class_str:
                tk.messagebox.showerror("Error", "Please enter at least one class number")
                return
            
            try:
                selected_classes = [int(x.strip()) for x in class_str.split(',')]
                # Validate class numbers
                invalid_classes = [x for x in selected_classes if x not in COCO_CLASSES]
                if invalid_classes:
                    tk.messagebox.showerror("Error", f"Invalid class numbers: {invalid_classes}")
                    return
                params["yolo_classes"] = selected_classes
            except ValueError:
                tk.messagebox.showerror("Error", "Invalid class number format. Use numbers separated by commas")
                return
            
            # Adiciona static_image_mode aos parâmetros
            static_mode_text = mp_static_mode.get().strip().lower()
            if static_mode_text not in ['true', 'false']:
                tk.messagebox.showerror("Error", "Static Image Mode must be 'True' or 'False'")
                return
            params["mp_static_mode"] = static_mode_text == 'true'
            
            params["mp_complexity"] = int(mp_complexity.get())
            params["mp_detection_conf"] = float(mp_detection_conf.get())
            params["mp_tracking_conf"] = float(mp_tracking_conf.get())
            
            dialog.quit()
        except ValueError as e:
            tk.messagebox.showerror("Error", "Please enter valid values")
    
    tk.Button(dialog, text="Start Processing", command=on_submit).pack(pady=10)
    
    # Center the dialog
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    dialog.mainloop()
    
    return params if params else None

def process_person_with_mediapipe(frame, bbox, pose, width, height, mp_drawing, mp_pose):
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validação das dimensões
        if width <= 0 or height <= 0:
            print("Invalid frame dimensions")
            return None, None, None, None
            
        if x1 >= x2 or y1 >= y2:
            print("Invalid bounding box coordinates")
            return None, None, None, None

        # Calculate dynamic padding based on bbox size
        box_w = x2 - x1
        box_h = y2 - y1
        
        # Add 15% padding to improve pose estimation
        pad_x = int(box_w * 0.15)
        pad_y = int(box_h * 0.15)
        
        # Apply padding with bounds checking
        x1_pad = max(0, x1 - pad_x)
        x2_pad = min(width, x2 + pad_x)
        y1_pad = max(0, y1 - pad_y)
        y2_pad = min(height, y2 + pad_y)
        
        # Create a masked version of the frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1_pad, y1_pad), (x2_pad, y2_pad), 255, -1)
        
        # Apply mask to frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Extract person crop with padding
        person_crop = masked_frame[y1_pad:y2_pad, x1_pad:x2_pad]
        
        if person_crop.size == 0:
            return None, None, None, None
        
        # Process with MediaPipe
        crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        results = pose.process(crop_rgb)
        
        if not results.pose_landmarks:
            return None, None, None, None
        
        # Create copies for normalized and pixel coordinates
        landmarks_normalized = []
        landmarks_pixels = []
        
        crop_width = x2_pad - x1_pad
        crop_height = y2_pad - y1_pad
        
        for landmark in results.pose_landmarks.landmark:
            # Get coordinates in crop space
            crop_x = landmark.x * crop_width
            crop_y = landmark.y * crop_height
            
            # Convert to global frame coordinates (pixels)
            global_x = int(crop_x + x1_pad)
            global_y = int(crop_y + y1_pad)
            
            # Store pixel coordinates
            landmarks_pixels.append({
                'x': global_x,
                'y': global_y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
            
            # Store normalized coordinates (0-1)
            landmarks_normalized.append({
                'x': global_x / width,
                'y': global_y / height,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        # Draw landmarks on crop
        annotated_crop = person_crop.copy()
        mp_drawing.draw_landmarks(
            annotated_crop,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 0, 255),  # Red for points
                thickness=2,
                circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 255, 255),  # White for connections
                thickness=1
            )
        )
        
        return landmarks_normalized, landmarks_pixels, annotated_crop, (x1_pad, y1_pad, x2_pad, y2_pad)

    except Exception as e:
        print(f"Error in process_person_with_mediapipe: {e}")
        return None, None, None, None

def save_person_data_to_csv(csv_path, frame_idx, person_id, bbox, confidence, 
                          landmarks_norm=None, landmarks_px=None):
    """
    Save person detection and pose data to CSV.
    
    Args:
        csv_path: Path to save CSV file
        frame_idx: Current frame number
        person_id: Person tracking ID
        bbox: YOLO bounding box [x1, y1, x2, y2]
        confidence: YOLO detection confidence
        landmarks_norm: List of normalized landmarks (0-1)
        landmarks_px: List of landmarks in pixel coordinates
    """
    row_data = {
        'frame': frame_idx,
        'person_id': person_id,
        'yolo_bbox_x1': bbox[0],
        'yolo_bbox_y1': bbox[1],
        'yolo_bbox_x2': bbox[2],
        'yolo_bbox_y2': bbox[3],
        'yolo_confidence': confidence
    }
    
    if landmarks_norm and landmarks_px:
        for idx, (norm, px) in enumerate(zip(landmarks_norm, landmarks_px)):
            prefix = f'landmark_{idx}'
            # Save normalized coordinates
            row_data[f'{prefix}_x_norm'] = norm['x']
            row_data[f'{prefix}_y_norm'] = norm['y']
            row_data[f'{prefix}_z_norm'] = norm['z']
            row_data[f'{prefix}_visibility'] = norm['visibility']
            # Save pixel coordinates
            row_data[f'{prefix}_x_px'] = px['x']
            row_data[f'{prefix}_y_px'] = px['y']
    else:
        # Fill with NaN when no landmarks detected
        for idx in range(33):
            prefix = f'landmark_{idx}'
            # Normalized
            row_data[f'{prefix}_x_norm'] = np.nan
            row_data[f'{prefix}_y_norm'] = np.nan
            row_data[f'{prefix}_z_norm'] = np.nan
            row_data[f'{prefix}_visibility'] = np.nan
            # Pixels
            row_data[f'{prefix}_x_px'] = np.nan
            row_data[f'{prefix}_y_px'] = np.nan
    
    df = pd.DataFrame([row_data])
    df.to_csv(csv_path, mode='a', header=False, index=False)

def process_yolo_tracking(video_path, output_dir, model, params):
    """First stage: YOLO detection and tracking"""
    tracking_data = {}  # Format: {class_id: {object_id: [frame_data]}}
    
    # Configurações para detecção múltipla
    model_params = {
        'conf': params["yolo_conf"],
        'iou': params["yolo_iou"],
        'classes': params["yolo_classes"],
        'persist': True,
        'stream': True,
        'max_det': 100,
        'verbose': False
    }
    
    print("\nStarting YOLO detection and tracking...")
    
    # Process video with YOLO
    results = model.track(source=video_path, **model_params)
    
    frame_idx = 0
    for result in results:
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs):
                class_id = int(cls_id)
                object_id = int(track_id)
                
                # Initialize dictionary structure if needed
                if class_id not in tracking_data:
                    tracking_data[class_id] = {}
                if object_id not in tracking_data[class_id]:
                    tracking_data[class_id][object_id] = []
                
                # Store detection data
                tracking_data[class_id][object_id].append({
                    'frame': frame_idx,
                    'bbox': box.tolist(),
                    'confidence': float(conf)
                })
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"\rProcessing frame {frame_idx}", end="")
    
    print(f"\nProcessed {frame_idx} frames")
    return tracking_data

def process_mediapipe_pose(video_path, output_dir, tracking_data, params):
    """Second stage: MediaPipe pose estimation for tracked persons"""
    print("\nStage 2: MediaPipe Pose Estimation")
    
    # Inicializa o gerenciador de poses
    pose_manager = MediaPipePoseManager(params)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process only person class (class_id = 0)
    if 0 in tracking_data:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Processa todas as pessoas no frame atual
            for object_id in tracking_data[0]:
                frame_data = next(
                    (data for data in tracking_data[0][object_id] 
                     if data['frame'] == frame_idx), 
                    None
                )
                
                if frame_data:
                    bbox = frame_data['bbox']
                    # Obtém o pose estimator apropriado para esta pessoa
                    pose_estimator, estimator_idx = pose_manager.get_pose_estimator(bbox, object_id)
                    
                    # Processa a pose
                    landmarks_px = process_single_pose(
                        frame, bbox, pose_estimator, width, height
                    )
                    
                    if landmarks_px:
                        # Salva os dados
                        csv_path = os.path.join(output_dir, f'pose_person_{object_id}.csv')
                        save_pose_to_csv(csv_path, frame_idx, object_id, landmarks_px)
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"\rProcessing: {(frame_idx/total_frames)*100:.1f}% "
                      f"({frame_idx}/{total_frames})", end="")
    
    cap.release()
    pose_manager.close_all()

def process_single_pose(frame, bbox, pose, width, height):
    """Process single person pose estimation"""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Add padding
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    
    x1_pad = max(0, x1 - pad_x)
    x2_pad = min(width, x2 + pad_x)
    y1_pad = max(0, y1 - pad_y)
    y2_pad = min(height, y2 + pad_y)
    
    person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
    if person_crop.size == 0:
        return None
        
    results = pose.process(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    
    # Convert to pixel coordinates
    landmarks_px = {}
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        # Convert from crop to global coordinates
        x_px = int(landmark.x * (x2_pad - x1_pad) + x1_pad)
        y_px = int(landmark.y * (y2_pad - y1_pad) + y1_pad)
        
        # Usar o mesmo formato de nome das colunas em todos os lugares
        landmarks_px[f'landmark_{idx}_x_norm'] = landmark.x
        landmarks_px[f'landmark_{idx}_y_norm'] = landmark.y
        landmarks_px[f'landmark_{idx}_z_norm'] = landmark.z
        landmarks_px[f'landmark_{idx}_visibility'] = landmark.visibility
        landmarks_px[f'landmark_{idx}_x_px'] = x_px
        landmarks_px[f'landmark_{idx}_y_px'] = y_px
    
    return landmarks_px

def create_visualization_video(video_path, output_dir, tracking_data, params):
    """
    Create visualization video with bounding boxes, IDs and pose landmarks
    """
    print("\nCreating visualization video...")
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary directory for frames
    temp_dir = os.path.join(output_dir, "temp_viz_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Load pose data for all persons
    pose_data = {}
    for object_id in tracking_data[0]:  # Only for persons (class_id = 0)
        pose_file = os.path.join(output_dir, f'pose_person_{object_id}.csv')
        if os.path.exists(pose_file):
            df = pd.read_csv(pose_file)
            pose_data[object_id] = df.set_index('frame').to_dict('index')

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    try:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw tracking boxes and IDs
            if 0 in tracking_data:  # Process persons
                for object_id in tracking_data[0]:
                    # Find frame data for this person
                    frame_data = next(
                        (data for data in tracking_data[0][object_id] 
                         if data['frame'] == frame_idx), 
                        None
                    )
                    
                    if frame_data:
                        # Draw bounding box
                        bbox = frame_data['bbox']
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw ID label
                        label = f"ID:{object_id}"
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        
                        # Draw background rectangle for text
                        cv2.rectangle(
                            frame,
                            (x1, y1 - label_height - 10),
                            (x1 + label_width + 10, y1),
                            (0, 255, 0),
                            -1
                        )
                        
                        # Draw ID text
                        cv2.putText(
                            frame,
                            label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 0),
                            2
                        )

                        # Draw pose landmarks if available
                        if object_id in pose_data and frame_idx in pose_data[object_id]:
                            frame_pose = pose_data[object_id][frame_idx]
                            
                            # Create landmark list
                            landmarks = []
                            for i in range(33):  # MediaPipe has 33 landmarks
                                try:
                                    x = frame_pose[f'landmark_{i}_x_px']
                                    y = frame_pose[f'landmark_{i}_y_px']
                                    vis = frame_pose[f'landmark_{i}_visibility']
                                    
                                    if not pd.isna(x) and not pd.isna(y):
                                        landmarks.append((int(x), int(y), vis))
                                    else:
                                        landmarks.append(None)
                                except KeyError:
                                    print(f"Warning: Missing landmark data for frame {frame_idx}, person {object_id}")
                                    landmarks.append(None)

                            # Draw landmarks and connections
                            for connection in mp_pose.POSE_CONNECTIONS:
                                start_idx = connection[0]
                                end_idx = connection[1]
                                
                                if (landmarks[start_idx] is not None and 
                                    landmarks[end_idx] is not None):
                                    cv2.line(frame, 
                                           landmarks[start_idx][:2],
                                           landmarks[end_idx][:2],
                                           (255, 255, 255), 1)

                            # Draw landmark points
                            for lm in landmarks:
                                if lm is not None:
                                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 
                                             2, (0, 0, 255), -1)

            # Save frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"\rCreating visualization: {(frame_idx/total_frames)*100:.1f}%", 
                      end="")

        # Create video using FFmpeg
        print("\nEncoding final video...")
        input_pattern = os.path.join(temp_dir, "frame_%06d.png")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video = os.path.join(output_dir, f"{video_name}_visualization.mp4")
        
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_video
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    finally:
        cap.release()
        # Cleanup temporary files
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

    print(f"\nVisualization video saved to: {output_video}")

class MediaPipePoseManager:
    def __init__(self, params):
        self.pose_estimators = []  # Lista de instâncias do MediaPipe Pose
        self.pose_dims = []  # Lista de dimensões (bbox) correspondentes
        self.params = params
        
    def compare_dist(self, dim1, dim2):
        """Calcula a distância entre duas bboxes"""
        x1_1, y1_1, x2_1, y2_1 = dim1
        x1_2, y1_2, x2_2, y2_2 = dim2
        
        # Calcula os centros das bboxes
        center1_x = (x1_1 + x2_1) / 2
        center1_y = (y1_1 + y2_1) / 2
        center2_x = (x1_2 + x2_2) / 2
        center2_y = (y1_2 + y2_2) / 2
        
        # Calcula distância euclidiana entre os centros
        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def get_pose_estimator(self, bbox, object_id):
        """Retorna o pose estimator apropriado para a bbox"""
        if len(self.pose_estimators) == 0:
            # Primeiro pose estimator
            pose = mp.solutions.pose.Pose(
                static_image_mode=self.params["mp_static_mode"],
                model_complexity=self.params["mp_complexity"],
                min_detection_confidence=self.params["mp_detection_conf"],
                min_tracking_confidence=self.params["mp_tracking_conf"]
            )
            self.pose_estimators.append(pose)
            self.pose_dims.append(bbox)
            return pose, 0
            
        elif object_id >= len(self.pose_estimators):
            # Verifica se precisa criar novo estimator
            threshold_for_new = 100
            best_score = float('inf')
            best_idx = 0
            
            # Procura o melhor match existente
            for idx, dim in enumerate(self.pose_dims):
                score = self.compare_dist(dim, bbox)
                if score < best_score:
                    best_score = score
                    best_idx = idx
            
            if best_score > threshold_for_new:
                # Cria novo estimator
                pose = mp.solutions.pose.Pose(
                    static_image_mode=self.params["mp_static_mode"],
                    model_complexity=self.params["mp_complexity"],
                    min_detection_confidence=self.params["mp_detection_conf"],
                    min_tracking_confidence=self.params["mp_tracking_conf"]
                )
                self.pose_estimators.append(pose)
                self.pose_dims.append(bbox)
                return pose, len(self.pose_estimators) - 1
            else:
                # Usa o melhor match encontrado
                self.pose_dims[best_idx] = bbox
                return self.pose_estimators[best_idx], best_idx
        else:
            # Atualiza as dimensões e retorna o estimator existente
            self.pose_dims[object_id] = bbox
            return self.pose_estimators[object_id], object_id
    
    def close_all(self):
        """Fecha todas as instâncias do MediaPipe"""
        for pose in self.pose_estimators:
            pose.close()

def save_pose_to_csv(csv_path, frame_idx, person_id, landmarks_px):
    """Save pose landmarks to CSV file."""
    row_data = {
        'frame': frame_idx,
        'person_id': person_id,
    }
    # Add landmarks data
    for idx, (key, value) in enumerate(landmarks_px.items()):
        row_data[key] = value

    df = pd.DataFrame([row_data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def run_markerless2d_mpyolo():
    root = tk.Tk()
    root.withdraw()

    # Get parameters first
    params = get_parameters_dialog()
    if not params:
        print("No parameters set. Exiting...")
        return

    # Select directory containing videos
    video_dir = filedialog.askdirectory(title="Select Directory with Videos")
    if not video_dir:
        print("No directory selected. Exiting...")
        return

    # Create main output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(video_dir, f"markerless2d_processed_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Initialize models
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolo11x.pt')
    print(f"Looking for YOLO model at: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: YOLO model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    mp_pose = mp.solutions.pose  # Módulo pose
    mp_drawing = mp.solutions.drawing_utils  # Módulo drawing

    # Get list of video files in directory (not in subdirectories)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
    video_files = [f for f in os.listdir(video_dir) 
                  if os.path.isfile(os.path.join(video_dir, f)) 
                  and f.endswith(video_extensions)]

    if not video_files:
        print("No video files found in the selected directory.")
        return

    print(f"\nFound {len(video_files)} video files to process.")
    
    # Process each video
    for video_idx, video_file in enumerate(video_files, 1):
        print(f"\n[{video_idx}/{len(video_files)}] Processing {video_file}")
        video_path = os.path.join(video_dir, video_file)
        output_dir = os.path.join(main_output_dir, os.path.splitext(video_file)[0])
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage 1: YOLO tracking
        tracking_data = process_yolo_tracking(video_path, output_dir, model, params)
        
        # Stage 2: MediaPipe pose estimation
        process_mediapipe_pose(video_path, output_dir, tracking_data, params)
        
        # Create visualization video
        create_visualization_video(video_path, output_dir, tracking_data, params)

    print(f"\nBatch processing complete!")
    print(f"All results saved to: {main_output_dir}")
    
    # Try to open output folder
    try:
        if os.name == 'nt':  # Windows
            os.startfile(main_output_dir)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['xdg-open', main_output_dir])
    except:
        pass

    root.destroy()

if __name__ == "__main__":
    run_markerless2d_mpyolo()