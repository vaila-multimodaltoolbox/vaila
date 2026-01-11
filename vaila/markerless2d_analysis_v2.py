"""
Project: vaila Multimodal Toolbox
Script: markerless_2D_analysis_v2.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation: 29 July 2024
Update: 11 January 2026
Version: 0.1.0

Description:
This script performs batch processing of videos for 2D pose estimation using
MediaPipe's Pose model (Tasks API 0.10.31+) with optional YOLOv11 integration for
enhanced person detection. It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files.

The user can configure key parameters via a graphical interface:
- MediaPipe: detection/tracking confidence, model complexity, segmentation
- YOLO: model selection (yolo11n to yolo11x), confidence threshold
- Modes: YOLO-only, YOLO+MediaPipe, or MediaPipe-only

New Features (v0.1.0):
- Updated to use MediaPipe Tasks API (0.10.31+) instead of deprecated mp.solutions
- Manual OpenCV drawing for landmarks (compatible with new MediaPipe version)
- Automatic model download for both MediaPipe and YOLO models

Usage example:
First activate the vaila environment:
conda activate vaila
Then run the markerless2d_analysis_v2.py script:
python markerless2d_analysis_v2.py

Requirements:
- Python 3.12.12
- OpenCV (pip install opencv-python)
- MediaPipe (pip install mediapipe>=0.10.31)
- Ultralytics (pip install ultralytics)
- Tkinter (usually included with Python installations)
- PyTorch (for GPU acceleration)

Output:
The following files are generated for each processed video:
1. Processed Video (*_mp.mp4):
   The video with the 2D pose landmarks overlaid on the original frames.
2. Normalized Landmark CSV (*_mp_norm.csv):
   A CSV file containing the landmark coordinates normalized to a scale between 0 and 1
   for each frame. These coordinates represent the relative positions of landmarks in the video.
3. Pixel Landmark CSV (*_mp_pixel.csv):
   A CSV file containing the landmark coordinates in pixel format. The x and y coordinates
   are scaled to the video's resolution, representing the exact pixel positions of the landmarks.
4. Log File (log_info.txt):
   A log file containing video metadata and processing information.

License:
    This project is licensed under the terms of AGPLv3.0.
"""

import warnings

# Suppress protobuf deprecation warning from MediaPipe
# This warning comes from MediaPipe using deprecated protobuf API
# The warning is: "SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead."
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype",
    category=UserWarning,
)

import datetime
import os
import platform
import shutil
import time
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import mediapipe as mp
import numpy as np
import torch

# --- NEW IMPORTS FOR THE TASKS API (MediaPipe 0.10.31+) ---
from ultralytics import YOLO

# MANUAL DEFINITION OF THE BODY CONNECTIONS (since mp.solutions was removed)
POSE_CONNECTIONS = frozenset(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 7),
        (0, 4),
        (4, 5),
        (5, 6),
        (6, 8),
        (9, 10),
        (11, 12),
        (11, 13),
        (13, 15),
        (15, 17),
        (15, 19),
        (15, 21),
        (17, 19),
        (12, 14),
        (14, 16),
        (16, 18),
        (16, 20),
        (16, 22),
        (18, 20),
        (11, 23),
        (12, 24),
        (23, 24),
        (23, 25),
        (24, 26),
        (25, 27),
        (26, 28),
        (27, 29),
        (28, 30),
        (29, 31),
        (30, 32),
        (27, 31),
        (28, 32),
    ]
)


def get_hardware_info():
    """Get detailed hardware information"""
    info = []
    info.append(f"Python version: {platform.python_version()}")
    info.append(f"OpenCV version: {cv2.__version__}")
    info.append(f"MediaPipe version: {mp.__version__}")
    info.append(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        info.append("CUDA available: Yes")
        info.append(f"CUDA version: {torch.version.cuda}")
        info.append(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            info.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            info.append(
                f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
            )
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

    device = "cuda:0"
    torch.cuda.set_device(0)

    # Clear cache for optimal performance
    torch.cuda.empty_cache()

    print("=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    print(get_hardware_info())
    print("=" * 60)
else:
    device = "cpu"
    print("=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    print(get_hardware_info())
    print("=" * 60)
    print("No GPU detected, using CPU")
    print("For better performance, consider using a CUDA-capable GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Optimize CPU performance
    torch.set_num_threads(os.cpu_count())  # Use all available CPU cores
    cv2.setNumThreads(os.cpu_count())
    cv2.setUseOptimized(True)

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


class ConfidenceInputDialog(simpledialog.Dialog):
    def body(self, master):
        # Available YOLO11-pose models
        yolo_models = [
            "yolo11n-pose.pt",  # Nano - fastest
            "yolo11s-pose.pt",  # Small
            "yolo11m-pose.pt",  # Medium
            "yolo11l-pose.pt",  # Large
            "yolo11x-pose.pt",  # Extra Large - most accurate
        ]

        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(row=0)
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(row=1)
        tk.Label(master, text="Enter model complexity (0, 1, or 2):").grid(row=2)
        tk.Label(master, text="Enable segmentation? (True/False):").grid(row=3)
        tk.Label(master, text="Smooth segmentation? (True/False):").grid(row=4)
        tk.Label(master, text="Static image mode? (True/False):").grid(row=5)
        tk.Label(master, text="Use YOLO detection? (True/False):").grid(row=6)
        tk.Label(master, text="YOLO mode (yolo_only/yolo_mediapipe):").grid(row=7)
        tk.Label(master, text="YOLO model (yolo11n-pose.pt, yolo11s-pose.pt, etc.):").grid(row=8)
        tk.Label(master, text="YOLO confidence threshold (0.0 - 1.0):").grid(row=9)
        tk.Label(master, text="Apply filter? (none/kalman/savgol):").grid(row=10)

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

        # YOLO mode selection (yolo_only or yolo_mediapipe)
        yolo_modes = ["yolo_only", "yolo_mediapipe"]
        self.yolo_mode_var = tk.StringVar(value="yolo_mediapipe")
        self.yolo_mode_combo = ttk.Combobox(
            master,
            textvariable=self.yolo_mode_var,
            values=yolo_modes,
            state="readonly",
            width=30,
        )
        self.yolo_mode_combo.grid(row=7, column=1, sticky="ew")

        # YOLO model selection with dropdown
        self.yolo_model_var = tk.StringVar(value="yolo11x-pose.pt")
        self.yolo_model_combo = ttk.Combobox(
            master,
            textvariable=self.yolo_model_var,
            values=yolo_models,
            state="readonly",
            width=30,
        )
        self.yolo_model_combo.grid(row=8, column=1, sticky="ew")

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
        self.yolo_conf_entry.grid(row=9, column=1)
        self.filter_type_entry.grid(row=10, column=1)

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
            "yolo_mode": self.yolo_mode_var.get(),  # yolo_only or yolo_mediapipe
            "yolo_model": self.yolo_model_var.get(),  # Get selected model
            "yolo_conf": float(self.yolo_conf_entry.get()),
            "filter_type": self.filter_type_entry.get().lower(),
        }


def get_pose_config(existing_root=None):
    """Get pose configuration from user via dialog"""
    if existing_root is not None:
        root = existing_root
    else:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

    # Prepare root for dialog (for all platforms, but especially for Windows)
    try:
        # Ensure root is ready for dialogs on all platforms
        root.deiconify()
        root.update_idletasks()
        root.lift()
        root.focus_force()
        root.geometry("1x1+100+100")
        root.update_idletasks()
        # Small delay to ensure window is ready (especially on Windows)
        root.after(100, lambda: None)
        root.update()
        print("Root window prepared for dialog.")
    except Exception as e:
        print(f"Warning: Could not prepare root for dialog: {e}")

    try:
        print("Creating pose configuration dialog...")
        # Ensure root is active before creating dialog
        root.update_idletasks()
        root.update()
        # Make sure root is on top and focused
        try:
            root.lift()
            root.focus_force()
            root.attributes("-topmost", True)
            root.update()
            # Remove topmost after a moment so dialog can be used
            root.after(100, lambda: root.attributes("-topmost", False))
        except Exception:
            pass

        dialog = ConfidenceInputDialog(root, title="Pose Configuration")
        print("Dialog created, checking result...")

        # Update root after dialog closes
        root.update_idletasks()
        root.update()

        if hasattr(dialog, "result") and dialog.result:
            print("Configuration accepted.")
            return dialog.result
        else:
            print("Configuration cancelled or no values entered.")
            if existing_root is None:
                # Only show error if we're not part of a larger app
                messagebox.showerror("Error", "Configuration cancelled or no values entered.")
            return None
    except KeyboardInterrupt:
        print("\nConfiguration cancelled by user (Ctrl+C).")
        return None
    except Exception as e:
        print(f"Error in pose configuration dialog: {e}")
        import traceback

        traceback.print_exc()
        if existing_root is None:
            try:
                messagebox.showerror("Error", f"Failed to open configuration dialog: {e}")
            except Exception:
                print(f"Could not show error dialog: {e}")
        return None
    finally:
        # Clean up root if we created it (just hide, don't quit)
        # On Windows, keep root visible for subsequent dialogs if needed
        if existing_root is None and platform.system() != "Windows":
            try:
                root.withdraw()
            except Exception:
                pass


def download_yolo_model(model_name):
    """
    Download a specific YOLO model to the vaila/vaila/models directory.
    Uses the same approach as markerless_live.py which works correctly.

    Args:
        model_name: Name of the model to download (e.g., "yolov11x.pt")

    Returns:
        Path to the downloaded model file
    """
    # Use the models directory in the vaila project
    script_dir = Path(__file__).parent.resolve()
    models_dir = script_dir / "models"
    model_path = models_dir / model_name

    # Ensure models directory exists
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created models directory: {models_dir}")

    # Check if model already exists
    if model_path.exists():
        print(f"Model {model_name} already exists at {model_path}, using existing file.")
        return str(model_path)

    print(f"Downloading {model_name} to {model_path}...")
    print("This may take a while for the first time (model size ~300-600 MB).")

    try:
        # Method 1: Try to download using YOLO (will download to Ultralytics cache)
        model = YOLO(model_name)

        # Get the path where YOLO downloaded the model
        source_path = getattr(model, "ckpt_path", None)

        if source_path and os.path.exists(source_path):
            # Copy the downloaded model to our models directory
            shutil.copy2(source_path, str(model_path))
            print(f"✓ Successfully saved {model_name} to {model_path}")
            return str(model_path)
        else:
            print(f"YOLO downloaded the model but couldn't find it at {source_path}")
            print("Trying alternative download method...")

    except Exception as e:
        print(f"Error downloading with YOLO: {e}")
        print("Trying alternative download method...")

    # Method 2: Try direct download from GitHub (same as markerless_live.py)
    try:
        import requests

        # URL for the model - updated to use the correct model name and version
        # Ultralytics models are available at: https://github.com/ultralytics/assets/releases
        if "yolo11" in model_name.lower() or "yolov11" in model_name.lower():
            version_tag = "v11.0.0"
        elif "yolo8" in model_name.lower() or "yolov8" in model_name.lower():
            version_tag = "v8.0.0"
        else:
            version_tag = "v0.0.0"

        # Clean model name for URL (ensure correct format)
        # Ultralytics uses 'yolo11x.pt' format (not 'yolov11x.pt')
        url_model_name = (
            model_name.replace("yolov", "yolo") if "yolov" in model_name.lower() else model_name
        )

        # Try multiple possible URLs
        possible_urls = [
            f"https://github.com/ultralytics/assets/releases/download/{version_tag}/{url_model_name}",
            f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{url_model_name}",
        ]

        url = possible_urls[0]  # Try the versioned URL first

        print(f"Downloading from: {url}")

        # Try each URL until one works
        response = None
        for attempt_url in possible_urls:
            try:
                print(f"Trying URL: {attempt_url}")
                response = requests.get(attempt_url, stream=True, timeout=30)
                if response.status_code == 200:
                    url = attempt_url
                    break
                else:
                    print(f"  URL returned status {response.status_code}, trying next...")
            except Exception as url_error:
                print(f"  Error with URL {attempt_url}: {url_error}")
                continue

        if not response or response.status_code != 200:
            raise Exception(
                f"Could not download from any URL. Last status: {response.status_code if response else 'No response'}"
            )

        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(str(model_path), "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Print every MB
                            print(
                                f"  Downloaded: {percent:.1f}% ({downloaded / (1024 * 1024):.1f} MB)"
                            )

        print(f"✓ Successfully downloaded {model_name} using requests")
        return str(model_path)

    except Exception as e2:
        print(f"All download methods failed for {model_name}: {e2}")
        print("Trying to find the model in Ultralytics cache...")

        # Method 3: Try to find in Ultralytics cache
        home_dir = Path.home()
        cache_locations = [
            home_dir / ".ultralytics" / "weights" / model_name,
            home_dir / ".cache" / "ultralytics" / model_name,
        ]

        for cache_path in cache_locations:
            if cache_path.exists():
                print(f"Found model in cache: {cache_path}")
                try:
                    shutil.copy2(str(cache_path), str(model_path))
                    print(f"✓ Copied model from cache to {model_path}")
                    return str(model_path)
                except Exception as copy_err:
                    print(f"Warning: Could not copy from cache: {copy_err}")

        print(f"Could not find model {model_name} locally or download it.")
        print("Please manually download the model and place it in the models directory.")
        return None


def get_mediapipe_model_path(complexity):
    """Download the correct MediaPipe Tasks model based on complexity (0=Lite, 1=Full, 2=Heavy)"""
    # Use vaila/models directory for storing models
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    models = {
        0: "pose_landmarker_lite.task",
        1: "pose_landmarker_full.task",
        2: "pose_landmarker_heavy.task",
    }
    model_name = models.get(complexity, "pose_landmarker_full.task")
    model_path = models_dir / model_name

    if not model_path.exists():
        print(f"Downloading MediaPipe Tasks model ({model_name})... please wait.")
        print(f"Download location: {model_path}")
        # Correct URLs for the models
        model_urls = {
            0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        }
        url = model_urls.get(complexity, model_urls[1])
        try:
            urllib.request.urlretrieve(url, str(model_path))
            print("Download completed!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise RuntimeError("Failed to download MediaPipe model.")
    return str(model_path.resolve())


def download_or_load_yolo_model(model_name=None):
    """Download or load YOLO model for pose detection

    Args:
        model_name: Name of the model to load (e.g., "yolo11x-pose.pt")
                   If None, defaults to "yolo11x-pose.pt"
    """
    # Default to pose model if not specified
    if model_name is None:
        model_name = "yolo11x-pose.pt"  # Extra large pose model for maximum accuracy

    # Use the models directory in the vaila project
    script_dir = Path(__file__).parent.resolve()
    models_dir = script_dir / "models"
    model_path = models_dir / model_name

    try:
        print(f"Loading YOLO model {model_name} for maximum accuracy...")
        print(f"Models directory: {models_dir}")

        # Check if model exists in project models directory
        if model_path.exists():
            print(f"Found local model at {model_path}")
            model = YOLO(str(model_path), verbose=False)
        else:
            # Download the model using the same method as markerless_live.py
            downloaded_path = download_yolo_model(model_name)

            if downloaded_path and os.path.exists(downloaded_path):
                print(f"Loading downloaded model from {downloaded_path}")
                model = YOLO(downloaded_path, verbose=False)
            else:
                # If download failed, try to use YOLO's automatic download
                print("Attempting to use YOLO's automatic download...")
                model = YOLO(model_name, verbose=False)
                # Try to save it after loading
                try:
                    if hasattr(model, "ckpt_path") and model.ckpt_path:
                        source = Path(model.ckpt_path)
                        if source.exists():
                            shutil.copy2(str(source), str(model_path))
                            print(f"✓ Model saved to {model_path}")
                except Exception:
                    pass

        # Configure for GPU or CPU
        model.to(device)
        print(f"✓ YOLO model loaded successfully on {device}")

        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Try fallback to a lighter model
        try:
            # Try fallback to nano pose model
            print("Trying fallback to yolo11n-pose.pt (nano - smaller model)...")
            fallback_name = "yolo11n-pose.pt"
            fallback_path = models_dir / fallback_name

            if fallback_path.exists():
                print(f"Found fallback model at {fallback_path}")
                model = YOLO(str(fallback_path), verbose=False)
            else:
                # Download fallback model
                downloaded_path = download_yolo_model(fallback_name)

                if downloaded_path and os.path.exists(downloaded_path):
                    model = YOLO(downloaded_path, verbose=False)
                else:
                    model = YOLO(fallback_name, verbose=False)

            model.to(device)
            print(f"✓ Fallback YOLO model loaded successfully on {device}")
            return model
        except Exception as e2:
            print(f"Failed to load any YOLO model: {e2}")
            print("Note: YOLO is used only for person detection (bounding boxes).")
            print("MediaPipe will still work for pose estimation without YOLO.")
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
    results = model(
        resized_frame,
        conf=conf_threshold,
        classes=0,  # only persons
        device=device,
        imgsz=target_size,
        verbose=False,
        max_det=10,  # Maximum 10 detections per image
        agnostic_nms=True,  # Class-agnostic NMS for better results
        retina_masks=True,
    )  # High quality masks if using segmentation

    persons = []
    if results and len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = box
            if cls == 0:  # person class
                # Scale back to original size
                persons.append(
                    {
                        "bbox": [
                            int(x1 / scale),
                            int(y1 / scale),
                            int(x2 / scale),
                            int(y2 / scale),
                        ],
                        "conf": float(conf),
                    }
                )

    return persons


def process_frame_with_yolo_pose_only(frame, yolo_model, conf_threshold=0.5, frame_count=0):
    """
    Process frame using YOLO11-pose only (no MediaPipe)
    Returns landmarks in MediaPipe format (33 landmarks, with YOLO's 17 mapped)

    Args:
        frame: Input frame (BGR format)
        yolo_model: YOLO pose model
        conf_threshold: Confidence threshold for detection
        frame_count: Current frame number (for debugging)
    """
    height, width = frame.shape[:2]

    # Run YOLO pose detection
    results = yolo_model(
        frame,
        conf=conf_threshold,
        classes=0,  # only persons
        device=device,
        verbose=False,
        max_det=1,  # Get only the best detection
    )

    if not results or len(results) == 0:
        return None, None

    result = results[0]

    # Check if keypoints are available
    if not hasattr(result, "keypoints") or result.keypoints is None:
        return None, None

    if len(result.keypoints.data) == 0:
        return None, None

    # Get keypoints from YOLO (shape: [num_persons, num_keypoints, 3] where 3 = [x, y, confidence])
    try:
        keypoints = result.keypoints.data[0].cpu().numpy()  # Get first person
        # Debug: print keypoints shape
        if frame_count == 0:  # Only print for first frame
            print(f"\n  YOLO keypoints shape: {keypoints.shape}")
            print(f"  YOLO keypoints dtype: {keypoints.dtype}")
    except (IndexError, AttributeError) as e:
        print(f"\n  Warning: Could not extract keypoints: {e}")
        return None, None

    # YOLO11-pose has 17 keypoints
    # Map YOLO keypoints to MediaPipe format (33 landmarks)
    # YOLO keypoints order (17):
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    # MediaPipe landmarks (33) - we'll map YOLO's 17 to closest MediaPipe indices
    # and fill the rest with NaN
    landmarks_norm = [[np.nan, np.nan, np.nan, 0.0] for _ in range(33)]
    landmarks_px = [[np.nan, np.nan, np.nan, 0.0] for _ in range(33)]

    # Mapping from YOLO keypoint index to MediaPipe landmark index
    yolo_to_mediapipe = {
        0: 0,  # nose -> nose
        1: 2,  # left_eye -> left_eye
        2: 5,  # right_eye -> right_eye
        3: 7,  # left_ear -> left_ear
        4: 8,  # right_ear -> right_ear
        5: 11,  # left_shoulder -> left_shoulder
        6: 12,  # right_shoulder -> right_shoulder
        7: 13,  # left_elbow -> left_elbow
        8: 14,  # right_elbow -> right_elbow
        9: 15,  # left_wrist -> left_wrist
        10: 16,  # right_wrist -> right_wrist
        11: 23,  # left_hip -> left_hip
        12: 24,  # right_hip -> right_hip
        13: 25,  # left_knee -> left_knee
        14: 26,  # right_knee -> right_knee
        15: 27,  # left_ankle -> left_ankle
        16: 28,  # right_ankle -> right_ankle
    }

    # Convert YOLO keypoints to MediaPipe format
    # YOLO keypoints are in pixel coordinates (x, y, confidence)
    for yolo_idx, mp_idx in yolo_to_mediapipe.items():
        if yolo_idx < len(keypoints):
            kp = keypoints[yolo_idx]
            # YOLO keypoints: [x_pixel, y_pixel, confidence]
            try:
                x_px = float(kp[0])
                y_px = float(kp[1])
                conf = float(kp[2]) if len(kp) > 2 else 1.0
            except (ValueError, IndexError, TypeError):
                # Skip invalid keypoints
                continue

            # Check for NaN or invalid values
            if (
                np.isnan(x_px)
                or np.isnan(y_px)
                or np.isnan(conf)
                or conf < 0.3
                or x_px <= 0
                or y_px <= 0
                or x_px >= width
                or y_px >= height
            ):
                continue

            # Normalized coordinates (0-1) for MediaPipe format
            landmarks_norm[mp_idx] = [x_px / width, y_px / height, 0.0, conf]
            # Pixel coordinates - ensure valid integers
            try:
                landmarks_px[mp_idx] = [int(round(x_px)), int(round(y_px)), 0.0, conf]
            except (ValueError, OverflowError):
                # Skip if conversion fails
                continue

    return landmarks_norm, landmarks_px


def process_frame_with_mediapipe_tasks(
    frame, landmarker, timestamp_ms, yolo_model=None, yolo_conf=0.4, use_yolo=True
):
    """
    Process a frame with MediaPipe Tasks API, optionally using YOLO for better detection.

    Args:
        frame: Input frame (BGR format)
        landmarker: MediaPipe PoseLandmarker object
        timestamp_ms: Timestamp in milliseconds for video mode
        yolo_model: YOLO model for person detection
        yolo_conf: YOLO confidence threshold
        use_yolo: Whether to use YOLO for person detection

    Returns:
        landmarks_norm: Normalized landmarks (0-1)
        landmarks_px: Pixel landmarks
        best_person_bbox: Bounding box if YOLO was used
    """
    height, width = frame.shape[:2]

    # If using YOLO, detect persons first
    best_person_bbox = None
    if use_yolo and yolo_model is not None:
        persons = detect_persons_with_yolo(frame, yolo_model, yolo_conf)

        # Select the person with highest confidence or largest bbox
        if persons:
            # Sort by bbox area (largest first) and confidence
            persons.sort(
                key=lambda p: (p["bbox"][2] - p["bbox"][0])
                * (p["bbox"][3] - p["bbox"][1])
                * p["conf"],
                reverse=True,
            )
            best_person_bbox = persons[0]["bbox"]

    # Process with MediaPipe Tasks API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect pose using Tasks API
    pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not pose_landmarker_result.pose_landmarks or len(pose_landmarker_result.pose_landmarks) == 0:
        return None, None, best_person_bbox

    # Get the first pose detected
    raw_landmarks = pose_landmarker_result.pose_landmarks[0]

    # If we have a YOLO bbox, check if the landmarks are inside it
    if best_person_bbox and use_yolo:
        x1, y1, x2, y2 = best_person_bbox

        # Check if most of the landmarks are inside the bbox
        inside_count = 0
        total_visible = 0

        for landmark in raw_landmarks:
            # Tasks API doesn't have visibility, use presence instead
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

    for landmark in raw_landmarks:
        # Tasks API landmarks have x, y, z but no visibility (use 1.0 as default)
        visibility = 1.0
        landmarks_norm.append([landmark.x, landmark.y, landmark.z, visibility])
        landmarks_px.append(
            [int(landmark.x * width), int(landmark.y * height), landmark.z, visibility]
        )

    return landmarks_norm, landmarks_px, best_person_bbox


def apply_temporal_filter(landmarks_history, current_landmarks, filter_type="none"):
    """Apply temporal filter to landmarks"""
    if filter_type == "none" or not landmarks_history:
        return current_landmarks

    if filter_type == "kalman":
        return apply_kalman_filter(landmarks_history, current_landmarks)
    elif filter_type == "savgol" and len(landmarks_history) >= 5:
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
                    curr[3] if len(curr) > 3 else 1.0,
                ]
                filtered.append(filtered_val)
            else:
                filtered.append(curr)
        else:
            filtered.append(current_landmarks[i])

    return filtered


def draw_yolo_landmarks(frame, landmarks_px, width, height):
    """
    Draw YOLO landmarks on frame using OpenCV (custom drawing for YOLO's 17 keypoints)
    Only draws connections between valid landmarks
    """
    if landmarks_px is None:
        return frame

    # YOLO keypoints mapped to MediaPipe indices
    yolo_landmark_indices = [
        0,
        2,
        5,
        7,
        8,
        11,
        12,
        13,
        14,
        15,
        16,
        23,
        24,
        25,
        26,
        27,
        28,
    ]

    # Connections for YOLO landmarks (based on MediaPipe connections but only for available landmarks)
    # Format: (start_idx, end_idx) where indices are MediaPipe landmark indices
    yolo_connections = [
        # Face
        (0, 2),  # nose to left_eye
        (0, 5),  # nose to right_eye
        (2, 7),  # left_eye to left_ear
        (5, 8),  # right_eye to right_ear
        # Upper body
        (11, 12),  # left_shoulder to right_shoulder
        (11, 13),  # left_shoulder to left_elbow
        (13, 15),  # left_elbow to left_wrist
        (12, 14),  # right_shoulder to right_elbow
        (14, 16),  # right_elbow to right_wrist
        # Torso
        (11, 23),  # left_shoulder to left_hip
        (12, 24),  # right_shoulder to right_hip
        (23, 24),  # left_hip to right_hip
        # Lower body
        (23, 25),  # left_hip to left_knee
        (25, 27),  # left_knee to left_ankle
        (24, 26),  # right_hip to right_knee
        (26, 28),  # right_knee to right_ankle
    ]

    # Draw landmarks (circles)
    for idx in yolo_landmark_indices:
        if idx < len(landmarks_px):
            lm = landmarks_px[idx]
            if not np.isnan(lm[0]) and not np.isnan(lm[1]):
                x, y = int(round(lm[0])), int(round(lm[1]))
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # Draw connections (lines)
    for start_idx, end_idx in yolo_connections:
        if start_idx < len(landmarks_px) and end_idx < len(landmarks_px):
            start_lm = landmarks_px[start_idx]
            end_lm = landmarks_px[end_idx]

            if (
                not np.isnan(start_lm[0])
                and not np.isnan(start_lm[1])
                and not np.isnan(end_lm[0])
                and not np.isnan(end_lm[1])
            ):
                start_pt = (int(round(start_lm[0])), int(round(start_lm[1])))
                end_pt = (int(round(end_lm[0])), int(round(end_lm[1])))

                if (
                    0 <= start_pt[0] < width
                    and 0 <= start_pt[1] < height
                    and 0 <= end_pt[0] < width
                    and 0 <= end_pt[1] < height
                ):
                    cv2.line(frame, start_pt, end_pt, (255, 0, 0), 2)

    return frame


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
                current_landmarks[i][3] if len(current_landmarks[i]) > 3 else 1.0,
            ]
            filtered.append(avg)
        else:
            filtered.append(current_landmarks[i])

    return filtered


def process_video(video_path, output_dir, pose_config, yolo_model=None):
    """Process a video file with optimized YOLO + MediaPipe Tasks API pipeline"""
    print(f"Processing video: {video_path}")
    start_time = time.time()

    # Open the video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Default FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configure output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_path.stem}_mp.mp4"
    output_file_path = output_dir / f"{video_path.stem}_mp_norm.csv"
    output_pixel_file_path = output_dir / f"{video_path.stem}_mp_pixel.csv"

    # Initialize MediaPipe Tasks API
    model_path = get_mediapipe_model_path(pose_config["model_complexity"])

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create options for PoseLandmarker
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=pose_config["min_detection_confidence"],
        min_pose_presence_confidence=pose_config["min_tracking_confidence"],
        output_segmentation_masks=pose_config["enable_segmentation"],
    )

    # Create the PoseLandmarker
    landmarker = PoseLandmarker.create_from_options(options)

    # History for temporal filtering
    landmarks_history = []

    # Prepare headers for CSV
    headers = ["frame_index"] + [f"{name}_x,{name}_y,{name}_z" for name in landmark_names]

    # Lists to store landmarks
    normalized_landmarks_list = []
    pixel_landmarks_list = []
    bbox_list = []
    frames_with_missing_data = []

    print(f"\n{'=' * 60}")
    print(f"PROCESSANDO VÍDEO: {video_path.name}")
    print(f"{'=' * 60}")
    print(f"Total de frames: {total_frames}")
    print(f"Resolução: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    if yolo_model is not None and pose_config.get("use_yolo", False):
        yolo_model_name = pose_config.get("yolo_model", "yolo11x-pose.pt")
        yolo_mode = pose_config.get("yolo_mode", "yolo_mediapipe")
        if yolo_mode == "yolo_only":
            print("Pipeline: YOLOv11-pose apenas")
        else:
            print("Pipeline: YOLOv11 + MediaPipe")
        print(f"YOLO Model: {yolo_model_name}")
        print(f"YOLO Confidence: {pose_config.get('yolo_conf', 0.5)}")
    else:
        print("Pipeline: MediaPipe apenas")
    print(f"{'=' * 60}\n")

    # Process video
    frame_count = 0
    frames_with_pose = 0
    frames_without_pose = 0

    print("Iniciando processamento de frames...")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Calculate timestamp for Tasks API (must be monotonically increasing)
        timestamp_ms = int((frame_count * 1000) / fps) if fps > 0 else frame_count * 33

        # Show progress every 10 frames with more detail
        if frame_count % 10 == 0:
            progress = (frame_count / total_frames) * 100
            pose_rate = (frames_with_pose / (frame_count + 1)) * 100 if frame_count > 0 else 0
            print(
                f"\r  Frame {frame_count}/{total_frames} ({progress:.1f}%) | "
                f"Pose detectado: {frames_with_pose} ({pose_rate:.1f}%) | "
                f"Sem pose: {frames_without_pose}",
                end="",
                flush=True,
            )

        # Show detailed progress every 30 frames
        if frame_count % 30 == 0 and frame_count > 0:
            elapsed = time.time() - start_time
            fps_processing = frame_count / elapsed if elapsed > 0 else 0
            remaining_frames = total_frames - frame_count
            eta = remaining_frames / fps_processing if fps_processing > 0 else 0
            print(
                f"\n  Velocidade: {fps_processing:.1f} fps | "
                f"Tempo decorrido: {elapsed:.1f}s | "
                f"ETA: {eta:.1f}s",
                end="",
                flush=True,
            )

        # Process frame based on mode
        yolo_mode = pose_config.get("yolo_mode", "yolo_mediapipe")
        if yolo_mode == "yolo_only" and yolo_model is not None:
            # Use YOLO pose only
            landmarks_norm, landmarks_px = process_frame_with_yolo_pose_only(
                frame, yolo_model, pose_config["yolo_conf"], frame_count
            )
            bbox = None
            # If no landmarks detected, set to None
            if landmarks_norm is None or landmarks_px is None:
                landmarks_norm = None
                landmarks_px = None
        else:
            # Use MediaPipe Tasks API (with or without YOLO for detection)
            landmarks_norm, landmarks_px, bbox = process_frame_with_mediapipe_tasks(
                frame,
                landmarker,
                timestamp_ms,
                yolo_model,
                pose_config["yolo_conf"],
                pose_config["use_yolo"] and yolo_mode == "yolo_mediapipe",
            )

        # Apply temporal filter if configured
        if landmarks_norm and pose_config["filter_type"] != "none":
            landmarks_norm = apply_temporal_filter(
                landmarks_history, landmarks_norm, pose_config["filter_type"]
            )
            # Update landmarks_px with filtered values
            landmarks_px = []
            for lm in landmarks_norm:
                # Check for NaN before conversion
                if lm is None or len(lm) < 2:
                    landmarks_px.append([np.nan, np.nan, np.nan, 0.0])
                else:
                    x_val = lm[0] if not np.isnan(lm[0]) else np.nan
                    y_val = lm[1] if not np.isnan(lm[1]) else np.nan
                    z_val = lm[2] if len(lm) > 2 and not np.isnan(lm[2]) else np.nan
                    conf_val = lm[3] if len(lm) > 3 and not np.isnan(lm[3]) else 1.0

                    if not np.isnan(x_val) and not np.isnan(y_val):
                        landmarks_px.append(
                            [
                                int(round(x_val * width)),
                                int(round(y_val * height)),
                                z_val,
                                conf_val,
                            ]
                        )
                    else:
                        landmarks_px.append([np.nan, np.nan, z_val, conf_val])

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
            frames_with_pose += 1
        else:
            num_landmarks = len(landmark_names)
            nan_landmarks = [[np.nan, np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
            normalized_landmarks_list.append(nan_landmarks)
            pixel_landmarks_list.append(nan_landmarks)
            bbox_list.append(bbox)
            frames_with_missing_data.append(frame_count)
            frames_without_pose += 1

        frame_count += 1

    # Final progress update
    print(
        f"\r  Frame {frame_count}/{total_frames} (100.0%) | "
        f"Pose detectado: {frames_with_pose} | "
        f"Sem pose: {frames_without_pose}"
    )
    print("\n✓ Processamento de frames concluído!")
    print(f"  Total processado: {frame_count} frames")
    if frame_count > 0:
        print(
            f"  Frames com pose: {frames_with_pose} ({frames_with_pose / frame_count * 100:.1f}%)"
        )
        print(
            f"  Frames sem pose: {frames_without_pose} ({frames_without_pose / frame_count * 100:.1f}%)"
        )

    # Close capture
    cap.release()
    # Close the landmarker
    if landmarker is not None:
        landmarker.close()

    print(f"\n{'=' * 60}")
    print("SALVANDO ARQUIVOS CSV...")
    print(f"{'=' * 60}")

    # Save CSVs
    with (
        open(output_file_path, "w") as f_norm,
        open(output_pixel_file_path, "w") as f_pixel,
    ):
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
                flat_landmarks_pixel.extend(lm_px[:3])  # Only x, y, z

            landmarks_norm_str = ",".join(
                "NaN" if np.isnan(value) else f"{value:.6f}" for value in flat_landmarks_norm
            )
            landmarks_pixel_str = ",".join(
                (
                    "NaN"
                    if np.isnan(value)
                    else str(int(round(value)))
                    if i % 3 != 2 and not np.isnan(value)
                    else f"{value:.6f}"
                )
                for i, value in enumerate(flat_landmarks_pixel)
            )

            f_norm.write(f"{frame_idx}," + landmarks_norm_str + "\n")
            f_pixel.write(f"{frame_idx}," + landmarks_pixel_str + "\n")

            # Progress for CSV writing
            if frame_idx % 50 == 0:
                csv_progress = (frame_idx / len(normalized_landmarks_list)) * 100
                print(f"\r  Saving CSVs: {csv_progress:.1f}%", end="", flush=True)

    print("\r  Saving CSVs: 100.0%")
    print("✓ CSVs saved successfully!")
    print(f"  - {output_file_path.name}")
    print(f"  - {output_pixel_file_path.name}")
    print(f"\n{'=' * 60}")
    print("Creating video with visualization...")
    print(f"{'=' * 60}")

    # Create video with visualization using manual OpenCV drawing
    cap = cv2.VideoCapture(str(video_path))
    codec = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            elapsed_video = time.time() - start_time
            print(
                f"\r  Creating video: {frame_idx}/{total_frames} ({progress:.1f}%) | "
                f"Time: {elapsed_video:.1f}s",
                end="",
                flush=True,
            )

        # Recover data
        if frame_idx < len(pixel_landmarks_list):
            landmarks_px = pixel_landmarks_list[frame_idx]
            bbox = bbox_list[frame_idx]

            # Draw bbox if available (only in yolo_mediapipe mode)
            yolo_mode = pose_config.get("yolo_mode", "yolo_mediapipe")
            if bbox and pose_config["use_yolo"] and yolo_mode == "yolo_mediapipe":
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Check if there are valid landmarks
            if not all(np.isnan(lm[0]) for lm in landmarks_px):
                # Check if using YOLO-only mode (draw with custom function)
                yolo_mode = pose_config.get("yolo_mode", "yolo_mediapipe")
                if yolo_mode == "yolo_only":
                    # Use custom drawing for YOLO landmarks
                    frame = draw_yolo_landmarks(frame, landmarks_px, width, height)
                else:
                    # Use manual OpenCV drawing for MediaPipe landmarks
                    # Draw landmarks (circles)
                    points = {}
                    for i, lm in enumerate(landmarks_px):
                        if i < 33 and not np.isnan(lm[0]) and not np.isnan(lm[1]):
                            x, y = int(round(lm[0])), int(round(lm[1]))
                            if 0 <= x < width and 0 <= y < height:
                                points[i] = (x, y)
                                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                    # Draw connections (using POSE_CONNECTIONS defined at module level)
                    for start_idx, end_idx in POSE_CONNECTIONS:
                        if start_idx in points and end_idx in points:
                            cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    # Close resources
    cap.release()
    out.release()

    print(f"\r  Creating video: {total_frames}/{total_frames} (100.0%)")
    print(f"Video created successfully: {output_video_path.name}")

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETED!")
    print(f"{'=' * 60}")

    # Create log
    log_info_path = output_dir / "log_info.txt"
    with open(log_info_path, "w") as log_file:
        log_file.write("=" * 60 + "\n")
        log_file.write("Processing Log\n")
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
        log_file.write(f"Average FPS: {frame_count / execution_time:.2f}\n")
        log_file.write(f"Processing device: {device}\n\n")
        log_file.write("=" * 60 + "\n")
        log_file.write("PIPELINE CONFIGURATION:\n")
        log_file.write("=" * 60 + "\n")
        if yolo_model is not None and pose_config.get("use_yolo", False):
            yolo_model_name = pose_config.get("yolo_model", "yolo11x-pose.pt")
            yolo_mode = pose_config.get("yolo_mode", "yolo_mediapipe")
            if yolo_mode == "yolo_only":
                log_file.write("Pipeline: YOLOv11-pose only\n")
            else:
                log_file.write("Pipeline: YOLOv11 + MediaPipe\n")
            log_file.write(f"YOLO Model: {yolo_model_name} (loaded successfully)\n")
            log_file.write("YOLO Confidence: {}\n".format(pose_config.get("yolo_conf", 0.5)))
        else:
            log_file.write("Pipeline: MediaPipe only\n")
            log_file.write("YOLO: Not used\n")
        log_file.write("\n")
        log_file.write("MEDIAPIPE CONFIGURATION:\n")
        log_file.write("=" * 60 + "\n")
        for key, value in pose_config.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")
        if frames_with_missing_data:
            log_file.write(f"Frames with missing data: {len(frames_with_missing_data)}\n")
            log_file.write(
                f"Missing data percentage: {len(frames_with_missing_data) / frame_count * 100:.2f}%\n"
            )
        else:
            log_file.write("No frames with missing data.\n")

        # Add memory usage if GPU
        if device != "cpu":
            log_file.write(f"\nGPU Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n")

    print(f"\n{'=' * 60}")
    print("RESUMO DO PROCESSAMENTO")
    print(f"{'=' * 60}")
    print(f"Tempo total: {execution_time:.2f} segundos ({execution_time / 60:.1f} minutos)")
    print(f"Velocidade média: {frame_count / execution_time:.2f} fps")
    print(f"Frames processados: {frame_count}")
    print(f"Frames com pose: {frames_with_pose} ({frames_with_pose / frame_count * 100:.1f}%)")
    print(
        f"Frames sem pose: {frames_without_pose} ({frames_without_pose / frame_count * 100:.1f}%)"
    )
    print("\nArquivos salvos em:")
    print(f"  {output_dir}")
    print(f"  - {output_video_path.name}")
    print(f"  - {output_file_path.name}")
    print(f"  - {output_pixel_file_path.name}")
    print(f"  - {log_info_path.name}")
    print(f"{'=' * 60}")
    print("✓ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"{'=' * 60}\n")


def process_videos_in_directory(existing_root=None):
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

    # Use existing root or create new one for dialogs
    print("Initializing Tkinter root window...")
    if existing_root is not None:
        root = existing_root
        print("Using existing root.")
    else:
        try:
            root = tk.Tk()
            root.withdraw()
            print("Tkinter root initialized and withdrawn.")
            # Keep dialogs on top (as before)
            try:
                root.attributes("-topmost", True)
                print("Root attributes set to topmost.")
            except Exception as e:
                print(f"Warning: Could not set topmost attribute: {e}")
                pass
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Tkinter: {e}")
            print("This usually indicates an issue with the graphical environment.")
            import traceback

            traceback.print_exc()
            return

    # Helper function to prepare root window for dialogs
    # Fixes issue where dialogs appear in wrong position (bottom corner) on macOS
    # On Windows, root must be visible for dialogs to appear
    def prepare_root_for_dialog():
        try:
            # Make root visible and prepare it for dialogs on all platforms
            root.deiconify()
            root.update_idletasks()
            root.lift()
            root.focus_force()
            root.geometry("1x1+100+100")
            root.update_idletasks()

            # On Windows, ensure root stays visible and active
            if platform.system() == "Windows":
                try:
                    root.attributes("-topmost", True)
                    root.after(100, lambda: root.attributes("-topmost", False))
                except Exception:
                    pass

            root.update()
        except Exception as e:
            print(f"Warning: Could not prepare root for dialog: {e}")

    # Select input directory
    print("\nPlease select the input directory containing videos...")
    prepare_root_for_dialog()

    try:
        # Ensure root is active and ready for dialog (especially on Windows)
        root.update_idletasks()
        root.update()
        # Small delay to ensure window is ready (especially on Windows)
        root.after(50, lambda: None)
        root.update()

        input_dir = filedialog.askdirectory(
            parent=root, title="Select the input directory containing videos"
        )
        root.update_idletasks()
        root.update()
    except Exception as e:
        print(f"Error opening directory dialog: {e}")
        import traceback

        traceback.print_exc()
        try:
            messagebox.showerror("Error", f"Failed to open directory dialog: {e}")
        except Exception:
            pass
        return
    finally:
        # Only hide root on macOS after dialog closes
        # On Windows, keep root visible for subsequent dialogs
        if platform.system() == "Darwin" and existing_root is None:
            root.withdraw()
        elif platform.system() == "Windows":
            # On Windows, keep root visible but bring it to front
            try:
                root.lift()
                root.focus_force()
                root.update()
            except Exception:
                pass

    if not input_dir:
        print("No input directory selected.")
        try:
            messagebox.showerror("Error", "No input directory selected.")
        except Exception:
            pass
        return

    print(f"Input directory selected: {input_dir}")

    # Select output base directory
    print("\nPlease select the base output directory...")
    prepare_root_for_dialog()

    try:
        # Ensure root is active and ready for dialog (especially on Windows)
        root.update_idletasks()
        root.update()
        # Small delay to ensure window is ready (especially on Windows)
        root.after(50, lambda: None)
        root.update()

        output_base = filedialog.askdirectory(parent=root, title="Select the base output directory")
        root.update_idletasks()
        root.update()
    except Exception as e:
        print(f"Error opening output directory dialog: {e}")
        import traceback

        traceback.print_exc()
        try:
            messagebox.showerror("Error", f"Failed to open output directory dialog: {e}")
        except Exception:
            pass
        return
    finally:
        # Only hide root on macOS after dialog closes
        # On Windows, keep root visible for subsequent dialogs
        if platform.system() == "Darwin" and existing_root is None:
            root.withdraw()
        elif platform.system() == "Windows":
            # On Windows, keep root visible but bring it to front
            try:
                root.lift()
                root.focus_force()
                root.update()
            except Exception:
                pass

    if not output_base:
        print("No output directory selected.")
        try:
            messagebox.showerror("Error", "No output directory selected.")
        except Exception:
            pass
        return

    print(f"Output directory selected: {output_base}")

    # Get pose configuration
    print("\nOpening pose configuration dialog...")
    # Ensure root is visible for the configuration dialog (especially on Windows)
    try:
        root.deiconify()
        root.update_idletasks()
        root.lift()
        root.focus_force()
        root.update()
    except Exception as e:
        print(f"Warning: Could not prepare root for configuration dialog: {e}")

    pose_config = get_pose_config(root)

    # Hide root again after configuration (only if we created it)
    if existing_root is None and platform.system() != "Windows":
        # On Windows, keep root visible for subsequent dialogs if needed
        try:
            root.withdraw()
        except Exception:
            pass

    if not pose_config:
        print("Pose configuration cancelled or failed.")
        return

    print("Pose configuration completed successfully.")

    # Load YOLO model if necessary
    yolo_model = None
    use_yolo_successfully = False
    if pose_config["use_yolo"]:
        # Get the selected model name from config, default to yolo11x-pose.pt
        selected_model = pose_config.get("yolo_model", "yolo11x-pose.pt")
        print(f"Loading YOLO model: {selected_model}")
        yolo_model = download_or_load_yolo_model(selected_model)
        if yolo_model is not None:
            use_yolo_successfully = True
            print(
                f"✓ YOLO model '{selected_model}' loaded successfully - using YOLOv11 + MediaPipe pipeline"
            )
        else:
            print("Warning: Could not load YOLO model, proceeding without it")
            pose_config["use_yolo"] = False

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use different directory names based on whether YOLO is being used
    if use_yolo_successfully and pose_config.get("use_yolo", False):
        output_base = Path(output_base) / f"yolov_{timestamp}"
        print(f"Output directory: yolov_{timestamp} (YOLOv11 + MediaPipe)")
    else:
        output_base = Path(output_base) / f"mediapipe_{timestamp}"
        print(f"Output directory: mediapipe_{timestamp} (MediaPipe only)")
    output_base.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    video_files = list(input_dir.glob("*.*"))
    video_files = [f for f in video_files if f.suffix.lower() in [".mp4", ".avi", ".mov"]]

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
