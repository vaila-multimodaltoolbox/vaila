"""
Project: vailá
Script: yolov26track.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 February 2025
Update Date: 12 January 2026
Version: 0.0.7

Description:
    This script performs object detection and tracking on video files using the YOLO model v26.
    It integrates multiple features, including:
      - Object detection and tracking using the Ultralytics YOLO library.
      - A graphical interface (Tkinter) for dynamic parameter configuration.
      - Video processing with OpenCV, including drawing bounding boxes and overlaying tracking data.
      - Generation of CSV files containing frame-by-frame tracking information per tracker ID.
      - Video conversion to more compatible formats using FFmpeg.

Usage:
    Run the script from the command line by passing the path to a video file as an argument:
            python yolov26track.py

Requirements:
    - Python 3.x
    - OpenCV
    - PyTorch
    - Ultralytics (YOLO)
    - Tkinter (for GUI operations)
    - FFmpeg (for video conversion)
    - Additional dependencies as imported (numpy, csv, etc.)

License:
    This project is licensed under the terms of the AGPL-3.0 License.

Change History:
    - 2026-01: Added ROI selection with improved visibility on macOS.
    - 2023-10: Initial version implemented, integrating detection and tracking with various configurable options.
    - 2025-03: Added color-coding for each tracker ID, improved GUI, and added more detailed help text.
    - 2025-03: Added support for multiple models and trackers.
    - 2025-03: Added support for video conversion to more compatible formats using FFmpeg.

Notes:
    - Ensure that all dependencies are installed.
    - Since the script uses a graphical interface (Tkinter) for model selection and configuration, a GUI-enabled environment is required.
    - If the ReID model is not found, the script will download it from the internet.

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
Visit the project repository: https://github.com/vaila-multimodaltoolbox

"""

import colorsys
import contextlib
import csv
import datetime
import glob
import os
import platform
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd

try:
    import pkg_resources  # type: ignore[import-not-found]
except ImportError:
    pkg_resources = None  # setuptools not installed; use importlib.resources only
import torch
import ultralytics
import yaml
from rich import print
from ultralytics import YOLO

from .hardware_manager import HardwareManager

# Import PIL for image display
try:
    from PIL import Image, ImageTk
except ImportError:
    print("Warning: PIL (Pillow) not found. Installing...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image, ImageTk

# Import TOML for ROI configuration
try:
    import toml
except ImportError:
    print("Warning: toml not found. Installing...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
    import toml

# Print the script version and directory
print(f"Running script: {Path(__file__).name}")
print(f"Script directory: {Path(__file__).parent}")
print("Starting YOLOv26Track...")
print("-" * 80)
print(f"Ultralytics version: {ultralytics.__version__}")
print("-" * 80)


# Ensure BoxMOT can be found
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Configure to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)  # Limits the number of threads to avoid conflicts

# ReID model paths
REID_MODELS = {
    "lmbn_n_cuhk03_d.pt": "Lightweight (LMBN CUHK03)",
    "osnet_x0_25_market1501.pt": "Lightweight (OSNet x0.25 Market1501)",
    "mobilenetv2_x1_4_msmt17.engine": "Medium (MobileNetV2 MSMT17)",
    "resnet50_msmt17.onnx": "Medium (ResNet50 MSMT17)",
    "osnet_x1_0_msmt17.pt": "Medium (OSNet x1.0 MSMT17)",
    "clip_market1501.pt": "Heavy (CLIP Market1501)",
    "clip_vehicleid.pt": "Heavy (CLIP VehicleID)",
}


def initialize_csv(output_dir, label, tracker_id, total_frames):
    """Initializes a CSV file for a specific tracker ID and label."""
    csv_file = os.path.join(output_dir, f"{label}_id_{int(tracker_id):02d}.csv")
    if not os.path.exists(csv_file):
        # Get color for this ID
        color = get_color_for_id(tracker_id)
        color_r, color_g, color_b = color[2], color[1], color[0]  # BGR to RGB

        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Frame",
                    "Tracker ID",
                    "Label",
                    "X_min",
                    "Y_min",
                    "X_max",
                    "Y_max",
                    "Confidence",
                    "Color_R",
                    "Color_G",
                    "Color_B",
                ]
            )
            for frame_idx in range(total_frames):
                writer.writerow(
                    [
                        frame_idx,
                        tracker_id,
                        label,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        color_r,
                        color_g,
                        color_b,
                    ]
                )
    return csv_file


def update_csv(csv_file, frame_idx, tracker_id, label, x_min, y_min, x_max, y_max, conf):
    """Updates the CSV file with detection data for the specific frame."""
    rows = []
    with open(csv_file, newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Get color info from first data row (already saved in CSV)
    if len(rows) > 1:
        color_r, color_g, color_b = rows[1][-3], rows[1][-2], rows[1][-1]
    else:
        # Fallback if color info isn't in CSV yet
        color = get_color_for_id(tracker_id)
        color_r, color_g, color_b = color[2], color[1], color[0]  # BGR to RGB

    # Update the specific frame line
    for i, row in enumerate(rows):
        if i > 0 and int(row[0]) == frame_idx:  # Skip the header
            rows[i] = [
                frame_idx,
                tracker_id,
                label,
                x_min,
                y_min,
                x_max,
                y_max,
                conf,
                color_r,
                color_g,
                color_b,
            ]
            break

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def get_hardware_info():
    """Get detailed hardware information for GPU/CPU/MPS detection"""
    info = []
    info.append(f"Python version: {sys.version}")
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

    # Check for MPS (Metal Performance Shaders) on macOS Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info.append("MPS (Metal) available: Yes")
        else:
            info.append("MPS (Metal) available: No")

        info.append(f"CPU cores: {os.cpu_count()}")

    return "\n".join(info)


def detect_optimal_device():
    """Detect and return the optimal device for processing"""
    # Check for CUDA first (NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Multiple GPUs detected ({gpu_count}). GPU 0 available for selection.")
        torch.cuda.empty_cache()
        return "cuda"
    # Check for MPS (Metal Performance Shaders) on macOS Apple Silicon
    elif platform.system() == "Darwin" and platform.machine() == "arm64":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    # Default to CPU
    return "cpu"


def validate_device_choice(user_device):
    """Validate user device choice and provide feedback"""
    device_lower = user_device.lower()
    if device_lower == "cuda":
        if torch.cuda.is_available():
            return True, "GPU (CUDA) - High performance"
        else:
            return False, "GPU (CUDA) requested but not available. Using CPU instead."
    elif device_lower == "mps":
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return True, "GPU (MPS/Metal) - Apple Silicon acceleration"
            else:
                return False, "MPS requested but not available. Using CPU instead."
        else:
            return False, "MPS only available on macOS Apple Silicon. Using CPU instead."
    elif device_lower == "cpu":
        return True, "CPU - Universal compatibility (default)"
    else:
        return False, f"Invalid device: {user_device}. Using CPU instead."


def save_roi_to_toml(video_path, roi_poly):
    """
    Save ROI polygon to a TOML file.
    Returns the path to the saved file, or None on error.
    """
    try:
        # Create ROI directory in the same location as the video
        video_dir = os.path.dirname(os.path.abspath(video_path))
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        roi_dir = os.path.join(video_dir, "roi_configs")
        os.makedirs(roi_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        roi_file = os.path.join(roi_dir, f"{video_name}_roi_{timestamp}.toml")

        # Convert ROI points to list of [x, y] pairs
        roi_points = [[int(pt[0]), int(pt[1])] for pt in roi_poly]

        # Get video dimensions for reference
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create TOML content
        roi_data = {
            "roi_info": {
                "video_path": str(Path(video_path).as_posix()),
                "video_name": video_name,
                "video_width": width,
                "video_height": height,
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_points": len(roi_points),
            },
            "roi_polygon": {"points": roi_points},
        }

        # Write to TOML file
        with open(roi_file, "w") as f:
            toml.dump(roi_data, f)

        print(f"ROI saved to: {roi_file}")
        return roi_file

    except Exception as e:
        print(f"Error saving ROI to TOML: {e}")
        import traceback

        traceback.print_exc()
        return None


def load_roi_from_toml(roi_file_path):
    """
    Load ROI polygon from a TOML file.
    Returns a numpy array of int32 points, or None on error.
    """
    try:
        if not os.path.exists(roi_file_path):
            print(f"ROI file not found: {roi_file_path}")
            return None

        with open(roi_file_path) as f:
            roi_data = toml.load(f)

        if "roi_polygon" not in roi_data or "points" not in roi_data["roi_polygon"]:
            print("Invalid ROI file format: missing polygon points")
            return None

        points = roi_data["roi_polygon"]["points"]
        if len(points) < 3:
            print(f"Invalid ROI: need at least 3 points, got {len(points)}")
            return None

        # Convert to numpy array
        roi_poly = np.array(points, dtype=np.int32)

        print(f"ROI loaded from: {roi_file_path}")
        print(f"  Points: {len(points)}")
        if "roi_info" in roi_data:
            info = roi_data["roi_info"]
            print(f"  Original video: {info.get('video_name', 'unknown')}")
            print(
                f"  Video dimensions: {info.get('video_width', '?')}x{info.get('video_height', '?')}"
            )

        return roi_poly

    except Exception as e:
        print(f"Error loading ROI from TOML: {e}")
        import traceback

        traceback.print_exc()
        return None


def select_free_polygon_roi(video_path):
    """
    Let the user draw a free polygon ROI on the first frame of the video.
    Left click adds points, right click removes the last point, Enter confirms,
    Esc skips, and 'r' resets. Returns a numpy array of int32 points or None.
    """
    # #region agent log
    log_file = "/Users/preto/Desktop/Preto/vaila/.cursor/debug.log"
    import json

    def log_debug(location, message, data, hypothesis_id):
        try:
            with open(log_file, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": hypothesis_id,
                            "location": location,
                            "message": message,
                            "data": data,
                            "timestamp": int(__import__("time").time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    # #endregion agent log

    cap = None
    window_name = None
    try:
        # #region agent log
        log_debug(
            "yolov26track.py:381", "ROI selection started", {"video_path": str(video_path)}, "A"
        )
        # #endregion agent log

        # Extract first frame from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None

        # Read first frame
        ret, frame = cap.read()
        cap.release()
        cap = None

        if not ret or frame is None:
            print("Error: Could not read first frame from video for ROI selection.")
            return None

        # #region agent log
        h_orig, w_orig = frame.shape[:2]
        log_debug(
            "yolov26track.py:395",
            "Original frame dimensions",
            {"width": w_orig, "height": h_orig},
            "A",
        )
        # #endregion agent log

        # Scale frame to reasonable size for display (much larger window for better visibility on macOS)
        scale = 1.0
        h, w = frame.shape[:2]
        # Use much larger max dimensions for better visibility on macOS (allows seeing all controls)
        max_h = 1800  # Increased from 1200
        max_w = 2400  # Increased from 1600
        # Scale down if too large, but keep aspect ratio
        if h > max_h or w > max_w:
            scale_h = max_h / h if h > max_h else 1.0
            scale_w = max_w / w if w > max_w else 1.0
            scale = min(scale_h, scale_w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # #region agent log
        h_scaled, w_scaled = frame.shape[:2]
        log_debug(
            "yolov26track.py:428",
            "After scaling",
            {"width": w_scaled, "height": h_scaled, "scale": scale},
            "A",
        )
        # #endregion agent log

        roi_points = []
        mouse_clicked = False  # Flag to provide visual feedback

        def mouse_callback(event, x, y, flags, param):
            nonlocal mouse_clicked
            # #region agent log
            log_debug(
                "yolov26track.py:437",
                "Mouse event",
                {"event": event, "x": x, "y": y, "flags": flags},
                "D",
            )
            # #endregion agent log
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
                mouse_clicked = True
                print(f"Point added: ({x}, {y}) - Total points: {len(roi_points)}")
                # #region agent log
                log_debug(
                    "yolov26track.py:443",
                    "Point added",
                    {"point": (x, y), "total_points": len(roi_points)},
                    "D",
                )
                # #endregion agent log
            elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
                removed = roi_points.pop()
                mouse_clicked = True
                print(f"Point removed: {removed} - Total points: {len(roi_points)}")
                # #region agent log
                log_debug(
                    "yolov26track.py:449",
                    "Point removed",
                    {"removed": removed, "total_points": len(roi_points)},
                    "D",
                )
                # #endregion agent log

        window_name = "Select ROI (Left: add, Right: undo, Enter: confirm, Esc: skip, r: reset)"

        # #region agent log
        log_debug(
            "yolov26track.py:455",
            "Creating window",
            {"window_name": window_name, "frame_size": (w_scaled, h_scaled)},
            "B",
        )
        # #endregion agent log

        # Create window with WINDOW_NORMAL flag for resizability
        # On macOS, use WINDOW_GUI_NORMAL for better window management
        if platform.system() == "Darwin":
            window_flags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO
        else:
            window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO

        cv2.namedWindow(window_name, window_flags)

        # Set mouse callback BEFORE showing window (important for macOS)
        cv2.setMouseCallback(window_name, mouse_callback)

        # #region agent log
        log_debug("yolov26track.py:473", "Mouse callback set", {}, "D")
        # #endregion agent log

        # Show window first to ensure it exists
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)  # Force window to appear

        # Set window size after showing (important for macOS resizing)
        try:
            # Calculate desired window size - make it larger to see all controls
            desired_w = max(1200, min(w_scaled, 2400))  # Increased from 800-1600
            desired_h = max(900, min(h_scaled, 1800))  # Increased from 600-1200

            # Resize window - try multiple times on macOS if needed
            cv2.resizeWindow(window_name, desired_w, desired_h)

            # On macOS, sometimes need to call resize multiple times
            if platform.system() == "Darwin":
                cv2.waitKey(10)  # Small delay
                cv2.resizeWindow(window_name, desired_w, desired_h)
                cv2.waitKey(10)  # Another small delay

            # #region agent log
            log_debug(
                "yolov26track.py:464",
                "Window resize attempted",
                {"width": desired_w, "height": desired_h},
                "B",
            )
            # #endregion agent log
        except Exception as e:
            # #region agent log
            log_debug("yolov26track.py:467", "Window resize failed", {"error": str(e)}, "B")
            # #endregion agent log
            pass

        # Colors optimized for visibility in both light and dark mode
        # Use bright cyan for polygon lines (high contrast)
        polygon_color = (255, 255, 0)  # Cyan (BGR) - bright and visible
        # Use bright yellow for points (high contrast)
        point_color = (0, 255, 255)  # Yellow (BGR) - bright and visible
        # Use bright magenta for closing line
        closing_line_color = (255, 0, 255)  # Magenta (BGR) - bright and visible

        # Add help text overlay (shown on first frame)
        help_text = [
            "Left Click: Add point",
            "Right Click: Remove last point",
            "Enter: Confirm selection",
            "Esc: Cancel",
            "R: Reset all points",
        ]

        loop_count = 0
        while True:
            display_img = frame.copy()

            # Draw help text (fade out after first few frames or when points are added)
            if loop_count < 100 or len(roi_points) == 0:
                y_offset = 10
                for i, text in enumerate(help_text):
                    y_pos = y_offset + i * 25
                    # White text with black outline for visibility
                    cv2.putText(
                        display_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3
                    )
                    cv2.putText(
                        display_img,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            if roi_points:
                pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
                # Draw polygon with thicker, brighter lines
                cv2.polylines(display_img, [pts], False, polygon_color, 3)
                # Draw points with larger, brighter circles
                for pt in roi_points:
                    cv2.circle(display_img, pt, 5, point_color, -1)
                    cv2.circle(display_img, pt, 5, (0, 0, 0), 1)  # Black outline for contrast
                # Draw closing line if more than 1 point
                if len(roi_points) > 1:
                    cv2.line(display_img, roi_points[-1], roi_points[0], closing_line_color, 2)
                # Show point count on image (bottom left)
                point_text = f"Points: {len(roi_points)} (min 3 required)"
                text_y = display_img.shape[0] - 20
                # White text with black outline for visibility
                cv2.putText(
                    display_img,
                    point_text,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    display_img,
                    point_text,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            # Visual feedback for mouse clicks
            if mouse_clicked:
                mouse_clicked = False
                # Brief flash effect could be added here if needed

            cv2.imshow(window_name, display_img)
            # Increased waitKey time for better event processing on macOS
            key = cv2.waitKey(30) & 0xFF

            # #region agent log
            if loop_count % 50 == 0:  # Log every 50 iterations to avoid spam
                log_debug(
                    "yolov26track.py:512",
                    "Loop iteration",
                    {"loop_count": loop_count, "key": key, "roi_points_count": len(roi_points)},
                    "C",
                )
            loop_count += 1
            # #endregion agent log

            if key == 13 or key == 10:  # Enter (13) or Return (10) - both work
                # #region agent log
                log_debug(
                    "yolov26track.py:518",
                    "Enter pressed - confirming",
                    {"roi_points_count": len(roi_points)},
                    "C",
                )
                # #endregion agent log
                if len(roi_points) >= 3:
                    print(f"ROI confirmed with {len(roi_points)} points")
                    break
                else:
                    print(f"Need at least 3 points. Currently have {len(roi_points)}")
            elif key == 27:  # Esc
                # #region agent log
                log_debug("yolov26track.py:526", "Esc pressed - cancelling", {}, "C")
                # #endregion agent log
                print("ROI selection cancelled")
                roi_points = []
                break
            elif key == ord("r") or key == ord("R"):  # Reset (case insensitive)
                # #region agent log
                log_debug("yolov26track.py:531", "R pressed - resetting", {}, "C")
                # #endregion agent log
                print("ROI points reset")
                roi_points = []

        if window_name:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)  # Ensure window is destroyed

        if len(roi_points) < 3:
            # #region agent log
            log_debug(
                "yolov26track.py:541",
                "Insufficient points",
                {"roi_points_count": len(roi_points)},
                "D",
            )
            # #endregion agent log
            print("ROI selection requires at least 3 points.")
            return None

        # Scale points back to original resolution
        final_points = (np.array(roi_points, dtype=np.float32) / scale).astype(np.int32)
        # #region agent log
        log_debug(
            "yolov26track.py:549",
            "ROI selection completed",
            {"final_points_count": len(final_points), "scale": scale},
            "A",
        )
        # #endregion agent log
        print(f"ROI selection completed with {len(final_points)} points")
        return final_points

    except Exception as e:
        # #region agent log
        log_debug(
            "yolov26track.py:491",
            "Exception in ROI selection",
            {"error": str(e), "error_type": type(e).__name__},
            "A",
        )
        # #endregion agent log
        print(f"Error in ROI selection: {e}")
        import traceback

        traceback.print_exc()
        # Clean up resources
        if cap is not None:
            cap.release()
        if window_name:
            try:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
            except Exception:
                pass
        cv2.destroyAllWindows()
        return None


def select_bbox_roi(video_path):
    """
    Let the user draw a bounding box (rectangle) ROI on the first frame of the video.
    Click and drag to draw rectangle, Enter confirms, Esc cancels.
    Returns a numpy array of int32 points (4 corners of rectangle) or None.
    """
    cap = None
    window_name = None
    try:
        # Extract first frame from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None

        # Read first frame
        ret, frame = cap.read()
        cap.release()
        cap = None

        if not ret or frame is None:
            print("Error: Could not read first frame from video for ROI selection.")
            return None

        h_orig, w_orig = frame.shape[:2]

        # Scale frame to reasonable size for display (larger window for better visibility)
        scale = 1.0
        h, w = frame.shape[:2]
        # Use larger max dimensions for better visibility on macOS
        max_h = 1800
        max_w = 2400
        # Scale down if too large, but keep aspect ratio
        if h > max_h or w > max_w:
            scale_h = max_h / h if h > max_h else 1.0
            scale_w = max_w / w if w > max_w else 1.0
            scale = min(scale_h, scale_w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        h_scaled, w_scaled = frame.shape[:2]

        # Variables for rectangle drawing
        drawing = False
        start_point = None
        end_point = None
        bbox_rect = None

        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, end_point, bbox_rect

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
                end_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    end_point = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                # Ensure we have a valid rectangle
                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    # Normalize to top-left and bottom-right
                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1, x2)
                    y_max = max(y1, y2)
                    bbox_rect = (x_min, y_min, x_max, y_max)
                    print(f"BBox selected: ({x_min}, {y_min}) to ({x_max}, {y_max})")

        window_name = "Select ROI BBox (Click & Drag: draw, Enter: confirm, Esc: cancel)"

        # Create window with WINDOW_NORMAL flag for resizability
        if platform.system() == "Darwin":
            window_flags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO
        else:
            window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO

        cv2.namedWindow(window_name, window_flags)

        # Set mouse callback BEFORE showing window
        cv2.setMouseCallback(window_name, mouse_callback)

        # Show window first
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

        # Set window size after showing (larger for macOS)
        try:
            desired_w = max(1200, min(w_scaled, 2400))
            desired_h = max(900, min(h_scaled, 1800))
            cv2.resizeWindow(window_name, desired_w, desired_h)
            if platform.system() == "Darwin":
                cv2.waitKey(10)
                cv2.resizeWindow(window_name, desired_w, desired_h)
                cv2.waitKey(10)
        except Exception:
            pass

        # Colors for rectangle
        rect_color = (0, 255, 0)  # Green (BGR)
        help_text_lines = [
            "Click and Drag: Draw bounding box",
            "Enter: Confirm selection",
            "Esc: Cancel",
        ]

        loop_count = 0
        while True:
            display_img = frame.copy()

            # Draw help text
            if loop_count < 100 or bbox_rect is None:
                y_offset = 10
                for i, text in enumerate(help_text_lines):
                    y_pos = y_offset + i * 25
                    cv2.putText(
                        display_img, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3
                    )
                    cv2.putText(
                        display_img,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            # Draw rectangle if drawing or if bbox is set
            if drawing and start_point and end_point:
                x1, y1 = start_point
                x2, y2 = end_point
                cv2.rectangle(display_img, (x1, y1), (x2, y2), rect_color, 2)
            elif bbox_rect:
                x_min, y_min, x_max, y_max = bbox_rect
                cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), rect_color, 3)
                # Show dimensions
                width = x_max - x_min
                height = y_max - y_min
                info_text = f"BBox: {width}x{height} pixels"
                text_y = display_img.shape[0] - 20
                cv2.putText(
                    display_img,
                    info_text,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    display_img,
                    info_text,
                    (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(30) & 0xFF

            loop_count += 1

            if key == 13 or key == 10:  # Enter
                if bbox_rect:
                    print(f"BBox ROI confirmed: {bbox_rect}")
                    break
                else:
                    print("No bounding box drawn. Please draw a rectangle first.")
            elif key == 27:  # Esc
                print("BBox ROI selection cancelled")
                bbox_rect = None
                break

        if window_name:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        if bbox_rect is None:
            print("BBox ROI selection cancelled or no rectangle drawn.")
            return None

        # Convert bbox to polygon (4 corners) and scale back to original resolution
        x_min, y_min, x_max, y_max = bbox_rect
        # Convert to original resolution
        x_min_orig = int(x_min / scale)
        y_min_orig = int(y_min / scale)
        x_max_orig = int(x_max / scale)
        y_max_orig = int(y_max / scale)

        # Create polygon from rectangle (4 corners)
        final_points = np.array(
            [
                [x_min_orig, y_min_orig],
                [x_max_orig, y_min_orig],
                [x_max_orig, y_max_orig],
                [x_min_orig, y_max_orig],
            ],
            dtype=np.int32,
        )

        print(f"BBox ROI selection completed: {len(final_points)} points (rectangle)")
        return final_points

    except Exception as e:
        print(f"Error in BBox ROI selection: {e}")
        import traceback

        traceback.print_exc()
        # Clean up resources
        if cap is not None:
            cap.release()
        if window_name:
            try:
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
            except Exception:
                pass
        cv2.destroyAllWindows()
        return None


class TrackerConfigDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None):
        self.tooltip = None
        self.hardware_info = get_hardware_info()
        self.optimal_device = detect_optimal_device()
        super().__init__(parent, title)

    def body(self, master):
        # Hardware Info Frame
        hw_frame = tk.LabelFrame(master, text="Hardware Information", padx=5, pady=5)
        hw_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        hw_text = tk.Text(hw_frame, height=6, width=60, wrap="word", font=("Consolas", 8))
        hw_text.insert(tk.END, self.hardware_info)
        hw_text.config(state="disabled")
        hw_text.pack(padx=5, pady=5)

        # Device Selection Frame
        device_frame = tk.LabelFrame(master, text="Device Selection", padx=5, pady=5)
        device_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Device choice
        tk.Label(device_frame, text="Processing Device:").grid(row=0, column=0, padx=5, pady=5)

        # Build device options based on available hardware
        device_options = ["cpu"]  # CPU is always available
        if torch.cuda.is_available():
            device_options.append("cuda")
        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device_options.append("mps")

        # Set default device to optimal device
        default_device = self.optimal_device if self.optimal_device in device_options else "cpu"
        self.device_var = tk.StringVar(value=default_device)
        device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.device_var,
            values=device_options,
            state="readonly",
            width=10,
        )
        device_combo.grid(row=0, column=1, padx=5, pady=5)

        # Device status
        self.device_status = tk.Label(device_frame, text="", fg="green")
        self.device_status.grid(row=0, column=2, padx=5, pady=5)

        # Update status when device changes
        def update_device_status(*args):
            device = self.device_var.get()
            is_valid, message = validate_device_choice(device)
            if is_valid:
                if device == "cpu":
                    self.device_status.config(text="[OK] " + message, fg="blue")  # Blue for CPU
                elif device == "mps":
                    self.device_status.config(text="[OK] " + message, fg="purple")  # Purple for MPS
                else:  # cuda
                    self.device_status.config(text="[OK] " + message, fg="green")  # Green for CUDA
            else:
                self.device_status.config(text="[WARNING] " + message, fg="orange")

        self.device_var.trace_add("write", update_device_status)
        update_device_status()  # Initial update

        # Help text for device selection
        help_text = tk.Label(device_frame, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=0, column=3, padx=5, pady=5)
        device_tooltip = (
            "Processing device options:\n"
            "'cpu'  - Use CPU\n"
            "        Universal compatibility\n"
            "        Works on all computers\n"
            "        Slower but reliable\n\n"
            "'cuda' - Use GPU (NVIDIA only)\n"
            "        Much faster processing (10-20x)\n"
            "        Requires NVIDIA GPU and CUDA\n"
            "        May have compatibility issues\n\n"
            "'mps'  - Use GPU (Apple Silicon only)\n"
            "        Fast processing via Metal Performance Shaders\n"
            "        Requires macOS on Apple Silicon (M1/M2/M3)\n"
            "        Recommended for Apple Silicon Macs\n\n"
            "CPU is recommended for most users."
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, device_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        # Confidence
        tk.Label(master, text="Confidence threshold:").grid(row=2, column=0, padx=5, pady=5)
        self.conf = tk.Entry(master)
        self.conf.insert(0, "0.15")
        self.conf.grid(row=2, column=1, padx=5, pady=5)
        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=2, column=2, padx=5, pady=5)
        conf_tooltip = (
            "Confidence threshold (0-1):\n"
            "Controls how confident the model must be to detect an object.\n"
            "Higher values (e.g., 0.7): Fewer but more accurate detections\n"
            "Lower values (e.g., 0.25): More detections but may include false positives\n"
            "Recommended: 0.25-0.5 for tracking"
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, conf_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        # IoU
        tk.Label(master, text="IoU threshold:").grid(row=3, column=0, padx=5, pady=5)
        self.iou = tk.Entry(master)
        self.iou.insert(0, "0.7")
        self.iou.grid(row=3, column=1, padx=5, pady=5)
        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=3, column=2, padx=5, pady=5)
        iou_tooltip = (
            "Intersection over Union threshold (0-1):\n"
            "Controls how much overlap is needed to merge multiple detections.\n"
            "Higher values (e.g., 0.9): Very strict matching\n"
            "Lower values (e.g., 0.5): More lenient matching\n"
            "Recommended: 0.7 for most cases"
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, iou_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        # Video stride
        tk.Label(master, text="Video stride:").grid(row=4, column=0, padx=5, pady=5)
        self.vid_stride = tk.Entry(master)
        self.vid_stride.insert(0, "1")
        self.vid_stride.grid(row=4, column=1, padx=5, pady=5)
        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=4, column=2, padx=5, pady=5)
        stride_tooltip = (
            "Video stride (frames to skip):\n"
            "1 = Process every frame\n"
            "2 = Process every other frame\n"
            "3 = Process every third frame\n\n"
            "Higher values = Faster processing\n"
            "Lower values = Better tracking accuracy\n"
            "Recommended: 1 for accurate tracking\n"
            "            2-3 for faster processing"
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, stride_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        # ROI selection section
        tk.Label(master, text="Region of Interest (ROI):").grid(row=5, column=0, padx=5, pady=5)
        self.roi_file_path = None
        self.roi_status_label = tk.Label(master, text="No ROI selected", fg="gray")
        self.roi_status_label.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Frame for ROI buttons
        roi_buttons_frame = tk.Frame(master)
        roi_buttons_frame.grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        btn_create_roi_poly = tk.Button(
            roi_buttons_frame,
            text="Create Polygon ROI",
            command=self.select_roi_from_video,
            bg="#4CAF50",
            fg="black",
            width=18,
        )
        btn_create_roi_poly.pack(side="left", padx=5)

        btn_create_roi_bbox = tk.Button(
            roi_buttons_frame,
            text="Create BBox ROI",
            command=self.select_bbox_roi_from_video,
            bg="#FF9800",
            fg="black",
            width=18,
        )
        btn_create_roi_bbox.pack(side="left", padx=5)

        btn_load_roi = tk.Button(
            roi_buttons_frame,
            text="Load Existing ROI",
            command=self.load_existing_roi,
            bg="#2196F3",
            fg="black",
            width=18,
        )
        btn_load_roi.pack(side="left", padx=5)

        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=5, column=2, padx=5, pady=5)
        roi_tooltip = (
            "ROI Options:\n"
            "'Create Polygon ROI' - Draw a free polygon on a video frame\n"
            "'Create BBox ROI' - Draw a rectangle (bounding box) on a video frame\n"
            "'Load Existing ROI' - Load a previously saved ROI from file\n\n"
            "The ROI will be applied to all videos in the batch.\n"
            "Tracking and detection will only run inside the selected area."
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, roi_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        return self.conf

    def show_help(self, event, text):
        # Hide any existing tooltip first
        self.hide_help()

        x = event.widget.winfo_rootx() + event.widget.winfo_width()
        y = event.widget.winfo_rooty()

        self.tooltip = tk.Toplevel()
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip,
            text=text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padx=5,
            pady=5,
        )
        label.pack()

    def hide_help(self, event=None):
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None

    def select_roi_from_video(self):
        """Open file dialog to select a video, then open polygon ROI selection window"""
        # Ask user to select a video file for ROI selection
        video_path = filedialog.askopenfilename(
            title="Select a video file for ROI selection",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
                ("All files", "*.*"),
            ],
        )

        if not video_path:
            return

        try:
            # Open polygon ROI selection window
            roi_poly = select_free_polygon_roi(video_path)

            if roi_poly is not None and len(roi_poly) >= 3:
                # Save ROI to TOML file
                roi_file = save_roi_to_toml(video_path, roi_poly)
                if roi_file:
                    self.roi_file_path = roi_file
                    self.roi_status_label.config(
                        text=f"ROI saved: {os.path.basename(roi_file)}", fg="green"
                    )
                    messagebox.showinfo(
                        "ROI Saved",
                        f"ROI polygon saved successfully!\n\n"
                        f"File: {os.path.basename(roi_file)}\n"
                        f"Points: {len(roi_poly)}\n\n"
                        f"This ROI will be applied to all videos in the batch.",
                    )
                else:
                    self.roi_status_label.config(text="Error saving ROI", fg="red")
            else:
                self.roi_status_label.config(text="ROI selection cancelled", fg="gray")
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting ROI: {str(e)}")
            self.roi_status_label.config(text="Error selecting ROI", fg="red")

    def select_bbox_roi_from_video(self):
        """Open file dialog to select a video, then open bbox ROI selection window"""
        # Ask user to select a video file for ROI selection
        video_path = filedialog.askopenfilename(
            title="Select a video file for ROI selection",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
                ("All files", "*.*"),
            ],
        )

        if not video_path:
            return

        try:
            # Open bbox ROI selection window
            roi_poly = select_bbox_roi(video_path)

            if roi_poly is not None and len(roi_poly) >= 3:
                # Save ROI to TOML file
                roi_file = save_roi_to_toml(video_path, roi_poly)
                if roi_file:
                    self.roi_file_path = roi_file
                    self.roi_status_label.config(
                        text=f"ROI saved: {os.path.basename(roi_file)}", fg="green"
                    )
                    messagebox.showinfo(
                        "ROI Saved",
                        f"ROI bounding box saved successfully!\n\n"
                        f"File: {os.path.basename(roi_file)}\n"
                        f"Points: {len(roi_poly)} (rectangle)\n\n"
                        f"This ROI will be applied to all videos in the batch.",
                    )
                else:
                    self.roi_status_label.config(text="Error saving ROI", fg="red")
            else:
                self.roi_status_label.config(text="ROI selection cancelled", fg="gray")
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting ROI: {str(e)}")
            self.roi_status_label.config(text="Error selecting ROI", fg="red")

    def load_existing_roi(self):
        """Open file dialog to load an existing ROI from TOML file"""
        roi_file_path = filedialog.askopenfilename(
            title="Select ROI configuration file",
            filetypes=[
                ("TOML files", "*.toml"),
                ("All files", "*.*"),
            ],
        )

        if not roi_file_path:
            return

        try:
            # Load ROI from TOML file
            roi_poly = load_roi_from_toml(roi_file_path)

            if roi_poly is not None and len(roi_poly) >= 3:
                self.roi_file_path = roi_file_path
                self.roi_status_label.config(
                    text=f"ROI loaded: {os.path.basename(roi_file_path)}", fg="green"
                )
                messagebox.showinfo(
                    "ROI Loaded",
                    f"ROI polygon loaded successfully!\n\n"
                    f"File: {os.path.basename(roi_file_path)}\n"
                    f"Points: {len(roi_poly)}\n\n"
                    f"This ROI will be applied to all videos in the batch.",
                )
            else:
                self.roi_status_label.config(text="Invalid ROI file", fg="red")
                messagebox.showerror(
                    "Error", "The selected ROI file is invalid or contains less than 3 points."
                )
        except Exception as e:
            messagebox.showerror("Error", f"Error loading ROI: {str(e)}")
            self.roi_status_label.config(text="Error loading ROI", fg="red")

    def validate(self):
        try:
            # Validate device choice
            device = self.device_var.get()
            is_valid, message = validate_device_choice(device)

            if not is_valid:
                messagebox.showwarning("Device Warning", message)
                device = "cpu"  # Fallback to CPU

            self.result = {
                "conf": float(self.conf.get()),
                "iou": float(self.iou.get()),
                "device": device,
                "vid_stride": int(self.vid_stride.get()),
                "roi_file": self.roi_file_path,  # Path to saved ROI TOML file
                "half": True,
                "persist": True,
                "verbose": False,
                "stream": True,
            }
            return True
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values")
            return False

    def apply(self):
        self.result = self.result


class ModelSelectorDialog(simpledialog.Dialog):
    def body(self, master):
        # Create main frame
        main_frame = tk.Frame(master)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Title
        title_label = tk.Label(main_frame, text="Select YOLO Model", font=("Arial", 12, "bold"))
        title_label.pack(pady=(0, 10))

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True)

        # Tab 1: Pre-trained models
        pretrained_frame = tk.Frame(notebook)
        notebook.add(pretrained_frame, text="Pre-trained Models")

        # Pre-trained models list
        models = [
            # Object Detection explanation: https://docs.ultralytics.com/tasks/detect/
            ("yolo26n.pt", "Detection - Nano (fastest)"),
            ("yolo26s.pt", "Detection - Small"),
            ("yolo26m.pt", "Detection - Medium"),
            ("yolo26l.pt", "Detection - Large"),
            ("yolo26x.pt", "Detection - XLarge (most accurate)"),
            # Pose Estimation explanation: https://docs.ultralytics.com/tasks/pose/
            ("yolo26n-pose.pt", "Pose - Nano (fastest)"),
            ("yolo26s-pose.pt", "Pose - Small"),
            ("yolo26m-pose.pt", "Pose - Medium"),
            ("yolo26l-pose.pt", "Pose - Large"),
            ("yolo26x-pose.pt", "Pose - XLarge (most accurate)"),
            # Segmentation explanation: https://docs.ultralytics.com/tasks/segment/
            ("yolo26n-seg.pt", "Segmentation - Nano"),
            ("yolo26s-seg.pt", "Segmentation - Small"),
            ("yolo26m-seg.pt", "Segmentation - Medium"),
            ("yolo26l-seg.pt", "Segmentation - Large"),
            ("yolo26x-seg.pt", "Segmentation - XLarge"),
            # OBB (Oriented Bounding Box) explanation: https://docs.ultralytics.com/tasks/obb/
            ("yolo26n-obb.pt", "OBB - Nano"),
            ("yolo26s-obb.pt", "OBB - Small"),
            ("yolo26m-obb.pt", "OBB - Medium"),
            ("yolo26l-obb.pt", "OBB - Large"),
            ("yolo26x-obb.pt", "OBB - XLarge"),
        ]

        # Create listbox with scrollbar for pre-trained models
        listbox_frame = tk.Frame(pretrained_frame)
        listbox_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.pretrained_listbox = tk.Listbox(listbox_frame, width=50, height=12)
        scrollbar = tk.Scrollbar(
            listbox_frame, orient="vertical", command=self.pretrained_listbox.yview
        )
        self.pretrained_listbox.configure(yscrollcommand=scrollbar.set)

        for model, desc in models:
            self.pretrained_listbox.insert(tk.END, f"{model} - {desc}")

        self.pretrained_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Tab 2: Custom model
        custom_frame = tk.Frame(notebook)
        notebook.add(custom_frame, text="Custom Model")

        # Custom model selection
        custom_label = tk.Label(custom_frame, text="Select custom model file:", font=("Arial", 10))
        custom_label.pack(pady=(10, 5))

        # Frame for path display and browse button
        path_frame = tk.Frame(custom_frame)
        path_frame.pack(fill="x", padx=10, pady=5)

        self.custom_path_var = tk.StringVar()
        self.custom_path_entry = tk.Entry(path_frame, textvariable=self.custom_path_var, width=50)
        self.custom_path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        browse_button = tk.Button(path_frame, text="Browse", command=self.browse_custom_model)
        browse_button.pack(side="right")

        # Help text for custom models
        help_text = tk.Label(
            custom_frame,
            text="Supported formats: .pt, .onnx, .engine\n"
            "Custom models should be trained with YOLO format\n"
            "Make sure the model file exists and is accessible",
            justify="left",
            font=("Arial", 9),
            fg="gray",
        )
        help_text.pack(pady=10)

        # Store the selected tab
        self.selected_tab = "pretrained"
        self.custom_model_path = None

        # Bind tab change event
        notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        return self.pretrained_listbox

    def browse_custom_model(self):
        """Browse for custom model file."""
        file_path = filedialog.askopenfilename(
            title="Select Custom Model File",
            filetypes=[
                ("YOLO models", "*.pt"),
                ("ONNX models", "*.onnx"),
                ("TensorRT models", "*.engine"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.custom_path_var.set(file_path)
            self.custom_model_path = file_path

    def on_tab_changed(self, event):
        """Handle tab change events."""
        notebook = event.widget
        current_tab = notebook.select()
        tab_id = notebook.index(current_tab)

        if tab_id == 0:  # Pre-trained models tab
            self.selected_tab = "pretrained"
        elif tab_id == 1:  # Custom model tab
            self.selected_tab = "custom"

    def validate(self):
        if self.selected_tab == "pretrained":
            if not self.pretrained_listbox.curselection():
                messagebox.showwarning("Warning", "Please select a pre-trained model")
                return False
        elif self.selected_tab == "custom":
            custom_path = self.custom_path_var.get().strip()
            if not custom_path:
                messagebox.showwarning("Warning", "Please select a custom model file")
                return False
            if not os.path.exists(custom_path):
                messagebox.showerror("Error", f"Custom model file not found: {custom_path}")
                return False
        return True

    def apply(self):
        if self.selected_tab == "pretrained":
            selection = self.pretrained_listbox.get(self.pretrained_listbox.curselection())
            self.result = selection.split(" - ")[0]
        elif self.selected_tab == "custom":
            self.result = self.custom_path_var.get().strip()


class TrackerSelectorDialog(simpledialog.Dialog):
    def body(self, master):
        trackers = [
            ("bytetrack", "ByteTrack - YOLO's default tracker"),
            ("botsort", "BoTSORT - YOLO's alternative tracker"),
        ]

        self.listbox = tk.Listbox(master, width=60, height=10)
        for tracker, desc in trackers:
            self.listbox.insert(tk.END, f"{tracker} - {desc}")
        self.listbox.pack(padx=5, pady=5)

        return self.listbox

    def validate(self):
        if not self.listbox.curselection():
            messagebox.showwarning("Warning", "Please select a tracker")
            return False
        return True

    def apply(self):
        selection = self.listbox.get(self.listbox.curselection())
        self.result = selection.split(" - ")[0]


class ReidModelSelectorDialog(simpledialog.Dialog):
    def body(self, master):
        reid_models = list(REID_MODELS.items())

        self.listbox = tk.Listbox(master, width=60, height=15)
        for model, desc in reid_models:
            self.listbox.insert(tk.END, f"{model} - {desc}")
        self.listbox.pack(padx=5, pady=5)

        help_text = tk.Label(
            master,
            text="ReID models help trackers maintain identity through occlusions\n"
            "Lightweight: Faster but less accurate\n"
            "Heavy: More accurate but uses more resources",
            justify="left",
        )
        help_text.pack(padx=5, pady=5)

        return self.listbox

    def validate(self):
        if not self.listbox.curselection():
            messagebox.showwarning("Warning", "Please select a ReID model")
            return False
        return True

    def apply(self):
        selection = self.listbox.get(self.listbox.curselection())
        self.result = selection.split(" - ")[0]


class ClassSelectorDialog(simpledialog.Dialog):
    def body(self, master):
        # Default COCO classes used by YOLO
        self.coco_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

        # Frame for the list of classes
        classes_frame = tk.Frame(master)
        classes_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Label and scrollbar for the list of classes
        tk.Label(classes_frame, text="Available classes:").grid(row=0, column=0, sticky="w")

        # Create a Text widget with scrollbar to display the classes
        self.classes_text = tk.Text(classes_frame, width=40, height=15)
        scrollbar = tk.Scrollbar(classes_frame, command=self.classes_text.yview)
        self.classes_text.config(yscrollcommand=scrollbar.set)

        self.classes_text.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")

        # Fill the Text widget with the list of classes
        for class_id, class_name in self.coco_classes.items():
            self.classes_text.insert(tk.END, f"{class_id}: {class_name}\n")

        self.classes_text.config(state="disabled")  # Make the text read-only

        # Frame for the selected classes input
        input_frame = tk.Frame(master)
        input_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(
            input_frame,
            text="Enter the classes you want to track (numbers separated by commas):",
        ).pack(anchor="w")

        # Input field for the selected classes
        self.classes_entry = tk.Entry(input_frame, width=40)
        self.classes_entry.pack(fill="x", pady=5)
        self.classes_entry.insert(0, "0, 32")  # Default: person and sports ball

        # Add examples and instructions
        tk.Label(
            input_frame,
            text="Examples:\n- '0' to track only people\n- '0, 2, 5, 7' to track people, cars, buses and trains\n- Leave empty to track all classes",
            justify="left",
        ).pack(anchor="w", pady=5)

        return self.classes_entry

    def validate(self):
        # Validate user input
        try:
            # If empty, accept (meaning all classes)
            if not self.classes_entry.get().strip():
                return True

            # Check if the input can be converted to a list of integers
            classes_input = self.classes_entry.get().replace(" ", "")
            if classes_input:
                class_ids = [int(x) for x in classes_input.split(",")]

                # Check if all IDs are valid
                for class_id in class_ids:
                    if class_id < 0 or class_id > 79:
                        messagebox.showwarning(
                            "Warning",
                            f"Class ID {class_id} out of valid range (0-79)",
                        )
                        return False
            return True
        except ValueError:
            messagebox.showwarning("Warning", "Invalid format. Use numbers separated by commas.")
            return False

    def apply(self):
        # Process the input and return the list of classes
        classes_input = self.classes_entry.get().strip().replace(" ", "")
        if not classes_input:
            self.result = None  # No specific class means all classes
        else:
            self.result = [int(x) for x in classes_input.split(",")]


def standardize_filename(filename: str) -> str:
    """
    Remove unwanted characters and replace spaces with underscores.
    This function ensures that the file name follows a safe pattern for processing.
    """
    # Replace spaces and colons, for example:
    new_name = filename.replace(" ", "_").replace(":", "-")
    # Alternatively, for a more general cleanup, uncomment the line below:
    # new_name = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    return new_name


def get_color_for_id_improved(tracker_id):
    """Generate a distinct color for each tracker ID using HSV color space."""
    # Use different prime numbers to create greater variation in hue
    prime_factors = [79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    h = ((tracker_id * prime_factors[tracker_id % len(prime_factors)]) % 359) / 359.0

    # Greater variation in saturation and value based on the ID
    s_values = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    v_values = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    s = s_values[tracker_id % len(s_values)]
    v = v_values[(tracker_id // len(s_values)) % len(v_values)]

    rgb = colorsys.hsv_to_rgb(h, s, v)
    # Convert to BGR (OpenCV format) with values 0-255
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def get_color_palette(num_colors=200):
    """Generates a maximally distinct color palette using HSV color space."""
    print(f"Generating a palette with {num_colors} distinct colors")

    # Base palette with distinct colors for the first few IDs - EXPANDED
    base_palette = [
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (0, 165, 255),  # Orange
        (255, 255, 255),  # White
        (128, 128, 128),  # Gray
        (128, 128, 255),  # Pink
        (255, 128, 128),  # Light Blue
        (128, 255, 128),  # Light Green
        (60, 20, 220),  # Crimson
        (32, 165, 218),  # Coral
        (0, 69, 255),  # Brown
        (204, 50, 153),  # Violet
        (79, 79, 47),  # Olive
        (143, 143, 188),  # Silver
        (209, 206, 0),  # Turquoise
        (0, 215, 255),  # Gold
        (34, 34, 178),  # Maroon
        (85, 128, 0),  # Forest Green
        (0, 0, 128),  # Dark Red
        (192, 192, 192),  # Light gray
        (0, 140, 255),  # Dark orange
        (128, 0, 0),  # Navy
        (0, 128, 128),  # Teal
        (220, 20, 60),  # Dark blue
        (122, 150, 233),  # Salmon
    ]

    # Pre-calculate additional colors using a more sophisticated method
    additional_colors = []

    # Generate additional colors by distributing evenly in the HSV space
    for i in range(num_colors - len(base_palette)):
        # Vary hue using prime numbers for more uniform distribution
        h = ((i * 47 + 13) % 360) / 360.0  # Distribute evenly across the spectrum

        # Alternate saturation and value levels for more variations
        s_level = 0.7 + 0.3 * ((i // 7) % 4) / 3  # Varies between 0.7 and 1.0
        v_level = 0.7 + 0.3 * ((i // 3) % 4) / 3  # Varies between 0.7 and 1.0

        rgb = colorsys.hsv_to_rgb(h, s_level, v_level)
        additional_colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))

    # Combine base + additional colors
    result = base_palette.copy() + additional_colors

    # Ensure we have exactly the requested number of colors
    if len(result) > num_colors:
        result = result[:num_colors]

    return result


# Replace the get_color_for_id function with this version that uses the palette
COLORS = None


def get_color_for_id(tracker_id):
    """Returns a distinct color for the ID using a pre-calculated palette."""
    global COLORS

    # Lazy-initialize the color palette (100 colors should be enough for most tracking needs)
    if COLORS is None:
        COLORS = get_color_palette(100)

    # Use modulo to reuse colors if we have more IDs than colors
    color_idx = tracker_id % len(COLORS)
    return COLORS[color_idx]


def create_combined_detection_csv(output_dir):
    """
    Creates a combined CSV file with detection data organized by ID columns.
    Each detected object gets their own set of columns (ID_n, X_n, Y_n, RGB_n).

    Args:
        output_dir: Directory containing the individual detection CSV files

    Returns:
        Path to the created combined CSV file
    """
    # Find all detection CSV files (any class with _id pattern)
    detection_csv_files = glob.glob(os.path.join(output_dir, "*_id_*.csv"))

    if not detection_csv_files:
        print(f"No detection tracking files found in {output_dir}")
        return None

    # Sort files by ID number in ascending order (01, 02, 03, ...)
    def extract_id(filepath):
        filename = os.path.basename(filepath)
        try:
            id_part = filename.rsplit("_id_", 1)[1].split(".")[0]
            return int(id_part)
        except (IndexError, ValueError):
            return 0

    detection_csv_files = sorted(detection_csv_files, key=extract_id)

    print(f"Found {len(detection_csv_files)} detection tracking files")
    for f in detection_csv_files:
        print(f"  - {os.path.basename(f)}")

    # Determine total frames from files (they are pre-initialized 0..N-1)
    max_frame = 0
    for csv_file in detection_csv_files:
        try:
            df = pd.read_csv(csv_file, usecols=("Frame",))  # type: ignore[call-overload]  # faster
            if not df.empty and df["Frame"].notna().any():
                max_frame = max(max_frame, int(df["Frame"].max()))
        except Exception:
            continue
    total_frames = max_frame + 1

    base = pd.DataFrame({"Frame": np.arange(int(total_frames))})

    # Merge each file's columns side-by-side with label+id suffix
    merged = base.copy()
    for csv_file in detection_csv_files:
        try:
            filename = os.path.basename(csv_file)
            label_part = filename.rsplit("_id_", 1)[0]
            id_part = filename.rsplit("_id_", 1)[1].split(".")[0]
            suffix = f"_{label_part}_id_{id_part}"

            df = pd.read_csv(csv_file)

            take_cols = [
                "Frame",
                "Tracker ID",
                "Label",
                "X_min",
                "Y_min",
                "X_max",
                "Y_max",
                "Confidence",
                "Color_R",
                "Color_G",
                "Color_B",
            ]
            for col in take_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # Keep only required columns and rename with unique suffix
            df = df[take_cols].copy()
            rename_map = {c: (c if c == "Frame" else f"{c}{suffix}") for c in take_cols}
            df = df.rename(columns=rename_map)

            # Merge on Frame
            merged = merged.merge(df, on="Frame", how="left")
        except Exception as e:
            print(f"Error merging file {filename}: {e}")
            continue

    output_file = os.path.join(output_dir, "all_id_detection.csv")
    merged.to_csv(output_file, index=False)
    print(f"Combined detection tracking data saved to: {output_file}")
    return output_file


def create_merged_detection_csv(output_dir, total_frames):
    """
    Create wide per-label CSV(s) with one row per frame (0..N-1) merging all IDs.
    Writes all_id_merge_<Label>.csv and an alias all_id_merge.csv if only one label.
    """
    csv_files = glob.glob(os.path.join(output_dir, "*_id_*.csv"))

    if not csv_files:
        print(f"No per-ID tracking files found in {output_dir}")
        return None

    print(f"Found {len(csv_files)} per-ID tracking files for merging")

    rows = []
    expected_columns = [
        "Frame",
        "Tracker ID",
        "Label",
        "X_min",
        "Y_min",
        "X_max",
        "Y_max",
        "Confidence",
        "Color_R",
        "Color_G",
        "Color_B",
    ]

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"Reading {os.path.basename(csv_file)}: {len(df)} rows")

            # Ensure all expected columns exist
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = np.nan

            # Keep only expected columns and append
            df = df[expected_columns]
            rows.append(df)
        except Exception as e:
            print(f"Error merging file {csv_file}: {e}")
            continue

    if not rows:
        print(f"No valid rows to merge in {output_dir}")
        return None

    df = pd.concat(rows, ignore_index=True)
    print(f"Total concatenated rows: {len(df)}")

    # Ensure numeric types where expected
    for col in ["Frame", "Tracker ID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in critical columns before grouping
    df = df.dropna(subset=["Label"])
    print(f"Rows after dropping NaN labels: {len(df)}")

    # Build one wide CSV per Label with exactly total_frames rows
    out_paths = []
    for label, df_label in df.groupby("Label"):
        print(f"Processing label: {label} with {len(df_label)} rows")

        # Ensure Frame is integer
        df_label = df_label.dropna(subset=["Frame"]).copy()
        df_label["Frame"] = df_label["Frame"].astype(int)

        # Flag rows that have actual detection (any of key coords present)
        present_cols = ["X_min", "X_max", "Y_max"]
        df_label["__present__"] = df_label[present_cols].notna().any(axis=1).astype(int)

        # Sort by Frame asc and presence desc so non-NaN rows come first
        df_label = df_label.sort_values(["Frame", "__present__"], ascending=[True, False])

        # Pick one row per Frame (prefer the one with data)
        picked = df_label.drop_duplicates(subset=["Frame"], keep="first").copy()
        picked = picked.sort_values("Frame")

        # Reindex to guarantee exactly total_frames rows (0..N-1)
        base = pd.DataFrame({"Frame": np.arange(int(total_frames))})
        merged = base.merge(picked.drop(columns=["__present__"]), on="Frame", how="left")

        out_path = os.path.join(output_dir, f"all_id_merge_{label}.csv")
        merged.to_csv(out_path, index=False)
        out_paths.append(out_path)
        print(f"Merged (stack-sort-pick) data saved to: {out_path}")

    # If only one label, also write canonical name for convenience
    if len(out_paths) == 1:
        alias = os.path.join(output_dir, "all_id_merge.csv")
        pd.read_csv(out_paths[0]).to_csv(alias, index=False)
        print(f"Alias written: {alias}")
        return alias
    return out_paths


def _get_pose_skeleton():
    """Return COCO keypoint skeleton edges (17-keypoint layout)."""
    return [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # head
        (0, 5),
        (0, 6),  # shoulders to nose
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # arms
        (5, 11),
        (6, 12),
        (11, 12),  # torso
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # legs
    ]


def _draw_keypoints_and_skeleton(frame, keypoints_abs, color=(0, 255, 0)):
    """
    Draw circles and skeleton on the frame.
    keypoints_abs: list/array with shape (N, 3) -> x, y, conf (absolute coords).
    """
    skeleton = _get_pose_skeleton()
    # Draw edges
    for i, j in skeleton:
        if i < len(keypoints_abs) and j < len(keypoints_abs):
            xi, yi, ci = keypoints_abs[i]
            xj, yj, cj = keypoints_abs[j]
            if not np.isnan(xi) and not np.isnan(yi) and not np.isnan(xj) and not np.isnan(yj):
                cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color, 2, cv2.LINE_AA)
    # Draw joints
    for kp in keypoints_abs:
        x, y, c = kp
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
    return frame


def merge_tracking_csvs(csv_files, output_csv_path, label="merged"):
    """
    Merge multiple tracking CSV files into a single CSV.
    For each frame, uses the first non-NaN bbox found across all CSVs.
    This is useful when multiple IDs track the same person over time.

    Args:
        csv_files: List of paths to CSV files to merge
        output_csv_path: Path where merged CSV will be saved
        label: Label to use in the merged CSV (default: "merged")

    Returns:
        Path to merged CSV file, or None on error
    """
    try:
        if not csv_files:
            print("Error: No CSV files provided for merging")
            return None

        print(f"Merging {len(csv_files)} tracking CSV files...")
        for csv_file in csv_files:
            print(f"  - {os.path.basename(csv_file)}")

        # Read all CSVs
        dfs = []
        max_frames = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
                if not df.empty:
                    max_frames = max(max_frames, int(df["Frame"].max()) + 1)
            except Exception as e:
                print(f"Warning: Error reading {csv_file}: {e}")
                continue

        if not dfs:
            print("Error: Could not read any CSV files")
            return None

        # Create merged DataFrame with all frames
        merged_data = []
        merged_id = 1  # Use ID 1 for merged data

        for frame_idx in range(max_frames):
            row_data = {
                "Frame": frame_idx,
                "Tracker ID": merged_id,
                "Label": label,
                "X_min": np.nan,
                "Y_min": np.nan,
                "X_max": np.nan,
                "Y_max": np.nan,
                "Confidence": np.nan,
                "Color_R": 255,
                "Color_G": 0,
                "Color_B": 0,
            }

            # Find first non-NaN bbox for this frame across all CSVs
            for df in dfs:
                frame_data = df[df["Frame"] == frame_idx]
                if not frame_data.empty:
                    row = frame_data.iloc[0]
                    # Check if bbox is valid (not NaN)
                    if (
                        pd.notna(row.get("X_min"))
                        and pd.notna(row.get("Y_min"))
                        and pd.notna(row.get("X_max"))
                        and pd.notna(row.get("Y_max"))
                    ):
                        row_data["X_min"] = float(row["X_min"])
                        row_data["Y_min"] = float(row["Y_min"])
                        row_data["X_max"] = float(row["X_max"])
                        row_data["Y_max"] = float(row["Y_max"])
                        if pd.notna(row.get("Confidence")):
                            row_data["Confidence"] = float(row["Confidence"])
                        # Use color from first valid row found
                        if pd.notna(row.get("Color_R")):
                            row_data["Color_R"] = int(row["Color_R"])
                        if pd.notna(row.get("Color_G")):
                            row_data["Color_G"] = int(row["Color_G"])
                        if pd.notna(row.get("Color_B")):
                            row_data["Color_B"] = int(row["Color_B"])
                        break  # Use first valid bbox found

            merged_data.append(row_data)

        # Create DataFrame and save
        merged_df = pd.DataFrame(merged_data)
        merged_df.to_csv(output_csv_path, index=False)

        # Count frames with valid bboxes
        valid_frames = merged_df[merged_df["X_min"].notna()].shape[0]
        print(f"Merged CSV created: {os.path.basename(output_csv_path)}")
        print(f"  Total frames: {len(merged_df)}")
        print(f"  Frames with valid bbox: {valid_frames}")

        return output_csv_path

    except Exception as e:
        print(f"Error merging CSV files: {e}")
        import traceback

        traceback.print_exc()
        return None


def pick_video_for_pose(tracking_dir, parent=None):
    """
    Find video file for pose estimation.
    Prioritizes processed_*.mp4 files, but also accepts any .mp4, .avi, .mov, .mkv file.
    If multiple videos found, prompts user to select one.
    If no videos found, allows manual file selection.

    Returns:
        Path to selected video file, or None if cancelled.
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = []

    # First, look for processed_*.mp4 files
    processed_videos = glob.glob(os.path.join(tracking_dir, "processed_*.mp4"))
    if processed_videos:
        video_files.extend(sorted(processed_videos))

    # Then, look for any other video files
    for ext in video_extensions:
        pattern = os.path.join(tracking_dir, f"*{ext}")
        found_videos = glob.glob(pattern)
        for video in found_videos:
            if video not in video_files:  # Avoid duplicates
                video_files.append(video)

    # Remove processed_* from general list if already in priority list
    video_files = sorted(set(video_files))

    if not video_files:
        # No videos found, allow manual selection
        if parent:
            video_path = filedialog.askopenfilename(
                parent=parent,
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
            )
            return video_path if video_path else None
        else:
            return None

    if len(video_files) == 1:
        # Only one video found, use it automatically
        return video_files[0]

    # Multiple videos found, prompt user to select
    if parent is None:
        root = tk.Tk()
        root.withdraw()
        created_root = True
    else:
        root = parent
        created_root = False

    selection_dialog = tk.Toplevel(root)
    selection_dialog.title("Select Video for Pose Estimation")
    selection_dialog.geometry("600x500")
    selection_dialog.transient(root)
    selection_dialog.grab_set()

    # Center window
    selection_dialog.update_idletasks()
    x = (selection_dialog.winfo_screenwidth() // 2) - (selection_dialog.winfo_width() // 2)
    y = (selection_dialog.winfo_screenheight() // 2) - (selection_dialog.winfo_height() // 2)
    selection_dialog.geometry(f"+{x}+{y}")

    tk.Label(
        selection_dialog,
        text="Multiple videos found. Please select one:",
        font=("Arial", 10, "bold"),
    ).pack(pady=10)

    listbox = tk.Listbox(selection_dialog, height=12)
    scrollbar = tk.Scrollbar(selection_dialog, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)

    for video_file in video_files:
        listbox.insert(tk.END, os.path.basename(video_file))

    listbox.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    scrollbar.pack(side="right", fill="y", pady=10)
    listbox.selection_set(0)

    selected_video = [None]

    def confirm_selection():
        if listbox.curselection():
            idx = listbox.curselection()[0]
            selected_video[0] = video_files[idx]
        selection_dialog.destroy()
        if created_root and root.winfo_exists():
            root.destroy()

    def cancel_selection():
        selection_dialog.destroy()
        if created_root and root.winfo_exists():
            root.destroy()

    buttons_frame = tk.Frame(selection_dialog)
    buttons_frame.pack(fill="x", padx=10, pady=10)

    tk.Button(
        buttons_frame, text="OK", command=confirm_selection, bg="#4CAF50", fg="white", padx=20
    ).pack(side="left", padx=5)

    tk.Button(buttons_frame, text="Cancel", command=cancel_selection, padx=20).pack(
        side="left", padx=5
    )

    selection_dialog.wait_window()
    if created_root and root.winfo_exists():
        root.destroy()

    return selected_video[0]


def select_id_and_run_pose():
    """
    GUI to select tracking directory, view video with bboxes/IDs, and select ID for pose estimation.
    """
    # Prefer existing Tk root to avoid multiple roots (pyimage errors); create only if needed
    created_root = False
    root = getattr(tk, "_default_root", None)
    if root is None or not isinstance(root, tk.Tk):
        root = tk.Tk()
        root.withdraw()
        created_root = True
    print(f"[pose] Using root: created={created_root}, exists={root.winfo_exists()}")

    # Select tracking directory
    tracking_dir = filedialog.askdirectory(title="Select Tracking Directory")
    if not tracking_dir:
        if created_root and root.winfo_exists():
            root.destroy()
        return

    # Find tracking CSV files
    csv_files = glob.glob(os.path.join(tracking_dir, "*_id_*.csv"))
    # Filter out combined/merged files (all_id_merge.csv, all_id_detection.csv, etc.)
    csv_files = [f for f in csv_files if not os.path.basename(f).startswith("all_id_")]

    if not csv_files:
        messagebox.showerror("Error", f"No tracking CSV files found in:\n{tracking_dir}")
        return

    # Find video (processed_* preferred, else any video; allow manual pick)
    video_path = pick_video_for_pose(tracking_dir, parent=root)
    if not video_path:
        messagebox.showerror("Error", f"No video found in:\n{tracking_dir}")
        return

    # Parse available IDs from CSV files
    available_ids = []
    id_info = {}  # {id: (label, csv_file)}

    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        parts = filename.replace(".csv", "").split("_id_")
        if len(parts) == 2:
            label = parts[0]
            try:
                tracker_id = int(parts[1])
                available_ids.append(tracker_id)
                id_info[tracker_id] = (label, csv_file)
            except ValueError:
                # Skip files with non-numeric ID parts
                continue

    if not available_ids:
        messagebox.showerror("Error", "No valid tracking IDs found in CSV files")
        return

    available_ids.sort()

    # Create selection dialog
    selection_dialog = tk.Toplevel(root)
    selection_dialog.title("Select ID for Pose Estimation")
    selection_dialog.geometry("1200x900")
    selection_dialog.transient(root)
    selection_dialog.grab_set()

    # Center window
    selection_dialog.update_idletasks()
    x = (selection_dialog.winfo_screenwidth() // 2) - (selection_dialog.winfo_width() // 2)
    y = (selection_dialog.winfo_screenheight() // 2) - (selection_dialog.winfo_height() // 2)
    selection_dialog.geometry(f"+{x}+{y}")

    # Video preview frame
    preview_frame = tk.Frame(selection_dialog)
    preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

    tk.Label(preview_frame, text="Video Preview with Tracked IDs", font=("Arial", 12, "bold")).pack(
        pady=5
    )

    # Canvas for video display
    canvas = tk.Canvas(preview_frame, bg="black", width=640, height=360)
    canvas.pack(pady=10, fill="both", expand=True)

    # Load first frame and draw bboxes
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Could not read video")
        cap.release()
        selection_dialog.destroy()
        root.destroy()
        return

    # Read tracking data for first frame
    tracking_data = {}
    for tracker_id, (label, csv_file) in id_info.items():
        df = pd.read_csv(csv_file)
        frame_data = df[df["Frame"] == 0]
        if not frame_data.empty and pd.notna(frame_data.iloc[0]["X_min"]):
            tracking_data[tracker_id] = {
                "label": label,
                "x_min": int(frame_data.iloc[0]["X_min"]),
                "y_min": int(frame_data.iloc[0]["Y_min"]),
                "x_max": int(frame_data.iloc[0]["X_max"]),
                "y_max": int(frame_data.iloc[0]["Y_max"]),
            }

    # Draw bboxes on frame
    display_frame = frame.copy()
    for tracker_id, data in tracking_data.items():
        color = get_color_for_id(tracker_id)
        label = data["label"]
        cv2.rectangle(
            display_frame, (data["x_min"], data["y_min"]), (data["x_max"], data["y_max"]), color, 2
        )
        cv2.putText(
            display_frame,
            f"id {tracker_id} {label}",
            (data["x_min"], data["y_min"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    # Resize frame to fit canvas (max 640x360)
    h, w = display_frame.shape[:2]
    scale = min(640 / w, 360 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    display_frame = cv2.resize(display_frame, (new_w, new_h))

    # Convert to PhotoImage
    display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(display_frame_rgb)
    img = ImageTk.PhotoImage(image=pil_image)

    # Center image on canvas
    canvas_width = 640
    canvas_height = 360
    x = canvas_width // 2
    y = canvas_height // 2
    canvas.create_image(x, y, image=img, anchor="center")
    canvas.image = img  # Keep a reference (dynamic attr for gc)

    cap.release()

    # ID selection frame
    id_frame = tk.Frame(selection_dialog)
    id_frame.pack(fill="x", padx=10, pady=10)

    tk.Label(id_frame, text="Available Tracked IDs:", font=("Arial", 10, "bold")).pack(anchor="w")

    # Listbox for ID selection (multiple selection enabled)
    listbox_frame = tk.Frame(id_frame)
    listbox_frame.pack(fill="both", expand=True, pady=5)

    listbox = tk.Listbox(listbox_frame, height=6, selectmode=tk.EXTENDED)
    scrollbar = tk.Scrollbar(listbox_frame, orient="vertical", command=listbox.yview)
    listbox.configure(yscrollcommand=scrollbar.set)

    for tracker_id in available_ids:
        label = id_info[tracker_id][0]
        listbox.insert(tk.END, f"ID {tracker_id:02d} ({label})")

    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    if available_ids:
        listbox.selection_set(0)  # Select first by default

    # Help text for multiple selection
    help_label = tk.Label(
        id_frame,
        text="Tip: Hold Ctrl/Cmd to select multiple IDs for merging",
        font=("Arial", 8),
        fg="gray",
    )
    help_label.pack(anchor="w", pady=(5, 0))

    # Buttons frame
    buttons_frame = tk.Frame(selection_dialog)
    buttons_frame.pack(fill="x", padx=10, pady=10)

    selected_ids = [None]  # Now can be a list of IDs
    selected_model = [None]
    selected_conf = [0.10]
    selected_iou = [0.70]

    def run_pose():
        if not listbox.curselection():
            messagebox.showwarning("Warning", "Please select at least one ID")
            return

        # Get all selected IDs
        selections = listbox.curselection()
        selected_id_list = []
        selected_csvs = []
        selected_labels = []

        for idx in selections:
            selection = listbox.get(idx)
            # Extract ID from "ID 01 (person)" format
            id_str = selection.split()[1]
            tracker_id = int(id_str)
            selected_id_list.append(tracker_id)

            # Get CSV file and label for this ID
            label, csv_file = id_info[tracker_id]
            selected_csvs.append(csv_file)
            selected_labels.append(label)

        # Check if all selected IDs have the same label
        unique_labels = set(selected_labels)
        merged_label = selected_labels[0] if len(unique_labels) == 1 else "merged"

        selected_ids[0] = selected_id_list

        # Ask for pose model
        pose_models = [
            ("yolo26n-pose.pt", "Pose - Nano (fastest)"),
            ("yolo26s-pose.pt", "Pose - Small"),
            ("yolo26m-pose.pt", "Pose - Medium"),
            ("yolo26l-pose.pt", "Pose - Large"),
            ("yolo26x-pose.pt", "Pose - XLarge (most accurate)"),
        ]

        model_dialog = tk.Toplevel(selection_dialog)
        model_dialog.title("Pose Configuration")
        model_dialog.geometry("450x400")
        model_dialog.transient(selection_dialog)
        model_dialog.grab_set()

        model_dialog.update_idletasks()
        x = (model_dialog.winfo_screenwidth() // 2) - (model_dialog.winfo_width() // 2)
        y = (model_dialog.winfo_screenheight() // 2) - (model_dialog.winfo_height() // 2)
        model_dialog.geometry(f"+{x}+{y}")

        # Model selection
        tk.Label(
            model_dialog, text="Select Pose Estimation Model:", font=("Arial", 10, "bold")
        ).pack(pady=(10, 5))

        model_listbox = tk.Listbox(model_dialog, height=5)
        for model, desc in pose_models:
            model_listbox.insert(tk.END, f"{model} - {desc}")
        model_listbox.pack(padx=10, pady=5, fill="both", expand=True)
        model_listbox.selection_set(0)

        # Configuration frame for conf and iou
        config_frame = tk.LabelFrame(model_dialog, text="Pose Detection Parameters", padx=5, pady=5)
        config_frame.pack(fill="x", padx=10, pady=10)

        # Confidence threshold
        conf_frame = tk.Frame(config_frame)
        conf_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(conf_frame, text="Confidence threshold:").pack(side="left", padx=(0, 10))
        conf_entry = tk.Entry(conf_frame, width=10)
        conf_entry.insert(0, "0.10")
        conf_entry.pack(side="left")
        help_conf = tk.Label(conf_frame, text="?", cursor="hand2", fg="blue")
        help_conf.pack(side="left", padx=(5, 0))
        conf_tooltip = (
            "Confidence threshold (0-1):\n"
            "Controls how confident the model must be to detect a pose.\n"
            "Lower values (e.g., 0.1): More detections but may include false positives\n"
            "Higher values (e.g., 0.5): Fewer but more accurate detections\n"
            "Recommended: 0.1-0.3 for pose estimation"
        )

        def show_conf_help(e):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.geometry(f"+{e.x_root + 10}+{e.y_root + 10}")
            label = tk.Label(
                tooltip, text=conf_tooltip, bg="yellow", justify="left", padx=5, pady=5
            )
            label.pack()

            def hide_help(e):
                tooltip.destroy()

            tooltip.bind("<Leave>", hide_help)

        help_conf.bind("<Enter>", show_conf_help)

        # IoU threshold
        iou_frame = tk.Frame(config_frame)
        iou_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(iou_frame, text="IoU threshold:").pack(side="left", padx=(0, 10))
        iou_entry = tk.Entry(iou_frame, width=10)
        iou_entry.insert(0, "0.70")
        iou_entry.pack(side="left")
        help_iou = tk.Label(iou_frame, text="?", cursor="hand2", fg="blue")
        help_iou.pack(side="left", padx=(5, 0))
        iou_tooltip = (
            "Intersection over Union threshold (0-1):\n"
            "Controls how much overlap is needed to merge multiple detections.\n"
            "Higher values (e.g., 0.9): Very strict matching\n"
            "Lower values (e.g., 0.5): More lenient matching\n"
            "Recommended: 0.7 for most cases"
        )

        def show_iou_help(e):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.geometry(f"+{e.x_root + 10}+{e.y_root + 10}")
            label = tk.Label(tooltip, text=iou_tooltip, bg="yellow", justify="left", padx=5, pady=5)
            label.pack()

            def hide_help(e):
                tooltip.destroy()

            tooltip.bind("<Leave>", hide_help)

        help_iou.bind("<Enter>", show_iou_help)

        def confirm_model():
            # Get model selection
            if model_listbox.curselection():
                selection = model_listbox.get(model_listbox.curselection())
                selected_model[0] = selection.split(" - ")[0]

            # Get conf and iou values
            try:
                conf_val = float(conf_entry.get())
                if not 0.0 <= conf_val <= 1.0:
                    messagebox.showerror(
                        "Error", "Confidence threshold must be between 0.0 and 1.0"
                    )
                    return
                selected_conf[0] = conf_val
            except ValueError:
                messagebox.showerror("Error", "Invalid confidence threshold value")
                return

            try:
                iou_val = float(iou_entry.get())
                if not 0.0 <= iou_val <= 1.0:
                    messagebox.showerror("Error", "IoU threshold must be between 0.0 and 1.0")
                    return
                selected_iou[0] = iou_val
            except ValueError:
                messagebox.showerror("Error", "Invalid IoU threshold value")
                return

            model_dialog.destroy()
            selection_dialog.destroy()
            if created_root and root.winfo_exists():
                root.destroy()

            # Process pose estimation for selected ID(s)
            if selected_ids[0] and selected_model[0]:
                selected_id_list = selected_ids[0]

                # If multiple IDs selected, merge CSVs first
                if len(selected_id_list) > 1:
                    print(f"\nMultiple IDs selected: {selected_id_list}")
                    print("Merging CSVs before pose estimation...")

                    # Get CSV files for selected IDs
                    csv_files_to_merge = []
                    for tracker_id in selected_id_list:
                        csv_files = glob.glob(
                            os.path.join(tracking_dir, f"*_id_{tracker_id:02d}.csv")
                        )
                        if csv_files:
                            csv_files_to_merge.append(csv_files[0])

                    if csv_files_to_merge:
                        # Create merged CSV in tracking directory
                        merged_csv_path = os.path.join(
                            tracking_dir, f"{merged_label}_id_merged.csv"
                        )
                        merged_csv = merge_tracking_csvs(
                            csv_files_to_merge, merged_csv_path, label=merged_label
                        )

                        if merged_csv:
                            # Process pose estimation using merged CSV
                            process_pose_for_merged_csv(
                                tracking_dir,
                                merged_csv_path,
                                video_path,
                                device=detect_optimal_device(),
                                pose_model_name=selected_model[0],
                                conf=selected_conf[0],
                                iou=selected_iou[0],
                            )
                        else:
                            messagebox.showerror("Error", "Failed to merge CSV files")
                    else:
                        messagebox.showerror("Error", "Could not find CSV files for selected IDs")
                else:
                    # Single ID selected, process normally
                    process_pose_for_single_id(
                        tracking_dir,
                        selected_id_list[0],
                        video_path,
                        device=detect_optimal_device(),
                        pose_model_name=selected_model[0],
                        conf=selected_conf[0],
                        iou=selected_iou[0],
                    )

        buttons_frame = tk.Frame(model_dialog)
        buttons_frame.pack(fill="x", padx=10, pady=10)

        tk.Button(
            buttons_frame, text="OK", command=confirm_model, bg="#4CAF50", fg="white", padx=20
        ).pack(side="left", padx=5)

        tk.Button(
            buttons_frame, text="Cancel", command=lambda: model_dialog.destroy(), padx=20
        ).pack(side="left", padx=5)

    tk.Button(
        buttons_frame,
        text="Run Pose Estimation",
        command=run_pose,
        bg="#FF9800",
        fg="white",
        font=("Arial", 10, "bold"),
        padx=20,
        pady=5,
    ).pack(side="left", padx=5)

    tk.Button(
        buttons_frame,
        text="Cancel",
        command=lambda: (
            selection_dialog.destroy(),
            root.destroy() if created_root and root.winfo_exists() else None,
        ),
        padx=20,
        pady=5,
    ).pack(side="left", padx=5)

    selection_dialog.wait_window()
    if created_root and root.winfo_exists() and not selection_dialog.winfo_exists():
        root.destroy()


def process_pose_for_merged_csv(
    tracking_dir,
    merged_csv_path,
    video_path,
    device=None,
    pose_model_name="yolo26n-pose.pt",
    conf=0.10,
    iou=0.70,
):
    """
    Process pose estimation using a merged CSV file (from multiple IDs).

    Args:
        tracking_dir: Directory containing tracking results
        merged_csv_path: Path to the merged CSV file
        video_path: Path to the video file
        device: Device to use for pose estimation
        pose_model_name: Name of the pose model to use
        conf: Confidence threshold for pose detection (default: 0.10)
        iou: IoU threshold for pose detection (default: 0.70)
    """
    print("\n" + "=" * 80)
    print("POSE ESTIMATION FOR MERGED CSV")
    print("=" * 80)

    # Auto-detect device if not specified
    if device is None:
        device = detect_optimal_device()
    print(f"Using device: {device}")

    if not os.path.exists(merged_csv_path):
        print(f"Error: Merged CSV file not found: {merged_csv_path}")
        messagebox.showerror("Error", f"Merged CSV file not found:\n{merged_csv_path}")
        return False

    # Get video basename (without extension) for directory and file naming
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    # Remove 'processed_' prefix if present
    if video_basename.startswith("processed_"):
        video_basename = video_basename[len("processed_") :]
    # Create pose output directory with timestamp
    pose_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_output_dir = os.path.join(tracking_dir, f"{video_basename}_pose_{pose_timestamp}")
    os.makedirs(pose_output_dir, exist_ok=True)

    csv_file = merged_csv_path
    filename = os.path.basename(csv_file)
    parts = filename.replace(".csv", "").split("_id_")
    label = parts[0] if len(parts) == 2 else "merged"

    print(f"Using video: {os.path.basename(video_path)}")
    print(f"Using merged CSV: {os.path.basename(merged_csv_path)}")
    print(f"Pose detection params: conf={conf}, iou={iou}")

    # Continue with same processing as single ID (using merged CSV)
    return _process_pose_from_csv(
        tracking_dir,
        video_path,
        csv_file,
        label,
        video_basename,
        pose_output_dir,
        device,
        pose_model_name,
        conf,
        iou,
    )


def process_pose_for_single_id(
    tracking_dir,
    tracker_id,
    video_path,
    device=None,
    pose_model_name="yolo26n-pose.pt",
    conf=0.10,
    iou=0.70,
):
    """
    Process pose estimation for a single tracked ID.

    Args:
        tracking_dir: Directory containing tracking results
        tracker_id: The ID to process
        video_path: Path to the video file
        device: Device to use for pose estimation
        pose_model_name: Name of the pose model to use
        conf: Confidence threshold for pose detection (default: 0.10)
        iou: IoU threshold for pose detection (default: 0.70)
    """
    print("\n" + "=" * 80)
    print(f"POSE ESTIMATION FOR ID {tracker_id:02d}")
    print("=" * 80)

    # Auto-detect device if not specified
    if device is None:
        device = detect_optimal_device()
    print(f"Using device: {device}")

    # Get video basename (without extension) for directory and file naming
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    # Remove 'processed_' prefix if present
    if video_basename.startswith("processed_"):
        video_basename = video_basename[len("processed_") :]
    # Create pose output directory with timestamp
    pose_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_output_dir = os.path.join(tracking_dir, f"{video_basename}_pose_{pose_timestamp}")
    os.makedirs(pose_output_dir, exist_ok=True)

    # Find CSV file for this ID
    csv_files = glob.glob(os.path.join(tracking_dir, f"*_id_{tracker_id:02d}.csv"))
    if not csv_files:
        print(f"Error: No tracking CSV found for ID {tracker_id:02d}")
        messagebox.showerror("Error", f"No tracking CSV found for ID {tracker_id:02d}")
        return False

    csv_file = csv_files[0]
    filename = os.path.basename(csv_file)
    parts = filename.replace(".csv", "").split("_id_")
    label = parts[0] if len(parts) == 2 else "unknown"

    print(f"Using video: {os.path.basename(video_path)}")
    print(f"Processing ID {tracker_id:02d} ({label})")
    print(f"Pose detection params: conf={conf}, iou={iou}")

    # Continue with shared processing function
    return _process_pose_from_csv(
        tracking_dir,
        video_path,
        csv_file,
        label,
        video_basename,
        pose_output_dir,
        device,
        pose_model_name,
        conf,
        iou,
        tracker_id=tracker_id,
    )


def _process_pose_from_csv(
    tracking_dir,
    video_path,
    csv_file,
    label,
    video_basename,
    pose_output_dir,
    device,
    pose_model_name,
    conf,
    iou,
    tracker_id=None,
):
    """
    Shared function to process pose estimation from a CSV file.
    Used by both single ID and merged CSV processing.

    Args:
        tracking_dir: Directory containing tracking results
        video_path: Path to the video file
        csv_file: Path to the CSV file (single ID or merged)
        label: Label for the tracking data
        video_basename: Basename of the video (without extension)
        pose_output_dir: Directory where pose outputs will be saved
        device: Device to use for pose estimation
        pose_model_name: Name of the pose model to use
        conf: Confidence threshold for pose detection
        iou: IoU threshold for pose detection
        tracker_id: Optional tracker ID (None for merged CSV)
    """

    # Load pose model
    # Models are downloaded to vaila/models/ directory
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    pose_model_path = os.path.join(models_dir, pose_model_name)

    # Download model if needed
    if not os.path.exists(pose_model_path):
        try:
            print(f"Downloading pose model {pose_model_name}...")
            current_dir = os.getcwd()
            os.chdir(models_dir)
            YOLO(pose_model_name)
            os.chdir(current_dir)
            print("Model downloaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download pose model: {str(e)}")
            return False

    # Initialize Hardware Manager
    hw = HardwareManager()

    try:
        # Check and auto-export optimized .engine model
        pose_model_path = hw.auto_export(pose_model_name)
        pose_model = YOLO(pose_model_path)
        print(f"Pose model loaded: {pose_model_name}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load pose model: {str(e)}")
        return False

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video:\n{video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.6f} FPS")

    # Read tracking CSV
    df = pd.read_csv(csv_file)

    # Create output CSV and video for pose keypoints using basename
    if tracker_id is not None:
        pose_csv = os.path.join(pose_output_dir, f"{video_basename}_id_{tracker_id:02d}_pose.csv")
        pose_video = os.path.join(pose_output_dir, f"{video_basename}_id_{tracker_id:02d}_pose.mp4")
        output_tracker_id = tracker_id
    else:
        # For merged CSV, use "merged" as ID
        pose_csv = os.path.join(pose_output_dir, f"{video_basename}_{label}_merged_pose.csv")
        pose_video = os.path.join(pose_output_dir, f"{video_basename}_{label}_merged_pose.mp4")
        output_tracker_id = 1  # Use ID 1 for merged data

    # Initialize pose CSV with headers
    keypoint_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    pose_headers: list[str] = ["Frame", "Tracker_ID", "Label"]
    for kp_name in keypoint_names:
        pose_headers.extend([f"{kp_name}_x", f"{kp_name}_y", f"{kp_name}_conf"])

    pose_data = []

    # Prepare video writer for skeleton overlay
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(pose_video, fourcc, fps if fps > 0 else 25.0, (frame_w, frame_h))

    # Process each frame
    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Get bbox for this frame from CSV
        frame_data = df[df["Frame"] == frame_idx]

        if not frame_data.empty and pd.notna(frame_data.iloc[0]["X_min"]):
            x_min = int(frame_data.iloc[0]["X_min"])
            y_min = int(frame_data.iloc[0]["Y_min"])
            x_max = int(frame_data.iloc[0]["X_max"])
            y_max = int(frame_data.iloc[0]["Y_max"])

            # Extract ROI from frame
            roi = frame[y_min:y_max, x_min:x_max]

            if roi.size > 0:
                # Run pose estimation on ROI
                results = pose_model.predict(
                    roi,
                    conf=conf,
                    iou=iou,
                    device=device,
                    verbose=False,
                    show=False,
                    save=False,
                )

                # Extract keypoints
                keypoints_row = [frame_idx, output_tracker_id, label]

                if results and len(results) > 0 and results[0].keypoints is not None:
                    kp_data = results[0].keypoints.data
                    if kp_data is not None and len(kp_data) > 0:
                        if hasattr(kp_data, "cpu"):
                            kp_data = cast(Any, kp_data).cpu().numpy()
                        elif hasattr(kp_data, "numpy"):
                            kp_data = cast(Any, kp_data).numpy()

                        keypoints = kp_data[0]

                        for i, _ in enumerate(keypoint_names):
                            if i < len(keypoints):
                                kp = keypoints[i]
                                kp_x = float(kp[0]) + x_min
                                kp_y = float(kp[1]) + y_min
                                kp_conf = float(kp[2]) if len(kp) > 2 else 1.0
                                keypoints_row.extend([kp_x, kp_y, kp_conf])
                            else:
                                keypoints_row.extend([np.nan, np.nan, np.nan])
                    else:
                        for _ in keypoint_names:
                            keypoints_row.extend([np.nan, np.nan, np.nan])
                else:
                    for _ in keypoint_names:
                        keypoints_row.extend([np.nan, np.nan, np.nan])

                pose_data.append(keypoints_row)

                # Draw skeleton on full frame using absolute keypoints
                abs_kps = []
                for i in range(len(keypoint_names)):
                    base = 3 * i
                    if base + 2 < len(keypoints_row):
                        abs_kps.append(
                            (
                                keypoints_row[3 + base],
                                keypoints_row[3 + base + 1],
                                keypoints_row[3 + base + 2],
                            )
                        )
                frame = _draw_keypoints_and_skeleton(frame, abs_kps, color=(0, 255, 0))
            else:
                keypoints_row = [frame_idx, output_tracker_id, label]
                for _ in keypoint_names:
                    keypoints_row.extend([np.nan, np.nan, np.nan])
                pose_data.append(keypoints_row)

            writer.write(frame)
        else:
            keypoints_row = [frame_idx, output_tracker_id, label]
            for _ in keypoint_names:
                keypoints_row.extend([np.nan, np.nan, np.nan])
            pose_data.append(keypoints_row)

        frame_idx += 1

        if frame_idx % 20 == 0:
            print(f"Processing frame {frame_idx}/{total_frames}", end="\r")

    cap.release()
    writer.release()

    # Save pose CSV (columns: list[str] is valid at runtime; cast for strict stubs)
    pose_df = pd.DataFrame(pose_data, columns=cast(Any, pose_headers))
    pose_df.to_csv(pose_csv, index=False)
    print(f"\nPose data saved: {os.path.basename(pose_csv)}")
    print(f"Pose video saved: {os.path.basename(pose_video)}")

    print("\n" + "=" * 80)
    print("POSE ESTIMATION COMPLETED!")
    print("=" * 80)

    if tracker_id is not None:
        completion_msg = f"Pose estimation completed for ID {tracker_id:02d}!\n\n"
    else:
        completion_msg = "Pose estimation completed for merged CSV!\n\n"

    messagebox.showinfo(
        "Pose Estimation Completed", completion_msg + f"Results saved in:\n{pose_csv}"
    )
    return True


def process_pose_in_bboxes(tracking_dir, device=None, pose_model_name="yolo26n-pose.pt"):
    """
    Process pose estimation within bounding boxes from tracking results.

    Args:
        tracking_dir: Directory containing tracking results (CSVs and processed video)
        device: Device to use for pose estimation (None for auto-detect, "cuda" or "cpu")
        pose_model_name: Name of the pose model to use (default: yolo26n-pose.pt)

    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("POSE ESTIMATION WITHIN TRACKED BBOXES")
    print("=" * 80)

    # Auto-detect device if not specified
    if device is None:
        device = detect_optimal_device()
    print(f"Using device: {device}")
    # Create pose output directory with timestamp
    pose_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_output_dir = os.path.join(tracking_dir, f"pose_{pose_timestamp}")
    os.makedirs(pose_output_dir, exist_ok=True)

    # Find tracking CSV files
    csv_files = glob.glob(os.path.join(tracking_dir, "*_id_*.csv"))
    if not csv_files:
        print(f"Error: No tracking CSV files found in {tracking_dir}")
        messagebox.showerror("Error", f"No tracking CSV files found in:\n{tracking_dir}")
        return False

    # Find video (processed_* preferred, else any; allow manual pick)
    video_path = pick_video_for_pose(tracking_dir)
    if not video_path:
        print(f"Error: No video found in {tracking_dir}")
        messagebox.showerror("Error", f"No video found in:\n{tracking_dir}")
        return False
    print(f"Using video: {os.path.basename(video_path)}")
    print(f"Found {len(csv_files)} tracking CSV files")

    # Load pose model
    # Models are downloaded to vaila/models/ directory
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    pose_model_path = os.path.join(models_dir, pose_model_name)

    # Download model if needed
    if not os.path.exists(pose_model_path):
        try:
            print(f"Downloading pose model {pose_model_name}...")
            current_dir = os.getcwd()
            os.chdir(models_dir)
            YOLO(pose_model_name)
            os.chdir(current_dir)
            print(f"Model downloaded successfully to {pose_model_path}")
        except Exception as e:
            print(f"Error downloading pose model: {e}")
            messagebox.showerror("Error", f"Failed to download pose model: {str(e)}")
            return False

    # Initialize Hardware Manager
    hw = HardwareManager()

    try:
        # Check and auto-export optimized .engine model
        pose_model_path = hw.auto_export(pose_model_name)
        pose_model = YOLO(pose_model_path)
        print(f"Pose model loaded: {pose_model_name}")
    except Exception as e:
        print(f"Error loading pose model: {e}")
        messagebox.showerror("Error", f"Failed to load pose model: {str(e)}")
        return False

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        messagebox.showerror("Error", f"Could not open video:\n{video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames @ {fps:.6f} FPS")

    # Process each tracking CSV file (each ID)
    for csv_file in csv_files:
        try:
            # Parse CSV filename to get label and ID
            filename = os.path.basename(csv_file)
            # Format: {label}_id_{id:02d}.csv
            parts = filename.replace(".csv", "").split("_id_")
            if len(parts) != 2:
                print(f"Skipping {filename}: invalid format")
                continue

            label = parts[0]
            tracker_id = int(parts[1])

            print(f"\nProcessing ID {tracker_id} ({label})...")

            # Read tracking CSV
            df = pd.read_csv(csv_file)

            # Create output CSV for pose keypoints
            pose_csv = os.path.join(pose_output_dir, f"{label}_id_{tracker_id:02d}_pose.csv")
            pose_video = os.path.join(pose_output_dir, f"{label}_id_{tracker_id:02d}_pose.mp4")

            # Initialize pose CSV with headers
            # YOLO pose has 17 keypoints (COCO format)
            keypoint_names = [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]

            pose_headers = ["Frame", "Tracker_ID", "Label"]  # type: list[str]
            for kp_name in keypoint_names:
                pose_headers.extend([f"{kp_name}_x", f"{kp_name}_y", f"{kp_name}_conf"])

            pose_data = []

            # Prepare pose video writer for this ID
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
            frame_idx = 0
            writer = cv2.VideoWriter(
                pose_video,
                cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
                fps if fps > 0 else 25.0,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )

            while frame_idx < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get bbox for this frame from CSV
                frame_data = df[df["Frame"] == frame_idx]

                if not frame_data.empty and pd.notna(frame_data.iloc[0]["X_min"]):
                    x_min = int(frame_data.iloc[0]["X_min"])
                    y_min = int(frame_data.iloc[0]["Y_min"])
                    x_max = int(frame_data.iloc[0]["X_max"])
                    y_max = int(frame_data.iloc[0]["Y_max"])

                    # Extract ROI from frame
                    roi = frame[y_min:y_max, x_min:x_max]

                    if roi.size > 0:
                        # Run pose estimation on ROI
                        results = pose_model.predict(
                            roi,
                            conf=0.10,
                            iou=0.70,
                            device=device,
                            verbose=False,
                            show=False,
                            save=False,
                        )

                        # Extract keypoints (use first detection if multiple)
                        keypoints_row = [frame_idx, tracker_id, label]

                        if results and len(results) > 0 and results[0].keypoints is not None:
                            # Get keypoints tensor and convert to numpy
                            kp_data = results[0].keypoints.data
                            if kp_data is not None and len(kp_data) > 0:
                                # Convert tensor to numpy if needed
                                if hasattr(kp_data, "cpu"):
                                    kp_data = cast(Any, kp_data).cpu().numpy()
                                elif hasattr(kp_data, "numpy"):
                                    kp_data = cast(Any, kp_data).numpy()

                                keypoints = kp_data[0]  # First person detected

                                # YOLO pose returns [num_keypoints, 3] where 3 = [x, y, conf]
                                for i, _ in enumerate(keypoint_names):
                                    if i < len(keypoints):
                                        kp = keypoints[i]
                                        # Keypoints are relative to ROI, convert to absolute coordinates
                                        kp_x = float(kp[0]) + x_min
                                        kp_y = float(kp[1]) + y_min
                                        kp_conf = float(kp[2]) if len(kp) > 2 else 1.0
                                        keypoints_row.extend([kp_x, kp_y, kp_conf])
                                    else:
                                        keypoints_row.extend([np.nan, np.nan, np.nan])
                            else:
                                # No keypoints detected
                                for _ in keypoint_names:
                                    keypoints_row.extend([np.nan, np.nan, np.nan])
                        else:
                            # No pose detected, fill with NaN
                            for _ in keypoint_names:
                                keypoints_row.extend([np.nan, np.nan, np.nan])

                        pose_data.append(keypoints_row)

                        # Draw skeleton on full frame using absolute keypoints
                        abs_kps = []
                        for i in range(len(keypoint_names)):
                            base = 3 * i
                            if base + 2 < len(keypoints_row):
                                abs_kps.append(
                                    (
                                        keypoints_row[3 + base],
                                        keypoints_row[3 + base + 1],
                                        keypoints_row[3 + base + 2],
                                    )
                                )
                        frame = _draw_keypoints_and_skeleton(
                            frame, abs_kps, color=get_color_for_id(tracker_id)
                        )
                    else:
                        # Empty ROI, fill with NaN
                        keypoints_row = [frame_idx, tracker_id, label]
                        for _ in keypoint_names:
                            keypoints_row.extend([np.nan, np.nan, np.nan])
                        pose_data.append(keypoints_row)
                else:
                    # No bbox for this frame, fill with NaN
                    keypoints_row = [frame_idx, tracker_id, label]
                    for _ in keypoint_names:
                        keypoints_row.extend([np.nan, np.nan, np.nan])
                    pose_data.append(keypoints_row)

                frame_idx += 1

                if frame_idx % 20 == 0:
                    print(f"  Frame {frame_idx}/{total_frames}", end="\r")
                writer.write(frame)

            # Save pose CSV (columns: list[str] valid at runtime; cast for strict stubs)
            pose_df = pd.DataFrame(pose_data, columns=cast(Any, pose_headers))
            pose_df.to_csv(pose_csv, index=False)
            print(f"\n  Pose data saved: {os.path.basename(pose_csv)}")
            print(f"  Pose video saved: {os.path.basename(pose_video)}")
            writer.release()

        except Exception as e:
            print(f"\nError processing {csv_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    cap.release()
    print("\n" + "=" * 80)
    print("POSE ESTIMATION COMPLETED!")
    print("=" * 80)
    messagebox.showinfo(
        "Pose Estimation Completed",
        f"Pose estimation completed successfully!\n\n"
        f"Results saved in:\n{tracking_dir}\n\n"
        f"Look for *_pose.csv files for each tracked ID.",
    )
    return True


def run_yolov26track():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    print("Starting yolov26track.py...")
    print("-" * 80)

    # Print hardware information
    print("=" * 60)
    print("HARDWARE CONFIGURATION")
    print("=" * 60)
    print(get_hardware_info())
    print("=" * 60)
    root = tk.Tk()
    root.withdraw()

    # Select directories
    video_dir = filedialog.askdirectory(title="Select Input Directory")
    if not video_dir:
        return

    output_base_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_base_dir:
        return

    # Create a single parent directory for all results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_base_dir, f"vailatracker_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Select model using ModelSelectorDialog
    model_dialog = ModelSelectorDialog(root, title="Select YOLO Model")
    if not model_dialog.result:
        return
    model_name = model_dialog.result

    # Select tracker using TrackerSelectorDialog
    tracker_dialog = TrackerSelectorDialog(root, title="Select Tracking Method")
    if not hasattr(tracker_dialog, "result"):
        return
    tracker_name = tracker_dialog.result

    # Handle model path based on whether it's a custom model or pre-trained
    if os.path.isabs(model_name) or model_name.startswith("./") or model_name.startswith("../"):
        # Custom model - use the path directly
        model_path = model_name
        print(f"Using custom model: {model_path}")

        # Validate custom model file
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Custom model file not found: {model_path}")
            return
    else:
        # Pre-trained model - build the path in models directory
        # Models are downloaded to vaila/models/ directory
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_name)

        # Download the model if it doesn't exist
        if not os.path.exists(model_path):
            try:
                print(f"Downloading model {model_name}...")
                current_dir = os.getcwd()
                os.chdir(models_dir)
                YOLO(model_name)
                os.chdir(current_dir)
                print(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to download model: {str(e)}")
                return

    # Get configuration using TrackerConfigDialog
    config_dialog = TrackerConfigDialog(root, title="Tracker Configuration")
    if not hasattr(config_dialog, "result"):
        return

    config = config_dialog.result

    # Print device information
    device = config["device"]
    print(f"\nSelected device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device == "mps":
        print("GPU: Apple Silicon (MPS/Metal)")
        print(f"CPU cores: {os.cpu_count()}")
    else:
        print(f"CPU cores: {os.cpu_count()}")

    # Initialize the YOLO model
    # Use HardwareManager to get optimal model for this GPU
    hw = HardwareManager()
    hw.print_report()

    try:
        # Auto-export if needed (creates .engine optimized for this GPU)
        model_path = hw.auto_export(model_name)
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return

    # Select classes for tracking
    class_dialog = ClassSelectorDialog(root, title="Select Classes for Tracking")
    if not hasattr(class_dialog, "result"):
        return

    target_classes = class_dialog.result

    # Count videos to process
    video_count = 0
    processed_count = 0

    # Process each video in the directory
    for video_file in os.listdir(video_dir):
        if video_file.endswith((".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")):
            video_count += 1
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Create a subdirectory for this specific video
            output_dir = os.path.join(main_output_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            # Read the dimensions and total frames of the original video
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_raw = cap.get(cv2.CAP_PROP_FPS)
            # Handle fractional FPS (common in some video formats)
            fps = float(fps_raw) if fps_raw > 0 else 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Specify temporary AVI path (MJPG codec is very reliable)
            temp_avi_path = os.path.join(output_dir, f"processed_{video_name}_temp.avi")
            # Final MP4 path
            out_video_path = os.path.join(output_dir, f"processed_{video_name}.mp4")

            # Use MJPG codec for AVI (highly compatible and reliable)
            # This ensures the video is written correctly without corruption
            writer = cv2.VideoWriter(
                temp_avi_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)  # type: ignore
            )
            if not writer.isOpened():
                print(f"Error creating video file: {temp_avi_path}")
                print("Trying alternative codec...")
                # Fallback to XVID if MJPG fails
                print("[WARNING] MJPG failed. Trying XVID...")
                writer = cv2.VideoWriter(
                    temp_avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height)  # type: ignore
                )
                if not writer.isOpened():
                    print("Error: Could not create video writer with any codec")
                    continue

            # Load ROI from saved TOML file if available
            roi_poly = None
            roi_file_path = config.get("roi_file")
            if isinstance(roi_file_path, (str, Path)) and os.path.exists(roi_file_path):
                print(f"Loading ROI from file: {os.path.basename(roi_file_path)}")
                try:
                    roi_poly = load_roi_from_toml(roi_file_path)
                    if roi_poly is not None and len(roi_poly) >= 3:
                        print("ROI loaded successfully. Detection limited to the polygon area.")
                    else:
                        print("ROI file invalid or empty. Using full frame.")
                        roi_poly = None
                except Exception as e:
                    print(f"Error loading ROI from file: {e}")
                    import traceback

                    traceback.print_exc()
                    print("Continuing with full frame processing...")
                    roi_poly = None
            elif roi_file_path:
                print(f"ROI file not found: {roi_file_path}")
                print("Continuing with full frame processing...")

            # Prepare ROI mask if loaded
            mask_img = None
            if roi_poly is not None:
                mask_img = np.zeros((height, width), dtype=np.uint8)
                cv2.fillPoly(mask_img, [roi_poly], 255)

            # Try to use the default Ultralytics tracker config first, then customize
            tracker_config = None
            try:
                # Option 1: Try to load default tracker config from Ultralytics
                # Use importlib.resources or fallback to pkg_resources for macOS compatibility
                ultralytics_path = None
                try:
                    # Try importlib.resources first (Python 3.9+, recommended)
                    try:
                        from importlib.resources import files

                        ultralytics_path = str(files("ultralytics") / "cfg" / "trackers")
                    except (ImportError, ModuleNotFoundError, AttributeError, TypeError):
                        # Fallback: try pkg_resources (deprecated but still works)
                        if pkg_resources is not None:
                            with contextlib.suppress(Exception):
                                ultralytics_path = pkg_resources.resource_filename(
                                    "ultralytics", "cfg/trackers"
                                )
                except Exception:
                    pass
                # If we couldn't find the path, skip to Option 2 (create from scratch)
                yaml_path = None
                if ultralytics_path and os.path.exists(ultralytics_path):
                    tracker_yaml = f"{tracker_name}.yaml"
                    yaml_path = os.path.join(ultralytics_path, tracker_yaml)

                if yaml_path and os.path.exists(yaml_path):
                    # Read the default config
                    with open(yaml_path) as f:
                        tracker_cfg = yaml.safe_load(f)

                    # Apply optimizations
                    tracker_cfg["track_buffer"] = 60  # Increased for better occlusion handling

                    # BoT-SORT specific optimizations
                    if tracker_name == "botsort":
                        tracker_cfg["gmc_method"] = "sparseOptFlow"
                        tracker_cfg["with_reid"] = True
                        tracker_cfg["proximity_thresh"] = 0.5
                        tracker_cfg["appearance_thresh"] = 0.25
                        print("BoT-SORT configured with GMC and ReID for maximum robustness")

                    # Save customized config
                    trackers_dir = os.path.join(models_dir, "trackers")
                    os.makedirs(trackers_dir, exist_ok=True)
                    custom_yaml = os.path.join(trackers_dir, f"{tracker_name}_custom.yaml")

                    with open(custom_yaml, "w") as f:
                        yaml.dump(tracker_cfg, f, default_flow_style=False, sort_keys=False)

                    tracker_config = custom_yaml
                    print(
                        f"Using optimized tracker config based on Ultralytics default: {tracker_config}"
                    )
                else:
                    # Option 2: Create config from scratch with all required fields
                    trackers_dir = os.path.join(models_dir, "trackers")
                    os.makedirs(trackers_dir, exist_ok=True)
                    custom_yaml = os.path.join(trackers_dir, f"{tracker_name}_custom.yaml")

                    # Create complete tracker config with all required fields
                    tracker_cfg = {
                        "tracker_type": tracker_name,
                        "track_high_thresh": 0.5,
                        "track_low_thresh": 0.1,
                        "new_track_thresh": 0.6,
                        "track_buffer": 60,
                        "match_thresh": 0.8,
                        "fuse_score": True,  # Required field that was missing
                    }

                    # BoT-SORT specific settings
                    if tracker_name == "botsort":
                        tracker_cfg["gmc_method"] = "sparseOptFlow"
                        tracker_cfg["with_reid"] = True
                        tracker_cfg["proximity_thresh"] = 0.5
                        tracker_cfg["appearance_thresh"] = 0.25
                        print("BoT-SORT configured with GMC and ReID for maximum robustness")

                    with open(custom_yaml, "w") as f:
                        yaml.dump(tracker_cfg, f, default_flow_style=False, sort_keys=False)

                    tracker_config = custom_yaml
                    print(f"Using custom tracker config: {tracker_config}")

            except Exception as e:
                print(f"Warning: Could not load default tracker config: {e}")
                print("Falling back to using tracker name directly")
                # Fallback: use tracker name directly (Ultralytics will use defaults)
                tracker_config = tracker_name

            print(f"Tracking with {tracker_name} (YOLO built-in tracker)")

            tracker_csv_files = {}
            # Map raw tracker IDs (from YOLO) to sequential per-label IDs starting at 1
            label_to_raw2seq = {}
            label_to_next = {}

            results = model.track(
                source=video_path,
                conf=config["conf"],
                iou=config["iou"],
                device=config["device"],
                vid_stride=config["vid_stride"],
                save=False,
                stream=True,
                persist=True,
                tracker=tracker_config,
                classes=target_classes,
                verbose=False,
            )

            frame_idx = 0
            for result in results:
                frame = result.orig_img

                # Overlay ROI outline for reference
                if roi_poly is not None:
                    cv2.polylines(frame, [roi_poly], True, (255, 255, 0), 2)

                boxes = result.boxes if result.boxes is not None else getattr(result, "obbs", None)

                if boxes is None:
                    writer.write(frame)
                    frame_idx += 1
                    continue

                for box in boxes:
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    raw_id = int(box.id[0]) if box.id is not None else -1
                    class_id = int(box.cls[0].item()) if box.cls is not None else -1
                    label = model.names.get(class_id, "unknown")

                    if raw_id < 0:
                        continue

                    # If ROI is defined, skip detections whose center is outside the polygon
                    if mask_img is not None:
                        cx = (x_min + x_max) // 2
                        cy = (y_min + y_max) // 2
                        inside = cv2.pointPolygonTest(
                            cast(Any, roi_poly), [float(cx), float(cy)], False
                        )
                        if inside < 0:
                            continue

                    if label not in label_to_raw2seq:
                        label_to_raw2seq[label] = {}
                        label_to_next[label] = 1
                    if raw_id not in label_to_raw2seq[label]:
                        label_to_raw2seq[label][raw_id] = label_to_next[label]
                        label_to_next[label] += 1
                    tracker_id = label_to_raw2seq[label][raw_id]

                    color = get_color_for_id(tracker_id)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(
                        frame,
                        f"id {tracker_id} {label}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    key = (tracker_id, label)
                    if key not in tracker_csv_files:
                        tracker_csv_files[key] = initialize_csv(
                            output_dir,
                            label,
                            tracker_id,
                            total_frames,
                        )
                    update_csv(
                        tracker_csv_files[key],
                        frame_idx,
                        tracker_id,
                        label,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        conf,
                    )

                writer.write(frame)

                if frame_idx % 20 == 0:
                    print(f"Processing frame {frame_idx}/{total_frames}", end="\r")

                frame_idx += 1

            writer.release()
            print("")  # newline after progress

            # Verify the AVI file was written successfully
            if not os.path.exists(temp_avi_path) or os.path.getsize(temp_avi_path) == 0:
                print(f"Error: Video file {temp_avi_path} was not created or is empty")
                continue

            # Convert AVI to MP4 using FFmpeg with robust settings for VLC compatibility
            print("Converting video to MP4 format for maximum compatibility...")
            try:
                # Use subprocess for better security and error handling
                # These FFmpeg parameters ensure VLC and other players can read the file
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",  # Overwrite output file if it exists
                    "-i",
                    temp_avi_path,
                    "-c:v",
                    "libx264",  # H.264 codec (universal compatibility)
                    "-preset",
                    "medium",  # Balance between speed and compression
                    "-crf",
                    "23",  # Quality (lower = better, 23 is good default)
                    "-pix_fmt",
                    "yuv420p",  # Pixel format (required for compatibility)
                    "-movflags",
                    "+faststart",  # Enable fast start for web playback
                    "-c:a",
                    "aac",  # Audio codec (if audio exists)
                    "-b:a",
                    "128k",  # Audio bitrate
                    "-strict",
                    "-2",  # Allow experimental codecs if needed
                    out_video_path,
                ]

                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False)

                if result.returncode == 0:
                    # Verify output file exists and has content
                    if os.path.exists(out_video_path) and os.path.getsize(out_video_path) > 0:
                        os.remove(temp_avi_path)
                        print(f"OK - Video converted successfully: {out_video_path}")
                        print(
                            f"  File size: {os.path.getsize(out_video_path) / (1024 * 1024):.2f} MB"
                        )
                    else:
                        print("Warning: FFmpeg completed but output file is missing or empty")
                        if os.path.exists(temp_avi_path):
                            print(f"Keeping temporary AVI file: {temp_avi_path}")
                else:
                    print(f"FFmpeg conversion error (return code {result.returncode}):")
                    if result.stderr:
                        print(f"  {result.stderr[:500]}")  # Print first 500 chars of error
                    # Keep temp AVI file if conversion fails
                    if os.path.exists(temp_avi_path):
                        print(f"Keeping temporary AVI file: {temp_avi_path}")
                        # Optionally rename AVI to final name
                        avi_final = out_video_path.replace(".mp4", ".avi")
                        try:
                            os.rename(temp_avi_path, avi_final)
                            print(f"Saved as AVI instead: {avi_final}")
                        except Exception:
                            pass

            except FileNotFoundError:
                print("Error: FFmpeg not found. Please install FFmpeg.")
                print(f"Keeping temporary AVI file: {temp_avi_path}")
            except Exception as e:
                print(f"Error in video conversion: {str(e)}")
                import traceback

                traceback.print_exc()
                # Keep temp AVI file if conversion fails
                if os.path.exists(temp_avi_path):
                    print(f"Keeping temporary AVI file: {temp_avi_path}")

            print(f"Processing completed for {video_file}. Results saved in '{output_dir}'.")

            # Create combined wide CSV and merged wide per-label CSV(s) after processing each video
            combined_csv = create_combined_detection_csv(output_dir)
            if combined_csv:
                print(f"Combined detection tracking file created: {combined_csv}")

            merged_csv = create_merged_detection_csv(output_dir, total_frames)
            if merged_csv:
                print(f"Merged detection tracking file created: {merged_csv}")

            processed_count += 1

    # Show completion message
    if video_count == 0:
        print("\n" + "=" * 80)
        print("WARNING: No video files found in the input directory!")
        print("=" * 80 + "\n")
        messagebox.showwarning(
            "No Videos Found",
            "No video files (.mp4, .avi, .mov, .mkv) were found in the input directory.\n\n"
            f"Input directory: {video_dir}",
        )
    else:
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Videos processed: {processed_count} / {video_count}")
        print(f"All results saved in: {main_output_dir}")
        print("=" * 80 + "\n")

        # Create custom completion dialog with pose estimation button
        completion_dialog = tk.Toplevel(root)
        completion_dialog.title("Processing Completed")
        completion_dialog.geometry("500x300")
        completion_dialog.transient(root)
        completion_dialog.grab_set()

        # Center the window
        completion_dialog.update_idletasks()
        x = (completion_dialog.winfo_screenwidth() // 2) - (completion_dialog.winfo_width() // 2)
        y = (completion_dialog.winfo_screenheight() // 2) - (completion_dialog.winfo_height() // 2)
        completion_dialog.geometry(f"+{x}+{y}")

        # Message
        msg_frame = tk.Frame(completion_dialog, padx=20, pady=20)
        msg_frame.pack(fill="both", expand=True)

        tk.Label(
            msg_frame,
            text="All videos have been processed successfully!",
            font=("Arial", 12, "bold"),
        ).pack(pady=(0, 10))

        tk.Label(
            msg_frame,
            text=f"Videos processed: {processed_count} / {video_count}\n\n"
            f"Results saved in:\n{main_output_dir}",
            justify="left",
        ).pack(pady=10)

        # Buttons frame
        buttons_frame = tk.Frame(completion_dialog, padx=20, pady=20)
        buttons_frame.pack(fill="x")

        def run_pose_estimation():
            completion_dialog.destroy()
            # Ask user to select a tracking directory
            tracking_dir = filedialog.askdirectory(
                title="Select Tracking Directory", initialdir=main_output_dir
            )
            if tracking_dir:
                # Simple pose model selection dialog
                pose_models = [
                    ("yolo26n-pose.pt", "Pose - Nano (fastest)"),
                    ("yolo26s-pose.pt", "Pose - Small"),
                    ("yolo26m-pose.pt", "Pose - Medium"),
                    ("yolo26l-pose.pt", "Pose - Large"),
                    ("yolo26x-pose.pt", "Pose - XLarge (most accurate)"),
                ]

                pose_dialog = tk.Toplevel(root)
                pose_dialog.title("Select Pose Model")
                pose_dialog.geometry("400x300")
                pose_dialog.transient(root)
                pose_dialog.grab_set()

                # Center window
                pose_dialog.update_idletasks()
                x = (pose_dialog.winfo_screenwidth() // 2) - (pose_dialog.winfo_width() // 2)
                y = (pose_dialog.winfo_screenheight() // 2) - (pose_dialog.winfo_height() // 2)
                pose_dialog.geometry(f"+{x}+{y}")

                tk.Label(
                    pose_dialog, text="Select Pose Estimation Model:", font=("Arial", 10, "bold")
                ).pack(pady=10)

                listbox = tk.Listbox(pose_dialog, width=50, height=8)
                for model, desc in pose_models:
                    listbox.insert(tk.END, f"{model} - {desc}")
                listbox.pack(padx=10, pady=10, fill="both", expand=True)
                listbox.selection_set(0)  # Select first by default

                selected_model = [None]

                def confirm():
                    if listbox.curselection():
                        selection = listbox.get(listbox.curselection())
                        selected_model[0] = selection.split(" - ")[0]
                    pose_dialog.destroy()

                btn_frame = tk.Frame(pose_dialog)
                btn_frame.pack(pady=10)

                tk.Button(
                    btn_frame, text="OK", command=confirm, bg="#4CAF50", fg="white", padx=20
                ).pack(side="left", padx=5)

                tk.Button(btn_frame, text="Cancel", command=pose_dialog.destroy, padx=20).pack(
                    side="left", padx=5
                )

                pose_dialog.wait_window()

                if selected_model[0]:
                    # Run pose estimation
                    process_pose_in_bboxes(
                        tracking_dir,
                        device=config.get("device", "cpu"),
                        pose_model_name=selected_model[0],
                    )

        def close_dialog():
            completion_dialog.destroy()
            root.destroy()

        # Pose estimation button
        btn_pose = tk.Button(
            buttons_frame,
            text="Run Pose Estimation on Tracked IDs",
            command=run_pose_estimation,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            pady=5,
        )
        btn_pose.pack(side="left", padx=5, fill="x", expand=True)

        # Close button
        btn_close = tk.Button(
            buttons_frame,
            text="Close",
            command=close_dialog,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10),
            padx=10,
            pady=5,
        )
        btn_close.pack(side="left", padx=5, fill="x", expand=True)

        # Wait for dialog to close
        completion_dialog.wait_window()

    root.destroy()


if __name__ == "__main__":
    # Launch GUI-based tracker application
    try:
        run_yolov26track()
    except KeyboardInterrupt:
        print("\n\nTracking interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
