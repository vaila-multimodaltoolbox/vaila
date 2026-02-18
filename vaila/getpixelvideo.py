"""
================================================================================
Pixel Coordinate Tool - getpixelvideo.py
================================================================================
vailá - Multimodal Toolbox
Authors: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 22 July 2025
Update: 3 February 2026
Version: 0.3.19
Python Version: 3.12.12

Description:
------------
This tool enables marking and saving pixel coordinates in video frames, with
zoom functionality for precise annotations. The window can now be resized dynamically,
and all UI elements adjust accordingly. Users can navigate the video frames, mark
points, and save results in CSV format.

New Features in This Version 0.3.17:
- Swap landmarks using file basename_swap.toml or Shift+W to load external config.
- Swap GUI to select start frame, end frame, marker 1 ID, and marker 2 ID.
- ClickPass Mode: Auto-advance frame after marking
- Zoom on Scroll: Use mouse wheel to zoom in/out
- Playback Speed: Use [ and ] to adjust speed
- Help Button: '?' button opens documentation
GUI Pygame in Linux to avoid conflicts with Tkinter.

New Features in This Version 0.3.14:
------------------------------
1. Button "Labeling" to label images in video frames for Machine Learning training.
2. YOLO dataset directory support.
3. YOLO tracking CSV visualization (all_id_detection.csv format) with bounding boxes overlay.

How to use:
------------
1. Select the video file to process.
2. Select the keypoint file to load.
3. Mark points in the video frame.
4. Save the results in CSV format.
5. Labeling mode to label images in video frames for Machine Learning training.

python getpixelvideo.py

Help:
------------
python getpixelvideo.py -h
Usage: python getpixelvideo.py [options]

Options:
  -h, --help            show this help message and exit
  -v, --version         show version information and exit
  -f FILE, --file FILE  specify the video file to process
  -k KEYPOINT, --keypoint KEYPOINT  specify the keypoint file to load
  -l, --labeling        label images in video frames for Machine Learning training
  --dataset DIR         set dataset folder (next Save appends; multi-video)
  -s, --save            save the results in CSV format
  -p, --persistence     show persistence mode
  -a, --auto            show auto-marking mode
  -c, --sequential      show sequential mode
  -h, --help            show this help message and exit

Key Bindings (Labeling Mode Only - Press 'L' to toggle):
  N                     Rename current object label
  F5                    Save Labeling Project (JSON) / export dataset (dataset_YYYYMMDD_HHMMSS or append)
  F6                    Load Labeling Project (JSON)
  F7                    Load dataset folder (next Save appends; multi-video)
  F8                    Open another video (keep dataset; no need to close app)
  -v, --version         show version information and exit

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
Visit the project repository: https://github.com/vaila-multimodaltoolbox

================================================================================
"""

import io
import json
import os

# Configure SDL environment variables BEFORE importing pygame
# to prevent EGL/OpenGL warnings on Linux systems
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from contextlib import redirect_stderr, suppress
from pathlib import Path

from rich import print

if platform.system() == "Linux":
    os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ["SDL_RENDER_DRIVER"] = "software"
    # Additional variables to suppress OpenGL/EGL warnings
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "0"
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

    # Suppress stderr during imports to hide EGL warnings
    # These warnings come from both pygame and cv2 on some Linux systems
    f = io.StringIO()
    with redirect_stderr(f):
        import cv2
        import pygame

    # Import cv2 again normally to ensure it's available in the global scope
    import cv2
else:
    import cv2
    import pygame

# Optional imports for advanced features
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not found. Pose estimation features will be disabled.")

try:
    import ultralytics

    # Mute YOLO unnecessary logs
    ultralytics.checks = lambda: None
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics (YOLO) not found. Advanced pose estimation will be limited.")

import os
from datetime import datetime

import numpy as np
import pandas as pd

# Optional import for TOML support (Python 3.11+)
try:
    import tomllib

    TOML_AVAILABLE = True
except ImportError:
    try:
        import tomli as tomllib  # pyright: ignore[reportMissingImports]

        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False
        print("Warning: tomllib/tomli not found. TOML configuration features will be disabled.")

# Removed native_file_dialog imports - now using Tkinter directly for all dialogs


def get_color_for_id(marker_id):
    """Generate a consistent color for a given marker ID."""
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Red
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]
    return colors[marker_id % len(colors)]


def apply_swap_config(coordinates, swap_config):
    """
    Apply marker swapping based on configuration list.
    swap_config: list of dicts with keys: start_frame, end_frame, marker_1, marker_2
    """
    if not swap_config:
        return coordinates

    count = 0
    for rule in swap_config:
        start = rule.get("start_frame", 0)
        end = rule.get("end_frame", float("inf"))
        m1_idx = rule.get("marker_1")
        m2_idx = rule.get("marker_2")  # logic uses 0-indexed internally if these are indices

        # Validation
        if m1_idx is None or m2_idx is None:
            continue

        # User input is likely 1-based (Marker 1, Marker 2...), convert to 0-based for internal list
        # If config is from internal logic, assume 0-based.
        # Let's assume input is 0-based for internal consistency, GUI handles conversion.

        for frame_idx, markers in coordinates.items():
            if start <= frame_idx <= end:
                # Ensure we have enough markers
                if len(markers) > max(m1_idx, m2_idx):
                    # Check if markers exist at these indices (not None)
                    # The markers list is [(x,y), (x,y), ...]. Some might be None or placeholders?
                    # getpixelvideo structure seems to be list of tuples.

                    # Swap
                    try:
                        temp = markers[m1_idx]
                        markers[m1_idx] = markers[m2_idx]
                        markers[m2_idx] = temp
                        count += 1
                    except IndexError:
                        pass

    print(f"Applied {count} marker swaps across frames.")
    return coordinates


def load_swap_toml(toml_path):
    """Load swap configuration from TOML file."""
    if not TOML_AVAILABLE:
        print("TOML library not available.")
        return []

    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
            raw_rules = data.get("swap", [])
            # Convert 1-based (User/TOML) to 0-based (Internal)
            corrected_rules = []
            for r in raw_rules:
                try:
                    corrected_rules.append(
                        {
                            "start_frame": int(r.get("start_frame", 1)) - 1,
                            "end_frame": int(r.get("end_frame", 1)) - 1,
                            "marker_1": int(r.get("marker_1", 1)) - 1,
                            "marker_2": int(r.get("marker_2", 2)) - 1,
                        }
                    )
                except (ValueError, TypeError):
                    continue
            return corrected_rules
    except Exception as e:
        print(f"Error loading TOML: {e}")
        return []


def save_swap_toml(toml_path, swap_rules):
    """Save swap configuration to TOML file with 1-based indexing."""
    if not swap_rules:
        return

    try:
        # Prepare data: Convert 0-based (Internal) to 1-based (User/TOML)
        export_rules = []
        for r in swap_rules:
            export_rules.append(
                {
                    "start_frame": r["start_frame"] + 1,
                    "end_frame": r["end_frame"] + 1,
                    "marker_1": r["marker_1"] + 1,
                    "marker_2": r["marker_2"] + 1,
                }
            )

        # Manually construct TOML string to avoid external dependency for writing if possible
        # but tomli_w is standard. Since we only have 'tomllib' (read-only) in stdlib 3.11+,
        # and simple structure, let's write manually to avoid adding 'tomli-w' dependency if not needed.

        with open(toml_path, "w") as f:
            f.write("# Marker Swap Configuration\n")
            f.write("# Generated by vailá getpixelvideo.py\n\n")

            for r in export_rules:
                f.write("[[swap]]\n")
                f.write(f"start_frame = {r['start_frame']}\n")
                f.write(f"end_frame = {r['end_frame']}\n")
                f.write(f"marker_1 = {r['marker_1']}\n")
                f.write(f"marker_2 = {r['marker_2']}\n")
                f.write("\n")

        print(f"Swap config with {len(export_rules)} rules saved to {toml_path}")
        return True
    except Exception as e:
        print(f"Error saving TOML: {e}")
        return False


POSE_LANDMARKER = None
YOLO_MODEL = None

# MediaPipe Pose Connections (Standard 33-keypoint topology)
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


def download_or_load_yolo_model(model_name=None):
    """Download or load YOLO model for pose detection"""
    if not YOLO_AVAILABLE:
        return None

    # Default to pose model if not specified
    if model_name is None:
        model_name = "yolo11x-pose.pt"  # Extra large pose model for accuracy

    # Use vaila/models directory
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / model_name

    # Check if we need to download (Ultralytics handles downloads automatically if we pass the name,
    # but we want to control where it goes to keep it self-contained)

    # If the file exists at our path, load it directly
    if model_path.exists():
        try:
            # Load the model
            print(f"Loading YOLO model from {model_path}...")
            return YOLO(str(model_path))
        except Exception as e:
            print(f"Error loading local YOLO model: {e}")
            return None

    # If not, let YOLO download it (it usually puts it in current dir or cache)
    # To be safe and clean, we can try to download it ourselves or let YOLO do it and then move it?
    # Simpler: Just rely on YOLO's auto download which is robust.
    # However, to store it in our models folder, we can use the same logic as markerless2d

    try:
        print(f"Loading/Downloading YOLO model {model_name}...")
        # This triggers download if not found
        model = YOLO(model_name)

        # If we successfully loaded/downloaded, check if the file is in CWD and move it to models_dir
        cwd_model = Path.cwd() / model_name
        if cwd_model.exists() and not model_path.exists():
            print(f"Moving downloaded model to {models_dir}...")
            shutil.move(str(cwd_model), str(model_path))

        return model
    except Exception:
        return model
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return None


def detect_person_box(frame, model):
    """
    Detect the most prominent person in the frame using YOLO.
    Returns [x1, y1, x2, y2] or None.
    """
    if model is None:
        return None

    # Run inference
    # conf=0.4 is a reasonable threshold
    results = model(frame, classes=0, verbose=False, conf=0.4)

    if not results or len(results) == 0:
        return None

    boxes = results[0].boxes
    if not boxes or len(boxes) == 0:
        return None

    # Find the largest person (by area * conf) to likely be the main subject
    best_box = None
    best_score = -1

    for box in boxes:
        # box.xyxy is [x1, y1, x2, y2]
        coords = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())

        x1, y1, x2, y2 = coords
        area = (x2 - x1) * (y2 - y1)
        score = area * conf

        if score > best_score:
            best_score = score
            best_box = coords

    return best_box


def get_mediapipe_model_path(complexity=1):
    """Download the correct MediaPipe Tasks model based on complexity (0=Lite, 1=Full, 2=Heavy)"""
    # Use vaila/models directory for storing models (relative to this script)
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
        # Correct URLs for the models
        model_urls = {
            0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        }
        url = model_urls.get(complexity, model_urls[1])
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, str(model_path))
            print("Download completed!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    return str(model_path.resolve())


def detect_pose_mediapipe(frame):
    """
    Detect Pose Landmarks using MediaPipe + YOLO Strategy.

    Strategy:
    1. Detect person with YOLO (if available).
    2. Crop and Upscale (2x) the person ROI.
    3. Run MediaPipe Tasks API on the crop.
    4. Map coordinates back to original frame.

    Falls back to simple full-frame MediaPipe if YOLO is missing or detection fails.
    """
    global POSE_LANDMARKER, YOLO_MODEL

    if not MEDIAPIPE_AVAILABLE:
        return []

    # --- Step 1: Initialize Models ---

    # Initialize MediaPipe Landmarker if needed
    if hasattr(mp, "tasks") and POSE_LANDMARKER is None:
        try:
            model_path = get_mediapipe_model_path(complexity=1)
            if model_path:
                BaseOptions = mp.tasks.BaseOptions
                PoseLandmarker = mp.tasks.vision.PoseLandmarker
                PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                )
                POSE_LANDMARKER = PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing MediaPipe Landmarker: {e}")

    # Initialize YOLO Model if available and not loaded
    if YOLO_AVAILABLE and YOLO_MODEL is None:
        YOLO_MODEL = download_or_load_yolo_model()

    height, width = frame.shape[:2]

    # --- Step 2: YOLO Detection & Crop Prep ---

    person_box = None
    if YOLO_AVAILABLE and YOLO_MODEL is not None:
        person_box = detect_person_box(frame, YOLO_MODEL)

    run_on_crop = False
    crop_info = None  # (x1, y1, scale_w, scale_h)

    # Prepare input image for MediaPipe
    # If we have a person box, we crop and upscale
    if person_box is not None:
        x1, y1, x2, y2 = person_box

        # Add 20% margin
        w_box = x2 - x1
        h_box = y2 - y1
        margin_w = int(w_box * 0.2)
        margin_h = int(h_box * 0.2)

        crop_x1 = max(0, int(x1 - margin_w))
        crop_y1 = max(0, int(y1 - margin_h))
        crop_x2 = min(width, int(x2 + margin_w))
        crop_y2 = min(height, int(y2 + margin_h))

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        if crop_w > 10 and crop_h > 10:
            # Crop
            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Upscale 2x
            upscale_factor = 2.0
            target_w = int(crop_w * upscale_factor)
            target_h = int(crop_h * upscale_factor)

            try:
                upscaled_crop = cv2.resize(
                    crop_img, (target_w, target_h), interpolation=cv2.INTER_CUBIC
                )
                rgb_frame = cv2.cvtColor(upscaled_crop, cv2.COLOR_BGR2RGB)
                run_on_crop = True
                crop_info = (crop_x1, crop_y1, crop_w, crop_h, target_w, target_h)
            except Exception as e:
                print(f"Resize failed: {e}, falling back to full frame.")
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        # Full frame fallback
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    landmarks = []

    # --- Step 3: Inference (Tasks API preferred) ---
    if hasattr(mp, "tasks") and POSE_LANDMARKER:
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = POSE_LANDMARKER.detect(mp_image)

            if detection_result.pose_landmarks:
                pose_landmarks = detection_result.pose_landmarks[0]

                for landmark in pose_landmarks:
                    if run_on_crop and crop_info:
                        # Map back to original frame
                        # 1. Un-normalize in Upscaled Crop
                        c_x1, c_y1, cw, ch, tw, th = crop_info

                        px_in_upscale_x = landmark.x * tw
                        px_in_upscale_y = landmark.y * th

                        # 2. Downscale to Original Crop
                        px_in_crop_x = px_in_upscale_x / (tw / cw)
                        px_in_crop_y = px_in_upscale_y / (th / ch)

                        # 3. Add Offset
                        px = int(px_in_crop_x + c_x1)
                        py = int(px_in_crop_y + c_y1)
                    else:
                        # Standard full frame normalization
                        px = min(int(landmark.x * width), width - 1)
                        py = min(int(landmark.y * height), height - 1)

                    landmarks.append((px, py))
                return landmarks
        except Exception as e:
            print(f"Tasks API inference failed: {e}")
            # Fallthrough to legacy

    # --- Step 4: Legacy Fallback (mp.solutions) ---
    # Only if Tasks API failed or not available, and we are NOT running on crop
    # (Legacy API might expect full frame or might handle crop, but let's keep it simple: fallback is full frame)
    if not run_on_crop and hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        try:
            with mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            ) as pose:
                # Need to re-convert if we were trying crop logic logic above and failed mid-way
                # But here we are assuming full-frame RGB is available or we convert
                if "rgb_frame" not in locals() or run_on_crop:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    for landmark in results.pose_landmarks.landmark:
                        px = min(int(landmark.x * width), width - 1)
                        py = min(int(landmark.y * height), height - 1)
                        landmarks.append((px, py))
                    return landmarks
        except Exception as e:
            print(f"Legacy API failed: {e}")

    return []


# Removed pygame_file_dialog - now using Tkinter directly for file dialogs
# Tkinter dialogs block Pygame events to avoid conflicts
def pygame_file_dialog(
    initial_dir=None, file_extensions=None, restore_screen=None, restore_size=None
):
    """
    Native pygame file dialog for selecting files.
    Works seamlessly within pygame without conflicts.

    Args:
        initial_dir: Starting directory (defaults to home or current directory)
        file_extensions: List of extensions to filter (e.g., ['.csv'])
        restore_screen: Screen surface to restore after dialog (optional)
        restore_size: Tuple (width, height) to restore window size (optional)

    Returns:
        Selected file path or None if cancelled
    """
    if file_extensions is None:
        file_extensions = [".csv"]

    # Normalize extensions (ensure they start with .)
    file_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in file_extensions]
    file_extensions_lower = [ext.lower() for ext in file_extensions]

    # Determine initial directory
    if initial_dir is None:
        # Try common locations
        for test_dir in [os.path.expanduser("~"), os.getcwd(), "/media", "/home"]:
            if os.path.exists(test_dir) and os.path.isdir(test_dir):
                initial_dir = test_dir
                break
        if initial_dir is None:
            initial_dir = os.getcwd()

    current_dir = os.path.abspath(initial_dir)
    selected_file = None
    scroll_offset = 0
    selected_index = 0
    items = []

    # Function to reload directory contents
    def load_directory(directory):
        new_items = []
        # Add parent directory option
        if directory != os.path.dirname(directory):
            new_items.append(("..", True))  # True = is directory

        try:
            # Get files and directories
            dir_items = sorted(os.listdir(directory))
            for item in dir_items:
                item_path = os.path.join(directory, item)
                try:
                    is_dir = os.path.isdir(item_path)
                    # Show directories and files matching extensions
                    if is_dir or any(item.lower().endswith(ext) for ext in file_extensions_lower):
                        new_items.append((item, is_dir))
                except (OSError, PermissionError):
                    continue
        except (OSError, PermissionError):
            pass
        return new_items

    # Initial load
    items = load_directory(current_dir)

    # Save current display mode if provided
    if restore_size:
        old_size = restore_size
    else:
        old_size = (
            pygame.display.get_surface().get_size() if pygame.display.get_init() else (800, 600)
        )

    # Create dialog window
    dialog_width = 800
    dialog_height = 600
    dialog_screen = pygame.display.set_mode((dialog_width, dialog_height))
    pygame.display.set_caption("Select File")

    # Use Verdana for better legibility (l vs I)
    font = pygame.font.SysFont("verdana", 12)
    small_font = pygame.font.SysFont("verdana", 12)

    running = True
    text_input = ""
    input_active = False
    last_click_time = 0
    last_click_index = -1
    double_click_delay = 500  # milliseconds

    # Layout constants for file list vs buttons (computed once, used in draw + events)
    list_y = 110
    item_height = 25
    button_instruction_height = 85  # space for buttons + instructions below list
    list_height = dialog_height - list_y - button_instruction_height
    visible_count = max(1, (list_height - 10) // item_height)

    while running:
        # Draw dialog
        dialog_screen.fill((40, 40, 40))

        # Draw title
        title = font.render(f"Select File ({len(items)} items)", True, (255, 255, 255))
        dialog_screen.blit(title, (10, 10))

        # Draw current path (truncate if too long so it doesn't overflow into list area)
        _max_path_chars = 90  # ~dialog_width/9 for Verdana 12
        _path_display = (
            current_dir
            if len(current_dir) <= _max_path_chars
            else "..." + current_dir[-(_max_path_chars - 3) :]
        )
        path_text = small_font.render(f"Path: {_path_display}", True, (200, 200, 200))
        dialog_screen.blit(path_text, (10, 40))

        # Draw path input field
        input_rect = pygame.Rect(10, 70, dialog_width - 20, 30)
        pygame.draw.rect(dialog_screen, (60, 60, 60), input_rect)
        if input_active:
            pygame.draw.rect(dialog_screen, (100, 150, 255), input_rect, 2)
        else:
            pygame.draw.rect(dialog_screen, (100, 100, 100), input_rect, 1)

        _input_display = text_input or current_dir
        _input_display = (
            _input_display
            if len(_input_display) <= _max_path_chars
            else "..." + _input_display[-(_max_path_chars - 3) :]
        )
        input_surface = small_font.render(_input_display, True, (255, 255, 255))
        dialog_screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))

        # Draw file list: reserve space so list never overlaps buttons/instructions
        list_rect = pygame.Rect(10, list_y, dialog_width - 20, list_height)
        pygame.draw.rect(dialog_screen, (30, 30, 30), list_rect)

        visible_items = items[scroll_offset : scroll_offset + visible_count]

        # Draw scrollbar if needed
        if len(items) > visible_count:
            scrollbar_height = max(20, int((visible_count / len(items)) * list_height))
            max_scroll = len(items) - visible_count
            if max_scroll > 0:
                scrollbar_y = list_y + int(
                    (scroll_offset / max_scroll) * (list_height - scrollbar_height)
                )
            else:
                scrollbar_y = list_y

            scrollbar_rect = pygame.Rect(dialog_width - 30, scrollbar_y, 20, scrollbar_height)
            pygame.draw.rect(dialog_screen, (100, 100, 100), scrollbar_rect)

        # Draw items (only visible_count rows; stays inside list_rect)
        for i, (item, is_dir) in enumerate(visible_items):
            y_pos = list_y + 5 + (i * item_height)

            # Draw background for item
            item_rect = pygame.Rect(12, y_pos - 2, dialog_width - 44, item_height)
            if is_dir:
                # Directory background (darker blue)
                bg_color = (40, 60, 80)
            else:
                # File background (darker gray)
                bg_color = (30, 30, 30)

            # Highlight selected item with brighter color
            if i == selected_index - scroll_offset:
                if is_dir:
                    bg_color = (80, 120, 200)  # Bright blue for selected directory
                else:
                    bg_color = (60, 60, 100)  # Brighter for selected file

            pygame.draw.rect(dialog_screen, bg_color, item_rect)

            # Draw icon (directory or file) - make directories more prominent
            if is_dir:
                icon_text = "[DIR]"
                color = (150, 220, 255)
            else:
                icon_text = "[FILE]"
                color = (200, 200, 200)

            # Try to render emoji, fallback to text if not supported
            try:
                icon = small_font.render(icon_text, True, color)
            except:
                icon_text = "[DIR]" if is_dir else "[FILE]"
                icon = small_font.render(icon_text, True, color)
            dialog_screen.blit(icon, (15, y_pos))

            # Draw item name - directories in brighter color
            if is_dir:
                name_color = (150, 220, 255)  # Bright blue for directories
                # Add "/" suffix to make it clearer it's a directory
                display_name = item + "/" if item != ".." else item
            else:
                name_color = (255, 255, 255)  # White for files
                display_name = item

            name_surface = small_font.render(display_name, True, name_color)
            dialog_screen.blit(name_surface, (100, y_pos))

        # Draw buttons
        button_y = dialog_height - 60
        up_button = pygame.Rect(10, button_y, 60, 35)
        open_button = pygame.Rect(dialog_width - 200, button_y, 80, 35)
        cancel_button = pygame.Rect(dialog_width - 100, button_y, 80, 35)

        # Up button (only if not at root)
        parent_dir = os.path.dirname(current_dir)
        if parent_dir != current_dir:  # Not at filesystem root
            pygame.draw.rect(dialog_screen, (80, 80, 150), up_button)
            up_text = font.render("↑ Up", True, (255, 255, 255))
            dialog_screen.blit(up_text, (up_button.x + 8, up_button.y + 8))

        pygame.draw.rect(dialog_screen, (50, 150, 50), open_button)
        pygame.draw.rect(dialog_screen, (150, 50, 50), cancel_button)

        open_text = font.render("Open", True, (255, 255, 255))
        cancel_text = font.render("Cancel", True, (255, 255, 255))
        dialog_screen.blit(open_text, (open_button.x + 15, open_button.y + 8))
        dialog_screen.blit(cancel_text, (cancel_button.x + 10, cancel_button.y + 8))

        # Draw instructions
        instructions = small_font.render(
            "Click folders to navigate | ↑↓: Navigate | Enter: Open | ESC: Cancel",
            True,
            (150, 150, 150),
        )
        dialog_screen.blit(instructions, (10, dialog_height - 25))

        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                selected_file = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    selected_file = None

                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    if visible_items and selected_index - scroll_offset < len(visible_items):
                        item, is_dir = visible_items[selected_index - scroll_offset]
                        if is_dir:
                            # Enter directory
                            if item == "..":
                                current_dir = os.path.dirname(current_dir)
                            else:
                                current_dir = os.path.join(current_dir, item)
                            items = load_directory(current_dir)
                            selected_index = 0
                            scroll_offset = 0
                            text_input = ""
                        else:
                            # Select file
                            selected_file = os.path.join(current_dir, item)
                            running = False
                    elif text_input and os.path.isfile(text_input):
                        selected_file = text_input
                        running = False

                elif event.key == pygame.K_UP:
                    if selected_index > 0:
                        selected_index -= 1
                        if selected_index < scroll_offset:
                            scroll_offset = max(0, selected_index)

                elif event.key == pygame.K_DOWN:
                    if selected_index < len(items) - 1:
                        selected_index += 1
                        if selected_index - scroll_offset >= visible_count:
                            scroll_offset = selected_index - visible_count + 1

                elif event.key == pygame.K_BACKSPACE:
                    if input_active:
                        text_input = text_input[:-1]

                elif event.key == pygame.K_TAB:
                    input_active = not input_active
                    if not input_active:
                        # Try to navigate to entered path
                        if text_input:
                            test_path = os.path.expanduser(text_input)
                            if os.path.isdir(test_path):
                                current_dir = test_path
                                items = load_directory(current_dir)
                                text_input = ""
                                selected_index = 0
                                scroll_offset = 0
                            elif os.path.isfile(test_path):
                                selected_file = test_path
                                running = False

                else:
                    if input_active:
                        text_input += event.unicode

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                if event.button == 1:  # Left click
                    # Check path input field
                    if input_rect.collidepoint(x, y):
                        input_active = True

                    # Check file list
                    elif list_rect.collidepoint(x, y):
                        rel_y = y - list_y - 5
                        clicked_index = rel_y // item_height
                        if 0 <= clicked_index < len(visible_items):
                            item, is_dir = visible_items[clicked_index]
                            clicked_absolute_index = scroll_offset + clicked_index

                            # Check for double-click
                            current_time = pygame.time.get_ticks()
                            is_double_click = (
                                clicked_absolute_index == last_click_index
                                and current_time - last_click_time < double_click_delay
                            )

                            if is_double_click:
                                # Double-click: open file or enter directory
                                if is_dir:
                                    # Enter directory
                                    if item == "..":
                                        current_dir = os.path.dirname(current_dir)
                                    else:
                                        current_dir = os.path.join(current_dir, item)
                                    items = load_directory(current_dir)
                                    selected_index = 0
                                    scroll_offset = 0
                                    text_input = ""
                                else:
                                    # Open file on double-click
                                    selected_file = os.path.join(current_dir, item)
                                    running = False
                            else:
                                # Single click: select item
                                selected_index = clicked_absolute_index
                                last_click_time = current_time
                                last_click_index = clicked_absolute_index

                                # For directories, enter on single click (user-friendly)
                                if is_dir:
                                    if item == "..":
                                        current_dir = os.path.dirname(current_dir)
                                    else:
                                        current_dir = os.path.join(current_dir, item)
                                    items = load_directory(current_dir)
                                    selected_index = 0
                                    scroll_offset = 0
                                    text_input = ""
                                    last_click_index = -1  # Reset for directory navigation

                    # Check buttons
                    elif up_button.collidepoint(x, y):
                        parent_dir = os.path.dirname(current_dir)
                        if parent_dir != current_dir:  # Not at filesystem root
                            # Go up one directory
                            current_dir = parent_dir
                            items = load_directory(current_dir)
                            selected_index = 0
                            scroll_offset = 0
                            text_input = ""

                    elif open_button.collidepoint(x, y):
                        if visible_items and selected_index - scroll_offset < len(visible_items):
                            item, is_dir = visible_items[selected_index - scroll_offset]
                            if not is_dir:
                                selected_file = os.path.join(current_dir, item)
                                running = False
                        elif text_input and os.path.isfile(text_input):
                            selected_file = text_input
                            running = False

                    elif cancel_button.collidepoint(x, y):
                        running = False
                        selected_file = None

            elif event.type == pygame.MOUSEWHEEL:
                # Scroll the list
                # event.y > 0 means scroll up (previous items) -> decrease offset
                # event.y < 0 means scroll down (next items) -> increase offset

                scroll_speed = 3  # Items per scroll click
                if event.y > 0:
                    scroll_offset = max(0, scroll_offset - scroll_speed)
                elif event.y < 0:
                    # Don't scroll past end
                    max_scroll = max(0, len(items) - visible_count)
                    scroll_offset = min(max_scroll, scroll_offset + scroll_speed)

        pygame.time.Clock().tick(60)

    # Restore original window size
    if restore_size:
        pygame.display.set_mode(restore_size, pygame.RESIZABLE)
    elif old_size:
        pygame.display.set_mode(old_size, pygame.RESIZABLE)

    return selected_file


def play_video_with_controls(
    video_path, coordinates=None, labels=None, initial_dataset_dir=None, initial_labeling_mode=None
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Default labels if not provided
    if labels is None:
        labels = []

    # Video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize coordinates if not provided
    if coordinates is None:
        coordinates = {i: [] for i in range(total_frames)}

    # Initialize active swap rules list to track session state
    active_swap_rules = []

    # Check for basename_swap.toml and apply initial swaps
    if video_path and TOML_AVAILABLE:
        try:
            base_dir = os.path.dirname(video_path)
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            swap_file = os.path.join(base_dir, f"{base_filename}_swap.toml")
            if os.path.exists(swap_file):
                print(f"Loading swap config from {swap_file}")
                swap_conf = load_swap_toml(swap_file)
                if swap_conf:
                    # Update session state
                    active_swap_rules = swap_conf
                    coordinates = apply_swap_config(coordinates, swap_conf)
                    print("Initial swaps applied from TOML.")
        except Exception as e:
            print(f"Auto-loading swap config failed: {e}")

    # Initialize Pygame
    pygame.init()
    screen_width, screen_height = (
        pygame.display.Info().current_w,
        pygame.display.Info().current_h,
    )
    window_width = min(original_width, screen_width - 100)
    window_height = min(original_height, screen_height - 150)
    screen = pygame.display.set_mode((window_width, window_height + 80), pygame.RESIZABLE)
    pygame.display.set_caption("Video Player with Controls")
    clock = pygame.time.Clock()

    # Initialize pygame.scrap for Ctrl+V clipboard paste support
    with suppress(Exception):
        pygame.scrap.init()

    # Control variables
    zoom_level = 1.0
    offset_x, offset_y = 0, 0
    frame_count = 0
    paused = True
    scrolling = False
    dragging_slider = False

    # Add marker navigation variables
    selected_marker_idx = 0  # Começar sempre com o marker 1 selecionado

    # Variables for the "1 line" mode (one-line marker mode)
    one_line_mode = False
    one_line_markers = []  # Each item: (frame_number, x, y)
    deleted_markers = set()  # Keep track of deleted marker indices

    # If no coordinates were loaded, initialize a dictionary with an empty list per frame.
    if coordinates is None:
        coordinates = {i: [] for i in range(total_frames)}

    # For regular mode, we'll track deleted positions
    deleted_positions = {i: set() for i in range(total_frames)}

    # Add persistence variables
    persistence_enabled = False
    persistence_frames = 10  # Default: show points from 10 previous frames

    # Add sequential mode variable
    sequential_mode = False

    # Add auto-marking mode variable
    auto_marking_mode = False

    # Bounding box labeling mode variables (preserved when switching video via F8)
    labeling_mode = False if initial_labeling_mode is None else initial_labeling_mode
    bboxes = {}  # Structure: {frame_index: [{'x': int, 'y': int, 'w': int, 'h': int, 'label': str}, ...]}
    drawing_box = False
    box_start_pos = None  # (x, y) in video coordinates
    current_box_rect = None  # pygame.Rect for preview
    current_label = "object"  # Default class label
    current_dataset_dir = (
        os.path.abspath(initial_dataset_dir)
        if initial_dataset_dir and os.path.isdir(initial_dataset_dir)
        else None
    )  # When set, Save (F5) appends to this dataset (multi-video)

    # When opening with a dataset (e.g. after F8 switch or --dataset), auto-load project for this video
    switch_to_video = None  # If set, exit and run_getpixelvideo will reopen with this path
    auto_load_project_done = False

    # Feature: Click & Pass
    click_pass_mode = False

    # Feature: Playback Speed
    playback_speed = 1.0  # 1.0 = Normal, 0.5 = Half speed, 2.0 = Double speed

    # YOLO tracking visualization variables
    tracking_data = {}  # Structure: {frame_index: [{'x1': int, 'y1': int, 'x2': int, 'y2': int, 'label': str, 'conf': float, 'color': (r, g, b)}, ...]}
    show_tracking = True  # Toggle to show/hide tracking boxes
    csv_loaded = False  # Flag to indicate if tracking CSV was loaded

    def export_video_with_annotations():
        """Export video with all annotations (markers, tracking boxes, labeling boxes) preserving audio"""
        nonlocal save_message_text, showing_save_message, save_message_timer
        import tempfile

        # Determine output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        output_path = os.path.join(video_dir, f"{base_name}_annotated.mp4")

        # Create temporary file for video without audio
        temp_video = None
        try:
            # Create temporary file
            temp_fd, temp_video = tempfile.mkstemp(suffix=".mp4", dir=video_dir)
            os.close(temp_fd)

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                save_message_text = "Error: Could not open video"
                showing_save_message = True
                save_message_timer = 90
                return

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30  # Default FPS if not available
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create video writer (temporary file without audio)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

            if not out.isOpened():
                save_message_text = "Error: Could not create output video"
                showing_save_message = True
                save_message_timer = 90
                cap.release()
                return

            # Process each frame - use frame-by-frame reading to ensure all frames are processed
            progress_update_interval = max(1, total_frames_video // 20)  # Update every 5%
            frames_written = 0

            # Process all frames explicitly by index
            for frame_idx in range(total_frames_video):
                # Set frame position explicitly to ensure we read the correct frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    print(f"Warning: Could not read frame {frame_idx}/{total_frames_video}")
                    # Try reading sequentially as fallback
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error: Could not read frame {frame_idx} even with sequential read")
                        # Create a black frame as last resort to maintain frame count
                        frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Verify frame dimensions
                if frame.shape[0] != height or frame.shape[1] != width:
                    print(f"Warning: Frame {frame_idx} has wrong dimensions, resizing...")
                    frame = cv2.resize(frame, (width, height))

                # OpenCV uses BGR natively, so we work directly with the frame

                # Draw markers (coordinates)
                if one_line_mode:
                    for idx, (f_num, x, y) in enumerate(one_line_markers):
                        if f_num == frame_idx and x is not None and y is not None:
                            # Draw marker circle (green in BGR)
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                            # Draw marker number
                            cv2.putText(
                                frame,
                                str(idx + 1),
                                (int(x) + 8, int(y) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                else:
                    if frame_idx in coordinates:
                        for i, (x, y) in enumerate(coordinates[frame_idx]):
                            if i not in deleted_positions.get(frame_idx, set()):
                                if x is not None and y is not None:
                                    # Draw marker circle (green in BGR)
                                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                                    # Draw marker number
                                    cv2.putText(
                                        frame,
                                        str(i + 1),
                                        (int(x) + 8, int(y) - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (255, 255, 255),
                                        1,
                                    )

                # Draw YOLO tracking boxes
                if csv_loaded and frame_idx in tracking_data:
                    for box in tracking_data[frame_idx]:
                        x1, y1 = int(box["x1"]), int(box["y1"])
                        x2, y2 = int(box["x2"]), int(box["y2"])
                        box_color = box.get("color", (0, 255, 0))
                        # Convert RGB to BGR for OpenCV (box_color is RGB tuple)
                        box_color_bgr = (box_color[2], box_color[1], box_color[0])

                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color_bgr, 2)

                        # Build label text
                        label = box.get("label", "object")
                        tracker_id = box.get("id")
                        conf = box.get("conf", 0)

                        label_parts = [f"Label:{label}"]
                        if tracker_id is not None:
                            label_parts.append(f"id:{tracker_id}")
                        if conf > 0:
                            label_parts.append(f"conf:{conf:.2f}")

                        label_text = " ".join(label_parts)

                        # Draw text above box
                        if label_text:
                            # Get text size
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                            )
                            # Draw background rectangle
                            cv2.rectangle(
                                frame,
                                (x1, y1 - text_height - baseline - 4),
                                (x1 + text_width + 4, y1),
                                box_color_bgr,
                                -1,
                            )
                            # Draw text
                            cv2.putText(
                                frame,
                                label_text,
                                (x1 + 2, y1 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )

                # Draw labeling bounding boxes
                if frame_idx in bboxes:
                    for bbox in bboxes[frame_idx]:
                        x = int(bbox["x"])
                        y = int(bbox["y"])
                        w = int(bbox["w"])
                        h = int(bbox["h"])
                        # Draw red rectangle (BGR: blue=0, green=0, red=255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # Draw label if available
                        label = bbox.get("label", "object")
                        cv2.putText(
                            frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                        )

                # Write frame directly (already in BGR format)
                out.write(frame)
                frames_written += 1

                # Update progress message
                if (frame_idx + 1) % progress_update_interval == 0:
                    progress = int(((frame_idx + 1) / total_frames_video) * 100)
                    save_message_text = f"Exporting video... {progress}% ({frame_idx + 1}/{total_frames_video} frames)"
                    showing_save_message = True
                    save_message_timer = 1  # Keep updating
                    pygame.event.pump()  # Keep UI responsive

            # Verify all frames were written
            print(f"[OK] Processed {frames_written} frames (expected {total_frames_video})")
            if frames_written != total_frames_video:
                print(
                    f"[WARNING] Warning: Frame count mismatch! Written: {frames_written}, Expected: {total_frames_video}"
                )

            # Clean up video writers
            cap.release()
            out.release()

            # Now use ffmpeg to combine video with audio from original
            save_message_text = "Adding audio to video..."
            showing_save_message = True
            save_message_timer = 1
            pygame.event.pump()

            # Check if ffmpeg is available
            try:
                subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # ffmpeg not available, just copy the temp file
                import shutil

                shutil.move(temp_video, output_path)
                save_message_text = f"Video exported (no audio): {os.path.basename(output_path)}"
                showing_save_message = True
                save_message_timer = 120
                print(f"[WARNING] ffmpeg not found. Video exported without audio: {output_path}")
                return

            # Use ffmpeg to combine video with audio
            # Command: ffmpeg -i temp_video -i original_video -c:v copy -c:a copy -map 0:v:0 -map 1:a:0? output
            # Note: We use -map 1:a:0? to optionally use audio
            # We DON'T use -shortest to ensure all video frames are preserved
            # The video length will determine the output length, audio will be trimmed if longer
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-i",
                temp_video,  # Video input (annotated, no audio) - this has all frames
                "-i",
                video_path,  # Original video (for audio)
                "-c:v",
                "copy",  # Copy video codec (no re-encoding) - preserves all frames
                "-c:a",
                "aac",  # Use AAC for audio (better compatibility)
                "-map",
                "0:v:0",  # Use video from first input (all frames - this is the master)
                "-map",
                "1:a:0?",  # Use audio from second input (if available)
                # Don't use -shortest - let video determine length (preserves all video frames)
                # Audio will be trimmed to match video length if needed
                output_path,
            ]

            # First, verify the temp video has the correct number of frames
            # Check frame count of temp video
            try:
                check_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_packets",
                    "-show_entries",
                    "stream=nb_read_packets",
                    "-of",
                    "csv=p=0",
                    temp_video,
                ]
                result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    temp_frame_count = int(result.stdout.strip())
                    print(
                        f"Temp video has {temp_frame_count} frames (expected {total_frames_video})"
                    )
                    if temp_frame_count != total_frames_video:
                        print("[WARNING] Warning: Temp video frame count mismatch!")
            except:
                pass  # ffprobe not available or failed, continue anyway

            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode == 0:
                    # Success - remove temp file
                    with suppress(BaseException):
                        os.remove(temp_video)
                    save_message_text = f"Video exported: {os.path.basename(output_path)}"
                    showing_save_message = True
                    save_message_timer = 120
                    print(f"[OK] Video exported successfully with audio to: {output_path}")
                else:
                    # ffmpeg failed, but video was created - try without audio
                    import shutil

                    if os.path.exists(temp_video):
                        shutil.move(temp_video, output_path)
                    save_message_text = (
                        f"Video exported (audio copy failed): {os.path.basename(output_path)}"
                    )
                    showing_save_message = True
                    save_message_timer = 120
                    print(
                        f"[WARNING] ffmpeg audio copy failed. Video exported without audio: {output_path}"
                    )
                    print(f"ffmpeg error: {result.stderr}")
            except subprocess.TimeoutExpired:
                # Timeout - copy temp file
                import shutil

                if os.path.exists(temp_video):
                    shutil.move(temp_video, output_path)
                save_message_text = f"Video exported (timeout): {os.path.basename(output_path)}"
                showing_save_message = True
                save_message_timer = 120
                print(f"[WARNING] ffmpeg timeout. Video exported without audio: {output_path}")
            except Exception as e:
                # Error - copy temp file
                import shutil

                if os.path.exists(temp_video):
                    shutil.move(temp_video, output_path)
                save_message_text = f"Video exported (error): {os.path.basename(output_path)}"
                showing_save_message = True
                save_message_timer = 120
                print(f"[WARNING] Error adding audio: {e}. Video exported without audio: {output_path}")

        except Exception as e:
            save_message_text = f"Error exporting video: {str(e)}"
            showing_save_message = True
            save_message_timer = 120
            print(f"Error exporting video: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Clean up temp file if it still exists
            if temp_video and os.path.exists(temp_video):
                with suppress(BaseException):
                    os.remove(temp_video)

    def load_tracking_csv():
        """Load YOLO tracking CSV file (all_id_detection.csv format)"""
        nonlocal \
            tracking_data, \
            csv_loaded, \
            save_message_text, \
            showing_save_message, \
            save_message_timer

        # Use platform-specific file dialog to avoid pygame/tkinter conflicts
        import platform

        initial_dir = os.path.dirname(video_path) if video_path else os.path.expanduser("~")

        if platform.system() == "Linux":
            # Use pygame native dialog on Linux to avoid conflicts
            csv_path = pygame_file_dialog(
                initial_dir=initial_dir,
                file_extensions=[".csv"],
                restore_size=(window_width, window_height + 80),
            )
        else:
            # Use Tkinter on Windows/Mac
            try:
                import tkinter as tk
                from tkinter import filedialog

                # Block Pygame events while dialog is open
                pygame.event.set_blocked(
                    [
                        pygame.MOUSEBUTTONDOWN,
                        pygame.MOUSEBUTTONUP,
                        pygame.MOUSEMOTION,
                        pygame.KEYDOWN,
                        pygame.KEYUP,
                    ]
                )
                pygame.event.clear()

                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                root.update_idletasks()

                csv_path = filedialog.askopenfilename(
                    title="Select Tracking CSV File",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                    initialdir=initial_dir,
                )

                root.destroy()

                # Re-enable Pygame events
                pygame.event.set_allowed(
                    [
                        pygame.MOUSEBUTTONDOWN,
                        pygame.MOUSEBUTTONUP,
                        pygame.MOUSEMOTION,
                        pygame.KEYDOWN,
                        pygame.KEYUP,
                    ]
                )
                pygame.event.clear()

            except Exception as e:
                print(f"Error opening file dialog: {e}")
                save_message_text = f"Error: {e}"
                showing_save_message = True
                save_message_timer = 60
                return

        if not csv_path:
            return

        try:
            print(f"Loading tracking CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            # Debug: Print CSV info
            print(f"CSV shape: {df.shape}")
            print(f"CSV columns: {list(df.columns)}")
            print(f"First few rows:\n{df.head()}")

            # Clear old tracking data
            tracking_data = {}

            # Check if CSV has the expected format
            if "Frame" not in df.columns:
                save_message_text = "CSV missing 'Frame' column"
                showing_save_message = True
                save_message_timer = 90
                print(f"ERROR: Missing 'Frame' column. Available columns: {list(df.columns)}")
                return

            # Identify object IDs from column suffixes
            # Format options:
            # 1. Multi-object (all_id_detection.csv): X_min_{label}_id_{id}, Y_min_{label}_id_{id}, etc.
            # 2. Individual file (person_id_01.csv): X_min, Y_min, X_max, Y_max (no suffix)
            object_suffixes = set()

            # First, check for individual file format (person_id_01.csv)
            # This format has simple column names without suffixes
            # Check exact column names (case-sensitive)
            has_individual_format = (
                "X_min" in df.columns
                and "Y_min" in df.columns
                and "X_max" in df.columns
                and "Y_max" in df.columns
            )

            if has_individual_format:
                # Individual file format detected
                print("[OK] Detected individual tracking CSV format (person_id_01.csv style)")
                print(
                    f"  Columns found: X_min={'X_min' in df.columns}, Y_min={'Y_min' in df.columns}, X_max={'X_max' in df.columns}, Y_max={'Y_max' in df.columns}"
                )
                object_suffixes.add("")  # Empty suffix for single object
            else:
                # Try to detect multi-object format with suffixes
                print("Checking for multi-object format...")
                for col in df.columns:
                    # Pattern 1: X_min_{label}_id_{id}
                    match = re.search(r"X_min_(.+)$", col)
                    if match:
                        suffix = match.group(1)
                        object_suffixes.add(suffix)
                        print(f"  Found multi-object suffix: {suffix}")
                    # Pattern 2: X_min_person_id_01 (simpler)
                    elif col.startswith("X_min_") and col != "X_min":
                        suffix = col.replace("X_min_", "")
                        object_suffixes.add(suffix)
                        print(f"  Found multi-object suffix (pattern 2): {suffix}")

            if not object_suffixes:
                save_message_text = f"Could not detect tracking format. Columns: {list(df.columns)}"
                showing_save_message = True
                save_message_timer = 120
                print(f"ERROR: Could not detect format. Available columns: {list(df.columns)}")
                print("  Looking for: X_min, Y_min, X_max, Y_max")
                return

            print(f"[OK] Found {len(object_suffixes)} object(s) in CSV")

            # Process each row to build tracking_data dictionary
            rows_processed = 0
            boxes_loaded = 0
            skipped_frames = 0
            for idx, row in df.iterrows():
                try:
                    frame_idx = int(row["Frame"])
                    if frame_idx not in tracking_data:
                        tracking_data[frame_idx] = []

                    # Process each object suffix
                    for suffix in object_suffixes:
                        if suffix == "":
                            # Single object format (individual CSV file)
                            x_min_col = "X_min"
                            y_min_col = "Y_min"
                            x_max_col = "X_max"
                            y_max_col = "Y_max"
                            conf_col = "Confidence"
                            label_col = "Label"
                            id_col = "ID"  # Try common ID column names
                            if "ID" not in df.columns:
                                if "Tracker_ID" in df.columns:
                                    id_col = "Tracker_ID"
                                elif "id" in df.columns:
                                    id_col = "id"
                                elif "tracker_id" in df.columns:
                                    id_col = "tracker_id"
                                else:
                                    id_col = None
                            color_r_col = "Color_R"
                            color_g_col = "Color_G"
                            color_b_col = "Color_B"
                        else:
                            # Multi-object format with suffix
                            x_min_col = f"X_min_{suffix}"
                            y_min_col = f"Y_min_{suffix}"
                            x_max_col = f"X_max_{suffix}"
                            y_max_col = f"Y_max_{suffix}"
                            conf_col = f"Confidence_{suffix}"
                            label_col = f"Label_{suffix}"
                            # Try to find ID column with suffix
                            id_col = f"ID_{suffix}"
                            if id_col not in df.columns:
                                if f"Tracker_ID_{suffix}" in df.columns:
                                    id_col = f"Tracker_ID_{suffix}"
                                elif f"id_{suffix}" in df.columns:
                                    id_col = f"id_{suffix}"
                                else:
                                    # Try to extract ID from suffix itself (e.g., "person_id_01" -> "01")
                                    id_match = re.search(r"id[_\s]*(\d+)", suffix, re.IGNORECASE)
                                    if id_match:
                                        id_col = f"ID_{id_match.group(1)}"  # Use extracted ID
                                    else:
                                        id_col = None
                            color_r_col = f"Color_R_{suffix}"
                            color_g_col = f"Color_G_{suffix}"
                            color_b_col = f"Color_B_{suffix}"

                        # Check if this object has valid data for this frame
                        # For individual format, check all required columns exist
                        if suffix == "":
                            # Individual format: require X_min, Y_min, X_max, Y_max
                            # Check if columns exist
                            cols_exist = (
                                x_min_col in df.columns
                                and y_min_col in df.columns
                                and x_max_col in df.columns
                                and y_max_col in df.columns
                            )

                            if cols_exist:
                                # Check if values are not NaN
                                x_min_val = row.get(x_min_col)
                                y_min_val = row.get(y_min_col)
                                x_max_val = row.get(x_max_col)
                                y_max_val = row.get(y_max_col)

                                has_valid_data = (
                                    pd.notna(x_min_val)
                                    and pd.notna(y_min_val)
                                    and pd.notna(x_max_val)
                                    and pd.notna(y_max_val)
                                )

                                if has_valid_data:
                                    try:
                                        # Extract ID from column or suffix
                                        tracker_id = None
                                        if (
                                            id_col
                                            and id_col in df.columns
                                            and pd.notna(row.get(id_col))
                                        ):
                                            with suppress(ValueError, TypeError):
                                                tracker_id = int(float(row[id_col]))

                                        # If ID not found in column, try to extract from suffix
                                        if tracker_id is None and suffix:
                                            id_match = re.search(
                                                r"id[_\s]*(\d+)", suffix, re.IGNORECASE
                                            )
                                            if id_match:
                                                with suppress(ValueError, TypeError):
                                                    tracker_id = int(id_match.group(1))

                                        box = {
                                            "x1": int(float(x_min_val)),
                                            "y1": int(float(y_min_val)),
                                            "x2": int(float(x_max_val)),
                                            "y2": int(float(y_max_val)),
                                            "conf": float(row[conf_col])
                                            if conf_col in df.columns
                                            and pd.notna(row.get(conf_col))
                                            else 0.0,
                                            "label": str(row[label_col])
                                            if label_col in df.columns
                                            and pd.notna(row.get(label_col))
                                            else "object",
                                            "id": tracker_id,  # Add tracker ID
                                            "color": (
                                                int(row[color_r_col])
                                                if color_r_col in df.columns
                                                and pd.notna(row.get(color_r_col))
                                                else 0,
                                                int(row[color_g_col])
                                                if color_g_col in df.columns
                                                and pd.notna(row.get(color_g_col))
                                                else 255,
                                                int(row[color_b_col])
                                                if color_b_col in df.columns
                                                and pd.notna(row.get(color_b_col))
                                                else 0,
                                            ),
                                        }

                                        # Validate box coordinates
                                        if box["x2"] > box["x1"] and box["y2"] > box["y1"]:
                                            tracking_data[frame_idx].append(box)
                                            boxes_loaded += 1
                                            if (
                                                boxes_loaded <= 3
                                            ):  # Print first 3 boxes for debugging
                                                print(
                                                    f"  Loaded box at frame {frame_idx}: x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}, label={box['label']}"
                                                )
                                        else:
                                            print(
                                                f"Warning: Invalid box coordinates at frame {frame_idx}: x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}"
                                            )
                                    except (ValueError, KeyError, TypeError) as e:
                                        # Skip invalid box data
                                        print(
                                            f"Warning: Skipping invalid box at frame {frame_idx}: {e}"
                                        )
                                        print(
                                            f"  Row data: x_min={x_min_val}, y_min={y_min_val}, x_max={x_max_val}, y_max={y_max_val}"
                                        )
                                else:
                                    # Frame has no detection data (NaN values) - this is normal
                                    skipped_frames += 1
                                    if skipped_frames <= 3:
                                        print(
                                            f"  Frame {frame_idx}: No detection data (NaN values)"
                                        )
                            else:
                                print(
                                    "ERROR: Required columns missing. Expected: X_min, Y_min, X_max, Y_max"
                                )
                                print(f"  Found columns: {list(df.columns)}")
                                return
                        else:
                            # Multi-object format: check if x_min_col exists and has data
                            if x_min_col in df.columns and pd.notna(row.get(x_min_col)):
                                try:
                                    # Extract ID from column or suffix
                                    tracker_id = None
                                    if (
                                        id_col
                                        and id_col in df.columns
                                        and pd.notna(row.get(id_col))
                                    ):
                                        with suppress(ValueError, TypeError):
                                            tracker_id = int(float(row[id_col]))

                                    # If ID not found in column, try to extract from suffix
                                    if tracker_id is None and suffix:
                                        id_match = re.search(
                                            r"id[_\s]*(\d+)", suffix, re.IGNORECASE
                                        )
                                        if id_match:
                                            with suppress(ValueError, TypeError):
                                                tracker_id = int(id_match.group(1))

                                    box = {
                                        "x1": int(float(row[x_min_col])),
                                        "y1": int(float(row[y_min_col]))
                                        if y_min_col in df.columns and pd.notna(row.get(y_min_col))
                                        else 0,
                                        "x2": int(float(row[x_max_col]))
                                        if x_max_col in df.columns and pd.notna(row.get(x_max_col))
                                        else int(float(row[x_min_col])),
                                        "y2": int(float(row[y_max_col]))
                                        if y_max_col in df.columns and pd.notna(row.get(y_max_col))
                                        else 0,
                                        "conf": float(row[conf_col])
                                        if conf_col in df.columns and pd.notna(row.get(conf_col))
                                        else 0.0,
                                        "label": str(row[label_col])
                                        if label_col in df.columns and pd.notna(row.get(label_col))
                                        else suffix,
                                        "id": tracker_id,  # Add tracker ID
                                        "color": (
                                            int(row[color_r_col])
                                            if color_r_col in df.columns
                                            and pd.notna(row.get(color_r_col))
                                            else 0,
                                            int(row[color_g_col])
                                            if color_g_col in df.columns
                                            and pd.notna(row.get(color_g_col))
                                            else 255,
                                            int(row[color_b_col])
                                            if color_b_col in df.columns
                                            and pd.notna(row.get(color_b_col))
                                            else 0,
                                        ),
                                    }

                                    # Validate box coordinates
                                    if box["x2"] > box["x1"] and box["y2"] > box["y1"]:
                                        tracking_data[frame_idx].append(box)
                                        boxes_loaded += 1
                                except (ValueError, KeyError, TypeError):
                                    # Skip invalid box data
                                    pass

                except (ValueError, KeyError) as e:
                    # Skip rows with invalid frame numbers or missing required columns
                    print(f"Warning: Skipping row {idx} due to error: {e}")

                rows_processed += 1

            csv_loaded = True
            total_boxes = sum(len(boxes) for boxes in tracking_data.values())
            frame_range = (
                f"{min(tracking_data.keys())}-{max(tracking_data.keys())}"
                if tracking_data
                else "none"
            )
            save_message_text = f"Tracking loaded: {len(tracking_data)} frames, {total_boxes} boxes"
            showing_save_message = True
            save_message_timer = 120
            print("[OK] Successfully loaded tracking data:")
            print(f"  - Frames with data: {len(tracking_data)}")
            print(f"  - Total boxes: {total_boxes}")
            print(f"  - Frame range: {frame_range}")
            print(f"  - Rows processed: {rows_processed}")
            if tracking_data:
                sample_frame = list(tracking_data.keys())[0]
                sample_box = tracking_data[sample_frame][0] if tracking_data[sample_frame] else None
                if sample_box:
                    print(
                        f"  - Sample box at frame {sample_frame}: x1={sample_box['x1']}, y1={sample_box['y1']}, x2={sample_box['x2']}, y2={sample_box['y2']}, label={sample_box['label']}, color={sample_box['color']}"
                    )
            else:
                print("  WARNING: No tracking data loaded! Check CSV format.")

        except Exception as e:
            save_message_text = f"Error loading CSV: {str(e)}"
            showing_save_message = True
            save_message_timer = 120
            print(f"Error loading tracking CSV: {e}")
            import traceback

            traceback.print_exc()

    def draw_controls():
        """
        Draw the control area on a separate surface.
        The frame slider is drawn across the bottom of the control area.
        In the lower-right corner (a bit above the slider) a compact cluster of four buttons is drawn:
          - Save
          - Help
          - "1 line" (toggle one-line marker mode)
          - "Persist" (toggle point persistence)
        """
        control_surface_height = 80
        control_surface = pygame.Surface((window_width, control_surface_height))
        control_surface.fill((30, 30, 30))
        # Use Verdana for better legibility (l vs I)
        font = pygame.font.SysFont("verdana", 12)

        # Draw slider for frames along the bottom.
        slider_margin_left = 10
        slider_margin_right = 10
        slider_width = window_width - slider_margin_left - slider_margin_right
        slider_height = 10
        slider_y = control_surface_height - slider_height - 5
        pygame.draw.rect(
            control_surface,
            (60, 60, 60),
            (slider_margin_left, slider_y, slider_width, slider_height),
        )
        # Guard against zero total_frames to avoid division by zero for short/merged videos
        denom_frames = total_frames if total_frames and total_frames > 0 else 1
        slider_pos = slider_margin_left + int((frame_count / denom_frames) * slider_width)
        pygame.draw.circle(
            control_surface,
            (255, 255, 255),
            (slider_pos, slider_y + slider_height // 2),
            8,
        )

        # Draw frame info above the slider.
        # Use a safe display total when total_frames is unknown/zero
        display_total = (
            total_frames if total_frames and total_frames > 0 else max(1, frame_count + 1)
        )
        frame_info = font.render(f"Frame: {frame_count + 1}/{display_total}", True, (255, 255, 255))
        control_surface.blit(frame_info, (slider_margin_left, slider_y - 25))

        # Draw auto-marking indicator if enabled
        if auto_marking_mode:
            auto_indicator = font.render("AUTO-MARKING ON", True, (255, 255, 0))
            control_surface.blit(auto_indicator, (slider_margin_left + 300, slider_y - 25))

        # Draw marker navigation and persistence info
        if one_line_mode:
            frame_markers = [m for m in one_line_markers if m[0] == frame_count]
            total_markers = len(frame_markers)
        else:
            total_markers = len(coordinates[frame_count])

        if total_markers > 0:
            marker_idx = selected_marker_idx + 1 if selected_marker_idx >= 0 else 0
            marker_info = font.render(
                f"Marker: {marker_idx}/{total_markers}", True, (255, 255, 255)
            )
            control_surface.blit(marker_info, (slider_margin_left + 200, slider_y - 25))

        # Draw button cluster in the lower-right corner.
        button_width = 50
        button_height = 20
        button_gap = 10
        persist_button_width = 70
        seq_button_width = 70
        auto_button_width = 70
        click_pass_button_width = 70
        labeling_button_width = 70
        tracking_csv_button_width = 120
        export_video_button_width = 100
        help_web_button_width = 30  # Width for '?' button
        total_buttons_width = (
            (button_width * 3)
            + persist_button_width
            + seq_button_width
            + auto_button_width
            + click_pass_button_width
            + labeling_button_width
            + tracking_csv_button_width
            + export_video_button_width
            + help_web_button_width
            + (button_gap * 9)
        )
        cluster_x = (window_width - total_buttons_width) // 2
        cluster_y = slider_y - button_height - 5

        # Helper to advance x position
        current_x = cluster_x

        # 1. "1 Line" mode toggle button.
        one_line_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            button_width,
            button_height,
        )
        current_x += button_width + button_gap

        btn_color = (150, 50, 50) if one_line_mode else (100, 100, 100)
        pygame.draw.rect(control_surface, btn_color, one_line_button_rect)
        one_line_text = font.render("1 Line", True, (255, 255, 255))
        control_surface.blit(
            one_line_text, one_line_text.get_rect(center=one_line_button_rect.center)
        )

        # 2. "Persist" mode toggle button.
        persist_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            persist_button_width,
            button_height,
        )
        current_x += persist_button_width + button_gap

        persist_color = (50, 150, 50) if persistence_enabled else (100, 100, 100)
        pygame.draw.rect(control_surface, persist_color, persist_button_rect)

        # Just show "Persist" or "Persist ON" depending on state
        persist_text = font.render(
            "Persist ON" if persistence_enabled else "Persist", True, (255, 255, 255)
        )
        control_surface.blit(persist_text, persist_text.get_rect(center=persist_button_rect.center))

        # 3. Sequential mode button
        seq_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            seq_button_width,
            button_height,
        )
        current_x += seq_button_width + button_gap

        seq_color = (50, 150, 50) if sequential_mode else (100, 100, 100)
        pygame.draw.rect(control_surface, seq_color, seq_button_rect)
        seq_text = font.render("Sequential", True, (255, 255, 255))
        control_surface.blit(seq_text, seq_text.get_rect(center=seq_button_rect.center))

        # 4. Auto-marking mode button
        auto_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            auto_button_width,
            button_height,
        )
        current_x += auto_button_width + button_gap

        auto_color = (150, 50, 150) if auto_marking_mode else (100, 100, 100)
        pygame.draw.rect(control_surface, auto_color, auto_button_rect)
        auto_text = font.render("Auto", True, (255, 255, 255))
        control_surface.blit(auto_text, auto_text.get_rect(center=auto_button_rect.center))

        # 5. ClickPass mode button
        click_pass_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            click_pass_button_width,
            button_height,
        )
        current_x += click_pass_button_width + button_gap

        click_pass_color = (50, 50, 150) if click_pass_mode else (100, 100, 100)
        pygame.draw.rect(control_surface, click_pass_color, click_pass_button_rect)
        click_pass_text = font.render("ClickPass", True, (255, 255, 255))
        control_surface.blit(
            click_pass_text, click_pass_text.get_rect(center=click_pass_button_rect.center)
        )

        # 6. Pose button REMOVED (Use 'J' hotkey)

        # 7. Labeling mode button
        labeling_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            labeling_button_width,
            button_height,
        )
        current_x += labeling_button_width + button_gap

        labeling_color = (50, 150, 150) if labeling_mode else (100, 100, 100)
        pygame.draw.rect(control_surface, labeling_color, labeling_button_rect)
        labeling_text = font.render("Labeling", True, (255, 255, 255))
        control_surface.blit(
            labeling_text, labeling_text.get_rect(center=labeling_button_rect.center)
        )

        # 7. Load Tracking CSV button
        tracking_csv_button_width = 120
        tracking_csv_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            tracking_csv_button_width,
            button_height,
        )
        current_x += tracking_csv_button_width  # No gap yet, indicator comes next?

        tracking_csv_color = (100, 150, 200) if csv_loaded else (100, 100, 100)
        pygame.draw.rect(control_surface, tracking_csv_color, tracking_csv_button_rect)
        tracking_csv_text = font.render("Load Track CSV", True, (255, 255, 255))
        control_surface.blit(
            tracking_csv_text, tracking_csv_text.get_rect(center=tracking_csv_button_rect.center)
        )

        # 8. Show Tracking checkbox indicator (small square next to button)
        show_tracking_indicator_size = 12
        show_tracking_indicator_rect = pygame.Rect(
            tracking_csv_button_rect.right + 5,
            tracking_csv_button_rect.centery - show_tracking_indicator_size // 2,
            show_tracking_indicator_size,
            show_tracking_indicator_size,
        )
        # Advance X past indicator and gap
        current_x += 5 + show_tracking_indicator_size + button_gap

        if show_tracking and csv_loaded:
            pygame.draw.rect(control_surface, (0, 255, 0), show_tracking_indicator_rect)
            pygame.draw.rect(control_surface, (255, 255, 255), show_tracking_indicator_rect, 1)
        else:
            pygame.draw.rect(control_surface, (60, 60, 60), show_tracking_indicator_rect)
            pygame.draw.rect(control_surface, (150, 150, 150), show_tracking_indicator_rect, 1)

        # 9. Load button (Moved here)
        load_button_rect = pygame.Rect(current_x, cluster_y, button_width, button_height)
        current_x += button_width + button_gap

        pygame.draw.rect(control_surface, (100, 100, 100), load_button_rect)
        load_text = font.render("Load", True, (255, 255, 255))
        control_surface.blit(load_text, load_text.get_rect(center=load_button_rect.center))

        # 10. Save button (Moved here)
        save_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            button_width,
            button_height,
        )
        current_x += button_width + button_gap

        pygame.draw.rect(control_surface, (100, 100, 100), save_button_rect)
        save_text = font.render("Save", True, (255, 255, 255))
        control_surface.blit(save_text, save_text.get_rect(center=save_button_rect.center))

        # 10b. Dataset button (Load dataset folder; next Save appends - multi-video)
        dataset_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            button_width,
            button_height,
        )
        current_x += button_width + button_gap
        pygame.draw.rect(control_surface, (80, 120, 80), dataset_button_rect)
        dataset_text = font.render("Dataset", True, (255, 255, 255))
        control_surface.blit(dataset_text, dataset_text.get_rect(center=dataset_button_rect.center))

        # 11. Help button (Moved here)
        help_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            button_width,
            button_height,
        )
        current_x += button_width + button_gap

        pygame.draw.rect(control_surface, (100, 100, 100), help_button_rect)
        help_text = font.render("Help", True, (255, 255, 255))
        control_surface.blit(help_text, help_text.get_rect(center=help_button_rect.center))

        # 12. Export Video button
        export_video_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            export_video_button_width,
            button_height,
        )
        current_x += export_video_button_width + button_gap

        export_video_color = (200, 100, 50)  # Orange color
        pygame.draw.rect(control_surface, export_video_color, export_video_button_rect)
        export_video_text = font.render("Export Video", True, (255, 255, 255))
        control_surface.blit(
            export_video_text, export_video_text.get_rect(center=export_video_button_rect.center)
        )

        # 13. Help Web button ('?')
        help_web_button_rect = pygame.Rect(
            current_x,
            cluster_y,
            help_web_button_width,
            button_height,
        )
        current_x += help_web_button_width + button_gap

        pygame.draw.rect(control_surface, (100, 100, 150), help_web_button_rect)
        help_web_text = font.render("?", True, (255, 255, 255))
        control_surface.blit(
            help_web_text, help_web_text.get_rect(center=help_web_button_rect.center)
        )

        # Display current class label when in labeling mode
        if labeling_mode:
            class_info = font.render(f"Class: {current_label}", True, (255, 255, 0))
            control_surface.blit(class_info, (slider_margin_left + 400, slider_y - 25))

        # Display tracking info when CSV is loaded
        if csv_loaded:
            tracking_info = font.render(
                f"Tracking: {len(tracking_data)} frames", True, (150, 255, 150)
            )
            control_surface.blit(tracking_info, (slider_margin_left + 200, slider_y - 45))

        # Display Speed Info
        speed_text = font.render(f"Speed: {playback_speed}X", True, (255, 255, 255))
        control_surface.blit(speed_text, (window_width - 100, slider_y - 45))

        screen.blit(control_surface, (0, window_height))
        return (
            one_line_button_rect,
            save_button_rect,
            help_button_rect,
            persist_button_rect,
            load_button_rect,
            seq_button_rect,  # Add sequential button to return
            auto_button_rect,  # Add auto button to return
            click_pass_button_rect,  # Add ClickPass button to return
            labeling_button_rect,  # Add labeling button to return
            tracking_csv_button_rect,  # Add tracking CSV button to return
            show_tracking_indicator_rect,  # Add tracking indicator to return
            export_video_button_rect,  # Add export video button to return
            help_web_button_rect,  # Add help web button to return
            dataset_button_rect,  # Load dataset folder (multi-video)
            slider_margin_left,
            slider_y,
            slider_width,
            slider_height,
        )

    def show_help_dialog():
        # Instead of using tkinter, display help directly in pygame
        help_lines_left = [
            "Video Player Controls:",
            "- Space: Play/Pause",
            "- Right Arrow: Next Frame (when paused)",
            "- Left Arrow: Previous Frame (when paused)",
            "- Up Arrow: Fast Forward (when paused)",
            "- Down Arrow: Rewind (when paused)",
            "- +: Zoom In",
            "- -: Zoom Out",
            "- Scroll M: Zoom In/Out",
            "- Left Click: Add Marker",
            "- Right Click: Remove Last Marker",
            "- Middle Click: Enable Pan/Move",
            "- Drag Slider: Jump to Frame",
            "- TAB: Next marker in current frame",
            "- SHIFT+TAB: Previous marker in current frame",
            "- DELETE: Delete selected marker",
            "- A: Add new empty marker to file",
            "- R: Remove last marker from file",
            "- J: Detect Pose (MediaPipe + YOLO)",  # Updated Pose help
            "- H: Show this help",
            "- D: Delete all markes in the current frame",
            "- ?: Open documentation in browser",
            "",
            "=== LABELING MODE (Bounding Boxes) ===",
            "",
            "STEP 1: Activate Labeling Mode",
            "  - Press L key, OR",
            "  - Click 'Labeling' button (turns green)",
            "",
            "STEP 2: Draw Boxes",
            "  - Click and DRAG on video to draw",
            "  - Red boxes appear while drawing",
            "  - Release mouse to save box",
            "",
            "STEP 3: Edit Boxes",
            "  - Press Z / Right Click: Remove last box",
            "  - Navigate frames to label more",
            "",
            "STEP 4: Export Dataset",
            "  - Click 'Save' button, OR",
            "  - Press ESC key",
            "  - Dataset saved: train/val/test",
        ]

        help_lines_right = [
            "Marker Modes:",
            "- Normal Mode (default): Clicking selects and",
            "  updates the current marker. Use TAB to navigate.",
            "  Each marker keeps its ID across all frames.",
            "",
            "- 1 Line Mode (C key): Creates points in sequence",
            "  in one frame. Each click adds a new marker.",
            "  Use for tracing paths or outlines.",
            "",
            "Sequential Mode (S/O key): Each click creates",
            "  a new marker with incrementing IDs. No need",
            "  to select markers first. Only in Normal mode.",
            "",
            "- Auto-marking Mode (M key): Automatically marks",
            "  points at mouse position during playback.",
            "  No clicking required - just move the mouse.",
            "",
            "- ClickPass Mode: Advances to next frame after",
            "  adding a marker (Normal Mode).",
            "",
            "Playback Speed:",
            "- [ : Slower",
            "- ] : Faster",
            "",
            "- ClickPass Mode: Advances to next frame after",
            "  adding a marker (Normal Mode).",
            "",
            "Playback Speed:",
            "- [ : Slower",
            "- ] : Faster",
            "",
            "Persistence Mode (P key):",
            "Shows markers from previous frames.",
            "- 1: Decrease persistence frames",
            "- 2: Increase persistence frames",
            "- 3: Toggle full persistence",
            "",
            "=== LABELING MODE DETAILS ===",
            "",
            "What it does:",
            "  Creates object detection dataset",
            "  for ML training (YOLO/COCO format)",
            "",
            "Export creates:",
            "  - train/ folder (70% of frames)",
            "  - val/ folder (20% of frames)",
            "  - test/ folder (10% of frames)",
            "  Each folder contains:",
            "    * images/ (frame images)",
            "    * labels/ (JSON annotations)",
            "",
            "Project Management:",
            "  - F5: Save Project & export dataset (dataset_YYYYMMDD_HHMMSS or append)",
            "  - F6: Load Labeling Project (JSON)",
            "  - F7: Load dataset folder (next Save appends; multi-video)",
            "  - F8: Open another video (keep dataset; no need to close app)",
            "  - N:  Rename Object Class",
            "",
            "Swap Markers:",
            "  - W: Open Swap Dialog (range swap)",
            "",
            "Press any key to close this help",
        ]

        # Create semi-transparent overlay
        overlay = pygame.Surface((window_width, window_height + 80))
        overlay.set_alpha(230)
        overlay.fill((0, 0, 0))

        # Render help text in two columns
        font = pygame.font.Font(None, 24)
        line_height = 28

        # Calculate column positions
        left_col_x = 20
        right_col_x = window_width // 2 + 10

        waiting_for_input = True
        scroll_offset = 0
        total_content_height = max(len(help_lines_left), len(help_lines_right)) * 28 + 40

        while waiting_for_input:
            # Re-render content based on scroll
            overlay.fill((0, 0, 0))  # Clear previous frame

            # Draw left column
            for i, line in enumerate(help_lines_left):
                y_pos = 20 + i * line_height - scroll_offset
                if -30 < y_pos < window_height + 30:  # Only draw visible
                    text_surface = font.render(line, True, (255, 255, 255))
                    overlay.blit(text_surface, (left_col_x, y_pos))

            # Draw right column
            for i, line in enumerate(help_lines_right):
                y_pos = 20 + i * line_height - scroll_offset
                if -30 < y_pos < window_height + 30:  # Only draw visible
                    text_surface = font.render(line, True, (255, 255, 255))
                    overlay.blit(text_surface, (right_col_x, y_pos))

            # Draw scroll bar if needed
            if total_content_height > window_height:
                scrollbar_x = window_width - 10
                view_ratio = window_height / total_content_height
                bar_height = max(30, int(window_height * view_ratio))
                scroll_ratio = scroll_offset / (total_content_height - window_height)
                bar_y = int(scroll_ratio * (window_height - bar_height))

                pygame.draw.rect(overlay, (100, 100, 100), (scrollbar_x, bar_y, 8, bar_height))

            screen.blit(overlay, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting_for_input = False
                    elif event.key == pygame.K_DOWN:
                        scroll_offset = min(
                            scroll_offset + 30, max(0, total_content_height - window_height)
                        )
                    elif event.key == pygame.K_UP:
                        scroll_offset = max(scroll_offset - 30, 0)
                    elif event.key == pygame.K_PAGEUP:
                        scroll_offset = max(scroll_offset - window_height + 50, 0)
                    elif event.key == pygame.K_PAGEDOWN:
                        scroll_offset = min(
                            scroll_offset + window_height - 50,
                            max(0, total_content_height - window_height),
                        )

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        waiting_for_input = False
                    elif event.button == 4:  # Scroll Up
                        scroll_offset = max(scroll_offset - 30, 0)
                    elif event.button == 5:  # Scroll Down
                        scroll_offset = min(
                            scroll_offset + 30, max(0, total_content_height - window_height)
                        )

                elif event.type == pygame.QUIT:
                    waiting_for_input = False
                    global running
                    running = False

    def show_persistence_settings():
        """Show a dialog to adjust persistence frames"""
        nonlocal persistence_frames

        # Create semi-transparent overlay
        overlay = pygame.Surface((window_width, window_height + 80))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))

        # Create UI elements
        if persistence_enabled and frame in bboxes:
            # Use a larger font for persistent boxes? Or same
            font = pygame.font.SysFont("verdana", 16)
        title = font.render("Persistence Settings", True, (255, 255, 255))

        instruction = font.render(
            "Use + and - keys to adjust frames, Enter to confirm", True, (255, 255, 255)
        )

        value_text = font.render(f"Frames: {persistence_frames}", True, (255, 255, 255))

        # Display overlay and UI
        screen.blit(overlay, (0, 0))
        screen.blit(
            title,
            (window_width // 2 - title.get_width() // 2, window_height // 2 - 100),
        )
        screen.blit(
            instruction,
            (window_width // 2 - instruction.get_width() // 2, window_height // 2 - 40),
        )
        screen.blit(
            value_text,
            (window_width // 2 - value_text.get_width() // 2, window_height // 2 + 20),
        )

        pygame.display.flip()

        # Handle input
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_input = False
                    global running
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        waiting_for_input = False
                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        persistence_frames += 1  # Sem limite máximo
                        value_text = font.render(
                            f"Frames: {persistence_frames}", True, (255, 255, 255)
                        )
                        # Redraw
                        screen.blit(overlay, (0, 0))
                        screen.blit(
                            title,
                            (
                                window_width // 2 - title.get_width() // 2,
                                window_height // 2 - 100,
                            ),
                        )
                        screen.blit(
                            instruction,
                            (
                                window_width // 2 - instruction.get_width() // 2,
                                window_height // 2 - 40,
                            ),
                        )
                        screen.blit(
                            value_text,
                            (
                                window_width // 2 - value_text.get_width() // 2,
                                window_height // 2 + 20,
                            ),
                        )
                        pygame.display.flip()
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        persistence_frames = max(1, persistence_frames - 1)
                        value_text = font.render(
                            f"Frames: {persistence_frames}", True, (255, 255, 255)
                        )
                        # Redraw
                        screen.blit(overlay, (0, 0))
                        screen.blit(
                            title,
                            (
                                window_width // 2 - title.get_width() // 2,
                                window_height // 2 - 100,
                            ),
                        )
                        screen.blit(
                            instruction,
                            (
                                window_width // 2 - instruction.get_width() // 2,
                                window_height // 2 - 40,
                            ),
                        )
                        screen.blit(
                            value_text,
                            (
                                window_width // 2 - value_text.get_width() // 2,
                                window_height // 2 + 20,
                            ),
                        )
                        pygame.display.flip()

    def show_input_dialog(prompt, initial_text=""):
        """Show a dialog to input text"""
        # Get current display size dynamically to support both Main Window and Dialog calls
        current_w, current_h = pygame.display.get_surface().get_size()

        # Create semi-transparent overlay
        overlay = pygame.Surface((current_w, current_h))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))

        # Create UI elements
        # Dialog font
        font = pygame.font.SysFont("verdana", 14)
        title = font.render(prompt, True, (255, 255, 255))

        input_text = initial_text

        waiting_for_input = True
        while waiting_for_input:
            overlay.fill((0, 0, 0))

            # Draw title
            screen.blit(overlay, (0, 0))
            screen.blit(title, (current_w // 2 - title.get_width() // 2, current_h // 2 - 50))

            # Draw input box
            input_surface = font.render(input_text + "_", True, (255, 255, 0))
            screen.blit(
                input_surface,
                (current_w // 2 - input_surface.get_width() // 2, current_h // 2 + 10),
            )

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_input = False
                    global running
                    running = False
                    return None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        waiting_for_input = False
                        return input_text
                    elif event.key == pygame.K_ESCAPE:
                        waiting_for_input = False
                        return None
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    else:
                        input_text += event.unicode
        return None

    def _deprecated_show_swap_dialog(current_frame, total_fr):
        """
        Show dialog to swap two markers over a frame range.
        Returns: dict with start_frame, end_frame, marker_1, marker_2, or None.
        """
        font = pygame.font.SysFont("verdana", 14)

        # Default values (1-based for user display)
        inputs = {
            "Start Frame": str(current_frame + 1),
            "End Frame": str(min(current_frame + 21, total_fr)),
            "Marker 1 ID": "1",
            "Marker 2 ID": "2",
        }
        order = ["Start Frame", "End Frame", "Marker 1 ID", "Marker 2 ID"]
        active_idx = 0

        box_width = 200
        box_height = 30
        margin = 10
        dialog_width = 400
        dialog_height = 300

        waiting = True
        while waiting:
            # Draw Overlay
            overlay = pygame.Surface((window_width, window_height + 80))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            # Dialog Box Background
            dx = (window_width - dialog_width) // 2
            dy = (window_height - dialog_height) // 2
            pygame.draw.rect(screen, (40, 40, 40), (dx, dy, dialog_width, dialog_height))
            pygame.draw.rect(screen, (100, 100, 100), (dx, dy, dialog_width, dialog_height), 2)

            # Title
            title = font.render("Swap Markers", True, (255, 255, 255))
            screen.blit(title, (dx + 20, dy + 20))

            # Draw Fields
            start_y = dy + 60
            for i, key in enumerate(order):
                # Label
                label = font.render(key + ":", True, (200, 200, 200))
                screen.blit(label, (dx + 30, start_y + i * (box_height + margin) + 5))

                # Input Box
                bx = dx + 180
                by = start_y + i * (box_height + margin)
                color = (255, 255, 255) if i == active_idx else (150, 150, 150)
                pygame.draw.rect(screen, (20, 20, 20), (bx, by, box_width - 40, box_height))
                pygame.draw.rect(screen, color, (bx, by, box_width - 40, box_height), 1)

                val_surf = font.render(inputs[key], True, (255, 255, 255))
                screen.blit(val_surf, (bx + 5, by + 5))

            # Submit Button
            submit_rect = pygame.Rect(dx + 100, dy + dialog_height - 50, 80, 30)
            cancel_rect = pygame.Rect(dx + 220, dy + dialog_height - 50, 80, 30)

            pygame.draw.rect(screen, (50, 150, 50), submit_rect)
            pygame.draw.rect(screen, (150, 50, 50), cancel_rect)

            submit_txt = font.render("Apply", True, (255, 255, 255))
            cancel_txt = font.render("Cancel", True, (255, 255, 255))

            screen.blit(submit_txt, (submit_rect.x + 20, submit_rect.y + 5))
            screen.blit(cancel_txt, (cancel_rect.x + 15, cancel_rect.y + 5))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    return None

                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    # Check fields
                    for i, _ in enumerate(order):
                        by = start_y + i * (box_height + margin)
                        bx = dx + 180
                        rect = pygame.Rect(bx, by, box_width - 40, box_height)
                        if rect.collidepoint(mx, my):
                            active_idx = i

                    if submit_rect.collidepoint(mx, my):
                        waiting = False
                        try:
                            return {
                                "start_frame": int(inputs["Start Frame"]) - 1,
                                "end_frame": int(inputs["End Frame"]) - 1,
                                "marker_1": int(inputs["Marker 1 ID"])
                                - 1,  # Convert 1-based to 0-based
                                "marker_2": int(inputs["Marker 2 ID"]) - 1,
                            }
                        except ValueError:
                            print("Invalid input")
                            return None

                    if cancel_rect.collidepoint(mx, my):
                        waiting = False
                        return None

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                        return None
                    elif event.key == pygame.K_TAB:
                        active_idx = (active_idx + 1) % 4
                    elif event.key == pygame.K_RETURN:
                        # Try submit
                        waiting = False
                        try:
                            return {
                                "start_frame": int(inputs["Start Frame"]),
                                "end_frame": int(inputs["End Frame"]),
                                "marker_1": int(inputs["Marker 1 ID"]) - 1,
                                "marker_2": int(inputs["Marker 2 ID"]) - 1,
                            }
                        except ValueError:
                            return None
                    elif event.key == pygame.K_BACKSPACE:
                        key = order[active_idx]
                        inputs[key] = inputs[key][:-1]
                    else:
                        key = order[active_idx]
                        if event.unicode.isnumeric():
                            inputs[key] += event.unicode
        return None

    def show_multi_swap_dialog(current_frame, total_fr, active_rules, markers_labels):
        """
        Show advanced Multi-Swap Manager Dialog.
        Allows selecting multiple pairs from a list and managing rules.
        """
        dialog_w, dialog_h = 900, 600
        d_surf = pygame.display.set_mode((dialog_w, dialog_h), pygame.RESIZABLE)
        pygame.display.set_caption("Multi-Swap Manager")

        font = pygame.font.SysFont("verdana", 14)
        title_font = pygame.font.SysFont("verdana", 16, bold=True)

        # Colors
        BG = (30, 30, 30)
        PANEL = (50, 50, 50)
        TEXT = (220, 220, 220)
        BTN_GREEN = (50, 150, 50)
        BTN_RED = (150, 50, 50)
        BTN_BLUE = (50, 80, 150)

        # Internal State
        temp_rules = active_rules.copy()  # List of dicts

        # Selection State
        selected_m1 = None  # Index
        selected_m2 = None  # Index

        # Frame Inputs
        input_start = str(current_frame + 1)
        input_end = str(min(current_frame + 21, total_fr))
        active_input = None  # 'start' or 'end'

        # Helper to draw text input
        def draw_input(surface, rect, text, active):
            color = (100, 150, 255) if active else (100, 100, 100)
            pygame.draw.rect(surface, color, rect, 1)
            txt_surf = font.render(text, True, TEXT)
            surface.blit(txt_surf, (rect.x + 5, rect.y + 5))

        # Helper to draw button
        def draw_button(surface, rect, text, color, hover=False):
            c = (
                (min(color[0] + 30, 255), min(color[1] + 30, 255), min(color[2] + 30, 255))
                if hover
                else color
            )
            pygame.draw.rect(surface, c, rect)
            txt = font.render(text, True, (255, 255, 255))
            surface.blit(
                txt,
                (
                    rect.x + (rect.w - txt.get_width()) // 2,
                    rect.y + (rect.h - txt.get_height()) // 2,
                ),
            )

        # Scroll state
        scroll_markers = 0
        scroll_rules = 0

        # Ensure labels are sufficient
        max_idx = 0
        if coordinates:
            for f_pts in coordinates.values():
                max_idx = max(max_idx, len(f_pts))

        display_labels = markers_labels.copy()
        if len(display_labels) < max_idx:
            for i in range(len(display_labels), max_idx):
                display_labels.append(f"Pixel {i + 1}")

        running = True
        while running:
            mx, my = pygame.mouse.get_pos()
            d_surf.fill(BG)

            # --- Left Panel: Markers List ---
            left_panel = pygame.Rect(10, 40, 300, dialog_h - 50)
            pygame.draw.rect(d_surf, PANEL, left_panel)
            d_surf.blit(title_font.render("Select Pair (Click 2)", True, TEXT), (10, 10))

            # Draw markers
            item_h = 30
            visible_count = left_panel.h // item_h

            for i in range(len(display_labels)):
                idx = i  # 0-based index
                if i < scroll_markers or i >= scroll_markers + visible_count:
                    continue

                y = left_panel.y + (i - scroll_markers) * item_h
                r = pygame.Rect(left_panel.x, y, left_panel.w, item_h - 1)

                # Determine color based on selection
                color = PANEL
                if idx == selected_m1:
                    color = (50, 100, 50)  # Greenish for first
                elif idx == selected_m2:
                    color = (100, 50, 50)  # Reddish for second
                elif r.collidepoint(mx, my):
                    color = (70, 70, 70)

                pygame.draw.rect(d_surf, color, r)

                # Text: "1: Left Hip"
                lbl = display_labels[i]
                d_surf.blit(font.render(f"{i + 1}: {lbl}", True, TEXT), (r.x + 10, r.y + 5))

            # --- Right Panel: Controls & Rules ---

            # Frame Range
            start_icon = font.render("Start Frame:", True, TEXT)
            d_surf.blit(start_icon, (330, 40))
            rect_start = pygame.Rect(430, 35, 100, 30)
            draw_input(d_surf, rect_start, input_start, active_input == "start")

            end_icon = font.render("End Frame:", True, TEXT)
            d_surf.blit(end_icon, (550, 40))
            rect_end = pygame.Rect(650, 35, 100, 30)
            draw_input(d_surf, rect_end, input_end, active_input == "end")

            # Selection Info
            sel_txt = "Selection: None"
            if selected_m1 is not None:
                l1 = display_labels[selected_m1]
                sel_txt = f"1: {l1}"
            if selected_m2 is not None:
                l2 = display_labels[selected_m2]
                sel_txt += f"  <-->  2: {l2}"
            d_surf.blit(font.render(sel_txt, True, (255, 200, 100)), (330, 80))

            # Add Swap Button
            btn_add = pygame.Rect(330, 110, 120, 35)
            draw_button(d_surf, btn_add, "Add Swap", BTN_BLUE, btn_add.collidepoint(mx, my))

            # Rules List
            d_surf.blit(
                title_font.render(f"Planned Swaps ({len(temp_rules)})", True, TEXT), (330, 160)
            )
            rule_panel = pygame.Rect(330, 190, 550, dialog_h - 260)
            pygame.draw.rect(d_surf, PANEL, rule_panel)

            rule_vis_count = rule_panel.h // 25
            for i, rule in enumerate(temp_rules):
                if i < scroll_rules or i >= scroll_rules + rule_vis_count:
                    continue
                y = rule_panel.y + (i - scroll_rules) * 25
                # Label indices: rule indices are 0-based, display as 1-based or Label name
                m1_idx = rule["marker_1"]
                m2_idx = rule["marker_2"]
                m1_name = (
                    display_labels[m1_idx]
                    if m1_idx < len(display_labels)
                    else f"Pixel {m1_idx + 1}"
                )
                m2_name = (
                    display_labels[m2_idx]
                    if m2_idx < len(display_labels)
                    else f"Pixel {m2_idx + 1}"
                )

                txt = f"[{rule['start_frame'] + 1}-{rule['end_frame'] + 1}] {m1_name} <-> {m2_name}"
                d_surf.blit(font.render(txt, True, TEXT), (rule_panel.x + 5, y + 2))

                # Delete X
                del_rect = pygame.Rect(rule_panel.right - 30, y, 25, 25)
                d_surf.blit(font.render("X", True, (255, 100, 100)), (del_rect.x + 5, del_rect.y))

            # Bottom Controls
            btn_apply = pygame.Rect(dialog_w - 150, dialog_h - 50, 130, 40)
            draw_button(d_surf, btn_apply, "Apply All", BTN_GREEN, btn_apply.collidepoint(mx, my))

            btn_cancel = pygame.Rect(dialog_w - 290, dialog_h - 50, 130, 40)
            draw_button(d_surf, btn_cancel, "Cancel", BTN_RED, btn_cancel.collidepoint(mx, my))

            btn_save = pygame.Rect(330, dialog_h - 50, 100, 40)
            draw_button(d_surf, btn_save, "Save TOML", BTN_BLUE, btn_save.collidepoint(mx, my))

            btn_load = pygame.Rect(440, dialog_h - 50, 100, 40)
            draw_button(d_surf, btn_load, "Load TOML", BTN_BLUE, btn_load.collidepoint(mx, my))

            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                elif event.type == pygame.VIDEORESIZE:
                    dialog_w, dialog_h = event.w, event.h
                    d_surf = pygame.display.set_mode((dialog_w, dialog_h), pygame.RESIZABLE)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left Click
                        # Handle Inputs
                        if rect_start.collidepoint(mx, my):
                            active_input = "start"
                        elif rect_end.collidepoint(mx, my):
                            active_input = "end"
                        else:
                            active_input = None

                        # Handle Marker List
                        if left_panel.collidepoint(mx, my):
                            idx = (my - left_panel.y) // item_h + scroll_markers
                            if 0 <= idx < len(display_labels):
                                if selected_m1 is None:
                                    selected_m1 = idx
                                elif selected_m2 is None and idx != selected_m1:
                                    selected_m2 = idx
                                else:
                                    # Reset if both selected or clicking same
                                    selected_m1 = idx
                                    selected_m2 = None

                        # Handle Add
                        if btn_add.collidepoint(mx, my):
                            if selected_m1 is not None and selected_m2 is not None:
                                try:
                                    s = int(input_start) - 1
                                    e = int(input_end) - 1
                                    temp_rules.append(
                                        {
                                            "start_frame": s,
                                            "end_frame": e,
                                            "marker_1": selected_m1,
                                            "marker_2": selected_m2,
                                        }
                                    )
                                    # Reset selection
                                    selected_m1 = None
                                    selected_m2 = None
                                except ValueError:
                                    print("Invalid frame range")

                        # Handle Delete Rule
                        if rule_panel.collidepoint(mx, my):
                            ridx = (my - rule_panel.y) // 25 + scroll_rules
                            if 0 <= ridx < len(temp_rules):
                                # We are just visualizing 'X' button logic roughly
                                if mx > rule_panel.right - 30:
                                    temp_rules.pop(ridx)

                        # Handle Footer
                        if btn_apply.collidepoint(mx, my):
                            return temp_rules
                        if btn_cancel.collidepoint(mx, my):
                            return None

                        # Save/Load
                        if btn_save.collidepoint(mx, my):
                            return ("SAVE", temp_rules)

                        if btn_load.collidepoint(mx, my):
                            return ("LOAD", temp_rules)

                    elif event.button == 4:  # Scroll Up
                        if left_panel.collidepoint(mx, my):
                            scroll_markers = max(0, scroll_markers - 1)
                        if rule_panel.collidepoint(mx, my):
                            scroll_rules = max(0, scroll_rules - 1)
                    elif event.button == 5:  # Scroll Down
                        if left_panel.collidepoint(mx, my):
                            scroll_markers = min(
                                max(0, len(display_labels) - visible_count), scroll_markers + 1
                            )
                        if rule_panel.collidepoint(mx, my):
                            scroll_rules = min(
                                max(0, len(temp_rules) - rule_vis_count), scroll_rules + 1
                            )

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None

                    if active_input:
                        if event.key == pygame.K_BACKSPACE:
                            if active_input == "start":
                                input_start = input_start[:-1]
                            else:
                                input_end = input_end[:-1]
                        elif event.unicode.isnumeric():
                            if active_input == "start":
                                input_start += event.unicode
                            else:
                                input_end += event.unicode
        return None

    def save_labeling_project():
        """
        Save the current labeling state to a JSON file INSIDE the dataset directory.
        If current_dataset_dir is set (e.g. after Load dataset folder), appends to that dataset.
        """
        nonlocal save_message_text, showing_save_message, save_message_timer

        # Export (new dataset next to video, or append to current_dataset_dir)
        dataset_dir, msg = export_labeling_dataset(
            video_path,
            bboxes,
            total_frames,
            original_width,
            original_height,
            output_dataset_dir=current_dataset_dir,
        )

        if not dataset_dir:
            save_message_text = f"Export failed: {msg}"
            showing_save_message = True
            save_message_timer = 120
            return

        # Prepare project metadata
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        project_file = os.path.join(dataset_dir, f"{base_name}_labeling_project.json")

        project_data = {
            "version": "0.3.0",
            "video_source": os.path.abspath(video_path),
            "dataset_root": os.path.abspath(dataset_dir),
            "current_label": current_label,
            "total_frames": total_frames,
            "bboxes": bboxes,
            "splits": {"train": "train", "val": "val", "test": "test"},
        }

        try:
            with open(project_file, "w") as f:
                json.dump(project_data, f, indent=4)

            save_message_text = "Project & Dataset Saved!" + (
                " (appended to dataset)" if current_dataset_dir else ""
            )
            showing_save_message = True
            save_message_timer = 90
            print(f"Project JSON saved to: {project_file}")
        except Exception as e:
            save_message_text = f"Error saving JSON: {e}"
            showing_save_message = True
            save_message_timer = 120

    def show_file_browser(start_dir, title="Select a file", extensions=None):
        """Pygame-based file browser with mouse navigation, Ctrl+V paste, and scroll.
        Returns selected file path or None on cancel (Escape).
        *extensions*: list of lowercase extensions to highlight, e.g. ['.yaml']
        """
        current_w, current_h = pygame.display.get_surface().get_size()
        font = pygame.font.SysFont("verdana", 14)
        small_font = pygame.font.SysFont("verdana", 12)
        title_font = pygame.font.SysFont("verdana", 15, bold=True)

        if extensions is None:
            extensions = []
        extensions = [e.lower() for e in extensions]

        # --- state ---
        cur_dir = (
            os.path.abspath(start_dir) if os.path.isdir(start_dir) else os.path.expanduser("~")
        )
        scroll_offset = 0
        selected_idx = -1  # hovered / clicked index
        input_text = cur_dir  # editable path bar
        input_active = False  # whether the path bar is focused
        entries = []  # list of (display_name, full_path, is_dir)
        filter_yaml_only = bool(extensions)  # toggle

        # Geometry constants
        margin = 12
        top_area = 90  # title + path bar
        bottom_area = 40  # status bar
        row_h = 24
        path_bar_h = 28
        scrollbar_w = 14

        def list_dir(d):
            """Return sorted entries: dirs first (alpha), then files (alpha)."""
            nonlocal entries, scroll_offset, selected_idx
            scroll_offset = 0
            selected_idx = -1
            dirs = []
            files = []
            try:
                for name in os.listdir(d):
                    full = os.path.join(d, name)
                    if os.path.isdir(full):
                        dirs.append((name + "/", full, True))
                    else:
                        if filter_yaml_only and extensions:
                            if not any(name.lower().endswith(ext) for ext in extensions):
                                continue
                        files.append((name, full, False))
            except PermissionError:
                pass
            dirs.sort(key=lambda x: x[0].lower())
            files.sort(key=lambda x: x[0].lower())
            # ".." entry to go up
            parent = os.path.dirname(d)
            entries = [("..", parent, True)] + dirs + files if parent != d else dirs + files

        list_dir(cur_dir)

        # Colors
        BG = (30, 30, 30)
        DIR_CLR = (100, 180, 255)
        FILE_CLR = (210, 210, 210)
        YAML_CLR = (120, 255, 120)
        HOVER_BG = (60, 60, 80)
        SEL_BG = (50, 80, 120)
        BAR_BG = (45, 45, 45)
        BAR_ACTIVE = (55, 55, 70)
        BORDER = (80, 80, 80)
        TITLE_CLR = (255, 255, 255)
        HINT_CLR = (160, 160, 160)

        clock = pygame.time.Clock()
        browsing = True
        result_path = None

        while browsing:
            clock.tick(30)
            current_w, current_h = pygame.display.get_surface().get_size()
            visible_rows = max(1, (current_h - top_area - bottom_area) // row_h)
            max_scroll = max(0, len(entries) - visible_rows)
            scroll_offset = max(0, min(scroll_offset, max_scroll))

            # ---- DRAW ----
            screen.fill(BG)

            # Title
            t_surf = title_font.render(title, True, TITLE_CLR)
            screen.blit(t_surf, (margin, margin))

            # Filter toggle hint
            filt_text = "[F] Filter: .yaml only" if filter_yaml_only else "[F] Filter: show all"
            filt_surf = small_font.render(filt_text, True, HINT_CLR)
            screen.blit(filt_surf, (current_w - filt_surf.get_width() - margin, margin + 4))

            # Path bar
            bar_y = margin + 30
            bar_rect = pygame.Rect(margin, bar_y, current_w - 2 * margin, path_bar_h)
            pygame.draw.rect(screen, BAR_ACTIVE if input_active else BAR_BG, bar_rect)
            pygame.draw.rect(screen, BORDER, bar_rect, 1)
            # Render path text (clip to bar)
            clip_w = bar_rect.width - 8
            display_path = input_text
            path_surf = font.render(
                display_path + ("_" if input_active else ""),
                True,
                (255, 255, 0) if input_active else (200, 200, 200),
            )
            # If text wider than bar, show end
            offset_x = path_surf.get_width() - clip_w if path_surf.get_width() > clip_w else 0
            screen.blit(
                path_surf,
                (bar_rect.x + 4 - offset_x, bar_rect.y + 5),
                area=pygame.Rect(offset_x, 0, clip_w, path_bar_h),
            )

            # File list area
            list_y = top_area
            list_h = current_h - top_area - bottom_area
            list_rect = pygame.Rect(0, list_y, current_w - scrollbar_w, list_h)

            mouse_pos = pygame.mouse.get_pos()

            for i in range(visible_rows):
                idx = i + scroll_offset
                if idx >= len(entries):
                    break
                name, full, is_dir = entries[idx]
                row_rect = pygame.Rect(0, list_y + i * row_h, current_w - scrollbar_w, row_h)

                # Hover highlight
                if row_rect.collidepoint(mouse_pos) and not input_active:
                    pygame.draw.rect(screen, HOVER_BG, row_rect)
                if idx == selected_idx:
                    pygame.draw.rect(screen, SEL_BG, row_rect)

                # Choose color
                if is_dir:
                    clr = DIR_CLR
                elif extensions and any(name.lower().endswith(ext) for ext in extensions):
                    clr = YAML_CLR
                else:
                    clr = FILE_CLR

                name_surf = font.render(name, True, clr)
                screen.blit(name_surf, (margin + 8, list_y + i * row_h + 3))

            # Scrollbar
            if len(entries) > visible_rows:
                sb_x = current_w - scrollbar_w
                sb_total_h = list_h
                sb_thumb_h = max(20, int(sb_total_h * visible_rows / len(entries)))
                sb_thumb_y = (
                    list_y + int((sb_total_h - sb_thumb_h) * scroll_offset / max_scroll)
                    if max_scroll
                    else list_y
                )
                pygame.draw.rect(screen, (50, 50, 50), (sb_x, list_y, scrollbar_w, sb_total_h))
                pygame.draw.rect(
                    screen, (120, 120, 120), (sb_x + 2, sb_thumb_y, scrollbar_w - 4, sb_thumb_h)
                )

            # Bottom status bar
            status_y = current_h - bottom_area
            pygame.draw.line(screen, BORDER, (0, status_y), (current_w, status_y))
            hint = "Click file to select | Enter=confirm path bar | Esc=cancel | Ctrl+V=paste | Scroll=mouse wheel"
            hint_surf = small_font.render(hint, True, HINT_CLR)
            screen.blit(hint_surf, (margin, status_y + 10))

            pygame.display.flip()

            # ---- EVENTS ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    browsing = False
                    global running
                    running = False
                    return None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Click on path bar?
                    if bar_rect.collidepoint(event.pos):
                        input_active = True
                        selected_idx = -1
                        continue
                    else:
                        if input_active:
                            input_active = False

                    # Click on file list?
                    if list_rect.collidepoint(event.pos):
                        row_i = (event.pos[1] - list_y) // row_h
                        idx = row_i + scroll_offset
                        if 0 <= idx < len(entries):
                            name, full, is_dir = entries[idx]
                            if is_dir:
                                # Navigate into directory
                                cur_dir = os.path.abspath(full)
                                input_text = cur_dir
                                list_dir(cur_dir)
                            else:
                                # Select file → return it
                                result_path = full
                                browsing = False

                    # Scroll wheel
                    if event.button == 4:  # scroll up
                        scroll_offset = max(0, scroll_offset - 3)
                    elif event.button == 5:  # scroll down
                        scroll_offset = min(max_scroll, scroll_offset + 3)

                elif event.type == pygame.MOUSEWHEEL:
                    scroll_offset = max(0, min(max_scroll, scroll_offset - event.y * 3))

                elif event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()

                    # Ctrl+V paste (works whether path bar is active or not)
                    if event.key == pygame.K_v and (mods & pygame.KMOD_CTRL):
                        try:
                            clip = pygame.scrap.get(pygame.SCRAP_TEXT)
                            if clip:
                                pasted = (
                                    clip.decode("utf-8", errors="ignore").rstrip("\x00").strip()
                                )
                                if pasted:
                                    input_text = pasted
                                    input_active = True
                        except Exception:
                            pass
                        continue

                    if event.key == pygame.K_ESCAPE:
                        browsing = False
                        result_path = None

                    elif event.key == pygame.K_F5 or (
                        event.key == pygame.K_r and (mods & pygame.KMOD_CTRL)
                    ):
                        # Refresh
                        list_dir(cur_dir)

                    elif event.key == pygame.K_f and not input_active:
                        # Toggle filter
                        filter_yaml_only = not filter_yaml_only
                        list_dir(cur_dir)

                    elif input_active:
                        if event.key == pygame.K_RETURN:
                            # Confirm typed path
                            typed = input_text.strip()
                            if os.path.isfile(typed):
                                result_path = typed
                                browsing = False
                            elif os.path.isdir(typed):
                                cur_dir = os.path.abspath(typed)
                                input_text = cur_dir
                                list_dir(cur_dir)
                                input_active = False
                            else:
                                # Try parent dir if partial
                                parent = os.path.dirname(typed)
                                if os.path.isdir(parent):
                                    cur_dir = os.path.abspath(parent)
                                    input_text = cur_dir
                                    list_dir(cur_dir)
                                    input_active = False
                        elif event.key == pygame.K_TAB:
                            # Tab autocomplete
                            typed = input_text.strip()
                            if os.path.isdir(typed):
                                base_dir = typed
                                prefix = ""
                            else:
                                base_dir = os.path.dirname(typed)
                                prefix = os.path.basename(typed).lower()
                            if os.path.isdir(base_dir):
                                try:
                                    matches = [
                                        n
                                        for n in os.listdir(base_dir)
                                        if n.lower().startswith(prefix)
                                    ]
                                except PermissionError:
                                    matches = []
                                if len(matches) == 1:
                                    completed = os.path.join(base_dir, matches[0])
                                    if os.path.isdir(completed):
                                        completed += os.sep
                                    input_text = completed
                                elif len(matches) > 1:
                                    # Find common prefix
                                    common = os.path.commonprefix(matches)
                                    if common and len(common) > len(prefix):
                                        input_text = os.path.join(base_dir, common)
                        elif event.key == pygame.K_BACKSPACE:
                            if mods & pygame.KMOD_CTRL:
                                # Delete last path component
                                input_text = os.path.dirname(input_text.rstrip(os.sep))
                            else:
                                input_text = input_text[:-1]
                        elif event.key == pygame.K_a and (mods & pygame.KMOD_CTRL):
                            pass  # select all – no-op for now
                        elif event.unicode and event.unicode.isprintable():
                            input_text += event.unicode

        return result_path

    def load_dataset_folder():
        """Select dataset by choosing its data.yaml file; future Saves (F5) will append (multi-video).
        On Linux uses the pygame file browser to avoid tk/zenity freezing."""
        nonlocal \
            current_dataset_dir, \
            current_label, \
            save_message_text, \
            showing_save_message, \
            save_message_timer

        initial_dir = os.path.dirname(video_path) if video_path else os.path.expanduser("~")
        start_browse = current_dataset_dir or initial_dir
        folder = ""
        yaml_path = None
        if sys.platform == "linux":
            # On Linux use the pygame file browser (tk/zenity freeze the app)
            yaml_path = show_file_browser(
                start_browse,
                title="Select data.yaml of the dataset",
                extensions=[".yaml"],
            )
        else:
            # On Windows/macOS use tkinter file dialog
            try:
                from tkinter import Tk, filedialog

                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                root.update_idletasks()
                yaml_path = filedialog.askopenfilename(
                    title="Select data.yaml of the dataset (folder will be used)",
                    initialdir=initial_dir,
                    filetypes=[
                        ("YAML files", "*.yaml"),
                        ("data.yaml", "data.yaml"),
                        ("All files", "*.*"),
                    ],
                )
                root.destroy()
            except Exception:
                pass
        if yaml_path and os.path.isfile(yaml_path):
            folder = os.path.dirname(yaml_path)
        elif yaml_path and os.path.isdir(yaml_path):
            folder = yaml_path
        if not folder or not os.path.isdir(folder):
            if folder:
                save_message_text = "Invalid or missing path."
            showing_save_message = True
            save_message_timer = 60
            return
        train_img = os.path.join(folder, "train", "images")
        classes_file = os.path.join(folder, "classes.txt")
        if not os.path.isdir(train_img):
            save_message_text = "Folder must contain train/images/ (YOLO dataset)."
            showing_save_message = True
            save_message_timer = 90
            return
        current_dataset_dir = os.path.abspath(folder)
        if os.path.exists(classes_file):
            try:
                with open(classes_file, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                if lines:
                    current_label = lines[0]
            except Exception:
                pass

        # Auto-load the project JSON for the current video (if it exists in this dataset)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        project_file = os.path.join(current_dataset_dir, f"{base_name}_labeling_project.json")
        if os.path.exists(project_file):
            load_labeling_project()  # will use current_dataset_dir we just set
        else:
            save_message_text = (
                f"Dataset set: {os.path.basename(folder)}. No previous labels for this video."
            )
            showing_save_message = True
            save_message_timer = 120

    def load_labeling_project():
        """Load labeling state from a JSON file inside the dataset directory.
        Uses current_dataset_dir when set (multi-video workflow); otherwise
        falls back to the default <videoname>_dataset/ beside the video."""
        nonlocal \
            bboxes, \
            current_label, \
            current_dataset_dir, \
            save_message_text, \
            showing_save_message, \
            save_message_timer

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)

        # Determine dataset directory: prefer current_dataset_dir if already set
        if current_dataset_dir and os.path.isdir(current_dataset_dir):
            dataset_dir = current_dataset_dir
        else:
            dataset_dir = os.path.join(video_dir, f"{base_name}_dataset")

        # Project file: <dataset_dir>/<video_base_name>_labeling_project.json
        project_file = os.path.join(dataset_dir, f"{base_name}_labeling_project.json")

        if not os.path.exists(project_file):
            # Also try without video prefix (legacy: frame_XXXXXX naming)
            alt_project = os.path.join(dataset_dir, "labeling_project.json")
            if os.path.exists(alt_project):
                project_file = alt_project
            else:
                save_message_text = "No project file for this video in dataset."
                showing_save_message = True
                save_message_timer = 60
                print(f"Tried loading: {project_file}")
                return

        try:
            with open(project_file) as f:
                project_data = json.load(f)

            # Restore data
            loaded_bboxes = project_data.get("bboxes", {})
            bboxes.clear()
            for k, v in loaded_bboxes.items():
                bboxes[int(k)] = v

            current_label = project_data.get("current_label", "object")
            # So next Save appends to same dataset when continuing from another video
            if "dataset_root" in project_data and os.path.isdir(project_data["dataset_root"]):
                current_dataset_dir = project_data["dataset_root"]

            save_message_text = (
                f"Project Loaded: {len(bboxes)} frames from {os.path.basename(dataset_dir)}"
            )
            showing_save_message = True
            save_message_timer = 90
        except Exception as e:
            save_message_text = f"Error loading project: {e}"
            showing_save_message = True
            save_message_timer = 120

    def save_1_line_coordinates(video_path, one_line_markers, deleted_markers=None):
        """Save markers created in one-line mode to a CSV file."""
        if not one_line_markers:
            print("No one line markers to save.")
            return

        if deleted_markers is None:
            deleted_markers = set()

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        output_file = os.path.join(video_dir, f"{base_name}_markers_1_line.csv")

        # Find largest marker index (accounting for deleted ones)
        max_marker = max(
            [idx for idx, _ in enumerate(one_line_markers) if idx not in deleted_markers],
            default=0,
        )

        # Create header: frame column and p1_x, p1_y, p2_x, p2_y, etc.
        header = ["frame"]
        for i in range(1, max_marker + 2):  # +2 because we need to add one more than max index
            header.extend([f"p{i}_x", f"p{i}_y"])

        # Get the frame number from the first non-deleted marker
        if deleted_markers is None:
            deleted_markers = set()

        # Get the frame number from the first non-deleted marker
        first_marker = next(
            (m for idx, m in enumerate(one_line_markers) if idx not in deleted_markers),
            None,
        )
        if first_marker is None:
            print("No valid markers to save after deletions.")
            return

        # Fill the row with values, preserving marker indices even with deletions
        row_values = [int(first_marker[0])]  # First marker's frame value

        # Initialize all positions with empty values
        for _ in range(max_marker + 1):
            row_values.extend(["", ""])  # Empty x, y values

        # Fill in the non-deleted marker positions
        for idx, (_, x, y) in enumerate(one_line_markers):
            if idx not in deleted_markers:
                # Verificar se é um marcador vazio (None)
                if x is not None and y is not None:
                    # Marker indices are 1-based in the CSV
                    row_values[idx * 2 + 1] = float(
                        x
                    )  # +1 for frame column, then multiply by 2 for x position
                    row_values[idx * 2 + 2] = float(y)  # +2 for y position
                # Se for None, deixar como vazio (já inicializado como "")

        df = pd.DataFrame([row_values], columns=header)
        df.to_csv(output_file, index=False)

        print(f"1 line coordinates saved to: {output_file}")
        return output_file

    def add_new_marker():
        """Adiciona um novo marcador vazio após o último marcador visível"""
        nonlocal \
            coordinates, \
            one_line_markers, \
            selected_marker_idx, \
            showing_save_message, \
            save_message_timer, \
            save_message_text

        if one_line_mode:
            # In one-line mode, find the largest visible marker index
            visible_markers = []
            for idx, _ in enumerate(one_line_markers):
                if idx not in deleted_markers:
                    visible_markers.append(idx)

            new_idx = max(visible_markers) + 1 if visible_markers else 0

            # Add empty marker at the current frame (using None instead of 0,0)
            one_line_markers.append((frame_count, None, None))
            selected_marker_idx = new_idx

            save_message_text = f"Added new empty marker {new_idx + 1}"
            showing_save_message = True
            save_message_timer = 60
        else:
            # In normal mode, we'll check the maximum number of visible markers
            max_visible_marker = -1

            for frame in range(total_frames):
                for i in range(len(coordinates[frame])):
                    if i not in deleted_positions[frame]:
                        max_visible_marker = max(max_visible_marker, i)

            new_marker_idx = max_visible_marker + 1

            # Add one more marker at each frame with empty position (None, None)
            for frame in range(total_frames):
                while len(coordinates[frame]) <= new_marker_idx:
                    coordinates[frame].append((None, None))

                # Add the marker as "deleted" in all frames except the current one
                if frame != frame_count:
                    deleted_positions[frame].add(new_marker_idx)

            # Remove the marker from the deleted list in the current frame to make it visible
            if new_marker_idx in deleted_positions[frame_count]:
                deleted_positions[frame_count].remove(new_marker_idx)

            # Select the new added marker
            selected_marker_idx = new_marker_idx

            save_message_text = f"Added new empty marker {new_marker_idx + 1}"
            showing_save_message = True
            save_message_timer = 60

        # Make automatic backup of the original file
        make_backup()

    def remove_marker():
        """Remove the selected marker only in the current frame"""
        nonlocal \
            coordinates, \
            one_line_markers, \
            selected_marker_idx, \
            showing_save_message, \
            save_message_timer, \
            save_message_text

        if one_line_mode:
            if selected_marker_idx >= 0:
                # Find and remove the selected marker only in the current frame
                for i, (f_num, _, _) in enumerate(one_line_markers):
                    if i == selected_marker_idx and f_num == frame_count:
                        # Instead of removing completely, add to the deleted markers list
                        deleted_markers.add(selected_marker_idx)
                        save_message_text = (
                            f"Removed marker {selected_marker_idx + 1} in the current frame"
                        )
                        showing_save_message = True
                        save_message_timer = 60
                        break
            else:
                save_message_text = "No marker selected to remove"
                showing_save_message = True
                save_message_timer = 60
        else:
            if selected_marker_idx >= 0:
                # Add the selected marker to the deleted list only in the current frame
                if selected_marker_idx < len(coordinates[frame_count]):
                    deleted_positions[frame_count].add(selected_marker_idx)
                    save_message_text = (
                        f"Removed marker {selected_marker_idx + 1} in the current frame"
                    )
                    showing_save_message = True
                    save_message_timer = 60
                else:
                    save_message_text = "Marker does not exist in this frame"
                    showing_save_message = True
                    save_message_timer = 60
            else:
                save_message_text = "No marker selected to remove"
                showing_save_message = True
                save_message_timer = 60

        # Make automatic backup of the original file
        make_backup()

    def make_backup():
        """Make a backup of the original coordinates file with timestamp"""
        if not os.path.exists(video_path):
            return

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if there is a normal coordinates file
        coords_file = os.path.join(video_dir, f"{base_name}_markers.csv")
        if os.path.exists(coords_file):
            backup_file = os.path.join(video_dir, f"{base_name}_markers_bk_{timestamp}.csv")
            try:
                import shutil

                shutil.copy2(coords_file, backup_file)
                print(f"Backup created: {backup_file}")
            except Exception as e:
                print(f"Error making backup: {e}")

        # Check if there is a sequential file
        seq_file = os.path.join(video_dir, f"{base_name}_markers_sequential.csv")
        if os.path.exists(seq_file):
            backup_file = os.path.join(
                video_dir, f"{base_name}_markers_sequential_bk_{timestamp}.csv"
            )
            try:
                import shutil

                shutil.copy2(seq_file, backup_file)
                print(f"Sequential backup created: {backup_file}")
            except Exception as e:
                print(f"Error making sequential backup: {e}")

        # Also check the 1 line file
        line_file = os.path.join(video_dir, f"{base_name}_markers_1_line.csv")
        if os.path.exists(line_file):
            backup_file = os.path.join(video_dir, f"{base_name}_markers_1_line_bk_{timestamp}.csv")
            try:
                import shutil

                shutil.copy2(line_file, backup_file)
                print(f"1-Line backup created: {backup_file}")
            except Exception as e:
                print(f"Error trying to backup 1-line: {e}")

    def show_file_path_dialog():
        """
        Select a CSV file or YOLO dataset directory.
        On Linux: Uses Pygame-native dialog to avoid conflicts/freezes.
        On others: Uses Tkinter dialog with option to select file or directory.
        """
        import platform

        # Determine initial directory from video path
        initial_dir = os.path.dirname(video_path) if video_path else os.path.expanduser("~")

        if platform.system() == "Linux":
            # Use the Pygame native dialog (CSV files only for now)
            # User can manually type directory path if needed
            return pygame_file_dialog(
                initial_dir=initial_dir,
                file_extensions=[".csv"],
                restore_size=(window_width, window_height + 80),
            )
        else:
            # Use Tkinter for Windows/Mac with option to select file or directory
            from tkinter import Tk, messagebox
            from tkinter.filedialog import askdirectory, askopenfilename

            # Block Pygame from processing mouse and keyboard events while Tkinter is open
            pygame.event.set_blocked(
                [
                    pygame.MOUSEBUTTONDOWN,
                    pygame.MOUSEBUTTONUP,
                    pygame.MOUSEMOTION,
                    pygame.KEYDOWN,
                    pygame.KEYUP,
                ]
            )

            # Clear any pending events
            pygame.event.clear()

            try:
                # Create Tkinter root window (hidden)
                root = Tk()
                root.withdraw()  # Hide the root window
                root.attributes("-topmost", True)  # Bring to front
                root.update_idletasks()  # Ensure window is ready

                # Ask user what they want to load
                choice = messagebox.askyesnocancel(
                    "Select Input Type",
                    "What do you want to load?\n\n"
                    "Yes = CSV file\n"
                    "No = YOLO dataset folder\n"
                    "Cancel = Cancel",
                )

                if choice is True:
                    # CSV file
                    filename = askopenfilename(
                        title="Select CSV File",
                        initialdir=initial_dir,
                        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                    )
                    result = filename if filename else None
                elif choice is False:
                    # YOLO dataset folder
                    directory = askdirectory(
                        title="Select YOLO Dataset Folder", initialdir=initial_dir
                    )
                    result = directory if directory else None
                else:
                    # Cancelled
                    result = None

            except Exception as e:
                print(f"Error in file dialog: {e}")
                result = None
            finally:
                # Clean up Tkinter
                with suppress(BaseException):
                    root.destroy()

                # Re-enable Pygame events
                pygame.event.set_allowed(
                    [
                        pygame.MOUSEBUTTONDOWN,
                        pygame.MOUSEBUTTONUP,
                        pygame.MOUSEMOTION,
                        pygame.KEYDOWN,
                        pygame.KEYUP,
                    ]
                )

                # Clear any events that accumulated while dialog was open
                pygame.event.clear()

            return result

    def reload_coordinates():
        """Load a new coordinates file or YOLO dataset during execution"""
        nonlocal \
            coordinates, \
            one_line_markers, \
            deleted_markers, \
            deleted_positions, \
            selected_marker_idx, \
            one_line_mode, \
            save_message_text, \
            showing_save_message, \
            save_message_timer

        # Make backup of the current before loading a new one
        make_backup()

        # Use simple pygame dialog to enter file/folder path
        # First, try to detect if user wants to select a folder (YOLO dataset)
        # We'll use a simple approach: check if the selected path is a directory

        # For now, use file dialog but allow directory selection
        # On Linux, use pygame dialog; on Windows/Mac, use tkinter with directory option
        import platform

        initial_dir = os.path.dirname(video_path) if video_path else os.path.expanduser("~")
        input_path = None

        if platform.system() == "Linux":
            # Use pygame dialog on Linux - first try CSV file dialog, then allow manual path entry
            # Try CSV file dialog first
            csv_path = pygame_file_dialog(
                initial_dir=initial_dir,
                file_extensions=[".csv"],
                restore_size=(window_width, window_height + 80),
            )

            if csv_path:
                input_path = csv_path
            else:
                # If cancelled, try input dialog for directory path
                input_path = show_input_dialog(
                    "Enter CSV file path or YOLO dataset folder path:", ""
                )
        else:
            # Use tkinter with both file and directory options on Windows/Mac
            from tkinter import Tk, filedialog

            pygame.event.set_blocked(
                [
                    pygame.MOUSEBUTTONDOWN,
                    pygame.MOUSEBUTTONUP,
                    pygame.MOUSEMOTION,
                    pygame.KEYDOWN,
                    pygame.KEYUP,
                ]
            )
            pygame.event.clear()

            try:
                root = Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                root.update_idletasks()

                # Ask user what they want to load
                import tkinter.messagebox as msgbox

                choice = msgbox.askyesnocancel(
                    "Select Input Type",
                    "What do you want to load?\n\n"
                    "Yes = CSV file\n"
                    "No = YOLO dataset folder\n"
                    "Cancel = Cancel",
                )

                if choice is True:
                    # CSV file
                    input_path = filedialog.askopenfilename(
                        title="Select CSV File",
                        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                        initialdir=initial_dir,
                    )
                elif choice is False:
                    # YOLO dataset folder
                    input_path = filedialog.askdirectory(
                        title="Select YOLO Dataset Folder", initialdir=initial_dir
                    )
                else:
                    # Cancelled
                    input_path = None

                root.destroy()
            except Exception as e:
                print(f"Error in file dialog: {e}")
                input_path = None
            finally:
                pygame.event.set_allowed(
                    [
                        pygame.MOUSEBUTTONDOWN,
                        pygame.MOUSEBUTTONUP,
                        pygame.MOUSEMOTION,
                        pygame.KEYDOWN,
                        pygame.KEYUP,
                    ]
                )
                pygame.event.clear()

        # If user cancelled or error occurred, show message
        if not input_path:
            save_message_text = "File/folder selection cancelled."
            showing_save_message = True
            save_message_timer = 60
            return

        # Check if it's a YOLO dataset directory
        is_yolo, images_dir, labels_dir, classes_file = is_yolo_dataset(input_path)

        if is_yolo:
            try:
                print(f"Loading YOLO dataset from: {input_path}")
                loaded_coords = load_yolo_dataset(
                    input_path, video_path, total_frames, original_width, original_height
                )

                if loaded_coords:
                    coordinates = loaded_coords
                    # Fix: Re-apply active swap rules after loading new data
                    if active_swap_rules:
                        print(f"Re-applying {len(active_swap_rules)} swap rules to loaded data...")
                        coordinates = apply_swap_config(coordinates, active_swap_rules)

                    deleted_positions = {i: set() for i in range(total_frames)}
                    one_line_mode = False
                    selected_marker_idx = 0

                    total_loaded = sum(len(coords) for coords in coordinates.values())
                    save_message_text = f"Loaded YOLO dataset: {total_loaded} annotations"
                    showing_save_message = True
                    save_message_timer = 90
                else:
                    save_message_text = "Failed to load YOLO dataset"
                    showing_save_message = True
                    save_message_timer = 90
                return
            except Exception as e:
                save_message_text = f"Error loading YOLO dataset: {e}"
                showing_save_message = True
                save_message_timer = 90
                return

        # Otherwise, treat as CSV file
        input_file = input_path

        try:
            # Check if it's a 1 line or normal file
            df = pd.read_csv(input_file)
            if "_1_line" in input_file or len(df) == 1:
                # Probably a 1 line file
                one_line_markers = []
                deleted_markers = set()

                for _, row in df.iterrows():
                    frame_num = int(row["frame"])
                    for i in range(1, 1001):  # Increased to support up to 1000 markers
                        x_col = f"p{i}_x"
                        y_col = f"p{i}_y"
                        if x_col in df.columns and y_col in df.columns:
                            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                                one_line_markers.append((frame_num, row[x_col], row[y_col]))

                save_message_text = f"Loaded 1 line file: {os.path.basename(input_file)}"
                # If it was in normal mode, switch to 1 line mode
                one_line_mode = True
            else:
                # Normal coordinates file
                coordinates = {i: [] for i in range(total_frames)}
                deleted_positions = {i: set() for i in range(total_frames)}

                for _, row in df.iterrows():
                    frame_num = int(row["frame"])
                    for i in range(1, 1001):  # Increased to support up to 1000 markers
                        x_col = f"p{i}_x"
                        y_col = f"p{i}_y"
                        if x_col in df.columns and y_col in df.columns:
                            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                                coordinates[frame_num].append((row[x_col], row[y_col]))

                # Fix: Re-apply active swap rules after loading new data
                if active_swap_rules:
                    print(f"Re-applying {len(active_swap_rules)} swap rules to loaded data...")
                    coordinates = apply_swap_config(coordinates, active_swap_rules)

                save_message_text = f"Loaded file: {os.path.basename(input_file)}"
                # If it was in 1 line mode, switch to normal mode
                one_line_mode = False

            # Always initialize on the first marker (index 0)
            selected_marker_idx = 0

            showing_save_message = True
            save_message_timer = 90

        except Exception as e:
            save_message_text = f"Error loading file: {e}"
            showing_save_message = True
            save_message_timer = 90

    # --- Key binding reference (main window only; check for overlaps when adding new keys) ---
    # ESC=quit | Space=play/pause | Left/Right/Up/Down=seek (when paused) | -=zoom out
    # C=1-line mode | M=auto-marking | D=remove last marker | L=labeling mode
    # Z=remove last bbox (labeling) | N=rename label (labeling) | F5=Save project | F6=Load project
    # F7=Load dataset folder | F8=Open another video | Tab=next marker | P=persistence
    # 1/2/3=persistence frames | H=help | A=add marker | R or D=remove last marker
    # ]=speed x2 | [=speed /2 | J=pose (MediaPipe) | W=swap dialog | O or S=sequential mode
    # (Save/Load coordinates = buttons only; no key overlap)
    running = True
    saved = False
    showing_save_message = False
    save_message_timer = 0
    save_message_text = ""

    last_valid_frame = None
    while running:
        # Once per run: if we have a dataset dir, load project for this video
        if current_dataset_dir and not auto_load_project_done:
            auto_load_project_done = True
            load_labeling_project()

        if paused:
            # When paused, we'll use the set method to position exactly on the frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
        else:
            # When playing, handle speed logic
            # Calculate frames to advance based on speed
            # Accumulate fractional frames for smooth slow playback is tricky in simple loop
            # Simple integer logic:
            if playback_speed >= 1.0:
                # Setup skip
                skip = int(playback_speed)
                for _ in range(skip):
                    ret, frame = cap.read()
                    if not ret:
                        break
                if ret:
                    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                # Slow motion: skip READING new frames based on speed
                # E.g. 0.5X -> read every 2nd loop
                # This requires keeping track of time or loop counts
                # Let's rely on slowing down the loop tick? No, simpler to just wait
                # To simulate slow motion without complex buffering, we just don't read every frame
                # Actually, cap.read() advances the frame. If we want slowmo, we must NOT call read()
                # and use the LAST read frame for multiple loop iterations.

                # Simple implementation: use a counter
                # Initialize static var for slow mo if not exists (hacky in local scope but works if var external)
                # Better: use floating point frame index target?

                # Let's try controlling the actual frame position
                # This is safer for consistency

                # We need to accumulate this... but we don't have a persistent accumulator easily here
                # without restructuring.

                # Alternative: Use Time-based approach
                # But for now, let's use the simple logic valid for high speeds, and for low speeds
                # we just modify wait time?
                # The loop runs at 30Hz (clock.tick(30)).
                # To get 0.5X speed (15fps effective), we should update frame every 2 ticks.

                if not hasattr(play_video_with_controls, "slow_mo_accumulator"):
                    play_video_with_controls.slow_mo_accumulator = 0.0

                play_video_with_controls.slow_mo_accumulator += playback_speed
                if play_video_with_controls.slow_mo_accumulator >= 1.0:
                    play_video_with_controls.slow_mo_accumulator -= 1.0
                    ret, frame = cap.read()
                    if ret:
                        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                else:
                    # Don't read, just keep current frame (last_valid_frame)
                    # But we need 'ret' to be True to show something
                    if last_valid_frame is not None:
                        frame = last_valid_frame.copy()
                        ret = True
                    else:
                        ret, frame = cap.read()  # Force read first time
                        if ret:
                            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if not ret:
                # End of video reached or short/merged video read failure.
                # Gracefully pause at the last known frame instead of exiting.
                paused = True
                # If total_frames is known, clamp to last frame; else keep current index
                if total_frames and total_frames > 0:
                    frame_count = max(0, min(frame_count, total_frames - 1))
                # Try to show the last valid frame if available
                if last_valid_frame is not None:
                    frame = last_valid_frame.copy()
                    ret = True
                else:
                    # Attempt to reposition and read the current frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()
                    if not ret:
                        # If still failing, do not crash: continue loop in paused state
                        # and wait for user input (e.g., ESC or navigation)
                        clock.tick(30)
                        continue

        if not ret:
            # If we still couldn't get a frame, try to keep showing the last valid one
            if last_valid_frame is not None:
                frame = last_valid_frame.copy()
                ret = True
            else:
                # Nothing to show; avoid abrupt exit, but limit CPU
                clock.tick(30)
                continue
        else:
            last_valid_frame = frame

        # Apply zoom
        zoomed_width = int(original_width * zoom_level)
        zoomed_height = int(original_height * zoom_level)
        zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))
        crop_x = int(max(0, min(zoomed_width - window_width, offset_x)))
        crop_y = int(max(0, min(zoomed_height - window_height, offset_y)))
        cropped_frame = zoomed_frame[
            crop_y : crop_y + window_height, crop_x : crop_x + window_width
        ]

        frame_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )
        screen.blit(frame_surface, (0, 0))

        # Draw persistent markers first (draw in order from oldest to newest)
        # Draw persistent markers first (draw in order from oldest to newest)
        font = pygame.font.SysFont("verdana", 14)
        if persistence_enabled:
            # Calculate range of frames to show
            start_frame = max(0, frame_count - persistence_frames)

            # Collect marker positions across frames for each ID
            marker_trails = {}  # Dictionary: marker_id -> list of (frame_num, x, y) points

            if one_line_mode:
                # In one_line mode, collect marker positions by their index
                for idx, (f_num, x, y) in enumerate(one_line_markers):
                    if idx in deleted_markers:
                        continue  # Skip deleted markers

                    if start_frame <= f_num <= frame_count:
                        if idx not in marker_trails:
                            marker_trails[idx] = []
                        # Store with frame number to sort by time later
                        marker_trails[idx].append((f_num, x, y))
            else:
                # In regular mode, need to track same marker ID across frames
                for f_num in range(start_frame, frame_count + 1):
                    for i, (x, y) in enumerate(coordinates[f_num]):
                        if i in deleted_positions[f_num]:
                            continue  # Skip deleted markers

                        if i not in marker_trails:
                            marker_trails[i] = []
                        marker_trails[i].append((f_num, x, y))

            # Draw trails for each marker
            for marker_id, positions in marker_trails.items():
                # Sort positions by frame number
                positions.sort(key=lambda p: p[0])

                # Need at least 2 points to draw a line
                if len(positions) < 2:
                    continue

                # Get marker's color - we'll base trail color on the marker's color
                color = get_color_for_id(marker_id)

                # Draw line segments with decreasing opacity
                for i in range(1, len(positions)):
                    # Calculate opacity based on how recent this segment is
                    segment_opacity = int(
                        200 * (positions[i][0] - start_frame) / (frame_count - start_frame + 1)
                    )

                    # Get screen coordinates for this segment
                    prev_frame, prev_x, prev_y = positions[i - 1]
                    curr_frame, curr_x, curr_y = positions[i]

                    # Skip if any coordinates are None
                    if prev_x is None or prev_y is None or curr_x is None or curr_y is None:
                        continue

                    prev_screen_x = int((prev_x * zoom_level) - crop_x)
                    prev_screen_y = int((prev_y * zoom_level) - crop_y)
                    curr_screen_x = int((curr_x * zoom_level) - crop_x)
                    curr_screen_y = int((curr_y * zoom_level) - crop_y)

                    # Create a temporary surface for the semi-transparent line
                    # Make it large enough to contain the line with padding
                    min_x = min(prev_screen_x, curr_screen_x) - 2
                    min_y = min(prev_screen_y, curr_screen_y) - 2
                    width = abs(curr_screen_x - prev_screen_x) + 4
                    height = abs(curr_screen_y - prev_screen_y) + 4

                    # Handle zero dimensions
                    width = max(width, 1)
                    height = max(height, 1)

                    line_surface = pygame.Surface((width, height), pygame.SRCALPHA)

                    # Line color with opacity
                    line_color = (color[0], color[1], color[2], segment_opacity)

                    # Draw the line segment on the surface
                    pygame.draw.line(
                        line_surface,
                        line_color,
                        (prev_screen_x - min_x, prev_screen_y - min_y),
                        (curr_screen_x - min_x, curr_screen_y - min_y),
                        max(3, int(6 * segment_opacity / 200)),  # Linha mais espessa
                    )

                    # Blit the line segment to the screen
                    screen.blit(line_surface, (min_x, min_y))

                    # Desenhar ponto no início do segmento (ponto adicional)
                    point_surface = pygame.Surface((8, 8), pygame.SRCALPHA)
                    point_color = (
                        0,
                        0,
                        0,
                        segment_opacity,
                    )  # Cor preta com a mesma opacidade da linha
                    pygame.draw.circle(point_surface, point_color, (4, 4), 2)  # Círculo de raio 2
                    screen.blit(point_surface, (prev_screen_x - 4, prev_screen_y - 4))

                # Still draw the most recent point as a small circle
                last_frame, last_x, last_y = positions[-1]
                if (
                    last_frame < frame_count and last_x is not None and last_y is not None
                ):  # Don't draw current frame again and check for None
                    last_screen_x = int((last_x * zoom_level) - crop_x)
                    last_screen_y = int((last_y * zoom_level) - crop_y)

                    # Draw a small circle for the most recent position
                    pygame.draw.circle(screen, color, (last_screen_x, last_screen_y), 2)

        # Draw Pose Connections if likely a full pose (33 points)
        if not one_line_mode and len(coordinates[frame_count]) == 33:
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < 33 and end_idx < 33:
                    # Check if indices exist (redundant but safe)
                    if start_idx < len(coordinates[frame_count]) and end_idx < len(
                        coordinates[frame_count]
                    ):
                        pt1 = coordinates[frame_count][start_idx]
                        pt2 = coordinates[frame_count][end_idx]

                        if pt1 is not None and pt2 is not None:
                            x1, y1 = pt1
                            x2, y2 = pt2

                            if (
                                x1 is not None
                                and y1 is not None
                                and x2 is not None
                                and y2 is not None
                            ):
                                sx1 = int((x1 * zoom_level) - crop_x)
                                sy1 = int((y1 * zoom_level) - crop_y)
                                sx2 = int((x2 * zoom_level) - crop_x)
                                sy2 = int((y2 * zoom_level) - crop_y)

                                # Draw simple blue line for skeleton
                                pygame.draw.line(screen, (0, 0, 255), (sx1, sy1), (sx2, sy2), 2)

        # Draw current frame markers

        if one_line_mode:
            for idx, (f_num, x, y) in enumerate(one_line_markers):
                if f_num == frame_count:
                    # Skip rendering if marker is empty/None
                    if x is None or y is None:
                        continue

                    screen_x = int((x * zoom_level) - crop_x)
                    screen_y = int((y * zoom_level) - crop_y)

                    # Highlight selected marker only in one_line_mode
                    if idx == selected_marker_idx:
                        pygame.draw.circle(
                            screen, (255, 165, 0), (screen_x, screen_y), 7
                        )  # Orange highlight

                    pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)
                    text_surface = font.render(str(idx + 1), True, (255, 255, 255))
                    screen.blit(text_surface, (screen_x + 5, screen_y - 15))
        else:
            for i, (x, y) in enumerate(coordinates[frame_count]):
                if i in deleted_positions[frame_count]:
                    continue

                if x is None or y is None:
                    continue

                screen_x = int((x * zoom_level) - crop_x)
                screen_y = int((y * zoom_level) - crop_y)

                # Adicionar destaque para o marcador selecionado
                if i == selected_marker_idx:
                    # Desenhar círculo laranja mais amplo como destaque
                    pygame.draw.circle(
                        screen, (255, 165, 0), (screen_x, screen_y), 7
                    )  # Orange highlight

                pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)
                text_surface = font.render(str(i + 1), True, (255, 255, 255))
                screen.blit(text_surface, (screen_x + 5, screen_y - 15))

        # Draw YOLO tracking bounding boxes
        if show_tracking and csv_loaded:
            # Check if current frame has tracking data
            if frame_count in tracking_data:
                boxes = tracking_data[frame_count]
                for box in boxes:
                    # Convert video coordinates to screen coordinates (account for zoom/offset)
                    screen_x1 = int((box["x1"] * zoom_level) - crop_x)
                    screen_y1 = int((box["y1"] * zoom_level) - crop_y)
                    screen_x2 = int((box["x2"] * zoom_level) - crop_x)
                    screen_y2 = int((box["y2"] * zoom_level) - crop_y)

                    # Only draw if box is visible on screen
                    if (
                        screen_x2 > 0
                        and screen_x1 < window_width
                        and screen_y2 > 0
                        and screen_y1 < window_height
                    ):
                        # Get color from box (RGB tuple)
                        box_color = box.get("color", (0, 255, 0))

                        # Draw rectangle outline with tracking color
                        pygame.draw.rect(
                            screen,
                            box_color,
                            (screen_x1, screen_y1, screen_x2 - screen_x1, screen_y2 - screen_y1),
                            2,
                        )

                        # Build complete label text: "Label:person id:1 conf:0.85"
                        label = box.get("label", "object")
                        tracker_id = box.get("id")
                        conf = box.get("conf", 0)

                        # Build text string
                        label_parts = [f"Label:{label}"]
                        if tracker_id is not None:
                            label_parts.append(f"id:{tracker_id}")
                        if conf > 0:
                            label_parts.append(f"conf:{conf:.2f}")

                        label_text = " ".join(label_parts)

                        # Draw complete label text above the box
                        if label_text:
                            # Create text surface with background for readability
                            text_surface = font.render(label_text, True, (255, 255, 255))
                            text_bg = pygame.Surface(
                                (text_surface.get_width() + 4, text_surface.get_height() + 2)
                            )
                            text_bg.fill(box_color)
                            text_bg.set_alpha(200)
                            # Position above the box
                            text_y = screen_y1 - text_surface.get_height() - 2
                            screen.blit(text_bg, (screen_x1, text_y))
                            screen.blit(text_surface, (screen_x1 + 2, text_y + 1))

        # Draw bounding boxes when in labeling mode
        if labeling_mode:
            # Draw existing boxes for current frame
            if frame_count in bboxes:
                for bbox in bboxes[frame_count]:
                    # Convert video coordinates to screen coordinates (account for zoom/offset)
                    screen_x = int((bbox["x"] * zoom_level) - crop_x)
                    screen_y = int((bbox["y"] * zoom_level) - crop_y)
                    screen_w = int(bbox["w"] * zoom_level)
                    screen_h = int(bbox["h"] * zoom_level)
                    # Draw rectangle outline (red color)
                    pygame.draw.rect(
                        screen, (255, 0, 0), (screen_x, screen_y, screen_w, screen_h), 2
                    )

            # Draw current box being drawn (preview)
            if drawing_box and current_box_rect is not None:
                pygame.draw.rect(screen, (255, 0, 0), current_box_rect, 2)

        (
            one_line_button_rect,
            save_button_rect,
            help_button_rect,
            persist_button_rect,
            load_button_rect,
            seq_button_rect,  # Add sequential button to return
            auto_button_rect,  # Add auto button to return
            click_pass_button_rect,  # Add ClickPass button to return
            labeling_button_rect,  # Add labeling button to return
            tracking_csv_button_rect,  # Add tracking CSV button to return
            show_tracking_indicator_rect,  # Add tracking indicator to return
            export_video_button_rect,  # Add export video button to return
            help_web_button_rect,  # Add help web button to return
            dataset_button_rect,  # Load dataset folder (multi-video)
            slider_x,
            slider_y,
            slider_width,
            slider_height,
        ) = draw_controls()

        # Show save message if needed
        if showing_save_message:
            save_message_timer -= 1
            if save_message_timer <= 0:
                showing_save_message = False
            else:
                # Draw save message notification at top of screen
                msg_font = pygame.font.SysFont("verdana", 14)
                msg_surface = msg_font.render(save_message_text, True, (255, 255, 255))
                msg_bg = pygame.Surface(
                    (msg_surface.get_width() + 20, msg_surface.get_height() + 10)
                )
                msg_bg.set_alpha(200)
                msg_bg.fill((0, 100, 0))
                screen.blit(msg_bg, (window_width // 2 - msg_bg.get_width() // 2, 10))
                screen.blit(msg_surface, (window_width // 2 - msg_surface.get_width() // 2, 15))

        pygame.display.flip()

        # Auto-marking logic - mark points automatically during playback
        if auto_marking_mode and not paused and not one_line_mode:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if mouse_y < window_height:  # Only mark if mouse is in video area
                # Convert screen coordinates to video coordinates
                video_x = (mouse_x + crop_x) / zoom_level
                video_y = (mouse_y + crop_y) / zoom_level

                # Mark the point at the selected marker index
                if selected_marker_idx >= 0:
                    # Update existing marker
                    while len(coordinates[frame_count]) <= selected_marker_idx:
                        coordinates[frame_count].append((None, None))
                    coordinates[frame_count][selected_marker_idx] = (video_x, video_y)
                    if selected_marker_idx in deleted_positions[frame_count]:
                        deleted_positions[frame_count].remove(selected_marker_idx)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                if new_h > 80:
                    window_width, window_height = new_w, new_h - 80
                    screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if labeling_mode and bboxes:
                        # Export labeling dataset
                        dataset_dir, message = export_labeling_dataset(
                            video_path, bboxes, total_frames, original_width, original_height
                        )
                        if dataset_dir:
                            saved = True
                            save_message_text = f"Dataset exported: {os.path.basename(dataset_dir)}"
                            showing_save_message = True
                            save_message_timer = 120
                        else:
                            save_message_text = message
                            showing_save_message = True
                            save_message_timer = 60
                    elif one_line_mode:
                        output_file = save_1_line_coordinates(
                            video_path, one_line_markers, deleted_markers
                        )
                        saved = True
                        save_message_text = f"Saved to: {os.path.basename(output_file)}"
                        showing_save_message = True
                        save_message_timer = 90  # Show for about 3 seconds at 30fps
                    else:
                        output_file = save_coordinates(
                            video_path,
                            coordinates,
                            total_frames,
                            deleted_positions,
                            is_sequential=sequential_mode,
                        )
                        # Save Swap Config if available
                        if active_swap_rules:
                            base_dir = os.path.dirname(video_path)
                            base_filename = os.path.splitext(os.path.basename(video_path))[0]
                            swap_file = os.path.join(base_dir, f"{base_filename}_swap.toml")
                            save_swap_toml(swap_file, active_swap_rules)

                        saved = True
                        save_message_text = f"Saved to: {os.path.basename(output_file)}"
                        showing_save_message = True
                        save_message_timer = 90  # Show for about 3 seconds at 30fps
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    frame_count = min(frame_count + 1, total_frames - 1)
                elif event.key == pygame.K_LEFT and paused:
                    frame_count = max(frame_count - 1, 0)
                elif event.key == pygame.K_UP and paused:
                    frame_count = min(frame_count + 60, total_frames - 1)
                elif event.key == pygame.K_DOWN and paused:
                    frame_count = max(frame_count - 60, 0)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    zoom_level *= 1.2
                elif event.key == pygame.K_MINUS:
                    zoom_level = max(0.2, zoom_level / 1.2)
                elif event.key == pygame.K_c:
                    one_line_mode = not one_line_mode
                    selected_marker_idx = -1  # Reset selected marker when changing modes
                elif event.key == pygame.K_m:
                    auto_marking_mode = not auto_marking_mode
                    save_message_text = (
                        f"Auto-marking {'enabled' if auto_marking_mode else 'disabled'}"
                    )
                    showing_save_message = True
                    save_message_timer = 30
                elif event.key == pygame.K_d:
                    # Delete all markers
                    if frame_count in coordinates:
                        deleted_positions[frame_count] = set(range(len(coordinates[frame_count])))
                        save_message_text = "All markers from this frame were deleted"
                        showing_save_message = True
                        save_message_timer = 30

                elif event.key == pygame.K_l:
                    labeling_mode = not labeling_mode
                    if labeling_mode:
                        # Disable other modes when labeling is active
                        one_line_mode = False
                        auto_marking_mode = False
                        sequential_mode = False
                        save_message_text = (
                            "LABELING MODE: Click and DRAG to draw boxes. Press Z to undo."
                        )
                    else:
                        save_message_text = "Labeling mode disabled"
                    showing_save_message = True
                    save_message_timer = 90
                elif event.key == pygame.K_z and labeling_mode:
                    # Undo last box in current frame
                    if frame_count in bboxes and bboxes[frame_count]:
                        bboxes[frame_count].pop()
                        save_message_text = "Removed last bounding box"
                        showing_save_message = True
                        save_message_timer = 30
                elif event.key == pygame.K_n and labeling_mode:
                    # Rename current label
                    new_label = show_input_dialog("Enter new label name:", current_label)
                    if new_label:
                        old_label = current_label
                        current_label = new_label

                        # Retroactively update existing boxes with the old label
                        updated_count = 0
                        for f_idx in bboxes:
                            for bbox in bboxes[f_idx]:
                                if bbox.get("label") == old_label:
                                    bbox["label"] = new_label
                                    updated_count += 1

                        save_message_text = (
                            f"Label changed: {old_label} -> {new_label} ({updated_count} updated)"
                        )
                        showing_save_message = True
                        save_message_timer = 60

                elif event.key == pygame.K_F5:
                    # Save Project
                    if labeling_mode:
                        save_labeling_project()
                    else:
                        save_message_text = "Enable Labeling Mode to save project"
                        showing_save_message = True
                        save_message_timer = 60
                elif event.key == pygame.K_F6:
                    # Load Project
                    if labeling_mode:
                        load_labeling_project()
                    else:
                        save_message_text = "Enable Labeling Mode to load project"
                        showing_save_message = True
                        save_message_timer = 60
                elif event.key == pygame.K_F7:
                    # Load dataset folder (next Save will append to it; multi-video)
                    if labeling_mode:
                        load_dataset_folder()
                    else:
                        save_message_text = "Enable Labeling Mode to set dataset folder"
                        showing_save_message = True
                        save_message_timer = 60
                elif event.key == pygame.K_F8:
                    # Open another video (keep dataset if set; no need to close app)
                    start_dir = (
                        os.path.dirname(video_path) if video_path else os.path.expanduser("~")
                    )
                    if sys.platform == "linux":
                        new_path = show_file_browser(
                            start_dir,
                            title="Select another video",
                            extensions=[".mp4", ".avi", ".mkv", ".mov", ".webm"],
                        )
                    else:
                        new_path = None
                        try:
                            from tkinter import Tk, filedialog

                            root = Tk()
                            root.withdraw()
                            root.attributes("-topmost", True)
                            new_path = filedialog.askopenfilename(
                                title="Select another video",
                                initialdir=start_dir,
                                filetypes=[
                                    ("Video", "*.mp4 *.avi *.mkv *.mov *.webm"),
                                    ("All files", "*.*"),
                                ],
                            )
                            root.destroy()
                        except Exception:
                            pass
                    if new_path and os.path.isfile(new_path) and new_path != video_path:
                        switch_to_video = new_path
                        running = False
                    elif new_path:
                        save_message_text = "Select a different video file."
                        showing_save_message = True
                        save_message_timer = 60
                elif event.key == pygame.K_TAB:
                    # Completely revamped marker navigation
                    if one_line_mode:
                        # Get all marker indices in current frame, including deleted ones
                        frame_marker_indices = []
                        deleted_frame_markers = []

                        # Collect available markers and the maximum index
                        max_marker_index = -1
                        for idx, (f_num, _, _) in enumerate(one_line_markers):
                            if idx > max_marker_index:
                                max_marker_index = idx

                            if f_num == frame_count:
                                if idx in deleted_markers:
                                    deleted_frame_markers.append(idx)
                                else:
                                    frame_marker_indices.append(idx)

                        # If no direction set (first tab press), start from 0 or current
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            # Previous marker
                            if selected_marker_idx == -1:
                                # If no selection, go to the last available marker
                                if frame_marker_indices:
                                    selected_marker_idx = frame_marker_indices[-1]
                                elif deleted_frame_markers:
                                    selected_marker_idx = deleted_frame_markers[-1]
                                elif max_marker_index >= 0:
                                    # If no marker in this frame, go to max index
                                    selected_marker_idx = max_marker_index
                            else:
                                # Find the previous marker (lower index)
                                prev_visible = [
                                    idx for idx in frame_marker_indices if idx < selected_marker_idx
                                ]
                                prev_deleted = [
                                    idx
                                    for idx in deleted_frame_markers
                                    if idx < selected_marker_idx
                                ]
                                prev_indices = prev_visible + prev_deleted

                                if prev_indices:
                                    selected_marker_idx = max(prev_indices)
                                else:
                                    # Wrap around to max marker
                                    all_markers = frame_marker_indices + deleted_frame_markers
                                    if all_markers:
                                        selected_marker_idx = max(all_markers)
                                    else:
                                        selected_marker_idx = max_marker_index
                        else:
                            # Next marker
                            if selected_marker_idx == -1:
                                # If no selection, start with the first marker
                                if frame_marker_indices:
                                    selected_marker_idx = min(frame_marker_indices)
                                elif deleted_frame_markers:
                                    selected_marker_idx = min(deleted_frame_markers)
                                elif max_marker_index >= 0:
                                    # If no marker in this frame, start from 0
                                    selected_marker_idx = 0
                            else:
                                # Find the next marker (higher index)
                                next_visible = [
                                    idx for idx in frame_marker_indices if idx > selected_marker_idx
                                ]
                                next_deleted = [
                                    idx
                                    for idx in deleted_frame_markers
                                    if idx > selected_marker_idx
                                ]
                                next_indices = next_visible + next_deleted

                                if next_indices:
                                    selected_marker_idx = min(next_indices)
                                else:
                                    # Wrap around to first marker
                                    all_markers = frame_marker_indices + deleted_frame_markers
                                    selected_marker_idx = min(all_markers) if all_markers else 0
                    else:
                        # Regular mode navigation
                        # Get indices of all markers in current frame
                        visible_markers = []
                        deleted_markers_in_frame = []

                        for i in range(len(coordinates[frame_count])):
                            if i in deleted_positions[frame_count]:
                                deleted_markers_in_frame.append(i)
                            else:
                                visible_markers.append(i)

                        # Get the max index that could be selected
                        max_index = max(
                            [len(coordinates[frame_count]) - 1]
                            + list(deleted_positions[frame_count])
                            + [-1]
                        )

                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            # Previous marker
                            if selected_marker_idx == -1:
                                # If no selection, go to the last available marker
                                if visible_markers:
                                    selected_marker_idx = max(visible_markers)
                                elif deleted_markers_in_frame:
                                    selected_marker_idx = max(deleted_markers_in_frame)
                                else:
                                    selected_marker_idx = max_index
                            else:
                                # Find the previous marker
                                prev_visible = [
                                    idx for idx in visible_markers if idx < selected_marker_idx
                                ]
                                prev_deleted = [
                                    idx
                                    for idx in deleted_markers_in_frame
                                    if idx < selected_marker_idx
                                ]
                                prev_indices = prev_visible + prev_deleted

                                if prev_indices:
                                    selected_marker_idx = max(prev_indices)
                                else:
                                    # Wrap around to highest marker
                                    all_markers = visible_markers + deleted_markers_in_frame
                                    if all_markers:
                                        selected_marker_idx = max(all_markers)
                                    else:
                                        selected_marker_idx = max_index
                        else:
                            # Next marker
                            if selected_marker_idx == -1:
                                # If no selection, start with the first marker
                                if visible_markers:
                                    selected_marker_idx = min(visible_markers)
                                elif deleted_markers_in_frame:
                                    selected_marker_idx = min(deleted_markers_in_frame)
                                else:
                                    selected_marker_idx = 0
                            else:
                                # Find the next marker
                                next_visible = [
                                    idx for idx in visible_markers if idx > selected_marker_idx
                                ]
                                next_deleted = [
                                    idx
                                    for idx in deleted_markers_in_frame
                                    if idx > selected_marker_idx
                                ]
                                next_indices = next_visible + next_deleted

                                if next_indices:
                                    selected_marker_idx = min(next_indices)
                                else:
                                    # Wrap around to first marker
                                    all_markers = visible_markers + deleted_markers_in_frame
                                    selected_marker_idx = min(all_markers) if all_markers else 0

                # Add persistence toggle with 'p' key
                elif event.key == pygame.K_p:
                    persistence_enabled = not persistence_enabled
                    # Show confirmation message
                    save_message_text = (
                        f"Persistence {'enabled' if persistence_enabled else 'disabled'}"
                    )
                    showing_save_message = True
                    save_message_timer = 30

                # Adjust persistence frames with '1', '2', and '3' keys
                elif event.key == pygame.K_1:  # Decrease persistence frames
                    if persistence_enabled:
                        persistence_frames = max(1, persistence_frames - 1)
                        save_message_text = f"Persistence: {persistence_frames} frames"
                        showing_save_message = True
                        save_message_timer = 30

                elif event.key == pygame.K_2:  # Increase persistence frames
                    if persistence_enabled:
                        persistence_frames += 1  # Sem limite máximo
                        save_message_text = f"Persistence: {persistence_frames} frames"
                        showing_save_message = True
                        save_message_timer = 30

                elif event.key == pygame.K_3:  # Alternar entre três modos
                    if not persistence_enabled:
                        # Modo 1: Ativar com persistência completa
                        persistence_enabled = True
                        persistence_frames = total_frames
                        save_message_text = "Full persistence enabled"
                    elif persistence_frames == total_frames:
                        # Modo 2: Mudar de full para número específico de frames
                        persistence_frames = 10
                        save_message_text = f"Persistence: {persistence_frames} frames"
                    else:
                        # Modo 3: Desativar completamente
                        persistence_enabled = False
                        save_message_text = "Persistence disabled"

                    showing_save_message = True
                    save_message_timer = 30

                # Show help dialog with 'h'
                elif event.key == pygame.K_h:
                    show_help_dialog()

                # Adicionar novo marcador
                elif event.key == pygame.K_a:
                    add_new_marker()

                # Remover marcador
                elif event.key == pygame.K_r or event.key == pygame.K_d:
                    remove_marker()

                # Playback Speed Control
                elif event.key == pygame.K_RIGHTBRACKET:  # ]
                    playback_speed *= 2.0
                    if playback_speed > 16.0:
                        playback_speed = 16.0
                    save_message_text = f"Speed: {playback_speed}X"
                    showing_save_message = True
                    save_message_text = f"Speed: {playback_speed}X"
                    showing_save_message = True
                    save_message_timer = 30

                # Add Pose detection hotkey 'J'
                elif event.key == pygame.K_j:
                    if MEDIAPIPE_AVAILABLE:
                        landmarks = detect_pose_mediapipe(frame)
                        if landmarks:
                            # Ensure coordinate list is large enough
                            while len(coordinates) <= frame_count:
                                coordinates.append([])

                            # Add points
                            for px, py in landmarks:
                                coordinates[frame_count].append((px, py))

                            save_message_text = f"Pose: Added {len(landmarks)} points"
                            showing_save_message = True
                            save_message_timer = 60
                        else:
                            save_message_text = "Pose: No person detected"
                            showing_save_message = True
                            save_message_timer = 60
                    else:
                        save_message_text = "MediaPipe not installed"
                        showing_save_message = True
                        save_message_timer = 60

                elif event.key == pygame.K_LEFTBRACKET:  # [
                    playback_speed /= 2.0
                    if playback_speed < 0.0625:
                        playback_speed = 0.0625
                    save_message_text = f"Speed: {playback_speed}X"
                    showing_save_message = True
                    save_message_timer = 30

                    save_message_timer = 30

                # Swap Hotkey (W) and Load Config (Shift+W)
                elif event.key == pygame.K_w:
                    # New Multi-Swap Manager
                    # Loop to allow saving/loading and returning to dialog
                    dialog_rules = active_swap_rules  # Start with current session rules

                    while True:
                        # Call Dialog
                        result = show_multi_swap_dialog(
                            frame_count, total_frames, dialog_rules, labels
                        )

                        if result is None:
                            # Cancelled
                            break

                        if isinstance(result, list):
                            # Apply All (result is the new list of rules)
                            new_rules = result
                            old_rules = active_swap_rules

                            # 1. Revert removed rules (Diff: Old - New)
                            # Swapping is its own inverse, so applying the same rule again undoes it.
                            for r in old_rules:
                                if r not in new_rules:
                                    print(f"Reverting swap: {r}")
                                    coordinates = apply_swap_config(coordinates, [r])

                            # 2. Apply new rules (Diff: New - Old)
                            for r in new_rules:
                                if r not in old_rules:
                                    print(f"Applying new swap: {r}")
                                    coordinates = apply_swap_config(coordinates, [r])

                            # Update session state
                            active_swap_rules = new_rules

                            save_message_text = (
                                f"Swap Rules Updated: {len(active_swap_rules)} active"
                            )
                            showing_save_message = True
                            save_message_timer = 60
                            break  # Close dialog after applying

                        elif isinstance(result, tuple):
                            action, r_data = result
                            if action == "SAVE":
                                # Input filename
                                default_name = (
                                    f"{os.path.basename(video_path).rsplit('.', 1)[0]}_swap.toml"
                                )
                                fname = show_input_dialog("Save Config As:", default_name)
                                if fname:
                                    # Ensure .toml
                                    if not fname.endswith(".toml"):
                                        fname += ".toml"
                                    # Path: use dirname of video
                                    save_path = os.path.join(os.path.dirname(video_path), fname)
                                    save_swap_toml(save_path, r_data)
                                    save_message_text = f"Saved {fname}"
                                    showing_save_message = True
                                    save_message_timer = 60
                                # Loop continues with same rules
                                dialog_rules = r_data

                            elif action == "LOAD":
                                # Select file
                                fpath = pygame_file_dialog(
                                    initial_dir=os.path.dirname(video_path),
                                    file_extensions=[".toml"],
                                    restore_size=(window_width, window_height),
                                )
                                # IMMEDIATE FIX: Restore Dialog Size because we are still in the loop!
                                pygame.display.set_mode((900, 600), pygame.RESIZABLE)

                                if fpath and os.path.exists(fpath):
                                    loaded = load_swap_toml(fpath)
                                    if loaded is not None:
                                        dialog_rules = loaded  # Replace internal dialog state
                                        save_message_text = f"Loaded {len(loaded)} rules"
                                        showing_save_message = True
                                        save_message_timer = 60
                                # Loop continues with loaded rules

                    # RESTORE MAIN WINDOW STATE
                    screen = pygame.display.set_mode(
                        (window_width, window_height + 80), pygame.RESIZABLE
                    )

                # Add sequential mode toggle with 'o' key
                elif (
                    event.key == pygame.K_o or event.key == pygame.K_s
                ):  # Toggle sequential mode with 'o' key
                    if not one_line_mode:  # Only toggle if not in one-line mode
                        sequential_mode = not sequential_mode
                        save_message_text = (
                            f"Sequential mode {'enabled' if sequential_mode else 'disabled'}"
                        )
                        showing_save_message = True
                        save_message_timer = 30

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y >= window_height:
                    # Clique na área de controles
                    rel_y = y - window_height
                    if load_button_rect.collidepoint(x, rel_y):
                        # Carregar novo arquivo
                        reload_coordinates()
                    elif one_line_button_rect.collidepoint(x, rel_y):
                        one_line_mode = not one_line_mode
                        selected_marker_idx = -1  # Reset selected marker
                    elif help_button_rect.collidepoint(x, rel_y):
                        show_help_dialog()
                    elif save_button_rect.collidepoint(x, rel_y):
                        if labeling_mode and bboxes:
                            # New Unified Save Logic
                            save_labeling_project()
                            showing_save_message = True
                            save_message_timer = 60
                        elif one_line_mode:
                            output_file = save_1_line_coordinates(
                                video_path, one_line_markers, deleted_markers
                            )
                            saved = True
                            save_message_text = f"Saved to: {os.path.basename(output_file)}"
                            showing_save_message = True
                            save_message_timer = 90  # Show for about 3 seconds at 30fps
                        else:
                            output_file = save_coordinates(
                                video_path,
                                coordinates,
                                total_frames,
                                deleted_positions,
                                is_sequential=sequential_mode,
                            )
                            saved = True
                            save_message_text = f"Saved to: {os.path.basename(output_file)}"
                            showing_save_message = True
                            save_message_timer = 90  # Show for about 3 seconds at 30fps
                    elif persist_button_rect.collidepoint(x, rel_y):
                        # Remove persistence settings dialog
                        persistence_enabled = not persistence_enabled
                        save_message_text = (
                            f"Persistence {'enabled' if persistence_enabled else 'disabled'}"
                        )
                        showing_save_message = True
                        save_message_timer = 30
                    elif seq_button_rect.collidepoint(x, rel_y):
                        if not one_line_mode:  # Only toggle if not in one-line mode
                            sequential_mode = not sequential_mode
                            save_message_text = (
                                f"Sequential mode {'enabled' if sequential_mode else 'disabled'}"
                            )
                            showing_save_message = True
                            save_message_timer = 30
                    elif auto_button_rect.collidepoint(x, rel_y):
                        if not one_line_mode:  # Only toggle if not in one-line mode
                            auto_marking_mode = not auto_marking_mode
                            save_message_text = (
                                f"Auto-marking {'enabled' if auto_marking_mode else 'disabled'}"
                            )
                            showing_save_message = True
                            save_message_timer = 30
                    elif click_pass_button_rect.collidepoint(x, rel_y):
                        if not one_line_mode:
                            click_pass_mode = not click_pass_mode
                            save_message_text = (
                                f"ClickPass {'enabled' if click_pass_mode else 'disabled'}"
                            )
                            showing_save_message = True
                            save_message_timer = 30
                    elif labeling_button_rect.collidepoint(x, rel_y):
                        labeling_mode = not labeling_mode
                        if labeling_mode:
                            # Disable other modes when labeling is active
                            one_line_mode = False
                            auto_marking_mode = False
                            sequential_mode = False
                            save_message_text = (
                                "LABELING MODE: Click and DRAG to draw boxes. Press Z to undo."
                            )
                        else:
                            save_message_text = "Labeling mode disabled"
                        showing_save_message = True
                        save_message_timer = 90
                    elif tracking_csv_button_rect.collidepoint(x, rel_y):
                        # Load tracking CSV
                        load_tracking_csv()
                    elif show_tracking_indicator_rect.collidepoint(x, rel_y):
                        # Toggle show tracking
                        show_tracking = not show_tracking
                        save_message_text = (
                            f"Tracking display {'enabled' if show_tracking else 'disabled'}"
                        )
                        showing_save_message = True
                        save_message_timer = 60
                        save_message_text = (
                            f"Tracking display {'enabled' if show_tracking else 'disabled'}"
                        )
                        showing_save_message = True
                        save_message_timer = 60
                    elif export_video_button_rect.collidepoint(x, rel_y):
                        # Export video with annotations
                        export_video_with_annotations()
                    elif dataset_button_rect.collidepoint(x, rel_y):
                        # Load dataset folder (next Save appends; multi-video). Same as F7.
                        load_dataset_folder()
                    elif help_web_button_rect.collidepoint(x, rel_y):
                        # Open documentation in browser
                        import webbrowser

                        help_url = "file://" + os.path.abspath(
                            os.path.join(os.path.dirname(__file__), "help", "getpixelvideo.html")
                        )
                        webbrowser.open(help_url)

                    elif slider_y <= rel_y <= slider_y + slider_height:
                        dragging_slider = True
                        rel_x = x - slider_x
                        rel_x = max(0, min(rel_x, slider_width))
                        denom_frames = (
                            total_frames
                            if total_frames and total_frames > 0
                            else max(1, frame_count + 1)
                        )
                        # Map slider position proportionally; clamp when total_frames unknown
                        frame_count = int((rel_x / slider_width) * denom_frames)
                        if total_frames and total_frames > 0:
                            frame_count = max(0, min(frame_count, total_frames - 1))
                        else:
                            frame_count = max(0, frame_count)
                        paused = True
                else:
                    # Clique na área do vídeo
                    video_x = (x + crop_x) / zoom_level
                    video_y = (y + crop_y) / zoom_level

                    if event.button == 1:  # Left click
                        if labeling_mode:
                            # Start drawing bounding box
                            drawing_box = True
                            box_start_pos = (video_x, video_y)
                            current_box_rect = None
                        elif one_line_mode:
                            # Simply append the new marker
                            one_line_markers.append((frame_count, video_x, video_y))
                        else:
                            if sequential_mode:
                                # Find the next available marker index
                                next_idx = len(coordinates[frame_count])
                                coordinates[frame_count].append((video_x, video_y))
                                selected_marker_idx = next_idx  # Auto-select the new marker
                            else:
                                # Use existing marker selection logic
                                if selected_marker_idx >= 0:
                                    # Update existing marker
                                    while len(coordinates[frame_count]) <= selected_marker_idx:
                                        coordinates[frame_count].append((None, None))
                                    coordinates[frame_count][selected_marker_idx] = (
                                        video_x,
                                        video_y,
                                    )
                                    if selected_marker_idx in deleted_positions[frame_count]:
                                        deleted_positions[frame_count].remove(selected_marker_idx)
                                else:
                                    # Add new marker at the end
                                    coordinates[frame_count].append((video_x, video_y))
                                    selected_marker_idx = len(coordinates[frame_count]) - 1

                                # Feature: ClickPass logic
                                if click_pass_mode:
                                    # Visual Feedback: Redraw frame with new marker before advancing
                                    # Force a single draw cycle
                                    # Note: This is a bit hacky but gives immediate feedback

                                    # We need to re-render everything to show the marker
                                    # Since we are inside the loop, we can't easily jump to the drawing code
                                    # So we just flip display? No, we haven't drawn the new marker yet.
                                    # The main loop draws at the start.

                                    # Actually, we just added the coordinate to `coordinates`.
                                    # If we wait here, the user sees the OLD frame.

                                    # To show the NEW marker, we'd need to draw it.
                                    # Let's do a quick draw of just the marker or everything?
                                    # Everything is safer.

                                    # Simplified draw for feedback (copy-pasting drawing logic is bad)
                                    # Let's just draw a circle on the screen directly on top

                                    # Convert video coord to screen coord
                                    screen_cx = int(video_x * zoom_level - crop_x)
                                    screen_cy = int(video_y * zoom_level - crop_y)

                                    # Draw green circle
                                    pygame.draw.circle(
                                        screen, (0, 255, 0), (screen_cx, screen_cy), 5
                                    )
                                    pygame.display.flip()

                                    pygame.time.delay(200)  # Wait 200ms

                                    frame_count = min(frame_count + 1, total_frames - 1)
                                    paused = True

                    elif event.button == 3:  # Right click
                        # Keep existing behavior for right-click (delete most recent)
                        if one_line_mode:
                            for i in range(len(one_line_markers) - 1, -1, -1):
                                if one_line_markers[i][0] == frame_count:
                                    del one_line_markers[i]
                                    break
                        else:
                            if coordinates[frame_count]:
                                coordinates[frame_count].pop()
                        # Reset selection if we removed the selected marker
                        if one_line_mode:
                            markers_in_frame = [
                                i for i, m in enumerate(one_line_markers) if m[0] == frame_count
                            ]
                            if not markers_in_frame:
                                selected_marker_idx = -1
                            elif selected_marker_idx >= len(markers_in_frame):
                                selected_marker_idx = len(markers_in_frame) - 1
                        else:
                            if not coordinates[frame_count]:
                                selected_marker_idx = -1
                            elif selected_marker_idx >= len(coordinates[frame_count]):
                                selected_marker_idx = len(coordinates[frame_count]) - 1

                    elif event.button == 2:  # Middle click for panning
                        scrolling = True
                        pygame.mouse.get_rel()  # Zera o acumulador de movimento relativo para pan contínuo

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    scrolling = False
                elif event.button == 1:
                    dragging_slider = False
                    # Finalize bounding box if drawing
                    if labeling_mode and drawing_box and box_start_pos is not None:
                        x, y = event.pos
                        if y < window_height:  # Only if released in video area
                            video_x = (x + crop_x) / zoom_level
                            video_y = (y + crop_y) / zoom_level

                            # Calculate box coordinates
                            x1, y1 = box_start_pos
                            x2, y2 = video_x, video_y

                            # Normalize coordinates (handle drag direction)
                            box_x = min(x1, x2)
                            box_y = min(y1, y2)
                            box_w = abs(x2 - x1)
                            box_h = abs(y2 - y1)

                            # Validate box (minimum size threshold)
                            if box_w >= 5 and box_h >= 5:
                                # Ensure coordinates are within video bounds
                                box_x = max(0, min(box_x, original_width - 1))
                                box_y = max(0, min(box_y, original_height - 1))
                                box_w = min(box_w, original_width - box_x)
                                box_h = min(box_h, original_height - box_y)

                                # Initialize frame list if needed
                                if frame_count not in bboxes:
                                    bboxes[frame_count] = []

                                # Add box to current frame
                                bboxes[frame_count].append(
                                    {
                                        "x": box_x,
                                        "y": box_y,
                                        "w": box_w,
                                        "h": box_h,
                                        "label": current_label,
                                    }
                                )

                        # Reset drawing state
                        drawing_box = False
                        box_start_pos = None
                        current_box_rect = None

            elif event.type == pygame.MOUSEMOTION:
                if scrolling:
                    rel_dx, rel_dy = pygame.mouse.get_rel()
                    offset_x = max(0, min(zoomed_width - window_width, offset_x - rel_dx))
                    offset_y = max(0, min(zoomed_height - window_height, offset_y - rel_dy))
                elif labeling_mode and drawing_box and box_start_pos is not None:
                    # Update preview box while dragging
                    x, y = event.pos
                    if y < window_height:  # Only if mouse is in video area
                        video_x = (x + crop_x) / zoom_level
                        video_y = (y + crop_y) / zoom_level

                        # Calculate box coordinates
                        x1, y1 = box_start_pos
                        x2, y2 = video_x, video_y

                        # Normalize coordinates
                        box_x = min(x1, x2)
                        box_y = min(y1, y2)
                        box_w = abs(x2 - x1)
                        box_h = abs(y2 - y1)

                        # Convert to screen coordinates for preview
                        screen_x = int((box_x * zoom_level) - crop_x)
                        screen_y = int((box_y * zoom_level) - crop_y)
                        screen_w = int(box_w * zoom_level)
                        screen_h = int(box_h * zoom_level)

                        # Create preview rect
                        current_box_rect = pygame.Rect(screen_x, screen_y, screen_w, screen_h)

                if dragging_slider:
                    rel_x = event.pos[0] - slider_x
                    rel_x = max(0, min(rel_x, slider_width))
                    denom_frames = (
                        total_frames
                        if total_frames and total_frames > 0
                        else max(1, frame_count + 1)
                    )
                    frame_count = int((rel_x / slider_width) * denom_frames)
                    if total_frames and total_frames > 0:
                        frame_count = max(0, min(frame_count, total_frames - 1))
                    else:
                        frame_count = max(0, frame_count)
                    paused = True

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom on Scroll
                # event.y > 0 means scroll up (zoom in)
                # event.y < 0 means scroll down (zoom out)

                old_zoom = zoom_level
                zoom_factor = 1.1

                if event.y > 0:
                    zoom_level = min(10.0, zoom_level * zoom_factor)
                elif event.y < 0:
                    zoom_level = max(0.1, zoom_level / zoom_factor)

                # Adjust offsets to keep view centered on mouse if possible
                mx, my = pygame.mouse.get_pos()
                if my < window_height:  # Only if in video area
                    # Center zoom on mouse cursor
                    target_vx = (mx + offset_x) / old_zoom
                    target_vy = (my + offset_y) / old_zoom

                    offset_x = (target_vx * zoom_level) - mx
                    offset_y = (target_vy * zoom_level) - my

                    # Clamp offsets
                    zoomed_width = original_width * zoom_level
                    zoomed_height = original_height * zoom_level

                    if zoomed_width > window_width:
                        offset_x = max(0, min(zoomed_width - window_width, offset_x))
                    else:
                        offset_x = 0

                    if zoomed_height > window_height:
                        offset_y = max(0, min(zoomed_height - window_height, offset_y))
                    else:
                        offset_y = 0

                save_message_text = f"Zoom: {zoom_level:.2f}X"
                showing_save_message = True
                save_message_timer = 30

        if paused:
            # Se pausado, não limitamos a taxa de FPS para que a interface seja responsiva
            clock.tick(60)  # Taxa de atualização da interface
        else:
            # Se em reprodução, limitamos à taxa de FPS do vídeo
            clock.tick(fps)

    if switch_to_video:
        cap.release()
        # Do not pygame.quit() so run_getpixelvideo can reopen with new video
        return ("switch_video", switch_to_video, current_dataset_dir, labeling_mode)

    cap.release()
    pygame.quit()

    if saved:
        print("Coordinates were saved.")
    else:
        print("Coordinates were not saved.")


def is_yolo_dataset(path):
    """
    Check if a path is a YOLO dataset directory (AnyLabeling format).

    Expected structure:
    - images/ (or train/images/, val/images/, test/images/)
    - labels/ (or train/labels/, val/labels/, test/labels/)
    - classes.txt (optional)

    Args:
        path: Path to check

    Returns:
        tuple: (is_yolo_dataset, images_dir, labels_dir, classes_file) or (False, None, None, None)
    """
    if not os.path.isdir(path):
        return False, None, None, None

    # Check for standard YOLO structure
    images_dir = None
    labels_dir = None
    classes_file = None

    # Check for root-level images/labels
    if os.path.isdir(os.path.join(path, "images")) and os.path.isdir(os.path.join(path, "labels")):
        images_dir = os.path.join(path, "images")
        labels_dir = os.path.join(path, "labels")
    # Check for train/val/test structure
    elif os.path.isdir(os.path.join(path, "train", "images")) and os.path.isdir(
        os.path.join(path, "train", "labels")
    ):
        # Use train set by default
        images_dir = os.path.join(path, "train", "images")
        labels_dir = os.path.join(path, "train", "labels")

    if images_dir and labels_dir:
        # Check for classes.txt
        classes_file = os.path.join(path, "classes.txt")
        if not os.path.exists(classes_file):
            classes_file = None
        return True, images_dir, labels_dir, classes_file

    return False, None, None, None


def load_yolo_dataset(dataset_path, video_path, total_frames, video_width, video_height):
    """
    Load YOLO dataset labels and convert bounding boxes to point coordinates.

    YOLO format: class_id center_x center_y width height (all normalized 0-1)
    Converts to: center points of bounding boxes as markers

    Args:
        dataset_path: Path to YOLO dataset directory
        video_path: Path to video file (to match frame names)
        total_frames: Total number of frames in video
        video_width: Video width in pixels
        video_height: Video height in pixels

    Returns:
        dict: Coordinates dictionary {frame_index: [(x, y), ...]}
    """
    is_yolo, images_dir, labels_dir, classes_file = is_yolo_dataset(dataset_path)

    if not is_yolo:
        return None

    print("Detected YOLO dataset structure:")
    print(f"  Images directory: {images_dir}")
    print(f"  Labels directory: {labels_dir}")
    if classes_file:
        print(f"  Classes file: {classes_file}")

    # Load class names if available
    class_names = []
    if classes_file and os.path.exists(classes_file):
        try:
            with open(classes_file, encoding="utf-8") as f:
                class_names = [line.strip() for line in f if line.strip()]
            print(f"  Found {len(class_names)} classes: {class_names}")
        except Exception as e:
            print(f"  Warning: Could not read classes.txt: {e}")

    # Get video base name to match with label files
    os.path.splitext(os.path.basename(video_path))[0]

    # Initialize coordinates dictionary
    coordinates = {i: [] for i in range(total_frames)}

    # Get list of label files
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    if not label_files:
        print(f"  Warning: No .txt label files found in {labels_dir}")
        return coordinates

    print(f"  Found {len(label_files)} label files")

    # Process each label file
    matched_frames = 0
    for label_file in label_files:
        label_path = os.path.join(labels_dir, label_file)

        # Try to match label file with video frame
        # Label files might be named: frame_000001.txt, image001.txt, etc.
        frame_num = None

        # Try to extract frame number from filename
        # Remove extension and try different patterns
        base_name = os.path.splitext(label_file)[0]

        # Pattern 1: frame_XXXXXX or frame_XXX
        if "frame_" in base_name.lower():
            try:
                frame_str = base_name.lower().split("frame_")[-1]
                frame_num = int(frame_str)
            except ValueError:
                pass

        # Pattern 2: imageXXX or imgXXX
        if frame_num is None:
            numbers = re.findall(r"\d+", base_name)
            if numbers:
                try:
                    # Use the last number found (usually the frame number)
                    frame_num = int(numbers[-1])
                except ValueError:
                    pass

        # Pattern 3: Just a number
        if frame_num is None:
            with suppress(ValueError):
                frame_num = int(base_name)

        # If still no match, try to match by image filename
        if frame_num is None:
            # Check if corresponding image exists
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            for ext in image_extensions:
                image_file = base_name + ext
                image_path = os.path.join(images_dir, image_file)
                if os.path.exists(image_path):
                    # Try to extract frame number from image filename
                    img_base = os.path.splitext(image_file)[0]
                    numbers = re.findall(r"\d+", img_base)
                    if numbers:
                        try:
                            frame_num = int(numbers[-1])
                            break
                        except ValueError:
                            pass

        if frame_num is None or frame_num < 0 or frame_num >= total_frames:
            # Skip if we can't determine frame number or it's out of range
            continue

        # Read YOLO label file
        try:
            with open(label_path, encoding="utf-8") as f:
                lines = f.readlines()

            frame_points = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    int(parts[0])
                    center_x_norm = float(parts[1])  # Normalized 0-1
                    center_y_norm = float(parts[2])  # Normalized 0-1
                    float(parts[3])  # Normalized 0-1
                    float(parts[4])  # Normalized 0-1

                    # Convert normalized coordinates to pixel coordinates
                    center_x = center_x_norm * video_width
                    center_y = center_y_norm * video_height

                    # Add center point as marker
                    frame_points.append((center_x, center_y))

                except (ValueError, IndexError) as e:
                    print(f"  Warning: Skipping invalid line in {label_file}: {line} ({e})")
                    continue

            if frame_points:
                coordinates[frame_num] = frame_points
                matched_frames += 1

        except Exception as e:
            print(f"  Warning: Error reading {label_file}: {e}")
            continue

    print(f"  Successfully loaded {matched_frames} frames with annotations")
    total_annotations = sum(len(coords) for coords in coordinates.values())
    print(f"  Total bounding boxes converted: {total_annotations}")

    return coordinates


def load_coordinates_from_file(total_frames, video_width=None, video_height=None):
    # Use CLI input for file path to avoid Tkinter issues on Linux
    # This is also more robust for remote execution or when GUI is busy

    print("\n--- Select Keypoint File or YOLO Dataset ---")
    print("Enter the full path to:")
    print("  - A .csv file with keypoints, OR")
    print("  - A YOLO dataset directory (with images/ and labels/ folders)")
    print("You can drag and drop the file/folder into this terminal.")

    # Try to offer a default if we are in a likely directory
    default_hint = "(Press Enter to skip/cancel)"

    input_path = input(f"File/Folder path {default_hint}: ").strip()

    # Remove quotes if user dragged and dropped 'filename'
    if (
        input_path.startswith("'")
        and input_path.endswith("'")
        or input_path.startswith('"')
        and input_path.endswith('"')
    ):
        input_path = input_path[1:-1]

    input_path = input_path.strip()

    if not input_path:
        # User cancelled or empty input
        input_path = None

    if not input_path:
        print("No file/folder selected. Starting fresh.")
    if not input_path:
        print("No file/folder selected. Starting fresh.")
        return {i: [] for i in range(total_frames)}, []

    # Check if it's a YOLO dataset directory
    is_yolo, images_dir, labels_dir, classes_file = is_yolo_dataset(input_path)
    if is_yolo:
        print(f"\nDetected YOLO dataset directory: {input_path}")
        # Note: We need video_path, video_width, video_height for YOLO loading
        # These will be passed from the calling function
        # For now, return a special marker that indicates YOLO dataset
        # For now, return a special marker that indicates YOLO dataset
        # YOLO doesn't have consistent keypoint labels in the same way, usually just class IDs
        # We can map class IDs if class file exists
        labels = []
        if classes_file and os.path.exists(classes_file):
            try:
                with open(classes_file) as f:
                    labels = [l.strip() for l in f.readlines()]
            except:
                pass
        return {"_yolo_dataset": input_path}, labels

    # Otherwise, treat as CSV file
    input_file = input_path

    try:
        print(f"Attempting to load coordinates from: {input_file}")
        df = pd.read_csv(input_file)
        print("File loaded successfully!")
        print(f"File columns: {list(df.columns)}")
        print(f"DataFrame shape: {df.shape}")

        # Validate that we have data
        if df.empty:
            print("WARNING: File is empty!")
            return {i: [] for i in range(total_frames)}

    except pd.errors.EmptyDataError:
        print(f"ERROR: File {input_file} is empty or contains no data")
        return {i: [] for i in range(total_frames)}
    except pd.errors.ParserError as e:
        print(f"ERROR: Failed to parse CSV file {input_file}: {e}")
        print("This might be due to malformed CSV data or wrong file format")
        return {i: [] for i in range(total_frames)}
    except FileNotFoundError:
        print(f"ERROR: File not found: {input_file}")
        return {i: [] for i in range(total_frames)}
    except PermissionError:
        print(f"ERROR: Permission denied accessing file: {input_file}")
        return {i: [] for i in range(total_frames)}
    except Exception as e:
        print(f"ERROR: Unexpected error reading file {input_file}: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
    except Exception as e:
        print(f"ERROR: Unexpected error reading file {input_file}: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return {i: [] for i in range(total_frames)}, []

    # Case A: vailá format (frame + pN_x/pN_y) - from markerless_2d_analysis.py
    if "frame" in df.columns and any(
        col.startswith("p") and col.endswith("_x") for col in df.columns
    ):
        print("Detected format: vailá (pN_x, pN_y)")
        coordinates = {i: [] for i in range(total_frames)}

        # Find the maximum marker number
        max_marker = 0
        for col in df.columns:
            if col.startswith("p") and col.endswith("_x"):
                try:
                    marker_num = int(col[1:-2])  # Extract number from "pN_x"
                    max_marker = max(max_marker, marker_num)
                except ValueError:
                    continue

        print(f"Found {max_marker} markers in vailá format")

        for row_idx, row in df.iterrows():
            try:
                frame_num = int(row.get("frame", 0)) if pd.notna(row.get("frame")) else 0
                pts = []
                for i in range(1, max_marker + 1):
                    try:
                        x_val = row.get(f"p{i}_x")
                        y_val = row.get(f"p{i}_y")
                        if pd.notna(x_val) and pd.notna(y_val):
                            pts.append((float(x_val), float(y_val)))
                        else:
                            pts.append((None, None))
                    except Exception as e:
                        print(f"ERROR processing marker p{i} in row {row_idx}: {e}")
                        pts.append((None, None))

                # Remove trailing None values
                while pts and (pts[-1][0] is None or pts[-1][1] is None):
                    pts.pop()

                coordinates[frame_num] = pts

            except Exception as e:
                print(f"ERROR processing row {row_idx} in vailá format: {e}")
                print(f"Row data: {dict(row)}")
                # Add empty coordinates for this frame
                coordinates[row_idx] = []

        print(f"Coordinates successfully loaded (vailá format): {max_marker} markers")
        print(f"Coordinates successfully loaded (vailá format): {max_marker} markers")
        labels = [f"Pixel {i}" for i in range(1, max_marker + 1)]
        return coordinates, labels

    # Case B: MediaPipe format (frame_index + landmark_x/y/z) - from markerless_2d_analysis.py
    if "frame_index" in df.columns and any(col.endswith("_x") for col in df.columns):
        print("Detected format: MediaPipe (landmark_x, landmark_y, landmark_z)")

        # Get landmark base names (e.g., "nose", "left_eye", etc.)
        base_names = []
        for col in df.columns:
            if col.endswith("_x") and col != "frame_index":
                base_name = col[:-2]  # Remove "_x"
                if f"{base_name}_y" in df.columns:
                    base_names.append(base_name)

        base_names = sorted(base_names)
        print(f"Found {len(base_names)} landmarks: {base_names[:5]}...")  # Show first 5

        # Detect if coordinates are normalized (0-1) or pixel values
        def _is_normalized(sample_cols):
            try:
                sample_vals = pd.concat([df[c].dropna().head(200) for c in sample_cols])
                max_val = sample_vals.max()
                min_val = sample_vals.min()
                # Normalized coordinates typically range from 0-1, pixel coordinates are much larger
                is_norm = (max_val <= 1.2) and (min_val >= -0.2)
                print(
                    f"Coordinate range: {min_val:.3f} to {max_val:.3f} - {'Normalized' if is_norm else 'Pixel'}"
                )
                return is_norm
            except Exception as e:
                print(f"ERROR detecting normalization: {e}")
                print(f"Error type: {type(e).__name__}")
                import traceback

                traceback.print_exc()
                return False

        # Check a sample of columns to determine if normalized
        sample_cols = [f"{base}_x" for base in base_names[: min(5, len(base_names))]]
        is_norm = _is_normalized(sample_cols)

        # Determine file type based on filename
        filename_lower = input_file.lower()
        if "_norm.csv" in filename_lower:
            file_type = "normalized"
            print("Detected _norm.csv file - will convert normalized coordinates to pixel")
            # Force conversion for normalized files
            sx = video_width if video_width else 1.0
            sy = video_height if video_height else 1.0
            print(
                f"Converting normalized coordinates to pixel coordinates using video dimensions: {video_width}x{video_height}"
            )
        elif "_pixel.csv" in filename_lower:
            file_type = "pixel"
            print("Detected _pixel.csv file - using pixel coordinates directly (ignoring Z)")
            # Use pixel coordinates as-is
            sx = 1.0
            sy = 1.0
            print("Using coordinates as-is (pixel coordinates, no scaling)")
        else:
            file_type = "auto"
            print("Auto-detecting coordinate type")
            # Use auto-detection
            sx = video_width if (is_norm and video_width) else 1.0
            sy = video_height if (is_norm and video_height) else 1.0
            if is_norm and video_width and video_height:
                print(
                    f"Converting normalized coordinates to pixel coordinates using video dimensions: {video_width}x{video_height}"
                )
            else:
                print("Using coordinates as-is (no scaling)")

        coordinates = {i: [] for i in range(total_frames)}

        for row_idx, row in df.iterrows():
            try:
                frame_num = int(row.get("frame_index", 0))
                pts = []

                for base in base_names:
                    try:
                        x_val = row.get(f"{base}_x")
                        y_val = row.get(f"{base}_y")
                        # Note: We ignore the Z coordinate (f"{base}_z") as requested

                        if pd.notna(x_val) and pd.notna(y_val):
                            # Apply scaling if coordinates are normalized
                            x_coord = float(x_val) * sx
                            y_coord = float(y_val) * sy
                            pts.append((x_coord, y_coord))
                        else:
                            pts.append((None, None))
                    except Exception as e:
                        print(f"ERROR processing landmark {base} in row {row_idx}: {e}")
                        pts.append((None, None))

                # Remove trailing None values
                while pts and (pts[-1][0] is None or pts[-1][1] is None):
                    pts.pop()

                coordinates[frame_num] = pts

            except Exception as e:
                print(f"ERROR processing row {row_idx}: {e}")
                print(f"Row data: {dict(row)}")
                # Add empty coordinates for this frame
                coordinates[row_idx] = []

        print(
            f"Coordinates successfully loaded (MediaPipe {file_type} format): {len(base_names)} landmarks"
        )
        return coordinates, base_names

    # Case C: Legacy format or other CSV formats
    print(f"Unknown format detected. Columns: {list(df.columns)}")
    print("Attempting to load as generic CSV format...")

    # Try to find coordinate columns
    coord_cols = []
    for col in df.columns:
        if any(suffix in col.lower() for suffix in ["_x", "_y", "x", "y"]):
            coord_cols.append(col)

    if coord_cols:
        coordinates = {i: [] for i in range(total_frames)}

        # Try to determine frame column
        frame_col = None
        for col in df.columns:
            if "frame" in col.lower() or col == "0":
                frame_col = col
                break

        if frame_col is None:
            frame_col = df.columns[0]  # Use first column as frame

        print(f"Using '{frame_col}' as frame column")

        for _, row in df.iterrows():
            frame_num = int(row.get(frame_col, 0)) if pd.notna(row.get(frame_col)) else 0
            pts = []

            # Group x,y pairs
            i = 0
            while i < len(coord_cols) - 1:
                x_col = coord_cols[i]
                y_col = coord_cols[i + 1]

                x_val = row.get(x_col)
                y_val = row.get(y_col)

                if pd.notna(x_val) and pd.notna(y_val):
                    pts.append((float(x_val), float(y_val)))
                else:
                    pts.append((None, None))

                i += 2

            # Remove trailing None values
            while pts and (pts[-1][0] is None or pts[-1][1] is None):
                pts.pop()

            coordinates[frame_num] = pts

        print(f"Coordinates loaded (generic CSV format): {len(coord_cols) // 2} coordinate pairs")
        print(f"Coordinates loaded (generic CSV format): {len(coord_cols) // 2} coordinate pairs")
        labels = [f"Pixel {i + 1}" for i in range(len(coord_cols) // 2)]
        return coordinates, labels

    print(f"File format not recognized: {input_file}. Starting fresh.")
    print("Supported formats:")
    print("  1. vailá format: 'frame', 'p1_x', 'p1_y', 'p2_x', 'p2_y', ...")
    print("  2. MediaPipe format: 'frame_index', 'landmark_x', 'landmark_y', 'landmark_z', ...")
    print("  3. Generic CSV with coordinate columns")
    print("  4. YOLO dataset directory (with images/ and labels/ folders)")
    print("  4. YOLO dataset directory (with images/ and labels/ folders)")
    return {i: [] for i in range(total_frames)}, []


def save_coordinates(
    video_path, coordinates, total_frames, deleted_positions=None, is_sequential=False
):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)

    # Create different filenames based on the mode
    if is_sequential:
        output_file = os.path.join(video_dir, f"{base_name}_markers_sequential.csv")
    else:
        output_file = os.path.join(video_dir, f"{base_name}_markers.csv")

    # Initialize deleted_positions if not provided
    if deleted_positions is None:
        deleted_positions = {i: set() for i in range(total_frames)}

    # Determina o número máximo de pontos marcados em qualquer frame.
    max_points = max((len(points) for points in coordinates.values()), default=0)

    # Cria o cabeçalho: a primeira coluna é 'frame' e para cada ponto,
    # adiciona as colunas 'p{i}_x' e 'p{i}_y'
    columns = ["frame"]
    for i in range(1, max_points + 1):
        columns.append(f"p{i}_x")
        columns.append(f"p{i}_y")

    # Cria o DataFrame inicializado com NaN para todos os frames.
    df = pd.DataFrame(np.nan, index=range(total_frames), columns=columns)
    df["frame"] = df.index

    # Preenche o DataFrame com os pontos marcados
    for frame_num, points in coordinates.items():
        for i, (x, y) in enumerate(points):
            if i not in deleted_positions[frame_num]:  # Only save non-deleted markers
                # Verificar se é um marcador vazio (None)
                if x is not None and y is not None:
                    df.at[frame_num, f"p{i + 1}_x"] = float(x)
                    df.at[frame_num, f"p{i + 1}_y"] = float(y)
                # Se for None, deixar como NaN (o que se tornará "" no CSV)

    # Salva o CSV com valores NaN representados como strings vazias
    df.to_csv(output_file, index=False, na_rep="")
    print(f"Coordinates saved to: {output_file}")
    return output_file


def export_labeling_dataset(
    video_path,
    bboxes,
    total_frames,
    original_width,
    original_height,
    output_dataset_dir=None,
):
    """
    Export bounding boxes to structured dataset format.
    Creates train/val/test split with images, JSON annotations, and YOLO .txt labels
    plus classes.txt and data.yaml so the dataset works with yolotrain.py / Ultralytics.

    If output_dataset_dir is set, appends to that dataset (multi-video): uses video
    base name as prefix for filenames and merges class names with existing classes.txt.
    """
    import json
    import random

    # Collect all annotated frames
    annotated_frames = [f for f in bboxes if bboxes[f]]
    if not annotated_frames:
        return None, "No bounding boxes to export"

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    is_append = output_dataset_dir is not None and os.path.isdir(output_dataset_dir)

    if is_append:
        dataset_dir = os.path.abspath(output_dataset_dir)
        # Load existing classes and merge with new labels
        existing_classes = []
        classes_file = os.path.join(dataset_dir, "classes.txt")
        if os.path.exists(classes_file):
            try:
                with open(classes_file, encoding="utf-8") as f:
                    existing_classes = [line.strip() for line in f if line.strip()]
            except Exception:
                pass
        new_labels = set()
        for frame_num in annotated_frames:
            for bbox in bboxes[frame_num]:
                new_labels.add(bbox.get("label", "object"))
        class_names = sorted(set(existing_classes) | new_labels)
        # Prefix for filenames to avoid collisions across videos
        file_prefix = f"{base_name}_"
    else:
        # Single shared dataset dir per export: dataset_<timestamp> (reuse by loading it)
        dataset_dir = os.path.join(video_dir, f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        all_labels = set()
        for frame_num in annotated_frames:
            for bbox in bboxes[frame_num]:
                all_labels.add(bbox.get("label", "object"))
        class_names = sorted(all_labels)
        file_prefix = ""

    label_to_id = {name: i for i, name in enumerate(class_names)}

    # Create split indices (70/20/10 for this export only)
    random.shuffle(annotated_frames)
    n_total = len(annotated_frames)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    splits = {
        "train": annotated_frames[:n_train],
        "val": annotated_frames[n_train : n_train + n_val],
        "test": annotated_frames[n_train + n_val :],
    }

    # Create directory structure
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dataset_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split, "labels"), exist_ok=True)

    # Write or update classes.txt
    classes_file = os.path.join(dataset_dir, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")

    # Open video to extract frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video"

    # Process each split
    for split_name, frames in splits.items():
        for frame_num in frames:
            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue

            # Save image (with optional prefix for multi-video append)
            img_filename = f"{file_prefix}frame_{frame_num:06d}.jpg"
            img_path = os.path.join(dataset_dir, split_name, "images", img_filename)
            cv2.imwrite(img_path, frame)

            # Create JSON annotation (kept for compatibility)
            annotation = {
                "image": img_filename,
                "width": original_width,
                "height": original_height,
                "annotations": [],
            }

            # YOLO .txt lines: class_id x_center y_center width height (normalized 0-1)
            yolo_lines = []
            for bbox in bboxes[frame_num]:
                x = int(bbox["x"])
                y = int(bbox["y"])
                w = int(bbox["w"])
                h = int(bbox["h"])
                annotation["annotations"].append(
                    {"x": x, "y": y, "w": w, "h": h, "label": bbox.get("label", "object")}
                )
                # Normalize: center and size relative to image dimensions
                x_center = (x + w / 2.0) / original_width
                y_center = (y + h / 2.0) / original_height
                w_norm = w / original_width
                h_norm = h / original_height
                cls_id = label_to_id.get(bbox.get("label", "object"), 0)
                yolo_lines.append(
                    f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )

            # Save JSON
            json_filename = f"{file_prefix}frame_{frame_num:06d}.json"
            json_path = os.path.join(dataset_dir, split_name, "labels", json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2)

            # Save YOLO .txt (same base name as image; required by Ultralytics)
            txt_filename = f"{file_prefix}frame_{frame_num:06d}.txt"
            txt_path = os.path.join(dataset_dir, split_name, "labels", txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

    cap.release()

    # Write data.yaml so yolotrain can use the dataset directly
    _write_data_yaml(dataset_dir)

    return (
        dataset_dir,
        f"Dataset exported: {len(annotated_frames)} frames (train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])})",
    )


def _write_data_yaml(dataset_dir):
    """Write data.yaml (YOLO/Ultralytics format) into dataset_dir. Reads classes from classes.txt."""
    classes_file = os.path.join(dataset_dir, "classes.txt")
    if not os.path.exists(classes_file):
        return
    class_names = []
    try:
        with open(classes_file, encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
    except Exception:
        return
    if not class_names:
        return
    train_path = os.path.join(dataset_dir, "train", "images")
    val_path = os.path.join(dataset_dir, "val", "images")
    train_path = os.path.abspath(train_path).replace("\\", "/")
    val_path = os.path.abspath(val_path).replace("\\", "/")
    test_path = os.path.join(dataset_dir, "test", "images")
    test_line = (
        f"\ntest: {os.path.abspath(test_path).replace(os.sep, '/')}"
        if os.path.isdir(test_path)
        else ""
    )
    yaml_content = f"""# YOLO dataset - generated by vailá getpixelvideo
path: .
train: {train_path}
val: {val_path}{test_line}
nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
    except Exception:
        pass


def get_video_path():
    # Use the same format as cutvideo.py which works on Linux
    # Single tuple with all extensions in one string works better on Linux tkinter
    file_types = [("Video Files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV")]

    try:
        import tkinter as tk
        from tkinter import filedialog

        # Create a hidden root window
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        root.attributes("-topmost", True)  # Bring to front
        root.update_idletasks()  # Process any pending events

        # Open file dialog - use same format as cutvideo.py (works on Linux)
        video_path = filedialog.askopenfilename(title="Select Video File", filetypes=file_types)

        if video_path:
            print(f"Video selected: {video_path}")

        root.destroy()  # Clean up

        # Convert to None if empty string
        video_path = video_path if video_path else None

    except Exception as e:
        print(f"Error with file dialog: {e}")
        import traceback

        traceback.print_exc()
        video_path = None

    return video_path


# This function is a duplicate/legacy version - keeping for compatibility
# The main load_coordinates_from_file function is defined above


def run_getpixelvideo(initial_dataset_dir=None):
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting GetPixelVideo...")
    print("-" * 80)

    video_path = get_video_path()
    if not video_path:
        print("No video selected. Exiting.")
        return

    # User requested to remove the startup prompt since there is a Load button in the GUI.
    # defaulting to False (starting fresh)
    print("\n" + "=" * 50)
    print("vailá - Pixel Coordinate Tool")
    print("=" * 50)
    print("Starting fresh. Use the 'Load' button to import keypoints.")

    load_existing = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if load_existing:
        loaded_data, labels = load_coordinates_from_file(total_frames, vw, vh)

        # Check if it's a YOLO dataset marker
        if isinstance(loaded_data, dict) and "_yolo_dataset" in loaded_data:
            # User selected a YOLO dataset, load it now
            dataset_path = loaded_data["_yolo_dataset"]
            print(f"\nLoading YOLO dataset from: {dataset_path}")
            coordinates = load_yolo_dataset(dataset_path, video_path, total_frames, vw, vh)
            if not coordinates:
                print("Failed to load YOLO dataset. Starting fresh.")
                coordinates = None
        else:
            coordinates = loaded_data
    else:
        coordinates = None
        labels = []

    while True:
        initial_labeling_mode = getattr(run_getpixelvideo, "_initial_labeling_mode", None)
        result = play_video_with_controls(
            video_path,
            coordinates,
            labels,
            initial_dataset_dir=initial_dataset_dir,
            initial_labeling_mode=initial_labeling_mode,
        )
        # F8 "Open another video" returns (switch_video, new_path, current_dataset_dir, labeling_mode)
        if result and len(result) >= 3 and result[0] == "switch_video":
            video_path = result[1]
            initial_dataset_dir = result[2]
            run_getpixelvideo._initial_labeling_mode = result[3] if len(result) > 3 else False
            coordinates = None
            labels = []
            continue
        break


if __name__ == "__main__":
    initial_dataset_dir = None
    if "--dataset" in sys.argv:
        idx = sys.argv.index("--dataset")
        if idx + 1 < len(sys.argv):
            initial_dataset_dir = sys.argv[idx + 1]
            if not os.path.isdir(initial_dataset_dir):
                print(f"Error: --dataset path is not a directory: {initial_dataset_dir}")
                initial_dataset_dir = None
    run_getpixelvideo(initial_dataset_dir=initial_dataset_dir)
