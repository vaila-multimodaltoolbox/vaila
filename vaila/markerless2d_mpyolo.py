"""
Script: markerless2d_mpyolo.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Created: 18 February 2025
Last Updated: 10 November 2025
Version: 0.0.2

Description:
This script combines YOLOv11 for person detection/tracking with MediaPipe for pose estimation.

Usage example:
python markerless2d_mpyolo.py -i input_directory -o output_directory -c config.toml

Requirements:
- Python 3.12.12
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Tkinter (usually included with Python installations)
- Pillow (if using image manipulation: `pip install Pillow`)
- Pandas (for coordinate conversion: `pip install pandas`)
- psutil (pip install psutil) - for memory monitoring

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
4. Original Coordinates CSV (`*_mp_original.csv`):
   If resize was used, coordinates converted back to original video dimensions.
5. Log File (`log_info.txt`):
   A log file containing video metadata and processing information.

License:
    This project is licensed under the terms of AGPLv3.0.
"""

import colorsys  # Adicionar esta importação no topo do arquivo
import datetime
import os
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Configurações para evitar conflitos
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

# COCO classes dictionary
COCO_CLASSES = {
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

# Adicionar estas funções após as importações, próximo ao início do arquivo
COLORS = None


def get_color_palette(num_colors=200):
    """Generates a maximally distinct color palette using HSV color space."""
    print(f"Generating a palette with {num_colors} distinct colors")

    # Base palette with distinct colors for the first few IDs
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


def get_color_for_id(tracker_id):
    """Returns a distinct color for the ID using a pre-calculated palette."""
    global COLORS

    # Lazy-initialize the color palette (100 colors should be enough for most tracking needs)
    if COLORS is None:
        COLORS = get_color_palette(100)

    # Use modulo to reuse colors if we have more IDs than colors
    color_idx = tracker_id % len(COLORS)
    return COLORS[color_idx]


def download_model(model_name):
    """
    Download a specific YOLO model to the vaila/vaila/models directory.

    Args:
        model_name: Name of the model to download (e.g., "yolo11x.pt")

    Returns:
        Path to the downloaded model
    """
    # Correto caminho para vaila/vaila/models
    script_dir = os.path.dirname(os.path.abspath(__file__))  # vaila/
    vaila_dir = os.path.dirname(script_dir)  # root directory
    models_dir = os.path.join(vaila_dir, "vaila", "models")  # vaila/vaila/models

    # Create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be downloaded to: {models_dir}")

    model_path = os.path.join(models_dir, model_name)

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}, using existing file.")
        return model_path

    print(f"Downloading {model_name} to {model_path}...")
    try:
        # Create a temporary YOLO model instance that will download the weights
        model = YOLO(model_name)

        # Get the path where YOLO downloaded the model
        source_path = model.ckpt_path

        if os.path.exists(source_path):
            # Copy the downloaded model to our models directory
            import shutil

            shutil.copy2(source_path, model_path)
            print(f"Successfully saved {model_name} to {model_path}")
        else:
            print(f"YOLO downloaded the model but couldn't find it at {source_path}")

    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        print("Trying alternative download method...")

        try:
            # Try downloading through Ultralytics Hub
            from ultralytics.utils.downloads import attempt_download

            attempt_download(model_path, model_name)
            if os.path.exists(model_path):
                print(f"Successfully downloaded {model_name} using attempt_download")
            else:
                print(f"Failed to download {model_name} to {model_path}")
        except Exception as e2:
            print(f"All download methods failed for {model_name}: {e2}")

    return model_path


def initialize_csv(output_dir, class_name, object_id, is_person=False):
    """Initialize a CSV file for a specific class."""
    csv_path = os.path.join(output_dir, f"{class_name}_{object_id}.csv")

    if is_person:
        # For persons, include MediaPipe pose landmarks
        columns = [
            "frame",
            "person_id",
            "yolo_bbox_x1",
            "yolo_bbox_y1",
            "yolo_bbox_x2",
            "yolo_bbox_y2",
            "yolo_confidence",
        ]
        for idx in range(33):
            columns.extend(
                [
                    f"landmark_{idx}_x",
                    f"landmark_{idx}_y",
                    f"landmark_{idx}_z",
                    f"landmark_{idx}_visibility",
                ]
            )
    else:
        # For other classes, save only bounding box coordinates
        columns = ["frame", "object_id", "x1", "y1", "x2", "y2", "confidence"]

    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, index=False)
    return csv_path


def save_detection_to_csv(csv_path, frame_idx, object_id, box, confidence):
    """Salva os dados da detecção no CSV."""
    x_min, y_min, x_max, y_max = map(float, box)
    row_data = {
        "frame": frame_idx,
        "object_id": object_id,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "confidence": confidence,
    }

    df = pd.DataFrame([row_data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def save_landmarks_to_csv(csv_path, frame_idx, person_id, landmarks):
    """Salva os dados dos landmarks no CSV."""
    row_data = {"frame": frame_idx, "person_id": person_id}

    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            prefix = f"landmark_{idx}"
            row_data[f"{prefix}_x"] = landmark.x
            row_data[f"{prefix}_y"] = landmark.y
            row_data[f"{prefix}_z"] = landmark.z
            row_data[f"{prefix}_visibility"] = landmark.visibility
    else:
        # Preenche com NaN quando não há landmarks
        for idx in range(33):  # MediaPipe Pose tem 33 landmarks
            prefix = f"landmark_{idx}"
            row_data[f"{prefix}_x"] = np.nan
            row_data[f"{prefix}_y"] = np.nan
            row_data[f"{prefix}_z"] = np.nan
            row_data[f"{prefix}_visibility"] = np.nan

    # Append ao CSV existente
    df = pd.DataFrame([row_data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def get_parameters_dialog():
    """Create a dialog for MediaPipe and YOLO parameters."""
    dialog = tk.Tk()
    dialog.title("Detection Parameters Yolo and MediaPipe")

    # Set a large initial size right after creating the dialog
    dialog.geometry("1024x768")

    # Create main scrollable frame
    main_frame = tk.Frame(dialog)
    main_frame.pack(fill="both", expand=True)

    # Add canvas with scrollbar
    canvas = tk.Canvas(main_frame)
    scrollbar = tk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    # Configure scrolling
    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Pack scrolling components
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Dictionary to store the results
    params = {}

    # Create frames inside scrollable_frame instead of main_frame
    mode_frame = tk.LabelFrame(scrollable_frame, text="Processing Mode", padx=5, pady=5)
    mode_frame.pack(padx=10, pady=5, fill="x")

    yolo_frame = tk.LabelFrame(scrollable_frame, text="YOLO Parameters", padx=5, pady=5)
    yolo_frame.pack(padx=10, pady=5, fill="x")

    mp_frame = tk.LabelFrame(scrollable_frame, text="MediaPipe Parameters", padx=5, pady=5)
    mp_frame.pack(padx=10, pady=5, fill="x")

    # Processing mode selection - using dropdown menu
    mode_select_frame = tk.Frame(mode_frame)
    mode_select_frame.pack(fill="x", padx=5, pady=5)

    # Label for the dropdown
    tk.Label(mode_select_frame, text="Select processing mode:").pack(side="left", padx=(0, 5))

    # Create dropdown (Combobox) for mode selection
    mode_var = tk.StringVar(value="sequential")
    mode_dropdown = ttk.Combobox(
        mode_select_frame, textvariable=mode_var, state="readonly", width=20
    )
    mode_dropdown["values"] = ("sequential", "multithreaded")
    mode_dropdown.pack(side="left", padx=5)

    # Set display names for the dropdown
    mode_display = {
        "sequential": "Sequential Processing",
        "multithreaded": "Multithreaded Tracking",
    }

    # Function to update the display text
    def update_dropdown_display():
        current_value = mode_var.get()
        mode_dropdown.set(mode_display.get(current_value, current_value))

    # Set initial display text
    update_dropdown_display()

    # Add help button with tooltip functionality
    def show_help_tooltip(_event=None):
        help_text = """
Sequential Processing:
• Process videos one at a time
• Full pipeline: detection, tracking, pose estimation
• Generates CSV files with normalized and pixel data
• Creates visualization video

Multithreaded Tracking:
• Process multiple videos simultaneously
• YOLO tracking only (no MediaPipe)
• Faster processing but less analytical data
• No pose data CSV files
        """
        x = help_button.winfo_rootx() + 25
        y = help_button.winfo_rooty() + 20

        # Create a toplevel window
        tooltip = tk.Toplevel(help_button)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tooltip,
            text=help_text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Arial", 10),
        )
        label.pack()

        def hide_tooltip():
            tooltip.destroy()

        tooltip.after(10000, hide_tooltip)  # Hide after 10 seconds
        help_button.bind("<Leave>", lambda e: tooltip.destroy())

    help_button = tk.Button(mode_select_frame, text="?", width=2, height=1)
    help_button.pack(side="left", padx=5)
    help_button.bind("<Button-1>", show_help_tooltip)

    # YOLO parameters with explanations
    tk.Label(yolo_frame, text="Confidence (0-1):").grid(row=0, column=0, sticky="e")
    yolo_conf = tk.Entry(yolo_frame)
    yolo_conf.insert(0, "0.25")  # Increased default for better quality detections
    yolo_conf.grid(row=0, column=1)
    tk.Label(
        yolo_frame,
        text="Confidence threshold for detections (higher = more selective)",
        font=("Arial", 8, "italic"),
    ).grid(row=1, column=0, columnspan=2)

    tk.Label(yolo_frame, text="IOU (0-1):").grid(row=2, column=0, sticky="e")
    yolo_iou = tk.Entry(yolo_frame)
    yolo_iou.insert(0, "0.7")
    yolo_iou.grid(row=2, column=1)
    tk.Label(
        yolo_frame,
        text="Intersection over Union threshold for overlap (higher = more overlapping allowed)",
        font=("Arial", 8, "italic"),
    ).grid(row=3, column=0, columnspan=2)

    # Classes display in grid
    classes_frame = tk.LabelFrame(
        scrollable_frame,
        text="Available Classes (Enter numbers separated by commas)",
        padx=5,
        pady=5,
    )
    classes_frame.pack(padx=10, pady=5, fill="both", expand=True)

    # Display classes in a 5-column grid
    cols = 5
    for idx, name in COCO_CLASSES.items():
        row = idx // cols
        col = idx % cols
        tk.Label(classes_frame, text=f"{idx}: {name}", font=("Courier", 8)).grid(
            row=row, column=col, sticky="w", padx=5
        )

    # Class selection entry
    tk.Label(yolo_frame, text="Selected Classes:").grid(row=4, column=0, sticky="e")
    class_entry = tk.Entry(yolo_frame)
    class_entry.insert(0, "0")  # Default to person class
    class_entry.grid(row=4, column=1)
    tk.Label(
        yolo_frame,
        text="Enter class numbers separated by commas (e.g., '0,1,2')",
        font=("Arial", 8, "italic"),
    ).grid(row=5, column=0, columnspan=2)

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

    # Enhanced processing parameters
    enhanced_frame = tk.LabelFrame(
        scrollable_frame, text="Enhanced Processing Parameters", padx=5, pady=5
    )
    enhanced_frame.pack(padx=10, pady=5, fill="x")

    tk.Label(enhanced_frame, text="Bbox Scale Factor (2-8):").grid(row=0, column=0, sticky="e")
    scale_factor = tk.Entry(enhanced_frame)
    scale_factor.insert(0, "4")
    scale_factor.grid(row=0, column=1)
    tk.Label(
        enhanced_frame,
        text="Higher values = better quality but slower",
        font=("Arial", 8, "italic"),
    ).grid(row=1, column=0, columnspan=2)

    tk.Label(enhanced_frame, text="Safety Margin (0.1-0.5):").grid(row=2, column=0, sticky="e")
    safety_margin = tk.Entry(enhanced_frame)
    safety_margin.insert(0, "0.25")
    safety_margin.grid(row=2, column=1)
    tk.Label(
        enhanced_frame,
        text="Larger values prevent landmarks from exploding outside bbox",
        font=("Arial", 8, "italic"),
    ).grid(row=3, column=0, columnspan=2)

    def on_submit():
        try:
            # Store the selected processing mode
            params["selected_mode"] = mode_var.get()
            print(
                f"Selected processing mode: {mode_display.get(params['selected_mode'], params['selected_mode'])}"
            )

            params["yolo_conf"] = float(yolo_conf.get())
            params["yolo_iou"] = float(yolo_iou.get())

            # Parse class numbers
            class_str = class_entry.get().strip()
            if not class_str:
                tk.messagebox.showerror("Error", "Please enter at least one class number")
                return

            try:
                selected_classes = [int(x.strip()) for x in class_str.split(",")]
                # Validate class numbers
                invalid_classes = [x for x in selected_classes if x not in COCO_CLASSES]
                if invalid_classes:
                    tk.messagebox.showerror("Error", f"Invalid class numbers: {invalid_classes}")
                    return
                params["yolo_classes"] = selected_classes
            except ValueError:
                tk.messagebox.showerror(
                    "Error",
                    "Invalid class number format. Use numbers separated by commas",
                )
                return

            # Adiciona static_image_mode aos parâmetros
            static_mode_text = mp_static_mode.get().strip().lower()
            if static_mode_text not in ["true", "false"]:
                tk.messagebox.showerror("Error", "Static Image Mode must be 'True' or 'False'")
                return
            params["mp_static_mode"] = static_mode_text == "true"

            params["mp_complexity"] = int(mp_complexity.get())
            params["mp_detection_conf"] = float(mp_detection_conf.get())
            params["mp_tracking_conf"] = float(mp_tracking_conf.get())

            # Enhanced processing parameters
            scale_factor_val = int(scale_factor.get())
            safety_margin_val = float(safety_margin.get())

            if not (2 <= scale_factor_val <= 8):
                tk.messagebox.showerror("Error", "Scale Factor must be between 2 and 8")
                return

            if not (0.1 <= safety_margin_val <= 0.5):
                tk.messagebox.showerror("Error", "Safety Margin must be between 0.1 and 0.5")
                return

            params["scale_factor"] = scale_factor_val
            params["safety_margin"] = safety_margin_val

            dialog.quit()
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter valid values")

    tk.Button(scrollable_frame, text="Start Processing", command=on_submit).pack(pady=10)

    # No final da função, configure o tamanho do canvas
    canvas.config(width=1004, height=748)  # Ligeiramente menor que a janela

    dialog.mainloop()

    return params if params else None


def process_person_with_mediapipe_enhanced(
    frame,
    bbox,
    pose,
    width,
    height,
    mp_drawing,
    mp_pose,
    scale_factor=4,
    safety_margin=0.25,
):
    """
    Enhanced MediaPipe processing with 4x scaling and safety margins to prevent landmarks from exploding outside bbox.

    Args:
        frame: Original video frame
        bbox: YOLO bounding box [x1, y1, x2, y2]
        pose: MediaPipe pose estimator
        width: Frame width
        height: Frame height
        mp_drawing: MediaPipe drawing utils
        mp_pose: MediaPipe pose solutions
        scale_factor: Factor to scale the bounding box (default 4)
        safety_margin: Safety margin as percentage of bbox size (default 0.25 = 25%)

    Returns:
        landmarks_normalized: Landmarks in original frame coordinates (0-1)
        landmarks_pixels: Landmarks in original frame pixel coordinates
        annotated_crop: Annotated crop image
        final_bbox: Final padded bounding box used
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)

        # Validação das dimensões
        if width <= 0 or height <= 0:
            print("Invalid frame dimensions")
            return None, None, None, None

        if x1 >= x2 or y1 >= y2:
            print("Invalid bounding box coordinates")
            return None, None, None, None

        # Calculate larger safety margins to prevent MediaPipe from exploding outside bbox
        box_w = x2 - x1
        box_h = y2 - y1

        # Use configurable safety margin (default 25%) for better containment
        pad_x = int(box_w * safety_margin)
        pad_y = int(box_h * safety_margin)

        # Apply padding with bounds checking
        x1_pad = max(0, x1 - pad_x)
        x2_pad = min(width, x2 + pad_x)
        y1_pad = max(0, y1 - pad_y)
        y2_pad = min(height, y2 + pad_y)

        # Extract person crop from original frame (no masking to avoid artifacts)
        person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

        if person_crop.size == 0:
            return None, None, None, None

        # Get original crop dimensions
        crop_height, crop_width = person_crop.shape[:2]

        # Scale the crop by the scale factor for better MediaPipe accuracy
        new_width = int(crop_width * scale_factor)
        new_height = int(crop_height * scale_factor)

        # Resize using high-quality interpolation
        scaled_crop = cv2.resize(
            person_crop, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        # Process with MediaPipe on the scaled image
        crop_rgb = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2RGB)
        results = pose.process(crop_rgb)

        if not results.pose_landmarks:
            return None, None, None, None

        # Convert landmarks back to original frame coordinates with validation
        landmarks_normalized = []
        landmarks_pixels = []

        # Original bbox bounds for validation (with some tolerance)
        original_x1, original_y1 = x1 - pad_x, y1 - pad_y
        original_x2, original_y2 = x2 + pad_x, y2 + pad_y

        # Add extra tolerance for landmark validation
        tolerance = max(box_w, box_h) * 0.1  # 10% of largest dimension

        valid_landmarks_count = 0

        for landmark in results.pose_landmarks.landmark:
            # Landmark coordinates are in the scaled crop (0-1)
            # Convert to pixel coordinates in the scaled crop
            scaled_x = landmark.x * new_width
            scaled_y = landmark.y * new_height

            # Convert back to original crop coordinates
            original_crop_x = scaled_x / scale_factor
            original_crop_y = scaled_y / scale_factor

            # Convert to global frame coordinates (pixels)
            global_x = original_crop_x + x1_pad
            global_y = original_crop_y + y1_pad

            # Validate landmark is within reasonable bounds (with tolerance)
            if (
                original_x1 - tolerance <= global_x <= original_x2 + tolerance
                and original_y1 - tolerance <= global_y <= original_y2 + tolerance
                and landmark.visibility > 0.3
            ):  # Only accept visible landmarks
                valid_landmarks_count += 1

                # Store pixel coordinates
                landmarks_pixels.append(
                    {
                        "x": int(global_x),
                        "y": int(global_y),
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                )

                # Store normalized coordinates (0-1)
                landmarks_normalized.append(
                    {
                        "x": global_x / width,
                        "y": global_y / height,
                        "z": landmark.z,
                        "visibility": landmark.visibility,
                    }
                )
            else:
                # Add invalid/outside landmarks as NaN
                landmarks_pixels.append(
                    {
                        "x": np.nan,
                        "y": np.nan,
                        "z": landmark.z,
                        "visibility": 0.0,
                    }
                )

                landmarks_normalized.append(
                    {
                        "x": np.nan,
                        "y": np.nan,
                        "z": landmark.z,
                        "visibility": 0.0,
                    }
                )

        # Only accept results if we have enough valid landmarks (at least 10 out of 33)
        if valid_landmarks_count < 10:
            print(
                f"Warning: Only {valid_landmarks_count}/33 landmarks are valid, rejecting detection"
            )
            return None, None, None, None

        # Create annotated crop from the scaled image for better visualization
        annotated_crop = scaled_crop.copy()

        # Create a temporary landmarks object for drawing
        temp_landmarks = type("", (), {})()
        temp_landmarks.landmark = []

        for i, landmark in enumerate(results.pose_landmarks.landmark):
            if not (np.isnan(landmarks_pixels[i]["x"]) or np.isnan(landmarks_pixels[i]["y"])):
                temp_landmarks.landmark.append(landmark)

        # Draw landmarks on scaled crop if we have valid landmarks
        if len(temp_landmarks.landmark) > 0:
            mp_drawing.draw_landmarks(
                annotated_crop,
                results.pose_landmarks,  # Use original landmarks for drawing
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=3,
                    circle_radius=4,  # Green, larger for scaled image
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=2,  # White connections
                ),
            )

        # Resize annotated crop back to original size for consistency
        annotated_crop = cv2.resize(
            annotated_crop, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR
        )

        return (
            landmarks_normalized,
            landmarks_pixels,
            annotated_crop,
            (x1_pad, y1_pad, x2_pad, y2_pad),
        )

    except Exception as e:
        print(f"Error in process_person_with_mediapipe_enhanced: {e}")
        return None, None, None, None


# Keep the original function for backward compatibility
def process_person_with_mediapipe(frame, bbox, pose, width, height, mp_drawing, mp_pose):
    """Original function - now calls the enhanced version with default parameters"""
    return process_person_with_mediapipe_enhanced(
        frame,
        bbox,
        pose,
        width,
        height,
        mp_drawing,
        mp_pose,
        scale_factor=4,
        safety_margin=0.25,
    )


def save_person_data_to_csv(
    csv_path,
    frame_idx,
    person_id,
    bbox,
    confidence,
    landmarks_norm=None,
    landmarks_px=None,
):
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
        "frame": frame_idx,
        "person_id": person_id,
        "yolo_bbox_x1": bbox[0],
        "yolo_bbox_y1": bbox[1],
        "yolo_bbox_x2": bbox[2],
        "yolo_bbox_y2": bbox[3],
        "yolo_confidence": confidence,
    }

    if landmarks_norm and landmarks_px:
        for idx, (norm, px) in enumerate(zip(landmarks_norm, landmarks_px, strict=False)):
            prefix = f"landmark_{idx}"
            # Save normalized coordinates
            row_data[f"{prefix}_x_norm"] = norm["x"]
            row_data[f"{prefix}_y_norm"] = norm["y"]
            row_data[f"{prefix}_z_norm"] = norm["z"]
            row_data[f"{prefix}_visibility"] = norm["visibility"]
            # Save pixel coordinates
            row_data[f"{prefix}_x_px"] = px["x"]
            row_data[f"{prefix}_y_px"] = px["y"]
    else:
        # Fill with NaN when no landmarks detected
        for idx in range(33):
            prefix = f"landmark_{idx}"
            # Normalized
            row_data[f"{prefix}_x_norm"] = np.nan
            row_data[f"{prefix}_y_norm"] = np.nan
            row_data[f"{prefix}_z_norm"] = np.nan
            row_data[f"{prefix}_visibility"] = np.nan
            # Pixels
            row_data[f"{prefix}_x_px"] = np.nan
            row_data[f"{prefix}_y_px"] = np.nan

    df = pd.DataFrame([row_data])
    df.to_csv(csv_path, mode="a", header=False, index=False)


def process_yolo_tracking(video_path, output_dir, model, params):
    """First stage: YOLO detection and tracking"""
    tracking_data = {}  # Format: {class_id: {object_id: [frame_data]}}

    # Configurações para detecção múltipla
    model_params = {
        "conf": params["yolo_conf"],
        "iou": params["yolo_iou"],
        "classes": params["yolo_classes"],
        "persist": True,
        "stream": True,
        "max_det": 100,
        "verbose": False,
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

            for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs, strict=False):
                class_id = int(cls_id)
                object_id = int(track_id)

                # Get color for this object ID
                color = get_color_for_id(object_id)

                # Initialize dictionary structure if needed
                if class_id not in tracking_data:
                    tracking_data[class_id] = {}
                if object_id not in tracking_data[class_id]:
                    tracking_data[class_id][object_id] = []
                    # Create a CSV file for this object immediately
                    class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                    csv_path = os.path.join(output_dir, f"{class_name}_{object_id}.csv")

                    # Initialize CSV with appropriate headers
                    if class_id == 0:  # Person class
                        columns = [
                            "frame",
                            "object_id",
                            "x1",
                            "y1",
                            "x2",
                            "y2",
                            "confidence",
                            "color_r",
                            "color_g",
                            "color_b",
                        ]
                    else:  # Other classes
                        columns = [
                            "frame",
                            "object_id",
                            "x1",
                            "y1",
                            "x2",
                            "y2",
                            "confidence",
                            "color_r",
                            "color_g",
                            "color_b",
                        ]

                    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
                    print(f"Created tracking CSV for {class_name} ID:{object_id}")

                # Store detection data with color
                tracking_data[class_id][object_id].append(
                    {
                        "frame": frame_idx,
                        "bbox": box.tolist(),
                        "confidence": float(conf),
                        "color": color,
                    }
                )

                # Save this detection to the object's CSV file immediately
                class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                csv_path = os.path.join(output_dir, f"{class_name}_{object_id}.csv")

                # Prepare row data
                x1, y1, x2, y2 = map(float, box.tolist())
                row_data = {
                    "frame": frame_idx,
                    "object_id": object_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(conf),
                    "color_r": color[0],
                    "color_g": color[1],
                    "color_b": color[2],
                }

                # Append to CSV
                pd.DataFrame([row_data]).to_csv(csv_path, mode="a", header=False, index=False)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"\rProcessing frame {frame_idx}", end="")

    print(f"\nProcessed {frame_idx} frames")

    # Print summary of detected objects
    print("\nDetection Summary:")
    for class_id, objects in tracking_data.items():
        class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
        print(f"  {class_name}: {len(objects)} objects")
        for object_id in objects:
            frames = len(tracking_data[class_id][object_id])
            print(f"    ID:{object_id} - {frames} frames")

    return tracking_data


def process_mediapipe_pose(video_path, output_dir, tracking_data, params):
    """
    Second stage: MediaPipe pose estimation for tracked persons

    This function processes each person detected in the video and saves:
    1. A normalized landmarks CSV file (values 0-1)
    2. A pixel coordinates CSV file (absolute pixel values)

    The format matches the one used in markerless_2D_analysis.py
    """
    print("\nStage 2: MediaPipe Pose Estimation")

    # Initialize the pose manager
    pose_manager = MediaPipePoseManager(params)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process only person class (class_id = 0)
    if 0 in tracking_data:
        for object_id in tracking_data[0]:
            # Create CSV files for each person
            norm_csv_path = os.path.join(output_dir, f"person_{object_id}_norm.csv")
            pixel_csv_path = os.path.join(output_dir, f"person_{object_id}_pixel.csv")

            # Initialize CSV files with headers
            headers = ["frame_index"]
            for idx in range(33):  # MediaPipe has 33 landmarks
                headers.extend(
                    [
                        f"landmark_{idx}_x",
                        f"landmark_{idx}_y",
                        f"landmark_{idx}_z",
                        f"landmark_{idx}_visibility",
                    ]
                )

            with (
                open(norm_csv_path, "w") as f_norm,
                open(pixel_csv_path, "w") as f_pixel,
            ):
                f_norm.write(",".join(headers) + "\n")
                f_pixel.write(",".join(headers) + "\n")

            # Initialize arrays for all frames with NaN values
            for frame_idx in range(total_frames):
                # Check if this frame has a detection for this person
                frame_data = next(
                    (data for data in tracking_data[0][object_id] if data["frame"] == frame_idx),
                    None,
                )

                if frame_data:
                    # Process this frame for the person
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    bbox = frame_data["bbox"]
                    pose_estimator, _ = pose_manager.get_pose_estimator(bbox, object_id)

                    # Process the pose and get landmarks
                    landmarks = process_single_pose(frame, bbox, pose_estimator, width, height)

                    if landmarks:
                        # Save normalized landmarks
                        flat_landmarks_norm = [
                            coord for landmark in landmarks for coord in landmark
                        ]
                        landmarks_norm_str = ",".join(
                            (
                                ""
                                if (isinstance(value, float) and np.isnan(value))
                                else f"{value:.6f}"
                            )
                            for value in flat_landmarks_norm
                        )

                        # Calculate and save pixel landmarks
                        pixel_landmarks = []
                        for landmark in landmarks:
                            x_px = int(landmark[0] * width)
                            y_px = int(landmark[1] * height)
                            z = landmark[2]
                            vis = landmark[3]
                            pixel_landmarks.append([x_px, y_px, z, vis])

                        flat_landmarks_pixel = [
                            coord for landmark in pixel_landmarks for coord in landmark
                        ]
                        landmarks_pixel_str = ",".join(
                            ("" if (isinstance(value, float) and np.isnan(value)) else str(value))
                            for value in flat_landmarks_pixel
                        )

                        with open(norm_csv_path, "a") as f_norm:
                            f_norm.write(f"{frame_idx}," + landmarks_norm_str + "\n")

                        with open(pixel_csv_path, "a") as f_pixel:
                            f_pixel.write(f"{frame_idx}," + landmarks_pixel_str + "\n")
                    else:
                        # Write empty values for frames where pose estimation failed
                        write_nan_row(norm_csv_path, frame_idx, 33)
                        write_nan_row(pixel_csv_path, frame_idx, 33)
                else:
                    # Write empty values for frames where the person was not detected
                    write_nan_row(norm_csv_path, frame_idx, 33)
                    write_nan_row(pixel_csv_path, frame_idx, 33)

                # Show progress
                if frame_idx % 30 == 0:
                    print(
                        f"\rProcessing person {object_id}: {(frame_idx / total_frames) * 100:.1f}% ({frame_idx}/{total_frames})",
                        end="",
                    )

            print(f"\nCompleted processing for person {object_id}")

    cap.release()
    pose_manager.close_all()


def process_single_pose(frame, bbox, pose, width, height):
    """
    Process the pose for a single frame, drawing landmarks using proper pixel scaling.

    Returns both normalized landmarks and pixel coordinates for saving to CSV files.
    The pixel coordinates are calculated as:
        x_px = int(landmark.x * frame_width)
        y_px = int(landmark.y * frame_height)
    """
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    # Draw landmarks on the frame
    for landmark in results.pose_landmarks.landmark:
        x_px = int(landmark.x * width)
        y_px = int(landmark.y * height)
        # Draw a small circle at the landmark position
        cv2.circle(frame, (x_px, y_px), 2, (0, 255, 0), -1)

    # Return the landmarks in normalized format
    landmarks = []
    for landmark in results.pose_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

    return landmarks


def create_visualization_video(video_path, output_dir, tracking_data, params):
    """
    Create visualization video with bounding boxes, IDs, and pose landmarks (if available).
    Supports all detected classes.
    """
    print("\nCreating visualization video...")

    cap = cv2.VideoCapture(video_path)
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create temporary directory for storing frames
    temp_dir = os.path.join(output_dir, "temp_viz_frames")
    os.makedirs(temp_dir, exist_ok=True)

    # Load pose data for persons (class 0) from individual pixel CSV files
    pose_data = {}
    for object_id in tracking_data.get(0, {}):
        pixel_csv_path = os.path.join(output_dir, f"person_{object_id}_pixel.csv")
        if os.path.exists(pixel_csv_path):
            try:
                df = pd.read_csv(pixel_csv_path)
                # Convert the dataframe to a dictionary for quick frame access
                pose_data[object_id] = df.set_index("frame_index").to_dict("index")
                print(f"Loaded pose data for person ID:{object_id}")
            except Exception as e:
                print(f"Error loading pose data for person ID:{object_id}: {e}")

    mp_pose = mp.solutions.pose

    try:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw tracking boxes and labels for all classes
            for class_id, objects in tracking_data.items():
                for object_id, detections in objects.items():
                    # Retrieve detection data for the current frame
                    frame_data = next(
                        (data for data in detections if data["frame"] == frame_idx),
                        None,
                    )
                    if frame_data is None:
                        continue

                    bbox = frame_data["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)

                    # Use the color stored with the detection data, or get a new color
                    color = frame_data.get("color", get_color_for_id(object_id))

                    # Draw bounding box with the object's color
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Create label with class name and object ID
                    label = f"{COCO_CLASSES.get(class_id, str(class_id))} ID:{object_id}"
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    # Draw background rectangle for label with the same color
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width + 10, y1),
                        color,
                        -1,
                    )
                    # Draw label text with black color for better readability
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                    )

                    # Se for detecção de pessoa (classe 0) e houver dados de pose, desenha os landmarks.
                    if class_id == 0 and object_id in pose_data:
                        # Get frame data from the pose CSV
                        frame_landmarks = pose_data[object_id].get(frame_idx)

                        if frame_landmarks:
                            landmarks = []
                            for i in range(33):  # MediaPipe Pose tem 33 landmarks
                                try:
                                    x = frame_landmarks.get(f"landmark_{i}_x")
                                    y = frame_landmarks.get(f"landmark_{i}_y")
                                    vis = frame_landmarks.get(f"landmark_{i}_visibility", 0.0)

                                    if (
                                        x is not None
                                        and y is not None
                                        and not pd.isna(x)
                                        and not pd.isna(y)
                                    ):
                                        landmarks.append((int(x), int(y), vis))
                                    else:
                                        landmarks.append(None)
                                except (KeyError, TypeError):
                                    landmarks.append(None)

                            # Draw connections between landmarks
                            for connection in mp_pose.POSE_CONNECTIONS:
                                start_idx, end_idx = connection
                                if (
                                    landmarks[start_idx] is not None
                                    and landmarks[end_idx] is not None
                                ):
                                    cv2.line(
                                        frame,
                                        landmarks[start_idx][:2],
                                        landmarks[end_idx][:2],
                                        color,  # Use the object's color for connections
                                        1,
                                    )

                            # Draw landmark points
                            for lm in landmarks:
                                if lm is not None:
                                    cv2.circle(
                                        frame,
                                        (int(lm[0]), int(lm[1])),
                                        3,  # Larger size for better visibility
                                        color,  # Use the object's color for points
                                        -1,  # Filled circle
                                    )

                            print(
                                f"\rDrawing pose for person ID:{object_id} on frame {frame_idx}",
                                end="",
                            )

            # Save the annotated frame
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(frame_path, frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(
                    f"\rCreating visualization: {(frame_idx / total_frames) * 100:.1f}%",
                    end="",
                )

        # Create video using FFmpeg from the saved frames
        print("\nEncoding final video...")
        input_pattern = os.path.join(temp_dir, "frame_%06d.png")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video = os.path.join(output_dir, f"{video_name}_visualization.mp4")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            output_video,
        ]
        subprocess.run(ffmpeg_cmd, check=True)

    finally:
        cap.release()
        # Cleanup temporary frame files
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
        return np.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

    def get_pose_estimator(self, bbox, object_id):
        """Retorna o pose estimator apropriado para a bbox"""
        if len(self.pose_estimators) == 0:
            # Primeiro pose estimator
            pose = mp.solutions.pose.Pose(
                static_image_mode=self.params["mp_static_mode"],
                model_complexity=self.params["mp_complexity"],
                min_detection_confidence=self.params["mp_detection_conf"],
                min_tracking_confidence=self.params["mp_tracking_conf"],
            )
            self.pose_estimators.append(pose)
            self.pose_dims.append(bbox)
            return pose, 0

        elif object_id >= len(self.pose_estimators):
            # Verifica se precisa criar novo estimator
            threshold_for_new = 100
            best_score = float("inf")
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
                    min_tracking_confidence=self.params["mp_tracking_conf"],
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
        "frame": frame_idx,
        "person_id": person_id,
    }
    # Add landmarks data
    for _idx, (key, value) in enumerate(landmarks_px.items()):
        row_data[key] = value

    df = pd.DataFrame([row_data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def run_tracker_in_thread(model_path, video_source, tracker_config, output_dir, params):
    """
    Run YOLO tracker in its own thread for concurrent processing.
    Args:
        model_path (str): Local path of the model (or model name for auto-download).
        video_source (str): Path to the video file.
        tracker_config (str): Tracker configuration file (e.g. 'bytetrack.yaml').
        output_dir (str): Directory to save tracking data.
        params (dict): YOLO parameters.
    """
    # Each thread uses its own model instance
    model = YOLO(model_path)
    print(f"Started tracking for: {video_source}")

    # Configure YOLO parameters
    model_params = {
        "conf": params["yolo_conf"],
        "iou": params["yolo_iou"],
        "classes": params["yolo_classes"],
        "persist": True,
        "stream": True,
        "max_det": 100,
        "verbose": False,
        "tracker": tracker_config,
    }

    # Dictionary to store tracking data
    tracking_data = {}  # Format: {class_id: {object_id: [frame_data]}}

    # Process video with YOLO
    results = model.track(source=video_source, **model_params)

    frame_idx = 0
    for result in results:
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs, strict=False):
                class_id = int(cls_id)
                object_id = int(track_id)

                # Initialize dictionary structure if needed
                if class_id not in tracking_data:
                    tracking_data[class_id] = {}
                if object_id not in tracking_data[class_id]:
                    tracking_data[class_id][object_id] = []
                    # Create a CSV file for this object immediately
                    class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                    csv_path = os.path.join(output_dir, f"{class_name}_{object_id}.csv")

                    # Initialize CSV with appropriate headers
                    columns = [
                        "frame",
                        "object_id",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "confidence",
                    ]
                    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
                    print(f" Created tracking CSV for {class_name} ID:{object_id}")

                # Store detection data
                tracking_data[class_id][object_id].append(
                    {
                        "frame": frame_idx,
                        "bbox": box.tolist(),
                        "confidence": float(conf),
                    }
                )

                # Save this detection to the object's CSV file immediately
                class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
                csv_path = os.path.join(output_dir, f"{class_name}_{object_id}.csv")

                # Prepare row data
                x1, y1, x2, y2 = map(float, box.tolist())
                row_data = {
                    "frame": frame_idx,
                    "object_id": object_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": float(conf),
                }

                # Append to CSV
                pd.DataFrame([row_data]).to_csv(csv_path, mode="a", header=False, index=False)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(
                f"\rProcessing {os.path.basename(video_source)}: frame {frame_idx}",
                end="",
            )

    print(f"\nFinished tracking for: {video_source}")

    # Print summary of detected objects
    print(f"\nDetection Summary for {os.path.basename(video_source)}:")
    for class_id, objects in tracking_data.items():
        class_name = COCO_CLASSES.get(class_id, f"class_{class_id}")
        print(f"  {class_name}: {len(objects)} objects")
        for object_id in objects:
            frames = len(tracking_data[class_id][object_id])
            print(f"    ID:{object_id} - {frames} frames")


def run_multithreaded_tracking():
    """
    Spawns a separate thread for each video in a chosen directory to run YOLO tracking concurrently.
    Inspired by the multithreaded tracking example at:
    https://docs.ultralytics.com/modes/track/#multithreaded-tracking
    """
    root = tk.Tk()
    root.withdraw()
    video_dir = filedialog.askdirectory(title="Select Directory with Videos for Tracking")
    if not video_dir:
        print("No directory selected for tracking. Exiting.")
        return

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if os.path.isfile(os.path.join(video_dir, f)) and f.endswith(video_extensions)
    ]
    if not video_files:
        print("No video files found in the selected directory.")
        return

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(video_dir, f"multithreaded_tracking_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Default tracker configuration (change as needed: e.g. "botsort.yaml" or "bytetrack.yaml")
    tracker_config = "botsort.yaml"

    # Update the model path construction
    model_name = "yolo11x.pt"

    # Correct path to vaila/vaila/models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vaila_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(vaila_dir, "vaila", "models")
    model_path = os.path.join(models_dir, model_name)

    # Check if the model exists, download only if necessary
    if not os.path.exists(model_path):
        model_path = download_model(model_name)
    else:
        print(f"Found local model for tracking: {model_path}")

    # Get parameters for YOLO
    params = get_parameters_dialog()
    if not params:
        print("No parameters set. Exiting...")
        return

    threads = []
    for video_file in video_files:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        t = threading.Thread(
            target=run_tracker_in_thread,
            args=(
                model_path,
                video_file,
                tracker_config,
                video_output_dir,
                params,
            ),
            daemon=True,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print("Multithreaded tracking complete.")
    print(f"All results saved to: {output_dir}")

    # Try to open output folder
    try:
        if os.name == "nt":  # Windows
            os.startfile(output_dir)
        elif os.name == "posix":  # macOS and Linux
            subprocess.run(["xdg-open", output_dir])
    except Exception as e:
        print(f"Could not open the output directory: {e}")

    root.destroy()


def save_tracking_data_to_csv(tracking_data, output_dir):
    """
    This function is no longer needed, as the data is saved during processing.
    Kept for compatibility, but does not perform any operation.
    """
    print("Tracking data already saved during processing.")
    pass


def process_video_enhanced(video_path, output_dir, model, params, mp_pose, mp_drawing):
    """
    Enhanced video processing pipeline with integrated YOLO tracking and MediaPipe pose estimation
    using 4x scaling and safety margins to prevent landmarks from exploding outside bbox.
    """
    print(f"Starting enhanced processing with scale factor: {params['scale_factor']}x")
    print(f"Safety margin: {params['safety_margin'] * 100:.0f}%")

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")

    # Initialize MediaPipe pose estimator
    pose = mp_pose.Pose(
        static_image_mode=params["mp_static_mode"],
        model_complexity=params["mp_complexity"],
        min_detection_confidence=params["mp_detection_conf"],
        min_tracking_confidence=params["mp_tracking_conf"],
    )

    # Storage for tracking data and CSV files
    tracking_data = {}
    csv_files = {}

    # Configure YOLO parameters
    model_params = {
        "conf": params["yolo_conf"],
        "iou": params["yolo_iou"],
        "classes": params["yolo_classes"],
        "persist": True,
        "stream": True,
        "max_det": 100,
        "verbose": False,
    }

    print("\nStarting enhanced YOLO tracking + MediaPipe pose estimation...")

    # Process video with YOLO tracking
    results = model.track(source=video_path, **model_params)

    frame_idx = 0
    successful_pose_detections = 0
    total_person_detections = 0

    try:
        for result in results:
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                clss = result.boxes.cls.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                # Get the current frame for MediaPipe processing
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                for box, track_id, cls_id, conf in zip(boxes, ids, clss, confs, strict=False):
                    class_id = int(cls_id)
                    object_id = int(track_id)

                    # Initialize tracking data structures
                    if class_id not in tracking_data:
                        tracking_data[class_id] = {}
                    if object_id not in tracking_data[class_id]:
                        tracking_data[class_id][object_id] = []

                    # Get color for this object
                    color = get_color_for_id(object_id)

                    # Store tracking data
                    tracking_data[class_id][object_id].append(
                        {
                            "frame": frame_idx,
                            "bbox": box.tolist(),
                            "confidence": float(conf),
                            "color": color,
                        }
                    )

                    # Process persons (class 0) with enhanced MediaPipe
                    if class_id == 0:  # Person class
                        total_person_detections += 1

                        # Initialize CSV file for this person if needed
                        if object_id not in csv_files:
                            csv_path = os.path.join(output_dir, f"person_{object_id}_enhanced.csv")
                            csv_files[object_id] = csv_path

                            # Create CSV with enhanced headers
                            columns = [
                                "frame",
                                "person_id",
                                "yolo_bbox_x1",
                                "yolo_bbox_y1",
                                "yolo_bbox_x2",
                                "yolo_bbox_y2",
                                "yolo_confidence",
                                "scale_factor",
                                "safety_margin",
                                "valid_landmarks_count",
                            ]
                            for idx in range(33):
                                columns.extend(
                                    [
                                        f"landmark_{idx}_x_norm",
                                        f"landmark_{idx}_y_norm",
                                        f"landmark_{idx}_z_norm",
                                        f"landmark_{idx}_visibility",
                                        f"landmark_{idx}_x_px",
                                        f"landmark_{idx}_y_px",
                                    ]
                                )

                            pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
                            print(f"Created enhanced CSV for person ID:{object_id}")

                        # Apply enhanced MediaPipe processing
                        landmarks_norm, landmarks_px, annotated_crop, final_bbox = (
                            process_person_with_mediapipe_enhanced(
                                frame,
                                box,
                                pose,
                                width,
                                height,
                                mp_drawing,
                                mp_pose,
                                scale_factor=params["scale_factor"],
                                safety_margin=params["safety_margin"],
                            )
                        )

                        # Count valid landmarks
                        valid_count = 0
                        if landmarks_norm and landmarks_px:
                            valid_count = sum(
                                1
                                for lm in landmarks_px
                                if not (np.isnan(lm["x"]) or np.isnan(lm["y"]))
                            )
                            successful_pose_detections += 1

                        # Save enhanced data to CSV
                        save_enhanced_person_data(
                            csv_files[object_id],
                            frame_idx,
                            object_id,
                            box,
                            conf,
                            landmarks_norm,
                            landmarks_px,
                            params,
                            valid_count,
                        )

            frame_idx += 1
            if frame_idx % 30 == 0:
                success_rate = (successful_pose_detections / max(total_person_detections, 1)) * 100
                print(
                    f"\rProcessing frame {frame_idx}/{total_frames} ({(frame_idx / total_frames) * 100:.1f}%) - "
                    f"Pose success: {success_rate:.1f}%",
                    end="",
                )

        print("\nCompleted enhanced processing!")
        print(f"Total person detections: {total_person_detections}")
        print(f"Successful pose detections: {successful_pose_detections}")
        print(
            f"Success rate: {(successful_pose_detections / max(total_person_detections, 1) * 100):.1f}%"
        )

        # Create enhanced visualization
        create_enhanced_visualization_video(video_path, output_dir, tracking_data, params)

        # Save processing parameters
        params_file = os.path.join(output_dir, "enhanced_processing_parameters.json")
        with open(params_file, "w") as f:
            import json

            enhanced_params = params.copy()
            enhanced_params["total_person_detections"] = total_person_detections
            enhanced_params["successful_pose_detections"] = successful_pose_detections
            enhanced_params["success_rate"] = (
                successful_pose_detections / max(total_person_detections, 1) * 100
            )
            json.dump(enhanced_params, f, indent=4)

    finally:
        cap.release()
        pose.close()

    print(f"\nEnhanced processing complete! Results saved to: {output_dir}")
    print("Key improvements applied:")
    print(f"  - {params['scale_factor']}x bbox scaling for better pose detection")
    print(f"  - {params['safety_margin'] * 100:.0f}% safety margin to prevent landmark explosion")
    print("  - Landmark validation with minimum 10/33 valid landmarks required")

    # Try to open output folder
    try:
        if os.name == "nt":  # Windows
            os.startfile(output_dir)
        elif os.name == "posix":  # macOS and Linux
            subprocess.run(["xdg-open", output_dir])
    except Exception as e:
        print(f"Could not open output directory: {e}")


def save_enhanced_person_data(
    csv_path,
    frame_idx,
    person_id,
    bbox,
    confidence,
    landmarks_norm=None,
    landmarks_px=None,
    params=None,
    valid_count=0,
):
    """Save enhanced person detection and pose data to CSV."""
    row_data = {
        "frame": frame_idx,
        "person_id": person_id,
        "yolo_bbox_x1": bbox[0],
        "yolo_bbox_y1": bbox[1],
        "yolo_bbox_x2": bbox[2],
        "yolo_bbox_y2": bbox[3],
        "yolo_confidence": confidence,
        "scale_factor": params["scale_factor"] if params else 4,
        "safety_margin": params["safety_margin"] if params else 0.25,
        "valid_landmarks_count": valid_count,
    }

    if landmarks_norm and landmarks_px:
        for idx, (norm, px) in enumerate(zip(landmarks_norm, landmarks_px, strict=False)):
            prefix = f"landmark_{idx}"
            # Save normalized coordinates
            row_data[f"{prefix}_x_norm"] = norm["x"]
            row_data[f"{prefix}_y_norm"] = norm["y"]
            row_data[f"{prefix}_z_norm"] = norm["z"]
            row_data[f"{prefix}_visibility"] = norm["visibility"]
            # Save pixel coordinates
            row_data[f"{prefix}_x_px"] = px["x"]
            row_data[f"{prefix}_y_px"] = px["y"]
    else:
        # Fill with NaN when no landmarks detected
        for idx in range(33):
            prefix = f"landmark_{idx}"
            # Normalized
            row_data[f"{prefix}_x_norm"] = np.nan
            row_data[f"{prefix}_y_norm"] = np.nan
            row_data[f"{prefix}_z_norm"] = np.nan
            row_data[f"{prefix}_visibility"] = np.nan
            # Pixels
            row_data[f"{prefix}_x_px"] = np.nan
            row_data[f"{prefix}_y_px"] = np.nan

    df = pd.DataFrame([row_data])
    df.to_csv(csv_path, mode="a", header=False, index=False)


def create_enhanced_visualization_video(video_path, output_dir, tracking_data, params):
    """Create enhanced visualization video showing improved pose detection."""
    print("\nCreating enhanced visualization video...")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video = os.path.join(output_dir, f"{video_name}_enhanced_visualization.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Load pose data for persons from enhanced CSV files
    pose_data = {}
    for object_id in tracking_data.get(0, {}):
        csv_path = os.path.join(output_dir, f"person_{object_id}_enhanced.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                pose_data[object_id] = df.set_index("frame").to_dict("index")
                print(f"Loaded enhanced pose data for person ID:{object_id}")
            except Exception as e:
                print(f"Error loading pose data for person ID:{object_id}: {e}")

    mp_pose = mp.solutions.pose

    try:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Add enhanced processing info overlay
            info_text = f"Enhanced: {params['scale_factor']}x scaling, {params['safety_margin'] * 100:.0f}% margin"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Draw tracking boxes and enhanced pose for all detected persons
            for class_id, objects in tracking_data.items():
                if class_id != 0:  # Only process persons (class 0)
                    continue

                for object_id, detections in objects.items():
                    # Get detection data for current frame
                    frame_data = next(
                        (data for data in detections if data["frame"] == frame_idx),
                        None,
                    )
                    if frame_data is None:
                        continue

                    bbox = frame_data["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)
                    color = frame_data.get("color", get_color_for_id(object_id))

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw enhanced label with pose info
                    if object_id in pose_data:
                        frame_landmarks = pose_data[object_id].get(frame_idx)
                        valid_count = (
                            frame_landmarks.get("valid_landmarks_count", 0)
                            if frame_landmarks
                            else 0
                        )
                        label = f"Person ID:{object_id} ({valid_count}/33)"
                    else:
                        label = f"Person ID:{object_id}"

                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_height - 10),
                        (x1 + label_width + 10, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                    )

                    # Draw enhanced pose if available
                    if object_id in pose_data:
                        frame_landmarks = pose_data[object_id].get(frame_idx)

                        if frame_landmarks:
                            landmarks = []
                            for i in range(33):
                                try:
                                    x = frame_landmarks.get(f"landmark_{i}_x_px")
                                    y = frame_landmarks.get(f"landmark_{i}_y_px")
                                    vis = frame_landmarks.get(f"landmark_{i}_visibility", 0.0)

                                    if (
                                        x is not None
                                        and y is not None
                                        and not pd.isna(x)
                                        and not pd.isna(y)
                                    ):
                                        landmarks.append((int(x), int(y), vis))
                                    else:
                                        landmarks.append(None)
                                except (KeyError, TypeError):
                                    landmarks.append(None)

                            # Draw connections with enhanced styling
                            for connection in mp_pose.POSE_CONNECTIONS:
                                start_idx, end_idx = connection
                                if (
                                    landmarks[start_idx] is not None
                                    and landmarks[end_idx] is not None
                                ):
                                    cv2.line(
                                        frame,
                                        landmarks[start_idx][:2],
                                        landmarks[end_idx][:2],
                                        color,
                                        3,
                                    )  # Thicker lines

                            # Draw landmark points with enhanced visibility
                            for lm in landmarks:
                                if lm is not None:
                                    cv2.circle(
                                        frame, (int(lm[0]), int(lm[1])), 5, color, -1
                                    )  # Larger points
                                    cv2.circle(
                                        frame,
                                        (int(lm[0]), int(lm[1])),
                                        5,
                                        (255, 255, 255),
                                        1,
                                    )  # White border

            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(
                    f"\rCreating enhanced visualization: {(frame_idx / total_frames) * 100:.1f}%",
                    end="",
                )

        print(f"\nEnhanced visualization video saved to: {output_video}")

    finally:
        cap.release()
        out.release()


def run_markerless2d_mpyolo():
    print("markerless2d_mpyolo.py - Enhanced with scaling and safety margins")
    # Print the script version and directory
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Update model path
    model_name = "yolo11x.pt"

    # Correct path to vaila/vaila/models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vaila_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(vaila_dir, "vaila", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_name)

    # Get parameters before downloading the model
    root = tk.Tk()
    root.withdraw()

    params = get_parameters_dialog()
    if not params:
        print("No parameters set. Exiting...")
        return

    # Download the model only after confirming that we will use it
    if not os.path.exists(model_path):
        model_path = download_model(model_name)
    else:
        print(f"Found model at: {model_path}. Using the local file.")

    # Select video file
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
    )

    if not video_path:
        print("No video file selected. Exiting...")
        return

    print(f"Selected video: {video_path}")

    # Create output directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(
        os.path.dirname(video_path), f"{video_name}_enhanced_analysis_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize YOLO model
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)

    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Process video with enhanced pipeline
    process_video_enhanced(video_path, output_dir, model, params, mp_pose, mp_drawing)

    print(f"\nProcessing complete! All results saved to: {output_dir}")

    # Try to open output folder
    try:
        if os.name == "nt":  # Windows
            os.startfile(output_dir)
        elif os.name == "posix":  # macOS and Linux
            subprocess.run(["xdg-open", output_dir])
    except Exception as e:
        print(f"Could not open the output directory: {e}")

    root.destroy()


def write_nan_row(csv_path, frame_idx, num_landmarks):
    """Write a row with empty values for frames without detections"""
    # Each landmark has 4 values (x, y, z, visibility)
    # Use empty values instead of "NaN" to save space
    empty_values = ",".join([""] * (num_landmarks * 4))
    with open(csv_path, "a") as f:
        f.write(f"{frame_idx}," + empty_values + "\n")


if __name__ == "__main__":
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")

    # Get parameters first
    params = get_parameters_dialog()
    if not params:
        print("No parameters set. Exiting...")
        exit()

    # Execute the selected processing mode
    selected_mode = params.get("selected_mode", "sequential")

    if selected_mode == "sequential":
        print("Starting Sequential Processing mode...")
        run_markerless2d_mpyolo()
    else:  # multithreaded
        print("Starting Multithreaded Tracking mode...")
        run_multithreaded_tracking()
