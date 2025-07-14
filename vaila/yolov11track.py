"""
Project: vailá
Script: yolov11track.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 February 2025
Update Date: 09 July 2025
Version: 0.2.0

Description:
    This script performs object detection and tracking on video files using the YOLO model v11.
    It integrates multiple features, including:
      - Object detection and tracking using the Ultralytics YOLO library.
      - A graphical interface (Tkinter) for dynamic parameter configuration.
      - Video processing with OpenCV, including drawing bounding boxes and overlaying tracking data.
      - Generation of CSV files containing frame-by-frame tracking information per tracker ID.
      - Video conversion to more compatible formats using FFmpeg.

Usage:
    Run the script from the command line by passing the path to a video file as an argument:
            python yolov11track.py

Requirements:
    - Python 3.x
    - OpenCV
    - PyTorch
    - Ultralytics (YOLO)
    - Tkinter (for GUI operations)
    - FFmpeg (for video conversion)
    - Additional dependencies as imported (numpy, csv, etc.)

License:
    This project is licensed under the terms of the MIT License (or another applicable license).

Change History:
    - 2023-10: Initial version implemented, integrating detection and tracking with various configurable options.
    - 2025-03: Added color-coding for each tracker ID, improved GUI, and added more detailed help text.
    - 2025-03: Added support for multiple models and trackers.
    - 2025-03: Added support for video conversion to more compatible formats using FFmpeg.

Notes:
    - Ensure that all dependencies are installed.
    - Since the script uses a graphical interface (Tkinter) for model selection and configuration, a GUI-enabled environment is required.
    - If the ReID model is not found, the script will download it from the internet.

"""

import os
from rich import print
import sys
import csv
import cv2
import torch
import numpy as np
import datetime
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess
import re
import colorsys
import pkg_resources
import glob
import pandas as pd
from boxmot import BotSort


print(f"Running script: {os.path.basename(__file__)}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
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
    csv_file = os.path.join(output_dir, f"{label}_id{tracker_id}.csv")
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


def update_csv(
    csv_file, frame_idx, tracker_id, label, x_min, y_min, x_max, y_max, conf
):
    """Updates the CSV file with detection data for the specific frame."""
    rows = []
    with open(csv_file, mode="r", newline="") as file:
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
    """Get detailed hardware information for GPU/CPU detection"""
    info = []
    info.append(f"Python version: {sys.version}")
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
        info.append(f"CPU cores: {os.cpu_count()}")
    
    return "\n".join(info)

def detect_optimal_device():
    """Detect and return the optimal device for processing"""
    # Default to CPU for better compatibility
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"Multiple GPUs detected ({gpu_count}). GPU 0 available for selection.")
        
        # Clear cache for optimal performance
        torch.cuda.empty_cache()
        # Still return "cpu" as default for better compatibility
        return "cpu"
    else:
        return "cpu"

def validate_device_choice(user_device):
    """Validate user device choice and provide feedback"""
    if user_device.lower() == "cuda":
        if torch.cuda.is_available():
            return True, "GPU (CUDA) - High performance"
        else:
            return False, "GPU (CUDA) requested but not available. Using CPU instead."
    elif user_device.lower() == "cpu":
        return True, "CPU - Universal compatibility (default)"
    else:
        return False, f"Invalid device: {user_device}. Using CPU instead."


class TrackerConfigDialog(tk.simpledialog.Dialog):
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
        
        self.device_var = tk.StringVar(value="cpu")  # Default to CPU
        device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                   values=["cpu", "cuda"], state="readonly", width=10)
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
                    self.device_status.config(text="✓ " + message, fg="blue")  # Blue for CPU
                else:
                    self.device_status.config(text="✓ " + message, fg="green")  # Green for GPU
            else:
                self.device_status.config(text="⚠ " + message, fg="orange")
        
        self.device_var.trace("w", update_device_status)
        update_device_status()  # Initial update
        
        # Help text for device selection
        help_text = tk.Label(device_frame, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=0, column=3, padx=5, pady=5)
        device_tooltip = (
            "Processing device options:\n"
            "'cpu'  - Use CPU (DEFAULT)\n"
            "        Universal compatibility\n"
            "        Works on all computers\n"
            "        Slower but reliable\n\n"
            "'cuda' - Use GPU (NVIDIA only)\n"
            "        Much faster processing (10-20x)\n"
            "        Requires NVIDIA GPU and CUDA\n"
            "        May have compatibility issues\n\n"
            "CPU is recommended for most users."
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, device_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        # Confidence
        tk.Label(master, text="Confidence threshold:").grid(
            row=2, column=0, padx=5, pady=5
        )
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

    def hide_help(self):
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None

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


class ModelSelectorDialog(tk.simpledialog.Dialog):
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
            ("yolo11n.pt", "Detection - Nano (fastest)"),
            ("yolo11s.pt", "Detection - Small"),
            ("yolo11m.pt", "Detection - Medium"),
            ("yolo11l.pt", "Detection - Large"),
            ("yolo11x.pt", "Detection - XLarge (most accurate)"),
            # Pose Estimation explanation: https://docs.ultralytics.com/tasks/pose/
            ("yolo11n-pose.pt", "Pose - Nano (fastest)"),
            ("yolo11s-pose.pt", "Pose - Small"),
            ("yolo11m-pose.pt", "Pose - Medium"),
            ("yolo11l-pose.pt", "Pose - Large"),
            ("yolo11x-pose.pt", "Pose - XLarge (most accurate)"),
            # Segmentation explanation: https://docs.ultralytics.com/tasks/segment/
            ("yolo11n-seg.pt", "Segmentation - Nano"),
            ("yolo11s-seg.pt", "Segmentation - Small"),
            ("yolo11m-seg.pt", "Segmentation - Medium"),
            ("yolo11l-seg.pt", "Segmentation - Large"),
            ("yolo11x-seg.pt", "Segmentation - XLarge"),
            # OBB (Oriented Bounding Box) explanation: https://docs.ultralytics.com/tasks/obb/
            ("yolo11n-obb.pt", "OBB - Nano"),
            ("yolo11s-obb.pt", "OBB - Small"),
            ("yolo11m-obb.pt", "OBB - Medium"),
            ("yolo11l-obb.pt", "OBB - Large"),
            ("yolo11x-obb.pt", "OBB - XLarge"),
        ]

        # Create listbox with scrollbar for pre-trained models
        listbox_frame = tk.Frame(pretrained_frame)
        listbox_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.pretrained_listbox = tk.Listbox(listbox_frame, width=50, height=12)
        scrollbar = tk.Scrollbar(listbox_frame, orient="vertical", command=self.pretrained_listbox.yview)
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
            fg="gray"
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
                ("All files", "*.*")
            ]
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


class TrackerSelectorDialog(tk.simpledialog.Dialog):
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


class ReidModelSelectorDialog(tk.simpledialog.Dialog):
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


class ClassSelectorDialog(tk.simpledialog.Dialog):
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
        tk.Label(classes_frame, text="Available classes:").grid(
            row=0, column=0, sticky="w"
        )

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
            messagebox.showwarning(
                "Warning", "Invalid format. Use numbers separated by commas."
            )
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


def process_video(input_file: str):
    # Normalize the file path (resolves slashes and separators appropriately)
    input_file = os.path.normpath(input_file)

    # Check if the file actually exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    # Separate directory and base file name, and standardize the base name if necessary
    dir_name = os.path.dirname(input_file)
    base_name = os.path.basename(input_file)
    standardized_name = standardize_filename(base_name)

    # If the name has been altered, rename the file temporarily
    if base_name != standardized_name:
        new_file_path = os.path.join(dir_name, standardized_name)
        os.rename(input_file, new_file_path)
        input_file = new_file_path
        print(f"Standardized file name: {input_file}")

    # On Windows, ffmpeg may misinterpret backslashes. Replace them with forward slashes.
    ffmpeg_input = input_file.replace("\\", "/")

    # Build the command as a list of arguments so that spaces don't break the command.
    cmd = [
        "ffmpeg",
        "-i",
        ffmpeg_input,
        "-filter_complex",
        "your_filter_here",
        "output.mp4",
    ]

    print("Executing command:", " ".join(cmd))
    # Run the command without using the shell
    subprocess.run(cmd)


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
        additional_colors.append(
            (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        )

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
    detection_csv_files = glob.glob(os.path.join(output_dir, "*_id*.csv"))

    if not detection_csv_files:
        print(f"No detection tracking files found in {output_dir}")
        return None

    print(f"Found {len(detection_csv_files)} detection tracking files")

    # First, get all unique frames and object IDs
    all_frames = set()
    object_ids = set()
    object_colors = {}  # Store RGB values for each object ID

    # Read all files to get frame and ID information
    for csv_file in detection_csv_files:
        try:
            df = pd.read_csv(csv_file)
            filename = os.path.basename(csv_file)
            # Extract object ID from filename (e.g., "person_id0.csv" -> 0)
            object_id = int(filename.split("_id")[1].split(".")[0])
            object_ids.add(object_id)

            # Get frames where this object appears
            valid_frames = df.dropna(subset=["X_min", "X_max", "Y_max"])[
                "Frame"
            ].unique()
            all_frames.update(valid_frames)

            # Store color information for this object
            if not df.empty:
                r = int(df["Color_R"].iloc[0])
                g = int(df["Color_G"].iloc[0])
                b = int(df["Color_B"].iloc[0])
                # Combine RGB values into a single integer
                rgb_int = (r << 16) + (g << 8) + b
                object_colors[object_id] = rgb_int

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    # Create a dictionary to store all data
    all_data = {}
    object_ids = sorted(list(object_ids))  # Sort IDs for consistent column order

    # Initialize the data structure for all frames
    for frame in all_frames:
        all_data[frame] = {"Frame": frame}
        for oid in object_ids:
            all_data[frame].update(
                {
                    f"ID_{oid}": oid,
                    f"X_{oid}": "",  # Empty string for missing values
                    f"Y_{oid}": "",
                    f"RGB_{oid}": object_colors.get(oid, ""),
                }
            )

    # Now fill in the position data from each file
    for csv_file in detection_csv_files:
        try:
            df = pd.read_csv(csv_file)
            object_id = int(os.path.basename(csv_file).split("_id")[1].split(".")[0])

            # Process only rows with valid detection data
            valid_data = df.dropna(subset=["X_min", "X_max", "Y_max"])

            for _, row in valid_data.iterrows():
                frame = int(row["Frame"])
                if frame in all_data:
                    x_center = int((float(row["X_min"]) + float(row["X_max"])) / 2)
                    y_point = int(float(row["Y_max"]))

                    all_data[frame][f"X_{object_id}"] = x_center
                    all_data[frame][f"Y_{object_id}"] = y_point

        except Exception as e:
            print(f"Error processing positions from {csv_file}: {e}")
            continue

    # Convert to DataFrame
    df_combined = pd.DataFrame(list(all_data.values()))

    # Sort by frame number
    df_combined = df_combined.sort_values("Frame")

    # Organize columns in the desired order
    columns = ["Frame"]
    for oid in object_ids:
        columns.extend([f"ID_{oid}", f"X_{oid}", f"Y_{oid}", f"RGB_{oid}"])

    df_combined = df_combined[columns]

    # Save to CSV with the new name
    output_file = os.path.join(output_dir, "all_id_detection.csv")
    df_combined.to_csv(output_file, index=False)
    print(f"Combined detection tracking data saved to: {output_file}")
    return output_file


def run_yolov11track():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
    if os.path.isabs(model_name) or model_name.startswith('./') or model_name.startswith('../'):
        # Custom model - use the path directly
        model_path = model_name
        print(f"Using custom model: {model_path}")
        
        # Validate custom model file
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Custom model file not found: {model_path}")
            return
    else:
        # Pre-trained model - build the path in models directory
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
    else:
        print(f"CPU cores: {os.cpu_count()}")

    # Initialize the YOLO model
    try:
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

    # Before initializing BotSort
    print("Checking if we are ready to initialize BotSort...")

    # Initialize BotSort
    print(f"Initializing BotSort")

    # Try create reid_weights if osnet_x0_25_msmt17.pt does not exist, download it from the internet https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    reid_weights_path = os.path.join(models_dir, "osnet_x0_25_msmt17.pt")
    if not os.path.exists(reid_weights_path):
        print("Downloading ReID model...")
        try:
            import requests

            url = "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt"
            response = requests.get(url)
            with open(reid_weights_path, "wb") as f:
                f.write(response.content)
            print(f"ReID model downloaded successfully to {reid_weights_path}")
        except Exception as e:
            print(f"Failed to download ReID model: {str(e)}")
            messagebox.showerror("Error", f"Failed to download ReID model: {str(e)}")
            return

    reid_weights = Path(models_dir) / "osnet_x0_25_msmt17.pt"

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    half = True  # Use reduced precision
    if not reid_weights.exists():
        print(f"ReID model not found at: {reid_weights}")
    try:
        tracker = BotSort(reid_weights=reid_weights, device=device, half=half)
        print("BotSort initialized successfully.")
    except Exception as e:
        print(f"Error initializing BotSort: {e}")

    # Process each video in the directory
    for video_file in os.listdir(video_dir):
        if video_file.endswith(
            (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
        ):
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Create a subdirectory for this specific video
            output_dir = os.path.join(main_output_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            # Read the dimensions of the original video
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            # Specify the output path for the processed video
            out_video_path = os.path.join(output_dir, f"processed_{video_name}.mp4")

            # Use the 'mp4v' codec which is more stable
            writer = cv2.VideoWriter(
                out_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                print(f"Error creating video file: {out_video_path}")
                continue

            # Process the frames
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0

            print(f"Tracking with {tracker_name} (YOLO built-in tracker)")

            # Find the tracker config file
            tracker_config = None

            # Option 1: Look for the tracker config in the Ultralytics package
            ultralytics_path = pkg_resources.resource_filename(
                "ultralytics", "cfg/trackers"
            )
            tracker_yaml = f"{tracker_name}.yaml"
            yaml_path = os.path.join(ultralytics_path, tracker_yaml)

            if os.path.exists(yaml_path):
                tracker_config = yaml_path
            else:
                # Option 2: Create local copy of tracker config in our models directory
                trackers_dir = os.path.join(models_dir, "trackers")
                os.makedirs(trackers_dir, exist_ok=True)
                local_yaml = os.path.join(trackers_dir, tracker_yaml)

                # Create basic tracker config file if it doesn't exist
                if not os.path.exists(local_yaml):
                    with open(local_yaml, "w") as f:
                        f.write(f"tracker_type: {tracker_name}\n")
                        f.write("track_high_thresh: 0.5\n")
                        f.write("track_low_thresh: 0.1\n")
                        f.write("new_track_thresh: 0.6\n")
                        f.write("track_buffer: 30\n")
                        f.write("match_thresh: 0.8\n")

                tracker_config = local_yaml

            print(f"Using tracker config: {tracker_config}")

            # Use YOLO's tracker with the config file
            results = model.track(
                source=video_path,
                conf=config["conf"],
                iou=config["iou"],
                device=config["device"],
                vid_stride=config["vid_stride"],
                save=False,
                stream=True,
                persist=True,
                tracker=tracker_config,  # Use the config file path
                classes=target_classes,  # Usa as classes selecionadas pelo usuário
            )

            tracker_csv_files = {}

            for result in results:
                frame = result.orig_img

                # For OBB models result.boxes might be None.
                # Try to use result.obbs if available:
                boxes = (
                    result.boxes
                    if result.boxes is not None
                    else getattr(result, "obbs", None)
                )

                if boxes is None:
                    print("No bounding boxes found in this frame.")
                    writer.write(frame)
                    frame_idx += 1
                    continue

                for box in boxes:
                    # Extract coordinates (this assumes the obb objects have a similar `xyxy` property)
                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                    conf = box.conf[0].item()
                    tracker_id = int(box.id[0]) if box.id is not None else -1
                    class_id = int(box.cls[0].item()) if box.cls is not None else -1
                    label = (
                        model.names[class_id] if class_id in model.names else "unknown"
                    )

                    # Get a unique color for this tracker ID
                    color = (
                        get_color_for_id(tracker_id) if tracker_id >= 0 else (0, 255, 0)
                    )

                    label_text = f"{label}_id{tracker_id}, Conf: {conf:.2f}"
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(
                        frame,
                        label_text,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    # Initialize and update the CSV for each tracker id
                    key = (tracker_id, label)
                    if key not in tracker_csv_files:
                        tracker_csv_files[key] = initialize_csv(
                            output_dir,
                            label,
                            tracker_id,
                            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
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

                # Write the processed frame to the output video
                writer.write(frame)
                frame_idx += 1

            # Release resources
            cap.release()
            writer.release()

            # Convert the video to a more compatible format using ffmpeg
            try:
                temp_path = out_video_path.replace(".mp4", "_temp.mp4")
                os.rename(out_video_path, temp_path)
                os.system(
                    f"ffmpeg -i {temp_path} -c:v libx264 -preset medium -crf 23 {out_video_path}"
                )
                os.remove(temp_path)
            except Exception as e:
                print(f"Error in video conversion: {str(e)}")

            print(
                f"Processing completed for {video_file}. Results saved in '{output_dir}'."
            )

            # Create combined detection CSV after processing each video
            combined_csv = create_combined_detection_csv(output_dir)
            if combined_csv:
                print(f"Combined detection tracking file created: {combined_csv}")

    root.destroy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolov11track.py [video_file]")
        sys.exit(1)

    video_file = sys.argv[1]
    try:
        process_video(video_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
