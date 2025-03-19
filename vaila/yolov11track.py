"""
Project: vailá
Script: yolov11track.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 16 February 2025
Update Date: 18 March 2025
Version: 0.02

Description:
    This script performs object detection and tracking on video files using the YOLO model.
    It integrates multiple features, including:
      - Object detection and tracking using the Ultralytics YOLO library.
      - A graphical interface (Tkinter) for dynamic parameter configuration.
      - Video processing with OpenCV, including drawing bounding boxes and overlaying tracking data.
      - Generation of CSV files containing frame-by-frame tracking information per tracker ID.
      - Video conversion to more compatible formats using FFmpeg.

Usage:
    Run the script from the command line by passing the path to a video file as an argument:
        python yolov11track.py /path/to/video.mp4

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

Notes:
    - Ensure that all dependencies are installed.
    - Since the script uses a graphical interface (Tkinter) for model selection and configuration, a GUI-enabled environment is required.

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
from tkinter import filedialog, messagebox, simpledialog
from boxmot import StrongSort, ByteTrack, OcSort, DeepOcSort, HybridSort
from pathlib import Path
import subprocess
import re
import colorsys

# Configuração para evitar conflitos de biblioteca
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)  # Limita o número de threads para evitar conflitos


def initialize_csv(output_dir, label, tracker_id, total_frames):
    """Inicializa um arquivo CSV para um ID de rastreador e rótulo específicos."""
    csv_file = os.path.join(output_dir, f"{label}_id{tracker_id}.csv")
    if not os.path.exists(csv_file):
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
                    ]
                )
    return csv_file


def update_csv(
    csv_file, frame_idx, tracker_id, label, x_min, y_min, x_max, y_max, conf
):
    """Atualiza o arquivo CSV com dados de detecção para o frame específico."""
    rows = []
    with open(csv_file, mode="r", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Atualiza a linha específica do frame
    for i, row in enumerate(rows):
        if i > 0 and int(row[0]) == frame_idx:  # Pula o cabeçalho
            rows[i] = [frame_idx, tracker_id, label, x_min, y_min, x_max, y_max, conf]
            break

    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


class TrackerConfigDialog(tk.simpledialog.Dialog):
    def __init__(self, parent, title=None):
        self.tooltip = None  # Initialize tooltip as None
        super().__init__(parent, title)

    def body(self, master):
        # Confidence
        tk.Label(master, text="Confidence threshold:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.conf = tk.Entry(master)
        self.conf.insert(0, "0.25")
        self.conf.grid(row=0, column=1, padx=5, pady=5)
        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=0, column=2, padx=5, pady=5)
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
        tk.Label(master, text="IoU threshold:").grid(row=1, column=0, padx=5, pady=5)
        self.iou = tk.Entry(master)
        self.iou.insert(0, "0.7")
        self.iou.grid(row=1, column=1, padx=5, pady=5)
        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=1, column=2, padx=5, pady=5)
        iou_tooltip = (
            "Intersection over Union threshold (0-1):\n"
            "Controls how much overlap is needed to merge multiple detections.\n"
            "Higher values (e.g., 0.9): Very strict matching\n"
            "Lower values (e.g., 0.5): More lenient matching\n"
            "Recommended: 0.7 for most cases"
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, iou_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        # Device
        tk.Label(master, text="Device (cuda/cpu):").grid(
            row=2, column=0, padx=5, pady=5
        )
        self.device = tk.Entry(master)
        self.device.insert(0, "cpu")
        self.device.grid(row=2, column=1, padx=5, pady=5)
        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=2, column=2, padx=5, pady=5)
        device_tooltip = (
            "Processing device options:\n"
            "'cuda' - Use GPU (NVIDIA only)\n"
            "        Much faster processing (10-20x)\n"
            "        Requires NVIDIA GPU and CUDA\n\n"
            "'cpu'  - Use CPU\n"
            "        Works on all computers\n"
            "        Slower but universally compatible"
        )
        help_text.bind("<Enter>", lambda e: self.show_help(e, device_tooltip))
        help_text.bind("<Leave>", self.hide_help)

        # Video stride
        tk.Label(master, text="Video stride:").grid(row=3, column=0, padx=5, pady=5)
        self.vid_stride = tk.Entry(master)
        self.vid_stride.insert(0, "1")
        self.vid_stride.grid(row=3, column=1, padx=5, pady=5)
        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=3, column=2, padx=5, pady=5)
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

    def hide_help(self, event=None):
        if self.tooltip is not None:
            self.tooltip.destroy()
            self.tooltip = None

    def validate(self):
        try:
            self.result = {
                "conf": float(self.conf.get()),
                "iou": float(self.iou.get()),
                "device": self.device.get(),
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

        self.listbox = tk.Listbox(master, width=50, height=15)
        for model, desc in models:
            self.listbox.insert(tk.END, f"{model} - {desc}")
        self.listbox.pack(padx=5, pady=5)

        return self.listbox

    def validate(self):
        if not self.listbox.curselection():
            messagebox.showwarning("Warning", "Please select a model")
            return False
        return True

    def apply(self):
        selection = self.listbox.get(self.listbox.curselection())
        self.result = selection.split(" - ")[0]


class TrackerSelectorDialog(tk.simpledialog.Dialog):
    def body(self, master):
        trackers = [
            ("bytetrack", "ByteTrack - YOLO's default tracker"),
            ("botsort", "BoTSORT - YOLO's alternative tracker"),
            ("strongsort", "StrongSort - Best precision with ReID"),
            ("ocsort", "OCSORT - Simple and efficient"),
            ("deepocsort", "DeepOCSORT - Enhanced OCSORT"),
            ("hybridsort", "HybridSORT - Hybrid method"),
        ]

        self.listbox = tk.Listbox(master, width=50, height=10)
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


def get_color_for_id(tracker_id):
    """Generate a distinct color for each tracker ID using a combination of techniques.

    This function uses a more sophisticated approach to ensure colors are visually distinct:
    1. Uses golden ratio conjugate for better distribution across color space
    2. Varies saturation for even/odd IDs
    3. Uses a predefined list of distinct colors for the first N IDs
    4. Ensures high contrast against common backgrounds
    """
    # Predefined distinct colors for the first 10 IDs (in BGR)
    distinct_colors = [
        (255, 0, 0),  # Blue
        (0, 0, 255),  # Red
        (0, 255, 0),  # Green
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Purple
        (0, 165, 255),  # Orange
        (255, 255, 255),  # White
        (0, 0, 0),  # Black
    ]

    # Use predefined colors for first few IDs
    if 0 <= tracker_id < len(distinct_colors):
        return distinct_colors[tracker_id]

    # For higher IDs, use golden ratio method for better distribution
    golden_ratio_conjugate = 0.618033988749895
    h = (tracker_id * golden_ratio_conjugate) % 1.0

    # Vary saturation and value based on ID to create more distinction
    s = 0.85 if tracker_id % 2 == 0 else 0.95
    v = 0.95 if tracker_id % 4 < 2 else 0.85

    rgb = colorsys.hsv_to_rgb(h, s, v)
    # Convert to BGR (OpenCV format) with values 0-255
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def run_yolov11track():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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

    # Build the full path for the model
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

    # Initialize the YOLO model
    model = YOLO(model_path)

    # Process each video in the directory
    for video_file in os.listdir(video_dir):
        if video_file.endswith((".mp4", ".avi", ".mov")):
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

            results = model.track(
                source=video_path,
                conf=config["conf"],
                iou=config["iou"],
                device=config["device"],
                vid_stride=config["vid_stride"],
                save=False,
                stream=True,
                persist=True,
                classes=[0, 32],
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
