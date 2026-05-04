"""
Project: vailá Multimodal Toolbox
Script: markerless2d_yolo26.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 4 May 2026
Update Date: 4 May 2026
Version: 0.1.0

Description:
This script performs standalone 2D markerless analysis using YOLOv26 (YOLOv11-Pose)
for 17 keypoint detection. It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files.

The 17 keypoints follow the COCO format:
nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
left_knee, right_knee, left_ankle, right_ankle.

Usage:
Run the script to open a graphical interface for selecting the input directory
containing video files and specifying the YOLO configuration parameters.
``uv run python vaila/markerless2d_yolo26.py``

License:
This program is licensed under the GNU Affero General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
"""

import csv
import datetime
import glob
import os
import platform
import shutil
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Standard 17 COCO Keypoints for YOLO Pose
KEYPOINT_NAMES = [
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


def ensure_yolo_pose_weights(model_filename: str) -> Path:
    """Resolve ``.pt`` under ``vaila/models``. Bare ``YOLO(name.pt)`` saves to process CWD (often repo root)."""
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_filename

    if model_path.exists():
        return model_path

    print(f"Downloading {model_filename} to {model_path}...")
    try:
        model = YOLO(str(model_path))
        source_path = getattr(model, "ckpt_path", None)

        if source_path and os.path.exists(source_path):
            if Path(source_path).resolve() != model_path.resolve():
                shutil.copy2(source_path, str(model_path))
            try:
                abs_src = os.path.abspath(source_path)
                if (
                    os.path.basename(abs_src) == model_filename
                    and abs_src.startswith(os.getcwd())
                    and Path(abs_src).resolve() != model_path.resolve()
                ):
                    os.remove(abs_src)
            except OSError:
                pass
            return model_path

        if model_path.exists():
            return model_path
        print(f"YOLO could not place weights at {model_path} (ckpt_path={source_path!r}).")
    except Exception as e:
        print(f"Ultralytics download path failed: {e}")

    try:
        import requests

        if "yolo11" in model_filename.lower() or "yolov11" in model_filename.lower():
            version_tag = "v11.0.0"
        elif "yolo8" in model_filename.lower() or "yolov8" in model_filename.lower():
            version_tag = "v8.0.0"
        else:
            version_tag = "v0.0.0"

        url_model_name = (
            model_filename.replace("yolov", "yolo")
            if "yolov" in model_filename.lower()
            else model_filename
        )
        possible_urls = [
            f"https://github.com/ultralytics/assets/releases/download/{version_tag}/{url_model_name}",
            f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{url_model_name}",
        ]

        response = None
        for attempt_url in possible_urls:
            try:
                response = requests.get(attempt_url, stream=True, timeout=60)
                if response.status_code == 200:
                    break
                response = None
            except Exception:
                continue

        if response is None or response.status_code != 200:
            raise RuntimeError("HTTP fallback: no asset URL returned success")

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with model_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024 * 1024) == 0:
                        print(f"  Downloaded: {(downloaded / total_size) * 100:.1f}%")

        print(f"[OK] Saved {model_filename} to {model_path}")
        return model_path

    except Exception as e2:
        print(f"HTTP fallback failed: {e2}")

    home = Path.home()
    for cache_path in (
        home / ".ultralytics" / "weights" / model_filename,
        home / ".cache" / "ultralytics" / model_filename,
    ):
        if cache_path.exists():
            shutil.copy2(str(cache_path), str(model_path))
            print(f"[OK] Copied from Ultralytics cache to {model_path}")
            return model_path

    stray = Path.cwd() / model_filename
    if stray.exists():
        shutil.copy2(str(stray), str(model_path))
        print(f"[OK] Moved stray weights from CWD to {model_path}")
        return model_path

    raise RuntimeError(
        f"Could not resolve YOLO weights {model_filename!r}. "
        f"Place the file under {models_dir} or check connectivity."
    )


class YoloConfigDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("YOLOv26 Pose Configuration")
        tk.Label(master, text="YOLO Pose Model:").grid(row=0, sticky="w", padx=5, pady=5)
        tk.Label(master, text="Confidence Threshold:").grid(row=1, sticky="w", padx=5, pady=5)

        self.model_var = tk.StringVar(value="yolo11x-pose.pt")
        models = [
            "yolo11n-pose.pt",
            "yolo11s-pose.pt",
            "yolo11m-pose.pt",
            "yolo11l-pose.pt",
            "yolo11x-pose.pt",
            "yolo26n-pose.pt",
            "yolo26s-pose.pt",
            "yolo26m-pose.pt",
            "yolo26l-pose.pt",
            "yolo26x-pose.pt",
        ]
        self.model_combo = ttk.Combobox(
            master, textvariable=self.model_var, values=models, state="readonly", width=25
        )
        self.model_combo.grid(row=0, column=1, padx=5, pady=5)

        self.conf_entry = tk.Entry(master, width=28)
        self.conf_entry.insert(0, "0.5")
        self.conf_entry.grid(row=1, column=1, padx=5, pady=5)

        return self.model_combo

    def apply(self):
        try:
            self.result = {
                "model": self.model_var.get(),
                "confidence": float(self.conf_entry.get()),
            }
        except ValueError:
            messagebox.showerror("Error", "Confidence must be a number between 0 and 1")
            self.result = None


def draw_yolo_pose(frame, keypoints, width, height):
    """Draw 17 keypoints and skeleton on frame"""
    # Define skeleton connections for COCO 17
    skeleton = [
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ]

    # Draw lines
    for p1, p2 in skeleton:
        if p1 < len(keypoints) and p2 < len(keypoints):
            pt1 = keypoints[p1]
            pt2 = keypoints[p2]
            if not np.isnan(pt1[0]) and not np.isnan(pt2[0]):
                cv2.line(
                    frame,
                    (int(pt1[0] * width), int(pt1[1] * height)),
                    (int(pt2[0] * width), int(pt2[1] * height)),
                    (0, 255, 0),
                    2,
                )

    # Draw points
    for pt in keypoints:
        if not np.isnan(pt[0]):
            cv2.circle(frame, (int(pt[0] * width), int(pt[1] * height)), 4, (0, 0, 255), -1)

    return frame


def process_video(video_path, config):
    print(f"\nProcessing: {video_path}")
    video_path_obj = Path(video_path)
    output_dir = video_path_obj.parent
    base_name = video_path_obj.stem

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"processed_yolo26_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    norm_csv_path = run_dir / f"{base_name}_yolo26_norm.csv"
    pixel_csv_path = run_dir / f"{base_name}_yolo26_pixel.csv"
    video_out_path = run_dir / f"{base_name}_yolo26.mp4"
    log_path = run_dir / "log_info.txt"

    try:
        weights_path = ensure_yolo_pose_weights(config["model"])
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    print(f"Loading model: {weights_path}")
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    out = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))

    with (
        open(norm_csv_path, "w", newline="") as norm_file,
        open(pixel_csv_path, "w", newline="") as pixel_file,
    ):
        header = ["frame"]
        for name in KEYPOINT_NAMES:
            header.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])

        norm_writer = csv.writer(norm_file)
        pixel_writer = csv.writer(pixel_file)
        norm_writer.writerow(header)
        pixel_writer.writerow(header)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            device = "mps"

        print(f"Using device: {device}")
        model.to(device)

        start_time = time.time()
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=config["confidence"], verbose=False)

            norm_row = [float(frame_idx)] + [np.nan] * (17 * 3)
            pixel_row = [float(frame_idx)] + [np.nan] * (17 * 3)
            kpts_norm_for_draw = [[np.nan, np.nan, np.nan]] * 17

            if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                # Take the detection with highest confidence or first detection
                kpts = results[0].keypoints.data[0].cpu().numpy()  # Shape (17, 3)

                norm_row = [float(frame_idx)]
                pixel_row = [float(frame_idx)]
                kpts_norm_for_draw = []

                for i in range(17):
                    # Default to NaNs if keypoint is missing or low confidence
                    kpt_norm = [np.nan, np.nan, np.nan]
                    kpt_pixel = [np.nan, np.nan, np.nan]

                    if i < len(kpts):
                        x, y, conf = kpts[i]
                        if conf > config["confidence"]:
                            kpt_norm = [x / width, y / height, conf]
                            kpt_pixel = [x, y, conf]

                    norm_row.extend(kpt_norm)
                    pixel_row.extend(kpt_pixel)
                    kpts_norm_for_draw.append(kpt_norm)

                frame = draw_yolo_pose(frame, kpts_norm_for_draw, width, height)

            norm_writer.writerow(norm_row)
            pixel_writer.writerow(pixel_row)
            out.write(frame)

            frame_idx += 1
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1f}%)", end="\r")

        end_time = time.time()
        duration = end_time - start_time
        print(f"\nFinished processing {frame_idx} frames in {duration:.1f}s.")

    cap.release()
    out.release()

    # Write log file
    with open(log_path, "w") as f:
        f.write("vailá Markerless 2D YOLOv26 Analysis Log\n")
        f.write("=" * 40 + "\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Video: {video_path_obj.name}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"FPS: {fps}\n")
        f.write(f"Total Frames: {total_frames}\n")
        f.write(f"Processing Duration: {duration:.1f}s\n")
        f.write(f"Average FPS: {frame_idx / duration:.1f}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model: {config['model']}\n")
        f.write(f"  Confidence: {config['confidence']}\n")
        f.write(f"  Device: {device}\n")

    print(f"Results saved to {run_dir}")


def run_markerless2d_yolo26():
    root = tk.Tk()
    root.withdraw()

    dialog = YoloConfigDialog(root)
    if not hasattr(dialog, "result") or not dialog.result:
        print("Configuration cancelled.")
        return

    config = dialog.result

    input_dir = filedialog.askdirectory(title="Select Directory with Videos")
    if not input_dir:
        print("No directory selected.")
        return

    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not video_files:
        messagebox.showinfo("Info", "No videos found in the selected directory.")
        return

    print(f"Found {len(video_files)} videos to process.")
    for video in video_files:
        process_video(video, config)

    messagebox.showinfo("Success", f"Processed {len(video_files)} videos successfully.")


if __name__ == "__main__":
    run_markerless2d_yolo26()
