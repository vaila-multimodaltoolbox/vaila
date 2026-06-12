"""
Project: vailá Multimodal Toolbox
Script: yolotrain.py - Simplified AnyLabeling YOLO Trainer

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 May 2025
Update Date: 12 June 2026
Version: 0.3.53

Description:
    YOLO training interface for vailá/getpixelvideo and YOLO-format datasets.
    Creates training datasets from getpixelvideo pixel CSVs and trains Ultralytics models.

Usage:
    Run the script from the command line:
        python yolotrain.py

Requirements:
    - Python 3.x
    - Ultralytics YOLO
    - Tkinter (for GUI operations)

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.3.53: Dataset build now runs in a worker thread with sequential video
               decode (no more shuffled cap.set seeks that froze the GUI on
               large CSVs); train/val splits no longer duplicate a train row
               into val; _resolve_yaml_path uses removeprefix("./") so
               "../images/train" and list-valued train/val entries resolve
               correctly.
    - v0.3.51: Added getpixelvideo/sam_points_georeid CSV to YOLO tracking dataset builder
    - v0.0.4: Simplified interface, AnyLabeling-focused, minimal YAML generation
    - v0.0.3: Added support for AnyLabeling data, improved UI
    - v0.0.2: Added validation and threading support
    - v0.0.1: First version
"""

import contextlib
import os
import pathlib
import random
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, simpledialog

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

# import datetime
# import re
# import webbrowser


# --- Class to redirect the console output to the Text Widget ---
class ConsoleRedirector:
    """Redirects stdout to both GUI and terminal."""

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout

    def write(self, text):
        # Write to GUI
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
        self.text_widget.update_idletasks()

        # Also write to original terminal
        self.original_stdout.write(text)
        self.original_stdout.flush()

    def flush(self):
        self.original_stdout.flush()


# --- Main Application Class ---
class YOLOTrainApp(tk.Tk):
    """YOLO Training Interface for vailá/getpixelvideo and YOLO datasets."""

    def __init__(self):
        super().__init__()
        self.title("YOLO Training - vailá getpixelvideo / YOLO")
        self.geometry("800x600")
        self.configure(padx=10, pady=10)

        # Available YOLO models (expanded list)
        self.available_models = [
            # YOLO26 Models (Latest - 2026)
            "yolo26x.pt",  # YOLO26 Extra Large
            "yolo26l.pt",  # YOLO26 Large
            "yolo26m.pt",  # YOLO26 Medium (Recommended default)
            "yolo26s.pt",  # YOLO26 Small
            "yolo26n.pt",  # YOLO26 Nano
            # YOLO11 Models
            "yolo11x.pt",  # YOLO11 Extra Large
            "yolo11l.pt",  # YOLO11 Large
            "yolo11m.pt",  # YOLO11 Medium
            "yolo11s.pt",  # YOLO11 Small
            "yolo11n.pt",  # YOLO11 Nano
            # YOLOv8 Models (Stable)
            "yolov8x.pt",  # YOLOv8 Extra Large
            "yolov8l.pt",  # YOLOv8 Large
            "yolov8m.pt",  # YOLOv8 Medium
            "yolov8s.pt",  # YOLOv8 Small
            "yolov8n.pt",  # YOLOv8 Nano
            # YOLOv9 Models
            "yolov9c.pt",  # YOLOv9 Compact
            "yolov9e.pt",  # YOLOv9 Extra Large
            "yolov9l.pt",  # YOLOv9 Large
            "yolov9m.pt",  # YOLOv9 Medium
            "yolov9s.pt",  # YOLOv9 Small
        ]

        # Tkinter variables
        self.dataset_path = tk.StringVar()
        self.yaml_path = tk.StringVar()
        self.base_model = tk.StringVar(value="yolo26m.pt")
        self.project_name = tk.StringVar(value="yolo_training")

        self.create_widgets()

    def create_widgets(self):
        """Creates simplified widgets."""
        # Configure grid
        self.grid_rowconfigure(9, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Title
        title_label = tk.Label(
            self,
            text="YOLO Training - vailá getpixelvideo / YOLO",
            font=("Arial", 16, "bold"),
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Help Button
        help_button = tk.Button(
            self,
            text="Dataset Guide",
            command=self.show_anylabeling_help,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10),
        )
        help_button.grid(row=1, column=0, columnspan=3, pady=5)

        # Model Location Info
        model_info_label = tk.Label(
            self,
            text="Models will be saved in: [dataset_folder]/runs/[run_name]/weights/",
            font=("Arial", 9),
            fg="#666666",
        )
        model_info_label.grid(row=2, column=0, columnspan=3, pady=2)

        # Dataset Path
        tk.Label(self, text="YOLO Dataset Folder:").grid(
            row=3, column=0, sticky="e", padx=5, pady=5
        )
        dataset_entry = tk.Entry(self, textvariable=self.dataset_path, width=50, state="readonly")
        dataset_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        tk.Button(self, text="Browse", command=self.browse_dataset).grid(
            row=3, column=2, padx=5, pady=5
        )

        self.csv_button = tk.Button(
            self,
            text="Create Dataset from getpixelvideo CSV",
            command=self.create_dataset_from_getpixelvideo_csv,
            bg="#673AB7",
            fg="white",
        )
        self.csv_button.grid(row=4, column=0, columnspan=3, pady=5, sticky="ew")

        # YAML Options Frame
        yaml_frame = tk.Frame(self)
        yaml_frame.grid(row=5, column=0, columnspan=3, pady=5, sticky="ew")

        tk.Label(yaml_frame, text="YAML Configuration:").grid(row=0, column=0, sticky="w", padx=5)

        # YAML Path Entry
        yaml_entry = tk.Entry(yaml_frame, textvariable=self.yaml_path, width=40, state="readonly")
        yaml_entry.grid(row=1, column=0, padx=5, pady=2, sticky="ew")

        # YAML Buttons Frame
        yaml_buttons_frame = tk.Frame(yaml_frame)
        yaml_buttons_frame.grid(row=1, column=1, padx=5, pady=2)

        # Browse YAML Button
        tk.Button(yaml_buttons_frame, text="Browse YAML", command=self.browse_yaml).grid(
            row=0, column=0, padx=2
        )

        # Create YAML Button
        tk.Button(yaml_buttons_frame, text="Create New YAML", command=self.create_new_yaml).grid(
            row=0, column=1, padx=2
        )

        # Start Training Button
        self.start_button = tk.Button(
            self,
            text="Start Training",
            command=self.start_training_thread,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
        )
        self.start_button.grid(row=6, column=0, columnspan=3, pady=10, sticky="ew")

        # Console
        tk.Label(self, text="Training Output:").grid(
            row=7, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )
        self.console = scrolledtext.ScrolledText(self, height=15, wrap="word", font=("Consolas", 9))
        self.console.grid(row=8, column=0, columnspan=3, padx=10, pady=5, sticky="nsew")

    def _update_model_list(self, event=None):
        """Updates the model list based on selected category."""
        category = self.model_category.get()  # ty: ignore[unresolved-attribute]

        # Filter models by category
        if category == "YOLO26":
            models = [m for m in self.available_models if m.startswith("yolo26")]
        elif category == "YOLO11":
            models = [m for m in self.available_models if m.startswith("yolo11")]
        elif category == "YOLOv8":
            models = [m for m in self.available_models if m.startswith("yolov8")]
        elif category == "YOLOv9":
            models = [m for m in self.available_models if m.startswith("yolov9")]
        else:
            models = self.available_models

        # Update combobox values
        self.model_combo["values"] = models  # ty: ignore[unresolved-attribute]

        # Set default model for category
        if models:
            if category == "YOLO26":
                self.base_model.set("yolo26m.pt")
            elif category == "YOLO11":
                self.base_model.set("yolo11m.pt")
            elif category == "YOLOv8":
                self.base_model.set("yolov8m.pt")
            elif category == "YOLOv9":
                self.base_model.set("yolov9m.pt")
            else:
                self.base_model.set(models[0])

    def show_model_help(self):
        """Shows help information about YOLO models."""
        help_text = """
YOLO MODEL SELECTION GUIDE
==========================

MODEL CATEGORIES:
• YOLO26: Latest version (2026) - Best performance (RECOMMENDED)
• YOLO11: Previous version (2024) - Well tested
• YOLOv9: Advanced features
• YOLOv8: Stable version - Legacy support

MODEL SIZES (n → s → m → l → x):
• n (Nano): ~6MB - Fastest, lowest accuracy
• s (Small): ~20MB - Fast, low accuracy
• m (Medium): ~50MB - Balanced (RECOMMENDED)
• l (Large): ~150MB - Slow, high accuracy
• x (Extra Large): ~300MB - Slowest, highest accuracy

RECOMMENDATIONS:
• For beginners: Use YOLO26m (default)
• For real-time: Use nano (n) models
• For best accuracy: Use large (l) or extra large (x) models
• For mobile/edge: Use nano (n) or small (s) models

TRAINING CONSIDERATIONS:
• Larger models need more GPU memory
• Smaller models may need more training epochs
• Medium models offer best balance of speed/accuracy

For more info: https://docs.ultralytics.com/models/yolo26/
        """

        messagebox.showinfo("YOLO Model Guide", help_text)

    def show_anylabeling_help(self):
        """Shows supported dataset help."""
        help_text = """
YOLO DATASET GUIDE
==================

1. GETPIXELVIDEO / SAM POINTS CSV:
   • Click "Create Dataset from getpixelvideo CSV"
   • Select sam_points_georeid.csv or any CSV with frame,p1_x,p1_y,... columns
   • Select the source video used to create the CSV
   • Enter the class name (for example: athlete, player, ball)
   • The tool extracts frames and writes YOLO labels for tracking/detection retrain

2. EXISTING YOLO DATASET:
   your_dataset/
   ├── train/images and train/labels
   ├── val/images and val/labels
   ├── test/images and test/labels (optional)
   ├── classes.txt
   └── data.yaml

3. USING THIS TOOL:
   • Browse a dataset folder or create one from CSV
   • Select or create data.yaml
   • Start training

4. NOTE:
   getpixelvideo point CSVs do not contain full boxes, so this builder creates
   fixed-size boxes centered on each visible point. Adjust box size when prompted.
        """

        messagebox.showinfo("Dataset Guide", help_text)

    def browse_dataset(self):
        """Browse for AnyLabeling dataset folder."""
        path = filedialog.askdirectory(title="Select AnyLabeling Dataset Folder")

        if path:
            self.dataset_path.set(path)
            # Check if YAML already exists
            yaml_file = os.path.join(path, "data.yaml")
            if os.path.exists(yaml_file):
                self.yaml_path.set(yaml_file)
                messagebox.showinfo(
                    "YAML Found",
                    f"Found existing YAML file:\n{yaml_file}\n\n"
                    "You can use this file or create a new one.",
                )
            else:
                # Ask user if they want to create YAML
                result = messagebox.askyesno(
                    "Create YAML",
                    f"No YAML file found in:\n{path}\n\nWould you like to create a new YAML file?",
                )
                if result:
                    self._auto_detect_and_create_yaml(path)

    def browse_yaml(self):
        """Browse for existing YAML file."""
        if not self.dataset_path.get():
            messagebox.showwarning("Warning", "Please select a dataset folder first!")
            return

        yaml_file = filedialog.askopenfilename(
            title="Select YAML Configuration File",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialdir=self.dataset_path.get(),
        )

        if yaml_file:
            self.yaml_path.set(yaml_file)
            print(f"Selected YAML file: {yaml_file}")

    def create_new_yaml(self):
        """Create new YAML file for current dataset."""
        if not self.dataset_path.get():
            messagebox.showwarning("Warning", "Please select a dataset folder first!")
            return

        # Ask for confirmation
        result = messagebox.askyesno(
            "Create New YAML",
            "This will create a new YAML file with default settings.\n"
            "Any existing YAML will be overwritten.\n\n"
            "Continue?",
        )

        if result:
            success = self._auto_detect_and_create_yaml(self.dataset_path.get())
            if success:
                print(f"New YAML created: {self.yaml_path.get()}")

    def create_dataset_from_getpixelvideo_csv(self):
        """Create a YOLO detection/tracking dataset from getpixelvideo point CSV + video.

        Heavy work (CSV parsing + sequential frame extraction) runs in a worker
        thread to keep the Tk main loop responsive on large CSVs.
        """
        csv_path = filedialog.askopenfilename(
            title="Select getpixelvideo / sam_points_georeid CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not csv_path:
            return

        video_path = filedialog.askopenfilename(
            title="Select source video for this CSV",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
                ("All files", "*.*"),
            ],
            initialdir=os.path.dirname(csv_path),
        )
        if not video_path:
            return

        class_name = simpledialog.askstring(
            "Class name",
            "YOLO class name for each marked point:",
            initialvalue="object",
            parent=self,
        )
        if not class_name:
            return
        class_name = " ".join(class_name.strip().split()) or "object"

        box_size = simpledialog.askinteger(
            "Box size",
            "Fixed box size in pixels around each point:",
            initialvalue=48,
            minvalue=4,
            maxvalue=512,
            parent=self,
        )
        if not box_size:
            return

        output_parent = filedialog.askdirectory(
            title="Select output folder for the generated dataset",
            initialdir=os.path.dirname(csv_path),
        )
        if not output_parent:
            return

        self._set_dataset_button_state("disabled")
        self._append_console("Building YOLO dataset in background thread...")
        worker = threading.Thread(
            target=self._dataset_build_worker,
            kwargs={
                "csv_path": csv_path,
                "video_path": video_path,
                "output_parent": output_parent,
                "class_name": class_name,
                "box_size": float(box_size),
            },
            daemon=True,
        )
        worker.start()

    def _set_dataset_button_state(self, state):
        """Toggle the CSV-dataset button so the user can't queue parallel builds."""
        button = getattr(self, "csv_button", None)
        if button is not None:
            with contextlib.suppress(tk.TclError):
                button.configure(state=state)

    def _append_console(self, message):
        """Thread-safe-friendly console append (call only from the Tk main thread)."""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)

    def _dataset_build_worker(self, csv_path, video_path, output_parent, class_name, box_size):
        """Run the dataset build off the Tk main thread; marshal results back via after()."""
        try:
            dataset_dir, yaml_file, message = self._build_tracking_dataset_from_pixel_csv(
                csv_path=csv_path,
                video_path=video_path,
                output_parent=output_parent,
                class_name=class_name,
                box_size=box_size,
                progress_callback=lambda msg: self.after(0, self._append_console, msg),
            )
        except Exception as exc:
            self.after(0, self._on_dataset_build_failed, str(exc))
            return
        self.after(0, self._on_dataset_build_succeeded, dataset_dir, yaml_file, message)

    def _on_dataset_build_failed(self, exc_msg):
        self._set_dataset_button_state("normal")
        self._append_console(f"Dataset build failed: {exc_msg}")
        messagebox.showerror("Dataset Error", f"Failed to create dataset:\n\n{exc_msg}")

    def _on_dataset_build_succeeded(self, dataset_dir, yaml_file, message):
        self._set_dataset_button_state("normal")
        self.dataset_path.set(dataset_dir)
        self.yaml_path.set(yaml_file)
        self._append_console(message)
        messagebox.showinfo("Dataset Created", message)

    def _build_tracking_dataset_from_pixel_csv(
        self,
        csv_path,
        video_path,
        output_parent,
        class_name,
        box_size,
        progress_callback=None,
    ):
        """Convert frame,pN_x,pN_y CSV markers into YOLO detect labels.

        Frames are read **sequentially** with cap.grab()/retrieve(); the previous
        implementation seeked in shuffled order via cap.set(POS_FRAMES, ...),
        which is both slow (keyframe re-decode per row) and codec-inaccurate
        (target frame may land on a neighbor). Iterating in ascending frame
        order also makes the build O(max_frame) instead of O(N_rows * seek).
        Split assignment is still random, but each frame is decoded once and
        each frame belongs to exactly one split (no train -> val duplication).
        """

        def report(msg):
            if progress_callback is not None:
                with contextlib.suppress(Exception):
                    progress_callback(msg)

        df = pd.read_csv(csv_path)
        if "frame" not in df.columns:
            raise ValueError("CSV must contain a 'frame' column.")

        point_ids = sorted(
            {
                col[1:-2]
                for col in df.columns
                if col.startswith("p") and col.endswith("_x") and col[1:-2].isdigit()
            },
            key=lambda value: int(value),
        )
        if not point_ids:
            raise ValueError("No pN_x/pN_y columns found in CSV.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                raise ValueError("Could not read video width/height.")

            video_stem = pathlib.Path(video_path).stem
            out_dir = (
                pathlib.Path(output_parent)
                / f"yolo_tracking_{video_stem}_{datetime.now():%Y%m%d_%H%M%S}"
            )

            valid_rows = []
            for _row_index, row in df.iterrows():
                frame_value = pd.to_numeric(row.get("frame"), errors="coerce")
                if pd.isna(frame_value) or frame_value < 0:
                    continue
                labels = []
                for point_id in point_ids:
                    point = self._row_point(row, point_id)
                    if point is None:
                        continue
                    x_px, y_px = point
                    labels.append(self._point_to_yolo_label(x_px, y_px, box_size, width, height))
                if labels:
                    valid_rows.append((int(frame_value), labels))

            if not valid_rows:
                raise ValueError("CSV has no visible points to export.")

            n_total = len(valid_rows)
            if n_total < 2:
                # Refuse to fabricate val by duplicating the single train row,
                # which would silently leak training data into validation.
                raise ValueError(
                    "Need at least 2 frames with visible points to build a "
                    "train/val split (got 1). Annotate more frames first."
                )

            # Allocate train and val first (both guaranteed >=1), test takes the rest.
            n_train = max(1, round(n_total * 0.7))
            n_train = min(n_train, n_total - 1)  # always leave at least 1 row for val
            remaining = n_total - n_train
            n_val = max(1, round(n_total * 0.2))
            n_val = min(n_val, remaining)
            n_test = remaining - n_val

            # Shuffle once for split assignment, then sort *within each split*
            # so the actual video decode below is purely ascending.
            shuffled = list(valid_rows)
            random.shuffle(shuffled)
            train_rows = sorted(shuffled[:n_train], key=lambda x: x[0])
            val_rows = sorted(shuffled[n_train : n_train + n_val], key=lambda x: x[0])
            test_rows = sorted(shuffled[n_train + n_val :], key=lambda x: x[0])

            # Materialize the per-frame split assignment. Disjoint slices of
            # `shuffled` guarantee a given frame never appears in two splits.
            frame_to_split: dict[int, tuple[str, list[str]]] = {}
            for frame_num, labels in train_rows:
                frame_to_split[frame_num] = ("train", labels)
            for frame_num, labels in val_rows:
                frame_to_split[frame_num] = ("val", labels)
            for frame_num, labels in test_rows:
                frame_to_split[frame_num] = ("test", labels)

            # Only create directories for splits we will actually populate.
            active_splits = {split for split, _ in frame_to_split.values()}
            for split in active_splits:
                (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
                (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

            sorted_targets = sorted(frame_to_split.keys())
            report(
                f"Extracting {len(sorted_targets)} frames sequentially "
                f"(train={n_train}, val={n_val}, test={n_test})..."
            )

            written = {"train": 0, "val": 0, "test": 0}
            target_idx = 0
            current_frame = 0
            progress_every = max(1, len(sorted_targets) // 20)

            while target_idx < len(sorted_targets):
                target = sorted_targets[target_idx]
                # Skip frames before the target with grab() (no decode cost).
                reached = True
                while current_frame < target:
                    if not cap.grab():
                        reached = False
                        break
                    current_frame += 1
                if not reached:
                    unreachable = len(sorted_targets) - target_idx
                    report(
                        f"Reached end of video; {unreachable} target frames "
                        f"are beyond the last decoded frame ({current_frame - 1})."
                    )
                    break

                ok, frame = cap.read()
                current_frame += 1
                if not ok or frame is None:
                    target_idx += 1
                    continue

                split, labels = frame_to_split[target]
                image_name = f"{video_stem}_frame_{target:06d}.jpg"
                label_name = f"{video_stem}_frame_{target:06d}.txt"
                cv2.imwrite(str(out_dir / split / "images" / image_name), frame)
                with open(out_dir / split / "labels" / label_name, "w", encoding="utf-8") as f:
                    f.write("\n".join(labels) + "\n")
                written[split] += 1
                target_idx += 1

                if target_idx % progress_every == 0:
                    report(
                        f"  {target_idx}/{len(sorted_targets)} frames extracted "
                        f"(train={written['train']}, val={written['val']}, "
                        f"test={written['test']})"
                    )

            if written["train"] == 0 or written["val"] == 0:
                raise ValueError(
                    "Could not extract enough video frames for train/val splits "
                    f"(written train={written['train']}, val={written['val']}). "
                    "Check that CSV frame numbers match the video."
                )

            with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
                f.write(class_name + "\n")

            train_path = (out_dir / "train" / "images").resolve().as_posix()
            val_path = (out_dir / "val" / "images").resolve().as_posix()
            test_path = (out_dir / "test" / "images").resolve().as_posix()
            yaml_file = out_dir / "data.yaml"
            yaml_file.write_text(
                "# YOLO detection/tracking dataset - generated by vaila yolotrain\n"
                f"path: {out_dir.resolve().as_posix()}\n"
                f"train: {train_path}\n"
                f"val: {val_path}\n"
                f"test: {test_path}\n"
                "nc: 1\n"
                f"names: {[class_name]!r}\n"
                "\n"
                "# Training defaults editable before Start Training\n"
                "model: yolo26m.pt\n"
                "epochs: 100\n"
                "batch: 16\n"
                "imgsz: 640\n"
                "device: cpu\n",
                encoding="utf-8",
            )

            message = (
                "YOLO tracking dataset created from getpixelvideo CSV:\n"
                f"  Dataset: {out_dir}\n"
                f"  CSV: {csv_path}\n"
                f"  Video: {video_path}\n"
                f"  Class: {class_name}\n"
                f"  Points exported as fixed {int(box_size)} px boxes\n"
                f"  Train/Val/Test images: "
                f"{written['train']}/{written['val']}/{written['test']}\n"
                f"  YAML: {yaml_file}"
            )
            return str(out_dir), str(yaml_file), message
        finally:
            cap.release()

    @staticmethod
    def _row_point(row, point_id):
        """Return pN point, preferring edited x/y and falling back to mask/centroid columns."""
        candidates = (
            (f"p{point_id}_x", f"p{point_id}_y"),
            (f"p{point_id}_mx", f"p{point_id}_my"),
            (f"p{point_id}_cx", f"p{point_id}_cy"),
        )
        for x_col, y_col in candidates:
            if x_col not in row.index or y_col not in row.index:
                continue
            x_val = pd.to_numeric(row.get(x_col), errors="coerce")
            y_val = pd.to_numeric(row.get(y_col), errors="coerce")
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            return float(x_val), float(y_val)
        return None

    @staticmethod
    def _point_to_yolo_label(x_px, y_px, box_size, width, height):
        """Build one YOLO detect label line from a center point and fixed box size."""
        half = box_size / 2.0
        x1 = max(0.0, x_px - half)
        y1 = max(0.0, y_px - half)
        x2 = min(float(width), x_px + half)
        y2 = min(float(height), y_px + half)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        cx = (x1 + box_w / 2.0) / float(width)
        cy = (y1 + box_h / 2.0) / float(height)
        return f"0 {cx:.6f} {cy:.6f} {box_w / width:.6f} {box_h / height:.6f}"

    @staticmethod
    def _resolve_yaml_path(yaml_data, dataset_dir, key):
        """Resolve Ultralytics train/val/test path entries.

        Supports absolute paths, ``./x``, ``../x``, bare ``train/images``,
        and the Ultralytics ``path: + images/train`` layout.
        """
        raw = yaml_data.get(key, "")
        # Ultralytics also allows a list of dirs; resolve the first non-empty one.
        if isinstance(raw, (list, tuple)):
            for item in raw:
                value = str(item).strip()
                if value:
                    break
            else:
                value = ""
        else:
            value = str(raw).strip()
        if not value:
            return value
        if os.path.isabs(value):
            return value
        base = str(yaml_data.get("path") or dataset_dir).strip() or dataset_dir
        if not os.path.isabs(base):
            base = os.path.join(dataset_dir, base)
        # Strip a leading "./" only; the previous str.lstrip("./") removed any
        # combination of '.' and '/' chars, corrupting e.g. "../images/train"
        # into "images/train" or ".datasets/x" into "datasets/x".
        return os.path.normpath(os.path.join(base, value.removeprefix("./")))

    def _auto_detect_and_create_yaml(self, dataset_path):
        """Automatically detects AnyLabeling structure and creates YAML."""
        # Check for classes.txt
        classes_file = os.path.join(dataset_path, "classes.txt")
        class_names = []

        if os.path.exists(classes_file):
            try:
                with open(classes_file, encoding="utf-8") as f:
                    class_names = [line.strip() for line in f if line.strip()]
                print(f"Found {len(class_names)} classes in classes.txt")
            except Exception as e:
                print(f"Error reading classes.txt: {e}")

        # Check for AnyLabeling structure (train/val already created)
        train_images = os.path.join(dataset_path, "train", "images")
        val_images = os.path.join(dataset_path, "val", "images")
        test_images = os.path.join(dataset_path, "test", "images")
        if not os.path.exists(train_images) or not os.path.exists(val_images):
            messagebox.showerror(
                "Error",
                "AnyLabeling structure not found!\n\n"
                "Expected:\n"
                f"  {dataset_path}/train/images/\n"
                f"  {dataset_path}/val/images/\n"
                f"  {dataset_path}/classes.txt\n\n"
                "Please export from AnyLabeling in YOLO format.",
            )
            return False  # Return False to indicate failure

        # Create comprehensive YAML with ABSOLUTE paths
        train_path = os.path.join(dataset_path, "train", "images")
        val_path = os.path.join(dataset_path, "val", "images")
        test_path = (
            os.path.join(dataset_path, "test", "images") if os.path.exists(test_images) else None
        )

        # Convert to forward slashes for YAML compatibility
        train_path = train_path.replace("\\", "/")
        val_path = val_path.replace("\\", "/")
        if test_path:
            test_path = test_path.replace("\\", "/")

        yaml_content = self._create_comprehensive_yaml(train_path, val_path, test_path, class_names)
        yaml_file_path = os.path.join(dataset_path, "data.yaml")

        try:
            with open(yaml_file_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            self.yaml_path.set(yaml_file_path)

            # Count files for verification
            train_count = len(
                [
                    f
                    for f in os.listdir(train_images)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
            )
            val_count = len(
                [
                    f
                    for f in os.listdir(val_images)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
                ]
            )

            messagebox.showinfo(
                "Success",
                f"Dataset ready for training!\n\n"
                f"Classes: {len(class_names)}\n"
                f"Train images: {train_count}\n"
                f"Val images: {val_count}\n"
                f"YAML: {yaml_file_path}\n\n"
                f"The YAML file contains all available options commented.\n"
                f"Uncomment and modify parameters as needed.\n\n"
                f"You can now start training.",
            )

            return True  # Return True to indicate success

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create YAML: {str(e)}")
            return False  # Return False to indicate failure

    def _create_comprehensive_yaml(self, train_path, val_path, test_path, class_names):
        """Creates comprehensive YAML file with all options commented."""
        names_str = str(class_names)

        # Add test path if exists
        test_line = f"\ntest: {test_path}" if test_path else ""

        # Create comprehensive YAML with ABSOLUTE paths
        yaml_content = f"""# YOLO Dataset Configuration
# Generated by vailá for AnyLabeling
# Based on Ultralytics YOLO11 documentation: https://docs.ultralytics.com/modes/train/

# =============================================================================
# DATASET CONFIGURATION (REQUIRED)
# =============================================================================
path: .
train: {train_path}  # train images (absolute path)
val: {val_path}  # val images (absolute path){test_line}

# =============================================================================
# CLASS CONFIGURATION (REQUIRED)
# =============================================================================
nc: {len(class_names)}  # number of classes
names: {names_str}  # class names

# =============================================================================
# TRAINING PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# Basic Training Parameters
# epochs: 100  # number of epochs to train
# batch: 16  # batch size (-1 for auto)
# imgsz: 640  # image size for training
# patience: 50  # early stopping patience

# Model Parameters
# model: yolo11m.pt  # starting model
# pretrained: true  # use pretrained weights

# Optimizer Settings
# optimizer: auto  # optimizer (auto, SGD, Adam, AdamW, RMSProp)
# lr0: 0.01  # initial learning rate
# lrf: 0.01  # final learning rate factor
# momentum: 0.937  # momentum
# weight_decay: 0.0005  # weight decay
# warmup_epochs: 3.0  # warmup epochs
# warmup_momentum: 0.8  # warmup momentum
# warmup_bias_lr: 0.1  # warmup bias learning rate

# Loss Weights
# box: 7.5  # box loss weight
# cls: 0.5  # classification loss weight
# dfl: 1.5  # distribution focal loss weight

# =============================================================================
# AUGMENTATION PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# HSV Color Augmentation
# hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
# hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
# hsv_v: 0.4  # image HSV-Value augmentation (fraction)

# Geometric Augmentation
# degrees: 0.0  # image rotation (+/- deg)
# translate: 0.1  # image translation (+/- fraction)
# scale: 0.5  # image scale (+/- gain)
# shear: 0.0  # image shear (+/- deg)
# perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
# flipud: 0.0  # image flip up-down (probability)
# fliplr: 0.5  # image flip left-right (probability)

# Advanced Augmentation
# mosaic: 1.0  # image mosaic (probability)
# mixup: 0.0  # image mixup (probability)
# copy_paste: 0.0  # segment copy-paste (probability)

# Classification Augmentation
# auto_augment: randaugment  # auto augmentation policy (randaugment, autoaugment, augmix)
# erasing: 0.4  # random erasing probability
# crop_fraction: 1.0  # crop fraction for classification

# =============================================================================
# VALIDATION AND SAVING PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# Validation
# val: true  # validate during training
# split: val  # dataset split to use for validation

# Saving
# save: true  # save checkpoints
# save_period: -1  # save every x epochs (disabled with -1)

# =============================================================================
# PERFORMANCE PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# Caching and Workers
# cache: false  # cache images for faster training
# workers: 8  # number of worker threads for data loading

# Device and Memory
# device: 0  # cuda device (0 for GPU, cpu for CPU)
# amp: true  # automatic mixed precision (AMP) training

# =============================================================================
# PROJECT AND EXPERIMENT PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# Project Settings
# project: runs/train  # project name
# name: exp  # experiment name
# exist_ok: false  # overwrite existing experiment
# plots: true  # create plots

# Resume and Checkpoints
# resume: false  # resume training from last checkpoint
# noval: false  # only validate final epoch
# nosave: false  # only save final checkpoint

# =============================================================================
# ADVANCED PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# Training Mode
# single_cls: false  # train as single-class dataset
# rect: false  # rectangular training
# cos_lr: false  # cosine learning rate scheduler
# close_mosaic: 10  # disable mosaic augmentation for final 10 epochs

# Loss and Metrics
# label_smoothing: 0.0  # label smoothing epsilon
# overlap_mask: true  # masks should overlap during training
# mask_ratio: 4  # mask downsample ratio
# dropout: 0.0  # use dropout regularization

# Deterministic Training
# seed: 0  # random seed for reproducibility
# deterministic: true  # deterministic training

# =============================================================================
# LOGGING PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# Logging
# verbose: true  # verbose output
# profile: false  # profile ONNX and TensorRT speeds during training

# =============================================================================
# DEPLOYMENT PARAMETERS (OPTIONAL - Uncomment to customize)
# =============================================================================

# Export
# half: true  # use FP16 half-precision training
# ddp: false  # use DistributedDataParallel mode
# evolve: false  # evolve hyperparameters for x generations
# multi_scale: false  # vary img-size +/- 50%%
# single_cls: false  # train as single-class dataset
# optimizer: auto  # optimizer (SGD, Adam, etc.)
# sync_bn: false  # use SyncBatchNorm, only available in DDP mode
# workers: 8  # max dataloader workers (per RANK in DDP mode)
# project: runs/train  # save to project/name
# name: exp  # save to project/name
# exist_ok: false  # existing project/name ok, do not increment
# quad: false  # quad dataloader
# cos_lr: false  # cosine learning rate scheduler
# label_smoothing: 0.0  # label smoothing epsilon
# patience: 100  # EarlyStopping patience (epochs)
# freeze: [0]  # layers to freeze (trainable layers)
# save_period: -1  # Save checkpoint every x epochs (disabled if < 1)
# local_rank: -1  # DDP parameter, do not modify
# entity: null  # Entity
# upload_dataset: false  # Upload dataset as W&B artifact table
# bbox_interval: -1  # Set bounding-box image logging interval
# artifact_alias: latest  # Version of dataset artifact to be used
"""
        return yaml_content

    def start_training_thread(self):
        """Starts training in a separate thread."""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select a dataset folder first!")
            return

        if not self.yaml_path.get():
            messagebox.showerror(
                "Error", "No YAML file selected! Please browse or create a YAML file."
            )
            return

        # Validate YAML before starting
        if not self._validate_existing_yaml(self.yaml_path.get()):
            return

        self.start_button.config(state=tk.DISABLED)
        self.console.delete("1.0", tk.END)
        self.console.insert(tk.END, "Starting training...\n")

        training_thread = threading.Thread(target=self._run_training)
        training_thread.daemon = True
        training_thread.start()

    def _run_training(self):
        """Runs the YOLO training."""
        original_stdout = sys.stdout
        sys.stdout = ConsoleRedirector(self.console)

        try:
            # Get parameters from GUI (only essential ones)
            dataset_folder = self.dataset_path.get()
            yaml_file = self.yaml_path.get()
            # model_name = self.base_model.get() # This line is removed

            # Print to both GUI and terminal
            print("=" * 60)
            print("YOLO TRAINING STARTING")
            print("=" * 60)
            print(f"Dataset folder: {dataset_folder}")
            print(f"YAML file: {yaml_file}")
            # print(f"Model: {model_name}") # This line is removed
            print(f"Run name: {self.project_name.get()}")  # Keep project_name for consistency
            print("=" * 60)

            # Validate YAML file
            print(f"\nValidating YAML file: {yaml_file}")
            if not os.path.exists(yaml_file):
                raise FileNotFoundError(f"YAML file not found: {yaml_file}")

            # Read and validate YAML content
            try:
                import yaml

                with open(yaml_file) as f:
                    yaml_data = yaml.safe_load(f)
                print("YAML loaded successfully")
                print(f"Classes: {yaml_data.get('nc', 'Not found')}")
                print(f"Class names: {yaml_data.get('names', 'Not found')}")
                print(f"Train path: {yaml_data.get('train', 'Not found')}")
                print(f"Val path: {yaml_data.get('val', 'Not found')}")

                # Get training parameters from YAML (with defaults)
                epochs = yaml_data.get("epochs", 100)
                batch_size = yaml_data.get("batch", 16)
                img_size = yaml_data.get("imgsz", 640)
                device = yaml_data.get("device", "cpu")

                # Get model name from YAML (with default)
                model_name = yaml_data.get("model", "yolo26m.pt")

                print("\nTraining parameters from YAML:")
                print(f"  Epochs: {epochs}")
                print(f"  Batch size: {batch_size}")
                print(f"  Image size: {img_size}")
                print(f"  Device: {device}")
                print(f"  Model: {model_name}")

                # Validate that the paths actually exist
                train_path = yaml_data.get("train", "")
                val_path = yaml_data.get("val", "")

                # Convert YAML paths to absolute paths. Supports absolute paths, ./x,
                # train/images, and Ultralytics path: + images/train layouts.
                train_path = self._resolve_yaml_path(yaml_data, os.path.dirname(yaml_file), "train")
                val_path = self._resolve_yaml_path(yaml_data, os.path.dirname(yaml_file), "val")

                print(f"Absolute train path: {train_path}")
                print(f"Absolute val path: {val_path}")

                if not os.path.exists(train_path):
                    raise FileNotFoundError(f"Train images not found: {train_path}")
                if not os.path.exists(val_path):
                    raise FileNotFoundError(f"Val images not found: {val_path}")

                print(f"Train images found: {len(os.listdir(train_path))} files")
                print(f"Val images found: {len(os.listdir(val_path))} files")

            except Exception as e:
                print(f"Error reading YAML: {e}")
                raise

            # Create models directory and download model if needed
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, model_name)

            print("\n--- MODEL INFORMATION ---")
            print(f"Selected model: {model_name}")
            print(f"Models directory: {models_dir}")
            print(f"Model path: {model_path}")

            # Show model characteristics
            self._show_model_characteristics(model_name)

            # Download the model if it doesn't exist
            if not os.path.exists(model_path):
                try:
                    print(f"\nDownloading model {model_name}...")
                    print("This may take a few minutes depending on your internet connection.")
                    print("Model size varies by type:")
                    print("  • Nano models (~6-20 MB)")
                    print("  • Small models (~20-50 MB)")
                    print("  • Medium models (~50-150 MB)")
                    print("  • Large models (~150-300 MB)")
                    print("  • Extra Large models (~300-600 MB)")

                    current_dir = os.getcwd()
                    os.chdir(models_dir)

                    # Download using YOLO with progress indication
                    # Use full path to avoid downloading to project root
                    print("\nInitiating download...")
                    model = YOLO(model_path)

                    os.chdir(current_dir)

                    # Check if file was actually downloaded
                    if os.path.exists(model_path):
                        size_mb = os.path.getsize(model_path) / (1024 * 1024)
                        print("Model downloaded successfully!")
                        print(f"  File size: {size_mb:.1f} MB")
                        print(f"  Location: {model_path}")
                    else:
                        raise FileNotFoundError(
                            f"Model file not found after download: {model_path}"
                        )

                except Exception as e:
                    print(f"Error downloading model: {e}")
                    print("Possible solutions:")
                    print("  • Check your internet connection")
                    print("  • Verify the model name is correct")
                    print("  • Try a different model")
                    print("  • Check if Ultralytics is up to date")
                    raise
            else:
                size_mb = os.path.getsize(model_path) / (1024 * 1024)
                print(f"Model already exists: {model_path}")
                print(f"  File size: {size_mb:.1f} MB")

            print(f"\nLoading model: {model_name}")
            model = YOLO(model_path)
            print("Model loaded successfully")

            # Define output directory structure
            output_dir = os.path.join(dataset_folder, "runs")
            run_output_dir = os.path.join(output_dir, self.project_name.get())  # Use project_name
            weights_dir = os.path.join(run_output_dir, "weights")

            print("\n--- Training Configuration ---")
            print(f"Dataset: {dataset_folder}")
            print(f"YAML: {yaml_file}")
            print(f"Model: {model_name}")
            print(f"Epochs: {epochs}")
            print(f"Batch Size: {batch_size}")
            print(f"Image Size: {img_size}")
            print(f"Device: {device}")
            print(f"Output Directory: {output_dir}")
            print(f"Run Directory: {run_output_dir}")
            print(f"Weights Directory: {weights_dir}")
            print("-----------------------------\n")

            # Start training with detailed output
            print("Starting YOLO training...")
            print("This may take several minutes depending on your dataset size.")
            print("Training progress will be shown below:\n")

            # Start training with parameters from YAML
            results = model.train(  # noqa: F841 - Training results object (important for future use)
                data=yaml_file,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=device,
                project=output_dir,
                name=self.project_name.get(),  # Use project_name
                exist_ok=True,
                verbose=True,  # This ensures detailed output
            )

            # Show completion with detailed info
            best_model_path = os.path.join(weights_dir, "best.pt")
            last_model_path = os.path.join(weights_dir, "last.pt")

            print("\n" + "=" * 60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("OUTPUT LOCATIONS:")
            print(f"   Best model: {best_model_path}")
            print(f"   Last model: {last_model_path}")
            print(f"   Results folder: {run_output_dir}")
            print(f"   Weights folder: {weights_dir}")
            print(f"   Training logs: {run_output_dir}")
            print(f"   Plots and graphs: {run_output_dir}")
            print("=" * 60)

            # Check if files exist and show file sizes
            self._show_model_info(best_model_path, last_model_path, run_output_dir)

            def show_completion():
                result = messagebox.askyesno(
                    "Training Completed",
                    f"Training completed successfully!\n\n"
                    f"OUTPUT LOCATIONS:\n"
                    f"   Best model: {best_model_path}\n"
                    f"   Last model: {last_model_path}\n"
                    f"   Results folder: {run_output_dir}\n\n"
                    f"The 'best.pt' model has the highest validation accuracy.\n"
                    f"The 'last.pt' model is from the final epoch.\n\n"
                    f"Open results folder?",
                )

                if result:
                    if sys.platform == "win32":
                        os.startfile(run_output_dir)
                    elif sys.platform == "darwin":
                        os.system(f"open '{run_output_dir}'")
                    else:
                        os.system(f"xdg-open '{run_output_dir}'")

            self.after(0, show_completion)

        except Exception as e:
            error_msg = str(e)
            print("\n" + "=" * 60)
            print("TRAINING ERROR!")
            print("=" * 60)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {error_msg}")
            print(f"Error details: {e}")
            print("=" * 60)

            # Also print to original stdout for terminal
            original_stdout.write(f"\nTRAINING ERROR: {error_msg}\n")
            original_stdout.flush()

            # Fix the lambda scope issue
            error_type = type(e).__name__
            error_details = str(e)  # noqa: F841 - Error details for potential future use

            self.after(
                0,
                lambda: messagebox.showerror(
                    "Training Error",
                    f"Error during training:\n\n"
                    f"Type: {error_type}\n"
                    f"Message: {error_msg}\n\n"
                    f"Check the console output above for details.",
                ),
            )
        finally:
            sys.stdout = original_stdout
            self.after(0, lambda: self.start_button.config(state=tk.NORMAL))

    def _show_model_characteristics(self, model_name):
        """Shows characteristics of the selected model."""
        print("\nModel characteristics:")

        # Model size and performance characteristics
        model_info = {
            # YOLO26 Models (Latest - 2026)
            "yolo26n.pt": {
                "size": "~6MB",
                "speed": "Fastest",
                "accuracy": "Lower",
                "use": "Edge devices, real-time",
            },
            "yolo26s.pt": {
                "size": "~20MB",
                "speed": "Fast",
                "accuracy": "Low",
                "use": "Mobile, embedded",
            },
            "yolo26m.pt": {
                "size": "~50MB",
                "speed": "Medium",
                "accuracy": "Good",
                "use": "General purpose (recommended)",
            },
            "yolo26l.pt": {
                "size": "~150MB",
                "speed": "Slow",
                "accuracy": "High",
                "use": "High accuracy needed",
            },
            "yolo26x.pt": {
                "size": "~300MB",
                "speed": "Slowest",
                "accuracy": "Highest",
                "use": "Best accuracy",
            },
            # YOLO11 Models
            "yolo11n.pt": {
                "size": "~6MB",
                "speed": "Fastest",
                "accuracy": "Lower",
                "use": "Edge devices, real-time",
            },
            "yolo11s.pt": {
                "size": "~20MB",
                "speed": "Fast",
                "accuracy": "Low",
                "use": "Mobile, embedded",
            },
            "yolo11m.pt": {
                "size": "~50MB",
                "speed": "Medium",
                "accuracy": "Good",
                "use": "General purpose",
            },
            "yolo11l.pt": {
                "size": "~150MB",
                "speed": "Slow",
                "accuracy": "High",
                "use": "High accuracy needed",
            },
            "yolo11x.pt": {
                "size": "~300MB",
                "speed": "Slowest",
                "accuracy": "Highest",
                "use": "Best accuracy",
            },
            # YOLOv8 Models
            "yolov8n.pt": {
                "size": "~6MB",
                "speed": "Fastest",
                "accuracy": "Lower",
                "use": "Edge devices, real-time",
            },
            "yolov8s.pt": {
                "size": "~20MB",
                "speed": "Fast",
                "accuracy": "Low",
                "use": "Mobile, embedded",
            },
            "yolov8m.pt": {
                "size": "~50MB",
                "speed": "Medium",
                "accuracy": "Good",
                "use": "General purpose",
            },
            "yolov8l.pt": {
                "size": "~150MB",
                "speed": "Slow",
                "accuracy": "High",
                "use": "High accuracy needed",
            },
            "yolov8x.pt": {
                "size": "~300MB",
                "speed": "Slowest",
                "accuracy": "Highest",
                "use": "Best accuracy",
            },
            # YOLOv9 Models
            "yolov9c.pt": {
                "size": "~10MB",
                "speed": "Fast",
                "accuracy": "Low",
                "use": "Compact, efficient",
            },
            "yolov9s.pt": {
                "size": "~20MB",
                "speed": "Fast",
                "accuracy": "Low",
                "use": "Mobile, embedded",
            },
            "yolov9m.pt": {
                "size": "~50MB",
                "speed": "Medium",
                "accuracy": "Good",
                "use": "General purpose",
            },
            "yolov9l.pt": {
                "size": "~150MB",
                "speed": "Slow",
                "accuracy": "High",
                "use": "High accuracy needed",
            },
            "yolov9e.pt": {
                "size": "~300MB",
                "speed": "Slowest",
                "accuracy": "Highest",
                "use": "Best accuracy",
            },
        }

        if model_name in model_info:
            info = model_info[model_name]
            print(f"  Size: {info['size']}")
            print(f"  Speed: {info['speed']}")
            print(f"  Accuracy: {info['accuracy']}")
            print(f"  Best for: {info['use']}")
        else:
            print(f"   Model: {model_name}")
            print("  Note: Model characteristics not available")

        # Show recommendations
        print("\nRecommendations:")
        if "n" in model_name:
            print("Good for: Real-time applications, edge devices")
            print("Consider: May need more training epochs for good accuracy")
        elif "s" in model_name:
            print("Good for: Mobile applications, embedded systems")
            print("Consider: Balance between speed and accuracy")
        elif "m" in model_name:
            print("Good for: General purpose training (recommended)")
            print("Consider: Best balance of speed, accuracy, and size")
        elif "l" in model_name:
            print("Good for: High accuracy requirements")
            print("Consider: Slower training and inference")
        elif "x" in model_name or "e" in model_name:
            print("Good for: Best possible accuracy")
            print("Consider: Requires more GPU memory and time")

    def _show_model_info(self, best_model_path, last_model_path, run_output_dir):
        """Shows detailed information about saved models."""
        print("\nMODEL INFORMATION:")

        # Check best model
        if os.path.exists(best_model_path):
            size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
            print(f"   Best model: {best_model_path}")
            print(f"      Size: {size_mb:.1f} MB")
        else:
            print(f"   Best model not found: {best_model_path}")

        # Check last model
        if os.path.exists(last_model_path):
            size_mb = os.path.getsize(last_model_path) / (1024 * 1024)
            print(f"   Last model: {last_model_path}")
            print(f"      Size: {size_mb:.1f} MB")
        else:
            print(f"   Last model not found: {last_model_path}")

        # List all files in the run directory
        if os.path.exists(run_output_dir):
            print("\nFILES IN RESULTS FOLDER:")
            for root, _dirs, files in os.walk(run_output_dir):
                level = root.replace(run_output_dir, "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    print(f"{subindent}{file} ({size_mb:.1f} MB)")

        print("\nUSAGE TIPS:")
        print("Use 'best.pt' for inference (highest accuracy)")
        print("Use 'last.pt' if you want to continue training")
        print(f"All training logs and plots are in: {run_output_dir}")
        print("You can copy the model files to use in other projects")

    def _validate_existing_yaml(self, yaml_file):
        """Validates existing YAML file."""
        try:
            import yaml

            with open(yaml_file) as f:
                yaml_data = yaml.safe_load(f)

            # Check required fields
            required_fields = ["train", "val", "nc", "names"]
            missing_fields = [field for field in required_fields if field not in yaml_data]

            if missing_fields:
                messagebox.showerror(
                    "Invalid YAML",
                    f"Missing required fields: {missing_fields}\n\n"
                    "Please use a valid YOLO YAML file.",
                )
                return False

            # Check if paths exist
            dataset_dir = os.path.dirname(yaml_file)
            train_path = yaml_data["train"]
            val_path = yaml_data["val"]

            # Convert YAML paths to absolute paths. Supports absolute paths, ./x,
            # train/images, and Ultralytics path: + images/train layouts.
            train_path = self._resolve_yaml_path(yaml_data, dataset_dir, "train")
            val_path = self._resolve_yaml_path(yaml_data, dataset_dir, "val")

            if not os.path.exists(train_path):
                messagebox.showerror("Invalid YAML", f"Train path not found: {train_path}")
                return False

            if not os.path.exists(val_path):
                messagebox.showerror("Invalid YAML", f"Val path not found: {val_path}")
                return False

            print("YAML validation successful:")
            print(f"  Classes: {yaml_data['nc']}")
            print(f"  Class names: {yaml_data['names']}")
            print(f"  Train path: {train_path}")
            print(f"  Val path: {val_path}")

            return True

        except Exception as e:
            messagebox.showerror("YAML Error", f"Error reading YAML file: {str(e)}")
            return False


# --- Application Entry Point ---
def run_yolotrain_gui():
    """Entry point function."""
    print(f"Running: {pathlib.Path(__file__).name}")
    print(f"Directory: {pathlib.Path(__file__).parent.resolve()}")

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch.set_num_threads(1)

    app = YOLOTrainApp()
    app.mainloop()


if __name__ == "__main__":
    run_yolotrain_gui()
