"""
Project: vailá
Script: yolov26track.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 February 2025
Update Date: 06 July 2026
Version: 0.3.72

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

Change History:
    - v0.3.66: Single-pass track+pose pipeline — geometric ID linker (SAM3-style
               IoU+centroid), upscaled bbox ROI for YOLO pose, global keypoint
               remap, ``all_id_pose.csv``, ``yolo_reid_links.csv``, and
               ``<stem>_track_pose_overlay.mp4``. GUI default mode is
               ``track+pose``; CLI adds ``--pose`` / ``--no-pose`` and related flags.
    - v0.3.65: Added headless ``track`` CLI subcommand
               (``python -m vaila.yolov26track track --model best.pt --source video.mp4``).
               Unlike Ultralytics ``yolo track`` (video only), it writes the biomechanics
               CSVs vailá's reconstruction pipeline needs:
                 * ``<stem>_markers.csv`` — getpixelvideo point format
                   ``frame,p1_x,p1_y,...,pN_x,pN_y`` (one ``--anchor`` point per player) →
                   direct input to REC2D (``rec2d.py``) / REC3D (``rec3d.py``);
                 * per-ID bbox CSVs (``{label}_id_NN.csv``) + wide ``all_id_detection.csv``
                   (loads straight into getpixelvideo);
                 * H.264 ``<stem>_track_overlay.mp4`` — always ``.mp4`` now (temp ``.avi``
                   is encoded then deleted; OpenCV ``mp4v`` fallback if FFmpeg missing), no
                   bulky AVI left on disk.
               Flags: ``--anchor``, ``--max-ids`` (global ID-cap rerank → stable ``p1..pN``),
               ``--classes``, ``--vid-stride``, ``--conf/--iou/--imgsz``. Vectorised per-ID
               and markers CSV writers (NumPy pre-fill, one ``to_csv`` each).
    - v0.3.61: ID-cap phase-1 buffer no longer retains full Ultralytics ``Results``
               (drops ``orig_img`` / GPU tensors per frame); phase-2 re-reads video from
               disk. Periodic ``gc`` + ``torch.cuda.empty_cache`` during long CUDA runs
               (SAM3-style); VRAM + host RAM logged at phase boundaries.
    - v0.3.60: Tracking terminal progress summarized by frame chunks (~100) with
               percentage only — no frame-by-frame spam; Ultralytics tqdm off.
    - v0.3.59: Terminal progress during tracking — phase banners and frame counters for
               YOLO inference (incl. ID-cap buffering) and output writing.
    - v0.3.58: Custom trained .pt paths no longer collapse to vaila/models/{stem}.pt during
               TensorRT auto-export; skip TRT for user-selected weights and load path directly.
    - 2026-06: Added "Max tracked IDs" post-tracking rerank cap (global rerank by persistence, applies to all sub-trackers via CSVs); clarified that BoT-SORT w/ ReID uses yolo26n-cls.pt for appearance features (not as detector) when model head is end2end.
    - 2026-05: Ultralytics dir bootstrap before YOLO import; recursive tracking-dir discovery; TRT10-style trtexec; SAM-shaped yolo_contours + manifest columns; FFmpeg for seg/all overlays.
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

from __future__ import annotations

import colorsys
import contextlib
import csv
import datetime
import faulthandler
import gc
import glob
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import tkinter as tk
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any, cast

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from rich import print

# Must set Ultralytics home before importing YOLO (avoids weights in repo root / CWD).
VAILA_MODELS_DIR = Path(__file__).resolve().parent / "models"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_custom_model_path(model_name: str) -> bool:
    """True when the user picked a concrete weights file outside the catalog list."""
    text = os.path.expanduser(str(model_name or "").strip())
    if not text:
        return False
    if os.path.isabs(text):
        return True
    if os.sep in text or text.startswith(("./", ".\\")):
        return True
    lower = text.lower()
    return lower.endswith((".pt", ".onnx", ".engine")) and os.path.isfile(text)


@dataclass(frozen=True)
class _RunLog:
    path: Path


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for s in self._streams:
            try:
                n = s.write(data)
            except Exception:
                continue
        return n

    def flush(self) -> None:
        for s in self._streams:
            with contextlib.suppress(Exception):
                s.flush()


def _setup_run_logging(tag: str) -> _RunLog:
    """Always-on per-run log file under `vaila/models/logs/`."""
    logs_dir = VAILA_MODELS_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{tag}_{ts}.log"

    # Configure python logging (keep idempotent in case multiple entrypoints called).
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)

    # Tee stdout/stderr into file too (GUI callbacks sometimes swallow tracebacks).
    f = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    sys.stdout = _Tee(sys.__stdout__, f)  # type: ignore[assignment]
    sys.stderr = _Tee(sys.__stderr__, f)  # type: ignore[assignment]

    # Dump fatal signals (segfault / abort) into same log.
    with contextlib.suppress(Exception):
        faulthandler.enable(file=f, all_threads=True)

    # Announce log path both to console and to file (keep plain text).
    msg = f"[log] Writing: {log_path}\n"
    with contextlib.suppress(Exception):
        out = sys.__stdout__
        if out is not None:
            out.write(msg)
    with contextlib.suppress(Exception):
        f.write(msg)
        f.flush()
    return _RunLog(path=log_path)


def _move_root_ultralytics_weights_to_models() -> None:
    """Best-effort: move accidental YOLO/Ultralytics *.pt drops from repo root into vaila/models/."""
    try:
        VAILA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for src in sorted(_REPO_ROOT.glob("yolo*.pt")):
            if not src.is_file():
                continue
            dst = VAILA_MODELS_DIR / src.name
            try:
                if not dst.exists():
                    src.replace(dst)
                else:
                    src.unlink(missing_ok=True)
            except OSError:
                continue
    except Exception:
        return


def _configure_ultralytics_dirs(models_dir: Path) -> None:
    """Force Ultralytics cache/runs/weights under vailá `vaila/models/`."""
    root = models_dir / "ultralytics"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["ULTRALYTICS_DIR"] = str(root)

    # Newer Ultralytics versions expose `settings.update`.
    try:
        from ultralytics import settings

        settings.update(
            {
                "runs_dir": str(root / "runs"),
                "weights_dir": str(models_dir),
                "datasets_dir": str(root / "datasets"),
            }
        )
    except Exception:
        pass

    _move_root_ultralytics_weights_to_models()


_configure_ultralytics_dirs(VAILA_MODELS_DIR)

from ultralytics import YOLO  # noqa: E402

# Mandatory dual-import pattern (package + standalone execution).
try:
    from .geometric_reid import (
        GeometricFrameLinker,
        GeometricLinkerConfig,
        write_reid_links_csv,
    )
    from .hardware_manager import HardwareManager
except ImportError:  # pragma: no cover
    from geometric_reid import (  # ty: ignore[unresolved-import]
        GeometricFrameLinker,
        GeometricLinkerConfig,
        write_reid_links_csv,
    )
    from hardware_manager import HardwareManager  # ty: ignore[unresolved-import]

try:
    from PIL import Image, ImageTk
except ImportError as e:
    raise ImportError(
        "Missing dependency: Pillow. Install via `uv sync` (or add extra that provides it)."
    ) from e

try:
    import toml
except ImportError as e:
    raise ImportError(
        "Missing dependency: toml. Install via `uv sync` (or add dependency in pyproject)."
    ) from e


# Ensure BoxMOT can be found
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Configure to avoid library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)  # Limits the number of threads to avoid conflicts

# ReID model catalogue (reserved for future BoT-SORT weight picker).
# Live tracking uses Ultralytics BoT-SORT ``with_reid`` (yolo26n-cls.pt via custom YAML).
# Post-track appearance ReID uses OSNet in ``reid_yolotrack`` / GUI ``appearance_reid``.
REID_MODELS = {
    "lmbn_n_cuhk03_d.pt": "Lightweight (LMBN CUHK03)",
    "osnet_x0_25_market1501.pt": "Lightweight (OSNet x0.25 Market1501)",
    "mobilenetv2_x1_4_msmt17.engine": "Medium (MobileNetV2 MSMT17)",
    "resnet50_msmt17.onnx": "Medium (ResNet50 MSMT17)",
    "osnet_x1_0_msmt17.pt": "Medium (OSNet x1.0 MSMT17)",
    "clip_market1501.pt": "Heavy (CLIP Market1501)",
    "clip_vehicleid.pt": "Heavy (CLIP VehicleID)",
}


def _discover_tracking_csv_roots(root: str | Path, max_depth: int = 6) -> list[Path]:
    """Directories under root (bounded depth) that contain per-ID tracking CSVs."""
    root_p = Path(root).resolve()
    if not root_p.is_dir():
        return []
    hits: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root_p):
        rel = Path(dirpath).relative_to(root_p)
        if len(rel.parts) > max_depth:
            dirnames.clear()
            continue
        if len(rel.parts) >= max_depth:
            dirnames.clear()
        if any(
            fn.endswith(".csv") and "_id_" in fn and not fn.startswith("all_id_")
            for fn in filenames
        ):
            hits.append(Path(dirpath))
    return sorted(set(hits))


def _pick_tracking_leaf_dir(
    parent: tk.Misc, title: str, message: str, candidates: list[Path]
) -> Path | None:
    """Tk listbox chooser for multiple tracking CSV roots."""
    pick = tk.Toplevel(parent)
    pick.title(title)
    pick.geometry("560x380")
    pick.transient(cast(tk.Wm, parent))
    with contextlib.suppress(Exception):
        pick.grab_set()
    tk.Label(pick, text=message, pady=10).pack()
    lb = tk.Listbox(pick, width=82, height=12)
    for c in candidates:
        lb.insert(tk.END, str(c))
    lb.pack(padx=10, pady=10, fill="both", expand=True)
    chosen: list[Path] = []

    def _ok() -> None:
        sel = lb.curselection()
        if not sel:
            return
        chosen.append(Path(lb.get(sel[0])))
        pick.destroy()

    def _cancel() -> None:
        pick.destroy()

    bf = tk.Frame(pick)
    bf.pack(pady=10)
    tk.Button(bf, text="OK", command=_ok, width=10).pack(side="left", padx=6)
    tk.Button(bf, text="Cancel", command=_cancel, width=10).pack(side="left", padx=6)
    pick.wait_window()
    return chosen[0] if chosen else None


def resolve_tracking_dir_with_csvs(tracking_dir: str, parent: tk.Misc | None = None) -> str | None:
    """Resolve a user-selected root to the directory that holds ``*_id_*.csv`` files."""
    candidates = _discover_tracking_csv_roots(tracking_dir)
    if not candidates:
        return None
    if len(candidates) == 1:
        return str(candidates[0])

    created = False
    if parent is None or not parent.winfo_exists():
        parent = tk.Tk()
        parent.withdraw()
        created = True
    try:
        pick = _pick_tracking_leaf_dir(
            parent,
            "Select tracking folder",
            "Multiple folders with tracking CSVs were found. Pick the folder to use:",
            candidates,
        )
    finally:
        if created and parent.winfo_exists():
            parent.destroy()
    return str(pick) if pick else None


def _ffmpeg_temp_avi_to_h264_mp4(temp_avi: str, out_mp4: str) -> bool:
    """Convert MJPG AVI from OpenCV to H.264 MP4 (no audio). Returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        temp_avi,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        out_mp4,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not os.path.exists(out_mp4) or os.path.getsize(out_mp4) == 0:
            if proc.stderr:
                print(f"FFmpeg: {proc.stderr[:800]}")
            return False
        return True
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg.")
        return False


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

        # Run mode (track / seg / pose)
        run_frame = tk.LabelFrame(master, text="Run Mode", padx=5, pady=5)
        run_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        tk.Label(run_frame, text="Mode:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.run_mode_var = tk.StringVar(value="track+pose")
        run_mode_combo = ttk.Combobox(
            run_frame,
            textvariable=self.run_mode_var,
            values=[
                "track",
                "track+pose",
                "track+seg",
                "run_all (track+seg+pose)",
            ],
            state="readonly",
            width=22,
        )
        run_mode_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.save_masks_var = tk.BooleanVar(value=True)
        self.save_contours_var = tk.BooleanVar(value=True)
        tk.Checkbutton(run_frame, text="Save masks (PNG)", variable=self.save_masks_var).grid(
            row=1, column=0, padx=5, pady=2, sticky="w"
        )
        tk.Checkbutton(
            run_frame, text="Save contours (JSON)", variable=self.save_contours_var
        ).grid(row=1, column=1, padx=5, pady=2, sticky="w")

        self.stabilize_ids_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            run_frame,
            text="Geometric ID stabilize (Hungarian + velocity)",
            variable=self.stabilize_ids_var,
        ).grid(row=2, column=0, columnspan=2, padx=5, pady=2, sticky="w")

        tk.Label(run_frame, text="ReID max gap:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.reid_max_gap = tk.Entry(run_frame, width=8)
        self.reid_max_gap.insert(0, "12")
        self.reid_max_gap.grid(row=3, column=1, padx=5, pady=2, sticky="w")

        tk.Label(run_frame, text="ReID max dist px:").grid(
            row=3, column=2, padx=5, pady=2, sticky="w"
        )
        self.reid_max_dist = tk.Entry(run_frame, width=8)
        self.reid_max_dist.insert(0, "180")
        self.reid_max_dist.grid(row=3, column=3, padx=5, pady=2, sticky="w")

        tk.Label(run_frame, text="ReID min IoU:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.reid_min_iou = tk.Entry(run_frame, width=8)
        self.reid_min_iou.insert(0, "0.05")
        self.reid_min_iou.grid(row=4, column=1, padx=5, pady=2, sticky="w")

        tk.Label(run_frame, text="Direction weight:").grid(
            row=4, column=2, padx=5, pady=2, sticky="w"
        )
        self.reid_direction_weight = tk.Entry(run_frame, width=8)
        self.reid_direction_weight.insert(0, "0.5")
        self.reid_direction_weight.grid(row=4, column=3, padx=5, pady=2, sticky="w")

        self.appearance_reid_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            run_frame,
            text="Post-track OSNet appearance ReID (reid_yolotrack)",
            variable=self.appearance_reid_var,
        ).grid(row=5, column=0, columnspan=3, padx=5, pady=2, sticky="w")

        # Pose sub-config (used when mode includes pose)
        pose_frame = tk.LabelFrame(master, text="Pose (optional)", padx=5, pady=5)
        pose_frame.grid(row=6, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        tk.Label(pose_frame, text="Pose model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pose_model_var = tk.StringVar(value="yolo26n-pose.pt")
        pose_model_combo = ttk.Combobox(
            pose_frame,
            textvariable=self.pose_model_var,
            values=[
                "yolo26n-pose.pt",
                "yolo26s-pose.pt",
                "yolo26m-pose.pt",
                "yolo26l-pose.pt",
                "yolo26x-pose.pt",
            ],
            state="readonly",
            width=18,
        )
        pose_model_combo.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        tk.Label(pose_frame, text="Pose conf:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.pose_conf = tk.Entry(pose_frame, width=10)
        self.pose_conf.insert(0, "0.10")
        self.pose_conf.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        tk.Label(pose_frame, text="Pose IoU:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.pose_iou = tk.Entry(pose_frame, width=10)
        self.pose_iou.insert(0, "0.70")
        self.pose_iou.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        tk.Label(pose_frame, text="ROI pad %:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.pose_pad_pct = tk.Entry(pose_frame, width=10)
        self.pose_pad_pct.insert(0, "0.15")
        self.pose_pad_pct.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        tk.Label(pose_frame, text="Min ROI px:").grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.pose_min_roi = tk.Entry(pose_frame, width=10)
        self.pose_min_roi.insert(0, "256")
        self.pose_min_roi.grid(row=2, column=3, padx=5, pady=5, sticky="w")

        # ROI selection section
        tk.Label(master, text="Region of Interest (ROI):").grid(row=7, column=0, padx=5, pady=5)
        self.roi_file_path = None
        self.roi_status_label = tk.Label(master, text="No ROI selected", fg="gray")
        self.roi_status_label.grid(row=7, column=1, padx=5, pady=5, sticky="w")

        # Frame for ROI buttons
        roi_buttons_frame = tk.Frame(master)
        roi_buttons_frame.grid(row=8, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

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

        # Max tracked IDs (post-tracking rerank cap)
        tk.Label(master, text="Max tracked IDs (cap):").grid(
            row=9, column=0, padx=5, pady=5, sticky="w"
        )
        self.max_tracked_ids = tk.Entry(master, width=10)
        self.max_tracked_ids.insert(0, "0")
        self.max_tracked_ids.grid(row=9, column=1, padx=5, pady=5, sticky="w")
        help_text_max_ids = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text_max_ids.grid(row=9, column=2, padx=5, pady=5, sticky="w")
        max_ids_tooltip = (
            "Max tracked IDs (post-tracking rerank cap):\n"
            "0  - Disabled (keep all IDs from BoT-SORT)\n"
            "N  - After tracking, keep only the top-N IDs ranked\n"
            "     by number of frames detected (persistence).\n\n"
            "Recommended for football (soccer 11x11 + 4 referees):\n"
            "    22 to 26\n\n"
            "How it works:\n"
            "  1. First pass: BoT-SORT runs normally (yolo26x detector +\n"
            "     ReID via yolo26n-cls, with GMC) and writes the full\n"
            "     per-ID stream to memory.\n"
            "  2. Count frames per raw tracker ID across the whole video.\n"
            "  3. Keep the N most persistent raw IDs; re-map them to\n"
            "     stable sequential IDs (1..N) by persistence rank.\n"
            "  4. All other IDs (short tracklets, noise, ghost tracks)\n"
            "     are dropped (label them '?').\n\n"
            "Side effects:\n"
            "  - Output CSVs and annotated video reflect the cleaned IDs.\n"
            "  - Applied uniformly to detection, pose and segmentation\n"
            "     sub-trackers.\n"
            "  - Increases total processing time because results are\n"
            "     buffered and a second pass re-renders outputs."
        )
        help_text_max_ids.bind("<Enter>", lambda e: self.show_help(e, max_ids_tooltip))
        help_text_max_ids.bind("<Leave>", self.hide_help)

        help_text = tk.Label(master, text="?", cursor="hand2", fg="blue")
        help_text.grid(row=7, column=2, padx=5, pady=5)
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
                "run_mode": self.run_mode_var.get(),
                "save_masks": bool(self.save_masks_var.get()),
                "save_contours": bool(self.save_contours_var.get()),
                "pose_model_name": self.pose_model_var.get(),
                "pose_conf": float(self.pose_conf.get()),
                "pose_iou": float(self.pose_iou.get()),
                "pose_pad_pct": float(self.pose_pad_pct.get()),
                "pose_min_roi": int(self.pose_min_roi.get()),
                "stabilize_ids": bool(self.stabilize_ids_var.get()),
                "reid_max_gap": int(self.reid_max_gap.get()),
                "reid_max_dist": float(self.reid_max_dist.get()),
                "reid_min_iou": float(self.reid_min_iou.get()),
                "reid_direction_weight": float(self.reid_direction_weight.get()),
                "appearance_reid": bool(self.appearance_reid_var.get()),
                "roi_file": self.roi_file_path,  # Path to saved ROI TOML file
                "half": True,
                "persist": True,
                "verbose": False,
                "stream": True,
                "max_tracked_ids": int(self.max_tracked_ids.get()),
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
    """Optional BoT-SORT ReID weight picker (not wired in main GUI yet).

    BoT-SORT online ReID uses Ultralytics defaults; offline OSNet merge is via
    ``appearance_reid`` / ``reid_yolotrack.run_appearance_reid_on_tracking_dir``.
    """

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


# ---------------------------------------------------------------------------
# Post-tracking ID cap (global rerank)
# ---------------------------------------------------------------------------
# BoT-SORT with ReID emits many short tracklets (entry/exit, occlusions, the
# ball, the referee near the bench, etc.). When the user knows roughly how many
# real objects are in the scene (e.g. 22 players + 4 referees = 26 for soccer
# 11x11), we can keep only the N most persistent raw IDs and re-map them to
# stable sequential IDs 1..N by persistence rank. Everything else is dropped.
#
# Strategy: "Rerank global pós-track"
#   1. Buffer the full Ultralytics Results stream (frame index + per-detection
#      data + the rendered annotated frame, if any).
#   2. Walk through detections once and count how many frames each raw ID
#      survived.
#   3. Sort raw IDs by persistence (desc) and keep the top-N. The remainder is
#      treated as noise/short tracklets.
#   4. Build a mapping {raw_id -> new_id 1..N} in persistence order. The
#      rerank is **deterministic** (no ties broken by raw id) so two runs over
#      the same input give the same output.
#   5. Re-emit results frame by frame, applying the mapping. Detections whose
#      raw id is not in the top-N are kept only if `keep_dropped=True` (they
#      are re-labeled to -1 and a placeholder label "?"). Default: dropped from
#      CSVs and from the annotated frame to keep outputs clean.
#   6. ID cap = 0 disables the entire pass and just yields the original stream
#      unchanged (so the feature is opt-in).
# ---------------------------------------------------------------------------

POSE_KEYPOINT_NAMES: tuple[str, ...] = (
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
)


# Backward-compatible alias used by tests and legacy call sites.
_GeometricTrackLinker = GeometricFrameLinker


def _load_reid_homography_from_config(config: dict[str, Any]) -> np.ndarray | None:
    """Load optional pitch homography for geometric Re-ID distance gating."""
    if isinstance(config.get("reid_homography"), np.ndarray):
        return config["reid_homography"]
    path = config.get("reid_homography_path") or config.get("reid_homography_file")
    if not path:
        return None
    try:
        try:
            from .reid_markers import load_homography_matrix
        except ImportError:
            from reid_markers import load_homography_matrix  # ty: ignore[unresolved-import]
        return load_homography_matrix(str(path))
    except Exception as exc:
        print(f"[yolov26track] WARNING: could not load ReID homography from {path}: {exc}")
        return None


def _linker_config_from_dict(config: dict[str, Any]) -> GeometricLinkerConfig:
    homography = _load_reid_homography_from_config(config)
    return GeometricLinkerConfig(
        max_gap=int(config.get("reid_max_gap", 12)),
        max_centroid_dist_px=float(config.get("reid_max_dist", 180.0)),
        min_iou=float(config.get("reid_min_iou", 0.05)),
        direction_weight=float(config.get("reid_direction_weight", 0.5)),
        homography_matrix=homography,
        mask_iou_weight=float(config.get("reid_mask_iou_weight", 0.0)),
    )


def _build_botsort_custom_yaml(models_dir: str, tracker_name: str) -> str:
    """Write BoT-SORT YAML with GMC + appearance ReID (mirrors GUI path)."""
    trackers_dir = os.path.join(models_dir, "trackers")
    os.makedirs(trackers_dir, exist_ok=True)
    custom_yaml = os.path.join(trackers_dir, f"{tracker_name}_custom.yaml")
    tracker_cfg: dict[str, Any] = {
        "tracker_type": tracker_name,
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.6,
        "track_buffer": 60,
        "match_thresh": 0.8,
        "fuse_score": True,
    }
    if tracker_name == "botsort":
        tracker_cfg["gmc_method"] = "sparseOptFlow"
        tracker_cfg["with_reid"] = True
        tracker_cfg["proximity_thresh"] = 0.5
        tracker_cfg["appearance_thresh"] = 0.25
    with open(custom_yaml, "w") as fh:
        yaml.dump(tracker_cfg, fh, default_flow_style=False, sort_keys=False)
    return custom_yaml


def prepare_pose_roi(
    frame: np.ndarray,
    xyxy: tuple[int, int, int, int],
    *,
    pad_pct: float = 0.15,
    min_side: int = 256,
    max_side: int = 1280,
) -> tuple[np.ndarray, float, int, int]:
    """Crop bbox (with padding), upscale so min side >= min_side; return ROI + map params."""
    x_min, y_min, x_max, y_max = xyxy
    fh, fw = frame.shape[:2]
    bw = max(1, x_max - x_min)
    bh = max(1, y_max - y_min)
    pad_x = int(bw * pad_pct)
    pad_y = int(bh * pad_pct)
    x0 = max(0, x_min - pad_x)
    y0 = max(0, y_min - pad_y)
    x1 = min(fw, x_max + pad_x)
    y1 = min(fh, y_max + pad_y)
    roi = frame[y0:y1, x0:x1].copy()
    rh, rw = roi.shape[:2]
    scale = 1.0
    if min(rh, rw) < min_side:
        scale = min_side / float(min(rh, rw))
    if max(rh, rw) * scale > max_side:
        scale = min(scale, max_side / float(max(rh, rw)))
    if abs(scale - 1.0) > 1e-6:
        new_w = max(1, int(round(rw * scale)))
        new_h = max(1, int(round(rh * scale)))
        roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return roi, scale, x0, y0


def map_pose_keypoints_to_global(
    kps_roi: np.ndarray,
    scale: float,
    offset_x: int,
    offset_y: int,
) -> list[tuple[float, float, float]]:
    """Map ROI keypoints back to full-frame coordinates."""
    out: list[tuple[float, float, float]] = []
    inv_scale = 1.0 / scale if scale else 1.0
    for kp in kps_roi:
        x = float(kp[0]) * inv_scale + offset_x
        y = float(kp[1]) * inv_scale + offset_y
        conf = float(kp[2]) if len(kp) > 2 else 1.0
        out.append((x, y, conf))
    return out


def select_pose_person(kps_list: np.ndarray, roi_wh: tuple[int, int]) -> int:
    """Pick the pose detection whose centroid is closest to the ROI center."""
    rw, rh = roi_wh
    cx, cy = rw * 0.5, rh * 0.5
    best_i = 0
    best_dist = float("inf")
    for i, person in enumerate(kps_list):
        xs: list[float] = []
        ys: list[float] = []
        for kp in person:
            if len(kp) >= 3 and float(kp[2]) > 0.1:
                xs.append(float(kp[0]))
                ys.append(float(kp[1]))
        if not xs:
            continue
        px = sum(xs) / len(xs)
        py = sum(ys) / len(ys)
        dist = (px - cx) ** 2 + (py - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def _pose_csv_headers() -> list[str]:
    headers: list[str] = ["Frame", "Tracker_ID", "Label"]
    for name in POSE_KEYPOINT_NAMES:
        headers.extend([f"{name}_x", f"{name}_y", f"{name}_conf"])
    return headers


def _pose_row_from_keypoints(
    frame_idx: int,
    stable_id: int,
    label: str,
    abs_kps: list[tuple[float, float, float]] | None,
) -> list[Any]:
    row: list[Any] = [frame_idx, stable_id, label]
    if abs_kps is None:
        for _ in POSE_KEYPOINT_NAMES:
            row.extend([np.nan, np.nan, np.nan])
        return row
    for i, _ in enumerate(POSE_KEYPOINT_NAMES):
        if i < len(abs_kps):
            x, y, c = abs_kps[i]
            row.extend([x, y, c])
        else:
            row.extend([np.nan, np.nan, np.nan])
    return row


def _load_pose_model(pose_model_name: str, models_dir: str) -> Any | None:
    """Load YOLO pose weights; prefer .pt when ROI sizes vary (skip broken TRT)."""
    _configure_ultralytics_dirs(VAILA_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    pose_model_path = os.path.join(models_dir, pose_model_name)
    if not os.path.exists(pose_model_path):
        try:
            print(f"Downloading pose model {pose_model_name}...")
            current_dir = os.getcwd()
            os.chdir(models_dir)
            YOLO(pose_model_path)
            os.chdir(current_dir)
        except Exception as e:
            print(f"Failed to download pose model: {e}")
            return None
    hw = HardwareManager(models_dir=models_dir)
    try:
        pt_path = Path(models_dir) / pose_model_name
        if pt_path.is_file():
            pose_model_path = str(pt_path)
        else:
            exported = hw.auto_export(pose_model_name, imgsz=640)
            pose_model_path = str(pt_path) if str(exported).endswith(".engine") else str(exported)
        return YOLO(pose_model_path, task="pose")
    except TypeError:
        return YOLO(pose_model_path)
    except Exception as e:
        print(f"Failed to load pose model: {e}")
        return None


def _infer_pose_in_bbox(
    pose_model: Any,
    frame: np.ndarray,
    xyxy: tuple[int, int, int, int],
    *,
    device: str,
    conf: float,
    iou: float,
    pad_pct: float,
    min_side: int,
) -> list[tuple[float, float, float]] | None:
    """Run upscaled pose inside bbox; return global keypoints or None."""
    roi, scale, off_x, off_y = prepare_pose_roi(frame, xyxy, pad_pct=pad_pct, min_side=min_side)
    if roi.size == 0:
        return None
    results = pose_model.predict(
        roi,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
        show=False,
        save=False,
    )
    if not results or results[0].keypoints is None:
        return None
    kp_data = results[0].keypoints.data
    if kp_data is None or len(kp_data) == 0:
        return None
    if hasattr(kp_data, "cpu"):
        kp_data = cast(Any, kp_data).cpu().numpy()
    elif hasattr(kp_data, "numpy"):
        kp_data = cast(Any, kp_data).numpy()
    roi_h, roi_w = roi.shape[:2]
    person_idx = select_pose_person(kp_data, (roi_w, roi_h))
    return map_pose_keypoints_to_global(kp_data[person_idx], scale, off_x, off_y)


def _write_yolo_reid_links_csv(output_dir: str, links: list[tuple[int, int, int]]) -> str | None:
    if not links:
        return None
    path = os.path.join(output_dir, "yolo_reid_links.csv")
    write_reid_links_csv(path, links, ("frame", "raw_id", "stable_id"))
    return path


def _write_all_id_pose_csv(output_dir: str, rows: list[list[Any]]) -> str | None:
    if not rows:
        return None
    path = os.path.join(output_dir, "all_id_pose.csv")
    pd.DataFrame(rows, columns=cast(Any, _pose_csv_headers())).to_csv(path, index=False)
    return path


def _flush_pose_csv_buffers(
    output_dir: str,
    video_basename: str,
    pose_buffers: dict[tuple[str, int], list[list[Any]]],
) -> list[str]:
    written: list[str] = []
    for (_label, stable_id), rows in sorted(pose_buffers.items()):
        if not rows:
            continue
        out_path = os.path.join(output_dir, f"{video_basename}_id_{stable_id:02d}_pose.csv")
        pd.DataFrame(rows, columns=cast(Any, _pose_csv_headers())).to_csv(out_path, index=False)
        written.append(out_path)
    return written


@dataclass
class _BufferedFrame:
    """One frame worth of tracking data captured from a YOLO Results stream."""

    frame_idx: int
    detections: list[dict]  # one per detection in this frame
    annotated_frame: np.ndarray | None  # BGR (post-plot), or None to skip
    raw_result: Any | None  # original Ultralytics Results object (for re-plot)


@dataclass
class _LightweightTrackFrame:
    """Phase-2 write item: video frame re-read from disk + lightweight detections."""

    frame_idx: int
    frame: np.ndarray
    detections: list[dict]


def buffer_tracking_stream(
    results_iter: Any,
    save_annotated: bool = True,
    progress_cb: Any | None = None,
    *,
    keep_raw_result: bool = True,
    gpu_gc_every: int = 0,
) -> tuple[list[_BufferedFrame], dict[int, int]]:
    """Materialize a YOLO `Results` stream into per-frame detection records.

    Args:
        results_iter: iterable of Ultralytics `Results` (e.g. from
            ``model.track(..., stream=True)``).
        save_annotated: when True, also keep the BGR frame produced by
            ``result.plot()`` so we can re-emit the annotated video after the
            rerank.
        progress_cb: optional ``callable(frame_idx)`` for progress reporting.
        keep_raw_result: when False, only detection metadata is kept (no
            ``orig_img`` / GPU tensors) — required for long clips with ID cap.
        gpu_gc_every: when > 0, run :func:`_release_yolo_gpu_memory` every N
            buffered frames during CUDA inference.

    Returns:
        Tuple ``(buffer, id_counts)`` where ``buffer`` is the list of
        ``_BufferedFrame`` and ``id_counts`` maps ``raw_tracker_id`` ->
        number of frames where the id was observed.
    """
    buffer: list[_BufferedFrame] = []
    id_counts: dict[int, int] = {}
    gc_interval = max(0, int(gpu_gc_every))

    for frame_idx, result in enumerate(results_iter):
        detections: list[dict] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            raw_ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(raw_ids))
            clses = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(raw_ids))
            for raw_id, (x1, y1, x2, y2), conf, cls in zip(
                raw_ids, xyxy, confs, clses, strict=True
            ):
                rid = int(raw_id)
                detections.append(
                    {
                        "raw_id": rid,
                        "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                        "conf": float(conf),
                        "cls": int(cls),
                    }
                )
                id_counts[rid] = id_counts.get(rid, 0) + 1

        annotated = result.plot() if save_annotated else None
        raw_keep = result if keep_raw_result else None
        buffer.append(
            _BufferedFrame(
                frame_idx=frame_idx,
                detections=detections,
                annotated_frame=annotated,
                raw_result=raw_keep,
            )
        )
        if not keep_raw_result:
            with contextlib.suppress(Exception):
                result.orig_img = None
            del result
        if gc_interval > 0 and (frame_idx + 1) % gc_interval == 0:
            _release_yolo_gpu_memory()
        if progress_cb is not None:
            with contextlib.suppress(Exception):
                progress_cb(frame_idx)

    if gc_interval > 0:
        _release_yolo_gpu_memory()

    return buffer, id_counts


def _iter_idcap_write_frames(
    video_path: str,
    buffer: list[_BufferedFrame],
    *,
    vid_stride: int = 1,
) -> Any:
    """Re-read source video for ID-cap phase 2 (no in-RAM ``orig_img`` buffer)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for ID-cap write pass: {video_path}")
    stride = max(1, int(vid_stride))
    try:
        for bf in buffer:
            src_idx = bf.frame_idx * stride
            if stride > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(src_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                _track_log(
                    f"ID-cap write pass: stopped at buffered frame {bf.frame_idx} (read failed)"
                )
                break
            yield _LightweightTrackFrame(
                frame_idx=bf.frame_idx,
                frame=frame,
                detections=bf.detections,
            )
    finally:
        cap.release()


def build_id_rerank_map(id_counts: dict[int, int], max_ids: int) -> dict[int, int]:
    """Build ``{raw_id -> new_id 1..N}`` keeping the top-N most persistent.

    Args:
        id_counts: mapping ``raw_id -> frame_count`` produced by
            :func:`buffer_tracking_stream`.
        max_ids: maximum number of IDs to keep. ``<= 0`` returns an empty
            mapping (cap disabled).

    Returns:
        Dictionary from raw id to a new sequential id starting at 1. Order is
        by ``(frame_count desc, raw_id asc)`` for determinism.
    """
    if max_ids <= 0 or not id_counts:
        return {}

    sorted_ids = sorted(id_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = sorted_ids[:max_ids]
    return {raw_id: new_id for new_id, (raw_id, _cnt) in enumerate(kept, start=1)}


def rerank_buffered_stream(
    buffer: list[_BufferedFrame],
    rerank_map: dict[int, int],
) -> list[_BufferedFrame]:
    """Return a copy of ``buffer`` with each detection's ``raw_id`` re-mapped.

    Detections whose raw id is not in ``rerank_map`` are dropped (default
    behavior of "Rerank global pós-track"). ``annotated_frame`` is left as-is;
    callers that need a clean re-render should re-plot from the re-mapped
    detections.
    """
    if not rerank_map:
        return buffer

    out: list[_BufferedFrame] = []
    for bf in buffer:
        kept: list[dict] = []
        for det in bf.detections:
            new_id = rerank_map.get(det["raw_id"])
            if new_id is None:
                continue
            kept.append({**det, "raw_id": new_id})
        out.append(
            _BufferedFrame(
                frame_idx=bf.frame_idx,
                detections=kept,
                annotated_frame=bf.annotated_frame,
                raw_result=bf.raw_result,
            )
        )
    return out


def rewrite_ultralytics_boxes_id(
    result: Any,
    rerank_map: dict[int, int],
) -> Any:
    """In-place rewrite of ``result.boxes.id`` using ``rerank_map``.

    Detections whose raw id is not in the map have their box **removed** from
    ``result.boxes`` so the rendered annotated frame does not show ghost
    tracks. Returns the (possibly modified) result.
    """
    if not rerank_map:
        return result
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None:
        return result
    import torch as _torch

    raw_ids = boxes.id.int().cpu().tolist()
    new_ids: list[int] = []
    keep_mask: list[bool] = []
    for rid in raw_ids:
        mapped = rerank_map.get(int(rid))
        if mapped is None:
            keep_mask.append(False)
            new_ids.append(-1)
        else:
            keep_mask.append(True)
            new_ids.append(mapped)
    if all(keep_mask):
        boxes.id = _torch.as_tensor(new_ids, dtype=_torch.int32, device=boxes.id.device)
        return result
    keep_idx = [i for i, k in enumerate(keep_mask) if k]
    if not keep_idx:
        boxes.id = _torch.zeros((0, 0), dtype=_torch.int32, device=boxes.id.device)
        return result
    keep_tensor = _torch.as_tensor(keep_idx, dtype=_torch.long, device=boxes.id.device)
    boxes.xyxy = boxes.xyxy[keep_tensor]
    if boxes.conf is not None:
        boxes.conf = boxes.conf[keep_tensor]
    if boxes.cls is not None:
        boxes.cls = boxes.cls[keep_tensor]
    boxes.id = _torch.as_tensor(
        [new_ids[i] for i in keep_idx], dtype=_torch.int32, device=boxes.id.device
    )
    return result


def create_combined_detection_csv(output_dir):
    """
    Creates a combined CSV file with detection data organized by ID columns.
    Each detected object gets their own set of columns (ID_n, X_n, Y_n, RGB_n).

    Args:
        output_dir: Directory containing the individual detection CSV files

    Returns:
        Path to the created combined CSV file
    """
    # Find all detection CSV files (any class with _id pattern); exclude merged vailá exports.
    detection_csv_files = [
        f
        for f in glob.glob(os.path.join(output_dir, "*_id_*.csv"))
        if not os.path.basename(f).startswith("all_id_")
    ]

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
            df = pd.read_csv(csv_file, usecols=("Frame",))  # ty: ignore[no-matching-overload]  # faster
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
    csv_files = [
        f
        for f in glob.glob(os.path.join(output_dir, "*_id_*.csv"))
        if not os.path.basename(f).startswith("all_id_")
    ]

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


def _console_hint(msg: str) -> None:
    """Print to real stdout (bypass Rich) and flush so CLI users see progress during GUI waits."""
    out = sys.__stdout__
    if out is not None:
        out.write(f"{msg}\n")
        out.flush()


def _track_log(message: str, *, flush: bool = True) -> None:
    """Emit a tracking progress line (``>>`` prefix survives absl logging)."""
    out = sys.__stdout__
    if out is not None:
        out.write(f">> yolov26track: {message}\n")
        if flush:
            out.flush()


def _track_banner(title: str, detail: str = "") -> None:
    """Boxed banner before long-running tracking phases."""
    out = sys.__stdout__
    if out is None:
        return
    bar = "=" * 70
    out.write(f"\n{bar}\n")
    out.write(f">> yolov26track: {title}\n")
    if detail:
        out.write(f"   {detail}\n")
    out.write(f"{bar}\n")
    out.flush()


def _release_yolo_gpu_memory() -> None:
    """Best-effort CUDA release between long YOLO track passes (SAM3-style).

      Ultralytics keeps per-frame tensors and allocator pools across a streaming
    ``model.track`` loop.  Periodic ``gc`` + ``empty_cache`` avoids VRAM climbing
      until the driver OOMs on long broadcast clips.
    """
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()


def _memory_snapshot() -> dict[str, float]:
    """Non-throwing VRAM + host RAM probe for terminal logging."""
    snap: dict[str, float] = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        snap["ram_available_gib"] = float(vm.available) / (1024**3)
        snap["ram_used_pct"] = float(vm.percent)
    except Exception:
        pass
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            free_b, total_b = torch.cuda.mem_get_info()
            snap["vram_free_gib"] = float(free_b) / (1024**3)
            snap["vram_total_gib"] = float(total_b) / (1024**3)
    return snap


def _log_memory_status(label: str) -> None:
    """Log VRAM (nvidia-smi aligned) and host RAM for operator monitoring."""
    snap = _memory_snapshot()
    if not snap:
        return
    parts: list[str] = []
    if "vram_free_gib" in snap and "vram_total_gib" in snap:
        parts.append(f"VRAM {snap['vram_free_gib']:.1f}/{snap['vram_total_gib']:.1f} GiB free")
    if "ram_available_gib" in snap:
        pct = snap.get("ram_used_pct", 0.0)
        parts.append(f"RAM {snap['ram_available_gib']:.1f} GiB avail ({pct:.0f}% used)")
    if parts:
        _track_log(f"{label}: " + " | ".join(parts))


PROGRESS_FRAME_CHUNK = 100
PROGRESS_MAX_LINES = 40


def _progress_chunk_size(total_frames: int) -> int:
    """Return frame interval between terminal progress lines (multiples of 100)."""
    chunk = PROGRESS_FRAME_CHUNK
    if total_frames <= chunk:
        return chunk
    # Long broadcast clips: cap terminal updates (~40 lines) instead of one per 100 frames.
    coarse = (total_frames + PROGRESS_MAX_LINES - 1) // PROGRESS_MAX_LINES
    if coarse <= chunk:
        return chunk
    return ((coarse + chunk - 1) // chunk) * chunk


def _emit_frame_progress(
    frame_idx: int,
    total_frames: int,
    *,
    phase: str,
    chunk_frames: int = PROGRESS_FRAME_CHUNK,
) -> None:
    """Emit summarized progress at chunk boundaries and on the last frame."""
    current = frame_idx + 1
    if total_frames > 0:
        at_chunk = current % chunk_frames == 0
        at_end = current >= total_frames
        if not at_chunk and not at_end:
            return
        pct = min(100.0, 100.0 * current / total_frames)
        _track_log(f"{phase}: {pct:.0f}% ({current}/{total_frames})")
    elif current % chunk_frames == 0:
        _track_log(f"{phase}: frame {current}")


def _make_frame_progress_logger(total_frames: int, phase: str, chunk_frames: int | None = None):
    """Build a ``progress_cb`` for :func:`buffer_tracking_stream`."""
    chunk = chunk_frames if chunk_frames is not None else _progress_chunk_size(total_frames)

    def _cb(frame_idx: int) -> None:
        _emit_frame_progress(frame_idx, total_frames, phase=phase, chunk_frames=chunk)

    return _cb


def run_yolov26pose_video(parent: tk.Misc | None = None) -> None:
    """Pose inference direct from video (no tracking required).

    Pass ``parent`` when called from an existing Tk app (e.g. main ``vaila.py``).
    A second ``tk.Tk()`` would deadlock or hide file dialogs on Linux.
    """
    _setup_run_logging("yolov26pose_video")
    _configure_ultralytics_dirs(VAILA_MODELS_DIR)

    created_root = False
    root: tk.Misc
    if parent is not None and parent.winfo_exists():
        root = parent
    else:
        existing = getattr(tk, "_default_root", None)
        if existing is not None and existing.winfo_exists():
            root = cast(Any, existing)
        else:
            root = tk.Tk()
            root.withdraw()
            created_root = True

    _console_hint("[pose] Open file dialog: select input video (check taskbar if nothing appears).")

    video_path = filedialog.askopenfilename(
        parent=root,
        title="Select Video for YOLOv26 Pose",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
            ("All files", "*.*"),
        ],
    )
    if not video_path:
        messagebox.showinfo("YOLOv26 Pose", "No video selected.")
        if created_root and isinstance(root, tk.Tk):
            root.destroy()
        return

    _console_hint("[pose] Open folder dialog: select output directory.")
    output_base_dir = filedialog.askdirectory(parent=root, title="Select Output Directory")
    if not output_base_dir:
        messagebox.showinfo("YOLOv26 Pose", "Pose run cancelled (no output directory).")
        if created_root and isinstance(root, tk.Tk):
            root.destroy()
        return

    # Quick config dialog
    cfg = tk.Toplevel(root)
    cfg.title("YOLOv26 Pose Inference")
    cfg.geometry("420x260")
    cfg.transient(cast(tk.Wm, root))
    with contextlib.suppress(Exception):
        cfg.grab_set()

    tk.Label(cfg, text="Pose model:").pack(pady=(12, 2))
    model_var = tk.StringVar(value="yolo26n-pose.pt")
    ttk.Combobox(
        cfg,
        textvariable=model_var,
        values=[
            "yolo26n-pose.pt",
            "yolo26s-pose.pt",
            "yolo26m-pose.pt",
            "yolo26l-pose.pt",
            "yolo26x-pose.pt",
        ],
        state="readonly",
        width=22,
    ).pack()

    tk.Label(cfg, text="Pose conf (0-1):").pack(pady=(10, 2))
    conf_entry = tk.Entry(cfg, width=10)
    conf_entry.insert(0, "0.10")
    conf_entry.pack()

    tk.Label(cfg, text="Pose IoU (0-1):").pack(pady=(10, 2))
    iou_entry = tk.Entry(cfg, width=10)
    iou_entry.insert(0, "0.70")
    iou_entry.pack()

    tk.Label(cfg, text="Device (cpu/cuda/mps):").pack(pady=(10, 2))
    dev_var = tk.StringVar(value=detect_optimal_device())
    ttk.Combobox(
        cfg, textvariable=dev_var, values=["cpu", "cuda", "mps"], state="readonly", width=10
    ).pack()

    result: dict[str, Any] = {}
    cancelled: list[bool] = [False]

    def _on_close() -> None:
        cancelled[0] = True
        cfg.destroy()

    cfg.protocol("WM_DELETE_WINDOW", _on_close)

    def _ok() -> None:
        try:
            result["pose_model_name"] = model_var.get()
            result["pose_conf"] = float(conf_entry.get())
            result["pose_iou"] = float(iou_entry.get())
            result["device"] = dev_var.get()
        except Exception as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
            return
        cfg.destroy()

    tk.Button(cfg, text="Run", command=_ok, width=12).pack(pady=14)
    cfg.wait_window()
    if cancelled[0] or not result:
        messagebox.showinfo("YOLOv26 Pose", "Pose run cancelled (close window or incomplete Run).")
        if created_root and isinstance(root, tk.Tk):
            root.destroy()
        return

    _print_pose_video_equivalent_cli(
        video_path=video_path,
        output_base_dir=output_base_dir,
        pose_model=cast(str, result["pose_model_name"]),
        pose_conf=float(result["pose_conf"]),
        pose_iou=float(result["pose_iou"]),
        device=cast(str, result["device"]),
    )

    video_stem = Path(video_path).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_base_dir) / f"{video_stem}_pose_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = str(VAILA_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    pose_model_name = cast(str, result["pose_model_name"])
    pose_model_path = os.path.join(models_dir, pose_model_name)
    if not os.path.exists(pose_model_path):
        try:
            cur = os.getcwd()
            os.chdir(models_dir)
            YOLO(pose_model_path)
            os.chdir(cur)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download pose model: {e}")
            if created_root and isinstance(root, tk.Tk):
                root.destroy()
            return

    hw = HardwareManager(models_dir=models_dir)
    pose_model_path = hw.auto_export(pose_model_name, imgsz=640)
    if str(pose_model_path).endswith(".engine"):
        p = Path(pose_model_path)
        with contextlib.suppress(OSError):
            if p.exists() and p.stat().st_size == 0:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                with contextlib.suppress(Exception):
                    p.replace(p.with_suffix(f".broken_{ts}.engine"))
                pose_model_path = str(Path(models_dir) / f"{Path(pose_model_name).stem}.pt")
                _console_hint(f"[pose] Zero-byte engine skipped; using PT: {pose_model_path}")
    try:
        pose_model = YOLO(pose_model_path, task="pose")
    except TypeError:
        pose_model = YOLO(pose_model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video:\n{video_path}")
        if created_root and isinstance(root, tk.Tk):
            root.destroy()
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_csv = out_dir / f"{video_stem}_pose.csv"
    pose_video = out_dir / f"{video_stem}_pose.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(str(pose_video), fourcc, fps, (frame_w, frame_h))

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
    headers: list[str] = ["Frame", "Tracker_ID", "Label"]
    for kp in keypoint_names:
        headers.extend([f"{kp}_x", f"{kp}_y", f"{kp}_conf"])

    rows: list[list[Any]] = []
    frame_idx = 0
    _console_hint(f"[pose] Processing {total_frames} frames → {out_dir}")
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model.predict(
            frame,
            conf=float(result["pose_conf"]),
            iou=float(result["pose_iou"]),
            device=cast(str, result["device"]),
            verbose=False,
            show=False,
            save=False,
        )

        row: list[Any] = [frame_idx, 1, "pose"]
        abs_kps: list[tuple[float, float, float]] = []
        if results and len(results) > 0 and results[0].keypoints is not None:
            kp_data = results[0].keypoints.data
            if kp_data is not None and len(kp_data) > 0:
                if hasattr(kp_data, "cpu"):
                    kp_data = cast(Any, kp_data).cpu().numpy()
                elif hasattr(kp_data, "numpy"):
                    kp_data = cast(Any, kp_data).numpy()
                kps = kp_data[0]
                for i in range(len(keypoint_names)):
                    if i < len(kps):
                        x = float(kps[i][0])
                        y = float(kps[i][1])
                        c = float(kps[i][2]) if len(kps[i]) > 2 else 1.0
                        row.extend([x, y, c])
                        abs_kps.append((x, y, c))
                    else:
                        row.extend([np.nan, np.nan, np.nan])
                        abs_kps.append((np.nan, np.nan, 0.0))
            else:
                for _ in keypoint_names:
                    row.extend([np.nan, np.nan, np.nan])
                abs_kps = [(np.nan, np.nan, 0.0) for _ in keypoint_names]
        else:
            for _ in keypoint_names:
                row.extend([np.nan, np.nan, np.nan])
            abs_kps = [(np.nan, np.nan, 0.0) for _ in keypoint_names]

        rows.append(row)
        frame = _draw_keypoints_and_skeleton(frame, abs_kps, color=(0, 255, 0))
        writer.write(frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"[pose] frame {frame_idx}/{total_frames}", end="\r")

    cap.release()
    writer.release()

    pd.DataFrame(rows, columns=cast(Any, headers)).to_csv(pose_csv, index=False)
    messagebox.showinfo("YOLOv26 Pose", f"Done.\n\nCSV:\n{pose_csv}\n\nVideo:\n{pose_video}")
    if created_root and isinstance(root, tk.Tk):
        root.destroy()


def _mask_to_polygons(mask_u8: np.ndarray) -> list[list[list[int]]]:
    """Binary mask (H,W uint8 {0,255}) -> polygons [[x,y], ...] list."""
    if mask_u8.ndim != 2:
        raise ValueError("mask_u8 must be 2D")
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[list[list[int]]] = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # cnt shape (N,1,2) -> (N,2)
        pts = cnt.reshape(-1, 2)
        polys.append([[int(x), int(y)] for x, y in pts])
    return polys


def _to_mask_u8(mask: Any) -> np.ndarray:
    """Ultralytics mask tensor/array -> uint8 0/255 mask."""
    if hasattr(mask, "cpu"):
        mask = cast(Any, mask).cpu().numpy()
    elif hasattr(mask, "numpy"):
        mask = cast(Any, mask).numpy()
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0.5).astype(np.uint8) * 255
    else:
        # assume 0/1 or 0/255
        if m.max() <= 1:
            m = m * 255
    if m.ndim == 3 and m.shape[0] == 1:
        # (N,H,W) caller should pass single slice, but accept N=1
        m = m[0]
    return m


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
    _setup_run_logging("yolov26pose_from_tracking")
    _configure_ultralytics_dirs(VAILA_MODELS_DIR)
    # Prefer existing Tk root to avoid multiple roots (pyimage errors); create only if needed
    created_root = False
    root = getattr(tk, "_default_root", None)
    if root is None or not isinstance(root, tk.Tk):
        root = tk.Tk()
        root.withdraw()
        created_root = True
    print(f"[pose] Using root: created={created_root}, exists={root.winfo_exists()}")

    # Select tracking directory (accept vailatracker_* root or per-video subdir)
    tracking_dir = filedialog.askdirectory(title="Select Tracking Directory")
    if not tracking_dir:
        if created_root and root.winfo_exists():
            root.destroy()
        return

    resolved = resolve_tracking_dir_with_csvs(tracking_dir, parent=root)
    if not resolved:
        messagebox.showerror(
            "Error",
            f"No tracking CSV files (*_id_*.csv) found under:\n{tracking_dir}",
        )
        if created_root and root.winfo_exists():
            root.destroy()
        return
    tracking_dir = resolved

    _print_pose_from_tracking_workflow_hint(tracking_dir)

    csv_files = glob.glob(os.path.join(tracking_dir, "*_id_*.csv"))
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
    canvas.image = img  # ty: ignore[unresolved-attribute] # Keep a reference (dynamic attr for gc)

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

    _configure_ultralytics_dirs(VAILA_MODELS_DIR)

    # Load pose model (download into vaila/models/)
    models_dir = str(VAILA_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    pose_model_path = os.path.join(models_dir, pose_model_name)

    # Download model if needed
    if not os.path.exists(pose_model_path):
        try:
            print(f"Downloading pose model {pose_model_name}...")
            current_dir = os.getcwd()
            os.chdir(models_dir)
            # Specify full path to YOLO to avoid downloading to root
            YOLO(pose_model_path)
            os.chdir(current_dir)
            print("Model downloaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download pose model: {str(e)}")
            return False

    # Initialize Hardware Manager
    hw = HardwareManager(models_dir=models_dir)

    try:
        # Check and auto-export optimized .engine model
        pose_model_path = hw.auto_export(pose_model_name, imgsz=640)
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
    parent_ui = getattr(tk, "_default_root", None)
    resolve_parent: tk.Misc | None = None
    if parent_ui is not None and parent_ui.winfo_exists():
        resolve_parent = cast(tk.Misc, parent_ui)

    resolved_tracking = resolve_tracking_dir_with_csvs(tracking_dir, parent=resolve_parent)
    if not resolved_tracking:
        print(f"Error: No tracking CSV files found under {tracking_dir}")
        messagebox.showerror("Error", f"No tracking CSV files found under:\n{tracking_dir}")
        return False
    tracking_dir = resolved_tracking

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
    video_path = pick_video_for_pose(tracking_dir, parent=parent_ui)
    if not video_path:
        print(f"Error: No video found in {tracking_dir}")
        messagebox.showerror("Error", f"No video found in:\n{tracking_dir}")
        return False
    print(f"Using video: {os.path.basename(video_path)}")
    print(f"Found {len(csv_files)} tracking CSV files")

    # Load pose model
    _configure_ultralytics_dirs(VAILA_MODELS_DIR)

    models_dir = str(VAILA_MODELS_DIR)
    os.makedirs(models_dir, exist_ok=True)
    pose_model_path = os.path.join(models_dir, pose_model_name)

    # Download model if needed
    if not os.path.exists(pose_model_path):
        try:
            print(f"Downloading pose model {pose_model_name}...")
            current_dir = os.getcwd()
            os.chdir(models_dir)
            # Specify full path to YOLO to avoid downloading to root
            YOLO(pose_model_path)
            os.chdir(current_dir)
            print(f"Model downloaded successfully to {pose_model_path}")
        except Exception as e:
            print(f"Error downloading pose model: {e}")
            messagebox.showerror("Error", f"Failed to download pose model: {str(e)}")
            return False

    # Initialize Hardware Manager
    hw = HardwareManager(models_dir=models_dir)

    try:
        # Check and auto-export optimized .engine model
        pose_model_path = hw.auto_export(pose_model_name, imgsz=640)
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
                        keypoints_row: list[Any] = [frame_idx, tracker_id, label]

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
                        keypoints_row: list[Any] = [frame_idx, tracker_id, label]
                        for _ in keypoint_names:
                            keypoints_row.extend([np.nan, np.nan, np.nan])
                        pose_data.append(keypoints_row)
                else:
                    # No bbox for this frame, fill with NaN
                    keypoints_row: list[Any] = [frame_idx, tracker_id, label]
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
    _setup_run_logging("yolov26track")
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    print("Starting yolov26track.py...")
    print("-" * 80)

    _configure_ultralytics_dirs(VAILA_MODELS_DIR)

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

    models_dir = str(VAILA_MODELS_DIR)

    is_custom_model = _is_custom_model_path(model_name)

    # Handle model path based on whether it's a custom model or pre-trained
    if is_custom_model:
        model_path = os.path.abspath(os.path.expanduser(model_name))
        print(f"Using custom model: {model_path}")

        if not os.path.isfile(model_path):
            messagebox.showerror("Error", f"Custom model file not found: {model_path}")
            return
    else:
        # Pre-trained model - build the path in models directory
        # Models are downloaded to vaila/models/ directory
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_name)

        # Download the model if it doesn't exist
        if not os.path.exists(model_path):
            try:
                print(f"Downloading model {model_name}...")
                current_dir = os.getcwd()
                os.chdir(models_dir)
                # Specify full path to YOLO to avoid downloading to root
                YOLO(model_path)
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

    run_mode = config.get("run_mode", "track")
    do_pose = run_mode in {"track+pose", "run_all (track+seg+pose)"}
    do_seg = run_mode in {"track+seg", "run_all (track+seg+pose)"}
    save_masks = bool(config.get("save_masks", True))
    save_contours = bool(config.get("save_contours", True))

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
    hw = HardwareManager(models_dir=models_dir)
    hw.print_report()

    def _guess_task(name: str) -> str:
        low = name.lower()
        if "-seg" in low or "segment" in low:
            return "segment"
        if "-pose" in low or "pose" in low:
            return "pose"
        return "detect"

    pt_fallback_path = (
        model_path if is_custom_model else str(Path(models_dir) / f"{Path(model_name).stem}.pt")
    )

    try:
        if is_custom_model:
            task = _guess_task(model_path)
            model = YOLO(model_path, task=task)
            print(f"Model loaded successfully: {model_path} (task={task}, custom weights)")
        else:
            # Auto-export if needed (creates .engine optimized for this GPU)
            model_path = hw.auto_export(model_name, imgsz=640)
            # Guard: some failed exports leave a 0-byte engine, which will crash inside `track()`.
            if str(model_path).endswith(".engine"):
                p = Path(model_path)
                with contextlib.suppress(OSError):
                    if p.exists() and p.stat().st_size == 0:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        broken_dst = p.with_suffix(f".broken_{ts}.engine")
                        with contextlib.suppress(Exception):
                            p.replace(broken_dst)
                            print(f"[warning] Zero-byte TensorRT engine moved to: {broken_dst}")
                        model_path = pt_fallback_path
                        print(f"[warning] Falling back to PT weights: {model_path}")
            task = _guess_task(model_name)
            # Ultralytics sometimes cannot infer task for TensorRT engines; pass explicit task.
            model = YOLO(model_path, task=task)
            print(f"Model loaded successfully: {model_path} (task={task})")
    except TypeError:
        # Back-compat if installed Ultralytics does not accept `task=` kwarg.
        if is_custom_model:
            model = YOLO(model_path)
        else:
            model_path = hw.auto_export(model_name, imgsz=640)
            model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path} (task=auto)")
    except Exception as e:
        # TensorRT engines can become stale/corrupt across Ultralytics/TensorRT updates.
        # When that happens, AutoBackend(tensorrt) may fail decoding embedded JSON metadata.
        mp = str(model_path) if "model_path" in locals() else ""
        if mp.endswith(".engine") and isinstance(e, json.JSONDecodeError):
            broken = Path(mp)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            broken_dst = broken.with_suffix(f".broken_{ts}.engine")
            with contextlib.suppress(Exception):
                broken.replace(broken_dst)
                print(f"[warning] Corrupt TensorRT engine moved to: {broken_dst}")

            # Fall back to PT weights (no TensorRT) so tracking can run.
            pt_path = Path(pt_fallback_path)
            try:
                task = _guess_task(model_path if is_custom_model else model_name)
                model = YOLO(str(pt_path), task=task)
                model_path = str(pt_path)
                print(f"[warning] Falling back to PT weights: {pt_path} (task={task})")
            except Exception:
                logging.exception("Failed to fall back to PT after engine JSONDecodeError")
                messagebox.showerror(
                    "Error",
                    "Failed to load TensorRT engine (corrupt metadata) and PT fallback also failed.\n\n"
                    f"Engine: {mp}\nPT: {pt_path}\n\nError: {e}",
                )
                return
        else:
            logging.exception("Failed to load YOLO model")
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            return

    # Select classes for tracking
    class_dialog = ClassSelectorDialog(root, title="Select Classes for Tracking")
    if not hasattr(class_dialog, "result"):
        return

    target_classes = class_dialog.result

    # Collect videos and print GUI→CLI mirror before processing
    pending_videos: list[tuple[str, str]] = []
    for video_file in os.listdir(video_dir):
        if video_file.endswith((".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")):
            video_path = os.path.abspath(os.path.join(video_dir, video_file))
            if not os.path.isfile(video_path):
                continue
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(main_output_dir, video_name)
            pending_videos.append((video_path, output_dir))

    if pending_videos:
        print("\n>> vaila/yolov26track: Equivalent CLI per video (copy/paste):", flush=True)
        tracker_cli = tracker_name if tracker_name.endswith(".yaml") else f"{tracker_name}.yaml"
        for video_path, output_dir in pending_videos:
            cmd = _format_track_cli_command(
                model_path=model_path,
                source=video_path,
                output_dir=output_dir,
                tracker=tracker_cli,
                config=config,
                target_classes=target_classes,
                do_pose=do_pose,
            )
            print(f">>   {cmd}", flush=True)
        print(
            ">> Note: GUI batch-processes a folder; CLI `track` runs one video per command.\n",
            flush=True,
        )

    # Count videos to process
    video_count = 0
    processed_count = 0

    # Process each video in the directory
    for video_file in os.listdir(video_dir):
        if video_file.endswith((".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")):
            video_count += 1
            video_path = os.path.abspath(os.path.join(video_dir, video_file))
            if not os.path.isfile(video_path):
                print(f"[yolov26track] Skipping missing video: {video_path}")
                continue
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"[yolov26track] Tracking video: {video_path}")

            # Create a subdirectory for this specific video
            output_dir = os.path.join(main_output_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            seg_masks_dir = Path(output_dir) / "yolo_masks"
            if do_seg and save_masks:
                seg_masks_dir.mkdir(parents=True, exist_ok=True)
            mask_manifest_rows: list[str] = ["frame,id,area,mask_png"]
            contours_out: dict[str, Any] = {
                "schema": "vaila_yolo_contours_v1",
                "frames": [],
            }

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
            seg_video_path = os.path.join(output_dir, f"processed_{video_name}_seg.mp4")
            all_video_path = os.path.join(output_dir, f"processed_{video_name}_all.mp4")
            temp_seg_avi_path = os.path.join(output_dir, f"processed_{video_name}_seg_temp.avi")
            temp_all_avi_path = os.path.join(output_dir, f"processed_{video_name}_all_temp.avi")
            track_pose_overlay_path = os.path.join(
                output_dir, f"{video_name}_track_pose_overlay.mp4"
            )
            temp_pose_avi_path = os.path.join(output_dir, f"{video_name}_track_pose_temp.avi")

            # Use MJPG codec for AVI (highly compatible and reliable)
            # This ensures the video is written correctly without corruption
            writer = cv2.VideoWriter(
                temp_avi_path,
                cv2.VideoWriter_fourcc(*"MJPG"),  # ty: ignore[unresolved-attribute]
                fps,
                (width, height),
            )
            if not writer.isOpened():
                print(f"Error creating video file: {temp_avi_path}")
                print("Trying alternative codec...")
                # Fallback to XVID if MJPG fails
                print("[WARNING] MJPG failed. Trying XVID...")
                writer = cv2.VideoWriter(
                    temp_avi_path,
                    cv2.VideoWriter_fourcc(*"XVID"),  # ty: ignore[unresolved-attribute]
                    fps,
                    (width, height),
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
                        # Fallback: try to find it manually or skip
                        pass
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

            print(f"[yolov26track] Tracking video: {video_path}")

            pose_model = None
            if do_pose:
                pose_model = _load_pose_model(
                    str(config.get("pose_model_name", "yolo26n-pose.pt")),
                    models_dir,
                )
                if pose_model is None:
                    messagebox.showerror(
                        "Error",
                        f"Failed to load pose model: {config.get('pose_model_name')}",
                    )
                    continue
                print(f"Pose model loaded for inline inference: {config.get('pose_model_name')}")

            geo_linker = GeometricFrameLinker(
                enabled=bool(config.get("stabilize_ids", True)),
                config=_linker_config_from_dict(config),
            )
            pose_buffers: dict[tuple[str, int], list[list[Any]]] = {}
            all_pose_rows: list[list[Any]] = []
            pose_pad_pct = float(config.get("pose_pad_pct", 0.15))
            pose_min_roi = int(config.get("pose_min_roi", 256))
            pose_conf = float(config.get("pose_conf", 0.10))
            pose_iou = float(config.get("pose_iou", 0.70))

            _track_banner(
                "Starting video tracking",
                f"video={os.path.basename(video_path)} | frames≈{total_frames} | "
                f"tracker={tracker_name} | device={config['device']}",
            )
            if max_tracked_ids := int(config.get("max_tracked_ids", 0) or 0):
                _track_log(
                    f"ID cap enabled (top-{max_tracked_ids}): phase 1 = YOLO inference, "
                    "phase 2 = write overlays/CSVs"
                )
            else:
                _track_log("Streaming track + writing overlays/CSVs in one pass")

            tracker_csv_files = {}
            # Map raw tracker IDs (from YOLO) to sequential per-label IDs starting at 1
            label_to_raw2seq = {}
            label_to_next = {}

            track_kwargs: dict[str, Any] = {
                "source": str(video_path),
                "conf": config["conf"],
                "iou": config["iou"],
                "device": config["device"],
                "vid_stride": config["vid_stride"],
                "save": False,
                "stream": True,
                "persist": True,
                "tracker": tracker_config,
                "verbose": False,
            }
            if target_classes is not None:
                track_kwargs["classes"] = target_classes

            _track_log(
                "Running YOLO track inference — progress every "
                f"~{_progress_chunk_size(total_frames)} frames on stdout"
            )

            try:
                results = model.track(**track_kwargs)
            except json.JSONDecodeError as e:
                # AutoBackend(TensorRT) can throw JSONDecodeError if engine metadata is empty/corrupt.
                if str(model_path).endswith(".engine"):
                    broken = Path(model_path)
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    broken_dst = broken.with_suffix(f".broken_{ts}.engine")
                    with contextlib.suppress(Exception):
                        broken.replace(broken_dst)
                        print(f"[warning] Corrupt TensorRT engine moved to: {broken_dst}")
                    pt_path = Path(pt_fallback_path)
                    task = _guess_task(model_path if is_custom_model else model_name)
                    model = YOLO(str(pt_path), task=task)
                    model_path = str(pt_path)
                    print(f"[warning] Retrying tracking with PT weights: {pt_path} (task={task})")
                    results = model.track(**track_kwargs)
                else:
                    raise e

            # Post-tracking ID cap (global rerank).
            write_progress_chunk = _progress_chunk_size(total_frames)
            if max_tracked_ids > 0:
                progress_chunk = write_progress_chunk
                _log_memory_status("Before phase 1/2")
                _track_log(
                    f"Phase 1/2 — buffering detections only (keep top-{max_tracked_ids} IDs; "
                    "video re-read in phase 2)..."
                )
                _buffer, _id_counts = buffer_tracking_stream(
                    results,
                    save_annotated=False,
                    keep_raw_result=False,
                    gpu_gc_every=progress_chunk,
                    progress_cb=_make_frame_progress_logger(
                        total_frames,
                        "Track inference",
                        chunk_frames=progress_chunk,
                    ),
                )
                del results
                _release_yolo_gpu_memory()
                _log_memory_status("After phase 1/2 buffer")
                _track_log(
                    f"Inference buffered {len(_buffer)} frames, "
                    f"{len(_id_counts)} unique raw tracker IDs"
                )
                rerank_map = build_id_rerank_map(_id_counts, max_tracked_ids)
                if rerank_map:
                    kept_ids = sorted(rerank_map.values())
                    _track_log(
                        f"ID cap: kept raw ids {sorted(rerank_map.keys())} "
                        f"-> new ids {kept_ids} (frames per id: "
                        f"{[(rid, _id_counts[rid]) for rid in sorted(rerank_map.keys())]})"
                    )
                else:
                    _track_log("ID cap: no IDs to keep (empty stream).")
                _reranked_buffer = rerank_buffered_stream(_buffer, rerank_map)
                del _buffer, _id_counts, rerank_map
                _release_yolo_gpu_memory()
                results = _iter_idcap_write_frames(
                    video_path,
                    _reranked_buffer,
                    vid_stride=int(config.get("vid_stride", 1) or 1),
                )
                label_to_raw2seq = {"__rerank__": {}}
                label_to_next = {"__rerank__": 1}
                _rerank_active = True
                _track_log("Phase 2/2 — writing annotated video and per-ID CSVs...")
            else:
                _rerank_active = False

            output_phase = "Writing output" if max_tracked_ids > 0 else "Track+write"
            frame_idx = 0
            stream_gc_every = write_progress_chunk if max_tracked_ids <= 0 else 0
            seg_writer: cv2.VideoWriter | None = None
            all_writer: cv2.VideoWriter | None = None
            if do_seg:
                seg_writer = cv2.VideoWriter(
                    temp_seg_avi_path,
                    cv2.VideoWriter_fourcc(*"MJPG"),  # ty: ignore[unresolved-attribute]
                    fps,
                    (width, height),
                )
                if not seg_writer.isOpened():
                    seg_writer = cv2.VideoWriter(
                        temp_seg_avi_path,
                        cv2.VideoWriter_fourcc(*"XVID"),  # ty: ignore[unresolved-attribute]
                        fps,
                        (width, height),
                    )
            if do_seg or do_pose:
                all_writer = cv2.VideoWriter(
                    temp_all_avi_path,
                    cv2.VideoWriter_fourcc(*"MJPG"),  # ty: ignore[unresolved-attribute]
                    fps,
                    (width, height),
                )
                if not all_writer.isOpened():
                    all_writer = cv2.VideoWriter(
                        temp_all_avi_path,
                        cv2.VideoWriter_fourcc(*"XVID"),  # ty: ignore[unresolved-attribute]
                        fps,
                        (width, height),
                    )
            pose_overlay_writer: cv2.VideoWriter | None = None
            if do_pose:
                pose_overlay_writer = cv2.VideoWriter(
                    temp_pose_avi_path,
                    cv2.VideoWriter_fourcc(*"MJPG"),  # ty: ignore[unresolved-attribute]
                    fps,
                    (width, height),
                )
                if not pose_overlay_writer.isOpened():
                    pose_overlay_writer = cv2.VideoWriter(
                        temp_pose_avi_path,
                        cv2.VideoWriter_fourcc(*"XVID"),  # ty: ignore[unresolved-attribute]
                        fps,
                        (width, height),
                    )
            for write_item in results:
                lightweight = isinstance(write_item, _LightweightTrackFrame)
                if lightweight:
                    frame = write_item.frame
                    frame_idx = write_item.frame_idx
                else:
                    frame = write_item.orig_img
                frame_seg = frame.copy() if do_seg else frame
                frame_all = frame.copy() if (do_seg or do_pose) else frame
                frame_contours: dict[str, Any] | None = None
                masks_data = None
                if not lightweight and do_seg and getattr(write_item, "masks", None) is not None:
                    masks_data = getattr(write_item.masks, "data", None)
                    if masks_data is not None and hasattr(masks_data, "cpu"):
                        masks_data = cast(Any, masks_data).cpu().numpy()

                # Overlay ROI outline for reference
                if roi_poly is not None:
                    cv2.polylines(frame, [roi_poly], True, (255, 255, 0), 2)
                    if do_seg:
                        cv2.polylines(frame_seg, [roi_poly], True, (255, 255, 0), 2)
                    if do_seg or do_pose:
                        cv2.polylines(frame_all, [roi_poly], True, (255, 255, 0), 2)

                if lightweight:
                    det_list = write_item.detections
                else:
                    boxes = (
                        write_item.boxes
                        if write_item.boxes is not None
                        else getattr(write_item, "obbs", None)
                    )
                    det_list = None

                if lightweight and not det_list:
                    writer.write(frame)
                    if seg_writer is not None:
                        seg_writer.write(frame_seg)
                    if all_writer is not None:
                        all_writer.write(frame_all)
                    if pose_overlay_writer is not None:
                        pose_overlay_writer.write(frame_all)
                    _emit_frame_progress(
                        frame_idx,
                        total_frames,
                        phase=output_phase,
                        chunk_frames=write_progress_chunk,
                    )
                    frame_idx += 1
                    continue

                if not lightweight and boxes is None:
                    writer.write(frame)
                    if seg_writer is not None:
                        seg_writer.write(frame_seg)
                    if all_writer is not None:
                        all_writer.write(frame_all)
                    if pose_overlay_writer is not None:
                        pose_overlay_writer.write(frame_all)
                    _emit_frame_progress(
                        frame_idx,
                        total_frames,
                        phase=output_phase,
                        chunk_frames=write_progress_chunk,
                    )
                    frame_idx += 1
                    if stream_gc_every > 0 and frame_idx % stream_gc_every == 0:
                        _release_yolo_gpu_memory()
                    continue

                if lightweight:
                    det_iter: Any = enumerate(det_list)
                else:
                    det_iter = enumerate(cast(Any, boxes))

                frame_dets: list[dict[str, Any]] = []
                for det_i, box in det_iter:
                    if lightweight:
                        det = cast(dict[str, Any], box)
                        x_min, y_min, x_max, y_max = map(int, det["xyxy"])
                        conf = float(det["conf"])
                        raw_id = int(det["raw_id"])
                        class_id = int(det["cls"])
                    else:
                        box = cast(Any, box)
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        conf = box.conf[0].item()
                        raw_id = int(box.id[0]) if box.id is not None else -1
                        class_id = int(box.cls[0].item()) if box.cls is not None else -1
                    label = model.names.get(class_id, "unknown")

                    if raw_id < 0:
                        continue

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
                    if _rerank_active:
                        tracker_id = raw_id
                    else:
                        if raw_id not in label_to_raw2seq[label]:
                            label_to_raw2seq[label][raw_id] = label_to_next[label]
                            label_to_next[label] += 1
                        tracker_id = label_to_raw2seq[label][raw_id]

                    frame_dets.append(
                        {
                            "raw_id": raw_id,
                            "tracker_id": tracker_id,
                            "xyxy": (x_min, y_min, x_max, y_max),
                            "conf": conf,
                            "label": label,
                            "det_i": det_i,
                        }
                    )

                if geo_linker.enabled:
                    frame_dets = geo_linker.assign_frame(frame_idx, frame_dets)
                else:
                    for det in frame_dets:
                        det["stable_id"] = det["tracker_id"]

                for det in frame_dets:
                    stable_id = int(det["stable_id"])
                    x_min, y_min, x_max, y_max = det["xyxy"]
                    conf = float(det["conf"])
                    label = str(det["label"])
                    det_i = int(det["det_i"])
                    color = get_color_for_id(stable_id)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    if do_seg:
                        cv2.rectangle(frame_seg, (x_min, y_min), (x_max, y_max), color, 2)
                    if do_seg or do_pose:
                        cv2.rectangle(frame_all, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(
                        frame,
                        f"id {stable_id} {label}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    if do_seg:
                        cv2.putText(
                            frame_seg,
                            f"id {stable_id} {label}",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )
                    if do_seg or do_pose:
                        cv2.putText(
                            frame_all,
                            f"id {stable_id} {label}",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                        )

                    key = (stable_id, label)
                    if key not in tracker_csv_files:
                        tracker_csv_files[key] = initialize_csv(
                            output_dir,
                            label,
                            stable_id,
                            total_frames,
                        )
                    update_csv(
                        tracker_csv_files[key],
                        frame_idx,
                        stable_id,
                        label,
                        x_min,
                        y_min,
                        x_max,
                        y_max,
                        conf,
                    )

                    if do_pose and pose_model is not None:
                        abs_kps = _infer_pose_in_bbox(
                            pose_model,
                            frame,
                            (x_min, y_min, x_max, y_max),
                            device=str(config["device"]),
                            conf=pose_conf,
                            iou=pose_iou,
                            pad_pct=pose_pad_pct,
                            min_side=pose_min_roi,
                        )
                        pose_row = _pose_row_from_keypoints(frame_idx, stable_id, label, abs_kps)
                        pose_key = (label, stable_id)
                        pose_buffers.setdefault(pose_key, []).append(pose_row)
                        all_pose_rows.append(pose_row)
                        if abs_kps:
                            frame_all = _draw_keypoints_and_skeleton(
                                frame_all, abs_kps, color=color
                            )

                    if do_seg and masks_data is not None and det_i < len(masks_data):
                        mask_u8 = _to_mask_u8(masks_data[det_i])
                        area_px = int(np.count_nonzero(mask_u8))
                        mask_rel = ""
                        if save_masks:
                            mask_name = f"frame_{frame_idx:06d}_id_{stable_id:02d}.png"
                            mask_path = seg_masks_dir / mask_name
                            cv2.imwrite(str(mask_path), mask_u8)
                            mask_rel = str(mask_path.relative_to(Path(output_dir)))
                            mask_manifest_rows.append(
                                f"{frame_idx},{stable_id},{area_px},{mask_rel}"
                            )

                        if save_contours:
                            polys = _mask_to_polygons(mask_u8)
                            if polys:
                                if frame_contours is None:
                                    frame_contours = {"frame": frame_idx, "objects": []}
                                obj_entry: dict[str, Any] = {
                                    "id": stable_id,
                                    "obj_id": stable_id,
                                    "label": label,
                                    "bbox_xyxy": [x_min, y_min, x_max, y_max],
                                    "area_px": area_px,
                                    "polygons": polys,
                                }
                                if save_masks and mask_rel:
                                    obj_entry["mask_png"] = mask_rel
                                frame_contours["objects"].append(obj_entry)

                        # Overlay segmentation mask on seg/all videos
                        colored = np.zeros_like(frame, dtype=np.uint8)
                        colored[:, :, 1] = mask_u8  # green channel
                        alpha = 0.35
                        frame_seg = cv2.addWeighted(frame_seg, 1.0, colored, alpha, 0.0)
                        frame_all = cv2.addWeighted(frame_all, 1.0, colored, alpha, 0.0)

                writer.write(frame)
                if seg_writer is not None:
                    seg_writer.write(frame_seg)
                if all_writer is not None:
                    all_writer.write(frame_all)
                if pose_overlay_writer is not None:
                    pose_overlay_writer.write(frame_all)
                if frame_contours is not None:
                    contours_out["frames"].append(frame_contours)

                _emit_frame_progress(
                    frame_idx,
                    total_frames,
                    phase=output_phase,
                    chunk_frames=write_progress_chunk,
                )

                frame_idx += 1
                if stream_gc_every > 0 and frame_idx % stream_gc_every == 0:
                    _release_yolo_gpu_memory()

            _release_yolo_gpu_memory()
            if max_tracked_ids > 0:
                _log_memory_status("After phase 2/2 write")

            writer.release()
            if seg_writer is not None:
                seg_writer.release()
                if (
                    do_seg
                    and os.path.exists(temp_seg_avi_path)
                    and os.path.getsize(temp_seg_avi_path) > 0
                ):
                    print("Converting seg overlay to MP4...")
                    if _ffmpeg_temp_avi_to_h264_mp4(temp_seg_avi_path, seg_video_path):
                        os.remove(temp_seg_avi_path)
                    else:
                        print(f"Seg FFmpeg failed; keeping {temp_seg_avi_path}")
            if all_writer is not None:
                all_writer.release()
                if (
                    (do_seg or do_pose)
                    and os.path.exists(temp_all_avi_path)
                    and os.path.getsize(temp_all_avi_path) > 0
                ):
                    print("Converting combined overlay to MP4...")
                    if _ffmpeg_temp_avi_to_h264_mp4(temp_all_avi_path, all_video_path):
                        os.remove(temp_all_avi_path)
                    else:
                        print(f"Combined FFmpeg failed; keeping {temp_all_avi_path}")
            if pose_overlay_writer is not None:
                pose_overlay_writer.release()
                if os.path.exists(temp_pose_avi_path) and os.path.getsize(temp_pose_avi_path) > 0:
                    print("Converting track+pose overlay to MP4...")
                    if _ffmpeg_temp_avi_to_h264_mp4(temp_pose_avi_path, track_pose_overlay_path):
                        os.remove(temp_pose_avi_path)
                    else:
                        print(f"Pose overlay FFmpeg failed; keeping {temp_pose_avi_path}")
            if frame_idx > 0 and total_frames > 0:
                _track_log(f"{output_phase}: 100% ({frame_idx}/{total_frames}) — done writing AVI")
            _track_log(f"Wrote {frame_idx} frames to {temp_avi_path}")

            if do_seg:
                if save_masks and len(mask_manifest_rows) > 1:
                    (Path(output_dir) / "yolo_masks_manifest.csv").write_text(
                        "\n".join(mask_manifest_rows) + "\n", encoding="utf-8"
                    )
                if save_contours and contours_out["frames"]:
                    oids: set[int] = set()
                    for fr in contours_out["frames"]:
                        for obj in fr.get("objects") or []:
                            oid = obj.get("obj_id", obj.get("id"))
                            if isinstance(oid, int):
                                oids.add(oid)
                    contours_out["frames"].sort(key=lambda d: int(d.get("frame", 0)))
                    contours_out["video"] = os.path.basename(video_path)
                    contours_out["width"] = int(width)
                    contours_out["height"] = int(height)
                    contours_out["fps"] = float(fps)
                    contours_out["n_frames"] = int(total_frames)
                    contours_out["object_ids"] = sorted(oids)
                    (Path(output_dir) / "yolo_contours.json").write_text(
                        json.dumps(contours_out, ensure_ascii=False) + "\n", encoding="utf-8"
                    )

            # Verify the AVI file was written successfully
            if not os.path.exists(temp_avi_path) or os.path.getsize(temp_avi_path) == 0:
                print(f"Error: Video file {temp_avi_path} was not created or is empty")
                continue

            # Convert AVI to MP4 using FFmpeg with robust settings for VLC compatibility
            _track_log("Converting annotated AVI to MP4 (FFmpeg)...")
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

            if config.get("stabilize_ids", True) and geo_linker.reid_links:
                links_path = _write_yolo_reid_links_csv(output_dir, geo_linker.reid_links)
                if links_path:
                    print(f"Geometric Re-ID links saved: {links_path}")

            if config.get("appearance_reid"):
                try:
                    from .reid_yolotrack import run_appearance_reid_on_tracking_dir
                except ImportError:
                    from reid_yolotrack import run_appearance_reid_on_tracking_dir  # ty: ignore
                print("[yolov26track] Running OSNet appearance ReID (post geometric stabilize)...")
                run_appearance_reid_on_tracking_dir(
                    output_dir,
                    video_file,
                    threshold=float(config.get("appearance_reid_threshold", 0.6)),
                )

            if do_pose:
                all_pose_path = _write_all_id_pose_csv(output_dir, all_pose_rows)
                if all_pose_path:
                    print(f"Combined pose CSV saved: {all_pose_path}")
                per_id_pose = _flush_pose_csv_buffers(output_dir, video_name, pose_buffers)
                for pose_csv_path in per_id_pose:
                    print(f"  Per-ID pose CSV: {os.path.basename(pose_csv_path)}")
                if track_pose_overlay_path and os.path.isfile(track_pose_overlay_path):
                    print(f"Track+pose overlay video: {track_pose_overlay_path}")
                pose_model = None
                _release_yolo_gpu_memory()

            processed_count += 1

            # Legacy offline pose pass removed — pose runs inline in the tracking loop.

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


_BBOX_ANCHOR_ALIASES_CLI: dict[str, str] = {
    "c": "center",
    "centre": "center",
    "middle": "center",
    "foot": "bottom",
    "feet": "bottom",
    "bottom-center": "bottom",
    "bottom-centre": "bottom",
    "head": "top",
    "top-center": "top",
    "top-centre": "top",
}


def _anchor_xy_from_bbox_cli(
    x1: float, y1: float, x2: float, y2: float, anchor: str
) -> tuple[float, float]:
    """Pixel anchor point of a bbox (mirrors getpixelvideo ``_anchor_xy_from_bbox``).

    ``center`` (centroid), ``bottom`` (foot/bottom-center), ``top`` (head),
    ``left``, ``right``, and the four corners. Used to reduce each tracked box
    to ONE marker point for REC2D/REC3D.
    """
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    a = _BBOX_ANCHOR_ALIASES_CLI.get(
        (anchor or "center").strip().lower(), (anchor or "center").strip().lower()
    )
    if a == "center":
        return cx, cy
    if a == "bottom":
        return cx, y2
    if a == "top":
        return cx, y1
    if a == "left":
        return x1, cy
    if a == "right":
        return x2, cy
    if a == "top-left":
        return x1, y1
    if a == "top-right":
        return x2, y1
    if a == "bottom-left":
        return x1, y2
    if a == "bottom-right":
        return x2, y2
    return cx, cy


def _write_markers_csv_from_buffer(
    buffer: list[_BufferedFrame],
    output_dir: str,
    video_stem: str,
    anchor: str = "bottom",
) -> tuple[str, int]:
    """Write a getpixelvideo-style ``<stem>_markers.csv`` for REC2D/REC3D.

    Format: ``frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y`` — exactly the vailá
    pixel-coordinate header that ``rec2d.py`` / ``rec3d.py`` consume. Each
    tracked ID becomes one stable marker slot (sorted by ID); empty cells are
    blank (NaN) so missing detections do not corrupt the reconstruction.
    """
    total_frames = len(buffer)
    oids = sorted({int(det["raw_id"]) for bf in buffer for det in bf.detections})
    slot_by_oid = {oid: i for i, oid in enumerate(oids)}
    n_slots = len(oids)

    arr = np.full((total_frames, n_slots * 2), np.nan, dtype=np.float64)
    for bf in buffer:
        for det in bf.detections:
            slot = slot_by_oid[int(det["raw_id"])]
            x1, y1, x2, y2 = det["xyxy"]
            x, y = _anchor_xy_from_bbox_cli(float(x1), float(y1), float(x2), float(y2), anchor)
            arr[bf.frame_idx, slot * 2] = x
            arr[bf.frame_idx, slot * 2 + 1] = y

    columns: list[str] = []
    for i in range(n_slots):
        columns.append(f"p{i + 1}_x")
        columns.append(f"p{i + 1}_y")
    df = pd.DataFrame(arr, columns=pd.Index(columns))
    df.insert(0, "frame", np.arange(total_frames, dtype=np.int64))

    out_path = os.path.join(output_dir, f"{video_stem}_markers.csv")
    df.to_csv(out_path, index=False, float_format="%.3f")
    return out_path, n_slots


def _write_per_id_csvs_from_buffer(
    buffer: list[_BufferedFrame],
    output_dir: str,
    class_name: str,
    names: dict[int, str] | None = None,
) -> list[str]:
    """Write one ``{label}_id_NN.csv`` per tracked ID from a buffered stream.

    Vectorised (NumPy pre-fill, single ``to_csv`` per ID) so 16k-frame clips
    write in well under a second — unlike the legacy per-frame ``update_csv``.
    Output schema matches the GUI tracker so ``create_combined_detection_csv``
    and ``getpixelvideo`` consume the files unchanged.
    """
    total_frames = len(buffer)
    id_rows: dict[int, list[tuple[int, float, float, float, float, float]]] = {}
    id_label: dict[int, str] = {}
    for bf in buffer:
        for det in bf.detections:
            rid = int(det["raw_id"])
            x1, y1, x2, y2 = det["xyxy"]
            id_rows.setdefault(rid, []).append(
                (bf.frame_idx, float(x1), float(y1), float(x2), float(y2), float(det["conf"]))
            )
            if rid not in id_label:
                cls = det.get("cls")
                if names and cls is not None and int(cls) in names:
                    id_label[rid] = str(names[int(cls)])
                else:
                    id_label[rid] = class_name

    written: list[str] = []
    for rid in sorted(id_rows):
        label = id_label.get(rid, class_name)
        color = get_color_for_id(rid)  # BGR
        color_r, color_g, color_b = color[2], color[1], color[0]
        n = total_frames
        x_min = np.full(n, np.nan)
        y_min = np.full(n, np.nan)
        x_max = np.full(n, np.nan)
        y_max = np.full(n, np.nan)
        conf = np.full(n, np.nan)
        for f_idx, x1, y1, x2, y2, c in id_rows[rid]:
            x_min[f_idx] = x1
            y_min[f_idx] = y1
            x_max[f_idx] = x2
            y_max[f_idx] = y2
            conf[f_idx] = c
        df = pd.DataFrame(
            {
                "Frame": np.arange(n),
                "Tracker ID": np.full(n, rid),
                "Label": [label] * n,
                "X_min": x_min,
                "Y_min": y_min,
                "X_max": x_max,
                "Y_max": y_max,
                "Confidence": conf,
                "Color_R": np.full(n, color_r),
                "Color_G": np.full(n, color_g),
                "Color_B": np.full(n, color_b),
            }
        )
        out_path = os.path.join(output_dir, f"{label}_id_{rid:02d}.csv")
        df.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def _export_overlay_video_from_buffer(
    buffer: list[_BufferedFrame],
    output_dir: str,
    video_stem: str,
    fps: float,
) -> str | None:
    """Write annotated frames to an **H.264 .mp4** (small, compatible).

    Primary path: MJPG into a temp ``.avi`` placed in the OS temp dir, then
    FFmpeg → H.264 ``.mp4``; the bulky AVI is always deleted. If FFmpeg is
    missing, fall back to OpenCV ``mp4v`` writing straight to ``.mp4``. The
    output directory therefore never keeps a large ``.avi``.
    """
    import tempfile

    frames = [bf.annotated_frame for bf in buffer if bf.annotated_frame is not None]
    if not frames:
        return None
    height, width = frames[0].shape[:2]
    out_mp4 = os.path.join(output_dir, f"{video_stem}_track_overlay.mp4")
    fps_val = float(fps or 30.0)

    tmp_dir = tempfile.mkdtemp(prefix="vaila_track_")
    temp_avi = os.path.join(tmp_dir, f"{video_stem}_track_tmp.avi")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # ty: ignore[unresolved-attribute]
        writer = cv2.VideoWriter(temp_avi, fourcc, fps_val, (width, height))
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        if _ffmpeg_temp_avi_to_h264_mp4(temp_avi, out_mp4):
            return out_mp4

        # FFmpeg unavailable/failed: write .mp4 directly with OpenCV (mp4v).
        print("[yolov26track] FFmpeg unavailable; encoding .mp4 directly with OpenCV mp4v.")
        fourcc_mp4 = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
        writer = cv2.VideoWriter(out_mp4, fourcc_mp4, fps_val, (width, height))
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        if os.path.exists(out_mp4) and os.path.getsize(out_mp4) > 0:
            return out_mp4
        return None
    finally:
        with contextlib.suppress(Exception):
            if os.path.exists(temp_avi):
                os.remove(temp_avi)
            os.rmdir(tmp_dir)


def _flush_stable_bbox_csvs(
    id_rows: dict[tuple[str, int], list[tuple[int, float, float, float, float, float]]],
    output_dir: str,
    total_frames: int,
) -> list[str]:
    written: list[str] = []
    for (label, stable_id), rows in sorted(id_rows.items()):
        color = get_color_for_id(stable_id)
        color_r, color_g, color_b = color[2], color[1], color[0]
        n = total_frames
        x_min_a = np.full(n, np.nan)
        y_min_a = np.full(n, np.nan)
        x_max_a = np.full(n, np.nan)
        y_max_a = np.full(n, np.nan)
        conf_a = np.full(n, np.nan)
        for f_idx, x1, y1, x2, y2, c in rows:
            x_min_a[f_idx] = x1
            y_min_a[f_idx] = y1
            x_max_a[f_idx] = x2
            y_max_a[f_idx] = y2
            conf_a[f_idx] = c
        df = pd.DataFrame(
            {
                "Frame": np.arange(n),
                "Tracker ID": np.full(n, stable_id),
                "Label": [label] * n,
                "X_min": x_min_a,
                "Y_min": y_min_a,
                "X_max": x_max_a,
                "Y_max": y_max_a,
                "Confidence": conf_a,
                "Color_R": np.full(n, color_r),
                "Color_G": np.full(n, color_g),
                "Color_B": np.full(n, color_b),
            }
        )
        out_path = os.path.join(output_dir, f"{label}_id_{stable_id:02d}.csv")
        df.to_csv(out_path, index=False)
        written.append(out_path)
    return written


def _apply_geometric_stabilize_to_buffer(
    buffer: list[_BufferedFrame],
    output_dir: str,
    class_name: str,
    names_map: dict[int, str] | None,
    *,
    stabilize_ids: bool,
    linker_config: GeometricLinkerConfig | None = None,
) -> tuple[list[str], str | None]:
    """Apply geometric linker to buffered tracks; write stable per-ID bbox CSVs."""
    total_frames = len(buffer)
    geo_linker = GeometricFrameLinker(
        enabled=stabilize_ids,
        config=linker_config or GeometricLinkerConfig(),
    )
    label_to_raw2seq: dict[str, dict[int, int]] = {}
    label_to_next: dict[str, int] = {}
    id_rows: dict[tuple[str, int], list[tuple[int, float, float, float, float, float]]] = {}

    for bf in buffer:
        frame_dets: list[dict[str, Any]] = []
        for det in bf.detections:
            raw_id = int(det["raw_id"])
            x1, y1, x2, y2 = det["xyxy"]
            cls = det.get("cls")
            if names_map and cls is not None and int(cls) in names_map:
                label = str(names_map[int(cls)])
            else:
                label = class_name
            if label not in label_to_raw2seq:
                label_to_raw2seq[label] = {}
                label_to_next[label] = 1
            if raw_id not in label_to_raw2seq[label]:
                label_to_raw2seq[label][raw_id] = label_to_next[label]
                label_to_next[label] += 1
            tracker_id = label_to_raw2seq[label][raw_id]
            frame_dets.append(
                {
                    "raw_id": raw_id,
                    "tracker_id": tracker_id,
                    "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(det["conf"]),
                    "label": label,
                }
            )

        if geo_linker.enabled:
            frame_dets = geo_linker.assign_frame(bf.frame_idx, frame_dets)
        else:
            for d in frame_dets:
                d["stable_id"] = d["tracker_id"]

        for det in frame_dets:
            stable_id = int(det["stable_id"])
            x_min, y_min, x_max, y_max = det["xyxy"]
            label = str(det["label"])
            key = (label, stable_id)
            id_rows.setdefault(key, []).append(
                (
                    bf.frame_idx,
                    float(x_min),
                    float(y_min),
                    float(x_max),
                    float(y_max),
                    float(det["conf"]),
                )
            )

    written = _flush_stable_bbox_csvs(id_rows, output_dir, total_frames)
    links_path = _write_yolo_reid_links_csv(output_dir, geo_linker.reid_links)
    return written, links_path


def _emit_track_pose_from_buffer(
    buffer: list[_BufferedFrame],
    video_path: str,
    output_dir: str,
    video_stem: str,
    *,
    device: str,
    names_map: dict[int, str] | None,
    class_name: str,
    pose_model_name: str,
    pose_conf: float,
    pose_iou: float,
    pose_pad_pct: float,
    pose_min_roi: int,
    stabilize_ids: bool,
    linker_config: GeometricLinkerConfig | None = None,
    fps: float,
    save_overlay: bool = True,
) -> dict[str, Any]:
    """Single-pass pose + optional geometric Re-ID from a buffered track stream."""
    models_dir = str(VAILA_MODELS_DIR)
    pose_model = _load_pose_model(pose_model_name, models_dir)
    if pose_model is None:
        return {}

    total_frames = len(buffer)
    geo_linker = GeometricFrameLinker(
        enabled=stabilize_ids,
        config=linker_config or GeometricLinkerConfig(),
    )
    label_to_raw2seq: dict[str, dict[int, int]] = {}
    label_to_next: dict[str, int] = {}
    pose_buffers: dict[tuple[str, int], list[list[Any]]] = {}
    all_pose_rows: list[list[Any]] = []
    id_rows: dict[tuple[str, int], list[tuple[int, float, float, float, float, float]]] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[yolov26track] ERROR: cannot open video for pose pass: {video_path}")
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_pose_avi = os.path.join(output_dir, f"{video_stem}_track_pose_temp.avi")
    track_pose_overlay_path = os.path.join(output_dir, f"{video_stem}_track_pose_overlay.mp4")
    writer: cv2.VideoWriter | None = None
    if save_overlay:
        writer = cv2.VideoWriter(
            temp_pose_avi,
            cv2.VideoWriter_fourcc(*"MJPG"),  # ty: ignore[unresolved-attribute]
            fps if fps > 0 else 25.0,
            (width, height),
        )
        if not writer.isOpened():
            writer = cv2.VideoWriter(
                temp_pose_avi,
                cv2.VideoWriter_fourcc(*"XVID"),  # ty: ignore[unresolved-attribute]
                fps if fps > 0 else 25.0,
                (width, height),
            )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for bf in buffer:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        frame_dets: list[dict[str, Any]] = []
        for det in bf.detections:
            raw_id = int(det["raw_id"])
            x1, y1, x2, y2 = det["xyxy"]
            x_min, y_min, x_max, y_max = int(x1), int(y1), int(x2), int(y2)
            cls = det.get("cls")
            if names_map and cls is not None and int(cls) in names_map:
                label = str(names_map[int(cls)])
            else:
                label = class_name
            if label not in label_to_raw2seq:
                label_to_raw2seq[label] = {}
                label_to_next[label] = 1
            if raw_id not in label_to_raw2seq[label]:
                label_to_raw2seq[label][raw_id] = label_to_next[label]
                label_to_next[label] += 1
            tracker_id = label_to_raw2seq[label][raw_id]
            frame_dets.append(
                {
                    "raw_id": raw_id,
                    "tracker_id": tracker_id,
                    "xyxy": (x_min, y_min, x_max, y_max),
                    "conf": float(det["conf"]),
                    "label": label,
                }
            )

        if geo_linker.enabled:
            frame_dets = geo_linker.assign_frame(bf.frame_idx, frame_dets)
        else:
            for det in frame_dets:
                det["stable_id"] = det["tracker_id"]

        for det in frame_dets:
            stable_id = int(det["stable_id"])
            x_min, y_min, x_max, y_max = det["xyxy"]
            label = str(det["label"])
            conf = float(det["conf"])
            color = get_color_for_id(stable_id)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(
                overlay,
                f"id {stable_id} {label}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
            key = (label, stable_id)
            id_rows.setdefault(key, []).append(
                (bf.frame_idx, float(x_min), float(y_min), float(x_max), float(y_max), conf)
            )
            abs_kps = _infer_pose_in_bbox(
                pose_model,
                frame,
                (x_min, y_min, x_max, y_max),
                device=device,
                conf=pose_conf,
                iou=pose_iou,
                pad_pct=pose_pad_pct,
                min_side=pose_min_roi,
            )
            pose_row = _pose_row_from_keypoints(bf.frame_idx, stable_id, label, abs_kps)
            pose_buffers.setdefault(key, []).append(pose_row)
            all_pose_rows.append(pose_row)
            if abs_kps:
                overlay = _draw_keypoints_and_skeleton(overlay, abs_kps, color=color)

        if writer is not None:
            writer.write(overlay)

    cap.release()
    if writer is not None:
        writer.release()
        if (
            os.path.exists(temp_pose_avi)
            and os.path.getsize(temp_pose_avi) > 0
            and _ffmpeg_temp_avi_to_h264_mp4(temp_pose_avi, track_pose_overlay_path)
        ):
            os.remove(temp_pose_avi)

    written_bbox = _flush_stable_bbox_csvs(id_rows, output_dir, total_frames)

    links_path = _write_yolo_reid_links_csv(output_dir, geo_linker.reid_links)
    all_pose_path = _write_all_id_pose_csv(output_dir, all_pose_rows)
    per_id_pose = _flush_pose_csv_buffers(output_dir, video_stem, pose_buffers)
    _release_yolo_gpu_memory()

    return {
        "bbox_csvs": written_bbox,
        "all_id_pose": all_pose_path,
        "yolo_reid_links": links_path,
        "per_id_pose": per_id_pose,
        "track_pose_overlay": track_pose_overlay_path
        if os.path.isfile(track_pose_overlay_path)
        else None,
    }


def _format_track_cli_command(
    *,
    model_path: str,
    source: str,
    output_dir: str,
    tracker: str,
    config: dict[str, Any],
    target_classes: list[int] | None,
    do_pose: bool,
) -> str:
    """Build copy-paste CLI equivalent to one GUI tracker run (``track`` subcommand)."""
    parts: list[str] = [
        "uv",
        "run",
        "python",
        "-m",
        "vaila.yolov26track",
        "track",
        "--model",
        model_path,
        "--source",
        source,
        "--output",
        output_dir,
        "--tracker",
        tracker,
        "--conf",
        str(config.get("conf", 0.25)),
        "--iou",
        str(config.get("iou", 0.7)),
        "--device",
        str(config.get("device", "auto")),
        "--vid-stride",
        str(config.get("vid_stride", 1)),
    ]
    if target_classes:
        parts.append("--classes")
        parts.extend(str(c) for c in target_classes)
    max_ids = int(config.get("max_tracked_ids", 0) or 0)
    if max_ids > 0:
        parts += ["--max-ids", str(max_ids)]
    if not do_pose:
        parts.append("--no-pose")
    else:
        parts += [
            "--pose-model",
            str(config.get("pose_model_name", "yolo26n-pose.pt")),
            "--pose-conf",
            str(config.get("pose_conf", 0.10)),
            "--pose-iou",
            str(config.get("pose_iou", 0.70)),
            "--pose-min-roi",
            str(config.get("pose_min_roi", 256)),
            "--pose-pad-pct",
            str(config.get("pose_pad_pct", 0.15)),
        ]
    if not config.get("stabilize_ids", True):
        parts.append("--no-stabilize-ids")
    else:
        parts += [
            "--reid-max-gap",
            str(config.get("reid_max_gap", 12)),
            "--reid-max-dist",
            str(config.get("reid_max_dist", 180.0)),
            "--reid-min-iou",
            str(config.get("reid_min_iou", 0.05)),
            "--reid-direction-weight",
            str(config.get("reid_direction_weight", 0.5)),
        ]
    if config.get("appearance_reid"):
        parts.append("--appearance-reid")
        parts += [
            "--appearance-reid-threshold",
            str(config.get("appearance_reid_threshold", 0.6)),
        ]
    homography = config.get("reid_homography_path") or config.get("reid_homography_file")
    if homography:
        parts += ["--reid-homography", str(homography)]
    return " ".join(shlex.quote(p) for p in parts)


def _print_pose_video_equivalent_cli(
    *,
    video_path: str,
    output_base_dir: str,
    pose_model: str,
    pose_conf: float,
    pose_iou: float,
    device: str,
) -> None:
    """Best-effort CLI hint for Pose (video) — no dedicated track subcommand."""
    print(
        "\n>> vaila/yolov26track: Pose (video) — GUI-only flow (no track subcommand).", flush=True
    )
    print(">> Launch from vailá: Frame B → YOLO + FB → Pose (video)", flush=True)
    print(">> Or open the GUI module:", flush=True)
    print(">>   uv run python -u -m vaila.yolov26track", flush=True)
    print(">> Selected parameters:", flush=True)
    print(f">>   video={shlex.quote(video_path)}", flush=True)
    print(f">>   output_parent={shlex.quote(output_base_dir)}", flush=True)
    print(
        f">>   pose_model={pose_model} pose_conf={pose_conf} pose_iou={pose_iou} device={device}",
        flush=True,
    )
    print("", flush=True)


def _print_pose_from_tracking_workflow_hint(tracking_dir: str) -> None:
    """CLI workflow hint for Pose (tracking) — step 1 is track; step 2 remains GUI."""
    print("\n>> vaila/yolov26track: Pose (tracking) workflow:", flush=True)
    print(">>   Step 1 — generate bbox CSVs with track CLI (one video example):", flush=True)
    print(
        ">>     uv run python -m vaila.yolov26track track "
        "--model vaila/models/yolo26n.pt --source VIDEO.mp4 --output OUT/",
        flush=True,
    )
    print(">>   Step 2 — select ID and run pose in GUI (no CLI subcommand yet):", flush=True)
    print(
        ">>     uv run python -u -m vaila.yolov26track  # then Pose (tracking) in chooser",
        flush=True,
    )
    print(f">>   Tracking dir selected: {shlex.quote(tracking_dir)}", flush=True)
    print("", flush=True)


def run_track_cli(argv: list[str] | None = None) -> int:
    """CLI: run YOLO detection+tracking and export biomechanics-ready CSVs.

    Outputs (for kinematics / REC2D / REC3D):
      * ``{label}_id_NN.csv`` + ``all_id_detection.csv`` — per-ID bbox tracks,
        loadable in ``getpixelvideo`` (same schema as the GUI tracker).
      * ``<stem>_markers.csv`` — getpixelvideo point format
        ``frame,p1_x,p1_y,...,pN_x,pN_y`` (one anchor point per player); feeds
        ``rec2d.py`` / ``rec3d.py`` directly.
      * ``<stem>_track_overlay.mp4`` — H.264 overlay (never a bulky ``.avi``).

    Unlike Ultralytics ``yolo track`` (which only saves a video), this emits the
    CSVs vailá's reconstruction pipeline needs.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vaila.yolov26track track",
        description=(
            "Run YOLO tracking and export per-ID bbox CSVs + a getpixelvideo "
            "markers CSV (frame,p1_x,p1_y,... for REC2D/REC3D) + an H.264 .mp4."
        ),
    )
    parser.add_argument(
        "--model", required=True, help="Path to trained weights (best.pt / .engine)."
    )
    parser.add_argument(
        "--source", "--video", dest="source", required=True, help="Input video file."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dir. Default: <video_dir>/processed_yolotrack_<stem>_<timestamp>.",
    )
    parser.add_argument("--class-name", default="person", help="Fallback label when names missing.")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)."
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (default 0.7).")
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Inference image size (match training)."
    )
    parser.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (default auto).")
    parser.add_argument(
        "--tracker", default="botsort.yaml", help="Tracker config: botsort.yaml or bytetrack.yaml."
    )
    parser.add_argument(
        "--vid-stride", type=int, default=1, help="Process every Nth frame (default 1)."
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these class indices (space separated).",
    )
    parser.add_argument(
        "--max-ids",
        type=int,
        default=0,
        help="Keep only the N most persistent IDs, re-ranked 1..N (0 = no cap).",
    )
    parser.add_argument(
        "--anchor",
        default="bottom",
        help=(
            "Marker point per bbox for the REC2D/REC3D markers CSV: "
            "center|bottom|top|left|right|corners (default bottom = foot)."
        ),
    )
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write annotated overlay MP4 (default on; use --no-save-video to skip).",
    )
    parser.add_argument(
        "--pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run inline upscaled pose inside tracked bboxes (default on).",
    )
    parser.add_argument(
        "--pose-model",
        default="yolo26n-pose.pt",
        help="YOLO pose weights filename under vaila/models/ (default yolo26n-pose.pt).",
    )
    parser.add_argument("--pose-conf", type=float, default=0.10, help="Pose confidence threshold.")
    parser.add_argument("--pose-iou", type=float, default=0.70, help="Pose NMS IoU threshold.")
    parser.add_argument(
        "--pose-min-roi",
        type=int,
        default=256,
        help="Minimum ROI side (px) before YOLO pose inference (default 256).",
    )
    parser.add_argument(
        "--pose-pad-pct",
        type=float,
        default=0.15,
        help="Fractional padding around bbox crop (default 0.15).",
    )
    parser.add_argument(
        "--stabilize-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Geometric ID linker after tracker (Hungarian, default on).",
    )
    parser.add_argument("--reid-max-gap", type=int, default=12, help="Max frame gap for ReID.")
    parser.add_argument(
        "--reid-max-dist", type=float, default=180.0, help="Max centroid distance (px)."
    )
    parser.add_argument("--reid-min-iou", type=float, default=0.05, help="Min IoU gate.")
    parser.add_argument(
        "--reid-direction-weight",
        type=float,
        default=0.5,
        help="Velocity-direction penalty weight (0=off).",
    )
    parser.add_argument(
        "--appearance-reid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run OSNet appearance ReID after geometric stabilize (reid_yolotrack).",
    )
    parser.add_argument(
        "--appearance-reid-threshold",
        type=float,
        default=0.6,
        help="Cosine similarity threshold for appearance ReID merge.",
    )
    parser.add_argument(
        "--reid-homography",
        type=str,
        default=None,
        help="Optional 3x3 homography (.npy/.npz/.csv) for pitch-plane Re-ID distances.",
    )
    args = parser.parse_args(argv)

    source = os.path.abspath(args.source)
    if not os.path.isfile(source):
        print(f"[yolov26track] ERROR: video not found: {source}")
        return 2
    if not os.path.isfile(os.path.abspath(args.model)):
        print(f"[yolov26track] ERROR: model not found: {os.path.abspath(args.model)}")
        return 2
    model_path = os.path.abspath(args.model)

    if args.device.lower() == "auto":
        device = detect_optimal_device()
    else:
        ok, msg = validate_device_choice(args.device)
        device = args.device.lower() if ok else "cpu"
        print(f"[yolov26track] device: {msg}")

    video_stem = os.path.splitext(os.path.basename(source))[0]
    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(source), f"processed_yolotrack_{video_stem}_{ts}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(source)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
    cap.release()

    _track_banner(
        "YOLO track (CLI)",
        f"model={os.path.basename(model_path)} | video={os.path.basename(source)} | "
        f"frames≈{total_frames} | tracker={args.tracker} | device={device}",
    )
    print(f"[yolov26track] output dir: {output_dir}")

    task = "detect"
    with contextlib.suppress(Exception):
        task = _guess_task_for_cli(model_path)
    model = YOLO(model_path, task=task)

    tracker_arg = args.tracker
    tracker_stem = Path(tracker_arg).stem.lower().replace(".yaml", "").replace(".yml", "")
    if tracker_stem == "botsort":
        models_dir = str(VAILA_MODELS_DIR)
        tracker_arg = _build_botsort_custom_yaml(models_dir, "botsort")
        print(f"[yolov26track] BoT-SORT with ReID + GMC: {tracker_arg}")

    linker_cfg_dict: dict[str, Any] = {
        "reid_max_gap": int(args.reid_max_gap),
        "reid_max_dist": float(args.reid_max_dist),
        "reid_min_iou": float(args.reid_min_iou),
        "reid_direction_weight": float(args.reid_direction_weight),
    }
    if args.reid_homography:
        linker_cfg_dict["reid_homography_path"] = args.reid_homography
    linker_config = _linker_config_from_dict(linker_cfg_dict)

    track_kwargs: dict[str, Any] = {
        "source": source,
        "conf": args.conf,
        "iou": args.iou,
        "device": device,
        "vid_stride": max(1, int(args.vid_stride)),
        "save": False,
        "stream": True,
        "persist": True,
        "tracker": tracker_arg,
        "verbose": False,
    }
    if args.imgsz:
        track_kwargs["imgsz"] = int(args.imgsz)
    if args.classes:
        track_kwargs["classes"] = list(args.classes)

    progress_cb = _make_frame_progress_logger(total_frames or 0, "Track inference")
    results = model.track(**track_kwargs)
    buffer, id_counts = buffer_tracking_stream(
        results,
        save_annotated=bool(args.save_video),
        keep_raw_result=False,
        gpu_gc_every=_progress_chunk_size(total_frames or 1) if device == "cuda" else 0,
        progress_cb=progress_cb,
    )

    if args.max_ids and args.max_ids > 0:
        rerank_map = build_id_rerank_map(id_counts, int(args.max_ids))
        kept = len(rerank_map)
        print(f"[yolov26track] ID cap: keeping top-{kept} of {len(id_counts)} raw IDs.")
        buffer = rerank_buffered_stream(buffer, rerank_map)

    names = getattr(model, "names", None)
    if isinstance(names, dict):
        names_map = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, (list, tuple)):
        names_map = {i: str(v) for i, v in enumerate(names)}
    else:
        names_map = None

    pose_outputs: dict[str, Any] = {}
    links_path: str | None = None
    if args.pose:
        _track_banner(
            "Inline track+pose (upscaled ROI)",
            f"pose={args.pose_model} | stabilize_ids={args.stabilize_ids}",
        )
        pose_outputs = _emit_track_pose_from_buffer(
            buffer,
            source,
            output_dir,
            video_stem,
            device=device,
            names_map=names_map,
            class_name=args.class_name,
            pose_model_name=args.pose_model,
            pose_conf=float(args.pose_conf),
            pose_iou=float(args.pose_iou),
            pose_pad_pct=float(args.pose_pad_pct),
            pose_min_roi=int(args.pose_min_roi),
            stabilize_ids=bool(args.stabilize_ids),
            linker_config=linker_config,
            fps=float(fps),
            save_overlay=bool(args.save_video),
        )
        written = pose_outputs.get("bbox_csvs") or []
        links_path = pose_outputs.get("yolo_reid_links")
    elif args.stabilize_ids:
        _track_banner("Geometric Re-ID stabilize", f"frames buffered={len(buffer)}")
        written, links_path = _apply_geometric_stabilize_to_buffer(
            buffer,
            output_dir,
            args.class_name,
            names_map,
            stabilize_ids=True,
            linker_config=linker_config,
        )
    else:
        _track_banner("Writing per-ID CSVs", f"frames buffered={len(buffer)}")
        written = _write_per_id_csvs_from_buffer(buffer, output_dir, args.class_name, names_map)

    if args.appearance_reid:
        try:
            from .reid_yolotrack import run_appearance_reid_on_tracking_dir
        except ImportError:
            from reid_yolotrack import run_appearance_reid_on_tracking_dir  # ty: ignore
        _track_banner("OSNet appearance ReID", f"threshold={args.appearance_reid_threshold}")
        run_appearance_reid_on_tracking_dir(
            output_dir,
            source,
            threshold=float(args.appearance_reid_threshold),
        )

    combined = create_combined_detection_csv(output_dir)

    _track_banner("Writing REC2D/REC3D markers CSV", f"anchor={args.anchor}")
    markers_csv, n_markers = _write_markers_csv_from_buffer(
        buffer, output_dir, video_stem, args.anchor
    )

    overlay = None
    if args.save_video:
        _track_banner("Encoding overlay video (.mp4)", "")
        overlay = _export_overlay_video_from_buffer(buffer, output_dir, video_stem, fps)

    _track_banner(
        "DONE",
        f"IDs={len(written)} | markers={n_markers} | output in {output_dir}",
    )
    print(f"[yolov26track] per-ID bbox CSVs: {len(written)} files")
    if combined:
        print(f"[yolov26track] combined bbox CSV (getpixelvideo): {combined}")
    print(f"[yolov26track] REC2D/REC3D markers CSV (frame,p1_x,p1_y,...): {markers_csv}")
    if overlay:
        print(f"[yolov26track] overlay video (.mp4): {overlay}")
    if pose_outputs.get("all_id_pose"):
        print(f"[yolov26track] combined pose CSV: {pose_outputs['all_id_pose']}")
    if pose_outputs.get("yolo_reid_links") or links_path:
        print(
            f"[yolov26track] geometric Re-ID links: "
            f"{pose_outputs.get('yolo_reid_links') or links_path}"
        )
    if pose_outputs.get("track_pose_overlay"):
        print(f"[yolov26track] track+pose overlay: {pose_outputs['track_pose_overlay']}")
    for p in pose_outputs.get("per_id_pose") or []:
        print(f"[yolov26track] per-ID pose CSV: {p}")
    print(
        "[yolov26track] Kinematics flow: feed the *_markers.csv into "
        "REC2D (rec2d.py) / REC3D (rec3d.py) with your DLT params."
    )
    print(
        "[yolov26track] To edit/inspect: open the video in getpixelvideo, "
        "Load Tracking CSV -> all_id_detection.csv (bbox) or the *_markers.csv (points)."
    )
    return 0


def _guess_task_for_cli(model_path: str) -> str:
    """Best-effort task guess from a weights filename for the CLI loader."""
    stem = os.path.basename(model_path).lower()
    if "pose" in stem:
        return "pose"
    if "seg" in stem:
        return "segment"
    if "cls" in stem:
        return "classify"
    return "detect"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "track":
        try:
            sys.exit(run_track_cli(sys.argv[2:]))
        except KeyboardInterrupt:
            print("\n\nTracking interrupted by user.")
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"\n\nAn error occurred: {e}")
            traceback.print_exc()
            sys.exit(1)

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
