"""
Project: vailá Multimodal Toolbox
Script: mp_facemesh_nvidia.py

Author: Abel Gonçalves Chinaglia
Email: abel.chinaglia@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 12 January 2026
Update Date: 12 January 2026
Version: 0.1.0 (NVIDIA/AMD GPU Optimized)

Description:
This script performs batch processing of videos for 2D face mesh detection using
MediaPipe's FaceMesh model with GPU acceleration support. It processes videos from
a specified input directory, overlays face landmarks on each video frame, and exports
both normalized and pixel-based landmark coordinates to CSV files.

This script is identical to mp_facemesh.py but includes GPU detection and selection
for NVIDIA and AMD GPUs. Note: MediaPipe FaceMesh currently uses CPU processing,
but GPU detection is included for future compatibility and system information.

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, max number of faces, and
whether to refine landmarks. The script supports ROI (Region of Interest) selection
for focused processing, video resize, filtering, and TOML configuration files.

Usage:
- Run the script to detect GPU availability (NVIDIA/AMD)
- Device selection dialog appears (CPU/GPU choice)
- Select input directory containing video files (.mp4, .avi, .mov)
- Configure parameters via GUI or load TOML file
- Process videos with selected device

Requirements:
- Python 3.12.12
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Tkinter (usually included with Python)
- Pandas (pip install pandas)
- psutil (pip install psutil)

License:
    This project is licensed under the terms of AGPLv3.0.
"""

import datetime
import gc
import json
import os
import platform
import time as _time_module
from pathlib import Path

# #region agent log
# Debug logging - uses script directory for portability across OS (Linux, macOS, Windows)
# Creates .cursor directory in project root if it doesn't exist
# MUST be defined BEFORE any calls to _debug_log
_debug_log_dir = Path(__file__).parent.parent / ".cursor"
try:
    _debug_log_dir.mkdir(exist_ok=True)
except (OSError, PermissionError):
    # If can't create .cursor, use script directory instead
    _debug_log_dir = Path(__file__).parent
_debug_log_path = _debug_log_dir / "debug.log"


def _debug_log(hypothesis_id, location, message, data=None):
    try:
        with open(_debug_log_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "debug-session",
                        "runId": "run1",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "message": message,
                        "data": data or {},
                        "timestamp": int(_time_module.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass


# #endregion

# Set CUDA_VISIBLE_DEVICES to use only NVIDIA GPU (device 0) and exclude Intel GPU (device 1)
# This ensures MediaPipe uses the NVIDIA GPU when GPU delegate is available
# #region agent log
_debug_log("A", "mp_facemesh_nvidia.py:54", "Before setting CUDA_VISIBLE_DEVICES", {
    "existing_cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET"),
    "all_cuda_env": {k: v for k, v in os.environ.items() if "CUDA" in k}
})
# #endregion
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force set, don't use setdefault
# #region agent log
_debug_log("A", "mp_facemesh_nvidia.py:58", "After setting CUDA_VISIBLE_DEVICES", {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
})
# #endregion
import shutil
import subprocess
import tempfile
import time
import tkinter as tk
import urllib.request
from collections import deque
from tkinter import filedialog, messagebox

import cv2
# #region agent log
_debug_log("B", "mp_facemesh_nvidia.py:67", "Before importing MediaPipe", {
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
})
# #endregion
import mediapipe as mp
# #region agent log
_debug_log("B", "mp_facemesh_nvidia.py:70", "After importing MediaPipe", {
    "mp_version": getattr(mp, "__version__", "unknown"),
    "has_tasks": hasattr(mp, "tasks"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
})
# #endregion
import numpy as np
import pandas as pd
import psutil

# Additional imports for filtering and interpolation
from pykalman import KalmanFilter  # noqa: E402
from rich import print  # noqa: E402
from scipy.interpolate import UnivariateSpline  # noqa: E402
from scipy.signal import butter, savgol_filter, sosfiltfilt  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

# Import TOML for configuration management
try:
    import toml
except ImportError:
    print("Warning: toml not found. Installing...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "toml"])
    import toml

# Additional imports for CPU throttling and better resource management
import multiprocessing  # noqa: F401, E402 - For future Linux batch processing
import signal  # noqa: F401, E402 - For future Linux process management
import threading  # noqa: F401, E402 - For future Linux thread management

# MediaPipe FaceMesh constants - Using Tasks API
# Face regions for MediaPipe FaceMesh (manually defined since Tasks API doesn't expose these)
# These are the standard FaceMesh connections (468 landmarks)
# We'll define key connections for visualization
FACE_CONNECTIONS = frozenset([
    # Face oval (contour)
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10),
    # Left eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (133, 173), (173, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 246), (246, 33),
    # Right eye
    (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
    # Lips
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 320),
    (320, 307), (307, 375), (375, 321), (321, 308), (308, 324), (324, 318), (318, 61),
    # Left eyebrow
    (276, 283), (283, 282), (282, 295), (295, 285), (285, 336), (336, 296), (296, 334), (334, 293),
    (293, 300), (300, 276),
    # Right eyebrow
    (46, 53), (53, 52), (52, 65), (65, 55), (55, 70), (70, 63), (63, 105), (105, 66),
    (66, 107), (107, 46),
])

# Drawing connections (for visualization)
DRAW_CONNECTIONS = list(FACE_CONNECTIONS)

# Define regions for compatibility (simplified)
MEDIAPIPE_REGIONS = {
    "face_oval": [(i, (i + 1) % 17) for i in range(17)],  # Simplified
    "left_eye": [(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133)],
    "right_eye": [(362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390), (390, 249), (249, 263)],
    "lips": [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 320)],
    "left_eyebrow": [(276, 283), (283, 282), (282, 295), (295, 285), (285, 336)],
    "right_eyebrow": [(46, 53), (53, 52), (52, 65), (65, 55), (55, 70)],
    "tessellation": [],  # Not used for drawing
    "irises": [],
    "contours": [],
}

# Human-readable names for key landmarks
LANDMARK_NAMES = {
    33: "right_eye_outer_corner",
    133: "right_eye_inner_corner",
    159: "right_eye_top",
    145: "right_eye_bottom",
    362: "left_eye_outer_corner",
    263: "left_eye_inner_corner",
    386: "left_eye_top",
    374: "left_eye_bottom",
    61: "mouth_left_corner",
    291: "mouth_right_corner",
    0: "mouth_center_top",
    17: "chin_center_bottom",
    13: "mouth_upper_inner",
    14: "mouth_lower_inner",
    1: "nose_tip",
    4: "nose_bottom",
    6: "nose_bridge",
    168: "nose_top",
    70: "right_eyebrow_inner",
    105: "right_eyebrow_middle",
    107: "right_eyebrow_outer",
    336: "left_eyebrow_inner",
    334: "left_eyebrow_middle",
    300: "left_eyebrow_outer",
    10: "forehead_center",
    152: "chin_tip",
    234: "left_cheek",
    454: "right_cheek",
    # Iris landmarks (468-477) - Essential for gaze/attention analysis
    468: "right_iris_center",
    469: "right_iris_1",
    470: "right_iris_2",
    471: "right_iris_3",
    472: "right_iris_4",
    473: "left_iris_center",
    474: "left_iris_1",
    475: "left_iris_2",
    476: "left_iris_3",
    477: "left_iris_4",
}

# Total number of face landmarks (MediaPipe FaceMesh has 478 landmarks including iris)
NUM_FACE_LANDMARKS = 478

VERBOSE_FRAMES = False
PAD_START_FRAMES = 120
PAD_START_FRAMES_DEFAULT = 30
ENABLE_PADDING_DEFAULT = True

# CPU throttling settings
CPU_USAGE_THRESHOLD = 150
FRAME_SLEEP_TIME = 0.01
MAX_CPU_CHECK_INTERVAL = 100

# Generate landmark mapping for CSV headers
def generate_landmark_mapping():
    """Generate mapping from landmark index to descriptive name"""
    regions_indices = {
        region: set(idx for connection in connections for idx in connection)
        for region, connections in MEDIAPIPE_REGIONS.items()
    }
    tess_indices = regions_indices.get("tessellation", set())
    tess_conns = MEDIAPIPE_REGIONS["tessellation"]

    max_idx = max(idx for indices in regions_indices.values() for idx in indices)
    landmark_to_name = {}
    for idx in range(max_idx + 1):
        anat_regions = [
            r
            for r, inds in regions_indices.items()
            if r != "tessellation" and idx in inds
        ]
        if anat_regions:
            regions_sorted = sorted(anat_regions)
            landmark_to_name[idx] = "_".join(regions_sorted) + f"_{idx}"
        elif idx in tess_indices:
            neighbors = {j for i, j in tess_conns if i == idx} | {
                i for i, j in tess_conns if j == idx
            }
            neighbor_regions = []
            for nb in neighbors:
                for r, inds in regions_indices.items():
                    if r != "tessellation" and nb in inds:
                        neighbor_regions.append(r)
            if neighbor_regions:
                region = sorted(neighbor_regions)[0]
                landmark_to_name[idx] = f"{region}_{idx}"
            else:
                landmark_to_name[idx] = f"tessellation_{idx}"
        else:
            landmark_to_name[idx] = f"unknown_{idx}"
    return landmark_to_name


# Generate CSV header with all landmarks
def generate_csv_header():
    """Generate CSV header with all face landmarks"""
    landmark_map = generate_landmark_mapping()
    header = ["frame_index", "face_idx"]  # Include face_idx to match row structure

    # Include all landmarks from 0 to 467
    indices_sorted = list(range(NUM_FACE_LANDMARKS))

    for idx in indices_sorted:
        name = LANDMARK_NAMES.get(idx, landmark_map.get(idx, f"landmark_{idx}"))
        header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    return header, indices_sorted


# Filter and smoothing functions (copied from markerless_2d_analysis.py)
def butter_filter(
    data,
    fs,
    filter_type="low",
    cutoff=None,
    lowcut=None,
    highcut=None,
    order=4,
    padding=True,
):
    """Applies a Butterworth filter (low-pass or band-pass) to the input data."""
    nyq = 0.5 * fs
    if filter_type == "low":
        if cutoff is None:
            raise ValueError("Cutoff frequency must be provided for low-pass filter.")
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    elif filter_type == "band":
        if lowcut is None or highcut is None:
            raise ValueError(
                "Lowcut and highcut frequencies must be provided for band-pass filter."
            )
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="band", analog=False, output="sos")
    else:
        raise ValueError("Unsupported filter type. Use 'low' for low-pass or 'band' for band-pass.")

    data = np.asarray(data)
    axis = 0

    if padding:
        data_len = data.shape[axis]
        max_padlen = data_len - 1
        padlen = min(int(fs), max_padlen, 15)

        if data_len <= padlen:
            raise ValueError(
                f"The length of the input data ({data_len}) must be greater than the padding length ({padlen})."
            )

        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (padlen, padlen)
        padded_data = np.pad(data, pad_width=pad_width, mode="reflect")
        filtered_padded_data = sosfiltfilt(sos, padded_data, axis=axis, padlen=0)
        idx = [slice(None)] * data.ndim
        idx[axis] = slice(padlen, -padlen)
        filtered_data = filtered_padded_data[tuple(idx)]
    else:
        filtered_data = sosfiltfilt(sos, data, axis=axis, padlen=0)

    return filtered_data


def savgol_smooth(data, window_length, polyorder):
    """Apply Savitzky-Golay filter to the data."""
    data = np.asarray(data)
    return savgol_filter(data, window_length, polyorder, axis=0)


def lowess_smooth(data, frac, it):
    """Apply LOWESS smoothing to the data."""
    data = np.asarray(data)
    x = np.arange(len(data)) if data.ndim == 1 else np.arange(data.shape[0])

    try:
        pad_len = int(len(data) * 0.1)
        if pad_len > 0:
            if data.ndim == 1:
                padded_data = np.pad(data, (pad_len, pad_len), mode="reflect")
                padded_x = np.arange(len(padded_data))
                smoothed = lowess(
                    endog=padded_data,
                    exog=padded_x,
                    frac=frac,
                    it=it,
                    return_sorted=False,
                    is_sorted=True,
                )
                return smoothed[pad_len:-pad_len]
            else:
                padded_data = np.pad(data, ((pad_len, pad_len), (0, 0)), mode="reflect")
                padded_x = np.arange(len(padded_data))
                smoothed = np.empty_like(data)
                for j in range(data.shape[1]):
                    smoothed[:, j] = lowess(
                        endog=padded_data[:, j],
                        exog=padded_x,
                        frac=frac,
                        it=it,
                        return_sorted=False,
                        is_sorted=True,
                    )[pad_len:-pad_len]
                return smoothed
        else:
            if data.ndim == 1:
                return lowess(
                    endog=data,
                    exog=x,
                    frac=frac,
                    it=it,
                    return_sorted=False,
                    is_sorted=True,
                )
            else:
                smoothed = np.empty_like(data)
                for j in range(data.shape[1]):
                    smoothed[:, j] = lowess(
                        endog=data[:, j],
                        exog=x,
                        frac=frac,
                        it=it,
                        return_sorted=False,
                        is_sorted=True,
                    )
                return smoothed
    except Exception as e:
        print(f"Error in LOWESS smoothing: {str(e)}")
        return data


def spline_smooth(data, s=1.0):
    """Apply spline smoothing to the data."""
    data = np.asarray(data)
    pad_len = int(len(data) * 0.1)
    if pad_len > 0:
        if data.ndim == 1:
            padded_data = np.pad(data, (pad_len, pad_len), mode="reflect")
            padded_x = np.arange(len(padded_data))
            spline = UnivariateSpline(padded_x, padded_data, s=s)
            return spline(padded_x)[pad_len:-pad_len]
        else:
            padded_data = np.pad(data, ((pad_len, pad_len), (0, 0)), mode="reflect")
            padded_x = np.arange(len(padded_data))
            filtered = np.empty_like(data)
            for j in range(data.shape[1]):
                spline = UnivariateSpline(padded_x, padded_data[:, j], s=s)
                filtered[:, j] = spline(padded_x)[pad_len:-pad_len]
            return filtered
    else:
        if data.ndim == 1:
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data, s=s)
            return spline(x)
        else:
            filtered = np.empty_like(data)
            x = np.arange(data.shape[0])
            for j in range(data.shape[1]):
                spline = UnivariateSpline(x, data[:, j], s=s)
                filtered[:, j] = spline(x)
            return filtered


def kalman_smooth(data, n_iter=5, mode=1):
    """Apply Kalman smoothing to data."""
    alpha = 0.7
    data = np.asarray(data)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_features = data.shape[1]

    try:
        if mode == 1:
            filtered_data = np.empty_like(data)
            for j in range(n_features):
                kf = KalmanFilter(
                    transition_matrices=np.array([[1, 1], [0, 1]]),
                    observation_matrices=np.array([[1, 0]]),
                    initial_state_mean=np.zeros(2),
                    initial_state_covariance=np.eye(2),
                    transition_covariance=np.eye(2) * 0.1,
                    observation_covariance=np.array([[0.1]]),
                    n_dim_obs=1,
                    n_dim_state=2,
                )
                smoothed_state_means, _ = kf.em(data[:, j : j + 1], n_iter=n_iter).smooth(
                    data[:, j : j + 1]
                )
                filtered_data[:, j] = alpha * smoothed_state_means[:, 0] + (1 - alpha) * data[:, j]
        else:
            if n_features % 2 != 0:
                raise ValueError("For 2D mode, number of features must be even (x,y pairs)")

            filtered_data = np.empty_like(data)
            for j in range(0, n_features, 2):
                transition_matrix = np.array(
                    [
                        [1, 0, 1, 0, 0.5, 0],
                        [0, 1, 0, 1, 0, 0.5],
                        [0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1],
                    ]
                )
                observation_matrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]])
                initial_state_mean = np.array([data[0, j], data[0, j + 1], 0, 0, 0, 0])
                initial_state_covariance = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
                transition_covariance = np.diag([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
                observation_covariance = np.array([[0.1, 0], [0, 0.1]])

                kf = KalmanFilter(
                    transition_matrices=transition_matrix,
                    observation_matrices=observation_matrix,
                    initial_state_mean=initial_state_mean,
                    initial_state_covariance=initial_state_covariance,
                    transition_covariance=transition_covariance,
                    observation_covariance=observation_covariance,
                    n_dim_obs=2,
                    n_dim_state=6,
                )

                observations = np.column_stack([data[:, j], data[:, j + 1]])
                smoothed_state_means, _ = kf.em(observations, n_iter=n_iter).smooth(observations)

                filtered_data[:, j] = alpha * smoothed_state_means[:, 0] + (1 - alpha) * data[:, j]
                filtered_data[:, j + 1] = (
                    alpha * smoothed_state_means[:, 1] + (1 - alpha) * data[:, j + 1]
                )

        return filtered_data

    except Exception as e:
        print(f"Error in Kalman smoothing: {str(e)}")
        return data


def arima_smooth(data, order=(1, 0, 0)):
    """Apply ARIMA smoothing to the input data."""
    data = np.asarray(data)

    if data.ndim == 1:
        try:
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data

            valid_data = data[valid_mask]
            if len(valid_data) < max(order) + 1:
                print("Warning: Not enough data points for ARIMA model")
                return data

            model = ARIMA(valid_data, order=order)
            result = model.fit(disp=False)

            output = data.copy()
            output[valid_mask] = result.fittedvalues
            return output

        except Exception as e:
            print(f"Error in ARIMA smoothing: {str(e)}")
            return data
    else:
        smoothed = np.empty_like(data)
        for j in range(data.shape[1]):
            try:
                col_data = data[:, j]
                valid_mask = ~np.isnan(col_data)

                if not np.any(valid_mask):
                    smoothed[:, j] = col_data
                    continue

                valid_data = col_data[valid_mask]
                if len(valid_data) < max(order) + 1:
                    print(f"Warning: Not enough data points for ARIMA model in column {j}")
                    smoothed[:, j] = col_data
                    continue

                model = ARIMA(valid_data, order=order)
                result = model.fit(disp=False)

                smoothed[:, j] = col_data.copy()
                smoothed[valid_mask, j] = result.fittedvalues

            except Exception as e:
                print(f"Error in ARIMA smoothing for column {j}: {str(e)}")
                smoothed[:, j] = data[:, j]
        return smoothed


def pad_signal(data, pad_width, mode="edge"):
    """Pad signal for better edge handling in filtering."""
    if pad_width == 0:
        return data
    return np.pad(data, (pad_width, pad_width), mode=mode)


def is_linux_system():
    """Check if running on Linux system"""
    return platform.system().lower() == "linux"


def get_system_memory_info():
    """Get current system memory usage"""
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
        }
    except Exception as e:
        print(f"Warning: Could not get memory info: {e}")
        return None


def get_cpu_usage():
    """Get current CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0


def should_throttle_cpu(frame_count):
    """Check if we should throttle CPU based on usage and frame count"""
    if frame_count % MAX_CPU_CHECK_INTERVAL == 0:
        cpu_usage = get_cpu_usage()
        if cpu_usage > CPU_USAGE_THRESHOLD:
            print(f"High CPU usage detected: {cpu_usage:.1f}% - Applying throttling")
            return True
    return False


def apply_cpu_throttling():
    """Apply CPU throttling by sleeping"""
    time.sleep(FRAME_SLEEP_TIME)


def cleanup_memory():
    """Aggressive memory cleanup for Linux systems"""
    if is_linux_system():
        gc.collect()
        time.sleep(0.1)


# GPU detection functions
def detect_nvidia_gpu():
    """
    Detect if NVIDIA GPU is available and accessible.
    Returns tuple: (is_available: bool, gpu_info: dict, error_message: str)
    """
    gpu_info = {}
    error_message = None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                parts = lines[0].split(",")
                if len(parts) >= 3:
                    gpu_info = {
                        "name": parts[0].strip(),
                        "driver_version": parts[1].strip(),
                        "memory_total_mb": int(parts[2].strip())
                        if parts[2].strip().isdigit()
                        else 0,
                        "count": len(lines),
                    }
                    return True, gpu_info, None
        else:
            error_message = "nvidia-smi found but no GPU detected"
            return False, gpu_info, error_message

    except FileNotFoundError:
        error_message = "nvidia-smi not found (NVIDIA drivers may not be installed)"
        return False, gpu_info, error_message
    except subprocess.TimeoutExpired:
        error_message = "nvidia-smi timeout (drivers may not be responding)"
        return False, gpu_info, error_message
    except Exception as e:
        error_message = f"Error checking GPU: {str(e)}"
        return False, gpu_info, error_message


def detect_amd_gpu():
    """
    Detect if AMD GPU is available.
    Returns tuple: (is_available: bool, gpu_info: dict, error_message: str)
    """
    gpu_info = {}
    error_message = None

    try:
        # Try to detect AMD GPU using rocm-smi or clinfo
        result = subprocess.run(
            ["rocm-smi", "--showid", "--showproductname"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if lines:
                gpu_info = {
                    "name": "AMD GPU (ROCm)",
                    "driver_version": "ROCm",
                    "count": len([l for l in lines if "card" in l.lower()]),
                }
                return True, gpu_info, None
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # Try alternative detection methods
    try:
        result = subprocess.run(
            ["lspci"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            text=True,
        )
        if result.returncode == 0:
            if "amd" in result.stdout.lower() or "radeon" in result.stdout.lower():
                gpu_info = {
                    "name": "AMD GPU (detected via lspci)",
                    "driver_version": "Unknown",
                    "count": 1,
                }
                return True, gpu_info, None
    except Exception:
        pass

    error_message = "AMD GPU not detected or ROCm not available"
    return False, gpu_info, error_message


def test_mediapipe_gpu():
    """
    Test if MediaPipe can use GPU (for future compatibility).
    Note: FaceMesh currently doesn't support GPU delegate like Pose does.
    Returns tuple: (works: bool, error_message: str)
    """
    # FaceMesh doesn't currently support GPU delegate in MediaPipe
    # This function is a placeholder for future compatibility
    return False, "MediaPipe FaceMesh GPU delegate not yet available"


# Device selection dialog
class DeviceSelectionDialog(tk.simpledialog.Dialog):
    """Dialog to select CPU or GPU for processing"""

    def __init__(self, parent, nvidia_available, nvidia_info, amd_available, amd_info):
        self.nvidia_available = nvidia_available
        self.nvidia_info = nvidia_info
        self.amd_available = amd_available
        self.amd_info = amd_info
        self.selected_device = "cpu"  # Default
        super().__init__(parent, title="Select Processing Device")

    def body(self, master):
        tk.Label(master, text="Select processing device:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 20), sticky="w"
        )

        self.device_var = tk.StringVar(value="cpu")

        # CPU option
        self.cpu_var = tk.Radiobutton(
            master,
            text="CPU (Standard Processing)",
            variable=self.device_var,
            value="cpu",
            command=lambda: setattr(self, "selected_device", "cpu"),
            font=("Arial", 9),
        )
        self.cpu_var.grid(row=1, column=0, columnspan=2, sticky="w", padx=20, pady=5)

        # NVIDIA GPU option
        nvidia_text = "GPU (NVIDIA CUDA)"
        if not self.nvidia_available:
            nvidia_text += " [NOT AVAILABLE]"

        self.nvidia_var = tk.Radiobutton(
            master,
            text=nvidia_text,
            variable=self.device_var,
            value="nvidia",
            command=lambda: setattr(self, "selected_device", "nvidia"),
            font=("Arial", 9),
            state="normal" if self.nvidia_available else "disabled",
        )
        self.nvidia_var.grid(row=2, column=0, columnspan=2, sticky="w", padx=20, pady=5)

        # AMD GPU option
        amd_text = "GPU (AMD ROCm)"
        if not self.amd_available:
            amd_text += " [NOT AVAILABLE]"

        self.amd_var = tk.Radiobutton(
            master,
            text=amd_text,
            variable=self.device_var,
            value="amd",
            command=lambda: setattr(self, "selected_device", "amd"),
            font=("Arial", 9),
            state="normal" if self.amd_available else "disabled",
        )
        self.amd_var.grid(row=3, column=0, columnspan=2, sticky="w", padx=20, pady=5)

        # GPU info frame
        info_frame = tk.LabelFrame(master, text="GPU Information", padx=10, pady=10)
        info_frame.grid(row=4, column=0, columnspan=2, padx=20, pady=10, sticky="ew")

        info_text = ""
        if self.nvidia_available and self.nvidia_info:
            info_text += f"NVIDIA: {self.nvidia_info.get('name', 'Unknown')}\n"
            info_text += f"Driver: {self.nvidia_info.get('driver_version', 'Unknown')}\n"
            if self.nvidia_info.get("memory_total_mb", 0) > 0:
                info_text += f"Memory: {self.nvidia_info['memory_total_mb'] / 1024:.1f} GB\n"
        if self.amd_available and self.amd_info:
            if info_text:
                info_text += "\n"
            info_text += f"AMD: {self.amd_info.get('name', 'Unknown')}\n"

        if not info_text:
            info_text = "No GPU detected.\n"
            info_text += "Note: MediaPipe FaceMesh currently uses CPU processing.\n"
            info_text += "GPU acceleration may be available in future MediaPipe versions."

        tk.Label(info_frame, text=info_text, justify="left", font=("Arial", 8)).grid(
            row=0, column=0, sticky="w"
        )

        return self.cpu_var

    def apply(self):
        selected = self.device_var.get()
        if (self.nvidia_available and selected == "nvidia") or (
            self.amd_available and selected == "amd"
        ):
            self.result = selected
        else:
            self.result = "cpu"


# Configuration management functions
def get_default_config():
    """Get default configuration dictionary for FaceMesh"""
    return {
        "facemesh": {
            "min_detection_confidence": 0.25,
            "min_tracking_confidence": 0.25,
            "max_num_faces": 1,
            "refine_landmarks": True,
            "apply_filtering": True,
        },
        "video_resize": {"enable_resize": False, "resize_scale": 2},
        "advanced_filtering": {
            "enable_advanced_filtering": False,
            "interp_method": "linear",
            "smooth_method": "none",
            "max_gap": 60,
        },
        "smoothing_params": {
            "savgol_window_length": 7,
            "savgol_polyorder": 3,
            "lowess_frac": 0.3,
            "lowess_it": 3,
            "butter_cutoff": 4.0,
            "butter_fs": 30.0,
            "kalman_iterations": 5,
            "kalman_mode": 1,
            "spline_smoothing_factor": 1.0,
            "arima_p": 1,
            "arima_d": 0,
            "arima_q": 0,
        },
        "enable_padding": ENABLE_PADDING_DEFAULT,
        "pad_start_frames": PAD_START_FRAMES_DEFAULT,
        "roi": {
            "enable_crop": False,
            "bbox_x_min": 0,
            "bbox_y_min": 0,
            "bbox_x_max": 1920,
            "bbox_y_max": 1080,
            "roi_polygon_points": [],
            "enable_resize_crop": False,
            "resize_crop_scale": 2,
        },
    }


# TOML configuration save/load functions
def save_config_to_toml(config, filepath):
    """Save configuration to TOML file"""
    try:
        toml_config = {
            "facemesh": {
                "min_detection_confidence": config.get("min_detection_confidence", 0.25),
                "min_tracking_confidence": config.get("min_tracking_confidence", 0.25),
                "max_num_faces": config.get("max_num_faces", 1),
                "refine_landmarks": config.get("refine_landmarks", True),
                "apply_filtering": config.get("apply_filtering", True),
            },
            "video_resize": {
                "enable_resize": config.get("enable_resize", False),
                "resize_scale": config.get("resize_scale", 2),
            },
            "advanced_filtering": {
                "enable_advanced_filtering": config.get("enable_advanced_filtering", False),
                "interp_method": config.get("interp_method", "linear"),
                "smooth_method": config.get("smooth_method", "none"),
                "max_gap": config.get("max_gap", 60),
            },
            "smoothing_params": config.get(
                "_all_smooth_params",
                get_default_config()["smoothing_params"],
            ),
            "enable_padding": str(config.get("enable_padding", ENABLE_PADDING_DEFAULT)).lower(),
            "pad_start_frames": config.get("pad_start_frames", PAD_START_FRAMES_DEFAULT),
            "roi": {
                "enable_crop": config.get("enable_crop", False),
                "bbox_x_min": config.get("bbox_x_min", 0),
                "bbox_y_min": config.get("bbox_y_min", 0),
                "bbox_x_max": config.get("bbox_x_max", 1920),
                "bbox_y_max": config.get("bbox_y_max", 1080),
                "roi_polygon_points": config.get("roi_polygon_points", []),
                "enable_resize_crop": config.get("enable_resize_crop", False),
                "resize_crop_scale": config.get("resize_crop_scale", 2),
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            toml.dump(toml_config, f)
        print(f"Configuration saved to: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def load_config_from_toml(filepath):
    """Load configuration from TOML file"""
    try:
        if not os.path.exists(filepath):
            print(f"Configuration file not found: {filepath}")
            return None

        with open(filepath, encoding="utf-8") as f:
            toml_config = toml.load(f)

        config = {}

        # FaceMesh settings
        if "facemesh" in toml_config:
            fm = toml_config["facemesh"]
            config.update(
                {
                    "min_detection_confidence": float(fm.get("min_detection_confidence", 0.25)),
                    "min_tracking_confidence": float(fm.get("min_tracking_confidence", 0.25)),
                    "max_num_faces": int(fm.get("max_num_faces", 1)),
                    "refine_landmarks": bool(fm.get("refine_landmarks", True)),
                    "apply_filtering": bool(fm.get("apply_filtering", True)),
                }
            )

        # Video resize settings
        if "video_resize" in toml_config:
            vr = toml_config["video_resize"]
            config.update(
                {
                    "enable_resize": bool(vr.get("enable_resize", False)),
                    "resize_scale": int(vr.get("resize_scale", 2)),
                }
            )

        # Advanced filtering settings
        if "advanced_filtering" in toml_config:
            af = toml_config["advanced_filtering"]
            config.update(
                {
                    "enable_advanced_filtering": bool(af.get("enable_advanced_filtering", False)),
                    "interp_method": str(af.get("interp_method", "linear")),
                    "smooth_method": str(af.get("smooth_method", "none")),
                    "max_gap": int(af.get("max_gap", 60)),
                }
            )

        # Smoothing parameters
        if "smoothing_params" in toml_config:
            sp = toml_config["smoothing_params"]
            smooth_method = config.get("smooth_method", "none")
            smooth_params = {}

            if smooth_method == "savgol":
                smooth_params = {
                    "window_length": int(sp.get("savgol_window_length", 7)),
                    "polyorder": int(sp.get("savgol_polyorder", 3)),
                }
            elif smooth_method == "lowess":
                smooth_params = {
                    "frac": float(sp.get("lowess_frac", 0.3)),
                    "it": int(sp.get("lowess_it", 3)),
                }
            elif smooth_method == "butterworth":
                smooth_params = {
                    "cutoff": float(sp.get("butter_cutoff", 4.0)),
                    "fs": float(sp.get("butter_fs", 30.0)),
                }
            elif smooth_method == "kalman":
                smooth_params = {
                    "n_iter": int(sp.get("kalman_iterations", 5)),
                    "mode": int(sp.get("kalman_mode", 1)),
                }
            elif smooth_method == "splines":
                smooth_params = {"smoothing_factor": float(sp.get("spline_smoothing_factor", 1.0))}
            elif smooth_method == "arima":
                smooth_params = {
                    "p": int(sp.get("arima_p", 1)),
                    "d": int(sp.get("arima_d", 0)),
                    "q": int(sp.get("arima_q", 0)),
                }

            config["smooth_params"] = smooth_params
            config["_all_smooth_params"] = {
                "savgol_window_length": int(sp.get("savgol_window_length", 7)),
                "savgol_polyorder": int(sp.get("savgol_polyorder", 3)),
                "lowess_frac": float(sp.get("lowess_frac", 0.3)),
                "lowess_it": int(sp.get("lowess_it", 3)),
                "butter_cutoff": float(sp.get("butter_cutoff", 4.0)),
                "butter_fs": float(sp.get("butter_fs", 30.0)),
                "kalman_iterations": int(sp.get("kalman_iterations", 5)),
                "kalman_mode": int(sp.get("kalman_mode", 1)),
                "spline_smoothing_factor": float(sp.get("spline_smoothing_factor", 1.0)),
                "arima_p": int(sp.get("arima_p", 1)),
                "arima_d": int(sp.get("arima_d", 0)),
                "arima_q": int(sp.get("arima_q", 0)),
            }

        # Padding section
        if "padding" in toml_config:
            pad = toml_config["padding"]
            config["enable_padding"] = bool(pad.get("enable_padding", ENABLE_PADDING_DEFAULT))
            config["pad_start_frames"] = int(pad.get("pad_start_frames", PAD_START_FRAMES_DEFAULT))
        else:
            config["enable_padding"] = ENABLE_PADDING_DEFAULT
            config["pad_start_frames"] = PAD_START_FRAMES_DEFAULT

        # ROI section
        if "roi" in toml_config:
            roi = toml_config["roi"]
            config.update(
                {
                    "enable_crop": bool(roi.get("enable_crop", False)),
                    "bbox_x_min": int(roi.get("bbox_x_min", 0)),
                    "bbox_y_min": int(roi.get("bbox_y_min", 0)),
                    "bbox_x_max": int(roi.get("bbox_x_max", 1920)),
                    "bbox_y_max": int(roi.get("bbox_y_max", 1080)),
                    "roi_polygon_points": roi.get("roi_polygon_points", []),
                    "enable_resize_crop": bool(roi.get("enable_resize_crop", False)),
                    "resize_crop_scale": int(roi.get("resize_crop_scale", 2)),
                }
            )

        print(f"Configuration loaded successfully from: {filepath}")
        return config

    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None


# ROI selection functions
def select_free_polygon_roi(video_path):
    """
    Let the user draw a free polygon ROI on the first frame of the video.
    Left click adds points, right click removes the last point, Enter confirms,
    Esc skips, and 'r' resets. Returns a numpy array of int32 points or None.
    """
    cap = None
    window_name = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None

        ret, frame = cap.read()
        cap.release()
        cap = None

        if not ret or frame is None:
            print("Error: Could not read first frame from video for ROI selection.")
            return None

        # Scale frame to reasonable size for display
        scale = 1.0
        h, w = frame.shape[:2]
        max_h = 1800
        max_w = 2400
        if h > max_h or w > max_w:
            scale_h = max_h / h if h > max_h else 1.0
            scale_w = max_w / w if w > max_w else 1.0
            scale = min(scale_h, scale_w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        roi_points = []
        mouse_clicked = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal mouse_clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
                mouse_clicked = True
                print(f"Point added: ({x}, {y}) - Total points: {len(roi_points)}")
            elif event == cv2.EVENT_RBUTTONDOWN and roi_points:
                removed = roi_points.pop()
                mouse_clicked = True
                print(f"Point removed: {removed} - Total points: {len(roi_points)}")

        window_name = "Select ROI (Left: add, Right: undo, Enter: confirm, Esc: skip, r: reset)"

        if platform.system() == "Darwin":
            window_flags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO
        else:
            window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO

        cv2.namedWindow(window_name, window_flags)
        cv2.setMouseCallback(window_name, mouse_callback)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

        try:
            h_scaled, w_scaled = frame.shape[:2]
            desired_w = max(1200, min(w_scaled, 2400))
            desired_h = max(900, min(h_scaled, 1800))
            cv2.resizeWindow(window_name, desired_w, desired_h)
            if platform.system() == "Darwin":
                cv2.waitKey(10)
                cv2.resizeWindow(window_name, desired_w, desired_h)
                cv2.waitKey(10)
        except Exception:
            pass

        polygon_color = (255, 255, 0)
        point_color = (0, 255, 255)
        closing_line_color = (255, 0, 255)

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

            if loop_count < 100 or len(roi_points) == 0:
                y_offset = 10
                for i, text in enumerate(help_text):
                    y_pos = y_offset + i * 25
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
                cv2.polylines(display_img, [pts], False, polygon_color, 3)
                for pt in roi_points:
                    cv2.circle(display_img, pt, 5, point_color, -1)
                    cv2.circle(display_img, pt, 5, (0, 0, 0), 1)
                if len(roi_points) > 1:
                    cv2.line(display_img, roi_points[-1], roi_points[0], closing_line_color, 2)
                point_text = f"Points: {len(roi_points)} (min 3 required)"
                text_y = display_img.shape[0] - 20
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

            if mouse_clicked:
                mouse_clicked = False

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(30) & 0xFF
            loop_count += 1

            if key == 13 or key == 10:  # Enter
                if len(roi_points) >= 3:
                    print(f"ROI confirmed with {len(roi_points)} points")
                    break
                else:
                    print(f"Need at least 3 points. Currently have {len(roi_points)}")
            elif key == 27:  # Esc
                print("ROI selection cancelled")
                roi_points = []
                break
            elif key == ord("r") or key == ord("R"):  # Reset
                print("ROI points reset")
                roi_points = []

        if window_name:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)

        if len(roi_points) < 3:
            print("ROI selection requires at least 3 points.")
            return None

        # Scale points back to original resolution
        final_points = (np.array(roi_points, dtype=np.float32) / scale).astype(np.int32)
        print(f"ROI selection completed with {len(final_points)} points")
        return final_points

    except Exception as e:
        print(f"Error in ROI selection: {e}")
        import traceback

        traceback.print_exc()
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


# Additional helper functions needed
def apply_interpolation_and_smoothing(df, config, progress_callback=None):
    """Apply interpolation and smoothing to MediaPipe face landmark data."""
    if not config.get("enable_advanced_filtering", False):
        return df

    if progress_callback:
        progress_callback("Applying advanced filtering and smoothing...")

    numeric_cols = [
        col
        for col in df.columns
        if col not in ["frame_index"] and ("_x" in col or "_y" in col or "_z" in col)
    ]

    if not numeric_cols:
        return df

    interp_method = config.get("interp_method", "none")
    max_gap = config.get("max_gap", 0)
    if interp_method not in ["none", "skip"] and progress_callback:
        progress_callback(f"Applying {interp_method} interpolation...")
    for col in numeric_cols:
        if interp_method == "linear":
            if max_gap > 0:
                df[col] = df[col].interpolate(method="linear", limit=max_gap)
            else:
                df[col] = df[col].interpolate(method="linear")
        elif interp_method == "cubic":
            if max_gap > 0:
                df[col] = df[col].interpolate(method="cubic", limit=max_gap)
            else:
                df[col] = df[col].interpolate(method="cubic")
        elif interp_method == "nearest":
            if max_gap > 0:
                df[col] = df[col].interpolate(method="nearest", limit=max_gap)
            else:
                df[col] = df[col].interpolate(method="nearest")

    smooth_method = config.get("smooth_method", "none")
    smooth_params = config.get("smooth_params", {})
    if smooth_method != "none" and progress_callback:
        progress_callback(f"Applying {smooth_method} smoothing...")
    data = df[numeric_cols].values
    try:
        if smooth_method == "savgol":
            window_length = smooth_params.get("window_length", 7)
            polyorder = smooth_params.get("polyorder", 3)
            for j in range(data.shape[1]):
                col = data[:, j]
                valid = ~np.isnan(col)
                col_valid = col[valid]
                pad_width = min(10, len(col_valid) // 2)
                if np.sum(valid) > window_length:
                    try:
                        col_padded = pad_signal(col_valid, pad_width, mode="edge")
                        col_filtered = savgol_filter(col_padded, window_length, polyorder)
                        col_filtered = col_filtered[pad_width:-pad_width]
                        col_smooth = col.copy()
                        col_smooth[valid] = col_filtered
                        data[:, j] = col_smooth
                    except Exception as e:
                        print(f"Savgol smoothing failed for column {j}: {e}")
        elif smooth_method == "lowess":
            frac = smooth_params.get("frac", 0.3)
            it = smooth_params.get("it", 3)
            for j in range(data.shape[1]):
                col = data[:, j]
                valid = ~np.isnan(col)
                col_valid = col[valid]
                pad_width = min(10, len(col_valid) // 2)
                if np.sum(valid) > 5:
                    try:
                        col_padded = pad_signal(col_valid, pad_width, mode="edge")
                        x = np.arange(len(col_padded))
                        col_filtered = lowess(col_padded, x, frac=frac, it=it, return_sorted=False)
                        col_filtered = col_filtered[pad_width:-pad_width]
                        col_smooth = col.copy()
                        col_smooth[valid] = col_filtered
                        data[:, j] = col_smooth
                    except Exception as e:
                        print(f"LOWESS smoothing failed for column {j}: {e}")
        elif smooth_method == "kalman":
            for j in range(data.shape[1]):
                col = data[:, j]
                valid = ~np.isnan(col)
                col_valid = col[valid]
                pad_width = min(10, len(col_valid) // 2)
                if np.sum(valid) > 5:
                    try:
                        col_padded = pad_signal(col_valid, pad_width, mode="edge")
                        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
                        col_filtered, _ = kf.smooth(col_padded)
                        col_filtered = col_filtered.flatten()[pad_width:-pad_width]
                        col_smooth = col.copy()
                        col_smooth[valid] = col_filtered
                        data[:, j] = col_smooth
                    except Exception as e:
                        print(f"Kalman smoothing failed for column {j}: {e}")
        elif smooth_method == "butterworth":
            cutoff = smooth_params.get("cutoff", 10)
            fs = smooth_params.get("fs", 100)
            for j in range(data.shape[1]):
                col = data[:, j]
                valid = ~np.isnan(col)
                col_valid = col[valid]
                pad_width = min(10, len(col_valid) // 2)
                if np.sum(valid) > 5:
                    try:
                        col_padded = pad_signal(col_valid, pad_width, mode="edge")
                        col_filtered = butter_filter(
                            col_padded,
                            fs=fs,
                            filter_type="low",
                            cutoff=cutoff,
                            order=4,
                            padding=False,
                        )
                        col_filtered = col_filtered[pad_width:-pad_width]
                        col_smooth = col.copy()
                        col_smooth[valid] = col_filtered
                        data[:, j] = col_smooth
                    except Exception as e:
                        print(f"Butterworth smoothing failed for column {j}: {e}")
        elif smooth_method == "splines":
            smoothing_factor = smooth_params.get("smoothing_factor", 1.0)
            for j in range(data.shape[1]):
                col = data[:, j]
                valid = ~np.isnan(col)
                col_valid = col[valid]
                pad_width = min(10, len(col_valid) // 2)
                if np.sum(valid) > 3:
                    try:
                        x = np.arange(len(col_valid))
                        x_padded = np.arange(-pad_width, len(col_valid) + pad_width)
                        col_padded = pad_signal(col_valid, pad_width, mode="edge")
                        spline = UnivariateSpline(x_padded, col_padded, s=smoothing_factor)
                        col_filtered = spline(x)
                        col_smooth = col.copy()
                        col_smooth[valid] = col_filtered
                        data[:, j] = col_smooth
                    except Exception as e:
                        print(f"Splines smoothing failed for column {j}: {e}")
        elif smooth_method == "arima":
            p = smooth_params.get("p", 1)
            d = smooth_params.get("d", 0)
            q = smooth_params.get("q", 0)
            order = (p, d, q)
            for j in range(data.shape[1]):
                col = data[:, j]
                valid = ~np.isnan(col)
                col_valid = col[valid]
                pad_width = min(10, len(col_valid) // 2)
                if np.sum(valid) > max(order) + 1:
                    try:
                        col_padded = pad_signal(col_valid, pad_width, mode="edge")
                        model = ARIMA(col_padded, order=order)
                        result = model.fit()
                        col_filtered = result.fittedvalues[pad_width:-pad_width]
                        col_smooth = col.copy()
                        col_smooth[valid] = col_filtered
                        data[:, j] = col_smooth
                    except Exception as e:
                        print(f"ARIMA smoothing failed for column {j}: {e}")
        df[numeric_cols] = data
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error during {smooth_method} smoothing: {str(e)}")
        print(f"Error during {smooth_method} smoothing: {str(e)}")
    return df


def map_landmarks_to_full_frame(
    landmarks,
    bbox_config,
    crop_width,
    crop_height,
    processing_width,
    processing_height,
    original_width=None,
    original_height=None,
):
    """
    Map landmarks from cropped/resized space to original video space.
    """
    enable_resize_crop = bbox_config.get("enable_resize_crop", False)
    resize_crop_scale = bbox_config.get("resize_crop_scale", 1)
    video_resized = bbox_config.get("video_resized", False)
    video_resize_scale = bbox_config.get("resize_scale", 1.0)

    if original_width is None or original_height is None:
        if video_resized and video_resize_scale > 1:
            original_width = int(processing_width / video_resize_scale)
            original_height = int(processing_height / video_resize_scale)
        else:
            original_width = processing_width
            original_height = processing_height

    bbox_x_min_processing = bbox_config.get("bbox_x_min", 0)
    bbox_y_min_processing = bbox_config.get("bbox_y_min", 0)

    mapped_landmarks = []
    for landmark in landmarks:
        x_norm_crop, y_norm_crop, z = landmark

        x_px_crop_resized = x_norm_crop * crop_width
        y_px_crop_resized = y_norm_crop * crop_height

        if enable_resize_crop and resize_crop_scale > 1:
            x_px_crop = x_px_crop_resized / resize_crop_scale
            y_px_crop = y_px_crop_resized / resize_crop_scale
        else:
            x_px_crop = x_px_crop_resized
            y_px_crop = y_px_crop_resized

        x_px_full_processing = x_px_crop + bbox_x_min_processing
        y_px_full_processing = y_px_crop + bbox_y_min_processing

        if video_resized and video_resize_scale > 1:
            x_px_full_original = x_px_full_processing / video_resize_scale
            y_px_full_original = y_px_full_processing / video_resize_scale
        else:
            x_px_full_original = x_px_full_processing
            y_px_full_original = y_px_full_processing

        x_norm_full = x_px_full_original / original_width if original_width > 0 else 0.0
        y_norm_full = y_px_full_original / original_height if original_height > 0 else 0.0

        mapped_landmarks.append([x_norm_full, y_norm_full, z])

    return mapped_landmarks


def convert_coordinates_to_original(df, metadata, progress_callback=None):
    """Convert coordinates from resized video back to original video dimensions."""
    if not metadata or metadata["scale_factor"] == 1:
        return df

    converted_df = df.copy()
    scale_factor = metadata["scale_factor"]

    if progress_callback:
        progress_callback(f"Converting coordinates back to original scale (/{scale_factor})")

    coord_columns = [col for col in df.columns if col.endswith("_x") or col.endswith("_y")]

    if progress_callback:
        progress_callback(f"Found {len(coord_columns)} coordinate columns to convert")

    processed = 0
    for col in coord_columns:
        if col.endswith("_x"):
            x_col = col
            y_col = col.replace("_x", "_y")

            if y_col in df.columns:
                for idx, row in df.iterrows():
                    if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                        original_x = row[x_col] / scale_factor
                        original_y = row[y_col] / scale_factor
                        converted_df.at[idx, x_col] = original_x
                        converted_df.at[idx, y_col] = original_y
                processed += 1

    if progress_callback:
        progress_callback(f"Converted {processed} coordinate pairs back to original scale")

    return converted_df


def resize_video_opencv(input_file, output_file, scale_factor, progress_callback=None):
    """Resize video using OpenCV and return metadata for coordinate conversion."""
    try:
        print(f"Resizing video: {input_file}")

        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_file}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        metadata = {
            "original_video": os.path.basename(input_file),
            "original_width": width,
            "original_height": height,
            "original_frames": total_frames,
            "scale_factor": scale_factor,
            "output_width": new_width,
            "output_height": new_height,
            "crop_applied": False,
        }

        if progress_callback:
            progress_callback(f"Resizing from {width}x{height} to {new_width}x{new_height}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_file, fourcc, 30, (new_width, new_height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )
            out.write(resized_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                if progress_callback:
                    progress_callback(f"Resizing: {progress:.1f}% ({frame_count}/{total_frames})")

        cap.release()
        out.release()

        if progress_callback:
            progress_callback(f"Video resized successfully: {output_file}")

        return metadata

    except Exception as e:
        print(f"Error during video resize: {str(e)}")
        if progress_callback:
            progress_callback(f"Error resizing video: {str(e)}")
        return None


# FaceMesh Configuration Dialog
class FaceMeshConfigDialog(tk.simpledialog.Dialog):
    def __init__(self, parent, input_dir=None):
        self.loaded_config = None
        self.use_toml = False
        self.toml_path = None
        self.input_dir = input_dir
        self.roi_polygon_points = None
        super().__init__(parent, title="FaceMesh Configuration (or TOML)")

    def body(self, master):
        # FaceMesh parameters
        tk.Label(master, text="min_detection_confidence (0.0 - 1.0):").grid(
            row=0, column=0, sticky="e"
        )
        tk.Label(master, text="min_tracking_confidence (0.0 - 1.0):").grid(
            row=1, column=0, sticky="e"
        )
        tk.Label(master, text="max_num_faces (1, 2, ...):").grid(row=2, column=0, sticky="e")
        tk.Label(master, text="refine_landmarks (True/False):").grid(row=3, column=0, sticky="e")
        tk.Label(master, text="apply_filtering (True/False):").grid(row=4, column=0, sticky="e")
        tk.Label(master, text="enable_resize (True/False):").grid(row=5, column=0, sticky="e")
        tk.Label(master, text="resize_scale (2, 3, ...):").grid(row=6, column=0, sticky="e")
        tk.Label(master, text="Enable initial frame padding? (True/False):").grid(
            row=7, column=0, sticky="e"
        )
        tk.Label(master, text="Number of padding frames:").grid(row=8, column=0, sticky="e")

        # Entries
        self.min_detection_entry = tk.Entry(master)
        self.min_detection_entry.insert(0, "0.25")
        self.min_tracking_entry = tk.Entry(master)
        self.min_tracking_entry.insert(0, "0.25")
        self.max_num_faces_entry = tk.Entry(master)
        self.max_num_faces_entry.insert(0, "1")
        self.refine_landmarks_entry = tk.Entry(master)
        self.refine_landmarks_entry.insert(0, "True")
        self.apply_filtering_entry = tk.Entry(master)
        self.apply_filtering_entry.insert(0, "True")
        self.enable_resize_entry = tk.Entry(master)
        self.enable_resize_entry.insert(0, "False")
        self.resize_scale_entry = tk.Entry(master)
        self.resize_scale_entry.insert(0, "2")
        self.enable_padding_entry = tk.Entry(master)
        self.enable_padding_entry.insert(0, str(ENABLE_PADDING_DEFAULT))
        self.pad_start_frames_entry = tk.Entry(master)
        self.pad_start_frames_entry.insert(0, str(PAD_START_FRAMES_DEFAULT))

        # Grid
        self.min_detection_entry.grid(row=0, column=1)
        self.min_tracking_entry.grid(row=1, column=1)
        self.max_num_faces_entry.grid(row=2, column=1)
        self.refine_landmarks_entry.grid(row=3, column=1)
        self.apply_filtering_entry.grid(row=4, column=1)
        self.enable_resize_entry.grid(row=5, column=1)
        self.resize_scale_entry.grid(row=6, column=1)
        self.enable_padding_entry.grid(row=7, column=1)
        self.pad_start_frames_entry.grid(row=8, column=1)

        # Bounding box section
        bbox_frame = tk.LabelFrame(master, text="Bounding Box (ROI) Selection", padx=10, pady=10)
        bbox_frame.grid(row=9, column=0, columnspan=2, pady=(10, 0), sticky="ew")

        self.enable_crop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bbox_frame, text="Enable bounding box cropping", variable=self.enable_crop_var
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

        tk.Label(bbox_frame, text="bbox_x_min (pixels):").grid(row=1, column=0, sticky="e", padx=5)
        self.bbox_x_min_entry = tk.Entry(bbox_frame, width=10)
        self.bbox_x_min_entry.insert(0, "0")
        self.bbox_x_min_entry.grid(row=1, column=1, sticky="w", padx=5)

        tk.Label(bbox_frame, text="bbox_y_min (pixels):").grid(row=2, column=0, sticky="e", padx=5)
        self.bbox_y_min_entry = tk.Entry(bbox_frame, width=10)
        self.bbox_y_min_entry.insert(0, "0")
        self.bbox_y_min_entry.grid(row=2, column=1, sticky="w", padx=5)

        tk.Label(bbox_frame, text="bbox_x_max (pixels):").grid(row=3, column=0, sticky="e", padx=5)
        self.bbox_x_max_entry = tk.Entry(bbox_frame, width=10)
        self.bbox_x_max_entry.insert(0, "1920")
        self.bbox_x_max_entry.grid(row=3, column=1, sticky="w", padx=5)

        tk.Label(bbox_frame, text="bbox_y_max (pixels):").grid(row=4, column=0, sticky="e", padx=5)
        self.bbox_y_max_entry = tk.Entry(bbox_frame, width=10)
        self.bbox_y_max_entry.insert(0, "1080")
        self.bbox_y_max_entry.grid(row=4, column=1, sticky="w", padx=5)

        tk.Label(bbox_frame, text="Normalized coordinates:").grid(
            row=5, column=0, sticky="e", padx=5, pady=(5, 0)
        )
        self.norm_coords_label = tk.Label(bbox_frame, text="(0.0, 0.0) to (1.0, 1.0)", fg="gray")
        self.norm_coords_label.grid(row=5, column=1, sticky="w", padx=5, pady=(5, 0))

        roi_buttons_frame = tk.Frame(bbox_frame)
        roi_buttons_frame.grid(row=6, column=0, columnspan=2, pady=10)

        select_bbox_roi_btn = tk.Button(
            roi_buttons_frame,
            text="Select BBox ROI",
            command=self.select_roi_from_video,
            bg="#FF9800",
            fg="black",
            width=18,
        )
        select_bbox_roi_btn.pack(side="left", padx=5)

        select_polygon_roi_btn = tk.Button(
            roi_buttons_frame,
            text="Select Polygon ROI",
            command=self.select_polygon_roi_from_video,
            bg="#4CAF50",
            fg="black",
            width=18,
        )
        select_polygon_roi_btn.pack(side="left", padx=5)

        self.enable_resize_crop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bbox_frame, text="Resize cropped region", variable=self.enable_resize_crop_var
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=5)

        tk.Label(bbox_frame, text="Resize crop scale:").grid(row=8, column=0, sticky="e", padx=5)
        self.resize_crop_scale_entry = tk.Entry(bbox_frame, width=10)
        self.resize_crop_scale_entry.insert(0, "2")
        self.resize_crop_scale_entry.grid(row=8, column=1, sticky="w", padx=5)

        for entry in [
            self.bbox_x_min_entry,
            self.bbox_y_min_entry,
            self.bbox_x_max_entry,
            self.bbox_y_max_entry,
        ]:
            entry.bind("<KeyRelease>", self.update_normalized_coords)

        # TOML section
        toml_frame = tk.LabelFrame(master, text="Advanced Configuration (TOML)", padx=10, pady=10)
        toml_frame.grid(row=10, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        btns_frame = tk.Frame(toml_frame)
        btns_frame.pack()
        tk.Button(btns_frame, text="Load Configuration TOML", command=self.load_config_file).pack(
            side="left", padx=5
        )
        tk.Button(
            btns_frame,
            text="Create Default TOML Template",
            command=self.create_default_toml_template,
        ).pack(side="left", padx=5)
        tk.Button(btns_frame, text="Help", command=self.show_help).pack(side="left", padx=5)
        self.toml_label = tk.Label(toml_frame, text="No TOML loaded", fg="gray")
        self.toml_label.pack()

        return self.min_detection_entry

    def update_normalized_coords(self, event=None):
        """Update normalized coordinates display based on pixel coordinates"""
        try:
            x_min = int(self.bbox_x_min_entry.get() or 0)
            y_min = int(self.bbox_y_min_entry.get() or 0)
            x_max = int(self.bbox_x_max_entry.get() or 1920)
            y_max = int(self.bbox_y_max_entry.get() or 1080)

            if self.input_dir:
                video_files = [
                    f
                    for f in Path(self.input_dir).glob("*.*")
                    if f.suffix.lower() in [".mp4", ".avi", ".mov"]
                ]
                if video_files:
                    cap = cv2.VideoCapture(str(video_files[0]))
                    if cap.isOpened():
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()

                        if width > 0 and height > 0:
                            x_min_norm = x_min / width
                            y_min_norm = y_min / height
                            x_max_norm = x_max / width
                            y_max_norm = y_max / height
                            self.norm_coords_label.config(
                                text=f"({x_min_norm:.3f}, {y_min_norm:.3f}) to ({x_max_norm:.3f}, {y_max_norm:.3f})"
                            )
                            return

            self.norm_coords_label.config(text="(Set video dimensions to calculate)")
        except Exception:
            self.norm_coords_label.config(text="(Invalid coordinates)")

    def select_roi_from_video(self):
        """Open first video and let user select ROI using cv2.selectROI"""
        if not self.input_dir:
            messagebox.showerror(
                "Error", "No input directory specified. Please select input directory first."
            )
            return

        video_files = [
            f
            for f in Path(self.input_dir).glob("*.*")
            if f.suffix.lower() in [".mp4", ".avi", ".mov"]
        ]

        if not video_files:
            messagebox.showerror("Error", "No video files found in input directory.")
            return

        first_video = video_files[0]
        cap = cv2.VideoCapture(str(first_video))

        if not cap.isOpened():
            messagebox.showerror("Error", f"Could not open video: {first_video}")
            return

        ret, original_frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Could not read first frame from video.")
            return

        orig_height, orig_width = original_frame.shape[:2]
        max_display_width = 1920
        max_display_height = 1080

        display_frame = original_frame.copy()
        scale_factor = 1.0

        if orig_width > max_display_width or orig_height > max_display_height:
            scale_w = max_display_width / orig_width
            scale_h = max_display_height / orig_height
            scale_factor = min(scale_w, scale_h)
            display_width = int(orig_width * scale_factor)
            display_height = int(orig_height * scale_factor)
            display_frame = cv2.resize(
                original_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR
            )
        else:
            display_width = orig_width
            display_height = orig_height

        window_name = "Select ROI - Press SPACE/ENTER to confirm, ESC to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)

        roi = cv2.selectROI(window_name, display_frame, False)

        if roi[2] == 0 or roi[3] == 0:
            print("ROI selection cancelled")
            cv2.destroyAllWindows()
            return

        x_display, y_display, w_display, h_display = roi

        if scale_factor != 1.0:
            x_min = int(x_display / scale_factor)
            y_min = int(y_display / scale_factor)
            x_max = int((x_display + w_display) / scale_factor)
            y_max = int((y_display + h_display) / scale_factor)
        else:
            x_min = x_display
            y_min = y_display
            x_max = x_display + w_display
            y_max = y_display + h_display

        x_min = max(0, min(x_min, orig_width - 1))
        y_min = max(0, min(y_min, orig_height - 1))
        x_max = max(x_min + 1, min(x_max, orig_width))
        y_max = max(y_min + 1, min(y_max, orig_height))

        self.bbox_x_min_entry.delete(0, tk.END)
        self.bbox_x_min_entry.insert(0, str(x_min))
        self.bbox_y_min_entry.delete(0, tk.END)
        self.bbox_y_min_entry.insert(0, str(y_min))
        self.bbox_x_max_entry.delete(0, tk.END)
        self.bbox_x_max_entry.insert(0, str(x_max))
        self.bbox_y_max_entry.delete(0, tk.END)
        self.bbox_y_max_entry.insert(0, str(y_max))

        self.update_normalized_coords()
        self.enable_crop_var.set(True)

        cv2.destroyAllWindows()
        print(f"ROI selected: ({x_min}, {y_min}) to ({x_max}, {y_max})")

    def select_polygon_roi_from_video(self):
        """Open first video and let user select polygon ROI"""
        if not self.input_dir:
            messagebox.showerror(
                "Error", "No input directory specified. Please select input directory first."
            )
            return

        video_files = [
            f
            for f in Path(self.input_dir).glob("*.*")
            if f.suffix.lower() in [".mp4", ".avi", ".mov"]
        ]

        if not video_files:
            messagebox.showerror("Error", "No video files found in input directory.")
            return

        first_video = video_files[0]
        roi_poly = select_free_polygon_roi(str(first_video))

        if roi_poly is not None and len(roi_poly) >= 3:
            x_coords = [pt[0] for pt in roi_poly]
            y_coords = [pt[1] for pt in roi_poly]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            self.bbox_x_min_entry.delete(0, tk.END)
            self.bbox_x_min_entry.insert(0, str(x_min))
            self.bbox_y_min_entry.delete(0, tk.END)
            self.bbox_y_min_entry.insert(0, str(y_min))
            self.bbox_x_max_entry.delete(0, tk.END)
            self.bbox_x_max_entry.insert(0, str(x_max))
            self.bbox_y_max_entry.delete(0, tk.END)
            self.bbox_y_max_entry.insert(0, str(y_max))

            self.roi_polygon_points = roi_poly.tolist() if hasattr(roi_poly, "tolist") else roi_poly

            self.update_normalized_coords()
            self.enable_crop_var.set(True)

            print(f"Polygon ROI selected with {len(roi_poly)} points")
            messagebox.showinfo(
                "ROI Selected",
                f"Polygon ROI selected with {len(roi_poly)} points.\nBounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})",
            )
        else:
            print("Polygon ROI selection cancelled or invalid")

    def create_default_toml_template(self):
        dialog_root = tk.Tk()
        dialog_root.withdraw()
        dialog_root.attributes("-topmost", True)

        file_path = filedialog.asksaveasfilename(
            parent=dialog_root,
            title="Create Default TOML Configuration Template",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialfile="facemesh_config_template.toml",
        )

        if file_path:
            default_config = get_default_config()
            save_config = {
                "min_detection_confidence": default_config["facemesh"]["min_detection_confidence"],
                "min_tracking_confidence": default_config["facemesh"]["min_tracking_confidence"],
                "max_num_faces": default_config["facemesh"]["max_num_faces"],
                "refine_landmarks": default_config["facemesh"]["refine_landmarks"],
                "apply_filtering": default_config["facemesh"]["apply_filtering"],
                "enable_resize": default_config["video_resize"]["enable_resize"],
                "resize_scale": default_config["video_resize"]["resize_scale"],
                "enable_advanced_filtering": default_config["advanced_filtering"][
                    "enable_advanced_filtering"
                ],
                "interp_method": default_config["advanced_filtering"]["interp_method"],
                "smooth_method": default_config["advanced_filtering"]["smooth_method"],
                "max_gap": default_config["advanced_filtering"]["max_gap"],
                "_all_smooth_params": default_config["smoothing_params"],
                "enable_padding": ENABLE_PADDING_DEFAULT,
                "pad_start_frames": PAD_START_FRAMES_DEFAULT,
                "enable_crop": default_config["roi"]["enable_crop"],
                "bbox_x_min": default_config["roi"]["bbox_x_min"],
                "bbox_y_min": default_config["roi"]["bbox_y_min"],
                "bbox_x_max": default_config["roi"]["bbox_x_max"],
                "bbox_y_max": default_config["roi"]["bbox_y_max"],
                "roi_polygon_points": default_config["roi"]["roi_polygon_points"],
                "enable_resize_crop": default_config["roi"]["enable_resize_crop"],
                "resize_crop_scale": default_config["roi"]["resize_crop_scale"],
            }
            ok = save_config_to_toml(save_config, file_path)
            if ok:
                messagebox.showinfo(
                    "Template Created",
                    f"Default TOML template created successfully:\n{file_path}",
                )
            else:
                messagebox.showerror("Error", "Failed to create template file.")

        dialog_root.destroy()

    def load_config_file(self):
        dialog_root = tk.Tk()
        dialog_root.withdraw()
        dialog_root.attributes("-topmost", True)

        file_path = filedialog.askopenfilename(
            parent=dialog_root,
            title="Select TOML file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )

        if file_path:
            try:
                config = load_config_from_toml(file_path)
                if config:
                    self.loaded_config = config
                    self.use_toml = True
                    self.toml_path = file_path
                    self.toml_label.config(
                        text=f"TOML loaded: {os.path.basename(file_path)}", fg="green"
                    )
                    self.populate_fields_from_config(config)
                    messagebox.showinfo("TOML Parameters Loaded", "Configuration loaded successfully!")
                else:
                    self.toml_label.config(text="Error loading TOML", fg="red")
            except Exception as e:
                self.toml_label.config(text=f"Error: {e}", fg="red")
                messagebox.showerror("Error", f"Failed to load TOML: {e}")

        dialog_root.destroy()

    def populate_fields_from_config(self, config):
        """Populate GUI fields with values from loaded TOML config"""
        self.min_detection_entry.delete(0, tk.END)
        self.min_detection_entry.insert(0, str(config.get("min_detection_confidence", 0.25)))
        self.min_tracking_entry.delete(0, tk.END)
        self.min_tracking_entry.insert(0, str(config.get("min_tracking_confidence", 0.25)))
        self.max_num_faces_entry.delete(0, tk.END)
        self.max_num_faces_entry.insert(0, str(config.get("max_num_faces", 1)))
        self.refine_landmarks_entry.delete(0, tk.END)
        self.refine_landmarks_entry.insert(0, str(config.get("refine_landmarks", True)))
        self.apply_filtering_entry.delete(0, tk.END)
        self.apply_filtering_entry.insert(0, str(config.get("apply_filtering", True)))
        self.enable_resize_entry.delete(0, tk.END)
        self.enable_resize_entry.insert(0, str(config.get("enable_resize", False)))
        self.resize_scale_entry.delete(0, tk.END)
        self.resize_scale_entry.insert(0, str(config.get("resize_scale", 2)))
        self.enable_padding_entry.delete(0, tk.END)
        self.enable_padding_entry.insert(0, str(config.get("enable_padding", ENABLE_PADDING_DEFAULT)))
        self.pad_start_frames_entry.delete(0, tk.END)
        self.pad_start_frames_entry.insert(0, str(config.get("pad_start_frames", PAD_START_FRAMES_DEFAULT)))

        enable_crop = config.get("enable_crop", False)
        self.enable_crop_var.set(enable_crop)
        self.bbox_x_min_entry.delete(0, tk.END)
        self.bbox_x_min_entry.insert(0, str(config.get("bbox_x_min", 0)))
        self.bbox_y_min_entry.delete(0, tk.END)
        self.bbox_y_min_entry.insert(0, str(config.get("bbox_y_min", 0)))
        self.bbox_x_max_entry.delete(0, tk.END)
        self.bbox_x_max_entry.insert(0, str(config.get("bbox_x_max", 1920)))
        self.bbox_y_max_entry.delete(0, tk.END)
        self.bbox_y_max_entry.insert(0, str(config.get("bbox_y_max", 1080)))
        self.enable_resize_crop_var.set(config.get("enable_resize_crop", False))
        self.resize_crop_scale_entry.delete(0, tk.END)
        self.resize_crop_scale_entry.insert(0, str(config.get("resize_crop_scale", 2)))

        roi_polygon = config.get("roi_polygon_points")
        if roi_polygon:
            self.roi_polygon_points = roi_polygon

        self.update_normalized_coords()

    def show_help(self):
        """Show help window"""
        help_window = tk.Toplevel()
        help_window.title("FaceMesh Analysis - Help")
        help_window.geometry("800x600")
        help_window.configure(bg="white")
        help_window.transient()
        help_window.attributes("-topmost", True)
        help_window.focus_set()

        text_frame = tk.Frame(help_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Arial", 10),
            bg="white",
            fg="black",
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        help_text = """
FACEMESH ANALYSIS - HELP GUIDE
===============================

OVERVIEW
--------
This script performs batch processing of videos for 2D face mesh detection using MediaPipe's FaceMesh model.
It processes videos from a specified input directory, overlays face landmarks on each video frame,
and exports both normalized and pixel-based landmark coordinates to CSV files.

FEATURES
--------
• Video resize functionality for better face detection
• Bounding Box (BBox) ROI selection for rectangular regions
• Polygon ROI selection for free-form regions
• Crop resize: Optional upscaling of cropped region
• Initial frame padding for MediaPipe stabilization
• Advanced filtering and smoothing options
• TOML configuration file support

ROI SELECTION
------------
1. Bounding Box ROI: Drag to select rectangular region
2. Polygon ROI: Left click to add points, right click to undo, Enter to confirm

OUTPUT FILES
-----------
For each processed video:
1. Processed Video (*_facemesh.mp4): Video with face landmarks overlaid
2. Normalized CSV (*_facemesh_norm.csv): Landmarks normalized to 0-1 scale
3. Pixel CSV (*_facemesh_pixel.csv): Landmarks in pixel coordinates
4. Configuration file: Settings used for processing

For more information, visit: https://github.com/vaila-multimodaltoolbox/vaila
        """

        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)

        close_btn = tk.Button(
            help_window,
            text="Close",
            command=help_window.destroy,
            font=("Arial", 10),
            bg="#4CAF50",
            fg="white",
            padx=20,
        )
        close_btn.pack(pady=10)

    def apply(self):
        if self.use_toml and self.loaded_config:
            self.result = self.loaded_config
        else:
            self.result = {
                "min_detection_confidence": float(self.min_detection_entry.get()),
                "min_tracking_confidence": float(self.min_tracking_entry.get()),
                "max_num_faces": int(self.max_num_faces_entry.get()),
                "refine_landmarks": self.refine_landmarks_entry.get().lower() == "true",
                "apply_filtering": self.apply_filtering_entry.get().lower() == "true",
                "enable_resize": self.enable_resize_entry.get().lower() == "true",
                "resize_scale": int(self.resize_scale_entry.get()),
                "enable_advanced_filtering": False,
                "interp_method": "linear",
                "smooth_method": "none",
                "max_gap": 60,
                "_all_smooth_params": get_default_config()["smoothing_params"],
                "enable_padding": self.enable_padding_entry.get().lower() == "true",
                "pad_start_frames": int(self.pad_start_frames_entry.get()),
                "enable_crop": self.enable_crop_var.get(),
                "bbox_x_min": int(self.bbox_x_min_entry.get() or 0),
                "bbox_y_min": int(self.bbox_y_min_entry.get() or 0),
                "bbox_x_max": int(self.bbox_x_max_entry.get() or 1920),
                "bbox_y_max": int(self.bbox_y_max_entry.get() or 1080),
                "enable_resize_crop": self.enable_resize_crop_var.get(),
                "resize_crop_scale": int(self.resize_crop_scale_entry.get() or 2),
                "roi_polygon_points": getattr(self, "roi_polygon_points", None),
            }


# Get face configuration from dialog
def get_face_model_path():
    """Download the FaceLandmarker model for MediaPipe Tasks API"""
    # Use vaila/models directory for storing models
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_name = "face_landmarker.task"
    model_path = models_dir / model_name

    if not model_path.exists():
        print(f"Downloading MediaPipe FaceLandmarker model ({model_name})... please wait.")
        print(f"Download location: {model_path}")
        # URL for FaceLandmarker model
        model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            urllib.request.urlretrieve(model_url, str(model_path))
            print("Download completed!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise RuntimeError("Failed to download MediaPipe FaceLandmarker model.") from e
    return str(model_path.resolve())


def get_face_config(existing_root=None, input_dir=None):
    """Get FaceMesh configuration from dialog"""
    if existing_root is not None:
        root = existing_root
    else:
        root = tk.Tk()
        root.withdraw()

    dialog = FaceMeshConfigDialog(root, input_dir=input_dir)
    if dialog.result:
        print("Configuration applied successfully!")
        return dialog.result
    else:
        messagebox.showerror("Error", "Configuration cancelled - no values entered.")
        return None


# Process frame with FaceMesh using Tasks API (adapted from process_frame_with_tasks_api)
def process_frame_with_facemesh(
    frame,
    landmarker,
    timestamp_ms,
    enable_crop,
    bbox_config,
    process_width,
    process_height,
    original_width,
    original_height,
    face_config,
):
    """
    Process a single frame using MediaPipe FaceLandmarker (Tasks API).
    Returns landmarks in normalized coordinates (mapped to full frame if cropping was used).
    Supports both bounding box and polygon ROI.
    """
    process_frame = frame
    actual_process_width = process_width
    actual_process_height = process_height

    if enable_crop:
        roi_polygon_points = bbox_config.get("roi_polygon_points")

        if roi_polygon_points and len(roi_polygon_points) >= 3:
            # Polygon ROI
            if bbox_config.get("video_resized", False):
                resize_scale = bbox_config.get("resize_scale", 1.0)
                scaled_polygon = [
                    [int(pt[0] * resize_scale), int(pt[1] * resize_scale)]
                    for pt in roi_polygon_points
                ]
            else:
                scaled_polygon = [[int(pt[0]), int(pt[1])] for pt in roi_polygon_points]

            mask = np.zeros((process_height, process_width), dtype=np.uint8)
            polygon_pts = np.array(scaled_polygon, dtype=np.int32)
            cv2.fillPoly(mask, [polygon_pts], 255)
            process_frame = cv2.bitwise_and(frame, frame, mask=mask)

            x_coords = [pt[0] for pt in scaled_polygon]
            y_coords = [pt[1] for pt in scaled_polygon]
            x_min = max(0, min(x_coords))
            y_min = max(0, min(y_coords))
            x_max = min(process_width, max(x_coords))
            y_max = min(process_height, max(y_coords))

            process_frame = process_frame[y_min:y_max, x_min:x_max]
            actual_process_width = x_max - x_min
            actual_process_height = y_max - y_min

            bbox_config["polygon_offset_x"] = x_min
            bbox_config["polygon_offset_y"] = y_min
        else:
            # Bounding box ROI
            x_min = max(0, min(bbox_config["bbox_x_min"], process_width - 1))
            y_min = max(0, min(bbox_config["bbox_y_min"], process_height - 1))
            x_max = max(x_min + 1, min(bbox_config["bbox_x_max"], process_width))
            y_max = max(y_min + 1, min(bbox_config["bbox_y_max"], process_height))
            process_frame = frame[y_min:y_max, x_min:x_max]
            actual_process_width = x_max - x_min
            actual_process_height = y_max - y_min

        if (
            bbox_config.get("enable_resize_crop", False)
            and bbox_config.get("resize_crop_scale", 1) > 1
        ):
            new_w = int(actual_process_width * bbox_config["resize_crop_scale"])
            new_h = int(actual_process_height * bbox_config["resize_crop_scale"])
            process_frame = cv2.resize(
                process_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            actual_process_width = new_w
            actual_process_height = new_h

    # Convert to RGB and create MP Image
    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect faces using Tasks API
    face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Parse results
    all_faces_landmarks = []
    if face_landmarker_result.face_landmarks and len(face_landmarker_result.face_landmarks) > 0:
        for face_idx, face_landmarks in enumerate(face_landmarker_result.face_landmarks):
            landmarks = [[lm.x, lm.y, lm.z] for lm in face_landmarks]

            # Map coordinates to full frame if cropping was used
            if enable_crop:
                if bbox_config.get("roi_polygon_points") and len(
                    bbox_config.get("roi_polygon_points", [])
                ) >= 3:
                    # Polygon ROI mapping
                    offset_x = bbox_config.get("polygon_offset_x", 0)
                    offset_y = bbox_config.get("polygon_offset_y", 0)
                    enable_resize_crop = bbox_config.get("enable_resize_crop", False)
                    resize_crop_scale = bbox_config.get("resize_crop_scale", 1)

                    for lm in landmarks:
                        x_px_crop_resized = lm[0] * actual_process_width
                        y_px_crop_resized = lm[1] * actual_process_height

                        if enable_resize_crop and resize_crop_scale > 1:
                            x_px_crop = x_px_crop_resized / resize_crop_scale
                            y_px_crop = y_px_crop_resized / resize_crop_scale
                        else:
                            x_px_crop = x_px_crop_resized
                            y_px_crop = y_px_crop_resized

                        x_px_full_processing = x_px_crop + offset_x
                        y_px_full_processing = y_px_crop + offset_y

                        if bbox_config.get("video_resized", False):
                            resize_scale = bbox_config.get("resize_scale", 1.0)
                            x_px_full_original = x_px_full_processing / resize_scale
                            y_px_full_original = y_px_full_processing / resize_scale
                        else:
                            x_px_full_original = x_px_full_processing
                            y_px_full_original = y_px_full_processing

                        lm[0] = x_px_full_original / original_width if original_width > 0 else 0.0
                        lm[1] = y_px_full_original / original_height if original_height > 0 else 0.0
                else:
                    # Bounding box ROI mapping
                    landmarks = map_landmarks_to_full_frame(
                        landmarks,
                        bbox_config,
                        actual_process_width,
                        actual_process_height,
                        process_width,
                        process_height,
                        original_width,
                        original_height,
                    )

            all_faces_landmarks.append((face_idx, landmarks))

    return all_faces_landmarks if all_faces_landmarks else None


# Process video with FaceMesh
def process_video(video_path, output_dir, face_config, use_gpu=False, gpu_type="nvidia"):
    """
    Process a video file using MediaPipe FaceMesh with optional video resize, ROI, and filtering.
    
    Note: use_gpu and gpu_type parameters are accepted for future compatibility,
    but MediaPipe FaceMesh currently uses CPU processing only.
    GPU acceleration may be available in future MediaPipe versions.
    """
    print("\n=== Parameters being used for this video ===")
    for k, v in face_config.items():
        if k != "roi_polygon_points":  # Skip large polygon data in print
            print(f"{k}: {v}")
    print("==========================================\n")

    print(f"Processing video: {video_path}")
    start_time = time.time()

    enable_resize = face_config.get("enable_resize", False)
    resize_scale = face_config.get("resize_scale", 2)

    orig_cap = cv2.VideoCapture(str(video_path))
    if not orig_cap.isOpened():
        print(f"Failed to open original video: {video_path}")
        return
    original_width = int(orig_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(orig_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_cap.release()

    output_dir.mkdir(parents=True, exist_ok=True)

    processing_video_path = video_path
    resize_metadata = None
    temp_resized_video = None

    # Step 1: Resize video if enabled
    if enable_resize and resize_scale > 1:
        print(f"Step 1/4: Resizing video by {resize_scale}x for better face detection")
        temp_dir = tempfile.mkdtemp()
        temp_resized_video = os.path.join(temp_dir, f"temp_resized_{resize_scale}x.mp4")
        resize_metadata = resize_video_opencv(
            str(video_path),
            temp_resized_video,
            resize_scale,
            lambda msg: print(f"  {msg}"),
        )
        if resize_metadata:
            processing_video_path = temp_resized_video
            print("Video resized successfully for processing")
        else:
            print("Failed to resize video, using original")
            enable_resize = False

    cap = cv2.VideoCapture(str(processing_video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {processing_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    # Output files
    output_video_path = output_dir / f"{video_path.stem}_facemesh.mp4"
    output_file_path = output_dir / f"{video_path.stem}_facemesh_norm.csv"
    output_pixel_file_path = output_dir / f"{video_path.stem}_facemesh_pixel.csv"

    # Initialize MediaPipe Tasks API
    # Note: GPU acceleration not yet available for FaceMesh in MediaPipe
    # The use_gpu parameter is accepted for future compatibility
    if use_gpu:
        print(f"Note: GPU ({gpu_type}) selected, but FaceMesh will use CPU processing")
        print("GPU acceleration may be available in future MediaPipe versions")
    
    model_path = get_face_model_path()

    BaseOptions = mp.tasks.BaseOptions  # noqa: N806 - MediaPipe class name
    FaceLandmarker = mp.tasks.vision.FaceLandmarker  # noqa: N806 - MediaPipe class name
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions  # noqa: N806 - MediaPipe class name
    VisionRunningMode = mp.tasks.vision.RunningMode  # noqa: N806 - MediaPipe class name

    # Create options with CPU or GPU Delegate
    # Note: MediaPipe FaceLandmarker may not fully support GPU delegate yet
    # Even when GPU delegate is set, it may fall back to CPU (XNNPACK)
    # This is a known limitation of MediaPipe FaceLandmarker
    # #region agent log
    _debug_log("C", "mp_facemesh_nvidia.py:2417", "Before delegate selection", {
        "use_gpu": use_gpu,
        "gpu_type": gpu_type,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "has_gpu_delegate": hasattr(BaseOptions.Delegate, "GPU") if hasattr(BaseOptions, "Delegate") else False,
    })
    # #endregion
    if use_gpu and gpu_type == "nvidia":
        # Ensure CUDA_VISIBLE_DEVICES is set to use only NVIDIA GPU (device 0)
        # This must be set BEFORE importing MediaPipe or creating options
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # #region agent log
        _debug_log("C", "mp_facemesh_nvidia.py:2425", "Setting CUDA_VISIBLE_DEVICES=0 for GPU", {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
        # #endregion
        try:
            # Try GPU delegate (NVIDIA CUDA)
            delegate = BaseOptions.Delegate.GPU
            # #region agent log
            _debug_log("C", "mp_facemesh_nvidia.py:2430", "GPU delegate selected", {
                "delegate": "GPU",
                "delegate_value": str(delegate),
            })
            # #endregion
            print("Attempting to use GPU (NVIDIA CUDA) for processing")
            print("Note: FaceLandmarker may still use CPU (XNNPACK) if GPU delegate is not supported")
        except Exception as e:
            # #region agent log
            _debug_log("C", "mp_facemesh_nvidia.py:2437", "Error setting GPU delegate", {
                "error": str(e),
                "error_type": type(e).__name__,
            })
            # #endregion
            print(f"Warning: Could not set GPU delegate: {e}")
            print("Falling back to CPU")
            delegate = BaseOptions.Delegate.CPU
            use_gpu = False
    else:
        delegate = BaseOptions.Delegate.CPU
        # #region agent log
        _debug_log("C", "mp_facemesh_nvidia.py:2445", "CPU delegate selected", {
            "use_gpu": use_gpu,
            "gpu_type": gpu_type,
            "reason": "use_gpu=False or gpu_type != nvidia",
        })
        # #endregion
        if use_gpu:
            print(f"Note: GPU ({gpu_type}) selected, but FaceMesh will use CPU processing")
            print("GPU acceleration may be available in future MediaPipe versions")
        else:
            print("Using CPU for processing")

    # Create options
    # Note: Even with GPU delegate, FaceLandmarker may use CPU (XNNPACK)
    # This is a known limitation - FaceLandmarker GPU support is experimental
    # #region agent log
    _debug_log("D", "mp_facemesh_nvidia.py:2442", "Before creating FaceLandmarkerOptions", {
        "delegate": str(delegate),
        "delegate_type": type(delegate).__name__ if hasattr(delegate, "__class__") else "unknown",
        "model_path": str(model_path),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    # #endregion
    try:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=face_config.get("max_num_faces", 1),
            min_face_detection_confidence=face_config.get("min_detection_confidence", 0.25),
            min_face_presence_confidence=face_config.get("min_tracking_confidence", 0.25),
            min_tracking_confidence=face_config.get("min_tracking_confidence", 0.25),
            output_face_blendshapes=face_config.get("refine_landmarks", True),
        )
        # #region agent log
        _debug_log("D", "mp_facemesh_nvidia.py:2455", "FaceLandmarkerOptions created successfully", {
            "delegate_used": str(delegate),
            "options_created": True,
        })
        # #endregion
        if delegate == BaseOptions.Delegate.GPU:
            print("GPU delegate configured, but FaceLandmarker may still use CPU")
            print("Check MediaPipe logs for actual device used (XNNPACK = CPU)")
    except Exception as e:
        # #region agent log
        _debug_log("D", "mp_facemesh_nvidia.py:2462", "Error creating FaceLandmarkerOptions", {
            "error": str(e),
            "error_type": type(e).__name__,
            "delegate": str(delegate),
        })
        # #endregion
        if "GPU" in str(e) or "delegate" in str(e).lower():
            print(f"Warning: GPU delegate failed, falling back to CPU: {e}")
            delegate = BaseOptions.Delegate.CPU
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path, delegate=delegate),
                running_mode=VisionRunningMode.VIDEO,
                num_faces=face_config.get("max_num_faces", 1),
                min_face_detection_confidence=face_config.get("min_detection_confidence", 0.25),
                min_face_presence_confidence=face_config.get("min_tracking_confidence", 0.25),
                min_tracking_confidence=face_config.get("min_tracking_confidence", 0.25),
                output_face_blendshapes=face_config.get("refine_landmarks", True),
            )
        else:
            raise

    # Generate CSV headers (will be adjusted if actual landmark count differs)
    header, indices = generate_csv_header()
    original_header = header.copy()

    # Lists to store landmarks
    all_frames_data = []  # List of dicts: {frame_idx: int, faces: list of face data}
    frames_with_missing_data = []

    enable_padding = face_config.get("enable_padding", ENABLE_PADDING_DEFAULT)
    pad_start_frames = face_config.get("pad_start_frames", PAD_START_FRAMES_DEFAULT)

    print(f"Padding: enable={enable_padding}, frames={pad_start_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read first frame.")
        cap.release()
        return

    padding_frame = first_frame.copy() if enable_padding and pad_start_frames > 0 else None

    # Extract ROI config
    enable_crop = face_config.get("enable_crop", False)
    if enable_resize and resize_metadata:
        resize_scale = resize_metadata["scale_factor"]
        bbox_x_min_orig = face_config.get("bbox_x_min", 0)
        bbox_y_min_orig = face_config.get("bbox_y_min", 0)
        bbox_x_max_orig = face_config.get("bbox_x_max", original_width)
        bbox_y_max_orig = face_config.get("bbox_y_max", original_height)
        bbox_x_min_scaled = int(bbox_x_min_orig * resize_scale)
        bbox_y_min_scaled = int(bbox_y_min_orig * resize_scale)
        bbox_x_max_scaled = int(bbox_x_max_orig * resize_scale)
        bbox_y_max_scaled = int(bbox_y_max_orig * resize_scale)
    else:
        bbox_x_min_scaled = face_config.get("bbox_x_min", 0)
        bbox_y_min_scaled = face_config.get("bbox_y_min", 0)
        bbox_x_max_scaled = face_config.get("bbox_x_max", width)
        bbox_y_max_scaled = face_config.get("bbox_y_max", height)

    bbox_config = {
        "enable_resize_crop": face_config.get("enable_resize_crop", False),
        "resize_crop_scale": face_config.get("resize_crop_scale", 2),
        "bbox_x_min": bbox_x_min_scaled,
        "bbox_y_min": bbox_y_min_scaled,
        "bbox_x_max": bbox_x_max_scaled,
        "bbox_y_max": bbox_y_max_scaled,
        "bbox_x_min_orig": face_config.get("bbox_x_min", 0),
        "bbox_y_min_orig": face_config.get("bbox_y_min", 0),
        "bbox_x_max_orig": face_config.get("bbox_x_max", original_width),
        "bbox_y_max_orig": face_config.get("bbox_y_max", original_height),
        "video_resized": enable_resize and resize_metadata is not None,
        "resize_scale": resize_metadata["scale_factor"] if (enable_resize and resize_metadata) else 1.0,
        "roi_polygon_points": face_config.get("roi_polygon_points"),
    }

    step_text = "Step 2/4" if enable_resize else "Step 1/3"
    print(f"\n{step_text}: Processing landmarks (total frames: {total_frames})")

    frame_count = 0

    # Process with Tasks API
    # #region agent log
    _debug_log("E", "mp_facemesh_nvidia.py:2478", "Before creating FaceLandmarker", {
        "delegate": str(delegate),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })
    # #endregion
    with FaceLandmarker.create_from_options(options) as landmarker:
        # #region agent log
        _debug_log("E", "mp_facemesh_nvidia.py:2482", "FaceLandmarker created", {
            "landmarker_created": True,
            "delegate_configured": str(delegate),
        })
        # #endregion
        # Process padding frames first if enabled
        if enable_padding and pad_start_frames > 0 and padding_frame is not None:
            print(f"Processing {pad_start_frames} padding frames...")
            for pad_idx in range(pad_start_frames):
                if should_throttle_cpu(frame_count):
                    apply_cpu_throttling()

                timestamp_ms = int((frame_count * 1000) / fps) if fps > 0 else frame_count * 33
                faces_result = process_frame_with_facemesh(
                    padding_frame,
                    landmarker,
                    timestamp_ms,
                    enable_crop,
                    bbox_config,
                    width,
                    height,
                    original_width,
                    original_height,
                    face_config,
                )

                if faces_result:
                    all_frames_data.append({"frame_idx": frame_count, "faces": faces_result})
                else:
                    all_frames_data.append({"frame_idx": frame_count, "faces": []})
                    frames_with_missing_data.append(frame_count)

                frame_count += 1
                time.sleep(FRAME_SLEEP_TIME)

        # Reset and process real frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if should_throttle_cpu(frame_count):
                apply_cpu_throttling()

            timestamp_ms = int((frame_count * 1000) / fps) if fps > 0 else frame_count * 33
            faces_result = process_frame_with_facemesh(
                frame,
                landmarker,
                timestamp_ms,
                enable_crop,
                bbox_config,
                width,
                height,
                original_width,
                original_height,
                face_config,
            )

            if faces_result:
                all_frames_data.append({"frame_idx": frame_count, "faces": faces_result})
            else:
                all_frames_data.append({"frame_idx": frame_count, "faces": []})
                frames_with_missing_data.append(frame_count)

            frame_count += 1
            time.sleep(FRAME_SLEEP_TIME)

            if frame_count % 100 == 0:
                print(f"  Processed {frame_count}/{total_frames + (pad_start_frames if enable_padding else 0)} frames")

    print(f"Finished processing all {frame_count} frames.")
    cap.release()
    cv2.destroyAllWindows()

    # Remove padding frames from results
    if enable_padding and pad_start_frames > 0:
        print(f"Removing {pad_start_frames} padding frames from results")
        all_frames_data = all_frames_data[pad_start_frames:]
        frames_with_missing_data = [f - pad_start_frames for f in frames_with_missing_data if f >= pad_start_frames]
        # Adjust frame indices
        for i, frame_data in enumerate(all_frames_data):
            frame_data["frame_idx"] = i

    # Convert to DataFrames
    step_text = "Step 3/4" if enable_resize else "Step 2/3"
    print(f"\n{step_text}: Converting landmarks to DataFrames")

    norm_rows = []
    pixel_rows = []

    # Determine actual number of landmarks from first frame with faces
    actual_num_landmarks = NUM_FACE_LANDMARKS
    for frame_data in all_frames_data:
        faces = frame_data["faces"]
        if faces and len(faces) > 0:
            # Get number of landmarks from first face
            first_face_landmarks = faces[0][1]  # faces[0] is (face_idx, landmarks)
            if len(first_face_landmarks) != NUM_FACE_LANDMARKS:
                actual_num_landmarks = len(first_face_landmarks)
                print(f"Warning: FaceLandmarker returned {actual_num_landmarks} landmarks, expected {NUM_FACE_LANDMARKS}")
                print(f"Adjusting CSV header to match actual landmark count")
                # Regenerate header with correct number of landmarks
                header = ["frame_index", "face_idx"]  # Include face_idx to match row structure
                for idx in range(actual_num_landmarks):
                    name = LANDMARK_NAMES.get(idx, f"landmark_{idx}")
                    header.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
                break
    
    # If header was adjusted, we need to ensure all rows match
    expected_cols = len(header)

    for frame_data in all_frames_data:
        frame_idx = frame_data["frame_idx"]
        faces = frame_data["faces"]

        if faces:
            for face_idx, landmarks in faces:
                norm_row = [frame_idx, face_idx]
                pixel_row = [frame_idx, face_idx]

                # Ensure we have the correct number of landmarks
                if len(landmarks) != actual_num_landmarks:
                    # Pad or truncate to match expected count
                    if len(landmarks) < actual_num_landmarks:
                        # Pad with NaN
                        landmarks = landmarks + [[np.nan, np.nan, np.nan]] * (actual_num_landmarks - len(landmarks))
                    else:
                        # Truncate
                        landmarks = landmarks[:actual_num_landmarks]

                # Add landmarks to rows (each landmark has 3 values: x, y, z)
                for landmark in landmarks:
                    if isinstance(landmark, (list, tuple)) and len(landmark) >= 3:
                        norm_row.extend([landmark[0], landmark[1], landmark[2]])  # Already normalized
                        pixel_row.extend([
                            int(landmark[0] * original_width) if not (np.isnan(landmark[0]) if hasattr(np, 'isnan') else landmark[0] != landmark[0]) else np.nan,
                            int(landmark[1] * original_height) if not (np.isnan(landmark[1]) if hasattr(np, 'isnan') else landmark[1] != landmark[1]) else np.nan,
                            landmark[2] if not (np.isnan(landmark[2]) if hasattr(np, 'isnan') else landmark[2] != landmark[2]) else np.nan
                        ])
                    else:
                        # Invalid landmark format, add NaN
                        norm_row.extend([np.nan, np.nan, np.nan])
                        pixel_row.extend([np.nan, np.nan, np.nan])
                
                # Verify row length matches header
                if len(norm_row) != expected_cols:
                    print(f"Warning: Row length mismatch for frame {frame_idx}, face {face_idx}: expected {expected_cols}, got {len(norm_row)}")
                    # Adjust row to match header
                    if len(norm_row) < expected_cols:
                        norm_row.extend([np.nan] * (expected_cols - len(norm_row)))
                        pixel_row.extend([np.nan] * (expected_cols - len(pixel_row)))
                    else:
                        norm_row = norm_row[:expected_cols]
                        pixel_row = pixel_row[:expected_cols]

                norm_rows.append(norm_row)
                pixel_rows.append(pixel_row)
        else:
            # No face detected - add row with NaN values
            norm_row = [frame_idx, 0]
            pixel_row = [frame_idx, 0]
            for _ in range(actual_num_landmarks):
                norm_row.extend([np.nan, np.nan, np.nan])
                pixel_row.extend([np.nan, np.nan, np.nan])
            norm_rows.append(norm_row)
            pixel_rows.append(pixel_row)

    # Create DataFrames
    df_norm = pd.DataFrame(norm_rows, columns=header)
    df_pixel = pd.DataFrame(pixel_rows, columns=header)

    # Apply advanced filtering if enabled
    if face_config.get("enable_advanced_filtering", False):
        step_text = "Step 4/5" if enable_resize else "Step 3/4"
        print(f"\n{step_text}: Applying advanced filtering and smoothing")
        df_norm = apply_interpolation_and_smoothing(
            df_norm, face_config, lambda msg: print(f"  Normalized: {msg}")
        )
        df_pixel = apply_interpolation_and_smoothing(
            df_pixel, face_config, lambda msg: print(f"  Pixel: {msg}")
        )

    # Convert pixel coordinates to original size if resize was used
    if enable_resize and resize_metadata:
        df_pixel = convert_coordinates_to_original(
            df_pixel, resize_metadata, lambda msg: print(f"  {msg}")
        )

    # Save CSVs
    step_text = "Step 4/4" if enable_resize else "Step 3/3"
    print(f"\n{step_text}: Saving processed CSVs")
    df_norm.to_csv(output_file_path, index=False, float_format="%.6f")
    df_pixel.to_csv(output_pixel_file_path, index=False)
    print(f"Saved: {output_file_path} (normalized)")
    print(f"Saved: {output_pixel_file_path} (pixel)")

    # Generate video with landmarks
    print(f"\n{step_text}: Generating video with processed landmarks")
    cap = cv2.VideoCapture(str(video_path))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    from fractions import Fraction
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if not orig_fps or np.isnan(orig_fps) or orig_fps < 1 or orig_fps > 240:
        safe_fps = 30.0
    else:
        safe_fps = float(Fraction(orig_fps).limit_denominator(1000))

    temp_output_video_path = output_dir / f"{video_path.stem}_facemesh_tmp.mp4"
    # Try H.264 (avc1) first for better compression/compatibility, fallback to mp4v
    try:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(
            str(temp_output_video_path), fourcc, safe_fps, (original_width, original_height)
        )
        if not out.isOpened():
            raise ValueError("avc1 codec not available")
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(temp_output_video_path), fourcc, safe_fps, (original_width, original_height)
        )

    if not out.isOpened():
        safe_fps = 30.0
        out = cv2.VideoWriter(
            str(temp_output_video_path),
            fourcc,
            safe_fps,
            (original_width, original_height),
        )

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"\r Generating video {frame_idx}/{total_frames} ({progress:.1f}%)", end="")

        # Get processed landmarks for this frame
        frame_faces = None
        for frame_data in all_frames_data:
            if frame_data["frame_idx"] == frame_idx:
                frame_faces = frame_data["faces"]
                break

        if frame_faces:
            for face_idx, landmarks in frame_faces:
                # Draw landmarks
                points = {}
                for i, lm in enumerate(landmarks):
                    if not np.isnan(lm[0]) and not np.isnan(lm[1]):
                        x = int(lm[0] * original_width)
                        y = int(lm[1] * original_height)
                        points[i] = (x, y)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                # Draw connections
                for connection in DRAW_CONNECTIONS:
                    if connection[0] in points and connection[1] in points:
                        cv2.line(frame, points[connection[0]], points[connection[1]], (255, 0, 0), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Finalize video with ffmpeg if available
    try:
        if temp_output_video_path.exists() and temp_output_video_path.stat().st_size > 0:
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path:
                cmd = [
                    ffmpeg_path,
                    "-y",
                    "-i",
                    str(temp_output_video_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_video_path),
                ]
                subprocess.run(cmd, check=False, capture_output=True)
                if output_video_path.exists() and output_video_path.stat().st_size > 0:
                    print(f"Saved final video (H.264): {output_video_path}")
                    os.remove(temp_output_video_path)
                else:
                    shutil.move(str(temp_output_video_path), str(output_video_path))
            else:
                shutil.move(str(temp_output_video_path), str(output_video_path))
                print(f"Saved final video (mp4v): {output_video_path}")
    except Exception as e:
        print(f"Warning: Could not finalize video: {e}")

    # Clean up temporary files
    if temp_resized_video and os.path.exists(temp_resized_video):
        try:
            os.remove(temp_resized_video)
            os.rmdir(os.path.dirname(temp_resized_video))
        except Exception:
            pass

    # Save configuration
    try:
        config_copy_path = output_dir / "configuration_used.toml"
        save_config_to_toml(face_config, str(config_copy_path))
        print(f"Configuration saved: {config_copy_path}")
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")

    # Create log
    end_time = time.time()
    execution_time = end_time - start_time

    log_info_path = output_dir / "log_info.txt"
    with open(log_info_path, "w") as log_file:
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write(f"Resolution: {original_width}x{original_height}\n")
        log_file.write(f"FPS: {safe_fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")
        log_file.write(f"FaceMesh Configuration: {face_config}\n")
        if frames_with_missing_data:
            log_file.write(f"Frames with missing data: {len(frames_with_missing_data)}\n")
        else:
            log_file.write("No frames with missing data.\n")

    print(f"\nCompleted processing {video_path.name}")
    print(f"Output saved to: {output_dir}")
    print(f"Processing time: {execution_time:.2f} seconds\n")


# Process videos in directory
def process_videos_in_directory(existing_root=None, use_gpu=False, gpu_type="nvidia"):
    """
    Process all video files in the selected directory for face mesh analysis.
    """
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

    if existing_root is not None:
        root = existing_root
    else:
        root = tk.Tk()
        root.withdraw()
        from contextlib import suppress

        with suppress(Exception):
            root.attributes("-topmost", True)

    def prepare_root_for_dialog():
        if platform.system() == "Darwin":
            root.deiconify()
            root.update_idletasks()
            root.geometry("1x1+100+100")
            root.lift()
            root.update_idletasks()

    prepare_root_for_dialog()
    input_dir = filedialog.askdirectory(
        parent=root, title="Select the input directory containing videos"
    )
    if platform.system() == "Darwin" and existing_root is None:
        root.withdraw()
    if not input_dir:
        messagebox.showerror("Error", "No input directory selected.")
        return

    prepare_root_for_dialog()
    output_base = filedialog.askdirectory(parent=root, title="Select the base output directory")
    if platform.system() == "Darwin" and existing_root is None:
        root.withdraw()
    if not output_base:
        messagebox.showerror("Error", "No output directory selected.")
        return

    # Face configuration (GUI or TOML via dialog)
    face_config = get_face_config(root, input_dir=input_dir)
    if not face_config:
        return

    # Timestamped output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_parts = []
    if face_config.get("enable_resize", False):
        suffix_parts.append(f"resize_{face_config.get('resize_scale', 2)}x")
    if face_config.get("enable_advanced_filtering", False):
        interp_method = face_config.get("interp_method", "none")
        smooth_method = face_config.get("smooth_method", "none")
        suffix_parts.append(f"filter_{interp_method}_{smooth_method}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    output_base = Path(output_base) / f"facemesh{suffix}_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    # Gather video files
    input_dir = Path(input_dir)
    video_files = [f for f in input_dir.glob("*.*") if f.suffix.lower() in [".mp4", ".avi", ".mov"]]

    if not video_files:
        messagebox.showerror(
            "Error", "No video files (.mp4, .avi, .mov) found in the selected folder."
        )
        return

    print(f"\nFound {len(video_files)} videos to process")
    if face_config.get("enable_resize", False):
        print(f"Video resize enabled: {face_config.get('resize_scale', 2)}x scaling")

    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file.name}")
        output_dir = output_base / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            process_video(video_file, output_dir, face_config, use_gpu=use_gpu, gpu_type=gpu_type)
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
        finally:
            try:
                import gc as _gc

                _gc.collect()
            except Exception:
                pass
            time.sleep(2)
            print("Memory released")

    print("\nAll videos processed!")


# Main entry point
if __name__ == "__main__":
    # Detect GPUs
    print("=" * 60)
    print("GPU DETECTION")
    print("=" * 60)

    nvidia_available, nvidia_info, nvidia_error = detect_nvidia_gpu()
    if nvidia_available:
        print(f"✓ NVIDIA GPU detected: {nvidia_info.get('name', 'Unknown')}")
        print(f"  Driver: {nvidia_info.get('driver_version', 'Unknown')}")
        if nvidia_info.get("memory_total_mb", 0) > 0:
            print(f"  Memory: {nvidia_info['memory_total_mb'] / 1024:.1f} GB")
    else:
        print(f"✗ NVIDIA GPU not available: {nvidia_error}")

    amd_available, amd_info, amd_error = detect_amd_gpu()
    if amd_available:
        print(f"✓ AMD GPU detected: {amd_info.get('name', 'Unknown')}")
    else:
        print(f"✗ AMD GPU not available: {amd_error}")

    print("\nNote: MediaPipe FaceMesh currently uses CPU processing.")
    print("GPU acceleration may be available in future MediaPipe versions.")
    print("=" * 60 + "\n")

    # Use existing root or create new one
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass

    # Show device selection dialog
    device_dialog = DeviceSelectionDialog(
        root, nvidia_available, nvidia_info, amd_available, amd_info
    )
    selected_device = device_dialog.result if device_dialog.result else "cpu"

    use_gpu = selected_device in ["nvidia", "amd"]
    gpu_type = selected_device if use_gpu else "cpu"

    if use_gpu:
        print(f"✓ {gpu_type.upper()} GPU processing selected")
        print("Note: FaceMesh will use CPU (GPU delegate not yet available)")
    else:
        print("✓ CPU processing selected")

    print()

    # Process videos
    process_videos_in_directory(root, use_gpu=use_gpu, gpu_type=gpu_type)
