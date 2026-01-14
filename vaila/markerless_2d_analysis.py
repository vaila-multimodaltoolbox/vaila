"""
Project: vailá Multimodal Toolbox
Script: markerless_2D_analysis.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 14 January 2026
Version: 0.8.1

Example of usage:
First activate the vaila environment:
conda activate vaila
Then run the markerless_2d_analysis.py script:
python markerless_2d_analysis.py -i input_directory -o output_directory -c config.toml

Description:
This script performs batch processing of videos for 2D pose estimation using
MediaPipe's Pose model (Tasks API 0.10.31+). It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files. The script also generates a
video with the landmarks overlaid on the original frames.

The script supports multiple processing backends:
- CPU: Standard processing (always available)
- NVIDIA/CUDA: GPU acceleration for NVIDIA GPUs (requires CUDA drivers)
- ROCm: GPU acceleration for AMD GPUs (requires ROCm installation)
- MPS: GPU acceleration for Apple Silicon (macOS arm64)

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, model complexity, and
whether to enable segmentation and smooth segmentation. The default settings
prioritize the highest detection accuracy and tracking precision, which may
increase computational cost.

New Features (v0.8.1):
- Unified CPU and GPU processing in single script
- Multi-GPU backend support (NVIDIA/CUDA, ROCm/AMD, MPS/Apple Silicon)
- Automatic GPU detection and testing
- Device selection dialog for choosing processing backend
- Bounding box (ROI) selection for small subjects or multi-person scenarios with zoom and window resize capabilities

Usage:
- Run the script to automatically detect available GPU backends (NVIDIA, ROCm, MPS)
- Select processing device (CPU or GPU) via device selection dialog
- Open a graphical interface for selecting the input directory containing video files
  (.mp4, .avi, .mov), the output directory, and for specifying the MediaPipe
  configuration parameters.
- Choose whether to enable video resize for better pose detection
- Optionally select a bounding box (ROI) from the first video frame for focused processing
- Optionally resize the cropped region for better detection of small subjects
- Load or save TOML configuration files for batch processing
- The script processes each video with the selected device, generating an output video
  with overlaid pose landmarks, and CSV files containing both normalized and pixel-based
  landmark coordinates in original video dimensions.

Requirements:
- Python 3.12.12
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Tkinter (usually included with Python installations)
- Pillow (if using image manipulation: `pip install Pillow`)
- Pandas (for coordinate conversion: `pip install pandas`)
- psutil (pip install psutil) - for memory monitoring

GPU Acceleration (Optional):
- NVIDIA/CUDA: NVIDIA GPU with CUDA Toolkit and drivers installed
- ROCm: AMD GPU with ROCm installed (Linux)
- MPS: Apple Silicon (arm64) on macOS (automatic)

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

import datetime
import gc
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import time as _time_module
import tkinter as tk
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import psutil

# #region agent log
# Debug logging - uses script directory for portability across OS (Linux, macOS, Windows)
# Creates .cursor directory in project root if it doesn't exist
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

# --- NEW IMPORTS FOR THE TASKS API (MediaPipe 0.10.31+) ---

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

# #region agent log
_debug_log(
    "A",
    "import:100",
    "After importing mediapipe Tasks API",
    {
        "mp_type": str(type(mp)),
        "has_tasks": hasattr(mp, "tasks"),
        "mp_attrs": str([x for x in dir(mp) if not x.startswith("_")])[:500],
    },
)
# #endregion

# Additional imports for filtering and interpolation
from pykalman import KalmanFilter  # noqa: E402
#from rich import print  # noqa: E402
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
# These are used conditionally for Linux-specific features
import multiprocessing  # noqa: F401, E402 - For future Linux batch processing
import signal  # noqa: F401, E402 - For future Linux process management
import threading  # noqa: F401, E402 - For future Linux thread management

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

VERBOSE_FRAMES = False  # Set to True to print per-frame pose detection status
PAD_START_FRAMES = 120  # Number of initial frames to pad for MediaPipe stabilization

# Add new defaults
PAD_START_FRAMES_DEFAULT = 30
ENABLE_PADDING_DEFAULT = True

# CPU throttling settings for high-resolution videos
CPU_USAGE_THRESHOLD = 150  # Percentage (across all cores)
FRAME_SLEEP_TIME = 0.01  # Sleep between frames when CPU is high
MAX_CPU_CHECK_INTERVAL = 100  # Check CPU every N frames


"""
Module: filter_utils.py
Description: This module provides a unified and flexible Butterworth filter function for low-pass and band-pass filtering of signals. The function supports edge effect mitigation through optional signal padding and uses second-order sections (SOS) for improved numerical stability.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.1
Date: 2024-09-12

Changelog:
- Version 1.1 (2024-09-12):
  - Modified `butter_filter` to handle multidimensional data.
  - Adjusted padding length dynamically based on data length.
  - Fixed issues causing errors when data length is less than padding length.

Usage Example:
- Low-pass filter:
  `filtered_data_low = butter_filter(data, fs=1000, filter_type='low', cutoff=10, order=4)`

- Band-pass filter:
  `filtered_data_band = butter_filter(data, fs=1000, filter_type='band', lowcut=5, highcut=15, order=4)`
"""


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
    """
    Applies a Butterworth filter (low-pass or band-pass) to the input data.

    Parameters:
    - data: array-like
        The input signal to be filtered. Can be 1D or multidimensional. Filtering is applied along the first axis.
    - fs: float
        The sampling frequency of the signal.
    - filter_type: str, default='low'
        The type of filter to apply: 'low' for low-pass or 'band' for band-pass.
    - cutoff: float, optional
        The cutoff frequency for a low-pass filter.
    - lowcut: float, optional
        The lower cutoff frequency for a band-pass filter.
    - highcut: float, optional
        The upper cutoff frequency for a band-pass filter.
    - order: int, default=4
        The order of the Butterworth filter.
    - padding: bool, default=True
        Whether to pad the signal to mitigate edge effects.

    Returns:
    - filtered_data: array-like
        The filtered signal.
    """
    # Check filter type and set parameters
    nyq = 0.5 * fs  # Nyquist frequency
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
    axis = 0  # Filtering along the first axis (rows)

    # Apply padding if needed to handle edge effects
    if padding:
        data_len = data.shape[axis]
        # Ensure padding length is suitable for data length
        max_padlen = data_len - 1
        padlen = min(int(fs), max_padlen, 15)

        if data_len <= padlen:
            raise ValueError(
                f"The length of the input data ({data_len}) must be greater than the padding length ({padlen})."
            )

        # Pad the data along the specified axis
        pad_width = [(0, 0)] * data.ndim
        pad_width[axis] = (padlen, padlen)
        padded_data = np.pad(data, pad_width=pad_width, mode="reflect")
        filtered_padded_data = sosfiltfilt(sos, padded_data, axis=axis, padlen=0)
        # Remove padding
        idx = [slice(None)] * data.ndim
        idx[axis] = slice(padlen, -padlen)
        filtered_data = filtered_padded_data[tuple(idx)]
    else:
        filtered_data = sosfiltfilt(sos, data, axis=axis, padlen=0)

    return filtered_data


# Smoothing and filtering functions
def savgol_smooth(data, window_length, polyorder):
    """Apply Savitzky-Golay filter to the data."""
    data = np.asarray(data)
    return savgol_filter(data, window_length, polyorder, axis=0)


def lowess_smooth(data, frac, it):
    """Apply LOWESS smoothing to the data."""
    data = np.asarray(data)
    x = np.arange(len(data)) if data.ndim == 1 else np.arange(data.shape[0])

    try:
        # Apply padding for better edge handling
        pad_len = int(len(data) * 0.1)  # 10% padding
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

    # Apply padding for better edge handling
    pad_len = int(len(data) * 0.1)  # 10% padding
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
    alpha = 0.7  # Blending factor for smoothing
    data = np.asarray(data)

    # Handle 1D data
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_features = data.shape[1]

    try:
        if mode == 1:  # 1D mode
            # Process each column independently
            filtered_data = np.empty_like(data)
            for j in range(n_features):
                # Initialize Kalman filter for 1D state (position and velocity)
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

                # Apply EM algorithm and smoothing
                smoothed_state_means, _ = kf.em(data[:, j : j + 1], n_iter=n_iter).smooth(
                    data[:, j : j + 1]
                )
                filtered_data[:, j] = alpha * smoothed_state_means[:, 0] + (1 - alpha) * data[:, j]

        else:  # mode == 2
            # Process x,y pairs together
            if n_features % 2 != 0:
                raise ValueError("For 2D mode, number of features must be even (x,y pairs)")

            filtered_data = np.empty_like(data)
            for j in range(0, n_features, 2):
                # Initialize Kalman filter for 2D state
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


def filter_pose_data(data, enable_median=True, enable_butterworth=True, order=4, fc=6, fs=30, median_kernel=5):
    """
    Applies:
    1. Median Filter (removes abrupt jumps/outliers).
    2. Butterworth Filter (smooths movement).
    """
    # Safe import
    try:
        from scipy import signal
    except ImportError:
        print("Scipy not installed. Filtering ignored.")
        return data

    filtered_data = data.copy()
    
    # Iterate over columns (handling numpy array)
    n_cols = data.shape[1]
    for j in range(n_cols):
        # Check if column is numeric (trivial for numpy float array, but good practice)
        if not np.issubdtype(filtered_data[:, j].dtype, np.number):
            continue

        # 1. REMOVE SPIKES (MEDIAN)
        # If median_kernel > 1, apply filter. Value must be odd.
        if enable_median and median_kernel and median_kernel > 1:
            k = int(median_kernel)
            if k % 2 == 0: k += 1 # Ensure odd
            try:
                filtered_data[:, j] = signal.medfilt(filtered_data[:, j], kernel_size=k)
            except Exception as e:
                pass # Ignore error if array is too small

        # 2. SMOOTHING (BUTTERWORTH)
        if enable_butterworth:
            nyquist = 0.5 * fs
            normal_cutoff = fc / nyquist
            if normal_cutoff >= 1: normal_cutoff = 0.99
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            try:
                filtered_data[:, j] = signal.filtfilt(b, a, filtered_data[:, j])
            except Exception as e:
                pass # Keep data with only median if butter fails
            
    return filtered_data

def arima_smooth(data, order=(1, 0, 0)):
    """Apply ARIMA smoothing to the input data."""
    data = np.asarray(data)

    # If data is 1D, process directly
    if data.ndim == 1:
        try:
            # Remove NaN values for ARIMA fitting
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data  # Return original if all NaN

            valid_data = data[valid_mask]
            if len(valid_data) < max(order) + 1:
                print("Warning: Not enough data points for ARIMA model")
                return data

            model = ARIMA(valid_data, order=order)
            result = model.fit(disp=False)  # Suppress output

            # Create output array
            output = data.copy()
            output[valid_mask] = result.fittedvalues
            return output

        except Exception as e:
            print(f"Error in ARIMA smoothing: {str(e)}")
            return data  # Return original data if smoothing fails
    else:
        # For 2D data, apply ARIMA smoothing column by column
        smoothed = np.empty_like(data)
        for j in range(data.shape[1]):
            try:
                col_data = data[:, j]
                valid_mask = ~np.isnan(col_data)

                if not np.any(valid_mask):
                    smoothed[:, j] = col_data  # Keep original if all NaN
                    continue

                valid_data = col_data[valid_mask]
                if len(valid_data) < max(order) + 1:
                    print(f"Warning: Not enough data points for ARIMA model in column {j}")
                    smoothed[:, j] = col_data
                    continue

                model = ARIMA(valid_data, order=order)
                result = model.fit(disp=False)  # Suppress output

                smoothed[:, j] = col_data.copy()
                smoothed[valid_mask, j] = result.fittedvalues

            except Exception as e:
                print(f"Error in ARIMA smoothing for column {j}: {str(e)}")
                smoothed[:, j] = data[:, j]  # Keep original data for failed columns
        return smoothed

# Configuration management functions
def get_default_config():
    """Get default configuration dictionary"""
    return {
        "mediapipe": {
            "min_detection_confidence": 0.1,
            "min_tracking_confidence": 0.1,
            "model_complexity": 2,
            "enable_segmentation": False,
            "smooth_segmentation": False,
            "static_image_mode": False,
            "apply_filtering": True,
            "estimate_occluded": True,
            # Simple Post-Processing
            "enable_median_filter": False,
            "median_kernel_size": 5,
        },
        "video_resize": {"enable_resize": False, "resize_scale": 2},
        # Legacy/Advanced sections removed for simplicity
        "bounding_box": {
            "enable_crop": False,
            "bbox_x_min": 0,
            "bbox_y_min": 0,
            "bbox_x_max": 1920,
            "bbox_y_max": 1080,
            "enable_resize_crop": False,
            "resize_crop_scale": 2,
            "roi_polygon_points": None,
        },
        "enable_padding": ENABLE_PADDING_DEFAULT,
        "pad_start_frames": PAD_START_FRAMES_DEFAULT,
        "enable_reverse_padding": True,
        "pad_end_frames": 30,
        "bounding_box_ranges": [],
    }


def save_config_to_toml(config, filepath):
    """Save configuration to TOML file"""
    try:
        # Convert config to TOML-friendly format
        toml_config = {
            "mediapipe": {
                "min_detection_confidence": config.get("min_detection_confidence", 0.1),
                "min_tracking_confidence": config.get("min_tracking_confidence", 0.1),
                "model_complexity": config.get("model_complexity", 2),
                "enable_segmentation": config.get("enable_segmentation", False),
                "smooth_segmentation": config.get("smooth_segmentation", False),
                "static_image_mode": config.get("static_image_mode", False),
                "apply_filtering": config.get("apply_filtering", True),
                "estimate_occluded": config.get("estimate_occluded", True),
                # New Post-Processing
                "enable_median_filter": config.get("enable_median_filter", False),
                "median_kernel_size": config.get("median_kernel_size", 5),
            },
            "video_resize": {
                "enable_resize": config.get("enable_resize", False),
                "resize_scale": config.get("resize_scale", 2),
            },
            # Sections advanced_filtering, smoothing_params, scientific_robustness REMOVED
            "enable_padding": str(config.get("enable_padding", ENABLE_PADDING_DEFAULT)).lower(),
            "pad_start_frames": config.get("pad_start_frames", PAD_START_FRAMES_DEFAULT),
            "bounding_box": {
                "enable_crop": config.get("enable_crop", False),
                "bbox_x_min": config.get("bbox_x_min", 0),
                "bbox_y_min": config.get("bbox_y_min", 0),
                "bbox_x_max": config.get("bbox_x_max", 1920),
                "bbox_y_max": config.get("bbox_y_max", 1080),
                "enable_resize_crop": config.get("enable_resize_crop", False),
                "resize_crop_scale": config.get("resize_crop_scale", 2),
                "roi_polygon_points": config.get("roi_polygon_points"),
            },
            "enable_reverse_padding": config.get("enable_reverse_padding", False),
            "pad_end_frames": config.get("pad_end_frames", 0),
            "bounding_box_ranges": config.get("bounding_box_ranges", []),
        }

        with open(filepath, "w") as f:
            # Write header comment
            f.write("# ================================================================\n")
            f.write("# MediaPipe 2D Analysis Configuration File\n")
            f.write(
                "# Generated automatically by markerless_2D_analysis.py in vaila Multimodal Analysis Toolbox\n"
            )
            f.write(f"# Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# ================================================================\n")
            f.write("#\n")
            f.write("# HOW TO USE THIS FILE:\n")
            f.write("# 1. Edit the values below to customize your analysis\n")
            f.write("# 2. Save this file\n")
            f.write("# 3. In the script, click 'Load Existing TOML Configuration'\n")
            f.write("# 4. Select this file and run your analysis\n")
            f.write("#\n")
            f.write("# IMPORTANT: Keep the format exactly as shown!\n")
            f.write("# - true/false must be lowercase\n")
            f.write("# - Numbers can have decimals (3.0) or not (30)\n")
            f.write('# - Text must be in quotes ("linear")\n')
            f.write("# ================================================================\n\n")

            # Write sections with comments
            # Write video info if available
            if "video_total_frames" in config:
                f.write("# ================================================================\n")
                f.write("# VIDEO INFO\n")
                f.write("# ================================================================\n")
                f.write(f"video_total_frames = {config['video_total_frames']}  # Total frames in the processed video\n\n")

            f.write("[mediapipe]\n")
            f.write("# ================================================================\n")
            f.write("# MEDIAPIPE POSE DETECTION SETTINGS\n")
            f.write("# ================================================================\n")
            f.write("# These control how MediaPipe detects poses in your video\n")

            mp = toml_config["mediapipe"]
            f.write(
                f"min_detection_confidence = {mp['min_detection_confidence']}    # How confident to start detecting (0.1-1.0)\n"
            )
            f.write(
                "#                                        # Lower = detect more poses (try 0.1-0.3)\n"
            )
            f.write(
                "#                                        # Higher = only clear poses (try 0.7-0.9)\n"
            )

            f.write(
                f"min_tracking_confidence = {mp['min_tracking_confidence']}     # How confident to keep tracking (0.1-1.0)\n"
            )
            f.write(
                "#                                       # Lower = track longer (try 0.1-0.3)\n"
            )
            f.write(
                "#                                       # Higher = drop unclear tracking (try 0.7-0.9)\n"
            )

            f.write(
                f"model_complexity = {mp['model_complexity']}                  # Model accuracy vs speed\n"
            )
            f.write("#                        # 0 = fastest, least accurate\n")
            f.write("#                        # 1 = balanced speed and accuracy\n")
            f.write("#                        # 2 = slowest, most accurate (recommended)\n")

            f.write(
                f"enable_segmentation = {str(mp['enable_segmentation']).lower()}           # Draw person outline (true/false)\n"
            )
            f.write("#                             # true = creates person mask (slower)\n")
            f.write("#                             # false = only landmarks (faster)\n")

            f.write(
                f"smooth_segmentation = {str(mp['smooth_segmentation']).lower()}           # Smooth the outline (true/false)\n"
            )
            f.write("#                             # Only works if enable_segmentation = true\n")

            f.write(
                f"static_image_mode = {str(mp['static_image_mode']).lower()}             # Treat each frame separately (true/false)\n"
            )
            f.write("#                           # false = track across frames (recommended)\n")
            f.write("#                           # true = detect fresh each frame (slower)\n")

            f.write(
                f"apply_filtering = {str(mp['apply_filtering']).lower()}               # Apply built-in smoothing (true/false)\n"
            )
            f.write("#                         # true = smoother movement (recommended)\n")
            f.write("#                         # false = raw detection results\n")

            f.write(
                f"estimate_occluded = {str(mp['estimate_occluded']).lower()}             # Guess hidden body parts (true/false)\n"
            )
            f.write(
                "#                           # true = fill in missing landmarks (recommended)\n"
            )
            f.write("#                           # false = leave gaps when parts are hidden\n\n")

            f.write("# ================================================================\n")
            f.write("# POST-PROCESSING (Optional)\n")
            f.write("# ================================================================\n")
            f.write(
                f"enable_median_filter = {str(mp['enable_median_filter']).lower()}          # Apply simple median smoothing (true/false)\n"
            )
            f.write(
                f"median_kernel_size = {mp['median_kernel_size']}               # Kernel size (odd integer: 3, 5, 7)\n"
            )
            f.write("#                           # Helps reduce jitter without heavy distortion.\n\n")

            f.write("\n[video_resize]\n")
            f.write("# ================================================================\n")
            f.write("# VIDEO RESIZING FOR BETTER DETECTION\n")
            f.write("# ================================================================\n")
            f.write("# Resize video before analysis to improve pose detection\n")
            f.write("# Useful for: small people, distant subjects, low resolution videos\n")

            vr = toml_config["video_resize"]
            f.write(
                f"enable_resize = {str(vr['enable_resize']).lower()}                 # Resize video before analysis (true/false)\n"
            )
            f.write("#                           # true = upscale for better detection\n")
            f.write("#                           # false = use original size (faster)\n")

            f.write(
                f"resize_scale = {vr['resize_scale']}                      # Scale factor (2-8)\n"
            )
            f.write("#                        # 2 = double size (good for most cases)\n")
            f.write("#                        # 3-4 = better for very small subjects\n")
            f.write("#                        # 5-8 = for very distant or tiny people\n")
            f.write("#                        # Higher = better detection but much slower\n")
            f.write("#                        # Coordinates are automatically converted back\n")


            f.write("\n[padding]\n")
            f.write("# ================================================================\n")
            f.write("# INITIAL FRAME PADDING FOR STABILIZATION\n")
            f.write("# ================================================================\n")
            f.write("# Add repeated frames at the start to help MediaPipe stabilize.\n")
            f.write(
                f"enable_padding = {str(config.get('enable_padding', ENABLE_PADDING_DEFAULT)).lower()}  # true/false\n"
            )
            f.write(
                f"pad_start_frames = {config.get('pad_start_frames', PAD_START_FRAMES_DEFAULT)}  # Number of frames to pad at start\n"
            )
            f.write(
                f"enable_reverse_padding = {str(config.get('enable_reverse_padding', False)).lower()}  # Reverse padding at end (true/false)\n"
            )
            f.write(
                f"pad_end_frames = {config.get('pad_end_frames', 0)}  # Number of frames to pad at end\n"
            )
            f.write("# Recommended: 30-60 for most videos.\n\n")

            f.write("[bounding_box]\n")
            f.write("# ================================================================\n")
            f.write("# BOUNDING BOX (ROI) SELECTION FOR SMALL SUBJECTS\n")
            f.write("# ================================================================\n")
            f.write("# Crop frames to a region of interest before MediaPipe processing.\n")
            f.write("# Useful when: multiple people in frame, subject is very small,\n")
            f.write("# or you want to focus on a specific area.\n\n")

            bb = toml_config["bounding_box"]
            f.write(
                f"enable_crop = {str(bb['enable_crop']).lower()}              # Enable bounding box cropping (true/false)\n"
            )
            f.write("#                         # true = crop frames to ROI before processing\n")
            f.write("#                         # false = process full frame (default)\n\n")

            f.write(
                f"bbox_x_min = {bb['bbox_x_min']}                   # Left edge of ROI in pixels\n"
            )
            f.write(
                f"bbox_y_min = {bb['bbox_y_min']}                   # Top edge of ROI in pixels\n"
            )
            f.write(
                f"bbox_x_max = {bb['bbox_x_max']}                  # Right edge of ROI in pixels\n"
            )
            f.write(
                f"bbox_y_max = {bb['bbox_y_max']}                  # Bottom edge of ROI in pixels\n"
            )
            f.write("# Note: Coordinates are in pixels. Use 'Select ROI from Video'\n")
            f.write("# button in GUI to visually select the region from first video frame.\n\n")

            f.write(
                f"enable_resize_crop = {str(bb['enable_resize_crop']).lower()}       # Resize the cropped region (true/false)\n"
            )
            f.write(
                "#                         # true = upscale cropped region for better detection\n"
            )
            f.write("#                         # false = use cropped region at original size\n\n")

            f.write(
                f"resize_crop_scale = {bb['resize_crop_scale']}            # Scale factor for cropped region (2-8)\n"
            )
            f.write("#                         # 2 = double size (good for most cases)\n")
            f.write("#                         # 3-4 = better for very small subjects\n")
            f.write("#                         # 5-8 = for very distant or tiny people\n")
            f.write("# This is separate from video_resize. Only the cropped region is resized,\n")
            f.write("# which is more efficient than resizing the entire video.\n\n")
            f.write("# Polygon ROI (optional):\n")
            if bb.get("roi_polygon_points"):
                # Save polygon points as TOML array
                polygon_str = "[\n"
                for pt in bb["roi_polygon_points"]:
                    polygon_str += f"    [{pt[0]}, {pt[1]}],\n"
                polygon_str = polygon_str.rstrip(",\n") + "\n]"
                f.write(f"roi_polygon_points = {polygon_str}  # Polygon ROI points\n")
            else:
                f.write("# roi_polygon_points = [[x1, y1], [x2, y2], ...]  # Polygon ROI points\n")
            f.write("# Note: If roi_polygon_points is set, it takes precedence over bbox coordinates.\n")
            f.write("# Use 'Select Polygon ROI' button in GUI to visually select polygon from video.\n\n")

            # NEW: Save multiple ROI ranges for different frame intervals
            bbox_ranges = config.get("bounding_box_ranges", [])
            if bbox_ranges and len(bbox_ranges) > 0:
                f.write("[bounding_box_ranges]\n")
                f.write("# ================================================================\n")
                f.write("# MULTIPLE ROI RANGES FOR DIFFERENT FRAME INTERVALS (ADVANCED)\n")
                f.write("# ================================================================\n")
                f.write("# Define different ROI and resize crop settings for different frame ranges.\n")
                f.write("# This is useful when the subject size changes significantly during the video.\n")
                f.write("# Example: Small subject at start (frames 0-100) needs larger ROI and 4x resize,\n")
                f.write("#          but larger subject later (frames 101+) needs smaller ROI and 2x resize.\n\n")

                for idx, range_config in enumerate(bbox_ranges):
                    f.write(f"[[bounding_box_ranges.range]]  # Range {idx + 1}\n")
                    f.write(f"frame_start = {range_config.get('frame_start', 0)}  # Start frame (0-based)\n")
                    # Handle infinity value for TOML (use quoted string)
                    frame_end_val = range_config.get('frame_end', float('inf'))
                    if frame_end_val == float('inf'):
                        f.write('frame_end = "inf"  # End frame (use "inf" for last frame)\n')
                    else:
                        f.write(f"frame_end = {int(frame_end_val)}  # End frame\n")
                    roi_type = range_config.get('roi_type', 'inclusion')
                    f.write(f'roi_type = "{roi_type}"  # "inclusion" = detect inside ROI, "exclusion" = ignore this area\n')
                    f.write(f"enable_crop = {str(range_config.get('enable_crop', False)).lower()}  # Enable ROI for this range\n")
                    f.write(f"bbox_x_min = {range_config.get('bbox_x_min', 0)}  # Left edge of ROI\n")
                    f.write(f"bbox_y_min = {range_config.get('bbox_y_min', 0)}  # Top edge of ROI\n")
                    f.write(f"bbox_x_max = {range_config.get('bbox_x_max', 1920)}  # Right edge of ROI\n")
                    f.write(f"bbox_y_max = {range_config.get('bbox_y_max', 1080)}  # Bottom edge of ROI\n")
                    f.write(f"enable_resize_crop = {str(range_config.get('enable_resize_crop', False)).lower()}  # Resize cropped region\n")
                    f.write(f"resize_crop_scale = {range_config.get('resize_crop_scale', 2)}  # Scale factor (2-8)\n")
                    if range_config.get("roi_polygon_points"):
                        polygon_str = "[\n"
                        for pt in range_config["roi_polygon_points"]:
                            polygon_str += f"    [{pt[0]}, {pt[1]}],\n"
                        polygon_str = polygon_str.rstrip(",\n") + "\n]"
                        f.write(f"roi_polygon_points = {polygon_str}  # Polygon ROI points (optional)\n")
                    f.write("\n")
            else:
                f.write("# [bounding_box_ranges]  # Uncomment to use multiple ROI ranges\n")
                f.write("# [[bounding_box_ranges.range]]\n")
                f.write("# frame_start = 0\n")
                f.write("# frame_end = 100\n")
                f.write("# enable_crop = true\n")
                f.write("# bbox_x_min = 100\n")
                f.write("# bbox_y_min = 100\n")
                f.write("# bbox_x_max = 500\n")
                f.write("# bbox_y_max = 400\n")
                f.write("# enable_resize_crop = true\n")
                f.write("# resize_crop_scale = 4\n")
                f.write("#\n")
                f.write("# [[bounding_box_ranges.range]]\n")
                f.write("# frame_start = 101\n")
                f.write('# frame_end = "inf"  # Use "inf" for last frame\n')
                f.write("# enable_crop = true\n")
                f.write("# bbox_x_min = 150\n")
                f.write("# bbox_y_min = 150\n")
                f.write("# bbox_x_max = 600\n")
                f.write("# bbox_y_max = 500\n")
                f.write("# enable_resize_crop = true\n")
                f.write("# resize_crop_scale = 2\n\n")

            # NEW: Scientific Robustness Section
            sr = toml_config["scientific_robustness"]
            f.write("\n[scientific_robustness]\n")
            f.write("# ================================================================\n")
            f.write("# SCIENTIFIC ROBUSTNESS PARAMETERS (Kalman + Optical Flow) - v0.9.0\n")
            f.write("# ================================================================\n")
            f.write("# Kalman Filter: Predicts motion when tracking is lost.\n")
            f.write(f"track_process_noise = {sr.get('track_process_noise', 1.0)}       # Process noise (model uncertainty)\n")
            f.write(f"track_measurement_noise = {sr.get('track_measurement_noise', 0.1)}   # Base measurement noise (sensor uncertainty)\n")
            f.write(f"max_pred_gap = {sr.get('max_pred_gap', 10)}              # Max frames to predict before declaring a gap\n")
            f.write("# Optical Flow: Tracks points when MediaPipe confidence is low.\n")
            f.write(f"optical_flow_threshold = {sr.get('optical_flow_threshold', 0.15)}    # Max error for optical flow tracking\n")
            f.write("# Gap Reconstruction: Fills true gaps (offline).\n")
            f.write(f"enable_gap_reconstruction = {str(sr.get('enable_gap_reconstruction', True)).lower()} # Enable offline gap reconstruction\n")
            f.write(f"max_reconstruction_gap = {sr.get('max_reconstruction_gap', 10)}      # Max gap size to fill (frames)\n")
            f.write("\n")

        print(f"Configuration saved to: {filepath}")
        return True
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False


def load_config_from_toml(filepath):
    """Load configuration from TOML file with detailed error handling"""
    try:
        if not os.path.exists(filepath):
            print(f"Configuration file not found: {filepath}")
            return None

        print(f"Reading TOML file: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            toml_config = toml.load(f)

        print(f"TOML parsed successfully. Found sections: {list(toml_config.keys())}")

        # Convert TOML config back to internal format
        config = {}

        # MediaPipe settings
        if "mediapipe" in toml_config:
            mp = toml_config["mediapipe"]
            print(f"Loading MediaPipe settings: {list(mp.keys())}")
            config.update(
                {
                    "min_detection_confidence": float(mp.get("min_detection_confidence", 0.1)),
                    "min_tracking_confidence": float(mp.get("min_tracking_confidence", 0.1)),
                    "model_complexity": int(mp.get("model_complexity", 2)),
                    "enable_segmentation": bool(mp.get("enable_segmentation", False)),
                    "smooth_segmentation": bool(mp.get("smooth_segmentation", False)),
                    "static_image_mode": bool(mp.get("static_image_mode", False)),
                    "apply_filtering": bool(mp.get("apply_filtering", True)),
                    "estimate_occluded": bool(mp.get("estimate_occluded", True)),
                    # New Post-Processing
                    "enable_median_filter": bool(mp.get("enable_median_filter", False)),
                    "median_kernel_size": int(mp.get("median_kernel_size", 5)),
                }
            )
        else:
            print("Warning: No [mediapipe] section found, using defaults")
            # Set defaults including new median filter
            defaults = get_default_config()
            config.update(defaults["mediapipe"])

        # Video resize settings
        if "video_resize" in toml_config:
            vr = toml_config["video_resize"]
            print(f"Loading video resize settings: {list(vr.keys())}")
            config.update(
                {
                    "enable_resize": bool(vr.get("enable_resize", False)),
                    "resize_scale": int(vr.get("resize_scale", 2)),
                }
            )
        else:
            print("Warning: No [video_resize] section found, using defaults")

        # Legacy advanced_filtering and smoothing_params sections removed
        # We silently ignore them if present in old config files.

        # Padding section
        if "padding" in toml_config:
            pad = toml_config["padding"]
            config["enable_padding"] = bool(pad.get("enable_padding", ENABLE_PADDING_DEFAULT))
            config["pad_start_frames"] = int(pad.get("pad_start_frames", PAD_START_FRAMES_DEFAULT))
            config["enable_reverse_padding"] = bool(pad.get("enable_reverse_padding", False))
            config["pad_end_frames"] = int(pad.get("pad_end_frames", 0))
        else:
            config["enable_padding"] = ENABLE_PADDING_DEFAULT
            config["pad_start_frames"] = PAD_START_FRAMES_DEFAULT
            config["enable_reverse_padding"] = False
            config["pad_end_frames"] = 0

        # Bounding box section
        if "bounding_box" in toml_config:
            bb = toml_config["bounding_box"]
            print(f"Loading bounding box settings: {list(bb.keys())}")
            bbox_x_min = int(bb.get("bbox_x_min", 0))
            bbox_y_min = int(bb.get("bbox_y_min", 0))
            bbox_x_max = int(bb.get("bbox_x_max", 1920))
            bbox_y_max = int(bb.get("bbox_y_max", 1080))

            # Validate coordinates
            if bbox_x_max <= bbox_x_min or bbox_y_max <= bbox_y_min:
                print("Warning: Invalid bounding box coordinates, using defaults")
                bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = 0, 0, 1920, 1080

            config.update(
                {
                    "enable_crop": bool(bb.get("enable_crop", False)),
                    "bbox_x_min": bbox_x_min,
                    "bbox_y_min": bbox_y_min,
                    "bbox_x_max": bbox_x_max,
                    "bbox_y_max": bbox_y_max,
                    "enable_resize_crop": bool(bb.get("enable_resize_crop", False)),
                    "resize_crop_scale": int(bb.get("resize_crop_scale", 2)),
                    "roi_polygon_points": bb.get("roi_polygon_points"),  # Load polygon ROI if available
                }
            )
        else:
            print("Warning: No [bounding_box] section found, using defaults")
            config.update(
                {
                    "enable_crop": False,
                    "bbox_x_min": 0,
                    "bbox_y_min": 0,
                    "bbox_x_max": 1920,
                    "bbox_y_max": 1080,
                    "enable_resize_crop": False,
                    "resize_crop_scale": 2,
                }
            )

        # Reverse padding (from padding section)
        # This was moved into the main padding block above.
        # if "padding" in toml_config:
        #     pad = toml_config["padding"]
        #     config["enable_reverse_padding"] = bool(pad.get("enable_reverse_padding", False))
        #     config["pad_end_frames"] = int(pad.get("pad_end_frames", 0))
        # else:
        #     config["enable_reverse_padding"] = False
        #     config["pad_end_frames"] = 0

        # NEW: Load multiple ROI ranges for different frame intervals
        if "bounding_box_ranges" in toml_config:
            bbox_ranges_config = toml_config["bounding_box_ranges"]
            if "range" in bbox_ranges_config:
                ranges_list = bbox_ranges_config["range"]
                # Handle both single dict and list of dicts
                if isinstance(ranges_list, dict):
                    ranges_list = [ranges_list]

                config["bounding_box_ranges"] = []
                for range_config in ranges_list:
                    frame_end = range_config.get("frame_end", float('inf'))
                    # Handle 'inf' string in TOML
                    if isinstance(frame_end, str) and frame_end.lower() == "inf":
                        frame_end = float('inf')

                    range_dict = {
                        "frame_start": int(range_config.get("frame_start", 0)),
                        "frame_end": float(frame_end) if isinstance(frame_end, (int, float)) else float('inf'),
                        "roi_type": str(range_config.get("roi_type", "inclusion")),  # "inclusion" or "exclusion"
                        "enable_crop": bool(range_config.get("enable_crop", False)),
                        "bbox_x_min": int(range_config.get("bbox_x_min", 0)),
                        "bbox_y_min": int(range_config.get("bbox_y_min", 0)),
                        "bbox_x_max": int(range_config.get("bbox_x_max", 1920)),
                        "bbox_y_max": int(range_config.get("bbox_y_max", 1080)),
                        "enable_resize_crop": bool(range_config.get("enable_resize_crop", False)),
                        "resize_crop_scale": int(range_config.get("resize_crop_scale", 2)),
                        "roi_polygon_points": range_config.get("roi_polygon_points"),
                    }
                    config["bounding_box_ranges"].append(range_dict)
                    print(f"Loaded ROI range: frames {range_dict['frame_start']}-{range_dict['frame_end']}")
            else:
                config["bounding_box_ranges"] = []
        else:
            config["bounding_box_ranges"] = []

        print(f"Configuration loaded successfully from: {filepath}")
        print(f"Total parameters: {len(config)}")
        print(f"Advanced filtering: {config.get('enable_advanced_filtering', False)}")
        print(f"Smoothing method: {config.get('smooth_method', 'none')}")
        if config.get("bounding_box_ranges"):
            print(f"Multiple ROI ranges loaded: {len(config['bounding_box_ranges'])} ranges")

        return config

    except toml.TomlDecodeError as e:
        print(f"TOML syntax error in {filepath}: {str(e)}")
        return None
    except ValueError as e:
        print(f"Value conversion error in {filepath}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error loading {filepath}: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None


def get_config_filepath():
    """Get the path for the configuration file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "mediapipe_config.toml")


class ConfidenceInputDialog(tk.simpledialog.Dialog):
    def __init__(self, parent, input_dir=None):
        self.loaded_config = None
        self.use_toml = False
        self.toml_path = None
        self.input_dir = input_dir
        self.roi_polygon_points = None
        super().__init__(parent, title="vaila Toolbox Configuration")

    def body(self, master):
        # Configuração da janela principal
        self.geometry("850x600")  # Tamanho inicial mais razoável
        self.resizable(True, True)

        # Estilo para deixar mais moderno
        style = ttk.Style()
        style.theme_use('clam') # Tenta usar um tema mais limpo se disponível

        # --- Criação do Sistema de Abas (Notebook) ---
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Criar os frames para cada aba
        self.tab_mediapipe = ttk.Frame(self.notebook)
        self.tab_processing = ttk.Frame(self.notebook)
        self.tab_roi = ttk.Frame(self.notebook)
        self.tab_scientific = ttk.Frame(self.notebook) # NEW
        self.tab_filters = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_mediapipe, text=' MediaPipe & Model ')
        self.notebook.add(self.tab_processing, text=' Simple Processing ')
        self.notebook.add(self.tab_roi, text=' Advanced ROI ')
        self.notebook.add(self.tab_filters, text=' Save & Load Config ')

        # --- ABA 1: MEDIAPIPE ---
        self._build_mediapipe_tab()

        # --- ABA 2: PROCESSAMENTO (Resize & Padding) ---
        self._build_processing_tab()

        # --- ABA 3: ROI (Bounding Box) ---
        self._build_roi_tab()

        # --- ABA 4: SAVE & LOAD ---
        self._build_filters_tab()

        # Retorna o foco inicial
        return self.min_detection_entry

    def _create_entry_row(self, parent, label_text, attr_name, default_val, row):
        """Helper para criar linhas de label + entry padronizadas"""
        lbl = ttk.Label(parent, text=label_text)
        lbl.grid(row=row, column=0, sticky="w", padx=10, pady=5)
        
        entry = ttk.Entry(parent)
        entry.insert(0, str(default_val))
        entry.grid(row=row, column=1, sticky="ew", padx=10, pady=5)
        
        # Garante que a coluna do entry expanda
        parent.grid_columnconfigure(1, weight=1)
        
        setattr(self, attr_name, entry)
        return entry

    def _build_mediapipe_tab(self):
        # Frame 1: Detection
        frame = ttk.LabelFrame(self.tab_mediapipe, text="Detection Parameters", padding=15)
        frame.pack(fill="x", padx=10, pady=5)

        mp_params = [
            ("Detection Confidence (0.0 - 1.0):", "min_detection_entry", "0.1"),
            ("Tracking Confidence (0.0 - 1.0):", "min_tracking_entry", "0.1"),
            ("Model Complexity (0, 1, 2):", "model_complexity_entry", "2"),
            ("Enable Segmentation (True/False):", "enable_segmentation_entry", "False"),
            ("Smooth Segmentation (True/False):", "smooth_segmentation_entry", "False"),
            ("Static Image Mode (True/False):", "static_image_mode_entry", "False"),
            ("Apply Internal Filtering (True/False):", "apply_filtering_entry", "True"),
            ("Estimate Occluded (True/False):", "estimate_occluded_entry", "True"),
        ]

        for i, (txt, attr, val) in enumerate(mp_params):
            self._create_entry_row(frame, txt, attr, val, i)
            
        # Frame 2: Post-Processing (Simple)
        pp_frame = ttk.LabelFrame(self.tab_mediapipe, text="Simple Post-Processing", padding=15)
        pp_frame.pack(fill="x", padx=10, pady=5)
        
        # Enable Median Filter
        self.enable_median_filter_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(pp_frame, text="Enable Median Filter (Reduce Jitter)", variable=self.enable_median_filter_var).grid(row=0, column=0, columnspan=2, sticky="w")
        
        # Kernel Size
        self._create_entry_row(pp_frame, "Median Kernel Size (3, 5, 7):", "median_kernel_size_entry", "5", 1)

    def _build_processing_tab(self):
        # Frame de Resize
        resize_frame = ttk.LabelFrame(self.tab_processing, text="Video Resizing", padding=15)
        resize_frame.pack(fill="x", padx=10, pady=10)

        self._create_entry_row(resize_frame, "Enable Resize (True/False):", "enable_resize_entry", "False", 0)
        self._create_entry_row(resize_frame, "Resize Scale (e.g., 2):", "resize_scale_entry", "2", 1)

        # Frame de Padding
        pad_frame = ttk.LabelFrame(self.tab_processing, text="Padding (Initial Stabilization)", padding=15)
        pad_frame.pack(fill="x", padx=10, pady=10)

        pad_params = [
            ("Enable Initial Padding (True/False):", "enable_padding_entry", str(ENABLE_PADDING_DEFAULT)),
            ("Initial Padding Frames:", "pad_start_frames_entry", str(PAD_START_FRAMES_DEFAULT)),
            ("Enable Reverse Padding (True/False):", "enable_reverse_padding_entry", "True"),
            ("Reverse Padding Frames:", "pad_end_frames_entry", "30"),
        ]
        
        for i, (txt, attr, val) in enumerate(pad_params):
            self._create_entry_row(pad_frame, txt, attr, val, i)

    def _build_roi_tab(self):
        roi_frame = ttk.LabelFrame(self.tab_roi, text="Region of Interest (ROI) Selection", padding=15)
        roi_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Checkbox
        self.enable_crop_var = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(roi_frame, text="Enable Cropping", variable=self.enable_crop_var)
        cb.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # Coordenadas
        bbox_params = [
            ("Minimum X (px):", "bbox_x_min_entry", "0"),
            ("Minimum Y (px):", "bbox_y_min_entry", "0"),
            ("Maximum X (px):", "bbox_x_max_entry", "1920"),
            ("Maximum Y (px):", "bbox_y_max_entry", "1080"),
        ]
        for i, (txt, attr, val) in enumerate(bbox_params):
            entry = self._create_entry_row(roi_frame, txt, attr, val, i+1)
            # Bind para atualizar coords normalizadas
            entry.bind("<KeyRelease>", self.update_normalized_coords)

        # Label coords normalizadas
        ttk.Label(roi_frame, text="Normalized Coords:").grid(row=5, column=0, sticky="e", padx=10, pady=5)
        self.norm_coords_label = ttk.Label(roi_frame, text="(0.0, 0.0) - (1.0, 1.0)", foreground="gray")
        self.norm_coords_label.grid(row=5, column=1, sticky="w", padx=10, pady=5)

        # Botões de Seleção Visual
        btn_frame = ttk.Frame(roi_frame)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=15)
        
        ttk.Button(btn_frame, text="Select BBox from Video", command=self.select_roi_from_video).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Select Polygon (Free)", command=self.select_polygon_roi_from_video).pack(side="left", padx=5)

        # Opções de Resize do Crop
        sep = ttk.Separator(roi_frame, orient="horizontal")
        sep.grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

        self.enable_resize_crop_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(roi_frame, text="Resize Cropped Region", variable=self.enable_resize_crop_var).grid(row=8, column=0, sticky="w")
        
        self._create_entry_row(roi_frame, "Crop Scale (2-8):", "resize_crop_scale_entry", "2", 9)

        # Multi ROI
        multi_frame = ttk.LabelFrame(self.tab_roi, text="Multiple ROIs (Advanced)", padding=10)
        multi_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(multi_frame, text="Num Ranges:").pack(side="left", padx=5)
        self.num_roi_ranges_entry = ttk.Entry(multi_frame, width=5)
        self.num_roi_ranges_entry.insert(0, "1")
        self.num_roi_ranges_entry.pack(side="left", padx=5)
        
        ttk.Button(multi_frame, text="Setup Ranges", command=self.setup_multiple_rois).pack(side="left", padx=5)
        
        self.roi_ranges_label = ttk.Label(multi_frame, text="Configured: 0", foreground="gray")
        self.roi_ranges_label.pack(side="left", padx=10)

        self.bounding_box_ranges = []

    def _build_filters_tab(self):
        # --- Configuração TOML ---
        toml_frame = ttk.LabelFrame(self.tab_filters, text="File Management (TOML)", padding=15)
        toml_frame.pack(fill="x", padx=10, pady=10)
        
        btn_grid = ttk.Frame(toml_frame)
        btn_grid.pack(fill="x")
        
        ttk.Button(btn_grid, text="Load TOML", command=self.load_config_file).pack(side="left", padx=2)
        ttk.Button(btn_grid, text="Save Current", command=self.save_current_config_to_toml).pack(side="left", padx=2)
        ttk.Button(btn_grid, text="Create Template", command=self.create_default_toml_template).pack(side="left", padx=2)
        ttk.Button(btn_grid, text="Help", command=self.show_help).pack(side="right", padx=2)
        
        self.toml_label = ttk.Label(toml_frame, text="No file loaded", foreground="gray")
        self.toml_label.pack(pady=5)


    def create_default_toml_template(self):
        from tkinter import filedialog, messagebox

        # Create a root window for the dialog
        dialog_root = tk.Tk()
        dialog_root.withdraw()
        dialog_root.attributes("-topmost", True)

        file_path = filedialog.asksaveasfilename(
            parent=dialog_root,
            title="Create Default TOML Configuration Template",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialfile="mediapipe_config_template.toml",
        )

        if file_path:
            default_config = get_default_config()
            # Build dict in format expected by save_config_to_toml
            save_config = {
                "min_detection_confidence": default_config["mediapipe"]["min_detection_confidence"],
                "min_tracking_confidence": default_config["mediapipe"]["min_tracking_confidence"],
                "model_complexity": default_config["mediapipe"]["model_complexity"],
                "enable_segmentation": default_config["mediapipe"]["enable_segmentation"],
                "smooth_segmentation": default_config["mediapipe"]["smooth_segmentation"],
                "static_image_mode": default_config["mediapipe"]["static_image_mode"],
                "apply_filtering": default_config["mediapipe"]["apply_filtering"],
                "estimate_occluded": default_config["mediapipe"]["estimate_occluded"],
                "enable_median_filter": default_config["mediapipe"].get("enable_median_filter", False),
                "median_kernel_size": default_config["mediapipe"].get("median_kernel_size", 5),
                "enable_resize": default_config["video_resize"]["enable_resize"],
                "resize_scale": default_config["video_resize"]["resize_scale"],
                "enable_padding": ENABLE_PADDING_DEFAULT,
                "pad_start_frames": PAD_START_FRAMES_DEFAULT,
                "enable_crop": default_config["bounding_box"]["enable_crop"],
                "bbox_x_min": default_config["bounding_box"]["bbox_x_min"],
                "bbox_y_min": default_config["bounding_box"]["bbox_y_min"],
                "bbox_x_max": default_config["bounding_box"]["bbox_x_max"],
                "bbox_y_max": default_config["bounding_box"]["bbox_y_max"],
                "enable_resize_crop": default_config["bounding_box"]["enable_resize_crop"],
                "resize_crop_scale": default_config["bounding_box"]["resize_crop_scale"],
                "enable_reverse_padding": default_config.get("enable_reverse_padding", False),
                "pad_end_frames": default_config.get("pad_end_frames", 0),
                "bounding_box_ranges": default_config.get("bounding_box_ranges", []),
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

    def setup_multiple_rois(self):
        """Setup multiple ROI ranges with visual bounding box selection for each"""

        try:
            num_ranges = int(self.num_roi_ranges_entry.get())
            if num_ranges < 1:
                messagebox.showerror("Error", "Number of ROI ranges must be at least 1.")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for ROI ranges.")
            return

        if not self.input_dir:
            messagebox.showerror(
                "Error", "No input directory specified. Please select input directory first."
            )
            return

        # Find first video file
        video_files = [
            f
            for f in Path(self.input_dir).glob("*.*")
            if f.suffix.lower() in [".mp4", ".avi", ".mov"]
        ]

        if not video_files:
            messagebox.showerror("Error", "No video files found in input directory.")
            return

        first_video = str(video_files[0])

        # Get video info for cap
        cap = cv2.VideoCapture(first_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Reset ranges list
        self.bounding_box_ranges = []

        for i in range(num_ranges):
            # Determine start frame default and min value based on previous range
            default_start = 0
            min_start = 0
            if i > 0 and self.bounding_box_ranges:
                last_end = self.bounding_box_ranges[-1].get("frame_end", 0)
                if last_end == float("inf"):
                    # If previous range ends at infinity, we can't really suggest a valid start > inf.
                    # We'll just default to Total Frames or 0, but user likely made a mistake or wants to override?
                    # Let's set it to total_frames - 1 as a placeholder or just 0 if robust.
                    default_start = total_frames
                    min_start = 0 # Allow overlap if user insists? Or just fail?
                else:
                    default_start = int(last_end) + 1
                    min_start = 0 # We don't strictly enforce > previous end here to allow flexibility/overlaps if needed, 
                                  # though typically sequential. minvalue=min_start in prompt enforces it if we want.

            frame_start = simpledialog.askinteger(
                f"ROI Range {i + 1}/{num_ranges}",
                f"Enter START frame for ROI range {i + 1}:\n(Video has {total_frames} frames, 0-indexed)",
                minvalue=0, 
                maxvalue=None, # Removed strict enforcement dependent on previous range to avoid complex validation logic here
                initialvalue=default_start,
            )
            if frame_start is None:
                messagebox.showinfo("Cancelled", f"Setup cancelled at range {i + 1}.")
                break

            # For last range, suggest "inf" equivalent (total_frames)
            default_end = total_frames - 1 if i == num_ranges - 1 else frame_start + 100
            frame_end_str = simpledialog.askstring(
                f"ROI Range {i + 1}/{num_ranges}",
                f"Enter END frame for ROI range {i + 1}:\n(Enter 'inf' for last frame, or a number up to {total_frames - 1})",
                initialvalue="inf" if i == num_ranges - 1 else str(min(default_end, total_frames - 1)),
            )
            if frame_end_str is None:
                messagebox.showinfo("Cancelled", f"Setup cancelled at range {i + 1}.")
                break

            # Parse frame_end
            if frame_end_str.lower() == "inf":
                frame_end = float("inf")
            else:
                try:
                    frame_end = int(frame_end_str)
                    if frame_end < frame_start:
                        messagebox.showerror("Error", "End frame cannot be before start frame.")
                        break
                except ValueError:
                    messagebox.showerror("Error", f"Invalid end frame: {frame_end_str}")
                    break

            # Ask for ROI type (inclusion or exclusion)
            roi_type_choice = simpledialog.askstring(
                f"ROI Range {i + 1}/{num_ranges} - Type",
                f"Enter ROI type for range {i + 1}:\n'inclusion' (or '1') = detect only inside this ROI\n'exclusion' (or '0') = ignore this area\n\n(Default: inclusion)",
                initialvalue="inclusion"
            )
            if roi_type_choice is None:
                roi_type_choice = "inclusion"
            
            # Map 1/0 shortcuts to full names
            choice_str = roi_type_choice.lower().strip()
            if choice_str == "1":
                roi_type = "inclusion"
            elif choice_str == "0":
                roi_type = "exclusion"
            else:
                roi_type = choice_str
                
            if roi_type not in ["inclusion", "exclusion"]:
                roi_type = "inclusion"  # Default to inclusion

            # Get resize crop scale for this range (only for inclusion ROIs)
            resize_scale = 1
            if roi_type == "inclusion":
                resize_scale = simpledialog.askinteger(
                    f"ROI Range {i + 1}/{num_ranges}",
                    f"Enter resize crop scale for range {i + 1} (2-8):",
                    minvalue=1,
                    maxvalue=10,
                    initialvalue=2,
                )
                if resize_scale is None:
                    resize_scale = 2

            # Now select the bounding box visually
            roi_type_text = "INCLUSION" if roi_type == "inclusion" else "EXCLUSION"
            messagebox.showinfo(
                f"Select {roi_type_text} ROI for Range {i + 1}",
                f"Select the ROI for frames {frame_start} to {frame_end_str}.\n\nType: {roi_type_text}\nDrag to select region, press SPACE or ENTER to confirm."
            )

            # Open video at the specified start frame
            cap = cv2.VideoCapture(first_video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                messagebox.showerror("Error", f"Could not read frame {frame_start} from video.")
                break

            # Let user select ROI
            cv2.namedWindow("Select ROI for Range " + str(i + 1), cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select ROI for Range " + str(i + 1), 1280, 720)
            bbox = cv2.selectROI("Select ROI for Range " + str(i + 1), frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            if bbox[2] == 0 or bbox[3] == 0:
                messagebox.showinfo("Skipped", f"ROI selection skipped for range {i + 1}.")
                continue

            x_min, y_min, w, h = bbox
            x_max = x_min + w
            y_max = y_min + h

            # Add to ranges list
            range_config = {
                "frame_start": frame_start,
                "frame_end": frame_end,
                "roi_type": roi_type,  # "inclusion" or "exclusion"
                "enable_crop": roi_type == "inclusion",  # Only enable crop for inclusion
                "bbox_x_min": x_min,
                "bbox_y_min": y_min,
                "bbox_x_max": x_max,
                "bbox_y_max": y_max,
                "enable_resize_crop": roi_type == "inclusion",  # Only resize crop for inclusion
                "resize_crop_scale": resize_scale if roi_type == "inclusion" else 1,
            }
            self.bounding_box_ranges.append(range_config)

            print(f"ROI Range {i + 1} configured: frames {frame_start}-{frame_end_str}, bbox ({x_min}, {y_min}) to ({x_max}, {y_max})")

        # Update status label
        self.roi_ranges_label.config(
            text=f"ROI ranges configured: {len(self.bounding_box_ranges)}",
            foreground="green" if self.bounding_box_ranges else "gray"
        )

        if self.bounding_box_ranges:
            messagebox.showinfo(
                "Setup Complete",
                f"Configured {len(self.bounding_box_ranges)} ROI range(s).\n\nUse 'Save Current Config' to save to TOML file."
            )

    def save_current_config_to_toml(self):
        """Salva o estado atual da GUI em um arquivo TOML."""
        
        # Pega o caminho do arquivo
        file_path = filedialog.asksaveasfilename(
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            title="Salvar Configuração TOML"
        )
        if not file_path:
            return

        try:
            # Constrói o dicionário com TODOS os campos da tela
            config = {
                # MediaPipe
                "min_detection_confidence": float(self.min_detection_entry.get()),
                "min_tracking_confidence": float(self.min_tracking_entry.get()),
                "model_complexity": int(self.model_complexity_entry.get()),
                "enable_segmentation": self.enable_segmentation_entry.get().lower() == 'true',
                "smooth_segmentation": self.smooth_segmentation_entry.get().lower() == 'true',
                "static_image_mode": self.static_image_mode_entry.get().lower() == 'true',
                "apply_filtering": self.apply_filtering_entry.get().lower() == 'true',
                "estimate_occluded": self.estimate_occluded_entry.get().lower() == 'true',
                
                # Simple Median Filter
                "enable_median_filter": self.enable_median_filter_var.get() if hasattr(self, 'enable_median_filter_var') else False,
                "median_kernel_size": int(self.median_kernel_size_entry.get()) if hasattr(self, 'median_kernel_size_entry') else 5,

                # Resize & Padding
                "enable_resize": self.enable_resize_entry.get().lower() == 'true',
                "resize_scale": float(self.resize_scale_entry.get()),
                "enable_padding": self.enable_padding_entry.get().lower() == 'true',
                "pad_start_frames": int(self.pad_start_frames_entry.get()),
                "enable_reverse_padding": self.enable_reverse_padding_entry.get().lower() == 'true',
                "pad_end_frames": int(self.pad_end_frames_entry.get()),

                # ROI
                "enable_crop": self.enable_crop_var.get(),
                "bbox_x_min": int(self.bbox_x_min_entry.get()),
                "bbox_y_min": int(self.bbox_y_min_entry.get()),
                "bbox_x_max": int(self.bbox_x_max_entry.get()),
                "bbox_y_max": int(self.bbox_y_max_entry.get()),
                "enable_resize_crop": self.enable_resize_crop_var.get(),
                "resize_crop_scale": float(self.resize_crop_scale_entry.get()),
                
                # Advanced ROI
                "roi_polygon_points": self.roi_polygon_points,
                "bounding_box_ranges": self.bounding_box_ranges,
            }

            # Try to get video total frames if input_dir is set
            try:
                if self.input_dir:
                    video_files = [
                        f
                        for f in Path(self.input_dir).glob("*.*")
                        if f.suffix.lower() in [".mp4", ".avi", ".mov"]
                    ]
                    if video_files:
                        cap = cv2.VideoCapture(str(video_files[0]))
                        if cap.isOpened():
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            config["video_total_frames"] = total_frames
                            cap.release()
            except Exception as e:
                print(f"Warning: Could not determine total frames: {e}")

            # Use the global save function to ensure correct formatting and headers
            save_config_to_toml(config, file_path)
            
            messagebox.showinfo("Sucesso", f"Configuração salva em:\n{file_path}")
            self.loaded_config = config
            
            # Atualiza label na aba de Filtros
            if hasattr(self, 'toml_label'):
                self.toml_label.config(text=os.path.basename(file_path), foreground="green")

        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Não foi possível salvar o arquivo:\n{e}")

    def load_config_file(self):
        # Create a root window for the dialog
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
                        text=f"TOML loaded: {os.path.basename(file_path)}", foreground="green"
                    )
                    # Populate GUI fields with loaded config values
                    self.populate_fields_from_config(config)
                    # Show summary of loaded config
                    summary = f"TOML loaded: {os.path.basename(file_path)}\n\n"
                    summary += (
                        f"min_detection_confidence: {config.get('min_detection_confidence')}\n"
                    )
                    summary += f"min_tracking_confidence: {config.get('min_tracking_confidence')}\n"
                    summary += f"model_complexity: {config.get('model_complexity')}\n"
                    summary += f"enable_segmentation: {config.get('enable_segmentation')}\n"
                    summary += f"smooth_segmentation: {config.get('smooth_segmentation')}\n"
                    summary += f"static_image_mode: {config.get('static_image_mode')}\n"
                    summary += f"apply_filtering: {config.get('apply_filtering')}\n"
                    summary += f"estimate_occluded: {config.get('estimate_occluded')}\n"
                    summary += f"enable_resize: {config.get('enable_resize')}\n"
                    summary += f"resize_scale: {config.get('resize_scale')}\n"
                    summary += (
                        f"enable_advanced_filtering: {config.get('enable_advanced_filtering')}\n"
                    )
                    summary += f"interp_method: {config.get('interp_method')}\n"
                    summary += f"smooth_method: {config.get('smooth_method')}\n"
                    summary += f"max_gap: {config.get('max_gap')}\n"
                    if "smooth_params" in config:
                        summary += f"smooth_params: {config['smooth_params']}\n"
                    summary += (
                        f"enable_padding: {config.get('enable_padding', ENABLE_PADDING_DEFAULT)}\n"
                    )
                    summary += f"pad_start_frames: {config.get('pad_start_frames', PAD_START_FRAMES_DEFAULT)}\n"
                    summary += f"enable_crop: {config.get('enable_crop', False)}\n"
                    if config.get("enable_crop", False):
                        summary += f"bbox: ({config.get('bbox_x_min', 0)}, {config.get('bbox_y_min', 0)}) to ({config.get('bbox_x_max', 1920)}, {config.get('bbox_y_max', 1080)})\n"
                        summary += (
                            f"enable_resize_crop: {config.get('enable_resize_crop', False)}\n"
                        )
                        if config.get("enable_resize_crop", False):
                            summary += f"resize_crop_scale: {config.get('resize_crop_scale', 2)}\n"
                    print("\n=== TOML configuration loaded and GUI updated ===\n" + summary)
                    from tkinter import messagebox

                    messagebox.showinfo(
                        "TOML Parameters Loaded",
                        "Configuration loaded and GUI fields updated!\n\n" + summary,
                    )
                else:
                    self.toml_label.config(text="Error loading TOML", foreground="red")
            except Exception as e:
                self.toml_label.config(text=f"Error: {e}", foreground="red")
                from tkinter import messagebox

                messagebox.showerror("Error", f"Failed to load TOML: {e}")

        dialog_root.destroy()

    def show_help(self):
        """Open local HTML help file in browser"""
        try:
            # Calculate path to help file relative to this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            help_path = os.path.join(script_dir, "help", "markerless_2d_analysis.html")
            
            if os.path.exists(help_path):
                webbrowser.open(f"file://{help_path}")
            else:
                from tkinter import messagebox
                messagebox.showwarning(
                    "Help Not Found", 
                    f"Could not find help file at:\n{help_path}\n\nPlease check your installation."
                )
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to open help: {str(e)}")
    def update_normalized_coords(self, event=None):
        """Update normalized coordinates display based on pixel coordinates"""
        try:
            x_min = int(self.bbox_x_min_entry.get() or 0)
            y_min = int(self.bbox_y_min_entry.get() or 0)
            x_max = int(self.bbox_x_max_entry.get() or 1920)
            y_max = int(self.bbox_y_max_entry.get() or 1080)

            # Get video dimensions if available
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

            # Default display if no video dimensions available
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

        # Find first video file
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

        # Read first frame
        ret, original_frame = cap.read()
        cap.release()

        if not ret:
            messagebox.showerror("Error", "Could not read first frame from video.")
            return

        # Get original dimensions
        orig_height, orig_width = original_frame.shape[:2]

        # Calculate maximum display size (fit to screen)
        max_display_width = 1920
        max_display_height = 1080

        # Check if frame needs to be resized to fit screen
        display_frame = original_frame.copy()
        scale_factor = 1.0

        if orig_width > max_display_width or orig_height > max_display_height:
            # Calculate scale to fit screen while maintaining aspect ratio
            scale_w = max_display_width / orig_width
            scale_h = max_display_height / orig_height
            scale_factor = min(scale_w, scale_h)

            # Resize frame for display
            display_width = int(orig_width * scale_factor)
            display_height = int(orig_height * scale_factor)
            display_frame = cv2.resize(
                original_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR
            )
            print(
                f"Frame resized for display: {orig_width}x{orig_height} -> {display_width}x{display_height} (scale: {scale_factor:.2f})"
            )
        else:
            display_width = orig_width
            display_height = orig_height

        # Let user select ROI on display frame
        print(f"Select ROI from first frame of: {first_video.name}")
        print("Instructions: Drag to select region, press SPACE or ENTER to confirm, ESC to cancel")
        print("Note: Window is resizable - you can resize it to see the full frame")

        # Create resizable window before selectROI
        window_name = "Select ROI - Press SPACE/ENTER to confirm, ESC to cancel"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)

        roi = cv2.selectROI(window_name, display_frame, False)

        # Check if user cancelled (roi is all zeros)
        if roi[2] == 0 or roi[3] == 0:
            print("ROI selection cancelled")
            cv2.destroyAllWindows()
            return

        # roi format: (x, y, width, height) - these are in display_frame coordinates
        x_display, y_display, w_display, h_display = roi

        # Convert ROI coordinates from display frame back to original video dimensions
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

        # Clamp to original frame dimensions
        x_min = max(0, min(x_min, orig_width - 1))
        y_min = max(0, min(y_min, orig_height - 1))
        x_max = max(x_min + 1, min(x_max, orig_width))
        y_max = max(y_min + 1, min(y_max, orig_height))

        # Update entry fields
        self.bbox_x_min_entry.delete(0, tk.END)
        self.bbox_x_min_entry.insert(0, str(x_min))
        self.bbox_y_min_entry.delete(0, tk.END)
        self.bbox_y_min_entry.insert(0, str(y_min))
        self.bbox_x_max_entry.delete(0, tk.END)
        self.bbox_x_max_entry.insert(0, str(x_max))
        self.bbox_y_max_entry.delete(0, tk.END)
        self.bbox_y_max_entry.insert(0, str(y_max))

        # Update normalized coordinates display
        self.update_normalized_coords()

        # Enable crop checkbox
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

        # Find first video file
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
            # Calculate bounding box from polygon for display
            x_coords = [pt[0] for pt in roi_poly]
            y_coords = [pt[1] for pt in roi_poly]
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)

            # Update entry fields with bounding box (for compatibility)
            self.bbox_x_min_entry.delete(0, tk.END)
            self.bbox_x_min_entry.insert(0, str(x_min))
            self.bbox_y_min_entry.delete(0, tk.END)
            self.bbox_y_min_entry.insert(0, str(y_min))
            self.bbox_x_max_entry.delete(0, tk.END)
            self.bbox_x_max_entry.insert(0, str(x_max))
            self.bbox_y_max_entry.delete(0, tk.END)
            self.bbox_y_max_entry.insert(0, str(y_max))

            # Store polygon points in config (will be saved to TOML)
            self.roi_polygon_points = roi_poly.tolist() if hasattr(roi_poly, "tolist") else roi_poly

            # Update normalized coordinates display
            self.update_normalized_coords()

            # Enable crop checkbox
            self.enable_crop_var.set(True)

            print(f"Polygon ROI selected with {len(roi_poly)} points")
            messagebox.showinfo(
                "ROI Selected", f"Polygon ROI selected with {len(roi_poly)} points.\nBounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})"
            )
        else:
            print("Polygon ROI selection cancelled or invalid")

    def populate_fields_from_config(self, config):
        """Preenche a GUI com valores de um dicionário carregado."""
        
        # --- MediaPipe & Model ---
        if 'min_detection_confidence' in config:
            self.min_detection_entry.delete(0, tk.END)
            self.min_detection_entry.insert(0, str(config['min_detection_confidence']))
        if 'min_tracking_confidence' in config:
            self.min_tracking_entry.delete(0, tk.END)
            self.min_tracking_entry.insert(0, str(config['min_tracking_confidence']))
        if 'model_complexity' in config:
            self.model_complexity_entry.delete(0, tk.END)
            self.model_complexity_entry.insert(0, str(config['model_complexity']))
        if 'enable_segmentation' in config:
            self.enable_segmentation_entry.delete(0, tk.END)
            self.enable_segmentation_entry.insert(0, str(config['enable_segmentation']))
        if 'smooth_segmentation' in config:
            self.smooth_segmentation_entry.delete(0, tk.END)
            self.smooth_segmentation_entry.insert(0, str(config['smooth_segmentation']))
        if 'static_image_mode' in config:
            self.static_image_mode_entry.delete(0, tk.END)
            self.static_image_mode_entry.insert(0, str(config['static_image_mode']))
        if 'apply_filtering' in config:
            self.apply_filtering_entry.delete(0, tk.END)
            self.apply_filtering_entry.insert(0, str(config['apply_filtering']))
        if 'estimate_occluded' in config:
            self.estimate_occluded_entry.delete(0, tk.END)
            self.estimate_occluded_entry.insert(0, str(config['estimate_occluded']))

        # --- Simple Median Filter ---
        if hasattr(self, 'enable_median_filter_var') and 'enable_median_filter' in config:
            self.enable_median_filter_var.set(config['enable_median_filter'])
        if hasattr(self, 'median_kernel_size_entry') and 'median_kernel_size' in config:
            self.median_kernel_size_entry.delete(0, tk.END)
            self.median_kernel_size_entry.insert(0, str(config['median_kernel_size']))

        # --- Processamento (Resize & Padding) ---
        if 'enable_resize' in config:
            self.enable_resize_entry.delete(0, tk.END)
            self.enable_resize_entry.insert(0, str(config['enable_resize']))
        if 'resize_scale' in config:
            self.resize_scale_entry.delete(0, tk.END)
            self.resize_scale_entry.insert(0, str(config['resize_scale']))
        if 'enable_padding' in config:
            self.enable_padding_entry.delete(0, tk.END)
            self.enable_padding_entry.insert(0, str(config['enable_padding']))
        if 'pad_start_frames' in config:
            self.pad_start_frames_entry.delete(0, tk.END)
            self.pad_start_frames_entry.insert(0, str(config['pad_start_frames']))
        if 'enable_reverse_padding' in config:
            self.enable_reverse_padding_entry.delete(0, tk.END)
            self.enable_reverse_padding_entry.insert(0, str(config['enable_reverse_padding']))
        if 'pad_end_frames' in config:
            self.pad_end_frames_entry.delete(0, tk.END)
            self.pad_end_frames_entry.insert(0, str(config['pad_end_frames']))

        # --- ROI ---
        if 'enable_crop' in config:
            # Checkbox var precisa ser setada
            val = str(config['enable_crop']).lower() == 'true'
            self.enable_crop_var.set(val)
        
        if 'bbox_x_min' in config:
            self.bbox_x_min_entry.delete(0, tk.END)
            self.bbox_x_min_entry.insert(0, str(config['bbox_x_min']))
        if 'bbox_y_min' in config:
            self.bbox_y_min_entry.delete(0, tk.END)
            self.bbox_y_min_entry.insert(0, str(config['bbox_y_min']))
        if 'bbox_x_max' in config:
            self.bbox_x_max_entry.delete(0, tk.END)
            self.bbox_x_max_entry.insert(0, str(config['bbox_x_max']))
        if 'bbox_y_max' in config:
            self.bbox_y_max_entry.delete(0, tk.END)
            self.bbox_y_max_entry.insert(0, str(config['bbox_y_max']))

        if 'enable_resize_crop' in config:
            val = str(config['enable_resize_crop']).lower() == 'true'
            self.enable_resize_crop_var.set(val)

        if 'resize_crop_scale' in config:
            self.resize_crop_scale_entry.delete(0, tk.END)
            self.resize_crop_scale_entry.insert(0, str(config['resize_crop_scale']))

        # --- ROI Polígono e Ranges ---
        if 'roi_polygon_points' in config:
            val = config['roi_polygon_points']
            # Tratamento para string "None" vindo do TOML
            if isinstance(val, str) and val.lower() == "none":
                self.roi_polygon_points = None
            else:
                self.roi_polygon_points = val

        if 'bounding_box_ranges' in config:
            self.bounding_box_ranges = config['bounding_box_ranges']
            self.roi_ranges_label.config(text=f"Configurados: {len(self.bounding_box_ranges)}")
            self.num_roi_ranges_entry.delete(0, tk.END)
            self.num_roi_ranges_entry.insert(0, str(len(self.bounding_box_ranges)))
            
        # Atualiza visualmente as coords normalizadas
        self.update_normalized_coords()

        # Median Filter (Simple)
        if hasattr(self, "enable_median_filter_var") and "enable_median_filter" in config:
            self.enable_median_filter_var.set(config["enable_median_filter"])
        if hasattr(self, "median_kernel_size_entry") and "median_kernel_size" in config:
            self.median_kernel_size_entry.delete(0, tk.END)
            self.median_kernel_size_entry.insert(0, str(config["median_kernel_size"]))

    def apply(self):
        if self.use_toml and self.loaded_config:
            self.result = self.loaded_config
        else:
            self.result = {
                # MediaPipe & Simple Processing
                "min_detection_confidence": float(self.min_detection_entry.get()),
                "min_tracking_confidence": float(self.min_tracking_entry.get()),
                "model_complexity": int(self.model_complexity_entry.get()),
                "enable_segmentation": self.enable_segmentation_entry.get().lower() == "true",
                "smooth_segmentation": self.smooth_segmentation_entry.get().lower() == "true",
                "static_image_mode": self.static_image_mode_entry.get().lower() == "true",
                "apply_filtering": self.apply_filtering_entry.get().lower() == "true",
                "estimate_occluded": self.estimate_occluded_entry.get().lower() == "true",
                # New Post-Processing
                "enable_median_filter": self.enable_median_filter_var.get(),
                "median_kernel_size": int(self.median_kernel_size_entry.get()),
                
                # Resizing & Padding
                "enable_resize": self.enable_resize_entry.get().lower() == "true",
                "resize_scale": int(self.resize_scale_entry.get()),
                "enable_padding": self.enable_padding_entry.get().lower() == "true",
                "pad_start_frames": int(self.pad_start_frames_entry.get()),
                "enable_reverse_padding": self.enable_reverse_padding_entry.get().lower() == "true",
                "pad_end_frames": int(self.pad_end_frames_entry.get()),
                
                # Bounding box settings
                "enable_crop": self.enable_crop_var.get(),
                "bbox_x_min": int(self.bbox_x_min_entry.get() or 0),
                "bbox_y_min": int(self.bbox_y_min_entry.get() or 0),
                "bbox_x_max": int(self.bbox_x_max_entry.get() or 1920),
                "bbox_y_max": int(self.bbox_y_max_entry.get() or 1080),
                "enable_resize_crop": self.enable_resize_crop_var.get(),
                "resize_crop_scale": int(self.resize_crop_scale_entry.get() or 2),
                "roi_polygon_points": getattr(self, "roi_polygon_points", None),  # Store polygon points if selected
                "bounding_box_ranges": getattr(self, "bounding_box_ranges", []),  # Store multiple ROI ranges
            }


class DeviceSelectionDialog(tk.simpledialog.Dialog):
    """Dialog to select CPU or GPU backend for processing"""

    def __init__(self, parent, available_backends):
        """
        Args:
            parent: Tkinter parent window
            available_backends: dict with {backend_name: (available, info_dict, test_result)}
        """
        self.available_backends = available_backends
        self.selected_device = "cpu"  # Default
        super().__init__(parent, title="Select Processing Device")

    def body(self, master):
        tk.Label(master, text="Select processing device:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(10, 20), sticky="w"
        )

        # Create a shared StringVar for radio buttons
        self.device_var = tk.StringVar(value="cpu")

        row = 1

        # CPU option (always available)
        self.cpu_var = tk.Radiobutton(
            master,
            text="CPU (Standard Processing)",
            variable=self.device_var,
            value="cpu",
            command=lambda: setattr(self, "selected_device", "cpu"),
            font=("Arial", 9),
        )
        self.cpu_var.grid(row=row, column=0, columnspan=2, sticky="w", padx=20, pady=5)
        row += 1

        # GPU backend options
        backend_labels = {
            "nvidia": "GPU (NVIDIA CUDA - High Performance)",
            "rocm": "GPU (AMD ROCm - High Performance)",
            "mps": "GPU (Apple Silicon MPS - High Performance)",
        }

        for backend_name in ["nvidia", "rocm", "mps"]:
            if backend_name in self.available_backends:
                available, info, error = self.available_backends[backend_name]
                test_result = self.available_backends.get(f"{backend_name}_test", (False, "Not tested"))

                backend_text = backend_labels.get(backend_name, f"GPU ({backend_name.upper()})")
                if not available:
                    backend_text += " [NOT AVAILABLE]"

                backend_var = tk.Radiobutton(
                    master,
                    text=backend_text,
                    variable=self.device_var,
                    value=backend_name,
                    command=lambda b=backend_name: setattr(self, "selected_device", b),
                    font=("Arial", 9),
                    state="normal" if available else "disabled",
                )
                backend_var.grid(row=row, column=0, columnspan=2, sticky="w", padx=20, pady=5)
                row += 1

        # Info frame for selected backend
        info_frame = tk.LabelFrame(master, text="Device Information", padx=10, pady=10)
        info_frame.grid(row=row, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        row += 1

        # Find first available GPU backend for info display
        info_text = ""
        for backend_name in ["nvidia", "rocm", "mps"]:
            if backend_name in self.available_backends:
                available, info, error = self.available_backends[backend_name]
                test_result = self.available_backends.get(f"{backend_name}_test", (False, "Not tested"))

                if available and info:
                    info_text = f"Backend: {backend_name.upper()}\n"
                    info_text += f"GPU: {info.get('name', 'Unknown')}\n"
                    info_text += f"Driver: {info.get('driver_version', 'Unknown')}\n"
                    if info.get("memory_total_mb", 0) > 0:
                        info_text += f"Memory: {info['memory_total_mb'] / 1024:.1f} GB\n"
                    if test_result[0]:
                        info_text += "Status: ✓ GPU tested and working"
                    else:
                        info_text += f"Status: ⚠ {test_result[1]}"
                    break

        if not info_text:
            # No GPU available
            info_text = "No GPU backends available.\n"
            info_text += "Requirements:\n"
            info_text += "• NVIDIA: NVIDIA GPU with CUDA support and drivers\n"
            info_text += "• ROCm: AMD GPU with ROCm installed\n"
            info_text += "• MPS: Apple Silicon (arm64) on macOS\n"
            info_text += "• MediaPipe with GPU delegate support"

        tk.Label(info_frame, text=info_text, justify="left", font=("Arial", 8)).grid(
            row=0, column=0, sticky="w"
        )

        return self.cpu_var

    def apply(self):
        # Get selected value from StringVar
        selected = self.device_var.get()
        if selected in ["nvidia", "rocm", "mps"]:
            # Check if this backend is available
            if selected in self.available_backends:
                available, _, _ = self.available_backends[selected]
                if available:
                    self.result = selected
                else:
                    self.result = "cpu"
            else:
                self.result = "cpu"
        else:
            self.result = "cpu"


def select_free_polygon_roi(video_path):
    """
    Let the user draw a free polygon ROI on the first frame of the video.
    Left click adds points, right click removes the last point, Enter confirms,
    Esc skips, and 'r' resets. Returns a numpy array of int32 points or None.
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

        # Create window with WINDOW_NORMAL flag for resizability
        if platform.system() == "Darwin":
            window_flags = cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_KEEPRATIO
        else:
            window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO

        cv2.namedWindow(window_name, window_flags)

        # Set mouse callback BEFORE showing window
        cv2.setMouseCallback(window_name, mouse_callback)

        # Show window first to ensure it exists
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

        # Set window size after showing
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

        # Colors for polygon
        polygon_color = (255, 255, 0)  # Cyan (BGR)
        point_color = (0, 255, 255)  # Yellow (BGR)
        closing_line_color = (255, 0, 255)  # Magenta (BGR)

        # Help text
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

            # Draw help text
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


def get_pose_config(existing_root=None, input_dir=None):
    if existing_root is not None:
        root = existing_root
    else:
        root = tk.Tk()
        root.withdraw()

    # Remove automatic creation of default TOML. Only create via button or after processing.

    dialog = ConfidenceInputDialog(root, input_dir=input_dir)
    if dialog.result:
        print("Configuration applied successfully!")
        return dialog.result
    else:
        messagebox.showerror("Error", "Configuration cancelled - no values entered.")
        return None


def resize_video_opencv(input_file, output_file, scale_factor, progress_callback=None):
    """
    Resize video using OpenCV and return metadata for coordinate conversion.
    """
    try:
        print(f"Resizing video: {input_file}")

        # Open input video
        cap = cv2.VideoCapture(input_file)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_file}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate new dimensions
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Store metadata for coordinate conversion
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

        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_file, fourcc, 30, (new_width, new_height))

        # Process the video frame by frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            resized_frame = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )
            out.write(resized_frame)

            # Update progress every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                if progress_callback:
                    progress_callback(f"Resizing: {progress:.1f}% ({frame_count}/{total_frames})")

        # Release resources
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


def convert_coordinates_to_original(df, metadata, progress_callback=None):
    """
    Convert coordinates from resized video back to original video dimensions.
    """
    if not metadata or metadata["scale_factor"] == 1:
        return df

    converted_df = df.copy()
    scale_factor = metadata["scale_factor"]

    if progress_callback:
        progress_callback(f"Converting coordinates back to original scale (/{scale_factor})")

    # Find coordinate columns (MediaPipe format: *_x, *_y)
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
                        # Convert back to original scale
                        original_x = row[x_col] / scale_factor
                        original_y = row[y_col] / scale_factor
                        converted_df.at[idx, x_col] = original_x
                        converted_df.at[idx, y_col] = original_y
                processed += 1

    if progress_callback:
        progress_callback(f"Converted {processed} coordinate pairs back to original scale")

    return converted_df


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

    Args:
        landmarks: List of [x, y, z] in normalized crop space (0-1)
        bbox_config: Dict with bbox coordinates and resize settings
        crop_width, crop_height: Dimensions of cropped region (after resize if enabled)
        processing_width, processing_height: Dimensions of processing frame (may be resized video)
        original_width, original_height: Dimensions of original video (optional, inferred if not provided)

    Returns:
        List of [x, y, z] in normalized original video space (0-1)
    """
    enable_resize_crop = bbox_config.get("enable_resize_crop", False)
    resize_crop_scale = bbox_config.get("resize_crop_scale", 1)
    video_resized = bbox_config.get("video_resized", False)
    video_resize_scale = bbox_config.get("resize_scale", 1.0)

    # Get original video dimensions
    if original_width is None or original_height is None:
        # Infer from processing dimensions and resize scale
        if video_resized and video_resize_scale > 1:
            original_width = int(processing_width / video_resize_scale)
            original_height = int(processing_height / video_resize_scale)
        else:
            original_width = processing_width
            original_height = processing_height

    # Use scaled bbox coordinates (they are in processing video coordinates)
    bbox_x_min_processing = bbox_config.get("bbox_x_min", 0)
    bbox_y_min_processing = bbox_config.get("bbox_y_min", 0)

    mapped_landmarks = []
    for landmark in landmarks:
        x_norm_crop, y_norm_crop, z = landmark

        # Step 1: Convert normalized crop → pixel crop-resized
        x_px_crop_resized = x_norm_crop * crop_width
        y_px_crop_resized = y_norm_crop * crop_height

        # Step 2: If resize crop enabled, convert pixel crop-resized → pixel crop
        if enable_resize_crop and resize_crop_scale > 1:
            x_px_crop = x_px_crop_resized / resize_crop_scale
            y_px_crop = y_px_crop_resized / resize_crop_scale
        else:
            x_px_crop = x_px_crop_resized
            y_px_crop = y_px_crop_resized

        # Step 3: Convert pixel crop → pixel full frame (in processing video coordinates)
        x_px_full_processing = x_px_crop + bbox_x_min_processing
        y_px_full_processing = y_px_crop + bbox_y_min_processing

        # Step 4: If video was resized, convert from processing video → original video
        if video_resized and video_resize_scale > 1:
            x_px_full_original = x_px_full_processing / video_resize_scale
            y_px_full_original = y_px_full_processing / video_resize_scale
        else:
            x_px_full_original = x_px_full_processing
            y_px_full_original = y_px_full_processing

        # Step 5: Convert pixel original video → normalized original video
        x_norm_full = x_px_full_original / original_width if original_width > 0 else 0.0
        y_norm_full = y_px_full_original / original_height if original_height > 0 else 0.0

        mapped_landmarks.append([x_norm_full, y_norm_full, z])

    return mapped_landmarks


def apply_interpolation_and_smoothing(df, config, progress_callback=None):
    """Apply interpolation and smoothing to MediaPipe landmark data."""
    if not config.get("enable_advanced_filtering", False):
        return df

    if progress_callback:
        progress_callback("Applying advanced filtering and smoothing...")

    # Get only landmark columns (exclude frame_index and any non-landmark columns)
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
                        # Kalman filter expects 2D array
                        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
                        col_filtered, _ = kf.smooth(col_padded)
                        col_filtered = col_filtered.flatten()[pad_width:-pad_width]
                        col_smooth = col.copy()
                        col_smooth[valid] = col_filtered
                        data[:, j] = col_smooth
                    except Exception as e:
                        print(f"Kalman smoothing failed for column {j}: {e}")
        elif smooth_method == "butterworth":
            # Use new user-controlled parameters if available, otherwise fall back to TOML/defaults
            # config comes from self.result in apply(). 
            # If loaded from TOML, these keys might be missing at top level, check smooth_params
            
            # Priority: Top-level config (GUI) > smooth_params (TOML) > Defaults
            cutoff = config.get("filter_cutoff")
            if cutoff is None:
                cutoff = smooth_params.get("cutoff", 6.0)
                
            median_kernel = config.get("median_kernel")
            if median_kernel is None:
                median_kernel = 5 # Default
                
            order = config.get("filter_order")
            if order is None:
                order = 4
                
            fs = smooth_params.get("fs", 30.0)
            
            # Apply the combined Median + Butterworth filter
            try:
                # filter_pose_data expects numpy array and handles iteration internally
                data = filter_pose_data(data, order=order, fc=cutoff, fs=fs, median_kernel=median_kernel)
                print(f"Applied Median (k={median_kernel}) + Butterworth (fc={cutoff}Hz) filter.")
            except Exception as e:
                print(f"Filtering failed: {e}")
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
        # Update DataFrame with smoothed data
        df[numeric_cols] = data
    except Exception as e:
        if progress_callback:
            progress_callback(f"Error during {smooth_method} smoothing: {str(e)}")
        print(f"Error during {smooth_method} smoothing: {str(e)}")
    return df


def apply_temporal_filter(landmarks_history, window=5):
    """Apply Savitzky-Golay filter to smooth landmark movement"""
    if len(landmarks_history) < window:
        return landmarks_history[-1]

    # Ensure window is odd
    if window % 2 == 0:
        window -= 1

    filtered_landmarks = []
    for i in range(len(landmarks_history[0])):
        # Extract history for this landmark
        landmark_data = [frame[i] for frame in landmarks_history]

        # Filter each dimension separately
        filtered_coords = []
        for dim in range(3):  # x, y, z
            values = [lm[dim] for lm in landmark_data if not np.isnan(lm[dim])]
            if len(values) >= window:
                try:
                    filtered = savgol_filter(values, window, 2)
                    filtered_coords.append(filtered[-1])
                except Exception as e:
                    print(f"Error during Savitzky-Golay filtering: {e}")
                    filtered_coords.append(landmark_data[-1][dim])
            else:
                filtered_coords.append(landmark_data[-1][dim])

        filtered_landmarks.append(filtered_coords)

    return filtered_landmarks


def estimate_occluded_landmarks(landmarks, landmarks_history=None):
    """Estimates occluded landmark positions based on anatomical constraints."""
    estimated = landmarks.copy()

    # Only proceed if we have some visible landmarks
    if all(np.isnan(lm[0]) for lm in landmarks):
        return landmarks

    # 1. Bilateral symmetry rules
    pairs = [
        (11, 12),  # shoulders
        (13, 14),  # elbows
        (15, 16),  # wrists
        (23, 24),  # hips
        (25, 26),  # knees
        (27, 28),  # ankles
    ]

    for left_idx, right_idx in pairs:
        left_visible = not np.isnan(landmarks[left_idx][0])
        right_visible = not np.isnan(landmarks[right_idx][0])

        if left_visible and not right_visible:
            if not np.isnan(landmarks[0][0]):
                center_x = landmarks[0][0]
                offset_x = landmarks[left_idx][0] - center_x
                estimated[right_idx][0] = center_x - offset_x
                estimated[right_idx][1] = landmarks[left_idx][1]
                estimated[right_idx][2] = landmarks[left_idx][2]

        elif right_visible and not left_visible and not np.isnan(landmarks[0][0]):
            center_x = landmarks[0][0]
            offset_x = landmarks[right_idx][0] - center_x
            estimated[left_idx][0] = center_x - offset_x
            estimated[left_idx][1] = landmarks[right_idx][1]
            estimated[left_idx][2] = landmarks[right_idx][2]

    # 2. Continuity rules for limbs
    if (
        not np.isnan(landmarks[11][0])
        and not np.isnan(landmarks[15][0])
        and np.isnan(landmarks[13][0])
    ):
        estimated[13][0] = (landmarks[11][0] + landmarks[15][0]) / 2
        estimated[13][1] = (landmarks[11][1] + landmarks[15][1]) / 2
        estimated[13][2] = (landmarks[11][2] + landmarks[15][2]) / 2

    if (
        not np.isnan(landmarks[12][0])
        and not np.isnan(landmarks[16][0])
        and np.isnan(landmarks[14][0])
    ):
        estimated[14][0] = (landmarks[12][0] + landmarks[16][0]) / 2
        estimated[14][1] = (landmarks[12][1] + landmarks[16][1]) / 2
        estimated[14][2] = (landmarks[12][2] + landmarks[16][2]) / 2

    # 3. Use history if available
    if landmarks_history and len(landmarks_history) > 0:
        for i, landmark in enumerate(estimated):
            if np.isnan(landmark[0]):
                for past_frame in reversed(landmarks_history):
                    if not np.isnan(past_frame[i][0]):
                        estimated[i] = past_frame[i]
                        break

    return estimated


def pad_signal(data, pad_width, mode="edge"):
    if pad_width == 0:
        return data
    return np.pad(data, (pad_width, pad_width), mode=mode)


def is_linux_system():
    """Check if running on Linux system"""
    return platform.system().lower() == "linux"


def detect_nvidia_gpu():
    """
    Detect if NVIDIA GPU is available and accessible.
    Returns tuple: (is_available: bool, gpu_info: dict, error_message: str)
    """
    gpu_info = {}
    error_message = None

    # Check for nvidia-smi command
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=5,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            if lines and lines[0]:
                # Parse first GPU info
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


def detect_rocm_gpu():
    """
    Detect if ROCm (AMD) GPU is available and accessible.
    Returns tuple: (is_available: bool, gpu_info: dict, error_message: str)
    """
    gpu_info = {}
    error_message = None

    # Check for rocm-smi command
    try:
        import subprocess

        # Try common ROCm paths
        rocm_paths = [
            "/opt/rocm/bin/rocm-smi",
            "/usr/bin/rocm-smi",
            "rocm-smi",
        ]

        rocm_smi_path = None
        for path in rocm_paths:
            if os.path.exists(path) or shutil.which(path):
                rocm_smi_path = path
                break

        if not rocm_smi_path:
            error_message = "rocm-smi not found (ROCm may not be installed)"
            return False, gpu_info, error_message

        result = subprocess.run(
            [
                rocm_smi_path,
                "--showid",
                "--showproductname",
                "--showdriverversion",
            ],
            capture_output=True,
            timeout=5,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Parse ROCm output (format may vary)
            lines = result.stdout.strip().split("\n")
            gpu_info = {
                "name": "AMD GPU (ROCm)",
                "driver_version": "Unknown",
                "memory_total_mb": 0,
                "count": 1,
            }
            # Try to extract GPU name from output
            for line in lines:
                if "Card series" in line or "Product Name" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        gpu_info["name"] = parts[1].strip()
                elif "Driver version" in line or "Driver Version" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        gpu_info["driver_version"] = parts[1].strip()

            return True, gpu_info, None
        else:
            error_message = "rocm-smi found but no GPU detected"
            return False, gpu_info, error_message

    except FileNotFoundError:
        error_message = "rocm-smi not found (ROCm may not be installed)"
        return False, gpu_info, error_message
    except subprocess.TimeoutExpired:
        error_message = "rocm-smi timeout (drivers may not be responding)"
        return False, gpu_info, error_message
    except Exception as e:
        error_message = f"Error checking ROCm GPU: {str(e)}"
        return False, gpu_info, error_message


def detect_mps_gpu():
    """
    Detect if MPS (Apple Silicon) GPU is available.
    Returns tuple: (is_available: bool, gpu_info: dict, error_message: str)
    """
    gpu_info = {}
    error_message = None

    # Check if running on macOS with Apple Silicon
    if platform.system() != "Darwin":
        error_message = "MPS is only available on macOS"
        return False, gpu_info, error_message

    # Check if it's Apple Silicon (arm64)
    machine = platform.machine().lower()
    if machine not in ["arm64", "aarch64"]:
        error_message = "MPS requires Apple Silicon (arm64)"
        return False, gpu_info, error_message

    # Try to get GPU info from system_profiler
    try:
        import subprocess

        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            timeout=5,
            text=True,
        )

        gpu_info = {
            "name": "Apple Silicon GPU",
            "driver_version": "MPS",
            "memory_total_mb": 0,  # Shared memory, not easily detectable
            "count": 1,
        }

        if result.returncode == 0 and result.stdout.strip():
            # Try to extract GPU name
            for line in result.stdout.split("\n"):
                if "Chipset Model" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        gpu_info["name"] = parts[1].strip()
                        break

        return True, gpu_info, None

    except Exception:
        # Even if we can't get detailed info, MPS might still be available
        gpu_info = {
            "name": "Apple Silicon GPU",
            "driver_version": "MPS",
            "memory_total_mb": 0,
            "count": 1,
        }
        return True, gpu_info, None


def detect_gpu_backends():
    """
    Detect all available GPU backends (NVIDIA, ROCm, MPS).
    Returns: dict with {backend_name: (available, info_dict, error_msg)}
    """
    backends = {}

    # NVIDIA/CUDA
    nvidia_available, nvidia_info, nvidia_error = detect_nvidia_gpu()
    backends["nvidia"] = (nvidia_available, nvidia_info, nvidia_error)

    # ROCm/AMD
    rocm_available, rocm_info, rocm_error = detect_rocm_gpu()
    backends["rocm"] = (rocm_available, rocm_info, rocm_error)

    # MPS/Apple Silicon
    mps_available, mps_info, mps_error = detect_mps_gpu()
    backends["mps"] = (mps_available, mps_info, mps_error)

    return backends


def test_mediapipe_gpu_delegate(backend="nvidia"):
    """
    Test if MediaPipe can use GPU delegate for a specific backend.
    Args:
        backend: "nvidia", "rocm", "mps", or "cpu"
    Returns tuple: (works: bool, error_message: str)
    """
    try:
        import mediapipe as mp

        base_options_cls = mp.tasks.BaseOptions
        pose_landmarker_cls = mp.tasks.vision.PoseLandmarker
        pose_landmarker_options_cls = mp.tasks.vision.PoseLandmarkerOptions
        vision_running_mode_cls = mp.tasks.vision.RunningMode

        # Check if GPU delegate is available
        if not hasattr(base_options_cls.Delegate, "GPU"):
            return False, "MediaPipe GPU delegate not available in this version"

        # For CPU, always return True (no test needed)
        if backend == "cpu":
            return True, None

        # Try to get a model path (use lite for quick test)
        try:
            model_path = get_model_path(0)  # Lite model for testing
        except Exception as e:
            return False, f"Could not get model for testing: {str(e)}"

        # Try to create options with GPU delegate
        try:
            options = pose_landmarker_options_cls(
                base_options=base_options_cls(
                    model_asset_path=model_path, delegate=base_options_cls.Delegate.GPU
                ),
                running_mode=vision_running_mode_cls.IMAGE,  # Use IMAGE mode for quick test
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )

            # Try to create landmarker (this will fail if GPU is not available)
            with pose_landmarker_cls.create_from_options(options) as landmarker:
                # Create a dummy test image
                test_image = np.zeros((100, 100, 3), dtype=np.uint8)
                rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

                # Try to detect (this will fail if GPU delegate doesn't work)
                landmarker.detect(mp_image)
                return True, None

        except Exception as e:
            error_msg = str(e)
            if "GPU" in error_msg or "delegate" in error_msg.lower() or "CUDA" in error_msg or "ROCm" in error_msg or "MPS" in error_msg:
                return False, f"GPU delegate test failed for {backend}: {error_msg}"
            else:
                # Might be a different error, but GPU delegate was created
                return True, None

    except ImportError as e:
        return False, f"MediaPipe import error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error testing GPU ({backend}): {str(e)}"


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


def should_use_batch_processing(video_path, pose_config):
    # Activate when:
    # - Linux system
    # - Resolution > 2.7K (2700px) OR > 1000 frames
    # - Memory < 4GB available OR > 80% used
    if not is_linux_system():
        return False

    # Check video resolution and frame count
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # High resolution threshold (2.7K = 2700 pixels width)
    is_high_res = width > 2700 or height > 2700

    # High frame count threshold (> 1000 frames)
    is_long_video = total_frames > 1000

    # Check available memory
    memory_info = get_system_memory_info()
    low_memory = False
    if memory_info:
        # If less than 4GB available or more than 80% used
        low_memory = memory_info["available_gb"] < 4.0 or memory_info["percent_used"] > 80

    should_use_batch = is_high_res or low_memory or is_long_video

    if should_use_batch:
        print("Batch processing enabled:")
        print(f"   - Resolution: {width}x{height}")
        print(f"   - Total frames: {total_frames}")
        if memory_info:
            print(
                f"   - Memory: {memory_info['available_gb']:.1f}GB available, {memory_info['percent_used']:.1f}% used"
            )
        print(f"   - High resolution: {is_high_res}")
        print(f"   - Long video: {is_long_video}")
        print(f"   - Low memory: {low_memory}")

    return should_use_batch


def calculate_batch_size(video_path, pose_config):
    # 4K+: 20 frames/batch (reduced for stability)
    # 2.7K-4K: 30 frames/batch
    # 1080p-2.7K: 50 frames/batch
    # <1080p: 100 frames/batch
    # Long videos (>1000 frames): further reduction
    # Ajusta baseado na memória disponível
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 50  # Conservative default fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Base batch size on resolution
    pixels_per_frame = width * height

    if pixels_per_frame > 8_000_000:  # 4K+
        batch_size = 20  # Reduced for stability
    elif pixels_per_frame > 4_000_000:  # 2.7K-4K
        batch_size = 30  # Reduced from 50
    elif pixels_per_frame > 2_000_000:  # 1080p-2.7K
        batch_size = 50  # Reduced from 100
    else:
        batch_size = 100  # Reduced from 200

    # Further reduction for long videos (>1000 frames)
    if total_frames > 5000:  # Very long videos
        batch_size = max(10, batch_size // 3)
    elif total_frames > 2000:  # Long videos
        batch_size = max(15, batch_size // 2)
    elif total_frames > 1000:  # Medium-long videos
        batch_size = max(20, int(batch_size * 0.7))

    # Adjust based on available memory
    memory_info = get_system_memory_info()
    if memory_info:
        available_gb = memory_info["available_gb"]
        if available_gb < 2.0:
            batch_size = max(10, batch_size // 2)
        elif available_gb < 4.0:
            batch_size = max(15, int(batch_size * 0.75))
        elif available_gb < 6.0:
            batch_size = max(20, int(batch_size * 0.85))

    # Ensure batch size doesn't exceed total frames
    batch_size = min(batch_size, total_frames)

    print("Batch processing configuration:")
    print(f"   - Resolution: {width}x{height} ({pixels_per_frame:,} pixels/frame)")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Batch size: {batch_size} frames")
    print(f"   - Estimated batches: {(total_frames + batch_size - 1) // batch_size}")
    if memory_info:
        print(f"   - Available memory: {memory_info['available_gb']:.1f}GB")

    return batch_size


def get_bbox_config_for_frame(
    frame_idx,
    pose_config,
    original_width,
    original_height,
    process_width=None,
    process_height=None,
    enable_resize=False,
    resize_scale=1.0
):
    """
    Get the bounding box configuration for a specific frame.
    
    Iterates through bounding_box_ranges (sorted by specificity) and returns the 
    config for the first matching range. Last match wins due to sorting.
    
    Args:
        frame_idx: Current frame index
        pose_config: Full pose configuration dictionary
        original_width: Original video width
        original_height: Original video height
        process_width: Processing width (if video was resized)
        process_height: Processing height (if video was resized)
        enable_resize: Whether video was resized
        resize_scale: Scale factor used for resizing
    
    Returns:
        dict: Bounding box configuration for this frame
    """
    if process_width is None:
        process_width = original_width
    if process_height is None:
        process_height = original_height
    
    # Default config (full frame, no crop)
    default_config = {
        "enable_crop": pose_config.get("enable_crop", False),
        "bbox_x_min": pose_config.get("bbox_x_min", 0),
        "bbox_y_min": pose_config.get("bbox_y_min", 0),
        "bbox_x_max": pose_config.get("bbox_x_max", process_width),
        "bbox_y_max": pose_config.get("bbox_y_max", process_height),
        "enable_resize_crop": pose_config.get("enable_resize_crop", False),
        "resize_crop_scale": pose_config.get("resize_crop_scale", 1),
        "roi_polygon_points": pose_config.get("roi_polygon_points"),
        "exclusion_rois": [],
        "video_resized": enable_resize,
        "resize_scale": resize_scale,
    }
    
    # Check if we have multiple ROI ranges
    bounding_box_ranges = pose_config.get("bounding_box_ranges", [])
    if not bounding_box_ranges:
        return default_config
    
    # Find matching ranges for this frame
    # Note: Ranges should be sorted General -> Specific (last match wins)
    matched_config = default_config.copy()
    exclusion_rois = []
    
    for roi_range in bounding_box_ranges:
        frame_start = roi_range.get("frame_start", 0)
        frame_end = roi_range.get("frame_end", float("inf"))
        
        # Check if frame is within range
        if frame_start <= frame_idx <= frame_end:
            roi_type = roi_range.get("roi_type", "inclusion")
            
            if roi_type == "exclusion":
                # Add to exclusion list
                exclusion_rois.append({
                    "bbox_x_min": roi_range.get("bbox_x_min", 0),
                    "bbox_y_min": roi_range.get("bbox_y_min", 0),
                    "bbox_x_max": roi_range.get("bbox_x_max", process_width),
                    "bbox_y_max": roi_range.get("bbox_y_max", process_height),
                    "roi_polygon_points": roi_range.get("roi_polygon_points"),
                })
            else:
                # Inclusion ROI - update the main config (last match wins)
                matched_config = {
                    "enable_crop": roi_range.get("enable_crop", True),
                    "bbox_x_min": roi_range.get("bbox_x_min", 0),
                    "bbox_y_min": roi_range.get("bbox_y_min", 0),
                    "bbox_x_max": roi_range.get("bbox_x_max", process_width),
                    "bbox_y_max": roi_range.get("bbox_y_max", process_height),
                    "enable_resize_crop": roi_range.get("enable_resize_crop", False),
                    "resize_crop_scale": roi_range.get("resize_crop_scale", 1),
                    "roi_polygon_points": roi_range.get("roi_polygon_points"),
                    "exclusion_rois": [],  # Will be populated below
                    "video_resized": enable_resize,
                    "resize_scale": resize_scale,
                }
    
    # Attach all matching exclusion ROIs
    matched_config["exclusion_rois"] = exclusion_rois
    
    return matched_config


def process_video_batch(
    frames,
    landmarker,
    pose_config,
    width,
    height,
    full_width=None,
    full_height=None,
    progress_callback=None,
    batch_index=0,
    fps=30.0,
    start_frame_idx=0,
):
    """
    Process a batch of frames using MediaPipe Tasks API and return landmarks with CPU throttling.
    Supports bounding box cropping and coordinate mapping.
    """
    # Use full dimensions if provided, otherwise use processing dimensions
    if full_width is None:
        full_width = width
    if full_height is None:
        full_height = height

    # Derive resize status from dimensions to ensure correctness even if config is stale
    derived_scale = width / full_width if full_width and full_width > 0 else 1.0
    derived_enable_resize = abs(derived_scale - 1.0) > 0.01

    batch_landmarks = []
    batch_pixel_landmarks = []
    landmarks_history = deque(maxlen=10)

    for i, frame in enumerate(frames):
        # CPU throttling check
        frame_global_index = batch_index * len(frames) + i
        if should_throttle_cpu(frame_global_index):
            apply_cpu_throttling()

        if progress_callback and i % 10 == 0:
            progress_callback(f"Processing frame {i + 1}/{len(frames)} in batch")

        # Calculate timestamp (must be monotonically increasing)
        global_frame_idx = start_frame_idx + i
        timestamp_ms = int((global_frame_idx * 1000) / fps) if fps > 0 else global_frame_idx * 33

        # Get dynamic bounding box configuration for this frame
        bbox_config = get_bbox_config_for_frame(
            global_frame_idx,
            pose_config,
            full_width,  # original_width
            full_height, # original_height
            width,       # processing_width
            height,      # processing_height
            derived_enable_resize,
            derived_scale
        )

        enable_crop = (
            bbox_config.get("bbox_x_min", 0) != 0 or
            bbox_config.get("bbox_y_min", 0) != 0 or
            bbox_config.get("bbox_x_max", width) != width or
            bbox_config.get("bbox_y_max", height) != height
        ) or pose_config.get("enable_crop", False)

        # Process frame with Tasks API
        landmarks = process_frame_with_tasks_api(
            frame,
            landmarker,
            timestamp_ms,
            enable_crop,
            bbox_config,
            width,
            height,
            full_width,
            full_height,
            pose_config,
            landmarks_history,
        )

        if landmarks:
            landmarks_history.append(landmarks)
            batch_landmarks.append(landmarks)

            # Convert to pixel coordinates (using full frame dimensions)
            pixel_landmarks = [
                [int(lm[0] * full_width), int(lm[1] * full_height), lm[2]] for lm in landmarks
            ]
            batch_pixel_landmarks.append(pixel_landmarks)
        else:
            # No pose detected
            num_landmarks = len(landmark_names)
            nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
            batch_landmarks.append(nan_landmarks)
            batch_pixel_landmarks.append(nan_landmarks)

        # Small sleep between frames to prevent CPU overload
        time.sleep(FRAME_SLEEP_TIME)

    return batch_landmarks, batch_pixel_landmarks


def cleanup_memory():
    """Aggressive memory cleanup for Linux systems"""
    # Force Python garbage collection multiple times
    for _ in range(3):
        gc.collect()

    if is_linux_system():
        # Linux-specific memory cleanup
        try:
            import ctypes

            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)  # Return freed memory to OS
        except Exception as e:
            print(f"Warning: Could not perform malloc_trim: {e}")

        # Additional memory pressure relief
        try:
            # Sync and drop caches if possible (requires privileges)
            import subprocess

            subprocess.run(["sync"], check=False, capture_output=True)
        except Exception:
            pass

    # Force memory usage report
    memory_info = get_system_memory_info()
    if memory_info:
        print(
            f"Memory cleanup: {memory_info['available_gb']:.1f}GB available, {memory_info['percent_used']:.1f}% used"
        )


def get_model_path(complexity):
    """Download the correct model based on complexity (0=Lite, 1=Full, 2=Heavy)"""
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
            raise RuntimeError("Failed to download MediaPipe model.") from e
    return str(model_path.resolve())



def process_frame_with_tasks_api(
    frame,
    landmarker,
    timestamp_ms,
    enable_crop,
    bbox_config,
    process_width,
    process_height,
    original_width,
    original_height,
    pose_config,
    landmarks_history,
):
    """
    Process a single frame using MediaPipe Tasks API.
    Returns landmarks in normalized coordinates (mapped to full frame if cropping was used).
    Supports both bounding box and polygon ROI, including exclusion ROIs.
    """
    # First, apply exclusion masks if any exclusion ROIs are defined
    exclusion_rois = bbox_config.get("exclusion_rois", [])
    if exclusion_rois:
        # Create a mask that excludes all exclusion ROIs
        exclusion_mask = np.ones((process_height, process_width), dtype=np.uint8) * 255

        for exclusion_roi in exclusion_rois:
            roi_polygon = exclusion_roi.get("roi_polygon_points")

            if roi_polygon and len(roi_polygon) >= 3:
                # Polygon exclusion ROI
                if bbox_config.get("video_resized", False):
                    resize_scale = bbox_config.get("resize_scale", 1.0)
                    scaled_polygon = [
                        [int(pt[0] * resize_scale), int(pt[1] * resize_scale)]
                        for pt in roi_polygon
                    ]
                else:
                    scaled_polygon = [[int(pt[0]), int(pt[1])] for pt in roi_polygon]

                polygon_pts = np.array(scaled_polygon, dtype=np.int32)
                # Fill exclusion area with black (0) in mask
                cv2.fillPoly(exclusion_mask, [polygon_pts], 0)
            else:
                # Bounding box exclusion ROI
                x_min = max(0, min(exclusion_roi.get("bbox_x_min", 0), process_width - 1))
                y_min = max(0, min(exclusion_roi.get("bbox_y_min", 0), process_height - 1))
                x_max = max(x_min + 1, min(exclusion_roi.get("bbox_x_max", process_width), process_width))
                y_max = max(y_min + 1, min(exclusion_roi.get("bbox_y_max", process_height), process_height))
                # Fill exclusion area with black (0) in mask
                exclusion_mask[y_min:y_max, x_min:x_max] = 0

        # Apply exclusion mask to frame
        frame = cv2.bitwise_and(frame, frame, mask=exclusion_mask)

    # Apply bounding box or polygon cropping if enabled (inclusion ROI)
    process_frame = frame
    actual_process_width = process_width
    actual_process_height = process_height

    if enable_crop:
        # Check if polygon ROI is available
        roi_polygon_points = bbox_config.get("roi_polygon_points")

        if roi_polygon_points and len(roi_polygon_points) >= 3:
            # Polygon ROI: create mask and apply it
            # Scale polygon points if video was resized
            if bbox_config.get("video_resized", False):
                resize_scale = bbox_config.get("resize_scale", 1.0)
                scaled_polygon = [
                    [int(pt[0] * resize_scale), int(pt[1] * resize_scale)]
                    for pt in roi_polygon_points
                ]
            else:
                scaled_polygon = [[int(pt[0]), int(pt[1])] for pt in roi_polygon_points]

            # Create mask for polygon ROI
            mask = np.zeros((process_height, process_width), dtype=np.uint8)
            polygon_pts = np.array(scaled_polygon, dtype=np.int32)
            cv2.fillPoly(mask, [polygon_pts], 255)

            # Apply mask to frame
            process_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Get bounding box of polygon for actual_process dimensions
            x_coords = [pt[0] for pt in scaled_polygon]
            y_coords = [pt[1] for pt in scaled_polygon]
            x_min = max(0, min(x_coords))
            y_min = max(0, min(y_coords))
            x_max = min(process_width, max(x_coords))
            y_max = min(process_height, max(y_coords))

            # Crop to bounding box for processing efficiency
            process_frame = process_frame[y_min:y_max, x_min:x_max]
            actual_process_width = x_max - x_min
            actual_process_height = y_max - y_min

            # Store polygon offset for coordinate mapping
            bbox_config["polygon_offset_x"] = x_min
            bbox_config["polygon_offset_y"] = y_min
        else:
            # Bounding box ROI (original behavior)
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

    # Use Standard Detection
    pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    if pose_landmarker_result.pose_landmarks and len(pose_landmarker_result.pose_landmarks) > 0:
        raw_landmarks = pose_landmarker_result.pose_landmarks[0]
        pose_landmarks_found = True
    else:
        raw_landmarks = []
        pose_landmarks_found = False

    # Parse results
    if pose_landmarks_found:
        landmarks = [[lm.x, lm.y, lm.z] for lm in raw_landmarks]

        # Map coordinates to full frame if cropping was used
        if enable_crop:

            # For polygon ROI, adjust coordinates for the offset
            if bbox_config.get("roi_polygon_points") and len(bbox_config.get("roi_polygon_points", [])) >= 3:
                # Polygon ROI: landmarks are in cropped bounding box coordinates (normalized 0-1)
                # Need to map back to full frame coordinates
                offset_x = bbox_config.get("polygon_offset_x", 0)
                offset_y = bbox_config.get("polygon_offset_y", 0)
                enable_resize_crop = bbox_config.get("enable_resize_crop", False)
                resize_crop_scale = bbox_config.get("resize_crop_scale", 1)

                # Step 1: Convert from normalized crop space (0-1) to pixel crop space
                # If resize_crop is enabled, landmarks are in resized crop space
                # Step 2: If resize_crop enabled, convert from resized crop to original crop
                # Step 3: Add offset to get coordinates in full processing frame
                # Step 4: If video was resized, convert to original dimensions
                # Step 5: Normalize to original video dimensions
                for lm in landmarks:
                    # Convert normalized crop to pixel crop (may be resized crop)
                    x_px_crop_resized = lm[0] * actual_process_width
                    y_px_crop_resized = lm[1] * actual_process_height

                    # If resize_crop is enabled, convert from resized crop to original crop
                    if enable_resize_crop and resize_crop_scale > 1:
                        x_px_crop = x_px_crop_resized / resize_crop_scale
                        y_px_crop = y_px_crop_resized / resize_crop_scale
                    else:
                        x_px_crop = x_px_crop_resized
                        y_px_crop = y_px_crop_resized

                    # Add offset to get coordinates in full processing frame
                    x_px_full_processing = x_px_crop + offset_x
                    y_px_full_processing = y_px_crop + offset_y

                    # If video was resized, convert from processing to original dimensions
                    if bbox_config.get("video_resized", False):
                        resize_scale = bbox_config.get("resize_scale", 1.0)
                        x_px_full_original = x_px_full_processing / resize_scale
                        y_px_full_original = y_px_full_processing / resize_scale
                    else:
                        x_px_full_original = x_px_full_processing
                        y_px_full_original = y_px_full_processing

                    # Normalize to original video dimensions
                    lm[0] = x_px_full_original / original_width if original_width > 0 else 0.0
                    lm[1] = y_px_full_original / original_height if original_height > 0 else 0.0
            else:
                # Bounding box ROI: use existing mapping function
                landmarks = map_landmarks_to_full_frame(
                    landmarks,
                    bbox_config,
                    actual_process_width,
                    actual_process_height,
                    process_width,
                    process_height,  # Processing frame dimensions (may be resized)
                    original_width,
                    original_height,  # Original video dimensions
                )

        # Apply occluded landmark estimation if enabled
        if pose_config.get("estimate_occluded", False):
            landmarks = estimate_occluded_landmarks(landmarks, list(landmarks_history))

        return landmarks
    else:
        # No pose detected
        return None

def process_video(video_path, output_dir, pose_config, use_gpu=False, gpu_backend=None):
    """
    Process a video file using MediaPipe Pose estimation with optional video resize.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save output files
        pose_config: Configuration dictionary with MediaPipe parameters
        use_gpu: Whether to use GPU acceleration (default: False)
        gpu_backend: GPU backend to use ("nvidia", "rocm", "mps") if use_gpu is True
    """
    # #region agent log
    try:
        import mediapipe as mp_test

        _debug_log(
            "E",
            "process_video:2475",
            "Function entry",
            {
                "video_path": str(video_path),
                "output_dir": str(output_dir),
                "has_mp_solutions": hasattr(mp_test, "solutions"),
                "mp_test_type": str(type(mp_test)),
            },
        )
    except Exception:
        # Log function itself may fail, but continue
        pass
    # #endregion
    print("\n=== Parameters being used for this video ===")
    for k, v in pose_config.items():
        print(f"{k}: {v}")
    print("==========================================\n")

    # --- REGRESSION FIX 1: Sort ROI ranges by specificity ---
    # We want specific (short duration) ranges to override general (long duration) ones.
    # Since "Last Match Wins" in get_bbox_config_for_frame, we sort so that:
    # - Long ranges (e.g. 0-inf) come FIRST
    # - Short ranges (e.g. 0-300) come LAST
    if "bounding_box_ranges" in pose_config and isinstance(pose_config["bounding_box_ranges"], list):
        print("Optimizing ROI Range priority...")
        def get_range_duration(r):
            start = r.get("frame_start", 0)
            end = r.get("frame_end", float('inf'))
            if isinstance(end, str) and end.lower() == "inf":
                end = float('inf')
            return end - start
        
        # Sort descending by duration: Longest (Inf) -> Shortest (Specific)
        # This ensures Specific overrides General in the loop.
        pose_config["bounding_box_ranges"].sort(key=get_range_duration, reverse=True)
        print("ROI Ranges sorted by specificity (General -> Specific).")

    print(f"Processing video: {video_path}")
    start_time = time.time()

    # Check if resize is enabled
    enable_resize = pose_config.get("enable_resize", False)
    resize_scale = pose_config.get("resize_scale", 2)

    # Get original video dimensions (before any resizing)
    orig_cap = cv2.VideoCapture(str(video_path))
    if not orig_cap.isOpened():
        print(f"Failed to open original video: {video_path}")
        return
    original_width = int(orig_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(orig_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_cap.release()

    # Prepare directories and output files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Video to process (original or resized)
    processing_video_path = video_path
    resize_metadata = None
    temp_resized_video = None

    # Step 1: Resize video if enabled
    if enable_resize and resize_scale > 1:
        print(f"Step 1/3: Resizing video by {resize_scale}x for better pose detection")

        # Create temporary file for resized video
        temp_dir = tempfile.mkdtemp()
        temp_resized_video = os.path.join(temp_dir, f"temp_resized_{resize_scale}x.mp4")

        # Resize the video
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

    # Initial configuration
    cap = cv2.VideoCapture(str(processing_video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {processing_video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output files
    output_video_path = output_dir / f"{video_path.stem}_mp.mp4"
    output_file_path = output_dir / f"{video_path.stem}_mp_norm.csv"
    output_pixel_file_path = output_dir / f"{video_path.stem}_mp_pixel.csv"

    # Initialize MediaPipe Tasks API
    # #region agent log
    from contextlib import suppress

    with suppress(Exception):
        _debug_log("A", "process_video:2610", "Initializing MediaPipe Tasks API", {})
    # #endregion

    model_path = get_model_path(pose_config["model_complexity"])

    BaseOptions = mp.tasks.BaseOptions  # noqa: N806 - MediaPipe class name
    PoseLandmarker = mp.tasks.vision.PoseLandmarker  # noqa: N806 - MediaPipe class name
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions  # noqa: N806 - MediaPipe class name
    VisionRunningMode = mp.tasks.vision.RunningMode  # noqa: N806 - MediaPipe class name

    # Create options with CPU or GPU Delegate
    delegate = BaseOptions.Delegate.CPU # Default to CPU
    if use_gpu and gpu_backend:
        try:
            # Try GPU delegate
            delegate = BaseOptions.Delegate.GPU
            backend_name = gpu_backend.upper()
            print(f"Using GPU ({backend_name}) for processing")
        except Exception as e:
            print(f"Warning: Could not use GPU delegate ({gpu_backend}): {e}")
            print("Falling back to CPU")
            delegate = BaseOptions.Delegate.CPU
            use_gpu = False
            gpu_backend = None
            
    # Create landmarker options
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=delegate
        ),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=pose_config.get("min_detection_confidence", 0.5),
        min_pose_presence_confidence=pose_config.get("min_detection_confidence", 0.5),
        min_tracking_confidence=pose_config.get("min_tracking_confidence", 0.5),
        num_poses=1,
    )

    # Setup Video Writer for annotated output
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    annotated_output_path = output_dir / f"{video_path.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(annotated_output_path), fourcc, fps, (width, height))

    # Initialize results containers
    normalized_landmarks_list = []
    pixel_landmarks_list = []
    last_frames_for_padding = deque(maxlen=30) 

    print(f"Starting analysis loop... (Total frames: {total_frames})")

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_count = 0
        landmarks_history = deque(maxlen=30)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Throttle CPU if needed
            if should_throttle_cpu(frame_count):
                apply_cpu_throttling()

            timestamp_ms = int((frame_count * 1000) / fps)
            
            # Determine ROI for this frame
            bbox_config = get_bbox_config_for_frame(frame_count, pose_config, width, height)
            
            # Show ROI details periodically (every 500 frames or at key frames)
            if frame_count == 0 or frame_count % 500 == 0:
                if bbox_config.get("enable_crop", False):
                    roi_info = f"ROI: ({bbox_config.get('bbox_x_min', 0)},{bbox_config.get('bbox_y_min', 0)}) -> ({bbox_config.get('bbox_x_max', width)},{bbox_config.get('bbox_y_max', height)})"
                    scale_info = f"Scale: {bbox_config.get('resize_crop_scale', 1)}x" if bbox_config.get('enable_resize_crop', False) else "No Scale"
                    print(f"  [Frame {frame_count}] {roi_info}, {scale_info}")
                else:
                    print(f"  [Frame {frame_count}] Using full frame (no ROI)")
            
            # --- PROCESS FRAME ---
            # Returns landmarks in normalized coordinates (w.r.t Original Video Size)
            landmarks = process_frame_with_tasks_api(
                frame,
                landmarker,
                timestamp_ms,
                bbox_config.get("enable_crop", False),
                bbox_config,
                width, # Process width
                height, # Process height
                original_width,
                original_height,
                pose_config,
                landmarks_history
            )
            
            # Draw on frame (Visualization)
            annotated_frame = frame.copy()
            
            if landmarks:
                normalized_landmarks_list.append(landmarks)
                
                # Convert to pixel for internal usage/drawing
                pixel_landmarks = [
                     [lm[0] * original_width, lm[1] * original_height, lm[2]] 
                     for lm in landmarks
                ]
                pixel_landmarks_list.append(pixel_landmarks)
                
                # Draw landmarks (Simple circle drawing for speed/robustness)
                for i, lm in enumerate(landmarks):
                     # Map normalized back to current frame dimensions for drawing
                     px = int(lm[0] * width)
                     py = int(lm[1] * height)
                     cv2.circle(annotated_frame, (px, py), 4, (0, 255, 0), -1)
                
                # Draw skeleton connections
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_lm = landmarks[start_idx]
                        end_lm = landmarks[end_idx]
                        # Skip if any coordinate is NaN
                        if not (np.isnan(start_lm[0]) or np.isnan(start_lm[1]) or 
                                np.isnan(end_lm[0]) or np.isnan(end_lm[1])):
                            start_pt = (int(start_lm[0] * width), int(start_lm[1] * height))
                            end_pt = (int(end_lm[0] * width), int(end_lm[1] * height))
                            cv2.line(annotated_frame, start_pt, end_pt, (255, 0, 0), 2)
                     
                # Draw ROI box if crop enabled
                if bbox_config.get("enable_crop", False):
                     x_min = bbox_config.get("bbox_x_min", 0)
                     y_min = bbox_config.get("bbox_y_min", 0)
                     x_max = bbox_config.get("bbox_x_max", width)
                     y_max = bbox_config.get("bbox_y_max", height)
                     cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                     
            else:
                 # NaNs for missing frame
                 num_landmarks = 33
                 nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
                 normalized_landmarks_list.append(nan_landmarks)
                 pixel_landmarks_list.append(nan_landmarks)
            
            out_video.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    print("\nAnalysis loop completed.")

    # --- POST PROCESSING ---
    
    # Define columns
    landmark_names = [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot_index", "right_foot_index"
    ]
    columns = ["frame_index"]
    for name in landmark_names:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    # Helper to convert list to array format for DataFrame
    def create_df(landmarks_list):
        data = []
        for idx, frame_landmarks in enumerate(landmarks_list):
            row = [idx]
            for lm in frame_landmarks:
                row.extend(lm)
            data.append(row)
        return pd.DataFrame(data, columns=columns)

    print("Converting results to DataFrame...")
    df_norm = create_df(normalized_landmarks_list)
    df_pixel = create_df(pixel_landmarks_list)
    
    # 2. Apply Median Filter (if enabled)
    # We only filter the coordinate columns (skip frame_index)
    coord_cols = df_norm.columns[1:] 
    
    if pose_config.get("enable_median_filter", False):
        kernel = pose_config.get("median_kernel_size", 5)
        print(f"Applying Median Filter (Kernel: {kernel})...")
        
        # Filter Normalized
        norm_values = df_norm[coord_cols].values
        filtered_norm = filter_pose_data(norm_values, enable_median=True, enable_butterworth=False, median_kernel=kernel)
        df_norm[coord_cols] = filtered_norm
        
        # Filter Pixel
        pixel_values = df_pixel[coord_cols].values
        filtered_pixel = filter_pose_data(pixel_values, enable_median=True, enable_butterworth=False, median_kernel=kernel)
        df_pixel[coord_cols] = filtered_pixel
        print("Median filtering complete.")

    # 3. Save CSVs
    print(f"Saving processed CSVs to {output_dir}")
    df_norm.to_csv(output_file_path, index=False, float_format="%.6f")
    
    # Save legacy/pixel format
    convert_mediapipe_to_vaila_format(df_pixel, output_dir / f"{video_path.stem}_pixel_vaila.csv")
    
    # Also save the raw MP pixel format if needed
    df_pixel.to_csv(output_pixel_file_path, index=False, float_format="%.1f")
    
    # 4. Save configuration used
    config_used_path = output_dir / "configuration_used.toml"
    try:
        # Add total frames to config for future reference
        pose_config["video_total_frames"] = int(total_frames)
        save_config_to_toml(pose_config, config_used_path)
        print(f"Configuration saved to: {config_used_path}")
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")
    
    # 5. Generate log_info.txt
    end_time = time.time()
    processing_time = end_time - start_time
    log_path = output_dir / "log_info.txt"
    try:
        with open(log_path, "w") as f:
            f.write(f"vailá Markerless 2D Analysis Log\n")
            f.write(f"================================\n\n")
            f.write(f"Video: {video_path.name}\n")
            f.write(f"Processing Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {processing_time:.2f} seconds\n\n")
            f.write(f"Video Info:\n")
            f.write(f"  - Original Resolution: {original_width}x{original_height}\n")
            f.write(f"  - Processing Resolution: {width}x{height}\n")
            f.write(f"  - Total Frames: {total_frames}\n")
            f.write(f"  - FPS: {fps:.2f}\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Model Complexity: {pose_config.get('model_complexity', 2)}\n")
            f.write(f"  - Detection Confidence: {pose_config.get('min_detection_confidence', 0.5)}\n")
            f.write(f"  - Tracking Confidence: {pose_config.get('min_tracking_confidence', 0.5)}\n")
            f.write(f"  - Enable Resize: {pose_config.get('enable_resize', False)}\n")
            f.write(f"  - Enable Median Filter: {pose_config.get('enable_median_filter', False)}\n")
            if pose_config.get('enable_median_filter', False):
                f.write(f"  - Median Kernel Size: {pose_config.get('median_kernel_size', 5)}\n")
            f.write(f"  - Enable Crop/ROI: {pose_config.get('enable_crop', False)}\n")
            if pose_config.get('bounding_box_ranges'):
                f.write(f"  - Multiple ROI Ranges: {len(pose_config.get('bounding_box_ranges', []))}\n")
            f.write(f"\nOutput Files:\n")
            f.write(f"  - Annotated Video: {video_path.stem}_annotated.mp4\n")
            f.write(f"  - Normalized CSV: {video_path.stem}_mp_norm.csv\n")
            f.write(f"  - Pixel CSV: {video_path.stem}_mp_pixel.csv\n")
            f.write(f"  - vailá Format CSV: {video_path.stem}_pixel_vaila.csv\n")
            f.write(f"  - Configuration: configuration_used.toml\n")
        print(f"Log saved to: {log_path}")
    except Exception as e:
        print(f"Warning: Could not save log: {e}")
    
    print(f"\nProcessing complete! Duration: {processing_time:.2f}s")
def process_videos_in_directory(existing_root=None):
    """
    Process all video files in the selected directory for markerless 2D analysis.

    This function analyzes video files to extract joint positions and calculate
    movement patterns using computer vision techniques.

    Args:
        existing_root: Optional existing Tkinter root window to use for dialogs.
                      If None, creates a new root window.

    Returns:
        None: Saves results to CSV files and generates plots

    Raises:
        FileNotFoundError: If no video files are found
        ValueError: If video files are corrupted
    """
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

    # Detect GPU backends availability
    print("\n=== Detecting GPU Backends Availability ===")
    backends = detect_gpu_backends()

    # Test each available backend
    available_backends = {}
    for backend_name in ["nvidia", "rocm", "mps"]:
        if backend_name in backends:
            available, info, error = backends[backend_name]
            if available:
                print(f"\nTesting {backend_name.upper()} backend...")
                test_result = test_mediapipe_gpu_delegate(backend_name)
                available_backends[backend_name] = (available, info, error)
                available_backends[f"{backend_name}_test"] = test_result
                if test_result[0]:
                    print(f"✓ {backend_name.upper()} backend test passed")
                else:
                    print(f"⚠ {backend_name.upper()} backend test failed: {test_result[1]}")
                    # Mark as unavailable if test fails
                    available_backends[backend_name] = (False, info, test_result[1])
            else:
                print(f"✗ {backend_name.upper()} not available: {error}")
                available_backends[backend_name] = (False, info, error)
                available_backends[f"{backend_name}_test"] = (False, error)

    print("=" * 40 + "\n")

    # Use existing root or create new one for dialogs
    if existing_root is not None:
        root = existing_root
    else:
        root = tk.Tk()
        root.withdraw()
        # Keep dialogs on top (as before)
        from contextlib import suppress

        with suppress(Exception):
            root.attributes("-topmost", True)

    # Show device selection dialog
    device_dialog = DeviceSelectionDialog(root, available_backends)
    selected_device = device_dialog.result if device_dialog.result else "cpu"

    use_gpu = selected_device in ["nvidia", "rocm", "mps"]
    gpu_backend = selected_device if use_gpu else None

    if use_gpu:
        print(f"✓ GPU processing selected ({selected_device.upper()})")
    else:
        print("✓ CPU processing selected")

    print()

    # Helper function to prepare root window for dialogs on macOS
    # Fixes issue where dialogs appear in wrong position (bottom corner) on macOS
    def prepare_root_for_dialog():
        if platform.system() == "Darwin":  # macOS
            root.deiconify()
            root.update_idletasks()
            # Position window in a visible location (small window at top-left)
            root.geometry("1x1+100+100")
            root.lift()
            root.update_idletasks()

    # Select input directory
    prepare_root_for_dialog()
    input_dir = filedialog.askdirectory(
        parent=root, title="Select the input directory containing videos"
    )
    if platform.system() == "Darwin" and existing_root is None:
        root.withdraw()  # Hide root window again after dialog closes
    if not input_dir:
        messagebox.showerror("Error", "No input directory selected.")
        return

    # Select output base directory
    prepare_root_for_dialog()
    output_base = filedialog.askdirectory(parent=root, title="Select the base output directory")
    if platform.system() == "Darwin" and existing_root is None:
        root.withdraw()  # Hide root window again after dialog closes
    if not output_base:
        messagebox.showerror("Error", "No output directory selected.")
        return

    # Pose configuration (GUI or TOML via dialog)
    pose_config = get_pose_config(root, input_dir=input_dir)
    if not pose_config:
        return

    # Timestamped output folder with descriptive suffix (as before)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix_parts = []
    if pose_config.get("enable_resize", False):
        suffix_parts.append(f"resize_{pose_config.get('resize_scale', 2)}x")
    if pose_config.get("enable_advanced_filtering", False):
        interp_method = pose_config.get("interp_method", "none")
        smooth_method = pose_config.get("smooth_method", "none")
        if smooth_method == "butterworth":
            params = pose_config.get("smooth_params", {})
            cutoff = params.get("cutoff", 10)
            fs = params.get("fs", 100)
            suffix_parts.append(f"filter_{interp_method}_{smooth_method}_c{cutoff}_fs{fs}")
        else:
            suffix_parts.append(f"filter_{interp_method}_{smooth_method}")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    output_base = Path(output_base) / f"mediapipe{suffix}_{timestamp}"
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
    if pose_config.get("enable_resize", False):
        print(f"Video resize enabled: {pose_config.get('resize_scale', 2)}x scaling")
    if pose_config.get("enable_advanced_filtering", False):
        print(
            f"Advanced filtering enabled: {pose_config.get('interp_method', 'none')} interpolation + "
            f"{pose_config.get('smooth_method', 'none')} smoothing"
        )

    # Process each video
    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file.name}")
        output_dir = output_base / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            process_video(video_file, output_dir, pose_config, use_gpu=use_gpu, gpu_backend=gpu_backend)
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
        finally:
            # Release memory between videos (as before)
            try:
                import gc as _gc

                _gc.collect()
            except Exception:
                pass
            time.sleep(2)
            print("Memory released")

    print("\nAll videos processed!")


def convert_mediapipe_to_vaila_format(df_pixel, output_path):
    """
    Convert MediaPipe format to vailá format (frame, p1_x, p1_y, p2_x, p2_y, ...)
    This mimics the convert_mediapipe_to_pixel_format function from rearrange_data.py
    """
    try:
        print("Converting to vailá format...")

        # Create the new DataFrame with the "frame" column and pX_x, pX_y coordinates
        new_df = pd.DataFrame()
        new_df["frame"] = df_pixel.iloc[:, 0]  # Use the first column as "frame"

        columns = df_pixel.columns[1:]  # Ignore the first column (frame_index)

        # Convert MediaPipe format (landmark_x, landmark_y, landmark_z) to vailá format (p1_x, p1_y, p2_x, p2_y)
        point_counter = 1
        for i in range(0, len(columns), 3):
            if i + 1 < len(columns):  # Ensure we have both x and y columns
                x_col = columns[i]  # landmark_x
                y_col = columns[i + 1]  # landmark_y
                # Note: We skip the z column (i + 2) as vailá format is 2D only

                new_df[f"p{point_counter}_x"] = df_pixel[x_col]
                new_df[f"p{point_counter}_y"] = df_pixel[y_col]
                point_counter += 1

        # Save the converted file
        new_df.to_csv(output_path, index=False, float_format="%.1f")
        print(f"vailá format CSV saved to: {output_path}")

        return True
    except Exception as e:
        print(f"Error converting to vailá format: {e}")
        return False


def get_cpu_usage():
    """Get current CPU usage percentage"""
    try:
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return 0


def should_throttle_cpu(frame_count):
    """Check if we should throttle CPU based on usage and frame count"""
    if frame_count % MAX_CPU_CHECK_INTERVAL == 0:  # Check every N frames
        cpu_usage = get_cpu_usage()
        if cpu_usage > CPU_USAGE_THRESHOLD:
            print(f"High CPU usage detected: {cpu_usage:.1f}% - Applying throttling")
            return True
    return False


def apply_cpu_throttling():
    """Apply CPU throttling by sleeping and reducing process priority"""
    time.sleep(FRAME_SLEEP_TIME * 2)  # Longer sleep for high CPU

    # Reduce process priority on Linux
    if is_linux_system():
        try:
            import os

            os.nice(5)  # Increase niceness (lower priority)
        except Exception:
            pass


if __name__ == "__main__":
    process_videos_in_directory()
