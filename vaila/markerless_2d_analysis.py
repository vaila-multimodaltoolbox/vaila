"""
Project: vailá Multimodal Toolbox
Script: markerless_2D_analysis.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 07 January 2026
Version: 0.7.2

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

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, model complexity, and
whether to enable segmentation and smooth segmentation. The default settings
prioritize the highest detection accuracy and tracking precision, which may
increase computational cost.

New Features (v0.7.2):
- Bounding box (ROI) selection for small subjects or multi-person scenarios with zoom and window resize capabilities

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the MediaPipe configuration parameters.
- Choose whether to enable video resize for better pose detection
- Optionally select a bounding box (ROI) from the first video frame for focused processing
- Optionally resize the cropped region for better detection of small subjects
- Load or save TOML configuration files for batch processing
- The script processes each video, generating an output video with overlaid pose
  landmarks, and CSV files containing both normalized and pixel-based landmark
  coordinates in original video dimensions.

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

import datetime
import gc
import os
import platform
import shutil
import tempfile
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import psutil
import json
import os
import time as _time_module
import urllib.request
import subprocess
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
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":hypothesis_id,"location":location,"message":message,"data":data or {},"timestamp":int(_time_module.time()*1000)})+"\n")
    except: pass
# #endregion

# --- NEW IMPORTS FOR THE TASKS API (MediaPipe 0.10.31+) ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MANUAL DEFINITION OF THE BODY CONNECTIONS (since mp.solutions was removed)
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23),
    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29),
    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])

# #region agent log
_debug_log("A", "import:100", "After importing mediapipe Tasks API", {"mp_type":str(type(mp)),"has_tasks":hasattr(mp,"tasks"),"mp_attrs":str([x for x in dir(mp) if not x.startswith("_")])[:500]})
# #endregion

# Additional imports for filtering and interpolation
from pykalman import KalmanFilter
from rich import print
from scipy.interpolate import UnivariateSpline
from scipy.signal import butter, savgol_filter, sosfiltfilt
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.arima.model import ARIMA

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
import multiprocessing  # noqa: F401 - For future Linux batch processing
import signal  # noqa: F401 - For future Linux process management
import threading  # noqa: F401 - For future Linux thread management

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
        "bounding_box": {
            "enable_crop": False,
            "bbox_x_min": 0,
            "bbox_y_min": 0,
            "bbox_x_max": 1920,
            "bbox_y_max": 1080,
            "enable_resize_crop": False,
            "resize_crop_scale": 2,
        },
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
                {
                    "savgol_window_length": 7,
                    "savgol_polyorder": 3,
                    "lowess_frac": 0.3,
                    "lowess_it": 3,
                    "butter_cutoff": 10.0,
                    "butter_fs": 100.0,
                    "kalman_iterations": 5,
                    "kalman_mode": 1,
                    "spline_smoothing_factor": 1.0,
                    "arima_p": 1,
                    "arima_d": 0,
                    "arima_q": 0,
                },
            ),
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
            },
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
            f.write("#                           # false = leave gaps when parts are hidden\n")

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

            f.write("\n[advanced_filtering]\n")
            f.write("# ================================================================\n")
            f.write("# ADVANCED FILTERING AND INTERPOLATION\n")
            f.write("# ================================================================\n")
            f.write("# Fill gaps and smooth the landmark data after detection\n")

            af = toml_config["advanced_filtering"]
            f.write(
                f"enable_advanced_filtering = {str(af['enable_advanced_filtering']).lower()}    # Use advanced processing (true/false)\n"
            )
            f.write(
                "#                                     # true = apply smoothing and gap filling\n"
            )
            f.write("#                                     # false = use raw MediaPipe output\n")

            f.write(
                f'interp_method = "{af["interp_method"]}"             # How to fill missing data\n'
            )
            f.write('#                         # "linear" = straight lines (most common)\n')
            f.write('#                         # "cubic" = curved lines (smoother)\n')
            f.write('#                         # "nearest" = copy nearest valid point\n')
            f.write('#                         # "kalman" = predictive filling\n')
            f.write('#                         # "none" = don\'t fill gaps\n')

            f.write(
                f'smooth_method = "{af["smooth_method"]}"               # Type of smoothing to apply\n'
            )
            f.write('#                       # "none" = no smoothing\n')
            f.write('#                       # "butterworth" = most common for biomechanics\n')
            f.write('#                       # "savgol" = preserves signal features\n')
            f.write('#                       # "lowess" = for very noisy data\n')
            f.write('#                       # "kalman" = for tracking applications\n')
            f.write('#                       # "splines" = very smooth curves\n')

            f.write(
                f"max_gap = {af['max_gap']}                         # Maximum gap size to fill (frames)\n"
            )
            f.write("#                         # 60 = fill gaps up to 2 seconds (at 30fps)\n")
            f.write("#                         # 30 = fill gaps up to 1 second (at 30fps)\n")
            f.write("#                         # 0 = fill all gaps regardless of size\n")

            f.write("\n[smoothing_params]\n")
            f.write("# ================================================================\n")
            f.write("# SMOOTHING PARAMETERS - Detailed Guide for Users\n")
            f.write("# ================================================================\n")
            f.write("# Only the parameters for your selected method will be used.\n")
            f.write("# To use smoothing, set 'smooth_method' above to the desired option:\n")
            f.write("#   - 'none' = No smoothing (default)\n")
            f.write("#   - 'savgol' = Savitzky-Golay filter (good for preserving peaks)\n")
            f.write("#   - 'lowess' = Local regression (good for noisy data)\n")
            f.write("#   - 'kalman' = Kalman filter (good for tracking)\n")
            f.write("#   - 'butterworth' = Butterworth filter (most common for biomechanics)\n")
            f.write("#   - 'splines' = Spline smoothing (very smooth curves)\n")
            f.write("#   - 'arima' = ARIMA model (for time series)\n")
            f.write("\n")

            # Group smoothing parameters by method
            sp = toml_config["smoothing_params"]

            f.write("# ----------------------------------------------------------------\n")
            f.write("# SAVITZKY-GOLAY FILTER (smooth_method = 'savgol')\n")
            f.write("# ----------------------------------------------------------------\n")
            f.write("# Best for: Preserving signal features while reducing noise\n")
            f.write("# Common use: When you want smooth movement but keep important details\n")
            f.write(
                f"savgol_window_length = {sp['savgol_window_length']}    # Window size (must be odd number)\n"
            )
            f.write("#                                 # Smaller = less smoothing (try 5-11)\n")
            f.write("#                                 # Larger = more smoothing (try 13-21)\n")
            f.write(
                f"savgol_polyorder = {sp['savgol_polyorder']}        # Polynomial order (usually 2 or 3)\n"
            )
            f.write("#                           # 2 = simpler curves, 3 = more complex curves\n\n")

            f.write("# ----------------------------------------------------------------\n")
            f.write("# LOWESS SMOOTHING (smooth_method = 'lowess')\n")
            f.write("# ----------------------------------------------------------------\n")
            f.write("# Best for: Very noisy data that needs strong smoothing\n")
            f.write("# Common use: Poor quality videos or tremor analysis\n")
            f.write(
                f"lowess_frac = {sp['lowess_frac']}        # Fraction of data to use (0.1 to 1.0)\n"
            )
            f.write("#                      # Smaller = less smoothing (try 0.1-0.3)\n")
            f.write("#                      # Larger = more smoothing (try 0.5-0.8)\n")
            f.write(
                f"lowess_it = {sp['lowess_it']}          # Number of iterations (usually 1-5)\n"
            )
            f.write("#                    # More iterations = more robust but slower\n\n")

            f.write("# ----------------------------------------------------------------\n")
            f.write("# BUTTERWORTH FILTER (smooth_method = 'butterworth') - MOST COMMON\n")
            f.write("# ----------------------------------------------------------------\n")
            f.write("# Best for: General biomechanics analysis (most used in research)\n")
            f.write("# Common use: Human movement, sports analysis, gait analysis\n")
            f.write("# IMPORTANT: Match butter_fs to your video frame rate!\n")
            f.write(
                f"butter_cutoff = {sp['butter_cutoff']}    # Cutoff frequency in Hz - ADJUST THIS!\n"
            )
            f.write("#                      # Lower = more smoothing:\n")
            f.write("#                      #   3Hz = heavy smoothing (slow movements)\n")
            f.write("#                      #   6Hz = medium smoothing (normal walking)\n")
            f.write("#                      #  10Hz = light smoothing (fast movements)\n")
            f.write("#                      #  15Hz = minimal smoothing (sports)\n")
            f.write(
                f"butter_fs = {sp['butter_fs']}       # Sampling frequency = VIDEO FRAME RATE!\n"
            )
            f.write("#                    # 30Hz for 30fps video, 60Hz for 60fps video\n")
            f.write("#                    # 120Hz for 120fps high-speed video\n\n")

            f.write("# ----------------------------------------------------------------\n")
            f.write("# KALMAN FILTER (smooth_method = 'kalman')\n")
            f.write("# ----------------------------------------------------------------\n")
            f.write("# Best for: Tracking objects with predictable motion\n")
            f.write("# Common use: Following specific body parts, sports tracking\n")
            f.write(
                f"kalman_iterations = {sp['kalman_iterations']}    # EM algorithm iterations (3-10)\n"
            )
            f.write("#                         # More = better fit but slower\n")
            f.write(
                f"kalman_mode = {sp['kalman_mode']}          # 1 = simple mode, 2 = advanced mode\n"
            )
            f.write("#                   # Mode 1 for most cases, Mode 2 for complex movements\n\n")

            f.write("# ----------------------------------------------------------------\n")
            f.write("# SPLINE SMOOTHING (smooth_method = 'splines')\n")
            f.write("# ----------------------------------------------------------------\n")
            f.write("# Best for: Creating very smooth curves for presentation\n")
            f.write("# Common use: Publication figures, smooth trajectories\n")
            f.write(
                f"spline_smoothing_factor = {sp['spline_smoothing_factor']}    # Smoothing factor\n"
            )
            f.write("#                               # 0 = no smoothing (follows data exactly)\n")
            f.write("#                               # 1 = moderate smoothing\n")
            f.write("#                               # 10+ = heavy smoothing\n\n")

            f.write("# ----------------------------------------------------------------\n")
            f.write("# ARIMA MODEL (smooth_method = 'arima')\n")
            f.write("# ----------------------------------------------------------------\n")
            f.write("# Best for: Time series analysis and prediction\n")
            f.write("# Common use: Advanced statistical analysis (requires expertise)\n")
            f.write(f"arima_p = {sp['arima_p']}    # Autoregressive order (usually 1-3)\n")
            f.write(f"arima_d = {sp['arima_d']}    # Differencing order (usually 0-1)\n")
            f.write(f"arima_q = {sp['arima_q']}    # Moving average order (usually 0-3)\n")
            f.write("# Note: ARIMA requires statistical knowledge to use effectively\n\n")

            f.write("# ================================================================\n")
            f.write("# QUICK REFERENCE FOR COMMON SCENARIOS:\n")
            f.write("# ================================================================\n")
            f.write("# Walking analysis:     butterworth, cutoff=6, fs=30\n")
            f.write("# Running analysis:     butterworth, cutoff=10, fs=60\n")
            f.write("# Sports (fast moves):  butterworth, cutoff=15, fs=120\n")
            f.write("# Tremor/shaky video:   lowess, frac=0.3, it=3\n")
            f.write("# Presentation plots:   splines, smoothing_factor=1.0\n")
            f.write("# Research publication: savgol, window=7, polyorder=3\n")
            f.write("# ================================================================\n")

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
            f.write("#                         # true = upscale cropped region for better detection\n")
            f.write("#                         # false = use cropped region at original size\n\n")

            f.write(
                f"resize_crop_scale = {bb['resize_crop_scale']}            # Scale factor for cropped region (2-8)\n"
            )
            f.write("#                         # 2 = double size (good for most cases)\n")
            f.write("#                         # 3-4 = better for very small subjects\n")
            f.write("#                         # 5-8 = for very distant or tiny people\n")
            f.write("# This is separate from video_resize. Only the cropped region is resized,\n")
            f.write("# which is more efficient than resizing the entire video.\n\n")

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
                }
            )
        else:
            print("Warning: No [mediapipe] section found, using defaults")

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

        # Advanced filtering settings
        if "advanced_filtering" in toml_config:
            af = toml_config["advanced_filtering"]
            print(f"Loading advanced filtering settings: {list(af.keys())}")
            config.update(
                {
                    "enable_advanced_filtering": bool(af.get("enable_advanced_filtering", False)),
                    "interp_method": str(af.get("interp_method", "linear")),
                    "smooth_method": str(af.get("smooth_method", "none")),
                    "max_gap": int(af.get("max_gap", 60)),
                }
            )
        else:
            print("Warning: No [advanced_filtering] section found, using defaults")

        # Smoothing parameters
        if "smoothing_params" in toml_config:
            sp = toml_config["smoothing_params"]
            print(f"Loading smoothing parameters: {list(sp.keys())}")

            # Determine which parameters to use based on smooth_method
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
                print(
                    f"Butterworth parameters loaded: cutoff={smooth_params['cutoff']}Hz, fs={smooth_params['fs']}Hz"
                )
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

            # Store all parameters for future use
            config["_all_smooth_params"] = {
                "savgol_window_length": int(sp.get("savgol_window_length", 7)),
                "savgol_polyorder": int(sp.get("savgol_polyorder", 3)),
                "lowess_frac": float(sp.get("lowess_frac", 0.3)),
                "lowess_it": int(sp.get("lowess_it", 3)),
                "butter_cutoff": float(sp.get("butter_cutoff", 10.0)),
                "butter_fs": float(sp.get("butter_fs", 100.0)),
                "kalman_iterations": int(sp.get("kalman_iterations", 5)),
                "kalman_mode": int(sp.get("kalman_mode", 1)),
                "spline_smoothing_factor": float(sp.get("spline_smoothing_factor", 1.0)),
                "arima_p": int(sp.get("arima_p", 1)),
                "arima_d": int(sp.get("arima_d", 0)),
                "arima_q": int(sp.get("arima_q", 0)),
            }
        else:
            print("Warning: No [smoothing_params] section found, using defaults")

        # Padding section
        if "padding" in toml_config:
            pad = toml_config["padding"]
            config["enable_padding"] = bool(pad.get("enable_padding", ENABLE_PADDING_DEFAULT))
            config["pad_start_frames"] = int(pad.get("pad_start_frames", PAD_START_FRAMES_DEFAULT))
        else:
            config["enable_padding"] = ENABLE_PADDING_DEFAULT
            config["pad_start_frames"] = PAD_START_FRAMES_DEFAULT

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

        print(f"Configuration loaded successfully from: {filepath}")
        print(f"Total parameters: {len(config)}")
        print(f"Advanced filtering: {config.get('enable_advanced_filtering', False)}")
        print(f"Smoothing method: {config.get('smooth_method', 'none')}")

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
        super().__init__(parent, title="MediaPipe and Resize Configuration (or TOML)")

    def body(self, master):
        # Labels
        tk.Label(master, text="min_detection_confidence (0.0 - 1.0):").grid(
            row=0, column=0, sticky="e"
        )
        tk.Label(master, text="min_tracking_confidence (0.0 - 1.0):").grid(
            row=1, column=0, sticky="e"
        )
        tk.Label(master, text="model_complexity (0, 1, 2):").grid(row=2, column=0, sticky="e")
        tk.Label(master, text="enable_segmentation (True/False):").grid(row=3, column=0, sticky="e")
        tk.Label(master, text="smooth_segmentation (True/False):").grid(row=4, column=0, sticky="e")
        tk.Label(master, text="static_image_mode (True/False):").grid(row=5, column=0, sticky="e")
        tk.Label(master, text="apply_filtering (True/False):").grid(row=6, column=0, sticky="e")
        tk.Label(master, text="estimate_occluded (True/False):").grid(row=7, column=0, sticky="e")
        tk.Label(master, text="enable_resize (True/False):").grid(row=8, column=0, sticky="e")
        tk.Label(master, text="resize_scale (2, 3, ...):").grid(row=9, column=0, sticky="e")
        tk.Label(master, text="Enable initial frame padding? (True/False):").grid(
            row=10, column=0, sticky="e"
        )
        tk.Label(master, text="Number of padding frames:").grid(row=11, column=0, sticky="e")

        # Entries
        self.min_detection_entry = tk.Entry(master)
        self.min_detection_entry.insert(0, "0.1")
        self.min_tracking_entry = tk.Entry(master)
        self.min_tracking_entry.insert(0, "0.1")
        self.model_complexity_entry = tk.Entry(master)
        self.model_complexity_entry.insert(0, "2")
        self.enable_segmentation_entry = tk.Entry(master)
        self.enable_segmentation_entry.insert(0, "False")
        self.smooth_segmentation_entry = tk.Entry(master)
        self.smooth_segmentation_entry.insert(0, "False")
        self.static_image_mode_entry = tk.Entry(master)
        self.static_image_mode_entry.insert(0, "False")
        self.apply_filtering_entry = tk.Entry(master)
        self.apply_filtering_entry.insert(0, "True")
        self.estimate_occluded_entry = tk.Entry(master)
        self.estimate_occluded_entry.insert(0, "True")
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
        self.model_complexity_entry.grid(row=2, column=1)
        self.enable_segmentation_entry.grid(row=3, column=1)
        self.smooth_segmentation_entry.grid(row=4, column=1)
        self.static_image_mode_entry.grid(row=5, column=1)
        self.apply_filtering_entry.grid(row=6, column=1)
        self.estimate_occluded_entry.grid(row=7, column=1)
        self.enable_resize_entry.grid(row=8, column=1)
        self.resize_scale_entry.grid(row=9, column=1)
        self.enable_padding_entry.grid(row=10, column=1)
        self.pad_start_frames_entry.grid(row=11, column=1)

        # Bounding box section
        bbox_frame = tk.LabelFrame(master, text="Bounding Box (ROI) Selection", padx=10, pady=10)
        bbox_frame.grid(row=12, column=0, columnspan=2, pady=(10, 0), sticky="ew")

        # Enable crop checkbox
        self.enable_crop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bbox_frame, text="Enable bounding box cropping", variable=self.enable_crop_var
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)

        # Bounding box coordinates
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

        # Normalized coordinates display (read-only)
        tk.Label(bbox_frame, text="Normalized coordinates:").grid(row=5, column=0, sticky="e", padx=5, pady=(5, 0))
        self.norm_coords_label = tk.Label(bbox_frame, text="(0.0, 0.0) to (1.0, 1.0)", fg="gray")
        self.norm_coords_label.grid(row=5, column=1, sticky="w", padx=5, pady=(5, 0))

        # Select ROI button
        select_roi_btn = tk.Button(
            bbox_frame,
            text="Select ROI from Video",
            command=self.select_roi_from_video,
            bg="#4CAF50",
            fg="white",
        )
        select_roi_btn.grid(row=6, column=0, columnspan=2, pady=10)

        # Resize crop options
        self.enable_resize_crop_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            bbox_frame, text="Resize cropped region", variable=self.enable_resize_crop_var
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=5)

        tk.Label(bbox_frame, text="Resize crop scale:").grid(row=8, column=0, sticky="e", padx=5)
        self.resize_crop_scale_entry = tk.Entry(bbox_frame, width=10)
        self.resize_crop_scale_entry.insert(0, "2")
        self.resize_crop_scale_entry.grid(row=8, column=1, sticky="w", padx=5)

        # Bind update function to coordinate entries
        for entry in [self.bbox_x_min_entry, self.bbox_y_min_entry, self.bbox_x_max_entry, self.bbox_y_max_entry]:
            entry.bind("<KeyRelease>", self.update_normalized_coords)

        # TOML section
        toml_frame = tk.LabelFrame(master, text="Advanced Configuration (TOML)", padx=10, pady=10)
        toml_frame.grid(row=13, column=0, columnspan=2, pady=(10, 0), sticky="ew")
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
                "enable_crop": default_config["bounding_box"]["enable_crop"],
                "bbox_x_min": default_config["bounding_box"]["bbox_x_min"],
                "bbox_y_min": default_config["bounding_box"]["bbox_y_min"],
                "bbox_x_max": default_config["bounding_box"]["bbox_x_max"],
                "bbox_y_max": default_config["bounding_box"]["bbox_y_max"],
                "enable_resize_crop": default_config["bounding_box"]["enable_resize_crop"],
                "resize_crop_scale": default_config["bounding_box"]["resize_crop_scale"],
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
                        text=f"TOML loaded: {os.path.basename(file_path)}", fg="green"
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
                        summary += f"enable_resize_crop: {config.get('enable_resize_crop', False)}\n"
                        if config.get("enable_resize_crop", False):
                            summary += f"resize_crop_scale: {config.get('resize_crop_scale', 2)}\n"
                    print("\n=== TOML configuration loaded and GUI updated ===\n" + summary)
                    from tkinter import messagebox

                    messagebox.showinfo("TOML Parameters Loaded", "Configuration loaded and GUI fields updated!\n\n" + summary)
                else:
                    self.toml_label.config(text="Error loading TOML", fg="red")
            except Exception as e:
                self.toml_label.config(text=f"Error: {e}", fg="red")
                from tkinter import messagebox
                messagebox.showerror("Error", f"Failed to load TOML: {e}")

        dialog_root.destroy()

    def show_help(self):
        """Show help window with script information and usage instructions"""
        help_window = tk.Toplevel()
        help_window.title("MediaPipe 2D Analysis - Help")
        help_window.geometry("800x600")
        help_window.configure(bg="white")

        # Make window modal and on top (without grab to avoid conflicts)
        help_window.transient()
        help_window.attributes("-topmost", True)
        help_window.focus_set()

        # Create scrollable text widget
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
MEDIAPIPE 2D ANALYSIS - HELP GUIDE
=====================================

OVERVIEW
--------
This script performs batch processing of videos for 2D pose estimation using MediaPipe's Pose model. 
It processes videos from a specified input directory, overlays pose landmarks on each video frame, 
and exports both normalized and pixel-based landmark coordinates to CSV files.

NEW FEATURES
------------
• Video resize functionality for better pose detection
• Batch processing for high-resolution videos (Linux only)
• Advanced filtering and smoothing options
• TOML configuration file support
• Automatic memory management

SYSTEM REQUIREMENTS
------------------
• Python 3.12.11
• OpenCV (pip install opencv-python)
• MediaPipe (pip install mediapipe)
• Tkinter (usually included with Python)
• Pandas (pip install pandas)
• psutil (pip install psutil) - for memory monitoring

BATCH PROCESSING (Linux Only)
----------------------------
For high-resolution videos (>2.7K) or low memory systems:
• Automatically detects when batch processing is needed
• Processes frames in small batches to prevent memory overflow
• Cleans memory after each batch
• Provides detailed progress information

VIDEO RESIZE FEATURE
-------------------
• Enable resize for better pose detection on small/distant subjects
• Scale factors: 2x to 8x (higher = better detection but slower)
• Coordinates automatically converted back to original video dimensions
• Useful for: small people, distant subjects, low resolution videos

ADVANCED FILTERING
-----------------
• Interpolation methods: linear, cubic, nearest, kalman
• Smoothing methods: none, butterworth, savgol, lowess, kalman, splines, arima
• Gap filling for missing landmark data
• Configurable parameters for each method

TOML CONFIGURATION
-----------------
• Create default TOML template with "Create Default TOML Template"
• Load existing configuration with "Load Configuration TOML"
• Save and reuse configurations for batch processing
• Detailed parameter descriptions in TOML files

OUTPUT FILES
-----------
For each processed video, the following files are generated:

1. Processed Video (*_mp.mp4):
   Video with 2D pose landmarks overlaid on original frames

2. Normalized Landmark CSV (*_mp_norm.csv):
   Landmark coordinates normalized to 0-1 scale for each frame

3. Pixel Landmark CSV (*_mp_pixel.csv):
   Landmark coordinates in pixel format (original video dimensions)

4. Log File (log_info.txt):
   Video metadata and processing information

5. Configuration File (configuration_used.toml):
   Settings used for processing (saved in output directory)

USAGE INSTRUCTIONS
-----------------
1. Run the script: python markerless_2D_analysis.py
2. Select input directory containing video files (.mp4, .avi, .mov)
3. Select output directory for processed files
4. Configure MediaPipe parameters or load TOML configuration
5. Click "OK" to start processing

PARAMETER GUIDE
--------------
MediaPipe Settings:
• min_detection_confidence (0.1-1.0): How confident to start detecting
• min_tracking_confidence (0.1-1.0): How confident to keep tracking
• model_complexity (0-2): 0=fastest, 1=balanced, 2=most accurate
• enable_segmentation: Draw person outline (slower but more detailed)
• static_image_mode: Treat each frame separately (slower)

Video Resize:
• enable_resize: Upscale video for better detection
• resize_scale (2-8): Scale factor (higher = better but slower)

Advanced Filtering:
• enable_advanced_filtering: Apply smoothing and gap filling
• interp_method: How to fill missing data
• smooth_method: Type of smoothing to apply
• max_gap: Maximum gap size to fill (frames)

RECOMMENDED SETTINGS
-------------------
For most cases:
• min_detection_confidence: 0.1-0.3
• min_tracking_confidence: 0.1-0.3
• model_complexity: 2 (for accuracy)
• enable_resize: true (for small/distant subjects)
• resize_scale: 2-3 (good balance)

For high-resolution videos:
• Use batch processing (automatic on Linux)
• Consider lower model_complexity (1) for speed
• Enable advanced filtering for smooth results

TROUBLESHOOTING
--------------
• If processing is slow: Reduce model_complexity or disable resize
• If memory issues: Batch processing will activate automatically on Linux
• If poor detection: Increase resize_scale or adjust confidence thresholds
• If system crashes: Reduce batch size or disable advanced filtering

BATCH PROCESSING DETAILS
-----------------------
• Automatically activated on Linux for high-resolution videos
• Batch sizes: 4K+ (30 frames), 2.7K-4K (50 frames), 1080p-2.7K (100 frames)
• Memory cleanup after each batch
• Progress tracking with detailed logs

For more information, visit: https://github.com/vaila-multimodaltoolbox/vaila
        """

        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Make read-only

        # Close button
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
            messagebox.showerror("Error", "No input directory specified. Please select input directory first.")
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
            display_frame = cv2.resize(original_frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            print(f"Frame resized for display: {orig_width}x{orig_height} -> {display_width}x{display_height} (scale: {scale_factor:.2f})")
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

    def populate_fields_from_config(self, config):
        """Populate GUI fields with values from loaded TOML config"""
        # MediaPipe parameters
        self.min_detection_entry.delete(0, tk.END)
        self.min_detection_entry.insert(0, str(config.get("min_detection_confidence", 0.1)))
        
        self.min_tracking_entry.delete(0, tk.END)
        self.min_tracking_entry.insert(0, str(config.get("min_tracking_confidence", 0.1)))
        
        self.model_complexity_entry.delete(0, tk.END)
        self.model_complexity_entry.insert(0, str(config.get("model_complexity", 2)))
        
        self.enable_segmentation_entry.delete(0, tk.END)
        self.enable_segmentation_entry.insert(0, str(config.get("enable_segmentation", False)))
        
        self.smooth_segmentation_entry.delete(0, tk.END)
        self.smooth_segmentation_entry.insert(0, str(config.get("smooth_segmentation", False)))
        
        self.static_image_mode_entry.delete(0, tk.END)
        self.static_image_mode_entry.insert(0, str(config.get("static_image_mode", False)))
        
        self.apply_filtering_entry.delete(0, tk.END)
        self.apply_filtering_entry.insert(0, str(config.get("apply_filtering", True)))
        
        self.estimate_occluded_entry.delete(0, tk.END)
        self.estimate_occluded_entry.insert(0, str(config.get("estimate_occluded", True)))
        
        self.enable_resize_entry.delete(0, tk.END)
        self.enable_resize_entry.insert(0, str(config.get("enable_resize", False)))
        
        self.resize_scale_entry.delete(0, tk.END)
        self.resize_scale_entry.insert(0, str(config.get("resize_scale", 2)))
        
        self.enable_padding_entry.delete(0, tk.END)
        self.enable_padding_entry.insert(0, str(config.get("enable_padding", ENABLE_PADDING_DEFAULT)))
        
        self.pad_start_frames_entry.delete(0, tk.END)
        self.pad_start_frames_entry.insert(0, str(config.get("pad_start_frames", PAD_START_FRAMES_DEFAULT)))
        
        # Bounding box parameters
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
        
        # Update normalized coordinates display
        self.update_normalized_coords()

    def apply(self):
        if self.use_toml and self.loaded_config:
            self.result = self.loaded_config
        else:
            self.result = {
                "min_detection_confidence": float(self.min_detection_entry.get()),
                "min_tracking_confidence": float(self.min_tracking_entry.get()),
                "model_complexity": int(self.model_complexity_entry.get()),
                "enable_segmentation": self.enable_segmentation_entry.get().lower() == "true",
                "smooth_segmentation": self.smooth_segmentation_entry.get().lower() == "true",
                "static_image_mode": self.static_image_mode_entry.get().lower() == "true",
                "apply_filtering": self.apply_filtering_entry.get().lower() == "true",
                "estimate_occluded": self.estimate_occluded_entry.get().lower() == "true",
                "enable_resize": self.enable_resize_entry.get().lower() == "true",
                "resize_scale": int(self.resize_scale_entry.get()),
                # Defaults for other advanced parameters
                "enable_advanced_filtering": False,
                "interp_method": "linear",
                "smooth_method": "none",
                "max_gap": 60,
                "_all_smooth_params": get_default_config()["smoothing_params"],
                "enable_padding": self.enable_padding_entry.get().lower() == "true",
                "pad_start_frames": int(self.pad_start_frames_entry.get()),
                # Bounding box settings
                "enable_crop": self.enable_crop_var.get(),
                "bbox_x_min": int(self.bbox_x_min_entry.get() or 0),
                "bbox_y_min": int(self.bbox_y_min_entry.get() or 0),
                "bbox_x_max": int(self.bbox_x_max_entry.get() or 1920),
                "bbox_y_max": int(self.bbox_y_max_entry.get() or 1080),
                "enable_resize_crop": self.enable_resize_crop_var.get(),
                "resize_crop_scale": int(self.resize_crop_scale_entry.get() or 2),
            }


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


def map_landmarks_to_full_frame(landmarks, bbox_config, crop_width, crop_height, processing_width, processing_height, original_width=None, original_height=None):
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

        elif right_visible and not left_visible:
            if not np.isnan(landmarks[0][0]):
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


def process_video_batch(
    frames, landmarker, pose_config, width, height, full_width=None, full_height=None, progress_callback=None, batch_index=0, fps=30.0, start_frame_idx=0
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

    # Extract bounding box config
    enable_crop = pose_config.get("enable_crop", False)
    bbox_config = {
        "enable_resize_crop": pose_config.get("enable_resize_crop", False),
        "resize_crop_scale": pose_config.get("resize_crop_scale", 2),
        "bbox_x_min": pose_config.get("bbox_x_min", 0),
        "bbox_y_min": pose_config.get("bbox_y_min", 0),
        "bbox_x_max": pose_config.get("bbox_x_max", width),
        "bbox_y_max": pose_config.get("bbox_y_max", height),
    }

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

        # Process frame with Tasks API
        landmarks = process_frame_with_tasks_api(
            frame, landmarker, timestamp_ms, enable_crop, bbox_config,
            width, height, full_width, full_height, pose_config, landmarks_history
        )

        if landmarks:
            landmarks_history.append(landmarks)
            batch_landmarks.append(landmarks)

            # Convert to pixel coordinates (using full frame dimensions)
            pixel_landmarks = [
                [int(lm[0] * full_width), int(lm[1] * full_height), lm[2]]
                for lm in landmarks
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
        2: "pose_landmarker_heavy.task"
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
            2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
        }
        url = model_urls.get(complexity, model_urls[1])
        try:
            urllib.request.urlretrieve(url, str(model_path))
            print("Download completed!")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise RuntimeError("Failed to download MediaPipe model.")  
    return str(model_path.resolve())


def process_frame_with_tasks_api(frame, landmarker, timestamp_ms, enable_crop, bbox_config, process_width, process_height, original_width, original_height, pose_config, landmarks_history):
    """
    Process a single frame using MediaPipe Tasks API.
    Returns landmarks in normalized coordinates (mapped to full frame if cropping was used).
    """
    # Apply bounding box cropping if enabled
    process_frame = frame
    actual_process_width = process_width
    actual_process_height = process_height
    
    if enable_crop:
        x_min = max(0, min(bbox_config["bbox_x_min"], process_width - 1))
        y_min = max(0, min(bbox_config["bbox_y_min"], process_height - 1))
        x_max = max(x_min + 1, min(bbox_config["bbox_x_max"], process_width))
        y_max = max(y_min + 1, min(bbox_config["bbox_y_max"], process_height))
        process_frame = frame[y_min:y_max, x_min:x_max]
        actual_process_width = x_max - x_min
        actual_process_height = y_max - y_min
        
        if bbox_config.get("enable_resize_crop", False) and bbox_config.get("resize_crop_scale", 1) > 1:
            new_w = int(actual_process_width * bbox_config["resize_crop_scale"])
            new_h = int(actual_process_height * bbox_config["resize_crop_scale"])
            process_frame = cv2.resize(process_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            actual_process_width = new_w
            actual_process_height = new_h
    
    # Convert to RGB and create MP Image
    rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect pose
    pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    # Parse results
    if pose_landmarker_result.pose_landmarks and len(pose_landmarker_result.pose_landmarks) > 0:
        # Get first pose
        raw_landmarks = pose_landmarker_result.pose_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in raw_landmarks]
        
        # Map coordinates to full frame if cropping was used
        if enable_crop:
            # Pass processing dimensions (may be resized) and original dimensions
            landmarks = map_landmarks_to_full_frame(
                landmarks, bbox_config, actual_process_width, actual_process_height, 
                process_width, process_height,  # Processing frame dimensions (may be resized)
                original_width, original_height  # Original video dimensions
            )
        
        # Apply occluded landmark estimation if enabled
        if pose_config.get("estimate_occluded", False):
            landmarks = estimate_occluded_landmarks(landmarks, list(landmarks_history))
        
        return landmarks
    else:
        # No pose detected
        return None


def process_video(video_path, output_dir, pose_config):
    """
    Process a video file using MediaPipe Pose estimation with optional video resize.
    """
    # #region agent log
    try:
        import mediapipe as mp_test
        _debug_log("E", "process_video:2475", "Function entry", {"video_path":str(video_path),"output_dir":str(output_dir),"has_mp_solutions":hasattr(mp_test,"solutions"),"mp_test_type":str(type(mp_test))})
    except Exception as log_e:
        # Log function itself may fail, but continue
        pass
    # #endregion
    print("\n=== Parameters being used for this video ===")
    for k, v in pose_config.items():
        print(f"{k}: {v}")
    print("==========================================\n")

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
    try:
        _debug_log("A", "process_video:2610", "Initializing MediaPipe Tasks API", {})
    except: pass
    # #endregion
    
    model_path = get_model_path(pose_config["model_complexity"])
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create options
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=pose_config["min_detection_confidence"],
        min_pose_presence_confidence=pose_config["min_tracking_confidence"],
        output_segmentation_masks=pose_config["enable_segmentation"]
    )
    
    # #region agent log
    try:
        _debug_log("B", "process_video:2625", "MediaPipe Tasks API options created", {"model_path": model_path})
    except: pass
    # #endregion

    # Prepare headers for CSV
    headers = ["frame_index"]
    for name in landmark_names:
        headers.extend([f"{name}_x", f"{name}_y", f"{name}_z"])

    # Lists to store landmarks
    normalized_landmarks_list = []
    pixel_landmarks_list = []
    frames_with_missing_data = []
    landmarks_history = deque(maxlen=10)

    step_text = "Step 2/3" if enable_resize else "Step 1/2"
    print(f"\n{step_text}: Processing landmarks (total frames: {total_frames})")

    # Get FPS for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # Default FPS

    # --- Frame padding for MediaPipe stabilization ---
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read first frame for padding.")
        cap.release()
        return

    enable_padding = pose_config.get("enable_padding", ENABLE_PADDING_DEFAULT)
    pad_start_frames = pose_config.get("pad_start_frames", PAD_START_FRAMES_DEFAULT)

    print(
        f"Padding configuration: enable_padding={enable_padding}, pad_start_frames={pad_start_frames}"
    )

    # Store first frame for padding (only one copy, we'll reuse it)
    padding_frame = first_frame.copy() if enable_padding and pad_start_frames > 0 else None

    # Extract bounding box config
    enable_crop = pose_config.get("enable_crop", False)
    # IMPORTANT: If video was resized, scale bbox coordinates to match resized video dimensions
    if enable_resize and resize_metadata:
        resize_scale = resize_metadata["scale_factor"]
        bbox_x_min_orig = pose_config.get("bbox_x_min", 0)
        bbox_y_min_orig = pose_config.get("bbox_y_min", 0)
        bbox_x_max_orig = pose_config.get("bbox_x_max", original_width)
        bbox_y_max_orig = pose_config.get("bbox_y_max", original_height)
        # Scale bbox coordinates to resized video
        bbox_x_min_scaled = int(bbox_x_min_orig * resize_scale)
        bbox_y_min_scaled = int(bbox_y_min_orig * resize_scale)
        bbox_x_max_scaled = int(bbox_x_max_orig * resize_scale)
        bbox_y_max_scaled = int(bbox_y_max_orig * resize_scale)
        print(f"Bbox scaled for resized video: ({bbox_x_min_scaled}, {bbox_y_min_scaled}) to ({bbox_x_max_scaled}, {bbox_y_max_scaled})")
    else:
        bbox_x_min_scaled = pose_config.get("bbox_x_min", 0)
        bbox_y_min_scaled = pose_config.get("bbox_y_min", 0)
        bbox_x_max_scaled = pose_config.get("bbox_x_max", width)
        bbox_y_max_scaled = pose_config.get("bbox_y_max", height)
    
    bbox_config = {
        "enable_resize_crop": pose_config.get("enable_resize_crop", False),
        "resize_crop_scale": pose_config.get("resize_crop_scale", 2),
        "bbox_x_min": bbox_x_min_scaled,  # Use scaled coordinates for processing
        "bbox_y_min": bbox_y_min_scaled,
        "bbox_x_max": bbox_x_max_scaled,
        "bbox_y_max": bbox_y_max_scaled,
        # Store original coordinates for mapping back
        "bbox_x_min_orig": pose_config.get("bbox_x_min", 0),
        "bbox_y_min_orig": pose_config.get("bbox_y_min", 0),
        "bbox_x_max_orig": pose_config.get("bbox_x_max", original_width),
        "bbox_y_max_orig": pose_config.get("bbox_y_max", original_height),
        "video_resized": enable_resize and resize_metadata is not None,
        "resize_scale": resize_metadata["scale_factor"] if (enable_resize and resize_metadata) else 1.0,
    }

    # Check if batch processing should be used
    use_batch_processing = should_use_batch_processing(video_path, pose_config)

    # Initialize frame_count and results lists
    frame_count = 0
    normalized_landmarks_list = []
    pixel_landmarks_list = []
    frames_with_missing_data = []
    landmarks_history = deque(maxlen=10)

    if use_batch_processing:
        print(
            "Using batch processing for high-resolution video (frame-by-frame to avoid memory out)"
        )
        batch_size = calculate_batch_size(video_path, pose_config)

        # Process with Tasks API
        with PoseLandmarker.create_from_options(options) as landmarker:
            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Process padding frames first if enabled
            if enable_padding and pad_start_frames > 0 and padding_frame is not None:
                print(f"Processing {pad_start_frames} padding frames...")
                for pad_idx in range(pad_start_frames):
                    if should_throttle_cpu(frame_count):
                        apply_cpu_throttling()

                    timestamp_ms = int((frame_count * 1000) / fps) if fps > 0 else frame_count * 33
                    landmarks = process_frame_with_tasks_api(
                        padding_frame, landmarker, timestamp_ms, enable_crop, bbox_config,
                        width, height, original_width, original_height, pose_config, landmarks_history
                    )

                    if landmarks:
                        landmarks_history.append(landmarks)
                        normalized_landmarks_list.append(landmarks)
                        pixel_landmarks = [
                            [int(lm[0] * original_width), int(lm[1] * original_height), lm[2]]
                            for lm in landmarks
                        ]
                        pixel_landmarks_list.append(pixel_landmarks)
                    else:
                        num_landmarks = len(landmark_names)
                        nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
                        normalized_landmarks_list.append(nan_landmarks)
                        pixel_landmarks_list.append(nan_landmarks)
                        frames_with_missing_data.append(frame_count)

                    frame_count += 1
                    time.sleep(FRAME_SLEEP_TIME)

            # Process video frames in batches (read frame-by-frame, process in batches)
            batch_frames = []
            batch_start_idx = frame_count
            current_batch = 0
            total_batches = (total_frames + batch_size - 1) // batch_size

            while True:
                ret, frame = cap.read()
                if not ret:
                    # Process remaining frames in batch
                    if batch_frames:
                        print(
                            f"Processing final batch {current_batch + 1}/{total_batches} ({len(batch_frames)} frames)"
                        )
                        batch_norm, batch_pixel = process_video_batch(
                            batch_frames,
                            landmarker,
                            pose_config,
                            width,
                            height,
                            full_width=original_width,
                            full_height=original_height,
                            progress_callback=lambda msg: print(f"    {msg}"),
                            batch_index=current_batch,
                            fps=fps,
                            start_frame_idx=batch_start_idx,
                        )
                        normalized_landmarks_list.extend(batch_norm)
                        pixel_landmarks_list.extend(batch_pixel)
                        for i, landmarks in enumerate(batch_norm):
                            if all(np.isnan(lm[0]) for lm in landmarks):
                                frames_with_missing_data.append(batch_start_idx + i)
                        batch_frames = []
                        cleanup_memory()
                    break

                batch_frames.append(frame.copy())  # Copy frame to avoid reference issues

                # When batch is full, process it
                if len(batch_frames) >= batch_size:
                    print(
                        f"Processing batch {current_batch + 1}/{total_batches} (frames {batch_start_idx}-{batch_start_idx + len(batch_frames) - 1})"
                    )

                    batch_norm, batch_pixel = process_video_batch(
                        batch_frames,
                        landmarker,
                        pose_config,
                        width,
                        height,
                        full_width=original_width,
                        full_height=original_height,
                        progress_callback=lambda msg: print(f"    {msg}"),
                        batch_index=current_batch,
                        fps=fps,
                        start_frame_idx=batch_start_idx,
                    )

                    normalized_landmarks_list.extend(batch_norm)
                    pixel_landmarks_list.extend(batch_pixel)

                    for i, landmarks in enumerate(batch_norm):
                        if all(np.isnan(lm[0]) for lm in landmarks):
                            frames_with_missing_data.append(batch_start_idx + i)

                    # Clear batch and free memory immediately
                    del batch_frames
                    batch_frames = []
                    cleanup_memory()
                    print(f"Batch {current_batch + 1} completed, memory cleaned")

                    batch_start_idx += len(batch_norm)
                    current_batch += 1
                    time.sleep(0.1)

        # Set frame_count for batch processing
        frame_count = len(normalized_landmarks_list)
        cap.release()
        cv2.destroyAllWindows()

    else:
        # Standard processing for normal resolution videos (frame-by-frame) using Tasks API
        print("Using standard processing (frame-by-frame to avoid memory out)")

        # Process with Tasks API
        with PoseLandmarker.create_from_options(options) as landmarker:
            # Process padding frames first if enabled
            if enable_padding and pad_start_frames > 0 and padding_frame is not None:
                print(f"Processing {pad_start_frames} padding frames...")
                for pad_idx in range(pad_start_frames):
                    if should_throttle_cpu(frame_count):
                        apply_cpu_throttling()

                    timestamp_ms = int((frame_count * 1000) / fps) if fps > 0 else frame_count * 33
                    landmarks = process_frame_with_tasks_api(
                        padding_frame, landmarker, timestamp_ms, enable_crop, bbox_config,
                        width, height, original_width, original_height, pose_config, landmarks_history
                    )

                    if landmarks:
                        if VERBOSE_FRAMES:
                            print(f"Padding frame {pad_idx}: pose detected!")
                        landmarks_history.append(landmarks)
                        if pose_config.get("apply_filtering", False) and len(landmarks_history) > 3:
                            landmarks = apply_temporal_filter(list(landmarks_history))
                        normalized_landmarks_list.append(landmarks)
                        pixel_landmarks = [
                            [int(lm[0] * original_width), int(lm[1] * original_height), lm[2]]
                            for lm in landmarks
                        ]
                        pixel_landmarks_list.append(pixel_landmarks)
                    else:
                        if VERBOSE_FRAMES:
                            print(f"Padding frame {pad_idx}: NO pose detected")
                        num_landmarks = len(landmark_names)
                        nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
                        normalized_landmarks_list.append(nan_landmarks)
                        pixel_landmarks_list.append(nan_landmarks)
                        frames_with_missing_data.append(frame_count)

                    frame_count += 1
                    time.sleep(FRAME_SLEEP_TIME)

            # Reset video to beginning and process real frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Process all frames frame-by-frame (no memory accumulation)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # CPU throttling check for standard processing
                if should_throttle_cpu(frame_count):
                    apply_cpu_throttling()

                timestamp_ms = int((frame_count * 1000) / fps) if fps > 0 else frame_count * 33
                landmarks = process_frame_with_tasks_api(
                    frame, landmarker, timestamp_ms, enable_crop, bbox_config,
                    width, height, original_width, original_height, pose_config, landmarks_history
                )

                if landmarks:
                    if VERBOSE_FRAMES:
                        print(f"Frame {frame_count}: pose detected!")
                    landmarks_history.append(landmarks)
                    if pose_config.get("apply_filtering", False) and len(landmarks_history) > 3:
                        landmarks = apply_temporal_filter(list(landmarks_history))
                    normalized_landmarks_list.append(landmarks)
                    pixel_landmarks = [
                        [int(lm[0] * original_width), int(lm[1] * original_height), lm[2]]
                        for lm in landmarks
                    ]
                    pixel_landmarks_list.append(pixel_landmarks)
                else:
                    if VERBOSE_FRAMES:
                        print(f"Frame {frame_count}: NO pose detected")
                    num_landmarks = len(landmark_names)
                    nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
                    normalized_landmarks_list.append(nan_landmarks)
                    pixel_landmarks_list.append(nan_landmarks)
                    frames_with_missing_data.append(frame_count)

                frame_count += 1

                # Small sleep between frames to prevent CPU overload
                time.sleep(FRAME_SLEEP_TIME)

                # Progress info every 100 frames
                if frame_count % 100 == 0:
                    print(
                        f"  Processed {frame_count}/{total_frames + (pad_start_frames if enable_padding else 0)} frames"
                    )

        cap.release()
        cv2.destroyAllWindows()

    # --- Remove padding frames from results ---
    if enable_padding and pad_start_frames > 0:
        print(f"Removing {pad_start_frames} padding frames from results")
        print(f"Before removal: {len(normalized_landmarks_list)} frames")
        # NÃO remover padding aqui - vamos aplicar o filtro primeiro
        # normalized_landmarks_list = normalized_landmarks_list[pad_start_frames:]
        # pixel_landmarks_list = pixel_landmarks_list[pad_start_frames:]
        # frames_with_missing_data = [f-pad_start_frames for f in frames_with_missing_data if f >= pad_start_frames]
        print(f"Keeping padding for advanced filtering: {len(normalized_landmarks_list)} frames")
    else:
        print("No padding frames to remove")

    # Convert landmarks to DataFrames for advanced processing
    step_text = "Step 3/4" if enable_resize else "Step 2/3"
    print(f"\n{step_text}: Converting landmarks to DataFrames")
    df_norm_data = []
    df_pixel_data = []
    for frame_idx in range(len(normalized_landmarks_list)):
        landmarks_norm = normalized_landmarks_list[frame_idx]
        landmarks_pixel = pixel_landmarks_list[frame_idx]
        norm_row = [frame_idx]
        pixel_row = [frame_idx]
        for landmark in landmarks_norm:
            norm_row.extend(landmark)
        for landmark in landmarks_pixel:
            pixel_row.extend(landmark)
        df_norm_data.append(norm_row)
        df_pixel_data.append(pixel_row)
    columns = ["frame_index"]
    for name in landmark_names:
        columns.extend([f"{name}_x", f"{name}_y", f"{name}_z"])
    df_norm = pd.DataFrame(df_norm_data, columns=columns)
    df_pixel = pd.DataFrame(df_pixel_data, columns=columns)
    # Keep an unfiltered copy for optional RAW vailá export
    df_pixel_unfiltered = df_pixel.copy()

    # Apply advanced filtering and smoothing if enabled (COM PADDING)
    if pose_config.get("enable_advanced_filtering", False):
        step_text = "Step 4/5" if enable_resize else "Step 3/4"
        print(f"\n{step_text}: Applying advanced filtering and smoothing (with padding)")

        # Apply to normalized data (with padding)
        df_norm = apply_interpolation_and_smoothing(
            df_norm, pose_config, lambda msg: print(f"  Normalized data: {msg}")
        )

        # Apply to pixel data (with padding)
        df_pixel = apply_interpolation_and_smoothing(
            df_pixel, pose_config, lambda msg: print(f"  Pixel data: {msg}")
        )

    # AGORA remover padding dos resultados filtrados
    if enable_padding and pad_start_frames > 0:
        print(f"Removing {pad_start_frames} padding frames from filtered results")
        print(f"Before removal: {len(df_norm)} frames")
        df_norm = df_norm.iloc[pad_start_frames:].reset_index(drop=True)
        df_pixel = df_pixel.iloc[pad_start_frames:].reset_index(drop=True)
        # Ajustar frame_index para começar do 0
        df_norm["frame_index"] = df_norm.index
        df_pixel["frame_index"] = df_pixel.index
        print(f"After removal: {len(df_norm)} frames")
        # Remove padding from unfiltered as well to keep alignment
        try:
            df_pixel_unfiltered = df_pixel_unfiltered.iloc[pad_start_frames:].reset_index(drop=True)
            df_pixel_unfiltered["frame_index"] = df_pixel_unfiltered.index
        except Exception:
            pass
    else:
        print("No padding frames to remove from filtered results")

    # Save processed CSVs
    print(f"\n{step_text}: Saving processed CSVs")
    # Always save normalized CSV
    df_norm.to_csv(output_file_path, index=False, float_format="%.6f")

    # Always convert pixel coordinates to original size if resize was used
    if enable_resize and resize_metadata:
        df_pixel_original = convert_coordinates_to_original(df_pixel, resize_metadata)
        df_pixel_unfiltered_original = convert_coordinates_to_original(
            df_pixel_unfiltered, resize_metadata
        )
    else:
        df_pixel_original = df_pixel
        df_pixel_unfiltered_original = df_pixel_unfiltered

    # Save only the pixel CSV in original size
    df_pixel_original.to_csv(output_pixel_file_path, index=False)
    print(f"Saved: {output_file_path} (normalized)")
    print(f"Saved: {output_pixel_file_path} (pixel, original size)")

    # Convert and save in vailá format (filtered as primary)
    vaila_file_path = output_dir / f"{video_path.stem}_mp_vaila.csv"
    success = convert_mediapipe_to_vaila_format(df_pixel_original, vaila_file_path)
    if success:
        print(f"Saved: {vaila_file_path} (vailá format)")
    else:
        print("Warning: Failed to save vailá format file")

    # If advanced filtering was applied, also export RAW (unfiltered) vailá
    if pose_config.get("enable_advanced_filtering", False):
        vaila_raw_path = output_dir / f"{video_path.stem}_mp_vaila_raw.csv"
        success_raw = convert_mediapipe_to_vaila_format(
            df_pixel_unfiltered_original, vaila_raw_path
        )
        if success_raw:
            print(f"Saved: {vaila_raw_path} (vailá RAW, unfiltered)")
        else:
            print("Warning: Failed to save vailá RAW file")

    # If smoothing/filtering was applied, save an extra CSV for smoothed pixel and norm data
    if (
        pose_config.get("enable_advanced_filtering", False)
        and pose_config.get("smooth_method", "none") != "none"
    ):
        smoothed_pixel_path = output_dir / f"{video_path.stem}_mp_pixel_smoothed.csv"
        df_pixel_smoothed = df_pixel_original.copy()
        df_pixel_smoothed = apply_interpolation_and_smoothing(
            df_pixel_smoothed,
            pose_config,
            lambda msg: print(f"  Pixel data (smoothed only): {msg}"),
        )
        df_pixel_smoothed.to_csv(smoothed_pixel_path, index=False)
        print(f"Saved: {smoothed_pixel_path} (pixel, smoothed)")

        smoothed_norm_path = output_dir / f"{video_path.stem}_mp_norm_smoothed.csv"
        df_norm_smoothed = df_norm.copy()
        df_norm_smoothed = apply_interpolation_and_smoothing(
            df_norm_smoothed,
            pose_config,
            lambda msg: print(f"  Norm data (smoothed only): {msg}"),
        )
        df_norm_smoothed.to_csv(smoothed_norm_path, index=False, float_format="%.6f")
        print(f"Saved: {smoothed_norm_path} (norm, smoothed)")

    # Determine final step number
    final_step = "Step 2/2"
    if enable_resize:
        final_step = (
            "Step 4/4" if pose_config.get("enable_advanced_filtering", False) else "Step 3/3"
        )
    elif pose_config.get("enable_advanced_filtering", False):
        final_step = "Step 4/4"

    print(f"\n{final_step}: Generating video with processed landmarks")

    # Generate the video using the processed landmarks (use original video for output)
    cap = cv2.VideoCapture(str(video_path))  # Use original video for final output
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sanitize FPS to avoid invalid timebase issues with short/merged videos
    from fractions import Fraction

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    # Safe defaults for problematic FPS values
    if not orig_fps or np.isnan(orig_fps) or orig_fps < 1 or orig_fps > 240:
        safe_fps = 30.0
    else:
        # Reduce denominator to avoid huge timebase that breaks MPEG-4 standard
        safe_fps = float(Fraction(orig_fps).limit_denominator(1000))

    # Write to a temporary file first; we'll transcode/move to final universal MP4
    temp_output_video_path = output_dir / f"{video_path.stem}_mp_tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(temp_output_video_path), fourcc, safe_fps, (original_width, original_height)
    )

    # Fallback if VideoWriter fails to open
    if not out.isOpened():
        print(f"Warning: VideoWriter failed with FPS {safe_fps}, trying fallback FPS 30.0")
        safe_fps = 30.0
        out = cv2.VideoWriter(
            str(temp_output_video_path),
            fourcc,
            safe_fps,
            (original_width, original_height),
        )

    # Use manual drawing with OpenCV (Tasks API doesn't have drawing utilities)
    # POSE_CONNECTIONS is defined at module level

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(
                f"\r Generating video {frame_idx}/{total_frames} ({progress:.1f}%)",
                end="",
            )

        # Get processed landmarks for this frame from DataFrame
        if frame_idx < len(df_pixel):
            frame_data = df_pixel.iloc[frame_idx]

            # Convert DataFrame row back to landmarks format
            landmarks_px = []
            for name in landmark_names:
                x_col = f"{name}_x"
                y_col = f"{name}_y"
                z_col = f"{name}_z"

                x_val = frame_data[x_col] if pd.notna(frame_data[x_col]) else np.nan
                y_val = frame_data[y_col] if pd.notna(frame_data[y_col]) else np.nan
                z_val = frame_data[z_col] if pd.notna(frame_data[z_col]) else np.nan

                landmarks_px.append([x_val, y_val, z_val])

            # Convert landmarks back to original scale if resize was used
            if enable_resize and resize_metadata:
                scale_factor = resize_metadata["scale_factor"]
                landmarks_px = [
                    [
                        lm[0] / scale_factor if not np.isnan(lm[0]) else lm[0],
                        lm[1] / scale_factor if not np.isnan(lm[1]) else lm[1],
                        lm[2],
                    ]
                    for lm in landmarks_px
                ]

            # Draw landmarks using processed data (manual drawing with OpenCV)
            if not all(np.isnan(lm[0]) for lm in landmarks_px):
                # Extract points
                points = {}
                for i, lm in enumerate(landmarks_px):
                    if not np.isnan(lm[0]) and not np.isnan(lm[1]):
                        x = int(lm[0])
                        y = int(lm[1])
                        points[i] = (x, y)
                        # Draw point
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw connections (using POSE_CONNECTIONS defined at module level)
                for start_idx, end_idx in POSE_CONNECTIONS:
                    if start_idx in points and end_idx in points:
                        cv2.line(frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    # Close resources
    cap.release()
    out.release()

    # Finalize single universal MP4: prefer H.264 if ffmpeg is available; otherwise keep mp4v
    try:
        # Check if temp video was created successfully and has content
        if not temp_output_video_path.exists() or temp_output_video_path.stat().st_size == 0:
            print("Warning: Temporary video file is empty or missing, skipping ffmpeg transcoding")
            # Try to move the temp file anyway (in case it exists but is empty)
            try:
                if temp_output_video_path.exists():
                    shutil.move(str(temp_output_video_path), str(output_video_path))
                    print(f"Saved final video (mp4v): {output_video_path}")
                else:
                    print("Error: No video file was created")
            except Exception as _move_e:
                print(f"Warning: Could not move temp video to final path: {_move_e}")
            return

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
            import subprocess as _sub

            res = _sub.run(cmd, check=False, capture_output=True)
            if output_video_path.exists() and output_video_path.stat().st_size > 0:
                print(f"Saved final video (H.264): {output_video_path}")
                try:
                    os.remove(temp_output_video_path)
                except Exception:
                    pass
            else:
                print("ffmpeg failed to produce final video; falling back to mp4v file.")
                try:
                    shutil.move(str(temp_output_video_path), str(output_video_path))
                except Exception as _move_e:
                    print(f"Warning: Could not move temp video to final path: {_move_e}")
        else:
            # No ffmpeg: move temp mp4v to final path to keep only one video
            try:
                shutil.move(str(temp_output_video_path), str(output_video_path))
                print(f"Saved final video (mp4v): {output_video_path}")
            except Exception as _move_e:
                print(f"Warning: Could not move temp video to final path: {_move_e}")
    except Exception as _e:
        print(f"Warning: Could not finalize video: {_e}")

    # Clean up temporary files
    if temp_resized_video and os.path.exists(temp_resized_video):
        try:
            os.remove(temp_resized_video)
            os.rmdir(os.path.dirname(temp_resized_video))
            print("\n  Cleaned up temporary resized video")
        except Exception as e:
            print(f"Warning: Could not clean up temporary resized video: {e}")

    # Save configuration used for this video
    try:
        config_copy_path = output_dir / "configuration_used.toml"
        save_config_to_toml(pose_config, str(config_copy_path))
        print(f"Configuration saved to output directory: {config_copy_path}")
    except Exception as e:
        print(f"Warning: Could not save configuration to output directory: {e}")

    # Create log
    end_time = time.time()
    execution_time = end_time - start_time

    log_info_path = output_dir / "log_info.txt"
    with open(log_info_path, "w") as log_file:
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write("Configuration File: configuration_used.toml (saved in this directory)\n")
        if enable_resize:
            log_file.write(f"Video Resize: Enabled ({resize_scale}x scaling)\n")
            log_file.write(
                f"Original Resolution: {resize_metadata['original_width']}x{resize_metadata['original_height']}\n"
            )
            log_file.write(
                f"Processing Resolution: {resize_metadata['output_width']}x{resize_metadata['output_height']}\n"
            )
        else:
            log_file.write("Video Resize: Disabled\n")
            log_file.write(f"Resolution: {original_width}x{original_height}\n")
        log_file.write(f"FPS: {safe_fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")
        log_file.write(f"MediaPipe Pose Configuration: {pose_config}\n")

        # Advanced filtering information
        if pose_config.get("enable_advanced_filtering", False):
            log_file.write("Advanced Filtering: Enabled\n")
            log_file.write(f"Interpolation Method: {pose_config.get('interp_method', 'none')}\n")
            log_file.write(f"Smoothing Method: {pose_config.get('smooth_method', 'none')}\n")
            log_file.write(f"Maximum Gap Size: {pose_config.get('max_gap', 0)} frames\n")
            if pose_config.get("smooth_params"):
                log_file.write(f"Smoothing Parameters: {pose_config['smooth_params']}\n")
        else:
            log_file.write("Advanced Filtering: Disabled\n")

        if frames_with_missing_data:
            log_file.write(f"Frames with missing data: {len(frames_with_missing_data)}\n")
        else:
            log_file.write("No frames with missing data.\n")

    print(f"\nCompleted processing {video_path.name}")
    print(f"Output saved to: {output_dir}")
    if enable_resize:
        print("Coordinates converted back to original video dimensions")
    print(f"Processing time: {execution_time:.2f} seconds\n")

    print(f"df_norm head:\n{df_norm.head()}")
    print(f"df_pixel head:\n{df_pixel.head()}")
    print(f"df_norm shape: {df_norm.shape}")
    print(f"df_pixel shape: {df_pixel.shape}")

    print(f"DEBUG: enable_padding = {enable_padding}")
    print(f"DEBUG: pad_start_frames = {pad_start_frames}")
    print(f"DEBUG: Total frames before padding removal: {len(normalized_landmarks_list)}")
    print(
        f"DEBUG: Total frames after padding removal: {len(normalized_landmarks_list[pad_start_frames:])}"
    )


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

    # Use existing root or create new one for dialogs
    if existing_root is not None:
        root = existing_root
    else:
        root = tk.Tk()
        root.withdraw()
        # Keep dialogs on top (as before)
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass

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
            process_video(video_file, output_dir, pose_config)
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
