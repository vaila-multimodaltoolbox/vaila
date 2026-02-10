"""
===============================================================================
mpangles.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 31 March 2025
Update Date: 5 February 2026
Version: 0.0.3
Python Version: 3.12.12

Description:
------------
This script calculates absolute and relative angles from landmark coordinates
obtained from MediaPipe pose estimation. It processes CSV files containing
landmark data and generates new CSV files with computed angles.

Key Features:
-------------
1. Absolute Angles:
   - Calculates angles between segments and horizontal axis
   - Uses arctan2 for robust angle calculation

2. Relative Angles:
   - Computes angles between connected segments
   - Uses arctan2 for dot product angle calculation

3. Supported Angles:
    - Elbow angle (between upper arm and forearm)
    - Shoulder angle (between trunk and upper arm)
    - Hip angle (between trunk and thigh)
    - Knee angle (between thigh and shank)
    - Ankle angle (between shank and foot)
    - Wrist angle (between hand and forearm)
    - Neck angle (between mid_shoulder and mid_ear)
    - Trunk angle (between mid_shoulder and mid_hip)

Usage example:
--------------
python mpangles.py

License:
---------
This program is licensed under the AGPL v3.0.
===============================================================================
"""

import argparse
import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import pandas as pd
from rich import print
from scipy import signal

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def select_directory():
    """
    Opens a dialog to select the directory containing CSV files.

    Returns:
    --------
    str or None
        Selected directory path or None if cancelled
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    try:
        directory = filedialog.askdirectory(title="Select Directory with CSV Files", mustexist=True)

        if directory:
            print(f"Selected directory: {directory}")
            return directory
        else:
            print("No directory selected")
            return None

    except Exception as e:
        print(f"Error selecting directory: {str(e)}")
        return None
    finally:
        root.destroy()


# MediaPipe pose uses 33 landmarks (x,y each) = 66 coordinate columns + optional frame column
MIN_MEDIAPIPE_COORD_COLUMNS = 66
MIN_MEDIAPIPE_TOTAL_COLUMNS = 67  # 1 frame + 66 coords (get_vector_landmark assumes col 0 = frame)


def is_mpangles_output_file(filename):
    """
    Return True if filename looks like an output of this module (angle CSVs), so we should not
    re-process it as landmark input.
    """
    if not filename or not filename.lower().endswith(".csv"):
        return False
    base = os.path.basename(filename)
    if base.startswith("processed_"):
        return True
    if "_rel.csv" in base.lower() or "_abs_180" in base.lower() or "_abs_360" in base.lower():
        return True
    if base.endswith("_rel.csv") or "_abs" in base and ".csv" in base:
        return True
    return False


def validate_mediapipe_csv(file_path):
    """
    Check that the CSV has enough columns for MediaPipe landmark data (66+ coord columns).
    Returns (True, None) if valid, or (False, reason_string) if invalid.
    """
    try:
        df = pd.read_csv(file_path, nrows=1)
    except Exception as e:
        return False, f"Could not read CSV: {e}"
    ncols = df.shape[1]
    if ncols < MIN_MEDIAPIPE_TOTAL_COLUMNS:
        return False, (
            f"CSV has {ncols} columns; MediaPipe landmark input needs at least "
            f"{MIN_MEDIAPIPE_TOTAL_COLUMNS} columns (frame + 33 landmarks × 2). "
            "This file may be an angle output or another format."
        )
    return True, None


def process_directory(directory_path=None, filter_config=None):
    """
    Process all CSV files in the given directory.
    If no directory is provided, opens a dialog to select one.

    Parameters:
    -----------
    directory_path : str, optional
        Path to the directory containing CSV files
    filter_config : dict, optional
        Filtering configuration parameters

    Returns:
    --------
    dict
        Dictionary with filenames as keys and processed data as values
    """
    # If no directory provided, ask user to select one
    if directory_path is None:
        directory_path = select_directory()
        if directory_path is None:
            return None

    # Dictionary to store results
    processed_files = {}

    # Get all CSV files in the directory, excluding our own angle outputs
    all_csv = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    csv_files = [f for f in all_csv if not is_mpangles_output_file(f)]

    skipped_outputs = len(all_csv) - len(csv_files)
    if skipped_outputs:
        print(f"Skipped {skipped_outputs} angle output file(s) (processed_*, *_rel, *_abs_*).")

    if not csv_files:
        print(f"No valid input CSV files found in {directory_path}")
        messagebox.showwarning(
            "No Files",
            "No valid input CSV files found. Angle outputs (processed_*, *_rel, *_abs_*) are ignored.",
        )
        return None

    print(f"Found {len(csv_files)} CSV file(s) to process")

    # Create output directory for processed files
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(directory_path, f"processed_angles_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)

        valid, reason = validate_mediapipe_csv(file_path)
        if not valid:
            print(f"Skipping {csv_file}: {reason}")
            continue

        print(f"\nProcessing {csv_file}...")
        output_path = os.path.join(output_dir, f"processed_{os.path.splitext(csv_file)[0]}")
        processed_files[csv_file] = {
            "input_path": file_path,
            "output_path": output_path,
        }

    return processed_files


# Video extensions for directory batch
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def process_directory_videos(directory_path, filter_config=None):
    """
    Process all videos in the directory in batch.
    For each video, finds a CSV with similar name (same dir) and runs
    process_video_with_visualization (skeleton + angles overlay).
    """
    if directory_path is None:
        directory_path = select_directory()
        if directory_path is None:
            return None

    videos = [f for f in os.listdir(directory_path) if f.lower().endswith(VIDEO_EXTENSIONS)]
    if not videos:
        return None

    results = []
    for video_file in sorted(videos):
        video_path = os.path.join(directory_path, video_file)
        video_basename = os.path.splitext(video_file)[0]
        # Find CSV matching this video; prefer vaila format (_pixel_vaila.csv)
        candidates = []
        for f in os.listdir(directory_path):
            if not f.endswith(".csv") or is_mpangles_output_file(f):
                continue
            csv_base = os.path.splitext(f)[0]
            if (
                csv_base == video_basename
                or csv_base.startswith(video_basename + ".")
                or f.startswith(video_basename)
            ):
                full = os.path.join(directory_path, f)
                valid, _ = validate_mediapipe_csv(full)
                if valid:
                    candidates.append(full)
        # Prefer _pixel_vaila.csv (vaila format), then _mp_norm, _mp_pixel, etc.
        csv_path = None
        for suf in ["_pixel_vaila.csv", "_mp_norm.csv", "_mp_pixel.csv", "_mp_data.csv"]:
            for c in candidates:
                if c.endswith(suf) or os.path.basename(c).endswith(suf):
                    csv_path = c
                    break
            if csv_path:
                break
        if csv_path is None and candidates:
            csv_path = candidates[0]
        if csv_path is None:
            print(f"Skipping {video_file}: no valid landmark CSV found with matching name.")
            continue
        try:
            out_dir = process_video_with_visualization(
                video_path, csv_path=csv_path, output_dir=None, filter_config=filter_config
            )
            if out_dir:
                results.append((video_file, out_dir))
            else:
                results.append((video_file, None))
        except Exception as e:
            print(f"Error processing {video_file}: {e}")
    return results


def get_vector_landmark(data, landmark):
    """
    Returns the x,y coordinates for a specific landmark from the data.
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        The input data array with shape (n_frames, n_columns)
        0 to 32 are the landmark indices
        First column is frame number, followed by p1_x,p1_y,p2_x,p2_y,...
    landmark : str
        The name of the landmark to extract (e.g., "nose", "left_shoulder", etc.)
    0 - nose
    1 - left eye (inner)
    2 - left eye
    3 - left eye (outer)
    4 - right eye (inner)
    5 - right eye
    6 - right eye (outer)
    7 - left ear
    8 - right ear
    9 - mouth (left)
    10 - mouth (right)
    11 - left shoulder
    12 - right shoulder
    13 - left elbow
    14 - right elbow
    15 - left wrist
    16 - right wrist
    17 - left pinky
    18 - right pinky
    19 - left index
    20 - right index
    21 - left thumb
    22 - right thumb
    23 - left hip
    24 - right hip
    25 - left knee
    26 - right knee
    27 - left ankle
    28 - right ankle
    29 - left heel
    30 - right heel
    31 - left foot index
    32 - right foot index
    Returns:
    --------
    numpy.ndarray
        Array of shape (n_frames, 2) with x,y coordinates for the requested landmark
    """
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

    # Find the index of the landmark in the list (0-based)
    try:
        landmark_idx = landmark_names.index(landmark)
    except ValueError:
        raise ValueError(f"Landmark '{landmark}' not found in the list of valid landmarks")

    # Calculate the column indices for x and y
    # Add 1 to skip the frame column, and multiply by 2 because each landmark has 2 columns
    x_col = (landmark_idx * 2) + 1
    y_col = x_col + 1

    # Convert to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Extract the x,y coordinates
    return data[:, [x_col, y_col]]


def butter_filter(data, fc, fs, order=4, filter_type="low"):
    """
    Apply Butterworth filter to the data.

    Args:
        data: Input data (1D or 2D array)
        fc: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order
        filter_type: 'low' or 'high' or 'band'

    Returns:
        Filtered data
    """
    try:
        nyquist = 0.5 * fs
        normal_cutoff = fc / nyquist

        # Safety check for cutoff
        if normal_cutoff >= 1.0:
            normal_cutoff = 0.99

        if filter_type == "low":
            b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
        else:
            # Simple fallback for now
            b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)

        # Apply filter
        if data.ndim == 1:
            return signal.filtfilt(b, a, data)
        else:
            # Apply along axis 0 (rows/frames)
            return signal.filtfilt(b, a, data, axis=0)
    except Exception as e:
        print(f"Error in Butterworth filter: {e}")
        return data


def filter_landmarks(df, fps=30, cutoff=6.0, order=4):
    """
    Apply Butterworth low-pass filter to all landmark coordinates in the DataFrame.

    Args:
        df: DataFrame with landmark coordinates
        fps: Frames per second (sampling rate)
        cutoff: Cutoff frequency
        order: Filter order

    Returns:
        DataFrame with filtered coordinates
    """
    filtered_df = df.copy()

    # Identify coordinate columns (all except frame number if present)
    # Assuming standard structure where first col might be frame
    start_col = 0
    if "frame" in df.columns[0].lower() or "index" in df.columns[0].lower():
        start_col = 1

    cols_to_filter = df.columns[start_col:]

    print(
        f"Applying Butterworth filter (fc={cutoff}Hz, order={order}) to {len(cols_to_filter)} columns..."
    )

    # Convert to numpy for faster processing
    data = filtered_df[cols_to_filter].values

    # Filter
    filtered_data = butter_filter(data, fc=cutoff, fs=fps, order=order)

    # Assign back
    filtered_df.iloc[:, start_col:] = filtered_data

    return filtered_df


def compute_midpoint(p1, p2):
    """
    Compute the midpoint between two 2D points.

    Args:
        p1: First point (2D or 3D vector)
        p2: Second point (2D or 3D vector)

    Returns:
        Midpoint as a numpy array
    """
    return (np.array(p1) + np.array(p2)) / 2


def compute_absolute_angle(p_proximal, p_distal, format_360=False):
    """
    Calculate the absolute angle (in degrees) between two points.

    Args:
        p_proximal: Proximal point (x, y)
        p_distal: Distal point (x, y)
        format_360: Boolean to control the angle format:
                   - True: returns angle in [0, 360) degree range
                   - False: returns angle in [-180, 180) degree range (default)

    Returns:
        absolute_angle (float): The computed absolute angle in degrees, formatted
                               according to the format_360 parameter.
    """
    dx = p_distal[0] - p_proximal[0]
    dy = p_distal[1] - p_proximal[1]
    angle = np.degrees(np.arctan2(dy, -dx))

    # Format angle based on user preference
    if format_360:
        absolute_angle = angle % 360  # [0, 360) format
    else:
        absolute_angle = (angle + 180) % 360 - 180  # [-180, 180) format

    return absolute_angle


def compute_relative_angle(a, b, c):
    """
    Compute the angle (in degrees) between two vectors defined by three points.

    This function calculates the angle at the middle point "b" between the vector from b to a
    and the vector from b to c. In other words, it computes the angle ∠ABC.

    The steps involved are:

    1. Convert the input points to NumPy arrays (if they are not already) and compute the vectors:
       - vector_ab = a - b, which points from b to a.
       - vector_cb = c - b, which points from b to c.

    2. Normalize both vectors by computing their Euclidean norms (lengths). If either norm is
       zero (i.e., if b coincides with a or c), the function returns 0.0 degrees to avoid
       division by zero.

    3. Calculate the dot product of the two normalized vectors.

    4. Clamp the dot product to the range [-1.0, 1.0] using np.clip to safeguard against
       possible floating-point inaccuracies that could take the value slightly outside this
       domain.

    5. Compute the angle (in radians) using np.arccos of the clamped dot product.

    6. Convert the angle from radians to degrees.

    Returns:
        angle_deg (float): The computed relative angle in degrees.
    """
    # Calculate vectors
    vector_ab = np.array(a) - np.array(b)
    vector_cb = np.array(c) - np.array(b)

    # Normalize vectors
    norm_ab = np.linalg.norm(vector_ab)
    norm_cb = np.linalg.norm(vector_cb)

    if norm_ab == 0 or norm_cb == 0:
        return 0.0

    vector_ab_normalized = vector_ab / norm_ab
    vector_cb_normalized = vector_cb / norm_cb

    # Calculate dot product
    dot_product = np.dot(vector_ab_normalized, vector_cb_normalized)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_knee_angle(hip, knee, ankle):
    """
    Compute the knee angle using thigh vector (hip-knee) and shank vector (ankle-knee).

    Args:
        hip: Hip point (2D or 3D vector)
        knee: Knee point (2D or 3D vector)
        ankle: Ankle point (2D or 3D vector)

    Returns:
        Knee angle in degrees
    """
    # Calculate thigh vector (hip to knee)
    thigh_vector = np.array(hip) - np.array(knee)

    # Calculate shank vector (ankle to knee)
    shank_vector = np.array(ankle) - np.array(knee)

    # Normalize vectors
    thigh_norm = np.linalg.norm(thigh_vector)
    shank_norm = np.linalg.norm(shank_vector)

    if thigh_norm == 0 or shank_norm == 0:
        return 0.0

    thigh_vector_normalized = thigh_vector / thigh_norm
    shank_vector_normalized = shank_vector / shank_norm

    # Calculate dot product
    dot_product = np.dot(thigh_vector_normalized, shank_vector_normalized)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    knee_angle_deg = np.degrees(angle_rad)

    return knee_angle_deg


def compute_hip_angle(hip, knee, trunk_vector):
    """
    Compute the hip angle using thigh vector (knee-hip) and trunk vector.

    Args:
        hip: Hip point (2D or 3D vector)
        knee: Knee point (2D or 3D vector)
        trunk_vector: Normalized trunk vector

    Returns:
        Hip angle in degrees
    """
    # Calculate thigh vector (knee to hip)
    thigh_vector = np.array(hip) - np.array(knee)

    # Normalize thigh vector
    thigh_norm = np.linalg.norm(thigh_vector)

    if thigh_norm == 0:
        return 0.0

    thigh_vector_normalized = thigh_vector / thigh_norm

    # Calculate dot product
    dot_product = np.dot(thigh_vector_normalized, trunk_vector)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_ankle_angle(knee, ankle, foot_index, heel):
    """
    Compute the ankle angle using shank vector (knee-ankle) and foot vector (foot_index-heel).

    Args:
        knee: Knee point (3D vector)
        ankle: Ankle point (3D vector)
        foot_index: Foot index point (3D vector)
        heel: Heel point (3D vector)

    Returns:
        Ankle angle in degrees
    """
    # Calculate shank vector (knee to ankle)
    shank_vector = np.array(knee) - np.array(ankle)

    # Calculate foot vector (foot_index to heel)
    foot_vector = np.array(foot_index) - np.array(heel)

    # Normalize vectors
    shank_norm = np.linalg.norm(shank_vector)
    foot_norm = np.linalg.norm(foot_vector)

    if shank_norm == 0 or foot_norm == 0:
        return 0.0

    shank_vector_normalized = shank_vector / shank_norm
    foot_vector_normalized = foot_vector / foot_norm

    # Calculate dot product
    dot_product = np.dot(foot_vector_normalized, shank_vector_normalized)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_shoulder_angle(shoulder, elbow, trunk_vector):
    """
    Compute the shoulder angle using upper arm vector (elbow-shoulder) and trunk vector.

    Args:
        shoulder: Shoulder point (2D vector)
        elbow: Elbow point (2D vector)
        trunk_vector: Normalized trunk vector (2D)

    Returns:
        Shoulder angle in degrees
    """
    # Calculate upper arm vector (elbow to shoulder)
    upper_arm_vector = np.array(elbow) - np.array(shoulder)

    # Normalize upper arm vector
    upper_arm_norm = np.linalg.norm(upper_arm_vector)

    if upper_arm_norm == 0:
        return 0.0

    upper_arm_normalized = upper_arm_vector / upper_arm_norm

    # Calculate dot product
    dot_product = np.dot(upper_arm_normalized, trunk_vector)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_elbow_angle(shoulder, elbow, wrist):
    """
    Compute the elbow angle using forearm vector (wrist-elbow) and upper arm vector (shoulder-elbow).

    Args:
        shoulder: Shoulder point (2D vector)
        elbow: Elbow point (2D vector)
        wrist: Wrist point (2D vector)

    Returns:
        Elbow angle in degrees
    """
    # Calculate forearm vector (wrist to elbow)
    forearm_vector = np.array(wrist) - np.array(elbow)

    # Calculate upper arm vector (shoulder to elbow)
    upper_arm_vector = np.array(shoulder) - np.array(elbow)

    # Normalize vectors
    forearm_norm = np.linalg.norm(forearm_vector)
    upper_arm_norm = np.linalg.norm(upper_arm_vector)

    if forearm_norm == 0 or upper_arm_norm == 0:
        return 0.0

    forearm_normalized = forearm_vector / forearm_norm
    upper_arm_normalized = upper_arm_vector / upper_arm_norm

    # Calculate dot product
    dot_product = np.dot(forearm_normalized, upper_arm_normalized)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    elbow_angle_deg = np.degrees(angle_rad)

    return elbow_angle_deg


def compute_neck_angle(mid_ear, mid_shoulder, trunk_vector):
    """
    Compute the neck angle using mid_ear vector (mid_shoulder to mid_ear) and trunk vector.

    Args:
        mid_ear: Mid ear point (2D vector)
        mid_shoulder: Mid shoulder point (2D vector)
        trunk_vector: Normalized trunk vector (2D)

    Returns:
        Neck angle in degrees
    """
    # Calculate head vector (mid_shoulder to mid_ear)
    head_vector = np.array(mid_ear) - np.array(mid_shoulder)

    # Normalize head vector
    head_norm = np.linalg.norm(head_vector)

    if head_norm == 0:
        return 0.0

    head_normalized = head_vector / head_norm

    # Calculate dot product
    dot_product = np.dot(head_normalized, trunk_vector)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    neck_angle_deg = np.degrees(angle_rad)

    return neck_angle_deg


def compute_wrist_angle(elbow, wrist, pinky, index):
    """
    Compute the wrist angle using hand vector (mid_hand-wrist) and forearm vector (elbow-wrist).

    Args:
        elbow: Elbow point (2D vector)
        wrist: Wrist point (2D vector)
        pinky: Pinky finger point (2D vector)
        index: Index finger point (2D vector)

    Returns:
        Wrist angle in degrees
    """
    # Calculate mid_hand point from pinky and index
    mid_hand = compute_midpoint(pinky, index)

    # Calculate hand vector (mid_hand to wrist)
    hand_vector = np.array(mid_hand) - np.array(wrist)

    # Calculate forearm vector (elbow to wrist)
    forearm_vector = np.array(elbow) - np.array(wrist)

    # Normalize vectors
    hand_norm = np.linalg.norm(hand_vector)
    forearm_norm = np.linalg.norm(forearm_vector)

    if hand_norm == 0 or forearm_norm == 0:
        return 0.0

    hand_normalized = hand_vector / hand_norm
    forearm_normalized = forearm_vector / forearm_norm

    # Calculate dot product
    dot_product = np.dot(hand_normalized, forearm_normalized)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    wrist_angle_deg = np.degrees(angle_rad)

    return wrist_angle_deg


def process_absolute_angles(input_csv, output_csv):
    """
    Process landmark data and compute absolute angles for all segments.
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_csv)
        print(f"Reading input file: {input_csv}")

        # Get total number of frames
        total_frames = len(df)
        frame_count = 0

        # Extract all landmarks
        # Right side landmarks
        right_shoulder = get_vector_landmark(df, "right_shoulder")
        right_elbow = get_vector_landmark(df, "right_elbow")
        right_wrist = get_vector_landmark(df, "right_wrist")
        right_hip = get_vector_landmark(df, "right_hip")
        right_knee = get_vector_landmark(df, "right_knee")
        right_ankle = get_vector_landmark(df, "right_ankle")
        right_heel = get_vector_landmark(df, "right_heel")
        right_foot_index = get_vector_landmark(df, "right_foot_index")
        right_pinky = get_vector_landmark(df, "right_pinky")
        right_index = get_vector_landmark(df, "right_index")
        right_ear = get_vector_landmark(df, "right_ear")

        # Left side landmarks
        left_shoulder = get_vector_landmark(df, "left_shoulder")
        left_elbow = get_vector_landmark(df, "left_elbow")
        left_wrist = get_vector_landmark(df, "left_wrist")
        left_hip = get_vector_landmark(df, "left_hip")
        left_knee = get_vector_landmark(df, "left_knee")
        left_ankle = get_vector_landmark(df, "left_ankle")
        left_heel = get_vector_landmark(df, "left_heel")
        left_foot_index = get_vector_landmark(df, "left_foot_index")
        left_pinky = get_vector_landmark(df, "left_pinky")
        left_index = get_vector_landmark(df, "left_index")
        left_ear = get_vector_landmark(df, "left_ear")

        # Get landmarks and calculate midpoints
        mid_ear = [compute_midpoint(left, right) for left, right in zip(left_ear, right_ear)]
        mid_shoulder = [
            compute_midpoint(left, right) for left, right in zip(left_shoulder, right_shoulder)
        ]
        mid_hip = [compute_midpoint(left, right) for left, right in zip(left_hip, right_hip)]

        # Calculate absolute angles for segments
        right_thigh_angles = np.array(
            [compute_absolute_angle(hip, knee) for hip, knee in zip(right_hip, right_knee)]
        )
        right_shank_angles = np.array(
            [compute_absolute_angle(knee, ankle) for knee, ankle in zip(right_knee, right_ankle)]
        )
        right_foot_angles = np.array(
            [
                compute_absolute_angle(heel, foot_index)
                for heel, foot_index in zip(right_heel, right_foot_index)
            ]
        )
        right_upperarm_angles = np.array(
            [
                compute_absolute_angle(shoulder, elbow)
                for shoulder, elbow in zip(right_shoulder, right_elbow)
            ]
        )
        right_forearm_angles = np.array(
            [compute_absolute_angle(elbow, wrist) for elbow, wrist in zip(right_elbow, right_wrist)]
        )
        right_hand_angles = np.array(
            [
                compute_absolute_angle(wrist, mid_hand)
                for wrist, mid_hand in zip(
                    right_wrist,
                    [compute_midpoint(p, i) for p, i in zip(right_pinky, right_index)],
                )
            ]
        )

        left_thigh_angles = np.array(
            [compute_absolute_angle(hip, knee) for hip, knee in zip(left_hip, left_knee)]
        )
        left_shank_angles = np.array(
            [compute_absolute_angle(knee, ankle) for knee, ankle in zip(left_knee, left_ankle)]
        )
        left_foot_angles = np.array(
            [
                compute_absolute_angle(heel, foot_index)
                for heel, foot_index in zip(left_heel, left_foot_index)
            ]
        )
        left_upperarm_angles = np.array(
            [
                compute_absolute_angle(shoulder, elbow)
                for shoulder, elbow in zip(left_shoulder, left_elbow)
            ]
        )
        left_forearm_angles = np.array(
            [compute_absolute_angle(elbow, wrist) for elbow, wrist in zip(left_elbow, left_wrist)]
        )
        left_hand_angles = np.array(
            [
                compute_absolute_angle(wrist, mid)
                for wrist, mid in zip(
                    left_wrist,
                    [compute_midpoint(p, i) for p, i in zip(left_pinky, left_index)],
                )
            ]
        )

        trunk_angles = np.array(
            [compute_absolute_angle(shoulder, hip) for shoulder, hip in zip(mid_shoulder, mid_hip)]
        )
        neck_angles = np.array(
            [compute_absolute_angle(ear, shoulder) for ear, shoulder in zip(mid_ear, mid_shoulder)]
        )

        # Create landmarks dictionary
        landmarks = {
            "right_ear": right_ear,
            "left_ear": left_ear,
            "mid_ear": mid_ear,
            "mid_shoulder": mid_shoulder,
            "mid_hip": mid_hip,
            "right_shoulder": right_shoulder,
            "right_elbow": right_elbow,
            "right_wrist": right_wrist,
            "right_hip": right_hip,
            "right_knee": right_knee,
            "right_ankle": right_ankle,
            "right_heel": right_heel,
            "right_foot_index": right_foot_index,
            "right_pinky": right_pinky,
            "right_index": right_index,
            "left_shoulder": left_shoulder,
            "left_elbow": left_elbow,
            "left_wrist": left_wrist,
            "left_hip": left_hip,
            "left_knee": left_knee,
            "left_ankle": left_ankle,
            "left_heel": left_heel,
            "left_foot_index": left_foot_index,
            "left_pinky": left_pinky,
            "left_index": left_index,
        }

        # Calculate angles
        angles = {
            # Right side
            "right_thigh_abs": right_thigh_angles,
            "right_shank_abs": right_shank_angles,
            "right_foot_abs": right_foot_angles,
            "right_upperarm_abs": right_upperarm_angles,
            "right_forearm_abs": right_forearm_angles,
            "right_hand_abs": right_hand_angles,
            # Left side
            "left_thigh_abs": left_thigh_angles,
            "left_shank_abs": left_shank_angles,
            "left_foot_abs": left_foot_angles,
            "left_upperarm_abs": left_upperarm_angles,
            "left_forearm_abs": left_forearm_angles,
            "left_hand_abs": left_hand_angles,
            # Central segments
            "trunk_abs": trunk_angles,
            "neck_abs": neck_angles,
        }

        # Create DataFrame with angles
        angles_df = pd.DataFrame(angles)
        angles_df.insert(0, "frame_index", df.iloc[:, 0])  # Add frame index as first column

        # Save to CSV
        angles_df.to_csv(output_csv, index=False, float_format="%.2f")
        print(f"\nAngles saved to: {output_csv}")

    except Exception as e:
        print(f"Error processing absolute angles: {str(e)}")
        raise


def process_angles(input_csv, output_csv, filter_config=None, video_dims=None):
    """
    Process landmark data and compute specified angles.
    Autmatically computes both -180..180 and 0..360 formats for absolute angles.

    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Base path for output CSV files
        filter_config (dict): Dictionary with filtering parameters (enabled, cutoff, order, fps)
        video_dims (tuple): (width, height) used to un-normalize coordinates if needed.

    Returns:
        tuple: (rel_df, abs180_df, abs360_df)
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_csv)
        print(f"Reading input file: {input_csv}")

        # Validate: require MediaPipe landmark layout (frame + 66 coord columns)
        ncols = df.shape[1]
        if ncols < MIN_MEDIAPIPE_TOTAL_COLUMNS:
            raise ValueError(
                f"This CSV has {ncols} columns. MediaPipe landmark input must have at least "
                f"{MIN_MEDIAPIPE_TOTAL_COLUMNS} columns (frame + 33 landmarks × 2). "
                "If this is an angle output file (_rel, _abs_180, _abs_360) or another format, "
                "please use a raw landmark CSV."
            )

        # Apply filtering if enabled
        if filter_config and filter_config.get("enabled", False):
            fps = filter_config.get("fps", 30)
            cutoff = filter_config.get("cutoff", 6.0)
            order = filter_config.get("order", 4)
            print(f"Applying Butterworth filter: fc={cutoff}Hz, order={order}, fs={fps}fps")
            df = filter_landmarks(df, fps=fps, cutoff=cutoff, order=order)

        # Handle coordinate scaling (Normalized -> Pixel)
        # Check if data appears normalized (all values <= 1.0)
        # We check columns 1 onwards (skipping frame/index)
        start_col = 0
        if "frame" in df.columns[0].lower() or "index" in df.columns[0].lower():
            start_col = 1

        # Check first row samples (or max of numeric cols)
        numeric_cols = df.iloc[:, start_col:]
        max_val = numeric_cols.max().max()

        is_normalized = max_val <= 1.5  # 1.5 to be safe (sometimes slightly >1.0)

        if is_normalized and video_dims:
            w, h = video_dims
            print(f"Detected normalized coordinates. Scaling to {w}x{h}...")

            # Identify x and y columns
            # Assuming format: p0_x, p0_y, p1_x, p1_y ...
            # x cols are start_col, start_col+2 ...
            # y cols are start_col+1, start_col+3 ...

            # Simple iteration to scale
            for col_idx in range(start_col, len(df.columns)):
                col_name = df.columns[col_idx]
                # Heuristic: ends with _x or is even/odd position?
                # Using the logic from get_vector_landmark, x is col 1, 3, 5... (if 0 is frame)
                # But here start_col handles the offset.
                # relative index
                rel_idx = col_idx - start_col
                if rel_idx % 2 == 0:  # X column
                    df.iloc[:, col_idx] = df.iloc[:, col_idx] * w
                else:  # Y column
                    df.iloc[:, col_idx] = df.iloc[:, col_idx] * h

            print("Coordinates scaled to pixel space.")
        elif is_normalized and not video_dims:
            print("Warning: Data appears normalized (0-1) but no video dimensions provided.")
            print("Angles may be distorted if aspect ratio is not 1:1.")

        # Get total number of frames
        total_frames = len(df)

        # Extract all landmarks needed for angle calculations
        # Right side landmarks
        right_shoulder = get_vector_landmark(df, "right_shoulder")
        right_elbow = get_vector_landmark(df, "right_elbow")
        right_wrist = get_vector_landmark(df, "right_wrist")
        right_hip = get_vector_landmark(df, "right_hip")
        right_knee = get_vector_landmark(df, "right_knee")
        right_ankle = get_vector_landmark(df, "right_ankle")
        right_foot_index = get_vector_landmark(df, "right_foot_index")
        right_heel = get_vector_landmark(df, "right_heel")
        right_pinky = get_vector_landmark(df, "right_pinky")
        right_index = get_vector_landmark(df, "right_index")
        right_ear = get_vector_landmark(df, "right_ear")

        # Left side landmarks
        left_shoulder = get_vector_landmark(df, "left_shoulder")
        left_elbow = get_vector_landmark(df, "left_elbow")
        left_wrist = get_vector_landmark(df, "left_wrist")
        left_hip = get_vector_landmark(df, "left_hip")
        left_knee = get_vector_landmark(df, "left_knee")
        left_ankle = get_vector_landmark(df, "left_ankle")
        left_foot_index = get_vector_landmark(df, "left_foot_index")
        left_heel = get_vector_landmark(df, "left_heel")
        left_pinky = get_vector_landmark(df, "left_pinky")
        left_index = get_vector_landmark(df, "left_index")
        left_ear = get_vector_landmark(df, "left_ear")

        # Get landmarks and calculate midpoints
        mid_ear = [compute_midpoint(left, right) for left, right in zip(left_ear, right_ear)]
        mid_shoulder = [
            compute_midpoint(left, right) for left, right in zip(left_shoulder, right_shoulder)
        ]
        mid_hip = [compute_midpoint(left, right) for left, right in zip(left_hip, right_hip)]

        # Calculate trunk vectors for all frames
        trunk_vectors = []
        for i in range(len(mid_hip)):
            trunk_vector = np.array(mid_hip[i]) - np.array(mid_shoulder[i])
            trunk_norm = np.linalg.norm(trunk_vector)
            if trunk_norm > 0:
                trunk_vector = trunk_vector / trunk_norm
            trunk_vectors.append(trunk_vector)

        # Calculate relative angles
        right_shoulder_angles = []
        right_elbow_angles = []
        right_hip_angles = []
        right_knee_angles = []
        right_ankle_angles = []
        right_wrist_angles = []
        left_shoulder_angles = []
        left_elbow_angles = []
        left_hip_angles = []
        left_knee_angles = []
        left_ankle_angles = []
        left_wrist_angles = []
        neck_angles = []
        trunk_angles = []

        # Calculate absolute angles (Lists for 180 and 360)
        # We will compute both for each segment
        abs_angles_180 = {}
        abs_angles_360 = {}

        segments = [
            "right_thigh",
            "right_shank",
            "right_foot",
            "right_upperarm",
            "right_forearm",
            "right_hand",
            "left_thigh",
            "left_shank",
            "left_foot",
            "left_upperarm",
            "left_forearm",
            "left_hand",
            "trunk",
            "neck",
        ]

        for seg in segments:
            abs_angles_180[f"{seg}_abs"] = []
            abs_angles_360[f"{seg}_abs"] = []

        # Process each frame
        for i in range(total_frames):
            try:
                # Relative angles
                # Right side
                right_shoulder_angles.append(
                    compute_shoulder_angle(right_shoulder[i], right_elbow[i], trunk_vectors[i])
                )
                right_elbow_angles.append(
                    compute_elbow_angle(right_shoulder[i], right_elbow[i], right_wrist[i])
                )
                right_hip_angles.append(
                    compute_hip_angle(right_hip[i], right_knee[i], trunk_vectors[i])
                )
                right_knee_angles.append(
                    compute_knee_angle(right_hip[i], right_knee[i], right_ankle[i])
                )
                right_ankle_angles.append(
                    compute_ankle_angle(
                        right_knee[i], right_ankle[i], right_foot_index[i], right_heel[i]
                    )
                )
                try:
                    right_wrist_angles.append(
                        compute_wrist_angle(
                            right_elbow[i], right_wrist[i], right_pinky[i], right_index[i]
                        )
                    )
                except:
                    right_wrist_angles.append(np.nan)

                # Left side
                left_shoulder_angles.append(
                    compute_shoulder_angle(left_shoulder[i], left_elbow[i], trunk_vectors[i])
                )
                left_elbow_angles.append(
                    compute_elbow_angle(left_shoulder[i], left_elbow[i], left_wrist[i])
                )
                left_hip_angles.append(
                    compute_hip_angle(left_hip[i], left_knee[i], trunk_vectors[i])
                )
                left_knee_angles.append(
                    compute_knee_angle(left_hip[i], left_knee[i], left_ankle[i])
                )
                left_ankle_angles.append(
                    compute_ankle_angle(
                        left_knee[i], left_ankle[i], left_foot_index[i], left_heel[i]
                    )
                )
                try:
                    left_wrist_angles.append(
                        compute_wrist_angle(
                            left_elbow[i], left_wrist[i], left_pinky[i], left_index[i]
                        )
                    )
                except:
                    left_wrist_angles.append(np.nan)

                # Central segments relative angles
                neck_angles.append(
                    compute_neck_angle(mid_ear[i], mid_shoulder[i], trunk_vectors[i])
                )
                trunk_angles.append(
                    compute_relative_angle(mid_shoulder[i], mid_hip[i], mid_shoulder[i])
                )

                # Absolute angles (Calculate both formats)
                pairs = {
                    "right_thigh": (right_hip[i], right_knee[i]),
                    "right_shank": (right_knee[i], right_ankle[i]),
                    "right_foot": (right_heel[i], right_foot_index[i]),
                    "right_upperarm": (right_shoulder[i], right_elbow[i]),
                    "right_forearm": (right_elbow[i], right_wrist[i]),
                    "left_thigh": (left_hip[i], left_knee[i]),
                    "left_shank": (left_knee[i], left_ankle[i]),
                    "left_foot": (left_heel[i], left_foot_index[i]),
                    "left_upperarm": (left_shoulder[i], left_elbow[i]),
                    "left_forearm": (left_elbow[i], left_wrist[i]),
                    "trunk": (mid_shoulder[i], mid_hip[i]),
                    "neck": (mid_ear[i], mid_shoulder[i]),
                }

                # Special cases for hand (midpoint)
                try:
                    rh_mid = compute_midpoint(right_pinky[i], right_index[i])
                    pairs["right_hand"] = (right_wrist[i], rh_mid)
                except:
                    pairs["right_hand"] = None

                try:
                    lh_mid = compute_midpoint(left_pinky[i], left_index[i])
                    pairs["left_hand"] = (left_wrist[i], lh_mid)
                except:
                    pairs["left_hand"] = None

                for name, pts in pairs.items():
                    key = f"{name}_abs"
                    if pts is not None:
                        val180 = compute_absolute_angle(pts[0], pts[1], format_360=False)
                        val360 = compute_absolute_angle(pts[0], pts[1], format_360=True)
                    else:
                        val180 = np.nan
                        val360 = np.nan

                    abs_angles_180[key].append(val180)
                    abs_angles_360[key].append(val360)

                # Show progress
                if (i + 1) % 50 == 0:
                    print(
                        f"Processing frame {i + 1}/{total_frames} ({((i + 1) / total_frames) * 100:.1f}%)"
                    )

            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                # Fill with NaNs if error (simplified for brevity, can be expanded if needed)

        # Create dictionaries
        relative_angles_dict = {
            "frame": df.iloc[:, 0]
            if "frame" in df.columns[0].lower()
            else np.arange(total_frames),  # Use existing or index
            "neck": neck_angles,
            "trunk": trunk_angles,
            "right_shoulder": right_shoulder_angles,
            "right_elbow": right_elbow_angles,
            "right_wrist": right_wrist_angles,
            "right_hip": right_hip_angles,
            "right_knee": right_knee_angles,
            "right_ankle": right_ankle_angles,
            "left_shoulder": left_shoulder_angles,
            "left_elbow": left_elbow_angles,
            "left_wrist": left_wrist_angles,
            "left_hip": left_hip_angles,
            "left_knee": left_knee_angles,
            "left_ankle": left_ankle_angles,
        }

        # Add frame to absolutes
        frame_idx = relative_angles_dict["frame"]
        abs_angles_180_dict = {"frame": frame_idx}
        abs_angles_360_dict = {"frame": frame_idx}

        abs_angles_180_dict.update(abs_angles_180)
        abs_angles_360_dict.update(abs_angles_360)

        # Create DataFrames
        rel_df = pd.DataFrame(relative_angles_dict)
        abs180_df = pd.DataFrame(abs_angles_180_dict)
        abs360_df = pd.DataFrame(abs_angles_360_dict)

        # Generate output names
        output_basename = os.path.splitext(output_csv)[0]

        path_rel = f"{output_basename}_rel.csv"
        path_abs180 = f"{output_basename}_abs_180.csv"
        path_abs360 = f"{output_basename}_abs_360.csv"

        # Save
        rel_df.to_csv(path_rel, index=False, float_format="%.2f")
        abs180_df.to_csv(path_abs180, index=False, float_format="%.2f")
        abs360_df.to_csv(path_abs360, index=False, float_format="%.2f")

        print(f"\nSaved relative angles: {path_rel}")
        print(f"Saved absolute angles (180): {path_abs180}")
        print(f"Saved absolute angles (360): {path_abs360}")

        return df, rel_df, abs180_df, abs360_df

    except Exception as e:
        print(f"Error processing angles: {str(e)}")
        import traceback

        traceback.print_exc()
        raise
    try:
        # Read CSV file
        df = pd.read_csv(input_csv)
        print(f"Reading input file: {input_csv}")

        # Get total number of frames
        total_frames = len(df)
        frame_count = 0

        # Extract all landmarks needed for angle calculations
        # Right side landmarks
        right_shoulder = get_vector_landmark(df, "right_shoulder")
        right_elbow = get_vector_landmark(df, "right_elbow")
        right_wrist = get_vector_landmark(df, "right_wrist")
        right_hip = get_vector_landmark(df, "right_hip")
        right_knee = get_vector_landmark(df, "right_knee")
        right_ankle = get_vector_landmark(df, "right_ankle")
        right_foot_index = get_vector_landmark(df, "right_foot_index")
        right_heel = get_vector_landmark(df, "right_heel")
        right_pinky = get_vector_landmark(df, "right_pinky")
        right_index = get_vector_landmark(df, "right_index")
        right_ear = get_vector_landmark(df, "right_ear")

        # Left side landmarks
        left_shoulder = get_vector_landmark(df, "left_shoulder")
        left_elbow = get_vector_landmark(df, "left_elbow")
        left_wrist = get_vector_landmark(df, "left_wrist")
        left_hip = get_vector_landmark(df, "left_hip")
        left_knee = get_vector_landmark(df, "left_knee")
        left_ankle = get_vector_landmark(df, "left_ankle")
        left_foot_index = get_vector_landmark(df, "left_foot_index")
        left_heel = get_vector_landmark(df, "left_heel")
        left_pinky = get_vector_landmark(df, "left_pinky")
        left_index = get_vector_landmark(df, "left_index")
        left_ear = get_vector_landmark(df, "left_ear")

        # Get landmarks and calculate midpoints
        mid_ear = [compute_midpoint(left, right) for left, right in zip(left_ear, right_ear)]
        mid_shoulder = [
            compute_midpoint(left, right) for left, right in zip(left_shoulder, right_shoulder)
        ]
        mid_hip = [compute_midpoint(left, right) for left, right in zip(left_hip, right_hip)]

        # Calculate trunk vectors for all frames
        trunk_vectors = []
        for i in range(len(mid_hip)):
            trunk_vector = np.array(mid_hip[i]) - np.array(mid_shoulder[i])
            trunk_norm = np.linalg.norm(trunk_vector)
            if trunk_norm > 0:
                trunk_vector = trunk_vector / trunk_norm
            trunk_vectors.append(trunk_vector)

        # Calculate relative angles for all frames
        right_shoulder_angles = []
        right_elbow_angles = []
        right_hip_angles = []
        right_knee_angles = []
        right_ankle_angles = []
        right_wrist_angles = []
        left_shoulder_angles = []
        left_elbow_angles = []
        left_hip_angles = []
        left_knee_angles = []
        left_ankle_angles = []
        left_wrist_angles = []
        neck_angles = []
        trunk_angles = []

        # Calculate absolute angles for all frames
        right_thigh_abs_angles = []
        right_shank_abs_angles = []
        right_foot_abs_angles = []
        right_upperarm_abs_angles = []
        right_forearm_abs_angles = []
        right_hand_abs_angles = []
        left_thigh_abs_angles = []
        left_shank_abs_angles = []
        left_foot_abs_angles = []
        left_upperarm_abs_angles = []
        left_forearm_abs_angles = []
        left_hand_abs_angles = []
        trunk_abs_angles = []
        neck_abs_angles = []

        # Process each frame
        for i in range(total_frames):
            try:
                # Relative angles
                # Right side
                right_shoulder_angles.append(
                    compute_shoulder_angle(right_shoulder[i], right_elbow[i], trunk_vectors[i])
                )
                right_elbow_angles.append(
                    compute_elbow_angle(right_shoulder[i], right_elbow[i], right_wrist[i])
                )
                right_hip_angles.append(
                    compute_hip_angle(right_hip[i], right_knee[i], trunk_vectors[i])
                )
                right_knee_angles.append(
                    compute_knee_angle(right_hip[i], right_knee[i], right_ankle[i])
                )
                right_ankle_angles.append(
                    compute_ankle_angle(
                        right_knee[i],
                        right_ankle[i],
                        right_foot_index[i],
                        right_heel[i],
                    )
                )
                try:
                    right_wrist_angles.append(
                        compute_wrist_angle(
                            right_elbow[i],
                            right_wrist[i],
                            right_pinky[i],
                            right_index[i],
                        )
                    )
                except:
                    right_wrist_angles.append(np.nan)

                # Left side
                left_shoulder_angles.append(
                    compute_shoulder_angle(left_shoulder[i], left_elbow[i], trunk_vectors[i])
                )
                left_elbow_angles.append(
                    compute_elbow_angle(left_shoulder[i], left_elbow[i], left_wrist[i])
                )
                left_hip_angles.append(
                    compute_hip_angle(left_hip[i], left_knee[i], trunk_vectors[i])
                )
                left_knee_angles.append(
                    compute_knee_angle(left_hip[i], left_knee[i], left_ankle[i])
                )
                left_ankle_angles.append(
                    compute_ankle_angle(
                        left_knee[i], left_ankle[i], left_foot_index[i], left_heel[i]
                    )
                )
                try:
                    left_wrist_angles.append(
                        compute_wrist_angle(
                            left_elbow[i], left_wrist[i], left_pinky[i], left_index[i]
                        )
                    )
                except:
                    left_wrist_angles.append(np.nan)

                # Central segments relative angles
                neck_angles.append(
                    compute_neck_angle(mid_ear[i], mid_shoulder[i], trunk_vectors[i])
                )
                trunk_angles.append(
                    compute_relative_angle(mid_shoulder[i], mid_hip[i], mid_shoulder[i])
                )

                # Absolute angles
                right_thigh_abs_angles.append(
                    compute_absolute_angle(right_hip[i], right_knee[i], format_360)
                )
                right_shank_abs_angles.append(
                    compute_absolute_angle(right_knee[i], right_ankle[i], format_360)
                )
                right_foot_abs_angles.append(
                    compute_absolute_angle(right_heel[i], right_foot_index[i], format_360)
                )
                right_upperarm_abs_angles.append(
                    compute_absolute_angle(right_shoulder[i], right_elbow[i], format_360)
                )
                right_forearm_abs_angles.append(
                    compute_absolute_angle(right_elbow[i], right_wrist[i], format_360)
                )

                try:
                    right_hand_mid = compute_midpoint(right_pinky[i], right_index[i])
                    right_hand_abs_angles.append(
                        compute_absolute_angle(right_wrist[i], right_hand_mid, format_360)
                    )
                except:
                    right_hand_abs_angles.append(np.nan)

                left_thigh_abs_angles.append(
                    compute_absolute_angle(left_hip[i], left_knee[i], format_360)
                )
                left_shank_abs_angles.append(
                    compute_absolute_angle(left_knee[i], left_ankle[i], format_360)
                )
                left_foot_abs_angles.append(
                    compute_absolute_angle(left_heel[i], left_foot_index[i], format_360)
                )
                left_upperarm_abs_angles.append(
                    compute_absolute_angle(left_shoulder[i], left_elbow[i], format_360)
                )
                left_forearm_abs_angles.append(
                    compute_absolute_angle(left_elbow[i], left_wrist[i], format_360)
                )

                try:
                    left_hand_mid = compute_midpoint(left_pinky[i], left_index[i])
                    left_hand_abs_angles.append(
                        compute_absolute_angle(left_wrist[i], left_hand_mid, format_360)
                    )
                except:
                    left_hand_abs_angles.append(np.nan)

                trunk_abs_angles.append(
                    compute_absolute_angle(mid_shoulder[i], mid_hip[i], format_360)
                )
                neck_abs_angles.append(
                    compute_absolute_angle(mid_ear[i], mid_shoulder[i], format_360)
                )

                # Show progress
                frame_count += 1
                if frame_count % 30 == 0:
                    print(
                        f"Processing frame {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)"
                    )
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                # Fill with zeros if there's an error
                # Relative angles
                if len(right_shoulder_angles) <= i:
                    right_shoulder_angles.append(np.nan)
                if len(right_elbow_angles) <= i:
                    right_elbow_angles.append(np.nan)
                if len(right_hip_angles) <= i:
                    right_hip_angles.append(np.nan)
                if len(right_knee_angles) <= i:
                    right_knee_angles.append(np.nan)
                if len(right_ankle_angles) <= i:
                    right_ankle_angles.append(np.nan)
                if len(right_wrist_angles) <= i:
                    right_wrist_angles.append(np.nan)
                if len(left_shoulder_angles) <= i:
                    left_shoulder_angles.append(np.nan)
                if len(left_elbow_angles) <= i:
                    left_elbow_angles.append(np.nan)
                if len(left_hip_angles) <= i:
                    left_hip_angles.append(np.nan)
                if len(left_knee_angles) <= i:
                    left_knee_angles.append(np.nan)
                if len(left_ankle_angles) <= i:
                    left_ankle_angles.append(np.nan)
                if len(left_wrist_angles) <= i:
                    left_wrist_angles.append(np.nan)
                if len(neck_angles) <= i:
                    neck_angles.append(np.nan)
                if len(trunk_angles) <= i:
                    trunk_angles.append(np.nan)

                # Absolute angles
                if len(right_thigh_abs_angles) <= i:
                    right_thigh_abs_angles.append(np.nan)
                if len(right_shank_abs_angles) <= i:
                    right_shank_abs_angles.append(np.nan)
                if len(right_foot_abs_angles) <= i:
                    right_foot_abs_angles.append(np.nan)
                if len(right_upperarm_abs_angles) <= i:
                    right_upperarm_abs_angles.append(np.nan)
                if len(right_forearm_abs_angles) <= i:
                    right_forearm_abs_angles.append(np.nan)
                if len(right_hand_abs_angles) <= i:
                    right_hand_abs_angles.append(np.nan)
                if len(left_thigh_abs_angles) <= i:
                    left_thigh_abs_angles.append(np.nan)
                if len(left_shank_abs_angles) <= i:
                    left_shank_abs_angles.append(np.nan)
                if len(left_foot_abs_angles) <= i:
                    left_foot_abs_angles.append(np.nan)
                if len(left_upperarm_abs_angles) <= i:
                    left_upperarm_abs_angles.append(np.nan)
                if len(left_forearm_abs_angles) <= i:
                    left_forearm_abs_angles.append(np.nan)
                if len(left_hand_abs_angles) <= i:
                    left_hand_abs_angles.append(np.nan)
                if len(trunk_abs_angles) <= i:
                    trunk_abs_angles.append(np.nan)
                if len(neck_abs_angles) <= i:
                    neck_abs_angles.append(np.nan)

        # Criar dicionários para os ângulos na ordem desejada
        relative_angles_dict = {
            "frame_index": df.iloc[:, 0],
            # Ângulos centrais
            "neck": neck_angles,
            "trunk": trunk_angles,
            # Lado direito
            "right_shoulder": right_shoulder_angles,
            "right_elbow": right_elbow_angles,
            "right_wrist": right_wrist_angles,
            "right_hip": right_hip_angles,
            "right_knee": right_knee_angles,
            "right_ankle": right_ankle_angles,
            # Lado esquerdo
            "left_shoulder": left_shoulder_angles,
            "left_elbow": left_elbow_angles,
            "left_wrist": left_wrist_angles,
            "left_hip": left_hip_angles,
            "left_knee": left_knee_angles,
            "left_ankle": left_ankle_angles,
        }

        absolute_angles_dict = {
            "frame_index": df.iloc[:, 0],
            # Ângulos centrais
            "neck_abs": neck_abs_angles,
            "trunk_abs": trunk_abs_angles,
            # Lado direito
            "right_upperarm_abs": right_upperarm_abs_angles,
            "right_forearm_abs": right_forearm_abs_angles,
            "right_hand_abs": right_hand_abs_angles,
            "right_thigh_abs": right_thigh_abs_angles,
            "right_shank_abs": right_shank_abs_angles,
            "right_foot_abs": right_foot_abs_angles,
            # Lado esquerdo
            "left_upperarm_abs": left_upperarm_abs_angles,
            "left_forearm_abs": left_forearm_abs_angles,
            "left_hand_abs": left_hand_abs_angles,
            "left_thigh_abs": left_thigh_abs_angles,
            "left_shank_abs": left_shank_abs_angles,
            "left_foot_abs": left_foot_abs_angles,
        }

        # Criar DataFrames separados
        relative_angles_df = pd.DataFrame(relative_angles_dict)
        absolute_angles_df = pd.DataFrame(absolute_angles_dict)

        # Gerar nomes para os arquivos de saída
        output_basename = os.path.splitext(output_csv)[0]
        relative_output_path = f"{output_basename}_rel.csv"
        absolute_output_path = f"{output_basename}_abs.csv"

        # Salvar CSVs com os ângulos ordenados
        relative_angles_df.to_csv(relative_output_path, index=False, float_format="%.2f")
        absolute_angles_df.to_csv(absolute_output_path, index=False, float_format="%.2f")

        print(f"\nÂngulos relativos salvos em: {relative_output_path}")
        print(f"Ângulos absolutos salvos em: {absolute_output_path}")

        # Não é necessário salvar o output_csv original, já que estamos criando dois arquivos específicos

    except Exception as e:
        print(f"Error processing angles: {str(e)}")
        raise


def draw_skeleton_enhanced(frame, landmarks, rel_angles, abs_angles, abs_angles_360=None):
    """
    Draw enhanced skeleton and angle values on the frame.
    abs_angles_360: optional dict for right-side absolute angles in 0..360; left uses abs_angles (-180..180).
    """
    height, width = frame.shape[:2]

    # Colors (BGR)
    # Right: Red-ish (Coral)
    C_RIGHT = (80, 80, 255)
    # Left: Blue-ish (Sky Blue)
    C_LEFT = (255, 191, 0)
    # Center: White/Gray
    C_CENTER = (240, 240, 240)
    # Joints: Green (Lime)
    C_JOINT = (0, 255, 0)
    # Text: White with shadow
    C_TEXT = (255, 255, 255)
    C_SHADOW = (10, 10, 10)

    # helper to draw line
    def dline(p1, p2, color, thick=3):
        if np.isnan(p1).any() or np.isnan(p2).any():
            return
        pt1 = (int(p1[0]), int(p1[1]))
        pt2 = (int(p2[0]), int(p2[1]))
        cv2.line(frame, pt1, pt2, color, thick, cv2.LINE_AA)

    # helper to draw circle
    def dcircle(p, color, radius=5):
        if np.isnan(p).any():
            return
        pt = (int(p[0]), int(p[1]))
        cv2.circle(frame, pt, radius, (255, 255, 255), -1, cv2.LINE_AA)  # white border
        cv2.circle(frame, pt, radius - 2, color, -1, cv2.LINE_AA)

    # Calculate mid_hand points if missing
    if "right_mid_hand" not in landmarks:
        landmarks["right_mid_hand"] = compute_midpoint(
            landmarks["right_pinky"], landmarks["right_index"]
        )
    if "left_mid_hand" not in landmarks:
        landmarks["left_mid_hand"] = compute_midpoint(
            landmarks["left_pinky"], landmarks["left_index"]
        )

    # Draw Segments
    # Right Side
    dline(landmarks["right_shoulder"], landmarks["right_elbow"], C_RIGHT)
    dline(landmarks["right_elbow"], landmarks["right_wrist"], C_RIGHT)
    dline(landmarks["right_wrist"], landmarks["right_mid_hand"], C_RIGHT)
    dline(landmarks["right_hip"], landmarks["right_knee"], C_RIGHT)
    dline(landmarks["right_knee"], landmarks["right_ankle"], C_RIGHT)
    dline(landmarks["right_ankle"], landmarks["right_heel"], C_RIGHT)
    dline(landmarks["right_heel"], landmarks["right_foot_index"], C_RIGHT)
    dline(landmarks["right_ankle"], landmarks["right_foot_index"], C_RIGHT)  # Foot top

    # Left Side
    dline(landmarks["left_shoulder"], landmarks["left_elbow"], C_LEFT)
    dline(landmarks["left_elbow"], landmarks["left_wrist"], C_LEFT)
    dline(landmarks["left_wrist"], landmarks["left_mid_hand"], C_LEFT)
    dline(landmarks["left_hip"], landmarks["left_knee"], C_LEFT)
    dline(landmarks["left_knee"], landmarks["left_ankle"], C_LEFT)
    dline(landmarks["left_ankle"], landmarks["left_heel"], C_LEFT)
    dline(landmarks["left_heel"], landmarks["left_foot_index"], C_LEFT)
    dline(landmarks["left_ankle"], landmarks["left_foot_index"], C_LEFT)

    # Center
    dline(landmarks["mid_shoulder"], landmarks["mid_hip"], C_CENTER)
    dline(landmarks["mid_ear"], landmarks["mid_shoulder"], C_CENTER)
    dline(landmarks["right_shoulder"], landmarks["left_shoulder"], C_CENTER, 2)
    dline(landmarks["right_hip"], landmarks["left_hip"], C_CENTER, 2)

    # Draw Joints
    for name, pt in landmarks.items():
        if "mid" in name:
            continue
        if name == "nose":
            continue
        if "eye" in name:
            continue
        dcircle(pt, C_JOINT, 5)

    # Draw angle values at/near each joint (on the figure)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1

    def put_at(pt, txt, color):
        if np.isnan(pt).any():
            return
        ix, iy = int(pt[0]), int(pt[1])
        # Offset so text is beside joint, not on top
        ox, oy = 12, -8
        cv2.putText(frame, txt, (ix + ox, iy + oy), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (ix + ox, iy + oy), font, scale, color, thick, cv2.LINE_AA)

    def ang(v):
        try:
            return f"{float(v):.0f}"
        except (TypeError, ValueError):
            return "—"

    # Relative angles at joints
    put_at(landmarks["right_shoulder"], ang(rel_angles.get("right_shoulder")), C_RIGHT)
    put_at(landmarks["right_elbow"], ang(rel_angles.get("right_elbow")), C_RIGHT)
    put_at(landmarks["right_wrist"], ang(rel_angles.get("right_wrist")), C_RIGHT)
    put_at(landmarks["right_hip"], ang(rel_angles.get("right_hip")), C_RIGHT)
    put_at(landmarks["right_knee"], ang(rel_angles.get("right_knee")), C_RIGHT)
    put_at(landmarks["right_ankle"], ang(rel_angles.get("right_ankle")), C_RIGHT)
    put_at(landmarks["left_shoulder"], ang(rel_angles.get("left_shoulder")), C_LEFT)
    put_at(landmarks["left_elbow"], ang(rel_angles.get("left_elbow")), C_LEFT)
    put_at(landmarks["left_wrist"], ang(rel_angles.get("left_wrist")), C_LEFT)
    put_at(landmarks["left_hip"], ang(rel_angles.get("left_hip")), C_LEFT)
    put_at(landmarks["left_knee"], ang(rel_angles.get("left_knee")), C_LEFT)
    put_at(landmarks["left_ankle"], ang(rel_angles.get("left_ankle")), C_LEFT)
    put_at(landmarks["mid_shoulder"], ang(rel_angles.get("neck")), C_CENTER)
    put_at(landmarks["mid_hip"], ang(rel_angles.get("trunk")), C_CENTER)

    # Draw Angles Text (panel) — didactic layout: all panels in corners, nothing in center
    scale = 0.55
    thick = 2
    step = 22
    margin = 12

    def put_txt(txt, x, y, color):
        cv2.putText(frame, txt, (x + 1, y + 1), font, scale, C_SHADOW, thick + 1, cv2.LINE_AA)
        cv2.putText(frame, txt, (x, y), font, scale, color, thick, cv2.LINE_AA)

    # Corners: left and right, top and bottom (no center text)
    x_left = margin
    x_right = width - 195
    y_top = 28
    y_bottom = height - 400

    # --- TOP-LEFT: RELATIVE (left side of body) ---
    put_txt("RELATIVE", x_left, y_top, C_TEXT)
    yl = y_top + step
    put_txt(f"L Shldr: {rel_angles.get('left_shoulder', 0):.1f}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Elbow: {rel_angles.get('left_elbow', 0):.1f}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Wrist: {rel_angles.get('left_wrist', 0):.1f}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Hip:   {rel_angles.get('left_hip', 0):.1f}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Knee:  {rel_angles.get('left_knee', 0):.1f}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Ankle: {rel_angles.get('left_ankle', 0):.1f}", x_left, yl, C_LEFT)
    yl += step

    # --- TOP CENTER: Neck (relative = neck with trunk) ---
    center_x = width // 2 - 80
    put_txt(f"Neck (rel): {rel_angles.get('neck', 0):.1f}", center_x, y_top, C_CENTER)

    # --- TOP-RIGHT: RELATIVE (right side of body, no neck here) ---
    put_txt("RELATIVE", x_right, y_top, C_TEXT)
    yr = y_top + step
    put_txt(f"R Shldr: {rel_angles.get('right_shoulder', 0):.1f}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Elbow: {rel_angles.get('right_elbow', 0):.1f}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Wrist: {rel_angles.get('right_wrist', 0):.1f}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Hip:   {rel_angles.get('right_hip', 0):.1f}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Knee:  {rel_angles.get('right_knee', 0):.1f}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Ankle: {rel_angles.get('right_ankle', 0):.1f}", x_right, yr, C_RIGHT)
    yr += step

    # --- BOTTOM-LEFT: ABSOLUTE left side (-180..180) ---
    def fmt_abs(v):
        try:
            x = float(v)
            return f"{x:.1f}" if np.isfinite(x) else "—"
        except (TypeError, ValueError):
            return "—"

    abs_left = abs_angles
    put_txt("ABSOLUTE (-180..180)", x_left, y_bottom, (0, 255, 255))
    yl = y_bottom + step
    put_txt(f"L Trunk:  {fmt_abs(abs_left.get('trunk_abs'))}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Neck:   {fmt_abs(abs_left.get('neck_abs'))}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Thigh:  {fmt_abs(abs_left.get('left_thigh_abs'))}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Shank:  {fmt_abs(abs_left.get('left_shank_abs'))}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Foot:   {fmt_abs(abs_left.get('left_foot_abs'))}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L UArm:   {fmt_abs(abs_left.get('left_upperarm_abs'))}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Forearm:{fmt_abs(abs_left.get('left_forearm_abs'))}", x_left, yl, C_LEFT)
    yl += step
    put_txt(f"L Hand:   {fmt_abs(abs_left.get('left_hand_abs'))}", x_left, yl, C_LEFT)
    yl += step

    # --- BOTTOM-RIGHT: ABSOLUTE right side (0..360 when abs_angles_360 provided) ---
    abs_right = abs_angles_360 if abs_angles_360 is not None else abs_angles
    put_txt(
        "ABSOLUTE (0..360)" if abs_angles_360 is not None else "ABSOLUTE (-180..180)",
        x_right,
        y_bottom,
        (0, 255, 255),
    )
    yr = y_bottom + step
    put_txt(f"R Trunk:  {fmt_abs(abs_right.get('trunk_abs'))}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Neck:   {fmt_abs(abs_right.get('neck_abs'))}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Thigh:  {fmt_abs(abs_right.get('right_thigh_abs'))}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Shank:  {fmt_abs(abs_right.get('right_shank_abs'))}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Foot:   {fmt_abs(abs_right.get('right_foot_abs'))}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R UArm:   {fmt_abs(abs_right.get('right_upperarm_abs'))}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Forearm:{fmt_abs(abs_right.get('right_forearm_abs'))}", x_right, yr, C_RIGHT)
    yr += step
    put_txt(f"R Hand:   {fmt_abs(abs_right.get('right_hand_abs'))}", x_right, yr, C_RIGHT)
    yr += step

    return frame


def process_video_with_visualization(
    video_path, csv_path=None, output_dir=None, filter_config=None
):
    """
    Process video file and create visualization with angles.
    Autodetects CSV if not provided. Use calculation logic from process_angles.
    """
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    if output_dir is None:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(video_path), f"angles_video_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Locate CSV if missing
    if csv_path is None:
        # Prefer vaila format (pixel_vaila), then mp_norm, etc.
        possible_suffixes = [
            "_pixel_vaila.csv",
            "_mp_norm.csv",
            "_mp_pixel.csv",
            "_mp_data.csv",
            ".csv",
        ]
        parent = os.path.dirname(video_path)
        found = False
        for s in possible_suffixes:
            p = os.path.join(parent, f"{video_basename}{s}")
            # Try also removing extension from basename
            # e.g. video.mp4 -> video_mp_norm.csv
            # But video_basename already removed extension

            # Check for "Saída 120fps.1" pattern if user used that
            # The user implied the CSV name matches.

            if os.path.exists(p):
                csv_path = p
                found = True
                break

        # If strict match fails, try relaxed match (any csv starting with basename)
        if not found:
            for f in os.listdir(parent):
                if f.startswith(video_basename) and f.endswith(".csv"):
                    csv_path = os.path.join(parent, f)
                    found = True
                    break

        if not found:
            print(f"Error: Could not find corresponding CSV for {video_path}")
            return

    # 2. Get Video Properties for Scaling
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info: {width}x{height} @ {fps}fps")

    # 3. Process Angles (Calculation)
    # This saves the CSVs and returns DFs
    output_csv_base = os.path.join(output_dir, video_basename)

    # If no filter config, maybe default to implicit 6Hz? User asked for option.
    # We pass what we have.

    try:
        df, rel_df, abs180_df, abs360_df = process_angles(
            input_csv=csv_path,
            output_csv=output_csv_base,
            filter_config=filter_config,
            video_dims=(width, height),
        )
    except Exception as e:
        print(f"Angle calculation failed: {e}")
        cap.release()
        return

    # 4. Generate Video
    output_video_path = os.path.join(output_dir, f"angles_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Generating visualization video: {output_video_path}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= len(df):
            break  # Stop if no more data

        # Extract landmarks for this frame (filtered) directly from df
        # Landmarks in df are p0_x, p0_y...
        # We need to construct a dict for draw function

        # Find which column index starts the filtered data
        # process_angles returns filtered 'df'.
        # We need to parse it back to dict landmarks

        row = df.iloc[frame_idx]

        # Construct landmarks dict
        # We rely on get_vector_landmark helper or just manual extraction
        # But get_vector_landmark takes the WHOLE df or array.
        # We can implement a fast extractor

        landmarks = {}
        # Reuse mapping logic (but fast)
        # names defined in global 'landmark_names' in markerless_2d... but here we have local list
        # mpangles.py has `get_vector_landmark` which uses names.

        # We need a list of names.
        l_names = [
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

        # Offset: 1 if frame col exists
        start_col = 0
        if "frame" in df.columns[0].lower() or "index" in df.columns[0].lower():
            start_col = 1

        for i, name in enumerate(l_names):
            idx = start_col + (i * 2)
            if idx + 1 < len(df.columns):
                x = row.iloc[idx]
                y = row.iloc[idx + 1]
                landmarks[name] = np.array([x, y])
            else:
                landmarks[name] = np.array([np.nan, np.nan])

        # Compute midpoints locally for drawing
        if (
            not np.isnan(landmarks["left_shoulder"]).any()
            and not np.isnan(landmarks["right_shoulder"]).any()
        ):
            landmarks["mid_shoulder"] = (
                landmarks["left_shoulder"] + landmarks["right_shoulder"]
            ) / 2
        else:
            landmarks["mid_shoulder"] = np.array([np.nan, np.nan])

        if not np.isnan(landmarks["left_hip"]).any() and not np.isnan(landmarks["right_hip"]).any():
            landmarks["mid_hip"] = (landmarks["left_hip"] + landmarks["right_hip"]) / 2
        else:
            landmarks["mid_hip"] = np.array([np.nan, np.nan])

        if not np.isnan(landmarks["left_ear"]).any() and not np.isnan(landmarks["right_ear"]).any():
            landmarks["mid_ear"] = (landmarks["left_ear"] + landmarks["right_ear"]) / 2
        else:
            landmarks["mid_ear"] = np.array([np.nan, np.nan])

        # Angles dict: rel, abs180 (left panel), abs360 (right panel)
        rel = rel_df.iloc[frame_idx].to_dict()
        abs180 = abs180_df.iloc[frame_idx].to_dict()
        abs360 = abs360_df.iloc[frame_idx].to_dict()

        # Draw: left absolute -180..180, right absolute 0..360
        frame = draw_skeleton_enhanced(frame, landmarks, rel, abs180, abs_angles_360=abs360)

        out.write(frame)

        if frame_idx % 30 == 0:
            print(f"Rendering frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()
    out.release()
    print("Video generation complete.")

    # Stick figure sequence for report
    output_video_basename = os.path.basename(output_video_path)
    stick_path = None
    try:
        stick_path = os.path.join(output_dir, f"{video_basename}_stick_sequence.png")
        plot_stick_sequence_mpangles(df, stick_path, num_frames=8, rel_df=rel_df, abs_df=abs180_df)
    except Exception as e:
        print(f"Stick figure sequence skipped: {e}")

    # Generate HTML Report (with video and optional stick figure)
    csv_paths = {
        "Relative Angles": os.path.join(output_dir, f"{video_basename}_rel.csv"),
        "Absolute Angles (180)": os.path.join(output_dir, f"{video_basename}_abs_180.csv"),
        "Absolute Angles (360)": os.path.join(output_dir, f"{video_basename}_abs_360.csv"),
    }
    generate_html_report(
        output_dir,
        video_path,
        csv_paths,
        output_video_basename=output_video_basename,
        stick_figure_path=stick_path,
    )
    return output_dir


# Landmark names in MediaPipe order (0..32) for stick figure
_STICK_LANDMARK_NAMES = [
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


# (start, end) -> color: left=blue, right=red, center=gray (like video)
def _segment_color(start, end):
    if "left" in start and "left" in end:
        return "#2563eb"  # blue
    if "right" in start and "right" in end:
        return "#dc2626"  # red
    return "#6b7280"  # gray (center/trunk/neck)


# Neck is drawn only as central segment (mid_ear to mid_shoulder) below; no left/right ear–shoulder
_STICK_SEGMENTS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
]
# Landmark -> relative angle key for labels on stick figure
_JOINT_ANGLE_KEYS = {
    "left_shoulder": "left_shoulder",
    "left_elbow": "left_elbow",
    "left_wrist": "left_wrist",
    "left_hip": "left_hip",
    "left_knee": "left_knee",
    "left_ankle": "left_ankle",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "right_hip": "right_hip",
    "right_knee": "right_knee",
    "right_ankle": "right_ankle",
    "mid_shoulder": "neck",
    "mid_hip": "trunk",
}


def plot_stick_sequence_mpangles(df, output_path, num_frames=8, rel_df=None, abs_df=None):
    """
    Plot stick figure sequence: colored segments (left=blue, right=red, center=gray),
    green joint circles, and angle values at joints when rel_df is provided.
    Adds neck segment (mid_ear to mid_shoulder) for head/neck.
    """
    if not _HAS_MPL:
        return None
    n = len(df)
    if n == 0:
        return None
    start_col = (
        1
        if (
            len(df.columns) > 0
            and ("frame" in str(df.columns[0]).lower() or "index" in str(df.columns[0]).lower())
        )
        else 0
    )
    name_to_idx = {name: i for i, name in enumerate(_STICK_LANDMARK_NAMES)}
    frames_idx = np.linspace(0, n - 1, min(num_frames, n), dtype=int)
    n_plot = len(frames_idx)
    ncols = min(4, n_plot)
    nrows = (n_plot + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if n_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, fi in enumerate(frames_idx):
        ax = axes[i]
        row = df.iloc[fi]
        pts = {}
        for name in _STICK_LANDMARK_NAMES:
            idx = name_to_idx.get(name, -1)
            if idx < 0:
                continue
            c0 = start_col + idx * 2
            c1 = start_col + idx * 2 + 1
            if c1 < len(row):
                try:
                    x, y = float(row.iloc[c0]), float(row.iloc[c1])
                    if np.isfinite(x) and np.isfinite(y):
                        pts[name] = (x, y)
                except (ValueError, TypeError):
                    pass
        # Midpoints for neck (head/neck segment)
        if "left_ear" in pts and "right_ear" in pts:
            pts["mid_ear"] = (
                (pts["left_ear"][0] + pts["right_ear"][0]) / 2,
                (pts["left_ear"][1] + pts["right_ear"][1]) / 2,
            )
        if "left_shoulder" in pts and "right_shoulder" in pts:
            pts["mid_shoulder"] = (
                (pts["left_shoulder"][0] + pts["right_shoulder"][0]) / 2,
                (pts["left_shoulder"][1] + pts["right_shoulder"][1]) / 2,
            )
        if "left_hip" in pts and "right_hip" in pts:
            pts["mid_hip"] = (
                (pts["left_hip"][0] + pts["right_hip"][0]) / 2,
                (pts["left_hip"][1] + pts["right_hip"][1]) / 2,
            )
        # Colored segments (like video)
        for start, end in _STICK_SEGMENTS:
            if start in pts and end in pts:
                color = _segment_color(start, end)
                ax.plot(
                    [pts[start][0], pts[end][0]], [pts[start][1], pts[end][1]], color=color, lw=2
                )
        # Neck: mid_ear to mid_shoulder (center)
        if "mid_ear" in pts and "mid_shoulder" in pts:
            ax.plot(
                [pts["mid_ear"][0], pts["mid_shoulder"][0]],
                [pts["mid_ear"][1], pts["mid_shoulder"][1]],
                color=_segment_color("mid", "mid"),
                lw=2,
            )
        # Joint circles (green, like video)
        for name, (x, y) in pts.items():
            if name in ("mid_ear", "mid_shoulder", "mid_hip"):
                continue
            if "eye" in name or name == "nose":
                continue
            ax.plot(
                x,
                y,
                "o",
                color="#16a34a",
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
        # Angle labels at joints (when rel_df provided)
        if rel_df is not None and fi < len(rel_df):
            rel_row = rel_df.iloc[fi]
            for lm_name, key in _JOINT_ANGLE_KEYS.items():
                if lm_name not in pts or key not in rel_row:
                    continue
                try:
                    val = rel_row[key]
                    if pd.isna(val):
                        continue
                    x, y = pts[lm_name][0], pts[lm_name][1]
                    ax.text(
                        x + 8,
                        y - 8,
                        f"{float(val):.0f}",
                        fontsize=7,
                        color="black",
                        ha="left",
                        va="bottom",
                    )
                except (TypeError, ValueError, KeyError):
                    pass
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(f"Frame {fi}")
        ax.grid(True, alpha=0.3)
    for j in range(len(frames_idx), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_html_report(
    output_dir, video_path, csv_paths, output_video_basename=None, stick_figure_path=None
):
    """
    Generate an HTML report summarizing the analysis.
    output_video_basename: filename of the generated video (e.g. angles_video.mp4) for the <video> src.
    stick_figure_path: optional path to stick figure sequence image (relative to output_dir or absolute).
    """
    try:
        report_path = os.path.join(output_dir, "analysis_report.html")

        video_name = os.path.basename(video_path) if video_path else "N/A"
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MP Angles Analysis Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; color: #333; }}
                .container {{ max-width: 1000px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                header {{ border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
                h1 {{ color: #2c3e50; margin: 0; }}
                .meta {{ color: #7f8c8d; font-size: 0.9em; margin-top: 5px; }}
                .section {{ margin-bottom: 30px; }}
                h2 {{ color: #3498db; border-left: 4px solid #3498db; padding-left: 10px; }}
                .file-list {{ background-color: #f9f9f9; padding: 15px; border-radius: 4px; }}
                .file-item {{ margin-bottom: 8px; font-family: monospace; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.85em; }}
                th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
                th {{ background-color: #3498db; color: white; text-align: center; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .angles-table-wrap {{ overflow-x: auto; max-height: 400px; overflow-y: auto; }}
                .footer {{ text-align: center; color: #bdc3c7; font-size: 0.8em; margin-top: 50px; border-top: 1px solid #eee; padding-top: 20px; }}
                .btn {{ display: inline-block; padding: 8px 15px; background-color: #3498db; color: white; text-decoration: none; border-radius: 4px; margin-top: 10px; }}
                .btn:hover {{ background-color: #2980b9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>MP Angles Analysis Report</h1>
                    <div class="meta">Generated on {date_str} | vailá Toolbox</div>
                </header>
                
                <div class="section">
                    <h2>Analysis Overview</h2>
                    <p><strong>Processed Video:</strong> {video_name}</p>
                    <p>This report summarizes the kinematic analysis performed using MediaPipe pose estimation.</p>
                </div>

                <div class="section">
                    <h2>Generated Files</h2>
                    <div class="file-list">
        """

        for name, path in csv_paths.items():
            rel_p = os.path.relpath(path, output_dir)
            html_content += f'<div class="file-item"><strong>{name}:</strong> <a href="{rel_p}">{rel_p}</a></div>'

        # Angles table (from Relative Angles CSV; preview first N rows)
        angles_table_html = ""
        try:
            rel_path = csv_paths.get("Relative Angles") or list(csv_paths.values())[0]
            if rel_path and os.path.exists(rel_path):
                df_ang = pd.read_csv(rel_path, nrows=21)
                cols = list(df_ang.columns)
                angles_table_html = '<div class="angles-table-wrap"><table><thead><tr>'
                for c in cols:
                    angles_table_html += f"<th>{c}</th>"
                angles_table_html += "</tr></thead><tbody>"
                for _, row in df_ang.iterrows():
                    angles_table_html += "<tr>"
                    for c in cols:
                        v = row[c]
                        if isinstance(v, (int, float)) and np.isfinite(v):
                            angles_table_html += (
                                f"<td>{int(v) if c.lower() == 'frame' else f'{v:.1f}'}</td>"
                            )
                        else:
                            angles_table_html += f"<td>{v}</td>"
                    angles_table_html += "</tr>"
                angles_table_html += "</tbody></table></div>"
                if len(df_ang) >= 20:
                    angles_table_html += (
                        "<p><em>Showing first 21 rows. Download CSV above for full data.</em></p>"
                    )
        except Exception as e:
            angles_table_html = f"<p>Could not load angles table: {e}</p>"

        html_content += f"""
                    </div>
                </div>

                <div class="section">
                    <h2>Angles</h2>
                    <p>Preview of computed angles (relative). Download CSV files above for full data.</p>
                    {angles_table_html}
                </div>
                """
        if stick_figure_path and os.path.exists(stick_figure_path):
            rel_stick = (
                os.path.basename(stick_figure_path)
                if os.path.dirname(stick_figure_path) == output_dir
                else os.path.relpath(stick_figure_path, output_dir)
            )
            html_content += f"""
                <div class="section">
                    <h2>Pose sequence (stick figures)</h2>
                    <p>PNG with skeleton overlay by frame — colored segments and angle values at joints.</p>
                    <img src="{rel_stick}" alt="Stick figure sequence" style="max-width:100%; height:auto;">
                </div>
                """
        html_content += """
                <div class="section">
                    <h2>Angle Definitions</h2>
                    <p><strong>Relative Angles:</strong> Angles between adjacent segments (e.g., Elbow flexion).</p>
                    <p><strong>Absolute Angles:</strong> Angles of segments relative to the global horizontal (e.g., Thigh angle).</p>
                </div>
                
                <div class="footer">
                    <p>Generated by vailá Multimodal Toolbox - MP Angles Module</p>
                </div>
            </div>
        </body>
        </html>
        """

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"Report generated: {report_path}")

    except Exception as e:
        print(f"Error generating HTML report: {e}")


def generate_batch_html_report(directory_path, results_with_dirs):
    """
    Generate an HTML report in directory_path listing all processed videos
    with links to each output folder's report and video.
    results_with_dirs: list of (video_file, output_dir) from process_directory_videos.
    """
    if not results_with_dirs:
        return
    try:
        report_path = os.path.join(directory_path, "batch_analysis_report.html")
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for video_file, out_dir in results_with_dirs:
            if not out_dir:
                rows.append(f"<tr><td>{video_file}</td><td>—</td><td>—</td></tr>")
                continue
            rel_dir = os.path.relpath(out_dir, directory_path)
            report_link = os.path.join(rel_dir, "analysis_report.html")
            video_name = "angles_" + video_file
            video_link = os.path.join(rel_dir, video_name)
            rows.append(
                f'<tr><td>{video_file}</td><td><a href="{report_link}">Report</a></td>'
                f'<td><a href="{video_link}">Video</a></td></tr>'
            )
        table_rows = "\n".join(rows)
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MP Angles Batch Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        .meta {{ color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>MP Angles Batch Report</h1>
        <div class="meta">Generated on {date_str} | Directory: {directory_path}</div>
        <p>Processed {len(results_with_dirs)} video(s). Open each report for CSV links, video player, and stick figure.</p>
        <table>
            <tr><th>Video</th><th>Report</th><th>Output video</th></tr>
            {table_rows}
        </table>
    </div>
</body>
</html>
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Batch report generated: {report_path}")
    except Exception as e:
        print(f"Error generating batch HTML report: {e}")


class MPAnglesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MP Angles Analysis")
        self.root.geometry("600x650")

        style = ttk.Style()
        style.theme_use("clam")

        # Header
        header = ttk.Label(root, text="MP Angles Processing", font=("Helvetica", 16, "bold"))
        header.pack(pady=15)

        # Input Section
        lf_input = ttk.LabelFrame(root, text="Input Selection", padding=15)
        lf_input.pack(fill="x", padx=15, pady=5)

        self.input_var = tk.StringVar()
        entry_input = ttk.Entry(lf_input, textvariable=self.input_var)
        entry_input.pack(side="left", fill="x", expand=True, padx=5)

        btn_browse_csv = ttk.Button(lf_input, text="CSV", command=self.browse_csv)
        btn_browse_csv.pack(side="left", padx=2)

        btn_browse_dir = ttk.Button(lf_input, text="Dir", command=self.browse_dir)
        btn_browse_dir.pack(side="left", padx=2)

        # Filter Section — clear UX: optional low-pass smoothing
        lf_filter = ttk.LabelFrame(root, text="Smoothing (optional)", padding=15)
        lf_filter.pack(fill="x", padx=15, pady=5)

        self.use_filter = tk.BooleanVar(value=False)
        row_switch = ttk.Frame(lf_filter)
        row_switch.pack(fill="x")
        # Use classic tk Checkbutton for larger, more visible box (ttk one is often too small)
        chk_filter = tk.Checkbutton(
            row_switch,
            text="Apply Butterworth low-pass filter (smooth joint trajectories)",
            variable=self.use_filter,
            command=self.toggle_filter,
            font=("TkDefaultFont", 10),
            padx=10,
            pady=8,
            selectcolor="#cce5ff",
        )
        chk_filter.pack(side="left")
        ttk.Label(
            lf_filter,
            text="Use when data is noisy; leave unchecked for raw angles.",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(anchor="w", pady=(2, 8))

        f_params = ttk.Frame(lf_filter)
        f_params.pack(fill="x", pady=(0, 5))
        ttk.Label(f_params, text="Cutoff (Hz):").pack(side="left")
        self.cutoff_var = tk.StringVar(value="6.0")
        self.entry_cutoff = tk.Entry(f_params, textvariable=self.cutoff_var, width=6)
        self.entry_cutoff.pack(side="left", padx=5)
        ttk.Label(f_params, text="Order:").pack(side="left", padx=5)
        self.order_var = tk.StringVar(value="4")
        self.entry_order = tk.Entry(f_params, textvariable=self.order_var, width=6)
        self.entry_order.pack(side="left", padx=5)
        ttk.Label(f_params, text="FPS:").pack(side="left", padx=5)
        self.fps_var = tk.StringVar(value="30")
        self.entry_fps = tk.Entry(f_params, textvariable=self.fps_var, width=6)
        self.entry_fps.pack(side="left", padx=5)

        self.toggle_filter()  # Ensure entries are editable (user can type anytime)

        # Processing Options
        lf_opts = ttk.LabelFrame(root, text="Processing Options", padding=15)
        lf_opts.pack(fill="x", padx=15, pady=5)

        ttk.Label(lf_opts, text="Calculates Rel, Abs(-180..180), Abs(0..360) automatically.").pack(
            anchor="w"
        )

        # Action
        self.btn_run = ttk.Button(root, text="RUN ANALYSIS", command=self.run_analysis)
        self.btn_run.pack(pady=20, ipadx=20, ipady=10)

        self.status_var = tk.StringVar(value="Ready")
        lbl_status = ttk.Label(root, textvariable=self.status_var, relief="sunken")
        lbl_status.pack(side="bottom", fill="x")

    def toggle_filter(self):
        # Keep entries always editable so user can type values anytime (only the checkbox enables/disables filter at run)
        self.entry_cutoff.configure(state="normal")
        self.entry_order.configure(state="normal")
        self.entry_fps.configure(state="normal")

    def browse_csv(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if f:
            self.input_var.set(f)

    def browse_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.input_var.set(d)

    def run_analysis(self):
        input_path = self.input_var.get()
        if not input_path:
            messagebox.showerror("Error", "Please select an input (CSV or Dir).")
            return

        try:
            cutoff = float(self.cutoff_var.get().strip() or 6.0)
        except (ValueError, TypeError):
            cutoff = 6.0
        try:
            order = int(float(self.order_var.get().strip() or 4))
        except (ValueError, TypeError):
            order = 4
        try:
            fps = int(float(self.fps_var.get().strip() or 30))
        except (ValueError, TypeError):
            fps = 30
        filter_config = {
            "enabled": self.use_filter.get(),
            "cutoff": cutoff,
            "order": order,
            "fps": fps,
        }

        self.status_var.set("Processing...")
        self.root.update()

        try:
            if os.path.isdir(input_path):
                # Directory mode: batch by video (match CSV by similar name), then skeleton+angles
                results = process_directory_videos(input_path, filter_config)
                if results:
                    video_names = [r[0] for r in results]
                    generate_batch_html_report(input_path, results)
                    messagebox.showinfo(
                        "Success",
                        f"Processed {len(results)} video(s). Open batch_analysis_report.html in the directory for links.",
                    )
                else:
                    # No videos in dir: fall back to CSV-only batch (angles only, no video overlay)
                    processed = process_directory(input_path, filter_config)
                    if processed:
                        for fname, info in processed.items():
                            process_angles(info["input_path"], info["output_path"], filter_config)
                        messagebox.showinfo(
                            "Success", f"Processed {len(processed)} CSV(s) (angles only)."
                        )
                    else:
                        messagebox.showwarning(
                            "No Files", "No videos or valid landmark CSVs found in directory."
                        )

            elif input_path.lower().endswith(".csv"):
                # CSV mode: ask which video to use, then draw skeleton and angles on that video
                video_path = filedialog.askopenfilename(
                    title="Select video for this CSV (skeleton + angles overlay)",
                    initialdir=os.path.dirname(input_path),
                    filetypes=[
                        ("All files", "*.*"),
                        ("Video (MP4)", "*.mp4"),
                        ("Video (AVI)", "*.avi"),
                        ("Video (MOV)", "*.mov"),
                        ("Video (MKV)", "*.mkv"),
                        ("Video (WebM)", "*.webm"),
                    ],
                )
                if not video_path:
                    self.status_var.set("Cancelled")
                    return
                process_video_with_visualization(
                    video_path, csv_path=input_path, output_dir=None, filter_config=filter_config
                )
                messagebox.showinfo("Success", "Video with skeleton and angles generated.")

            else:
                messagebox.showerror("Error", "Input must be a CSV file or a Directory.")
                self.status_var.set("Error")
                return

            self.status_var.set("Finished")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set("Error")
            print(e)


def run_mp_angles():
    """
    Entry point for integration with vaila.py.
    Launches the GUI application.
    """
    root = tk.Tk()
    app = MPAnglesApp(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Angles Calculation Module")
    parser.add_argument(
        "-i", "--input", help="Input: CSV file or Directory (with videos + landmark CSVs)"
    )
    parser.add_argument(
        "-v",
        "--video",
        help="Video path (required for CSV input if you want skeleton overlay and report)",
    )
    parser.add_argument("-o", "--output", help="Output directory (optional)")
    parser.add_argument("--filter", action="store_true", help="Enable Butterworth filter")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Filter cutoff (Hz)")
    parser.add_argument("--order", type=int, default=4, help="Filter order")
    parser.add_argument("--fps", type=int, default=30, help="Sampling FPS")

    args = parser.parse_args()

    if args.input:
        # CLI Mode
        filter_config = {
            "enabled": args.filter,
            "cutoff": args.cutoff,
            "order": args.order,
            "fps": args.fps,
        }

        path = args.input
        if os.path.isdir(path):
            # Directory: batch by video (like GUI)
            results = process_directory_videos(path, filter_config)
            if results:
                generate_batch_html_report(path, results)
                print(
                    f"Processed {len(results)} video(s). See batch_analysis_report.html in the directory."
                )
            else:
                # No videos: CSV-only batch
                processed = process_directory(path, filter_config)
                if processed:
                    for fname, info in processed.items():
                        process_angles(info["input_path"], info["output_path"], filter_config)
                    print(f"Processed {len(processed)} CSV(s) (angles only).")
                else:
                    print("No videos or valid landmark CSVs found in directory.")

        elif path.lower().endswith(".csv"):
            if args.video:
                # CSV + video: full visualization and report
                process_video_with_visualization(
                    args.video, csv_path=path, output_dir=args.output, filter_config=filter_config
                )
                print("Video with skeleton and angles generated.")
            else:
                # CSV only: angles CSVs, no video/report
                out = args.output if args.output else os.path.dirname(path)
                base = os.path.splitext(os.path.basename(path))[0]
                out_base = os.path.join(out, f"processed_{base}")
                process_angles(path, out_base, filter_config)
                print("Angles CSVs saved. Use --video to generate overlay video and report.")

        else:
            # Path is a video file: find CSV by name and run visualization
            process_video_with_visualization(
                path, output_dir=args.output, filter_config=filter_config
            )

        print("Done.")

    else:
        # GUI Mode
        root = tk.Tk()
        app = MPAnglesApp(root)
        root.mainloop()


if __name__ == "__main__":
    main()
