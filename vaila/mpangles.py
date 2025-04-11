"""
===============================================================================
mpangles.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 31 March 2025
Update Date: 11 April 2025
Version: 0.1.1
Python Version: 3.12.9

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

Usage:
------
python relativeangle.py input_csv output_csv [--segments SEGMENTS]

Arguments:
  input_csv    Path to input CSV file with landmark coordinates
  output_csv   Path to output CSV file for computed angles
  --segments   Optional: List of segments to analyze (default: all)
               Options: left_arm, right_arm, left_leg, right_leg, trunk

Example:
  python relativeangle.py pose_landmarks.csv angles.csv --segments left_arm,right_leg

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================
"""

import os
from rich import print
import pandas as pd
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2


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
        directory = filedialog.askdirectory(
            title="Select Directory with CSV Files", mustexist=True
        )

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


def process_directory(directory_path=None):
    """
    Process all CSV files in the given directory.
    If no directory is provided, opens a dialog to select one.

    Parameters:
    -----------
    directory_path : str, optional
        Path to the directory containing CSV files

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

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        messagebox.showwarning("No Files", f"No CSV files found in {directory_path}")
        return None

    print(f"Found {len(csv_files)} CSV files to process")

    # Create output directory for processed files
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(directory_path, f"processed_angles_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        print(f"\nProcessing {csv_file}...")

        try:
            data = pd.read_csv(file_path)
            # Convert to numpy array for faster processing
            data_array = data.values
            processed_files[csv_file] = {
                "data": data_array,
                "output_path": os.path.join(output_dir, f"processed_{csv_file}"),
            }
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue

    return processed_files


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
        raise ValueError(
            f"Landmark '{landmark}' not found in the list of valid landmarks"
        )

    # Calculate the column indices for x and y
    # Add 1 to skip the frame column, and multiply by 2 because each landmark has 2 columns
    x_col = (landmark_idx * 2) + 1
    y_col = x_col + 1

    # Convert to numpy array if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Extract the x,y coordinates
    return data[:, [x_col, y_col]]


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
        mid_ear = [
            compute_midpoint(left, right) for left, right in zip(left_ear, right_ear)
        ]
        mid_shoulder = [
            compute_midpoint(left, right)
            for left, right in zip(left_shoulder, right_shoulder)
        ]
        mid_hip = [
            compute_midpoint(left, right) for left, right in zip(left_hip, right_hip)
        ]

        # Calculate absolute angles for segments
        right_thigh_angles = np.array(
            [
                compute_absolute_angle(hip, knee)
                for hip, knee in zip(right_hip, right_knee)
            ]
        )
        right_shank_angles = np.array(
            [
                compute_absolute_angle(knee, ankle)
                for knee, ankle in zip(right_knee, right_ankle)
            ]
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
            [
                compute_absolute_angle(elbow, wrist)
                for elbow, wrist in zip(right_elbow, right_wrist)
            ]
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
            [
                compute_absolute_angle(hip, knee)
                for hip, knee in zip(left_hip, left_knee)
            ]
        )
        left_shank_angles = np.array(
            [
                compute_absolute_angle(knee, ankle)
                for knee, ankle in zip(left_knee, left_ankle)
            ]
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
            [
                compute_absolute_angle(elbow, wrist)
                for elbow, wrist in zip(left_elbow, left_wrist)
            ]
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
            [
                compute_absolute_angle(shoulder, hip)
                for shoulder, hip in zip(mid_shoulder, mid_hip)
            ]
        )
        neck_angles = np.array(
            [
                compute_absolute_angle(ear, shoulder)
                for ear, shoulder in zip(mid_ear, mid_shoulder)
            ]
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
        angles_df.insert(
            0, "frame_index", df.iloc[:, 0]
        )  # Add frame index as first column

        # Save to CSV
        angles_df.to_csv(output_csv, index=False, float_format="%.2f")
        print(f"\nAngles saved to: {output_csv}")

    except Exception as e:
        print(f"Error processing absolute angles: {str(e)}")
        raise


def process_angles(input_csv, output_csv, segments=None, format_360=False):
    """
    Process landmark data and compute specified angles.

    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        segments (list): List of segments to analyze (default: all)
        format_360: Boolean to control the angle format:
                   - True: returns angle in [0, 360) degree range
                   - False: returns angle in [-180, 180) degree range (default)
    """
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
        mid_ear = [
            compute_midpoint(left, right) for left, right in zip(left_ear, right_ear)
        ]
        mid_shoulder = [
            compute_midpoint(left, right)
            for left, right in zip(left_shoulder, right_shoulder)
        ]
        mid_hip = [
            compute_midpoint(left, right) for left, right in zip(left_hip, right_hip)
        ]

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
                    compute_shoulder_angle(
                        right_shoulder[i], right_elbow[i], trunk_vectors[i]
                    )
                )
                right_elbow_angles.append(
                    compute_elbow_angle(
                        right_shoulder[i], right_elbow[i], right_wrist[i]
                    )
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
                    compute_shoulder_angle(
                        left_shoulder[i], left_elbow[i], trunk_vectors[i]
                    )
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
                        f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
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
        relative_angles_df.to_csv(
            relative_output_path, index=False, float_format="%.2f"
        )
        absolute_angles_df.to_csv(
            absolute_output_path, index=False, float_format="%.2f"
        )

        print(f"\nÂngulos relativos salvos em: {relative_output_path}")
        print(f"Ângulos absolutos salvos em: {absolute_output_path}")

        # Não é necessário salvar o output_csv original, já que estamos criando dois arquivos específicos

    except Exception as e:
        print(f"Error processing angles: {str(e)}")
        raise


def draw_skeleton_and_angles(frame, landmarks, angles, absolute_angles):
    """
    Draw skeleton segments, joints and angle values (both relative and absolute) on the frame.
    """
    height, width = frame.shape[:2]

    # Colors
    RED = (0, 0, 255)  # Right side
    BLUE = (255, 0, 0)  # Left side
    GREEN = (0, 255, 0)  # Joints
    WHITE = (255, 255, 255)  # Text
    YELLOW = (0, 255, 255)  # Absolute angles

    # Calculate mid_hand points
    right_mid_hand = compute_midpoint(
        landmarks["right_pinky"], landmarks["right_index"]
    )
    left_mid_hand = compute_midpoint(landmarks["left_pinky"], landmarks["left_index"])

    # Add mid_hand to landmarks dictionary
    landmarks["right_mid_hand"] = right_mid_hand
    landmarks["left_mid_hand"] = left_mid_hand

    # Draw segments
    # Right side (in RED)
    cv2.line(
        frame,
        tuple(landmarks["right_shoulder"].astype(int)),
        tuple(landmarks["right_elbow"].astype(int)),
        RED,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["right_elbow"].astype(int)),
        tuple(landmarks["right_wrist"].astype(int)),
        RED,
        2,
    )
    # Add line from right wrist to right mid_hand
    cv2.line(
        frame,
        tuple(landmarks["right_wrist"].astype(int)),
        tuple(landmarks["right_mid_hand"].astype(int)),
        RED,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["right_hip"].astype(int)),
        tuple(landmarks["right_knee"].astype(int)),
        RED,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["right_knee"].astype(int)),
        tuple(landmarks["right_ankle"].astype(int)),
        RED,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["right_heel"].astype(int)),
        tuple(landmarks["right_foot_index"].astype(int)),
        RED,
        2,
    )

    # Left side (in BLUE)
    cv2.line(
        frame,
        tuple(landmarks["left_shoulder"].astype(int)),
        tuple(landmarks["left_elbow"].astype(int)),
        BLUE,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["left_elbow"].astype(int)),
        tuple(landmarks["left_wrist"].astype(int)),
        BLUE,
        2,
    )
    # Add line from left wrist to left mid_hand
    cv2.line(
        frame,
        tuple(landmarks["left_wrist"].astype(int)),
        tuple(landmarks["left_mid_hand"].astype(int)),
        BLUE,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["left_hip"].astype(int)),
        tuple(landmarks["left_knee"].astype(int)),
        BLUE,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["left_knee"].astype(int)),
        tuple(landmarks["left_ankle"].astype(int)),
        BLUE,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["left_heel"].astype(int)),
        tuple(landmarks["left_foot_index"].astype(int)),
        BLUE,
        2,
    )

    # Draw trunk and neck
    cv2.line(
        frame,
        tuple(landmarks["mid_shoulder"].astype(int)),
        tuple(landmarks["mid_hip"].astype(int)),
        WHITE,
        2,
    )
    cv2.line(
        frame,
        tuple(landmarks["mid_ear"].astype(int)),
        tuple(landmarks["mid_shoulder"].astype(int)),
        WHITE,
        2,
    )  # Neck segment

    # Draw joints (circles) - exclude nose
    joint_radius = 4
    for landmark_name, landmark in landmarks.items():
        if landmark_name != "nose":  # Skip nose landmark
            cv2.circle(frame, tuple(landmark.astype(int)), joint_radius, GREEN, -1)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Vertical spacing for text
    line_height = 30

    # Margins and positioning
    left_margin = 10
    right_margin = width - 300
    center_x = width // 2 - 50

    # Reordenação dos ângulos relativos centrais
    y_offset = line_height
    cv2.putText(
        frame,
        f"Neck Rel: {angles['neck']:.1f}",
        (center_x - 200, y_offset),
        font,
        font_scale,
        WHITE,
        thickness,
    )
    cv2.putText(
        frame,
        f"Trunk Rel: {angles['trunk']:.1f}",
        (center_x + 100, y_offset),
        font,
        font_scale,
        WHITE,
        thickness,
    )

    # Reordenação dos ângulos absolutos centrais
    y_offset += line_height
    cv2.putText(
        frame,
        f"Neck Abs: {absolute_angles['neck_abs']:.1f}",
        (center_x - 200, y_offset),
        font,
        font_scale,
        YELLOW,
        thickness,
    )
    cv2.putText(
        frame,
        f"Trunk Abs: {absolute_angles['trunk_abs']:.1f}",
        (center_x + 100, y_offset),
        font,
        font_scale,
        YELLOW,
        thickness,
    )

    # Right side relative angles (in RED) na nova ordem
    y_offset = line_height
    cv2.putText(
        frame,
        f"R Shoulder Rel: {angles['right_shoulder']:.1f}",
        (left_margin, y_offset),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"R Elbow Rel: {angles['right_elbow']:.1f}",
        (left_margin, y_offset),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"R Wrist Rel: {angles['right_wrist']:.1f}",
        (left_margin, y_offset),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"R Hip Rel: {angles['right_hip']:.1f}",
        (left_margin, y_offset),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"R Knee Rel: {angles['right_knee']:.1f}",
        (left_margin, y_offset),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"R Ankle Rel: {angles['right_ankle']:.1f}",
        (left_margin, y_offset),
        font,
        font_scale,
        RED,
        thickness,
    )

    # Left side relative angles (in BLUE) na nova ordem
    y_offset = line_height
    cv2.putText(
        frame,
        f"L Shoulder Rel: {angles['left_shoulder']:.1f}",
        (right_margin, y_offset),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"L Elbow Rel: {angles['left_elbow']:.1f}",
        (right_margin, y_offset),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"L Wrist Rel: {angles['left_wrist']:.1f}",
        (right_margin, y_offset),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"L Hip Rel: {angles['left_hip']:.1f}",
        (right_margin, y_offset),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"L Knee Rel: {angles['left_knee']:.1f}",
        (right_margin, y_offset),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset += line_height
    cv2.putText(
        frame,
        f"L Ankle Rel: {angles['left_ankle']:.1f}",
        (right_margin, y_offset),
        font,
        font_scale,
        BLUE,
        thickness,
    )

    # Right side absolute angles (in RED) na nova ordem
    y_offset_abs = height - 210
    cv2.putText(
        frame,
        f"R UpperArm Abs: {absolute_angles['right_upperarm_abs']:.1f}",
        (left_margin, y_offset_abs),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"R Forearm Abs: {absolute_angles['right_forearm_abs']:.1f}",
        (left_margin, y_offset_abs),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"R Hand Abs: {absolute_angles['right_hand_abs']:.1f}",
        (left_margin, y_offset_abs),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"R Thigh Abs: {absolute_angles['right_thigh_abs']:.1f}",
        (left_margin, y_offset_abs),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"R Shank Abs: {absolute_angles['right_shank_abs']:.1f}",
        (left_margin, y_offset_abs),
        font,
        font_scale,
        RED,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"R Foot Abs: {absolute_angles['right_foot_abs']:.1f}",
        (left_margin, y_offset_abs),
        font,
        font_scale,
        RED,
        thickness,
    )

    # Left side absolute angles (in BLUE) na nova ordem
    y_offset_abs = height - 210
    cv2.putText(
        frame,
        f"L UpperArm Abs: {absolute_angles['left_upperarm_abs']:.1f}",
        (right_margin, y_offset_abs),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"L Forearm Abs: {absolute_angles['left_forearm_abs']:.1f}",
        (right_margin, y_offset_abs),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"L Hand Abs: {absolute_angles['left_hand_abs']:.1f}",
        (right_margin, y_offset_abs),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"L Thigh Abs: {absolute_angles['left_thigh_abs']:.1f}",
        (right_margin, y_offset_abs),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"L Shank Abs: {absolute_angles['left_shank_abs']:.1f}",
        (right_margin, y_offset_abs),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    y_offset_abs += line_height
    cv2.putText(
        frame,
        f"L Foot Abs: {absolute_angles['left_foot_abs']:.1f}",
        (right_margin, y_offset_abs),
        font,
        font_scale,
        BLUE,
        thickness,
    )

    return frame


def process_video_with_visualization(video_path, csv_path, output_dir, format_360=False):
    """
    Process video file and create visualization with both relative and absolute angles.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV file with coordinates
    try:
        df = pd.read_csv(csv_path)
        print(f"Reading coordinates from: {csv_path}")
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Create video writer
    output_video_path = os.path.join(
        output_dir, f"angles_{os.path.basename(video_path)}"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0

    # Initialize lists to store angles for CSV
    relative_angles_list = []
    absolute_angles_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= len(df):
            break

        # Get landmarks from CSV for current frame
        landmarks = {}

        # Add nose landmark
        landmarks["nose"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "nose")[0]
        )

        # Add ear landmarks
        landmarks["right_ear"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_ear")[0]
        )
        landmarks["left_ear"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_ear")[0]
        )

        # Right side landmarks
        landmarks["right_shoulder"] = np.array(
            get_vector_landmark(
                df.iloc[frame_count : frame_count + 1], "right_shoulder"
            )[0]
        )
        landmarks["right_elbow"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_elbow")[
                0
            ]
        )
        landmarks["right_wrist"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_wrist")[
                0
            ]
        )
        landmarks["right_pinky"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_pinky")[
                0
            ]
        )
        landmarks["right_index"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_index")[
                0
            ]
        )
        landmarks["right_hip"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_hip")[0]
        )
        landmarks["right_knee"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_knee")[0]
        )
        landmarks["right_ankle"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_ankle")[
                0
            ]
        )
        landmarks["right_foot_index"] = np.array(
            get_vector_landmark(
                df.iloc[frame_count : frame_count + 1], "right_foot_index"
            )[0]
        )
        landmarks["right_heel"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "right_heel")[0]
        )

        # Left side landmarks
        landmarks["left_shoulder"] = np.array(
            get_vector_landmark(
                df.iloc[frame_count : frame_count + 1], "left_shoulder"
            )[0]
        )
        landmarks["left_elbow"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_elbow")[0]
        )
        landmarks["left_wrist"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_wrist")[0]
        )
        landmarks["left_pinky"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_pinky")[0]
        )
        landmarks["left_index"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_index")[0]
        )
        landmarks["left_hip"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_hip")[0]
        )
        landmarks["left_knee"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_knee")[0]
        )
        landmarks["left_ankle"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_ankle")[0]
        )
        landmarks["left_foot_index"] = np.array(
            get_vector_landmark(
                df.iloc[frame_count : frame_count + 1], "left_foot_index"
            )[0]
        )
        landmarks["left_heel"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_heel")[0]
        )

        # Calculate midpoints
        landmarks["mid_hip"] = compute_midpoint(
            landmarks["left_hip"], landmarks["right_hip"]
        )
        landmarks["mid_shoulder"] = compute_midpoint(
            landmarks["left_shoulder"], landmarks["right_shoulder"]
        )
        landmarks["mid_ear"] = compute_midpoint(
            landmarks["left_ear"], landmarks["right_ear"]
        )

        # Initialize angles dictionary with default values
        angles = {
            # Right side
            "right_shoulder": 0,
            "right_elbow": 0,
            "right_hip": 0,
            "right_knee": 0,
            "right_ankle": 0,
            "right_wrist": 0,
            # Left side
            "left_shoulder": 0,
            "left_elbow": 0,
            "left_hip": 0,
            "left_knee": 0,
            "left_ankle": 0,
            "left_wrist": 0,
            # Central segments
            "neck": 0,
            "trunk": 0,
        }

        # Calculate trunk vector
        try:
            trunk_vector = np.array(landmarks["mid_hip"]) - np.array(
                landmarks["mid_shoulder"]
            )
            trunk_norm = np.linalg.norm(trunk_vector)
            if trunk_norm > 0:
                trunk_vector = trunk_vector / trunk_norm

                # Calculate relative angles
                # Right side
                angles["right_shoulder"] = compute_shoulder_angle(
                    landmarks["right_shoulder"], landmarks["right_elbow"], trunk_vector
                )
                angles["right_elbow"] = compute_elbow_angle(
                    landmarks["right_shoulder"],
                    landmarks["right_elbow"],
                    landmarks["right_wrist"],
                )
                angles["right_hip"] = compute_hip_angle(
                    landmarks["right_hip"], landmarks["right_knee"], trunk_vector
                )
                angles["right_knee"] = compute_knee_angle(
                    landmarks["right_hip"],
                    landmarks["right_knee"],
                    landmarks["right_ankle"],
                )
                angles["right_ankle"] = compute_ankle_angle(
                    landmarks["right_knee"],
                    landmarks["right_ankle"],
                    landmarks["right_foot_index"],
                    landmarks["right_heel"],
                )
                try:
                    angles["right_wrist"] = compute_wrist_angle(
                        landmarks["right_elbow"],
                        landmarks["right_wrist"],
                        landmarks["right_pinky"],
                        landmarks["right_index"],
                    )
                except:
                    angles["right_wrist"] = 0

                # Left side
                angles["left_shoulder"] = compute_shoulder_angle(
                    landmarks["left_shoulder"], landmarks["left_elbow"], trunk_vector
                )
                angles["left_elbow"] = compute_elbow_angle(
                    landmarks["left_shoulder"],
                    landmarks["left_elbow"],
                    landmarks["left_wrist"],
                )
                angles["left_hip"] = compute_hip_angle(
                    landmarks["left_hip"], landmarks["left_knee"], trunk_vector
                )
                angles["left_knee"] = compute_knee_angle(
                    landmarks["left_hip"], landmarks["left_knee"], landmarks["left_ankle"]
                )
                angles["left_ankle"] = compute_ankle_angle(
                    landmarks["left_knee"],
                    landmarks["left_ankle"],
                    landmarks["left_foot_index"],
                    landmarks["left_heel"],
                )
                try:
                    angles["left_wrist"] = compute_wrist_angle(
                        landmarks["left_elbow"],
                        landmarks["left_wrist"],
                        landmarks["left_pinky"],
                        landmarks["left_index"],
                    )
                except:
                    angles["left_wrist"] = 0

                # Central segments relative angles
                angles["neck"] = compute_neck_angle(
                    landmarks["mid_ear"], landmarks["mid_shoulder"], trunk_vector
                )
                angles["trunk"] = compute_relative_angle(
                    landmarks["mid_shoulder"],
                    landmarks["mid_hip"],
                    landmarks["mid_shoulder"],
                )
        except Exception as e:
            print(f"Error calculating relative angles: {str(e)}")

        # Initialize absolute angles dictionary with default values
        absolute_angles = {
            # Right side
            "right_thigh_abs": 0,
            "right_shank_abs": 0,
            "right_foot_abs": 0,
            "right_upperarm_abs": 0,
            "right_forearm_abs": 0,
            "right_hand_abs": 0,
            # Left side
            "left_thigh_abs": 0,
            "left_shank_abs": 0,
            "left_foot_abs": 0,
            "left_upperarm_abs": 0,
            "left_forearm_abs": 0,
            "left_hand_abs": 0,
            # Central segments
            "trunk_abs": 0,
            "neck_abs": 0,
        }

        try:
            # Calculate absolute angles
            absolute_angles["right_thigh_abs"] = compute_absolute_angle(
                landmarks["right_hip"], landmarks["right_knee"], format_360
            )
            absolute_angles["right_shank_abs"] = compute_absolute_angle(
                landmarks["right_knee"], landmarks["right_ankle"], format_360
            )
            absolute_angles["right_foot_abs"] = compute_absolute_angle(
                landmarks["right_heel"], landmarks["right_foot_index"], format_360
            )
            absolute_angles["right_upperarm_abs"] = compute_absolute_angle(
                landmarks["right_shoulder"], landmarks["right_elbow"], format_360
            )
            absolute_angles["right_forearm_abs"] = compute_absolute_angle(
                landmarks["right_elbow"], landmarks["right_wrist"], format_360
            )

            try:
                right_hand_mid = compute_midpoint(
                    landmarks["right_pinky"], landmarks["right_index"]
                )
                absolute_angles["right_hand_abs"] = compute_absolute_angle(
                    landmarks["right_wrist"], right_hand_mid, format_360
                )
            except:
                absolute_angles["right_hand_abs"] = 0

            absolute_angles["left_thigh_abs"] = compute_absolute_angle(
                landmarks["left_hip"], landmarks["left_knee"], format_360
            )
            absolute_angles["left_shank_abs"] = compute_absolute_angle(
                landmarks["left_knee"], landmarks["left_ankle"], format_360
            )
            absolute_angles["left_foot_abs"] = compute_absolute_angle(
                landmarks["left_heel"], landmarks["left_foot_index"], format_360
            )
            absolute_angles["left_upperarm_abs"] = compute_absolute_angle(
                landmarks["left_shoulder"], landmarks["left_elbow"], format_360
            )
            absolute_angles["left_forearm_abs"] = compute_absolute_angle(
                landmarks["left_elbow"], landmarks["left_wrist"], format_360
            )

            try:
                left_hand_mid = compute_midpoint(
                    landmarks["left_pinky"], landmarks["left_index"]
                )
                absolute_angles["left_hand_abs"] = compute_absolute_angle(
                    landmarks["left_wrist"], left_hand_mid, format_360
                )
            except:
                absolute_angles["left_hand_abs"] = 0

            absolute_angles["trunk_abs"] = compute_absolute_angle(
                landmarks["mid_shoulder"], landmarks["mid_hip"], format_360
            )
            absolute_angles["neck_abs"] = compute_absolute_angle(
                landmarks["mid_ear"], landmarks["mid_shoulder"], format_360
            )
        except Exception as e:
            print(f"Error calculating absolute angles: {str(e)}")

        # Salve os ângulos relativos e absolutos nas listas
        relative_angles_list.append(angles)
        absolute_angles_list.append(absolute_angles)

        # Draw visualization with both relative and absolute angles
        frame = draw_skeleton_and_angles(frame, landmarks, angles, absolute_angles)

        # Write frame
        out.write(frame)

        # Show progress
        frame_count += 1
        if frame_count % 30 == 0:
            print(
                f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
            )

    # Release resources
    cap.release()
    out.release()

    # Save relative and absolute angles to CSV
    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    # Reorganize relative angles in the desired order
    relative_angles_list_ordered = []
    for angles_dict in relative_angles_list:
        ordered_dict = {
            "neck": angles_dict["neck"],
            "trunk": angles_dict["trunk"],
            # Right side
            "right_shoulder": angles_dict["right_shoulder"],
            "right_elbow": angles_dict["right_elbow"],
            "right_wrist": angles_dict["right_wrist"],
            "right_hip": angles_dict["right_hip"],
            "right_knee": angles_dict["right_knee"],
            "right_ankle": angles_dict["right_ankle"],
            # Left side
            "left_shoulder": angles_dict["left_shoulder"],
            "left_elbow": angles_dict["left_elbow"],
            "left_wrist": angles_dict["left_wrist"],
            "left_hip": angles_dict["left_hip"],
            "left_knee": angles_dict["left_knee"],
            "left_ankle": angles_dict["left_ankle"],
        }
        relative_angles_list_ordered.append(ordered_dict)

    # Reorganize absolute angles in the desired order
    absolute_angles_list_ordered = []
    for angles_dict in absolute_angles_list:
        ordered_dict = {
            "neck_abs": angles_dict["neck_abs"],
            "trunk_abs": angles_dict["trunk_abs"],
            # Right side
            "right_upperarm_abs": angles_dict["right_upperarm_abs"],
            "right_forearm_abs": angles_dict["right_forearm_abs"],
            "right_hand_abs": angles_dict["right_hand_abs"],
            "right_thigh_abs": angles_dict["right_thigh_abs"],
            "right_shank_abs": angles_dict["right_shank_abs"],
            "right_foot_abs": angles_dict["right_foot_abs"],
            # Left side
            "left_upperarm_abs": angles_dict["left_upperarm_abs"],
            "left_forearm_abs": angles_dict["left_forearm_abs"],
            "left_hand_abs": angles_dict["left_hand_abs"],
            "left_thigh_abs": angles_dict["left_thigh_abs"],
            "left_shank_abs": angles_dict["left_shank_abs"],
            "left_foot_abs": angles_dict["left_foot_abs"],
        }
        absolute_angles_list_ordered.append(ordered_dict)

    # Create DataFrames with the ordered angles
    relative_angles_df = pd.DataFrame(relative_angles_list_ordered)
    absolute_angles_df = pd.DataFrame(absolute_angles_list_ordered)

    # Create a frame index for both DataFrames
    frame_index = np.arange(len(relative_angles_df))

    # Insert the frame index as the first column
    relative_angles_df.insert(0, "frame", frame_index)
    absolute_angles_df.insert(0, "frame", frame_index)

    relative_angles_csv_path = os.path.join(output_dir, f"{video_basename}_rel.csv")
    absolute_angles_csv_path = os.path.join(output_dir, f"{video_basename}_abs.csv")

    relative_angles_df.to_csv(
        relative_angles_csv_path, index=False, float_format="%.2f"
    )
    absolute_angles_df.to_csv(
        absolute_angles_csv_path, index=False, float_format="%.2f"
    )

    print(f"Video processing complete. Output saved to: {output_video_path}")
    print(f"Relative angles saved to: {relative_angles_csv_path}")
    print(f"Absolute angles saved to: {absolute_angles_csv_path}")


def select_video_file():
    """
    Opens a dialog to select a video file.

    Returns:
        str or None: Path to selected video file
    """
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
            ("All files", "*.*"),
        ],
    )

    if file_path:
        return file_path
    return None


def select_csv_file():
    """
    Opens a dialog to select a CSV file with coordinates.

    Returns:
        str or None: Path to selected CSV file
    """
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select CSV File with Coordinates",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )

    if file_path:
        return file_path
    return None


def run_mp_angles():
    """Runs the MP Angles module with options for CSV or video processing."""
    print("\nStarting MP Angles module...")

    # Ask user what type of processing they want
    root = tk.Tk()
    root.withdraw()

    # Primeiro perguntar sobre o formato do ângulo
    angle_format = messagebox.askquestion(
        "Angle Format",
        "Choose the angle format for absolute angles:\n\n"
        + "Yes: 0 to 360 degrees format\n"
        + "No: -180 to +180 degrees format"
    )
    
    # Converter para boolean
    use_format_360 = (angle_format == "yes")
    
    # Armazenar a escolha em uma variável global para uso em todo o código
    global ANGLE_FORMAT_360
    ANGLE_FORMAT_360 = use_format_360
    
    format_text = "0 to 360°" if use_format_360 else "-180 to +180°"
    print(f"Selected angle format: {format_text}")

    process_type = messagebox.askquestion(
        "Processing Type",
        "Do you want to process a video file?\n\n"
        + "Yes: Process video with visualization\n"
        + "No: Process CSV files",
    )

    if process_type == "yes":
        # First select CSV file with coordinates
        csv_path = select_csv_file()
        if not csv_path:
            print("No CSV file selected. Exiting.")
            return

        # After selecting the CSV file, select the video file for visualization
        video_path = select_video_file()
        if not video_path:
            print("No video file selected. Exiting.")
            return

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(video_path), f"angles_video_{timestamp}")
        process_video_with_visualization(video_path, csv_path, output_dir, format_360=use_format_360)

    else:
        # CSV processing (existing functionality)
        input_dir = select_directory()
        if not input_dir:
            print("No input directory selected. Exiting.")
            return

        processed_files = process_directory(input_dir)
        if not processed_files:
            print("No files were processed. Exiting.")
            return

        # Process each file in the directory
        for csv_file, file_info in processed_files.items():
            try:
                input_path = os.path.join(input_dir, csv_file)
                output_path = file_info["output_path"]
                
                # Processar e salvar ângulos relativos e absolutos de uma vez
                process_angles(input_path, output_path, format_360=use_format_360)
                print(f"Successfully processed angles: {csv_file}")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")
                continue

    print("\nProcessing complete!")


def main():
    """
    Main function to run the angle processing pipeline.
    """
    try:
        run_mp_angles()
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
    finally:
        print("\nProgram finished. You can close this window.")


if __name__ == "__main__":
    main()
