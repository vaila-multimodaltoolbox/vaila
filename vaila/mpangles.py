"""
===============================================================================
mpangles.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 31 March 2025
Update Date: 09 April 2025
Version: 0.1.2
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
   - Uses dot product and cross product for angle calculation

3. Supported Angles:
    - Elbow angle (between upper arm and forearm)
    - Shoulder angle (between trunk and upper arm)
    - Hip angle (between trunk and thigh)
    - Knee angle (between thigh and shank)
    - Ankle angle (between shank and foot)

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
import sys
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp


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

    Parameters:
    -----------
    data : numpy.ndarray or pandas.DataFrame
        The input data array with shape (n_frames, n_columns)
        First column is frame number, followed by p1_x,p1_y,p2_x,p2_y,...
    landmark : str
        The name of the landmark to extract (e.g., "nose", "left_shoulder", etc.)

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


def compute_absolute_angle(p1, p2):
    """
    Calculates the absolute angle (in degrees) of the vector from p1 to p2
    relative to the horizontal axis.

    Args:
        p1 (list): [x, y] coordinates of first point
        p2 (list): [x, y] coordinates of second point

    Returns:
        float: Angle in degrees (-180 to 180)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle


def compute_relative_angle(a, b, c):
    """
    Compute the angle between three points.

    Args:
        a: First point (3D vector)
        b: Middle point (3D vector)
        c: Third point (3D vector)

    Returns:
        Angle in degrees
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
        hip: Hip point (3D vector)
        knee: Knee point (3D vector)
        ankle: Ankle point (3D vector)

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
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_midpoint(p1, p2):
    """
    Compute the midpoint between two 2D points.

    Args:
        p1: First point (2D vector)
        p2: Second point (2D vector)

    Returns:
        Midpoint as a numpy array
    """
    return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])


def compute_hip_angle(hip, knee, trunk_vector):
    """
    Compute the hip angle using thigh vector (knee-hip) and trunk vector.

    Args:
        hip: Hip point (3D vector)
        knee: Knee point (3D vector)
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
    shank_vector = np.array(ankle) - np.array(knee)

    # Calculate foot vector (foot_index to heel)
    foot_vector = np.array(heel) - np.array(foot_index)

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
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def compute_neck_angle(nose, mid_shoulder, trunk_vector):
    """
    Compute the neck angle using head-nose vector (nose-mid_shoulder) and trunk vector.

    Args:
        nose: Nose point (2D vector)
        mid_shoulder: Mid shoulder point (2D vector)
        trunk_vector: Normalized trunk vector (2D)

    Returns:
        Neck angle in degrees
    """
    # Calculate head-nose vector (nose to mid_shoulder)
    headnose_vector = np.array(nose) - np.array(mid_shoulder)

    # Normalize head-nose vector
    headnose_norm = np.linalg.norm(headnose_vector)

    if headnose_norm == 0:
        return 0.0

    headnose_normalized = headnose_vector / headnose_norm

    # Calculate dot product
    dot_product = np.dot(headnose_normalized, trunk_vector)

    # Clamp dot product to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate angle in radians
    angle_rad = np.arccos(dot_product)

    # Convert to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


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
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def process_angles(input_csv, output_csv, segments=None):
    """
    Process landmark data and compute specified angles.

    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        segments (list): List of segments to analyze (default: all)
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_csv)
        print(f"Reading input file: {input_csv}")

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

        # Left side landmarks
        left_shoulder = get_vector_landmark(df, "left_shoulder")
        left_elbow = get_vector_landmark(df, "left_elbow")
        left_wrist = get_vector_landmark(df, "left_wrist")
        left_hip = get_vector_landmark(df, "left_hip")
        left_knee = get_vector_landmark(df, "left_knee")
        left_ankle = get_vector_landmark(df, "left_ankle")
        left_foot_index = get_vector_landmark(df, "left_foot_index")
        left_heel = get_vector_landmark(df, "left_heel")

        # Calculate midpoints for trunk vector
        mid_hip = [
            compute_midpoint(l_hip, r_hip) for l_hip, r_hip in zip(left_hip, right_hip)
        ]
        mid_shoulder = [
            compute_midpoint(l_shoulder, r_shoulder)
            for l_shoulder, r_shoulder in zip(left_shoulder, right_shoulder)
        ]

        # Calculate trunk vector (mid_hip - mid_shoulder) and normalize
        trunk_vectors = []
        for m_hip, m_shoulder in zip(mid_hip, mid_shoulder):
            trunk_vector = np.array(m_hip) - np.array(
                m_shoulder
            )  # Changed direction to hip - shoulder
            trunk_norm = np.linalg.norm(trunk_vector)

            if trunk_norm == 0:
                trunk_vectors.append(np.array([0, 0]))
            else:
                trunk_vectors.append(trunk_vector / trunk_norm)

        # Right side angles
        right_elbow_angles = np.array(
            [
                compute_elbow_angle(shoulder, elbow, wrist)
                for shoulder, elbow, wrist in zip(
                    right_shoulder, right_elbow, right_wrist
                )
            ]
        )

        right_shoulder_angles = np.array(
            [
                compute_shoulder_angle(shoulder, elbow, trunk_vector)
                for shoulder, elbow, trunk_vector in zip(
                    right_shoulder, right_elbow, trunk_vectors
                )
            ]
        )

        right_hip_angles = np.array(
            [
                compute_hip_angle(hip, knee, trunk_vector)
                for hip, knee, trunk_vector in zip(right_hip, right_knee, trunk_vectors)
            ]
        )

        right_knee_angles = np.array(
            [
                compute_knee_angle(hip, knee, ankle)
                for hip, knee, ankle in zip(right_hip, right_knee, right_ankle)
            ]
        )

        right_ankle_angles = np.array(
            [
                compute_ankle_angle(knee, ankle, foot_index, heel)
                for knee, ankle, foot_index, heel in zip(
                    right_knee, right_ankle, right_foot_index, right_heel
                )
            ]
        )

        # Left side angles
        left_elbow_angles = np.array(
            [
                compute_elbow_angle(shoulder, elbow, wrist)
                for shoulder, elbow, wrist in zip(left_shoulder, left_elbow, left_wrist)
            ]
        )

        left_shoulder_angles = np.array(
            [
                compute_shoulder_angle(shoulder, elbow, trunk_vector)
                for shoulder, elbow, trunk_vector in zip(
                    left_shoulder, left_elbow, trunk_vectors
                )
            ]
        )

        left_hip_angles = np.array(
            [
                compute_hip_angle(hip, knee, trunk_vector)
                for hip, knee, trunk_vector in zip(left_hip, left_knee, trunk_vectors)
            ]
        )

        left_knee_angles = np.array(
            [
                compute_knee_angle(hip, knee, ankle)
                for hip, knee, ankle in zip(left_hip, left_knee, left_ankle)
            ]
        )

        left_ankle_angles = np.array(
            [
                compute_ankle_angle(knee, ankle, foot_index, heel)
                for knee, ankle, foot_index, heel in zip(
                    left_knee, left_ankle, left_foot_index, left_heel
                )
            ]
        )

        # Extract nose landmark
        nose = get_vector_landmark(df, "nose")

        # Calculate neck angles
        neck_angles = np.array(
            [
                compute_neck_angle(n, m_shoulder, trunk_vector)
                for n, m_shoulder, trunk_vector in zip(
                    nose, mid_shoulder, trunk_vectors
                )
            ]
        )

        # Extract additional landmarks for wrist angle
        right_pinky = get_vector_landmark(df, "right_pinky")
        right_index = get_vector_landmark(df, "right_index")
        left_pinky = get_vector_landmark(df, "left_pinky")
        left_index = get_vector_landmark(df, "left_index")

        # Calculate wrist angles
        right_wrist_angles = np.array(
            [
                compute_wrist_angle(elbow, wrist, pinky, index)
                for elbow, wrist, pinky, index in zip(
                    right_elbow, right_wrist, right_pinky, right_index
                )
            ]
        )

        left_wrist_angles = np.array(
            [
                compute_wrist_angle(elbow, wrist, pinky, index)
                for elbow, wrist, pinky, index in zip(
                    left_elbow, left_wrist, left_pinky, left_index
                )
            ]
        )

        # Create output DataFrame
        angles_df = pd.DataFrame(
            {
                "frame_index": df.iloc[:, 0],
                "neck_rel": neck_angles,  # Added neck angle
                # Right side angles
                "right_elbow_rel": right_elbow_angles,
                "right_shoulder_rel": right_shoulder_angles,
                "right_hip_rel": right_hip_angles,
                "right_knee_rel": right_knee_angles,
                "right_ankle_rel": right_ankle_angles,
                "right_wrist_rel": right_wrist_angles,  # Added wrist angle
                # Left side angles
                "left_elbow_rel": left_elbow_angles,
                "left_shoulder_rel": left_shoulder_angles,
                "left_hip_rel": left_hip_angles,
                "left_knee_rel": left_knee_angles,
                "left_ankle_rel": left_ankle_angles,
                "left_wrist_rel": left_wrist_angles,  # Added wrist angle
            }
        )

        # Save to CSV
        angles_df.to_csv(output_csv, index=False, float_format="%.2f")
        print(f"\nAngles saved to: {output_csv}")
        print(f"Computed angles: {list(angles_df.columns)[1:]}")  # Skip frame_index

    except Exception as e:
        print(f"Error processing angles: {str(e)}")
        raise


def draw_skeleton_and_angles(frame, landmarks, angles):
    """
    Draw skeleton segments, joints and angle values on the frame.

    Args:
        frame: Video frame (numpy array)
        landmarks: Dictionary containing landmark coordinates
        angles: Dictionary containing angle values
    """
    height, width = frame.shape[:2]

    # Colors
    RED = (0, 0, 255)  # Right side
    BLUE = (255, 0, 0)  # Left side
    GREEN = (0, 255, 0)  # Joints
    WHITE = (255, 255, 255)  # Text

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
        tuple(landmarks["nose"].astype(int)),
        tuple(landmarks["mid_shoulder"].astype(int)),
        WHITE,
        2,
    )  # Neck segment

    # Draw joints (circles)
    joint_radius = 4
    for landmark in landmarks.values():
        cv2.circle(frame, tuple(landmark.astype(int)), joint_radius, GREEN, -1)

    # Add angle values with larger font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7  # Aumentado de 0.5 para 0.7
    thickness = 2  # Aumentado de 1 para 2

    # Neck angle (centralizado no topo)
    cv2.putText(
        frame,
        f"Neck: {angles['neck']:.1f}",
        (width // 2 - 50, 30),
        font,
        font_scale,
        WHITE,
        thickness,
    )

    # Right side angles (in RED)
    cv2.putText(
        frame,
        f"R Shoulder: {angles['right_shoulder']:.1f}",
        (10, 30),
        font,
        font_scale,
        RED,
        thickness,
    )
    cv2.putText(
        frame,
        f"R Elbow: {angles['right_elbow']:.1f}",
        (10, 60),
        font,
        font_scale,
        RED,
        thickness,
    )
    cv2.putText(
        frame,
        f"R Hip: {angles['right_hip']:.1f}",
        (10, 90),
        font,
        font_scale,
        RED,
        thickness,
    )
    cv2.putText(
        frame,
        f"R Knee: {angles['right_knee']:.1f}",
        (10, 120),
        font,
        font_scale,
        RED,
        thickness,
    )
    cv2.putText(
        frame,
        f"R Ankle: {angles['right_ankle']:.1f}",
        (10, 150),
        font,
        font_scale,
        RED,
        thickness,
    )
    cv2.putText(
        frame,
        f"R Wrist: {angles['right_wrist']:.1f}",
        (10, 180),
        font,
        font_scale,
        RED,
        thickness,
    )

    # Left side angles (in BLUE)
    cv2.putText(
        frame,
        f"L Shoulder: {angles['left_shoulder']:.1f}",
        (width - 200, 30),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    cv2.putText(
        frame,
        f"L Elbow: {angles['left_elbow']:.1f}",
        (width - 200, 60),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    cv2.putText(
        frame,
        f"L Hip: {angles['left_hip']:.1f}",
        (width - 200, 90),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    cv2.putText(
        frame,
        f"L Knee: {angles['left_knee']:.1f}",
        (width - 200, 120),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    cv2.putText(
        frame,
        f"L Ankle: {angles['left_ankle']:.1f}",
        (width - 200, 150),
        font,
        font_scale,
        BLUE,
        thickness,
    )
    cv2.putText(
        frame,
        f"L Wrist: {angles['left_wrist']:.1f}",
        (width - 200, 180),
        font,
        font_scale,
        BLUE,
        thickness,
    )

    return frame


def process_video_with_visualization(video_path, csv_path, output_dir):
    """
    Process video file and create visualization with angles using coordinates from CSV.

    Args:
        video_path: Path to input video file
        csv_path: Path to CSV file with pixel coordinates
        output_dir: Directory to save output files
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

    # Create video writer
    output_video_path = os.path.join(
        output_dir, f"visualization_{os.path.basename(video_path)}"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
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

        # Calculate trunk vector
        trunk_vector = np.array(landmarks["mid_hip"]) - np.array(
            landmarks["mid_shoulder"]
        )
        trunk_norm = np.linalg.norm(trunk_vector)
        if trunk_norm > 0:
            trunk_vector = trunk_vector / trunk_norm

        # Calculate angles
        angles = {}

        # Neck angle
        angles["neck"] = compute_neck_angle(
            landmarks["nose"], landmarks["mid_shoulder"], trunk_vector
        )

        # Right side angles
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
            landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"]
        )
        angles["right_ankle"] = compute_ankle_angle(
            landmarks["right_knee"],
            landmarks["right_ankle"],
            landmarks["right_foot_index"],
            landmarks["right_heel"],
        )

        # Left side angles
        angles["left_shoulder"] = compute_shoulder_angle(
            landmarks["left_shoulder"], landmarks["left_elbow"], trunk_vector
        )
        angles["left_elbow"] = compute_elbow_angle(
            landmarks["left_shoulder"], landmarks["left_elbow"], landmarks["left_wrist"]
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

        # Get additional landmarks for wrist angle
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
        landmarks["left_pinky"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_pinky")[0]
        )
        landmarks["left_index"] = np.array(
            get_vector_landmark(df.iloc[frame_count : frame_count + 1], "left_index")[0]
        )

        # Calculate wrist angles
        angles["right_wrist"] = compute_wrist_angle(
            landmarks["right_elbow"],
            landmarks["right_wrist"],
            landmarks["right_pinky"],
            landmarks["right_index"],
        )
        angles["left_wrist"] = compute_wrist_angle(
            landmarks["left_elbow"],
            landmarks["left_wrist"],
            landmarks["left_pinky"],
            landmarks["left_index"],
        )

        # Draw visualization
        frame = draw_skeleton_and_angles(frame, landmarks, angles)

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

    print(f"Video processing complete. Output saved to: {output_video_path}")


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
        filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")],
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

        # Then select video file for visualization
        video_path = select_video_file()
        if not video_path:
            print("No video file selected. Exiting.")
            return

        output_dir = os.path.join(os.path.dirname(video_path), "processed_video")
        process_video_with_visualization(video_path, csv_path, output_dir)

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
                process_angles(input_path, output_path)
                print(f"Successfully processed: {csv_file}")
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
