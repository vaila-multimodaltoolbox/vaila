#!/usr/bin/env python3
"""
===============================================================================
relativeangle.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 31 March 2025
Update Date: 31 March 2025
Version: 0.0.1
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
   - Upper arm angles (shoulder to elbow)
   - Forearm angles (elbow to wrist)
   - Elbow angles (between upper arm and forearm)
   - Thigh angles (hip to knee)
   - Shank angles (knee to ankle)
   - Knee angles (between thigh and shank)
   - Trunk angle (mid-shoulders to mid-hips)

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
from rich import print
import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path


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


def compute_relative_angle(p1, p2, p3):
    """
    Calculates the relative angle (in degrees) between vector p1->p2 and p2->p3.
    Uses cross product and dot product for robust angle calculation.

    Args:
        p1 (list): [x, y] coordinates of first point
        p2 (list): [x, y] coordinates of second (joint) point
        p3 (list): [x, y] coordinates of third point

    Returns:
        float: Angle in degrees (0 to 180)
    """
    # Create vectors
    u = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    # Normalize vectors
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    # Check for zero-length vectors
    if u_norm == 0 or v_norm == 0:
        return np.nan

    u = u / u_norm
    v = v / v_norm

    # Calculate cross and dot products
    cross = u[0] * v[1] - u[1] * v[0]
    dot = np.dot(u, v)

    # Calculate angle
    angle = np.degrees(np.arctan2(cross, dot))

    # Ensure angle is positive
    if angle < 0:
        angle = 180 + angle

    return angle


def compute_midpoint(p1, p2):
    """
    Calculates the midpoint between two points.

    Args:
        p1 (list): [x, y] coordinates of first point
        p2 (list): [x, y] coordinates of second point

    Returns:
        list: [x, y] coordinates of midpoint
    """
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


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

        # Initialize dictionary for angles
        angles_dict = {"frame_index": df.iloc[:, 0]}

        # Define segments to process
        all_segments = ["left_arm", "right_arm", "left_leg", "right_leg", "trunk"]
        if segments is None:
            segments = all_segments
        else:
            segments = [seg.strip() for seg in segments.split(",")]
            invalid_segments = [seg for seg in segments if seg not in all_segments]
            if invalid_segments:
                raise ValueError(f"Invalid segments: {invalid_segments}")

        print(f"Processing segments: {segments}")

        # Process each frame
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"Processing frame {idx}/{len(df)}")

            # Process left arm
            if "left_arm" in segments:
                shoulder = [row["left_shoulder_x"], row["left_shoulder_y"]]
                elbow = [row["left_elbow_x"], row["left_elbow_y"]]
                wrist = [row["left_wrist_x"], row["left_wrist_y"]]

                angles_dict.setdefault("left_upper_arm_abs", []).append(
                    compute_absolute_angle(shoulder, elbow)
                )
                angles_dict.setdefault("left_forearm_abs", []).append(
                    compute_absolute_angle(elbow, wrist)
                )
                angles_dict.setdefault("left_elbow_rel", []).append(
                    compute_relative_angle(shoulder, elbow, wrist)
                )

            # Process right arm
            if "right_arm" in segments:
                shoulder = [row["right_shoulder_x"], row["right_shoulder_y"]]
                elbow = [row["right_elbow_x"], row["right_elbow_y"]]
                wrist = [row["right_wrist_x"], row["right_wrist_y"]]

                angles_dict.setdefault("right_upper_arm_abs", []).append(
                    compute_absolute_angle(shoulder, elbow)
                )
                angles_dict.setdefault("right_forearm_abs", []).append(
                    compute_absolute_angle(elbow, wrist)
                )
                angles_dict.setdefault("right_elbow_rel", []).append(
                    compute_relative_angle(shoulder, elbow, wrist)
                )

            # Process left leg
            if "left_leg" in segments:
                hip = [row["left_hip_x"], row["left_hip_y"]]
                knee = [row["left_knee_x"], row["left_knee_y"]]
                ankle = [row["left_ankle_x"], row["left_ankle_y"]]

                angles_dict.setdefault("left_thigh_abs", []).append(
                    compute_absolute_angle(hip, knee)
                )
                angles_dict.setdefault("left_shank_abs", []).append(
                    compute_absolute_angle(knee, ankle)
                )
                angles_dict.setdefault("left_knee_rel", []).append(
                    compute_relative_angle(hip, knee, ankle)
                )

            # Process right leg
            if "right_leg" in segments:
                hip = [row["right_hip_x"], row["right_hip_y"]]
                knee = [row["right_knee_x"], row["right_knee_y"]]
                ankle = [row["right_ankle_x"], row["right_ankle_y"]]

                angles_dict.setdefault("right_thigh_abs", []).append(
                    compute_absolute_angle(hip, knee)
                )
                angles_dict.setdefault("right_shank_abs", []).append(
                    compute_absolute_angle(knee, ankle)
                )
                angles_dict.setdefault("right_knee_rel", []).append(
                    compute_relative_angle(hip, knee, ankle)
                )

            # Process trunk
            if "trunk" in segments:
                left_shoulder = [row["left_shoulder_x"], row["left_shoulder_y"]]
                right_shoulder = [row["right_shoulder_x"], row["right_shoulder_y"]]
                left_hip = [row["left_hip_x"], row["left_hip_y"]]
                right_hip = [row["right_hip_x"], row["right_hip_y"]]

                mid_shoulder = compute_midpoint(left_shoulder, right_shoulder)
                mid_hip = compute_midpoint(left_hip, right_hip)

                angles_dict.setdefault("trunk_abs", []).append(
                    compute_absolute_angle(mid_hip, mid_shoulder)
                )

        # Create DataFrame with computed angles
        angles_df = pd.DataFrame(angles_dict)

        # Save to CSV
        angles_df.to_csv(output_csv, index=False, float_format="%.2f")
        print(f"\nAngles saved to: {output_csv}")
        print(f"Computed angles: {list(angles_df.columns)[1:]}")  # Skip frame_index

    except Exception as e:
        print(f"Error processing angles: {str(e)}")
        raise


def run_relativeangle():
    parser = argparse.ArgumentParser(
        description="Calculate angles from landmark coordinates."
    )
    parser.add_argument(
        "input_csv", help="Path to input CSV file with landmark coordinates"
    )
    parser.add_argument(
        "output_csv", help="Path to output CSV file for computed angles"
    )
    parser.add_argument(
        "--segments",
        help="Comma-separated list of segments to analyze "
        "(options: left_arm,right_arm,left_leg,right_leg,trunk)",
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input_csv}")
        return

    # Process angles
    try:
        process_angles(args.input_csv, args.output_csv, args.segments)
    except Exception as e:
        print(f"Error: {str(e)}")
        return


if __name__ == "__main__":
    run_relativeangle()
