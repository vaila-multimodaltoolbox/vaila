"""
===============================================================================
mpangles.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 31 March 2025
Update Date: 03 April 2025
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

import os
from rich import print
import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

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
            title="Select Directory with CSV Files",
            mustexist=True
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
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
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
                'data': data_array,
                'output_path': os.path.join(output_dir, f"processed_{csv_file}")
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
    landmark_names = ["nose",
                     "left_eye_inner", "left_eye", "left_eye_outer",
                     "right_eye_inner", "right_eye", "right_eye_outer",
                     "left_ear", "right_ear",
                     "mouth_left", "mouth_right",
                     "left_shoulder", "right_shoulder",
                     "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist",
                     "left_pinky", "right_pinky",
                     "left_index", "right_index",   
                     "left_thumb", "right_thumb",
                     "left_hip", "right_hip",
                     "left_knee", "right_knee",
                     "left_ankle", "right_ankle",
                     "left_heel", "right_heel",
                     "left_foot_index", "right_foot_index"]
    
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
        
        # Get landmarks for angle calculation
        # Example for left arm angles:
        left_shoulder = get_vector_landmark(df, "left_shoulder")
        left_elbow = get_vector_landmark(df, "left_elbow")
        left_wrist = get_vector_landmark(df, "left_wrist")
        
        # Calculate angles
        # Absolute angle of upper arm (shoulder to elbow)
        left_upper_arm_angles = np.array([
            compute_absolute_angle(shoulder, elbow) 
            for shoulder, elbow in zip(left_shoulder, left_elbow)
        ])
        
        # Absolute angle of forearm (elbow to wrist)
        left_forearm_angles = np.array([
            compute_absolute_angle(elbow, wrist)
            for elbow, wrist in zip(left_elbow, left_wrist)
        ])
        
        # Relative angle at elbow
        left_elbow_angles = np.array([
            compute_relative_angle(shoulder, elbow, wrist)
            for shoulder, elbow, wrist in zip(left_shoulder, left_elbow, left_wrist)
        ])
        
        # Para extrair outros marcadores:
        right_shoulder = get_vector_landmark(df, "right_shoulder")
        right_hip = get_vector_landmark(df, "right_hip")
        right_knee = get_vector_landmark(df, "right_knee")

        # Para calcular ângulos do quadril direito:
        right_hip_angles = np.array([
            compute_absolute_angle(hip, knee)
            for hip, knee in zip(right_hip, right_knee)
        ])
        
        # Create output DataFrame
        angles_df = pd.DataFrame({
            'frame_index': df.iloc[:, 0],
            'left_upper_arm_abs': left_upper_arm_angles,
            'left_forearm_abs': left_forearm_angles,
            'left_elbow_rel': left_elbow_angles,
            'right_hip_abs': right_hip_angles
        })
        
        # Save to CSV
        angles_df.to_csv(output_csv, index=False, float_format="%.2f")
        print(f"\nAngles saved to: {output_csv}")
        print(f"Computed angles: {list(angles_df.columns)[1:]}")  # Skip frame_index

    except Exception as e:
        print(f"Error processing angles: {str(e)}")
        raise

def run_mp_angles():
    """Runs the MP Angles module with directory selection for batch processing."""
    print("\nStarting MP Angles module...")
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Select directory containing CSV files
    input_dir = select_directory()
    if not input_dir:
        print("No input directory selected. Exiting.")
        return

    # Process all CSV files in the directory
    processed_files = process_directory(input_dir)
    if not processed_files:
        print("No files were processed. Exiting.")
        return

    # Process each file in the directory
    for csv_file, file_info in processed_files.items():
        try:
            input_path = os.path.join(input_dir, csv_file)
            output_path = file_info['output_path']
            
            print(f"\nProcessing file: {csv_file}")
            print(f"Output will be saved to: {output_path}")
            
            # Process angles for all segments
            process_angles(input_path, output_path)
            
            print(f"Successfully processed: {csv_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue

    print("\nProcessing complete!")
    print(f"Output files are saved in: {os.path.dirname(output_path)}")

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
