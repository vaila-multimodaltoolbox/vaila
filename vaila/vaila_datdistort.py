"""
===============================================================================
vaila_datdistort.py
===============================================================================
Author: Based on vaila_lensdistortvideo.py by Prof. Paulo R. P. Santiago
Date: March 2024
===============================================================================

This script applies lens distortion correction to 2D coordinates from a DAT file
using the same camera calibration parameters as vaila_lensdistortvideo.py.
"""

import cv2
import numpy as np
import pandas as pd
import os
from rich import print
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import subprocess
import sys


def load_distortion_parameters(csv_path):
    """Load distortion parameters from a CSV file."""
    df = pd.read_csv(csv_path)
    return df.iloc[0].to_dict()


def undistort_points(points, camera_matrix, dist_coeffs, image_size):
    """
    Undistort 2D points using camera calibration parameters.
    
    Args:
        points: Nx2 array of (x,y) coordinates
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        image_size: (width, height) of the original image
    
    Returns:
        Nx2 array of undistorted (x,y) coordinates
    """
    # Convert points to float32 numpy array
    points = np.array(points, dtype=np.float32)
    if points.size == 0:  # Check if points array is empty
        return points
        
    # Ensure camera matrix and dist_coeffs are float32
    camera_matrix = np.array(camera_matrix, dtype=np.float32)
    dist_coeffs = np.array(dist_coeffs, dtype=np.float32)
    
    # Reshape points to Nx1x2 format required by cv2.undistortPoints
    points = points.reshape(-1, 1, 2)
    
    # Get optimal new camera matrix
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 1, image_size
    )
    
    try:
        # Undistort points
        undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, None, new_camera_matrix)
        # Reshape back to Nx2
        return undistorted.reshape(-1, 2)
    except cv2.error as e:
        print(f"Error undistorting points: {e}")
        print(f"Points shape: {points.shape}")
        print(f"Points dtype: {points.dtype}")
        print(f"Camera matrix shape: {camera_matrix.shape}")
        print(f"Distortion coefficients shape: {dist_coeffs.shape}")
        raise


def process_dat_file(input_path, output_path, parameters, image_size=(1920, 1080)):
    """
    Process a DAT/CSV file to apply lens distortion correction to coordinates.
    """
    # Read the DAT file - try both comma and semicolon as separators
    try:
        df = pd.read_csv(input_path)
    except:
        df = pd.read_csv(input_path, sep=';')
    
    # Create camera matrix
    camera_matrix = np.array([
        [parameters["fx"], 0, parameters["cx"]],
        [0, parameters["fy"], parameters["cy"]],
        [0, 0, 1]
    ])
    
    # Create distortion coefficients array
    dist_coeffs = np.array([
        parameters["k1"],
        parameters["k2"],
        parameters["p1"],
        parameters["p2"],
        parameters["k3"]
    ])
    
    # Get all x,y column pairs (assuming format p1_x, p1_y, p2_x, p2_y, etc)
    columns = df.columns.tolist()
    x_columns = [col for col in columns if col.endswith('_x')]
    y_columns = [col for col in columns if col.endswith('_y')]
    
    # Sort columns to ensure matching pairs
    x_columns.sort()
    y_columns.sort()
    
    result_frames = []
    
    # Process each frame
    for _, row in df.iterrows():
        frame_num = row['frame']
        
        # Collect valid points for this frame
        points = []
        for x_col, y_col in zip(x_columns, y_columns):
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                # Only include valid coordinates (not 0,0 or NaN)
                if pd.notna(x) and pd.notna(y) and not (x == 0 and y == 0):
                    points.append([x, y])
            except (ValueError, TypeError):
                continue
        
        points = np.array(points)
        
        # Skip if no valid points
        if len(points) == 0:
            result_frames.append(row.to_dict())
            continue
            
        # Undistort valid points
        try:
            undistorted_points = undistort_points(points, camera_matrix, dist_coeffs, image_size)
            
            # Create new row with undistorted coordinates
            new_row = {'frame': frame_num}
            point_idx = 0
            
            # Reconstruct all columns, replacing coordinates with undistorted ones
            for x_col, y_col in zip(x_columns, y_columns):
                if point_idx < len(undistorted_points):
                    # Get original values
                    orig_x = row[x_col]
                    orig_y = row[y_col]
                    
                    # Only update if original point was valid
                    if pd.notna(orig_x) and pd.notna(orig_y) and not (orig_x == 0 and orig_y == 0):
                        new_row[x_col] = undistorted_points[point_idx][0]
                        new_row[y_col] = undistorted_points[point_idx][1]
                        point_idx += 1
                    else:
                        # Keep original invalid/zero values
                        new_row[x_col] = orig_x
                        new_row[y_col] = orig_y
                else:
                    # Keep original values for any remaining columns
                    new_row[x_col] = row[x_col]
                    new_row[y_col] = row[y_col]
                
            result_frames.append(new_row)
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            result_frames.append(row.to_dict())
    
    # Create output DataFrame and save
    result_df = pd.DataFrame(result_frames)
    result_df.to_csv(output_path, index=False)


def select_file(title="Select a file", filetypes=(("All Files", "*.*"),)):
    """Open a dialog to select a file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path


def select_directory(title="Select a directory"):
    """Open a dialog to select a directory."""
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title=title)
    return dir_path


def run_datdistort():
    """Main function to process DAT/CSV files using distortion parameters."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    print("Select the distortion parameters CSV file:")
    parameters_path = select_file(
        title="Select Calibration Parameters File",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    if not parameters_path:
        print("No parameters file selected. Exiting.")
        return

    print("Select the directory containing CSV/DAT files to process:")
    input_dir = select_directory(title="Select Directory with CSV/DAT Files")
    if not input_dir:
        print("No directory selected. Exiting.")
        return

    # Load parameters once
    parameters = load_distortion_parameters(parameters_path)

    # Process all CSV and DAT files in the directory
    processed_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = os.path.join(input_dir, f"distorted_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.csv', '.dat')):
            input_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}_distorted.csv")

            try:
                print(f"\nProcessing: {filename}")
                process_dat_file(input_path, output_path, parameters)
                print(f"Saved as: {os.path.basename(output_path)}")
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    print(f"\nProcessing complete!")
    print(f"Files processed: {processed_count}")
    print(f"Output directory: {output_dir}")

    # Open output directory in file explorer
    if processed_count > 0:
        try:
            if os.name == 'nt':  # Windows
                os.startfile(output_dir)
            elif os.name == 'posix':  # macOS and Linux
                subprocess.run(['xdg-open', output_dir])
        except:
            pass  # Ignore if can't open file explorer


if __name__ == "__main__":
    run_datdistort()
