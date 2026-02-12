"""
================================================================================
Script: rec3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.2
Created: August 03, 2025
Last Updated: August 03, 2025

Description:
    Optimized batch processing of 3D coordinates reconstruction using corresponding DLT3D parameters for each frame.
    Processes multiple CSV files containing pixel coordinates and reconstructs them to 3D real-world coordinates.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    Uses DLT3D parameters that can vary per frame (11 parameters per frame).

    Optimizations:
    - Pre-allocated NumPy arrays to eliminate dynamic memory allocation
    - Progress tracking for large datasets
    - Reduced debug output for cleaner processing feedback
    - User-defined output directory and data frequency upfront
    - Vectorized operations for better performance
    - Support for multiple cameras with different DLT3D parameters per frame
"""

import os
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from rich import print


def rec3d_multicam(dlt_list, pixel_list):
    """
    Reconstructs a 3D point using multiple camera observations and their corresponding DLT3D parameters.

    Args:
        dlt_list (list of np.array): List of DLT3D parameter arrays (each of 11 elements) for each camera.
        pixel_list (list of tuple): List of observed pixel coordinates (x, y) for each camera.

    Returns:
        np.array: Reconstructed 3D point [X, Y, Z] using a least squares solution.
    """
    num_cameras = len(dlt_list)
    A_matrix = np.zeros((num_cameras * 2, 3))
    b_vector = np.zeros(num_cameras * 2)

    for i, (A_params, (x, y)) in enumerate(zip(dlt_list, pixel_list, strict=False)):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = A_params

        # Equations for camera i:
        # (a1 - x*a9)*X + (a2 - x*a10)*Y + (a3 - x*a11)*Z = x - a4
        # (a5 - y*a9)*X + (a6 - y*a10)*Y + (a7 - y*a11)*Z = y - a8
        row_idx = i * 2
        A_matrix[row_idx] = [a1 - x * a9, a2 - x * a10, a3 - x * a11]
        A_matrix[row_idx + 1] = [a5 - y * a9, a6 - y * a10, a7 - y * a11]
        b_vector[row_idx] = x - a4
        b_vector[row_idx + 1] = y - a8

    solution, residuals, rank, s = lstsq(A_matrix, b_vector, rcond=None)
    return solution  # [X, Y, Z]


def process_files_in_directory(dlt_params_dfs, input_directory, output_directory, data_rate):
    """
    Process multiple CSV files in a directory using multiple DLT3D parameter sets.

    Args:
        dlt_params_dfs (list): List of DataFrames containing DLT3D parameters for each camera
        input_directory (str): Directory containing CSV files to process
        output_directory (str): Directory to save output files
        data_rate (int): Data frequency in Hz
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_directory, f"vaila_rec3d_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Extract DLT parameters for each camera
    dlt_params_list = []
    frames_list = []
    for df in dlt_params_dfs:
        dlt_params = df.to_numpy()
        frames = dlt_params[:, 0]
        dlt_params = dlt_params[:, 1:]  # Remove frame column, keep 11 DLT parameters
        dlt_params_list.append(dlt_params)
        frames_list.append(frames)

    csv_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".csv")])

    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in the selected directory!")
        return

    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Using {len(dlt_params_list)} camera DLT parameter sets")

    total_files = len(csv_files)
    files_processed = 0

    for csv_file in csv_files:
        files_processed += 1
        progress = (files_processed / total_files) * 100
        print(f"Processing file {files_processed}/{total_files} ({progress:.1f}%): {csv_file}")

        pixel_file = os.path.join(input_directory, csv_file)
        pixel_coords_df = pd.read_csv(pixel_file)

        # Calculate number of coordinate pairs (excluding frame column)
        num_coords = (pixel_coords_df.shape[1] - 1) // 2
        total_frames = len(pixel_coords_df)

        # Pre-allocate array: frame + (x,y,z) for each coordinate pair
        total_cols = 1 + (num_coords * 3)  # frame + x,y,z for each point
        rec_coords_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)

        # Set frame numbers in first column
        rec_coords_array[:, 0] = pixel_coords_df["frame"].values

        # Process each frame with pre-allocated array
        for i, row in pixel_coords_df.iterrows():
            frame_num = int(row["frame"])

            # Check if frame exists in all DLT parameter sets
            frame_exists_in_all = True
            dlt_params_for_frame = []

            for _camera_idx, (dlt_params, frames) in enumerate(zip(dlt_params_list, frames_list, strict=False)):
                if frame_num in frames:
                    A_index = np.where(frames == frame_num)[0][0]
                    A = dlt_params[A_index]
                    if not np.isnan(A).any():
                        dlt_params_for_frame.append(A)
                    else:
                        frame_exists_in_all = False
                        break
                else:
                    frame_exists_in_all = False
                    break

            if frame_exists_in_all and len(dlt_params_for_frame) == len(dlt_params_list):
                # Process each marker for this frame
                for marker in range(1, num_coords + 1):
                    pixel_obs_list = []
                    valid_marker = True

                    for _camera_idx in range(len(dlt_params_list)):
                        try:
                            x_obs = float(row[f"p{marker}_x"])
                            y_obs = float(row[f"p{marker}_y"])
                            if np.isnan(x_obs) or np.isnan(y_obs):
                                valid_marker = False
                                break
                            pixel_obs_list.append((x_obs, y_obs))
                        except:
                            valid_marker = False
                            break

                    if valid_marker and len(pixel_obs_list) == len(dlt_params_for_frame):
                        # Calculate 3D reconstruction
                        point3d = rec3d_multicam(dlt_params_for_frame, pixel_obs_list)

                        # Fill the pre-allocated array directly
                        col_start = 1 + (marker - 1) * 3  # x, y, z columns for this marker
                        rec_coords_array[i, col_start : col_start + 3] = point3d
                # NaN values already pre-allocated for invalid frames/markers

        # Convert to DataFrame with original column names but with _z added
        header = ["frame"]
        for marker in range(1, num_coords + 1):
            header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])

        rec_coords_df = pd.DataFrame(rec_coords_array, columns=header)

        output_file = os.path.join(output_dir, f"{os.path.splitext(csv_file)[0]}_{timestamp}.3d")
        rec_coords_df.to_csv(output_file, index=False, float_format="%.6f")

    print("\n=== Processing Complete ===")
    print(f"Processed {total_files} files")
    print(f"Data rate used: {data_rate} Hz")
    print(f"Output directory: {output_dir}")

    messagebox.showinfo(
        "Processing Complete",
        f"3D reconstruction completed successfully!\n\n"
        f"Processed: {total_files} files\n"
        f"Data rate: {data_rate} Hz\n"
        f"Output directory: {os.path.basename(output_dir)}",
    )
    print(f"Reconstructed 3D coordinates saved to {output_dir}")


def run_rec3d():
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec3d.py...")
    print("-" * 80)

    root = Tk()
    root.withdraw()

    # Step 1: Select DLT3D parameters files (multiple cameras)
    print("Step 1: Selecting DLT3D parameters files...")
    dlt_files = filedialog.askopenfilenames(
        title="Select DLT3D Parameters Files (one per camera)",
        filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")],
    )
    if not dlt_files:
        print("DLT file selection cancelled.")
        return

    # Step 2: Select input directory with CSV files
    print("Step 2: Selecting input directory...")
    input_directory = filedialog.askdirectory(title="Select Directory Containing CSV Files")
    if not input_directory:
        print("Input directory selection cancelled.")
        return

    # Step 3: Select output directory
    print("Step 3: Selecting output directory...")
    output_directory = filedialog.askdirectory(title="Select Output Directory for Results")
    if not output_directory:
        print("Output directory selection cancelled.")
        return

    # Step 4: Ask for data frequency
    print("Step 4: Setting data frequency...")
    data_rate = simpledialog.askinteger(
        "Data Frequency", "Enter the data frequency (Hz):", minvalue=1, initialvalue=100
    )
    if data_rate is None:
        messagebox.showerror("Error", "Data frequency is required. Operation cancelled.")
        return

    # Load and validate DLT parameters for each camera
    print("Loading DLT3D parameters...")
    dlt_params_dfs = []
    for dlt_file in dlt_files:
        df = pd.read_csv(dlt_file)
        if df.empty:
            messagebox.showerror("Error", f"DLT3D file {os.path.basename(dlt_file)} is empty!")
            return
        dlt_params_dfs.append(df)

    print("Configuration complete:")
    print(f"  - DLT3D files: {len(dlt_files)} cameras")
    print(f"  - Input directory: {input_directory}")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {data_rate} Hz")
    for i, df in enumerate(dlt_params_dfs):
        print(f"  - Camera {i + 1}: {len(df)} frames")
    print("-" * 80)

    # Process files
    process_files_in_directory(dlt_params_dfs, input_directory, output_directory, data_rate)

    root.destroy()


if __name__ == "__main__":
    run_rec3d()
