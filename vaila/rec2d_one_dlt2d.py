"""
================================================================================
Script: rec2d_one_dlt2d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.3
Created: August 9, 2024
Last Updated: August 02, 2025

Description:
    Optimized batch processing of 2D coordinates reconstruction using corresponding DLT parameters for each frame.
    Processes multiple CSV files containing pixel coordinates and reconstructs them to 2D real-world coordinates.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    Uses DLT2D parameters that can vary per frame.

    Optimizations:
    - Pre-allocated NumPy arrays to eliminate dynamic memory allocation
    - Progress tracking for large datasets
    - Reduced debug output for cleaner processing feedback
    - User-defined output directory and data frequency upfront
    - Vectorized operations for better performance
"""

import os
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
from numpy.linalg import inv
from rich import print


def read_coordinates(file_path, usecols=None):
    df = pd.read_csv(file_path, usecols=usecols)
    coordinates = df.to_numpy()  # Não descartar NaN aqui
    return coordinates


def rec2d(A, cc2d):
    nlin = np.size(cc2d, 0)
    H = np.matrix(np.zeros((nlin, 2)))
    for k in range(nlin):
        x = cc2d[k, 0]
        y = cc2d[k, 1]
        cc2d1 = np.matrix([[A[0] - x * A[6], A[1] - x * A[7]], [A[3] - y * A[6], A[4] - y * A[7]]])
        cc2d2 = np.matrix([[x - A[2]], [y - A[5]]])
        G1 = inv(cc2d1) * cc2d2
        H[k, :] = G1.transpose()
    return np.asarray(H)


def process_files_in_directory(dlt_params, input_directory, output_directory, data_rate):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_directory, f"vaila_rec2d_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".csv")])

    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in the selected directory!")
        return

    print(f"Found {len(csv_files)} CSV files to process")

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

        # Pre-allocate array: frame + (x,y) for each coordinate pair
        total_cols = 1 + (num_coords * 2)  # frame + x,y for each point
        rec_coords_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)

        # Set frame numbers in first column
        rec_coords_array[:, 0] = pixel_coords_df["frame"].values

        # Process each frame with pre-allocated array
        for i, row in pixel_coords_df.iterrows():
            pixel_coords = row[1:].to_numpy().reshape(-1, 2)
            if not np.isnan(pixel_coords).all():
                rec2d_coords = rec2d(dlt_params, pixel_coords)
                # Fill the pre-allocated array directly
                rec_coords_array[i, 1:] = rec2d_coords.flatten()
            # NaN values already pre-allocated, so skip invalid data

        # Convert to DataFrame with original column names
        rec_coords_df = pd.DataFrame(rec_coords_array, columns=pixel_coords_df.columns)

        # Save with timestamp
        output_file = os.path.join(output_dir, f"{os.path.splitext(csv_file)[0]}_{timestamp}.2d")
        rec_coords_df.to_csv(output_file, index=False, float_format="%.6f")

    print("\n=== Processing Complete ===")
    print(f"Processed {total_files} files")
    print(f"Data rate used: {data_rate} Hz")
    print(f"Output directory: {output_dir}")

    messagebox.showinfo(
        "Processing Complete",
        f"2D reconstruction completed successfully!\n\n"
        f"Processed: {total_files} files\n"
        f"Data rate: {data_rate} Hz\n"
        f"Output directory: {os.path.basename(output_dir)}",
    )
    print(f"Reconstructed 2D coordinates saved to {output_dir}")


def run_rec2d_one_dlt2d():
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec2d_one_dlt2d.py...")
    print("-" * 80)

    root = Tk()
    root.withdraw()

    # Step 1: Select DLT parameters file
    print("Step 1: Selecting DLT parameters file...")
    dlt_file = filedialog.askopenfilename(
        title="Select DLT Parameters File", filetypes=[("DLT2D files", "*.dlt2d")]
    )
    if not dlt_file:
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

    # Load and validate DLT parameters
    print("Loading DLT parameters...")
    dlt_params_df = pd.read_csv(dlt_file)

    if dlt_params_df.shape[0] < 1:
        messagebox.showerror("Error", "DLT file should contain at least one set of DLT parameters.")
        return

    # Use the first set of DLT parameters
    dlt_params = dlt_params_df.iloc[0, 1:].to_numpy()

    print("Configuration complete:")
    print(f"  - DLT file: {os.path.basename(dlt_file)}")
    print(f"  - Input directory: {input_directory}")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {data_rate} Hz")
    print("-" * 80)

    # Process files
    process_files_in_directory(dlt_params, input_directory, output_directory, data_rate)

    root.destroy()


if __name__ == "__main__":
    run_rec2d_one_dlt2d()
