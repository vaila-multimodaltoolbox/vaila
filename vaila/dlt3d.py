"""
================================================================================
Script: dlt3d.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Roberto Pereira Santiago
Version: 0.0.3
Create: 24 February, 2025
Last Updated: 02 August, 2025

Description:
    This script calculates the Direct Linear Transformation (DLT) parameters for 3D coordinate transformations.
    It uses pixel coordinates from video calibration data and corresponding real-world 3D coordinates to compute the 11
    DLT parameters for each frame (or uses a single row of real-world coordinates for all frames).

    New Features:
      - Generates a REF3D template (with _x, _y, _z columns) from the pixel file.
      - Validates that the REF3D file contains the three axes for each point.
      - Updated calculation of DLT parameters (11 parameters) using least squares.
      - Graphical file selection using Tkinter.
      - Improved console output.

Usage:
    Run the script and select a pixel coordinate CSV file. Then, choose whether to create a REF3D template.
    If you opt to create the template, edit it with the real-world coordinates and run the DLT process again.
    Otherwise, select the edited REF3D file, and the script will calculate the parameters and save an output file
    with the .dlt3d extension.
"""

import os
import numpy as np
import pandas as pd
from rich import print
from tkinter import filedialog, messagebox, Tk


def read_pixel_file(file_path):
    """Reads the pixel coordinate CSV file."""
    df = pd.read_csv(file_path)
    return df


def read_ref3d_file(file_path):
    """Reads the REF3D file and checks if the _z columns are present."""
    df = pd.read_csv(file_path)
    # Dynamically determine the number of points from the input file
    # instead of hardcoding to 25 points
    input_columns = list(df.columns)

    # Find all point columns (p1_x, p1_y, p2_x, p2_y, etc.)
    point_columns = [
        col
        for col in input_columns
        if col.startswith("p") and ("_x" in col or "_y" in col)
    ]

    # Determine the highest point number
    point_numbers = set()
    for col in point_columns:
        # Extract the number from column names like "p1_x", "p20_y", etc.
        if "_" in col:
            parts = col.split("_")
            if len(parts) >= 2:
                point_num = parts[0][1:]  # Remove 'p' from 'p1', 'p20', etc.
                if point_num.isdigit():
                    point_numbers.add(int(point_num))

    num_points = max(point_numbers) if point_numbers else 0

    # Generate expected columns for the 3D reference file
    expected_columns = []
    for i in range(1, num_points + 1):
        expected_columns.extend([f"p{i}_x", f"p{i}_y", f"p{i}_z"])
    if not all(col in df.columns for col in expected_columns):
        print(
            "Error: REF3D file does not contain the expected columns with _z coordinates!"
        )
        return None
    return df


def calculate_dlt3d_params(pixel_coords, ref_coords):
    """
    Computes the 11 DLT3d parameters using the following models:

      u = (L1*X + L2*Y + L3*Z + L4) / (L9*X + L10*Y + L11*Z + 1)
      v = (L5*X + L6*Y + L7*Z + L8) / (L9*X + L10*Y + L11*Z + 1)

    The equations are rearranged to form a linear system:
      X   Y   Z   1   0   0   0   0  -uX  -uY  -uZ = u
      0   0   0   0   X   Y   Z   1  -vX  -vY  -vZ = v
    """
    n = pixel_coords.shape[0]
    A = np.zeros((2 * n, 11))
    B = np.zeros((2 * n,))
    for i in range(n):
        X, Y, Z = ref_coords[i, :]
        u, v = pixel_coords[i, :]
        # First equation (for u)
        A[2 * i, 0:4] = [X, Y, Z, 1]
        A[2 * i, 8:11] = -u * np.array([X, Y, Z])
        B[2 * i] = u
        # Second equation (for v)
        A[2 * i + 1, 4:8] = [X, Y, Z, 1]
        A[2 * i + 1, 8:11] = -v * np.array([X, Y, Z])
        B[2 * i + 1] = v
    # Solve the system A * L = B using least squares
    L, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return L


def process_files(pixel_file, ref3d_file):
    """
    Processes the pixel and REF3D files.
    If the REF3D file contains only one row, the same real-world points are used for every frame.
    """
    pixel_df = read_pixel_file(pixel_file)
    ref_df = read_ref3d_file(ref3d_file)
    if ref_df is None:
        return None

    # Determine the number of points from the pixel file columns
    pixel_columns = list(pixel_df.columns)
    point_columns = [
        col
        for col in pixel_columns
        if col.startswith("p") and ("_x" in col or "_y" in col)
    ]
    point_numbers = set()
    for col in point_columns:
        if "_" in col:
            parts = col.split("_")
            if len(parts) >= 2:
                point_num = parts[0][1:]  # Remove 'p' from 'p1', 'p20', etc.
                if point_num.isdigit():
                    point_numbers.add(int(point_num))

    num_points = max(point_numbers) if point_numbers else 0

    dlt_params_all = {}
    # If the REF3D file consists of only one row, use it for all frames:
    if len(ref_df) == 1:
        ref_coords_arr = []
        ref_line = ref_df.iloc[0]
        for i in range(1, num_points + 1):
            ref_coords_arr.append(
                [ref_line[f"p{i}_x"], ref_line[f"p{i}_y"], ref_line[f"p{i}_z"]]
            )
        ref_coords_arr = np.array(ref_coords_arr)
        for _, row in pixel_df.iterrows():
            pixel_coords_arr = []
            for i in range(1, num_points + 1):
                pixel_coords_arr.append([row[f"p{i}_x"], row[f"p{i}_y"]])
            pixel_coords_arr = np.array(pixel_coords_arr)
            L = calculate_dlt3d_params(pixel_coords_arr, ref_coords_arr)
            frame = row["frame"]
            dlt_params_all[frame] = L
    else:
        # If REF3D contains multiple rows, match the frame numbers
        for _, row in pixel_df.iterrows():
            frame = row["frame"]
            ref_line = ref_df[ref_df["frame"] == frame]
            if ref_line.empty:
                print(f"Frame {frame} not found in REF3D file.")
                continue
            ref_line = ref_line.iloc[0]
            pixel_coords_arr = []
            ref_coords_arr = []
            for i in range(1, num_points + 1):
                pixel_coords_arr.append([row[f"p{i}_x"], row[f"p{i}_y"]])
                ref_coords_arr.append(
                    [ref_line[f"p{i}_x"], ref_line[f"p{i}_y"], ref_line[f"p{i}_z"]]
                )
            pixel_coords_arr = np.array(pixel_coords_arr)
            ref_coords_arr = np.array(ref_coords_arr)
            L = calculate_dlt3d_params(pixel_coords_arr, ref_coords_arr)
            dlt_params_all[frame] = L
    return dlt_params_all


def save_dlt_parameters(output_file, dlt_params):
    """Saves the computed DLT3d parameters to a CSV file without spaces after commas."""
    with open(output_file, "w") as f:
        f.write(
            "frame,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11\n"
        )  # Please verify header names if needed
        for frame, params in dlt_params.items():
            param_str = ",".join([f"{p:.6f}" for p in params])
            f.write(f"{frame},{param_str}\n")
    # Show a message box indicating success
    messagebox.showinfo("Success", f"DLT3d file saved successfully: {output_file}")
    print(f"DLT3d parameters saved to {output_file}")


def main():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting DLT3D module...")

    root = Tk()
    root.withdraw()
    pixel_file = filedialog.askopenfilename(
        title="Select the pixel coordinate file", filetypes=[("CSV files", "*.csv")]
    )
    if not pixel_file:
        print("Pixel file selection canceled.")
        return

    # Determine the number of points from the pixel file
    pixel_df = read_pixel_file(pixel_file)
    pixel_columns = list(pixel_df.columns)
    point_columns = [
        col
        for col in pixel_columns
        if col.startswith("p") and ("_x" in col or "_y" in col)
    ]
    point_numbers = set()
    for col in point_columns:
        if "_" in col:
            parts = col.split("_")
            if len(parts) >= 2:
                point_num = parts[0][1:]  # Remove 'p' from 'p1', 'p20', etc.
                if point_num.isdigit():
                    point_numbers.add(int(point_num))

    num_points = max(point_numbers) if point_numbers else 0

    # Ask the user if they want to generate a REF3D template
    mode = messagebox.askquestion("Mode", "Do you want to create a REF3D template?")
    if mode == "yes":
        real_file = os.path.splitext(pixel_file)[0] + ".ref3d"
        # Create a template with header for 25 points with _x, _y, _z (default value 0.0)
        template_data = {"frame": [0]}
        for i in range(1, num_points + 1):
            template_data[f"p{i}_x"] = [0]
            template_data[f"p{i}_y"] = [0]
            template_data[f"p{i}_z"] = [0]
        template_df = pd.DataFrame(template_data)
        template_df.to_csv(real_file, index=False)
        messagebox.showinfo("Success", f"REF3D template created: {real_file}")
        print(f"REF3D template created: {real_file}")
        print(
            "Please edit the REF3D file with the real coordinates and run the DLT process again."
        )
        return
    else:
        real_file = filedialog.askopenfilename(
            title="Select the real 3D coordinates file",
            filetypes=[("REF3D files", "*.ref3d")],
        )
        if not real_file:
            print("Real file selection canceled.")
            return

    dlt_params = process_files(pixel_file, real_file)
    if dlt_params is None:
        print("Error processing the files.")
        return
    output_file = os.path.splitext(pixel_file)[0] + ".dlt3d"
    save_dlt_parameters(output_file, dlt_params)


if __name__ == "__main__":
    main()
