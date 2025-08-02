"""
================================================================================
Script: dlt2d.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.0.3
Created: November 26, 2024  
Last Updated: August 02, 2025
================================================================================
Description:
    This script calculates the Direct Linear Transformation (DLT) parameters for 2D coordinate transformations.
    It uses pixel coordinates from video calibration data and corresponding real-world coordinates to compute
    the DLT parameters for each frame in a dataset.

    The script also allows users to create a template REF2D file from a pixel file, which can then be edited
    manually to include real-world coordinates. The main functionality includes reading input files,
    performing DLT calculations, and saving the results to an output file.

New Features:
    - Automatic generation of REF2D templates from pixel coordinate files.
    - Validation of input coordinate pairs to ensure compatibility before processing.
    - Detailed logging of DLT parameter calculation for each frame.
    - User-friendly graphical interface for file selection using Tkinter.
    - Integration with the Rich library for enhanced console output.

Usage:
    1. Run the script to start the Direct Linear Transformation (DLT) process.
    2. A graphical interface will prompt you to select:
       - A pixel coordinate file (CSV format) for calibration.
       - (Optional) Create a REF2D template file or use an existing REF2D file for real-world coordinates.
    3. The script processes the input files and calculates the DLT parameters.
    4. The calculated parameters are saved as a `.dlt2d` CSV file in the same directory as the input pixel file.

How to Execute:
    1. Ensure you have all dependencies installed:
       - Install numpy: `pip install numpy`
       - Install pandas: `pip install pandas`
       - Install rich: `pip install rich`
    2. Open a terminal and navigate to the directory where `dlt2d.py` is located.
    3. Run the script using Python:

       python dlt2d.py

    4. Follow the graphical prompts to select input files and process the data.

Requirements:
    - Python 3.11.9
    - Numpy (`pip install numpy`)
    - Pandas (`pip install pandas`)
    - Rich (`pip install rich`)
    - Tkinter (usually included with Python installations)

Output:
    - REF2D Template File (`*.ref2d`):
      A template file for real-world coordinates created from the pixel file.
    - DLT Parameters File (`*.dlt2d`):
      A CSV file containing the DLT parameters for each processed frame.

Output Structure:
    - Frame: The frame index from the input pixel file.
    - DLT Parameters: The 8 calculated DLT parameters for each frame.

Example Workflow:
    1. Select a pixel coordinate CSV file containing calibration data.
    2. Choose to create a REF2D template file or use an existing REF2D file with real-world coordinates.
    3. Edit the REF2D template file to include the real-world coordinates.
    4. Re-run the script and process the files to generate the DLT parameters.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of
    the GNU General Public License as published by the Free Software Foundation, either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU General Public License for more details.

    You should have received a copy of the GNU GPLv3 (General Public License Version 3) along with this program.
    If not, see <https://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import pandas as pd
import csv
from numpy.linalg import inv
from tkinter import filedialog, Tk, messagebox
from rich import print


def read_coordinates(file_path, usecols):
    """
    Function to read coordinates from a CSV file.

    Args:
    file_path (str): Path to the CSV file.
    usecols (list or function): Columns to be used.

    Returns:
    np.array: Array containing the coordinates.
    """
    df = pd.read_csv(file_path, usecols=usecols)
    coordinates = df.to_numpy()  # Keep all rows, including NaNs
    return coordinates


def dlt2d(F, L):
    """
    Calculate DLT (2D) parameters.

    Args:
    F (np.array): Matrix containing real-world global coordinates (X, Y).
    L (np.array): Matrix containing pixel coordinates of calibration points.

    Returns:
    np.array: DLT parameters.
    """
    F = np.matrix(F)
    L = np.matrix(L)
    Lt = L.transpose()
    C = Lt.flatten("F").transpose()
    m = np.size(F, 0)

    B = np.zeros((2 * m, 8))
    for i in range(m):
        B[2 * i, 0:3] = [F[i, 0], F[i, 1], 1]
        B[2 * i, 6:8] = [-F[i, 0] * L[i, 0], -F[i, 1] * L[i, 0]]
        B[2 * i + 1, 3:6] = [F[i, 0], F[i, 1], 1]
        B[2 * i + 1, 6:8] = [-F[i, 0] * L[i, 1], -F[i, 1] * L[i, 1]]

    A = inv(B.T @ B) @ B.T @ C
    return np.asarray(A).flatten()


def create_ref2d_template(pixel_file):
    """
    Create a REF2D template based on the pixel file.

    Args:
    pixel_file (str): Path to the pixel file.

    Returns:
    str: Path to the generated REF2D file.
    """
    df = pd.read_csv(pixel_file)
    template = df.copy()
    template.iloc[:, 1:] = np.nan  # Clear all coordinate data but keep frame numbers
    template_file = os.path.splitext(pixel_file)[0] + ".ref2d"
    template.to_csv(template_file, index=False)
    return template_file


def filter_and_shape_coordinates(coords):
    """
    Filter coordinates and remove NaNs.

    Args:
    coords (np.array): Array of coordinates.

    Returns:
    np.array: Filtered and reshaped coordinates.
    tuple: Indices of valid coordinate pairs.
    """
    valid_pairs = []
    filtered_coords = []

    # Process coordinates in pairs (x,y)
    for i in range(0, len(coords), 2):
        if (
            i + 1 < len(coords)
            and not np.isnan(coords[i])
            and not np.isnan(coords[i + 1])
        ):
            filtered_coords.append([coords[i], coords[i + 1]])
            valid_pairs.append(i // 2)  # Store the index of the valid pair

    return np.array(filtered_coords), valid_pairs


def process_files(pixel_file, real_file):
    """
    Process the coordinate files to calculate the DLT parameters.
    Supports two scenarios:
    1. Single reference line in .ref2d file - applies to all frames
    2. Multiple reference lines - each line corresponds to a frame in the pixel file

    Args:
    pixel_file (str): Path to the pixel coordinate file.
    real_file (str): Path to the real-world coordinate file.

    Returns:
    list: List of DLT parameters for each frame.
    """
    # Read the full dataframes
    pixel_df = pd.read_csv(pixel_file)
    real_df = pd.read_csv(real_file)

    # Ensure both files have the same column structure
    if len(pixel_df.columns) != len(real_df.columns):
        print("The number of columns in the two files must match.")
        return

    # Extract coordinate column names (excluding 'frame')
    coord_cols = [col for col in pixel_df.columns if col != "frame"]

    # Check if we're using a single reference row for all frames
    single_ref_mode = len(real_df) == 1
    if single_ref_mode:
        print(
            "Single reference mode: Using the same reference coordinates for all frames"
        )
        # Store the single reference row for repeated use
        ref_row = real_df.iloc[0]
    elif len(real_df) != len(pixel_df):
        print(
            f"Warning: Pixel file has {len(pixel_df)} frames but reference file has {len(real_df)} frames."
        )
        print(
            "Files should have either the same number of frames or reference file should have exactly 1 frame."
        )
        return

    dlt_params = []

    # Process each frame from the pixel file
    for i in range(len(pixel_df)):
        frame = pixel_df.iloc[i]["frame"]
        print(f"Processing frame {frame}...")

        # Get pixel coordinates for this frame
        pixel_row = pixel_df.iloc[i]

        # Get reference coordinates - either from the same row or from the single reference row
        if single_ref_mode:
            real_row = ref_row
        else:
            real_row = real_df.iloc[i]

        # Filter coordinates to keep only complete pairs
        L_coords = []
        F_coords = []

        # Process coordinates in pairs (x,y)
        for j in range(0, len(coord_cols), 2):
            if j + 1 < len(coord_cols):  # Ensure we have a complete pair
                point_name = coord_cols[j].split("_")[
                    0
                ]  # Extract point name (e.g., p1 from p1_x)
                px_x = pixel_row[coord_cols[j]]
                px_y = pixel_row[coord_cols[j + 1]]
                real_x = real_row[coord_cols[j]]
                real_y = real_row[coord_cols[j + 1]]

                # Only use pairs where both pixel and real coordinates are valid
                if (
                    not pd.isna(px_x)
                    and not pd.isna(px_y)
                    and not pd.isna(real_x)
                    and not pd.isna(real_y)
                ):
                    L_coords.append([px_x, px_y])
                    F_coords.append([real_x, real_y])
                    print(f"  Using point {point_name} for frame {frame}")
                else:
                    if single_ref_mode:
                        note = "(missing in pixel or reference data)"
                    else:
                        note = "(missing in frame data)"
                    print(f"  Skipping point {point_name} for frame {frame} {note}")

        # Convert to numpy arrays
        L = np.array(L_coords)
        F = np.array(F_coords)

        # Calculate DLT parameters if we have enough points
        if len(L) >= 4 and len(F) >= 4:
            print(f"  Frame {frame}: Using {len(L)} valid coordinate pairs")
            try:
                params = dlt2d(F, L)
                dlt_params.append((frame, params))
            except Exception as e:
                print(f"  Error calculating DLT for frame {frame}: {e}")
                dlt_params.append((frame, [np.nan] * 8))
        else:
            print(
                f"  Frame {frame}: Not enough valid coordinate pairs ({len(L)}). Need at least 4."
            )
            dlt_params.append((frame, [np.nan] * 8))

    return dlt_params


def save_dlt_parameters(output_file, dlt_params):
    """
    Save the DLT parameters to a CSV file.

    Args:
    output_file (str): Path to the output file.
    dlt_params (list): List of DLT parameters.
    """
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["frame"] + [f"dlt_param_{j}" for j in range(1, 9)])
        for frame, params in dlt_params:
            # Converter o frame para inteiro antes de salvar
            frame = int(frame)
            csvwriter.writerow([frame] + list(params))

    messagebox.showinfo("Success", f"DLT parameters saved to {output_file}")
    print(f"DLT parameters saved to {output_file}")


def run_dlt2d():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting DLT2D calculation...")

    root = Tk()
    root.withdraw()

    pixel_file = filedialog.askopenfilename(
        title="Select the PIXEL coordinate file to be used for calibration.",
        filetypes=[("CSV files", "*.csv")],
    )
    if not pixel_file:
        print("Pixel file selection cancelled.")
        return

    create_ref = messagebox.askyesno(
        "Create REF2D File",
        "Do you want to create a REF2D file based on the pixel file?",
    )
    if create_ref:
        real_file = create_ref2d_template(pixel_file)
        messagebox.showinfo("Success", f"Template REF2D file created: {real_file}")
        print(f"Template REF2D file created: {real_file}")
        print(
            "Please edit the REF2D file with real coordinates and run the DLT process again."
        )
        return
    else:
        real_file = filedialog.askopenfilename(
            title="Select Real 2D Coordinates File",
            filetypes=[("REF2D files", "*.ref2d")],
        )
        if not real_file:
            print("Real file selection cancelled.")
            return

    dlt_params = process_files(pixel_file, real_file)
    output_file = os.path.splitext(pixel_file)[0] + ".dlt2d"
    save_dlt_parameters(output_file, dlt_params)


if __name__ == "__main__":
    run_dlt2d()
