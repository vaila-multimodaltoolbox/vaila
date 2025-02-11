"""
Script: dlt3d.py
Author: vailá
Version: 0.02
Last Updated: January 30, 2025

Description:
    This script calculates the Direct Linear Transformation (DLT) parameters for 3D coordinate
    transformations. It uses pixel coordinates from video calibration data and corresponding 
    real-world 3D coordinates to compute the DLT parameters (a row vector with 11 numbers).

    The procedure is as follows:
      - The user selects a pixel coordinate CSV file containing calibration data.
      - Optionally, the user can create a REF3D template file from the pixel file. If not, the user 
        must select an existing REF3D file (which contains the real-world 3D coordinates).
      - The calibration points from the files are used to compute the DLT parameters via a linear 
        solution obtained by inverting the normal equations.
      
Usage:
    1. Run the script.
    2. A graphical interface will prompt you to select:
         - A pixel coordinate CSV file for calibration.
         - Optionally, create a REF3D template based on the pixel file.
           If you choose not to create one, then select an existing REF3D file.
    3. The script calculates the DLT parameters and saves them as a "dlt3d" file.
    
Output:
    A CSV file with a header row: "frame", "dlt_param_1", ..., "dlt_param_11"
    and one data row containing the computed parameters.
    
License:
    This program is free software: you can redistribute it and/or modify it under the terms of 
    the GNU General Public License.
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
    Reads coordinates from a CSV file using the specified columns.
    """
    df = pd.read_csv(file_path, usecols=usecols)
    return df.to_numpy()


def dlt_calib(cp3d, cp2d):
    """
    Calculates the DLT (3D) calibration parameters (11 parameters).

    Args:
        cp3d (array-like): Calibration 3D points. Shape (m, 3). If extra columns are present (e.g. frame),
                           the first column is ignored.
        cp2d (array-like): Corresponding pixel coordinates. Shape (m, 2).

    Returns:
        np.array: 1D array with 11 DLT parameters.
    """
    cp3d = np.asarray(cp3d)
    # If there is an extra column (e.g., frame numbers), ignore the first column.
    if cp3d.shape[1] > 3:
        cp3d = cp3d[:, 1:]

    cp2d = np.asarray(cp2d)
    m = cp3d.shape[0]
    M = np.zeros((2 * m, 11))
    N = np.zeros((2 * m, 1))

    for i in range(m):
        X, Y, Z = cp3d[i, :]
        x, y = cp2d[i, :]
        M[2 * i, :] = [X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z]
        M[2 * i + 1, :] = [0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z]
        N[2 * i, 0] = x
        N[2 * i + 1, 0] = y

    DLT = inv(M.T.dot(M)).dot(M.T).dot(N)
    return DLT.flatten()


def create_ref3d_template(pixel_file):
    """
    Creates a REF3D template file from the pixel coordinate file by clearing all coordinate data
    (except, for example, the frame number).

    Returns:
        str: Path to the generated REF3D file.
    """
    df = pd.read_csv(pixel_file)
    template = df.copy()
    # Zero out all columns except the first one (usually the frame column)
    template.iloc[:, 1:] = np.nan
    template_file = os.path.splitext(pixel_file)[0] + ".ref3d"
    template.to_csv(template_file, index=False)
    return template_file


def process_files(pixel_file, real_file):
    """
    Processes the pixel and real-world coordinate files to compute DLT parameters.

    Args:
        pixel_file (str): Path to the pixel coordinate CSV file.
        real_file (str): Path to the REF3D file containing real-world 3D coordinates.

    Returns:
        np.array: The computed DLT parameters (row vector with 11 parameters).
    """
    # Read calibration points from the pixel file (ignore the 'frame' column)
    cp2d = read_coordinates(pixel_file, usecols=lambda c: c != "frame")
    # Read real-world coordinates from the REF3D file (ignore the 'frame' column)
    cp3d = read_coordinates(real_file, usecols=lambda c: c != "frame")

    if cp2d.shape[0] != cp3d.shape[0]:
        print(
            "The number of calibration points in the pixel file and the REF3D file do not match!"
        )
        return None

    return dlt_calib(cp3d, cp2d)


def save_dlt_parameters(output_file, dlt_params):
    """
    Saves the DLT parameters to a CSV file.
    """
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ["frame"] + [f"dlt_param_{j}" for j in range(1, 12)]
        csvwriter.writerow(header)
        # Using frame index 0 as a placeholder
        csvwriter.writerow([0] + list(dlt_params))
    messagebox.showinfo("Success", f"DLT parameters saved to {output_file}")
    print(f"DLT parameters saved to {output_file}")


def main():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting DLT3D calibration...")

    root = Tk()
    root.withdraw()

    pixel_file = filedialog.askopenfilename(
        title="Select the PIXEL coordinate file for calibration.",
        filetypes=[("CSV files", "*.csv")],
    )
    if not pixel_file:
        print("Pixel file selection cancelled.")
        return

    create_ref = messagebox.askyesno(
        "Create REF3D File",
        "Do you want to create a REF3D template based on the pixel file?",
    )
    if create_ref:
        real_file = create_ref3d_template(pixel_file)
        messagebox.showinfo(
            "Template Created",
            f"Template REF3D file created:\n{real_file}\nPlease edit it with the real-world coordinates and run the process again.",
        )
        print(f"Template REF3D file created: {real_file}")
        return
    else:
        real_file = filedialog.askopenfilename(
            title="Select the REF3D file with real-world coordinates.",
            filetypes=[("REF3D files", "*.ref3d")],
        )
        if not real_file:
            print("Real-world coordinates file selection cancelled.")
            return

    dlt_params = process_files(pixel_file, real_file)
    if dlt_params is None:
        messagebox.showerror("Error", "Calibration failed due to mismatched data.")
        return

    output_file = os.path.splitext(pixel_file)[0] + ".dlt3d"
    save_dlt_parameters(output_file, dlt_params)
    root.destroy()


if __name__ == "__main__":
    main()
