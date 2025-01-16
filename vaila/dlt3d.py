"""
Script: dlt3d.py
Author: vailá
Version: 0.01
Last Updated: January 16, 2025

Description:
    This script calculates the Direct Linear Transformation (DLT) parameters for 3D coordinate transformations.
    It uses pixel coordinates from video calibration data and corresponding real-world 3D coordinates to compute
    the DLT parameters for each frame in a dataset.

    The script also allows users to create a template REF3D file from a pixel file, which can then be edited
    manually to include real-world coordinates. The main functionality includes reading input files,
    performing DLT calculations, and saving the results to an output file.

Usage:
    1. Run the script to start the Direct Linear Transformation (DLT) process.
    2. A graphical interface will prompt you to select:
       - A pixel coordinate file (CSV format) for calibration.
       - (Optional) Create a REF3D template file or use an existing REF3D file for real-world coordinates.
    3. The script processes the input files and calculates the DLT parameters.
    4. The calculated parameters are saved as a `.dlt3d` CSV file in the same directory as the input pixel file.

Requirements:
    - Python 3.11.9
    - Numpy (`pip install numpy`)
    - Pandas (`pip install pandas`)
    - Rich (`pip install rich`)
    - Tkinter (usually included with Python installations)

Output:
    - REF3D Template File (`*.ref3d`):
      A template file for real-world coordinates created from the pixel file.
    - DLT Parameters File (`*.dlt3d`):
      A CSV file containing the DLT parameters for each processed frame.

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
    df = pd.read_csv(file_path, usecols=usecols)
    coordinates = df.to_numpy()  # Keep all rows, including NaNs
    return coordinates


def dlt3d(F, L):
    """
    Calculate DLT (3D) parameters.

    Args:
    F (np.array): Matrix containing real-world global coordinates (X, Y, Z).
    L (np.array): Matrix containing pixel coordinates of calibration points.

    Returns:
    np.array: DLT parameters.
    """
    F = np.matrix(F)
    L = np.matrix(L)
    Lt = L.transpose()
    C = Lt.flatten("F").transpose()
    m = np.size(F, 0)

    B = np.zeros((2 * m, 11))
    for i in range(m):
        B[2 * i, 0:4] = [F[i, 0], F[i, 1], F[i, 2], 1]
        B[2 * i, 8:11] = [-F[i, 0] * L[i, 0], -F[i, 1] * L[i, 0], -F[i, 2] * L[i, 0]]
        B[2 * i + 1, 4:8] = [F[i, 0], F[i, 1], F[i, 2], 1]
        B[2 * i + 1, 8:11] = [-F[i, 0] * L[i, 1], -F[i, 1] * L[i, 1], -F[i, 2] * L[i, 1]]

    A = inv(B.T @ B) @ B.T @ C
    return np.asarray(A).flatten()


def create_ref3d_template(pixel_file):
    df = pd.read_csv(pixel_file)
    template = df.copy()
    template.iloc[:, 1:] = np.nan  # Clear all coordinate data but keep frame numbers
    template_file = os.path.splitext(pixel_file)[0] + ".ref3d"
    template.to_csv(template_file, index=False)
    return template_file


def process_files(pixel_file, real_file):
    pixel_coords = read_coordinates(pixel_file, usecols=lambda x: x != "frame")
    real_coords = read_coordinates(real_file, usecols=lambda x: x != "frame")

    if pixel_coords.shape[1] // 2 != real_coords.shape[0]:
        print("The number of 2D views and 3D coordinate sets must match.")
        return

    dlt_params = []
    for i in range(len(pixel_coords)):
        if not np.isnan(pixel_coords[i]).any() and not np.isnan(real_coords[i]).any():
            L = pixel_coords[i].reshape(-1, 2)
            F = real_coords[i].reshape(-1, 3)
            if F.shape[0] >= 6:  # At least 6 points are needed for DLT3D
                print(f"Frame {i}: F = {F}, L = {L}")
                dlt_params.append((i, dlt3d(F, L)))
            else:
                dlt_params.append((i, [np.nan] * 11))
        else:
            dlt_params.append((i, [np.nan] * 11))

    return dlt_params


def save_dlt_parameters(output_file, dlt_params):
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["frame"] + [f"dlt_param_{j}" for j in range(1, 12)])
        for frame, params in dlt_params:
            csvwriter.writerow([frame] + list(params))

    messagebox.showinfo("Success", f"DLT parameters saved to {output_file}")
    print(f"DLT parameters saved to {output_file}")


def main():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting DLT3D calculation...")

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
        "Create REF3D File",
        "Do you want to create a REF3D file based on the pixel file?",
    )
    if create_ref:
        real_file = create_ref3d_template(pixel_file)
        messagebox.showinfo("Success", f"Template REF3D file created: {real_file}")
        print(f"Template REF3D file created: {real_file}")
        print(
            "Please edit the REF3D file with real coordinates and run the DLT process again."
        )
        return
    else:
        real_file = filedialog.askopenfilename(
            title="Select Real 3D Coordinates File",
            filetypes=[("REF3D files", "*.ref3d")],
        )
        if not real_file:
            print("Real file selection cancelled.")
            return

    dlt_params = process_files(pixel_file, real_file)
    output_file = os.path.splitext(pixel_file)[0] + ".dlt3d"
    save_dlt_parameters(output_file, dlt_params)


if __name__ == "__main__":
    main()
