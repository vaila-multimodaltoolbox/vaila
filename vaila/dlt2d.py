"""
Code Name: dlt2d.py
Version: v0.02
Date and Time: 2024-09-30
Creator: vailá
Email: vailamultimodaltoolbox@gmail.com
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
    """
    filtered_coords = [row for row in coords if not np.isnan(row).any()]
    return np.array(filtered_coords).reshape(-1, 2)


def process_files(pixel_file, real_file):
    """
    Process the coordinate files to calculate the DLT parameters.

    Args:
    pixel_file (str): Path to the pixel coordinate file.
    real_file (str): Path to the real-world coordinate file.

    Returns:
    list: List of DLT parameters for each frame.
    """
    pixel_coords = read_coordinates(pixel_file, usecols=lambda x: x != "frame")
    real_coords = read_coordinates(real_file, usecols=lambda x: x != "frame")

    if pixel_coords.shape[1] != real_coords.shape[1]:
        print("The number of coordinate pairs in the two files must match.")
        return

    dlt_params = []
    for i in range(len(pixel_coords)):
        if not np.isnan(pixel_coords[i]).any() and not np.isnan(real_coords[i]).any():
            L = filter_and_shape_coordinates(pixel_coords[i])
            F = filter_and_shape_coordinates(real_coords[i])
            if F.shape[0] == L.shape[0] and F.shape[0] >= 4:
                print(f"Frame {i}: F = {F}, L = {L}")
                dlt_params.append((i, dlt2d(F, L)))
            else:
                dlt_params.append((i, [np.nan] * 8))
        else:
            dlt_params.append((i, [np.nan] * 8))

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
            csvwriter.writerow([frame] + list(params))

    messagebox.showinfo("Success", f"DLT parameters saved to {output_file}")
    print(f"DLT parameters saved to {output_file}")


def main():
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
    main()
