"""
rec3d_one_dlt3d.py
Author: Paulo Santiago
Version: 0.0.1
Last Updated: January 30, 2025

Description:
    This script performs 3D reconstruction using a single set of DLT3D parameters.
    It reads a DLT3D parameters file (with 11 parameters) and a CSV file containing
    pixel coordinates (with columns "frame", "x", "y"). The projection model is:

        x = (a1*X + a2*Y + a3*Z + a4) / (a9*X + a10*Y + a11*Z + 1)
        y = (a5*X + a6*Y + a7*Z + a8) / (a9*X + a10*Y + a11*Z + 1)
    
    Multiplying and rearranging, we get:
    
        (a1 - x*a9)*X + (a2 - x*a10)*Y + (a3 - x*a11)*Z = x - a4
        (a5 - y*a9)*X + (a6 - y*a10)*Y + (a7 - y*a11)*Z = y - a8
    
    Since the system is under-determined (2 equations, 3 unknowns), we obtain the 
    least-squares (minimal norm) solution using np.linalg.lstsq.
    
Usage:
    - Execute the script.
    - A GUI will prompt you to select the DLT3D parameters file (.dlt3d)
      and the pixel coordinate CSV file (which must have columns 'frame', 'x', 'y').
    - The script will generate a CSV file with columns: frame, X, Y, Z.
"""

import numpy as np
import pandas as pd
import os
from numpy.linalg import lstsq
from tkinter import filedialog, Tk, messagebox
from datetime import datetime


def rec3d(A, pixel):
    """
    Reconstructs a 3D point in a least-squares sense given DLT3D parameters and the observed pixel coordinates.

    Args:
        A (np.array): 1D array of 11 DLT3D parameters: [a1, a2, ..., a11].
        pixel (tuple): Observed pixel coordinate (x, y).

    Returns:
        np.array: Reconstructed 3D point (X, Y, Z) as computed by the least-squares solution.
    """
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = A
    x, y = pixel

    # Form the linear system derived from:
    # (a1 - x*a9)*X + (a2 - x*a10)*Y + (a3 - x*a11)*Z = x - a4
    # (a5 - y*a9)*X + (a6 - y*a10)*Y + (a7 - y*a11)*Z = y - a8
    M = np.array(
        [
            [a1 - x * a9, a2 - x * a10, a3 - x * a11],
            [a5 - y * a9, a6 - y * a10, a7 - y * a11],
        ]
    )
    b = np.array([x - a4, y - a8])
    sol, residuals, rank, s = lstsq(M, b, rcond=None)
    return sol  # Returns [X, Y, Z]


def process_reconstruction(dlt_file, pixel_file):
    """
    Process a DLT3D file and a pixel coordinate file to compute 3D reconstructed points.

    Args:
        dlt_file (str): Path to the .dlt3d file containing the DLT3D parameters.
        pixel_file (str): Path to the CSV file with pixel coordinates (must include columns "frame", "x", "y").

    Returns:
        pd.DataFrame: DataFrame with columns ['frame', 'X', 'Y', 'Z'].
    """
    # Read DLT3D parameters. Assume the file has a header: frame, dlt_param_1, ..., dlt_param_11.
    dlt_df = pd.read_csv(dlt_file)
    if dlt_df.empty:
        raise ValueError("DLT3D parameters file is empty.")
    # Use the first set (first row) of parameters.
    A = dlt_df.iloc[0, 1:].to_numpy().astype(float)

    # Read pixel coordinates CSV file (must include columns "frame", "x", and "y")
    pixel_df = pd.read_csv(pixel_file)
    required_cols = {"frame", "x", "y"}
    if not required_cols.issubset(pixel_df.columns):
        raise ValueError(
            "Pixel coordinate file must contain columns 'frame', 'x', and 'y'."
        )

    reconstructed_points = []
    for i, row in pixel_df.iterrows():
        frame = row["frame"]
        try:
            x = float(row["x"])
            y = float(row["y"])
        except Exception:
            x, y = np.nan, np.nan
        if np.isnan(x) or np.isnan(y):
            rec_point = (np.nan, np.nan, np.nan)
        else:
            rec_point = rec3d(A, (x, y))
        reconstructed_points.append((frame, rec_point[0], rec_point[1], rec_point[2]))

    rec_df = pd.DataFrame(reconstructed_points, columns=["frame", "X", "Y", "Z"])
    return rec_df


def main():
    root = Tk()
    root.withdraw()

    messagebox.showinfo("Instructions", "Select the DLT3D parameters file (.dlt3d)")
    dlt_file = filedialog.askopenfilename(
        title="Select DLT3D parameters file",
        filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")],
    )
    if not dlt_file:
        messagebox.showerror("Error", "No DLT3D file selected!")
        return

    messagebox.showinfo(
        "Instructions",
        "Select the pixel coordinate CSV file.\nThe file must contain columns: frame, x, y",
    )
    pixel_file = filedialog.askopenfilename(
        title="Select pixel coordinate CSV file", filetypes=[("CSV files", "*.csv")]
    )
    if not pixel_file:
        messagebox.showerror("Error", "No pixel coordinate file selected!")
        return

    try:
        rec_df = process_reconstruction(dlt_file, pixel_file)
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(os.path.dirname(pixel_file), f"rec3d_{timestamp}.csv")
    rec_df.to_csv(output_file, index=False, float_format="%.6f")
    messagebox.showinfo(
        "Success", f"3D Reconstruction completed.\nOutput saved to:\n{output_file}"
    )
    root.destroy()


if __name__ == "__main__":
    main()
