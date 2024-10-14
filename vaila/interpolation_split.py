"""
===============================================================================
interpolation_split.py
===============================================================================
Author: Paulo R. P. Santiago
Date: 14 October 2024
Version: 1.3.0
Python Version: 3.11.9

Description:
------------
This script provides functionality to either fill missing data in CSV files using 
linear interpolation, Kalman filter, Savitzky-Golay filter, nearest value fill, or NaN fill, 
or split data into a separate CSV file. It is intended for use in biomechanical data analysis, 
where gaps in time-series data can be filled and datasets can be split for further analysis.

Key Features:
-------------
1. **Gap Filling with Interpolation**: 
   - Fills gaps in numerical data using linear interpolation, Kalman filter, Savitzky-Golay filter, nearest value fill, or NaN fill.
2. **Data Splitting**: 
   - Splits CSV files into two halves for easier data management and analysis.

Usage:
------
- Run this script to launch a graphical user interface (GUI) that provides options 
  to perform interpolation on CSV files or to split them into two parts.
- The filled or split files are saved in new directories to avoid overwriting the 
  original files.

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================
"""

import os
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
from tkinter import filedialog, messagebox, Toplevel, Button, Label
import tkinter as tk
from rich import print
import ipdb


def run_fill_split_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    dialog = Toplevel()
    dialog.title("Interpolation or Split Data")
    dialog.geometry("300x350")

    label = Label(dialog, text="Choose an action to perform:")
    label.pack(pady=10)

    fill_btn_linear = Button(
        dialog,
        text="Run Linear Interpolation",
        command=lambda: run_interpolation(method="linear"),
    )
    fill_btn_linear.pack(pady=5)

    fill_btn_kalman = Button(
        dialog,
        text="Run Kalman Filter",
        command=lambda: run_interpolation(method="kalman"),
    )
    fill_btn_kalman.pack(pady=5)

    fill_btn_savgol = Button(
        dialog,
        text="Run Savitzky-Golay Filter",
        command=lambda: run_interpolation(method="savgol"),
    )
    fill_btn_savgol.pack(pady=5)

    fill_btn_nearest = Button(
        dialog,
        text="Run Nearest Value Fill",
        command=lambda: run_interpolation(method="nearest"),
    )
    fill_btn_nearest.pack(pady=5)

    fill_btn_nan = Button(
        dialog,
        text="Leave NaNs as is",
        command=lambda: run_interpolation(method="nan"),
    )
    fill_btn_nan.pack(pady=5)

    split_btn = Button(dialog, text="Split Data", command=lambda: split_data())
    split_btn.pack(pady=5)

def run_interpolation(method):
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        return

    dest_dir = os.path.join(
        source_dir,
        f"interpolated_{method}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(dest_dir, exist_ok=True)

    processed_files = []  # List to track processed files

    ipdb.set_trace()
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(source_dir, filename)
            df = pd.read_csv(file_path)

            original_first_col_name = df.columns[0]  # Preserve the original name of the first column

            # Ensure the first column is treated as integers (no interpolation)
            df[original_first_col_name] = df[original_first_col_name].astype(int)

            # Store decimal precision for each column
            decimals_info = {
                col: df[col].dropna().apply(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0).max()
                for col in df.select_dtypes(include=[np.float64]).columns
            }

            # Detect gaps in the first column (using np.diff to find gaps larger than 1)
            diff = np.diff(df[original_first_col_name])
            gap_indices = np.where(diff > 1)[0]  # Identify gaps where diff > 1
            for idx in gap_indices:
                start_value = df.loc[idx, original_first_col_name]
                gap_size = int(diff[idx])
                df.loc[idx + 1:idx + gap_size, original_first_col_name] = np.arange(
                    start_value + 1, start_value + gap_size + 1
                )

            numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(original_first_col_name)

            missing_ranges = {}  # Dictionary to store missing ranges for each column

            # Capture missing ranges before interpolation
            for col in numeric_cols:
                missing_ranges[col] = df[df[col].isna()].index.to_list()

            for col in numeric_cols:
                if method == "linear":
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
                elif method == "kalman":
                    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
                    df[col] = kf.smooth(df[col].fillna(0).values)[0]
                elif method == "savgol":
                    window_length = (
                        min(7, len(df[col]))
                        if len(df[col]) % 2 != 0
                        else min(7, len(df[col]) - 1)
                    )
                    df[col] = savgol_filter(df[col].fillna(0), window_length, polyorder=2)
                elif method == "nearest":
                    df[col] = df[col].interpolate(method="nearest", limit_direction="both")
                elif method == "nan":
                    # Leave NaN values as they are, no interpolation
                    df[col] = df[col]  # This effectively skips interpolation for NaN method

                # Preserve original significant digits after interpolation
                decimals = decimals_info.get(col, 0)
                df[col] = df[col].apply(
                    lambda x: f"{x:.{decimals}f}" if pd.notna(x) else x
                ).astype(float)

            # Save data to CSV, preserving number formatting
            dest_file_path = os.path.join(dest_dir, f"{filename.split('.csv')[0]}_{method}_fill.csv")
            df.to_csv(dest_file_path, index=False)

            # Create a report of the ranges that were filled based on np.diff of the first column
            report_path = os.path.join(
                dest_dir, f"{filename.split('.csv')[0]}_{method}_fill.info"
            )
            with open(report_path, "w") as report_file:
                report_file.write(f"Gaps detected and filled for file: {filename}\n")
                report_file.write(f"Method used: {method}\n")
                
                # Report gaps in the first column where np.diff != 1
                report_file.write(f"Gaps in {original_first_col_name}:\n")
                for idx in gap_indices:
                    report_file.write(f"Gap at index {idx}, start value: {df.loc[idx, original_first_col_name]}, gap size: {int(diff[idx])}\n")

                # Report missing indices in other columns
                for col, missing_indices in missing_ranges.items():
                    report_file.write(f"Missing values in column {col}: {missing_indices}\n")

            # Add file to the list of processed files
            processed_files.append(filename)

    # Display a single confirmation message after all files have been processed
    if processed_files:
        messagebox.showinfo(
            "Interpolation", f"All files processed and saved in {dest_dir}."
        )

def split_data():
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        return

    dest_dir = os.path.join(
        source_dir, f"split_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(source_dir, filename)
            df = pd.read_csv(file_path)

            midpoint = len(df) // 2
            df_part2 = df.iloc[midpoint:].reset_index(drop=True)
            df_part2.index += 1
            df_part2.reset_index(inplace=True)
            df_part2.rename(columns={"index": "frame_index"}, inplace=True)

            part2_file_path = os.path.join(
                dest_dir, f"{filename.split('.csv')[0]}_part2_split.csv"
            )
            df_part2.to_csv(part2_file_path, index=False)
            messagebox.showinfo("Split Data", f"Split data saved to {part2_file_path}")


if __name__ == "__main__":
    run_fill_split_dialog()
