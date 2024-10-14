"""
===============================================================================
interpolation_split.py
===============================================================================
Author: Paulo R. P. Santiago
Date: 14 October 2024
Version: 1.3.1
Python Version: 3.11.9

Description:
------------
This script provides functionality to fill missing data in CSV files using 
linear interpolation, Kalman filter, Savitzky-Golay filter, nearest value fill, 
or to split data into a separate CSV file. It is intended for use in biomechanical 
data analysis, where gaps in time-series data can be filled and datasets can be 
split for further analysis.

Key Features:
-------------
1. **Gap Filling with Interpolation**: 
   - Fills gaps in numerical data using linear interpolation, Kalman filter, 
     Savitzky-Golay filter, nearest value fill, or leaves NaNs as is.
   - Only the missing data (NaNs) are filled; existing data remains unchanged.
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


def run_fill_split_dialog():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting in interpolation_split.py script...")

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

    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(source_dir, filename)
            df = pd.read_csv(file_path)

            original_first_col_name = df.columns[0]  # Preserve the original name of the first column

            # Ensure the first column is treated as integers (no interpolation)
            df[original_first_col_name] = df[original_first_col_name].astype(int)

            # Generate the full sequence of frame_index
            full_index = pd.Series(range(df[original_first_col_name].min(), df[original_first_col_name].max() + 1))
            
            # Detect missing indices
            missing_indices = full_index[~full_index.isin(df[original_first_col_name])]
            
            # Create rows with NaN for missing indices
            nan_rows = pd.DataFrame({original_first_col_name: missing_indices})
            for col in df.columns[1:]:
                nan_rows[col] = np.nan

            # Append missing rows and sort by frame_index
            df = pd.concat([df, nan_rows]).sort_values(by=original_first_col_name).reset_index(drop=True)

            numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(original_first_col_name)

            # Capture the number of decimal places for each column
            decimals_info = {
                col: df[col].dropna().apply(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0).max()
                for col in numeric_cols
            }

            for col in numeric_cols:
                original_data = df[col].copy()  # Preserve original data
                if method == "linear":
                    df[col] = df[col].interpolate(method="linear", limit_direction="both")
                elif method == "nearest":
                    df[col] = df[col].interpolate(method="nearest", limit_direction="both")
                elif method == "kalman":
                    # Apply Kalman filter only to NaN positions
                    observations = df[col].values
                    nan_mask = np.isnan(observations)
                    if nan_mask.any():
                        # Initialize Kalman Filter
                        kf = KalmanFilter()
                        # Use existing data to fit the filter
                        smoothed_state_means, _ = kf.em(observations, n_iter=5).smooth(observations)
                        # Update only NaN positions
                        df.loc[nan_mask, col] = smoothed_state_means[nan_mask, 0]
                elif method == "savgol":
                    # Apply Savitzky-Golay filter only to NaN positions
                    observations = df[col].values
                    nan_mask = np.isnan(observations)
                    if nan_mask.any():
                        # Fill NaNs temporarily for filtering
                        observations_filled = pd.Series(observations).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
                        window_length = (
                            min(7, len(observations_filled))
                            if len(observations_filled) % 2 != 0
                            else min(7, len(observations_filled) - 1)
                        )
                        filtered_values = savgol_filter(observations_filled, window_length, polyorder=2)
                        # Update only NaN positions
                        df.loc[nan_mask, col] = filtered_values[nan_mask]
                elif method == "nan":
                    continue  # Leave NaN values as they are

                # Restore original data where not NaN
                df[col] = df[col].where(df[col].isna(), original_data)

                # Apply the number of decimal places when saving the data
                decimals = decimals_info.get(col, 0)
                df[col] = df[col].apply(lambda x: round(x, decimals) if pd.notna(x) else x)

            # Save data to CSV, preserving the number formatting
            dest_file_path = os.path.join(dest_dir, f"{filename.split('.csv')[0]}_{method}_fill.csv")
            df.to_csv(dest_file_path, index=False)

            # Create a report of the gaps in the frame index and the ranges filled
            report_path = os.path.join(
                dest_dir, f"{filename.split('.csv')[0]}_{method}_fill.info"
            )
            with open(report_path, "w") as report_file:
                report_file.write(f"Gaps detected and filled for file: {filename}\n")
                report_file.write(f"Method used: {method}\n")
                
                # Report gaps in the first column where np.diff != 1
                report_file.write(f"Gaps in {original_first_col_name}:\n")
                for idx in missing_indices:
                    report_file.write(f"Gap at index {idx}, missing frame_index: {int(idx)}\n")

            # Add file to the list of processed files
            processed_files.append(filename)

    # Display a single confirmation message after all files have been processed
    if processed_files:
        messagebox.showinfo(
            "Interpolation", f"All files processed and saved in {dest_dir}."
        )


def split_data():
    print("Splitting data...")
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

            # Split the data into two parts
            df_part2 = df.iloc[midpoint:].copy()

            # Reset the 'frame' and 'frame_index' columns to start from 0 for the second part
            df_part2['frame'] = range(len(df_part2))
            df_part2['frame_index'] = range(len(df_part2))

            # Save the second part with correct 'frame' and 'frame_index' starting from 0
            part2_file_path = os.path.join(
                dest_dir, f"{filename.split('.csv')[0]}_part2_split.csv"
            )
            df_part2.to_csv(part2_file_path, index=False)
            
            messagebox.showinfo("Split Data", f"Split data saved to {part2_file_path}")


if __name__ == "__main__":
    run_fill_split_dialog()
