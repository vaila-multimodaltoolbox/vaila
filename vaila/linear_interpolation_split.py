"""
===============================================================================
linear_interpolation_split.py
===============================================================================
Author: Paulo R. P. Santiago
Created: 7 October 2024
Updated: 20 February 2026
Version: 0.0.12
Python Version: 3.12.12

Description:
------------
This script provides functionality to either fill missing data in CSV files using
linear interpolation or split data into a separate CSV file. It is intended for
use in biomechanical data analysis, where gaps in time-series data can be filled
and datasets can be split for further analysis.

Key Features:
-------------
1. **Linear Interpolation**:
   - Fills gaps in numerical data using linear interpolation.
2. **Data Splitting**:
   - Splits CSV files into two halves for easier data management and analysis.

Usage:
------
- Run this script to launch a graphical user interface (GUI) that provides options
  to perform linear interpolation on CSV files or to split them into two parts.
- The filled or split files are saved in new directories to avoid overwriting the
  original files.

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================
"""

import os
import tkinter as tk
from tkinter import Button, Label, Toplevel, filedialog, messagebox

import numpy as np
import pandas as pd


def run_fill_split_dialog():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting Linear Interpolation or Split Data...")

    root = tk.Tk()
    root.withdraw()  # Hide the root window

    dialog = Toplevel()
    dialog.title("Linear Interpolation or Split Data")
    dialog.geometry("300x200")

    label = Label(dialog, text="Choose an action to perform:")
    label.pack(pady=10)

    # Button for running Linear Interpolation
    fill_btn = Button(
        dialog,
        text="Run Linear Interpolation",
        command=lambda: run_linear_interpolation(),
    )
    fill_btn.pack(pady=5)

    # Button for splitting data
    split_btn = Button(dialog, text="Split Data", command=lambda: split_data())
    split_btn.pack(pady=5)


def run_linear_interpolation():
    # Prompt user to select the source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        return

    # Create a new directory to save the filled data files
    dest_dir = os.path.join(
        source_dir, f"vaila_fill_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(dest_dir, exist_ok=True)

    # Process each CSV file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(source_dir, filename)
            df = pd.read_csv(file_path)

            # Identify gaps in the frame_index column
            if "frame_index" in df.columns:
                df["frame_index"] = df["frame_index"].astype(int)
                full_index = pd.RangeIndex(
                    start=df["frame_index"].min(),
                    stop=df["frame_index"].max() + 1,
                    step=1,
                )
                df = df.set_index("frame_index").reindex(full_index).reset_index()

            # Assuming the columns that need interpolation are numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                # Apply linear interpolation to fill the gaps
                df[col] = df[col].interpolate(method="linear", limit_direction="both")

                # Round to match the original number of decimal places
                original_decimals = (
                    df[col]
                    .dropna()
                    .apply(lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0)
                    .max()
                )
                df[col] = df[col].round(original_decimals)

            # Save the filled data to the destination directory with '_fill' suffix
            dest_file_path = os.path.join(dest_dir, f"{filename.split('.csv')[0]}_fill.csv")
            df.to_csv(dest_file_path, index=False)
            messagebox.showinfo("Linear Interpolation", f"Filled data saved to {dest_file_path}")


def split_data():
    # Prompt user to select the source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        return

    # Create a new directory to save the split data file
    dest_dir = os.path.join(
        source_dir, f"vaila_split_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(dest_dir, exist_ok=True)

    # Process each CSV file in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(source_dir, filename)
            df = pd.read_csv(file_path)

            # Split the data in half and keep the second part
            midpoint = len(df) // 2
            df_part2 = df.iloc[midpoint:].reset_index(drop=True)
            df_part2.index += 1
            df_part2.reset_index(inplace=True)
            df_part2.rename(columns={"index": "frame_index"}, inplace=True)

            # Save the split data to the destination directory with '_split' suffix
            part2_file_path = os.path.join(dest_dir, f"{filename.split('.csv')[0]}_part2_split.csv")
            df_part2.to_csv(part2_file_path, index=False)
            messagebox.showinfo("Split Data", f"Split data saved to {part2_file_path}")


if __name__ == "__main__":
    run_fill_split_dialog()
