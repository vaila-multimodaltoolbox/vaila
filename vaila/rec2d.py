"""
================================================================================
Script: rec2d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.3
Created: August 9, 2024
Last Updated: March 19, 2026

Description:
    Optimized batch processing of 2D coordinates reconstruction using corresponding
    DLT2D parameters for each frame.
    Processes multiple CSV files containing pixel coordinates and reconstructs them
    to 2D real-world coordinates.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    Uses DLT2D parameters that can vary per frame (8 parameters per frame).

    Optimizations:
    - Pre-allocated NumPy arrays to eliminate dynamic memory allocation
    - Progress tracking for large datasets
    - Reduced debug output for cleaner processing feedback
    - User-defined output directory and data frequency upfront
    - Vectorized operations for better performance
    - Output saves both `.2d` and `.csv`
    - `Frame` saved as integer (no float formatting)
    - Coordinate columns saved as float with %.6f precision
    - CLI / headless mode support via argparse
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
from rich import print


def rec2d(A, cc2d):
    """Reconstruct 2D real-world coordinates from pixel coordinates using DLT2D parameters.

    Args:
        A: DLT2D parameter array (8 elements).
        cc2d: Pixel coordinates array of shape (N, 2).

    Returns:
        np.ndarray of shape (N, 2) with reconstructed 2D coordinates.
    """
    nlin = cc2d.shape[0]
    H = np.zeros((nlin, 2))
    for k in range(nlin):
        x = cc2d[k, 0]
        y = cc2d[k, 1]
        coeff = np.array([[A[0] - x * A[6], A[1] - x * A[7]], [A[3] - y * A[6], A[4] - y * A[7]]])
        rhs = np.array([[x - A[2]], [y - A[5]]])
        G1 = np.linalg.solve(coeff, rhs)
        H[k, :] = G1.flatten()
    return H


def process_files_in_directory(dlt_params_df, input_directory, output_directory, data_rate):
    """Process multiple CSV files in a directory using DLT2D parameters.

    Args:
        dlt_params_df: DataFrame with DLT2D parameters (frame + 8 params).
        input_directory: Directory containing CSV files to process.
        output_directory: Directory to save output files.
        data_rate: Data frequency in Hz.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_directory, f"vaila_rec2d_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    dlt_params = dlt_params_df.to_numpy()
    frames = dlt_params[:, 0]
    dlt_params = dlt_params[:, 1:]

    csv_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".csv")])

    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in the selected directory!")
        return

    print(f"Found {len(csv_files)} CSV files to process")

    total_files = len(csv_files)
    files_processed = 0

    for i_file, csv_file in enumerate(csv_files):
        files_processed = i_file + 1
        progress = (files_processed / total_files) * 100
        print(f"Processing file {files_processed}/{total_files} ({progress:.1f}%): {csv_file}")

        try:
            pixel_file = os.path.join(input_directory, csv_file)
            pixel_coords_df = pd.read_csv(pixel_file)
        except Exception as e:
            print(f"[red]Error reading {csv_file}: {e}. Skipping file.[/red]")
            continue

        num_coords = (pixel_coords_df.shape[1] - 1) // 2
        total_frames = len(pixel_coords_df)

        total_cols = 1 + (num_coords * 2)
        rec_coords_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)

        rec_coords_array[:, 0] = pixel_coords_df["frame"].values

        for i, row in pixel_coords_df.iterrows():
            frame_num = int(row["frame"])
            if frame_num in frames:
                A_index = np.where(frames == frame_num)[0][0]
                A = dlt_params[A_index]
                if not np.isnan(A).any():
                    pixel_coords = row.iloc[1:].to_numpy().reshape(-1, 2)
                    if np.isnan(pixel_coords).any():
                        continue
                    rec2d_coords = rec2d(A, pixel_coords)
                    rec_coords_array[i, 1:] = rec2d_coords.flatten()

        rec_coords_df = pd.DataFrame(rec_coords_array, columns=pixel_coords_df.columns)

        # Normalize output to vaila's expected convention: `Frame` as integer.
        if "frame" in rec_coords_df.columns:
            rec_coords_df["frame"] = rec_coords_df["frame"].astype(int)
            rec_coords_df = rec_coords_df.rename(columns={"frame": "Frame"})

        def _write_output(df: pd.DataFrame, out_path: str) -> None:
            coord_cols_local = [c for c in df.columns if c != "Frame"]
            with open(out_path, "w") as fh:
                fh.write(",".join(df.columns) + "\n")
                for _, row in df.iterrows():
                    vals: list[str] = []
                    for col in df.columns:
                        v = row[col]
                        if col == "Frame":
                            vals.append(str(int(v)))
                        elif pd.isna(v):
                            vals.append("")
                        else:
                            # Coordinates as float with fixed precision.
                            vals.append(f"{float(v):.6f}" if col in coord_cols_local else str(v))
                    fh.write(",".join(vals) + "\n")

        base = os.path.splitext(csv_file)[0]
        output_file_2d = os.path.join(output_dir, f"{base}_{timestamp}.2d")
        output_file_csv = os.path.join(output_dir, f"{base}_{timestamp}.csv")
        _write_output(rec_coords_df, output_file_2d)
        _write_output(rec_coords_df, output_file_csv)

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


def run_rec2d(dlt_file=None, input_directory=None, output_directory=None, data_rate=None):
    """Main entry point for 2D reconstruction.

    When called without arguments, opens GUI dialogs. When all arguments are
    provided, runs in headless mode (CLI).
    """
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec2d.py...")
    print("-" * 80)

    if dlt_file is None:
        root = Tk()
        root.withdraw()

        print("Step 1: Selecting DLT parameters file...")
        dlt_file = filedialog.askopenfilename(
            title="Select DLT Parameters File",
            filetypes=[("DLT2D files", "*.dlt2d"), ("CSV files", "*.csv")],
        )
        if not dlt_file:
            print("DLT file selection cancelled.")
            return

        print("Step 2: Selecting input directory...")
        input_directory = filedialog.askdirectory(title="Select Directory Containing CSV Files")
        if not input_directory:
            print("Input directory selection cancelled.")
            return

        print("Step 3: Selecting output directory...")
        output_directory = filedialog.askdirectory(title="Select Output Directory for Results")
        if not output_directory:
            print("Output directory selection cancelled.")
            return

        print("Step 4: Setting data frequency...")
        data_rate = simpledialog.askinteger(
            "Data Frequency", "Enter the data frequency (Hz):", minvalue=1, initialvalue=100
        )
        if data_rate is None:
            messagebox.showerror("Error", "Data frequency is required. Operation cancelled.")
            return

        root.destroy()
    else:
        if input_directory is None or output_directory is None or data_rate is None:
            print(
                "Error: dlt-file, input-dir, output-dir, and rate are required for headless mode."
            )
            return

    print("Loading DLT parameters...")
    dlt_params_df = pd.read_csv(dlt_file)
    if dlt_params_df.empty:
        print(f"Error: DLT2D file {os.path.basename(dlt_file)} is empty!")
        return

    print("Configuration complete:")
    print(f"  - DLT file: {os.path.basename(dlt_file)}")
    print(f"  - Input directory: {input_directory}")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {data_rate} Hz")
    print(f"  - DLT parameters for {len(dlt_params_df)} frames")
    print("-" * 80)

    process_files_in_directory(dlt_params_df, input_directory, output_directory, data_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reconstruct 2D Coordinates using DLT2D parameters"
    )
    parser.add_argument("--dlt-file", help="Path to DLT2D parameter file")
    parser.add_argument("--input-dir", help="Directory containing CSV files to process")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--rate", type=int, help="Data frequency in Hz")
    args = parser.parse_args()

    run_rec2d(
        dlt_file=args.dlt_file,
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        data_rate=args.rate,
    )
