"""
================================================================================
Script: rec3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.3.94
Created: August 03, 2025
Last Updated: 01 August 2026

Description:
    Batch 3D reconstruction using per-frame Direct Linear Transformation (DLT3D)
    parameters — i.e. a DLT "matrix" that can change frame by frame (moving or
    re-calibrated cameras), as opposed to rec3d_one_dlt3d.py which uses one
    fixed set of DLT3D parameters per camera for the whole clip.

    For each camera you provide:
      - One DLT3D parameter file with ONE ROW PER FRAME (frame, 11 coefficients).
      - One pixel-coordinate CSV, in the same directory as the other cameras'
        pixel files, with the same number of rows as the other cameras.

    All camera pixel files must be placed together in a single input directory
    (one CSV per camera) so they can be correlated frame-by-frame; the files are
    paired with --dlt-files by sorted filename order — same convention used to
    pair --dlt3d/--pixels in rec3d_one_dlt3d.py, just via a directory instead of
    an explicit file list.

    Column headers are NOT required to follow any particular naming: only the
    COLUMN ORDER matters — column 0 is the frame identifier and every pair of
    columns after that is one marker's (x, y). This makes the pixel files
    compatible with trackers that use different label conventions (vailá
    p1_x/p1_y, SAM3, YOLO, MediaPipe named joints, etc.) as long as the same
    markers appear in the same order in every camera's file.

    Output: one reconstructed 3D result (CSV + .3d) in a timestamped
    vaila_rec3d_<timestamp>/ subfolder — not one output per input file.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from rich import print


def rec3d_multicam(dlt_list, pixel_list):
    """
    Reconstructs a 3D point using multiple camera observations and their corresponding DLT3D parameters.

    Args:
        dlt_list (list of np.array): List of DLT3D parameter arrays (each of 11 elements) for each camera.
        pixel_list (list of tuple): List of observed pixel coordinates (x, y) for each camera.

    Returns:
        np.array: Reconstructed 3D point [X, Y, Z] using a least squares solution.
    """
    num_cameras = len(dlt_list)
    A_matrix = np.zeros((num_cameras * 2, 3))
    b_vector = np.zeros(num_cameras * 2)

    for i, (A_params, (x, y)) in enumerate(zip(dlt_list, pixel_list, strict=False)):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = A_params

        # Equations for camera i:
        # (a1 - x*a9)*X + (a2 - x*a10)*Y + (a3 - x*a11)*Z = x - a4
        # (a5 - y*a9)*X + (a6 - y*a10)*Y + (a7 - y*a11)*Z = y - a8
        row_idx = i * 2
        A_matrix[row_idx] = [a1 - x * a9, a2 - x * a10, a3 - x * a11]
        A_matrix[row_idx + 1] = [a5 - y * a9, a6 - y * a10, a7 - y * a11]
        b_vector[row_idx] = x - a4
        b_vector[row_idx + 1] = y - a8

    solution, residuals, rank, s = lstsq(A_matrix, b_vector, rcond=None)
    return solution  # [X, Y, Z]


def load_pixel_csv_positional(file_path):
    """
    Load a pixel-coordinate CSV using COLUMN ORDER instead of column names.

    Column 0 (whatever it is named) is treated as the frame identifier; every
    pair of columns after that is one marker's (x, y) pixel position. A single
    header row is still expected (and discarded) so the file remains a normal
    CSV, but its label text is never inspected.

    Args:
        file_path (str): Path to the pixel CSV file.

    Returns:
        tuple[np.ndarray, np.ndarray]: (frame_values shape (n_rows,),
        xy shape (n_rows, num_markers, 2)).

    Raises:
        ValueError: if the file has fewer than 3 columns, or the number of
        coordinate columns (all columns after the first) is not even.
    """
    df = pd.read_csv(file_path)
    n_cols = df.shape[1]
    if n_cols < 3:
        raise ValueError(f"expected at least 3 columns (frame + one marker x,y), found {n_cols}")
    n_coord_cols = n_cols - 1
    if n_coord_cols % 2 != 0:
        raise ValueError(
            f"expected an even number of coordinate columns after the frame column, "
            f"found {n_coord_cols} (columns must be frame, x1, y1, x2, y2, ...)"
        )

    values = df.to_numpy(dtype=np.float64)
    frame_values = values[:, 0]
    if len(frame_values) == 1:
        # Single-row files (e.g. a single calibration/reference frame) are
        # conventionally treated as frame 0.
        frame_values = np.array([0.0])

    num_markers = n_coord_cols // 2
    xy = values[:, 1:].reshape(-1, num_markers, 2)
    return frame_values, xy


def find_common_frames(frame_arrays):
    """Sorted intersection of frame numbers present in every array (as ints)."""
    frame_sets = [{int(f) for f in arr} for arr in frame_arrays]
    common = set.intersection(*frame_sets) if frame_sets else set()
    return np.array(sorted(common), dtype=np.int64)


def _write_rec3d_output(rec_coords_df, out_path):
    """Write frame as integer, coordinates as float with 6-decimal precision."""
    with open(out_path, "w") as fh:
        fh.write(",".join(rec_coords_df.columns) + "\n")
        for _, row in rec_coords_df.iterrows():
            vals = []
            for col in rec_coords_df.columns:
                v = row[col]
                if col == "frame":
                    vals.append(str(int(v)))
                elif pd.isna(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.6f}")
            fh.write(",".join(vals) + "\n")


def process_files_in_directory(dlt_params_dfs, input_directory, output_directory, data_rate):
    """
    Reconstruct 3D coordinates from N camera pixel CSV files (one per camera,
    matching the order of dlt_params_dfs) using per-frame-varying DLT3D
    parameters for each camera.

    Args:
        dlt_params_dfs (list of pd.DataFrame): Per-camera DLT3D parameter
            tables (frame + 11 coefficients per row), same order as the pixel
            CSV files once sorted by filename.
        input_directory (str): Directory containing exactly len(dlt_params_dfs)
            pixel CSV files, one per camera.
        output_directory (str): Directory to save the reconstructed output.
        data_rate (float): Data frequency in Hz (recorded in the console summary).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_cameras = len(dlt_params_dfs)

    # Per-camera DLT3D frames + parameters (positional: col 0 = frame, rest = 11 coeffs)
    dlt_frames_list = []
    dlt_values_list = []
    for df in dlt_params_dfs:
        arr = df.to_numpy(dtype=np.float64)
        dlt_frames_list.append(arr[:, 0])
        dlt_values_list.append(arr[:, 1:])

    csv_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".csv")])

    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in the selected directory!")
        return
    if len(csv_files) != num_cameras:
        messagebox.showerror(
            "Error",
            f"Expected {num_cameras} pixel CSV file(s) in the input directory "
            f"(one per camera, matching --dlt-files), found {len(csv_files)}.",
        )
        return

    print(
        f"Found {len(csv_files)} camera pixel file(s), matching {num_cameras} DLT3D camera set(s)"
    )

    # Load every camera's pixel file (column-order based; labels are ignored)
    pixel_frames_list = []
    pixel_xy_list = []
    for csv_file in csv_files:
        path = os.path.join(input_directory, csv_file)
        try:
            frames_arr, xy_arr = load_pixel_csv_positional(path)
        except Exception as e:
            print(f"[red]Error reading {csv_file}: {e}. Aborting.[/red]")
            messagebox.showerror("Error", f"Error reading {csv_file}: {e}")
            return
        pixel_frames_list.append(frames_arr)
        pixel_xy_list.append(xy_arr)

    num_markers = min(xy.shape[1] for xy in pixel_xy_list)
    if any(xy.shape[1] != num_markers for xy in pixel_xy_list):
        print(
            f"[yellow]Warning: camera pixel files have different marker counts; "
            f"using the smallest common count: {num_markers}[/yellow]"
        )

    common_frames = find_common_frames(pixel_frames_list)
    if common_frames.size == 0:
        messagebox.showerror("Error", "No common frames found among the camera pixel files!")
        return
    print(f"Processing {len(common_frames)} common frame(s) across {num_cameras} camera(s)...")

    # Only create the output folder once every validation above has passed.
    output_dir = os.path.join(output_directory, f"vaila_rec3d_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    total_frames = len(common_frames)
    total_cols = 1 + (num_markers * 3)
    reconstruction_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)
    reconstruction_array[:, 0] = common_frames

    pixel_frame_to_row = [{int(f): i for i, f in enumerate(frames)} for frames in pixel_frames_list]
    dlt_frame_to_row = [{int(f): i for i, f in enumerate(frames)} for frames in dlt_frames_list]

    progress_step = max(1, total_frames // 20)
    for frame_idx, frame in enumerate(common_frames):
        if frame_idx % progress_step == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

        frame_int = int(frame)

        # Per-camera DLT3D parameters for THIS frame (the "DLT matrix" lookup)
        dlt_params_for_frame = []
        frame_ok = True
        for cam_idx in range(num_cameras):
            dlt_row = dlt_frame_to_row[cam_idx].get(frame_int)
            if dlt_row is None:
                frame_ok = False
                break
            A = dlt_values_list[cam_idx][dlt_row]
            if np.isnan(A).any():
                frame_ok = False
                break
            dlt_params_for_frame.append(A)
        if not frame_ok:
            continue

        pixel_row_for_cam = [pixel_frame_to_row[c][frame_int] for c in range(num_cameras)]

        for marker in range(num_markers):
            pixel_obs_list = []
            valid_marker = True
            for cam_idx in range(num_cameras):
                x_obs, y_obs = pixel_xy_list[cam_idx][pixel_row_for_cam[cam_idx], marker]
                if np.isnan(x_obs) or np.isnan(y_obs):
                    valid_marker = False
                    break
                pixel_obs_list.append((float(x_obs), float(y_obs)))
            if not valid_marker:
                continue

            point3d = rec3d_multicam(dlt_params_for_frame, pixel_obs_list)
            col_start = 1 + marker * 3
            reconstruction_array[frame_idx, col_start : col_start + 3] = point3d

    header = ["frame"]
    for marker in range(1, num_markers + 1):
        header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])

    rec_coords_df = pd.DataFrame(reconstruction_array, columns=header)  # type: ignore
    rec_coords_df["frame"] = rec_coords_df["frame"].astype(int)

    output_file_3d = os.path.join(output_dir, f"rec3d_{timestamp}.3d")
    output_file_csv = os.path.join(output_dir, f"rec3d_{timestamp}.csv")
    _write_rec3d_output(rec_coords_df, output_file_3d)
    _write_rec3d_output(rec_coords_df, output_file_csv)

    print("\n=== Processing Complete ===")
    print(f"Cameras: {num_cameras}")
    print(f"Frames reconstructed: {total_frames}")
    print(f"Markers: {num_markers}")
    print(f"Data rate used: {data_rate} Hz")
    print(f"Output directory: {output_dir}")

    messagebox.showinfo(
        "Processing Complete",
        f"3D reconstruction completed successfully!\n\n"
        f"Cameras: {num_cameras}\n"
        f"Frames: {total_frames}\n"
        f"Markers: {num_markers}\n"
        f"Data rate: {data_rate} Hz\n"
        f"Output directory: {os.path.basename(output_dir)}",
    )
    print(f"Reconstructed 3D coordinates saved to {output_dir}")


def run_rec3d(dlt_files=None, input_directory=None, output_directory=None, data_rate=None):
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec3d.py...")
    print("-" * 80)

    if dlt_files is None:
        root = Tk()
        root.withdraw()

        # Step 1: Select DLT3D parameters files (multiple cameras)
        print("Step 1: Selecting DLT3D parameters files...")
        dlt_files = filedialog.askopenfilenames(
            title="Select DLT3D Parameters Files (one per camera, matching pixel file order)",
            filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")],
        )
        if not dlt_files:
            print("DLT file selection cancelled.")
            return

        # Step 2: Select input directory with CSV files
        print("Step 2: Selecting input directory...")
        input_directory = filedialog.askdirectory(
            title="Select Directory Containing Pixel CSV Files (one per camera)"
        )
        if not input_directory:
            print("Input directory selection cancelled.")
            return

        # Step 3: Select output directory
        print("Step 3: Selecting output directory...")
        output_directory = filedialog.askdirectory(title="Select Output Directory for Results")
        if not output_directory:
            print("Output directory selection cancelled.")
            return

        # Step 4: Ask for data frequency
        print("Step 4: Setting data frequency...")
        data_rate = simpledialog.askfloat(
            "Data Frequency",
            "Enter the data frequency (Hz), e.g. 119.88012001 for a real NTSC-derived rate:",
            minvalue=0.0001,
            initialvalue=100.0,
        )
        if data_rate is None:
            messagebox.showerror("Error", "Data frequency is required. Operation cancelled.")
            return
        root.destroy()
    else:
        # Headless mode
        if input_directory is None or output_directory is None or data_rate is None:
            print(
                "Error: dlt-files, input-dir, output-dir, and rate are required for headless mode."
            )
            return

    # Load and validate DLT parameters for each camera
    print("Loading DLT3D parameters...")
    dlt_params_dfs = []
    for dlt_file in dlt_files:
        df = pd.read_csv(dlt_file)
        if df.empty:
            print(f"Error: DLT3D file {os.path.basename(dlt_file)} is empty!")
            return
        dlt_params_dfs.append(df)

    print("Configuration complete:")
    print(f"  - DLT3D files: {len(dlt_files)} cameras")
    print(f"  - Input directory: {input_directory}")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {data_rate} Hz")
    for i, df in enumerate(dlt_params_dfs):
        print(f"  - Camera {i + 1}: {len(df)} frames")
    print("-" * 80)

    # Process files
    process_files_in_directory(dlt_params_dfs, input_directory, output_directory, data_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct 3D coordinates from N camera pixel CSV files using "
            "per-frame-varying DLT3D parameters (a DLT matrix, one row per frame, "
            "per camera). --input-dir must contain exactly one pixel CSV per "
            "camera, matching --dlt-files in count (paired by sorted filename)."
        )
    )
    parser.add_argument(
        "--dlt-files", nargs="+", help="Path to DLT3D parameter files (one per camera)"
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing exactly one pixel CSV per camera (matching --dlt-files)",
    )
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument(
        "--rate",
        type=float,
        help="Data frequency in Hz (accepts fractional rates, e.g. 119.88012001)",
    )
    args = parser.parse_args()

    run_rec3d(
        dlt_files=args.dlt_files,
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        data_rate=args.rate,
    )
