"""
================================================================================
vaila_datdistort.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 03 April 2025
Update: 10 February 2026
Version: 0.0.5
Python Version: 3.12.12

Description:
------------
This tool applies lens distortion correction to 2D coordinates from a DAT/CSV file
using the same camera calibration parameters as vaila_lensdistortvideo.py.

New Features in This Version:
------------------------------
1. CLI support for pipeline integration.
2. Fixed issue with column order in output file.
3. Improved error handling and logging.

How to use:
------------
GUI Mode (Default):
    uv run vaila/vaila_datdistort.py
    python vaila/vaila_datdistort.py
    (Follow the on-screen dialogs to select parameters file and input directory)

CLI Mode (from project root):
    uv run vaila/vaila_datdistort.py --params_file /path/to/params.toml --input /path/to/file_or_dir [--output_dir /path/to/output]
    python vaila/vaila_datdistort.py --params_file /path/to/params.toml --input /path/to/file_or_dir [--output_dir /path/to/output]

  Or run as module:
    uv run python -m vaila.vaila_datdistort --params_file /path/to/params.toml --input /path/to/file_or_dir [--output_dir /path/to/output]
    python -m vaila.vaila_datdistort --params_file /path/to/params.toml --input /path/to/file_or_dir [--output_dir /path/to/output]

  Arguments:
    --params_file  Path to the camera calibration parameters TOML file (required in CLI).
    --input        Single CSV/DAT file or directory containing CSV/DAT files to process.
    --output_dir   (Optional) Output directory. If omitted, a new subdirectory distorted_TIMESTAMP
                   is created under the input path; if given, files are written directly there.
  The parameters file is never processed (excluded from the batch).
  Output CSV "frame" column is written as integer.
  CLI help:  uv run vaila/vaila_datdistort.py --help

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
================================================================================
"""

import argparse
import json
import os
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

import cv2
import numpy as np
import pandas as pd
from rich import print


def _decimal_places(s):
    """Return number of decimal places in a string number, or -1 if not a float string."""
    s = str(s).strip()
    if not s or s.lower() in ("nan", "inf", "-inf"):
        return -1
    if "." not in s:
        return 0  # integer
    try:
        parts = s.split(".")
        if len(parts) != 2:
            return -1
        return len(parts[1])  # e.g. "1.0" -> 1, "1.00" -> 2
    except Exception:
        return -1


def _infer_float_precision(file_path, columns_of_interest, sep=",", max_rows=100):
    """
    Infer decimal places from the raw file for given columns.
    Returns a dict: column -> max decimal places (0 = int, 1 = .1f, etc.).
    """
    prec = dict.fromkeys(columns_of_interest, 0)
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            first = f.readline()
        if sep not in first and ";" in first:
            sep = ";"
        with open(file_path, encoding="utf-8", errors="replace") as f:
            df_raw = pd.read_csv(f, sep=sep, dtype=str, nrows=max_rows)
    except Exception:
        return prec
    for col in columns_of_interest:
        if col not in df_raw.columns:
            continue
        max_dp = 0
        for v in df_raw[col].dropna():
            dp = _decimal_places(v)
            if dp >= 0 and dp > max_dp:
                max_dp = dp
        prec[col] = max_dp
    return prec


def load_distortion_parameters(toml_path):
    """Load distortion parameters from a TOML file (fx, fy, cx, cy, k1, k2, k3, p1, p2)."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    params = {
        k: float(v)
        for k, v in data.items()
        if k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")
    }
    # #region agent log
    try:
        with open(
            "/home/preto/Preto/vaila/.cursor/debug-a5f5a000-975d-4bfc-9676-f9748629bda8.log", "a"
        ) as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "a5f5a000-975d-4bfc-9676-f9748629bda8",
                        "id": "datdistort_load_toml",
                        "timestamp": int(time.time() * 1000),
                        "location": "vaila_datdistort.load_distortion_parameters",
                        "message": "TOML params loaded",
                        "data": {
                            "script": "vaila_datdistort",
                            "path": toml_path,
                            "ext": os.path.splitext(toml_path)[1],
                            "keys_count": len(params),
                        },
                        "runId": "distort",
                        "hypothesisId": "A",
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    return params


def undistort_points(points, camera_matrix, dist_coeffs, image_size):
    """
    Undistort 2D points using camera calibration parameters.

    Args:
        points: Nx2 array of (x,y) coordinates
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        image_size: (width, height) of the original image

    Returns:
        Nx2 array of undistorted (x,y) coordinates
    """
    # Convert points to float32 numpy array
    points = np.array(points, dtype=np.float32)
    if points.size == 0:  # Check if points array is empty
        return points

    # Ensure camera matrix and dist_coeffs are float32
    camera_matrix = np.array(camera_matrix, dtype=np.float32)
    dist_coeffs = np.array(dist_coeffs, dtype=np.float32)

    # Reshape points to Nx1x2 format required by cv2.undistortPoints
    points = points.reshape(-1, 1, 2)

    # Get optimal new camera matrix
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 1, image_size
    )

    try:
        # Undistort points
        undistorted = cv2.undistortPoints(
            points, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        # Reshape back to Nx2
        return undistorted.reshape(-1, 2)
    except cv2.error as e:
        print(f"Error undistorting points: {e}")
        print(f"Points shape: {points.shape}")
        print(f"Points dtype: {points.dtype}")
        print(f"Camera matrix shape: {camera_matrix.shape}")
        print(f"Distortion coefficients shape: {dist_coeffs.shape}")
        raise


def process_dat_file(input_path, output_path, parameters, image_size=(1920, 1080)):
    """
    Process a DAT/CSV file to apply lens distortion correction to coordinates.
    """
    # Read the DAT file - try both comma and semicolon as separators
    try:
        df = pd.read_csv(input_path)
    except Exception:
        df = pd.read_csv(input_path, sep=";")

    # Create camera matrix
    camera_matrix = np.array(
        [
            [parameters["fx"], 0, parameters["cx"]],
            [0, parameters["fy"], parameters["cy"]],
            [0, 0, 1],
        ]
    )

    # Create distortion coefficients array
    dist_coeffs = np.array(
        [
            parameters["k1"],
            parameters["k2"],
            parameters["p1"],
            parameters["p2"],
            parameters["k3"],
        ]
    )

    # Use original column order from the file to preserve header order
    columns = df.columns.tolist()
    x_columns = [col for col in columns if col.endswith("_x")]
    y_columns = [col for col in columns if col.endswith("_y")]

    # Frame column: used only for row identification/logging, not for distortion math
    has_frame_col = "frame" in columns

    result_frames = []

    # Process each frame
    for idx, row in df.iterrows():
        frame_num = int(row["frame"]) if has_frame_col else idx

        # Collect valid points for this frame
        points = []
        for x_col, y_col in zip(x_columns, y_columns):
            try:
                x = float(row[x_col])
                y = float(row[y_col])
                # Only include valid coordinates (not 0,0 or NaN)
                if pd.notna(x) and pd.notna(y) and not (x == 0 and y == 0):
                    points.append([x, y])
            except (ValueError, TypeError):
                continue

        points = np.array(points)

        # Skip if no valid points
        if len(points) == 0:
            result_frames.append(row.to_dict())
            continue

        # Undistort valid points
        try:
            undistorted_points = undistort_points(points, camera_matrix, dist_coeffs, image_size)

            # Start with a copy of the original row to preserve all columns
            new_row = row.to_dict()
            point_idx = 0

            # Update only the coordinate columns with undistorted values
            for x_col, y_col in zip(x_columns, y_columns):
                if point_idx < len(undistorted_points):
                    # Get original values
                    orig_x = row[x_col]
                    orig_y = row[y_col]

                    # Only update if original point was valid
                    if pd.notna(orig_x) and pd.notna(orig_y) and not (orig_x == 0 and orig_y == 0):
                        new_row[x_col] = undistorted_points[point_idx][0]
                        new_row[y_col] = undistorted_points[point_idx][1]
                        point_idx += 1
                    # For invalid/zero values, keep original (already in new_row)
                # For remaining columns, keep original (already in new_row)

            result_frames.append(new_row)
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
            result_frames.append(row.to_dict())

    # Create output DataFrame with the same column order as input
    result_df = pd.DataFrame(result_frames, columns=df.columns)
    # Ensure "frame" column is integer in output (not float)
    if "frame" in result_df.columns:
        result_df["frame"] = result_df["frame"].astype(int)

    # Infer decimal precision from input file so output matches input
    coord_columns = x_columns + y_columns
    try:
        sep = "," if "," in open(input_path, encoding="utf-8", errors="replace").readline() else ";"
    except Exception:
        sep = ","
    prec = _infer_float_precision(input_path, coord_columns, sep=sep)
    # Use same precision for all float columns (match input, e.g. .1f -> .1f)
    max_decimals = max(prec.values()) if prec else 4
    # Format float columns to same number of decimals as input before writing
    for col in result_df.columns:
        if col == "frame":
            continue  # keep as int, already set above
        if result_df[col].dtype in (np.floating, float):
            n = prec.get(col, max_decimals)
            result_df[col] = result_df[col].apply(
                lambda x, nd=n: f"{x:.{nd}f}" if pd.notna(x) else ""
            )
    result_df.to_csv(output_path, index=False)


def select_file(title="Select a file", filetypes=(("All Files", "*.*"),)):
    """Open a dialog to select a file."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path


def select_directory(title="Select a directory"):
    """Open a dialog to select a directory."""
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title=title)
    return dir_path


def run_datdistort():
    """Main function to process DAT/CSV files using distortion parameters."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Batch lens distortion correction for CSV/DAT files. GUI mode if --input and --params_file are omitted."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a single CSV/DAT file or to a directory containing CSV/DAT files to process",
    )
    parser.add_argument(
        "--params_file",
        type=str,
        help="Path to the camera calibration parameters TOML file (required in CLI mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for processed files. If omitted, a new subdirectory (distorted_TIMESTAMP) is created under the input path.",
    )
    args = parser.parse_args()

    # Determine parameters file
    # #region agent log
    _log_path = "/home/preto/Preto/vaila/.cursor/debug-a5f5a000-975d-4bfc-9676-f9748629bda8.log"
    # #endregion
    if args.params_file:
        parameters_path = os.path.abspath(args.params_file)
        if not os.path.isfile(parameters_path):
            print(f"Error: Parameters file not found: {parameters_path}")
            return
        print(f"Using parameters file: {parameters_path}")
        # #region agent log
        try:
            with open(_log_path, "a") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "a5f5a000-975d-4bfc-9676-f9748629bda8",
                            "id": "datdistort_mode",
                            "timestamp": int(time.time() * 1000),
                            "location": "vaila_datdistort.run_datdistort",
                            "message": "Params source",
                            "data": {
                                "script": "vaila_datdistort",
                                "mode": "cli",
                                "params_path": parameters_path,
                            },
                            "runId": "distort",
                            "hypothesisId": "B",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
    else:
        print("Select the distortion parameters TOML file:")
        parameters_path = select_file(
            title="Select Calibration Parameters File",
            filetypes=(("TOML Files", "*.toml"), ("All Files", "*.*")),
        )
        if not parameters_path:
            print("No parameters file selected. Exiting.")
            return
        parameters_path = os.path.abspath(parameters_path)
        # #region agent log
        try:
            with open(_log_path, "a") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "a5f5a000-975d-4bfc-9676-f9748629bda8",
                            "id": "datdistort_mode",
                            "timestamp": int(time.time() * 1000),
                            "location": "vaila_datdistort.run_datdistort",
                            "message": "Params source",
                            "data": {
                                "script": "vaila_datdistort",
                                "mode": "gui",
                                "params_path": parameters_path,
                            },
                            "runId": "distort",
                            "hypothesisId": "B",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

    # Determine input: single file or directory
    if args.input:
        input_path = os.path.abspath(args.input)
        if os.path.isfile(input_path):
            input_dir = os.path.dirname(input_path)
            file_list = (
                [os.path.basename(input_path)]
                if input_path.lower().endswith((".csv", ".dat"))
                else []
            )
        elif os.path.isdir(input_path):
            input_dir = input_path
            file_list = [f for f in os.listdir(input_dir) if f.lower().endswith((".csv", ".dat"))]
        else:
            print(f"Error: Input not found: {input_path}")
            return
        print(f"Using input: {args.input}")
    else:
        print("Select the directory containing CSV/DAT files to process:")
        input_dir = select_directory(title="Select Directory with CSV/DAT Files")
        if not input_dir:
            print("No directory selected. Exiting.")
            return
        input_dir = os.path.abspath(input_dir)
        file_list = [f for f in os.listdir(input_dir) if f.lower().endswith((".csv", ".dat"))]
        single_file_mode = False

    # Load parameters once
    parameters = load_distortion_parameters(parameters_path)

    processed_count = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directory: if given, use as-is; else create new subdir under input
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        output_dir = os.path.join(input_dir, f"distorted_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    for filename in file_list:
        input_path = os.path.join(input_dir, filename)
        # Do not process the parameters file (avoid applying distortion to the params TOML)
        if os.path.abspath(input_path) == parameters_path:
            print(f"\nSkipping parameters file: {filename}")
            continue
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}_distorted.csv")

        try:
            print(f"\nProcessing: {filename}")
            process_dat_file(input_path, output_path, parameters)
            print(f"Saved as: {os.path.basename(output_path)}")
            processed_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print("\nProcessing complete!")
    print(f"Files processed: {processed_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    run_datdistort()
