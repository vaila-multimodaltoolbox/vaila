"""
================================================================================
Script: dlt3d.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Roberto Pereira Santiago
Version: 0.3.120
Create: 24 February, 2025
Last Updated: 03 September 2026

Description:
    This script calculates the Direct Linear Transformation (DLT) parameters for 3D coordinate transformations.
    It uses pixel coordinates from video calibration data and corresponding real-world 3D coordinates to compute the 11
    DLT parameters for each frame (or uses a single row of real-world coordinates for all frames).

    New Features:
      - Generates a REF3D template (with _x, _y, _z columns) from the pixel file.
      - Auto-detects REF3D layout (format 1 wide CSV, format 2 xyz rows, format 3 indexed xyz rows)
        and normalizes internally to format 1 before DLT3D.
      - Validates that the REF3D file contains the three axes for each point.
      - Updated calculation of DLT parameters (11 parameters) using least squares.
      - Graphical file selection using Tkinter.
      - Improved console output.

    Point matching (pixel <-> REF3D): points are correlated by LABEL (the "p3"
    in "p3_x"/"p3_y"/"p3_z"), not by column order, since a REF3D file may
    calibrate more points than a given pixel file tracks. Only the common
    points are used; if fewer than 6 common points are found (the minimum for
    an 11-parameter DLT3D solve), no parameters are computed and a clear
    message is printed instead of a raw KeyError.

Usage:
    Run the script and select a pixel coordinate CSV file. Then, choose whether to create a REF3D template.
    If you opt to create the template, edit it with the real-world coordinates and run the DLT process again.
    Otherwise, select the edited REF3D file, and the script will calculate the parameters and save an output file
    with the .dlt3d extension.
"""

import argparse
import os
from tkinter import Tk, filedialog, messagebox

import numpy as np
import pandas as pd
from rich import print


def read_pixel_file(file_path):
    """Reads the pixel coordinate CSV file."""
    df = pd.read_csv(file_path)
    return df


def _point_numbers_from_columns(columns) -> set[int]:
    """Point indices (the N in pN_x/pN_y/pN_z) present in a column list."""
    numbers: set[int] = set()
    for col in columns:
        if col.startswith("p") and "_" in col:
            parts = col.split("_")
            if len(parts) >= 2 and parts[0][1:].isdigit():
                numbers.add(int(parts[0][1:]))
    return numbers


def _is_format1_dataframe(df: pd.DataFrame) -> bool:
    """True when *df* already uses the wide format-1 header (frame, p1_x, …)."""
    if "frame" not in df.columns:
        return False
    return bool(_point_numbers_from_columns(df.columns))


def _validate_format1_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    """Ensure format-1 REF3D has _x/_y/_z columns for every detected point index.

    Point indices are NOT assumed to be a 1-based contiguous range: pixel
    files from getpixelvideo.py's default/sequential mode start at p0, so a
    REF3D file may legitimately use p0..p(N-1) instead of p1..pN.
    """
    point_numbers = _point_numbers_from_columns(df.columns)
    if not point_numbers:
        return None
    expected_columns = []
    for i in sorted(point_numbers):
        expected_columns.extend([f"p{i}_x", f"p{i}_y", f"p{i}_z"])
    if not all(col in df.columns for col in expected_columns):
        return None
    return df


def _build_format1_from_point_rows(rows: list[tuple[int, float, float, float]]) -> pd.DataFrame:
    """Convert (p_index, x, y, z) tuples into a single-row format-1 DataFrame."""
    row_map = {idx: (x, y, z) for idx, x, y, z in rows}
    data: dict[str, list[float | int]] = {"frame": [0]}
    for idx in sorted(row_map):
        x, y, z = row_map[idx]
        data[f"p{idx}_x"] = [x]
        data[f"p{idx}_y"] = [y]
        data[f"p{idx}_z"] = [z]
    return pd.DataFrame(data)


def detect_ref3d_format(file_path: str) -> int | None:
    """
    Detect REF3D file layout.

    Returns 1 (wide CSV with header), 2 (xyz rows, no header), 3 (index + xyz rows),
    or None when the file cannot be classified.
    """
    path = os.path.abspath(file_path)
    if not os.path.isfile(path):
        return None

    with open(path, encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return None

    lowered = first_line.lower()
    if lowered.startswith("frame,") or ",p1_x," in lowered or lowered.endswith(",p1_x"):
        return 1

    raw = pd.read_csv(path, header=None)
    if raw.empty:
        return None

    ncols = raw.shape[1]
    if ncols == 3:
        numeric = pd.to_numeric(raw.stack(), errors="coerce")
        if numeric.notna().all():
            return 2
    if ncols == 4:
        index_col = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        coords = pd.to_numeric(raw.iloc[:, 1:].stack(), errors="coerce")
        if (
            index_col.notna().all()
            and coords.notna().all()
            and np.allclose(index_col.to_numpy(), np.round(index_col.to_numpy()))
        ):
            return 3

    # Last chance: header row without the literal substring checks above.
    headed = pd.read_csv(path)
    if _is_format1_dataframe(headed):
        return 1
    return None


def normalize_ref3d_to_format1(file_path: str) -> pd.DataFrame | None:
    """
    Load any supported REF3D variant and return the canonical format-1 DataFrame.

    Format 1: ``frame,p1_x,p1_y,p1_z,...`` (wide, optional multi-row per frame).
    Format 2: one ``x,y,z`` triplet per row, no header; row order defines p1..pN.
    Format 3: one ``index,x,y,z`` row per point, no header; index column defines pN.
    """
    fmt = detect_ref3d_format(file_path)
    if fmt is None:
        return None

    if fmt == 1:
        df = pd.read_csv(file_path)
        return _validate_format1_dataframe(df)

    raw = pd.read_csv(file_path, header=None)
    rows: list[tuple[int, float, float, float]] = []
    if fmt == 2:
        for row_idx in range(len(raw)):
            row = raw.iloc[row_idx]
            x, y, z = (float(row[0]), float(row[1]), float(row[2]))
            rows.append((row_idx + 1, x, y, z))
    else:
        for _, row in raw.iterrows():
            idx = int(row[0])
            x, y, z = (float(row[1]), float(row[2]), float(row[3]))
            rows.append((idx, x, y, z))

    if len(rows) < 6:
        return None
    return _validate_format1_dataframe(_build_format1_from_point_rows(rows))


def read_ref3d_file(file_path):
    """Read REF3D (formats 1–3) and return normalized format-1 coordinates."""
    df = normalize_ref3d_to_format1(file_path)
    if df is None:
        print("Error: REF3D file does not contain the expected columns with _z coordinates!")
    return df


def calculate_dlt3d_params(pixel_coords, ref_coords):
    """
    Computes the 11 DLT3d parameters using the following models:

      u = (L1*X + L2*Y + L3*Z + L4) / (L9*X + L10*Y + L11*Z + 1)
      v = (L5*X + L6*Y + L7*Z + L8) / (L9*X + L10*Y + L11*Z + 1)

    The equations are rearranged to form a linear system:
      X   Y   Z   1   0   0   0   0  -uX  -uY  -uZ = u
      0   0   0   0   X   Y   Z   1  -vX  -vY  -vZ = v
    """
    n = pixel_coords.shape[0]
    A = np.zeros((2 * n, 11))
    B = np.zeros((2 * n,))
    for i in range(n):
        X, Y, Z = ref_coords[i, :]
        u, v = pixel_coords[i, :]
        # First equation (for u)
        A[2 * i, 0:4] = [X, Y, Z, 1]
        A[2 * i, 8:11] = -u * np.array([X, Y, Z])
        B[2 * i] = u
        # Second equation (for v)
        A[2 * i + 1, 4:8] = [X, Y, Z, 1]
        A[2 * i + 1, 8:11] = -v * np.array([X, Y, Z])
        B[2 * i + 1] = v
    # Solve the system A * L = B using least squares
    L, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    return L


def _point_has_axes(row, point_idx: int, suffixes: tuple[str, ...]) -> bool:
    """True when *row* has non-empty (non-NaN) values for point_idx's given axes."""
    return all(pd.notna(row.get(f"p{point_idx}{suffix}")) for suffix in suffixes)


def process_files(pixel_file, ref3d_file):
    """
    Processes the pixel and REF3D files.
    If the REF3D file contains only one row, the same real-world points are used for every frame.

    Empty/NaN pixel coordinates are skipped per point rather than crashing:
    getpixelvideo.py's sequential-mode CSV keeps one row per video frame even
    when markers were only placed on a single frame, leaving every other
    frame's pN_x/pN_y blank. Feeding those NaNs straight into the
    least-squares solve raised ``numpy.linalg.LinAlgError: SVD did not
    converge`` instead of a clear message. Now, for each frame, only the
    common points that actually have data are used; a frame is skipped (with
    a message) when fewer than 6 such points remain.
    """
    pixel_df = read_pixel_file(pixel_file)
    ref_df = read_ref3d_file(ref3d_file)
    if ref_df is None:
        return None

    def _point_numbers(columns):
        """Point indices (the N in pN_x/pN_y/pN_z) present in a column list."""
        numbers = set()
        for col in columns:
            if col.startswith("p") and "_" in col:
                parts = col.split("_")
                if len(parts) >= 2 and parts[0][1:].isdigit():
                    numbers.add(int(parts[0][1:]))
        return numbers

    # Only use points present in BOTH files: the REF3D file may calibrate a
    # different (often larger) set of points than what a given pixel/video
    # file actually tracks. Assuming the pixel file's point range also exists
    # in the REF3D file crashes with a raw KeyError as soon as they diverge.
    pixel_points = _point_numbers(pixel_df.columns)
    ref_points = _point_numbers(ref_df.columns)
    common_points = sorted(pixel_points & ref_points)

    if len(common_points) < 6:
        print(
            f"Error: only {len(common_points)} common point(s) found between the pixel and "
            "REF3D files; DLT3D needs at least 6 non-coplanar points to be well-determined "
            "(11 unknowns, 2 equations per point)."
        )
        return None

    missing_in_ref = sorted(pixel_points - ref_points)
    if missing_in_ref:
        print(
            f"Warning: pixel file has point(s) {missing_in_ref} not present in the REF3D "
            f"file; using only the {len(common_points)} common point(s): {common_points}"
        )

    dlt_params_all = {}
    skipped_frames = []

    # If the REF3D file consists of only one row, use it for all frames:
    if len(ref_df) == 1:
        ref_line = ref_df.iloc[0]
        # Drop common points whose real-world coordinates are themselves empty.
        ref_valid_points = [
            i for i in common_points if _point_has_axes(ref_line, i, ("_x", "_y", "_z"))
        ]
        ref_coords_map = {
            i: [ref_line[f"p{i}_x"], ref_line[f"p{i}_y"], ref_line[f"p{i}_z"]]
            for i in ref_valid_points
        }
        for _, row in pixel_df.iterrows():
            frame = row["frame"]
            frame_points = [i for i in ref_valid_points if _point_has_axes(row, i, ("_x", "_y"))]
            if len(frame_points) < 6:
                skipped_frames.append((frame, len(frame_points)))
                continue
            pixel_coords_arr = np.array([[row[f"p{i}_x"], row[f"p{i}_y"]] for i in frame_points])
            ref_coords_arr = np.array([ref_coords_map[i] for i in frame_points])
            L = calculate_dlt3d_params(pixel_coords_arr, ref_coords_arr)
            dlt_params_all[frame] = L
    else:
        # If REF3D contains multiple rows, match the frame numbers
        for _, row in pixel_df.iterrows():
            frame = row["frame"]
            ref_line = ref_df[ref_df["frame"] == frame]
            if ref_line.empty:
                print(f"Frame {frame} not found in REF3D file.")
                continue
            ref_line = ref_line.iloc[0]
            frame_points = [
                i
                for i in common_points
                if _point_has_axes(ref_line, i, ("_x", "_y", "_z"))
                and _point_has_axes(row, i, ("_x", "_y"))
            ]
            if len(frame_points) < 6:
                skipped_frames.append((frame, len(frame_points)))
                continue
            pixel_coords_arr = np.array([[row[f"p{i}_x"], row[f"p{i}_y"]] for i in frame_points])
            ref_coords_arr = np.array(
                [
                    [ref_line[f"p{i}_x"], ref_line[f"p{i}_y"], ref_line[f"p{i}_z"]]
                    for i in frame_points
                ]
            )
            L = calculate_dlt3d_params(pixel_coords_arr, ref_coords_arr)
            dlt_params_all[frame] = L

    if skipped_frames:
        preview = ", ".join(f"{f} ({n} pt)" for f, n in skipped_frames[:10])
        more = f", … +{len(skipped_frames) - 10} more" if len(skipped_frames) > 10 else ""
        print(
            f"Warning: skipped {len(skipped_frames)} frame(s) with fewer than 6 valid "
            f"(non-empty) common points: {preview}{more}"
        )

    if not dlt_params_all:
        print("Error: no frame had at least 6 valid (non-empty) common points; nothing to solve.")
        return None

    return dlt_params_all


def save_dlt_parameters(output_file, dlt_params, show_gui=True):
    """Saves the computed DLT3d parameters to a CSV file without spaces after commas."""
    with open(output_file, "w") as f:
        f.write(
            "frame,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11\n"
        )  # Please verify header names if needed
        for frame, params in dlt_params.items():
            param_str = ",".join([f"{p:.6f}" for p in params])
            f.write(f"{frame},{param_str}\n")
    # Show a message box indicating success
    if show_gui:
        try:
            import tkinter as tk

            if tk._default_root is not None:
                messagebox.showinfo("Success", f"DLT3d file saved successfully: {output_file}")
        except Exception:
            pass
    print(f"DLT3d parameters saved to {output_file}")


def main(pixel_file=None, real_file=None, create_ref=False):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting DLT3D module...")

    if pixel_file is None:
        root = Tk()
        root.withdraw()
        pixel_file = filedialog.askopenfilename(
            title="Select the pixel coordinate file", filetypes=[("CSV files", "*.csv")]
        )
        if not pixel_file:
            print("Pixel file selection canceled.")
            return

        # Ask the user if they want to generate a REF3D template
        create_ref = (
            messagebox.askquestion("Mode", "Do you want to create a REF3D template?") == "yes"
        )

    # Determine the number of points from the pixel file
    pixel_df = read_pixel_file(pixel_file)
    pixel_columns = list(pixel_df.columns)
    point_columns = [
        col for col in pixel_columns if col.startswith("p") and ("_x" in col or "_y" in col)
    ]
    point_numbers = set()
    for col in point_columns:
        if "_" in col:
            parts = col.split("_")
            if len(parts) >= 2:
                point_num = parts[0][1:]  # Remove 'p' from 'p1', 'p20', etc.
                if point_num.isdigit():
                    point_numbers.add(int(point_num))

    if create_ref:
        real_file = os.path.splitext(pixel_file)[0] + ".ref3d"
        # Create a template with header for points with _x, _y, _z (default value 0.0).
        # IMPORTANT: use the ACTUAL point indices found in the pixel file, not
        # range(1, max+1). getpixelvideo.py's default/sequential mode numbers
        # markers starting at p0 (keypoint_start_idx=0, keypoint_index_base=0),
        # so a pixel file with 10 markers has columns p0_x..p9_x. Assuming a
        # 1-based range here silently dropped p0 and produced a REF3D template
        # with one fewer point than the pixel file, shifted by one label.
        template_data = {"frame": [0]}
        for i in sorted(point_numbers):
            template_data[f"p{i}_x"] = [0]
            template_data[f"p{i}_y"] = [0]
            template_data[f"p{i}_z"] = [0]
        template_df = pd.DataFrame(template_data)
        template_df.to_csv(real_file, index=False)
        if pixel_file is None:  # only show messagebox if GUI was used
            messagebox.showinfo("Success", f"REF3D template created: {real_file}")
        print(f"REF3D template created: {real_file}")
        print("Please edit the REF3D file with the real coordinates and run the DLT process again.")
        return
    else:
        if real_file is None:
            real_file = filedialog.askopenfilename(
                title="Select the real 3D coordinates file",
                filetypes=[("REF3D files", "*.ref3d")],
            )
            if not real_file:
                print("Real file selection canceled.")
                return

    dlt_params = process_files(pixel_file, real_file)
    if dlt_params is None:
        print("Error processing the files.")
        return
    output_file = os.path.splitext(pixel_file)[0] + ".dlt3d"
    show_gui = pixel_file is None
    save_dlt_parameters(output_file, dlt_params, show_gui=show_gui)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLT3D Reconstruction")
    parser.add_argument("--pixel", help="Path to pixel coordinate CSV file")
    parser.add_argument("--real", help="Path to real-world coordinate REF3D file")
    parser.add_argument(
        "--create-ref", action="store_true", help="Create a REF3D file from the pixel file"
    )
    args = parser.parse_args()

    main(pixel_file=args.pixel, real_file=args.real, create_ref=args.create_ref)
