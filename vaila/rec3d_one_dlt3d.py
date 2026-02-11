"""
================================================================================
Script: rec3d_one_dlt3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.5
Created: 02 August 2025
Last Updated: 11 February 2026

================================================================================
Description
================================================================================

Batch 3D reconstruction using the Direct Linear Transformation (DLT) method with
multiple cameras. For each camera you provide:
  - One DLT3D parameter file (11 coefficients per camera, e.g. from dlt3d.py).
  - One pixel-coordinate CSV with columns: frame, p1_x, p1_y, p2_x, p2_y, ..., pN_x, pN_y.

Frames common to all pixel files are reconstructed; output is written to a
timestamped subfolder in the chosen output directory.

Output files (same base name, in the output subfolder):
  - rec3d_YYYYMMDD_HHMMSS.csv   — 3D points (frame, p1_x, p1_y, p1_z, ...)
  - rec3d_YYYYMMDD_HHMMSS.3d    — same data, duplicate copy
  - rec3d_YYYYMMDD_HHMMSS_m.c3d — C3D in meters (POINT:UNITS=m, POINT:FRAMES set)
  - rec3d_YYYYMMDD_HHMMSS_mm.c3d — C3D in millimeters (POINT:UNITS=mm)

C3D files are compatible with viewc3d, viewc3d_pyvista, readc3d_export (inspect/
convert), and standard C3D tools. They are produced via readcsv_export.auto_create_c3d_from_csv.

================================================================================
Input file formats
================================================================================

DLT3D file:
  - CSV with one row of 11 DLT coefficients (e.g. from vaila dlt3d module).
  - One file per camera; order must match the pixel file order.

Pixel CSV:
  - Header: frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y (vailá standard).
  - One file per camera; same number of markers and matching frame sets recommended.
  - Files may be in different directories (GUI: one dialog per camera).

================================================================================
Usage
================================================================================

GUI (default):
  Run with no arguments or --gui. You are prompted for:
  1) Number of cameras
  2) One DLT3D file per camera (file dialogs; may be in different folders)
  3) One pixel CSV per camera (file dialogs; may be in different folders)
  4) Output directory
  5) Data rate in Hz (e.g. 60, 100)

CLI:
  Require --dlt3d, --pixels, --output. Optional --fps. Order of files must match.
  Example:
    python -m vaila.rec3d_one_dlt3d --dlt3d c1.dlt3d c2.dlt3d --pixels c1.csv c2.csv --fps 60 -o ./out
  Help:
    python -m vaila.rec3d_one_dlt3d --help

Documentation: vaila/help/rec3d_one_dlt3d.md and rec3d_one_dlt3d.html

Related modules:
  - dlt3d.py          — compute DLT3D coefficients from calibration data
  - readcsv_export.py — CSV to C3D (used internally); batch convert
  - readc3d_export.py — C3D to CSV; inspect C3D
  - viewc3d / viewc3d_pyvista — visualize C3D files
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import ezc3d
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

    for i, (A_params, (x, y)) in enumerate(zip(dlt_list, pixel_list)):
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


def save_rec3d_as_c3d(rec3d_df, output_dir, default_filename, point_rate=100, conversion_factor=1):
    """
    Converte o DataFrame de reconstrução 3D para um arquivo C3D e o salva.

    Args:
        rec3d_df (pd.DataFrame): DataFrame com os resultados (colunas "frame", "p1_x", "p1_y", "p1_z", ..., "p25_x", "p25_y", "p25_z").
        output_dir (str): Diretório onde o arquivo será salvo.
        default_filename (str): Nome padrão para o arquivo C3D.
        point_rate (int): Taxa de amostragem dos pontos (Hz).
        conversion_factor (float): Fator de conversão para as coordenadas (se necessário).
    """
    from tkinter import filedialog, messagebox

    num_frames = rec3d_df.shape[0]
    # Define the markers based on the actual columns
    x_columns = [col for col in rec3d_df.columns if col.endswith("_x") and col.startswith("p")]
    num_markers = len(x_columns)
    marker_labels = [f"p{i}" for i in range(1, num_markers + 1)]

    # Inicializa a matriz de pontos com shape (4, num_markers, num_frames)
    points_data = np.zeros((4, num_markers, num_frames))
    for i, marker in enumerate(marker_labels):
        try:
            points_data[0, i, :] = rec3d_df[f"{marker}_x"].values * conversion_factor
            points_data[1, i, :] = rec3d_df[f"{marker}_y"].values * conversion_factor
            points_data[2, i, :] = rec3d_df[f"{marker}_z"].values * conversion_factor
        except KeyError as e:
            messagebox.showerror("Error", f"Dados ausentes para o marcador {marker}: {e}")
            return
    points_data[3, :, :] = 1  # Coordenada homogênea

    c3d = ezc3d.c3d()
    # Usar a estrutura POINT já existente no ezc3d (preserva __METADATA__ para write())
    units_str = "mm" if conversion_factor == 1000 else "m"
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_labels
    c3d["parameters"]["POINT"]["RATE"]["value"] = [point_rate]
    c3d["parameters"]["POINT"]["UNITS"]["value"] = [units_str]
    c3d["parameters"]["POINT"]["FRAMES"]["value"] = [num_frames]
    c3d["data"]["points"] = points_data

    output_c3d = filedialog.asksaveasfilename(
        title="Salvar arquivo C3D",
        initialdir=output_dir,
        initialfile=default_filename,
        defaultextension=".c3d",
        filetypes=[("C3D files", "*.c3d")],
    )
    if output_c3d:
        try:
            c3d.write(output_c3d)
            messagebox.showinfo("Success", f"Arquivo C3D salvo em:\n{output_c3d}")
        except Exception as e:
            messagebox.showerror("Error", f"Erro ao salvar arquivo C3D: {e}")
    else:
        messagebox.showwarning("Warning", "Operação de salvamento do C3D cancelada.")


def run_reconstruction(dlt_files, pixel_files, output_directory, point_rate, gui=True):
    """
    Run 3D reconstruction from DLT3D and pixel CSV paths. Used by both GUI and CLI.

    Args:
        dlt_files: list of paths to DLT3D parameter files (one per camera)
        pixel_files: list of paths to pixel coordinate CSV files (one per camera)
        output_directory: directory where output subdir and files will be written
        point_rate: point data rate in Hz (e.g. 60, 100)
        gui: if True use messagebox for errors/success; if False use print only

    Returns:
        (new_dir, file_base) on success, None on failure.
    """
    def _err(msg):
        if gui:
            messagebox.showerror("Error", msg)
        else:
            print(f"Error: {msg}")
        return None

    # Load DLT3D parameters for each camera
    print("Loading DLT3D calibration parameters...")
    dlt_params_list = []
    for file in dlt_files:
        df = pd.read_csv(file)
        if df.empty:
            return _err(f"DLT3D file {os.path.basename(file)} is empty!")
        params = df.iloc[0, 1:].to_numpy().astype(float)
        dlt_params_list.append(params)

    # Load pixel coordinate data for each camera
    print("Loading pixel coordinate data...")
    pixel_dfs = []
    for file in pixel_files:
        df = pd.read_csv(file)
        required_cols = {"frame", "p1_x", "p1_y"}
        if not required_cols.issubset(df.columns):
            return _err(
                f"Pixel coordinate file {os.path.basename(file)} does not contain required columns 'frame', 'p1_x', and 'p1_y'."
            )
        if df.shape[0] == 1:
            df["frame"] = 0
        pixel_dfs.append(df)

    first_df = pixel_dfs[0]
    x_columns = [col for col in first_df.columns if col.endswith("_x") and col.startswith("p")]
    num_markers = len(x_columns)
    print(f"Detected {num_markers} markers for 3D reconstruction")

    frame_sets = [set(df["frame"]) for df in pixel_dfs]
    common_frames = set.intersection(*frame_sets)
    if not common_frames:
        return _err("No common frames found among pixel files!")
    common_frames = sorted(common_frames)
    print(f"Processing {len(common_frames)} common frames...")

    total_frames = len(common_frames)
    total_cols = 1 + (num_markers * 3)
    print(f"Pre-allocating array for {total_frames} frames x {num_markers} markers...")
    reconstruction_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)
    reconstruction_array[:, 0] = common_frames

    progress_step = max(1, total_frames // 20)
    for frame_idx, frame in enumerate(common_frames):
        if frame_idx % progress_step == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

        frame_data = []
        valid_frame = True
        for df in pixel_dfs:
            frame_row = df[df["frame"] == frame]
            if frame_row.empty:
                valid_frame = False
                break
            frame_data.append(frame_row.iloc[0])

        if not valid_frame:
            continue

        for marker in range(1, num_markers + 1):
            pixel_obs_list = []
            valid_marker = True
            for frame_row in frame_data:
                try:
                    x_obs = float(frame_row[f"p{marker}_x"])
                    y_obs = float(frame_row[f"p{marker}_y"])
                    if np.isnan(x_obs) or np.isnan(y_obs):
                        valid_marker = False
                        break
                    pixel_obs_list.append((x_obs, y_obs))
                except Exception:
                    valid_marker = False
                    break

            col_start = 1 + (marker - 1) * 3
            if not valid_marker or len(pixel_obs_list) != len(dlt_params_list):
                pass
            else:
                point3d = rec3d_multicam(dlt_params_list, pixel_obs_list)
                reconstruction_array[frame_idx, col_start : col_start + 3] = point3d

    print("3D reconstruction completed!")

    header = ["frame"]
    for marker in range(1, num_markers + 1):
        header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])

    rec3d_df = pd.DataFrame(reconstruction_array, columns=header)
    valid_frames_mask = ~rec3d_df.iloc[:, 1:].isna().all(axis=1)
    rec3d_df = rec3d_df[valid_frames_mask].reset_index(drop=True)

    if rec3d_df.empty:
        return _err("No valid 3D reconstruction could be performed!")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(output_directory, f"vaila_rec3d_{timestamp}")
    os.makedirs(new_dir, exist_ok=True)

    file_base = f"rec3d_{timestamp}"
    file_3d_path = os.path.join(new_dir, f"{file_base}.3d")
    file_csv_path = os.path.join(new_dir, f"{file_base}.csv")

    print("Saving 3D reconstruction results...")
    rec3d_df.to_csv(file_3d_path, index=False, float_format="%.6f")
    rec3d_df.to_csv(file_csv_path, index=False, float_format="%.6f")

    rec3d_df_for_c3d = rec3d_df.copy()
    new_columns = []
    for col in rec3d_df_for_c3d.columns:
        if col.lower() != "frame":
            parts = col.split("_")
            if len(parts) == 2:
                new_columns.append(parts[0] + "_" + parts[1].upper())
            else:
                new_columns.append(col)
        else:
            new_columns.append(col)
    rec3d_df_for_c3d.columns = new_columns

    m_conversion = 1
    mm_conversion = 1000

    import vaila.readcsv_export as readcsv_export

    c3d_output_path_m = os.path.join(new_dir, f"{file_base}_m.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_m,
            point_rate=point_rate,
            conversion_factor=m_conversion,
        )
        if gui:
            messagebox.showinfo("Success", f"C3D file (meters) saved at:\n{c3d_output_path_m}")
        print("C3D file (meters) created successfully")
    except Exception as e:
        if gui:
            messagebox.showerror("Error", f"Failed to save C3D file (meters): {e}")
        print("Error saving C3D file (meters):", e)
        return None

    c3d_output_path_mm = os.path.join(new_dir, f"{file_base}_mm.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_mm,
            point_rate=point_rate,
            conversion_factor=mm_conversion,
        )
        if gui:
            messagebox.showinfo("Success", f"C3D file (millimeters) saved at:\n{c3d_output_path_mm}")
        print("C3D file (millimeters) created successfully")
    except Exception as e:
        if gui:
            messagebox.showerror("Error", f"Failed to save C3D file (millimeters): {e}")
        print("Error saving C3D file (millimeters):", e)
        return None

    print("\n=== Processing Complete ===")
    print(f"Processed {len(common_frames)} frames with {num_markers} markers")
    print(f"Output directory: {new_dir}")
    print("Files created:")
    print(f"  - {file_base}.csv (CSV format)")
    print(f"  - {file_base}.3d (3D format)")
    print(f"  - {file_base}_m.c3d (C3D in meters)")
    print(f"  - {file_base}_mm.c3d (C3D in millimeters)")

    if gui:
        messagebox.showinfo(
            "Processing Complete",
            f"3D reconstruction completed successfully!\n\n"
            f"Processed: {len(common_frames)} frames with {num_markers} markers\n"
            f"Output directory: {os.path.basename(new_dir)}\n\n"
            f"Files created:\n"
            f"• CSV and 3D format files\n"
            f"• C3D files (meters and millimeters)",
        )

    return (new_dir, file_base)


def run_rec3d_one_dlt3d():
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec3d_one_dlt3d.py...")
    print("-" * 80)

    # --- New changes: Save files in a user-selected directory ---
    root = Tk()
    root.withdraw()

    # Step 0: Ask number of cameras (allows selecting files from different directories)
    print("Step 0: Number of cameras...")
    n_cameras = simpledialog.askinteger(
        "Number of cameras",
        "Enter number of cameras (e.g. 2):",
        minvalue=1,
        maxvalue=20,
        initialvalue=2,
    )
    if n_cameras is None:
        messagebox.showwarning("Cancelled", "Operation cancelled.")
        root.destroy()
        return

    # Step 1: Select DLT3D parameter files (one dialog per camera so each can be from a different directory)
    print("Step 1: Selecting DLT3D parameter files...")
    dlt_files = []
    for i in range(1, n_cameras + 1):
        path = filedialog.askopenfilename(
            title=f"Select DLT3D file for camera {i}",
            filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")],
        )
        if not path:
            messagebox.showerror("Error", f"No DLT3D file selected for camera {i}!")
            root.destroy()
            return
        dlt_files.append(path)

    # Step 2: Select pixel coordinate CSV files (one dialog per camera)
    print("Step 2: Selecting pixel coordinate CSV files...")
    pixel_files = []
    for i in range(1, n_cameras + 1):
        path = filedialog.askopenfilename(
            title=f"Select pixel coordinate CSV for camera {i}",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            messagebox.showerror("Error", f"No pixel coordinate file selected for camera {i}!")
            root.destroy()
            return
        pixel_files.append(path)

    if len(dlt_files) != len(pixel_files):
        messagebox.showerror(
            "Error",
            "The number of DLT3D files must match the number of pixel coordinate files!",
        )
        root.destroy()
        return

    # Step 3: Select output directory
    print("Step 3: Selecting output directory...")
    output_directory = filedialog.askdirectory(title="Select Output Directory for Results")
    if not output_directory:
        messagebox.showerror("Error", "No output directory selected. Operation cancelled.")
        return

    # Step 4: Ask for data frequency
    print("Step 4: Setting data frequency...")
    point_rate = simpledialog.askinteger(
        "Data Frequency",
        "Enter the point data rate (Hz):",
        minvalue=1,
        initialvalue=100,
    )
    if point_rate is None:
        messagebox.showerror("Error", "Point data rate is required. Operation cancelled.")
        return

    # Configuration summary
    print("Configuration complete:")
    print(f"  - DLT3D files: {len(dlt_files)} cameras")
    print(f"  - Pixel files: {len(pixel_files)} cameras")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {point_rate} Hz")
    print("-" * 80)

    run_reconstruction(dlt_files, pixel_files, output_directory, point_rate, gui=True)
    root.destroy()


def _cli_run():
    """CLI entry: argparse for --dlt3d, --pixels, --fps, --output."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch 3D reconstruction from DLT3D parameter files and pixel CSV files "
            "(one file per camera). Output: CSV, .3d, and C3D files (meters and mm) in a "
            "timestamped subfolder under the given output directory. "
            "Without --dlt3d/--pixels/--output, or with --gui, launches the GUI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input:
  DLT3D files: one per camera (CSV with 11 DLT coefficients; e.g. from dlt3d module).
  Pixel files:  one per camera (CSV with header frame,p1_x,p1_y,p2_x,p2_y,...).
  Order must match: first DLT3D with first pixel file, etc.

Output:
  A new subfolder is created under DIR with name rec3d_YYYYMMDD_HHMMSS containing:
  rec3d_*.csv, rec3d_*.3d, rec3d_*_m.c3d, rec3d_*_mm.c3d.

Examples:
  %(prog)s --dlt3d cam1.dlt3d cam2.dlt3d --pixels cam1.csv cam2.csv --fps 60 -o ./out
  %(prog)s -o ./results --dlt3d a.dlt3d b.dlt3d --pixels a.csv b.csv
  %(prog)s --gui

See also: vaila/help/rec3d_one_dlt3d.md
        """,
    )
    parser.add_argument(
        "--dlt3d",
        nargs="+",
        metavar="FILE",
        help="DLT3D parameter files, one per camera (order must match --pixels)",
    )
    parser.add_argument(
        "--pixels",
        nargs="+",
        metavar="FILE",
        help="Pixel coordinate CSV files, one per camera (order must match --dlt3d)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=100,
        metavar="HZ",
        help="Point data rate in Hz for C3D/CSV (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        dest="output",
        help="Output directory; a timestamped subfolder will be created here",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI (file dialogs) instead of CLI",
    )
    args = parser.parse_args()

    if args.gui or (not args.dlt3d and not args.pixels and not args.output):
        run_rec3d_one_dlt3d()
        return

    if not args.dlt3d or not args.pixels:
        print("Error: CLI mode requires --dlt3d and --pixels.", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: CLI mode requires --output.", file=sys.stderr)
        sys.exit(1)
    if len(args.dlt3d) != len(args.pixels):
        print(
            "Error: Number of --dlt3d files must match number of --pixels files.",
            file=sys.stderr,
        )
        sys.exit(1)

    result = run_reconstruction(
        args.dlt3d,
        args.pixels,
        os.path.abspath(args.output),
        args.fps,
        gui=False,
    )
    if result is None:
        sys.exit(1)
    print(f"Output directory: {result[0]}")


if __name__ == "__main__":
    _cli_run()
