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
Version: 0.0.4
Created: August 02, 2025
Last Updated: August 02, 2025

Description:
    Optimized batch 3D reconstruction using the Direct Linear Transformation (DLT) method with multiple cameras.
    Each camera has a corresponding DLT3D parameter file (one set of 11 parameters per file) and a pixel coordinate CSV file.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    For each common frame found among the pixel files, the script reconstructs 3D points for all detected markers.
    The output includes CSV, 3D format, and C3D files with reconstructed coordinates (x,y,z) for each marker.
    
    Optimizations:
    - Reduced debug output for cleaner processing feedback
    - Vectorized NumPy operations for faster computation
    - Progress tracking for large datasets
    - Pre-allocated NumPy arrays to eliminate dynamic memory allocation
    - Direct array indexing instead of list.append() operations
    - Improved memory efficiency and cache locality
"""

import os
from pathlib import Path
from rich import print
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from tkinter import filedialog, Tk, messagebox, simpledialog
from datetime import datetime
import ezc3d


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


def save_rec3d_as_c3d(
    rec3d_df, output_dir, default_filename, point_rate=100, conversion_factor=1
):
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
    x_columns = [
        col for col in rec3d_df.columns if col.endswith("_x") and col.startswith("p")
    ]
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
            messagebox.showerror(
                "Error", f"Dados ausentes para o marcador {marker}: {e}"
            )
            return
    points_data[3, :, :] = 1  # Coordenada homogênea

    c3d = ezc3d.c3d()
    # Adiciona o grupo __METADATA__ para evitar o erro na escrita do arquivo C3D
    c3d["parameters"]["__METADATA__"] = {}

    # Define parâmetros básicos do C3D
    c3d["parameters"]["POINT"] = {}
    c3d["parameters"]["POINT"]["LABELS"] = {"value": marker_labels}
    c3d["parameters"]["POINT"]["RATE"] = {"value": [point_rate]}
    c3d["parameters"]["POINT"]["UNITS"] = {"value": ["m"]}

    # Atribui os pontos diretamente (sem reatribuir "c3d['data']")
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
            print("[DEBUG] Arquivo C3D salvo em", output_c3d)
        except Exception as e:
            messagebox.showerror("Error", f"Erro ao salvar arquivo C3D: {e}")
            print("[DEBUG] Erro ao escrever arquivo C3D:", e)
    else:
        messagebox.showwarning("Warning", "Operação de salvamento do C3D cancelada.")


def run_rec3d_one_dlt3d():
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec3d_one_dlt3d.py...")
    print("-" * 80)
    
    # --- New changes: Save files in a user-selected directory ---
    root = Tk()
    root.withdraw()

    # Step 1: Select DLT3D parameter files
    print("Step 1: Selecting DLT3D parameter files...")
    dlt_files = filedialog.askopenfilenames(
        title="Select DLT3D parameter files (one per camera)",
        filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")],
    )
    if not dlt_files:
        messagebox.showerror("Error", "No DLT3D files selected!")
        return

    # Step 2: Select pixel coordinate CSV files
    print("Step 2: Selecting pixel coordinate CSV files...")
    pixel_files = filedialog.askopenfilenames(
        title="Select pixel coordinate CSV files (one per camera)",
        filetypes=[("CSV files", "*.csv")],
    )

    if not pixel_files:
        messagebox.showerror("Error", "No pixel coordinate files selected!")
        return

    if len(dlt_files) != len(pixel_files):
        messagebox.showerror(
            "Error",
            "The number of DLT3D files must match the number of pixel coordinate files!",
        )
        return

    # Step 3: Select output directory
    print("Step 3: Selecting output directory...")
    output_directory = filedialog.askdirectory(
        title="Select Output Directory for Results"
    )
    if not output_directory:
        messagebox.showerror("Error", "No output directory selected. Operation cancelled.")
        return

    # Step 4: Ask for data frequency
    print("Step 4: Setting data frequency...")
    point_rate = simpledialog.askinteger(
        "Data Frequency", 
        "Enter the point data rate (Hz):", 
        minvalue=1, 
        initialvalue=100
    )
    if point_rate is None:
        messagebox.showerror("Error", "Point data rate is required. Operation cancelled.")
        return

    # Configuration summary
    print(f"Configuration complete:")
    print(f"  - DLT3D files: {len(dlt_files)} cameras")
    print(f"  - Pixel files: {len(pixel_files)} cameras")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {point_rate} Hz")
    print("-" * 80)

    # Load DLT3D parameters for each camera (use first row from each file, skipping the frame column)
    print("Loading DLT3D calibration parameters...")
    dlt_params_list = []
    for file in dlt_files:
        df = pd.read_csv(file)
        if df.empty:
            messagebox.showerror(
                "Error", f"DLT3D file {os.path.basename(file)} is empty!"
            )
            return
        params = df.iloc[0, 1:].to_numpy().astype(float)
        dlt_params_list.append(params)

    # Load pixel coordinate data for each camera using Vailá's standard header
    print("Loading pixel coordinate data...")
    pixel_dfs = []
    for file in pixel_files:
        df = pd.read_csv(file)
        required_cols = {"frame", "p1_x", "p1_y"}
        if not required_cols.issubset(df.columns):
            messagebox.showerror(
                "Error",
                f"Pixel coordinate file {os.path.basename(file)} does not contain required columns 'frame', 'p1_x', and 'p1_y'.",
            )
            return
        # If the file has only one row, force the frame to 0 for consistency across cameras.
        if df.shape[0] == 1:
            df["frame"] = 0
        pixel_dfs.append(df)

    # Calculate number of markers by counting the columns that match the pattern "p{number}_x"
    first_df = pixel_dfs[0]
    x_columns = [
        col for col in first_df.columns if col.endswith("_x") and col.startswith("p")
    ]
    num_markers = len(x_columns)
    print(f"Detected {num_markers} markers for 3D reconstruction")

    # Determine the common frames across all pixel files
    frame_sets = [set(df["frame"]) for df in pixel_dfs]
    common_frames = set.intersection(*frame_sets)
    if not common_frames:
        messagebox.showerror("Error", "No common frames found among pixel files!")
        return
    common_frames = sorted(common_frames)
    print(f"Processing {len(common_frames)} common frames...")

    # Pre-allocate NumPy array for better performance
    total_frames = len(common_frames)
    total_cols = 1 + (num_markers * 3)  # frame + (x,y,z) for each marker
    
    print(f"Pre-allocating array for {total_frames} frames x {num_markers} markers...")
    reconstruction_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)
    
    # Set frame numbers in first column
    reconstruction_array[:, 0] = common_frames
    
    # Progress tracking
    progress_step = max(1, total_frames // 20)  # Show progress every 5%
    
    for frame_idx, frame in enumerate(common_frames):
        if frame_idx % progress_step == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")
        
        # Get frame data from all cameras at once
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
            
        # Loop through all detected markers
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
                except:
                    valid_marker = False
                    break
                    
            # Calculate column indices for this marker
            col_start = 1 + (marker - 1) * 3  # x, y, z columns for this marker
            
            if not valid_marker or len(pixel_obs_list) != len(dlt_params_list):
                # NaN values already pre-allocated, so skip
                pass
            else:
                point3d = rec3d_multicam(dlt_params_list, pixel_obs_list)
                reconstruction_array[frame_idx, col_start:col_start+3] = point3d
    
    print("3D reconstruction completed!")

    # Prepare header for all markers
    header = ["frame"]
    for marker in range(1, num_markers + 1):
        header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])

    # Convert pre-allocated array to DataFrame
    rec3d_df = pd.DataFrame(reconstruction_array, columns=header)
    
    # Remove frames that were skipped (all NaN except frame number)
    valid_frames_mask = ~rec3d_df.iloc[:, 1:].isna().all(axis=1)
    rec3d_df = rec3d_df[valid_frames_mask].reset_index(drop=True)
    
    if rec3d_df.empty:
        messagebox.showerror("Error", "No valid 3D reconstruction could be performed!")
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(output_directory, f"vaila_rec3d_{timestamp}")
    os.makedirs(new_dir, exist_ok=True)

    file_base = f"rec3d_{timestamp}"
    file_3d_path = os.path.join(new_dir, f"{file_base}.3d")
    file_csv_path = os.path.join(new_dir, f"{file_base}.csv")

    print("Saving 3D reconstruction results...")
    rec3d_df.to_csv(file_3d_path, index=False, float_format="%.6f")
    rec3d_df.to_csv(file_csv_path, index=False, float_format="%.6f")

    # Adjust column names to the expected format (e.g., convert "p1_x" to "p1_X")
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

    # Set conversion factors
    m_conversion = 1  # Conversion factor for meters
    mm_conversion = 1000  # Conversion factor for millimeters

    import vaila.readcsv_export as readcsv_export

    # Save C3D file in meters
    c3d_output_path_m = os.path.join(new_dir, f"{file_base}_m.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_m,
            point_rate=point_rate,
            conversion_factor=m_conversion,
        )
        messagebox.showinfo(
            "Success", f"C3D file (meters) saved at:\n{c3d_output_path_m}"
        )
        print("C3D file (meters) created successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save C3D file (meters): {e}")
        print("Error saving C3D file (meters):", e)

    # Save C3D file in millimeters
    c3d_output_path_mm = os.path.join(new_dir, f"{file_base}_mm.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_mm,
            point_rate=point_rate,
            conversion_factor=mm_conversion,
        )
        messagebox.showinfo(
            "Success", f"C3D file (millimeters) saved at:\n{c3d_output_path_mm}"
        )
        print("C3D file (millimeters) created successfully")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save C3D file (millimeters): {e}")
        print("Error saving C3D file (millimeters):", e)

    # Final summary message
    print(f"\n=== Processing Complete ===")
    print(f"Processed {len(common_frames)} frames with {num_markers} markers")
    print(f"Output directory: {new_dir}")
    print(f"Files created:")
    print(f"  - {file_base}.csv (CSV format)")
    print(f"  - {file_base}.3d (3D format)")
    print(f"  - {file_base}_m.c3d (C3D in meters)")
    print(f"  - {file_base}_mm.c3d (C3D in millimeters)")
    
    messagebox.showinfo(
        "Processing Complete",
        f"3D reconstruction completed successfully!\n\n"
        f"Processed: {len(common_frames)} frames with {num_markers} markers\n"
        f"Output directory: {os.path.basename(new_dir)}\n\n"
        f"Files created:\n"
        f"• CSV and 3D format files\n"
        f"• C3D files (meters and millimeters)"
    )

    root.destroy()


if __name__ == "__main__":
    run_rec3d_one_dlt3d()
