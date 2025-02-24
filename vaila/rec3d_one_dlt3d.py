"""
rec3d_one_dlt3d.py
Author: Paulo Santiago
Version: 0.1.0
Last Updated: February 24, 2025

Description:
    Batch 3D reconstruction using the Direct Linear Transformation (DLT) method with multiple cameras.
    Each camera has a corresponding DLT3D parameter file (one set of 11 parameters per file) and a pixel coordinate CSV file.
    The pixel files are expected to use Vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,p25_x,p25_y
    For each common frame found among the pixel files, the script reconstructs 3D points for all 25 markers.
    The output file contains the 3D reconstructed coordinates (x,y,z) for each marker.
"""

import os
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
    A_matrix = []
    b_vector = []
    for i in range(num_cameras):
        A_params = dlt_list[i]
        x, y = pixel_list[i]
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = A_params
        # Equations for camera i:
        # (a1 - x*a9)*X + (a2 - x*a10)*Y + (a3 - x*a11)*Z = x - a4
        # (a5 - y*a9)*X + (a6 - y*a10)*Y + (a7 - y*a11)*Z = y - a8
        row1 = [a1 - x * a9, a2 - x * a10, a3 - x * a11]
        row2 = [a5 - y * a9, a6 - y * a10, a7 - y * a11]
        A_matrix.append(row1)
        A_matrix.append(row2)
        b_vector.append(x - a4)
        b_vector.append(y - a8)
    A_matrix = np.array(A_matrix)
    b_vector = np.array(b_vector)
    print("[DEBUG] Solving least squares system for 3D reconstruction...")
    solution, residuals, rank, s = lstsq(A_matrix, b_vector, rcond=None)
    return solution  # [X, Y, Z]

def run_rec3d_one_dlt3d():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = Tk()
    root.withdraw()
    
    print("[DEBUG] Prompting for DLT3D parameter files.")
    dlt_files = filedialog.askopenfilenames(
        title="Select DLT3D parameter files (one per camera)",
        filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")]
    )
    print("[DEBUG] Selected DLT files:", dlt_files)
    if not dlt_files:
        messagebox.showerror("Error", "No DLT3D files selected!")
        return
    
    print("[DEBUG] Prompting for pixel coordinate CSV files.")
    pixel_files = filedialog.askopenfilenames(
        title="Select pixel coordinate CSV files (one per camera)",
        filetypes=[("CSV files", "*.csv")]
    )
    print("[DEBUG] Selected pixel files:", pixel_files)
    if not pixel_files:
        messagebox.showerror("Error", "No pixel coordinate files selected!")
        return
    
    if len(dlt_files) != len(pixel_files):
        messagebox.showerror("Error", "The number of DLT3D files must match the number of pixel coordinate files!")
        return
    
    # Load DLT3D parameters for each camera (use first row from each file, skipping the frame column)
    dlt_params_list = []
    for file in dlt_files:
        print(f"[DEBUG] Processing DLT file: {file}")
        df = pd.read_csv(file)
        if df.empty:
            messagebox.showerror("Error", f"DLT3D file {os.path.basename(file)} is empty!")
            return
        params = df.iloc[0, 1:].to_numpy().astype(float)
        dlt_params_list.append(params)
        print(f"[DEBUG] DLT parameters from {os.path.basename(file)}:", params)
    
    # Load pixel coordinate data for each camera using Vailá's standard header
    pixel_dfs = []
    for file in pixel_files:
        print(f"[DEBUG] Processing pixel file: {file}")
        df = pd.read_csv(file)
        print(f"[DEBUG] Pixel data from {os.path.basename(file)} loaded with shape:", df.shape)
        required_cols = {"frame", "p1_x", "p1_y"}
        if not required_cols.issubset(df.columns):
            messagebox.showerror("Error", f"Pixel coordinate file {os.path.basename(file)} does not contain required columns 'frame', 'p1_x', and 'p1_y'.")
            return
        # If the file has only one row, force the frame to 0 for consistency across cameras.
        if df.shape[0] == 1:
            df["frame"] = 0
        pixel_dfs.append(df)
    
    # Debug: List unique frame values for each pixel file
    for idx, df in enumerate(pixel_dfs, start=1):
        frames = df["frame"].unique()
        print(f"[DEBUG] Pixel file {idx} ({os.path.basename(pixel_files[idx-1])}) frames:", frames)
    
    # Determine the common frames across all pixel files
    frame_sets = [set(df["frame"]) for df in pixel_dfs]
    common_frames = set.intersection(*frame_sets)
    print("[DEBUG] Common frames found:", common_frames)
    if not common_frames:
        messagebox.showerror("Error", "No common frames found among pixel files!")
        return
    common_frames = sorted(common_frames)
    
    # For each common frame, perform reconstruction for all 25 markers
    reconstruction_results = []
    for frame in common_frames:
        print(f"[DEBUG] Processing frame: {frame}")
        row_results = [frame]
        # Loop through markers 1 to 25
        for marker in range(1, 26):
            pixel_obs_list = []
            valid_marker = True
            for df in pixel_dfs:
                row = df[df["frame"] == frame]
                if row.empty:
                    valid_marker = False
                    break
                row = row.iloc[0]
                try:
                    x_obs = float(row[f"p{marker}_x"])
                    y_obs = float(row[f"p{marker}_y"])
                except Exception as e:
                    print(f"[DEBUG] Error parsing marker {marker} in frame {frame}: {e}")
                    valid_marker = False
                    break
                if np.isnan(x_obs) or np.isnan(y_obs):
                    valid_marker = False
                    break
                pixel_obs_list.append((x_obs, y_obs))
            if not valid_marker or len(pixel_obs_list) != len(dlt_params_list):
                print(f"[DEBUG] Marker {marker} in frame {frame} has invalid data. Appending NaNs.")
                row_results.extend([np.nan, np.nan, np.nan])
            else:
                point3d = rec3d_multicam(dlt_params_list, pixel_obs_list)
                print(f"[DEBUG] Reconstructed 3D point for marker {marker} in frame {frame}:", point3d)
                row_results.extend([point3d[0], point3d[1], point3d[2]])
        reconstruction_results.append(row_results)
    
    if not reconstruction_results:
        messagebox.showerror("Error", "No valid 3D reconstruction could be performed!")
        return
    
    # Prepara header: frame, depois p1_x, p1_y, p1_z, ..., p25_x, p25_y, p25_z
    header = ["frame"]
    for marker in range(1, 26):
        header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])
    
    rec3d_df = pd.DataFrame(reconstruction_results, columns=header)
    
    # --- NOVAS ALTERAÇÕES: SALVAR ARQUIVOS EM UM DIRETÓRIO ESCOLHIDO PELO USUÁRIO ---
    output_dir = filedialog.askdirectory(title="Selecione o diretório para salvar os arquivos")
    if not output_dir:
        messagebox.showerror("Erro", "Nenhum diretório selecionado. Operação cancelada.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(output_dir, f"vaila_rec3d_{timestamp}")
    os.makedirs(new_dir, exist_ok=True)
    
    file_base = f"rec3d_{timestamp}"
    file_3d_path = os.path.join(new_dir, f"{file_base}.3d")
    file_csv_path = os.path.join(new_dir, f"{file_base}.csv")
    
    print(f"[DEBUG] Salvando resultados 3D em: {file_3d_path} e {file_csv_path}")
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

    # Ask the user for the point data rate in Hz
    point_rate = simpledialog.askinteger("Point Rate", "Enter the point data rate (Hz):", minvalue=1, initialvalue=100)
    if point_rate is None:
        messagebox.showerror("Error", "Point data rate is not defined. Operation cancelled.")
        return

    # Set conversion factors
    m_conversion = 1       # Conversion factor for meters
    mm_conversion = 1000   # Conversion factor for millimeters

    import vaila.readcsv_export as readcsv_export

    # Save C3D file in meters
    c3d_output_path_m = os.path.join(new_dir, f"{file_base}_m.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_m,
            point_rate=point_rate,
            conversion_factor=m_conversion
        )
        messagebox.showinfo("Success", f"C3D file (meters) saved at:\n{c3d_output_path_m}")
        print("[DEBUG] C3D file (meters) saved at", c3d_output_path_m)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save C3D file (meters): {e}")
        print("[DEBUG] Error saving C3D file (meters):", e)

    # Save C3D file in millimeters
    c3d_output_path_mm = os.path.join(new_dir, f"{file_base}_mm.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_mm,
            point_rate=point_rate,
            conversion_factor=mm_conversion
        )
        messagebox.showinfo("Success", f"C3D file (millimeters) saved at:\n{c3d_output_path_mm}")
        print("[DEBUG] C3D file (millimeters) saved at", c3d_output_path_mm)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save C3D file (millimeters): {e}")
        print("[DEBUG] Error saving C3D file (millimeters):", e)
    
    root.destroy()

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
    # Trabalhamos com 25 marcadores; define os rótulos.
    marker_labels = [f"p{i}" for i in range(1, 26)]
    num_markers = len(marker_labels)
    
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
        filetypes=[("C3D files", "*.c3d")]
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

if __name__ == "__main__":
    run_rec3d_one_dlt3d()
