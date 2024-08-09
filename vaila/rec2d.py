# rec2d.py
# Author: Paulo Santiago
# Version: 0.0.3
# Last Updated: August 9, 2024
# Description: Reconstruct 2D coordinates for each frame using corresponding DLT parameters.
# --------------------------------------------------
# Usage Instructions:
# - Select the DLT parameters file (can contain multiple sets of DLT parameters).
# - Select the CSV file containing pixel coordinates.
# - The reconstructed 2D coordinates will be saved to a new file with the same name but with a .2d extension.
# - The script matches each frame in the pixel coordinates file with the corresponding DLT parameters based on frame number.
# --------------------------------------------------

import numpy as np
import pandas as pd
from numpy.linalg import inv
from tkinter import filedialog, Tk, messagebox
from rich import print

def read_coordinates(file_path, usecols):
    df = pd.read_csv(file_path, usecols=usecols)
    coordinates = df.to_numpy()  # Não descartar NaN aqui
    return coordinates

def rec2d(A, cc2d):
    nlin = np.size(cc2d, 0)
    H = np.matrix(np.zeros((nlin, 2)))
    for k in range(nlin):
        x = cc2d[k, 0]
        y = cc2d[k, 1]
        cc2d1 = np.matrix([[A[0] - x * A[6], A[1] - x * A[7]], 
                           [A[3] - y * A[6], A[4] - y * A[7]]])
        cc2d2 = np.matrix([[x - A[2]], 
                           [y - A[5]]])
        G1 = inv(cc2d1) * cc2d2
        H[k, :] = G1.transpose()
    return np.asarray(H)

def main():
    root = Tk()
    root.withdraw()

    dlt_file = filedialog.askopenfilename(title="Select DLT Parameters File", filetypes=[("DLT2D files", "*.dlt2d")])
    if not dlt_file:
        print("DLT file selection cancelled.")
        return

    pixel_file = filedialog.askopenfilename(title="Select Pixel Coordinates File", filetypes=[("CSV files", "*.csv")])
    if not pixel_file:
        print("Pixel file selection cancelled.")
        return

    dlt_params_df = pd.read_csv(dlt_file)
    pixel_coords_df = pd.read_csv(pixel_file)

    dlt_params = dlt_params_df.to_numpy()
    frames = dlt_params[:, 0]
    dlt_params = dlt_params[:, 1:]

    rec_coords = []
    for i, row in pixel_coords_df.iterrows():
        frame_num = int(row['frame'])
        if frame_num in frames:
            A_index = np.where(frames == frame_num)[0][0]
            A = dlt_params[A_index]
            if not np.isnan(A).any():
                pixel_coords = row[1:].to_numpy().reshape(-1, 2)
                rec2d_coords = rec2d(A, pixel_coords)
                rec_coords.append((frame_num, *rec2d_coords.flatten()))
            else:
                rec_coords.append((frame_num, *[np.nan] * (pixel_coords_df.shape[1] - 1)))
        else:
            rec_coords.append((frame_num, *[np.nan] * (pixel_coords_df.shape[1] - 1)))

    rec_coords_df = pd.DataFrame(rec_coords, columns=pixel_coords_df.columns)

    output_file = pixel_file.replace(".csv", ".2d")
    rec_coords_df.to_csv(output_file, index=False, float_format='%.6f')

    messagebox.showinfo("Success", f"Reconstructed 2D coordinates saved to {output_file}")
    print(f"Reconstructed 2D coordinates saved to {output_file}")

if __name__ == "__main__":
    main()
