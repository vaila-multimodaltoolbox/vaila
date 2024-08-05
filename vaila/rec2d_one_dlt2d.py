import numpy as np
import pandas as pd
from numpy.linalg import inv
from tkinter import filedialog, Tk, messagebox

def read_coordinates(file_path, usecols=None):
    df = pd.read_csv(file_path, usecols=usecols)
    coordinates = df.dropna().to_numpy()
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

    dlt_params = read_coordinates(dlt_file)
    pixel_coords_df = pd.read_csv(pixel_file)

    if dlt_params.shape[0] != 1:
        print("DLT file should contain only one set of DLT parameters.")
        return

    A = dlt_params[0]

    rec_coords = []
    for i, row in pixel_coords_df.iterrows():
        pixel_coords = row[1:].to_numpy().reshape(-1, 2)
        if not np.isnan(pixel_coords).all():
            rec2d_coords = rec2d(A, pixel_coords)
            rec_coords.append((row[0], *rec2d_coords.flatten()))
        else:
            rec_coords.append((row[0], *[np.nan] * (len(row) - 1)))

    rec_coords_df = pd.DataFrame(rec_coords, columns=pixel_coords_df.columns)

    output_file = pixel_file.replace(".csv", ".2d")
    rec_coords_df.to_csv(output_file, index=False, float_format='%.6f')

    messagebox.showinfo("Success", f"Reconstructed 2D coordinates saved to {output_file}")
    print(f"Reconstructed 2D coordinates saved to {output_file}")

if __name__ == "__main__":
    main()
