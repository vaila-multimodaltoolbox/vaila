"""
Script Name: dlt3d.py
Version: v0.01
Date and Time: 2024-08-13
Author: Prof. PhD. Paulo Santiago
Email: vailamultimodaltoolbox@gmail.com
Description: This script performs 3D calibration using the Direct Linear 
             Transformation (DLT) method. It calibrates based on correspondences 
             between 2D coordinates in multiple views and their corresponding 
             3D coordinates.

Dependencies:
    - Python 3.11.8
    - numpy
    - pandas
    - tkinter
    - rich
    - csv

    
Case DLT 3D Calibration not correctly performed, try
def dlt_calib(cp3d, cp2d):
    '''Calibration DLT... [rest of your docstring]'''
    '''Calibration DLT
    =============================================================================
    DLT 3D
    Calcula os parametros do DLT
    para executá-la, digite os comandos abaixos
    import rec3d
    DLT = rec3d.dlt_calib(cp3d, cd2d)
    onde:
    DLT  = vetor linha com os parametros do DLT calculados
    [L1,L2,L3...L11]
    cp3d = matriz retangular com as coordenadas 3d (X, Y, Z) dos pontos (p) do calibrador
    Xp1 Yp1 Zp1
    Xp2 Yp2 Zp2
    Xp3 Yp3 Zp3
    .   .   .
    .   .   .
    Xpn Ypn Zpn
    cp2d = matriz retangular com as coordenadas de tela (X, Y) dos pontos (p) do calibrador
    xp1 yp1
    xp2 yp2
    xp3 yp3
    .   .
    .   .
    xpn ypn
    =============================================================================
    '''
    cp3d = np.asarray(cp3d)
    if np.size(cp3d, 1) > 3:
        cp3d = cp3d[:, 1:]

    m = np.size(cp3d[:, 0], 0)
    M = np.zeros([m * 2, 11])
    N = np.zeros([m * 2, 1])

    cp2d = np.asarray(cp2d)

    for i in range(m):
        M[i*2, :] = [cp3d[i, 0], cp3d[i, 1], cp3d[i, 2], 1, 0, 0, 0, 0, -cp2d[i, 0]
                     * cp3d[i, 0], -cp2d[i, 0] * cp3d[i, 1], -cp2d[i, 0] * cp3d[i, 2]]

        M[i*2+1, :] = [0, 0, 0, 0, cp3d[i, 0], cp3d[i, 1], cp3d[i, 2], 1, -
                       cp2d[i, 1] * cp3d[i, 0], -cp2d[i, 1] * cp3d[i, 1], -cp2d[i, 1] * cp3d[i, 2]]

        N[[i*2, i*2+1], 0] = cp2d[i, :]

    Mt = M.T
    M1 = inv(Mt.dot(M))
    MN = Mt.dot(N)

    DLT = (M1).dot(MN).T

    return [DLT.tolist()]
"""

import os
import numpy as np
import pandas as pd
import csv  # Importando a biblioteca csv
from numpy.linalg import inv
from tkinter import filedialog, Tk, messagebox
from rich import print


def read_coordinates(file_path, usecols):
    df = pd.read_csv(file_path, usecols=usecols)
    coordinates = df.to_numpy()  # Mantém todas as linhas, incluindo aquelas com NaNs
    return coordinates


def dlt_calib(cp3d, cp2d):
    """
    Perform DLT 3D calibration.

    Parameters:
    - cp3d: Real 3D coordinates (X, Y, Z) of the calibration points.
    - cp2d: Corresponding 2D coordinates from the camera views.

    Returns:
    - DLT parameters as a list [L1, L2, ..., L11].
    """
    cp3d = np.asarray(cp3d)
    if np.size(cp3d, 1) > 3:
        cp3d = cp3d[:, 1:]

    m = np.size(cp3d[:, 0], 0)
    M = np.zeros([m * 2, 11])
    N = np.zeros([m * 2, 1])

    cp2d = np.asarray(cp2d)

    for i in range(m):
        M[i * 2, :] = [
            cp3d[i, 0],
            cp3d[i, 1],
            cp3d[i, 2],
            1,
            0,
            0,
            0,
            0,
            -cp2d[i, 0] * cp3d[i, 0],
            -cp2d[i, 0] * cp3d[i, 1],
            -cp2d[i, 0] * cp3d[i, 2],
        ]

        M[i * 2 + 1, :] = [
            0,
            0,
            0,
            0,
            cp3d[i, 0],
            cp3d[i, 1],
            cp3d[i, 2],
            1,
            -cp2d[i, 1] * cp3d[i, 0],
            -cp2d[i, 1] * cp3d[i, 1],
            -cp2d[i, 1] * cp3d[i, 2],
        ]

        N[[i * 2, i * 2 + 1], 0] = cp2d[i, :]

    Mt = M.T
    M1 = inv(Mt.dot(M))
    MN = Mt.dot(N)

    DLT = (M1).dot(MN).T

    return DLT.flatten()


def create_ref3d_template(pixel_file):
    df = pd.read_csv(pixel_file)
    template = df.copy()
    template.iloc[:, 1:] = (
        np.nan
    )  # Limpa todos os dados de coordenadas, mas mantém os números dos frames
    template_file = os.path.splitext(pixel_file)[0] + ".ref3d"
    template.to_csv(template_file, index=False)
    return template_file


def main():
    root = Tk()
    root.withdraw()

    pixel_file = filedialog.askopenfilename(
        title="Select the PIXEL coordinate file that will be used for calibration.",
        filetypes=[("CSV files", "*.csv")],
    )
    if not pixel_file:
        print("Pixel file selection cancelled.")
        return

    create_ref = messagebox.askyesno(
        "Create REF3D File",
        "Do you want to create a REF3D file based on the pixel file?",
    )
    if create_ref:
        real_file = create_ref3d_template(pixel_file)
        messagebox.showinfo("Success", f"Template REF3D file created: {real_file}")
        print(f"Template REF3D file created: {real_file}")
        print(
            "Please edit the REF3D file with real coordinates and run the DLT process again."
        )
        return
    else:
        real_file = filedialog.askopenfilename(
            title="Select Real 3D Coordinates File",
            filetypes=[("REF3D files", "*.ref3d")],
        )
        if not real_file:
            print("Real file selection cancelled.")
            return

    pixel_coords = read_coordinates(pixel_file, usecols=lambda x: x != "frame")
    real_coords = read_coordinates(real_file, usecols=lambda x: x != "frame")

    if pixel_coords.shape[1] // 2 != real_coords.shape[0]:
        print("The number of 2D views and 3D coordinate sets must match.")
        return

    dlt_params = []
    for i in range(len(pixel_coords)):
        if not np.isnan(pixel_coords[i]).any() and not np.isnan(real_coords[i]).any():
            L = pixel_coords[i].reshape(-1, 2)
            F = real_coords[i].reshape(-1, 3)
            if F.shape[0] >= 6:  # Necessita pelo menos 6 pontos para DLT3D
                print(f"Frame {i}: F = {F}, L = {L}")
                dlt_params.append((i, dlt_calib(F, L)))
            else:
                dlt_params.append((i, [np.nan] * 11))
        else:
            dlt_params.append((i, [np.nan] * 11))

    output_file = os.path.splitext(pixel_file)[0] + ".dlt3d"
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["frame"] + [f"dlt_param_{j}" for j in range(1, 12)])
        for frame, params in dlt_params:
            csvwriter.writerow([frame] + list(params))

    messagebox.showinfo("Success", f"DLT parameters saved to {output_file}")
    print(f"DLT parameters saved to {output_file}")


if __name__ == "__main__":
    main()
