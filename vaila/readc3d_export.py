"""
File: readc3d_export.py

Description:
This script processes .c3d files, extracting marker data, analog data, and points residuals,
and saves them into CSV files. It also allows the option to save the data in Excel format.
The script leverages Dask for efficient data handling and processing, particularly useful
when working with large datasets.

Features:
- Extracts and saves marker data with time columns.
- Extracts and saves analog data with time columns.
- Extracts and saves points residuals with time columns.
- Supports saving the data in CSV format.
- Optionally saves the data in Excel format (can be slow for large files).
- Generates an info file containing metadata about markers and analogs.

Dependencies:
- Python 3.x
- ezc3d
- Dask
- Pandas
- Tkinter
- Tqdm
- Openpyxl (optional, for saving Excel files)

Version: 1.0
Date: August 2024
Author: Prof. Paulo Santiago

Usage:
- Run the script, select the input directory containing .c3d files, and specify an output directory.
- Choose whether to save the files in Excel format.
- The script will process each .c3d file in the input directory and save the results in the specified output directory.

Example:
$ python readc3d_export.py

Notes:
- Ensure that all necessary libraries are installed.
- This script is designed to handle large datasets efficiently, but saving to Excel format may take significant time depending on the dataset size.
"""

import os
import pandas as pd
from ezc3d import c3d
from datetime import datetime
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm
import numpy as np


def importc3d(dat):
    datac3d = c3d(dat)
    print(f"\nProcessing file: {dat}")
    print(f'Number of markers = {datac3d["parameters"]["POINT"]["USED"]["value"][0]}')

    point_data = datac3d["data"]["points"]
    points_residuals = datac3d["data"]["meta_points"]["residuals"]
    analogs = datac3d["data"]["analogs"]
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]

    # Verifica se há dados de pontos
    if datac3d["parameters"]["POINT"]["USED"]["value"][0] > 0:
        markers = point_data[0:3, :, :].T.reshape(-1, len(marker_labels) * 3)
    else:
        markers = np.array([])  # Usa um array vazio do NumPy

    marker_freq = datac3d["header"]["points"]["frame_rate"]
    analog_freq = datac3d["header"]["analogs"]["frame_rate"]

    # Print summary information
    num_analog_channels = datac3d["parameters"]["ANALOG"]["USED"]["value"][0]
    print(f"Number of marker labels = {len(marker_labels)}")
    print(f"Number of analog channels = {num_analog_channels}")
    print(f"Marker frequency = {marker_freq} Hz")
    print(f"Analog frequency = {analog_freq} Hz")

    return (
        markers,
        marker_labels,
        marker_freq,
        analogs,
        points_residuals,
        analog_labels,
        analog_freq,
    )


def save_info_file(
    marker_labels, marker_freq, analog_labels, analog_freq, dir_name, file_name
):
    print(f"Saving info file for {file_name}")
    info_file_path = os.path.join(dir_name, f"{file_name}.info")
    with open(info_file_path, "w") as f:
        f.write(f"Marker frequency: {marker_freq} Hz\n")
        f.write(f"Analog frequency: {analog_freq} Hz\n\n")
        f.write("Marker labels:\n")
        for label in marker_labels:
            f.write(f"{label}\n")
        f.write("\nAnalog labels:\n")
        for label in analog_labels:
            f.write(f"{label}\n")


def save_empty_file(file_path):
    # Save an empty CSV file
    print(f"Saving empty file: {file_path}")
    with open(file_path, "w") as f:
        f.write("")


def save_to_files(
    markers,
    marker_labels,
    marker_freq,
    analogs,
    points_residuals,
    analog_labels,
    analog_freq,
    file_name,
    output_dir,
    save_excel,
):
    print(f"Saving data to files for {file_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.join(
        output_dir, "multimodal_c3d_to_csv", f"{file_name}_{timestamp}"
    )
    os.makedirs(dir_name, exist_ok=True)
    print(f"Directory created: {dir_name}")

    # Prepare marker columns
    marker_columns = [
        f"{label}_{axis}" for label in marker_labels for axis in ["X", "Y", "Z"]
    ]

    # Save markers data
    if markers.size > 0:
        markers_df = pd.DataFrame(markers, columns=marker_columns)
        markers_df.insert(
            0,
            "Time",
            pd.Series(
                [f"{i / marker_freq:.3f}" for i in range(markers_df.shape[0])],
                name="Time",
            ),
        )
        print(f"Saving markers CSV for {file_name}")
        markers_df.to_csv(
            os.path.join(dir_name, f"{file_name}_markers.csv"), index=False
        )
    else:
        print(f"No markers found for {file_name}, saving empty file.")
        save_empty_file(os.path.join(dir_name, f"{file_name}_markers.csv"))

    # Save analog data
    if analogs.size > 0:
        analogs_df = pd.DataFrame(analogs.squeeze(axis=0).T, columns=analog_labels)
        analogs_df.insert(
            0,
            "Time",
            pd.Series(
                [f"{i / analog_freq:.3f}" for i in range(analogs_df.shape[0])],
                name="Time",
            ),
        )
        print(f"Saving analogs CSV for {file_name}")
        analogs_df.to_csv(
            os.path.join(dir_name, f"{file_name}_analogs.csv"), index=False
        )
    else:
        print(f"No analogs found for {file_name}, saving empty file.")
        save_empty_file(os.path.join(dir_name, f"{file_name}_analogs.csv"))

    # Save points residuals data
    if points_residuals.size > 0:
        points_residuals_df = pd.DataFrame(points_residuals.squeeze(axis=0).T)
        points_residuals_df.insert(
            0,
            "Time",
            pd.Series(
                [f"{i / marker_freq:.3f}" for i in range(points_residuals_df.shape[0])],
                name="Time",
            ),
        )
        print(f"Saving points residuals CSV for {file_name}")
        points_residuals_df.to_csv(
            os.path.join(dir_name, f"{file_name}_points_residuals.csv"), index=False
        )
    else:
        print(f"No points residuals found for {file_name}, saving empty file.")
        save_empty_file(os.path.join(dir_name, f"{file_name}_points_residuals.csv"))

    # Optionally save to Excel
    if save_excel:
        print("Saving to Excel. This process can take a long time...")
        with pd.ExcelWriter(
            os.path.join(dir_name, f"{file_name}.xlsx"), engine="openpyxl"
        ) as writer:
            if markers.size > 0:
                markers_df.to_excel(writer, sheet_name="Markers", index=False)
            if analogs.size > 0:
                analogs_df.to_excel(writer, sheet_name="Analogs", index=False)
            if points_residuals.size > 0:
                points_residuals_df.to_excel(
                    writer, sheet_name="Points Residuals", index=False
                )

    # Save the info file
    save_info_file(
        marker_labels, marker_freq, analog_labels, analog_freq, dir_name, file_name
    )

    print(f"Files for {file_name} saved successfully!")


def convert_c3d_to_csv():
    root = Tk()
    root.withdraw()

    save_excel = messagebox.askyesno(
        "Save as Excel",
        "Do you want to save the data as Excel files? This process can be very slow.",
    )
    print(f"Debug: save_excel = {save_excel}")

    input_directory = filedialog.askdirectory(title="Select Input Directory")
    print(f"Debug: input_directory = {input_directory}")

    output_directory = filedialog.askdirectory(title="Select Output Directory")
    print(f"Debug: output_directory = {output_directory}")

    if input_directory and output_directory:
        c3d_files = sorted(
            [f for f in os.listdir(input_directory) if f.endswith(".c3d")]
        )
        print(f"Found {len(c3d_files)} .c3d files in the input directory.")

        # Barra de progresso simplificada
        progress_bar = tqdm(
            total=len(c3d_files), desc="Processing C3D files", unit="file"
        )

        for c3d_file in c3d_files:
            print(f"Processing file: {c3d_file}")
            try:
                file_path = os.path.join(input_directory, c3d_file)
                (
                    markers,
                    marker_labels,
                    marker_freq,
                    analogs,
                    points_residuals,
                    analog_labels,
                    analog_freq,
                ) = importc3d(file_path)
                file_name = os.path.splitext(c3d_file)[0]
                save_to_files(
                    markers,
                    marker_labels,
                    marker_freq,
                    analogs,
                    points_residuals,
                    analog_labels,
                    analog_freq,
                    file_name,
                    output_directory,
                    save_excel,
                )
            except Exception as e:
                print(f"Error processing {c3d_file}: {e}")

            # Atualiza a barra de progresso após cada arquivo
            progress_bar.update(1)

        progress_bar.close()
        print("All files have been processed and saved successfully!")
        messagebox.showinfo(
            "Information", "C3D files conversion completed successfully!"
        )
    else:
        print("Input or output directory not selected.")
        messagebox.showwarning("Warning", "Input or output directory not selected.")


if __name__ == "__main__":
    convert_c3d_to_csv()
