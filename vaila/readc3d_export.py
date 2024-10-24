"""
File: readc3d_export.py

Description:
This script processes .c3d files, extracting marker data, analog data, events, and points residuals,
and saves them into CSV files. It also allows the option to save the data in Excel format.
The script leverages Dask for efficient data handling and processing, particularly useful
when working with large datasets.

Features:
- Extracts and saves marker data with time columns.
- Extracts and saves analog data with time columns, including their units.
- Extracts and saves events with their labels and times.
- Extracts and saves points residuals with time columns.
- Supports saving the data in CSV format.
- Optionally saves the data in Excel format (can be slow for large files).
- Generates an info file containing metadata about markers, analogs, and their units.
- Generates a simplified short info file with key parameters and headers.
- Handles encoding errors to avoid crashes due to unexpected characters.

Dependencies:
- Python 3.x
- ezc3d
- Pandas
- Tkinter
- Tqdm
- Numpy
- Openpyxl (optional, for saving Excel files)

Version: 1.8
Date: October 2024
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


def save_info_file(datac3d, file_name, output_dir):
    """
    Save all parameters and data from the C3D file into a detailed .info text file.
    """
    print(f"Saving all data to .info file for {file_name}")

    info_file_path = os.path.join(output_dir, f"{file_name}.info")
    # Use encoding='utf-8' and ignore errors
    with open(info_file_path, "w", encoding="utf-8", errors="ignore") as info_file:
        # Write header information
        info_file.write(f"File: {file_name}\n")
        info_file.write("--- Parameters in C3D File ---\n\n")

        # Iterate over all groups in parameters and write them to the .info file
        for group_name, group_content in datac3d["parameters"].items():
            info_file.write(f"Group: {group_name}\n")
            for param_name, param_content in group_content.items():
                info_file.write(f"  Parameter: {param_name}\n")
                info_file.write(
                    f"    Description: {param_content.get('description', 'No description')}\n"
                )
                info_file.write(
                    f"    Value: {param_content.get('value', 'No value')}\n"
                )
                info_file.write(f"    Type: {param_content.get('type', 'No type')}\n")
                info_file.write(
                    f"    Dimension: {param_content.get('dimension', 'No dimension')}\n"
                )
                info_file.write("\n")

    print(f".info file saved at: {info_file_path}")

def save_short_info_file(
    marker_labels,
    marker_freq,
    analog_labels,
    analog_units,
    analog_freq,
    dir_name,
    file_name,
):
    """
    Save a simplified version of the info file with only the main parameters and headers.
    """
    print(f"Saving short info file for {file_name}")
    short_info_file_path = os.path.join(dir_name, f"{file_name}_short.info")
    # Use encoding='utf-8' and ignore errors
    with open(short_info_file_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(f"Marker frequency: {marker_freq} Hz\n")
        f.write(f"Analog frequency: {analog_freq} Hz\n\n")
        f.write("Marker labels:\n")
        for label in marker_labels:
            f.write(f"{label}\n")
        f.write("\nAnalog labels and units:\n")
        for label, unit in zip(analog_labels, analog_units):
            f.write(f"{label} ({unit})\n")

    print(f"Short info file saved at: {short_info_file_path}")

def save_events(datac3d, file_name, output_dir):
    """
    Save events data from the C3D file into a CSV file, including the frame number.
    """
    print(f"Saving events for {file_name}")

    events = datac3d["parameters"]["EVENT"]["CONTEXTS"]["value"]
    event_labels = datac3d["parameters"]["EVENT"]["LABELS"]["value"]
    event_times = datac3d["parameters"]["EVENT"]["TIMES"]["value"][1, :]
    event_contexts = datac3d["parameters"]["EVENT"]["CONTEXTS"]["value"]
    marker_freq = datac3d["header"]["points"]["frame_rate"]

    events_data = []
    for context, label, time in zip(event_contexts, event_labels, event_times):
        frame = int(round(time * marker_freq))
        events_data.append({"Context": context, "Label": label, "Time": time, "Frame": frame})

    # Save to a CSV file
    events_df = pd.DataFrame(events_data)
    events_file_path = os.path.join(output_dir, f"{file_name}_events.csv")
    events_df.to_csv(events_file_path, index=False)
    print(f"Events CSV saved at: {events_file_path}")


def importc3d(dat):
    """
    Import C3D file data and parameters.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    datac3d = c3d(dat)
    print(f"\nProcessing file: {dat}")
    print(f'Number of markers = {datac3d["parameters"]["POINT"]["USED"]["value"][0]}')

    point_data = datac3d["data"]["points"]
    points_residuals = datac3d["data"]["meta_points"]["residuals"]
    analogs = datac3d["data"]["analogs"]
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
    analog_units = (
        datac3d["parameters"]["ANALOG"]
        .get("UNITS", {})
        .get("value", ["Unknown"] * len(analog_labels))
    )

    # Check if there are point data
    if datac3d["parameters"]["POINT"]["USED"]["value"][0] > 0:
        markers = point_data[0:3, :, :].T.reshape(-1, len(marker_labels) * 3)
    else:
        markers = np.array([])  # Use an empty NumPy array if no markers

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
        analog_units,
        analog_freq,
        datac3d,
    )


def save_empty_file(file_path):
    """
    Save an empty CSV file.
    """
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
    analog_units,
    analog_freq,
    file_name,
    output_dir,
    save_excel,
    datac3d,
):
    """
    Save extracted data to CSV files and all parameters to .info files.
    """
    print(f"Saving data to files for {file_name}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.join(output_dir, "vaila_c3d_to_csv", f"{file_name}_{timestamp}")
    os.makedirs(dir_name, exist_ok=True)
    print(f"Directory created: {dir_name}")

    # Save all data to detailed .info file
    save_info_file(datac3d, file_name, dir_name)
    # Save simplified short .info file
    save_short_info_file(
        marker_labels,
        marker_freq,
        analog_labels,
        analog_units,
        analog_freq,
        dir_name,
        file_name,
    )
    # Save events data
    save_events(datac3d, file_name, dir_name)

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

    print(f"Files for {file_name} saved successfully!")


def convert_c3d_to_csv():
    """
    Main function to convert C3D files to CSV and .info files.
    """
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

    root.destroy()  # Use root.destroy() to properly close the Tkinter resources

    if input_directory and output_directory:
        c3d_files = sorted(
            [f for f in os.listdir(input_directory) if f.endswith(".c3d")]
        )
        print(f"Found {len(c3d_files)} .c3d files in the input directory.")

        # Simplified progress bar
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
                    analog_units,
                    analog_freq,
                    datac3d,
                ) = importc3d(file_path)
                file_name = os.path.splitext(c3d_file)[0]
                save_to_files(
                    markers,
                    marker_labels,
                    marker_freq,
                    analogs,
                    points_residuals,
                    analog_labels,
                    analog_units,
                    analog_freq,
                    file_name,
                    output_directory,
                    save_excel,
                    datac3d,
                )
            except Exception as e:
                print(f"Error processing {c3d_file}: {e}")
                messagebox.showerror("Error", f"Failed to process {c3d_file}: {e}")

            # Update progress bar after each file
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
