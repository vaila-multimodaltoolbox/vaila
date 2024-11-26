"""
================================================================================
IMU Analysis Tool - imu_analysis.py
================================================================================
Author: Prof. Ph.D. Paulo Santiago (paulosantiago@usp.br)
Date: 2024-11-21
Version: 1.2

Description:
------------
This script performs IMU sensor data analysis from CSV and C3D files. 
It processes accelerometer and gyroscope data, calculates tilt angles and Euler angles,
and generates graphs and CSV files with the processed results. File names for outputs 
include the prefix of the processed file.

Features:
- Support for CSV and C3D files.
- Processing of accelerometer and gyroscope data.
- Calculation of tilt angles and Euler angles.
- Saving graphs in PNG format with file-specific prefixes.
- Exporting processed data to uniquely named CSV files.
- Graphical interface for selecting directories and headers.
- Automatic processing of default headers if no selection is made.

Requirements:
- Python 3.x
- Libraries: numpy, pandas, imufusion, matplotlib, tkinter, ezc3d, rich
- Custom modules: filtering, readcsv, dialogsuser

================================================================================
"""

import os
import numpy as np
import pandas as pd
import imufusion
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import messagebox, filedialog, Tk
import ezc3d
from .filtering import apply_filter
from .readcsv import select_headers_gui, get_csv_headers
from .dialogsuser import get_user_inputs
from rich import print


def importc3d(file_path):
    """Reads C3D file and extracts markers, analog data, and other metadata."""
    datac3d = ezc3d.c3d(file_path)
    print(f"\nProcessing file: {file_path}")
    print(f'Number of markers: {datac3d["parameters"]["POINT"]["USED"]["value"][0]}')

    point_data = datac3d["data"]["points"]
    analogs = datac3d["data"]["analogs"]
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
    marker_freq = datac3d["header"]["points"]["frame_rate"]
    analog_freq = datac3d["header"]["analogs"]["frame_rate"]

    markers = point_data[0:3, :, :].T.reshape(-1, len(marker_labels) * 3)

    return (
        markers,
        marker_labels,
        marker_freq,
        analogs,
        None,  # Placeholder for points residuals (not used)
        analog_labels,
        analog_freq,
    )


def imu_orientations(accelerometer, gyroscope, time, sample_rate, sensor_name):
    """Process accelerometer and gyroscope data to compute orientations and plot results."""
    tilt_x_rad = np.arctan2(
        accelerometer[:, 0],
        np.sqrt(accelerometer[:, 1] ** 2 + accelerometer[:, 2] ** 2),
    )
    tilt_y_rad = np.arctan2(
        accelerometer[:, 1],
        np.sqrt(accelerometer[:, 0] ** 2 + accelerometer[:, 2] ** 2),
    )
    tilt_z_rad = np.arctan2(
        np.sqrt(accelerometer[:, 0] ** 2 + accelerometer[:, 1] ** 2),
        accelerometer[:, 2],
    )

    tilt_deg = np.stack(
        (np.degrees(tilt_x_rad), np.degrees(tilt_y_rad), np.degrees(tilt_z_rad)),
        axis=-1,
    )

    ahrs = imufusion.Ahrs()
    euler = np.empty((len(time), 3))
    quaternions = np.empty((len(time), 4))

    for index in range(len(time)):
        ahrs.update_no_magnetometer(
            gyroscope[index], accelerometer[index], (1 / sample_rate)
        )
        euler[index] = ahrs.quaternion.to_euler()
        quaternions[index] = [
            ahrs.quaternion.w,
            ahrs.quaternion.x,
            ahrs.quaternion.y,
            ahrs.quaternion.z,
        ]

    return tilt_deg, euler, quaternions


def plot_and_save_graphs(
    time, data, labels, title, xlabel, ylabel, sensor_name, save_path, file_prefix
):
    colors = ["red", "green", "blue"]
    plt.figure(figsize=(10, 6))
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(time, data[:, i], label=label, color=color)
    plt.title(f"{sensor_name} {title}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    save_file_path = os.path.join(
        save_path, f"{file_prefix}_{sensor_name}_{title.replace(' ', '_')}.png"
    )
    plt.savefig(save_file_path)
    plt.close()
    print(f"Saved plot: {save_file_path}")


def save_results_to_csv(
    base_dir,
    time,
    gyroscope,
    accelerometer,
    euler,
    tilt_deg,
    quaternions,
    sensor_name,
    file_prefix,
):
    results = {
        "Time": time,
        "Gyro_X": gyroscope[:, 0],
        "Gyro_Y": gyroscope[:, 1],
        "Gyro_Z": gyroscope[:, 2],
        "Acc_X": accelerometer[:, 0],
        "Acc_Y": accelerometer[:, 1],
        "Acc_Z": accelerometer[:, 2],
        "Euler_X": euler[:, 0],
        "Euler_Y": euler[:, 1],
        "Euler_Z": euler[:, 2],
        "Tilt_X": tilt_deg[:, 0],
        "Tilt_Y": tilt_deg[:, 1],
        "Tilt_Z": tilt_deg[:, 2],
        "Quat_W": quaternions[:, 0],
        "Quat_X": quaternions[:, 1],
        "Quat_Y": quaternions[:, 2],
        "Quat_Z": quaternions[:, 3],
    }

    df = pd.DataFrame(results)
    csv_path = os.path.join(base_dir, f"{file_prefix}_{sensor_name}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")


def plot_and_save_sensor_data(
    time,
    gyroscope,
    accelerometer,
    euler,
    tilt_deg,
    quaternions,
    sensor_name,
    base_dir_figures,
    file_prefix,
):
    plot_and_save_graphs(
        time,
        accelerometer,
        ["Acc X", "Acc Y", "Acc Z"],
        "Accelerometer Data",
        "Time (s)",
        "Acceleration (g)",
        sensor_name,
        base_dir_figures,
        file_prefix,
    )
    plot_and_save_graphs(
        time,
        gyroscope,
        ["Gyro X", "Gyro Y", "Gyro Z"],
        "Gyroscope Data",
        "Time (s)",
        "Gyroscope (°/s)",
        sensor_name,
        base_dir_figures,
        file_prefix,
    )
    plot_and_save_graphs(
        time,
        tilt_deg,
        ["Tilt X", "Tilt Y", "Tilt Z"],
        "Tilt Angles",
        "Time (s)",
        "Tilt (°)",
        sensor_name,
        base_dir_figures,
        file_prefix,
    )
    plot_and_save_graphs(
        time,
        euler,
        ["Euler X", "Euler Y", "Euler Z"],
        "Euler Angles",
        "Time (s)",
        "Euler (°)",
        sensor_name,
        base_dir_figures,
        file_prefix,
    )


def analyze_imu_data():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = Tk()
    root.withdraw()

    user_inputs = get_user_inputs()
    file_type = user_inputs.get("file_type")
    sample_rate = user_inputs.get("sample_rate")

    if file_type not in ["csv", "c3d"]:
        messagebox.showerror("Error", "Invalid file type selected.")
        return

    if sample_rate is None or sample_rate <= 0:
        messagebox.showerror("Error", "Invalid sample rate provided.")
        return

    directory_path = filedialog.askdirectory(title="Select directory to read files")
    if not directory_path:
        messagebox.showerror("Error", "No input directory selected.")
        return

    output_directory = filedialog.askdirectory(
        title="Choose directory to save analysis"
    )
    if not output_directory:
        messagebox.showerror("Error", "No output directory selected.")
        return

    select_headers_for_all = messagebox.askyesno(
        "Header Selection", "Select headers for all files?"
    )

    selected_headers = None
    if select_headers_for_all:
        selected_file = filedialog.askopenfilename(
            title="Pick file to select headers",
            filetypes=[("CSV files", "*.csv"), ("C3D files", "*.c3d")],
        )
        if not selected_file:
            messagebox.showerror("Error", "No file selected to choose headers.")
            return
        if selected_file.endswith(".csv"):
            headers = get_csv_headers(selected_file)
            selected_headers = select_headers_gui(headers)
        elif selected_file.endswith(".c3d"):
            _, _, _, analogs, _, analog_labels, analog_freq = importc3d(selected_file)
            df_analogs = pd.DataFrame(
                analogs.reshape(
                    analogs.shape[0] * analogs.shape[1], analogs.shape[2]
                ).T,
                columns=analog_labels,
            )
            selected_headers = select_headers_gui(df_analogs.columns)

    filter_method = "fir"
    gravity = 9.81

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if not os.path.isfile(file_path):
            continue

        file_name_without_extension, _ = os.path.splitext(file_name)
        file_prefix = file_name_without_extension.replace(" ", "_").replace("-", "_")

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir_figures = os.path.join(
            output_directory,
            "IMU_multimodal_results",
            f"figure/IMU_{file_prefix}_{current_time}",
        )
        base_dir_processed_data = os.path.join(
            output_directory,
            "IMU_multimodal_results",
            f"processed_data/IMU_{file_prefix}_{current_time}",
        )
        os.makedirs(base_dir_figures, exist_ok=True)
        os.makedirs(base_dir_processed_data, exist_ok=True)

        if file_type == "csv" and file_name.endswith(".csv"):
            try:
                data = pd.read_csv(file_path)
                if selected_headers:
                    data = data[selected_headers]
                accelerometer = data.iloc[:, [0, 2, 1]].values / gravity
                gyroscope = data.iloc[:, [3, 5, 4]].values * (180 / np.pi)

                time = np.linspace(
                    0, (len(accelerometer) - 1) / sample_rate, len(accelerometer)
                )
                tilt, euler, quaternions = imu_orientations(
                    accelerometer, gyroscope, time, sample_rate, "Sensor"
                )

                plot_and_save_sensor_data(
                    time,
                    gyroscope,
                    accelerometer,
                    euler,
                    tilt,
                    quaternions,
                    "Sensor",
                    base_dir_figures,
                    file_prefix,
                )

                save_results_to_csv(
                    base_dir_processed_data,
                    time,
                    gyroscope,
                    accelerometer,
                    euler,
                    tilt,
                    quaternions,
                    "Sensor",
                    file_prefix,
                )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred while processing {file_name}: {e}"
                )
        elif file_type == "c3d" and file_name.endswith(".c3d"):
            try:
                _, _, _, analogs, _, analog_labels, analog_freq = importc3d(file_path)
                df_analogs = pd.DataFrame(
                    analogs.reshape(
                        analogs.shape[0] * analogs.shape[1], analogs.shape[2]
                    ).T,
                    columns=analog_labels,
                )
                if selected_headers:
                    data = df_analogs[selected_headers]
                else:
                    data = df_analogs.iloc[:, :18]  # Default to first 18 channels

                accelerometer = data.iloc[:, [0, 2, 1]].values / gravity
                gyroscope = data.iloc[:, [3, 5, 4]].values * (180 / np.pi)

                time = np.linspace(
                    0, (len(accelerometer) - 1) / analog_freq, len(accelerometer)
                )
                tilt, euler, quaternions = imu_orientations(
                    accelerometer, gyroscope, time, analog_freq, "Sensor"
                )

                plot_and_save_sensor_data(
                    time,
                    gyroscope,
                    accelerometer,
                    euler,
                    tilt,
                    quaternions,
                    "Sensor",
                    base_dir_figures,
                    file_prefix,
                )

                save_results_to_csv(
                    base_dir_processed_data,
                    time,
                    gyroscope,
                    accelerometer,
                    euler,
                    tilt,
                    quaternions,
                    "Sensor",
                    file_prefix,
                )
            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred while processing {file_name}: {e}"
                )

    root.destroy()
    messagebox.showinfo(
        "Success", "All files have been processed and saved successfully."
    )


if __name__ == "__main__":
    analyze_imu_data()
