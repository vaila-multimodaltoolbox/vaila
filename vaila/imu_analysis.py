"""
Nome: imu_analysis.py
Data: 2024-07-28
Versão: 1.0
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
from .readcsv import show_csv, select_headers_gui, get_csv_headers
from .dialogsuser import get_user_inputs
from rich import print


def imu_orientations(accelerometer, gyroscope, time, sample_rate, sensor_name):
    """Process accelerometer and gyroscope data to compute orientations and plot results."""

    # Calculate tilt angles in radians
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

    # Convert radians to degrees and combine into a single array
    tilt_deg = np.stack(
        (np.degrees(tilt_x_rad), np.degrees(tilt_y_rad), np.degrees(tilt_z_rad)),
        axis=-1,
    )

    # Initialize the AHRS algorithm
    ahrs = imufusion.Ahrs()
    euler = np.empty((len(time), 3))

    # Initialize array numpy to quaternions
    quaternions = np.empty((len(time), 4))  # Create a four-column array for w, x, y, z

    for index in range(len(time)):
        ahrs.update_no_magnetometer(
            gyroscope[index], accelerometer[index], (1 / sample_rate)
        )  # Assuming Hz sampling rate
        euler[index] = ahrs.quaternion.to_euler()
        # quaternion in numpy array
        quaternions[index] = [
            ahrs.quaternion.w,
            ahrs.quaternion.x,
            ahrs.quaternion.y,
            ahrs.quaternion.z,
        ]

    return tilt_deg, euler, quaternions


def plot_and_save_graphs(
    time, data, labels, title, xlabel, ylabel, sensor_name, save_path
):
    colors = ["red", "green", "blue"]  # Specific colors for X, Y, and Z axes
    plt.figure(figsize=(10, 6))
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(time, data[:, i], label=label, color=color)
    plt.title(f"{sensor_name} {title}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_results_to_csv(
    base_dir, time, gyroscope, accelerometer, euler, tilt_deg, quaternions, sensor_name
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
    csv_path = os.path.join(base_dir, f"{sensor_name}_results.csv")
    df.to_csv(csv_path, index=False)


def plot_and_save_sensor_data(
    time,
    gyroscope,
    accelerometer,
    euler,
    tilt_deg,
    quaternions,
    sensor_name,
    base_dir_figures,
):
    plot_and_save_graphs(
        time,
        accelerometer,
        ["Acc X", "Acc Y", "Acc Z"],
        "Accelerometer Data",
        "Time (s)",
        "Acceleration (g)",
        sensor_name,
        f"{base_dir_figures}/{sensor_name}_Accelerometer_Data.png",
    )
    plot_and_save_graphs(
        time,
        gyroscope,
        ["Gyro X", "Gyro Y", "Gyro Z"],
        "Gyroscope Data",
        "Time (s)",
        "Gyroscope (°/s)",
        sensor_name,
        f"{base_dir_figures}/{sensor_name}_Gyroscope_Data.png",
    )
    plot_and_save_graphs(
        time,
        tilt_deg,
        ["Tilt X", "Tilt Y", "Tilt Z"],
        "Tilt Angles",
        "Time (s)",
        "Tilt (°)",
        sensor_name,
        f"{base_dir_figures}/{sensor_name}_Tilt_Angles.png",
    )
    plot_and_save_graphs(
        time,
        euler,
        ["Euler X", "Euler Y", "Euler Z"],
        "Euler Angles",
        "Time (s)",
        "Euler (°)",
        sensor_name,
        f"{base_dir_figures}/{sensor_name}_Euler_Angles.png",
    )


def importc3d(dat):
    datac3d = ezc3d.c3d(dat)
    print(f"\nProcessing file: {dat}")
    print(f'Number of markers = {datac3d["parameters"]["POINT"]["USED"]["value"][0]}')
    point_data = datac3d["data"]["points"]
    points_residuals = datac3d["data"]["meta_points"]["residuals"]
    analogs = datac3d["data"]["analogs"]
    marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
    analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
    markers = point_data[0:3, :, :].T.reshape(-1, len(marker_labels) * 3)
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


def analyze_imu_data():
    """
    Analyzes all IMU CSV and C3D files in the specified directory.
    """
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Get user inputs
    user_inputs = get_user_inputs()
    file_type = user_inputs.get("file_type")
    sample_rate = user_inputs.get("sample_rate")

    if file_type not in ["csv", "c3d"]:
        messagebox.showerror("Error", "Invalid file type selected.")
        return

    if sample_rate is None or sample_rate <= 0:
        messagebox.showerror("Error", "Invalid sample rate provided.")
        return

    # Request the input directory from the user
    directory_path = filedialog.askdirectory(title="Select directory to read files")
    if not directory_path:
        messagebox.showerror("Error", "No input directory selected.")
        return

    # Request the output directory from the user
    output_directory = filedialog.askdirectory(
        title="Choose directory to save analysis"
    )
    if not output_directory:
        messagebox.showerror("Error", "No output directory selected.")
        return

    # Ask the user if they want to select headers for all files
    select_headers_for_all = messagebox.askyesno(
        "Header Selection", "Select headers for all files?"
    )

    selected_headers = None
    if select_headers_for_all:
        # Select a file to choose headers from
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
            # Convert analogs to 2D array
            analogs_2d = analogs.reshape(
                analogs.shape[0] * analogs.shape[1], analogs.shape[2]
            )
            df_analogs = pd.DataFrame(
                analogs_2d.T, columns=analog_labels
            )  # Convert to DataFrame for easier selection
            selected_headers = select_headers_gui(df_analogs.columns)

    # Apply the chosen filter to accelerometer and gyroscope data ('butterworth' or 'fir')
    filter_method = "fir"
    gravity = 9.81

    # List all files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if not os.path.isfile(file_path):
            continue

        if file_type == "csv" and file_name.endswith(".csv"):
            try:
                # Read the CSV file with selected headers if applicable
                if selected_headers:
                    data = pd.read_csv(file_path, usecols=selected_headers).values
                else:
                    data = show_csv(
                        file_path
                    )  # Assuming show_csv returns the selected data
                if data is not None:
                    dataf = apply_filter(data, sample_rate, method=filter_method)
                    accelerometer_1 = (
                        dataf[:, [0, 2, 1]] * np.array([1, 1, -1])
                    ) / gravity
                    gyroscope_1 = (dataf[:, [3, 5, 4]] * np.array([1, 1, -1])) * (
                        180 / np.pi
                    )
                    accelerometer_2 = (
                        dataf[:, [9, 11, 10]] * np.array([1, 1, -1])
                    ) / gravity
                    gyroscope_2 = (dataf[:, [12, 14, 13]] * np.array([1, 1, -1])) * (
                        180 / np.pi
                    )

                    time = np.linspace(
                        0, len(accelerometer_1) / sample_rate, len(accelerometer_1)
                    )

                    tilt_1, euler_1, quaternions_1 = imu_orientations(
                        accelerometer_1, gyroscope_1, time, sample_rate, "Sensor Trunk"
                    )
                    tilt_2, euler_2, quaternions_2 = imu_orientations(
                        accelerometer_2, gyroscope_2, time, sample_rate, "Sensor Pelvis"
                    )

                    print(f"Processed {file_name}:")
                    # print(f"Sensor 1 - Tilt shape: {tilt_1.shape}, Euler angles shape: {euler_1.shape}, Quaternions shape: {quaternions_1.shape}")
                    # print(f"Sensor 2 - Tilt shape: {tilt_2.shape}, Euler angles shape: {euler_2.shape}, Quaternions shape: {quaternions_2.shape}")

                    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_dir_figures = os.path.join(
                        output_directory,
                        "IMU_multimodal_results",
                        f"figure/IMU_{file_name}_{current_time}",
                    )
                    base_dir_processed_data = os.path.join(
                        output_directory,
                        "IMU_multimodal_results",
                        f"processed_data/IMU_{file_name}_{current_time}",
                    )
                    os.makedirs(base_dir_figures, exist_ok=True)
                    os.makedirs(base_dir_processed_data, exist_ok=True)

                    # Plot and save the data for each sensor
                    plot_and_save_sensor_data(
                        time,
                        gyroscope_1,
                        accelerometer_1,
                        euler_1,
                        tilt_1,
                        quaternions_1,
                        "Sensor Trunk",
                        base_dir_figures,
                    )
                    plot_and_save_sensor_data(
                        time,
                        gyroscope_2,
                        accelerometer_2,
                        euler_2,
                        tilt_2,
                        quaternions_2,
                        "Sensor Pelvis",
                        base_dir_figures,
                    )

                    # Save processed data for each sensor
                    save_results_to_csv(
                        base_dir_processed_data,
                        time,
                        gyroscope_1,
                        accelerometer_1,
                        euler_1,
                        tilt_1,
                        quaternions_1,
                        "Sensor Trunk",
                    )
                    save_results_to_csv(
                        base_dir_processed_data,
                        time,
                        gyroscope_2,
                        accelerometer_2,
                        euler_2,
                        tilt_2,
                        quaternions_2,
                        "Sensor Pelvis",
                    )

            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred while processing {file_name}: {e}"
                )

        elif file_type == "c3d" and file_name.endswith(".c3d"):
            try:
                # _, _, _, analogs, _, analog_labels, analog_freq = importc3d(file_path)
                # Convert analogs to 2D array
                # analogs_2d = analogs.reshape(analogs.shape[0] * analogs.shape[1], analogs.shape[2])
                # df_analogs = pd.DataFrame(analogs_2d.T, columns=analog_labels)  # Convert to DataFrame for easier selection
                analogs_selected = df_analogs[
                    selected_headers
                ].values  # Select columns from DataFrame
                dataf = apply_filter(
                    analogs_selected, sample_rate, method=filter_method
                )
                accelerometer_1 = (dataf[:, [0, 2, 1]] * np.array([1, 1, -1])) / gravity
                gyroscope_1 = (dataf[:, [3, 5, 4]] * np.array([1, 1, -1])) * (
                    180 / np.pi
                )
                accelerometer_2 = (
                    dataf[:, [9, 11, 10]] * np.array([1, 1, -1])
                ) / gravity
                gyroscope_2 = (dataf[:, [12, 14, 13]] * np.array([1, 1, -1])) * (
                    180 / np.pi
                )

                time = np.linspace(
                    0, len(accelerometer_1) / sample_rate, len(accelerometer_1)
                )

                tilt_1, euler_1, quaternions_1 = imu_orientations(
                    accelerometer_1, gyroscope_1, time, sample_rate, "Sensor Trunk"
                )
                tilt_2, euler_2, quaternions_2 = imu_orientations(
                    accelerometer_2, gyroscope_2, time, sample_rate, "Sensor Pelvis"
                )

                print(f"Processed {file_name}:")
                # print(f"Sensor 1 - Tilt shape: {tilt_1.shape}, Euler angles shape: {euler_1.shape}, Quaternions shape: {quaternions_1.shape}")
                # print(f"Sensor 2 - Tilt shape: {tilt_2.shape}, Euler angles shape: {euler_2.shape}, Quaternions shape: {quaternions_2.shape}")

                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_dir_figures = os.path.join(
                    output_directory,
                    "IMU_multimodal_results",
                    f"figure/IMU_{file_name}_{current_time}",
                )
                base_dir_processed_data = os.path.join(
                    output_directory,
                    "IMU_multimodal_results",
                    f"processed_data/IMU_{file_name}_{current_time}",
                )
                os.makedirs(base_dir_figures, exist_ok=True)
                os.makedirs(base_dir_processed_data, exist_ok=True)

                # Plot and save the data for each sensor
                plot_and_save_sensor_data(
                    time,
                    gyroscope_1,
                    accelerometer_1,
                    euler_1,
                    tilt_1,
                    quaternions_1,
                    "Sensor Trunk",
                    base_dir_figures,
                )
                plot_and_save_sensor_data(
                    time,
                    gyroscope_2,
                    accelerometer_2,
                    euler_2,
                    tilt_2,
                    quaternions_2,
                    "Sensor Pelvis",
                    base_dir_figures,
                )

                # Save processed data for each sensor
                save_results_to_csv(
                    base_dir_processed_data,
                    time,
                    gyroscope_1,
                    accelerometer_1,
                    euler_1,
                    tilt_1,
                    quaternions_1,
                    "Sensor Trunk",
                )
                save_results_to_csv(
                    base_dir_processed_data,
                    time,
                    gyroscope_2,
                    accelerometer_2,
                    euler_2,
                    tilt_2,
                    quaternions_2,
                    "Sensor Pelvis",
                )

            except Exception as e:
                messagebox.showerror(
                    "Error", f"An error occurred while processing {file_name}: {e}"
                )

    root.destroy()

    messagebox.showinfo(
        "Success", "All files have been processed and saved successfully."
    )


# Main function call
if __name__ == "__main__":
    analyze_imu_data()
