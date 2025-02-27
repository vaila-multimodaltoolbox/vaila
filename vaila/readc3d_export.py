"""
File: readc3d_export.py

Version: 0.1.9
Date: February 2025
Author: Prof. Paulo Santiago

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
- Extracts and saves force platform data including Center of Pressure (COP) from the C3D file into CSV files.
- Saves the COP data in a combined CSV file with time columns and platform indices.
- Saves a summary file with platform information.

Dependencies:
- Python 3.12.9
- ezc3d
- Pandas
- Tkinter
- Tqdm
- Numpy
- Openpyxl (optional, for saving Excel files)
- Rich (optional, for console output)

Usage:
- Run the script, select the input directory containing .c3d files, and specify an output directory.
- Choose whether to save the files in Excel format.
- The script will process each .c3d file in the input directory and save the results in the specified output directory.

Example:
Windows:
$ python readc3d_export.py

Linux:
$ python3 readc3d_export.py

macOS:
$ python3 readc3d_export.py

Notes:
- Ensure that all necessary libraries are installed.
- This script is designed to handle large datasets efficiently, but saving to Excel format may take significant time depending on the dataset size.
- The calculation and export of COP data were removed from this script.
  The COP processing will be performed later in the cop_calculate.py script.
"""

import os
from rich import print
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

    # Verify if the necessary parameters for events are available
    if "EVENT" not in datac3d["parameters"]:
        print(f"No events found for {file_name}, saving empty file.")
        save_empty_file(os.path.join(output_dir, f"{file_name}_events.csv"))
        return

    event_params = datac3d["parameters"]["EVENT"]
    required_keys = ["CONTEXTS", "LABELS", "TIMES"]
    if not all(key in event_params for key in required_keys):
        print(f"Event parameters incomplete for {file_name}, saving empty file.")
        save_empty_file(os.path.join(output_dir, f"{file_name}_events.csv"))
        return

    # Collect the event data
    event_contexts = event_params["CONTEXTS"]["value"]
    event_labels = event_params["LABELS"]["value"]
    event_times = event_params["TIMES"]["value"][1, :]  # Only the times (line 1)
    marker_freq = datac3d["header"]["points"]["frame_rate"]

    # Build the event data
    events_data = []
    for context, label, time in zip(event_contexts, event_labels, event_times):
        frame = int(round(time * marker_freq))
        events_data.append(
            {"Context": context, "Label": label, "Time": time, "Frame": frame}
        )

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

    # Load the C3D file with force platform data extraction
    datac3d = c3d(dat, extract_forceplat_data=True)
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

    # Check for force platform data
    has_platforms = "platform" in datac3d["data"]
    num_platforms = len(datac3d["data"]["platform"]) if has_platforms else 0

    # Print summary information
    num_analog_channels = datac3d["parameters"]["ANALOG"]["USED"]["value"][0]
    print(f"Number of marker labels = {len(marker_labels)}")
    print(f"Number of analog channels = {num_analog_channels}")
    print(f"Number of force platforms = {num_platforms}")
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


def save_platform_data(datac3d, file_name, output_dir):
    """
    Save force platform data including Center of Pressure (COP) from the C3D file into CSV files.
    """
    print(f"Checking for platform data in {file_name}")

    if "platform" not in datac3d["data"] or not datac3d["data"]["platform"]:
        print(f"No force platform data found for {file_name}")
        save_empty_file(os.path.join(output_dir, f"{file_name}_cop_all.csv"))
        return

    platforms = datac3d["data"]["platform"]
    num_platforms = len(platforms)
    print(f"Found {num_platforms} force platforms for {file_name}")

    # Get analog frequency for time column
    analog_freq = datac3d["header"]["analogs"]["frame_rate"]

    # Prepare combined dataframe for all COPs
    all_cop_data = []
    max_frames = 0

    # Process each platform
    for platform_idx, platform in enumerate(platforms):
        print(f"Processing platform {platform_idx} for {file_name}")

        # Extract and save center of pressure data
        if "center_of_pressure" in platform:
            cop_data = platform["center_of_pressure"]
            num_frames = cop_data.shape[1]
            max_frames = max(max_frames, num_frames)

            # Create DataFrame with time and cop x, y, z
            time_values = [f"{i / analog_freq:.3f}" for i in range(num_frames)]

            cop_df = pd.DataFrame(
                {
                    "Time": time_values,
                    f"COP_X_P{platform_idx}": cop_data[0, :],
                    f"COP_Y_P{platform_idx}": cop_data[1, :],
                    f"COP_Z_P{platform_idx}": cop_data[2, :],
                }
            )

            # Save individual platform COP data to CSV
            cop_file_path = os.path.join(
                output_dir, f"{file_name}_platform{platform_idx}_cop.csv"
            )
            cop_df.to_csv(cop_file_path, index=False)
            print(f"COP data saved to: {cop_file_path}")

            # Add to combined data
            all_cop_data.append({"platform": platform_idx, "data": cop_data})

        # Extract and save force data
        if "force" in platform:
            force_data = platform["force"]

            # Create DataFrame with time and force x, y, z
            num_frames = force_data.shape[1]
            time_values = [f"{i / analog_freq:.3f}" for i in range(num_frames)]

            force_df = pd.DataFrame(
                {
                    "Time": time_values,
                    "Force_X": force_data[0, :],
                    "Force_Y": force_data[1, :],
                    "Force_Z": force_data[2, :],
                }
            )

            # Save to CSV
            force_file_path = os.path.join(
                output_dir, f"{file_name}_platform{platform_idx}_force.csv"
            )
            force_df.to_csv(force_file_path, index=False)
            print(f"Force data saved to: {force_file_path}")

        # Extract and save moment data
        if "moment" in platform:
            moment_data = platform["moment"]

            # Create DataFrame with time and moment x, y, z
            num_frames = moment_data.shape[1]
            time_values = [f"{i / analog_freq:.3f}" for i in range(num_frames)]

            moment_df = pd.DataFrame(
                {
                    "Time": time_values,
                    "Moment_X": moment_data[0, :],
                    "Moment_Y": moment_data[1, :],
                    "Moment_Z": moment_data[2, :],
                }
            )

            # Save to CSV
            moment_file_path = os.path.join(
                output_dir, f"{file_name}_platform{platform_idx}_moment.csv"
            )
            moment_df.to_csv(moment_file_path, index=False)
            print(f"Moment data saved to: {moment_file_path}")

    # Create combined COP CSV with data from all platforms
    if all_cop_data:
        # Create time column
        time_values = [f"{i / analog_freq:.3f}" for i in range(max_frames)]
        combined_cop = {"Time": time_values}

        # Add data from each platform
        for platform_data in all_cop_data:
            platform_idx = platform_data["platform"]
            cop_data = platform_data["data"]
            num_frames = cop_data.shape[1]

            # Add columns for this platform
            combined_cop[f"P{platform_idx}_COP_X"] = np.pad(
                cop_data[0, :],
                (0, max_frames - num_frames),
                "constant",
                constant_values=np.nan,
            )
            combined_cop[f"P{platform_idx}_COP_Y"] = np.pad(
                cop_data[1, :],
                (0, max_frames - num_frames),
                "constant",
                constant_values=np.nan,
            )
            combined_cop[f"P{platform_idx}_COP_Z"] = np.pad(
                cop_data[2, :],
                (0, max_frames - num_frames),
                "constant",
                constant_values=np.nan,
            )

        # Save combined COP data
        combined_cop_df = pd.DataFrame(combined_cop)
        combined_cop_path = os.path.join(output_dir, f"{file_name}_cop_all.csv")
        combined_cop_df.to_csv(combined_cop_path, index=False)
        print(f"Combined COP data saved to: {combined_cop_path}")

    # Save a summary file with platform information
    try:
        platform_info_path = os.path.join(output_dir, f"{file_name}_platform_info.csv")
        platform_info = []

        for platform_idx, platform in enumerate(platforms):
            if "origin" in platform and "corners" in platform:
                origin = platform["origin"]
                corners = platform["corners"]

                platform_info.append(
                    {
                        "Platform": platform_idx,
                        "Origin_X": origin[0],
                        "Origin_Y": origin[1],
                        "Origin_Z": origin[2],
                        "Corner1_X": corners[0, 0],
                        "Corner1_Y": corners[1, 0],
                        "Corner2_X": corners[0, 1],
                        "Corner2_Y": corners[1, 1],
                        "Corner3_X": corners[0, 2],
                        "Corner3_Y": corners[1, 2],
                        "Corner4_X": corners[0, 3],
                        "Corner4_Y": corners[1, 3],
                    }
                )

        if platform_info:
            pd.DataFrame(platform_info).to_csv(platform_info_path, index=False)
            print(f"Platform information saved to: {platform_info_path}")
    except Exception as e:
        print(f"Error saving platform information: {e}")


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
    run_save_dir,
    save_excel,
    datac3d,
):
    """
    Save the extracted data to CSV files and .info files within a specific directory
    created for the specific file, within the execution save directory.
    """
    print(f"Saving data to files for {file_name}")
    # Create a subfolder for the file within the execution save directory
    file_dir = os.path.join(run_save_dir, file_name)
    os.makedirs(file_dir, exist_ok=True)
    print(f"Directory created: {file_dir}")

    # Save detailed .info file and the short .info file
    save_info_file(datac3d, file_name, file_dir)
    save_short_info_file(
        marker_labels,
        marker_freq,
        analog_labels,
        analog_units,
        analog_freq,
        file_dir,
        file_name,
    )
    # Save events
    save_events(datac3d, file_name, file_dir)

    # Save force platform data
    save_platform_data(datac3d, file_name, file_dir)

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
            os.path.join(file_dir, f"{file_name}_markers.csv"), index=False
        )
    else:
        print(f"No markers found for {file_name}, saving empty file.")
        save_empty_file(os.path.join(file_dir, f"{file_name}_markers.csv"))

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
            os.path.join(file_dir, f"{file_name}_analogs.csv"), index=False
        )
    else:
        print(f"No analogs found for {file_name}, saving empty file.")
        save_empty_file(os.path.join(file_dir, f"{file_name}_analogs.csv"))

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
            os.path.join(file_dir, f"{file_name}_points_residuals.csv"), index=False
        )
    else:
        print(f"No points residuals found for {file_name}, saving empty file.")
        save_empty_file(os.path.join(file_dir, f"{file_name}_points_residuals.csv"))

    # Optionally save to Excel
    if save_excel:
        print("Saving to Excel. This process can take a long time...")
        with pd.ExcelWriter(
            os.path.join(file_dir, f"{file_name}.xlsx"), engine="openpyxl"
        ) as writer:
            if markers.size > 0:
                markers_df.to_excel(writer, sheet_name="Markers", index=False)
            if analogs.size > 0:
                analogs_df.to_excel(writer, sheet_name="Analogs", index=False)
            if points_residuals.size > 0:
                points_residuals_df.to_excel(
                    writer, sheet_name="Points Residuals", index=False
                )

            # Add platform data to Excel if available
            if "platform" in datac3d["data"] and datac3d["data"]["platform"]:
                for platform_idx, platform in enumerate(datac3d["data"]["platform"]):
                    if "center_of_pressure" in platform:
                        cop_data = platform["center_of_pressure"]
                        num_frames = cop_data.shape[1]
                        time_values = [
                            f"{i / analog_freq:.3f}" for i in range(num_frames)
                        ]

                        cop_df = pd.DataFrame(
                            {
                                "Time": time_values,
                                "COP_X": cop_data[0, :],
                                "COP_Y": cop_data[1, :],
                                "COP_Z": cop_data[2, :],
                            }
                        )
                        cop_df.to_excel(
                            writer,
                            sheet_name=f"Platform{platform_idx}_COP",
                            index=False,
                        )

    print(f"Files for {file_name} saved successfully!")
    return file_dir


def convert_c3d_to_csv():
    """
    Main function to convert C3D files to CSV and .info files.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
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

    root.destroy()  # Close the Tkinter resources

    if input_directory and output_directory:
        c3d_files = sorted(
            [f for f in os.listdir(input_directory) if f.endswith(".c3d")]
        )
        print(f"Found {len(c3d_files)} .c3d files in the input directory.")

        # Create the root directory for saving with timestamp in the name
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_save_dir = os.path.join(
            output_directory, f"vaila_c3d_to_csv_{run_timestamp}"
        )
        os.makedirs(run_save_dir, exist_ok=True)
        print(f"Run-level save directory created: {run_save_dir}")

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
                # Save the extracted data in CSV files and .info files within the execution save directory
                out_dir = save_to_files(
                    markers,
                    marker_labels,
                    marker_freq,
                    analogs,
                    points_residuals,
                    analog_labels,
                    analog_units,
                    analog_freq,
                    file_name,
                    run_save_dir,
                    save_excel,
                    datac3d,
                )
            except Exception as e:
                print(f"Error processing {c3d_file}: {e}")
                messagebox.showerror("Error", f"Failed to process {c3d_file}: {e}")
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
