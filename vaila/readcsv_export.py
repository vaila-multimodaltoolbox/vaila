"""
readcsv_export.py
Author: Paulo R. P. Santiago
Version: 2024-09-25 11:07:00

Description:
This script provides functionality to convert CSV files containing point and analog data 
into the C3D format, commonly used for motion capture data analysis. The script uses the 
ezc3d library to create C3D files from CSV inputs while sanitizing and formatting the data
to ensure compatibility with the C3D standard.

Main Features:
- Reads point and analog data from user-selected CSV files.
- Sanitizes headers to remove unwanted characters and ensure proper naming conventions.
- Handles user input for data rates, unit conversions, and sorting preferences.
- Converts the CSV data into a C3D file with appropriately formatted point and analog data.
- Provides a user interface for selecting files and entering required information using Tkinter.

Functions:
- sanitize_header: Cleans and formats CSV headers to conform to expected data formats.
- convert_csv_to_c3d: Handles user input and coordinates the conversion process from CSV to C3D.
- create_c3d_from_csv: Constructs the C3D file from the sanitized data.
- validate_and_filter_columns: Validates and filters CSV columns to ensure correct formatting.
- get_conversion_factor: Provides a user interface for unit conversion selection.

Dependencies:
- numpy: For numerical data handling.
- pandas: For data manipulation and reading CSV files.
- ezc3d: To create and write C3D files.
- tkinter: For GUI elements, including file dialogs and message boxes.

Usage:
Run the script, select the necessary CSV files for point and analog data, provide the required 
parameters, and save the resulting C3D file to the desired location.

"""

import numpy as np
import pandas as pd
import re
import ezc3d
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# Dictionary for metric unit conversions with abbreviations
CONVERSIONS = {
    "meters": (1, "m"),
    "centimeters": (100, "cm"),
    "millimeters": (1000, "mm"),
    "kilometers": (0.001, "km"),
    "inches": (39.3701, "in"),
    "feet": (3.28084, "ft"),
    "yards": (1.09361, "yd"),
    "miles": (0.000621371, "mi"),
    "seconds": (1, "s"),
    "minutes": (1 / 60, "min"),
    "hours": (1 / 3600, "hr"),
    "days": (1 / 86400, "day"),
    "volts": (1, "V"),
    "millivolts": (1000, "mV"),
    "microvolts": (1e6, "µV"),
    "degrees": (1, "deg"),
    "radians": (3.141592653589793 / 180, "rad"),
    "km_per_hour": (1, "km/h"),
    "meters_per_second": (1000 / 3600, "m/s"),
    "miles_per_hour": (0.621371, "mph"),
    "kilograms": (1, "kg"),
    "newtons": (9.80665, "N"),
    "angular_rotation_per_second": (1, "rps"),
    "rpm": (1 / 60, "rpm"),
    "radians_per_second": (2 * 3.141592653589793 / 60, "rad/s"),
    "radians_per_minute": (2 * 3.141592653589793 / 3600, "rad/min"),
    "radians_per_hour": (2 * 3.141592653589793 / 86400, "rad/hr"),
    "watts": (1, "W"),
    "pounds": (0.453592, "lb"),
    "joules": (1, "J"),
    "kilojoules": (1000, "kJ"),
    "watt_hours": (3600, "Wh"),
    "kilojoules_per_hour": (3600, "kWh"),
    "calories": (4.184, "cal"),
    "kilocalories": (4.184, "kcal"),
}


def sanitize_header(header):
    """
    Sanitize the CSV header to ensure it is in the correct format.
    - Remove any existing suffixes (_X, _Y, _Z or .X, .Y, .Z) from column names.
    - Replace any characters that are not letters, numbers, or underscores.
    - Ensure each coordinate column has the correct suffix (_X, _Y, _Z).
    - Fix any empty or incorrectly named columns.
    """
    new_header = [
        header[0]
    ]  # Keep the first column unchanged (typically 'Time' or 'Frame')
    empty_column_counter = 0  # Counter for empty columns

    for i, col in enumerate(header[1:]):  # Start from the second column
        col = col.strip().upper()  # Remove extra spaces and convert to uppercase

        if not col:  # Handle empty columns
            base_name = f"VAILA{empty_column_counter // 3 + 1}"
            suffix = ["_X", "_Y", "_Z"][empty_column_counter % 3]
            new_col = base_name + suffix
            empty_column_counter += 1
        else:
            # Remove any unwanted characters (anything not a letter, number, or underscore)
            col = re.sub(r"[^A-Z0-9_]", "_", col)

            # Remove existing suffixes "_X", "_Y", "_Z", ".X", ".Y", ".Z" if present
            if col.endswith(("_X", "_Y", "_Z")):
                col = col.rsplit("_", 1)[0]

            # Add the correct suffix based on the (i) position to ensure starting at "_X"
            correct_suffix = ["_X", "_Y", "_Z"][(i) % 3]
            new_col = col + correct_suffix

        new_header.append(new_col)

    # Ensure all markers have their _X, _Y, and _Z coordinates
    sanitized_header = []
    i = 0
    while i < len(new_header):
        sanitized_header.append(new_header[i])
        if i > 0 and (i + 2) < len(
            new_header
        ):  # Ensure we have space for a set of _X, _Y, _Z
            base_name = new_header[i].rsplit("_", 1)[
                0
            ]  # Get the base name without the suffix
            sanitized_header.append(base_name + "_Y")
            sanitized_header.append(base_name + "_Z")
            i += 2
        i += 1

    return sanitized_header


def validate_and_filter_columns(df):
    """
    Filter only the columns that are in the correct format, ignoring the first column.
    """
    valid_columns = [df.columns[0]] + [
        col
        for col in df.columns[1:]
        if "_" in col and col.split("_")[-1] in ["X", "Y", "Z"]
    ]
    return df[valid_columns]


def get_conversion_factor():
    """
    Display a window to select the conversion factor for units.
    """
    convert_window = tk.Toplevel()
    convert_window.title("Conversion Factor")
    convert_window.geometry("400x600")

    unit_options = list(CONVERSIONS.keys())

    current_unit_label = tk.Label(convert_window, text="Current Unit:")
    current_unit_label.pack(pady=5)
    current_unit_listbox = tk.Listbox(
        convert_window, selectmode=tk.SINGLE, exportselection=False
    )
    current_unit_listbox.pack(pady=5)
    for unit in unit_options:
        current_unit_listbox.insert(tk.END, unit)

    target_unit_label = tk.Label(convert_window, text="Target Unit:")
    target_unit_label.pack(pady=5)
    target_unit_listbox = tk.Listbox(
        convert_window, selectmode=tk.SINGLE, exportselection=False
    )
    target_unit_listbox.pack(pady=5)
    for unit in unit_options:
        target_unit_listbox.insert(tk.END, unit)

    def on_submit():
        current_unit = current_unit_listbox.get(tk.ACTIVE)
        target_unit = target_unit_listbox.get(tk.ACTIVE)
        conversion_factor = CONVERSIONS[target_unit][0] / CONVERSIONS[current_unit][0]
        convert_window.conversion_factor = conversion_factor
        convert_window.destroy()

    submit_button = tk.Button(convert_window, text="Submit", command=on_submit)
    submit_button.pack(pady=10)

    convert_window.transient()
    convert_window.grab_set()
    convert_window.wait_window()

    return (
        convert_window.conversion_factor
        if hasattr(convert_window, "conversion_factor")
        else 1
    )


def convert_csv_to_c3d():
    """
    Handle the CSV to C3D conversion process, including file selection and user inputs.
    """
    root = tk.Tk()
    root.withdraw()

    point_file_path = filedialog.askopenfilename(
        title="Select Point Data CSV", filetypes=[("CSV files", "*.csv")]
    )
    if not point_file_path:
        messagebox.showerror("Error", "No point data file selected.")
        return

    point_df = pd.read_csv(point_file_path)

    print(f"Loaded point data from {point_file_path}")
    print(f"Point data header: {point_df.columns.tolist()}")
    print(f"Point data shape: {point_df.shape}")

    point_df.columns = sanitize_header(point_df.columns)

    use_analog = messagebox.askyesno(
        "Analog Data", "Do you have an analog data CSV file to add?"
    )
    analog_df = None

    if use_analog:
        analog_file_path = filedialog.askopenfilename(
            title="Select Analog Data CSV", filetypes=[("CSV files", "*.csv")]
        )
        if analog_file_path:
            analog_df = pd.read_csv(analog_file_path)
            analog_df.columns = sanitize_header(analog_df.columns)

            print(f"Loaded analog data from {analog_file_path}")
            print(f"Analog data header: {analog_df.columns.tolist()}")
            print(f"Analog data shape: {analog_df.shape}")

    point_rate = simpledialog.askinteger(
        "Point Rate", "Enter the point data rate (Hz):", minvalue=1, initialvalue=100
    )
    analog_rate = 1000
    if analog_df is not None:
        analog_rate = simpledialog.askinteger(
            "Analog Rate",
            "Enter the analog data rate (Hz):",
            minvalue=1,
            initialvalue=1000,
        )

    conversion_factor = get_conversion_factor()

    sort_markers = messagebox.askyesno(
        "Sort Markers", "Do you want to sort markers alphabetically?"
    )

    try:
        create_c3d_from_csv(
            point_df,
            analog_df,
            point_rate,
            analog_rate,
            conversion_factor,
            sort_markers,
        )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while creating C3D file: {e}")


def create_c3d_from_csv(
    points_df,
    analog_df=None,
    point_rate=100,
    analog_rate=1000,
    conversion_factor=1,
    sort_markers=False,
):
    """
    Create a C3D file from the given point and analog data.
    """
    print("Creating C3D from CSV...")

    c3d = ezc3d.c3d()
    print("Initialized empty C3D object.")

    points_df = validate_and_filter_columns(points_df)
    print("Filtered and sanitized columns for points:", points_df.columns.tolist())

    marker_labels = [col.rsplit("_", 1)[0] for col in points_df.columns[1::3]]
    if sort_markers:
        marker_labels.sort()
    print("Extracted marker labels:", marker_labels)

    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_labels
    c3d["parameters"]["POINT"]["RATE"]["value"] = [point_rate]

    num_markers = len(marker_labels)
    num_frames = len(points_df)
    print(f"Number of markers: {num_markers}, Number of frames: {num_frames}")

    points_data = np.zeros((4, num_markers, num_frames))
    print("Initialized points data array with shape:", points_data.shape)

    for i, label in enumerate(marker_labels):
        try:
            points_data[0, i, :] = points_df[f"{label}_X"].values * conversion_factor
            points_data[1, i, :] = points_df[f"{label}_Y"].values * conversion_factor
            points_data[2, i, :] = points_df[f"{label}_Z"].values * conversion_factor
            points_data[3, i, :] = 1  # Coordenada homogênea
        except KeyError as e:
            print(f"Error accessing data for label '{label}': {e}")
            raise

    print("Points data populated successfully.")
    c3d["data"]["points"] = points_data
    print("Assigned points data to C3D.")

    if analog_df is not None:
        analog_labels = list(analog_df.columns[1:])
        num_analog = len(analog_labels)
        print("Analog labels:", analog_labels)
        c3d["parameters"]["ANALOG"]["LABELS"]["value"] = analog_labels
        c3d["parameters"]["ANALOG"]["RATE"]["value"] = [analog_rate]

        num_analog_frames = analog_df.shape[0]
        analog_data = np.zeros((1, num_analog, num_analog_frames))

        for i, label in enumerate(analog_labels):
            try:
                analog_data[0, i, :] = analog_df[label].values
            except KeyError as e:
                print(f"Error accessing analog data for label '{label}': {e}")
                raise

        print(f"Analog data shape: {analog_data.shape}")
        c3d["data"]["analogs"] = analog_data
        print("Analog data assigned to C3D.")

    print(f"Final POINT RATE: {c3d['parameters']['POINT']['RATE']['value']}")
    print(f"Final ANALOG RATE: {c3d['parameters']['ANALOG']['RATE']['value']}")
    print(f"Final POINT SHAPE: {c3d['data']['points'].shape}")
    print(f"Final ANALOG SHAPE: {c3d['data']['analogs'].shape}")

    output_path = filedialog.asksaveasfilename(
        defaultextension=".c3d", filetypes=[("C3D files", "*.c3d")]
    )
    if output_path:
        try:
            c3d.write(output_path)
            print(f"C3D file saved to {output_path}.")
            messagebox.showinfo("Success", f"C3D file saved to {output_path}")
        except Exception as e:
            print(f"Error writing C3D file: {e}")
            messagebox.showerror("Error", f"Failed to save C3D file: {e}")
    else:
        print("Save operation cancelled.")
        messagebox.showwarning("Warning", "Save operation cancelled.")


if __name__ == "__main__":
    convert_csv_to_c3d()
