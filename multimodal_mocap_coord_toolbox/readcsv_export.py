"""
readcsv_export.py
Version: 2024-07-25 18:00:00
"""

import numpy as np
import pandas as pd
import ezc3d
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import os

# Dictionary for metric unit conversions with abbreviations
CONVERSIONS = {
    'meters': (1, 'm'),
    'centimeters': (100, 'cm'),
    'millimeters': (1000, 'mm'),
    'kilometers': (0.001, 'km'),
    'inches': (39.3701, 'in'),
    'feet': (3.28084, 'ft'),
    'yards': (1.09361, 'yd'),
    'miles': (0.000621371, 'mi'),
    'seconds': (1, 's'),
    'minutes': (1/60, 'min'),
    'hours': (1/3600, 'hr'),
    'days': (1/86400, 'day'),
    'volts': (1, 'V'),
    'millivolts': (1000, 'mV'),
    'microvolts': (1e6, 'µV'),
    'degrees': (1, 'deg'),
    'radians': (3.141592653589793/180, 'rad'),
    'km_per_hour': (1, 'km/h'),
    'meters_per_second': (1000/3600, 'm/s'),
    'miles_per_hour': (0.621371, 'mph'),
    'kilograms': (1, 'kg'),
    'newtons': (9.80665, 'N'),
    'angular_rotation_per_second': (1, 'rps'),
    'rpm': (1/60, 'rpm'),
    'radians_per_second': (2 * 3.141592653589793 / 60, 'rad/s')
}

def create_c3d_from_csv(points_df, analog_df=None, point_rate=100, analog_rate=1000, conversion_factor=1):
    c3d = ezc3d.c3d()

    # Extract marker labels
    marker_labels = [col.rsplit('_', 1)[0] for col in points_df.columns[1::3]]
    c3d['parameters']['POINT']['LABELS']['value'] = marker_labels

    # Convert points data to the required shape (4xNxT)
    num_markers = len(marker_labels)
    num_frames = len(points_df)
    points_data = np.zeros((4, num_markers, num_frames))

    for i, label in enumerate(marker_labels):
        points_data[0, i, :] = points_df[f"{label}_X"].values * conversion_factor
        points_data[1, i, :] = points_df[f"{label}_Y"].values * conversion_factor
        points_data[2, i, :] = points_df[f"{label}_Z"].values * conversion_factor
        points_data[3, i, :] = 1  # Homogeneous coordinate

    c3d['data']['points'] = points_data
    c3d['parameters']['POINT']['RATE']['value'] = [point_rate]

    # Handle analog data if provided
    if analog_df is not None:
        analog_labels = list(analog_df.columns)[1:]
        num_analog = len(analog_labels)
        c3d['parameters']['ANALOG']['LABELS']['value'] = analog_labels
        analog_data = np.zeros((1, num_analog, num_frames))

        for i, label in enumerate(analog_labels):
            analog_data[0, i, :] = analog_df[label].values

        c3d['data']['analogs'] = analog_data
        c3d['parameters']['ANALOG']['RATE']['value'] = [analog_rate]

    # Write the C3D file
    output_path = filedialog.asksaveasfilename(defaultextension=".c3d", filetypes=[("C3D files", "*.c3d")])
    if output_path:
        c3d.write(output_path)
        messagebox.showinfo("Success", f"C3D file saved to {output_path}")
    else:
        messagebox.showwarning("Warning", "Save operation cancelled.")

def get_conversion_factor():
    convert_window = tk.Toplevel()
    convert_window.title("Conversion Factor")
    convert_window.geometry("400x300")

    unit_options = list(CONVERSIONS.keys())

    current_unit_label = tk.Label(convert_window, text="Current Unit:")
    current_unit_label.pack(pady=5)
    current_unit_listbox = tk.Listbox(convert_window, selectmode=tk.SINGLE, exportselection=False)
    current_unit_listbox.pack(pady=5)
    for unit in unit_options:
        current_unit_listbox.insert(tk.END, unit)

    target_unit_label = tk.Label(convert_window, text="Target Unit:")
    target_unit_label.pack(pady=5)
    target_unit_listbox = tk.Listbox(convert_window, selectmode=tk.SINGLE, exportselection=False)
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

    return convert_window.conversion_factor if hasattr(convert_window, 'conversion_factor') else 1

def convert_csv_to_c3d():
    root = tk.Tk()
    root.withdraw()

    point_file_path = filedialog.askopenfilename(title="Select Point Data CSV", filetypes=[("CSV files", "*.csv")])
    if not point_file_path:
        messagebox.showerror("Error", "No point data file selected.")
        return

    point_df = pd.read_csv(point_file_path)

    use_analog = messagebox.askyesno("Analog Data", "Do you have an analog data CSV file to add?")
    analog_df = None

    if use_analog:
        analog_file_path = filedialog.askopenfilename(title="Select Analog Data CSV", filetypes=[("CSV files", "*.csv")])
        if analog_file_path:
            analog_df = pd.read_csv(analog_file_path)

    point_rate = simpledialog.askinteger("Point Rate", "Enter the point data rate (Hz):", minvalue=1, initialvalue=100)
    analog_rate = 1000
    if analog_df is not None:
        analog_rate = simpledialog.askinteger("Analog Rate", "Enter the analog data rate (Hz):", minvalue=1, initialvalue=1000)

    conversion_factor = get_conversion_factor()

    try:
        create_c3d_from_csv(point_df, analog_df, point_rate, analog_rate, conversion_factor)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while creating C3D file: {e}")

if __name__ == "__main__":
    convert_csv_to_c3d()
