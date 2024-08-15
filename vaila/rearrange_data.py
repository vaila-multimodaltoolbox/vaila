"""
Script Name: rearrange_data.py
Version: 2024-08-10 14:30:00
Author: Paulo Santiago

Description:
-------------
This script provides tools for rearranging and processing CSV data files. 
It includes functions for:
- Reordering columns.
- Merging and stacking CSV files.
- Converting MediaPipe data to a format compatible with 'getpixelvideo.py'.
- Detecting precision and scientific notation in the data.
- Converting units between various metric systems.
- Modifying lab reference systems.

Features:
---------
- Batch processing of CSV files.
- Support for GUI-based column reordering and data reshaping.
- MediaPipe data conversion to a compatible format for pixel coordinate visualization.
- Flexible unit conversion and custom lab reference adjustments.
- User-friendly interface with options for saving intermediate and final processed data.

Changelog:
----------
- 2024-08-10: Added functionality to batch convert MediaPipe CSV files and save them in a new directory.
- 2024-08-10: Implemented automatic directory creation for saving converted MediaPipe data.
- 2024-07-31: Initial version with core functionalities for CSV reordering and unit conversion.

Usage:
------
- Run the script to launch a GUI for reordering CSV columns.
- Use the "Convert MediaPipe" button to batch convert MediaPipe CSV files to a compatible format.
- Save the processed files in a timestamped directory.

Requirements:
-------------
- Python 3.x
- pandas
- numpy
- tkinter

"""

import os
import pandas as pd
#import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Scrollbar
from datetime import datetime
from vaila import modifylabref
from vaila.mergestack import select_file, merge_csv_files, stack_csv_files

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
}


# Function to detect scientific notation and maximum precision in the data
def detect_precision_and_notation(file_path):
    df = pd.read_csv(file_path, dtype=str)
    max_decimal_places = 0
    scientific_notation = False
    for col in df.columns:
        col_max_decimal = (
            df[col].apply(lambda x: len(x.split(".")[1]) if "." in str(x) else 0).max()
        )
        max_decimal_places = max(max_decimal_places, col_max_decimal)
        if df[col].str.contains(r"[eE]").any():
            scientific_notation = True
    return max_decimal_places, scientific_notation


# Function to save the DataFrame with the detected precision
def save_dataframe(df, file_path, columns, max_decimal_places):
    float_format = f"%.{max_decimal_places}f"
    df.to_csv(file_path, index=False, columns=columns, float_format=float_format)


# Function to get headers from the CSV file
def get_headers(file_path):
    try:
        df = pd.read_csv(file_path, nrows=0)
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading headers from {file_path}: {e}")
        return []


# Function to reshape data
def reshapedata(file_path, new_order, save_directory, suffix, max_decimal_places):
    try:
        print(f"Starting reshapedata for {file_path}")
        headers = get_headers(file_path)

        # Read CSV with Pandas
        df = pd.read_csv(file_path)

        print("Headers read from the CSV:")
        print(headers)

        print("First 5 rows of the original DataFrame:")
        print(df.head())

        new_order_indices = [headers.index(col) for col in new_order]
        df_reordered = df.iloc[:, new_order_indices]

        print("First 5 rows of the reordered DataFrame:")
        print(df_reordered.head())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_name = f"{base_name}_{timestamp}{suffix}.csv"
        new_file_path = os.path.join(save_directory, new_file_name)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        save_dataframe(df_reordered, new_file_path, new_order, max_decimal_places)

        print(f"Reordered data saved to {new_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# Function to convert MediaPipe data to the format compatible with getpixelvideo.py
def convert_mediapipe_to_pixel_format(file_path, save_directory):
    df = pd.read_csv(file_path)

    # Adjust the "frame" column to start from 0
    df.iloc[:, 0] = df.iloc[:, 0] - 1

    # Create the new DataFrame with the "frame" column and pX_x, pX_y coordinates
    new_df = pd.DataFrame()
    new_df["frame"] = df.iloc[:, 0]  # Use the first column as "frame"

    columns = df.columns[
        1:
    ]  # Ignore the first column, which we already used for "frame"
    for i in range(0, len(columns), 3):
        if i + 1 < len(columns):
            x_col = columns[i]
            y_col = columns[i + 1]
            new_df[f"p{i//3 + 1}_x"] = df[x_col]
            new_df[f"p{i//3 + 1}_y"] = df[y_col]

    # Save the new CSV file in the desired format
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    new_file_name = f"{base_name}_converted.csv"
    new_file_path = os.path.join(save_directory, new_file_name)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    new_df.to_csv(new_file_path, index=False)
    print(f"Converted MediaPipe data saved to {new_file_path}")
    messagebox.showinfo("Success", f"Converted MediaPipe data saved to {new_file_path}")


# Function to batch convert all MediaPipe CSV files in a directory
def batch_convert_mediapipe(directory_path):
    if not directory_path:
        print("No directory selected.")
        return

    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # Create a new directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(directory_path, f"Convert_MediaPipe_{timestamp}")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for file_name in csv_files:
        file_path = os.path.join(directory_path, file_name)
        convert_mediapipe_to_pixel_format(file_path, save_directory)

    print(f"All files have been converted and saved to {save_directory}")
    messagebox.showinfo(
        "Success", f"All files have been converted and saved to {save_directory}"
    )


class ColumnReorderGUI(tk.Tk):
    def __init__(self, original_headers, file_names, directory_path):
        super().__init__()
        self.original_headers = original_headers
        self.current_order = original_headers.copy()
        self.file_names = file_names
        self.directory_path = directory_path
        self.rearranged_path = os.path.join(directory_path, "data_rearranged")
        self.history = []

        base_file_name = file_names[0]

        # Detect precision and scientific notation in the first file
        self.max_decimal_places, self.scientific_notation = (
            detect_precision_and_notation(os.path.join(directory_path, base_file_name))
        )

        # Read CSV with Pandas
        self.df = pd.read_csv(os.path.join(directory_path, base_file_name))

        self.title(f"Reorder CSV Columns - {base_file_name}")
        self.geometry("1024x960")

        # Scrollbar
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Instructions and buttons at the top
        self.instructions = tk.Label(
            scrollable_frame,
            text="Click to select a Column and press Enter to reorder. Select and press 'd' to delete.\nPress 'm' to manually select range. Press 'l' to edit rows. Press Ctrl+S to save. Press Ctrl+Z to undo.\nPress Esc to save and exit.",
            font=("default", 14),
        )
        self.instructions.grid(row=0, column=0, columnspan=2, pady=10, sticky="n")

        # Frame for headers and listboxes with adjusted height
        self.header_frame = tk.Frame(scrollable_frame)
        self.header_frame.grid(
            row=1, column=0, columnspan=3, pady=10, padx=10, sticky="nsew"
        )

        self.number_label = tk.Label(
            self.header_frame, text="Number", font=("default", 12, "bold")
        )
        self.number_label.grid(row=0, column=0, padx=(10, 5), pady=(10, 0))

        self.name_label = tk.Label(
            self.header_frame, text="Name", font=("default", 12, "bold")
        )
        self.name_label.grid(row=0, column=1, padx=(5, 10), pady=(10, 0))

        self.shape_label = tk.Label(
            self.header_frame, text=f"Shape: {self.df.shape}", font=("default", 12)
        )
        self.shape_label.grid(row=0, column=2, padx=(5, 10), pady=(10, 0))

        self.order_listbox = tk.Listbox(
            self.header_frame, selectmode=tk.MULTIPLE, width=5, height=30
        )
        self.order_listbox.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="ns")

        self.header_listbox = tk.Listbox(
            self.header_frame, selectmode=tk.MULTIPLE, width=50, height=30
        )
        self.header_listbox.grid(row=1, column=1, padx=(5, 10), pady=10, sticky="ns")

        self.update_listbox()
        self.update_shape_label()

        # Buttons for unit conversion and reference system modification
        button_frame = tk.Frame(scrollable_frame)
        button_frame.grid(row=1, column=3, pady=10, padx=10, sticky="ns")

        self.convert_button = tk.Button(
            button_frame, text="Convert Units", command=self.convert_units
        )
        self.convert_button.grid(row=0, column=0, padx=5, pady=5, sticky="n")

        self.modify_labref_button = tk.Button(
            button_frame, text="Modify Lab Ref System", command=self.modify_labref
        )
        self.modify_labref_button.grid(row=1, column=0, padx=5, pady=5, sticky="n")

        # Adding buttons for merge and stack
        merge_button = tk.Button(button_frame, text="Merge CSV", command=self.merge_csv)
        merge_button.grid(row=2, column=0, padx=5, pady=5, sticky="n")

        stack_button = tk.Button(
            button_frame, text="Stack/Append CSV", command=self.stack_csv
        )
        stack_button.grid(row=3, column=0, padx=5, pady=5, sticky="n")

        # Button to convert MediaPipe CSVs
        mediapipe_button = tk.Button(
            button_frame,
            text="Convert MediaPipe to vailá",
            command=lambda: batch_convert_mediapipe(self.directory_path),
        )
        mediapipe_button.grid(row=4, column=0, padx=5, pady=5, sticky="n")

        self.bind("<Return>", self.swap_columns)
        self.bind("d", self.delete_columns)
        self.bind("m", self.manual_selection)
        self.bind("l", self.edit_rows)
        self.bind("<Control-s>", self.save_intermediate)
        self.bind("<Control-z>", self.undo)
        self.bind("<Escape>", self.save_and_exit)

    def update_listbox(self):
        self.header_listbox.delete(0, tk.END)
        self.order_listbox.delete(0, tk.END)
        for i, header in enumerate(self.current_order):
            self.order_listbox.insert(tk.END, i + 1)
            self.header_listbox.insert(tk.END, f"{i + 1}: {header}")

    def update_shape_label(self):
        shape = (self.df.shape[0], len(self.current_order))
        self.shape_label.config(text=f"Shape: {shape}")

    def save_state(self):
        self.history.append(self.current_order.copy())

    def undo(self, event):
        if self.history:
            self.current_order = self.history.pop()
            self.update_listbox()
            self.update_shape_label()

    def swap_columns(self, event):
        selected_idx = self.header_listbox.curselection()
        if selected_idx:
            self.save_state()
            selected_idx = list(selected_idx)
            new_position = simpledialog.askinteger(
                "Swap Column",
                f"Enter the new starting position for '{self.current_order[selected_idx[0]]}' (1-{len(self.current_order)}):",
                minvalue=1,
                maxvalue=len(self.current_order),
            )
            if new_position is not None:
                new_position -= 1  # Adjust for 0-based indexing
                selected_headers = [self.current_order[i] for i in selected_idx]
                for i in sorted(selected_idx, reverse=True):
                    del self.current_order[i]
                for i, header in enumerate(selected_headers):
                    self.current_order.insert(new_position + i, header)
                self.update_listbox()
                self.update_shape_label()

    def delete_columns(self, event=None):
        selected_idx = self.header_listbox.curselection()
        if selected_idx:
            self.save_state()
            selected_idx = list(selected_idx)
            for i in sorted(selected_idx, reverse=True):
                del self.current_order[i]
            self.update_listbox()
            self.update_shape_label()

    def manual_selection(self, event=None):
        selection_range = simpledialog.askstring(
            "Manual Selection",
            "Enter the selection range in the format 'start:end' (1-based indexing):",
        )
        if selection_range:
            try:
                self.save_state()
                start, end = map(int, selection_range.split(":"))
                start -= 1  # Adjust for 0-based indexing
                end -= 1  # Adjust for 0-based indexing
                self.header_listbox.selection_clear(0, tk.END)
                for i in range(start, end + 1):
                    self.header_listbox.selection_set(i)

                new_position = simpledialog.askinteger(
                    "Insert Position",
                    f"Enter the position to insert the selected columns (1-{len(self.current_order)}):",
                    minvalue=1,
                    maxvalue=len(self.current_order),
                )
                if new_position is not None:
                    new_position -= 1  # Adjust for 0-based indexing
                    selected_headers = [
                        self.current_order[i] for i in range(start, end + 1)
                    ]
                    for i in range(start, end + 1):
                        del self.current_order[start]
                    for i, header in enumerate(selected_headers):
                        self.current_order.insert(new_position + i, header)
                    self.update_listbox()
                    self.update_shape_label()
            except ValueError:
                messagebox.showerror(
                    "Error", "Invalid format. Please enter the range as 'start:end'."
                )

    def edit_rows(self, event):
        row_range = simpledialog.askstring(
            "Edit Rows",
            "Enter the row range to edit in the format 'start:end' (1-based indexing):",
        )
        if row_range:
            try:
                start, end = map(int, row_range.split(":"))
                start -= 1  # Adjust for 0-based indexing
                end -= 1  # Adjust for 0-based indexing
                row_shape = (
                    end - start + 1,
                    len(self.current_order),
                )  # Assuming all rows are equal length

                row_edit_window = tk.Toplevel(self)
                row_edit_window.title("Edit Rows")
                row_edit_window.geometry("600x400")

                shape_label = tk.Label(
                    row_edit_window, text=f"Shape: {row_shape}", font=("default", 12)
                )
                shape_label.pack(pady=10)

                button_frame = tk.Frame(row_edit_window)
                button_frame.pack(pady=10)

                save_button = tk.Button(
                    button_frame,
                    text="Save range",
                    command=lambda: self.save_row_range(row_edit_window, start, end),
                )
                save_button.grid(row=0, column=0, padx=5)

                delete_button = tk.Button(
                    button_frame,
                    text="Delete range",
                    command=lambda: self.delete_row_range(row_edit_window, start, end),
                )
                delete_button.grid(row=0, column=1, padx=5)

                close_button = tk.Button(
                    button_frame, text="Close", command=row_edit_window.destroy
                )
                close_button.grid(row=1, column=0, columnspan=2, pady=10)

            except ValueError:
                messagebox.showerror(
                    "Error", "Invalid format. Please enter the range as 'start:end'."
                )

    def save_row_range(self, window, start, end):
        print(f"Saving row range from {start} to {end}")
        row_df = self.df.iloc[start : end + 1]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.file_names[0]))[0]
        new_file_name = f"{base_name}_{timestamp}_selrows_{start + 1}_{end + 1}.csv"
        new_file_path = os.path.join(self.rearranged_path, new_file_name)

        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        save_dataframe(row_df, new_file_path, row_df.columns, self.max_decimal_places)
        messagebox.showinfo("Success", f"Selected row range saved to {new_file_path}")
        window.destroy()

    def delete_row_range(self, window, start, end):
        print(f"Deleting row range from {start} to {end}")
        deleted_df = self.df.drop(self.df.index[start : end + 1])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.file_names[0]))[0]
        new_file_name = f"{base_name}_{timestamp}_delrows_{start + 1}_{end + 1}.csv"
        new_file_path = os.path.join(self.rearranged_path, new_file_name)

        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        save_dataframe(
            deleted_df, new_file_path, deleted_df.columns, self.max_decimal_places
        )
        messagebox.showinfo("Success", f"Deleted row range saved to {new_file_path}")
        window.destroy()

    def save_intermediate(self, event=None):
        try:
            print("Starting intermediate save")
            if not os.path.exists(self.rearranged_path):
                os.makedirs(self.rearranged_path)

            max_decimal_places = simpledialog.askinteger(
                "Decimal Places",
                "Enter the number of decimal places for saving:",
                initialvalue=self.max_decimal_places,
            )
            if max_decimal_places is None:
                max_decimal_places = self.max_decimal_places

            for file_name in self.file_names:
                file_path = os.path.join(self.directory_path, file_name)
                reshapedata(
                    file_path,
                    self.current_order,
                    self.rearranged_path,
                    "",
                    max_decimal_places,
                )
            messagebox.showinfo(
                "Success",
                f"Intermediate save completed. Files are saved in {self.rearranged_path} with a timestamp.",
            )
        except Exception as e:
            print(f"Error during intermediate save: {e}")

    def save_and_exit(self, event=None):
        try:
            print("Starting save and exit")
            if not os.path.exists(self.rearranged_path):
                os.makedirs(self.rearranged_path)

            max_decimal_places = simpledialog.askinteger(
                "Decimal Places",
                "Enter the number of decimal places for saving:",
                initialvalue=self.max_decimal_places,
            )
            if max_decimal_places is None:
                max_decimal_places = self.max_decimal_places

            for file_name in self.file_names:
                file_path = os.path.join(self.directory_path, file_name)
                reshapedata(
                    file_path,
                    self.current_order,
                    self.rearranged_path,
                    "_final",
                    max_decimal_places,
                )
            messagebox.showinfo(
                "Success",
                f"Reordering completed for all files. Final files are saved in {self.rearranged_path} with '_final' in the name.",
            )
            self.destroy()
        except Exception as e:
            print(f"Error during save and exit: {e}")

    def convert_units(self):
        convert_window = tk.Toplevel(self)
        convert_window.title("Convert Units")
        convert_window.geometry("800x720")  # Adjusted height for better visibility

        unit_options = list(CONVERSIONS.keys())

        # Current Unit
        current_unit_label = tk.Label(convert_window, text="Current Unit:")
        current_unit_label.grid(row=0, column=0, pady=5, padx=10)
        self.current_unit_var = tk.StringVar()
        self.current_unit_listbox = tk.Listbox(
            convert_window,
            selectmode=tk.SINGLE,
            width=20,
            height=27,
            exportselection=False,
        )  # Adjusted height
        self.current_unit_listbox.grid(row=1, column=0, pady=5, padx=10)
        for unit in unit_options:
            self.current_unit_listbox.insert(tk.END, unit)

        # Spacer
        spacer_label = tk.Label(convert_window, text="")
        spacer_label.grid(row=0, column=1, rowspan=4, padx=20)  # Adjusted spacing

        # Target Unit
        target_unit_label = tk.Label(convert_window, text="Target Unit:")
        target_unit_label.grid(row=0, column=2, pady=5, padx=10)
        self.target_unit_var = tk.StringVar()
        self.target_unit_listbox = tk.Listbox(
            convert_window,
            selectmode=tk.SINGLE,
            width=20,
            height=27,
            exportselection=False,
        )  # Adjusted height
        self.target_unit_listbox.grid(row=1, column=2, pady=5, padx=10)
        for unit in unit_options:
            self.target_unit_listbox.insert(tk.END, unit)

        # Time column checkbox
        time_column_label = tk.Label(convert_window, text="Is the first column time?")
        time_column_label.grid(row=2, column=0, columnspan=3, pady=5)
        self.time_column_var = tk.BooleanVar()
        time_column_check = tk.Checkbutton(
            convert_window, variable=self.time_column_var
        )
        time_column_check.grid(row=3, column=0, columnspan=3, pady=5)

        # Convert button
        convert_button = tk.Button(
            convert_window,
            text="Convert",
            command=lambda: self.apply_conversion(convert_window),
        )
        convert_button.grid(row=4, column=0, columnspan=3, pady=10)

        # Adjust column weights to ensure centering
        for i in range(3):
            convert_window.grid_columnconfigure(i, weight=1)

        convert_window.transient(self)
        convert_window.grab_set()
        self.wait_window(convert_window)

    def apply_conversion(self, convert_window):
        current_unit = self.current_unit_listbox.get(tk.ACTIVE)
        target_unit = self.target_unit_listbox.get(tk.ACTIVE)
        time_column = self.time_column_var.get()

        if current_unit == target_unit:
            messagebox.showinfo(
                "Information", "Selected units are the same. No conversion is needed."
            )
            return

        conversion_factor = CONVERSIONS[target_unit][0] / CONVERSIONS[current_unit][0]

        # Iterate over all CSV files in the directory
        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        for file_name in self.file_names:
            file_path = os.path.join(self.directory_path, file_name)
            df = pd.read_csv(file_path)

            if time_column:
                df.iloc[:, 1:] = df.iloc[:, 1:] * conversion_factor
            else:
                df = df * conversion_factor

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(file_name)[0]
            new_file_name = f"{base_name}_unit_{CONVERSIONS[current_unit][1]}_{CONVERSIONS[target_unit][1]}_{timestamp}.csv"
            new_file_path = os.path.join(self.rearranged_path, new_file_name)

            max_decimal_places, _ = detect_precision_and_notation(file_path)
            save_dataframe(df, new_file_path, df.columns, max_decimal_places)

        messagebox.showinfo(
            "Success",
            f"Conversion from {current_unit} to {target_unit} applied successfully to all files. Files saved in {self.rearranged_path}.",
        )
        convert_window.destroy()
        self.update_shape_label()

    def modify_labref(self):
        modify_window = tk.Toplevel(self)
        modify_window.title("Modify Lab Ref System")
        modify_window.geometry("800x400")  # Increasing window width for better text fit

        # Current Option
        current_option_label = tk.Label(modify_window, text="Current Option:")
        current_option_label.grid(row=0, column=0, pady=5, padx=10)
        self.current_option_var = tk.StringVar()
        self.current_option_listbox = tk.Listbox(
            modify_window,
            selectmode=tk.SINGLE,
            width=70,
            height=4,
            exportselection=False,
        )  # Increasing listbox width
        self.current_option_listbox.grid(row=1, column=0, pady=5, padx=10)
        options = [
            "(A) Rotate 180 degrees in Z axis",
            "(B) Rotate 90 degrees clockwise in Z axis",
            "(C) Rotate 90 degrees counter clockwise in Z axis",
            "Custom: Angles in degrees for x, y, and z axes (e.g., [20, -90, 100]) - NOT Working yet",
        ]
        for option in options:
            self.current_option_listbox.insert(tk.END, option)

        # Spacer
        spacer_label = tk.Label(modify_window, text="")
        spacer_label.grid(row=0, column=1, rowspan=4, padx=20)

        # Modify button
        modify_button = tk.Button(
            modify_window,
            text="Modify",
            command=lambda: self.apply_modify(modify_window),
        )
        modify_button.grid(row=2, column=0, columnspan=3, pady=10)

        # Adjust column weights to ensure centering
        for i in range(2):
            modify_window.grid_columnconfigure(i, weight=1)

        modify_window.transient(self)
        modify_window.grab_set()
        self.wait_window(modify_window)

    def apply_modify(self, modify_window):
        selected_text = self.current_option_listbox.get(tk.ACTIVE)
        if selected_text.startswith("Custom"):
            custom_angles = simpledialog.askstring(
                "Input",
                "Enter rotation angles in degrees for x, y, and z axes (e.g., [20, -90, 100]):",
            )
            if custom_angles:
                selected_text = custom_angles
        else:
            selected_text = selected_text.split(" ")[0].strip("()")

        if not selected_text:
            messagebox.showinfo(
                "Information", "No option selected. Please select an option."
            )
            return

        modifylabref.run_modify_labref(selected_text, self.directory_path)
        messagebox.showinfo("Success", "Modification applied successfully.")
        modify_window.destroy()

    def merge_csv(self):
        base_file = select_file("Select the base CSV file")
        merge_file = select_file("Select the CSV file to merge")

        if base_file and merge_file:
            insert_position = simpledialog.askinteger(
                "Insert Position",
                "Enter the column position to insert the merge file (leave empty for last column):",
                initialvalue=None,
                minvalue=1,
            )
            save_path = filedialog.asksaveasfilename(
                title="Save Merged File As",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if save_path:
                merge_csv_files(base_file, merge_file, save_path, insert_position)

    def stack_csv(self):
        base_file = select_file("Select the base CSV file")
        stack_file = select_file("Select the CSV file to stack/append")

        if base_file and stack_file:
            stack_position = simpledialog.askstring(
                "Stack Position",
                "Enter 'start' to stack at the beginning, or leave empty to stack at the end:",
                initialvalue="end",
            )
            save_path = filedialog.asksaveasfilename(
                title="Save Stacked File As",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
            )
            if save_path:
                stack_csv_files(base_file, stack_file, save_path, stack_position)


def rearrange_data_in_directory():
    root = tk.Tk()
    root.withdraw()

    selected_directory = filedialog.askdirectory(
        title="Select Directory Containing CSV Files"
    )
    if not selected_directory:
        print("No directory selected.")
        return

    file_names = sorted(
        [f for f in os.listdir(selected_directory) if f.endswith(".csv")]
    )

    if not file_names:
        print("No CSV files found in the directory.")
        return

    example_file = os.path.join(selected_directory, file_names[0])
    original_headers = get_headers(example_file)

    print("-" * 80)
    print("Original Headers:")
    print(original_headers)
    print("-" * 80)
    print("")

    app = ColumnReorderGUI(original_headers, file_names, selected_directory)
    app.mainloop()


if __name__ == "__main__":
    rearrange_data_in_directory()
