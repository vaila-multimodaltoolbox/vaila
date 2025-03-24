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
from rich import print
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Scrollbar
from datetime import datetime
from vaila import modifylabref
from vaila.mergestack import select_file, merge_csv_files, stack_csv_files
from vaila.standardize_header import standardize_header
from vaila.dlc2vaila import batch_convert_dlc

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
    "meters_per_second_squared": (1, "m/s²"),
    "gravitational_force": (1 / 9.80665, "g"),
}


# Function to detect scientific notation and maximum precision in the data
def detect_precision_and_notation(file_path):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    float_format = f"%.{max_decimal_places}f"
    df.to_csv(file_path, index=False, columns=columns, float_format=float_format)


# Function to get headers from the CSV file
def get_headers(file_path):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    try:
        df = pd.read_csv(file_path, nrows=0)
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading headers from {file_path}: {e}")
        return []


# Function to reshape data
def reshapedata(file_path, new_order, save_directory, suffix, max_decimal_places):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    df = pd.read_csv(file_path)

    # Adjust the "frame" column to start from 0
    df.iloc[:, 0] = df.iloc[:, 0]

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
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    if not directory_path:
        print("No directory selected.")
        return

    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # Create a new directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(
        directory_path, f"Convert_MediaPipe_to_vaila_{timestamp}"
    )

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for file_name in csv_files:
        file_path = os.path.join(directory_path, file_name)
        convert_mediapipe_to_pixel_format(file_path, save_directory)

    print(f"All files have been converted and saved to {save_directory}")
    messagebox.showinfo(
        "Success", f"All files have been converted and saved to {save_directory}"
    )


def convert_kinovea_to_vaila(file_path, save_directory):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Generate the correct header based on the number of trajectories
        num_columns = len(df.columns) - 1  # Exclude the first column (frame/time)
        num_points = num_columns // 2  # Each point has X and Y
        correct_header = ["frame"] + [
            f"p{i+1}_{coord}" for i in range(num_points) for coord in ["x", "y"]
        ]

        # Assign the correct header to the DataFrame
        df.columns = correct_header[: len(df.columns)]

        # Format the columns with adjusted decimal precision
        float_format = "%.1f"
        if "frame" in df.columns:
            df.iloc[:, 1:] = df.iloc[:, 1:].astype(float).round(1)
        else:
            df = df.astype(float).round(1)

        # Save the corrected file in Vailá format
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_name = f"{base_name}_vaila.csv"
        new_file_path = os.path.join(save_directory, new_file_name)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        df.to_csv(new_file_path, index=False, float_format=float_format)
        print(f"Corrected and converted Kinovea data saved to {new_file_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")


def batch_convert_kinovea(directory_path):
    """
    Processes all files in the Kinovea format within a directory,
    converts them to the Vailá format, and saves the output in a subdirectory.
    """
    if not directory_path:
        print("No directory selected.")
        return

    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the directory.")
        messagebox.showwarning("No Files Found", "No CSV files found in the directory.")
        return

    # Create a subdirectory to save the converted files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(
        directory_path, f"Convert_Kinovea_to_vaila_{timestamp}"
    )
    os.makedirs(save_directory, exist_ok=True)

    converted_files = []
    errors = []
    for file_name in csv_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            # Call the conversion function
            convert_kinovea_to_vaila(file_path, save_directory)
            converted_files.append(file_name)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            errors.append((file_name, str(e)))

    # Final summary message
    if converted_files:
        success_message = (
            f"Conversion completed for {len(converted_files)} file(s).\n"
            f"Files have been saved in: {save_directory}"
        )
        print(success_message)
        if errors:
            error_message = (
                f"\nHowever, there were errors with {len(errors)} file(s):\n"
                + "\n".join(f"{name}: {error}" for name, error in errors)
            )
            success_message += error_message
        messagebox.showinfo("Batch Conversion Completed", success_message)
    elif errors:
        error_message = f"All files failed to convert.\nErrors:\n" + "\n".join(
            f"{name}: {error}" for name, error in errors
        )
        print(error_message)
        messagebox.showerror("Batch Conversion Failed", error_message)


class ColumnReorderGUI(tk.Tk):
    def __init__(self, original_headers, file_names, directory_path):
        super().__init__()
        self.original_headers = original_headers
        self.current_order = original_headers.copy()
        self.file_names = file_names
        self.directory_path = directory_path
        self.rearranged_path = os.path.join(directory_path, "data_rearranged")
        self.history = []

        # Verifique se não há arquivos CSV e simule um CSV vazio
        if self.file_names == ["Empty"]:
            print("No CSV files found. Simulating an empty CSV file.")
            self.df = pd.DataFrame(columns=["Column1", "Column2", "Column3"])
            self.file_names = ["Simulated_Empty_File.csv"]
            self.max_decimal_places = 2
            self.scientific_notation = False
        else:
            base_file_name = file_names[0]
            full_path = os.path.join(directory_path, base_file_name)

            try:
                # Primeira tentativa de ler o arquivo
                self.df = pd.read_csv(full_path)
                self.max_decimal_places, self.scientific_notation = (
                    detect_precision_and_notation(full_path)
                )
            except pd.errors.ParserError:
                # Se houver erro de parsing, standardize o header e encerre a GUI
                print("Parser error detected. Standardizing header...")
                self.withdraw()
                standardize_header()
                self.quit()  # Encerra a GUI atual
                return

        # Em vez de um tamanho fixo, define o tamanho relativo à tela
        self.title(f"Reorder CSV Columns - {self.file_names[0]}")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.geometry(f"{window_width}x{window_height}")

        self.setup_gui()

    def setup_gui(self):
        # Function to set up the GUI elements
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

        self.instructions = tk.Label(
            scrollable_frame,
            text="Click to select a Column and press Enter to reorder. Select and press 'd' to delete.\nPress 'm' to manually select range. Press 'l' to edit rows. Press Ctrl+S to save. Press Ctrl+Z to undo.\nPress Esc to save and exit.",
            font=("default", 10),
        )
        self.instructions.grid(row=0, column=0, columnspan=3, pady=10, sticky="n")

        self.header_frame = tk.Frame(scrollable_frame)
        self.header_frame.grid(
            row=1, column=0, columnspan=2, pady=10, padx=10, sticky="nsew"
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

        # Adjust the button frame to be placed next to the column names listbox
        button_frame = tk.Frame(self.header_frame)
        button_frame.grid(row=1, column=2, padx=10, pady=10, sticky="ns")

        self.convert_button = tk.Button(
            button_frame, text="Convert Units", command=self.convert_units
        )
        self.convert_button.grid(row=0, column=0, padx=5, pady=5, sticky="n")

        self.modify_labref_button = tk.Button(
            button_frame, text="Modify Lab Ref System", command=self.modify_labref
        )
        self.modify_labref_button.grid(row=1, column=0, padx=5, pady=5, sticky="n")

        merge_button = tk.Button(button_frame, text="Merge CSV", command=self.merge_csv)
        merge_button.grid(row=2, column=0, padx=5, pady=5, sticky="n")

        stack_button = tk.Button(
            button_frame, text="Stack/Append CSV", command=self.stack_csv
        )
        stack_button.grid(row=3, column=0, padx=5, pady=5, sticky="n")

        # Add new YOLO Tracker button
        yolo_tracker_button = tk.Button(
            button_frame,
            text="Convert YOLO Tracker to vailá",
            command=lambda: batch_convert_yolo_tracker(self.directory_path),
        )
        yolo_tracker_button.grid(row=4, column=0, padx=5, pady=5, sticky="n")

        # Shift the existing buttons down one row
        mediapipe_button = tk.Button(
            button_frame,
            text="Convert MediaPipe to vailá",
            command=lambda: batch_convert_mediapipe(self.directory_path),
        )
        mediapipe_button.grid(row=5, column=0, padx=5, pady=5, sticky="n")

        dvideo_button = tk.Button(
            button_frame,
            text="Convert Dvideo to vailá",
            command=lambda: batch_convert_dvideo(self.directory_path),
        )
        dvideo_button.grid(row=6, column=0, padx=5, pady=5, sticky="n")

        dlc_button = tk.Button(
            button_frame,
            text="Convert DLC to vailá",
            command=lambda: batch_convert_dlc(self.directory_path),
        )
        dlc_button.grid(row=7, column=0, padx=5, pady=5, sticky="n")

        standardize_button = tk.Button(
            button_frame, text="Standardize Header", command=standardize_header
        )
        standardize_button.grid(row=8, column=0, padx=5, pady=5, sticky="n")

        kinovea_button = tk.Button(
            button_frame,
            text="Convert Kinovea to vailá",
            command=lambda: batch_convert_kinovea(self.directory_path),
        )
        kinovea_button.grid(row=9, column=0, padx=5, pady=5, sticky="n")

        # Bind events to functions
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

        # Se não houver arquivos, exibir "Empty"
        if self.file_names == ["Empty"]:
            self.header_listbox.insert(tk.END, "No CSV files found.")
        else:
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
                start -= 1  # Ajusta para índices 0-based
                end -= 1
                row_shape = (end - start + 1, len(self.current_order))

                row_edit_window = tk.Toplevel(self)
                row_edit_window.title("Edit Rows")
                # Calcula o tamanho ideal baseado na resolução da tela
                screen_width = row_edit_window.winfo_screenwidth()
                screen_height = row_edit_window.winfo_screenheight()
                # Define 80% do tamanho da tela, mas não excede 600x400 se a tela permitir
                window_width = min(600, int(screen_width * 0.8))
                window_height = min(400, int(screen_height * 0.8))
                row_edit_window.geometry(f"{window_width}x{window_height}")

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
        # Ensure the directory for rearranged data exists
        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        print(f"Saving row range from {start + 1} to {end + 1}")

        # Walk through the original directory, ignoring files with timestamps
        for root, _, files in os.walk(self.directory_path):
            for file_name in files:
                if file_name.endswith(".csv") and not self.is_file_already_processed(
                    file_name
                ):
                    file_path = os.path.join(root, file_name)
                    df = pd.read_csv(file_path)
                    row_df = df.iloc[
                        start : end + 1
                    ]  # The end + 1 ensures 'end' is included
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = os.path.splitext(os.path.basename(file_name))[0]
                    new_file_name = (
                        f"{base_name}_{timestamp}_selrows_{start + 1}_{end + 1}.csv"
                    )
                    new_file_path = os.path.join(self.rearranged_path, new_file_name)

                    # Save only the file with the timestamp
                    save_dataframe(
                        row_df, new_file_path, row_df.columns, self.max_decimal_places
                    )
                    print(f"Selected row range saved to {new_file_path}")

        messagebox.showinfo(
            "Success",
            f"Selected row range ({start + 1} - {end + 1}) saved for all files.",
        )

        # Force GUI to update and ensure all operations are processed
        self.update_idletasks()
        window.destroy()

    def delete_row_range(self, window, start, end):
        # Ensure the directory for rearranged data exists
        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        print(f"Deleting row range from {start + 1} to {end + 1}")

        # Walk through the original directory, ignoring files with timestamps
        for root, _, files in os.walk(self.directory_path):
            for file_name in files:
                if file_name.endswith(".csv") and not self.is_file_already_processed(
                    file_name
                ):
                    file_path = os.path.join(root, file_name)
                    df = pd.read_csv(file_path)
                    deleted_df = df.drop(
                        df.index[start : end + 1]
                    )  # The end + 1 ensures 'end' is included in deletion
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = os.path.splitext(os.path.basename(file_name))[0]
                    new_file_name = (
                        f"{base_name}_{timestamp}_delrows_{start + 1}_{end + 1}.csv"
                    )
                    new_file_path = os.path.join(self.rearranged_path, new_file_name)

                    # Save only the file with the timestamp
                    save_dataframe(
                        deleted_df,
                        new_file_path,
                        deleted_df.columns,
                        self.max_decimal_places,
                    )
                    print(f"Deleted row range saved to {new_file_path}")

        messagebox.showinfo(
            "Success",
            f"Deleted row range ({start + 1} - {end + 1}) saved for all files.",
        )

        # Force GUI to update and ensure all operations are processed
        self.update_idletasks()
        window.destroy()

    def is_file_already_processed(self, file_name):
        """
        Helper function to check if a file already contains a timestamp,
        meaning it has already been processed.
        """
        # A regular expression or simple check can be used to see if the file name contains a timestamp
        # For example, checking if the file contains '_selrows_' or '_delrows_' which indicates it's a processed file
        return "_selrows_" in file_name or "_delrows_" in file_name

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

        # Special case for m/s2 and g conversion
        if current_unit in [
            "meters_per_second_squared",
            "gravitational_force",
        ] and target_unit in ["meters_per_second_squared", "gravitational_force"]:
            if (
                current_unit == "meters_per_second_squared"
                and target_unit == "gravitational_force"
            ):
                conversion_factor = 1 / 9.80665
            elif (
                current_unit == "gravitational_force"
                and target_unit == "meters_per_second_squared"
            ):
                conversion_factor = 9.80665
            else:
                conversion_factor = (
                    CONVERSIONS[target_unit][0] / CONVERSIONS[current_unit][0]
                )
        else:
            conversion_factor = (
                CONVERSIONS[target_unit][0] / CONVERSIONS[current_unit][0]
            )

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
            # Adjust the unit names to avoid special characters and upper case
            current_unit_name = (
                CONVERSIONS[current_unit][1].replace("/", "").replace("²", "2").lower()
            )
            target_unit_name = (
                CONVERSIONS[target_unit][1].replace("/", "").replace("²", "2").lower()
            )
            new_file_name = f"{base_name}_unit_{current_unit_name}_{target_unit_name}_{timestamp}.csv"
            new_file_path = os.path.join(self.rearranged_path, new_file_name)

            # Ensure the directory exists before saving
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))

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


def convert_dvideo_to_vaila(file_path, save_directory):
    try:
        # Read the .dat file with space-separated values using raw string literal for regex separator
        df = pd.read_csv(file_path, sep=r"\s+", header=None)

        # Calculate the number of points (assuming the first column is 'frame')
        num_points = (df.shape[1] - 1) // 2

        # Create headers in the format: frame, p1_x, p1_y, p2_x, p2_y, ..., pN_x, pN_y
        headers = ["frame"] + [
            f"p{i + 1}_x" if j % 2 == 0 else f"p{i + 1}_y"
            for i in range(num_points)
            for j in range(2)
        ]
        df.columns = headers

        # Ensure the output formatting with a fixed number of decimal places
        float_format = "%.6f"

        # Create the output file path
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_name = f"{base_name}_vaila.csv"
        new_file_path = os.path.join(save_directory, new_file_name)

        # Ensure save directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the dataframe as CSV with specified float format and headers
        df.to_csv(new_file_path, index=False, float_format=float_format)
        print(f"Converted Dvideo data saved to {new_file_path}")

    except Exception as e:
        print(f"Error converting {file_path}: {e}")


def batch_convert_dvideo(directory_path):
    if not directory_path:
        print("No directory selected.")
        return

    dat_files = [f for f in os.listdir(directory_path) if f.endswith(".dat")]
    if not dat_files:
        print("No .dat files found in the directory.")
        return

    # Create a new directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(
        directory_path, f"Convert_Dvideo_to_vaila_{timestamp}"
    )

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Convert each .dat file and save in the new directory
    for file_name in dat_files:
        file_path = os.path.join(directory_path, file_name)
        convert_dvideo_to_vaila(file_path, save_directory)

    print(f"All files have been converted and saved to {save_directory}")


def convert_yolo_tracker_to_pixel_format(tracker_file, save_directory=None):
    """
    Converts YOLO tracker output format (all_persons_positions.csv) to a format compatible with getpixelvideo.py.

    The YOLO tracker format has columns: Frame,ID_1,X_1,Y_1,RGB_1,ID_2,X_2,Y_2,RGB_2,...
    The pixel format needs columns: frame,p1_x,p1_y,p2_x,p2_y,...

    Args:
        tracker_file: Path to the YOLO tracker output file (all_persons_positions.csv)
        save_directory: Directory to save the converted file (if None, saves in the same directory)

    Returns:
        Path to the converted file
    """
    print(f"Converting YOLO tracker file: {tracker_file}")

    try:
        # Read the YOLO tracker file
        df = pd.read_csv(tracker_file)

        # Create a new DataFrame for the pixel format
        new_df = pd.DataFrame()
        new_df["frame"] = df["Frame"]

        # Find all unique person IDs in the file
        person_columns = [col for col in df.columns if col.startswith("ID_")]

        # Create dictionary to store person data by ID
        # This preserves the ID numbering in the output file
        person_data = {}

        # Extract each person's data
        for col in person_columns:
            person_id = col.split("_")[1]  # Extract the ID number
            x_col = f"X_{person_id}"
            y_col = f"Y_{person_id}"

            if x_col in df.columns and y_col in df.columns:
                # Store this person's data with their original ID
                person_data[int(person_id)] = {"x": df[x_col], "y": df[y_col]}

        # Sort the person IDs to ensure consistent ordering
        sorted_ids = sorted(person_data.keys())

        # Add each person's coordinates to the new DataFrame
        for idx, person_id in enumerate(sorted_ids):
            marker_idx = idx + 1  # Marker indices start at 1
            new_df[f"p{marker_idx}_x"] = person_data[person_id]["x"]
            new_df[f"p{marker_idx}_y"] = person_data[person_id]["y"]

        # Determine the output file path
        if save_directory is None:
            save_directory = os.path.dirname(tracker_file)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        base_name = os.path.splitext(os.path.basename(tracker_file))[0]
        output_file = os.path.join(save_directory, f"{base_name}_pixelformat.csv")

        # Save the new DataFrame
        new_df.to_csv(output_file, index=False)
        print(f"Converted file saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"Error converting YOLO tracker file: {e}")
        return None


def batch_convert_yolo_tracker(directory_path=None):
    """
    Batch converts all YOLO tracker files in a directory to pixel format.

    Args:
        directory_path: Directory containing YOLO tracker files
    """
    if directory_path is None:
        # Open a file dialog to select the directory
        directory_path = filedialog.askdirectory(
            title="Select Directory with YOLO Tracker Files"
        )

    if not directory_path:
        print("No directory selected.")
        return

    # Find all potential YOLO tracker files
    potential_files = [
        f
        for f in os.listdir(directory_path)
        if f.endswith(".csv") and ("all_persons_positions" in f or "person" in f)
    ]

    if not potential_files:
        print("No YOLO tracker files found in the directory.")
        messagebox.showwarning(
            "No Files Found", "No YOLO tracker files found in the directory."
        )
        return

    # Create a timestamp directory to save the converted files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(directory_path, f"Convert_YOLO_to_Pixel_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)

    # Convert each file
    converted_files = []
    errors = []

    for file_name in potential_files:
        try:
            file_path = os.path.join(directory_path, file_name)
            result_path = convert_yolo_tracker_to_pixel_format(
                file_path, save_directory
            )
            if result_path:
                converted_files.append(file_name)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            errors.append((file_name, str(e)))

    # Show summary message
    if converted_files:
        success_message = (
            f"Conversion completed for {len(converted_files)} file(s).\n"
            f"Files have been saved in: {save_directory}"
        )
        print(success_message)
        if errors:
            error_message = (
                f"\nHowever, there were errors with {len(errors)} file(s):\n"
                + "\n".join(f"{name}: {error}" for name, error in errors)
            )
            success_message += error_message
        messagebox.showinfo("YOLO Tracker Conversion Completed", success_message)
    elif errors:
        error_message = f"All files failed to convert.\nErrors:\n" + "\n".join(
            f"{name}: {error}" for name, error in errors
        )
        print(error_message)
        messagebox.showerror("YOLO Tracker Conversion Failed", error_message)


def rearrange_data_in_directory():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    root = tk.Tk()
    root.withdraw()

    selected_directory = filedialog.askdirectory(
        title="Select Directory Containing CSV Files"
    )
    if not selected_directory:
        print("No directory selected.")
        return

    # Find all CSV files in the selected directory
    file_names = sorted(
        [f for f in os.listdir(selected_directory) if f.endswith(".csv")]
    )

    # If no CSV files are found, continue and open the GUI
    if not file_names:
        print("No CSV files found in the directory.")
        file_names = ["Empty"]
        original_headers = []
    else:
        # Get headers from the first CSV file if it exists
        example_file = os.path.join(selected_directory, file_names[0])
        try:
            original_headers = get_headers(example_file)
            print("-" * 80)
            print("Original Headers:")
            print(original_headers)
            print("-" * 80)
            print("")

            # Try to create the GUI
            app = ColumnReorderGUI(original_headers, file_names, selected_directory)
            app.mainloop()
        except pd.errors.ParserError:
            print("Parser error detected. Running header standardization...")
            standardize_header()
            # After standardize_header, restart the process
            rearrange_data_in_directory()
        except Exception as e:
            print(f"Unexpected error: {e}")
            return


if __name__ == "__main__":
    rearrange_data_in_directory()
