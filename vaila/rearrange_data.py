"""
Project: vailá Multimodal Toolbox
Script: rearrange_data.py - CSV Data Rearrangement and Processing Tool

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 08 Oct 2024
Update Date: 15 Oct 2025
Version: 0.0.7

Description:
    This script provides tools for rearranging and processing CSV data files.
    It includes functions for:
    - Reordering columns.
    - Merging and stacking CSV files.
    - Converting MediaPipe data to a format compatible with 'getpixelvideo.py'.
    - Detecting precision and scientific notation in the data.
    - Converting units between various metric systems.
    - Modifying lab reference systems.
    - Saving the second half of each CSV file.

Usage:
    Run the script from the command line:
        python rearrange_data.py

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - tkinter

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.7: Added custom math operation feature with NumPy, Pandas, and SciPy support
    - v0.0.6: Added functionality to save the second half of each CSV file
    - v0.0.4: Added functionality to save the second half of each CSV file
    - v0.0.3: Added batch convert MediaPipe CSV files functionality
    - v0.0.2: Added automatic directory creation for saving converted MediaPipe data
    - v0.0.1: Initial version with core functionalities for CSV reordering and unit conversion
"""

import gc
import os
import pathlib
import tkinter as tk
from datetime import datetime
from tkinter import Scrollbar, filedialog, messagebox, simpledialog

import numpy as np
import pandas as pd
from rich import print

from vaila import modifylabref
from vaila.dlc2vaila import batch_convert_dlc
from vaila.mergestack import merge_csv_files, select_file, stack_csv_files
from vaila.standardize_header import standardize_header

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


def detect_column_precision_detailed(file_path):
    """
    Detect precision for each column individually.

    Returns:
        dict: Column index -> number of decimal places
    """
    try:
        # Read as string to preserve exact formatting
        df_str = pd.read_csv(file_path, dtype=str)

        column_precision = {}

        for col_idx, _col_name in enumerate(df_str.columns):
            max_decimal_places = 0

            # Check multiple rows to get representative precision
            for row_idx in range(min(10, len(df_str))):
                value_str = str(df_str.iloc[row_idx, col_idx])

                if value_str != "nan" and "." in value_str:
                    decimal_part = value_str.split(".")[1]
                    max_decimal_places = max(max_decimal_places, len(decimal_part))

            column_precision[col_idx] = max_decimal_places

        return column_precision
    except Exception as e:
        print(f"Error detecting precision: {e}")
        return {}


def save_dataframe_with_precision(df, file_path, column_precision):
    """
    Save DataFrame with specific precision for each column.

    Args:
        df: DataFrame to save
        file_path: Output file path
        column_precision: Dict with column index -> decimal places
    """
    try:
        # Apply formatting to each column
        formatted_data = df.copy()

        for i, col in enumerate(formatted_data.columns):
            precision = column_precision.get(i, 6)  # Default to 6 if not found

            if precision == 0:
                # Format as integer
                formatted_data[col] = formatted_data[col].apply(
                    lambda x: f"{int(x)}" if pd.notna(x) else ""
                )
            else:
                # Format with specific decimal places
                formatted_data[col] = formatted_data[col].apply(
                    lambda x: f"{x:.{precision}f}" if pd.notna(x) else ""
                )

        # Save the formatted data
        formatted_data.to_csv(file_path, index=False)
    except Exception as e:
        print(f"Error saving with precision: {e}")
        # Fallback to regular save
        df.to_csv(file_path, index=False, float_format="%.6f")


# Function to detect scientific notation and maximum precision in the data
def detect_precision_and_notation(file_path):
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

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
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

    float_format = f"%.{max_decimal_places}f"
    df.to_csv(file_path, index=False, columns=columns, float_format=float_format)


# Function to get headers from the CSV file
def get_headers(file_path):
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

    try:
        df = pd.read_csv(file_path, nrows=0)
        return df.columns.tolist()
    except Exception as e:
        print(f"Error reading headers from {file_path}: {e}")
        return []


# Function to reshape data
def reshapedata(file_path, new_order, save_directory, suffix, column_precision=None):
    try:
        print(f"Starting reshapedata for {file_path}")
        headers = get_headers(file_path)

        # Read CSV with Pandas
        df = pd.read_csv(file_path)

        print("Headers read from the CSV:")
        print(headers)

        print("First 5 rows of the original DataFrame:")
        print(df.head())

        # Only keep columns from new_order that exist in headers
        existing_cols = [col for col in new_order if col in headers]
        df_reordered = df[existing_cols]

        print("First 5 rows of the reordered DataFrame:")
        print(df_reordered.head())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_name = f"{base_name}_{timestamp}{suffix}.csv"
        new_file_path = os.path.join(save_directory, new_file_name)

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Detect precision if not provided
        if column_precision is None:
            column_precision = detect_column_precision_detailed(file_path)

        # Adjust precision mapping for reordered columns
        reordered_precision = {}
        for new_idx, col in enumerate(existing_cols):
            old_idx = headers.index(col)
            reordered_precision[new_idx] = column_precision.get(old_idx, 6)

        save_dataframe_with_precision(df_reordered, new_file_path, reordered_precision)

        print(f"Reordered data saved to {new_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# Function to convert MediaPipe data to the format compatible with getpixelvideo.py
def convert_mediapipe_to_pixel_format(file_path, save_directory):
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

    df = pd.read_csv(file_path)

    # Adjust the "frame" column to start from 0
    df.iloc[:, 0] = df.iloc[:, 0]

    # Create the new DataFrame with the "frame" column and pX_x, pX_y coordinates
    new_df = pd.DataFrame()
    new_df["frame"] = df.iloc[:, 0]  # Use the first column as "frame"

    columns = df.columns[1:]  # Ignore the first column, which we already used for "frame"
    for i in range(0, len(columns), 3):
        if i + 1 < len(columns):
            x_col = columns[i]
            y_col = columns[i + 1]
            new_df[f"p{i // 3 + 1}_x"] = df[x_col]
            new_df[f"p{i // 3 + 1}_y"] = df[y_col]

    # Save the new CSV file in the desired format
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    new_file_name = f"{base_name}_converted.csv"
    new_file_path = os.path.join(save_directory, new_file_name)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    new_df.to_csv(new_file_path, index=False)
    print(f"Converted MediaPipe data saved to {new_file_path}")


# Function to batch convert all MediaPipe CSV files in a directory
def batch_convert_mediapipe(directory_path):
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

    if not directory_path:
        print("No directory selected.")
        return

    csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # Create a new directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(directory_path, f"Convert_MediaPipe_to_vaila_{timestamp}")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for file_name in csv_files:
        file_path = os.path.join(directory_path, file_name)
        convert_mediapipe_to_pixel_format(file_path, save_directory)

    print(f"All files have been converted and saved to {save_directory}")


def convert_kinovea_to_vaila(file_path, save_directory):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Generate the correct header based on the number of trajectories
        num_columns = len(df.columns) - 1  # Exclude the first column (frame/time)
        num_points = num_columns // 2  # Each point has X and Y
        correct_header = ["frame"] + [
            f"p{i + 1}_{coord}" for i in range(num_points) for coord in ["x", "y"]
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
    save_directory = os.path.join(directory_path, f"Convert_Kinovea_to_vaila_{timestamp}")
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
            print(error_message)
            success_message += error_message
        messagebox.showinfo("Batch Conversion Completed", success_message)
    elif errors:
        error_message = "All files failed to convert.\nErrors:\n" + "\n".join(
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

        # Verificar o tamanho do arquivo antes de carregar
        if self.file_names == ["Empty"]:
            self.is_large_file = False
            self.setup_empty_gui()
        else:
            base_file_name = file_names[0]
            full_path = os.path.join(directory_path, base_file_name)
            file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
            self.is_large_file = (
                file_size_mb > 100
            )  # Arquivos maiores que 100MB são considerados grandes

            if self.is_large_file:
                print(f"Large file detected ({file_size_mb:.2f} MB). Loading in simplified mode...")
                self.setup_large_file_gui(full_path)
            else:
                self.setup_normal_gui(full_path)

        # Configure the window
        self.title(f"Reorder CSV Columns - {self.file_names[0]}")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.geometry(f"{window_width}x{window_height}")

    def setup_empty_gui(self):
        """Configuration for empty CSV files"""
        print("No CSV files found. Simulating an empty CSV file.")
        self.df = pd.DataFrame(columns=pd.Index(["Column1", "Column2", "Column3"]))
        self.file_names = ["Simulated_Empty_File.csv"]
        self.max_decimal_places = 2
        self.scientific_notation = False
        self.setup_gui()

    def setup_large_file_gui(self, file_path):
        """Configuration for large files"""
        try:
            # Ler apenas o cabeçalho e as primeiras linhas
            print("Reading file headers...")
            self.df = pd.read_csv(file_path, nrows=5)  # Ler apenas 5 linhas para exemplo
            self.max_decimal_places = 3  # Valor padrão para arquivos grandes
            self.scientific_notation = False

            # Mostrar aviso sobre modo simplificado
            messagebox.showinfo(
                "Large File Mode",
                "File is too large for full preview. Running in simplified mode.\n"
                "Only headers and first 5 rows will be shown.",
            )

            self.setup_gui(is_large_file=True)

        except Exception as e:
            print(f"Error loading large file: {e}")
            self.setup_empty_gui()

    def setup_normal_gui(self, file_path):
        """Configuration for small files"""
        try:
            self.df = pd.read_csv(file_path)
            self.max_decimal_places, self.scientific_notation = detect_precision_and_notation(
                file_path
            )
            self.setup_gui(is_large_file=False)
        except pd.errors.ParserError:
            print("Parser error detected. Standardizing header...")
            self.withdraw()
            standardize_header()
            self.quit()
        except Exception as e:
            print(f"Error loading file: {e}")
            self.setup_empty_gui()

    def setup_gui(self, is_large_file=False):
        """Configuration for the GUI with large file mode"""
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

        # Add warning for large files
        if is_large_file:
            warning_label = tk.Label(
                scrollable_frame,
                text="Large File Mode: Limited Preview Available",
                font=("default", 12, "bold"),
                fg="red",
            )
            warning_label.grid(row=0, column=0, columnspan=3, pady=5)

        # Normal instructions
        instructions_text = (
            "Click to select a Column and press Enter to reorder. Select and press 'd' to delete.\n"
            "Press 'm' to manually select range. Press 'l' to edit rows. Press Ctrl+S to save. "
            "Press Ctrl+Z to undo.\nPress Esc to save and exit."
        )
        if is_large_file:
            instructions_text += "\nLarge File Mode: Changes will be applied to the entire file."

        self.instructions = tk.Label(scrollable_frame, text=instructions_text, font=("default", 10))
        self.instructions.grid(row=1, column=0, columnspan=3, pady=10, sticky="n")

        # Rest of the GUI configuration remains the same
        self.header_frame = tk.Frame(scrollable_frame)
        self.header_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")

        # Labels for number and name
        self.number_label = tk.Label(self.header_frame, text="Number", font=("default", 12, "bold"))
        self.number_label.grid(row=0, column=0, padx=(10, 5), pady=(10, 0))

        self.name_label = tk.Label(self.header_frame, text="Name", font=("default", 12, "bold"))
        self.name_label.grid(row=0, column=1, padx=(5, 10), pady=(10, 0))

        # Show shape with additional information for large files
        if is_large_file:
            shape_text = (
                f"Shape: {self.df.shape[0]} rows (showing first 5) x {self.df.shape[1]} columns"
            )
        else:
            shape_text = f"Shape: {self.df.shape}"

        self.shape_label = tk.Label(self.header_frame, text=shape_text, font=("default", 12))
        self.shape_label.grid(row=0, column=2, padx=(5, 10), pady=(10, 0))

        # Listboxes
        self.order_listbox = tk.Listbox(
            self.header_frame, selectmode=tk.MULTIPLE, width=5, height=30
        )
        self.order_listbox.grid(row=1, column=0, padx=(10, 5), pady=10, sticky="ns")

        self.header_listbox = tk.Listbox(
            self.header_frame, selectmode=tk.MULTIPLE, width=50, height=30
        )
        self.header_listbox.grid(row=1, column=1, padx=(5, 10), pady=10, sticky="ns")

        # Update listboxes
        self.update_listbox()

        # Add button frame
        button_frame = tk.Frame(self.header_frame)
        button_frame.grid(row=1, column=2, padx=10, pady=10, sticky="ns")

        # Add all buttons
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

        stack_button = tk.Button(button_frame, text="Stack/Append CSV", command=self.stack_csv)
        stack_button.grid(row=3, column=0, padx=5, pady=5, sticky="n")

        # Add new YOLO Tracker button
        yolo_tracker_button = tk.Button(
            button_frame,
            text="Convert YOLO Tracker to vailá",
            command=lambda: batch_convert_yolo_tracker(self.directory_path),
        )
        yolo_tracker_button.grid(row=4, column=0, padx=5, pady=5, sticky="n")

        # MediaPipe button
        mediapipe_button = tk.Button(
            button_frame,
            text="Convert MediaPipe to vailá",
            command=lambda: batch_convert_mediapipe(self.directory_path),
        )
        mediapipe_button.grid(row=5, column=0, padx=5, pady=5, sticky="n")

        # DVideo button
        dvideo_button = tk.Button(
            button_frame,
            text="Convert Dvideo to vailá",
            command=lambda: batch_convert_dvideo(self.directory_path),
        )
        dvideo_button.grid(row=6, column=0, padx=5, pady=5, sticky="n")

        # DLC button
        dlc_button = tk.Button(
            button_frame,
            text="Convert DLC to vailá",
            command=lambda: batch_convert_dlc(self.directory_path),
        )
        dlc_button.grid(row=7, column=0, padx=5, pady=5, sticky="n")

        # Standardize button
        standardize_button = tk.Button(
            button_frame, text="Standardize Header", command=standardize_header
        )
        standardize_button.grid(row=8, column=0, padx=5, pady=5, sticky="n")

        # Kinovea button
        kinovea_button = tk.Button(
            button_frame,
            text="Convert Kinovea to vailá",
            command=lambda: batch_convert_kinovea(self.directory_path),
        )
        kinovea_button.grid(row=9, column=0, padx=5, pady=5, sticky="n")

        # Save 2nd Half button
        second_half_button = tk.Button(
            button_frame, text="Save 2nd Half CSV", command=self.save_second_half
        )
        second_half_button.grid(row=10, column=0, padx=5, pady=5, sticky="n")

        # Reset index column 0 button
        reset_index_button = tk.Button(
            button_frame, text="Reset Index Col 0", command=self.reset_index_column_0
        )
        reset_index_button.grid(row=11, column=0, padx=5, pady=5, sticky="n")

        # Custom Math Operation button
        custom_math_button = tk.Button(
            button_frame,
            text="Custom Math Operation",
            command=self.custom_math_operation,
        )
        custom_math_button.grid(row=12, column=0, padx=5, pady=5, sticky="n")

        # Configure bindings
        self.setup_bindings()

    def setup_bindings(self):
        """Configure all keyboard shortcuts"""
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

        # If there are no files, display "Empty"
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
                    selected_headers = [self.current_order[i] for i in range(start, end + 1)]
                    for _ in range(start, end + 1):
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
                end -= 1
                row_shape = (end - start + 1, len(self.current_order))

                row_edit_window = tk.Toplevel(self)
                row_edit_window.title("Edit Rows")
                # Calculate the ideal size based on the screen resolution
                screen_width = row_edit_window.winfo_screenwidth()
                screen_height = row_edit_window.winfo_screenheight()
                # Define 80% of the screen size, but not exceed 600x400 if the screen allows
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
                if file_name.endswith(".csv") and not self.is_file_already_processed(file_name):
                    file_path = os.path.join(root, file_name)
                    df = pd.read_csv(file_path)
                    row_df = df.iloc[start : end + 1]  # The end + 1 ensures 'end' is included
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = os.path.splitext(os.path.basename(file_name))[0]
                    new_file_name = f"{base_name}_{timestamp}_selrows_{start + 1}_{end + 1}.csv"
                    new_file_path = os.path.join(self.rearranged_path, new_file_name)

                    # Save only the file with the timestamp
                    save_dataframe(row_df, new_file_path, row_df.columns, self.max_decimal_places)
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
                if file_name.endswith(".csv") and not self.is_file_already_processed(file_name):
                    file_path = os.path.join(root, file_name)
                    df = pd.read_csv(file_path)
                    deleted_df = df.drop(
                        df.index[start : end + 1].tolist()
                    )  # Convert slice to list for drop method
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = os.path.splitext(os.path.basename(file_name))[0]
                    new_file_name = f"{base_name}_{timestamp}_delrows_{start + 1}_{end + 1}.csv"
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
                initialvalue=(
                    int(self.max_decimal_places)
                    if isinstance(self.max_decimal_places, (int, float))
                    else 6
                ),
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
                    dict.fromkeys(range(len(self.current_order)), max_decimal_places),
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
                initialvalue=(
                    int(self.max_decimal_places)
                    if isinstance(self.max_decimal_places, (int, float))
                    else 6
                ),
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
                    dict.fromkeys(range(len(self.current_order)), max_decimal_places),
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
            height=25,  # Reduced height to make room for radio buttons
            exportselection=False,
        )
        self.current_unit_listbox.grid(row=1, column=0, pady=5, padx=10)
        for unit in unit_options:
            self.current_unit_listbox.insert(tk.END, unit)

        # Spacer
        spacer_label = tk.Label(convert_window, text="")
        spacer_label.grid(row=0, column=1, rowspan=4, padx=20)

        # Target Unit
        target_unit_label = tk.Label(convert_window, text="Target Unit:")
        target_unit_label.grid(row=0, column=2, pady=5, padx=10)
        self.target_unit_var = tk.StringVar()
        self.target_unit_listbox = tk.Listbox(
            convert_window,
            selectmode=tk.SINGLE,
            width=20,
            height=25,  # Reduced height to make room for radio buttons
            exportselection=False,
        )
        self.target_unit_listbox.grid(row=1, column=2, pady=5, padx=10)
        for unit in unit_options:
            self.target_unit_listbox.insert(tk.END, unit)

        # Column conversion options using radiobuttons (more reliable than checkbox)
        conversion_frame = tk.LabelFrame(
            convert_window,
            text="Column Conversion Options",
            font=("default", 10, "bold"),
        )
        conversion_frame.grid(row=2, column=0, columnspan=3, pady=20, padx=20, sticky="ew")

        self.conversion_option = tk.StringVar(value="all_columns")  # Default to convert all columns

        # Radio button for converting all columns
        all_columns_radio = tk.Radiobutton(
            conversion_frame,
            text="Convert ALL columns (including first column)",
            variable=self.conversion_option,
            value="all_columns",
            font=("default", 10),
            bg="lightblue",
        )
        all_columns_radio.grid(row=0, column=0, pady=5, padx=10, sticky="w")

        # Radio button for ignoring first column
        ignore_first_radio = tk.Radiobutton(
            conversion_frame,
            text="Ignore FIRST column (time/frame numbers) - convert only data columns",
            variable=self.conversion_option,
            value="ignore_first",
            font=("default", 10),
            bg="lightyellow",
        )
        ignore_first_radio.grid(row=1, column=0, pady=5, padx=10, sticky="w")

        # Explanation label
        explanation_label = tk.Label(
            conversion_frame,
            text="Select 'Ignore FIRST column' if your first column contains time, frame numbers,\n"
            "or any other data that should NOT be converted with the unit conversion.",
            font=("default", 9),
            fg="darkblue",
        )
        explanation_label.grid(row=2, column=0, pady=10, padx=10, sticky="w")

        # Convert button
        convert_button = tk.Button(
            convert_window,
            text="Convert Units",
            command=lambda: self.apply_conversion(convert_window),
            font=("default", 12, "bold"),
            bg="lightgreen",
        )
        convert_button.grid(row=3, column=0, columnspan=3, pady=20)

        # Adjust column weights to ensure centering
        for i in range(3):
            convert_window.grid_columnconfigure(i, weight=1)

        convert_window.transient(self)
        convert_window.grab_set()
        self.wait_window(convert_window)

    def apply_conversion(self, convert_window):
        current_unit = self.current_unit_listbox.get(tk.ACTIVE)
        target_unit = self.target_unit_listbox.get(tk.ACTIVE)

        # Get the conversion option from radiobuttons (more reliable)
        conversion_choice = self.conversion_option.get()
        ignore_first_column = conversion_choice == "ignore_first"

        print(f"Conversion choice: {conversion_choice}")
        print(f"Ignore first column: {ignore_first_column}")

        if current_unit == target_unit:
            messagebox.showinfo(
                "Information", "Selected units are the same. No conversion is needed."
            )
            return

        # Validate selections
        if not current_unit or not target_unit:
            messagebox.showwarning(
                "Selection Required", "Please select both current and target units."
            )
            return

        # Calculate conversion factor
        if current_unit in [
            "meters_per_second_squared",
            "gravitational_force",
        ] and target_unit in ["meters_per_second_squared", "gravitational_force"]:
            if current_unit == "meters_per_second_squared" and target_unit == "gravitational_force":
                conversion_factor = 1 / 9.80665
            elif (
                current_unit == "gravitational_force" and target_unit == "meters_per_second_squared"
            ):
                conversion_factor = 9.80665
            else:
                conversion_factor = CONVERSIONS[target_unit][0] / CONVERSIONS[current_unit][0]
        else:
            conversion_factor = CONVERSIONS[target_unit][0] / CONVERSIONS[current_unit][0]

        # Iterate over all CSV files in the directory
        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        processed_files = 0
        for file_name in self.file_names:
            if file_name == "Empty":
                continue

            file_path = os.path.join(self.directory_path, file_name)
            df = pd.read_csv(file_path)

            # Detect original precision
            column_precision = detect_column_precision_detailed(file_path)

            # Create a copy for conversion
            df_converted = df.copy()

            if ignore_first_column:
                print(f"IGNORING first column (time/frame) for {file_name}")
                # Only convert columns from index 1 onwards (ignore first column)
                if len(df_converted.columns) > 1:
                    df_converted.iloc[:, 1:] = df_converted.iloc[:, 1:] * conversion_factor
                    print(
                        f"Applied conversion factor {conversion_factor} to columns 1-{len(df_converted.columns) - 1}"
                    )
                else:
                    print(f"Warning: {file_name} has only one column. No conversion applied.")
            else:
                print(f"CONVERTING ALL columns for {file_name}")
                # Convert all columns
                df_converted = df_converted * conversion_factor
                print(f"Applied conversion factor {conversion_factor} to all columns")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(file_name)[0]
            # Adjust the unit names to avoid special characters and upper case
            current_unit_name = (
                CONVERSIONS[current_unit][1].replace("/", "").replace("²", "2").lower()
            )
            target_unit_name = (
                CONVERSIONS[target_unit][1].replace("/", "").replace("²", "2").lower()
            )

            # Add clear indication in filename about what was converted
            conversion_suffix = "_FIRST_IGNORED" if ignore_first_column else "_ALL_CONVERTED"
            new_file_name = f"{base_name}_unit_{current_unit_name}_to_{target_unit_name}{conversion_suffix}_{timestamp}.csv"
            new_file_path = os.path.join(self.rearranged_path, new_file_name)

            # Ensure the directory exists before saving
            if not os.path.exists(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))

            # Save with original precision
            save_dataframe_with_precision(df_converted, new_file_path, column_precision)
            print(f"Saved converted file: {new_file_path}")
            processed_files += 1

        # Show detailed success message
        if processed_files > 0:
            success_message = f"SUCCESS! Converted {processed_files} file(s) from {current_unit} to {target_unit}."

            if ignore_first_column:
                success_message += "\n\nCONVERSION DETAILS:"
                success_message += "\n- FIRST column was IGNORED (preserved original values)"
                success_message += "\n- Only DATA columns were converted"
            else:
                success_message += "\n\nCONVERSION DETAILS:"
                success_message += "\n- ALL columns were converted"

            success_message += f"\n\nFiles saved in: {self.rearranged_path}"
            success_message += f"\nConversion factor applied: {conversion_factor}"

            messagebox.showinfo("Conversion Complete", success_message)
        else:
            messagebox.showwarning(
                "No Files Processed", "No valid CSV files were found to process."
            )

        convert_window.destroy()
        self.update_shape_label()

    def modify_labref(self):
        modify_window = tk.Toplevel(self)
        modify_window.title("Modify Lab Ref System")
        modify_window.geometry("800x500")  # Increased height for custom input

        # Current Option
        current_option_label = tk.Label(modify_window, text="Select Rotation Option:")
        current_option_label.grid(row=0, column=0, pady=5, padx=10)

        self.current_option_var = tk.StringVar()
        self.current_option_listbox = tk.Listbox(
            modify_window,
            selectmode=tk.SINGLE,
            width=70,
            height=5,
            exportselection=False,
        )
        self.current_option_listbox.grid(row=1, column=0, pady=5, padx=10)

        options = [
            "(A) Rotate 180 degrees in Z axis",
            "(B) Rotate 90 degrees clockwise in Z axis",
            "(C) Rotate 90 degrees counter clockwise in Z axis",
            "Custom: Enter angles in degrees for x, y, and z axes",
        ]
        for option in options:
            self.current_option_listbox.insert(tk.END, option)

        # Custom angles input frame (initially hidden)
        self.custom_frame = tk.Frame(modify_window)
        self.custom_frame.grid(row=2, column=0, pady=10, padx=10, sticky="ew")
        self.custom_frame.grid_remove()  # Hide initially

        custom_label = tk.Label(
            self.custom_frame,
            text="Custom angles (format: [x, y, z] or [x, y, z], xyz):",
        )
        custom_label.grid(row=0, column=0, pady=5, sticky="w")

        self.custom_entry = tk.Entry(self.custom_frame, width=50)
        self.custom_entry.grid(row=1, column=0, pady=5, sticky="ew")
        self.custom_entry.insert(0, "[0, -45, 0]")  # Example

        examples_label = tk.Label(
            self.custom_frame,
            text="Examples: [0, -45, 0] or [90, 0, 180], zyx",
            font=("default", 9),
        )
        examples_label.grid(row=2, column=0, pady=2, sticky="w")

        # Bind selection change to show/hide custom input
        self.current_option_listbox.bind("<<ListboxSelect>>", self.on_option_select)

        # Modify button
        modify_button = tk.Button(
            modify_window,
            text="Apply Rotation",
            command=lambda: self.apply_modify(modify_window),
        )
        modify_button.grid(row=3, column=0, pady=20)

        # Configure grid weights
        modify_window.grid_columnconfigure(0, weight=1)
        self.custom_frame.grid_columnconfigure(0, weight=1)

        modify_window.transient(self)
        modify_window.grab_set()
        self.wait_window(modify_window)

    def on_option_select(self, event):
        """Show/hide custom input based on selection"""
        try:
            selection = self.current_option_listbox.curselection()
            if selection:
                selected_text = self.current_option_listbox.get(selection[0])
                if selected_text.startswith("Custom"):
                    self.custom_frame.grid()  # Show custom input
                else:
                    self.custom_frame.grid_remove()  # Hide custom input
        except:
            pass

    def apply_modify(self, modify_window):
        """Apply the rotation to the CSV files in the directory."""
        # Print the directory and name of the script being executed
        print(f"Running script: {pathlib.Path(__file__).name}")
        print(f"Script directory: {pathlib.Path(__file__).parent}")

        try:
            selection = self.current_option_listbox.curselection()
            if not selection:
                messagebox.showinfo("Information", "Please select a rotation option.")
                return

            selected_text = self.current_option_listbox.get(selection[0])

            if selected_text.startswith("Custom"):
                # Get custom angles from entry widget
                custom_angles = self.custom_entry.get().strip()
                if not custom_angles:
                    messagebox.showinfo("Information", "Please enter custom angles.")
                    return
                selected_option = custom_angles
            else:
                # Extract letter from predefined options
                selected_option = selected_text.split(" ")[0].strip("()")

            print(f"Applying rotation with option: {selected_option}")

            # Call the modifylabref function
            modifylabref.run_modify_labref(selected_option, self.directory_path)

            messagebox.showinfo(
                "Success",
                "Rotation applied successfully! Check the 'rotated_files' folder.",
            )
            modify_window.destroy()

        except Exception as e:
            print(f"Error in apply_modify: {e}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            modify_window.destroy()

    def merge_csv(self):
        """Merge two CSV files into one."""
        # Print the directory and name of the script being executed
        print(f"Running script: {pathlib.Path(__file__).name}")
        print(f"Script directory: {pathlib.Path(__file__).parent}")

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
        """Stack/append two CSV files into one."""
        # Print the directory and name of the script being executed
        print(f"Running script: {pathlib.Path(__file__).name}")
        print(f"Script directory: {pathlib.Path(__file__).parent}")

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
                stack_csv_files(base_file, stack_file, save_path, stack_position or "end")

    def save_second_half(self):
        """Save the second half of each CSV file into `self.rearranged_path`."""
        # Print the directory and name of the script being executed
        print(f"Running script: {pathlib.Path(__file__).name}")
        print(f"Script directory: {pathlib.Path(__file__).parent}")

        # Cria o diretório de saída, se necessário
        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        # Processa cada arquivo CSV
        for file_name in self.file_names:
            if file_name == "Empty" or not file_name.lower().endswith(".csv"):
                continue
            file_path = os.path.join(self.directory_path, file_name)
            df = pd.read_csv(file_path)
            half_idx = len(df) // 2
            second_half = df.iloc[half_idx:].reset_index(drop=True)

            # Reset the first column (assumed to be frame numbers) to start from 0
            if len(second_half.columns) > 0:
                first_col = second_half.columns[0]
                second_half[first_col] = range(len(second_half))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(file_name)[0]
            new_name = f"{base}_{timestamp}_2ndhalf.csv"
            new_path = os.path.join(self.rearranged_path, new_name)

            second_half.to_csv(new_path, index=False)

        messagebox.showinfo("Success", f"Second half of CSV files saved in: {self.rearranged_path}")

    def reset_index_column_0(self):
        """
        Resets the index of the first column to start from 0 and go until the end of the data,
        saving a new file in the rearranged directory.
        """
        # Print the directory and name of the script being executed
        print(f"Running script: {pathlib.Path(__file__).name}")
        print(f"Script directory: {pathlib.Path(__file__).parent}")

        if not os.path.exists(self.rearranged_path):
            os.makedirs(self.rearranged_path)

        for file_name in self.file_names:
            if file_name == "Empty" or not file_name.lower().endswith(".csv"):
                continue
            file_path = os.path.join(self.directory_path, file_name)
            df = pd.read_csv(file_path)
            # Reset the first column
            if len(df.columns) > 0:
                first_col = df.columns[0]
                df[first_col] = range(len(df))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(file_name)[0]
            new_name = f"{base}_{timestamp}_resetidx.csv"
            new_path = os.path.join(self.rearranged_path, new_name)
            df.to_csv(new_path, index=False)
        messagebox.showinfo(
            "Success", f"Index of column 0 reset and saved in: {self.rearranged_path}"
        )

    def custom_math_operation(self):
        """
        Apply custom mathematical operations to selected columns.
        Allows user to enter any Python expression using numpy functions.
        """
        print(f"Running script: {pathlib.Path(__file__).name}")
        print(f"Script directory: {pathlib.Path(__file__).parent}")

        # Create custom operation window
        operation_window = tk.Toplevel(self)
        operation_window.title("Custom Math Operation")
        operation_window.geometry("900x650")

        # Title
        title_label = tk.Label(
            operation_window,
            text="Apply Custom Mathematical Operation",
            font=("default", 14, "bold"),
        )
        title_label.pack(pady=10)

        # Instructions
        instructions_text = (
            "Select column(s) and enter a mathematical expression.\n\n"
            "Available: NumPy, Pandas, SciPy functions and operators\n"
            "Use 'x' to represent the column value.\n\n"
            "Examples:\n"
            "  x * 2.5              → Multiply by 2.5\n"
            "  x / 1000             → Divide by 1000 (mm to m)\n"
            "  np.sqrt(x)           → Square root\n"
            "  x ** 2               → Square\n"
            "  np.abs(x)            → Absolute value\n"
            "  np.log(x)            → Natural logarithm\n"
            "  np.deg2rad(x)        → Degrees to radians\n"
            "  scipy.signal.medfilt(x, 5)  → Median filter\n"
            "  pd.Series(x).rolling(5).mean()  → Moving average\n"
        )
        instructions_label = tk.Label(
            operation_window,
            text=instructions_text,
            font=("default", 9),
            justify=tk.LEFT,
            bg="lightyellow",
        )
        instructions_label.pack(pady=10, padx=10, fill=tk.X)

        # Column selection frame
        column_frame = tk.LabelFrame(
            operation_window,
            text="Select Columns to Apply Operation",
            font=("default", 10, "bold"),
        )
        column_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        # Scrollbar and listbox for column selection
        scrollbar = tk.Scrollbar(column_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        column_listbox = tk.Listbox(
            column_frame,
            selectmode=tk.MULTIPLE,
            width=60,
            height=10,
            yscrollcommand=scrollbar.set,
        )
        column_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=column_listbox.yview)

        # Populate with current columns
        for i, header in enumerate(self.current_order):
            column_listbox.insert(tk.END, f"{i + 1}: {header}")

        # Select all / Clear all buttons
        select_button_frame = tk.Frame(operation_window)
        select_button_frame.pack(pady=5)

        select_all_btn = tk.Button(
            select_button_frame,
            text="Select All",
            command=lambda: column_listbox.select_set(0, tk.END),
        )
        select_all_btn.pack(side=tk.LEFT, padx=5)

        clear_all_btn = tk.Button(
            select_button_frame,
            text="Clear Selection",
            command=lambda: column_listbox.selection_clear(0, tk.END),
        )
        clear_all_btn.pack(side=tk.LEFT, padx=5)

        # Expression input frame
        expression_frame = tk.LabelFrame(
            operation_window,
            text="Mathematical Expression",
            font=("default", 10, "bold"),
        )
        expression_frame.pack(pady=10, padx=10, fill=tk.X)

        expression_label = tk.Label(
            expression_frame,
            text="Enter expression (use 'x' for column value):",
            font=("default", 10),
        )
        expression_label.pack(pady=5, padx=10, anchor=tk.W)

        expression_entry = tk.Entry(expression_frame, width=60, font=("default", 11))
        expression_entry.pack(pady=5, padx=10, fill=tk.X)
        expression_entry.insert(0, "x * 1.0")  # Default example
        expression_entry.focus()

        # Test button
        test_button = tk.Button(
            expression_frame,
            text="Test Expression",
            command=lambda: self.test_expression(expression_entry.get(), column_listbox),
            bg="lightblue",
        )
        test_button.pack(pady=5)

        # Apply and Cancel buttons
        button_frame = tk.Frame(operation_window)
        button_frame.pack(pady=15)

        apply_button = tk.Button(
            button_frame,
            text="Apply Operation",
            command=lambda: self.apply_math_operation(
                column_listbox, expression_entry.get(), operation_window
            ),
            font=("default", 11, "bold"),
            bg="lightgreen",
            width=15,
        )
        apply_button.pack(side=tk.LEFT, padx=10)

        cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=operation_window.destroy,
            font=("default", 11),
            width=15,
        )
        cancel_button.pack(side=tk.LEFT, padx=10)

        operation_window.transient(self)
        operation_window.grab_set()
        self.wait_window(operation_window)

    def test_expression(self, expression, column_listbox):
        """Test the mathematical expression with sample data"""
        try:
            # Import all necessary libraries first
            import numpy as np
            import pandas as pd

            # Get selected columns
            selected_idx = column_listbox.curselection()
            if not selected_idx:
                messagebox.showwarning("No Selection", "Please select at least one column to test.")
                return

            # Get first selected column for testing
            first_selected = selected_idx[0]
            column_name = self.current_order[first_selected]

            # Load sample data from first file
            if self.file_names[0] == "Empty":
                messagebox.showwarning("No Data", "No CSV files loaded for testing.")
                return

            file_path = os.path.join(self.directory_path, self.file_names[0])
            df = pd.read_csv(file_path, nrows=5)  # Read only first 5 rows for testing

            if column_name not in df.columns:
                messagebox.showerror("Error", f"Column '{column_name}' not found in file.")
                return

            # Get sample values
            sample_values = df[column_name].values

            result_values = []
            for val in sample_values:
                try:
                    result = eval(expression)
                    # Handle pandas Series results (e.g., from rolling operations)
                    if isinstance(result, pd.Series):
                        result = result.iloc[0] if len(result) > 0 else np.nan
                    result_values.append(result)
                except Exception as e:
                    messagebox.showerror(
                        "Expression Error",
                        f"Error evaluating expression with value {val}:\n{str(e)}\n\n"
                        f"Make sure to use 'np.' prefix for NumPy functions (e.g., np.sqrt(x))",
                    )
                    return

            # Show test results
            test_result = f"Testing expression: {expression}\n"
            test_result += f"Column: {column_name}\n\n"
            test_result += "Sample Results (first 5 rows):\n"
            test_result += "-" * 50 + "\n"
            for i, (orig, result) in enumerate(zip(sample_values, result_values, strict=False), 1):
                test_result += f"Row {i}: {orig} → {result}\n"

            messagebox.showinfo("Test Results", test_result)

        except Exception as e:
            messagebox.showerror("Test Error", f"Error testing expression:\n{str(e)}")

    def apply_math_operation(self, column_listbox, expression, operation_window):
        """Apply the mathematical operation to selected columns and save results"""
        try:
            # Get selected columns
            selected_idx = column_listbox.curselection()
            if not selected_idx:
                messagebox.showwarning("No Selection", "Please select at least one column.")
                return

            if not expression or expression.strip() == "":
                messagebox.showwarning("No Expression", "Please enter a mathematical expression.")
                return

            # Get selected column names
            selected_columns = [self.current_order[i] for i in selected_idx]

            # Confirm operation
            confirm_msg = (
                f"Apply operation '{expression}' to {len(selected_columns)} column(s):\n"
                f"{', '.join(selected_columns[:5])}"
            )
            if len(selected_columns) > 5:
                confirm_msg += f"\n... and {len(selected_columns) - 5} more"

            if not messagebox.askyesno("Confirm Operation", confirm_msg):
                return

            # Create output directory
            if not os.path.exists(self.rearranged_path):
                os.makedirs(self.rearranged_path)

            # Import all necessary libraries for eval
            import numpy as np
            import pandas as pd

            # Process each file
            processed_files = 0
            for file_name in self.file_names:
                if file_name == "Empty" or not file_name.lower().endswith(".csv"):
                    continue

                file_path = os.path.join(self.directory_path, file_name)
                df = pd.read_csv(file_path)

                # Detect original precision
                column_precision = detect_column_precision_detailed(file_path)

                # Create a copy for modification
                df_modified = df.copy()

                # Apply operation to each selected column
                for column_name in selected_columns:
                    if column_name in df_modified.columns:
                        print(f"Applying '{expression}' to column '{column_name}' in {file_name}")

                        # Apply the expression to the column
                        original_values = df_modified[column_name].values
                        new_values = []

                        for val in original_values:
                            try:
                                result = eval(expression)
                                # Handle pandas Series results
                                if isinstance(result, pd.Series):
                                    result = result.iloc[0] if len(result) > 0 else np.nan
                                new_values.append(result)
                            except Exception as e:
                                print(f"Error evaluating expression for value {val}: {e}")
                                new_values.append(np.nan)

                        df_modified[column_name] = new_values
                    else:
                        print(f"Warning: Column '{column_name}' not found in {file_name}")

                # Generate output filename - clean expression for safe filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(file_name)[0]

                # Create a safe filename by removing special characters
                expr_clean = expression.replace(" ", "")
                # Replace operators with words
                replacements = {
                    "*": "mult",
                    "/": "div",
                    "+": "plus",
                    "-": "minus",
                    "**": "pow",
                    "(": "",
                    ")": "",
                    ".": "_",
                    "[": "",
                    "]": "",
                    ",": "_",
                    "=": "eq",
                    "<": "lt",
                    ">": "gt",
                    "!": "not",
                }
                for old, new in replacements.items():
                    expr_clean = expr_clean.replace(old, new)

                # Limit length and ensure alphanumeric
                expr_clean = "".join(c for c in expr_clean if c.isalnum() or c == "_")[:30]

                new_file_name = f"{base_name}_{timestamp}_mathop_{expr_clean}.csv"
                new_file_path = os.path.join(self.rearranged_path, new_file_name)

                # Save with original precision
                save_dataframe_with_precision(df_modified, new_file_path, column_precision)
                print(f"Saved: {new_file_path}")
                processed_files += 1

            # Show success message
            if processed_files > 0:
                success_message = (
                    f"SUCCESS! Applied operation to {len(selected_columns)} column(s) in {processed_files} file(s).\n\n"
                    f"Operation: {expression}\n"
                    f"Columns: {', '.join(selected_columns[:3])}"
                )
                if len(selected_columns) > 3:
                    success_message += f"... and {len(selected_columns) - 3} more"
                success_message += f"\n\nFiles saved in: {self.rearranged_path}"

                messagebox.showinfo("Operation Complete", success_message)
            else:
                messagebox.showwarning("No Files Processed", "No valid CSV files were processed.")

            operation_window.destroy()

        except Exception as e:
            print(f"Error applying math operation: {e}")
            import traceback

            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


def convert_dvideo_to_vaila(file_path, save_directory):
    """Convert a Dvideo file to a Vaila CSV file."""
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

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
    """Batch convert all Dvideo files in a directory to Vaila CSV files."""
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

    if not directory_path:
        print("No directory selected.")
        return

    dat_files = [f for f in os.listdir(directory_path) if f.endswith(".dat")]
    if not dat_files:
        print("No .dat files found in the directory.")
        return

    # Create a new directory with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(directory_path, f"Convert_Dvideo_to_vaila_{timestamp}")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Convert each .dat file and save in the new directory
    for file_name in dat_files:
        file_path = os.path.join(directory_path, file_name)
        convert_dvideo_to_vaila(file_path, save_directory)

    print(f"All files have been converted and saved to {save_directory}")


def convert_yolo_tracker_to_pixel_format(tracker_file, save_directory=None, chunk_size=10000):
    """Convert a YOLO tracker file to a Vaila CSV file."""
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")

    print(f"Converting YOLO tracker file: {tracker_file}")

    try:
        # Configuração do diretório de saída
        if save_directory is None:
            save_directory = os.path.dirname(tracker_file)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        base_name = os.path.splitext(os.path.basename(tracker_file))[0]
        output_file = os.path.join(save_directory, f"{base_name}_pixelformat.csv")

        # Primeira passagem: identificar IDs únicos
        print("Analyzing file structure...")
        column_names = pd.read_csv(tracker_file, nrows=1).columns
        person_ids = sorted(
            [int(col.split("_")[1]) for col in column_names if col.startswith("ID_")]
        )

        # Processar o arquivo em chunks
        print(f"Processing file in chunks of {chunk_size} rows...")
        first_chunk = True

        # Usar chunked reading para processar o arquivo
        for chunk_number, chunk in enumerate(pd.read_csv(tracker_file, chunksize=chunk_size)):
            print(f"Processing chunk {chunk_number + 1}...")

            # Criar todos os dados em um dicionário primeiro
            data = {"frame": chunk["Frame"]}

            # Processar dados de cada pessoa
            for idx, person_id in enumerate(person_ids):
                x_col = f"X_{person_id}"
                y_col = f"Y_{person_id}"

                if x_col in chunk.columns and y_col in chunk.columns:
                    data[f"p{idx + 1}_x"] = chunk[x_col]
                    data[f"p{idx + 1}_y"] = chunk[y_col]
                else:
                    data[f"p{idx + 1}_x"] = np.nan
                    data[f"p{idx + 1}_y"] = np.nan

            # Criar o DataFrame de uma vez só com todas as colunas
            new_chunk = pd.DataFrame(data)

            # Escrever chunk no arquivo
            mode = "w" if first_chunk else "a"
            header = first_chunk
            new_chunk.to_csv(
                output_file, mode=mode, header=header, index=False, float_format="%.3f"
            )

            first_chunk = False

            # Liberar memória
            del new_chunk
            del data
            gc.collect()

        print(f"Conversion completed. File saved to: {output_file}")
        return output_file

    except Exception as e:
        print(f"Error converting YOLO tracker file: {e}")
        return None


def batch_convert_yolo_tracker(directory_path=None):
    """
    Batch converts all YOLO tracker files in a directory to pixel format.
    """
    if directory_path is None:
        directory_path = filedialog.askdirectory(title="Select Directory with YOLO Tracker Files")

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
        messagebox.showwarning("No Files Found", "No YOLO tracker files found in the directory.")
        return

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(directory_path, f"Convert_YOLO_to_Pixel_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)

    # Process each file
    converted_files = []
    errors = []

    # Ask for chunk size
    try:
        chunk_size = simpledialog.askinteger(
            "Chunk Size",
            "Enter chunk size for processing (larger files need smaller chunks):\n"
            + "Recommended:\n"
            + "- Small files: 10000\n"
            + "- Medium files: 5000\n"
            + "- Large files: 1000\n"
            + "- Very large files: 500",
            initialvalue=5000,
            minvalue=100,
            maxvalue=50000,
        )

        if chunk_size is None:
            chunk_size = 5000  # Default value if dialog is cancelled
    except:
        chunk_size = 5000  # Fallback value

    total_files = len(potential_files)
    for idx, file_name in enumerate(potential_files, 1):
        try:
            print(f"\nProcessing file {idx}/{total_files}: {file_name}")
            file_path = os.path.join(directory_path, file_name)

            # Get file size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"File size: {file_size_mb:.2f} MB")

            result_path = convert_yolo_tracker_to_pixel_format(
                file_path, save_directory, chunk_size=chunk_size
            )

            if result_path:
                converted_files.append(file_name)
                print(f"Successfully converted: {file_name}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            errors.append((file_name, str(e)))

        # Force garbage collection
        gc.collect()

    # Show summary
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
            print(error_message)
            success_message += error_message
        messagebox.showinfo("Conversion Complete", success_message)
    elif errors:
        error_message = "All files failed to convert.\nErrors:\n" + "\n".join(
            f"{name}: {error}" for name, error in errors
        )
        print(error_message)
        messagebox.showerror("Conversion Failed", error_message)


def rearrange_data_in_directory():
    # Print the directory and name of the script being executed
    print("Running script: rearrange_data")
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    root = tk.Tk()
    root.withdraw()

    selected_directory = filedialog.askdirectory(title="Select Directory Containing CSV Files")
    if not selected_directory:
        print("No directory selected.")
        return

    # Find all CSV files in the selected directory
    file_names = sorted([f for f in os.listdir(selected_directory) if f.endswith(".csv")])

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
