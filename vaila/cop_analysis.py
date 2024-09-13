"""
Module: cop_analysis.py
Description: This module provides tools for analyzing Center of Pressure (CoP) data from force plate measurements. 
It includes functions for data filtering, selecting relevant data headers, calculating confidence ellipses, 
and plotting CoP pathways with confidence intervals.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.0
Date: 2024-09-12

References:
- Jirsa, V. K., & Kelso, J. A. S. (2004). Coordination dynamics: Issues and trends. In Coordination: Neural, Behavioral and Social Dynamics (pp. 39-56). Springer.
- Liu, S., Zhai, P., Wang, L., Qiu, J., Liu, L., & Wang, H. (2021). Application of entropy-based measures in postural control system research. Frontiers in Bioengineering and Biotechnology, 9, 776326. https://doi.org/10.3389/fbioe.2021.776326
- GitHub Repository: Code Descriptors Postural Control. https://github.com/Jythen/code_descriptors_postural_control/blob/main/main.py
- Liu, S., Zhai, P., Wang, L., Qiu, J., Liu, L., & Wang, H. (2021). A Review of Entropy-Based Methods in Postural Control Evaluation. Frontiers in Neuroscience, 15, 776326. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8623280/

Changelog:
- Version 1.0 (2024-09-12):
  - Initial implementation of CoP data analysis functions.
  - Included flexible Butterworth filtering for data processing.
  - Added functions for selecting data headers and plotting CoP pathways.
  - Integrated GUI components for user input and file handling.
  - Added conversion utilities for different measurement units.

Usage:
- To run the CoP analysis, use the `main` function:

  if __name__ == "__main__":
      main()

- The module provides functions to:
  - Read CSV files:
    read_csv_full(filename)
    Reads the full CSV file and returns its content as a DataFrame.
  
  - Filter CoP data:
    butter_filter(data, fs, filter_type='low', cutoff=10, order=4)
    Applies a flexible Butterworth filter to the data.
  
  - Select Data Headers:
    select_nine_headers(file_path)
    Displays a GUI to select nine headers for force plate data analysis.
  
  - Compute Confidence Ellipses:
    plot_ellipse_pca(data, confidence=0.95)
    Calculates the ellipse using PCA with a specified confidence level.
  
  - Plot CoP Pathways:
    plot_cop_pathway_with_ellipse(cop_x, cop_y, area, angle, ellipse_data, title, output_path)
    Plots the CoP pathway along with the 95% confidence ellipse and saves the figure.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import (
    Tk, Toplevel, Canvas, Scrollbar, Frame, Button, Checkbutton,
    BooleanVar, messagebox, filedialog, simpledialog
)
from .ellipse import plot_ellipse_pca, plot_cop_pathway_with_ellipse
from .filter_utils import butter_filter  # Import the updated filter function

def convert_to_cm(data, unit):
    """Converts the data to centimeters based on the provided unit."""
    conversion_factors = {
        "m": 100, "mm": 0.1, "ft": 30.48, "in": 2.54, "yd": 91.44, "cm": 1
    }
    if unit not in conversion_factors:
        raise ValueError(f"Unsupported unit '{unit}'. Please use m, mm, ft, in, yd, or cm.")
    return data * conversion_factors[unit]

def read_csv_full(filename):
    """Reads the full CSV file."""
    try:
        data = pd.read_csv(filename, delimiter=",")
        return data
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {str(e)}")

def select_nine_headers(file_path):
    """Displays a GUI to select nine headers for force plate data analysis."""

    def get_csv_headers(file_path):
        """Reads the headers from a CSV file."""
        df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [
            header for header, var in zip(headers, header_vars) if var.get()
        ]
        selection_window.quit()
        selection_window.destroy()

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select Nine Headers for Force Plate Data")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.8)}"
    )

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    header_vars = [BooleanVar() for _ in headers]
    num_columns = 8  # Number of columns for header labels

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, anchor="center")

    Button(btn_frame, text="Select All", command=select_all).pack(side="top", pady=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(
        side="top", pady=5
    )
    Button(btn_frame, text="Confirm", command=on_select).pack(side="top", pady=5)

    selection_window.mainloop()

    if len(selected_headers) != 9:
        messagebox.showinfo("Info", "Please select exactly nine headers for analysis.")
        return None, None

    selected_data = df[selected_headers]
    return selected_headers, selected_data

def analyze_data_2d(data, output_dir, file_name, fs, plate_width, plate_height, timestamp):
    """Analyzes selected 2D data and saves the results."""

    # Apply the updated Butterworth filter to the data
    try:
        data = butter_filter(data, fs, filter_type='low', cutoff=10, order=4, padding=True)
    except ValueError as e:
        print(f"Filtering error: {e}")
        messagebox.showerror("Filtering Error", f"An error occurred during filtering:\n{e}")
        return

    # Defining variables for the analysis
    cop_x = data[:, 6]     # Center of Pressure in the X direction (Cx)
    cop_y = data[:, 7]     # Center of Pressure in the Y direction (Cy)

    # Create time vector based on the sampling frequency
    time = np.linspace(0, (len(cop_x) - 1) / fs, len(cop_x))

    # Calculate the confidence ellipse for Cx and Cy
    area, angle, bounds, ellipse_data = plot_ellipse_pca(data[:, [6, 7]], confidence=0.95)

    # Define the output path for the plots
    output_path = os.path.join(output_dir, f"{file_name}_cop_analysis_{timestamp}")

    # Delegate the plotting to the ellipse module
    plot_cop_pathway_with_ellipse(cop_x, cop_y, area, angle, ellipse_data, file_name, output_path)

def main():
    """Function to run the CoP analysis"""
    root = Tk()
    root.withdraw()  # Hides the main Tkinter window

    # Request input and output directories
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    if not input_dir:
        print("No input directory selected.")
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected.")
        return

    # Request sampling frequency and plate dimensions
    fs = simpledialog.askfloat(
        "Signal Frequency",
        "Enter the sampling frequency (Fs) in Hz:",
        initialvalue=1000.0,
    )
    if not fs:
        print("No valid frequency provided.")
        return

    plate_width = simpledialog.askfloat(
        "Force Plate Width",
        "Enter the width of the force plate in cm:",
        initialvalue=46.4,
    )
    plate_height = simpledialog.askfloat(
        "Force Plate Height",
        "Enter the height of the force plate in cm:",
        initialvalue=50.75,
    )

    if not plate_width or not plate_height:
        print("Invalid force plate dimensions provided.")
        return

    # Ask user for the unit of measurement
    unit = simpledialog.askstring(
        "Unit of Measurement",
        "Enter the unit of measurement for the CoP data (e.g., cm, m, mm, ft, in, yd):",
        initialvalue="mm"
    )
    if not unit:
        print("No unit provided.")
        return

    # Select sample file
    sample_file_path = filedialog.askopenfilename(
        title="Select a Sample CSV File", filetypes=[("CSV files", "*.csv")]
    )
    if not sample_file_path:
        print("No sample file selected.")
        return

    # Select nine headers for force plate data analysis
    selected_headers, _ = select_nine_headers(sample_file_path)
    if not selected_headers:
        print("No valid headers selected.")
        return

    # Create timestamp for output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Create main output directory using timestamp
    main_output_dir = os.path.join(output_dir, f"vaila_cop_balance_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Process each CSV file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_dir, file_name)
            data = read_csv_full(file_path)[selected_headers].to_numpy()

            # Convert data to cm if necessary
            try:
                data = convert_to_cm(data, unit)
            except ValueError as e:
                print(e)
                messagebox.showerror("Unit Conversion Error", f"An error occurred during unit conversion:\n{e}")
                return

            # Create output directory for the current file without the '.csv' extension
            file_name_without_extension = os.path.splitext(file_name)[0]
            file_output_dir = os.path.join(main_output_dir, file_name_without_extension)
            os.makedirs(file_output_dir, exist_ok=True)

            # Analyze and save results
            analyze_data_2d(
                data,
                file_output_dir,
                file_name_without_extension,  # Use file name without extension
                fs,
                plate_width,
                plate_height,
                timestamp,
            )

    # Inform the user that the analysis is complete
    print("Analysis complete.")
    messagebox.showinfo("Information", "Analysis complete! The window will close in 10 seconds.")
    root.after(10000, root.destroy)  # Wait for 10 seconds and then destroy the window
    root.mainloop()

if __name__ == "__main__":
    main()

