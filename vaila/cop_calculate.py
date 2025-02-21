"""
cop_calculate.py
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2025-02-18
Update: 2025-02-18
Version: 0.0.1
Python Version: 3.12.9
Description:
------------
This script allows users to select six (6) headers for force plate data analysis.
How to Run:
-----------
1. Ensure required dependencies are installed:
    pip install numpy pandas matplotlib scipy
2. Run the script:
    python cop_calculate.py
3. Follow on-screen prompts to select data and define parameters.
License: GNU GPLv3
--------
vailá Multimodal Toolbox
""" 
import os
import pandas as pd
import numpy as np
from tkinter import (
    Tk,
    filedialog,
    simpledialog,
    messagebox,
    Toplevel,
    Canvas,
    Scrollbar,
    Frame,
    Button,
    Checkbutton,
    BooleanVar,
)
import re


def read_csv_full(filename):
    """Reads the full CSV file."""
    try:
        data = pd.read_csv(filename, delimiter=",")
        return data
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {str(e)}")


def select_headers(file_path):
    """Displays a GUI to select six (6) headers for force plate data analysis."""
    def get_csv_headers(file_path):
        """Reads the headers from a CSV file."""
        df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [header for header, var in zip(headers, header_vars) if var.get()]
        if len(selected_headers) != 6:
            messagebox.showinfo("Info", "Please select exactly six (6) headers for analysis.")
            return
        selection_window.quit()
        selection_window.destroy()

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select six (6) headers for Force Plate Data")
    selection_window.geometry(f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.9)}")

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    header_vars = [BooleanVar() for _ in headers]
    num_columns = 7  # number of columns for header labels

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill="y", anchor="center")
    Button(btn_frame, text="Select All", command=select_all).pack(side="top", pady=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(side="top", pady=5)
    Button(btn_frame, text="Confirm", command=on_select).pack(side="top", pady=5)

    selection_window.mainloop()

    if not selected_headers:
        messagebox.showinfo("Info", "No headers were selected.")
        return None

    return selected_headers


def calc_cop(data, board_height_m: float = 0.0):
    """
    Converts force (N) and moment (N.m) data into center of pressure (m) coordinates.
    
    Parameters:
        data: numpy array with columns [Fx, Fy, Fz, Mx, My, Mz]
        board_height_m: float, height of the board over the force plate (in meters). 
                        If 0, assumes no board is present.
    
    Returns:
        cop_xyz: numpy array with columns [CP_ap, CP_ml, h] in meters.
                 For no board (h = 0), CP_ap = -My/Fz and CP_ml = Mx/Fz.
                 For board present, formulas are:
                   CP_ap = (-h * Fx - My) / Fz
                   CP_ml = (h * Fy + Mx) / Fz
    """
    data = np.asarray(data)
    fx = data[:, 0]
    fy = data[:, 1]
    fz = data[:, 2]
    mx = data[:, 3]
    my = data[:, 4]
    # Mz is not used in these calculations

    # Garantir que Fz seja positivo
    if np.any(fz < 0):
        fz = -fz

    if board_height_m == 0.0:
        # Sem board: CP_ap = -My/Fz, CP_ml = Mx/Fz, CP_z = 0
        cp_ap = -my / fz
        cp_ml = mx / fz
        cp_z = np.zeros_like(fz)
    else:
        h = board_height_m  # altura em metros
        cp_ap = (-h * fx - my) / fz
        cp_ml = (h * fy + mx) / fz
        cp_z = np.full_like(fz, h)

    cop_xyz = np.column_stack((cp_ap, cp_ml, cp_z))
    return cop_xyz


def main():
    """Main function to run the CoP calculation."""
    root = Tk()
    root.withdraw()  # Hides the main Tkinter window

    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Select input and output directories
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    if not input_dir:
        print("No input directory selected.")
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected.")
        return

    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    # Ask for moment unit
    moment_unit = simpledialog.askstring("Moment Unit of Measurement",
                                           "Enter the unit of measurement for your moment data (e.g. N.m, N.mm)",
                                           initialvalue="N.m",
                                           parent=root)
    print(f"Moment unit of measurement: {moment_unit}")

    # Ask for force plate dimensions in mm (convert to meters)
    dimensions_input = simpledialog.askstring("Force Plate Dimensions",
                                                "Enter the force plate dimensions (length [X-axis], width [Y-axis]) in mm, separated by a comma:",
                                                initialvalue="508,464",
                                                parent=root)
    if dimensions_input:
        try:
            fp_dimensions_xy = [float(dim) / 1000 for dim in dimensions_input.split(",")]
            if len(fp_dimensions_xy) != 2:
                raise ValueError("Please provide exactly two values: length and width.")
        except ValueError as e:
            print(f"Invalid input for force plate dimensions: {e}")
            return
    else:
        print("No force plate dimensions provided. Using default [0.508, 0.464].")
        fp_dimensions_xy = [0.508, 0.464]

    # Ask for board height (in mm) and convert to meters
    board_height_mm = simpledialog.askfloat("Board Height",
                                              "Enter the height of the board over the force plate (in mm):",
                                              initialvalue=0.0,
                                              parent=root)
    if board_height_mm is not None:
        board_height_m = board_height_mm / 1000
        print(f"Using provided board height: {board_height_m} m")
    else:
        print("No board height provided. Assuming board_height_m = 0.")
        board_height_m = 0.0

    # Get list of CSV files in input directory
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if not csv_files:
        print("No CSV files found in the selected directory.")
        return

    # Let user select headers from the first CSV file
    first_file_path = os.path.join(input_dir, csv_files[0])
    selected_headers = select_headers(first_file_path)
    if not selected_headers:
        print("No valid headers selected.")
        return

    print(f"Selected Headers: {selected_headers}")

    # Create output directory with timestamp
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_dir, f"vaila_cop_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Main output directory created: {main_output_dir}")

    # Process each CSV file
    for file_name in csv_files:
        print(f"Processing file: {file_name}")
        file_path = os.path.join(input_dir, file_name)
        try:
            df_full = read_csv_full(file_path)
        except Exception as e:
            print(f"Error reading CSV file {file_name}: {e}")
            continue

        if not all(header in df_full.columns for header in selected_headers):
            messagebox.showerror("Header Error", f"Selected headers not found in file {file_name}.")
            print(f"Error: Selected headers not found in file {file_name}. Skipping file.")
            continue

        data = df_full[selected_headers].to_numpy()

        try:
            cop_xyz_m = calc_cop(data, board_height_m=board_height_m)
        except Exception as e:
            print(f"Error processing data for file {file_name}: {e}")
            messagebox.showerror("Data Processing Error", f"Error processing data for file {file_name}: {e}")
            continue

        # Convert CoP from meters to millimeters
        cop_xyz_mm = cop_xyz_m * 1000

        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(main_output_dir, f"{file_name_without_extension}_{timestamp}.csv")
        output_df = pd.DataFrame(cop_xyz_mm, columns=["cop_ap_mm", "cop_ml_mm", "cop_z_mm"])
        output_df.to_csv(output_file_path, index=False)
        print(f"Saved CoP data to: {output_file_path}")

    print("All files processed.")
    messagebox.showinfo("Information", "CoP calculation complete! The window will close in 5 seconds.")


if __name__ == "__main__":
    main()

