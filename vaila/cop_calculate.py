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

def convert_to_mm(data, unit):
    """Converts the data to millimeters based on the provided unit."""
    conversion_factors = {
        "m": 1000,
        "cm": 10,
        "mm": 1,
        "ft": 304.8,
        "in": 25.4,
        "yd": 914.4,
    }
    if unit not in conversion_factors:
        raise ValueError(
            f"Unsupported unit '{unit}'. Please use m, cm, mm, ft, in, or yd."
        )
    return data * conversion_factors[unit]

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
        selected_headers = [
            header for header, var in zip(headers, header_vars) if var.get()
        ]
        if len(selected_headers) != 6:
            messagebox.showinfo(
                "Info", "Please select exactly six (6) headers for analysis."
            )
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
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.9)}"
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

    num_columns = 7  # Number of columns for header labels

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

def N_Nm2COP_OR67(N_Nm_Data):
    """
    Converts force (N) and moment (Nmm) data to CoP (mm) coordinates.
    Inputs:
        N_Nm_Data: numpy array with columns [Fx, Fy, Fz, Mx, My, Mz]
    Outputs:
        COPxy_mm: numpy array with columns [COPx, COPy] in mm
    """
    # Default location (in meters) of the center of the force plate
    C0 = [0.0000e-3, -0.0001e-3, -50.000e-3]

    # CoP calculations based on provided formulas
    COPx = ((-N_Nm_Data[:, 4] + N_Nm_Data[:, 0] * C0[2]) / N_Nm_Data[:, 2]) + C0[0]
    COPy = ((N_Nm_Data[:, 3] + N_Nm_Data[:, 1] * C0[2]) / N_Nm_Data[:, 2]) + C0[1]

    # Convert from meters to millimeters
    COPxy_mm = np.column_stack((COPx, COPy)) * 1000
    return COPxy_mm

def main():
    """Function to run the CoP calculation."""
    root = Tk()
    root.withdraw()  # Hides the main Tkinter window

    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Request input and output directories
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

    # Set default units for forces (N) and moments (Nmm)
    unit = "mm"
    print(f"Default unit of measurement: {unit}")

    # Sort files to select the first CSV for header selection
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if not csv_files:
        print("No CSV files found in the selected directory.")
        return

    first_file_path = os.path.join(input_dir, csv_files[0])
    selected_headers = select_headers(first_file_path)
    if not selected_headers:
        print("No valid headers selected.")
        return

    print(f"Selected Headers: {selected_headers}")

    # Create timestamp for output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Create main output directory using timestamp
    main_output_dir = os.path.join(output_dir, f"vaila_cop_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    print(f"Main output directory created: {main_output_dir}")

    # Process each CSV file in the input directory
    for file_name in csv_files:
        print(f"Processing file: {file_name}")
        file_path = os.path.join(input_dir, file_name)
        try:
            df_full = read_csv_full(file_path)
        except Exception as e:
            print(f"Error reading the CSV file: {e}")
            continue

        if not all(header in df_full.columns for header in selected_headers):
            messagebox.showerror(
                "Header Error", f"Selected headers not found in file {file_name}."
            )
            print(f"Error: Selected headers not found in file {file_name}. Skipping file.")
            continue

        data = df_full[selected_headers].to_numpy()

        # Convert data to mm if necessary
        try:
            data = convert_to_mm(data, unit)
        except ValueError as e:
            print(f"Error converting units for file {file_name}: {e}")
            messagebox.showerror(
                "Unit Conversion Error",
                f"An error occurred during unit conversion:\n{e}",
            )
            continue

        # Calculate CoP from force and moment data
        COPxy_mm = N_Nm2COP_OR67(data)

        # Add Cz (0.0) to CoP data
        Cz = np.zeros((COPxy_mm.shape[0], 1))
        COP_xyz = np.hstack((COPxy_mm, Cz))

        # Save the CoP data to a new CSV file
        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(main_output_dir, f"{file_name_without_extension}_{timestamp}.csv")
        output_df = pd.DataFrame(COP_xyz, columns=['CoP_x (mm)', 'CoP_y (mm)', 'CoP_z (mm)'])
        output_df.to_csv(output_file_path, index=False)

        print(f"Saved CoP data to: {output_file_path}")

    # Inform the user that the analysis is complete
    print("All files processed.")
    messagebox.showinfo(
        "Information", "CoP calculation complete! The window will close in 5 seconds."
    )
    root.after(5000, lambda: root.quit())  # Wait for 5 seconds and then quit safely
    root.mainloop()

if __name__ == "__main__":
    main()

