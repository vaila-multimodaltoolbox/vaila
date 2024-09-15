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
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(
        side="top", pady=5
    )
    Button(btn_frame, text="Confirm", command=on_select).pack(side="top", pady=5)

    selection_window.mainloop()

    if not selected_headers:
        messagebox.showinfo("Info", "No headers were selected.")
        return None

    return selected_headers


def calc_cop(data, fp_dimensions_xy=None, h=None):
    """
    Converts force (N) and moment (N.mm) data to CoP (mm) coordinates.
    Inputs:
        data: numpy array with columns [Fx, Fy, Fz, Mx, My, Mz]
        fp_dimensions_xy: array or None, the dimensions of the force plate in mm [width, length]
        h: float or None, the height of the board over the force plate (in mm)
    Outputs:
        cop_xyz_mm: numpy array with columns [cop_x, cop_y, cop_z] in mm

    Calculations based on:

    Case 1 - No board over the force plate (h is None):
        cop_x = My / Fz
        cop_y = Mx / Fz
        cop_z = 0 (height at force plate level)

    Case 2 - Board exists over the force plate (h is provided):
        cop_x = (-h * Fx - My) / Fz
        cop_y = (-h * Fy + Mx) / Fz
        cop_z = h (height of the board over the force plate)

    Where:
        - Mx, My: Components of the moment (N.mm)
        - Fx, Fy, Fz: Force components (N)
        - h: Height of the board over the force plate (mm)
    """

    # Convert data to numpy array if it's not already
    data = np.asarray(data)

    # Use default force plate dimensions if none are provided
    if fp_dimensions_xy is None:
        fp_dimensions_xy = np.array([500, 500])

    # Calculate the center of the force plate
    fp_center = fp_dimensions_xy / 2

    # Initialize CoP arrays with the default values as the center of the force plate
    cop_x = np.full(data.shape[0], fp_center[0])
    cop_y = np.full(data.shape[0], fp_center[1])

    # Identify rows where Fz is not zero
    fz_nonzero = data[:, 2] != 0

    # Case 1 - No board over the force plate (h is None)
    if h is None:
        cop_x[fz_nonzero] = data[fz_nonzero, 4] / data[fz_nonzero, 2]
        cop_y[fz_nonzero] = data[fz_nonzero, 3] / data[fz_nonzero, 2]
        cop_z = np.zeros(data.shape[0])  # CoP Z is zero when no board is present
    # Case 2 - Board exists over the force plate (h is provided)
    else:
        # Ensure h is a valid float
        if not isinstance(h, (int, float)):
            raise ValueError("The height 'h' must be a number.")

        cop_x[fz_nonzero] = (-h * data[fz_nonzero, 0] - data[fz_nonzero, 4]) / data[
            fz_nonzero, 2
        ]
        cop_y[fz_nonzero] = (-h * data[fz_nonzero, 1] + data[fz_nonzero, 3]) / data[
            fz_nonzero, 2
        ]
        cop_z = np.full(data.shape[0], h)  # CoP Z is the height of the board

    # Combine cop_x, cop_y, and cop_z into a single array
    cop_xyz = np.column_stack((cop_x, cop_y, cop_z))

    return cop_xyz


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

    # Request force plate dimensions from user
    dimensions_input = simpledialog.askstring(
        "Force Plate Dimensions",
        "Enter the force plate dimensions (width, length) in mm, separated by a comma:",
        parent=root,
    )
    if dimensions_input:
        try:
            fp_dimensions_xy = [float(dim) for dim in dimensions_input.split(",")]
            if len(fp_dimensions_xy) != 2:
                raise ValueError("Please provide exactly two values: width and length.")
        except ValueError as e:
            print(f"Invalid input for force plate dimensions: {e}")
            return
    else:
        print("No force plate dimensions provided. Using default [500, 500].")
        fp_dimensions_xy = [500, 500]

    # Request the height 'h' from user
    h = simpledialog.askfloat(
        "Board Height (h)",
        "Enter the height of the board over the force plate (in mm):",
        parent=root,
    )
    if h is not None:
        print(f"Using provided board height: {h} mm")
    else:
        print(
            "No height provided. Calculations will assume no board over the force plate (h=None)."
        )
        h = None

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
            print(
                f"Error: Selected headers not found in file {file_name}. Skipping file."
            )
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
        cop_xyz = calc_cop(data, fp_dimensions_xy, h)

        # Save the CoP data to a new CSV file
        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(
            main_output_dir, f"{file_name_without_extension}_{timestamp}.csv"
        )
        output_df = pd.DataFrame(cop_xyz, columns=["cop_x_mm", "cop_y_mm", "cop_z_mm"])
        output_df.to_csv(output_file_path, index=False)

        print(f"Saved CoP data to: {output_file_path}")

    # Inform the user that the analysis is complete
    print("All files processed.")
    messagebox.showinfo(
        "Information", "CoP calculation complete! The window will close in 5 seconds."
    )
    # root.after(5000, lambda: root.quit())  # Wait for 5 seconds and then quit safely
    # root.mainloop()


if __name__ == "__main__":
    main()
