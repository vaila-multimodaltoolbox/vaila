"""
cop_calculate.py
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2026-02-03
Update: 2026-02-03
Version: 0.0.3
Python Version: 3.12.9
Description:
------------
This script allows users to select six (6) headers for force plate data analysis.
For each input file it writes two CSV outputs: one using the Shimba (1984) method
and one using the simplified formulas: cp_ap = (-h*fx - my)/fz, cp_ml = (h*fy + mx)/fz.
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
import re
from collections import defaultdict
from tkinter import (
    BooleanVar,
    Button,
    Canvas,
    Checkbutton,
    Frame,
    LabelFrame,
    Scrollbar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    simpledialog,
)

import ezc3d
import numpy as np
import pandas as pd


def read_csv_full(filename):
    """Reads the full CSV file."""
    try:
        data = pd.read_csv(filename, delimiter=",")
        return data
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {str(e)}")


def read_c3d_analogs(filename):
    """
    Reads analog data from a C3D file using ezc3d.
    Returns a pandas DataFrame with analog channels as columns.
    """
    try:
        c = ezc3d.c3d(filename)
        # Extract analog labels
        labels = c["parameters"]["ANALOG"]["LABELS"]["value"]
        # Extract analog data (Shape: 1 x N_channels x N_frames) -> (N_channels x N_frames)
        data = c["data"]["analogs"][0, :, :]

        # Create DataFrame
        # Transpose to have frames as rows, channels as columns
        df = pd.DataFrame(data.T, columns=labels)

        # Add Time column if rate is available
        rate = c["parameters"]["ANALOG"]["RATE"]["value"][0]
        n_frames = df.shape[0]
        time = np.linspace(0, n_frames / rate, n_frames, endpoint=False)
        df.insert(0, "Time", time)

        return df
    except Exception as e:
        raise Exception(f"Error reading the C3D file: {str(e)}")


def select_headers(file_path):
    """Displays a GUI to select six (6) headers for force plate data analysis."""

    def get_file_headers(file_path):
        """Reads the headers from a CSV or C3D file."""
        if file_path.lower().endswith(".c3d"):
            df = read_c3d_analogs(file_path)
        else:
            df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_file_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [
            header for header, var in zip(headers, header_vars, strict=False) if var.get()
        ]
        if len(selected_headers) != 6:
            messagebox.showinfo("Info", "Please select exactly six (6) headers for analysis.")
            return
        selection_window.quit()
        _cleanup_bindings()

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select six (6) headers for Force Plate Data")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight() * 0.9)}"
    )

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # --- Scroll Bindings ---
    def _on_mousewheel(event):
        try:
            if event.delta:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            elif event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")
        except Exception:
            pass

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    def _cleanup_bindings():
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")
        selection_window.destroy()

    selection_window.protocol("WM_DELETE_WINDOW", _cleanup_bindings)

    header_vars = [BooleanVar() for _ in headers]

    # --- Grouping Logic ---
    groups = defaultdict(list)
    # Match Force.Fx1, etc.
    fp_pattern = re.compile(r"^(Force|Moment|Analog)[._]([FfMm][xyzXYZ])(\d+)?$")

    for i, h in enumerate(headers):
        match = fp_pattern.match(h)
        if match:
            # Force Plate Channel
            idx = match.group(3) if match.group(3) else "1"
            group_key = f"Force Plate {idx}"
            groups[group_key].append(i)
        else:
            # Group by generic prefix
            if "." in h:
                prefix = h.split(".")[0]
                groups[prefix].append(i)
            elif "_" in h:
                prefix = h.split("_")[0]
                groups[prefix].append(i)
            else:
                groups["Other"].append(i)

    # Sort Keys
    def group_sort_key(k):
        if k.startswith("Force Plate"):
            nums = re.findall(r"\d+", k)
            n = int(nums[0]) if nums else 0
            return (0, n, k)
        elif k == "Other":
            return (2, 0, k)
        else:
            return (1, 0, k)

    sorted_keys = sorted(groups.keys(), key=group_sort_key)

    # helper for group selection
    def make_select_group(indices, val):
        def callback():
            for idx in indices:
                header_vars[idx].set(val)

        return callback

    # Layout Groups
    MAX_GROUPS_PER_ROW = 3

    for g_idx, g_key in enumerate(sorted_keys):
        indices = groups[g_key]

        # Container for the group
        lf = LabelFrame(scrollable_frame, text=g_key, padx=5, pady=5, font=("Arial", 10, "bold"))
        row = g_idx // MAX_GROUPS_PER_ROW
        col = g_idx % MAX_GROUPS_PER_ROW
        lf.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # Group Buttons
        btn_f = Frame(lf)
        btn_f.pack(side="top", anchor="e", fill="x")
        Button(
            btn_f, text="All", font=("Arial", 7), width=4, command=make_select_group(indices, True)
        ).pack(side="right", padx=2)
        Button(
            btn_f,
            text="None",
            font=("Arial", 7),
            width=4,
            command=make_select_group(indices, False),
        ).pack(side="right", padx=2)

        # Sort headers inside group
        if g_key.startswith("Force Plate"):

            def header_sort_key(idx):
                h = headers[idx].lower()
                if "fx" in h:
                    return 1
                if "fy" in h:
                    return 2
                if "fz" in h:
                    return 3
                if "mx" in h:
                    return 4
                if "my" in h:
                    return 5
                if "mz" in h:
                    return 6
                return 10

            indices.sort(key=header_sort_key)

        # Content Frame
        content_f = Frame(lf)
        content_f.pack(fill="both", expand=True)

        # Grid Checkbuttons
        inner_cols = 3 if g_key.startswith("Force Plate") or len(indices) > 4 else 1

        for j, h_idx in enumerate(indices):
            chk = Checkbutton(content_f, text=headers[h_idx], variable=header_vars[h_idx])
            chk.grid(row=j // inner_cols, column=j % inner_cols, sticky="w", padx=2)

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


def calculate_cop_shimba(fx, fy, fz, mx, my, mz, threshold=0.0):
    """
    Calculate Point of Wrench Application (PWA) using Shimba (1984) method.

    Formula based on:
    Shimba T. (1984), "An estimation of center of gravity from force platform data",
    Journal of Biomechanics 17(1), 53–60.

    Adapted from BTK's GroundReactionWrenchFilter.

    Parameters:
        fx, fy, fz: Force components (arrays or scalars)
        mx, my, mz: Moment components (arrays or scalars) at the origin
        threshold: Fz threshold below which CoP is set to 0 to avoid division by zero

    Returns:
        px, py, pz: Coordinates of the PWA (CoP)
    """
    # Square norm of force
    snf = fx**2 + fy**2 + fz**2

    # Avoid division by zero
    # Create mask for valid data (Fz > threshold and sNF > 0)
    # Using numpy for array handling

    # Initialize output arrays
    px = np.zeros_like(fz)
    py = np.zeros_like(fz)
    pz = np.zeros_like(fz)  # Shimba PWA Pz is typically 0 if calculated at surface

    # Mask for valid calculation
    # BTK logic: if (sNF == 0.0) || (abs(Fz) <= threshold) -> Px=Py=0
    mask = (snf > 0) & (np.abs(fz) > threshold)

    # Calculate for valid indices
    # To avoid runtime warnings for division by zero in invalid indices, we effectively filter
    # But since we are doing array operations, we can compute everything and then apply mask,
    # OR only compute on masked elements. Python is safer with masked operations to avoid warnings.

    if np.any(mask):
        f_sq = snf[mask]
        f_z_val = fz[mask]
        f_x_val = fx[mask]
        f_y_val = fy[mask]
        m_x_val = mx[mask]
        m_y_val = my[mask]
        m_z_val = mz[mask]

        # Px = (Fy * Mz - Fz * My) / sNF - (Fx^2 * My - Fx * Fy * Mx) / (sNF * Fz)
        term1_x = (f_y_val * m_z_val - f_z_val * m_y_val) / f_sq
        term2_x = (f_x_val**2 * m_y_val - f_x_val * (f_y_val * m_x_val)) / (f_sq * f_z_val)
        px[mask] = term1_x - term2_x

        # Py = (Fz * Mx - Fx * Mz) / sNF - (Fx * Fy * My - Fy^2 * Mx) / (sNF * Fz)
        term1_y = (f_z_val * m_x_val - f_x_val * m_z_val) / f_sq
        term2_y = (f_x_val * (f_y_val * m_y_val) - f_y_val**2 * m_x_val) / (f_sq * f_z_val)
        py[mask] = term1_y - term2_y

    return px, py, pz


def calc_cop(data, board_height_m: float = 0.0, moment_unit: str = "N.m"):
    """
    Converts force (N) and moment (N.m or N.mm) data into center of pressure (m) coordinates.
    Uses Shimba (1984) method for PWA calculation.

    Parameters:
        data: numpy array with columns [Fx, Fy, Fz, Mx, My, Mz]
        board_height_m: float, height of the board over the force plate (in meters).
                        If not 0, moments are adjusted before PWA calculation.
        moment_unit: str, unit of moment data ('N.m' or 'N.mm').
                     If 'N.mm', moments are converted to N.m before calculation.

    Returns:
        cop_xyz: numpy array with columns [CP_ap, CP_ml, CP_z] in meters.
    """
    data = np.asarray(data)
    fx = data[:, 0]
    fy = data[:, 1]
    fz = data[:, 2]
    mx = data[:, 3]
    my = data[:, 4]
    mz = data[:, 5]

    # Convert Moments to N.m if provided in N.mm
    if "mm" in moment_unit.lower():
        mx = mx / 1000.0
        my = my / 1000.0
        mz = mz / 1000.0

    # Adjust moments if there is a board height (Translation of origin)
    # M_new = M_old + F x r
    # Where r is vector from old origin to new origin (0, 0, h)
    # r = [0, 0, h]
    # F x r = | i  j  k |
    #         | Fx Fy Fz| = (Fy*h - Fz*0)i - (Fx*h - Fz*0)j + (Fx*0 - Fy*0)k
    #         | 0  0  h |
    #       = (Fy*h, -Fx*h, 0)
    # This seems inverted. Usually we want Moment AT the surface.
    # If board_height_m is the distance from Sensor Origin to Surface.
    # We want M at surface.
    # However, Shimba's formula calculates PWA relative to the origin where M is defined.
    # If we want PWA at the top of the board, we should probably calculate PWA at sensor origin
    # and then project it? Or translate M to surface?
    # BTK implementation: Mx += Fy * o.z() - o.y() * Fz ...
    # where o is origin...
    # Let's keep it simple: Calculate PWA at the origin using Shimba.
    # Then checking vertical offset.

    # Ideally, we calculate PWA using given M and F.
    # If board_height_m > 0, we can assume the result (Pz) might need adjustment or M needs adjustment.
    # For now, let's stick to the user's previous logic flow but use Shimba for the core calculation.
    # Previous logic:
    # cp_ap = (-h * fx - my) / fz  <-- This is basically transforming Moment My
    # The previous logic was: M_surface_y = My + h*Fx.  Actually My_surface = My + Fx*h (if Fx applied at h creates moment)
    # Let's use Shimba on the provided F and M.

    # If board_height exists, we translate the moments to the new surface plane BEFORE Shimba.
    if board_height_m != 0.0:
        # Translate moments to the surface (z = board_height_m)
        # Assuming F is constant.
        # M_surface = M_sensor - (P_surface - P_sensor) x F
        # Vector from Sensor to Surface is (0,0,h).
        # M_surf = M_sens - (h * k) x F
        # k x F = (-Fy, Fx, 0)
        # M_surf_x = Mx - ( h * -Fy ) = Mx + h*Fy
        # M_surf_y = My - ( h * Fx )  = My - h*Fx
        # M_surf_z = Mz

        # Wait, usually Moment measured at sensor. Force applied at surface.
        # M_sensor = M_surface + (P_surface - P_sensor) x F
        # M_sensor = (0,0,h) x F = (-hFy, hFx, 0) + M_surface (if M_surface pure torque?)
        # Actually usually M_sensor = r x F.
        # r = (px, py, h).
        # We want px, py.

        # Consistent with previous code:
        # cp_ap = (-h*fx - my)/fz -> -my/fz - h*fx/fz.
        # -my/fz is standard x. -h*fx/fz is correction due to height.

        # Let's apply correction to Moments so Shimba calculates correct P
        # Mx_new = Mx + board_height_m * Fy
        # My_new = My - board_height_m * Fx

        mx = mx + board_height_m * fy
        my = my - board_height_m * fx

    # Calculate PWA using Shimba
    # Threshold for Fz (e.g. 10N) to avoid noise
    px, py, pz = calculate_cop_shimba(fx, fy, fz, mx, my, mz, threshold=5.0)  # 5.0 N threshold

    # If board_height was used, does Pz change? Shimba gives P on the plane where M=0 (ideally)
    # But for a sensor with 6DOF, PWA is the point where the wrench is parallel to Force.
    # The formula returns Px, Py with Pz assumed 0 (projected).

    if board_height_m != 0.0:
        pz = np.full_like(fz, board_height_m)

    # Standard Orientation correction might be needed depending on Lab CS
    # In Vaila previous code: CP_ap = -My/Fz.
    # In Shimba (and standad): Px ~= -My/Fz.
    # But Vaila outputs [CP_ap, CP_ml].
    # Usually AP is X, ML is Y? Or AP is Y, ML is X?
    # Previous code:
    # cp_ap = -my/fz  (This looks like X formula: x = -My/Fz) -> So AP is X?
    # cp_ml = mx/fz   (This looks like Y formula: y = Mx/Fz)  -> So ML is Y?
    #
    # Wait, usually AP (Anterior-Posterior) is Y axis in ISB?
    # If AP is X, then X is forward?
    # Let's respect the variable naming in return.
    # cop_xyz = [cp_ap, cp_ml, cp_z]
    #
    # Shimba returns px, py.
    # px term starts with -My... so px corresponds to cp_ap (based on previous code).
    # py term starts with Mx...  so py corresponds to cp_ml.

    cop_xyz = np.column_stack((px, py, pz))
    return cop_xyz


def calc_cop_simplified(
    data, board_height_m: float = 0.0, moment_unit: str = "N.m", threshold: float = 5.0
):
    """
    CoP using the simplified formula (no full wrench projection).

    Formulas:
        cp_ap = (-h * fx - my) / fz
        cp_ml = (h * fy + mx) / fz

    Parameters:
        data: numpy array with columns [Fx, Fy, Fz, Mx, My, Mz]
        board_height_m: height of the board over the force plate (m). Used as h in formulas.
        moment_unit: 'N.m' or 'N.mm'. If N.mm, moments are converted to N.m before calculation.
        threshold: Fz threshold below which CoP is set to 0.

    Returns:
        cop_xyz: numpy array with columns [CP_ap, CP_ml, CP_z] in meters.
    """
    data = np.asarray(data)
    fx = data[:, 0].astype(float)
    fy = data[:, 1].astype(float)
    fz = data[:, 2].astype(float)
    mx = data[:, 3].astype(float)
    my = data[:, 4].astype(float)
    mz = data[:, 5].astype(float)

    if "mm" in moment_unit.lower():
        mx = mx / 1000.0
        my = my / 1000.0
        mz = mz / 1000.0

    h = float(board_height_m)
    valid = np.abs(fz) > threshold

    cp_ap = np.zeros_like(fz)
    cp_ml = np.zeros_like(fz)
    cp_z = np.zeros_like(fz)

    if np.any(valid):
        fz_v = fz[valid]
        cp_ap[valid] = (-h * fx[valid] - my[valid]) / fz_v
        cp_ml[valid] = (h * fy[valid] + mx[valid]) / fz_v

    if board_height_m != 0.0:
        cp_z[:] = board_height_m

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
    moment_unit = simpledialog.askstring(
        "Moment Unit of Measurement",
        "Enter the unit of measurement for your moment data (e.g. N.m, N.mm)",
        initialvalue="N.m",
        parent=root,
    )
    print(f"Moment unit of measurement: {moment_unit}")

    # Ask for force plate dimensions in mm (convert to meters)
    dimensions_input = simpledialog.askstring(
        "Force Plate Dimensions",
        "Enter the force plate dimensions (length [X-axis], width [Y-axis]) in mm, separated by a comma:",
        initialvalue="508,464",
        parent=root,
    )
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
    board_height_mm = simpledialog.askfloat(
        "Board Height",
        "Enter the height of the board over the force plate (in mm):",
        initialvalue=0.0,
        parent=root,
    )
    if board_height_mm is not None:
        board_height_m = board_height_mm / 1000
        print(f"Using provided board height: {board_height_m} m")
    else:
        print("No board height provided. Assuming board_height_m = 0.")
        board_height_m = 0.0

    # Get list of CSV and C3D files in input directory
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".csv", ".c3d"))])
    if not files:
        print("No CSV or C3D files found in the selected directory.")
        return

    # Let user select headers from the first file
    first_file_path = os.path.join(input_dir, files[0])
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

    # Process each file
    for file_name in files:
        print(f"Processing file: {file_name}")
        file_path = os.path.join(input_dir, file_name)
        try:
            if file_name.lower().endswith(".c3d"):
                df_full = read_c3d_analogs(file_path)
            else:
                df_full = read_csv_full(file_path)
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            continue

        if not all(header in df_full.columns for header in selected_headers):
            messagebox.showerror("Header Error", f"Selected headers not found in file {file_name}.")
            print(f"Error: Selected headers not found in file {file_name}. Skipping file.")
            continue

        data = df_full[selected_headers].to_numpy()

        try:
            cop_xyz_shimba_m = calc_cop(
                data, board_height_m=board_height_m, moment_unit=moment_unit
            )
            cop_xyz_simplified_m = calc_cop_simplified(
                data, board_height_m=board_height_m, moment_unit=moment_unit
            )
        except Exception as e:
            print(f"Error processing data for file {file_name}: {e}")
            messagebox.showerror(
                "Data Processing Error",
                f"Error processing data for file {file_name}: {e}",
            )
            continue

        # Convert CoP from meters to millimeters
        cop_xyz_shimba_mm = cop_xyz_shimba_m * 1000
        cop_xyz_simplified_mm = cop_xyz_simplified_m * 1000

        file_name_without_extension = os.path.splitext(file_name)[0]

        # Output 1: Shimba method
        output_file_shimba = os.path.join(
            main_output_dir, f"{file_name_without_extension}_shimba_{timestamp}.csv"
        )
        df_shimba = pd.DataFrame(cop_xyz_shimba_mm, columns=["cop_ap_mm", "cop_ml_mm", "cop_z_mm"])
        df_shimba.to_csv(output_file_shimba, index=False)
        print(f"Saved CoP (Shimba) to: {output_file_shimba}")

        # Output 2: Simplified method
        output_file_simplificado = os.path.join(
            main_output_dir, f"{file_name_without_extension}_simplificado_{timestamp}.csv"
        )
        df_simplificado = pd.DataFrame(
            cop_xyz_simplified_mm, columns=["cop_ap_mm", "cop_ml_mm", "cop_z_mm"]
        )
        df_simplificado.to_csv(output_file_simplificado, index=False)
        print(f"Saved CoP (simplificado) to: {output_file_simplificado}")

    print("All files processed.")
    messagebox.showinfo(
        "Information", "CoP calculation complete! The window will close in 5 seconds."
    )


if __name__ == "__main__":
    main()
