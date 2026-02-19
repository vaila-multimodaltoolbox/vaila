"""
Project: vailá Multimodal Toolbox
Script: readcsv.py - Read CSV File

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 03 September 2025
Version: 0.0.4

Description:
    This script provides tools for reading CSV files and displaying their contents.
    It includes functions for:
    - Detecting the delimiter used in the file.
    - Detecting if the file has a header.
    - Selecting markers to display.
    - Selecting headers to display.
    - Visualizing the data using PyVista (same backend as C3D viewer).

Usage:
    Run the script from the command line:
        python readcsv.py

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - pyvista (for 3D viewer)
    - tkinter
    - rich

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.3: Added support for CSV, TXT and TSV files, improved UI
    - v0.0.2: Added support for CSV, TXT and TSV files, improved UI
    - v0.0.1: First version
"""

import os
import tkinter as tk
from pathlib import Path
from tkinter import Button, Frame, Listbox, filedialog, messagebox

import numpy as np
import pandas as pd
from rich import print


# ==================== PERFORMANCE OPTIMIZATIONS ====================
def _npz_path_for(csv_path: str) -> str:
    """Generate NPZ cache path for CSV file"""
    base = os.path.basename(csv_path)
    stem, _ = os.path.splitext(base)
    return os.path.join(os.path.dirname(csv_path), f"{stem}_cache.npz")


def _load_with_cache(csv_path: str, use_cache: bool = True):
    """Load data with NPZ cache for speed"""
    npz_file = _npz_path_for(csv_path)

    if use_cache and os.path.exists(npz_file):
        try:
            print(f"[bold green]Loading from cache: {npz_file}[/bold green]")
            data = np.load(npz_file)
            return data["index_vector"], data["marker_names"], data["points"]
        except Exception as e:
            print(f"Cache loading failed: {e}")

    # Load from CSV and save cache
    print("[bold yellow]Loading from CSV and creating cache...[/bold yellow]")
    index_vector, marker_data, valid_markers, delimiter = read_csv_generic(csv_path)
    points = np.stack([marker_data[m] for m in valid_markers], axis=1)

    if use_cache:
        try:
            np.savez_compressed(
                npz_file,
                index_vector=index_vector,
                marker_names=list(valid_markers.keys()),
                points=points,
            )
            print(f"[bold green]Cache saved: {npz_file}[/bold green]")
        except Exception as e:
            print(f"Cache saving failed: {e}")

    return index_vector, list(valid_markers.keys()), points


def show_csv_optimized(file_path=None, *, use_cache=True, turbo=False, fps=30, stride=1):
    """
    Optimized version of show_csv with performance improvements

    Args:
        file_path: CSV file path
        use_cache: Use NPZ cache for faster loading
        turbo: Use optimized visualization mode
        fps: Frames per second for animation
        stride: Skip frames (1=all, 2=every other, etc.)
    """
    print(f"Running optimized script: {os.path.basename(__file__)}")
    print(f"Performance settings: cache={use_cache}, turbo={turbo}, fps={fps}, stride={stride}")

    # File selection
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = select_file()
        if not file_path:
            print("No file selected.")
            return

    if not os.path.exists(file_path):
        print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        return

    # Load data with cache
    try:
        index_vector, marker_names, points = _load_with_cache(file_path, use_cache)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading data: {e}")
        return

    # Apply stride to reduce data size
    if stride > 1:
        points = points[::stride]
        index_vector = index_vector[::stride]
        print(
            f"Applied stride {stride}: reduced frames from {len(index_vector) * stride} to {len(index_vector)}"
        )

    # Select markers
    selected_markers = select_markers_csv(marker_names)
    if not selected_markers:
        messagebox.showwarning("Warning", "No markers selected.")
        return

    # Filter points for selected markers
    marker_indices = [marker_names.index(m) for m in selected_markers]
    points = points[:, marker_indices, :]

    print("[bold green]Data ready for visualization![/bold green]")
    print(f"Shape: {points.shape}, FPS: {fps}")

    viewer_choice = choose_visualizer()
    if viewer_choice == "open3d":
        from vaila.viewc3d import run_viewc3d_from_array

        run_viewc3d_from_array(points, selected_markers, fps, file_path)
    else:
        from vaila.viewc3d_pyvista import MokkaLikeViewer

        MokkaLikeViewer.from_array(
            points, selected_markers, frame_rate=fps, title="Vaila - PyVista CSV Viewer"
        )


###############################################################################
# Function: headersidx
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def headersidx(headers, prefix):
    """
    Given a list of headers and a prefix, returns the indices of headers that start with the prefix.

    Args:
        headers (list): List of header names.
        prefix (str): Prefix to check.

    Returns:
        List[int]: List of indices of headers that start with the prefix.
    """
    return [i for i, header in enumerate(headers) if header.startswith(prefix)]


###############################################################################
# Function: reshapedata
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def reshapedata(df, selected_markers):
    """
    Given a DataFrame `df` containing the time column and marker columns in the format:
      marker_x, marker_y, marker_z,
    and a list with the names of selected markers, returns a NumPy array of shape:
      (num_frames, num_markers, 3)

    If the mean absolute value is high (> 100), data is converted from millimeters to meters.

    Args:
        df (DataFrame): DataFrame containing the data, where the first column is Time.
        selected_markers (list): List of marker names.

    Returns:
        numpy.ndarray: Array with shape (num_frames, num_markers, 3) containing marker data.
    """
    num_frames = df.shape[0]
    num_markers = len(selected_markers)
    points = np.zeros((num_frames, num_markers, 3))
    for i, marker in enumerate(selected_markers):
        x_col = f"{marker}_x"
        y_col = f"{marker}_y"
        z_col = f"{marker}_z"
        if x_col in df.columns and y_col in df.columns and z_col in df.columns:
            points[:, i, 0] = df[x_col].values
            points[:, i, 1] = df[y_col].values
            points[:, i, 2] = df[z_col].values
        else:
            raise ValueError(f"Columns for marker '{marker}' not found in expected format.")
    if np.mean(np.abs(points)) > 100:
        points = points * 0.001  # Convert from millimeters to meters
    return points


###############################################################################
# Function: detect_delimiter
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_delimiter(file_path):
    """
    Detects the delimiter used in the file by trying common delimiters.
    Analyzes multiple lines for more accurate detection.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Detected delimiter (',', ';', '\t', or ' ')
    """
    import csv

    # Try different delimiters and check consistency
    delimiters = [",", ";", "\t", " "]
    delimiter_scores = {}

    try:
        with open(file_path, encoding="utf-8") as file:
            # Read first few lines
            sample_lines = []
            file.seek(0)
            for i, line in enumerate(file):
                if i >= 10:  # Read max 10 lines
                    break
                sample_lines.append(line.strip())

            for delimiter in delimiters:
                scores = []
                try:
                    # Use csv.Sniffer for better detection
                    file.seek(0)
                    sample = file.read(1024)
                    if sample:  # Check if sample is not empty
                        dialect = csv.Sniffer().sniff(sample, delimiters=[delimiter])
                        if dialect.delimiter == delimiter:
                            scores.append(10)  # High score for csv.Sniffer detection
                except Exception:
                    pass

                # Check consistency of column count
                column_counts = []
                for line in sample_lines:
                    if line:
                        column_counts.append(len(line.split(delimiter)))

                if column_counts:
                    # Prefer delimiters that give consistent column counts
                    most_common_count = max(set(column_counts), key=column_counts.count)
                    consistency = column_counts.count(most_common_count) / len(column_counts)
                    avg_columns = sum(column_counts) / len(column_counts)

                    # Score based on consistency and reasonable column count
                    score = consistency * 10 + (avg_columns if avg_columns > 1 else 0)
                    delimiter_scores[delimiter] = score

            # Return delimiter with highest score
            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
                print(
                    f"Delimiter detected: '{best_delimiter}' (score: {delimiter_scores[best_delimiter]:.2f})"
                )
                return best_delimiter

    except Exception as e:
        print(f"Warning: Error detecting delimiter: {e}")

    print("Using default delimiter: ','")
    return ","  # Default to comma


###############################################################################
# Function: detect_has_header
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_has_header(file_path, delimiter):
    """
    Detects if the file has a header by checking if the first line contains non-numeric values.

    Args:
        file_path (str): Path to the file
        delimiter (str): Delimiter used in the file

    Returns:
        bool: True if file has header, False otherwise
    """
    with open(file_path, encoding="utf-8") as file:
        first_line = file.readline().strip()
        values = first_line.split(delimiter)

        # Check if any value in the first line is not numeric
        for value in values:
            try:
                float(value)
            except ValueError:
                return True
    return False


###############################################################################
# Function: select_file
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def select_file():
    """Displays the dialog box for selecting CSV, TXT or TSV file."""
    return filedialog.askopenfilename(
        title="Select data file",
        filetypes=[
            ("Data files", ("*.csv", "*.txt", "*.tsv")),
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("TSV files", "*.tsv"),
            ("All files", "*.*"),
        ],
    )


###############################################################################
# Function: select_markers_csv
# Displays a marker selection window using a Listbox.
###############################################################################
def select_markers_csv(marker_labels):
    """
    Displays a Tkinter window with a Listbox for marker selection.

    Args:
        marker_labels (list): list of marker names.

    Returns:
        List of selected marker names.
    """
    root = tk.Tk()
    root.title("Select Markers to Display")
    listbox = Listbox(root, selectmode="multiple", width=50, height=15)
    for label in marker_labels:
        listbox.insert(tk.END, label)
    listbox.pack(padx=10, pady=10)

    btn_frame = Frame(root)
    btn_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    Button(btn_frame, text="Select All", command=select_all).pack(side="left", padx=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(side="left", padx=5)

    Button(root, text="Select", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    # Return the marker names corresponding to the selected indices
    return [marker_labels[int(i)] for i in selected_indices]


###############################################################################
# Function: choose_visualizer
# Dialog to choose between PyVista and Open3D 3D viewer for CSV data.
###############################################################################
def choose_visualizer():
    """
    Displays a dialog to choose the 3D viewer: PyVista or Open3D.

    Returns:
        str: "pyvista" or "open3d"
    """
    root = tk.Tk()
    root.title("Choose 3D Viewer")
    root.resizable(False, False)
    result = [None]

    def choose(which):
        result[0] = which
        root.quit()

    tk.Label(
        root,
        text="Visualize CSV markers with:",
        font=("", 11),
    ).pack(pady=(14, 10))
    tk.Button(root, text="PyVista viewer", command=lambda: choose("pyvista"), width=22).pack(pady=4)
    tk.Button(root, text="Open3D viewer", command=lambda: choose("open3d"), width=22).pack(pady=4)
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_reqwidth()) // 2
    y = (root.winfo_screenheight() - root.winfo_reqheight()) // 2
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    root.destroy()
    return result[0] if result[0] else "pyvista"


###############################################################################
# Function: select_headers_gui
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def select_headers_gui(headers):
    """
    Displays a Tkinter window with a Listbox for header (column) selection.

    Args:
        headers (list): List of CSV headers.

    Returns:
        List of selected headers.
    """
    root = tk.Tk()
    root.title("Select CSV Headers")
    listbox = Listbox(root, selectmode="multiple", width=50, height=15)
    for header in headers:
        listbox.insert(tk.END, header)
    listbox.pack(padx=10, pady=10)

    btn_frame = Frame(root)
    btn_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    Button(btn_frame, text="Select All", command=select_all).pack(side="left", padx=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(side="left", padx=5)

    Button(root, text="OK", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    return [headers[int(i)] for i in selected_indices]


###############################################################################
# Function: get_csv_headers
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def get_csv_headers(file_path):
    """
    Reads the CSV file at the given file_path and returns its headers (column names).

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List of headers (str).
    """
    try:
        df = pd.read_csv(file_path, nrows=0)
        return list(df.columns)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV headers: {e}")
        return []


###############################################################################
# Function: detect_units
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_units(points):
    """
    Detects if the data is in millimeters or meters based on the magnitude of values.

    Args:
        points (numpy.ndarray): Array of shape (num_frames, num_markers, 3) containing the coordinates

    Returns:
        bool: True if data is in millimeters (needs conversion), False if already in meters
    """
    # Calculate the mean absolute value of all coordinates
    mean_abs = np.mean(np.abs(points))

    # If mean absolute value is greater than 100, likely in millimeters
    return mean_abs > 100


def ask_user_units():
    """
    Ask user about the units of the coordinate data.

    Returns:
        str: 'mm' for millimeters, 'm' for meters, 'auto' for automatic detection
    """
    import tkinter as tk
    from tkinter import messagebox

    try:
        # Try simple dialog first (more compatible)
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Simple approach using messagebox
        response = messagebox.askyesnocancel(
            "Select Coordinate Units",
            "What are the units of your coordinate data?\n\n"
            "• YES = Millimeters (mm) - typical values like 1234.56\n"
            "• NO = Meters (m) - typical values like 1.23\n"
            "• CANCEL = Auto-detect (let program guess)\n\n"
            "Choose based on your data range shown in terminal.",
        )

        root.destroy()

        if response is True:
            return "mm"
        elif response is False:
            return "m"
        else:
            return "auto"

    except Exception as e:
        print(f"Warning: Could not show unit selection dialog: {e}")
        print("Using auto-detection for units.")
        return "auto"


def read_csv_generic(file_path):
    """
    Reads a CSV or TXT file considering:
      - Automatically detects the delimiter (',', ';', '\t', ' ')
      - Detects if the file has a header
      - If it doesn't have a header, uses default names (Index, p1_x, p1_y, p1_z, p2_x, ...)
      - The first column contains any index data (time, frames, or any identifier)
      - The subsequent columns are organized in groups of three (x, y, z)
      - Asks user about units and converts accordingly

    Returns:
      index_vector: pd.Series with the first column data (time/frames/index)
      marker_data: dictionary mapping the marker name to a numpy array Nx3
      valid_markers: dictionary mapping the marker name to the list of columns used
      delimiter: the detected delimiter
    """
    print(f"Processing file: {file_path}")

    # Detect delimiter
    delimiter = detect_delimiter(file_path)

    # Detect if file has header
    has_header = detect_has_header(file_path, delimiter)
    print(f"Header detected: {has_header}")

    # Read the file
    try:
        if has_header:
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            # Create default column names for files without headers
            # First read to count columns
            temp_df = pd.read_csv(file_path, delimiter=delimiter, nrows=1)
            num_columns = len(temp_df.columns)

            default_columns = ["Index"]  # Generic name for first column
            for i in range(1, (num_columns - 1) // 3 + 1):
                default_columns.extend([f"p{i}_x", f"p{i}_y", f"p{i}_z"])

            # Add remaining columns if any
            while len(default_columns) < num_columns:
                default_columns.append(f"col_{len(default_columns)}")

            df = pd.read_csv(file_path, delimiter=delimiter, names=default_columns)

        print(f"Successfully read CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}") from e

    if df.empty:
        raise ValueError("The file is empty or could not be read.")

    # The first column is generic (time/frames/index/etc.)
    index_vector = df.iloc[:, 0]

    # Process the remaining columns: each column must have the format marker_coord (ex.: MARKER_X)
    # Support multiple naming conventions
    marker_headers = {}
    for col in df.columns[1:]:
        marker_name = None
        coord = None

        # Try different naming patterns
        # Pattern 1: marker_X, marker_Y, marker_Z or marker_x, marker_y, marker_z
        if "_" in col:
            parts = col.rsplit("_", 1)  # split by the last occurrence of '_'
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()

        # Pattern 2: markerX, markerY, markerZ (no underscore)
        elif col[-1].upper() in ["X", "Y", "Z"]:
            marker_name = col[:-1]
            coord = col[-1].upper()

        # Pattern 3: marker.x, marker.y, marker.z
        elif "." in col:
            parts = col.rsplit(".", 1)
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()

        # Pattern 4: marker:x, marker:y, marker:z
        elif ":" in col:
            parts = col.rsplit(":", 1)
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()

        if marker_name:
            if marker_name not in marker_headers:
                marker_headers[marker_name] = {}
            marker_headers[marker_name][coord] = col
        else:
            print(f"Warning: Could not parse column '{col}' - no matching pattern found")

    print(f"Found marker patterns: {list(marker_headers.keys())}")

    # Debug: Show details of marker parsing
    if len(marker_headers) <= 10:  # Only show details if reasonable number
        for marker, coords in marker_headers.items():
            print(f"  {marker}: {coords}")

    # Select only the markers that have the complete set of 3 columns
    valid_markers = {}
    for marker, coord_dict in marker_headers.items():
        if len(coord_dict) == 3 and all(coord in coord_dict for coord in ["X", "Y", "Z"]):
            # Ensure correct order: X, Y, Z
            ordered_cols = [coord_dict["X"], coord_dict["Y"], coord_dict["Z"]]
            valid_markers[marker] = ordered_cols
        else:
            available_coords = list(coord_dict.keys()) if coord_dict else []
            print(
                f"Warning: The marker '{marker}' has incomplete data. Available coordinates: {available_coords}"
            )

    print(f"Valid markers found: {list(valid_markers.keys())}")

    # Extract the data for each marker into an Nx3 array
    marker_data = {}
    for marker, cols in valid_markers.items():
        marker_data[marker] = df[cols].to_numpy()

    # Handle unit conversion based on user input
    if valid_markers:
        # Create a temporary array with all points to check units
        temp_points = np.stack([marker_data[marker] for marker in valid_markers], axis=1)

        # Calculate statistics for user information
        valid_coords = temp_points[~np.isnan(temp_points)]
        if len(valid_coords) > 0:
            mean_abs_value = np.mean(np.abs(valid_coords))
            min_value = np.min(valid_coords)
            max_value = np.max(valid_coords)

            print("\nCoordinate data statistics:")
            print(f"  Mean absolute value: {mean_abs_value:.3f}")
            print(f"  Range: [{min_value:.3f}, {max_value:.3f}]")
            print("  Data preview (first marker, first 3 records):")
            first_marker = list(marker_data.keys())[0]
            preview_data = marker_data[first_marker][:3]
            for i, coords in enumerate(preview_data):
                if not np.isnan(coords).any():
                    print(
                        f"    Record {i}: X={coords[0]:.6f}, Y={coords[1]:.6f}, Z={coords[2]:.6f}"
                    )

            # Ask user about units
            print("\nPlease select units in the dialog box...")
            user_units = ask_user_units()
            print(f"User selected: {user_units}")

            if user_units == "mm":
                print("Converting from millimeters to meters...")
                # Convert all marker data from millimeters to meters
                for marker in marker_data:
                    marker_data[marker] = marker_data[marker] * 0.001
                print("Unit conversion completed: mm → m")

                # Show converted preview
                print("After conversion (first marker, first 3 records):")
                preview_data = marker_data[first_marker][:3]
                for i, coords in enumerate(preview_data):
                    if not np.isnan(coords).any():
                        print(
                            f"    Record {i}: X={coords[0]:.6f}, Y={coords[1]:.6f}, Z={coords[2]:.6f}"
                        )

        elif user_units == "m":
            print("No conversion needed - data is already in meters.")

        elif user_units == "auto":
            print("Using auto-detection for units...")
            if detect_units(temp_points):
                print("Auto-detection: Data appears to be in millimeters. Converting to meters...")
                # Convert all marker data from millimeters to meters
                for marker in marker_data:
                    marker_data[marker] = marker_data[marker] * 0.001
                print("Unit conversion completed: mm → m")
            else:
                print("Auto-detection: Data appears to be already in meters.")

        # Check for and filter extreme outliers after unit conversion
        print("\nChecking for extreme outliers...")
        temp_points_converted = np.stack([marker_data[marker] for marker in valid_markers], axis=1)
        valid_coords = temp_points_converted[~np.isnan(temp_points_converted)]

        if len(valid_coords) > 0:
            # Calculate statistics to detect outliers
            q25 = np.percentile(valid_coords, 25)
            q75 = np.percentile(valid_coords, 75)
            iqr = q75 - q25
            median = np.median(valid_coords)

            # Define outlier bounds (more conservative for coordinate data)
            outlier_threshold = 10 * iqr  # Allow larger range for coordinate data
            lower_bound = q25 - outlier_threshold
            upper_bound = q75 + outlier_threshold

            # Count outliers
            outliers = (valid_coords < lower_bound) | (valid_coords > upper_bound)
            outlier_count = np.sum(outliers)
            outlier_percentage = (outlier_count / len(valid_coords)) * 100

            print("Data quality check:")
            print(f"  Median: {median:.3f}")
            print(f"  Q25-Q75: [{q25:.3f}, {q75:.3f}]")
            print(f"  IQR: {iqr:.3f}")
            print(f"  Outlier bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
            print(
                f"  Outliers found: {outlier_count}/{len(valid_coords)} ({outlier_percentage:.1f}%)"
            )

            if outlier_percentage > 5:  # If more than 5% outliers
                print("Warning: High percentage of outliers detected!")
                print("This may indicate data quality issues or incorrect units.")
                print("Consider reviewing your data source.")

    else:
        print("Warning: No valid coordinate data found.")

    return index_vector, marker_data, valid_markers, delimiter


###############################################################################
# Function: show_csv (Main Function)
# - Opens a file selection dialog to pick the CSV file.
# - Extracts the marker names (ignoring the first "Time" column).
# - Opens a marker selection dialog.
# - Constructs an array of marker positions of shape (num_frames, num_markers, 3).
# - Prompts the user to choose a visualization method.
# - Launches the PyVista 3D viewer (same as C3D viewer).
###############################################################################
def show_csv(file_path=None):
    """
    Main function to load the CSV, select the markers and plot the data.

    Args:
        file_path (str, optional): Path to the CSV file. If None, opens file dialog.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Running CSV viewer")
    print("================================================")

    # If no file path provided, show file dialog
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = select_file()
        if not file_path:
            print("No file selected.")
            return
    else:
        # Validate file exists
        if not os.path.exists(file_path):
            print(f"[bold red]Error:[/bold red] File not found: {file_path}")
            return
        if not os.path.isfile(file_path):
            print(f"[bold red]Error:[/bold red] Path is not a file: {file_path}")
            return

    try:
        index_vector, marker_data, valid_markers, delimiter = read_csv_generic(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error reading the CSV file: {e}")
        return

    # List the available markers
    available_markers = list(valid_markers.keys())
    print("\n[bold green]File loaded successfully![/bold green]")
    print(f"File: {file_path}")
    print(f"Delimiter detected: '{delimiter}'")
    print(f"First column: '{index_vector.name}' (contains {index_vector.dtype} data)")
    print(f"Number of records: {len(index_vector)}")
    print(f"Number of markers found: {len(available_markers)}")
    print("\n[bold]Available markers:[/bold]")
    for i, marker in enumerate(available_markers, 1):
        print(f"  {i:2d}. {marker}")

    # Allow the user to select the markers to be visualized (multiple selection)
    selected_markers = select_markers_csv(available_markers)
    if not selected_markers:
        messagebox.showwarning("Warning", "No markers selected.")
        return

    # Build an array of points with shape (num_frames, num_markers, 3)
    # CORREÇÃO: Garantir que as colunas estão na ordem correta X, Y, Z
    points_list = []
    for marker in selected_markers:
        marker_coords = marker_data[marker]  # Nx3 array [X, Y, Z]
        points_list.append(marker_coords)

    points = np.stack(points_list, axis=1)  # Shape: (num_frames, num_markers, 3)
    num_frames = points.shape[0]
    num_markers = points.shape[1]

    file_name = os.path.basename(file_path)

    # Debug: Print first few coordinate values to verify correctness
    print("\n[bold]Debug Info:[/bold]")
    print(f"Points array shape: {points.shape}")
    if num_markers > 0:
        first_marker = selected_markers[0]
        first_frame_coords = points[0, 0, :]
        print(f"First marker '{first_marker}' coordinates at first record:")
        print(f"  X: {first_frame_coords[0]:.6f}")
        print(f"  Y: {first_frame_coords[1]:.6f}")
        print(f"  Z: {first_frame_coords[2]:.6f}")

    print("Final coordinate ranges (see data above).")
    viewer_choice = choose_visualizer()
    if viewer_choice == "open3d":
        from vaila.viewc3d import run_viewc3d_from_array

        run_viewc3d_from_array(points, selected_markers, 60.0, file_path)
    else:
        from vaila.viewc3d_pyvista import MokkaLikeViewer

        title = f"Vaila - CSV Viewer (PyVista) | {file_name} | {len(selected_markers)} markers | {num_frames} records"
        MokkaLikeViewer.from_array(points, selected_markers, frame_rate=60.0, title=title)


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    import argparse
    import sys

    # Enhanced command line interface with performance options
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Parse arguments
        parser = argparse.ArgumentParser(description="CSV Viewer with performance options")
        parser.add_argument("file", nargs="?", help="CSV file path")
        parser.add_argument("--no-cache", action="store_true", help="Disable NPZ cache")
        parser.add_argument(
            "--turbo", action="store_true", help="Use turbo mode (faster visualization)"
        )
        parser.add_argument("--fps", type=int, default=30, help="Frames per second")
        parser.add_argument(
            "--stride", type=int, default=1, help="Skip frames (1=all, 2=every other)"
        )

        args = parser.parse_args()

        if args.file:
            show_csv_optimized(
                args.file,
                use_cache=not args.no_cache,
                turbo=args.turbo,
                fps=args.fps,
                stride=args.stride,
            )
        else:
            show_csv()
    else:
        # Original simple interface
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            show_csv(file_path)
        else:
            show_csv()
