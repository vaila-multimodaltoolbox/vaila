"""
===============================================================================
readcsv_export.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Version: 25 September 2024 
Update: 04 September 2025
Version updated: 0.1.1
Python Version: 3.12.11

Description:
This script provides functionality to convert CSV files containing point and analog data
into the C3D format, commonly used for motion capture data analysis. The script uses the
ezc3d library to create C3D files from CSV inputs while sanitizing and formatting the data
to ensure compatibility with the C3D standard.

Main Features:
- Reads point and analog data from user-selected CSV files.
- Sanitizes headers to remove unwanted characters and ensure proper naming conventions.
- Handles user input for data rates, unit conversions, and sorting preferences.
- Converts the CSV data into a C3D file with appropriately formatted point and analog data.
- Provides a user interface for selecting files and entering required information using Tkinter.
- **NEW**: Batch processing capability to convert multiple CSV files in a directory automatically.
- **NEW**: Automatic output directory creation with timestamps for organized file management.
- **NEW**: Comprehensive logging and error reporting system.
- **NEW**: Cross-platform path handling using pathlib.Path.

Functions:
- sanitize_header: Cleans and formats CSV headers to conform to expected data formats.
- convert_csv_to_c3d: Handles user input and coordinates the conversion process from CSV to C3D.
- create_c3d_from_csv: Constructs the C3D file from the sanitized data.
- validate_and_filter_columns: Validates and filters CSV columns to ensure correct formatting.
- get_conversion_factor: Provides a user interface for unit conversion selection.
- **NEW**: batch_convert_csv_to_c3d: Processes all CSV files in a directory automatically.
- **NEW**: auto_create_c3d_from_csv: Creates C3D files without user prompts for batch processing.

Dependencies:
- numpy: For numerical data handling.
- pandas: For data manipulation and reading CSV files.
- ezc3d: To create and write C3D files.
- tkinter: For GUI elements, including file dialogs and message boxes.

Usage:
Run the script and choose between:
1. **Single File Processing**: Select individual CSV files and convert them one by one.
2. **Batch Processing**: Process all CSV files in a directory automatically with common parameters.

For batch processing, the script will:
- Ask for input directory containing CSV files
- Request output directory for results
- Apply common parameters to all files
- Create timestamped output directory
- Process all files automatically
- Show summary of successful/failed conversions
- Generate detailed conversion log file

"""

from pathlib import Path
from rich import print
import numpy as np
import pandas as pd
import re
import ezc3d
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from datetime import datetime

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
    "radians_per_minute": (2 * 3.141592653589793 / 3600, "rad/min"),
    "radians_per_hour": (2 * 3.141592653589793 / 86400, "rad/hr"),
    "watts": (1, "W"),
    "pounds": (0.453592, "lb"),
    "joules": (1, "J"),
    "kilojoules": (1000, "kJ"),
    "watt_hours": (3600, "Wh"),
    "kilojoules_per_hour": (3600, "kWh"),
    "calories": (4.184, "cal"),
    "kilocalories": (4.184, "kcal"),
}


def sanitize_header(header):
    """
    Sanitize the CSV header to ensure it is in the correct format.
    - Remove any existing suffixes (_X, _Y, _Z or .X, .Y, .Z) from column names.
    - Replace any characters that are not letters, numbers, or underscores.
    - Ensure each coordinate column has the correct suffix (_X, _Y, _Z).
    - Fix any empty or incorrectly named columns.
    """
    new_header = [
        header[0]
    ]  # Keep the first column unchanged (typically 'Time' or 'Frame')
    empty_column_counter = 0  # Counter for empty columns

    for i, col in enumerate(header[1:]):  # Start from the second column
        col = col.strip().upper()  # Remove extra spaces and convert to uppercase

        if not col:  # Handle empty columns
            base_name = f"VAILA{empty_column_counter // 3 + 1}"
            suffix = ["_X", "_Y", "_Z"][empty_column_counter % 3]
            new_col = base_name + suffix
            empty_column_counter += 1
        else:
            # Remove any unwanted characters (anything not a letter, number, or underscore)
            col = re.sub(r"[^A-Z0-9_]", "_", col)

            # Remove existing suffixes "_X", "_Y", "_Z", ".X", ".Y", ".Z" if present
            if col.endswith(("_X", "_Y", "_Z")):
                col = col.rsplit("_", 1)[0]

            # Add the correct suffix based on the (i) position to ensure starting at "_X"
            correct_suffix = ["_X", "_Y", "_Z"][(i) % 3]
            new_col = col + correct_suffix

        new_header.append(new_col)

    # Ensure all markers have their _X, _Y, and _Z coordinates
    sanitized_header = []
    i = 0
    while i < len(new_header):
        sanitized_header.append(new_header[i])
        if i > 0 and (i + 2) < len(
            new_header
        ):  # Ensure we have space for a set of _X, _Y, _Z
            base_name = new_header[i].rsplit("_", 1)[
                0
            ]  # Get the base name without the suffix
            sanitized_header.append(base_name + "_Y")
            sanitized_header.append(base_name + "_Z")
            i += 2
        i += 1

    return sanitized_header


def validate_and_filter_columns(df):
    """
    Filter only the columns that are in the correct format, ignoring the first column.
    """
    valid_columns = [df.columns[0]] + [
        col
        for col in df.columns[1:]
        if "_" in col and col.split("_")[-1] in ["X", "Y", "Z"]
    ]
    return df[valid_columns]


def get_conversion_factor():
    """
    Display a window to select the conversion factor for units.
    """
    convert_window = tk.Toplevel()
    convert_window.title("Conversion Factor")
    convert_window.geometry("400x600")

    unit_options = list(CONVERSIONS.keys())

    current_unit_label = tk.Label(convert_window, text="Current Unit:")
    current_unit_label.pack(pady=5)
    current_unit_listbox = tk.Listbox(
        convert_window, selectmode=tk.SINGLE, exportselection=False
    )
    current_unit_listbox.pack(pady=5)
    for unit in unit_options:
        current_unit_listbox.insert(tk.END, unit)

    target_unit_label = tk.Label(convert_window, text="Target Unit:")
    target_unit_label.pack(pady=5)
    target_unit_listbox = tk.Listbox(
        convert_window, selectmode=tk.SINGLE, exportselection=False
    )
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

    return (
        convert_window.conversion_factor
        if hasattr(convert_window, "conversion_factor")
        else 1
    )


def convert_csv_to_c3d():
    """
    Handle the CSV to C3D conversion process, including file selection and user inputs.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Running CSV to C3D conversion")
    print("================================================")
    
    # Always ask user first if they want batch processing
    root = tk.Tk()
    root.withdraw()
    
    print("Asking user for processing mode...")
    choice = messagebox.askyesno(
        "Processing Mode", 
        "Do you want to process all CSV files in a directory?\n\nYes = Batch processing (recommended)\nNo = Single file processing"
    )
    
    if choice:
        print("User chose BATCH processing")
        batch_convert_csv_to_c3d()
        return
    
    print("User chose SINGLE file processing")
    
    # Single file processing
    point_file_path = filedialog.askopenfilename(
        title="Select Point Data CSV", filetypes=[("CSV files", "*.csv")]
    )
    if not point_file_path:
        messagebox.showerror("Error", "No point data file selected.")
        return

    point_df = pd.read_csv(point_file_path)

    print(f"Loaded point data from {point_file_path}")
    print(f"Point data header: {point_df.columns.tolist()}")
    print(f"Point data shape: {point_df.shape}")

    point_df.columns = sanitize_header(point_df.columns)

    use_analog = messagebox.askyesno(
        "Analog Data", "Do you have an analog data CSV file to add?"
    )
    analog_df = None

    if use_analog:
        analog_file_path = filedialog.askopenfilename(
            title="Select Analog Data CSV", filetypes=[("CSV files", "*.csv")]
        )
        if analog_file_path:
            analog_df = pd.read_csv(analog_file_path)
            analog_df.columns = sanitize_header(analog_df.columns)

            print(f"Loaded analog data from {analog_file_path}")
            print(f"Analog data header: {analog_df.columns.tolist()}")
            print(f"Analog data shape: {analog_df.shape}")

    point_rate = simpledialog.askinteger(
        "Point Rate", "Enter the point data rate (Hz):", minvalue=1, initialvalue=100
    )
    analog_rate = 1000
    if analog_df is not None:
        analog_rate = simpledialog.askinteger(
            "Analog Rate",
            "Enter the analog data rate (Hz):",
            minvalue=1,
            initialvalue=1000,
        )

    conversion_factor = get_conversion_factor()

    sort_markers = messagebox.askyesno(
        "Sort Markers", "Do you want to sort markers alphabetically?"
    )

    try:
        create_c3d_from_csv(
            point_df,
            analog_df,
            point_rate,
            analog_rate,
            conversion_factor,
            sort_markers,
        )
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while creating C3D file: {e}")


def create_c3d_from_csv(
    points_df,
    analog_df=None,
    point_rate=100,
    analog_rate=1000,
    conversion_factor=1,
    sort_markers=False,
):
    """
    Create a C3D file from the given point and analog data.
    """
    print("Creating C3D from CSV...")

    c3d = ezc3d.c3d()
    print("Initialized empty C3D object.")

    points_df = validate_and_filter_columns(points_df)
    print("Filtered and sanitized columns for points:", points_df.columns.tolist())

    marker_labels = [col.rsplit("_", 1)[0] for col in points_df.columns[1::3]]
    if sort_markers:
        marker_labels.sort()
    print("Extracted marker labels:", marker_labels)

    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_labels
    c3d["parameters"]["POINT"]["RATE"]["value"] = [point_rate]

    num_markers = len(marker_labels)
    num_frames = len(points_df)
    print(f"Number of markers: {num_markers}, Number of frames: {num_frames}")

    points_data = np.zeros((4, num_markers, num_frames))
    print("Initialized points data array with shape:", points_data.shape)

    for i, label in enumerate(marker_labels):
        try:
            points_data[0, i, :] = points_df[f"{label}_X"].values * conversion_factor
            points_data[1, i, :] = points_df[f"{label}_Y"].values * conversion_factor
            points_data[2, i, :] = points_df[f"{label}_Z"].values * conversion_factor
            points_data[3, i, :] = 1  # Coordenada homogênea
        except KeyError as e:
            print(f"Error accessing data for label '{label}': {e}")
            raise

    print("Points data populated successfully.")
    c3d["data"]["points"] = points_data
    print("Assigned points data to C3D.")

    if analog_df is not None:
        analog_labels = list(analog_df.columns[1:])
        num_analog = len(analog_labels)
        print("Analog labels:", analog_labels)
        c3d["parameters"]["ANALOG"]["LABELS"]["value"] = analog_labels
        c3d["parameters"]["ANALOG"]["RATE"]["value"] = [analog_rate]

        num_analog_frames = analog_df.shape[0]
        analog_data = np.zeros((1, num_analog, num_analog_frames))

        for i, label in enumerate(analog_labels):
            try:
                analog_data[0, i, :] = analog_df[label].values
            except KeyError as e:
                print(f"Error accessing analog data for label '{label}': {e}")
                raise

        print(f"Analog data shape: {analog_data.shape}")
        c3d["data"]["analogs"] = analog_data
        print("Analog data assigned to C3D.")

    print(f"Final POINT RATE: {c3d['parameters']['POINT']['RATE']['value']}")
    print(f"Final ANALOG RATE: {c3d['parameters']['ANALOG']['RATE']['value']}")
    print(f"Final POINT SHAPE: {c3d['data']['points'].shape}")
    print(f"Final ANALOG SHAPE: {c3d['data']['analogs'].shape}")

    output_path = filedialog.asksaveasfilename(
        defaultextension=".c3d", filetypes=[("C3D files", "*.c3d")]
    )
    if output_path:
        try:
            c3d.write(output_path)
            print(f"C3D file saved to {output_path}.")
            messagebox.showinfo("Success", f"C3D file saved to {output_path}")
        except Exception as e:
            print(f"Error writing C3D file: {e}")
            messagebox.showerror("Error", f"Failed to save C3D file: {e}")
    else:
        print("Save operation cancelled.")
        messagebox.showwarning("Warning", "Save operation cancelled.")


def batch_convert_csv_to_c3d():
    """
    Handle batch CSV to C3D conversion process for all CSV files in a directory.
    """
    print("="*60)
    print("BATCH CSV TO C3D CONVERSION")
    print("="*60)
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting BATCH processing mode...")
    print("="*60)
    
    root = tk.Tk()
    root.withdraw()

    print("Step 1: Selecting input directory...")
    # Select input directory containing CSV files
    input_directory = filedialog.askdirectory(title="Select Input Directory with CSV Files")
    if not input_directory:
        print("No input directory selected. Exiting.")
        messagebox.showerror("Error", "No input directory selected.")
        return
    
    print(f"Input directory selected: {input_directory}")

    print("Step 2: Selecting output directory...")
    # Select output directory
    output_directory = filedialog.askdirectory(title="Select Output Directory")
    if not output_directory:
        print("No output directory selected. Exiting.")
        messagebox.showerror("Error", "No output directory selected.")
        return
    
    print(f"Output directory selected: {output_directory}")

    print("Step 3: Scanning for CSV files...")
    # Get all CSV files in the input directory (excluding hidden files that start with '.')
    input_path = Path(input_directory)
    csv_files = [f.name for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == '.csv' and not f.name.startswith('.')]
    if not csv_files:
        print(f"ERROR: No visible CSV files found in {input_directory}")
        messagebox.showerror("Error", f"No visible CSV files found in {input_directory}")
        return

    print(f"Found {len(csv_files)} visible CSV files in directory")

    # Filter out analog files to avoid double processing
    point_csv_files = []
    for csv_file in csv_files:
        if not any(suffix in csv_file.lower() for suffix in ['_analog', '_force', '_emg', '_sensor', '_analog_data']):
            point_csv_files.append(csv_file)
    
    if not point_csv_files:
        print(f"ERROR: No point data CSV files found in {input_directory}")
        messagebox.showerror("Error", f"No point data CSV files found in {input_directory}")
        return

    print(f"Processing {len(point_csv_files)} point data files:")
    for f in point_csv_files:
        print(f"  - {f}")
    
    print(f"Filtered out {len(csv_files) - len(point_csv_files)} analog/data files")

    print("Step 4: Getting user parameters...")
    # Ask user for common parameters
    point_rate = simpledialog.askinteger(
        "Point Rate", "Enter the point data rate (Hz):", minvalue=1, initialvalue=100
    )
    if not point_rate:
        print("No point rate specified. Using default: 100 Hz")
        point_rate = 100
    
    print(f"Point rate set to: {point_rate} Hz")
    
    use_analog = messagebox.askyesno(
        "Analog Data", "Do you have analog data CSV files to add? (Should be in same directory)"
    )
    print(f"Analog data processing: {'Yes' if use_analog else 'No'}")
    
    analog_rate = 1000
    if use_analog:
        analog_rate = simpledialog.askinteger(
            "Analog Rate",
            "Enter the analog data rate (Hz):",
            minvalue=1,
            initialvalue=1000,
        )
        if not analog_rate:
            print("No analog rate specified. Using default: 1000 Hz")
            analog_rate = 1000
        print(f"Analog rate set to: {analog_rate} Hz")

    print("Getting conversion factor...")
    conversion_factor = get_conversion_factor()
    print(f"Conversion factor: {conversion_factor}")
    
    sort_markers = messagebox.askyesno(
        "Sort Markers", "Do you want to sort markers alphabetically?"
    )
    print(f"Sort markers: {'Yes' if sort_markers else 'No'}")

    print("Step 5: Creating output directory...")
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_dir = Path(output_directory) / f"csv2c3d_{timestamp}"
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Batch output directory created: {batch_output_dir}")
    print(f"Timestamp: {timestamp}")

    print("Step 6: Starting batch processing...")
    print("="*50)
    
    # Initialize tracking variables
    successful_conversions = 0
    failed_conversions = 0
    error_details = []
    successful_files = []
    failed_files = []
    
    # Create log file
    log_file_path = batch_output_dir / "conversion_log.txt"
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # Write header to log file
    log_file.write("="*80 + "\n")
    log_file.write("CSV TO C3D BATCH CONVERSION LOG\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Input Directory: {input_directory}\n")
    log_file.write(f"Output Directory: {batch_output_dir}\n")
    log_file.write(f"Point Rate: {point_rate} Hz\n")
    log_file.write(f"Analog Rate: {analog_rate} Hz\n")
    log_file.write(f"Conversion Factor: {conversion_factor}\n")
    log_file.write(f"Sort Markers: {'Yes' if sort_markers else 'No'}\n")
    log_file.write(f"Visible CSV Files Found: {len(csv_files)}\n")
    log_file.write(f"Point Data Files to Process: {len(point_csv_files)}\n")
    log_file.write("="*80 + "\n\n")
    
    # Process each CSV file
    for i, csv_file in enumerate(point_csv_files, 1):
        print(f"\nProcessing file {i}/{len(point_csv_files)}: {csv_file}")
        log_file.write(f"\n--- Processing File {i}/{len(point_csv_files)}: {csv_file} ---\n")
        
        try:
            print(f"\nProcessing: {csv_file}")
            log_file.write(f"Status: Processing started\n")
            
            # Read the CSV file
            csv_path = Path(input_directory) / csv_file
            point_df = pd.read_csv(csv_path)
            
            log_file.write(f"CSV loaded successfully - Shape: {point_df.shape}\n")
            log_file.write(f"Original columns: {list(point_df.columns)}\n")
            
            # Sanitize headers
            point_df.columns = sanitize_header(point_df.columns)
            log_file.write(f"Sanitized columns: {list(point_df.columns)}\n")
            
            # Look for corresponding analog file (more flexible naming)
            analog_df = None
            if use_analog:
                # Try multiple naming patterns for analog files
                possible_analog_names = [
                    csv_file.replace('.csv', '_analog.csv'),
                    csv_file.replace('.csv', '_analog_data.csv'),
                    csv_file.replace('.csv', '_analog.csv'),
                    csv_file.replace('.csv', '_force.csv'),
                    csv_file.replace('.csv', '_emg.csv'),
                    csv_file.replace('.csv', '_sensor.csv')
                ]
                
                for analog_name in possible_analog_names:
                    analog_path = Path(input_directory) / analog_name
                    if analog_path.exists():
                        try:
                            analog_df = pd.read_csv(analog_path)
                            analog_df.columns = sanitize_header(analog_df.columns)
                            print(f"Found analog file: {analog_name}")
                            log_file.write(f"Analog file found: {analog_name} - Shape: {analog_df.shape}\n")
                            break
                        except Exception as e:
                            print(f"Warning: Could not read analog file {analog_name}: {e}")
                            log_file.write(f"Warning: Could not read analog file {analog_name}: {e}\n")
                            continue
            
            # Create output filename
            base_name = Path(csv_file).stem
            output_filename = f"{base_name}.c3d"
            output_path = batch_output_dir / output_filename
            
            # Convert to C3D
            auto_create_c3d_from_csv(
                point_df,
                output_path,
                analog_df,
                point_rate,
                analog_rate,
                conversion_factor,
                sort_markers,
            )
            
            successful_conversions += 1
            successful_files.append(csv_file)
            print(f"Successfully converted: {csv_file} -> {output_filename}")
            log_file.write(f"Status: SUCCESS - C3D file created: {output_filename}\n")
            
        except Exception as e:
            failed_conversions += 1
            failed_files.append(csv_file)
            error_msg = str(e)
            error_details.append((csv_file, error_msg))
            
            print(f"ERROR processing {csv_file}: {e}")
            print("Continuing with next file...")
            
            log_file.write(f"Status: FAILED - Error: {error_msg}\n")
            log_file.write(f"Error type: {type(e).__name__}\n")
            
            # Add more context for common errors
            if "utf-8" in error_msg.lower():
                log_file.write(f"Context: This appears to be a UTF-8 encoding issue\n")
            elif "keyerror" in error_msg.lower():
                log_file.write(f"Context: This appears to be a column/key access issue\n")
            elif "shape" in error_msg.lower():
                log_file.write(f"Context: This appears to be a data shape/dimension issue\n")
            
            continue

    # Write summary to log file
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("CONVERSION SUMMARY\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Visible CSV files found: {len(csv_files)}\n")
    log_file.write(f"Point data files processed: {len(point_csv_files)}\n")
    log_file.write(f"Successful conversions: {successful_conversions}\n")
    log_file.write(f"Failed conversions: {failed_conversions}\n")
    log_file.write(f"Success rate: {(successful_conversions/len(point_csv_files)*100):.1f}%\n")
    
    if successful_files:
        log_file.write(f"\nSUCCESSFUL CONVERSIONS ({len(successful_files)}):\n")
        for file in successful_files:
            log_file.write(f"  ✓ {file}\n")
    
    if failed_files:
        log_file.write(f"\nFAILED CONVERSIONS ({len(failed_files)}):\n")
        for file, error in error_details:
            log_file.write(f"  ✗ {file} - Error: {error}\n")
    
    # Analyze error patterns
    if error_details:
        log_file.write(f"\nERROR ANALYSIS:\n")
        error_types = {}
        for file, error in error_details:
            error_type = type(error).__name__ if hasattr(error, '__class__') else "Unknown"
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(file)
        
        for error_type, files in error_types.items():
            log_file.write(f"  {error_type}: {len(files)} files\n")
            for file in files:
                log_file.write(f"    - {file}\n")
    
    log_file.write(f"\nOutput directory: {batch_output_dir}\n")
    log_file.write(f"Log file: {log_file_path}\n")
    log_file.write("="*80 + "\n")
    log_file.close()
    
    print(f"\nDetailed log saved to: {log_file_path}")
    
    # Show final results
    print(f"\n{'='*60}")
    print("BATCH CONVERSION COMPLETED")
    print(f"{'='*60}")
    print(f"Visible CSV files found: {len(csv_files)}")
    print(f"Point data files processed: {len(point_csv_files)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Success rate: {(successful_conversions/len(point_csv_files)*100):.1f}%")
    print(f"Output directory: {batch_output_dir}")
    print(f"Detailed log: {log_file_path}")
    print(f"{'='*60}")
    
    if failed_conversions == 0:
        print("PERFECT! All files converted successfully!")
    elif successful_conversions > failed_conversions:
        print("Good! Most files converted successfully.")
    else:
        print("Warning: Many files failed to convert.")
    
    # Show error summary if there were failures
    if failed_conversions > 0:
        print("\nERROR SUMMARY:")
        print(f"Failed files: {failed_conversions}")
        print("Most common errors:")
        
        error_counts = {}
        for file, error in error_details:
            error_msg = str(error)
            if error_msg not in error_counts:
                error_counts[error_msg] = 0
            error_counts[error_msg] += 1
        
        # Show top 5 most common errors
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (error_msg, count) in enumerate(sorted_errors[:5]):
            print(f"  {i+1}. {error_msg[:100]}{'...' if len(error_msg) > 100 else ''} ({count} files)")
    
    message = f"Batch conversion completed!\n\nVisible CSV files: {len(csv_files)}\nPoint data files: {len(point_csv_files)}\nSuccessful: {successful_conversions}\nFailed: {failed_conversions}\nSuccess rate: {(successful_conversions/len(point_csv_files)*100):.1f}%\n\nOutput directory: {batch_output_dir}\nDetailed log: {log_file_path.name}"
    messagebox.showinfo("Batch Conversion Complete", message)


def auto_create_c3d_from_csv(
    points_df,
    output_path,
    analog_df=None,
    point_rate=100,
    analog_rate=1000,
    conversion_factor=1,
    sort_markers=False,
):
    """
    Create a C3D file from the given points DataFrame and automatically
    saves it to the specified output_path without prompting the user.
    
    Args:
        points_df (pd.DataFrame): DataFrame containing point data with headers.
        output_path (str): Full file path where the C3D file should be saved.
        analog_df (pd.DataFrame, optional): DataFrame with analog data if available.
        point_rate (int): Point data sampling rate.
        analog_rate (int): Analog data sampling rate.
        conversion_factor (float): Conversion factor for the point coordinates.
        sort_markers (bool): Whether to sort marker labels alphabetically.
    
    Raises:
        Exception: If there is an error writing the C3D file.
    """
    print("Creating C3D from CSV (auto mode)...")
    
    try:
        c3d = ezc3d.c3d()
        print("Initialized empty C3D object.")
    except Exception as e:
        raise Exception(f"Failed to initialize C3D object: {e}")
    
    try:
        points_df = validate_and_filter_columns(points_df)
        print("Filtered and sanitized columns for points:", points_df.columns.tolist())
    except Exception as e:
        raise Exception(f"Failed to validate and filter columns: {e}")
    
    try:
        marker_labels = [col.rsplit("_", 1)[0] for col in points_df.columns[1::3]]
        if sort_markers:
            marker_labels.sort()
        print("Marker labels for C3D:", marker_labels)
        
        if not marker_labels:
            raise Exception("No valid marker labels found. Check if CSV has proper X, Y, Z column structure.")
    except Exception as e:
        raise Exception(f"Failed to extract marker labels: {e}")
    
    try:
        c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]
        c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_labels
        c3d["parameters"]["POINT"]["RATE"]["value"] = [point_rate]
    except Exception as e:
        raise Exception(f"Failed to set C3D point parameters: {e}")

    num_markers = len(marker_labels)
    num_frames = len(points_df)
    print(f"Number of markers: {num_markers}, Number of frames: {num_frames}")
    
    if num_frames == 0:
        raise Exception("CSV file contains no data rows")
    
    if num_markers == 0:
        raise Exception("No valid markers found in CSV data")

    try:
        points_data = np.zeros((4, num_markers, num_frames))
        print("Initialized points data array with shape:", points_data.shape)
    except Exception as e:
        raise Exception(f"Failed to initialize points data array: {e}")
    
    # Populate points data with better error handling
    for i, label in enumerate(marker_labels):
        try:
            x_col = f"{label}_X"
            y_col = f"{label}_Y"
            z_col = f"{label}_Z"
            
            # Check if columns exist
            if x_col not in points_df.columns:
                raise KeyError(f"Column {x_col} not found in CSV")
            if y_col not in points_df.columns:
                raise KeyError(f"Column {y_col} not found in CSV")
            if z_col not in points_df.columns:
                raise KeyError(f"Column {z_col} not found in CSV")
            
            # Check for NaN values
            x_data = points_df[x_col].values
            y_data = points_df[y_col].values
            z_data = points_df[z_col].values
            
            if np.any(np.isnan(x_data)) or np.any(np.isnan(y_data)) or np.any(np.isnan(z_data)):
                print(f"Warning: NaN values found in marker {label}, replacing with 0")
                x_data = np.nan_to_num(x_data, nan=0.0)
                y_data = np.nan_to_num(y_data, nan=0.0)
                z_data = np.nan_to_num(z_data, nan=0.0)
            
            points_data[0, i, :] = x_data * conversion_factor
            points_data[1, i, :] = y_data * conversion_factor
            points_data[2, i, :] = z_data * conversion_factor
            points_data[3, i, :] = 1  # Homogeneous coordinate
            
        except KeyError as e:
            raise KeyError(f"Error accessing data for marker '{label}': {e}")
        except Exception as e:
            raise Exception(f"Error processing data for marker '{label}': {e}")
    
    try:
        c3d["data"]["points"] = points_data
        print("Points data assigned to C3D successfully.")
    except Exception as e:
        raise Exception(f"Failed to assign points data to C3D: {e}")

    # Handle analog data if provided
    if analog_df is not None:
        try:
            analog_labels = list(analog_df.columns[1:])
            num_analog = len(analog_labels)
            print(f"Processing {num_analog} analog channels")
            
            if num_analog > 0:
                c3d["parameters"]["ANALOG"]["LABELS"]["value"] = analog_labels
                c3d["parameters"]["ANALOG"]["RATE"]["value"] = [analog_rate]
                
                num_analog_frames = analog_df.shape[0]
                analog_data = np.zeros((1, num_analog, num_analog_frames))
                
                for i, label in enumerate(analog_labels):
                    try:
                        analog_values = analog_df[label].values
                        # Handle NaN values in analog data
                        if np.any(np.isnan(analog_values)):
                            print(f"Warning: NaN values found in analog channel {label}, replacing with 0")
                            analog_values = np.nan_to_num(analog_values, nan=0.0)
                        analog_data[0, i, :] = analog_values
                    except KeyError as e:
                        raise KeyError(f"Error accessing analog data for channel '{label}': {e}")
                    except Exception as e:
                        raise Exception(f"Error processing analog data for channel '{label}': {e}")
                
                c3d["data"]["analogs"] = analog_data
                print(f"Analog data assigned to C3D successfully. Shape: {analog_data.shape}")
            else:
                print("No analog channels found, skipping analog data")
                
        except Exception as e:
            print(f"Warning: Failed to process analog data: {e}")
            print("Continuing without analog data...")

    try:
        # Convert Path object to string for ezc3d compatibility
        output_path_str = str(output_path)
        c3d.write(output_path_str)
        print(f"C3D file saved successfully to {output_path_str}")
    except Exception as e:
        raise Exception(f"Failed to save C3D file to {output_path}: {e}")


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("CSV TO C3D CONVERTER - VAILA")
    print("="*60)
    print("Starting application...")
    
    # Check command line arguments for batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        print("BATCH MODE forced via command line argument")
        try:
            batch_convert_csv_to_c3d()
        except Exception as e:
            print(f"Error in batch processing: {e}")
            messagebox.showerror("Error", f"Batch processing failed: {e}")
        sys.exit(0)
    
    # Ask user if they want batch processing or single file
    root = tk.Tk()
    root.withdraw()
    
    print("Showing processing mode selection dialog...")
    choice = messagebox.askyesno(
        "Processing Mode", 
        "Do you want to process all CSV files in a directory?\n\nYes = Batch processing (recommended)\nNo = Single file processing"
    )
    
    print(f"User choice: {'Batch processing' if choice else 'Single file processing'}")
    
    if choice:
        print("Starting BATCH processing...")
        try:
            batch_convert_csv_to_c3d()
        except Exception as e:
            print(f"Error in batch processing: {e}")
            messagebox.showerror("Error", f"Batch processing failed: {e}")
    else:
        print("Starting SINGLE file processing...")
        try:
            convert_csv_to_c3d()
        except Exception as e:
            print(f"Error in single file processing: {e}")
            messagebox.showerror("Error", f"Single file processing failed: {e}")
    
    print("Application finished.")
