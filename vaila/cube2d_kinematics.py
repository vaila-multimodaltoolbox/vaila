"""
===============================================================================
cube2d_kinematics.py
===============================================================================
Author: Prof. Dr. Paulo Roberto Pereira Santiago
created: 2025-01-16
updated: 2025-06-09
version: 0.0.3
python version: 3.12.9

Description:
This module provides functionality for analyzing 2D kinematics data from cube-based movement assessments.

Key Features:
- Processes CSV files containing 2D position data (x,y coordinates)
- Calculates kinematic metrics including:
  - Total distance traveled
  - Average speed
  - Time spent stationary
  - Total movement time
- Divides movement space into 9 quadrants for spatial analysis
- Generates visualizations:
  - Movement pathway plots with quadrant overlays
  - Color-coded speed profiles
- Handles batch processing of multiple files
- Outputs results to organized directories with timestamps
- Provides GUI interface for:
  - File/directory selection
  - Parameter input
  - Results viewing

Dependencies:
- NumPy: Numerical computations and array operations
- Pandas: Data manipulation and analysis
- Matplotlib: Data visualization and plotting
- Tkinter: Graphical user interface
- SciPy: Signal processing functions

The module is designed for research and clinical applications in movement science,
particularly for analyzing confined space movement patterns like those seen in
agility or balance tests using a 9-square grid layout.

How to use:

1. Run the module using the run_cube2d_kinematics() function.
2. Select the data directory containing the CSV files.
3. Select the quadrants file (optional, default is a 9-square grid).
4. Select the output directory for the results.
5. Enter the sampling frequency (fs) for all files.
6. The module will process all CSV files in the selected directory,
   generate pathway plots with quadrants, and save the results to the output directory.

File need to be in the following format:
The input CSV files should contain two columns:
- Column 1: X coordinates (medio-lateral position in meters)
- Column 2: Y coordinates (antero-posterior position in meters)

Example format:
x,y
0.1,0.2
0.15,0.25
0.2,0.3
...

The data should be sampled at a consistent frequency (fs) which will be specified
during processing. The coordinates should represent positions in meters relative
to a defined origin point (typically the center of the movement area).

Example files can be found in the tests/Cube2d_kinematics directory.
conda activate vaila
python vaila/cube2d_kinematics.py
"""

import os
from datetime import datetime
from tkinter import (
    BOTH,
    RIGHT,
    WORD,
    Button,
    Frame,
    Scrollbar,
    Text,
    Tk,
    Toplevel,
    Y,
    filedialog,
    messagebox,
    simpledialog,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import butter, filtfilt

# Define the default quadrants using numpy arrays
quadrants = np.array(
    [
        [1, 0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0],
        [2, 0.0, 0.5, 0.0, 1.0, 0.5, 1.0, 0.5, 0.5],
        [3, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5],
        [4, 1.0, 0.5, 1.0, 1.0, 1.5, 1.0, 1.5, 0.5],
        [5, 1.0, 0.0, 1.0, 0.5, 1.5, 0.5, 1.5, 0.0],
        [6, 1.0, -0.5, 1.0, 0.0, 1.5, 0.0, 1.5, -0.5],
        [7, 0.5, -0.5, 0.5, 0.0, 1.0, 0.0, 1.0, -0.5],
        [8, 0.0, -0.5, 0.0, 0.0, 0.5, 0.0, 0.5, -0.5],
        [9, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0],
    ]
)

# Define column names for better understanding
column_names = [
    "quadrant",
    "vertex1_x",
    "vertex1_y",
    "vertex2_x",
    "vertex2_y",
    "vertex3_x",
    "vertex3_y",
    "vertex4_x",
    "vertex4_y",
]


def show_instructions():
    """
    Display a comprehensive instruction window before file selection.
    """
    root = Tk()
    root.withdraw()  # Hide the main window

    # Create instruction window
    instruction_window = Toplevel(root)
    instruction_window.title("CUBE 2D Kinematics Analysis - Instructions")
    instruction_window.geometry("700x600")
    instruction_window.grab_set()  # Make it modal

    # Create scrollable text widget
    frame = Frame(instruction_window)
    frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

    scrollbar = Scrollbar(frame)
    scrollbar.pack(side=RIGHT, fill=Y)

    text_widget = Text(frame, wrap=WORD, yscrollcommand=scrollbar.set, font=("Arial", 11))
    text_widget.pack(side="left", fill=BOTH, expand=True)
    scrollbar.config(command=text_widget.yview)

    instructions = """
CUBE 2D KINEMATICS ANALYSIS - SETUP INSTRUCTIONS

This tool analyzes 2D (meters) movement patterns in a cube/grid environment. Please follow these steps:

═══════════════════════════════════════════════════════════════════════════════════

1. PREPARE YOUR DATA FILES

   Required Files:
   • CSV files containing position data in meters
   • All CSV files should be in the same directory
   • Files must have consistent sampling frequency

   CSV File Format (SIMPLIFIED):
   • Any header row (will be ignored)
   • Exactly 3 columns minimum:
     - Column 0: Any data (frame, time, etc.) - will be ignored
     - Column 1: X coordinates in meters (medio-lateral)
     - Column 2: Y coordinates in meters (antero-posterior)
   • Coordinates relative to center origin (0,0)

   Example CSV content:
   frame,x,y
   0,0.656807,0.194554
   1,0.582177,0.299958
   2,0.560044,0.310084
   ...

   OR:

   time,p33_x,p33_y
   0.0,0.656807,0.194554
   0.033,0.582177,0.299958
   0.066,0.560044,0.310084
   ...

   The system will ALWAYS use:
   • Column 1 as X coordinates
   • Column 2 as Y coordinates
   • Header names are ignored

═══════════════════════════════════════════════════════════════════════════════════

2. OUTPUT FILES GENERATED

   For each input file, the system creates:
   • Movement pathway plot (PNG)
   • Individual results CSV (database-ready)
   • Summary text file (human-readable)
   • Consolidated database CSV (all results combined)

   Database CSV Format:
   - file_name: Input file name
   - analysis_date: Processing date
   - analysis_time: Processing time
   - sampling_frequency_hz: Sampling rate
   - total_distance_m: Total distance traveled
   - average_speed_ms: Average movement speed
   - maximum_speed_ms: Peak speed
   - time_stationary_s: Time below 0.05 m/s
   - total_time_s: Total recording time
   - movement_percentage: Percentage of time moving
   - data_points: Number of data samples

═══════════════════════════════════════════════════════════════════════════════════

3. QUADRANTS FILE (OPTIONAL)

   • Text file defining the 9-quadrant grid layout
   • If not provided, default 3x3 grid will be used
   • Default quadrants cover a 1.5m x 1.5m area centered at origin

═══════════════════════════════════════════════════════════════════════════════════

4. PROCESSING DETAILS

   The analysis will:
   • Apply low-pass filtering (6 Hz cutoff) to smooth trajectories
   • Calculate comprehensive kinematic metrics
   • Generate pathway visualization with time-based color gradient
   • Create database-ready CSV files for easy analysis
   • Provide both individual and consolidated results

═══════════════════════════════════════════════════════════════════════════════════

CLICK 'CONTINUE' TO START FILE SELECTION
"""

    text_widget.insert("1.0", instructions)
    text_widget.config(state="disabled")  # Make text read-only

    # Add continue button
    button_frame = Frame(instruction_window)
    button_frame.pack(pady=10)

    continue_clicked = [False]  # Use list to allow modification in nested function

    def on_continue():
        continue_clicked[0] = True
        instruction_window.destroy()
        root.quit()

    continue_button = Button(
        button_frame,
        text="CONTINUE",
        command=on_continue,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=20,
        pady=10,
    )
    continue_button.pack()

    # Center the window
    instruction_window.update_idletasks()
    x = (instruction_window.winfo_screenwidth() // 2) - (instruction_window.winfo_width() // 2)
    y = (instruction_window.winfo_screenheight() // 2) - (instruction_window.winfo_height() // 2)
    instruction_window.geometry(f"+{x}+{y}")

    root.mainloop()
    root.destroy()

    return continue_clicked[0]


def load_quadrants(file_path=None):
    """
    Load quadrants from a file or return default quadrants as a pandas DataFrame.
    """
    # Convert default quadrants to DataFrame
    default_df = pd.DataFrame(quadrants, columns=column_names)

    if file_path and file_path.endswith(".txt"):
        try:
            # Read the txt file into a DataFrame and convert to numeric values
            loaded_df = pd.read_csv(file_path)
            # Convert all columns except 'quadrant' to float
            for col in loaded_df.columns:
                if col != "quadrant":
                    loaded_df[col] = pd.to_numeric(loaded_df[col], errors="coerce")
            return loaded_df
        except Exception as e:
            print(f"Error reading quadrants file: {e}. Using default quadrants.")
            return default_df
    return default_df


def load_data(input_file):
    """
    Loads the input file using column positions: column 1 = X, column 2 = Y.
    Column 0 can be frame, time, or any other data.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        tuple: Tuple containing arrays for X and Y coordinates.
    """
    try:
        # Load the data, skipping the header
        data = np.genfromtxt(input_file, delimiter=",", skip_header=1)

        # Handle single column case
        if data.ndim == 1:
            raise ValueError("File must contain at least 3 columns (any, X, Y)")

        # Check if we have at least 3 columns
        if data.shape[1] < 3:
            raise ValueError("File must contain at least 3 columns (any, X, Y)")

        # Always use column 1 as X and column 2 as Y
        x = data[:, 1]  # Column 1 = X coordinates
        y = data[:, 2]  # Column 2 = Y coordinates

        return x, y

    except Exception as e:
        raise ValueError(f"Error loading data from {input_file}: {e}")


def butter_lowpass_filter(data, cutoff, fs, order=4, padding=True):
    """
    Applies a Butterworth low-pass filter to the input data with optional padding.

    Parameters:
    - data: array-like
        The input signal to be filtered.
    - cutoff: float
        The cutoff frequency for the low-pass filter.
    - fs: float
        The sampling frequency of the signal.
    - order: int, default=4
        The order of the Butterworth filter.
    - padding: bool, default=True
        Whether to pad the signal to mitigate edge effects.

    Returns:
    - filtered_data: array-like
        The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    data = np.asarray(data)

    if padding:
        data_len = len(data)
        max_padlen = data_len - 1
        padlen = min(int(fs), max_padlen, 15)

        if data_len <= padlen:
            raise ValueError(
                f"The length of the input data ({data_len}) must be greater than the padding length ({padlen})."
            )

        # Apply reflection padding
        padded_data = np.pad(data, pad_width=(padlen, padlen), mode="reflect")
        filtered_padded_data = filtfilt(b, a, padded_data, padlen=0)
        filtered_data = filtered_padded_data[padlen:-padlen]
    else:
        filtered_data = filtfilt(b, a, data, padlen=0)

    return filtered_data


def calculate_distance(x, y):
    """Calculate instantaneous distance between consecutive points."""
    distance = np.insert(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2), 0, 0)
    return distance


def calculate_speed(distance, fs):
    """Calculate instantaneous speed from distance and sampling frequency."""
    return distance * fs


def plot_pathway_with_quadrants(x, y, quadrants_df, time_vector):
    """Plot pathway with time-based color gradient and quadrants."""

    # Plot the quadrants and pathway
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw each quadrant
    for _, quad in quadrants_df.iterrows():
        # Extract vertices
        vertices_x = [
            quad["vertex1_x"],
            quad["vertex2_x"],
            quad["vertex3_x"],
            quad["vertex4_x"],
            quad["vertex1_x"],
        ]
        vertices_y = [
            quad["vertex1_y"],
            quad["vertex2_y"],
            quad["vertex3_y"],
            quad["vertex4_y"],
            quad["vertex1_y"],
        ]

        # Draw quadrant boundaries
        ax.plot(vertices_x, vertices_y, color="gray", linewidth=2, alpha=0.7)

        # Calculate quadrant center for numbering
        center_x = np.mean(vertices_x[:-1])  # Exclude repeated last point
        center_y = np.mean(vertices_y[:-1])

        # Adjust position for quadrant 1 text
        if int(quad["quadrant"]) == 1:
            center_y -= 0.25

        # Add quadrant number
        ax.text(
            float(center_x),
            float(center_y),
            str(int(quad["quadrant"])),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="circle", facecolor="white", alpha=0.8),
        )

    # Create color gradient for pathway
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(0, 1)
    lc = LineCollection(segments.tolist(), cmap="plasma", norm=norm)
    lc.set_array(np.linspace(0, 1, len(time_vector)))
    lc.set_linewidth(3)
    line = ax.add_collection(lc)

    # Plot start point (green) and end point (red)
    ax.scatter(
        x[0],
        y[0],
        color="green",
        s=150,
        zorder=5,
        label="Start",
        edgecolors="black",
        linewidth=2,
    )
    ax.scatter(
        x[-1],
        y[-1],
        color="red",
        s=150,
        zorder=5,
        label="End",
        edgecolors="black",
        linewidth=2,
    )

    # Add colorbar
    cbar = plt.colorbar(line, ax=ax, orientation="vertical", shrink=0.8)
    cbar.set_label("Time (s)", fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f"{time_vector[0]:.2f}", f"{time_vector[-1]:.2f}"])

    # Configure plot
    ax.set_title("CUBE 2D Pathway with Time-Based Color Gradient", fontsize=16, fontweight="bold")
    ax.set_xlabel("X - Medio-lateral (m)", fontsize=12)
    ax.set_ylabel("Y - Antero-posterior (m)", fontsize=12)
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.axvline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_aspect("equal", "box")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def process_file(file_path, quadrants_df, output_dir, fs, base_name):
    """Process a single file and generate results."""
    try:
        # Load and filter data
        x, y = load_data(file_path)
        x = butter_lowpass_filter(x, 6, fs)
        y = butter_lowpass_filter(y, 6, fs)

        # Calculate metrics
        distance = calculate_distance(x, y)
        speed = calculate_speed(distance, fs)
        total_distance = np.sum(distance)
        avg_speed = np.mean(speed)
        max_speed = np.max(speed)
        time_stationary = np.sum(speed < 0.05) / fs  # Time below 0.05 m/s
        total_time = len(x) / fs  # Total time in seconds
        movement_percentage = (total_time - time_stationary) / total_time * 100

        # Create time vector
        time_vector = np.linspace(0, (len(x) - 1) / fs, len(x))

        # Plot and save pathway with quadrants
        fig = plot_pathway_with_quadrants(x, y, quadrants_df, time_vector)
        plt.savefig(
            os.path.join(output_dir, f"{base_name}_cube2d_result.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Save metrics to CSV file (database-friendly format)
        results_csv = os.path.join(output_dir, f"{base_name}_cube2d_results.csv")
        results_data = {
            "file_name": [base_name],
            "analysis_date": [datetime.now().strftime("%Y-%m-%d")],
            "analysis_time": [datetime.now().strftime("%H:%M:%S")],
            "sampling_frequency_hz": [fs],
            "total_distance_m": [round(total_distance, 3)],
            "average_speed_ms": [round(avg_speed, 3)],
            "maximum_speed_ms": [round(max_speed, 3)],
            "time_stationary_s": [round(time_stationary, 3)],
            "total_time_s": [round(total_time, 3)],
            "movement_percentage": [round(movement_percentage, 1)],
            "data_points": [len(x)],
        }

        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_csv, index=False)

        # Also save detailed text file for human reading
        with open(os.path.join(output_dir, f"{base_name}_cube2d_summary.txt"), "w") as f:
            f.write("CUBE 2D KINEMATICS ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n")
            f.write(f"File: {base_name}\n")
            f.write(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Sampling frequency: {fs} Hz\n")
            f.write(f"Data points: {len(x)}\n\n")
            f.write("KINEMATIC METRICS:\n")
            f.write(f"Total distance: {total_distance:.3f} m\n")
            f.write(f"Average speed: {avg_speed:.3f} m/s\n")
            f.write(f"Maximum speed: {max_speed:.3f} m/s\n")
            f.write(f"Time stationary: {time_stationary:.3f} s\n")
            f.write(f"Total time: {total_time:.3f} s\n")
            f.write(f"Movement percentage: {movement_percentage:.1f}%\n")

        return True

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return False


def run_cube2d_kinematics():
    """Main function to run the CUBE 2D kinematics analysis."""

    # Show instructions first
    if not show_instructions():
        print("Analysis cancelled by user.")
        return

    # Initialize Tkinter interface
    root = Tk()
    root.withdraw()  # Hide main window

    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting CUBE 2D Kinematics analysis...")

    # Select data directory
    data_dir = filedialog.askdirectory(title="Select the Data Directory (containing CSV files)")
    if not data_dir:
        print("No data directory selected. Exiting.")
        return

    # Count CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in the selected directory!")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    # Select quadrants file (optional)
    quadrants_file = filedialog.askopenfilename(
        title="Select the Quadrants File (optional - cancel for default)",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )

    # Select output directory
    output_dir = filedialog.askdirectory(title="Select the Output Directory")
    if not output_dir:
        print("No output directory selected. Exiting.")
        return

    # Get sampling frequency
    fs = None
    while fs is None:
        try:
            fs_value = simpledialog.askstring(
                "Sampling Frequency",
                f"Enter the sampling frequency (fs) in Hz for all {len(csv_files)} files:\n\n"
                "Common values: 30, 60, 100, 120, 250 Hz",
            )
            if fs_value is None:
                print("No sampling frequency entered. Exiting.")
                return
            fs = float(fs_value)
            if fs <= 0:
                raise ValueError("Sampling frequency must be positive")
        except ValueError as e:
            messagebox.showerror(
                "Invalid Input", f"Error: {e}\nPlease enter a valid positive number."
            )

    # Load quadrants
    quadrants_df = load_quadrants(quadrants_file)
    print(f"Using {'custom' if quadrants_file else 'default'} quadrants configuration.")

    # Create base output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(output_dir, f"vaila_cube2d_{timestamp}")
    os.makedirs(base_output_dir, exist_ok=True)

    # Process all CSV files
    files_processed = 0
    files_failed = 0
    all_results = []  # Store all results for consolidated database

    for file_name in csv_files:
        file_path = os.path.join(data_dir, file_name)
        base_name = os.path.splitext(file_name)[0]

        # Create subdirectory for each file
        file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_output_dir = os.path.join(base_output_dir, f"{base_name}_cube2d_{file_timestamp}")
        os.makedirs(file_output_dir, exist_ok=True)

        # Process file
        print(f"Processing: {file_name}")
        if process_file(file_path, quadrants_df, file_output_dir, fs, base_name):
            files_processed += 1

            # Collect data for consolidated database
            try:
                x, y = load_data(file_path)
                x = butter_lowpass_filter(x, 6, fs)
                y = butter_lowpass_filter(y, 6, fs)

                distance = calculate_distance(x, y)
                speed = calculate_speed(distance, fs)
                total_distance = np.sum(distance)
                avg_speed = np.mean(speed)
                max_speed = np.max(speed)
                time_stationary = np.sum(speed < 0.05) / fs
                total_time = len(x) / fs
                movement_percentage = (total_time - time_stationary) / total_time * 100

                all_results.append(
                    {
                        "file_name": base_name,
                        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
                        "analysis_time": datetime.now().strftime("%H:%M:%S"),
                        "sampling_frequency_hz": fs,
                        "total_distance_m": round(total_distance, 3),
                        "average_speed_ms": round(avg_speed, 3),
                        "maximum_speed_ms": round(max_speed, 3),
                        "time_stationary_s": round(time_stationary, 3),
                        "total_time_s": round(total_time, 3),
                        "movement_percentage": round(movement_percentage, 1),
                        "data_points": len(x),
                    }
                )
            except Exception as e:
                print(f"Error collecting data for consolidated database: {e}")
        else:
            files_failed += 1

    # Create consolidated database CSV
    if all_results:
        consolidated_df = pd.DataFrame(all_results)
        consolidated_csv = os.path.join(base_output_dir, "consolidated_cube2d_database.csv")
        consolidated_df.to_csv(consolidated_csv, index=False)
        print(f"Consolidated database saved: {consolidated_csv}")

    # Show completion message
    message = "Analysis completed!\n\n"
    message += f"Files processed successfully: {files_processed}\n"
    if files_failed > 0:
        message += f"Files failed: {files_failed}\n"
    message += f"\nOutput directory:\n{base_output_dir}\n\n"
    message += "Database file: consolidated_cube2d_database.csv"

    messagebox.showinfo("Processing Complete", message)
    print(f"Analysis complete. Results saved to: {base_output_dir}")


if __name__ == "__main__":
    run_cube2d_kinematics()
