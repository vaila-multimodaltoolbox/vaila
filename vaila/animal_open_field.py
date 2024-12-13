"""
===============================================================================
heatmap_pathway_plot.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 13 December 2024
Version: 2.0.0
Python Version: 3.11.11

Description:
------------
This script processes movement data of animals in an open field test, performing 
kinematic analyses including total distance traveled, average speed, and time 
spent in specific zones.

Key Features:
-------------
- Reads movement data from .csv files (X and Y positions over time).
- Calculates the total distance traveled.
- Calculates the average speed of the animal.
- Analyzes the time spent in different zones of a 60x60 cm open field, divided into 
  3x3 grid cells of 20x20 cm each.
- Generates visualizations including heatmaps and pathways of the animal's movement.
- Saves results and figures in an organized directory structure.

Dependencies:
-------------
- Python 3.x
- numpy
- matplotlib
- seaborn
- scipy
- tkinter

Usage:
------
- Run the script, select the directory containing .csv files with movement data.
- The .csv files should contain columns:
  - time(s), position_x(m), position_y(m).
- Results, including figures and a text summary, will be saved in a timestamped 
  directory with subdirectories for each processed file.

Example:
--------
$ python heatmap_pathway_plot.py

Notes:
------
- Ensure input .csv files are correctly formatted with positions in meters.
===============================================================================
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from tkinter import Tk, filedialog, simpledialog, messagebox
from datetime import datetime
from pathlib import Path

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the data.

    Args:
        data (array-like): Input data to filter.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Order of the filter.

    Returns:
        array-like: Filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)

def generate_time_vector(num_samples, fs):
    """
    Generate a time vector based on the number of samples and sampling frequency.

    Args:
        num_samples (int): Number of samples in the dataset.
        fs (float): Sampling frequency in Hz.

    Returns:
        numpy.ndarray: Time vector.
    """
    return np.linspace(0, (num_samples - 1) / fs, num_samples)


def plot_heatmap(x, y, output_dir, base_name):
    """
    Plot a heatmap of the animal's movement in the open field using seaborn KDE.

    Args:
        x (array-like): X coordinates of the movement (in meters).
        y (array-like): Y coordinates of the movement (in meters).
        output_dir (str): Directory to save the plot.
        base_name (str): Base name of the input file.

    Returns:
        str: Path to the saved heatmap file.
    """
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted to be square
    sns.kdeplot(x=x, y=y, cmap="coolwarm", fill=True, levels=40, ax=ax)

    # Adjust plot limits and labels
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.set_title("Heatmap of Animal Movement (in m)")

    # Set aspect ratio to equal
    ax.set_aspect('equal')

    # Save the plot
    output_file_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
    print(f"Heatmap plot saved at: {output_file_path}")
    return output_file_path


def plot_pathway(x, y, output_dir, base_name):
    """
    Plot the pathway of the animal's movement in the open field.

    Args:
        x (array-like): X coordinates of the movement (in meters).
        y (array-like): Y coordinates of the movement (in meters).
        output_dir (str): Directory to save the plot.
        base_name (str): Base name of the input file.

    Returns:
        str: Path to the saved pathway plot file.
    """
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjusted to be square
    ax.plot(x, y, color="blue", label="Pathway", alpha=0.7, linewidth=2)

    # Highlight start and end points
    ax.scatter(x[0], y[0], color="green", label="Start", s=100, edgecolors="black")
    ax.scatter(x[-1], y[-1], color="red", label="End", s=100, edgecolors="black")

    # Adjust plot limits and labels
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")
    ax.set_title("Pathway of Animal Movement (in m)")
    ax.legend()

    # Set aspect ratio to equal
    ax.set_aspect('equal')

    # Save the plot
    output_file_path = os.path.join(output_dir, f"{base_name}_pathway.png")
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
    print(f"Pathway plot saved at: {output_file_path}")
    return output_file_path

def calculate_total_distance_traveled(x, y):
    """
    Calculate the total distance traveled by the animal based on X and Y coordinates over time.

    Args:
        x (array-like): X coordinates of the movement.
        y (array-like): Y coordinates of the movement.

    Returns:
        float: Total distance traveled (in meters).
    """
    return np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

def calculate_average_speed(total_distance, total_time):
    """
    Calculate the average speed of the animal.

    Args:
        total_distance (float): Total distance traveled by the animal (in meters).
        total_time (float): Total time of the movement (in seconds).

    Returns:
        float: Average speed (in meters per second).
    """
    if total_time > 0:
        return total_distance / total_time
    else:
        return 0.0

def calculate_time_in_zones(x, y, time_vector):
    """
    Calculate the time spent in each 20x20 cm zone of a 60x60 cm grid.

    Args:
        x (array-like): X coordinates of the movement (in meters).
        y (array-like): Y coordinates of the movement (in meters).
        time_vector (array-like): Time vector corresponding to the coordinates.

    Returns:
        dict: Time spent in each zone (keys are zone indices, values are times in seconds).
    """
    zones = {(i, j): 0 for i in range(3) for j in range(3)}  # 3x3 grid

    for i in range(1, len(x)):
        # Skip coordinates outside the grid (0 to 60 cm)
        if not (0 <= x[i] <= 0.6 and 0 <= y[i] <= 0.6):
            continue

        # Determine the current zone
        zone_x = min(int(x[i] // 0.2), 2)
        zone_y = min(int(y[i] // 0.2), 2)
        
        # Calculate time spent in the current zone
        dt = time_vector[i] - time_vector[i - 1]
        zones[(zone_x, zone_y)] += dt

    return zones

def save_results_to_txt(results, output_dir, base_name):
    """
    Save the results and file paths to a .txt file.

    Args:
        results (dict): Dictionary containing the results.
        output_dir (str): Directory to save the .txt file.
        base_name (str): Base name of the input file.
    """
    try:
        txt_file_path = os.path.join(output_dir, f"{base_name}_results.txt")

        # Ensure directory exists and is writable
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.access(output_dir, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {output_dir}")

        # Save results to file
        with open(txt_file_path, "w", encoding="utf-8") as f:
            f.write("Analysis Results\n")
            f.write("=" * 20 + "\n")
            for key, value in results.items():
                if isinstance(value, dict):  # For nested dictionaries (e.g., zones)
                    f.write(f"{key}:\n")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (float, int)):
                            f.write(f"  Zone {sub_key}: {sub_value:.2f} seconds\n")
                        else:
                            f.write(f"  Zone {sub_key}: {sub_value}\n")
                else:
                    if isinstance(value, (float, int)):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        print(f"Results saved to: {txt_file_path}")
    except PermissionError as e:
        print(f"Permission error: {e}")
    except Exception as e:
        print(f"Error saving results to {txt_file_path}: {e}")
        raise

def process_open_field_data(input_file, main_output_dir, fs):
    """
    Process data from an open field test file and save results to the output directory.

    Args:
        input_file (str): Path to the input CSV file.
        main_output_dir (str): Path to the main output directory.
        fs (float): Sampling frequency in Hz.
    """
    try:
        # Determine base name of the file
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        # Create a specific output directory for this file
        output_dir = os.path.join(main_output_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Load data using numpy and select columns 1 and 2 (x and y coordinates)
        data = np.loadtxt(input_file, delimiter=",", skiprows=1, usecols=(1, 2))
        x, y = data[:, 0], data[:, 1]

        # Apply Butterworth filter to coordinates
        x = butter_lowpass_filter(x, cutoff=6, fs=fs, order=4)
        y = butter_lowpass_filter(y, cutoff=6, fs=fs, order=4)

        # Generate time vector
        time_vector = generate_time_vector(len(x), fs)

        # Plot heatmap and pathway
        heatmap_path = plot_heatmap(x, y, output_dir, base_name)
        pathway_path = plot_pathway(x, y, output_dir, base_name)

        # Calculate total distance
        total_distance = calculate_total_distance_traveled(x, y)

        # Calculate average speed
        total_time = time_vector[-1] - time_vector[0]
        average_speed = calculate_average_speed(total_distance, total_time)

        # Calculate time spent in zones
        time_in_zones = calculate_time_in_zones(x, y, time_vector)

        # Save results to a .txt file
        results = {
            "total_distance_m": total_distance,
            "average_speed_m/s": average_speed,
            "time_in_zones_s": time_in_zones,
            "heatmap_path": heatmap_path,
            "pathway_path": pathway_path,
        }
        save_results_to_txt(results, output_dir, base_name)

        print(f"Processing of file {input_file} completed successfully.")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {str(e)}")
        raise

def process_all_files_in_directory(target_dir, fs):
    """
    Process all .csv files in the specified directory and save the results.

    Args:
        target_dir (str): Path to the directory containing CSV files.
        fs (float): Sampling frequency in Hz.
    """
    # Generate the main output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(target_dir, f"vaila_openfield_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # List all .csv files in the target directory
    csv_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".csv")
    ]

    # Process each .csv file
    for input_file in csv_files:
        print(f"Processing file: {input_file}")
        process_open_field_data(input_file, main_output_dir, fs)

    print("All files have been processed successfully.")

def run_animal_open_field():
    """
    Main function to handle user input and execute the open field analysis
    on all .csv files in a directory.
    """
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

    root = Tk()
    root.withdraw()

    # Ask user for the target directory
    target_dir = filedialog.askdirectory(
        title="Select the directory containing .csv files of open field data"
    )
    if not target_dir:
        messagebox.showwarning("Warning", "No directory selected.")
        return

    # Ask user for the frame rate
    fs = simpledialog.askfloat(
        "Frame Rate",
        "Enter the frame rate of the data (fps in Hz):",
        minvalue=0.1,
        maxvalue=1000,
    )
    if not fs:
        messagebox.showwarning("Warning", "No frame rate provided.")
        return

    # Process all files in the directory
    process_all_files_in_directory(target_dir, fs)

    root.destroy()
    messagebox.showinfo(
        "Success", "All .csv files have been processed and results saved."
    )

if __name__ == "__main__":
    run_animal_open_field()
