"""
===============================================================================
animal_open_field.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 09 December 2024
Version: 1.0.0
Python Version: 3.11.11

Description:
------------
This script processes movement data of mice in an open field test,
calculating kinematic variables such as total distance traveled, average speed,
time spent in each zone, and others.

In this initial version, only the code structure is presented. The calculation
functions are defined, but they currently do not implement any logic. In future
versions, these functions will be filled with the necessary equations to perform
the desired analyses.

Planned Future Features:
------------------------
- Read movement data from .csv files (X and Y positions over time).
- Calculate the total distance traveled by the animal.
- Calculate the average and/or instantaneous speed of the animal.
- Calculate the time spent in different zones of the open field.
- Save results in an output directory with a timestamp.

Dependencies:
-------------
- Python 3.x
- pandas
- math
- tkinter

Usage:
------
- Run the script, select the directory containing .csv files with data.
- The .csv files are expected to contain the animal's position over time.
- Results will be saved in a new directory.

Future Example:
---------------
$ python animal_open_field.py

Notes:
------
- The .csv files are expected to have columns, for example:
  - time(s), position_x(m), position_y(m)
- The exact calculation logic will be added in future versions.
===============================================================================
"""

import os
import math
import pandas as pd
from tkinter import Tk, filedialog, messagebox
from datetime import datetime
from pathlib import Path
from rich import print


def calculate_total_distance_traveled(position_data):
    """
    Calculate the total distance traveled by the animal based on X and Y coordinates over time.

    Args:
        position_data (pd.DataFrame): DataFrame with columns for X position, Y position, and time.

    Returns:
        float: Total distance traveled (in meters).
    """
    # Future logic: Calculate the sum of distances between consecutive points.
    return None


def calculate_speed(position_data):
    """
    Calculate the animal's speed(s) (instantaneous, average, etc.) based on its position.

    Args:
        position_data (pd.DataFrame): DataFrame with columns for X position, Y position, and time.

    Returns:
        float or dict: Average speed (and potentially instantaneous speeds).
    """
    # Future logic: Calculate instantaneous speeds = delta_distance / delta_time, then average.
    return None


def calculate_time_in_each_zone(position_data, zones_definition):
    """
    Calculate the total time spent in each zone of the open field.

    Args:
        position_data (pd.DataFrame): DataFrame with positions and time.
        zones_definition (dict): Definition of zones, for example, specific regions of the field.

    Returns:
        dict: Total time in each zone (keys: zone names, values: time in seconds).
    """
    # Future logic: Check position at each time point, categorize into zones, sum up time.
    return None


def process_open_field_data(input_file, output_dir):
    """
    Process data from an open field test file and save results to the output directory.

    Args:
        input_file (str): Path to the input CSV file.
        output_dir (str): Path to the output directory.
    """
    try:
        # Data reading (expected columns: time, position_x, position_y)
        data = pd.read_csv(input_file)

        # Calculations (future)
        total_distance = calculate_total_distance_traveled(data)
        avg_speed = calculate_speed(data)

        # Example zones - just a placeholder
        zones_definition = {
            "center": [
                (0, 0),
                (1, 1),
            ],  # Example: define the central region of the field
            "periphery": [(1, 1), (2, 2)],  # Example: define the periphery
        }
        time_zones = calculate_time_in_each_zone(data, zones_definition)

        # Assemble results into a dictionary
        results = {
            "total_distance_m": total_distance,
            "average_speed_m/s": avg_speed,
        }
        # Include time in zones, if available
        if time_zones:
            for zone_name, t in time_zones.items():
                results[f"time_in_{zone_name}_s"] = t

        # Convert results to DataFrame
        results_df = pd.DataFrame([results])

        # Generate output file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file_path = os.path.join(
            output_dir, f"{base_name}_openfield_{timestamp}.csv"
        )
        results_df.to_csv(output_file_path, index=False)

        print(f"Results saved successfully at: {output_file_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {str(e)}")


def process_all_files_in_directory(target_dir):
    """
    Process all .csv files in the specified directory and save the results.

    Args:
        target_dir (str): Path to the target directory containing .csv files.
    """
    # Generate the output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"animal_open_field_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # List all .csv files in the target directory
    csv_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".csv")
    ]

    # Process each .csv file
    for input_file in csv_files:
        print(f"Processing file: {input_file}")
        process_open_field_data(input_file, output_dir)

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

    # Process all files in the directory
    process_all_files_in_directory(target_dir)

    root.destroy()
    messagebox.showinfo(
        "Success", "All .csv files have been processed and results saved."
    )


if __name__ == "__main__":
    run_animal_open_field()
