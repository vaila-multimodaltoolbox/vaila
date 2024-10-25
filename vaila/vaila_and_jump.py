"""
===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 24 Oct 2024
Version: 1.0.3
Python Version: 3.11.9

Description:
------------
This script processes jump data from multiple .csv files in a specified directory, 
performing biomechanical calculations based on either the time of flight or the 
jump height. The results are saved in a new output directory with a timestamp 
for each processed file.

Features:
---------
- Supports two calculation modes: 
  - Based on time of flight (calculates jump height)
  - Based on measured jump height (uses the height directly)
- Calculates various metrics for each jump:
  - Required Force
  - Exit Velocity
  - Potential Energy
  - Kinetic Energy
  - Average Power (if contact time is provided)
  - Total time (if time of flight and contact time are available)
- Processes all .csv files in the specified directory (ignores subdirectories).
- Saves the results in a new output directory named "vaila_verticaljump_<timestamp>".
- Formats numerical values to a fixed number of decimal places for better readability.
- Adds units of measurement in the header for each result.

Dependencies:
-------------
- Python 3.x
- pandas
- math
- tkinter

Usage:
------
- Run the script, select the target directory containing .csv files, and specify 
  whether the data is based on time of flight or jump height.
- The script will process each .csv file, performing calculations and saving 
  the results in a new directory.

Example:
--------
$ python vaila_and_jump.py

Notes:
------
- The .csv files are expected to have the following columns in order:
  - Column 0: mass
  - Column 1: time of flight or jump height, depending on the selected mode
  - Column 2 (optional): contact time
- Ensure that all necessary libraries are installed.
===============================================================================
"""

import os
import math
import pandas as pd
from tkinter import Tk, filedialog, messagebox, simpledialog
from datetime import datetime
from pathlib import Path
from rich import print

def calculate_required_force(mass, velocity, contact_time, gravity=9.81):
    """Calculate the force required to achieve the given exit velocity."""
    if pd.notna(contact_time) and contact_time > 0:
        acceleration = velocity / contact_time
        return mass * (gravity + acceleration)
    else:
        return mass * gravity  # If contact time is not available, fallback to body weight

def calculate_exit_velocity(height, time_of_flight, gravity=9.81):
    """Calculate the exit velocity based on height or time of flight."""
    if pd.notna(height):
        # Use height to calculate exit velocity
        return math.sqrt(2 * gravity * height)
    elif pd.notna(time_of_flight):
        # Use time of flight to calculate exit velocity
        return (gravity * time_of_flight) / 2
    else:
        return None

def calculate_potential_energy(mass, height, gravity=9.81):
    return mass * gravity * height

def calculate_kinetic_energy(mass, velocity):
    return 0.5 * mass * velocity ** 2

def calculate_average_power(potential_energy, contact_time):
    return potential_energy / contact_time

def format_number(num, decimals=3):
    """Format a number to a fixed number of decimal places."""
    if pd.isna(num):
        return ""
    return f"{num:.{decimals}f}"

def process_jump_data(input_file, output_dir, use_time_of_flight):
    """
    Process the jump data from an input file and save results to the output directory.
    """
    try:
        # Read data from the input file (assumed to be CSV format)
        data = pd.read_csv(input_file)
        results = []

        for index, row in data.iterrows():
            # Assuming the columns are in the following order:
            # Column 0: mass, Column 1: time_of_flight or height, Column 2: contact_time (optional)
            mass = row.iloc[0]
            second_value = row.iloc[1]
            contact_time = row.iloc[2] if len(row) > 2 else None

            # Determine height based on whether we are using time_of_flight or height directly
            if use_time_of_flight:
                # Calculate height based on time_of_flight
                time_of_flight = second_value
                height = (9.81 * time_of_flight ** 2) / 8
            else:
                # Use height directly
                height = second_value
                time_of_flight = None

            # Calculate the exit velocity
            velocity = calculate_exit_velocity(height, time_of_flight)

            # Calculate the required force
            force = calculate_required_force(mass, velocity, contact_time)

            # Perform energy calculations
            potential_energy = calculate_potential_energy(mass, height)
            kinetic_energy = calculate_kinetic_energy(mass, velocity)
            total_time = time_of_flight + contact_time if time_of_flight is not None and pd.notna(contact_time) else None
            average_power = calculate_average_power(potential_energy, contact_time) if pd.notna(contact_time) else None

            # Append the results for each row
            results.append({
                'height_m': format_number(height),
                'required_force_N': format_number(force),
                'exit_velocity_m/s': format_number(velocity),
                'potential_energy_J': format_number(potential_energy),
                'kinetic_energy_J': format_number(kinetic_energy),
                'average_power_W': format_number(average_power),
                'total_time_s': format_number(total_time)
            })

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Generate output file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file_path = os.path.join(output_dir, f"{base_name}_vjump_{timestamp}.csv")
        results_df.to_csv(output_file_path, index=False)

        print(f"Results saved successfully at: {output_file_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {str(e)}")

def process_all_files_in_directory(target_dir, use_time_of_flight):
    """
    Process all .csv files in the specified directory and save the results in a new output directory.
    """
    # Generate the output directory with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_verticaljump_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # List all .csv files in the target directory (ignore subdirectories)
    csv_files = [f for f in os.listdir(target_dir) if f.endswith(".csv")]

    # Process each .csv file found
    for file in csv_files:
        input_file = os.path.join(target_dir, file)
        print(f"Processing file: {input_file}")
        process_jump_data(input_file, output_dir, use_time_of_flight)

    print("All files have been processed successfully.")

def vaila_and_jump():
    """
    Main function to handle user input and execute the jump analysis for all .csv files in a directory.
    """
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    
    root = Tk()
    root.withdraw()

    # Ask user for the target directory
    target_dir = filedialog.askdirectory(title="Select the target directory containing .csv files")
    if not target_dir:
        messagebox.showwarning("Warning", "No target directory selected.")
        return

    # Ask if the data is based on time of flight or height
    use_time_of_flight = messagebox.askyesno(
        "Data Type", 
        "Are the files based on time of flight data? (Select 'No' if based on jump height)"
    )

    # Perform the analysis for all .csv files in the selected directory
    process_all_files_in_directory(target_dir, use_time_of_flight)

    root.destroy()
    messagebox.showinfo("Success", "All .csv files have been processed and results saved.")

if __name__ == "__main__":
    vaila_and_jump()
