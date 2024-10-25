"""
===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 24 Oct 2024
Version: 1.0.0
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
  - Force
  - Velocity
  - Potential Energy
  - Kinetic Energy
  - Average Power (if contact time is provided)
  - Total time (if time of flight and contact time are available)
- Processes all .csv files in the selected directory.
- Saves the results in a new output directory named "vaila_verticaljump_<timestamp>".

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

def calculate_force(mass, acceleration):
    """
    Calculate the force exerted by the jumper.
    
    The force is calculated based on mass and acceleration. This method
    calculates the force needed to achieve the acceleration.
    
    Parameters:
    - mass: Mass of the jumper (kg).
    - acceleration: Acceleration during takeoff (m/s^2).
    
    Returns:
    - The force in Newtons (N).
    """
    return mass * acceleration

def calculate_jump_height(time_of_flight, gravity=9.81):
    """
    Calculate the jump height based on the time of flight.
    
    The height is calculated using the equation: h = g * t^2 / 8.
    
    Parameters:
    - time_of_flight: The duration of the flight phase of the jump (s).
    - gravity: The gravitational acceleration (default is 9.81 m/s^2).
    
    Returns:
    - The jump height in meters (m).
    """
    return (gravity * time_of_flight ** 2) / 8

def calculate_power(force, height, contact_time):
    """
    Calculate the power generated during the jump.
    
    The power is calculated as the work done (force x height) divided by the 
    contact time.
    
    Parameters:
    - force: The force exerted by the jumper (N).
    - height: The height of the jump (m).
    - contact_time: The duration of the contact phase (s).
    
    Returns:
    - The power in Watts (W).
    """
    work = force * height
    return work / contact_time

def calculate_velocity(height, gravity=9.81):
    """
    Calculate the takeoff velocity needed to reach a given height.
    
    The velocity is calculated using the equation: v = sqrt(2 * g * h).
    
    Parameters:
    - height: The height of the jump (m).
    - gravity: The gravitational acceleration (default is 9.81 m/s^2).
    
    Returns:
    - The velocity in meters per second (m/s).
    """
    return math.sqrt(2 * gravity * height)

def calculate_kinetic_energy(mass, velocity):
    """
    Calculate the kinetic energy at takeoff.
    
    The kinetic energy is calculated using the equation: KE = 0.5 * m * v^2.
    
    Parameters:
    - mass: Mass of the jumper (kg).
    - velocity: The takeoff velocity (m/s).
    
    Returns:
    - The kinetic energy in Joules (J).
    """
    return 0.5 * mass * velocity ** 2

def calculate_potential_energy(mass, height, gravity=9.81):
    """
    Calculate the potential energy at the peak of the jump.
    
    The potential energy is calculated using the equation: PE = m * g * h.
    
    Parameters:
    - mass: Mass of the jumper (kg).
    - height: The height of the jump (m).
    - gravity: The gravitational acceleration (default is 9.81 m/s^2).
    
    Returns:
    - The potential energy in Joules (J).
    """
    return mass * gravity * height

def calculate_average_power(potential_energy, contact_time):
    """
    Calculate the average power output during the jump.
    
    The average power is calculated as the potential energy divided by the
    contact time.
    
    Parameters:
    - potential_energy: The potential energy at the peak of the jump (J).
    - contact_time: The duration of the contact phase (s).
    
    Returns:
    - The average power in Watts (W).
    """
    return potential_energy / contact_time

def calculate_relative_power(average_power, mass):
    """
    Calculate the relative power output during the jump.
    
    The relative power is calculated as the average power divided by the
    mass of the jumper.
    
    Parameters:
    - average_power: The average power output during the jump (W).
    - mass: Mass of the jumper (kg).
    
    Returns:
    - The relative power in Watts per kilogram (W/kg).
    """
    return average_power / mass

def calculate_jump_performance_index(height, force, contact_time):
    """
    Calculate the Jump Performance Index (JPI).
    
    The JPI is calculated using the formula: JPI = (height * force) / contact_time.
    
    Parameters:
    - height: The height of the jump (m).
    - force: The force exerted during the jump (N).
    - contact_time: The contact time (s).
    
    Returns:
    - The JPI as a unitless number.
    """
    return (height * force) / contact_time

def process_jump_data(input_file, output_dir, use_time_of_flight, gravity=9.81, decimals=3):
    """
    Process the jump data from an input file and save results to the output directory.
    
    This function reads the data from the input file, performs biomechanical calculations
    for each jump based on either time of flight or jump height, and saves the results to 
    a specified output directory.
    
    Parameters:
    - input_file: The path to the input .csv file.
    - output_dir: The directory where the results will be saved.
    - use_time_of_flight: Boolean indicating whether to use time of flight for calculations.
    - gravity: The gravitational acceleration (default is 9.81 m/s^2).
    - decimals: The number of decimal places to round the results.
    """
    try:
        # Read data from the input file (assumed to be CSV format)
        data = pd.read_csv(input_file)
        results = []

        for index, row in data.iterrows():
            # Assuming the columns are in the following order:
            # Column 0: mass, Column 1: time_of_flight or height, Column 2: contact_time (optional)
            mass = row[0]
            second_value = row[1]
            contact_time = row[2] if len(row) > 2 else None

            # Determine height based on whether we are using time_of_flight or height directly
            if use_time_of_flight:
                # Calculate height based on time_of_flight
                time_of_flight = second_value
                height = calculate_jump_height(time_of_flight, gravity)
            else:
                # Use height directly
                height = second_value
                time_of_flight = None

            # Calculate takeoff velocity
            velocity = calculate_velocity(height, gravity)

            # Calculate force based on the velocity and contact time (if provided)
            if pd.notna(contact_time) and contact_time > 0:
                acceleration = velocity / contact_time  # Assuming linear acceleration
                force = calculate_force(mass, acceleration)
            else:
                force = None

            # Calculate potential energy at the peak of the jump
            potential_energy = calculate_potential_energy(mass, height, gravity)

            # Calculate kinetic energy at takeoff
            kinetic_energy = calculate_kinetic_energy(mass, velocity)

            # Calculate average power if contact time is provided
            average_power = calculate_average_power(potential_energy, contact_time) if pd.notna(contact_time) else None

            # Calculate relative power (average power normalized by body mass)
            relative_power = calculate_relative_power(average_power, mass) if average_power is not None else None

            # Calculate Jump Performance Index (JPI)
            jpi = calculate_jump_performance_index(height, force, contact_time) if pd.notna(contact_time) and force is not None else None

            # Round the results to the specified number of decimal places
            def format_number(value):
                return round(value, decimals) if value is not None else None

            # Append the results for each row
            results.append({
                'height_m': format_number(height),
                'force_N': format_number(force),
                'velocity_m/s': format_number(velocity),
                'potential_energy_J': format_number(potential_energy),
                'kinetic_energy_J': format_number(kinetic_energy),
                'average_power_W': format_number(average_power),
                'relative_power_W/kg': format_number(relative_power),
                'jump_performance_index': format_number(jpi),
                'total_time_s': format_number(time_of_flight + contact_time) if time_of_flight is not None and pd.notna(contact_time) else None
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

def get_csv_files_in_directory(target_dir):
    """
    Get a list of all CSV files in the specified directory.
    
    This function scans the target directory for files with the .csv extension and 
    returns a list of their paths.
    
    Parameters:
    - target_dir: The directory to search for .csv files.
    
    Returns:
    - A list of file paths for all .csv files found in the directory.
    """
    csv_files = []
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def process_all_files_in_directory(target_dir, use_time_of_flight, gravity=9.81, decimals=3):
    """
    Process all .csv files in the specified directory and save the results in a new output directory.
    
    This function collects all .csv files in the specified target directory, processes each file 
    using biomechanical calculations, and saves the results to a new output directory with a 
    timestamp. The results for each file are saved individually.
    
    Parameters:
    - target_dir: The directory containing .csv files to process.
    - use_time_of_flight: Boolean indicating whether to use time of flight for calculations.
    - gravity: The gravitational acceleration (default is 9.81 m/s^2).
    - decimals: The number of decimal places to round the results.
    """
    # Generate the output directory with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_verticaljump_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Get all .csv files in the directory
    csv_files = get_csv_files_in_directory(target_dir)

    # Process each file
    for input_file in csv_files:
        print(f"Processing file: {input_file}")
        process_jump_data(input_file, output_dir, use_time_of_flight, gravity, decimals)

    print("All files have been processed successfully.")

def vaila_and_jump():
    """
    Main function to handle user input and execute the jump analysis for all .csv files in a directory.
    
    This function interacts with the user to get the target directory containing .csv files, 
    prompts the user to choose whether the data is based on time of flight or jump height, 
    and then processes all files using biomechanical calculations.
    """
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
