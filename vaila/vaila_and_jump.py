"""
===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 Oct 2024
Update Date: 01 May 2025
Version: 0.0.3
Python Version: 3.12.9

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
  - Liftoff Force (Thrust)
  - Velocity (takeoff)
  - Potential Energy
  - Kinetic Energy
  - Average Power (if contact time is provided)
  - Relative Power (W/kg)
  - Jump Performance Index (JPI)
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

Input File Format:
-----------------
The input CSV files should have the following format:

1. Time-of-flight based format:
   mass(kg), time_of_flight(s), contact_time(s)[optional]
   
   Example:
   ```
   mass_kg,time_of_flight_s,contact_time_s
   75.0,0.45,0.22
   80.2,0.42,0.25
   65.5,0.48,0.20
   ```

2. Jump-height based format:
   mass(kg), height(m), contact_time(s)[optional]
   
   Example:
   ```
   mass_kg,heigth_m,contact_time_s
   75.0,0.25,0.22
   80.2,0.22,0.25
   65.5,0.28,0.20
   ```

Output:
-------
The script generates a CSV file with the following columns:
- height_m: Jump height in meters
- liftoff_force_N: Liftoff force in Newtons (if contact time is provided)
- velocity_m/s: Takeoff velocity in meters per second
- potential_energy_J: Potential energy in Joules
- kinetic_energy_J: Kinetic energy in Joules
- average_power_W: Average power in Watts (if contact time is provided)
- relative_power_W/kg: Power relative to body mass (if contact time is provided)
- jump_performance_index: Jump Performance Index (if both time of flight and contact time are available)
- total_time_s: Total time in seconds (if both time of flight and contact time are available)

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


def calculate_force(mass, gravity=9.81):
    """
    Calculate the weight force based on the mass and gravity.

    Args:
        mass (float): The mass of the jumper in kilograms.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The weight force in Newtons.
    """
    return mass * gravity


def calculate_jump_height(time_of_flight, gravity=9.81):
    """
    Calculate the jump height from the time of flight.

    Args:
        time_of_flight (float): The time of flight in seconds.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The jump height in meters.
    """
    return (gravity * time_of_flight**2) / 8


def calculate_power(force, height, contact_time):
    """
    Calculate the power output during the jump.

    Args:
        force (float): The force exerted during the jump in Newtons.
        height (float): The jump height in meters.
        contact_time (float): The contact time in seconds.

    Returns:
        float: The power output in Watts.
    """
    work = force * height
    return work / contact_time


def calculate_velocity(height, gravity=9.81):
    """
    Calculate the takeoff velocity needed to reach the given height.

    Args:
        height (float): The jump height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The takeoff velocity in meters per second.
    """
    return math.sqrt(2 * gravity * height)


def calculate_kinetic_energy(mass, velocity):
    """
    Calculate the kinetic energy based on mass and velocity.

    Args:
        mass (float): The mass of the jumper in kilograms.
        velocity (float): The velocity in meters per second.

    Returns:
        float: The kinetic energy in Joules.
    """
    return 0.5 * mass * velocity**2


def calculate_potential_energy(mass, height, gravity=9.81):
    """
    Calculate the potential energy based on mass and height.

    Args:
        mass (float): The mass of the jumper in kilograms.
        height (float): The height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The potential energy in Joules.
    """
    return mass * gravity * height


def calculate_average_power(potential_energy, contact_time):
    """
    Calculate the average power output during the contact phase.

    Args:
        potential_energy (float): The potential energy in Joules.
        contact_time (float): The contact time in seconds.

    Returns:
        float: The average power in Watts.
    """
    return potential_energy / contact_time


def calculate_liftoff_force(mass, velocity, contact_time, gravity=9.81):
    """
    Calculate the total liftoff force (thrust) required during the contact phase.

    Args:
        mass (float): The mass of the jumper in kilograms.
        velocity (float): The takeoff velocity in m/s.
        contact_time (float): The contact time in seconds.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The liftoff force in Newtons.
    """
    # Calculate the weight force
    weight_force = mass * gravity

    # Calculate the force needed to accelerate to the takeoff velocity
    acceleration_force = (mass * velocity) / contact_time

    # Total liftoff force
    liftoff_force = weight_force + acceleration_force

    return liftoff_force


def calculate_time_of_flight(height, gravity=9.81):
    """
    Calculate the time of flight from the jump height.
    
    Args:
        height (float): The jump height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.
        
    Returns:
        float: The time of flight in seconds.
    """
    return math.sqrt(8 * height / gravity)


def process_jump_data(input_file, output_dir, use_time_of_flight):
    """
    Process the jump data from an input file and save results to the output directory.
    """
    try:
        # Read data from the input file
        data = pd.read_csv(input_file)
        results = []
        
        # Auto-detect column names
        columns = list(data.columns)
        
        # Look for appropriate columns by name pattern
        mass_col = next((col for col in columns if 'mass' in col.lower()), columns[0])
        
        if use_time_of_flight:
            value_col = next((col for col in columns if 'time' in col.lower() and 'contact' not in col.lower()), columns[1])
        else:
            value_col = next((col for col in columns if 'height' in col.lower() or 'heigth' in col.lower()), columns[1])
            
        contact_col = next((col for col in columns if 'contact' in col.lower()), None)
        
        print(f"Using columns: Mass={mass_col}, {'Time' if use_time_of_flight else 'Height'}={value_col}, Contact={contact_col}")
        
        for index, row in data.iterrows():
            # Get values using detected column names
            mass = row[mass_col]
            second_value = row[value_col]
            contact_time = row[contact_col] if contact_col else None
            
            # Determine height based on whether we are using time_of_flight or height directly
            if use_time_of_flight:
                # Calculate height based on time_of_flight
                time_of_flight = second_value
                height = calculate_jump_height(time_of_flight)
            else:
                # Use height directly and calculate time_of_flight from it
                height = second_value
                time_of_flight = calculate_time_of_flight(height)

            # Perform calculations
            velocity = calculate_velocity(height)
            potential_energy = calculate_potential_energy(mass, height)
            kinetic_energy = calculate_kinetic_energy(mass, velocity)
            liftoff_force = (
                calculate_liftoff_force(mass, velocity, contact_time)
                if contact_time
                else None
            )
            average_power = (
                calculate_average_power(potential_energy, contact_time)
                if contact_time
                else None
            )
            relative_power = (average_power / mass) if average_power else None
            jump_performance_index = (
                (average_power * time_of_flight)
                if average_power and time_of_flight
                else None
            )
            total_time = (
                time_of_flight + contact_time
                if time_of_flight and contact_time
                else None
            )

            # Append the results for each row
            results.append(
                {
                    "height_m": round(height, 3),
                    "liftoff_force_N": (
                        round(liftoff_force, 3) if liftoff_force else None
                    ),
                    "velocity_m/s": round(velocity, 3),
                    "potential_energy_J": round(potential_energy, 3),
                    "kinetic_energy_J": round(kinetic_energy, 3),
                    "average_power_W": (
                        round(average_power, 3) if average_power else None
                    ),
                    "relative_power_W/kg": (
                        round(relative_power, 3) if relative_power else None
                    ),
                    "jump_performance_index": (
                        round(jump_performance_index, 3)
                        if jump_performance_index
                        else None
                    ),
                    "total_time_s": round(total_time, 3) if total_time else None,
                }
            )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Generate output file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file_path = os.path.join(
            output_dir, f"{base_name}_vjump_{timestamp}.csv"
        )
        results_df.to_csv(output_file_path, index=False)

        print(f"Results saved successfully at: {output_file_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {str(e)}")


def process_all_files_in_directory(target_dir, use_time_of_flight):
    """
    Process all .csv files in the specified directory and save the results in a new output directory.

    Args:
        target_dir (str): The path to the target directory containing .csv files.
        use_time_of_flight (bool): Whether to use time of flight data.
    """
    # Generate the output directory with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_verticaljump_{timestamp}")
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
    target_dir = filedialog.askdirectory(
        title="Select the target directory containing .csv files"
    )
    if not target_dir:
        messagebox.showwarning("Warning", "No target directory selected.")
        return

    # Use a dialog box with more descriptive options
    data_type_options = ["Time of Flight Data", "Jump Height Data"]
    data_type = simpledialog.askinteger(
        "Select Data Type",
        "Select the type of data in your CSV files:\n\n1. Time of Flight Data\n2. Jump Height Data",
        minvalue=1, maxvalue=2
    )
    
    if data_type is None:  # User cancelled
        messagebox.showwarning("Warning", "No data type selected. Exiting.")
        return
    
    use_time_of_flight = (data_type == 1)

    # Perform the analysis for all .csv files in the selected directory
    process_all_files_in_directory(target_dir, use_time_of_flight)

    root.destroy()
    
    # Show more detailed success message
    msg = "All CSV files have been processed and results saved.\n\n"
    msg += f"Input data type: {'Time of Flight' if use_time_of_flight else 'Jump Height'}\n"
    msg += f"Output directory: {os.path.join(target_dir, 'vaila_verticaljump_*')}"
    
    messagebox.showinfo("Success", msg)


if __name__ == "__main__":
    vaila_and_jump()
