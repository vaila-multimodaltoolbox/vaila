"""
===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 Oct 2024
Update Date: 31 March 2025
Version: 0.0.2
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


def process_jump_data(input_file, output_dir, use_time_of_flight):
    """
    Process the jump data from an input file and save results to the output directory.

    Args:
        input_file (str): The path to the input CSV file.
        output_dir (str): The path to the output directory.
        use_time_of_flight (bool): Whether to use time of flight data.
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
                height = calculate_jump_height(time_of_flight)
            else:
                # Use height directly
                height = second_value
                time_of_flight = None

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

    # Ask if the data is based on time of flight or height
    use_time_of_flight = messagebox.askyesno(
        "Data Type",
        "Are the files based on time of flight data? (Select 'No' if based on jump height)",
    )

    # Perform the analysis for all .csv files in the selected directory
    process_all_files_in_directory(target_dir, use_time_of_flight)

    root.destroy()
    messagebox.showinfo(
        "Success", "All .csv files have been processed and results saved."
    )


if __name__ == "__main__":
    vaila_and_jump()
