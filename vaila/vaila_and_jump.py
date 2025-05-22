"""
===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 Oct 2024
Update Date: 21 May 2025
Version: 0.0.4
Python Version: 3.12.9

Description:
------------
This script processes jump data from multiple .csv files in a specified directory,
performing biomechanical calculations based on either the time of flight or the
jump height. The results are saved in a new output directory with a timestamp
for each processed file.

For MediaPipe data, the script automatically inverts y-coordinates (1.0 - y) to
transform from screen coordinates (where y increases downward) to biomechanical
coordinates (where y increases upward). This allows proper visualization and
analysis of the jumping motion.

Features:
---------
- Supports two calculation modes:
  - Based on time of flight (calculates jump height)
  - Based on measured jump height (uses the height directly)
  - Processes MediaPipe (in vailá) pose estimation data:
    - Automatically inverts y-coordinates for proper biomechanical analysis
    - Converts normalized coordinates to meters using shank length as reference
    - Calculates center of gravity (CG) position for accurate jump height
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
- Generates visualizations and HTML reports for data analysis.

Dependencies:
-------------
- Python 3.x
- pandas
- numpy
- matplotlib
- tkinter
- math
- datetime

Usage:
------
- Run the script, select the target directory containing .csv files, and specify
  whether the data is based on time of flight, jump height, or MediaPipe data.
- The script will process each .csv file, performing calculations and saving
  the results in a new directory.

Example:
--------
$ python vaila_and_jump.py

Input File Format:
-----------------
The input CSV files should have one of the following formats:

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
   mass_kg,height_m,contact_time_s
   75.0,0.25,0.22
   80.2,0.22,0.25
   65.5,0.28,0.20
   ```

3. MediaPipe pose estimation format:
   CSV file with MediaPipe pose landmark coordinates
   (frame_index, nose_x, nose_y, nose_z, left_eye_inner_x, etc.)

Coordinate System:
-----------------
For MediaPipe data, coordinates are transformed to use the following system:
- Origin: Bottom-left corner
- X-axis: Increases from left to right
- Y-axis: Increases from bottom to top (inverted from screen coordinates)
- Z-axis: Increases from back to front

Output:
-------
The script generates various output files:
- CSV files with jump metrics
- Full processed data with all original and calculated values
- Calibrated data in meters with proper coordinate orientation
- Visualizations of the jump performance
- HTML report summarizing all analysis

Notes:
------
- Ensure that all necessary libraries are installed.
- For accurate results, the MediaPipe landmark detection should be of good quality.

License:
--------
This script is licensed under the GNU General Public License v3.0.
===============================================================================
"""

import os
import math
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, messagebox, simpledialog
from datetime import datetime
from pathlib import Path
from rich import print
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


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


def calculate_baseline(data, n_frames=10):
    """
    Calculate the baseline using the first n frames.

    Args:
        data (pd.DataFrame): DataFrame with MediaPipe data
        n_frames (int): Number of frames to calculate the baseline average

    Returns:
        tuple: (baseline_feet, baseline_cg)
    """
    # Calculate the average of the first n frames for the feet
    right_foot_baseline = data["right_foot_index_y"].iloc[:n_frames].mean()
    left_foot_baseline = data["left_foot_index_y"].iloc[:n_frames].mean()
    feet_baseline = (right_foot_baseline + left_foot_baseline) / 2

    # Calculate the average of the first n frames for the CG
    cg_y_baseline = data["cg_y"].iloc[:n_frames].mean()

    return feet_baseline, cg_y_baseline


def identify_jump_phases(data, feet_baseline, cg_baseline, fps):
    """
    Identify the different phases of the jump.

    Args:
        data (pd.DataFrame): DataFrame with the data
        feet_baseline (float): Baseline for the feet
        cg_baseline (float): Baseline for the CG
        fps (int): Frames per second

    Returns:
        dict: Dictionary with information about the jump phases
    """
    # Convert frame indices to time
    time = data.index / fps

    # Find propulsion phase (when CG starts to rise)
    propulsion_start = data[data["cg_y"] > cg_baseline].index[0]

    # Find the start of the airborne phase (when the feet leave the ground)
    takeoff_idx = data[data["right_foot_index_y"] < feet_baseline].index[0]

    # Find the highest point (lowest value of CG_y since y increases downwards)
    max_height_idx = data["cg_y"].idxmin()

    # Find landing (when the feet return to the ground)
    landing_idx = data[data.index > max_height_idx][
        data[data.index > max_height_idx]["right_foot_index_y"] >= feet_baseline
    ].index[0]

    # Calculate airborne phase time
    flight_time = (landing_idx - takeoff_idx) / fps

    # Calculate maximum height relative to the baseline
    max_height = abs(data["cg_y"].min() - cg_baseline)

    return {
        "propulsion_start_frame": propulsion_start,
        "takeoff_frame": takeoff_idx,
        "max_height_frame": max_height_idx,
        "landing_frame": landing_idx,
        "flight_time_s": flight_time,
        "max_height_m": max_height,
        "propulsion_time_s": (takeoff_idx - propulsion_start) / fps,
        "ascent_time_s": (max_height_idx - takeoff_idx) / fps,
        "descent_time_s": (landing_idx - max_height_idx) / fps,
    }


def generate_jump_plots(data, results, output_dir, base_name):
    """
    Generate time series plots for CG and feet position data.

    Args:
        data (pd.DataFrame): DataFrame with processed data
        results (dict): Dictionary with jump metrics
        output_dir (str): Directory to save the plots
        base_name (str): Base name for the output files

    Returns:
        list: Paths to the generated plot files
    """
    plot_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Create time series plot of CG and feet positions
    plt.figure(figsize=(12, 8))

    # Add frame numbers on x-axis
    frames = (
        data.index.values if "frame_index" not in data.columns else data["frame_index"]
    )
    time_seconds = frames / results["fps"]

    # Plot NORMALIZED CG position
    plt.plot(
        time_seconds,
        data["cg_y_normalized"],
        "b-",
        linewidth=2,
        label="Center of Gravity (normalized)",
    )

    # Plot NORMALIZED feet position if available
    if "left_foot_index_y_m" in data.columns:
        normalized_left_foot = data["left_foot_index_y_m"] - data["reference_cg_y"]
        plt.plot(
            time_seconds,
            normalized_left_foot,
            "g-",
            linewidth=1.5,
            label="Left Foot (normalized)",
        )
    if "right_foot_index_y_m" in data.columns:
        normalized_right_foot = data["right_foot_index_y_m"] - data["reference_cg_y"]
        plt.plot(
            time_seconds,
            normalized_right_foot,
            "r-",
            linewidth=1.5,
            label="Right Foot (normalized)",
        )

    # Mark important frames
    takeoff_time = time_seconds[results["takeoff_frame"]]
    max_height_time = time_seconds[results["max_height_frame"]]
    landing_time = time_seconds[results["landing_frame"]]

    # Mark takeoff, max height, and landing points
    plt.axvline(x=takeoff_time, color="g", linestyle="--", label="Takeoff")
    plt.axvline(x=max_height_time, color="r", linestyle="--", label="Max Height")
    plt.axvline(x=landing_time, color="k", linestyle="--", label="Landing")

    # Add reference line (zero)
    plt.axhline(
        y=0, color="gray", linestyle="-", label="Initial CG Position (reference)"
    )

    # Add labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Position (meters from initial CG) - Up is positive")
    plt.title("Jump Analysis - Normalized CG and Feet Positions")
    plt.legend(loc="best")
    plt.grid(True)

    # Save plot
    ts_plot_path = os.path.join(output_dir, f"{base_name}_time_series_{timestamp}.png")
    plt.savefig(ts_plot_path, dpi=300, bbox_inches="tight")
    plot_files.append(ts_plot_path)
    plt.close()

    # 2. Create a jump phases visualization
    plt.figure(figsize=(12, 6))

    # Create flight phase rectangle
    flight_start = takeoff_time
    flight_end = landing_time
    flight_duration = flight_end - flight_start

    # Plot rectangles for each phase
    plt.axvspan(
        flight_start, flight_end, alpha=0.2, color="skyblue", label="Flight Phase"
    )
    plt.axvspan(
        flight_start, max_height_time, alpha=0.2, color="lightgreen", label="Ascent"
    )
    plt.axvspan(max_height_time, flight_end, alpha=0.2, color="salmon", label="Descent")

    # Plot NORMALIZED CG path
    plt.plot(
        time_seconds,
        data["cg_y_normalized"],
        "b-",
        linewidth=2.5,
        label="CG Path (normalized)",
    )

    # Mark max height using normalized value
    max_height_value = data["cg_y_normalized"].iloc[results["max_height_frame"]]
    plt.scatter(
        [max_height_time],
        [max_height_value],
        color="red",
        s=100,
        marker="o",
        label=f'Max Height: {results["height_m"]:.3f}m from initial CG',
    )

    # Add annotations
    plt.annotate(
        f"Flight Time: {results['flight_time_s']:.3f}s",
        xy=(flight_start + flight_duration / 2, max_height_value - 0.05),
        xytext=(flight_start + flight_duration / 2, max_height_value - 0.1),
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3),
    )

    plt.annotate(
        f"Velocity: {results['velocity_m/s']:.2f} m/s",
        xy=(flight_start, max_height_value - 0.15),
        xytext=(flight_start, max_height_value - 0.15),
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3),
    )

    # Add labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Position (meters from initial CG)")
    plt.title("Jump Phases Analysis - Normalized from Initial CG Position")
    plt.legend(loc="best")
    plt.grid(True)

    # Save plot
    phases_plot_path = os.path.join(
        output_dir, f"{base_name}_jump_phases_{timestamp}.png"
    )
    plt.savefig(phases_plot_path, dpi=300, bbox_inches="tight")
    plot_files.append(phases_plot_path)
    plt.close()

    return plot_files


def generate_html_report(data, results, plot_files, output_dir, base_name):
    """
    Generate an HTML report with jump metrics and plots.

    Args:
        data (pd.DataFrame): DataFrame with processed data
        results (dict): Dictionary with jump metrics
        plot_files (list): List of paths to plot files
        output_dir (str): Directory to save the report
        base_name (str): Base name for the output file

    Returns:
        str: Path to the generated HTML report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{base_name}_report_{timestamp}.html")

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Jump Analysis Report - {base_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 30px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .img-container {{
                text-align: center;
                margin: 30px 0;
            }}
            .img-container img {{
                max-width: 100%;
                height: auto;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .note {{
                background-color: #f8f9fa;
                border-left: 4px solid #4caf50;
                padding: 15px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 50px;
                border-top: 1px solid #ddd;
                padding-top: 20px;
                color: #7f8c8d;
                font-size: 0.9em;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>Jump Analysis Report</h1>
        <p><strong>Subject:</strong> {base_name}</p>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="note">
            <h3>Coordinate System</h3>
            <p>This analysis uses a biomechanical coordinate system where:</p>
            <ul>
                <li>Origin is at the bottom left</li>
                <li>X-axis: positive to the right</li>
                <li>Y-axis: positive upward</li>
                <li>Z-axis: positive forward</li>
            </ul>
            <p>MediaPipe coordinates were transformed to match this convention, and all measurements are in meters.</p>
            <p><strong>Important:</strong> Jump height is measured relative to the initial center of gravity (CG) position,
            which is calculated as the average CG position during the first 10 frames. This reference position is set as zero,
            so all vertical measurements represent displacement from this initial position.</p>
        </div>
        
        <h2>Jump Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
            <tr>
                <td>Mass</td>
                <td>{results["mass_kg"]}</td>
                <td>kg</td>
            </tr>
            <tr>
                <td>Jump Height</td>
                <td>{results["height_m"]}</td>
                <td>m (from initial CG position)</td>
            </tr>
            <tr>
                <td>Flight Time</td>
                <td>{results["flight_time_s"]}</td>
                <td>s</td>
            </tr>
            <tr>
                <td>Takeoff Velocity</td>
                <td>{results["velocity_m/s"]}</td>
                <td>m/s</td>
            </tr>
            <tr>
                <td>Potential Energy</td>
                <td>{results["potential_energy_J"]}</td>
                <td>J</td>
            </tr>
            <tr>
                <td>Kinetic Energy</td>
                <td>{results["kinetic_energy_J"]}</td>
                <td>J</td>
            </tr>
        </table>
        
        <h2>Jump Phase Frames</h2>
        <table>
            <tr>
                <th>Phase</th>
                <th>Frame</th>
                <th>Time (s)</th>
            </tr>
            <tr>
                <td>Takeoff</td>
                <td>{results["takeoff_frame"]}</td>
                <td>{results["takeoff_frame"] / results["fps"]:.3f}</td>
            </tr>
            <tr>
                <td>Maximum Height</td>
                <td>{results["max_height_frame"]}</td>
                <td>{results["max_height_frame"] / results["fps"]:.3f}</td>
            </tr>
            <tr>
                <td>Landing</td>
                <td>{results["landing_frame"]}</td>
                <td>{results["landing_frame"] / results["fps"]:.3f}</td>
            </tr>
        </table>
        
        <h2>Jump Analysis Visualizations</h2>
    """

    # Add images to the report
    for plot_file in plot_files:
        plot_filename = os.path.basename(plot_file)
        # Use relative paths in the HTML
        html_content += f"""
        <div class="img-container">
            <img src="{plot_filename}" alt="Jump analysis plot">
            <p><em>{plot_filename}</em></p>
        </div>
        """

    # Close the HTML content
    html_content += """
        <div class="footer">
            <p>Generated by vailá - Vertical Jump Analysis Tool</p>
        </div>
    </body>
    </html>
    """

    # Write the HTML file
    with open(report_path, "w") as f:
        f.write(html_content)

    return report_path


def process_mediapipe_data(input_file, output_dir):
    """
    Process MediaPipe data and generate visualizations and report.
    """
    try:
        data = pd.read_csv(input_file)

        # Invert all y coordinates (1.0 - y) to fix orientation
        for col in [c for c in data.columns if c.endswith("_y")]:
            data[col] = 1.0 - data[col]

        results = {}

        # Request mass and FPS
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # Force dialogs to be on top

        mass = simpledialog.askfloat(
            "Mass Input",
            "Enter the subject's mass (kg):",
            parent=root,  # Set parent window
            minvalue=20.0,
            maxvalue=200.0,
        )
        root.lift()  # Bring window to the front

        if mass is None:
            print(f"Processing cancelled for {input_file} - no mass provided.")
            return

        fps = simpledialog.askinteger(
            "FPS Input",
            "Enter the video FPS (frames per second):",
            parent=root,  # Set parent window
            minvalue=1,
            maxvalue=240,
        )
        root.lift()  # Bring window to the front

        if fps is None:
            fps = 30
            print(f"Using default FPS value: {fps}")

        # Define a scale factor for meters
        shank_length_real = simpledialog.askfloat(
            "Scale Factor",
            "Enter the approximate shank length in meters (e.g., 0.4):",
            parent=root,  # Set parent window
            minvalue=0.1,
            maxvalue=1.0,
        )
        root.lift()  # Bring window to the front
        if shank_length_real is None:
            shank_length_real = 0.4  # Default value

        # Calculate the conversion factor for normalized pixels to meters
        conversion_factor = calc_fator_convert_mediapipe(data, shank_length_real)
        print(f"Conversion factor: {conversion_factor:.6f} m/unit")

        # Processing to calculate the CG (already in meters)
        cg_x_m_list = []
        cg_y_m_list = []

        # Process in chunks to calculate the CG
        chunk_size = 50
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i : i + chunk_size]
            for _, row in chunk.iterrows():
                temp_df = pd.DataFrame([row])
                # Calculate CG and store the results
                cg_x, cg_y = calculate_cg_frame(temp_df, conversion_factor)
                # Fix the issue of Series vs scalar
                cg_x_m_list.append(
                    float(cg_x.iloc[0]) if hasattr(cg_x, "iloc") else float(cg_x)
                )
                cg_y_m_list.append(
                    float(cg_y.iloc[0]) if hasattr(cg_y, "iloc") else float(cg_y)
                )

        # Add the CG columns in meters to the DataFrame
        data["cg_x_m"] = cg_x_m_list
        data["cg_y_m"] = cg_y_m_list

        # Convert all coordinates to meters
        cols_to_convert = {}

        # Convert x coordinates
        for col in [c for c in data.columns if c.endswith("_x")]:
            base_name = col[:-2]
            cols_to_convert[f"{base_name}_x_m"] = data[col] * conversion_factor

        # Convert y coordinates
        for col in [c for c in data.columns if c.endswith("_y")]:
            base_name = col[:-2]
            cols_to_convert[f"{base_name}_y_m"] = data[col] * conversion_factor

        # Convert z coordinates
        for col in [c for c in data.columns if c.endswith("_z")]:
            base_name = col[:-2]
            cols_to_convert[f"{base_name}_z_m"] = data[col] * conversion_factor

        # Add all converted columns at once
        conv_df = pd.DataFrame(cols_to_convert)
        data = pd.concat([data, conv_df], axis=1)

        # Calculate the reference (averages of the first 10 frames)
        n_baseline_frames = 10

        # Calculate reference for the CG (to be used as zero)
        cg_y_ref = data["cg_y_m"].iloc[:n_baseline_frames].mean()
        cg_x_ref = data["cg_x_m"].iloc[:n_baseline_frames].mean()

        # Add this block to create relative versions of all y-coordinates
        print("Creating relative coordinates referenced to initial CG position...")
        # Colete todas as colunas em dicionários
        rel_cols = {}

        # Y coordinates relative to CG reference
        for col in [c for c in data.columns if c.endswith("_y_m")]:
            rel_cols[f"{col}_rel"] = data[col] - cg_y_ref
            print(f"  Created {col}_rel column")

        # X coordinates relative to CG reference
        for col in [c for c in data.columns if c.endswith("_x_m")]:
            rel_cols[f"{col}_rel"] = data[col] - cg_x_ref
            print(f"  Created {col}_rel column")

        # Adicione todas as colunas de uma vez
        data = pd.concat([data, pd.DataFrame(rel_cols)], axis=1)

        # Normalize CG so that the reference is zero
        data["cg_x_normalized"] = data["cg_x_m"] - cg_x_ref
        data["cg_y_normalized"] = data["cg_y_m"] - cg_y_ref

        # Calculate baseline for the feet
        has_left_foot = "left_foot_index_y_m" in data.columns
        has_right_foot = "right_foot_index_y_m" in data.columns

        if has_left_foot and has_right_foot:
            feet_y_values = (
                data["left_foot_index_y_m"] + data["right_foot_index_y_m"]
            ) / 2
        elif has_left_foot:
            feet_y_values = data["left_foot_index_y_m"]
        elif has_right_foot:
            feet_y_values = data["right_foot_index_y_m"]
        else:
            # Fallback to ankles
            if "left_ankle_y_m" in data.columns and "right_ankle_y_m" in data.columns:
                feet_y_values = (data["left_ankle_y_m"] + data["right_ankle_y_m"]) / 2
            elif "left_ankle_y_m" in data.columns:
                feet_y_values = data["left_ankle_y_m"]
            elif "right_ankle_y_m" in data.columns:
                feet_y_values = data["right_ankle_y_m"]
            else:
                feet_y_values = data["cg_y_m"] * 0.8  # Estimate

        feet_baseline = feet_y_values.iloc[:n_baseline_frames].mean()

        # Add basic information
        data["mass_kg"] = mass
        data["fps"] = fps
        data["reference_cg_y"] = cg_y_ref
        data["reference_cg_x"] = cg_x_ref
        data["reference_feet_y"] = feet_baseline

        # Identify the lowest point (squat) - before the jump
        squat_frame = data["cg_y_normalized"].idxmin()

        # Identify the highest point (jump) - now it will be the maximum
        max_height_frame = data["cg_y_normalized"].idxmax()

        # Calculate the jump height relative to the reference position
        jump_height = data["cg_y_normalized"].iloc[max_height_frame]

        # Identify takeoff: the first frame after the squat where the CG rises above zero
        takeoff_candidates = data.index[
            (data.index > squat_frame)
            & (data.index < max_height_frame)
            & (data["cg_y_normalized"] > 0)
        ]
        takeoff_frame = (
            takeoff_candidates.min() if len(takeoff_candidates) > 0 else squat_frame
        )

        # Identify landing: the first frame after the highest point where the CG returns to zero or below
        landing_candidates = data.index[
            (data.index > max_height_frame) & (data["cg_y_normalized"] < 0)
        ]
        landing_frame = (
            landing_candidates.min() if len(landing_candidates) > 0 else len(data) - 1
        )

        # Calculate flight time
        flight_time = (landing_frame - takeoff_frame) / fps

        # ======= Correct calculation of Kinetic Energy (takeoff) and Potential Energy (maximum height) =======
        # Time interval between frames
        dt = 1.0 / fps

        # Central derivative for vertical velocity at the takeoff frame (CG in meters)
        if takeoff_frame > 0 and takeoff_frame < len(data) - 1:
            vel_takeoff = (
                data["cg_y_m"].iloc[takeoff_frame + 1]
                - data["cg_y_m"].iloc[takeoff_frame - 1]
            ) / (2 * dt)
        else:
            vel_takeoff = (
                data["cg_y_m"].iloc[takeoff_frame]
                - data["cg_y_m"].iloc[takeoff_frame - 1]
            ) / dt

        # Kinetic energy at takeoff (in Joules)
        kinetic_energy = 0.5 * mass * vel_takeoff**2

        # Potential energy at the maximum height of the jump (in Joules)
        potential_energy = mass * 9.81 * jump_height

        # Calculate jump metrics
        velocity = calculate_velocity(jump_height)

        # Prepare results
        results = {
            "height_m": round(jump_height, 3),
            "mass_kg": mass,
            "fps": fps,
            "flight_time_s": round(flight_time, 3),
            "velocity_m/s": round(
                vel_takeoff, 3
            ),  # Actual velocity of the CG at takeoff
            "potential_energy_J": round(potential_energy, 3),
            "kinetic_energy_J": round(kinetic_energy, 3),
            "squat_frame": squat_frame,
            "max_height_frame": max_height_frame,
            "takeoff_frame": takeoff_frame,
            "landing_frame": landing_frame,
            "conversion_factor": round(conversion_factor, 6),
        }

        # Save results in DataFrame
        results_df = pd.DataFrame([results])

        # Generate base name for files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        # Não é necessário salvar o arquivo processed, pois é redundante
        # output_data_file = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.csv")
        # data.to_csv(output_data_file, index=False)

        # 1. Identify all original columns (keeping the order of the read file)
        orig_cols = list(pd.read_csv(input_file, nrows=1).columns)

        # 2. For each original column ending with _x, _y, _z, add its _x_m, _y_m, _z_m version
        orig_m_cols = []
        for col in orig_cols:
            if col.endswith(("_x", "_y", "_z")):
                orig_m_cols.append(
                    f"{col[:-2]}_x_m"
                    if col.endswith("_x")
                    else f"{col[:-2]}_y_m" if col.endswith("_y") else f"{col[:-2]}_z_m"
                )
            else:
                # If it's not a coordinate, keep it if it exists (e.g., frame_index)
                if col in data.columns:
                    orig_m_cols.append(col)

        # 3. New columns: relative to the initial CG (_rel) and normalized (cg_y_normalized etc)
        rel_cols = [c for c in data.columns if c.endswith("_rel")]
        norm_cols = [c for c in data.columns if "normalized" in c]
        metadata_cols = [
            "mass_kg",
            "fps",
            "reference_cg_y",
            "reference_cg_x",
            "reference_feet_y",
        ]

        # 4. Build the final list of columns, keeping all originals (now in meters), and only then the new ones
        final_cols = []
        for col in orig_cols:
            if col.endswith(("_x", "_y", "_z")):
                metr = (
                    f"{col[:-2]}_x_m"
                    if col.endswith("_x")
                    else f"{col[:-2]}_y_m" if col.endswith("_y") else f"{col[:-2]}_z_m"
                )
                if metr in data.columns:
                    final_cols.append(metr)
            else:
                # frame_index or other non-coordinate columns
                if col in data.columns:
                    final_cols.append(col)
        # Add the new ones at the end
        final_cols += rel_cols + norm_cols + metadata_cols

        # Make sure CG columns are included
        cg_columns = ["cg_x_m", "cg_y_m", "cg_x_normalized", "cg_y_normalized"]
        for col in cg_columns:
            if col in data.columns and col not in final_cols:
                final_cols.append(col)

        # 5. Save the calibrated file with the correct column order
        calibrated_data = data[final_cols].copy()
        output_calibrated_file = os.path.join(
            output_dir, f"{base_name}_calibrated_{timestamp}.csv"
        )
        calibrated_data.to_csv(output_calibrated_file, index=False)
        print(
            f"Calibrated data saved (ordered, all original headers first, then _rel and normalized): {output_calibrated_file}"
        )

        # Initialize the list of plot files
        plot_files = []

        # 1. Generate the stick figures phases (subplot) primeiro
        stickfig_phases_file = os.path.join(
            output_dir, f"{base_name}_stickfigures_phases_{timestamp}.png"
        )
        plot_jump_stickfigures_subplot(output_calibrated_file, stickfig_phases_file)
        plot_files.append(stickfig_phases_file)

        # 2. Generate the stick figure with CG pathway
        stickfig_output_file = os.path.join(
            output_dir, f"{base_name}_stickfigures_cg_{timestamp}.png"
        )
        plot_jump_stickfigures_with_cg(output_calibrated_file, stickfig_output_file)
        plot_files.append(stickfig_output_file)

        # 3. Generate other jump plots (time series, phases, etc.)
        other_plot_files = generate_jump_plots(data, results, output_dir, base_name)
        plot_files.extend(other_plot_files)

        # 4. Generate diagnostic plot and add it
        diagnostic_plot = generate_normalized_diagnostic_plot(
            data,
            takeoff_frame,
            landing_frame,
            max_height_frame,
            squat_frame,
            fps,
            output_dir,
            base_name,
        )
        plot_files.append(diagnostic_plot)

        # 5. Generate HTML report with all plots
        output_metrics_file = os.path.join(
            output_dir, f"{base_name}_jump_metrics_{timestamp}.csv"
        )
        results_df.to_csv(output_metrics_file, index=False)
        report_path = generate_html_report(
            data, results, plot_files, output_dir, base_name
        )

        print(f"Jump metrics saved at: {output_metrics_file}")
        # print(f"Complete data saved at: {output_data_file}")
        print(f"Calibrated data (in meters) saved at: {output_calibrated_file}")
        print(f"Jump analysis plots saved in: {output_dir}")
        print(f"HTML report generated: {report_path}")

        # Print diagnostic information
        print(f"Diagnostic info (with corrected orientation):")
        print(
            f"  Reference CG position: {cg_y_ref:.3f} m (set as zero reference for normalized values)"
        )
        print(f"  Squat frame (lowest): {squat_frame} (time: {squat_frame/fps:.3f} s)")
        print(f"  Takeoff frame: {takeoff_frame} (time: {takeoff_frame/fps:.3f} s)")
        print(
            f"  Max height frame: {max_height_frame} (time: {max_height_frame/fps:.3f} s)"
        )
        print(f"  Landing frame: {landing_frame} (time: {landing_frame/fps:.3f} s)")
        print(f"  Flight time: {flight_time:.3f} s")
        print(f"  Jump height: {jump_height:.3f} m (from initial CG position)")

        # In function process_mediapipe_data, after calculating cg_x_ref and cg_y_ref:

        # First, adjust the coordinates in meters to the same reference as the CG
        print("Adjusting all coordinates to the same reference as the CG...")
        for col in [c for c in data.columns if c.endswith("_x_m")]:
            # Do not adjust the CG, as this is already done in the _normalized columns
            if col != "cg_x_m":
                data[col] = data[col] - cg_x_ref
                print(f"  Adjusted {col} - referenced to the initial CG")

        for col in [c for c in data.columns if c.endswith("_y_m")]:
            # Não ajuste o CG, pois isso já é feito nas colunas _normalized
            if col != "cg_y_m":
                data[col] = data[col] - cg_y_ref
                print(f"  Adjusted {col} - referenced to the initial CG")

        # Now update the cg_x_m and cg_y_m columns to use the same reference
        data["cg_x_m"] = data["cg_x_m"] - cg_x_ref
        data["cg_y_m"] = data["cg_y_m"] - cg_y_ref

        # Then, create the _rel versions as copies (now they will be identical, since they are already referenced)
        print(
            "Creating _rel versions (identical, since coordinates are already referenced)..."
        )
        rel_cols = {}

        # Colete todas as colunas x_m em um dicionário
        for col in [c for c in data.columns if c.endswith("_x_m")]:
            rel_cols[f"{col[:-2]}_rel"] = data[col].copy()
            print(f"  Created {col[:-2]}_rel")

        # Colete todas as colunas y_m em um dicionário
        for col in [c for c in data.columns if c.endswith("_y_m")]:
            rel_cols[f"{col[:-2]}_rel"] = data[col].copy()
            print(f"  Created {col[:-2]}_rel")

        # Add all the new columns at once
        data = pd.concat([data, pd.DataFrame(rel_cols)], axis=1)

        # The normalized columns can remain as they are
        data["cg_x_normalized"] = data["cg_x_m"]
        data["cg_y_normalized"] = data["cg_y_m"]

        # Store the original references as metadata for conversion back if needed
        data["reference_cg_x"] = cg_x_ref
        data["reference_cg_y"] = cg_y_ref

        return True

    except Exception as e:
        print(f"Error processing {input_file} (MediaPipe): {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def process_all_mediapipe_files(target_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_mediapipejump_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".csv")
    ]

    for input_file in csv_files:
        print(f"Processing MediaPipe file: {input_file}")
        process_mediapipe_data(input_file, output_dir)
    print("All MediaPipe files have been processed successfully.")


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
        mass_col = next((col for col in columns if "mass" in col.lower()), columns[0])

        if use_time_of_flight:
            value_col = next(
                (
                    col
                    for col in columns
                    if "time" in col.lower() and "contact" not in col.lower()
                ),
                columns[1],
            )
        else:
            value_col = next(
                (
                    col
                    for col in columns
                    if "height" in col.lower() or "heigth" in col.lower()
                ),
                columns[1],
            )

        contact_col = next((col for col in columns if "contact" in col.lower()), None)

        print(
            f"Using columns: Mass={mass_col}, {'Time' if use_time_of_flight else 'Height'}={value_col}, Contact={contact_col}"
        )

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


def calc_fator_convert_mediapipe(
    df, knee="right_knee", ankle="right_ankle", shank_length_real=0.4
):
    # frame 0
    rkx, rky = df[f"{knee}_x"].iloc[0], df[f"{knee}_y"].iloc[0]
    rax, ray = df[f"{ankle}_x"].iloc[0], df[f"{ankle}_y"].iloc[0]
    normalized_length = np.sqrt((rkx - rax) ** 2 + (rky - ray) ** 2)
    factor = shank_length_real / normalized_length
    return factor


def calc_fator_convert_mediapipe(df, shank_length_real):
    rkx, rky = df["right_knee_x"].iloc[0], df["right_knee_y"].iloc[0]
    rax, ray = df["right_ankle_x"].iloc[0], df["right_ankle_y"].iloc[0]
    normalized_length = np.sqrt((rkx - rax) ** 2 + (rky - ray) ** 2)
    factor = shank_length_real / normalized_length
    return factor


def midpoint(df, p1, p2):
    return (df[f"{p1}_x"] + df[f"{p2}_x"]) / 2, (df[f"{p1}_y"] + df[f"{p2}_y"]) / 2


def calculate_cg_frame(df, factor):
    # Locations of CGs in proportion to the segment
    locations = {
        "head": 0.5,
        "trunk": 0.5,
        "upperarm": 0.436,
        "forearm": 0.430,
        "hand": 0.506,
        "thigh": 0.433,
        "shank": 0.433,
        "foot": 0.5,
    }
    # Relative masses
    masses = {
        "head": 0.081,
        "trunk": 0.497,
        "upperarm": 0.028,
        "forearm": 0.016,
        "hand": 0.006,
        "thigh": 0.100,
        "shank": 0.047,
        "foot": 0.014,
    }
    # Note: y coordinates have already been inverted (1.0 - y) at the beginning of processing

    # Head: midpoint between eyes and shoulders
    head_prox_x, head_prox_y = midpoint(df, "left_eye", "right_eye")
    head_dist_x, head_dist_y = midpoint(df, "left_shoulder", "right_shoulder")
    cg_head_x = head_prox_x + locations["head"] * (head_dist_x - head_prox_x)
    cg_head_y = head_prox_y + locations["head"] * (head_dist_y - head_prox_y)
    # Trunk: midpoint shoulders and hips
    trunk_prox_x, trunk_prox_y = midpoint(df, "left_shoulder", "right_shoulder")
    trunk_dist_x, trunk_dist_y = midpoint(df, "left_hip", "right_hip")
    cg_trunk_x = trunk_prox_x + locations["trunk"] * (trunk_dist_x - trunk_prox_x)
    cg_trunk_y = trunk_prox_y + locations["trunk"] * (trunk_dist_y - trunk_prox_y)
    # Upper left limb segments
    cg_upperarm_l_x = df["left_shoulder_x"] + locations["upperarm"] * (
        df["left_elbow_x"] - df["left_shoulder_x"]
    )
    cg_upperarm_l_y = df["left_shoulder_y"] + locations["upperarm"] * (
        df["left_elbow_y"] - df["left_shoulder_y"]
    )
    cg_forearm_l_x = df["left_elbow_x"] + locations["forearm"] * (
        df["left_wrist_x"] - df["left_elbow_x"]
    )
    cg_forearm_l_y = df["left_elbow_y"] + locations["forearm"] * (
        df["left_wrist_y"] - df["left_elbow_y"]
    )
    cg_hand_l_x = df["left_wrist_x"] + locations["hand"] * (
        midpoint(df, "left_pinky", "left_index")[0] - df["left_wrist_x"]
    )
    cg_hand_l_y = df["left_wrist_y"] + locations["hand"] * (
        midpoint(df, "left_pinky", "left_index")[1] - df["left_wrist_y"]
    )
    # Upper right limb segments
    cg_upperarm_r_x = df["right_shoulder_x"] + locations["upperarm"] * (
        df["right_elbow_x"] - df["right_shoulder_x"]
    )
    cg_upperarm_r_y = df["right_shoulder_y"] + locations["upperarm"] * (
        df["right_elbow_y"] - df["right_shoulder_y"]
    )
    cg_forearm_r_x = df["right_elbow_x"] + locations["forearm"] * (
        df["right_wrist_x"] - df["right_elbow_x"]
    )
    cg_forearm_r_y = df["right_elbow_y"] + locations["forearm"] * (
        df["right_wrist_y"] - df["right_elbow_y"]
    )
    cg_hand_r_x = df["right_wrist_x"] + locations["hand"] * (
        midpoint(df, "right_pinky", "right_index")[0] - df["right_wrist_x"]
    )
    cg_hand_r_y = df["right_wrist_y"] + locations["hand"] * (
        midpoint(df, "right_pinky", "right_index")[1] - df["right_wrist_y"]
    )
    # Left thigh, shank, foot
    cg_thigh_l_x = df["left_hip_x"] + locations["thigh"] * (
        df["left_knee_x"] - df["left_hip_x"]
    )
    cg_thigh_l_y = df["left_hip_y"] + locations["thigh"] * (
        df["left_knee_y"] - df["left_hip_y"]
    )
    cg_shank_l_x = df["left_knee_x"] + locations["shank"] * (
        df["left_ankle_x"] - df["left_knee_x"]
    )
    cg_shank_l_y = df["left_knee_y"] + locations["shank"] * (
        df["left_ankle_y"] - df["left_knee_y"]
    )
    cg_foot_l_x = df["left_heel_x"] + locations["foot"] * (
        df["left_foot_index_x"] - df["left_heel_x"]
    )
    cg_foot_l_y = df["left_heel_y"] + locations["foot"] * (
        df["left_foot_index_y"] - df["left_heel_y"]
    )
    # Right thigh, shank, foot
    cg_thigh_r_x = df["right_hip_x"] + locations["thigh"] * (
        df["right_knee_x"] - df["right_hip_x"]
    )
    cg_thigh_r_y = df["right_hip_y"] + locations["thigh"] * (
        df["right_knee_y"] - df["right_hip_y"]
    )
    cg_shank_r_x = df["right_knee_x"] + locations["shank"] * (
        df["right_ankle_x"] - df["right_knee_x"]
    )
    cg_shank_r_y = df["right_knee_y"] + locations["shank"] * (
        df["right_ankle_y"] - df["right_knee_y"]
    )
    cg_foot_r_x = df["right_heel_x"] + locations["foot"] * (
        df["right_foot_index_x"] - df["right_heel_x"]
    )
    cg_foot_r_y = df["right_heel_y"] + locations["foot"] * (
        df["right_foot_index_y"] - df["right_heel_y"]
    )
    # Vectors of the coordinates of the CGs of the segments
    cg_x = (
        masses["head"] * cg_head_x
        + masses["trunk"] * cg_trunk_x
        + masses["upperarm"] * (cg_upperarm_l_x + cg_upperarm_r_x)
        + masses["forearm"] * (cg_forearm_l_x + cg_forearm_r_x)
        + masses["hand"] * (cg_hand_l_x + cg_hand_r_x)
        + masses["thigh"] * (cg_thigh_l_x + cg_thigh_r_x)
        + masses["shank"] * (cg_shank_l_x + cg_shank_r_x)
        + masses["foot"] * (cg_foot_l_x + cg_foot_r_x)
    ) / 1.0

    cg_y = (
        masses["head"] * cg_head_y
        + masses["trunk"] * cg_trunk_y
        + masses["upperarm"] * (cg_upperarm_l_y + cg_upperarm_r_y)
        + masses["forearm"] * (cg_forearm_l_y + cg_forearm_r_y)
        + masses["hand"] * (cg_hand_l_y + cg_hand_r_y)
        + masses["thigh"] * (cg_thigh_l_y + cg_thigh_r_y)
        + masses["shank"] * (cg_shank_l_y + cg_shank_r_y)
        + masses["foot"] * (cg_foot_l_y + cg_foot_r_y)
    ) / 1.0
    # Apply conversion factor to meters
    cg_x_m = cg_x * factor
    cg_y_m = cg_y * factor
    return cg_x_m, cg_y_m


def jump_height_mediapipe(df, shank_length_real):
    factor = calc_fator_convert_mediapipe(df, shank_length_real)
    cg_x, cg_y = calculate_cg_frame(df, factor)
    # Now y increases upward, so the highest value is the peak!
    jump_height = cg_y.max() - cg_y.iloc[:10].mean()
    return jump_height, cg_x, cg_y


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


def generate_normalized_diagnostic_plot(
    data,
    takeoff_frame,
    landing_frame,
    max_height_frame,
    squat_frame,
    fps,
    output_dir,
    base_name,
):
    """
    Generate a diagnostic plot showing the normalized CG position and jump phases.
    With corrected orientation: y increases upward, so jump appears as a peak.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.figure(figsize=(14, 10)), plt.subplot(111)

    # Convert frames to time
    time = data.index / fps

    # Plot normalized CG trajectory
    ax.plot(
        time,
        data["cg_y_normalized"],
        "b-",
        linewidth=2,
        label="CG Position (normalized m)",
    )

    # Plot feet positions
    if "left_foot_index_y_m" in data.columns:
        normalized_left_foot = data["left_foot_index_y_m"] - data["reference_cg_y"]
        ax.plot(
            time,
            normalized_left_foot,
            "g-",
            linewidth=1,
            alpha=0.7,
            label="Left Foot (normalized)",
        )
    if "right_foot_index_y_m" in data.columns:
        normalized_right_foot = data["right_foot_index_y_m"] - data["reference_cg_y"]
        ax.plot(
            time,
            normalized_right_foot,
            "r-",
            linewidth=1,
            alpha=0.7,
            label="Right Foot (normalized)",
        )

    # Plot reference line (zero)
    ax.axhline(y=0, color="gray", linestyle="-", label="Reference Position")

    # Mark important frames
    squat_time = squat_frame / fps
    takeoff_time = takeoff_frame / fps
    landing_time = landing_frame / fps
    max_height_time = max_height_frame / fps

    ax.axvline(
        x=squat_time, color="brown", linestyle="-", label=f"Squat (Frame {squat_frame})"
    )
    ax.axvline(
        x=takeoff_time,
        color="purple",
        linestyle="-",
        label=f"Takeoff (Frame {takeoff_frame})",
    )
    ax.axvline(
        x=landing_time,
        color="orange",
        linestyle="-",
        label=f"Landing (Frame {landing_frame})",
    )
    ax.axvline(
        x=max_height_time,
        color="red",
        linestyle="-",
        label=f"Max Height (Frame {max_height_frame})",
    )

    # Shade the flight phase
    ax.axvspan(
        takeoff_time,
        landing_time,
        alpha=0.2,
        color="yellow",
        label=f"Flight Phase: {landing_time-takeoff_time:.3f} s",
    )

    # Add annotations
    jump_height = data["cg_y_normalized"].iloc[max_height_frame]
    ax.annotate(
        f"Jump Height: {jump_height:.3f} m (from initial CG position)",
        xy=(max_height_time, jump_height),
        xytext=(max_height_time, jump_height * 0.8),
        arrowprops=dict(facecolor="black", shrink=0.05),
        ha="center",
    )

    # Add frame numbers at regular intervals
    frames_to_show = np.arange(0, len(data), 20)
    # Position frame numbers at bottom of plot
    y_pos = min(data["cg_y_normalized"]) - 0.05
    for frame in frames_to_show:
        ax.text(frame / fps, y_pos, f"{frame}", fontsize=8, ha="center")

    # Set labels and title
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Position (meters from reference) - Up is positive")
    ax.set_title("Jump Analysis - Normalized CG Position (Corrected Orientation)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Save plot
    plot_path = os.path.join(
        output_dir, f"{base_name}_normalized_diagnostic_{timestamp}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path


def plot_jump_stickfigures_with_cg(
    csv_file,
    output_file,
    frames_plot=None,
    labels_plot=None,
    colors=None,
    body_segments=None,
    figsize=(10, 7),
):
    """
    Plot stick figures at key jump frames and the path of the center of gravity (CG).
    Uses normalized coordinates for consistency.
    """
    df = pd.read_csv(csv_file)

    # Check if we have _rel columns for body segments
    rel_segments_available = ("left_shoulder_x_rel" in df.columns) and (
        "left_shoulder_y_rel" in df.columns
    )

    # If we have _rel columns, use those + cg_x/y_normalized
    if rel_segments_available:
        suffix = "_rel"
        possible_cg_y = [
            c for c in ["cg_y_normalized", "cg_y_m_rel"] if c in df.columns
        ]
        possible_cg_x = [
            c for c in ["cg_x_normalized", "cg_x_m_rel"] if c in df.columns
        ]
        print("Using relative coordinates (_rel) for combined visualization")
    else:
        # Fall back to meter coordinates if relative not available
        suffix = "_m"
        possible_cg_y = [c for c in ["cg_y_m"] if c in df.columns]
        possible_cg_x = [c for c in ["cg_x_m"] if c in df.columns]
        print("Using meter coordinates (_m) for combined visualization")

    if not possible_cg_x or not possible_cg_y:
        raise ValueError("Cannot find CG coordinates in the CSV file")

    cg_x_col = possible_cg_x[0]
    cg_y_col = possible_cg_y[0]
    print(f"Using CG columns: {cg_x_col} / {cg_y_col} for stick figure plot")
    print(f"Using body segment suffix: '{suffix}'")

    # Auto-select frames if not provided
    frame_initial = 0
    frame_squat = df[cg_y_col].idxmin()
    frame_peak = df[cg_y_col].idxmax()
    takeoff_candidates = df.index[
        (df.index > frame_squat) & (df[cg_y_col] > df[cg_y_col].iloc[frame_initial])
    ]
    frame_takeoff = (
        takeoff_candidates.min() if len(takeoff_candidates) > 0 else frame_initial
    )
    landing_candidates = df.index[
        (df.index > frame_peak)
        & (df[cg_y_col] < df[cg_y_col].iloc[frame_initial] + 0.01)
    ]
    frame_landing = (
        landing_candidates.min() if len(landing_candidates) > 0 else df.index[-1]
    )

    if frames_plot is None:
        frames_plot = [
            frame_initial,
            frame_squat,
            frame_takeoff,
            frame_peak,
            frame_landing,
        ]
    if labels_plot is None:
        labels_plot = ["Initial", "Squat", "Takeoff", "Peak", "Landing"]
    if colors is None:
        colors = ["black", "red", "blue", "green", "purple"]
    if body_segments is None:
        body_segments = [
            ("left_ankle", "left_knee"),
            ("left_knee", "left_hip"),
            ("right_ankle", "right_knee"),
            ("right_knee", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "left_shoulder"),
            ("right_hip", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "nose"),
            ("right_shoulder", "nose"),
        ]

    # Check which segments are available with the current suffix
    available_segments = []
    for start, end in body_segments:
        x_start = f"{start}_x{suffix}"
        y_start = f"{start}_y{suffix}"
        x_end = f"{end}_x{suffix}"
        y_end = f"{end}_y{suffix}"

        if all(col in df.columns for col in [x_start, y_start, x_end, y_end]):
            available_segments.append((start, end))

    print(
        f"Available segments with suffix '{suffix}': {len(available_segments)} of {len(body_segments)}"
    )

    # If no segments available, try alternative suffix
    if len(available_segments) == 0:
        alt_suffix = "_m" if suffix == "_rel" else "_rel"
        print(f"No segments available with '{suffix}', trying '{alt_suffix}'")

        alt_available_segments = []
        for start, end in body_segments:
            x_start = f"{start}_x{alt_suffix}"
            y_start = f"{start}_y{alt_suffix}"
            x_end = f"{end}_x{alt_suffix}"
            y_end = f"{end}_y{alt_suffix}"

            if all(col in df.columns for col in [x_start, y_start, x_end, y_end]):
                alt_available_segments.append((start, end))

        if len(alt_available_segments) > 0:
            suffix = alt_suffix
            available_segments = alt_available_segments
            print(
                f"Using alternative suffix '{suffix}' with {len(available_segments)} segments"
            )

            # Update CG columns to match
            if suffix == "_rel":
                possible_cg_y = [
                    c for c in ["cg_y_normalized", "cg_y_m_rel"] if c in df.columns
                ]
                possible_cg_x = [
                    c for c in ["cg_x_normalized", "cg_x_m_rel"] if c in df.columns
                ]
            else:
                possible_cg_y = [c for c in ["cg_y_m"] if c in df.columns]
                possible_cg_x = [c for c in ["cg_x_m"] if c in df.columns]

            cg_x_col = possible_cg_x[0] if possible_cg_x else cg_x_col
            cg_y_col = possible_cg_y[0] if possible_cg_y else cg_y_col

    # If still no segments available, create warning
    if len(available_segments) == 0:
        print(
            "WARNING: No body segments could be matched. Creating a simple plot with CG points only."
        )
    else:
        body_segments = available_segments

    plt.figure(figsize=figsize)
    # Plot CG pathway
    plt.plot(df[cg_x_col], df[cg_y_col], "--", color="gray", label="CG Path")

    # Plot stick figures and CG at key moments
    for idx, (frame, label, color) in enumerate(zip(frames_plot, labels_plot, colors)):
        row = df.iloc[frame]
        for start, end in body_segments:
            x_start = f"{start}_x{suffix}"
            y_start = f"{start}_y{suffix}"
            x_end = f"{end}_x{suffix}"
            y_end = f"{end}_y{suffix}"

            if all(col in row.index for col in [x_start, y_start, x_end, y_end]):
                if not any(
                    pd.isna(row[col]) for col in [x_start, y_start, x_end, y_end]
                ):
                    plt.plot(
                        [row[x_start], row[x_end]],
                        [row[y_start], row[y_end]],
                        color=color,
                        lw=2,
                    )

        plt.plot(
            row[cg_x_col],
            row[cg_y_col],
            "o",
            color=color,
            label=f"{label} (frame {frame})",
            markersize=10,
        )

    plt.xlabel("X (m) - relative to initial CG")
    plt.ylabel("Y (m) - relative to initial CG")
    plt.legend()
    plt.title("Stick Figures and CG Pathway during Counter-Movement Jump")
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    return output_file


def plot_jump_stickfigures_subplot(
    csv_file,
    output_file,
    frames_plot=None,
    labels_plot=None,
    colors=None,
    body_segments=None,
    figsize=(20, 7),
):
    """
    Plot a sequence of stick figures (one per subplot) with the CG at each phase.
    Uses normalized coordinates for consistency.
    """
    df = pd.read_csv(csv_file)

    # Debug info about the available columns
    print(f"DEBUG: CSV file has {len(df.columns)} columns")
    coord_cols = [
        c for c in df.columns if c.endswith(("_x_m", "_y_m", "_x_rel", "_y_rel"))
    ]
    print(f"DEBUG: Found {len(coord_cols)} coordinate columns")
    print(
        f"DEBUG: Sample coordinate columns: {coord_cols[:10] if len(coord_cols) > 10 else coord_cols}"
    )
    cg_cols = [c for c in df.columns if "cg" in c]
    print(f"DEBUG: CG columns found: {cg_cols}")

    # First, determine if we're using relative/normalized coordinates
    # Check if we have body segment columns with _rel suffix
    rel_segments_available = ("left_shoulder_x_rel" in df.columns) and (
        "left_shoulder_y_rel" in df.columns
    )

    # If we have _rel columns for body segments, use those + cg_x/y_normalized
    if rel_segments_available:
        suffix = "_rel"
        possible_cg_y = [
            c for c in ["cg_y_normalized", "cg_y_m_rel"] if c in df.columns
        ]
        possible_cg_x = [
            c for c in ["cg_x_normalized", "cg_x_m_rel"] if c in df.columns
        ]
        print("Using relative coordinates (_rel) for visualization")
    else:
        # Fall back to meter coordinates if relative not available
        suffix = "_m"
        possible_cg_y = [c for c in ["cg_y_m"] if c in df.columns]
        possible_cg_x = [c for c in ["cg_x_m"] if c in df.columns]
        print("Using meter coordinates (_m) for visualization")

    if not possible_cg_x or not possible_cg_y:
        raise ValueError("Cannot find CG coordinates in the CSV file")

    cg_x_col = possible_cg_x[0]
    cg_y_col = possible_cg_y[0]
    print(f"Using CG columns: {cg_x_col} / {cg_y_col} for stick figure visualization")
    print(f"Using body segment suffix: '{suffix}'")

    # Identify key frames for jump phases if not provided
    if frames_plot is None:
        frame_initial = 0
        frame_squat = df[cg_y_col].idxmin()  # Lowest position (squat)
        frame_peak = df[cg_y_col].idxmax()  # Highest position (peak)

        # Find takeoff: first frame after squat where CG rises above initial position
        takeoff_candidates = df.index[
            (df.index > frame_squat) & (df[cg_y_col] > df[cg_y_col].iloc[frame_initial])
        ]
        frame_takeoff = (
            takeoff_candidates.min() if len(takeoff_candidates) > 0 else frame_squat
        )

        # Find landing: first frame after peak where CG returns near initial position
        landing_candidates = df.index[
            (df.index > frame_peak)
            & (df[cg_y_col] < df[cg_y_col].iloc[frame_initial] + 0.01)
        ]
        frame_landing = (
            landing_candidates.min() if len(landing_candidates) > 0 else df.index[-1]
        )

        frames_plot = [
            frame_initial,
            frame_squat,
            frame_takeoff,
            frame_peak,
            frame_landing,
        ]
        print(
            f"Auto-detected frames - Initial: {frame_initial}, Squat: {frame_squat}, "
            f"Takeoff: {frame_takeoff}, Peak: {frame_peak}, Landing: {frame_landing}"
        )

    if labels_plot is None:
        labels_plot = ["Initial", "Squat", "Takeoff", "Peak", "Landing"]

    if colors is None:
        colors = ["black", "red", "blue", "green", "purple"]

    if body_segments is None:
        body_segments = [
            ("left_ankle", "left_knee"),
            ("left_knee", "left_hip"),
            ("right_ankle", "right_knee"),
            ("right_knee", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "left_shoulder"),
            ("right_hip", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "nose"),
            ("right_shoulder", "nose"),
        ]

    # Check which segments are available with the current suffix
    available_segments = []
    for start, end in body_segments:
        x_start = f"{start}_x{suffix}"
        y_start = f"{start}_y{suffix}"
        x_end = f"{end}_x{suffix}"
        y_end = f"{end}_y{suffix}"

        if all(col in df.columns for col in [x_start, y_start, x_end, y_end]):
            available_segments.append((start, end))

    print(
        f"Available segments with suffix '{suffix}': {len(available_segments)} of {len(body_segments)}"
    )

    # If no segments available with current suffix, try alternative
    if len(available_segments) == 0:
        alt_suffix = "_m" if suffix == "_rel" else "_rel"
        print(f"No segments available with '{suffix}', trying '{alt_suffix}'")

        alt_available_segments = []
        for start, end in body_segments:
            x_start = f"{start}_x{alt_suffix}"
            y_start = f"{start}_y{alt_suffix}"
            x_end = f"{end}_x{alt_suffix}"
            y_end = f"{end}_y{alt_suffix}"

            if all(col in df.columns for col in [x_start, y_start, x_end, y_end]):
                alt_available_segments.append((start, end))

        print(
            f"Available segments with '{alt_suffix}': {len(alt_available_segments)} of {len(body_segments)}"
        )

        if len(alt_available_segments) > 0:
            suffix = alt_suffix
            available_segments = alt_available_segments

            # Update CG columns to match the suffix type
            if suffix == "_rel":
                possible_cg_y = [
                    c for c in ["cg_y_normalized", "cg_y_m_rel"] if c in df.columns
                ]
                possible_cg_x = [
                    c for c in ["cg_x_normalized", "cg_x_m_rel"] if c in df.columns
                ]
            else:
                possible_cg_y = [c for c in ["cg_y_m"] if c in df.columns]
                possible_cg_x = [c for c in ["cg_x_m"] if c in df.columns]

            cg_x_col = possible_cg_x[0] if possible_cg_x else cg_x_col
            cg_y_col = possible_cg_y[0] if possible_cg_y else cg_y_col
            print(f"Updated to use CG columns: {cg_x_col} / {cg_y_col}")

    # If still no segments available, report error
    if len(available_segments) == 0:
        print(
            "WARNING: No body segments could be matched. Creating a simple plot with CG points only."
        )
    else:
        body_segments = available_segments

    # Create figure with subplots - one for each phase
    n_phases = len(frames_plot)
    fig, axes = plt.subplots(1, n_phases, figsize=figsize)

    # Ensure axes is always an array (even with a single subplot)
    if n_phases == 1:
        axes = [axes]

    # Get common y limits to ensure consistent scaling
    all_y_values = []
    all_x_values = []

    for frame in frames_plot:
        if frame >= len(df):
            print(f"WARNING: Frame {frame} exceeds DataFrame length ({len(df)})")
            continue

        row = df.iloc[frame]

        # Add CG to the limits calculation
        if cg_x_col in row and cg_y_col in row:
            all_x_values.append(float(row[cg_x_col]))
            all_y_values.append(float(row[cg_y_col]))

        # Add body segments to the limits calculation
        for start, end in body_segments:
            x_start = f"{start}_x{suffix}"
            y_start = f"{start}_y{suffix}"
            x_end = f"{end}_x{suffix}"
            y_end = f"{end}_y{suffix}"

            if all(col in row.index for col in [x_start, y_start, x_end, y_end]):
                if not any(
                    pd.isna(row[col]) for col in [x_start, y_start, x_end, y_end]
                ):
                    all_x_values.extend([float(row[x_start]), float(row[x_end])])
                    all_y_values.extend([float(row[y_start]), float(row[y_end])])

    if all_y_values and all_x_values:
        y_min, y_max = min(all_y_values), max(all_y_values)
        x_min, x_max = min(all_x_values), max(all_x_values)
        # Add padding
        y_range = y_max - y_min
        x_range = x_max - x_min
        y_padding = y_range * 0.2  # Increased padding for better visualization
        x_padding = x_range * 0.2
        y_min -= y_padding
        y_max += y_padding
        x_min -= x_padding
        x_max += x_padding

    # Plot each phase in its own subplot
    for i, (ax, frame, label, color) in enumerate(
        zip(axes, frames_plot, labels_plot, colors)
    ):
        if frame >= len(df):
            ax.text(
                0.5,
                0.5,
                f"Frame {frame} out of range",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        row = df.iloc[frame]

        # Draw body segments (stick figure)
        for start, end in body_segments:
            x_start = f"{start}_x{suffix}"
            y_start = f"{start}_y{suffix}"
            x_end = f"{end}_x{suffix}"
            y_end = f"{end}_y{suffix}"

            if all(col in row.index for col in [x_start, y_start, x_end, y_end]):
                if not any(
                    pd.isna(row[col]) for col in [x_start, y_start, x_end, y_end]
                ):
                    ax.plot(
                        [row[x_start], row[x_end]],
                        [row[y_start], row[y_end]],
                        color=color,
                        lw=2,
                    )

        # Plot the CG with a distinct marker
        if cg_x_col in row and cg_y_col in row:
            cg_x = row[cg_x_col]
            cg_y = row[cg_y_col]
            if not pd.isna(cg_x) and not pd.isna(cg_y):
                ax.plot(
                    cg_x,
                    cg_y,
                    "o",
                    color="orange",
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label="CG",
                )

        # Set common limits for consistent scale across subplots
        if all_y_values and all_x_values:
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)

        # Add frame info and title
        ax.set_title(f"{label}\n(frame {frame})", fontsize=12)
        ax.set_xlabel("X (m) - relative to initial CG")
        ax.set_aspect("equal", "box")  # Force equal aspect ratio

        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel("Y (m) - relative to initial CG")

        # Add grid for better visibility
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend only to the first subplot to avoid redundancy
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)

    # Add overall title
    fig.suptitle("Vertical Jump Phases with Center of Gravity (CG)", fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    # Save the figure
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Stick figure phases visualization saved: {output_file}")
    return output_file


def vaila_and_jump():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Force dialogs to be on top

    target_dir = filedialog.askdirectory(
        title="Select the target directory containing .csv files", parent=root
    )
    root.lift()  # Bring window to the front

    if not target_dir:
        messagebox.showwarning("Warning", "No target directory selected.", parent=root)
        root.destroy()
        return

    # Now with option 3!
    data_type = simpledialog.askinteger(
        "Select Data Type",
        "Select the type of data in your CSV files:\n\n"
        "1. Time of Flight Data\n"
        "2. Jump Height Data\n"
        "3. MediaPipe Ankle Data",
        parent=root,  # Set parent window
        minvalue=1,
        maxvalue=3,
    )
    root.lift()  # Bring window to the front

    if data_type is None:
        messagebox.showwarning(
            "Warning", "No data type selected. Exiting.", parent=root
        )
        root.destroy()
        return

    if data_type == 1 or data_type == 2:
        use_time_of_flight = data_type == 1
        process_all_files_in_directory(target_dir, use_time_of_flight)
    elif data_type == 3:
        process_all_mediapipe_files(target_dir)

    msg = "All CSV files have been processed and results saved.\n\n"
    if data_type == 1:
        msg += f"Input data type: Time of Flight\n"
    elif data_type == 2:
        msg += f"Input data type: Jump Height\n"
    else:
        msg += f"Input data type: MediaPipe Ankle\n"
    msg += f"Output directory: {os.path.join(target_dir, 'vaila_verticaljump_*')}"
    messagebox.showinfo("Success", msg, parent=root)

    root.destroy()


if __name__ == "__main__":
    vaila_and_jump()
