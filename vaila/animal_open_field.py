"""
===============================================================================
heatmap_pathway_plot.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 16 December 2024
Version: 2.1.0
Python Version: 3.11.11

Description:
------------
This script processes movement data of animals in an open field test, performing 
comprehensive kinematic analyses and generating visualizations to evaluate animal behavior.

Key Features:
-------------
- Reads movement data from .csv files (X and Y positions over time).
- Calculates key metrics, including:
  - Total distance traveled.
  - Average speed.
  - Time stationary (speed < 0.05 m/s).
  - Time spent in defined speed ranges (0-45 m/min).
- Analyzes the time spent in specific zones of a 60x60 cm open field, divided into 
  3x3 grid cells of 20x20 cm each, including:
  - Percentage and count of time in each zone.
  - Percentage and count of time in the center zone and border areas.
- Generates the following visualizations:
  - Pathway plots with color gradients indicating time progression.
  - Heatmaps of positional density, including zone annotations.
  - Heatmaps highlighting center and border occupancy.
  - Speed over time plots with speed ranges and smoothed curves using moving averages.
  - Bar charts showing time distribution across speed ranges.
- Results and visualizations are saved in a structured directory format for each input file.

Dependencies:
-------------
- Python 3.x
- numpy
- matplotlib
- seaborn
- tkinter

Usage:
------
1. Run the script, and a dialog will prompt for user input.
2. Select the directory containing `.csv` files with movement data.
   - Expected columns in the `.csv` files:
     - `time(s)` - Time in seconds.
     - `position_x(m)` - X-coordinate in meters.
     - `position_y(m)` - Y-coordinate in meters.
3. Input the sampling frequency (Hz) and the cutoff frequency (Hz) for filtering.
4. The script will process all `.csv` files in the selected directory.
5. Results, including figures and a detailed text summary, will be saved in a timestamped 
   directory, with subdirectories for each processed file.

Example:
--------
$ python heatmap_pathway_plot.py

Notes:
------
- Ensure input `.csv` files are correctly formatted with positions in meters.
- Speed smoothing is performed using a moving average over a user-defined window size (e.g., 2 seconds).
- Handles boundary constraints by clipping positional data within the defined 60x60 cm open field.

Changelog:
----------
- v2.1.0:
  - Replaced Butterworth filter with moving average smoothing for speed analysis.
  - Added dynamic window size for speed smoothing based on sampling frequency (e.g., 2 seconds).
  - Enhanced pathway plots with time-based color gradients.
  - Improved directory structure and error handling.
===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tkinter import Tk, filedialog, simpledialog, messagebox
from datetime import datetime
from pathlib import Path
from scipy.signal import butter, filtfilt


def load_and_preprocess_data(input_file):
    """
    Loads the input file, processes the columns for X and Y coordinates,
    and computes their mean if there are multiple X and Y columns.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        tuple: Tuple containing arrays for X and Y coordinates.
    """
    # Load the data, skipping the header
    data = np.loadtxt(input_file, delimiter=",", skiprows=1)

    # Exclude the first column (time/frame)
    data = data[:, 1:]

    # Separate X and Y columns (odd columns for X, even columns for Y)
    x = data[:, ::2]  # Columns at indices 1, 3, 5, ...
    y = data[:, 1::2]  # Columns at indices 2, 4, 6, ...

    # Compute the mean along the rows if there are multiple columns
    x_mean = np.mean(x, axis=1) if x.shape[1] > 1 else x.flatten()
    y_mean = np.mean(y, axis=1) if y.shape[1] > 1 else y.flatten()

    return x_mean, y_mean


def adjust_to_bounds(x, y, xmin=0, xmax=0.6, ymin=0, ymax=0.6):
    """
    Adjusts the values of x and y to ensure they are within the specified bounds.

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        xmin (float): Minimum allowed value for x.
        xmax (float): Maximum allowed value for x.
        ymin (float): Minimum allowed value for y.
        ymax (float): Maximum allowed value for y.

    Returns:
        adjusted_x (array-like): Adjusted X coordinates.
        adjusted_y (array-like): Adjusted Y coordinates.
    """
    adjusted_x = np.clip(x, xmin, xmax)
    adjusted_y = np.clip(y, ymin, ymax)
    return adjusted_x, adjusted_y


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


def define_zones():
    """
    Define os limites das 9 zonas fixas do grid (3x3) com margens de tolerância.
    Returns:
        dict: Coordenadas das zonas com limites xmin, xmax, ymin, ymax.
    """
    zones = {
        "Z1": {"xmin": -0.1, "xmax": 0.2, "ymin": -0.1, "ymax": 0.2},
        "Z2": {"xmin": 0.2000001, "xmax": 0.4, "ymin": -0.1, "ymax": 0.2},
        "Z3": {"xmin": 0.4000001, "xmax": 0.7, "ymin": -0.1, "ymax": 0.2},
        "Z4": {"xmin": -0.1, "xmax": 0.2, "ymin": 0.2000001, "ymax": 0.4},
        "Z5": {"xmin": 0.2000001, "xmax": 0.4, "ymin": 0.20000001, "ymax": 0.4},
        "Z6": {"xmin": 0.4000001, "xmax": 0.7, "ymin": 0.2000001, "ymax": 0.4},
        "Z7": {"xmin": -0.1, "xmax": 0.2, "ymin": 0.4000001, "ymax": 0.7},
        "Z8": {"xmin": 0.2000001, "xmax": 0.4, "ymin": 0.4000001, "ymax": 0.7},
        "Z9": {"xmin": 0.4000001, "xmax": 0.7, "ymin": 0.4000001, "ymax": 0.7},
    }
    return zones


def define_center_zone():
    """
    Define os limites da zona central fixa.
    Returns:
        dict: Coordenadas da zona central com limites xmin, xmax, ymin, ymax.
    """
    center_zone = {"xmin": 0.1, "xmax": 0.5, "ymin": 0.1, "ymax": 0.5}
    return center_zone


def calculate_zone_occupancy(x, y):
    """
    Calculates the number of points and percentages in each zone (3x3 grid).

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.

    Returns:
        zones_count (dict): Count of points in each zone.
        zones_percentage (dict): Percentage of points in each zone.
    """
    zones = define_zones()  # Get zone definitions
    zones_count = {zone: 0 for zone in zones}  # Initialize counts for each zone
    total_points = len(x)

    # Count points in zones
    for xi, yi in zip(x, y):
        for zone_name, limits in zones.items():
            if (
                limits["xmin"] <= xi <= limits["xmax"]
                and limits["ymin"] <= yi <= limits["ymax"]
            ):
                zones_count[zone_name] += 1
                break  # Avoid double counting

    # Calculate percentages
    zones_percentage = {
        zone: (count / total_points) * 100 for zone, count in zones_count.items()
    }

    return zones_count, zones_percentage


def calculate_center_and_border_occupancy(x, y):
    """
    Calculates the number of points and percentages in the center and border zones.

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.

    Returns:
        dict: Count and percentages of points in the center and border zones.
    """
    center_zone = define_center_zone()  # Get the limits of the center zone
    total_points = len(x)

    # Count the points within the center zone
    points_in_center = sum(
        1
        for xi, yi in zip(x, y)
        if center_zone["xmin"] <= xi <= center_zone["xmax"]
        and center_zone["ymin"] <= yi <= center_zone["ymax"]
    )

    # Points outside the center zone are considered to be in the border zone
    points_in_border = total_points - points_in_center

    # Return the results as a formatted dictionary
    return {
        "points_in_center": points_in_center,
        "percentage_in_center": (points_in_center / total_points) * 100,
        "points_in_border": points_in_border,
        "percentage_in_border": (points_in_border / total_points) * 100,
    }


def calculate_kinematics(x, y, fs):
    """
    Calculate kinematic metrics including distance, speed, time stationary, and speed ranges.

    Args:
        x (array-like): Filtered X coordinates.
        y (array-like): Filtered Y coordinates.
        fs (float): Sampling frequency in Hz.

    Returns:
        distance (array-like): Distance traveled between consecutive points.
        speed (array-like): Speed calculated from the distance and sampling frequency.
        time_stationary (float): Total time the animal was stationary (speed < 0.05 m/s).
        speed_range_counts_frames (dict): Count of occurrences in each speed range (frames).
        speed_range_counts_seconds (dict): Total time in seconds spent in each speed range.
    """
    # Calculate distance between consecutive points
    distance = np.insert(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2), 0, 0)

    # Calculate speed as distance divided by time (1/fs for each interval)
    speed = np.insert(distance[1:] / (1 / fs), 0, 0)

    # Calculate time stationary (speed < 0.05 m/s)
    stationary_threshold = 0.05  # m/s
    time_stationary = np.sum(speed < stationary_threshold) / fs  # Total time in seconds

    # Define speed ranges (convert to m/min: multiply by 60)
    speed_ranges = [(3 * i, 3 * (i + 1)) for i in range(15)]  # From 0 to 45 m/min
    speed_range_counts_frames = {f"{low}-{high} m/min": 0 for low, high in speed_ranges}
    speed_range_counts_seconds = {
        f"{low}-{high} m/min": 0 for low, high in speed_ranges
    }

    # Count occurrences in each speed range and calculate time in seconds
    for s in speed * 60:  # Convert speed to m/min
        for low, high in speed_ranges:
            if low <= s < high:
                speed_range_counts_frames[
                    f"{low}-{high} m/min"
                ] += 1  # Increment frame count
                speed_range_counts_seconds[f"{low}-{high} m/min"] += (
                    1 / fs
                )  # Convert to seconds
                break

    return (
        distance,
        speed,
        time_stationary,
        speed_range_counts_frames,
        speed_range_counts_seconds,
    )


def plot_pathway(x, y, time_vector, total_distance, output_dir, base_name):
    """
    Plots the pathway of the animal's movement with a color gradient indicating progression
    over time in minutes and shows the total distance covered in the title.

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        time_vector (array-like): Time vector in seconds.
        total_distance (float): Total distance covered in meters.
        output_dir (str): Directory to save the output figure.
        base_name (str): Base name for the output file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert time to minutes
    time_in_minutes = time_vector / 60

    # Create a colormap for the pathway (e.g., from blue to red)
    cmap = LinearSegmentedColormap.from_list(
        "PathwayProgress", ["blue", "green", "yellow", "red"]
    )

    # Normalize the time to range [0, 1] for color mapping
    progress = (time_in_minutes - time_in_minutes.min()) / (
        time_in_minutes.max() - time_in_minutes.min()
    )

    # Create the pathway plot
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(x) - 1):
        ax.plot(
            x[i : i + 2],
            y[i : i + 2],
            color=cmap(progress[i]),  # Color based on time progression
            linewidth=2,
            alpha=0.8,
        )

    # Add start and end points
    ax.scatter(x[0], y[0], color="green", s=50, label="Start", zorder=5)  # Start point
    ax.scatter(x[-1], y[-1], color="red", s=50, label="End", zorder=5)  # End point

    # Add grid and axis limits
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Position X (m)")
    ax.set_ylabel("Position Y (m)")

    # Add title with total distance
    ax.set_title(f"Pathway of Animal Movement\nTotal Distance: {total_distance:.2f} m")

    # Add grid lines for the zones (3x3)
    for i in range(1, 3):
        ax.axvline(i * 0.2, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(i * 0.2, color="black", linestyle="--", linewidth=0.8)

    # Add legend for Start and End points
    # ax.legend()

    # Add a colorbar for the pathway progression
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=time_in_minutes.min(), vmax=time_in_minutes.max()),
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Time (minutes)", rotation=270, labelpad=15)

    # Save the plot
    output_file_path = os.path.join(output_dir, f"{base_name}_pathway_colored.png")
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
    print(f"Pathway plot saved at: {output_file_path}")


def plot_heatmap(x, y, output_dir, base_name, results):
    """
    Plots a corrected heatmap with zones and their respective percentages.

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        output_dir (str): Directory to save the output.
        base_name (str): Base name of the output file.
        results (dict): Processed results containing counts and percentages for each zone.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if the data is empty
    if len(x) == 0 or len(y) == 0:
        print(f"Warning: Empty data for heatmap in {base_name}. Skipping plot.")
        return

    # Create the heatmap
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.kdeplot(
            x=x,
            y=y,
            cmap="coolwarm",
            fill=True,
            levels=100,
            bw_adjust=2,
            thresh=0,
            ax=ax,
        )
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel("Position X (m)")
        ax.set_ylabel("Position Y (m)")
        ax.set_title("Heatmap with Zone Grid")

        # Add grid lines and labels for the zones
        zones = define_zones()
        for zone_name, limits in zones.items():
            # Calculate the center of the zone for text positioning
            center_x = (limits["xmin"] + limits["xmax"]) / 2
            center_y = (limits["ymin"] + limits["ymax"]) / 2

            # Retrieve the percentage for the zone
            percentage = results["zone_percentages"].get(zone_name, 0)

            # Add the zone name and percentage as text
            ax.text(
                center_x,
                center_y,
                f"{zone_name}\n{percentage:.1f}%",
                color="black",
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
            )

            # Add the zone border lines
            ax.axvline(limits["xmin"], color="black", linestyle="--", linewidth=0.8)
            ax.axvline(limits["xmax"], color="black", linestyle="--", linewidth=0.8)
            ax.axhline(limits["ymin"], color="black", linestyle="--", linewidth=0.8)
            ax.axhline(limits["ymax"], color="black", linestyle="--", linewidth=0.8)

        # Save the heatmap
        output_file_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
        plt.savefig(output_file_path, bbox_inches="tight")
        plt.close()
        print(f"Heatmap plot saved at: {output_file_path}")

    except ValueError as e:
        print(f"Error generating heatmap for {base_name}: {e}. Skipping plot.")
        return


def plot_center_and_border_heatmap(x, y, output_dir, base_name, center_border_results):
    """
    Plots a heatmap highlighting the center and border zones.

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        output_dir (str): Directory to save the output.
        base_name (str): Base name of the output file.
        center_border_results (dict): Processed results with counts and percentages
                                      for the center and border zones.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if the data is empty
    if len(x) == 0 or len(y) == 0:
        print(
            f"Warning: Empty data for center and border heatmap in {base_name}. Skipping plot."
        )
        return

    # Create the heatmap
    try:
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.kdeplot(
            x=x,
            y=y,
            cmap="coolwarm",
            fill=True,
            levels=100,
            bw_adjust=1.5,
            thresh=0,
            ax=ax,
        )

        # Add a rectangle for the center zone
        center_zone = define_center_zone()
        rect = plt.Rectangle(
            (center_zone["xmin"], center_zone["ymin"]),
            center_zone["xmax"] - center_zone["xmin"],
            center_zone["ymax"] - center_zone["ymin"],
            linewidth=2,
            edgecolor="black",
            facecolor="none",
            label="Center Zone",
        )
        ax.add_patch(rect)

        # Add text with the percentages for the center and border zones
        ax.text(
            0.3,
            0.3,
            f"Center\n{center_border_results['percentage_in_center']:.1f}%",
            color="black",
            ha="center",
            va="center",
            fontsize=12,
            weight="bold",
        )
        ax.text(
            0.05,
            0.55,
            f"Border\n{center_border_results['percentage_in_border']:.1f}%",
            color="black",
            ha="left",
            va="center",
            fontsize=12,
            weight="bold",
        )

        # Additional configurations
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 0.6)
        ax.set_xlabel("Position X (m)")
        ax.set_ylabel("Position Y (m)")
        ax.set_title("Heatmap with Central and Border Areas")

        # Save the heatmap
        output_file_path = os.path.join(
            output_dir, f"{base_name}_center_border_heatmap.png"
        )
        plt.savefig(output_file_path, bbox_inches="tight")
        plt.close()
        print(f"Central and border heatmap saved at: {output_file_path}")

    except ValueError as e:
        print(
            f"Error generating center and border heatmap for {base_name}: {e}. Skipping plot."
        )
        return


def plot_speed_ranges(
    speed_range_counts_frames, time_stationary_seconds, fs, output_dir, base_name
):
    """
    Plots a bar chart of speed ranges (e.g., 3-6 m/min, 6-9 m/min) including stationary time in seconds.

    Args:
        speed_range_counts_frames (dict): Counts of occurrences in each speed range (frames).
        time_stationary_seconds (float): Total time stationary (speed < 0.05 m/s) in seconds.
        fs (float): Sampling frequency in Hz.
        output_dir (str): Directory to save the plot.
        base_name (str): Base name for the output file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert frames to seconds (frames / fs)
    speed_range_counts_seconds = {
        k: v / fs for k, v in speed_range_counts_frames.items()
    }

    # Add stationary time to the first range (0-3 m/min)
    speed_range_counts_seconds["0-3 m/min"] += time_stationary_seconds

    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ranges = list(speed_range_counts_seconds.keys())
    times = list(speed_range_counts_seconds.values())
    ax.bar(ranges, times, color="blue", alpha=0.7)

    # Add labels and title
    ax.set_xlabel("Speed Range (m/min)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Distribution of Speed Ranges")
    plt.xticks(rotation=45, ha="right")

    # Save the plot
    output_file_path = os.path.join(output_dir, f"{base_name}_speed_ranges.png")
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()
    print(f"Speed ranges plot saved at: {output_file_path}")


def plot_speed_over_time_with_tags(
    time_vector, speed, window_size, output_dir, base_name
):
    """
    Plots speed over time with horizontal lines indicating speed ranges
    and tags for each range. Additionally, overlays a smoothed speed curve
    using a moving average over a 1-second window (30 points).

    Args:
        time_vector (array-like): Time vector in seconds.
        speed (array-like): Speed in m/s.
        output_dir (str): Directory to save the plot.
        base_name (str): Base name for the output file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert speed to m/min
    speed_m_per_min = speed * 60

    # Moving average smoothing using numpy.convolve
    speed_smoothed = np.convolve(
        speed_m_per_min, np.ones(window_size) / window_size, mode="same"
    )

    # Speed ranges (3 m/min to 45 m/min)
    speed_ranges = [3 * i for i in range(16)]  # 0, 3, 6, ..., 45 m/min

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        time_vector, speed_m_per_min, color="blue", linewidth=1.5, label="Speed (m/min)"
    )
    plt.plot(
        time_vector,
        speed_smoothed,
        color="red",
        linewidth=1,
        label="Moving Average (m/min)",
    )

    # Add dashed horizontal lines and tags for the speed ranges
    for y in speed_ranges:
        plt.axhline(y=y, color="black", linestyle="--", linewidth=1, alpha=0.8)
        plt.text(
            x=max(time_vector) * 1.01,  # Place the text to the right of the graph
            y=y,
            s=f"{y} m/min",
            fontsize=10,
            color="black",
            va="center",
        )

    # Add title and labels
    plt.title("Speed Over Time", fontsize=16, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Speed (m/min)", fontsize=12)

    # Configure grid and limits
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(0, max(speed_m_per_min) + 5)
    plt.xlim(
        0, max(time_vector) + (max(time_vector) * 0.05)
    )  # Add space on the X-axis for tags

    # Add legend
    plt.legend(fontsize=10)

    # Save the plot
    output_file_path = os.path.join(
        output_dir, f"{base_name}_speed_over_time_with_tags.png"
    )
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()

    print(f"Speed over time plot with tags saved at: {output_file_path}")


def save_results_to_csv(results, center_border_results, fs, output_dir, base_name):
    """
    Save results including zone occupancy, stationary time, and speed range counts
    in both frames and seconds.

    Args:
        results (dict): Processed results containing zone counts, percentages, and speed data.
        center_border_results (dict): Results for center and border occupancy.
        fs (float): Sampling frequency in Hz.
        output_dir (str): Directory to save the output CSV.
        base_name (str): Base name for the output file.
    """
    try:
        combined_file_path = os.path.join(output_dir, f"{base_name}_summary_zones.csv")

        # Create column headers
        zone_headers_points = [f"z{i}_npoints" for i in range(1, 10)]
        zone_headers_percentage = [f"z{i}_percentage" for i in range(1, 10)]
        additional_headers = [
            "zcenter_npoints",
            "zborder_npoints",
            "zcenter_percentage",
            "zborder_percentage",
        ]
        time_headers = ["time_stationary_seconds"]
        speed_range_headers_frames = [
            f"{k}_frames" for k in results["speed_range_counts_frames"].keys()
        ]
        speed_range_headers_seconds = [
            f"{k}_seconds" for k in results["speed_range_counts_seconds"].keys()
        ]

        headers = (
            zone_headers_points
            + zone_headers_percentage
            + additional_headers
            + time_headers
            + speed_range_headers_frames
            + speed_range_headers_seconds
        )

        # Create data row
        zone_points = [results["zone_counts"].get(f"Z{i}", 0) for i in range(1, 10)]
        zone_percentages = [
            results["zone_percentages"].get(f"Z{i}", 0.0) for i in range(1, 10)
        ]
        center_border_data = [
            center_border_results["points_in_center"],
            center_border_results["points_in_border"],
            center_border_results["percentage_in_center"],
            center_border_results["percentage_in_border"],
        ]
        time_data = [results["time_stationary"]]

        # Extract frames and seconds for speed ranges
        speed_range_frames = list(results["speed_range_counts_frames"].values())
        speed_range_seconds = list(results["speed_range_counts_seconds"].values())

        row_data = (
            zone_points
            + zone_percentages
            + center_border_data
            + time_data
            + speed_range_frames
            + speed_range_seconds
        )

        # Write to the CSV file
        write_header = not os.path.exists(combined_file_path)
        with open(combined_file_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(headers) + "\n")
            f.write(",".join(map(str, row_data)) + "\n")

        print(f"Results saved to: {combined_file_path}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        raise


def save_position_data(time_vector, x, y, distance, speed, output_dir, base_name):
    try:
        position_file_path = os.path.join(output_dir, f"{base_name}_position_data.csv")
        with open(position_file_path, "w", encoding="utf-8") as f:
            f.write("time_s,x_m,y_m,distance_m,speed_m/s\n")  # Cabeçalho
            for i in range(len(time_vector)):
                f.write(
                    f"{time_vector[i]:.6f},{x[i]:.6f},{y[i]:.6f},{distance[i]:.6f},{speed[i]:.6f}\n"
                )
        print(f"Position data saved to: {position_file_path}")
    except Exception as e:
        print(f"Error saving position data to CSV: {e}")
        raise


def process_open_field_data(input_file, main_output_dir, fs, cutoff):
    try:
        # Extract base name and create output directory
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(main_output_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Load and preprocess data
        x, y = load_and_preprocess_data(input_file)

        # Apply time vector
        time_vector = np.linspace(0, len(x) / fs, len(x))

        # Apply Butterworth low-pass filter with padding
        x_filtered = butter_lowpass_filter(
            x, cutoff=cutoff, fs=fs, order=4, padding=True
        )
        y_filtered = butter_lowpass_filter(
            y, cutoff=cutoff, fs=fs, order=4, padding=True
        )

        # Adjust x and y to stay within the bounds
        x_filtered, y_filtered = adjust_to_bounds(
            x_filtered, y_filtered, xmin=0, xmax=0.6, ymin=0, ymax=0.6
        )

        # Calculate kinematics
        (
            distance,
            speed,
            time_stationary,
            speed_range_counts_frames,
            speed_range_counts_seconds,
        ) = calculate_kinematics(x_filtered, y_filtered, fs)

        # Analyze zones and center/border occupancy
        zones, zone_percentages = calculate_zone_occupancy(x_filtered, y_filtered)
        center_border_results = calculate_center_and_border_occupancy(
            x_filtered, y_filtered
        )

        # Save results and data
        results = {
            "zone_counts": zones,
            "zone_percentages": zone_percentages,
            "time_stationary": time_stationary,
            "speed_range_counts_frames": speed_range_counts_frames,
            "speed_range_counts_seconds": speed_range_counts_seconds,
        }
        save_results_to_csv(results, center_border_results, fs, output_dir, base_name)
        save_position_data(
            time_vector, x_filtered, y_filtered, distance, speed, output_dir, base_name
        )

        # Generate visualizations
        plot_pathway(
            x_filtered, y_filtered, time_vector, sum(distance), output_dir, base_name
        )
        plot_heatmap(x_filtered, y_filtered, output_dir, base_name, results)
        plot_center_and_border_heatmap(
            x_filtered, y_filtered, output_dir, base_name, center_border_results
        )
        plot_speed_ranges(
            speed_range_counts_frames, time_stationary, fs, output_dir, base_name
        )
        plot_speed_over_time_with_tags(
            time_vector, speed, int(2 * fs), output_dir, base_name
        )

        print(f"Processing of file {input_file} completed successfully.")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {e}")
        raise


def process_all_files_in_directory(target_dir, fs, cutoff):
    """
    Processes all files in the selected directory, applying the specified
    sampling frequency and cutoff frequency for the Butterworth filter.

    Args:
        target_dir (str): Directory containing CSV files.
        fs (float): Sampling frequency in Hz.
        cutoff (float): Butterworth filter cutoff frequency in Hz.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(target_dir, f"openfield_results_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    csv_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".csv")
    ]
    for input_file in csv_files:
        print(f"Processing file: {input_file}")
        process_open_field_data(input_file, main_output_dir, fs, cutoff)
    print("All files have been processed successfully.")


def run_animal_open_field():
    """
    Run the open field analysis process, allowing the user to select the directory
    and input the sampling frequency and Butterworth cutoff frequency.
    """
    print("Running open field analysis...")
    root = Tk()
    root.withdraw()

    # Ask user to select the directory containing CSV files
    target_dir = filedialog.askdirectory(
        title="Select the directory containing .csv files of open field data"
    )
    if not target_dir:
        messagebox.showwarning("Warning", "No directory selected.")
        return

    # Ask user to input the sampling frequency (Hz)
    fs = simpledialog.askfloat(
        "Sampling Frequency", "Enter the sampling frequency (Hz):", minvalue=0.1
    )
    if not fs:
        messagebox.showwarning("Warning", "Sampling frequency not provided.")
        return

    # Ask user to input the Butterworth cutoff frequency (Hz)
    cutoff = simpledialog.askfloat(
        "Butterworth Cutoff Frequency",
        "Enter the cutoff frequency for the Butterworth filter (Hz):",
        minvalue=0.1,
    )
    if not cutoff:
        messagebox.showwarning("Warning", "Cutoff frequency not provided.")
        return

    # Process all files in the selected directory
    process_all_files_in_directory(target_dir, fs, cutoff)
    root.destroy()
    messagebox.showinfo(
        "Success", "All .csv files have been processed and results saved."
    )


if __name__ == "__main__":
    run_animal_open_field()
