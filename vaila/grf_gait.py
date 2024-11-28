"""
================================================================================
Gait Data Analysis Toolkit - grf_gait.py
================================================================================
Author: [Your Name]
Date: [Current Date]
Version: 1.0
Python Version: 3.x

Description:
------------
This script processes gait data from force platforms, analyzing a single
contact strike to compute key metrics, including:
- Peak forces, impulse, rate of force development (RFD), and contact time.

The results are visualized through interactive plots and saved to CSV files for
further analysis.

Key Functionalities:
---------------------
1. Data Selection:
   - Allows the user to select an input CSV file containing gait data.
   - Prompts the user to specify output directories.
   - Prompts the user for input parameters (sampling frequency, thresholds, etc.).
2. Data Processing:
   - Normalizes data, applies Butterworth filters, and computes key biomechanical metrics.
3. Visualization:
   - Generates and saves plots for force-time curves with relevant markers and highlighted regions.
4. Output:
   - Saves results to a CSV file and generates plots for the analyzed contact strike.

Input:
------
- CSV File:
   The CSV file should contain force data recorded from a force platform during gait.
   The file must include a column for vertical ground reaction force (VGRF).

   Example format:
   Sample, GRF (N)
   0, 50.25
   1, 51.60
   2, 49.80
   ...

- User Input:
   - Sampling frequency (Fs in Hz).
   - Threshold for activity detection.

Output:
-------
- CSV File:
   A CSV file containing results for key metrics such as:
   * Peak force
   * Impulse
   * Rate of force development (RFD)
   * Contact time

- Plot Files:
   PNG and SVG plots of force-time curves, highlighting key events.

How to Run:
-----------
1. Ensure required dependencies are installed:
   pip install numpy pandas matplotlib scipy
2. Run the script:
   python grf_gait.py
3. Follow on-screen prompts to select data and define parameters.

License:
--------
This script is provided "as is," without warranty. It is intended for academic 
and research purposes only.

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from tkinter import Tk, filedialog, simpledialog, messagebox
import os


def select_input_file():
    """
    Opens a file dialog to select the input CSV file containing GRF data.
    """
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Input CSV File", filetypes=[("CSV Files", "*.csv")]
    )
    root.destroy()
    return file_path


def select_output_directory():
    """
    Opens a dialog to select the output directory for saving results.
    """
    root = Tk()
    root.withdraw()
    output_dir = filedialog.askdirectory(title="Select Output Directory for Results")
    root.destroy()
    return output_dir


def load_csv_file(file_path):
    """
    Loads the CSV file and returns a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None


def butterworth_filter(data, cutoff, fs):
    """
    Applies a Butterworth low-pass filter to the data.
    """
    nyquist = fs / 2
    b, a = butter(4, cutoff / nyquist, btype="low")
    return filtfilt(b, a, data)


def prompt_user_input():
    """
    Prompts the user to input sampling frequency and threshold.
    """
    root = Tk()
    root.withdraw()

    fs = simpledialog.askfloat(
        "Input", "Enter Sampling Frequency (Hz):", initialvalue=1000.0
    )
    threshold = simpledialog.askfloat(
        "Input", "Enter Threshold (N):", initialvalue=50.0
    )

    root.destroy()
    return fs, threshold


def calculate_metrics(data, fs):
    """
    Calculate key metrics from the GRF data for a single contact strike.
    """
    time = np.linspace(0, total_time, num_samples)  # Time vector
    total_time = time[-1]
    peak_force = np.max(data)
    time_to_peak = time[np.argmax(data)]
    impulse = np.trapz(data, time)
    rfd = peak_force / time_to_peak if time_to_peak > 0 else np.nan
    contact_time = total_time

    metrics = {
        "Peak Force (N)": peak_force,
        "Time to Peak Force (s)": time_to_peak,
        "Impulse (N·s)": impulse,
        "Rate of Force Development (N/s)": rfd,
        "Contact Time (s)": contact_time,
    }
    return metrics


def plot_results(time, data, metrics, output_dir, file_name):
    """
    Plots the GRF data and highlights key metrics.
    """
    fig, ax = plt.subplots()
    ax.plot(time, data, label="GRF (N)", linewidth=2)
    ax.axhline(
        metrics["Peak Force (N)"],
        color="r",
        linestyle="--",
        label=f"Peak Force: {metrics['Peak Force (N)']:.2f} N",
    )
    ax.axvline(
        metrics["Time to Peak Force (s)"],
        color="g",
        linestyle="--",
        label=f"Time to Peak: {metrics['Time to Peak Force (s)']:.2f} s",
    )
    ax.set_title(f"Gait GRF Analysis - {file_name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.legend()
    ax.grid()

    # Save the plot
    plot_png = os.path.join(output_dir, f"{file_name}_plot.png")
    plot_svg = os.path.join(output_dir, f"{file_name}_plot.svg")
    plt.savefig(plot_png)
    plt.savefig(plot_svg)
    plt.show()


def save_results(metrics, output_dir, file_name):
    """
    Saves the calculated metrics to a CSV file.
    """
    output_file = os.path.join(output_dir, f"{file_name}_results.csv")
    pd.DataFrame([metrics]).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    """
    Main function to process GRF data for a single contact strike.
    """
    file_path = select_input_file()
    if not file_path:
        print("No file selected.")
        return

    output_dir = select_output_directory()
    if not output_dir:
        print("No output directory selected.")
        return

    df = load_csv_file(file_path)
    if df is None or df.empty:
        print("Failed to load data.")
        return

    column_name = simpledialog.askstring(
        "Column Selection", "Enter the column name for GRF data:"
    )
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the file.")
        return

    grf_data = df[column_name].to_numpy()
    fs, threshold = prompt_user_input()

    # Filter data
    filtered_data = butterworth_filter(grf_data, cutoff=20, fs=fs)

    # Identify contact region
    contact_region = filtered_data > threshold
    contact_data = filtered_data[contact_region]

    if len(contact_data) == 0:
        print("No contact data detected above the threshold.")
        return

    # Time vector for the contact region
    contact_time = np.linspace(0, len(contact_data) / fs, len(contact_data))

    # Calculate metrics
    metrics = calculate_metrics(contact_data, fs)

    # Save and plot results
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_results(metrics, output_dir, file_name)
    plot_results(contact_time, contact_data, metrics, output_dir, file_name)


if __name__ == "__main__":
    main()
