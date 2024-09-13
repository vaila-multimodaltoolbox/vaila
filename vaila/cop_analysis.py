"""
Module: cop_analysis.py
Description: This module provides tools for analyzing Center of Pressure (CoP) data from force plate measurements.
It includes functions for data filtering, selecting relevant data headers, calculating confidence ellipses,
and plotting CoP pathways with confidence intervals.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.4
Date: 2024-09-12

Changelog:
- Version 1.4 (2024-09-12):
  - Integrated spectral features calculations.
  - Adjusted metrics dictionary to include new spectral features.
  - Standardized headers in the metrics dictionary for CSV output.
  - Updated plotting functions to include maximum PSD values and median frequencies.
  - Added print statements to provide user feedback during processing.

Usage:
- To run the CoP analysis, use the `main` function:

  if __name__ == "__main__":
      main()
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import (
    Tk, Toplevel, Canvas, Scrollbar, Frame, Button, Checkbutton,
    BooleanVar, messagebox, filedialog, simpledialog
)
from .ellipse import plot_ellipse_pca, plot_cop_pathway_with_ellipse
from .filter_utils import butter_filter
from .stabilogram_analysis import (
    compute_rms,
    compute_speed,
    compute_power_spectrum,
    compute_msd,
    count_zero_crossings,
    count_peaks,
    compute_sway_density,
    compute_total_path_length,
    plot_stabilogram,
    plot_power_spectrum,
    save_metrics_to_csv
)
from .spectral_features import (
    total_power,
    power_frequency_50,
    power_frequency_95,
    power_mode,
    centroid_frequency,
    frequency_dispersion,
    energy_content_below_0_5,
    energy_content_0_5_2,
    energy_content_above_2,
    frequency_quotient
)


def convert_to_cm(data, unit):
    """Converts the data to centimeters based on the provided unit."""
    conversion_factors = {
        "m": 100, "mm": 0.1, "ft": 30.48, "in": 2.54, "yd": 91.44, "cm": 1
    }
    if unit not in conversion_factors:
        raise ValueError(f"Unsupported unit '{unit}'. Please use m, mm, ft, in, yd, or cm.")
    return data * conversion_factors[unit]


def read_csv_full(filename):
    """Reads the full CSV file."""
    try:
        data = pd.read_csv(filename, delimiter=",")
        return data
    except Exception as e:
        raise Exception(f"Error reading the CSV file: {str(e)}")


def select_nine_headers(file_path):
    """Displays a GUI to select nine headers for force plate data analysis."""

    def get_csv_headers(file_path):
        """Reads the headers from a CSV file."""
        df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [
            header for header, var in zip(headers, header_vars) if var.get()
        ]
        if len(selected_headers) != 9:
            messagebox.showinfo("Info", "Please select exactly nine headers for analysis.")
            return
        selection_window.quit()
        selection_window.destroy()

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select Nine Headers for Force Plate Data")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.8)}"
    )

    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    header_vars = [BooleanVar() for _ in headers]
    num_columns = 8  # Number of columns for header labels

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, anchor="center")

    Button(btn_frame, text="Select All", command=select_all).pack(side="top", pady=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(side="top", pady=5)
    Button(btn_frame, text="Confirm", command=on_select).pack(side="top", pady=5)

    selection_window.mainloop()

    if len(selected_headers) != 9:
        messagebox.showinfo("Info", "Please select exactly nine headers for analysis.")
        return None, None

    selected_data = df[selected_headers]
    return selected_headers, selected_data


def analyze_data_2d(data, output_dir, file_name, fs, plate_width, plate_height, timestamp):
    """Analyzes selected 2D data and saves the results."""

    print(f"Starting analysis for file: {file_name}")

    # Keep the original data in 'data'
    # Apply the Butterworth filter to the data, creating 'dataf'
    try:
        print("Applying Butterworth filter...")
        dataf = butter_filter(data, fs, filter_type='low', cutoff=10, order=4, padding=True)
    except ValueError as e:
        print(f"Filtering error: {e}")
        messagebox.showerror("Filtering Error", f"An error occurred during filtering:\n{e}")
        return

    # Extract variables from filtered data 'dataf'
    cop_x_f = dataf[:, 6]  # Filtered CoP in the X direction (ML)
    cop_y_f = dataf[:, 7]  # Filtered CoP in the Y direction (AP)

    # Create time vector based on the sampling frequency
    N = len(cop_x_f)
    T = N / fs  # Total duration
    time = np.linspace(0, T, N)

    # Calculate mean values of ML and AP coordinates
    mean_ML = np.mean(cop_x_f)
    mean_AP = np.mean(cop_y_f)

    # Centered coordinates
    X_n = cop_x_f - mean_ML
    Y_n = cop_y_f - mean_AP

    # Radius
    R_n = np.sqrt(X_n ** 2 + Y_n ** 2)

    # Covariance
    COV = np.mean(X_n * Y_n)

    # Velocity components
    V_xn = np.gradient(X_n, time)
    V_yn = np.gradient(Y_n, time)
    V_n = np.sqrt(V_xn ** 2 + V_yn ** 2)

    # Mean speed
    mean_speed_ml = np.mean(np.abs(V_xn))
    mean_speed_ap = np.mean(np.abs(V_yn))
    mean_velocity_norm = np.mean(V_n)

    # Compute power spectrum
    print("Computing power spectrum...")
    freqs_ml, psd_ml, freqs_ap, psd_ap = compute_power_spectrum(X_n, Y_n, fs)

    # Compute spectral features
    print("Computing spectral features...")
    # ML direction
    total_power_ml = total_power(freqs_ml, psd_ml)
    power_freq_50_ml = power_frequency_50(freqs_ml, psd_ml)
    power_freq_95_ml = power_frequency_95(freqs_ml, psd_ml)
    power_mode_ml = power_mode(freqs_ml, psd_ml)
    centroid_freq_ml = centroid_frequency(freqs_ml, psd_ml)
    freq_dispersion_ml = frequency_dispersion(freqs_ml, psd_ml)
    energy_below_0_5_ml = energy_content_below_0_5(freqs_ml, psd_ml)
    energy_0_5_2_ml = energy_content_0_5_2(freqs_ml, psd_ml)
    energy_above_2_ml = energy_content_above_2(freqs_ml, psd_ml)
    freq_quotient_ml = frequency_quotient(freqs_ml, psd_ml)

    # AP direction
    total_power_ap = total_power(freqs_ap, psd_ap)
    power_freq_50_ap = power_frequency_50(freqs_ap, psd_ap)
    power_freq_95_ap = power_frequency_95(freqs_ap, psd_ap)
    power_mode_ap = power_mode(freqs_ap, psd_ap)
    centroid_freq_ap = centroid_frequency(freqs_ap, psd_ap)
    freq_dispersion_ap = frequency_dispersion(freqs_ap, psd_ap)
    energy_below_0_5_ap = energy_content_below_0_5(freqs_ap, psd_ap)
    energy_0_5_2_ap = energy_content_0_5_2(freqs_ap, psd_ap)
    energy_above_2_ap = energy_content_above_2(freqs_ap, psd_ap)
    freq_quotient_ap = frequency_quotient(freqs_ap, psd_ap)

    # Compute MSD for a time interval Δt
    delta_t = 0.1  # Adjust as necessary
    print(f"Computing MSD with delta_t = {delta_t} seconds...")
    msd_ml = compute_msd(X_n, fs, delta_t)
    msd_ap = compute_msd(Y_n, fs, delta_t)

    # Count zero-crossings
    zero_crossings_ml = count_zero_crossings(X_n)
    zero_crossings_ap = count_zero_crossings(Y_n)

    # Count peaks
    num_peaks_ml = count_peaks(X_n)
    num_peaks_ap = count_peaks(Y_n)

    # Compute RMS using centered data
    rms_ml, rms_ap = compute_rms(X_n, Y_n)

    # Compute total path length
    print("Calculating total path length...")
    total_path_length = compute_total_path_length(cop_x_f, cop_y_f)

    # Compute sway density using centered data
    print("Computing sway density...")
    sway_density_ml = compute_sway_density(X_n, fs, radius=0.3)
    sway_density_ap = compute_sway_density(Y_n, fs, radius=0.3)

    # Update metrics dictionary
    metrics = {
        'Total Duration (s)': T,
        'Number of Points': N,
        'Sampling Frequency (Hz)': fs,
        'Mean ML (cm)': mean_ML,
        'Mean AP (cm)': mean_AP,
        'Min ML (cm)': np.min(X_n),
        'Max ML (cm)': np.max(X_n),
        'Min AP (cm)': np.min(Y_n),
        'Max AP (cm)': np.max(Y_n),
        'RMS ML (cm)': rms_ml,
        'RMS AP (cm)': rms_ap,
        'Covariance (cm²)': COV,
        'Total Path Length (cm)': total_path_length,
        'Mean Speed ML (cm/s)': mean_speed_ml,
        'Mean Speed AP (cm/s)': mean_speed_ap,
        'Mean Velocity Norm (cm/s)': mean_velocity_norm,
        'MSD ML (cm²)': msd_ml,
        'MSD AP (cm²)': msd_ap,
        'Zero Crossings ML': zero_crossings_ml,
        'Zero Crossings AP': zero_crossings_ap,
        'Number of Peaks ML': num_peaks_ml,
        'Number of Peaks AP': num_peaks_ap,
        'Total Power ML': total_power_ml,
        'Total Power AP': total_power_ap,
        'Power Frequency 50 ML': power_freq_50_ml,
        'Power Frequency 50 AP': power_freq_50_ap,
        'Power Frequency 95 ML': power_freq_95_ml,
        'Power Frequency 95 AP': power_freq_95_ap,
        'Power Mode ML': power_mode_ml,
        'Power Mode AP': power_mode_ap,
        'Centroid Frequency ML': centroid_freq_ml,
        'Centroid Frequency AP': centroid_freq_ap,
        'Frequency Dispersion ML': freq_dispersion_ml,
        'Frequency Dispersion AP': freq_dispersion_ap,
        'Energy Content Below 0.5 ML': energy_below_0_5_ml,
        'Energy Content Below 0.5 AP': energy_below_0_5_ap,
        'Energy Content 0.5-2 ML': energy_0_5_2_ml,
        'Energy Content 0.5-2 AP': energy_0_5_2_ap,
        'Energy Content Above 2 ML': energy_above_2_ml,
        'Energy Content Above 2 AP': energy_above_2_ap,
        'Frequency Quotient ML': freq_quotient_ml,
        'Frequency Quotient AP': freq_quotient_ap,
    }

    # Define output path to save files
    output_path = os.path.join(output_dir, f"{file_name}_cop_analysis_{timestamp}")

    # Save metrics to CSV
    print("Saving metrics to CSV...")
    save_metrics_to_csv(metrics, output_path)

    # Plot and save stabilogram using centered data and time vector
    print("Plotting stabilogram...")
    plot_stabilogram(time, X_n, Y_n, output_path)

    # Plot and save power spectrum
    print("Plotting power spectrum...")
    plot_power_spectrum(freqs_ml, psd_ml, freqs_ap, psd_ap, output_path)

    # Calculate and plot confidence ellipse
    print("Calculating and plotting confidence ellipse...")
    area, angle, bounds, ellipse_data = plot_ellipse_pca(np.column_stack((X_n, Y_n)), confidence=0.95)
    plot_cop_pathway_with_ellipse(X_n, Y_n, area, angle, ellipse_data, file_name, output_path)

    print(f"Analysis complete for file: {file_name}\n")


def main():
    """Function to run the CoP analysis"""
    root = Tk()
    root.withdraw()  # Hides the main Tkinter window

    print("Starting CoP analysis...")

    # Request input and output directories
    input_dir = filedialog.askdirectory(title="Select Input Directory")
    if not input_dir:
        print("No input directory selected.")
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_dir:
        print("No output directory selected.")
        return

    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    # Request sampling frequency and plate dimensions
    fs = simpledialog.askfloat(
        "Signal Frequency",
        "Enter the sampling frequency (Fs) in Hz:",
        initialvalue=1000.0,
    )
    if not fs:
        print("No valid frequency provided.")
        return

    plate_width = simpledialog.askfloat(
        "Force Plate Width",
        "Enter the width of the force plate in cm:",
        initialvalue=46.4,
    )
    plate_height = simpledialog.askfloat(
        "Force Plate Height",
        "Enter the height of the force plate in cm:",
        initialvalue=50.75,
    )

    if not plate_width or not plate_height:
        print("Invalid force plate dimensions provided.")
        return

    # Ask user for the unit of measurement
    unit = simpledialog.askstring(
        "Unit of Measurement",
        "Enter the unit of measurement for the CoP data (e.g., cm, m, mm, ft, in, yd):",
        initialvalue="mm"
    )
    if not unit:
        print("No unit provided.")
        return

    print(f"Sampling Frequency: {fs} Hz")
    print(f"Force Plate Dimensions: {plate_width} cm x {plate_height} cm")
    print(f"Unit of Measurement: {unit}")

    # Select sample file
    sample_file_path = filedialog.askopenfilename(
        title="Select a Sample CSV File", filetypes=[("CSV files", "*.csv")]
    )
    if not sample_file_path:
        print("No sample file selected.")
        return

    print(f"Sample file selected: {sample_file_path}")

    # Select nine headers for force plate data analysis
    selected_headers, _ = select_nine_headers(sample_file_path)
    if not selected_headers:
        print("No valid headers selected.")
        return

    print(f"Selected Headers: {selected_headers}")

    # Create timestamp for output directory
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Create main output directory using timestamp
    main_output_dir = os.path.join(output_dir, f"cop_balance_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    print(f"Main output directory created: {main_output_dir}")

    # Process each CSV file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            print(f"Processing file: {file_name}")
            file_path = os.path.join(input_dir, file_name)
            df_full = read_csv_full(file_path)
            if not all(header in df_full.columns for header in selected_headers):
                messagebox.showerror("Header Error", f"Selected headers not found in file {file_name}.")
                print(f"Error: Selected headers not found in file {file_name}. Skipping file.")
                continue
            data = df_full[selected_headers].to_numpy()

            # Convert data to cm if necessary
            try:
                data = convert_to_cm(data, unit)
            except ValueError as e:
                print(e)
                messagebox.showerror("Unit Conversion Error", f"An error occurred during unit conversion:\n{e}")
                print(f"Error converting units for file {file_name}. Skipping file.")
                continue

            # Create output directory for the current file without the '.csv' extension
            file_name_without_extension = os.path.splitext(file_name)[0]
            file_output_dir = os.path.join(main_output_dir, file_name_without_extension)
            os.makedirs(file_output_dir, exist_ok=True)

            # Analyze and save results
            analyze_data_2d(
                data,
                file_output_dir,
                file_name_without_extension,  # Use file name without extension
                fs,
                plate_width,
                plate_height,
                timestamp,
            )
        else:
            print(f"Skipping non-CSV file: {file_name}")

    # Inform the user that the analysis is complete
    print("All files processed.")
    messagebox.showinfo("Information", "Analysis complete! The window will close in 10 seconds.")
    root.after(10000, root.destroy)  # Wait for 10 seconds and then destroy the window
    root.mainloop()


if __name__ == "__main__":
    main()

