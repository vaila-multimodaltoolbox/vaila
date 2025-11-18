"""
# Plot the CoP pathway with points only (no connecting lines)
ax2.plot(X_n, Y_n, label='CoP Pathway', color='blue', marker='.', markersize=3, linestyle='None')
Module: cop_analysis.py
Description: This module provides a comprehensive set of tools for analyzing Center of Pressure (CoP) data from force plate measurements.
             CoP data is critical in understanding balance and postural control in various fields such as biomechanics, rehabilitation, and sports science.

             The module includes:
             - Functions for data filtering, specifically using a Butterworth filter, to remove noise and smooth the CoP data.
             - GUI components to allow users to select relevant data headers interactively for analysis.
             - Methods to calculate various postural stability metrics such as Mean Square Displacement (MSD), Power Spectral Density (PSD), and sway density.
             - Spectral feature analysis to quantify different frequency components of the CoP signals, which is essential for evaluating balance strategies and postural control.
             - Plotting functions to visualize CoP pathways and their associated confidence ellipses, stabilograms, and power spectrums.

             The module is designed to handle large datasets efficiently, allowing batch processing of multiple files and providing detailed feedback throughout the analysis.

Inputs:
- `convert_to_cm(data, unit)`:
  - `data` (numpy array): Numerical data to be converted.
  - `unit` (str): The current unit of the data (e.g., 'm', 'mm', 'ft', 'in', 'yd', 'cm').

- `read_csv_full(filename)`:
  - `filename` (str): The path to the CSV file containing CoP data.

- `select2headers(file_path)`:
  - `file_path` (str): Path to a sample CSV file to determine the headers available for analysis.

- `analyze_data_2d(data, output_dir, file_name, fs, plate_width, plate_height, timestamp)`:
  - `data` (numpy array): Filtered CoP data for analysis.
  - `output_dir` (str): Directory path to save analysis outputs.
  - `file_name` (str): Base name for output files.
  - `fs` (float): Sampling frequency in Hz.
  - `plate_width` (float): Width of the force plate in centimeters.
  - `plate_height` (float): Height of the force plate in centimeters.
  - `timestamp` (str): Timestamp for output file naming.

Outputs:
- Saves processed data and metrics in CSV format to the specified output directory.
- Generates and saves visualizations of CoP pathways, stabilograms, power spectrums, and confidence ellipses in PNG and SVG formats.

Usage:
- To use this module for CoP data analysis, execute the `main` function. The script will prompt the user to select input and output directories, sampling frequency, force plate dimensions, and unit of measurement for the CoP data.

Example:
```python
if __name__ == "__main__":
    main()
```
    Example flow of using the functions in this module:
        Data Import and Preparation: Use read_csv_full to load CoP data from a CSV file and convert_to_cm to ensure the data is in the desired unit.
        Header Selection: Use select2headers to allow the user to select which headers from the CSV file should be analyzed.
        Data Analysis: Use analyze_data_2d to filter the CoP data, compute various metrics, perform spectral feature analysis, and generate visualizations.
        Batch Processing: The main function orchestrates the entire process, handling multiple CSV files in a selected directory.

Author: Prof. Dr. Paulo R. P. Santiago Version: 1.4 Date: 2024-09-12

Changelog:

    Version 1.4 (2024-09-12):
        Integrated spectral features calculations for enhanced analysis of CoP data.
        Adjusted metrics dictionary to include new spectral features such as total power, frequency dispersion, and energy content at various frequency bands.
        Standardized headers in the metrics dictionary for consistent CSV output.
        Updated plotting functions to display maximum PSD values and median frequencies for improved visual analysis.
        Enhanced user feedback with print statements during various processing stages.

References:

    GitHub Repository: Code Descriptors Postural Control. https://github.com/Jythen/code_descriptors_postural_control
    Further reading on entropy-based methods in postural control: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8623280

"""

import os
from tkinter import (
    BooleanVar,
    Button,
    Canvas,
    Checkbutton,
    Frame,
    Scrollbar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
    simpledialog,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ellipse import plot_cop_pathway_with_ellipse, plot_ellipse_pca
from .filter_utils import butter_filter
from .spectral_features import (
    centroid_frequency,
    energy_content_0_5_2,
    energy_content_above_2,
    energy_content_below_0_5,
    frequency_dispersion,
    frequency_quotient,
    power_frequency_50,
    power_frequency_95,
    power_mode,
    total_power,
)
from .stabilogram_analysis import (
    compute_msd,
    compute_power_spectrum,
    compute_rms,
    compute_sway_density,
    compute_total_path_length,
    count_peaks,
    count_zero_crossings,
    plot_power_spectrum,
    plot_stabilogram,
    save_metrics_to_csv,
)


def convert_to_cm(data, unit):
    """Converts the data to centimeters based on the provided unit."""
    conversion_factors = {
        "m": 100,
        "mm": 0.1,
        "ft": 30.48,
        "in": 2.54,
        "yd": 91.44,
        "cm": 1,
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


def select2headers(file_path):
    """Displays a GUI to select two (2) headers for force plate data analysis."""

    def get_csv_headers(file_path):
        """Reads the headers from a CSV file."""
        df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [header for header, var in zip(headers, header_vars) if var.get()]
        if len(selected_headers) != 2:
            messagebox.showinfo("Info", "Please select exactly two (2) headers for analysis.")
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
    selection_window.title("Select two (2) Cx and Cy components from Headers for Force Plate Data")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight() * 0.8)}"
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

    if len(selected_headers) != 2:
        messagebox.showinfo("Info", "Please select exactly 2 headers for analysis.")
        return None, None

    selected_data = df[selected_headers]
    return selected_headers, selected_data


def plot_final_figure(
    time,
    X_n,
    Y_n,
    freqs_ml,
    psd_ml,
    freqs_ap,
    psd_ap,
    metrics,
    output_path,
    area,
    angle,
    bounds,
    ellipse_data,
):
    """
    Creates the final figure with the stabilogram, CoP pathway, and a text with all the presented result variables.
    """
    fig = plt.figure(figsize=(12, 8))

    # Subplot for the stabilogram (row 1, column 1)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time, X_n, label="CoP ML")
    ax1.plot(time, Y_n, label="CoP AP")
    ax1.grid(color="gray", linestyle=":", linewidth=0.5)
    ax1.set_title("Stabilogram")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Displacement (cm)")
    ax1.legend()

    # Subplot for the CoP pathway with confidence ellipse (row 2, column 1)
    ax2 = fig.add_subplot(2, 2, 3)

    # Plot the CoP pathway with points only (no connecting lines)
    ax2.plot(
        X_n,
        Y_n,
        label="CoP Pathway",
        color="blue",
        marker=".",
        markersize=3,
        linestyle="None",
    )

    # Unpack the ellipse data to plot it correctly
    ellipse_x, ellipse_y = ellipse_data[0], ellipse_data[1]
    eigvecs, scaled_eigvals, pca_mean = (
        ellipse_data[2],
        ellipse_data[3],
        ellipse_data[4],
    )

    # Plot the confidence ellipse
    ax2.plot(
        ellipse_x,
        ellipse_y,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Confidence Ellipse",
    )

    # Plot major and minor axes of the ellipse
    major_axis_start = pca_mean - eigvecs[0] * scaled_eigvals[0]
    major_axis_end = pca_mean + eigvecs[0] * scaled_eigvals[0]
    ax2.plot(
        [major_axis_start[0], major_axis_end[0]],
        [major_axis_start[1], major_axis_end[1]],
        color="grey",
        linestyle=":",
        linewidth=1.7,
        label="Major Axis",
    )

    minor_axis_start = pca_mean - eigvecs[1] * scaled_eigvals[1]
    minor_axis_end = pca_mean + eigvecs[1] * scaled_eigvals[1]
    ax2.plot(
        [minor_axis_start[0], minor_axis_end[0]],
        [minor_axis_start[1], minor_axis_end[1]],
        color="grey",
        linestyle=":",
        linewidth=1.5,
        label="Minor Axis",
    )

    # Set the title and labels
    ax2.set_title("CoP Pathway with Confidence Ellipse")
    ax2.set_xlabel("CoP ML (cm)")
    ax2.set_ylabel("CoP AP (cm)")

    # Set the aspect ratio to equal to ensure equal proportions
    ax2.set_aspect("equal", adjustable="box")

    # Adjust the limits of the plot to ensure both the CoP pathway and ellipse are visible
    x_margin = 0.02 * (
        np.max([np.max(ellipse_x), np.max(X_n)]) - np.min([np.min(ellipse_x), np.min(X_n)])
    )
    y_margin = 0.02 * (
        np.max([np.max(ellipse_y), np.max(Y_n)]) - np.min([np.min(ellipse_y), np.min(Y_n)])
    )

    ax2.set_xlim(
        min(np.min(ellipse_x), np.min(X_n)) - x_margin,
        max(np.max(ellipse_x), np.max(X_n)) + x_margin,
    )
    ax2.set_ylim(
        min(np.min(ellipse_y), np.min(Y_n)) - y_margin,
        max(np.max(ellipse_y), np.max(Y_n)) + y_margin,
    )

    # Subplot for result variables (combined column 2)
    ax3 = fig.add_subplot(1, 2, 2)  # Use a single subplot that spans both rows in the second column
    ax3.axis("off")  # Hide axes to focus on the text
    text_str = "\n".join(
        [f"{key}: {value}" for key, value in metrics.items()]
    )  # Prepare text from metrics dictionary
    ax3.text(
        0.05,
        0.5,
        text_str,
        fontsize=10,
        verticalalignment="center",
        transform=ax3.transAxes,
        wrap=True,
    )  # Display text

    # Save the figure in PNG and SVG formats
    plt.tight_layout()
    plt.savefig(f"{output_path}_final_figure.png", dpi=300, format="png")
    plt.savefig(f"{output_path}_final_figure.svg", format="svg")
    plt.close()


def analyze_data_2d(data, output_dir, file_name, fs, plate_width, plate_height, timestamp):
    """Analyzes selected 2D data and saves the results."""
    print(f"Starting analysis for file: {file_name}")

    # Apply the Butterworth filter to the data
    try:
        print("Applying Butterworth filter...")
        dataf = butter_filter(data, fs, filter_type="low", cutoff=10, order=4, padding=True)
    except ValueError as e:
        print(f"Filtering error: {e}")
        return

    # Extract filtered CoP data
    cop_x_f = dataf[:, 0]  # Filtered CoP in the X direction (ML)
    cop_y_f = dataf[:, 1]  # Filtered CoP in the Y direction (AP)

    # Create time vector
    N = len(cop_x_f)
    T = N / fs
    time = np.linspace(0, (len(cop_x_f) - 1) / fs, len(cop_x_f))

    # Mean values and centered coordinates
    mean_ML = np.mean(cop_x_f)
    mean_AP = np.mean(cop_y_f)
    X_n = cop_x_f - mean_ML
    Y_n = cop_y_f - mean_AP

    # Radius and covariance
    R_n = np.sqrt(X_n**2 + Y_n**2)
    COV = np.mean(X_n * Y_n)

    # Velocity components
    V_xn = np.gradient(X_n, time)
    V_yn = np.gradient(Y_n, time)
    V_n = np.sqrt(V_xn**2 + V_yn**2)

    # Mean speed
    mean_speed_ml = np.mean(np.abs(V_xn))
    mean_speed_ap = np.mean(np.abs(V_yn))
    mean_velocity_norm = np.mean(V_n)

    # Compute power spectrum
    print("Computing power spectrum...")
    freqs_ml, psd_ml, freqs_ap, psd_ap = compute_power_spectrum(X_n, Y_n, fs)

    # Compute spectral features
    print("Computing spectral features...")
    total_power_ml = total_power(freqs_ml, psd_ml)
    total_power_ap = total_power(freqs_ap, psd_ap)
    power_freq_50_ml = power_frequency_50(freqs_ml, psd_ml)
    power_freq_50_ap = power_frequency_50(freqs_ap, psd_ap)
    power_freq_95_ml = power_frequency_95(freqs_ml, psd_ml)
    power_freq_95_ap = power_frequency_95(freqs_ap, psd_ap)
    power_mode_ml = power_mode(freqs_ml, psd_ml)
    power_mode_ap = power_mode(freqs_ap, psd_ap)
    centroid_freq_ml = centroid_frequency(freqs_ml, psd_ml)
    centroid_freq_ap = centroid_frequency(freqs_ap, psd_ap)
    freq_dispersion_ml = frequency_dispersion(freqs_ml, psd_ml)
    freq_dispersion_ap = frequency_dispersion(freqs_ap, psd_ap)
    energy_below_0_5_ml = energy_content_below_0_5(freqs_ml, psd_ml)
    energy_below_0_5_ap = energy_content_below_0_5(freqs_ap, psd_ap)
    energy_0_5_2_ml = energy_content_0_5_2(freqs_ml, psd_ml)
    energy_0_5_2_ap = energy_content_0_5_2(freqs_ap, psd_ap)
    energy_above_2_ml = energy_content_above_2(freqs_ml, psd_ml)
    energy_above_2_ap = energy_content_above_2(freqs_ap, psd_ap)
    freq_quotient_ml = frequency_quotient(freqs_ml, psd_ml)
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

    # Calculate and plot confidence ellipse
    print("Calculating confidence ellipse...")
    area, angle, bounds, ellipse_data = plot_ellipse_pca(
        np.column_stack((X_n, Y_n)), confidence=0.95
    )

    # Update metrics dictionary with all required variables
    metrics = {
        "Total_Duration_s": T,
        "Number_of_Points": N,
        "Sampling_Frequency_Hz": fs,
        "Mean_ML_cm": mean_ML,
        "Mean_AP_cm": mean_AP,
        "Min_ML_cm": np.min(X_n),
        "Max_ML_cm": np.max(X_n),
        "Min_AP_cm": np.min(Y_n),
        "Max_AP_cm": np.max(Y_n),
        "RMS_ML_cm": rms_ml,
        "RMS_AP_cm": rms_ap,
        "Covariance_cm2": COV,
        "Total_Path_Length_cm": total_path_length,
        "Mean_Speed_ML_cmps": mean_speed_ml,
        "Mean_Speed_AP_cmps": mean_speed_ap,
        "Mean_Velocity_Norm_cmps": mean_velocity_norm,
        "Sway_Area_cm2": area,
        "Ellipse_Angle_degrees": angle,
        "MSD_ML_cm2": msd_ml,
        "MSD_AP_cm2": msd_ap,
        "Zero_Crossings_ML": zero_crossings_ml,
        "Zero_Crossings_AP": zero_crossings_ap,
        "Number_of_Peaks_ML": num_peaks_ml,
        "Number_of_Peaks_AP": num_peaks_ap,
        "Total_Power_ML": total_power_ml,
        "Total_Power_AP": total_power_ap,
        "Power_Frequency_50_ML": power_freq_50_ml,
        "Power_Frequency_50_AP": power_freq_50_ap,
        "Power_Frequency_95_ML": power_freq_95_ml,
        "Power_Frequency_95_AP": power_freq_95_ap,
        "Power_Mode_ML": power_mode_ml,
        "Power_Mode_AP": power_mode_ap,
        "Centroid_Frequency_ML": centroid_frequency(freqs_ml, psd_ml),
        "Centroid_Frequency_AP": centroid_frequency(freqs_ap, psd_ap),
        "Frequency_Dispersion_ML": freq_dispersion_ml,
        "Frequency_Dispersion_AP": freq_dispersion_ap,
        "Energy_Content_Below_0.5_ML": energy_below_0_5_ml,
        "Energy_Content_Below_0.5_AP": energy_below_0_5_ap,
        "Energy_Content_0.5_2_ML": energy_0_5_2_ml,
        "Energy_Content_0.5_2_AP": energy_0_5_2_ap,
        "Energy_Content_Above_2_ML": energy_above_2_ml,
        "Energy_Content_Above_2_AP": energy_above_2_ap,
        "Frequency_Quotient_ML": freq_quotient_ml,
        "Frequency_Quotient_AP": freq_quotient_ap,
    }

    # Save metrics to CSV
    print("Saving metrics to CSV...")
    metrics_file = os.path.join(output_dir, f"{file_name}_metrics.csv")
    save_metrics_to_csv(metrics, metrics_file)

    # Plot and save stabilogram
    print("Plotting stabilogram...")
    stabilogram_file = os.path.join(output_dir, f"{file_name}_stabilogram.png")
    plot_stabilogram(time, X_n, Y_n, stabilogram_file)

    # Plot power spectrum
    print("Plotting power spectrum...")
    power_spectrum_file = os.path.join(output_dir, f"{file_name}_power_spectrum.png")
    plot_power_spectrum(freqs_ml, psd_ml, freqs_ap, psd_ap, power_spectrum_file)

    # Plot CoP pathway with confidence ellipse
    print("Plotting CoP pathway with ellipse...")
    cop_pathway_file = os.path.join(output_dir, f"{file_name}_cop_pathway.png")
    plot_cop_pathway_with_ellipse(X_n, Y_n, area, angle, ellipse_data, file_name, cop_pathway_file)

    # Create final figure
    print("Creating final figure...")
    final_figure_file = os.path.join(output_dir, f"{file_name}_final_figure.png")
    plot_final_figure(
        time,
        X_n,
        Y_n,
        freqs_ml,
        psd_ml,
        freqs_ap,
        psd_ap,
        metrics,
        final_figure_file,
        area,
        angle,
        bounds,
        ellipse_data,
    )

    print(f"Analysis completed for file: {file_name}")


def main():
    """Function to run the CoP analysis"""
    root = Tk()
    root.withdraw()  # Hides the main Tkinter window

    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
        initialvalue="mm",
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

    # Select two (2) headers for force plate data analysis
    selected_headers, _ = select2headers(sample_file_path)
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
                messagebox.showerror(
                    "Header Error", f"Selected headers not found in file {file_name}."
                )
                print(f"Error: Selected headers not found in file {file_name}. Skipping file.")
                continue
            data = df_full[selected_headers].to_numpy()

            # Convert data to cm if necessary
            try:
                data = convert_to_cm(data, unit)
            except ValueError as e:
                print(e)
                messagebox.showerror(
                    "Unit Conversion Error",
                    f"An error occurred during unit conversion:\n{e}",
                )
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
    messagebox.showinfo("Information", "Analysis complete! Close this window.")
    root.mainloop()


if __name__ == "__main__":
    main()
