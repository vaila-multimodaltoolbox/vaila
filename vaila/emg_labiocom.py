"""
================================================================================
EMG Analysis Toolkit - emg_labiocom
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 23 September 2024
Version: 1.1
Python Version: 3.11.9

Description:
------------
This toolkit provides advanced functionalities to analyze EMG (Electromyography) signals. 
It includes filtering, full-wave rectification, linear envelope calculation, RMS calculation, 
and median frequency analysis. The toolkit now supports recursive processing of all CSV files 
within a selected directory for batch analysis.

Key Functionalities:
---------------------
1. Butterworth Filtering: Lowpass and Bandpass filtering to clean EMG data.
2. Full-Wave Rectification: Rectifies EMG signals to prepare for envelope extraction.
3. Linear Envelope Calculation: Computes a smoothed signal for muscle activity analysis.
4. RMS and Median Frequency Calculation: Calculates Root Mean Square (RMS) and median frequency.
5. Welch's Power Spectral Density (PSD) Analysis: Analyzes frequency components of the EMG signals.
6. Batch Processing: Recursive processing of all CSV files in a directory.

Input:
------
- CSV files: The script reads CSV or text files containing raw EMG data. 
  The first column is assumed to be sample numbers.
  The second column is the EMG signal (in µVolts).

Example of Input Format (CSV):
------------------------------
Sample, EMG (µVolts)
0, 0.001
1, 0.002
2, 0.003
...

Output:
-------
For each processed file, the following outputs are generated:
1. Processed Data (CSV):
    - CSV files contain RMS, median frequency, linear envelope, maximum PSD, and other relevant metrics.
    - These files are saved in a timestamped folder, named according to the input file.
2. Graphs and Visualizations (PNG/SVG):
    - Plots of raw, filtered, and rectified EMG signals.
    - RMS and median frequency trends over time.
    - Power Spectral Density (PSD) plot using Welch's method.
3. Summary Report (TXT):
    - A summary text file with key metrics for each EMG file, including RMS, median frequency, and maximum PSD.

Example of Output Files:
------------------------
- filename_results_emg_labiocom.csv: Contains processed metrics for each EMG file.
- filename_filtered_emg.png: Plot of filtered EMG signal.
- filename_rms.png: Plot of RMS values over time.
- filename_median_frequency.png: Plot of median frequency over time.
- log_info.txt: A summary report with all key metrics and processing details.

How to Use:
-----------
1. Run the script:
   python3 emg_labiocom.py
2. The GUI will prompt you to:
   - Select a directory containing EMG CSV files.
   - Input the sampling rate and specify start and end indices for analysis.
3. The script will recursively process all CSV files in the selected directory and save the results in a timestamped folder.
4. You can choose to plot the data interactively or save the plots without displaying them.

Example of Usage:
-----------------
1. Select a folder with multiple CSV files containing EMG data.
2. Enter the sampling rate (e.g., 2000 Hz).
3. Specify the start and end indices for the data range you want to process (e.g., start: 0, end: 10000).
4. The script will process each file in the folder and save the results in the output directory.

Dependencies:
-------------
- Python Standard Libraries: os, datetime, tkinter.
- External Libraries: numpy, scipy, matplotlib (Install via pip install numpy scipy matplotlib).

New Features in v1.1:
----------------------
- Recursive File Processing: Batch processing of all CSV files within a selected directory.
- GUI Enhancements: Improved interface for selecting files, inputting parameters, and displaying results.
- Error Handling: Enhanced error messages for out-of-bounds selections and empty signals.

License:
--------
This program is free software: you can redistribute it and/or modify it under the terms of 
the GNU General Public License as published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. 
If not, see https://www.gnu.org/licenses/.

Disclaimer:
-----------
This script is provided "as is," without any warranty, express or implied, and is intended for 
academic and research purposes only.

Changelog:
----------
- 2024-07-26: Initial release with core EMG analysis functionalities.
- 2024-09-23: Added recursive file processing and improved GUI interaction.
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import messagebox, filedialog, Tk, simpledialog


def butter_lowpass(cutoff, fs, order=4):
    from scipy.signal import butter

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    from scipy.signal import filtfilt

    padding_length = fs  # 1 second padding
    padded_data = np.concatenate(
        [data[:padding_length][::-1], data, data[-padding_length:][::-1]]
    )
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, padded_data)
    return y[padding_length:-padding_length]


def butter_bandpass(lowcut, highcut, fs, order=4):
    from scipy.signal import butter

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    from scipy.signal import filtfilt

    padding_length = fs  # 1 second padding
    padded_data = np.concatenate(
        [data[:padding_length][::-1], data, data[-padding_length:][::-1]]
    )
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, padded_data)
    return y[padding_length:-padding_length]


def full_wave_rectification(emg_signal):
    from scipy.signal import detrend

    emg_detrend = detrend(emg_signal)
    emg_abs = np.abs(emg_detrend)
    return emg_abs


def linear_envelope(emg_abs, cutoff, fs):
    emg_envelope = butter_lowpass_filter(emg_abs, cutoff, fs)
    time = np.linspace(0, (len(emg_abs) - 1) / fs, len(emg_abs))
    signal_integ = np.trapz(emg_envelope, time)
    return emg_envelope, signal_integ


def calculate_rms(semg, window_length, overlap):
    start = 0
    rms_values = []
    while start + window_length < len(semg):
        window = semg[start : start + window_length]
        rms_value = np.sqrt(np.mean(window**2))
        rms_values.append(rms_value)
        start += overlap
    return rms_values


def calculate_median_frequency(semg, fs, window_length, overlap):
    start = 0
    median_freq_values = []
    while start + window_length < len(semg):
        window = semg[start : start + window_length]
        median_freq_value = calculate_median_frequency_for_window(window, fs)
        median_freq_values.append(median_freq_value)
        start += overlap
    return median_freq_values


def calculate_median_frequency_for_window(window, fs):
    from scipy.signal import welch

    nperseg = len(window)
    noverlap = int(nperseg / 2)
    nfft = 1024
    freqs, psd = welch(
        window, fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft
    )
    median_freq_idx = np.where(np.cumsum(psd) >= np.sum(psd) / 2)
    if median_freq_idx[0].size > 0:
        median_freq = freqs[median_freq_idx[0][0]]
    else:
        median_freq = np.nan
    return median_freq


def polynomial_fit(x, y, poly_deg=2):
    poly_coeff = np.polyfit(x, y, poly_deg)
    poly_vals = np.polyval(poly_coeff, x)
    return poly_vals


def emg_analysis(emg_file, fs, start_index, end_index, no_plot, selected_path):
    from scipy.signal import welch

    emg_signal = np.genfromtxt(
        emg_file, delimiter=",", skip_header=1, filling_values=0.0
    )
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000
    time_full = np.linspace(0, (len(emg_signal) - 1), len(emg_signal))
    emg_signal = emg_signal[:, 1]

    print(f"Initial selection - Start index: {start_index}, End index: {end_index}")

    if end_index is None or end_index > len(emg_signal):
        end_index = len(emg_signal)

    if start_index >= len(emg_signal):
        messagebox.showerror(
            "Invalid Start Index", "Start index is out of bounds. Exiting."
        )
        return

    print(f"Final selection - Start index: {start_index}, End index: {end_index}")

    emg_signal_cut = emg_signal[start_index:end_index]

    if len(emg_signal_cut) == 0:
        messagebox.showerror(
            "Empty Signal", "The selected range results in an empty signal. Exiting."
        )
        return

    lowcut = 10.0
    highcut = 450.0
    emg_filtered = butter_bandpass_filter(emg_signal_cut, lowcut, highcut, fs, order=4)

    emg_detrend = full_wave_rectification(emg_filtered)
    emg_abs = np.abs(emg_detrend)

    if len(emg_abs) == 0:
        messagebox.showerror(
            "Empty Signal After Rectification",
            "The rectified signal is empty. Exiting.",
        )
        return

    emg_envelope, signal_integ = linear_envelope(emg_abs, cutoff=10, fs=fs)

    window_length = int(fs * 0.25)
    overlap = int(window_length / 2)
    rms_values = calculate_rms(emg_filtered, window_length, overlap)
    median_freq_values = calculate_median_frequency(
        emg_filtered, fs, window_length, overlap
    )

    time = np.linspace(start_index, end_index - 1, end_index - start_index)
    time_rms = np.linspace(
        start_index, start_index + (len(rms_values) - 1) * overlap, len(rms_values)
    )

    poly2_rms = polynomial_fit(time_rms, rms_values, 2)
    poly2_mdf = polynomial_fit(time_rms, median_freq_values, 2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(emg_file)
    filename = os.path.splitext(base)[0]
    output_dir = os.path.join(selected_path, f"emg_labiocom_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{filename}_results_emg_labiocom.txt")

    max_len = max(len(rms_values), len(median_freq_values))
    rms_array = np.array(rms_values + [0] * (max_len - len(rms_values)))
    freq_array = np.array(
        median_freq_values + [0] * (max_len - len(median_freq_values))
    )
    time_array = np.array(time_rms.tolist() + [0] * (max_len - len(time_rms)))
    linear_envelope_values = np.array([signal_integ] + [0] * (max_len - 1))
    freqs, psd = welch(emg_filtered, fs)
    index_max = np.argmax(psd)
    freq_max = freqs[index_max]
    freq_max_array = np.array([freq_max] + [0] * (max_len - 1))
    psd_max_array = np.array([psd[index_max]] + [0] * (max_len - 1))
    data_matrix = np.vstack(
        (
            time_array,
            rms_array,
            freq_array,
            linear_envelope_values,
            freq_max_array,
            psd_max_array,
        )
    ).T

    with open(output_file, "w") as f:
        f.write(
            "Sample,RMS_µVolts,MedianFrequency_Hz,Linear_envelope_µVolts,Freq_Max,PSD_Max\n"
        )
        np.savetxt(f, data_matrix, fmt="%f", delimiter=",")

    print(f"Results written to {output_file}\n Have a good study!")

    # Plotting and saving figures
    save_formats = ["png", "svg"]  # Formats to save the plots

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(time_full, emg_signal, label="Raw EMG", color="blue", linewidth=1)
    axs[0].set_title("Raw EMG Full-Signal")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("sEMG (µ Volts)")
    axs[0].axis("tight")
    axs[0].grid(True)
    axs[0].axvline(start_index, color="g", linestyle="--", label="Start Index")
    axs[0].axvline(end_index, color="r", linestyle="--", label="End Index")
    axs[0].legend()
    axs[1].plot(
        time,
        emg_signal[start_index:end_index],
        label="Raw EMG",
        color="blue",
        linewidth=3,
    )
    axs[1].plot(time, emg_filtered, label="Filtered EMG", color="red", linewidth=1)
    axs[1].set_title("Cut and Filtered EMG Signal")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("sEMG (µ Volts)")
    axs[1].axis("tight")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()

    for fmt in save_formats:
        plt.savefig(os.path.join(output_dir, f"{filename}_filtered_emg.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig)

    fig2 = plt.figure()
    plt.plot(time, emg_abs)
    plt.plot(time[: len(emg_envelope)], emg_envelope, color="r", linewidth=1)
    plt.xlabel("Sample")
    plt.ylabel("Rectified EMG (µ Volts)")
    plt.title(f"FULL-WAVE & LINEAR ENVELOPE = {signal_integ:.1f} µVolts.s")
    plt.grid(True)
    plt.axis("tight")

    for fmt in save_formats:
        plt.savefig(os.path.join(output_dir, f"{filename}_rectified_emg.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(time_rms, rms_values)
    plt.plot(time_rms, poly2_rms, color="red", linestyle="--")
    plt.title("EMG - RMS")
    plt.xlabel("Sample")
    plt.ylabel("RMS (µ Volts)")
    plt.grid(True)

    for fmt in save_formats:
        plt.savefig(os.path.join(output_dir, f"{filename}_rms.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig3)

    fig4 = plt.figure()
    plt.plot(time_rms, median_freq_values)
    plt.plot(time_rms, poly2_mdf, color="red", linestyle="--")
    plt.title("EMG - Median Frequency")
    plt.xlabel("Sample")
    plt.ylabel("Frequency (Hz)")
    plt.axis("tight")
    plt.grid(True)

    for fmt in save_formats:
        plt.savefig(os.path.join(output_dir, f"{filename}_median_frequency.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig4)

    fig5 = plt.figure()
    plt.plot(freqs, psd)
    plt.title("EMG - PWelch")
    plt.ylabel("PSD (dB/Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.axis("tight")
    plt.grid(True)
    plt.plot(freq_max, psd[index_max], "ro")
    plt.annotate(
        "Max: {:.2f}, {:.2f}".format(freq_max, psd[index_max]),
        xy=(freq_max, psd[index_max]),
        xycoords="data",
        xytext=(+10, +30),
        textcoords="offset points",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

    for fmt in save_formats:
        plt.savefig(os.path.join(output_dir, f"{filename}_pwelch.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig5)


def plot_initial_emg(emg_file, fs):
    emg_signal = np.genfromtxt(
        emg_file, delimiter=",", skip_header=1, filling_values=0.0
    )
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000
    samples = np.arange(len(emg_signal))
    emg_signal = emg_signal[:, 1]

    plt.figure(figsize=(12, 6))
    plt.plot(samples, emg_signal, label="Raw EMG", color="blue", linewidth=1)
    plt.title(
        "Raw sEMG - if you wish, write down the START and END index value to inform in the next window. Then you can close this Figure."
    )
    plt.xlabel("Sample")
    plt.ylabel("sEMG (µ Volts)")
    plt.grid(True)
    plt.show()


def run_emg_gui():
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    selected_path = filedialog.askdirectory(title="Select Directory with EMG Files")
    if not selected_path:
        messagebox.showerror("No Directory Selected", "No directory selected. Exiting.")
        return

    emg_file = filedialog.askopenfilename(
        title="Select EMG File",
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
    )
    if not emg_file:
        messagebox.showerror("No File Selected", "No file selected. Exiting.")
        return

    fs = simpledialog.askinteger(
        "Input", "Enter Sampling Rate (Hz):", initialvalue=2000, minvalue=1
    )
    if fs is None:
        messagebox.showerror("No Sampling Rate", "No sampling rate provided. Exiting.")
        return

    plot_initial_emg(emg_file, fs)  # Now this function is recognized

    input_dialog = simpledialog.askstring(
        "Input",
        "Enter Start (integer sample index), End (integer sample index), Plot (y/n) separated by commas:",
        initialvalue="0,None,n",
    )
    if not input_dialog:
        messagebox.showerror("No Input", "No input provided. Exiting.")
        return

    inputs = input_dialog.split(",")
    if len(inputs) != 3:
        messagebox.showerror(
            "Invalid Input",
            "Invalid input format. Provide three values separated by commas. Exiting.",
        )
        return

    try:
        start_index = int(inputs[0])
        end_index = int(inputs[1]) if inputs[1].lower() != "none" else None
        plot_data = inputs[2].strip().lower() in ["y", "yes"]
    except ValueError:
        messagebox.showerror("Invalid Input", "Invalid input values. Exiting.")
        return

    no_plot = not plot_data

    for filename in os.listdir(selected_path):
        if filename.endswith(".txt") or filename.endswith(".csv"):
            emg_file = os.path.join(selected_path, filename)
            emg_analysis(emg_file, fs, start_index, end_index, no_plot, selected_path)

    messagebox.showinfo("Success", "EMG analysis completed!")


if __name__ == "__main__":
    run_emg_gui()
