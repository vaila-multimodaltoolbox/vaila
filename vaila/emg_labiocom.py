"""
================================================================================
EMG Analysis Toolkit - emg_labiocom (Improved Version)
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Improved Version: 2025
Created: 01.Oct.2024
Updated: 28.May.2025
Version: 0.1.0
Python Version: 3.12.9

Description:
------------
This improved toolkit provides advanced functionalities to analyze EMG (Electromyography) signals.
Improvements include:
- Fixed duplicate plotting issue on Linux
- Interactive selection with mouse clicks
- Multiple segment selection support
- CSV output format for results
- Statistical summary generation
- No special characters in column names
- Advanced EMG analysis techniques (wavelets, fatigue analysis)

Key New Features:
-----------------
1. Interactive Selection: Click-based interval selection with mouse
2. Multiple Segments: Select multiple intervals for analysis
3. Mouse Controls:
   - Left click: Set start point
   - Shift + Left click: Set end point
   - Right click: Remove last selection
4. Enhanced Output: CSV format with statistical summaries
5. Cross-platform compatibility improvements
6. Advanced time-frequency analysis with wavelets
7. Improved fatigue detection algorithms
8. Spectogram visualization

Dependencies:
-------------
- Python Standard Libraries: os, datetime, tkinter
- External Libraries: numpy, scipy, matplotlib, pandas, PyWavelets
Install via: pip install numpy scipy matplotlib pandas PyWavelets
================================================================================
"""

import os
import re
from datetime import datetime
from tkinter import Tk, filedialog, messagebox, simpledialog

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
from scipy.signal import welch

# Import PyWavelets for wavelet analysis if available
try:
    import pywt

    WAVELET_AVAILABLE = True
except ImportError:
    WAVELET_AVAILABLE = False
    print("PyWavelets not available. Wavelet analysis will be disabled.")
    print("Install with: pip install PyWavelets")

# HTML report template
HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EMG Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            box-sizing: border-box;
        }
        h1, h2, h3, h4 {
            color: #2c3e50;
            margin-top: 1.5em;
        }
        h1 {
            text-align: center;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #2c3e50;
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.85em;
            overflow-x: auto;
            display: block;
            white-space: nowrap;
        }
        table thead, table tbody {
            display: table;
            width: 100%;
            table-layout: fixed;
        }
        th, td {
            padding: 8px 6px;
            border: 1px solid #ddd;
            text-align: left;
            word-wrap: break-word;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
            font-size: 0.9em;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .figure {
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .figure img {
            max-width: 100%;
            height: auto;
            margin-bottom: 10px;
        }
        .figure-caption {
            font-style: italic;
            color: #666;
        }
        .reference {
            margin-top: 10px;
            padding-left: 25px;
            text-indent: -25px;
        }
        .abstract {
            background-color: #f8f9fa;
            padding: 15px;
            border-left: 5px solid #3498db;
            margin: 20px 0;
        }
        .metrics-section {
            margin: 30px 0;
        }
        .code {
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
        .box {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
            overflow-x: auto;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 0.9em;
            color: #666;
        }
        .dataframe {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.8em;
            border-radius: 5px 5px 0 0;
            overflow: hidden;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
        .dataframe thead, .dataframe tbody {
            display: table;
            width: 100%;
            table-layout: auto;
        }
        .dataframe thead tr {
            background-color: #3498db;
            color: #ffffff;
            text-align: left;
            font-weight: bold;
        }
        .dataframe th, .dataframe td {
            padding: 8px 6px;
            text-align: right;
            color: #333;
            border: 1px solid #ddd;
            word-wrap: break-word;
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .dataframe tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .dataframe tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .dataframe tbody tr:last-of-type {
            border-bottom: 2px solid #3498db;
        }
        .dataframe th:first-child, .dataframe td:first-child {
            text-align: left;
            font-weight: bold;
            max-width: 200px;
        }
        .segments-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }
        .segments-table th, .segments-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
            color: #333;
        }
        .segments-table th {
            background-color: #3498db;
            color: white;
        }
    </style>
</head>
<body>
    <h1>Advanced EMG Analysis Report</h1>
    <p class="abstract">
        This report presents a comprehensive analysis of Electromyography (EMG) signals using advanced signal processing techniques.
        The analysis was performed using the EMG module of the vailá (Versatile Anarcho Integrated Liberation Ánalysis) Multimodal Toolbox,
        a Python-based platform designed for human movement analysis across multiple biomechanical systems.
        Sampling Rate: <span class="sampling-frequency">2000</span> Hz.
    </p>

    <h2>1. Introduction</h2>
    <p>
        Electromyography (EMG) is a technique for evaluating and recording the electrical activity produced by skeletal muscles.
        EMG signals are complex and non-stationary, requiring sophisticated signal processing methods to extract meaningful information.
        This report details the analysis performed on EMG data using time-domain, frequency-domain, and time-frequency domain methods.
    </p>

    <h2>2. Time Domain Analysis</h2>
    <p>
        The time domain analysis focuses on the amplitude characteristics of the EMG signal. After band-pass filtering
        (10-450 Hz) to remove noise, the signal is processed using the following techniques:
    </p>
    <ul>
        <li>Full-wave rectification: Converting negative amplitudes to positive</li>
        <li>Linear envelope: Low-pass filtering (10 Hz cutoff) of the rectified signal</li>
        <li>Root Mean Square (RMS): Calculating the RMS value in sliding windows</li>
    </ul>

    <h3>2.1. Raw and Filtered EMG</h3>
    <div class="figure">
        <img src="PLACEHOLDER_filtered_emg.png" alt="Raw and Filtered EMG Signal">
        <p class="figure-caption">Top: Raw EMG signal with selected segments highlighted. Bottom: Band-pass filtered EMG signal from selected segment.</p>
    </div>

    <h3>2.2. Rectified EMG and Linear Envelope</h3>
    <div class="figure">
        <img src="PLACEHOLDER_rectified_emg.png" alt="Rectified EMG and Linear Envelope">
        <p class="figure-caption">Full-wave rectified EMG signal (blue) and linear envelope (red). The integral value represents overall muscle activity.</p>
    </div>

    <h3>2.3. Root Mean Square (RMS)</h3>
    <div class="figure">
        <img src="PLACEHOLDER_rms.png" alt="RMS of EMG Signal">
        <p class="figure-caption">RMS values calculated using a 250ms window with 50% overlap. The red dashed line represents a polynomial fit.</p>
    </div>

    <h2>3. Frequency Domain Analysis</h2>
    <p>
        Frequency domain analysis reveals the power distribution across different frequencies, providing insights into
        muscle fiber composition, fatigue, and motor unit recruitment strategies.
    </p>

    <h3>3.1. Median Frequency</h3>
    <div class="figure">
        <img src="PLACEHOLDER_median_frequency.png" alt="Median Frequency Analysis">
        <p class="figure-caption">Median frequency over time. A decreasing trend can indicate muscle fatigue.</p>
    </div>

    <h3>3.2. Power Spectral Density</h3>
    <div class="figure">
        <img src="PLACEHOLDER_pwelch.png" alt="Power Spectral Density">
        <p class="figure-caption">Power spectral density using Welch's method. The red dot indicates the frequency with maximum power.</p>
    </div>

    <h2>4. Time-Frequency Analysis</h2>
    <p>
        Time-frequency analysis provides insights into how the frequency content of the EMG signal changes over time,
        which is particularly useful for non-stationary signals like EMG during dynamic contractions.
    </p>

    <h3>4.1. Spectrogram (STFT)</h3>
    <div class="figure">
        <img src="PLACEHOLDER_spectrogram.png" alt="EMG Spectrogram">
        <p class="figure-caption">Short-time Fourier transform (STFT) showing frequency content over time.</p>
    </div>

    <h3>4.2. Wavelet Analysis</h3>
    <div class="figure">
        <img src="PLACEHOLDER_wavelet.png" alt="Wavelet Analysis">
        <p class="figure-caption">Continuous wavelet transform using Morlet wavelets, providing multi-resolution analysis of the EMG signal.</p>
    </div>

    <h2>5. Statistical Summary</h2>
    <p>
        The table below presents statistical metrics calculated from the EMG signal analysis. These metrics include standard
        statistical measures (mean, median, standard deviation) as well as specialized EMG parameters.
    </p>

    <h2>References</h2>
    <div class="references">
        <p class="reference">
            [1] Merletti, R., & Parker, P. A. (2004). <em>Electromyography: Physiology, Engineering, and Non-Invasive Applications</em>.
            IEEE Press. <a href="https://doi.org/10.1002/0471678384">https://doi.org/10.1002/0471678384</a>
        </p>
        <p class="reference">
            [2] De Luca, C. J. (1997). The use of surface electromyography in biomechanics.
            <em>Journal of Applied Biomechanics</em>, 13(2), 135-163.
            <a href="https://doi.org/10.1123/jab.13.2.135">https://doi.org/10.1123/jab.13.2.135</a>
        </p>
        <p class="reference">
            [3] Cifrek, M., Medved, V., Tonković, S., & Ostojić, S. (2009). Surface EMG based muscle fatigue evaluation in biomechanics.
            <em>Clinical Biomechanics</em>, 24(4), 327-340.
            <a href="https://doi.org/10.1016/j.clinbiomech.2009.01.010">https://doi.org/10.1016/j.clinbiomech.2009.01.010</a>
        </p>
        <p class="reference">
            [4] Santiago, P. R. P. et al. (2024). vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox.
            <em>arXiv preprint</em>. <a href="https://doi.org/10.48550/arXiv.2410.07238">https://doi.org/10.48550/arXiv.2410.07238</a>
        </p>
    </div>
</body>
</html>"""


class EMGSelector:
    def __init__(self, fig, ax, emg_signal, samples):
        self.fig = fig
        self.ax = ax
        self.emg_signal = emg_signal
        self.samples = samples
        self.selections = []
        self.current_start = None
        self.temp_patches = []
        self.selection_patches = []

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Add instructions
        self.ax.set_title(
            "EMG Signal - Click to select intervals\n"
            "Left click: start point, Shift+Left click: end point\n"
            "Right click: remove last selection, Close window when done"
        )

        # Add buttons
        self.setup_buttons()

    def setup_buttons(self):
        # Create button axes
        ax_clear = plt.axes([0.7, 0.01, 0.1, 0.05])
        ax_done = plt.axes([0.81, 0.01, 0.1, 0.05])

        # Create buttons
        self.btn_clear = Button(ax_clear, "Clear All")
        self.btn_done = Button(ax_done, "Done")

        # Connect button events
        self.btn_clear.on_clicked(self.clear_all)
        self.btn_done.on_clicked(self.done_selection)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left click
            if event.key == "shift":  # Shift + left click for end point
                if self.current_start is not None:
                    end_point = int(event.xdata)
                    start_point = self.current_start

                    if end_point > start_point:
                        self.selections.append((start_point, end_point))
                        self.add_selection_patch(start_point, end_point)
                        print(f"Selection added: {start_point} to {end_point}")

                    self.current_start = None
                    self.clear_temp_patches()
            else:  # Regular left click for start point
                self.current_start = int(event.xdata)
                print(f"Start point set: {self.current_start}")
                self.clear_temp_patches()
                self.add_temp_patch(self.current_start)

        elif event.button == 3:  # Right click - remove last selection
            if self.selections:
                removed = self.selections.pop()
                self.remove_last_patch()
                print(f"Removed selection: {removed[0]} to {removed[1]}")

        self.fig.canvas.draw()

    def on_key_press(self, event):
        if event.key == "escape":
            self.current_start = None
            self.clear_temp_patches()
            self.fig.canvas.draw()

    def add_temp_patch(self, start):
        # Add temporary patch for start point
        patch = self.ax.axvline(start, color="orange", linestyle="--", alpha=0.7)
        self.temp_patches.append(patch)

    def clear_temp_patches(self):
        for patch in self.temp_patches:
            patch.remove()
        self.temp_patches.clear()

    def add_selection_patch(self, start, end):
        # Add patch for confirmed selection
        y_min, y_max = self.ax.get_ylim()
        rect = patches.Rectangle(
            (start, y_min),
            end - start,
            y_max - y_min,
            linewidth=2,
            edgecolor="green",
            facecolor="green",
            alpha=0.2,
        )
        self.ax.add_patch(rect)
        self.selection_patches.append(rect)

    def remove_last_patch(self):
        if self.selection_patches:
            patch = self.selection_patches.pop()
            patch.remove()

    def clear_all(self, event):
        self.selections.clear()
        self.current_start = None
        self.clear_temp_patches()
        for patch in self.selection_patches:
            patch.remove()
        self.selection_patches.clear()
        self.fig.canvas.draw()
        print("All selections cleared")

    def done_selection(self, event):
        plt.close(self.fig)


def butter_lowpass(cutoff, fs, order=4):
    from scipy.signal import butter

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    from scipy.signal import filtfilt

    padding_length = fs  # 1 second padding
    padded_data = np.concatenate([data[:padding_length][::-1], data, data[-padding_length:][::-1]])
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
    padded_data = np.concatenate([data[:padding_length][::-1], data, data[-padding_length:][::-1]])
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
    signal_integ = np.trapezoid(emg_envelope, time)
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
    freqs, psd = welch(window, fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    median_freq_idx = np.where(np.cumsum(psd) >= np.sum(psd) / 2)
    median_freq = freqs[median_freq_idx[0][0]] if median_freq_idx[0].size > 0 else np.nan
    return median_freq


def polynomial_fit(x, y, poly_deg=2):
    poly_coeff = np.polyfit(x, y, poly_deg)
    poly_vals = np.polyval(poly_coeff, x)
    return poly_vals


def create_statistical_summary(df, output_dir, filename):
    """Create statistical summary of the results with advanced statistics"""
    # Calculate statistics for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    stats_dict = {}
    for col in numeric_cols:
        if col != "Sample":  # Skip sample column
            data = df[col].dropna()
            if len(data) > 0:
                stats_dict[col] = {
                    "Mean": data.mean(),
                    "Std": data.std(),
                    "Median": data.median(),
                    "Min": data.min(),
                    "Max": data.max(),
                    "Q25": data.quantile(0.25),
                    "Q75": data.quantile(0.75),
                    "CI_Lower_95": data.mean() - 1.96 * data.std() / np.sqrt(len(data)),
                    "CI_Upper_95": data.mean() + 1.96 * data.std() / np.sqrt(len(data)),
                    # New statistical indicators
                    "Skewness": data.skew(),  # Skewness
                    "Kurtosis": data.kurtosis(),  # Kurtosis
                    "IQR": data.quantile(0.75) - data.quantile(0.25),  # Interquartile range
                    "CV": (
                        data.std() / data.mean() if data.mean() != 0 else np.nan
                    ),  # Coefficient of variation
                }

    # Create summary DataFrame
    summary_df = pd.DataFrame(stats_dict).T

    # Save summary
    summary_file = os.path.join(output_dir, f"{filename}_statistical_summary.csv")
    summary_df.to_csv(summary_file)
    print(f"Statistical summary saved to: {summary_file}")

    return summary_df


def plot_and_save_figures(
    emg_signal,
    emg_filtered,
    emg_abs,
    emg_envelope,
    time,
    time_full,
    time_rms,
    rms_values,
    median_freq_values,
    poly2_rms,
    poly2_mdf,
    freqs,
    psd,
    freq_max,
    index_max,
    signal_integ,
    start_index,
    end_index,
    selections,
    output_dir,
    filename,
    no_plot,
    wavelet_data=None,
    spectrogram_data=None,
):
    """Create and save all plots including advanced visualizations"""

    save_formats = ["png", "svg"]

    # Plot 1: Raw and filtered EMG
    fig1, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(time_full, emg_signal, label="Raw EMG", color="blue", linewidth=1)
    axs[0].set_title("Raw EMG Full-Signal")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("sEMG (microVolts)")
    axs[0].axis("tight")
    axs[0].grid(True)

    # Highlight all selections
    for i, (sel_start, sel_end) in enumerate(selections):
        axs[0].axvspan(
            sel_start,
            sel_end,
            alpha=0.3,
            color="green",
            label=f"Selection {i + 1}: {sel_start}-{sel_end}",
        )

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
    axs[1].set_ylabel("sEMG (microVolts)")
    axs[1].axis("tight")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()

    for fmt in save_formats:
        fig1.savefig(os.path.join(output_dir, f"{filename}_filtered_emg.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig1)

    # Plot 2: Rectified and envelope
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, emg_abs, label="Rectified EMG", alpha=0.7)
    ax.plot(
        time[: len(emg_envelope)],
        emg_envelope,
        color="r",
        linewidth=2,
        label="Linear Envelope",
    )
    ax.set_xlabel("Sample")
    axs[1].set_ylabel("Rectified EMG (microVolts)")
    ax.set_title(f"FULL-WAVE & LINEAR ENVELOPE = {signal_integ:.1f} microVolts.s")
    ax.grid(True)
    ax.axis("tight")
    ax.legend()

    for fmt in save_formats:
        fig2.savefig(os.path.join(output_dir, f"{filename}_rectified_emg.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig2)

    # Plot 3: RMS
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_rms, rms_values, label="RMS", linewidth=2)
    ax.plot(time_rms, poly2_rms, color="red", linestyle="--", label="Polynomial Fit")
    ax.set_title("EMG - RMS")
    ax.set_xlabel("Sample")
    ax.set_ylabel("RMS (microVolts)")
    ax.grid(True)
    ax.legend()

    for fmt in save_formats:
        fig3.savefig(os.path.join(output_dir, f"{filename}_rms.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig3)

    # Plot 4: Median Frequency
    fig4, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_rms, median_freq_values, label="Median Frequency", linewidth=2)
    ax.plot(time_rms, poly2_mdf, color="red", linestyle="--", label="Polynomial Fit")
    ax.set_title("EMG - Median Frequency")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Frequency (Hz)")
    ax.axis("tight")
    ax.grid(True)
    ax.legend()

    for fmt in save_formats:
        fig4.savefig(os.path.join(output_dir, f"{filename}_median_frequency.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig4)

    # Plot 5: Power Spectral Density
    fig5, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freqs, psd, linewidth=2)
    ax.set_title("EMG - Power Spectral Density (Welch)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.axis("tight")
    ax.grid(True)
    ax.plot(freq_max, psd[index_max], "ro", markersize=8)
    ax.annotate(
        f"Max: {freq_max:.2f} Hz, {psd[index_max]:.2e} dB/Hz",
        xy=(freq_max, psd[index_max]),
        xycoords="data",
        xytext=(+10, +30),
        textcoords="offset points",
        fontsize=12,
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=.2"},
    )

    for fmt in save_formats:
        fig5.savefig(os.path.join(output_dir, f"{filename}_pwelch.{fmt}"))

    if not no_plot:
        plt.show()
    plt.close(fig5)

    # Novo Plot 6: Espectrograma do sinal EMG
    if spectrogram_data is not None:
        fig6, ax = plt.subplots(figsize=(10, 6))
        times, frequencies, Sxx = spectrogram_data
        pcm = ax.pcolormesh(
            times, frequencies, 10 * np.log10(Sxx), shading="gouraud", cmap="viridis"
        )
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_title("EMG Spectrogram (STFT)")
        fig6.colorbar(pcm, ax=ax, label="Intensity [dB]")

        for fmt in save_formats:
            fig6.savefig(os.path.join(output_dir, f"{filename}_spectrogram.{fmt}"))

        if not no_plot:
            plt.show()
        plt.close(fig6)

    # Novo Plot 7: Análise Wavelet (CWT)
    if wavelet_data is not None and WAVELET_AVAILABLE:
        fig7, ax = plt.subplots(figsize=(10, 6))
        coefs, freqs = wavelet_data
        pcm = ax.contourf(time, freqs, abs(coefs), cmap="viridis")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_title("Continuous Wavelet Transform (CWT)")
        fig7.colorbar(pcm, ax=ax, label="Magnitude")

        for fmt in save_formats:
            fig7.savefig(os.path.join(output_dir, f"{filename}_wavelet.{fmt}"))

        if not no_plot:
            plt.show()
        plt.close(fig7)


def emg_analysis_segment(emg_signal, fs, start_index, end_index, segment_name):
    """Analyze a single EMG segment with advanced techniques"""

    # Extract segment
    emg_signal_cut = emg_signal[start_index:end_index]

    # Check minimum segment size (2 seconds of data)
    min_samples = 2 * fs
    if len(emg_signal_cut) < min_samples:
        print(
            f"Warning: Segment {segment_name} is too short (less than 2 seconds of data). Minimum required: {min_samples} samples, Got: {len(emg_signal_cut)} samples. Skipping."
        )
        return None

    # Check if there's enough data for filtering padding
    padding_length = fs  # 1 second of padding
    if len(emg_signal_cut) <= 3 * padding_length:  # We need at least 3x the padding
        print(
            f"Warning: Segment {segment_name} is too short for filtering. Minimum required: {3 * padding_length} samples, Got: {len(emg_signal_cut)} samples. Skipping."
        )
        return None

    try:
        # Filter
        lowcut = 10.0
        highcut = 450.0
        emg_filtered = butter_bandpass_filter(emg_signal_cut, lowcut, highcut, fs, order=4)

        # Rectify
        emg_abs = full_wave_rectification(emg_filtered)

        # Linear envelope
        emg_envelope, signal_integ = linear_envelope(emg_abs, cutoff=10, fs=fs)

        # RMS and median frequency
        window_length = int(fs * 0.25)  # 250ms window
        if len(emg_filtered) < window_length:
            print(
                f"Warning: Segment {segment_name} is too short for RMS/MDF calculation. Minimum required: {window_length} samples. Skipping."
            )
            return None

        overlap = int(window_length / 2)
        rms_values = calculate_rms(emg_filtered, window_length, overlap)
        median_freq_values = calculate_median_frequency(emg_filtered, fs, window_length, overlap)

        # Check for valid calculation results
        if len(rms_values) == 0 or len(median_freq_values) == 0:
            print(f"Warning: Segment {segment_name} produced no valid RMS/MDF values. Skipping.")
            return None

        # Time arrays
        time = np.linspace(start_index, end_index - 1, end_index - start_index)
        time_rms = np.linspace(
            start_index, start_index + (len(rms_values) - 1) * overlap, len(rms_values)
        )

        # Polynomial fits
        if len(time_rms) > 2:  # Need at least 3 points for polynomial fit
            poly2_rms = polynomial_fit(time_rms, rms_values, 2)
            poly2_mdf = polynomial_fit(time_rms, median_freq_values, 2)
        else:
            print(
                f"Warning: Segment {segment_name} has too few points for polynomial fitting. Using linear fit."
            )
            poly2_rms = polynomial_fit(time_rms, rms_values, 1)
            poly2_mdf = polynomial_fit(time_rms, median_freq_values, 1)

        # PSD analysis
        freqs, psd = welch(emg_filtered, fs)
        index_max = np.argmax(psd)
        freq_max = freqs[index_max]

        # Advanced analyses
        advanced_features = {}

        # 1. Calculate advanced features
        advanced_features.update(calculate_muscle_coordination(emg_filtered))
        advanced_features.update(calculate_frequency_domain_features(emg_filtered, fs))
        advanced_features.update(calculate_fatigue_indices(median_freq_values))

        # 2. Spectrogram Analysis (STFT)
        from scipy.signal import spectrogram

        f, t, Sxx = spectrogram(emg_filtered, fs)
        spectrogram_data = (t, f, Sxx)

        # 3. Wavelet Analysis (if available)
        wavelet_data = None
        if WAVELET_AVAILABLE:
            wavelet_data = calculate_wavelet_transform(emg_filtered, fs)

        # 4. Calculate advanced statistics
        stats = calculate_advanced_statistics(emg_filtered)
        advanced_features.update(stats)

        return {
            "emg_filtered": emg_filtered,
            "emg_abs": emg_abs,
            "emg_envelope": emg_envelope,
            "signal_integ": signal_integ,
            "rms_values": rms_values,
            "median_freq_values": median_freq_values,
            "time": time,
            "time_rms": time_rms,
            "poly2_rms": poly2_rms,
            "poly2_mdf": poly2_mdf,
            "freqs": freqs,
            "psd": psd,
            "freq_max": freq_max,
            "psd_max": psd[index_max],
            "advanced_features": advanced_features,
            "spectrogram_data": spectrogram_data,
            "wavelet_data": wavelet_data,
        }

    except Exception as e:
        print(f"Error processing segment {segment_name}: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def emg_analysis(emg_file, fs, selections, no_plot, output_dir, generate_report=False):
    """Main EMG analysis function with multiple segment support"""

    # Load EMG data
    emg_signal = np.genfromtxt(emg_file, delimiter=",", skip_header=1, filling_values=0.0)
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000  # Convert to microVolts
    time_full = np.linspace(0, len(emg_signal) - 1, len(emg_signal))
    emg_signal = emg_signal[:, 1]

    base = os.path.basename(emg_file)
    filename = os.path.splitext(base)[0]

    # Create subdirectory for this file
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)

    # Process each selection, adjusting for file size
    all_results = []
    signal_length = len(emg_signal)

    for i, (start_index, end_index) in enumerate(selections):
        # Adjust indices for the file size
        adjusted_start = min(start_index, signal_length - 1)
        adjusted_end = min(end_index, signal_length)

        if adjusted_start >= adjusted_end:
            print(f"Warning: Invalid segment {i + 1} for file {filename}")
            print(f"Original selection: {start_index} to {end_index}")
            print(f"File length: {signal_length} samples")
            print("Skipping this segment.\n")
            continue

        if adjusted_end != end_index:
            print(
                f"Note: Segment {i + 1} end point adjusted from {end_index} to {adjusted_end} due to file length"
            )

        print(f"Processing segment {i + 1} for {filename}: {adjusted_start} to {adjusted_end}")

        # Analyze segment
        segment_name = f"segment_{i + 1}"
        result = emg_analysis_segment(emg_signal, fs, adjusted_start, adjusted_end, segment_name)

        if result is None:
            continue

        # Prepare data for CSV
        max_len = max(len(result["rms_values"]), len(result["median_freq_values"]))

        # Create arrays with consistent length
        segment_data = []
        for j in range(max_len):
            # Basic metrics
            row = {
                "Segment": segment_name,
                "Sample": (result["time_rms"][j] if j < len(result["time_rms"]) else np.nan),
                "RMS_microVolts": (
                    result["rms_values"][j] if j < len(result["rms_values"]) else np.nan
                ),
                "MedianFrequency_Hz": (
                    result["median_freq_values"][j]
                    if j < len(result["median_freq_values"])
                    else np.nan
                ),
                "Linear_envelope_microVolts": (result["signal_integ"] if j == 0 else np.nan),
                "Freq_Max_Hz": result["freq_max"] if j == 0 else np.nan,
                "PSD_Max": result["psd_max"] if j == 0 else np.nan,
            }

            # Add all advanced features to the first row only
            if j == 0 and "advanced_features" in result:
                for key, value in result["advanced_features"].items():
                    row[key] = value

            segment_data.append(row)

        all_results.extend(segment_data)

        # Create plots for this segment
        plot_and_save_figures(
            emg_signal,
            result["emg_filtered"],
            result["emg_abs"],
            result["emg_envelope"],
            result["time"],
            time_full,
            result["time_rms"],
            result["rms_values"],
            result["median_freq_values"],
            result["poly2_rms"],
            result["poly2_mdf"],
            result["freqs"],
            result["psd"],
            result["freq_max"],
            np.argmax(result["psd"]),
            result["signal_integ"],
            adjusted_start,
            adjusted_end,
            selections,
            file_output_dir,
            f"{filename}_{segment_name}",
            no_plot,
            result["wavelet_data"],
            result["spectrogram_data"],
        )

    # Save combined results to CSV
    summary_df = None
    if all_results:
        # Create DataFrame directly from the list of dictionaries
        df = pd.DataFrame(all_results)
        results_file = os.path.join(file_output_dir, f"{filename}_results_emg_labiocom.csv")
        df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")

        # Create statistical summary
        summary_df = create_statistical_summary(df, file_output_dir, filename)

        # Generate HTML report if requested
        if generate_report:
            generate_html_report(file_output_dir, filename, summary_df, fs)

    print(f"Analysis completed for {filename}")


def get_interactive_selection(emg_file, fs):
    """Get interactive selection using mouse clicks"""

    # Load EMG data for display
    emg_signal = np.genfromtxt(emg_file, delimiter=",", skip_header=1, filling_values=0.0)
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000
    samples = np.arange(len(emg_signal))
    emg_signal = emg_signal[:, 1]

    # Create interactive plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(samples, emg_signal, label="Raw EMG", color="blue", linewidth=1)
    ax.set_xlabel("Sample")
    ax.set_ylabel("sEMG (microVolts)")
    ax.grid(True)

    # Create selector
    selector = EMGSelector(fig, ax, emg_signal, samples)

    # Show plot and wait for user interaction
    plt.show()

    return selector.selections


def get_manual_selection(emg_file, fs):
    """Get manual selection after showing the signal"""
    # Show the signal first
    emg_signal = np.genfromtxt(emg_file, delimiter=",", skip_header=1, filling_values=0.0)
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000
    samples = np.arange(len(emg_signal))
    emg_signal = emg_signal[:, 1]

    # Create plot for visualization
    plt.figure(figsize=(15, 8))
    plt.plot(samples, emg_signal, label="Raw EMG", color="blue", linewidth=1)
    plt.title("Raw EMG Signal - Note the intervals you want to analyze")
    plt.xlabel("Sample")
    plt.ylabel("sEMG (microVolts)")
    plt.grid(True)
    plt.show()

    # Now ask for the intervals
    input_dialog = simpledialog.askstring(
        "Manual Input",
        "Enter selections as: start1,end1;start2,end2;... (e.g., 1000,5000;10000,15000):",
        initialvalue="0,1000",
    )

    if not input_dialog:
        return []

    selections = []
    try:
        pairs = input_dialog.split(";")
        for pair in pairs:
            start, end = map(int, pair.split(","))
            selections.append((start, end))
    except ValueError:
        messagebox.showerror("Invalid Input", "Invalid selection format.")
        return []

    return selections


def run_emg_gui():
    """Main GUI function"""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    # Select input directory with EMG files
    input_path = filedialog.askdirectory(title="Select Directory with EMG Files")
    if not input_path:
        messagebox.showerror("No Directory Selected", "No input directory selected. Exiting.")
        return

    # Select reference file for segmentation
    ref_file = filedialog.askopenfilename(
        title="Select Reference EMG File for Segmentation",
        initialdir=input_path,
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")],
    )
    if not ref_file:
        messagebox.showerror("No Reference File", "No reference file selected. Exiting.")
        return

    # Get sampling rate
    fs = simpledialog.askinteger(
        "Input", "Enter Sampling Rate (Hz):", initialvalue=2000, minvalue=1
    )
    if fs is None:
        messagebox.showerror("No Sampling Rate", "No sampling rate provided. Exiting.")
        return

    # Ask for selection method
    selection_method = messagebox.askyesnocancel(
        "Selection Method",
        "Choose selection method:\nYes = Interactive (mouse clicks)\nNo = Manual (type values)\nCancel = Exit",
    )

    if selection_method is None:
        return

    # Get segments from reference file
    print(f"\nProcessing reference file: {os.path.basename(ref_file)}")
    if selection_method:  # Interactive
        selections = get_interactive_selection(ref_file, fs)
    else:  # Manual
        selections = get_manual_selection(ref_file, fs)

    if not selections:
        messagebox.showerror("No Selections", "No segments selected. Exiting.")
        return

    print(f"Selected intervals from reference file: {selections}")

    # Ask about plotting
    plot_data = messagebox.askyesno("Plotting", "Show plots during analysis?")
    no_plot = not plot_data

    # Ask about HTML report generation
    generate_report = messagebox.askyesno(
        "HTML Report", "Generate comprehensive HTML reports for each file?"
    )

    # Select output directory (different from input)
    output_path = filedialog.askdirectory(
        title="Select Output Directory for Results",
        initialdir=os.path.dirname(input_path),  # Start from parent of input dir
    )
    if not output_path:
        messagebox.showerror("No Output Directory", "No output directory selected. Exiting.")
        return

    # Create timestamped directory in output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_path, f"emg_labiocom_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Process all CSV/TXT files in directory using the same segments
    processed_files = 0

    for filename in os.listdir(input_path):
        if filename.endswith((".txt", ".csv")):
            emg_file = os.path.join(input_path, filename)
            print(f"\nProcessing file: {filename}")

            # Analyze EMG using segments from reference file
            emg_analysis(emg_file, fs, selections, no_plot, output_dir, generate_report)
            processed_files += 1

    if processed_files > 0:
        # Create index.html that links to all reports if HTML reports were generated
        if generate_report:
            create_index_html(output_dir)

        messagebox.showinfo(
            "Success",
            f"EMG analysis completed!\n"
            f"{processed_files} files processed using segments from {os.path.basename(ref_file)}.\n"
            f"Results saved in: {output_dir}",
        )
    else:
        messagebox.showinfo("No Files", "No files were processed.")


def create_index_html(output_dir):
    """Create an index.html file that links to all generated reports"""
    import os

    # Find all subdirectories (one for each processed file)
    subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    # Find HTML reports in each subdirectory
    reports = []
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        html_files = [f for f in os.listdir(subdir_path) if f.endswith("_emg_report.html")]
        for html_file in html_files:
            reports.append((subdir, html_file))

    if not reports:
        return

    # Create index.html
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write('<html lang="en">\n')
        f.write("<head>\n")
        f.write('    <meta charset="UTF-8">\n')
        f.write('    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n')
        f.write("    <title>EMG Analysis Reports</title>\n")
        f.write("    <style>\n")
        f.write(
            "        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }\n"
        )
        f.write("        h1 { text-align: center; color: #2c3e50; }\n")
        f.write("        ul { list-style-type: none; padding: 0; }\n")
        f.write(
            "        li { margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }\n"
        )
        f.write("        a { color: #3498db; text-decoration: none; }\n")
        f.write("        a:hover { text-decoration: underline; }\n")
        f.write("        .timestamp { color: #666; font-size: 0.8em; margin-top: 5px; }\n")
        f.write("    </style>\n")
        f.write("</head>\n")
        f.write("<body>\n")
        f.write("    <h1>EMG Analysis Reports</h1>\n")
        f.write("    <p>Generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</p>\n")
        f.write("    <ul>\n")

        for subdir, html_file in reports:
            rel_path = os.path.join(subdir, html_file)
            f.write(f'        <li><a href="{rel_path}">{html_file}</a></li>\n')

        f.write("    </ul>\n")
        f.write('    <p style="text-align: center; margin-top: 30px; color: #666;">\n')
        f.write("        Generated using vailá Multimodal Toolbox | EMG Analysis Module<br>\n")
        f.write(
            '        <a href="https://doi.org/10.48550/arXiv.2410.07238">vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox</a>\n'
        )
        f.write("    </p>\n")
        f.write("</body>\n")
        f.write("</html>\n")

    print(f"Index file created: {index_path}")


def calculate_entropy(signal, window_length, overlap):
    """Calculate Sample Entropy (SampEn) for EMG signal"""
    from entropy import sample_entropy

    start = 0
    entropy_values = []
    while start + window_length < len(signal):
        window = signal[start : start + window_length]
        # Calculate Sample Entropy with m=2, r=0.2*std
        entropy = sample_entropy(window, order=2, metric="chebyshev")
        entropy_values.append(entropy)
        start += overlap
    return entropy_values


def calculate_fatigue_indices(median_freq_values):
    """Calculate fatigue indices based on median frequency trend"""
    # Linear regression on median frequency
    x = np.arange(len(median_freq_values))
    slope, intercept = np.polyfit(x, median_freq_values, 1)

    # Fatigue Index (FI) - normalized slope
    fi = slope / intercept if intercept != 0 else 0

    # Modified Fatigue Index (MFI) - area under MF curve
    mfi = np.trapezoid(median_freq_values) / len(median_freq_values)

    # New: Dimitrov Fatigue Index (DFI)
    # Based on ratio between low and high frequency powers
    if len(median_freq_values) > 2:
        half_idx = len(median_freq_values) // 2
        first_half = np.mean(median_freq_values[:half_idx])
        second_half = np.mean(median_freq_values[half_idx:])
        dfi = first_half / second_half if second_half != 0 else 0
    else:
        dfi = 0

    return {
        "fatigue_index": fi,
        "modified_fatigue_index": mfi,
        "mf_slope": slope,
        "mf_intercept": intercept,
        "dimitrov_fatigue_index": dfi,
    }


def calculate_muscle_coordination(emg_filtered):
    """Calculate muscle coordination metrics"""
    # Zero Crossing Rate (ZCR)
    zcr = np.sum(np.diff(np.signbit(emg_filtered).astype(int))) / len(emg_filtered)

    # Mean Absolute Value (MAV)
    mav = np.mean(np.abs(emg_filtered))

    # Variance of EMG (VAR)
    var_emg = np.var(emg_filtered)

    # Waveform Length (WL)
    wl = np.sum(np.abs(np.diff(emg_filtered)))

    # Novo: Wilson Amplitude (WAMP) - contagem de vezes que a diferença entre amostras excede um limiar
    threshold = 0.1 * np.std(emg_filtered)
    wamp = np.sum(np.abs(np.diff(emg_filtered)) > threshold)

    # Novo: Slope Sign Changes (SSC) - mudanças no sinal da primeira derivada
    slope_changes = np.diff(np.sign(np.diff(emg_filtered)))
    ssc = np.sum(slope_changes != 0)

    return {
        "zero_crossing_rate": zcr,
        "mean_absolute_value": mav,
        "variance": var_emg,
        "waveform_length": wl,
        "wilson_amplitude": wamp,
        "slope_sign_changes": ssc,
    }


def calculate_frequency_domain_features(emg_filtered, fs):
    """Calculate frequency domain features"""
    from scipy.signal import welch

    # Calculate Power Spectral Density
    freqs, psd = welch(emg_filtered, fs)

    # Mean Frequency
    mean_freq = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0

    # Peak Frequency
    peak_freq = freqs[np.argmax(psd)]

    # Spectral Moments
    (
        np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
    )  # First moment (mean freq)
    moment2 = (
        np.sum((freqs - mean_freq) ** 2 * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
    )  # Second moment (variance)

    # Power Bands
    low_band = np.sum(psd[(freqs >= 20) & (freqs <= 50)])
    mid_band = np.sum(psd[(freqs > 50) & (freqs <= 100)])
    high_band = np.sum(psd[(freqs > 100) & (freqs <= 150)])

    # Novo: Razão de potência entre bandas (útil para detecção de fadiga)
    power_ratio = low_band / high_band if high_band > 0 else 0

    # Novo: Média Modificada da Frequência de Potência (MMFP)
    # Mais robusta ao ruído que a média tradicional
    np.sum(psd)
    mmfp = np.sum(freqs * np.sqrt(psd)) / np.sum(np.sqrt(psd)) if np.sum(np.sqrt(psd)) > 0 else 0

    return {
        "mean_frequency": mean_freq,
        "peak_frequency": peak_freq,
        "spectral_variance": moment2,
        "low_band_power": low_band,
        "mid_band_power": mid_band,
        "high_band_power": high_band,
        "power_ratio": power_ratio,
        "modified_mean_power_freq": mmfp,
    }


def calculate_wavelet_transform(signal, fs):
    """Calculate continuous wavelet transform of EMG signal"""
    if not WAVELET_AVAILABLE:
        return None

    # Configurar escalas para frequências de interesse em EMG (10-500 Hz)
    widths = np.arange(1, 31)

    # Aplicar CWT com wavelet Morlet (adequada para sinais biomédicos)
    coefs, freqs = pywt.cwt(signal, widths, "morl", 1 / fs)

    return coefs, freqs


def calculate_advanced_statistics(signal):
    """Calculate advanced statistical features of EMG signal"""
    # Prepare signal
    if len(signal) == 0:
        return {
            "skewness": 0,
            "kurtosis": 0,
            "hjorth_mobility": 0,
            "hjorth_complexity": 0,
        }

    # Skewness (assimetria)
    mean = np.mean(signal)
    std = np.std(signal)
    skewness = 0 if std == 0 else np.mean(((signal - mean) / std) ** 3)

    # Kurtosis (curtose)
    kurtosis = 0 if std == 0 else np.mean(((signal - mean) / std) ** 4) - 3

    # Hjorth Parameters (mobility and complexity)
    # Utils parameters for EEG/EMG characterization
    var0 = np.var(signal)

    # Avoid division by zero
    if var0 == 0:
        hjorth_mobility = 0
        hjorth_complexity = 0
    else:
        # First derivative
        d1 = np.diff(signal)
        var1 = np.var(d1)

        # Second derivative
        d2 = np.diff(d1)
        var2 = np.var(d2)

        # Mobility
        hjorth_mobility = np.sqrt(var1 / var0) if var0 > 0 else 0

        # Complexity
        hjorth_complexity = (
            np.sqrt(var2 / var1) / hjorth_mobility if var1 > 0 and hjorth_mobility > 0 else 0
        )

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
    }


def detect_muscle_activation(emg_envelope, threshold_factor=0.3):
    """Detect muscle activation periods based on envelope threshold"""
    # Calculate adaptive threshold (30% of maximum by default)
    threshold = threshold_factor * np.max(emg_envelope)

    # Identify where signal exceeds threshold
    activations = emg_envelope > threshold

    # Find activation start and end points
    activation_starts = []
    activation_ends = []

    # Identify transitions (0->1 and 1->0)
    transitions = np.diff(activations.astype(int))
    start_indices = np.where(transitions == 1)[0] + 1
    end_indices = np.where(transitions == -1)[0] + 1

    # Ensure we have complete pairs
    if len(start_indices) > 0:
        # If starting activated, add beginning at 0
        if activations[0]:
            start_indices = np.insert(start_indices, 0, 0)

        # If ending activated, add end at last point
        if activations[-1]:
            end_indices = np.append(end_indices, len(activations) - 1)

        # Associate starts and ends (assuming equal number)
        min_length = min(len(start_indices), len(end_indices))
        activation_starts = start_indices[:min_length]
        activation_ends = end_indices[:min_length]

    return list(zip(activation_starts, activation_ends, strict=False))


def generate_html_report(output_dir, filename_prefix, summary_df, fs):
    """Generate an HTML report with all analysis results and figures

    Args:
        output_dir: Directory containing results
        filename_prefix: Base filename for the analyzed file
        summary_df: DataFrame with statistical summary
        fs: Sampling frequency used in analysis
    """
    import glob
    import os
    import shutil

    # Define report output path
    report_path = os.path.join(output_dir, f"{filename_prefix}_emg_report.html")

    try:
        # Start with the template
        html_content = HTML_REPORT_TEMPLATE

        # Update document title with filename
        html_content = html_content.replace(
            "<title>EMG Analysis Report</title>",
            f"<title>EMG Analysis Report - {filename_prefix}</title>",
        )

        # Update report title
        html_content = html_content.replace(
            "<h1>Advanced EMG Analysis Report</h1>",
            f"<h1>Advanced EMG Analysis Report - {filename_prefix}</h1>",
        )

        # Update sampling frequency
        html_content = html_content.replace(
            '<span class="sampling-frequency">2000</span>',
            f'<span class="sampling-frequency">{fs}</span>',
        )

        # Find all segment files for this analysis
        results_csv = os.path.join(output_dir, f"{filename_prefix}_results_emg_labiocom.csv")
        segments_info = ""
        segments = []

        if os.path.exists(results_csv):
            try:
                results_df = pd.read_csv(results_csv)
                segments = results_df["Segment"].unique()

                # Create segment information
                segments_info = "<h4>Analyzed Segments</h4>"
                segments_info += "<table class='segments-table'>"
                segments_info += "<tr><th>Segment</th><th>Samples</th><th>RMS (microVolts)</th><th>Median Frequency (Hz)</th></tr>"

                for segment in segments:
                    segment_data = results_df[results_df["Segment"] == segment]
                    samples_range = (
                        f"{int(segment_data['Sample'].min())} - {int(segment_data['Sample'].max())}"
                    )
                    rms_mean = segment_data["RMS_microVolts"].mean()
                    mf_mean = segment_data["MedianFrequency_Hz"].mean()

                    segments_info += f"<tr><td>{segment}</td><td>{samples_range}</td>"
                    segments_info += f"<td>{rms_mean:.2f}</td><td>{mf_mean:.2f}</td></tr>"

                segments_info += "</table>"

                # Insert segment information after abstract
                html_content = html_content.replace(
                    "</p>\n\n    <h2>1. Introduction</h2>",
                    f'</p>\n\n    <div class="segment-info">\n{segments_info}\n    </div>\n\n    <h2>1. Introduction</h2>',
                )
            except Exception as e:
                print(f"Warning: Could not parse results CSV file: {str(e)}")

        # Get list of all segment names for the current file
        segment_names = [f"segment_{i + 1}" for i in range(len(segments))]

        # Update image paths with actual paths - create image galleries for multiple segments
        figure_types = [
            "filtered_emg",
            "rectified_emg",
            "rms",
            "median_frequency",
            "pwelch",
            "spectrogram",
            "wavelet",
        ]

        for fig_type in figure_types:
            # First look for segment-specific images
            found_images = []
            for segment in segment_names:
                pattern = os.path.join(output_dir, f"{filename_prefix}_{segment}_{fig_type}.png")
                matching_files = glob.glob(pattern)
                if matching_files:
                    found_images.append((segment, matching_files[0]))

            # If no segment-specific images found, try generic pattern
            if not found_images:
                pattern = os.path.join(output_dir, f"{filename_prefix}*{fig_type}.png")
                matching_files = glob.glob(pattern)
                if matching_files:
                    found_images.append(("", matching_files[0]))

            if found_images:
                # Create image gallery for multiple segments or single image for one segment
                if len(found_images) > 1:
                    # Create gallery
                    gallery_html = '<div class="image-gallery">\n'
                    for segment, img_path in found_images:
                        # Copy the image to the HTML file directory if needed
                        dest_img_path = os.path.join(output_dir, os.path.basename(img_path))
                        if img_path != dest_img_path:
                            shutil.copy2(img_path, dest_img_path)

                        gallery_html += '<div class="gallery-item">\n'
                        gallery_html += (
                            f'<img src="{os.path.basename(img_path)}" alt="{segment} {fig_type}">\n'
                        )
                        gallery_html += f'<div class="caption">{segment}</div>\n'
                        gallery_html += "</div>\n"
                    gallery_html += "</div>\n"

                    # Replace placeholder with gallery
                    placeholder = f'<div class="figure">\n        <img src="PLACEHOLDER_{fig_type}.png" alt=".*?">\n        <p class="figure-caption">.*?</p>\n    </div>'
                    html_content = re.sub(placeholder, gallery_html, html_content, flags=re.DOTALL)
                else:
                    # Single image
                    segment, img_path = found_images[0]
                    # Copy the image to the HTML file directory
                    dest_img_path = os.path.join(output_dir, os.path.basename(img_path))
                    if img_path != dest_img_path:
                        shutil.copy2(img_path, dest_img_path)

                    # Replace placeholder with actual image path
                    html_content = html_content.replace(
                        f"PLACEHOLDER_{fig_type}.png", os.path.basename(img_path)
                    )

        # Add gallery CSS styles
        gallery_style = """
        .image-gallery {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        .gallery-item {
            flex: 1 1 300px;
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            background-color: #f8f9fa;
            text-align: center;
        }
        .gallery-item img {
            max-width: 100%;
            height: auto;
            margin-bottom: 10px;
        }
        .gallery-item .caption {
            font-style: italic;
            color: #666;
        }
        """
        html_content = html_content.replace("</style>", f"{gallery_style}</style>")

        # Format the summary DataFrame to avoid scientific notation
        if not summary_df.empty:
            # Format numbers to avoid scientific notation
            pd.set_option("display.float_format", "{:.6f}".format)

            # Format specific columns with appropriate precision
            formatters = {}
            for col in summary_df.columns:
                if "microVolts" in col or "RMS" in col or "Frequency" in col or "Hz" in col:
                    formatters[col] = lambda x: f"{x:.2f}"
                elif "ratio" in col.lower() or "index" in col.lower():
                    formatters[col] = lambda x: f"{x:.4f}"
                else:
                    formatters[col] = lambda x: f"{x:.6f}"

            # Convert DataFrame to HTML with formatted numbers
            summary_html = summary_df.to_html(classes="dataframe", border=0, formatters=formatters)

            # Replace scientific notation in the entire HTML content
            import re

            html_content = re.sub(
                r"(\d+\.\d+)e[+-]\d+",
                lambda m: f"{float(m.group(0)):.6f}",
                html_content,
            )

            # Insert the summary table BEFORE the References section, not at the end
            html_content = html_content.replace(
                "    <h2>References</h2>",
                f'    <div class="box">\n{summary_html}\n    </div>\n\n    <h2>References</h2>',
            )

        # Add timestamp to the report
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = html_content.replace(
            "</body>", f'<div class="footer">Generated on {timestamp}</div>\n</body>'
        )

        # Write the report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"HTML report generated: {report_path}")

    except Exception as e:
        print(f"Error generating HTML report: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_emg_gui()
