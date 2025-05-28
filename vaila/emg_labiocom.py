"""
================================================================================
EMG Analysis Toolkit - emg_labiocom (Improved Version)
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Improved Version: 2025
Created: 01.Oct.2024
Updated: 28.May.2025
Version: 0.0.2
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

Dependencies:
-------------
- Python Standard Libraries: os, datetime, tkinter
- External Libraries: numpy, scipy, matplotlib, pandas
Install via: pip install numpy scipy matplotlib pandas
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tkinter import messagebox, filedialog, Tk, simpledialog
from matplotlib.widgets import Button
import matplotlib.patches as patches


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
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add instructions
        self.ax.set_title('EMG Signal - Click to select intervals\n'
                         'Left click: start point, Shift+Left click: end point\n'
                         'Right click: remove last selection, Close window when done')
        
        # Add buttons
        self.setup_buttons()
    
    def setup_buttons(self):
        # Create button axes
        ax_clear = plt.axes([0.7, 0.01, 0.1, 0.05])
        ax_done = plt.axes([0.81, 0.01, 0.1, 0.05])
        
        # Create buttons
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_done = Button(ax_done, 'Done')
        
        # Connect button events
        self.btn_clear.on_clicked(self.clear_all)
        self.btn_done.on_clicked(self.done_selection)
    
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left click
            if event.key == 'shift':  # Shift + left click for end point
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
        if event.key == 'escape':
            self.current_start = None
            self.clear_temp_patches()
            self.fig.canvas.draw()
    
    def add_temp_patch(self, start):
        # Add temporary patch for start point
        patch = self.ax.axvline(start, color='orange', linestyle='--', alpha=0.7)
        self.temp_patches.append(patch)
    
    def clear_temp_patches(self):
        for patch in self.temp_patches:
            patch.remove()
        self.temp_patches.clear()
    
    def add_selection_patch(self, start, end):
        # Add patch for confirmed selection
        y_min, y_max = self.ax.get_ylim()
        rect = patches.Rectangle((start, y_min), end-start, y_max-y_min, 
                               linewidth=2, edgecolor='green', facecolor='green', alpha=0.2)
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


def create_statistical_summary(df, output_dir, filename):
    """Create statistical summary of the results"""
    # Calculate statistics for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats_dict = {}
    for col in numeric_cols:
        if col != 'Sample':  # Skip sample column
            data = df[col].dropna()
            if len(data) > 0:
                stats_dict[col] = {
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Median': data.median(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Q25': data.quantile(0.25),
                    'Q75': data.quantile(0.75),
                    'CI_Lower_95': data.mean() - 1.96 * data.std() / np.sqrt(len(data)),
                    'CI_Upper_95': data.mean() + 1.96 * data.std() / np.sqrt(len(data))
                }
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(stats_dict).T
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{filename}_statistical_summary.csv")
    summary_df.to_csv(summary_file)
    print(f"Statistical summary saved to: {summary_file}")
    
    return summary_df


def plot_and_save_figures(emg_signal, emg_filtered, emg_abs, emg_envelope, 
                         time, time_full, time_rms, rms_values, median_freq_values,
                         poly2_rms, poly2_mdf, freqs, psd, freq_max, index_max,
                         signal_integ, start_index, end_index, selections,
                         output_dir, filename, no_plot):
    """Create and save all plots"""
    
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
        axs[0].axvspan(sel_start, sel_end, alpha=0.3, color='green', 
                      label=f'Selection {i+1}: {sel_start}-{sel_end}')
    
    axs[0].legend()
    
    axs[1].plot(time, emg_signal[start_index:end_index], label="Raw EMG", color="blue", linewidth=3)
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
    ax.plot(time[:len(emg_envelope)], emg_envelope, color='r', linewidth=2, label="Linear Envelope")
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
        "Max: {:.2f} Hz, {:.2e} dB/Hz".format(freq_max, psd[index_max]),
        xy=(freq_max, psd[index_max]),
        xycoords="data",
        xytext=(+10, +30),
        textcoords="offset points",
        fontsize=12,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )
    
    for fmt in save_formats:
        fig5.savefig(os.path.join(output_dir, f"{filename}_pwelch.{fmt}"))
    
    if not no_plot:
        plt.show()
    plt.close(fig5)


def emg_analysis_segment(emg_signal, fs, start_index, end_index, segment_name):
    """Analyze a single EMG segment"""
    
    # Extract segment
    emg_signal_cut = emg_signal[start_index:end_index]
    
    # Verificar tamanho mínimo do segmento (2 segundos de dados)
    min_samples = 2 * fs
    if len(emg_signal_cut) < min_samples:
        print(f"Warning: Segment {segment_name} is too short (less than 2 seconds of data). Minimum required: {min_samples} samples, Got: {len(emg_signal_cut)} samples. Skipping.")
        return None
    
    # Verificar se há dados suficientes para o padding da filtragem
    padding_length = fs  # 1 segundo de padding
    if len(emg_signal_cut) <= 3 * padding_length:  # Precisamos de pelo menos 3x o padding
        print(f"Warning: Segment {segment_name} is too short for filtering. Minimum required: {3 * padding_length} samples, Got: {len(emg_signal_cut)} samples. Skipping.")
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
            print(f"Warning: Segment {segment_name} is too short for RMS/MDF calculation. Minimum required: {window_length} samples. Skipping.")
            return None
            
        overlap = int(window_length / 2)
        rms_values = calculate_rms(emg_filtered, window_length, overlap)
        median_freq_values = calculate_median_frequency(emg_filtered, fs, window_length, overlap)
        
        # Verificar se os cálculos geraram resultados válidos
        if len(rms_values) == 0 or len(median_freq_values) == 0:
            print(f"Warning: Segment {segment_name} produced no valid RMS/MDF values. Skipping.")
            return None
            
        # Time arrays
        time = np.linspace(start_index, end_index - 1, end_index - start_index)
        time_rms = np.linspace(
            start_index, start_index + (len(rms_values) - 1) * overlap, len(rms_values)
        )
        
        # Polynomial fits
        if len(time_rms) > 2:  # Precisamos de pelo menos 3 pontos para fit polinomial
            poly2_rms = polynomial_fit(time_rms, rms_values, 2)
            poly2_mdf = polynomial_fit(time_rms, median_freq_values, 2)
        else:
            print(f"Warning: Segment {segment_name} has too few points for polynomial fitting. Using linear fit.")
            poly2_rms = polynomial_fit(time_rms, rms_values, 1)
            poly2_mdf = polynomial_fit(time_rms, median_freq_values, 1)
        
        # PSD analysis
        from scipy.signal import welch
        freqs, psd = welch(emg_filtered, fs)
        index_max = np.argmax(psd)
        freq_max = freqs[index_max]
        
        return {
            'emg_filtered': emg_filtered,
            'emg_abs': emg_abs,
            'emg_envelope': emg_envelope,
            'signal_integ': signal_integ,
            'rms_values': rms_values,
            'median_freq_values': median_freq_values,
            'time': time,
            'time_rms': time_rms,
            'poly2_rms': poly2_rms,
            'poly2_mdf': poly2_mdf,
            'freqs': freqs,
            'psd': psd,
            'freq_max': freq_max,
            'psd_max': psd[index_max]
        }
        
    except Exception as e:
        print(f"Error processing segment {segment_name}: {str(e)}")
        return None


def emg_analysis(emg_file, fs, selections, no_plot, output_dir):
    """Main EMG analysis function with multiple segment support"""
    from scipy.signal import welch
    
    # Load EMG data
    emg_signal = np.genfromtxt(emg_file, delimiter=",", skip_header=1, filling_values=0.0)
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000  # Convert to microVolts
    time_full = np.linspace(0, len(emg_signal) - 1, len(emg_signal))
    emg_signal = emg_signal[:, 1]
    
    base = os.path.basename(emg_file)
    filename = os.path.splitext(base)[0]
    
    # Criar subdiretório específico para este arquivo
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)
    
    # Process each selection, adjusting for file size
    all_results = []
    signal_length = len(emg_signal)
    
    for i, (start_index, end_index) in enumerate(selections):
        # Ajustar índices para o tamanho do arquivo atual
        adjusted_start = min(start_index, signal_length - 1)
        adjusted_end = min(end_index, signal_length)
        
        if adjusted_start >= adjusted_end:
            print(f"Warning: Invalid segment {i+1} for file {filename}")
            print(f"Original selection: {start_index} to {end_index}")
            print(f"File length: {signal_length} samples")
            print(f"Skipping this segment.\n")
            continue
        
        if adjusted_end != end_index:
            print(f"Note: Segment {i+1} end point adjusted from {end_index} to {adjusted_end} due to file length")
        
        print(f"Processing segment {i+1} for {filename}: {adjusted_start} to {adjusted_end}")
        
        # Analyze segment
        segment_name = f"segment_{i+1}"
        result = emg_analysis_segment(emg_signal, fs, adjusted_start, adjusted_end, segment_name)
        
        if result is None:
            continue
        
        # Prepare data for CSV
        max_len = max(len(result['rms_values']), len(result['median_freq_values']))
        
        # Create arrays with consistent length
        segment_data = []
        for j in range(max_len):
            row = {
                'Segment': segment_name,
                'Sample': result['time_rms'][j] if j < len(result['time_rms']) else np.nan,
                'RMS_microVolts': result['rms_values'][j] if j < len(result['rms_values']) else np.nan,
                'MedianFrequency_Hz': result['median_freq_values'][j] if j < len(result['median_freq_values']) else np.nan,
                'Linear_envelope_microVolts': result['signal_integ'] if j == 0 else np.nan,
                'Freq_Max_Hz': result['freq_max'] if j == 0 else np.nan,
                'PSD_Max': result['psd_max'] if j == 0 else np.nan
            }
            segment_data.append(row)
        
        all_results.extend(segment_data)
        
        # Create plots for this segment
        plot_and_save_figures(
            emg_signal, result['emg_filtered'], result['emg_abs'], result['emg_envelope'],
            result['time'], time_full, result['time_rms'], result['rms_values'],
            result['median_freq_values'], result['poly2_rms'], result['poly2_mdf'],
            result['freqs'], result['psd'], result['freq_max'], 
            np.argmax(result['psd']), result['signal_integ'],
            adjusted_start, adjusted_end, selections, file_output_dir, f"{filename}_{segment_name}", no_plot
        )
    
    # Save combined results to CSV
    if all_results:
        df = create_results_dataframe(all_results)
        results_file = os.path.join(file_output_dir, f"{filename}_results_emg_labiocom.csv")
        df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        
        # Create statistical summary
        create_statistical_summary(df, file_output_dir, filename)
    
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
    # Mostrar o sinal primeiro
    emg_signal = np.genfromtxt(emg_file, delimiter=",", skip_header=1, filling_values=0.0)
    emg_signal[:, 1] = emg_signal[:, 1] * 1000000
    samples = np.arange(len(emg_signal))
    emg_signal = emg_signal[:, 1]
    
    # Criar plot para visualização
    plt.figure(figsize=(15, 8))
    plt.plot(samples, emg_signal, label="Raw EMG", color="blue", linewidth=1)
    plt.title("Raw EMG Signal - Note the intervals you want to analyze")
    plt.xlabel("Sample")
    plt.ylabel("sEMG (microVolts)")
    plt.grid(True)
    plt.show()
    
    # Agora pedir os intervalos
    input_dialog = simpledialog.askstring(
        "Manual Input",
        "Enter selections as: start1,end1;start2,end2;... (e.g., 1000,5000;10000,15000):",
        initialvalue="0,1000"
    )
    
    if not input_dialog:
        return []
    
    selections = []
    try:
        pairs = input_dialog.split(';')
        for pair in pairs:
            start, end = map(int, pair.split(','))
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
        filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")]
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
        "Choose selection method:\nYes = Interactive (mouse clicks)\nNo = Manual (type values)\nCancel = Exit"
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
    
    # Select output directory (different from input)
    output_path = filedialog.askdirectory(
        title="Select Output Directory for Results",
        initialdir=os.path.dirname(input_path)  # Start from parent of input dir
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
            emg_analysis(emg_file, fs, selections, no_plot, output_dir)
            processed_files += 1
    
    if processed_files > 0:
        messagebox.showinfo(
            "Success", 
            f"EMG analysis completed!\n"
            f"{processed_files} files processed using segments from {os.path.basename(ref_file)}.\n"
            f"Results saved in: {output_dir}"
        )
    else:
        messagebox.showinfo("No Files", "No files were processed.")


def calculate_entropy(signal, window_length, overlap):
    """Calculate Sample Entropy (SampEn) for EMG signal"""
    from entropy import sample_entropy
    start = 0
    entropy_values = []
    while start + window_length < len(signal):
        window = signal[start : start + window_length]
        # Calculate Sample Entropy with m=2, r=0.2*std
        entropy = sample_entropy(window, order=2, metric='chebyshev')
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
    mfi = np.trapz(median_freq_values) / len(median_freq_values)
    
    return {
        'fatigue_index': fi,
        'modified_fatigue_index': mfi,
        'mf_slope': slope,
        'mf_intercept': intercept
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
    
    return {
        'zero_crossing_rate': zcr,
        'mean_absolute_value': mav,
        'variance': var_emg,
        'waveform_length': wl
    }


def calculate_frequency_domain_features(emg_filtered, fs):
    """Calculate frequency domain features"""
    from scipy.signal import welch
    
    # Calculate Power Spectral Density
    freqs, psd = welch(emg_filtered, fs)
    
    # Mean Frequency
    mean_freq = np.sum(freqs * psd) / np.sum(psd)
    
    # Peak Frequency
    peak_freq = freqs[np.argmax(psd)]
    
    # Spectral Moments
    moment1 = np.sum(freqs * psd) / np.sum(psd)  # First moment (mean freq)
    moment2 = np.sum((freqs - mean_freq)**2 * psd) / np.sum(psd)  # Second moment (variance)
    
    # Power Bands
    low_band = np.sum(psd[(freqs >= 20) & (freqs <= 50)])
    mid_band = np.sum(psd[(freqs > 50) & (freqs <= 100)])
    high_band = np.sum(psd[(freqs > 100) & (freqs <= 150)])
    
    return {
        'mean_frequency': mean_freq,
        'peak_frequency': peak_freq,
        'spectral_variance': moment2,
        'low_band_power': low_band,
        'mid_band_power': mid_band,
        'high_band_power': high_band
    }


def create_results_dataframe(all_results):
    """Create comprehensive results DataFrame"""
    df = pd.DataFrame(all_results)
    
    # Adicionar novas colunas para as métricas adicionais
    for result in all_results:
        if 'coordination_metrics' in result:
            for key, value in result['coordination_metrics'].items():
                df[f'Coord_{key}'] = value
        if 'frequency_features' in result:
            for key, value in result['frequency_features'].items():
                df[f'Freq_{key}'] = value
        if 'fatigue_indices' in result:
            for key, value in result['fatigue_indices'].items():
                df[f'Fatigue_{key}'] = value
    
    return df


if __name__ == "__main__":
    run_emg_gui()
