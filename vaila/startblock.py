"""
================================================================================
startblock.py - Ground Reaction Force (GRF) Analysis for Start Block
================================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 05 December 2025
Update Date: 12 January 2026
Version: 0.1.1
Python Version: 3.12.12

Description:
------------
This script performs analysis of force plate data from load cell measurements,
specifically designed for start block reaction time analysis. It detects the
initial negative peak (synchronization surge) and the return to baseline after
the positive force peak (ground push), calculating reaction time metrics.

The script processes CSV files containing force data (in grams) and time data
(in milliseconds), converting them to kilograms and analyzing key events:
1. Detects the initial negative peak (impact/synchronization surge)
2. Identifies the positive force peak (ground push)
3. Calculates reaction time from negative peak to positive peak + 100ms maximum
4. Generates comprehensive reports with plots and statistics

Features:
---------
- Single file processing: Analyze individual CSV files with interactive plots
- Batch processing: Process all CSV files in a directory automatically
- Graphical User Interface: Easy selection between single file or batch mode
- Butterworth Low-Pass Filter: Removes high-frequency noise while preserving signal characteristics
  * Configurable cutoff frequency (default: 10 Hz, typical for strain gauge data)
  * Configurable filter order (default: 4th order)
  * Zero-phase filtering using forward-backward filtering (filtfilt)
  * Edge effect mitigation through signal padding
- Comprehensive reporting: Generates HTML reports with embedded plots
- Multiple visualizations: Raw signal, filtered data, zoomed view, and positive force only
- Statistical analysis: Reaction time, peak forces, duration, area under curve
- Data export: Saves processed CSV files and high-resolution plots

Key Metrics Calculated:
-----------------------
- Reaction Time: Time from negative peak to positive peak + up to 100ms after positive peak
- Negative Peak Time: Time of initial negative surge (synchronization)
- Positive Peak Time: Time of maximum positive force (ground push)
- Force Statistics: Maximum, mean, area under curve, duration of positive force
- Baseline Analysis: Initial baseline calculation and signal characteristics

Usage:
------
1. Command Line (Single File):
   python startblock.py [caminho_do_arquivo.csv]

2. Command Line (GUI Mode):
   python startblock.py
   # Opens GUI to select single file or directory for batch processing

3. From *vailá* GUI:
   - Click "Start Block" button in Multimodal Analysis section
   - Choose between single file or batch directory processing

Input Format:
-------------
CSV file with:
- Column 0: Time in milliseconds
- Column 1: Force in grams

Output Files:
-------------
For each processed file:
1. {filename}_processed_plot.png: High-resolution plot with 4 subplots
2. {filename}_processed.csv: Processed data with converted units
3. {filename}_report.html: Comprehensive HTML report with embedded plots and statistics

Requirements:
-------------
- Python 3.12.12
- pandas: For CSV file handling
- numpy: For numerical computations
- matplotlib: For plotting and visualization
- scipy: For signal processing (trapezoid integration)
- tkinter: For GUI (usually included with Python)

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
Visit the project repository: https://github.com/vaila-multimodaltoolbox
================================================================================
"""

import base64
import sys
import tkinter as tk
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

# Try to import toml for configuration file support
try:
    import toml

    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

# Import Butterworth filter function from filter_utils module
try:
    from vaila.filter_utils import butter_filter
except ImportError:
    # Fallback: local implementation if module is not available
    from scipy.signal import butter, sosfiltfilt

    def butter_filter(data, fs, filter_type="low", cutoff=None, order=4, padding=True):
        """Applies Butterworth low-pass filter to data."""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")

        data = np.asarray(data)
        if padding:
            data_len = len(data)
            padlen = min(int(fs), data_len - 1, 15)
            if data_len > padlen:
                padded_data = np.pad(data, (padlen, padlen), mode="reflect")
                filtered_padded = sosfiltfilt(sos, padded_data, padlen=0)
                filtered_data = filtered_padded[padlen:-padlen]
            else:
                filtered_data = sosfiltfilt(sos, data, padlen=0)
        else:
            filtered_data = sosfiltfilt(sos, data, padlen=0)

        return filtered_data


def spectral_analysis_baseline(signal_data, time_data, fs, last_second=1.0):
    """
    Performs spectral analysis on the last second of baseline data to suggest cutoff frequency.

    Args:
        signal_data: Array of force signal data
        time_data: Array of time data in milliseconds
        fs: Sampling frequency in Hz
        last_second: Duration in seconds to analyze from the end (default: 1.0)

    Returns:
        dict: Dictionary with spectral analysis results including suggested cutoff frequency
    """
    # Convert time from ms to seconds
    time_seconds = time_data / 1000.0

    # Find indices for the last second
    total_duration = time_seconds[-1] - time_seconds[0]
    if total_duration < last_second:
        # If signal is shorter than last_second, use all data
        baseline_data = signal_data
        print(
            f"  Signal duration ({total_duration:.2f} s) is shorter than requested ({last_second} s). Using all data."
        )
    else:
        # Extract last second
        start_time = time_seconds[-1] - last_second
        start_idx = np.argmin(np.abs(time_seconds - start_time))
        baseline_data = signal_data[start_idx:]

    if len(baseline_data) < 10:
        print(
            f"  Warning: Baseline segment too short ({len(baseline_data)} samples). Using default cutoff."
        )
        return {
            "suggested_cutoff": 10.0,
            "dominant_frequency": None,
            "peak_frequency": None,
            "mean_frequency": None,
            "analysis_valid": False,
        }

    # Remove DC component (mean)
    baseline_data_centered = baseline_data - np.mean(baseline_data)

    # Calculate Power Spectral Density using Welch's method
    nperseg = min(256, len(baseline_data) // 4)  # Segment length for Welch method
    if nperseg < 8:
        nperseg = len(baseline_data)

    try:
        freqs, psd = welch(
            baseline_data_centered, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann"
        )
    except Exception as e:
        print(f"  Error in spectral analysis: {e}. Using default cutoff.")
        return {
            "suggested_cutoff": 10.0,
            "dominant_frequency": None,
            "peak_frequency": None,
            "mean_frequency": None,
            "analysis_valid": False,
        }

    # Find dominant frequencies (peaks in PSD)
    # Focus on frequencies up to Nyquist (fs/2)
    max_freq_idx = np.argmin(np.abs(freqs - fs / 2))
    freqs_analysis = freqs[: max_freq_idx + 1]
    psd_analysis = psd[: max_freq_idx + 1]

    # Find peak frequency (frequency with maximum power)
    peak_idx = np.argmax(psd_analysis)
    peak_frequency = freqs_analysis[peak_idx]

    # Calculate mean frequency (weighted by power)
    total_power = np.sum(psd_analysis)
    mean_frequency = np.sum(freqs_analysis * psd_analysis) / total_power if total_power > 0 else 0.0

    # Find dominant frequency (highest peak above a threshold)
    # Use 95th percentile of PSD as threshold to find significant peaks
    psd_threshold = np.percentile(psd_analysis, 95)
    significant_peaks = psd_analysis > psd_threshold
    if np.any(significant_peaks):
        dominant_idx = np.argmax(psd_analysis[significant_peaks])
        dominant_frequency = freqs_analysis[significant_peaks][dominant_idx]
    else:
        dominant_frequency = peak_frequency

    # Suggest cutoff frequency: 2-3x the dominant frequency, but cap at reasonable values
    # For strain gauge data, typical noise is below 20 Hz, so we cap at 20 Hz
    suggested_cutoff = min(max(dominant_frequency * 2.5, 5.0), 20.0)

    # If dominant frequency is very low (< 2 Hz), it might be drift, use default
    if dominant_frequency < 2.0:
        suggested_cutoff = 10.0
        print(
            f"  Low dominant frequency ({dominant_frequency:.2f} Hz) detected. May be drift. Using default cutoff."
        )

    print(f"\nSpectral Analysis Results (last {last_second} s of baseline):")
    print(f"  Peak frequency: {peak_frequency:.2f} Hz")
    print(f"  Mean frequency: {mean_frequency:.2f} Hz")
    print(f"  Dominant frequency: {dominant_frequency:.2f} Hz")
    print(f"  Suggested cutoff frequency: {suggested_cutoff:.2f} Hz")

    return {
        "suggested_cutoff": suggested_cutoff,
        "dominant_frequency": dominant_frequency,
        "peak_frequency": peak_frequency,
        "mean_frequency": mean_frequency,
        "frequencies": freqs_analysis,
        "psd": psd_analysis,
        "analysis_valid": True,
    }


def get_template_config_path():
    """Get the path to the template config file in tests/startblock/"""
    script_dir = Path(__file__).parent.parent
    template_path = script_dir / "tests" / "startblock" / "startblock_config.toml"
    return template_path


def get_default_config():
    """Get default configuration values."""
    return {
        "processing": {"mode": "batch", "input_path": "", "output_dir": ""},
        "signal": {"sampling_rate_hz": 80.0},
        "detection": {"baseline_window": 50, "negative_threshold_offset": 0.5},
        "output": {
            "show_plot": True,
            "generate_html_report": True,
            "generate_csv": True,
            "generate_plot": True,
        },
    }


def load_config(config_path):
    """
    Load configuration from TOML file.

    Args:
        config_path: Path to config file (required, no auto-search)

    Returns:
        dict: Configuration dictionary with defaults merged if file not found or invalid
    """
    defaults = get_default_config()

    if not TOML_AVAILABLE:
        return defaults

    config_path = Path(config_path)
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = toml.load(f)
            # Merge with defaults
            for key, value in defaults.items():
                if key not in config:
                    config[key] = value
                else:
                    # Merge nested dicts
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
            print(f"Configuration loaded from: {config_path}")
            return config
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration")

    return defaults


def create_config_from_template(output_path=None):
    """
    Create a config file from the template in tests/startblock/startblock_config.toml

    Args:
        output_path: Path where to save the config file. If None, saves to current directory.

    Returns:
        Path: Path to created config file, or None if failed
    """
    if not TOML_AVAILABLE:
        return None

    template_path = get_template_config_path()

    if not template_path.exists():
        print(f"Warning: Template config not found at {template_path}")
        return None

    if output_path is None:
        output_path = Path.cwd() / "startblock_config.toml"
    else:
        output_path = Path(output_path)

    try:
        # Read template
        with open(template_path, encoding="utf-8") as f:
            template_content = f.read()

        # Write to output location
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(template_content)

        print(f"Config file created from template at: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating config file: {e}")
        return None


def process_single_file(
    csv_file,
    show_plot=True,
    output_dir=None,
    sampling_rate_hz=80.0,
    baseline_window=50,
    negative_threshold_offset=0.5,
):
    """
    Process a single CSV file and generate results.

    Args:
        csv_file: Path to the CSV file
        show_plot: If True, displays the plot. If False, only saves.
        output_dir: Output directory for processed files (default: creates 'startblock_output' in same directory)
        sampling_rate_hz: Sampling rate in Hz (default: 80.0)
        baseline_window: Number of points for baseline calculation
        negative_threshold_offset: Offset from baseline to detect negative peak (in kg)

    Returns:
        dict: Dictionary with analysis results
    """
    csv_file = Path(csv_file)

    # Check if file exists
    if not csv_file.exists():
        print(f"Error: File not found: {csv_file}")
        return None

    # Create output directory
    # If output_dir is empty string or None, use same directory as input file
    if output_dir is None or output_dir == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = csv_file.parent / f"startblock_output_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory created: {output_dir}")
    print(f"Original data location: {csv_file.parent}")

    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"File loaded: {csv_file}")
        print(f"Available columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Extract columns 0 (x) and 1 (y)
    x_raw = df.iloc[:, 0].values  # First column (time) - original values
    y_raw = df.iloc[:, 1].values  # Second column (force) - raw signal in grams

    # Convert time to specified sampling rate base
    frame_time_ms = 1000.0 / sampling_rate_hz  # ms per frame

    # Convert time to frames based on sampling rate
    # If time is already in ms, convert to frames and back to precise ms
    x = np.arange(len(x_raw)) * frame_time_ms  # Time based on sampling rate
    print(f"\nTime converted to {sampling_rate_hz} Hz base ({frame_time_ms:.2f} ms per frame)")

    # Convert from grams to kilograms (divide by 1000)
    y = y_raw / 1000.0
    print("Data converted from grams to kilograms (÷ 1000)")

    print("\nOriginal data statistics (in kg):")
    print(f"  Min: {y.min():.2f} kg")
    print(f"  Max: {y.max():.2f} kg")
    print(f"  Mean: {y.mean():.2f} kg")

    # 2. Calculate initial baseline (mean of first N points)
    # Use specified baseline_window or 10% of data, whichever is smaller
    if isinstance(baseline_window, float) and baseline_window < 1.0:
        # Percentage
        baseline_window = int(len(y) * baseline_window)
    baseline_window = min(baseline_window, len(y) // 10)
    baseline_initial = np.mean(y[:baseline_window])
    print(f"\nInitial baseline (first {baseline_window} points): {baseline_initial:.2f} kg")

    # Use raw data (no filtering)
    y_data = y.copy()
    print("\nUsing raw data (no filtering applied)")

    # 3. NEW APPROACH: Find positive peak by searching from back to front
    # Search from the end of the signal backwards to find the positive peak
    print("\nSearching for positive peak from back to front...")

    # Find positive peak by searching backwards (from end to beginning)
    positive_peak_idx = None
    positive_peak_value = None
    positive_peak_time = None

    # Search from end to beginning for maximum positive value
    # Track the maximum positive value found when searching backwards
    for i in range(len(y_data) - 1, -1, -1):
        # Only consider positive values above baseline
        # Track the maximum positive value found so far (going backwards)
        if y_data[i] > baseline_initial and (positive_peak_idx is None or y_data[i] > positive_peak_value):
            positive_peak_idx = i
            positive_peak_value = y_data[i]
            positive_peak_time = x[i]

    if positive_peak_idx is None:
        # Fallback: find maximum in entire signal
        positive_peak_idx = np.argmax(y_data)
        positive_peak_value = y_data[positive_peak_idx]
        positive_peak_time = x[positive_peak_idx]
        print("   No positive value above baseline found, using global maximum")

    print("\nPositive peak detected (from back to front):")
    print(f"  Index: {positive_peak_idx}")
    print(f"  Time: {positive_peak_time:.2f} ms ({positive_peak_time / 1000:.3f} s)")
    print(f"  Value: {positive_peak_value:.2f} kg")

    # 4. Calculate derivative to find abrupt changes
    # Use gradient to calculate derivative (rate of change)
    print("\nCalculating signal derivative to detect abrupt changes...")

    # Calculate derivative: dy/dx (change in force / change in time)
    # np.gradient handles edge cases and gives central differences
    dy = np.gradient(y_data)
    dx = np.gradient(x)
    derivative = dy / dx  # Derivative in kg/ms

    print("  Derivative statistics:")
    print(f"    Min: {derivative.min():.4f} kg/ms")
    print(f"    Max: {derivative.max():.4f} kg/ms")
    print(f"    Mean: {derivative.mean():.4f} kg/ms")

    # 5. Find surge start: point of maximum negative derivative (most abrupt negative change)
    # Search before positive peak for the most negative derivative
    print("\nSearching for surge start (maximum negative derivative) before positive peak...")

    # Limit search to region before positive peak
    search_end = min(positive_peak_idx, len(derivative))
    if search_end > 0:
        # Find index with most negative derivative (steepest downward slope)
        derivative_before_peak = derivative[:search_end]
        surge_start_idx = np.argmin(derivative_before_peak)  # Most negative = steepest drop
        surge_start_time = x[surge_start_idx]
        surge_start_derivative = derivative[surge_start_idx]
    else:
        # Fallback
        surge_start_idx = 0
        surge_start_time = x[0]
        surge_start_derivative = derivative[0]
        print("  Using first point as surge start")

    print("\nSurge start (maximum negative derivative) detected:")
    print(f"  Index: {surge_start_idx}")
    print(f"  Time: {surge_start_time:.2f} ms ({surge_start_time / 1000:.3f} s)")
    print(f"  Value: {y_data[surge_start_idx]:.2f} kg")
    print(f"  Derivative: {surge_start_derivative:.4f} kg/ms (most negative)")

    # 6. Find the minimum point (bottom of the negative U)
    print("\nFinding minimum point (bottom of negative U)...")

    # Search between surge start and positive peak for the minimum
    search_start = surge_start_idx
    search_end = min(positive_peak_idx + 1, len(y_data))
    if search_end > search_start:
        search_range = y_data[search_start:search_end]
        min_idx_local = np.argmin(search_range)
        minimum_idx = search_start + min_idx_local
        minimum_time = x[minimum_idx]
        minimum_value = y_data[minimum_idx]
    else:
        minimum_idx = surge_start_idx
        minimum_time = surge_start_time
        minimum_value = y_data[minimum_idx]

    print("  Minimum point:")
    print(f"    Index: {minimum_idx}")
    print(f"    Time: {minimum_time:.2f} ms ({minimum_time / 1000:.3f} s)")
    print(f"    Value: {minimum_value:.2f} kg")

    # 7. Find end of surge: point of maximum positive derivative AFTER the minimum
    # This marks where the signal starts rising most rapidly after the bottom of the U
    print("\nSearching for end of surge (maximum positive derivative after minimum)...")

    end_surge_idx = None
    end_surge_time = None

    # Search after minimum point, before positive peak
    search_start = minimum_idx + 1
    search_end = min(positive_peak_idx + 1, len(derivative))

    if search_end > search_start:
        # Find index with most positive derivative (steepest upward slope)
        derivative_after_min = derivative[search_start:search_end]
        if len(derivative_after_min) > 0:
            max_deriv_idx_local = np.argmax(derivative_after_min)  # Most positive = steepest rise
            end_surge_idx = search_start + max_deriv_idx_local
            end_surge_time = x[end_surge_idx]
            end_surge_derivative = derivative[end_surge_idx]
        else:
            end_surge_idx = minimum_idx
            end_surge_time = minimum_time
            end_surge_derivative = derivative[minimum_idx]
    else:
        # Fallback: use point just after minimum
        end_surge_idx = min(minimum_idx + 1, len(y_data) - 1)
        end_surge_time = x[end_surge_idx]
        end_surge_derivative = derivative[end_surge_idx]
        print("  Using point after minimum as end surge")

    print("\nEnd of negative surge (maximum positive derivative after minimum) detected:")
    print(f"  Index: {end_surge_idx}")
    print(f"  Time: {end_surge_time:.2f} ms ({end_surge_time / 1000:.3f} s)")
    print(f"  Value: {y_data[end_surge_idx]:.2f} kg")
    print(f"  Derivative: {end_surge_derivative:.4f} kg/ms (most positive after minimum)")

    # 8. Calculate Reaction Times
    # Reaction Time 1: from surge start (first negative) to positive peak
    reaction_time_surge_to_peak = positive_peak_time - surge_start_time

    # Reaction Time 2: from end of surge to positive peak
    reaction_time_end_surge_to_peak = positive_peak_time - end_surge_time

    print(f"\n{'=' * 60}")
    print("Reaction Time Analysis:")
    print(f"{'=' * 60}")
    print(
        f"  Surge start (first negative): {surge_start_time:.2f} ms ({surge_start_time / 1000:.3f} s)"
    )
    print(f"  End of surge: {end_surge_time:.2f} ms ({end_surge_time / 1000:.3f} s)")
    print(f"  Positive peak: {positive_peak_time:.2f} ms ({positive_peak_time / 1000:.3f} s)")
    print(
        f"\n  Reaction Time (Surge Start → Peak): {reaction_time_surge_to_peak:.2f} ms ({reaction_time_surge_to_peak / 1000:.3f} s)"
    )
    print(
        f"  Reaction Time (End Surge → Peak): {reaction_time_end_surge_to_peak:.2f} ms ({reaction_time_end_surge_to_peak / 1000:.3f} s)"
    )
    print(f"{'=' * 60}")

    # Create figure with subplots (4 plots)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))

    # Plot 1: Raw signal (in grams)
    ax0 = axes[0]
    ax0.plot(x, y_raw, "k-", linewidth=2, label="Raw signal (grams)")
    ax0.set_xlabel(df.columns[0] if len(df.columns) > 0 else "Time (ms)", fontsize=12)
    ax0.set_ylabel(df.columns[1] + " (g)" if len(df.columns) > 1 else "Force (g)", fontsize=12)
    ax0.set_title("Raw Signal - Original Data in Grams", fontsize=14, fontweight="bold")
    ax0.legend(loc="best", fontsize=10)
    ax0.grid(True, alpha=0.3)

    # Plot 2: Complete data with markers
    ax1 = axes[1]
    ax1.plot(x, y_data, "b-", linewidth=1.5, label="Raw data (kg)")
    ax1.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5, label="Zero")
    ax1.axvline(
        x=surge_start_time,
        color="brown",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Surge start (max neg derivative)",
    )
    ax1.axvline(
        x=minimum_time,
        color="orange",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
        label="Minimum (bottom of U)",
    )
    ax1.axvline(
        x=end_surge_time,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="End of surge (max pos derivative)",
    )
    ax1.axvline(
        x=positive_peak_time,
        color="purple",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label="Positive peak",
    )
    # Add baseline line
    ax1.axhline(
        y=baseline_initial,
        color="green",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Initial baseline",
    )
    # Mark minimum point
    ax1.plot(minimum_time, minimum_value, "o", color="orange", markersize=8, alpha=0.7)

    # Add reaction time text
    reaction_time_text = f"Reaction Time (Surge Start → Peak): {reaction_time_surge_to_peak:.2f} ms ({reaction_time_surge_to_peak / 1000:.3f} s)\nReaction Time (End Surge → Peak): {reaction_time_end_surge_to_peak:.2f} ms ({reaction_time_end_surge_to_peak / 1000:.3f} s)"
    ax1.text(
        0.02,
        0.02,
        reaction_time_text,
        transform=ax1.transAxes,
        fontsize=11,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.7},
    )
    ax1.set_xlabel(df.columns[0] if len(df.columns) > 0 else "Time (ms)", fontsize=12)
    ax1.set_ylabel(df.columns[1] + " (kg)" if len(df.columns) > 1 else "Force (kg)", fontsize=12)
    ax1.set_title("Load Cell Analysis - Complete Data", fontsize=14, fontweight="bold")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 3: Zoom on region of interest (from surge start to peak)
    ax2 = axes[2]
    start_idx = max(0, surge_start_idx - 20)
    end_idx = min(len(y_data), positive_peak_idx + 20)
    ax2.plot(
        x[start_idx:end_idx], y_data[start_idx:end_idx], "b-", linewidth=2, label="Raw data (kg)"
    )
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5, label="Zero")
    ax2.axhline(
        y=baseline_initial,
        color="green",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Initial baseline",
    )
    ax2.plot(
        surge_start_time,
        y_data[surge_start_idx],
        "s",
        color="brown",
        markersize=12,
        label="Surge start (max neg derivative)",
    )
    ax2.plot(
        minimum_time,
        minimum_value,
        "o",
        color="orange",
        markersize=12,
        label=f"Minimum ({minimum_value:.2f} kg)",
    )
    ax2.plot(
        end_surge_time,
        y_data[end_surge_idx],
        "s",
        color="red",
        markersize=12,
        label="End of surge (max pos derivative)",
    )
    ax2.plot(
        positive_peak_time,
        y_data[positive_peak_idx],
        "o",
        color="purple",
        markersize=12,
        label=f"Positive peak ({positive_peak_value:.2f} kg)",
    )
    ax2.set_xlabel(df.columns[0] if len(df.columns) > 0 else "Time (ms)", fontsize=12)
    ax2.set_ylabel(df.columns[1] + " (kg)" if len(df.columns) > 1 else "Force (kg)", fontsize=12)
    ax2.set_title(
        "Zoom: Surge Start → Minimum → End Surge → Positive Peak", fontsize=14, fontweight="bold"
    )
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Optional: Add derivative plot as subplot or overlay
    # We could add a 5th subplot showing derivative, but for now keep it simple

    # Plot 4: Positive force from end surge to peak
    ax3 = axes[3]
    # Extract positive data from end surge to peak
    positive_force_start = end_surge_idx
    positive_force_end = positive_peak_idx + 1
    positive_force_x = x[positive_force_start:positive_force_end]
    positive_force_y = y_data[positive_force_start:positive_force_end]

    # Filter only values above initial baseline
    positive_mask = positive_force_y > baseline_initial
    positive_force_x_filtered = positive_force_x[positive_mask]
    positive_force_y_filtered = positive_force_y[positive_mask]

    if len(positive_force_x_filtered) > 0:
        ax3.plot(
            positive_force_x_filtered,
            positive_force_y_filtered,
            "g-",
            linewidth=2.5,
            label="Positive force",
        )
        ax3.fill_between(
            positive_force_x_filtered,
            0,
            positive_force_y_filtered,
            alpha=0.3,
            color="green",
            label="Area under curve",
        )
        ax3.axhline(y=0, color="k", linestyle="-", linewidth=1, alpha=0.5)
        ax3.plot(
            positive_peak_time,
            positive_peak_value,
            "o",
            color="purple",
            markersize=12,
            label=f"Positive peak ({positive_peak_value:.2f} kg)",
        )

        # Calculate statistics
        positive_force_max = np.max(positive_force_y_filtered)
        positive_force_mean = np.mean(positive_force_y_filtered)
        positive_force_area = np.trapezoid(positive_force_y_filtered, positive_force_x_filtered)
        positive_duration = positive_force_x_filtered[-1] - positive_force_x_filtered[0]

        # Create text with statistics
        stats_text = f"Max: {positive_force_max:.2f} kg | Mean: {positive_force_mean:.2f} kg | Area: {positive_force_area:.2f} kg·ms | Duration: {positive_duration:.2f} ms ({positive_duration / 1000:.3f} s)"

        # Add statistics text
        ax3.text(
            0.02,
            0.98,
            stats_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        print("\nPositive Force Statistics:")
        print(f"  Maximum force: {positive_force_max:.2f} kg")
        print(f"  Mean force: {positive_force_mean:.2f} kg")
        print(f"  Area under curve: {positive_force_area:.2f} kg·ms")
        print(f"  Duration: {positive_duration:.2f} ms ({positive_duration / 1000:.3f} s)")
    else:
        positive_force_max = 0
        positive_force_mean = 0
        positive_force_area = 0
        positive_duration = 0
        ax3.text(
            0.5,
            0.5,
            "No positive force detected",
            transform=ax3.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    ax3.set_xlabel(df.columns[0] if len(df.columns) > 0 else "Time (ms)", fontsize=12)
    ax3.set_ylabel(
        df.columns[1] + " (kg)" if len(df.columns) > 1 else "Positive Force (kg)", fontsize=12
    )
    ax3.set_title("Positive Force (End Surge → Peak)", fontsize=14, fontweight="bold")
    ax3.legend(loc="best", fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot to output directory
    output_plot = output_dir / f"{csv_file.stem}_processed_plot.png"
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_plot}")

    # Save processed data to output directory
    output_csv = output_dir / f"{csv_file.stem}_processed.csv"
    df_processed = pd.DataFrame({df.columns[0]: x, df.columns[1]: y_data})
    df_processed.to_csv(output_csv, index=False)
    print(f"Processed data saved to: {output_csv}")

    # Prepare results dictionary for report
    results_dict = {
        "reaction_time_surge_to_peak": reaction_time_surge_to_peak,
        "reaction_time_end_surge_to_peak": reaction_time_end_surge_to_peak,
        "peak_positive": positive_peak_value,
        "positive_force_max": positive_force_max,
        "positive_force_mean": positive_force_mean,
        "positive_force_area": positive_force_area,
        "positive_duration": positive_duration,
        "surge_start_time": surge_start_time,
        "end_surge_time": end_surge_time,
        "positive_peak_time": positive_peak_time,
        "baseline_initial": baseline_initial,
    }

    # Generate HTML report in output directory
    html_output = output_dir / f"{csv_file.stem}_report.html"
    generate_html_report(fig, csv_file, results_dict, html_output, baseline_initial, x)

    # Mostrar o plot apenas se solicitado
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return results_dict


def generate_html_report(fig, csv_file, results_dict, output_file, baseline_initial, x):
    """Generates an HTML report with analysis results."""

    # Save current figure to buffer
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    # Prepare statistics
    stats_html = f"""
    <div class="stats-grid">
        <div class="stat-card">
            <h3>Reaction Time (Surge Start → Peak)</h3>
            <p class="stat-value">{results_dict["reaction_time_surge_to_peak"]:.2f} ms</p>
            <p class="stat-value-small">({results_dict["reaction_time_surge_to_peak"] / 1000:.3f} s)</p>
        </div>
        <div class="stat-card">
            <h3>Reaction Time (End Surge → Peak)</h3>
            <p class="stat-value">{results_dict["reaction_time_end_surge_to_peak"]:.2f} ms</p>
            <p class="stat-value-small">({results_dict["reaction_time_end_surge_to_peak"] / 1000:.3f} s)</p>
        </div>
        <div class="stat-card">
            <h3>Positive Peak</h3>
            <p class="stat-value">{results_dict["peak_positive"]:.2f} kg</p>
        </div>
        <div class="stat-card">
            <h3>Maximum Positive Force</h3>
            <p class="stat-value">{results_dict["positive_force_max"]:.2f} kg</p>
        </div>
        <div class="stat-card">
            <h3>Mean Positive Force</h3>
            <p class="stat-value">{results_dict["positive_force_mean"]:.2f} kg</p>
        </div>
        <div class="stat-card">
            <h3>Area Under Curve</h3>
            <p class="stat-value">{results_dict["positive_force_area"]:.2f} kg·ms</p>
        </div>
        <div class="stat-card">
            <h3>Positive Force Duration</h3>
            <p class="stat-value">{results_dict["positive_duration"]:.2f} ms</p>
        </div>
    </div>
    """

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ground Reaction Force Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .info-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .info-section h2 {{
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .info-item {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .info-item strong {{
            color: #667eea;
            display: block;
            margin-bottom: 5px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-card h3 {{
            font-size: 0.9em;
            margin-bottom: 10px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 5px 0;
        }}
        .stat-value-small {{
            font-size: 1em;
            opacity: 0.8;
        }}
        .plot-section {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-section img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #ddd;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Ground Reaction Force Analysis Report</h1>
            <p>Load Cell Analysis - Ground Reaction Force (GRF)</p>
        </div>
        <div class="content">
            <div class="info-section">
                <h2>📁 File Information</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Analyzed File:</strong>
                        {csv_file.name}
                    </div>
                    <div class="info-item">
                        <strong>Directory:</strong>
                        {csv_file.parent}
                    </div>
                    <div class="info-item">
                        <strong>Total Samples:</strong>
                        {len(x)}
                    </div>
                    <div class="info-item">
                        <strong>Total Duration:</strong>
                        {x[-1] - x[0]:.2f} ms
                    </div>
                </div>
            </div>

            <div class="info-section">
                <h2>📈 Main Statistics</h2>
                {stats_html}
            </div>

            <div class="info-section">
                <h2>🔍 Analysis Details</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>Surge Start (First Negative):</strong>
                        {results_dict["surge_start_time"]:.2f} ms ({results_dict["surge_start_time"] / 1000:.3f} s)
                    </div>
                    <div class="info-item">
                        <strong>End of Surge:</strong>
                        {results_dict["end_surge_time"]:.2f} ms ({results_dict["end_surge_time"] / 1000:.3f} s)
                    </div>
                    <div class="info-item">
                        <strong>Positive Peak:</strong>
                        {results_dict["positive_peak_time"]:.2f} ms ({results_dict["peak_positive"]:.2f} kg)
                    </div>
                    <div class="info-item">
                        <strong>Initial Baseline:</strong>
                        {baseline_initial:.2f} kg
                    </div>
                    <div class="info-item" style="background: #fff3cd; border-left-color: #ffc107;">
                        <strong> Reaction Times:</strong>
                        <br>Surge Start → Peak: {results_dict["reaction_time_surge_to_peak"]:.2f} ms ({results_dict["reaction_time_surge_to_peak"] / 1000:.3f} s)
                        <br>End Surge → Peak: {results_dict["reaction_time_end_surge_to_peak"]:.2f} ms ({results_dict["reaction_time_end_surge_to_peak"] / 1000:.3f} s)
                    </div>
                </div>
            </div>

            <div class="plot-section">
                <h2 style="color: #667eea; margin-bottom: 20px;">📉 Analysis Plots</h2>
                <img src="data:image/png;base64,{plot_data}" alt="Analysis Plots">
            </div>
        </div>
        <div class="footer">
            <p>Report automatically generated by <i>vailá</i> - Multimodal Toolbox</p>
            <p class="timestamp">Date: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
        </div>
    </div>
</body>
</html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report saved to: {output_file}")


def batch_process_directory(
    directory_path,
    sampling_rate_hz=80.0,
    baseline_window=50,
    negative_threshold_offset=0.5,
    output_dir=None,
):
    """
    Process all CSV files in a directory.

    Args:
        directory_path: Path to directory containing CSV files
        sampling_rate_hz: Sampling rate in Hz (default: 80.0)
        baseline_window: Number of points for baseline calculation
        negative_threshold_offset: Offset from baseline to detect negative peak (in kg)
        output_dir: Output directory (if None, creates timestamped directory)
    """
    directory_path = Path(directory_path)

    if not directory_path.is_dir():
        print(f"Error: {directory_path} is not a valid directory")
        return

    # Find all CSV files in directory
    csv_files = sorted([f for f in directory_path.glob("*.csv") if not f.name.startswith(".")])

    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    # Create main output directory
    # If output_dir is empty string or None, use same directory as input directory
    if output_dir is None or output_dir == "":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = directory_path / f"startblock_output_batch_{timestamp}"
    else:
        main_output_dir = Path(output_dir)
    main_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nMain output directory: {main_output_dir}")
    print(f"Original data location: {directory_path}")

    print(f"\n{'=' * 60}")
    print(f"Batch Processing - {len(csv_files)} file(s) found")
    print(f"{'=' * 60}\n")

    processed_count = 0
    failed_count = 0

    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {csv_file.name}")
        print("-" * 60)

        try:
            # Each file gets its own subdirectory in the main output directory
            file_output_dir = main_output_dir / csv_file.stem
            result = process_single_file(
                csv_file,
                show_plot=False,
                output_dir=file_output_dir,
                sampling_rate_hz=sampling_rate_hz,
                baseline_window=baseline_window,
                negative_threshold_offset=negative_threshold_offset,
            )
            if result:
                processed_count += 1
                print(f"✓ Successfully processed: {csv_file.name}")
            else:
                failed_count += 1
                print(f"✗ Failed to process: {csv_file.name}")
        except Exception as e:
            failed_count += 1
            print(f"✗ Error processing {csv_file.name}: {e}")

    print(f"\n{'=' * 60}")
    print("Batch processing completed!")
    print(f"  ✓ Successfully processed: {processed_count}")
    print(f"  ✗ Failures: {failed_count}")
    print(f"  Output directory: {main_output_dir}")
    print(f"{'=' * 60}\n")


def run_startblock_gui():
    """
    Simple GUI: only path selection and TOML config management.
    All processing configuration comes from TOML file.
    """
    root = Tk()
    root.title("Start Block Analysis")
    root.geometry("500x600")

    # Main frame
    main_frame = ttk.Frame(root, padding="20")
    main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Config file section
    config_frame = ttk.LabelFrame(main_frame, text="Configuration File (TOML)", padding="10")
    config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

    # Variable to track current config file path
    current_config_path = tk.StringVar(value="")
    config_path_label = ttk.Label(
        config_frame, text="No config loaded", foreground="gray", wraplength=350
    )
    config_path_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

    def load_config_file():
        """Load config from any TOML file chosen by user"""
        config_file = filedialog.askopenfilename(
            title="Select configuration file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if config_file:
            try:
                config_path = Path(config_file)
                if config_path.exists():
                    current_config_path.set(str(config_path))
                    config_path_label.config(
                        text=f"Loaded: {config_path.name}\n{config_path.parent}", foreground="green"
                    )
                    messagebox.showinfo("Success", f"Configuration loaded from:\n{config_file}")
                else:
                    messagebox.showerror("Error", f"File not found:\n{config_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config file:\n{e}")

    def create_config():
        """Create config file from template"""
        output_path = filedialog.asksaveasfilename(
            title="Save config file as",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if output_path:
            created = create_config_from_template(output_path)
            if created:
                current_config_path.set(str(created))
                config_path_label.config(
                    text=f"Created: {Path(created).name}\n{Path(created).parent}", foreground="blue"
                )
                messagebox.showinfo("Success", f"Config file created at:\n{created}")
            else:
                messagebox.showerror("Error", "Failed to create config file")

    ttk.Button(config_frame, text="Load Config File", command=load_config_file, width=20).grid(
        row=1, column=0, padx=5, pady=5
    )
    ttk.Button(config_frame, text="Create Config File", command=create_config, width=20).grid(
        row=1, column=1, padx=5, pady=5
    )

    # Path selection section
    path_frame = ttk.LabelFrame(main_frame, text="Select Input Path", padding="10")
    path_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

    input_path_var = tk.StringVar(value="")
    input_path_label = ttk.Label(
        path_frame, text="No path selected", foreground="gray", wraplength=350
    )
    input_path_label.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)

    def select_file():
        """Select single CSV file"""
        csv_file = filedialog.askopenfilename(
            title="Select CSV file for analysis",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if csv_file:
            input_path_var.set(csv_file)
            input_path_label.config(
                text=f"File: {Path(csv_file).name}\n{Path(csv_file).parent}", foreground="green"
            )

    def select_directory():
        """Select directory for batch processing"""
        directory = filedialog.askdirectory(
            title="Select directory with CSV files for batch processing"
        )
        if directory:
            input_path_var.set(directory)
            input_path_label.config(
                text=f"Directory: {Path(directory).name}\n{Path(directory).parent}",
                foreground="green",
            )

    ttk.Button(path_frame, text="Select File", command=select_file, width=20).grid(
        row=1, column=0, padx=5, pady=5
    )
    ttk.Button(path_frame, text="Select Directory", command=select_directory, width=20).grid(
        row=1, column=1, padx=5, pady=5
    )

    # Run button
    def run_analysis():
        """Run analysis using config TOML and selected path"""
        config_file = current_config_path.get()
        input_path = input_path_var.get()

        if not config_file:
            messagebox.showerror("Error", "Please load or create a configuration file first!")
            return

        if not input_path:
            messagebox.showerror("Error", "Please select an input file or directory!")
            return

        # Load config
        try:
            config = load_config(config_file)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config:\n{e}")
            return

        root.withdraw()  # Hide window during processing

        try:
            input_path_obj = Path(input_path)

            # Get output_dir from config
            # If empty string, use same directory as input (will create timestamped subdirectory)
            # If specified, use that path
            output_dir_config = config["processing"].get("output_dir", "")
            if output_dir_config == "":
                # Empty string means use same directory as input (None triggers default behavior)
                output_dir_config = None
            elif output_dir_config:
                # Use the configured path
                output_dir_config = str(output_dir_config)
            else:
                # None or not specified, use default behavior
                output_dir_config = None

            if input_path_obj.is_file():
                # Single file mode
                process_single_file(
                    input_path_obj,
                    show_plot=config["output"].get("show_plot", False),
                    sampling_rate_hz=config["signal"].get("sampling_rate_hz", 80.0),
                    baseline_window=config["detection"].get("baseline_window", 50),
                    negative_threshold_offset=config["detection"].get(
                        "negative_threshold_offset", 0.5
                    ),
                    output_dir=output_dir_config,
                )
                messagebox.showinfo("Success", "Processing completed!")
            elif input_path_obj.is_dir():
                # Batch mode
                batch_process_directory(
                    input_path_obj,
                    sampling_rate_hz=config["signal"].get("sampling_rate_hz", 80.0),
                    baseline_window=config["detection"].get("baseline_window", 50),
                    negative_threshold_offset=config["detection"].get(
                        "negative_threshold_offset", 0.5
                    ),
                    output_dir=output_dir_config,
                )
                messagebox.showinfo("Success", "Batch processing completed!")
            else:
                messagebox.showerror("Error", f"Invalid path: {input_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing:\n{e}")
        finally:
            root.destroy()

    button_frame = ttk.Frame(main_frame)
    button_frame.grid(row=2, column=0, columnspan=2, pady=20)

    ttk.Button(button_frame, text="Run Analysis", command=run_analysis, width=25).grid(
        row=0, column=0, padx=5
    )
    ttk.Button(button_frame, text="Cancel", command=root.destroy, width=25).grid(
        row=0, column=1, padx=5
    )

    root.mainloop()


if __name__ == "__main__":
    # If a file was passed as argument, process directly with defaults
    if len(sys.argv) > 1:
        csv_file = Path(sys.argv[1])
        if csv_file.exists():
            # Use default values
            defaults = get_default_config()
            process_single_file(
                csv_file,
                show_plot=defaults["output"].get("show_plot", True),
                sampling_rate_hz=defaults["signal"].get("sampling_rate_hz", 80.0),
                baseline_window=defaults["detection"].get("baseline_window", 50),
                negative_threshold_offset=defaults["detection"].get(
                    "negative_threshold_offset", 0.5
                ),
            )
        else:
            print(f"Error: File not found: {csv_file}")
            sys.exit(1)
    else:
        # Open GUI (no auto-loading)
        run_startblock_gui()
