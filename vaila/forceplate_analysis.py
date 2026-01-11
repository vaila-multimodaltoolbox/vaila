"""
================================================================================
forceplate_analysis.py
================================================================================
Author: Prof. Paulo Santiago
Date: 9 September 2024
Update: 30 December 2025
Version: 2.0

Description:
------------
This script serves as the central control interface for the VAILA (Virtual Analysis
for Interactive Learning in Biomechanics) toolbox, providing a user-friendly graphical
user interface (GUI) for selecting and executing various biomechanical analysis scripts.

The analyses offered in this toolbox support the study and evaluation of postural control,
dynamic balance, gait data, and force measurements, which are critical in both clinical
and research settings.

Key Analyses Supported:
-----------------------
1. Force Cube Analysis: Examines force data captured in a cubic arrangement, allowing for
   multidirectional force vector analysis.
2. Center of Pressure (CoP) Balance Analysis: Evaluates postural stability by analyzing
   the center of pressure data, providing insights into balance control and sway
   characteristics. Includes:
   - Stabilometric analysis (RMS, MSD, PSD, sway density, total path length)
   - Spectral feature analysis (total power, frequency dispersion, energy content)
   - 2D RMS (DRMS) calculation for total sway area magnitude
   - Heatmap visualizations (KDE and histogram methods with topographic contours)
3. Force CMJ (Countermovement Jump) Analysis: Analyzes the forces involved in a
   countermovement jump to assess athletic performance, muscle power, and explosiveness.
4. Noise Signal Fixing: Identifies and corrects noise artifacts in force signals, ensuring
   data accuracy for subsequent analyses.
5. Calculate CoP: Executes a process to calculate the CoP data from force and moment data.
6. Gait Analysis (Single Strike): Analyzes a single contact strike in gait data, calculating
   key metrics such as peak force, impulse, and rate of force development.

This module unifies functionality previously distributed across multiple separate modules:
- cop_analysis.py (now integrated)
- cop_calculate.py (now integrated)
- stabilogram_analysis.py (now integrated)
- spectral_features.py (now integrated)

All CoP-related analyses are now self-contained within this module.

Functionalities:
----------------
1. GUI for Analysis Selection: Utilizes Python's Tkinter library to present a straightforward
   interface where users can choose their desired analysis with ease.
2. Self-Contained CoP Analysis: All CoP balance and calculation functions are integrated
   directly into this module, providing a single-file solution for force plate analyses.
3. Dynamic Module Importing: For non-CoP analyses (force_cube_fig, force_cmj, etc.),
   efficiently loads only the necessary module for the selected analysis.
4. Execution of Analysis: Automatically runs the selected analysis, guiding users through
   any necessary data input or parameter settings.

Modules and Packages Required:
------------------------------
- Python Standard Libraries: tkinter for GUI creation and management.
- External Libraries: numpy, pandas, matplotlib, scipy, sklearn (for PCA in ellipse calculations)
- VAILA Toolbox Modules (external, imported only when needed):
  * force_cube_fig: For analyzing force cube data.
  * force_cmj: For analyzing countermovement jump dynamics.
  * fixnoise: For correcting noise in force data.
  * grf_gait: For analyzing single strike gait data.
  * sit2stand: For sit-to-stand analysis.
- VAILA Toolbox Modules (internal, imported for CoP analysis):
  * ellipse: For confidence ellipse calculations using PCA.
  * filter_utils: For Butterworth filtering.

How to Use:
-----------
1. Run the Script:
   Execute the script from the terminal using:
   python -m vaila.forceplate_analysis
   Alternatively, double-click the script file if your environment supports it.

2. Select Analysis:
   A GUI window will appear, prompting you to select the type of analysis to perform. Click on
   the corresponding button for the desired analysis.

3. Follow Instructions:
   The selected analysis will start, and you will be guided through the necessary steps, such as
   choosing data files or entering parameters. Follow the on-screen instructions to complete the
   analysis.

License:
--------
This script is licensed under the MIT License. For more details, please refer to the LICENSE file
located in the project root.

Disclaimer:
-----------
This script is provided "as is," without any warranty, express or implied. The authors are not
liable for any damage or data loss resulting from the use of this script. It is intended solely for
academic and research purposes.

Changelog:
----------
- 2025-12-30 (Version 2.0):
  - Unified cop_analysis.py, cop_calculate.py, stabilogram_analysis.py, and spectral_features.py
    into this single module.
  - Added 2D RMS (DRMS) calculation for total sway area magnitude.
  - Added heatmap visualizations (KDE and histogram methods with topographic contours).
  - All CoP-related analyses are now self-contained within this module.
  - Removed external dependencies on separate CoP analysis modules.
- 2024-09-09: Initial creation of the script with dynamic analysis selection functionality.
- 2024-09-10: Added support for CoP Balance Analysis (cop_analysis.py).
- 2024-09-14: Added "Calculate CoP" button and functionality (cop_calculate.py).
- 2024-11-27: Added "Gait Analysis (Single Strike)" button and functionality (gait_analysis.py).
- 2025-10-10: Added "Sit to Stand" button and functionality (sit_to_stand.py).
================================================================================
"""

import os
from tkinter import (
    BooleanVar,
    Button,
    Canvas,
    Checkbutton,
    Frame,
    Label,
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
from scipy import stats
from scipy.interpolate import griddata
from scipy.signal import find_peaks, savgol_filter, welch

from .ellipse import plot_cop_pathway_with_ellipse, plot_ellipse_pca
from .filter_utils import butter_filter

# ============================================================================
# GUI FUNCTIONS - Analysis Selection
# ============================================================================


def choose_analysis_type():
    """
    Opens a GUI to choose which analysis code to run.
    """
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    choice = []

    def select_force_cube_fig():
        choice.append("force_cube_fig")
        choice_window.quit()
        choice_window.destroy()

    def select_cop_balance():
        choice.append("cop_balance")
        choice_window.quit()
        choice_window.destroy()

    def select_force_cmj():
        choice.append("force_cmj")
        choice_window.quit()
        choice_window.destroy()

    def select_fix_noise():
        choice.append("fix_noise")
        choice_window.quit()
        choice_window.destroy()

    def select_calculate_cop():
        choice.append("calculate_cop")
        choice_window.quit()
        choice_window.destroy()

    def select_gait_analysis():
        choice.append("gait_analysis")
        choice_window.quit()
        choice_window.destroy()

    def select_sit_to_stand():
        choice.append("sit_to_stand")
        choice_window.quit()
        choice_window.destroy()

    choice_window = Toplevel()
    choice_window.title("Choose Analysis Type")
    choice_window.geometry("300x500")

    Label(choice_window, text="Select which analysis to run:").pack(pady=17)

    btn_force_cube = Button(
        choice_window, text="Force Cube Analysis", command=select_force_cube_fig
    )
    btn_force_cube.pack(pady=5)

    btn_cop_balance = Button(
        choice_window, text="CoP Balance Analysis", command=select_cop_balance
    )
    btn_cop_balance.pack(pady=5)

    btn_force_cmj = Button(choice_window, text="Force CMJ Analysis", command=select_force_cmj)
    btn_force_cmj.pack(pady=5)

    btn_fix_noise = Button(choice_window, text="Fix Noise Signal", command=select_fix_noise)
    btn_fix_noise.pack(pady=5)

    btn_calculate_cop = Button(choice_window, text="Calculate CoP", command=select_calculate_cop)
    btn_calculate_cop.pack(pady=5)

    btn_gait_analysis = Button(
        choice_window,
        text="Gait Analysis (Fz vertical)",
        command=select_gait_analysis,
    )
    btn_gait_analysis.pack(pady=7)

    btn_sit_to_stand = Button(choice_window, text="Sit to Stand", command=select_sit_to_stand)
    btn_sit_to_stand.pack(pady=7)

    choice_window.mainloop()

    return choice[0] if choice else None


# ============================================================================
# UTILITY FUNCTIONS FOR CoP ANALYSIS
# ============================================================================


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
        raise Exception(f"Error reading the CSV file: {str(e)}") from e


# ============================================================================
# STABILOMETRIC ANALYSIS FUNCTIONS
# ============================================================================


def compute_rms(cop_x, cop_y):
    """
    Calculates the RMS displacement in the ML and AP directions.

    Parameters:
    - cop_x: array-like, CoP data in the ML direction.
    - cop_y: array-like, CoP data in the AP direction.

    Returns:
    - rms_ml: float, RMS displacement in the ML direction.
    - rms_ap: float, RMS displacement in the AP direction.
    """
    rms_ml = np.sqrt(np.mean(cop_x**2))
    rms_ap = np.sqrt(np.mean(cop_y**2))
    return rms_ml, rms_ap


def compute_drms(cop_x, cop_y):
    """
    Calculates the 2D RMS (DRMS - Distance Root Mean Square) combining ML and AP directions.
    Provides a single global measure of postural sway magnitude.
    This represents the RMS of the total sway area (not separated into ML and AP).

    Parameters:
    - cop_x: array-like, CoP data in the ML direction (centered).
    - cop_y: array-like, CoP data in the AP direction (centered).

    Returns:
    - drms: float, 2D RMS value in cm, representing the magnitude of total sway area.
    """
    rms_ml = np.sqrt(np.mean(cop_x**2))
    rms_ap = np.sqrt(np.mean(cop_y**2))
    drms = np.sqrt(rms_ml**2 + rms_ap**2)
    return drms


def compute_speed(cop_x, cop_y, fs, window_length=5, polyorder=3):
    """
    Calculates the speed of the CoP signal using the Savitzky-Golay filter.

    Parameters:
    - cop_x: array-like, CoP data in the ML direction.
    - cop_y: array-like, CoP data in the AP direction.
    - fs: float, Sampling frequency in Hz.
    - window_length: int, default=5, Length of the filter window.
    - polyorder: int, default=3, Order of the polynomial.

    Returns:
    - speed_ml: array-like, Speed in the ML direction.
    - speed_ap: array-like, Speed in the AP direction.
    """
    delta = 1 / fs
    window_length = min(window_length, len(cop_x) // 2 * 2 - 1)
    if window_length % 2 == 0:
        window_length += 1
    speed_ml = savgol_filter(cop_x, window_length, polyorder, deriv=1, delta=delta)
    speed_ap = savgol_filter(cop_y, window_length, polyorder, deriv=1, delta=delta)
    return speed_ml, speed_ap


def compute_power_spectrum(cop_x, cop_y, fs):
    """
    Calculates the Power Spectral Density (PSD) of the CoP signals.

    Parameters:
    - cop_x: array-like, CoP data in the ML direction.
    - cop_y: array-like, CoP data in the AP direction.
    - fs: float, Sampling frequency in Hz.

    Returns:
    - freqs_ml, psd_ml, freqs_ap, psd_ap: Power spectral density data.
    """
    freqs_ml, psd_ml = welch(cop_x, fs=fs, nperseg=256)
    freqs_ap, psd_ap = welch(cop_y, fs=fs, nperseg=256)
    return freqs_ml, psd_ml, freqs_ap, psd_ap


def compute_msd(S_n, fs, delta_t):  # noqa: N803
    """
    Calculates the Mean Square Displacement (MSD) for a time interval delta_t.

    Parameters:
    - S_n: array-like, Centered signal (X_n or Y_n).
    - fs: float, Sampling frequency in Hz.
    - delta_t: float, Time interval in seconds.

    Returns:
    - msd: float, MSD value.
    """
    delta_n = int(delta_t * fs)
    N = len(S_n)  # noqa: N806
    if delta_n >= N:
        raise ValueError("delta_t is too large for the signal length.")
    diff = S_n[delta_n:] - S_n[:-delta_n]
    msd = np.mean(diff**2)
    return msd


def count_zero_crossings(signal):
    """Counts the number of zero-crossings in a signal."""
    zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
    return zero_crossings


def count_peaks(signal):
    """Counts the number of peaks in a signal."""
    peaks, _ = find_peaks(signal)
    return len(peaks)


def compute_sway_density(cop_signal, fs, radius=0.3):
    """
    Calculates the sway density of the CoP signal.

    Parameters:
    - cop_signal: array-like, CoP data.
    - fs: float, Sampling frequency in Hz.
    - radius: float, default=0.3, Radius in cm for sway density calculation.

    Returns:
    - sway_density: array-like, Sway density values.
    """
    N = len(cop_signal)  # noqa: N806
    sway_density = np.zeros(N)
    for t in range(N):
        distances = np.abs(cop_signal - cop_signal[t])
        sway_density[t] = np.sum(distances <= radius) / N
    return sway_density


def compute_total_path_length(cop_x, cop_y):
    """Calculates the total path length of the CoP trajectory."""
    diffs_x = np.diff(cop_x)
    diffs_y = np.diff(cop_y)
    distances = np.sqrt(diffs_x**2 + diffs_y**2)
    return np.sum(distances)


def plot_stabilogram(time, cop_x, cop_y, output_path):
    """Plots and saves the stabilogram as time series plots for ML and AP displacements."""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(time, cop_x, color="black", linewidth=2)
    plt.title("Stabilogram - ML Displacement")
    plt.xlabel("Time (s)")
    plt.ylabel("ML Displacement (cm)")
    plt.grid(True)

    min_ml = np.min(cop_x)
    max_ml = np.max(cop_x)
    rms_ml = np.sqrt(np.mean(cop_x**2))
    plt.axhline(min_ml, color="grey", linestyle="--", label=f"Min: {min_ml:.2f} cm")
    plt.axhline(max_ml, color="grey", linestyle="-.", label=f"Max: {max_ml:.2f} cm")
    plt.axhline(rms_ml, color="grey", linestyle=":", label=f"RMS: {rms_ml:.2f} cm")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time, cop_y, color="black", linewidth=2)
    plt.title("Stabilogram - AP Displacement")
    plt.xlabel("Time (s)")
    plt.ylabel("AP Displacement (cm)")
    plt.grid(True)

    min_ap = np.min(cop_y)
    max_ap = np.max(cop_y)
    rms_ap = np.sqrt(np.mean(cop_y**2))
    plt.axhline(min_ap, color="grey", linestyle="--", label=f"Min: {min_ap:.2f} cm")
    plt.axhline(max_ap, color="grey", linestyle="-.", label=f"Max: {max_ap:.2f} cm")
    plt.axhline(rms_ap, color="grey", linestyle=":", label=f"RMS: {rms_ap:.2f} cm")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}_stabilogram.png", dpi=300)
    plt.savefig(f"{output_path}_stabilogram.svg", dpi=300)
    plt.close()


def plot_power_spectrum(freqs_ml, psd_ml, freqs_ap, psd_ap, output_path):
    """Plots and saves the power spectrum of the CoP signals."""
    plt.figure(figsize=(10, 8))
    plt.semilogy(freqs_ml, psd_ml, label="ML")
    plt.semilogy(freqs_ap, psd_ap, label="AP")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (cm²/Hz)")
    plt.title("Power Spectral Density")
    plt.grid(True)

    max_psd_ml = np.max(psd_ml)
    max_freq_ml = freqs_ml[np.argmax(psd_ml)]
    max_psd_ap = np.max(psd_ap)
    max_freq_ap = freqs_ap[np.argmax(psd_ap)]

    plt.scatter(
        max_freq_ml,
        max_psd_ml,
        color="blue",
        marker="v",
        label=f"Max ML PSD: {max_psd_ml:.2e} at {max_freq_ml:.2f} Hz",
    )
    plt.scatter(
        max_freq_ap,
        max_psd_ap,
        color="orange",
        marker="v",
        label=f"Max AP PSD: {max_psd_ap:.2e} at {max_freq_ap:.2f} Hz",
    )

    median_freq_ml = np.median(freqs_ml)
    median_freq_ap = np.median(freqs_ap)

    plt.axvline(
        median_freq_ml,
        color="blue",
        linestyle="--",
        label=f"Median ML Frequency: {median_freq_ml:.2f} Hz",
    )
    plt.axvline(
        median_freq_ap,
        color="orange",
        linestyle="--",
        label=f"Median AP Frequency: {median_freq_ap:.2f} Hz",
    )

    plt.legend()
    plt.savefig(f"{output_path}_psd.png", dpi=300)
    plt.savefig(f"{output_path}_psd.svg", dpi=300)
    plt.close()


def save_metrics_to_csv(metrics_dict, output_path):
    """Saves the calculated metrics to a CSV file with standardized headers."""
    standardized_metrics = {}
    for key, value in metrics_dict.items():
        new_key = (
            key.replace(" ", "_")
            .replace("(", "_")
            .replace(")", "")
            .replace("²", "2")
            .replace("·", "_")
            .replace("³", "3")
        )
        standardized_metrics[new_key] = value

    df = pd.DataFrame([standardized_metrics])
    df.to_csv(f"{output_path}_metrics.csv", index=False)


# ============================================================================
# SPECTRAL FEATURE FUNCTIONS
# ============================================================================


def adjust_frequency_range(freqs, fmin, fmax):
    """Adjusts the frequency range to ensure it fits within the bounds of available frequencies."""
    if fmin < freqs.min():
        fmin = freqs.min()
    if fmax > freqs.max():
        fmax = freqs.max()
    return fmin, fmax


def total_power(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the total power within the specified frequency range."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_powers = psd[idx]
    if len(selected_powers) == 0:
        return np.nan
    return np.sum(selected_powers)


def power_frequency_50(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency at which 50% of the total power is reached."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]

    if len(selected_powers) == 0:
        return np.nan

    cum_power = np.cumsum(selected_powers)
    total_power = cum_power[-1]

    if total_power == 0:
        return np.nan

    freq_50_idx = np.where(cum_power >= total_power * 0.5)[0]
    if freq_50_idx.size == 0:
        return np.nan
    return selected_freqs[freq_50_idx[0]]


def power_frequency_95(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency at which 95% of the total power is reached."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]

    if len(selected_powers) == 0:
        return np.nan

    cum_power = np.cumsum(selected_powers)
    total_power = cum_power[-1]

    if total_power == 0:
        return np.nan

    freq_95_idx = np.where(cum_power >= total_power * 0.95)[0]
    if freq_95_idx.size == 0:
        return np.nan
    return selected_freqs[freq_95_idx[0]]


def power_mode(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency with the maximum power within the specified range."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    if idx[0].size == 0:
        return np.nan
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    mode_idx = np.argmax(selected_powers)
    return selected_freqs[mode_idx]


def spectral_moment(freqs, psd, moment=1, fmin=0.15, fmax=5):
    """Calculates the spectral moment of the given order within the specified frequency range."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))
    selected_freqs = freqs[idx]
    selected_powers = psd[idx]
    if len(selected_powers) == 0:
        return np.nan
    return np.sum((selected_freqs**moment) * selected_powers)


def centroid_frequency(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the centroid frequency."""
    m0 = spectral_moment(freqs, psd, moment=0, fmin=fmin, fmax=fmax)
    m2 = spectral_moment(freqs, psd, moment=2, fmin=fmin, fmax=fmax)
    if m0 == 0:
        return np.nan
    return np.sqrt(m2 / m0)


def frequency_dispersion(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency dispersion."""
    m0 = spectral_moment(freqs, psd, moment=0, fmin=fmin, fmax=fmax)
    m1 = spectral_moment(freqs, psd, moment=1, fmin=fmin, fmax=fmax)
    m2 = spectral_moment(freqs, psd, moment=2, fmin=fmin, fmax=fmax)
    if m0 * m2 == 0:
        return np.nan
    return np.sqrt(1 - (m1**2) / (m0 * m2))


def energy_content(freqs, psd, f_low, f_high, fmin=0.15, fmax=5):
    """Calculates the energy content between f_low and f_high Hz."""
    fmin, fmax = adjust_frequency_range(freqs, fmin, fmax)
    idx = np.where((freqs >= f_low) & (freqs <= f_high) & (freqs >= fmin) & (freqs <= fmax))
    selected_powers = psd[idx]
    if len(selected_powers) == 0:
        return np.nan
    return np.sum(selected_powers)


def energy_content_below_0_5(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the energy content below 0.5 Hz."""
    return energy_content(freqs, psd, f_low=0, f_high=0.5, fmin=fmin, fmax=fmax)


def energy_content_0_5_2(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the energy content between 0.5 Hz and 2 Hz."""
    return energy_content(freqs, psd, f_low=0.5, f_high=2, fmin=fmin, fmax=fmax)


def energy_content_above_2(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the energy content above 2 Hz."""
    return energy_content(freqs, psd, f_low=2, f_high=fmax, fmin=fmin, fmax=fmax)


def frequency_quotient(freqs, psd, fmin=0.15, fmax=5):
    """Calculates the frequency quotient."""
    power_below_2 = energy_content(freqs, psd, f_low=0, f_high=2, fmin=fmin, fmax=fmax)
    power_above_2 = energy_content(freqs, psd, f_low=2, f_high=fmax, fmin=fmin, fmax=fmax)
    if power_below_2 == 0:
        return np.nan
    return power_above_2 / power_below_2


# ============================================================================
# HEATMAP VISUALIZATION FUNCTIONS
# ============================================================================


def plot_heatmap_kde(cop_x, cop_y, output_path, title="CoP Density Heatmap (KDE)", bandwidth=None):
    """Creates a heatmap using Kernel Density Estimation (KDE) for smooth density visualization."""
    x_min, x_max = np.min(cop_x) - 0.5, np.max(cop_x) + 0.5
    y_min, y_max = np.min(cop_y) - 0.5, np.max(cop_y) + 0.5
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)  # noqa: N806

    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    values = np.vstack([cop_x, cop_y])

    if bandwidth is None:
        kde = stats.gaussian_kde(values)
    else:
        kde = stats.gaussian_kde(values, bw_method=bandwidth)

    density = kde(positions).reshape(X_grid.shape)

    plt.figure(figsize=(10, 8))
    plt.contourf(X_grid, Y_grid, density, levels=50, cmap="hot")
    plt.colorbar(label="Density")
    plt.plot(cop_x, cop_y, "w.", markersize=2, alpha=0.5, label="CoP Pathway")
    plt.plot(cop_x[0], cop_y[0], "go", markersize=10, label="Start")
    plt.plot(cop_x[-1], cop_y[-1], "ro", markersize=10, label="End")
    plt.xlabel("CoP ML (cm)")
    plt.ylabel("CoP AP (cm)")
    plt.title(title)
    plt.grid(True, linestyle=":", color="lightgray", alpha=0.5)
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(f"{output_path}_heatmap_kde.png", dpi=300)
    plt.savefig(f"{output_path}_heatmap_kde.svg", dpi=300)
    plt.close()


def plot_heatmap_histogram(cop_x, cop_y, output_path, title="CoP Density Heatmap (Histogram)", bins=50):
    """Creates a heatmap using 2D histogram for discrete density bins visualization."""
    hist, x_edges, y_edges = np.histogram2d(cop_x, cop_y, bins=bins)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    hist = hist.T

    plt.figure(figsize=(10, 8))
    plt.contourf(x_centers, y_centers, hist, levels=50, cmap="hot")
    plt.colorbar(label="Density (count)")
    plt.plot(cop_x, cop_y, "w.", markersize=2, alpha=0.5, label="CoP Pathway")
    plt.plot(cop_x[0], cop_y[0], "go", markersize=10, label="Start")
    plt.plot(cop_x[-1], cop_y[-1], "ro", markersize=10, label="End")
    plt.xlabel("CoP ML (cm)")
    plt.ylabel("CoP AP (cm)")
    plt.title(title)
    plt.grid(True, linestyle=":", color="lightgray", alpha=0.5)
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(f"{output_path}_heatmap_histogram.png", dpi=300)
    plt.savefig(f"{output_path}_heatmap_histogram.svg", dpi=300)
    plt.close()


def plot_heatmap_with_contours(cop_x, cop_y, output_path, method="kde", title="CoP Density Heatmap with Topographic Contours", bins=50, bandwidth=None, contour_levels=10):
    """Creates a heatmap with topographic contour lines overlay showing density levels."""
    x_min, x_max = np.min(cop_x) - 0.5, np.max(cop_x) + 0.5
    y_min, y_max = np.min(cop_y) - 0.5, np.max(cop_y) + 0.5
    x_grid = np.linspace(x_min, x_max, 100)
    y_grid = np.linspace(y_min, y_max, 100)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)  # noqa: N806

    if method == "kde":
        positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
        values = np.vstack([cop_x, cop_y])

        if bandwidth is None:
            kde = stats.gaussian_kde(values)
        else:
            kde = stats.gaussian_kde(values, bw_method=bandwidth)

        density = kde(positions).reshape(X_grid.shape)
        density_label = "Density (KDE)"
    elif method == "histogram":
        hist, x_edges, y_edges = np.histogram2d(cop_x, cop_y, bins=bins)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        x_hist_grid, y_hist_grid = np.meshgrid(x_centers, y_centers)
        points = np.column_stack([x_hist_grid.ravel(), y_hist_grid.ravel()])
        values_hist = hist.T.ravel()
        density = griddata(points, values_hist, (X_grid, Y_grid), method="cubic", fill_value=0)
        density_label = "Density (count)"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kde' or 'histogram'.")

    plt.figure(figsize=(10, 8))
    cf = plt.contourf(X_grid, Y_grid, density, levels=50, cmap="hot", alpha=0.8)
    plt.colorbar(cf, label=density_label)

    contours = plt.contour(X_grid, Y_grid, density, levels=contour_levels, colors="white", linewidths=1.5, alpha=0.7)
    plt.clabel(contours, inline=True, fontsize=8, fmt="%1.2f")

    plt.plot(cop_x, cop_y, "cyan", linewidth=0.5, alpha=0.6, label="CoP Pathway")
    plt.plot(cop_x[0], cop_y[0], "go", markersize=12, label="Start", zorder=5)
    plt.plot(cop_x[-1], cop_y[-1], "ro", markersize=12, label="End", zorder=5)

    plt.xlabel("CoP ML (cm)")
    plt.ylabel("CoP AP (cm)")
    plt.title(f"{title}\n(Method: {method.upper()})")
    plt.grid(True, linestyle=":", color="lightgray", alpha=0.3)
    plt.legend(loc="upper right")
    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(f"{output_path}_heatmap_contours_{method}.png", dpi=300)
    plt.savefig(f"{output_path}_heatmap_contours_{method}.svg", dpi=300)
    plt.close()


# ============================================================================
# CoP BALANCE ANALYSIS FUNCTIONS
# ============================================================================


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
        selected_headers = [header for header, var in zip(headers, header_vars, strict=True) if var.get()]
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
    num_columns = 8

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


def plot_final_figure(  # noqa: N803
    time,
    X_n,  # noqa: N803
    Y_n,  # noqa: N803
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
    """Creates the final figure with the stabilogram, CoP pathway, and a text with all the presented result variables."""
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(time, X_n, label="CoP ML")
    ax1.plot(time, Y_n, label="CoP AP")
    ax1.grid(color="gray", linestyle=":", linewidth=0.5)
    ax1.set_title("Stabilogram")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Displacement (cm)")
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(X_n, Y_n, label="CoP Pathway", color="blue", marker=".", markersize=3, linestyle="None")

    ellipse_x, ellipse_y = ellipse_data[0], ellipse_data[1]
    eigvecs, scaled_eigvals, pca_mean = (
        ellipse_data[2],
        ellipse_data[3],
        ellipse_data[4],
    )

    ax2.plot(ellipse_x, ellipse_y, color="red", linestyle="--", linewidth=2, label="Confidence Ellipse")

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

    ax2.set_title("CoP Pathway with Confidence Ellipse")
    ax2.set_xlabel("CoP ML (cm)")
    ax2.set_ylabel("CoP AP (cm)")
    ax2.set_aspect("equal", adjustable="box")

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

    ax3 = fig.add_subplot(1, 2, 2)
    ax3.axis("off")
    text_str = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
    ax3.text(0.05, 0.5, text_str, fontsize=10, verticalalignment="center", transform=ax3.transAxes, wrap=True)

    plt.tight_layout()
    plt.savefig(f"{output_path}_final_figure.png", dpi=300, format="png")
    plt.savefig(f"{output_path}_final_figure.svg", format="svg")
    plt.close()


def analyze_data_2d(data, output_dir, file_name, fs, plate_width, plate_height, timestamp):
    """Analyzes selected 2D data and saves the results."""
    print(f"Starting analysis for file: {file_name}")

    try:
        print("Applying Butterworth filter...")
        dataf = butter_filter(data, fs, filter_type="low", cutoff=10, order=4, padding=True)
    except ValueError as e:
        print(f"Filtering error: {e}")
        return

    cop_x_f = dataf[:, 0]
    cop_y_f = dataf[:, 1]

    N = len(cop_x_f)  # noqa: N806
    T = N / fs  # noqa: N806
    time = np.linspace(0, (len(cop_x_f) - 1) / fs, len(cop_x_f))

    mean_ML = np.mean(cop_x_f)  # noqa: N806
    mean_AP = np.mean(cop_y_f)  # noqa: N806
    X_n = cop_x_f - mean_ML  # noqa: N806
    Y_n = cop_y_f - mean_AP  # noqa: N806

    _ = np.sqrt(X_n**2 + Y_n**2)
    COV = np.mean(X_n * Y_n)  # noqa: N806

    V_xn = np.gradient(X_n, time)  # noqa: N806
    V_yn = np.gradient(Y_n, time)  # noqa: N806
    V_n = np.sqrt(V_xn**2 + V_yn**2)  # noqa: N806

    mean_speed_ml = np.mean(np.abs(V_xn))
    mean_speed_ap = np.mean(np.abs(V_yn))
    mean_velocity_norm = np.mean(V_n)

    print("Computing power spectrum...")
    freqs_ml, psd_ml, freqs_ap, psd_ap = compute_power_spectrum(X_n, Y_n, fs)

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

    delta_t = 0.1
    print(f"Computing MSD with delta_t = {delta_t} seconds...")
    msd_ml = compute_msd(X_n, fs, delta_t)
    msd_ap = compute_msd(Y_n, fs, delta_t)

    zero_crossings_ml = count_zero_crossings(X_n)
    zero_crossings_ap = count_zero_crossings(Y_n)

    num_peaks_ml = count_peaks(X_n)
    num_peaks_ap = count_peaks(Y_n)

    rms_ml, rms_ap = compute_rms(X_n, Y_n)

    print("Calculating 2D RMS (DRMS)...")
    drms = compute_drms(X_n, Y_n)

    print("Calculating total path length...")
    total_path_length = compute_total_path_length(cop_x_f, cop_y_f)

    print("Computing sway density...")
    _ = compute_sway_density(X_n, fs, radius=0.3)
    _ = compute_sway_density(Y_n, fs, radius=0.3)

    print("Calculating confidence ellipse...")
    area, angle, bounds, ellipse_data = plot_ellipse_pca(np.column_stack((X_n, Y_n)), confidence=0.95)

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
        "RMS_2D_DRMS_cm": drms,
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
        "Centroid_Frequency_ML": centroid_freq_ml,
        "Centroid_Frequency_AP": centroid_freq_ap,
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

    print("Saving metrics to CSV...")
    metrics_file = os.path.join(output_dir, f"{file_name}_metrics.csv")
    save_metrics_to_csv(metrics, metrics_file)

    print("Plotting stabilogram...")
    stabilogram_file = os.path.join(output_dir, f"{file_name}_stabilogram.png")
    plot_stabilogram(time, X_n, Y_n, stabilogram_file)

    print("Plotting power spectrum...")
    power_spectrum_file = os.path.join(output_dir, f"{file_name}_power_spectrum.png")
    plot_power_spectrum(freqs_ml, psd_ml, freqs_ap, psd_ap, power_spectrum_file)

    print("Plotting CoP pathway with ellipse...")
    cop_pathway_file = os.path.join(output_dir, f"{file_name}_cop_pathway.png")
    plot_cop_pathway_with_ellipse(X_n, Y_n, area, angle, ellipse_data, file_name, cop_pathway_file)

    print("Creating final figure...")
    final_figure_file = os.path.join(output_dir, f"{file_name}_final_figure.png")
    plot_final_figure(
        time, X_n, Y_n, freqs_ml, psd_ml, freqs_ap, psd_ap, metrics,
        final_figure_file, area, angle, bounds, ellipse_data,
    )

    print("Generating heatmaps...")
    heatmap_base_path = os.path.join(output_dir, file_name)

    try:
        plot_heatmap_kde(X_n, Y_n, heatmap_base_path)
        print("  ✓ KDE heatmap created")
    except Exception as e:
        print(f"  ✗ Error creating KDE heatmap: {e}")

    try:
        plot_heatmap_histogram(X_n, Y_n, heatmap_base_path)
        print("  ✓ Histogram heatmap created")
    except Exception as e:
        print(f"  ✗ Error creating histogram heatmap: {e}")

    try:
        plot_heatmap_with_contours(X_n, Y_n, heatmap_base_path, method="kde")
        print("  ✓ KDE heatmap with contours created")
    except Exception as e:
        print(f"  ✗ Error creating KDE heatmap with contours: {e}")

    try:
        plot_heatmap_with_contours(X_n, Y_n, heatmap_base_path, method="histogram")
        print("  ✓ Histogram heatmap with contours created")
    except Exception as e:
        print(f"  ✗ Error creating histogram heatmap with contours: {e}")

    print(f"Analysis completed for file: {file_name}")


def main_cop_balance():
    """Main function to run the CoP Balance Analysis."""
    root = Tk()
    root.withdraw()

    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting CoP analysis...")

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

    sample_file_path = filedialog.askopenfilename(
        title="Select a Sample CSV File", filetypes=[("CSV files", "*.csv")]
    )
    if not sample_file_path:
        print("No sample file selected.")
        return

    print(f"Sample file selected: {sample_file_path}")

    selected_headers, _ = select2headers(sample_file_path)
    if not selected_headers:
        print("No valid headers selected.")
        return

    print(f"Selected Headers: {selected_headers}")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_dir, f"cop_balance_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    print(f"Main output directory created: {main_output_dir}")

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

            file_name_without_extension = os.path.splitext(file_name)[0]
            file_output_dir = os.path.join(main_output_dir, file_name_without_extension)
            os.makedirs(file_output_dir, exist_ok=True)

            analyze_data_2d(
                data, file_output_dir, file_name_without_extension,
                fs, plate_width, plate_height, timestamp,
            )
        else:
            print(f"Skipping non-CSV file: {file_name}")

    print("All files processed.")
    messagebox.showinfo("Information", "Analysis complete! Close this window.")
    root.mainloop()


# ============================================================================
# CoP CALCULATION FUNCTIONS
# ============================================================================


def select_headers_calculate(file_path):
    """Displays a GUI to select six (6) headers for force plate data analysis."""
    def get_csv_headers(file_path):
        df = pd.read_csv(file_path)
        return list(df.columns), df

    headers, df = get_csv_headers(file_path)
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [header for header, var in zip(headers, header_vars, strict=True) if var.get()]
        if len(selected_headers) != 6:
            messagebox.showinfo("Info", "Please select exactly six (6) headers for analysis.")
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
    selection_window.title("Select six (6) headers for Force Plate Data")
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight() * 0.9)}"
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
    num_columns = 7

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill="y", anchor="center")
    Button(btn_frame, text="Select All", command=select_all).pack(side="top", pady=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(side="top", pady=5)
    Button(btn_frame, text="Confirm", command=on_select).pack(side="top", pady=5)

    selection_window.mainloop()

    if not selected_headers:
        messagebox.showinfo("Info", "No headers were selected.")
        return None

    return selected_headers


def calc_cop(data, board_height_m: float = 0.0):
    """
    Converts force (N) and moment (N.m) data into center of pressure (m) coordinates.

    Parameters:
        data: numpy array with columns [Fx, Fy, Fz, Mx, My, Mz]
        board_height_m: float, height of the board over the force plate (in meters).
                        If 0, assumes no board is present.

    Returns:
        cop_xyz: numpy array with columns [CP_ap, CP_ml, h] in meters.
    """
    data = np.asarray(data)
    fx = data[:, 0]
    fy = data[:, 1]
    fz = data[:, 2]
    mx = data[:, 3]
    my = data[:, 4]

    if np.any(fz < 0):
        fz = -fz

    if board_height_m == 0.0:
        cp_ap = -my / fz
        cp_ml = mx / fz
        cp_z = np.zeros_like(fz)
    else:
        h = board_height_m
        cp_ap = (-h * fx - my) / fz
        cp_ml = (h * fy + mx) / fz
        cp_z = np.full_like(fz, h)

    cop_xyz = np.column_stack((cp_ap, cp_ml, cp_z))
    return cop_xyz


def main_calculate_cop():
    """Main function to run the CoP calculation."""
    root = Tk()
    root.withdraw()

    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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

    moment_unit = simpledialog.askstring(
        "Moment Unit of Measurement",
        "Enter the unit of measurement for your moment data (e.g. N.m, N.mm)",
        initialvalue="N.m",
        parent=root,
    )
    print(f"Moment unit of measurement: {moment_unit}")

    dimensions_input = simpledialog.askstring(
        "Force Plate Dimensions",
        "Enter the force plate dimensions (length [X-axis], width [Y-axis]) in mm, separated by a comma:",
        initialvalue="508,464",
        parent=root,
    )
    if dimensions_input:
        try:
            fp_dimensions_xy = [float(dim) / 1000 for dim in dimensions_input.split(",")]
            if len(fp_dimensions_xy) != 2:
                raise ValueError("Please provide exactly two values: length and width.")
        except ValueError as e:
            print(f"Invalid input for force plate dimensions: {e}")
            return
    else:
        print("No force plate dimensions provided. Using default [0.508, 0.464].")
        fp_dimensions_xy = [0.508, 0.464]

    board_height_mm = simpledialog.askfloat(
        "Board Height",
        "Enter the height of the board over the force plate (in mm):",
        initialvalue=0.0,
        parent=root,
    )
    if board_height_mm is not None:
        board_height_m = board_height_mm / 1000
        print(f"Using provided board height: {board_height_m} m")
    else:
        print("No board height provided. Assuming board_height_m = 0.")
        board_height_m = 0.0

    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
    if not csv_files:
        print("No CSV files found in the selected directory.")
        return

    first_file_path = os.path.join(input_dir, csv_files[0])
    selected_headers = select_headers_calculate(first_file_path)
    if not selected_headers:
        print("No valid headers selected.")
        return

    print(f"Selected Headers: {selected_headers}")

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_dir, f"vaila_cop_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    print(f"Main output directory created: {main_output_dir}")

    for file_name in csv_files:
        print(f"Processing file: {file_name}")
        file_path = os.path.join(input_dir, file_name)
        try:
            df_full = read_csv_full(file_path)
        except Exception as e:
            print(f"Error reading CSV file {file_name}: {e}")
            continue

        if not all(header in df_full.columns for header in selected_headers):
            messagebox.showerror("Header Error", f"Selected headers not found in file {file_name}.")
            print(f"Error: Selected headers not found in file {file_name}. Skipping file.")
            continue

        data = df_full[selected_headers].to_numpy()

        try:
            cop_xyz_m = calc_cop(data, board_height_m=board_height_m)
        except Exception as e:
            print(f"Error processing data for file {file_name}: {e}")
            messagebox.showerror(
                "Data Processing Error",
                f"Error processing data for file {file_name}: {e}",
            )
            continue

        cop_xyz_mm = cop_xyz_m * 1000

        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(
            main_output_dir, f"{file_name_without_extension}_{timestamp}.csv"
        )
        output_df = pd.DataFrame(cop_xyz_mm, columns=["cop_ap_mm", "cop_ml_mm", "cop_z_mm"])
        output_df.to_csv(output_file_path, index=False)
        print(f"Saved CoP data to: {output_file_path}")

    print("All files processed.")
    messagebox.showinfo(
        "Information", "CoP calculation complete! The window will close in 5 seconds."
    )


# ============================================================================
# ANALYSIS RUNNER FUNCTIONS (for external modules)
# ============================================================================


def run_force_cube_analysis():
    """Runs the Force Cube Analysis."""
    try:
        from . import force_cube_fig
        force_cube_fig.run_force_cube_fig()
    except ImportError as e:
        print(f"Error importing force_cube_fig: {e}")


def run_cop_balance_analysis():
    """Runs the CoP Balance Analysis."""
    main_cop_balance()


def run_force_cmj_analysis():
    """Runs the Force CMJ Analysis."""
    try:
        from . import force_cmj
        force_cmj.main()
    except ImportError as e:
        print(f"Error importing force_cmj: {e}")


def run_fix_noise():
    """Runs the Fix Noise Signal Analysis."""
    try:
        from . import fixnoise
        fixnoise.main()
    except ImportError as e:
        print(f"Error importing fixnoise: {e}")


def run_calculate_cop():
    """Runs the Calculate CoP process."""
    main_calculate_cop()


def run_gait_analysis():
    """Runs the Gait Analysis for a single strike."""
    try:
        from . import grf_gait
        grf_gait.main()
    except ImportError as e:
        print(f"Error importing gait_analysis: {e}")


def run_sit_to_stand():
    """Runs the Sit to Stand."""
    try:
        from vaila import sit2stand
        sit2stand.main()
    except ImportError as e:
        print(f"Error importing sit2stand: {e}")


def run_force_analysis():
    """Main function to execute the chosen force analysis."""
    root = Tk()
    root.withdraw()

    analysis_type = choose_analysis_type()

    if analysis_type == "force_cube_fig":
        run_force_cube_analysis()
    elif analysis_type == "cop_balance":
        run_cop_balance_analysis()
    elif analysis_type == "force_cmj":
        run_force_cmj_analysis()
    elif analysis_type == "fix_noise":
        run_fix_noise()
    elif analysis_type == "calculate_cop":
        run_calculate_cop()
    elif analysis_type == "gait_analysis":
        run_gait_analysis()
    elif analysis_type == "sit_to_stand":
        run_sit_to_stand()
    else:
        print("No analysis type selected.")


if __name__ == "__main__":
    run_force_analysis()
