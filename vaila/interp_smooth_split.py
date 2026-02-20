"""
===============================================================================
interp_smooth_split.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 14 October 2024
Update Date: 19 February 2026
Version: 0.0.8
Python Version: 3.12.12

Description:
------------
This script provides functionality to fill missing data in CSV files using
linear interpolation, Kalman filter, Savitzky-Golay filter, nearest value fill,
or to split data into a separate CSV file. It is intended for use in biomechanical
data analysis, where gaps in time-series data can be filled and datasets can be
split for further analysis.

Key Features:
-------------
1. **New Features**:
   - Hampel filter for spike removal.
   - Median filter for spike removal.
2. **Data Splitting**:
   - Splits CSV files into two halves for easier data management and analysis.
3. **Padding**:
   - Pads the data with the last valid value to avoid edge effects.

    padding_length = 0.1 * len(data)  # 10% padding of the data length before and after the data
    padded_data = np.concatenate(
        [data[:padding_length][::-1], data, data[-padding_length:][::-1]]
    )
    y = Filtering or smoothing the data method
    return y[padding_length:-padding_length]

4. **Filtering/Smoothing**:
   - Applies Kalman filter, Savitzky-Golay filter, or nearest value fill to
     numerical data.
5. **Gap Filling with Interpolation**:
   - Fills gaps in numerical data using linear interpolation, Kalman filter,
     Savitzky-Golay filter, nearest value fill, or leaves NaNs as is.
   - Only the missing data (NaNs) are filled; existing data remains unchanged.

Usage:
------
1) GUI mode (default): no arguments, or --gui
   $ uv run vaila/interp_smooth_split.py
   $ uv run vaila/interp_smooth_split.py --gui
   Opens the configuration dialog; after Apply you choose the source directory.
   Output is written to a timestamped subdir (e.g. processed_linear_lowess_YYYYMMDD_HHMMSS).
   Configuration can be saved/loaded as smooth_config.toml.

2) CLI mode: pass -i/--input (and optionally -o/--output, -c/--config)
   Config is read from (in priority order):
     a) -c/--config PATH   if given explicitly
     b) smooth_config.toml in the input directory
     c) smooth_config.toml in the current working directory
   If no config is found, an error is printed.

   Arguments:
     -i, --input  DIR   Input directory containing CSV files (required)
     -o, --output DIR   Output directory (default: timestamped subdir inside input)
     -c, --config TOML  Path to smooth_config.toml configuration file
     --gui              Launch GUI instead of CLI

   Examples:
     # Process using config from input dir or cwd:
     $ uv run vaila/interp_smooth_split.py -i /path/to/csv_dir

     # Process with explicit output directory:
     $ uv run vaila/interp_smooth_split.py -i ./data -o ./results

     # Process with explicit config file:
     $ uv run vaila/interp_smooth_split.py -i ./data -c ./my_config.toml

     # Full example with all flags:
     $ uv run vaila/interp_smooth_split.py -i ./data -o ./out -c ./smooth_config.toml

   Workflow tip:
     1. First run in GUI mode to configure and test your parameters.
     2. Click 'Save Template' to save a smooth_config.toml.
     3. Then use CLI mode with -i and -c for batch processing.

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================
"""

import contextlib
import datetime
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import toml
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from pykalman import KalmanFilter
from rich import print
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.arima.model import ARIMA

# Import filter_utils - handle both relative and absolute imports
try:
    from .filter_utils import butter_filter
except ImportError:
    try:
        from filter_utils import butter_filter
    except ImportError:
        print("Warning: filter_utils not found. Butterworth filtering will be disabled.")

        def butter_filter(data, **kwargs):
            print("Butterworth filter not available - filter_utils not found")
            return data


# =============================================================================
# ROBUST SPIKE REMOVAL AND NAN HANDLING FUNCTIONS
# =============================================================================

# Numba-optimized Hampel filter functions
try:
    from numba import jit, njit, prange

    @jit(nopython=True)
    def _calc_medians(window_size, arr, medians):
        """Calculate rolling medians using Numba JIT."""
        for i in range(window_size, len(arr) - window_size, 1):
            id0 = i - window_size
            id1 = i + window_size
            median = np.median(arr[id0:id1])
            medians[i] = median

    @jit(nopython=True)
    def _calc_medians_std(window_size, arr, medians, medians_diff):
        """Calculate rolling MAD (scaled to σ) using Numba JIT."""
        k = 1.4826  # Scale factor to convert MAD to σ estimate
        for i in range(window_size, len(arr) - window_size, 1):
            id0 = i - window_size
            id1 = i + window_size
            x = arr[id0:id1]
            medians_diff[i] = k * np.median(np.abs(x - np.median(x)))

    @njit(parallel=True)
    def _calc_medians_parallel(window_size, arr, medians):
        """Calculate rolling medians using Numba parallel JIT."""
        for i in prange(window_size, len(arr) - window_size, 1):
            id0 = i - window_size
            id1 = i + window_size
            median = np.median(arr[id0:id1])
            medians[i] = median

    @njit(parallel=True)
    def _calc_medians_std_parallel(window_size, arr, medians, medians_diff):
        """Calculate rolling MAD (scaled to σ) using Numba parallel JIT."""
        k = 1.4826  # Scale factor to convert MAD to σ estimate
        for i in prange(window_size, len(arr) - window_size, 1):
            id0 = i - window_size
            id1 = i + window_size
            x = arr[id0:id1]
            medians_diff[i] = k * np.median(np.abs(x - np.median(x)))

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available. Hampel filter will use slower pandas implementation.")


def sanitize_filename(name):
    """Sanitize filename/directory name by removing or replacing special characters.

    Args:
        name: String to sanitize

    Returns:
        str: Sanitized string safe for filesystem use
    """
    # Replace dots and special characters with underscores
    replacements = {
        ".": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
        " ": "_",
        "=": "eq",
        "+": "plus",
        "-": "minus",
        "(": "",
        ")": "",
        "[": "",
        "]": "",
        "{": "",
        "}": "",
        ",": "_",
    }

    sanitized = name
    for old_char, new_char in replacements.items():
        sanitized = sanitized.replace(old_char, new_char)

    # Remove multiple consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    # Ensure it's not empty
    if not sanitized:
        sanitized = "sanitized"

    return sanitized


def _hampel_numba(arr, window_size=5, n=3, parallel=True):
    """
    Hampel filter using Numba for high performance.

    Returns indices of outliers detected.
    """
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.values

    arr = np.asarray(arr, dtype=np.float64)
    medians = np.ones_like(arr, dtype=float) * np.nan
    medians_diff = np.ones_like(arr, dtype=float) * np.nan

    if parallel:
        _calc_medians_parallel(window_size, arr, medians)
        _calc_medians_std_parallel(window_size, arr, medians, medians_diff)
    else:
        _calc_medians(window_size, arr, medians)
        _calc_medians_std(window_size, arr, medians, medians_diff)

    ind = np.abs(arr - medians) > n * medians_diff
    outlier_indices = np.where(ind)[0]
    return outlier_indices, medians


def _hampel_pandas(data, window_size=7, n_sigmas=3):
    """Fallback Hampel filter using pandas (slower, for when Numba unavailable)."""
    data = np.asarray(data).copy().astype(float)
    series = pd.Series(data)

    if window_size % 2 == 0:
        window_size += 1

    rolling_median = series.rolling(window=window_size, center=True, min_periods=1).median()

    def get_mad(x):
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med))  # k=1.4826 for proper scaling

    rolling_mad = series.rolling(window=window_size, center=True, min_periods=1).apply(
        get_mad, raw=True
    )
    rolling_mad = rolling_mad.replace(0, np.nan).fillna(1e-10)

    outliers = np.abs(series - rolling_median) > n_sigmas * rolling_mad
    cleaned = series.copy()
    cleaned[outliers] = rolling_median[outliers]

    return cleaned.values


def hampel_filter(data, window_size=7, n_sigmas=3, parallel=True):
    """
    Detect and remove spikes using Hampel filter with Numba optimization.

    Uses rolling Median and MAD (Median Absolute Deviation) with k=1.4826
    scale factor for proper MAD-to-σ conversion.

    Parameters:
    - data: array-like, 1D array
    - window_size: int, half-window size for Numba version, full window for pandas
    - n_sigmas: float, threshold in σ units (default 3)
    - parallel: bool, use parallel Numba computation (default True)

    Returns:
    - cleaned_data: array-like, data with spikes replaced by local median
    """
    data = np.asarray(data).copy().astype(float)

    if NUMBA_AVAILABLE:
        # Use Numba optimized version
        half_window = window_size // 2 if window_size > 2 else 2
        outlier_indices, medians = _hampel_numba(data, half_window, n_sigmas, parallel)

        # Replace outliers with median values
        cleaned = data.copy()
        for idx in outlier_indices:
            if not np.isnan(medians[idx]):
                cleaned[idx] = medians[idx]
        return cleaned
    else:
        # Fallback to pandas version
        return _hampel_pandas(data, window_size, n_sigmas)


def robust_zscore_cleaning(data, threshold=3.5):
    """
    Remove global spikes using robust Z-Score based on MAD.

    Z = 0.6745 * (x - median) / MAD

    Points with |Z| > threshold are replaced with linear interpolation.

    Parameters:
    - data: array-like, 1D array
    - threshold: float, z-score threshold (default 3.5)

    Returns:
    - cleaned_data: array-like, data with outliers interpolated
    """
    data = np.asarray(data).copy().astype(float)

    # Calculate robust z-score
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))

    if mad == 0:
        return data  # No variation, nothing to clean

    z_scores = 0.6745 * (data - median) / mad

    # Mark outliers as NaN
    outliers = np.abs(z_scores) > threshold
    data[outliers] = np.nan

    # Interpolate the outlier positions
    return pd.Series(data).interpolate(method="linear", limit_direction="both").values


def median_filter_smooth(data, kernel_size=5):
    """
    Apply median filter smoothing to remove impulsive noise.

    Parameters:
    - data: array-like, 1D array
    - kernel_size: int, size of the median filter window (must be odd)

    Returns:
    - filtered_data: array-like, median-filtered data
    """
    from scipy.signal import medfilt

    data = np.asarray(data).copy().astype(float)

    # Ensure kernel_size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Handle NaN values by interpolating them first
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        data_clean = pd.Series(data).interpolate(method="linear", limit_direction="both").values
        # Handle edge NaNs
        data_clean = pd.Series(data_clean).ffill().bfill().values
        filtered = medfilt(data_clean, kernel_size)
        # Restore NaN positions
        filtered[nan_mask] = np.nan
        return filtered

    return medfilt(data, kernel_size)


def apply_smoothing_with_nan_handling(data, smooth_func, preserve_nans=False, **kwargs):
    """
    Apply a smoothing function while handling NaN values.

    If preserve_nans is True, the function:
    1. Saves the NaN mask
    2. Temporarily interpolates NaN positions
    3. Applies the smoothing function
    4. Restores NaN positions in the output

    This is used to fix the "Skip" interpolation option bug where smoothing
    filters would fail on data containing NaNs.

    Parameters:
    - data: array-like, 1D array (may contain NaNs)
    - smooth_func: callable, the smoothing function to apply
    - preserve_nans: bool, if True, NaN positions are restored after smoothing
    - **kwargs: additional arguments passed to smooth_func

    Returns:
    - smoothed_data: array-like, smoothed data (with or without NaNs restored)
    """
    data = np.asarray(data).copy().astype(float)

    # Save original NaN mask
    nan_mask = np.isnan(data)

    if nan_mask.any():
        # Temporarily interpolate NaN positions for smoothing
        data_filled = pd.Series(data).interpolate(method="linear", limit_direction="both").values

        # Handle edge cases where interpolation couldn't fill all NaNs
        remaining_nans = np.isnan(data_filled)
        if remaining_nans.any():
            # Use forward/backward fill for edge NaNs
            data_filled = pd.Series(data_filled).ffill().bfill().values
    else:
        data_filled = data

    # Apply smoothing
    smoothed = smooth_func(data_filled, **kwargs)

    # Restore NaN positions if requested
    if preserve_nans and nan_mask.any():
        smoothed = np.asarray(smoothed).astype(float)
        smoothed[nan_mask] = np.nan

    return smoothed


def save_config_to_toml(config, filepath):
    """Save the current configuration to a TOML file with didactic comments for non-experts."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# ================================================================\n")
        f.write("# Interp/Smooth Split - Configuration File\n")
        f.write("# Generated automatically by interp_smooth_split.py\n")
        f.write(f"# Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# ================================================================\n")
        f.write("#\n")
        f.write("# HOW TO USE THIS FILE:\n")
        f.write("# 1. Edit the values below to customize your interpolation and smoothing.\n")
        f.write("# 2. Save this file.\n")
        f.write("# 3. In the script, click 'Load TOML configuration'.\n")
        f.write("# 4. Select this file and run your analysis.\n")
        f.write("#\n")
        f.write("# IMPORTANT: Keep the format exactly as shown!\n")
        f.write("# - true/false must be lowercase\n")
        f.write("# - Numbers can have decimals (3.0) or not (30)\n")
        f.write('# - Text must be in quotes ("linear")\n')
        f.write("#\n")
        f.write("# Each section below controls a part of the processing.\n")
        f.write("# All options are explained with examples.\n")
        f.write("# ================================================================\n\n")

        # Interpolation section
        f.write("[interpolation]\n")
        f.write("# Method for filling gaps in your data.\n")
        f.write('# Options: "linear", "cubic", "nearest", "kalman", "none", "skip"\n')
        f.write("#   - linear: straight lines between points (most common)\n")
        f.write("#   - cubic: smooth curves\n")
        f.write("#   - nearest: copy nearest valid value\n")
        f.write("#   - kalman: predictive filling (advanced)\n")
        f.write("#   - none: leave gaps as NaN\n")
        f.write("#   - skip: do not fill gaps, only apply smoothing\n")
        f.write('method = "linear"\n')
        f.write("# Maximum gap size to fill (in frames).\n")
        f.write("# 0 = fill all gaps, 60 = fill up to 2 seconds at 30fps.\n")
        f.write("max_gap = 60\n\n")

        # Smoothing section
        f.write("[smoothing]\n")
        f.write("# Method for smoothing the data after filling gaps.\n")
        f.write(
            '# Options: "none", "savgol", "lowess", "kalman", "butterworth", "splines", "arima", "median", "hampel"\n'
        )
        f.write("#   - none: no smoothing\n")
        f.write("#   - savgol: Savitzky-Golay filter (preserves peaks)\n")
        f.write("#   - lowess: Local regression (for noisy data)\n")
        f.write("#   - kalman: Kalman filter (for tracking)\n")
        f.write("#   - butterworth: Butterworth filter (biomechanics standard)\n")
        f.write("#   - splines: Spline smoothing (very smooth curves)\n")
        f.write("#   - arima: ARIMA model (time series)\n")
        f.write("#   - median: Moving Median filter (outlier removal)\n")
        f.write("#   - hampel: Hampel filter (robust outlier detection)\n")
        f.write('method = "none"\n')
        f.write("#\n")
        f.write(
            "# --- Parameters for each smoothing method (only the relevant ones are used) ---\n"
        )
        f.write("# Savitzky-Golay (savgol):\n")
        f.write("window_length = 7    # Odd number, e.g. 5, 7, 9\n")
        f.write("polyorder = 3        # Usually 2 or 3\n")
        f.write("# LOWESS:\n")
        f.write("frac = 0.3           # Fraction of data (0.1-1.0)\n")
        f.write("it = 3               # Number of iterations\n")
        f.write("# Kalman:\n")
        f.write("n_iter = 5           # EM algorithm iterations (3-10)\n")
        f.write("mode = 1             # 1 = simple, 2 = advanced\n")
        f.write("# Butterworth:\n")
        f.write("cutoff = 4.0        # Cutoff frequency in Hz (e.g. 4.0, 10.0)\n")
        f.write("fs = 30.0           # Sampling frequency (video FPS, e.g. 30.0, 100.0)\n")
        f.write("# Splines:\n")
        f.write("smoothing_factor = 1.0   # 0 = no smoothing, 1 = moderate, 10+ = strong\n")
        f.write("# ARIMA:\n")
        f.write("p = 1                # AR order\n")
        f.write("d = 0                # Difference order\n")
        f.write("q = 0                # MA order\n")
        f.write("# Moving Median:\n")
        f.write("kernel_size = 5      # Odd number, e.g. 3, 5, 7\n")
        f.write("# Hampel Filter:\n")
        f.write("window_size = 7      # Odd number, e.g. 5, 7, 9\n")
        f.write("n_sigmas = 3.0       # Sigma multiplier (e.g. 3.0)\n\n")

        # Padding section
        f.write("[padding]\n")
        f.write("# Add extra frames at the start and end to avoid edge effects.\n")
        f.write("# percent = how much padding to add (as percent of data length).\n")
        f.write("# 0 = no padding, 10 = 10%% of data length (recommended).\n")
        f.write("percent = 10.0\n\n")

        # Split section
        f.write("[split]\n")
        f.write("# Split the data into two parts?\n")
        f.write("# enabled = true/false\n")
        f.write("enabled = false\n\n")

        # Sample rate section
        f.write("[time_column]\n")
        f.write("# Sample rate for Time column recalculation.\n")
        f.write(
            "# If the first column is 'Time', enter the sample rate (Hz) to recalculate time values.\n"
        )
        f.write("# Leave empty or set to 0 to use original time values from the file.\n")
        f.write("# Example: 2000.0 for 2000 Hz, 10000.0 for 10000 Hz\n")
        f.write("sample_rate = 0.0  # Set to 0 to use original time values\n\n")
    print(f"Configuration saved in: {filepath}")


def load_config_from_toml(filepath):
    """Load the configuration from a TOML file and return a dictionary."""
    with open(filepath, encoding="utf-8") as f:
        config = toml.load(f)
    print(f"Configuration loaded from: {filepath}")
    return config


SMOOTH_CONFIG_FILENAME = "smooth_config.toml"


def _smooth_config_path_for_dialog(dialog):
    """Return path for smooth_config.toml: directory of test data if set, else cwd."""
    base = (
        os.path.dirname(dialog.test_data_path)
        if getattr(dialog, "test_data_path", None)
        else os.getcwd()
    )
    return os.path.join(base, SMOOTH_CONFIG_FILENAME)


def _write_smooth_config_toml_from_result(dialog):
    """Write current applied config to smooth_config.toml so it is the single source of truth."""
    path = _smooth_config_path_for_dialog(dialog)
    save_smooth_config_toml(dialog.result, path)


def save_smooth_config_toml(config_result, filepath):
    """
    Save the applied configuration to smooth_config.toml so it is the single source of truth.
    config_result: dict with keys padding, interp_method, interp_params, smooth_method,
                   smooth_params, max_gap, do_split, sample_rate (optional).
    """
    interp = {
        "method": config_result.get("interp_method", "linear"),
        "max_gap": config_result.get("max_gap", 60),
    }
    smooth = {
        "method": config_result.get("smooth_method", "none"),
        **config_result.get("smooth_params", {}),
    }
    padding = {"percent": config_result.get("padding", 10.0)}
    split = {"enabled": config_result.get("do_split", False)}
    sr = config_result.get("sample_rate")
    time_col = {"sample_rate": float(sr) if sr is not None and sr > 0 else 0.0}
    data = {
        "interpolation": interp,
        "smoothing": smooth,
        "padding": padding,
        "split": split,
        "time_column": time_col,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        toml.dump(data, f)
    print(f"Configuration saved to {filepath} (will be used for analysis and processing).")


def load_smooth_config_for_analysis(filepath):
    """
    Load smooth_config.toml and return config in the format expected by
    perform_analysis / process_file (interp_method, smooth_method, smooth_params, padding, max_gap,
    do_split, sample_rate).
    """
    with open(filepath, encoding="utf-8") as f:
        data = toml.load(f)
    interp = data.get("interpolation", {})
    smoothing = data.get("smoothing", {})
    padding_pct = data.get("padding", {}).get("percent", 10.0)
    split = data.get("split", {})
    time_col = data.get("time_column", {})
    smooth_params = {k: v for k, v in smoothing.items() if k != "method"}
    sample_rate = time_col.get("sample_rate") or 0.0
    try:
        sample_rate = float(sample_rate)
        if sample_rate <= 0:
            sample_rate = None
    except (TypeError, ValueError):
        sample_rate = None
    return {
        "interp_method": interp.get("method", "linear"),
        "interp_params": {k: v for k, v in interp.items() if k not in ["method", "max_gap"]},
        "smooth_method": smoothing.get("method", "none"),
        "smooth_params": smooth_params,
        "padding": float(padding_pct),
        "max_gap": int(interp.get("max_gap", 60)),
        "do_split": bool(split.get("enabled", False)),
        "sample_rate": sample_rate,
    }


class InterpolationConfigDialog:
    """Standard tkinter Configuration dialog with Notebook tabs."""

    def __init__(self, parent=None):
        import tkinter as tk
        from tkinter import ttk
        
        self.result = None
        
        if parent is None:
            self.root = tk.Tk()
            self.root.title("Vaila - Data Processing")
            self.window = self.root
        else:
            self.root = parent
            self.window = tk.Toplevel(parent)
            self.window.title("Vaila - Data Processing")
            self.window.transient(parent)
            self.window.grab_set()

        self.window.geometry("1400x900")
        self.window.minsize(1200, 800)
        
        # Setup Variables (Internal State)
        self.setup_variables()
        self.create_dialog_content()
        self.center_window()
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        if parent is None:
            self.window.focus_force()

    def setup_variables(self):
        import tkinter as tk
        self.savgol_window = tk.StringVar(value="7")
        self.savgol_poly = tk.StringVar(value="3")
        self.lowess_frac = tk.StringVar(value="0.3")
        self.lowess_it = tk.StringVar(value="3")
        self.butter_cutoff = tk.StringVar(value="10.0")
        self.butter_fs = tk.StringVar(value="100.0")
        self.kalman_iterations = tk.StringVar(value="5")
        self.kalman_mode = tk.StringVar(value="1")
        self.spline_smoothing = tk.StringVar(value="1.0")
        self.arima_p = tk.StringVar(value="1")
        self.arima_d = tk.StringVar(value="0")
        self.arima_q = tk.StringVar(value="0")
        self.median_kernel = tk.StringVar(value="5")
        self.hampel_window = tk.StringVar(value="7")
        self.hampel_sigma = tk.StringVar(value="3.0")
        self.sample_rate = tk.StringVar(value="")
        
        self.split_var = tk.BooleanVar(value=False)
        self.interp_method_var = tk.StringVar(value="1") # 1=linear
        self.smooth_method_var = tk.StringVar(value="1") # 1=none
        self.padding_var = tk.StringVar(value="10")
        self.max_gap_var = tk.StringVar(value="60")
        
        self.loaded_toml = None
        self.use_toml = False
        self.test_data = None
        self.test_data_path = None
        self.param_frames = {}

    def center_window(self):
        self.window.update_idletasks()
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()
        x = max(0, (self.window.winfo_screenwidth() - width) // 2)
        y = max(0, (self.window.winfo_screenheight() - height) // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def create_dialog_content(self):
        import tkinter as tk
        from tkinter import ttk
        
        # Header
        self.header_frame = tk.Frame(self.window)
        self.header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        tk.Label(self.header_frame, text="Interpolation & Smoothing Config", font=("Arial", 16, "bold")).pack()
        tk.Label(self.header_frame, text="Configure gap filling, smoothing and validate your data.", fg="gray").pack()

        # Split pane (Left: Tabs right: Analysis)
        self.main_split = tk.PanedWindow(self.window, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        self.main_split.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.config_panel = tk.Frame(self.main_split, width=400)
        self.main_split.add(self.config_panel, minsize=350)
        
        self.analysis_panel = tk.Frame(self.main_split)
        self.main_split.add(self.analysis_panel, minsize=600)

        # Tabs
        self.notebook = ttk.Notebook(self.config_panel)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_interp = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.tab_interp, text="Gap Filling")
        
        self.tab_smooth = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.tab_smooth, text="Smoothing")
        
        self.tab_general = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.tab_general, text="General")
        
        self.build_interp_tab()
        self.build_smooth_tab()
        self.build_general_tab()
        
        # Buttons
        self.buttons_frame = tk.Frame(self.config_panel)
        self.buttons_frame.pack(fill="x", padx=10, pady=10)
        
        tk.Button(self.buttons_frame, text="💾 Save Template", command=self.create_toml_template, width=15).grid(row=0, column=0, padx=5, pady=5)
        tk.Button(self.buttons_frame, text="📂 Load TOML", command=self.load_toml_config, width=15).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.buttons_frame, text="✅ Apply & Run Batch", command=self.ok, bg="#a8df65", font=("Arial", 10, "bold")).grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(15, 5))
        
        self.build_analysis_panel()

    def build_interp_tab(self):
        import tkinter as tk
        from tkinter import ttk
        tk.Label(self.tab_interp, text="Select Interpolation Method:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        
        methods = {
            "1": "Linear (Straight lines)",
            "2": "Cubic (Smooth curves)",
            "3": "Nearest (Copy nearest)",
            "4": "Kalman (Predictive)",
            "5": "None (Leave gaps)",
            "6": "Skip (Only smooth)",
        }
        
        self.interp_combo = ttk.Combobox(self.tab_interp, values=list(methods.values()), state="readonly")
        self.interp_combo.pack(fill="x", pady=5)
        self.interp_combo.set("Linear (Straight lines)")
        self.interp_combo.bind("<<ComboboxSelected>>", self.on_interp_change)
        
        tk.Label(self.tab_interp, text="Max Gap Size (frames):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 5))
        tk.Entry(self.tab_interp, textvariable=self.max_gap_var).pack(fill="x", pady=5)
        tk.Label(self.tab_interp, text="0 = Fill all gaps\n60 = Up to 2 seconds at 30 fps", fg="gray", justify="left").pack(anchor="w")

    def on_interp_change(self, event=None):
        mapping = {
            "Linear (Straight lines)": "1",
            "Cubic (Smooth curves)": "2",
            "Nearest (Copy nearest)": "3",
            "Kalman (Predictive)": "4",
            "None (Leave gaps)": "5",
            "Skip (Only smooth)": "6",
        }
        choice = self.interp_combo.get()
        if choice in mapping:
            self.interp_method_var.set(mapping[choice])

    def build_smooth_tab(self):
        import tkinter as tk
        from tkinter import ttk
        tk.Label(self.tab_smooth, text="Select Smoothing Method:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        
        methods = {
            "1": "None",
            "2": "Savitzky-Golay",
            "3": "LOWESS",
            "4": "Kalman",
            "5": "Butterworth",
            "6": "Splines",
            "7": "ARIMA",
            "8": "Moving Median",
            "9": "Hampel Filter",
        }
        
        self.smooth_combo = ttk.Combobox(self.tab_smooth, values=list(methods.values()), state="readonly")
        self.smooth_combo.pack(fill="x", pady=5)
        self.smooth_combo.set("None")
        self.smooth_combo.bind("<<ComboboxSelected>>", self.on_smooth_change)
        
        self.dynamic_params_container = tk.Frame(self.tab_smooth)
        self.dynamic_params_container.pack(fill="both", expand=True, pady=15)
        
        self.build_all_param_frames()
        self.on_smooth_change()

    def on_smooth_change(self, event=None):
        mapping = {
            "None": "1",
            "Savitzky-Golay": "2",
            "LOWESS": "3",
            "Kalman": "4",
            "Butterworth": "5",
            "Splines": "6",
            "ARIMA": "7",
            "Moving Median": "8",
            "Hampel Filter": "9",
        }
        choice = self.smooth_combo.get()
        if choice in mapping:
            self.smooth_method_var.set(mapping[choice])
            
            for frame in self.param_frames.values():
                frame.pack_forget()
                
            method_id = mapping[choice]
            if method_id in self.param_frames:
                self.param_frames[method_id].pack(fill="both", expand=True)

    def build_all_param_frames(self):
        import tkinter as tk
        # 2: Savgol
        f2 = tk.Frame(self.dynamic_params_container)
        tk.Label(f2, text="Window Length (Odd):").pack(anchor="w")
        tk.Entry(f2, textvariable=self.savgol_window).pack(fill="x", pady=(0, 10))
        tk.Label(f2, text="Polynomial Order:").pack(anchor="w")
        tk.Entry(f2, textvariable=self.savgol_poly).pack(fill="x")
        self.param_frames["2"] = f2
        
        # 3: Lowess
        f3 = tk.Frame(self.dynamic_params_container)
        tk.Label(f3, text="Fraction (0.1 - 1.0):").pack(anchor="w")
        tk.Entry(f3, textvariable=self.lowess_frac).pack(fill="x", pady=(0, 10))
        tk.Label(f3, text="Iterations:").pack(anchor="w")
        tk.Entry(f3, textvariable=self.lowess_it).pack(fill="x")
        self.param_frames["3"] = f3
        
        # 4: Kalman
        f4 = tk.Frame(self.dynamic_params_container)
        tk.Label(f4, text="EM Iterations:").pack(anchor="w")
        tk.Entry(f4, textvariable=self.kalman_iterations).pack(fill="x", pady=(0, 10))
        tk.Label(f4, text="Mode (1=1D, 2=2D):").pack(anchor="w")
        tk.Entry(f4, textvariable=self.kalman_mode).pack(fill="x")
        self.param_frames["4"] = f4
        
        # 5: Butterworth
        f5 = tk.Frame(self.dynamic_params_container)
        tk.Label(f5, text="Cutoff Frequency (Hz):").pack(anchor="w")
        tk.Entry(f5, textvariable=self.butter_cutoff).pack(fill="x", pady=(0, 2))
        tk.Label(f5, text="Tip: 4-10 Hz for biomechanics", fg="gray", font=("Arial", 9)).pack(anchor="w", pady=(0, 10))
        
        tk.Label(f5, text="Sampling Freq (fs, Hz):").pack(anchor="w")
        tk.Entry(f5, textvariable=self.butter_fs).pack(fill="x", pady=(0, 2))
        tk.Label(f5, text="Tip: fps of the video or capture freq", fg="gray", font=("Arial", 9)).pack(anchor="w")
        self.param_frames["5"] = f5
        
        # 6: Splines
        f6 = tk.Frame(self.dynamic_params_container)
        tk.Label(f6, text="Smoothing Factor:").pack(anchor="w")
        tk.Entry(f6, textvariable=self.spline_smoothing).pack(fill="x")
        self.param_frames["6"] = f6
        
        # 7: ARIMA
        f7 = tk.Frame(self.dynamic_params_container)
        tk.Label(f7, text="P (AR):").pack(anchor="w")
        tk.Entry(f7, textvariable=self.arima_p).pack(fill="x", pady=(0, 5))
        tk.Label(f7, text="D (Diff):").pack(anchor="w")
        tk.Entry(f7, textvariable=self.arima_d).pack(fill="x", pady=(0, 5))
        tk.Label(f7, text="Q (MA):").pack(anchor="w")
        tk.Entry(f7, textvariable=self.arima_q).pack(fill="x")
        self.param_frames["7"] = f7
        
        # 8: Median
        f8 = tk.Frame(self.dynamic_params_container)
        tk.Label(f8, text="Kernel Size (Odd):").pack(anchor="w")
        tk.Entry(f8, textvariable=self.median_kernel).pack(fill="x")
        self.param_frames["8"] = f8
        
        # 9: Hampel
        f9 = tk.Frame(self.dynamic_params_container)
        tk.Label(f9, text="Window Size (Odd):").pack(anchor="w")
        tk.Entry(f9, textvariable=self.hampel_window).pack(fill="x", pady=(0, 10))
        tk.Label(f9, text="Sigma Multiplier (e.g. 3.0):").pack(anchor="w")
        tk.Entry(f9, textvariable=self.hampel_sigma).pack(fill="x")
        self.param_frames["9"] = f9

    def build_general_tab(self):
        import tkinter as tk
        tk.Label(self.tab_general, text="Padding (%):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 5))
        tk.Entry(self.tab_general, textvariable=self.padding_var).pack(fill="x", pady=5)
        
        tk.Label(self.tab_general, text="Split Dataset:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 5))
        tk.Checkbutton(self.tab_general, text="Enable Splitting", variable=self.split_var).pack(anchor="w", pady=5)
        
        tk.Label(self.tab_general, text="Sample Rate Override (optional):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 5))
        tk.Entry(self.tab_general, textvariable=self.sample_rate).pack(fill="x", pady=5)

    def build_analysis_panel(self):
        import tkinter as tk
        self.analysis_top = tk.Frame(self.analysis_panel)
        self.analysis_top.pack(fill="x", padx=10, pady=10)
        
        tk.Button(self.analysis_top, text="📊 Load Test CSV", command=self.load_test_data).pack(side="left", padx=5)
        
        self.test_label = tk.Label(self.analysis_top, text="No test data loaded.", fg="gray")
        self.test_label.pack(side="left", padx=10)
        
        self.col_frame = tk.Frame(self.analysis_top)
        self.col_frame.pack(side="right", padx=5)
        
        self.col_combo = None
        
        self.plot_frame = tk.Frame(self.analysis_panel, bg="white")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def load_test_data(self):
        from tkinter import filedialog, messagebox
        import tkinter as tk
        from tkinter import ttk
        import os
        import pandas as pd
        import numpy as np

        file_path = filedialog.askopenfilename(title="Select CSV file for testing", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file_path:
            try:
                self.test_data_path = file_path
                self.test_data = pd.read_csv(file_path)
                self.test_label.configure(text=os.path.basename(file_path))
                
                numeric_cols = self.test_data.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    messagebox.showerror("Error", "No numeric columns found.")
                    return
                
                if self.col_combo:
                    for widget in self.col_frame.winfo_children():
                        widget.destroy()
                    
                self.test_col_var = tk.StringVar(value=numeric_cols[0])
                self.col_combo = ttk.Combobox(self.col_frame, values=numeric_cols, textvariable=self.test_col_var, state="readonly")
                self.col_combo.pack(side="left", padx=5)
                self.col_combo.bind("<<ComboboxSelected>>", lambda e: self.run_analysis(self.test_col_var.get()))
                
                tk.Button(self.col_frame, text="▶ Run Check", command=lambda: self.run_analysis(self.test_col_var.get())).pack(side="left", padx=5)
                tk.Button(self.col_frame, text="❄ Winter Check", command=lambda: self.perform_winter_residual_analysis(self.test_col_var.get()), fg="blue").pack(side="left", padx=5)

                self.run_analysis(numeric_cols[0])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def get_current_analysis_config(self):
        return self.get_config()

    def run_analysis(self, column_name):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import numpy as np

        if self.test_data is None:
            return
            
        for child in self.plot_frame.winfo_children():
            child.destroy()
            
        config = self.get_config()
        original_data = self.test_data[column_name].values
        
        if self.test_data.select_dtypes(include=[np.number]).columns[0] == self.test_data.columns[0]:
            frame_numbers = self.test_data.iloc[:, 0].values
        else:
            frame_numbers = np.arange(len(original_data))
            
        processed_data, padded_data = self.process_column_for_analysis(original_data, config)
        
        first_derivative = np.gradient(processed_data)
        second_derivative = np.gradient(first_derivative)
        
        valid_mask = ~np.isnan(original_data)
        residuals = np.full_like(original_data, np.nan)
        residuals[valid_mask] = original_data[valid_mask] - processed_data[valid_mask]
        filtered_residuals = self.apply_filter_to_residuals(residuals, config)
        
        fig = Figure(figsize=(10, 8), facecolor='#ebebeb')

        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(frame_numbers, original_data, "o", label="Original", alpha=0.5, markersize=3, color="blue")
        ax1.plot(frame_numbers, processed_data, ".", label="Processed", markersize=4, color="red", alpha=0.7)
        ax1.set_title(f"Original vs Processed - {column_name}", fontweight="bold")
        ax1.legend(loc="best")
        
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(frame_numbers[valid_mask], residuals[valid_mask], "o", markersize=3, label="Og. Residuals", alpha=0.4, color="green")
        ax2.plot(frame_numbers[valid_mask], filtered_residuals[valid_mask], ".", markersize=5, label="Filtered", alpha=0.7, color="red")
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1.5)
        ax2.set_title("Residuals", fontweight="bold")
        
        rms_error = np.sqrt(np.nanmean(residuals**2))
        ax2.text(0.02, 0.98, f"RMS Error: {rms_error:.4f}", transform=ax2.transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(frame_numbers, first_derivative, "-", linewidth=1.5, color="magenta", alpha=0.7)
        ax3.set_title("First Derivative (Velocity)", fontweight="bold")

        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(frame_numbers, second_derivative, "-", linewidth=1.5, color="cyan", alpha=0.7)
        ax4.set_title("Second Derivative (Acceleration)", fontweight="bold")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def validate(self):
        try:
            return True
        except ValueError as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Invalid input: {e}")
            return False

    def get_config(self):
        interp_map = {1: "linear", 2: "cubic", 3: "nearest", 4: "kalman", 5: "none", 6: "skip"}
        smooth_map = {1: "none", 2: "savgol", 3: "lowess", 4: "kalman", 5: "butterworth", 6: "splines", 7: "arima", 8: "median", 9: "hampel"}
        
        interp_method = int(self.interp_method_var.get())
        smooth_method = int(self.smooth_method_var.get())
        
        smooth_params = {}
        if smooth_method == 2:
            smooth_params = {"window_length": int(self.savgol_window.get()), "polyorder": int(self.savgol_poly.get())}
        elif smooth_method == 3:
            smooth_params = {"frac": float(self.lowess_frac.get()), "it": int(self.lowess_it.get())}
        elif smooth_method == 4:
            smooth_params = {"n_iter": int(self.kalman_iterations.get()), "mode": int(self.kalman_mode.get())}
        elif smooth_method == 5:
            smooth_params = {"cutoff": float(self.butter_cutoff.get()), "fs": float(self.butter_fs.get())}
        elif smooth_method == 6:
            smooth_params = {"smoothing_factor": float(self.spline_smoothing.get())}
        elif smooth_method == 7:
            smooth_params = {"p": int(self.arima_p.get()), "d": int(self.arima_d.get()), "q": int(self.arima_q.get())}
        elif smooth_method == 8:
            smooth_params = {"kernel_size": int(self.median_kernel.get())}
        elif smooth_method == 9:
            smooth_params = {"window_size": int(self.hampel_window.get()), "n_sigmas": float(self.hampel_sigma.get())}
            
        interp_params = {}
        
        sr_val = self.sample_rate.get().strip()
        sr = float(sr_val) if sr_val else None

        return {
            "padding": float(self.padding_var.get()),
            "interp_method": interp_map[interp_method],
            "interp_params": interp_params,
            "smooth_method": smooth_map[smooth_method],
            "smooth_params": smooth_params,
            "max_gap": int(self.max_gap_var.get()),
            "do_split": self.split_var.get(),
            "sample_rate": sr,
        }

    def ok(self):
        from tkinter import messagebox
        if self.validate():
            self.result = self.get_config()
            conf_str = f"Methods: Gap={self.result['interp_method']}, Smooth={self.result['smooth_method']}."
            if messagebox.askokcancel("Confirm Parameters", f"Apply the following processing?\n\n{conf_str}"):
                _write_smooth_config_toml_from_result(self)
                self.window.destroy()
            
    def cancel(self):
        self.result = None
        self.window.destroy()

    def apply_toml_to_gui(self, config):
        interp_rev = {"linear": "1", "cubic": "2", "nearest": "3", "kalman": "4", "none": "5", "skip": "6"}
        smooth_rev = {"none": "1", "savgol": "2", "lowess": "3", "kalman": "4", "butterworth": "5", "splines": "6", "arima": "7", "median": "8", "hampel": "9"}
        
        interp = config.get("interpolation", {})
        if interp.get("method") in interp_rev:
            name_mapping = {
                "1": "Linear (Straight lines)", "2": "Cubic (Smooth curves)", "3": "Nearest (Copy nearest)",
                "4": "Kalman (Predictive)", "5": "None (Leave gaps)", "6": "Skip (Only smooth)"
            }
            mapped = name_mapping[interp_rev[interp.get("method")]]
            self.interp_combo.set(mapped)
            self.on_interp_change()
            
        self.max_gap_var.set(str(interp.get("max_gap", 60)))
        
        smoothing = config.get("smoothing", {})
        if smoothing.get("method") in smooth_rev:
            name_mapping = {
                "1": "None", "2": "Savitzky-Golay", "3": "LOWESS", "4": "Kalman", "5": "Butterworth",
                "6": "Splines", "7": "ARIMA", "8": "Moving Median", "9": "Hampel Filter"
            }
            mapped = name_mapping[smooth_rev[smoothing.get("method")]]
            self.smooth_combo.set(mapped)
            self.on_smooth_change()
            
        if smoothing.get("method") == "butterworth":
            self.butter_cutoff.set(str(smoothing.get("cutoff", 10.0)))
            self.butter_fs.set(str(smoothing.get("fs", 100.0)))
        elif smoothing.get("method") == "savgol":
            self.savgol_window.set(str(smoothing.get("window_length", 7)))
            self.savgol_poly.set(str(smoothing.get("polyorder", 3)))
            
        self.padding_var.set(str(config.get("padding", {}).get("percent", 10.0)))
        self.split_var.set(config.get("split", {}).get("enabled", False))
        
        sr = config.get("time_column", {}).get("sample_rate", 0.0)
        if sr > 0:
            self.sample_rate.set(str(sr))

    def load_toml_config(self):
        from tkinter import filedialog, messagebox
        import os
        file_path = filedialog.askopenfilename(title="Load TOML configuration", filetypes=[("TOML files", "*.toml"), ("All files", "*.*")])
        if file_path:
            config = load_config_from_toml(file_path)
            self.loaded_toml = config
            self.use_toml = True
            self.apply_toml_to_gui(config)
            messagebox.showinfo("Loaded", f"Loaded configuration from {os.path.basename(file_path)}")
            
    def create_toml_template(self):
        from tkinter import filedialog, messagebox
        file_path = filedialog.asksaveasfilename(title="Save Template", defaultextension=".toml", filetypes=[("TOML files", "*.toml")], initialfile="smooth_config.toml")
        if file_path:
            save_config_to_toml(self.get_config(), file_path)
            messagebox.showinfo("Template created", f"Template TOML created in:\n{file_path}")

    def process_column_for_analysis(self, data, config):
        import numpy as np
        import pandas as pd
        padding_percent = config["padding"]
        pad_len = int(len(data) * padding_percent / 100) if padding_percent > 0 else 0

        padded_data = np.pad(data, pad_len, mode="edge") if pad_len > 0 else data.copy()

        if config["interp_method"] not in ["none", "skip"]:
            series = pd.Series(padded_data)
            if config["interp_method"] in ["linear", "cubic", "nearest"]:
                series = series.interpolate(method=config["interp_method"], limit_direction="both")
            padded_data = series.values

        if config["smooth_method"] != "none":
            try:
                if config["smooth_method"] == "savgol":
                    padded_data = savgol_smooth(padded_data, config["smooth_params"]["window_length"], config["smooth_params"]["polyorder"])
                elif config["smooth_method"] == "lowess":
                    padded_data = lowess_smooth(padded_data, config["smooth_params"]["frac"], config["smooth_params"]["it"])
                elif config["smooth_method"] == "kalman":
                    padded_data = kalman_smooth(padded_data, config["smooth_params"]["n_iter"], config["smooth_params"]["mode"]).flatten()
                elif config["smooth_method"] == "butterworth":
                    if not np.isnan(padded_data).all():
                        padded_data = butter_filter(padded_data, fs=config["smooth_params"]["fs"], filter_type="low", cutoff=config["smooth_params"]["cutoff"], order=4)
                elif config["smooth_method"] == "splines":
                    padded_data = spline_smooth(padded_data, s=config["smooth_params"]["smoothing_factor"])
                elif config["smooth_method"] == "arima":
                    padded_data = arima_smooth(padded_data, order=(config["smooth_params"]["p"], config["smooth_params"]["d"], config["smooth_params"]["q"]))
                elif config["smooth_method"] == "median":
                    padded_data = median_filter_smooth(padded_data, kernel_size=config["smooth_params"].get("kernel_size", 5))
            except Exception as e:
                print(f"Error in smoothing: {str(e)}")

        processed_data = padded_data[pad_len:-pad_len] if pad_len > 0 else padded_data
        return processed_data, padded_data

    def apply_filter_to_residuals(self, residuals, config):
        import numpy as np
        try:
            if config["smooth_method"] == "none":
                return residuals

            padding_percent = config["padding"]
            pad_len = int(len(residuals) * padding_percent / 100) if padding_percent > 0 else 0
            padded_residuals = np.pad(residuals, pad_len, mode="edge") if pad_len > 0 else residuals.copy()

            if config["smooth_method"] == "savgol":
                filtered_residuals = savgol_smooth(padded_residuals, config["smooth_params"]["window_length"], config["smooth_params"]["polyorder"])
            elif config["smooth_method"] == "lowess":
                filtered_residuals = lowess_smooth(padded_residuals, config["smooth_params"]["frac"], config["smooth_params"]["it"])
            elif config["smooth_method"] == "kalman":
                filtered_residuals = kalman_smooth(padded_residuals, config["smooth_params"]["n_iter"], config["smooth_params"]["mode"]).flatten()
            elif config["smooth_method"] == "butterworth":
                if not np.isnan(padded_residuals).all():
                    filtered_residuals = butter_filter(padded_residuals, fs=config["smooth_params"]["fs"], filter_type="low", cutoff=config["smooth_params"]["cutoff"], order=4)
                else:
                    filtered_residuals = padded_residuals
            elif config["smooth_method"] == "splines":
                filtered_residuals = spline_smooth(padded_residuals, s=config["smooth_params"]["smoothing_factor"])
            elif config["smooth_method"] == "arima":
                filtered_residuals = arima_smooth(padded_residuals, order=(config["smooth_params"]["p"], config["smooth_params"]["d"], config["smooth_params"]["q"]))
            elif config["smooth_method"] == "median":
                filtered_residuals = median_filter_smooth(padded_residuals, kernel_size=config["smooth_params"]["kernel_size"])
            else:
                filtered_residuals = padded_residuals

            if pad_len > 0:
                filtered_residuals = filtered_residuals[pad_len:-pad_len]
            return filtered_residuals
        except Exception as e:
            return residuals

    def perform_winter_residual_analysis(self, column_name):
        from tkinter import messagebox
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import numpy as np
        import pandas as pd
        
        if self.test_data is None:
            return
            
        fs_str = self.butter_fs.get()
        try:
            fs = float(fs_str)
            if fs <= 0:
                messagebox.showerror("Error", "Sampling frequency (fs) must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid fs. Use a numeric value for Butterworth Sampling Freq.")
            return

        for child in self.plot_frame.winfo_children():
            child.destroy()
            
        data = self.test_data[column_name].values
        if np.isnan(data).any():
            data = pd.Series(data).interpolate(method="linear", limit_direction="both").values
            
        try:
            fc_arr, res_arr, opt_fc = winter_residual_analysis(data=data, fs=fs, fc_min=1.0, fc_max=15.0, n_fc=29, order=4)
        except Exception as e:
            messagebox.showerror("Error", f"Winter analysis failed: {str(e)}")
            return
            
        fig = Figure(figsize=(8, 6), facecolor='#ebebeb')
            
        ax = fig.add_subplot(111)
        ax.plot(fc_arr, res_arr, 'o-', linewidth=2, color='tab:blue', label='Residual RMS')
        ax.axvline(x=opt_fc, color='tab:red', linestyle='--', linewidth=2, label=f'Optimal Cutoff ≈ {opt_fc:.2f} Hz')
        
        ax.set_title(f"Winter residual analysis - {column_name} (fs={fs} Hz)", fontweight='bold')
        ax.set_xlabel('Cutoff Frequency (fc) [Hz]', fontweight='bold')
        ax.set_ylabel('Residual RMS', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
            
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


def winter_residual_analysis(data, fs, fc_min=1.0, fc_max=15.0, n_fc=29, order=4):
    """
    Winter-style residual analysis: Butterworth low-pass at multiple cutoff frequencies,
    compute RMS(residual) = RMS(raw - filtered). Used to find optimal cutoff (elbow).

    Parameters:
    - data: 1D array (NaN filled with linear interpolation for the sweep)
    - fs: sampling frequency (Hz)
    - fc_min, fc_max: range of cutoff frequencies (Hz)
    - n_fc: number of fc points
    - order: Butterworth order (4 = dual 2nd for zero phase in filter_utils)

    Returns:
    - fc_list: array of cutoff frequencies
    - rms_list: array of RMS residual for each fc
    - suggested_fc: cutoff at elbow (where relative decrease in RMS drops below ~5%)
    """
    data = np.asarray(data, dtype=float)
    valid = ~np.isnan(data)
    if not np.any(valid):
        return np.array([]), np.array([]), None
    # Fill NaN for filtering
    series = pd.Series(data)
    filled = series.interpolate(method="linear", limit_direction="both").values
    fc_list = np.linspace(fc_min, fc_max, num=n_fc)
    rms_list = []
    for fc in fc_list:
        try:
            filtered = butter_filter(
                filled, fs=fs, filter_type="low", cutoff=fc, order=order, padding=True
            )
            residual = filled - filtered
            rms_list.append(np.sqrt(np.nanmean(residual[valid] ** 2)))
        except Exception:
            rms_list.append(np.nan)
    rms_list = np.array(rms_list)
    # Elbow: first fc where relative decrease in RMS is below threshold
    suggested_fc = None
    for i in range(1, len(rms_list)):
        if rms_list[i - 1] > 0 and not np.isnan(rms_list[i]):
            rel_decrease = (rms_list[i - 1] - rms_list[i]) / rms_list[i - 1]
            if rel_decrease < 0.05:  # less than 5% decrease
                suggested_fc = float(fc_list[i])
                break
    if suggested_fc is None and len(fc_list) > 0:
        suggested_fc = float(fc_list[-1])
    return fc_list, rms_list, suggested_fc


def spline_smooth(data, s=1.0):
    """
    Applies spline smoothing to the data.

    Parameters:
    - data: array-like, 1D or 2D array
    - s: float, smoothing factor

    Returns:
    - filtered_data: array-like, smoothed data
    """
    data = np.asarray(data)

    # Apply padding for better edge handling
    pad_len = int(len(data) * 0.1)  # 10% padding
    if pad_len > 0:
        if data.ndim == 1:
            padded_data = np.pad(data, (pad_len, pad_len), mode="reflect")
            padded_x = np.arange(len(padded_data))
            spline = UnivariateSpline(padded_x, padded_data, s=s)
            return spline(padded_x)[pad_len:-pad_len]
        else:
            padded_data = np.pad(data, ((pad_len, pad_len), (0, 0)), mode="reflect")
            padded_x = np.arange(len(padded_data))
            filtered = np.empty_like(data)
            for j in range(data.shape[1]):
                spline = UnivariateSpline(padded_x, padded_data[:, j], s=s)
                filtered[:, j] = spline(padded_x)[pad_len:-pad_len]
            return filtered
    else:
        if data.ndim == 1:
            x = np.arange(len(data))
            spline = UnivariateSpline(x, data, s=s)
            return spline(x)
        else:
            filtered = np.empty_like(data)
            x = np.arange(data.shape[0])
            for j in range(data.shape[1]):
                spline = UnivariateSpline(x, data[:, j], s=s)
                filtered[:, j] = spline(x)
            return filtered


def kalman_smooth(data, n_iter=5, mode=1):
    """
    Apply Kalman smoothing to data.

    Parameters:
    - data: input data (1D or 2D array)
    - n_iter: number of EM iterations
    - mode: 1 for 1D processing, 2 for 2D (x,y pairs)

    Returns:
    - smoothed data
    """
    alpha = 0.7  # Blending factor for smoothing
    data = np.asarray(data)  # Ensure it's a numpy array

    # Handle 1D data
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_features = data.shape[1]

    try:
        if mode == 1:  # 1D mode
            # Process each column independently
            filtered_data = np.empty_like(data)
            for j in range(n_features):
                # Initialize Kalman filter for 1D state (position and velocity)
                kf = KalmanFilter(
                    transition_matrices=np.array([[1, 1], [0, 1]]),
                    observation_matrices=np.array([[1, 0]]),
                    initial_state_mean=np.zeros(2),
                    initial_state_covariance=np.eye(2),
                    transition_covariance=np.eye(2) * 0.1,
                    observation_covariance=np.array([[0.1]]),
                    n_dim_obs=1,
                    n_dim_state=2,
                )

                # Apply EM algorithm and smoothing
                smoothed_state_means, _ = kf.em(data[:, j : j + 1], n_iter=n_iter).smooth(
                    data[:, j : j + 1]
                )
                filtered_data[:, j] = alpha * smoothed_state_means[:, 0] + (1 - alpha) * data[:, j]

        else:  # mode == 2
            # Process x,y pairs together
            if n_features % 2 != 0:
                raise ValueError("For 2D mode, number of features must be even (x,y pairs)")

            filtered_data = np.empty_like(data)
            for j in range(0, n_features, 2):
                # Initialize Kalman filter for 2D state (x,y positions and velocities)
                # State vector: [x, y, vx, vy, ax, ay]
                # Transition matrix models constant acceleration motion
                transition_matrix = np.array(
                    [
                        [1, 0, 1, 0, 0.5, 0],  # x = x + vx + 0.5*ax
                        [0, 1, 0, 1, 0, 0.5],  # y = y + vy + 0.5*ay
                        [0, 0, 1, 0, 1, 0],  # vx = vx + ax
                        [0, 0, 0, 1, 0, 1],  # vy = vy + ay
                        [0, 0, 0, 0, 1, 0],  # ax = ax
                        [0, 0, 0, 0, 0, 1],  # ay = ay
                    ]
                )

                # Observation matrix: observe x and y positions
                observation_matrix = np.array(
                    [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]  # observe x  # observe y
                )

                # Initialize state mean with first observation and zero velocities/accelerations
                initial_state_mean = np.array(
                    [
                        data[0, j],  # initial x
                        data[0, j + 1],  # initial y
                        0,  # initial vx
                        0,  # initial vy
                        0,  # initial ax
                        0,  # initial ay
                    ]
                )

                # Initialize state covariance with high uncertainty in velocities and accelerations
                initial_state_covariance = np.array(
                    [
                        [0.5, 0, 0, 0, 0, 0],  # x uncertainty
                        [0, 0.5, 0, 0, 0, 0],  # y uncertainty
                        [0, 0, 0.5, 0, 0, 0],  # vx uncertainty
                        [0, 0, 0, 0.5, 0, 0],  # vy uncertainty
                        [0, 0, 0, 0, 0.5, 0],  # ax uncertainty
                        [0, 0, 0, 0, 0, 0.5],  # ay uncertainty
                    ]
                )

                # Process noise (smaller for positions, larger for velocities and accelerations)
                transition_covariance = np.array(
                    [
                        [0.1, 0, 0, 0, 0, 0],  # x process noise
                        [0, 0.1, 0, 0, 0, 0],  # y process noise
                        [0, 0, 0.2, 0, 0, 0],  # vx process noise
                        [0, 0, 0, 0.2, 0, 0],  # vy process noise
                        [0, 0, 0, 0, 0.3, 0],  # ax process noise
                        [0, 0, 0, 0, 0, 0.3],  # ay process noise
                    ]
                )

                # Measurement noise (small for position measurements)
                observation_covariance = np.array(
                    [[0.1, 0], [0, 0.1]]  # x measurement noise  # y measurement noise
                )

                # Create Kalman filter instance
                kf = KalmanFilter(
                    transition_matrices=transition_matrix,
                    observation_matrices=observation_matrix,
                    initial_state_mean=initial_state_mean,
                    initial_state_covariance=initial_state_covariance,
                    transition_covariance=transition_covariance,
                    observation_covariance=observation_covariance,
                    n_dim_obs=2,
                    n_dim_state=6,
                )

                # Prepare observations for the x,y pair
                observations = np.column_stack([data[:, j], data[:, j + 1]])

                # Apply EM algorithm and smoothing
                smoothed_state_means, _ = kf.em(observations, n_iter=n_iter).smooth(observations)

                # Extract x,y positions from smoothed state means
                filtered_data[:, j] = alpha * smoothed_state_means[:, 0] + (1 - alpha) * data[:, j]
                filtered_data[:, j + 1] = (
                    alpha * smoothed_state_means[:, 1] + (1 - alpha) * data[:, j + 1]
                )

        return filtered_data

    except Exception as e:
        print(f"Error in Kalman smoothing: {str(e)}")
        return data  # Return original data if smoothing fails


def arima_smooth(data, order=(1, 0, 0)):
    """
    Applies ARIMA smoothing to the input data.

    Parameters:
        data (array-like): The input time series data. Can be 1D or 2D.
        order (tuple): The ARIMA model order (p, d, q):
            p: Number of AR terms (autoregressive)
            d: Number of differences
            q: Number of MA terms (moving average)

    Returns:
        filtered_data (array-like): The smoothed data.
    """
    data = np.asarray(data)

    # If data is 1D, process directly
    if data.ndim == 1:
        try:
            # Remove NaN values for ARIMA fitting
            valid_mask = ~np.isnan(data)
            if not np.any(valid_mask):
                return data  # Return original if all NaN

            valid_data = data[valid_mask]
            if len(valid_data) < max(order) + 1:
                print("Warning: Not enough data points for ARIMA model")
                return data

            model = ARIMA(valid_data, order=order)
            result = model.fit(disp=False)  # Suppress output

            # Create output array
            output = data.copy()
            output[valid_mask] = result.fittedvalues
            return output

        except Exception as e:
            print(f"Error in ARIMA smoothing: {str(e)}")
            return data  # Return original data if smoothing fails
    else:
        # For 2D data, apply ARIMA smoothing column by column
        smoothed = np.empty_like(data)
        for j in range(data.shape[1]):
            try:
                col_data = data[:, j]
                valid_mask = ~np.isnan(col_data)

                if not np.any(valid_mask):
                    smoothed[:, j] = col_data  # Keep original if all NaN
                    continue

                valid_data = col_data[valid_mask]
                if len(valid_data) < max(order) + 1:
                    print(f"Warning: Not enough data points for ARIMA model in column {j}")
                    smoothed[:, j] = col_data
                    continue

                model = ARIMA(valid_data, order=order)
                result = model.fit(disp=False)  # Suppress output

                smoothed[:, j] = col_data.copy()
                smoothed[valid_mask, j] = result.fittedvalues

            except Exception as e:
                print(f"Error in ARIMA smoothing for column {j}: {str(e)}")
                smoothed[:, j] = data[:, j]  # Keep original data for failed columns
        return smoothed


def process_file(file_path, dest_dir, config):
    try:
        # Get base filename without extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]

        # Create method suffix based on smoothing method
        method_suffix = "original"  # default if no smoothing
        if config["smooth_method"] != "none":
            method_suffix = config["smooth_method"]

        # Sanitize both base filename and method suffix
        sanitized_base = sanitize_filename(base_filename)
        sanitized_method = sanitize_filename(method_suffix)

        # Create output filename with sanitized method suffix
        output_filename = f"{sanitized_base}_{sanitized_method}.csv"
        output_path = os.path.join(dest_dir, output_filename)

        file_info = {
            "original_path": file_path,
            "original_filename": os.path.basename(file_path),
            "output_path": output_path,
            "warnings": [],
        }

        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)

        # Record original size
        file_info["original_size"] = len(df)
        file_info["original_columns"] = len(df.columns)

        # Detect if first column is Time
        first_col = df.columns[0]
        is_time_column = first_col.lower() in ["time", "t", "tempo"]

        # Get sample rate from config
        sample_rate = config.get("sample_rate")

        # Store original first column values
        original_first_col = df[first_col].copy()

        if is_time_column:
            print(f"Detected Time column: {first_col}")
            # If time column, preserve original values and calculate based on sample rate if provided
            if sample_rate is not None and sample_rate > 0:
                print(f"Using sample rate: {sample_rate} Hz to recalculate time")
                # Calculate time based on sample rate
                df[first_col] = np.arange(len(df)) / sample_rate
                # Store original time values for reference
            else:
                print("Using original time values from file")
                # Use original time values
                df[first_col] = original_first_col
                # Try to detect sample rate from time differences
                if len(df) > 1:
                    time_diffs = np.diff(df[first_col].dropna().values)
                    if len(time_diffs) > 0:
                        avg_diff = np.mean(time_diffs[time_diffs > 0])
                        if avg_diff > 0:
                            detected_sample_rate = 1.0 / avg_diff
                            print(
                                f"Detected sample rate from time column: {detected_sample_rate:.2f} Hz"
                            )
                            sample_rate = detected_sample_rate

            # For time column, we work with indices internally
            df["_internal_index"] = np.arange(len(df))
            min_frame = 0
            max_frame = len(df) - 1
            print(f"Time column detected. Data range: {min_frame} to {max_frame} rows")
        else:
            # For non-time columns, treat as frame numbers
            print(f"First column '{first_col}' treated as frame numbers")
            # Try to preserve as integers if possible, otherwise use as-is
            try:
                df[first_col] = df[first_col].astype(int)
                min_frame = int(df[first_col].min())
                max_frame = int(df[first_col].max())
                print(f"Frame range: {min_frame} to {max_frame}")

                # Create DataFrame with all frames
                all_frames = pd.DataFrame({first_col: range(min_frame, max_frame + 1)})

                # Merge with original data to identify gaps
                df = pd.merge(all_frames, df, on=first_col, how="left")
                print(f"Shape after adding missing frames: {df.shape}")
            except (ValueError, TypeError):
                # If conversion fails, treat as continuous data
                print(
                    f"Warning: Could not convert '{first_col}' to integers. Treating as continuous data."
                )
                df["_internal_index"] = np.arange(len(df))
                min_frame = 0
                max_frame = len(df) - 1

        # Count missing values
        file_info["total_missing"] = df.isna().sum().sum()
        file_info["columns_with_missing"] = {}

        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                file_info["columns_with_missing"][col] = missing

        # Apply padding if necessary
        padding_percent = config["padding"]
        print(f"Using exact padding value: {padding_percent}%")

        pad_len = 0
        if padding_percent > 0:
            pad_len = int(len(df) * padding_percent / 100)
            print(f"Applying padding of {pad_len} frames")

            if is_time_column:
                # For time column, calculate time values for padding
                if sample_rate is not None and sample_rate > 0:
                    # Calculate time before and after
                    first_time = df[first_col].iloc[0]
                    last_time = df[first_col].iloc[-1]
                    time_step = 1.0 / sample_rate

                    # Create time values for padding
                    pad_before_times = np.arange(
                        first_time - pad_len * time_step, first_time, time_step
                    )
                    pad_after_times = np.arange(
                        last_time + time_step,
                        last_time + (pad_len + 1) * time_step,
                        time_step,
                    )

                    pad_before = pd.DataFrame({first_col: pad_before_times})
                    pad_after = pd.DataFrame({first_col: pad_after_times})
                else:
                    # Use original time values pattern
                    first_time = df[first_col].iloc[0]
                    last_time = df[first_col].iloc[-1]
                    if len(df) > 1:
                        time_step = df[first_col].iloc[1] - df[first_col].iloc[0]
                    else:
                        time_step = 0.001  # Default fallback

                    pad_before_times = np.arange(
                        first_time - pad_len * time_step, first_time, time_step
                    )
                    pad_after_times = np.arange(
                        last_time + time_step,
                        last_time + (pad_len + 1) * time_step,
                        time_step,
                    )

                    pad_before = pd.DataFrame({first_col: pad_before_times})
                    pad_after = pd.DataFrame({first_col: pad_after_times})
            else:
                # For frame numbers
                pad_before = pd.DataFrame({first_col: range(min_frame - pad_len, min_frame)})
                pad_after = pd.DataFrame({first_col: range(max_frame + 1, max_frame + pad_len + 1)})

            # Fill padding with edge values for other columns
            for col in df.columns:
                if col != first_col:
                    # Use the value of the first record for initial padding
                    pad_before[col] = df[col].iloc[0]
                    # Use the value of the last record for final padding
                    pad_after[col] = df[col].iloc[-1]

            # Concatenate with padding
            df = pd.concat([pad_before, df, pad_after]).reset_index(drop=True)
            print(f"Shape after padding: {df.shape}")

        # Process numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(first_col)
        print(f"Processing {len(numeric_cols)} numeric columns")

        # STEP 1: Apply interpolation to each column
        print("\nSTEP 1: Applying interpolation to each column")
        for col in numeric_cols:
            print(f"\nProcessing column: {col}")
            nan_mask = df[col].isna()
            print(f"Found {nan_mask.sum()} NaN values in column {col}")

            if nan_mask.any() and config["interp_method"] not in ["none", "skip"]:
                print(f"Applying {config['interp_method']} interpolation")

                # Check maximum gap size
                max_gap = config["max_gap"]
                print(f"Using maximum gap size: {max_gap} frames")

                if max_gap > 0:
                    # Find gaps larger than max_gap
                    gap_starts = []
                    gap_ends = []
                    in_gap = False
                    gap_start = 0

                    for i in range(len(df)):
                        if df[col].isna().iloc[i] and not in_gap:
                            in_gap = True
                            gap_start = i
                        elif not df[col].isna().iloc[i] and in_gap:
                            in_gap = False
                            gap_end = i
                            gap_size = gap_end - gap_start
                            if gap_size > max_gap:
                                gap_starts.append(gap_start)
                                gap_ends.append(gap_end)

                    # Handle gap at the end of the data
                    if in_gap:
                        gap_end = len(df)
                        gap_size = gap_end - gap_start
                        if gap_size > max_gap:
                            gap_starts.append(gap_start)
                            gap_ends.append(gap_end)

                    # Create a copy of the column for interpolation
                    interpolated = df[col].copy()

                    # Apply interpolation only to gaps smaller than max_gap
                    if config["interp_method"] in ["linear", "hampel"]:
                        # Hampel uses linear interpolation after spike removal
                        interpolated = interpolated.interpolate(
                            method="linear", limit_direction="both"
                        )
                    elif config["interp_method"] == "nearest":
                        interpolated = interpolated.interpolate(
                            method="nearest", limit_direction="both"
                        )
                    elif config["interp_method"] == "cubic":
                        interpolated = interpolated.interpolate(
                            method="cubic", limit_direction="both"
                        )
                    elif config["interp_method"] == "kalman":
                        # For Kalman, we need to handle the entire column
                        # We'll apply it after this block
                        pass

                    # Restore NaN values for gaps larger than max_gap
                    for start, end in zip(gap_starts, gap_ends, strict=False):
                        interpolated.iloc[start:end] = np.nan

                    # Update the column with interpolated values
                    df[col] = interpolated

                    # Apply Kalman filter if selected
                    if config["interp_method"] == "kalman":
                        # Apply Kalman filter to the entire column
                        # This is a simplified approach - in practice, you might want
                        # to apply it only to specific regions
                        try:
                            kf = KalmanFilter(
                                transition_matrices=np.array([[1, 1], [0, 1]]),
                                observation_matrices=np.array([[1, 0]]),
                                initial_state_mean=np.zeros(2),
                                initial_state_covariance=np.eye(2),
                                transition_covariance=np.eye(2) * 0.1,
                                observation_covariance=np.array([[0.1]]),
                                n_dim_obs=1,
                                n_dim_state=2,
                            )

                            # Get non-NaN values for training
                            valid_data = np.array(df[col].dropna().values).reshape(-1, 1)
                            if len(valid_data) > 0:
                                # Train the filter
                                kf = kf.em(valid_data, n_iter=5)

                                # Apply smoothing
                                # Convert to numpy array and handle NaN values
                                data = df[col].to_numpy()
                                # Reshape for Kalman filter
                                data_reshaped = data.reshape(-1, 1)
                                # Apply smoothing
                                smoothed_state_means, _ = kf.smooth(data_reshaped)
                                df[col] = smoothed_state_means[:, 0]
                        except Exception as e:
                            print(f"Error applying Kalman filter: {str(e)}")
                else:  # No gap size limit
                    if config["interp_method"] in ["linear", "hampel"]:
                        # Hampel uses linear interpolation after spike removal
                        df[col] = df[col].interpolate(method="linear", limit_direction="both")

                    elif config["interp_method"] == "nearest":
                        df[col] = df[col].interpolate(method="nearest", limit_direction="both")
                    elif config["interp_method"] == "cubic":
                        df[col] = df[col].interpolate(method="cubic", limit_direction="both")
                    elif config["interp_method"] == "kalman":
                        try:
                            kf = KalmanFilter(
                                transition_matrices=np.array([[1, 1], [0, 1]]),
                                observation_matrices=np.array([[1, 0]]),
                                initial_state_mean=np.zeros(2),
                                initial_state_covariance=np.eye(2),
                                transition_covariance=np.eye(2) * 0.1,
                                observation_covariance=np.array([[0.1]]),
                                n_dim_obs=1,
                                n_dim_state=2,
                            )

                            # Get non-NaN values for training
                            valid_data = np.array(df[col].dropna().values).reshape(-1, 1)
                            if len(valid_data) > 0:
                                # Train the filter
                                kf = kf.em(valid_data, n_iter=5)

                                # Apply smoothing
                                # Convert to numpy array and handle NaN values
                                data = df[col].to_numpy()
                                # Reshape for Kalman filter
                                data_reshaped = data.reshape(-1, 1)
                                # Apply smoothing
                                smoothed_state_means, _ = kf.smooth(data_reshaped)
                                df[col] = smoothed_state_means[:, 0]
                        except Exception as e:
                            print(f"Error applying Kalman filter: {str(e)}")

                remaining_nans = df[col].isna().sum()
                print(f"After interpolation, {remaining_nans} NaN values remain")

        # STEP 2: Apply smoothing to each column
        if config["smooth_method"] != "none":
            print("\nSTEP 2: Applying smoothing to each column")

            # Check if we need to preserve NaNs (Skip interpolation mode)
            preserve_nans = config["interp_method"] == "skip"
            if preserve_nans:
                print("Note: Skip mode - NaN positions will be preserved after smoothing")

            for col in numeric_cols:
                print(f"\nSmoothing column: {col}")

                # Save original NaN mask for Skip mode
                original_nan_mask = df[col].isna().copy() if preserve_nans else None

                try:
                    data = df[col].values.copy().astype(float)

                    # For Skip mode, temporarily fill NaNs before smoothing
                    if preserve_nans and np.any(np.isnan(data)):
                        data_for_smoothing = (
                            pd.Series(data)
                            .interpolate(method="linear", limit_direction="both")
                            .ffill()
                            .bfill()
                            .values
                        )
                        print(
                            f"Temporarily interpolated {original_nan_mask.sum()} NaN values for smoothing"
                        )
                    else:
                        data_for_smoothing = data

                    # Apply the selected smoothing method
                    smoothed_result = None

                    if config["smooth_method"] == "savgol":
                        params = config["smooth_params"]
                        smoothed_result = savgol_smooth(
                            data_for_smoothing, params["window_length"], params["polyorder"]
                        )
                        print(
                            f"Applied Savitzky-Golay filter with window={params['window_length']}, order={params['polyorder']}"
                        )

                    elif config["smooth_method"] == "lowess":
                        params = config["smooth_params"]
                        smoothed_result = lowess_smooth(
                            data_for_smoothing, params["frac"], params["it"]
                        )
                        print(
                            f"Applied LOWESS smoothing with fraction={params['frac']}, iterations={params['it']}"
                        )

                    elif config["smooth_method"] == "kalman":
                        params = config["smooth_params"]
                        smoothed_result = kalman_smooth(
                            data_for_smoothing, params["n_iter"], params["mode"]
                        )
                        print(
                            f"Applied Kalman filter with {params['n_iter']} iterations in {params['mode']} mode"
                        )

                    elif config["smooth_method"] == "butterworth":
                        params = config["smooth_params"]
                        try:
                            fs = float(params["fs"])
                            cutoff = float(params["cutoff"])

                            # Ensure cutoff frequency is valid
                            if cutoff >= fs / 2:
                                cutoff = fs / 2 - 1
                                print(f"Warning: Adjusted cutoff frequency to {cutoff} Hz")

                            # Use butter_filter from filter_utils.py
                            smoothed_result = butter_filter(
                                data_for_smoothing,
                                fs=fs,
                                filter_type="low",
                                cutoff=cutoff,
                                order=4,
                                padding=True,
                            )
                            print(f"Applied Butterworth filter with cutoff={cutoff} Hz, fs={fs} Hz")

                        except Exception as e:
                            print(f"Error filtering column {col}: {str(e)}")
                            print("Keeping original data for this column")
                            smoothed_result = data_for_smoothing

                    elif config["smooth_method"] == "splines":
                        params = config["smooth_params"]
                        smoothed_result = spline_smooth(
                            data_for_smoothing, s=float(params["smoothing_factor"])
                        )
                        print(
                            f"Applied Spline smoothing with smoothing factor={params['smoothing_factor']}"
                        )

                    elif config["smooth_method"] == "arima":
                        params = config["smooth_params"]
                        order = (int(params["p"]), int(params["d"]), int(params["q"]))
                        smoothed_result = arima_smooth(data_for_smoothing, order=order)
                        print(f"Applied ARIMA filter with order={order}")

                    elif config["smooth_method"] == "median":
                        params = config["smooth_params"]
                        kernel_size = int(params.get("kernel_size", 5))
                        smoothed_result = median_filter_smooth(
                            data_for_smoothing, kernel_size=kernel_size
                        )
                        print(f"Applied Moving Median filter with kernel_size={kernel_size}")

                    elif config["smooth_method"] == "hampel":
                        params = config["smooth_params"]
                        window_size = int(params.get("window_size", 7))
                        n_sigmas = float(params.get("n_sigmas", 3))
                        smoothed_result = hampel_filter(
                            data_for_smoothing, window_size=window_size, n_sigmas=n_sigmas
                        )
                        print(
                            f"Applied Hampel filter with window_size={window_size}, n_sigmas={n_sigmas}"
                        )

                    # Apply the smoothed result
                    if smoothed_result is not None:
                        # Restore NaN positions for Skip mode
                        if (
                            preserve_nans
                            and original_nan_mask is not None
                            and original_nan_mask.any()
                        ):
                            smoothed_result = np.asarray(smoothed_result).astype(float)
                            smoothed_result[original_nan_mask.values] = np.nan
                            print(f"Restored {original_nan_mask.sum()} NaN positions")

                        df[col] = smoothed_result

                except Exception as e:
                    error_msg = f"Error smoothing column {col}: {str(e)}"
                    print(error_msg)
                    file_info["warnings"].append(error_msg)

        # Remove padding
        print("\nRemoving padding")
        if is_time_column:
            # For time column, remove padding based on original time range
            original_length = file_info["original_size"]
            pad_len = int(original_length * padding_percent / 100) if padding_percent > 0 else 0

            if pad_len > 0:
                df = df.iloc[pad_len:-pad_len].reset_index(drop=True)

            # Recalculate time for the final data
            if sample_rate is not None and sample_rate > 0:
                # Use sample rate to calculate time
                df[first_col] = np.arange(len(df)) / sample_rate
                print(f"Recalculated time using sample rate: {sample_rate} Hz")
            else:
                # Use original time values pattern
                if len(original_first_col) > 0:
                    first_time = float(original_first_col.iloc[0])
                    if len(original_first_col) > 1:
                        # Detect time step from original
                        time_step = float(original_first_col.iloc[1] - original_first_col.iloc[0])
                        df[first_col] = np.arange(len(df)) * time_step + first_time
                        print(f"Recalculated time using detected time step: {time_step} s")
                    else:
                        # Single value, use sample rate if available
                        if sample_rate is not None and sample_rate > 0:
                            df[first_col] = np.arange(len(df)) / sample_rate
                        else:
                            df[first_col] = (
                                np.arange(len(df)) * 0.001 + first_time
                            )  # Default 1ms step
                else:
                    # Fallback: use sample rate or default
                    if sample_rate is not None and sample_rate > 0:
                        df[first_col] = np.arange(len(df)) / sample_rate
                    else:
                        df[first_col] = np.arange(len(df)) * 0.001  # Default 1ms step
        else:
            # For frame numbers
            print(f"Keeping only frames from {min_frame} to {max_frame}")
            df = df[df[first_col].between(min_frame, max_frame)].reset_index(drop=True)

        print(f"Final shape after removing padding: {df.shape}")

        # Remove internal index column if it exists
        if "_internal_index" in df.columns:
            df = df.drop(columns=["_internal_index"])

        # Detect float format from original file
        float_format = detect_float_format(file_info["original_path"])
        print(f"Using float format: {float_format}")

        # For time column, ensure proper precision
        time_precision = None
        if is_time_column:
            if sample_rate is not None and sample_rate > 0:
                # Use precision based on sample rate (same logic as readc3d_export.py)
                try:
                    from .readc3d_export import get_time_precision
                except ImportError:
                    try:
                        from readc3d_export import get_time_precision
                    except ImportError:
                        # Fallback: define function locally
                        import math

                        def get_time_precision(freq):
                            if freq <= 1000:
                                return 3
                            else:
                                interval = 1.0 / freq
                                decimal_places = max(3, int(math.ceil(-math.log10(interval))))
                                return decimal_places

                time_precision = get_time_precision(sample_rate)
                print(f"Time column will be formatted with {time_precision} decimal places")
            else:
                # Try to detect precision from original time values
                if len(original_first_col) > 1:
                    # Check decimal places in original time values
                    sample_time = float(original_first_col.iloc[1])
                    time_str = f"{sample_time:.10f}".rstrip("0").rstrip(".")
                    if "." in time_str:
                        time_precision = len(time_str.split(".")[1])
                    else:
                        time_precision = 3  # Default
                    print(
                        f"Detected time precision from original file: {time_precision} decimal places"
                    )

        # Save processed DataFrame
        print(f"\nSaving processed file to: {output_path}")

        # If time column has specific precision, format it separately
        if is_time_column and time_precision is not None:
            # Create a copy for saving
            df_to_save = df.copy()
            # Format time column with specific precision
            df_to_save[first_col] = df_to_save[first_col].apply(lambda x: f"{x:.{time_precision}f}")
            # Save with default float format for other columns
            df_to_save.to_csv(output_path, index=False, float_format=float_format)
        else:
            # Save normally
            df.to_csv(output_path, index=False, float_format=float_format)

        print("File saved successfully!")

        return file_info

    except Exception as e:
        # Return basic info with error in case of failure
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        sanitized_base = sanitize_filename(base_name)
        output_filename = f"{sanitized_base}_processed.csv"
        output_path = os.path.join(dest_dir, output_filename)

        return {
            "original_path": file_path,
            "original_filename": filename,
            "output_path": output_path,
            "warnings": [f"Error processing file: {str(e)}"],
            "error": True,
            "original_size": 0,
            "original_columns": 0,
            "total_missing": 0,
            "columns_with_missing": {},
        }


def detect_float_format(original_path):
    """Detecta o formato de float com base no número máximo de casas decimais do arquivo original.

    Args:
        original_path: Caminho do arquivo CSV original

    Returns:
        str: String de formato para float (ex: '%.6f')
    """
    try:
        original_df = pd.read_csv(original_path)
        max_decimals = 0
        for col in original_df.select_dtypes(include=[np.number]).columns:
            # Considere somente valores não-nulos
            valid_values = original_df[col].dropna().astype(str)
            if not valid_values.empty:
                # Extrai a parte decimal usando expressão regular
                decimals = valid_values.str.extract(r"\.(\d+)", expand=False)
                if not decimals.empty:
                    # Calcula o número máximo de dígitos encontrados na parte decimal
                    col_max = decimals.dropna().str.len().max()
                    if pd.notna(col_max) and col_max > max_decimals:
                        max_decimals = col_max
        # Se encontrou casas decimais, constrói o formato; caso contrário, usa 6
        return f"%.{int(max_decimals)}f" if max_decimals > 0 else "%.6f"
    except Exception as e:
        print(f"Error: Could not detect float format: {str(e)}")
        return "%.6f"


def generate_report(dest_dir, config, processed_files):
    """
    Generates a detailed processing report and saves it to a text file.

    Args:
        dest_dir: Directory where the processed files were saved
        config: Configuration settings used in processing
        processed_files: List of dictionaries with information about processed files
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = os.path.join(dest_dir, "processing_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("PROCESSING REPORT - VAILA INTERPOLATION AND SMOOTHING TOOL\n")
        f.write(f"Date and Time: {timestamp}\n")
        f.write("=" * 80 + "\n\n")

        # General configuration
        f.write("GENERAL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Gap Filling Method: {config['interp_method']}\n")

        if config["interp_method"] not in ["none", "skip"]:
            f.write(f"Maximum Gap Size to Fill: {config['max_gap']} frames")
            if config["max_gap"] == 0:
                f.write(" (no limit - all gaps filled)\n")
            else:
                f.write("\n")

        f.write(f"Smoothing Method: {config['smooth_method']}\n")
        f.write(f"Padding: {config['padding']}%\n")
        f.write(f"Split Data: {'Yes' if config['do_split'] else 'No'}\n\n")

        # Specific parameters
        if config["smooth_method"] != "none":
            f.write("SMOOTHING PARAMETERS\n")
            f.write("-" * 80 + "\n")

            if config["smooth_method"] == "savgol":
                f.write(f"Window Length: {config['smooth_params'].get('window_length', 7)}\n")
                f.write(f"Polynomial Order: {config['smooth_params'].get('polyorder', 2)}\n")

            elif config["smooth_method"] == "lowess":
                f.write(f"Fraction: {config['smooth_params'].get('frac', 0.3)}\n")
                f.write(f"Iterations: {config['smooth_params'].get('it', 3)}\n")

            elif config["smooth_method"] == "kalman":
                f.write(f"EM Iterations: {config['smooth_params'].get('n_iter', 5)}\n")

            elif config["smooth_method"] == "butterworth":
                f.write(f"Cutoff Frequency: {config['smooth_params'].get('cutoff', 10)} Hz\n")
                f.write(f"Sampling Frequency: {config['smooth_params'].get('fs', 100)} Hz\n")

            elif config["smooth_method"] == "splines":
                f.write(
                    f"Smoothing Factor: {config['smooth_params'].get('smoothing_factor', 1.0)}\n"
                )

            elif config["smooth_method"] == "median":
                f.write(f"Kernel Size: {config['smooth_params'].get('kernel_size', 5)}\n")

            elif config["smooth_method"] == "hampel":
                f.write(f"Window Size: {config['smooth_params'].get('window_size', 7)}\n")
                f.write(f"N Sigmas: {config['smooth_params'].get('n_sigmas', 3.0)}\n")

            f.write("\n")

        # Processed files
        f.write("PROCESSED FILES\n")
        f.write("-" * 80 + "\n")

        for idx, file_info in enumerate(processed_files, 1):
            f.write(f"File {idx}: {file_info['original_filename']}\n")
            f.write(f"  - Original Path: {file_info['original_path']}\n")
            f.write(
                f"  - Original Size: {file_info['original_size']} frames, {file_info['original_columns']} columns\n"
            )
            f.write(f"  - Total Missing Values: {file_info['total_missing']}\n")
            f.write(f"  - Processed Output: {file_info['output_path']}\n")

            # If split, show both parts
            if config["do_split"] and "output_part2_path" in file_info:
                f.write(
                    f"  - Split Part 1: {file_info['output_part1_path']} ({file_info['part1_size']} frames)\n"
                )
                f.write(
                    f"  - Split Part 2: {file_info['output_part2_path']} ({file_info['part2_size']} frames)\n"
                )

            # Details of columns with interpolated values
            if file_info["columns_with_missing"]:
                f.write("  - Columns with missing values:\n")
                for col_name, missing_count in file_info["columns_with_missing"].items():
                    f.write(f"    - {col_name}: {missing_count} missing values\n")

            # Additional information if applicable
            if file_info.get("warnings"):
                f.write("  - Warnings during processing:\n")
                for warning in file_info["warnings"]:
                    f.write(f"    - {warning}\n")

            f.write("\n")

        # Add smoothing verification section
        if config["smooth_method"] != "none":
            f.write("SMOOTHING VERIFICATION\n")
            f.write("-" * 80 + "\n")
            f.write(
                "Comparing first 10 values of the first numeric column between original and processed files:\n\n"
            )

            for idx, file_info in enumerate(processed_files, 1):
                try:
                    # Read original and processed files
                    original_df = pd.read_csv(file_info["original_path"])
                    processed_df = pd.read_csv(file_info["output_path"])

                    # Find first numeric column (excluding the first column which is usually frame number)
                    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:  # Skip first column if it's numeric
                        first_numeric_col = numeric_cols[1]
                    else:
                        first_numeric_col = numeric_cols[0]

                    # Get first 10 values from both files
                    original_values = original_df[first_numeric_col].head(10).values
                    processed_values = processed_df[first_numeric_col].head(10).values

                    # Calculate percentage differences for first 10 values
                    differences = np.abs(
                        (np.array(processed_values) - np.array(original_values))
                        / np.array(original_values)
                        * 100
                    )

                    f.write(f"File {idx}: {file_info['original_filename']}\n")
                    f.write(f"Column: {first_numeric_col}\n")
                    f.write(
                        "Original Values: "
                        + ", ".join([f"{x:.6f}" for x in original_values])
                        + "\n"
                    )
                    f.write(
                        "Processed Values: "
                        + ", ".join([f"{x:.6f}" for x in processed_values])
                        + "\n"
                    )
                    f.write(
                        "Percentage Differences: "
                        + ", ".join([f"{x:.2f}%" for x in differences])
                        + "\n"
                    )
                    f.write(f"Average Difference (first 10): {np.mean(differences):.2f}%\n")
                    f.write("-" * 40 + "\n")

                    # Complete column comparison
                    f.write("\nComplete Column Analysis:\n")
                    f.write("-" * 40 + "\n")

                    # Get all values from both columns
                    all_original = original_df[first_numeric_col].values
                    all_processed = processed_df[first_numeric_col].values

                    # Calculate differences for all values
                    all_differences = np.abs(
                        (np.array(all_processed) - np.array(all_original))
                        / np.array(all_original)
                        * 100
                    )

                    # Calculate statistics
                    mean_diff = np.mean(all_differences)
                    std_diff = np.std(all_differences)
                    max_diff = np.max(all_differences)
                    min_diff = np.min(all_differences)

                    # Write statistics
                    f.write(f"Total number of values compared: {len(all_differences)}\n")
                    f.write(f"Mean difference: {mean_diff:.2f}%\n")
                    f.write(f"Standard deviation: {std_diff:.2f}%\n")
                    f.write(f"Maximum difference: {max_diff:.2f}%\n")
                    f.write(f"Minimum difference: {min_diff:.2f}%\n")

                    # Add smoothing effectiveness summary
                    f.write("\nSmoothing Effectiveness Summary:\n")
                    if mean_diff > 0.01:  # If there's significant change
                        f.write(
                            "[OK] Smoothing was effectively applied (significant changes detected)\n"
                        )
                        if mean_diff < 1.0:
                            f.write("  - Light smoothing effect\n")
                        elif mean_diff < 5.0:
                            f.write("  - Moderate smoothing effect\n")
                        else:
                            f.write("  - Strong smoothing effect\n")
                    else:
                        f.write(
                            "[WARNING] Warning: Very small changes detected. Verify if smoothing was properly applied.\n"
                        )

                    f.write("\n" + "=" * 80 + "\n\n")

                except Exception as e:
                    print(f"Error: {str(e)}")
                    f.write(f"File {idx}: {file_info['original_filename']}\n")
                    f.write(f"Error during verification: {str(e)}\n")
                    f.write("-" * 40 + "\n\n")

        # Additional information
        f.write("PROCESSING DETAILS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Files Processed: {len(processed_files)}\n")
        f.write(f"Output Directory: {dest_dir}\n")
        f.write(f"Python Version: {sys.version.split()[0]}\n")
        f.write(f"Processing Completed at: {timestamp}\n\n")

        # Instructions for the user
        f.write("NOTES\n")
        f.write("-" * 80 + "\n")
        f.write(
            "- Processed files include the interpolation and smoothing method in their filenames.\n"
        )
        f.write("- Files are saved with the same number of decimal places as the original files.\n")
        f.write(
            "- The output directory includes the processing methods in its name for easy reference.\n"
        )
        f.write("- For questions or issues, please contact: paulosantiago@usp.br\n")

    print(f"Generated detailed processing report: {report_path}")
    return report_path


def run_fill_split_dialog(parent=None):
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting script: interp_smooth_split.py")
    print("================================================")

    # Open configuration dialog
    config_dialog = InterpolationConfigDialog(parent=parent)

    print("GUI window should now be visible...")

    # Wait for dialog to complete
    # When running standalone (no parent), we need mainloop() to start the event loop.
    # When called from vaila.py (parent exists), use wait_window() since
    # the parent already has a running event loop.
    if parent is None:
        config_dialog.window.mainloop()
    else:
        config_dialog.window.wait_window()

    config = None
    if hasattr(config_dialog, "result") and config_dialog.result is not None:
        config = config_dialog.result
    else:
        # No result from dialog: try smooth_config.toml in cwd so batch can use last saved config
        path_cwd = os.path.join(os.getcwd(), SMOOTH_CONFIG_FILENAME)
        if os.path.isfile(path_cwd):
            config = load_smooth_config_for_analysis(path_cwd)
            print(f"Using config from {path_cwd}")
    if config is None:
        print("Operation canceled by user (no config and no smooth_config.toml).")
        print("================================================")
        return

    # Select source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        print("Operation canceled by user.")
        print("================================================")
        return

    # Prefer smooth_config.toml in source directory if present (per-project config)
    path_in_source = os.path.join(source_dir, SMOOTH_CONFIG_FILENAME)
    if os.path.isfile(path_in_source):
        config = load_smooth_config_for_analysis(path_in_source)
        print(f"Using config from {path_in_source}")

    run_batch(source_dir, config, dest_dir=None, use_messagebox=True)


def run_batch(source_dir, config, dest_dir=None, use_messagebox=True):
    """
    Process all CSV files in source_dir with the given config.
    If dest_dir is None, create a timestamped subdir inside source_dir.
    If use_messagebox is False (CLI), print results instead of showing dialogs.
    Returns (dest_dir, processed_files, report_path or None).
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    interp_name = config["interp_method"]
    smooth_info = "no_smooth"
    if config["smooth_method"] != "none":
        smooth_info = config["smooth_method"]
        try:
            if config["smooth_method"] == "butterworth":
                smooth_info += f"_cut{config['smooth_params'].get('cutoff', 10)}"
            elif config["smooth_method"] == "savgol":
                smooth_info += f"_w{config['smooth_params'].get('window_length', 7)}p{config['smooth_params'].get('polyorder', 2)}"
            elif config["smooth_method"] == "lowess":
                frac = config["smooth_params"].get("frac", 0.3)
                it = config["smooth_params"].get("it", 3)
                smooth_info += f"_frac{int(frac * 100)}_it{it}"
            elif config["smooth_method"] == "kalman":
                smooth_info += f"_iter{config['smooth_params'].get('n_iter', 5)}_mode{config['smooth_params'].get('mode', 1)}"
            elif config["smooth_method"] == "splines":
                smooth_info += f"_s{config['smooth_params'].get('smoothing_factor', 1.0)}"
            elif config["smooth_method"] == "arima":
                p = config["smooth_params"].get("p", 1)
                d = config["smooth_params"].get("d", 0)
                q = config["smooth_params"].get("q", 0)
                smooth_info += f"_p{p}d{d}q{q}"
        except Exception:
            smooth_info = config["smooth_method"]
    sanitized_interp = sanitize_filename(interp_name)
    sanitized_smooth = sanitize_filename(smooth_info)
    if dest_dir is None:
        dest_dir_name = f"processed_{sanitized_interp}_{sanitized_smooth}_{timestamp}"
        dest_dir = os.path.join(source_dir, dest_dir_name)
    os.makedirs(dest_dir, exist_ok=True)
    save_smooth_config_toml(config, os.path.join(dest_dir, SMOOTH_CONFIG_FILENAME))

    processed_files = []
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            try:
                file_info = process_file(os.path.join(source_dir, filename), dest_dir, config)
                if file_info is not None:
                    processed_files.append(file_info)
                else:
                    print(f"Warning: No information returned for file {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                processed_files.append(
                    {
                        "original_path": os.path.join(source_dir, filename),
                        "original_filename": filename,
                        "warnings": [f"Error: {str(e)}"],
                        "error": True,
                        "original_size": 0,
                        "original_columns": 0,
                        "total_missing": 0,
                        "columns_with_missing": {},
                        "output_path": None,
                    }
                )
    processed_files = [pf for pf in processed_files if pf is not None]

    report_path = None
    if processed_files:
        report_path = generate_report(dest_dir, config, processed_files)
        if use_messagebox:
            messagebox.showinfo(
                "Complete",
                f"Processing complete. Results saved in {dest_dir}\nReport: {report_path}",
            )
        else:
            print(f"Processing complete. Results saved in {dest_dir}")
            print(f"Report: {report_path}")
    else:
        if use_messagebox:
            messagebox.showwarning("Warning", "No files were successfully processed.")
        else:
            print("Warning: No CSV files were successfully processed.")
    return dest_dir, processed_files, report_path


def _cli_run():
    """CLI entry: argparse for --input, --output, --config (TOML)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="interp_smooth_split.py",
        description="""
Interpolate, smooth, and split CSV data using configurable methods.
Configuration is read from a TOML file (smooth_config.toml) which can be
created via the GUI 'Save Template' button or manually.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Open GUI (default when no arguments):
  %(prog)s
  %(prog)s --gui

  # CLI: process CSVs in ./data using config from that dir or cwd:
  %(prog)s -i ./data

  # CLI: process with explicit output directory:
  %(prog)s -i ./data -o ./results

  # CLI: process with explicit config file:
  %(prog)s -i ./data -c ./smooth_config.toml

  # CLI: all flags together:
  %(prog)s -i ./data -o ./out -c ./my_config.toml

Workflow:
  1. Run in GUI mode first to configure parameters and test them.
  2. Click 'Save Template' to create a smooth_config.toml.
  3. Use CLI mode (-i, -o, -c) for automated batch processing.
        """,
    )
    parser.add_argument("-i", "--input", metavar="DIR", help="Input directory containing CSV files")
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        help="Output directory (default: timestamped subdir inside input)",
    )
    parser.add_argument(
        "-c",
        "--config",
        metavar="TOML",
        help="Path to smooth_config.toml (default: smooth_config.toml in input dir or cwd)",
    )
    parser.add_argument("--gui", action="store_true", help="Launch GUI instead of CLI")
    args = parser.parse_args()

    if args.gui or (not args.input and not args.output and not args.config):
        run_fill_split_dialog()
        return

    if not args.input:
        print("Error: --input is required for CLI mode. Use --gui to open the graphical interface.")
        sys.exit(1)

    source_dir = os.path.abspath(args.input)
    if not os.path.isdir(source_dir):
        print(f"Error: Input is not a directory: {source_dir}")
        sys.exit(1)

    config = None
    if args.config and os.path.isfile(args.config):
        config = load_smooth_config_for_analysis(os.path.abspath(args.config))
        print(f"Using config from {args.config}")
    else:
        path_in_source = os.path.join(source_dir, SMOOTH_CONFIG_FILENAME)
        if os.path.isfile(path_in_source):
            config = load_smooth_config_for_analysis(path_in_source)
            print(f"Using config from {path_in_source}")
        else:
            path_cwd = os.path.join(os.getcwd(), SMOOTH_CONFIG_FILENAME)
            if os.path.isfile(path_cwd):
                config = load_smooth_config_for_analysis(path_cwd)
                print(f"Using config from {path_cwd}")
    if config is None:
        print(
            "Error: No configuration found. Create smooth_config.toml (e.g. via GUI Apply) or pass --config PATH."
        )
        sys.exit(1)

    dest_dir = os.path.abspath(args.output) if args.output else None
    run_batch(source_dir, config, dest_dir=dest_dir, use_messagebox=False)


if __name__ == "__main__":
    _cli_run()
