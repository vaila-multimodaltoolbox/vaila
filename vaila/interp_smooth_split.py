"""
===============================================================================
interp_smooth_split.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 14 October 2024
Update Date: 16 September 2025
Version: 0.0.7
Python Version: 3.12.9

Description:
------------
This script provides functionality to fill missing data in CSV files using
linear interpolation, Kalman filter, Savitzky-Golay filter, nearest value fill,
or to split data into a separate CSV file. It is intended for use in biomechanical
data analysis, where gaps in time-series data can be filled and datasets can be
split for further analysis.

Key Features:
-------------
1. **Data Splitting**:
   - Splits CSV files into two halves for easier data management and analysis.
2. **Padding**:
   - Pads the data with the last valid value to avoid edge effects.

    padding_length = 0.1 * len(data)  # 10% padding of the data length before and after the data
    padded_data = np.concatenate(
        [data[:padding_length][::-1], data, data[-padding_length:][::-1]]
    )
    y = Filtering or smoothing the data method
    return y[padding_length:-padding_length]

3. **Filtering/Smoothing**:
   - Applies Kalman filter, Savitzky-Golay filter, or nearest value fill to
     numerical data.
4. **Gap Filling with Interpolation**:
   - Fills gaps in numerical data using linear interpolation, Kalman filter,
     Savitzky-Golay filter, nearest value fill, or leaves NaNs as is.
   - Only the missing data (NaNs) are filled; existing data remains unchanged.

Usage:
------
- Run this script to launch a graphical user interface (GUI) that provides options
  to perform interpolation on CSV files or to split them into two parts.
- The filled or split files are saved in new directories to avoid overwriting the
  original files.

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from tkinter import filedialog, messagebox
from scipy.interpolate import UnivariateSpline
import tkinter as tk
from rich import print
from statsmodels.tsa.arima.model import ARIMA
import sys
import toml
import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Import filter_utils - handle both relative and absolute imports
try:
    from .filter_utils import butter_filter
except ImportError:
    try:
        from filter_utils import butter_filter
    except ImportError:
        print(
            "Warning: filter_utils not found. Butterworth filtering will be disabled."
        )

        def butter_filter(data, **kwargs):
            print("Butterworth filter not available - filter_utils not found")
            return data


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
        f.write(
            "# 1. Edit the values below to customize your interpolation and smoothing.\n"
        )
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
        f.write(
            "# ================================================================\n\n"
        )

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
            '# Options: "none", "savgol", "lowess", "kalman", "butterworth", "splines", "arima"\n'
        )
        f.write("#   - none: no smoothing\n")
        f.write("#   - savgol: Savitzky-Golay filter (preserves peaks)\n")
        f.write("#   - lowess: Local regression (for noisy data)\n")
        f.write("#   - kalman: Kalman filter (for tracking)\n")
        f.write("#   - butterworth: Butterworth filter (biomechanics standard)\n")
        f.write("#   - splines: Spline smoothing (very smooth curves)\n")
        f.write("#   - arima: ARIMA model (time series)\n")
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
        f.write(
            "fs = 30.0           # Sampling frequency (video FPS, e.g. 30.0, 100.0)\n"
        )
        f.write("# Splines:\n")
        f.write(
            "smoothing_factor = 1.0   # 0 = no smoothing, 1 = moderate, 10+ = strong\n"
        )
        f.write("# ARIMA:\n")
        f.write("p = 1                # AR order\n")
        f.write("d = 0                # Difference order\n")
        f.write("q = 0                # MA order\n\n")

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
    print(f"Configuration saved in: {filepath}")


def load_config_from_toml(filepath):
    """Load the configuration from a TOML file and return a dictionary."""
    with open(filepath, "r", encoding="utf-8") as f:
        config = toml.load(f)
    print(f"Configuration loaded from: {filepath}")
    return config


class InterpolationConfigDialog:
    def __init__(self, parent=None):
        # Remove singleton pattern - it's causing issues
        self.result = None

        # Create root window if no parent provided
        if parent is None:
            self.root = tk.Tk()
            self.root.title("Interpolation and Smoothing Tool")
            self.window = self.root
        else:
            self.root = parent
            self.window = tk.Toplevel(parent)
            self.window.title("Interpolation Configuration")
            self.window.transient(parent)
            self.window.grab_set()

        # Configure window
        self.window.geometry("1400x900")
        self.window.minsize(1200, 800)
        self.window.resizable(True, True)

        # Initialize variables
        self.setup_variables()

        # Create the dialog content
        self.create_dialog_content()

        # Center window
        self.center_window()

        # Bind close event
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        # Force window to be visible
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
        self.window.update()

    def setup_variables(self):
        """Initialize all StringVar variables"""
        self.savgol_window = tk.StringVar(value="7")
        self.savgol_poly = tk.StringVar(value="3")
        self.lowess_frac = tk.StringVar(value="0.3")
        self.lowess_it = tk.StringVar(value="3")
        self.butter_cutoff = tk.StringVar(value="10")
        self.butter_fs = tk.StringVar(value="100")
        self.kalman_iterations = tk.StringVar(value="5")
        self.kalman_mode = tk.StringVar(value="1")
        self.spline_smoothing = tk.StringVar(value="1.0")
        self.arima_p = tk.StringVar(value="1")
        self.arima_d = tk.StringVar(value="0")
        self.arima_q = tk.StringVar(value="0")
        self.loaded_toml = None
        self.use_toml = False
        self.test_data = None
        self.test_data_path = None

    def center_window(self):
        """Center the window on the screen"""
        self.window.update_idletasks()

        # Get window dimensions
        width = self.window.winfo_reqwidth()
        height = self.window.winfo_reqheight()

        # Get screen dimensions
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        # Calculate center position
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)

        # Set the geometry
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def on_window_resize(self, event):
        """Handle window resize events for better responsiveness"""
        try:
            # Update canvas scroll region when window is resized
            if hasattr(self, "canvas"):
                self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            # Ensure window stays resizable
            if event.widget == self.window:
                self.window.resizable(True, True)

        except Exception as e:
            print(f"Error in window resize handler: {e}")

    def create_dialog_content(self):
        """Create the dialog content - simplified version"""
        print("Creating dialog content...")

        # Main container
        main_container = tk.Frame(self.window, padx=15, pady=15)
        main_container.pack(fill="both", expand=True)

        # Add main title at the top
        title_frame = tk.Frame(main_container)
        title_frame.pack(fill="x", pady=(0, 15))

        title_label = tk.Label(
            title_frame,
            text="Interpolation and Smoothing Configuration Tool",
            font=("Arial", 16, "bold"),
            fg="#2E7D32",
        )
        title_label.pack()

        subtitle_label = tk.Label(
            title_frame,
            text="Configure gap filling and smoothing parameters for your data",
            font=("Arial", 11),
            fg="#666666",
        )
        subtitle_label.pack()

        # Create scrollable frame
        canvas = tk.Canvas(main_container, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            main_container, orient="vertical", command=canvas.yview
        )
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Two column layout
        left_column = tk.Frame(scrollable_frame)
        right_column = tk.Frame(scrollable_frame)

        left_column.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        right_column.grid(row=0, column=1, sticky="nsew", padx=(20, 0))

        scrollable_frame.grid_columnconfigure(0, weight=1)
        scrollable_frame.grid_columnconfigure(1, weight=1)

        # LEFT COLUMN - Method Selection
        self.create_interpolation_section(left_column)
        self.create_smoothing_section(left_column)
        self.create_split_section(left_column)

        # RIGHT COLUMN - Parameters
        self.create_parameters_section(right_column)
        self.create_padding_section(right_column)
        self.create_gap_section(right_column)

        # Bottom buttons
        self.create_buttons(scrollable_frame)

        print("Dialog content creation completed")

    def create_interpolation_section(self, parent):
        """Create interpolation method selection"""
        frame = tk.LabelFrame(
            parent,
            text="Gap Filling Method",
            padx=15,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        frame.pack(fill="x", pady=(0, 15))

        methods_text = """1 - Linear Interpolation (simple, works well for most cases)
2 - Cubic Spline (smooth transitions between points)
3 - Nearest Value (use closest available value)
4 - Kalman Filter (good for movement data, models physics)
5 - None (leave gaps as NaN)
6 - Skip (keep original data, apply only smoothing)"""

        tk.Label(frame, text=methods_text, justify="left", font=("Arial", 11)).pack(
            anchor="w", padx=10, pady=5
        )

        tk.Label(
            frame, text="Enter gap filling method (1-6):", font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=10, pady=8)
        self.interp_entry = tk.Entry(frame, font=("Arial", 12))
        self.interp_entry.insert(0, "1")
        self.interp_entry.pack(fill="x", padx=10, pady=5)

    def create_smoothing_section(self, parent):
        """Create smoothing method selection"""
        frame = tk.LabelFrame(
            parent,
            text="Smoothing Method",
            padx=15,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        frame.pack(fill="x", pady=(0, 15))

        methods_text = """1 - None (no smoothing)
2 - Savitzky-Golay Filter (preserves peaks and valleys)
3 - LOWESS (adapts to local trends)
4 - Kalman Filter (state estimation with noise reduction)
5 - Butterworth Filter (4th order, frequency domain filtering)
6 - Spline Smoothing (flexible curve fitting)
7 - ARIMA (time series modeling and filtering)"""

        tk.Label(frame, text=methods_text, justify="left", font=("Arial", 11)).pack(
            anchor="w", padx=10, pady=5
        )

        tk.Label(
            frame, text="Enter smoothing method (1-7):", font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=10, pady=8)
        self.smooth_entry = tk.Entry(frame, font=("Arial", 12))
        self.smooth_entry.insert(0, "1")
        self.smooth_entry.pack(fill="x", padx=10, pady=5)

        # Update parameters button
        tk.Button(
            frame,
            text="Update Parameters",
            command=self.update_params_frame,
            font=("Arial", 11, "bold"),
            height=2,
        ).pack(pady=10, padx=10, fill="x")

    def create_split_section(self, parent):
        """Create split configuration"""
        frame = tk.LabelFrame(
            parent,
            text="Split Configuration",
            padx=15,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        frame.pack(fill="x", pady=(0, 15))

        self.split_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame,
            text="Split data into two parts",
            variable=self.split_var,
            font=("Arial", 11),
        ).pack(anchor="w", padx=10, pady=5)

    def create_parameters_section(self, parent):
        """Create parameters section"""
        self.params_frame = tk.LabelFrame(
            parent,
            text="Method Parameters",
            padx=15,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        self.params_frame.pack(fill="both", expand=True, pady=(0, 15))

        self.params_widgets = []
        self.param_entries = {}

        # Initialize with empty parameters
        self.update_params_frame()

    def create_padding_section(self, parent):
        """Create padding configuration"""
        frame = tk.LabelFrame(
            parent,
            text="Padding Configuration",
            padx=15,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        frame.pack(fill="x", pady=(0, 15))

        tk.Label(
            frame, text="Padding length (% of data):", font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        self.padding_entry = tk.Entry(frame, font=("Arial", 12))
        self.padding_entry.insert(0, "10")
        self.padding_entry.pack(fill="x", padx=10, pady=5)

    def create_gap_section(self, parent):
        """Create gap configuration"""
        frame = tk.LabelFrame(
            parent,
            text="Gap Configuration",
            padx=15,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        frame.pack(fill="x", pady=(0, 15))

        tk.Label(
            frame, text="Maximum gap size to fill (frames):", font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        self.max_gap_entry = tk.Entry(frame, font=("Arial", 12))
        self.max_gap_entry.insert(0, "60")
        self.max_gap_entry.pack(fill="x", padx=10, pady=5)

        tk.Label(
            frame,
            text="Note: Gaps larger than this value will be left as NaN. Set to 0 to fill all gaps.",
            foreground="blue",
            justify="left",
            wraplength=400,
            font=("Arial", 10),
        ).pack(anchor="w", padx=10, pady=5)

    def create_buttons(self, parent):
        """Create OK and Cancel buttons"""
        # Analysis section
        analysis_frame = tk.LabelFrame(
            parent,
            text="Quality Analysis",
            padx=15,
            pady=12,
            font=("Arial", 12, "bold"),
        )
        analysis_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=15, pady=(20, 10)
        )
        analysis_btns_frame = tk.Frame(analysis_frame)
        analysis_btns_frame.pack(pady=10)

        # Add test data button
        tk.Button(
            analysis_btns_frame,
            text="Load Test Data",
            command=self.load_test_data,
            width=18,
            height=2,
            bg="#2196F3",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(side="left", padx=10)

        # Add analyze quality button
        tk.Button(
            analysis_btns_frame,
            text="Analyze Quality",
            command=self.analyze_quality,
            width=18,
            height=2,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11, "bold"),
        ).pack(side="left", padx=10)

        self.test_data_label = tk.Label(
            analysis_frame, text="No test data loaded", fg="gray", font=("Arial", 11)
        )
        self.test_data_label.pack(pady=10)

        # OK and Cancel buttons
        button_frame = tk.Frame(parent)
        button_frame.grid(row=2, column=0, columnspan=2, pady=20)

        self.ok_button = tk.Button(
            button_frame,
            text="OK",
            command=self.ok,
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.ok_button.pack(side="left", padx=15)

        self.cancel_button = tk.Button(
            button_frame,
            text="Cancel",
            command=self.cancel,
            width=15,
            height=2,
            bg="#f44336",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.cancel_button.pack(side="right", padx=15)

    def update_params_frame(self):
        """Update parameters frame based on selected method"""
        try:
            # Clear existing widgets
            for widget in self.params_widgets:
                widget.destroy()
            self.params_widgets.clear()
            self.param_entries.clear()

            smooth_method = int(self.smooth_entry.get())

            if smooth_method == 2:  # Savitzky-Golay
                self.create_savgol_params()
            elif smooth_method == 3:  # LOWESS
                self.create_lowess_params()
            elif smooth_method == 4:  # Kalman
                self.create_kalman_params()
            elif smooth_method == 5:  # Butterworth
                self.create_butterworth_params()
            elif smooth_method == 6:  # Splines
                self.create_splines_params()
            elif smooth_method == 7:  # ARIMA
                self.create_arima_params()
            else:
                label = tk.Label(
                    self.params_frame,
                    text="No additional parameters needed",
                    font=("Arial", 11),
                )
                label.pack(anchor="w", padx=5, pady=5)
                self.params_widgets.append(label)

        except ValueError:
            label = tk.Label(
                self.params_frame, text="Please enter a valid method number (1-7)"
            )
            label.pack(anchor="w", padx=5, pady=5)
            self.params_widgets.append(label)

    def create_savgol_params(self):
        """Create Savitzky-Golay parameters"""
        # Window length
        label1 = tk.Label(
            self.params_frame, text="Window length (must be odd):", font=("Arial", 11)
        )
        label1.pack(anchor="w", padx=5, pady=2)
        entry1 = tk.Entry(
            self.params_frame, textvariable=self.savgol_window, font=("Arial", 11)
        )
        entry1.pack(fill="x", padx=5, pady=2)

        # Tooltip for window length
        tooltip1 = tk.Label(
            self.params_frame,
            text="💡 Tip: Use 5-15 for smooth data, 15-31 for noisy data. Must be odd number.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip1.pack(anchor="w", padx=5, pady=(0, 5))

        # Polynomial order
        label2 = tk.Label(
            self.params_frame, text="Polynomial order:", font=("Arial", 11)
        )
        label2.pack(anchor="w", padx=5, pady=2)
        entry2 = tk.Entry(
            self.params_frame, textvariable=self.savgol_poly, font=("Arial", 11)
        )
        entry2.pack(fill="x", padx=5, pady=2)

        # Tooltip for polynomial order
        tooltip2 = tk.Label(
            self.params_frame,
            text="💡 Tip: Use 2-3 for most cases. Must be < window length. Higher = more flexible.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip2.pack(anchor="w", padx=5, pady=(0, 5))

        self.params_widgets.extend([label1, entry1, tooltip1, label2, entry2, tooltip2])
        self.param_entries["window_length"] = entry1
        self.param_entries["polyorder"] = entry2

    def create_lowess_params(self):
        """Create LOWESS parameters"""
        # Fraction
        label1 = tk.Label(self.params_frame, text="Fraction (0-1):", font=("Arial", 11))
        label1.pack(anchor="w", padx=5, pady=2)
        entry1 = tk.Entry(
            self.params_frame, textvariable=self.lowess_frac, font=("Arial", 11)
        )
        entry1.pack(fill="x", padx=5, pady=2)

        # Tooltip for fraction
        tooltip1 = tk.Label(
            self.params_frame,
            text="💡 Tip: 0.1-0.3 for smooth data, 0.3-0.5 for noisy data. Higher = smoother.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip1.pack(anchor="w", padx=5, pady=(0, 5))

        # Iterations
        label2 = tk.Label(
            self.params_frame, text="Number of iterations:", font=("Arial", 11)
        )
        label2.pack(anchor="w", padx=5, pady=2)
        entry2 = tk.Entry(
            self.params_frame, textvariable=self.lowess_it, font=("Arial", 11)
        )
        entry2.pack(fill="x", padx=5, pady=2)

        # Tooltip for iterations
        tooltip2 = tk.Label(
            self.params_frame,
            text="💡 Tip: 2-4 iterations usually sufficient. More iterations = more robust to outliers.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip2.pack(anchor="w", padx=5, pady=(0, 5))

        self.params_widgets.extend([label1, entry1, tooltip1, label2, entry2, tooltip2])
        self.param_entries["frac"] = entry1
        self.param_entries["it"] = entry2

    def create_butterworth_params(self):
        """Create Butterworth parameters"""
        # Cutoff frequency
        label1 = tk.Label(
            self.params_frame, text="Cutoff frequency (Hz):", font=("Arial", 11)
        )
        label1.pack(anchor="w", padx=5, pady=2)
        entry1 = tk.Entry(
            self.params_frame, textvariable=self.butter_cutoff, font=("Arial", 11)
        )
        entry1.pack(fill="x", padx=5, pady=2)

        # Tooltip for cutoff frequency
        tooltip1 = tk.Label(
            self.params_frame,
            text="💡 Tip: 4-10 Hz for biomechanics, 1-5 Hz for slow movements. Must be < fs/2.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip1.pack(anchor="w", padx=5, pady=(0, 5))

        # Sampling frequency
        label2 = tk.Label(
            self.params_frame, text="Sampling frequency (Hz):", font=("Arial", 11)
        )
        label2.pack(anchor="w", padx=5, pady=2)
        entry2 = tk.Entry(
            self.params_frame, textvariable=self.butter_fs, font=("Arial", 11)
        )
        entry2.pack(fill="x", padx=5, pady=2)

        # Tooltip for sampling frequency
        tooltip2 = tk.Label(
            self.params_frame,
            text="💡 Tip: 30 Hz for video, 100-1000 Hz for motion capture. Must be > 2×cutoff.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip2.pack(anchor="w", padx=5, pady=(0, 5))

        self.params_widgets.extend([label1, entry1, tooltip1, label2, entry2, tooltip2])
        self.param_entries["cutoff"] = entry1
        self.param_entries["fs"] = entry2

    def create_kalman_params(self):
        """Create Kalman parameters"""
        # EM iterations
        label1 = tk.Label(
            self.params_frame, text="Number of EM iterations:", font=("Arial", 11)
        )
        label1.pack(anchor="w", padx=5, pady=2)
        entry1 = tk.Entry(
            self.params_frame, textvariable=self.kalman_iterations, font=("Arial", 11)
        )
        entry1.pack(fill="x", padx=5, pady=2)

        # Tooltip for EM iterations
        tooltip1 = tk.Label(
            self.params_frame,
            text="💡 Tip: 3-10 iterations. More iterations = better parameter estimation but slower.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip1.pack(anchor="w", padx=5, pady=(0, 5))

        # Processing mode
        label2 = tk.Label(
            self.params_frame, text="Processing Mode (1=1D, 2=2D):", font=("Arial", 11)
        )
        label2.pack(anchor="w", padx=5, pady=2)
        entry2 = tk.Entry(
            self.params_frame, textvariable=self.kalman_mode, font=("Arial", 11)
        )
        entry2.pack(fill="x", padx=5, pady=2)

        # Tooltip for processing mode
        tooltip2 = tk.Label(
            self.params_frame,
            text="💡 Tip: 1=process each column independently, 2=process x,y pairs together.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip2.pack(anchor="w", padx=5, pady=(0, 5))

        self.params_widgets.extend([label1, entry1, tooltip1, label2, entry2, tooltip2])
        self.param_entries["n_iter"] = entry1
        self.param_entries["mode"] = entry2

    def create_splines_params(self):
        """Create Splines parameters"""
        # Smoothing factor
        label = tk.Label(
            self.params_frame, text="Smoothing factor (s):", font=("Arial", 11)
        )
        label.pack(anchor="w", padx=5, pady=2)
        entry = tk.Entry(
            self.params_frame, textvariable=self.spline_smoothing, font=("Arial", 11)
        )
        entry.pack(fill="x", padx=5, pady=2)

        # Tooltip for smoothing factor
        tooltip = tk.Label(
            self.params_frame,
            text="💡 Tip: 0.1-1.0 for light smoothing, 1.0-10 for moderate, 10+ for strong smoothing.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip.pack(anchor="w", padx=5, pady=(0, 5))

        self.params_widgets.extend([label, entry, tooltip])
        self.param_entries["smoothing_factor"] = entry

    def create_arima_params(self):
        """Create ARIMA parameters"""
        # AR order (p)
        label1 = tk.Label(self.params_frame, text="AR order (p):", font=("Arial", 11))
        label1.pack(anchor="w", padx=5, pady=2)
        entry1 = tk.Entry(
            self.params_frame, textvariable=self.arima_p, font=("Arial", 11)
        )
        entry1.pack(fill="x", padx=5, pady=2)

        # Tooltip for AR order
        tooltip1 = tk.Label(
            self.params_frame,
            text="💡 Tip: 1-3 for most cases. Higher values for complex patterns.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip1.pack(anchor="w", padx=5, pady=(0, 5))

        # Difference order (d)
        label2 = tk.Label(
            self.params_frame, text="Difference order (d):", font=("Arial", 11)
        )
        label2.pack(anchor="w", padx=5, pady=2)
        entry2 = tk.Entry(
            self.params_frame, textvariable=self.arima_d, font=("Arial", 11)
        )
        entry2.pack(fill="x", padx=5, pady=2)

        # Tooltip for difference order
        tooltip2 = tk.Label(
            self.params_frame,
            text="💡 Tip: 0 for stationary data, 1-2 for trending data.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip2.pack(anchor="w", padx=5, pady=(0, 5))

        # MA order (q)
        label3 = tk.Label(self.params_frame, text="MA order (q):", font=("Arial", 11))
        label3.pack(anchor="w", padx=5, pady=2)
        entry3 = tk.Entry(
            self.params_frame, textvariable=self.arima_q, font=("Arial", 11)
        )
        entry3.pack(fill="x", padx=5, pady=2)

        # Tooltip for MA order
        tooltip3 = tk.Label(
            self.params_frame,
            text="💡 Tip: 0-2 for most cases. Higher values for complex noise patterns.",
            font=("Arial", 9),
            fg="#666666",
            wraplength=300,
        )
        tooltip3.pack(anchor="w", padx=5, pady=(0, 5))

        self.params_widgets.extend(
            [
                label1,
                entry1,
                tooltip1,
                label2,
                entry2,
                tooltip2,
                label3,
                entry3,
                tooltip3,
            ]
        )
        self.param_entries.update({"p": entry1, "d": entry2, "q": entry3})

    def update_parameter_value(self, event, stringvar):
        """Update the value of the StringVar when the user presses Enter"""
        widget = event.widget
        value = widget.get()
        stringvar.set(value)
        # Move the focus to the next widget
        widget.tk_focusNext().focus()

    def validate(self):
        try:
            interp_num = int(self.interp_entry.get())
            smooth_num = int(self.smooth_entry.get())

            if not (1 <= interp_num <= 7):
                messagebox.showerror(
                    "Error", "Gap filling method must be between 1 and 7"
                )
                return False

            if not (1 <= smooth_num <= 7):
                messagebox.showerror(
                    "Error", "Smoothing method must be between 1 and 7"
                )
                return False

            # Validate parameters specifically
            if smooth_num == 2:  # Savitzky-Golay
                if not self.savgol_window.get() or not self.savgol_poly.get():
                    messagebox.showerror(
                        "Error", "Savitzky-Golay parameters are required"
                    )
                    return False

                window = int(self.savgol_window.get())
                poly = int(self.savgol_poly.get())
                if window % 2 == 0:
                    messagebox.showerror("Error", "Window length must be an odd number")
                    return False
                if poly >= window:
                    messagebox.showerror(
                        "Error", "Polynomial order must be less than window length"
                    )
                    return False

            elif smooth_num == 3:  # LOWESS
                if not self.lowess_frac.get() or not self.lowess_it.get():
                    messagebox.showerror("Error", "LOWESS parameters are required")
                    return False

                frac = float(self.lowess_frac.get())
                if not (0 < frac <= 1):
                    messagebox.showerror("Error", "Fraction must be between 0 and 1")
                    return False

            elif smooth_num == 4:  # Kalman
                if not self.kalman_iterations.get():
                    messagebox.showerror(
                        "Error", "Kalman filter iterations are required"
                    )
                    return False

                n_iter = int(self.kalman_iterations.get())
                if n_iter <= 0:
                    messagebox.showerror(
                        "Error", "Number of iterations must be positive"
                    )
                    return False

            elif smooth_num == 5:  # Butterworth
                if not self.butter_cutoff.get() or not self.butter_fs.get():
                    messagebox.showerror(
                        "Error",
                        "Butterworth filter requires both cutoff and sampling frequencies",
                    )
                    return False

                cutoff = float(self.butter_cutoff.get())
                fs = float(self.butter_fs.get())
                if cutoff <= 0 or fs <= 0:
                    messagebox.showerror("Error", "Frequencies must be positive")
                    return False
                if cutoff >= fs / 2:
                    messagebox.showerror(
                        "Error",
                        "Cutoff frequency must be less than half of sampling frequency (Nyquist frequency)",
                    )
                    return False

            # Validate general parameters
            if not self.padding_entry.get():
                messagebox.showerror("Error", "Padding percentage is required")
                return False

            padding = float(self.padding_entry.get())

            if not self.max_gap_entry.get():
                messagebox.showerror("Error", "Maximum gap size is required")
                return False

            max_gap = int(self.max_gap_entry.get())

            if not (0 <= padding <= 100):
                messagebox.showerror("Error", "Padding must be between 0 and 100%")
                return False

            if max_gap < 0:
                messagebox.showerror("Error", "Maximum gap size must be non-negative")
                return False

            if smooth_num == 6:  # Splines
                if not self.spline_smoothing.get():
                    messagebox.showerror("Error", "Spline smoothing factor is required")
                    return False

                s = float(self.spline_smoothing.get())
                if s < 0:
                    messagebox.showerror(
                        "Error", "Smoothing factor must be non-negative"
                    )
                return False

            return True

        except ValueError as e:
            print(f"Error: Please enter valid numeric values: {str(e)}")
            return False

    def confirm_parameters(self):
        """Confirm and update parameters before processing"""
        try:
            # Force focus loss of all input widgets
            self.focus()

            # Force explicit update of Entry widget values
            if "cutoff" in self.param_entries:
                self.butter_cutoff.set(self.param_entries["cutoff"].get())
            if "fs" in self.param_entries:
                self.butter_fs.set(self.param_entries["fs"].get())
            if "window_length" in self.param_entries:
                self.savgol_window.set(self.param_entries["window_length"].get())
            if "polyorder" in self.param_entries:
                self.savgol_poly.set(self.param_entries["polyorder"].get())
            if "frac" in self.param_entries:
                self.lowess_frac.set(self.param_entries["frac"].get())
            if "it" in self.param_entries:
                self.lowess_it.set(self.param_entries["it"].get())
            if "smoothing_factor" in self.param_entries:
                self.spline_smoothing.set(self.param_entries["smoothing_factor"].get())
            if "n_iter" in self.param_entries:
                self.kalman_iterations.set(self.param_entries["n_iter"].get())
            if "mode" in self.param_entries:
                self.kalman_mode.set(self.param_entries["mode"].get())

            # Force update of widgets
            self.update_idletasks()

            # Capture the current smoothing method
            smooth_method = int(self.smooth_entry.get())

            # Print confirmed parameters in terminal
            print("\n" + "=" * 50)
            print("CONFIRMED PARAMETERS:")
            print("=" * 50)
            print(f"Gap Filling Method: {self.interp_entry.get()}")
            print(f"Smoothing Method: {self.smooth_entry.get()}")
            print(f"Max Gap Size: {self.max_gap_entry.get()} frames")
            print(f"Padding: {self.padding_entry.get()}%")
            print(f"Split Data: {'Yes' if self.split_var.get() else 'No'}")

            # Define params_text and print specific method parameters
            if smooth_method == 1:  # None
                params_text = "No smoothing parameters needed"
                print("\nNo smoothing parameters needed")
            elif smooth_method == 2:  # Savitzky-Golay
                window = int(self.savgol_window.get())
                poly = int(self.savgol_poly.get())
                params_text = f"Window Length: {window}, Polynomial Order: {poly}"
                print("\nSavitzky-Golay Parameters:")
                print(f"- Window Length: {window}")
                print(f"- Polynomial Order: {poly}")
            elif smooth_method == 3:  # LOWESS
                frac = float(self.lowess_frac.get())
                it = int(self.lowess_it.get())
                params_text = f"Fraction: {frac}, Iterations: {it}"
                print("\nLOWESS Parameters:")
                print(f"- Fraction: {frac}")
                print(f"- Iterations: {it}")
            elif smooth_method == 4:  # Kalman
                n_iter = int(self.kalman_iterations.get())
                mode = int(self.kalman_mode.get())
                if mode not in [1, 2]:
                    messagebox.showerror(
                        "Error", "Kalman mode must be 1 (1D) or 2 (2D)"
                    )
                    return False
                params_text = f"EM Iterations: {n_iter}, Processing Mode: {mode}"
                print(f"APPLY: Kalman settings - n_iter={n_iter}, mode={mode}")
            elif smooth_method == 5:  # Butterworth
                cutoff = float(self.butter_cutoff.get())
                fs = float(self.butter_fs.get())
                params_text = f"Cutoff: {cutoff} Hz, Sampling Frequency: {fs} Hz"
                print("\nButterworth Filter Parameters:")
                print(f"- Cutoff Frequency: {cutoff} Hz")
                print(f"- Sampling Frequency: {fs} Hz")

                # Additional validation for Butterworth
                if cutoff >= fs / 2:
                    raise ValueError(
                        "Cutoff frequency must be less than half of sampling frequency (Nyquist frequency)"
                    )
            elif smooth_method == 6:  # Splines
                s = float(self.spline_smoothing.get())
                params_text = f"Smoothing Factor: {s}"
                print("\nSpline Smoothing Parameters:")
                print(f"- Smoothing Factor: {s}")
            elif smooth_method == 7:  # ARIMA
                p = int(self.arima_p.get())
                d = int(self.arima_d.get())
                q = int(self.arima_q.get())
                params_text = f"ARIMA Order: p={p}, d={d}, q={q}"
                print("\nARIMA Parameters:")
                print(f"- AR order (p): {p}")
                print(f"- Difference order (d): {d}")
                print(f"- MA order (q): {q}")

            print("=" * 50)

            # Show confirmation message with current values
            confirmation = f"""Current Parameters:
            
Interpolation Method: {self.interp_entry.get()}
Smoothing Method: {self.smooth_entry.get()}
Max Gap Size: {self.max_gap_entry.get()}
Padding: {self.padding_entry.get()}%

Method Specific Parameters:
{params_text}

Split Data: {'Yes' if self.split_var.get() else 'No'}

Parameters have been confirmed and will be used for processing.
"""
            messagebox.showinfo("Parameters Confirmed", confirmation)

            # Change the color of the button to indicate that the parameters have been confirmed
            self.confirm_button.configure(
                bg="pale green", text="Parameters Confirmed ✓"
            )

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")

    def apply(self):
        if self.use_toml and self.loaded_toml:
            # Build the result from the loaded TOML
            interp = self.loaded_toml.get("interpolation", {})
            smoothing = self.loaded_toml.get("smoothing", {})
            padding = self.loaded_toml.get("padding", {})
            split = self.loaded_toml.get("split", {})
            smooth_params = {k: v for k, v in smoothing.items() if k != "method"}
            self.result = {
                "padding": float(padding.get("percent", 10)),
                "interp_method": interp.get("method", "linear"),
                "smooth_method": smoothing.get("method", "none"),
                "smooth_params": smooth_params,
                "max_gap": int(interp.get("max_gap", 60)),
                "do_split": bool(split.get("enabled", False)),
            }
        else:
            try:
                # Check if the parameters have been confirmed
                if self.confirm_button["text"] != "Parameters Confirmed ✓":
                    if not messagebox.askyesno(
                        "Warning",
                        "Parameters have not been confirmed. Do you want to proceed anyway?",
                    ):
                        return

                # Force update of widget values before collecting them
                self.update_idletasks()

                # Debug: Print the values of the StringVar before processing
                print("\nDEBUG - StringVar Values:")
                print(f"Savgol Window: {self.savgol_window.get()}")
                print(f"Savgol Poly: {self.savgol_poly.get()}")
                print(f"LOWESS Frac: {self.lowess_frac.get()}")
                print(f"LOWESS It: {self.lowess_it.get()}")
                print(f"Butter Cutoff: {self.butter_cutoff.get()}")
                print(f"Butter Fs: {self.butter_fs.get()}")
                print(f"Kalman Iterations: {self.kalman_iterations.get()}")

                interp_map = {
                    1: "linear",
                    2: "cubic",
                    3: "nearest",
                    4: "kalman",
                    5: "none",
                    6: "skip",
                }

                smooth_map = {
                    1: "none",
                    2: "savgol",
                    3: "lowess",
                    4: "kalman",
                    5: "butterworth",
                    6: "splines",
                    7: "arima",
                }

                # Debug: Print the values of the Entry widgets
                print("\nDEBUG - Entry Values:")
                print(f"Interpolation Method: {self.interp_entry.get()}")
                print(f"Smoothing Method: {self.smooth_entry.get()}")
                print(f"Max Gap: {self.max_gap_entry.get()}")
                print(f"Padding: {self.padding_entry.get()}")

                smooth_method = int(self.smooth_entry.get())
                interp_method = int(self.interp_entry.get())
                max_gap = int(self.max_gap_entry.get())
                padding = float(self.padding_entry.get())
                do_split = self.split_var.get()

                # Prepare the smoothing parameters based on the chosen method
                smooth_params = {}
                if smooth_method == 2:  # Savitzky-Golay
                    window_length = int(self.savgol_window.get())
                    polyorder = int(self.savgol_poly.get())
                    smooth_params = {
                        "window_length": window_length,
                        "polyorder": polyorder,
                    }
                    print(
                        f"APPLY: Savitzky-Golay settings - window={window_length}, polyorder={polyorder}"
                    )

                elif smooth_method == 3:  # LOWESS
                    frac = float(self.lowess_frac.get())
                    it = int(self.lowess_it.get())
                    smooth_params = {"frac": frac, "it": it}
                    print(f"APPLY: LOWESS settings - frac={frac}, it={it}")

                elif smooth_method == 4:  # Kalman
                    n_iter = int(self.kalman_iterations.get())
                    mode = int(self.kalman_mode.get())
                    if mode not in [1, 2]:
                        messagebox.showerror(
                            "Error", "Kalman mode must be 1 (1D) or 2 (2D)"
                        )
                        return False
                    smooth_params = {"n_iter": n_iter, "mode": mode}
                    print(f"APPLY: Kalman settings - n_iter={n_iter}, mode={mode}")

                elif smooth_method == 5:  # Butterworth
                    cutoff = float(self.butter_cutoff.get())
                    fs = float(self.butter_fs.get())
                    smooth_params = {"cutoff": cutoff, "fs": fs}
                    print(
                        f"APPLY: Butterworth settings - cutoff={cutoff} Hz, fs={fs} Hz"
                    )

                elif smooth_method == 6:  # Splines
                    smoothing_factor = float(self.spline_smoothing.get())
                    smooth_params = {"smoothing_factor": smoothing_factor}
                    print(
                        f"APPLY: Spline Smoothing settings - smoothing_factor={smoothing_factor}"
                    )

                elif smooth_method == 7:  # ARIMA
                    p = int(self.arima_p.get())
                    d = int(self.arima_d.get())
                    q = int(self.arima_q.get())
                    smooth_params = {"p": p, "d": d, "q": q}
                    print(f"APPLY: ARIMA settings - order=({p},{d},{q})")

                # Display a summary of the chosen parameters
                summary = f"""
=== CONFIGURATION SUMMARY ===
- Gap Filling Method: {interp_map[interp_method]}
- Max Gap Size: {max_gap} frames
- Smoothing Method: {smooth_map[smooth_method]}
- Padding: {padding}%
- Split Data: {'Yes' if do_split else 'No'}
"""

                if smooth_method == 2:  # Savitzky-Golay
                    window = int(self.savgol_window.get())
                    poly = int(self.savgol_poly.get())
                    summary += f"\nSavitzky-Golay Parameters:\n- Window Length: {window}\n- Polynomial Order: {poly}"
                elif smooth_method == 3:  # LOWESS
                    frac = float(self.lowess_frac.get())
                    it = int(self.lowess_it.get())
                    summary += (
                        f"\nLOWESS Parameters:\n- Fraction: {frac}\n- Iterations: {it}"
                    )
                elif smooth_method == 4:  # Kalman
                    n_iter = int(self.kalman_iterations.get())
                    mode = int(self.kalman_mode.get())
                    summary += f"\nKalman Parameters:\n- EM Iterations: {n_iter}\n- Processing Mode: {mode}"
                elif smooth_method == 5:  # Butterworth
                    cutoff = float(self.butter_cutoff.get())
                    fs = float(self.butter_fs.get())
                    summary += f"\nButterworth Parameters:\n- Cutoff Frequency: {cutoff} Hz\n- Sampling Frequency: {fs} Hz"
                elif smooth_method == 6:  # Splines
                    s = float(self.spline_smoothing.get())
                    summary += (
                        f"\nSpline Smoothing Parameters:\n- Smoothing Factor: {s}"
                    )
                elif smooth_method == 7:  # ARIMA
                    order = (
                        int(smooth_params["p"]),
                        int(smooth_params["d"]),
                        int(smooth_params["q"]),
                    )
                    summary += f"\nARIMA Parameters:\n- Order: {order}"

                if messagebox.askokcancel("Confirm Parameters", summary):
                    config_result = {
                        "padding": padding,
                        "interp_method": interp_map[interp_method],
                        "smooth_method": smooth_map[smooth_method],
                        "smooth_params": smooth_params,
                        "max_gap": max_gap,
                        "do_split": do_split,
                    }

                    print("\nDEBUG - Final Configuration:")
                    print(f"FINAL CONFIG: {config_result}")
                    self.result = config_result
                else:
                    self.result = None

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
                self.result = None

    def update_value(self, event):
        self.update_idletasks()

    def bind_entries(self):
        self.interp_entry.bind("<FocusOut>", self.update_value)
        self.smooth_entry.bind("<FocusOut>", self.update_value)
        self.padding_entry.bind("<FocusOut>", self.update_value)
        self.max_gap_entry.bind("<FocusOut>", self.update_value)

    def create_toml_template(self):
        from tkinter import filedialog, messagebox

        file_path = filedialog.asksaveasfilename(
            title="Create TOML template",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialfile="interp_smooth_config_template.toml",
        )
        if file_path:
            config = self.get_current_config()
            save_config_to_toml(config, file_path)
            messagebox.showinfo(
                "Template created", f"Template TOML created in:\n{file_path}"
            )

    def load_toml_config(self):
        from tkinter import filedialog, messagebox

        file_path = filedialog.askopenfilename(
            title="Load TOML configuration",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if file_path:
            config = load_config_from_toml(file_path)
            self.loaded_toml = config
            self.use_toml = True
            self.toml_label.config(
                text=f"TOML loaded: {os.path.basename(file_path)}", fg="green"
            )
            self.apply_toml_to_gui(config)
            summary = f"TOML loaded: {os.path.basename(file_path)}\n"
            summary += f"[interpolation] method: {config.get('interpolation', {}).get('method')}\n"
            summary += f"max_gap: {config.get('interpolation', {}).get('max_gap')}\n"
            summary += (
                f"[smoothing] method: {config.get('smoothing', {}).get('method')}\n"
            )
            summary += (
                f"[padding] percent: {config.get('padding', {}).get('percent')}\n"
            )
            summary += f"[split] enabled: {config.get('split', {}).get('enabled')}\n"
            print("\n=== TOML configuration loaded and will be used ===\n" + summary)
            messagebox.showinfo("TOML Parameters Loaded", summary)

    def get_current_config(self):
        # Collect the current values from the interface and build the dict for TOML
        interp_map = {
            1: "linear",
            2: "cubic",
            3: "nearest",
            4: "kalman",
            5: "none",
            6: "skip",
        }
        smooth_map = {
            1: "none",
            2: "savgol",
            3: "lowess",
            4: "kalman",
            5: "butterworth",
            6: "splines",
            7: "arima",
        }
        interp_method = interp_map.get(int(self.interp_entry.get()), "linear")
        smooth_method = smooth_map.get(int(self.smooth_entry.get()), "none")
        max_gap = int(self.max_gap_entry.get())
        padding = float(self.padding_entry.get())
        do_split = self.split_var.get()
        # Specific parameters
        smoothing_params = {}
        if smooth_method == "savgol":
            smoothing_params = {
                "window_length": int(self.savgol_window.get()),
                "polyorder": int(self.savgol_poly.get()),
            }
        elif smooth_method == "lowess":
            smoothing_params = {
                "frac": float(self.lowess_frac.get()),
                "it": int(self.lowess_it.get()),
            }
        elif smooth_method == "kalman":
            smoothing_params = {
                "n_iter": int(self.kalman_iterations.get()),
                "mode": int(self.kalman_mode.get()),
            }
        elif smooth_method == "butterworth":
            smoothing_params = {
                "cutoff": float(self.butter_cutoff.get()),
                "fs": float(self.butter_fs.get()),
            }
        elif smooth_method == "splines":
            smoothing_params = {"smoothing_factor": float(self.spline_smoothing.get())}
        elif smooth_method == "arima":
            smoothing_params = {
                "p": int(self.arima_p.get()),
                "d": int(self.arima_d.get()),
                "q": int(self.arima_q.get()),
            }
        config = {
            "interpolation": {"method": interp_method, "max_gap": max_gap},
            "smoothing": {"method": smooth_method, **smoothing_params},
            "padding": {"percent": padding},
            "split": {"enabled": do_split},
        }
        return config

    def apply_toml_to_gui(self, config):
        # Fill the interface fields with the values from the TOML
        interp_map_rev = {
            "linear": 1,
            "cubic": 2,
            "nearest": 3,
            "kalman": 4,
            "none": 5,
            "skip": 6,
        }
        smooth_map_rev = {
            "none": 1,
            "savgol": 2,
            "lowess": 3,
            "kalman": 4,
            "butterworth": 5,
            "splines": 6,
            "arima": 7,
        }
        interp = config.get("interpolation", {})
        smoothing = config.get("smoothing", {})
        padding = config.get("padding", {})
        split = config.get("split", {})
        self.interp_entry.delete(0, tk.END)
        self.interp_entry.insert(
            0, str(interp_map_rev.get(interp.get("method", "linear"), 1))
        )
        self.smooth_entry.delete(0, tk.END)
        self.smooth_entry.insert(
            0, str(smooth_map_rev.get(smoothing.get("method", "none"), 1))
        )
        self.max_gap_entry.delete(0, tk.END)
        self.max_gap_entry.insert(0, str(interp.get("max_gap", 60)))
        self.padding_entry.delete(0, tk.END)
        self.padding_entry.insert(0, str(padding.get("percent", 10)))
        self.split_var.set(bool(split.get("enabled", False)))
        # Specific smoothing parameters
        if smoothing.get("method") == "savgol":
            self.savgol_window.set(str(smoothing.get("window_length", 7)))
            self.savgol_poly.set(str(smoothing.get("polyorder", 3)))
        elif smoothing.get("method") == "lowess":
            self.lowess_frac.set(str(smoothing.get("frac", 0.3)))
            self.lowess_it.set(str(smoothing.get("it", 3)))
        elif smoothing.get("method") == "kalman":
            self.kalman_iterations.set(str(smoothing.get("n_iter", 5)))
            self.kalman_mode.set(str(smoothing.get("mode", 1)))
        elif smoothing.get("method") == "butterworth":
            self.butter_cutoff.set(str(smoothing.get("cutoff", 10)))
            self.butter_fs.set(str(smoothing.get("fs", 100)))
        elif smoothing.get("method") == "splines":
            self.spline_smoothing.set(str(smoothing.get("smoothing_factor", 1.0)))
        elif smoothing.get("method") == "arima":
            self.arima_p.set(str(smoothing.get("p", 1)))
            self.arima_d.set(str(smoothing.get("d", 0)))
            self.arima_q.set(str(smoothing.get("q", 0)))

    def buttonbox(self):
        """Override to avoid default OK/Cancel outside the scroll area."""
        # Do not create any external buttons - only use the ones inside scroll
        pass

    def validate(self):
        """Validate input parameters"""
        try:
            interp_num = int(self.interp_entry.get())
            smooth_num = int(self.smooth_entry.get())

            if not (1 <= interp_num <= 6):
                messagebox.showerror(
                    "Error", "Gap filling method must be between 1 and 6"
                )
                return False

            if not (1 <= smooth_num <= 7):
                messagebox.showerror(
                    "Error", "Smoothing method must be between 1 and 7"
                )
                return False

            # Validate specific method parameters
            if smooth_num == 2:  # Savitzky-Golay
                window = int(self.savgol_window.get())
                poly = int(self.savgol_poly.get())
                if window % 2 == 0:
                    messagebox.showerror("Error", "Window length must be an odd number")
                    return False
                if poly >= window:
                    messagebox.showerror(
                        "Error", "Polynomial order must be less than window length"
                    )
                    return False

            elif smooth_num == 3:  # LOWESS
                frac = float(self.lowess_frac.get())
                if not (0 < frac <= 1):
                    messagebox.showerror("Error", "Fraction must be between 0 and 1")
                    return False

            elif smooth_num == 5:  # Butterworth
                cutoff = float(self.butter_cutoff.get())
                fs = float(self.butter_fs.get())
                if cutoff <= 0 or fs <= 0:
                    messagebox.showerror("Error", "Frequencies must be positive")
                    return False
                if cutoff >= fs / 2:
                    messagebox.showerror(
                        "Error",
                        "Cutoff frequency must be less than half of sampling frequency",
                    )
                    return False

            # Validate general parameters
            padding = float(self.padding_entry.get())
            max_gap = int(self.max_gap_entry.get())

            if not (0 <= padding <= 100):
                messagebox.showerror("Error", "Padding must be between 0 and 100%")
                return False

            if max_gap < 0:
                messagebox.showerror("Error", "Maximum gap size must be non-negative")
                return False

            return True

        except ValueError as e:
            messagebox.showerror(
                "Error", f"Please enter valid numeric values: {str(e)}"
            )
            return False

    def get_config(self):
        """Get configuration from GUI"""
        interp_map = {
            1: "linear",
            2: "cubic",
            3: "nearest",
            4: "kalman",
            5: "none",
            6: "skip",
        }
        smooth_map = {
            1: "none",
            2: "savgol",
            3: "lowess",
            4: "kalman",
            5: "butterworth",
            6: "splines",
            7: "arima",
        }

        interp_method = int(self.interp_entry.get())
        smooth_method = int(self.smooth_entry.get())

        # Get smoothing parameters
        smooth_params = {}
        if smooth_method == 2:  # Savitzky-Golay
            smooth_params = {
                "window_length": int(self.savgol_window.get()),
                "polyorder": int(self.savgol_poly.get()),
            }
        elif smooth_method == 3:  # LOWESS
            smooth_params = {
                "frac": float(self.lowess_frac.get()),
                "it": int(self.lowess_it.get()),
            }
        elif smooth_method == 4:  # Kalman
            smooth_params = {
                "n_iter": int(self.kalman_iterations.get()),
                "mode": int(self.kalman_mode.get()),
            }
        elif smooth_method == 5:  # Butterworth
            smooth_params = {
                "cutoff": float(self.butter_cutoff.get()),
                "fs": float(self.butter_fs.get()),
            }
        elif smooth_method == 6:  # Splines
            smooth_params = {"smoothing_factor": float(self.spline_smoothing.get())}
        elif smooth_method == 7:  # ARIMA
            smooth_params = {
                "p": int(self.arima_p.get()),
                "d": int(self.arima_d.get()),
                "q": int(self.arima_q.get()),
            }

        return {
            "padding": float(self.padding_entry.get()),
            "interp_method": interp_map[interp_method],
            "smooth_method": smooth_map[smooth_method],
            "smooth_params": smooth_params,
            "max_gap": int(self.max_gap_entry.get()),
            "do_split": self.split_var.get(),
        }

    def ok(self):
        """Handle OK button click"""
        if self.validate():
            self.result = self.get_config()
            self.window.destroy()

    def cancel(self):
        """Handle Cancel button click"""
        self.result = None
        self.window.destroy()

    def load_test_data(self):
        """Load a CSV file for testing the configuration"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file for testing",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )

        if file_path:
            try:
                self.test_data = pd.read_csv(file_path)
                self.test_data_path = file_path
                self.test_data_label.config(
                    text=f"Test data: {os.path.basename(file_path)} ({len(self.test_data)} rows, {len(self.test_data.columns)} columns)",
                    fg="green",
                )
                messagebox.showinfo(
                    "Success", f"Loaded test data: {os.path.basename(file_path)}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load test data: {str(e)}")
                self.test_data = None
                self.test_data_path = None
                self.test_data_label.config(text="Failed to load test data", fg="red")

    def analyze_quality(self):
        """Analyze the quality of smoothing/interpolation on test data"""
        if self.test_data is None:
            messagebox.showwarning("Warning", "Please load test data first!")
            return

        # Create a new window for analysis
        analysis_window = tk.Toplevel(self.window)
        analysis_window.title("Quality Analysis")
        analysis_window.geometry("1200x800")

        # Get all numeric columns for analysis
        numeric_cols = self.test_data.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        if len(numeric_cols) == 0:
            messagebox.showerror("Error", "No numeric columns found in test data!")
            analysis_window.destroy()
            return

        # Create column selection frame
        selection_frame = tk.Frame(analysis_window, padx=10, pady=10)
        selection_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(selection_frame, text="Select column to analyze:").pack(
            side=tk.LEFT, padx=5
        )

        # Column selection dropdown
        selected_column = tk.StringVar(value=numeric_cols[0])
        column_menu = tk.OptionMenu(selection_frame, selected_column, *numeric_cols)
        column_menu.pack(side=tk.LEFT, padx=5)

        # Analyze button
        tk.Button(
            selection_frame,
            text="Analyze Column",
            command=lambda: self.perform_analysis(
                analysis_window, selected_column.get()
            ),
            bg="#4CAF50",
            fg="white",
        ).pack(side=tk.LEFT, padx=5)

        # Close button
        tk.Button(
            selection_frame,
            text="Close",
            command=analysis_window.destroy,
            bg="#f44336",
            fg="white",
        ).pack(side=tk.RIGHT, padx=5)

        # Create frame for plots
        self.plot_frame = tk.Frame(analysis_window)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Perform initial analysis on first column
        self.perform_analysis(analysis_window, numeric_cols[0])

    def apply_filter_to_residuals(self, residuals, config):
        """Apply the same filter used in processing to the residuals"""
        try:
            if config["smooth_method"] == "none":
                return residuals  # No filtering applied

            # Apply padding if necessary
            padding_percent = config["padding"]
            pad_len = (
                int(len(residuals) * padding_percent / 100)
                if padding_percent > 0
                else 0
            )

            if pad_len > 0:
                # Pad with edge values
                padded_residuals = np.pad(residuals, pad_len, mode="edge")
            else:
                padded_residuals = residuals.copy()

            # Apply the same smoothing method to residuals
            if config["smooth_method"] == "savgol":
                params = config["smooth_params"]
                filtered_residuals = savgol_smooth(
                    padded_residuals, params["window_length"], params["polyorder"]
                )
            elif config["smooth_method"] == "lowess":
                params = config["smooth_params"]
                filtered_residuals = lowess_smooth(
                    padded_residuals, params["frac"], params["it"]
                )
            elif config["smooth_method"] == "kalman":
                params = config["smooth_params"]
                filtered_residuals = kalman_smooth(
                    padded_residuals, params["n_iter"], params["mode"]
                ).flatten()
            elif config["smooth_method"] == "butterworth":
                params = config["smooth_params"]
                if not np.isnan(padded_residuals).all():
                    filtered_residuals = butter_filter(
                        padded_residuals,
                        fs=params["fs"],
                        filter_type="low",
                        cutoff=params["cutoff"],
                        order=4,
                    )
                else:
                    filtered_residuals = padded_residuals
            elif config["smooth_method"] == "splines":
                params = config["smooth_params"]
                filtered_residuals = spline_smooth(
                    padded_residuals, s=params["smoothing_factor"]
                )
            elif config["smooth_method"] == "arima":
                params = config["smooth_params"]
                order = (params["p"], params["d"], params["q"])
                filtered_residuals = arima_smooth(padded_residuals, order=order)
            else:
                filtered_residuals = padded_residuals

            # Remove padding
            if pad_len > 0:
                filtered_residuals = filtered_residuals[pad_len:-pad_len]

            return filtered_residuals

        except Exception as e:
            print(f"Error applying filter to residuals: {str(e)}")
            return residuals  # Return original residuals if filtering fails

    def perform_analysis(self, window, column_name):
        """Perform quality analysis on selected column"""
        # Force update of all widgets before analysis
        print("\n[DEBUG] Performing analysis...")
        print(f"[DEBUG] Column: {column_name}")
        
        # Force focus to trigger any pending updates
        self.window.focus_force()
        self.window.update()
        self.window.update_idletasks()
        
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        # Get current configuration with forced parameter update
        config = self.get_current_analysis_config()
        
        print(f"[DEBUG] Using config for analysis:")
        print(f"  Interp method: {config['interp_method']}")
        print(f"  Smooth method: {config['smooth_method']}")
        print(f"  Smooth params: {config['smooth_params']}")
        print(f"  Padding: {config['padding']}")
        print(f"  Max gap: {config['max_gap']}")

        # Process the selected column
        original_data = self.test_data[column_name].values

        # Use first column as frame numbers if it's numeric, otherwise create index
        if (
            self.test_data.select_dtypes(include=[np.number]).columns[0]
            == self.test_data.columns[0]
        ):
            frame_numbers = self.test_data.iloc[:, 0].values
        else:
            frame_numbers = np.arange(len(original_data))

        # Apply current configuration to process data
        processed_data, padded_data = self.process_column_for_analysis(
            original_data, config
        )

        # Calculate derivatives
        first_derivative = np.gradient(processed_data)
        second_derivative = np.gradient(first_derivative)

        # Calculate residuals (only where original data is not NaN)
        valid_mask = ~np.isnan(original_data)
        residuals = np.full_like(original_data, np.nan)
        residuals[valid_mask] = original_data[valid_mask] - processed_data[valid_mask]

        # Apply the same filter to residuals to check for signal
        filtered_residuals = self.apply_filter_to_residuals(residuals, config)

        # Create figure with subplots
        fig = Figure(figsize=(12, 10))

        # Plot 1: Original vs Processed Data (usando pontos para melhor visualização)
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(
            frame_numbers,
            original_data,
            "o",
            label="Original",
            alpha=0.5,
            markersize=3,
            color="blue",
        )
        ax1.plot(
            frame_numbers, 
            processed_data, 
            ".", 
            label="Processed", 
            markersize=4,
            color="red",
            alpha=0.7
        )
        ax1.set_title(f"Original vs Processed - {column_name}", fontweight='bold')
        ax1.set_xlabel("Frame", fontweight='bold')
        ax1.set_ylabel("Value", fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Plot 2: Residuals (Original and Filtered) - usando pontos
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(
            frame_numbers[valid_mask],
            residuals[valid_mask],
            "o",
            markersize=3,
            label="Original Residuals",
            alpha=0.4,
            color="green",
        )
        ax2.plot(
            frame_numbers[valid_mask],
            filtered_residuals[valid_mask],
            ".",
            markersize=5,
            label="Filtered Residuals",
            alpha=0.7,
            color="red",
        )
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1.5)
        ax2.set_title("Residuals (Original - Processed)", fontweight='bold')
        ax2.set_xlabel("Frame", fontweight='bold')
        ax2.set_ylabel("Residual", fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Calculate and display RMS error
        rms_error = np.sqrt(np.nanmean(residuals**2))
        ax2.text(
            0.02,
            0.98,
            f"RMS Error: {rms_error:.4f}",
            transform=ax2.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Plot 3: First Derivative (Velocity) - mantém linha
        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(frame_numbers, first_derivative, "-", linewidth=1.5, color="magenta", alpha=0.7)
        ax3.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
        ax3.set_title("First Derivative (Velocity)", fontweight='bold')
        ax3.set_xlabel("Frame", fontweight='bold')
        ax3.set_ylabel("dY/dX", fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')

        # Plot 4: Second Derivative (Acceleration) - mantém linha
        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(frame_numbers, second_derivative, "-", linewidth=1.5, color="cyan", alpha=0.7)
        ax4.axhline(y=0, color="k", linestyle="-", alpha=0.3, linewidth=0.5)
        ax4.set_title("Second Derivative (Acceleration)", fontweight='bold')
        ax4.set_xlabel("Frame", fontweight='bold')
        ax4.set_ylabel("d²Y/dX²", fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')

        # Plot 5: Histogram of Residuals (melhorado)
        ax5 = fig.add_subplot(3, 2, 5)
        if np.any(valid_mask):
            ax5.hist(residuals[valid_mask], bins=30, edgecolor="black", alpha=0.7, color='steelblue')
            ax5.set_title("Distribution of Residuals", fontweight='bold')
            ax5.set_xlabel("Residual Value", fontweight='bold')
            ax5.set_ylabel("Frequency", fontweight='bold')
            ax5.grid(True, alpha=0.3, linestyle='--', axis="y")

            # Add normal distribution overlay
            from scipy import stats

            mu = np.nanmean(residuals)
            sigma = np.nanstd(residuals)
            x = np.linspace(np.nanmin(residuals), np.nanmax(residuals), 100)
            ax5.plot(
                x,
                stats.norm.pdf(x, mu, sigma)
                * len(residuals[valid_mask])
                * (x[1] - x[0])
                * 30,
                "r-",
                linewidth=2,
                label=f"Normal(μ={mu:.3f}, σ={sigma:.3f})",
            )
            ax5.legend()

        # Plot 6: Spectral Analysis (FFT of processed signal) - melhorado
        ax6 = fig.add_subplot(3, 2, 6)
        if len(processed_data) > 1:
            # Remove mean and apply window
            signal = processed_data - np.mean(processed_data)
            window = np.hanning(len(signal))
            signal_windowed = signal * window

            # Compute FFT
            fft = np.fft.rfft(signal_windowed)
            freq = np.fft.rfftfreq(len(signal), 1.0)  # Assuming 1 frame = 1 time unit

            ax6.semilogy(freq[1:], np.abs(fft[1:]), "-", linewidth=1.5, color='darkblue', alpha=0.7)
            ax6.set_title("Frequency Spectrum (FFT)", fontweight='bold')
            ax6.set_xlabel("Frequency (cycles/frame)", fontweight='bold')
            ax6.set_ylabel("Magnitude (log scale)", fontweight='bold')
            ax6.grid(True, alpha=0.3, linestyle='--', which='both')

        # Add configuration info as title
        config_text = f"Config: {config['interp_method']} interp, {config['smooth_method']} smooth"
        if config["smooth_method"] != "none":
            params_str = ", ".join(
                [f"{k}={v}" for k, v in config["smooth_params"].items()]
            )
            config_text += f" ({params_str})"
        config_text += " | Residuals filtered with same method"
        fig.suptitle(config_text, y=0.98, fontsize=12)

        # Adjust layout with more space for title
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Embed the figure in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar for navigation
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def get_current_analysis_config(self):
        """Get current configuration for analysis - with forced parameter update"""
        try:
            # Force update of all parameter values from Entry widgets
            self.window.update_idletasks()
            
            # Force explicit update from param_entries if they exist
            if hasattr(self, 'param_entries'):
                if "cutoff" in self.param_entries:
                    try:
                        cutoff_value = self.param_entries["cutoff"].get()
                        self.butter_cutoff.set(cutoff_value)
                        print(f"[DEBUG] Updated cutoff to: {cutoff_value}")
                    except Exception as e:
                        print(f"[DEBUG] Error updating cutoff: {e}")
                
                if "fs" in self.param_entries:
                    try:
                        fs_value = self.param_entries["fs"].get()
                        self.butter_fs.set(fs_value)
                        print(f"[DEBUG] Updated fs to: {fs_value}")
                    except Exception as e:
                        print(f"[DEBUG] Error updating fs: {e}")
            
            interp_map = {
                1: "linear",
                2: "cubic",
                3: "nearest",
                4: "kalman",
                5: "none",
                6: "skip",
            }
            smooth_map = {
                1: "none",
                2: "savgol",
                3: "lowess",
                4: "kalman",
                5: "butterworth",
                6: "splines",
                7: "arima",
            }

            smooth_method = int(self.smooth_entry.get())
            smooth_params = {}

            if smooth_method == 2:  # Savitzky-Golay
                smooth_params = {
                    "window_length": int(self.savgol_window.get()),
                    "polyorder": int(self.savgol_poly.get()),
                }
                print(f"[DEBUG] Savgol params: window={smooth_params['window_length']}, poly={smooth_params['polyorder']}")
            elif smooth_method == 3:  # LOWESS
                smooth_params = {
                    "frac": float(self.lowess_frac.get()),
                    "it": int(self.lowess_it.get()),
                }
                print(f"[DEBUG] LOWESS params: frac={smooth_params['frac']}, it={smooth_params['it']}")
            elif smooth_method == 4:  # Kalman
                smooth_params = {
                    "n_iter": int(self.kalman_iterations.get()),
                    "mode": int(self.kalman_mode.get()),
                }
                print(f"[DEBUG] Kalman params: n_iter={smooth_params['n_iter']}, mode={smooth_params['mode']}")
            elif smooth_method == 5:  # Butterworth
                smooth_params = {
                    "cutoff": float(self.butter_cutoff.get()),
                    "fs": float(self.butter_fs.get()),
                }
                print(f"[DEBUG] Butterworth params: cutoff={smooth_params['cutoff']}, fs={smooth_params['fs']}")
            elif smooth_method == 6:  # Splines
                smooth_params = {"smoothing_factor": float(self.spline_smoothing.get())}
                print(f"[DEBUG] Splines params: smoothing={smooth_params['smoothing_factor']}")
            elif smooth_method == 7:  # ARIMA
                smooth_params = {
                    "p": int(self.arima_p.get()),
                    "d": int(self.arima_d.get()),
                    "q": int(self.arima_q.get()),
                }
                print(f"[DEBUG] ARIMA params: p={smooth_params['p']}, d={smooth_params['d']}, q={smooth_params['q']}")

            config = {
                "interp_method": interp_map[int(self.interp_entry.get())],
                "smooth_method": smooth_map[smooth_method],
                "smooth_params": smooth_params,
                "padding": float(self.padding_entry.get()),
                "max_gap": int(self.max_gap_entry.get()),
            }
            
            print(f"[DEBUG] Final analysis config: {config}")
            return config
            
        except Exception as e:
            print(f"[ERROR] Error getting analysis config: {e}")
            import traceback
            traceback.print_exc()
            # Return default config if any error
            return {
                "interp_method": "linear",
                "smooth_method": "none",
                "smooth_params": {},
                "padding": 10,
                "max_gap": 60,
            }

    def process_column_for_analysis(self, data, config):
        """Process a single column with current configuration for analysis"""
        # Apply padding if necessary
        padding_percent = config["padding"]
        pad_len = int(len(data) * padding_percent / 100) if padding_percent > 0 else 0

        if pad_len > 0:
            # Pad with edge values
            padded_data = np.pad(data, pad_len, mode="edge")
        else:
            padded_data = data.copy()

        # Apply interpolation
        if config["interp_method"] not in ["none", "skip"]:
            # Create pandas series for interpolation
            series = pd.Series(padded_data)

            if config["interp_method"] == "linear":
                series = series.interpolate(method="linear", limit_direction="both")
            elif config["interp_method"] == "cubic":
                series = series.interpolate(method="cubic", limit_direction="both")
            elif config["interp_method"] == "nearest":
                series = series.interpolate(method="nearest", limit_direction="both")

            padded_data = series.values

        # Apply smoothing
        if config["smooth_method"] != "none":
            try:
                if config["smooth_method"] == "savgol":
                    params = config["smooth_params"]
                    padded_data = savgol_smooth(
                        padded_data, params["window_length"], params["polyorder"]
                    )
                elif config["smooth_method"] == "lowess":
                    params = config["smooth_params"]
                    padded_data = lowess_smooth(
                        padded_data, params["frac"], params["it"]
                    )
                elif config["smooth_method"] == "kalman":
                    params = config["smooth_params"]
                    padded_data = kalman_smooth(
                        padded_data, params["n_iter"], params["mode"]
                    ).flatten()
                elif config["smooth_method"] == "butterworth":
                    params = config["smooth_params"]
                    if not np.isnan(padded_data).all():
                        padded_data = butter_filter(
                            padded_data,
                            fs=params["fs"],
                            filter_type="low",
                            cutoff=params["cutoff"],
                            order=4,
                        )
                elif config["smooth_method"] == "splines":
                    params = config["smooth_params"]
                    padded_data = spline_smooth(
                        padded_data, s=params["smoothing_factor"]
                    )
                elif config["smooth_method"] == "arima":
                    params = config["smooth_params"]
                    order = (params["p"], params["d"], params["q"])
                    padded_data = arima_smooth(padded_data, order=order)
            except Exception as e:
                print(f"Error in smoothing: {str(e)}")

        # Remove padding
        if pad_len > 0:
            processed_data = padded_data[pad_len:-pad_len]
        else:
            processed_data = padded_data

        return processed_data, padded_data


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
                f.write(
                    f"Window Length: {config['smooth_params'].get('window_length', 7)}\n"
                )
                f.write(
                    f"Polynomial Order: {config['smooth_params'].get('polyorder', 2)}\n"
                )

            elif config["smooth_method"] == "lowess":
                f.write(f"Fraction: {config['smooth_params'].get('frac', 0.3)}\n")
                f.write(f"Iterations: {config['smooth_params'].get('it', 3)}\n")

            elif config["smooth_method"] == "kalman":
                f.write(f"EM Iterations: {config['smooth_params'].get('n_iter', 5)}\n")

            elif config["smooth_method"] == "butterworth":
                f.write(
                    f"Cutoff Frequency: {config['smooth_params'].get('cutoff', 10)} Hz\n"
                )
                f.write(
                    f"Sampling Frequency: {config['smooth_params'].get('fs', 100)} Hz\n"
                )

            elif config["smooth_method"] == "splines":
                f.write(
                    f"Smoothing Factor: {config['smooth_params'].get('smoothing_factor', 1.0)}\n"
                )

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
                for col_name, missing_count in file_info[
                    "columns_with_missing"
                ].items():
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
                    numeric_cols = original_df.select_dtypes(
                        include=[np.number]
                    ).columns
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
                    f.write(
                        f"Average Difference (first 10): {np.mean(differences):.2f}%\n"
                    )
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
                    f.write(
                        f"Total number of values compared: {len(all_differences)}\n"
                    )
                    f.write(f"Mean difference: {mean_diff:.2f}%\n")
                    f.write(f"Standard deviation: {std_diff:.2f}%\n")
                    f.write(f"Maximum difference: {max_diff:.2f}%\n")
                    f.write(f"Minimum difference: {min_diff:.2f}%\n")

                    # Add smoothing effectiveness summary
                    f.write("\nSmoothing Effectiveness Summary:\n")
                    if mean_diff > 0.01:  # If there's significant change
                        f.write(
                            "✓ Smoothing was effectively applied (significant changes detected)\n"
                        )
                        if mean_diff < 1.0:
                            f.write("  - Light smoothing effect\n")
                        elif mean_diff < 5.0:
                            f.write("  - Moderate smoothing effect\n")
                        else:
                            f.write("  - Strong smoothing effect\n")
                    else:
                        f.write(
                            "⚠ Warning: Very small changes detected. Verify if smoothing was properly applied.\n"
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
        f.write(
            "- Files are saved with the same number of decimal places as the original files.\n"
        )
        f.write(
            "- The output directory includes the processing methods in its name for easy reference.\n"
        )
        f.write("- For questions or issues, please contact: paulosantiago@usp.br\n")

    print(f"Generated detailed processing report: {report_path}")
    return report_path


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


def savgol_smooth(data, window_length, polyorder):
    """
    Applies the Savitzky-Golay filter to the data.

    Parameters:
    - data: array-like, 1D or 2D array
    - window_length: int, length of the filter window (must be odd)
    - polyorder: int, order of the polynomial to fit

    Returns:
    - filtered_data: array-like, smoothed data
    """
    data = np.asarray(data)
    return savgol_filter(data, window_length, polyorder, axis=0)


def lowess_smooth(data, frac, it):
    """
    Applies LOWESS smoothing to the data.

    Parameters:
    - data: array-like, 1D or 2D array (assumed to have no NaNs after gap filling)
    - frac: float, between 0 and 1, fraction of data to use for smoothing
    - it: int, number of iterations

    Returns:
    - filtered_data: array-like, smoothed data
    """
    data = np.asarray(data)
    x = np.arange(len(data)) if data.ndim == 1 else np.arange(data.shape[0])

    try:
        # Apply padding for better edge handling
        pad_len = int(len(data) * 0.1)  # 10% padding
        if pad_len > 0:
            if data.ndim == 1:
                padded_data = np.pad(data, (pad_len, pad_len), mode="reflect")
                padded_x = np.arange(len(padded_data))
                smoothed = lowess(
                    endog=padded_data,
                    exog=padded_x,
                    frac=frac,
                    it=it,
                    return_sorted=False,
                    is_sorted=True,
                )
                return smoothed[pad_len:-pad_len]
            else:
                padded_data = np.pad(data, ((pad_len, pad_len), (0, 0)), mode="reflect")
                padded_x = np.arange(len(padded_data))
                smoothed = np.empty_like(data)
                for j in range(data.shape[1]):
                    smoothed[:, j] = lowess(
                        endog=padded_data[:, j],
                        exog=padded_x,
                        frac=frac,
                        it=it,
                        return_sorted=False,
                        is_sorted=True,
                    )[pad_len:-pad_len]
                return smoothed
        else:
            if data.ndim == 1:
                return lowess(
                    endog=data,
                    exog=x,
                    frac=frac,
                    it=it,
                    return_sorted=False,
                    is_sorted=True,
                )
            else:
                smoothed = np.empty_like(data)
                for j in range(data.shape[1]):
                    smoothed[:, j] = lowess(
                        endog=data[:, j],
                        exog=x,
                        frac=frac,
                        it=it,
                        return_sorted=False,
                        is_sorted=True,
                    )
                return smoothed
    except Exception as e:
        print(f"Error in LOWESS smoothing: {str(e)}")
        return data  # Return original data if smoothing fails


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
                smoothed_state_means, _ = kf.em(
                    data[:, j : j + 1], n_iter=n_iter
                ).smooth(data[:, j : j + 1])
                filtered_data[:, j] = (
                    alpha * smoothed_state_means[:, 0] + (1 - alpha) * data[:, j]
                )

        else:  # mode == 2
            # Process x,y pairs together
            if n_features % 2 != 0:
                raise ValueError(
                    "For 2D mode, number of features must be even (x,y pairs)"
                )

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
                smoothed_state_means, _ = kf.em(observations, n_iter=n_iter).smooth(
                    observations
                )

                # Extract x,y positions from smoothed state means
                filtered_data[:, j] = (
                    alpha * smoothed_state_means[:, 0] + (1 - alpha) * data[:, j]
                )
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
                    print(
                        f"Warning: Not enough data points for ARIMA model in column {j}"
                    )
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

        # Create output filename with method suffix
        output_filename = f"{base_filename}_{method_suffix}.csv"
        output_path = os.path.join(dest_dir, output_filename)

        file_info = {
            "original_path": file_path,
            "original_filename": os.path.basename(file_path),
            "output_path": output_path,
            "warnings": [],
        }

        # Debug: print configuration parameters
        print("\n" + "=" * 80)
        print("DEBUG - PROCESSING PARAMETERS:")
        print(f"Interpolation Method: {config['interp_method']}")
        print(f"Maximum Gap Size: {config['max_gap']} frames")
        print(f"Smoothing Method: {config['smooth_method']}")
        print("=" * 80 + "\n")

        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)

        # Record original size
        file_info["original_size"] = len(df)
        file_info["original_columns"] = len(df.columns)

        # Preserve the first column as integers
        first_col = df.columns[0]
        df[first_col] = df[first_col].astype(int)

        # Generate complete sequence of indices
        min_frame = df[first_col].min()
        max_frame = df[first_col].max()
        print(f"Frame range: {min_frame} to {max_frame}")

        # Create DataFrame with all frames
        all_frames = pd.DataFrame({first_col: range(min_frame, max_frame + 1)})

        # Merge with original data to identify gaps
        df = pd.merge(all_frames, df, on=first_col, how="left")
        print(f"Shape after adding missing frames: {df.shape}")

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

            # Criar frames para padding
            pad_before = pd.DataFrame(
                {first_col: range(min_frame - pad_len, min_frame)}
            )
            pad_after = pd.DataFrame(
                {first_col: range(max_frame + 1, max_frame + pad_len + 1)}
            )

            # Em vez de preencher com NaN, preencher com os valores das bordas
            for col in df.columns[1:]:
                # Usar o valor do primeiro registro para o padding inicial
                pad_before[col] = df[col].iloc[0]
                # Usar o valor do último registro para o padding final
                pad_after[col] = df[col].iloc[-1]

            # Concatenar com padding
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
                    if config["interp_method"] == "linear":
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
                    for start, end in zip(gap_starts, gap_ends):
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
                            valid_data = np.array(df[col].dropna().values).reshape(
                                -1, 1
                            )
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
                    if config["interp_method"] == "linear":
                        df[col] = df[col].interpolate(
                            method="linear", limit_direction="both"
                        )
                    elif config["interp_method"] == "nearest":
                        df[col] = df[col].interpolate(
                            method="nearest", limit_direction="both"
                        )
                    elif config["interp_method"] == "cubic":
                        df[col] = df[col].interpolate(
                            method="cubic", limit_direction="both"
                        )
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
                            valid_data = np.array(df[col].dropna().values).reshape(
                                -1, 1
                            )
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
            for col in numeric_cols:
                print(f"\nSmoothing column: {col}")
                try:
                    if config["smooth_method"] == "savgol":
                        params = config["smooth_params"]
                        df[col] = savgol_smooth(
                            df[col].values, params["window_length"], params["polyorder"]
                        )
                        print(
                            f"Applied Savitzky-Golay filter with window={params['window_length']}, order={params['polyorder']}"
                        )

                    elif config["smooth_method"] == "lowess":
                        params = config["smooth_params"]
                        df[col] = lowess_smooth(
                            df[col].values, params["frac"], params["it"]
                        )
                        print(
                            f"Applied LOWESS smoothing with fraction={params['frac']}, iterations={params['it']}"
                        )

                    elif config["smooth_method"] == "kalman":
                        params = config["smooth_params"]
                        df[col] = kalman_smooth(
                            df[col].values, params["n_iter"], params["mode"]
                        )
                        print(
                            f"Applied Kalman filter with {params['n_iter']} iterations in {params['mode']} mode"
                        )

                    elif config["smooth_method"] == "butterworth":
                        params = config["smooth_params"]
                        try:
                            data = df[
                                col
                            ].to_numpy()  # Use to_numpy() instead of .values
                            if np.isnan(data).any():
                                print(
                                    f"Warning: Column {col} contains NaN values. Interpolating before filtering..."
                                )
                                data = (
                                    pd.Series(data).interpolate(method="linear").values
                                )

                            fs = float(params["fs"])
                            cutoff = float(params["cutoff"])

                            # Garantir que a frequência de corte seja razoável
                            if cutoff >= fs / 2:
                                cutoff = fs / 2 - 1
                                print(
                                    f"Warning: Adjusted cutoff frequency to {cutoff} Hz"
                                )

                            # Usar a função butter_filter do filter_utils.py
                            filtered = butter_filter(
                                data,
                                fs=fs,
                                filter_type="low",
                                cutoff=cutoff,
                                order=4,
                                padding=True,
                            )

                            if not np.array_equal(
                                np.array(filtered), np.array(data)
                            ):  # Verificar se houve mudança
                                df[col] = filtered
                                print(f"Successfully filtered column {col}")
                            else:
                                print(
                                    f"Warning: No change detected after filtering column {col}"
                                )

                        except Exception as e:
                            print(f"Error filtering column {col}: {str(e)}")
                            # Manter dados originais em caso de erro
                            print("Keeping original data for this column")

                    elif config["smooth_method"] == "splines":
                        params = config["smooth_params"]
                        df[col] = spline_smooth(
                            df[col].values, s=float(params["smoothing_factor"])
                        )
                        print(
                            f"Applied Spline smoothing with smoothing factor={params['smoothing_factor']}"
                        )

                    elif config["smooth_method"] == "arima":
                        params = config["smooth_params"]
                        order = (int(params["p"]), int(params["d"]), int(params["q"]))
                        df[col] = arima_smooth(df[col].values, order=order)
                        print(f"Applied ARIMA filter with order={order}")

                except Exception as e:
                    error_msg = f"Error smoothing column {col}: {str(e)}"
                    print(error_msg)
                    file_info["warnings"].append(error_msg)

        # Remove padding
        print(
            f"\nRemoving padding (keeping only frames from {min_frame} to {max_frame})"
        )
        df = df[df[first_col].between(min_frame, max_frame)].reset_index(drop=True)
        print(f"Final shape after removing padding: {df.shape}")

        # Detect float format from original file
        float_format = detect_float_format(file_info["original_path"])
        print(f"Using float format: {float_format}")

        # Save processed DataFrame
        print(f"\nSaving processed file to: {output_path}")
        df.to_csv(output_path, index=False, float_format=float_format)
        print("File saved successfully!")

        return file_info

    except Exception as e:
        # Return basic info with error in case of failure
        filename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(filename)[0]}_processed.csv"
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


def run_fill_split_dialog(parent=None):
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting script: interp_smooth_split.py")
    print("================================================")

    # Open configuration dialog as main window
    config_dialog = InterpolationConfigDialog(parent=parent)

    # Ensure window is visible before starting main loop
    config_dialog.window.update()
    config_dialog.window.update_idletasks()
    config_dialog.window.deiconify()
    config_dialog.window.lift()
    config_dialog.window.focus_force()

    # Additional visibility commands
    config_dialog.window.state("normal")
    config_dialog.window.wm_attributes("-alpha", 1.0)
    config_dialog.window.wm_attributes("-disabled", False)
    config_dialog.window.attributes("-topmost", True)
    config_dialog.window.attributes("-topmost", False)

    print("GUI window should now be visible...")

    # Wait for dialog to complete
    config_dialog.window.wait_window()

    if not hasattr(config_dialog, "result") or config_dialog.result is None:
        print("Operation canceled by user.")
        print("================================================")
        return

    config = config_dialog.result

    # Select source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        print("Operation canceled by user.")
        print("================================================")
        return

    # Create destination directory with method names
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Create descriptive names for methods
    interp_name = config["interp_method"]

    # Add information about smoothing parameters if used
    smooth_info = "no_smooth"
    if config["smooth_method"] != "none":
        smooth_info = config["smooth_method"]
        try:
            if config["smooth_method"] == "butterworth":
                cutoff = config["smooth_params"].get("cutoff", 10)
                smooth_info += f"_cut{cutoff}"
            elif config["smooth_method"] == "savgol":
                window = config["smooth_params"].get("window_length", 7)
                poly = config["smooth_params"].get("polyorder", 2)
                smooth_info += f"_w{window}p{poly}"
            elif config["smooth_method"] == "lowess":
                frac = config["smooth_params"].get("frac", 0.3)
                it = config["smooth_params"].get("it", 3)
                smooth_info += f"_frac{int(frac*100)}_it{it}"
                print("\nDEBUG - LOWESS Configuration:")
                print(f"Fraction: {frac}")
                print(f"Iterations: {it}")
            elif config["smooth_method"] == "kalman":
                n_iter = config["smooth_params"].get("n_iter", 5)
                mode = config["smooth_params"].get("mode", 1)
                smooth_info += f"_iter{n_iter}_mode{mode}"
            elif config["smooth_method"] == "splines":
                s = config["smooth_params"].get("smoothing_factor", 1.0)
                smooth_info += f"_s{s}"
            elif config["smooth_method"] == "arima":
                p = config["smooth_params"].get("p", 1)
                d = config["smooth_params"].get("d", 0)
                q = config["smooth_params"].get("q", 0)
                smooth_info += f"_p{p}d{d}q{q}"
        except Exception as e:
            print(f"Warning: Error formatting smooth_info: {str(e)}")
            smooth_info = config["smooth_method"]  # Fallback to basic name

    # Directory with informative name
    dest_dir_name = f"processed_{interp_name}_{smooth_info}_{timestamp}"
    dest_dir = os.path.join(source_dir, dest_dir_name)
    os.makedirs(dest_dir, exist_ok=True)

    # List to store information about processed files
    processed_files = []

    # Process each file
    for filename in os.listdir(source_dir):
        if filename.endswith(".csv"):
            try:
                file_info = process_file(
                    os.path.join(source_dir, filename), dest_dir, config
                )
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
                        "warnings": [f"Error processing file: {str(e)}"],
                        "error": True,
                        "original_size": 0,
                        "original_columns": 0,
                        "total_missing": 0,
                        "columns_with_missing": {},
                        "output_path": None,
                    }
                )

    # Filtra arquivos processados para remover None
    processed_files = [pf for pf in processed_files if pf is not None]

    # Generate detailed processing report
    if processed_files:
        report_path = generate_report(dest_dir, config, processed_files)
        messagebox.showinfo(
            "Complete",
            f"Processing complete. Results saved in {dest_dir}\n"
            f"A detailed processing report has been saved to:\n{report_path}",
        )
    else:
        messagebox.showwarning("Warning", "No files were successfully processed.")


if __name__ == "__main__":
    run_fill_split_dialog()
