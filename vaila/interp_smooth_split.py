"""
===============================================================================
interp_smooth_split.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 14 October 2024
Update Date: 01 April 2025
Version: 0.0.2
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
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from scipy.signal import savgol_filter, butter, filtfilt, sosfiltfilt, firwin
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from tkinter import filedialog, messagebox, Toplevel, Button, Label
from scipy.interpolate import UnivariateSpline
import tkinter as tk
from rich import print
from statsmodels.tsa.arima.model import ARIMA


class InterpolationConfigDialog(tk.simpledialog.Dialog):
    def __init__(self, parent):
        # Initialize StringVar variables with default values before calling parent constructor
        self.savgol_window = tk.StringVar(value="7")  # Default window length
        self.savgol_poly = tk.StringVar(value="3")  # Default polynomial order
        self.lowess_frac = tk.StringVar(value="0.3")  # Default fraction
        self.lowess_it = tk.StringVar(value="3")  # Default iterations
        self.butter_cutoff = tk.StringVar(value="10")  # Default cutoff frequency
        self.butter_fs = tk.StringVar(value="100")  # Default sampling frequency
        self.kalman_iterations = tk.StringVar(value="5")  # Default Kalman iterations
        self.kalman_mode = tk.StringVar(value="1")  # Default Kalman mode
        self.spline_smoothing = tk.StringVar(value="1.0")  # Default smoothing factor
        self.arima_p = tk.StringVar(value="1")  # AR order
        self.arima_d = tk.StringVar(value="0")  # Difference order
        self.arima_q = tk.StringVar(value="0")  # MA order
        
        # Call parent constructor after initializing variables
        super().__init__(parent, title="Interpolation Configuration")

    def body(self, master):
        # Create main frame with scrollbar
        main_container = tk.Frame(master)
        main_container.pack(fill="both", expand=True)
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(main_container, width=800, height=600)
        scrollbar = tk.Scrollbar(
            main_container, orient="vertical", command=canvas.yview
        )
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Create two columns
        left_column = tk.Frame(scrollable_frame)
        right_column = tk.Frame(scrollable_frame)
        
        left_column.grid(row=0, column=0, sticky="nw", padx=10)
        right_column.grid(row=0, column=1, sticky="nw", padx=10)
        
        # ====== LEFT COLUMN - METHODS SELECTION ======
        
        # Frame for the gap filling method
        interp_frame = tk.LabelFrame(
            left_column, text="Gap Filling Method", padx=5, pady=5
        )
        interp_frame.pack(fill="x", pady=5, anchor="n")
        
        # List of interpolation methods
        interp_text = """
1 - Linear Interpolation (simple, works well for most cases)
2 - Cubic Spline (smooth transitions between points)
3 - Nearest Value (use closest available value)
4 - Kalman Filter (good for movement data, models physics)
5 - None (leave gaps as NaN)
6 - Skip (keep original data, apply only smoothing)"""
        
        tk.Label(interp_frame, text=interp_text, justify="left").pack(
            anchor="w", padx=5
        )
        
        tk.Label(interp_frame, text="Enter gap filling method (1-6):").pack(
            anchor="w", padx=5, pady=5
        )
        self.interp_entry = tk.Entry(interp_frame)
        self.interp_entry.insert(0, "1")  # Default: linear
        self.interp_entry.pack(fill="x", padx=5)
        
        # Frame for smoothing method
        smooth_frame = tk.LabelFrame(
            left_column, text="Smoothing Method", padx=5, pady=5
        )
        smooth_frame.pack(fill="x", pady=5, anchor="n")
        
        # List of smoothing methods
        smooth_text = """
1 - None (no smoothing)
2 - Savitzky-Golay Filter (preserves peaks and valleys)
3 - LOWESS (adapts to local trends)
4 - Kalman Filter (state estimation with noise reduction)
5 - Butterworth Filter (4th order, frequency domain filtering)
6 - Spline Smoothing (flexible curve fitting)
7 - ARIMA (time series modeling and filtering)"""
        
        tk.Label(smooth_frame, text=smooth_text, justify="left").pack(
            anchor="w", padx=5
        )
        
        # Important explanatory note
        tk.Label(
            smooth_frame,
            text="Note: Smoothing is applied to the entire data after filling gaps",
            foreground="blue",
            justify="left",
        ).pack(anchor="w", padx=5)

        tk.Label(smooth_frame, text="Enter smoothing method (1-7):").pack(
            anchor="w", padx=5, pady=5
        )
        self.smooth_entry = tk.Entry(smooth_frame)
        self.smooth_entry.insert(0, "1")  # Default: no smoothing
        self.smooth_entry.pack(fill="x", padx=5)
        
        # Add button to update parameters based on selection
        update_button = tk.Button(
            smooth_frame, text="Update Parameters", command=self.update_params_frame
        )
        update_button.pack(pady=5)

        # Frame for split option
        split_frame = tk.LabelFrame(
            left_column, text="Split Configuration", padx=5, pady=5
        )
        split_frame.pack(fill="x", pady=5, anchor="n")
        
        self.split_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            split_frame, text="Split data into two parts", variable=self.split_var
        ).pack(anchor="w")
        
        # ====== RIGHT COLUMN - PARAMETERS ======
        
        # Frame for specific method parameters
        self.params_frame = tk.LabelFrame(
            right_column, text="Method Parameters", padx=5, pady=5
        )
        self.params_frame.pack(fill="x", pady=5, anchor="n")
        
        # Create empty widgets for parameters
        self.params_widgets = []
        self.param_entries = {}  # Dictionary to keep track of parameter entries

        # Adicionar botão de confirmação de parâmetros
        self.confirm_button = tk.Button(
            right_column,
            text="Confirm Parameters",
            command=self.confirm_parameters,
            bg="lightgreen",
            font=("Arial", 10, "bold"),
        )
        self.confirm_button.pack(pady=10)
        
        # Frame for padding
        padding_frame = tk.LabelFrame(
            right_column, text="Padding Configuration", padx=5, pady=5
        )
        padding_frame.pack(fill="x", pady=5, anchor="n")
        
        tk.Label(padding_frame, text="Padding length (% of data):").pack(anchor="w")
        self.padding_entry = tk.Entry(padding_frame)
        self.padding_entry.insert(0, "10")  # Default 10%
        self.padding_entry.pack(fill="x", padx=5)
        
        # Frame for gap configuration
        gap_frame = tk.LabelFrame(
            right_column, text="Gap Configuration", padx=5, pady=5
        )
        gap_frame.pack(fill="x", pady=5, anchor="n")
        
        tk.Label(gap_frame, text="Maximum gap size to fill (frames):").pack(anchor="w")
        self.max_gap_entry = tk.Entry(gap_frame)
        self.max_gap_entry.insert(0, "60")  # Default 60 frames
        self.max_gap_entry.pack(fill="x", padx=5)
        
        # Explanatory label
        tk.Label(
            gap_frame,
                text="Note: Gaps larger than this value will be left as NaN. Set to 0 to fill all gaps.", 
            foreground="blue",
            justify="left",
            wraplength=350,
        ).pack(anchor="w", padx=5, pady=2)
        
        # Initialize the parameters frame
        self.update_params_frame()
        
        # Bind the mouse wheel to the canvas for scrolling
        self.bind_mousewheel(canvas)
        
        return self.interp_entry  # Initial focus
    
    def bind_mousewheel(self, canvas):
        """Bind mouse wheel to scrolling canvas"""

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def update_params_frame(self):
        try:
            # Clear existing widgets
            for widget in self.params_widgets:
                widget.destroy()
            self.params_widgets.clear()
            self.param_entries.clear()
            
            smooth_method = int(self.smooth_entry.get())
            
            if smooth_method == 2:  # Savitzky-Golay
                # Add section header with special emphasis
                header = tk.Label(
                    self.params_frame,
                    text="Savitzky-Golay Parameters:",
                    font=("", 10, "bold"),
                )
                header.pack(anchor="w", padx=5, pady=5)
                
                label1 = tk.Label(
                    self.params_frame,
                    text="Window length (must be odd):",
                    foreground="red",
                )
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(self.params_frame, textvariable=self.savgol_window)
                entry1.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry1.bind(
                    "<Return>",
                    lambda e: self.update_parameter_value(e, self.savgol_window),
                )
                
                label2 = tk.Label(
                    self.params_frame, text="Polynomial order:", foreground="red"
                )
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.savgol_poly)
                entry2.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry2.bind(
                    "<Return>",
                    lambda e: self.update_parameter_value(e, self.savgol_poly),
                )

                # Add detailed explanation
                explanation = tk.Label(
                    self.params_frame,
                    text=(
                        "Savitzky-Golay Filter Parameters:\n"
                        "• Window Length: Size of the window (must be odd)\n"
                        "  - Smaller values (3-7): preserve fine details\n"
                        "  - Larger values (9-21): stronger smoothing\n"
                        "  - Recommended: 7-11 for biomechanical data\n\n"
                        "• Polynomial Order: Degree of the fitting polynomial\n"
                        "  - Order 2: stronger smoothing\n"
                        "  - Order 3-4: preserves peaks and valleys\n"
                        "  - Must be less than window length"
                    ),
                    foreground="blue",
                    justify="left",
                )
                explanation.pack(anchor="w", padx=5, pady=2)
                self.params_widgets.extend(
                    [header, label1, entry1, label2, entry2, explanation]
                )

                # Guardar referências aos entries para acesso posterior
                self.param_entries["window_length"] = entry1
                self.param_entries["polyorder"] = entry2
                
            elif smooth_method == 3:  # LOWESS
                # Add section header with special emphasis
                header = tk.Label(
                    self.params_frame, text="LOWESS Parameters:", font=("", 10, "bold")
                )
                header.pack(anchor="w", padx=5, pady=5)
                
                label1 = tk.Label(
                    self.params_frame, text="Fraction (0-1):", foreground="red"
                )
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(self.params_frame, textvariable=self.lowess_frac)
                entry1.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry1.bind(
                    "<Return>",
                    lambda e: self.update_parameter_value(e, self.lowess_frac),
                )
                
                label2 = tk.Label(
                    self.params_frame, text="Number of iterations:", foreground="red"
                )
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.lowess_it)
                entry2.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry2.bind(
                    "<Return>", lambda e: self.update_parameter_value(e, self.lowess_it)
                )

                # Add detailed explanation
                explanation = tk.Label(
                    self.params_frame,
                    text=(
                        "LOWESS Filter Parameters:\n"
                        "• Fraction: Proportion of points used (0-1)\n"
                        "  - 0.2-0.3: preserves local details\n"
                        "  - 0.4-0.6: moderate smoothing\n"
                        "  - 0.7-1.0: strong smoothing\n\n"
                        "• Iterations: Number of robustifying iterations\n"
                        "  - 1-2: faster processing\n"
                        "  - 3-4: better outlier removal\n"
                        "  - 5+: more robust smoothing"
                    ),
                    foreground="blue",
                    justify="left",
                )
                explanation.pack(anchor="w", padx=5, pady=2)
                self.params_widgets.extend(
                    [header, label1, entry1, label2, entry2, explanation]
                )

                # Guardar referências aos entries para acesso posterior
                self.param_entries["frac"] = entry1
                self.param_entries["it"] = entry2
                
            elif smooth_method == 4:  # Kalman
                # Add section header with special emphasis
                header = tk.Label(
                    self.params_frame,
                    text="Kalman Filter Parameters:",
                    font=("", 10, "bold"),
                )
                header.pack(anchor="w", padx=5, pady=5)
                
                # Number of iterations
                label1 = tk.Label(
                    self.params_frame, text="Number of EM iterations:", foreground="red"
                )
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(
                    self.params_frame, textvariable=self.kalman_iterations
                )
                entry1.pack(fill="x", padx=5, pady=2)
                entry1.bind(
                    "<Return>",
                    lambda e: self.update_parameter_value(e, self.kalman_iterations),
                )

                # Processing mode selection
                label2 = tk.Label(
                    self.params_frame,
                    text="Processing Mode (1=1D, 2=2D):",
                    foreground="red",
                )
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.kalman_mode)
                entry2.pack(fill="x", padx=5, pady=2)
                entry2.bind(
                    "<Return>",
                    lambda e: self.update_parameter_value(e, self.kalman_mode),
                )

                # Add detailed explanation
                explanation = tk.Label(
                    self.params_frame,
                    text=(
                        "Kalman Filter Parameters:\n"
                        "• EM Iterations: Algorithm iterations\n"
                        "  - 3-5: basic estimation\n"
                        "  - 6-10: better convergence\n"
                        "  - 10+: more precise convergence\n\n"
                        "• Processing Mode:\n"
                        "  - 1: Process each column independently (1D)\n"
                        "  - 2: Process X,Y pairs together (2D, requires even number of columns)\n\n"
                        "Characteristics:\n"
                        "• Optimal for Gaussian noise\n"
                        "• Preserves movement trends\n"
                        "• Adapts to velocity changes"
                    ),
                    foreground="blue",
                    justify="left",
                )
                explanation.pack(anchor="w", padx=5, pady=2)
                self.params_widgets.extend(
                    [header, label1, entry1, label2, entry2, explanation]
                )

                # Guardar referências aos entries para acesso posterior
                self.param_entries["n_iter"] = entry1
                self.param_entries["mode"] = entry2
                
            elif smooth_method == 5:  # Butterworth
                # Add section header with special emphasis
                header = tk.Label(
                    self.params_frame,
                    text="Butterworth Filter Parameters:",
                    font=("", 10, "bold"),
                )
                header.pack(anchor="w", padx=5, pady=5)
                
                label1 = tk.Label(
                    self.params_frame,
                    text="Cutoff frequency (Hz, e.g. 4):",
                    foreground="red",
                )
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(self.params_frame, textvariable=self.butter_cutoff)
                entry1.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry1.bind(
                    "<Return>",
                    lambda e: self.update_parameter_value(e, self.butter_cutoff),
                )

                label2 = tk.Label(
                    self.params_frame,
                    text="Sampling frequency (Hz, e.g. 50):",
                    foreground="red",
                )
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.butter_fs)
                entry2.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry2.bind(
                    "<Return>", lambda e: self.update_parameter_value(e, self.butter_fs)
                )
                
                # Strong reminder for Butterworth
                reminder = tk.Label(
                    self.params_frame,
                                  text="* You MUST enter both cutoff and sampling frequency values.\nNo default values will be used.",
                    foreground="red",
                    font=("", 9, "bold"),
                    justify="left",
                )
                reminder.pack(anchor="w", padx=5, pady=5)
                
                # Add detailed explanation
                explanation = tk.Label(
                    self.params_frame,
                    text=(
                        "Butterworth Filter Parameters:\n"
                        "• Cutoff Frequency (Hz):\n"
                        "  - 4-6 Hz: slow movements/posture\n"
                        "  - 7-12 Hz: normal movements\n"
                        "  - 13-20 Hz: fast movements\n\n"
                        "• Sampling Frequency (Hz):\n"
                        "  - Data capture frequency\n"
                        "  - Common in biomechanics: 50-200 Hz\n"
                        "  - Must be > 2x cutoff (Nyquist)\n\n"
                        "Tip: Lower cutoff = more smoothing"
                    ),
                    foreground="blue",
                    justify="left",
                )
                explanation.pack(anchor="w", padx=5, pady=2)
                self.params_widgets.extend(
                    [header, label1, entry1, label2, entry2, reminder, explanation]
                )

                # Guardar referências aos entries para acesso posterior
                self.param_entries["cutoff"] = entry1
                self.param_entries["fs"] = entry2

            elif smooth_method == 6:  # Splines
                # Add section header with special emphasis
                header = tk.Label(
                    self.params_frame,
                    text="Spline Smoothing Parameters:",
                    font=("", 10, "bold"),
                )
                header.pack(anchor="w", padx=5, pady=5)

                label = tk.Label(
                    self.params_frame,
                    text="Smoothing factor (s):",
                    foreground="red",
                )
                label.pack(anchor="w", padx=5, pady=2)
                entry = tk.Entry(self.params_frame, textvariable=self.spline_smoothing)
                entry.pack(fill="x", padx=5, pady=2)
                entry.bind(
                    "<Return>",
                    lambda e: self.update_parameter_value(e, self.spline_smoothing),
                )

                # Add detailed explanation
                explanation = tk.Label(
                    self.params_frame,
                    text=(
                        "Spline Smoothing Parameters:\n"
                        "• Smoothing Factor (s):\n"
                        "  - s = 0: Pure interpolation\n"
                        "  - 0 < s < 1: Light smoothing\n"
                        "  - s = 1: Balanced smoothing\n"
                        "  - 1 < s < 10: Moderate smoothing\n"
                        "  - s ≥ 10: Strong smoothing\n\n"
                        "Characteristics:\n"
                        "• Preserves curve continuity\n"
                        "• Ideal for biomechanical data\n"
                        "• Prevents artificial oscillations"
                    ),
                    foreground="blue",
                    justify="left",
                )
                explanation.pack(anchor="w", padx=5, pady=2)

                self.params_widgets.extend([header, label, entry, explanation])
                self.param_entries["smoothing_factor"] = entry

            elif smooth_method == 7:  # ARIMA
                # Add section header with special emphasis
                header = tk.Label(
                    self.params_frame,
                    text="ARIMA Parameters:",
                    font=("", 10, "bold"),
                )
                header.pack(anchor="w", padx=5, pady=5)

                # AR order (p)
                label1 = tk.Label(
                    self.params_frame,
                    text="AR order (p):",
                    foreground="red",
                )
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(self.params_frame, textvariable=self.arima_p)
                entry1.pack(fill="x", padx=5, pady=2)

                # Difference order (d)
                label2 = tk.Label(
                    self.params_frame,
                    text="Difference order (d):",
                    foreground="red",
                )
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.arima_d)
                entry2.pack(fill="x", padx=5, pady=2)

                # MA order (q)
                label3 = tk.Label(
                    self.params_frame,
                    text="MA order (q):",
                    foreground="red",
                )
                label3.pack(anchor="w", padx=5, pady=2)
                entry3 = tk.Entry(self.params_frame, textvariable=self.arima_q)
                entry3.pack(fill="x", padx=5, pady=2)

                # Add detailed explanation
                explanation = tk.Label(
                    self.params_frame,
                    text=(
                        "ARIMA Model Parameters (p,d,q):\n"
                        "• p: Autoregressive Order\n"
                        "  - 0: No AR component\n"
                        "  - 1: Short-term dependence\n"
                        "  - 2+: Long-term dependence\n\n"
                        "• d: Differencing Order\n"
                        "  - 0: Stationary data\n"
                        "  - 1: Removes linear trend\n"
                        "  - 2: Removes quadratic trend\n\n"
                        "• q: Moving Average Order\n"
                        "  - 0: No MA component\n"
                        "  - 1: Short-term correction\n"
                        "  - 2+: Long-term correction\n\n"
                        "Common combinations:\n"
                        "• (1,0,0): Simple AR model\n"
                        "• (0,1,1): Simple exponential smoothing\n"
                        "• (1,1,1): ARIMA with trend"
                    ),
                    foreground="blue",
                    justify="left",
                )
                explanation.pack(anchor="w", padx=5, pady=2)

                self.params_widgets.extend(
                    [
                        header,
                        label1,
                        entry1,
                        label2,
                        entry2,
                        label3,
                        entry3,
                        explanation,
                    ]
                )
                self.param_entries.update({"p": entry1, "d": entry2, "q": entry3})

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
            label = tk.Label(
                self.params_frame, text="Please enter valid method numbers"
            )
            label.pack(anchor="w", padx=5, pady=5)
            self.params_widgets.append(label)
    
    def update_parameter_value(self, event, stringvar):
        """Atualiza o valor da StringVar quando o usuário pressiona Enter"""
        widget = event.widget
        value = widget.get()
        stringvar.set(value)
        # Move o foco para o próximo widget
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
        """Confirma e atualiza os parâmetros antes do processamento"""
        try:
            # Força a perda de foco de todos os widgets de entrada
            self.focus()

            # Força atualização explícita dos valores dos Entry widgets
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

            # Força a atualização dos widgets
            self.update_idletasks()

            # Captura o método de suavização atual
            smooth_method = int(self.smooth_entry.get())

            # Print dos parâmetros confirmados no terminal
            print("\n" + "=" * 50)
            print("CONFIRMED PARAMETERS:")
            print("=" * 50)
            print(f"Gap Filling Method: {self.interp_entry.get()}")
            print(f"Smoothing Method: {self.smooth_entry.get()}")
            print(f"Max Gap Size: {self.max_gap_entry.get()} frames")
            print(f"Padding: {self.padding_entry.get()}%")
            print(f"Split Data: {'Yes' if self.split_var.get() else 'No'}")

            # Define params_text e imprime parâmetros específicos do método
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
                smooth_params = {"n_iter": n_iter, "mode": mode}
                params_text = f"EM Iterations: {n_iter}, Processing Mode: {mode}"
                print(f"APPLY: Kalman settings - n_iter={n_iter}, mode={mode}")
            elif smooth_method == 5:  # Butterworth
                cutoff = float(self.butter_cutoff.get())
                fs = float(self.butter_fs.get())
                params_text = f"Cutoff: {cutoff} Hz, Sampling Frequency: {fs} Hz"
                print("\nButterworth Filter Parameters:")
                print(f"- Cutoff Frequency: {cutoff} Hz")
                print(f"- Sampling Frequency: {fs} Hz")

                # Validação adicional para Butterworth
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

            # Mostra mensagem de confirmação com os valores atuais
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

            # Muda a cor do botão para indicar que os parâmetros foram confirmados
            self.confirm_button.configure(
                bg="pale green", text="Parameters Confirmed ✓"
            )

        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")

    def apply(self):
        try:
            # Verifica se os parâmetros foram confirmados
            if self.confirm_button["text"] != "Parameters Confirmed ✓":
                if not messagebox.askyesno(
                    "Warning",
                    "Parameters have not been confirmed. Do you want to proceed anyway?",
                ):
                    return

            # Força a atualização dos valores dos widgets antes de coletá-los
            self.update_idletasks()

            # Debug: Print dos valores das StringVar antes do processamento
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

            # Debug: Print dos valores dos Entry widgets
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
            
            # Prepara os parâmetros de smoothing com base no método escolhido
            smooth_params = {}
            if smooth_method == 2:  # Savitzky-Golay
                window_length = int(self.savgol_window.get())
                polyorder = int(self.savgol_poly.get())
                smooth_params = {"window_length": window_length, "polyorder": polyorder}
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
                print(f"APPLY: Butterworth settings - cutoff={cutoff} Hz, fs={fs} Hz")
            
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

            # Exibe um resumo dos parâmetros escolhidos
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
                summary += f"\nSpline Smoothing Parameters:\n- Smoothing Factor: {s}"
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
        f.write(f"PROCESSING REPORT - VAILA INTERPOLATION AND SMOOTHING TOOL\n")
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
                        (processed_values - original_values) / original_values * 100
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
                        (all_processed - all_original) / all_original * 100
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
        f.write(f"Python Version: {os.sys.version.split()[0]}\n")
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


# Scripts to filter/smooth data
def butter_filter(data, fs, filter_type="low", cutoff=None, lowcut=None, highcut=None, order=4, padding=True):
    """
    Applies a Butterworth filter to the input data.
    """
    # Convert data to numpy array and handle NaN values
    data = np.asarray(data)
    if np.isnan(data).any():
        # Interpolate NaN values before filtering
        nan_mask = np.isnan(data)
        x = np.arange(len(data))
        data = pd.Series(data).interpolate(method='linear', limit_direction='both').values

    # Calculate Nyquist frequency
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq

    # Design the Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Apply padding if requested
    if padding:
        # Use a fixed padding length that's reasonable for your data
        pad_len = min(int(len(data) * 0.1), 20)  # 10% of data length or 20 points, whichever is smaller
        padded_data = np.pad(data, pad_len, mode='reflect')
        filtered_data = filtfilt(b, a, padded_data)
        filtered_data = filtered_data[pad_len:-pad_len]
        else:
        filtered_data = filtfilt(b, a, data)

    # Restore NaN values if they existed
    if np.isnan(data).any():
        filtered_data[nan_mask] = np.nan

    return filtered_data


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
    Applies Kalman smoothing to the data.

    Parameters:
    - data: array-like, 1D or 2D array
    - n_iter: int, number of EM iterations
    - mode: int, 1 for 1D processing or 2 for 2D processing

    Returns:
    - filtered_data: array-like, smoothed data
    """
    data = np.asarray(data)

    # Handle 1D data
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples = data.shape[0]
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
                alpha = 0.7  # Smoothing factor
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
                observation_matrix = np.array([
                    [1, 0, 0, 0, 0, 0],  # observe x
                    [0, 1, 0, 0, 0, 0]   # observe y
                ])
                
                # Initialize state mean with first observation and zero velocities/accelerations
                initial_state_mean = np.array([
                    data[0, j],    # initial x
                    data[0, j+1],  # initial y
                    0,             # initial vx
                    0,             # initial vy
                    0,             # initial ax
                    0              # initial ay
                ])
                
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
                observations = np.column_stack([data[:, j], data[:, j+1]])
                
                # Apply EM algorithm and smoothing
                smoothed_state_means, _ = kf.em(observations, n_iter=n_iter).smooth(observations)
                
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
    Applies ARIMA smoothing to the input data using the ARIMA.filter method.

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
            model = ARIMA(data, order=order)
            result = model.fit()  # Primeiro ajusta o modelo
            return result.fittedvalues  # Depois obtém os valores suavizados
        except Exception as e:
            print(f"Error in ARIMA smoothing: {str(e)}")
            return data  # Return original data if smoothing fails
    else:
        # For 2D data, apply ARIMA smoothing column by column
        smoothed = np.empty_like(data)
        for j in range(data.shape[1]):
            try:
                model = ARIMA(data[:, j], order=order)
                result = model.fit()  # Primeiro ajusta o modelo
                smoothed[:, j] = (
                    result.fittedvalues
                )  # Depois obtém os valores suavizados
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
        
        # Create frames for padding
            pad_before = pd.DataFrame(
                {first_col: range(min_frame - pad_len, min_frame)}
            )
            pad_after = pd.DataFrame(
                {first_col: range(max_frame + 1, max_frame + pad_len + 1)}
            )
        
        # Add empty columns in paddings
        for col in df.columns[1:]:
            pad_before[col] = np.nan
            pad_after[col] = np.nan
        
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
                print(f"Limiting gap filling to gaps of {max_gap} frames or less")
                
                # Identify consecutive NaN groups
                nan_groups = []
                start_idx = None
                original_nan_mask = df[col].isna()

                for i in range(len(original_nan_mask)):
                    if original_nan_mask.iloc[i]:
                        if start_idx is None:
                            start_idx = i
                    elif start_idx is not None:
                        nan_groups.append((start_idx, i - 1))
                        start_idx = None

                if start_idx is not None:
                    nan_groups.append((start_idx, len(original_nan_mask) - 1))

                # Apply interpolation method
                if config["interp_method"] == "linear":
                    df[col] = df[col].interpolate(
                        method="linear", limit_direction="both"
                    )
                elif config["interp_method"] == "nearest":
                    df[col] = df[col].interpolate(
                        method="nearest", limit_direction="both"
                    )
                elif config["interp_method"] == "cubic":
                    try:
                        valid_mask = ~nan_mask
                        if valid_mask.sum() >= 3:
                            x_valid = df[first_col][valid_mask].values
                            y_valid = df[col][valid_mask].values
                            sort_idx = np.argsort(x_valid)
                            x_valid = x_valid[sort_idx]
                            y_valid = y_valid[sort_idx]
                            cs = CubicSpline(x_valid, y_valid, bc_type="natural")
                            x_nan = df[first_col][nan_mask].values
                            df.loc[nan_mask, col] = cs(x_nan)
                        else:
                            print(
                                "Not enough points for cubic spline, falling back to linear"
                            )
                            df[col] = df[col].interpolate(
                                method="linear", limit_direction="both"
                            )
                    except Exception as e:
                        print(f"Error applying Cubic Spline: {str(e)}")
                        print("Falling back to linear interpolation")
                        df[col] = df[col].interpolate(
                            method="linear", limit_direction="both"
                        )
                elif config["interp_method"] == "kalman":
                    try:
                        temp_series = df[col].copy()
                        temp_filled = temp_series.interpolate(
                            method="linear", limit_direction="both"
                        )
                        kf = KalmanFilter(
                            initial_state_mean=np.mean(temp_filled.dropna()),
                            n_dim_obs=1,
                            n_dim_state=2,
                        )
                        observations = np.array(
                            [[x] if not np.isnan(x) else None for x in df[col].values]
                        )
                        n_iter = config["smooth_params"].get("n_iter", 5)
                        smoothed_state_means, _ = kf.em(
                            observations, n_iter=n_iter
                        ).smooth(observations)
                        df.loc[nan_mask, col] = smoothed_state_means[nan_mask, 0]
                    except Exception as e:
                        print(f"Error applying Kalman for gap filling: {str(e)}")
                        print("Falling back to linear interpolation")
                        df[col] = df[col].interpolate(
                            method="linear", limit_direction="both"
                        )

                # Revert interpolation for gaps larger than max_gap
                for start, end in nan_groups:
                    gap_size = end - start + 1
                    if gap_size > max_gap:
                        print(
                            f"Reverting interpolation for gap of size {gap_size} at frames {start}-{end}"
                        )
                        df.loc[df.index[start : end + 1], col] = np.nan

                remaining_nans = df[col].isna().sum()
                print(f"After interpolation, {remaining_nans} NaN values remain")
            
            else:  # No gap size limit
            if config["interp_method"] == "linear":
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
            elif config["interp_method"] == "nearest":
                df[col] = df[col].interpolate(method="nearest", limit_direction="both")
            elif config["interp_method"] == "cubic":
                    try:
                        valid_mask = ~nan_mask
                    if valid_mask.sum() >= 3:
                            x_valid = df[first_col][valid_mask].values
                            y_valid = df[col][valid_mask].values
                            sort_idx = np.argsort(x_valid)
                            x_valid = x_valid[sort_idx]
                            y_valid = y_valid[sort_idx]
                        cs = CubicSpline(x_valid, y_valid, bc_type="natural")
                            x_nan = df[first_col][nan_mask].values
                            df.loc[nan_mask, col] = cs(x_nan)
                        else:
                        print(
                            "Not enough points for cubic spline, falling back to linear"
                        )
                        df[col] = df[col].interpolate(
                            method="linear", limit_direction="both"
                        )
                    except Exception as e:
                        print(f"Error applying Cubic Spline: {str(e)}")
                    print("Falling back to linear interpolation")
                    df[col] = df[col].interpolate(
                        method="linear", limit_direction="both"
                    )
            elif config["interp_method"] == "kalman":
                try:
                        temp_series = df[col].copy()
                    temp_filled = temp_series.interpolate(
                        method="linear", limit_direction="both"
                    )
                        kf = KalmanFilter(
                            initial_state_mean=np.mean(temp_filled.dropna()),
                            n_dim_obs=1,
                        n_dim_state=2,
                    )
                    observations = np.array(
                        [[x] if not np.isnan(x) else None for x in df[col].values]
                    )
                    n_iter = config["smooth_params"].get("n_iter", 5)
                    smoothed_state_means, _ = kf.em(observations, n_iter=n_iter).smooth(
                        observations
                    )
                        df.loc[nan_mask, col] = smoothed_state_means[nan_mask, 0]
                    except Exception as e:
                        print(f"Error applying Kalman for gap filling: {str(e)}")
                    print("Falling back to linear interpolation")
                    df[col] = df[col].interpolate(
                        method="linear", limit_direction="both"
                    )

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
                        df[col] = butter_filter(
                            df[col].values, fs=params["fs"], cutoff=params["cutoff"]
                        )
                        print(
                            f"Applied Butterworth filter with cutoff={params['cutoff']}Hz, fs={params['fs']}Hz"
                        )

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


def run_fill_split_dialog():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    root = tk.Tk()
    root.withdraw()
    
    # Open configuration dialog
    config_dialog = InterpolationConfigDialog(root)
    if not hasattr(config_dialog, "result") or config_dialog.result is None:
        print("Operation canceled by user.")
        return
        
    config = config_dialog.result
    
    # Select source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
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
                print(f"\nDEBUG - LOWESS Configuration:")
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
