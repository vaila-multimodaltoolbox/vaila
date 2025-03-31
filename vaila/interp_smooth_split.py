"""
===============================================================================
interpolation_split.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 14 October 2024
Update Date: 31 March 2025
Version: 0.0.1
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
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
from tkinter import filedialog, messagebox, Toplevel, Button, Label
import tkinter as tk
from rich import print


class InterpolationConfigDialog(tk.simpledialog.Dialog):
    def __init__(self, parent):
        # Initialize StringVar variables with default values before calling parent constructor
        self.savgol_window = tk.StringVar(value="7")  # Default window length
        self.savgol_poly = tk.StringVar(value="3")    # Default polynomial order
        self.lowess_frac = tk.StringVar(value="0.3")  # Default fraction
        self.lowess_it = tk.StringVar(value="3")      # Default iterations
        self.butter_cutoff = tk.StringVar(value="10")  # Default cutoff frequency
        self.butter_fs = tk.StringVar(value="100")     # Default sampling frequency
        self.kalman_iterations = tk.StringVar(value="5")  # Default Kalman iterations
        
        # Call parent constructor after initializing variables
        super().__init__(parent, title="Interpolation Configuration")

    def body(self, master):
        # Create main frame with scrollbar
        main_container = tk.Frame(master)
        main_container.pack(fill="both", expand=True)
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(main_container, width=800, height=600)
        scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
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
        interp_frame = tk.LabelFrame(left_column, text="Gap Filling Method", padx=5, pady=5)
        interp_frame.pack(fill="x", pady=5, anchor="n")
        
        # List of interpolation methods
        interp_text = """Gap Filling Methods:
1 - Linear Interpolation (simple, works well for most cases)
2 - Cubic Spline (smooth transitions between points)
3 - Nearest Value (use closest available value)
4 - Kalman Filter (good for movement data, models physics)
5 - None (leave gaps as NaN)
6 - Skip (keep original data, apply only smoothing)"""
        
        tk.Label(interp_frame, text=interp_text, justify="left").pack(anchor="w", padx=5)
        
        tk.Label(interp_frame, text="Enter gap filling method (1-6):").pack(anchor="w", padx=5, pady=5)
        self.interp_entry = tk.Entry(interp_frame)
        self.interp_entry.insert(0, "1")  # Default: linear
        self.interp_entry.pack(fill="x", padx=5)
        
        # Frame for smoothing method
        smooth_frame = tk.LabelFrame(left_column, text="Smoothing Method", padx=5, pady=5)
        smooth_frame.pack(fill="x", pady=5, anchor="n")
        
        # List of smoothing methods
        smooth_text = """Smoothing Methods:
1 - None (no smoothing)
2 - Savitzky-Golay Filter (preserves peaks and valleys)
3 - LOWESS (adapts to local trends)
4 - Kalman Filter (state estimation with noise reduction)
5 - Butterworth Filter (4th order, frequency domain filtering)"""
        
        tk.Label(smooth_frame, text=smooth_text, justify="left").pack(anchor="w", padx=5)
        
        # Important explanatory note
        tk.Label(smooth_frame, text="Note: Smoothing is applied to the entire data after filling gaps",
                foreground="blue", justify="left").pack(anchor="w", padx=5)
        
        tk.Label(smooth_frame, text="Enter smoothing method (1-5):").pack(anchor="w", padx=5, pady=5)
        self.smooth_entry = tk.Entry(smooth_frame)
        self.smooth_entry.insert(0, "1")  # Default: no smoothing
        self.smooth_entry.pack(fill="x", padx=5)
        
        # Add button to update parameters based on selection
        update_button = tk.Button(smooth_frame, text="Update Parameters", 
                                 command=self.update_params_frame)
        update_button.pack(pady=5)

        # Frame for split option
        split_frame = tk.LabelFrame(left_column, text="Split Configuration", padx=5, pady=5)
        split_frame.pack(fill="x", pady=5, anchor="n")
        
        self.split_var = tk.BooleanVar(value=False)
        tk.Checkbutton(split_frame, text="Split data into two parts", 
                      variable=self.split_var).pack(anchor="w")
        
        # ====== RIGHT COLUMN - PARAMETERS ======
        
        # Frame for specific method parameters
        self.params_frame = tk.LabelFrame(right_column, text="Method Parameters", padx=5, pady=5)
        self.params_frame.pack(fill="x", pady=5, anchor="n")
        
        # Create empty widgets for parameters
        self.params_widgets = []
        self.param_entries = {}  # Dictionary to keep track of parameter entries
        
        # Adicionar botão de confirmação de parâmetros
        self.confirm_button = tk.Button(
            right_column,
            text="Confirm Parameters",
            command=self.confirm_parameters,
            bg='lightgreen',
            font=('Arial', 10, 'bold')
        )
        self.confirm_button.pack(pady=10)
        
        # Frame for padding
        padding_frame = tk.LabelFrame(right_column, text="Padding Configuration", padx=5, pady=5)
        padding_frame.pack(fill="x", pady=5, anchor="n")
        
        tk.Label(padding_frame, text="Padding length (% of data):").pack(anchor="w")
        self.padding_entry = tk.Entry(padding_frame)
        self.padding_entry.insert(0, "10")  # Default 10%
        self.padding_entry.pack(fill="x", padx=5)
        
        # Frame for gap configuration
        gap_frame = tk.LabelFrame(right_column, text="Gap Configuration", padx=5, pady=5)
        gap_frame.pack(fill="x", pady=5, anchor="n")
        
        tk.Label(gap_frame, text="Maximum gap size to fill (frames):").pack(anchor="w")
        self.max_gap_entry = tk.Entry(gap_frame)
        self.max_gap_entry.insert(0, "30")  # Default 30 frames
        self.max_gap_entry.pack(fill="x", padx=5)
        
        # Explanatory label
        tk.Label(gap_frame, 
                text="Note: Gaps larger than this value will be left as NaN. Set to 0 to fill all gaps.", 
                foreground="blue", justify="left", wraplength=350).pack(anchor="w", padx=5, pady=2)
        
        # Initialize the parameters frame
        self.update_params_frame()
        
        # Bind the mouse wheel to the canvas for scrolling
        self.bind_mousewheel(canvas)
        
        return self.interp_entry  # Initial focus
    
    def bind_mousewheel(self, canvas):
        """Bind mouse wheel to scrolling canvas"""
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
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
                header = tk.Label(self.params_frame, text="Savitzky-Golay Parameters:", font=("", 10, "bold"))
                header.pack(anchor="w", padx=5, pady=5)
                
                label1 = tk.Label(self.params_frame, text="Window length (must be odd):", foreground="red")
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(self.params_frame, textvariable=self.savgol_window)
                entry1.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry1.bind('<Return>', lambda e: self.update_parameter_value(e, self.savgol_window))
                
                label2 = tk.Label(self.params_frame, text="Polynomial order:", foreground="red")
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.savgol_poly)
                entry2.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry2.bind('<Return>', lambda e: self.update_parameter_value(e, self.savgol_poly))
                
                self.params_widgets.extend([header, label1, entry1, label2, entry2])
                
                # Guardar referências aos entries para acesso posterior
                self.param_entries['window_length'] = entry1
                self.param_entries['polyorder'] = entry2
                
            elif smooth_method == 3:  # LOWESS
                # Add section header with special emphasis
                header = tk.Label(self.params_frame, text="LOWESS Parameters:", font=("", 10, "bold"))
                header.pack(anchor="w", padx=5, pady=5)
                
                label1 = tk.Label(self.params_frame, text="Fraction (0-1):", foreground="red")
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(self.params_frame, textvariable=self.lowess_frac)
                entry1.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry1.bind('<Return>', lambda e: self.update_parameter_value(e, self.lowess_frac))
                
                label2 = tk.Label(self.params_frame, text="Number of iterations:", foreground="red")
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.lowess_it)
                entry2.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry2.bind('<Return>', lambda e: self.update_parameter_value(e, self.lowess_it))
                
                self.params_widgets.extend([header, label1, entry1, label2, entry2])
                
                # Guardar referências aos entries para acesso posterior
                self.param_entries['frac'] = entry1
                self.param_entries['it'] = entry2
                
            elif smooth_method == 4:  # Kalman
                # Add section header with special emphasis
                header = tk.Label(self.params_frame, text="Kalman Filter Parameters:", font=("", 10, "bold"))
                header.pack(anchor="w", padx=5, pady=5)
                
                label = tk.Label(self.params_frame, text="Number of EM iterations:", foreground="red")
                label.pack(anchor="w", padx=5, pady=2)
                entry = tk.Entry(self.params_frame, textvariable=self.kalman_iterations)
                entry.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry.bind('<Return>', lambda e: self.update_parameter_value(e, self.kalman_iterations))
                
                self.params_widgets.extend([header, label, entry])
                
                # Guardar referência ao entry para acesso posterior
                self.param_entries['n_iter'] = entry
                
            elif smooth_method == 5:  # Butterworth
                # Add section header with special emphasis
                header = tk.Label(self.params_frame, text="Butterworth Filter Parameters:", font=("", 10, "bold"))
                header.pack(anchor="w", padx=5, pady=5)
                
                label1 = tk.Label(self.params_frame, text="Cutoff frequency (Hz, e.g. 4):", foreground="red")
                label1.pack(anchor="w", padx=5, pady=2)
                entry1 = tk.Entry(self.params_frame, textvariable=self.butter_cutoff)
                entry1.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry1.bind('<Return>', lambda e: self.update_parameter_value(e, self.butter_cutoff))
                
                label2 = tk.Label(self.params_frame, text="Sampling frequency (Hz, e.g. 50):", foreground="red")
                label2.pack(anchor="w", padx=5, pady=2)
                entry2 = tk.Entry(self.params_frame, textvariable=self.butter_fs)
                entry2.pack(fill="x", padx=5, pady=2)
                # Adicionar binding para Enter
                entry2.bind('<Return>', lambda e: self.update_parameter_value(e, self.butter_fs))
                
                # Strong reminder for Butterworth
                reminder = tk.Label(self.params_frame, 
                                  text="* You MUST enter both cutoff and sampling frequency values.\nNo default values will be used.",
                                  foreground="red", font=("", 9, "bold"), justify="left")
                reminder.pack(anchor="w", padx=5, pady=5)
                
                self.params_widgets.extend([header, label1, entry1, label2, entry2, reminder])
                
                # Guardar referências aos entries para acesso posterior
                self.param_entries['cutoff'] = entry1
                self.param_entries['fs'] = entry2
                
        except ValueError:
            label = tk.Label(self.params_frame, text="Please enter valid method numbers")
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
            
            if not (1 <= interp_num <= 6):
                messagebox.showerror("Error", "Gap filling method must be between 1 and 6")
                return False
            
            if not (1 <= smooth_num <= 5):
                messagebox.showerror("Error", "Smoothing method must be between 1 and 5")
                return False
            
            # Validate parameters specifically
            if smooth_num == 2:  # Savitzky-Golay
                if not self.savgol_window.get() or not self.savgol_poly.get():
                    messagebox.showerror("Error", "Savitzky-Golay parameters are required")
                    return False
                
                window = int(self.savgol_window.get())
                poly = int(self.savgol_poly.get())
                if window % 2 == 0:
                    messagebox.showerror("Error", "Window length must be an odd number")
                    return False
                if poly >= window:
                    messagebox.showerror("Error", "Polynomial order must be less than window length")
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
                    messagebox.showerror("Error", "Kalman filter iterations are required")
                    return False
                
                n_iter = int(self.kalman_iterations.get())
                if n_iter <= 0:
                    messagebox.showerror("Error", "Number of iterations must be positive")
                    return False
            
            elif smooth_num == 5:  # Butterworth
                if not self.butter_cutoff.get() or not self.butter_fs.get():
                    messagebox.showerror("Error", "Butterworth filter requires both cutoff and sampling frequencies")
                    return False
                
                cutoff = float(self.butter_cutoff.get())
                fs = float(self.butter_fs.get())
                if cutoff <= 0 or fs <= 0:
                    messagebox.showerror("Error", "Frequencies must be positive")
                    return False
                if cutoff >= fs/2:
                    messagebox.showerror("Error", "Cutoff frequency must be less than half of sampling frequency (Nyquist frequency)")
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
                
            return True
            
        except ValueError as e:
            messagebox.showerror("Error", f"Please enter valid numeric values: {str(e)}")
            return False
    
    def confirm_parameters(self):
        """Confirma e atualiza os parâmetros antes do processamento"""
        try:
            # Força a perda de foco de todos os widgets de entrada
            self.focus()
            
            # Força atualização explícita dos valores dos Entry widgets
            if 'cutoff' in self.param_entries:
                self.butter_cutoff.set(self.param_entries['cutoff'].get())
            if 'fs' in self.param_entries:
                self.butter_fs.set(self.param_entries['fs'].get())
            if 'window_length' in self.param_entries:
                self.savgol_window.set(self.param_entries['window_length'].get())
            if 'polyorder' in self.param_entries:
                self.savgol_poly.set(self.param_entries['polyorder'].get())
            
            # Força a atualização dos widgets
            self.update_idletasks()
            
            # Captura o método de suavização atual
            smooth_method = int(self.smooth_entry.get())
            
            # Print dos parâmetros confirmados no terminal
            print("\n" + "="*50)
            print("CONFIRMED PARAMETERS:")
            print("="*50)
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
                params_text = f"EM Iterations: {n_iter}"
                print("\nKalman Filter Parameters:")
                print(f"- EM Iterations: {n_iter}")
            elif smooth_method == 5:  # Butterworth
                cutoff = float(self.butter_cutoff.get())
                fs = float(self.butter_fs.get())
                params_text = f"Cutoff: {cutoff} Hz, Sampling Frequency: {fs} Hz"
                print("\nButterworth Filter Parameters:")
                print(f"- Cutoff Frequency: {cutoff} Hz")
                print(f"- Sampling Frequency: {fs} Hz")
                
                # Validação adicional para Butterworth
                if cutoff >= fs/2:
                    raise ValueError("Cutoff frequency must be less than half of sampling frequency (Nyquist frequency)")
            else:
                params_text = "Unknown smoothing method"
            
            print("="*50)
            
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
            self.confirm_button.configure(bg='pale green', text="Parameters Confirmed ✓")
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter value: {str(e)}")
    
    def apply(self):
        try:
            # Verifica se os parâmetros foram confirmados
            if self.confirm_button['text'] != "Parameters Confirmed ✓":
                if not messagebox.askyesno("Warning", 
                    "Parameters have not been confirmed. Do you want to proceed anyway?"):
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
                1: 'linear',
                2: 'cubic',
                3: 'nearest',
                4: 'kalman',
                5: 'none',
                6: 'skip'
            }
            
            smooth_map = {
                1: 'none',
                2: 'savgol',
                3: 'lowess',
                4: 'kalman',
                5: 'butterworth'
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
                smooth_params = {
                    'window_length': window_length,
                    'polyorder': polyorder
                }
                print(f"APPLY: Savitzky-Golay settings - window={window_length}, polyorder={polyorder}")
                
            elif smooth_method == 3:  # LOWESS
                frac = float(self.lowess_frac.get())
                it = int(self.lowess_it.get())
                smooth_params = {
                    'frac': frac,
                    'it': it
                }
                print(f"APPLY: LOWESS settings - frac={frac}, it={it}")
                
            elif smooth_method == 4:  # Kalman
                n_iter = int(self.kalman_iterations.get())
                smooth_params = {
                    'n_iter': n_iter
                }
                print(f"APPLY: Kalman settings - n_iter={n_iter}")
                
            elif smooth_method == 5:  # Butterworth
                cutoff = float(self.butter_cutoff.get())
                fs = float(self.butter_fs.get())
                smooth_params = {
                    'cutoff': cutoff,
                    'fs': fs
                }
                print(f"APPLY: Butterworth settings - cutoff={cutoff} Hz, fs={fs} Hz")
            
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
                summary += f"\nSavitzky-Golay Parameters:\n- Window Length: {window_length}\n- Polynomial Order: {polyorder}"
            elif smooth_method == 3:  # LOWESS
                summary += f"\nLOWESS Parameters:\n- Fraction: {frac}\n- Iterations: {it}"
            elif smooth_method == 4:  # Kalman
                summary += f"\nKalman Parameters:\n- EM Iterations: {n_iter}"
            elif smooth_method == 5:  # Butterworth
                summary += f"\nButterworth Parameters:\n- Cutoff Frequency: {cutoff} Hz\n- Sampling Frequency: {fs} Hz"
            
            if messagebox.askokcancel("Confirm Parameters", summary):
                config_result = {
                    'padding': padding,
                    'interp_method': interp_map[interp_method],
                    'smooth_method': smooth_map[smooth_method],
                    'smooth_params': smooth_params,
                    'max_gap': max_gap,
                    'do_split': do_split
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
        self.interp_entry.bind('<FocusOut>', self.update_value)
        self.smooth_entry.bind('<FocusOut>', self.update_value)
        self.padding_entry.bind('<FocusOut>', self.update_value)
        self.max_gap_entry.bind('<FocusOut>', self.update_value)


def generate_report(dest_dir, config, processed_files):
    """
    Generates a detailed processing report and saves it to a text file.
    
    Args:
        dest_dir: Directory where the processed files were saved
        config: Configuration settings used in processing
        processed_files: List of dictionaries with information about processed files
    """
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    report_path = os.path.join(dest_dir, "processing_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"PROCESSING REPORT - VAILA INTERPOLATION AND SMOOTHING TOOL\n")
        f.write(f"Date and Time: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # General configuration
        f.write("GENERAL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Gap Filling Method: {config['interp_method']}\n")
        
        if config['interp_method'] not in ['none', 'skip']:
            f.write(f"Maximum Gap Size to Fill: {config['max_gap']} frames")
            if config['max_gap'] == 0:
                f.write(" (no limit - all gaps filled)\n")
            else:
                f.write("\n")
        
        f.write(f"Smoothing Method: {config['smooth_method']}\n")
        f.write(f"Padding: {config['padding']}%\n")
        f.write(f"Split Data: {'Yes' if config['do_split'] else 'No'}\n\n")
        
        # Specific parameters
        if config['smooth_method'] != 'none':
            f.write("SMOOTHING PARAMETERS\n")
            f.write("-" * 80 + "\n")
            
            if config['smooth_method'] == 'savgol':
                f.write(f"Window Length: {config['smooth_params'].get('window_length', 7)}\n")
                f.write(f"Polynomial Order: {config['smooth_params'].get('polyorder', 2)}\n")
            
            elif config['smooth_method'] == 'lowess':
                f.write(f"Fraction: {config['smooth_params'].get('frac', 0.3)}\n")
                f.write(f"Iterations: {config['smooth_params'].get('it', 3)}\n")
            
            elif config['smooth_method'] == 'kalman':
                f.write(f"EM Iterations: {config['smooth_params'].get('n_iter', 5)}\n")
            
            elif config['smooth_method'] == 'butterworth':
                f.write(f"Cutoff Frequency: {config['smooth_params'].get('cutoff', 10)} Hz\n")
                f.write(f"Sampling Frequency: {config['smooth_params'].get('fs', 100)} Hz\n")
            
            f.write("\n")
        
        # Processed files
        f.write("PROCESSED FILES\n")
        f.write("-" * 80 + "\n")
        
        for idx, file_info in enumerate(processed_files, 1):
            f.write(f"File {idx}: {file_info['original_filename']}\n")
            f.write(f"  - Original Path: {file_info['original_path']}\n")
            f.write(f"  - Original Size: {file_info['original_size']} frames, {file_info['original_columns']} columns\n")
            f.write(f"  - Total Missing Values: {file_info['total_missing']}\n")
            f.write(f"  - Processed Output: {file_info['output_path']}\n")
            
            # If split, show both parts
            if config['do_split'] and 'output_part2_path' in file_info:
                f.write(f"  - Split Part 1: {file_info['output_part1_path']} ({file_info['part1_size']} frames)\n")
                f.write(f"  - Split Part 2: {file_info['output_part2_path']} ({file_info['part2_size']} frames)\n")
            
            # Details of columns with interpolated values
            if file_info['columns_with_missing']:
                f.write("  - Columns with missing values:\n")
                for col_name, missing_count in file_info['columns_with_missing'].items():
                    f.write(f"    - {col_name}: {missing_count} missing values\n")
            
            # Additional information if applicable
            if file_info.get('warnings'):
                f.write("  - Warnings during processing:\n")
                for warning in file_info['warnings']:
                    f.write(f"    - {warning}\n")
            
            f.write("\n")
        
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
        f.write("- Processed files include the interpolation and smoothing method in their filenames.\n")
        f.write("- Files are saved with the same number of decimal places as the original files.\n")
        f.write("- The output directory includes the processing methods in its name for easy reference.\n")
        f.write("- For questions or issues, please contact: paulosantiago@usp.br\n")
    
    print(f"Generated detailed processing report: {report_path}")
    return report_path


def process_file(file_path, dest_dir, config):
    try:
        # Initialize dictionary with file information
        filename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(filename)[0]}_processed.csv"
        output_path = os.path.join(dest_dir, output_filename)
        
        file_info = {
            'original_path': file_path,
            'original_filename': filename,
            'output_path': output_path,
            'warnings': []
        }
        
        # Debug: print configuration parameters
        print("\n" + "="*80)
        print("DEBUG - PROCESSING PARAMETERS:")
        print(f"Interpolation Method: {config['interp_method']}")
        print(f"Maximum Gap Size: {config['max_gap']} frames")
        print(f"Smoothing Method: {config['smooth_method']}")
        
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        
        # Record original size
        file_info['original_size'] = len(df)
        file_info['original_columns'] = len(df.columns)
        
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
        df = pd.merge(all_frames, df, on=first_col, how='left')
        print(f"Shape after adding missing frames: {df.shape}")
        
        # Count missing values
        file_info['total_missing'] = df.isna().sum().sum()
        file_info['columns_with_missing'] = {}
        
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                file_info['columns_with_missing'][col] = missing
        
        # Apply padding if necessary 
        # Get exact padding value from config
        padding_percent = config['padding']
        print(f"Using exact padding value: {padding_percent}%")
        
        pad_len = 0
        if padding_percent > 0:
            pad_len = int(len(df) * padding_percent / 100)
            print(f"Applying padding of {pad_len} frames")
            
            # Create frames for padding
            pad_before = pd.DataFrame({first_col: range(min_frame - pad_len, min_frame)})
            pad_after = pd.DataFrame({first_col: range(max_frame + 1, max_frame + pad_len + 1)})
            
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

        for col in numeric_cols:
            print(f"\nProcessing column: {col}")
            nan_mask = df[col].isna()
            print(f"Found {nan_mask.sum()} NaN values in column {col}")
            
            # STEP 1: Gap Filling
            if nan_mask.any() and config['interp_method'] not in ['none', 'skip']:
                print(f"STEP 1: Filling gaps with {config['interp_method']} method")
                
                # Check maximum gap size
                max_gap = config['max_gap']
                print(f"Using exact maximum gap size: {max_gap} frames")
                
                if max_gap > 0:
                    print(f"Limiting gap filling to gaps of {max_gap} frames or less")
                    
                    # Identificar grupos de NaNs consecutivos para todos os métodos
                    nan_groups = []
                    start_idx = None
                    original_nan_mask = df[col].isna()
                    
                    for i in range(len(original_nan_mask)):
                        if original_nan_mask.iloc[i]:
                            if start_idx is None:
                                start_idx = i
                        elif start_idx is not None:
                            nan_groups.append((start_idx, i-1))
                            start_idx = None
                    
                    # Não esquecer do último grupo se terminar com NaN
                    if start_idx is not None:
                        nan_groups.append((start_idx, len(original_nan_mask)-1))
                    
                    # Aplicar o método de interpolação escolhido
                    if config['interp_method'] == 'linear':
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    
                    elif config['interp_method'] == 'nearest':
                        df[col] = df[col].interpolate(method='nearest', limit_direction='both')
                    
                    elif config['interp_method'] == 'cubic':
                        try:
                            valid_mask = ~nan_mask
                            if valid_mask.sum() >= 3:
                                x_valid = df[first_col][valid_mask].values
                                y_valid = df[col][valid_mask].values
                                
                                sort_idx = np.argsort(x_valid)
                                x_valid = x_valid[sort_idx]
                                y_valid = y_valid[sort_idx]
                                
                                cs = CubicSpline(x_valid, y_valid, bc_type='natural')
                                x_nan = df[first_col][nan_mask].values
                                df.loc[nan_mask, col] = cs(x_nan)
                            else:
                                print(f"Not enough points for cubic spline, falling back to linear")
                                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                        except Exception as e:
                            print(f"Error applying Cubic Spline: {str(e)}")
                            print(f"Falling back to linear interpolation")
                            df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    
                    elif config['interp_method'] == 'kalman':
                        try:
                            temp_series = df[col].copy()
                            temp_filled = temp_series.interpolate(method='linear', limit_direction='both')
                            
                            kf = KalmanFilter(
                                initial_state_mean=np.mean(temp_filled.dropna()),
                                n_dim_obs=1,
                                n_dim_state=2
                            )
                            
                            observations = np.array([
                                [x] if not np.isnan(x) else None 
                                for x in df[col].values
                            ])
                            
                            n_iter = 5  # Fallback value
                            if 'n_iter' in config['smooth_params']:
                                n_iter = config['smooth_params']['n_iter']
                            
                            smoothed_state_means, _ = kf.em(observations, n_iter=n_iter).smooth(observations)
                            df.loc[nan_mask, col] = smoothed_state_means[nan_mask, 0]
                        except Exception as e:
                            print(f"Error applying Kalman for gap filling: {str(e)}")
                            print(f"Falling back to linear interpolation")
                            df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    
                    # Reverter interpolação para gaps maiores que max_gap para TODOS os métodos
                    for start, end in nan_groups:
                        gap_size = end - start + 1
                        if gap_size > max_gap:
                            print(f"Reverting interpolation for gap of size {gap_size} at frames {start}-{end}")
                            df.loc[df.index[start:end+1], col] = np.nan
                    
                    # Verificar o resultado após a interpolação
                remaining_nans = df[col].isna().sum()
                print(f"After gap filling with max_gap={max_gap}, {remaining_nans} NaN values remain")
            
            else:  # No gap size limit
                    # Aplicar interpolação sem limite de gap
                if config['interp_method'] == 'linear':
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                elif config['interp_method'] == 'nearest':
                    df[col] = df[col].interpolate(method='nearest', limit_direction='both')
                elif config['interp_method'] == 'cubic':
                    try:
                        valid_mask = ~nan_mask
                        if valid_mask.sum() >= 3:
                            x_valid = df[first_col][valid_mask].values
                            y_valid = df[col][valid_mask].values
                            
                            sort_idx = np.argsort(x_valid)
                            x_valid = x_valid[sort_idx]
                            y_valid = y_valid[sort_idx]
                            
                            cs = CubicSpline(x_valid, y_valid, bc_type='natural')
                            x_nan = df[first_col][nan_mask].values
                            df.loc[nan_mask, col] = cs(x_nan)
                        else:
                            print(f"Not enough points for cubic spline, falling back to linear")
                            df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    except Exception as e:
                        print(f"Error applying Cubic Spline: {str(e)}")
                        print(f"Falling back to linear interpolation")
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                elif config['interp_method'] == 'kalman':
                    try:
                        temp_series = df[col].copy()
                        temp_filled = temp_series.interpolate(method='linear', limit_direction='both')
                        
                        kf = KalmanFilter(
                            initial_state_mean=np.mean(temp_filled.dropna()),
                            n_dim_obs=1,
                            n_dim_state=2
                        )
                        
                        observations = np.array([
                            [x] if not np.isnan(x) else None 
                            for x in df[col].values
                        ])
                        
                        print(f"Falling back to linear interpolation")
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
                    except Exception as e:
                        print(f"Error applying Kalman for gap filling: {str(e)}")
                        print(f"Falling back to linear interpolation")
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # Remover o padding: manter apenas os frames originais
        print(f"\nRemoving padding (keeping only frames from {min_frame} to {max_frame})")
        df = df[df[first_col].between(min_frame, max_frame)].reset_index(drop=True)
        print(f"Final shape after removing padding: {df.shape}")
        
        # Salvar o DataFrame processado
        print(f"\nSaving processed file to: {output_path}")
        df.to_csv(output_path, index=False)
        print(f"File saved successfully!")
        
        return file_info
        
    except Exception as e:
        # Return basic info with error in case of failure
        filename = os.path.basename(file_path)
        output_filename = f"{os.path.splitext(filename)[0]}_processed.csv"
        output_path = os.path.join(dest_dir, output_filename)
        
        return {
            'original_path': file_path,
            'original_filename': filename,
            'output_path': output_path,
            'warnings': [f"Error processing file: {str(e)}"],
            'error': True,
            'original_size': 0,
            'original_columns': 0,
            'total_missing': 0,
            'columns_with_missing': {}
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
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    # Create descriptive names for methods
    interp_name = config['interp_method']
    
    # Add information about smoothing parameters if used
    smooth_info = "no_smooth"
    if config['smooth_method'] != 'none':
        smooth_info = config['smooth_method']
        # Add main parameters in summary form
        if config['smooth_method'] == 'butterworth':
            cutoff = config['smooth_params'].get('cutoff', 10)
            smooth_info += f"_cut{cutoff}"
        elif config['smooth_method'] == 'savgol':
            window = config['smooth_params'].get('window_length', 7)
            poly = config['smooth_params'].get('polyorder', 2)
            smooth_info += f"_w{window}p{poly}"
        elif config['smooth_method'] == 'lowess':
            frac = config['smooth_params'].get('frac', 0.3)
            smooth_info += f"_frac{int(frac*100)}"
    
    # Directory with informative name
    dest_dir_name = f'processed_{interp_name}_{smooth_info}_{timestamp}'
    dest_dir = os.path.join(source_dir, dest_dir_name)
    os.makedirs(dest_dir, exist_ok=True)

    # List to store information about processed files
    processed_files = []
    
    # Process each file
    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            try:
                file_info = process_file(
                    os.path.join(source_dir, filename),
                    dest_dir,
                    config
                )
                if file_info is not None:
                    processed_files.append(file_info)
                else:
                    print(f"Warning: No information returned for file {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                # Adiciona informação básica do arquivo com erro
                processed_files.append({
                    'original_path': os.path.join(source_dir, filename),
                    'original_filename': filename,
                    'warnings': [f"Error processing file: {str(e)}"],
                    'error': True,
                    'original_size': 0,
                    'original_columns': 0,
                    'total_missing': 0,
                    'columns_with_missing': {},
                    'output_path': None
                })
    
    # Filtra arquivos processados para remover None
    processed_files = [pf for pf in processed_files if pf is not None]
    
    # Generate detailed processing report
    if processed_files:
        report_path = generate_report(dest_dir, config, processed_files)
        messagebox.showinfo("Complete", 
                           f"Processing complete. Results saved in {dest_dir}\n"
                           f"A detailed processing report has been saved to:\n{report_path}")
    else:
        messagebox.showwarning("Warning", 
                             "No files were successfully processed.")


if __name__ == "__main__":
    run_fill_split_dialog()
