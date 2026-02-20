class InterpolationConfigDialog:
    """Modern UI Configuration dialog for interpolation, smoothing, and splitting using CustomTkinter."""

    def __init__(self, parent=None):
        import customtkinter as ctk
        self.result = None
        
        if parent is None:
            self.root = ctk.CTk()
            self.root.title("Vaila - Data Processing")
            self.window = self.root
        else:
            self.root = parent
            self.window = ctk.CTkToplevel(parent)
            self.window.title("Vaila - Data Processing")
            self.window.transient(parent)
            self.window.grab_set()

        self.window.geometry("1400x900")
        self.window.minsize(1200, 800)
        
        # Appearance
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # Setup Variables (Internal State)
        self.setup_variables()
        self.create_dialog_content()
        self.center_window()
        self.window.protocol("WM_DELETE_WINDOW", self.cancel)

        if parent is None:
            self.window.focus_force()

    def setup_variables(self):
        import customtkinter as ctk
        self.savgol_window = ctk.StringVar(value="7")
        self.savgol_poly = ctk.StringVar(value="3")
        self.lowess_frac = ctk.StringVar(value="0.3")
        self.lowess_it = ctk.StringVar(value="3")
        self.butter_cutoff = ctk.StringVar(value="10.0")
        self.butter_fs = ctk.StringVar(value="100.0")
        self.kalman_iterations = ctk.StringVar(value="5")
        self.kalman_mode = ctk.StringVar(value="1")
        self.spline_smoothing = ctk.StringVar(value="1.0")
        self.arima_p = ctk.StringVar(value="1")
        self.arima_d = ctk.StringVar(value="0")
        self.arima_q = ctk.StringVar(value="0")
        self.median_kernel = ctk.StringVar(value="5")
        self.hampel_window = ctk.StringVar(value="7")
        self.hampel_sigma = ctk.StringVar(value="3.0")
        self.sample_rate = ctk.StringVar(value="")
        
        self.split_var = ctk.BooleanVar(value=False)
        self.interp_method_var = ctk.StringVar(value="1") # 1=linear
        self.smooth_method_var = ctk.StringVar(value="1") # 1=none
        self.padding_var = ctk.StringVar(value="10")
        self.max_gap_var = ctk.StringVar(value="60")
        
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
        import customtkinter as ctk
        self.header_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        self.title_label = ctk.CTkLabel(self.header_frame, text="Interpolation & Smoothing Config", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack()
        self.subtitle_label = ctk.CTkLabel(self.header_frame, text="Configure gap filling, smoothing and validate your data.", font=ctk.CTkFont(size=14), text_color="gray")
        self.subtitle_label.pack()

        self.main_split = ctk.CTkFrame(self.window, fg_color="transparent")
        self.main_split.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.config_panel = ctk.CTkFrame(self.main_split, width=400)
        self.config_panel.pack(side="left", fill="y", padx=(0, 10))
        
        self.analysis_panel = ctk.CTkFrame(self.main_split)
        self.analysis_panel.pack(side="right", fill="both", expand=True)

        self.tabview = ctk.CTkTabview(self.config_panel, width=380)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.tab_interp = self.tabview.add("Gap Filling")
        self.tab_smooth = self.tabview.add("Smoothing")
        self.tab_general = self.tabview.add("General")
        
        self.build_interp_tab()
        self.build_smooth_tab()
        self.build_general_tab()
        
        self.buttons_frame = ctk.CTkFrame(self.config_panel, fg_color="transparent")
        self.buttons_frame.pack(fill="x", padx=10, pady=10)
        
        self.save_btn = ctk.CTkButton(self.buttons_frame, text="💾 Save Template", command=self.create_toml_template, width=140)
        self.save_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.load_btn = ctk.CTkButton(self.buttons_frame, text="📂 Load TOML", command=self.load_toml_config, width=140)
        self.load_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.apply_btn = ctk.CTkButton(self.buttons_frame, text="✅ Apply & Run Batch", command=self.ok, fg_color="#2E7D32", hover_color="#1B5E20")
        self.apply_btn.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(15, 5))
        
        self.build_analysis_panel()

    def build_interp_tab(self):
        import customtkinter as ctk
        lbl = ctk.CTkLabel(self.tab_interp, text="Select Interpolation Method:", font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor="w", pady=(10, 5))
        
        methods = {
            "1": "Linear (Straight lines)",
            "2": "Cubic (Smooth curves)",
            "3": "Nearest (Copy nearest)",
            "4": "Kalman (Predictive)",
            "5": "None (Leave gaps)",
            "6": "Skip (Only smooth)",
        }
        
        self.interp_combo = ctk.CTkOptionMenu(self.tab_interp, values=list(methods.values()), command=self.on_interp_change)
        self.interp_combo.pack(fill="x", pady=5)
        self.interp_combo.set("Linear (Straight lines)")
        
        lbl_gap = ctk.CTkLabel(self.tab_interp, text="Max Gap Size (frames):", font=ctk.CTkFont(weight="bold"))
        lbl_gap.pack(anchor="w", pady=(15, 5))
        
        self.gap_entry = ctk.CTkEntry(self.tab_interp, textvariable=self.max_gap_var)
        self.gap_entry.pack(fill="x", pady=5)
        
        tip = ctk.CTkLabel(self.tab_interp, text="0 = Fill all gaps\n60 = Up to 2 seconds at 30 fps", font=ctk.CTkFont(size=11), text_color="gray", justify="left")
        tip.pack(anchor="w")

    def on_interp_change(self, choice):
        mapping = {
            "Linear (Straight lines)": "1",
            "Cubic (Smooth curves)": "2",
            "Nearest (Copy nearest)": "3",
            "Kalman (Predictive)": "4",
            "None (Leave gaps)": "5",
            "Skip (Only smooth)": "6",
        }
        self.interp_method_var.set(mapping[choice])

    def build_smooth_tab(self):
        import customtkinter as ctk
        lbl = ctk.CTkLabel(self.tab_smooth, text="Select Smoothing Method:", font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor="w", pady=(10, 5))
        
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
        
        self.smooth_combo = ctk.CTkOptionMenu(self.tab_smooth, values=list(methods.values()), command=self.on_smooth_change)
        self.smooth_combo.pack(fill="x", pady=5)
        
        self.dynamic_params_container = ctk.CTkFrame(self.tab_smooth, fg_color="transparent")
        self.dynamic_params_container.pack(fill="both", expand=True, pady=15)
        
        self.build_all_param_frames()
        self.smooth_combo.set("None")
        self.on_smooth_change("None")

    def on_smooth_change(self, choice):
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
        self.smooth_method_var.set(mapping[choice])
        
        for frame in self.param_frames.values():
            frame.pack_forget()
            
        method_id = mapping[choice]
        if method_id in self.param_frames:
            self.param_frames[method_id].pack(fill="both", expand=True)

    def build_all_param_frames(self):
        import customtkinter as ctk
        # 2: Savgol
        f2 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f2, text="Window Length (Odd):").pack(anchor="w")
        ctk.CTkEntry(f2, textvariable=self.savgol_window).pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(f2, text="Polynomial Order:").pack(anchor="w")
        ctk.CTkEntry(f2, textvariable=self.savgol_poly).pack(fill="x")
        self.param_frames["2"] = f2
        
        # 3: Lowess
        f3 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f3, text="Fraction (0.1 - 1.0):").pack(anchor="w")
        ctk.CTkEntry(f3, textvariable=self.lowess_frac).pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(f3, text="Iterations:").pack(anchor="w")
        ctk.CTkEntry(f3, textvariable=self.lowess_it).pack(fill="x")
        self.param_frames["3"] = f3
        
        # 4: Kalman
        f4 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f4, text="EM Iterations:").pack(anchor="w")
        ctk.CTkEntry(f4, textvariable=self.kalman_iterations).pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(f4, text="Mode (1=1D, 2=2D):").pack(anchor="w")
        ctk.CTkEntry(f4, textvariable=self.kalman_mode).pack(fill="x")
        self.param_frames["4"] = f4
        
        # 5: Butterworth
        f5 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f5, text="Cutoff Frequency (Hz):").pack(anchor="w")
        ctk.CTkEntry(f5, textvariable=self.butter_cutoff).pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(f5, text="Tip: 4-10 Hz for biomechanics", text_color="gray", font=("Arial", 10)).pack(anchor="w", pady=(0, 10))
        
        ctk.CTkLabel(f5, text="Sampling Freq (fs, Hz):").pack(anchor="w")
        ctk.CTkEntry(f5, textvariable=self.butter_fs).pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(f5, text="Tip: fps of the video or capture freq", text_color="gray", font=("Arial", 10)).pack(anchor="w")
        self.param_frames["5"] = f5
        
        # 6: Splines
        f6 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f6, text="Smoothing Factor:").pack(anchor="w")
        ctk.CTkEntry(f6, textvariable=self.spline_smoothing).pack(fill="x")
        self.param_frames["6"] = f6
        
        # 7: ARIMA
        f7 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f7, text="P (AR):").pack(anchor="w")
        ctk.CTkEntry(f7, textvariable=self.arima_p).pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(f7, text="D (Diff):").pack(anchor="w")
        ctk.CTkEntry(f7, textvariable=self.arima_d).pack(fill="x", pady=(0, 5))
        ctk.CTkLabel(f7, text="Q (MA):").pack(anchor="w")
        ctk.CTkEntry(f7, textvariable=self.arima_q).pack(fill="x")
        self.param_frames["7"] = f7
        
        # 8: Median
        f8 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f8, text="Kernel Size (Odd):").pack(anchor="w")
        ctk.CTkEntry(f8, textvariable=self.median_kernel).pack(fill="x")
        self.param_frames["8"] = f8
        
        # 9: Hampel
        f9 = ctk.CTkFrame(self.dynamic_params_container, fg_color="transparent")
        ctk.CTkLabel(f9, text="Window Size (Odd):").pack(anchor="w")
        ctk.CTkEntry(f9, textvariable=self.hampel_window).pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(f9, text="Sigma Multiplier (e.g. 3.0):").pack(anchor="w")
        ctk.CTkEntry(f9, textvariable=self.hampel_sigma).pack(fill="x")
        self.param_frames["9"] = f9

    def build_general_tab(self):
        import customtkinter as ctk
        lbl = ctk.CTkLabel(self.tab_general, text="Padding (%):", font=ctk.CTkFont(weight="bold"))
        lbl.pack(anchor="w", pady=(10, 5))
        ctk.CTkEntry(self.tab_general, textvariable=self.padding_var).pack(fill="x", pady=5)
        
        lbl_split = ctk.CTkLabel(self.tab_general, text="Split Dataset:", font=ctk.CTkFont(weight="bold"))
        lbl_split.pack(anchor="w", pady=(15, 5))
        ctk.CTkSwitch(self.tab_general, text="Enable Splitting", variable=self.split_var).pack(anchor="w", pady=5)
        
        lbl_rate = ctk.CTkLabel(self.tab_general, text="Sample Rate Override (optional):", font=ctk.CTkFont(weight="bold"))
        lbl_rate.pack(anchor="w", pady=(15, 5))
        ctk.CTkEntry(self.tab_general, textvariable=self.sample_rate).pack(fill="x", pady=5)

    def build_analysis_panel(self):
        import customtkinter as ctk
        self.analysis_top = ctk.CTkFrame(self.analysis_panel, fg_color="transparent")
        self.analysis_top.pack(fill="x", padx=10, pady=10)
        
        self.load_test_btn = ctk.CTkButton(self.analysis_top, text="📊 Load Test CSV", command=self.load_test_data)
        self.load_test_btn.pack(side="left", padx=5)
        
        self.test_label = ctk.CTkLabel(self.analysis_top, text="No test data loaded.")
        self.test_label.pack(side="left", padx=10)
        
        self.col_frame = ctk.CTkFrame(self.analysis_top, fg_color="transparent")
        self.col_frame.pack(side="right", padx=5)
        
        self.col_combo = None
        
        self.plot_frame = ctk.CTkFrame(self.analysis_panel)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def load_test_data(self):
        from tkinter import filedialog, messagebox
        import customtkinter as ctk
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
                    self.col_combo.destroy()
                    
                self.test_col_var = ctk.StringVar(value=numeric_cols[0])
                self.col_combo = ctk.CTkOptionMenu(self.col_frame, values=numeric_cols, variable=self.test_col_var, command=self.run_analysis)
                self.col_combo.pack(side="left", padx=5)
                
                btn = ctk.CTkButton(self.col_frame, text="▶ Run Check", command=lambda: self.run_analysis(self.test_col_var.get()), width=100)
                btn.pack(side="left", padx=5)
                
                # Check for Winter analysis
                win_btn = ctk.CTkButton(self.col_frame, text="❄ Winter Check", command=lambda: self.perform_winter_residual_analysis(self.test_col_var.get()), width=100, fg_color="#ea7317", hover_color="#d3610b")
                win_btn.pack(side="left", padx=5)

                self.run_analysis(numeric_cols[0])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load: {e}")

    def get_current_analysis_config(self):
        # In this CTk version, everything is always linked to variables seamlessly
        return self.get_config()

    def run_analysis(self, column_name):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import numpy as np

        if self.test_data is None:
            return
            
        for child in self.plot_frame.winfo_children():
            child.destroy()
            
        # Get config seamlessly from our variables, no state resets
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
        if getattr(self.plot_frame, "cget", lambda x: "Light")("fg_color") != "transparent":
            fig.patch.set_facecolor("#2b2b2b")
        else:
            fig.patch.set_facecolor("#ebebeb")

        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(frame_numbers, original_data, "o", label="Original", alpha=0.5, markersize=3, color="blue")
        ax1.plot(frame_numbers, processed_data, ".", label="Processed", markersize=4, color="red", alpha=0.7)
        ax1.set_title(f"Original vs Processed - {column_name}", fontweight="bold", color="white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")
        ax1.legend(loc="best")
        
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(frame_numbers[valid_mask], residuals[valid_mask], "o", markersize=3, label="Og. Residuals", alpha=0.4, color="green")
        ax2.plot(frame_numbers[valid_mask], filtered_residuals[valid_mask], ".", markersize=5, label="Filtered", alpha=0.7, color="red")
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1.5)
        ax2.set_title("Residuals", fontweight="bold", color="white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")
        
        rms_error = np.sqrt(np.nanmean(residuals**2))
        ax2.text(0.02, 0.98, f"RMS Error: {rms_error:.4f}", transform=ax2.transAxes, va="top", bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

        ax3 = fig.add_subplot(3, 2, 3)
        ax3.plot(frame_numbers, first_derivative, "-", linewidth=1.5, color="magenta", alpha=0.7)
        ax3.set_title("First Derivative (Velocity)", fontweight="bold", color="white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")

        ax4 = fig.add_subplot(3, 2, 4)
        ax4.plot(frame_numbers, second_derivative, "-", linewidth=1.5, color="cyan", alpha=0.7)
        ax4.set_title("Second Derivative (Acceleration)", fontweight="bold", color="white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor("#333333" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "white")
            ax.tick_params(colors="white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")
            for spine in ax.spines.values():
                spine.set_color("white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")

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
                self.window.quit()
                self.window.destroy()
            
    def cancel(self):
        self.result = None
        self.window.quit()
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
            self.interp_combo.set(name_mapping[interp_rev[interp.get("method")]])
            self.on_interp_change(name_mapping[interp_rev[interp.get("method")]])
            
        self.max_gap_var.set(str(interp.get("max_gap", 60)))
        
        smoothing = config.get("smoothing", {})
        if smoothing.get("method") in smooth_rev:
            name_mapping = {
                "1": "None", "2": "Savitzky-Golay", "3": "LOWESS", "4": "Kalman", "5": "Butterworth",
                "6": "Splines", "7": "ARIMA", "8": "Moving Median", "9": "Hampel Filter"
            }
            self.smooth_combo.set(name_mapping[smooth_rev[smoothing.get("method")]])
            self.on_smooth_change(name_mapping[smooth_rev[smoothing.get("method")]])
            
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
        if getattr(self.plot_frame, "cget", lambda x: "Light")("fg_color") != "transparent":
            fig.patch.set_facecolor("#2b2b2b")
        else:
            fig.patch.set_facecolor("#ebebeb")
            
        ax = fig.add_subplot(111)
        ax.plot(fc_arr, res_arr, 'o-', linewidth=2, color='tab:blue', label='Residual RMS')
        ax.axvline(x=opt_fc, color='tab:red', linestyle='--', linewidth=2, label=f'Optimal Cutoff ≈ {opt_fc:.2f} Hz')
        
        ax.set_title(f"Winter residual analysis - {column_name} (fs={fs} Hz)", fontweight='bold', color='white' if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else 'black')
        ax.set_xlabel('Cutoff Frequency (fc) [Hz]', fontweight='bold')
        ax.set_ylabel('Residual RMS', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.set_facecolor("#333333" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "white")
        ax.tick_params(colors="white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")
        for spine in ax.spines.values():
            spine.set_color("white" if fig.patch.get_facecolor() == (0.16862745098039217, 0.16862745098039217, 0.16862745098039217, 1.0) else "black")
            
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
