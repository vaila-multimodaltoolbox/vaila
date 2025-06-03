"""
================================================================================
vailaplot2d.py
================================================================================
Author: Prof. Paulo Santiago
Created: 23 September 2024
Updated: 03 June 2025
Version: 0.0.2

Description:
------------
This script provides functionality for generating 2D plots within vailá:Versatile
Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox. It includes
a graphical user interface (GUI) for selecting and plotting various graph types,
such as scatter plots, angle-angle plots, and confidence intervals. Additionally,
the script offers buttons to clear all plots from memory, clear cached data,
and create new figure windows for refreshed plotting.

Plot Types Supported:
---------------------
1. Time Scatter Plot: Plots time-series data across multiple headers from selected
   files.
2. Angle-Angle Plot: Displays relationships between two angles based on header pairs
   from the selected files.
3. Confidence Interval Plot: Plots data with corresponding confidence intervals,
   highlighting statistical variability.
4. Boxplot: Generates boxplots for selected data columns.
5. SPM (Statistical Parametric Mapping): Conducts SPM analysis using data from
   multiple headers.

Functionalities:
----------------
1. GUI for Plot Selection: A simple, intuitive GUI using Python's Tkinter library allows
   users to select files, headers, and the desired plot type.
2. Plot and Data Management:
   - Clear All Plots: Button to clear all matplotlib plots from memory.
   - Clear Data: Button to clear loaded data from memory (pandas DataFrames, selected
     files, and headers).
   - New Figure: Button to create a new figure window for fresh plotting.
3. Dynamic Plotting: The script dynamically loads and plots data based on user selections
   from CSV files.

New Features in v0.1.0:
---------------------
- Improved plot memory management: properly closes figures and cleans memory
- Redesigned GUI with status bar and better visual feedback
- Added functionality to clear all data from memory
- Added save functionality for generated plots
- Improved error handling and user feedback
================================================================================
"""

import matplotlib.pyplot as plt
import pandas as pd
from tkinter import (
    Tk,
    Button,
    filedialog,
    Toplevel,
    Checkbutton,
    BooleanVar,
    Canvas,
    Scrollbar,
    Frame,
    messagebox,
    Label,
    LabelFrame,
    StringVar,
)
from spm1d import stats
import os
import matplotlib.colors as mcolors
import numpy as np
from rich import print
from pathlib import Path
import gc  # For garbage collection

# Global variables to store user selections
selected_files = []
selected_headers = []
plot_type = None
loaded_data_cache = {}  # Store loaded DataFrames to avoid reloading
current_figures = []  # Keep track of generated matplotlib figures

# Defining a list of colors starting with R, G, B, followed by matplotlib's color palette
base_colors = ["r", "g", "b"]
additional_colors = list(mcolors.TABLEAU_COLORS.keys())
predefined_colors = base_colors + additional_colors


# Function to clear all plots from memory
def clear_plots():
    """Clear all matplotlib plots from memory and reset figure list"""
    plt.close("all")
    global current_figures
    current_figures = []
    print("All plots cleared!")
    return "All plots cleared!"


# Function to clear all data
def clear_data():
    """Clear all loaded data and selections from memory"""
    global selected_files, selected_headers, loaded_data_cache, plot_type
    selected_files = []
    selected_headers = []
    loaded_data_cache = {}
    plot_type = None

    # Force garbage collection to free memory
    gc.collect()
    print("All data cleared!")
    return "All data cleared!"


# Function to create a new figure
def new_figure():
    """Create a new matplotlib figure and add to tracking list"""
    # Always close current figure if any exists
    if plt.get_fignums():
        plt.figure()  # Create new figure
    else:
        plt.figure()  # Create first figure

    global current_figures
    current_figures.append(plt.gcf())  # Track the new figure
    print("New figure created!")
    return "New figure created!"


# Function to save current figure
def save_figure():
    """Save the current matplotlib figure to a file"""
    if not plt.get_fignums():
        return "No figure to save!"

    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    file_path = filedialog.asksaveasfilename(
        title="Save Figure",
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf"),
            ("SVG files", "*.svg"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if file_path:
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to: {file_path}")
        return f"Figure saved to: {file_path}"
    return "Save cancelled."


class PlotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("vailá Plot 2D")
        self.root.geometry("700x500")

        # Initialize status variable
        self.status_var = StringVar()
        self.status_var.set("Ready")

        self._create_widgets()

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main frame
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = Label(
            main_frame, text="vailá 2D Plotting Tool", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Plot type selection frame
        plot_type_frame = LabelFrame(
            main_frame, text="Select Plot Type", padx=10, pady=10
        )
        plot_type_frame.pack(fill="x", pady=10)

        # Create 3x2 grid of plot type buttons
        self.plot_buttons = []

        # Row 1
        btn_time = Button(
            plot_type_frame,
            text="Time Scatter",
            command=lambda: self._set_plot_type("time_scatter"),
            width=15,
            height=2,
        )
        btn_time.grid(row=0, column=0, padx=5, pady=5)
        self.plot_buttons.append(btn_time)

        btn_angle = Button(
            plot_type_frame,
            text="Angle-Angle",
            command=lambda: self._set_plot_type("angle_angle"),
            width=15,
            height=2,
        )
        btn_angle.grid(row=0, column=1, padx=5, pady=5)
        self.plot_buttons.append(btn_angle)

        btn_ci = Button(
            plot_type_frame,
            text="Confidence Interval",
            command=lambda: self._set_plot_type("confidence_interval"),
            width=15,
            height=2,
        )
        btn_ci.grid(row=0, column=2, padx=5, pady=5)
        self.plot_buttons.append(btn_ci)

        # Row 2
        btn_box = Button(
            plot_type_frame,
            text="Boxplot",
            command=lambda: self._set_plot_type("boxplot"),
            width=15,
            height=2,
        )
        btn_box.grid(row=1, column=0, padx=5, pady=5)
        self.plot_buttons.append(btn_box)

        btn_spm = Button(
            plot_type_frame,
            text="SPM",
            command=lambda: self._set_plot_type("spm"),
            width=15,
            height=2,
        )
        btn_spm.grid(row=1, column=1, padx=5, pady=5)
        self.plot_buttons.append(btn_spm)

        # Control buttons frame
        control_frame = LabelFrame(main_frame, text="Plot Controls", padx=10, pady=10)
        control_frame.pack(fill="x", pady=10)

        # Row 1 of controls
        btn_clear_plots = Button(
            control_frame, text="Clear All Plots", command=self._clear_plots, width=15
        )
        btn_clear_plots.grid(row=0, column=0, padx=5, pady=5)

        btn_clear_data = Button(
            control_frame, text="Clear All Data", command=self._clear_data, width=15
        )
        btn_clear_data.grid(row=0, column=1, padx=5, pady=5)

        # Row 2 of controls
        btn_new_figure = Button(
            control_frame, text="New Figure", command=self._new_figure, width=15
        )
        btn_new_figure.grid(row=1, column=0, padx=5, pady=5)

        btn_save_figure = Button(
            control_frame,
            text="Save Current Figure",
            command=self._save_figure,
            width=15,
        )
        btn_save_figure.grid(row=1, column=1, padx=5, pady=5)

        # Status bar at bottom
        status_bar = Label(
            self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w"
        )
        status_bar.pack(side="bottom", fill="x")

    def _set_plot_type(self, ptype):
        """Set the plot type and start the file selection process"""
        global plot_type
        plot_type = ptype

        # Update status
        self.status_var.set(f"Selected plot type: {ptype}")

        # Highlight selected button
        for btn in self.plot_buttons:
            btn.config(bg="SystemButtonFace")  # Reset all buttons

        for btn in self.plot_buttons:
            if btn["text"].lower().replace("-", "_").replace(" ", "_") == ptype:
                btn.config(bg="lightblue")  # Highlight selected button

        # Start file selection
        self.root.withdraw()  # Hide main window temporarily
        select_and_plot()
        self.root.deiconify()  # Show main window again

    def _clear_plots(self):
        """Clear all plots and update status"""
        result = clear_plots()
        self.status_var.set(result)

    def _clear_data(self):
        """Clear all data and update status"""
        result = clear_data()
        self.status_var.set(result)

    def _new_figure(self):
        """Create new figure and update status"""
        result = new_figure()
        self.status_var.set(result)

    def _save_figure(self):
        """Save current figure and update status"""
        result = save_figure()
        self.status_var.set(result)


def select_plot_type():
    """Create the main GUI window and start the application"""
    root = Tk()
    app = PlotGUI(root)
    root.mainloop()


def select_and_plot():
    """Function to handle file and header selection, then plot"""
    global selected_files, selected_headers

    def on_select_file():
        """Handler for selecting a file"""
        file_path = select_file()
        if file_path:
            selected_files.append(file_path)
            # Load file into memory
            if file_path not in loaded_data_cache:
                try:
                    loaded_data_cache[file_path] = pd.read_csv(file_path)
                    status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                    return

            headers = get_csv_headers(file_path)
            selected_headers.extend(select_headers_gui(headers))

            # Update status
            file_count_label.config(text=f"Files selected: {len(selected_files)}")
            header_count_label.config(text=f"Headers selected: {len(selected_headers)}")

    def on_plot():
        """Handler for plotting the selected data"""
        if not selected_files or not selected_headers:
            messagebox.showwarning(
                "Warning", "Please select at least one file and header."
            )
            return

        # Create a new figure for the plot
        new_figure()

        if plot_type == "time_scatter":
            plot_time_scatter()
        elif plot_type == "angle_angle":
            plot_angle_angle()
        elif plot_type == "confidence_interval":
            plot_confidence_interval()
        elif plot_type == "boxplot":
            plot_boxplot()
        elif plot_type == "spm":
            plot_spm()

        # Update status
        status_label.config(text=f"Plot created: {plot_type}")
        root.destroy()

    def on_add_more():
        """Handler for adding more files/headers"""
        root.destroy()
        select_and_plot()

    def on_clear():
        """Handler for clearing selections"""
        global selected_files, selected_headers
        selected_files = []
        selected_headers = []
        file_count_label.config(text="Files selected: 0")
        header_count_label.config(text="Headers selected: 0")
        status_label.config(text="Selections cleared")

    root = Tk()
    root.title("Select Files and Headers")
    root.geometry("500x300")

    # Main frame with padding
    main_frame = Frame(root, padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)

    # Status display section
    status_frame = LabelFrame(main_frame, text="Status", padx=10, pady=10)
    status_frame.pack(fill="x", pady=(0, 10))

    file_count_label = Label(
        status_frame, text=f"Files selected: {len(selected_files)}"
    )
    file_count_label.pack(anchor="w")

    header_count_label = Label(
        status_frame, text=f"Headers selected: {len(selected_headers)}"
    )
    header_count_label.pack(anchor="w")

    status_label = Label(status_frame, text="Ready")
    status_label.pack(anchor="w")

    # Buttons frame
    button_frame = Frame(main_frame)
    button_frame.pack(fill="x", pady=10)

    Button(button_frame, text="Select File", command=on_select_file, width=12).pack(
        side="left", padx=5
    )
    Button(button_frame, text="Clear Selections", command=on_clear, width=12).pack(
        side="left", padx=5
    )
    Button(button_frame, text="Plot", command=on_plot, width=12).pack(
        side="left", padx=5
    )
    Button(button_frame, text="Add More", command=on_add_more, width=12).pack(
        side="left", padx=5
    )

    root.mainloop()


def plot_time_scatter():
    """Plot time series data from selected files and headers"""
    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    for file_idx, file_path in enumerate(selected_files):
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            data = pd.read_csv(file_path)
            loaded_data_cache[file_path] = data

        # Get the first column name (could be Time, Frame, or anything else)
        x_column = data.columns[0]

        for header_idx, header in enumerate(selected_headers):
            if header not in data.columns:
                continue

            if header == x_column:  # Skip if the header is the x-axis column
                continue

            color = predefined_colors[
                (file_idx * len(selected_headers) + header_idx) % len(predefined_colors)
            ]
            plt.plot(
                data[x_column],
                data[header],
                label=f"{os.path.basename(file_path)}-{header}",
                color=color,
            )

    # Set x-label based on the first column name
    plt.xlabel(x_column)
    plt.ylabel("Values")
    plt.title("Time Series Plot")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


def plot_angle_angle():
    """Plot angle-angle diagrams from selected files and headers"""
    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    for file_idx, file_path in enumerate(selected_files):
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            data = pd.read_csv(file_path)
            loaded_data_cache[file_path] = data

        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) % 2 != 0:
            messagebox.showwarning(
                "Warning", "Please select pairs of headers for angle-angle plot."
            )
            return

        for i in range(0, len(headers), 2):
            color = predefined_colors[
                (file_idx * len(headers) // 2 + i // 2) % len(predefined_colors)
            ]
            plt.plot(
                data[headers[i]],
                data[headers[i + 1]],
                label=f"{os.path.basename(file_path)}-{headers[i]} vs {headers[i + 1]}",
                color=color,
            )
    plt.xlabel("Angle 1")
    plt.ylabel("Angle 2")
    plt.title("Angle-Angle Plot")
    plt.legend(loc="best", fontsize="small")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_confidence_interval():
    """Plot data with confidence intervals"""
    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    for file_idx, file_path in enumerate(selected_files):
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            data = pd.read_csv(file_path)
            loaded_data_cache[file_path] = data

        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) < 1:
            messagebox.showwarning(
                "Warning",
                "Please select at least one header for confidence interval plot.",
            )
            return

        # Calculate median across all selected columns for each time point
        median_values = data[headers].median(axis=1)

        # Calculate confidence interval using bootstrap
        n_bootstrap = 1000
        bootstrap_medians = np.zeros((len(data), n_bootstrap))

        for i in range(n_bootstrap):
            # Randomly sample columns with replacement
            sampled_cols = np.random.choice(headers, size=len(headers), replace=True)
            bootstrap_medians[:, i] = data[sampled_cols].median(axis=1)

        # Calculate 95% confidence interval
        ci_lower = np.percentile(bootstrap_medians, 2.5, axis=1)
        ci_upper = np.percentile(bootstrap_medians, 97.5, axis=1)

        # Plot median and confidence interval
        color = predefined_colors[file_idx % len(predefined_colors)]
        plt.plot(
            median_values, label=f"{os.path.basename(file_path)} - Median", color=color
        )
        plt.fill_between(
            range(len(median_values)),
            ci_lower,
            ci_upper,
            alpha=0.2,
            color=color,
            label=f"{os.path.basename(file_path)} - 95% CI",
        )

    plt.xlabel("Time Points")
    plt.ylabel("Values")
    plt.title("Median and 95% Confidence Interval")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


def plot_boxplot():
    """Create boxplots for selected data"""
    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    data_dict = {}
    for file_idx, file_path in enumerate(selected_files):
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            data = pd.read_csv(file_path)
            loaded_data_cache[file_path] = data

        headers = [header for header in selected_headers if header in data.columns]
        for header in headers:
            key = f"{os.path.basename(file_path)}-{header}"
            data_dict[key] = data[header].dropna().values

    if not data_dict:
        messagebox.showwarning("Warning", "No valid data for boxplot")
        return

    plt.boxplot(data_dict.values(), notch=True, patch_artist=True)
    colors_list = [
        predefined_colors[i % len(predefined_colors)] for i in range(len(data_dict))
    ]
    for patch, color in zip(plt.gca().artists, colors_list):
        patch.set_facecolor(color)

    # Format x-tick labels for better readability
    plt.xticks(range(1, len(data_dict) + 1), data_dict.keys(), rotation=45, ha="right")
    plt.xlabel("Headers")
    plt.ylabel("Values")
    plt.title("Boxplot")
    plt.tight_layout()
    plt.show()


def plot_spm():
    """Perform statistical parametric mapping analysis"""
    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    for file_path in selected_files:
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            data = pd.read_csv(file_path)
            loaded_data_cache[file_path] = data

        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) < 2:
            messagebox.showwarning(
                "Warning", "Please select at least two headers for SPM plot."
            )
            return

        # Perform t-test between first two selected headers
        yA = data[headers[0]].dropna().values
        yB = data[headers[1]].dropna().values
        t = stats.ttest2(yA, yB)
        ti = t.inference(0.05, two_tailed=True)
        ti.plot()
        plt.title(f"SPM: {headers[0]} vs {headers[1]}")

    plt.tight_layout()
    plt.show()


def select_file():
    """Open a dialog to select a CSV file"""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Pick file to select headers", filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    return file_path


def get_csv_headers(file_path):
    """Extract column headers from a CSV file"""
    # Use cached data if available
    if file_path in loaded_data_cache:
        return list(loaded_data_cache[file_path].columns)

    df = pd.read_csv(file_path)
    loaded_data_cache[file_path] = df  # Cache it for later use
    return list(df.columns)


def select_headers_gui(headers):
    """GUI for selecting headers from a CSV file"""
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_headers = [
            header for header, var in zip(headers, header_vars) if var.get()
        ]
        selection_window.quit()  # End the main Tkinter loop
        selection_window.destroy()  # Destroy the selection window

    def select_all():
        for var in header_vars:
            var.set(True)

    def unselect_all():
        for var in header_vars:
            var.set(False)

    selection_window = Toplevel()
    selection_window.title("Select Headers")
    selection_window.geometry("800x600")

    # Create a frame with a title
    title_frame = Frame(selection_window)
    title_frame.pack(fill="x", pady=10)

    Label(
        title_frame, text="Select Headers for Analysis", font=("Arial", 14, "bold")
    ).pack()

    # Create scrollable frame for headers
    canvas = Canvas(selection_window)
    scrollbar = Scrollbar(selection_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    header_vars = [BooleanVar() for _ in headers]

    num_columns = 4  # Number of columns for the headers

    for i, label in enumerate(headers):
        chk = Checkbutton(
            scrollable_frame, text=label, variable=header_vars[i], padx=5, pady=2
        )
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Button frame at bottom
    btn_frame = Frame(selection_window)
    btn_frame.pack(side="bottom", fill="x", padx=10, pady=10)

    Button(btn_frame, text="Select All", command=select_all, width=15).pack(
        side="left", padx=5
    )
    Button(btn_frame, text="Unselect All", command=unselect_all, width=15).pack(
        side="left", padx=5
    )
    Button(btn_frame, text="Confirm", command=on_select, width=15).pack(
        side="right", padx=5
    )

    selection_window.mainloop()

    return selected_headers


def run_plot_2d():
    """Main function to run the plotting application"""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

    select_plot_type()

    # Show completion message only if plots were created
    if current_figures:
        messagebox.showinfo(
            "Plotting Completed", "All selected plots have been successfully generated."
        )


if __name__ == "__main__":
    run_plot_2d()
