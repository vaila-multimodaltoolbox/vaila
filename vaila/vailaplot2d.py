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

from tkinter import (
    END,
    MULTIPLE,
    Button,
    Frame,
    Label,
    LabelFrame,
    Listbox,
    Scrollbar,
    StringVar,
    Tk,
    Toplevel,
    filedialog,
    messagebox,
)

import matplotlib.pyplot as plt
import pandas as pd

try:
    from spm1d import stats

    SPM_SUPPORT = True
except ImportError:
    SPM_SUPPORT = False
    stats = None
    print("Warning: spm1d not available. SPM plot support disabled.")
import gc  # For garbage collection
import os
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
from rich import print

# Try to import additional libraries for different file formats
try:
    import openpyxl  # For Excel files

    EXCEL_SUPPORT = True
except ImportError:
    EXCEL_SUPPORT = False
    print("Warning: openpyxl not available. Excel file support limited.")

# Optional ODS support
ODS_SUPPORT = False
try:
    from odf import opendocument  # type: ignore[import]

    ODS_SUPPORT = True
except ImportError:
    print("Warning: odfpy not available. ODS file support limited.")

try:
    from ezc3d import c3d  # For C3D files

    C3D_SUPPORT = True
except ImportError:
    C3D_SUPPORT = False
    print("Warning: ezc3d not available. C3D file support limited.")

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
        self.root.geometry("900x600")  # Increased window size

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
        title_label = Label(main_frame, text="vailá 2D Plotting Tool", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # Plot type selection frame
        plot_type_frame = LabelFrame(main_frame, text="Select Plot Type", padx=15, pady=15)
        plot_type_frame.pack(fill="x", pady=10)

        # Create 2 rows with 3 columns each for better layout
        self.plot_buttons = []

        # Row 1 - 3 columns
        btn_time = Button(
            plot_type_frame,
            text="Time Scatter",
            command=lambda: self._set_plot_type("time_scatter"),
            width=20,
            height=2,
            font=("Arial", 10),
        )
        btn_time.grid(row=0, column=0, padx=8, pady=8, sticky="ew")
        self.plot_buttons.append(btn_time)

        btn_angle = Button(
            plot_type_frame,
            text="Angle-Angle",
            command=lambda: self._set_plot_type("angle_angle"),
            width=20,
            height=2,
            font=("Arial", 10),
        )
        btn_angle.grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        self.plot_buttons.append(btn_angle)

        btn_ci = Button(
            plot_type_frame,
            text="Confidence Interval",
            command=lambda: self._set_plot_type("confidence_interval"),
            width=20,
            height=2,
            font=("Arial", 10),
        )
        btn_ci.grid(row=0, column=2, padx=8, pady=8, sticky="ew")
        self.plot_buttons.append(btn_ci)

        # Row 2 - 3 columns
        btn_box = Button(
            plot_type_frame,
            text="Boxplot",
            command=lambda: self._set_plot_type("boxplot"),
            width=20,
            height=2,
            font=("Arial", 10),
        )
        btn_box.grid(row=1, column=0, padx=8, pady=8, sticky="ew")
        self.plot_buttons.append(btn_box)

        btn_spm = Button(
            plot_type_frame,
            text="SPM",
            command=lambda: self._set_plot_type("spm"),
            width=20,
            height=2,
            font=("Arial", 10),
        )
        btn_spm.grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        self.plot_buttons.append(btn_spm)

        # Empty cell in row 2, column 2 for symmetry
        empty_label = Label(plot_type_frame, text="")
        empty_label.grid(row=1, column=2, padx=8, pady=8)

        # Control buttons frame
        control_frame = LabelFrame(main_frame, text="Plot Controls", padx=15, pady=15)
        control_frame.pack(fill="x", pady=10)

        # Create a more organized layout with better spacing
        btn_clear_plots = Button(
            control_frame,
            text="Clear All Plots",
            command=self._clear_plots,
            width=18,
            height=2,
            font=("Arial", 10),
        )
        btn_clear_plots.grid(row=0, column=0, padx=8, pady=8, sticky="ew")

        btn_clear_data = Button(
            control_frame,
            text="Clear All Data",
            command=self._clear_data,
            width=18,
            height=2,
            font=("Arial", 10),
        )
        btn_clear_data.grid(row=0, column=1, padx=8, pady=8, sticky="ew")

        btn_new_figure = Button(
            control_frame,
            text="New Figure",
            command=self._new_figure,
            width=18,
            height=2,
            font=("Arial", 10),
        )
        btn_new_figure.grid(row=0, column=2, padx=8, pady=8, sticky="ew")

        btn_save_figure = Button(
            control_frame,
            text="Save Current Figure",
            command=self._save_figure,
            width=18,
            height=2,
            font=("Arial", 10),
        )
        btn_save_figure.grid(row=1, column=0, padx=8, pady=8, sticky="ew")

        # Empty cells for better layout
        empty_label1 = Label(control_frame, text="")
        empty_label1.grid(row=1, column=1, padx=8, pady=8)

        empty_label2 = Label(control_frame, text="")
        empty_label2.grid(row=1, column=2, padx=8, pady=8)

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

        # Reset all buttons to default (using SystemButtonFace for cross-platform support)
        # Linux/X11 can throw an error if passed an empty string
        for btn in self.plot_buttons:
            try:
                btn.config(bg="SystemButtonFace")
            except Exception:
                # Fallback for systems where SystemButtonFace isn't recognized
                try:
                    btn.config(bg="#d9d9d9")
                except:
                    pass

        # Start file selection in a separate window
        self.selection_window = FileSelectionWindow(self.root, ptype)

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


class FileSelectionWindow:
    """Window for file and header selection with proper layout management"""

    def __init__(self, parent, plot_type):
        self.parent = parent
        self.plot_type = plot_type
        self.selected_files = []
        self.selected_headers = []

        self.create_window()

    def create_window(self):
        """Create the file selection window"""
        self.window = Toplevel(self.parent)
        self.window.title("Select Files and Headers")
        self.window.geometry("700x600")  # Larger window
        self.window.transient(self.parent)  # Make it modal
        self.window.grab_set()  # Modal

        # Center the window on screen
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.window.winfo_screenheight() // 2) - (600 // 2)
        self.window.geometry(f"700x600+{x}+{y}")

        self.create_widgets()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        """Create all widgets for the file selection window"""
        # Main frame with padding
        main_frame = Frame(self.window, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = Label(
            main_frame, text="File and Header Selection", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 25))

        # Status display section
        status_frame = LabelFrame(main_frame, text="Status", padx=15, pady=15)
        status_frame.pack(fill="x", pady=(0, 25))

        self.file_count_label = Label(status_frame, text="Files selected: 0", font=("Arial", 10))
        self.file_count_label.pack(anchor="w")

        self.header_count_label = Label(
            status_frame, text="Headers selected: 0", font=("Arial", 10)
        )
        self.header_count_label.pack(anchor="w")

        self.status_label = Label(status_frame, text="Ready", font=("Arial", 10))
        self.status_label.pack(anchor="w")

        # Buttons frame with better organization
        button_frame = Frame(main_frame)
        button_frame.pack(fill="x", pady=(0, 25))

        # Top row buttons
        Button(
            button_frame,
            text="Select File",
            command=self.on_select_file,
            width=15,
            height=2,
            font=("Arial", 10),
        ).grid(row=0, column=0, padx=8, pady=5)

        Button(
            button_frame,
            text="Clear Selections",
            command=self.on_clear,
            width=18,
            height=2,
            font=("Arial", 10),
        ).grid(row=0, column=1, padx=8, pady=5)

        # Bottom row buttons
        Button(
            button_frame,
            text="Plot",
            command=self.on_plot,
            width=12,
            height=2,
            font=("Arial", 10),
        ).grid(row=1, column=0, padx=8, pady=5)

        Button(
            button_frame,
            text="Add More",
            command=self.on_add_more,
            width=12,
            height=2,
            font=("Arial", 10),
        ).grid(row=1, column=1, padx=8, pady=5)

        # Close button at bottom
        close_btn = Button(
            main_frame,
            text="Close",
            command=self.on_close,
            width=12,
            height=2,
            font=("Arial", 10),
        )
        close_btn.pack(pady=(25, 0))

    def on_select_file(self):
        """Handler for selecting a file"""
        file_path = select_file()
        if file_path:
            self.selected_files.append(file_path)

            # Load file into memory using appropriate method
            if file_path not in loaded_data_cache:
                try:
                    # Determine file format and read accordingly
                    file_ext = file_path.lower().split(".")[-1]

                    if file_ext == "csv":
                        loaded_data_cache[file_path] = read_csv_with_encoding(
                            file_path, skipfooter=0
                        )
                    elif file_ext == "xlsx":
                        loaded_data_cache[file_path] = read_excel_with_sheet_selection(file_path)
                    elif file_ext == "ods":
                        if ODS_SUPPORT:
                            loaded_data_cache[file_path] = pd.read_excel(file_path, engine="odf")
                        else:
                            messagebox.showerror(
                                "Error", "ODS support not available. Install odfpy."
                            )
                            return
                    elif file_ext == "c3d":
                        loaded_data_cache[file_path] = read_c3d_file(file_path)
                    else:
                        loaded_data_cache[file_path] = read_csv_with_encoding(
                            file_path, skipfooter=0
                        )

                    if loaded_data_cache[file_path] is None:
                        messagebox.showerror(
                            "Error",
                            f"Failed to load file: {os.path.basename(file_path)}",
                        )
                        self.selected_files.remove(file_path)
                        return

                    self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                    if file_path in self.selected_files:
                        self.selected_files.remove(file_path)
                    return

            headers = get_file_headers(file_path)
            if not headers:
                messagebox.showerror("Error", f"No headers found in {os.path.basename(file_path)}")
                return

            selected = select_headers_gui(headers, file_path)
            if selected:
                self.selected_headers.extend(selected)
                print(f"[DEBUG] Headers selected from GUI: {selected}")

            # Update status
            self.file_count_label.config(text=f"Files selected: {len(self.selected_files)}")
            self.header_count_label.config(text=f"Headers selected: {len(self.selected_headers)}")

    def on_clear(self):
        """Handler for clearing selections"""
        self.selected_files = []
        self.selected_headers = []
        self.file_count_label.config(text="Files selected: 0")
        self.header_count_label.config(text="Headers selected: 0")
        self.status_label.config(text="Selections cleared")

    def on_plot(self):
        """Handler for plotting the selected data"""
        if not self.selected_files or not self.selected_headers:
            messagebox.showwarning("Warning", "Please select at least one file and header.")
            return

        # Update global variables with local selections
        global selected_files, selected_headers
        selected_files = self.selected_files.copy()
        selected_headers = self.selected_headers.copy()

        print(f"Plotting with {len(selected_files)} file(s) and {len(selected_headers)} header(s)")
        print(f"Files: {[os.path.basename(f) for f in selected_files]}")
        print(f"Headers: {selected_headers}")

        # Create a new figure for the plot
        new_figure()

        if self.plot_type == "time_scatter":
            plot_time_scatter()
        elif self.plot_type == "angle_angle":
            plot_angle_angle()
        elif self.plot_type == "confidence_interval":
            plot_confidence_interval()
        elif self.plot_type == "boxplot":
            plot_boxplot()
        elif self.plot_type == "spm":
            plot_spm()

        # Update parent status
        if hasattr(self.parent, "status_var"):
            self.parent.status_var.set(f"Plot created: {self.plot_type}")

        self.window.destroy()

    def on_add_more(self):
        """Handler for adding more files/headers"""
        # Keep current selections and allow adding more
        pass  # Selections are maintained, user can select more files

    def on_close(self):
        """Handler for closing the window"""
        self.window.destroy()


def select_plot_type():
    """Create the main GUI window and start the application"""
    root = Tk()
    PlotGUI(root)
    root.mainloop()


# Legacy function removed - replaced by FileSelectionWindow class


def plot_time_scatter():
    """Plot time series data from selected files and headers"""
    print("[DEBUG] plot_time_scatter called")
    print(f"[DEBUG] selected_files: {selected_files}")
    print(f"[DEBUG] selected_headers: {selected_headers}")

    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    plot_count = 0
    for file_idx, file_path in enumerate(selected_files):
        print(
            f"[DEBUG] Processing file {file_idx + 1}/{len(selected_files)}: {os.path.basename(file_path)}"
        )

        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
            print(f"[DEBUG] Using cached data for {os.path.basename(file_path)}")
        else:
            # Determine file format and read accordingly
            file_ext = file_path.lower().split(".")[-1]
            try:
                if file_ext == "csv":
                    data = read_csv_with_encoding(file_path, skipfooter=0)
                elif file_ext == "xlsx":
                    data = pd.read_excel(file_path)
                elif file_ext == "ods":
                    data = pd.read_excel(file_path, engine="odf")
                elif file_ext == "c3d":
                    data = read_c3d_file(file_path)
                    if data is None:
                        continue
                else:
                    data = read_csv_with_encoding(file_path, skipfooter=0)  # Default fallback

                loaded_data_cache[file_path] = data
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue

        # Get the first column name (could be Time, Frame, or anything else)
        if len(data.columns) == 0:
            print(f"[DEBUG] No columns found in {os.path.basename(file_path)}")
            continue

        x_column = data.columns[0]
        print(f"[DEBUG] X-axis column: {x_column}")

        # Check if Time column is problematic (only 0s and 1s or not increasing properly)
        if x_column.lower() in ["time", "tempo", "frame"]:
            unique_time_values = sorted(data[x_column].unique())
            if len(unique_time_values) <= 2 or not data[x_column].is_monotonic_increasing:
                print(f"[WARNING] Time column has issues: {unique_time_values}")
                print("[DEBUG] Using row index as X-axis instead")
                # Create a proper time axis using row index
                data = data.copy()
                data.insert(0, "Time_Index", range(1, len(data) + 1))
                x_column = "Time_Index"

        for header_idx, header in enumerate(selected_headers):
            print(f"[DEBUG] Checking header {header_idx + 1}/{len(selected_headers)}: {header}")

            if header not in data.columns:
                print(f"[WARNING] Header '{header}' not found in data columns")
                print(f"[DEBUG] Available columns: {list(data.columns)}")
                continue

            if header == x_column:  # Skip if the header is the x-axis column
                print(f"[DEBUG] Skipping {header} (same as x-axis)")
                continue

            color = predefined_colors[
                (file_idx * len(selected_headers) + header_idx) % len(predefined_colors)
            ]

            print(f"[DEBUG] Plotting {header} with color {color}")
            plt.plot(
                data[x_column],
                data[header],
                label=f"{os.path.basename(file_path)}-{header}",
                color=color,
            )
            plot_count += 1

    print(f"[DEBUG] Total plots created: {plot_count}")

    if plot_count == 0:
        messagebox.showwarning(
            "Warning",
            "No data was plotted. Please check your file and header selections.",
        )
        return

    # Set x-label based on the first column name
    if selected_files:
        first_file_data = loaded_data_cache.get(selected_files[0])
        if first_file_data is None:
            file_ext = selected_files[0].lower().split(".")[-1]
            try:
                if file_ext == "csv":
                    first_file_data = read_csv_with_encoding(selected_files[0], skipfooter=1)
                elif file_ext == "xlsx":
                    first_file_data = pd.read_excel(selected_files[0]) if EXCEL_SUPPORT else None
                elif file_ext == "ods":
                    if ODS_SUPPORT:
                        first_file_data = pd.read_excel(selected_files[0], engine="odf")
                    else:
                        first_file_data = None
                elif file_ext == "c3d":
                    first_file_data = read_c3d_file(selected_files[0])
                else:
                    first_file_data = None
            except:
                first_file_data = None

        if first_file_data is not None and len(first_file_data.columns) > 0:
            # Check if we're using Time_Index as x-axis
            x_label = first_file_data.columns[0]
            if x_label.lower() in ["time", "tempo", "frame"]:
                unique_time_values = sorted(first_file_data[x_label].unique())
                if (
                    len(unique_time_values) <= 2
                    or not first_file_data[x_label].is_monotonic_increasing
                ):
                    x_label = "Sample Index"
            plt.xlabel(x_label)

    plt.ylabel("Values")
    plt.title("Time Series Plot")
    plt.legend(loc="best", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_angle_angle():
    """Plot angle-angle diagrams from selected files and headers"""
    print("[DEBUG] plot_angle_angle called")
    print(f"[DEBUG] selected_files: {selected_files}")
    print(f"[DEBUG] selected_headers: {selected_headers}")

    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    plot_count = 0
    for file_idx, file_path in enumerate(selected_files):
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            # Determine file format and read accordingly
            file_ext = file_path.lower().split(".")[-1]
            try:
                if file_ext == "csv":
                    data = read_csv_with_encoding(file_path, skipfooter=0)
                elif file_ext == "xlsx":
                    data = pd.read_excel(file_path)
                elif file_ext == "ods":
                    data = pd.read_excel(file_path, engine="odf")
                elif file_ext == "c3d":
                    data = read_c3d_file(file_path)
                    if data is None:
                        continue
                else:
                    data = read_csv_with_encoding(file_path, skipfooter=0)  # Default fallback

                loaded_data_cache[file_path] = data
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue

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
            plot_count += 1

    if plot_count == 0:
        messagebox.showwarning("Warning", "No valid data pairs found for angle-angle plot.")
        return

    plt.xlabel("Angle 1")
    plt.ylabel("Angle 2")
    plt.title("Angle-Angle Plot")
    plt.legend(loc="best", fontsize="small")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_confidence_interval():
    """Plot data with confidence intervals"""
    print("[DEBUG] plot_confidence_interval called")
    print(f"[DEBUG] selected_files: {selected_files}")
    print(f"[DEBUG] selected_headers: {selected_headers}")

    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    plot_count = 0
    for file_idx, file_path in enumerate(selected_files):
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            # Determine file format and read accordingly
            file_ext = file_path.lower().split(".")[-1]
            try:
                if file_ext == "csv":
                    data = read_csv_with_encoding(file_path, skipfooter=0)
                elif file_ext == "xlsx":
                    data = pd.read_excel(file_path)
                elif file_ext == "ods":
                    data = pd.read_excel(file_path, engine="odf")
                elif file_ext == "c3d":
                    data = read_c3d_file(file_path)
                    if data is None:
                        continue
                else:
                    data = read_csv_with_encoding(file_path, skipfooter=0)  # Default fallback

                loaded_data_cache[file_path] = data
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue

        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) < 1:
            messagebox.showwarning(
                "Warning",
                "Please select at least one header for confidence interval plot.",
            )
            return

        try:
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
                median_values,
                label=f"{os.path.basename(file_path)} - Median",
                color=color,
            )
            plt.fill_between(
                range(len(median_values)),
                ci_lower,
                ci_upper,
                alpha=0.2,
                color=color,
                label=f"{os.path.basename(file_path)} - 95% CI",
            )
            plot_count += 1

        except Exception as e:
            messagebox.showwarning(
                "Warning", f"Error processing {os.path.basename(file_path)}: {str(e)}"
            )
            continue

    if plot_count == 0:
        messagebox.showwarning("Warning", "No valid data found for confidence interval plot.")
        return

    plt.xlabel("Time Points")
    plt.ylabel("Values")
    plt.title("Median and 95% Confidence Interval")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


def plot_boxplot():
    """Create boxplots for selected data"""
    print("[DEBUG] plot_boxplot called")
    print(f"[DEBUG] selected_files: {selected_files}")
    print(f"[DEBUG] selected_headers: {selected_headers}")

    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    data_dict = {}
    for _file_idx, file_path in enumerate(selected_files):
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            # Determine file format and read accordingly
            file_ext = file_path.lower().split(".")[-1]
            try:
                if file_ext == "csv":
                    data = read_csv_with_encoding(file_path, skipfooter=0)
                elif file_ext == "xlsx":
                    data = pd.read_excel(file_path)
                elif file_ext == "ods":
                    data = pd.read_excel(file_path, engine="odf")
                elif file_ext == "c3d":
                    data = read_c3d_file(file_path)
                    if data is None:
                        continue
                else:
                    data = read_csv_with_encoding(file_path, skipfooter=0)  # Default fallback

                loaded_data_cache[file_path] = data
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue

        headers = [header for header in selected_headers if header in data.columns]
        for header in headers:
            key = f"{os.path.basename(file_path)}-{header}"
            values = data[header].dropna().values
            if len(values) > 0:
                data_dict[key] = values

    if not data_dict:
        messagebox.showwarning("Warning", "No valid data for boxplot")
        return

    plt.boxplot(data_dict.values(), notch=True, patch_artist=True)
    colors_list = [predefined_colors[i % len(predefined_colors)] for i in range(len(data_dict))]
    for patch, color in zip(plt.gca().artists, colors_list, strict=False):
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
    print("[DEBUG] plot_spm called")
    print(f"[DEBUG] selected_files: {selected_files}")
    print(f"[DEBUG] selected_headers: {selected_headers}")

    if not SPM_SUPPORT:
        messagebox.showerror(
            "Error",
            "SPM analysis requires spm1d library.\nInstall with: pip install spm1d",
        )
        return

    # Always ensure we're plotting to a new figure
    if not plt.get_fignums():
        plt.figure()

    plot_count = 0
    for file_path in selected_files:
        # Use cached data if available
        if file_path in loaded_data_cache:
            data = loaded_data_cache[file_path]
        else:
            # Determine file format and read accordingly
            file_ext = file_path.lower().split(".")[-1]
            try:
                if file_ext == "csv":
                    data = read_csv_with_encoding(file_path, skipfooter=0)
                elif file_ext == "xlsx":
                    data = pd.read_excel(file_path)
                elif file_ext == "ods":
                    data = pd.read_excel(file_path, engine="odf")
                elif file_ext == "c3d":
                    data = read_c3d_file(file_path)
                    if data is None:
                        continue
                else:
                    data = read_csv_with_encoding(file_path, skipfooter=0)  # Default fallback

                loaded_data_cache[file_path] = data
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
                continue

        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) < 2:
            messagebox.showwarning("Warning", "Please select at least two headers for SPM plot.")
            return

        try:
            # Perform t-test between first two selected headers
            yA = data[headers[0]].dropna().values
            yB = data[headers[1]].dropna().values

            if len(yA) == 0 or len(yB) == 0:
                messagebox.showwarning(
                    "Warning",
                    f"No valid data in {os.path.basename(file_path)} for SPM analysis.",
                )
                continue

            t = stats.ttest2(yA, yB)
            ti = t.inference(0.05, two_tailed=True)
            ti.plot()
            plt.title(f"SPM: {os.path.basename(file_path)} - {headers[0]} vs {headers[1]}")
            plot_count += 1

        except Exception as e:
            messagebox.showwarning(
                "Warning",
                f"Error in SPM analysis for {os.path.basename(file_path)}: {str(e)}",
            )
            continue

    if plot_count == 0:
        messagebox.showwarning("Warning", "No valid SPM plots could be generated.")
        return

    plt.tight_layout()
    plt.show()


def select_file():
    """Open a dialog to select a file (CSV, C3D, XLSX, ODS)"""
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Pick file to select headers",
        filetypes=[
            ("All supported files", "*.csv;*.c3d;*.xlsx;*.ods"),
            ("CSV files", "*.csv"),
            ("C3D files", "*.c3d"),
            ("Excel files", "*.xlsx"),
            ("LibreOffice files", "*.ods"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path


def read_excel_with_sheet_selection(file_path):
    """Read Excel file with sheet selection support"""
    try:
        if not EXCEL_SUPPORT:
            print("Excel support not available - openpyxl library not installed")
            return None

        # Get available sheets
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names

        if len(sheet_names) == 1:
            # Only one sheet, use it directly
            df = pd.read_excel(file_path, sheet_name=sheet_names[0])
            print(f"Loaded single sheet: {sheet_names[0]}")
        else:
            # Multiple sheets, let user choose
            print(f"Available sheets: {sheet_names}")

            # For now, use the first sheet as default
            # TODO: Add GUI selection for sheets in future version
            df = pd.read_excel(file_path, sheet_name=sheet_names[0])
            print(f"Loaded first sheet: {sheet_names[0]} (use GUI for sheet selection in future)")

        return df

    except Exception as e:
        print(f"Error reading Excel file {file_path}: {str(e)}")
        return None


def get_file_headers(file_path):
    """Extract column headers from various file formats with enhanced Excel support"""
    # Use cached data if available
    if file_path in loaded_data_cache:
        return list(loaded_data_cache[file_path].columns)

    # Determine file format and read accordingly
    file_ext = file_path.lower().split(".")[-1]

    try:
        if file_ext == "csv":
            # Use simple approach that works with any CSV structure
            try:
                df = pd.read_csv(file_path, nrows=0)
                loaded_data_cache[file_path] = df
                columns = list(df.columns)
                print(
                    f"Extracted {len(columns)} columns from {file_path}: {columns[:5]}..."
                )  # Show first 5
                return columns
            except Exception as e:
                print(f"Error reading CSV headers: {e}")
                # Fallback to encoding detection with footer skip for problematic files
                df = read_csv_with_encoding(file_path, skipfooter=0)
                if df is not None:
                    loaded_data_cache[file_path] = df
                    columns = list(df.columns)
                    print(f"Extracted {len(columns)} columns from {file_path}: {columns[:5]}...")
                    return columns
                return []
        elif file_ext == "xlsx":
            df = read_excel_with_sheet_selection(file_path)
            if df is None:
                print(f"Failed to read Excel file: {file_path}")
                return []
        elif file_ext == "ods":
            if ODS_SUPPORT:
                try:
                    # Get available sheets for ODS
                    try:
                        from odf import opendocument  # type: ignore[import]

                        doc = opendocument.load(file_path)
                        sheet_names = [
                            sheet.get_attribute("name")
                            for sheet in doc.spreadsheet.getElementsByType(doc.table.Table)
                        ]
                        if len(sheet_names) == 1:
                            df = pd.read_excel(file_path, engine="odf", sheet_name=sheet_names[0])
                        else:
                            df = pd.read_excel(file_path, engine="odf", sheet_name=sheet_names[0])
                            print(f"Loaded first sheet: {sheet_names[0]} from ODS file")
                    except ImportError:
                        # Fallback if odf library structure is different
                        df = pd.read_excel(file_path, engine="odf")
                except Exception as e:
                    print(f"Error reading ODS file: {e}")
                    df = None
            else:
                print(f"ODS support not available for {file_path}")
                return []
        elif file_ext == "c3d":
            df = read_c3d_file(file_path)
            if df is None:
                print(f"Failed to read C3D file: {file_path}")
                return []
        else:
            # Default to CSV with simple approach
            try:
                df = pd.read_csv(file_path, nrows=0)
                loaded_data_cache[file_path] = df
                columns = list(df.columns)
                print(f"Extracted {len(columns)} columns from {file_path}: {columns[:5]}...")
                return columns
            except Exception as e:
                print(f"Error reading file as CSV: {e}")
                return []

        if df is None:
            print(f"DataFrame is None for {file_path}")
            return []

        loaded_data_cache[file_path] = df  # Cache it for later use
        columns = list(df.columns)
        print(
            f"Extracted {len(columns)} columns from {file_path}: {columns[:5]}..."
        )  # Show first 5
        return columns

    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return []


def detect_c3d_units(pts):
    """
    Enhanced detection of C3D data units (millimeters vs meters) using multiple criteria.

    Args:
        pts: np.ndarray with shape (num_frames, num_markers, 3) - raw data from C3D

    Returns:
        bool: True if data is in millimeters (needs conversion), False if already in meters
        str: Detection method used for logging
    """
    # Remove NaN values for analysis
    valid_data = pts[~np.isnan(pts)]

    if len(valid_data) == 0:
        print("[yellow]Warning: No valid data found, assuming meters[/yellow]")
        return False, "no_valid_data"

    confidence_score = 0
    detection_reasons = []

    # Method 1: Check absolute magnitude of values
    mean_abs_value = np.mean(np.abs(valid_data))

    if mean_abs_value > 100:
        confidence_score += 3
        detection_reasons.append(f"high_magnitude (mean: {mean_abs_value:.1f})")
    elif mean_abs_value > 10:
        confidence_score += 1
        detection_reasons.append(f"moderate_magnitude (mean: {mean_abs_value:.1f})")

    # Method 2: Check data range
    data_range = np.max(valid_data) - np.min(valid_data)
    if data_range > 2000:
        confidence_score += 3
        detection_reasons.append(f"large_range ({data_range:.1f})")
    elif data_range > 100:
        confidence_score += 1
        detection_reasons.append(f"moderate_range ({data_range:.1f})")

    # Method 3: Inter-marker distances analysis
    if pts.shape[0] > 0 and pts.shape[1] > 1:
        # Use multiple frames for better statistics
        frame_indices = np.linspace(0, pts.shape[0] - 1, min(10, pts.shape[0]), dtype=int)
        all_distances = []

        for frame_idx in frame_indices:
            frame = pts[frame_idx]
            valid_markers = frame[~np.isnan(frame).any(axis=1)]

            if len(valid_markers) > 1:
                for i in range(len(valid_markers)):
                    for j in range(i + 1, len(valid_markers)):
                        dist = np.linalg.norm(valid_markers[i] - valid_markers[j])
                        all_distances.append(dist)

        if all_distances:
            avg_distance = np.mean(all_distances)
            max_distance = np.max(all_distances)
            percentile_95 = np.percentile(all_distances, 95)

            # More sophisticated distance analysis
            if avg_distance > 200 or max_distance > 4000:
                confidence_score += 3
                detection_reasons.append(
                    f"large_distances (avg: {avg_distance:.1f}, max: {max_distance:.1f})"
                )
            elif avg_distance > 50 or percentile_95 > 2000:
                confidence_score += 2
                detection_reasons.append(
                    f"moderate_distances (avg: {avg_distance:.1f}, p95: {percentile_95:.1f})"
                )

    # Method 4: Statistical analysis of coordinate values
    if len(valid_data) > 100:
        # Check percentage of values in different ranges
        very_large = np.sum(np.abs(valid_data) > 1000)
        large = np.sum(np.abs(valid_data) > 100)
        moderate = np.sum(np.abs(valid_data) > 10)

        very_large_pct = (very_large / len(valid_data)) * 100
        large_pct = (large / len(valid_data)) * 100
        moderate_pct = (moderate / len(valid_data)) * 100

        if very_large_pct > 5:  # More than 5% of values > 1000
            confidence_score += 3
            detection_reasons.append(f"very_large_values ({very_large_pct:.1f}%)")
        elif large_pct > 50:  # More than 50% of values > 100
            confidence_score += 2
            detection_reasons.append(f"many_large_values ({large_pct:.1f}%)")
        elif moderate_pct > 80:  # More than 80% of values > 10
            confidence_score += 1
            detection_reasons.append(f"mostly_moderate_values ({moderate_pct:.1f}%)")

    # Method 5: Standard deviation analysis
    std_dev = np.std(valid_data)
    if std_dev > 500:
        confidence_score += 2
        detection_reasons.append(f"high_std_dev ({std_dev:.1f})")
    elif std_dev > 100:
        confidence_score += 1
        detection_reasons.append(f"moderate_std_dev ({std_dev:.1f})")

    # Decision based on confidence score
    is_millimeters = confidence_score >= 3

    detection_summary = ", ".join(detection_reasons) if detection_reasons else "no_clear_indicators"
    final_method = f"confidence_score_{confidence_score} ({detection_summary})"

    return is_millimeters, final_method


def read_c3d_file(file_path):
    """Read C3D file and extract data as DataFrame using ezc3d with enhanced unit detection"""
    try:
        if not C3D_SUPPORT:
            print("C3D support not available - ezc3d library not installed")
            return None

        print(f"Loading C3D file: {file_path}")

        # Load C3D file using ezc3d
        c3d_data = c3d(file_path, extract_forceplat_data=True)

        print("C3D file loaded successfully")

        # Get point (marker) data
        try:
            point_data = c3d_data["data"]["points"]
            marker_labels = c3d_data["parameters"]["POINT"]["LABELS"]["value"]
            if isinstance(marker_labels[0], list):
                marker_labels = marker_labels[0]

            print(f"Point data shape: {point_data.shape}")
            print(f"Number of markers: {len(marker_labels)}")
            print(f"Marker labels: {marker_labels[:5]}...")  # Show first 5
        except KeyError as e:
            print(f"Point data not found in C3D file: {e}")
            return None

        # Get analog data
        try:
            analogs = c3d_data["data"]["analogs"]
            analog_labels = c3d_data["parameters"]["ANALOG"]["LABELS"]["value"]
            if isinstance(analog_labels[0], list):
                analog_labels = analog_labels[0]
            (
                c3d_data["parameters"]["ANALOG"]
                .get("UNITS", {})
                .get("value", ["Unknown"] * len(analog_labels))
            )
            print(f"Analog data shape: {analogs.shape}")
            print(f"Number of analog channels: {len(analog_labels)}")
            print(f"Analog labels: {analog_labels[:5]}...")  # Show first 5
        except KeyError as e:
            print(f"Analog data not found in C3D file: {e}")
            # Create empty analog data if not present
            analogs = np.array([]).reshape(0, 0)
            analog_labels = []
            print("No analog data found, continuing with marker data only")

        # Get frequencies
        marker_freq = c3d_data["header"]["points"]["frame_rate"]
        analog_freq = c3d_data["header"]["analogs"]["frame_rate"]

        print(f"Marker frequency: {marker_freq} Hz")
        print(f"Analog frequency: {analog_freq} Hz")

        # Reshape point data to (num_frames, num_markers, 3)
        pts = point_data[:3, :, :]  # use only x, y, z coordinates
        pts = np.transpose(pts, (2, 1, 0))  # shape (num_frames, num_markers, 3)

        print(f"Raw data loaded: {len(marker_labels)} markers, {pts.shape[0]} frames")

        # Auto-detect units and convert if necessary
        is_mm, detection_method = detect_c3d_units(pts)

        # Show preview data for user reference
        valid_sample = pts[~np.isnan(pts)][:10]  # First 10 valid values
        if len(valid_sample) > 0:
            print(f"Data preview (first 10 valid values): {valid_sample}")

        # Apply conversion
        if is_mm:
            pts = pts * 0.001  # Convert from millimeters to meters
            print("[bold green][OK] Applied conversion: MILLIMETERS → METERS[/bold green]")
            print(f"  Method: {detection_method}")
        else:
            print("[bold green][OK] No conversion applied: Data already in METERS[/bold green]")
            print(f"  Method: {detection_method}")

        # Show data statistics after conversion
        valid_data = pts[~np.isnan(pts)]
        if len(valid_data) > 0:
            print(
                f"  Final data range: [{np.min(valid_data):.3f}, {np.max(valid_data):.3f}] meters"
            )
            print(f"  Mean absolute value: {np.mean(np.abs(valid_data)):.3f} meters")

        # Create time columns
        marker_time_values = np.arange(pts.shape[0]) / marker_freq

        # Create DataFrame with all data
        data_dict = {}

        # Add marker data (X, Y, Z for each marker)
        for i, label in enumerate(marker_labels):
            if pts.shape[2] >= 3:  # Check if we have 3D data
                data_dict[f"{label}_X"] = pts[:, i, 0]
                data_dict[f"{label}_Y"] = pts[:, i, 1]
                data_dict[f"{label}_Z"] = pts[:, i, 2]
                print(f"Added marker: {label}")

        # Add analog data (if available)
        if len(analog_labels) > 0 and len(analogs.shape) == 2:
            analog_time_values = np.arange(analogs.shape[1]) / analog_freq

            for i, label in enumerate(analog_labels):
                data_dict[label] = analogs[i, :]
                print(f"Added analog: {label}")

            # Add analog time column
            data_dict["Analog_Time"] = analog_time_values

        # Add time columns (use marker time as primary)
        data_dict["Time"] = marker_time_values
        print(f"Using marker time as primary (length: {len(marker_time_values)})")

        df = pd.DataFrame(data_dict)
        print(f"Created DataFrame with shape: {df.shape}")
        print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns

        return df

    except Exception as e:
        print(f"Error reading C3D file: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def read_csv_with_encoding(file_path, skipfooter=0, engine=None):
    """Read CSV file with automatic encoding detection"""
    # Common encodings to try
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]

    # Prepare read_csv arguments
    read_args = {"encoding": None}
    if skipfooter > 0:
        read_args["skipfooter"] = skipfooter
        if engine is None:
            engine = "python"  # Force python engine when using skipfooter

    if engine is not None:
        read_args["engine"] = engine

    for encoding in encodings:
        try:
            read_args["encoding"] = encoding
            df = pd.read_csv(file_path, **read_args)
            print(f"Successfully read {file_path} with encoding: {encoding}")
            if skipfooter > 0:
                print(f"  Skipped {skipfooter} footer lines")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # If it's not a Unicode error, it might be a different issue
            if "codec" in str(e).lower() and "decode" in str(e).lower():
                continue
            else:
                # Re-raise non-encoding errors
                raise e

    # If all encodings fail, try with error handling
    try:
        read_args["encoding"] = "utf-8"
        read_args["errors"] = "replace"
        df = pd.read_csv(file_path, **read_args)
        print(f"Read {file_path} with UTF-8 and error replacement")
        return df
    except Exception as e:
        print(f"Failed to read {file_path} with all encodings: {str(e)}")
        raise e


# Keep backward compatibility
def get_csv_headers(file_path):
    """Extract column headers from a CSV file (backward compatibility)"""
    return get_file_headers(file_path)


def select_headers_gui(headers, file_path=None):
    """Simple GUI for selecting headers from various file formats - similar to rearrange_data.py style"""
    selected_headers = []

    def on_select():
        nonlocal selected_headers
        selected_indices = listbox.curselection()
        selected_headers = [headers[i] for i in selected_indices]
        selection_window.quit()
        selection_window.destroy()

    def select_all():
        listbox.select_set(0, END)

    def unselect_all():
        listbox.selection_clear(0, END)

    def save_selection():
        """Save current selection to JSON file"""
        selected_indices = listbox.curselection()
        selected = [headers[i] for i in selected_indices]
        if selected and file_path:
            try:
                import json
                import os

                base_name = os.path.splitext(os.path.basename(file_path))[0]
                selection_file = os.path.join(
                    os.path.dirname(file_path), f"{base_name}_header_selection.json"
                )

                with open(selection_file, "w") as f:
                    json.dump(selected, f)
                messagebox.showinfo("Success", f"Selection saved to:\n{selection_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save selection:\n{str(e)}")

    def load_selection():
        """Load previous selection from JSON file"""
        if file_path:
            try:
                import json
                import os

                base_name = os.path.splitext(os.path.basename(file_path))[0]
                selection_file = os.path.join(
                    os.path.dirname(file_path), f"{base_name}_header_selection.json"
                )

                if os.path.exists(selection_file):
                    with open(selection_file) as f:
                        saved_selection = json.load(f)

                    # Clear current selection
                    listbox.selection_clear(0, END)

                    # Set saved selection
                    for header in saved_selection:
                        if header in headers:
                            idx = headers.index(header)
                            listbox.selection_set(idx)

                    messagebox.showinfo("Success", f"Selection loaded from:\n{selection_file}")
                else:
                    messagebox.showwarning(
                        "Warning", f"No saved selection found at:\n{selection_file}"
                    )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load selection:\n{str(e)}")

    selection_window = Toplevel()
    selection_window.title("Select Headers for Analysis")
    selection_window.geometry("800x600")

    # Instructions
    instructions = Label(
        selection_window,
        text="Click to select headers (hold Ctrl for multiple). Press Enter to confirm.",
        font=("Arial", 10),
    )
    instructions.pack(pady=10)

    # Create main frame
    main_frame = Frame(selection_window)
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Create listbox for headers
    listbox_frame = Frame(main_frame)
    listbox_frame.pack(fill="both", expand=True)

    scrollbar = Scrollbar(listbox_frame)
    scrollbar.pack(side="right", fill="y")

    listbox = Listbox(
        listbox_frame,
        selectmode=MULTIPLE,
        width=80,
        height=25,
        yscrollcommand=scrollbar.set,
        font=("Arial", 10),
    )
    listbox.pack(side="left", fill="both", expand=True)

    scrollbar.config(command=listbox.yview)

    # Populate listbox with headers
    for i, header in enumerate(headers):
        listbox.insert(END, f"{i + 1:2d}: {header}")

    # Button frame
    btn_frame = Frame(selection_window)
    btn_frame.pack(fill="x", pady=10)

    # Left side buttons
    left_frame = Frame(btn_frame)
    left_frame.pack(side="left")

    Button(left_frame, text="Select All", command=select_all, width=12).pack(side="left", padx=5)
    Button(left_frame, text="Clear All", command=unselect_all, width=12).pack(side="left", padx=5)

    if file_path:
        Button(left_frame, text="Save Selection", command=save_selection, width=15).pack(
            side="left", padx=5
        )
        Button(left_frame, text="Load Selection", command=load_selection, width=15).pack(
            side="left", padx=5
        )

    # Right side button
    Button(
        btn_frame,
        text="Confirm Selection",
        command=on_select,
        width=18,
        bg="#4CAF50",
        fg="white",
        font=("Arial", 10, "bold"),
    ).pack(side="right")

    # Bind Enter key to confirm
    selection_window.bind("<Return>", lambda e: on_select())
    selection_window.bind("<Escape>", lambda e: selection_window.quit())

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


def test_csv_reading():
    """Test function to verify CSV reading with skipfooter parameter"""
    test_file = r"C:\Users\paulo\Preto\vaila\tests\sit2stand\processed\sit2stand_corrigido.csv"

    print("Testing CSV reading WITHOUT skipfooter:")
    try:
        df_no_skip = read_csv_with_encoding(test_file, skipfooter=0)
        print(f"  Shape without skipfooter: {df_no_skip.shape}")
        print(f"  Last row first few values: {df_no_skip.iloc[-1, :5].values}")
        print(f"  Contains problematic last row: {df_no_skip.iloc[-1, 0] == 1.0}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nTesting CSV reading WITH skipfooter=1:")
    try:
        df_with_skip = read_csv_with_encoding(test_file, skipfooter=1)
        print(f"  Shape with skipfooter: {df_with_skip.shape}")
        print(f"  Last row first few values: {df_with_skip.iloc[-1, :5].values}")
        print(f"  Contains problematic last row: {df_with_skip.iloc[-1, 0] == 1.0}")

        # Check if Force.Fz3 column has non-zero values
        if "Force.Fz3" in df_with_skip.columns:
            non_zero_count = (df_with_skip["Force.Fz3"] != 0).sum()
            print(f"  Non-zero values in Force.Fz3: {non_zero_count}/{len(df_with_skip)}")
        else:
            print("  Force.Fz3 column not found in DataFrame")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == "__main__":
    # Uncomment the line below to test CSV reading
    # test_csv_reading()

    run_plot_2d()
