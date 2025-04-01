"""
================================================================================
vailaplot2d.py
================================================================================
Author: Prof. Paulo Santiago
Created: 23 September 2024
Updated: 01 April 2025
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

New Features in v1.1:
---------------------
- Added functionality to clear all loaded data (pandas DataFrames) from memory
  when clearing plots.
- Improved memory management by invoking garbage collection after clearing data.
- Enhanced user feedback in the console for clearer actions.

Modules and Packages Required:
------------------------------
- Python Standard Libraries: tkinter for GUI creation, os for file management.
- External Libraries:
  * matplotlib for plotting.
  * pandas for CSV data handling.
  * spm1d for Statistical Parametric Mapping analysis.
  * matplotlib.colors for advanced color management in plots.
  * gc for memory management and garbage collection.

How to Use:
-----------
1. Run the Script:
   Execute the script using the following command:
   python -m vaila.plot_2d_graphs

2. Select Plot Type:
   A GUI window will appear, offering multiple plot types. Select the desired plot
   type and follow the prompts to choose files and headers for plotting.

3. Use Plot Controls:
   The interface provides buttons to:
   - Clear all plots and cached data.
   - Generate a new figure window for plotting.
   - Plot data dynamically based on the user's file and header selections.

License:
--------
This script is licensed under the GNU General Public License v3.0. For more details,
refer to the LICENSE file located in the project root, or visit:
https://www.gnu.org/licenses/gpl-3.0.en.html

Disclaimer:
-----------
This script is provided "as is," without any warranty, express or implied. The authors
are not liable for any damage or data loss resulting from the use of this script. It is
intended solely for academic and research purposes.

Changelog:
----------
- 2024-09-23: Initial creation of the script with support for multiple plot types
  and dynamic plot management.
- 2024-09-24: Added functionality to clear plots and data from memory, improved
  memory management.
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
)
from spm1d import stats
import os
import matplotlib.colors as mcolors

# Global variables to store user selections
selected_files = []
selected_headers = []
plot_type = None

# Defining a list of colors starting with R, G, B, followed by matplotlib's color palette
base_colors = ["r", "g", "b"]
additional_colors = list(mcolors.TABLEAU_COLORS.keys())
predefined_colors = base_colors + additional_colors


# Function to clear all plots from memory
def clear_plots():
    plt.close("all")
    print("All plots cleared!")


# Function to create a new figure
def new_figure():
    plt.figure()
    print("New figure created!")


def select_plot_type():
    global plot_type

    def set_plot_type(ptype):
        global plot_type
        plot_type = ptype
        root.destroy()
        select_and_plot()

    root = Tk()
    root.title("Select Plot Type")

    Button(
        root, text="Time Scatter", command=lambda: set_plot_type("time_scatter")
    ).pack()
    Button(
        root, text="Angle-Angle", command=lambda: set_plot_type("angle_angle")
    ).pack()
    Button(
        root,
        text="Confidence Interval",
        command=lambda: set_plot_type("confidence_interval"),
    ).pack()
    Button(root, text="Boxplot", command=lambda: set_plot_type("boxplot")).pack()
    Button(root, text="SPM", command=lambda: set_plot_type("spm")).pack()

    # Button to clear all plots
    Button(root, text="Clear All Plots", command=clear_plots).pack()

    # Button to create a new figure
    Button(root, text="New Figure", command=new_figure).pack()

    root.mainloop()


def select_and_plot():
    global selected_files, selected_headers

    def on_select_file():
        file_path = select_file()
        if file_path:
            selected_files.append(file_path)
            headers = get_csv_headers(file_path)
            selected_headers.extend(select_headers_gui(headers))

    def on_plot():
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

    def on_add_more():
        root.destroy()
        select_and_plot()

    root = Tk()
    root.title("Select File and Headers")

    Button(root, text="Select File", command=on_select_file).pack()
    Button(root, text="Plot", command=on_plot).pack()
    Button(root, text="Add More", command=on_add_more).pack()

    root.mainloop()


def plot_time_scatter():
    plt.figure()
    for file_idx, file_path in enumerate(selected_files):
        data = pd.read_csv(file_path)
        # Get the first column name (could be Time, Frame, or anything else)
        x_column = data.columns[0]
        
        for header_idx, header in enumerate(selected_headers):
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
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


def plot_angle_angle():
    plt.figure()
    for file_idx, file_path in enumerate(selected_files):
        data = pd.read_csv(file_path)
        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) % 2 != 0:
            print("Please select pairs of headers for angle-angle plot.")
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
    plt.legend(loc="best", fontsize="small")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_confidence_interval():
    plt.figure()
    for file_idx, file_path in enumerate(selected_files):
        data = pd.read_csv(file_path)
        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) < 2:
            print("Please select at least two headers for confidence interval plot.")
            return
        for header_idx, header in enumerate(headers):
            values = data[header].dropna()
            ci = 1.96 * values.std() / (len(values) ** 0.5)
            color = predefined_colors[
                (file_idx * len(headers) + header_idx) % len(predefined_colors)
            ]
            plt.errorbar(
                x=range(len(values)),
                y=values,
                yerr=ci,
                label=f"{os.path.basename(file_path)}-{header}",
                color=color,
            )
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.show()


def plot_boxplot():
    plt.figure()
    data_dict = {}
    for file_idx, file_path in enumerate(selected_files):
        data = pd.read_csv(file_path)
        headers = [header for header in selected_headers if header in data.columns]
        for header in headers:
            if header not in data_dict:
                data_dict[header] = []
            data_dict[header].extend(data[header].dropna().values)
    plt.boxplot(data_dict.values(), notch=True, patch_artist=True)
    colors_list = [
        predefined_colors[i % len(predefined_colors)] for i in range(len(data_dict))
    ]
    for patch, color in zip(plt.gca().artists, colors_list):
        patch.set_facecolor(color)
    plt.xticks(range(1, len(data_dict) + 1), data_dict.keys())
    plt.xlabel("Headers")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.show()


def plot_spm():
    plt.figure()
    for file_path in selected_files:
        data = pd.read_csv(file_path)
        headers = [header for header in selected_headers if header in data.columns]
        if len(headers) < 2:
            print("Please select at least two headers for SPM plot.")
            return
        yA = data[headers[0]].dropna().values
        yB = data[headers[1]].dropna().values
        t = stats.ttest2(yA, yB)
        ti = t.inference(0.05, two_tailed=True)
        ti.plot()
    plt.tight_layout()
    plt.show()


def select_file():
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    file_path = filedialog.askopenfilename(
        title="Pick file to select headers", filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    return file_path


def get_csv_headers(file_path):
    df = pd.read_csv(file_path)
    return list(df.columns)


def select_headers_gui(headers):
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
    selection_window.geometry(
        f"{selection_window.winfo_screenwidth()}x{int(selection_window.winfo_screenheight()*0.9)}"
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

    num_columns = 6  # Number of columns for the labels

    for i, label in enumerate(headers):
        chk = Checkbutton(scrollable_frame, text=label, variable=header_vars[i])
        chk.grid(row=i // num_columns, column=i % num_columns, sticky="w")

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    btn_frame = Frame(selection_window)
    btn_frame.pack(side="right", padx=10, pady=10, fill="y", anchor="center")

    btn_select_all = Button(btn_frame, text="Select All", command=select_all)
    btn_select_all.pack(side="top", pady=5)

    btn_unselect_all = Button(btn_frame, text="Unselect All", command=unselect_all)
    btn_unselect_all.pack(side="top", pady=5)

    btn_select = Button(btn_frame, text="Confirm", command=on_select)
    btn_select.pack(side="top", pady=5)

    selection_window.mainloop()

    return selected_headers


def plot_2d():
    select_plot_type()
    messagebox.showinfo(
        "Plotting Completed", "All selected plots have been successfully generated."
    )


if __name__ == "__main__":
    plot_2d()
