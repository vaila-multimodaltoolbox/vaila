"""
================================================================================
Marker Re-identification Tool - reid_markers.py
================================================================================
Author: Adapted from getpixelvideo.py by Prof. Dr. Paulo R. P. Santiago
Date: Current
Version: 0.1.0
Python Version: 3.12.9

Description:
------------
This tool allows correcting identification issues in marker files generated
by getpixelvideo.py. It offers the following functionalities:

1. Marker merging: Combine markers that represent the same object
2. Gap filling: Fill gaps where a marker temporarily disappears
3. Swaps: Fix cases where IDs were swapped in certain frame intervals

================================================================================
"""

import json
import os
import shutil  # Para operações de diretório
import tkinter as tk
import warnings
from tkinter import Button as TkButton
from tkinter import Frame, Label, Tk, filedialog, messagebox, simpledialog, ttk

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import (
    Button,
    CheckButtons,
    RangeSlider,
    SpanSelector,
    TextBox,
)
from rich import print
from statsmodels.tsa.arima.model import ARIMA

matplotlib.use("TkAgg")  # Força backend interativo
from pathlib import Path


def load_markers_file():
    """Load a CSV file containing markers."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Markers File",
        filetypes=[("CSV Files", "*.csv")],
    )
    if not file_path:
        print("No file selected. Exiting.")
        return None, None

    try:
        df = pd.read_csv(file_path)
        print(f"File loaded: {file_path}")
        return df, file_path
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None


def save_markers_file(df, original_path, suffix="_reid"):
    """Save corrected markers to a new CSV file."""
    base_path, ext = os.path.splitext(original_path)
    new_path = f"{base_path}{suffix}{ext}"

    df.to_csv(new_path, index=False)
    print(f"Markers saved to: {new_path}")
    return new_path


def create_temp_dir(original_path):
    """Cria um diretório temporário para armazenar arquivos de edição."""
    base_dir = os.path.dirname(original_path)
    base_name = os.path.basename(original_path).split(".")[0]
    temp_dir = os.path.join(base_dir, f"{base_name}_temp")

    # Criar diretório se não existir
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    return temp_dir


def create_temp_file(df, original_path, suffix="_temp"):
    """Create a temporary file with current changes."""
    temp_dir = create_temp_dir(original_path)
    base_name = os.path.basename(original_path).split(".")[0]
    ext = os.path.splitext(original_path)[1]
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    temp_path = os.path.join(temp_dir, f"{base_name}{suffix}_{timestamp}{ext}")

    df.to_csv(temp_path, index=False)
    print(f"Temporary file created: {temp_path}")
    return temp_path


def clear_temp_dir(original_path):
    """Remove o diretório temporário e seus arquivos."""
    temp_dir = create_temp_dir(original_path)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"Temporary directory removed: {temp_dir}")


def detect_markers(df):
    """Detect which markers exist in the file."""
    markers = []
    for col in df.columns:
        if col.endswith("_x") and col.startswith("p"):
            marker_id = int(col[1:-2])  # Extract marker number
            markers.append(marker_id)

    return sorted(markers)


def get_marker_coords(df, marker_id):
    """Get x, y coordinates of a specific marker across all frames."""
    x_col = f"p{marker_id}_x"
    y_col = f"p{marker_id}_y"

    if x_col not in df.columns or y_col not in df.columns:
        return None, None

    x_values = df[x_col].values
    y_values = df[y_col].values

    return x_values, y_values


def detect_gaps(x_values, y_values, min_gap_size=3):
    """Detect gaps in a marker's trajectory."""
    gaps = []
    current_gap = None

    for i in range(len(x_values)):
        is_missing = (
            pd.isna(x_values[i])
            or pd.isna(y_values[i])
            or (x_values[i] == "" and y_values[i] == "")
        )

        if is_missing and current_gap is None:
            current_gap = [i, i]
        elif is_missing and current_gap is not None:
            current_gap[1] = i
        elif not is_missing and current_gap is not None:
            if current_gap[1] - current_gap[0] + 1 >= min_gap_size:
                gaps.append(current_gap)
            current_gap = None

    if current_gap is not None and current_gap[1] - current_gap[0] + 1 >= min_gap_size:
        gaps.append(current_gap)

    return gaps


def fill_gaps(df, marker_id, max_gap_size=30, method="linear"):
    """Fill gaps in a marker's trajectory."""
    x_col = f"p{marker_id}_x"
    y_col = f"p{marker_id}_y"

    if x_col not in df.columns or y_col not in df.columns:
        return df

    x_values = df[x_col].values.copy()
    y_values = df[y_col].values.copy()

    gaps = detect_gaps(x_values, y_values)
    filled_gaps = 0

    for gap_start, gap_end in gaps:
        gap_size = gap_end - gap_start + 1

        if gap_size <= max_gap_size:
            # Find valid points before and after the gap
            before_idx = gap_start - 1
            after_idx = gap_end + 1

            while before_idx >= 0 and (
                pd.isna(x_values[before_idx]) or pd.isna(y_values[before_idx])
            ):
                before_idx -= 1

            while after_idx < len(x_values) and (
                pd.isna(x_values[after_idx]) or pd.isna(y_values[after_idx])
            ):
                after_idx += 1

            if before_idx >= 0 and after_idx < len(x_values):
                # Interpolate values
                frames = [before_idx, after_idx]
                x_known = [x_values[before_idx], x_values[after_idx]]
                y_known = [y_values[before_idx], y_values[after_idx]]

                if method == "linear":
                    x_interp = np.interp(range(gap_start, gap_end + 1), frames, x_known)
                    y_interp = np.interp(range(gap_start, gap_end + 1), frames, y_known)

                    for i, idx in enumerate(range(gap_start, gap_end + 1)):
                        x_values[idx] = x_interp[i]
                        y_values[idx] = y_interp[i]

                    filled_gaps += 1

    if filled_gaps > 0:
        df[x_col] = x_values
        df[y_col] = y_values
        print(f"Filled {filled_gaps} gaps for marker {marker_id}")

    return df


def merge_markers(df, source_id, target_id, frame_range=None):
    """Combine two markers, copying data from source to target where target is empty."""
    source_x_col = f"p{source_id}_x"
    source_y_col = f"p{source_id}_y"
    target_x_col = f"p{target_id}_x"
    target_y_col = f"p{target_id}_y"

    if (
        source_x_col not in df.columns
        or source_y_col not in df.columns
        or target_x_col not in df.columns
        or target_y_col not in df.columns
    ):
        print(f"Columns for markers {source_id} or {target_id} not found.")
        return df

    if frame_range is None:
        frame_range = (0, len(df) - 1)

    start_frame, end_frame = frame_range

    for i in range(start_frame, end_frame + 1):
        source_x_valid = not (pd.isna(df.at[i, source_x_col]) or df.at[i, source_x_col] == "")
        source_y_valid = not (pd.isna(df.at[i, source_y_col]) or df.at[i, source_y_col] == "")

        # Copy source data to target if source has valid coordinates,
        # regardless of whether target already has coordinates
        if source_x_valid and source_y_valid:
            df.at[i, target_x_col] = df.at[i, source_x_col]
            df.at[i, target_y_col] = df.at[i, source_y_col]

    print(
        f"Marker {source_id} merged with marker {target_id} in frame range {start_frame}-{end_frame}"
    )
    return df


def swap_markers(df, marker_id1, marker_id2, frame_range):
    """Swap data between two markers in a specific frame range."""
    x_col1 = f"p{marker_id1}_x"
    y_col1 = f"p{marker_id1}_y"
    x_col2 = f"p{marker_id2}_x"
    y_col2 = f"p{marker_id2}_y"

    if (
        x_col1 not in df.columns
        or y_col1 not in df.columns
        or x_col2 not in df.columns
        or y_col2 not in df.columns
    ):
        print(f"Columns for markers {marker_id1} or {marker_id2} not found.")
        return df

    start_frame, end_frame = frame_range

    # Store values temporarily
    temp_x = df.loc[start_frame:end_frame, x_col1].copy()
    temp_y = df.loc[start_frame:end_frame, y_col1].copy()

    # Swap
    df.loc[start_frame:end_frame, x_col1] = df.loc[start_frame:end_frame, x_col2]
    df.loc[start_frame:end_frame, y_col1] = df.loc[start_frame:end_frame, y_col2]
    df.loc[start_frame:end_frame, x_col2] = temp_x
    df.loc[start_frame:end_frame, y_col2] = temp_y

    print(f"Markers {marker_id1} and {marker_id2} swapped in frame range {start_frame}-{end_frame}")
    return df


def save_operations_log(operations, file_path):
    """Save a log of performed operations."""
    log_path = file_path.replace(".csv", "_operations.json")

    with open(log_path, "w") as f:
        json.dump(operations, f, indent=4)

    print(f"Operations log saved to: {log_path}")


def visualize_markers(df, marker_ids=None):
    """Visualize marker trajectories for analysis with interactive selector."""
    if df is None:
        return

    if len(df.columns) == 0 or len(df) == 0:
        messagebox.showerror("Error", "The data file appears to be empty or malformed.")
        return

    frame_col = df.columns[0]
    frames = df[frame_col].values

    # Detect all markers in the file
    all_markers = detect_markers(df)

    # Ensure marker_ids is either None or a list of integers
    if isinstance(marker_ids, str):  # Fix for when a file path is passed instead of marker IDs
        marker_ids = None

    # Handle empty dataframe
    if "frame" not in df.columns or len(df) == 0:
        messagebox.showerror(
            "Error", "The data file appears to be empty or missing the 'frame' column."
        )
        return

    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 10))
    ax1 = plt.subplot2grid((2, 1), (0, 0))  # X coordinates
    ax2 = plt.subplot2grid((2, 1), (1, 0))  # Y coordinates

    plt.subplots_adjust(left=0.2, bottom=0.25, right=0.85)  # Make room for legend

    # Create an area for the markers checkbox
    markers_checkbox_ax = plt.axes([0.05, 0.2, 0.15, 0.7])

    # Initial visibility - all markers visible by default if none specified
    if marker_ids is None:
        initial_visibility = [True for _ in all_markers]
    else:
        initial_visibility = [marker_id in marker_ids for marker_id in all_markers]

    # Create checkbox for marker selection
    marker_selection = CheckButtons(
        markers_checkbox_ax, [f"Marker {m}" for m in all_markers], initial_visibility
    )

    # Add frame range slider
    frames_slider_ax = plt.axes([0.3, 0.15, 0.5, 0.03])
    frames_range = RangeSlider(frames_slider_ax, "Frames", 0, len(df) - 1, valinit=(0, len(df) - 1))

    # Dictionary to store plot lines
    lines_x = {}
    lines_y = {}

    # Function to update plot based on checkbox selection
    def update_plot():
        # Get current checkbox states
        checked = marker_selection.get_status()
        visible_markers = [
            all_markers[i] for i, checked_state in enumerate(checked) if checked_state
        ]

        # Clear current axes
        ax1.clear()
        ax2.clear()

        # Set up axes labels and grids
        ax1.set_title("X Coordinates of Markers")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("X Position")
        ax1.grid(True)

        ax2.set_title("Y Coordinates of Markers")
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Y Position")
        ax2.grid(True)

        # Plot only selected markers
        for marker_id in visible_markers:
            x_values, y_values = get_marker_coords(df, marker_id)
            if x_values is not None and y_values is not None:
                lines_x[marker_id] = ax1.plot(frames, x_values, label=f"Marker {marker_id}")[0]
                lines_y[marker_id] = ax2.plot(frames, y_values, label=f"Marker {marker_id}")[0]

        # Show frame range with vertical lines
        start_frame, end_frame = frames_range.val
        ax1.axvline(start_frame, color="r", linestyle="--")
        ax1.axvline(end_frame, color="r", linestyle="--")
        ax2.axvline(start_frame, color="r", linestyle="--")
        ax2.axvline(end_frame, color="r", linestyle="--")

        ax1.set_xlim(0, len(df) - 1)
        ax2.set_xlim(0, len(df) - 1)

        # Place legend outside the plot to avoid overlap
        if visible_markers and any(marker_id in lines_x for marker_id in visible_markers):
            handles1 = [lines_x[m] for m in visible_markers if m in lines_x]
            handles2 = [lines_y[m] for m in visible_markers if m in lines_y]

            if handles1:
                ax1.legend(
                    handles=handles1,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1),
                    borderaxespad=0,
                )
            if handles2:
                ax2.legend(
                    handles=handles2,
                    loc="upper left",
                    bbox_to_anchor=(1.05, 1),
                    borderaxespad=0,
                )

        fig.canvas.draw_idle()

    # Connect checkbox callback
    marker_selection.on_clicked(lambda event: update_plot())

    # Add save button
    save_button_ax = plt.axes([0.05, 0.05, 0.15, 0.04])
    save_button = Button(save_button_ax, "Close")

    # Add select all/none buttons
    select_all_ax = plt.axes([0.05, 0.12, 0.07, 0.04])
    select_none_ax = plt.axes([0.13, 0.12, 0.07, 0.04])

    select_all_button = Button(select_all_ax, "All")
    select_none_button = Button(select_none_ax, "None")

    def on_save(event):
        plt.close(fig)  # Close the figure when done

    def select_all(event):
        for i in range(len(all_markers)):
            if not marker_selection.get_status()[i]:
                marker_selection.set_active(i)
        update_plot()

    def select_none(event):
        for i in range(len(all_markers)):
            if marker_selection.get_status()[i]:
                marker_selection.set_active(i)
        update_plot()

    save_button.on_clicked(on_save)
    select_all_button.on_clicked(select_all)
    select_none_button.on_clicked(select_none)

    # Initial plot
    update_plot()

    plt.tight_layout(
        rect=[0.2, 0.25, 0.85, 1]
    )  # Adjust layout to make room for checkboxes and legend
    plt.show()

    # Return which markers were selected when the figure was closed
    return [all_markers[i] for i, checked in enumerate(marker_selection.get_status()) if checked]


def detect_markers_dynamic(df, coord_cols):
    """Detect markers dynamically based on selected coordinate columns."""
    markers = {}

    # Group columns by marker
    for col in coord_cols:
        # Try to extract marker name and coordinate type
        parts = col.split("_")
        if len(parts) >= 2:
            marker_name = "_".join(parts[:-1])  # Everything except last part
            coord_type = parts[-1].lower()  # x, y, z, etc.
        else:
            # If no underscore, try to detect pattern
            # Look for patterns like "p1x", "marker1x", etc.
            import re

            match = re.match(r"([a-zA-Z]+\d+)([xyz])$", col.lower())
            if match:
                marker_name = match.group(1)
                coord_type = match.group(2)
            else:
                # Fallback: use column name as marker name
                marker_name = col
                coord_type = "unknown"

        if marker_name not in markers:
            markers[marker_name] = {}
        markers[marker_name][coord_type] = col

    return markers


def get_marker_coords_dynamic(df, marker_info, coord_types=None):
    """Get coordinates of a marker dynamically based on available coordinate types."""
    if coord_types is None:
        coord_types = ["x", "y", "z"]
    coords = {}
    for coord_type in coord_types:
        if coord_type in marker_info:
            col_name = marker_info[coord_type]
            if col_name in df.columns:
                coords[coord_type] = df[col_name].values
    return coords


def detect_gaps_dynamic(coords_dict, min_gap_size=3):
    """Detect gaps in a marker's trajectory for multiple coordinates."""
    if not coords_dict:
        return []

    # Get the length from any coordinate
    length = len(list(coords_dict.values())[0])
    gaps = []
    current_gap = None

    for i in range(length):
        # Check if any coordinate is missing at this frame
        is_missing = any(pd.isna(coords[i]) or coords[i] == "" for coords in coords_dict.values())

        if is_missing and current_gap is None:
            current_gap = [i, i]
        elif is_missing and current_gap is not None:
            current_gap[1] = i
        elif not is_missing and current_gap is not None:
            if current_gap[1] - current_gap[0] + 1 >= min_gap_size:
                gaps.append(current_gap)
            current_gap = None

    if current_gap is not None and current_gap[1] - current_gap[0] + 1 >= min_gap_size:
        gaps.append(current_gap)

    return gaps


def visualize_markers_dynamic(df, frame_col, coord_cols, marker_ids=None):
    """Visualize marker trajectories for analysis with dynamic coordinate support."""
    if df is None or len(df) == 0:
        messagebox.showerror("Error", "The data file appears to be empty.")
        return

    frames = df[frame_col].values
    markers = detect_markers_dynamic(df, coord_cols)

    if not markers:
        messagebox.showerror("Error", "No markers detected in the selected columns.")
        return

    # Determine dimensions (how many coordinate types we have)
    all_coord_types = set()
    for marker_info in markers.values():
        all_coord_types.update(marker_info.keys())

    coord_types = sorted([ct for ct in ["x", "y", "z"] if ct in all_coord_types])
    n_dims = len(coord_types)

    if n_dims < 2:
        messagebox.showerror("Error", "Need at least 2 coordinate dimensions (x, y).")
        return

    # Create a Tkinter window for marker selection with scrollbar
    marker_select_root = tk.Toplevel()
    marker_select_root.title("Select Markers")
    marker_select_root.geometry("300x400")
    marker_select_root.grab_set()

    marker_names = list(markers.keys())
    marker_vars = {}

    tk.Label(
        marker_select_root,
        text="Select Markers to Display:",
        font=("Arial", 10, "bold"),
    ).pack(pady=5)

    # Create scrollable frame
    canvas = tk.Canvas(marker_select_root, height=300)
    scrollbar = tk.Scrollbar(marker_select_root, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create checkboxes for all markers
    for marker_name in marker_names:
        var = tk.BooleanVar(value=True)
        tk.Checkbutton(scrollable_frame, text=marker_name, variable=var).pack(anchor="w", padx=20)
        marker_vars[marker_name] = var

    canvas.pack(side="left", fill="both", expand=True, padx=(20, 0))
    scrollbar.pack(side="right", fill="y")

    # Add Select All/None buttons
    button_frame = tk.Frame(marker_select_root)
    button_frame.pack(pady=10)

    def select_all():
        for var in marker_vars.values():
            var.set(True)

    def select_none():
        for var in marker_vars.values():
            var.set(False)

    tk.Button(button_frame, text="Select All", command=select_all).pack(side="left", padx=5)
    tk.Button(button_frame, text="Select None", command=select_none).pack(side="left", padx=5)

    # Function to show the plot with selected markers
    selected_markers = []

    def on_ok():
        nonlocal selected_markers
        selected_markers = [name for name, var in marker_vars.items() if var.get()]
        if not selected_markers:
            messagebox.showwarning("Warning", "Please select at least one marker.")
            return
        marker_select_root.destroy()

    def on_cancel():
        marker_select_root.destroy()

    tk.Button(marker_select_root, text="OK", command=on_ok).pack(pady=5)
    tk.Button(marker_select_root, text="Cancel", command=on_cancel).pack(pady=5)

    marker_select_root.wait_window()

    if not selected_markers:  # User cancelled or closed without selecting
        return []

    # Now create the plot with matplotlib
    fig = plt.figure(figsize=(16, 4 * n_dims))
    axes = []

    for i in range(n_dims):
        ax = plt.subplot(n_dims, 1, i + 1)
        axes.append(ax)
        ax.set_title(f"{coord_types[i].upper()} Coordinates of Markers")
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"{coord_types[i].upper()} Position")
        ax.grid(True)

    plt.subplots_adjust(
        left=0.1, bottom=0.15, right=0.95, top=0.95, hspace=0.4
    )  # Ajustar right para usar mais espaço

    # Dictionary to store plot lines
    {coord_type: {} for coord_type in coord_types}

    # Add frame range slider
    frames_slider_ax = plt.axes([0.3, 0.05, 0.5, 0.03])
    frames_range = RangeSlider(frames_slider_ax, "Frames", 0, len(df) - 1, valinit=(0, len(df) - 1))

    # Track current slider values to avoid NameError and enable optional bindings
    start_var = tk.IntVar(value=0)
    end_var = tk.IntVar(value=len(df) - 1)

    def update_plot():
        # Clear current axes
        for ax in axes:
            ax.clear()
            ax.grid(True)

        # Set up axes labels
        for i, coord_type in enumerate(coord_types):
            axes[i].set_title(f"{coord_type.upper()} Coordinates of Markers")
            axes[i].set_xlabel("Frame")
            axes[i].set_ylabel(f"{coord_type.upper()} Position")

        # Plot selected markers (without labels)
        for marker_name in selected_markers:
            marker_info = markers[marker_name]
            coords = get_marker_coords_dynamic(df, marker_info, coord_types)

            for i, coord_type in enumerate(coord_types):
                if coord_type in coords:
                    axes[i].plot(frames, coords[coord_type])  # Removido o label

        # Show frame range with vertical lines
        start_frame, end_frame = frames_range.val
        start_var.set(int(start_frame))
        end_var.set(int(end_frame))

        for ax in axes:
            ax.axvline(start_frame, color="r", linestyle="--")
            ax.axvline(end_frame, color="r", linestyle="--")
            ax.set_xlim(0, len(df) - 1)
            # Removido if visible_markers: ax.legend(loc='upper right')

        fig.canvas.draw_idle()

    # Add close button
    close_button_ax = plt.axes([0.85, 0.05, 0.1, 0.04])
    close_button = Button(close_button_ax, "Close")

    def on_close(event):
        plt.close(fig)

    # Connect callbacks
    frames_range.on_changed(lambda val: update_plot())
    close_button.on_clicked(on_close)

    # Initial plot
    update_plot()
    plt.show()

    return selected_markers


def select_columns_dialog(df):
    """Dialog to select frame and coordinate columns."""
    import tkinter as tk

    root = tk.Toplevel()
    root.title("Select Columns")
    root.geometry("600x600")  # Aumentar largura para acomodar múltiplas colunas
    root.grab_set()

    result = {"cancelled": True}

    # Frame column selection
    tk.Label(root, text="Select Frame Column:", font=("Arial", 12, "bold")).pack(pady=10)
    frame_var = tk.StringVar(value=df.columns[0])
    frame_combo = ttk.Combobox(
        root,
        textvariable=frame_var,
        values=list(df.columns),
        state="readonly",
        width=40,
    )
    frame_combo.pack(pady=5)

    # Coordinate columns selection with scrollbar and multiple columns
    tk.Label(root, text="Select Coordinate Columns:", font=("Arial", 12, "bold")).pack(pady=(20, 5))

    # Create main container with scrollbar
    main_container = tk.Frame(root)
    main_container.pack(fill="both", expand=True, padx=20, pady=5)

    # Create canvas and scrollbar
    canvas = tk.Canvas(main_container, height=350)
    scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create coordinate column checkboxes in multiple columns
    coord_vars = {}
    coordinate_columns = [col for col in df.columns if col != df.columns[0]]  # Skip frame column

    # Calculate number of columns (max 4 columns)
    total_cols = len(coordinate_columns)
    num_columns = min(4, max(1, total_cols // 10))  # At least 1, max 4 columns
    items_per_column = (total_cols + num_columns - 1) // num_columns  # Ceiling division

    # Create grid of checkboxes
    for i, col in enumerate(coordinate_columns):
        row = i % items_per_column
        column = i // items_per_column

        var = tk.BooleanVar()
        # Auto-select if column looks like coordinates
        if any(col.lower().endswith(suffix) for suffix in ["_x", "_y", "_z", "x", "y", "z"]):
            var.set(True)

        checkbox = tk.Checkbutton(
            scrollable_frame,
            text=col,
            variable=var,
            font=("Arial", 9),
            anchor="w",
            width=25,  # Fixed width for consistent column alignment
        )
        checkbox.grid(row=row, column=column, sticky="w", padx=10, pady=1)
        coord_vars[col] = var

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Add Select All/None buttons for coordinates
    coord_buttons_frame = tk.Frame(root)
    coord_buttons_frame.pack(pady=10)

    def select_all_coords():
        for var in coord_vars.values():
            var.set(True)

    def select_none_coords():
        for var in coord_vars.values():
            var.set(False)

    def auto_select_coords():
        """Auto-select columns that look like coordinates"""
        for col, var in coord_vars.items():
            if any(col.lower().endswith(suffix) for suffix in ["_x", "_y", "_z", "x", "y", "z"]):
                var.set(True)
            else:
                var.set(False)

    tk.Button(coord_buttons_frame, text="Select All", command=select_all_coords, width=12).pack(
        side="left", padx=5
    )
    tk.Button(coord_buttons_frame, text="Select None", command=select_none_coords, width=12).pack(
        side="left", padx=5
    )
    tk.Button(coord_buttons_frame, text="Auto Select", command=auto_select_coords, width=12).pack(
        side="left", padx=5
    )

    # Info label
    info_label = tk.Label(
        root,
        text="Auto Select will choose columns ending with _x, _y, _z, x, y, z",
        font=("Arial", 8),
        fg="gray",
    )
    info_label.pack(pady=5)

    # Buttons
    def on_ok():
        selected_coords = [col for col, var in coord_vars.items() if var.get()]
        if len(selected_coords) < 2:
            messagebox.showerror("Error", "Select at least 2 coordinate columns!")
            return
        result.update(
            {
                "cancelled": False,
                "frame_col": frame_var.get(),
                "coord_cols": selected_coords,
            }
        )
        root.destroy()

    def on_cancel():
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.pack(pady=15)
    tk.Button(button_frame, text="OK", command=on_ok, width=15, font=("Arial", 10, "bold")).pack(
        side="left", padx=10
    )
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=15).pack(side="left", padx=10)

    # Enable mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind("<MouseWheel>", on_mousewheel)

    root.wait_window()
    return result


def create_gui_menu():
    """Create main GUI menu for the application."""

    # Print the script version and directory
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = Tk()
    root.title("Marker Re-identification Tool")
    root.geometry("1000x500")  # Aumentar ainda mais a largura e altura

    # Set up main frame with padding
    main_frame = Frame(root, padx=30, pady=30)
    main_frame.pack(expand=True)

    # Title label
    title_label = Label(
        main_frame, text="Marker Re-identification Tool", font=("Arial", 24, "bold")
    )
    title_label.grid(row=0, column=0, pady=(0, 30), columnspan=3)  # 3 colunas agora

    # Description
    desc_label = Label(main_frame, text="Select an option:", font=("Arial", 16))
    desc_label.grid(row=1, column=0, pady=(0, 20), sticky="w", columnspan=3)

    # Buttons for the different options - organized in 3 columns
    btn_width = 25
    btn_height = 3
    font_size = ("Arial", 11)

    # Function to load file and then proceed with selected action
    def load_and_process(action_type):
        """Load file once and then proceed with the selected action"""
        root.destroy()

        # Load file only once
        df, file_path = load_markers_file()
        if df is None:
            return

        # Select columns dialog
        column_config = select_columns_dialog(df)
        if column_config.get("cancelled", False):
            return

        frame_col = column_config["frame_col"]
        coord_cols = column_config["coord_cols"]

        # Proceed with the selected action
        if action_type == "interactive":
            advanced_reid_gui_with_data(df, file_path, frame_col, coord_cols)
        elif action_type == "visualize":
            visualize_markers_dynamic(df, frame_col, coord_cols)
        elif action_type == "arima":
            auto_fill_gaps_arima_with_data(df, file_path)
        elif action_type == "reidswap_auto":
            run_reid_swap_auto_with_data(df, file_path)
        elif action_type == "reidswap_manual":
            run_reid_swap_manual_with_data(df, file_path)

    # Row 1 - Main functions
    option1_btn = TkButton(
        main_frame,
        text="Interactive\nRe-identification",
        width=btn_width,
        height=btn_height,
        font=font_size,
        command=lambda: load_and_process("interactive"),
    )
    option1_btn.grid(row=2, column=0, padx=15, pady=15)

    option2_btn = TkButton(
        main_frame,
        text="Auto Fill Gaps\n(ARIMA)",
        width=btn_width,
        height=btn_height,
        font=font_size,
        command=lambda: load_and_process("arima"),
    )
    option2_btn.grid(row=2, column=1, padx=15, pady=15)

    option3_btn = TkButton(
        main_frame,
        text="Visualize\nMarkers",
        width=btn_width,
        height=btn_height,
        font=font_size,
        command=lambda: load_and_process("visualize"),
    )
    option3_btn.grid(row=2, column=2, padx=15, pady=15)

    # Row 2 - Additional functions
    option4_btn = TkButton(
        main_frame,
        text="Auto Swap L/R\n(reidmplrswap)",
        width=btn_width,
        height=btn_height,
        font=font_size,
        command=lambda: load_and_process("reidswap_auto"),
    )
    option4_btn.grid(row=3, column=0, padx=15, pady=15)

    option5_btn = TkButton(
        main_frame,
        text="Manual Swap L/R\n(reidmplrswap)",
        width=btn_width,
        height=btn_height,
        font=font_size,
        command=lambda: load_and_process("reidswap_manual"),
    )
    option5_btn.grid(row=3, column=1, padx=15, pady=15)

    option6_btn = TkButton(
        main_frame,
        text="Settings\n(Coming Soon)",
        width=btn_width,
        height=btn_height,
        font=font_size,
        state="disabled",  # Disabled for now
        command=lambda: print("Settings selected"),
    )
    option6_btn.grid(row=3, column=2, padx=15, pady=15)

    # Exit button - spanning all columns
    exit_btn = TkButton(
        main_frame,
        text="Exit",
        width=btn_width,
        height=2,
        font=font_size,
        command=root.destroy,
    )
    exit_btn.grid(row=4, column=0, columnspan=3, pady=30)

    # Version and info
    info_frame = Frame(main_frame)
    info_frame.grid(row=5, column=0, columnspan=3, pady=(20, 0))

    version_label = Label(info_frame, text="Version 0.1.0", font=("Arial", 10))
    version_label.pack()

    author_label = Label(info_frame, text="Paulo R. P. Santiago", font=("Arial", 9), fg="gray")
    author_label.pack()

    root.mainloop()


def run_reid_swap_auto_with_data(df, file_path):
    """Run automatic L/R swap detection using reidmplrswap on the provided df."""
    try:
        # Reuse logic from reidmplrswap: detect L/R pairs from columns
        from vaila.reidmplrswap import (
            auto_fix_swaps,
            find_lr_pairs,
            save_csv_with_suffix,
        )

        pairs = find_lr_pairs(df)
        if not pairs:
            messagebox.showinfo("Info", "No L/R pairs detected in the selected columns.")
            return
        proposals = auto_fix_swaps(df, pairs, max_len=30, min_gap=1)
        out_csv = save_csv_with_suffix(Path(file_path), df, suffix="_reidswap")
        # Also write a quick report
        from vaila.reidmplrswap import write_report

        write_report(Path(file_path), proposals)
        messagebox.showinfo("Done", f"Auto swaps applied. Saved: {out_csv}")
    except Exception as exc:
        messagebox.showerror("Error", f"Auto swap failed: {exc}")


def run_reid_swap_manual_with_data(df, file_path):
    """Prompt for a pair base name and frame range, and swap L/R using reidmplrswap."""
    try:
        from vaila.reidmplrswap import (
            apply_swap_for_pair,
            find_lr_pairs,
            save_csv_with_suffix,
        )

        pairs = find_lr_pairs(df)
        if not pairs:
            messagebox.showinfo("Info", "No L/R pairs detected in the selected columns.")
            return
        # Ask for pair (base name)
        base_names = ", ".join(sorted({p.base_name for p in pairs}))
        pair_name = simpledialog.askstring(
            "Manual Swap", f"Enter pair base name (options: {base_names})"
        )
        if not pair_name:
            return
        # Find target pair (case-insensitive, allow partial)
        target = None
        for p in pairs:
            if p.base_name.lower() == pair_name.lower() or pair_name.lower() in p.base_name.lower():
                target = p
                break
        if target is None:
            messagebox.showerror("Error", f"Pair '{pair_name}' not found.")
            return
        # Choose method: graphical or typing
        use_graphical = messagebox.askyesno(
            "Selection Method",
            "Would you like to select the frame range graphically on X/Y plots?\nYes: graphical selection\nNo: type start/end",
        )

        if use_graphical:
            import matplotlib.pyplot as plt

            frames = df[df.columns[0]].values if len(df.columns) > 0 else np.arange(len(df))
            colsLx = target.left.cols.get("x")
            colsLy = target.left.cols.get("y")
            colsRx = target.right.cols.get("x")
            colsRy = target.right.cols.get("y")
            if not colsLx or not colsLy or not colsRx or not colsRy:
                messagebox.showerror("Error", "Selected pair does not have x/y columns.")
                return
            Lx = df[colsLx].values
            Ly = df[colsLy].values
            Rx = df[colsRx].values
            Ry = df[colsRy].values

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            fig.subplots_adjust(bottom=0.22, top=0.92, hspace=0.25)
            ax1.set_title(f"{target.base_name} - X (L/R)")
            ax1.plot(frames, Lx, color="g", label="Lx")
            ax1.plot(frames, Rx, color="orange", label="Rx")
            ax1.legend(loc="upper right")
            ax1.grid(True)
            ax2.set_title(f"{target.base_name} - Y (L/R)")
            ax2.plot(frames, Ly, color="g", label="Ly")
            ax2.plot(frames, Ry, color="orange", label="Ry")
            ax2.legend(loc="upper right")
            ax2.grid(True)

            start_ax = plt.axes([0.10, 0.08, 0.12, 0.05])
            end_ax = plt.axes([0.25, 0.08, 0.12, 0.05])
            apply_ax = plt.axes([0.41, 0.08, 0.12, 0.05])
            cancel_ax = plt.axes([0.56, 0.08, 0.12, 0.05])
            start_tb = TextBox(start_ax, "Start", initial="0")
            end_tb = TextBox(end_ax, "End", initial=str(len(df) - 1))
            apply_bt = Button(apply_ax, "Apply Swap")
            cancel_bt = Button(cancel_ax, "Cancel")

            v1 = ax1.axvline(0, color="r", linestyle="--")
            v2 = ax1.axvline(len(df) - 1, color="r", linestyle="--")
            w1 = ax2.axvline(0, color="r", linestyle="--")
            w2 = ax2.axvline(len(df) - 1, color="r", linestyle="--")
            sel = {"start": 0, "end": len(df) - 1}

            def _update_lines():
                s = int(max(0, min(len(df) - 1, sel["start"])))
                e = int(max(0, min(len(df) - 1, sel["end"])))
                v1.set_xdata([s, s])
                v2.set_xdata([e, e])
                w1.set_xdata([s, s])
                w2.set_xdata([e, e])
                fig.canvas.draw_idle()

            def on_select(xmin, xmax):
                s = int(round(min(xmin, xmax)))
                e = int(round(max(xmin, xmax)))
                s = max(0, min(len(df) - 1, s))
                e = max(0, min(len(df) - 1, e))
                sel["start"] = s
                sel["end"] = e
                start_tb.set_val(str(s))
                end_tb.set_val(str(e))
                _update_lines()

            # Use a Matplotlib-compatible signature across versions (no span_stays/interactive)
            SpanSelector(ax2, on_select, "horizontal", useblit=True)

            def on_start_submit(text):
                try:
                    sel["start"] = int(text)
                    _update_lines()
                except Exception:
                    pass

            def on_end_submit(text):
                try:
                    sel["end"] = int(text)
                    _update_lines()
                except Exception:
                    pass

            start_tb.on_submit(on_start_submit)
            end_tb.on_submit(on_end_submit)

            result = {"done": False}

            def do_apply(event):
                s = int(min(sel["start"], sel["end"]))
                e = int(max(sel["start"], sel["end"]))
                if e <= s:
                    return
                apply_swap_for_pair(df, target, s, e)
                out_csv = save_csv_with_suffix(Path(file_path), df, suffix="_reidswap")
                print(f"Manual swap applied for '{target.base_name}' {s}-{e}. Saved: {out_csv}")
                result["done"] = True
                plt.close(fig)

            def do_cancel(event):
                plt.close(fig)

            apply_bt.on_clicked(do_apply)
            cancel_bt.on_clicked(do_cancel)
            plt.show()
            if not result["done"]:
                return
            messagebox.showinfo("Done", f"Manual swap applied for '{target.base_name}'.")
        else:
            start = simpledialog.askinteger(
                "Manual Swap", "Start frame:", minvalue=0, maxvalue=len(df) - 1
            )
            end = simpledialog.askinteger(
                "Manual Swap", "End frame:", minvalue=0, maxvalue=len(df) - 1
            )
            if start is None or end is None or end <= start:
                messagebox.showerror("Error", "Invalid frame range.")
                return
            apply_swap_for_pair(df, target, start, end)
            out_csv = save_csv_with_suffix(Path(file_path), df, suffix="_reidswap")
            messagebox.showinfo(
                "Done",
                f"Manual swap applied for '{target.base_name}' {start}-{end}. Saved: {out_csv}",
            )
    except Exception as exc:
        messagebox.showerror("Error", f"Manual swap failed: {exc}")


def advanced_reid_gui_with_data(df, file_path, frame_col, coord_cols):
    """Advanced re-identification GUI with pre-loaded data."""
    if len(df.columns) == 0 or len(df) == 0:
        messagebox.showerror("Error", "The data file appears to be empty or malformed.")
        return

    frames = df[frame_col].values

    # Detect markers dynamically
    markers = detect_markers_dynamic(df, coord_cols)
    marker_names = list(markers.keys())

    # Determine coordinate types available
    all_coord_types = set()
    for marker_info in markers.values():
        all_coord_types.update(marker_info.keys())
    coord_types = sorted([ct for ct in ["x", "y", "z"] if ct in all_coord_types])
    n_dims = len(coord_types)

    # Add variables for tracking changes
    original_df = df.copy()
    df_history = [df.copy()]

    # Set up the interface with dynamic subplot configuration
    fig = plt.figure(figsize=(20, 4 * n_dims + 6))  # Aumentar a largura e altura
    axes = []

    for i in range(n_dims):
        ax = plt.subplot(n_dims, 1, i + 1)
        axes.append(ax)
        ax.set_title(f"{coord_types[i].upper()} Coordinates of Markers")
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"{coord_types[i].upper()} Position")
        ax.grid(True)

    plt.subplots_adjust(left=0.2, bottom=0.3, right=0.85, top=0.95, hspace=0.4)

    # Create a Tkinter window for marker selection with scrollbar
    marker_select_root = tk.Toplevel()
    marker_select_root.title("Select Markers")
    marker_select_root.geometry("400x700")  # Aumentar largura

    marker_vars = {}

    tk.Label(marker_select_root, text="Select Markers:", font=("Arial", 12, "bold")).pack(pady=5)

    # Create scrollable frame with multiple columns for markers
    main_container = tk.Frame(marker_select_root)
    main_container.pack(fill="both", expand=True, padx=20, pady=5)

    canvas = tk.Canvas(main_container, height=400)
    scrollbar = tk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Organize markers in multiple columns (max 3 columns)
    total_markers = len(marker_names)
    num_columns = min(3, max(1, total_markers // 15))  # At least 1, max 3 columns
    items_per_column = (total_markers + num_columns - 1) // num_columns

    # Create checkboxes for all markers in grid layout
    for i, marker_name in enumerate(marker_names):
        row = i % items_per_column
        column = i // items_per_column

        var = tk.BooleanVar(value=True)
        checkbox = tk.Checkbutton(
            scrollable_frame,
            text=marker_name,
            variable=var,
            font=("Arial", 9),
            anchor="w",
            width=20,
        )
        checkbox.grid(row=row, column=column, sticky="w", padx=10, pady=1)
        marker_vars[marker_name] = var

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Add Select All/None buttons
    button_frame = tk.Frame(marker_select_root)
    button_frame.pack(pady=10)

    def select_all():
        for var in marker_vars.values():
            var.set(True)
        update_plot()

    def select_none():
        for var in marker_vars.values():
            var.set(False)
        update_plot()

    tk.Button(button_frame, text="Select All", command=select_all, width=12).pack(
        side="left", padx=5
    )
    tk.Button(button_frame, text="Select None", command=select_none, width=12).pack(
        side="left", padx=5
    )

    # Add controls in multiple columns
    controls_frame = tk.Frame(marker_select_root)
    controls_frame.pack(pady=10, fill="x")

    # Organize buttons in a grid (2 columns)
    tk.Button(controls_frame, text="Fill Gaps", command=lambda: on_fill_gaps(), width=12).grid(
        row=0, column=0, padx=5, pady=5
    )
    tk.Button(
        controls_frame,
        text="Merge Markers",
        command=lambda: on_merge_markers(),
        width=12,
    ).grid(row=0, column=1, padx=5, pady=5)
    tk.Button(
        controls_frame, text="Swap Markers", command=lambda: on_swap_markers(), width=12
    ).grid(row=1, column=0, padx=5, pady=5)
    tk.Button(
        controls_frame,
        text="Delete Marker",
        command=lambda: on_delete_marker(),
        width=12,
    ).grid(row=1, column=1, padx=5, pady=5)

    # Frame range selection with Tkinter
    range_frame = tk.Frame(marker_select_root)
    range_frame.pack(pady=10, fill="x")

    tk.Label(range_frame, text="Frame Range:", font=("Arial", 10, "bold")).pack()

    range_entry_frame = tk.Frame(range_frame)
    range_entry_frame.pack(pady=5)

    tk.Label(range_entry_frame, text="Start:").grid(row=0, column=0, padx=5)
    start_var = tk.IntVar(value=0)
    start_entry = tk.Entry(range_entry_frame, textvariable=start_var, width=8)
    start_entry.grid(row=0, column=1, padx=5)

    tk.Label(range_entry_frame, text="End:").grid(row=0, column=2, padx=5)
    end_var = tk.IntVar(value=len(df) - 1)
    end_entry = tk.Entry(range_entry_frame, textvariable=end_var, width=8)
    end_entry.grid(row=0, column=3, padx=5)

    def update_range():
        try:
            start = start_var.get()
            end = end_var.get()
            if 0 <= start < end <= len(df):
                frames_range.set_val((start, end))
                update_plot()
            else:
                messagebox.showerror("Error", f"Range must be between 0 and {len(df) - 1}")
        except:
            messagebox.showerror("Error", "Please enter valid numbers")

    tk.Button(range_entry_frame, text="Apply", command=update_range).grid(row=0, column=4, padx=5)

    # Save and exit buttons
    save_frame = tk.Frame(marker_select_root)
    save_frame.pack(pady=20)

    tk.Button(save_frame, text="Save Changes", command=lambda: on_save(), width=15).pack(
        side="left", padx=10
    )
    tk.Button(save_frame, text="Exit", command=lambda: on_close(), width=15).pack(
        side="left", padx=10
    )

    # Create sliders for the frame range in matplotlib
    frames_slider_ax = plt.axes([0.3, 0.15, 0.5, 0.03])
    frames_range = RangeSlider(frames_slider_ax, "Frames", 0, len(df) - 1, valinit=(0, len(df) - 1))

    # Function to get selected markers
    def get_selected_markers():
        return [name for name, var in marker_vars.items() if var.get()]

    # Define the update plot function
    def update_plot():
        # Get selected markers
        visible_markers = get_selected_markers()

        # Clear current axes
        for ax in axes:
            ax.clear()
            ax.grid(True)

        # Set up axes labels
        for i, coord_type in enumerate(coord_types):
            axes[i].set_title(f"{coord_type.upper()} Coordinates of Markers")
            axes[i].set_xlabel("Frame")
            axes[i].set_ylabel(f"{coord_type.upper()} Position")

        # Plot only selected markers (without labels)
        for marker_name in visible_markers:
            marker_info = markers[marker_name]
            coords = get_marker_coords_dynamic(df, marker_info, coord_types)

            for i, coord_type in enumerate(coord_types):
                if coord_type in coords:
                    axes[i].plot(frames, coords[coord_type])  # Removido o label

        # Show frame range with vertical lines
        start_frame, end_frame = frames_range.val
        start_var.set(int(start_frame))
        end_var.set(int(end_frame))

        for ax in axes:
            ax.axvline(start_frame, color="r", linestyle="--")
            ax.axvline(end_frame, color="r", linestyle="--")
            ax.set_xlim(0, len(df) - 1)
            # Removido if visible_markers: ax.legend(loc='upper right')

        fig.canvas.draw_idle()

    # Define callback functions
    def on_fill_gaps():
        selected_markers = get_selected_markers()
        if not selected_markers:
            messagebox.showinfo("Info", "Please select at least one marker to fill gaps.")
            return

        # Save current state to history
        df_history.append(df.copy())

        # Get frame range
        start_frame, end_frame = [int(val) for val in frames_range.val]

        # Fill gaps for each selected marker
        for marker_name in selected_markers:
            marker_info = markers[marker_name]
            for coord_type in coord_types:
                if coord_type in marker_info:
                    col_name = marker_info[coord_type]
                    values = df[col_name].values.copy()

                    # Simple linear interpolation on the selected range
                    values_range = values[start_frame : end_frame + 1]
                    interpolated = pd.Series(values_range).interpolate(method="linear").values
                    values[start_frame : end_frame + 1] = interpolated
                    df[col_name] = values

        update_plot()
        messagebox.showinfo("Complete", f"Gaps filled for {len(selected_markers)} marker(s).")

    def on_merge_markers():
        # TODO: Implement marker merging functionality
        messagebox.showinfo("Info", "Merge Markers function not yet implemented.")

    def on_swap_markers():
        # Manual swap within GUI for two selected markers using start/end
        visible_markers = get_selected_markers()
        if len(visible_markers) != 2:
            messagebox.showinfo("Info", "Select exactly two markers to swap.")
            return
        start_frame, end_frame = [int(v) for v in frames_range.val]
        if end_frame <= start_frame:
            messagebox.showerror("Error", "Invalid frame range.")
            return
        # Perform swap across all available coords
        m1, m2 = visible_markers
        for coord in coord_types:
            c1 = markers[m1].get(coord)
            c2 = markers[m2].get(coord)
            if c1 and c2 and c1 in df.columns and c2 in df.columns:
                tmp = df.loc[start_frame:end_frame, c1].copy()
                df.loc[start_frame:end_frame, c1] = df.loc[start_frame:end_frame, c2].values
                df.loc[start_frame:end_frame, c2] = tmp.values
        update_plot()
        messagebox.showinfo("Done", f"Swapped {m1} and {m2} in {start_frame}-{end_frame}.")

    def on_delete_marker():
        # TODO: Implement marker deletion functionality
        messagebox.showinfo("Info", "Delete Marker function not yet implemented.")

    def on_save():
        if not messagebox.askyesno("Save Changes", "Save changes to a new file?"):
            return

        new_file = save_markers_file(df, file_path)
        messagebox.showinfo("Success", f"Changes saved to: {new_file}")

    def on_close():
        if df.equals(original_df) or messagebox.askyesno(
            "Discard Changes", "Discard changes and exit?"
        ):
            marker_select_root.destroy()
            plt.close(fig)

    # Connect matplotlib callbacks
    frames_range.on_changed(lambda val: update_plot())

    # Enable mouse wheel scrolling
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind("<MouseWheel>", on_mousewheel)

    # Set callback for window close
    marker_select_root.protocol("WM_DELETE_WINDOW", on_close)

    # Initial plot
    update_plot()
    plt.show()

    # This will keep the Tkinter window running alongside matplotlib
    marker_select_root.mainloop()


def auto_fill_gaps_arima_with_data(df, file_path):
    """ARIMA gap filling with pre-loaded data."""
    # Use the existing auto_fill_gaps_arima but modify to accept data
    # For now, just call the original function
    auto_fill_gaps_arima()


def fill_gaps_arima(df, marker_id, max_gap_size=30, order=(1, 1, 1)):
    """
    Fill gaps in a marker's trajectory using ARIMA modeling.

    Parameters:
        df: DataFrame containing marker data
        marker_id: ID of the marker to process
        max_gap_size: Maximum size of gaps to fill
        order: ARIMA model order (p,d,q)
               p: autoregressive order
               d: differencing order
               q: moving average order
    """
    x_col = f"p{marker_id}_x"
    y_col = f"p{marker_id}_y"

    if x_col not in df.columns or y_col not in df.columns:
        return df, 0

    x_values = df[x_col].values.copy()
    y_values = df[y_col].values.copy()

    gaps = detect_gaps(x_values, y_values)
    filled_gaps = 0
    total_points_filled = 0

    # Skip if no gaps found
    if not gaps:
        return df, 0

    # Create copies of the arrays for processing
    x_filled = x_values.copy()
    y_filled = y_values.copy()

    # First pass: identify valid segments (continuous data points)
    valid_indices = ~(pd.isna(x_values) | pd.isna(y_values))

    # If there are fewer than 10 valid points, fall back to linear interpolation
    if np.sum(valid_indices) < 10:
        print(f"Too few valid points for marker {marker_id}, falling back to linear interpolation")
        return fill_gaps(df, marker_id, max_gap_size, method="linear"), 0

    # Process each gap
    for gap_start, gap_end in gaps:
        gap_size = gap_end - gap_start + 1

        if gap_size > max_gap_size:
            print(f"Gap too large for marker {marker_id} ({gap_size} frames), skipping")
            continue

        # Find valid points before and after
        before_segment = []
        after_segment = []

        # Collect up to 20 valid points before the gap
        i = gap_start - 1
        count = 0
        while i >= 0 and count < 20:
            if not pd.isna(x_values[i]) and not pd.isna(y_values[i]):
                before_segment.insert(0, (x_values[i], y_values[i]))
                count += 1
            i -= 1

        # Collect up to 20 valid points after the gap
        i = gap_end + 1
        count = 0
        while i < len(x_values) and count < 20:
            if not pd.isna(x_values[i]) and not pd.isna(y_values[i]):
                after_segment.append((x_values[i], y_values[i]))
                count += 1
            i += 1

        # Skip if not enough context
        if len(before_segment) < 3 or len(after_segment) < 3:
            print(
                f"Not enough context around gap for marker {marker_id} (frames {gap_start}-{gap_end})"
            )
            continue

        # Combine segments for ARIMA modeling
        full_segment_x = (
            [p[0] for p in before_segment] + [np.nan] * gap_size + [p[0] for p in after_segment]
        )
        full_segment_y = (
            [p[1] for p in before_segment] + [np.nan] * gap_size + [p[1] for p in after_segment]
        )

        # Create pandas series for ARIMA
        full_segment_x_series = pd.Series(full_segment_x)
        full_segment_y_series = pd.Series(full_segment_y)

        try:
            # Handle X values with ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress convergence warnings

                # Fill X values
                valid_x = ~pd.isna(full_segment_x_series)
                if sum(valid_x) > 10:  # Need enough data for ARIMA
                    model_x = ARIMA(full_segment_x_series[valid_x], order=order)
                    results_x = model_x.fit()

                    # Apply Kalman filter for better prediction
                    smooth_x = results_x.filter()

                    # Create a forecast for the missing values
                    forecast_x = smooth_x.get_forecast(steps=gap_size)
                    forecast_values_x = forecast_x.predicted_mean

                    # Fill in the gap
                    for i, idx in enumerate(range(gap_start, gap_end + 1)):
                        if i < len(forecast_values_x):
                            x_filled[idx] = forecast_values_x[i]

                # Fill Y values
                valid_y = ~pd.isna(full_segment_y_series)
                if sum(valid_y) > 10:  # Need enough data for ARIMA
                    model_y = ARIMA(full_segment_y_series[valid_y], order=order)
                    results_y = model_y.fit()

                    # Apply Kalman filter for better prediction
                    smooth_y = results_y.filter()

                    # Create a forecast for the missing values
                    forecast_y = smooth_y.get_forecast(steps=gap_size)
                    forecast_values_y = forecast_y.predicted_mean

                    # Fill in the gap
                    for i, idx in enumerate(range(gap_start, gap_end + 1)):
                        if i < len(forecast_values_y):
                            y_filled[idx] = forecast_values_y[i]

            filled_gaps += 1
            total_points_filled += gap_size
            print(
                f"Successfully filled gap for marker {marker_id} using ARIMA (frames {gap_start}-{gap_end})"
            )

        except Exception as e:
            print(
                f"ARIMA failed for marker {marker_id}: {str(e)}. Falling back to linear interpolation."
            )
            # Fall back to linear interpolation for this gap
            before_idx = gap_start - 1
            after_idx = gap_end + 1

            while before_idx >= 0 and (
                pd.isna(x_values[before_idx]) or pd.isna(y_values[before_idx])
            ):
                before_idx -= 1

            while after_idx < len(x_values) and (
                pd.isna(x_values[after_idx]) or pd.isna(y_values[after_idx])
            ):
                after_idx += 1

            if before_idx >= 0 and after_idx < len(x_values):
                frames = [before_idx, after_idx]
                x_known = [x_values[before_idx], x_values[after_idx]]
                y_known = [y_values[before_idx], y_values[after_idx]]

                x_interp = np.interp(range(gap_start, gap_end + 1), frames, x_known)
                y_interp = np.interp(range(gap_start, gap_end + 1), frames, y_known)

                for i, idx in enumerate(range(gap_start, gap_end + 1)):
                    x_filled[idx] = x_interp[i]
                    y_filled[idx] = y_interp[i]

                filled_gaps += 1
                total_points_filled += gap_size

    if filled_gaps > 0:
        df[x_col] = x_filled
        df[y_col] = y_filled
        print(
            f"Filled {filled_gaps} gaps ({total_points_filled} points) for marker {marker_id} using ARIMA"
        )

    return df, total_points_filled


def auto_fill_gaps_arima():
    """Automatically fill gaps for all markers using the ARIMA model."""
    df, file_path = load_markers_file()
    if df is None:
        return

    if len(df.columns) == 0 or len(df) == 0:
        messagebox.showerror("Error", "The data file appears to be empty or malformed.")
        return

    frame_col = df.columns[0]
    df[frame_col].values

    markers = detect_markers(df)
    total_filled = 0
    total_points_filled = 0

    # Add metadata column to track which markers had gaps filled
    if "processed_info" not in df.columns:
        df["processed_info"] = ""

    # Default values - these will be used as initial values in the UI
    default_p = 1
    default_d = 1
    default_q = 1
    default_max_gap = 20

    # Initialize arima_params with defaults to ensure they exist even if window is closed
    arima_params = {
        "p": default_p,
        "d": default_d,
        "q": default_q,
        "max_gap": default_max_gap,
    }

    # Create parameters window with enhanced instructions
    arima_params_window = Tk()
    arima_params_window.title("ARIMA Model Parameters")
    arima_params_window.geometry("450x400")

    # Parameter descriptions and help text (hidden initially)
    help_text = {
        "p": "Autoregressive Order:\n• Controls how previous values influence current value\n• Higher values capture longer-term patterns\n• Recommended: 1-3 (start with 1)",
        "d": "Differencing Order:\n• Makes data stationary by removing trends\n• 0: No differencing, 1: Remove linear trend, 2: Remove quadratic trend\n• Recommended: 0-1 for marker trajectories",
        "q": "Moving Average Order:\n• Incorporates error terms from previous predictions\n• Higher values smooth out irregular patterns\n• Recommended: 1-2 (start with 1)",
        "max_gap": "Maximum number of consecutive frames to fill\n• Larger gaps are less reliable to fill\n• Recommended: 15-30 frames",
    }

    # Header label
    header_label = Label(
        arima_params_window, text="ARIMA Model Parameters", font=("Arial", 12, "bold")
    )
    header_label.pack(pady=(15, 10))

    # Create frame for parameter inputs
    params_frame = Frame(arima_params_window)
    params_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # Help popup function
    def show_help_popup(param):
        help_window = tk.Toplevel(arima_params_window)
        help_window.title(f"Help for {param}")
        help_window.geometry("350x200")
        help_window.transient(arima_params_window)
        help_window.grab_set()

        # Make window appear near the button
        help_window.geometry(
            f"+{arima_params_window.winfo_rootx() + 100}+{arima_params_window.winfo_rooty() + 100}"
        )

        help_label = Label(help_window, text=help_text[param], justify="left", padx=15, pady=15)
        help_label.pack(fill="both", expand=True)

        ok_button = TkButton(help_window, text="OK", command=help_window.destroy)
        ok_button.pack(pady=10)

    # ARIMA p parameter row
    p_frame = Frame(params_frame)
    p_frame.pack(fill="x", pady=5)
    Label(p_frame, text="p (AR):", width=10, anchor="w").pack(side=tk.LEFT, padx=10)
    p_var = tk.StringVar(value=str(default_p))
    p_entry = tk.Entry(p_frame, textvariable=p_var, width=5)
    p_entry.pack(side=tk.LEFT, padx=5)
    Label(p_frame, text=f"Default: {default_p}", width=10).pack(side=tk.LEFT)
    p_help_btn = TkButton(p_frame, text="?", width=2, command=lambda: show_help_popup("p"))
    p_help_btn.pack(side=tk.LEFT)

    # ARIMA d parameter row
    d_frame = Frame(params_frame)
    d_frame.pack(fill="x", pady=5)
    Label(d_frame, text="d (Diff):", width=10, anchor="w").pack(side=tk.LEFT)
    d_var = tk.StringVar(value=str(default_d))
    d_entry = tk.Entry(d_frame, textvariable=d_var, width=5)
    d_entry.pack(side=tk.LEFT, padx=5)
    Label(d_frame, text=f"Default: {default_d}", width=10).pack(side=tk.LEFT)
    d_help_btn = TkButton(d_frame, text="?", width=2, command=lambda: show_help_popup("d"))
    d_help_btn.pack(side=tk.LEFT)

    # ARIMA q parameter row
    q_frame = Frame(params_frame)
    q_frame.pack(fill="x", pady=5)
    Label(q_frame, text="q (MA):", width=10, anchor="w").pack(side=tk.LEFT)
    q_var = tk.StringVar(value=str(default_q))
    q_entry = tk.Entry(q_frame, textvariable=q_var, width=5)
    q_entry.pack(side=tk.LEFT, padx=5)
    Label(q_frame, text=f"Default: {default_q}", width=10).pack(side=tk.LEFT)
    q_help_btn = TkButton(q_frame, text="?", width=2, command=lambda: show_help_popup("q"))
    q_help_btn.pack(side=tk.LEFT)

    # Max gap size parameter row
    max_gap_frame = Frame(params_frame)
    max_gap_frame.pack(fill="x", pady=5)
    Label(max_gap_frame, text="Max Gap Size:", width=10, anchor="w").pack(side=tk.LEFT)
    max_gap_var = tk.StringVar(value=str(default_max_gap))
    max_gap_entry = tk.Entry(max_gap_frame, textvariable=max_gap_var, width=5)
    max_gap_entry.pack(side=tk.LEFT, padx=5)
    Label(max_gap_frame, text=f"Default: {default_max_gap}", width=10).pack(side=tk.LEFT)
    max_gap_help_btn = TkButton(
        max_gap_frame, text="?", width=2, command=lambda: show_help_popup("max_gap")
    )
    max_gap_help_btn.pack(side=tk.LEFT)

    # Separator
    separator = Frame(arima_params_window, height=2, bd=1, relief=tk.SUNKEN)
    separator.pack(fill="x", padx=20, pady=15)

    # Processing status
    status_var = tk.StringVar(value="")
    status_label = Label(arima_params_window, textvariable=status_var, font=("Arial", 9), fg="blue")
    status_label.pack(pady=5)

    def reset_to_defaults():
        """Reset all parameters to defaults"""
        p_var.set(str(default_p))
        d_var.set(str(default_d))
        q_var.set(str(default_q))
        max_gap_var.set(str(default_max_gap))

    def start_processing():
        """Starts the actual ARIMA processing"""
        status_var.set("Processing started. This may take a few minutes...")
        arima_params_window.update_idletasks()  # Force GUI update

        # Process each marker with ARIMA
        nonlocal total_filled, total_points_filled
        for marker_id in markers:
            df_updated, points_filled = fill_gaps_arima(
                df,
                marker_id,
                max_gap_size=arima_params["max_gap"],
                order=(arima_params["p"], arima_params["d"], arima_params["q"]),
            )

            # Update the dataframe with the filled values
            df.update(df_updated)

            if points_filled > 0:
                total_filled += 1
                total_points_filled += points_filled
                # Update the metadata to include information about filled gaps
                df.loc[0, "processed_info"] += (
                    f"ARIMA filled {points_filled} points for marker {marker_id}; "
                )

            # Update status to show progress
            status_var.set(
                f"Processing marker {marker_id}... ({markers.index(marker_id) + 1}/{len(markers)})"
            )
            arima_params_window.update_idletasks()  # Force GUI update

        # Add timestamp and processing info
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.loc[0, "processed_info"] += (
            f"ARIMA processed on {timestamp} with parameters (p={arima_params['p']},d={arima_params['d']},q={arima_params['q']})"
        )

        # Save with a more descriptive suffix that includes parameters
        param_text = f"p{arima_params['p']}d{arima_params['d']}q{arima_params['q']}"
        new_file = save_markers_file(df, file_path, suffix=f"_arima_{param_text}_{timestamp}")

        # Create operations log
        operations_log = {
            "operation": "auto_fill_gaps_arima",
            "timestamp": timestamp,
            "arima_parameters": {
                "p": arima_params["p"],
                "d": arima_params["d"],
                "q": arima_params["q"],
                "max_gap_size": arima_params["max_gap"],
            },
            "markers_processed": markers,
            "markers_with_changes": total_filled,
            "total_points_filled": total_points_filled,
        }

        save_operations_log(operations_log, new_file)

        status_var.set("Processing complete! Saved to: " + os.path.basename(new_file))
        arima_params_window.update_idletasks()  # Force GUI update

        # Show confirmation with details after a short delay
        arima_params_window.after(
            1000,
            lambda: [
                arima_params_window.destroy(),
                messagebox.showinfo(
                    "ARIMA Gap Filling Complete",
                    f"All gaps have been processed using ARIMA({arima_params['p']},{arima_params['d']},{arima_params['q']}).\n\n"
                    f"{total_filled} out of {len(markers)} markers had gaps filled.\n"
                    f"Total of {total_points_filled} data points were filled.\n"
                    f"Results saved to: {os.path.basename(new_file)}",
                ),
                (
                    lambda: (
                        visualize_markers(df)
                        if messagebox.askyesno(
                            "View Results", "Would you like to visualize the results?"
                        )
                        else None
                    )
                )(),
            ],
        )

    def on_confirm():
        try:
            # Get parameters from UI
            arima_params["p"] = int(p_var.get())
            arima_params["d"] = int(d_var.get())
            arima_params["q"] = int(q_var.get())
            arima_params["max_gap"] = int(max_gap_var.get())

            # Disable all inputs during processing
            p_entry.config(state="disabled")
            d_entry.config(state="disabled")
            q_entry.config(state="disabled")
            max_gap_entry.config(state="disabled")
            confirm_button.config(state="disabled")
            reset_button.config(state="disabled")
            help_all_button.config(state="disabled")

            # Start processing in the same window (don't destroy it yet)
            start_processing()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers for all parameters")

    # Button frame
    button_frame = Frame(arima_params_window)
    button_frame.pack(pady=10)

    # Add a reset button
    reset_button = TkButton(button_frame, text="Reset to Defaults", command=reset_to_defaults)
    reset_button.pack(side=tk.LEFT, padx=10)

    # Add confirm button with a more descriptive label
    confirm_button = TkButton(button_frame, text="Start Processing", command=on_confirm)
    confirm_button.pack(side=tk.LEFT, padx=10)

    # Show all help text in one window
    def show_all_help():
        help_window = tk.Toplevel(arima_params_window)
        help_window.title("ARIMA Parameters Help")
        help_window.geometry("450x400")
        help_window.transient(arima_params_window)
        help_window.grab_set()

        # Make window appear centered relative to main window
        help_window.geometry(
            f"+{arima_params_window.winfo_rootx() + 50}+{arima_params_window.winfo_rooty() + 50}"
        )

        # Title
        Label(help_window, text="ARIMA Parameter Guide", font=("Arial", 12, "bold")).pack(
            pady=(10, 15)
        )

        # Create scrollable frame for help text
        canvas = tk.Canvas(help_window)
        scrollbar = tk.Scrollbar(help_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y", padx=(0, 10))

        # Add parameter descriptions
        Label(
            scrollable_frame,
            text="p (AR) - Autoregressive Order:",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(5, 0), padx=10)
        Label(
            scrollable_frame,
            text="• Controls how previous values influence current value\n• Higher values capture longer-term patterns\n• Recommended: 1-3 (start with 1)",
            justify="left",
        ).pack(fill="x", pady=(0, 10), padx=20)

        Label(
            scrollable_frame,
            text="d (Diff) - Differencing Order:",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(5, 0), padx=10)
        Label(
            scrollable_frame,
            text="• Makes data stationary by removing trends\n• 0: No differencing, 1: Remove linear trend, 2: Remove quadratic trend\n• Recommended: 0-1 for marker trajectories",
            justify="left",
        ).pack(fill="x", pady=(0, 10), padx=20)

        Label(
            scrollable_frame,
            text="q (MA) - Moving Average Order:",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(5, 0), padx=10)
        Label(
            scrollable_frame,
            text="• Incorporates error terms from previous predictions\n• Higher values smooth out irregular patterns\n• Recommended: 1-2 (start with 1)",
            justify="left",
        ).pack(fill="x", pady=(0, 10), padx=20)

        Label(
            scrollable_frame,
            text="Max Gap Size:",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(5, 0), padx=10)
        Label(
            scrollable_frame,
            text="• Maximum number of consecutive frames to fill\n• Larger gaps are less reliable to fill\n• Recommended: 15-30 frames",
            justify="left",
        ).pack(fill="x", pady=(0, 10), padx=20)

        # Add practical advice
        Label(
            scrollable_frame,
            text="Practical Tips:",
            font=("Arial", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(10, 0), padx=10)
        Label(
            scrollable_frame,
            text="• Start with default values (p=1, d=1, q=1)\n• For smoother trajectories, increase q\n• For cyclical motion, increase p\n• For complex patterns, try p=2, d=1, q=2",
            justify="left",
        ).pack(fill="x", pady=(0, 10), padx=20)

        # OK button
        TkButton(help_window, text="OK", command=help_window.destroy).pack(pady=10)

    # Add help button for all parameters
    help_all_button = TkButton(
        arima_params_window, text="View Complete Parameter Guide", command=show_all_help
    )
    help_all_button.pack(pady=10)

    arima_params_window.mainloop()


if __name__ == "__main__":
    create_gui_menu()
