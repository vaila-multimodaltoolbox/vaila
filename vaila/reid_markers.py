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

import os
import sys
from rich import print
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RangeSlider, CheckButtons, TextBox, RadioButtons
from tkinter import Tk, filedialog, messagebox, Frame, Label, Button as TkButton
import tkinter as tk
from scipy.interpolate import interp1d
import json
from statsmodels.tsa.arima.model import ARIMA
import warnings
import shutil  # Para operações de diretório
from tkinter import ttk


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
        source_x_valid = not (
            pd.isna(df.at[i, source_x_col]) or df.at[i, source_x_col] == ""
        )
        source_y_valid = not (
            pd.isna(df.at[i, source_y_col]) or df.at[i, source_y_col] == ""
        )

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

    print(
        f"Markers {marker_id1} and {marker_id2} swapped in frame range {start_frame}-{end_frame}"
    )
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

    # Detect all markers in the file
    all_markers = detect_markers(df)

    # Ensure marker_ids is either None or a list of integers
    if isinstance(
        marker_ids, str
    ):  # Fix for when a file path is passed instead of marker IDs
        marker_ids = None

    # Handle empty dataframe
    if "frame" not in df.columns or len(df) == 0:
        messagebox.showerror(
            "Error", "The data file appears to be empty or missing the 'frame' column."
        )
        return

    frames = df["frame"].values

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
    frames_range = RangeSlider(
        frames_slider_ax, "Frames", 0, len(df) - 1, valinit=(0, len(df) - 1)
    )

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
                lines_x[marker_id] = ax1.plot(
                    frames, x_values, label=f"Marker {marker_id}"
                )[0]
                lines_y[marker_id] = ax2.plot(
                    frames, y_values, label=f"Marker {marker_id}"
                )[0]

        # Show frame range with vertical lines
        start_frame, end_frame = frames_range.val
        ax1.axvline(start_frame, color="r", linestyle="--")
        ax1.axvline(end_frame, color="r", linestyle="--")
        ax2.axvline(start_frame, color="r", linestyle="--")
        ax2.axvline(end_frame, color="r", linestyle="--")

        ax1.set_xlim(0, len(df) - 1)
        ax2.set_xlim(0, len(df) - 1)

        # Place legend outside the plot to avoid overlap
        if visible_markers and any(
            marker_id in lines_x for marker_id in visible_markers
        ):
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
    return [
        all_markers[i]
        for i, checked in enumerate(marker_selection.get_status())
        if checked
    ]


def advanced_reid_gui():
    # Load file
    df, file_path = load_markers_file()
    if df is None:
        return

    # Add variables for tracking history and temp files
    latest_temp_file = file_path
    temp_history = []

    # Detect markers
    all_markers = detect_markers(df)
    frames = df["frame"].values

    # Set up the interface with a larger figure for better visibility
    fig = plt.figure(figsize=(18, 10))

    # Create subplots for X and Y coordinates
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0))

    plt.subplots_adjust(
        left=0.2, bottom=0.3, right=0.85, top=0.95
    )  # Aumentei o bottom para dar mais espaço

    ax1.set_title("X Coordinates of Markers")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("X Position")
    ax1.grid(True)

    ax2.set_title("Y Coordinates of Markers")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Y Position")
    ax2.grid(True)

    # Create areas for buttons and controls
    fill_button_ax = plt.axes([0.3, 0.1, 0.12, 0.04])
    merge_button_ax = plt.axes([0.44, 0.1, 0.12, 0.04])
    swap_button_ax = plt.axes([0.58, 0.1, 0.12, 0.04])
    delete_button_ax = plt.axes([0.72, 0.1, 0.12, 0.04])

    # Areas for marker selection - REDUCED WIDTH from 0.18 to 0.12
    markers_checkbox_ax = plt.axes([0.04, 0.25, 0.12, 0.65])

    # Slider for frame range selection
    frames_slider_ax = plt.axes([0.3, 0.15, 0.5, 0.03])
    frames_range = RangeSlider(
        frames_slider_ax, "Frames", 0, len(df) - 1, valinit=(0, len(df) - 1)
    )

    # Save and close buttons
    save_button_ax = plt.axes([0.3, 0.05, 0.15, 0.04])
    close_button_ax = plt.axes([0.5, 0.05, 0.15, 0.04])

    # Add select all/none buttons - REDUCED WIDTH from 0.08 to 0.05
    select_all_ax = plt.axes([0.04, 0.2, 0.05, 0.03])
    select_none_ax = plt.axes([0.10, 0.2, 0.05, 0.03])

    # Add undo button
    undo_button_ax = plt.axes([0.68, 0.05, 0.12, 0.04])

    # Add help button
    help_button_ax = plt.axes([0.04, 0.05, 0.08, 0.03])

    # Create interactive controls
    marker_selection = CheckButtons(
        markers_checkbox_ax,
        [f"Marker {m}" for m in all_markers],
        [True] * len(all_markers),
    )

    fill_button = Button(fill_button_ax, "Fill Gaps")
    merge_button = Button(merge_button_ax, "Merge Markers")
    swap_button = Button(swap_button_ax, "Swap Markers")
    delete_button = Button(delete_button_ax, "Delete Marker")
    save_button = Button(save_button_ax, "Save Changes and Exit")
    close_button = Button(close_button_ax, "Close")

    select_all_button = Button(select_all_ax, "All")
    select_none_button = Button(select_none_ax, "None")
    undo_button = Button(undo_button_ax, "Undo")
    help_button = Button(help_button_ax, "Help")

    # Dictionary to store plot lines
    lines_x = {}
    lines_y = {}

    # Track operations performed
    operations_log = {
        "fill_gaps": [],
        "merge_markers": [],
        "swap_markers": [],
        "delete_markers": [],
        "removed_markers": [],
    }

    def create_temp_dir(original_path):
        """Create a temporary directory for storing edit files."""
        base_dir = os.path.dirname(original_path)
        base_name = os.path.basename(original_path).split(".")[0]
        temp_dir = os.path.join(base_dir, f"{base_name}_temp")

        # Create directory if it doesn't exist
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
        """Remove the temporary directory and its files."""
        temp_dir = create_temp_dir(original_path)
        if os.path.exists(temp_dir):
            import shutil

            shutil.rmtree(temp_dir)
            print(f"Temporary directory removed: {temp_dir}")

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
                lines_x[marker_id] = ax1.plot(
                    frames, x_values, label=f"Marker {marker_id}"
                )[0]
                lines_y[marker_id] = ax2.plot(
                    frames, y_values, label=f"Marker {marker_id}"
                )[0]

        # Show frame range with vertical lines
        start_frame, end_frame = frames_range.val
        ax1.axvline(start_frame, color="r", linestyle="--")
        ax1.axvline(end_frame, color="r", linestyle="--")
        ax2.axvline(start_frame, color="r", linestyle="--")
        ax2.axvline(end_frame, color="r", linestyle="--")

        ax1.set_xlim(0, len(df) - 1)
        ax2.set_xlim(0, len(df) - 1)

        # Place legend outside the plot to avoid overlap
        if visible_markers and any(
            marker_id in lines_x for marker_id in visible_markers
        ):
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

    # Get selected markers
    def get_selected_markers():
        checked = marker_selection.get_status()
        return [
            all_markers[i] for i, checked_state in enumerate(checked) if checked_state
        ]

    # Connect callbacks
    def on_marker_select(event):
        update_plot()

    def on_fill_gaps(event):
        nonlocal latest_temp_file, df

        selected_markers = get_selected_markers()
        if not selected_markers:
            return

        # Backup para operação de desfazer
        temp_history.append(df.copy())

        # Para cada marcador selecionado
        for marker_id in selected_markers:
            x_col = f"p{marker_id}_x"
            y_col = f"p{marker_id}_y"

            if x_col not in df.columns or y_col not in df.columns:
                continue

            x_values = df[x_col].values.copy()
            y_values = df[y_col].values.copy()

            # Detectar gaps
            gaps = detect_gaps(x_values, y_values)
            if not gaps:
                print(f"Marker {marker_id}: No gaps detected")
                continue

            # Analisar características dos dados para configurar Kalman
            valid_indices = ~(pd.isna(x_values) | pd.isna(y_values))
            data_length = np.sum(valid_indices)

            # Preparar arrays para resultados
            x_filled = x_values.copy()
            y_filled = y_values.copy()

            # Processar cada gap
            for gap_start, gap_end in gaps:
                gap_size = gap_end - gap_start + 1

                # Escolher método baseado no tamanho do gap e dados disponíveis
                if gap_size <= 5:
                    # Gaps pequenos: interpolação linear simples
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

                        x_interp = np.interp(
                            range(gap_start, gap_end + 1), frames, x_known
                        )
                        y_interp = np.interp(
                            range(gap_start, gap_end + 1), frames, y_known
                        )

                        for i, idx in enumerate(range(gap_start, gap_end + 1)):
                            x_filled[idx] = x_interp[i]
                            y_filled[idx] = y_interp[i]
                else:
                    # Gaps maiores: Kalman filter com parâmetros estimados
                    # Coletar contexto antes e depois do gap
                    before_points = []
                    after_points = []

                    # Pontos antes do gap (até 20)
                    i = gap_start - 1
                    count = 0
                    while i >= 0 and count < 20:
                        if not pd.isna(x_values[i]) and not pd.isna(y_values[i]):
                            before_points.insert(0, (i, x_values[i], y_values[i]))
                            count += 1
                        i -= 1

                    # Pontos depois do gap (até 20)
                    i = gap_end + 1
                    count = 0
                    while i < len(x_values) and count < 20:
                        if not pd.isna(x_values[i]) and not pd.isna(y_values[i]):
                            after_points.append((i, x_values[i], y_values[i]))
                            count += 1
                        i += 1

                    # Verificar se temos contexto suficiente
                    if len(before_points) >= 3 and len(after_points) >= 3:
                        # Implementação simples de Kalman
                        # Estimar velocidade média dos pontos de contexto
                        if len(before_points) > 1:
                            dx_before = (before_points[-1][1] - before_points[0][1]) / (
                                before_points[-1][0] - before_points[0][0]
                            )
                            dy_before = (before_points[-1][2] - before_points[0][2]) / (
                                before_points[-1][0] - before_points[0][0]
                            )
                        else:
                            dx_before = 0
                            dy_before = 0

                        if len(after_points) > 1:
                            dx_after = (after_points[-1][1] - after_points[0][1]) / (
                                after_points[-1][0] - after_points[0][0]
                            )
                            dy_after = (after_points[-1][2] - after_points[0][2]) / (
                                after_points[-1][0] - after_points[0][0]
                            )
                        else:
                            dx_after = 0
                            dy_after = 0

                        # Média ponderada das velocidades
                        dx = (dx_before + dx_after) / 2
                        dy = (dy_before + dy_after) / 2

                        # Ponto inicial
                        x_start = before_points[-1][1]
                        y_start = before_points[-1][2]

                        # Predição linear com ajuste para o ponto final
                        x_end = after_points[0][1]
                        y_end = after_points[0][2]

                        # Ajustar para combinar o ponto final
                        for i, idx in enumerate(range(gap_start, gap_end + 1)):
                            t = i / gap_size  # Fator de interpolação (0 a 1)
                            # Combinação de predição linear e correção para o ponto final
                            x_filled[idx] = (
                                x_start + dx * i * (1 - t) + (x_end - x_start) * t
                            )
                            y_filled[idx] = (
                                y_start + dy * i * (1 - t) + (y_end - y_start) * t
                            )
                    else:
                        # Fallback para interpolação linear se não tivermos contexto suficiente
                        before_idx = gap_start - 1
                        after_idx = gap_end + 1

                        while before_idx >= 0 and (
                            pd.isna(x_values[before_idx])
                            or pd.isna(y_values[before_idx])
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

                            x_interp = np.interp(
                                range(gap_start, gap_end + 1), frames, x_known
                            )
                            y_interp = np.interp(
                                range(gap_start, gap_end + 1), frames, y_known
                            )

                            for i, idx in enumerate(range(gap_start, gap_end + 1)):
                                x_filled[idx] = x_interp[i]
                                y_filled[idx] = y_interp[i]

            # Atualizar o DataFrame com valores preenchidos
            df[x_col] = x_filled
            df[y_col] = y_filled
            print(f"Filled gaps intelligently for marker {marker_id}")

            # Registrar operação
            operations_log["fill_gaps"].append(
                {
                    "marker_id": marker_id,
                    "method": "intelligent_kalman",
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        # Criar arquivo temporário
        latest_temp_file = create_temp_file(df, file_path)

        # Carregar dados do arquivo temporário
        df_temp = pd.read_csv(latest_temp_file)
        df = df_temp

        # Atualizar o gráfico
        update_plot()

        messagebox.showinfo(
            "Processo Concluído",
            f"Preenchimento inteligente de gaps concluído para {len(selected_markers)} marcador(es).",
        )

    def on_merge_markers(event):
        nonlocal latest_temp_file, df

        selected_markers = get_selected_markers()

        if len(selected_markers) < 2:
            return

        # Backup para operação de desfazer
        temp_history.append(df.copy())

        # Use o marcador com o menor ID como o destino (a ser mantido)
        source_id = min(selected_markers)
        # Todos os outros marcadores selecionados serão fontes (a serem mesclados e removidos)
        sources = [m for m in selected_markers if m != source_id]

        start_frame, end_frame = map(int, frames_range.val)

        # Obter colunas do alvo (o que será mantido)
        target_x_col = f"p{source_id}_x"
        target_y_col = f"p{source_id}_y"

        # Processar cada marcador fonte (a ser removido)
        for source_id_to_remove in sources:
            source_x_col = f"p{source_id_to_remove}_x"
            source_y_col = f"p{source_id_to_remove}_y"

            # Transferir TODAS as coordenadas válidas da fonte para o alvo
            for i in range(start_frame, end_frame + 1):
                # Verificar se a fonte tem dados válidos neste quadro
                source_x_valid = not (
                    pd.isna(df.at[i, source_x_col]) or df.at[i, source_x_col] == ""
                )
                source_y_valid = not (
                    pd.isna(df.at[i, source_y_col]) or df.at[i, source_y_col] == ""
                )

                if source_x_valid and source_y_valid:
                    # Transferir dados da fonte para o alvo
                    df.at[i, target_x_col] = df.at[i, source_x_col]
                    df.at[i, target_y_col] = df.at[i, source_y_col]

            # Adicionar à lista de marcadores para remover ao salvar
            if source_id_to_remove not in operations_log["removed_markers"]:
                operations_log["removed_markers"].append(source_id_to_remove)

            # Registrar a operação
            operations_log["merge_markers"].append(
                {
                    "target_id": source_id,  # O marcador que estamos mantendo
                    "source_id": source_id_to_remove,  # O marcador sendo mesclado e removido
                    "frame_range": [start_frame, end_frame],
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        # REMOVER OS MARCADORES APÓS A TRANSFERÊNCIA DE DADOS
        columns_to_remove = []
        for marker_id in sources:
            columns_to_remove.append(f"p{marker_id}_x")
            columns_to_remove.append(f"p{marker_id}_y")

        df = df.drop(columns=columns_to_remove)

        # Criar arquivo temporário
        latest_temp_file = create_temp_file(df, file_path)

        # Carregar dados do arquivo temporário para atualizar o gráfico
        df_temp = pd.read_csv(latest_temp_file)
        df = df_temp  # Atualizar o DataFrame com dados do arquivo temporário

        update_plot()

    def on_swap_markers(event):
        nonlocal latest_temp_file, df

        selected_markers = get_selected_markers()

        if len(selected_markers) != 2:
            print(
                "Erro: Por favor, selecione exatamente dois marcadores para trocar dados."
            )
            return

        marker_id1, marker_id2 = selected_markers

        # Backup para operação de desfazer
        temp_history.append(df.copy())

        # Trocar dados entre os dois marcadores selecionados
        df = swap_markers(df, marker_id1, marker_id2, (0, len(df) - 1))

        # Criar arquivo temporário
        latest_temp_file = create_temp_file(df, file_path)

        # Carregar dados do arquivo temporário para atualizar o gráfico
        df_temp = pd.read_csv(latest_temp_file)
        df = df_temp  # Atualizar o DataFrame com dados do arquivo temporário

        update_plot()

    def on_delete_markers(event):
        nonlocal latest_temp_file, df

        selected_markers = get_selected_markers()

        if not selected_markers:
            return

        # Backup para operação de desfazer
        temp_history.append(df.copy())

        # Remover marcadores selecionados
        df = df.drop(
            columns=[f"p{m}_x" for m in selected_markers]
            + [f"p{m}_y" for m in selected_markers]
        )

        # Criar arquivo temporário
        latest_temp_file = create_temp_file(df, file_path)

        # Carregar dados do arquivo temporário para atualizar o gráfico
        df_temp = pd.read_csv(latest_temp_file)
        df = df_temp  # Atualizar o DataFrame com dados do arquivo temporário

        update_plot()

    def on_save(event):
        nonlocal latest_temp_file, df

        # Salvar o arquivo final com as modificações
        save_markers_file(df, file_path)
        print(f"Arquivo final salvo com modificações")

        # Limpar diretório temporário
        clear_temp_dir(file_path)

        # Fechar a janela
        plt.close(fig)

    def on_close(event):
        # Limpar diretório temporário antes de fechar
        clear_temp_dir(file_path)
        plt.close(fig)  # Close the figure when done

    def on_undo(event):
        nonlocal latest_temp_file, df

        if temp_history:
            df = temp_history.pop()
            latest_temp_file = create_temp_file(df, file_path)
            update_plot()
        else:
            print("Não há mais operações para desfazer.")

    def open_help(event):
        import webbrowser

        webbrowser.open("https://vaila.readthedocs.io/en/latest/")

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

    # Connect callbacks
    marker_selection.on_clicked(on_marker_select)
    frames_range.on_changed(lambda val: update_plot())
    fill_button.on_clicked(on_fill_gaps)
    merge_button.on_clicked(on_merge_markers)
    swap_button.on_clicked(on_swap_markers)
    delete_button.on_clicked(on_delete_markers)
    save_button.on_clicked(on_save)
    close_button.on_clicked(on_close)
    select_all_button.on_clicked(select_all)
    select_none_button.on_clicked(select_none)
    undo_button.on_clicked(on_undo)
    help_button.on_clicked(open_help)

    # Initialize plot
    update_plot()

    plt.tight_layout(rect=[0.2, 0.25, 0.85, 1])  # Adjust layout
    plt.show()


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
        print(
            f"Too few valid points for marker {marker_id}, falling back to linear interpolation"
        )
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
            [p[0] for p in before_segment]
            + [np.nan] * gap_size
            + [p[0] for p in after_segment]
        )
        full_segment_y = (
            [p[1] for p in before_segment]
            + [np.nan] * gap_size
            + [p[1] for p in after_segment]
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

        help_label = Label(
            help_window, text=help_text[param], justify="left", padx=15, pady=15
        )
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
    p_help_btn = TkButton(
        p_frame, text="?", width=2, command=lambda: show_help_popup("p")
    )
    p_help_btn.pack(side=tk.LEFT)

    # ARIMA d parameter row
    d_frame = Frame(params_frame)
    d_frame.pack(fill="x", pady=5)
    Label(d_frame, text="d (Diff):", width=10, anchor="w").pack(side=tk.LEFT)
    d_var = tk.StringVar(value=str(default_d))
    d_entry = tk.Entry(d_frame, textvariable=d_var, width=5)
    d_entry.pack(side=tk.LEFT, padx=5)
    Label(d_frame, text=f"Default: {default_d}", width=10).pack(side=tk.LEFT)
    d_help_btn = TkButton(
        d_frame, text="?", width=2, command=lambda: show_help_popup("d")
    )
    d_help_btn.pack(side=tk.LEFT)

    # ARIMA q parameter row
    q_frame = Frame(params_frame)
    q_frame.pack(fill="x", pady=5)
    Label(q_frame, text="q (MA):", width=10, anchor="w").pack(side=tk.LEFT)
    q_var = tk.StringVar(value=str(default_q))
    q_entry = tk.Entry(q_frame, textvariable=q_var, width=5)
    q_entry.pack(side=tk.LEFT, padx=5)
    Label(q_frame, text=f"Default: {default_q}", width=10).pack(side=tk.LEFT)
    q_help_btn = TkButton(
        q_frame, text="?", width=2, command=lambda: show_help_popup("q")
    )
    q_help_btn.pack(side=tk.LEFT)

    # Max gap size parameter row
    max_gap_frame = Frame(params_frame)
    max_gap_frame.pack(fill="x", pady=5)
    Label(max_gap_frame, text="Max Gap Size:", width=10, anchor="w").pack(side=tk.LEFT)
    max_gap_var = tk.StringVar(value=str(default_max_gap))
    max_gap_entry = tk.Entry(max_gap_frame, textvariable=max_gap_var, width=5)
    max_gap_entry.pack(side=tk.LEFT, padx=5)
    Label(max_gap_frame, text=f"Default: {default_max_gap}", width=10).pack(
        side=tk.LEFT
    )
    max_gap_help_btn = TkButton(
        max_gap_frame, text="?", width=2, command=lambda: show_help_popup("max_gap")
    )
    max_gap_help_btn.pack(side=tk.LEFT)

    # Separator
    separator = Frame(arima_params_window, height=2, bd=1, relief=tk.SUNKEN)
    separator.pack(fill="x", padx=20, pady=15)

    # Processing status
    status_var = tk.StringVar(value="")
    status_label = Label(
        arima_params_window, textvariable=status_var, font=("Arial", 9), fg="blue"
    )
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
                df.loc[
                    0, "processed_info"
                ] += f"ARIMA filled {points_filled} points for marker {marker_id}; "

            # Update status to show progress
            status_var.set(
                f"Processing marker {marker_id}... ({markers.index(marker_id)+1}/{len(markers)})"
            )
            arima_params_window.update_idletasks()  # Force GUI update

        # Add timestamp and processing info
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        df.loc[
            0, "processed_info"
        ] += f"ARIMA processed on {timestamp} with parameters (p={arima_params['p']},d={arima_params['d']},q={arima_params['q']})"

        # Save with a more descriptive suffix that includes parameters
        param_text = f"p{arima_params['p']}d{arima_params['d']}q{arima_params['q']}"
        new_file = save_markers_file(
            df, file_path, suffix=f"_arima_{param_text}_{timestamp}"
        )

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
            messagebox.showerror(
                "Error", "Please enter valid integers for all parameters"
            )

    # Button frame
    button_frame = Frame(arima_params_window)
    button_frame.pack(pady=10)

    # Add a reset button
    reset_button = TkButton(
        button_frame, text="Reset to Defaults", command=reset_to_defaults
    )
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
        Label(
            help_window, text="ARIMA Parameter Guide", font=("Arial", 12, "bold")
        ).pack(pady=(10, 15))

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


def create_gui_menu():
    """Create main GUI menu for the application."""

    # Print the script version and directory
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = Tk()
    root.title("Marker Re-identification Tool")
    root.geometry("500x300")  # Reduced height since we're removing an option

    # Set up frame for buttons
    frame = Frame(root, padx=20, pady=20)
    frame.pack(expand=True)

    # Title label
    title_label = Label(
        frame, text="Marker Re-identification Tool", font=("Arial", 16, "bold")
    )
    title_label.grid(row=0, column=0, pady=(0, 20))

    # Description
    desc_label = Label(frame, text="Select an option:", font=("Arial", 12))
    desc_label.grid(row=1, column=0, pady=(0, 20), sticky="w")

    # Buttons for the different options
    btn_width = 30
    btn_height = 2

    # Interactive Re-identification button
    option1_btn = TkButton(
        frame,
        text="Interactive Re-identification",
        width=btn_width,
        height=btn_height,
        command=lambda: [root.destroy(), advanced_reid_gui()],
    )
    option1_btn.grid(row=2, column=0, pady=10)

    # Auto Fill Gaps (ARIMA) button
    option2_btn = TkButton(
        frame,
        text="Auto Fill Gaps (ARIMA)",
        width=btn_width,
        height=btn_height,
        command=lambda: [root.destroy(), auto_fill_gaps_arima()],
    )
    option2_btn.grid(row=3, column=0, pady=10)

    # Exit button - row changed from 5 to 4 since we removed an option
    option4_btn = TkButton(
        frame, text="Exit", width=btn_width, height=btn_height, command=root.destroy
    )
    option4_btn.grid(row=4, column=0, pady=10)

    # Version info - row changed from 6 to 5
    version_label = Label(frame, text="Version 0.1.0", font=("Arial", 8))
    version_label.grid(row=5, column=0, pady=(20, 0))

    root.mainloop()


if __name__ == "__main__":
    create_gui_menu()
