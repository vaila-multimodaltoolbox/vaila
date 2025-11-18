"""
Project: vailá Multimodal Toolbox
Script: readcsv.py - Read CSV File

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 03 September 2025
Version: 0.0.4

Description:
    This script provides tools for reading CSV files and displaying their contents.
    It includes functions for:
    - Detecting the delimiter used in the file.
    - Detecting if the file has a header.
    - Selecting markers to display.
    - Selecting headers to display.
    - Visualizing the data using Matplotlib or Open3D.

Usage:
    Run the script from the command line:
        python readcsv.py

Requirements:
    - Python 3.x
    - pandas
    - numpy
    - matplotlib
    - tkinter
    - rich

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.3: Added support for CSV, TXT and TSV files, improved UI
    - v0.0.2: Added support for CSV, TXT and TSV files, improved UI
    - v0.0.1: First version
"""

import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import Button, Frame, Label, Listbox, filedialog, messagebox
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.backend_bases import TimerBase
from matplotlib.widgets import Button as MplButton
from matplotlib.widgets import Slider, TextBox
from mpl_toolkits.mplot3d import Axes3D
from rich import print


# ==================== PERFORMANCE OPTIMIZATIONS ====================
def _npz_path_for(csv_path: str) -> str:
    """Generate NPZ cache path for CSV file"""
    base = os.path.basename(csv_path)
    stem, _ = os.path.splitext(base)
    return os.path.join(os.path.dirname(csv_path), f"{stem}_cache.npz")


def _load_with_cache(csv_path: str, use_cache: bool = True):
    """Load data with NPZ cache for speed"""
    npz_file = _npz_path_for(csv_path)

    if use_cache and os.path.exists(npz_file):
        try:
            print(f"[bold green]Loading from cache: {npz_file}[/bold green]")
            data = np.load(npz_file)
            return data["index_vector"], data["marker_names"], data["points"]
        except Exception as e:
            print(f"Cache loading failed: {e}")

    # Load from CSV and save cache
    print("[bold yellow]Loading from CSV and creating cache...[/bold yellow]")
    index_vector, marker_data, valid_markers, delimiter = read_csv_generic(csv_path)
    points = np.stack([marker_data[m] for m in valid_markers.keys()], axis=1)

    if use_cache:
        try:
            np.savez_compressed(
                npz_file,
                index_vector=index_vector,
                marker_names=list(valid_markers.keys()),
                points=points,
            )
            print(f"[bold green]Cache saved: {npz_file}[/bold green]")
        except Exception as e:
            print(f"Cache saving failed: {e}")

    return index_vector, list(valid_markers.keys()), points


def show_csv_optimized(file_path=None, *, use_cache=True, turbo=False, fps=30, stride=1):
    """
    Optimized version of show_csv with performance improvements

    Args:
        file_path: CSV file path
        use_cache: Use NPZ cache for faster loading
        turbo: Use optimized visualization mode
        fps: Frames per second for animation
        stride: Skip frames (1=all, 2=every other, etc.)
    """
    print(f"Running optimized script: {os.path.basename(__file__)}")
    print(f"Performance settings: cache={use_cache}, turbo={turbo}, fps={fps}, stride={stride}")

    # File selection
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = select_file()
        if not file_path:
            print("No file selected.")
            return

    if not os.path.exists(file_path):
        print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        return

    # Load data with cache
    try:
        index_vector, marker_names, points = _load_with_cache(file_path, use_cache)
    except Exception as e:
        messagebox.showerror("Error", f"Error loading data: {e}")
        return

    # Apply stride to reduce data size
    if stride > 1:
        points = points[::stride]
        index_vector = index_vector[::stride]
        print(
            f"Applied stride {stride}: reduced frames from {len(index_vector) * stride} to {len(index_vector)}"
        )

    # Select markers
    selected_markers = select_markers_csv(marker_names)
    if not selected_markers:
        messagebox.showwarning("Warning", "No markers selected.")
        return

    # Filter points for selected markers
    marker_indices = [marker_names.index(m) for m in selected_markers]
    points = points[:, marker_indices, :]

    print("[bold green]Data ready for visualization![/bold green]")
    print(f"Shape: {points.shape}, FPS: {fps}")

    if turbo:
        show_csv_matplotlib_turbo(points, selected_markers, fps)
    else:
        show_csv_matplotlib(points, selected_markers, fps)


def show_csv_matplotlib_turbo(points, marker_names, fps=30):
    """Ultra-fast matplotlib visualization with optimizations"""

    # Pre-filter NaN values
    valid_frames = []
    for frame in range(points.shape[0]):
        valid_mask = ~np.isnan(points[frame]).any(axis=1)
        valid_frames.append(valid_mask)

    # Pre-calculate colors
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(marker_names), 10)))

    # Create figure with optimized settings
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Turbo Mode - {len(marker_names)} markers, {points.shape[0]} frames")

    # Set limits once
    valid_points = points[~np.isnan(points)]
    if len(valid_points) > 0:
        x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
        y_min, y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
        z_min, z_max = valid_points[:, 2].min(), valid_points[:, 2].max()

        # Add padding
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
        padding = 0.1
        ax.set_xlim(x_min - x_range * padding, x_max + x_range * padding)
        ax.set_ylim(y_min - y_range * padding, y_max + y_range * padding)
        ax.set_zlim(z_min - z_range * padding, z_max + z_range * padding)

    # Initial plot
    valid_mask = valid_frames[0]
    valid_points = points[0][valid_mask]
    if len(valid_points) > 0:
        scatter = ax.scatter(
            valid_points[:, 0],
            valid_points[:, 1],
            valid_points[:, 2],
            c=colors[: len(valid_points)],
            s=30,
            alpha=0.8,
        )
    else:
        scatter = ax.scatter([], [], [], c=[], s=30)

    # Add simple controls
    ax_slider = plt.axes((0.25, 0.02, 0.5, 0.03))
    slider = Slider(ax_slider, "Frame", 0, points.shape[0] - 1, valinit=0, valfmt="%d")

    def update(frame):
        frame_idx = int(frame)
        valid_mask = valid_frames[frame_idx]
        valid_points = points[frame_idx][valid_mask]

        if len(valid_points) > 0:
            # Update scatter positions only (faster than clear/redraw)
            scatter._offsets3d = (
                valid_points[:, 0],
                valid_points[:, 1],
                valid_points[:, 2],
            )
            scatter.set_array(colors[: len(valid_points)])
        else:
            scatter._offsets3d = ([], [], [])

        ax.set_title(f"Frame {frame_idx}/{points.shape[0] - 1}")
        return (scatter,)

    slider.on_changed(update)

    # Add play/pause button
    ax_play = plt.axes((0.82, 0.02, 0.1, 0.05))
    btn_play = MplButton(ax_play, "Play")

    playing = [False]
    timer = [None]

    def play_pause(event):
        if not playing[0]:
            playing[0] = True
            btn_play.label.set_text("Pause")
            timer[0] = fig.canvas.new_timer(interval=int(1000 / fps))
            timer[0].add_callback(lambda: slider.set_val((slider.val + 1) % points.shape[0]))
            timer[0].start()
        else:
            playing[0] = False
            btn_play.label.set_text("Play")
            if timer[0]:
                timer[0].stop()

    btn_play.on_clicked(play_pause)

    # Keyboard shortcuts
    def on_key(event):
        if event.key == " ":
            play_pause(None)
        elif event.key == "right":
            slider.set_val(min(slider.val + 1, points.shape[0] - 1))
        elif event.key == "left":
            slider.set_val(max(slider.val - 1, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


###############################################################################
# Function: headersidx
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def headersidx(headers, prefix):
    """
    Given a list of headers and a prefix, returns the indices of headers that start with the prefix.

    Args:
        headers (list): List of header names.
        prefix (str): Prefix to check.

    Returns:
        List[int]: List of indices of headers that start with the prefix.
    """
    return [i for i, header in enumerate(headers) if header.startswith(prefix)]


###############################################################################
# Function: reshapedata
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def reshapedata(df, selected_markers):
    """
    Given a DataFrame `df` containing the time column and marker columns in the format:
      marker_x, marker_y, marker_z,
    and a list with the names of selected markers, returns a NumPy array of shape:
      (num_frames, num_markers, 3)

    If the mean absolute value is high (> 100), data is converted from millimeters to meters.

    Args:
        df (DataFrame): DataFrame containing the data, where the first column is Time.
        selected_markers (list): List of marker names.

    Returns:
        numpy.ndarray: Array with shape (num_frames, num_markers, 3) containing marker data.
    """
    num_frames = df.shape[0]
    num_markers = len(selected_markers)
    points = np.zeros((num_frames, num_markers, 3))
    for i, marker in enumerate(selected_markers):
        x_col = f"{marker}_x"
        y_col = f"{marker}_y"
        z_col = f"{marker}_z"
        if x_col in df.columns and y_col in df.columns and z_col in df.columns:
            points[:, i, 0] = df[x_col].values
            points[:, i, 1] = df[y_col].values
            points[:, i, 2] = df[z_col].values
        else:
            raise ValueError(f"Columns for marker '{marker}' not found in expected format.")
    if np.mean(np.abs(points)) > 100:
        points = points * 0.001  # Convert from millimeters to meters
    return points


###############################################################################
# Function: detect_delimiter
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_delimiter(file_path):
    """
    Detects the delimiter used in the file by trying common delimiters.
    Analyzes multiple lines for more accurate detection.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Detected delimiter (',', ';', '\t', or ' ')
    """
    import csv

    # Try different delimiters and check consistency
    delimiters = [",", ";", "\t", " "]
    delimiter_scores = {}

    try:
        with open(file_path, encoding="utf-8") as file:
            # Read first few lines
            sample_lines = []
            file.seek(0)
            for i, line in enumerate(file):
                if i >= 10:  # Read max 10 lines
                    break
                sample_lines.append(line.strip())

            for delimiter in delimiters:
                scores = []
                try:
                    # Use csv.Sniffer for better detection
                    file.seek(0)
                    sample = file.read(1024)
                    if sample:  # Check if sample is not empty
                        dialect = csv.Sniffer().sniff(sample, delimiters=[delimiter])
                        if dialect.delimiter == delimiter:
                            scores.append(10)  # High score for csv.Sniffer detection
                except Exception:
                    pass

                # Check consistency of column count
                column_counts = []
                for line in sample_lines:
                    if line:
                        column_counts.append(len(line.split(delimiter)))

                if column_counts:
                    # Prefer delimiters that give consistent column counts
                    most_common_count = max(set(column_counts), key=column_counts.count)
                    consistency = column_counts.count(most_common_count) / len(column_counts)
                    avg_columns = sum(column_counts) / len(column_counts)

                    # Score based on consistency and reasonable column count
                    score = consistency * 10 + (avg_columns if avg_columns > 1 else 0)
                    delimiter_scores[delimiter] = score

            # Return delimiter with highest score
            if delimiter_scores:
                best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
                print(
                    f"Delimiter detected: '{best_delimiter}' (score: {delimiter_scores[best_delimiter]:.2f})"
                )
                return best_delimiter

    except Exception as e:
        print(f"Warning: Error detecting delimiter: {e}")

    print("Using default delimiter: ','")
    return ","  # Default to comma


###############################################################################
# Function: detect_has_header
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_has_header(file_path, delimiter):
    """
    Detects if the file has a header by checking if the first line contains non-numeric values.

    Args:
        file_path (str): Path to the file
        delimiter (str): Delimiter used in the file

    Returns:
        bool: True if file has header, False otherwise
    """
    with open(file_path, encoding="utf-8") as file:
        first_line = file.readline().strip()
        values = first_line.split(delimiter)

        # Check if any value in the first line is not numeric
        for value in values:
            try:
                float(value)
            except ValueError:
                return True
    return False


###############################################################################
# Function: select_file
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def select_file():
    """Displays the dialog box for selecting CSV, TXT or TSV file."""
    return filedialog.askopenfilename(
        title="Select data file",
        filetypes=[
            ("Data files", ("*.csv", "*.txt", "*.tsv")),
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("TSV files", "*.tsv"),
            ("All files", "*.*"),
        ],
    )


###############################################################################
# Function: choose_visualizer
# Opens a window for the user to choose the visualization method.
###############################################################################
def choose_visualizer():
    """
    Opens a window for the user to choose the visualization method.
    Returns:
        "matplotlib" or "open3d" based on user choice.
    """
    root = tk.Tk()
    root.title("Select Visualizer")
    choice: list[str | None] = [None]  # Add type annotation

    def choose_matplotlib():
        choice[0] = "matplotlib"
        root.quit()

    def choose_open3d():
        choice[0] = "open3d"
        root.quit()

    Label(root, text="Choose visualization method:").pack(pady=10)
    Button(root, text="Matplotlib Visualizer", command=choose_matplotlib).pack(pady=5)
    Button(root, text="Open3D Visualizer", command=choose_open3d).pack(pady=5)
    root.mainloop()
    root.destroy()
    return choice[0]


###############################################################################
# Function: select_markers_csv
# Displays a marker selection window using a Listbox.
###############################################################################
def select_markers_csv(marker_labels):
    """
    Displays a Tkinter window with a Listbox for marker selection.

    Args:
        marker_labels (list): list of marker names.

    Returns:
        List of selected marker names.
    """
    root = tk.Tk()
    root.title("Select Markers to Display")
    listbox = Listbox(root, selectmode="multiple", width=50, height=15)
    for label in marker_labels:
        listbox.insert(tk.END, label)
    listbox.pack(padx=10, pady=10)

    btn_frame = Frame(root)
    btn_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    Button(btn_frame, text="Select All", command=select_all).pack(side="left", padx=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(side="left", padx=5)

    Button(root, text="Select", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    # Return the marker names corresponding to the selected indices
    return [marker_labels[int(i)] for i in selected_indices]


###############################################################################
# Function: select_headers_gui
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def select_headers_gui(headers):
    """
    Displays a Tkinter window with a Listbox for header (column) selection.

    Args:
        headers (list): List of CSV headers.

    Returns:
        List of selected headers.
    """
    root = tk.Tk()
    root.title("Select CSV Headers")
    listbox = Listbox(root, selectmode="multiple", width=50, height=15)
    for header in headers:
        listbox.insert(tk.END, header)
    listbox.pack(padx=10, pady=10)

    btn_frame = Frame(root)
    btn_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    Button(btn_frame, text="Select All", command=select_all).pack(side="left", padx=5)
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(side="left", padx=5)

    Button(root, text="OK", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    return [headers[int(i)] for i in selected_indices]


###############################################################################
# Function: get_csv_headers
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def get_csv_headers(file_path):
    """
    Reads the CSV file at the given file_path and returns its headers (column names).

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List of headers (str).
    """
    try:
        df = pd.read_csv(file_path, nrows=0)
        return list(df.columns)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read CSV headers: {e}")
        return []


###############################################################################
# Function: show_csv_open3d
# Visualizes marker data using Open3D (similar to viewc3d.py).
###############################################################################
def show_csv_open3d(points, marker_names, fps=30):
    """
    Visualizes CSV marker data using Open3D.

    Args:
        points: numpy array of shape (num_frames, num_markers, 3)
        marker_names: list of marker names corresponding to the second dimension
        fps: frames per second for animation
    """
    # Open3D visualization stub - not yet implemented
    pass

    num_frames, num_markers, _ = points.shape
    print(f"Open3D visualization stub: {num_frames} frames, {num_markers} markers.")
    # Here you would implement the actual Open3D visualization.
    # For this stub, we just simulate a delay.
    time.sleep(2)
    print("Open3D visualization complete.")


###############################################################################
# Function: show_csv_matplotlib
# Visualizes marker data using Matplotlib (similar to showc3d.py).
###############################################################################
def show_csv_matplotlib(points, marker_names, fps=30):
    """
    Visualizes CSV marker data using Matplotlib.

    Args:
        points: numpy array of shape (num_frames, num_markers, 3)
        marker_names: list of marker names corresponding to the second dimension
        fps: frames per second for the playback animation
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button as MplButton
        from matplotlib.widgets import Slider
    except ImportError:
        print("matplotlib is not installed. Please install it with 'pip install matplotlib'.")
        return

    num_frames, num_markers, _ = points.shape
    fig = plt.figure(figsize=(10, 8))
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    ax.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2], c="blue", s=20)
    ax.set_title(f"CSV Data Visualization (Matplotlib) | Frames: {num_frames} | FPS: {fps}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    x_min, x_max = points[:, :, 0].min(), points[:, :, 0].max()
    y_min, y_max = points[:, :, 1].min(), points[:, :, 1].max()
    z_min, z_max = points[:, :, 2].min(), points[:, :, 2].max()
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_zlim((z_min, z_max))

    ax_slider = plt.axes((0.25, 0.02, 0.5, 0.03))
    slider = Slider(ax_slider, "Frame", 0, num_frames - 1, valinit=0, valfmt="%d")

    current_frame = [0]

    def update_frame(val):
        # Update the scatter plot with the new points of the selected frame.
        frame = int(slider.val) if isinstance(val, float) else int(val)
        current_frame[0] = frame
        new_positions = points[frame]

        # Clear and redraw
        ax.clear()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Filter out NaN values for this frame
        valid_mask = ~np.isnan(new_positions).any(axis=1)
        valid_positions = new_positions[valid_mask]

        if len(valid_positions) > 0:
            ax.scatter(
                valid_positions[:, 0],
                valid_positions[:, 1],
                zs=valid_positions[:, 2],
                c="blue",
                s=20,
            )

        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        ax.set_zlim((z_min, z_max))
        fig.canvas.draw_idle()

    slider.on_changed(update_frame)

    # Variables for automatic playback control
    playing = [False]
    timer: list[TimerBase | None] = [None]

    def timer_callback():
        current_frame[0] = (current_frame[0] + 1) % num_frames
        slider.set_val(current_frame[0])
        update_frame(current_frame[0])

    def play_pause(event):
        if not playing[0]:
            playing[0] = True
            btn_play.label.set_text("Pause")
            timer[0] = fig.canvas.new_timer(interval=int(1000 / 30))  # Assuming 30 fps
            try:
                timer[0].single_shot = False
            except AttributeError:
                pass
            timer[0].add_callback(timer_callback)
            timer[0].start()
        else:
            playing[0] = False
            btn_play.label.set_text("Play")
            if timer[0] is not None:
                timer[0].stop()
                timer[0] = None

    # Create play button
    ax_play = fig.add_axes((0.82, 0.02, 0.1, 0.05))
    btn_play = MplButton(ax_play, "Play")
    btn_play.on_clicked(play_pause)
    # Add record button
    ax_record = plt.axes((0.82, 0.08, 0.1, 0.05))  # Keep consistent with ax_slider format
    btn_record = MplButton(ax_record, "Record")

    def record_animation(event):
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")],
                title="Save animation as",
            )
            if not file_path:
                return

            # Create animation writer
            writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="VAILA"), bitrate=1800)

            # Show recording message
            btn_record.label.set_text("Recording...")
            fig.canvas.draw_idle()

            # Create animation
            def update(frame):
                new_positions = points[frame]

                # Clear and redraw
                ax.clear()
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                # Filter out NaN values for this frame
                valid_mask = ~np.isnan(new_positions).any(axis=1)
                valid_positions = new_positions[valid_mask]

                if len(valid_positions) > 0:
                    ax.scatter(
                        valid_positions[:, 0],
                        valid_positions[:, 1],
                        valid_positions[:, 2],
                        c="blue",
                        s=20,
                    )

                ax.set_xlim((x_min, x_max))
                ax.set_ylim((y_min, y_max))
                ax.set_zlim((z_min, z_max))
                return (ax,)

            anim = animation.FuncAnimation(
                fig, update, frames=num_frames, interval=1000 / 30, blit=True
            )

            # Save animation
            anim.save(file_path, writer=writer)

            # Reset button text
            btn_record.label.set_text("Record")
            fig.canvas.draw_idle()

            messagebox.showinfo("Success", "Animation saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save animation: {e}")
            btn_record.label.set_text("Record")
            fig.canvas.draw_idle()

    btn_record.on_clicked(record_animation)

    # Add space key functionality
    def on_key(event):
        if event.key == " ":
            play_pause(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


###############################################################################
# Function: detect_units
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def detect_units(points):
    """
    Detects if the data is in millimeters or meters based on the magnitude of values.

    Args:
        points (numpy.ndarray): Array of shape (num_frames, num_markers, 3) containing the coordinates

    Returns:
        bool: True if data is in millimeters (needs conversion), False if already in meters
    """
    # Calculate the mean absolute value of all coordinates
    mean_abs = np.mean(np.abs(points))

    # If mean absolute value is greater than 100, likely in millimeters
    return mean_abs > 100


def ask_user_units():
    """
    Ask user about the units of the coordinate data.

    Returns:
        str: 'mm' for millimeters, 'm' for meters, 'auto' for automatic detection
    """
    import tkinter as tk
    from tkinter import messagebox

    try:
        # Try simple dialog first (more compatible)
        root = tk.Tk()
        root.withdraw()  # Hide the main window

        # Simple approach using messagebox
        response = messagebox.askyesnocancel(
            "Select Coordinate Units",
            "What are the units of your coordinate data?\n\n"
            "• YES = Millimeters (mm) - typical values like 1234.56\n"
            "• NO = Meters (m) - typical values like 1.23\n"
            "• CANCEL = Auto-detect (let program guess)\n\n"
            "Choose based on your data range shown in terminal.",
        )

        root.destroy()

        if response is True:
            return "mm"
        elif response is False:
            return "m"
        else:
            return "auto"

    except Exception as e:
        print(f"Warning: Could not show unit selection dialog: {e}")
        print("Using auto-detection for units.")
        return "auto"


def read_csv_generic(file_path):
    """
    Reads a CSV or TXT file considering:
      - Automatically detects the delimiter (',', ';', '\t', ' ')
      - Detects if the file has a header
      - If it doesn't have a header, uses default names (Index, p1_x, p1_y, p1_z, p2_x, ...)
      - The first column contains any index data (time, frames, or any identifier)
      - The subsequent columns are organized in groups of three (x, y, z)
      - Asks user about units and converts accordingly

    Returns:
      index_vector: pd.Series with the first column data (time/frames/index)
      marker_data: dictionary mapping the marker name to a numpy array Nx3
      valid_markers: dictionary mapping the marker name to the list of columns used
      delimiter: the detected delimiter
    """
    print(f"Processing file: {file_path}")

    # Detect delimiter
    delimiter = detect_delimiter(file_path)

    # Detect if file has header
    has_header = detect_has_header(file_path, delimiter)
    print(f"Header detected: {has_header}")

    # Read the file
    try:
        if has_header:
            df = pd.read_csv(file_path, delimiter=delimiter)
        else:
            # Create default column names for files without headers
            # First read to count columns
            temp_df = pd.read_csv(file_path, delimiter=delimiter, nrows=1)
            num_columns = len(temp_df.columns)

            default_columns = ["Index"]  # Generic name for first column
            for i in range(1, (num_columns - 1) // 3 + 1):
                default_columns.extend([f"p{i}_x", f"p{i}_y", f"p{i}_z"])

            # Add remaining columns if any
            while len(default_columns) < num_columns:
                default_columns.append(f"col_{len(default_columns)}")

            df = pd.read_csv(file_path, delimiter=delimiter, names=default_columns)

        print(f"Successfully read CSV with shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    if df.empty:
        raise ValueError("The file is empty or could not be read.")

    # The first column is generic (time/frames/index/etc.)
    index_vector = df.iloc[:, 0]

    # Process the remaining columns: each column must have the format marker_coord (ex.: MARKER_X)
    # Support multiple naming conventions
    marker_headers = {}
    for col in df.columns[1:]:
        marker_name = None
        coord = None

        # Try different naming patterns
        # Pattern 1: marker_X, marker_Y, marker_Z or marker_x, marker_y, marker_z
        if "_" in col:
            parts = col.rsplit("_", 1)  # split by the last occurrence of '_'
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()

        # Pattern 2: markerX, markerY, markerZ (no underscore)
        elif col[-1].upper() in ["X", "Y", "Z"]:
            marker_name = col[:-1]
            coord = col[-1].upper()

        # Pattern 3: marker.x, marker.y, marker.z
        elif "." in col:
            parts = col.rsplit(".", 1)
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()

        # Pattern 4: marker:x, marker:y, marker:z
        elif ":" in col:
            parts = col.rsplit(":", 1)
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                coord = parts[1].upper()

        if marker_name:
            if marker_name not in marker_headers:
                marker_headers[marker_name] = {}
            marker_headers[marker_name][coord] = col
        else:
            print(f"Warning: Could not parse column '{col}' - no matching pattern found")

    print(f"Found marker patterns: {list(marker_headers.keys())}")

    # Debug: Show details of marker parsing
    if len(marker_headers) <= 10:  # Only show details if reasonable number
        for marker, coords in marker_headers.items():
            print(f"  {marker}: {coords}")

    # Select only the markers that have the complete set of 3 columns
    valid_markers = {}
    for marker, coord_dict in marker_headers.items():
        if len(coord_dict) == 3 and all(coord in coord_dict for coord in ["X", "Y", "Z"]):
            # Ensure correct order: X, Y, Z
            ordered_cols = [coord_dict["X"], coord_dict["Y"], coord_dict["Z"]]
            valid_markers[marker] = ordered_cols
        else:
            available_coords = list(coord_dict.keys()) if coord_dict else []
            print(
                f"Warning: The marker '{marker}' has incomplete data. Available coordinates: {available_coords}"
            )

    print(f"Valid markers found: {list(valid_markers.keys())}")

    # Extract the data for each marker into an Nx3 array
    marker_data = {}
    for marker, cols in valid_markers.items():
        marker_data[marker] = df[cols].to_numpy()

    # Handle unit conversion based on user input
    if valid_markers:
        # Create a temporary array with all points to check units
        temp_points = np.stack([marker_data[marker] for marker in valid_markers], axis=1)

        # Calculate statistics for user information
        valid_coords = temp_points[~np.isnan(temp_points)]
        if len(valid_coords) > 0:
            mean_abs_value = np.mean(np.abs(valid_coords))
            min_value = np.min(valid_coords)
            max_value = np.max(valid_coords)

            print("\nCoordinate data statistics:")
            print(f"  Mean absolute value: {mean_abs_value:.3f}")
            print(f"  Range: [{min_value:.3f}, {max_value:.3f}]")
            print("  Data preview (first marker, first 3 records):")
            first_marker = list(marker_data.keys())[0]
            preview_data = marker_data[first_marker][:3]
            for i, coords in enumerate(preview_data):
                if not np.isnan(coords).any():
                    print(
                        f"    Record {i}: X={coords[0]:.6f}, Y={coords[1]:.6f}, Z={coords[2]:.6f}"
                    )

            # Ask user about units
            print("\nPlease select units in the dialog box...")
            user_units = ask_user_units()
            print(f"User selected: {user_units}")

            if user_units == "mm":
                print("Converting from millimeters to meters...")
                # Convert all marker data from millimeters to meters
                for marker in marker_data:
                    marker_data[marker] = marker_data[marker] * 0.001
                print("Unit conversion completed: mm → m")

                # Show converted preview
                print("After conversion (first marker, first 3 records):")
                preview_data = marker_data[first_marker][:3]
                for i, coords in enumerate(preview_data):
                    if not np.isnan(coords).any():
                        print(
                            f"    Record {i}: X={coords[0]:.6f}, Y={coords[1]:.6f}, Z={coords[2]:.6f}"
                        )

        elif user_units == "m":
            print("No conversion needed - data is already in meters.")

        elif user_units == "auto":
            print("Using auto-detection for units...")
            if detect_units(temp_points):
                print("Auto-detection: Data appears to be in millimeters. Converting to meters...")
                # Convert all marker data from millimeters to meters
                for marker in marker_data:
                    marker_data[marker] = marker_data[marker] * 0.001
                print("Unit conversion completed: mm → m")
            else:
                print("Auto-detection: Data appears to be already in meters.")

        # Check for and filter extreme outliers after unit conversion
        print("\nChecking for extreme outliers...")
        temp_points_converted = np.stack(
            [marker_data[marker] for marker in valid_markers], axis=1
        )
        valid_coords = temp_points_converted[~np.isnan(temp_points_converted)]

        if len(valid_coords) > 0:
            # Calculate statistics to detect outliers
            q25 = np.percentile(valid_coords, 25)
            q75 = np.percentile(valid_coords, 75)
            iqr = q75 - q25
            median = np.median(valid_coords)

            # Define outlier bounds (more conservative for coordinate data)
            outlier_threshold = 10 * iqr  # Allow larger range for coordinate data
            lower_bound = q25 - outlier_threshold
            upper_bound = q75 + outlier_threshold

            # Count outliers
            outliers = (valid_coords < lower_bound) | (valid_coords > upper_bound)
            outlier_count = np.sum(outliers)
            outlier_percentage = (outlier_count / len(valid_coords)) * 100

            print("Data quality check:")
            print(f"  Median: {median:.3f}")
            print(f"  Q25-Q75: [{q25:.3f}, {q75:.3f}]")
            print(f"  IQR: {iqr:.3f}")
            print(f"  Outlier bounds: [{lower_bound:.3f}, {upper_bound:.3f}]")
            print(
                f"  Outliers found: {outlier_count}/{len(valid_coords)} ({outlier_percentage:.1f}%)"
            )

            if outlier_percentage > 5:  # If more than 5% outliers
                print("Warning: High percentage of outliers detected!")
                print("This may indicate data quality issues or incorrect units.")
                print("Consider reviewing your data source.")

    else:
        print("Warning: No valid coordinate data found.")

    return index_vector, marker_data, valid_markers, delimiter


###############################################################################
# Function: show_csv (Main Function)
# - Opens a file selection dialog to pick the CSV file.
# - Extracts the marker names (ignoring the first "Time" column).
# - Opens a marker selection dialog.
# - Constructs an array of marker positions of shape (num_frames, num_markers, 3).
# - Prompts the user to choose a visualization method.
# - Launches the visualization using either Open3D or Matplotlib.
###############################################################################
def show_csv(file_path=None):
    """
    Main function to load the CSV, select the markers and plot the data.

    Args:
        file_path (str, optional): Path to the CSV file. If None, opens file dialog.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Running CSV viewer")
    print("================================================")

    # If no file path provided, show file dialog
    if file_path is None:
        root = tk.Tk()
        root.withdraw()
        file_path = select_file()
        if not file_path:
            print("No file selected.")
            return
    else:
        # Validate file exists
        if not os.path.exists(file_path):
            print("[bold red]Error:[/bold red] File not found: {file_path}")
            return
        if not os.path.isfile(file_path):
            print(f"[bold red]Error:[/bold red] Path is not a file: {file_path}")
            return

    try:
        index_vector, marker_data, valid_markers, delimiter = read_csv_generic(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error reading the CSV file: {e}")
        return

    # List the available markers
    available_markers = list(valid_markers.keys())
    print("\n[bold green]File loaded successfully![/bold green]")
    print(f"File: {file_path}")
    print(f"Delimiter detected: '{delimiter}'")
    print(f"First column: '{index_vector.name}' (contains {index_vector.dtype} data)")
    print(f"Number of records: {len(index_vector)}")
    print(f"Number of markers found: {len(available_markers)}")
    print("\n[bold]Available markers:[/bold]")
    for i, marker in enumerate(available_markers, 1):
        print(f"  {i:2d}. {marker}")

    # Allow the user to select the markers to be visualized (multiple selection)
    selected_markers = select_markers_csv(available_markers)
    if not selected_markers:
        messagebox.showwarning("Warning", "No markers selected.")
        return

    # Build an array of points with shape (num_frames, num_markers, 3)
    # CORREÇÃO: Garantir que as colunas estão na ordem correta X, Y, Z
    points_list = []
    for marker in selected_markers:
        marker_coords = marker_data[marker]  # Nx3 array [X, Y, Z]
        points_list.append(marker_coords)

    points = np.stack(points_list, axis=1)  # Shape: (num_frames, num_markers, 3)
    num_frames = points.shape[0]
    num_markers = points.shape[1]

    file_name = os.path.basename(file_path)

    # Debug: Print first few coordinate values to verify correctness
    print("\n[bold]Debug Info:[/bold]")
    print(f"Points array shape: {points.shape}")
    if num_markers > 0:
        first_marker = selected_markers[0]
        first_frame_coords = points[0, 0, :]
        print(f"First marker '{first_marker}' coordinates at first record:")
        print(f"  X: {first_frame_coords[0]:.6f}")
        print(f"  Y: {first_frame_coords[1]:.6f}")
        print(f"  Z: {first_frame_coords[2]:.6f}")

    # Create the 3D figure with improved layout
    fig = plt.figure(figsize=(14, 10))
    ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
    # Ajustar posição do plot para dar mais espaço aos controles
    ax.set_position((0.0, 0.25, 0.75, 0.75))  # Mais espaço embaixo para controles

    # Filter out NaN values for initial plotting
    valid_mask = ~np.isnan(points[0]).any(axis=1)
    valid_points_frame0 = points[0][valid_mask]

    if len(valid_points_frame0) > 0:
        ax.scatter(
            valid_points_frame0[:, 0],
            valid_points_frame0[:, 1],
            valid_points_frame0[:, 2],
            c="blue",
            s=30,
        )
    else:
        ax.scatter(0, 0, 0, c="blue", s=30, alpha=0)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Dynamic title
    first_column_name = index_vector.name if index_vector.name else "Index"
    ax.set_title(
        f"CSV Viewer | File: {file_name} | Markers: {len(selected_markers)}/{len(available_markers)} | Records: {num_frames}"
    )

    # Calculate initial limits from data, excluding NaN values
    valid_points_all = points[~np.isnan(points)]
    if len(valid_points_all) > 0:
        x_valid = points[:, :, 0][~np.isnan(points[:, :, 0])]
        y_valid = points[:, :, 1][~np.isnan(points[:, :, 1])]
        z_valid = points[:, :, 2][~np.isnan(points[:, :, 2])]

        if len(x_valid) > 0 and len(y_valid) > 0 and len(z_valid) > 0:
            x_min, x_max = x_valid.min(), x_valid.max()
            y_min, y_max = y_valid.min(), y_valid.max()
            z_min, z_max = z_valid.min(), z_valid.max()
        else:
            x_min, x_max = -1, 1
            y_min, y_max = -1, 1
            z_min, z_max = -1, 1
    else:
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        z_min, z_max = -1, 1

    # Add padding to the limits
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    z_range = max(z_max - z_min, 0.1)
    padding = 0.1

    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding

    # Set initial limits
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_zlim((z_min, z_max))

    print("Final coordinate ranges:")
    print(f"  X: [{x_min:.6f}, {x_max:.6f}]")
    print(f"  Y: [{y_min:.6f}, {y_max:.6f}]")
    print(f"  Z: [{z_min:.6f}, {z_max:.6f}]")

    # Define equal aspect
    ax.set_aspect("equal")

    # CORREÇÃO: Reorganizar layout para evitar sobreposição
    # Controles de limite - parte superior esquerda
    def update_x_limits(text):
        try:
            x_min_new, x_max_new = map(float, text.split(","))
            if x_min_new < x_max_new:
                nonlocal x_min, x_max
                x_min, x_max = x_min_new, x_max_new
                ax.set_xlim((x_min, x_max))
                fig.canvas.draw_idle()
        except ValueError:
            pass

    def update_y_limits(text):
        try:
            y_min_new, y_max_new = map(float, text.split(","))
            if y_min_new < y_max_new:
                nonlocal y_min, y_max
                y_min, y_max = y_min_new, y_max_new
                ax.set_ylim((y_min, y_max))
                fig.canvas.draw_idle()
        except ValueError:
            pass

    def update_z_limits(text):
        try:
            z_min_new, z_max_new = map(float, text.split(","))
            if z_min_new < z_max_new:
                nonlocal z_min, z_max
                z_min, z_max = z_min_new, z_max_new
                ax.set_zlim((z_min, z_max))
                fig.canvas.draw_idle()
        except ValueError:
            pass

    def reset_limits(event):
        nonlocal x_min, x_max, y_min, y_max, z_min, z_max
        # Recalculate original limits
        valid_points_all = points[~np.isnan(points)]
        if len(valid_points_all) > 0:
            x_valid = points[:, :, 0][~np.isnan(points[:, :, 0])]
            y_valid = points[:, :, 1][~np.isnan(points[:, :, 1])]
            z_valid = points[:, :, 2][~np.isnan(points[:, :, 2])]

            if len(x_valid) > 0 and len(y_valid) > 0 and len(z_valid) > 0:
                x_min_orig, x_max_orig = x_valid.min(), x_valid.max()
                y_min_orig, y_max_orig = y_valid.min(), y_valid.max()
                z_min_orig, z_max_orig = z_valid.min(), z_valid.max()

                x_range = max(x_max_orig - x_min_orig, 0.1)
                y_range = max(y_max_orig - y_min_orig, 0.1)
                z_range = max(z_max_orig - z_min_orig, 0.1)

                x_min = x_min_orig - x_range * 0.1
                x_max = x_max_orig + x_range * 0.1
                y_min = y_min_orig - y_range * 0.1
                y_max = y_max_orig + y_range * 0.1
                z_min = z_min_orig - z_range * 0.1
                z_max = z_max_orig + z_range * 0.1

        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        ax.set_zlim((z_min, z_max))
        textbox_x.set_val(f"{x_min:.2f},{x_max:.2f}")
        textbox_y.set_val(f"{y_min:.2f},{y_max:.2f}")
        textbox_z.set_val(f"{z_min:.2f},{z_max:.2f}")
        fig.canvas.draw_idle()

    # Text boxes para limites - posicionamento corrigido
    ax_textbox_x = fig.add_axes((0.02, 0.20, 0.12, 0.03))
    ax_textbox_y = fig.add_axes((0.02, 0.16, 0.12, 0.03))
    ax_textbox_z = fig.add_axes((0.02, 0.12, 0.12, 0.03))
    ax_reset = fig.add_axes((0.15, 0.12, 0.06, 0.08))

    textbox_x = TextBox(ax_textbox_x, "X:", initial=f"{x_min:.2f},{x_max:.2f}")
    textbox_y = TextBox(ax_textbox_y, "Y:", initial=f"{y_min:.2f},{y_max:.2f}")
    textbox_z = TextBox(ax_textbox_z, "Z:", initial=f"{z_min:.2f},{z_max:.2f}")
    btn_reset = MplButton(ax_reset, "Reset\nLimits")

    textbox_x.on_submit(update_x_limits)
    textbox_y.on_submit(update_y_limits)
    textbox_z.on_submit(update_z_limits)
    btn_reset.on_clicked(reset_limits)

    # Slider para controle de frame - posicionamento corrigido
    ax_frame = fig.add_axes((0.25, 0.02, 0.4, 0.04))
    slider_frame = Slider(ax_frame, "Frame", 0, num_frames - 1, valinit=0, valfmt="%d")

    # Speed control - separado dos limites
    ax_speed = fig.add_axes((0.25, 0.08, 0.4, 0.03))
    slider_speed = Slider(ax_speed, "Speed (FPS)", 10, 120, valinit=60, valfmt="%d")

    # Variáveis de controle
    current_frame = [0]
    playing = [False]
    timer: list[TimerBase | None] = [None]
    playback_speed = [60]
    show_labels = [False]
    show_connections = [False]
    show_trajectory = [False]
    show_legend = [True]
    color_mode = [0]  # 0: blue, 1: multicolor
    trajectory_length = [30]
    connections = []

    # Pre-generate color schemes
    multicolor_colors = plt.cm.tab10(np.linspace(0, 1, min(num_markers, 10)))
    if num_markers > 10:
        extra_colors = plt.cm.Set3(np.linspace(0, 1, num_markers - 10))
        multicolor_colors = np.vstack([multicolor_colors, extra_colors])

    def update_frame(val):
        frame = int(slider_frame.val) if isinstance(val, float) else int(val)
        current_frame[0] = frame
        new_positions = points[frame]

        # Clear and redraw
        ax.clear()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Updated title to show current index value
        current_index_value = index_vector.iloc[frame]
        if pd.api.types.is_numeric_dtype(index_vector):
            index_display = f"{current_index_value:.3f}"
        else:
            index_display = str(current_index_value)

        ax.set_title(f"Record {frame}/{num_frames - 1} | {first_column_name}: {index_display}")

        # Plot markers
        for i, (marker_pos, marker_name) in enumerate(zip(new_positions, selected_markers)):
            if not np.isnan(marker_pos).any():
                # Only add label if legend is enabled and we have reasonable number of markers
                marker_label = marker_name if (show_legend[0] and num_markers <= 15) else None

                if color_mode[0] == 0:
                    # Blue mode
                    ax.scatter(
                        marker_pos[0],
                        marker_pos[1],
                        marker_pos[2],
                        c="blue",
                        s=40,
                        label=marker_label,
                        alpha=0.8,
                    )
                else:
                    # Multicolor mode
                    color_idx = i % len(multicolor_colors)
                    color = multicolor_colors[color_idx]
                    ax.scatter(
                        marker_pos[0],
                        marker_pos[1],
                        marker_pos[2],
                        c=[color],
                        s=40,
                        label=marker_label,
                        alpha=0.8,
                    )

                # Add marker name as text if enabled
                if show_labels[0]:
                    ax.text(
                        marker_pos[0],
                        marker_pos[1],
                        marker_pos[2],
                        f"  {marker_name}",
                        fontsize=8,
                        alpha=0.7,
                    )

        # Draw connections if enabled
        if show_connections[0] and len(connections) > 0:
            for conn in connections:
                idx1, idx2 = conn
                if idx1 < len(new_positions) and idx2 < len(new_positions):
                    pos1, pos2 = new_positions[idx1], new_positions[idx2]
                    if not np.isnan(pos1).any() and not np.isnan(pos2).any():
                        ax.plot(
                            [pos1[0], pos2[0]],
                            [pos1[1], pos2[1]],
                            [pos1[2], pos2[2]],
                            "k-",
                            alpha=0.3,
                            linewidth=1,
                        )

        # Show trajectory if enabled
        if show_trajectory[0] and frame > 0:
            for i in range(num_markers):
                trail_length = min(trajectory_length[0], frame)
                trail_start = max(0, frame - trail_length)
                trail = points[trail_start : frame + 1, i, :]

                valid_trail = trail[~np.isnan(trail).any(axis=1)]
                if len(valid_trail) > 1:
                    if color_mode[0] == 0:
                        color = "blue"
                    else:
                        color_idx = i % len(multicolor_colors)
                        color = multicolor_colors[color_idx]
                    ax.plot(
                        valid_trail[:, 0],
                        valid_trail[:, 1],
                        valid_trail[:, 2],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                    )

        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        ax.set_zlim((z_min, z_max))

        # Add legend if enabled and we have reasonable number of markers
        if show_legend[0] and num_markers <= 15:
            # Get handles and labels, filter out None labels
            handles, labels = ax.get_legend_handles_labels()
            if handles and labels:  # Only create legend if we have valid handles and labels
                # Filter out empty labels
                filtered_handles_labels = [
                    (h, label)
                    for h, label in zip(handles, labels)
                    if label is not None and label != ""
                ]
                if filtered_handles_labels:
                    filtered_handles, filtered_labels = zip(*filtered_handles_labels)
                    ax.legend(
                        filtered_handles,
                        filtered_labels,
                        loc="upper right",
                        fontsize=8,
                        framealpha=0.8,
                        fancybox=True,
                        shadow=True,
                    )

        fig.canvas.draw_idle()

    slider_frame.on_changed(update_frame)

    def update_speed(val):
        playback_speed[0] = int(slider_speed.val)
        if playing[0] and timer[0] is not None:
            timer[0].stop()
            timer[0] = fig.canvas.new_timer(interval=int(1000 / playback_speed[0]))
            try:
                timer[0].single_shot = False
            except AttributeError:
                pass
            timer[0].add_callback(timer_callback)
            timer[0].start()

    slider_speed.on_changed(update_speed)

    def timer_callback():
        current_frame[0] = (current_frame[0] + 1) % num_frames
        slider_frame.set_val(current_frame[0])

    def play_pause(event):
        if not playing[0]:
            playing[0] = True
            btn_play.label.set_text("Pause")
            timer[0] = fig.canvas.new_timer(interval=int(1000 / playback_speed[0]))
            try:
                timer[0].single_shot = False
            except AttributeError:
                pass
            timer[0].add_callback(timer_callback)
            timer[0].start()
        else:
            playing[0] = False
            btn_play.label.set_text("Play")
            if timer[0] is not None:
                timer[0].stop()
                timer[0] = None

    # Control buttons - posicionamento à direita sem sobreposição
    controls_x_start = 0.78
    button_width = 0.1
    button_height = 0.04
    button_spacing = 0.05

    # Play/Pause button
    ax_play = fig.add_axes((controls_x_start, 0.02, button_width, button_height))
    btn_play = MplButton(ax_play, "Play")
    btn_play.on_clicked(play_pause)

    # Color mode button
    ax_color = fig.add_axes((controls_x_start, 0.02 + button_spacing, button_width, button_height))
    btn_color = MplButton(ax_color, "Blue Mode")

    def toggle_color_mode(event):
        color_mode[0] = (color_mode[0] + 1) % 2
        if color_mode[0] == 0:
            btn_color.label.set_text("Blue Mode")
        else:
            btn_color.label.set_text("Color Mode")
        update_frame(current_frame[0])

    btn_color.on_clicked(toggle_color_mode)

    # Labels button
    ax_labels = fig.add_axes(
        (controls_x_start, 0.02 + 2 * button_spacing, button_width, button_height)
    )
    btn_labels = MplButton(ax_labels, "Labels OFF")

    def toggle_labels(event):
        show_labels[0] = not show_labels[0]
        btn_labels.label.set_text("Labels ON" if show_labels[0] else "Labels OFF")
        update_frame(current_frame[0])

    btn_labels.on_clicked(toggle_labels)

    # Trajectory button
    ax_trajectory = fig.add_axes(
        (controls_x_start, 0.02 + 3 * button_spacing, button_width, button_height)
    )
    btn_trajectory = MplButton(ax_trajectory, "Trails OFF")

    def toggle_trajectory(event):
        show_trajectory[0] = not show_trajectory[0]
        btn_trajectory.label.set_text("Trails ON" if show_trajectory[0] else "Trails OFF")
        update_frame(current_frame[0])

    btn_trajectory.on_clicked(toggle_trajectory)

    # Legend button
    ax_legend = fig.add_axes(
        (controls_x_start, 0.02 + 4 * button_spacing, button_width, button_height)
    )
    initial_legend_text = "Legend ON" if show_legend[0] else "Legend OFF"
    btn_legend = MplButton(ax_legend, initial_legend_text)

    def toggle_legend(event):
        show_legend[0] = not show_legend[0]
        btn_legend.label.set_text("Legend ON" if show_legend[0] else "Legend OFF")
        update_frame(current_frame[0])

        # Show info about legend limits
        if show_legend[0] and num_markers > 15:
            print(
                f"Warning: Legend disabled for {num_markers} markers (limit: 15). Use fewer markers or text labels instead."
            )

    btn_legend.on_clicked(toggle_legend)

    # Record button
    ax_record = fig.add_axes(
        (controls_x_start, 0.02 + 5 * button_spacing, button_width, button_height)
    )
    btn_record = MplButton(ax_record, "Record")

    def record_animation(event):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")],
                title="Save animation as",
            )
            if not file_path:
                return

            writer = animation.FFMpegWriter(fps=30, metadata=dict(artist="VAILA"), bitrate=1800)

            btn_record.label.set_text("Recording...")
            fig.canvas.draw_idle()

            def update_for_record(frame):
                new_positions = points[frame]

                ax.clear()
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_title(f"Frame {frame}/{num_frames - 1}")

                for i, (marker_pos, marker_name) in enumerate(zip(new_positions, selected_markers)):
                    if not np.isnan(marker_pos).any():
                        if color_mode[0] == 0:
                            ax.scatter(
                                marker_pos[0],
                                marker_pos[1],
                                marker_pos[2],
                                c="blue",
                                s=40,
                                alpha=0.8,
                            )
                        else:
                            color_idx = i % len(multicolor_colors)
                            color = multicolor_colors[color_idx]
                            ax.scatter(
                                marker_pos[0],
                                marker_pos[1],
                                marker_pos[2],
                                c=[color],
                                s=40,
                                alpha=0.8,
                            )

                ax.set_xlim((x_min, x_max))
                ax.set_ylim((y_min, y_max))
                ax.set_zlim((z_min, z_max))
                return (ax,)

            anim = animation.FuncAnimation(
                fig, update_for_record, frames=num_frames, interval=1000 / 30, blit=True
            )

            anim.save(file_path, writer=writer)

            btn_record.label.set_text("Record")
            fig.canvas.draw_idle()

            messagebox.showinfo("Success", "Animation saved successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save animation: {e}")
            btn_record.label.set_text("Record")
            fig.canvas.draw_idle()

    btn_record.on_clicked(record_animation)

    # Export button
    ax_export = fig.add_axes(
        (controls_x_start, 0.02 + 6 * button_spacing, button_width, button_height)
    )
    btn_export = MplButton(ax_export, "Export Data")

    def export_data(event):
        try:
            export_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export processed data as",
            )
            if not export_path:
                return

            export_df = pd.DataFrame()
            export_df[first_column_name] = index_vector

            for i, marker in enumerate(selected_markers):
                export_df[f"{marker}_X"] = points[:, i, 0]
                export_df[f"{marker}_Y"] = points[:, i, 1]
                export_df[f"{marker}_Z"] = points[:, i, 2]

            export_df.to_csv(export_path, index=False)

            print("\n[bold green]Data exported successfully![/bold green]")
            print(f"File: {export_path}")
            print(f"Records: {num_frames}, Markers: {len(selected_markers)}")

            messagebox.showinfo("Success", f"Data exported successfully to:\n{export_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export data: {e}")

    btn_export.on_clicked(export_data)

    # Add keyboard shortcuts
    def on_key(event):
        if event.key == " ":
            play_pause(None)
        elif event.key == "c":
            toggle_color_mode(None)
        elif event.key == "l":
            toggle_labels(None)
        elif event.key == "t":
            toggle_trajectory(None)
        elif event.key == "right":
            new_frame = min(current_frame[0] + 1, num_frames - 1)
            slider_frame.set_val(new_frame)
        elif event.key == "left":
            new_frame = max(current_frame[0] - 1, 0)
            slider_frame.set_val(new_frame)
        elif event.key == "up":
            new_frame = min(current_frame[0] + 10, num_frames - 1)
            slider_frame.set_val(new_frame)
        elif event.key == "down":
            new_frame = max(current_frame[0] - 10, 0)
            slider_frame.set_val(new_frame)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Add instructions text
    instructions = (
        "Keyboard Shortcuts:\n"
        "Space: Play/Pause\n"
        "C: Toggle colors\n"
        "L: Toggle labels\n"
        "T: Toggle trails\n"
        "←→: ±1 record\n"
        "↑↓: ±10 records"
    )

    fig.text(
        0.02,
        0.95,
        instructions,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    plt.show()


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    import argparse
    import sys

    # Enhanced command line interface with performance options
    if len(sys.argv) > 1 and sys.argv[1].startswith("--"):
        # Parse arguments
        parser = argparse.ArgumentParser(description="CSV Viewer with performance options")
        parser.add_argument("file", nargs="?", help="CSV file path")
        parser.add_argument("--no-cache", action="store_true", help="Disable NPZ cache")
        parser.add_argument(
            "--turbo", action="store_true", help="Use turbo mode (faster visualization)"
        )
        parser.add_argument("--fps", type=int, default=30, help="Frames per second")
        parser.add_argument(
            "--stride", type=int, default=1, help="Skip frames (1=all, 2=every other)"
        )

        args = parser.parse_args()

        if args.file:
            show_csv_optimized(
                args.file,
                use_cache=not args.no_cache,
                turbo=args.turbo,
                fps=args.fps,
                stride=args.stride,
            )
        else:
            show_csv()
    else:
        # Original simple interface
        if len(sys.argv) > 1:
            file_path = sys.argv[1]
            show_csv(file_path)
        else:
            show_csv()
