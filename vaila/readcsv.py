"""
readcsv.py

Name: Your Name
Date: 29/07/2024

Description:
Script to visualize data from .csv files using Open3D and Matplotlib,
with marker selection interface and frame animation. This module is now similar
to viewc3d.py and showc3d.py for visualization.

Version: 0.2
"""

import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, Toplevel, Button, Label, Listbox, Frame, messagebox
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button as MplButton, TextBox
from matplotlib import animation
from rich import print


###############################################################################
# Function: headersidx
# (Adicionado para manter compatibilidade com vaila/__init__.py)
###############################################################################
def headersidx(headers, prefix):
    """
    Dada uma lista de cabeçalhos e um prefixo, retorna os índices dos cabeçalhos que começam com o prefixo.

    Args:
        headers (list): Lista de nomes de cabeçalhos.
        prefix (str): Prefixo a ser verificado.

    Returns:
        List[int]: Lista de índices dos cabeçalhos que começam com o prefixo.
    """
    return [i for i, header in enumerate(headers) if header.startswith(prefix)]


###############################################################################
# Function: reshapedata
# (Adicionado para manter compatibilidade com vaila/__init__.py)
###############################################################################
def reshapedata(df, selected_markers):
    """
    Dado um DataFrame `df` que contém a coluna de tempo e as colunas dos marcadores no formato:
      marker_x, marker_y, marker_z,
    e uma lista com os nomes dos marcadores selecionados, retorna um array NumPy de forma:
      (num_frames, num_markers, 3)

    Se a média dos valores absolutos for alta (> 100), os dados são convertidos de milímetros para metros.

    Args:
        df (DataFrame): DataFrame contendo os dados, em que a primeira coluna é Time.
        selected_markers (list): Lista de nomes dos marcadores.

    Returns:
        numpy.ndarray: Array com forma (num_frames, num_markers, 3) contendo os dados dos marcadores.
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
            raise ValueError(
                f"Columns for marker '{marker}' not found in expected format."
            )
    if np.mean(np.abs(points)) > 100:
        points = points * 0.001  # Converte de milímetros para metros
    return points


###############################################################################
# Function: detect_delimiter
# (Adicionado para manter compatibilidade com vaila/__init__.py)
###############################################################################
def detect_delimiter(file_path):
    """
    Detects the delimiter used in the file by trying common delimiters.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Detected delimiter (',', ';', '\t', or ' ')
    """
    delimiters = [',', ';', '\t', ' ']
    max_columns = 0
    best_delimiter = ','
    
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
        
        for delimiter in delimiters:
            columns = len(first_line.split(delimiter))
            if columns > max_columns:
                max_columns = columns
                best_delimiter = delimiter
    
    return best_delimiter


###############################################################################
# Function: detect_has_header
# (Adicionado para manter compatibilidade com vaila/__init__.py)
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
    with open(file_path, 'r', encoding='utf-8') as file:
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
# (Adicionado para manter compatibilidade com vaila/__init__.py)
###############################################################################
def select_file():
    """Exibe a caixa de diálogo para seleção do arquivo CSV ou TXT."""
    return filedialog.askopenfilename(
        title="Selecione o arquivo CSV ou TXT",
        filetypes=[("Data files", "*.csv;*.txt"), ("CSV files", "*.csv"), ("Text files", "*.txt")]
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
    choice = [None]

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
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(
        side="left", padx=5
    )

    Button(root, text="Select", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    # Return the marker names corresponding to the selected indices
    return [marker_labels[int(i)] for i in selected_indices]


###############################################################################
# Function: select_headers_gui
# (Adicionado para manter compatibilidade com vaila/__init__.py)
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
    Button(btn_frame, text="Unselect All", command=unselect_all).pack(
        side="left", padx=5
    )

    Button(root, text="OK", command=root.quit).pack(pady=10)
    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    return [headers[int(i)] for i in selected_indices]


###############################################################################
# Function: get_csv_headers
# (Adicionado para manter compatibilidade com vaila/__init__.py)
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
    try:
        import open3d as o3d
    except ImportError:
        print("open3d is not installed. Install it with 'pip install open3d'.")
        return

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
        from matplotlib.widgets import Slider, Button as MplButton
    except ImportError:
        print(
            "matplotlib is not installed. Please install it with 'pip install matplotlib'."
        )
        return

    num_frames, num_markers, _ = points.shape
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        points[0, :, 0], points[0, :, 1], points[0, :, 2], c="blue", s=20
    )
    ax.set_title(
        f"CSV Data Visualization (Matplotlib) | Frames: {num_frames} | FPS: {fps}"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    x_min, x_max = points[:, :, 0].min(), points[:, :, 0].max()
    y_min, y_max = points[:, :, 1].min(), points[:, :, 1].max()
    z_min, z_max = points[:, :, 2].min(), points[:, :, 2].max()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
    slider = Slider(ax_slider, "Frame", 0, num_frames - 1, valinit=0, valfmt="%d")

    current_frame = [0]

    def update_frame(val):
        # Update the scatter plot with the new points of the selected frame.
        frame = int(slider.val) if isinstance(val, float) else int(val)
        current_frame[0] = frame
        new_positions = points[frame]
        scatter._offsets3d = (
            new_positions[:, 0],
            new_positions[:, 1],
            new_positions[:, 2],
        )
        fig.canvas.draw_idle()

    slider.on_changed(update_frame)

    # Variables for automatic playback control
    playing = [False]
    timer = [None]

    def timer_callback():
        current_frame[0] = (current_frame[0] + 1) % num_frames
        slider.set_val(current_frame[0])
        update_frame(current_frame[0])

    def play_pause(event):
        if not playing[0]:
            playing[0] = True
            btn_play.label.set_text("Pause")
            timer[0] = fig.canvas.new_timer(interval=1000 / 30)  # Assuming 30 fps
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
    ax_play = fig.add_axes([0.82, 0.02, 0.1, 0.05])
    btn_play = MplButton(ax_play, "Play")
    btn_play.on_clicked(play_pause)

    # Add record button
    ax_record = fig.add_axes([0.82, 0.08, 0.1, 0.05])
    btn_record = MplButton(ax_record, "Record")
    
    def record_animation(event):
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")],
                title="Save animation as"
            )
            if not file_path:
                return

            # Create animation writer
            writer = animation.FFMpegWriter(
                fps=30,
                metadata=dict(artist='VAILA'),
                bitrate=1800
            )

            # Show recording message
            btn_record.label.set_text("Recording...")
            fig.canvas.draw_idle()

            # Create animation
            def update(frame):
                new_positions = points[frame]
                scatter._offsets3d = (
                    new_positions[:, 0],
                    new_positions[:, 1],
                    new_positions[:, 2],
                )
                return scatter,

            anim = animation.FuncAnimation(
                fig, update, frames=num_frames,
                interval=1000/30, blit=True
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
        if event.key == ' ':
            play_pause(None)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


###############################################################################
# Function: detect_units
# (Adicionado para manter compatibilidade com vaila/__init__.py)
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


###############################################################################
# Function: read_csv_generic
# (Added to maintain compatibility with vaila/__init__.py)
###############################################################################
def read_csv_generic(file_path):
    """
    Reads a CSV or TXT file considering:
      - Automatically detects the delimiter (',', ';', '\t', ' ')
      - Detects if the file has a header
      - If it doesn't have a header, uses default names (p1_x, p1_y, p1_z, p2_x, ...)
      - The first column contains the time or frames
      - The subsequent columns are organized in groups of three (x, y, z)
      - Automatically detects and converts units from millimeters to meters if necessary

    Returns:
      time_vector: pd.Series with the time/frames data
      marker_data: dictionary mapping the marker name to a numpy array Nx3
      valid_markers: dictionary mapping the marker name to the list of columns used
    """
    # Detect delimiter
    delimiter = detect_delimiter(file_path)
    
    # Detect if file has header
    has_header = detect_has_header(file_path, delimiter)
    
    # Read the file
    if has_header:
        df = pd.read_csv(file_path, delimiter=delimiter)
    else:
        # Create default column names for files without headers
        num_columns = len(pd.read_csv(file_path, delimiter=delimiter, nrows=0).columns)
        default_columns = ['Time']
        for i in range(1, (num_columns - 1) // 3 + 1):
            default_columns.extend([f'p{i}_x', f'p{i}_y', f'p{i}_z'])
        df = pd.read_csv(file_path, delimiter=delimiter, names=default_columns)
    
    if df.empty:
        raise ValueError("The file is empty or could not be read.")

    # The first column is the time/frames
    time_vector = df.iloc[:, 0]

    # Process the remaining columns: each column must have the format marker_coord (ex.: PELO_X)
    marker_headers = {}
    for col in df.columns[1:]:
        if "_" in col:
            parts = col.rsplit("_", 1)  # split by the last occurrence of '_'
            if len(parts) == 2 and parts[1].upper() in ["X", "Y", "Z"]:
                marker_name = parts[0]
                if marker_name not in marker_headers:
                    marker_headers[marker_name] = []
                marker_headers[marker_name].append(col)

    # Select only the markers that have the complete set of 3 columns
    valid_markers = {}
    for marker, cols in marker_headers.items():
        if len(cols) == 3:
            # Sort the columns to ensure the order: X, Y, Z
            sorted_cols = sorted(cols, key=lambda c: c.upper().split("_")[-1])
            valid_markers[marker] = sorted_cols
        else:
            print(f"Warning: The marker '{marker}' has incomplete data: {cols}")

    # Extract the data for each marker into an Nx3 array
    marker_data = {}
    for marker, cols in valid_markers.items():
        marker_data[marker] = df[cols].to_numpy()

    # Check if data needs unit conversion
    if valid_markers:
        # Create a temporary array with all points to check units
        temp_points = np.stack([marker_data[marker] for marker in valid_markers.keys()], axis=1)
        if detect_units(temp_points):
            print("Converting units from millimeters to meters...")
            # Convert all marker data from millimeters to meters
            for marker in marker_data:
                marker_data[marker] = marker_data[marker] * 0.001

    return time_vector, marker_data, valid_markers


###############################################################################
# Function: show_csv (Main Function)
# - Opens a file selection dialog to pick the CSV file.
# - Extracts the marker names (ignoring the first "Time" column).
# - Opens a marker selection dialog.
# - Constructs an array of marker positions of shape (num_frames, num_markers, 3).
# - Prompts the user to choose a visualization method.
# - Launches the visualization using either Open3D or Matplotlib.
###############################################################################
def show_csv():
    """
    Main function to load the CSV, select the markers and plot the data.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = tk.Tk()
    root.withdraw()
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    try:
        time_vector, marker_data, valid_markers = read_csv_generic(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Error reading the CSV file: {e}")
        return

    # List the available markers
    available_markers = list(valid_markers.keys())
    print("Available markers:")
    for marker in available_markers:
        print(marker)

    # Allow the user to select the markers to be visualized (multiple selection)
    selected_markers = select_markers_csv(available_markers)
    if not selected_markers:
        messagebox.showwarning("Warning", "No markers selected.")
        return

    # Build an array of points with shape (num_frames, num_markers, 3)
    # for the selected markers using the data in marker_data.
    points = np.stack([marker_data[marker] for marker in selected_markers], axis=1)
    num_frames = points.shape[0]
    num_markers = points.shape[1]

    file_name = os.path.basename(file_path)

    # Create the 3D figure with the initial frame (frame 0) markers
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.0, 0.15, 1.0, 0.85], projection="3d")  # Increased bottom margin
    scat = ax.scatter(points[0, :, 0], points[0, :, 1], points[0, :, 2], c="blue", s=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"C3D CSV Viewer | File: {file_name} | Markers: {len(selected_markers)}/{len(available_markers)} | Frames: {num_frames}"
    )

    # Calculate initial limits from data
    x_min, x_max = points[:, :, 0].min(), points[:, :, 0].max()
    y_min, y_max = points[:, :, 1].min(), points[:, :, 1].max()
    z_min, z_max = points[:, :, 2].min(), points[:, :, 2].max()

    # Add some padding to the limits
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    padding = 0.1  # 10% padding

    x_min -= x_range * padding
    x_max += x_range * padding
    y_min -= y_range * padding
    y_max += y_range * padding
    z_min -= z_range * padding
    z_max += z_range * padding

    # Set initial limits
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # Define the equal aspect to avoid distortions
    ax.set_aspect("equal")

    # Add text boxes for axis limits
    def update_x_limits(text):
        try:
            x_min_new, x_max_new = map(float, text.split(','))
            if x_min_new < x_max_new:
                ax.set_xlim([x_min_new, x_max_new])
                fig.canvas.draw_idle()
        except ValueError:
            pass

    def update_y_limits(text):
        try:
            y_min_new, y_max_new = map(float, text.split(','))
            if y_min_new < y_max_new:
                ax.set_ylim([y_min_new, y_max_new])
                fig.canvas.draw_idle()
        except ValueError:
            pass

    def update_z_limits(text):
        try:
            z_min_new, z_max_new = map(float, text.split(','))
            if z_min_new < z_max_new:
                ax.set_zlim([z_min_new, z_max_new])
                fig.canvas.draw_idle()
        except ValueError:
            pass

    def reset_limits(event):
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        textbox_x.set_val(f"{x_min:.2f},{x_max:.2f}")
        textbox_y.set_val(f"{y_min:.2f},{y_max:.2f}")
        textbox_z.set_val(f"{z_min:.2f},{z_max:.2f}")
        fig.canvas.draw_idle()

    # Create text boxes for limits with better positioning
    ax_textbox_x = fig.add_axes([0.02, 0.08, 0.12, 0.03])
    ax_textbox_y = fig.add_axes([0.02, 0.05, 0.12, 0.03])
    ax_textbox_z = fig.add_axes([0.02, 0.02, 0.12, 0.03])
    ax_reset = fig.add_axes([0.15, 0.02, 0.06, 0.09])

    textbox_x = TextBox(ax_textbox_x, 'X:', initial=f"{x_min:.2f},{x_max:.2f}")
    textbox_y = TextBox(ax_textbox_y, 'Y:', initial=f"{y_min:.2f},{y_max:.2f}")
    textbox_z = TextBox(ax_textbox_z, 'Z:', initial=f"{z_min:.2f},{z_max:.2f}")
    btn_reset = MplButton(ax_reset, 'Reset\nLimits')

    textbox_x.on_submit(update_x_limits)
    textbox_y.on_submit(update_y_limits)
    textbox_z.on_submit(update_z_limits)
    btn_reset.on_clicked(reset_limits)

    # Create a slider for frame control, positioned at the bottom
    ax_frame = fig.add_axes([0.25, 0.02, 0.5, 0.04])
    slider_frame = Slider(ax_frame, "Frame", 0, num_frames - 1, valinit=0, valfmt="%d")

    current_frame = [0]

    def update_frame(val):
        # Update the scatter plot with the new points of the selected frame.
        frame = int(slider_frame.val) if isinstance(val, float) else int(val)
        current_frame[0] = frame
        new_positions = points[frame]
        scat._offsets3d = (
            new_positions[:, 0],
            new_positions[:, 1],
            new_positions[:, 2],
        )
        fig.canvas.draw_idle()

    slider_frame.on_changed(update_frame)

    # Variables for automatic playback control
    playing = [False]
    timer = [None]

    def timer_callback():
        current_frame[0] = (current_frame[0] + 1) % num_frames
        slider_frame.set_val(current_frame[0])
        update_frame(current_frame[0])

    def play_pause(event):
        if not playing[0]:
            playing[0] = True
            btn_play.label.set_text("Pause")
            timer[0] = fig.canvas.new_timer(interval=1000 / 30)  # Assuming 30 fps
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
    ax_play = fig.add_axes([0.82, 0.02, 0.1, 0.05])
    btn_play = MplButton(ax_play, "Play")
    btn_play.on_clicked(play_pause)

    # Add record button
    ax_record = fig.add_axes([0.82, 0.08, 0.1, 0.05])
    btn_record = MplButton(ax_record, "Record")
    
    def record_animation(event):
        try:
            # Ask for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mp4",
                filetypes=[("MP4 files", "*.mp4")],
                title="Save animation as"
            )
            if not file_path:
                return

            # Create animation writer
            writer = animation.FFMpegWriter(
                fps=30,
                metadata=dict(artist='VAILA'),
                bitrate=1800
            )

            # Show recording message
            btn_record.label.set_text("Recording...")
            fig.canvas.draw_idle()

            # Create animation
            def update(frame):
                new_positions = points[frame]
                scat._offsets3d = (
                    new_positions[:, 0],
                    new_positions[:, 1],
                    new_positions[:, 2],
                )
                return scat,

            anim = animation.FuncAnimation(
                fig, update, frames=num_frames,
                interval=1000/30, blit=True
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
        if event.key == ' ':
            play_pause(None)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    show_csv()
