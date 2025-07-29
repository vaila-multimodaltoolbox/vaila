"""
Project: vailá Multimodal Toolbox
Script: viewc3d.py - View C3D File

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 6 February 2025
Update Date: 28 July 2025
Version: 0.0.4

Description:
------------
This script launches a 3D viewer for C3D files, providing efficient visualization of marker data.
Marker positions are converted from millimeters to meters and displayed using Open3D.
The viewer features:
  - Display of markers (optionally selected by the user), Cartesian coordinate axes, a dark gray ground with a 1x1 grid, and boundary markers.
  - Detailed window title displaying file name, number of markers (selected vs. total), number of frames, FPS, and keyboard/mouse shortcuts.
  - Keyboard shortcuts for navigation:
       'N': Next frame
       'P': Previous frame
       'F': Advance 10 frames
       'B': Go back 10 frames
       'Space': Toggle automatic playback
       'O': Display current camera parameters
  - Mouse actions:
       Left Drag: Rotate
       Middle or Right Drag: Pan
       Mouse Wheel: Zoom

Usage:
------
1. Ensure the required dependencies are installed:
   - open3d (pip install open3d)
   - ezc3d (pip install ezc3d)
   - numpy
   - tkinter (typically included with Python)
2. Run the script.
3. When prompted, choose a C3D file via the file selection dialog.
4. In the marker selection window, choose the markers you want to display and press "Select".
5. Use the specified keyboard and mouse commands to navigate through the frames in the 3D viewer.

License:
--------
This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of
the License, or (at your option) any later version. This program is distributed in the hope that
it will be useful, but WITHOUT ANY WARRANTY.
"""

import os
import open3d as o3d
import ezc3d
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import time
from rich import print
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button


def _create_centered_tk_root():
    """Creates a temporary, centered, hidden tkinter root window to force dialogs to appear in the center on macOS."""
    root = tk.Tk()
    root.withdraw()
    # Move the dummy window to the center of the screen to hint to the OS where to place the dialog
    root.update_idletasks()
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    root.geometry(f'+{ws//2}+{hs//2}')
    root.attributes("-topmost", True)
    return root


def detect_c3d_units(pts):
    """
    Detect if C3D data is in millimeters or meters based on multiple criteria.
    
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
    
    # Method 1: Check absolute magnitude of values
    mean_abs_value = np.mean(np.abs(valid_data))
    if mean_abs_value > 100:
        return True, f"magnitude_check (mean_abs: {mean_abs_value:.1f})"
    
    # Method 2: Check range of values
    data_range = np.max(valid_data) - np.min(valid_data)
    if data_range > 1000:
        return True, f"range_check (range: {data_range:.1f})"
    
    # Method 3: Inter-marker distances (human body scale)
    if pts.shape[0] > 0 and pts.shape[1] > 1:
        first_frame = pts[0]
        valid_markers = first_frame[~np.isnan(first_frame).any(axis=1)]
        
        if len(valid_markers) > 1:
            # Calculate pairwise distances between markers
            distances = []
            for i in range(len(valid_markers)):
                for j in range(i+1, len(valid_markers)):
                    dist = np.linalg.norm(valid_markers[i] - valid_markers[j])
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                max_distance = np.max(distances)
                
                # Human body markers typically span 0.5-2.5 meters
                # If average distance > 50 or max > 3000, likely in mm
                if avg_distance > 50 or max_distance > 3000:
                    return True, f"marker_distance_check (avg: {avg_distance:.1f}, max: {max_distance:.1f})"
    
    # Method 4: Check for typical human measurement ranges
    if len(valid_data) > 100:
        # For human motion capture in meters:
        # - Most coordinates should be between -5 and +5 meters
        # - Very few should exceed 10 meters
        extreme_values = np.sum(np.abs(valid_data) > 10)
        extreme_percentage = (extreme_values / len(valid_data)) * 100
        
        if extreme_percentage > 10:  # More than 10% of values are "extreme"
            return True, f"human_scale_check (extreme_values: {extreme_percentage:.1f}%)"
    
    return False, "passed_all_checks"


def ask_user_units_c3d():
    """
    Ask user about the units of the coordinate data if auto-detection is uncertain.
    
    Returns:
        str: 'mm' for millimeters, 'm' for meters, 'auto' for automatic detection
    """
    root = tk.Tk()
    root.title("C3D Units Selection")
    root.geometry("400x300")
    root.attributes("-topmost", True)
    
    result = {'choice': 'auto'}
    
    # Create main frame
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = tk.Label(main_frame, text="C3D Data Units", font=("Arial", 14, "bold"))
    title_label.pack(pady=(0, 10))
    
    # Explanation
    explanation = tk.Label(main_frame, 
        text="Please select the units of your C3D coordinate data:",
        font=("Arial", 10), justify=tk.LEFT)
    explanation.pack(pady=(0, 15))
    
    # Radio buttons
    choice_var = tk.StringVar(value="auto")
    
    auto_radio = tk.Radiobutton(main_frame, text="Auto-detect (recommended)", 
                               variable=choice_var, value="auto", font=("Arial", 10))
    auto_radio.pack(anchor=tk.W, pady=2)
    
    mm_radio = tk.Radiobutton(main_frame, text="Millimeters (mm) - typical values like 1234.56", 
                             variable=choice_var, value="mm", font=("Arial", 10))
    mm_radio.pack(anchor=tk.W, pady=2)
    
    m_radio = tk.Radiobutton(main_frame, text="Meters (m) - typical values like 1.23", 
                            variable=choice_var, value="m", font=("Arial", 10))
    m_radio.pack(anchor=tk.W, pady=2)
    
    # Buttons
    button_frame = tk.Frame(main_frame)
    button_frame.pack(pady=(20, 0))
    
    def on_ok():
        result['choice'] = choice_var.get()
        root.quit()
        root.destroy()
    
    def on_cancel():
        result['choice'] = 'auto'
        root.quit()
        root.destroy()
    
    ok_button = tk.Button(button_frame, text="OK", command=on_ok, 
                         bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), width=10)
    ok_button.pack(side=tk.LEFT, padx=(0, 10))
    
    cancel_button = tk.Button(button_frame, text="Cancel", command=on_cancel, 
                             bg="#f44336", fg="white", font=("Arial", 10, "bold"), width=10)
    cancel_button.pack(side=tk.LEFT)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()
    
    return result['choice']


def load_c3d_file():
    """
    Opens a dialog to select a C3D file and loads the marker data with automatic unit detection.
    General and robust handling of C3D files in both mm and meters.

    Returns:
        pts: np.ndarray with shape (num_frames, num_markers, 3) – the points in meters
        filepath: path of the selected file.
        fps: frames per second (Hz).
        marker_labels: list of marker labels.
    """
    root = _create_centered_tk_root()
    filepath = filedialog.askopenfilename(
        title="Select a C3D file", filetypes=[("C3D Files", "*.c3d")]
    )
    root.destroy()
    if not filepath:
        print("No file was selected. Exiting.")
        exit(0)

    # Display loading message
    print("[bold blue]Loading C3D file...[/bold blue]")
    
    try:
        c3d = ezc3d.c3d(filepath)
        fps = c3d["header"]["points"]["frame_rate"]
        pts = c3d["data"]["points"]
        pts = pts[:3, :, :]  # use only x, y, z coordinates
        pts = np.transpose(pts, (2, 1, 0))  # shape (num_frames, num_markers, 3)

        # Extract marker labels
        marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
        if isinstance(marker_labels[0], list):
            marker_labels = marker_labels[0]

        print(f"Raw data loaded: {len(marker_labels)} markers, {pts.shape[0]} frames")
        
        # Auto-detect units and convert if necessary
        is_mm, detection_method = detect_c3d_units(pts)
        
        # Show preview data for user reference
        valid_sample = pts[~np.isnan(pts)][:10]  # First 10 valid values
        if len(valid_sample) > 0:
            print(f"Data preview (first 10 valid values): {valid_sample}")
        
        # Check if detection is uncertain and ask user if needed
        uncertain_detection = False
        if "passed_all_checks" in detection_method:
            # Data passed all checks - could be meters but want to double-check
            mean_val = np.mean(np.abs(valid_sample)) if len(valid_sample) > 0 else 0
            if 0.1 < mean_val < 10:  # Ambiguous range
                uncertain_detection = True
        
        # Ask user for confirmation if detection is uncertain
        if uncertain_detection:
            print(f"[yellow]⚠ Uncertain unit detection (method: {detection_method})[/yellow]")
            user_choice = ask_user_units_c3d()
            
            if user_choice == 'mm':
                is_mm = True
                detection_method = "user_override_mm"
            elif user_choice == 'm':
                is_mm = False
                detection_method = "user_override_m"
            # If 'auto', keep the original detection
        
        # Apply conversion
        if is_mm:
            pts = pts * 0.001  # Convert from millimeters to meters
            print("[bold green]✓ Applied conversion: MILLIMETERS → METERS[/bold green]")
            print(f"  Method: {detection_method}")
        else:
            print("[bold green]✓ No conversion applied: Data already in METERS[/bold green]")
            print(f"  Method: {detection_method}")
        
        # Show data statistics after conversion
        valid_data = pts[~np.isnan(pts)]
        if len(valid_data) > 0:
            print(f"  Final data range: [{np.min(valid_data):.3f}, {np.max(valid_data):.3f}] meters")
            print(f"  Mean absolute value: {np.mean(np.abs(valid_data)):.3f} meters")
        
        print("[bold green]Successfully loaded and processed C3D file![/bold green]")
        return pts, filepath, fps, marker_labels
        
    except Exception as e:
        messagebox.showerror("Error Loading File", f"Failed to load C3D file:\n{str(e)}")
        print(f"[bold red]Error details:[/bold red] {str(e)}")
        exit(1)


def toggle_theme(theme, window=None):
    """
    Toggle between light and dark themes by changing UI colors.

    Args:
        theme (str): The theme to set ('light' or 'dark').
        window: The tkinter window to apply the theme to.
    """
    if window is None:
        return
        
    style = ttk.Style()
    style.theme_use('clam')
    
    if theme == 'dark':
        # Dark theme
        window.configure(bg='black')
        style.configure('.', background='black', foreground='white')
        style.map('TButton', background=[('active', 'gray')])
        # Configure all widgets for dark theme
        for widget in window.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg='black')
            elif isinstance(widget, tk.Label):
                widget.configure(bg='black', fg='white')
            elif isinstance(widget, tk.Entry):
                widget.configure(bg='gray', fg='white')
            elif isinstance(widget, tk.Listbox):
                widget.configure(bg='gray', fg='white', selectbackground='blue')
            elif isinstance(widget, tk.Button):
                widget.configure(bg='gray', fg='white')
    else:
        # Light theme
        window.configure(bg='white')
        style.configure('.', background='white', foreground='black')
        style.map('TButton', background=[('active', 'lightgray')])
        # Configure all widgets for light theme
        for widget in window.winfo_children():
            if isinstance(widget, tk.Frame):
                widget.configure(bg='white')
            elif isinstance(widget, tk.Label):
                widget.configure(bg='white', fg='black')
            elif isinstance(widget, tk.Entry):
                widget.configure(bg='white', fg='black')
            elif isinstance(widget, tk.Listbox):
                widget.configure(bg='white', fg='black', selectbackground='lightblue')
            elif isinstance(widget, tk.Button):
                widget.configure(bg='lightgray', fg='black')


def select_markers(marker_labels, c3d_filepath=None):
    """
    Displays a Tkinter window with a list of marker labels so the user can select which markers to display.

    Args:
        marker_labels (list): list of marker labels.
        c3d_filepath (str): path to the C3D file being loaded
    Returns:
        List of selected marker indices.
    """
    root = tk.Tk()
    root.title("Select Markers to Display")
    root.geometry("600x500")
    
    # Determinar o diretório onde salvar o arquivo de seleção
    if c3d_filepath:
        c3d_dir = os.path.dirname(c3d_filepath)
        c3d_basename = os.path.splitext(os.path.basename(c3d_filepath))[0]
        selection_file = os.path.join(c3d_dir, f"{c3d_basename}_marker_selection.json")
    else:
        selection_file = "marker_selection.json"  # Fallback se não tiver filepath
    
    # Remove theme toggle - it doesn't work in the viewer window
    search_frame = tk.Frame(root)
    search_frame.pack(pady=5)
    tk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var, width=30)
    search_entry.pack(side=tk.LEFT, padx=5)
    
    # Create frame for listbox and scrollbar
    list_frame = tk.Frame(root)
    list_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    listbox = tk.Listbox(list_frame, selectmode="multiple", width=50, height=15, 
                         yscrollcommand=scrollbar.set)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)
    
    # Function to populate listbox based on search
    def update_listbox():
        search_term = search_var.get().lower()
        listbox.delete(0, tk.END)
        for i, label in enumerate(marker_labels):
            if search_term in label.lower() or not search_term:
                listbox.insert(tk.END, f"{i}: {label}")
    
    # Initial population
    update_listbox()
    search_var.trace('w', lambda *args: update_listbox())

    # Create a frame for extra control buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)
        
    def save_selection():
        selected = [int(listbox.get(i).split(':')[0]) for i in listbox.curselection()]
        if selected:
            try:
                with open(selection_file, 'w') as f:
                    json.dump(selected, f)
                messagebox.showinfo("Success", f"Selection saved to:\n{selection_file}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save selection:\n{str(e)}")
    
    def load_selection():
        try:
            with open(selection_file, 'r') as f:
                saved_indices = json.load(f)
            listbox.selection_clear(0, tk.END)
            for idx in saved_indices:
                for i in range(listbox.size()):
                    if int(listbox.get(i).split(':')[0]) == idx:
                        listbox.selection_set(i)
                        break
            messagebox.showinfo("Success", f"Selection loaded from:\n{selection_file}")
        except FileNotFoundError:
            messagebox.showwarning("Warning", f"No saved selection found at:\n{selection_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load selection:\n{str(e)}")

    # Add the buttons
    btn_select_all = tk.Button(button_frame, text="Select All", command=select_all)
    btn_unselect_all = tk.Button(button_frame, text="Unselect All", command=unselect_all)
    btn_save = tk.Button(button_frame, text="Save Selection", command=save_selection)
    btn_load = tk.Button(button_frame, text="Load Selection", command=load_selection)
    
    btn_select_all.pack(side=tk.LEFT, padx=5)
    btn_unselect_all.pack(side=tk.LEFT, padx=5)
    btn_save.pack(side=tk.LEFT, padx=5)
    btn_load.pack(side=tk.LEFT, padx=5)

    # Add quick filter buttons
    filter_frame = tk.Frame(root)
    filter_frame.pack(pady=5)
    
    def filter_by_prefix(prefix):
        listbox.selection_clear(0, tk.END)
        for i in range(listbox.size()):
            if listbox.get(i).lower().startswith(prefix.lower()):
                listbox.selection_set(i)
    
    tk.Button(filter_frame, text="Left", command=lambda: filter_by_prefix("left")).pack(side=tk.LEFT, padx=2)
    tk.Button(filter_frame, text="Right", command=lambda: filter_by_prefix("right")).pack(side=tk.LEFT, padx=2)
    tk.Button(filter_frame, text="Head", command=lambda: filter_by_prefix("head")).pack(side=tk.LEFT, padx=2)
    tk.Button(filter_frame, text="Spine", command=lambda: filter_by_prefix("spine")).pack(side=tk.LEFT, padx=2)

    def on_select():
        root.quit()

    btn_select = tk.Button(root, text="Select", command=on_select, bg="#4CAF50", 
                          fg="white", font=("Arial", 12, "bold"))
    btn_select.pack(pady=(0, 10))

    # Center the window to prevent it from getting stuck in a corner on macOS
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry(f'{w}x{h}+{int(x)}+{int(y)}')

    root.mainloop()
    selected_indices = [int(listbox.get(i).split(':')[0]) for i in listbox.curselection()]
    root.destroy()
    return selected_indices


def create_coordinate_lines(axis_length=0.25):
    """
    Creates lines representing the Cartesian coordinate axes:
      - X axis in red
      - Y axis in green
      - Z axis in blue
    """
    points = np.array(
        [[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]
    )
    lines = np.array([[0, 1], [0, 2], [0, 3]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def create_ground_plane(width=5.0, height=5.0):
    """
    Creates a plane on the XY axis (ground) with dimensions width x height and dark gray color.
    The plane is defined at z = 0.
    """
    half_w = width / 2.0
    half_h = height / 2.0
    vertices = [
        [-half_w, -half_h, 0],
        [half_w, -half_h, 0],
        [half_w, half_h, 0],
        [-half_w, half_h, 0],
    ]
    triangles = [[0, 1, 2], [0, 2, 3]]
    ground = o3d.geometry.TriangleMesh()
    ground.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    ground.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    ground.paint_uniform_color([0.2, 0.2, 0.2])  # dark gray ground
    ground.compute_vertex_normals()
    return ground


def create_x_marker(position, size=0.2):
    """
    Creates an "X" marker on the XY plane to indicate boundaries.

    Args:
        position (np.ndarray): (x, y, z) coordinate.
        size (float): Size of the lines forming the "X".

    Returns:
        A LineSet representing the "X".
    """
    half = size / 2.0
    x, y, z = position
    points = np.array(
        [
            [x - half, y - half, z],
            [x + half, y + half, z],
            [x - half, y + half, z],
            [x + half, y - half, z],
        ]
    )
    lines = np.array([[0, 1], [2, 3]])
    x_marker = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    x_marker.paint_uniform_color([1, 0, 0])
    return x_marker


def get_marker_customization():
    """
    Opens a GUI dialog for marker customization (radius and color).
    
    Returns:
        tuple: (marker_radius, marker_color) where marker_color is a list of [R, G, B] values
    """
    root = tk.Tk()
    root.title("Marker Customization")
    root.geometry("500x600")  # Increase window size
    
    # Variables to store values
    radius_var = tk.DoubleVar(value=0.015)
    
    # Radius input
    tk.Label(root, text="Marker Radius:", font=("Arial", 12, "bold")).pack(pady=10)
    radius_frame = tk.Frame(root)
    radius_frame.pack(pady=10)
    radius_scale = tk.Scale(radius_frame, from_=0.005, to=0.05, resolution=0.001, 
                           orient=tk.HORIZONTAL, variable=radius_var, length=300)
    radius_scale.pack(side=tk.LEFT)
    radius_label = tk.Label(radius_frame, textvariable=radius_var, width=8, font=("Arial", 10))
    radius_label.pack(side=tk.LEFT, padx=10)
    
    # Color selection with predefined colors
    tk.Label(root, text="Marker Color:", font=("Arial", 12, "bold")).pack(pady=15)
    
    # Color preview
    preview_frame = tk.Frame(root, width=150, height=60, relief=tk.SOLID, borderwidth=3)
    preview_frame.pack(pady=10)
    
    # Selected color variable
    selected_color = tk.StringVar(value="Orange")
    
    def update_preview(color_name):
        selected_color.set(color_name)
        colors = {
            "Orange": [1.0, 0.65, 0.0],
            "Blue": [0.0, 0.5, 1.0],
            "Green": [0.0, 1.0, 0.0],
            "Red": [1.0, 0.0, 0.0],
            "White": [1.0, 1.0, 1.0],
            "Yellow": [1.0, 1.0, 0.0],
            "Purple": [0.5, 0.0, 1.0],
            "Cyan": [0.0, 1.0, 1.0],
            "Pink": [1.0, 0.0, 1.0],
            "Gray": [0.5, 0.5, 0.5],
            "Black": [0.0, 0.0, 0.0]
        }
        r, g, b = colors[color_name]
        color_hex = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
        preview_frame.configure(bg=color_hex)
    
    # Color buttons in a grid
    color_frame = tk.Frame(root)
    color_frame.pack(pady=15)
    
    colors = [
        ("Orange", "Orange"), ("Blue", "Blue"), ("Green", "Green"), ("Red", "Red"),
        ("White", "White"), ("Yellow", "Yellow"), ("Purple", "Purple"), ("Cyan", "Cyan"),
        ("Pink", "Pink"), ("Gray", "Gray"), ("Black", "Black")
    ]
    
    for i, (name, color) in enumerate(colors):
        row = i // 4
        col = i % 4
        btn = tk.Button(color_frame, text=name, width=10, height=2,
                       command=lambda c=color: update_preview(c))
        btn.grid(row=row, column=col, padx=3, pady=3)
    
    # Initial preview update
    update_preview("Orange")
    
    # OK button - mover para o final e garantir que apareça
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    result = {}
    
    def on_ok():
        colors = {
            "Orange": [1.0, 0.65, 0.0],
            "Blue": [0.0, 0.5, 1.0],
            "Green": [0.0, 1.0, 0.0],
            "Red": [1.0, 0.0, 0.0],
            "White": [1.0, 1.0, 1.0],
            "Yellow": [1.0, 1.0, 0.0],
            "Purple": [0.5, 0.0, 1.0],
            "Cyan": [0.0, 1.0, 1.0],
            "Pink": [1.0, 0.0, 1.0],
            "Gray": [0.5, 0.5, 0.5],
            "Black": [0.0, 0.0, 0.0]
        }
        result['radius'] = radius_var.get()
        result['color'] = colors[selected_color.get()]
        root.quit()
    
    ok_btn = tk.Button(button_frame, text="OK", command=on_ok, bg="#4CAF50", fg="white", 
                      font=("Arial", 14, "bold"), width=15, height=2)
    ok_btn.pack()
    
    # Center the window to prevent it from getting stuck in a corner on macOS
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws/2) - (w/2)
    y = (hs/2) - (h/2)
    root.geometry(f'{w}x{h}+{int(x)}+{int(y)}')

    root.mainloop()
    root.destroy()
    
    return result.get('radius', 0.015), result.get('color', [1.0, 0.65, 0.0])


def create_ground_grid(x_min=-1, x_max=5, y_min=-1, y_max=6, spacing=1.0):
    """
    Creates a grid of lines with a spacing of 1 meter for the ground.
    """
    points = []
    lines = []
    for y in np.arange(y_min, y_max + spacing, spacing):
        idx = len(points)
        points.append([x_min, y, 0])
        points.append([x_max, y, 0])
        lines.append([idx, idx + 1])
    for x in np.arange(x_min, x_max + spacing, spacing):
        idx = len(points)
        points.append([x, y_min, 0])
        points.append([x, y_max, 0])
        lines.append([idx, idx + 1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(points)),
        lines=o3d.utility.Vector2iVector(np.array(lines)),
    )
    grid.paint_uniform_color([1.0, 1.0, 1.0])
    return grid


def check_display_environment():
    """
    Check if the system has a proper display environment for OpenGL.
    
    Returns:
        bool: True if display is available, False otherwise
        str: Description of the environment
    """
    import os
    
    # Check if we're in a remote session or headless environment
    if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
        return False, "SSH remote session detected"
    
    if 'DISPLAY' not in os.environ and os.name != 'nt':
        return False, "No DISPLAY environment variable (headless system)"
    
    # Check for common virtualization/container environments
    if any(env in os.environ for env in ['CONTAINER', 'DOCKER', 'KUBERNETES']):
        return False, "Container/virtualization environment detected"
    
    return True, "Display environment available"


def check_opengl_support():
    """
    Check if OpenGL/Open3D visualization is supported on this system.
    
    Returns:
        bool: True if supported, False otherwise
        str: Error message if not supported
    """
    # First check display environment
    display_ok, display_msg = check_display_environment()
    if not display_ok:
        return False, f"Display issue: {display_msg}"
    
    try:
        # Try to create a minimal Open3D window to test OpenGL support
        test_vis = o3d.visualization.Visualizer()
        success = test_vis.create_window(window_name="OpenGL Test", width=100, height=100, visible=False)
        
        if success:
            test_vis.destroy_window()
            return True, "OpenGL support confirmed"
        else:
            return False, "Failed to create Open3D window - OpenGL not supported"
            
    except Exception as e:
        error_msg = str(e).lower()
        if 'glx' in error_msg or 'glfw' in error_msg:
            return False, f"OpenGL/GLX error: {str(e)}"
        elif 'mesa' in error_msg or 'swrast' in error_msg:
            return False, f"Mesa/software rendering issue: {str(e)}"
        else:
            return False, f"OpenGL initialization error: {str(e)}"


def run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices):
    """
    Fallback visualization using matplotlib when OpenGL is not available.
    Provides interactive frame navigation and playback.
    """
    try:
        # Filter points to selected markers
        points = points[:, selected_indices, :]
        selected_markers = [marker_labels[i] for i in selected_indices]
        num_frames, num_markers, _ = points.shape
        file_name = os.path.basename(filepath)
        
        print("[yellow] Using matplotlib fallback visualization[/yellow]")
        print(f"[yellow] File: {file_name}[/yellow]")
        print(f"[yellow] Markers: {num_markers}/{len(marker_labels)}[/yellow]")
        print(f"[yellow] Selected markers: {', '.join(selected_markers)}[/yellow]")
        print(f"[yellow] Frames: {num_frames}[/yellow]")
        print(f"[yellow] FPS: {fps}[/yellow]")
        
        # Create figure with extra space for controls
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.15)
        
        # Calculate global limits for consistent scaling
        all_valid = points[~np.isnan(points)]
        if len(all_valid) > 0:
            x_coords = points[:, :, 0][~np.isnan(points[:, :, 0])]
            y_coords = points[:, :, 1][~np.isnan(points[:, :, 1])]
            z_coords = points[:, :, 2][~np.isnan(points[:, :, 2])]
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            
            # Add margin
            x_range = max(x_max - x_min, 0.1) * 0.1
            y_range = max(y_max - y_min, 0.1) * 0.1
            z_range = max(z_max - z_min, 0.1) * 0.1
            
            x_min -= x_range
            x_max += x_range
            y_min -= y_range
            y_max += y_range
            z_min -= z_range
            z_max += z_range
        else:
            x_min = y_min = z_min = -1
            x_max = y_max = z_max = 1
        
        # Initialize plot
        current_frame = [0]
        scatter_plot = None
        
        def update_plot(frame_idx):
            nonlocal scatter_plot
            ax.clear()
            
            # Get current frame data
            frame_data = points[frame_idx]
            valid_mask = ~np.isnan(frame_data).any(axis=1)
            valid_points = frame_data[valid_mask]
            
            if len(valid_points) > 0:
                scatter_plot = ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], 
                                        c='blue', s=60, alpha=0.8, edgecolors='navy', linewidth=0.5)
            
            # Set consistent limits and labels
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel('X (meters)', fontsize=10)
            ax.set_ylabel('Y (meters)', fontsize=10)
            ax.set_zlabel('Z (meters)', fontsize=10)
            
            # Update title with frame info
            time_sec = frame_idx / fps if fps > 0 else 0
            ax.set_title(f'C3D Viewer (Matplotlib Fallback)\n'
                        f'{file_name} | Frame {frame_idx+1}/{num_frames} | '
                        f'Time: {time_sec:.2f}s | Valid: {len(valid_points)}/{num_markers}',
                        fontsize=11, pad=20)
            
            fig.canvas.draw_idle()
        
        # Create slider for frame control
        ax_slider = plt.axes([0.15, 0.02, 0.5, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valfmt='%d')
        
        def on_slider_change(val):
            frame_idx = int(slider.val)
            current_frame[0] = frame_idx
            update_plot(frame_idx)
        
        slider.on_changed(on_slider_change)
        
        # Create play/pause button
        ax_button = plt.axes([0.7, 0.02, 0.08, 0.05])
        button = Button(ax_button, 'Play')
        
        # Animation control
        anim = [None]
        is_playing = [False]
        
        def toggle_play(event):
            if not is_playing[0]:
                # Start playing
                def animate(frame):
                    new_frame = (current_frame[0] + 1) % num_frames
                    current_frame[0] = new_frame
                    slider.set_val(new_frame)
                    return []
                
                anim[0] = animation.FuncAnimation(fig, animate, frames=num_frames, 
                                                interval=int(1000/fps), repeat=True, blit=False)
                button.label.set_text('Pause')
                is_playing[0] = True
            else:
                # Stop playing
                if anim[0]:
                    anim[0].event_source.stop()
                button.label.set_text('Play')
                is_playing[0] = False
        
        button.on_clicked(toggle_play)
        
        # Add keyboard shortcuts
        def on_key(event):
            if event.key == ' ':  # Space bar
                toggle_play(None)
            elif event.key == 'right' and current_frame[0] < num_frames - 1:
                current_frame[0] += 1
                slider.set_val(current_frame[0])
            elif event.key == 'left' and current_frame[0] > 0:
                current_frame[0] -= 1
                slider.set_val(current_frame[0])
            elif event.key == 'up' and current_frame[0] < num_frames - 10:
                current_frame[0] += 10
                slider.set_val(current_frame[0])
            elif event.key == 'down' and current_frame[0] >= 10:
                current_frame[0] -= 10
                slider.set_val(current_frame[0])
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Add instructions
        instructions = (" Controls: Space=Play/Pause | ←→=Frame | ↑↓=10 Frames | Mouse=Rotate View")
        plt.figtext(0.02, 0.95, instructions, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Plot initial frame
        update_plot(0)
        
        print("[green] Matplotlib visualization ready![/green]")
        print("[cyan] Use slider, play button, or keyboard for navigation[/cyan]")
        
        plt.show()
        
        print("[green] Matplotlib visualization completed successfully![/green]")
        return True
        
    except ImportError:
        print("[red] Error: matplotlib not available for fallback visualization[/red]")
        print("[red]   Install with: pip install matplotlib[/red]")
        return False
    except Exception as e:
        print(f"[red] Error in matplotlib fallback: {str(e)}[/red]")  # noqa: F841 - Error details for potential future use
        return False


def run_viewc3d():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Load data from the C3D file, including FPS and marker labels
    points, filepath, fps, marker_labels = load_c3d_file()
    num_frames, total_markers, _ = points.shape

    # Let user select which markers to display - passar o filepath
    selected_indices = select_markers(marker_labels, filepath)
    if not selected_indices:
        print("No markers selected, exiting.")
        exit(0)
    
    # Check OpenGL support before proceeding
    opengl_supported, error_msg = check_opengl_support()
    
    if not opengl_supported:
        print("[yellow] OpenGL/Open3D not supported on this system:[/yellow]")
        print(f"[yellow]  {error_msg}[/yellow]")
        print("[yellow]  This is common on older Linux systems or remote connections[/yellow]")
        print("[cyan] Switching to matplotlib fallback visualization...[/cyan]")
        
        success = run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices)
        if success:
            return
        else:
            print("[red]❌ Both Open3D and matplotlib visualization failed[/red]")
            print("[red]Please check your system's graphics drivers and Python environment[/red]")
            return
    
    # Filter the points array to only the selected markers
    points = points[:, selected_indices, :]
    num_markers = len(selected_indices)

    # Extract file name from full path (cross-platform)
    file_name = os.path.basename(filepath)

    # Build a detailed window title with file info, FPS, and control instructions
    window_title = (
        f"C3D Viewer | File: {file_name} | Markers: {num_markers}/{total_markers} | Frames: {num_frames} | FPS: {fps} | "
        "Keys: [←→: Frame, ↑↓: 60 Frames, +/-: Size, C: Color, Space: Play, H: Help] | "
        "Mouse: [Left: Rotate, Middle/Right: Pan, Wheel: Zoom]"
    )

    try:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        window_created = vis.create_window(window_name=window_title)
        
        if not window_created:
            print("[red] Failed to create Open3D window[/red]")
            print("[cyan] Trying matplotlib fallback...[/cyan]")
            success = run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices)
            return
            
    except Exception as e:
        print(f"[red] Open3D window creation failed: {str(e)}[/red]")
        print("[cyan] Trying matplotlib fallback...[/cyan]")
        success = run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices)
        return

    # Remove this line:
    # marker_radius, marker_color = get_marker_customization()
    
    # Set default values for radius and color
    current_radius = 0.015  # Default value
    current_color_index = 0  # Orange as default
    available_colors = [
        ([1.0, 0.65, 0.0], "Orange"),
        ([0.0, 0.5, 1.0], "Blue"), 
        ([0.0, 1.0, 0.0], "Green"),
        ([1.0, 0.0, 0.0], "Red"),
        ([1.0, 1.0, 1.0], "White"),
        ([1.0, 1.0, 0.0], "Yellow"),
        ([0.5, 0.0, 1.0], "Purple"),
        ([0.0, 1.0, 1.0], "Cyan"),
        ([1.0, 0.0, 1.0], "Pink"),
        ([0.5, 0.5, 0.5], "Gray"),
        ([0.0, 0.0, 0.0], "Black")
    ]
    
    # Use color from the default list
    default_color = available_colors[current_color_index][0]

    spheres = []
    spheres_bases = []
    for i in range(num_markers):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=current_radius, resolution=8
        )
        base_vertices = np.asarray(sphere.vertices).copy()
        initial_pos = points[0][i]
        sphere.vertices = o3d.utility.Vector3dVector(base_vertices + initial_pos)
        sphere.paint_uniform_color(default_color)
        spheres.append(sphere)
        spheres_bases.append(base_vertices)

    for sphere in spheres:
        vis.add_geometry(sphere)

    # Add Cartesian axes, ground, and grid
    axes = create_coordinate_lines(axis_length=0.25)
    vis.add_geometry(axes)

    # Modificar o ground plane e grid para o tamanho de um campo de futebol:
    ground = create_ground_plane(width=120.0, height=80.0)  # 120x80m (maior que campo oficial)
    ground.translate(np.array([52.5, 34.0, 0.0]))  # Centralizar no campo
    vis.add_geometry(ground)

    # Grid com espaçamento maior para campo de futebol
    grid = create_ground_grid(x_min=0, x_max=105, y_min=0, y_max=68, spacing=5.0)  # Espaçamento de 5m
    vis.add_geometry(grid)

    # Store default field geometries to be removed if a custom field is loaded
    default_field_geometries = [ground, grid]

    # Ajustar os marcadores "X" nos cantos do campo:
    corners = [
        np.array([0, 0, 0]),      # bottom_left_corner
        np.array([105, 0, 0]),    # bottom_right_corner  
        np.array([105, 68, 0]),   # top_right_corner
        np.array([0, 68, 0]),     # top_left_corner
    ]
    for corner in corners:
        x_marker = create_x_marker(corner, size=1.0)  # Marcadores maiores
        vis.add_geometry(x_marker)
        default_field_geometries.append(x_marker)

    # Configurar câmera inicial para escala de campo de futebol:
    ctr = vis.get_view_control()
    
    # Check if view control was created successfully
    if ctr is None:
        print("[red] Failed to get view control - camera setup failed[/red]")
        print("[cyan] Trying matplotlib fallback...[/cyan]")
        vis.destroy_window()
        success = run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices)
        return
    
    try:
        bbox_center = np.array([52.5, 34.0, 0.0])  # Centro do campo de futebol
        ctr.set_lookat(bbox_center)
        ctr.set_front(np.array([0, -1, -0.5]))  # Vista diagonal para melhor perspectiva
        ctr.set_up(np.array([0, 0, 1]))
        ctr.set_zoom(0.003)  # Zoom extremamente pequeno para mostrar campo completo

        # Melhorar limites de zoom ainda mais:
        try:
            ctr.set_constant_z_near(0.00001)     # Ultra próximo
            ctr.set_constant_z_far(1000000.0)    # Ultra distante (1000km)
        except AttributeError:
            print("[yellow] Advanced zoom controls not available on this Open3D version[/yellow]")
            
    except Exception as e:
        print(f"[red] Camera setup failed: {str(e)}[/red]")
        print("[cyan] Trying matplotlib fallback...[/cyan]")
        vis.destroy_window()
        success = run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices)
        return

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.line_width = 5.0
    render_option.background_color = np.array([0, 0, 0])
    render_option.light_on = False

    # Inicializar com linhas básicas do campo
    current_field_lines = create_football_field_lines()
    vis.add_geometry(current_field_lines)
    
    # Controle de visibilidade das linhas
    show_field_lines = [True]
    
    # Frame control variables and callback definitions
    current_frame = 0
    is_playing = False

    def toggle_field_lines(_vis_obj):
        """Toggle field lines or load new field lines from CSV"""
        nonlocal current_field_lines, default_field_geometries
        
        if show_field_lines[0]:
            # Se está mostrando, esconder
            vis.remove_geometry(current_field_lines, reset_bounding_box=False)
            print("\nField lines hidden")
            show_field_lines[0] = False
        else:
            # Se está escondido, perguntar se quer carregar novo arquivo
            root = _create_centered_tk_root()
            choice = messagebox.askyesnocancel(
                "Field Lines", 
                "Show current lines or load new CSV file?\n\n"
                "Yes = Show current lines\n"
                "No = Load new CSV file with auto-adjust\n"
                "Cancel = Keep hidden",
                parent=root
            )
            root.destroy()
            
            if choice is True:
                # Mostrar linhas atuais
                vis.add_geometry(current_field_lines, reset_bounding_box=False)
                print("\nField lines visible")
                show_field_lines[0] = True
                
            elif choice is False:
                # Carregar novo arquivo CSV com auto-ajuste
                result = load_field_lines_from_csv()
                if result:
                    # Remove default ground/grid if they exist
                    if default_field_geometries:
                        print("\nRemoving default ground plane and grid.")
                        for geom in default_field_geometries:
                            try:
                                vis.remove_geometry(geom, reset_bounding_box=False)
                            except Exception:
                                pass
                        default_field_geometries = [] # Clear so we don't remove again

                    if len(result) == 3:  # Novo formato com centro e zoom
                        new_lines, center, optimal_zoom = result
                        
                        # Remover linhas antigas se existirem
                        try:
                            vis.remove_geometry(current_field_lines, reset_bounding_box=False)
                        except Exception:
                            pass
                        
                        # Adicionar novas linhas
                        current_field_lines = new_lines
                        vis.add_geometry(current_field_lines, reset_bounding_box=False)
                        
                        # Ajustar câmera automaticamente
                        ctr.set_lookat([center[0], center[1], center[2]])
                        ctr.set_front([0, -1, -0.3])  # Vista ligeiramente diagonal
                        ctr.set_up([0, 0, 1])
                        ctr.set_zoom(optimal_zoom)
                        
                        # Atualizar render options para linhas mais espessas
                        render_option.line_width = 10.0  # Linhas muito mais espessas
                        
                        print("\nNew field lines loaded with auto-adjusted view")
                        print(f"Camera centered at: [{center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}]")
                        print(f"Zoom set to: {optimal_zoom:.4f}")
                        show_field_lines[0] = True
                    else:
                        # Formato antigo (retrocompatibilidade)
                        new_lines = result
                        try:
                            vis.remove_geometry(current_field_lines, reset_bounding_box=False)
                        except Exception:
                            pass
                        current_field_lines = new_lines
                        vis.add_geometry(current_field_lines, reset_bounding_box=False)
                        render_option.line_width = 10.0
                        show_field_lines[0] = True
        
        return False
    
    # Add after creating the spheres
    def create_frame_indicator():
        """Create a visual indicator of the current frame"""
        # Create a coordinate frame at the origin
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)
        # Position at the origin (0, 0, 0)
        coordinate_frame.translate([0, 0, 0])
        return coordinate_frame

    # Add the indicator to the visualization
    frame_indicator = create_frame_indicator()
    vis.add_geometry(frame_indicator)

    # Function to update the radius of the markers
    def update_marker_radius(new_radius):
        nonlocal current_radius, spheres_bases
        current_radius = new_radius
        
        # Recreate the bases of the spheres with the new radius
        for i in range(num_markers):
            # Remove the old sphere
            vis.remove_geometry(spheres[i], reset_bounding_box=False)
            
            # Create new sphere with new radius
            new_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=current_radius, resolution=8
            )
            base_vertices = np.asarray(new_sphere.vertices).copy()
            
            # Position at the current frame position
            current_pos = points[current_frame][i]
            new_sphere.vertices = o3d.utility.Vector3dVector(base_vertices + current_pos)
            new_sphere.paint_uniform_color(available_colors[current_color_index][0])
            
            # Update the references
            spheres[i] = new_sphere
            spheres_bases[i] = base_vertices
            
            # Add the new sphere
            vis.add_geometry(spheres[i], reset_bounding_box=False)
        
        print(f"\nMarker radius: {current_radius:.3f}")
        return False

    # Function to change the color of the markers
    def change_marker_color():
        nonlocal current_color_index
        current_color_index = (current_color_index + 1) % len(available_colors)
        new_color, color_name = available_colors[current_color_index]
        
        # Update the color of all markers
        for sphere in spheres:
            sphere.paint_uniform_color(new_color)
            vis.update_geometry(sphere)
        
        print(f"\nMarker color: {color_name}")
        return False

    # Functions to increase and decrease the radius
    def increase_radius(vis_obj):
        new_radius = min(current_radius + 0.005, 0.05)  # Maximum 0.05
        if new_radius != current_radius:
            update_marker_radius(new_radius)
        return False

    def decrease_radius(vis_obj):
        new_radius = max(current_radius - 0.005, 0.005)  # Minimum 0.005
        if new_radius != current_radius:
            update_marker_radius(new_radius)
        return False

    def cycle_color(vis_obj):
        change_marker_color()
        return False

    # Modified update_spheres function
    def update_spheres(frame_data):
        for i, sphere in enumerate(spheres):
            new_pos = frame_data[i]
            new_vertices = spheres_bases[i] + new_pos
            sphere.vertices = o3d.utility.Vector3dVector(new_vertices)
            vis.update_geometry(sphere)
        
        # Update frame indicator in the terminal
        frame_info = f"Frame {current_frame+1}/{num_frames}"
        print(f"\r{frame_info} - Time: {(current_frame/fps):.3f}s", end="", flush=True)
        
        vis.poll_events()
        vis.update_renderer()

    # Update the window title with the current frame
    frame_info = f"Frame {current_frame+1}/{num_frames}"
    new_title = (
        f"C3D Viewer | File: {file_name} | Markers: {num_markers}/{total_markers} | {frame_info} | FPS: {fps} | "
        "Keys: [←→: Frame, ↑↓: 60 Frames, +/-: Size, C: Color, Space: Play, H: Help] | "
        "Mouse: [Left: Rotate, Middle/Right: Pan, Wheel: Zoom]"
    )
    
    # Use the correct method to update the window title
    try:
        # Try using set_window_name if available
        vis.set_window_name(new_title)
    except AttributeError:
        # If not available, just print the current frame in the terminal
        print(f"\rFrame {current_frame+1}/{num_frames}", end="", flush=True)
    
    vis.poll_events()
    vis.update_renderer()

    def next_frame(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame + 1) % num_frames
        update_spheres(points[current_frame])
        return False

    def previous_frame(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame - 1) % num_frames
        update_spheres(points[current_frame])
        return False

    # Modificar as funções para avançar/voltar 60 frames:
    def forward_60_frames(_vis_obj):
        nonlocal current_frame
        current_frame = (current_frame + 60) % num_frames
        update_spheres(points[current_frame])
        return False

    def backward_60_frames(_vis_obj):
        nonlocal current_frame
        current_frame = (current_frame - 60) % num_frames
        update_spheres(points[current_frame])
        return False

    def toggle_play(vis_obj):
        nonlocal is_playing
        is_playing = not is_playing
        return False

    # Add after the other callback functions
    def set_view_limits(vis_obj):
        """Allow setting custom limits for visualization"""
        
        # Create window for input of limits
        root = _create_centered_tk_root()
        
        try:
            # Get current approximate limits from data
            all_points = points.reshape(-1, 3)
            valid_points = all_points[~np.isnan(all_points).any(axis=1)]
            
            if len(valid_points) > 0:
                current_x_min, current_x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
                current_y_min, current_y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
                current_z_min, current_z_max = valid_points[:, 2].min(), valid_points[:, 2].max()
            else:
                current_x_min = current_x_max = 0
                current_y_min = current_y_max = 0
                current_z_min = current_z_max = 0
            
            print("\nCurrent data bounds:")
            print(f"X: [{current_x_min:.3f}, {current_x_max:.3f}]")
            print(f"Y: [{current_y_min:.3f}, {current_y_max:.3f}]")
            print(f"Z: [{current_z_min:.3f}, {current_z_max:.3f}]")
            
            # Request new limits
            x_min = simpledialog.askfloat("X Axis", f"X minimum (current: {current_x_min:.3f}):", 
                                         initialvalue=current_x_min - 10.0, parent=root)  # 10m de margem
            if x_min is None:
                root.destroy()
                return False
            
            x_max = simpledialog.askfloat("X Axis", f"X maximum (current: {current_x_max:.3f}):", 
                                         initialvalue=current_x_max + 10.0, parent=root)
            if x_max is None:
                root.destroy()
                return False
            
            y_min = simpledialog.askfloat("Y Axis", f"Y minimum (current: {current_y_min:.3f}):", 
                                         initialvalue=current_y_min - 10.0, parent=root)
            if y_min is None:
                root.destroy()
                return False
            
            y_max = simpledialog.askfloat("Y Axis", f"Y maximum (current: {current_y_max:.3f}):", 
                                         initialvalue=current_y_max + 10.0, parent=root)
            if y_max is None:
                root.destroy()
                return False
            
            z_min = simpledialog.askfloat("Z Axis", f"Z minimum (current: {current_z_min:.3f}):", 
                                         initialvalue=current_z_min - 2.0, parent=root)
            if z_min is None:
                root.destroy()
                return False
            
            z_max = simpledialog.askfloat("Z Axis", f"Z maximum (current: {current_z_max:.3f}):", 
                                         initialvalue=current_z_max + 10.0, parent=root)  # 10m de altura
            if z_max is None:
                root.destroy()
                return False
            
            root.destroy()
            
            # Apply new limits adjusting the camera position
            new_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
            
            # Calculate appropriate distance from the camera
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            
            # Adjust visualization
            ctr.set_lookat(new_center)
            # Calculate zoom based on the maximum range
            optimal_zoom = 0.5 / max_range if max_range > 0 else 0.5
            ctr.set_zoom(optimal_zoom)
            
            print("\nNew view limits set:")
            print(f"X: [{x_min:.3f}, {x_max:.3f}]")
            print(f"Y: [{y_min:.3f}, {y_max:.3f}]")
            print(f"Z: [{z_min:.3f}, {z_max:.3f}]")
            print(f"Center: [{new_center[0]:.3f}, {new_center[1]:.3f}, {new_center[2]:.3f}]")
            
        except Exception as e:
            print(f"Error setting view limits: {e}")
            root.destroy()
        
        return False

    # Function to reset the camera
    def reset_camera_view(_vis_obj):
        """Reset camera to default view"""
        ctr.set_lookat([52.5, 34.0, 0.0])  # Centro do campo
        ctr.set_front([0, -1, -0.5])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.003)  # Zoom para mostrar campo completo
        print("\nCamera view reset to football field scale")
        return False

    # Add the show_help function and register the shortcut
    def show_help(_vis_obj):
        """Show help in a matplotlib GUI window"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Add title
        ax.text(5, 9.5, 'C3D VIEWER - KEYBOARD SHORTCUTS', ha='center', va='top', 
                fontsize=16, fontweight='bold')
        
        # Add shortcuts text
        shortcuts_text = """NAVIGATION:
← → - Previous/Next frame
↑ ↓ - Forward/Backward 60 frames
S/E - Jump to start/end
Space - Play/Pause animation

MARKERS:
+/= - Increase marker size
- - Decrease marker size
C - Change marker color

VIEW:
T - Change background color
Y - Change ground plane color
G - Toggle football field lines
R - Reset camera view
L - Set view limits

DATA:
U - Override unit conversion (mm/m)

INFO:
I - Show frame info
O - Show camera parameters
H - Show this help"""
        
        ax.text(0.5, 8.5, shortcuts_text, ha='left', va='top', fontsize=11, 
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
        return False

    # Add these missing functions before the callback registrations:

    def jump_to_start(_vis_obj):
        nonlocal current_frame
        current_frame = 0
        update_spheres(points[current_frame])
        return False

    def jump_to_end(_vis_obj):
        nonlocal current_frame
        current_frame = num_frames - 1
        update_spheres(points[current_frame])
        return False

    # Add variables to control the ground and background colors
    current_ground_color_index = 2  # Green as default
    current_bg_color_index = 0      # Black as default
    
    # Available colors for ground and background
    ground_colors = [
        ([1.0, 0.65, 0.0], "Orange"),
        ([0.0, 0.5, 1.0], "Blue"), 
        ([0.0, 0.8, 0.0], "Green"),    # Darker green for the ground
        ([0.8, 0.0, 0.0], "Red"),      # Darker red
        ([0.8, 0.8, 0.8], "Light Gray"),
        ([1.0, 1.0, 0.0], "Yellow"),
        ([0.5, 0.0, 1.0], "Purple"),
        ([0.0, 0.8, 0.8], "Cyan"),     # Darker cyan
        ([1.0, 0.0, 1.0], "Pink"),
        ([0.4, 0.4, 0.4], "Dark Gray"),
        ([0.2, 0.2, 0.2], "Very Dark Gray")
    ]
    
    # Colors for background
    background_colors = [
        ([0, 0, 0], "Black"),
        ([0.1, 0.1, 0.1], "Very Dark Gray"),
        ([0.2, 0.2, 0.2], "Dark Gray"),
        ([0.4, 0.4, 0.4], "Gray"),
        ([0.6, 0.6, 0.6], "Light Gray"),
        ([0.8, 0.8, 0.8], "Very Light Gray"),
        ([0.9, 0.9, 0.9], "Almost White"),
        ([1.0, 1.0, 1.0], "White"),
        ([0.0, 0.1, 0.2], "Dark Blue"),
        ([0.1, 0.0, 0.1], "Dark Purple"),
        ([0.1, 0.1, 0.0], "Dark Yellow")
    ]

    # Function to change the ground plane color
    def change_ground_color(_vis_obj):
        nonlocal current_ground_color_index
        current_ground_color_index = (current_ground_color_index + 1) % len(ground_colors)
        new_color, color_name = ground_colors[current_ground_color_index]
        
        # Update the ground plane color
        ground.paint_uniform_color(new_color)
        vis.update_geometry(ground)
        
        print(f"\nGround color: {color_name}")
        return False

    # Improved function to change the background
    def toggle_background_advanced(_vis_obj):
        nonlocal current_bg_color_index
        current_bg_color_index = (current_bg_color_index + 1) % len(background_colors)
        new_bg_color, bg_color_name = background_colors[current_bg_color_index]
        
        # Update background color
        render_option.background_color = np.array(new_bg_color)
        
        # Adjust grid color based on background to maintain contrast
        if sum(new_bg_color) < 1.5:  # Dark background
            grid.paint_uniform_color([0.8, 0.8, 0.8])  # Grid claro
        else:  # Light background
            grid.paint_uniform_color([0.2, 0.2, 0.2])  # Grid escuro
            
        vis.update_geometry(grid)
        
        print(f"\nBackground color: {bg_color_name}")
        return False

    def show_frame_info(_vis_obj):
        """Show detailed information about the current frame"""
        if current_frame < len(points):
            frame_data = points[current_frame]
            valid_points = frame_data[~np.isnan(frame_data).any(axis=1)]
            
            if len(valid_points) > 0:
                center = np.mean(valid_points, axis=0)
                height = np.max(valid_points[:, 2]) - np.min(valid_points[:, 2])
                print(f"\n=== Frame {current_frame+1}/{num_frames} ===")
                print(f"Valid markers: {len(valid_points)}/{num_markers}")
                print(f"Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                print(f"Height range: {height:.3f}m")
                print(f"Time: {(current_frame/fps):.3f}s")
            else:
                print(f"Frame {current_frame+1}: No valid markers")
        return False

    # Register all shortcuts
    # Setas esquerda/direita para frame anterior/próximo
    vis.register_key_callback(262, next_frame)      # Seta direita (→)
    vis.register_key_callback(263, previous_frame)  # Seta esquerda (←)

    # Setas cima/baixo para +60/-60 frames
    vis.register_key_callback(264, backward_60_frames)  # Seta baixo (↓) - volta 60
    vis.register_key_callback(265, forward_60_frames)   # Seta cima (↑) - avança 60

    # Manter as teclas N/P e F/B como alternativas
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), previous_frame)
    vis.register_key_callback(ord("F"), forward_60_frames)
    vis.register_key_callback(ord("B"), backward_60_frames)

    # Outros atalhos
    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(ord("O"), lambda _vis_obj: print(ctr.convert_to_pinhole_camera_parameters().extrinsic))
    vis.register_key_callback(ord("+"), increase_radius)
    vis.register_key_callback(ord("="), increase_radius)
    vis.register_key_callback(ord("-"), decrease_radius)
    vis.register_key_callback(ord("C"), cycle_color)
    vis.register_key_callback(ord("S"), jump_to_start)
    vis.register_key_callback(ord("E"), jump_to_end)
    
    # Novos atalhos para cores
    vis.register_key_callback(ord("T"), toggle_background_advanced)  # Background colorido
    vis.register_key_callback(ord("Y"), change_ground_color)         # Ground plane colorido
    vis.register_key_callback(ord("L"), set_view_limits)
    vis.register_key_callback(ord("I"), show_frame_info)
    vis.register_key_callback(ord("H"), show_help)
    vis.register_key_callback(ord("G"), toggle_field_lines)  # Use G ao invés de F
    
    # Add unit conversion override key
    def override_units(_vis_obj):
        """Allow user to manually override unit conversion"""
        nonlocal points, spheres, spheres_bases
        
        user_choice = ask_user_units_c3d()
        
        if user_choice == 'mm':
            # Convert from meters back to mm, then to meters (effectively *1000 then *0.001 = *1)
            # But first we need to "undo" the previous conversion if it was done
            print("[yellow]Converting data assuming current data is in millimeters...[/yellow]")
            points = points * 0.001  # Assuming current is mm, convert to meters
            
        elif user_choice == 'm':
            # Convert from mm to meters (multiply by 1000 then by 0.001)
            print("[yellow]Converting data assuming current data is in meters...[/yellow]")
            points = points * 1000 * 0.001  # This keeps it the same but for demonstration
            
        # Recreate spheres with new positions
        for i in range(num_markers):
            vis.remove_geometry(spheres[i], reset_bounding_box=False)
            
            new_sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=current_radius, resolution=8
            )
            base_vertices = np.asarray(new_sphere.vertices).copy()
            current_pos = points[current_frame][i]
            new_sphere.vertices = o3d.utility.Vector3dVector(base_vertices + current_pos)
            new_sphere.paint_uniform_color(available_colors[current_color_index][0])
            
            spheres[i] = new_sphere
            spheres_bases[i] = base_vertices
            vis.add_geometry(spheres[i], reset_bounding_box=False)
        
        # Update current frame
        update_spheres(points[current_frame])
        
        print("[bold green]Unit conversion override applied![/bold green]")
        return False
    
    vis.register_key_callback(ord("U"), override_units)  # U for Units

    # --- Field Drawing Logic from soccerfield.py ---

    def draw_line_3d(vis, p1, p2, color=[1, 1, 1], width=0.02):
        """Draws a 3D line in Open3D"""
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([p1, p2])
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set.colors = o3d.utility.Vector3dVector([color])
        vis.add_geometry(line_set, reset_bounding_box=False)
        return line_set

    def draw_circle_3d(vis, center, normal, radius, color=[1, 1, 1], resolution=100):
        """Draws a 3D circle in Open3D"""
        # Create a circle on the XY plane
        t = np.linspace(0, 2 * np.pi, resolution)
        points = np.vstack([radius * np.cos(t), radius * np.sin(t), np.zeros(resolution)]).T
        
        # Find rotation to align the circle's normal with the given normal
        z_axis = np.array([0, 0, 1])
        rotation = rotation_matrix_from_vectors(z_axis, normal)
        points = points @ rotation.T
        
        # Translate to the center
        points += center
        
        lines = [[i, i + 1] for i in range(resolution - 1)] + [[resolution - 1, 0]]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
        vis.add_geometry(line_set, reset_bounding_box=False)
        return line_set

    def draw_arc_3d(vis, center, normal, radius, start_angle, end_angle, color=[1, 1, 1], resolution=50):
        """Draws a 3D arc in Open3D"""
        t = np.linspace(np.deg2rad(start_angle), np.deg2rad(end_angle), resolution)
        points = np.vstack([radius * np.cos(t), radius * np.sin(t), np.zeros(resolution)]).T

        z_axis = np.array([0, 0, 1])
        rotation = rotation_matrix_from_vectors(z_axis, normal)
        points = points @ rotation.T
        
        points += center
        
        lines = [[i, i + 1] for i in range(resolution - 1)]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
        vis.add_geometry(line_set, reset_bounding_box=False)
        return line_set
        
    def rotation_matrix_from_vectors(vec1, vec2):
        """Find the rotation matrix that aligns vec1 to vec2"""
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s == 0:
            return np.eye(3)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    # Replace the simple line drawing with a full field drawing
    def create_full_soccer_field(vis, df):
        points = {row["point_name"]: (row["x"], row["y"], row["z"]) for _, row in df.iterrows()}
        geometries = []
        
        # Perimeter
        p = points
        geometries.append(draw_line_3d(vis, p['bottom_left_corner'], p['top_left_corner']))
        geometries.append(draw_line_3d(vis, p['top_left_corner'], p['top_right_corner']))
        geometries.append(draw_line_3d(vis, p['top_right_corner'], p['bottom_right_corner']))
        geometries.append(draw_line_3d(vis, p['bottom_right_corner'], p['bottom_left_corner']))
        
        # Center line
        geometries.append(draw_line_3d(vis, p['midfield_left'], p['midfield_right']))
        
        # Center circle
        radius = np.linalg.norm(np.array(p['center_circle_top_intersection']) - np.array(p['center_field']))
        geometries.append(draw_circle_3d(vis, p['center_field'], [0, 0, 1], radius))
        
        # Penalty areas and goal areas (rectangles)
        # Left penalty area
        geometries.append(draw_line_3d(vis, p['left_penalty_area_bottom_left'], p['left_penalty_area_top_left']))
        geometries.append(draw_line_3d(vis, p['left_penalty_area_top_left'], p['left_penalty_area_top_right']))
        geometries.append(draw_line_3d(vis, p['left_penalty_area_top_right'], p['left_penalty_area_bottom_right']))
        geometries.append(draw_line_3d(vis, p['left_penalty_area_bottom_right'], p['left_penalty_area_bottom_left']))

        # Right penalty area
        geometries.append(draw_line_3d(vis, p['right_penalty_area_bottom_left'], p['right_penalty_area_top_left']))
        geometries.append(draw_line_3d(vis, p['right_penalty_area_top_left'], p['right_penalty_area_top_right']))
        geometries.append(draw_line_3d(vis, p['right_penalty_area_top_right'], p['right_penalty_area_bottom_right']))
        geometries.append(draw_line_3d(vis, p['right_penalty_area_bottom_right'], p['right_penalty_area_bottom_left']))

        # Penalty Arcs
        # Left
        center_l = p['left_penalty_spot']
        p1 = np.array(p['left_penalty_arc_left_intersection'])
        p2 = np.array(p['left_penalty_arc_right_intersection'])
        radius_l = np.linalg.norm(p1 - center_l)
        angle1_l = np.rad2deg(np.arctan2(p1[1]-center_l[1], p1[0]-center_l[0]))
        angle2_l = np.rad2deg(np.arctan2(p2[1]-center_l[1], p2[0]-center_l[0]))
        geometries.append(draw_arc_3d(vis, center_l, [0,0,1], radius_l, angle1_l, angle2_l))

        # Right
        center_r = p['right_penalty_spot']
        p1_r = np.array(p['right_penalty_arc_left_intersection'])
        p2_r = np.array(p['right_penalty_arc_right_intersection'])
        radius_r = np.linalg.norm(p1_r - center_r)
        angle1_r = np.rad2deg(np.arctan2(p1_r[1]-center_r[1], p1_r[0]-center_r[0]))
        angle2_r = np.rad2deg(np.arctan2(p2_r[1]-center_r[1], p2_r[0]-center_r[0]))
        geometries.append(draw_arc_3d(vis, center_r, [0,0,1], radius_r, angle2_r, angle1_r))

        return geometries

    # --- End of Field Drawing Logic ---

    # Atualizar o título da janela
    window_title = (
        f"C3D Viewer | File: {file_name} | Markers: {num_markers}/{total_markers} | Frames: {num_frames} | FPS: {fps} | "
        "Keys: [←→: Frame, ↑↓: 60 Frames, C: Color, U: Units, G: Field, H: Help] | "
        "Mouse: [Left: Rotate, Middle/Right: Pan, Wheel: Zoom]"
    )

    # Main loop for automatic playback (using FPS from file)
    last_time = time.time()
    while True:
        if not vis.poll_events():
            break
        if is_playing:
            current_time = time.time()
            if current_time - last_time >= 1.0 / fps:
                next_frame(vis)
                last_time = current_time
        else:
            time.sleep(0.01)  # Small sleep to reduce CPU usage when not playing
    vis.destroy_window()


def load_field_lines_from_csv():
    """Load field lines from a CSV file"""
    root = _create_centered_tk_root()
    csv_file = filedialog.askopenfilename(
        title="Select field lines CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        parent=root
    )
    root.destroy()
    
    if not csv_file:
        return None
    
    try:
        # Ler o arquivo CSV
        df = pd.read_csv(csv_file)
        
        # Check for named points required for full field drawing
        required_points = ['bottom_left_corner', 'center_field', 'left_penalty_spot']
        if 'point_name' in df.columns and all(p in df['point_name'].values for p in required_points):
             print("Detected a full field definition CSV. Drawing all features.")
             # The create_full_soccer_field function will be called inside run_viewc3d
             # We just need to return the dataframe.
             return df
        else:
            print("Detected a simple line CSV. Drawing connected points.")
            # Fallback to simple line drawing
            lines_points = []
            lines_indices = []
            current_line_start_index = 0

            # Process points, treating blank rows as line breaks
            for i, row in df.iterrows():
                if row.isnull().all():
                    # End of a line strip. Connect points if there are any.
                    if len(lines_points) > current_line_start_index + 1:
                        for j in range(current_line_start_index, len(lines_points) - 1):
                            lines_indices.append([j, j + 1])
                    # The start of the next line will be the current number of points
                    current_line_start_index = len(lines_points)
                else:
                    lines_points.append([row['x'], row['y'], row['z']])

            # Add the last line strip if it exists
            if len(lines_points) > current_line_start_index + 1:
                for j in range(current_line_start_index, len(lines_points) - 1):
                    lines_indices.append([j, j + 1])
            
            if not lines_indices:
                 messagebox.showwarning("Warning", "No lines could be generated from the CSV. Ensure it contains at least two points per line segment.")
                 return None

            # Criar LineSet com linhas mais espessas
            field_lines = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(lines_points),
                lines=o3d.utility.Vector2iVector(lines_indices)
            )
            field_lines.paint_uniform_color([1.0, 1.0, 1.0])  # Linhas brancas
            
            # Calcular limites dos dados para ajuste automático da câmera
            points_array = np.array(lines_points)
            x_min, x_max = points_array[:, 0].min(), points_array[:, 0].max()
            y_min, y_max = points_array[:, 1].min(), points_array[:, 1].max()
            z_min, z_max = points_array[:, 2].min(), points_array[:, 2].max()
            
            # Calcular centro e escala
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_z = (z_min + z_max) / 2
            
            # Calcular range máximo para determinar zoom apropriado
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            
            # Calcular zoom baseado no tamanho dos dados
            # Para um campo de futebol (105m), zoom ~0.003 funciona bem
            optimal_zoom = 0.3 / max_range if max_range > 0 else 0.003
            optimal_zoom = max(0.001, min(optimal_zoom, 0.1))  # Limitar entre 0.001 e 0.1
            
            print(f"Loaded {len(lines_points)} points and {len(lines_indices)} lines from {os.path.basename(csv_file)}")
            print(f"Data bounds: X[{x_min:.1f}, {x_max:.1f}] Y[{y_min:.1f}, {y_max:.1f}] Z[{z_min:.1f}, {z_max:.1f}]")
            print(f"Center: [{center_x:.1f}, {center_y:.1f}, {center_z:.1f}], Optimal zoom: {optimal_zoom:.4f}")
            
            return field_lines, (center_x, center_y, center_z), optimal_zoom
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
        return None

def create_football_field_lines():
    """Creates basic football field markings as default"""
    lines_points = []
    lines_indices = []
    
    # Campo oficial: 105m x 68m
    field_length = 105.0
    field_width = 68.0
    
    # Apenas as linhas básicas do campo
    # Retângulo externo
    corners = [
        [-field_length/2, -field_width/2, 0],
        [field_length/2, -field_width/2, 0],
        [field_length/2, field_width/2, 0],
        [-field_length/2, field_width/2, 0]
    ]
    
    for i in range(len(corners)):
        lines_points.append(corners[i])
        lines_indices.append([i, (i + 1) % len(corners)])
    
    # Linha do meio
    lines_points.extend([[0, -field_width/2, 0], [0, field_width/2, 0]])
    lines_indices.append([len(lines_points)-2, len(lines_points)-1])
    
    # Criar LineSet
    field_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lines_points),
        lines=o3d.utility.Vector2iVector(lines_indices)
    )
    field_lines.paint_uniform_color([1.0, 1.0, 1.0])
    return field_lines


if __name__ == "__main__":
    run_viewc3d()
