"""
================================================================================
Script: viewc3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.9
Created: 06 February 2025
Last Updated: 15 September 2025

Description:
------------
Advanced 3D viewer for C3D files with adaptive visualization for different scales.
Automatically detects and converts units (millimeters/meters) with enhanced confidence scoring.
Features adaptive ground plane, grid, and camera positioning based on data scale.
Features soccer field lines and penalty areas.

    Key Features:
    - Adaptive visualization for small (lab) to large (soccer field) scales
    - Automatic unit detection (mm/m) with confidence scoring
    - Interactive marker selection with search and filter options
    - Real-time marker labels with color coding
    - Ground grid toggle and field line customization
    - Matplotlib fallback for systems without OpenGL support
    
    Keyboard Shortcuts:
    Navigation:
      ← → - Previous/Next frame
      ↑ ↓ - Forward/Backward 60 frames  
      S/E - Jump to start/end
      Space - Play/Pause animation
    
    Markers:
      +/-/= - Increase/Decrease marker size
      C - Change marker color
      X - Toggle marker labels (names)
    
    View:
      T - Change background color
      Y - Change ground plane color
      G - Toggle football field lines
      M - Toggle ground grid
      R - Reset camera view
      L - Set custom view limits
    
    Data:
      U - Override unit conversion (mm/m)
    
    Info:
      I - Show frame info
      O - Show camera parameters
      H - Show help
    
    Mouse Controls:
      Left Drag - Rotate view
      Middle/Right Drag - Pan view
      Mouse Wheel - Zoom in/out
    
    Optimizations:
    - Adaptive ground plane and grid based on data bounds
    - Intelligent camera positioning for different scales
    - Enhanced unit detection with statistical analysis
    - Real-time label updates during animation
    - Robust fallback visualization system
"""

import os
from pathlib import Path
import open3d as o3d
import ezc3d
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import time
from rich import print
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import subprocess
import tempfile
import shutil
import webbrowser


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
        frame_indices = np.linspace(0, pts.shape[0]-1, min(10, pts.shape[0]), dtype=int)
        all_distances = []
        
        for frame_idx in frame_indices:
            frame = pts[frame_idx]
            valid_markers = frame[~np.isnan(frame).any(axis=1)]
            
            if len(valid_markers) > 1:
                for i in range(len(valid_markers)):
                    for j in range(i+1, len(valid_markers)):
                        dist = np.linalg.norm(valid_markers[i] - valid_markers[j])
                        all_distances.append(dist)
        
        if all_distances:
            avg_distance = np.mean(all_distances)
            max_distance = np.max(all_distances)
            percentile_95 = np.percentile(all_distances, 95)
            
            # More sophisticated distance analysis
            if avg_distance > 200 or max_distance > 4000:
                confidence_score += 3
                detection_reasons.append(f"large_distances (avg: {avg_distance:.1f}, max: {max_distance:.1f})")
            elif avg_distance > 50 or percentile_95 > 2000:
                confidence_score += 2
                detection_reasons.append(f"moderate_distances (avg: {avg_distance:.1f}, p95: {percentile_95:.1f})")
    
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
    # Allow bypassing dialog via environment variable for automated tests
    env_path = os.environ.get("VIEWC3D_FILE", "").strip()
    if env_path and os.path.exists(env_path):
        filepath = env_path
        print(f"[cyan]Using C3D file from VIEWC3D_FILE:[/cyan] {filepath}")
    else:
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
    # Non-interactive bypass for automated runs
    env_sel = os.environ.get("VIEWC3D_MARKERS", "").strip().lower()
    if env_sel == "all":
        return list(range(len(marker_labels)))

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
    import platform
    
    # Check if we're in a remote session or headless environment
    if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
        return False, "SSH remote session detected"
    
    # macOS doesn't use DISPLAY variable - it has its own display system
    if platform.system() == 'Darwin':  # macOS
        return True, "macOS display environment"
    
    # For Linux and other Unix-like systems, check DISPLAY
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
    import platform
    
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
        
        # Special handling for macOS GLFW issues
        if platform.system() == 'Darwin' and 'glfw' in error_msg:
            return False, f"macOS GLFW error (common on Apple Silicon M-series): {str(e)}"
        elif 'glx' in error_msg or 'glfw' in error_msg:
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
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting viewc3d.py...")
    print("-" * 80)

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
        import platform
        print("[yellow] OpenGL/Open3D not supported on this system:[/yellow]")
        print(f"[yellow]  {error_msg}[/yellow]")
        
        # Platform-specific messages
        if platform.system() == 'Darwin':
            print("[yellow]  This is common on macOS with Apple Silicon (M1/M2/M3/M4) due to GLFW compatibility issues[/yellow]")
            print("[yellow]  The matplotlib fallback provides full functionality for C3D visualization[/yellow]")
        else:
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
        # Create window with Blender-like aspect (16:9)
        window_created = vis.create_window(window_name=window_title, width=1280, height=720)
        
        if not window_created:
            print("[red] Failed to create Open3D window[/red]")
            print("[cyan] Trying matplotlib fallback...[/cyan]")
            success = run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices)
            return
            
    except Exception as e:
        import platform
        error_msg = str(e).lower()
        
        # Special handling for macOS Apple Silicon GLFW issues
        if platform.system() == 'Darwin' and 'glfw' in error_msg:
            print(f"[yellow] macOS GLFW error (common on Apple Silicon M1/M2/M3/M4): {str(e)}[/yellow]")
            print("[yellow] This is a known issue with Open3D on Apple Silicon Macs[/yellow]")
            print("[cyan] Switching to matplotlib fallback visualization...[/cyan]")
        else:
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

    # Calculate data bounds for adaptive ground plane and grid
    all_points_data = points.reshape(-1, 3)
    valid_data_points = all_points_data[~np.isnan(all_points_data).any(axis=1)]
    
    if len(valid_data_points) > 0:
        data_x_min, data_x_max = valid_data_points[:, 0].min(), valid_data_points[:, 0].max()
        data_y_min, data_y_max = valid_data_points[:, 1].min(), valid_data_points[:, 1].max()
        data_z_min, data_z_max = valid_data_points[:, 2].min(), valid_data_points[:, 2].max()
        
        # Add margins (20% of range or minimum 2 meters)
        x_range = data_x_max - data_x_min
        y_range = data_y_max - data_y_min
        margin_x = max(x_range * 0.2, 2.0)
        margin_y = max(y_range * 0.2, 2.0)
        
        ground_x_min = data_x_min - margin_x
        ground_x_max = data_x_max + margin_x
        ground_y_min = data_y_min - margin_y
        ground_y_max = data_y_max + margin_y
        ground_z = 0.0  # Ground always at Z=0 level
        
        # Determine appropriate grid spacing based on data scale
        max_range = max(x_range, y_range)
        if max_range > 50:  # Large scale (like soccer field)
            grid_spacing = 5.0
        elif max_range > 10:  # Medium scale
            grid_spacing = 1.0
        else:  # Small scale
            grid_spacing = 0.5
    else:
        # Fallback for no valid data
        ground_x_min, ground_x_max = -10, 10
        ground_y_min, ground_y_max = -10, 10
        ground_z = 0
        grid_spacing = 1.0
        # Ensure max_range is defined for later usage (e.g., corner X markers sizing)
        max_range = max(ground_x_max - ground_x_min, ground_y_max - ground_y_min)
    
    # Create adaptive ground plane
    ground_width = ground_x_max - ground_x_min
    ground_height = ground_y_max - ground_y_min
    ground_center_x = (ground_x_min + ground_x_max) / 2
    ground_center_y = (ground_y_min + ground_y_max) / 2
    
    ground = create_ground_plane(width=ground_width, height=ground_height)
    ground.translate(np.array([ground_center_x, ground_center_y, ground_z]))
    vis.add_geometry(ground)

    # Create adaptive grid
    grid = create_ground_grid(x_min=ground_x_min, x_max=ground_x_max, 
                             y_min=ground_y_min, y_max=ground_y_max, 
                             spacing=grid_spacing)
    # Adjust grid Z position to be slightly above ground
    grid_points = np.asarray(grid.points)
    grid_points[:, 2] = ground_z + 0.001  # Slightly above ground to avoid Z-fighting
    grid.points = o3d.utility.Vector3dVector(grid_points)
    vis.add_geometry(grid)

    # Store default field geometries to be removed if a custom field is loaded
    default_field_geometries = [ground, grid]

    # Add corner markers at ground level
    marker_size = max(0.1, max_range * 0.01)  # Adaptive marker size
    corners = [
        np.array([ground_x_min, ground_y_min, ground_z]),
        np.array([ground_x_max, ground_y_min, ground_z]),
        np.array([ground_x_max, ground_y_max, ground_z]),
        np.array([ground_x_min, ground_y_max, ground_z])
    ]
    for corner in corners:
        x_marker = create_x_marker(corner, size=marker_size)
        vis.add_geometry(x_marker)
        default_field_geometries.append(x_marker)

    # Configure adaptive camera
    ctr = vis.get_view_control()
    
    # Check if view control was created successfully
    if ctr is None:
        print("[red] Failed to get view control - camera setup failed[/red]")
        print("[cyan] Trying matplotlib fallback...[/cyan]")
        vis.destroy_window()
        success = run_viewc3d_fallback(points, filepath, fps, marker_labels, selected_indices)
        return
    
    try:
        def compute_lookat_extrinsic(eye, center, up):
            forward = center - eye
            forward = forward / (np.linalg.norm(forward) + 1e-12)
            upn = up / (np.linalg.norm(up) + 1e-12)
            right = np.cross(forward, upn)
            right = right / (np.linalg.norm(right) + 1e-12)
            up2 = np.cross(right, forward)
            R = np.eye(4)
            R[0, :3] = right
            R[1, :3] = up2
            R[2, :3] = -forward
            T = np.eye(4)
            T[:3, 3] = -eye
            return R @ T

        def set_camera_blender_like(view_ctl, center, x_range, y_range, z_range, fov_x_deg=40.0, width=1280, height=720, margin=1.3):
            params = view_ctl.convert_to_pinhole_camera_parameters()
            # Intrinsics like Blender 16:9 with ~40° horizontal FOV
            fov_x = np.deg2rad(fov_x_deg)
            fx = width / (2.0 * np.tan(fov_x / 2.0))
            fy = fx  # square pixels
            params.intrinsic.width = width
            params.intrinsic.height = height
            params.intrinsic.set_intrinsics(width, height, fx, fy, width / 2.0, height / 2.0)

            # Vertical FOV implied
            fov_y = 2.0 * np.arctan((height / 2.0) / fy)

            # Fit both axes
            half_w = max(x_range, 1e-6) * 0.5
            half_h = max(y_range, 1e-6) * 0.5
            dist_x = half_w / np.tan(fov_x / 2.0)
            dist_y = half_h / np.tan(fov_y / 2.0)
            base_dist = max(dist_x, dist_y)
            # Consider vertical spread a bit
            base_dist = max(base_dist, (z_range * 0.5) / np.tan(fov_y / 3.0))
            distance = base_dist * margin

            front_dir = np.array([0.0, -1.0, -0.5])
            front_dir = front_dir / (np.linalg.norm(front_dir) + 1e-12)
            eye = center - front_dir * distance
            up = np.array([0.0, 0.0, 1.0])
            params.extrinsic = compute_lookat_extrinsic(eye, center, up)

            view_ctl.convert_from_pinhole_camera_parameters(params)
            try:
                # Generous near/far planes based on scale
                max_dimension = max(x_range, y_range, z_range, 1.0)
                view_ctl.set_constant_z_near(max_dimension * 0.0001)
                view_ctl.set_constant_z_far(max_dimension * 2000.0)
            except AttributeError:
                pass

        # Determine center and ranges (include origin to show axes)
        data_center = np.array([ground_center_x, ground_center_y, (data_z_min + data_z_max) / 2])
        x_range = ground_width
        y_range = ground_height
        z_range = max(0.1, data_z_max - data_z_min)

        # Slightly extend ranges to include origin and give more air
        x_range = max(x_range, abs(data_center[0]) * 2.0, 1.0)
        y_range = max(y_range, abs(data_center[1]) * 2.0, 1.0)
        z_range = max(z_range, abs(data_center[2]) * 2.0, 0.5)

        set_camera_blender_like(ctr, data_center, x_range, y_range, z_range, fov_x_deg=40.0, width=1280, height=720, margin=1.35)
        print("[green]Camera configured with Blender-like FOV (~40° horiz) and 16:9 aspect[/green]")
            
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

    # Inicializar com linhas básicas do campo no nível correto do solo
    current_field_lines = [create_football_field_lines(ground_z)]
    for geom in current_field_lines:
        vis.add_geometry(geom)
    
    # Controle de visibilidade das linhas e grid
    show_field_lines = [True]
    show_grid = [True]  # Grid is initially visible
    
    # Initialize marker labels system
    marker_labels_visible = [False]  # Labels initially hidden
    marker_label_texts = []  # Will store text geometries
    
    # Get selected marker names for labeling
    selected_marker_names = [marker_labels[i] for i in selected_indices]
    
    # ---- Advanced Features State (Trails, Skeleton, Measurements, Capture) ----
    trails_enabled = [False]
    trail_length = [120]  # frames
    trails_positions = [deque(maxlen=trail_length[0]) for _ in range(num_markers)]
    trails_linesets = [None for _ in range(num_markers)]

    skeleton_connections = []           # list of (idx_a, idx_b) in selected marker indices space
    skeleton_linesets = []

    measurement_pair = [None, None]     # (idx_a, idx_b)
    measurement_lineset = [None]

    screenshot_serial = [0]

    def _safe_add_geometry(geometry):
        try:
            vis.add_geometry(geometry, reset_bounding_box=False)
        except Exception as exc:
            print(f"[yellow]Could not add geometry: {exc}[/yellow]")

    def _safe_remove_geometry(geometry):
        try:
            vis.remove_geometry(geometry, reset_bounding_box=False)
        except Exception:
            pass

    def _color_map_speed(speeds):
        """Map speeds (array) to RGB colors (blue->green->red)."""
        if len(speeds) == 0:
            return np.zeros((0, 3))
        v = np.array(speeds)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v_min, v_max = float(np.min(v)), float(np.max(v))
        if v_max - v_min < 1e-9:
            t = np.zeros_like(v)
        else:
            t = (v - v_min) / (v_max - v_min)
        # simple RGB mapping
        colors = np.stack([t, 1.0 - np.abs(t - 0.5) * 2.0, 1.0 - t], axis=1)
        colors = np.clip(colors, 0.0, 1.0)
        return colors

    def _update_single_trail(marker_index, new_pos):
        """Append position and update/create its LineSet with per-segment speed color."""
        if np.isnan(new_pos).any():
            return
        trails_positions[marker_index].append(new_pos.copy())
        positions = np.array(trails_positions[marker_index])
        if len(positions) < 2:
            return
        # Build lines between consecutive positions
        lines = [[i, i + 1] for i in range(len(positions) - 1)]
        # Speeds magnitude between segments
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        colors = _color_map_speed(diffs)

        if trails_linesets[marker_index] is None:
            ls = o3d.geometry.LineSet()
            trails_linesets[marker_index] = ls
            _safe_add_geometry(ls)

        ls = trails_linesets[marker_index]
        ls.points = o3d.utility.Vector3dVector(positions)
        ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
        if len(colors) > 0:
            ls.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(ls)

    def toggle_trails(_vis_obj):
        trails_enabled[0] = not trails_enabled[0]
        if not trails_enabled[0]:
            # Remove existing trails
            for ls in trails_linesets:
                if ls is not None:
                    _safe_remove_geometry(ls)
            for i in range(num_markers):
                trails_linesets[i] = None
                trails_positions[i].clear()
            print("\nTrails disabled")
        else:
            # Initialize with current frame positions
            for i in range(num_markers):
                trails_positions[i].clear()
                if not np.isnan(points[0, i]).any():
                    trails_positions[i].append(points[0, i].copy())
            print(f"\nTrails enabled (length={trail_length[0]} frames)")
        return False

    def load_skeleton_from_json(_vis_obj):
        """Load a skeleton connections JSON and draw dynamic lines following markers.
        Robust name mapping with normalization and first-valid-frame initialization."""
        nonlocal skeleton_connections, skeleton_linesets
        root = _create_centered_tk_root()
        json_file = filedialog.askopenfilename(
            title="Select skeleton JSON",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        root.destroy()
        if not json_file:
            return False
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            conns = data.get("connections", [])
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load JSON: {exc}")
            return False

        def _norm(s: str) -> str:
            s = s.strip().lower()
            # remove separators
            out = []
            for ch in s:
                if ch.isalnum():
                    out.append(ch)
            return ''.join(out)

        # Build normalized map from selected names to indices
        norm_to_indices = {}
        for idx, name in enumerate(selected_marker_names):
            n = _norm(name)
            norm_to_indices.setdefault(n, []).append(idx)

        def _resolve_name(target: str):
            # Direct pN mapping to original marker index (1-based -> 0-based)
            t = target.strip()
            if len(t) >= 2 and (t[0] == 'p' or t[0] == 'P') and t[1:].isdigit():
                n = int(t[1:])
                global_idx = n - 1
                if 0 <= global_idx < len(marker_labels):
                    try:
                        # map original index to selected space
                        sel_pos = selected_indices.index(global_idx)
                        return sel_pos
                    except ValueError:
                        # not selected; cannot map into current points subset
                        return None
            nt = _norm(target)
            # Exact normalized match
            if nt in norm_to_indices and len(norm_to_indices[nt]) == 1:
                return norm_to_indices[nt][0]
            # Heuristic: unique contains match
            candidates = []
            for nsel, idcs in norm_to_indices.items():
                if nt in nsel or nsel in nt:
                    candidates.extend(idcs)
            candidates = list(dict.fromkeys(candidates))  # unique order
            if len(candidates) == 1:
                return candidates[0]
            return None

        mapped = []
        unresolved = []
        for pair in conns:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = str(pair[0]), str(pair[1])
            ia = _resolve_name(a)
            ib = _resolve_name(b)
            if ia is not None and ib is not None:
                mapped.append((ia, ib))
            else:
                unresolved.append((a, b))

        # Remove previous skeleton
        for ls in skeleton_linesets:
            _safe_remove_geometry(ls)
        skeleton_linesets = []
        skeleton_connections = mapped

        # Create linesets for each connection with first valid frame
        for (ia, ib) in skeleton_connections:
            # Find first frame where both are valid
            init_pts = None
            for fidx in range(num_frames):
                pa = points[fidx, ia]
                pb = points[fidx, ib]
                if not (np.isnan(pa).any() or np.isnan(pb).any()):
                    init_pts = np.array([pa, pb])
                    break
            if init_pts is None:
                # No valid data for this pair; skip
                continue
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(init_pts)
            ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
            ls.paint_uniform_color([1.0, 1.0, 1.0])
            _safe_add_geometry(ls)
            skeleton_linesets.append(ls)

        print(f"\nLoaded skeleton connections: {len(skeleton_connections)}")
        if unresolved:
            print("Unresolved label pairs (no match in selected markers):")
            for a, b in unresolved[:10]:
                print(f"  - {a} / {b}")
            if len(unresolved) > 10:
                print(f"  ... and {len(unresolved) - 10} more")
        if not skeleton_linesets:
            messagebox.showwarning("Skeleton", "No skeleton segments were created. Check marker names in the JSON vs selected markers.")
        vis.update_renderer()
        return False

    def _update_skeleton_lines(frame_idx):
        if not skeleton_linesets:
            return
        frame = points[frame_idx]
        for (ls, (ia, ib)) in zip(skeleton_linesets, skeleton_connections):
            pa = frame[ia]
            pb = frame[ib]
            if np.isnan(pa).any() or np.isnan(pb).any():
                # keep previous to avoid flicker
                continue
            ls.points = o3d.utility.Vector3dVector(np.array([pa, pb]))
            vis.update_geometry(ls)

    def measure_distance_between_two_markers(_vis_obj):
        """Prompt two marker names and display/update a measurement line."""
        nonlocal measurement_pair, measurement_lineset
        root = _create_centered_tk_root()
        try:
            a = simpledialog.askstring(
                "Distance Measurement",
                "First marker name (case-sensitive):\n" + ", ".join(selected_marker_names[:30]) + (" ..." if len(selected_marker_names) > 30 else "")
            )
            if a is None:
                return False
            b = simpledialog.askstring(
                "Distance Measurement",
                "Second marker name (case-sensitive):\n" + ", ".join(selected_marker_names[:30]) + (" ..." if len(selected_marker_names) > 30 else "")
            )
            if b is None:
                return False
        finally:
            root.destroy()

        name_to_idx = {name: idx for idx, name in enumerate(selected_marker_names)}
        if a not in name_to_idx or b not in name_to_idx:
            messagebox.showwarning("Warning", "One or both marker names not found among selected markers.")
            return False
        ia, ib = name_to_idx[a], name_to_idx[b]
        measurement_pair[0], measurement_pair[1] = ia, ib

        # Create or update the measurement line now
        frame = points[current_frame]
        pa, pb = frame[ia], frame[ib]
        if np.isnan(pa).any() or np.isnan(pb).any():
            print("\nCannot draw measurement: marker position is NaN on current frame")
            return False
        if measurement_lineset[0] is None:
            ls = o3d.geometry.LineSet()
            measurement_lineset[0] = ls
            _safe_add_geometry(ls)
        ls = measurement_lineset[0]
        ls.points = o3d.utility.Vector3dVector(np.array([pa, pb]))
        ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        ls.paint_uniform_color([1.0, 1.0, 0.0])
        vis.update_geometry(ls)

        dist = float(np.linalg.norm(pb - pa))
        print(f"\nDistance {a} - {b}: {dist:.3f} m")
        return False

    def _update_measurement_line(frame_idx):
        if measurement_lineset[0] is None or measurement_pair[0] is None:
            return
        ia, ib = measurement_pair
        frame = points[frame_idx]
        pa, pb = frame[ia], frame[ib]
        if np.isnan(pa).any() or np.isnan(pb).any():
            return
        measurement_lineset[0].points = o3d.utility.Vector3dVector(np.array([pa, pb]))
        vis.update_geometry(measurement_lineset[0])

    def save_screenshot(_vis_obj):
        """Save a screenshot PNG next to the C3D file."""
        out_dir = os.path.dirname(filepath)
        screenshot_serial[0] += 1
        out_path = os.path.join(out_dir, f"viewc3d_frame{current_frame:05d}_{screenshot_serial[0]:03d}.png")
        try:
            vis.capture_screen_image(out_path, do_render=True)
            print(f"\nSaved screenshot: {out_path}")
        except Exception as exc:
            print(f"\n[red]Failed to save screenshot:[/red] {exc}")
        return False

    def export_png_sequence(_vis_obj):
        """Export the whole sequence as PNG images in a chosen directory."""
        nonlocal current_frame, is_playing
        root = _create_centered_tk_root()
        out_dir = filedialog.askdirectory(title="Select output directory for PNG sequence")
        root.destroy()
        if not out_dir:
            return False
        was_playing = is_playing
        is_playing = False
        start_frame = current_frame
        print(f"\nExporting PNG sequence to: {out_dir}")
        for fidx in range(num_frames):
            current_frame = fidx
            update_spheres(points[current_frame])
            out_path = os.path.join(out_dir, f"frame_{fidx:05d}.png")
            try:
                vis.capture_screen_image(out_path, do_render=True)
            except Exception as exc:
                print(f"\n[red]Failed at frame {fidx}:[/red] {exc}")
                break
        current_frame = start_frame
        update_spheres(points[current_frame])
        is_playing = was_playing
        print("\nPNG sequence export done")
        return False

    # --- Blender-like quick views and video export ---
    def _data_center_for_views():
        try:
            return np.array([ground_center_x, ground_center_y, (data_z_min + data_z_max) / 2])
        except Exception:
            return np.array([0.0, 0.0, 0.0])

    def view_front(_vis_obj):
        center = _data_center_for_views()
        ctr.set_lookat(center)
        ctr.set_front(np.array([0, 1, 0]))
        ctr.set_up(np.array([0, 0, 1]))
        vis.update_renderer()
        print("\nView: Front")
        return False

    def view_right(_vis_obj):
        center = _data_center_for_views()
        ctr.set_lookat(center)
        ctr.set_front(np.array([1, 0, 0]))
        ctr.set_up(np.array([0, 0, 1]))
        vis.update_renderer()
        print("\nView: Right")
        return False

    def view_top(_vis_obj):
        center = _data_center_for_views()
        ctr.set_lookat(center)
        ctr.set_front(np.array([0, 0, 1]))
        ctr.set_up(np.array([0, 1, 0]))
        vis.update_renderer()
        print("\nView: Top")
        return False

    def export_video_mp4(_vis_obj):
        """Export animation to MP4 using ffmpeg (temp PNGs -> MP4)."""
        nonlocal current_frame, is_playing
        root = _create_centered_tk_root()
        default_name = os.path.splitext(os.path.basename(filepath))[0] + "_viewc3d.mp4"
        out_path = filedialog.asksaveasfilename(
            title="Save MP4 video",
            defaultextension=".mp4",
            initialfile=default_name,
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")]
        )
        root.destroy()
        if not out_path:
            return False
        was_playing = is_playing
        is_playing = False
        start_frame = current_frame
        tmp_dir = tempfile.mkdtemp(prefix="viewc3d_frames_")
        print(f"\nExporting MP4 via ffmpeg → {out_path}")
        try:
            for fidx in range(num_frames):
                current_frame = fidx
                update_spheres(points[current_frame])
                png_path = os.path.join(tmp_dir, f"frame_{fidx:05d}.png")
                vis.capture_screen_image(png_path, do_render=True)
            # Build ffmpeg command
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(max(1, int(fps))),
                "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                out_path
            ]
            try:
                subprocess.run(cmd, check=True)
                print("\nMP4 export finished")
            except FileNotFoundError:
                print("\n[red]ffmpeg not found. Install ffmpeg and ensure it is in PATH.[/red]")
            except subprocess.CalledProcessError as exc:
                print(f"\n[red]ffmpeg failed:[/red] {exc}")
        finally:
            current_frame = start_frame
            update_spheres(points[current_frame])
            is_playing = was_playing
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        return False

    def render_turntable(_vis_obj):
        """Render a simple turntable MP4 rotating the view around the scene center."""
        nonlocal is_playing
        root = _create_centered_tk_root()
        default_name = os.path.splitext(os.path.basename(filepath))[0] + "_turntable.mp4"
        out_path = filedialog.asksaveasfilename(
            title="Save turntable MP4",
            defaultextension=".mp4",
            initialfile=default_name,
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")]
        )
        root.destroy()
        if not out_path:
            return False
        was_playing = is_playing
        is_playing = False
        tmp_dir = tempfile.mkdtemp(prefix="viewc3d_turn_")
        print(f"\nRendering turntable → {out_path}")
        frames = 180
        try:
            for i in range(frames):
                # Small horizontal rotate per frame
                try:
                    ctr.rotate(10, 0)
                except Exception:
                    pass
                vis.update_renderer()
                png_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
                vis.capture_screen_image(png_path, do_render=True)
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                out_path
            ]
            try:
                subprocess.run(cmd, check=True)
                print("\nTurntable export finished")
            except FileNotFoundError:
                print("\n[red]ffmpeg not found. Install ffmpeg and ensure it is in PATH.[/red]")
            except subprocess.CalledProcessError as exc:
                print(f"\n[red]ffmpeg failed:[/red] {exc}")
        finally:
            is_playing = was_playing
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        return False
    
    # Frame control variables and callback definitions
    current_frame = 0
    is_playing = False
    verbose_frame = [False]  # Print per-frame info in console when True
    playback_rates = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    playback_rate_index = [4]  # start at 1.0x

    def toggle_grid(_vis_obj):
        """Toggle ground grid visibility"""
        nonlocal show_grid, grid
        
        if show_grid[0]:
            # Hide grid
            vis.remove_geometry(grid, reset_bounding_box=False)
            print("\nGround grid hidden")
            show_grid[0] = False
        else:
            # Show grid
            vis.add_geometry(grid, reset_bounding_box=False)
            print("\nGround grid visible")
            show_grid[0] = True
        
        return False
    
    def create_text_geometry(text, position, size=0.1, color=[1, 1, 1]):
        """Create a 3D text geometry (simplified using a small cube as placeholder)"""
        # Note: Open3D doesn't have native text rendering in 3D space
        # We'll create a small cube with the label as a visual indicator
        # The cube will be positioned above the marker
        text_cube = o3d.geometry.TriangleMesh.create_box(
            width=size * 0.3, height=size * 0.3, depth=size * 0.3
        )
        # Position above the marker
        offset_pos = position + np.array([0, 0, size * 1.5])
        text_cube.translate(offset_pos)
        text_cube.paint_uniform_color(color)
        return text_cube
    
    def toggle_marker_labels(_vis_obj):
        """Toggle marker labels visibility"""
        nonlocal marker_labels_visible, marker_label_texts
        
        if marker_labels_visible[0]:
            # Hide labels
            for label_geom in marker_label_texts:
                vis.remove_geometry(label_geom, reset_bounding_box=False)
            marker_label_texts.clear()
            print("\nMarker labels hidden")
            marker_labels_visible[0] = False
        else:
            # Show labels - create text geometries for current frame
            frame_data = points[current_frame]
            for i, (marker_name, marker_pos) in enumerate(zip(selected_marker_names, frame_data)):
                if not np.isnan(marker_pos).any():
                    # Choose color based on marker name for better distinction
                    first_char = marker_name.lower()[0] if marker_name else 'z'
                    if first_char in 'abc':
                        label_color = [1, 0, 0]  # Red
                    elif first_char in 'def':
                        label_color = [0, 1, 0]  # Green
                    elif first_char in 'ghi':
                        label_color = [0, 0, 1]  # Blue
                    elif first_char in 'jklm':
                        label_color = [1, 1, 0]  # Yellow
                    elif first_char in 'nopq':
                        label_color = [1, 0, 1]  # Magenta
                    elif first_char in 'rst':
                        label_color = [0, 1, 1]  # Cyan
                    else:
                        label_color = [1, 1, 1]  # White
                    
                    label_geom = create_text_geometry(marker_name, marker_pos, 
                                                    size=current_radius * 3, 
                                                    color=label_color)
                    vis.add_geometry(label_geom, reset_bounding_box=False)
                    marker_label_texts.append(label_geom)
            
            print(f"\nMarker labels visible ({len(marker_label_texts)} labels)")
            marker_labels_visible[0] = True
        
        return False
    
    def update_marker_labels():
        """Update marker label positions when frame changes"""
        if marker_labels_visible[0] and marker_label_texts:
            # Remove old labels
            for label_geom in marker_label_texts:
                vis.remove_geometry(label_geom, reset_bounding_box=False)
            marker_label_texts.clear()
            
            # Add new labels for current frame
            frame_data = points[current_frame]
            for i, (marker_name, marker_pos) in enumerate(zip(selected_marker_names, frame_data)):
                if not np.isnan(marker_pos).any():
                    # Choose color based on marker name for better distinction
                    first_char = marker_name.lower()[0] if marker_name else 'z'
                    if first_char in 'abc':
                        label_color = [1, 0, 0]  # Red
                    elif first_char in 'def':
                        label_color = [0, 1, 0]  # Green
                    elif first_char in 'ghi':
                        label_color = [0, 0, 1]  # Blue
                    elif first_char in 'jklm':
                        label_color = [1, 1, 0]  # Yellow
                    elif first_char in 'nopq':
                        label_color = [1, 0, 1]  # Magenta
                    elif first_char in 'rst':
                        label_color = [0, 1, 1]  # Cyan
                    else:
                        label_color = [1, 1, 1]  # White
                    
                    label_geom = create_text_geometry(marker_name, marker_pos, 
                                                    size=current_radius * 3, 
                                                    color=label_color)
                    vis.add_geometry(label_geom, reset_bounding_box=False)
                    marker_label_texts.append(label_geom)

    def toggle_field_lines(_vis_obj):
        """Toggle field lines or load new field lines from CSV"""
        nonlocal current_field_lines, default_field_geometries
        
        if show_field_lines[0]:
            # If showing, hide them
            for geom in current_field_lines:
                vis.remove_geometry(geom, reset_bounding_box=False)
            print("\nField lines hidden")
            show_field_lines[0] = False
        else:
            # If hidden, ask to show or load new
            root = _create_centered_tk_root()
            choice = messagebox.askyesnocancel(
                "Field Lines", 
                "Show current lines or load new CSV file?\n\n"
                "Yes = Show current lines\n"
                "No = Load new CSV file with auto-adjust\n"
                "Cancel = Keep hidden"
            )
            root.destroy()
            
            if choice is True:
                # Show current lines
                for geom in current_field_lines:
                    vis.add_geometry(geom, reset_bounding_box=False)
                print("\nField lines visible")
                show_field_lines[0] = True
                
            elif choice is False:
                # Load new CSV file with auto-adjust
                result = load_field_lines_from_csv()
                
                if result is not None:
                    # Remove default ground/grid if they exist
                    if default_field_geometries:
                        print("\nRemoving default ground plane and grid.")
                        for geom in default_field_geometries:
                            try:
                                vis.remove_geometry(geom, reset_bounding_box=False)
                            except Exception:
                                pass
                        default_field_geometries = [] # Clear so we don't remove again

                    # Remove old field lines
                    for geom in current_field_lines:
                        try:
                            vis.remove_geometry(geom, reset_bounding_box=False)
                        except Exception:
                            pass
                    
                    if isinstance(result, pd.DataFrame):
                        # --- Full Soccer Field from CSV ---
                        df = result
                        current_field_lines = create_full_soccer_field(vis, df)

                        # Auto-adjust camera
                        points_array = df[['x', 'y', 'z']].to_numpy()
                        x_min, x_max = points_array[:, 0].min(), points_array[:, 0].max()
                        y_min, y_max = points_array[:, 1].min(), points_array[:, 1].max()
                        z_min, z_max = points_array[:, 2].min(), points_array[:, 2].max()
                        center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
                        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
                        optimal_zoom = 0.3 / max_range if max_range > 0 else 0.003
                        optimal_zoom = max(0.001, min(optimal_zoom, 0.1))

                        ctr.set_lookat(center)
                        ctr.set_front([0, -1, -0.3])
                        ctr.set_up([0, 0, 1])
                        ctr.set_zoom(optimal_zoom)
                        render_option.line_width = 10.0
                        print("\nFull soccer field loaded with auto-adjusted view")
                        show_field_lines[0] = True
                        
                    elif isinstance(result, tuple) and len(result) == 3:
                        # --- Simple Lines from CSV ---
                        new_lines, center, optimal_zoom = result
                        current_field_lines = [new_lines]
                        vis.add_geometry(new_lines, reset_bounding_box=False)

                        # Adjust camera automatically
                        ctr.set_lookat([center[0], center[1], center[2]])
                        ctr.set_front([0, -1, -0.3])
                        ctr.set_up([0, 0, 1])
                        ctr.set_zoom(optimal_zoom)
                        render_option.line_width = 10.0
                        print("\nNew field lines loaded with auto-adjusted view")
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
            if np.isnan(current_pos).any():
                current_pos = np.zeros(3)
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
            # Skip update if the new position is invalid to avoid corrupting geometry
            if np.isnan(new_pos).any():
                continue
            new_vertices = spheres_bases[i] + new_pos
            sphere.vertices = o3d.utility.Vector3dVector(new_vertices)
            vis.update_geometry(sphere)
        
        # Update marker labels if they are visible
        update_marker_labels()
        
        # Update trails if enabled
        if trails_enabled[0]:
            for i in range(num_markers):
                _update_single_trail(i, frame_data[i])

        # Update dynamic skeleton
        _update_skeleton_lines(current_frame)

        # Update measurement line
        _update_measurement_line(current_frame)
        
        # Optional per-frame console info
        if verbose_frame[0]:
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

    def toggle_play(_vis_obj=None):
        nonlocal is_playing
        is_playing = not is_playing
        print(f"\nPlay: {'ON' if is_playing else 'PAUSE'}")
        return False

    def faster_playback(_vis_obj):
        # Increase playback rate
        if playback_rate_index[0] < len(playback_rates) - 1:
            playback_rate_index[0] += 1
        print(f"\nPlayback rate: {playback_rates[playback_rate_index[0]]}x")
        return False

    def slower_playback(_vis_obj):
        # Decrease playback rate
        if playback_rate_index[0] > 0:
            playback_rate_index[0] -= 1
        print(f"\nPlayback rate: {playback_rates[playback_rate_index[0]]}x")
        return False

    def toggle_verbose(_vis_obj):
        verbose_frame[0] = not verbose_frame[0]
        print(f"\nPer-frame console info: {'ON' if verbose_frame[0] else 'OFF'}")
        return False

    # Add after the other callback functions
    def set_view_limits(vis_obj):
        """Allow setting custom limits for visualization and regenerate the ground/grid."""
        nonlocal default_field_geometries
        
        root = _create_centered_tk_root()
        
        try:
            # Get current data bounds to use as default values
            all_points_data = points.reshape(-1, 3)
            valid_points = all_points_data[~np.isnan(all_points_data).any(axis=1)]
            
            if len(valid_points) > 0:
                current_x_min, current_x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
                current_y_min, current_y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
                current_z_min, current_z_max = valid_points[:, 2].min(), valid_points[:, 2].max()
            else:
                current_x_min, current_x_max, current_y_min, current_y_max, current_z_min, current_z_max = 0, 0, 0, 0, 0, 0
            
            print("\nCurrent data bounds:")
            print(f"X: [{current_x_min:.3f}, {current_x_max:.3f}]")
            print(f"Y: [{current_y_min:.3f}, {current_y_max:.3f}]")
            print(f"Z: [{current_z_min:.3f}, {current_z_max:.3f}]")
            
            # Request new limits from user
            x_min = simpledialog.askfloat("X Axis", "X minimum:", initialvalue=current_x_min - 2.0)
            if x_min is None:
                return False
            x_max = simpledialog.askfloat("X Axis", "X maximum:", initialvalue=current_x_max + 2.0)
            if x_max is None:
                return False
            y_min = simpledialog.askfloat("Y Axis", "Y minimum:", initialvalue=current_y_min - 2.0)
            if y_min is None:
                return False
            y_max = simpledialog.askfloat("Y Axis", "Y maximum:", initialvalue=current_y_max + 2.0)
            if y_max is None:
                return False
            z_min = simpledialog.askfloat("Z Axis", "Z minimum:", initialvalue=current_z_min - 2.0)
            if z_min is None:
                return False
            z_max = simpledialog.askfloat("Z Axis", "Z maximum:", initialvalue=current_z_max + 2.0)
            if z_max is None:
                return False
            
        finally:
            root.destroy()

        # Remove old default geometries
        for geom in default_field_geometries:
            try:
                vis.remove_geometry(geom, reset_bounding_box=False)
            except Exception:
                pass
        default_field_geometries.clear()

        # Create new ground, grid, and markers based on user limits
        new_width, new_height = x_max - x_min, y_max - y_min
        x_range, y_range, z_range = new_width, new_height, z_max - z_min
        max_range = max(x_range, y_range, z_range, 1) # Avoid division by zero
        
        if new_width > 0 and new_height > 0:
            new_ground = create_ground_plane(width=new_width, height=new_height)
            new_ground.translate(np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0]))
            spacing = max(1.0, int(max(new_width, new_height) / 20))
            new_grid = create_ground_grid(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, spacing=spacing)
            
            vis.add_geometry(new_ground, reset_bounding_box=False)
            vis.add_geometry(new_grid, reset_bounding_box=False)
            default_field_geometries.extend([new_ground, new_grid])

        # Add corner markers for the new bounds
        new_corners = [np.array([x_min, y_min, z_min]), np.array([x_max, y_min, z_min]),
                       np.array([x_max, y_max, z_min]), np.array([x_min, y_max, z_min])]
        for corner in new_corners:
            x_marker = create_x_marker(corner, size=max_range * 0.02)
            vis.add_geometry(x_marker, reset_bounding_box=False)
            default_field_geometries.append(x_marker)

        # Apply new camera view
        new_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
        optimal_zoom = 0.5 / max_range if max_range > 0 else 0.5
        ctr.set_lookat(new_center)
        ctr.set_zoom(optimal_zoom)
        
        print("\nNew view limits set:")
        print(f"X: [{x_min:.3f}, {x_max:.3f}] | Y: [{y_min:.3f}, {y_max:.3f}] | Z: [{z_min:.3f}, {z_max:.3f}]")
        print(f"Center: [{new_center[0]:.3f}, {new_center[1]:.3f}, {new_center[2]:.3f}]")
        
        return False


    # Function to reset the camera
    def reset_camera_view(_vis_obj):
        """Reset camera to default view based on data bounds"""
        # Calculate data center and optimal zoom
        all_points_data = points.reshape(-1, 3)
        valid_data_points = all_points_data[~np.isnan(all_points_data).any(axis=1)]
        
        if len(valid_data_points) > 0:
            data_center = np.mean(valid_data_points, axis=0)
            data_range = np.max(valid_data_points, axis=0) - np.min(valid_data_points, axis=0)
            max_dimension = np.max(data_range)
            
            # Calculate optimal zoom based on data scale
            if max_dimension > 100:  # Very large scale (soccer field)
                optimal_zoom = 0.003
            elif max_dimension > 20:  # Large scale
                optimal_zoom = 0.02
            elif max_dimension > 5:   # Medium scale
                optimal_zoom = 0.1
            else:                     # Small scale
                optimal_zoom = 0.3
        else:
            data_center = np.array([0, 0, 0])
            optimal_zoom = 0.1
        
        ctr.set_lookat(data_center)
        ctr.set_front(np.array([0, -1, -0.5]))  # Diagonal view
        ctr.set_up(np.array([0, 0, 1]))
        ctr.set_zoom(optimal_zoom)
        print(f"\nCamera view reset to data center: {data_center}")
        return False

    # Add the show_help function and register the shortcut
    def show_help(_vis_obj):
        """Open help in the default browser (non-blocking, scrollable)."""
        # Prefer existing docs if available
        candidate_paths = [
            Path(__file__).parent / "help" / "view3d_eng_help.html",
            Path(__file__).parent / "help" / "view3d_help.html"
        ]
        for p in candidate_paths:
            if p.exists():
                webbrowser.open(p.as_uri())
                print(f"\nOpened help: {p}")
                return False
        # Fallback: generate a temporary HTML
        html = """
        <html><head><meta charset='utf-8'><title>C3D Viewer Help</title></head>
        <body style='font-family: monospace; white-space: pre-wrap;'>
        <h3>C3D VIEWER - KEYBOARD SHORTCUTS</h3>
        NAVIGATION
← → - Previous/Next frame
↑ ↓ - Forward/Backward 60 frames
        F/B - Previous/Next frame
S/E - Jump to start/end
        Space or Enter - Play/Pause animation

        MARKERS
        +/-= - Increase marker size
- - Decrease marker size
C - Change marker color
X - Toggle marker labels (names)
        W - Toggle trails with velocity coloring
        J - Load skeleton connections from JSON
        D - Measure distance between two markers

        VIEW
T - Change background color
Y - Change ground plane color
G - Toggle football field lines
M - Toggle ground grid
R - Reset camera view
L - Set view limits

        DATA
U - Override unit conversion (mm/m)

        INFO
I - Show frame info
O - Show camera parameters
        H - Show this help

        CAPTURE
        K - Save screenshot (PNG)
        Z - Export PNG sequence
        V - Export MP4 (requires ffmpeg)
        9 - Render turntable MP4 (requires ffmpeg)

        NUMPAD-LIKE VIEWS
        1 - Front, 3 - Right, 7 - Top

        PLAYBACK SPEED
        [ - Slower, ] - Faster (rates: 0.1×, 0.2×, 0.25×, 0.5×, 1×, 2×, 4×, 8×)
        Q - Toggle per-frame console info
        </body></html>
        """
        tmp = Path(tempfile.gettempdir()) / "viewc3d_help.html"
        tmp.write_text(html, encoding="utf-8")
        webbrowser.open(tmp.as_uri())
        print(f"\nOpened help: {tmp}")
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
            grid.paint_uniform_color([0.8, 0.8, 0.8])  # Light grid
        else:  # Light background
            grid.paint_uniform_color([0.2, 0.2, 0.2])  # Dark grid
            
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
    # Left/right arrows for previous/next frame
    vis.register_key_callback(262, next_frame)      # Right arrow (→)
    vis.register_key_callback(263, previous_frame)  # Left arrow (←)

    # Arrows up/down for +60/-60 frames
    vis.register_key_callback(264, backward_60_frames)  # Down arrow (↓) - go back 60
    vis.register_key_callback(265, forward_60_frames)   # Up arrow (↑) - go forward 60

    # Keep N/P and F/B as alternatives
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), previous_frame)
    vis.register_key_callback(ord("F"), forward_60_frames)
    vis.register_key_callback(ord("B"), backward_60_frames)

    # Other shortcuts
    vis.register_key_callback(ord(" "), toggle_play)  # Space
    vis.register_key_callback(257, toggle_play)       # Enter/Return
    vis.register_key_callback(ord("O"), lambda _vis_obj: print(ctr.convert_to_pinhole_camera_parameters().extrinsic))
    vis.register_key_callback(ord("+"), increase_radius)
    vis.register_key_callback(ord("="), increase_radius)
    vis.register_key_callback(ord("-"), decrease_radius)
    vis.register_key_callback(ord("C"), cycle_color)
    vis.register_key_callback(ord("S"), jump_to_start)
    vis.register_key_callback(ord("E"), jump_to_end)
    
    # New shortcuts for colors and features
    vis.register_key_callback(ord("T"), toggle_background_advanced)  # Background colored
    vis.register_key_callback(ord("Y"), change_ground_color)         # Ground plane colored
    vis.register_key_callback(ord("L"), set_view_limits)
    vis.register_key_callback(ord("I"), show_frame_info)
    vis.register_key_callback(ord("H"), show_help)
    vis.register_key_callback(ord("G"), toggle_field_lines)  # Field lines
    vis.register_key_callback(ord("M"), toggle_grid)         # Grid visibility
    vis.register_key_callback(ord("X"), toggle_marker_labels)  # Marker labels
    # New advanced features
    vis.register_key_callback(ord("W"), toggle_trails)                     # Trails
    vis.register_key_callback(ord("J"), load_skeleton_from_json)           # Skeleton JSON
    vis.register_key_callback(ord("D"), measure_distance_between_two_markers)  # Distance
    vis.register_key_callback(ord("K"), save_screenshot)                   # Screenshot
    vis.register_key_callback(ord("Z"), export_png_sequence)               # PNG sequence
    # Blender-like views and video
    vis.register_key_callback(ord("1"), view_front)   # Front
    vis.register_key_callback(ord("3"), view_right)   # Right
    vis.register_key_callback(ord("7"), view_top)     # Top
    vis.register_key_callback(ord("V"), export_video_mp4)  # MP4 export
    vis.register_key_callback(ord("9"), render_turntable)  # Turntable render (moved from R)
    vis.register_key_callback(ord("R"), reset_camera_view) # Reset camera view
    # Playback speed and verbose toggle
    vis.register_key_callback(ord("]"), faster_playback)
    vis.register_key_callback(ord("["), slower_playback)
    vis.register_key_callback(ord("Q"), toggle_verbose)
    # Quit (ESC)
    vis.register_key_callback(256, lambda _v: (vis.destroy_window(), False)[1])
    
    # Add unit conversion override key
    def override_units(_vis_obj):
        """Allow user to manually override unit conversion"""
        nonlocal points, spheres, spheres_bases
        
        user_choice = ask_user_units_c3d()
        
        if user_choice == 'mm':
            # Interpret current data as millimeters and convert to meters
            print("[yellow]Applying unit override: interpreting current data as millimeters → converting to meters[/yellow]")
            points = points * 0.001
        elif user_choice == 'm':
            # Interpret current data as meters; no scaling needed
            print("[yellow]Applying unit override: interpreting current data as meters (no scaling)[/yellow]")
        else:
            # Keep as is if user kept auto
            print("[yellow]Unit override cancelled or auto-selected; keeping current scaling[/yellow]")
            return False
        
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

        # Left goal area (small area)
        geometries.append(draw_line_3d(vis, p['left_goal_area_bottom_left'], p['left_goal_area_top_left']))
        geometries.append(draw_line_3d(vis, p['left_goal_area_top_left'], p['left_goal_area_top_right']))
        geometries.append(draw_line_3d(vis, p['left_goal_area_top_right'], p['left_goal_area_bottom_right']))
        geometries.append(draw_line_3d(vis, p['left_goal_area_bottom_right'], p['left_goal_area_bottom_left']))

        # Right goal area (small area)
        geometries.append(draw_line_3d(vis, p['right_goal_area_bottom_left'], p['right_goal_area_top_left']))
        geometries.append(draw_line_3d(vis, p['right_goal_area_top_left'], p['right_goal_area_top_right']))
        geometries.append(draw_line_3d(vis, p['right_goal_area_top_right'], p['right_goal_area_bottom_right']))
        geometries.append(draw_line_3d(vis, p['right_goal_area_bottom_right'], p['right_goal_area_bottom_left']))

        # Penalty Arcs
        # Left
        center_l = p['left_penalty_spot']
        p1 = np.array(p['left_penalty_arc_left_intersection'])
        p2 = np.array(p['left_penalty_arc_right_intersection'])
        radius_l = np.linalg.norm(p1 - center_l)
        angle1_l = np.rad2deg(np.arctan2(p1[1]-center_l[1], p1[0]-center_l[0]))
        angle2_l = np.rad2deg(np.arctan2(p2[1]-center_l[1], p2[0]-center_l[0]))
        
        # For the left penalty arc, we need to draw the OUTER arc (towards center field, not towards goal)
        # Left penalty area center is at x=11, penalty area edge is at x=16.5
        # Current angles: angle1_l ≈ -53° (left intersection), angle2_l ≈ +53° (right intersection)
        # The direct path from angle1_l to angle2_l goes through 0° (towards center - CORRECT)
        geometries.append(draw_arc_3d(vis, center_l, [0,0,1], radius_l, angle1_l, angle2_l))

        # Right
        center_r = p['right_penalty_spot']
        p1_r = np.array(p['right_penalty_arc_left_intersection'])
        p2_r = np.array(p['right_penalty_arc_right_intersection'])
        radius_r = np.linalg.norm(p1_r - center_r)
        angle1_r = np.rad2deg(np.arctan2(p1_r[1]-center_r[1], p1_r[0]-center_r[0]))
        angle2_r = np.rad2deg(np.arctan2(p2_r[1]-center_r[1], p2_r[0]-center_r[0]))
        
        # For the right penalty arc, we need to draw the OUTER arc (towards center field, not towards goal)
        # Right penalty area center is at x=94, penalty area edge is at x=88.5
        # Current angles: angle1_r ≈ -127° (left intersection), angle2_r ≈ +127° (right intersection)
        # The direct path from angle1_r to angle2_r goes through 0° (towards goal - WRONG)
        # We need the complementary arc that goes through 180° (towards center field - CORRECT)
        geometries.append(draw_arc_3d(vis, center_r, [0,0,1], radius_r, angle2_r, angle1_r + 360))

        return geometries

    # --- End of Field Drawing Logic ---

    # Update window title
    window_title = (
        f"C3D Viewer | File: {file_name} | Markers: {num_markers}/{total_markers} | Frames: {num_frames} | FPS: {fps} | "
        "Keys: [←→: Frame, ↑↓: 60 Frames, C: Color, X: Labels, M: Grid, G: Field, H: Help] | "
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
        try:
            vis.update_renderer()
        except Exception:
            pass
    vis.destroy_window()


def load_field_lines_from_csv():
    """Load field lines from a CSV file"""
    root = _create_centered_tk_root()
    csv_file = filedialog.askopenfilename(
        title="Select field lines CSV file",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
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

            # Adjust Z coordinates to ground level if they are close to zero
            lines_points_array = np.array(lines_points)
            if np.max(np.abs(lines_points_array[:, 2])) < 0.1:  # If Z values are very small
                lines_points_array[:, 2] += 0.001  # Raise slightly above ground
                lines_points = lines_points_array.tolist()
            
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
            
            # Calculate center and scale
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            center_z = (z_min + z_max) / 2
            
            # Calculate maximum range to determine appropriate zoom
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            max_range = max(x_range, y_range, z_range)
            
            # Calculate zoom based on data size
            # For a football field (105m), zoom ~0.003 works well
            optimal_zoom = 0.3 / max_range if max_range > 0 else 0.003
            optimal_zoom = max(0.001, min(optimal_zoom, 0.1))  # Limitar entre 0.001 e 0.1
            
            print(f"Loaded {len(lines_points)} points and {len(lines_indices)} lines from {os.path.basename(csv_file)}")
            print(f"Data bounds: X[{x_min:.1f}, {x_max:.1f}] Y[{y_min:.1f}, {y_max:.1f}] Z[{z_min:.1f}, {z_max:.1f}]")
            print(f"Center: [{center_x:.1f}, {center_y:.1f}, {center_z:.1f}], Optimal zoom: {optimal_zoom:.4f}")
            
            return field_lines, (center_x, center_y, center_z), optimal_zoom
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load CSV file:\n{str(e)}")
        return None

def create_football_field_lines(ground_level=0):
    """Creates basic football field markings at the specified ground level"""
    lines_points = []
    lines_indices = []
    
    # Official field: 105m x 68m
    field_length = 105.0
    field_width = 68.0
    
    # Only the basic field lines
    # Outer rectangle
    corners = [
        [-field_length/2, -field_width/2, ground_level + 0.001],  # Slightly above ground
        [field_length/2, -field_width/2, ground_level + 0.001],
        [field_length/2, field_width/2, ground_level + 0.001],
        [-field_length/2, field_width/2, ground_level + 0.001]
    ]
    
    for i in range(len(corners)):
        lines_points.append(corners[i])
        lines_indices.append([i, (i + 1) % len(corners)])
    
    # Middle line
    lines_points.extend([[0, -field_width/2, ground_level + 0.001], 
                        [0, field_width/2, ground_level + 0.001]])
    lines_indices.append([len(lines_points)-2, len(lines_points)-1])
    
    # Create LineSet
    field_lines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(lines_points),
        lines=o3d.utility.Vector2iVector(lines_indices)
    )
    field_lines.paint_uniform_color([1.0, 1.0, 1.0])
    return field_lines


if __name__ == "__main__":
    run_viewc3d()
