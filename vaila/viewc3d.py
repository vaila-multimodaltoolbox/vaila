"""
Script: viewc3d.py
Author: Prof. Dr. Paulo Santiago
Version: 0.0.2
Created: February 6, 2025
Last Updated: February 7, 2025

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

import open3d as o3d
import ezc3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time


def load_c3d_file():
    """
    Opens a dialog to select a C3D file and loads the marker data along with
    the file's frame rate and marker labels.

    Returns:
        pts: np.ndarray with shape (num_frames, num_markers, 3) – the points converted (in meters)
        filepath: path of the selected file.
        fps: frames per second (Hz).
        marker_labels: list of marker labels.
    """
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select a C3D file", filetypes=[("C3D Files", "*.c3d")]
    )
    root.destroy()
    if not filepath:
        print("No file was selected. Exiting.")
        exit(0)

    c3d = ezc3d.c3d(filepath)
    fps = c3d["header"]["points"]["frame_rate"]
    pts = c3d["data"]["points"]
    pts = pts[:3, :, :]  # use only x, y, z coordinates
    pts = np.transpose(pts, (2, 1, 0))  # shape (num_frames, num_markers, 3)
    pts = pts * 0.001  # convert from millimeters to meters

    # Extract marker labels (assuming they are stored in the C3D parameters)
    marker_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    if isinstance(marker_labels[0], list):
        marker_labels = marker_labels[0]

    return pts, filepath, fps, marker_labels


def select_markers(marker_labels):
    """
    Displays a Tkinter window with a list of marker labels so the user can select which markers to display.

    Args:
        marker_labels (list): list of marker labels.
    Returns:
        List of selected marker indices.
    """
    root = tk.Tk()
    root.title("Select Markers to Display")

    listbox = tk.Listbox(root, selectmode="multiple", width=50, height=15)
    for i, label in enumerate(marker_labels):
        listbox.insert(tk.END, f"{i}: {label}")
    listbox.pack(padx=10, pady=10)

    # Create a frame for extra control buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    # Add the "Select All" and "Unselect All" buttons.
    btn_select_all = tk.Button(button_frame, text="Select All", command=select_all)
    btn_unselect_all = tk.Button(
        button_frame, text="Unselect All", command=unselect_all
    )
    btn_select_all.pack(side=tk.LEFT, padx=5)
    btn_unselect_all.pack(side=tk.LEFT, padx=5)

    def on_select():
        root.quit()

    btn_select = tk.Button(root, text="Select", command=on_select)
    btn_select.pack(pady=(0, 10))

    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    return [int(i) for i in selected_indices]


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


def run_viewc3d():
    # Load data from the C3D file, including FPS and marker labels
    points, filepath, fps, marker_labels = load_c3d_file()
    num_frames, total_markers, _ = points.shape

    # Let user select which markers to display
    selected_indices = select_markers(marker_labels)
    if not selected_indices:
        print("No markers selected, exiting.")
        exit(0)
    # Filter the points array to only the selected markers
    points = points[:, selected_indices, :]
    num_markers = len(selected_indices)

    # Extract file name from full path
    file_name = filepath.split("/")[-1]

    # Build a detailed window title with file info, FPS, and control instructions
    window_title = (
        f"C3D Viewer | File: {file_name} | Markers: {num_markers}/{total_markers} | Frames: {num_frames} | FPS: {fps} | "
        "Keys: [N: Next, P: Prev, F: +10, B: -10, Space: Play/Pause, O: Cam Params] | "
        "Mouse: [Left Drag: Rotate, Middle/Right Drag: Pan, Mouse Wheel: Zoom]"
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_title)

    # Create a sphere for each selected marker and paint them in orange
    marker_radius = 0.015
    spheres = []
    spheres_bases = []
    for i in range(num_markers):
        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=marker_radius, resolution=8
        )
        base_vertices = np.asarray(sphere.vertices).copy()
        initial_pos = points[0][i]
        sphere.vertices = o3d.utility.Vector3dVector(base_vertices + initial_pos)
        sphere.paint_uniform_color([1.0, 0.65, 0.0])
        spheres.append(sphere)
        spheres_bases.append(base_vertices)

    for sphere in spheres:
        vis.add_geometry(sphere)

    # Add Cartesian axes, ground, and grid
    axes = create_coordinate_lines(axis_length=0.25)
    vis.add_geometry(axes)

    ground = create_ground_plane(width=6.0, height=7.0)
    ground.translate(np.array([2.0, 2.5, 0.0]))
    vis.add_geometry(ground)

    grid = create_ground_grid(x_min=-1, x_max=5, y_min=-1, y_max=6, spacing=1.0)
    vis.add_geometry(grid)

    # Set up view control parameters:
    ctr = vis.get_view_control()
    bbox_center = np.array([2.0, 2.5, 0.0])
    ctr.set_lookat(bbox_center)
    ctr.set_front(np.array([0, -1, 0]))
    ctr.set_up(np.array([0, 0, 1]))
    ctr.set_zoom(0.6)

    # Add "X" markers at the four corners of the ground
    corners = [
        np.array([-1, -1, 0]),
        np.array([5, -1, 0]),
        np.array([5, 6, 0]),
        np.array([-1, 6, 0]),
    ]
    for corner in corners:
        x_marker = create_x_marker(corner, size=0.2)
        vis.add_geometry(x_marker)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.line_width = 5.0
    render_option.background_color = np.array([0, 0, 0])
    render_option.light_on = False

    # Frame control variables and callback definitions
    current_frame = 0
    is_playing = False

    def update_spheres(frame_data):
        for i, sphere in enumerate(spheres):
            new_pos = frame_data[i]
            new_vertices = spheres_bases[i] + new_pos
            sphere.vertices = o3d.utility.Vector3dVector(new_vertices)
            vis.update_geometry(sphere)
        new_title = f"C3D Viewer - Frame {current_frame+1}/{num_frames}"
        print(new_title)
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

    def forward_10_frames(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame + 10) % num_frames
        update_spheres(points[current_frame])
        return False

    def backward_10_frames(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame - 10) % num_frames
        update_spheres(points[current_frame])
        return False

    def toggle_play(vis_obj):
        nonlocal is_playing
        is_playing = not is_playing
        return False

    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), previous_frame)
    vis.register_key_callback(ord("F"), forward_10_frames)
    vis.register_key_callback(ord("B"), backward_10_frames)
    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(
        ord("O"),
        lambda vis_obj: print(ctr.convert_to_pinhole_camera_parameters().extrinsic),
    )

    # Main loop for automatic playback (using FPS from file)
    while True:
        if not vis.poll_events():
            break
        if is_playing:
            next_frame(vis)
            time.sleep(1.0 / fps)
    vis.destroy_window()


if __name__ == "__main__":
    run_viewc3d()
