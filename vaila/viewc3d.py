"""
Script: viewc3d.py
Author: Prof. Dr. Paulo Santiago
Version: 0.0.1
Created: April 5, 2025
Last Updated: April 6, 2025

Description:
------------
This script launches a 3D viewer for C3D files, providing efficient visualization of marker data.
Marker positions are converted from millimeters to meters and displayed using Open3D.
The viewer features:
  - Display of markers, Cartesian coordinate axes, a ground plane with a 1x1 grid, and boundary markers.
  - Customizable camera view parameters (with an initial view rotated by 90°).
  - Keyboard shortcuts for navigation:
       'N': Next frame
       'P': Previous frame
       'F': Advance 10 frames
       'B': Go back 10 frames
       'Space': Toggle automatic playback
       'O': Display current camera parameters
  - A black background with thicker rendering lines for enhanced visibility.

Usage:
------
1. Ensure the required dependencies are installed:
   - open3d (pip install open3d)
   - ezc3d (pip install ezc3d)
   - numpy
   - tkinter (typically included with Python)
2. Run the script.
3. When prompted, choose a C3D file via the file selection dialog.
4. Use the specified keyboard commands to navigate through the frames in the 3D viewer.

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
    Opens a dialog to select a C3D file and loads the marker data along with the file's frame rate.
    
    Returns:
        pts: np.ndarray with shape (num_frames, num_markers, 3) – the points converted (in meters)
        filepath: path of the selected file.
        fps: frames per second (Hz) extracted from the C3D header.
    """
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Select a C3D file",
                                          filetypes=[("C3D Files", "*.c3d")])
    root.destroy()
    if not filepath:
        print("No file was selected. Exiting.")
        exit(0)
    
    c3d = ezc3d.c3d(filepath)
    # Extract the frame rate (fps) from the header
    fps = c3d["header"]["points"]["frame_rate"]
    pts = c3d["data"]["points"]
    pts = pts[:3, :, :]  # use only x, y, z coordinates
    pts = np.transpose(pts, (2, 1, 0))  # shape (num_frames, num_markers, 3)
    pts = pts * 0.001  # convert from millimeters to meters
    return pts, filepath, fps

def create_coordinate_lines(axis_length=0.5):
    """
    Creates lines representing the Cartesian coordinate axes:
      - X axis in red
      - Y axis in green
      - Z axis in blue
    """
    points = np.array([
        [0, 0, 0],                # origin
        [axis_length, 0, 0],       # X axis
        [0, axis_length, 0],       # Y axis
        [0, 0, axis_length]        # Z axis
    ])
    lines = np.array([
        [0, 1],  # line for X
        [0, 2],  # line for Y
        [0, 3]   # line for Z
    ])
    colors = np.array([
        [1, 0, 0],    # red for X
        [0, 1, 0],    # green for Y
        [0, 0, 1]     # blue for Z
    ])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
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
        [ half_w, -half_h, 0],
        [ half_w,  half_h, 0],
        [-half_w,  half_h, 0]
    ]
    triangles = [
        [0, 1, 2],
        [0, 2, 3]
    ]
    ground = o3d.geometry.TriangleMesh()
    ground.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    ground.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    ground.paint_uniform_color([0.2, 0.2, 0.2])  # ground in dark gray
    ground.compute_vertex_normals()
    return ground

def create_x_marker(position, size=0.2):
    """
    Creates an "X" marker on the XY plane to indicate boundaries.
    
    Args:
        position (np.ndarray): (x, y, z) coordinate where the "X" is placed.
        size (float): Size of the lines forming the "X".
    
    Returns:
        A LineSet representing the "X".
    """
    half = size / 2.0
    x, y, z = position
    points = np.array([
        [x - half, y - half, z],
        [x + half, y + half, z],
        [x - half, y + half, z],
        [x + half, y - half, z]
    ])
    lines = np.array([
        [0, 1],
        [2, 3]
    ])
    x_marker = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    x_marker.paint_uniform_color([1, 0, 0])  # red for emphasis
    return x_marker

def create_ground_grid(x_min=-1, x_max=5, y_min=-1, y_max=6, spacing=1.0):
    """
    Creates a grid of lines with a spacing of 1 meter for the ground.
    
    For the ground corners:
      Bottom left: (x_min, y_min, 0)
      Bottom right: (x_max, y_min, 0)
      Top right: (x_max, y_max, 0)
      Top left: (x_min, y_max, 0)
    """
    points = []
    lines = []
    # Horizontal lines (iterating over y)
    for y in np.arange(y_min, y_max + spacing, spacing):
        idx = len(points)
        points.append([x_min, y, 0])
        points.append([x_max, y, 0])
        lines.append([idx, idx + 1])
    # Vertical lines (iterating over x)
    for x in np.arange(x_min, x_max + spacing, spacing):
        idx = len(points)
        points.append([x, y_min, 0])
        points.append([x, y_max, 0])
        lines.append([idx, idx + 1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(points)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    grid.paint_uniform_color([1.0, 1.0, 1.0])  # grid in white for visualization
    return grid

def main():
    # Load data from the C3D file, including the fps information
    points, filepath, fps = load_c3d_file()
    num_frames, num_markers, _ = points.shape

    # Extract file name from full path
    file_name = filepath.split("/")[-1]

    # Build a detailed window title with file info, fps and control instructions
    window_title = (
        f"C3D Viewer | File: {file_name} | Markers: {num_markers} | Frames: {num_frames} | FPS: {fps} | "
        "Keys: [N: Next, P: Prev, F: +10, B: -10, Space: Play/Pause, O: Cam Params] | "
        "Mouse: [Left Drag: Rotate, Middle/Right Drag: Pan, Mouse Wheel: Zoom]"
    )
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=window_title)

    # Create a sphere for each marker (with reduced size) and paint them in orange
    marker_radius = 0.015  # reduced marker size
    spheres = []
    spheres_bases = []
    for i in range(num_markers):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius, resolution=8)
        base_vertices = np.asarray(sphere.vertices).copy()  # base centered at the origin
        initial_pos = points[0][i]
        sphere.vertices = o3d.utility.Vector3dVector(base_vertices + initial_pos)
        sphere.paint_uniform_color([1.0, 0.65, 0.0])  # orange color
        spheres.append(sphere)
        spheres_bases.append(base_vertices)
    
    # Add the markers (spheres) to the scene
    for sphere in spheres:
        vis.add_geometry(sphere)

    # Add Cartesian axes (X, Y, Z)
    axes = create_coordinate_lines(axis_length=0.5)
    vis.add_geometry(axes)

    # Create the ground (black) and add the grid
    ground = create_ground_plane(width=6.0, height=7.0)
    # Translate the ground so that the corners are at (-1,-1), (5,-1), (5,6), (-1,6)
    ground.translate(np.array([2.0, 2.5, 0.0]))
    vis.add_geometry(ground)

    grid = create_ground_grid(x_min=-1, x_max=5, y_min=-1, y_max=6, spacing=1.0)
    vis.add_geometry(grid)

    # Set up view control parameters:
    ctr = vis.get_view_control()
    bbox_center = np.array([2.0, 2.5, 0.0])
    ctr.set_lookat(bbox_center)
    ctr.set_front(np.array([0, -1, 0]))  # rotated view by 90 degrees
    ctr.set_up(np.array([0, 0, 1]))
    ctr.set_zoom(0.6)  # set the camera further away from the center

    # Add "X" markers at the four corners of the ground
    corners = [
        np.array([-1, -1, 0]),
        np.array([5, -1, 0]),
        np.array([5, 6, 0]),
        np.array([-1, 6, 0])
    ]
    for corner in corners:
        x_marker = create_x_marker(corner, size=0.2)
        vis.add_geometry(x_marker)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.line_width = 3.0  # thicker lines for the axes and grid
    render_option.background_color = np.array([0, 0, 0])  # background in black
    render_option.light_on = False

    # Frame control variables
    current_frame = 0
    is_playing = False

    # Helper function to update markers' positions:
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

    # Register key callbacks
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), previous_frame)
    vis.register_key_callback(ord("F"), forward_10_frames)
    vis.register_key_callback(ord("B"), backward_10_frames)
    vis.register_key_callback(ord(" "), toggle_play)
    vis.register_key_callback(ord("O"), lambda vis_obj: print(vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic))

    # Main loop with playback speed determined by the file's fps (1/fps seconds per frame)
    while True:
        if not vis.poll_events():
            break
        if is_playing:
            next_frame(vis)
            time.sleep(1.0 / fps)
    vis.destroy_window()

if __name__ == "__main__":
    main() 