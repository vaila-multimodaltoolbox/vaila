"""
Script: showc3d.py
Author: Prof. Paulo Roberto Pereira Santiago
Date: 29/07/2024
Updated: 07/02/2025

Description:
------------
This script visualizes marker data from a C3D file using Matplotlib.
Marker positions are converted from millimeters to meters.
The user is prompted to select which markers to display.
A Matplotlib 3D scatter plot is used to animate the data,
complete with a slider to choose frames and a play/pause button.
The FPS from the C3D file is used to ensure correct playback speed.

Usage:
------
1. Ensure you have installed:
   - ezc3d (pip install ezc3d)
   - numpy
   - matplotlib (pip install matplotlib)
   - tkinter (usually included with Python)
2. Run the script and select a C3D file and markers to display.
3. Use the slider or Play/Pause button to control the animation.
"""

import os
import numpy as np
import ezc3d
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import Slider, Button


def load_c3d_file():
    """
    Opens a dialog to select a C3D file and loads the marker data,
    the file's frame rate, and marker labels.

    Returns:
        pts: np.ndarray with shape (num_frames, num_markers, 3) – points converted to meters
        filepath: path of the selected file.
        fps: frames per second from the C3D file.
        marker_labels: list of marker labels.
    """
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select a C3D file", filetypes=[("C3D Files", "*.c3d")]
    )
    root.destroy()
    if not filepath:
        print("No file selected. Exiting.")
        exit(0)

    c3d = ezc3d.c3d(filepath)
    fps = c3d["header"]["points"]["frame_rate"]
    pts = c3d["data"]["points"]
    pts = pts[:3, :, :]  # use only x, y, z coordinates
    pts = np.transpose(pts, (2, 1, 0))  # shape becomes (num_frames, num_markers, 3)
    pts = pts * 0.001  # convert from millimeters to meters

    # Extract marker labels (assumed stored in PARAMETERs)
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

    # Add Select All and Unselect All buttons.
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


def draw_cartesian_axes(ax, axis_length=0.25):
    """
    Draws the Cartesian axes on the given Matplotlib 3D axes.
      - X axis: red
      - Y axis: green
      - Z axis: blue
    """
    # X axis (red)
    ax.plot([0, axis_length], [0, 0], [0, 0], color="red", linewidth=2)
    # Y axis (green)
    ax.plot([0, 0], [0, axis_length], [0, 0], color="green", linewidth=2)
    # Z axis (blue)
    ax.plot([0, 0], [0, 0], [0, axis_length], color="blue", linewidth=2)


def main():
    # Load data from the C3D file
    pts, filepath, fps, marker_labels = load_c3d_file()
    num_frames, total_markers, _ = pts.shape

    # Let the user select which markers to display
    selected_indices = select_markers(marker_labels)
    if not selected_indices:
        print("No markers selected, exiting.")
        exit(0)
    pts = pts[:, selected_indices, :]
    num_markers = len(selected_indices)

    file_name = os.path.basename(filepath)

    # Create a figure that fills most of the window and a main 3D axes that occupies nearly the entire figure.
    fig = plt.figure(figsize=(10, 8))
    # Ensure the axes are 3D by setting projection='3d'
    ax = fig.add_axes([0.0, 0.12, 1.0, 0.88], projection="3d")

    # Plot initial marker positions (frame 0) using a scatter plot with smaller blue markers.
    scat = ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0, :, 2], c="blue", s=20)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(
        f"C3D Viewer | File: {file_name} | Markers: {num_markers}/{total_markers} | Frames: {num_frames} | FPS: {fps}\n"
        "Controls: [Slider] | [Play/Pause]"
    )

    # Compute global limits from all frames and markers
    x_min, x_max = pts[:, :, 0].min(), pts[:, :, 0].max()
    y_min, y_max = pts[:, :, 1].min(), pts[:, :, 1].max()
    z_min, z_max = pts[:, :, 2].min(), pts[:, :, 2].max()
    margin_x = 0.1 * (x_max - x_min) if x_max != x_min else 0.5
    margin_y = 0.1 * (y_max - y_min) if y_max != y_min else 0.5
    margin_z = 0.1 * (z_max - z_min) if z_max != z_min else 0.5
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_zlim(z_min - margin_z, z_max + margin_z)

    # Standard Matplotlib rotation and set aspect ratio equal.
    ax.set_aspect("equal")

    # Draw Cartesian axes on the plot
    draw_cartesian_axes(ax, axis_length=0.25)

    # Create a slider for frame selection; note the axes for the slider now lie at the very bottom.
    ax_frame = fig.add_axes([0.25, 0.02, 0.5, 0.04])
    slider_frame = Slider(ax_frame, "Frame", 0, num_frames - 1, valinit=0, valfmt="%d")

    # Global variable for current frame (mutable via a list)
    current_frame = [0]

    def update_frame(val):
        frame = int(slider_frame.val) if isinstance(val, float) else int(val)
        current_frame[0] = frame
        new_positions = pts[frame]
        # Update scatter plot data using internal _offsets3d
        scat._offsets3d = (
            new_positions[:, 0],
            new_positions[:, 1],
            new_positions[:, 2],
        )
        fig.canvas.draw_idle()

    slider_frame.on_changed(update_frame)

    # Use a Matplotlib timer to advance frames
    playing = [False]  # mutable flag for play state
    timer = [None]  # mutable container for the timer instance

    def timer_callback():
        current_frame[0] = (current_frame[0] + 1) % num_frames
        slider_frame.set_val(current_frame[0])
        update_frame(current_frame[0])

    def play_pause(event):
        if not playing[0]:
            playing[0] = True
            btn_play.label.set_text("Pause")
            timer[0] = fig.canvas.new_timer(interval=1000 / fps)
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

    # Add a play/pause button for animation; its axes are defined near the bottom.
    ax_play = fig.add_axes([0.82, 0.02, 0.1, 0.05])
    btn_play = Button(ax_play, "Play")
    btn_play.on_clicked(play_pause)

    plt.show()


def show_c3d():
    main()


if __name__ == "__main__":
    main()
