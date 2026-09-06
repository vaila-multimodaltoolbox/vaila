"""
Script: showc3d.py
Author: Prof. Paulo Roberto Pereira Santiago
Date: 29/07/2024
Updated: 06/09/2026
Version: 0.3.122

Description:
------------
This script visualizes marker and calibration keypoint data from C3D files
or direct numpy arrays using Matplotlib in 3D.
Features:
- Automatic unit detection (meters vs millimeters).
- Adapts spatial environment for soccer field / sports court scale
  (pitch boundary lines, halfway line, center circle, ground plane).
- 3D marker text labels with toggle button.
- Frame slider and Play/Pause animation for multi-frame MoCap,
  with streamlined display for single-frame calibration models.
- Standalone CLI execution, GUI invocation, and programmatic show_points_3d().

Usage:
------
1. uv run vaila/showc3d.py
2. uv run vaila/showc3d.py path/to/model.c3d
3. From python:
   from vaila.showc3d import show_c3d, show_points_3d
   show_c3d("soccerfield_kiki.c3d")
"""

from __future__ import annotations

import contextlib
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import TYPE_CHECKING, Any

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

if TYPE_CHECKING:
    pass


def load_c3d_file(
    filepath: str | Path | None = None,
) -> tuple[np.ndarray, str, float, list[str]]:
    """Opens a dialog (or uses provided filepath) to load marker data from a C3D file.

    Returns:
        pts: np.ndarray with shape (num_frames, num_markers, 3) – points in meters.
        filepath: path of the selected file.
        fps: frames per second from the C3D file.
        marker_labels: list of marker labels.
    """
    if filepath is None:
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            title="Select a C3D file",
            filetypes=[("C3D Files", "*.c3d"), ("All Files", "*.*")],
        )
        root.destroy()
        if not filepath:
            print("No file selected. Exiting.")
            sys.exit(0)

    filepath_str = str(filepath)
    if not os.path.isfile(filepath_str):
        raise FileNotFoundError(f"C3D file not found: {filepath_str}")

    c3d = ezc3d.c3d(filepath_str)

    # Frame rate
    try:
        fps = float(c3d["header"]["points"]["frame_rate"])
        if fps <= 0:
            fps = 60.0
    except Exception:
        fps = 60.0

    pts = c3d["data"]["points"]
    pts = pts[:3, :, :]  # use only x, y, z
    pts = np.transpose(pts, (2, 1, 0)).astype(np.float64)  # (num_frames, num_markers, 3)

    # Extract marker labels
    marker_labels: list[str] = []
    try:
        raw_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
        if raw_labels and isinstance(raw_labels[0], list):
            raw_labels = raw_labels[0]
        marker_labels = [str(lbl).strip() for lbl in raw_labels]
    except Exception:
        marker_labels = [f"pt_{i}" for i in range(pts.shape[1])]

    # Detect units: mm vs m
    units = "m"
    try:
        raw_units = c3d["parameters"]["POINT"]["UNITS"]["value"]
        u_str = str(raw_units[0] if isinstance(raw_units, list) else raw_units).strip().lower()
        if "mm" in u_str:
            units = "mm"
        elif u_str in ("m", "meter", "meters"):
            units = "m"
    except Exception:
        pass

    # If units parameter was ambiguous, check coordinate magnitude
    max_val = float(np.nanmax(np.abs(pts))) if pts.size > 0 else 0.0
    if units == "mm" or (units == "m" and max_val > 500.0):
        pts = pts * 0.001  # convert from millimeters to meters
        print(f">> C3D Viewer: Converted coordinates from mm to meters (max was {max_val:.1f} mm)")
    else:
        print(f">> C3D Viewer: Coordinates loaded in meters (max coordinate: {max_val:.2f} m)")

    return pts, filepath_str, fps, marker_labels


def select_markers(
    marker_labels: list[str],
    auto_select_all: bool = False,
) -> list[int]:
    """Displays a Tkinter window with marker labels to select which markers to display.

    Args:
        marker_labels: list of marker labels.
        auto_select_all: if True, skips dialog and selects all markers.

    Returns:
        List of selected marker indices.
    """
    if auto_select_all or len(marker_labels) <= 48:
        # Default to selecting all markers when list is calibration-sized
        return list(range(len(marker_labels)))

    root = tk.Tk()
    root.title("Select Markers to Display")
    root.geometry("400x480")

    lbl_info = tk.Label(root, text=f"Select markers ({len(marker_labels)} available):")
    lbl_info.pack(padx=10, pady=(10, 4))

    listbox = tk.Listbox(root, selectmode="multiple", width=45, height=18)
    for i, label in enumerate(marker_labels):
        listbox.insert(tk.END, f"{i}: {label}")
    listbox.pack(padx=10, pady=4, fill=tk.BOTH, expand=True)

    # Pre-select all
    listbox.select_set(0, tk.END)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=6)

    def select_all():
        listbox.select_set(0, tk.END)

    def unselect_all():
        listbox.selection_clear(0, tk.END)

    btn_select_all = tk.Button(button_frame, text="Select All", command=select_all)
    btn_unselect_all = tk.Button(button_frame, text="Unselect All", command=unselect_all)
    btn_select_all.pack(side=tk.LEFT, padx=5)
    btn_unselect_all.pack(side=tk.LEFT, padx=5)

    def on_select():
        root.quit()

    btn_select = tk.Button(
        root, text="Display Selected", command=on_select, bg="#1976D2", fg="white", padx=12, pady=4
    )
    btn_select.pack(pady=(4, 10))

    root.mainloop()
    selected_indices = listbox.curselection()
    root.destroy()
    return [int(i) for i in selected_indices]


def draw_cartesian_axes(ax: Any, axis_length: float = 0.25) -> None:
    """Draws Cartesian axes on the given Matplotlib 3D axes (X: red, Y: green, Z: blue)."""
    ax.plot([0, axis_length], [0, 0], [0, 0], color="red", linewidth=2.5, label="X")
    ax.plot([0, 0], [0, axis_length], [0, 0], color="green", linewidth=2.5, label="Y")
    ax.plot([0, 0], [0, 0], [0, axis_length], color="blue", linewidth=2.5, label="Z")


def draw_soccer_field_features(
    ax: Any,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_ground: float = 0.0,
) -> None:
    """Draws soccer field pitch outline, center line, and center circle on Z=ground plane."""
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    span_x = x_max - x_min
    span_y = y_max - y_min

    # Outer pitch boundary rectangle (white/green)
    rect_x = [x_min, x_max, x_max, x_min, x_min]
    rect_y = [y_min, y_min, y_max, y_max, y_min]
    rect_z = [z_ground] * 5
    ax.plot(
        rect_x,
        rect_y,
        rect_z,
        color="#2E7D32",
        linewidth=2.0,
        linestyle="-",
        label="Pitch Boundary",
    )

    # Halfway line
    ax.plot(
        [cx, cx],
        [y_min, y_max],
        [z_ground, z_ground],
        color="#388E3C",
        linewidth=1.5,
        linestyle="-",
    )

    # Center circle (approx 9.15m radius, or 10% of field width)
    circle_radius = min(9.15, span_y * 0.15) if span_y > 20.0 else span_y * 0.15
    theta = np.linspace(0, 2 * np.pi, 60)
    circ_x = cx + circle_radius * np.cos(theta)
    circ_y = cy + circle_radius * np.sin(theta)
    circ_z = np.full_like(theta, z_ground)
    ax.plot(circ_x, circ_y, circ_z, color="#388E3C", linewidth=1.2)

    # Center mark
    ax.plot([cx], [cy], [z_ground], "o", color="#2E7D32", markersize=4)

    # Penalty areas if field is standard-sized (~105m x ~68m)
    if span_x > 60.0 and span_y > 40.0:
        box_depth = 16.5
        box_width = 40.3
        y_box_min = cy - box_width / 2.0
        y_box_max = cy + box_width / 2.0

        # Left penalty area
        bx_l = [x_min, x_min + box_depth, x_min + box_depth, x_min]
        by_l = [y_box_min, y_box_min, y_box_max, y_box_max]
        bz_l = [z_ground] * 4
        ax.plot(bx_l, by_l, bz_l, color="#388E3C", linewidth=1.2, alpha=0.7)

        # Right penalty area
        bx_r = [x_max, x_max - box_depth, x_max - box_depth, x_max]
        by_r = [y_box_min, y_box_min, y_box_max, y_box_max]
        bz_r = [z_ground] * 4
        ax.plot(bx_r, by_r, bz_r, color="#388E3C", linewidth=1.2, alpha=0.7)


def show_points_3d(
    points: np.ndarray,
    marker_labels: list[str],
    fps: float = 60.0,
    title: str = "vailá 3D Keypoints Viewer",
    file_name: str = "",
) -> None:
    """Core 3D scatter and animation viewer for marker / calibration data.

    Parameters:
        points: np.ndarray with shape (num_frames, num_markers, 3) in meters.
        marker_labels: list of marker names.
        fps: playback frame rate.
        title: window title / plot header.
        file_name: source file name string.
    """
    num_frames, num_markers, _ = points.shape
    if num_markers == 0:
        print(">> No markers to display.")
        return

    # Compute bounding box and span
    valid_mask = ~np.isnan(points[:, :, 0])
    if np.any(valid_mask):
        valid_x = points[:, :, 0][valid_mask]
        valid_y = points[:, :, 1][valid_mask]
        valid_z = points[:, :, 2][valid_mask]
        x_min, x_max = float(valid_x.min()), float(valid_x.max())
        y_min, y_max = float(valid_y.min()), float(valid_y.max())
        z_min, z_max = float(valid_z.min()), float(valid_z.max())
    else:
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = 0.0, 1.0

    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min
    max_span = max(span_x, span_y, span_z, 1.0)
    is_field_scale = span_x > 20.0 or span_y > 15.0

    # Create figure
    fig = plt.figure(figsize=(11, 8.5))
    bottom_margin = 0.12 if num_frames > 1 else 0.08
    ax: Any = fig.add_axes((0.02, bottom_margin, 0.96, 0.96 - bottom_margin), projection="3d")

    # Plot initial markers (frame 0)
    pt_color = "#1E88E5" if not is_field_scale else "#E65100"
    pt_size = 28 if not is_field_scale else 36
    scat: Any = ax.scatter(
        points[0, :, 0],
        points[0, :, 1],
        points[0, :, 2],
        c=pt_color,
        s=pt_size,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.9,
    )

    # 3D Marker Text Labels
    label_actors: list[Any] = []
    offset_z = max_span * 0.015
    for i in range(num_markers):
        lbl = marker_labels[i] if i < len(marker_labels) else f"pt_{i}"
        txt = ax.text(
            points[0, i, 0],
            points[0, i, 1],
            points[0, i, 2] + offset_z,
            f" {lbl}",
            fontsize=7,
            color="#0D47A1",
            fontweight="bold",
        )
        label_actors.append(txt)

    # Draw soccer field pitch outline if field-scale
    if is_field_scale:
        draw_soccer_field_features(ax, x_min, x_max, y_min, y_max, z_ground=0.0)
        ax.view_init(elev=32, azim=-62)
    else:
        ax.view_init(elev=20, azim=-60)

    # Cartesian axes scaled appropriately
    axis_len = max(0.25, max_span * 0.06)
    draw_cartesian_axes(ax, axis_length=axis_len)

    # Labels and limits
    ax.set_xlabel("X (m)", labelpad=8)
    ax.set_ylabel("Y (m)", labelpad=8)
    ax.set_zlabel("Z (m)", labelpad=8)

    margin_x = max(0.1 * span_x, 1.0)
    margin_y = max(0.1 * span_y, 1.0)
    margin_z = max(0.1 * span_z, 0.5)
    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_zlim(min(z_min - margin_z, -0.5), max(z_max + margin_z, 2.5))

    mode_info = "Soccer Field Environment" if is_field_scale else "Human MoCap Volume"
    frame_info = (
        f"{num_frames} frames @ {fps:.1f} fps"
        if num_frames > 1
        else "Single Frame (Calibration Model)"
    )
    header_text = f"{title}\nFile: {file_name or 'Array'} | Markers: {num_markers} | {frame_info} | {mode_info}"
    ax.set_title(header_text, fontsize=10, pad=12)

    # Label toggle button
    labels_visible = [True]
    ax_toggle_btn = fig.add_axes((0.02, 0.02, 0.15, 0.04))
    btn_labels = Button(ax_toggle_btn, "Labels: ON", color="#E0E0E0", hovercolor="#BDBDBD")

    def toggle_labels(event=None):
        labels_visible[0] = not labels_visible[0]
        for t in label_actors:
            t.set_visible(labels_visible[0])
        btn_labels.label.set_text(f"Labels: {'ON' if labels_visible[0] else 'OFF'}")
        fig.canvas.draw_idle()

    btn_labels.on_clicked(toggle_labels)

    # Multi-frame Animation controls
    if num_frames > 1:
        ax_frame = fig.add_axes((0.22, 0.02, 0.52, 0.04))
        slider_frame = Slider(ax_frame, "Frame", 0, num_frames - 1, valinit=0, valfmt="%d")
        current_frame = [0]

        def update_frame(val):
            frame = int(slider_frame.val) if isinstance(val, float) else int(val)
            current_frame[0] = frame
            new_positions = points[frame]
            scat._offsets3d = (
                new_positions[:, 0],
                new_positions[:, 1],
                new_positions[:, 2],
            )
            for i, t in enumerate(label_actors):
                t.set_position((new_positions[i, 0], new_positions[i, 1]))
                t.set_3d_properties(new_positions[i, 2] + offset_z, "z")
            fig.canvas.draw_idle()

        slider_frame.on_changed(update_frame)

        playing = [False]
        timer = [None]

        def timer_callback():
            current_frame[0] = (current_frame[0] + 1) % num_frames
            slider_frame.set_val(current_frame[0])
            update_frame(current_frame[0])

        def play_pause(event=None):
            if not playing[0]:
                playing[0] = True
                btn_play.label.set_text("Pause")
                timer[0] = fig.canvas.new_timer(interval=max(10, int(1000.0 / fps)))
                with contextlib.suppress(AttributeError):
                    timer[0].single_shot = False
                timer[0].add_callback(timer_callback)
                timer[0].start()
            else:
                playing[0] = False
                btn_play.label.set_text("Play")
                if timer[0] is not None:
                    timer[0].stop()
                    timer[0] = None

        ax_play = fig.add_axes((0.78, 0.02, 0.12, 0.04))
        btn_play = Button(ax_play, "Play", color="#E8F5E9", hovercolor="#C8E6C9")
        btn_play.on_clicked(play_pause)

    plt.show()


def main(filepath: str | Path | None = None, auto_select_all: bool = False) -> None:
    """Main CLI entry point for showc3d."""
    if filepath is None and len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        filepath = sys.argv[1]

    pts, loaded_path, fps, marker_labels = load_c3d_file(filepath)
    file_name = os.path.basename(loaded_path)

    # Let user select markers
    selected_indices = select_markers(marker_labels, auto_select_all=auto_select_all)
    if not selected_indices:
        print("No markers selected. Exiting.")
        return

    pts = pts[:, selected_indices, :]
    selected_labels = [marker_labels[i] for i in selected_indices]

    show_points_3d(
        pts,
        selected_labels,
        fps=fps,
        title="C3D 3D Viewer",
        file_name=file_name,
    )


def show_c3d(filepath: str | Path | None = None) -> None:
    """Convenience alias to launch showc3d."""
    main(filepath)


if __name__ == "__main__":
    main()
