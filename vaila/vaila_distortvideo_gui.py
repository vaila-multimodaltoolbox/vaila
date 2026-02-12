"""
===============================================================================
vaila_distortvideo_gui.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 06 Feb 2026
Version: 0.0.3
Python Version: 3.12.12
===============================================================================

This script processes videos applying lens distortion correction based on
intrinsic camera parameters and distortion coefficients. Parameters are loaded
and saved as TOML (not CSV). It is possible to adjust them interactively
through a graphical interface with sliders and buttons. The first frame of the
video and the result (undistorted image) are displayed in an updated preview in real time.
===============================================================================
"""

import json
import math
import os
import subprocess
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

import cv2
import numpy as np
from PIL import Image, ImageTk  # To convert images for display with Tkinter
from rich import print
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def load_distortion_parameters(toml_path):
    """
    Load distortion parameters from a TOML file (fx, fy, cx, cy, k1, k2, k3, p1, p2).
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    return {
        k: float(v)
        for k, v in data.items()
        if k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")
    }


def process_video(input_path, output_path, parameters):
    """Process video applying lens distortion correction."""
    console = Console()

    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create camera matrix and distortion coefficients
    camera_matrix = np.array(
        [
            [parameters["fx"], 0, parameters["cx"]],
            [0, parameters["fy"], parameters["cy"]],
            [0, 0, 1],
        ]
    )

    dist_coeffs = np.array(
        [
            parameters["k1"],
            parameters["k2"],
            parameters["p1"],
            parameters["p2"],
            parameters["k3"],
        ]
    )

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )

    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Add task
            process_task = progress.add_task("[cyan]Processing frames...", total=total_frames)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Undistort frame
                undistorted = cv2.undistort(
                    frame, camera_matrix, dist_coeffs, None, new_camera_matrix
                )

                # Save frame como PNG (lossless)
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_path, undistorted)

                frame_count += 1
                progress.update(process_task, advance=1)

                # Display additional information every 100 frames
                if frame_count % 100 == 0:
                    elapsed = progress.tasks[0].elapsed
                    if elapsed:
                        fps_processing = frame_count / elapsed
                        remaining = (total_frames - frame_count) / fps_processing
                        progress.console.print(
                            f"[dim]Processing speed: {fps_processing:.1f} fps | "
                            f"Estimated time remaining: {remaining:.1f}s[/dim]"
                        )

        # Create the final video with FFmpeg
        rprint("\n[yellow]Creating final video with FFmpeg...[/yellow]")
        input_pattern = os.path.join(temp_dir, "frame_%06d.png")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]

        subprocess.run(ffmpeg_cmd, check=True)

    finally:
        # Release the video capture
        cap.release()

        # Remove temporary files
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

    rprint("\n[green]Video processing complete![/green]")
    rprint(f"[blue]Output saved as: {output_path}[/blue]")


def select_directory(title="Select a directory"):
    """
    Open a dialog to select a directory.
    """
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory


def select_file(title="Select a file", filetypes=(("TOML Files", "*.toml"),)):
    """
    Open a dialog to select a file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path


def distort_video_gui():
    """
    GUI to adjust distortion parameters interactively
    using the first frame of a video as an example.

    After adjustment, the parameters (fx,fy,cx,cy,k1,k2,k3,p1,p2) are saved
    in a TOML file for later use.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Create the root window (hidden)
    root = tk.Tk()
    root.withdraw()

    # Select the video and extract the first frame
    video_path = filedialog.askopenfilename(
        title="Select the video to extract the first frame",
        filetypes=(
            (
                "Video Files",
                "*.mp4;*.avi;*.mov;*.mkv;*.webm;*.MP4;*.AVI;*.MOV;*.MKV;*.WEBM",
            ),
            ("All Files", "*.*"),
        ),
    )
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Could not read the video frame.")
        return None

    original_frame = frame.copy()
    orig_height, orig_width = original_frame.shape[:2]

    # Estimate initial parameters using a 90° FOV
    fov = 90
    default_fx = int((orig_width / 2) / math.tan(math.radians(fov / 2)))
    default_fy = default_fx
    default_cx = orig_width // 2
    default_cy = orig_height // 2

    # Create the preview window
    preview_win = tk.Toplevel(root)
    preview_win.title("Preview - Distortion Correction")
    init_width = min(800, orig_width)
    init_height = int(init_width * (orig_height / orig_width))
    preview_win.geometry(f"{init_width}x{init_height}")
    preview_label = tk.Label(preview_win)
    preview_label.pack(expand=True, fill="both")

    # Create the controls window
    control_win = tk.Toplevel(root)
    control_win.title("Controls of Parameters")
    control_win.geometry("350x700")
    controls_frame = tk.Frame(control_win)
    controls_frame.pack(expand=True, fill="both", padx=5, pady=5)

    # Define the variables for the parameters
    fx_var = tk.DoubleVar(value=default_fx)
    fy_var = tk.DoubleVar(value=default_fy)
    cx_var = tk.DoubleVar(value=default_cx)
    cy_var = tk.DoubleVar(value=default_cy)
    k1_var = tk.DoubleVar(value=0.0)
    k2_var = tk.DoubleVar(value=0.0)
    k3_var = tk.DoubleVar(value=0.0)
    p1_var = tk.DoubleVar(value=0.0)
    p2_var = tk.DoubleVar(value=0.0)
    scale_var = tk.DoubleVar(value=1.0)

    def update_preview():
        # Get the current parameters
        fx = fx_var.get()
        fy = fy_var.get()
        cx = cx_var.get()
        cy = cy_var.get()
        k1 = k1_var.get()
        k2 = k2_var.get()
        k3 = k3_var.get()
        p1 = p1_var.get()
        p2 = p2_var.get()

        # Create the camera matrix and distortion coefficients
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        # Calculate the new camera matrix (optional, but useful)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (orig_width, orig_height),
            1,
            (orig_width, orig_height),
        )
        undistorted = cv2.undistort(
            original_frame, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        # Resize the image to fit in the preview window
        scale = scale_var.get()
        preview_win.update_idletasks()
        win_w = preview_win.winfo_width()
        win_h = preview_win.winfo_height()
        new_w = int(win_w * scale)
        new_h = int(win_h * scale)
        resized = cv2.resize(undistorted, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert to RGB and create the image for Tkinter
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(resized_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        preview_label.configure(image=tk_image)
        preview_label.image = tk_image  # Keep reference
        preview_win.after(100, update_preview)

    update_preview()

    # Helper function to create sliders (kept, but without keyboard bindings)
    slider_row = 0

    def add_slider(label_text, var, from_val, to_val, resolution):
        nonlocal slider_row
        # Create a frame to group the label, the slider and the entry field
        frame = tk.Frame(controls_frame)
        frame.grid(row=slider_row, column=0, columnspan=2, sticky="we", padx=2, pady=2)

        # Label of the slider
        lbl = tk.Label(frame, text=label_text)
        lbl.pack(side="left")

        # Slider itself
        slider = tk.Scale(
            frame,
            variable=var,
            from_=from_val,
            to=to_val,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            length=150,
            takefocus=True,
        )
        slider.pack(side="left", fill="x", expand=True)

        # Entry field for manual value input
        entry = tk.Entry(frame, width=8)
        entry.pack(side="left", padx=5)
        entry.insert(0, str(var.get()))

        # Update the entry field when the slider is moved
        def slider_changed(val):
            try:
                fval = float(val)
            except ValueError:
                fval = 0
            # If the resolution is less than 1, use float formatting; otherwise, integer.
            if resolution < 1:
                entry_value = f"{fval:.3f}"
            else:
                entry_value = f"{int(round(fval))}"
            entry.delete(0, tk.END)
            entry.insert(0, entry_value)

        slider.config(command=slider_changed)

        # Update the slider when the user manually inputs the value
        def entry_changed(event):
            try:
                new_val = float(entry.get())
            except ValueError:
                new_val = slider.get()
            # Ensure the value is within the limits
            if new_val < from_val:
                new_val = from_val
            elif new_val > to_val:
                new_val = to_val
            slider.set(new_val)

        entry.bind("<Return>", entry_changed)
        entry.bind("<FocusOut>", entry_changed)

        # Configure the mouse scroll to increment/decrement exactly 1 unit of 'resolution'
        def on_mousewheel(event):
            r = float(slider.cget("resolution"))
            if event.delta:
                # Ignore the absolute value; use only the sign
                step = 1 if event.delta > 0 else -1
                slider.set(slider.get() + step * r)
            elif hasattr(event, "num"):
                if event.num == 4:
                    slider.set(slider.get() + r)
                elif event.num == 5:
                    slider.set(slider.get() - r)
            return "break"

        slider.bind("<MouseWheel>", on_mousewheel)
        slider.bind("<Button-4>", on_mousewheel)
        slider.bind("<Button-5>", on_mousewheel)

        slider_row += 1

    add_slider("fx", fx_var, default_fx * 0.5, default_fx * 1.5, 1)
    add_slider("fy", fy_var, default_fy * 0.5, default_fy * 1.5, 1)
    add_slider("cx", cx_var, 0, orig_width, 1)
    add_slider("cy", cy_var, 0, orig_height, 1)
    add_slider("k1", k1_var, -1.0, 1.0, 0.001)
    add_slider("k2", k2_var, -1.0, 1.0, 0.001)
    add_slider("k3", k3_var, -1.0, 1.0, 0.001)
    add_slider("p1", p1_var, -1.0, 1.0, 0.001)
    add_slider("p2", p2_var, -1.0, 1.0, 0.001)
    add_slider("Scale", scale_var, 0.5, 1.5, 0.001)

    # --- Global keyboard event binding for sliders --- #
    def on_key_global(event):
        focused_widget = control_win.focus_get()
        if isinstance(focused_widget, tk.Scale):
            current_val = focused_widget.get()
            # Get the resolution configured for the slider
            resolution = float(focused_widget.cget("resolution"))
            if event.keysym == "Left":
                focused_widget.set(current_val - resolution)
            elif event.keysym == "Right":
                focused_widget.set(current_val + resolution)

    control_win.bind_all("<KeyPress-Left>", on_key_global)
    control_win.bind_all("<KeyPress-Right>", on_key_global)
    # ----------------------------------------------------------------- #

    # Class to maintain the confirmation state
    class State:
        def __init__(self):
            self.confirmed = False
            self.results = {}

    state = State()

    def confirm():
        state.results = {
            "fx": fx_var.get(),
            "fy": fy_var.get(),
            "cx": cx_var.get(),
            "cy": cy_var.get(),
            "k1": k1_var.get(),
            "k2": k2_var.get(),
            "k3": k3_var.get(),
            "p1": p1_var.get(),
            "p2": p2_var.get(),
        }
        state.confirmed = True
        preview_win.destroy()
        control_win.destroy()
        root.quit()

    def cancel():
        preview_win.destroy()
        control_win.destroy()
        root.quit()

    btn_confirm = tk.Button(controls_frame, text="Confirmar", command=confirm)
    btn_confirm.grid(row=slider_row, column=0, columnspan=2, pady=5)
    slider_row += 1
    btn_cancel = tk.Button(controls_frame, text="Cancelar", command=cancel)
    btn_cancel.grid(row=slider_row, column=0, columnspan=2, pady=5)

    root.mainloop()

    if state.confirmed:
        # Select where to save the parameters file (TOML)
        params_file = filedialog.asksaveasfilename(
            title="Save parameters",
            defaultextension=".toml",
            filetypes=(("TOML Files", "*.toml"), ("All Files", "*.*")),
        )
        if params_file:
            r = state.results
            toml_content = (
                "# Camera distortion parameters\n"
                f"fx = {r['fx']:.6f}\n"
                f"fy = {r['fy']:.6f}\n"
                f"cx = {r['cx']:.6f}\n"
                f"cy = {r['cy']:.6f}\n"
                f"k1 = {r['k1']:.17f}\n"
                f"k2 = {r['k2']:.17f}\n"
                f"k3 = {r['k3']:.17f}\n"
                f"p1 = {r['p1']:.17f}\n"
                f"p2 = {r['p2']:.17f}\n"
            )
            with open(params_file, "w", encoding="utf-8") as f:
                f.write(toml_content)
        return state.results
    else:
        return None


def distort_video_gui_cv2():
    """
    Adjust distortion parameters using an OpenCV-based interface.

    The user selects a video file (to extract the first frame) and can use the OpenCV trackbars
    to modify the correction parameters: [fx, fy, cx, cy, k1, k2, k3, p1, p2] and the scale factor.

    Press 'c' to confirm or 'q' to cancel.

    Returns:
        dict: Confirmed parameters, or None if canceled.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Use Tkinter to select the video file for frame extraction
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select the video for frame extraction",
        filetypes=[
            (
                "Video Files",
                "*.mp4 *.avi *.mov *.mkv *.webm *.MP4 *.AVI *.MOV *.MKV *.WEBM",
            ),
            ("All Files", "*.*"),
        ],
    )
    if not video_path:
        return None
    # Destroy the Tkinter root, freeing the main thread for OpenCV usage
    root.destroy()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Error", "Could not read the video frame.")
        return None
    original_frame = frame.copy()
    orig_height, orig_width = original_frame.shape[:2]
    aspect_ratio = orig_width / orig_height

    # Estimate initial parameters assuming a 90° FOV
    fov = 90
    default_fx = int((orig_width / 2) / math.tan(math.radians(fov / 2)))
    default_fy = default_fx
    default_cx = orig_width // 2
    default_cy = orig_height // 2

    # Create a window for preview and parameter adjustment using OpenCV
    window_name = "Parameter Adjustment (Press 'c' to confirm, 'q' to cancel, 'r' to reset view)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, orig_width, orig_height)

    # Define trackbars with their ranges and default values
    trackbars = {
        "fx": {
            "min": int(default_fx * 0.5),
            "max": int(default_fx * 1.5),
            "default": default_fx,
        },
        "fy": {
            "min": int(default_fy * 0.5),
            "max": int(default_fy * 1.5),
            "default": default_fy,
        },
        "cx": {"min": 0, "max": orig_width, "default": default_cx},
        "cy": {"min": 0, "max": orig_height, "default": default_cy},
        "k1": {"min": -1000, "max": 1000, "default": 0},
        "k2": {"min": -1000, "max": 1000, "default": 0},
        "k3": {"min": -1000, "max": 1000, "default": 0},
        "p1": {"min": -1000, "max": 1000, "default": 0},
        "p2": {"min": -1000, "max": 1000, "default": 0},
        # Scale factor for visualization (0.5 to 1.5), multiplied by 100 for 0.01 resolution
        "scale": {"min": 50, "max": 150, "default": 100},
    }

    def nothing(x):
        pass

    # Create the OpenCV trackbars in the window
    for name, params in trackbars.items():
        cv2.createTrackbar(
            name,
            window_name,
            params["default"] - params["min"],
            params["max"] - params["min"],
            nothing,
        )

    def get_trackbar_value(trackbar_name):
        params = trackbars[trackbar_name]
        pos = cv2.getTrackbarPos(trackbar_name, window_name)
        return params["min"] + pos

    while True:
        # Verify that the window is still open
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return None

        fx = float(get_trackbar_value("fx"))
        fy = float(get_trackbar_value("fy"))
        cx = float(get_trackbar_value("cx"))
        cy = float(get_trackbar_value("cy"))
        k1 = get_trackbar_value("k1") / 1000.0
        k2 = get_trackbar_value("k2") / 1000.0
        k3 = get_trackbar_value("k3") / 1000.0
        p1 = get_trackbar_value("p1") / 1000.0
        p2 = get_trackbar_value("p2") / 1000.0
        scale = get_trackbar_value("scale") / 100.0

        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (orig_width, orig_height),
            1,
            (orig_width, orig_height),
        )
        undistorted = cv2.undistort(
            original_frame, camera_matrix, dist_coeffs, None, new_camera_matrix
        )

        # Get current window size to maintain aspect ratio
        window_rect = cv2.getWindowImageRect(window_name)
        if window_rect[2] > 0 and window_rect[3] > 0:  # Valid window size
            # Use a padding value for controls
            control_padding = 150  # Espaço aproximado ocupado pelos controles

            # Calculate available height for the image (total height minus controls)
            available_height = window_rect[3] - control_padding
            if available_height <= 0:
                available_height = window_rect[3]  # Fallback if window too small

            # Calculate image dimensions based on aspect ratio
            display_height = available_height
            display_width = int(display_height * aspect_ratio)

            # Check if width fits in window
            if display_width > window_rect[2]:
                display_width = window_rect[2]
                display_height = int(display_width / aspect_ratio)

            # Apply scale factor
            display_width = int(display_width * scale)
            display_height = int(display_height * scale)

            # Resize maintaining aspect ratio
            preview = cv2.resize(undistorted, (display_width, display_height))

            # Create a canvas of window size to place the image
            canvas = np.zeros((window_rect[3], window_rect[2], 3), dtype=np.uint8)

            # Calculate position to center the image in the available space
            x_offset = (window_rect[2] - display_width) // 2
            y_offset = (available_height - display_height) // 2

            # Place the image on the canvas
            if y_offset >= 0 and x_offset >= 0:
                try:
                    canvas[
                        y_offset : y_offset + display_height,
                        x_offset : x_offset + display_width,
                    ] = preview
                except ValueError:
                    # Fallback if dimensions don't match
                    preview_resized = cv2.resize(undistorted, (window_rect[2], window_rect[3]))
                    canvas = preview_resized
            else:
                # Fallback if offsets are negative
                preview_resized = cv2.resize(undistorted, (window_rect[2], window_rect[3]))
                canvas = preview_resized

            cv2.imshow(window_name, canvas)
        else:
            # Fallback if window dimensions are not valid
            cv2.imshow(window_name, undistorted)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("c"):
            cv2.destroyAllWindows()
            return {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "p1": p1,
                "p2": p2,
            }
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None
        elif key == ord("r"):
            # Reset window size to original dimensions
            cv2.resizeWindow(window_name, orig_width, orig_height)


def run_distortvideo_gui():
    """Main function to run lens distortion correction using a single video and an OpenCV-based GUI."""
    rprint("[yellow]Running lens distortion correction with OpenCV GUI...[/yellow]")

    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")

    # Extract parameters via the OpenCV-based interface
    parameters = distort_video_gui_cv2()
    if parameters is None:
        rprint("[red]Parameter extraction was canceled.[/red]")
        return

    # Use Tkinter to select the video for processing (can be the same as used for adjustment)
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select the video for processing",
        filetypes=[
            (
                "Video Files",
                "*.mp4 *.avi *.mov *.mkv *.webm *.MP4 *.AVI *.MOV *.MKV *.WEBM",
            ),
            ("All Files", "*.*"),
        ],
    )
    if not video_path:
        rprint("[red]No video was selected for processing.[/red]")
        return
    # Destroy the Tkinter root after the video selection
    root.destroy()

    # Save the parameters to a TOML file in the same directory as the selected video
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    params_toml = os.path.join(os.path.dirname(video_path), f"{base_name}_parameters.toml")
    try:
        p = parameters
        toml_content = (
            "# Camera distortion parameters\n"
            f"fx = {p['fx']:.6f}\n"
            f"fy = {p['fy']:.6f}\n"
            f"cx = {p['cx']:.6f}\n"
            f"cy = {p['cy']:.6f}\n"
            f"k1 = {p['k1']:.17f}\n"
            f"k2 = {p['k2']:.17f}\n"
            f"k3 = {p['k3']:.17f}\n"
            f"p1 = {p['p1']:.17f}\n"
            f"p2 = {p['p2']:.17f}\n"
        )
        with open(params_toml, "w", encoding="utf-8") as f:
            f.write(toml_content)
        rprint(f"\n[blue]Parameters saved to: {params_toml}[/blue]")
        # #region agent log
        try:
            with open(
                "/home/preto/Preto/vaila/.cursor/debug-a5f5a000-975d-4bfc-9676-f9748629bda8.log",
                "a",
            ) as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "a5f5a000-975d-4bfc-9676-f9748629bda8",
                            "id": "distortvideo_gui_save_toml",
                            "timestamp": int(time.time() * 1000),
                            "location": "vaila_distortvideo_gui.run_distortvideo_gui",
                            "message": "Params saved as TOML",
                            "data": {
                                "script": "vaila_distortvideo_gui",
                                "mode": "gui",
                                "params_path": params_toml,
                                "ext": ".toml",
                            },
                            "runId": "distort",
                            "hypothesisId": "C",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
    except Exception as e:
        rprint(f"[red]Error saving parameters: {e}[/red]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        os.path.dirname(video_path), f"{base_name}_undistorted_{timestamp}.mp4"
    )

    try:
        rprint(f"\n[cyan]Processing video: {video_path}[/cyan]")
        process_video(video_path, output_path, parameters)
    except Exception as e:
        rprint(f"[red]Error processing video: {e}[/red]")

    rprint("\n[green]Processing complete![/green]")
    rprint(f"[blue]Output saved as: {output_path}[/blue]")


if __name__ == "__main__":
    run_distortvideo_gui()
