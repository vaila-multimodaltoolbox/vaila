"""
================================================================================
Pixel Coordinate Tool - getpixelvideo.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 15 August 2024
Version: 0.0.9
Python Version: 3.11.9

Description:
------------
This tool enables marking and saving pixel coordinates in video frames, with zoom 
functionality for more precise annotations. Users can interact with the video, marking points 
on each frame and loading pre-marked points from a CSV file.

Key Functionalities:
---------------------
1. Frame Navigation: Move through video frames to mark or adjust points.
2. Zoom: Zoom in and out on the video for precise pixel selection.
3. Point Marking: Left-click to mark a point, right-click to remove the last marked point.
4. CSV Loading: Load pre-existing points from a CSV file for review and adjustment.
5. Batch Processing: Save marked points as CSV, creating a log for each processed video.

Input:
------
- Video Files: Supported formats include .mp4, .avi, .mov, .mkv.
- CSV Files (Optional): CSV files containing pre-marked points for review and adjustment.

Example of Input Format (CSV):
------------------------------
frame, p1_x, p1_y, p2_x, p2_y, ...
0, 12, 34, 56, 78, ...
1, 23, 45, 67, 89, ...
...

Output:
-------
For each processed video, the following outputs are generated:
1. Marked Coordinates (CSV):
    - CSV file containing the pixel coordinates of each marked point for each frame.
    - Saved in the same directory as the input video with a suffix "_getpixel.csv".
2. Log File (TXT):
    - Summary of the video processing session, including the total number of frames processed and the coordinates saved.

Example of Output Files:
------------------------
- video_filename_getpixel.csv: Contains the pixel coordinates for each frame of the video.
- log_info.txt: A summary log with frame count, markers, and session details.

How to Use:
-----------
1. Run the script:
   python3 getpixelvideo.py
2. The GUI will prompt you to:
   - Select a video file to process.
   - Optionally, load a CSV file with pre-marked points.
3. Use the provided controls to navigate frames, mark points, and save the results.

Example of Usage:
-----------------
1. Select a video file in .mp4 format.
2. Optionally load an existing CSV file with pre-marked points.
3. Navigate the video frames and mark new points or adjust existing ones.
4. Save the marked coordinates and close the video.

Controls:
---------
- Space: Play/Pause the video.
- Escape: Close the video and exit the application.
- A / Left Arrow: Move to the previous frame.
- D / Right Arrow: Move to the next frame.
- W / Up Arrow: Select the next point.
- S / Down Arrow: Select the previous point.
- N: Go to the first frame.
- P: Go to the last frame.
- Ctrl+m: Zoom in.
- Ctrl+l: Zoom out.
- Ctrl+h: Reset zoom.
- Left-click: Mark a point.
- Right-click: Remove the last marked point.

Dependencies:
-------------
- Python Standard Libraries: os, tkinter, pandas.
- External Libraries: numpy, OpenCV, matplotlib (Install via pip install numpy opencv-python matplotlib).

New Features in v0.0.9:
-----------------------
- Zoom Functionality: Users can now zoom in/out for precise pixel marking.
- CSV Loading: Load pre-existing marked points for review and adjustments.
- Enhanced GUI Interaction: Improved interface for easier navigation and point marking.

License:
--------
This program is free software: you can redistribute it and/or modify it under the terms of 
the GNU General Public License as published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU General Public License for more details.

Disclaimer:
-----------
This script is provided "as is", without any warranty, express or implied, and is intended 
for academic and research purposes only.

Changelog:
----------
- 2024-08-10: Initial release of the pixel marking tool with basic navigation.
- 2024-08-15: Added zoom functionality and CSV loading support.
================================================================================
"""

import cv2
import os
import pandas as pd
from tkinter import filedialog, Tk, Toplevel, Scale, HORIZONTAL, Button, messagebox
import numpy as np

# Variável global para rastrear se o salvamento foi solicitado
should_save = False


def show_help_message():
    root = Tk()
    root.withdraw()
    messagebox.showinfo(
        "Help",
        "This tool allows you to mark and save pixel coordinates in a video.\n\n"
        "Instructions:\n"
        "- Press 'Space' to toggle play/pause of the video.\n"
        "- Press 'Escape' to close the video and exit the application.\n"
        "- Press 'A' or 'Left Arrow' to go to the previous frame.\n"
        "- Press 'D' or 'Right Arrow' to go to the next frame.\n"
        "- Press 'W' or 'Up Arrow' to move to the next point.\n"
        "- Press 'S' or 'Down Arrow' to move to the previous point.\n"
        "- Press 'N' to go to the first frame.\n"
        "- Press 'P' to go to the last frame.\n"
        "- Press 'Ctrl m' to zoom in on the video.\n"
        "- Press 'Ctrl l' to zoom out on the video.\n"
        "- Press 'Ctrl h' to reset the zoom.\n"
        "- Left-click to mark a point on the video.\n"
        "- Right-click to remove the last marked point.\n\n"
        "Note: The current marker is not visually highlighted. Use the navigation counter to track your position.\n"
        "To control the video with zoom and player, you must select the control window.\n\n"
        "For more detailed help, click the link below:\n"
        "docs/help.html",
        icon="info",
    )


def get_video_path():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
    )
    return video_path


def load_existing_coordinates(video_path):
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select CSV File with Pre-marked Points",
        filetypes=[("CSV Files", "*.csv")],
    )
    if csv_path:
        df = pd.read_csv(csv_path)

        frame_column = df.columns[0]
        coordinates = {}

        for _, row in df.iterrows():
            frame_num = int(row[frame_column])
            points = []
            for i in range(1, (len(row) - 1) // 2 + 1):
                x = row.get(f"p{i}_x")
                y = row.get(f"p{i}_y")
                if pd.notna(x) and pd.notna(y):
                    points.append((int(x), int(y)))
            coordinates[frame_num] = points
        return coordinates
    return None


def save_coordinates(video_path, coordinates, total_frames):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_file = os.path.join(video_dir, f"{base_name}_getpixel.csv")

    columns = ["frame"] + [f"p{i}_{c}" for i in range(1, 101) for c in ["x", "y"]]
    df = pd.DataFrame(np.nan, index=range(total_frames), columns=columns)
    df["frame"] = df.index

    for frame_num, points in coordinates.items():
        for i, (x, y) in enumerate(points):
            df.at[frame_num, f"p{i+1}_x"] = x
            df.at[frame_num, f"p{i+1}_y"] = y

    last_point = 0
    for frame_num, points in coordinates.items():
        if points:
            last_point = max(last_point, len(points))

    if last_point < 100:
        df = df.iloc[:, : 1 + 2 * last_point]

    df.to_csv(output_file, index=False)
    print(f"Coordinates saved to {output_file}")


def get_pixel_coordinates(video_path, initial_coordinates=None):
    global should_save
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    coordinates = (
        initial_coordinates
        if initial_coordinates
        else {i: [] for i in range(total_frames)}
    )
    paused = True
    frame = None
    zoom_level = 1.0
    current_point = 0  # Initialize current_point

    def draw_point(frame, x, y, num, is_new=False):
        outer_color = (0, 0, 0)  # Always black for the outer circle
        inner_color = (
            (0, 255, 0) if is_new else (0, 0, 255)
        )  # Green for new points, Blue for loaded points
        outer_radius = 6  # Size of the circle outer
        inner_radius = 4  # Size of the circle inner
        thickness = -1  # Fill the circle

        screen_x = int(x * zoom_level)
        screen_y = int(y * zoom_level)

        cv2.circle(frame, (screen_x, screen_y), outer_radius, outer_color, thickness)
        cv2.circle(frame, (screen_x, screen_y), inner_radius, inner_color, thickness)
        cv2.putText(
            frame,
            f"{num}",
            (screen_x + 10, screen_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            inner_color,
            1,
        )

    def apply_zoom(frame, zoom_level):
        height, width = frame.shape[:2]
        new_width = int(width * zoom_level)
        new_height = int(height * zoom_level)
        frame = cv2.resize(
            frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        if zoom_level > 1.0:
            x_offset = (new_width - width) // 2
            y_offset = (new_height - height) // 2
            frame = frame[y_offset : y_offset + height, x_offset : x_offset + width]

        return frame

    def click_event(event, x, y, flags, param):
        nonlocal frame, current_point
        if event == cv2.EVENT_LBUTTONDOWN:
            original_x = int(x / zoom_level)
            original_y = int(y / zoom_level)
            coordinates[frame_count].append((original_x, original_y))
            draw_point(
                frame,
                original_x,
                original_y,
                len(coordinates[frame_count]),
                is_new=True,
            )
            current_point = (
                len(coordinates[frame_count]) - 1
            )  # Set current point to the latest
            cv2.imshow("Frame", frame)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if coordinates[frame_count]:
                coordinates[frame_count].pop()
                current_point = max(0, len(coordinates[frame_count]) - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if ret:
                    frame = apply_zoom(frame, zoom_level)
                    for i, point in enumerate(coordinates[frame_count]):
                        draw_point(frame, point[0], point[1], i + 1, is_new=False)
                    cv2.imshow("Frame", frame)

    def update_frame(new_frame_count):
        nonlocal frame_count, frame, paused
        frame_count = new_frame_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if ret:
            frame = apply_zoom(frame, zoom_level)
            for i, point in enumerate(coordinates[frame_count]):
                draw_point(frame, point[0], point[1], i + 1, is_new=False)
            cv2.imshow("Frame", frame)
            window.title(
                f"Frame {frame_count} - Point {current_point + 1} of {len(coordinates[frame_count])}"
            )
        paused = True

    def on_key(event):
        nonlocal paused, zoom_level, current_point
        key = event.keysym.lower()
        if key == "space":
            toggle_play_pause()
        elif key == "escape":
            cap.release()
            cv2.destroyAllWindows()
            window.quit()
        elif key in ["a", "left"]:
            update_frame(max(frame_count - 1, 0))
        elif key in ["d", "right"]:
            update_frame(min(frame_count + 1, total_frames - 1))
        elif key in ["w", "up"]:
            if coordinates[frame_count]:
                current_point = (current_point + 1) % len(coordinates[frame_count])
                update_frame(frame_count)
        elif key in ["s", "down"]:
            if coordinates[frame_count]:
                current_point = (current_point - 1) % len(coordinates[frame_count])
                update_frame(frame_count)
        elif key == "n":
            update_frame(0)  # Go to the first frame
        elif key == "p":
            update_frame(total_frames - 1)  # Go to the last frame
        elif event.state == 4:  # Ctrl is pressed
            if key == "m":  # Ctrl m
                zoom_level *= 1.2
                update_frame(frame_count)
            elif key == "l":  # Ctrl l
                zoom_level /= 1.2
                update_frame(frame_count)
            elif key == "h":  # Ctrl h
                zoom_level = 1.0
                update_frame(frame_count)
        slider.set(frame_count)

    def play_video():
        nonlocal frame_count, frame
        if not paused:
            frame_count += 1
            if frame_count >= total_frames:
                frame_count = 0
            ret, frame = cap.read()
            if ret:
                frame = apply_zoom(frame, zoom_level)
                for i, point in enumerate(coordinates[frame_count]):
                    draw_point(frame, point[0], point[1], i + 1, is_new=False)
                cv2.imshow("Frame", frame)
                window.title(
                    f"Frame {frame_count} - Point {current_point + 1} of {len(coordinates[frame_count])}"
                )
                slider.set(frame_count)
            window.after(30, play_video)  # Adjust delay as needed
        else:
            window.after(30, play_video)

    def toggle_play_pause():
        nonlocal paused
        paused = not paused

    def close_video_without_saving():
        global should_save
        should_save = False  # Indica que não deve salvar
        cap.release()
        cv2.destroyAllWindows()
        window.quit()

    def save_and_close_video():
        global should_save
        should_save = True  # Indica que deve salvar
        cap.release()
        cv2.destroyAllWindows()
        window.quit()

    root = Tk()
    root.withdraw()

    window = Toplevel(root)
    window.title("Frame Viewer")
    window.bind("<KeyPress>", on_key)
    window.geometry("800x100")

    slider = Scale(
        window,
        from_=0,
        to=total_frames - 1,
        orient=HORIZONTAL,
        command=lambda pos: update_frame(int(pos)),
    )
    slider.pack(fill="x", expand=True)

    play_pause_button = Button(window, text="Next Frame", command=toggle_play_pause)
    play_pause_button.pack(side="left")

    close_button = Button(
        window, text="Close video", command=close_video_without_saving
    )
    close_button.pack(side="right")

    save_close_button = Button(
        window, text="Save and close video", command=save_and_close_video
    )
    save_close_button.pack(side="right")

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_event)

    update_frame(0)  # Start with the first frame paused

    window.after(30, play_video)
    window.mainloop()

    return coordinates, total_frames


def main():
    show_help_message()
    video_path = get_video_path()
    if video_path:
        root = Tk()
        root.withdraw()
        load_csv = messagebox.askyesno(
            "Load CSV",
            "Do you want to load an existing CSV file with pre-marked points?",
        )
        if load_csv:
            initial_coordinates = load_existing_coordinates(video_path)
            coordinates, total_frames = get_pixel_coordinates(
                video_path, initial_coordinates
            )
        else:
            coordinates, total_frames = get_pixel_coordinates(video_path)

        # Verifica se deve salvar
        if should_save:
            save_coordinates(video_path, coordinates, total_frames)


if __name__ == "__main__":
    main()
