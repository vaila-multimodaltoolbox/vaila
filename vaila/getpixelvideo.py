"""
================================================================================
Pixel Coordinate Tool - getpixelvideo.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 21 November 2024
Version: 2.1.0
Python Version: 3.11

Description:
------------
This tool enables marking and saving pixel coordinates in video frames, with 
zoom functionality for precise annotations. The window can now be resized dynamically, 
and all UI elements adjust accordingly. Users can navigate the video frames, mark 
points, and save results in CSV format.

Key Functionalities:
---------------------
1. Frame Navigation: Navigate through video frames to mark or adjust points.
2. Resizable Window: Adjust the window size, and all elements will scale accordingly.
3. Zoom: Zoom in and out of the video for precise pixel selection.
4. Point Marking: Left-click to mark a point, right-click to remove the last marked point.
5. CSV Export: Save marked points as a CSV file for further processing.

Input:
------
- Video Files: Supported formats include .mp4, .avi, .mov, .mkv.

Example of Input Format (CSV):
------------------------------
frame, p1_x, p1_y, p2_x, p2_y, ...
0, 12, 34, 56, 78, ...
1, 23, 45, 67, 89, ...
...

Output:
-------
1. Marked Coordinates (CSV):
    - A CSV file containing the pixel coordinates of each marked point for each frame.
    - Saved in the same directory as the input video with a "_getpixel.csv" suffix.

Example of Output Files:
------------------------
- video_filename_getpixel.csv: Contains the pixel coordinates for each frame.

Controls:
---------
- Space: Play/Pause the video.
- Escape: Exit the application.
- Left Arrow: Move to the previous frame.
- Right Arrow: Move to the next frame.
- Left-click: Mark a point.
- Right-click: Remove the last marked point.

Dependencies:
-------------
- Python Standard Libraries: os, pandas, numpy.
- External Libraries: pygame, opencv-python (Install via pip install pygame opencv-python).

Changelog:
----------
- 2024-08-10: Initial release of the pixel marking tool.
- 2024-08-15: Added zoom functionality and CSV loading support.
- 2024-11-21: Added dynamic resizing of the window using Pygame.
================================================================================
"""

import os
import pygame
import pandas as pd
import numpy as np
import cv2
from rich import print

# Global variable to track if save is requested
should_save = False

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

    last_point = max(len(points) for points in coordinates.values())
    if last_point < 100:
        df = df.iloc[:, : 1 + 2 * last_point]

    df.to_csv(output_file, index=False)
    print(f"[bold green]Coordinates saved to:[/bold green] {output_file}")


def load_coordinates_from_csv(video_path):
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select CSV File with Pre-marked Points",
        filetypes=[("CSV Files", "*.csv")],
    )
    if not csv_path:
        return None

    df = pd.read_csv(csv_path)
    coordinates = {}

    for _, row in df.iterrows():
        frame = int(row["frame"])
        points = []
        for i in range(1, (len(row) - 1) // 2 + 1):
            x = row.get(f"p{i}_x")
            y = row.get(f"p{i}_y")
            if not pd.isna(x) and not pd.isna(y):
                points.append((x, y))
        coordinates[frame] = points

    print(f"[bold yellow]Loaded coordinates from:[/bold yellow] {csv_path}")
    return coordinates


def play_video(video_path, initial_coordinates=None):
    global should_save
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[bold red]Error opening video.[/bold red]")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height + 50), pygame.RESIZABLE)
    pygame.display.set_caption("Pixel Coordinate Tool - Pygame")
    clock = pygame.time.Clock()

    # Initialize variables
    frame_count = 0
    paused = True
    zoom_level = 1.0
    coordinates = initial_coordinates if initial_coordinates else {i: [] for i in range(total_frames)}

    def draw_points(frame, points, scale_factor):
        for i, (x, y) in enumerate(points):
            pygame.draw.circle(frame, (0, 255, 0), (int(x * scale_factor), int(y * scale_factor)), 5)
            font = pygame.font.Font(None, 24)
            text = font.render(str(i + 1), True, (255, 255, 255))
            frame.blit(text, (int(x * scale_factor) + 10, int(y * scale_factor) - 10))

    def draw_controls():
        control_surface = pygame.Surface((width, 50))
        control_surface.fill((30, 30, 30))

        font = pygame.font.Font(None, 24)
        text = font.render(f"Frame: {frame_count + 1}/{total_frames}", True, (255, 255, 255))
        control_surface.blit(text, (10, 10))

        screen.blit(control_surface, (0, height))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and paused:
                    frame_count = min(frame_count + 1, total_frames - 1)
                elif event.key == pygame.K_LEFT and paused:
                    frame_count = max(frame_count - 1, 0)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    zoom_level *= 1.2
                elif event.key == pygame.K_MINUS:
                    zoom_level /= 1.2
                elif event.key == pygame.K_c:
                    # Load coordinates from CSV
                    loaded_coordinates = load_coordinates_from_csv(video_path)
                    if loaded_coordinates:
                        coordinates.update(loaded_coordinates)
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                scale_factor = min(screen.get_width() / width, (screen.get_height() - 50) / height)
                if y > height:
                    # Click in the control area (can implement buttons here)
                    pass
                else:
                    if event.button == 1:  # Left mouse button
                        coordinates[frame_count].append((x / scale_factor, y / scale_factor))
                    elif event.button == 3:  # Right mouse button
                        if coordinates[frame_count]:
                            coordinates[frame_count].pop()

        if not paused:
            frame_count = (frame_count + 1) % total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        scale_factor = min(screen.get_width() / width, (screen.get_height() - 50) / height)
        scaled_frame = pygame.transform.scale(frame, (int(width * scale_factor), int(height * scale_factor)))
        screen.blit(scaled_frame, (0, 0))
        draw_points(screen, coordinates[frame_count], scale_factor)
        draw_controls()
        pygame.display.flip()
        clock.tick(fps)

    cap.release()
    pygame.quit()

    # Save coordinates on exit
    save_coordinates(video_path, coordinates, total_frames)


def get_video_path():
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
    )
    return video_path


def main():
    # Print script name and directory
    print(f"[bold green]Running script:[/bold green] {os.path.basename(__file__)}")
    print(f"[bold blue]Script directory:[/bold blue] {os.path.dirname(os.path.abspath(__file__))}")

    video_path = get_video_path()
    if video_path:
        initial_coordinates = None  # If you want to implement CSV loading
        play_video(video_path, initial_coordinates)


if __name__ == "__main__":
    main()

