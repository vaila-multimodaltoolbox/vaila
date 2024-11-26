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
import cv2
import pandas as pd
import numpy as np


def play_video_with_controls(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize Pygame
    pygame.init()
    screen_width, screen_height = (
        pygame.display.Info().current_w,
        pygame.display.Info().current_h,
    )
    window_width = min(original_width, screen_width - 100)
    window_height = min(original_height, screen_height - 150)
    screen = pygame.display.set_mode(
        (window_width, window_height + 80), pygame.RESIZABLE
    )
    pygame.display.set_caption("Video Player with Controls")
    clock = pygame.time.Clock()

    # Scaling factors and zoom
    zoom_level = 1.0
    offset_x, offset_y = 0, 0
    frame_count = 0
    paused = True
    coordinates = {i: [] for i in range(total_frames)}

    def draw_controls():
        """
        Draws the slider and frame information on the screen.
        Returns the slider's position and dimensions for handling clicks.
        """
        slider_surface = pygame.Surface((window_width, 80))
        slider_surface.fill((30, 30, 30))

        # Draw slider bar
        slider_width = int(window_width * 0.8)
        slider_x = (window_width - slider_width) // 2
        slider_y = 30
        slider_height = 10
        pygame.draw.rect(
            slider_surface,
            (60, 60, 60),
            (slider_x, slider_y, slider_width, slider_height),
        )

        # Draw slider handle
        slider_pos = slider_x + int((frame_count / total_frames) * slider_width)
        pygame.draw.circle(
            slider_surface,
            (255, 255, 255),
            (slider_pos, slider_y + slider_height // 2),
            8,
        )

        # Draw frame information
        font = pygame.font.Font(None, 24)
        frame_text = font.render(
            f"Frame: {frame_count + 1}/{total_frames}", True, (255, 255, 255)
        )
        slider_surface.blit(frame_text, (10, 10))

        screen.blit(slider_surface, (0, window_height))

        return slider_x, slider_width, slider_y, slider_height

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
                elif event.key == pygame.K_UP and paused:
                    frame_count = min(frame_count + 60, total_frames - 1)
                elif event.key == pygame.K_DOWN and paused:
                    frame_count = max(frame_count - 60, 0)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    zoom_level *= 1.2
                elif event.key == pygame.K_MINUS:
                    zoom_level = max(
                        0.2, zoom_level / 1.2
                    )  # Allow zoom out below original size
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                slider_x, slider_width, slider_y, slider_height = draw_controls()

                # Handle slider clicks
                if slider_y <= y <= slider_y + slider_height:
                    rel_x = x - slider_x
                    frame_count = int((rel_x / slider_width) * total_frames)
                    frame_count = max(0, min(frame_count, total_frames - 1))
                    paused = True
                elif event.button == 1:  # Left-click to add a marker
                    if y < window_height:
                        video_x = (x - offset_x) / zoom_level
                        video_y = (y - offset_y) / zoom_level
                        coordinates[frame_count].append((video_x, video_y))
                elif event.button == 3:  # Right-click to remove the last marker
                    if coordinates[frame_count]:
                        coordinates[frame_count].pop()
                elif event.button == 4:  # Scroll up (move video up)
                    offset_y = max(offset_y - 20, 0)
                elif event.button == 5:  # Scroll down (move video down)
                    offset_y = min(
                        offset_y + 20, int(original_height * zoom_level) - window_height
                    )
            elif (
                event.type == pygame.MOUSEMOTION and event.buttons[1]
            ):  # Middle button drag
                dx, dy = event.rel
                offset_x = max(
                    0,
                    min(offset_x - dx, int(original_width * zoom_level) - window_width),
                )
                offset_y = max(
                    0,
                    min(
                        offset_y - dy, int(original_height * zoom_level) - window_height
                    ),
                )

        if not paused:
            frame_count = (frame_count + 1) % total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Apply zoom
        zoomed_width = int(original_width * zoom_level)
        zoomed_height = int(original_height * zoom_level)
        zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))
        crop_x = max(0, min(zoomed_width - window_width, offset_x))
        crop_y = max(0, min(zoomed_height - window_height, offset_y))
        cropped_frame = zoomed_frame[
            crop_y : crop_y + window_height, crop_x : crop_x + window_width
        ]

        frame_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )
        screen.blit(frame_surface, (0, 0))

        # Draw markers
        for i, (x, y) in enumerate(coordinates[frame_count]):
            screen_x, screen_y = int((x * zoom_level) - crop_x), int(
                (y * zoom_level) - crop_y
            )
            pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 5)
            font = pygame.font.Font(None, 24)
            text_surface = font.render(str(i + 1), True, (255, 255, 255))
            screen.blit(text_surface, (screen_x + 5, screen_y - 15))

        draw_controls()
        pygame.display.flip()
        clock.tick(fps)

    cap.release()
    pygame.quit()

    # Save coordinates to CSV on exit
    save_coordinates(video_path, coordinates, total_frames)


def save_coordinates(video_path, coordinates, total_frames):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_file = os.path.join(video_dir, f"{base_name}_markers.csv")

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
    print(f"Coordinates saved to: {output_file}")


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
    video_path = get_video_path()
    if video_path:
        play_video_with_controls(video_path)


if __name__ == "__main__":
    main()