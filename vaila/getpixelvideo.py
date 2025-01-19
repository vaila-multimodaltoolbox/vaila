"""
================================================================================
Pixel Coordinate Tool - getpixelvideo.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 09 December 2024
Update: 16 January 2025
Version: 0.0.2
Python Version: 3.12.8

Description:
------------
This tool enables marking and saving pixel coordinates in video frames, with 
zoom functionality for precise annotations. The window can now be resized dynamically, 
and all UI elements adjust accordingly. Users can navigate the video frames, mark 
points, and save results in CSV format.

New Features in This Version:
------------------------------
1. Prompts the user to load existing keypoints from a saved file before starting.
2. Allows the user to choose the keypoint file via a file dialog.

================================================================================
"""

import os
import pygame
import cv2
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog, messagebox


def play_video_with_controls(video_path, coordinates=None):
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

    # Initialize variables
    zoom_level = 1.0
    offset_x, offset_y = 0, 0
    frame_count = 0
    paused = True
    scrolling = False

    # Initialize fresh coordinates if not loaded
    if coordinates is None:
        coordinates = {i: [] for i in range(total_frames)}

    def draw_controls():
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

        # Draw help and save buttons (smaller size and better aligned)
        button_width = 60  # Reduced from 80
        button_height = 25  # Reduced from 30
        button_y = 10

        # Save button
        save_button_rect = pygame.Rect(
            window_width - 140, button_y, button_width, button_height
        )
        pygame.draw.rect(slider_surface, (100, 100, 100), save_button_rect)
        save_text = font.render("Save", True, (255, 255, 255))
        text_rect = save_text.get_rect(center=save_button_rect.center)
        slider_surface.blit(save_text, text_rect)

        # Help button
        help_button_rect = pygame.Rect(
            window_width - 70, button_y, button_width, button_height
        )
        pygame.draw.rect(slider_surface, (100, 100, 100), help_button_rect)
        help_text = font.render("Help", True, (255, 255, 255))
        text_rect = help_text.get_rect(center=help_button_rect.center)
        slider_surface.blit(help_text, text_rect)

        # Blit the slider surface at the bottom of the window area
        screen.blit(slider_surface, (0, window_height))

        return (
            slider_x,
            slider_width,
            slider_y,
            slider_height,
            help_button_rect,
            save_button_rect,
        )

    def show_help_dialog():
        help_message = (
            "Video Player Controls:\n"
            "- Space: Play/Pause\n"
            "- Right Arrow: Next Frame\n"
            "- Left Arrow: Previous Frame\n"
            "- Up Arrow: Fast Forward\n"
            "- Down Arrow: Rewind\n"
            "- +: Zoom In\n"
            "- -: Zoom Out\n"
            "- Left Click: Add Marker\n"
            "- Right Click: Remove Last Marker\n"
            "- Drag Slider: Jump to Frame\n"
        )
        messagebox.showinfo("Help", help_message)

    running = True
    saved = False
    while running:
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
            pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)
            font = pygame.font.Font(None, 24)
            text_surface = font.render(str(i + 1), True, (255, 255, 255))
            screen.blit(text_surface, (screen_x + 5, screen_y - 15))

        (
            slider_x,
            slider_width,
            slider_y,
            slider_height,
            help_button_rect,
            save_button_rect,
        ) = draw_controls()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                if new_h > 80:
                    window_width, window_height = new_w, new_h - 80
                    screen = pygame.display.set_mode((new_w, new_h), pygame.RESIZABLE)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    save_coordinates(video_path, coordinates, total_frames)
                    saved = True
                    messagebox.showinfo("Success", "Coordinates saved successfully!")
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
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    zoom_level *= 1.2
                elif event.key == pygame.K_MINUS:
                    zoom_level = max(0.2, zoom_level / 1.2)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                if event.button == 2:  # Botão do meio do mouse
                    pygame.mouse.get_rel()  # Resetar o movimento relativo
                    scrolling = True
                elif help_button_rect.collidepoint(x, y - window_height):  # Corrigido
                    show_help_dialog()
                elif slider_y <= y <= slider_y + slider_height:
                    rel_x = x - slider_x
                    frame_count = int((rel_x / slider_width) * total_frames)
                    frame_count = max(0, min(frame_count, total_frames - 1))
                    paused = True
                else:
                    if event.button == 1:
                        if y < window_height:
                            video_x = (x + crop_x) / zoom_level
                            video_y = (y + crop_y) / zoom_level
                            coordinates[frame_count].append((video_x, video_y))
                    elif event.button == 3:
                        if coordinates[frame_count]:
                            coordinates[frame_count].pop()

                if save_button_rect.collidepoint(x, y - window_height):
                    save_coordinates(video_path, coordinates, total_frames)
                    saved = True
                    messagebox.showinfo("Success", "Coordinates saved successfully!")

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:  # Botão do meio do mouse
                    scrolling = False

            elif event.type == pygame.MOUSEMOTION:
                if scrolling:
                    rel_x, rel_y = pygame.mouse.get_rel()
                    offset_x = max(
                        0, min(zoomed_width - window_width, offset_x - rel_x)
                    )
                    offset_y = max(
                        0, min(zoomed_height - window_height, offset_y - rel_y)
                    )

        if not paused:
            frame_count = (frame_count + 1) % total_frames

        clock.tick(fps)

    cap.release()
    pygame.quit()

    if saved:
        print("Coordinates were saved.")
    else:
        print("Coordinates were not saved.")


def load_coordinates_from_file(total_frames):
    root = Tk()
    root.withdraw()
    input_file = filedialog.askopenfilename(
        title="Select Keypoint File",
        filetypes=[("CSV Files", "*.csv")],
    )
    if not input_file:
        print("No keypoint file selected. Starting fresh.")
        return {i: [] for i in range(total_frames)}

    try:
        df = pd.read_csv(input_file)
        coordinates = {i: [] for i in range(total_frames)}
        for _, row in df.iterrows():
            frame_num = int(row["frame"])
            for i in range(1, 101):
                x_col = f"p{i}_x"
                y_col = f"p{i}_y"
                if pd.notna(row.get(x_col)) and pd.notna(row.get(y_col)):
                    coordinates[frame_num].append((row[x_col], row[y_col]))
        print(f"Coordinates successfully loaded from: {input_file}")
        return coordinates
    except Exception as e:
        print(f"Error loading coordinates from {input_file}: {e}. Starting fresh.")
        return {i: [] for i in range(total_frames)}


def save_coordinates(video_path, coordinates, total_frames):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_file = os.path.join(video_dir, f"{base_name}_markers.csv")

    columns = ["frame"] + [f"p{i}_{c}" for i in range(1, 101) for c in ["x", "y"]]
    df = pd.DataFrame(np.nan, index=range(total_frames), columns=columns)
    df["frame"] = df.index

    for frame_num, points in coordinates.items():
        for i, (x, y) in enumerate(points):
            df.at[frame_num, f"p{i+1}_x"] = int(x)
            df.at[frame_num, f"p{i+1}_y"] = int(y)

    df.to_csv(output_file, index=False)
    print(f"Coordinates saved to: {output_file}")


def get_video_path():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV")],
    )
    return video_path


def main():
    video_path = get_video_path()
    if not video_path:
        print("No video selected. Exiting.")
        return

    load_existing = messagebox.askyesno(
        "Load Existing Keypoints",
        "Do you want to load existing keypoints from a saved file?",
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if load_existing:
        coordinates = load_coordinates_from_file(total_frames)
    else:
        coordinates = None

    play_video_with_controls(video_path, coordinates)


if __name__ == "__main__":
    main()
