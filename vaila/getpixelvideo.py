"""
================================================================================
Pixel Coordinate Tool - getpixelvideo.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 09 December 2024
Update: 21 March 2025
Version: 0.0.4
Python Version: 3.12.9

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
3. Select keypoint number in the video frame.

================================================================================
"""

import os
from rich import print
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

    # Control variables
    zoom_level = 1.0
    offset_x, offset_y = 0, 0
    frame_count = 0
    paused = True
    scrolling = False
    dragging_slider = False

    # Add marker navigation variables
    selected_marker_idx = -1  # -1 means no marker selected

    # Variables for the "1 line" mode (one-line marker mode)
    one_line_mode = False
    one_line_markers = []  # Each item: (frame_number, x, y)
    deleted_markers = set()  # Keep track of deleted marker indices

    # If no coordinates were loaded, initialize a dictionary with an empty list per frame.
    if coordinates is None:
        coordinates = {i: [] for i in range(total_frames)}

    # For regular mode, we'll track deleted positions
    deleted_positions = {i: set() for i in range(total_frames)}

    # Add persistence variables
    persistence_enabled = False
    persistence_frames = 10  # Default: show points from 10 previous frames

    def draw_controls():
        """
        Draw the control area on a separate surface.
        The frame slider is drawn across the bottom of the control area.
        In the lower-right corner (a bit above the slider) a compact cluster of four buttons is drawn:
          - Save
          - Help
          - "1 line" (toggle one-line marker mode)
          - "Persist" (toggle point persistence)
        """
        control_surface_height = 80
        control_surface = pygame.Surface((window_width, control_surface_height))
        control_surface.fill((30, 30, 30))
        font = pygame.font.Font(None, 20)

        # Draw slider for frames along the bottom.
        slider_margin_left = 10
        slider_margin_right = 10
        slider_width = window_width - slider_margin_left - slider_margin_right
        slider_height = 10
        slider_y = control_surface_height - slider_height - 5
        pygame.draw.rect(
            control_surface,
            (60, 60, 60),
            (slider_margin_left, slider_y, slider_width, slider_height),
        )
        slider_pos = slider_margin_left + int(
            (frame_count / total_frames) * slider_width
        )
        pygame.draw.circle(
            control_surface,
            (255, 255, 255),
            (slider_pos, slider_y + slider_height // 2),
            8,
        )

        # Draw frame info above the slider.
        frame_info = font.render(
            f"Frame: {frame_count + 1}/{total_frames}", True, (255, 255, 255)
        )
        control_surface.blit(frame_info, (slider_margin_left, slider_y - 25))

        # Draw marker navigation and persistence info
        if one_line_mode:
            frame_markers = [m for m in one_line_markers if m[0] == frame_count]
            total_markers = len(frame_markers)
        else:
            total_markers = len(coordinates[frame_count])

        if total_markers > 0:
            marker_idx = selected_marker_idx + 1 if selected_marker_idx >= 0 else 0
            marker_info = font.render(
                f"Marker: {marker_idx}/{total_markers}", True, (255, 255, 255)
            )
            control_surface.blit(marker_info, (slider_margin_left + 200, slider_y - 25))

        # Draw button cluster in the lower-right corner.
        button_width = 50
        button_height = 20
        button_gap = 5
        total_buttons_width = button_width * 4 + button_gap * 3  # Now 4 buttons
        margin_right = 10
        cluster_x = window_width - total_buttons_width - margin_right
        cluster_y = slider_y - button_height - 5

        # Save button.
        save_button_rect = pygame.Rect(
            cluster_x, cluster_y, button_width, button_height
        )
        pygame.draw.rect(control_surface, (100, 100, 100), save_button_rect)
        save_text = font.render("Save", True, (255, 255, 255))
        control_surface.blit(
            save_text, save_text.get_rect(center=save_button_rect.center)
        )

        # Help button.
        help_button_rect = pygame.Rect(
            cluster_x + button_width + button_gap,
            cluster_y,
            button_width,
            button_height,
        )
        pygame.draw.rect(control_surface, (100, 100, 100), help_button_rect)
        help_text = font.render("Help", True, (255, 255, 255))
        control_surface.blit(
            help_text, help_text.get_rect(center=help_button_rect.center)
        )

        # "1 Line" mode toggle button.
        one_line_button_rect = pygame.Rect(
            cluster_x + 2 * (button_width + button_gap),
            cluster_y,
            button_width,
            button_height,
        )
        btn_color = (150, 50, 50) if one_line_mode else (100, 100, 100)
        pygame.draw.rect(control_surface, btn_color, one_line_button_rect)
        one_line_text = font.render("1 Line", True, (255, 255, 255))
        control_surface.blit(
            one_line_text, one_line_text.get_rect(center=one_line_button_rect.center)
        )

        # "Persist" mode toggle button.
        persist_button_rect = pygame.Rect(
            cluster_x + 3 * (button_width + button_gap),
            cluster_y,
            button_width,
            button_height,
        )
        persist_color = (50, 150, 50) if persistence_enabled else (100, 100, 100)
        pygame.draw.rect(control_surface, persist_color, persist_button_rect)

        # Just show "Persist" or "Persist ON" depending on state
        persist_text = font.render(
            "Persist ON" if persistence_enabled else "Persist", True, (255, 255, 255)
        )
        control_surface.blit(
            persist_text, persist_text.get_rect(center=persist_button_rect.center)
        )

        screen.blit(control_surface, (0, window_height))
        return (
            one_line_button_rect,
            save_button_rect,
            help_button_rect,
            persist_button_rect,
            slider_margin_left,
            slider_y,
            slider_width,
            slider_height,
        )

    def show_help_dialog():
        # Instead of using tkinter, display help directly in pygame
        help_lines = [
            "Video Player Controls:",
            "- Space: Play/Pause",
            "- Right Arrow: Next Frame (when paused)",
            "- Left Arrow: Previous Frame (when paused)",
            "- Up Arrow: Fast Forward (when paused)",
            "- Down Arrow: Rewind (when paused)",
            "- +: Zoom In",
            "- -: Zoom Out",
            "- Left Click on video: Add Marker",
            "- Right Click on video: Remove Last Marker",
            "- Middle Click on video: Enable Pan/Move",
            "- Drag Slider: Jump to Frame",
            "- C: Toggle 1 line marker mode",
            "- P: Toggle persistence mode",
            "- 1: Decrease persistence frames by 1",
            "- 2: Increase persistence frames by 1",
            "- 3: Toggle full persistence (all frames)",
            "- TAB: Next marker in current frame",
            "- SHIFT+TAB: Previous marker in current frame",
            "- DELETE: Delete currently selected marker",
            "",
            "Persistence Mode:",
            "Shows markers from previous frames. Use 1 and 2 to adjust",
            "the number of frames, or 3 to toggle full persistence.",
            "",
            "Press any key to close this help",
        ]

        # Create semi-transparent overlay
        overlay = pygame.Surface((window_width, window_height + 80))
        overlay.set_alpha(230)
        overlay.fill((0, 0, 0))

        # Render help text
        font = pygame.font.Font(None, 24)
        line_height = 28

        for i, line in enumerate(help_lines):
            text_surface = font.render(line, True, (255, 255, 255))
            overlay.blit(text_surface, (20, 20 + i * line_height))

        # Display help and wait for key/click
        screen.blit(overlay, (0, 0))
        pygame.display.flip()

        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    waiting_for_input = False
                if event.type == pygame.QUIT:
                    waiting_for_input = False
                    global running
                    running = False

    def show_persistence_settings():
        """Show a dialog to adjust persistence frames"""
        nonlocal persistence_frames

        # Create semi-transparent overlay
        overlay = pygame.Surface((window_width, window_height + 80))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))

        # Create UI elements
        font = pygame.font.Font(None, 36)
        title = font.render("Persistence Settings", True, (255, 255, 255))

        instruction = font.render(
            "Use + and - keys to adjust frames, Enter to confirm", True, (255, 255, 255)
        )

        value_text = font.render(f"Frames: {persistence_frames}", True, (255, 255, 255))

        # Display overlay and UI
        screen.blit(overlay, (0, 0))
        screen.blit(
            title,
            (window_width // 2 - title.get_width() // 2, window_height // 2 - 100),
        )
        screen.blit(
            instruction,
            (window_width // 2 - instruction.get_width() // 2, window_height // 2 - 40),
        )
        screen.blit(
            value_text,
            (window_width // 2 - value_text.get_width() // 2, window_height // 2 + 20),
        )

        pygame.display.flip()

        # Handle input
        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting_for_input = False
                    global running
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                        waiting_for_input = False
                    elif event.key in (
                        pygame.K_PLUS,
                        pygame.K_EQUALS,
                        pygame.K_KP_PLUS,
                    ):
                        persistence_frames = min(persistence_frames + 1, 100)
                        value_text = font.render(
                            f"Frames: {persistence_frames}", True, (255, 255, 255)
                        )
                        # Redraw
                        screen.blit(overlay, (0, 0))
                        screen.blit(
                            title,
                            (
                                window_width // 2 - title.get_width() // 2,
                                window_height // 2 - 100,
                            ),
                        )
                        screen.blit(
                            instruction,
                            (
                                window_width // 2 - instruction.get_width() // 2,
                                window_height // 2 - 40,
                            ),
                        )
                        screen.blit(
                            value_text,
                            (
                                window_width // 2 - value_text.get_width() // 2,
                                window_height // 2 + 20,
                            ),
                        )
                        pygame.display.flip()
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        persistence_frames = max(persistence_frames - 1, 1)
                        value_text = font.render(
                            f"Frames: {persistence_frames}", True, (255, 255, 255)
                        )
                        # Redraw
                        screen.blit(overlay, (0, 0))
                        screen.blit(
                            title,
                            (
                                window_width // 2 - title.get_width() // 2,
                                window_height // 2 - 100,
                            ),
                        )
                        screen.blit(
                            instruction,
                            (
                                window_width // 2 - instruction.get_width() // 2,
                                window_height // 2 - 40,
                            ),
                        )
                        screen.blit(
                            value_text,
                            (
                                window_width // 2 - value_text.get_width() // 2,
                                window_height // 2 + 20,
                            ),
                        )
                        pygame.display.flip()

    def save_1_line_coordinates(video_path, one_line_markers, deleted_markers=None):
        """Save markers created in one-line mode to a CSV file."""
        if not one_line_markers:
            print("No one line markers to save.")
            return

        if deleted_markers is None:
            deleted_markers = set()

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        output_file = os.path.join(video_dir, f"{base_name}_markers_1_line.csv")

        # Find largest marker index (accounting for deleted ones)
        max_marker = max(
            [
                idx
                for idx, _ in enumerate(one_line_markers)
                if idx not in deleted_markers
            ],
            default=0,
        )

        # Create header: frame column and p1_x, p1_y, p2_x, p2_y, etc.
        header = ["frame"]
        for i in range(
            1, max_marker + 2
        ):  # +2 because we need to add one more than max index
            header.extend([f"p{i}_x", f"p{i}_y"])

        # Get the frame number from the first non-deleted marker
        first_marker = next(
            (m for idx, m in enumerate(one_line_markers) if idx not in deleted_markers),
            None,
        )
        if not first_marker:
            print("No valid markers to save after deletions.")
            return

        # Fill the row with values, preserving marker indices even with deletions
        row_values = [int(first_marker[0])]  # First marker's frame value

        # Initialize all positions with empty values
        for _ in range(max_marker + 1):
            row_values.extend(["", ""])  # Empty x, y values

        # Fill in the non-deleted marker positions
        for idx, (_, x, y) in enumerate(one_line_markers):
            if idx not in deleted_markers:
                # Marker indices are 1-based in the CSV
                row_values[idx * 2 + 1] = int(
                    x
                )  # +1 for frame column, then multiply by 2 for x position
                row_values[idx * 2 + 2] = int(y)  # +2 for y position

        df = pd.DataFrame([row_values], columns=header)
        df.to_csv(output_file, index=False)

        print(f"1 line coordinates saved to: {output_file}")
        return output_file

    running = True
    saved = False
    showing_save_message = False
    save_message_timer = 0
    save_message_text = ""

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

        # Draw persistent markers first (draw in order from oldest to newest)
        font = pygame.font.Font(None, 24)
        if persistence_enabled:
            # Calculate range of frames to show
            start_frame = max(0, frame_count - persistence_frames)

            # Draw markers from previous frames with decreasing opacity
            for past_frame in range(start_frame, frame_count):
                # Calculate opacity based on how far back this frame is
                opacity = int(
                    200
                    * (past_frame - start_frame + 1)
                    / (frame_count - start_frame + 1)
                )

                if one_line_mode:
                    for idx, (f_num, x, y) in enumerate(one_line_markers):
                        if idx in deleted_markers:
                            continue  # Skip deleted markers

                        if f_num == past_frame:
                            screen_x = int((x * zoom_level) - crop_x)
                            screen_y = int((y * zoom_level) - crop_y)

                            # Draw past markers in gray with decreasing opacity
                            # Create a temporary surface for the semi-transparent circle
                            circle_surface = pygame.Surface((14, 14), pygame.SRCALPHA)
                            pygame.draw.circle(
                                circle_surface, (150, 150, 150, opacity), (7, 7), 3
                            )
                            screen.blit(circle_surface, (screen_x - 7, screen_y - 7))

                            # Draw the marker number with reduced opacity
                            text_surface = font.render(
                                str(idx + 1), True, (200, 200, 200, opacity)
                            )
                            text_surface.set_alpha(opacity)
                            screen.blit(text_surface, (screen_x + 5, screen_y - 15))
                else:
                    for i, (x, y) in enumerate(coordinates[past_frame]):
                        if i in deleted_positions[past_frame]:
                            continue  # Skip deleted markers

                        screen_x = int((x * zoom_level) - crop_x)
                        screen_y = int((y * zoom_level) - crop_y)

                        # Draw past markers in gray with decreasing opacity
                        # Create a temporary surface for the semi-transparent circle
                        circle_surface = pygame.Surface((14, 14), pygame.SRCALPHA)
                        pygame.draw.circle(
                            circle_surface, (150, 150, 150, opacity), (7, 7), 3
                        )
                        screen.blit(circle_surface, (screen_x - 7, screen_y - 7))

                        # Draw the marker number with reduced opacity
                        text_surface = font.render(
                            str(i + 1), True, (200, 200, 200, opacity)
                        )
                        text_surface.set_alpha(opacity)
                        screen.blit(text_surface, (screen_x + 5, screen_y - 15))

        # Draw current frame markers (same as before, on top of persistence markers)
        if one_line_mode:
            frame_markers = []
            for idx, (f_num, x, y) in enumerate(one_line_markers):
                if idx in deleted_markers:
                    continue  # Skip deleted markers

                if f_num == frame_count:
                    frame_markers.append((idx, x, y))
                    screen_x = int((x * zoom_level) - crop_x)
                    screen_y = int((y * zoom_level) - crop_y)

                    # Highlight selected marker
                    if idx == selected_marker_idx:
                        pygame.draw.circle(
                            screen, (255, 165, 0), (screen_x, screen_y), 7
                        )  # Orange highlight

                    pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)
                    text_surface = font.render(str(idx + 1), True, (255, 255, 255))
                    screen.blit(text_surface, (screen_x + 5, screen_y - 15))
        else:
            for i, (x, y) in enumerate(coordinates[frame_count]):
                if i in deleted_positions[frame_count]:
                    continue  # Skip deleted markers

                screen_x = int((x * zoom_level) - crop_x)
                screen_y = int((y * zoom_level) - crop_y)

                # Highlight selected marker
                if i == selected_marker_idx:
                    pygame.draw.circle(
                        screen, (255, 165, 0), (screen_x, screen_y), 7
                    )  # Orange highlight

                pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)
                text_surface = font.render(str(i + 1), True, (255, 255, 255))
                screen.blit(text_surface, (screen_x + 5, screen_y - 15))

        (
            one_line_button_rect,
            save_button_rect,
            help_button_rect,
            persist_button_rect,
            slider_x,
            slider_y,
            slider_width,
            slider_height,
        ) = draw_controls()

        # Show save message if needed
        if showing_save_message:
            save_message_timer -= 1
            if save_message_timer <= 0:
                showing_save_message = False
            else:
                # Draw save message notification at top of screen
                msg_font = pygame.font.Font(None, 28)
                msg_surface = msg_font.render(save_message_text, True, (255, 255, 255))
                msg_bg = pygame.Surface(
                    (msg_surface.get_width() + 20, msg_surface.get_height() + 10)
                )
                msg_bg.set_alpha(200)
                msg_bg.fill((0, 100, 0))
                screen.blit(msg_bg, (window_width // 2 - msg_bg.get_width() // 2, 10))
                screen.blit(
                    msg_surface, (window_width // 2 - msg_surface.get_width() // 2, 15)
                )

        # Add current marker indicator if markers exist
        if one_line_mode:
            frame_markers_indices = [
                i
                for i, m in enumerate(one_line_markers)
                if m[0] == frame_count and i not in deleted_markers
            ]
            total_markers = len(frame_markers_indices)

            # Find the position of the selected marker in this frame's markers
            if selected_marker_idx >= 0:
                visible_markers = [i for i in frame_markers_indices]
                try:
                    marker_position = visible_markers.index(selected_marker_idx) + 1
                except ValueError:
                    marker_position = 0
            else:
                marker_position = 0
        else:
            visible_markers = [
                i
                for i in range(len(coordinates[frame_count]))
                if i not in deleted_positions[frame_count]
            ]
            total_markers = len(visible_markers)

            marker_position = 0
            if selected_marker_idx >= 0:
                try:
                    marker_position = visible_markers.index(selected_marker_idx) + 1
                except ValueError:
                    marker_position = 0

        if total_markers > 0 and marker_position > 0:
            indicator_font = pygame.font.Font(None, 36)
            indicator_text = f"Marker {selected_marker_idx+1}/{len(visible_markers)}"
            indicator_surface = indicator_font.render(
                indicator_text, True, (255, 255, 255)
            )

            # Create background for text
            indicator_bg = pygame.Surface(
                (
                    indicator_surface.get_width() + 20,
                    indicator_surface.get_height() + 10,
                )
            )
            indicator_bg.set_alpha(180)
            indicator_bg.fill((50, 50, 50))

            # Position in top-right corner
            screen.blit(
                indicator_bg, (window_width - indicator_bg.get_width() - 10, 10)
            )
            screen.blit(
                indicator_surface,
                (window_width - indicator_surface.get_width() - 20, 15),
            )

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
                    if one_line_mode:
                        output_file = save_1_line_coordinates(
                            video_path, one_line_markers, deleted_markers
                        )
                    else:
                        output_file = save_coordinates(
                            video_path, coordinates, total_frames, deleted_positions
                        )
                    saved = True
                    save_message_text = f"Saved to: {os.path.basename(output_file)}"
                    showing_save_message = True
                    save_message_timer = 90  # Show for about 3 seconds at 30fps
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
                elif event.key == pygame.K_c:
                    one_line_mode = not one_line_mode
                    selected_marker_idx = (
                        -1
                    )  # Reset selected marker when changing modes
                elif event.key == pygame.K_TAB:
                    # Completely revamped marker navigation
                    if one_line_mode:
                        # Get all marker indices in current frame, including deleted ones
                        frame_marker_indices = []
                        deleted_frame_markers = []

                        # Collect available markers and the maximum index
                        max_marker_index = -1
                        for idx, (f_num, _, _) in enumerate(one_line_markers):
                            if idx > max_marker_index:
                                max_marker_index = idx

                            if f_num == frame_count:
                                if idx in deleted_markers:
                                    deleted_frame_markers.append(idx)
                                else:
                                    frame_marker_indices.append(idx)

                        # If no direction set (first tab press), start from 0 or current
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            # Previous marker
                            if selected_marker_idx == -1:
                                # If no selection, go to the last available marker
                                if frame_marker_indices:
                                    selected_marker_idx = frame_marker_indices[-1]
                                elif deleted_frame_markers:
                                    selected_marker_idx = deleted_frame_markers[-1]
                                elif max_marker_index >= 0:
                                    # If no marker in this frame, go to max index
                                    selected_marker_idx = max_marker_index
                            else:
                                # Find the previous marker (lower index)
                                prev_visible = [
                                    idx
                                    for idx in frame_marker_indices
                                    if idx < selected_marker_idx
                                ]
                                prev_deleted = [
                                    idx
                                    for idx in deleted_frame_markers
                                    if idx < selected_marker_idx
                                ]
                                prev_indices = prev_visible + prev_deleted

                                if prev_indices:
                                    selected_marker_idx = max(prev_indices)
                                else:
                                    # Wrap around to max marker
                                    all_markers = (
                                        frame_marker_indices + deleted_frame_markers
                                    )
                                    if all_markers:
                                        selected_marker_idx = max(all_markers)
                                    else:
                                        selected_marker_idx = max_marker_index
                        else:
                            # Next marker
                            if selected_marker_idx == -1:
                                # If no selection, start with the first marker
                                if frame_marker_indices:
                                    selected_marker_idx = min(frame_marker_indices)
                                elif deleted_frame_markers:
                                    selected_marker_idx = min(deleted_frame_markers)
                                elif max_marker_index >= 0:
                                    # If no marker in this frame, start from 0
                                    selected_marker_idx = 0
                            else:
                                # Find the next marker (higher index)
                                next_visible = [
                                    idx
                                    for idx in frame_marker_indices
                                    if idx > selected_marker_idx
                                ]
                                next_deleted = [
                                    idx
                                    for idx in deleted_frame_markers
                                    if idx > selected_marker_idx
                                ]
                                next_indices = next_visible + next_deleted

                                if next_indices:
                                    selected_marker_idx = min(next_indices)
                                else:
                                    # Wrap around to first marker
                                    all_markers = (
                                        frame_marker_indices + deleted_frame_markers
                                    )
                                    if all_markers:
                                        selected_marker_idx = min(all_markers)
                                    else:
                                        selected_marker_idx = 0
                    else:
                        # Regular mode navigation
                        # Get indices of all markers in current frame
                        visible_markers = []
                        deleted_markers_in_frame = []

                        for i in range(len(coordinates[frame_count])):
                            if i in deleted_positions[frame_count]:
                                deleted_markers_in_frame.append(i)
                            else:
                                visible_markers.append(i)

                        # Get the max index that could be selected
                        max_index = max(
                            [len(coordinates[frame_count]) - 1]
                            + list(deleted_positions[frame_count])
                            + [-1]
                        )

                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            # Previous marker
                            if selected_marker_idx == -1:
                                # If no selection, go to the last available marker
                                if visible_markers:
                                    selected_marker_idx = max(visible_markers)
                                elif deleted_markers_in_frame:
                                    selected_marker_idx = max(deleted_markers_in_frame)
                                else:
                                    selected_marker_idx = max_index
                            else:
                                # Find the previous marker
                                prev_visible = [
                                    idx
                                    for idx in visible_markers
                                    if idx < selected_marker_idx
                                ]
                                prev_deleted = [
                                    idx
                                    for idx in deleted_markers_in_frame
                                    if idx < selected_marker_idx
                                ]
                                prev_indices = prev_visible + prev_deleted

                                if prev_indices:
                                    selected_marker_idx = max(prev_indices)
                                else:
                                    # Wrap around to highest marker
                                    all_markers = (
                                        visible_markers + deleted_markers_in_frame
                                    )
                                    if all_markers:
                                        selected_marker_idx = max(all_markers)
                                    else:
                                        selected_marker_idx = max_index
                        else:
                            # Next marker
                            if selected_marker_idx == -1:
                                # If no selection, start with the first marker
                                if visible_markers:
                                    selected_marker_idx = min(visible_markers)
                                elif deleted_markers_in_frame:
                                    selected_marker_idx = min(deleted_markers_in_frame)
                                else:
                                    selected_marker_idx = 0
                            else:
                                # Find the next marker
                                next_visible = [
                                    idx
                                    for idx in visible_markers
                                    if idx > selected_marker_idx
                                ]
                                next_deleted = [
                                    idx
                                    for idx in deleted_markers_in_frame
                                    if idx > selected_marker_idx
                                ]
                                next_indices = next_visible + next_deleted

                                if next_indices:
                                    selected_marker_idx = min(next_indices)
                                else:
                                    # Wrap around to first marker
                                    all_markers = (
                                        visible_markers + deleted_markers_in_frame
                                    )
                                    if all_markers:
                                        selected_marker_idx = min(all_markers)
                                    else:
                                        selected_marker_idx = 0

                elif event.key == pygame.K_DELETE:
                    # Mark the currently selected marker as deleted without changing indices
                    if selected_marker_idx >= 0:
                        if one_line_mode:
                            # Add to deleted set instead of removing
                            if selected_marker_idx < len(one_line_markers):
                                deleted_markers.add(selected_marker_idx)

                                # Get non-deleted markers in this frame
                                frame_markers_indices = [
                                    i
                                    for i, m in enumerate(one_line_markers)
                                    if m[0] == frame_count and i not in deleted_markers
                                ]

                                # Select next visible marker if available
                                if frame_markers_indices:
                                    next_indices = [
                                        i
                                        for i in frame_markers_indices
                                        if i > selected_marker_idx
                                    ]
                                    if next_indices:
                                        selected_marker_idx = next_indices[0]
                                    else:
                                        # No marker after this one, select previous one
                                        prev_indices = [
                                            i
                                            for i in frame_markers_indices
                                            if i < selected_marker_idx
                                        ]
                                        if prev_indices:
                                            selected_marker_idx = prev_indices[-1]
                                        else:
                                            selected_marker_idx = -1  # No markers left
                                else:
                                    selected_marker_idx = -1  # No visible markers left
                        else:
                            # Mark as deleted in the current frame
                            if selected_marker_idx < len(coordinates[frame_count]):
                                deleted_positions[frame_count].add(selected_marker_idx)

                                # Find next visible marker
                                frame_markers_indices = [
                                    i
                                    for i in range(len(coordinates[frame_count]))
                                    if i not in deleted_positions[frame_count]
                                ]

                                if frame_markers_indices:
                                    next_indices = [
                                        i
                                        for i in frame_markers_indices
                                        if i > selected_marker_idx
                                    ]
                                    if next_indices:
                                        selected_marker_idx = next_indices[0]
                                    else:
                                        # No marker after this one, select previous one
                                        prev_indices = [
                                            i
                                            for i in frame_markers_indices
                                            if i < selected_marker_idx
                                        ]
                                        if prev_indices:
                                            selected_marker_idx = prev_indices[-1]
                                        else:
                                            selected_marker_idx = -1  # No markers left
                                else:
                                    selected_marker_idx = -1  # No visible markers left

                # Add persistence toggle with 'p' key
                elif event.key == pygame.K_p:
                    persistence_enabled = not persistence_enabled
                    # Show confirmation message
                    save_message_text = f"Persistence {'enabled' if persistence_enabled else 'disabled'}"
                    showing_save_message = True
                    save_message_timer = 30

                # Adjust persistence frames with '1', '2', and '3' keys
                elif event.key == pygame.K_1:  # Decrease persistence frames
                    if persistence_enabled:
                        persistence_frames = max(1, persistence_frames - 1)
                        save_message_text = f"Persistence: {persistence_frames} frames"
                        showing_save_message = True
                        save_message_timer = 30

                elif event.key == pygame.K_2:  # Increase persistence frames
                    if persistence_enabled:
                        persistence_frames = min(100, persistence_frames + 1)
                        save_message_text = f"Persistence: {persistence_frames} frames"
                        showing_save_message = True
                        save_message_timer = 30

                elif event.key == pygame.K_3:  # Toggle full persistence
                    if persistence_enabled and persistence_frames < total_frames:
                        # Save current frames before switching to full mode
                        previous_frames = persistence_frames
                        persistence_frames = total_frames
                        save_message_text = "Full persistence enabled"
                    else:
                        # If already in full mode or just enabling, set to 10 frames
                        persistence_frames = 10
                        save_message_text = f"Persistence: {persistence_frames} frames"
                    showing_save_message = True
                    save_message_timer = 30

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y >= window_height:
                    # Clique na área de controles
                    rel_y = y - window_height
                    if one_line_button_rect.collidepoint(x, rel_y):
                        one_line_mode = not one_line_mode
                        selected_marker_idx = -1  # Reset selected marker
                    elif help_button_rect.collidepoint(x, rel_y):
                        show_help_dialog()
                    elif save_button_rect.collidepoint(x, rel_y):
                        if one_line_mode:
                            output_file = save_1_line_coordinates(
                                video_path, one_line_markers, deleted_markers
                            )
                        else:
                            output_file = save_coordinates(
                                video_path, coordinates, total_frames, deleted_positions
                            )
                        saved = True
                        save_message_text = f"Saved to: {os.path.basename(output_file)}"
                        showing_save_message = True
                        save_message_timer = 90  # Show for about 3 seconds at 30fps
                    elif persist_button_rect.collidepoint(x, rel_y):
                        # Remove persistence settings dialog
                        persistence_enabled = not persistence_enabled
                        save_message_text = f"Persistence {'enabled' if persistence_enabled else 'disabled'}"
                        showing_save_message = True
                        save_message_timer = 30
                    elif slider_y <= rel_y <= slider_y + slider_height:
                        dragging_slider = True
                        rel_x = x - slider_x
                        rel_x = max(0, min(rel_x, slider_width))
                        frame_count = int((rel_x / slider_width) * total_frames)
                        frame_count = max(0, min(frame_count, total_frames - 1))
                        paused = True
                else:
                    # Clique na área do vídeo
                    video_x = (x + crop_x) / zoom_level
                    video_y = (y + crop_y) / zoom_level

                    if event.button == 1:  # Left click
                        # Check if a marker is selected
                        if selected_marker_idx >= 0:
                            if one_line_mode:
                                # Find if this marker exists in the current frame
                                marker_in_frame = False
                                for i, (f_num, _, _) in enumerate(one_line_markers):
                                    if (
                                        i == selected_marker_idx
                                        and f_num == frame_count
                                    ):
                                        # Update existing marker
                                        one_line_markers[i] = (
                                            frame_count,
                                            video_x,
                                            video_y,
                                        )
                                        save_message_text = (
                                            f"Updated marker {selected_marker_idx+1}"
                                        )
                                        showing_save_message = True
                                        save_message_timer = 30
                                        marker_in_frame = True
                                        break

                                # If marker doesn't exist or was deleted, create/recreate it
                                if not marker_in_frame:
                                    if selected_marker_idx in deleted_markers:
                                        deleted_markers.remove(selected_marker_idx)

                                    # Find appropriate position to insert marker
                                    insert_pos = len(one_line_markers)
                                    for i, (f_num, _, _) in enumerate(one_line_markers):
                                        if (
                                            f_num == frame_count
                                            and i > selected_marker_idx
                                        ):
                                            insert_pos = i
                                            break

                                    # Create new marker at selected index
                                    one_line_markers.insert(
                                        insert_pos, (frame_count, video_x, video_y)
                                    )
                                    save_message_text = f"{'Recreated' if selected_marker_idx < len(one_line_markers) else 'Added'} marker {selected_marker_idx+1}"
                                    showing_save_message = True
                                    save_message_timer = 30
                            else:
                                # Ensure the coordinates list is long enough for this marker
                                while (
                                    len(coordinates[frame_count]) <= selected_marker_idx
                                ):
                                    coordinates[frame_count].append((0, 0))

                                # Update marker position
                                coordinates[frame_count][selected_marker_idx] = (
                                    video_x,
                                    video_y,
                                )

                                # If this was a deleted marker, remove it from deleted list
                                if (
                                    selected_marker_idx
                                    in deleted_positions[frame_count]
                                ):
                                    deleted_positions[frame_count].remove(
                                        selected_marker_idx
                                    )

                                save_message_text = (
                                    f"Updated marker {selected_marker_idx+1}"
                                )
                                showing_save_message = True
                                save_message_timer = 30
                        else:
                            # No marker selected, add a new one
                            if one_line_mode:
                                one_line_markers.append((frame_count, video_x, video_y))
                                selected_marker_idx = len(one_line_markers) - 1
                            else:
                                coordinates[frame_count].append((video_x, video_y))
                                selected_marker_idx = len(coordinates[frame_count]) - 1

                    elif event.button == 3:  # Right click
                        # Keep existing behavior for right-click (delete most recent)
                        if one_line_mode:
                            for i in range(len(one_line_markers) - 1, -1, -1):
                                if one_line_markers[i][0] == frame_count:
                                    del one_line_markers[i]
                                    break
                        else:
                            if coordinates[frame_count]:
                                coordinates[frame_count].pop()
                        # Reset selection if we removed the selected marker
                        if one_line_mode:
                            markers_in_frame = [
                                i
                                for i, m in enumerate(one_line_markers)
                                if m[0] == frame_count
                            ]
                            if not markers_in_frame:
                                selected_marker_idx = -1
                            elif selected_marker_idx >= len(markers_in_frame):
                                selected_marker_idx = len(markers_in_frame) - 1
                        else:
                            if not coordinates[frame_count]:
                                selected_marker_idx = -1
                            elif selected_marker_idx >= len(coordinates[frame_count]):
                                selected_marker_idx = len(coordinates[frame_count]) - 1

                    elif event.button == 2:  # Middle click for panning
                        scrolling = True
                        pygame.mouse.get_rel()  # Zera o acumulador de movimento relativo para pan contínuo

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    scrolling = False
                elif event.button == 1:
                    dragging_slider = False

            elif event.type == pygame.MOUSEMOTION:
                if scrolling:
                    rel_dx, rel_dy = pygame.mouse.get_rel()
                    offset_x = max(
                        0, min(zoomed_width - window_width, offset_x - rel_dx)
                    )
                    offset_y = max(
                        0, min(zoomed_height - window_height, offset_y - rel_dy)
                    )
                if dragging_slider:
                    rel_x = event.pos[0] - slider_x
                    rel_x = max(0, min(rel_x, slider_width))
                    frame_count = int((rel_x / slider_width) * total_frames)
                    frame_count = max(0, min(frame_count, total_frames - 1))
                    paused = True

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


def save_coordinates(video_path, coordinates, total_frames, deleted_positions=None):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_file = os.path.join(video_dir, f"{base_name}_markers.csv")

    # Initialize deleted_positions if not provided
    if deleted_positions is None:
        deleted_positions = {i: set() for i in range(total_frames)}

    # Determina o número máximo de pontos marcados em qualquer frame.
    max_points = max((len(points) for points in coordinates.values()), default=0)

    # Cria o cabeçalho: a primeira coluna é 'frame' e para cada ponto,
    # adiciona as colunas 'p{i}_x' e 'p{i}_y'
    columns = ["frame"]
    for i in range(1, max_points + 1):
        columns.append(f"p{i}_x")
        columns.append(f"p{i}_y")

    # Cria o DataFrame inicializado com NaN para todos os frames.
    df = pd.DataFrame(np.nan, index=range(total_frames), columns=columns)
    df["frame"] = df.index

    # Preenche o DataFrame com os pontos marcados, convertendo pra int.
    for frame_num, points in coordinates.items():
        for i, (x, y) in enumerate(points):
            if i not in deleted_positions[frame_num]:  # Only save non-deleted markers
                df.at[frame_num, f"p{i+1}_x"] = int(x)
                df.at[frame_num, f"p{i+1}_y"] = int(y)

    # Substitui os valores NaN por strings vazias.
    df.fillna("", inplace=True)

    df.to_csv(output_file, index=False)
    print(f"Coordinates saved to: {output_file}")
    return output_file


def get_video_path():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV")],
    )
    return video_path


def main():
    # Print the script version and directory
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("-" * 80)

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
