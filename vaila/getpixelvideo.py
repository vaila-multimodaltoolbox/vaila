"""
================================================================================
Pixel Coordinate Tool - getpixelvideo.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 09 December 2024
Update: 10 March 2025
Version: 0.0.3
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

    # Variables for the "1 line" mode (one-line marker mode)
    one_line_mode = False
    one_line_markers = []  # Each item: (frame_number, x, y)

    # If no coordinates were loaded, initialize a dictionary with an empty list per frame.
    if coordinates is None:
        coordinates = {i: [] for i in range(total_frames)}

    def draw_controls():
        """
        Draw the control area on a separate surface.
        The frame slider is drawn across the bottom of the control area.
        In the lower-right corner (a bit above the slider) a compact cluster of three buttons is drawn:
          - Save
          - Help
          - "1 line" (toggle one-line marker mode)
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

        # Draw button cluster in the lower-right corner.
        button_width = 50
        button_height = 20
        button_gap = 5
        total_buttons_width = button_width * 3 + button_gap * 2
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

        screen.blit(control_surface, (0, window_height))
        return (
            one_line_button_rect,
            save_button_rect,
            help_button_rect,
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
            "     (When enabled, all markers are appended sequentially in one line.)",
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

    def save_1_line_coordinates(video_path, one_line_markers):
        if not one_line_markers:
            print("No one line markers to save.")
            return

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        output_file = os.path.join(video_dir, f"{base_name}_markers_1_line.csv")

        # Monta o header: coluna 0 é "frame" e as demais são p1_x, p1_y, p2_x, p2_y, ...
        header = ["frame"]
        n = len(one_line_markers)
        for i in range(1, n + 1):
            header.extend([f"p{i}_x", f"p{i}_y"])

        # Preenche a linha de valores:
        # Utiliza o frame do primeiro marker para a coluna "frame" e,
        # em seguida, coloca apenas os valores x e y de cada marker.
        row_values = [int(one_line_markers[0][0])]  # primeiro marker: valor de frame
        for marker in one_line_markers:
            # marker: (frame, x, y)
            _, x, y = marker
            row_values.extend([int(x), int(y)])

        df = pd.DataFrame([row_values], columns=header)
        df.to_csv(output_file, index=False)

        # Display save confirmation on screen instead of messagebox
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

        # Draw markers
        font = pygame.font.Font(None, 24)
        if one_line_mode:
            for idx, (f_num, x, y) in enumerate(one_line_markers):
                if f_num == frame_count:
                    screen_x = int((x * zoom_level) - crop_x)
                    screen_y = int((y * zoom_level) - crop_y)
                    pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)
                    text_surface = font.render(str(idx + 1), True, (255, 255, 255))
                    screen.blit(text_surface, (screen_x + 5, screen_y - 15))
        else:
            for i, (x, y) in enumerate(coordinates[frame_count]):
                screen_x = int((x * zoom_level) - crop_x)
                screen_y = int((y * zoom_level) - crop_y)
                pygame.draw.circle(screen, (0, 255, 0), (screen_x, screen_y), 3)
                text_surface = font.render(str(i + 1), True, (255, 255, 255))
                screen.blit(text_surface, (screen_x + 5, screen_y - 15))

        (
            one_line_button_rect,
            save_button_rect,
            help_button_rect,
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
                            video_path, one_line_markers
                        )
                    else:
                        output_file = save_coordinates(
                            video_path, coordinates, total_frames
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

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y >= window_height:
                    # Clique na área de controles
                    rel_y = y - window_height
                    if one_line_button_rect.collidepoint(x, rel_y):
                        one_line_mode = not one_line_mode
                    elif help_button_rect.collidepoint(x, rel_y):
                        show_help_dialog()
                    elif save_button_rect.collidepoint(x, rel_y):
                        if one_line_mode:
                            output_file = save_1_line_coordinates(
                                video_path, one_line_markers
                            )
                        else:
                            output_file = save_coordinates(
                                video_path, coordinates, total_frames
                            )
                        saved = True
                        save_message_text = f"Saved to: {os.path.basename(output_file)}"
                        showing_save_message = True
                        save_message_timer = 90  # Show for about 3 seconds at 30fps
                    elif slider_y <= rel_y <= slider_y + slider_height:
                        dragging_slider = True
                        rel_x = x - slider_x
                        rel_x = max(0, min(rel_x, slider_width))
                        frame_count = int((rel_x / slider_width) * total_frames)
                        frame_count = max(0, min(frame_count, total_frames - 1))
                        paused = True
                else:
                    # Clique na área do vídeo
                    if event.button == 1:  # Botão esquerdo: adiciona marcador.
                        video_x = (x + crop_x) / zoom_level
                        video_y = (y + crop_y) / zoom_level
                        if one_line_mode:
                            one_line_markers.append((frame_count, video_x, video_y))
                        else:
                            coordinates[frame_count].append((video_x, video_y))
                    elif event.button == 3:  # Botão direito: remove o último marcador.
                        if one_line_mode:
                            for i in range(len(one_line_markers) - 1, -1, -1):
                                if one_line_markers[i][0] == frame_count:
                                    del one_line_markers[i]
                                    break
                        else:
                            if coordinates[frame_count]:
                                coordinates[frame_count].pop()
                    elif event.button == 2:  # Botão do meio: ativa o movimento (pan).
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


def save_coordinates(video_path, coordinates, total_frames):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_file = os.path.join(video_dir, f"{base_name}_markers.csv")

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
