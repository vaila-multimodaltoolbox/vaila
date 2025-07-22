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
from datetime import datetime


def get_color_for_id(marker_id):
    """Generate a consistent color for a given marker ID."""
    colors = [
        (0, 255, 0),  # Green
        (255, 0, 0),  # Red
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (128, 255, 0),  # Lime
    ]
    return colors[marker_id % len(colors)]


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
    selected_marker_idx = 0  # Começar sempre com o marker 1 selecionado

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

    # Add sequential mode variable
    sequential_mode = False

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
        button_gap = 10
        persist_button_width = 70
        seq_button_width = 70
        total_buttons_width = (
            (button_width * 3)
            + persist_button_width
            + seq_button_width
            + (button_gap * 4)
        )
        cluster_x = (window_width - total_buttons_width) // 2
        cluster_y = slider_y - button_height - 5

        # Load button
        load_button_rect = pygame.Rect(
            cluster_x, cluster_y, button_width, button_height
        )
        pygame.draw.rect(control_surface, (100, 100, 100), load_button_rect)
        load_text = font.render("Load", True, (255, 255, 255))
        control_surface.blit(
            load_text, load_text.get_rect(center=load_button_rect.center)
        )

        # Save button.
        save_button_rect = pygame.Rect(
            cluster_x + button_width + button_gap,
            cluster_y,
            button_width,
            button_height,
        )
        pygame.draw.rect(control_surface, (100, 100, 100), save_button_rect)
        save_text = font.render("Save", True, (255, 255, 255))
        control_surface.blit(
            save_text, save_text.get_rect(center=save_button_rect.center)
        )

        # Help button.
        help_button_rect = pygame.Rect(
            cluster_x + 2 * (button_width + button_gap),
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
            cluster_x + 3 * (button_width + button_gap),
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
            cluster_x + 4 * (button_width + button_gap),
            cluster_y,
            persist_button_width,
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

        # Add Sequential mode button after persist button
        seq_button_rect = pygame.Rect(
            cluster_x
            + 4 * (button_width + button_gap)
            + persist_button_width
            + button_gap,
            cluster_y,
            seq_button_width,
            button_height,
        )
        seq_color = (50, 150, 50) if sequential_mode else (100, 100, 100)
        pygame.draw.rect(control_surface, seq_color, seq_button_rect)
        seq_text = font.render("Sequential", True, (255, 255, 255))
        control_surface.blit(seq_text, seq_text.get_rect(center=seq_button_rect.center))

        screen.blit(control_surface, (0, window_height))
        return (
            one_line_button_rect,
            save_button_rect,
            help_button_rect,
            persist_button_rect,
            load_button_rect,
            seq_button_rect,  # Add sequential button to return
            slider_margin_left,
            slider_y,
            slider_width,
            slider_height,
        )

    def show_help_dialog():
        # Instead of using tkinter, display help directly in pygame
        help_lines_left = [
            "Video Player Controls:",
            "- Space: Play/Pause",
            "- Right Arrow: Next Frame (when paused)",
            "- Left Arrow: Previous Frame (when paused)",
            "- Up Arrow: Fast Forward (when paused)",
            "- Down Arrow: Rewind (when paused)",
            "- +: Zoom In",
            "- -: Zoom Out",
            "- Left Click: Add Marker",
            "- Right Click: Remove Last Marker",
            "- Middle Click: Enable Pan/Move",
            "- Drag Slider: Jump to Frame",
            "- TAB: Next marker in current frame",
            "- SHIFT+TAB: Previous marker in current frame",
            "- DELETE: Delete selected marker",
            "- A: Add new empty marker to file",
            "- R: Remove last marker from file",
        ]

        help_lines_right = [
            "Marker Modes:",
            "- Normal Mode (default): Clicking selects and",
            "  updates the current marker. Use TAB to navigate.",
            "  Each marker keeps its ID across all frames.",
            "",
            "- 1 Line Mode (C key): Creates points in sequence",
            "  in one frame. Each click adds a new marker.",
            "  Use for tracing paths or outlines.",
            "",
            "- Sequential Mode (S key): Each click creates",
            "  a new marker with incrementing IDs. No need",
            "  to select markers first. Only in Normal mode.",
            "",
            "Persistence Mode (P key):",
            "Shows markers from previous frames.",
            "- 1: Decrease persistence frames",
            "- 2: Increase persistence frames",
            "- 3: Toggle full persistence",
            "",
            "Press any key to close this help",
        ]

        # Create semi-transparent overlay
        overlay = pygame.Surface((window_width, window_height + 80))
        overlay.set_alpha(230)
        overlay.fill((0, 0, 0))

        # Render help text in two columns
        font = pygame.font.Font(None, 24)
        line_height = 28

        # Calculate column positions
        col_width = window_width // 2 - 30
        left_col_x = 20
        right_col_x = window_width // 2 + 10

        # Draw left column
        for i, line in enumerate(help_lines_left):
            text_surface = font.render(line, True, (255, 255, 255))
            overlay.blit(text_surface, (left_col_x, 20 + i * line_height))

        # Draw right column
        for i, line in enumerate(help_lines_right):
            text_surface = font.render(line, True, (255, 255, 255))
            overlay.blit(text_surface, (right_col_x, 20 + i * line_height))

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
                        persistence_frames += 1  # Sem limite máximo
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
                        persistence_frames = max(1, persistence_frames - 1)
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
                # Verificar se é um marcador vazio (None)
                if x is not None and y is not None:
                    # Marker indices are 1-based in the CSV
                    row_values[idx * 2 + 1] = float(
                        x
                    )  # +1 for frame column, then multiply by 2 for x position
                    row_values[idx * 2 + 2] = float(y)  # +2 for y position
                # Se for None, deixar como vazio (já inicializado como "")

        df = pd.DataFrame([row_values], columns=header)
        df.to_csv(output_file, index=False)

        print(f"1 line coordinates saved to: {output_file}")
        return output_file

    def add_new_marker():
        """Adiciona um novo marcador vazio após o último marcador visível"""
        nonlocal coordinates, one_line_markers, selected_marker_idx, showing_save_message, save_message_timer, save_message_text

        if one_line_mode:
            # No modo 1 line, encontrar o maior número de marcador visível
            visible_markers = []
            for idx, _ in enumerate(one_line_markers):
                if idx not in deleted_markers:
                    visible_markers.append(idx)

            if visible_markers:
                new_idx = max(visible_markers) + 1
            else:
                new_idx = 0

            # Adicionar marcador vazio no frame atual (usando None em vez de 0,0)
            one_line_markers.append((frame_count, None, None))
            selected_marker_idx = new_idx

            save_message_text = f"Adicionado novo marcador vazio {new_idx+1}"
            showing_save_message = True
            save_message_timer = 60
        else:
            # No modo normal, vamos verificar o número máximo de marcadores visíveis
            max_visible_marker = -1

            for frame in range(total_frames):
                for i in range(len(coordinates[frame])):
                    if i not in deleted_positions[frame]:
                        max_visible_marker = max(max_visible_marker, i)

            new_marker_idx = max_visible_marker + 1

            # Adicionar mais um marcador a cada frame com posição vazia (None, None)
            for frame in range(total_frames):
                while len(coordinates[frame]) <= new_marker_idx:
                    coordinates[frame].append((None, None))

                # Adicione o marcador como "deletado" em todos os frames exceto o atual
                if frame != frame_count:
                    deleted_positions[frame].add(new_marker_idx)

            # Remova o marcador da lista de deletados no frame atual para torná-lo visível
            if new_marker_idx in deleted_positions[frame_count]:
                deleted_positions[frame_count].remove(new_marker_idx)

            # Selecione o novo marcador adicionado
            selected_marker_idx = new_marker_idx

            save_message_text = f"Adicionado novo marcador vazio {new_marker_idx+1}"
            showing_save_message = True
            save_message_timer = 60

        # Fazer backup automático do arquivo original
        make_backup()

    def remove_marker():
        """Remove o marcador selecionado apenas no frame atual"""
        nonlocal coordinates, one_line_markers, selected_marker_idx, showing_save_message, save_message_timer, save_message_text

        if one_line_mode:
            if selected_marker_idx >= 0:
                # Encontre e remova o marcador selecionado apenas no frame atual
                for i, (f_num, _, _) in enumerate(one_line_markers):
                    if i == selected_marker_idx and f_num == frame_count:
                        # Em vez de remover completamente, adicione à lista de marcadores deletados
                        deleted_markers.add(selected_marker_idx)
                        save_message_text = (
                            f"Removido marcador {selected_marker_idx+1} no frame atual"
                        )
                        showing_save_message = True
                        save_message_timer = 60
                        break
            else:
                save_message_text = "Nenhum marcador selecionado para remover"
                showing_save_message = True
                save_message_timer = 60
        else:
            if selected_marker_idx >= 0:
                # Adicione o marcador selecionado à lista de deletados apenas no frame atual
                if selected_marker_idx < len(coordinates[frame_count]):
                    deleted_positions[frame_count].add(selected_marker_idx)
                    save_message_text = (
                        f"Removido marcador {selected_marker_idx+1} no frame atual"
                    )
                    showing_save_message = True
                    save_message_timer = 60
                else:
                    save_message_text = "Marcador não existe neste frame"
                    showing_save_message = True
                    save_message_timer = 60
            else:
                save_message_text = "Nenhum marcador selecionado para remover"
                showing_save_message = True
                save_message_timer = 60

        # Fazer backup automático do arquivo original
        make_backup()

    def make_backup():
        """Faz um backup do arquivo original de coordenadas com timestamp"""
        if not os.path.exists(video_path):
            return

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Verifica se existe um arquivo de coordenadas normal
        coords_file = os.path.join(video_dir, f"{base_name}_markers.csv")
        if os.path.exists(coords_file):
            backup_file = os.path.join(
                video_dir, f"{base_name}_markers_bk_{timestamp}.csv"
            )
            try:
                import shutil

                shutil.copy2(coords_file, backup_file)
                print(f"Backup created: {backup_file}")
            except Exception as e:
                print(f"Erro ao fazer backup: {e}")

        # Verifica se existe arquivo sequential
        seq_file = os.path.join(video_dir, f"{base_name}_markers_sequential.csv")
        if os.path.exists(seq_file):
            backup_file = os.path.join(
                video_dir, f"{base_name}_markers_sequential_bk_{timestamp}.csv"
            )
            try:
                import shutil

                shutil.copy2(seq_file, backup_file)
                print(f"Sequential backup created: {backup_file}")
            except Exception as e:
                print(f"Erro ao fazer backup sequential: {e}")

        # Também verifica o arquivo de 1 line
        line_file = os.path.join(video_dir, f"{base_name}_markers_1_line.csv")
        if os.path.exists(line_file):
            backup_file = os.path.join(
                video_dir, f"{base_name}_markers_1_line_bk_{timestamp}.csv"
            )
            try:
                import shutil

                shutil.copy2(line_file, backup_file)
                print(f"1-Line backup created: {backup_file}")
            except Exception as e:
                print(f"Erro ao fazer backup 1-line: {e}")

    def reload_coordinates():
        """Carrega um novo arquivo de coordenadas durante a execução"""
        nonlocal coordinates, one_line_markers, deleted_markers, deleted_positions, selected_marker_idx

        # Fazer backup do atual antes de carregar um novo
        make_backup()

        # Criar uma nova instância do Tkinter para o diálogo de arquivo
        root = Tk()
        root.withdraw()
        input_file = filedialog.askopenfilename(
            title="Selecionar Arquivo de Keypoints",
            filetypes=[("Arquivos CSV", "*.csv")],
        )
        if not input_file:
            save_message_text = "Carregamento cancelado."
            showing_save_message = True
            save_message_timer = 60
            return

        try:
            # Verificar se é arquivo de 1 line ou normal
            df = pd.read_csv(input_file)
            if "_1_line" in input_file or len(df) == 1:
                # Provavelmente é um arquivo de 1 line
                one_line_markers = []
                deleted_markers = set()

                for _, row in df.iterrows():
                    frame_num = int(row["frame"])
                    for i in range(1, 101):  # Assumindo no máximo 100 pontos
                        x_col = f"p{i}_x"
                        y_col = f"p{i}_y"
                        if x_col in df.columns and y_col in df.columns:
                            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                                one_line_markers.append(
                                    (frame_num, row[x_col], row[y_col])
                                )

                save_message_text = (
                    f"Carregado arquivo de 1 line: {os.path.basename(input_file)}"
                )
                # Se estava no modo normal, alternar para o modo 1 line
                one_line_mode = True
            else:
                # Arquivo de coordenadas normal
                coordinates = {i: [] for i in range(total_frames)}
                deleted_positions = {i: set() for i in range(total_frames)}

                for _, row in df.iterrows():
                    frame_num = int(row["frame"])
                    for i in range(1, 101):  # Assumindo no máximo 100 pontos
                        x_col = f"p{i}_x"
                        y_col = f"p{i}_y"
                        if x_col in df.columns and y_col in df.columns:
                            if pd.notna(row[x_col]) and pd.notna(row[y_col]):
                                coordinates[frame_num].append((row[x_col], row[y_col]))

                save_message_text = f"Carregado arquivo: {os.path.basename(input_file)}"
                # Se estava no modo 1 line, alternar para o modo normal
                one_line_mode = False

            # Inicializar sempre no primeiro marcador (índice 0)
            selected_marker_idx = 0

            showing_save_message = True
            save_message_timer = 90

        except Exception as e:
            save_message_text = f"Erro ao carregar arquivo: {e}"
            showing_save_message = True
            save_message_timer = 90

    running = True
    saved = False
    showing_save_message = False
    save_message_timer = 0
    save_message_text = ""

    while running:
        if paused:
            # Quando pausado, vamos usar o método set para posicionar no frame exato
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
        else:
            # Quando em reprodução, apenas leia o próximo frame sem reposicionar
            ret, frame = cap.read()
            if ret:
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                # Final do vídeo alcançado, reiniciar
                frame_count = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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

            # Collect marker positions across frames for each ID
            marker_trails = (
                {}
            )  # Dictionary: marker_id -> list of (frame_num, x, y) points

            if one_line_mode:
                # In one_line mode, collect marker positions by their index
                for idx, (f_num, x, y) in enumerate(one_line_markers):
                    if idx in deleted_markers:
                        continue  # Skip deleted markers

                    if start_frame <= f_num <= frame_count:
                        if idx not in marker_trails:
                            marker_trails[idx] = []
                        # Store with frame number to sort by time later
                        marker_trails[idx].append((f_num, x, y))
            else:
                # In regular mode, need to track same marker ID across frames
                for f_num in range(start_frame, frame_count + 1):
                    for i, (x, y) in enumerate(coordinates[f_num]):
                        if i in deleted_positions[f_num]:
                            continue  # Skip deleted markers

                        if i not in marker_trails:
                            marker_trails[i] = []
                        marker_trails[i].append((f_num, x, y))

            # Draw trails for each marker
            for marker_id, positions in marker_trails.items():
                # Sort positions by frame number
                positions.sort(key=lambda p: p[0])

                # Need at least 2 points to draw a line
                if len(positions) < 2:
                    continue

                # Get marker's color - we'll base trail color on the marker's color
                color = get_color_for_id(marker_id)

                # Draw line segments with decreasing opacity
                for i in range(1, len(positions)):
                    # Calculate opacity based on how recent this segment is
                    segment_opacity = int(
                        200
                        * (positions[i][0] - start_frame)
                        / (frame_count - start_frame + 1)
                    )

                    # Get screen coordinates for this segment
                    prev_frame, prev_x, prev_y = positions[i - 1]
                    curr_frame, curr_x, curr_y = positions[i]

                    # Skip if any coordinates are None
                    if (
                        prev_x is None
                        or prev_y is None
                        or curr_x is None
                        or curr_y is None
                    ):
                        continue

                    prev_screen_x = int((prev_x * zoom_level) - crop_x)
                    prev_screen_y = int((prev_y * zoom_level) - crop_y)
                    curr_screen_x = int((curr_x * zoom_level) - crop_x)
                    curr_screen_y = int((curr_y * zoom_level) - crop_y)

                    # Create a temporary surface for the semi-transparent line
                    # Make it large enough to contain the line with padding
                    min_x = min(prev_screen_x, curr_screen_x) - 2
                    min_y = min(prev_screen_y, curr_screen_y) - 2
                    width = abs(curr_screen_x - prev_screen_x) + 4
                    height = abs(curr_screen_y - prev_screen_y) + 4

                    # Handle zero dimensions
                    width = max(width, 1)
                    height = max(height, 1)

                    line_surface = pygame.Surface((width, height), pygame.SRCALPHA)

                    # Line color with opacity
                    line_color = (color[0], color[1], color[2], segment_opacity)

                    # Draw the line segment on the surface
                    pygame.draw.line(
                        line_surface,
                        line_color,
                        (prev_screen_x - min_x, prev_screen_y - min_y),
                        (curr_screen_x - min_x, curr_screen_y - min_y),
                        max(3, int(6 * segment_opacity / 200)),  # Linha mais espessa
                    )

                    # Blit the line segment to the screen
                    screen.blit(line_surface, (min_x, min_y))

                    # Desenhar ponto no início do segmento (ponto adicional)
                    point_surface = pygame.Surface((8, 8), pygame.SRCALPHA)
                    point_color = (
                        0,
                        0,
                        0,
                        segment_opacity,
                    )  # Cor preta com a mesma opacidade da linha
                    pygame.draw.circle(
                        point_surface, point_color, (4, 4), 2
                    )  # Círculo de raio 2
                    screen.blit(point_surface, (prev_screen_x - 4, prev_screen_y - 4))

                # Still draw the most recent point as a small circle
                last_frame, last_x, last_y = positions[-1]
                if (
                    last_frame < frame_count
                    and last_x is not None
                    and last_y is not None
                ):  # Don't draw current frame again and check for None
                    last_screen_x = int((last_x * zoom_level) - crop_x)
                    last_screen_y = int((last_y * zoom_level) - crop_y)

                    # Draw a small circle for the most recent position
                    pygame.draw.circle(screen, color, (last_screen_x, last_screen_y), 2)

        # Draw current frame markers
        if one_line_mode:
            for idx, (f_num, x, y) in enumerate(one_line_markers):
                if f_num == frame_count:
                    # Skip rendering if marker is empty/None
                    if x is None or y is None:
                        continue

                    screen_x = int((x * zoom_level) - crop_x)
                    screen_y = int((y * zoom_level) - crop_y)

                    # Highlight selected marker only in one_line_mode
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
                    continue

                if x is None or y is None:
                    continue

                screen_x = int((x * zoom_level) - crop_x)
                screen_y = int((y * zoom_level) - crop_y)

                # Adicionar destaque para o marcador selecionado
                if i == selected_marker_idx:
                    # Desenhar círculo laranja mais amplo como destaque
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
            load_button_rect,
            seq_button_rect,  # Add sequential button to return
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
                            video_path, one_line_markers, deleted_markers
                        )
                    else:
                        output_file = save_coordinates(
                            video_path,
                            coordinates,
                            total_frames,
                            deleted_positions,
                            is_sequential=sequential_mode,
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
                        persistence_frames += 1  # Sem limite máximo
                        save_message_text = f"Persistence: {persistence_frames} frames"
                        showing_save_message = True
                        save_message_timer = 30

                elif event.key == pygame.K_3:  # Alternar entre três modos
                    if not persistence_enabled:
                        # Modo 1: Ativar com persistência completa
                        persistence_enabled = True
                        persistence_frames = total_frames
                        save_message_text = "Full persistence enabled"
                    elif persistence_frames == total_frames:
                        # Modo 2: Mudar de full para número específico de frames
                        persistence_frames = 10
                        save_message_text = f"Persistence: {persistence_frames} frames"
                    else:
                        # Modo 3: Desativar completamente
                        persistence_enabled = False
                        save_message_text = "Persistence disabled"

                    showing_save_message = True
                    save_message_timer = 30

                # Adicionar novo marcador
                elif event.key == pygame.K_a:
                    add_new_marker()

                # Remover marcador
                elif event.key == pygame.K_r:
                    remove_marker()

                # Atualize a função de teclado para usar a tecla 'd' para remover marcadores
                elif event.key == pygame.K_d:
                    remove_marker()

                # Add sequential mode toggle with 'o' key
                elif event.key == pygame.K_o:  # Toggle sequential mode with 'o' key
                    if not one_line_mode:  # Only toggle if not in one-line mode
                        sequential_mode = not sequential_mode
                        save_message_text = f"Sequential mode {'enabled' if sequential_mode else 'disabled'}"
                        showing_save_message = True
                        save_message_timer = 30

                # Add sequential mode toggle with 's' key
                elif event.key == pygame.K_s:  # Toggle sequential mode with 's' key
                    if not one_line_mode:  # Only toggle if not in one-line mode
                        sequential_mode = not sequential_mode
                        save_message_text = f"Sequential mode {'enabled' if sequential_mode else 'disabled'}"
                        showing_save_message = True
                        save_message_timer = 30

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if y >= window_height:
                    # Clique na área de controles
                    rel_y = y - window_height
                    if load_button_rect.collidepoint(x, rel_y):
                        # Carregar novo arquivo
                        reload_coordinates()
                    elif one_line_button_rect.collidepoint(x, rel_y):
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
                                video_path,
                                coordinates,
                                total_frames,
                                deleted_positions,
                                is_sequential=sequential_mode,
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
                    elif seq_button_rect.collidepoint(x, rel_y):
                        if not one_line_mode:  # Only toggle if not in one-line mode
                            sequential_mode = not sequential_mode
                            save_message_text = f"Sequential mode {'enabled' if sequential_mode else 'disabled'}"
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
                        if one_line_mode:
                            # Simply append the new marker
                            one_line_markers.append((frame_count, video_x, video_y))
                        else:
                            if sequential_mode:
                                # Find the next available marker index
                                next_idx = len(coordinates[frame_count])
                                coordinates[frame_count].append((video_x, video_y))
                                selected_marker_idx = (
                                    next_idx  # Auto-select the new marker
                                )
                            else:
                                # Use existing marker selection logic
                                if selected_marker_idx >= 0:
                                    # Update existing marker
                                    while (
                                        len(coordinates[frame_count])
                                        <= selected_marker_idx
                                    ):
                                        coordinates[frame_count].append((None, None))
                                    coordinates[frame_count][selected_marker_idx] = (
                                        video_x,
                                        video_y,
                                    )
                                    if (
                                        selected_marker_idx
                                        in deleted_positions[frame_count]
                                    ):
                                        deleted_positions[frame_count].remove(
                                            selected_marker_idx
                                        )
                                else:
                                    # Add new marker at the end
                                    coordinates[frame_count].append((video_x, video_y))
                                    selected_marker_idx = (
                                        len(coordinates[frame_count]) - 1
                                    )

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

        if paused:
            # Se pausado, não limitamos a taxa de FPS para que a interface seja responsiva
            clock.tick(60)  # Taxa de atualização da interface
        else:
            # Se em reprodução, limitamos à taxa de FPS do vídeo
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

        # Determinar o número máximo de marcadores no arquivo
        max_marker = 0
        for i in range(1, 101):  # Limite máximo de 100 marcadores
            if f"p{i}_x" in df.columns:
                max_marker = i

        for _, row in df.iterrows():
            frame_num = int(row["frame"])

            # Garantir que todos os frames tenham o mesmo número de marcadores
            for i in range(1, max_marker + 1):
                x_col = f"p{i}_x"
                y_col = f"p{i}_y"

                if x_col in df.columns and y_col in df.columns:
                    # Se x ou y for NaN/vazio, ambos serão considerados None
                    x_val = row.get(x_col)
                    y_val = row.get(y_col)

                    if pd.isna(x_val) or pd.isna(y_val):
                        coordinates[frame_num].append((None, None))
                    else:
                        coordinates[frame_num].append((x_val, y_val))

        print(f"Coordinates successfully loaded from: {input_file}")
        return coordinates
    except Exception as e:
        print(f"Error loading coordinates from {input_file}: {e}. Starting fresh.")
        return {i: [] for i in range(total_frames)}


def save_coordinates(
    video_path, coordinates, total_frames, deleted_positions=None, is_sequential=False
):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)

    # Create different filenames based on the mode
    if is_sequential:
        output_file = os.path.join(video_dir, f"{base_name}_markers_sequential.csv")
    else:
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

    # Preenche o DataFrame com os pontos marcados
    for frame_num, points in coordinates.items():
        for i, (x, y) in enumerate(points):
            if i not in deleted_positions[frame_num]:  # Only save non-deleted markers
                # Verificar se é um marcador vazio (None)
                if x is not None and y is not None:
                    df.at[frame_num, f"p{i+1}_x"] = float(x)
                    df.at[frame_num, f"p{i+1}_y"] = float(y)
                # Se for None, deixar como NaN (o que se tornará "" no CSV)

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


def run_getpixelvideo():
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
    run_getpixelvideo()