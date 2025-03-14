"""
================================================================================
Video Cutting Tool - cutvideo.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 24 January 2025
Version: 0.0.2
Python Version: 3.12.8

Description:
------------
This tool enables marking start and end frames for video cutting/trimming with
frame-by-frame navigation. Users can mark multiple segments and generate new
videos for each marked segment. Cuts can be saved to a text file and loaded
later for video generation.

Controls:
---------
- Space: Play/Pause
- Right Arrow: Next Frame
- Left Arrow: Previous Frame
- Up Arrow: Fast Forward (60 frames)
- Down Arrow: Rewind (60 frames)
- S: Mark start frame for cut
- E: Mark end frame for cut
- R: Reset current cut
- DELETE: Remove last cut
- L: List all cuts
- ESC: Save cuts to file and optionally generate videos

Features:
---------
- Multiple cuts before generating videos
- Save cuts to text file for later use
- Load existing cuts from text file
- Visual feedback for current cut and total cuts
- Resizable window with maintained aspect ratio
================================================================================
"""

import os
import pygame
import cv2
import datetime
from tkinter import Tk, filedialog, messagebox
from pathlib import Path


def save_cuts_to_txt(video_path, cuts):
    """Save cuts information to a text file."""
    video_name = Path(video_path).stem
    txt_path = Path(video_path).parent / f"{video_name}_cuts.txt"

    with open(txt_path, "w") as f:
        f.write(f"Cuts for video: {video_name}\n")
        f.write(f"Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 50 + "\n")
        for i, (start, end) in enumerate(cuts, 1):
            f.write(f"Cut {i}: Frame {start} to {end}\n")

    return txt_path


def load_cuts_from_txt(video_path):
    """Load existing cuts from a text file if it exists."""
    video_name = Path(video_path).stem
    txt_path = Path(video_path).parent / f"{video_name}_cuts.txt"

    cuts = []
    if txt_path.exists():
        with open(txt_path, "r") as f:
            lines = f.readlines()[3:]  # Skip header lines
            for line in lines:
                if line.strip():
                    # Extract start and end frames from line
                    parts = line.split("Frame ")[1].split(" to ")
                    start = int(parts[0])
                    end = int(parts[1])
                    cuts.append((start, end))

    return cuts


def play_video_with_cuts(video_path):
    pygame.init()

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize window with adjusted size
    screen_info = pygame.display.Info()
    max_width = screen_info.current_w - 100  # Leave some margin
    max_height = screen_info.current_h - 100  # Leave some margin

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate window size maintaining aspect ratio
    if original_width > max_width:
        window_width = max_width
        window_height = int(window_width / aspect_ratio)
    elif original_height > max_height:
        window_height = max_height
        window_width = int(window_height * aspect_ratio)
    else:
        window_width = original_width
        window_height = original_height

    # Ensure minimum size
    window_width = max(640, min(window_width, max_width))
    window_height = max(480, min(window_height, max_height))

    # Initialize window
    screen = pygame.display.set_mode(
        (window_width, window_height + 80), pygame.RESIZABLE
    )
    pygame.display.set_caption(
        "Space:Play/Pause | ←→:Frame | S:Start | E:End | R:Reset | DEL:Remove | L:List | ESC:Save"
    )

    # Initialize variables
    clock = pygame.time.Clock()
    frame_count = 0
    paused = True
    cuts = []  # List to store (start, end) frame pairs
    current_start = None

    # Load existing cuts if available
    cuts = load_cuts_from_txt(video_path)
    # Flag to track if cuts were loaded from a sync file
    using_sync_file = len(cuts) > 0
    if using_sync_file:
        print(f"Loaded {len(cuts)} cuts from synchronization file")

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

        # Draw frame information and cut markers
        font = pygame.font.Font(None, 24)
        frame_text = font.render(
            f"Frame: {frame_count + 1}/{total_frames}", True, (255, 255, 255)
        )
        slider_surface.blit(frame_text, (10, 10))

        # Draw current cut information
        if current_start is not None:
            cut_text = font.render(
                f"Current Cut Start: {current_start}", True, (0, 255, 0)
            )
            slider_surface.blit(cut_text, (10, 50))

        # Draw number of cuts
        cuts_text = font.render(f"Cuts: {len(cuts)}", True, (255, 255, 255))
        slider_surface.blit(cuts_text, (window_width - 100, 50))

        # Draw help button
        help_button_rect = pygame.Rect(window_width - 70, 10, 60, 25)
        pygame.draw.rect(slider_surface, (100, 100, 100), help_button_rect)
        help_text = font.render("Help", True, (255, 255, 255))
        text_rect = help_text.get_rect(center=help_button_rect.center)
        slider_surface.blit(help_text, text_rect)

        screen.blit(slider_surface, (0, window_height))
        return slider_x, slider_width, slider_y, slider_height, help_button_rect

    def show_help_dialog():
        help_message = (
            "Video Cutting Controls:\n\n"
            "Navigation:\n"
            "- Space: Play/Pause\n"
            "- Right Arrow: Next Frame\n"
            "- Left Arrow: Previous Frame\n"
            "- Up Arrow: Fast Forward (60 frames)\n"
            "- Down Arrow: Rewind (60 frames)\n\n"
            "Cutting Operations:\n"
            "- S: Mark Start Frame\n"
            "- E: Mark End Frame\n"
            "- R: Reset Current Cut\n"
            "- DELETE: Remove Last Cut\n"
            "- L: List All Cuts\n\n"
            "File Operations:\n"
            "- ESC: Save cuts to file and optionally generate videos\n\n"
            "Mouse Controls:\n"
            "- Click on slider: Jump to frame\n"
            "- Click 'Help': Show this dialog\n"
            "- Drag window edges: Resize window"
        )
        messagebox.showinfo("Help", help_message)

    def save_and_generate_videos():
        if not cuts:
            messagebox.showinfo("Info", "No cuts were marked!")
            return False

        # First save cuts to text file
        txt_path = save_cuts_to_txt(video_path, cuts)

        # Garantir que o pygame está fechado
        if pygame.get_init():
            pygame.quit()
            print("Pygame closed before video processing")

        # Ask if user wants to generate videos now
        if messagebox.askyesno(
            "Generate Videos",
            "Cuts saved to text file. Do you want to generate video files now?",
        ):
            success = save_cuts(video_path, cuts, using_sync_file)

            # Ask if user wants to apply the same cuts to all videos in the directory
            if success and messagebox.askyesno(
                "Batch Processing",
                "Do you want to apply these same cuts to all other videos in this directory?",
            ):
                batch_process_videos(video_path, cuts, using_sync_file)

            return success
        return True

    def batch_process_videos(source_video_path, cuts, from_sync_file=False):
        """Apply the same cuts to all videos in the same directory."""
        if not cuts:
            messagebox.showinfo("Info", "No cuts to apply!")
            return

        # Get the directory of the source video
        source_dir = Path(source_video_path).parent
        source_name = Path(source_video_path).name

        # Get all video files in the directory
        video_extensions = [
            ".mp4",
            ".MP4",
            ".avi",
            ".AVI",
            ".mov",
            ".MOV",
            ".mkv",
            ".MKV",
        ]
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(source_dir.glob(f"*{ext}")))

        # Remove the source video from the list
        video_files = [v for v in video_files if v.name != source_name]

        if not video_files:
            messagebox.showinfo("Info", "No other video files found in this directory.")
            return

        # Create a progress dialog
        root = Tk()
        root.title("Batch Processing")
        root.geometry("400x150")

        from tkinter import ttk

        label = ttk.Label(root, text="Processing videos in batch...")
        label.pack(pady=10)

        progress = ttk.Progressbar(
            root, orient="horizontal", length=300, mode="determinate"
        )
        progress.pack(pady=10)
        progress["maximum"] = len(video_files)

        status_label = ttk.Label(root, text="")
        status_label.pack(pady=5)

        # Create output directory with improved naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "sync_" if from_sync_file else ""
        output_dir = source_dir / f"vailacut_{prefix}batch_{timestamp}"
        output_dir.mkdir(exist_ok=True)

        processed_count = 0

        def process_next_video():
            nonlocal processed_count

            if processed_count < len(video_files):
                video_path = str(video_files[processed_count])
                video_name = Path(video_path).stem

                # Salvar informações de corte para cada vídeo processado
                save_cuts_to_txt(video_path, cuts)

                status_label.config(text=f"Processing: {video_name}")

                try:
                    # Get video properties
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        status_label.config(text=f"Error opening: {video_name}")
                        root.after(100, process_next_video)
                        processed_count += 1
                        progress["value"] = processed_count
                        return

                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Process each cut
                    for i, (start_frame, end_frame) in enumerate(cuts):
                        # Skip if end frame is beyond video length
                        if start_frame >= total_frames:
                            continue

                        # Adjust end frame if needed
                        actual_end_frame = min(end_frame, total_frames - 1)

                        output_path = (
                            output_dir
                            / f"{video_name}_frame_{start_frame}_to_{actual_end_frame}.mp4"
                        )
                        out = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height),
                        )

                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        for _ in range(actual_end_frame - start_frame + 1):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            out.write(frame)

                        out.release()

                    cap.release()

                except Exception as e:
                    status_label.config(text=f"Error processing {video_name}: {str(e)}")

                processed_count += 1
                progress["value"] = processed_count
                root.after(100, process_next_video)
            else:
                status_label.config(text="Batch processing complete!")
                root.after(2000, root.destroy)

        # Start processing
        root.after(100, process_next_video)
        root.mainloop()

        messagebox.showinfo(
            "Batch Complete",
            f"Processed {processed_count} videos. Output saved to {output_dir}",
        )

    def save_cuts(video_path, cuts, from_sync_file=False):
        if not cuts:
            messagebox.showinfo("Info", "No cuts were marked!")
            return False

        # Create output directory with improved naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        prefix = "sync_" if from_sync_file else ""
        output_dir = Path(video_path).parent / f"vailacut_{prefix}{timestamp}"
        output_dir.mkdir(exist_ok=True)

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Process each cut
        for i, (start_frame, end_frame) in enumerate(cuts):
            output_path = (
                output_dir / f"{video_name}_frame_{start_frame}_to_{end_frame}.mp4"
            )
            out = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame + 1):
                ret, frame = cap.read()
                if ret:
                    out.write(frame)

            out.release()

        cap.release()
        return True

    running = True
    while running:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to fit window while maintaining aspect ratio
        frame_height = window_height
        frame_width = window_width
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Convert frame to pygame surface
        frame_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )
        screen.blit(frame_surface, (0, 0))

        # Draw controls
        slider_x, slider_width, slider_y, slider_height, help_button_rect = (
            draw_controls()
        )
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                if new_h > 80:
                    # Maintain minimum size
                    window_width = max(640, new_w)
                    window_height = max(480, new_h - 80)
                    screen = pygame.display.set_mode(
                        (window_width, window_height + 80), pygame.RESIZABLE
                    )

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if save_and_generate_videos():
                        messagebox.showinfo(
                            "Success",
                            "Cuts saved to text file and videos generated (if selected)!",
                        )
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
                elif event.key == pygame.K_s and paused:
                    current_start = frame_count
                    print(
                        f"Start frame marked: {frame_count}"
                    )  # Feedback apenas no terminal
                elif event.key == pygame.K_e and paused and current_start is not None:
                    if frame_count > current_start:
                        cuts.append((current_start, frame_count))
                        print(
                            f"Cut marked: {current_start} to {frame_count}"
                        )  # Feedback apenas no terminal
                        current_start = None
                    else:
                        print(
                            "Error: End frame must be after start frame!"
                        )  # Feedback apenas no terminal
                elif event.key == pygame.K_r:  # Reset current cut
                    current_start = None
                    print("Current cut reset")  # Mudado para terminal
                elif event.key == pygame.K_DELETE:  # Delete last cut
                    if cuts:
                        cuts.pop()
                        print("Last cut removed")  # Mudado para terminal
                elif event.key == pygame.K_l:  # List all cuts
                    if cuts:
                        cuts_info = "\n".join(
                            [
                                f"Cut {i+1}: Frame {start} to {end}"
                                for i, (start, end) in enumerate(cuts)
                            ]
                        )
                        messagebox.showinfo(
                            "Cuts List", cuts_info
                        )  # Mantém a janela apenas para listar
                    else:
                        messagebox.showinfo("Cuts List", "No cuts marked yet")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if help_button_rect.collidepoint(x, y - window_height):
                    show_help_dialog()
                elif slider_y <= y - window_height <= slider_y + slider_height:
                    rel_x = x - slider_x
                    frame_count = int((rel_x / slider_width) * total_frames)
                    frame_count = max(0, min(frame_count, total_frames - 1))
                    paused = True

        if not paused:
            frame_count = (frame_count + 1) % total_frames

        clock.tick(fps)

    cap.release()
    pygame.quit()


def get_video_path():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV")],
    )
    return video_path


def main():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting cutvideo.py...")

    video_path = get_video_path()
    if not video_path:
        print("No video selected. Exiting.")
        return

    play_video_with_cuts(video_path)


if __name__ == "__main__":
    main()
