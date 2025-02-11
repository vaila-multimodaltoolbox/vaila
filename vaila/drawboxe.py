"""
drawboxe.py

Description:
-----------
This script is designed to add bounding boxes to videos using coordinates
obtained from clicks on an image. It also supports extracting frames and
applying boxes to specific frame intervals or directly to videos. The script
can be used for batch processing of videos in a directory.

Version:
--------
0.0.2
date: 2025-01-22

Author:
-------
Prof. PhD. Paulo Santiago

License:
--------
This code is licensed under the MIT License. See the LICENSE file for more details.

Version History:
----------------
- v1.0 (2024-08-12): Initial version with support for adding boxes to videos and extracting frames.
- v0.0.2 (2025-01-22): Added draw shape trapezoid.

Contact:
--------
For questions or contributions, please contact the author at: paulo.santiago@example.com.

Contributions:
--------------
Contributions are welcome. Please follow the contribution guidelines provided in the CONTRIBUTING.md file of this repository.

Dependencies:
-------------
- Python 3.12.8 (Anaconda environment)
- os
- ffmpeg (installed via Conda or available in PATH)
- matplotlib
- opencv-python
- tkinter

Additional Notes:
-----------------
- Ensure FFMPEG is installed on your system for this script to function correctly.
- The script assumes that the input videos are in a format supported by OpenCV.
"""

import numpy as np
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import time
import shutil


def save_first_frame(video_path, frame_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(frame_path, image)
    vidcap.release()


def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    command = ["ffmpeg", "-i", video_path, os.path.join(frames_dir, "frame_%09d.png")]
    subprocess.run(command, check=True)


def apply_boxes_directly_to_video(input_path, output_path, coordinates, selections):
    vidcap = cv2.VideoCapture(input_path)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = vidcap.read()
        if not ret:
            break

        for coords, (mode, shape_type) in zip(coordinates, selections):
            if shape_type == "rectangle":
                # Usar apenas os pontos inicial e final para o retângulo
                x1, y1 = int(coords[0][0]), int(coords[0][1])
                x2, y2 = int(coords[2][0]), int(coords[2][1])

                # Garantir que x1,y1 seja o ponto superior esquerdo
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                if mode == "outside":
                    # Criar uma cópia do frame preenchida com preto
                    black_frame = np.zeros_like(frame)
                    # Copiar apenas a região do retângulo do frame original
                    black_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    frame = black_frame
                else:  # inside mode
                    # Desenhar retângulo preto diretamente no frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

            elif shape_type == "trapezoid":
                pts = np.array(coords, np.int32)
                if mode == "inside":
                    cv2.fillPoly(frame, [pts], (0, 0, 0))

        out.write(frame)
        frame_count += 1
        print(
            f"Processed {frame_count}/{total_frames} frames for {os.path.basename(input_path)}",
            end="\r",
        )

    print(f"\nCompleted processing: {os.path.basename(input_path)}")
    print(f"Saved to: {output_path}")
    out.release()
    vidcap.release()


def apply_boxes_to_frames(frames_dir, coordinates, selections, frame_intervals):
    for filename in sorted(os.listdir(frames_dir)):
        frame_number = int(filename.split("_")[1].split(".")[0])
        for start_frame, end_frame in frame_intervals:
            if start_frame <= frame_number <= end_frame:
                frame_path = os.path.join(frames_dir, filename)
                img = cv2.imread(frame_path)
                for coords, (mode, shape_type) in zip(coordinates, selections):
                    if shape_type == "rectangle":
                        # Usar apenas os pontos inicial e final para o retângulo
                        x1, y1 = int(coords[0][0]), int(coords[0][1])
                        x2, y2 = int(coords[2][0]), int(coords[2][1])

                        # Garantir que x1,y1 seja o ponto superior esquerdo
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)

                        if mode == "outside":
                            mask = np.zeros_like(img)
                            cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
                            mask = cv2.bitwise_not(mask)
                            img = cv2.bitwise_and(img, mask)
                        else:  # inside mode
                            # Desenhar diretamente o retângulo preenchido
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    elif shape_type == "trapezoid":
                        pts = np.array(coords, np.int32)
                        if mode == "inside":
                            cv2.fillPoly(img, [pts], (0, 0, 0))

                cv2.imwrite(frame_path, img)


def reassemble_video(frames_dir, output_path, fps):
    command = [
        "ffmpeg",
        "-framerate",
        str(fps),
        "-i",
        os.path.join(frames_dir, "frame_%09d.png"),
        "-c:v",
        "libx264",
        output_path,
    ]
    subprocess.run(command, check=True)


def clean_up(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)
    os.rmdir(directory)


def get_box_coordinates(image_path):
    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    selection_mode = {"mode": "inside", "shape": "rectangle"}  # Added shape type

    def update_title():
        shape_text = (
            "trapezoid" if selection_mode["shape"] == "trapezoid" else "rectangle"
        )
        ax.set_title(
            f'Red box: inside, Blue box: outside\nCurrent mode: {selection_mode["mode"]}, Shape: {shape_text}\n'
            'Click to select corners. Press "e" to toggle mode, "t" to toggle shape, Enter to finish.'
        )
        fig.canvas.draw()

    ax.imshow(img)
    update_title()

    points = []
    shapes = []
    selections = []
    temp_points = []

    def on_key(event):
        if event.key == "e":  # Toggle mode
            selection_mode["mode"] = (
                "outside" if selection_mode["mode"] == "inside" else "inside"
            )
            update_title()
        elif event.key == "t":  # Toggle shape type
            selection_mode["shape"] = (
                "trapezoid" if selection_mode["shape"] == "rectangle" else "rectangle"
            )
            temp_points.clear()  # Clear temporary points when switching modes
            update_title()
        elif event.key == "enter":  # Close the window
            plt.close()

    def on_click(event):
        nonlocal points, shapes, selections, temp_points

        if event.button == 3:  # Right mouse button to remove the last shape
            if shapes:
                shapes[-1].remove()
                shapes.pop()
                selections.pop()
                points = points[:-4] if points else []
                temp_points.clear()
                plt.draw()
            return

        if event.button == 1:  # Left mouse button to add a point
            if selection_mode["shape"] == "rectangle":
                temp_points.append((event.xdata, event.ydata))
                if len(temp_points) == 2:
                    # Calculate all 4 corners of rectangle
                    x1, y1 = temp_points[0]
                    x2, y2 = temp_points[1]
                    rect_points = [
                        (x1, y1),  # Top-left
                        (x2, y1),  # Top-right
                        (x2, y2),  # Bottom-right
                        (x1, y2),  # Bottom-left
                    ]
                    points.extend(rect_points)

                    color = "b" if selection_mode["mode"] == "outside" else "r"
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(rect)
                    shapes.append(rect)
                    selections.append((selection_mode["mode"], "rectangle"))
                    temp_points.clear()
                    plt.draw()
            else:
                # Trapezoid logic
                temp_points.append((event.xdata, event.ydata))
                if len(temp_points) == 4:
                    color = "b" if selection_mode["mode"] == "outside" else "r"
                    trap = patches.Polygon(
                        temp_points,
                        linewidth=1,
                        edgecolor=color,
                        facecolor="none",
                    )
                    ax.add_patch(trap)
                    shapes.append(trap)
                    points.extend(temp_points)
                    selections.append((selection_mode["mode"], "trapezoid"))
                    temp_points.clear()
                    plt.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if temp_points:  # Check for incomplete shapes
        raise ValueError("An incomplete shape was defined.")

    # Convert points to the format expected by the video processing functions
    boxes = []
    for i in range(0, len(points), 4):
        if i + 3 < len(points):
            box_points = [
                (int(points[j][0]), int(points[j][1])) for j in range(i, i + 4)
            ]
            boxes.append(box_points)

    return boxes, selections


def load_frame_intervals(file_path):
    intervals = []
    with open(file_path, "r") as file:
        for line in file:
            start, end = map(int, line.strip().split(","))
            intervals.append((start, end))
    return intervals


def show_feedback_message():
    print("vailá!")
    time.sleep(2)  # Simulate processing time


def run_drawboxe():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = tk.Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos"
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    video_files = sorted(
        [
            f
            for f in os.listdir(video_directory)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )

    if not video_files:
        messagebox.showerror("Error", "No video files found in the selected directory.")
        return

    first_video = video_files[0]
    first_frame_path = os.path.join(video_directory, "first_frame.jpg")
    save_first_frame(os.path.join(video_directory, first_video), first_frame_path)

    coordinates, selections = get_box_coordinates(first_frame_path)
    os.remove(first_frame_path)

    use_intervals = messagebox.askyesno(
        "Frame Intervals", "Do you want to use frame intervals from a .txt file?"
    )
    frame_intervals = None
    if use_intervals:
        intervals_file = filedialog.askopenfilename(
            title="Select the .txt file with frame intervals",
            filetypes=[("Text files", "*.txt")],
        )
        if intervals_file:
            frame_intervals = load_frame_intervals(intervals_file)
        else:
            messagebox.showerror("Error", "No .txt file selected.")
            return

    output_dir = os.path.join(video_directory, "video_2_drawbox")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete the existing directory and its contents

    os.makedirs(output_dir, exist_ok=True)

    for video_file in video_files:
        input_path = os.path.join(video_directory, video_file)
        final_output_path = os.path.join(
            output_dir, f"{os.path.splitext(video_file)[0]}_dbox.mp4"
        )

        vidcap = cv2.VideoCapture(input_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidcap.release()

        if frame_intervals:
            frames_dir = os.path.join(video_directory, "frames_temp")
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            extract_frames(input_path, frames_dir)
            apply_boxes_to_frames(frames_dir, coordinates, selections, frame_intervals)
            reassemble_video(frames_dir, final_output_path, fps)
            clean_up(frames_dir)
        else:
            apply_boxes_directly_to_video(
                input_path, final_output_path, coordinates, selections
            )

    show_feedback_message()
    print("All videos processed and saved to the output directory.")
    messagebox.showinfo("Completed", "All videos have been processed successfully!")


if __name__ == "__main__":
    run_drawboxe()
