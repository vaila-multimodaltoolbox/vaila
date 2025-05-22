"""
drawboxe.py

Description:
-----------
This script is designed to add bounding boxes to videos using coordinates
obtained from clicks on an image. It also supports extracting frames and
applying boxes to specific frame intervals or directly to videos. The script
can be used for batch processing of videos in a directory.

Version: 0.0.5
created: 2025-02-28
updated: 2025-05-18 (modes logic fully fixed)

Author:
-------
Prof. PhD. Paulo Santiago

License:
--------
This code is licensed under the MIT License. See the LICENSE file for more details.

Dependencies:
-------------
- Python 3.12.9 (Anaconda environment)
- os
- ffmpeg (installed via Conda or available in PATH)
- matplotlib
- opencv-python
- tkinter

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
import datetime


def save_first_frame(video_path, frame_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(frame_path, image)
    vidcap.release()


def extract_frames(video_path, frames_dir):
    try:
        os.makedirs(frames_dir, exist_ok=True)
        video_path = os.path.normpath(os.path.abspath(video_path))
        frames_dir = os.path.normpath(os.path.abspath(frames_dir))
        if os.name == "nt":
            command = [
                "ffmpeg",
                "-i",
                video_path,
                os.path.join(frames_dir, "frame_%09d.png"),
            ]
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, shell=True
            )
        else:
            command = [
                "ffmpeg",
                "-i",
                video_path,
                os.path.join(frames_dir, "frame_%09d.png"),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e.stderr}")
        raise
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        raise


def apply_boxes_directly_to_video(input_path, output_path, coordinates, selections):
    vidcap = cv2.VideoCapture(input_path)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        mask_all = np.zeros(frame.shape[:2], dtype=np.uint8)
        for coords, (mode, shape_type) in zip(coordinates, selections):
            if shape_type == "rectangle":
                x1, y1 = int(coords[0][0]), int(coords[0][1])
                x2, y2 = int(coords[2][0]), int(coords[2][1])
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if mode == "inside":
                    frame[y1:y2, x1:x2] = 0
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    mask[y1:y2, x1:x2] = 255
                    mask_all = cv2.bitwise_or(mask_all, mask)
            elif shape_type == "trapezoid":
                pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                if mode == "inside":
                    cv2.fillPoly(frame, [pts], (0, 0, 0))
                else:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    mask_all = cv2.bitwise_or(mask_all, mask)
        if np.any(mask_all):
            frame = cv2.bitwise_and(frame, frame, mask=mask_all)
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
                mask_all = np.zeros(img.shape[:2], dtype=np.uint8)
                for coords, (mode, shape_type) in zip(coordinates, selections):
                    if shape_type == "rectangle":
                        x1, y1 = int(coords[0][0]), int(coords[0][1])
                        x2, y2 = int(coords[2][0]), int(coords[2][1])
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        if mode == "inside":
                            img[y1:y2, x1:x2] = 0
                        else:
                            mask = np.zeros(img.shape[:2], dtype=np.uint8)
                            mask[y1:y2, x1:x2] = 255
                            mask_all = cv2.bitwise_or(mask_all, mask)
                    elif shape_type == "trapezoid":
                        pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                        if mode == "inside":
                            cv2.fillPoly(img, [pts], (0, 0, 0))
                        else:
                            mask = np.zeros(img.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(mask, [pts], 255)
                            mask_all = cv2.bitwise_or(mask_all, mask)
                if np.any(mask_all):
                    img = cv2.bitwise_and(img, img, mask=mask_all)
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
    selection_mode = {"mode": "inside", "shape": "rectangle"}

    def update_title():
        shape_text = (
            "trapezoid" if selection_mode["shape"] == "trapezoid" else "rectangle"
        )
        ax.set_title(
            f"Color Guide:\n"
            f"Rectangle: Red (inside) / Blue (outside)\n"
            f"Trapezoid: Green (inside) / Yellow (outside)\n"
            f'Current mode: {selection_mode["mode"]}, Shape: {shape_text}\n'
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
        if event.key == "e":
            selection_mode["mode"] = (
                "outside" if selection_mode["mode"] == "inside" else "inside"
            )
            update_title()
        elif event.key == "t":
            selection_mode["shape"] = (
                "trapezoid" if selection_mode["shape"] == "rectangle" else "rectangle"
            )
            temp_points.clear()
            update_title()
        elif event.key == "enter":
            plt.close()

    def on_click(event):
        nonlocal points, shapes, selections, temp_points
        if event.button == 3:
            if shapes:
                shapes[-1].remove()
                shapes.pop()
                selections.pop()
                points = points[:-4] if points else []
                temp_points.clear()
                plt.draw()
            return
        if event.button == 1:
            if selection_mode["shape"] == "rectangle":
                temp_points.append((event.xdata, event.ydata))
                if len(temp_points) == 2:
                    x1, y1 = temp_points[0]
                    x2, y2 = temp_points[1]
                    rect_points = [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2),
                    ]
                    points.extend(rect_points)
                    color = "blue" if selection_mode["mode"] == "outside" else "red"
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
                temp_points.append((event.xdata, event.ydata))
                if len(temp_points) == 4:
                    color = "yellow" if selection_mode["mode"] == "outside" else "green"
                    trap = patches.Polygon(
                        temp_points,
                        linewidth=1.5,
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
    if temp_points:
        raise ValueError("An incomplete shape was defined.")
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
    time.sleep(2)


def run_drawboxe():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = tk.Tk()
    root.withdraw()
    if os.name == "nt":
        initial_dir = os.path.expanduser("~")
    elif os.name == "posix":
        initial_dir = os.path.expanduser("~")
    else:
        initial_dir = os.getcwd()
    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos", initialdir=initial_dir
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return
    video_directory = os.path.normpath(os.path.abspath(video_directory))
    if not os.path.exists(video_directory):
        messagebox.showerror("Error", f"Directory does not exist: {video_directory}")
        return
    try:
        video_files = sorted(
            [
                f
                for f in os.listdir(video_directory)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]
        )
    except PermissionError:
        messagebox.showerror(
            "Error", f"Permission denied to access directory: {video_directory}"
        )
        return
    except Exception as e:
        messagebox.showerror("Error", f"Error accessing directory: {str(e)}")
        return
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(video_directory, f"video_2_drawbox_{timestamp}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
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
