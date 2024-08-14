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
2024-08-12 08:00:00 (New York Time)

Author:
-------
Prof. PhD. Paulo Santiago

License:
--------
This code is licensed under the MIT License. See the LICENSE file for more details.

Version History:
----------------
- v1.0 (2024-08-12): Initial version with support for adding boxes to videos and extracting frames.

Contact:
--------
For questions or contributions, please contact the author at: paulo.santiago@example.com.

Contributions:
--------------
Contributions are welcome. Please follow the contribution guidelines provided in the CONTRIBUTING.md file of this repository.

Dependencies:
-------------
- Python 3.11.8 (Anaconda environment)
- os
- ffmpeg-python
- matplotlib
- opencv-python
- tkinter

Additional Notes:
-----------------
- Ensure FFMPEG is installed on your system for this script to function correctly.
- The script assumes that the input videos are in a format supported by OpenCV.
"""

import os
from ffmpeg import FFmpeg
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
    ffmpeg = (
        FFmpeg()
        .input(video_path)
        .output(os.path.join(frames_dir, 'frame_%09d.png'))
    )
    ffmpeg.execute()

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
        for (x1, y1, x2, y2), selection in zip(coordinates, selections):
            if selection == 'outside':
                frame[:y1, :] = (0, 0, 0)
                frame[y2:, :] = (0, 0, 0)
                frame[y1:y2, :x1] = (0, 0, 0)
                frame[y1:y2, x2:] = (0, 0, 0)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        out.write(frame)
        frame_count += 1
        print(f"Processed {frame_count}/{total_frames} frames for {os.path.basename(input_path)}", end="\r")

    print(f"\nCompleted processing: {os.path.basename(input_path)}")
    print(f"Saved to: {output_path}")
    out.release()
    vidcap.release()

def apply_boxes_to_frames(frames_dir, coordinates, selections, frame_intervals):
    for filename in sorted(os.listdir(frames_dir)):
        frame_number = int(filename.split('_')[1].split('.')[0])
        for start_frame, end_frame in frame_intervals:
            if start_frame <= frame_number <= end_frame:
                frame_path = os.path.join(frames_dir, filename)
                img = cv2.imread(frame_path)
                for (x1, y1, x2, y2), selection in zip(coordinates, selections):
                    if selection == 'outside':
                        img[:y1, :] = (0, 0, 0)
                        img[y2:, :] = (0, 0, 0)
                        img[y1:y2, :x1] = (0, 0, 0)
                        img[y1:y2, x2:] = (0, 0, 0)
                    else:
                        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                cv2.imwrite(frame_path, img)

def reassemble_video(frames_dir, output_path, fps):
    ffmpeg = (
        FFmpeg()
        .input(os.path.join(frames_dir, 'frame_%09d.png'), framerate=fps)
        .output(output_path)
    )
    ffmpeg.execute()

def clean_up(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)
    os.rmdir(directory)

def get_box_coordinates(image_path):
    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    selection_mode = {'mode': 'inside'}  # Default mode is 'inside'

    def update_title():
        ax.set_title(f'Red box: inside, Blue box: outside\nCurrent mode: {selection_mode["mode"]}\n'
                     'Click to select corners of the box. Press "e" to toggle mode, Enter to finish.')
        fig.canvas.draw()

    ax.imshow(img)
    update_title()

    points = []
    rects = []
    selections = []

    def on_key(event):
        if event.key == 'e':  # Toggle mode
            selection_mode['mode'] = 'outside' if selection_mode['mode'] == 'inside' else 'inside'
            update_title()
        elif event.key == 'enter':  # Close the window
            plt.close()

    def on_click(event):
        nonlocal points, rects, selections

        if event.button == 3:  # Right mouse button to remove the last box
            if len(points) > 0:
                points = points[:-2]  # Remove the last pair of points
                if rects:
                    rects[-1].remove()  # Remove the last rectangle from the plot
                    rects.pop()
                    selections.pop()
                plt.draw()
            return

        if event.button == 1:  # Left mouse button to add a point
            if len(points) % 2 == 0:
                points.append((event.xdata, event.ydata))
            else:
                points.append((event.xdata, event.ydata))
                color = 'b' if selection_mode['mode'] == 'outside' else 'r'
                rect = patches.Rectangle(
                    (points[-2][0], points[-2][1]),
                    points[-1][0] - points[-2][0],
                    points[-1][1] - points[-2][1],
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                rects.append(rect)
                selections.append(selection_mode['mode'])
                plt.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if len(points) % 2 != 0:
        raise ValueError("An incomplete box was defined.")

    boxes = [(int(points[i][0]), int(points[i][1]), int(points[i+1][0]), int(points[i+1][1])) for i in range(0, len(points), 2)]

    return boxes, selections

def load_frame_intervals(file_path):
    intervals = []
    with open(file_path, 'r') as file:
        for line in file:
            start, end = map(int, line.strip().split(','))
            intervals.append((start, end))
    return intervals

def show_feedback_message():
    print("vailá!")
    time.sleep(2)  # Simulate processing time

def run_drawboxe():
    root = tk.Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(title="Select the directory containing videos")
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    video_files = sorted([f for f in os.listdir(video_directory) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])

    if not video_files:
        messagebox.showerror("Error", "No video files found in the selected directory.")
        return

    first_video = video_files[0]
    first_frame_path = os.path.join(video_directory, 'first_frame.jpg')
    save_first_frame(os.path.join(video_directory, first_video), first_frame_path)

    coordinates, selections = get_box_coordinates(first_frame_path)
    os.remove(first_frame_path)

    use_intervals = messagebox.askyesno("Frame Intervals", "Do you want to use frame intervals from a .txt file?")
    frame_intervals = None
    if use_intervals:
        intervals_file = filedialog.askopenfilename(title="Select the .txt file with frame intervals", filetypes=[("Text files", "*.txt")])
        if intervals_file:
            frame_intervals = load_frame_intervals(intervals_file)
        else:
            messagebox.showerror("Error", "No .txt file selected.")
            return

    output_dir = os.path.join(video_directory, 'video_2_drawbox')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete the existing directory and its contents

    os.makedirs(output_dir, exist_ok=True)

    for video_file in video_files:
        input_path = os.path.join(video_directory, video_file)
        final_output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_dbox.mp4")
        
        vidcap = cv2.VideoCapture(input_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidcap.release()

        if frame_intervals:
            frames_dir = os.path.join(video_directory, 'frames_temp')
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            extract_frames(input_path, frames_dir)
            apply_boxes_to_frames(frames_dir, coordinates, selections, frame_intervals)
            reassemble_video(frames_dir, final_output_path, fps)
            clean_up(frames_dir)
        else:
            apply_boxes_directly_to_video(input_path, final_output_path, coordinates, selections)
    
    show_feedback_message()
    print("All videos processed and saved to the output directory.")
    messagebox.showinfo("Completed", "All videos have been processed successfully!")

if __name__ == "__main__":
    run_drawboxe()
