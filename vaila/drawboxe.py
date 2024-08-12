"""
drawboxe.py

Description:
-----------
This script is designed to add bounding boxes to videos using coordinates obtained from clicks on an image. It also supports extracting frames and applying boxes to specific frame intervals or directly to videos. The script can be used for batch processing of videos in a directory.

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
- pandas
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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

# Funções do common_utils
def determine_header_lines(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            first_element = line.split(',')[0].strip()
            if first_element.replace('.', '', 1).isdigit():
                return i
    return 0

def headersidx(file_path):
    try:
        header_lines = determine_header_lines(file_path)
        df = pd.read_csv(file_path, header=list(range(header_lines)))

        print("Headers with indices:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i}: {col}")

        print("\nExample of new order:")
        new_order = ['Time']
        for i in range(1, len(df.columns), 3):
            new_order.append(df.columns[i][0])
        print(new_order)

        return new_order

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def reshapedata(file_path, new_order, save_directory):
    try:
        header_lines = determine_header_lines(file_path)
        df = pd.read_csv(file_path, skiprows=header_lines, header=None)
        actual_header = pd.read_csv(file_path, nrows=header_lines, header=None).values
        new_order_indices = [0]

        for header in new_order[1:]:
            base_idx = [i for i, col in enumerate(actual_header[0]) if col == header][0]
            new_order_indices.extend([base_idx, base_idx + 1, base_idx + 2])

        df_reordered = df.iloc[:, new_order_indices]
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        new_file_path = os.path.join(save_directory, f"{base_name}_reord.csv")
        
        new_header = []
        for header in new_order:
            if header == 'Time':
                new_header.append(header)
            else:
                new_header.extend([header + '_x', header + '_y', header + '_z'])

        df_reordered.to_csv(new_file_path, index=False, header=new_header)
        print(f"Reordered data saved to {new_file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Funções específicas de drawboxe
def get_box_coordinates(image_path):
    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.title('Click on the top-left and bottom-right corners of the box. Press Enter to finish.')

    points = []

    def onclick(event):
        if event.key == 'enter':
            plt.close()
            return
        if len(points) % 2 == 0:
            points.append((event.xdata, event.ydata))
        else:
            points.append((event.xdata, event.ydata))
            rect = patches.Rectangle(
                (points[-2][0], points[-2][1]),
                points[-1][0] - points[-2][0],
                points[-1][1] - points[-2][1],
                linewidth=1,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)
            plt.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onclick)
    plt.show()

    if len(points) % 2 != 0:
        raise ValueError("An incomplete box was defined.")

    return [(int(points[i][0]), int(points[i][1]), int(points[i+1][0]), int(points[i+1][1])) for i in range(0, len(points), 2)]

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

def apply_boxes_directly_to_video(input_path, output_path, coordinates):
    vidcap = cv2.VideoCapture(input_path)
    width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        for (x1, y1, x2, y2) in coordinates:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        out.write(frame)
    
    vidcap.release()
    out.release()

def apply_boxes_to_frames(frames_dir, coordinates, frame_intervals):
    for filename in sorted(os.listdir(frames_dir)):
        frame_number = int(filename.split('_')[1].split('.')[0])
        for start_frame, end_frame in frame_intervals:
            if start_frame <= frame_number <= end_frame:
                frame_path = os.path.join(frames_dir, filename)
                img = cv2.imread(frame_path)
                for (x1, y1, x2, y2) in coordinates:
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

def load_frame_intervals(file_path):
    intervals = []
    with open(file_path, 'r') as file:
        for line in file:
            start, end = map(int, line.strip().split(','))
            intervals.append((start, end))
    return intervals

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

    coordinates = get_box_coordinates(first_frame_path)
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
    os.makedirs(output_dir, exist_ok=True)

    for video_file in video_files:
        input_path = os.path.join(video_directory, video_file)
        vidcap = cv2.VideoCapture(input_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        vidcap.release()
        
        if frame_intervals:
            frames_dir = os.path.join(video_directory, 'frames_temp')
            extract_frames(input_path, frames_dir)
            apply_boxes_to_frames(frames_dir, coordinates, frame_intervals)
            final_output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_dbox.mp4")
            reassemble_video(frames_dir, final_output_path, fps)
            clean_up(frames_dir)
        else:
            final_output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_dbox.mp4")
            apply_boxes_directly_to_video(input_path, final_output_path, coordinates)

    print("All videos processed and saved to the output directory.")

if __name__ == "__main__":
    run_drawboxe()
