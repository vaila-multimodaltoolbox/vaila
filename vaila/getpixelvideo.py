# --------------------------------------------------
# Script Name: getpixelvideo.py
# Version: 0.0.1
# Last Updated: 8 Aug 2024
# Description: A tool for marking and saving pixel
# coordinates in a video.
# --------------------------------------------------
# Usage Instructions:
# - Press 'Space' to toggle play/pause of the video.
# - Press 'Escape' to close the video and exit the application.
# - Press 'A' or 'Left Arrow' to go to the previous frame.
# - Press 'D' or 'Right Arrow' to go to the next frame.
# - Press 'W' or 'Up Arrow' to jump to the last frame.
# - Press 'S' or 'Down Arrow' to jump to the first frame.
# - Left-click to mark a point on the video.
# - Right-click to remove the last marked point.
# --------------------------------------------------

import cv2
import os
import pandas as pd
from tkinter import filedialog, Tk, Toplevel, Scale, HORIZONTAL, Button, messagebox
import numpy as np

def show_help_message():
    root = Tk()
    root.withdraw()
    messagebox.showinfo(
        "Help",
        "This tool allows you to mark and save pixel coordinates in a video.\n\n"
        "Instructions:\n"
        "- Press 'Space' to toggle play/pause of the video.\n"
        "- Press 'Escape' to close the video and exit the application.\n"
        "- Press 'A' or 'Left Arrow' to go to the previous frame.\n"
        "- Press 'D' or 'Right Arrow' to go to the next frame.\n"
        "- Press 'W' or 'Up Arrow' to jump to the last frame.\n"
        "- Press 'S' or 'Down Arrow' to jump to the first frame.\n"
        "- Left-click to mark a point on the video.\n"
        "- Right-click to remove the last marked point.\n\n"
        "For more detailed help, click the link below:\n"
        "docs/help.html",
        icon='info'
    )

def get_video_path():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
    return video_path

def save_coordinates(video_path, coordinates, total_frames):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    output_file = os.path.join(video_dir, f"{base_name}_getpixel.csv")

    columns = ['frame'] + [f'p{i}_{c}' for i in range(1, 101) for c in ['x', 'y']]
    df = pd.DataFrame(np.nan, index=range(total_frames), columns=columns)
    df['frame'] = df.index

    for frame_num, points in coordinates.items():
        for i, (x, y) in enumerate(points):
            df.at[frame_num, f'p{i+1}_x'] = x
            df.at[frame_num, f'p{i+1}_y'] = y

    # Determine the last marked point
    last_point = 0
    for frame_num, points in coordinates.items():
        if points:
            last_point = max(last_point, len(points))

    # Remove columns beyond the last marked point
    if last_point < 100:
        df = df.iloc[:, :1 + 2 * last_point]

    df.to_csv(output_file, index=False)
    print(f"Coordinates saved to {output_file}")

def get_pixel_coordinates(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    coordinates = {i: [] for i in range(total_frames)}
    paused = True
    frame = None

    def draw_point(frame, x, y, num):
        color = (0, 255, 0)
        radius = 5
        thickness = -1  # Filled circle
        cv2.circle(frame, (x, y), radius, color, thickness)
        cv2.putText(frame, f'{num}', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def click_event(event, x, y, flags, param):
        nonlocal frame
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates[frame_count].append((x, y))
            draw_point(frame, x, y, len(coordinates[frame_count]))
            cv2.imshow('Frame', frame)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if coordinates[frame_count]:
                coordinates[frame_count].pop()
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if ret:
                    for i, point in enumerate(coordinates[frame_count]):
                        draw_point(frame, point[0], point[1], i + 1)
                    cv2.imshow('Frame', frame)

    def update_frame(new_frame_count):
        nonlocal frame_count, frame, paused
        frame_count = new_frame_count
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if ret:
            for i, point in enumerate(coordinates[frame_count]):
                draw_point(frame, point[0], point[1], i + 1)
            cv2.imshow('Frame', frame)
            window.title(f'Frame {frame_count}')
        paused = True

    def on_key(event):
        nonlocal paused
        key = event.keysym.lower()
        if key == 'space':
            toggle_play_pause()
        elif key == 'escape':
            cap.release()
            cv2.destroyAllWindows()
            window.quit()
        elif key in ['a', 'left']:
            update_frame(max(frame_count - 1, 0))
        elif key in ['d', 'right']:
            update_frame(min(frame_count + 1, total_frames - 1))
        elif key in ['w', 'up']:
            update_frame(total_frames - 1)
        elif key in ['s', 'down']:
            update_frame(0)
        slider.set(frame_count)

    def play_video():
        nonlocal frame_count, frame
        if not paused:
            frame_count += 1
            if frame_count >= total_frames:
                frame_count = 0
            ret, frame = cap.read()
            if ret:
                for i, point in enumerate(coordinates[frame_count]):
                    draw_point(frame, point[0], point[1], i + 1)
                cv2.imshow('Frame', frame)
                window.title(f'Frame {frame_count}')
                slider.set(frame_count)
            window.after(30, play_video)  # Adjust delay as needed
        else:
            window.after(30, play_video)

    def toggle_play_pause():
        nonlocal paused
        paused = not paused

    def close_video():
        cap.release()
        cv2.destroyAllWindows()
        window.quit()

    root = Tk()
    root.withdraw()

    window = Toplevel(root)
    window.title('Frame Viewer')
    window.bind('<KeyPress>', on_key)
    window.geometry('800x100')

    slider = Scale(window, from_=0, to=total_frames-1, orient=HORIZONTAL, command=lambda pos: update_frame(int(pos)))
    slider.pack(fill='x', expand=True)

    play_pause_button = Button(window, text="Next Frame", command=toggle_play_pause)
    play_pause_button.pack(side='left')

    close_button = Button(window, text="Close Video", command=close_video)
    close_button.pack(side='right')

    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', click_event)

    update_frame(0)  # Start with the first frame paused

    window.after(30, play_video)
    window.mainloop()

    return coordinates, total_frames

def main():
    show_help_message()  # Show the help message and open the help file
    video_path = get_video_path()
    if video_path:
        coordinates, total_frames = get_pixel_coordinates(video_path)
        save_coordinates(video_path, coordinates, total_frames)

if __name__ == "__main__":
    main()
