"""
syncvid.py

This script is used for manually synchronizing videos, allowing the user to input
keyframes for multiple video files and select a main camera for synchronization.

Features:
- Manual synchronization: Allows the user to manually input keyframes for
  synchronization.
- Automatic synchronization (using flash): Automatically extracts the median of
  the R, G, and B values from a specific region of each video to automatically
  synchronize the videos based on a flash or brightness change.

Dependencies:
- tkinter: For the graphical user interface (GUI).
- cv2 (OpenCV): For video processing, used in automatic synchronization.
- numpy: For numerical array manipulation, used in automatic synchronization.

Usage:
- Run the script and follow the instructions in the GUI to select the videos
  and choose the synchronization method.
- Save the resulting synchronization file in the desired format.

Author: [Your Name]
Date: [Current Date]

"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from vaila.sync_flash import (
    get_median_brightness,
)  # Imports the automatic synchronization feature


def get_video_files(directory_path):
    return sorted(
        [
            f
            for f in os.listdir(directory_path)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )


def write_sync_file(sync_data, output_file):
    with open(output_file, "a") as f:
        for data in sync_data:
            f.write(" ".join(map(str, data)) + "\n")


def get_sync_info(video_files):
    sync_data = []

    class SyncDialog(tk.Toplevel):
        def __init__(self, master, video_files):
            super().__init__(master)
            self.video_files = video_files
            self.sync_data = []
            self.entries = {}
            self.main_video = None

            self.title("Video Synchronization")
            self.geometry("800x600")

            self.create_widgets()

        def create_widgets(self):
            tk.Label(
                self, text="Enter Keyframes for Synchronization", font=("Arial", 14)
            ).pack(pady=10)

            frame = tk.Frame(self)
            frame.pack(fill=tk.BOTH, expand=True)

            canvas = tk.Canvas(frame)
            scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

            canvas.configure(yscrollcommand=scrollbar.set)

            # Header labels
            header_frame = tk.Frame(scrollable_frame)
            header_frame.pack(fill=tk.X, pady=5)
            tk.Label(header_frame, text="Video File", width=40, anchor="w").pack(
                side=tk.LEFT, padx=5
            )
            tk.Label(header_frame, text="Keyframe", width=10, anchor="w").pack(
                side=tk.LEFT, padx=5
            )

            for video_file in self.video_files:
                row = tk.Frame(scrollable_frame)
                row.pack(fill=tk.X, pady=5)

                tk.Label(row, text=video_file, width=40, anchor="w").pack(side=tk.LEFT)
                keyframe_entry = tk.Entry(row, width=10)
                keyframe_entry.pack(side=tk.LEFT, padx=5)

                self.entries[video_file] = keyframe_entry

            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

            tk.Button(self, text="Next", command=self.on_next).pack(pady=10)

        def on_next(self):
            for video_file, keyframe_entry in self.entries.items():
                try:
                    keyframe = int(keyframe_entry.get())
                    self.sync_data.append([video_file, keyframe])
                except ValueError:
                    messagebox.showerror(
                        "Error",
                        f"Invalid input for video {video_file}. Please enter a valid keyframe.",
                    )
                    return

            self.select_main_camera()

        def select_main_camera(self):
            self.withdraw()
            MainCameraDialog(self, self.video_files)

        def set_main_camera(self, main_video, frame_initial, frame_final):
            self.main_video = main_video
            self.frame_initial = frame_initial
            self.frame_final = frame_final
            self.destroy()

        def get_sync_data(self):
            self.wait_window()
            return self.sync_data, self.main_video, self.frame_initial, self.frame_final

    class MainCameraDialog(tk.Toplevel):
        def __init__(self, master, video_files):
            super().__init__(master)
            self.master = master
            self.video_files = video_files
            self.main_video_var = tk.StringVar(value="")

            self.title("Select Main Camera")
            self.geometry("600x400")

            tk.Label(
                self,
                text="Select the Main Camera for Synchronization",
                font=("Arial", 14),
            ).pack(pady=10)

            frame = tk.Frame(self)
            frame.pack(pady=10)

            frame_initial_label = tk.Label(frame, text="Start Frame:")
            frame_initial_label.pack(side=tk.LEFT, padx=5)
            self.frame_initial_entry = tk.Entry(frame, width=10)
            self.frame_initial_entry.pack(side=tk.LEFT, padx=5)

            frame_final_label = tk.Label(frame, text="End Frame:")
            frame_final_label.pack(side=tk.LEFT, padx=5)
            self.frame_final_entry = tk.Entry(frame, width=10)
            self.frame_final_entry.pack(side=tk.LEFT, padx=5)

            self.main_video_combobox = tk.Listbox(self, selectmode=tk.SINGLE)
            self.main_video_combobox.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
            for video_file in video_files:
                self.main_video_combobox.insert(tk.END, video_file)

            tk.Button(self, text="OK", command=self.on_ok).pack(pady=10)

        def on_ok(self):
            selected_index = self.main_video_combobox.curselection()
            if not selected_index:
                messagebox.showerror(
                    "Error", "Please select the main video for synchronization."
                )
                return

            try:
                frame_initial = int(self.frame_initial_entry.get())
                frame_final = int(self.frame_final_entry.get())
            except ValueError:
                messagebox.showerror(
                    "Error", "Please enter valid start and end frames."
                )
                return

            main_video = self.video_files[selected_index[0]]
            print(f"Selected main video: {main_video}")  # Debugging print
            self.master.set_main_camera(main_video, frame_initial, frame_final)
            self.destroy()

    root = tk.Tk()
    root.withdraw()

    dialog = SyncDialog(root, video_files)
    sync_data, main_video, frame_initial, frame_final = dialog.get_sync_data()

    return sync_data, main_video, frame_initial, frame_final


def sync_videos():
    root = tk.Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos"
    )
    if not video_directory:
        print("No video directory selected.")
        return

    output_file = filedialog.asksaveasfilename(
        title="Save sync file as",
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if not output_file:
        print("No output file selected.")
        return

    video_files = get_video_files(video_directory)

    # Ask the user if they want to use the flash for automatic synchronization
    use_flash = messagebox.askyesno(
        "Use Flash Synchronization",
        "Do you want to use flash detection for automatic synchronization?",
    )

    if use_flash:
        for video_file in video_files:
            region = (50, 50, 100, 100)  # Example region, customize as needed
            median_brightness = get_median_brightness(video_file, region)
            print(f"Median brightness for {video_file}: {median_brightness}")

    sync_data, main_video, frame_initial, frame_final = get_sync_info(video_files)
    if not main_video:
        messagebox.showerror("Error", "Main video was not selected.")
        return

    main_keyframe = None

    for video_file, keyframe in sync_data:
        if main_video in video_file:
            main_keyframe = keyframe
            break

    if main_keyframe is None:
        messagebox.showerror("Error", "Main video keyframe is not set.")
        return

    adjusted_sync_data = []
    for video_file, keyframe in sync_data:
        if main_video in video_file:
            adjusted_sync_data.append(
                [
                    video_file,
                    f"{os.path.splitext(video_file)[0]}_{keyframe}_{frame_initial}_{frame_final}.mp4",
                    frame_initial,
                    frame_final,
                ]
            )
            continue

        initial_frame = frame_initial - (main_keyframe - keyframe)
        final_frame = frame_final - (main_keyframe - keyframe)
        new_name = f"{os.path.splitext(video_file)[0]}_{keyframe}_{initial_frame}_{final_frame}.mp4"
        adjusted_sync_data.append([video_file, new_name, initial_frame, final_frame])

    write_sync_file(adjusted_sync_data, output_file)
    print(
        "Sync file created successfully! Now use Cut Videos to synchronize the videos."
    )

    # Display success message
    messagebox.showinfo(
        "Success",
        f"Sync file '{output_file}' was created successfully. Now use Cut Videos to synchronize the videos.",
    )


if __name__ == "__main__":
    sync_videos()
