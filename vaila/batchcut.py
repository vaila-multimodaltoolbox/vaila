"""
batchcut.py
Version: 2024-07-25

Script para cortar vídeos em batch baseado em um arquivo de lista de frames.
"""

import os
import sys
from ffmpeg import FFmpeg
import tkinter as tk
from tkinter import filedialog

def batch_cut_videos(video_directory, list_file_path, output_directory):
    if not os.path.isfile(list_file_path):
        print(f"List file {list_file_path} does not exist.")
        return
    
    # Cria o subdiretório "cut_videos" dentro do diretório de saída
    output_directory = os.path.join(output_directory, "cut_videos")
    os.makedirs(output_directory, exist_ok=True)
    
    with open(list_file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if not line.strip():
            continue
        
        original_name, new_name, start_frame, end_frame = line.split()
        start_frame, end_frame = int(start_frame), int(end_frame)
        
        original_path = os.path.join(video_directory, original_name)
        new_path = os.path.join(output_directory, f"{new_name}.mp4")
        
        try:
            print(f"Processing {original_name} from frame {start_frame} to {end_frame}")
            probe = FFmpeg().input(original_path).output('pipe:').run(capture_stdout=True, capture_stderr=True)
            video_stream = next((stream for stream in probe[1]['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream:
                print(f"No video stream found in {original_name}. Skipping.")
                continue

            fps = eval(video_stream['r_frame_rate'])
            ffmpeg = (
                FFmpeg()
                .input(original_path)
                .trim(start_frame=start_frame, end_frame=end_frame)
                .setpts('PTS-STARTPTS')
                .output(new_path, r=fps)
                .global_args('-hide_banner', '-nostats', '-loglevel', 'quiet')
            )
            ffmpeg.execute()
            print(f"{new_name} completed!")
        except Exception as e:
            print(f"Error processing {original_name}: {str(e)}", file=sys.stderr)

def cut_videos():
    root = tk.Tk()
    root.withdraw()
    
    video_directory = filedialog.askdirectory(title="Select the directory containing videos")
    if not video_directory:
        print("No video directory selected.")
        return

    list_file_path = filedialog.askopenfilename(
        title="Select the list file",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
    )
    if not list_file_path:
        print("No list file selected.")
        return

    output_directory = filedialog.askdirectory(title="Select the output directory")
    if not output_directory:
        print("No output directory selected.")
        return

    root.destroy()

    batch_cut_videos(video_directory, list_file_path, output_directory)

if __name__ == "__main__":
    cut_videos()
