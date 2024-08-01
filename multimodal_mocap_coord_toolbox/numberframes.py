import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

def get_video_info(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = f"{chr(codec & 0xFF)}{chr((codec >> 8) & 0xFF)}{chr((codec >> 16) & 0xFF)}{chr((codec >> 24) & 0xFF)}"
        
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        contrast = cap.get(cv2.CAP_PROP_CONTRAST)
        saturation = cap.get(cv2.CAP_PROP_SATURATION)
        hue = cap.get(cv2.CAP_PROP_HUE)
        gain = cap.get(cv2.CAP_PROP_GAIN)
        convert_rgb = cap.get(cv2.CAP_PROP_CONVERT_RGB)
        
        duration = frame_count / fps if fps else 0
        
        cap.release()

        return {
            "file_name": os.path.basename(video_path),
            "frame_count": frame_count,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "codec": codec_str,
            "duration": duration,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
            "gain": gain,
            "convert_rgb": convert_rgb
        }
    except Exception as e:
        print(f"Error parsing video info for {video_path}: {str(e)}")
        return None

def count_frames_in_videos(directory_path):
    video_files = sorted([f for f in os.listdir(directory_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    video_infos = []
    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        video_info = get_video_info(video_path)
        if video_info is not None:
            video_infos.append(video_info)
        else:
            video_infos.append({"file_name": video_file, "error": "Error retrieving video info"})
    
    output_file = save_metadata_to_file(video_infos, directory_path)
    print(f"Metadata saved to: {output_file}")
    display_video_info(video_infos, output_file)
    print_ffprobe_commands(video_files, directory_path)

def display_video_info(video_infos, output_file):
    def on_closing():
        root.destroy()
        show_save_success_message(output_file)
    
    root = tk.Tk()
    root.title("Video Information")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    canvas = tk.Canvas(frame, width=1080, height=720)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    # Adding headers
    headers = ["Video File", "Frames", "FPS", "Resolution", "Codec", "Duration (s)", "Brightness", "Contrast", "Saturation", "Hue", "Gain", "Convert RGB"]
    for i, header in enumerate(headers):
        ttk.Label(scrollable_frame, text=header, font=("Arial", 10, "bold")).grid(row=0, column=i, padx=10, pady=5, sticky=tk.W)
    
    for i, info in enumerate(video_infos, start=1):
        ttk.Label(scrollable_frame, text=info['file_name'], font=("Arial", 10, "bold")).grid(row=i, column=0, sticky=tk.W, pady=5)
        if 'error' in info:
            ttk.Label(scrollable_frame, text=info['error']).grid(row=i, column=1, columnspan=11, sticky=tk.W, padx=10)
        else:
            ttk.Label(scrollable_frame, text=info['frame_count']).grid(row=i, column=1, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=info['fps']).grid(row=i, column=2, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=info['resolution']).grid(row=i, column=3, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=info['codec']).grid(row=i, column=4, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=f"{info['duration']:.2f}").grid(row=i, column=5, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=f"{info['brightness']:.2f}").grid(row=i, column=6, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=f"{info['contrast']:.2f}").grid(row=i, column=7, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=f"{info['saturation']:.2f}").grid(row=i, column=8, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=f"{info['hue']:.2f}").grid(row=i, column=9, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=f"{info['gain']:.2f}").grid(row=i, column=10, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=f"{info['convert_rgb']:.2f}").grid(row=i, column=11, sticky=tk.W, padx=10)
    
    root.mainloop()

def save_metadata_to_file(video_infos, directory_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(directory_path, f"video_metadata_{timestamp}.txt")
    
    with open(output_file, 'w') as f:
        for info in video_infos:
            if 'error' in info:
                f.write(f"File: {info['file_name']}\nError: {info['error']}\n\n")
            else:
                f.write(f"File: {info['file_name']}\n")
                f.write(f"Frames: {info['frame_count']}\n")
                f.write(f"FPS: {info['fps']}\n")
                f.write(f"Resolution: {info['resolution']}\n")
                f.write(f"Codec: {info['codec']}\n")
                f.write(f"Duration (s): {info['duration']:.2f}\n")
                f.write(f"Brightness: {info['brightness']:.2f}\n")
                f.write(f"Contrast: {info['contrast']:.2f}\n")
                f.write(f"Saturation: {info['saturation']:.2f}\n")
                f.write(f"Hue: {info['hue']:.2f}\n")
                f.write(f"Gain: {info['gain']:.2f}\n")
                f.write(f"Convert RGB: {info['convert_rgb']:.2f}\n\n")
    
    return output_file

def print_ffprobe_commands(video_files, directory_path):
    print("\nTo get more detailed metadata, you can use the following ffprobe commands in the shell:\n")
    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        print(f"ffprobe -v quiet -print_format json -show_format -show_streams \"{video_path}\"")

def show_save_success_message(output_file):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Save Success", f"Metadata successfully saved!\n\nOutput file: {output_file}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(title="Select the directory containing videos")
    if directory_path:
        count_frames_in_videos(directory_path)
    else:
        messagebox.showerror("Error", "No directory selected.")
