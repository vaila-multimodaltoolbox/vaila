import os
from ffmpeg import FFmpeg
from tkinter import filedialog, messagebox, Tk

def run_compress_videos_h265(video_directory, preset='medium', crf=23):
    output_directory = os.path.join(video_directory, 'compressed_h265')
    os.makedirs(output_directory, exist_ok=True)
    
    print("!!!ATTENTION!!!")
    print("This process might take several hours depending on your computer and the size of your videos. Please be patient or use a high-performance computer!")
    
    for video_file in sorted([f for f in os.listdir(video_directory) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]):
        input_path = os.path.join(video_directory, video_file)
        output_path = os.path.join(output_directory, f"{os.path.splitext(video_file)[0]}_h265.mp4")
        
        print(f"Compressing {video_file}...")
        
        ffmpeg = (
            FFmpeg()
            .option("y")
            .input(input_path)
            .output(
                output_path,
                {"codec:v": "libx265"},
                preset=preset,
                crf=crf
            )
        )

        try:
            ffmpeg.execute()
            print(f"Done compressing {video_file} to H.265/HEVC.")
        except Exception as e:
            print(f"Error compressing {video_file}: {e}")
    
    print("All videos have been compressed successfully!")

def compress_videos_h265_gui():
    root = Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(title="Select the directory containing videos to compress")
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return
    
    preset = 'medium'
    crf = 23
    
    run_compress_videos_h265(video_directory, preset, crf)
    messagebox.showinfo("Success", "Video compression completed. All videos have been compressed successfully!")

if __name__ == "__main__":
    compress_videos_h265_gui()
