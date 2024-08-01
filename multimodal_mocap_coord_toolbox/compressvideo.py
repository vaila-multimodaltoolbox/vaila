import os
from ffmpeg import FFmpeg
from tkinter import filedialog, messagebox, Tk, simpledialog, Toplevel, Label, Radiobutton, StringVar, Button, W

def check_ffmpeg_encoder(encoder):
    try:
        ffmpeg = FFmpeg().input('dummy').output('dummy.mp4', vcodec=encoder).global_args('-hide_banner', '-nostats')
        ffmpeg.execute()
        return True
    except Exception as e:
        if "Unknown encoder" in str(e):
            return False
        else:
            raise

def run_compress_videos(video_directory, codec, preset='medium', crf=23):
    output_directory = os.path.join(video_directory, 'compressed_' + codec)
    os.makedirs(output_directory, exist_ok=True)
    
    print("!!!ATTENTION!!!")
    print("This process might take several hours depending on your computer and the size of your videos. Please be patient or use a high-performance computer!")
    
    if not check_ffmpeg_encoder(codec):
        print(f"Error: Your ffmpeg installation does not support the {codec} encoder.")
        return
    
    for video_file in sorted([f for f in os.listdir(video_directory) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]):
        input_path = os.path.join(video_directory, video_file)
        output_path = os.path.join(output_directory, f"{os.path.splitext(video_file)[0]}_{codec}.mp4")
        
        print(f"Compressing {video_file}...")
        
        ffmpeg = (
            FFmpeg()
            .input(input_path)
            .output(
                output_path,
                vcodec=codec,
                preset=preset,
                crf=crf
            )
            .global_args('-hide_banner', '-nostats')
        )

        try:
            ffmpeg.execute()
            print(f"Done compressing {video_file} to {codec}.")
        except Exception as e:
            print(f"Error compressing {video_file}: {e}")

def ask_codec_selection(root):
    codecs = [("H264", "libx264"), ("H265/HEVC", "libx265")]
    
    codec_var = StringVar(value="libx264")
    selection_dialog = Toplevel(root)
    selection_dialog.title("Select Codec")

    Label(selection_dialog, text="Select the codec to use:").pack(pady=10)

    for text, codec in codecs:
        Radiobutton(selection_dialog, text=text, variable=codec_var, value=codec).pack(anchor=W, padx=20)

    def on_ok():
        selection_dialog.destroy()

    Button(selection_dialog, text="OK", command=on_ok).pack(pady=10)

    root.wait_window(selection_dialog)
    return codec_var.get()

def compress_videos_gui():
    root = Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(title="Select the directory containing videos to compress")
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    codec = ask_codec_selection(root)
    if not codec:
        messagebox.showerror("Error", "No codec selected.")
        return

    preset = 'medium'  # You can add GUI components to select these values if needed
    crf = 23           # You can add GUI components to select these values if needed
    
    run_compress_videos(video_directory, codec, preset, crf)
    messagebox.showinfo("Success", "Video compression completed.")

if __name__ == "__main__":
    compress_videos_gui()
