import os
from ffmpeg import FFmpeg
from tkinter import filedialog, messagebox, Tk

def extract_png_from_videos():
    # Ocultar a janela principal do Tkinter
    root = Tk()
    root.withdraw()

    # Selecionar o diretório de origem
    src = filedialog.askdirectory(title="Select the source directory containing videos")
    if not src:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Selecionar o diretório de destino
    dest = filedialog.askdirectory(title="Select the destination directory for PNG files")
    if not dest:
        messagebox.showerror("Error", "No destination directory selected.")
        return

    try:
        for item in os.listdir(src):
            if item.endswith(('.avi', '.mp4', '.mov', '.mkv')):
                video_path = os.path.join(src, item)
                video_name = os.path.splitext(item)[0]
                output_dir = os.path.join(dest, f"{video_name}_png")
                os.makedirs(output_dir, exist_ok=True)
                output_pattern = os.path.join(output_dir, "%09d.png")

                ffmpeg = (
                    FFmpeg()
                    .input(video_path)
                    .output(
                        output_pattern,
                        vf='scale=in_range=pc:out_range=pc,format=rgb24',
                        vcodec='png',
                        q=1
                    )
                )

                ffmpeg.execute()

        messagebox.showinfo("Success", "PNG frames have been extracted from the videos.")
    except Exception as e:
        messagebox.showerror("Error", f"Error extracting PNG frames: {e}")

if __name__ == "__main__":
    extract_png_from_videos()
