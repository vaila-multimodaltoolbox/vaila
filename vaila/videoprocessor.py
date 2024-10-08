"""
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

videoprocessor.py

Description:
This script allows users to process and edit video files, enabling batch processing of videos. It concatenates pairs of videos
based on instructions provided in a text file. The script supports custom text files for batch processing and includes a GUI 
for directory and file selection.

Key Features:
- Graphical User Interface (GUI) for easy selection of directories and file inputs.
- Batch processing using a text file (`videos_e_frames.txt`) with custom instructions for each pair of videos to merge.
- Each line in the text file contains two video filenames, which will be concatenated into a single output video.
- Automatic creation of output directories based on a timestamp for organized file management.
- Detailed console output for tracking progress and handling errors.

Usage:
- Run the script to open a graphical interface. After selecting the source and target directories, 
  the script will process videos based on the instructions provided in the text file.
- The processed videos will be saved in a new output directory named with a timestamp.

Requirements:
- FFmpeg must be installed and accessible in the system PATH.
- Python 3.x environment.

Installation of FFmpeg (for video processing):
- **Conda (recommended)**: 
  ```bash
  conda install -c conda-forge ffmpeg
"""

import os
import time
import subprocess
from tkinter import filedialog, messagebox, simpledialog

def process_videos(
    source_dir,
    target_dir,
    use_text_file=False,
    text_file_path=None,
):
    # Cria um novo diretório com timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"mergedvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Usa o arquivo de texto, se especificado
    if use_text_file and text_file_path:
        video_files = []
        with open(text_file_path, "r") as file:
            for line in file.readlines():
                line = line.strip()
                if "," in line:
                    # Dividir por vírgula para separar o vídeo de origem e de destino
                    source_video, target_video = line.split(",")
                    video_files.append(
                        (
                            os.path.join(source_dir, source_video.strip()),  # Caminho do vídeo de origem
                            os.path.join(source_dir, target_video.strip()),  # Caminho do segundo vídeo
                            os.path.join(output_dir, f"{source_video.strip()}_merged.mp4")  # Nome do arquivo final
                        )
                    )
                else:
                    print(f"Erro: A linha '{line}' não está formatada corretamente. Deve haver uma vírgula separando os dois vídeos.")
    else:
        print("Erro: Nenhum arquivo de texto foi fornecido ou selecionado.")
        return

    # Itera sobre os pares de vídeos e aplica o processo de merge
    for source_video, target_video, output_video in video_files:
        try:
            print(f"Processing video pair: {source_video} and {target_video}")

            # Comando do FFmpeg para concatenar os dois vídeos
            ffmpeg_command = [
                "ffmpeg", 
                "-i", source_video,
                "-i", target_video,
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",  # Concatena os dois vídeos
                "-c:v", "libx264", 
                output_video
            ]
            
            # Executa o comando do FFmpeg
            subprocess.run(ffmpeg_command, check=True)

            print(f"Video processed and saved to: {output_video}")
        except Exception as e:
            print(f"Error processing video {source_video} and {target_video}: {e}")

def process_videos_gui():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting video processing...")

    # Ask user to select source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Ask user to select target directory
    target_dir = filedialog.askdirectory(title="Select Target Directory")
    if not target_dir:
        messagebox.showerror("Error", "No target directory selected.")
        return

    # Ask if the user wants to use a text file
    use_text_file = messagebox.askyesno(
        "Use Text File", "Do you want to use a text file (videos_e_frames.txt)?"
    )
    text_file_path = None
    if use_text_file:
        text_file_path = filedialog.askopenfilename(
            title="Select videos_e_frames.txt", filetypes=[("Text files", "*.txt")]
        )
        if not text_file_path:
            messagebox.showerror("Error", "No text file selected.")
            return

    # Call the process_videos function with the collected inputs
    process_videos(
        source_dir, target_dir, use_text_file, text_file_path
    )

if __name__ == "__main__":
    process_videos_gui()
