"""
================================================================================
YouTube High Quality Downloader - vaila_ytdown.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: March 2025
Version: 0.2.0

Description:
------------
This script downloads videos from YouTube in the highest quality possible,
prioritizing highest resolution and framerate (FPS).

Key Features:
- Downloads videos in highest resolution available (up to 8K)
- Prioritizes streams with higher FPS (60fps when available)
- Automatically selects best video and audio quality
- Shows detailed video information including resolution and FPS
- Downloads with progress tracking
- Batch download support for playlists
- Uses yt-dlp for maximum compatibility and regular updates

Requirements:
- yt-dlp (pip install yt-dlp)
- ffmpeg (must be installed on your system and in PATH)
================================================================================
"""

import argparse
import os
import re
import shutil
import sys
from datetime import datetime

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

# Try to import yt-dlp
try:
    import yt_dlp  # type: ignore
except ImportError:
    print("Error: yt-dlp package is required. Install it with:")
    print("pip install yt-dlp")
    sys.exit(1)

# Try to import tkinter for GUI
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. Running in CLI mode only.")

# Rich console for pretty output
console = Console()


# Simplified function to read URLs from file - no resolution parsing needed
def read_urls_from_file(file_path):
    """Read YouTube URLs from a text file (one per line)."""
    urls = []
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith("#"):  # Ignore empty lines and comments
                    urls.append(url)
        return urls
    except Exception as e:
        raise Exception(f"Error reading URL file: {str(e)}")


class YTDownloader:
    def __init__(self):
        """Initialize the downloader with default settings."""
        self.output_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        self.current_video_title = ""
        self.progress_callback = None
        self.status_callback = None

        # Check if ffmpeg is available
        self.ffmpeg_available = self._check_ffmpeg()
        if not self.ffmpeg_available:
            console.print(
                "[yellow]Warning: ffmpeg not found in PATH. Using yt-dlp's embedded version.[/yellow]"
            )

    def _check_ffmpeg(self):
        """Check if ffmpeg is available in the system path."""
        return shutil.which("ffmpeg") is not None

    def get_video_info(self, url):
        """Get detailed information about the video with enhanced resolution and FPS tracking."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": "best",
            "simulate": True,
            "dump_single_json": True,
            "no_check_certificate": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Process all available formats to get comprehensive quality options
                available_formats = []

                if "formats" in info:
                    video_formats = [
                        f for f in info["formats"] if f.get("vcodec", "none") != "none"
                    ]

                    # Group formats by resolution and find best FPS for each
                    resolution_formats = {}
                    for fmt in video_formats:
                        height = fmt.get("height", 0)
                        fps = fmt.get("fps", 0)

                        key = f"{fmt.get('width', 0)}x{height}"
                        if key not in resolution_formats or fps > resolution_formats[key]["fps"]:
                            resolution_formats[key] = {
                                "resolution": key,
                                "height": height,
                                "width": fmt.get("width", 0),
                                "fps": fps,
                                "format_id": fmt.get("format_id", ""),
                                "ext": fmt.get("ext", ""),
                                "filesize": fmt.get("filesize", 0),
                                "vcodec": fmt.get("vcodec", ""),
                            }

                    # Convert to sorted list (highest resolution first)
                    available_formats = sorted(
                        resolution_formats.values(),
                        key=lambda x: (x["height"], x["fps"]),
                        reverse=True,
                    )

                return {
                    "title": info.get("title", "Unknown"),
                    "uploader": info.get("uploader", "Unknown"),
                    "duration": info.get("duration", 0),
                    "upload_date": info.get("upload_date", ""),
                    "available_formats": available_formats,
                    "url": url,
                }
        except Exception as e:
            console.print(f"[red]Error getting video info: {str(e)}[/red]")
            # Return basic info so download can still proceed
            return {
                "title": "Unknown",
                "url": url,
                "available_formats": [],
            }

    def download_video(self, url, output_dir=None, filename_prefix=""):
        """Download video prioritizing highest FPS regardless of resolution."""
        if output_dir:
            self.output_dir = output_dir

        # Create timestamp for unique folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.output_dir, f"vaila_ytdownload_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # Format spec: Let yt-dlp decide best quality available (default behavior)
        format_spec = "bestvideo+bestaudio/best"
        console.print("[blue]Downloading best available quality (video+audio)[/blue]")

        # Progress hook for download updates
        def progress_hook(d):
            if d["status"] == "downloading":
                if self.progress_callback:
                    self.progress_callback(d)

                # Also print progress to console
                p = d.get("_percent_str", "0%")
                size = d.get("_total_bytes_str", "Unknown")
                speed = d.get("_speed_str", "Unknown speed")
                eta = d.get("_eta_str", "Unknown")

                status_msg = f"\rDownloading: {p} of {size} at {speed}, ETA: {eta}"
                console.print(status_msg, end="")

            elif d["status"] == "finished":
                console.print("\nDownload complete. Processing video...")
                if self.status_callback:
                    self.status_callback("Merging video and audio...")

        # Prepare filename template with prefix if provided
        outtmpl = os.path.join(
            save_dir,
            f"{filename_prefix + '_' if filename_prefix else ''}%(title)s.%(ext)s",
        )

        # Set up download options with max FPS preference
        ydl_opts = {
            "format": format_spec,
            "outtmpl": outtmpl,
            "progress_hooks": [progress_hook],
            "merge_output_format": "mp4",
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            "writethumbnail": False,
            "writeinfojson": False,
            "no_check_certificate": True,
        }

        try:
            # First get extended video information for the detailed info file
            try:
                console.print(f"[blue]Getting detailed info for: {url}[/blue]")
                video_info = self.get_video_info(url)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get detailed info: {str(e)}[/yellow]")
                video_info = {"url": url, "available_formats": []}

            # Now download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                self.current_video_title = info.get("title", "Unknown")

                # Get the actual filename that was downloaded
                if info.get("requested_downloads"):
                    actual_filename = info["requested_downloads"][0]["filepath"]
                else:
                    # Try to guess the filename
                    actual_filename = os.path.join(save_dir, f"{self.current_video_title}.mp4")

                # Create a comprehensive information file with available resolutions and FPS
                info_file = os.path.join(save_dir, "video_info.txt")
                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(f"Title: {info.get('title', 'Unknown')}\n")
                    f.write(f"Channel: {info.get('uploader', 'Unknown')}\n")
                    f.write(f"URL: {url}\n")
                    f.write(
                        f"Downloaded resolution: {info.get('width', 0)}x{info.get('height', 0)}\n"
                    )
                    f.write(f"Downloaded FPS: {info.get('fps', 0)}\n")
                    f.write(f"Duration: {info.get('duration', 0)} seconds\n")
                    f.write(f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    # Add available formats information sorted by FPS first, then resolution
                    f.write("AVAILABLE RESOLUTIONS AND FPS OPTIONS (sorted by FPS):\n")
                    f.write("=================================================\n")

                    if video_info.get("available_formats"):
                        # Sort formats by FPS first, then by resolution
                        sorted_formats = sorted(
                            video_info["available_formats"],
                            key=lambda x: (x.get("fps", 0), x.get("height", 0)),
                            reverse=True,
                        )

                        for i, fmt in enumerate(sorted_formats, 1):
                            f.write(
                                f"{i}. FPS: {fmt.get('fps')} | Resolution: {fmt.get('resolution')}\n"
                            )
                            f.write(f"   Video codec: {fmt.get('vcodec')}\n")
                            filesize_mb = fmt.get("filesize", 0) / (1024 * 1024)
                            if filesize_mb > 0:
                                f.write(f"   Approximate size: {filesize_mb:.1f} MB\n")
                            f.write("\n")
                    else:
                        f.write("Could not retrieve detailed format information.\n")

                console.print(f"\n[green]Download successful:[/green] {self.current_video_title}")
                console.print(f"[blue]Saved to:[/blue] {actual_filename}")
                console.print(
                    f"[blue]Resolution:[/blue] {info.get('width', 0)}x{info.get('height', 0)}"
                )
                console.print(f"[blue]FPS:[/blue] {info.get('fps', 0)}")

                if self.status_callback:
                    self.status_callback(f"Download complete: {actual_filename}")

                return actual_filename
        except Exception as e:
            error_msg = f"Error downloading video: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            if self.status_callback:
                self.status_callback(f"Error: {error_msg}")
            raise Exception(error_msg)

    def download_audio(self, url, output_dir=None, filename_prefix=""):
        """Download audio only from a YouTube URL as MP3."""
        if output_dir:
            self.output_dir = output_dir

        # Create timestamp for unique folder (optional, maybe save directly to output_dir?)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Define diretório de salvamento (pode ser ajustado, talvez sem subpasta timestamp para áudio?)
        save_dir = os.path.join(self.output_dir, f"vaila_ytaudio_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        console.print(f"[blue]Downloading audio only (MP3) for: {url}[/blue]")

        def progress_hook(d):
            if d["status"] == "downloading":
                if self.progress_callback:
                    self.progress_callback(d)
                p = d.get("_percent_str", "0%")
                size = d.get("_total_bytes_str", "Unknown")
                speed = d.get("_speed_str", "Unknown speed")
                eta = d.get("_eta_str", "Unknown")
                status_msg = f"\rDownloading Audio: {p} of {size} at {speed}, ETA: {eta}"
                console.print(status_msg, end="")
            elif d["status"] == "finished":
                console.print("\nAudio download complete. Converting to MP3...")
                if self.status_callback:
                    self.status_callback("Converting audio to MP3...")

        outtmpl = os.path.join(
            save_dir,
            f"{filename_prefix + '_' if filename_prefix else ''}%(title)s.%(ext)s",
        )

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "progress_hooks": [progress_hook],
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",  # Pode ajustar a qualidade (ex: '320')
                }
            ],
            "writethumbnail": False,
            "writeinfojson": False,  # Pode querer manter True para ter info
            "no_check_certificate": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                self.current_video_title = info.get("title", "Unknown")

                # yt-dlp geralmente ajusta a extensão no post-processing
                # mas podemos tentar obter o nome final se necessário
                actual_filename = ydl.prepare_filename(info)
                # Corrige a extensão para .mp3 se o prepare_filename não o fez
                base, _ = os.path.splitext(actual_filename)
                actual_filename_mp3 = base + ".mp3"

                # Renomeia se o arquivo final não for .mp3 (caso raro)
                if os.path.exists(actual_filename) and not os.path.exists(actual_filename_mp3):
                    try:
                        os.rename(actual_filename, actual_filename_mp3)
                        actual_filename = actual_filename_mp3
                    except OSError as e:
                        console.print(
                            f"[yellow]Warning: Could not rename output file to .mp3: {e}[/yellow]"
                        )
                        actual_filename = actual_filename  # Mantém o nome original se falhar

                console.print(
                    f"\n[green]Audio download successful:[/green] {self.current_video_title}"
                )
                console.print(f"[blue]Saved as MP3 to:[/blue] {actual_filename}")

                if self.status_callback:
                    self.status_callback(f"Audio download complete: {actual_filename}")

                return actual_filename
        except Exception as e:
            error_msg = f"Error downloading audio: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            if self.status_callback:
                self.status_callback(f"Error: {error_msg}")
            raise Exception(error_msg)

    def download_playlist(self, playlist_url, output_dir=None):
        """Download all videos in a YouTube playlist."""
        if output_dir:
            self.output_dir = output_dir

        try:
            # First get playlist info
            info = self.get_video_info(playlist_url)

            if not info.get("is_playlist"):
                raise Exception("The URL does not appear to be a playlist.")

            playlist_title = info.get("title", "Unknown_Playlist")
            entries = info.get("entries", [])

            if not entries:
                raise Exception("No videos found in this playlist.")

            # Create timestamp for unique folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = re.sub(r'[\\/*?:"<>|]', "_", playlist_title)
            playlist_dir = os.path.join(
                self.output_dir, f"vaila_ytplaylist_{safe_title}_{timestamp}"
            )
            os.makedirs(playlist_dir, exist_ok=True)

            # Create playlist info file
            info_file = os.path.join(playlist_dir, "playlist_info.txt")
            with open(info_file, "w", encoding="utf-8") as f:
                f.write(f"Playlist: {playlist_title}\n")
                f.write(f"Downloaded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total videos: {len(entries)}\n\n")
                f.write("Videos in this playlist:\n")
                f.write("-" * 60 + "\n")

            console.print(f"[bold]Downloading playlist:[/bold] {playlist_title}")
            console.print(f"[bold]Total videos:[/bold] {len(entries)}")
            console.print(f"[bold]Output directory:[/bold] {playlist_dir}")

            # Download each video
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("[cyan]Downloading playlist", total=len(entries))

                for i, entry in enumerate(entries, 1):
                    video_url = entry.get("url")
                    video_title = entry.get("title", f"Video {i}")

                    progress.update(task, description=f"[{i}/{len(entries)}] {video_title[:40]}...")

                    try:
                        # Add number prefix to keep videos in order
                        prefix = f"{i:03d}"
                        self.download_video(
                            video_url, output_dir=playlist_dir, filename_prefix=prefix
                        )

                        # Add to info file
                        with open(info_file, "a", encoding="utf-8") as f:
                            f.write(f"{i}. {video_title}\n")

                        # Update progress
                        progress.update(task, advance=1)

                    except Exception as e:
                        console.print(f"[red]Error downloading video {i}: {str(e)}[/red]")

                        # Add error to info file
                        with open(info_file, "a", encoding="utf-8") as f:
                            f.write(f"{i}. ERROR: {video_title} - {str(e)}\n")

                        # Update progress despite error
                        progress.update(task, advance=1)
                        continue

            console.print("\n[green]Playlist download complete![/green]")
            console.print(f"[blue]Saved to:[/blue] {playlist_dir}")

            return playlist_dir

        except Exception as e:
            error_msg = f"Error downloading playlist: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            raise Exception(error_msg)

    def download_from_file(self, file_path, output_dir=None, audio_only=False):
        """Download all items listed in a text file (video or audio)."""
        if output_dir:
            self.output_dir = output_dir

        urls = read_urls_from_file(file_path)
        if not urls:
            raise Exception("No URLs found in the file.")

        # Create timestamp folder for all downloads
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_type = "audio" if audio_only else "batch"
        batch_dir = os.path.join(self.output_dir, f"vaila_{folder_type}_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)

        # Create log file
        log_file = os.path.join(batch_dir, "download_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Batch download started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Download type: {'Audio (MP3)' if audio_only else 'Video (Highest FPS)'}\n")
            f.write(f"Total URLs: {len(urls)}\n\n")
            f.write("Results:\n")
            f.write("-" * 60 + "\n")

        content_type = "audio tracks" if audio_only else "videos"
        console.print(f"[bold]Starting batch download of {len(urls)} {content_type}[/bold]")
        console.print(f"[bold]Output directory:[/bold] {batch_dir}")

        if not audio_only:
            console.print("[bold]Priority:[/bold] Highest FPS available for each video")

        success_count = 0
        fail_count = 0

        for i, url in enumerate(urls, 1):
            console.print(f"\n[bold][{i}/{len(urls)}] Processing:[/bold] {url}")

            # Create a numbered folder for each item
            item_dir = os.path.join(batch_dir, f"{i:03d}")
            os.makedirs(item_dir, exist_ok=True)

            try:
                # Choose download function based on audio_only flag
                if audio_only:
                    output_file = self.download_audio(
                        url, output_dir=item_dir, filename_prefix=f"{i:03d}"
                    )
                else:
                    output_file = self.download_video(
                        url, output_dir=item_dir, filename_prefix=f"{i:03d}"
                    )

                # Log success
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{i}. SUCCESS: {url} -> {os.path.basename(output_file)}\n")

                success_count += 1

            except Exception as e:
                error_msg = str(e)
                content_type = "audio" if audio_only else "video"
                console.print(f"[red]Error downloading {content_type} {i}: {error_msg}[/red]")

                # Log error
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{i}. ERROR: {url} - {error_msg}\n")

                fail_count += 1

        # Final summary
        console.print("\n[green]Batch download complete![/green]")
        console.print(
            f"[blue]Total: {len(urls)}, Success: {success_count}, Failures: {fail_count}[/blue]"
        )
        console.print(f"[blue]Saved to:[/blue] {batch_dir}")

        # No final do método load_url_file, antes de limpar recursos
        # Trazer a janela para frente antes da mensagem final
        self.root.lift()
        self.root.focus_force()
        self.root.update()

        messagebox.showinfo(
            "Batch Download Complete",
            f"All videos have been downloaded to:\n{batch_dir}",
            parent=self.root,
        )

        return batch_dir


# GUI Implementation (if tkinter is available)
if TKINTER_AVAILABLE:

    class DownloaderGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("vailá YOUTUBE DOWNLOADER")
            self.root.geometry("800x720")  # Reduzir o tamanho inicial da janela
            self.root.minsize(700, 600)  # Ajustar o tamanho mínimo da janela

            # Create downloader instance
            self.downloader = YTDownloader()

            # Main frame with scrollbar
            canvas = tk.Canvas(root)
            scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Main frame content
            main_frame = ttk.Frame(scrollable_frame, padding=15)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Criar um frame para o título
            title_frame = ttk.Frame(main_frame)
            title_frame.pack(pady=(0, 5), fill=tk.X)

            # Usar dois labels separados em vez do widget Text
            # O primeiro label para "vailá" em itálico
            vaila_label = ttk.Label(title_frame, text="vailá", font=("Arial", 16, "bold", "italic"))
            vaila_label.pack(side=tk.LEFT)

            # O segundo label para "YOUTUBE DOWNLOADER" em fonte normal
            downloader_label = ttk.Label(
                title_frame, text=" YOUTUBE DOWNLOADER", font=("Arial", 16, "bold")
            )
            downloader_label.pack(side=tk.LEFT)

            desc_label = ttk.Label(
                main_frame,
                text="Download YouTube videos with highest framerate (FPS)",
                font=("Arial", 12),
            )
            desc_label.pack(pady=(0, 15))

            # Output directory section
            dir_frame = ttk.LabelFrame(main_frame, text="Save Location", padding=10)
            dir_frame.pack(fill=tk.X, pady=10)

            # Directory display
            self.output_dir_var = tk.StringVar(value=os.path.expanduser("~/Downloads"))
            ttk.Label(dir_frame, text="Videos will be saved to:").pack(anchor=tk.W, pady=(0, 5))

            dir_path = ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=60)
            dir_path.pack(fill=tk.X, pady=5)

            browse_btn = ttk.Button(dir_frame, text="Browse...", command=self.browse_dir)
            browse_btn.pack(anchor=tk.W, pady=5)

            # Status area
            status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
            status_frame.pack(fill=tk.X, pady=10)

            self.status_var = tk.StringVar(
                value="Ready - Please select output directory and load a URL file"
            )
            ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 10, "bold")).pack(
                anchor=tk.W
            )

            self.progress_bar = ttk.Progressbar(
                status_frame, orient=tk.HORIZONTAL, length=100, mode="determinate"
            )
            self.progress_bar.pack(fill=tk.X, expand=True, pady=10)

            # Log display
            log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
            log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            self.log_text = tk.Text(
                log_frame, height=5, width=80, wrap=tk.WORD
            )  # Reduzir a altura do log
            self.log_text.pack(fill=tk.BOTH, expand=True)

            # Load URL file section
            file_frame = ttk.LabelFrame(main_frame, text="Load URLs from File", padding=10)
            file_frame.pack(fill=tk.X, pady=10)

            ttk.Label(
                file_frame,
                text="Load a text file with YouTube URLs (one per line)",
                font=("Arial", 11),
            ).pack(anchor=tk.W, pady=5)

            file_button_frame = ttk.Frame(file_frame)
            file_button_frame.pack(fill=tk.X, pady=5)

            load_file_button = ttk.Button(
                file_button_frame,
                text="SELECT URL FILE",
                command=self.load_url_file,
                style="Accent.TButton",
            )
            load_file_button.pack(side=tk.LEFT, padx=5, pady=5)

            # Add instructions for URL file format
            instructions_frame = ttk.Frame(main_frame)
            instructions_frame.pack(fill=tk.X, pady=5)

            ttk.Label(
                instructions_frame,
                text="File Format: One YouTube URL per line, lines starting with # are ignored",
                font=("Arial", 9),
                foreground="gray",
            ).pack(pady=5)

            # Thread management
            self.download_thread = None

            # Log startup
            self.log("Downloader started - loading highest FPS version for all videos")
            self.log("Please select output directory and then load a URL file")

        def log(self, message):
            """Add a message to the log display safely."""
            try:
                if hasattr(self, "log_text") and self.log_text.winfo_exists():
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
                    self.log_text.see(tk.END)  # Scroll to the end
                print(f"[LOG] {message}")  # Sempre imprimir no console, independente do widget
            except Exception as e:
                print(f"[LOG ERROR] Couldn't log to UI: {str(e)}")
                print(f"[LOG] {message}")  # Garantir que a mensagem é impressa

        def update_status(self, message):
            """Update the status message."""
            self.status_var.set(message)
            self.log(message)
            self.root.update_idletasks()

        def browse_dir(self):
            """Open directory browser dialog."""
            try:
                # Trazer a janela para frente e forçar o foco
                self.root.lift()
                self.root.focus_force()
                # Breve pausa para garantir que a janela principal esteja visível
                self.root.update()

                directory = filedialog.askdirectory(
                    initialdir=os.path.expanduser("~"),
                    title="Select folder to save videos",
                    parent=self.root,  # Explicitamente definir a janela pai
                )

                if directory:
                    # Atualiza a variável e força a atualização da interface
                    self.output_dir_var.set(directory)
                    self.root.update_idletasks()  # Força a atualização dos widgets

                    # Garante que o Entry seja atualizado explicitamente
                    for widget in self.root.winfo_children():
                        widget.update()

                    self.update_status(f"Output directory set to: {directory}")

                    # Log para debug
                    self.log(f"Directory selected: {directory}")
                    self.log(f"Variable value: {self.output_dir_var.get()}")
            except Exception as e:
                self.log(f"Error selecting directory: {str(e)}")

        def load_url_file(self):
            """Load URLs from a text file and download with highest FPS priority or as MP3 audio."""
            try:
                # Trazer a janela para frente e forçar o foco
                self.root.lift()
                self.root.focus_force()
                # Breve pausa para garantir que a janela principal esteja visível
                self.root.update()

                file_path = filedialog.askopenfilename(
                    initialdir=os.path.expanduser("~"),
                    title="Select file with YouTube URLs",
                    filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
                    parent=self.root,  # Explicitamente definir a janela pai
                )

                if not file_path:
                    return

                # Read URLs from file
                self.log(f"Reading URLs from file: {file_path}")
                urls = read_urls_from_file(file_path)

                if not urls:
                    # Trazer a janela para frente antes de mostrar a mensagem
                    self.root.lift()
                    self.root.focus_force()
                    self.root.update()
                    messagebox.showwarning("Warning", "No URLs found in the file", parent=self.root)
                    return

                # Trazer a janela para frente antes dos próximos diálogos
                self.root.lift()
                self.root.focus_force()
                self.root.update()

                # Ask user for download type
                download_type = messagebox.askquestion(
                    "Download Type",
                    "Do you want to download as MP3 audio files?\n\n"
                    "Select 'Yes' for MP3 audio only.\n"
                    "Select 'No' for video with highest FPS.",
                    icon="question",
                    parent=self.root,
                )

                audio_only = download_type == "yes"
                content_type = "MP3 audio tracks" if audio_only else "videos with highest FPS"

                # Trazer a janela para frente antes do próximo diálogo
                self.root.lift()
                self.root.focus_force()
                self.root.update()

                # Show confirmation with URL preview
                confirm = messagebox.askyesno(
                    "Confirm Batch Download",
                    f"Do you want to download {len(urls)} {content_type}?\n\n"
                    f"First 3 URLs:\n" + "\n".join(urls[:3]) + ("\n..." if len(urls) > 3 else ""),
                    parent=self.root,
                )

                if not confirm:
                    return

                # Process each URL
                self.log(f"Processing {len(urls)} URLs from file")

                # Get output directory
                output_dir = self.output_dir_var.get()

                # Create a batch folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_type = "audio" if audio_only else "batch"
                batch_dir = os.path.join(output_dir, f"yt{folder_type}_{timestamp}")
                os.makedirs(batch_dir, exist_ok=True)

                # Create log file
                log_file = os.path.join(batch_dir, "batch_log.txt")
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(
                        f"Batch download started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"Total URLs: {len(urls)}\n\n")

                # Process each URL
                for i, url in enumerate(urls, 1):
                    try:
                        # Update status
                        self.update_status(f"Processing URL {i}/{len(urls)}")

                        # Create a subfolder for this video
                        item_dir = os.path.join(batch_dir, f"{i:03d}")
                        os.makedirs(item_dir, exist_ok=True)

                        if audio_only:
                            # Download as MP3 audio
                            self.downloader.download_audio(
                                url, output_dir=item_dir, filename_prefix=f"{i:03d}"
                            )
                            self.log(f"Downloaded MP3 ({i}/{len(urls)}): {url}")
                        else:
                            # Download as video with the existing code
                            # ... código existente para download de vídeo ...

                            # Format specification for best quality
                            format_spec = "bestvideo+bestaudio/best"

                            # Download options
                            ydl_opts = {
                                "format": format_spec,
                                "outtmpl": os.path.join(item_dir, "%(title)s.%(ext)s"),
                                "merge_output_format": "mp4",
                                "no_check_certificate": True,
                            }

                            # Download the video
                            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info = ydl.extract_info(url, download=True)
                                title = info.get("title", "Unknown")

                                # Create info file with FPS details
                                info_file = os.path.join(item_dir, "video_info.txt")
                                with open(info_file, "w", encoding="utf-8") as f:
                                    f.write(f"Title: {title}\n")
                                    f.write(f"URL: {url}\n")
                                    f.write(
                                        f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                    )
                                    f.write(
                                        f"Resolution: {info.get('width', 'unknown')}x{info.get('height', 'unknown')}\n"
                                    )
                                    f.write(f"FPS: {info.get('fps', 'unknown')}\n")
                                    f.write("Priority: Best overall quality\n")

                                # Log success
                                with open(log_file, "a", encoding="utf-8") as f:
                                    f.write(
                                        f"{i}. SUCCESS: {url} -> {title} ({info.get('fps', 'unknown')} FPS)\n"
                                    )

                                self.log(
                                    f"Downloaded ({i}/{len(urls)}): {title} @ {info.get('fps', 'unknown')} FPS"
                                )

                    except Exception as e:
                        # Log error
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(f"{i}. ERROR: {url} -> {str(e)}\n")

                        self.log(f"Error processing URL {i}/{len(urls)}: {str(e)}")

                # Show completion message
                self.update_status("Batch download complete")
                # No final do método load_url_file, antes de limpar recursos
                # Trazer a janela para frente antes da mensagem final
                self.root.lift()
                self.root.focus_force()
                self.root.update()

                messagebox.showinfo(
                    "Batch Download Complete",
                    f"All videos have been downloaded to:\n{batch_dir}",
                    parent=self.root,
                )

                # Adicionar esta linha para limpar recursos após conclusão
                self.cleanup_resources()

            except Exception as e:
                self.log(f"Error in batch download: {str(e)}")
                messagebox.showerror("Batch Download Error", str(e))

        def cleanup_resources(self):
            """Clean up resources to prevent hanging after download completion."""
            try:
                # Limpar referências a processos e threads
                if hasattr(self, "download_thread") and self.download_thread:
                    self.download_thread = None

                # Força o coletor de lixo para liberar recursos
                import gc

                gc.collect()

                # Atualizar a interface se ainda existir
                if self.root and self.root.winfo_exists():
                    self.root.update_idletasks()

                # Log a mensagem com segurança usando o método de log modificado
                self.log("Resources cleaned up successfully")
            except Exception as e:
                # Tentar registrar o erro, mas não lance outra exceção se falhar
                try:
                    print(f"[Error cleaning up resources] {str(e)}")
                except:
                    pass  # Último recurso - silenciar completamente


def run_ytdown():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Download YouTube videos or audio")
    parser.add_argument("-u", "--url", help="YouTube video or playlist URL")
    parser.add_argument("-f", "--file", help="Text file with YouTube URLs (one per line)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--no-gui", action="store_true", help="Force CLI mode (no GUI)")
    parser.add_argument(
        "-a", "--audio-only", action="store_true", help="Download audio only as MP3"
    )

    args = parser.parse_args()

    # Print script information
    console.print("=" * 70)
    console.print("[bold]YouTube Cinematic Downloader[/bold]")
    console.print("=" * 70)
    console.print(f"Running script: {os.path.basename(__file__)}")
    console.print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # CLI mode: if URL or file provided, or GUI not available or forced CLI
    if args.url or args.file or not TKINTER_AVAILABLE or args.no_gui:
        downloader = YTDownloader()

        if args.output:
            downloader.output_dir = args.output

        # Handle file with URLs
        if args.file:
            try:
                content_type = "MP3 audio" if args.audio_only else "videos"
                console.print(
                    f"\n[bold]Loading URLs from file to download {content_type}:[/bold] {args.file}"
                )
                downloader.download_from_file(
                    args.file, output_dir=args.output, audio_only=args.audio_only
                )
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
                sys.exit(1)

        # Handle YouTube URL
        elif args.url:
            try:
                url = args.url
                # Handle URLs with @ prefix (remove it if present)
                if url.startswith("@"):
                    url = url[1:]

                if args.audio_only:
                    console.print("\n[green]Starting audio download (MP3)...[/green]")
                    console.print(f"[bold]URL:[/bold] {url}")
                    downloader.download_audio(url, output_dir=args.output)
                else:
                    console.print("\n[green]Starting download of highest FPS version...[/green]")
                    console.print(f"[bold]URL:[/bold] {url}")

                    # Simplifiquei a chamada aqui para usar o método da classe
                    # em vez de recriar as opções do ydl_opts aqui.
                    # Assumindo que download_video use a lógica de maior FPS.
                    downloader.download_video(url, output_dir=args.output)

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
                sys.exit(1)
        else:
            # If no URL or file provided, prompt for URL
            console.print("\n[bold yellow]Please enter YouTube URL:[/bold yellow]")
            url = input().strip()

            if url:
                try:
                    console.print(
                        "\n[green]Starting download of highest quality version...[/green]"
                    )
                    downloader.download_video(url, output_dir=args.output)
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    sys.exit(1)
            else:
                console.print("[red]No URL provided. Exiting.[/red]")
                sys.exit(1)

    # GUI mode
    else:
        try:
            root = tk.Tk()
            # Fazer com que a janela sempre fique no topo
            root.attributes("-topmost", True)

            # Add custom style for better visibility
            style = ttk.Style()
            style.configure("TButton", font=("Arial", 10))
            style.configure("Accent.TButton", font=("Arial", 10, "bold"))
            style.configure("TLabel", font=("Arial", 10))
            style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))

            app = DownloaderGUI(root)

            # Após criar os componentes, podemos desligar o topmost
            # para permitir que o usuário alterne entre janelas se quiser
            root.after(1000, lambda: root.attributes("-topmost", False))

            # If URL was provided, pre-fill it
            if args.url:
                app.url_var.set(args.url)

            # If output dir was provided, set it
            if args.output:
                app.output_dir_var.set(args.output)

            root.mainloop()
        except Exception as e:
            console.print(f"[bold red]Error starting GUI: {str(e)}[/bold red]")
            console.print("Falling back to command line mode...")
            # If GUI fails, prompt for URL
            console.print("\n[bold yellow]Please enter YouTube URL:[/bold yellow]")
            url = input().strip()

            if url:
                try:
                    downloader = YTDownloader()
                    console.print(
                        "\n[green]Starting download of highest quality version...[/green]"
                    )
                    downloader.download_video(url, output_dir=args.output)
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    sys.exit(1)


if __name__ == "__main__":
    run_ytdown()
