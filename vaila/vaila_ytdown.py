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

import os
import sys
import re
import time
import subprocess
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import threading
import json
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
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
    from tkinter import ttk, filedialog, messagebox

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
        with open(file_path, "r", encoding="utf-8") as f:
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
                        if (
                            key not in resolution_formats
                            or fps > resolution_formats[key]["fps"]
                        ):
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

        # Format spec that prioritizes FPS over resolution
        # The 'fps>30' filter will prioritize videos with FPS greater than 30
        # When multiple high FPS versions exist, it will choose the one with best resolution
        format_spec = "bestvideo[fps>30]+bestaudio/bestvideo+bestaudio/best"
        console.print(f"[blue]Prioritizing highest FPS available[/blue]")

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
            f"{filename_prefix+'_' if filename_prefix else ''}%(title)s.%(ext)s",
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
                console.print(
                    f"[yellow]Warning: Could not get detailed info: {str(e)}[/yellow]"
                )
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
                    actual_filename = os.path.join(
                        save_dir, f"{self.current_video_title}.mp4"
                    )

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
                    f.write(
                        f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    )

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

                console.print(
                    f"\n[green]Download successful:[/green] {self.current_video_title}"
                )
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
                f.write(
                    f"Downloaded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
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
                task = progress.add_task(
                    f"[cyan]Downloading playlist", total=len(entries)
                )

                for i, entry in enumerate(entries, 1):
                    video_url = entry.get("url")
                    video_title = entry.get("title", f"Video {i}")

                    progress.update(
                        task, description=f"[{i}/{len(entries)}] {video_title[:40]}..."
                    )

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
                        console.print(
                            f"[red]Error downloading video {i}: {str(e)}[/red]"
                        )

                        # Add error to info file
                        with open(info_file, "a", encoding="utf-8") as f:
                            f.write(f"{i}. ERROR: {video_title} - {str(e)}\n")

                        # Update progress despite error
                        progress.update(task, advance=1)
                        continue

            console.print(f"\n[green]Playlist download complete![/green]")
            console.print(f"[blue]Saved to:[/blue] {playlist_dir}")

            return playlist_dir

        except Exception as e:
            error_msg = f"Error downloading playlist: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            raise Exception(error_msg)

    def download_from_file(self, file_path, output_dir=None):
        """Download all videos listed in a text file with highest FPS priority."""
        if output_dir:
            self.output_dir = output_dir

        urls = read_urls_from_file(file_path)
        if not urls:
            raise Exception("No URLs found in the file.")

        # Create timestamp folder for all downloads
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(self.output_dir, f"vaila_batch_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)

        # Create log file
        log_file = os.path.join(batch_dir, "download_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(
                f"Batch download started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total URLs: {len(urls)}\n\n")
            f.write("Results:\n")
            f.write("-" * 60 + "\n")

        console.print(f"[bold]Starting batch download of {len(urls)} URLs[/bold]")
        console.print(f"[bold]Output directory:[/bold] {batch_dir}")
        console.print(f"[bold]Priority:[/bold] Highest FPS available for each video")

        success_count = 0
        fail_count = 0

        for i, url in enumerate(urls, 1):
            console.print(f"\n[bold][{i}/{len(urls)}] Processing:[/bold] {url}")

            # Create a numbered folder for each video
            video_dir = os.path.join(batch_dir, f"{i:03d}")
            os.makedirs(video_dir, exist_ok=True)

            try:
                output_file = self.download_video(
                    url, output_dir=video_dir, filename_prefix=f"{i:03d}"
                )

                # Log success
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{i}. SUCCESS: {url} -> {os.path.basename(output_file)}\n")

                success_count += 1

            except Exception as e:
                error_msg = str(e)
                console.print(f"[red]Error downloading video {i}: {error_msg}[/red]")

                # Log error
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{i}. ERROR: {url} - {error_msg}\n")

                fail_count += 1

        # Final summary
        console.print(f"\n[green]Batch download complete![/green]")
        console.print(
            f"[blue]Total: {len(urls)}, Success: {success_count}, Failures: {fail_count}[/blue]"
        )
        console.print(f"[blue]Saved to:[/blue] {batch_dir}")

        return batch_dir


# GUI Implementation (if tkinter is available)
if TKINTER_AVAILABLE:

    class DownloaderGUI:
        def __init__(self, root):
            self.root = root
            self.root.title("vailá YOUTUBE DOWNLOADER")
            self.root.geometry("1024x960")
            self.root.minsize(800, 800)

            # Create downloader instance
            self.downloader = YTDownloader()

            # Main frame
            main_frame = ttk.Frame(root, padding=15)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Criar um frame para o título
            title_frame = ttk.Frame(main_frame)
            title_frame.pack(pady=(0, 5), fill=tk.X)
            
            # Usar dois labels separados em vez do widget Text
            # O primeiro label para "vailá" em itálico
            vaila_label = ttk.Label(
                title_frame, 
                text="vailá",
                font=("Arial", 16, "bold", "italic")
            )
            vaila_label.pack(side=tk.LEFT)
            
            # O segundo label para "YOUTUBE DOWNLOADER" em fonte normal
            downloader_label = ttk.Label(
                title_frame, 
                text=" YOUTUBE DOWNLOADER",
                font=("Arial", 16, "bold")
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
            ttk.Label(dir_frame, text="Videos will be saved to:").pack(
                anchor=tk.W, pady=(0, 5)
            )

            dir_path = ttk.Entry(dir_frame, textvariable=self.output_dir_var, width=60)
            dir_path.pack(fill=tk.X, pady=5)

            browse_btn = ttk.Button(
                dir_frame, text="Browse...", command=self.browse_dir
            )
            browse_btn.pack(anchor=tk.W, pady=5)

            # Status area
            status_frame = ttk.LabelFrame(main_frame, text="Status", padding=10)
            status_frame.pack(fill=tk.X, pady=10)

            self.status_var = tk.StringVar(
                value="Ready - Please select output directory and load a URL file"
            )
            ttk.Label(
                status_frame, textvariable=self.status_var, font=("Arial", 10, "bold")
            ).pack(anchor=tk.W)

            self.progress_bar = ttk.Progressbar(
                status_frame, orient=tk.HORIZONTAL, length=100, mode="determinate"
            )
            self.progress_bar.pack(fill=tk.X, expand=True, pady=10)

            # Log display
            log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
            log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

            self.log_text = tk.Text(log_frame, height=10, width=80, wrap=tk.WORD)
            self.log_text.pack(fill=tk.BOTH, expand=True)

            # Load URL file section
            file_frame = ttk.LabelFrame(
                main_frame, text="Load URLs from File", padding=10
            )
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
            """Add a message to the log display."""
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)  # Scroll to the end
            print(f"[LOG] {message}")  # Also print to console

        def update_status(self, message):
            """Update the status message."""
            self.status_var.set(message)
            self.log(message)
            self.root.update_idletasks()

        def browse_dir(self):
            """Open directory browser dialog."""
            try:
                directory = filedialog.askdirectory(
                    initialdir=os.path.expanduser("~"),
                    title="Select folder to save videos",
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
            """Load URLs from a text file and download with highest FPS priority."""
            try:
                file_path = filedialog.askopenfilename(
                    initialdir=os.path.expanduser("~"),
                    title="Select file with YouTube URLs",
                    filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
                )

                if not file_path:
                    return

                # Read URLs from file
                self.log(f"Reading URLs from file: {file_path}")
                urls = read_urls_from_file(file_path)

                if not urls:
                    messagebox.showwarning("Warning", "No URLs found in the file")
                    return

                # Show confirmation
                confirm = messagebox.askyesno(
                    "Confirm Batch Download",
                    f"Do you want to download {len(urls)} videos with highest FPS?\n\nFirst 3 URLs:\n"
                    + "\n".join(urls[:3])
                    + ("\n..." if len(urls) > 3 else ""),
                )

                if not confirm:
                    return

                # Process each URL
                self.log(f"Processing {len(urls)} URLs from file")

                # Get output directory
                output_dir = self.output_dir_var.get()

                # Create a batch folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_dir = os.path.join(output_dir, f"ytbatch_{timestamp}")
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
                        video_dir = os.path.join(batch_dir, f"{i:03d}")
                        os.makedirs(video_dir, exist_ok=True)

                        # Format specification for highest FPS
                        format_spec = (
                            "bestvideo[fps>30]+bestaudio/bestvideo+bestaudio/best"
                        )

                        # Download options
                        ydl_opts = {
                            "format": format_spec,
                            "outtmpl": os.path.join(video_dir, "%(title)s.%(ext)s"),
                            "merge_output_format": "mp4",
                            "no_check_certificate": True,
                        }

                        # Download the video
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=True)
                            title = info.get("title", "Unknown")

                            # Create info file with FPS details
                            info_file = os.path.join(video_dir, "video_info.txt")
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
                                f.write(f"Priority: Highest FPS available\n")

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
                messagebox.showinfo(
                    "Batch Download Complete",
                    f"All videos have been downloaded to:\n{batch_dir}",
                )

            except Exception as e:
                self.log(f"Error in batch download: {str(e)}")
                messagebox.showerror("Batch Download Error", str(e))


def run_ytdown():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download YouTube videos in highest quality"
    )
    parser.add_argument("-u", "--url", help="YouTube video or playlist URL")
    parser.add_argument(
        "-f", "--file", help="Text file with YouTube URLs (one per line)"
    )
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--no-gui", action="store_true", help="Force CLI mode (no GUI)")

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
                console.print(f"\n[bold]Loading URLs from file:[/bold] {args.file}")
                downloader.download_from_file(args.file, output_dir=args.output)
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

                console.print(
                    f"\n[green]Starting download of highest quality version...[/green]"
                )
                console.print(f"[bold]URL:[/bold] {url}")

                # Directly use yt-dlp without intermediate steps for more reliability
                ydl_opts = {
                    "format": "bestvideo+bestaudio/best",
                    "outtmpl": os.path.join(
                        args.output or os.path.expanduser("~/Downloads"),
                        "vaila_ytdownload_%(upload_date)s",
                        "%(title)s.%(ext)s",
                    ),
                    "merge_output_format": "mp4",
                    "no_check_certificate": True,
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    title = info.get("title", "Unknown")
                    filepath = ydl.prepare_filename(info)

                    console.print(f"[green]Download successful:[/green] {title}")
                    console.print(f"[blue]Saved to:[/blue] {filepath}")
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
                        f"\n[green]Starting download of highest quality version...[/green]"
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
            # Add custom style for better visibility
            style = ttk.Style()
            style.configure("TButton", font=("Arial", 10))
            style.configure("Accent.TButton", font=("Arial", 10, "bold"))
            style.configure("TLabel", font=("Arial", 10))
            style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))

            app = DownloaderGUI(root)

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
                        f"\n[green]Starting download of highest quality version...[/green]"
                    )
                    downloader.download_video(url, output_dir=args.output)
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    sys.exit(1)


if __name__ == "__main__":
    run_ytdown()
