# vaila_ytdown

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/vaila_ytdown.py`
- **Version:** 0.3.19
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes
- **HTML Help:** [vaila_ytdown.html](vaila_ytdown.html) (open in browser or use **? Help** in the GUI)

---

## How to run

The script can be used via **GUI** (default) or **CLI** (command line).

### GUI (default)

```bash
uv run python vaila/vaila_ytdown.py
```

1. Set **Save Location** (Browse…).
2. Choose **Video (highest FPS)** or **Audio Only (MP3)**.
3. **Direct:** paste URLs in the text box and click **DOWNLOAD FROM TEXT BOX**.
4. **Batch:** click **LOAD FROM FILE...** and select a `.txt` file with one URL per line.
5. Confirm; progress appears in the log. Use **? Help** in the window for more.

### CLI (command line)

```bash
# Single URL (video, best quality)
uv run python vaila/vaila_ytdown.py -u "https://www.youtube.com/watch?v=..."

# Single URL (audio only, MP3)
uv run python vaila/vaila_ytdown.py -u "https://www.youtube.com/watch?v=..." -a

# Batch from file (video)
uv run python vaila/vaila_ytdown.py -f urls.txt -o ~/Videos

# Batch from file (audio only)
uv run python vaila/vaila_ytdown.py -f urls.txt -a -o ~/Music

# Force CLI (no GUI)
uv run python vaila/vaila_ytdown.py --no-gui -u "https://..."

# Show all options
uv run python vaila/vaila_ytdown.py -h
```

---

## 📖 Description

YouTube High Quality Downloader for the vailá toolbox. Downloads videos in the highest quality possible, prioritizing resolution and framerate (FPS), or audio only as MP3.

### Key Features

- Downloads in highest resolution available (up to 8K)
- Prioritizes higher FPS (e.g. 60 fps when available)
- Automatically selects best video + audio quality
- Shows detailed video info (resolution, FPS)
- Progress tracking and batch download from URL files
- **Direct URL input** for quick downloads
- Audio-only mode: MP3 from YouTube URLs
- Uses **yt-dlp** for compatibility and updates

### Requirements

- **yt-dlp:** `pip install yt-dlp`
- **ffmpeg:** must be installed and in PATH (for merging video/audio)

---

## 🖥️ Usage

### GUI mode (default)

```bash
python -m vaila.vaila_ytdown
# or
uv run python vaila/vaila_ytdown.py
```

1. **Configuration**:
   - Set **Save Location** (Browse...).
   - Select **Download Type**: "Video (highest FPS)" or "Audio Only (MP3)".
2. **Download Methods**:
   - **Direct Input**: Paste URLs into the text box and click **DOWNLOAD FROM TEXT BOX**.
   - **File Batch**: Click **LOAD FROM FILE...** and choose a `.txt` file with one YouTube URL per line.
3. Confirm the action; downloads run with progress in the log.

Use **? Help** in the window to open the full HTML documentation in your browser.

### CLI mode

```bash
# Single URL (video, best quality)
python -m vaila.vaila_ytdown -u "https://www.youtube.com/watch?v=..."

# Single URL (audio only, MP3)
python -m vaila.vaila_ytdown -u "https://www.youtube.com/watch?v=..." -a

# File with URLs (video)
python -m vaila.vaila_ytdown -f urls.txt -o ~/Videos

# File with URLs (audio only)
python -m vaila.vaila_ytdown -f urls.txt -a -o ~/Music

# Force CLI (no GUI)
python -m vaila.vaila_ytdown --no-gui -u "https://..."
```

### URL file format

- One YouTube URL per line.
- Empty lines and lines starting with `#` are ignored.

---

## 🔧 Main Functions

- `read_urls_from_file` — Read URLs from a text file
- `run_ytdown` — Main entry point (CLI + GUI)
- `get_video_info` — Get video metadata and formats
- `download_video` — Download video (best quality)
- `download_audio` — Download audio only (MP3)
- `download_playlist` — Download full playlist
- `download_from_file` — Batch download from URL file
- `browse_dir` — GUI: choose output directory
- `load_url_file` — GUI: load URL file and start batch
- `cleanup_resources` — GUI: cleanup after batch

---

📅 **Updated:** 2026  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
