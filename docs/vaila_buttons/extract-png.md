# Video ↔ PNG (`extractpng`)

**Button:** Frame C → Video and Image → **C_B_r1_c1 - Video↔PNG**  
**Handler:** `extract_png_from_videos`  
**Module:** [`vaila/extractpng.py`](../../vaila/extractpng.py)  
**Help:** [`vaila/help/extractpng.md`](../../vaila/help/extractpng.md) · [HTML](../../vaila/help/extractpng.html)  
**Version:** 0.4.7 · **Updated:** 02 October 2026

## Overview

One simple Tkinter window (and a full CLI) to:

1. **Video → PNG** — batch-extract frames from every video in a folder  
2. **PNG → Video** — build `.mp4` from any folder of PNGs (contiguous `%09d` from 0, or every PNG in filename order)  
3. **Select frames** — grab specific frame indices from one video  
4. **Random sample** — extract well-spaced, non-contiguous random frames from videos across directory trees (default 20 frames per video)

On GUI **Run**, the terminal prints a copy-paste CLI mirror (`>> vaila/extractpng`).

## GUI

1. Launch *vailá* (`uv run vaila.py`) → **Video↔PNG**, or `uv run vaila/extractpng.py`
2. Choose mode (radio buttons)
3. Browse **Input** (folder of videos / PNG dirs, or one video for Select frames)
4. Optional **Output** (empty → timestamped folder next to input)
5. Set options for the mode → **Run**

## CLI

```bash
uv run vaila/extractpng.py
uv run vaila/extractpng.py extract -i /path/to/videos
uv run vaila/extractpng.py extract -i /path/to/videos -o /path/to/out --pattern %09d.png
uv run vaila/extractpng.py create -i /path/to/png_dirs --fps 30 --codec 264
uv run vaila/extractpng.py frames -i VIDEO.mp4 --frames 0,3,5,7
uv run vaila/extractpng.py sample -i /path/to/videos -n 20 --recursive
```

## Notes

- FFmpeg `-hwaccel auto` is an **input** option (before `-i`). No forced `hevc_cuvid`; software decode is the fallback.
- Default PNG pattern: `%09d.png`. Create encodes every PNG in natural filename order through a temporary numbered sequence, with quality 18 and GOP 1 by default.
- Batch extract writes `vaila_extractpng_<timestamp>/<stem>_png/` plus `video_info.txt` per video.
- Random sampling writes `vaila_random_samples_<timestamp>/<stem>_random<N>/frame_<number>.png` and `manifest.csv` per video, plus `sampling_summary.csv` in the output root.
- Requires `ffmpeg` / `ffprobe` on `PATH`.
