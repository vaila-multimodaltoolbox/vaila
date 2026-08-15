# extractpng

## Module Information

- **Category:** Tools
- **File:** `vaila/extractpng.py`
- **Version:** 0.3.105
- **Updated:** 13 August 2026
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** Yes (one window)

## Description

*vailá* Video ↔ PNG tool: extract PNG frames from videos, build videos from PNG
sequences, or grab selected frames. One simple Tkinter window; full CLI for
headless runs.

## GUI

Frame C → **Video↔PNG**, or:

```bash
uv run vaila/extractpng.py
```

One window:

1. Choose mode: **Video → PNG** / **PNG → Video** / **Select frames**
2. Browse input (and optional output)
3. Set options for the mode
4. **Run**

On Run, the terminal prints a copy-paste CLI mirror (`>> vaila/extractpng`).

## CLI

```bash
uv run vaila/extractpng.py extract -i /path/to/videos
uv run vaila/extractpng.py extract -i /path/to/videos -o /path/to/out --pattern %09d.png
uv run vaila/extractpng.py create -i /path/to/png_dirs --fps 30 --codec 264
uv run vaila/extractpng.py frames -i VIDEO.mp4 --frames 0,3,5,7
```

## Notes

- FFmpeg `-hwaccel auto` is an **input** option (placed before `-i`). There is
  no forced `hevc_cuvid` decoder; software decode is the automatic fallback.
- Default PNG pattern: `%09d.png`
- Batch extract writes `vaila_extractpng_<timestamp>/<stem>_png/` plus
  `video_info.txt` per video.

---

**Part of** *vailá* — Multimodal Toolbox
