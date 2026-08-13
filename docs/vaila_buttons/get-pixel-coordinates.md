# getpixelvideo.py — Get Pixel Coordinates

**Button:** Frame C → Video and Image → **GetPixelCoord**  
**Module:** [`vaila/getpixelvideo.py`](../../vaila/getpixelvideo.py)  
**Help:** [`vaila/help/getpixelvideo.md`](../../vaila/help/getpixelvideo.md) · [HTML](../../vaila/help/getpixelvideo.html)  
**Version:** 0.3.105 · **Updated:** 13 August 2026

## Overview

Mark and save pixel coordinates on video frames or PNG sequences (zoom, pan,
templates, FIFA pitch, ML dataset export).

## Opening media (v0.3.105)

One file picker — **no** video-vs-PNG question. Type is auto-detected:

- Video file → video
- Lone PNG → single image
- Any PNG in a folder with multiple PNGs → PNG sequence
- Directory (CLI) → PNG sequence

```bash
uv run vaila/getpixelvideo.py
uv run vaila/getpixelvideo.py -f VIDEO.mp4
uv run vaila/getpixelvideo.py -d /path/to/png_frames
```

See the full help page for key bindings, Load Track CSV, FIFA mode, and ML export.
