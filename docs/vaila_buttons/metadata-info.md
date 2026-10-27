# Metadata Info Tool

## Overview

The **Metadata info** tool (`numberframes.py`) provides scientific-grade video metadata extraction and **Adjust real FPS** (retag container FPS before Cut / Sync).

GUI path: *vailá* Frame C → Video → **Metadata info** (chooser: Extract metadata | Adjust real FPS).

## Key Features

- High-precision ffprobe metadata (frame count, display/avg FPS, resolution, codec, duration)
- Parallel directory scan; writes `video_metadata_basic_<ts>.txt` + `video_metadata_full_<ts>.json`
- **Adjust real FPS**: remux (stream copy via `setts` + MP4 timescale) keeping every frame; re-encode fallback; writes `processed_fpsfix_<timestamp>/` + `fpsfix_summary.csv`

## How to Use

### Extract metadata

1. Launch *vailá* → **Metadata info** → **Extract metadata**
2. Select directory containing videos

### Adjust real FPS

Use when joined clips were stamped with the wrong FPS but frame content is at the capture rate. Cut Video and Make Sync file both use **container** FPS.

1. **Metadata info** → **Adjust real FPS**
2. Enter real FPS once for the batch (`29.97` or `30000/1001`)
3. Select the directory

CLI:

```bash
uv run vaila/numberframes.py --set-fps 29.97 --dir /path/to/videos --cli
uv run vaila/numberframes.py --set-fps 30000/1001 video1.mp4 video2.mp4 --cli
```

Semantics: does **not** drop/duplicate frames to stretch playback. Duration becomes `nb_frames / real_fps`.

## Related Tools

- **Cut Videos** (`cutvideo.py`) — cuts using container FPS
- **Make Sync file** (`syncvid.py`) — sync math must match file playback rate; retag here if stamped FPS is wrong

## Documentation

- [Full Help](../../vaila/help/numberframes.html)
- [Markdown](../../vaila/help/numberframes.md)
