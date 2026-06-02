# Cut Videos Tool

## Overview

The **Cut Videos** tool ([`vaila/cutvideo.py`](../../vaila/cutvideo.py)) provides an interactive pygame-based video cutting interface with **scientific precision** for research applications. Mark cut points frame-by-frame, save cuts to TOML, and export segments with preserved FPS metadata (ffmpeg-first, OpenCV fallback).

## Key Features

- **Frame-accurate cutting** via ffmpeg (ffprobe metadata; inclusive frame counts)
- **TOML cut storage** (`{video}_cuts.toml`) with 1-based frames and `.6f` timestamps
- **Custom output base name** — prefix for exported clips (`trial01_frame_10_to_20.mp4`) via **B** or **Base name** button
- **Per-cut CSV/TXT names** — clean clip names (`jump01.mp4`, `jump02.mp4`) via **N** or **Names CSV**
- **Cut timeline feedback** — clickable/draggable strip: blue range, green start, orange end, yellow pending start, white playhead
- **Cut-marker keyboard navigation** — Shift+Left/Right cycles previous/next start/end marker with wraparound, matching Get Pixel Video
- **Responsive export progress** — cancellable dialog remains active while ffmpeg/OpenCV renders final clips
- **Audio waveform panel** (A/M), loop control, zoom/pan, auto-fit (0)
- **Sync file support** (TXT from Make Sync File) and batch apply to other videos in the folder
- **Scrollable in-app help** (H)

## How to Use

### Accessing the Tool

1. Launch vailá (`uv run vaila.py`)
2. **Frame C → Video and Image → C_B_r4_c2 - Cut Video**
3. Select a video file (`.mp4`, `.avi`, `.mov`, `.mkv`)

### Workflow

1. **Navigate** — arrows, slider, G (frame), T (time)
2. **Optional: set names** — **B** / **Base name** for one prefix, or **N** / **Names CSV** for one clean name per cut
3. **Mark cuts** — **S** start, **E** end (paused); review colored feedback strip
4. **Review** — **L** list cuts; click/drag strip to revisit marker positions
5. **Save / export** — **ESC** → saves TOML → optional ffmpeg export with cancellable progress → optional batch on sibling videos

### Controls (summary)

| Area | Keys / UI |
|------|-----------|
| Playback | Space, ←/→, Shift+←/→ (previous/next cut marker), ↑/↓ (±60), slider, cut-strip click/drag |
| Audio | A (panel), M (mute), Loop button |
| Window | 0 (auto-fit), +/- / wheel (zoom), MMB (pan) |
| Cuts | S, E, R, D/Delete, L, C (labels CSV) |
| Markers | Shift+←/→, Home/End, PageUp/PageDown |
| Files | F (load TOML/sync), **B** (base name), **N** (per-cut names), ESC (save + render) |
| Help | H or Help button |

### Bottom bar layout

- **Left:** colored cut strip above slider; `Out: {prefix}_frame_…` preview of export filenames
- **Right:** **Loop**, **Help**, **Names CSV**, **Base name** (active options show ✓)

## Output Files

| Item | Pattern |
|------|---------|
| Cuts metadata | `{video_stem}_cuts.toml` (next to source video) |
| Output folder (single) | `{video}_vailacut_{timestamp}` |
| Output folder (sync) | `{video}_sync_vailacut_{timestamp}` |
| Output folder (batch) | `vailacut_{source}_batch_{timestamp}` |
| Cut clips | `{base}_frame_{start}_to_{end}.mp4` (1-based; `{base}` = custom prefix or video stem) |

The **folder** name always uses the source video stem; only **clip filenames** use the custom base name when set.

## Scientific Precision

- Exact FPS from ffprobe (`r_frame_rate` / `avg_frame_rate`)
- Duration from inclusive `frame_count = end - start + 1`
- Times stored/displayed with 6 decimal places

## Requirements

- **ffmpeg / ffprobe** — recommended for precise cuts and audio waveform
- **OpenCV**, **pygame** — player and fallback cutter

## Related Tools

- **Metadata Info** (`numberframes.py`) — analyze video metadata
- **Make Sync File** (`syncvid.py`) — multi-camera sync TXT
- **Resize Video** (`resize_video.py`)

## Documentation

- [Help HTML](../../vaila/help/cutvideo.html)
- [Help Markdown](../../vaila/help/cutvideo.md)

**Last updated:** 2026-06-02 (v0.3.47)
