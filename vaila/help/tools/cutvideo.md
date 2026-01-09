# cutvideo.py

## 📋 Module Information

- **Category:** Tools  
- **File:** `vaila/cutvideo.py`  
- **Version:** 0.2.0  
- **Author:** Paulo Roberto Pereira Santiago  
- **Email:** paulosantiago@usp.br  
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila  
- **GUI Interface:** ✅ Yes  
- **License:** AGPL v3.0  

## 📖 Description
Interactive video cutting with frame-accurate navigation, TOML-based cut storage, and batch/sync workflows. Designed for biomechanics-grade precision with inclusive frame counts and high-resolution timestamps (.6f).

### Key Updates
- **TOML cuts**: Saves/loads `*_cuts.toml` with 1-based frames, `.6f` precision, `frame_count = end - start + 1` (duration = `frame_count / fps`), plus `output_dir` and per-cut `output_file` entries. Windows paths stored as POSIX to avoid escaping issues.
- **Audio waveform visualization**: VirtualDub-style audio panel with orange waveform line, synchronized playback, and mute/unmute control (hotkey `A` for panel, `M` for mute).
- **Audio playback**: Synchronized audio playback with video; automatically loops when loop mode is enabled.
- **Loop control**: Loop button (column layout above Help) to enable/disable video and audio looping.
- **Auto-fit window**: Hotkey `0` automatically adjusts window to maximize screen space usage while maintaining aspect ratio.
- **Marker navigation**: PageUp/PageDown cycle through every start/end marker; Home/End jump to current/last cut bounds.
- **Manual FPS input**: Hotkey `I`/`P` lets you override FPS via Tk dialog; UI updates instantly.
- **Scrollable help**: Help overlay scrolls via mouse wheel or arrow keys.
- **Output naming**: Single video → `{video}_vailacut_{timestamp}`; batch → `vailacut_{prefix}{source}_batch_{timestamp}`; sync batch → `vailacut_sync_{video}_{timestamp}`; files named with 1-based frame ranges.
- **Time display**: Caption shows filename + FPS; slider shows frame and time with `.6f` precision (`Frame: X/Y (t.TTTTTT/T.TTTTTT)`).

## 🔧 Main Functions (high level)
- `play_video_with_cuts()` — main UI/player and cut workflow.  
- `save_cuts_to_toml()` — writes TOML (1-based frames, .6f times, frame-count duration).  
- `load_cuts_from_toml()` / `load_cuts_from_toml_file()` — reads TOML; legacy `.txt` as fallback.  
- `cut_video_with_ffmpeg()` (precise) and `cut_video_with_opencv()` (fallback).  
- `batch_process_videos()` and `batch_process_sync_videos()` — reuse marked cuts across files.  
- `get_precise_video_metadata()` — ffprobe-first, OpenCV fallback.  
- `run_cutvideo()` — entry point (uses Tk file dialog then launches pygame UI).  

## 🎮 Controls (UI)
- **Playback/Navigation:** Space (play/pause), →/← (frame step), ↑/↓ (±60 frames), mouse on slider (jump).  
- **Audio Controls:** A (toggle audio waveform panel), M (mute/unmute audio), Loop button (enable/disable looping).  
- **Window Controls:** 0 (auto-fit window to screen), drag window edges (manual resize).  
- **Markers/Cuts:** S (start), E (end), R (reset start), D or Delete (remove last cut), L (list cuts).  
- **Navigation to markers:** PageUp/PageDown (prev/next marker), Home/End (start/end of current or first/last cut).  
- **Jump inputs:** G (go to frame), T (go to time in seconds), I or P (manual FPS input).  
- **File ops:** F (load cuts TOML or sync TXT), ESC (save TOML, optionally render cuts; batch prompt).  
- **Help:** H or click "Help" button (scrollable overlay with mouse wheel or arrow keys).  

## 📁 Formats
- **Input videos:** `.mp4`, `.avi`, `.mov`, `.mkv` (anything ffmpeg/OpenCV can read).  
- **Cuts file:** `{basename}_cuts.toml`  
  - Metadata: `video_name`, `fps` (.6f), `created`, `source_file` (POSIX), `output_dir` (POSIX).  
  - Cuts (1-based): `index`, `start_frame`, `end_frame`, `frame_count`, `start_time`, `end_time`, `duration` (= `frame_count/fps`), `output_file`, `output_dir` (POSIX).  
- **Sync file (TXT):** `video_file new_name initial_frame final_frame` (integers).  
- **Outputs:**  
  - Folder (single): `{video}_vailacut_{timestamp}` (or `{video}_sync_vailacut_{timestamp}` when using sync).  
  - Folder (batch): `vailacut_{prefix}{source}_batch_{timestamp}`.  
  - Files: `{video}_frame_{start}_to_{end}.mp4` (1-based in names).  

## 🔬 Precision Notes
- FPS stored/written with `.6f`; durations use frame counts to include the last frame.  
- All time displays use `.6f` precision (6 decimal places) for scientific accuracy in biomechanics.  
- UI displays frames as 1-based; internal math is 0-based.  
- PageUp/PageDown iterate over every start and end marker (sorted, deduped).  
- Audio waveform displays synchronized audio data with orange line visualization; playback matches video frame position.  

## 🐛 Troubleshooting Quick Hits
- **No ffmpeg:** Falls back to OpenCV; install ffmpeg for frame-accurate cuts and audio extraction.  
- **Windows paths in TOML:** Paths are stored with `/` to avoid escape issues.  
- **Cannot load cuts:** Ensure the file is `.toml` with `[[cuts]]` entries; legacy `.txt` still loads but is deprecated.  
- **No audio playback:** Check if video file contains audio track; audio panel (A key) shows "No Audio Data" if unavailable.  
- **Audio not syncing:** Ensure audio is loaded (A key) and not muted (M key); audio syncs on play/pause and frame navigation.  

---
📅 **Last Updated:** January 2026 (v0.2.0 - Audio waveform, loop control, auto-fit)  
🔗 **Part of vailá - Multimodal Toolbox**  
