# multicam_recorder

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/multicam_recorder.py`
- **Version:** 0.4.5
- **GUI Interface:** ✅ Yes
- **CLI Interface:** ❌ No (GUI only; prints an equivalent CLI command per camera)

## 📖 Description

Records video from **N cameras simultaneously** through a simple Tkinter GUI,
replacing hand-typed per-camera `ffmpeg` commands launched in parallel from
the shell. Each camera is captured with its own `ffmpeg` subprocess using
**stream copy** (`-c:v copy`) — no re-encoding, so quality and CPU usage match
manual capture exactly.

### Key Features

- **Cross-platform camera detection**: `/dev/video*` on Linux, `ffmpeg -f avfoundation -list_devices` on macOS, `ffmpeg -f dshow -list_devices` on Windows
- **Per-camera resolution/fps**: each detected camera gets its own editable resolution and FPS pickers (best-effort probed via OpenCV)
- **Single Start/Stop toggle**: one button starts/stops every selected camera together
- **Graceful stop**: sends ffmpeg's interactive `q` command on each process's stdin so every `.mp4` is finalized correctly, instead of killing processes
- **Fault isolation**: a disconnected/hung camera never blocks stopping the others (per-process timeout with `terminate`/`kill` escalation as a last resort)
- **Session output layout**: `<output_dir>/<session_name>_<timestamp>/cam<N>_<session_name>.mp4` (`.mp4` so recordings are directly readable by vailá's Markerless 2D tools, no renaming needed)
- **Live preview**: every detected camera previews automatically in one small mosaic window, kept live (via a low-res JPEG ffmpeg also writes) even while recording, so you can catch a bumped/out-of-focus camera mid-session
- **GUI→CLI mirror**: prints the exact equivalent `ffmpeg` command for each camera on Start, so it can be copy-pasted and reproduced from a terminal

## 🚀 Usage

### GUI Mode (from vailá)

Select **Record Cameras** in the vailá toolbox (Video and Image tools).

1. Click **Rescan cameras** if a camera was connected after opening the window.
2. Check the cameras to record, adjusting resolution/fps per camera if needed.
3. Choose an output directory and enter a session/trial name.
4. Click **Start Recording**; click the same button (now **Stop Recording**) to finish.

### Standalone

```bash
uv run python vaila/multicam_recorder.py
```

## 📋 Requirements

- **FFmpeg** installed and resolvable via `vaila.ffmpeg_utils.get_ffmpeg_path`
- **OpenCV** (only used to probe supported resolution/fps per camera; capture itself uses ffmpeg)
- Python 3.12 with Tkinter

## ⚠️ Known limitations (v1)

- Windows DirectShow duplicate device names (two identical webcam models) are
  disambiguated via the device's `@device_pnp_...` alternative name, not by a
  friendlier label.
- No hardware frame-level synchronization between cameras — relies on
  near-simultaneous process launch, same as manual `&`/`wait` shell capture.

---

📅 **Updated:** 26/09/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
