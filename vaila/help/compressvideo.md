# compressvideo (⚠️ Deprecated)

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila/compressvideo.py`
- **GUI Interface:** ✅ Yes
- **CLI Interface:** ❌ No
- **Status:** ⚠️ **DEPRECATED**

## ⚠️ Deprecation Notice

This module is **deprecated**. Please use the newer, more capable scripts instead:

- **`compress_videos_h264.py`** — H.264 compression with GPU support, CLI, and resolution control
- **`compress_videos_h265.py`** — H.265 compression with GPU support, CLI, and resolution control

## 📖 Description

Legacy script that compresses videos to either H.264 or H.265 format using FFmpeg.
Provides a basic GUI for codec selection but lacks GPU acceleration, resolution
control, preset selection, and CLI support.

## 🔧 Main Functions

- `check_ffmpeg_encoder` — Test if an FFmpeg encoder is available
- `run_compress_videos` — Basic compression logic
- `ask_codec_selection` — GUI codec picker
- `compress_videos_gui` — GUI entry point (shows deprecation warning)

---

📅 **Updated:** 18/02/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
