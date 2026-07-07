# cutvideo.py

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/cutvideo.py`
- **Version:** 0.3.76
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
- **Loop control**: Loop button (stacked above Help / Base name on the right) to enable/disable video and audio looping.
- **Auto-fit window**: Hotkey `0` automatically adjusts window to maximize screen space usage while maintaining aspect ratio.
- **Marker navigation**: Shift+Left/Right cycles through every start/end marker with wraparound, matching `getpixelvideo`; PageUp/PageDown and Home/End remain available.
- **Cut timeline feedback**: Clickable/draggable strip above the scrub slider mirrors `getpixelvideo`: blue ranges, green starts, orange ends, yellow pending start, white playhead. Clicking a marker column snaps to that cut point.
- **Responsive final render**: ffmpeg/OpenCV exports show a cancellable Tk progress dialog. The dialog stays responsive while ffmpeg runs and terminates the active subprocess when cancelled.
- **Manual FPS input**: Hotkey `I`/`P` lets you override FPS via Tk dialog; UI updates instantly.
- **Scrollable help**: Help overlay scrolls via mouse wheel or arrow keys.
- **Output naming**: Single video → `{video}_vailacut_{timestamp}`; batch → `vailacut_{prefix}{source}_batch_{timestamp}`; sync batch → `vailacut_sync_{video}_{timestamp}`; cut files use 1-based frame ranges.
- **Custom output base name**: Set a prefix for cut files (e.g. `trial01` → `trial01_frame_10_to_20.mp4`) via **B** or the **Base name** button; empty input resets to the video stem. Output folder name still uses the source video stem.
- **Per-cut output names (CSV/TXT):** **V** or **N** (or **Cut names** button) — one name per line → `name.mp4` per cut. Does **not** load cuts or replace TOML (**F**); only sets exported video filenames. Example for 3 cuts:

```text
CarlosMiguel_cod_02
Coutinho_cod_02
Heittor_cod_02
```

  **C** loads labels for TOML metadata only (does not rename files). Per-cut names take priority over **B** (single base name).
- **Runs in its own process**: launched from the main GUI as a subprocess (like `getpixelvideo`), so closing the cut window always returns control to vailá without freezing the main window.
- **Playback speed (v0.3.76):** `[` / `]` step through discrete speeds (0.0625×–16×, always includes **1×**); HUD overlay upper-right on the video (same style as `getpixelvideo`).
- **Time display**: Caption shows filename + FPS + active base prefix (or `Names:CSV(n)` when a per-cut list is loaded); slider shows frame and time with `.6f` precision (`Frame: X/Y (t.TTTTTT/T.TTTTTT)`).

## 🔧 Main Functions (high level)
- `play_video_with_cuts()` — main UI/player and cut workflow.
- `timeline_x_for_frame()` / `frame_index_from_cut_timeline_x()` — timeline drawing and marker-aware click mapping.
- `adjacent_cut_marker_frame()` — Shift+Left/Right previous/next cut-marker lookup with wraparound.
- `RenderProgressDialog` — responsive, cancellable export feedback while ffmpeg runs.
- `sanitize_output_basename()` — filesystem-safe custom prefix.
- `effective_cut_basename()` — resolves custom prefix or video stem.
- `cut_output_filename()` — builds `{base}_frame_{start}_to_{end}.mp4` (1-based).
- `prompt_output_basename_dialog()` — Tk dialog for single base name (**B** / button).
- `prompt_output_basenames_csv()` — Tk file dialog to load a per-cut name list (**V** / **N** / **Cut names** button).
- `parse_basename_list()` — parse a CSV/TXT list of base names (comma or newline; `name,label` OK).
- `build_cut_output_filenames()` — resolve one collision-safe output filename per cut (per-cut CSV > single base > video stem).
- `save_cuts_to_toml()` — writes TOML (1-based frames, .6f times, frame-count duration).
- `load_cuts_from_toml()` / `load_cuts_from_toml_file()` — reads TOML; legacy `.txt` as fallback.
- `cut_video_with_ffmpeg()` (precise) and `cut_video_with_opencv()` (fallback).
- `batch_process_videos()` and `batch_process_sync_videos()` — reuse marked cuts across files.
- `get_precise_video_metadata()` — ffprobe-first, OpenCV fallback.
- `run_cutvideo()` — entry point (uses Tk file dialog then launches pygame UI).

## 🎮 Controls (UI)

### Bottom control bar
- **Left:** cut timeline strip above the frame/time slider, cut count, hint `Out: {prefix}_frame_…` (or `Out: per-cut CSV names (n)`) for the active output naming. Strip colors: blue range, green start, orange end, yellow pending start, white playhead.
- **Right (top to bottom):** **Loop**, **Help**, then a row with **Cut names** (**V** / **N**; shows **Names ✓n** when loaded) and **Base name** (**B**).

### Keyboard and mouse
- **Playback/Navigation:** Space (play/pause), `[` / `]` (slower/faster playback: halve/double, 0.0625×–16×), →/← (frame step), Shift+→/← (next/previous cut marker with wraparound), ↑/↓ (±60 frames), mouse on slider (jump), click/drag cut strip (jump with marker snap).
- **Audio Controls:** A (toggle audio waveform panel), M (mute/unmute audio).
- **Window Controls:** 0 (auto-fit window to screen), +/- or wheel (zoom), middle-mouse drag (pan when zoomed), drag window edges (manual resize).
- **Markers/Cuts:** S (start), E (end), R (reset start), D or Delete (remove last cut), L (list cuts with planned output filenames), C (labels only).
- **Navigation to markers:** Shift+Left/Right (cycle previous/next marker), PageUp/PageDown (prev/next marker), Home/End (start/end of current or first/last cut).
- **Jump inputs:** G (go to frame), T (go to time in seconds), I or P (manual FPS input).
- **File ops:** F (load cuts TOML or sync TXT), **V** or **N** (per-cut output names CSV/TXT), B (single base name), C (labels only), ESC (save TOML, optionally render cuts; batch prompt).
- **Help:** H or click **Help** button (scrollable overlay).

## 📁 Formats
- **Input videos:** `.mp4`, `.avi`, `.mov`, `.mkv` (anything ffmpeg/OpenCV can read).
- **Cuts file:** `{basename}_cuts.toml`
  - Metadata: `video_name`, `fps` (.6f), `created`, `source_file` (POSIX), `output_dir` (POSIX).
  - Cuts (1-based): `index`, `start_frame`, `end_frame`, `frame_count`, `start_time`, `end_time`, `duration` (= `frame_count/fps`), `output_file`, `output_dir` (POSIX).
- **Sync file (TXT):** `video_file new_name initial_frame final_frame` (integers).
- **Outputs:**
  - Folder (single): `{video}_vailacut_{timestamp}` (or `{video}_sync_vailacut_{timestamp}` when using sync).
  - Folder (batch): `vailacut_{prefix}{source}_batch_{timestamp}`.
  - Files: `{base}_frame_{start}_to_{end}.mp4` (1-based in names; `{base}` = video stem or custom prefix from **B**).
  - Files (per-cut CSV via **N**): `{name}.mp4`, one name per cut in order, de-duplicated as needed.
- **Base-name list (CSV/TXT for N):** single comma-separated line (`jump1, jump2, jump3`) or one name per line; optional second column treated as a label (`run_a,sprint`).

## 🔬 Precision Notes
- FPS stored/written with `.6f`; durations use frame counts to include the last frame.
- All time displays use `.6f` precision (6 decimal places) for scientific accuracy in biomechanics.
- UI displays frames as 1-based; internal math is 0-based.
- Shift+Left/Right iterate over sorted, deduplicated start/end markers and wrap at timeline edges; PageUp/PageDown remain supported.
- Audio waveform displays synchronized audio data with orange line visualization; playback matches video frame position.

## 🐛 Troubleshooting Quick Hits
- **Player does not open after metadata loads:** Update to v0.3.47+ (fixes `output_basename` initialization before the pygame UI).
- **No ffmpeg:** Falls back to OpenCV; install ffmpeg for frame-accurate cuts and audio extraction.
- **Windows paths in TOML:** Paths are stored with `/` to avoid escape issues.
- **Cannot load cuts:** Ensure the file is `.toml` with `[[cuts]]` entries; legacy `.txt` still loads but is deprecated.
- **No audio playback:** Check if video file contains audio track; audio panel (A key) shows "No Audio Data" if unavailable.
- **Audio not syncing:** Ensure audio is loaded (A key) and not muted (M key); audio syncs on play/pause and frame navigation.

- **Main vailá window froze after cutting / had to `kill`:** Fixed — the cut tool runs in its own subprocess and final ffmpeg/OpenCV export now has a responsive cancellable progress dialog.

---
📅 **Last Updated:** 02 June 2026 (v0.3.47 - Shift+Left/Right cut-marker navigation; timeline feedback; responsive cancellable final render)
🔗 **Part of vailá - Multimodal Toolbox**
