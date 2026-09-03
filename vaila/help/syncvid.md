# syncvid — Pygame Multi-Video Synchronization

## Module information

- **Category:** Tools → Video and Image
- **File:** `vaila/syncvid.py`
- **Version:** 0.3.120
- **Updated:** 2026-09-03
- **GUI:** Frame C → **Make Sync file**
- **CLI:** `uv run vaila/syncvid.py`

## Purpose

`syncvid` opens every supported video from one target directory in a fast Pygame player. Playback and frame navigation follow the same conventions as `cutvideo.py`. The synchronization values are typed directly in the side panel, avoiding separate Mark, Reference, Start, and End buttons.

The generated versioned TXT maps one inclusive reference-camera interval into equal-duration ranges for every camera. **Save + Cut Video** launches Cut Video with both the reference video and TXT already selected.

## Fast workflow

1. Select a directory containing at least two videos.
2. Click a camera name in the right panel, or use **Page Up/Page Down**.
3. Use playback, the timeline, or frame stepping to locate the common event.
4. Type its displayed 1-based frame in that camera's **Sync frame** field.
5. Repeat for every camera.
6. In **Reference video**, type any one of:
   - the 1-based row number;
   - the exact filename;
   - the filename without extension;
   - a unique text fragment such as `frontal`.
7. Type the inclusive **Reference start** and **Reference end** frame numbers.
8. Choose:
   - **Save sync** — save and return to vailá;
   - **Save + Cut Video** — save and open Cut Video immediately without another file chooser.

Clicking a field selects its current contents for quick replacement. **Tab/Shift+Tab** moves through all fields and **Enter** validates the active field. All displayed and saved frames are **1-based and inclusive**; internal Python calculations remain 0-based.

## Load / Resume

Every **Save sync** or **Save + Cut Video** also writes a `.json` checkpoint next to the sync TXT (same name, `.json` extension) capturing every field's text, the current camera, the current playback frame, and the output path.

Click **Load / Resume** at any time to reopen a checkpoint through a file picker and restore that state:

- All **Sync frame** fields, **Reference video**, **Reference start**, and **Reference end** are refilled.
- The active camera and its exact playback frame are restored.
- Works for a checkpoint saved from **Save sync** (an in-progress session) or **Save + Cut Video** (a completed one) — the status bar names which.
- A camera the checkpoint mentions that is not present in the currently opened directory is skipped, not fatal; the status bar lists any skipped camera name.
- A missing file, corrupt JSON, or a checkpoint missing required keys shows a clear status-bar error instead of crashing the player.

## Help viewer (Linux/WSL)

The **Help** button opens this page in a browser. It tries, in order, the `BROWSER` environment variable, native Linux GUI browsers (`xdg-open`, `google-chrome`, `firefox`, `chromium`) found on `PATH`, then `wslview` only when WSL interop is actually available, then Python's `webbrowser` module. If every attempt fails, the page's `file://` path is printed to the console instead of crashing — this avoids the `wslview`/`reg.exe` failure (`grep: /proc/sys/fs/binfmt_misc/WSLInterop: No such file or directory`) seen in WSL setups without working interop.

## Pygame controls

| Control | Action |
|---|---|
| Space | Play/pause |
| Left / Right | Previous/next frame while paused |
| Down or `-` | Move 60 frames backward |
| Up or `+` | Move 60 frames forward |
| `[` / `]` | Slower/faster playback: 0.0625× through 16× |
| Home / End | First/last frame of the active video |
| Page Up / Page Down | Previous/next camera |
| Timeline click/drag | Seek directly |
| Mouse click | Select camera, field, or action |
| Tab / Shift+Tab | Next/previous input field |
| Enter | Validate the active field and seek to a typed sync frame |
| Ctrl+S | Save the synchronization TXT |
| Escape | Leave a field; when no field is active, cancel |

The visible step buttons use the traditional ASCII labels **-60**, **-1**, **+1**, and **+60**.

Playback reads consecutive frames sequentially and redraws cached paused frames, avoiding repeated random H.264 seeks. This is substantially smoother on long 120-fps recordings.

## Synchronization file v2

The output remains a UTF-8 tab-separated TXT:

```text
# vaila sync file v2
# frame_base=1; frame ranges are inclusive
video_file	output_file	start_frame	end_frame	sync_frame
camera 01.mp4	camera_01_sync_1001_frames_901_to_1101.mp4	901	1101	1001
```

Tab-separated CSV quoting supports filenames containing spaces. Cut Video also reads legacy whitespace rows:

```text
camera01.mp4 camera01_sync.mp4 901 1101
```

Saving atomically replaces an existing sync file instead of appending duplicate or stale rows.

## Direct Cut Video handoff

```bash
uv run vaila/cutvideo.py \
  --video /path/to/reference.mp4 \
  --sync-file /path/to/vaila_sync_YYYYMMDD_HHMMSS.txt
```

Cut Video validates the TXT and every source video before its Pygame window opens. The interval is already loaded with `(SYNC)` status, and saving can render all synchronized cameras.

## CLI and validation

```bash
# Pygame player with a preselected target directory
uv run vaila/syncvid.py -i /path/to/camera_directory

# Preselect the TXT destination
uv run vaila/syncvid.py -i /path/to/camera_directory -o /path/to/session_sync.txt

# Probe/decode all videos without opening Pygame
uv run vaila/syncvid.py -i /path/to/camera_directory --dry-run
```

## Validation and security

- Every typed frame must be an integer inside its video's timeline.
- Reference text must resolve to exactly one camera; ambiguous fragments are rejected.
- The common interval is checked before writing, so every output has equal duration.
- Empty/broken videos, invalid FPS/frame counts, duplicate filenames, and fewer than two cameras are rejected.
- Absolute paths, traversal, external symlinks, control characters, duplicate TXT entries, oversized files, and invalid ranges are rejected.
- The writer is atomic and the Cut Video handoff uses an argument list with `shell=False`.

---

Part of **vailá — Multimodal Toolbox**. Updated 2 September 2026.
