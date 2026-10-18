# syncvid — Pygame Multi-Video Synchronization

## Module information

- **Category:** Tools → Video and Image
- **File:** `vaila/syncvid.py`
- **Version:** 0.4.10
- **Updated:** 2026-10-18
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
| Left / Right | Previous/next frame while paused; hold to keep stepping (as in getpixelvideo) |
| Down or `-` | Move 60 frames backward (repeats while held) |
| Up or `+` | Move 60 frames forward (repeats while held) |
| `[` / `]` | Slower/faster playback: 0.0625× through 16× |
| Home / End | First/last frame of the active video |
| Page Up / Page Down | Previous/next camera |
| Timeline click/drag | Seek directly |
| `D` or Delete | Mark the current frame as repeated (red border, red timeline tick); again to keep it |
| Mouse wheel over the video | Zoom in/out at the cursor (1× to 10×) |
| Middle-button drag, or left drag when zoomed | Pan the zoomed video |
| `0` | Reset zoom |
| Mouse click | Select camera, field, or action |
| Tab / Shift+Tab | Next/previous input field |
| Enter | Validate the active field and seek to a typed sync frame |
| Ctrl+S | Save the synchronization TXT |
| Escape | Leave a field; when no field is active, cancel |

The visible step buttons use the traditional ASCII labels **-60**, **-1**, **+1**, and **+60**.

Playback reads consecutive frames sequentially and redraws cached paused frames, avoiding repeated random H.264 seeks. This is substantially smoother on long 120-fps recordings.

## Remove repeated frames

Videos edited for YouTube or a broadcast often splice several cameras into one timeline. When a camera
ran slower than the timeline, the editor repeats some of its frames, so the file claims 59.94 fps but
holds fewer real images, and the real frame rate is hidden.

Two side-panel buttons and one key handle them:

| control | action |
|---|---|
| **Detect repeats** | analyzes every video and marks its repeated frames |
| `D` or Delete | marks the current frame as repeated, or keeps a marked one (safety for anything detection missed or got wrong) |
| **Save w/o repeats** | writes repeat-free copies of every video with marks (below) |
| **Save sync** / **Save + Cut Video** | also stores the marks in the sync file (see Synchronization file v2) |

A marked frame shows a red border and the label *REPEATED*; the timeline shows a red tick per mark,
the camera row shows `(-N)` marks, and the frame line shows the recounted number, for example
`Frame 45/149  (new 30/93)`. Marks are saved in the Load / Resume checkpoint.

How **Detect repeats** judges each frame, as a viewer would:

- The frame and the previous one are blurred slightly. The **motion score** is the largest mean
  grey-level change in a 32 x 18 grid of blocks, that is, the place where the image changed most.
- A copy re-encoded by the editor (H.264/H.265) is never pixel-identical: edges and the image top
  and bottom flicker a little. That codec noise scores low everywhere. A real new frame scores high
  where something moved.
- The cut is chosen **per video**. Sorted scores show a gap of at least 2x between the noise group
  and the motion group. When several such gaps exist, the one that best separates the two groups is
  used. Without such a gap the video has no repeats (only changes scoring at most 1 grey level count).
- On the 1080p examples: `cobrador.mp4` copies scored at most 16.5 and real frames at least 40.2;
  `lateral.mp4` 8.4 against 30.6. `cabine1.mp4` and `gol.mp4` showed a continuous range (largest gap
  1.09x) and no repeats.

**Save w/o repeats** (or the CLI `--remove-repeated`, which uses the automatic marks) writes a copy of
each video with marked frames in `processed_norepeat_<timestamp>/`, next to the videos:

| file | content |
|---|---|
| `<name>_norepeat_<fps>fps.mp4` | H.264, only the distinct frames. The container frame rate is the estimated real rate, so the duration is kept; the rate is also in the file name and in the `comment` metadata. No audio |
| `<name>_norepeat_frames.csv` | `new_frame`, `original_frame` (both 1-based), `original_time_s` |
| `<name>_motion_scores.csv` | every video: `frame`, `motion_score`, `repeated` (1 = removed), to check the decision |
| `norepeat_summary.csv` | every video: frames, unique, repeated (marked), fps, estimated real fps, threshold, gap ratio, automatically detected repeats, repeated frame numbers |

Estimated real fps = fps x unique frames / total frames. Example: `lateral.mp4`, 141 frames at
59.94 fps, keeps 54 unique frames, about 22.96 fps (`lateral_norepeat_22.96fps.mp4`). The real frames
are not evenly spaced in the source (here every 2nd to 4th frame), so the new video spaces them evenly
at the average rate; `<name>_norepeat_frames.csv` keeps the original times. Videos without repeats are
listed in the summary but not rewritten.

A shot where nothing moves at all scores like codec noise. Check `<name>_motion_scores.csv` before
using a copy of such a video.

```bash
uv run vaila/syncvid.py -i /path/to/camera_directory --remove-repeated
uv run vaila/syncvid.py -i /path/to/one_video.mp4 --remove-repeated
```

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

When frames are marked as repeated, a sixth column `removed_frames` lists the marked source frames
inside each range (1-based, space-separated). The synchronization itself is unchanged: ranges stay
frame-aligned on the original timeline. Cut Video renders every synchronized cut as before and,
for cameras with marks, also writes `<cut>_norepeat_<fps>fps.mp4` without them (real frame rate,
same duration) and `<cut>_norepeat_frames.csv` mapping each new frame to its source frame.

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

# Write repeat-free copies of videos with editor-repeated frames
uv run vaila/syncvid.py -i /path/to/camera_directory --remove-repeated
```

## Validation and security

- Every typed frame must be an integer inside its video's timeline.
- Reference text must resolve to exactly one camera; ambiguous fragments are rejected.
- The common interval is checked before writing, so every output has equal duration.
- Empty/broken videos, invalid FPS/frame counts, duplicate filenames, and fewer than two cameras are rejected.
- Absolute paths, traversal, external symlinks, control characters, duplicate TXT entries, oversized files, and invalid ranges are rejected.
- The writer is atomic and the Cut Video handoff uses an argument list with `shell=False`.

---

Part of *vailá* — Multimodal Toolbox. Updated 18 October 2026.
