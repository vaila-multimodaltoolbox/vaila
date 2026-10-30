# syncvid — Pygame Multi-Video Synchronization

## Module information

- **Category:** Tools → Video and Image
- **File:** `vaila/syncvid.py`
- **Version:** 0.4.12
- **Updated:** 2026-10-29
- **GUI:** Frame C → **Make Sync file**
- **CLI:** `uv run vaila/syncvid.py`

## Purpose

`syncvid` opens every supported video from one target directory in a fast Pygame player. Playback and frame navigation follow the same conventions as `cutvideo.py`. The synchronization values are typed directly in the side panel, avoiding separate Mark, Reference, Start, and End buttons.

The generated versioned TXT maps one inclusive reference-camera interval into equal-duration ranges for every camera. **Save + Cut Video** launches Cut Video with both the reference video and TXT already selected.

## Equalize by events (RIFE)

Open **Frame C → Make Sync file**, select the video folder, then click
**Equalize by events (RIFE)** at the bottom of the synchronization panel. A separate
Tkinter window opens; Syncvid remains available. This action uses the reference
camera's frame count and exact rational FPS by default. **Output FPS** and
**Output frame count** can define a different common grid. The existing **Events: interpolate**
mode still uses Farnebäck and its grid that preserves every original.

1. Choose **Reference video** (the desired output timeline) and **Source video**.
2. Enter **Reference: start, contact, end** and **Source: start, contact, end**.
   Values are original, 1-based, inclusive frames. Start/end initialize from the
   video counts when browsing; contact must be entered or reviewed. Syncvid's
   launcher prefills the current sync-frame values. Use Browse after changing videos
   to refresh the bounds. Removal marks and FPS overrides from Syncvid do not apply.
3. Optionally set **Output FPS** (decimal/fraction; **60** and **120** shortcuts)
   and/or **Output frame count** (integer ≥3). Blank fields retain the reference grid.
   FPS alone derives `N = round_half_up(reference anchor span × FPS) + 1`;
   count alone uses reference FPS; both are honored independently. `29.970` is exactly
   `2997/100`, distinct from NTSC `30000/1001`. FPS must be positive and finite;
   FPS-only requests calculating fewer than 3 frames are rejected.
4. Set **Practical-RIFE checkout**, **RIFE model directory** and **Output parent directory**.
5. Click **Preview**. Review exact FPS/count, original/generated counts in both cameras,
   first-to-last duration `(N−1)/FPS`, file duration `N/FPS`, change relative to the
   reference anchor span, and contact timing adjustments. Click **Run** after review;
   editing settings requires another preview. Progress appears in the window and terminal, including the equivalent
   CLI. **Cancel export** stops the worker; partial directories remain marked INCOMPLETE.

The output anchors are start=1 and end=N. Contact follows the reference interval
proportion, rounded to nearest with ties up, then clamped to 2…N−1. Each camera's
source positions are linear within its two segments. All three anchors are copied
exactly; fractional positions in either camera, including the reference, use RIFE.
See [full output controls and audit fields](video_event_equalizer.md#output-fps-and-frame-count).

With FPS/count blank, for the Roseguine example, choose reference `c2_16mm_left.mp4`, events `1,9,22`,
and source `c1_main.mp4`, events `1,9,19`. Both outputs have 22 frames at **2997/100 FPS**
and retain their original resolution. C1 frames 1–9 and its final frame 19 are copied;
the other 12 output images are generated. For output frame `j > 9`, the source
position is `9 + (j − 9) × 10/13`. This re-samples the segment rather than merely
inserting three pictures. Not every post-contact original appears in the output.
With **Output FPS=60** and count blank, both have 43 frames, contact=17, span=0.7 s,
and file duration=43/60 s. With **120** they have 85 frames and contact=33.
Setting both FPS=120 and count=31 gives a 0.25 s span and changes playback speed.

Each new `processed_event_equalize_TIMESTAMP/` contains:

- `reference_<stem>_equalized.mp4` and `source_<stem>_equalized.mp4`: lossless RGB H.264,
  exact selected FPS, **no audio**. Originals are never overwritten.
- `reference_frames.csv` and `source_frames.csv`: output/reference index, rational
  output time, fractional source position, left/right source frames, interpolation
  fraction, and `original`/`generated` label. Added columns record rational reference
  position, anchor labels, requested/effective FPS/count, output anchors and timing adjustments.
  `reference_frame` can be rational on a custom grid; default integer values are preserved.
- `event_equalizer.json`: source paths/counts/FPS/events, output policy, RIFE revision,
  code hashes, weight SHA256 and GPU/runtime information, requested/effective output
  parameters, rational durations, contact adjustments and per-camera original/generated
  counts; `README.txt` states assumptions and the preview.
- `INCOMPLETE.txt` exists during processing and remains after failure/cancellation.

**Interpretation:** event equivalence is supplied by the operator. Uniform spacing
between each pair of events and the reference clock are assumptions. Generated ball,
feet and occlusion pixels can contain artifacts and are not acquired measurements.
Increasing FPS generates visual estimates; freely choosing FPS and count changes
playback speed. Contact rounding can shift event timing; the preview records that shift.
For quantitative 3D work, retain measured source coordinates and their provenance.

### RIFE setup

Requires NVIDIA CUDA, CUDA-enabled PyTorch, FFmpeg and the official
[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) checkout plus its matching
**4.25** model bundle. The existing CUDA environment supplies cuDNN; do not replace
its torch packages by installing upstream's full requirements file.

**Automatic installation:** `install_vaila_linux.sh` and `install_vaila_win.ps1`
run RIFE setup when the CUDA/GPU profile is selected. The unified bootstraps
`bin/setup_pyproject.sh --target=cuda` and `bin/setup_pyproject.ps1 -Target cuda`
also run it after dependency sync. CPU/macOS profiles do not install RIFE.
`--no-sync` / `-NoSync` skip this step too. Use `--no-rife` / `-NoRife` to skip
only RIFE in the unified bootstrap. Failed optional setup prints a retry command
and leaves the rest of vailá available.

**Install or reinstall only RIFE**, from the vailá repository root after the CUDA
environment has been installed:

```bash
# Linux / WSL / Git Bash
bash bin/setup_rife.sh
```

```powershell
# Windows PowerShell
pwsh bin/setup_rife.ps1
```

Both call `bin/setup_rife.py` using the existing vailá Python environment with
`uv run --no-sync`. They clone the validated revision, download the official
4.25 ZIP (~23 MB), verify the archive and model SHA256 hashes, install the model,
and run actual CUDA inference. Repeating the command reuses verified assets.
No Hugging Face login, separate cuDNN installation, or new Python extra is needed.
The NVIDIA driver must support the CUDA PyTorch build selected by vailá.

Installed assets, relative to the repository:

| Location | Contents |
| --- | --- |
| `.local/third_party/Practical-RIFE/` | Official code, pinned Git revision |
| `vaila/models/rife/train_log/` | `flownet.pkl` and its matching Python model files |
| `.local/downloads/rife425.zip` | Verified download cache, created on automatic download |
| `vaila/models/rife/setup_manifest.json` | Model identity and CUDA inference verification |

These assets are ignored by Git: **each computer needs its own setup**. To move
them offline, preserve both the complete pinned checkout (including `.git`) and
the model directory at these same relative locations, then re-run setup to test
the destination GPU. A previously downloaded official ZIP can also be supplied:

```bash
uv run --no-sync python bin/setup_rife.py --archive /path/to/rife425.zip
```

The ZIP avoids downloading weights; cloning still needs network if the checkout
is absent. `--skip-check` stages assets without CUDA testing and explicitly does
not report GPU readiness. Existing modified/different checkouts are preserved
and require review before switching to the validated revision. Linux CUDA setup
was tested from an empty directory and on reinstall; Windows integration has
not yet been executed on a Windows host.

**Manual recovery** if the download service is unavailable:

```bash
git clone https://github.com/hzwer/Practical-RIFE.git .local/third_party/Practical-RIFE
git -C .local/third_party/Practical-RIFE checkout bbfd2ea90910789a860ea3e2b32a240cd577b75e
```

Download the official **4.25** bundle linked in that project's README:
[Google Drive](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view).
Extract it into `vaila/models/rife/`, retaining the `train_log/` directory.
The model directory must contain `RIFE_HDv3.py`, `IFNet_HDv3.py`, and `flownet.pkl`.
Keep the accompanying Python files together. All model-bundle files are ignored by Git.
Expected `flownet.pkl` SHA256 for the validated bundle:
`6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2`.
The loader checks inference weights strictly, excluding only training-only
`teacher.*` and `caltime.*` tensors, and performs an actual-resolution preflight.
The upstream README lists Python ≤3.11; this adapter was functionally validated
with vailá's Python 3.12, RIFE 4.25 and an RTX 4090.

```bash
uv run --no-sync python -m vaila.video_event_equalizer \
  --reference /path/to/c2_16mm_left.mp4 --source /path/to/c1_main.mp4 \
  --reference-events 1,9,22 --source-events 1,9,19 \
  --output-dir /path/to/results
```

Use `--rife-dir` and `--weights` to select other local locations. Add `--dry-run`
to probe/decode-count videos and print exact frame maps without loading RIFE or
writing output. Run with `--gui` to open the same window directly.
Use `--output-fps 60` / `--output-fps 120` / `--output-fps 30000/1001` and optionally
`--output-frame-count 85` for the same output controls as the GUI.

## Manual cuts side by side

Open **Frame C → Make Sync file**, select the source folder, then click **Manual cuts side by side (M)**
above the event-mode button. The keyboard shortcut **M** enters this mode from any synchronization mode;
**Back to synchronization (M)** returns with the independent manual edits still available.
You can also launch it directly:

```bash
uv run --no-sync python -m vaila.syncvid --manual-cuts --input-dir /path/to/videos
```

Two video previews appear side by side. Click a pane to select the **ACTIVE** camera; playback,
frame stepping, Go to frame, zoom and **D/Delete** apply to that camera only. The other camera stays
at its own frame. **PgUp/PgDn**, the camera rows and **Prev/Next video** switch cameras; folders
with more than two cameras are shown in pairs. A one-video folder works too.

1. Navigate to the first frame to keep and press **I** or **Set Start (I)**.
2. Navigate to the last frame to keep and press **O** or **Set End (O)**.
3. Alternatively, type each camera's own **Start** and **End** in its row. Values are original,
   **1-based and inclusive**, and are never changed to match another camera.
4. Press **D/Delete** on a frame to mark removal. A red pane indicates a removed frame;
   pressing **D** again restores it. Originals remain visible during editing so you can undo marks.
5. Watch **kept / removed** in each row and **Remaining: … | equal / diff …** above the list.
   Counts include only frames inside each interval. Marks outside an interval have no effect on that cut.
6. Click **Save all (Ctrl+S)** to export every loaded camera, or **Save current video** to export only
   the active camera. Choose an output parent; a new `manual_cuts_TIMESTAMP/` folder is created.

Manual deletion marks are separate from synchronization/repeat-detector marks. No detector runs in this
mode. **Save** directly writes the requested new videos; it does not create a synchronization plan,
launch Cut Video, interpolate images, duplicate frames or adjust the source FPS. Removing a frame
shortens playback by one source-frame period. Different source FPS remain different: equal frame
counts imply equal durations only when FPS also matches. Unequal counts are allowed if that is what
you choose to save. Invalid intervals or an interval with every frame removed block that export;
**Save current video** validates only the active camera.

Exported files:

- `<original filename>_manual.mp4`: selected originals in order, lossless RGB H.264, **no audio**,
  at the source's exact rational FPS. Existing files and export folders are preserved.
- `<original filename>_manual_frames.csv`: output/source frame indices (1-based), exact rational
  source time and output time. Frame removal compresses the output timeline.
- `manual_cuts.json`: independent edit definitions, source FPS/count, selected bounds and removal
  marks. This is an audit/replay file, not a sync file. **Load manual cuts JSON** restores these edits.
- `INCOMPLETE.txt` remains if export is cancelled or fails; the active partial MP4 is deleted.
  Completed earlier camera videos may remain in that folder. Replay into a new folder.

Export progress stays visible; **Cancel export** or **Esc** stops rendering and keeps the editor open.
**Close** exits when idle. The equivalent CLI is printed with `>>` when opening the mode and saving.
To replay cuts without any GUI (Ctrl+C cancels):

```bash
uv run --no-sync python -m vaila.syncvid --manual-cuts --input-dir /path/to/videos \
  --cuts-file /path/to/export/manual_cuts.json --output-dir /tmp/new_manual_export
```

Omit `--output-dir` to open the side-by-side editor with a saved `--cuts-file` loaded.
Source frame count and FPS must still match the saved definitions. Existing synchronization
checkpoints are edited through the synchronization modes; they are not manual-cut definitions.

## Sync by events + interpolation

Choose a mode directly: **FPS** (default), **Events: nearest** or **Events: interpolate**.
The selected button stays highlighted; clicking it again keeps that mode. **Events: nearest** selects
the nearest retained source image and creates no interpolated images.
The interpolation option keeps **every original frame** in each inclusive selected interval, even for cameras
with equal container FPS. Enter at least **Start, Sync, End**, in that order, in every camera's event row.
Additional ordered correspondences are supported. These pairs must represent the same physical instants.

**Reference clock FPS** starts from the reference file's rate and is editable independently of output FPS.
When an imported checkpoint records a different clock from the current reference file (for example 59.94
versus 29.97), saving is blocked until you review the field and click **Confirm reference clock**.
CLI users confirm by explicitly passing `--reference-fps`. Confirming a clock is a timing assumption,
not proof of the camera's acquisition rate. Sessions save the confirmed clock and source rate.

Between consecutive events, each original receives a time linearly according to its source-frame ordinal.
Fractions compute the **minimum exact output FPS** containing every sample time in every camera.
Set **Max output FPS (e.g. 480)** to authorize the required grid. The default is **240**; use **480**
when the calculated rate is higher. This is a ceiling, so selecting 480 does not force 480 FPS.
Before saving, the panel shows the exact and decimal output FPS, final count per camera and each
camera's original/interpolated counts. The same interpolation applies to every camera, including
the one with fewer frames. Above the configured limit, validation reports the required rational rate
and blocks export; there is no silent rounding or frame dropping. Raising the limit leaves the recorded
reference clock unchanged. Change that clock only when explicitly intended.
Removal marks **inside** any interval are rejected.
Clear such marks with **D/Delete**, or revise the interval; marks outside it can remain for audit.
Unordered, duplicate, conflicting, out-of-bounds or unequal-length event lists are rejected. No images
are extrapolated beyond the first and last events.

At each exact sample position the original decoded BGR pixels pass through unchanged. Intermediate
positions use OpenCV **bidirectional Farnebäck** motion, inverse warps and temporal blending of the
two neighboring originals. **Interpolated images are estimates**: ball, feet and occlusions may show
ghosting, deformation or other artifacts. They do not recover missing physical measurements.

For `c1: 63, 71, 81` and `c2: 54, 58, 62`, a `30000/1001` reference clock yields **37 frames per camera
at `60000/1001` FPS**. Events occupy output frames **1, 17, 37** (1-based). The sample span is 0.6006 s;
the video duration includes the last frame's display period and is about 0.617283 s.

For `c1: 1, 9, 19` and `c2: 1, 9, 22`, keeping the saved `979001/32666` reference clock and
setting **Max output FPS** to **480** yields **235 frames per camera at `12727013/32666` FPS**
(389.610390), with events at **1, 105, 235**. C1 keeps 19 originals and adds 216 interpolated images;
C2 keeps 22 originals and adds 213. The default 240 limit blocks this session.

```bash
uv run --no-sync python -m vaila.syncvid --events --interpolate \
  --input-dir /path/to/cameras --checkpoint /path/to/session.json \
  --reference-fps 30000/1001 --output /tmp/interpolated.events.json
uv run --no-sync python -m vaila.cutvideo --video /path/to/cameras/c1_main.mp4 \
  --sync-file /tmp/interpolated.events.json --output-dir /tmp/new_interpolated_export
```

Use a checkpoint whose Start/Sync/End or event lists contain the requested correspondences. Omit
`--output-fps`: interpolation computes it automatically. To raise the ceiling, add `--max-output-fps 480`;
omit `--reference-fps` to retain an already confirmed saved clock. The CLI uses an explicit limit first,
then the checkpoint's limit, then 240. The GUI prints the equivalent export command
with `>>`. **Save sync** saves the manifest and resume checkpoint; **Save + Cut Video** exports into
a new timestamped folder. Existing export folders are never overwritten. A progress dialog supports
**Cancel**; the CLI prints progress and accepts Ctrl+C. Interrupted output keeps `INCOMPLETE.txt`,
deletes the active partial MP4 and never writes a completed set manifest. Retry into a new folder.

Outputs use lossless **RGB H.264**, without audio. Each camera gets a video, retained PNGs/timing CSV,
and `_time_map.csv` with `output_frame` (1-based), `requested_time_s`, `requested_time_rational`,
`type` (`original`/`interpolated`), `original_frame` (empty for estimates), `source_frame_left`,
`source_frame_right`, and exact rational `interpolation_fraction`. Original rows have identical left/right
indices and fraction 0. The legacy time/error columns remain present; assigned-time error is zero in
this mode, which does **not** measure estimation quality. Manifest/checkpoint fields include
`interpolation_method: farneback_bidirectional`, `fps_policy: auto_preserve_originals` and
`max_output_fps` (rational text). The manifest also records `event_output_frames` and per-camera
`original_output_count` / `interpolated_output_count`. Older sessions without `max_output_fps` use 240;
sessions without an interpolation method retain nearest-image behavior. Encoder and MP4 time bases
preserve the calculated rational clock exactly.

Review the foot–ball contact visually, including the synthesized images around the contact event.
Matching FPS, counts and anchors alone do not validate optical-flow quality.

## Sync by events: physical-time calibration

For edited cameras with repeated images, click **Load / Resume** to load the existing checkpoint,
then select **Events: nearest**. This imports each camera's explicit **Start, Sync, End** as three
events, without shortening a range because container FPS matches. The event list in each row is
editable: enter comma- or space-separated original frame numbers in the same event order for every
camera. Additional events are supported. The reference field accepts the usual camera name or row.
In this mode the top two fields are **Reference clock FPS** (initially the reference file's rate, or the imported clock) and **Output grid
FPS** (`30000/1001`). These are exact rational rates. Container FPS is informational here.

Review repeats with **Detect repeats** and **D/Delete** before saving. The model numbers only the
retained frames between the first and last event and assigns physical time by piecewise linear
interpolation of their retained ordinals. Reference event times come from original reference frame
offsets divided by the assumed reference clock. Event frames must be strictly increasing, in bounds,
and retained; errors identify the camera and event to correct. Nothing is extrapolated beyond the
first or last event. The hypothesis is uniform physical spacing of retained samples within each
segment. The current camera's inferred segment rates are visible below the camera list;
**Validate event timing** prints every camera's rates. These rates do not prove acquisition FPS.

**Save sync** writes `<name>.events.json` plus `<name>.checkpoint.json` for Load / Resume. The
manifest records its checkpoint, imported checkpoint, events, all removal marks (including marks
outside the event interval), rational clocks, method and assumptions. **Save + Cut Video** launches
the event export directly. Edit event synchronization in Sync Video; legacy TXT files retain their
existing behavior.

Reproducible CLI (the output directory must be new):

```bash
uv run --no-sync python -m vaila.syncvid --events \
  --input-dir /path/to/cameras --checkpoint /path/to/session.json \
  --reference-fps 60000/1001 --output-fps 30000/1001 \
  --output /tmp/session.events.json
uv run --no-sync python -m vaila.cutvideo --video /path/to/cameras/c1_main.mp4 \
  --sync-file /tmp/session.events.json --output-dir /tmp/session_event_export
```

Cut Video exports the closest retained image for each output time, with exact ties choosing the
earlier image. The grid includes times `k / output_fps` up to and including the physical endpoint
when it lies on the grid: `floor(sample_span * output_fps) + 1` frames. Events off the output grid
remain exact calibration anchors, but cannot occupy an exact output timestamp.

Each camera produces:

- `<camera filename>_events.mp4`: lossless RGB H.264 at the exact output FPS, without audio.
- `<camera filename>_time_map.csv`: `output_frame`, `requested_time_s`, `original_frame`,
  `physical_time_s`, `temporal_error_s`, `reused`, and exact rational forms of all times/errors.
  Signed error is **selected physical time minus requested time**; `reused=1` means the same source
  image was already selected earlier in that output video.
- `<camera filename>_retained/`: lossless PNG sequence of every retained observation in the interval.
- `<camera filename>_retained_times.csv`: retained ordinal, original frame, physical time (decimal
  and rational) and PNG filename. These sequences have no imposed constant playback clock.
- Shared `event_sync_manifest.json` and `README.txt`: versioned calibration and export audit.

For the confirmed pairs **c1: 1,15,79 / c2: 1,20,109**, with 31 removals in c2's interval,
c1 retains 79 images and c2 retains 78. Both outputs have **40 frames**, with events at **1,8,40**.
Physical sample span is **1.301300000 s**; playback duration is **1.334666667 s**. The intermediate
event is at **0.233566667 s**. Inferred c2 rates are 55.658627 and 59.940060 fps.

**Output correspondence does not imply simultaneous acquisition.** Consult every image's temporal
error. For 3D, measure 2D coordinates on the retained observations, interpolate those coordinates
onto shared physical times, then triangulate. Coordinate interpolation creates no new observations.
This mode exports videos and temporal maps; reconstruction is outside its scope.

## Fast workflow (sync frame + FPS)

1. Select a directory containing at least two videos.
2. Click a camera name in the right panel, or use **Page Up/Page Down**.
3. Use playback, the timeline, or frame stepping to locate the common event.
4. Type its displayed 1-based frame in that camera's **Sync** field (the keyframe, e.g. ball contact).
5. Repeat for every camera. **Start/End** of a non-reference camera are optional (see
   *Cameras with different frame rates*).
6. In **Reference video**, type any one of:
   - the 1-based row number;
   - the exact filename;
   - the filename without extension;
   - a unique text fragment such as `frontal`.
7. Type the inclusive **Reference start** and **Reference end** frame numbers.
8. Choose:
   - **Save sync** — write the sync file and stay in the player (edit and re-save as needed; Cancel or the window close ends the session);
   - **Save + Cut Video** — save, close Sync Video, and open Cut Video immediately without another file chooser.

Clicking a field selects its current contents for quick replacement. **Tab/Shift+Tab** moves through all fields and **Enter** validates the active field. All displayed and saved frames are **1-based and inclusive**; internal Python calculations remain 0-based.

## Cameras with different frame rates

Each camera is aligned in **time** at its Sync (key) frame, at its **own fps**, so cameras at 59.94, 50,
37.41, 30 or 22.96 fps can be cut to the same event window even when no frame is shared:

```text
start = sync + round((reference_start - reference_sync) / reference_fps * fps)
end   = start + round((reference_end - reference_start) / reference_fps * fps)
```

Rounding the length keeps every cut within half a frame of the reference duration; with equal fps it is
the same frame offset as before.

- Every row shows the camera's fps and frame count; the video's top bar shows `fps` and resolution.
- **Start/End** of a non-reference row are optional. Empty: they follow the reference from Sync + fps and
  the computed frames appear in grey as a preview. Typed (both): that camera is cut there and its Sync
  frame becomes optional; use it when the same event start and end can be seen in that camera.
- On save, the terminal lists every camera's range, frames, fps and duration. When a typed Start/End
  gives a duration that differs from the reference by more than one frame, it also shows the fps the
  range implies, a hint that the camera's fps (or a mark) is wrong. The cut keeps the file's fps.
- Cut Video renders each range at its own fps; the sync TXT format is unchanged.

Example (`sync_videos_different_fps`, reference `cabine1.mp4` 39–111 with keyframe 50 = ball contact):

| camera | fps | keyframe | range | frames |
|---|---:|---:|---|---:|
| cabine1.mp4 (reference) | 59.940 | 50 | 39–111 | 73 |
| cobrador_norepeat_37.41fps.mp4 | 37.412 | 19 | 12–57 | 46 |
| gol.mp4 | 59.940 | 42 | 31–103 | 73 |
| lateral_norepeat_22.96fps.mp4 | 22.956 | 9 | 5–33 | 29 |
| reporter_fundo.mp4 | 30.000 | 96 | 90–126 | 37 |

All cuts last 1.20–1.22 s and show the ball contact at the same time (0.1835 s).

## Load / Resume

Every **Save sync** or **Save + Cut Video** also writes a `.json` checkpoint next to the sync TXT (same name, `.json` extension) capturing every field's text (including each camera's optional Start/End), the current camera, the current playback frame, and the output path.

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
| `G`, or **Go to frame** + Enter/**Go** | Jump to a typed 1-based frame of the current video |
| `D` or Delete | Mark the current frame as repeated (red border, red timeline tick); again to keep it |
| Mouse wheel over the video | Zoom in/out at the cursor (1× to 10×) |
| Middle-button drag, or left drag when zoomed | Pan the zoomed video |
| `0` | Reset zoom |
| Mouse click | Select camera, field, or action |
| Tab / Shift+Tab | Next/previous input field |
| Enter | Validate the active field and seek to a typed sync frame |
| Ctrl+S | Save sync (writes files; GUI stays open) |
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
  used. Without such a gap this rule marks only changes scoring at most 1 grey level.
- **Interlaced sources** (e.g. 50 fps material on a 59.94 timeline) repeat a frame built from the other
  field: the same image shifted about 1.2 px vertically. A small, purely vertical global shift
  (phase correlation on the centre of the frame) is compensated before scoring again, and the lower
  score is kept. Horizontal or larger shifts are real camera motion and are not compensated.
- A second rule detects **alternating fields** even when repeat scores overlap real-motion scores.
  A candidate requires `|dx| <= 0.1 px`, `0.75 <= |dy| <= 1.75 px`, and phase-correlation response
  `>= 0.9`. Only runs of at least **three candidates with alternating vertical signs** are marked.
  Real frames may occur between candidates, with variable intervals; there is no fixed cadence.
  A same-sign candidate ends the run and starts another. Isolated candidates and pairs do not qualify.
  The automatic decision is the union of both rules. When both apply, the audit reason is
  `motion_threshold` (counts are exclusive); otherwise it is `field_alternation` or `none`.
- Validation on the Veiga pair matched all **32 manually marked frames** in `c2_16mm.mp4`, with no
  extra or missed marks, and found zero in `c1_main.mp4`. The 112-frame clip exports 80 frames at
  about **42.814 fps**. This is evidence for this pair, not a guarantee for other camera movements;
  review automatic marks with **D** before saving.
- On the 1080p examples: `cobrador.mp4` copies scored at most 6.7 and real frames at least 40.2;
  `lateral.mp4` 5.1 against 30.6; `gol.mp4` (field-shifted copies, raw score ~20 like slow real frames)
  8.3 against 22.9, 24 repeats, about 49.88 fps, i.e. 50 fps material. `cabine1.mp4` and the four
  30 fps phone videos showed a continuous range and no repeats.

**Save w/o repeats** (or the CLI `--remove-repeated`, which uses the automatic marks) writes a copy of
each video with marked frames in `processed_norepeat_<timestamp>/`, next to the videos:

| file | content |
|---|---|
| `<name>_norepeat_<fps>fps.mp4` | H.264, only the distinct frames. The container frame rate is the estimated real rate, so the duration is kept; the rate is also in the file name and in the `comment` metadata. No audio |
| `<name>_norepeat_frames.csv` | `new_frame`, `original_frame` (both 1-based), `original_time_s` |
| `<name>_motion_scores.csv` | every analyzed video: existing `frame`, `motion_score`, `repeated` (1 = actually removed), `shift_x_px`, `shift_y_px`; appended `phase_response`, `auto_repeated`, `auto_reason` (`motion_threshold`, `field_alternation`, `none`). The first frame has no pair score/response and is never auto-marked |
| `norepeat_summary.csv` | every video: existing counts, rates, threshold, gap, automatic total, marked frame numbers and output; appended `motion_threshold_repeats`, `field_alternation_repeats`, `manual_added_repeats`, `manual_kept_frames` |

Automatic marks remain separate from the effective export selection: **D** can keep an automatic
mark or add a removal, without changing `auto_repeated` or `auto_reason`. Summary rule counts describe
the original automatic decision; review counts describe additions and kept automatic marks relative
to it. When there is no analysis (e.g. a resumed checkpoint), automatic counts are blank and current
marks count as review additions. Checkpoints and sync files preserve the effective marks.

Estimated real fps = fps x unique frames / total frames. Example: `lateral.mp4`, 141 frames at
59.94 fps, keeps 54 unique frames, about 22.96 fps (`lateral_norepeat_22.96fps.mp4`). The real frames
are not evenly spaced in the source (here every 2nd to 4th frame), so the new video spaces them evenly
at the average rate; `<name>_norepeat_frames.csv` keeps the original times. Videos without repeats are
listed in the summary but not rewritten.

### Real FPS + Fill Start/End (after norepeat)

After **Save w/o repeats**, open the `*_norepeat_*.mp4` folder (or a mix of norepeat + original
cameras). Container fps may still be wrong for some files, or you may want one shared rate:

| control | action |
|---|---|
| **Real FPS** field | type e.g. `13.91` or `29.97` |
| **FPS this** / Enter | apply that rate to the **current** camera only (`*` next to fps in the row) |
| **FPS all** | apply that rate to **every** camera |
| clear the field + Enter | drop the override on the current camera (back to container fps) |
| **Fill Start/End from FPS** | recompute every non-reference Start/End so Start/End are the **same temporal events** as the reference window (from Sync + real fps) |

Rules after Fill / Save:

- **Same temporal window** — every camera’s Start/End map to the reference event start/end in time.
- **Equal fps** (file playback rate within ~0.05 fps) → **identical frame counts**.
- **Different fps** → frame counts may differ; durations still match in time.
- Real FPS must **match the file** (e.g. `13.91` on `*_norepeat_13.91fps.mp4`). Typing `29.97` on a
  13.91 file is rejected/ignored: Cut Video still plays at 13.91, so equal frame counts would
  **not** be the same events in time. If the container was stamped wrong at join time, retag with
  Frame C → **Metadata info** → **Adjust real FPS** before Fill Start/End.

Workflow tip: after norepeat, use each camera’s real rate (from the filename) → Sync marks →
Reference start/end → **Fill Start/End from FPS** → **Save + Cut Video**. Only cameras that
truly share a playback rate get equal frame counts (e.g. c1 and c4 both ~29.97).

Overrides are stored in the Load/Resume `.json` checkpoint (`fps_overrides`).

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

Part of *vailá* — Multimodal Toolbox. Updated 29 October 2026 (v0.4.12).
