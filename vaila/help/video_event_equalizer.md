# Video Event Equalizer

- **Version:** 0.4.14
- **Updated:** 2026-11-01
- **Category:** Tools
- **GUI:** Frame C → Make Sync file → Equalize by events (RIFE)
- **CLI:** `uv run --no-sync python -m vaila.video_event_equalizer --help`

Equalize two or more camera videos using three corresponding events each and local CUDA RIFE.
The first (reference) camera sets the grid; every other camera is aligned to it.
Outputs use a shared grid with configurable FPS and frame count; CSV maps identify
original and generated images in both cameras, including the reference.
See [setup, fields, example and limitations](syncvid.md#equalize-by-events-rife).

## Output FPS and frame count

- Leave **Output FPS** and **Output frame count** blank to keep the reference grid.
- Set only FPS to use `N = round_half_up(reference anchor span × FPS) + 1`.
  Buttons **60** and **120** fill the FPS field. Decimals are exact:
  `29.970 = 2997/100`; `30000/1001` is a different rate (NTSC).
- Set only the count to use exactly that many frames at the reference FPS.
- Set both to choose FPS and count independently. This changes playback speed.
- FPS must be positive and finite; count must be an integer of at least 3.
  An FPS-only request that calculates fewer than 3 frames is rejected.

Click **Preview**, review exact/decimal FPS, count, original/generated counts per
camera, first-to-last duration `(N−1)/FPS`, file duration `N/FPS`, and the duration
change relative to the reference anchor span. Then click **Run**. Editing any field
requires a new preview; probing runs in a worker so the window stays responsive.

Start and end occupy frames 1 and N. Contact is placed at
`1 + round_half_up((reference_contact − reference_start) / (reference_end − reference_start) × (N−1))`,
clamped to frames 2…N−1. Preview shows the contact rounding/clamping adjustment in
seconds and the total contact-time change relative to the reference. Each camera
maps its own two segments linearly onto these anchors. All three anchor images
are copied exactly; fractional positions use the installed CUDA RIFE model.
Increasing FPS creates visual estimates, not additional acquired measurements.

```bash
uv run --no-sync python -m vaila.video_event_equalizer \
  --reference /path/to/c2.mp4 --source /path/to/c1.mp4 \
  --reference-events 1,9,22 --source-events 1,9,19 \
  --output-fps 120 --output-frame-count 85 --output-dir /path/to/results
```

Add `--dry-run` for the same plan and full maps as JSON without loading RIFE or
writing outputs. Without `--dry-run`, the CLI prints the preview before export.
GUI Run prints the equivalent command with the selected output parameters.

### Main camera and frame count

Any camera can be the **Main**: pick the Main button of its row in the window (on the
command line, it is `--reference`). Its events set the contact and the timing, and its
video is the reference of the run. The other cameras are mapped onto its timeline.

**Frame count from** chooses how many output frames there are:

- **main camera (as before):** the main camera's own span. The main keeps its frames.
- **shortest camera (reduce):** the shortest event span in seconds. Longer cameras are
  compressed onto it, so they lose frames.
- **longest camera (increase):** the longest event span in seconds. Shorter cameras are
  stretched onto it with RIFE, so they gain generated frames.

The mode only sets the default count: a typed output frame count overrides it. FPS
stays the main camera's unless you type Output FPS. Preview prints the target span.

```bash
uv run --no-sync python -m vaila.video_event_equalizer \
  --reference /path/to/c1_passe_main.mp4 --reference-events 1,1,26 \
  --source /path/to/c2_passe_16mm.mp4 --source-events 1,1,39 \
  --mode increase --output-dir /path/to/results
```

### Contact on the start or end

When only the start and end of the sync are known, the contact may repeat one of
them: `1,1,39` puts the contact on the start, `1,26,26` on the end. The output grid
then places the contact on frame 1 or frame N, and the zero-length segment is skipped.
Start and end must differ, the contact must lie between them, and every camera must
use the same pattern as the reference (all on the start, all on the end, or all with
their own contact frame). Mixed patterns are refused. Outputs of a repeated contact
keep the reference grid exactly like any other run:

```bash
uv run --no-sync python -m vaila.video_event_equalizer \
  --reference /path/to/c2_passe_16mm.mp4 --reference-events 1,1,39 \
  --source /path/to/c1_passe_main.mp4 --source-events 1,1,26 \
  --output-dir /path/to/results
```

The reference sets the output frame count: here 39 frames, so the 26-frame camera
gets 37 generated frames. Choosing the 26-frame camera as reference instead gives
26 output frames, and the 39-frame camera is compressed onto them.

### Three or more cameras

Repeat `--source` and `--source-events` together, in the same order. Each camera
gets its own three events (1-based start, contact, end) and its own video in the
output folder, named `<role>_<video>_equalized.mp4`:

```bash
uv run --no-sync python -m vaila.video_event_equalizer \
  --reference /path/to/c2_16mm_left.mp4 --reference-events 1,9,22 \
  --source /path/to/c1_main.mp4 --source-events 1,9,19 \
  --source /path/to/c3_top.mp4 --source-events 1,8,20 \
  --output-dir /path/to/results
```

The manifest lists the cameras as `reference`, `source`, `source_2`, `source_3`,
and so on. Each camera has its own `*_frames.csv` map. The FreeKiki one-click reads
only `reference` and `source`. The GUI has **Add camera** and **Remove** for the
same cameras, one table row each; the window scrolls vertically and horizontally (wheel;
Shift + wheel). GUI Run prints the standard `>> vaila/video_event_equalizer: Equivalent CLI`
banner (`uv run --no-sync vaila/video_event_equalizer.py ...`). Two different files are
required; the same video twice is refused.

The manifest keeps the existing effective `output_fps` / `output_frame_count`
fields and records requested values, output anchors, rational durations and contact
adjustments. Both CSV maps retain their existing columns and add the same output
parameters and timing audit. `reference_frame` remains an integer on the default
grid and may contain a rational position on a custom grid; `reference_position`
always supplies that exact rational position. `event` labels the three anchors.
Original/generated counts are available in both the preview and final manifest.
Both cameras are preflighted at their own dimensions when interpolation is needed.
No new dependency or model download is used for these controls.

## Installation and reinstallation

**Practical-RIFE 4.25 is optional and is no longer installed by default.** Install it
(about 50 MB) in a folder you choose, then verify CUDA, in an existing *vailá* environment:

```bash
bash bin/setup_rife.sh --home /big/disk/rife   # portable folder on any disk
bash bin/setup_rife.sh                         # or inside the vailá folder (legacy layout)
```

On Windows: `pwsh bin/setup_rife.ps1 --home D:\rife`. Setup downloads the pinned
official code and model, verifies SHA256 hashes and runs CUDA inference. No new
extra, HF login or separate cuDNN installation is required.

**Choosing and moving the folder.** The equalizer reads RIFE from `--rife-home DIR`,
else from the environment variable `VAILA_RIFE_HOME`, else from the legacy folders
in the *vailá* checkout (`.local/third_party/Practical-RIFE/` and
`vaila/models/rife/train_log/`). `--rife-dir` and `--weights` override one folder
each. A chosen folder without RIFE is an error, never a silent fallback.
To copy an installed RIFE to another disk or computer with no network, hashes
verified and a real CUDA test on the copy:

```bash
bash bin/setup_rife.sh --export /other/disk/rife
uv run --no-sync python -m vaila.video_event_equalizer --gui --rife-home /other/disk/rife
```

A portable folder holds `Practical-RIFE/`, `train_log/` (with `flownet.pkl`) and
`setup_manifest.json`. Assets remain outside Git. Re-running setup reuses verified
files; see the [full setup instructions](syncvid.md#rife-setup) for offline restore
and recovery.
