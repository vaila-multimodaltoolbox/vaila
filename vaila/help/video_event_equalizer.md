# Video Event Equalizer

- **Version:** 0.4.13
- **Updated:** 2026-10-30
- **Category:** Tools
- **GUI:** Frame C → Make Sync file → Equalize by events (RIFE)
- **CLI:** `uv run --no-sync python -m vaila.video_event_equalizer --help`

Equalize two camera videos using three corresponding events and local CUDA RIFE.
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
