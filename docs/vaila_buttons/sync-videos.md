# Make Sync File — Syncvid

**Button:** Frame C → Video and Image → C_B_r2_c3 — Make Sync file  
**Module:** [`vaila/syncvid.py`](../../vaila/syncvid.py)  
**Version:** 0.4.12 · **Updated:** 2026-10-28

Make Sync file opens the multi-camera Pygame player. The default workflow creates the familiar
sync TXT and hands it to Cut Video. The event selector adds physical-time calibration workflows;
the separate manual-cut mode exports clips directly.

## Choose the workflow

| Workflow | What it does | Export behavior |
|---|---|---|
| **FPS** (default) | Aligns a selected event and reference interval using each camera's FPS. | **Save + Cut Video** creates synchronized cuts through the existing TXT workflow. |
| **Events: nearest** | Maps ordered event pairs to physical times, optionally accounting for reviewed repeated frames. | Cut Video exports a common time grid with a per-frame time/error map; it selects the nearest retained source image. |
| **Events: interpolate** | Keeps every original frame in each selected interval and inserts bidirectional Farnebäck image estimates between event anchors. | Exports a shared frame grid at the minimum exact FPS needed for all source sample times, subject to **Max output FPS** (default 240, configurable, e.g. 480). |
| **Manual cuts side by side (M)** | Lets each camera have independent inclusive Start/End bounds and manual D/Delete frame removals. | Save all or the active camera writes new clips at their respective source FPS, without synchronization or interpolation. |

For event-based modes, paired events must mark the same physical instants. The reference clock is an
explicit timing assumption; confirm a checkpoint clock when it differs from the current reference
file rate. Interpolated images are estimates and can show artifacts around the ball, feet and
occlusions. They do not create new physical observations. See the detailed
[Syncvid help](../../vaila/help/syncvid.md) for checkpoint handling, keyboard controls, output maps,
CLI examples and limitations.

Select the desired mode directly; the active button is highlighted. The preview shows calculated
FPS, final count per camera and original/interpolated counts in each row before saving. A 480 ceiling
authorizes a higher grid without forcing 480 FPS or changing the recorded reference clock.
For `1/9/19 ↔ 1/9/22` at `979001/32666`, use 480 to produce 235 frames per camera at
`12727013/32666` FPS, with events at 1, 105 and 235. Both cameras preserve every original.

## Manual cuts

In side-by-side mode, click a pane to make its camera active. I/O set that camera's inclusive bounds;
D/Delete toggles removal of the current original frame. The live remaining-frame counts help the user
choose matching counts. Saving only exports the selected originals in order; removing frames shortens
duration, and equal counts imply equal duration only when source FPS also matches. This mode does not
create or apply a sync file. See [Manual Video Cuts help](../../vaila/help/manual_video_cuts.md).

## Command-line examples

```bash
# Open the usual synchronization player
uv run --no-sync python -m vaila.syncvid --input-dir /path/to/cameras

# Calibrate physical event times and save a manifest
uv run --no-sync python -m vaila.syncvid --events --input-dir /path/to/cameras \
  --checkpoint /path/to/session.json --reference-fps 30000/1001 \
  --output /tmp/session.events.json

# Preserve all originals and synthesize intermediate images during Cut Video export
uv run --no-sync python -m vaila.syncvid --events --interpolate \
  --input-dir /path/to/cameras --checkpoint /path/to/session.json \
  --reference-fps 30000/1001 --output /tmp/session.interpolated.events.json

# Raise the limit for a saved, confirmed clock; keep that clock unchanged
uv run --no-sync python -m vaila.syncvid --events --interpolate --max-output-fps 480 \
  --input-dir /path/to/cameras --checkpoint /path/to/session.json \
  --output /tmp/session.high_rate.events.json

# Independently trim/remove frames and export without synchronization
uv run --no-sync python -m vaila.syncvid --manual-cuts --input-dir /path/to/cameras
```

The GUI prints its equivalent CLI command with the `>>` prefix. For video export, choose a new output
directory; existing files are preserved. The interpolation CLI computes its output FPS automatically.
