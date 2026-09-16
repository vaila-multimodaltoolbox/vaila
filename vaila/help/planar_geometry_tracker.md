# planar_geometry_tracker

## Module information

| Field | Value |
| --- | --- |
| **Category** | Tools → Video and Image (**Planar Geo**); also Processing via getpixelvideo |
| **File** | `vaila/planar_geometry_tracker.py` |
| **Version** | 0.4.3 |
| **Updated** | 15 September 2026 |
| **Author** | Paulo R. P. Santiago |
| **GUI** | Yes — Frame C → **Video and Image → Planar Geo**, and **Geo Homog** in `getpixelvideo.py` |
| **CLI** | Yes — `python -m vaila.planar_geometry_tracker --config ... --measurements-csv ...` |

---

## Description

For a stabilized video from fixed pixel markers, use [Video Stabilizer](video_stabilizer.md).
Its default similarity warp preserves shape; planar fits remain a separate metric
diagnostic layer. Background markers do not need metric coordinates.

Standalone planar-geometry gap-filler / extrapolator. Takes a 2D pixel-space
marker CSV from `getpixelvideo.py` and a metric target-geometry TOML (EVA
tatame, soccer pitch, …). For every frame it:

1. **Fits per-frame DLT2D** on *visible measured* world↔pixel pairs
   (`dlt2d.py`). Digitized markers are calibration observations only.
2. **Reprojects the full metric geometry** through that DLT so the output
   square/rectangle stays projectively consistent (collinear world edges →
   collinear image edges; e.g. 0.96 m tatame).
3. **Topology bootstrap** — if fewer than 4 measured points, fill gaps via
   midpoints / line intersections, then refit DLT when possible.
4. **Temporal DLT fallback** — when a frame cannot be calibrated, reuses the
   previous frame's DLT parameters.

Output is always a **new** `*_imputed.csv` (input measurements are never
overwritten). Optional `geometry_animation.html` shows pixel + world (rec2d)
geometry frame by frame. Extrapolated coordinates are never clamped to the
frame boundary.

## Usage

### From the GUI

**Main vailá:** Frame C → **Video and Image → Planar Geo**. Pick the marker
CSV, target-geometry TOML, optional reference video, and output directory
(Cancel on output = auto `processed_planar_geom_<timestamp>/` next to the CSV).
With a video selected, `--debug-viz` is enabled automatically.

**Inside getpixelvideo:** click **Geo Homog** (next to **Gap Fill**). Current
annotations are saved first, then a wizard asks:

1. **Mode** — `1` = shipped/custom TOML profile, or `2` = generic rectangle
   (4 corner marker IDs + width/height in metres).
2. **TOML path** (mode 1) — `1=tatame_1x1m`, `2=soccerfield_broadcast`, or
   `3=browse…`.
3. **Marker → geometry map** (mode 1) — pairs `marker_id:geom_id`
   (e.g. `0:0,1:1,2:2,3:3`); need ≥4 pairs. Optional `W,H` resize of the
   profile bounding box.
4. **Rectangle** (mode 2) — four CSV marker indices in order SW, SE, NE, NW,
   then `width,height` metres.
5. **Edit pause** — session files are written to `processed_geom_<timestamp>/`;
   edit the TOML/CSV in a terminal (**imagination!** / `uv` venv) if needed,
   then press Enter to run.
6. **Optional save** — after success, choose whether to write DLT geometry
   markers to a **new** `<stem>_geom_dlt_markers.csv` (never overwrites the
   original `*_markers.csv`).

The getpixelvideo path runs with `--debug-viz` always on. On success,
*getpixelvideo* reloads the imputed CSV, enables a live wireframe overlay
drawn from resolved marker pixels (**Shift+G** toggles), writes
`debug_projected_wireframe.mp4`, and `geometry_animation.html`.

No-args CLI also opens the Planar Geo file-dialog GUI:

```bash
uv run python -m vaila.planar_geometry_tracker
```

### From the CLI

```bash
uv run python -m vaila.planar_geometry_tracker \
  --config vaila/models/planar_targets/tatame_1x1m.toml \
  --measurements-csv path/to/markers.csv \
  --video-path path/to/video.mp4 \
  --output-dir ./vaila_tracker_output \
  --extended-canvas
```

### CLI arguments

| Flag | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--config` | Yes | — | Path to the target-geometry TOML profile. |
| `--measurements-csv` | Yes | — | Marker CSV from `getpixelvideo.py`. Wide format (`frame,p0_x,p0_y,p1_x,p1_y,...`, what `getpixelvideo.py` actually writes) or long format (`frame,point_id,u,v`). |
| `--video-path` | No | `None` | Source video, for pixel-dimension reference and (with `--debug-viz`) the overlay render. |
| `--output-dir` | No | `./vaila_tracker_output` | Timestamp-free output directory (created if missing) — caller controls the path so it can point straight back at the GUI's session folder. |
| `--ransac-thresh` | No | `3.0` | RANSAC reprojection-error threshold (pixels) passed to `cv2.findHomography`. |
| `--canvas-scale` | No | `1.8` | Extended-canvas size multiplier relative to the source frame, when `--extended-canvas` is set. |
| `--extended-canvas` | No | off | Render the oversized-canvas diagnostic video so extrapolated (off-frame) projections stay visible instead of being clipped. |
| `--debug-viz` | No | off | Write a debug overlay video with the projected wireframe drawn over the source frames. Requires `--video-path`. |

## Target-geometry TOML profile

A profile describes a rigid planar target in real-world metric units. Two are
shipped in `vaila/models/planar_targets/`:

- `tatame_1x1m.toml` — an 8-point EVA tatame mat (4 corners + 4 edge
  midpoints), perimeter + edge lines only.
- `soccerfield_broadcast.toml` — a full 105 x 68 m FIFA pitch with corner
  flags, halfway line, both penalty boxes, and the center circle + both
  penalty arcs via `[[circles]]`/`[[arcs]]`.

### Schema

```toml
[target]
name = "tatame_eva_1x1m"
description = "Interlocking martial arts EVA mat (0.96m modular interlocked pitch)"
type = "polygon"

[points]
0 = { name = "corner_sw", x = 0.000, y = 0.000, z = 0.0 }
1 = { name = "mid_s",     x = 0.480, y = 0.000, z = 0.0 }
# ... one entry per marker, numbered to match the CSV's p{N}_x/p{N}_y columns

[topology]
perimeter = [0, 1, 2, 3, 4, 5, 6, 7]   # optional, closed outline for the wireframe
lines = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [4, 5], [5, 6], [6, 7], [7, 0]
]                                       # index pairs, each drawn as a segment

# Optional, repeatable — circular/arc features (e.g. a center circle, a penalty arc)
[[circles]]
name = "center_circle"
center_point = 6        # index into [points]
radius = 9.15            # metres
samples = 64              # polyline vertex count for the projected circle

[[arcs]]
name = "left_penalty_arc"
center_point = 11
radius = 9.15
angle_start_deg = -53.0
angle_end_deg = 53.0
samples = 32
```

- `[points]` entries are keyed by the same integer index used in the marker
  CSV's `p{N}_x`/`p{N}_y` columns (or `point_id` in long-format CSVs) — the
  point count and numbering in the TOML must match what was digitized in
  `getpixelvideo.py`.
- `[topology].perimeter` is optional (used to close the outline); `lines` is
  the general index-pair list drawn for every frame's wireframe.
- `[[circles]]`/`[[arcs]]` are optional array-of-tables; each is parametrized
  in the metric target plane and projected through that frame's homography,
  so a true circle on the ground renders as the correct ellipse in image
  space.

## Outputs

Written to `--output-dir`:

| File | Contents |
| --- | --- |
| `<stem>_imputed.csv` | Wide-format marker CSV with **full DLT reprojection** of the TOML geometry (same `p{N}_x`/`p{N}_y` convention). New file — never overwrites the input. |
| `dense_projected_points_long.csv` | Long-format per-frame projection; `is_measured` + `reproj_error_px` are diagnostic vs. the digitized observations. |
| `homographies.npz` | Per-frame 3x3 homography matrices (DLT→H), for downstream reuse without refitting. |
| `geometry_animation.html` | Interactive slider: pixel wireframe + world (rec2d) vs TOML ideal. |
| `target_calibration.ref3d` | Metric target-geometry reference in the repo's existing `.ref3d` convention (see `drawsportsfields.py`), readable by `dlt3d.py`, `quickmeasure.py`, `sapiens2_3d.py`. |
| `debug_projected_wireframe.mp4` | Only with `--debug-viz`: source video with the projected wireframe overlay drawn per frame. |

## See also

- [getpixelvideo.md](getpixelvideo.md) — marker digitizing tool that produces
  the input CSV and hosts the **Geo Homog** button.
- [dlt2d.md](dlt2d.md) / [rec2d_one_dlt2d.md](rec2d_one_dlt2d.md) — one-time
  DLT2D calibration fit, a different use case (fixed exact point set, no
  per-frame RANSAC refit).
