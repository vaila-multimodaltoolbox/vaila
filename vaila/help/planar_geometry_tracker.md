# planar_geometry_tracker

## Module information

| Field | Value |
| --- | --- |
| **Category** | Processing |
| **File** | `vaila/planar_geometry_tracker.py` |
| **Version** | 0.4.1 |
| **Updated** | 15 September 2026 |
| **Author** | Paulo R. P. Santiago |
| **GUI** | Yes — **Geo Homog** button in `getpixelvideo.py` |
| **CLI** | Yes — `python -m vaila.planar_geometry_tracker --config ... --measurements-csv ...` |

---

## Description

For a stabilized video from fixed pixel markers, use [Video Stabilizer](video_stabilizer.md).
Its default similarity warp preserves shape; planar fits remain a separate metric
diagnostic layer. Background markers do not need metric coordinates.

Standalone planar-geometry homography tracker/extrapolator/gap-filler. Takes a
2D pixel-space marker CSV exported by `getpixelvideo.py` and a metric
target-geometry profile (TOML — EVA tatame mat, soccer pitch, court, …) and,
for every frame, fits the planar homography that maps the metric target plane
onto image pixels. With that per-frame homography it:

- **imputes** markers that are missing/occluded on a frame but whose position
  on the metric target plane is known (i.e. every other visible marker still
  pins down the plane);
- **extrapolates** points that have left the camera field of view, without
  clamping to the frame boundary — the projected pixel coordinate can fall
  outside `[0, width) x [0, height)`, which is expected and by design (a
  point 2 m past the sideline should project 2 m past the sideline in pixel
  space, not snap to the edge);
- **projects a wireframe** of the target geometry (perimeter/interior lines,
  circles, arcs) into image space for every frame, for visual QA and for
  building an "extended canvas" diagnostic video larger than the source
  frame so off-frame projections stay visible.

Each frame's homography is fit independently with `cv2.findHomography(...,
cv2.RANSAC, ransac_thresh)` from whatever markers are visible and
non-collinear on that frame (minimum 4 non-collinear points). When fewer than
4 usable points are available on a frame, the module falls back to
`cv2.estimateAffinePartial2D` propagated from the nearest frame that did have
a full homography, so a brief marker dropout does not stop tracking.

This is a different tool from `dlt2d.py`/`rec2d_one_dlt2d.py`: those do a
one-time least-squares DLT2D calibration fit from an exact point set (no
outlier rejection, no per-frame refit). `planar_geometry_tracker.py` refits a
robust RANSAC homography every frame from noisy, partial, changing
correspondences — the right tool when the visible marker set varies frame to
frame (occlusion, markers leaving/entering the FOV) rather than staying fixed
for one calibration shot.

## Usage

### From the GUI

In `getpixelvideo.py`, click **Geo Homog** (next to **Gap Fill**). Current
annotations are saved first, then you are prompted to choose a target
geometry TOML profile (see below); the module runs as a subprocess against
the just-saved CSV and the video currently loaded, and success/failure is
reported in the save toast. The equivalent `>>` CLI command is printed to the
console before the subprocess launches.

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
| `<stem>_imputed.csv` | Wide-format marker CSV (same `p{N}_x`/`p{N}_y` column convention `getpixelvideo.py` writes and reads) with occluded/missing markers filled in and out-of-FOV markers extrapolated — loads straight back into `getpixelvideo.py` via its Load button. |
| `dense_projected_points_long.csv` | Long-format per-frame projection of every target-geometry point (not just measured markers), including the dense circle/arc sample points. |
| `homographies.npz` | Per-frame 3x3 homography matrices (and affine-fallback frames flagged), for downstream reuse without refitting. |
| `target_calibration.ref3d` | Metric target-geometry reference in the repo's existing `.ref3d` convention (see `drawsportsfields.py`), readable by `dlt3d.py`, `quickmeasure.py`, `sapiens2_3d.py`. |
| `debug_projected_wireframe.mp4` | Only with `--debug-viz`: source video with the projected wireframe overlay drawn per frame. |

## See also

- [getpixelvideo.md](getpixelvideo.md) — marker digitizing tool that produces
  the input CSV and hosts the **Geo Homog** button.
- [dlt2d.md](dlt2d.md) / [rec2d_one_dlt2d.md](rec2d_one_dlt2d.md) — one-time
  DLT2D calibration fit, a different use case (fixed exact point set, no
  per-frame RANSAC refit).
