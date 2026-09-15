# Codex CLI implementation prompt — VAILA fixed-scene video stabilizer

You are working inside the **vailá Multimodal Toolbox** repository.

## Goal

Design and implement **one standalone production module**:

`vaila/video_stabilizer.py`

Its primary job is simple:

> Given a video and a `getpixelvideo.py` marker CSV containing 2D pixel coordinates of points that are physically fixed in the scene, generate a new stabilized video that looks as if the camera had remained fixed, while minimizing visual deformation.

The module must support the real use case where some fixed points belong to a **metric planar target** (e.g. a tatame on Z=0 with known X,Y coordinates), while other fixed points are **background-only 2D stabilization anchors** (wall/columns/etc.) with no known metric coordinates.

Do NOT require metric coordinates for video stabilization.

---

# Mandatory repository study before coding

First inspect and understand these existing modules and reuse code where appropriate instead of duplicating behavior:

- `vaila/getpixelvideo.py`
- `vaila/planar_geometry_tracker.py`
- `vaila/dlt2d.py`
- `vaila/rec2d.py`
- `vaila/rec2d_one_dlt2d.py`
- existing video metadata / ffmpeg helpers in the repo
- existing help/documentation conventions under `vaila/help/`
- existing CLI/style conventions

Pay special attention to `planar_geometry_tracker.py`:

- `load_target_geometry()`
- `load_measurements_csv()`
- `solve_frame_homographies()`
- `FrameHomography`
- `project_points()`
- `write_outputs()`

The planar tracker is useful, but its per-frame planar homography MUST NOT be used blindly to warp the whole image because it is only geometrically valid for the calibrated plane.

---

# Real input example

The input marker CSV is wide, as produced by `getpixelvideo.py`:

```text
frame,p0_x,p0_y,p1_x,p1_y,...,p10_x,p10_y
```

Example semantic grouping:

- `p0..p7`: physically fixed points on a tatame/floor.
  - Pixel coordinates vary with camera motion.
  - Their real-world coordinates may be known in a TOML planar geometry profile.
- `p8,p9,p10`: physically fixed points on walls/columns/background.
  - Visible through the whole clip.
  - Pixel coordinates vary with camera motion.
  - NO metric X,Y,Z coordinates are known.
  - They are valid stabilization anchors and must never be rejected merely because they do not exist in the metric geometry TOML.

Example metric floor geometry:

```toml
[target]
description = "Tatame de artes marciais com passo modular util intertravado de 0.96m"
type = "polygon"

[points]
0 = { name = "canto_inf_esq", x = 0.000, y = 0.000, z = 0.0 }
1 = { name = "meio_inf",      x = 0.480, y = 0.000, z = 0.0 }
2 = { name = "canto_inf_dir", x = 0.960, y = 0.000, z = 0.0 }
3 = { name = "meio_dir",      x = 0.960, y = 0.480, z = 0.0 }
4 = { name = "canto_sup_dir", x = 0.960, y = 0.960, z = 0.0 }
5 = { name = "meio_sup",      x = 0.480, y = 0.960, z = 0.0 }
6 = { name = "canto_sup_esq", x = 0.000, y = 0.960, z = 0.0 }
7 = { name = "meio_esq",      x = 0.000, y = 0.480, z = 0.0 }
```

---

# Critical geometric rule

Separate **visual stabilization** from **metric planar calibration**.

There are TWO different transforms:

## A. Visual stabilization transform `S_t`

Purpose:
- render the final stabilized video;
- preserve human/object shape;
- stabilize the fixed scene globally.

Default transform family:
- 2D **similarity transform** only:
  - translation X/Y
  - rotation
  - uniform scale
- no shear;
- no independent X/Y scale;
- no projective terms.

Mathematically:

```text
x' = a*x - b*y + tx
y' = b*x + a*y + ty
```

This must be the default because it strongly limits image deformation.

## B. Metric floor homography `H_floor,t`

Purpose:
- represent the known Z=0 plane;
- calculate floor reprojection quality;
- optionally impute missing floor control points;
- optionally export DLT/homography diagnostics for biomechanical reconstruction.

It MUST NOT be the default warp used on the entire video.

Do not try to force wall points and floor points into one projective homography: they are not coplanar.

---

# Required operating modes

One Python module, but support these modes through CLI.

## 1. `visual` — DEFAULT

Requires only:
- video
- marker CSV

Uses fixed pixel markers only.

Example:

```bash
uv run python -m vaila.video_stabilizer \
  --video JJ_Kabuto.mp4 \
  --markers tatame_markers.csv \
  --stabilization-markers 0-10 \
  --anchor-markers 8,9,10 \
  --mode visual \
  --model similarity \
  --reference auto \
  --canvas union
```

No TOML geometry is required.

## 2. `hybrid`

Requires:
- video
- marker CSV
- planar geometry TOML

Example:

```bash
uv run python -m vaila.video_stabilizer \
  --video JJ_Kabuto.mp4 \
  --markers tatame_markers.csv \
  --geometry-config vaila/models/planar_targets/tatame_1x1m.toml \
  --metric-markers 0-7 \
  --stabilization-markers 0-10 \
  --anchor-markers 8,9,10 \
  --mode hybrid \
  --model similarity \
  --canvas union
```

In hybrid mode:
- use planar metric geometry only for floor homography, diagnostics, validation and optional floor-marker imputation;
- use ALL selected 2D fixed markers for the shape-preserving visual stabilization transform;
- `p8+` do not need to exist in the TOML;
- do not warp the final video using `H_floor,t`.

## 3. optional `floor-lock` diagnostic mode

Allow a deliberately planar stabilization for research/debug comparison:

```bash
--mode floor-lock
```

This may warp frames with the floor homography relative to a reference floor homography.

It MUST:
- be explicitly labeled as planar-only;
- print a warning that non-coplanar scene regions can distort;
- never be the default or auto-selected mode.

---

# Reference frame selection

Support:

```text
--reference auto
--reference first
--reference frame:123
```

`auto` must choose a REAL frame, not a synthetic average image.

Recommended algorithm:

1. require good visibility of priority anchors;
2. prefer high total marker count;
3. prefer broad spatial distribution / convex hull area;
4. compute a robust median marker configuration over time;
5. choose the real frame nearest to that central configuration (medoid-like);
6. reject frames with poor geometry.

Never create a synthetic reference marker layout by independently averaging every marker unless explicitly requested.

Save the chosen reference frame in diagnostics.

---

# Marker roles

Implement explicit marker groups:

```text
--stabilization-markers 0-10
--anchor-markers 8,9,10
--metric-markers 0-7
```

Parsing should support:
- `0-10`
- `0,1,2,8,9,10`
- combinations such as `0-7,10,12-14`
- `all`

Definitions:

### stabilization markers
All physically static pixel-space points allowed to contribute to `S_t`.

### anchor markers
Highly reliable fixed background points. They remain pixel-only.
They should receive configurable extra weight.

### metric markers
Subset with known real-world XY coordinates in the planar geometry TOML.

Do not assume marker IDs are contiguous.

---

# Visual transform estimation

Default estimator must NOT be a full homography.

Implement a robust weighted similarity solve.

Preferred strategy:

1. collect current-frame and reference-frame correspondences;
2. apply spatial balancing so a dense cluster of floor points cannot dominate a few widely distributed background points;
3. add `anchor_weight` for explicitly selected anchors;
4. estimate a weighted 2D similarity transform with least squares / Procrustes / Umeyama-style solution;
5. perform robust residual reweighting (Huber or Tukey) for obvious digitization mistakes;
6. solve again;
7. compute residual diagnostics.

RANSAC may be offered as an optional estimator:

```text
--estimator robust-lsq   # default
--estimator ransac
```

But do NOT use RANSAC as default to classify valid background points as outliers merely because floor and wall exhibit parallax.

Add:

```text
--anchor-weight 2.0
```

Default around 2.0, but choose a defensible value after testing.

Spatial balancing:
- calculate marker distribution in normalized image coordinates;
- points in dense local clusters should receive less aggregate weight than isolated anchors;
- keep behavior deterministic.

---

# Missing markers / fallback

The CSV may contain NaNs.

Requirements:

- Do not fail when some points disappear.
- Use all currently available valid correspondences.
- Similarity transform requires at least 2 useful non-coincident correspondences, but prefer >=3.
- If a frame is underconstrained:
  1. interpolate transform parameters temporally between trustworthy neighboring frames;
  2. for leading/trailing gaps, propagate nearest trustworthy transform;
  3. label the method in diagnostics.
- In hybrid mode, the planar tracker may be used to impute missing `p0..p7`, but imputed planar points must be flagged and optionally down-weighted relative to manual observations.

Do NOT chain frame-to-frame transforms as the primary method.
Always estimate each trustworthy frame against the fixed reference to prevent drift.

---

# Temporal regularization

Do NOT smooth arbitrary 3x3 homography coefficients for visual stabilization.

For a similarity transform decompose into:

```text
tx
ty
theta
log_scale
```

Unwrap theta over time.

Support:

```text
--smooth none
--smooth savgol
--smooth lowpass
```

Choose a conservative default based on the tests.

The goal is:
- remove digitization jitter;
- avoid lag;
- avoid rubber-band motion.

Raw and smoothed values must both be exported.

---

# Hybrid planar geometry behavior

When `--geometry-config` is supplied:

1. load geometry using existing `planar_geometry_tracker.py` functionality;
2. only use marker IDs shared by `--metric-markers` and the geometry profile;
3. estimate `H_floor,t` from real XY -> pixel coordinates;
4. use normalized / numerically stable homography estimation;
5. report:
   - number of floor correspondences;
   - floor reprojection RMSE;
   - RANSAC/inlier info if applicable;
   - condition / degeneracy flags;
6. optionally use the floor homography to impute missing metric control points;
7. NEVER add `p8,p9,p10` to the floor homography unless metric world coordinates actually exist for them;
8. NEVER apply `H_floor,t` to the entire final video in `visual` or `hybrid` mode.

The known metric floor is a physical constraint and diagnostic layer, not permission to projectively distort the full scene.

---

# Canvas policy

Support:

```text
--canvas union
--canvas original
--canvas crop
```

Default: `union`.

## union

Two-pass workflow:

Pass 1:
- estimate final `S_t`;
- transform the four source-frame corners for every frame;
- calculate global min/max X/Y;
- add one constant translation matrix so all stabilized frames use ONE fixed output canvas.

Pass 2:
- warp every frame onto that fixed canvas.

Important:
- preserve every pixel that was actually captured when geometrically possible;
- black/empty borders are valid;
- do not dynamically resize or recenter canvas frame by frame;
- no auto-zoom by default;
- no generative filling.

Provide:

```text
--border black
```

as the initial production behavior.

---

# Video warp

For similarity/affine-compatible transforms use `cv2.warpAffine`.
Avoid `warpPerspective` in visual mode.

Use high-quality interpolation appropriate for video.

Keep:
- exact frame order;
- original effective FPS;
- original orientation;
- original duration as closely as possible.

Read metadata with existing repo helpers/ffprobe where possible.

---

# Audio

The generated stabilized video should preserve the original audio when present.

Recommended pipeline:

1. render temporary silent stabilized MP4;
2. use `ffmpeg` to mux/copy audio from the source;
3. preserve timestamps/duration safely;
4. if no audio exists, simply finalize the silent video;
5. report audio status in diagnostics/log.

Do not require audio for success.

---

# Outputs

Default output directory:

```text
<video_dir>/vaila_stabilized_<timestamp>/
```

Write at least:

```text
<video_stem>_stabilized.mp4
stabilization_transforms.csv
stabilized_markers.csv
stabilization_diagnostics.csv
stabilization_report.html
```

Hybrid mode additionally:

```text
floor_homographies.npz
floor_diagnostics.csv
```

Optional:

```text
debug_overlay.mp4
```

---

# stabilization_transforms.csv

One row per frame with at least:

```text
frame
reference_frame
method
n_markers
n_manual_markers
n_imputed_markers
tx_raw
ty_raw
rotation_deg_raw
scale_raw
tx
ty
rotation_deg
scale
matrix_00
matrix_01
matrix_02
matrix_10
matrix_11
matrix_12
marker_rmse_px
anchor_rmse_px
floor_marker_rmse_px
is_interpolated
```

Hybrid-specific metrics may be NaN in visual mode.

---

# stabilized_markers.csv

Apply the exact final video transform `S_t` to every valid marker coordinate from the original input CSV.

Output in normal VAILA wide format:

```text
frame,p0_x,p0_y,...
```

This allows checking that fixed points are actually stationary in stabilized coordinates.

Do not overwrite the input CSV.

---

# Diagnostics / report

Compute BEFORE and AFTER stabilization motion for each static marker.

Report at least:

- per-marker RMS displacement relative to reference;
- per-marker median displacement;
- all-marker aggregate;
- anchor-only aggregate;
- floor-only aggregate when metric markers exist;
- percentage of frames solved directly;
- percentage interpolated;
- max scale deviation;
- max rotation correction;
- canvas dimensions;
- black-border fraction estimate if feasible;
- chosen reference frame;
- audio preservation result.

HTML report should contain simple plots/tables if the repo already has a lightweight convention; otherwise keep it dependency-light.

---

# Debug overlay

Optional:

```text
--debug-overlay
```

Generate a separate diagnostic video showing:
- source marker locations;
- stabilized marker locations;
- reference locations;
- marker IDs;
- residual vectors;
- method/fallback text;
- current rotation/scale/translation;
- floor RMSE in hybrid mode.

Do not burn overlays into the production stabilized MP4.

---

# CLI

Create a full `argparse` CLI.

Minimum flags:

```text
--video PATH
--markers PATH
--output-dir PATH
--mode visual|hybrid|floor-lock
--model similarity|affine
--estimator robust-lsq|ransac
--stabilization-markers SPEC
--anchor-markers SPEC
--anchor-weight FLOAT
--metric-markers SPEC
--geometry-config PATH
--reference auto|first|frame:N
--smooth none|savgol|lowpass
--canvas union|original|crop
--border black
--debug-overlay
--no-audio
```

Defaults:
- mode = visual
- model = similarity
- estimator = robust-lsq
- reference = auto
- canvas = union
- preserve audio = yes

`affine` is an explicit opt-in research alternative because it can introduce shear / anisotropic scale.

---

# API

Expose a callable entry point:

```python
run_video_stabilizer(
    video_path: Path,
    markers_csv: Path,
    output_dir: Path | None = None,
    *,
    mode: str = "visual",
    model: str = "similarity",
    stabilization_markers: str = "all",
    anchor_markers: str | None = None,
    anchor_weight: float = 2.0,
    metric_markers: str | None = None,
    geometry_config: Path | None = None,
    reference: str = "auto",
    smooth: str = "savgol",
    canvas: str = "union",
    preserve_audio: bool = True,
    debug_overlay: bool = False,
) -> dict[str, Path]:
    ...
```

Keep computational functions independently testable.

---

# Reuse rules

Prefer importing and reusing proven helpers from `planar_geometry_tracker.py` for:
- TOML target parsing;
- marker CSV parsing;
- planar point projection;
- planar diagnostics.

But avoid coupling the visual stabilizer to GUI state.

Do not mutate `getpixelvideo.py` for the first implementation unless strictly necessary.

Do not duplicate a generic ffprobe/audio helper if the repository already contains one.

---

# Important numerical / geometry requirements

- use float64 for transform estimation;
- normalize coordinates before homography solves;
- reject NaN/inf;
- detect coincident / degenerate correspondences;
- avoid explicit `inv(B.T @ B)` normal-equation solves for new DLT code;
- use SVD / `lstsq` / stable OpenCV implementations;
- use `np.linalg.solve` instead of explicit inverse where applicable;
- never silently clamp projected metric coordinates to the frame;
- do not silently substitute a homography when the user selected similarity;
- log every fallback.

---

# Acceptance tests using the supplied JJ_Kabuto example

Run the real regression test with:
- `JJ_Kabuto.mp4` or the available JJ_Kabuto input video;
- `tatame_markers.csv`;
- p0..p7 metric floor points;
- p8,p9,p10 upper-background static markers.

Test at least:

```bash
--mode visual --model similarity --stabilization-markers 0-10 --anchor-markers 8,9,10
```

and, when the TOML profile is available:

```bash
--mode hybrid --model similarity --metric-markers 0-7 \
--stabilization-markers 0-10 --anchor-markers 8,9,10
```

Verify automatically:

1. output MP4 opens;
2. output frame count equals source frame count;
3. output FPS matches source within metadata tolerance;
4. audio is preserved when source audio exists;
5. output uses one constant canvas size;
6. no NaN/inf transforms;
7. `p8,p9,p10` displacement after stabilization is substantially smaller than before;
8. total fixed-marker displacement is substantially smaller than before;
9. scale remains near 1 unless source motion requires otherwise;
10. no projective perspective coefficients exist in similarity mode;
11. hybrid floor diagnostics are written;
12. p8..p10 are NEVER interpreted as metric world points unless explicitly defined in geometry.

Do not assert an arbitrary pixel threshold before inspecting manual-marker noise and parallax. Report actual before/after metrics.

---

# Unit tests

Add focused tests for:

- marker range parser;
- wide CSV parser with missing values;
- reference-frame selection;
- exact synthetic similarity recovery;
- weighted similarity recovery;
- robust outlier handling;
- angle unwrap;
- temporal interpolation;
- union-canvas bounds;
- marker transform output;
- planar metric marker / visual-only marker separation;
- hybrid mode accepting p8+ without TOML coordinates;
- failure messages for insufficient correspondences.

Synthetic tests should generate known transforms and verify numerical recovery.

---

# Documentation

Create:

```text
vaila/help/video_stabilizer.md
vaila/help/video_stabilizer.html
```

Follow existing help-page conventions.

Document clearly:

> Metric coordinates are optional for visual stabilization.

and:

> A planar homography is valid for the calibrated plane and is not used as the default whole-frame video warp.

Include the JJ_Kabuto-style example with:
- p0-p7 metric floor;
- p8-p10 pixel-only upper anchors.

Run the repository help-index generator after adding the topic.

---

# Implementation workflow

Do this in two explicit phases.

## Phase 1 — PLAN

Before editing code, inspect the repository and write a concrete implementation plan containing:

1. files to create/change;
2. existing functions to reuse;
3. data flow;
4. transform mathematics;
5. marker-role logic;
6. fallback logic;
7. canvas strategy;
8. audio strategy;
9. diagnostics;
10. unit/regression tests;
11. risks, especially parallax and non-coplanarity.

Call out any inconsistency you find in this prompt versus the current repo.

Do not code until the plan is internally coherent.

## Phase 2 — IMPLEMENT

Then implement the plan.

After implementation:

1. run formatter/linter used by the repo;
2. run targeted unit tests;
3. run relevant existing tests;
4. run the real JJ_Kabuto smoke/regression test;
5. inspect output diagnostics;
6. report exact before/after stabilization metrics;
7. report generated output paths;
8. list any remaining limitations.

Do not claim visual success solely from a low floor homography reprojection error.

The key success criterion is:

> fixed scene becomes substantially more stable while the video retains natural, shape-preserving geometry.

No commit unless explicitly requested.
