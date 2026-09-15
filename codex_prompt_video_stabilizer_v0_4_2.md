# Codex CLI Prompt — Improve VAILA Video Stabilizer to v0.4.2

You are working inside the **vailá Multimodal Toolbox** repository.

Your task is to review and improve the existing fixed-scene video stabilizer implementation, not rewrite it from scratch.

Target version: **0.4.2**

Primary implementation:
- `vaila/video_stabilizer.py`

Compatibility launcher:
- `vaila/vaila_ground_stabilizer.py`

Related modules to inspect and reuse:
- `vaila/planar_geometry_tracker.py`
- `vaila/dlt2d.py`
- `vaila/rec2d.py`
- `vaila/rec2d_one_dlt2d.py`
- `vaila/getpixelvideo.py`
- `vaila/ffmpeg_utils.py`
- `vaila/numberframes.py`

Documentation:
- `vaila/help/video_stabilizer.md`
- `vaila/help/video_stabilizer.html`
- help index generator / index files

Tests / real regression material:
- the JJ_Kabuto tatame stabilization example if present in the repo/test assets
- marker CSV with p0..p10
- p0..p7 = metric floor controls
- p8,p9,p10 = pixel-only upper-background anchors

Do not commit unless explicitly requested.

---

# 0. First principle: preserve what is already correct

The current implementation already has several good design decisions. Keep them unless a test proves they are wrong.

Preserve:

1. Visual stabilization and planar metric calibration are separate concepts.
2. Default video warp is a **shape-preserving 2D similarity transform**.
3. Similarity allows only:
   - translation x/y
   - rotation
   - uniform scale
4. Do NOT introduce projective terms into visual/hybrid output.
5. `floor-lock` remains an explicit planar diagnostic mode and may use a homography.
6. p8+ pixel-only anchors do NOT require metric coordinates.
7. Every trustworthy frame is estimated against a fixed canonical reference, not chained frame-to-frame as the primary method.
8. Spatial marker balancing and anchor weighting remain.
9. Robust weighted least squares remains the default estimator.
10. RANSAC remains optional, not default.
11. Savitzky-Golay remains a valid default temporal regularizer unless tests show otherwise.
12. Union canvas remains the default.
13. Black borders are allowed; do not invent content.
14. Input CSV is never overwritten.
15. Debug overlay remains separate from the production video.
16. GUI remains thread-safe and CLI-capable.
17. The legacy launcher remains a thin compatibility wrapper.

The current weighted similarity solver should NOT be replaced casually. Add tests around it first.

---

# 1. Required first step: audit current repository state

Before editing code:

- inspect the full current `video_stabilizer.py`;
- inspect the compatibility launcher;
- inspect the current Markdown + HTML help;
- inspect repo video encoding utilities;
- inspect current tests;
- inspect current planar tracker;
- run the existing targeted tests if available;
- run `python -m py_compile` on the relevant modules;
- run the current JJ_Kabuto regression if assets are available.

Write a concise implementation plan before modifying code.

The plan must explicitly state:
- what currently works;
- what is incomplete;
- which functions will be changed;
- which new helpers/tests will be added;
- any behavior or CLI compatibility risks.

---

# 2. Main correction: make `hybrid` actually hybrid

Current problem:

`hybrid` computes floor homographies and floor diagnostics, but the visual transform `S_t` is effectively estimated the same way as `visual`. Metric floor information does not materially improve the visual stabilization.

Fix this without ever warping the whole frame with `H_floor,t`.

## 2.1 Role separation

Maintain three marker concepts:

```text
stabilization markers:
    all static 2D points allowed to estimate visual S_t

anchor markers:
    highly reliable static 2D background points
    e.g. p8,p9,p10
    no metric coordinates required

metric markers:
    subset of planar controls with known real XY
    e.g. p0..p7
```

## 2.2 Safe hybrid use of floor geometry

Use metric floor geometry only to provide:

- floor fit quality;
- planar diagnostics;
- optional reconstruction of MISSING metric control pixels;
- confidence information.

Do NOT:
- use p8+ inside the floor homography unless they really have world coordinates;
- turn the whole visual warp into a homography;
- generate dense virtual floor anchors that dominate visual stabilization;
- force non-coplanar wall/background points to obey the floor plane.

## 2.3 Floor-derived imputation

Implement optional low-confidence imputation of missing p0..p7 observations in hybrid mode.

Rules:

1. Fit `H_floor,t` only from valid observed metric controls.
2. If the fit is high-quality, project a missing metric world point through `H_floor,t`.
3. Mark the projected point as imputed.
4. Add it to visual similarity estimation only with a low configurable weight.

Add CLI/API option:

```text
--hybrid-imputed-weight 0.25
```

Default around `0.25`.

Manual observed points always remain more important than imputed points.

Do not count imputed points as manual points.

If floor quality is poor, do not impute.

If fewer than 4 non-degenerate metric controls exist and no trustworthy floor fit is available, do not invent a projective fit from too little data.

For short gaps, if useful, interpolate the projected pixel position of the missing floor marker over time rather than interpolating arbitrary homography matrix coefficients.

Do NOT smooth raw homography coefficients for the visual stabilizer.

## 2.4 Hybrid quality gating

Create an explicit floor quality function using metrics such as:

- correspondence count;
- all-point RMSE;
- inlier RMSE;
- inlier ratio;
- maximum residual;
- condition number;
- degeneracy.

Return a quality state such as:

```text
good
usable
poor
unavailable
```

Only `good` / `usable` floor fits may contribute imputed visual correspondences.

Export the quality state per frame.

---

# 3. Improve floor homography estimation and diagnostics

The current floor diagnostics are not sufficient because a RANSAC homography may fit a small subset extremely well while leaving other known physical controls with large residuals.

## 3.1 Export separate residual metrics

For every frame export at least:

```text
frame
n_floor_correspondences
n_inliers
floor_inlier_ratio
floor_rmse_all_px
floor_rmse_inliers_px
floor_median_all_px
floor_max_residual_px
condition
degenerate
fit_quality
method
```

Do not call an all-point residual simply `floor_rmse_px` if that hides the RANSAC distinction.

Keep a compatibility alias only if needed, documented clearly.

## 3.2 Fit methods

Inspect the current floor fitter and compare:

A. all-point normalized/direct homography fit;
B. robust all-point IRLS / M-estimator if practical;
C. OpenCV RANSAC / USAC method as diagnostic/outlier-resistant alternative.

Because these are manually declared physical correspondences, do not assume RANSAC-selected four-point subsets are automatically superior.

Prefer a numerically stable method:
- float64;
- normalized coordinates;
- SVD / stable OpenCV solver;
- no explicit normal-equation inversion.

Add an internal or CLI-selectable floor method only if useful, for example:

```text
--floor-estimator robust-all
--floor-estimator ransac
```

Default should be chosen from actual JJ_Kabuto and synthetic tests.

Regardless of method, always compute residuals over ALL observed metric controls.

---

# 4. Re-anchor transforms after temporal smoothing

Current issue:

Smoothing `tx, ty, theta, log_scale` can move the selected reference frame away from exact identity.

After interpolation/smoothing and before canvas calculation:

1. reconstruct all smoothed matrices `M_t`;
2. obtain `M_ref`;
3. compute:

```python
A = np.linalg.inv(M_ref)
M_t = A @ M_t
```

for every frame.

For similarity/affine this remains in the same transform family up to floating-point tolerance.

Then verify:

```text
M_ref ≈ identity
```

to strict numerical tolerance.

This must happen BEFORE union/crop/original canvas translation is added.

After this operation:
- reference frame has exactly zero correction in stabilization coordinates;
- canvas translation is tracked separately.

Add a unit test that intentionally smooths a sequence where the reference would otherwise shift and verifies exact re-anchoring.

---

# 5. Separate transform stages in diagnostics

Current names such as `tx_raw` are ambiguous because the current "raw" timeline may already contain interpolation.

Refactor transform diagnostics into explicit stages.

Recommended columns:

```text
frame
reference_frame
solve_method
n_markers
n_manual_markers
n_imputed_markers

direct_tx
direct_ty
direct_rotation_deg
direct_scale

filled_tx
filled_ty
filled_rotation_deg
filled_scale

smoothed_tx
smoothed_ty
smoothed_rotation_deg
smoothed_scale

reanchored_tx
reanchored_ty
reanchored_rotation_deg
reanchored_scale

canvas_offset_x
canvas_offset_y

output_matrix_00
output_matrix_01
output_matrix_02
output_matrix_10
output_matrix_11
output_matrix_12
output_matrix_20
output_matrix_21
output_matrix_22

marker_rmse_px
anchor_rmse_px
floor_marker_rmse_px
is_interpolated
is_propagated
```

For direct values:
- use NaN when no direct visual solve existed.

For filled values:
- include interpolation/propagation.

For smoothed values:
- before re-anchoring.

For reanchored values:
- after forcing reference identity but before canvas translation.

For output matrix:
- exact matrix used to render/write coordinates, including constant canvas translation.

If maintaining old column names is important for compatibility, retain them as documented aliases rather than leaving ambiguous semantics.

---

# 6. Build a canonical reference atlas

Current limitation:

A marker only contributes if it exists in the selected reference frame.

This is unnecessary and prevents long videos from using static markers that become visible later.

Implement a canonical reference atlas.

## 6.1 Goal

Produce:

```text
reference_positions[marker_id] -> canonical 2D position
```

for as many static stabilization markers as possible, even if they were not visible in the chosen reference frame.

## 6.2 Bootstrap algorithm

Suggested deterministic approach:

1. Initialize atlas with markers visible in the chosen real reference frame.
2. Solve frames that share enough atlas markers.
3. Transform other observed static markers from those frames into canonical reference coordinates.
4. Accumulate multiple canonical observations per marker.
5. Robustly aggregate them, e.g. weighted median / robust mean.
6. Add newly established marker targets to the atlas.
7. Iterate until no more marker IDs can be added.
8. Require minimum support count / residual quality before accepting a new atlas point.

Do not create a synthetic reference image.

The selected reference frame remains a real frame; the atlas only extends the set of static target coordinates.

Add diagnostics:

```text
marker_id
atlas_source
atlas_support_frames
atlas_rmse_px
```

This should improve robustness when no single frame contains all static points.

Update motion diagnostics to use canonical atlas targets rather than returning unavailable statistics solely because a marker was absent in the reference frame.

---

# 7. Keep robust similarity estimation, but harden it

Do not replace the current weighted solver unless tests show a problem.

Add / verify:

- exact synthetic similarity recovery;
- weighted recovery;
- robust recovery with one gross click error;
- deterministic output;
- finite determinant;
- no reflection;
- sensible failure for coincident points.

For affine research mode:
- inspect current parameter decomposition;
- use a numerically meaningful polar/SVD decomposition if current rotation/scale/shear reporting is not correct for a general affine matrix;
- do not change visual default to affine.

---

# 8. Replace production `mp4v` encoding with repository FFmpeg/H.264 pipeline

Current production encoding through OpenCV `mp4v` should be replaced.

Goal:
- one high-quality production encode;
- H.264 by default using existing VAILA encoder selection logic;
- preserve frame count and FPS;
- then mux original audio without re-encoding video.

Inspect `vaila/ffmpeg_utils.py` first and reuse existing encoder selection/preferences.

Preferred architecture:

```text
OpenCV decode + warp
        ↓
raw frames / pipe
        ↓
FFmpeg H.264 encoder
        ↓
temporary stabilized video
        ↓
audio mux
        ↓
final MP4
```

Do not introduce unnecessary double lossy encoding.

Use existing VAILA hardware acceleration selection when safe:
- h264_nvenc
- VideoToolbox
- libx264 fallback
or whatever the repository currently supports.

Add a CLI/API quality control only if consistent with repo conventions, for example:

```text
--video-quality high
```

or reuse existing encoder defaults instead of inventing duplicate settings.

Debug overlay may use the same helper or a simpler codec if desired, but production output must use the high-quality path.

---

# 9. Harden audio/timestamp handling

Inspect the current audio mux logic critically.

Do not blindly seek audio using video `start_time` unless that is actually correct for the source stream layout.

Requirements:

- detect source audio stream;
- preserve audio when requested;
- use `-map` safely;
- tolerate no-audio input;
- use stream copy when compatible;
- AAC fallback if needed;
- no silent desynchronization;
- verify final duration.

After muxing, validate with ffprobe:

```text
output frame count
output fps
output duration
audio presence
audio start time
video start time
```

For CFR inputs, output frame count must equal source exactly.

For VFR inputs:
- detect VFR explicitly;
- do not silently claim original timestamps are preserved when rendering at average FPS;
- either preserve PTS properly or issue a clear warning / explicit resampling status.
- if full VFR preservation is outside this patch, document it honestly and add a regression check that the warning exists.

Add media validation results to summary JSON/report.

---

# 10. Improve media validation after rendering

After final MP4 creation, reopen/probe it.

Fail the run if production output is structurally invalid.

Validate at least:

```text
video opens
expected width/height
frame count
fps
duration tolerance
audio expected/present
no truncated final frame
```

Write useful fields into `stabilization_summary.json`, for example:

```json
"media_validation": {
  "frame_count_ok": true,
  "fps_ok": true,
  "duration_error_s": 0.001,
  "audio_ok": true,
  "video_codec": "h264"
}
```

The production video codec should normally report H.264.

---

# 11. Improve stabilization metrics

Keep before/after displacement, but add metrics that distinguish solver fit from smoothed output.

For each frame/group export or summarize:

```text
direct_fit_rmse_px
final_fit_rmse_px
anchor_direct_rmse_px
anchor_final_rmse_px
floor_direct_rmse_px
floor_final_rmse_px
```

For global summary report:

```text
all RMS before -> after
anchors RMS before -> after
floor RMS before -> after
all median before -> after
anchor median before -> after
direct solve percentage
interpolated percentage
propagated percentage
max absolute rotation correction
max scale deviation
mean black border fraction
reference frame
atlas marker coverage
hybrid imputed observation count
```

Do not evaluate stabilization quality using floor homography error alone.

---

# 12. Lens distortion: diagnostic only in v0.4.2

Do NOT turn this patch into a full camera-calibration project.

However, add a note / optional diagnostic to the report:

If floor residuals remain spatially systematic across many frames despite good manual points, report something like:

```text
Persistent systematic planar residuals may indicate lens distortion or imperfect target geometry.
Consider a future camera-intrinsics/distortion calibration step.
```

Do not estimate camera intrinsics/distortion in this patch unless the repository already has a proven reusable calibration module and integration is trivial.

Create a follow-up TODO rather than expanding scope.

---

# 13. Canvas and borders

Keep:
- union default;
- original;
- crop.

Re-anchoring must happen before canvas bounds.

Canvas translation must be constant over the whole output.

Add diagnostics:

```text
canvas_left
canvas_top
canvas_width
canvas_height
canvas_offset_x
canvas_offset_y
mean_black_border_fraction
max_black_border_fraction
```

Do not dynamically recenter per frame.

Do not auto-zoom unless explicitly implemented as a future opt-in mode.

---

# 14. GUI behavior

Preserve the existing thread-safe GUI architecture.

Add GUI controls only for new options that are useful to normal users.

At minimum, if hybrid imputation is implemented, expose:

```text
Hybrid imputed weight
```

but hide or de-emphasize advanced floor estimator controls if they would clutter the GUI.

GUI must continue:
- to print equivalent CLI;
- not touch Tk state from worker thread;
- to support cancel;
- to keep completed outputs;
- to open report/help;
- to avoid freezing during encoding.

---

# 15. Compatibility launcher

Keep `vaila_ground_stabilizer.py` as a very small compatibility wrapper.

Preserve:
- `--estimator global` -> `robust-lsq`
- `--reference-frame N` -> `--reference frame:N`

Only:
- bump version/date if repository conventions require it;
- add compatibility translations only when genuinely necessary.

Do not duplicate production stabilization code there.

---

# 16. Documentation updates

Update BOTH:

```text
vaila/help/video_stabilizer.md
vaila/help/video_stabilizer.html
```

Update help index if required.

The documentation must now explain that `hybrid` can use good floor geometry to provide **low-weight imputed missing floor controls**, while final visual rendering remains shape-preserving similarity/affine.

Explicitly state:

> The metric floor homography is never the default whole-frame visual warp.

Explain transform stages and CSV column meaning.

Update outputs and new floor metrics.

Document H.264/FFmpeg production encoding and audio validation.

Document reference atlas behavior.

Keep the tatame example:

```text
p0-p7 = metric floor
p8-p10 = pixel-only background anchors
```

---

# 17. Tests

Add targeted unit tests before relying on the real video.

## Similarity / affine core

- exact synthetic similarity recovery;
- weighted similarity recovery;
- robust outlier recovery;
- degenerate input rejection;
- no reflection;
- affine diagnostics if affine is retained.

## Temporal

- gap interpolation;
- leading/trailing propagation;
- theta unwrap;
- smoothing;
- exact re-anchor identity after smoothing.

## Reference atlas

- marker absent from real reference but visible later becomes usable;
- atlas robust aggregation;
- atlas does not accept a marker with insufficient support;
- diagnostics no longer depend on marker being visible in reference.

## Hybrid

- p8/p9/p10 never become metric unless TOML + metric selection explicitly define them;
- direct floor fit with all valid p0..p7;
- missing p0 is imputed only when floor quality is acceptable;
- imputed point receives lower weight;
- bad floor fit does not contribute imputed visual anchors;
- manual/imputed counts correct;
- visual output remains similarity, not projective;
- hybrid differs from visual when useful missing floor controls are supplied by metric geometry.

## Floor metrics

Synthetic planar data:
- known exact homography -> near-zero all/inlier residual;
- one corrupted click -> all-RMSE reflects corruption;
- robust estimator identifies/reduces influence;
- inlier ratio and max residual correct;
- degeneracy detected.

## Canvas

- reference reanchored before bounds;
- union includes all warped corners;
- crop fixed;
- original fixed;
- constant output dimensions.

## Media

If feasible:
- tiny synthetic video;
- output H.264;
- exact frame count;
- exact/acceptable FPS;
- with-audio source preserves audio;
- no-audio source succeeds;
- final ffprobe validation passes.

---

# 18. Real JJ_Kabuto regression

Run the real example if available.

Recommended visual test:

```bash
python -m vaila.video_stabilizer \
  --video JJ_Kabuto.mp4 \
  --markers tatame_markers.csv \
  --stabilization-markers 0-10 \
  --anchor-markers 8,9,10 \
  --mode visual \
  --model similarity \
  --estimator robust-lsq \
  --reference auto \
  --smooth savgol \
  --canvas union
```

Recommended hybrid test:

```bash
python -m vaila.video_stabilizer \
  --video JJ_Kabuto.mp4 \
  --markers tatame_markers.csv \
  --geometry-config vaila/models/planar_targets/tatame_1x1m.toml \
  --metric-markers 0-7 \
  --stabilization-markers 0-10 \
  --anchor-markers 8,9,10 \
  --hybrid-imputed-weight 0.25 \
  --mode hybrid \
  --model similarity \
  --estimator robust-lsq \
  --reference auto \
  --smooth savgol \
  --canvas union
```

Compare visual vs hybrid numerically.

Previously observed baseline behavior on this example was approximately:

```text
all static RMS:     ~77.7 px -> ~9.5 px
upper anchors RMS:  ~90.2 px -> ~10.1 px
331 frames
100% direct visual solves in that specific marked dataset
max rotation correction ~2.5 deg
max scale deviation ~14.6%
```

Treat these only as historical sanity checks, not immutable golden values.

The improved version must not regress grossly.

For this specific dataset, require at least:

- output frame count = source frame count;
- output H.264 video opens;
- p8-p10 after-RMS << before-RMS;
- all-marker after-RMS << before-RMS;
- reference transform is identity before canvas translation;
- no NaN/inf output transforms;
- final audio present if source has audio;
- floor metrics distinguish all-point vs inlier errors;
- hybrid never creates projective terms in final visual matrices.

If hybrid has no missing floor markers in a frame, it is acceptable for visual and hybrid transforms to be nearly identical there.

Do not force hybrid to differ merely for the sake of difference.

---

# 19. Report exact results after implementation

At the end, report:

1. files changed;
2. new functions/classes added;
3. old functions modified;
4. tests run;
5. test results;
6. visual JJ_Kabuto metrics;
7. hybrid JJ_Kabuto metrics;
8. output codec;
9. source/output frame count and FPS;
10. audio validation;
11. reference frame;
12. atlas coverage;
13. number of manual vs imputed correspondences;
14. floor fit statistics;
15. any remaining limitations.

Include explicit limitations:

- one global similarity cannot remove true multi-depth parallax caused by camera translation;
- floor homography is only geometrically exact on the floor plane;
- black borders represent genuinely uncaptured image area;
- VFR timestamp preservation status must be stated honestly;
- lens distortion is not fully solved in v0.4.2.

---

# 20. Scope guard

Do NOT:

- rewrite the application from scratch;
- replace the stable similarity solver without evidence;
- make homography the default visual transform;
- put p8-p10 into metric geometry automatically;
- overfit JJ_Kabuto with hard-coded marker IDs in core logic;
- add deep learning;
- add generative border filling;
- add camera intrinsics calibration unless trivially reusing an existing proven module;
- break the current CLI;
- break the GUI;
- remove the compatibility launcher;
- silently overwrite output directories;
- commit automatically.

The v0.4.2 goal is a **scientifically clearer, numerically safer, better encoded and genuinely useful hybrid stabilizer**, while preserving the current shape-preserving visual behavior.
