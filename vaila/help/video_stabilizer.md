# video_stabilizer

**Category:** Tools → Video and Image
**Version:** 0.4.5
**Updated:** 24 September 2026
**Author:** Paulo R. P. Santiago
**GUI:** Yes — Video Stabilizer, between Compress Video and Make Sync file
**CLI:** `uv run vaila/video_stabilizer.py` (from repo root) or `python -m vaila.video_stabilizer`

## Description

Stabilize a video from physically fixed scene markers exported by getpixelvideo.
Metric coordinates are optional for visual stabilization. The default similarity
transform permits translation, rotation and uniform scale, preserving shape.
The optional visual homography model adds a full 8-DOF projective warp when
perspective motion remains after similarity or affine fitting.

Markers must belong to fixed objects, not moving people. Floor and wall points
may both contribute to visual stabilization. Parallax between different depths
can remain; a global similarity cannot eliminate it without deforming the image.

## GUI

Open **Video and Image → Video Stabilizer** (C_B_r2_c2). Select the video and
wide marker CSV. Leave the output field blank for a new method-named timestamped
directory beside the video, or select an output base path. Select your static marker IDs,
optionally specify priority anchors, then click **Stabilize** (Ctrl+Enter).
Click **Sweep all methods** to rank the full method grid and render its best
candidates; **Sweep render top** controls how many are encoded.
The equivalent CLI is printed before execution, with quoted paths.

Use **visual** without a TOML. Use **hybrid** with a geometry TOML and metric
marker subset to add floor diagnostics. **floor-lock** is an explicit planar-only
comparison and can distort people, walls and columns. **affine** allows shear
and independent axis scales; similarity remains recommended.

### `visual` vs `hybrid` vs `floor-lock`, in plain terms

All three modes stabilize the video the same way: they fit a shape-preserving
transform (`similarity` or `affine`) from your `--stabilization-markers` pixel
positions and warp every frame so the scene stops shaking/panning/zooming.
The difference is only about **extra metric (real-world) diagnostics**:

- **`visual`** (default) — pixel-only. No TOML needed. Use this if you just
  want a steady video and don't care about real-world distances. This is the
  right choice for most users.
- **`hybrid`** — does the exact same visual stabilization as `visual`, **plus**
  fits an independent floor homography (pixel ↔ real-world meters, from
  `--geometry-config`'s TOML and `--metric-markers`) per frame, purely for
  diagnostics. It writes two extra files, `floor_homographies.npz` and
  `floor_diagnostics.csv`, so you can check floor-fit quality (RMSE,
  condition number) or later convert floor-plane pixels to meters. The
  stabilized video itself looks the same as `visual` with the same
`--model`/`--canvas`/`--stabilization-markers`. Requires a TOML with the
  real-world positions of your metric markers (see the tatame example below:
  `vaila/models/planar_targets/tatame_1x1m.toml` defines the mat's known
  0.96 m module in real coordinates) and `--metric-markers` naming the CSV
  marker IDs that correspond to those known points.
- **`floor-lock`** — a different, more aggressive mode: instead of a single
  shape-preserving transform for the whole frame, it warps each frame with
  the *relative floor homography itself*, so the floor plane stays pixel-exact
  across frames. This deliberately distorts anything not on the floor plane
  (people, walls, columns) because it is forcing a different geometric
  constraint. Only use it if you specifically need the floor pixel-locked for
  planar measurement work and accept that people/walls will look warped.

If you don't have a calibrated TOML for your scene, or don't need metric
floor diagnostics, use `visual` — it's simpler and requires fewer inputs.
`hybrid` only makes sense once you have a geometry TOML for that specific
camera setup (see [planar_geometry_tracker](planar_geometry_tracker.md) for
how to build one).

The resizable child window stays above its parent. Encoding runs in one worker;
progress reaches Tk through a queue. **Cancel** or closing the window requests
cancellation at frame boundaries; during audio mux the current step finishes
first. Completed files remain available. A cancelled run may contain incomplete
diagnostic files; retry with the same base path and the interface creates a new
method-suffixed directory. **Help** and
**Open report** open the local HTML pages in your browser.

## Tatame example

The supplied `tests/video_stabilizer/tatame.mp4` contains 331 portrait frames with
audio. `tatame_markers.csv` contains floor points p0–p7 and six fully observed
upper/background points p8–p13. Frame IDs are zero-based and refer to decoded
source frames.

```bash
uv run vaila/video_stabilizer.py \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --stabilization-markers all --anchor-markers 8-13 \
  --mode visual --model homography --canvas union

uv run vaila/video_stabilizer.py \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --geometry-config vaila/models/planar_targets/tatame_1x1m.toml \
  --metric-markers 0-7 --sweep
```

Both commands run from the repo root. `python -m vaila.video_stabilizer ...` (same
flags) works identically for users who prefer package-module invocation.

The full 866-combination sweep on this updated fixture ranked
`visual-homography-rlsq-savgol-refauto-mkall-anc8-13w4` first. Its marker RMS
was 3.43 px at the top, 6.27 px in the middle and 6.28 px at the bottom. The
old p0–p7 similarity baseline measured 3.78 px at the bottom but had no top or
middle marker coverage, so it cannot support a whole-frame stability claim.
Use the montage for the final visual choice; these figures are reproducible
marker diagnostics, not a pixel-domain image-quality score.

The TOML defines the interlocked 0.96 m floor module. Background points p8–p13
never enter the floor fit unless they are explicitly defined as metric controls
in both the TOML and metric selection. The current implementation uses manual
CSV observations only; it does not synthesize/impute floor points.

`python -m vaila.vaila_ground_stabilizer` is a compatibility launcher for the
same implementation. It also translates legacy `--estimator global` and
`--reference-frame N`. Use `--output-dir`, not the prototype's `--output`.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--video`, `--markers` | required in CLI | Input video and wide `frame,pN_x,pN_y,...` CSV. |
| `--output-dir` | new method-named timestamped directory | CLI/GUI append the deterministic method slug, then `_v2`, `_v3`, ... if needed. Direct Python callers retain strict paths and refuse non-empty directories. |
| `--mode` | `visual` | `visual`, `hybrid`, `floor-lock`. Last two require geometry. |
| `--model` | `similarity` | `similarity`, `affine`, or 8-DOF projective `homography`. |
| `--estimator` | `robust-lsq` | Spatially balanced weighted fit with Huber reweighting, or `ransac`. |
| `--stabilization-markers` | `all` | Static pixel markers; ranges such as `0-7,10,12-14`. |
| `--anchor-markers` | none | Subset of static markers given priority and extra weight. |
| `--anchor-weight` | `2.0` | Positive weight multiplier for anchors. |
| `--hybrid-imputed-weight` | `0.25` | Weight for trusted floor-reconstructed controls in hybrid mode. |
| `--metric-markers` | CSV ∩ TOML | Metric subset; unknown world IDs are excluded from floor fitting. |
| `--geometry-config` | none | Existing planar target TOML. Also enables diagnostics in visual mode. |
| `--floor-estimator` | `robust-all` | Floor fit using all controls with robust weights, or `ransac`. |
| `--reference` | `auto` | Real central frame; `first` means frame 0, `frame:N` selects explicitly. |
| `--smooth` | `savgol` | `none`, short centered Savitzky-Golay (up to 7 frames), or zero-phase `lowpass` (up to 6 Hz). |
| `--canvas` | `union` | Union of all warped corners; `original` keeps source extent; `crop` uses a fixed centered rectangle inside their intersection. |
| `--border` | `black` | Black empty borders; no filling or automatic zoom. |
| `--debug-overlay` | off | Separate side-by-side source/stabilized video with marker IDs, reference targets and residuals. |
| `--no-audio` | off | Explicitly disable audio preservation. |
| `--sweep` | off | Triage the full method grid without video encoding, rank it, then render the best candidates. |
| `--sweep-grid` | built-in grid | TOML arrays overriding any sweep axis. |
| `--sweep-render-top` | `5` | Number of ranked candidates to render, or `all`. |

No arguments opens the GUI. `--help` prints all flags. CLI success exits 0;
invalid inputs, encoder/audio failures and cancellation return nonzero.

### Flags, one at a time (plain language)

- **`--video`** — path to the shaky source video.
- **`--markers`** — the wide CSV from getpixelvideo: one row per frame, columns
  `pN_x`, `pN_y` for each marker ID `N`. These are the fixed points the
  algorithm tracks to figure out how the camera moved.
- **`--output-dir`** — where results go. Leave it **blank** to auto-create a
  fresh timestamped folder next to the video (`vaila_stabilized_<method>_<timestamp>`)
  — the safe default. If you pass a path yourself and it already has files in
  it (e.g. rerunning with the same `--output-dir`), the CLI and GUI
  append a method slug and automatically pick `_v2`, `_v3`, ... instead of overwriting — you
  never need to pick a fresh empty folder by hand. (Programmatic callers of
  `run_video_stabilizer()` directly still get the strict original behavior:
  `FileExistsError` if the directory isn't empty — pointing `--output-dir` at
  the same folder as `--video`/`--markers` is still a bad idea, since either
  path leaves your source files where the tool has to reason about them.)
- **`--mode`** — `visual`, `hybrid`, or `floor-lock`; see the comparison above.
- **`--model`** — the shape of transform fit per frame. `similarity` (default)
  only allows translation + rotation + uniform zoom, so people/objects keep
  their true proportions. `affine` also allows shear and different
  horizontal/vertical scale — more flexible, but can subtly distort shapes;
  `homography` also models perspective with eight degrees of freedom. It can
  stabilize projective camera motion but may deform content when the fixed
  points lie at different depths.
- **`--estimator`** — how the per-frame transform is fit from your marker
  correspondences. `robust-lsq` (default) uses every marker but automatically
  down-weights ones that look noisy/inconsistent that frame. `ransac`
  instead throws out correspondences it judges as outliers entirely — more
  aggressive, and empirically worse when markers are sparse/occluded (see the
  tatame finding below).
- **`--stabilization-markers`** — which marker IDs (from the CSV) to use for
  the visual fit. `all` (default), or a list/range like `0-7,10,12-14`.
- **`--anchor-markers`** — a subset of the above you trust more (e.g. rock-solid
  background points), given extra weight (`--anchor-weight`, default `2.0`)
  in the fit. Optional.
- **`--hybrid-imputed-weight`** — relative weight for missing metric controls
  reconstructed from a trusted floor fit; the default 0.25 keeps manual clicks
  dominant.
- **`--metric-markers`** — only used in `hybrid`/`floor-lock`: the marker IDs
  that also have known real-world coordinates in `--geometry-config`'s TOML,
  used for the floor homography fit. Ignored in `visual` mode.
- **`--geometry-config`** — path to a TOML describing real-world (metric)
  positions for a subset of markers on a flat plane (e.g. a floor mat with
  known dimensions). Required for `hybrid`/`floor-lock`; optional in `visual`
  (adds diagnostics only, no effect on the stabilized video).
- **`--floor-estimator`** — `robust-all` keeps declared controls and reweights
  noisy clicks; `ransac` can reject controls as outliers.
- **`--reference`** — which frame's marker positions become the fixed target
  every other frame is warped toward. `auto` (default) picks a real frame
  with good marker visibility and spread near the temporal middle; `first`
  forces frame 0; `frame:N` picks an exact frame index yourself.
- **`--smooth`** — light temporal smoothing of the fitted transform parameters
  before warping, to reduce frame-to-frame jitter beyond what the marker fit
  alone gives. `savgol` (default), `lowpass`, or `none`. Differences are
  usually small; try `none` only if smoothing seems to be lagging fast motion.
- **`--canvas`** — how big the output frame is. `union` (default) grows the
  canvas to fit every warped frame's corners so nothing gets cropped;
  `original` keeps the source video's exact size (may crop warped edges);
  `crop` picks a fixed centered rectangle guaranteed to stay inside every
  frame (no black borders, but a smaller field of view).
- **`--border`** — border fill for empty canvas area exposed by warping.
  Currently always `black`; no auto-zoom or content-aware fill.
- **`--debug-overlay`** — also render a separate side-by-side diagnostic video
  showing marker IDs, the reference target positions, and residuals, useful
  for judging fit quality visually.
- **`--no-audio`** — skip copying the source audio track into the output.
- **`--sweep`** — evaluates 432 visual combinations without geometry, or 866
  combinations with geometry (432 visual, 432 hybrid, two floor-lock fits).
  It then renders only the requested top candidates.
- **`--sweep-grid`** — TOML overrides. Put arrays under `[sweep]`; supported
  keys are `modes`, `models`, `estimators`, `smooth`, `references`,
  `stabilization_markers`, `anchors` (for example `"8-13@4.0"`),
  `hybrid_imputed_weights`, and `floor_estimators`.
- **`--sweep-render-top`** — positive count or `all`.

## Geometry and missing observations

Transforms are estimated against one real reference, never chained frame to
frame. Auto reference prioritizes anchor visibility, total marker count, broad
spatial support and proximity to the robust temporal median configuration.
That median is only a selection criterion; the reference uses actual coordinates.

At least two non-coincident correspondences are needed for similarity, three
non-collinear for affine and four non-collinear for homography. Missing coordinates/rows are allowed; internal gaps
are interpolated in translation, unwrapped angle and log-scale. Ends use nearest
valid parameters. Every fallback is logged. Sparse rows never shorten the video.
Duplicate, fractional, negative or out-of-video frame IDs are rejected.

Floor diagnostics use independent normalized OpenCV RANSAC homographies from
real XY to pixels. Insufficient floor observations produce NaN floor matrices
and an explicit unavailable status, while visual processing continues. No
projected coordinates are clamped. Floor-lock uses relative floor homographies
with nearest-fit propagation, is unsmoothed, and is always explicitly labeled.

Region diagnostics split canonical marker Y coordinates into equal top, middle
and bottom thirds of the source frame. They measure whether digitized fixed
points hold still. Every marker in the CSV counts, including markers left out of
`--stabilization-markers`; those also appear as the `held_out` row, so a subset
fit cannot look better by ignoring off-plane points (a floor-only homography
can move a wall 30 px). A region without any marker is unmeasured and is
skipped: the worst-region score uses the covered regions only. The metric cannot
see pixel motion between markers. Runs rank by the largest regional marker
RMS, then the regional mean and scale deviation. The sweep skips grid values
whose marker or anchor IDs are absent from the CSV and validates
`--sweep-render-top` before triage. Visual
inspection of the montage remains the final selection step.

## Outputs

| File | Contents |
| --- | --- |
| `<stem>_stabilized.mp4` | Production video; no overlay. Every source frame, constant canvas, effective source FPS and displayed orientation. |
| `stabilization_transforms.csv` | One row per frame: reference, method, observed/imputed counts, raw and smoothed parameters, exact final 3×3 video matrix including canvas translation, residuals, fallback flag. |
| `stabilized_markers.csv` | Every valid original pixel marker transformed by the exact final video matrix; same IDs and missing values. |
| `stabilization_diagnostics.csv` | Per-marker and all/anchor/floor group RMS and median displacement before/after, relative to the real reference. |
| `stabilization_report.html` | Summary, SVG comparison chart, marker table, canvas, fallback proportions, scale/rotation limits, border estimate and audio result. |
| `stabilization_summary.json` | Machine-readable run settings and aggregate information. |
| `floor_homographies.npz` | With geometry: `H` (world→pixel), `frame_ids`, `metric_ids`; unavailable fits are NaN. |
| `floor_diagnostics.csv` | With geometry: correspondence/inlier counts, all-observation floor RMSE, condition number, degeneracy flag and method. |
| `debug_overlay.mp4` | Optional diagnostic video, separate from the production output. |

A sweep writes `sweep_ranking.csv`, `sweep_report.html`,
`sweep_best_command.txt`, `sweep_montage.mp4`, no-video artifacts under
`triage/`, full artifacts under `renders/`, and rank-prefixed MP4 files in one
flat `videos/` directory.

`tx_raw`, `ty_raw` are before canvas translation; final matrix/tx/ty include it.
Similarity has exactly `[0,0,1]` as its last matrix row. Imputed counts are zero.
For floor-lock, scalar similarity parameters are not applicable; use the full
matrix. Affine matrices additionally carry anisotropic scale/shear.
Displacement for a marker absent in the reference is unavailable, not invented.
The border fraction is a geometric area estimate, independent of dark content.
`union` never resizes the source image: it expands the canvas as needed and
fills genuinely uncaptured areas with black.

Audio is copied from the source when possible, with AAC fallback if required,
aligned to the source video start. No-audio input is valid. An audio preservation
failure is reported as a failed run rather than silently discarding audio.
FFmpeg/ffprobe and OpenCV are existing vailá dependencies. Variable-frame-rate
sources are represented at their effective average FPS; individual variable
timestamps are not reproduced. Encoded FPS may differ by at most codec rounding.

## Downstream Multimodal AI Workflow

Video stabilization is the foundational first stage of the vailá multimodal field-calibration and 3D capture pipeline:

1. **Stage 1: Video Stabilization (`video_stabilizer.py`)** — produces `<stem>_stabilized.mp4` and `stabilized_markers.csv`.
2. **Stage 2: Planar Target Geometry Calibration (`planar_geometry_tracker.py`)** — uses `stabilized_markers.csv` and a target TOML (e.g. `tatame_1x1m.toml`) to fit per-frame DLT2D, impute occluded points while preserving square shape and collinearity, and output `geometry_animation.html` and `debug_projected_wireframe.mp4`.
3. **Stage 3: Markerless 3D Mesh & Keypoints (`sam3dinov3.py`)** — processes `<stem>_stabilized.mp4` through SAM 3 identity tracking and SAM 3D Body (DINOv3 backbone) to regress metric MHR70 3D joints and 36k-face human meshes (`meshes/*.npz`). Stabilizing the video first eliminates camera ego-motion jitter from silhouettes and joint angles.
4. **Stage 4: World Calibration Alignment (`monocular_dlt_align.py`)** — aligns camera-relative 3D pose onto the calibrated floor coordinate system (meters).

For the complete end-to-end tutorial with reproducible benchmarks and installation instructions, see **[`tests/video_stabilizer/README_stabilizer.md`](../../tests/video_stabilizer/README_stabilizer.md)**.
See also [planar_geometry_tracker](planar_geometry_tracker.md) and [sam3dinov3](sam3dinov3.md).
