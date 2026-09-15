# video_stabilizer

**Category:** Tools → Video and Image
**Version:** 0.4.2
**Updated:** 15 September 2026
**Author:** Paulo R. P. Santiago
**GUI:** Yes — Video Stabilizer, between Compress Video and Make Sync file
**CLI:** `uv run vaila/video_stabilizer.py` (from repo root) or `python -m vaila.video_stabilizer`

## Description

Stabilize a video from physically fixed scene markers exported by getpixelvideo.
Metric coordinates are optional for visual stabilization. The default similarity
transform permits translation, rotation and uniform scale, preserving shape.
A planar homography is valid for the calibrated plane and is not used as the
default whole-frame video warp.

Markers must belong to fixed objects, not moving people. Floor and wall points
may both contribute to visual stabilization. Parallax between different depths
can remain; a global similarity cannot eliminate it without deforming the image.

## GUI

Open **Video and Image → Video Stabilizer** (C_B_r2_c2). Select the video and
wide marker CSV. Leave the output field blank for a new timestamped directory
beside the video, or select an empty directory. Select your static marker IDs,
optionally specify priority anchors, then click **Stabilize** (Ctrl+Enter).
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
diagnostic files; use a new/empty output directory to retry. **Help** and
**Open report** open the local HTML pages in your browser.

## Tatame example

The supplied `tests/video_stabilizer/tatame.mp4` contains 331 portrait frames with
audio. `tatame_markers.csv` contains floor points p0–p7 and upper-background
anchors p8–p10. Frame IDs are zero-based and refer to decoded source frames.

```bash
uv run vaila/video_stabilizer.py \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --stabilization-markers 0-10 --anchor-markers 8,9,10 \
  --mode visual --model similarity --canvas union

uv run vaila/video_stabilizer.py \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --geometry-config vaila/models/planar_targets/tatame_1x1m.toml \
  --metric-markers 0-7 --stabilization-markers 0-10 --anchor-markers 8,9,10 \
  --mode hybrid --model similarity --canvas union
```

Both commands run from the repo root. `python -m vaila.video_stabilizer ...` (same
flags) works identically for users who prefer package-module invocation.

For this specific clip, empirically, `--model affine` roughly halves the
stabilization residual RMS versus the default `similarity` (about 8.6 px to
4.0 px "all" RMS) — the tatame camera has some non-rigid scale/shear drift.
Try `--model affine` first on this footage; `similarity` remains the
recommended default for arbitrary videos. `--estimator ransac` performed
markedly worse here (about 20.6 px RMS) given the sparse/occluded marker set —
keep the default `robust-lsq`.

The TOML defines the interlocked 0.96 m floor module. Background points p8–p10
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
| `--output-dir` | new `vaila_stabilized_<timestamp>` | Output directory. If it already has files, CLI/GUI runs auto-suffix `_v2`, `_v3`, ... instead of overwriting. |
| `--mode` | `visual` | `visual`, `hybrid`, `floor-lock`. Last two require geometry. |
| `--model` | `similarity` | Similarity or explicit research `affine`. |
| `--estimator` | `robust-lsq` | Spatially balanced weighted fit with Huber reweighting, or `ransac`. |
| `--stabilization-markers` | `all` | Static pixel markers; ranges such as `0-7,10,12-14`. |
| `--anchor-markers` | none | Subset of static markers given priority and extra weight. |
| `--anchor-weight` | `2.0` | Positive weight multiplier for anchors. |
| `--metric-markers` | CSV ∩ TOML | Metric subset; unknown world IDs are excluded from floor fitting. |
| `--geometry-config` | none | Existing planar target TOML. Also enables diagnostics in visual mode. |
| `--reference` | `auto` | Real central frame; `first` means frame 0, `frame:N` selects explicitly. |
| `--smooth` | `savgol` | `none`, short centered Savitzky-Golay (up to 7 frames), or zero-phase `lowpass` (up to 6 Hz). |
| `--canvas` | `union` | Union of all warped corners; `original` keeps source extent; `crop` uses a fixed centered rectangle inside their intersection. |
| `--border` | `black` | Black empty borders; no filling or automatic zoom. |
| `--debug-overlay` | off | Separate side-by-side source/stabilized video with marker IDs, reference targets and residuals. |
| `--no-audio` | off | Explicitly disable audio preservation. |

No arguments opens the GUI. `--help` prints all flags. CLI success exits 0;
invalid inputs, encoder/audio failures and cancellation return nonzero.

### Flags, one at a time (plain language)

- **`--video`** — path to the shaky source video.
- **`--markers`** — the wide CSV from getpixelvideo: one row per frame, columns
  `pN_x`, `pN_y` for each marker ID `N`. These are the fixed points the
  algorithm tracks to figure out how the camera moved.
- **`--output-dir`** — where results go. Leave it **blank** to auto-create a
  fresh timestamped folder next to the video (`vaila_stabilized_<timestamp>`)
  — the safe default. If you pass a path yourself and it already has files in
  it (e.g. rerunning with the same `--output-dir`), the CLI and GUI
  automatically pick `<dir>_v2`, `<dir>_v3`, ... instead of overwriting — you
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
  use it only if `similarity` residuals look poor for your specific footage.
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
- **`--metric-markers`** — only used in `hybrid`/`floor-lock`: the marker IDs
  that also have known real-world coordinates in `--geometry-config`'s TOML,
  used for the floor homography fit. Ignored in `visual` mode.
- **`--geometry-config`** — path to a TOML describing real-world (metric)
  positions for a subset of markers on a flat plane (e.g. a floor mat with
  known dimensions). Required for `hybrid`/`floor-lock`; optional in `visual`
  (adds diagnostics only, no effect on the stabilized video).
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

## Geometry and missing observations

Transforms are estimated against one real reference, never chained frame to
frame. Auto reference prioritizes anchor visibility, total marker count, broad
spatial support and proximity to the robust temporal median configuration.
That median is only a selection criterion; the reference uses actual coordinates.

At least two non-coincident correspondences are needed for similarity (three
non-collinear for affine). Missing coordinates/rows are allowed; internal gaps
are interpolated in translation, unwrapped angle and log-scale. Ends use nearest
valid parameters. Every fallback is logged. Sparse rows never shorten the video.
Duplicate, fractional, negative or out-of-video frame IDs are rejected.

Floor diagnostics use independent normalized OpenCV RANSAC homographies from
real XY to pixels. Insufficient floor observations produce NaN floor matrices
and an explicit unavailable status, while visual processing continues. No
projected coordinates are clamped. Floor-lock uses relative floor homographies
with nearest-fit propagation, is unsmoothed, and is always explicitly labeled.

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

`tx_raw`, `ty_raw` are before canvas translation; final matrix/tx/ty include it.
Similarity has exactly `[0,0,1]` as its last matrix row. Imputed counts are zero.
For floor-lock, scalar similarity parameters are not applicable; use the full
matrix. Affine matrices additionally carry anisotropic scale/shear.
Displacement for a marker absent in the reference is unavailable, not invented.
The border fraction is a geometric area estimate, independent of dark content.

Audio is copied from the source when possible, with AAC fallback if required,
aligned to the source video start. No-audio input is valid. An audio preservation
failure is reported as a failed run rather than silently discarding audio.
FFmpeg/ffprobe and OpenCV are existing vailá dependencies. Variable-frame-rate
sources are represented at their effective average FPS; individual variable
timestamps are not reproduced. Encoded FPS may differ by at most codec rounding.

See [planar_geometry_tracker](planar_geometry_tracker.md) for metric plane
projection and extrapolation; use its REF3D/DLT outputs for metric reconstruction.
