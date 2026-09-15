# Video stabilizer v0.4.2 implementation plan

Version: 0.4.2 · 15 September 2026

Required audit-and-plan step from `codex_prompt_video_stabilizer_v0_4_2.md` §1.
This document records the audited state of the code *before* editing, the exact
functions that will change, the helpers and tests that will be added, and the
behaviour/CLI compatibility risks.

## Audit: what was inspected

- `vaila/video_stabilizer.py` (1134 lines, v0.4.1, 15 September 2026) in full,
  including `StabilizerGUI`.
- `vaila/vaila_ground_stabilizer.py` (34 lines) — compatibility launcher.
- `vaila/ffmpeg_utils.py` encoder API.
- `vaila/planar_geometry_tracker.py` re-exported helpers
  (`cv2_find_homography`, `has_non_collinear_quad`, `project_points`,
  `load_measurements_csv`, `load_target_geometry`, `FrameHomography`).
- `tests/test_video_stabilizer.py` (552 lines, 27 tests).
- `vaila/help/video_stabilizer.md` / `.html`.
- `vaila/models/planar_targets/tatame_1x1m.toml` (8 metric points, p0–p7).
- `tests/video_stabilizer/tatame_markers.csv` and the user-supplied
  `tests/video_stabilizer/processed_skip_butterworth_cut6_0_20260915_010603/tatame_markers_butterworth.csv`.

Baseline runs before any edit:

```text
uv run python -m py_compile vaila/video_stabilizer.py vaila/vaila_ground_stabilizer.py vaila/planar_geometry_tracker.py   # OK
uv run pytest tests/test_video_stabilizer.py -q                                                                          # 25 passed, 2 skipped
```

The two skips are `test_real_tatame_visual_and_hybrid_regression` (its asset
paths point at `tests/interp_geometry/`, which no longer exists — the real
assets now live in `tests/video_stabilizer/`) and `test_gui_display_smoke`
(needs `VAILA_GUI_SMOKE=1` and a display).

## What currently works and must be preserved

1. Visual stabilization and planar metric calibration are already separate
   concepts. The default warp is a shape-preserving 2D similarity and no
   projective term ever reaches the rendered video.
2. `fit_weighted_transform` / `spatial_weights` / `estimate_transform`:
   float64 centred weighted least squares with spatial density balancing,
   anchor weighting and Huber IRLS. Exact on synthetic similarity data and
   already covered by tests. It is not replaced.
3. `matrix_to_params` / `params_to_matrix` form an exact RQ decomposition
   (`theta = atan2(m10, m00)`, `scale = hypot(m00, m10)`,
   `triangular = R.T @ A` has `triangular[1,0] == 0` algebraically) and round
   trip for both `similarity` and `affine`. No rewrite; a proving test is added.
4. Estimation is against one fixed canonical reference frame, never
   frame-to-frame chaining, so there is no drift accumulation.
5. `regularize_transforms` interpolates gaps in physical parameter space,
   unwraps rotation, propagates ends and smooths with Savitzky-Golay or a
   zero-phase lowpass. Homography coefficients are never smoothed.
6. `compute_canvas` union/original/crop plus one constant canvas translation.
7. `solve_floor` rejects non-finite, rank-deficient, ill-conditioned and
   degenerate (collinear) configurations and writes NaN into the NPZ for them.
8. CSV inputs are never modified; outputs go to a caller-chosen directory.
9. The debug overlay is a separate file from the production video.
10. `StabilizerGUI` is thread-safe: the worker thread only pushes onto a
    `queue.Queue`, and `drain()` on the Tk main loop is the only writer of Tk
    state. Cancel, Help, Open report and `format_cli_command` all work.
11. `vaila_ground_stabilizer.py` is a thin wrapper that only rewrites
    `--estimator global` and `--reference-frame N`.

## What is incomplete

1. **`hybrid` is inert.** There is no `mode == "hybrid"` branch anywhere; the
   only `mode` tests are for canvas modes, geometry-config validation and
   `floor-lock`. The visual solve in `hybrid` is byte-identical to `visual`
   (a current test even asserts `allclose(visual, hybrid, atol=1e-10)`), and
   floor data only reaches `floor_homographies.npz` / `floor_diagnostics.csv`.
   `n_imputed_markers` is hard-coded to `0`.
2. **No re-anchoring after smoothing.** This is a real metric bug, not only a
   cosmetic one: `motion_diagnostics` computes
   `target = project_points(canvas_translation, table.xy[reference_frame])`,
   which is the true post-warp reference position *only* when `M_ref == I`.
   Savitzky-Golay moves `M_ref` off identity, so every `rms_after_px` is biased.
3. **Floor diagnostics conflate all-point and inlier error.** `solve_floor`
   exports `floor_rmse_px` computed over *all* correspondences while the fit is
   RANSAC, so a fit that nails four points and misses the rest looks good.
   There is no inlier RMSE, inlier ratio, median, max residual or quality state.
4. **Ambiguous transform-stage names.** `tx_raw` is already post-interpolation,
   so "raw" hides both the direct solve and the filled stage.
5. **No reference atlas.** `valid` requires finiteness in the reference frame,
   so a static marker absent from that one frame contributes nothing, ever.
   With the user's Butterworth CSV this matters a lot: `p0` is missing on
   221/331 frames and `p6` on 183/331.
6. **Production encode is OpenCV `mp4v`**, not H.264, and does not use the
   repository's encoder selection.
7. **Audio mux is fragile.** `_finish_video` places `-ss <video start_time>`
   before the *audio* input and truncates with `-t nb_frames/fps`.
8. **No post-render validation.** Nothing reopens the final MP4.
9. **No direct-vs-final fit metrics**, no canvas geometry diagnostics beyond
   `mean_black_border_fraction`, no VFR honesty statement.

## Functions that will change

| Function | Change |
| --- | --- |
| `solve_floor` | float64 normalized fit, optional `robust-all` IRLS estimator, residuals over **all** observed controls plus inlier subset, new metric columns, `fit_quality` state |
| `estimate_transform` / visual solve loop in `run_video_stabilizer` | accept imputed correspondences with a separate low weight; record manual vs imputed counts per frame |
| `regularize_transforms` | return the smoothed stage separately so `direct`/`filled`/`smoothed` can all be exported |
| `run_video_stabilizer` | insert re-anchoring before `compute_canvas`; wire atlas, imputation, new diagnostics, H.264 encode, media validation |
| `motion_diagnostics` | use atlas canonical targets, report direct-fit and final-fit RMSE separately, and rely on the re-anchored reference |
| `_finish_video` | ffprobe-driven audio detection, no blind `-ss`, no `-t` truncation, copy-then-AAC fallback |
| `_write_report` | new sections: transform stages, floor quality, atlas coverage, media validation, canvas geometry |
| `build_parser` / `_run_args` / `format_cli_command` | `--hybrid-imputed-weight`, `--floor-estimator`, `--video-quality` |
| `StabilizerGUI.__init__` | one new field, `Hybrid imputed weight` |
| `vaila_ground_stabilizer.py` | version/date header only |

## New helpers

- `build_reference_atlas(table, ids, reference_frame, ...)` — deterministic
  iterative bootstrap returning canonical positions plus
  `marker_id / atlas_source / atlas_support_frames / atlas_rmse_px`.
- `floor_fit_quality(...)` — `good` / `usable` / `poor` / `unavailable`.
- `fit_floor_homography(src, dst, estimator)` — normalized DLT + IRLS branch.
- `impute_floor_markers(...)` — projects missing metric controls through a
  trusted `H_floor,t`, with short-gap interpolation of the projected **pixel**
  positions (never of homography coefficients).
- `reanchor_transforms(matrices, reference_frame)` — `A = inv(M_ref)`,
  `M_t <- A @ M_t`, assert `M_ref ≈ I`.
- `encode_frames_h264(...)` — writes the production video through
  `ffmpeg_utils.run_ffmpeg_encode_with_fallback` from a rawvideo pipe.
- `validate_output_media(...)` — reopen/probe and build `media_validation`.

## New tests

Added to `tests/test_video_stabilizer.py`: parameter round trip for a general
affine matrix; no-reflection and coincident-point failure; exact re-anchor
identity after smoothing; atlas admits a marker absent from the reference frame
and rejects one with insufficient support; floor metrics on synthetic planar
data with one corrupted click (all-RMSE reflects it, inlier RMSE does not);
imputation happens only when floor quality is acceptable and receives the lower
weight; `p8`–`p10` never become metric; hybrid output stays a similarity;
H.264 codec, exact frame count and audio presence on a tiny synthetic clip;
no-audio source succeeds. The stale `tests/interp_geometry/` regression paths
are repointed at `tests/video_stabilizer/`.

## Compatibility risks and how they are handled

1. **Renamed transform CSV columns.** Mitigation: the new staged columns are
   added and the old names (`tx_raw`, `ty_raw`, `rotation_deg_raw`,
   `scale_raw`, `tx`, `ty`, `rotation_deg`, `scale`, `matrix_ij`) are kept as
   documented aliases, so existing readers and `test_explicit_alternatives` /
   the real-artifact test keep working.
2. **`hybrid` will no longer be bit-identical to `visual`** whenever a trusted
   floor fit supplies a missing control. The existing assertion
   `assert_allclose(visual, hybrid, atol=1e-10)` is therefore replaced by an
   assertion that hybrid stays a *similarity* with no projective terms and is
   identical only on frames with no imputable control.
3. **Re-anchoring changes the output matrices** (by a constant
   `inv(M_ref)`) and therefore also `stabilized_markers.csv` and the reported
   `rms_after_px`. This is the bug fix; the README numbers are re-measured.
4. **Encoder change.** `libx264` CPU fallback is always available through
   `run_ffmpeg_encode_with_fallback`, and hardware failures already fall back
   there, so no new hard dependency is introduced.
5. **CLI additions only.** No flag is removed or renamed; the launcher
   translations are untouched.
6. **Global version.** The user named v0.4.2 and CLAUDE.md requires touched
   scripts to carry the global vailá version, so `vaila.py` (header + two
   banner strings) moves 0.4.1 → 0.4.2 together with the stabilizer, launcher,
   help pages and help index.

## Out of scope (spec §20)

No rewrite, no replacement of the similarity solver, homography never becomes
the default visual warp, p8–p10 never enter metric geometry automatically, no
marker IDs hard-coded in core logic, no deep learning, no generative border
fill, no intrinsics calibration, no CLI/GUI break, launcher kept, no automatic
commit.
