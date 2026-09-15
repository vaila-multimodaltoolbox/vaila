# Fixed-scene video stabilizer implementation plan

Version: 0.4.1 · 15 September 2026

1. Create `vaila/video_stabilizer.py` as the sole processing and GUI owner;
   replace the supplied `vaila_ground_stabilizer.py` prototype with a compatible
   launcher. Wire Video Stabilizer into Frame C, Video and Image, C_B_r2_c2.
   Create focused tests, module help (MD/HTML), button help and sample results.
2. Reuse `planar_geometry_tracker.load_measurements_csv`, `load_target_geometry`,
   `cv2_find_homography`, `FrameHomography` and `project_points`. Make its sports
   field export imports local to `write_outputs` so headless parsing has no GUI
   dependency. Reuse `numberframes.get_precise_video_metadata` and
   `ffmpeg_utils.get_ffmpeg_path`; the existing prototype audio helper needs
   explicit status and must not truncate frames with `-shortest`.
3. Validate zero-based CSV frame IDs against the entire video timeline; sparse
   rows become missing observations, never a shorter video. Preserve all marker
   IDs and NaNs. Choose a real reference, estimate transforms, regularize, compute
   one canvas, render, mux audio, export exact transforms/markers and report.
4. Default S is a float64 weighted similarity least-squares fit with centered
   coordinates, spatial-density balancing, anchor weight 2, robust Huber IRLS.
   Affine and RANSAC remain explicit alternatives. Never silently change model.
5. Visual IDs and priority anchors are pixel-only. Metric IDs intersect the TOML
   and CSV; only those feed direct RANSAC floor fits and floor diagnostics.
   Imputation is optional and omitted in this first implementation (counts = 0).
6. Estimate each usable frame against one fixed reference. Fill internal gaps
   by parameter interpolation, ends by nearest propagation. Unwrap rotation;
   smooth translation/angle/log-scale with short Savitzky-Golay or zero-phase
   lowpass. Export raw and final values and log each fallback. Floor-lock uses
   explicitly labeled planar transforms and nearest floor fallback.
7. Union maps all corners and adds one translation, rounding codec dimensions
   outward to even pixels. Original keeps source bounds. Crop finds a fixed
   centered rectangle inside the intersection of transformed source polygons.
8. Read oriented dimensions and precise FPS via the existing metadata helper.
   Render silent MP4 then mux source audio without shortening video; try copy
   then AAC. Report missing/disabled audio; fail explicitly if existing audio
   cannot be preserved. Outputs use a fresh directory and do not overwrite data.
9. Write frame transforms, transformed wide CSV, marker/group before-after RMS
   and median displacement, floor fits/inliers/condition flags, HTML tables and
   inline SVG plot, plus optional overlay video separate from production video.
10. Unit tests cover parser, missing frames, reference, exact/weighted/robust
    estimation, unwrap/interpolation, smoothing, canvas and marker consistency,
    hybrid separation, errors and GUI lifecycle. Run existing planar tests and
    real visual/hybrid runs on the supplied 331-frame portrait tatame with audio.
11. Parallax cannot be removed globally without shape deformation; report actual
    anchor and floor residuals instead of treating floor RMSE as visual success.
    Arbitrary variable-frame timing is represented at effective average FPS.

Repository differences: user names `vaila_ground_stabilizer.py`, spec names
`video_stabilizer.py`; keep both launch names with one implementation. Existing
planar solver chains and smooths homographies; reuse its direct solver for floor
diagnostics instead, never its fallback trajectory for visual stabilization.
Existing DLT2D uses normal equations, so do not reuse it for new numerical fits.
No new dependencies or automatic metric imputation are needed.
