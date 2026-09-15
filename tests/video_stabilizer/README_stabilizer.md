# Tatame fixed-scene stabilization example

Version: 0.4.1 · Updated: 15 September 2026.

Inputs: `tatame.mp4` (331 frames, portrait 1080×1920, effective
59.940060001 FPS, AAC audio) and `tatame_markers.csv` (zero-based wide CSV).
Use p0–p7 as floor controls and p8–p10 as pixel-only background anchors.
Source files are not modified. Generated results are gitignored under
`stabilizer_results/`.

Run from the repository root into fresh/empty output directories:

```bash
.venv/bin/python -m vaila.video_stabilizer \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --stabilization-markers 0-10 --anchor-markers 8,9,10 \
  --mode visual --model similarity --canvas union --debug-overlay \
  --output-dir tests/video_stabilizer/stabilizer_results/visual

.venv/bin/python -m vaila.video_stabilizer \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --stabilization-markers 0-10 --anchor-markers 8,9,10 --metric-markers 0-7 \
  --geometry-config vaila/models/planar_targets/tatame_1x1m.toml \
  --mode hybrid --model similarity --canvas union \
  --output-dir tests/video_stabilizer/stabilizer_results/hybrid
```

## Measured result

Both modes choose real frame 127 and produce the same visual matrices.

| Static markers | RMS before (px) | RMS after (px) |
| --- | ---: | ---: |
| All selected markers | 77.656 | 9.484 |
| Background anchors p8–p10 | 90.244 | 10.109 |
| Floor p0–p7 (hybrid diagnostics) | 70.733 | 9.165 |
| p8 | 74.583 | 14.517 |
| p9 | 92.423 | 6.320 |
| p10 | 101.623 | 7.478 |

These are RMS displacements relative to the real reference, with all available
manual observations retained. Residuals reflect annotation noise and parallax.
100% of visual transforms were solved directly. Scale varies from 0.99113 to
1.14621; rotation from −1.50486° to 2.46632°. The scale correction agrees with
changes in observed background-marker distances (checked within 0.04), rather
than arbitrary zoom. All matrices preserve uniform scale and have no projective
terms.

The constant output canvas is 1372×2330. Both MP4s decode all 331 frames,
retain AAC audio, and report 59.94 FPS (codec rounding of 0.000060001 FPS).
Video duration is 5.522189 s. Geometric black-border area averages 32.82%; union
canvas intentionally retains captured pixels instead of cropping/zooming them.

Hybrid floor fits are separate: 290 usable fits, 5 frames with too few controls,
and 36 degenerate configurations. This includes frames 245 and 317, where the
initial numerical fit had returned a matrix despite three aligned controls.
Invalid homographies are NaN in the NPZ and flagged in the CSV.
The visual video is unaffected. Floor reprojection error is not a visual
stability criterion.

## Artifacts and verification

- `stabilizer_results/visual/tatame_stabilized.mp4`
- `stabilizer_results/visual/stabilization_report.html`
- `stabilizer_results/visual/debug_overlay.mp4` (source/stabilized panels)
- `stabilizer_results/hybrid/tatame_stabilized.mp4`
- `stabilizer_results/hybrid/stabilization_report.html`
- `stabilizer_results/hybrid/floor_diagnostics.csv`
- `stabilizer_results/contact_sheet.jpg` (source/output frames 0, 127, 330)
- `stabilizer_results/main_button.png`, `stabilizer_integrated.png`,
  `stabilizer_result.png`, `stabilizer_standalone.png` (desktop smoke captures)

```bash
.venv/bin/python -m pytest tests/test_video_stabilizer.py tests/test_planar_geometry_tracker.py -q
# Existing menu tests require desktop access because Vaila(gui=False) still creates Tk:
.venv/bin/python -m pytest tests/test_vaila_cli_menu.py -q
# Opt-in real desktop test; opens and closes windows, no external downloads:
VAILA_GUI_SMOKE=1 .venv/bin/python -m pytest tests/test_video_stabilizer.py -q -k gui_display_smoke
```

The real-artifact acceptance test decodes every output frame, compares FPS,
duration/audio, verifies constant dimensions, finite similarity matrices, exact
matrix-to-marker-CSV agreement, individual/aggregate background improvements,
scale consistency and metric/visual marker separation. It skips only when the
example outputs have not yet been generated. Synthetic tests also cover sparse
CSV rows, degenerate fits, weighted/robust estimation, interpolation/unwrap,
affine/RANSAC options, canvas bounds, cancellation and duplicate GUI runs.

The desktop smoke invokes the actual main-window button, checks its grid and
parent relationship, renders a synthetic clip while processing GUI events,
checks Help/report URI dispatch, captures resizable layouts, closes the child
without closing the parent and opens/closes the standalone window. Browser
dispatch is mocked in this test; HTML files are real local documents.

Installers were reviewed: NumPy, pandas, SciPy, OpenCV, Tkinter and FFmpeg are
already provided by the project; no dependency or installer change was needed.
Variable frame timestamps are represented at effective FPS. Floor-only
homographies cannot stabilize non-coplanar objects without potential distortion.
