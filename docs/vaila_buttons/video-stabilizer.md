# Video Stabilizer

**Button:** C_B_r2_c2 — Video and Image, between Compress Video and Make Sync file.
**Handler:** `Vaila.video_stabilizer()` → `run_video_stabilizer_gui(parent=self)`.
**Version:** 0.4.1 · **Updated:** 15 September 2026.

Select a video and its getpixelvideo marker CSV, choose static marker IDs and
optional priority anchors, then Stabilize. Metric coordinates are optional.
The default similarity transform preserves shapes; hybrid mode adds floor-only
geometry diagnostics. Cancel is cooperative and progress stays responsive.

```bash
python -m vaila.video_stabilizer \
  --video tests/interp_geometry/tatame.mp4 \
  --markers tests/interp_geometry/tatame_markers.csv \
  --stabilization-markers 0-10 --anchor-markers 8,9,10
```

Outputs include the stabilized video with audio, exact transforms, transformed
markers, before/after measurements and an HTML report. The GUI prints the exact
CLI before starting. [Full module help](../../vaila/help/video_stabilizer.md).
