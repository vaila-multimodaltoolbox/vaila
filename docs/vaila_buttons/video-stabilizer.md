# Video Stabilizer

**Button:** C_B_r2_c2 — Video and Image, between Compress Video and Make Sync file.
**Handler:** `Vaila.video_stabilizer()` → `run_video_stabilizer_gui(parent=self)`.
**Version:** 0.4.3 · **Updated:** 15 September 2026.

Select a video and its getpixelvideo marker CSV, choose static marker IDs and
optional priority anchors, then Stabilize. Metric coordinates are optional.
The default similarity transform preserves shapes; hybrid mode adds floor-only
geometry diagnostics. **Sweep all methods** triages every configured method
without video encoding, ranks regional marker residuals, and renders the chosen
top candidates. Cancel is cooperative and progress stays responsive.

```bash
python -m vaila.video_stabilizer \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --geometry-config vaila/models/planar_targets/tatame_1x1m.toml \
  --metric-markers 0-7 --sweep
```

Outputs include the stabilized video with audio, exact transforms, transformed
markers, before/after measurements and an HTML report. A sweep additionally
writes a ranked CSV, comparison HTML, best command, montage, and flat videos/
folder. The GUI prints the exact CLI before starting.
[Full module help](../../vaila/help/video_stabilizer.md).
