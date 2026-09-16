# Planar Geo

**Button:** C_B_r1_c2 — Video and Image, between Video↔PNG and Draw Box.
**Handler:** `Vaila.planar_geometry_tracker()` → `run_planar_geometry_tracker_gui(parent=self)`.
**Version:** 0.4.3 · **Updated:** 15 September 2026.

Standalone planar-geometry DLT2D tracker / gap-filler / extrapolator. Select a
getpixelvideo marker CSV, a metric target-geometry TOML (e.g.
`vaila/models/planar_targets/tatame_1x1m.toml`), optional reference video, and
an output directory. With a video selected, writes `debug_projected_wireframe.mp4`.

The same module remains available inside **Get Pixel Coord** as the toolbar
**Geo Homog** wizard (interactive TOML/rectangle map + edit pause + live
wireframe). That path is unchanged.

```bash
uv run python -m vaila.planar_geometry_tracker
uv run python -m vaila.planar_geometry_tracker \
  --config vaila/models/planar_targets/tatame_1x1m.toml \
  --measurements-csv path/to/markers.csv \
  --video-path path/to/video.mp4 \
  --output-dir ./vaila_tracker_output \
  --debug-viz
```

[Full module help](../../vaila/help/planar_geometry_tracker.md).
