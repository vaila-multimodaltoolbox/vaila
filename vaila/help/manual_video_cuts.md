# Manual Video Cuts

- **Version:** 0.4.13
- **Updated:** 2026-10-28
- **Category:** Tools → Video and Image
- **File:** `vaila/manual_video_cuts.py` (export backend)
- **GUI:** Frame C → Make Sync file → Manual cuts side by side (M)

Independent video trims with inclusive Start/End and manually removed frames. The GUI lives in
`syncvid.py`; this helper validates edit definitions and exports exact selected originals.
Use [the complete GUI/CLI workflow](syncvid.md#manual-cuts-side-by-side).

Save all or the current camera into a new folder. Output preserves source FPS and decoded pixels,
without audio, synchronization, interpolation or automatic frame-count adjustment. CSV frame maps
and `manual_cuts.json` support audit and replay. Incomplete exports retain `INCOMPLETE.txt`.

```bash
uv run --no-sync python -m vaila.syncvid --manual-cuts -i /path/to/videos
uv run --no-sync python -m vaila.syncvid --manual-cuts -i /path/to/videos \
  --cuts-file /path/to/export/manual_cuts.json --output-dir /tmp/new_manual_export
```
