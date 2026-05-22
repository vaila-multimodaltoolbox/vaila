# rf_trackers

## Module Information

- **Category:** ML
- **File:** `vaila/rf_trackers.py`
- **Version:** 0.3.44
- **Updated:** 2026-05-14
- **Author:** Paulo Roberto Pereira Santiago
- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila

## Description

Combines **Ultralytics YOLO26** (default checkpoint **`yolo26x.pt`** under `vaila/models/`) with **[Roboflow trackers](https://github.com/roboflow/trackers) 2.4.x** (SORT, ByteTrack, OC-SORT, BoT-SORT) and [supervision](https://github.com/roboflow/supervision). BoT-SORT with **camera motion compensation (CMC)** must receive the **raw BGR frame** in `tracker.update(detections, frame)` (see [trackers v2.4.0](https://github.com/roboflow/trackers/releases/tag/2.4.0)).

## `--help` and CLI progress

CLI prints a **status banner** and a **`tqdm` frame bar** on **stderr** while processing. Suppress with **`--quiet`** / **`-q`** (final line on stdout is unchanged).

```bash
uv run python -m vaila.rf_trackers --help
```

That lists all flags (`-i` / `--input`, `-w` / `--weights`, `--tracker`, `--conf`, `--no-save-video`, `--no-cmc`, `--cmc-method`, **`-q`**) and an **epilog** with copy-paste examples.

## GUI (default when no `-i`)

```bash
uv run python -m vaila.rf_trackers
```

Or: **Frame B → Video AI tools → Roboflow trackers (v2.4)** in `vaila.py`.

## CLI (headless)

Omit the GUI by passing a video:

```bash
uv run python -m vaila.rf_trackers -i /path/to/video.mp4 \
  -w vaila/models/yolo26x.pt \
  --tracker botsort --conf 0.25
```

### Example: FIFA broadcast clip

```bash
uv run python -m vaila.rf_trackers \
  -i /path/to/FIFA/to_sent/BRA_KOR_234113.mp4 \
  -w vaila/models/yolo26x.pt \
  --tracker botsort \
  --conf 0.25
```

Adjust `-w` if you prefer another `.pt` (e.g. `yolo26m.pt` for speed). Use `--no-save-video` to skip the overlay MP4 (CSV only). Use `--no-cmc` only for **BoT-SORT** when the camera is fixed. Use `--cmc-method` one of: `sparseOptFlow`, `orb`, `sift`, `ecc`.

## Output

Next to the input video: `processed_rf_trackers_YYYYMMDD_HHMMSS/`

| File | Purpose |
|------|---------|
| `rf_tracks.csv` | Long format: frame, track_id, box, confidence, class (debug / analysis) |
| `{Label}_id_XX.csv` | Per tracker ID, same columns as `yolov26track` / `initialize_csv` — one row per frame (`0..N-1`), NaN when missing |
| `all_id_detection.csv` | Wide merge from those per-ID files (for **Get Pixel Video** multi-object CSV) |
| `<stem>_rf_tracked.mp4` | Optional overlay (unless `--no-save-video` in CLI) |

Use **`all_id_detection.csv`** or individual **`*_id_*.csv`** in **Get Pixel Video** to load boxes and tracker IDs. Prefer a **fresh output directory** when re-running the combined merge (avoid mixing an old `all_id_detection.csv` with new per-ID files). **Reconstruction (`rec2d`)** expects the **pixel** CSV exported from Get Pixel video (landmarks), not these bbox files.

## Requirements

- `trackers==2.4.0` (see `pyproject.toml`; installs `supervision`).

## License

AGPL-3.0-or-later (vailá). Upstream **trackers** library: Apache-2.0.
