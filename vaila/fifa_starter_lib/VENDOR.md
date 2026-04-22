# Vendor — FIFA Skeletal Tracking Light 2026 starter kit

This directory contains a **partial, light-weight vendoring** of the official
FIFA Skeletal Tracking Light 2026 starter kit so that `vaila.fifa_skeletal_pipeline`
can run without a separate `git clone` or an editable install. The upstream
project is MIT-licensed (see [`LICENSE_MIT`](LICENSE_MIT)).

- **Upstream:** <https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026>
- **License:** MIT © 2025 G3P-Workshop
- **Snapshot:** `main` branch as of April 2026

## Files

| File | Source |
|------|--------|
| [`camera_tracker.py`](camera_tracker.py) | `lib/camera_tracker.py` |
| [`postprocess.py`](postprocess.py) | `lib/postprocess.py` |
| [`LICENSE_MIT`](LICENSE_MIT) | `LICENSE` (repo root) |

## Local modifications

- Added SPDX header + vendor banner in each Python file.
- Reformatted with ruff (line length, default Python formatting) — **no** logic
  changes.
- Safer initialization of `dist_map`/`labels`/`label2yx` inside `CameraTracker.track`
  to stop `ty` from complaining about possibly-unbound names when
  `refine_interval` is large.

## Updating the vendor snapshot

```bash
# From the repo root, with network access:
git clone --depth 1 \
  https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026.git \
  /tmp/fifa_sk
cp /tmp/fifa_sk/lib/camera_tracker.py vaila/fifa_starter_lib/camera_tracker.py
cp /tmp/fifa_sk/lib/postprocess.py    vaila/fifa_starter_lib/postprocess.py
# Reapply the SPDX banner at the top of each file and reformat:
uv run ruff format vaila/fifa_starter_lib/
```

Then verify no behavior drift:

```bash
uv run pytest tests/test_fifa_skeletal_pipeline.py -v
```
