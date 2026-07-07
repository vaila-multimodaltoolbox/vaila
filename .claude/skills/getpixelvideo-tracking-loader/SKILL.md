# getpixelvideo Smart Tracking-CSV Loader (vailá)

Use when the user works on **`vaila/getpixelvideo.py` Load Tracking CSV**,
needs to import **SAM3** / **YOLO** tracking files as editable markers,
mentions **bbox → marker anchor conversion** (center / bottom / top / left /
right), or hits a *“Frame column not found”* error in the legacy loader.

## vailá maintenance rule (version/date)

Whenever you edit `vaila/getpixelvideo.py` (or any `*.py` in repo), also
update:

- script header **Update Date** (today) + **Version** (global, from `vaila.py`
  header/banner)
- root `README.md` line `Last updated: YYYY-MM-DD`
- help docs: `vaila/help/getpixelvideo.md` + `.html`, plus `vaila/help/index.md`
  + `.html` (“Generated on”)
- installers / `vaila.py` if change impacts install/run UX

Reference checklist: `AGENTS.md` § *Mandatory: Update metadata on any script
change*.

---

## What the smart loader does (v0.3.54+)

The **Load Tracking CSV** button in `getpixelvideo.py` used to fail whenever
a file did not have a literal `Frame` column or used vailá’s own SAM3 / YOLO
export schema. The smart loader auto-detects 5 formats and offers a
**bbox → marker anchor** conversion so the data can be edited like any other
vailá marker set.

### Supported formats (auto-detected)

| Format key | Typical file | Notes |
|------------|--------------|-------|
| `sam_points` | `sam_points.csv` | Point markers, loads directly |
| `sam_tracks` | `sam_tracks.csv` (or `sam_bbox_tracks.csv` alias) | Long bbox+area per (frame, obj_id) |
| `sam_frames_meta` | `sam_frames_meta.csv` | Per-frame meta with normalised bbox (`xc, yc, w, h ∈ [0,1]`) |
| `yolo_multi` | `all_id_detection.csv` | One row per detection, multi-id |
| `yolo_single` | `person_id_NN.csv` | Single-id YOLO export |
| `sapiens_pose_long` | `*_sapiens_vaila.csv` | Sapiens2 long pose; prompt for `person_id` |
| `sapiens_tracks` | `sapiens_tracks.csv` (legacy) | `stable_id` + `x1..y2` bbox rows |
| `sam_tracks` | `sapiens_bbox_tracks.csv` | Same SAM schema as `sam_tracks.csv` |

**Sapiens2 wide pose (308 kp):** use the regular **Load** button on
``<stem>_id_NN_sapiens_pose.csv`` or ``<stem>_getpixelvideo_pose.csv`` from
``vaila/vaila_sapiens.py`` — not the Load Tracking CSV button.

Anything unrecognised falls back to the legacy YOLO parser, so old files
still load.

### Save behaviour (v0.3.55)

After a successful anchor conversion, the loader sets
``bbox_converted_to_markers = True``. The **Save** button is then routed to
the regular ``save_coordinates`` writer (vailá ``*_markers.csv``) and **does
NOT** trigger ``export_labeling_dataset`` — the legacy bbox-to-YOLO export
path. Why this matters: ``export_labeling_dataset`` extracts every annotated
frame to disk; on a 16k-frame broadcast clip with 248k bboxes that froze the
pygame loop for tens of minutes and users were forced to ``sudo kill -9`` the
process.

If you want the YOLO dataset export instead, **answer the anchor prompt with
Enter** (skip conversion, bboxes stay as overlay only); Save will then go
through the dataset exporter.

``save_coordinates`` itself is **vectorised** (NumPy bulk assignment instead
of per-cell ``df.at[]`` loops). The same 248k-bbox case writes in ~0.5 s.

### Terminal feedback for ML dataset saves (v0.3.55, later same day)

The YOLO dataset writers are legitimately slow — they extract + re-encode
every annotated frame. While Python is blocked the pygame window appears
frozen. Three helpers in `vaila/getpixelvideo.py` give the user terminal
visibility:

- ``_save_banner(title, detail)`` — `=`-boxed banner before each save begins
  with destination + counts. **Prefix is `>>` not `[...]`** because absl
  logging (installed by mediapipe / opencv on import) silently eats
  bracketed prefixes from stdout. The symptom is confusing because
  `inspect.getsource()` still shows the original `[...]` code but the
  output is missing the prefix. tqdm writes to stderr so its `desc=...`
  brackets are unaffected.
- ``_save_done(message)`` — single-line completion tail.
- ``_try_import_tqdm()`` — soft import. tqdm is already a transitive dep via
  ultralytics / pytorch.

Wired into ``export_labeling_dataset`` (YOLO detection),
``export_pose_dataset`` (YOLO pose), and ``_export_all_labels_view``. Each
writer shows a tqdm bar per train/val/test split.

### Anchor prompt (bbox formats)

After detecting a bbox format, `load_tracking_csv` populates `tracking_data`
for overlay and then asks (via `show_input_dialog`):

```
1 = center      (default if you just type "center")
2 = bottom      (good for foot contact / ground contact studies)
3 = top         (head reference)
4 = left
5 = right
Enter = skip conversion, keep bbox overlay only
```

Choosing an anchor calls `bboxes_to_marker_coordinates(bboxes, total_frames,
anchor)` which builds the regular `coordinates: dict[int, list]` plus a
`labels` vector. From that moment on the data behaves exactly like any other
marker set: editable, TAB-navigable, saveable.

---

## Public API (single-source-of-truth)

| Function | Purpose |
|----------|---------|
| `_BBOX_ANCHOR_ALIASES` | Maps friendly names + numeric shortcuts to canonical anchor keys |
| `_anchor_xy_from_bbox(x1, y1, x2, y2, anchor)` | Returns `(x, y)` int pixel for the chosen anchor |
| `_detect_frame_col(columns)` | Case-insensitive locator for the frame column |
| `_detect_tracking_format(df)` | Returns one of `sam_tracks / sam_frames_meta / sam_points / yolo_multi / yolo_single / sapiens_pose_long / sapiens_tracks / unknown` |
| `_sapiens_long_to_marker_df(df, person_id, kpt_thr)` | Pivot Sapiens2 long CSV to wide marker layout |
| `_iter_bboxes_from_df(df, fmt, video_width, video_height)` | Generator of uniform `{frame, obj_id, x1, y1, x2, y2, label, score}` dicts (normalised → pixel handled here) |
| `bboxes_to_marker_coordinates(bboxes, total_frames, anchor)` | Build `(coordinates, labels)` from a list of bbox dicts |
| `load_tracking_csv(...)` | Top-level loader; orchestrates detect → iterate → optional anchor prompt |

---

## Typical usage

### GUI flow

1. Launch `uv run vaila.py` → Frame C → **Pixel Coordinate Tool**, or
   `uv run vaila/getpixelvideo.py -f /path/to/video.mp4`.
2. Click **Load Tracking CSV**.
3. Pick `sam_tracks.csv` / `sam_frames_meta.csv` / `all_id_detection.csv` / etc.
4. Anchor prompt appears (only for bbox formats). Type `1`–`5` or a friendly
   name. Press Enter to keep the overlay only (no marker creation).
5. Edit / save the resulting markers as you would with any other set.

### Headless / scripting

When integrating into other scripts:

```python
from vaila.getpixelvideo import (
    _detect_tracking_format,
    _iter_bboxes_from_df,
    bboxes_to_marker_coordinates,
)

import pandas as pd

df = pd.read_csv("sam_tracks.csv")
fmt = _detect_tracking_format(df)          # "sam_tracks"
bboxes = list(_iter_bboxes_from_df(df, fmt, video_width=1920, video_height=1080))
coords, labels = bboxes_to_marker_coordinates(
    bboxes, total_frames=1189, anchor="bottom"
)
```

`coords` is the canonical vailá `dict[int, list[(x, y)]]` keyed by frame
index, ready to feed `save_markers` or any downstream module.

---

## Troubleshooting

### *“Frame column not found”* but file looks correct

Old loader required exact `Frame`. The smart loader uses
`_detect_frame_col(columns)` which accepts `frame`, `Frame`, `FRAME`,
`frame_id`, `frame_idx`. If you still see the error you are on a pre-v0.3.54
copy → update `getpixelvideo.py` (header should say `Version: 0.3.54`).

### Normalised bboxes appear in the wrong place

`sam_frames_meta.csv` stores bboxes as `xc, yc, w, h ∈ [0, 1]`. The loader
converts them with the video’s **actual** width/height as read by OpenCV at
load time. If you opened a different video before loading the CSV you’ll see
the offset bug — switch to the matching video first.

### Anchor prompt did not appear

Anchor prompt only fires for **bbox** formats. `sam_points.csv` is loaded
directly as markers (no bbox, no anchor needed).

---

## Tests

```bash
uv run pytest tests/test_sam_postprocess.py tests/test_vaila_sam.py -q
```

(No dedicated test module yet for `getpixelvideo.py`’s smart loader — the
SAM postprocess tests cover the upstream `sam_tracks.csv` /
`sam_frames_meta.csv` schemas that the loader consumes.)

---

## Open follow-ups (see `docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md` § 6.1)

- Wire the toolbar “Load Tracking CSV” button through the same
  `load_tracking_csv` path so the anchor prompt also fires there (currently
  only the file-dialog code path does).
- Expose `--load-tracking-csv FILE --bbox-anchor center` as CLI flags for
  headless conversion.
- Persist the chosen anchor in the marker file metadata so re-opening picks
  up the same convention without re-asking.

---

## Code map

| File | Role |
|------|------|
| `vaila/getpixelvideo.py` | Pixel coordinate GUI + smart loader |
| `vaila/help/getpixelvideo.md` / `.html` | User-facing help |
| `vaila/vaila_sam.py` | Produces `sam_tracks.csv`, `sam_frames_meta.csv`, `sam_points.csv`, `sam_contours.json` |
| `vaila/sam_postprocess.py` | Converts SAM3 batch output to vailá pixel CSVs |
| `tests/test_sam_postprocess.py` | Validates the SAM3 CSV schemas the loader consumes |
| `docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md` | Full session history (read first on continuation) |
