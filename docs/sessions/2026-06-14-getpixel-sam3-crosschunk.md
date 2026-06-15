# Session 2026-06-14 — getpixelvideo intelligent loader + SAM3 cross-chunk tracklet linking

**Branch:** `reidtrain`
**vailá version:** `0.3.54`
**Update date stamped in headers:** `14 June 2026`
**Author of session:** Paulo Roberto Pereira Santiago (+ AI assistant)
**Audience:** any LLM / IDE agent picking up this work next (Claude Code, Cursor, Codex, Antigravity, Windsurf, …).

> Read alongside `AGENTS.md` § *History (cross-IDE memory)*, `CLAUDE.md`,
> `.claude/skills/sam3-video/SKILL.md`, and the new
> `.claude/skills/getpixelvideo-tracking-loader/SKILL.md`.

---

## 1. Goals requested by the user

1. **Version sync** — bump everything edited today to **v0.3.54 / 14 June 2026**
   (match `vaila.py`’s header/banner): `vaila_sam.py`, `sam_postprocess.py`,
   `getpixelvideo.py`, help files for SAM, `README.md`.
2. **Fix “silly problems” in `getpixelvideo.py`’s Load Tracking CSV** — it was
   failing whenever the `Frame` column was missing/renamed or the file came
   from vailá’s own YOLO / SAM3 exports.
3. **Intelligent CSV loader** in `getpixelvideo.py`:
   - Auto-detect SAM3 outputs (`sam_tracks.csv`, `sam_frames_meta.csv`,
     `sam_points.csv`) and YOLO multi/single-id exports.
   - Allow loading **bounding boxes** (bbox), not just point markers.
   - Convert bbox → vailá marker using an **anchor option** (center / bottom /
     top / left / right), so the data becomes editable as normal markers.
4. **Cross-Chunk Tracklet Linking** in `vaila_sam.py`:
   - Sliding window overlap of **2 frames** between adjacent chunks.
   - Bipartite cost matrix (IoU + centroid distance) on shared frames.
   - Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) for the 1:1
     matching, with a greedy fallback when SciPy is missing.
   - Linked-list merging so chunk-N+1 IDs inherit chunk-N persistent global IDs.
5. **History/handoff documentation** — keep `AGENTS.md` + `CLAUDE.md` + skills
   in sync so any LLM can pick the work up later.

---

## 2. Files changed today

```
modified: README.md
modified: tests/test_sam_postprocess.py
modified: tests/test_vaila_sam.py
modified: uv.lock
modified: vaila.py
modified: vaila/getpixelvideo.py
modified: vaila/help/index.md
modified: vaila/help/vaila_sam.html
modified: vaila/help/vaila_sam.md
modified: vaila/sam_postprocess.py
modified: vaila/vaila_sam.py
```

Untracked artifacts (test outputs, gitignored):

```
tests/Train_yolo/processed_sam_20260612_162157/
tests/Train_yolo/processed_sam_20260614_121915/
```

All headers carry `Update Date: 14 June 2026` and `Version: 0.3.54`. `README.md`
`Last updated` is `2026-06-14`. Help index “Generated on” line is in sync.

---

## 3. `vaila/getpixelvideo.py` — intelligent CSV loader

### 3.1 New module-level helpers

- `_BBOX_ANCHOR_ALIASES: dict[str, str]` — maps friendly names (and numeric
  shortcuts `1`–`5`) to a canonical anchor key.
- `_anchor_xy_from_bbox(x1, y1, x2, y2, anchor) -> tuple[int, int]` — computes
  the pixel (x, y) for the requested anchor (center, top, bottom, left, right,
  top-left, top-right, bottom-left, bottom-right).
- `_detect_frame_col(columns) -> str | None` — case-insensitive locator for the
  `frame` column (also accepts `frame_id`, `frame_idx`, etc.).
- `_detect_tracking_format(df) -> str` — returns one of:
  `sam_tracks | sam_frames_meta | sam_points | yolo_multi | yolo_single |
  unknown`.
- `_iter_bboxes_from_df(df, fmt, video_width, video_height)` —
  generator yielding uniform dicts:
  `{frame, obj_id, x1, y1, x2, y2, label, score}`. Normalised SAM coordinates
  (`xc, yc, w, h` in `[0,1]`) are converted to pixel space using the video
  width/height.
- `bboxes_to_marker_coordinates(bboxes, total_frames, anchor)` — converts a
  list of bbox dicts to vailá `coordinates: dict[int, list]` plus a `labels`
  vector, using the requested anchor.

### 3.2 `load_tracking_csv()` behaviour

1. Reads the CSV (tolerates BOM/whitespace).
2. Calls `_detect_tracking_format(df)`.
3. **If `sam_points`** → load directly as markers via `load_marker_csv_df`.
4. **If bbox format (`sam_tracks` / `sam_frames_meta` / `yolo_single` /
   `yolo_multi`)** → iterate with `_iter_bboxes_from_df`, populate
   `tracking_data` for overlay, then **prompt the user** via `show_input_dialog`:
   ```
   1 = center
   2 = bottom
   3 = top
   4 = left
   5 = right
   (Enter = skip conversion, keep bbox overlay only)
   ```
   If an anchor is chosen, `bboxes_to_marker_coordinates` is used and the
   bboxes become regular, editable, saveable markers.
5. **Else** → fall back to legacy YOLO parser.

The helper updates `nonlocal coordinates, labels, deleted_positions,
selected_marker_idx, one_line_mode` so the GUI stays consistent with the new
data.

### 3.3 Docstring updates

`vaila/getpixelvideo.py` top docstring now documents the smart loader and the
anchor prompt (see lines around 46–53).

### 3.4 Lint fixes

Removed `E702 multiple statements on one line` flagged by ruff inside
`_iter_bboxes_from_df` (split `x1 = …; y1 = …` into separate lines).

---

## 4. `vaila/vaila_sam.py` — Cross-Chunk Tracklet Linking

### 4.1 What was broken

`_build_cross_chunk_id_maps` referenced `_assignment_min_cost`, but that
function had been imported informally from `reid_markers.py` and was **never
defined in `vaila_sam.py`** → `NameError` at runtime whenever chunked merge
fired. Result: chunked path produced random per-chunk IDs.

### 4.2 The fix

- `_assignment_min_cost` is now defined inline in `vaila_sam.py`
  (single definition, after a duplicate was cleaned up later in this session).
  Uses `scipy.optimize.linear_sum_assignment` if available, with a greedy
  minimum-cost fallback otherwise. SciPy is already a hard dependency in
  `pyproject.toml` so the greedy path is only a defensive fallback.
- `_split_video_into_chunks(..., overlap_frames=2)` is the default, and
  `_process_video_chunked` explicitly passes `overlap_frames=2` and logs it.
- `_build_cross_chunk_id_maps` docstring rewritten to spell out the 5-step
  pipeline (sliding window → feature caching → graph association → optimal
  matching → linked-list merge).

### 4.3 Algorithm (implemented)

For each chunk N → N+1 boundary:

1. **Collect overlap detections** from `sam_tracks.csv` of both chunks for the
   shared frames (`start_frame_N+1 ≤ global_frame < end_frame_N`).
2. **Build the cost matrix** of shape `(local_ids_N+1, global_ids_N_history)`.
   Per shared frame the per-pair score is
   `(1 − IoU) + min(1, dist / max_centroid_dist_px)` averaged across all
   overlap frames where both objects appear.
3. **Gate matches** with `min_iou=0.05` AND `max_centroid_dist_px=180`.
4. **Solve** with Hungarian (`_assignment_min_cost`).
5. **Merge**: matched chunk-N+1 local IDs inherit the persistent global ID
   from chunk-N. Unmatched IDs allocate the next free `next_gid`.
6. **Per-chunk mapping** is appended to `maps` and applied when assembling the
   final outputs (`sam_tracks.csv`, mask filenames, etc.).

### 4.4 Help / docs

- `vaila/help/vaila_sam.md` — new bullet under *Main features* for Cross-Chunk
  Tracklet Linking.
- `vaila/help/vaila_sam.html` — new entry in *Recent updates* describing the
  `NameError` fix and the Hungarian-based stitch.

---

## 5. Tests

### 5.1 Updates

- `tests/test_vaila_sam.py::test_merge_chunk_outputs` — the assertions for
  remapped mask filenames were updated: with non-overlapping chunks, chunk-0
  local ID 1 → global ID 0, chunk-1 local ID 1 → global ID 1, so we now expect
  `frame_000000_obj_0.png` and `frame_000019_obj_1.png` (previously both
  expected `_obj_1`).
- `tests/test_sam_postprocess.py` — minor version/date sync.

### 5.2 What passes

```bash
uv run ruff check vaila/vaila_sam.py vaila/sam_postprocess.py vaila/getpixelvideo.py
uv run ruff format vaila/getpixelvideo.py
uv run pytest tests/test_sam_postprocess.py tests/test_vaila_sam.py -q
uv run pytest tests/ -q --ignore=tests/SAM --ignore=tests/Train_yolo \
    --deselect tests/test_tugturn_integration.py::test_cli_end_to_end
```

All green. The `tugturn_integration` CLI deselect is an unrelated **Qt
platform plugin** environment failure (see *AGENTS.md* History for the
`drawsportsfields` fix that taught us to always force `matplotlib.use("TkAgg")`
before importing `pyplot`). It’s not a regression introduced today.

---

## 6. How to continue this work in a new session

Open a new chat in any LLM (Claude Code, Cursor, Codex, etc.) and paste this
prompt:

```
We are continuing the vailá session from 2026-06-14.
Read these in order:
  1. docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md   (this file)
  2. .claude/skills/sam3-video/SKILL.md                     (cross-chunk + OOM)
  3. .claude/skills/getpixelvideo-tracking-loader/SKILL.md  (smart loader)
  4. AGENTS.md  § History (cross-IDE memory)
  5. CLAUDE.md  § Recent GUI Notes

Project version is 0.3.54 / 14 June 2026, branch reidtrain.
Astral toolchain only (uv / ruff / ty); never use bare pip / black / mypy.
```

### 6.1 Likely next steps (not yet done)

- **Run SAM3 on a real long broadcast clip** (>1500 frames at 1080p, ≥48-frame
  chunks) and verify that **global IDs survive across all chunk boundaries**,
  not just synthetic 20-frame test fixtures.
- **Stretch the cost function** to include an embedding term (Re-ID) instead
  of pure IoU + centroid distance, per the user’s “Appearance Embeddings”
  bullet in the spec. Today we only use spatial features because SAM3 does not
  expose a stable Re-ID head.
- **Expose `min_iou` / `max_centroid_dist_px` as CLI flags** of `vaila_sam.py`
  (`--xchunk-min-iou`, `--xchunk-max-centroid-px`).
- **Persist the global-ID mapping** to `sam_global_id_map.json` per video, so
  external tools (post-processing, ML training) can audit the stitching.
- **getpixelvideo**: add the same anchor prompt to the *toolbar button*
  (currently only the file dialog path triggers it). Then expose
  `--load-tracking-csv FILE --bbox-anchor center` as CLI flags for headless use.

---

## 7. Quick command reference

```bash
# Run the GUI
uv run vaila.py

# Run the smart loader CLI (after launch, use Load Tracking CSV)
uv run vaila/getpixelvideo.py -f /path/to/video.mp4

# SAM3 single video with chunked fallback
uv run vaila/vaila_sam.py -i video.mp4 -o out/ -t person \
    --max-frames 48 --max-input-long-edge 1280

# SAM3 batch directory
uv run vaila/vaila_sam.py -i videos_dir/ -o out/ -t player

# Lint + format + type check
uv run ruff check vaila/ --fix && \
uv run ruff format vaila/ && \
uv run ty check vaila/

# Focused test
uv run pytest tests/test_vaila_sam.py tests/test_sam_postprocess.py -q
```

---

## 8. Acknowledged caveats

- **Cross-chunk linking is spatial-only**: no appearance embedding. This is
  acceptable for short overlaps (2 frames) where motion is small, but if the
  user later needs robust Re-ID across long gaps, a CLIP or ReID-Net descriptor
  should be added to the cost matrix.
- **The smart loader currently asks anchor only on the file-dialog path.**
  Toolbar “Load Tracking CSV” still routes through the legacy parser; a small
  refactor will route both through `load_tracking_csv`.
- **`tests/test_tugturn_integration.py::test_cli_end_to_end`** is a Qt plugin
  / display environment failure (`xcb`). Not in scope for today’s work.

---

**End of session 2026-06-14.** Next agent: read the SKILLs first, then jump to
§ 6.1 for the open todo list.
