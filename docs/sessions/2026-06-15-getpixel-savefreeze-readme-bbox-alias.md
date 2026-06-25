# Session 2026-06-15 — Save-freeze fix + verbose SAM3 README + bbox alias

**Branch:** `reidtrain`
**vailá version:** `0.3.55`
**Update date stamped in headers:** `15 June 2026`
**Predecessor session:** `docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md`
**Author of session:** Paulo Roberto Pereira Santiago (+ AI assistant)
**Audience:** any LLM / IDE agent picking up this work next.

> Read alongside `AGENTS.md` § *History (cross-IDE memory)*, `CLAUDE.md`
> § *Recent GUI Notes*, `.claude/skills/sam3-video/SKILL.md`, and
> `.claude/skills/getpixelvideo-tracking-loader/SKILL.md`.

---

## 1. Three concrete problems reported

1. The `README_sam.txt` produced by `vaila_sam.py` was an 8-line file with
   chunk stats only. The user wanted **every produced CSV/JSON explained**:
   schema, units, downstream role.
2. The file containing bboxes (`sam_tracks.csv`) didn't have **"bbox"** in
   the name. Hard to spot in a long directory listing.
3. **`vaila/getpixelvideo.py` Save froze the GUI** after loading a SAM3
   `sam_tracks.csv` (248 K bboxes, 16 693 frames) and converting bboxes to
   markers via the anchor prompt. The freeze was so total the user had to
   `sudo kill -9` the pid.

---

## 2. Root causes

### 2.1 README_sam.txt was sparse

`vaila_sam.py` had two duplicate `readme.write_text(...)` blocks (one in
`_merge_chunk_outputs`'s caller around line 2464, one in `run_sam3_on_video`
around line 3541) that only wrote the run header. No glossary, no schema.

### 2.2 sam_tracks.csv name had no "bbox" tag

The canonical filename `sam_tracks.csv` is hard-coded across ~10 modules
(`reid_markers.py`, `sam_postprocess.py`, `sam_validate.py`,
`soccerfield_calib.py`, `fifa_to_dlt.py`, …). Renaming would cascade.

### 2.3 Save freeze in getpixelvideo

Two compounding bugs:

**A. Wrong Save branch chosen.** After the smart loader converted bboxes to
markers, `tracking_data` was still populated (for overlay). The Save handler
priority list was:

```python
if labeling_mode and bboxes:               save_labeling_project()
elif csv_loaded and tracking_data:         export_labeling_dataset(...)   # <-- this one
elif one_line_mode:                        save_1_line_coordinates(...)
else:                                      save_coordinates(...)
```

`export_labeling_dataset` extracts **every annotated frame** from the video
(seek + decode in OpenCV) and writes a YOLO-format image + JSON + .txt per
frame. For 16 693 frames that's ~50 K file writes + 16 693 video seeks on
the main pygame thread.

**B. `save_coordinates` itself was O(N×slots) `df.at[]` calls.** Even if the
right branch ran, with `coordinates` of shape 16 693 × 62 and ~248 K real
entries, the per-cell `df.at[frame, "pN_x"] = value` loop is minutes of
pandas label lookups.

Both bugs hit at once → "frozen GUI" symptom.

---

## 3. Fixes implemented

### 3.1 `vaila/vaila_sam.py`

- New constant `SAM_OUTPUT_FILE_GLOSSARY` — plain-text glossary of every
  file vailá's SAM3 path can produce (`sam_tracks.csv`,
  `sam_bbox_tracks.csv`, `sam_frames_meta.csv`, `sam_points.csv`,
  `sam_id_map.csv`, `sam_contours.json[.gz]`, `sam_masks_manifest.csv`,
  `<video>_sam_overlay.mp4`, `masks/`, `FAILED_sam.txt`) with schema, units,
  and downstream role.
- New helper `_write_sam_run_readme(output_dir, *, header)` — appends
  `SAM_OUTPUT_FILE_GLOSSARY` to a run-specific header and writes
  `README_sam.txt`.
- New helper `_make_sam_bbox_tracks_alias(output_dir)` — creates
  `sam_bbox_tracks.csv` next to `sam_tracks.csv`. Tries POSIX `os.link`
  first (zero disk cost, same inode), falls back to `shutil.copy2` on
  filesystems where hardlinking is disallowed.
- Both writers (`_merge_chunk_outputs` caller + `run_sam3_on_video`) now go
  through these helpers — single source of truth.
- Cleaned up an unrelated duplicate `_assignment_min_cost` definition that
  slipped in yesterday (the function existed twice; Python kept the second).
  Lint clean.

### 3.2 `vaila/getpixelvideo.py`

- New state flag `bbox_converted_to_markers = False`. Set to `True` in
  `load_tracking_csv()` after:
  - a `sam_points.csv` direct load (already a marker file), or
  - a successful bbox → marker anchor conversion (user picked
    `1`/`2`/`3`/`4`/`5` for center/bottom/top/left/right).
- Save handler now reads:
  ```python
  elif csv_loaded and tracking_data and not bbox_converted_to_markers:
      ...export_labeling_dataset(...)        # YOLO dataset path (slow)
  ```
  So once the user converts bboxes to markers, Save goes through the
  vectorised `save_coordinates` and writes a regular `*_markers.csv`.
  Users who want the YOLO dataset export answer the anchor prompt with
  Enter (skip conversion) — `tracking_data` stays as overlay only and the
  dataset path still runs.
- `save_coordinates` rewritten to use **vectorised NumPy bulk assignment**:
  allocate `(n_frames, max_points*2)` float64 array, fill with bulk
  `arr[frame, slot*2 : slot*2+2] = (x, y)` style assignment, then wrap in
  a single `pd.DataFrame`. The legacy per-cell `df.at[]` loop is gone.
- New helper `_flush_save_message(screen, text)` — paints a 36 px yellow
  banner directly to the pygame surface and flips the display **before**
  any long save begins (both the dataset path and the marker save path).
- Updated docstring on `load_tracking_csv` print line to say
  "Save will write `*_markers.csv`" so the user knows what's about to
  happen.

### 3.3 Version metadata sync

All headers bumped to `0.3.55 / 15 June 2026`:
- `vaila.py`, `vaila/vaila_sam.py`, `vaila/sam_postprocess.py`,
  `vaila/getpixelvideo.py`
- `vaila/help/vaila_sam.{md,html}`, `vaila/help/getpixelvideo.{md,html}`
- `README.md` already at `2026-06-15` (pre-existing housekeeping)

---

## 4. Verification

### 4.1 Lint / format

```bash
uv run ruff check vaila/vaila_sam.py vaila/getpixelvideo.py vaila/sam_postprocess.py
uv run ruff format --check vaila/vaila_sam.py vaila/getpixelvideo.py vaila/sam_postprocess.py
```
All checks passed.

### 4.2 Tests

```bash
uv run pytest tests/ -q --ignore=tests/SAM --ignore=tests/Train_yolo \
    --deselect tests/test_tugturn_integration.py::test_cli_end_to_end
# 338 passed, 1 skipped, 1 deselected, 4 warnings
```

The `tugturn` deselect is the unrelated Qt-platform-plugin (`xcb`) env
failure documented in *AGENTS.md* History for the `drawsportsfields` fix.
Not a regression introduced today.

### 4.3 Functional benchmark for `save_coordinates`

```python
# 16693 frames × 62 slots × 248310 entries, mimicking user's failing case
# Before: hung (minutes / sudo kill -9)
# After:  0.56 s, 3.4 MB output
```

### 4.4 Functional test for SAM helpers

```python
_write_sam_run_readme(out, header="...") + _make_sam_bbox_tracks_alias(out)
# -> verbose README written
# -> sam_bbox_tracks.csv exists; same inode as sam_tracks.csv (hardlinked)
```

---

## 5. Files touched

```
modified: AGENTS.md                                          # History entry (v0.3.55)
modified: .claude/skills/sam3-video/SKILL.md                 # Output Format section + helpers
modified: .claude/skills/getpixelvideo-tracking-loader/SKILL.md  # Save behaviour (v0.3.55)
modified: README.md                                          # already at 2026-06-15
modified: vaila.py                                           # version 0.3.55
modified: vaila/vaila_sam.py                                 # helpers + version
modified: vaila/sam_postprocess.py                           # version
modified: vaila/getpixelvideo.py                             # flag, route, vectorise, banner
modified: vaila/help/vaila_sam.md                            # v0.3.55 entry, bbox alias
modified: vaila/help/vaila_sam.html                          # v0.3.55 entry, bbox alias
modified: vaila/help/getpixelvideo.md                        # v0.3.55 entry
modified: vaila/help/getpixelvideo.html                      # v0.3.55 entry
new:      docs/sessions/2026-06-15-getpixel-savefreeze-readme-bbox-alias.md
```

---

## 6. How to continue this work in a new session

Paste in a new chat (any LLM / IDE):

```
We are continuing the vailá session from 2026-06-15.
Read these in order:
  1. docs/sessions/2026-06-15-getpixel-savefreeze-readme-bbox-alias.md  (this file)
  2. docs/sessions/2026-06-14-getpixel-sam3-crosschunk.md               (predecessor)
  3. .claude/skills/sam3-video/SKILL.md
  4. .claude/skills/getpixelvideo-tracking-loader/SKILL.md
  5. AGENTS.md  § History (cross-IDE memory)
  6. CLAUDE.md  § Recent GUI Notes

Project version is 0.3.55 / 15 June 2026, branch reidtrain.
Astral toolchain only (uv / ruff / ty); never use bare pip / black / mypy.
```

### 6.1 Likely next steps (not yet done)

- **Verify the Save fix end-to-end** on the real user dataset
  (`/home/preto/data/ComercialFC/processed_sam_20260614_121915/yoyo_comercial_01052026_dbox/sam_tracks.csv`):
  load, convert with anchor `2 (bottom)`, hit Save, confirm a
  `yoyo_comercial_01052026_dbox_markers.csv` is written under 5 s with no
  freeze.
- **Run a real long SAM3 batch** on a fresh video and `cat README_sam.txt`
  to confirm the verbose glossary is what the user actually wants. If they
  want different sections (e.g. add `--postprocess-points` flag values
  used, total runtime), extend `_write_sam_run_readme`'s `header` arg or
  the glossary constant.
- **Optional second pass on `getpixelvideo.py`:** the Save handler is now
  ~15 lines per branch. Consider extracting a small `_do_save(...) -> str`
  helper so future fixes don't have to touch the pygame event loop.
- **Optional**: add a per-video `sam_global_id_map.json` (cross-chunk id
  remapping audit trail) as suggested in the 2026-06-14 session §6.1; this
  was deferred again today.

---

## 7. Quick command reference

```bash
# Run the GUI
uv run vaila.py

# Smart loader CLI (use Load Tracking CSV → pick sam_tracks.csv or sam_bbox_tracks.csv)
uv run vaila/getpixelvideo.py -f /path/to/video.mp4

# SAM3 single video (writes verbose README + bbox alias)
uv run vaila/vaila_sam.py -i video.mp4 -o out/ -t person \
    --max-frames 48 --max-input-long-edge 1280

# Lint + format + type check + focused tests
uv run ruff check vaila/ --fix && \
uv run ruff format vaila/ && \
uv run ty check vaila/ && \
uv run pytest tests/test_vaila_sam.py tests/test_sam_postprocess.py \
              tests/test_getpixelvideo_markers_io.py -q
```

---

---

## 8. Follow-up edit (same day, later): terminal progress for ML saves

After the GUI freeze was fixed the user noticed that the **ML dataset save**
path (`export_labeling_dataset`, `export_pose_dataset`) is **legitimately
slow** — extracting and re-encoding thousands of video frames takes minutes.
The pygame window appears frozen during that time; without any terminal
feedback users still assume the process hung.

### Fix

Three new top-level helpers in `vaila/getpixelvideo.py`:

| Helper | Role |
|--------|------|
| `_save_banner(title, detail)` | Print a `=` boxed banner with destination + counts before any long save begins. Prefix is `>> vaila/getpixelvideo:` (square brackets avoided because `absl` logging — installed by mediapipe/opencv — eats `[...]` tags from stdout). |
| `_save_done(message)` | Print a one-line `>> vaila/getpixelvideo DONE: …` tail when the save completes. |
| `_try_import_tqdm()` | Soft import: returns `tqdm.tqdm` if available, else `None`. tqdm is already a transitive dep via ultralytics / pytorch. |

Wired into three writers:

1. **`export_labeling_dataset`** (YOLO detection) — banner with frame count
   and split breakdown, then a `tqdm` bar per split iterating annotated
   frames.
2. **`export_pose_dataset`** (YOLO pose) — banner with kpt count + split
   breakdown, then a `tqdm` bar per split.
3. **`_export_all_labels_view`** — banner + `tqdm` bar over copied `.txt`
   files (each split).

Final `_save_done(...)` tail confirms completion + output dir.

### Lessons learned (worth keeping for future agents)

- **`absl` logging eats `[tag]` prefixes from stdout.** mediapipe (and
  several other ML libs) install absl on import. The `print(f"[vaila/...] …")`
  call returned no error but the bracketed prefix was silently stripped
  before reaching the terminal. The fix is purely cosmetic — switch to a
  `>>` prefix — but the symptom can be confusing because
  `inspect.getsource()` still shows the original `[...]` code. **Always
  test stdout output through a real script, not just by source inspection.**
- **tqdm writes to stderr**, so its bracketed `desc=...` survives the absl
  stdout filter unchanged. No need to rewrite tqdm prefixes.

### Files touched in this follow-up

```
modified: vaila/getpixelvideo.py            # 3 helpers + wired into 3 writers
modified: vaila/help/getpixelvideo.md       # v0.3.55 changelog extended
modified: vaila/help/getpixelvideo.html     # v0.3.55 changelog extended
modified: AGENTS.md                         # History entry extended
modified: docs/sessions/2026-06-15-getpixel-savefreeze-readme-bbox-alias.md  # this file
```

No new version bump (still v0.3.55 / 15 June 2026); same release.

---

**End of session 2026-06-15.** Next agent: read § 6.1 for open follow-ups.
