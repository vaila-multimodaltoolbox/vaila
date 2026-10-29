# Independent side-by-side manual cuts — 28 October 2026

Version: 0.4.12.

Syncvid now offers **Manual cuts side by side (M)**. Each camera keeps its own
position, inclusive Start/End and manual deletion marks. Click a pane to select
the active camera; I/O set bounds and D/Delete toggles removal. Counts show the
remaining frames and their difference. One camera works; larger folders display
two cameras at a time. Zoom, pan, frame stepping and Go to frame operate on the
active camera.

**Save all / Save current video** exports original frames in order at each source's
FPS. Frame deletion shortens the output. There is no sync planning, event mapping,
interpolation, frame duplication or automatic count/FPS adjustment. Manual marks
are separate from sync/repeat-detector marks. Save current validates only that
camera, so invalid fields in another row do not block it.

Output is a new timestamped directory with lossless RGB H.264 MP4s (no audio),
source/output frame CSVs and `manual_cuts.json`. The JSON defines edits for audit
and replay, not synchronization. INCOMPLETE.txt identifies interrupted exports;
the active partial MP4 is removed. GUI progress accepts Cancel export/Esc, while
headless CLI accepts Ctrl+C. Existing files are never overwritten.

```bash
uv run --no-sync python -m vaila.syncvid --manual-cuts -i /path/to/videos
uv run --no-sync python -m vaila.syncvid --manual-cuts -i /path/to/videos \
  --cuts-file /path/to/manual_cuts.json --output-dir /tmp/new_manual_export
```

The UI and CLI share `manual_video_cuts.py` validation/export and the existing
sequential lossless encoder. Saved frame counts and rates are checked against
the source before output is created; derived counts and output names are recomputed.
No dependencies or installer changes are needed. Script, README and help metadata
remain at the current global version/date, with previous local changes preserved.

## Verification

```bash
.venv/bin/ruff check vaila/syncvid.py vaila/manual_video_cuts.py tests/test_manual_video_cuts.py
.venv/bin/ty check vaila/syncvid.py vaila/manual_video_cuts.py
.venv/bin/python -m pytest tests/test_manual_video_cuts.py tests/test_sync_events.py \
  tests/test_syncvid.py tests/test_cutvideo.py tests/test_syncvid_repaint.py \
  tests/test_syncvid_real_videos.py -q
```

Result: **160 passed, 13 skipped**; Ruff, formatting and Ty pass.

Coverage includes independent bounds/positions/marks, D restoration, live counts,
equal and unequal FPS/counts, original pixels after decoding, single/all exports,
saved edit replay, current-only export with another invalid camera, cancellation,
invalid or empty intervals, changed sources, preserving existing outputs, zoom/pan,
single-camera layout, paging three cameras, resizing, and existing sync workflows.
The full repository suite was not run. Optional tests requiring the absent
`tests/syncvid/` video bank are skipped.

## Separate real-source experiment

Read-only sources:
`/home/preto/data/veiga/forsync_after_norepeat/processed_fpsfix_20261027_113644`.
Artifacts: `/tmp/syncvid-manual-cuts-20261028/`.

For this verification only, c1 used 63–81 with ten explicit test deletion marks
and c2 used 54–62 without deletions. These are test choices, not proposed edits
to the user's data. Both exports contain **9 frames at 60000/1001 FPS**, without
audio. All 18 decoded output frames compare pixel-identically against their
explicit CSV source identities. The source clips/checkpoints were not modified.

`validation.json` records the source indices, rates and pixel checks; `export/`
holds the new videos, maps and definitions. `side_by_side.png` was rendered with
dummy SDL and visually reviewed for simultaneous panes, active selection, red
removal state, independent Start/End and equal remaining counts. It does not
substitute for a live desktop file-chooser/display review.
