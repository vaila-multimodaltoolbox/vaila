# Syncvid event interpolation — 28 October 2026

Version: 0.4.12. The existing FPS and nearest-image event workflows remain available.
Select FPS, Events: nearest or Events: interpolate directly; the active button is highlighted.

## Behavior

- Every original in each inclusive selected interval has an exact rational assigned time.
  Reference event times use source-frame offsets divided by the assumed reference clock;
  each camera is calibrated linearly by frame ordinal between consecutive events.
- The reciprocal of the rational GCD of all assigned sample times gives the minimum
  preserving FPS. The configurable maximum defaults to 240 FPS; rates above the limit are reported,
  not rounded. Max output FPS / `--max-output-fps` can authorize 480 without forcing that output rate.
- Start, Sync and End are mandatory; additional correspondences are allowed. Invalid
  ordering, inconsistent event counts and removal marks inside intervals block export.
- Exact samples bypass synthesis. Intermediate images use bidirectional OpenCV Farnebäck
  flow, inverse warps and temporal blending. Two neighboring originals and their flows
  are cached, without loading the whole clip into memory.
- RGB H.264 at CRF 0 preserves decoded original pixels, with no audio. Each camera has
  an original/interpolated CSV with exact times, neighbors and blend fractions, plus
  the existing retained PNG/timing audit. Manifests/checkpoints save method and FPS policy.
- Imported clock/source-rate discrepancies require GUI confirmation or explicit CLI
  `--reference-fps`. A saved confirmation is tied to the probed reference rate.
- Both Cut Video GUI export paths use cancellable progress. Headless export prints
  progress and accepts Ctrl+C. An incomplete set retains `INCOMPLETE.txt`, and active
  partial MP4s are deleted. Export directories must be new.

## Automated verification

```bash
.venv/bin/ruff check vaila.py vaila/sync_events.py vaila/syncvid.py vaila/cutvideo.py tests/test_sync_events.py
.venv/bin/ty check vaila/sync_events.py vaila/syncvid.py vaila/cutvideo.py
.venv/bin/python -m pytest tests/test_sync_events.py tests/test_syncvid.py \
  tests/test_cutvideo.py tests/test_syncvid_repaint.py tests/test_syncvid_real_videos.py -q
```

Result: **140 passed, 13 skipped**. The skipped tests require the absent `tests/syncvid/`
fixture bank; they do not represent the separate real-source experiment below. Ruff/Ty pass.
Focused coverage includes the requested event example, exact original pixels after video
decoding, rational FPS probing, ordered extra events, invalid intervals and marks,
240-FPS boundary and over-limit rejection, known synthetic translation, cancellation,
old checkpoints, existing FPS behavior, and GUI/CLI checkpoint parity.
The whole repository test suite was not run. Installers were reviewed: no new dependencies
or installation/run setup are required.

## Real-source experiment

Read-only source directory:
`/home/preto/data/veiga/forsync_after_norepeat/processed_fpsfix_20261027_113644`.
The local copies both declare `60000/1001` FPS. For this experiment the reference
physical clock was explicitly set to **30000/1001**, following the requested assumption;
it is independent of the declared container rate.

Artifacts: `/tmp/syncvid-interpolation-20261028/` (not committed).
`example.checkpoint.json` is a separate copy with the requested event lists and no
removal marks. The existing source checkpoint and videos were not rewritten.

| Camera | Events (source, 1-based) | Output FPS | Output count | Originals checked |
|---|---|---|---|---|
| c1_main.mp4 | 63, 71, 81 | 60000/1001 | 37 | 19 |
| c2_16mm.mp4 | 54, 58, 62 | 60000/1001 | 37 | 9 |

All three event pairs occupy **1, 17, 37** in the output. Each selected source frame occurs
exactly once as an original row. All 28 originals compare pixel-identically after decoding
the lossless output; FFprobe confirms the rational FPS, 37 frames and no audio stream.
The 0.6006 s sample span differs from the 37-frame playback duration (about 0.617283 s).
`validation.json` records these checks. `export/` contains both videos and the audit files.

The output sheet and full-resolution frames were visually inspected. These particular
source intervals show the ball already in flight, so they cannot validate foot–ball
contact timing. A **separate per-camera diagnostic**, `foot_ball_diagnostic.png` and `.json`,
compares originals c1 14–16 and c2 13–15 with synthesized half-frame positions around the
kick. No new cross-camera event correspondences were asserted. Soft edges and blur remain
visible around the kicking foot; the source images themselves also have motion blur.
This inspection establishes neither sub-frame contact time nor physical accuracy of
the synthesized ball/foot geometry. Occlusion and fast small-object motion remain limitations.

`gui_interpolation.png` is a dummy-SDL GUI render confirming visible controls and values;
it does not substitute for live desktop interaction. Matching FPS/counts and exact anchors
alone cannot prove image-estimation quality.

## Configurable ceiling and shorter-camera regression

The follow-up implements `max_output_fps` in manifests/checkpoints and `--max-output-fps` in
the CLI. Default: 240. GUI: **Max output FPS (e.g. 480)**. Explicit CLI overrides the saved
ceiling; older checkpoints/manifests without it use 240. The ceiling authorizes the computed
minimum and never forces that rate. FPS, Events: nearest and Events: interpolate are now
direct, idempotent choices with an active highlight. The GUI previews exact/decimal FPS,
final count and each camera's original/interpolated counts; GUI/CLI also print the summary
before saving/exporting. Both cameras use the existing bidirectional rendering path.

Source session, read only:
`/home/preto/data/apagarser/manual_cuts_20261028_221957_603267/vai.checkpoint.json`.
It records **nearest**, with events **c1: 1/9/19**, **c2: 1/9/22** and a confirmed reference
clock of **979001/32666**. That clock was retained verbatim. With interpolation the default
240 limit correctly blocks the required **12727013/32666** FPS (389.610390). Limit 480
permits **235 frames per camera**, with events at **1, 105, 235**. C1 keeps **19 originals**
and adds **216 interpolated** images; C2 keeps **22 originals** and adds **213 interpolated**.

FFmpeg's default encoder/muxer time base rounded this particular rational to `30000/77`.
The exporter now explicitly sets the reciprocal encoder time base and MP4 track timescale.
FFprobe confirms both actual and average FPS as `12727013/32666`; all **470 packet timestamps
and durations** agree exactly with the rational grid. All **41 decoded originals**, matched
through their time-map rows, have **zero pixel error** at full 1920×1080 resolution.

Artifacts: `/tmp/syncvid-max-fps-20261028/` (not committed). Use **`export_exact/`** for the
verified final videos and audits; `validation.json` records the checks. The earlier `export/`
is retained only as evidence of the rounded-clock experiment. A separate copied checkpoint
and manifest live alongside the exports. Existing source files and results were not rewritten.
The verification script is `verify_sample.py` (uses `export_exact/`); packet checks are also
covered by the automated regression tests.

The per-camera `*.contact.png` sheets and native-pixel `*.motion_detail.png` crops were
visually reviewed around the middle event. Interpolation is visible on both cameras, with
soft/ghosted foot and ball edges; small fast-moving ball estimates can show a second faint
image. These artifacts remain limitations of the existing estimator. Original samples remain
unchanged. `gui.png` was reviewed for visible mode/ceiling/counts using dummy SDL; live desktop
interaction was not exercised.

Final focused verification:

```bash
.venv/bin/ruff check vaila/sync_events.py vaila/syncvid.py tests/test_sync_events.py
.venv/bin/ruff format --check vaila/sync_events.py vaila/syncvid.py tests/test_sync_events.py
.venv/bin/ty check vaila/sync_events.py vaila/syncvid.py vaila/cutvideo.py
.venv/bin/python -m pytest tests/test_sync_events.py tests/test_syncvid.py \
  tests/test_cutvideo.py tests/test_syncvid_repaint.py tests/test_syncvid_real_videos.py \
  tests/test_manual_video_cuts.py tests/test_gui_help.py -q
```

**172 passed, 16 skipped**; Ruff, formatting and Ty pass. Thirteen skipped tests need the
absent `tests/syncvid/` fixture bank; three help tests need an accessible Tk display. Coverage
includes shorter/longer references, decoded originals in both cameras, exact encoded packet
times, 240 rejection, inclusive configured ceilings, malformed limits, old sessions, GUI
save/resume, explicit mode selection and CLI override/persistence. The full suite was not run.
Metadata remains **0.4.12 / 28 October 2026**, including README and help indexes. Installers
require no changes: dependencies and application launch remain the same.
