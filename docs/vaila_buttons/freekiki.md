# FreeKiki Button

The **FreeKiki (49 field KPs)** button (Frame B → Soccer Tools) launches `vaila/freekiki.py`.

**Version:** 0.4.7 · **Updated:** 15 October 2026

## Overview

Trains, retrains and runs a YOLO-pose network that detects the 49 soccer-field keypoints of the
kiki template (`vaila/models/soccerfield_kiki.csv`) in broadcast video, as the first step toward
per-frame camera calibration.

## Key Features

- **Portable workspace:** dataset, runs, model registry and active model live in one folder that
  can be copied and reused for new trainings.
- **Dataset import + check:** copies a kiki49 YOLO-pose build (source read-only) and validates it
  against the 49-point schema. **Audit dataset** (read-only) adds per-point availability by
  split/source/recording, label issues, label-vs-homography outliers, cross-split leakage and
  near-duplicates.
- **Oversampling manifest:** `manifests/vNNN/` repeats train images that show rare
  keypoints (repeat-factor sampling). The dataset is not modified. **Train**
  with Manifest `v001` uses that list; val and test stay the original split.
- **Train / Retrain:** from `yolo26*-pose.pt` or from the current best (`active`); `best.pt` is
  compared with the active model on **val** and promoted only when the configurable
  `[promotion]` gate passes (pose mAP, PCK, recall/error of critical points, homography rate);
  every decision goes to `models/promotion_log.csv`. **Bench batch/VRAM** measures throughput and
  peak VRAM per batch/workers before a long run.
- **Presets:** *Smoke* (`yolo26n-pose`, 5 epochs, 10 % data, 640 px, ~5 min — pipeline check only) and
  *Full* (`yolo26m-pose`, 150 epochs, 1280 px — the real model).
- **Resume after Stop / crash / power loss:** every finished epoch is saved in
  `runs/<run>/weights/last.pt`; **Runs status** shows which runs are `running`, `resumable`,
  `finished`, `registered` or `no-checkpoint`, and **Resume interrupted** continues from the last
  completed epoch (also on another machine with a copied workspace).
- **Evaluate:** labelled split (**val** for decisions, test for the final report) → pose mAP,
  keypoint recall/precision with an explicit distance match, error percentiles, PCK over matched
  and over all labelled points (px @1920), per-point/per-source tables, failures CSV, per-image
  homography; results are appended to `models/evaluations.csv`. **Sweep thresholds (val)**
  re-scores saved predictions over a box/keypoint threshold grid. **Compare with its slot** runs the
  promotion gate against the model of the candidate's size slot without promoting.
- **Model slots:** one promoted model per network size, `models/freekiki_{n,s,m,l,x}.pt` (YOLO26-pose)
  and `freekiki_hm_*` (heatmap); `active` is the default slot (`models --default l`). An old
  `models/active.pt` workspace is migrated once by copy (kept). `grow` deepens a trained m into a
  function-preserving l initialisation; n/s/x start from `yolo26*-pose.pt`.
- **Detect:** one video or a folder; getpixelvideo-compatible CSV (`p0..p48`), confidence, raw and
  per-point status CSVs (detected / rejected with reason), optional gap filling inside a shot,
  diagnostic frames, overlay MP4, snapshot PNG and `quality.json` (detection rate, visible kps,
  valid-homography rate, cuts, camera displacement vs residual jitter); folders also get
  `quality_summary.csv`.
- **Field geometry (Geometry: fill / fix / off in section 5):**
  - A camera is fitted to the accepted points per frame: a homography; a DLT3D when flags or post
    tops are seen; otherwise a pinhole camera recovered from the homography.
  - `fill` adds missing points the network half-sees (G); `fix` also replaces points that disagree
    with the camera (Gx).
  - Output: `field_kps_geom_*.csv`. `evaluate` reports network vs fill vs fix on the same labels.
  - Per-frame calibration: `<video>.dlt2d` (ground homography, for `rec2d.py`) and
    `<video>.dlt3d` (11 DLT coefficients, for `rec3d.py`), in pitch metres. A blank row means no
    camera in that frame.
- **Mining (CLI):** `queue --need p5,p29,p39,p47 --montage` picks, across many detected videos, the
  frames whose field camera puts those points in the picture (missed by the network first), a few per
  video. It builds one review video (`montage.mp4` + `montage_frames.csv`); `ingest` then takes one
  match per source video.
- **Correct and retrain:** **Correct in getpixelvideo** (section 5), or getpixelvideo **Tpl: → L = FreeKiki
  Load** (point at the run folder; it finds the video), opens the detected video with its labels. Right-click picks a point, left-click places it, Del / Del Range = not visible, F10 accepts a
  ghost, F3 = frame OK, F9 = save dataset folder. Put that folder in section 3 **Corrections** and
  Train. Only complete frames are saved. The CLI keeps `queue`, `ingest` and `--split hard`.
- **Snap (`snap`, CLI):** moves the accepted keypoints of a detect run onto the painted lines (lines refitted to the
  paint, corners = intersections); writes `<run>/snap/` with a corrected CSV, the moves and an overlay video.
- **Detect screens the cameras:** a folder `detect` skips videos from cameras the network was not trained for
  (zoom, behind the goal, a close-up on a player: fewer than 30 % of the sampled frames have a valid pitch camera),
  lists them in `skipped_videos.csv`, and never deletes them. `--no-screen` runs everything.
- **Select frames in video (`select`):** detects on downloaded games at the normal keypoint confidence, finds the
  moments where the rare keypoints (p5, p29, p39, p47) show and cuts them with vailá Cut Video's ffmpeg cutter to a
  clips folder (`<video>_frame_<a>_to_<b>.mp4`, `selected_clips.csv`, `rejected_windows.csv`). A MediaPipe **pose
  gate** drops close-up cameras (a pose taller than 1/10 of the image) and a **camera check** drops frames whose fitted
  camera does not fit the pitch (rmse above 5 px, height outside 8-45 m: behind the net, zoom).
- **Pseudo-labelled frames (`pseudo`, CLI):** turns the detect run of the `select` clips into complete labels (network
  points + fitted camera, with camera, aux-point and line-support gates), staged for `extend --src` (preview, then
  `--commit` to train). No human reviews them; `--trust-csv` rejects frames showing keypoints the reviewers kept
  correcting. `remove-source` takes a source out of train reversibly.
- **External datasets (`extend`):** converts Roboflow Universe keypoint projects, SoccerNet-GSR and the
  rejected SoccerNet-calibration images into 49-point labels. A Roboflow project's own keypoint layout is
  aligned by network votes plus pitch geometry (`mapping.csv`, editable). Frames are staged in
  `incoming/external_*`, then `--commit` appends them to train only, dropping val/test/hard groups and
  near-duplicates.

## Usage

1. Soccer Tools → **FreeKiki (49 field KPs)**.
2. Choose a workspace → **Init workspace** → **Import (copy)** → **Check dataset**.
3. **Smoke preset** → **Train** to prove the pipeline; then **Full preset** → **Train** (hours);
   later retrain with base `active`.
   If the training stops, press **Runs status** then **Resume interrupted**.
4. **Evaluate model** on val (and **Sweep thresholds (val)** / **Compare with its slot**); keep the
   test split for the final report.
5. **Detect** on a broadcast video or a folder of videos; check `quality_summary.csv` and snapshots.
6. Corrections: **Correct in getpixelvideo** → fix → **F9 Save dataset** → section 3 **Corrections**
   → **Train** (base `active`).

## GUI ↔ CLI

Every button prints its `>>` command in the terminal (`WS` = workspace; CUDA machines use
`uv run --no-sync`):

| Button | CLI |
|---|---|
| Init workspace | `uv run vaila/freekiki.py init -w WS` |
| Import (copy) | `uv run vaila/freekiki.py import-dataset -w WS --src DATASET` |
| Check dataset | `uv run vaila/freekiki.py check -w WS` |
| Build oversampling manifest | `uv run vaila/freekiki.py manifest -w WS --rfs-t 0.05 --cap 4 --seed 0` |
| Train (Smoke / Full / Retrain) | `uv run vaila/freekiki.py train -w WS --base BASE --epochs N --imgsz PX` |
| Train on manifest | `uv run vaila/freekiki.py train -w WS --manifest v001 --base active` |
| Runs status | `uv run vaila/freekiki.py status -w WS` |
| Resume interrupted | `uv run vaila/freekiki.py resume -w WS [--name RUN]` |
| Bench batch/VRAM | `uv run vaila/freekiki.py bench -w WS --batches 2,8,16 --fraction 0.02` |
| Evaluate model | `uv run vaila/freekiki.py evaluate -w WS --split val` |
| Sweep thresholds (val) | `uv run vaila/freekiki.py sweep -w WS --eval-dir EVAL_DIR` |
| Compare with its slot | `uv run vaila/freekiki.py compare -w WS --candidate BEST.pt` |
| Model slots | `uv run vaila/freekiki.py models -w WS [--default l]` |
| Select frames in video → Plan / Select + cut | `uv run vaila/freekiki.py select -w WS --src GAMES --out CLIPS [--plan-only]` |
| External data → Stage | `uv run vaila/freekiki.py extend -w WS --source roboflow --search "class:pitch"` |
| External data → Preview / Commit | `uv run vaila/freekiki.py extend -w WS --src STAGING [--commit]` |
| (CLI only) grow m to l | `uv run vaila/freekiki.py grow -w WS --src m --to l --out models/freekiki_l_init.pt` |
| Audit dataset | `uv run vaila/freekiki.py audit -w WS` |
| Detect | `uv run vaila/freekiki.py detect -w WS --video VIDEO_OR_FOLDER` |
| Correct in getpixelvideo | `uv run vaila/getpixelvideo.py -f VIDEO --freekiki --freekiki-workspace WS --freekiki-predictions OUTPUT_DIR` |
| Train with Corrections | `uv run vaila/freekiki.py train -w WS --base active --add-dataset FOLDER` |

---
See also: [FreeKiki Help](../../vaila/help/freekiki.html), [Field KPs (AI)](soccerfield-keypoints-ai.md)
