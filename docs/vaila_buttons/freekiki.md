# FreeKiki Button

The **FreeKiki (49 field KPs)** button (Frame B → Soccer Tools) launches `vaila/freekiki.py`.

## Overview

Trains, retrains and runs a YOLO-pose network that detects the 49 soccer-field keypoints of the
kiki template (`vaila/models/soccerfield_kiki.csv`) in broadcast video, as the first step toward
per-frame camera calibration.

## Key Features

- **Portable workspace:** dataset, runs, model registry and active model live in one folder that
  can be copied and reused for new trainings.
- **Dataset import + check:** copies a kiki49 YOLO-pose build (source read-only) and validates it
  against the 49-point schema.
- **Train / Retrain:** from `yolo26*-pose.pt` or from the current best (`active`); the active model
  is only replaced when validation pose mAP50-95 improves.
- **Presets:** *Smoke* (`yolo26n-pose`, 5 epochs, 10 % data, 640 px, ~5 min — pipeline check only) and
  *Full* (`yolo26m-pose`, 150 epochs, 1280 px — the real model).
- **Resume after Stop / crash / power loss:** every finished epoch is saved in
  `runs/<run>/weights/last.pt`; **Runs status** shows which runs are `running`, `resumable`,
  `finished`, `registered` or `no-checkpoint`, and **Resume interrupted** continues from the last
  completed epoch (also on another machine with a copied workspace).
- **Evaluate:** labelled test split → pose mAP, keypoint recall/precision, median pixel error and
  PCK10/25 (px @1920), per-keypoint table; every result is appended to `models/evaluations.csv`.
- **Detect:** one video or a folder; getpixelvideo-compatible CSV (`p0..p48`), confidence CSV,
  overlay MP4, snapshot PNG and `quality.json` (detection rate, visible kps, calibration-ready rate,
  jitter); folders also get `quality_summary.csv`.

## Usage

1. Soccer Tools → **FreeKiki (49 field KPs)**.
2. Choose a workspace → **Init workspace** → **Import (copy)** → **Check dataset**.
3. **Smoke preset** → **Train** to prove the pipeline; then **Full preset** → **Train** (hours);
   later retrain with base `active`.
   If the training stops, press **Runs status** then **Resume interrupted**.
4. **Evaluate model** to measure quality on the labelled test split.
5. **Detect** on a broadcast video or a folder of videos; check `quality_summary.csv` and snapshots.

## GUI ↔ CLI

Every button prints its `>>` command in the terminal (`WS` = workspace; CUDA machines use
`uv run --no-sync`):

| Button | CLI |
|---|---|
| Init workspace | `uv run vaila/freekiki.py init -w WS` |
| Import (copy) | `uv run vaila/freekiki.py import-dataset -w WS --src DATASET` |
| Check dataset | `uv run vaila/freekiki.py check -w WS` |
| Train (Smoke / Full / Retrain) | `uv run vaila/freekiki.py train -w WS --base BASE --epochs N --imgsz PX` |
| Runs status | `uv run vaila/freekiki.py status -w WS` |
| Resume interrupted | `uv run vaila/freekiki.py resume -w WS [--name RUN]` |
| Evaluate model | `uv run vaila/freekiki.py evaluate -w WS` |
| Detect | `uv run vaila/freekiki.py detect -w WS --video VIDEO_OR_FOLDER` |

---
See also: [FreeKiki Help](../../vaila/help/freekiki.html), [Field KPs (AI)](soccerfield-keypoints-ai.md)
