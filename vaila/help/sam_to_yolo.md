# sam_to_yolo

## Module Information

- **Category:** ML
- **File:** `vaila/sam_to_yolo.py`
- **Version:** 0.3.67
- **Updated:** 2026-06-29
- **GUI Interface:** Yes (Tkinter file pickers)
- **CLI Interface:** Yes

## Description

Converts a **SAM 3 video tracking export** (`sam_tracks.csv` or its
`sam_bbox_tracks.csv` alias) into a **YOLO detection dataset**: one bounding box
per tracked instance per frame, a single class (`person` by default).

This is the **correct** path when the goal is to detect and track N moving
people/objects with a YOLO detector + BoT-SORT. The per-instance IDs `1..N` are
assigned by the **tracker** at inference time — not by detector classes.

## Why this module exists

A common and costly mistake:

1. Run SAM 3 (`vaila_sam.py`) → `sam_tracks.csv` (≈N boxes per frame, one per person).
2. Load the tracks into `getpixelvideo.py` and convert each bbox to a single marker.
3. Press **F9** (pose export).

Step 3 collapses **all** instances of a frame into **one** object whose
keypoints are the markers: `nc: 1, names: ['object'], kpt_shape: [N, 3]`. A model
trained on that predicts one box + N keypoints and can never detect N separate
people — at tracking time everything appears as a single `object` class.

`sam_to_yolo` skips that detour and writes a proper detection dataset straight
from the SAM boxes.

## Label format

Each label file (`frame_NNNNNN.txt`) holds one line per instance, normalized:

```text
0 cx cy w h      # instance 1 (class 0 = person)
0 cx cy w h      # instance 2
...              # ~N lines, one per SAM track present in the frame
```

`data.yaml` is a detection set: `nc: 1`, `names: ['person']`, **no `kpt_shape`**.

## Variable number of instances

The detector + tracker do **not** require a fixed count. Each frame's label
holds exactly as many boxes as SAM tracked in that frame (7, 16, 18, ...).
There is no `--max` and nothing is hard-coded to 16.

## Class name is metadata (rename anytime)

YOLO **detection** labels store only the class **index** (`0`), never the text.
So changing `object` → `person` is a metadata edit — the thousands of `.txt`
files never change. You can pick the name at build time, rename the dataset
before training, or rename the trained weights after training.

## CLI

Build a detection dataset:

```bash
uv run python -m vaila.sam_to_yolo build \
  --sam-tracks /path/processed_sam_*/<video>/sam_tracks.csv \
  --video /path/<video>.mp4 \
  --class-name person \
  --reuse-images-dir /path/vaila_dataset_*   # optional, reuse extracted frames \
  --output /path/out_dataset                 # optional
```

(A bare `--sam-tracks ...` with no subcommand still implies `build`.)

Rename the class of an existing dataset (metadata only; labels untouched):

```bash
uv run python -m vaila.sam_to_yolo rename-dataset \
  --dataset /path/out_dataset --class-name person
# multi-class: --class-name "person, ball"
```

Rename the class baked into a trained `.pt` (no retraining):

```bash
uv run python -m vaila.sam_to_yolo rename-model \
  --weights best.pt --class-name person --output best_person.pt
```

Useful `build` flags:

- `--split-mode {temporal,random}`: **`temporal` is the default.** Frames are
  split into chronological train/val/test blocks (first frames → train, middle
  → val, last → test). On video, consecutive frames are near-duplicates, so a
  random split leaks them across train and val and inflates mAP. Use
  `--split-mode random` only if you really want the old shuffle behaviour.
- `--frame-stride N`: keep only every Nth frame (by frame index, so `10` keeps
  frames 0, 10, 20, ...). `1` (default) = full extract. Use a larger stride to
  build a smaller, less-redundant labelling set from the SAM3 tracks — handy
  when SAM3 is your "labeller" and you only need a subset to train/correct.
  Progress is shown with a banner + `tqdm` bar; frame extraction now does a
  single sequential decode pass (much faster than per-frame seeking).
- `--reuse-images-dir DIR`: hardlink already-extracted `frame_NNNNNN.*` images
  (e.g. from a previous pose-dataset export) instead of re-decoding the video.
  Zero extra disk via hardlinks; falls back to copy/symlink.
- `--min-box-px N`: drop boxes smaller than N pixels (w or h).
- `--min-score S`: drop boxes whose SAM `score` is below S.
- `--width / --height`: override frame size (skips video probing).
- `--seed N`: deterministic train/val/test split (default 70/20/10; only
  affects `--split-mode random`).

## GUI

```bash
uv run python -m vaila.sam_to_yolo
```

File pickers: SAM tracks CSV → source video → optional reuse-images dir.

## Train the resulting dataset

The task auto-detects to `detect` (there is no `kpt_shape`):

```bash
uv run python -m vaila.yolotrain \
  --data /path/out_dataset/data.yaml \
  --task detect --model yolo26x.pt --epochs 100 --imgsz 1280 --device 0
```

or directly with Ultralytics:

```bash
yolo detect train data=/path/out_dataset/data.yaml model=yolo26x.pt epochs=100 imgsz=1280
```

After training, track with `yolov26track` using your `best.pt`; BoT-SORT
assigns the persistent IDs.

## Output layout

```text
out_dataset/
├── data.yaml          # nc: 1, names: ['person'], no kpt_shape
├── classes.txt        # person
├── README_dataset.txt # provenance + ready-to-run train command
├── train/images  train/labels
├── val/images    val/labels
└── test/images   test/labels
```

---

**Last Updated:** 2026-06-29 (v0.3.67 - added --split-mode temporal, now the default)
