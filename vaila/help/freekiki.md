# freekiki

## Module Information

- **Category:** Multimodal Analysis / Sports Field Calibration
- **File:** `vaila/freekiki.py`
- **Version:** 0.4.7
- **Updated:** 07 October 2026
- **Author:** Paulo Santiago — paulosantiago@usp.br
- **GUI Interface:** Yes (Tkinter) — **Frame B → Soccer Tools → FreeKiki (49 field KPs)**
- **CLI Interface:** Yes
- **License:** AGPL-3.0

---

## What it does (in one paragraph)

### Correct and retrain (3 steps)

1. **Detect**, then **Correct in getpixelvideo** (section 5). A folder of videos opens them one after the
   other.
   - Or, in getpixelvideo, press **Tpl:** and type **L** (FreeKiki Load), or use the **Load run folder** button
     of the FreeKiki panel. Open the run or batch folder in the
     browser and press **Enter** to use it. It finds the original video and opens it in correction mode; for a
     batch it asks which video. The CLI equivalent is `getpixelvideo.py --freekiki-run DIR`.
   - The predicted labels appear on the video, already in edit (INSERT) mode.
2. **Fix** every frame you want to use.
   - **Right-click** a point to pick it; **left-click** to place it where it belongs.
   - Missing points: pick them in the list or with the ghost, then left-click. **F10** accepts a ghost.
   - A point that is inside the picture but not visible: **Del** (this frame) or **Del Range** (that point over
     many frames). Points outside the picture are ignored.
   - **F3 Frame OK** when the frame is right. **Next draft (PgDn)** goes to the next frame.
   - **X / Discard** drops a bad frame (advert, wrong camera, blur, replay graphics). It loses its
     points, is never exported, and **disappears from navigation**: arrows, slider, playback and
     PgDn jump over it. **Shift+X** shows the discarded frames again (dark, labelled
     *DISCARDED*); X on one of them restores the network draft. The montage video itself is not
     cut, so the frame numbers of labels already ingested stay valid.
   - **F9 Save** (or panel **Save (F9)** button) asks whether to save **Full** (1) or **Lite** (2):
     - **Full mode (1):** saves the complete dataset with both human-reviewed frames and uncorrected AI predicted frames (images + labels) into `freekiki_full_<video>`, allowing the network to train on the entire footage.
     - **Lite mode (2):** writes only human-reviewed frames into `freekiki_corrections_<video>`; incomplete frames remain as drafts and are listed in `incomplete_frames.csv`.
3. In section 3, **Corrections → Add folder…** (several videos: several folders), then **Train**.
   - Base `active` gives a fine-tune.
   - The folders are added to the train split once, with the same leakage and duplicate checks as `ingest`, and
     then the training starts.
   - **Match id** is optional. The default is the video name; give clips of one match the same id.
   - CLI: `train -w WS --base active --add-dataset FOLDER [--add-dataset FOLDER2] [--match-id M]`.
     Do not combine it with `--manifest`, because a manifest is a fixed list without the new frames.

### Mine new videos for the missing keypoints (`queue --need`, one montage)

Use the trained network to find, in many new videos, the frames that show the
keypoints it still misses. Put them in **one** review video.

```bash
# 1. download (vailá-ready H.264, clean names), then detect every video (geometry on by default)
uv run --no-sync vaila/vaila_ytdown.py --file urls.txt --output ~/data/yt_mining --no-gui
uv run --no-sync vaila/freekiki.py detect -w WS --video ~/data/yt_mining/vaila_batch_<ts> --model l --stride 5 --no-overlay --output-dir ~/data/yt_mining/detect
# 2. frames whose field camera puts p5 / p29 / p39 / p47 in the picture -> one montage session
uv run --no-sync vaila/freekiki.py queue -w WS --batch ~/data/yt_mining/detect/processed_freekiki_batch_<ts> --need p5,p29,p39,p47 --montage
# 3. review the montage (command printed by queue), F9 saves the dataset in the montage folder
# 4. one match per source video, taken from montage_frames.csv
uv run --no-sync vaila/freekiki.py ingest -w WS --src WS/incoming/montage_<ts>            # preview
uv run --no-sync vaila/freekiki.py ingest -w WS --src WS/incoming/montage_<ts> --commit
```

How the frames are chosen. For each frame, the field camera is fitted to the
points the network accepted (see *Field geometry*). It says whether a
**needed** keypoint lies inside the picture, even when the network does not
find it. The priority:

1. camera: the point is in the picture and the network **misses** it (the most
   useful labels);
2. only with `--half-seen`: no camera, but the network half-sees the point (conf ≥ 0.1).
   It is off by default because reviewers discarded all such frames (105 of 105): adverts,
   replays and close-ups;
3. camera: the point is in the picture and the network already accepts it.

Frames without any needed point are skipped. With `--need`, at most
`--per-video` (default **5**) frames are taken per video, at least `--min-gap`
apart: many videos with few frames each give the variety the network lacks.
`--batch` takes several folders.

`--montage` writes `WS/incoming/montage_<ts>/` with:

- `montage.mp4`: H.264 with every frame a key frame, so seeking is exact;
  videos of another size are letterboxed to the most common size;
- `session.json`: the network predictions drafted on every frame;
- `montage_frames.csv`: for each montage frame, its source `video`, `frame`,
  `time_s`, letterbox `scale` / `offset`, the reason it was chosen and its
  `match`.

`match` defaults to the source video name; cuts of one video
(`…_frame_A_to_B`) share it. Edit the column before `ingest` to give clips of
the same match one id. `ingest` without `--match-id` groups each frame by that
column, so the val/test/hard leakage checks work per match. The manifest
keeps `source_video` / `source_frame`.

### Advanced (CLI): label queue, ingest, hard holdout

```bash
# 1. detect, then queue the frames worth labelling (one review session per clip)
uv run --no-sync vaila/freekiki.py detect -w WS --video /path/new_videos
uv run --no-sync vaila/freekiki.py queue  -w WS --batch /path/new_videos/processed_freekiki_batch_<ts>
# 2. review each session (the queue prints this command per clip)
uv run --no-sync vaila/getpixelvideo.py --freekiki --freekiki-workspace WS --freekiki-session WS/incoming/SESSION_ID/session.json
# 3. preview, then commit to train (or to the hard holdout)
uv run --no-sync vaila/freekiki.py ingest -w WS --src WS/incoming/SESSION_ID --match-id MATCH_ID
uv run --no-sync vaila/freekiki.py ingest -w WS --src WS/incoming/SESSION_ID --match-id MATCH_ID --commit
uv run --no-sync vaila/freekiki.py ingest -w WS --src WS/incoming/SESSION_ID --match-id HARD_MATCH --split hard --commit
# 4. check, audit, oversample, retrain; measure on val, then hard, then test once
uv run --no-sync vaila/freekiki.py check    -w WS
uv run --no-sync vaila/freekiki.py audit    -w WS
uv run --no-sync vaila/freekiki.py manifest -w WS
uv run --no-sync vaila/freekiki.py train    -w WS --manifest vNNN --base active
uv run --no-sync vaila/freekiki.py evaluate -w WS --split hard --model runs/RUN/weights/best.pt
```

**Label every visible point, or hide it.** An empty point is written as
`0 0 0` ("not visible"), and the pose loss then teaches the network that the
point is *absent*. A frame with only a few labelled points therefore makes
the missed points worse. FreeKiki checks every frame. The labelled pitch
points give a field homography, and every point whose projection falls inside
the image must be either labelled or hidden with **Del** (occluded or not
there). Frames that fail the check:

- **Mark Reviewed (F3)** refuses them and names the missing points
  (e.g. `p5 bottom_left_corner probably visible`).
- **Export (F9)** leaves them as drafts and lists them in `incomplete_frames.csv`.
- **ingest** rejects them.

A frame needs at least 4 spread pitch points before the check can run.

**Ghosts help with the missing points.** In the review window:

- A **cyan circle** is the AI position of a point the network saw with low
  confidence (below `kp_conf`). Its label shows the confidence.
- A **pink cross** is the projection of the labelled points.
- **F10** (or *Accept ghost*) copies the ghost of the selected point and moves
  to the next missing one.
- The panel line `missing: ...` turns green (`complete`) when the frame can
  be exported.

**Which frames to label.** `queue` picks up to `--per-video` (25) frames per
clip, at least `--min-gap` (5) frames apart, in this order:

1. frames that do not calibrate (homography not ok or < 4 accepted points);
2. frames where p5 / p29 / p39 / p47 are half-seen (conf 0.1–kp_conf);
3. the rest.

Only the queued frames are drafted, so **PageDown / PageUp** walks exactly
the queue. `queue.csv` in the session folder says why each frame was chosen.

**Train or hard.**

- `ingest` (default `--split train`) adds frames to training.
- `--split hard` adds them to a **labelled holdout of difficult footage**. It
  is never trained on, and you measure it with `evaluate --split hard`.
- A match lives in one split only:
  - train refuses a match of val/test/hard, and any frame that is a
    near-duplicate of their images;
  - hard refuses a match of train/val/test, and near-duplicates of train
    frames.
  - So copies of a holdout clip in another folder cannot leak into training.
- Like test, hard is for reporting. Choose thresholds and models on val.

**Before ingesting.**

- The first ingest command previews and validates every PNG/TXT pair.
- `--commit` copies the pairs and updates `manifest.csv` last.
- Use one `--match-id` for a match or sequence shared across cameras and cuts.
- Val/test stay unchanged. A hard commit only adds `hard: images/hard` to the
  workspace `data.yaml`.
- A repeated commit of an unchanged session is safe.
- `human_vs_ai.csv` describes training annotations and is not an independent
  evaluation. Keep using `evaluate --split val` and `compare` for model
  selection.

**FreeKiki** teaches a YOLO-pose network to find the **49 soccer-field
keypoints** of the *kiki* template (`vaila/models/soccerfield_kiki.csv`,
skeleton `vaila/skeletons/soccerfield_kiki49.json`) in broadcast video —
pitch-line intersections, penalty-arc intersections, goal posts and nets,
corner flags and the centre — and then finds them in your videos. Those
points are what a later step needs to calibrate the camera frame by frame;
FreeKiki already checks, per frame, whether the accepted points give a
**geometrically valid field homography** (RANSAC, inliers, reprojection
error, orientation). Compared with **Field KPs
(AI)** (32-point FIFA/Roboflow schema), FreeKiki uses the richer 49-point
schema with 3D-aware points (posts and flags with z ≠ 0).

## Quick start (5 steps)

| Step | GUI button | What happens | Time |
|---|---|---|---|
| 1 | **Init workspace** | creates the workspace folder layout | seconds |
| 2 | **Import (copy)** → **Check dataset** | copies the kiki49 dataset in and validates it | ~10 min (11 GB) |
| 3 | **Smoke preset** → **Train** | tiny training to prove everything works | ~5 min |
| 4 | **Full preset** → **Train** | the real training | many hours |
| 5 | **Evaluate model** and **Detect** | measure quality, then run on your videos | minutes |

Training stopped (Stop button, crash, power loss)? Nothing finished is lost:
press **Resume interrupted** — see *If the training stops* below.

Every GUI action prints the equivalent `>>` command in the terminal, so any
step can be repeated from the command line. The full mapping is in
*GUI button ↔ CLI command* below.

To run detection with an existing model, use **section 5 alone**: choose the
FreeKiki **Workspace** folder containing `freekiki.toml` and a promoted
model (`models/freekiki_<slot>.pt`; *Model* `active` = the default slot), choose a video or video folder, then press **Detect**.
The Workspace field in sections 1 and 5 is shared. The model file by itself
does not replace the workspace because detection also reads its settings.
Browsing to a `.pt` inside a workspace fills that Workspace field automatically.
**Init workspace**, dataset import, and training are only needed when creating
a new workspace/model.

## GUI button ↔ CLI command

Open the GUI with `uv run vaila.py` → **Frame B → Soccer Tools → FreeKiki**, or
directly with `uv run vaila/freekiki.py`. `WS` is the workspace folder.

| GUI section → button | Same thing from the terminal |
|---|---|
| 1. Workspace → **Init workspace** | `uv run vaila/freekiki.py init -w WS` |
| 2. Dataset → **Import (copy)** | `uv run vaila/freekiki.py import-dataset -w WS --src /path/kiki49_dataset` |
| 2. Dataset → **Check dataset** | `uv run vaila/freekiki.py check -w WS` |
| 3. Train → **Smoke preset** then **Train** | `uv run vaila/freekiki.py train -w WS --base yolo26n-pose.pt --epochs 5 --imgsz 640 --batch 16 --fraction 0.1` |
| 3. Train → **Full preset** then **Train** | `uv run vaila/freekiki.py train -w WS --base yolo26m-pose.pt --epochs 150 --imgsz 1280 --batch -1` |
| 3. Train → **Heatmap preset** (Backend `heatmap`) then **Train** | `uv run vaila/freekiki.py train -w WS --backend heatmap --epochs 60 --imgsz 1024 --batch 16` |
| 3. Train → Base model `active` then **Train** (retrain) | `uv run vaila/freekiki.py train -w WS --base active --epochs 60` |
| 3. Train → **Runs status** | `uv run vaila/freekiki.py status -w WS` |
| 3. Train → **Resume interrupted** | `uv run vaila/freekiki.py resume -w WS` (add `--name RUN` to choose the run) |
| 3. Train → **Build oversampling manifest** | `uv run vaila/freekiki.py manifest -w WS --rfs-t 0.05 --cap 4 --seed 0` |
| 3. Train → Manifest `v001` then **Train** | `uv run vaila/freekiki.py train -w WS --manifest v001 --base active` |
| 3. Train → **Bench batch/VRAM** | `uv run vaila/freekiki.py bench -w WS --batches 2,8,16 --fraction 0.02` |
| 4. Evaluate → **Evaluate model** | `uv run vaila/freekiki.py evaluate -w WS --split val` |
| 4. Evaluate → **Sweep thresholds (val)** | `uv run vaila/freekiki.py sweep -w WS --eval-dir outputs/processed_freekiki_eval_val_<ts>` |
| 4. Evaluate → **Compare with its slot** | `uv run vaila/freekiki.py compare -w WS --candidate runs/RUN/weights/best.pt` (baseline = the candidate's size slot) |
| 4. Evaluate → **Model slots** | `uv run vaila/freekiki.py models -w WS` |
| (CLI only) set which slot `active` means | `uv run vaila/freekiki.py models -w WS --default l` |
| (CLI only) weight overlap per size | `uv run vaila/freekiki.py sizes -w WS --src m` |
| (CLI only) deepen a trained m into an l init | `uv run vaila/freekiki.py grow -w WS --src m --to l --out models/freekiki_l_init.pt` |
| 4. Evaluate → **Audit dataset** | `uv run vaila/freekiki.py audit -w WS` |
| 5. Detect → Workspace + Video + **Detect** | `uv run vaila/freekiki.py detect -w WS --video match.mp4` (add `--fill-gaps 3` to also write the interpolated CSV) |
| 5. Detect → Workspace + **Folder** + **Detect** | `uv run vaila/freekiki.py detect -w WS --video /path/folder_of_videos` |
| 5. Detect → Video + Output dir + **Correct in getpixelvideo** | `uv run vaila/getpixelvideo.py -f VIDEO --freekiki --freekiki-workspace WS --freekiki-predictions OUTPUT_DIR` |
| 3. Train → **Corrections** folder(s) + **Train** | `uv run vaila/freekiki.py train -w WS --base active --add-dataset FOLDER [--match-id M]` |
| 3. Train → **External data** → Source + query + **Stage** | `uv run vaila/freekiki.py extend -w WS --source roboflow --search "class:pitch"` (or `--source gsr` / `soccernet-rescue`) |
| 3. Train → **External data** → staging folder + **Preview** / **Commit** | `uv run vaila/freekiki.py extend -w WS --src WS/incoming/external_<source>_<ts> [--commit]` |
| (CLI only) label queue / ingest / hard holdout | `queue -w WS --batch DIR`, `ingest -w WS --src DIR --match-id M [--split hard] [--commit]` |
| 4. Evaluate → Split `hard` + **Evaluate model** | `uv run vaila/freekiki.py evaluate -w WS --split hard` |
| **Stop** | `Ctrl+C` in the terminal |
| **Help** | opens this page |

The GUI fields map to the options of the same name (`Epochs` → `--epochs`,
`Device` → `--device`, `KP conf` → `--kp-conf`, `Stride` → `--stride`, ...).
On NVIDIA/CUDA machines write `uv run --no-sync` instead of `uv run`.

## If the training stops (Stop, crash, power loss) — resume it

**What is saved.** At the end of **every epoch** Ultralytics writes
`runs/<run>/weights/last.pt` (model **and** optimizer state, epoch number)
and one row in `runs/<run>/results.csv`. If the computer switches off in the
middle of epoch 38, epochs 1–37 are safe; only the unfinished epoch is
repeated.

**Step 1 — see what is there:** **Runs status** (or `status -w WS`) lists every
run:

```
[freekiki] smoke_n640_e5            registered     epochs 5/5 imgsz 640 base yolo26n-pose.pt
[freekiki] kiki49_20260925_222907   resumable      epochs 37/150 imgsz 1280 base yolo26m-pose.pt
[freekiki] to continue: uv run vaila/freekiki.py resume -w WS --name kiki49_20260925_222907
```

| State | Meaning | What to do |
|---|---|---|
| `running` | a FreeKiki process is training it right now | wait (do **not** resume it twice) |
| `resumable` | interrupted; `last.pt` keeps the optimizer | **Resume interrupted** |
| `finished` | training ended but was not registered (e.g. closed during the final validation) | **Resume interrupted** only registers it (no training) |
| `registered` | finished and logged in `models/registry.csv` | nothing; to improve it, retrain with Base model `active` |
| `no-checkpoint` | stopped before the first epoch ended | start **Train** again |

**Step 2 — continue:** **Resume interrupted** (or `resume -w WS`) continues
the newest interrupted run from its last completed epoch, with the same
settings (base, epochs, imgsz, learning-rate schedule). When it ends the run is
registered and promoted into its size slot (`models/freekiki_<slot>.pt`) if it is better — exactly as a
normal Train.

Good to know:

- Choose a specific run with `resume -w WS --name RUN`.
- A run started with `--manifest vNNN` resumes on the same oversampled train
  list (the manifest is re-checked against the current labels first).
- `--device 0` or `--batch 8` may be changed on resume (e.g. after an
  out-of-memory error); everything else comes from the run.
- A workspace **copied to another machine** can be resumed there: the dataset
  path and run folder are taken from the current workspace.
- **Do not** press **Train** again for an interrupted run: it would start a new
  run from epoch 1. FreeKiki refuses a Train that would overwrite a resumable or
  running run of the same name.
- A run that **finished** (all epochs, or early stop by `patience`) cannot be
  resumed — its optimizer is removed. To train more, use **Retrain**
  (Base model `active`).

## Portable workspace

Pick any folder (fast local disk recommended). Everything FreeKiki needs is
kept inside it, with relative paths in `freekiki.toml`, so the folder can be
copied to another machine and pointed at for new trainings.

```
<workspace>/
  freekiki.toml            settings + [models] slots (default = what 'active' means)
  spec/                    soccerfield_kiki.csv + soccerfield_kiki49.json snapshot
  datasets/kiki49/         imported YOLO-pose dataset (data.yaml, images, labels, ...)
  runs/<name>/             Ultralytics training runs (weights, results.csv, plots)
  models/registry.csv      every finished training and its validation pose mAP
  models/evaluations.csv   every "Evaluate" result (compare models over time)
  models/freekiki_<slot>.pt  best model per network size: n s m l x (YOLO26-pose),
                           hm_n..hm_x (heatmap backend); the default slot is 'active'
  models/active.pt         schema-1 file, kept after migration (never overwritten)
  outputs/                 evaluation reports
```

## Workflow in detail

1. **Init workspace** — creates the layout above. Safe to repeat; never deletes.
2. **Import (copy)** — copies a kiki49 dataset build (`data.yaml`, `images/`,
   `labels/`, `manifest.csv`, `keypoints_kiki49.csv`, `cameras.jsonl`,
   `reports/`) into `datasets/kiki49/`. The source is only read. Re-running
   resumes an interrupted copy. `data.yaml` `path:` is rewritten to the new
   absolute folder (and refreshed before every training).
3. **Check dataset** — `kpt_shape [49, 3]`, `flip_idx` and `kpt_names` equal
   the CSV, per-split image/label counts, label column count (5 + 147).
4. **Train** —
   - **Smoke preset**: `yolo26n-pose.pt`, 5 epochs, 10 % of the images,
     `imgsz 640`. Only proves the pipeline works; the model is weak.
   - **Full preset** (recommended real run): `yolo26m-pose.pt`, 150 epochs,
     `imgsz 1280`, `batch -1` (AutoBatch), patience 30.
   - **Retrain**: set *Base model* to `active` to fine-tune the best model so
     far (e.g. after adding new labelled clips, or on an oversampling
     manifest). That path sets AdamW, `lr0=1e-4`, `lrf=0.01` (final lr
     `1e-6`), `warmup_epochs=1`, cosine schedule and `mosaic=0`. The previous
     full run ended with mosaic off; turning mosaic back on at `lr0=0.001`
     dropped pose mAP50-95 in a single epoch. The backbone stays trainable.
     A first train from `yolo26*-pose.pt` keeps the Ultralytics
     `optimizer=auto` defaults.
   - **Build oversampling manifest**: repeat-factor sampling (LVIS) of rare
     keypoints. Writes an immutable `manifests/vNNN/` next to the dataset
     (the dataset is not modified). Default `t = 0.05`, cap `4`, seed `0`.
     A keypoint seen in a fraction `f` of train images gets repeat
     `max(1, sqrt(t / f))`; each image repeats by its rarest visible point,
     and the fraction is rounded with the seed. `--exclude FILE` drops image
     names (one per line, or a CSV with an `image` column). Put `v001` in
     the Manifest field (or `train --manifest v001`). Val and test stay the
     original split, so evaluate and the promotion gate stay comparable.
     The first training on that list may rebuild `labels/train.cache`
     (a derived file; images and labels are unchanged).

   After each run `best.pt` is logged in `models/registry.csv`, evaluated on
   **val** and compared with the model of its size slot (see *Model slots*
   and *Model promotion*); `models/freekiki_<slot>.pt` is replaced **only**
   when the configurable gate passes.
5. **Evaluate model** — measures a model on a labelled split: **val** to choose
   thresholds and compare models, **test** only for the final report. See
   *How to read the quality numbers* below.
6. **Detect** — runs the active (or any) model on **one video or a whole
   folder of videos**.

## How to read the quality numbers

### Evaluate (labelled split — the objective measure)

Use **`--split val`** to choose thresholds and compare models. Keep **test**
for the final report only: every look at test that changes a decision makes
it less independent (`sweep` refuses a test evaluation unless `--allow-test`).

Report folder `outputs/processed_freekiki_eval_<split>_<timestamp>/`:

| File | Content |
|---|---|
| `eval_summary.json` | model sha256, split, image-list sha1, thresholds, OKS sigmas, Ultralytics mAP, FreeKiki keypoint metrics, homography summary, **definitions** of every metric |
| `per_keypoint.csv` | one row per point `p0..p48` (counts, precision, recall, error percentiles, PCK, identity swaps) |
| `per_source.csv` | the same numbers per dataset source (soccernet, ts_worldcup, ...) |
| `failures.csv` | every missed / mislocalised / false point with coordinates and the index it was confused with |
| `calibration_per_image.csv` | homography status, inliers, RMSE per image (from the ground-truth-visible accepted points) and the error of the labelled points reprojected through it |

Homography has two rates. **valid** (`calib.ok_rate`) only checks that the
predicted points agree with each other (RANSAC, RMSE, spread, orientation);
a model that always outputs an average field layout can pass it. **correct**
(`calib.correct_rate`) also requires the labelled points to reproject within
10 px@1920 (median). The promotion gate uses **correct**.
| `predictions.npz` | raw predictions down to box conf 0.01 — `sweep` re-scores them offline |

How a predicted point is matched (same index only, never another index):

| Status | Meaning |
|---|---|
| **TP** | labelled (v > 0) and predicted (box ≥ det conf, kp ≥ kp conf) within `--match-px` (default 25 px@1920) |
| **MIS** | labelled and predicted, but farther than `--match-px` (counted as a miss *and* as a false point) |
| **FN** | labelled, not predicted (below a threshold or no box) — its error is **undefined**, never 0 |
| **FP** | predicted where there is no label |

| Number | Formula | Unit |
|---|---|---|
| **recall** | TP / labelled | 0–1 |
| **precision** | TP / predicted | 0–1 |
| **err_mean / median / p90 / p95** | pixel distance over TP only | px@1920 (distance × 1920 / image width) |
| **err_any_median / p90** | distance over every labelled+predicted point (TP + MIS) | px@1920 |
| **pckN_pred** | TP within N px / predicted-and-labelled (TP + MIS) | 0–1 |
| **pckN_all** | TP within N px / **all labelled** — a miss counts as a failure | 0–1 |
| **swap_mirror / rot180 / other** | a labelled point missed while another index was predicted on it (left↔right mirror, 180° field rotation, anything else) | count |
| **pose mAP50-95** | Ultralytics OKS score, uniform sigmas (the field has no human-like per-point scale) | 0–1 |

`n_ref` and `n_groups_ref` (distinct matches/recordings) give the support:
a point with n < 30 or few groups is a weak estimate. The log lists the
**hardest keypoints** with their support and swap counts.

### Choosing thresholds (val only)

`sweep` re-applies a grid of box (`--det-confs`) and keypoint (`--kp-confs`)
thresholds to the saved `predictions.npz` and writes `sweep.csv` (recall,
precision, false points, `pck*_all`, share of images with a valid and a
correct homography). It shows the trade-off; pick on val, never on test.

### Model slots (one promoted model per network size)

`freekiki.toml` keeps one promoted model per size in `[models.<slot>]`
(`file`, `backend`, `arch`, `base`, `run`, `pose_map50_95`, `pck10_all`,
`sha256`). Slots are `n s m l x` (YOLO26-pose, file
`models/freekiki_<slot>.pt`) and `hm_n`..`hm_x` (heatmap backend).
`[models] default` says which slot the name `active` means; switch it with
`models --default l`. `models` lists the slots (`*` marks the default).

```toml
[models]
default = "m"

[models.m]
file = "models/freekiki_m.pt"
backend = "yolo"
arch = "yolo26m-pose"
run = "kiki49_20260925_222907"
pose_map50_95 = 0.8576
sha256 = "..."
```

**Migration from the old `[active]` table.** The first time a schema-1
workspace is loaded, `models/active.pt` is copied to `models/freekiki_m.pt`
(the SHA-256 of the copy is checked) and the old table is kept as
`[legacy_active]`. `active.pt` stays on disk and is only read. The migration
runs once; if `freekiki_m.pt` already exists with other content it stops
with an error instead of overwriting it.

**Sizes do not mix.** The size (slot) of a checkpoint is read from the
checkpoint itself. A run competes only with the model of its own slot; an
empty slot takes its first run when its best-epoch pose mAP50-95 is above
`[promotion] new_slot_min_pose_map50_95` (0.5). `sizes` shows why weights
cannot move between sizes: an m checkpoint matches 94 % of l's parameters by
name and shape, but only 45 % of n, 57 % of s and 5 % of x, and a shape match
between different widths is not a semantic match. So n, s and x start from
the official `yolo26{n,s,x}-pose.pt`.

**Grow m → l.** m and l have the same width; l is deeper. `grow` copies every
m tensor into l, zero-pads the extra input channels of the fusion convs and
starts the new attention block as the identity, so the grown l computes the
same function as m before training (the maximum output difference on a
random image is printed and stored in the checkpoint; on the kiki49 m it was
5.5e-4 px). The grown file is only an initialisation: train it
(`train --base models/freekiki_l_init.pt`, a continued fine-tune) and it
fills the `l` slot only through the gate below.

Any base that is not an official `yolo26*-pose.pt` (a slot, a run's
`best.pt`, a grown init) is trained as a continued fine-tune: AdamW, `lr0`
1e-4 with cosine decay, one warmup epoch, `mosaic=0`.

### Model promotion (candidate vs its slot)

After `train`, the new `best.pt` is evaluated on **val** and compared with
the model of its own slot on the same images (`compare`). It is copied to
`models/freekiki_<slot>.pt` only if the gate passes; every decision (both
model hashes, reasons) is appended to `models/promotion_log.csv`. Candidate
weights are always kept in `models/`. Configure it in `freekiki.toml`:

```toml
[promotion]
mode = "gate"            # gate | map (old: pose mAP only) | never
split = "val"
pck_px = 10
max_map_drop = 0.0       # pose mAP50-95 may not drop
max_pck_all_drop = 0.0
critical_kps = [0, 5, 13, 16, 24, 29, 32, 33, 40, 41, 48]
min_support = 30         # critical points with fewer labels are reported, not gated
max_recall_drop = 0.02
max_median_err_increase_px = 1.0
max_calib_correct_drop = 0.01   # valid AND <= 10 px@1920 vs labels
```

`best.pt` itself is chosen by Ultralytics during training: the epoch with the
highest **pose mAP50-95 + box mAP50-95** (first maximum), not pose mAP alone.

### Dataset audit

`audit` never modifies the dataset. It writes `kp_availability.csv` (labels
per point, split, source and recording group), `label_issues.csv`
(visibility, out-of-range, duplicated points, mirrored labels),
`label_outliers.csv` (label vs the homography fitted to the other labels),
`label_missing_suspects.csv` (points labelled "not visible" whose projection
falls inside the image; usable as `manifest --exclude` — judge that on val),
`leakage_groups.csv` (recordings present in more than one split) and
`near_duplicates.csv` (dHash across splits).

### Detect (your videos — no labels, so indirect indicators)

These are **not accuracy**: without labels a confident, stable point can
still be the wrong landmark. Each video gets `quality.json`; a folder run
also writes `quality_summary.csv` comparing all videos.

| Number | Meaning |
|---|---|
| **detection_rate** | frames with a field box ≥ *Conf* |
| **mean_visible_kps** | accepted points (box ≥ *Conf* and kp ≥ *KP conf*) per frame |
| **min4_kps_rate** | frames with ≥ 4 accepted points — only the *minimum count*, not a valid calibration |
| **calib_ok_rate** / **calib_status** | (no labels here, so plausibility, not correctness) frames whose accepted ground points give a valid homography: ≥ 6 RANSAC inliers (8 px@1920), RMSE ≤ 6 px@1920, field spread ≥ 1 m on the minor axis, not mirrored. Otherwise `few_points`, `degenerate`, `ransac_fail`, `few_inliers`, `high_error`, `mirrored` |
| **cuts** | shot changes found (colour-histogram jump); temporal measures never cross a cut |
| **displacement_median/p90_px** | frame-to-frame movement of the accepted points — mostly **camera motion** |
| **residual_median/p90_px** | movement left after removing the camera motion (homography, or translation when too few points) — the keypoint **jitter** |
| **point_codes** | counts of the per-point status codes below |

Also look at `diag_frames/` (green accepted, red rejected with name,
confidence and code) and the overlay MP4 — wrong points are obvious there.

## Detect outputs

Written to `freekiki_predict_<video>_<timestamp>/` inside the output folder
(default: the video's folder). A folder of videos writes
`processed_freekiki_batch_<timestamp>/` with one such sub-folder per video.
Older runs named `processed_freekiki_<video>_<timestamp>/` still open.

| File | Content |
|---|---|
| `field_kps_getpixelvideo.csv` | `frame,p0_x,p0_y,...,p48_x,p48_y` (0-based, pixels; accepted points only, blank otherwise) — loads in getpixelvideo |
| `field_kps_conf.csv` | per-keypoint confidence `p0_conf..p48_conf` |
| `field_kps_raw.csv` | best instance down to box conf 0.01, no filtering (for re-analysis) |
| `field_kps_status.csv` | per frame: cut flag, homography status/inliers/RMSE and one code per point |
| `field_kps_filled_getpixelvideo.csv` / `_source.csv` | only with `--fill-gaps N`: gaps ≤ N frames inside one shot linearly interpolated, each point marked `D` or `I` |
| `frame_issues.csv` | frames with a cut, no box or no valid homography |
| `diag_frames/` | a few annotated frames (`--diag-frames`) |
| `<video>_freekiki_overlay.mp4` | keypoints + kiki49 skeleton lines (optional) |
| `<video>_freekiki_snapshot.png` | middle frame with keypoints (quick look) |
| `quality.json` | the detect indicators above |
| `README.txt` | parameters used and the status codes |

Point status codes: **D** detected (box ≥ *Conf*, kp ≥ *KP conf*); **N** no
field instance; **Rb** rejected, box below *Conf*; **Rk** rejected, keypoint
below *KP conf*; **Ro** outside the image; **Rd** same pixel (< 3 px@1920) as
a higher-confidence index; **I** interpolated (filled CSV only). Gap filling
never crosses a cut and never extrapolates past the first/last detection.
Geometry codes (geometry CSVs only): **G** filled by the field camera, **Gx**
replaced by it, **Dg** accepted but far from it.

## Field geometry (fill / fix) — AI + mathematics

The pitch is a known 3D object (`vaila/models/soccerfield_kiki.csv`, metres,
z up). In every frame FreeKiki fits a **camera** to the keypoints the network
accepted, then predicts where every other keypoint must be.

The camera:

- **Planar homography** of the z = 0 points (RANSAC).
- With ≥ 7 points including ≥ 2 off the ground (flag tops z = 1.5 m, goal-post
  tops z = 2.44 m): a **DLT3D** of 11 parameters, the same as `dlt3d.py`. It is
  pruned of points that disagree and checked for square pixels, no skew and
  the camera above the pitch.
- Otherwise a **pinhole camera recovered from the homography**:
  - closed-form focal length (Zhang), refined on the points;
  - principal point at the image centre, square pixels.
  - This is what places flags and post tops when only line intersections are
    seen.

What each mode does (`detect --geom`, GUI section 5 *Geometry*):

| mode | effect |
|---|---|
| `fill` (default) | A point the network did not accept is **filled** (code **G**) when its projection is inside the image **and** the network still half-sees it (raw conf ≥ 0.05, so occluded points are not invented). It keeps the network position when it agrees with the camera (≤ 15 px@1920). |
| `fix` | Like `fill`, plus an accepted point more than 20 px@1920 from its projection is **replaced** (**Gx**). In `fill` mode it is only flagged (**Dg**). |
| `off` | Network only. |

Outputs:

- `field_kps_geom_getpixelvideo.csv` holds the network points plus geometry.
- `field_kps_geom_status.csv` has, per frame: the camera source
  (dlt3d / homography / planar / none), the focal length, the camera height,
  the fit RMSE and one code per point.
- `quality.json` gets the `geom_*` keys: camera rate, fills and fixes per
  point.
- The network CSVs are never changed.
- A geometry point says where the keypoint **must be**; it may be occluded by a
  player.

### Per-frame DLT calibration for rec2d / rec3d (tracking players)

With geometry on, every detect run also writes the field calibration of each
processed frame:

- **`<video>.dlt2d`**: columns `frame,p1..p8`. The ground-plane homography
  (pitch metres → pixels, H[2,2] = 1) for `rec2d.py`. Use it for **positions on
  the pitch**: feet / ground contact of pose keypoints, or the bottom-centre of
  detection boxes.
- **`<video>.dlt3d`**: columns `frame,p1..p11`, the full camera (P[2,3] = 1)
  for `rec3d.py`. 3D reconstruction needs **≥ 2 cameras** of the same instant;
  with one broadcast camera, use the DLT2D for ground positions.
- The world frame is in metres, with the origin at the centre spot, x towards
  the right goal, y towards the "top" touchline and z up
  (`soccerfield_kiki.csv`).
- A blank row means there was no camera in that frame. `rec2d` / `rec3d` skip
  it.
- Run detect with `--stride 1` so every frame of the tracking has a row; the
  `frame` column is the 0-based video frame, the same as getpixelvideo and the
  tracking CSVs.

```bash
uv run --no-sync vaila/freekiki.py detect -w WS --video match.mp4 --model l --stride 1 --output-dir OUT
uv run --no-sync vaila/rec2d.py --dlt-file OUT/freekiki_predict_match_<ts>/match.dlt2d --input-dir TRACK_CSV_DIR --output-dir REC_DIR --rate 25
```

Each frame is calibrated on its own. The camera quality is in
`field_kps_geom_status.csv` (`camera`, `rmse_px`, `cam_height_m`): the camera
height must stay nearly constant within one match. Expect small frame-to-frame
jitter, so smooth the reconstructed tracks if needed.

Limits:

- Geometry needs a camera, so ≥ 6 consistent pitch points. In frames where the
  network finds only 2–3 points it cannot help; a better network is still
  needed there.
- `evaluate` scores the **same labels** three ways (network, fill, fix). It
  writes `per_keypoint_geom_fill.csv` / `_fix.csv` and an
  `eval_summary.json["geometry"]` block.
- Choose `--geom-min-conf` / `--geom-fix-px` on **val**, never on test.
- `audit` also flags flag and post-top labels far from the camera of the other
  labels (`label_outliers.csv`, `kind=geom3d`).
- In review (getpixelvideo), flags and post tops get camera ghosts (F10
  accepts).

## External datasets (super-dataset): `extend`

More labelled frames without labelling by hand. External soccer-field datasets
are converted to the 49 FreeKiki points and appended to **train** only (val,
test and hard stay frozen, so models remain comparable).

| Source | What it brings | How the 49 points are made |
|---|---|---|
| `roboflow` | Roboflow Universe keypoint projects (`--search "class:pitch"` or `--project ws/proj[@ver]`; key `ROBOFLOW_API_KEY` in the environment or the gitignored `.env`) | Each project uses its own keypoint layout. The FreeKiki network predicts the 49 points on up to 150 images and every project keypoint **votes** for the nearest accepted prediction (12 px@1920, at least 10 votes, 70 % agreement, one-to-one). Points the network rarely accepts (the corners) are mapped by **geometry**: the mapped points fit a homography and the click goes back to the pitch (within 1.5 m). The result is `mapping.csv` (editable; reused next time). Square "Stretch" exports are resized back to 16:9. |
| `gsr` | SoccerNet Game State Reconstruction (200 clips) | Human pitch-line annotations, like SoccerNet calibration. 1 frame per second, plus every 5th frame when a rare point (p5/p16/p29/p39/p47) is in view. |
| `soccernet-rescue` | SoccerNet calibration train images the original build rejected (too few lines) | The network's points anchor the plane; the **human lines must agree** (median ≤ 3 px@1280) and the projected field must sit on painted lines. |

Every label keeps the human points (annotated) and fills the other points in
view from the fitted plane / camera; projected labels must pass the
line-support gate. Conversion code: `vaila/kiki49_build/` (ported from
mkvis3d `openbiomech/soccer_field`).

```bash
W=/path/FreeKiki
uv run --no-sync vaila/freekiki.py extend -w $W --source roboflow --search "class:pitch" --search "soccer field keypoints"
uv run --no-sync vaila/freekiki.py extend -w $W --source gsr
uv run --no-sync vaila/freekiki.py extend -w $W --source soccernet-rescue
# each writes incoming/external_<source>_<ts>/ (images, labels, rows.csv, report.md, preview/)
uv run --no-sync vaila/freekiki.py extend -w $W --src $W/incoming/external_roboflow_<ts>            # preview
uv run --no-sync vaila/freekiki.py extend -w $W --src $W/incoming/external_roboflow_<ts> --commit   # append to train
```

The commit drops (and counts) frames whose group belongs to val/test/hard,
near-duplicates of val/test/hard images, of train images or of another staged
frame, and images already registered; re-running it is safe. Then build a new
manifest and retrain. GUI: section 3, row **External data** (Source, query or
`ws/proj@ver` entries separated by `;`, **Stage**, staging folder, **Preview**,
**Commit**).

## CLI

```bash
uv run vaila/freekiki.py init     -w /path/FreeKiki
uv run vaila/freekiki.py import-dataset -w /path/FreeKiki --src /path/kiki49_dataset
uv run vaila/freekiki.py check    -w /path/FreeKiki
uv run vaila/freekiki.py manifest -w /path/FreeKiki --rfs-t 0.05 --cap 4 --seed 0   # oversample rare points; no training
uv run vaila/freekiki.py train    -w /path/FreeKiki --manifest v001 --base active     # train on that list
uv run vaila/freekiki.py train    -w /path/FreeKiki --base yolo26n-pose.pt --epochs 5 --imgsz 640 --batch 16 --fraction 0.1   # smoke
uv run vaila/freekiki.py train    -w /path/FreeKiki --base yolo26m-pose.pt --epochs 150 --imgsz 1280                           # full
uv run vaila/freekiki.py train    -w /path/FreeKiki --base active --epochs 60                                                  # retrain
uv run vaila/freekiki.py train    -w /path/FreeKiki --backend heatmap --epochs 1 --fraction 0.01                              # heatmap pilot
uv run vaila/freekiki.py train    -w /path/FreeKiki --backend heatmap --epochs 60 --imgsz 1024 --batch 16                     # heatmap full
uv run vaila/freekiki.py status   -w /path/FreeKiki                        # runs, which can be resumed
uv run vaila/freekiki.py resume   -w /path/FreeKiki [--name RUN]           # continue after stop / power loss
uv run vaila/freekiki.py evaluate -w /path/FreeKiki --split val            # choose / compare on val
uv run vaila/freekiki.py sweep    -w /path/FreeKiki --eval-dir /path/FreeKiki/outputs/processed_freekiki_eval_val_<ts>
uv run vaila/freekiki.py compare  -w /path/FreeKiki --candidate /path/FreeKiki/runs/RUN/weights/best.pt   # vs its slot
uv run vaila/freekiki.py models   -w /path/FreeKiki [--default l]          # model slots; 'active' = default
uv run vaila/freekiki.py sizes    -w /path/FreeKiki --src m                # weight overlap with n/s/m/l/x
uv run vaila/freekiki.py grow     -w /path/FreeKiki --src m --to l --out models/freekiki_l_init.pt
uv run vaila/freekiki.py audit    -w /path/FreeKiki                        # read-only dataset audit
uv run vaila/freekiki.py bench    -w /path/FreeKiki --batches 2,8,16 --workers 8 --fraction 0.02
uv run vaila/freekiki.py evaluate -w /path/FreeKiki --split test           # final report only
uv run vaila/freekiki.py detect   -w /path/FreeKiki --video match.mp4 --stride 5
uv run vaila/freekiki.py detect   -w /path/FreeKiki --video /path/folder_of_videos
uv run vaila/freekiki.py queue    -w /path/FreeKiki --batch /path/processed_freekiki_batch_<ts> --per-video 25 --min-gap 5
uv run vaila/freekiki.py queue    -w /path/FreeKiki --batch BATCH1 BATCH2 --need p5,p29,p39,p47 --montage   # one review video
uv run vaila/freekiki.py ingest   -w /path/FreeKiki --src /path/FreeKiki/incoming/SESSION --match-id MATCH [--split hard] [--commit]
uv run vaila/freekiki.py evaluate -w /path/FreeKiki --split hard
uv run vaila/freekiki.py extend   -w /path/FreeKiki --source roboflow --search "class:pitch"   # external data -> staging
uv run vaila/freekiki.py extend   -w /path/FreeKiki --src /path/FreeKiki/incoming/external_roboflow_<ts> --commit
uv run vaila/freekiki.py detect   -w /path/FreeKiki --video match.mp4 --geom fix      # fill (default) | fix | off
uv run vaila/freekiki.py evaluate -w /path/FreeKiki --split val --geom-min-conf 0.1   # network vs fill vs fix           # labelled holdout, report only
```

On NVIDIA/CUDA machines use `uv run --no-sync`.

Useful options: `manifest --rfs-t --cap --seed --exclude FILE --name`;
`train --manifest v001 --batch --device 0 --name --patience --seed --workers
--backend {yolo,heatmap} --backbone resnet50 --no-pretrained --lr`;
`resume --name --device --batch`;
`evaluate --model PATH.pt --split val --det-conf --kp-conf --match-px --pck 5,10,25 --max-images --origin 1`
(`--origin`: extra table `per_keypoint_origin_<digits>.csv` that scores only labels of that provenance, 1 annotated,
2 plane-projected, 3 camera-projected; a projected label may be occluded, so it is not held against the network);
`compare --baseline SLOT --promote` (copies to `models/freekiki_<slot>.pt` only if the gate passes);
`detect --model PATH.pt --start --max-frames --conf --kp-conf --imgsz
--no-overlay --output-dir --fill-gaps N --diag-frames N`.

Evaluate and Detect use the **image size the model was trained with**
(read from the `.pt`: 640 for the smoke preset, 1280 for the full preset).
`--imgsz` overrides it from the CLI; the GUI always uses the model's size.

## Base models

Official Ultralytics `yolo26{n,s,m,l,x}-pose.pt` (cached in `vaila/models/`
or downloaded by Ultralytics). Any `.pt` pose model can be used as base,
e.g. a public 32-keypoint pitch model; only its backbone/neck transfer
because the 49-keypoint head is re-initialised.

`--base` also accepts a slot name (`active`, `m`, `l`, `freekiki_l`, ...).
A trained m does not seed n, s or x (different widths, see *Model slots*);
the only size change that keeps the trained weights is `grow` m → l.

## Heatmap backend (vailá-native, no company-owned framework)

`train --backend heatmap` trains a second kind of network on the **same
labels** (`datasets/kiki49`, no rebuild): a torchvision ResNet trunk
(default `resnet50`, ImageNet weights from the local torch cache — nothing is
downloaded) + 3 deconvolutions + one 49-channel heatmap head (stride 4,
Gaussian targets σ = 2). The pitch is one instance per frame, so no box head
is needed. Code: `vaila/freekiki_heatmap.py`, only PyTorch/torchvision
(BSD-3-Clause) — the model and its training loop are part of *vailá*
(AGPL-3.0) and do not depend on Ultralytics or its enterprise license.

| Item | Value |
|---|---|
| Input | 16:9 letterbox, `--imgsz` = canvas width (1024 → 1024×576) |
| Augmentation | scale ±20 %, rotation ±5°, shift ±5 %, h-flip with `flip_idx`, brightness/contrast |
| Loss / optimizer | MSE on heatmaps (v=0 points have an empty target), AdamW + warmup + cosine, bf16 on CUDA |
| Decode | arg-max + log-parabola sub-cell refinement, then the inverse letterbox to frame pixels; conf = heatmap peak (0..1) |
| Per epoch | `results.csv` (loss, val recall/precision/median error, `val/pck10_all` on ≤ 500 val images), `weights/last.pt` (resumable), `weights/best.pt` |
| `best.pt` | epoch with the highest val PCK10 **with misses** (not mAP) |
| Resume | `resume -w WS [--name RUN]`, same as YOLO (same train subset for `--fraction`) |
| `--base` | only a heatmap checkpoint (continued fine-tune); a `yolo*.pt` base is ignored |

**Never promoted automatically.** A heatmap run is registered
(`backend = heatmap` in `models/registry.csv`) but no slot file is
touched, even in a new workspace. Compare it with the active YOLO model on
**val** — the FreeKiki keypoint tables (recall, precision, PCK with misses,
median error, critical points) are computed the same way for both
backends; pose mAP exists only for YOLO:

```bash
uv run --no-sync vaila/freekiki.py compare -w WS --baseline active \
    --candidate models/kiki49_<RUN>.pt              # add --promote only after reading it
                                                    # (--promote fills its hm_* slot)
```

`evaluate`, `detect`, `compare` and the getpixelvideo **Predict** button load
either kind of `.pt` (the backend is read from the checkpoint). A heatmap
model has a fixed input size, so `--imgsz` is ignored for it.

## See also

- `vaila/help/soccerfield_keypoints_ai.html` — 32-point Field KPs (AI)
- `vaila/help/soccerfield_calib.html` — DLT2D field calibration
- `vaila/help/yolotrain.html` — generic YOLO training used by FreeKiki
- `docs/fifa_workflow.md`


FreeKiki correction: Delete marks only the selected keypoint in the current frame as not visible (Undo supported). Save the session with F4 or close normally. Reopen the original video and Load the original detect CSV/folder, or the corrected dataset/session.json: saved corrections, hidden points, review states and the last frame/keypoint are restored automatically. F9 exports reviewed frames; F4 also preserves unfinished drafts.
