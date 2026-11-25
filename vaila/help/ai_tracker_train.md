# Train AI Tracker

Version: 0.4.19  
Updated: 24 November 2026

Offline fine-tuning for *vailá* Track AI assets, using YOLO datasets produced by **getpixelvideo → Save ML**, or the points you clicked while AI Track was on (`--from-samples`, see below). The assets live in the AI Track folder (`<home>/ai_tracker/`; see [ai_tracker_models](ai_tracker_models.md)). If no folder was chosen, the old `vaila/models/ai_tracker/` is used.

## Goal

Improve tracking quality by:

1. Fine-tuning a backbone `.pth` (ResNet50 / ResNet152 / MobileNetV3 / EfficientNet-B0)
   with multi-class crops from your labeled dataset.
2. Training an offline appearance **discriminator** (`discriminator_<profile>.npz`)
   with more samples and time than the online JIT path.

## Workflow

1. In **getpixelvideo**, mark targets (`p0`, `p1`, …).
2. Click **Rename** (or **Ctrl+N**): map slots to COCO-80 or custom names
   (e.g. `person`, `sports ball`). Option 5 sets the pose object class.
3. Click **Save ML**:
   - `1 = pose` — markers become keypoints (one object class).
   - `2 = detect` — each marker becomes a class box (uses Track AI block size when available).
4. Open main GUI → **Markerless 2D** → **Train AI Tracker**.
5. Select the dataset `data.yaml`, pick a base `.pth` from `ai_tracker/`, choose
   mode **backbone** / **discriminator** / **both**, Run.

The same dataset also trains Ultralytics via **Train YOLOv26**.

## CLI

```bash
uv run --no-sync python -m vaila.ai_tracker_train \
  --data /path/to/vaila_dataset_*/data.yaml \
  --weights vaila/models/ai_tracker/resnet50_imagenet.pth \
  --mode both --epochs 10 --profile athletics
```

Outputs:

- `<home>/ai_tracker/backbones/{variant}_finetuned_YYYYMMDD_HHMMSS.pth`
- `<home>/ai_tracker/discriminators/discriminator_<profile>__<key>.npz`

Never overwrites `*_imagenet.pth` automatically. Load the finetuned `.pth` in
getpixelvideo Track AI (Deep / Cfg / browse). Discriminator profiles load via
the existing Track AI checkpoint path.

## Retrain from your own clicks (`--from-samples`)

AI Track keeps the points you click while it is on (only mouse clicks) as samples in
`<home>/ai_tracker/samples/`. `--from-samples ROOT` (ROOT = the `<home>/ai_tracker` folder)
retrains from them. The **Retrain (GPU)** button of the Net window in getpixelvideo runs the
same command.

```bash
uv run --no-sync python -m vaila.ai_tracker_train \
  --from-samples ~/vaila_ai_models/ai_tracker --mode both --device auto
uv run --no-sync python -m vaila.ai_tracker_train --from-samples ~/vaila_ai_models/ai_tracker --dry-run
```

- **Backbone:** a marker / background classifier on the context crops. By default only the last
  stage plus a new 2-class head is trained (`--unfreeze last`); `--unfreeze all` trains the whole
  network. Defaults: 5 epochs, batch 32, learning rate 1e-4. Needs at least 40 training samples
  (`--min-samples`).
- **Hold-out and activation rule:** with samples from two or more videos, whole videos are held
  out (about `--val-fraction`, default 0.2); with one video, the last block of frames is. The
  measure is the held-out ROC AUC of a discriminator trained on the train split with the live
  tracker features, with the old backbone (before) and the new one (after). The new backbone is
  saved as `backbones/<variant>_finetuned_<time>.pth` and becomes active in `manifest.json`
  only if after >= before. Otherwise it stays on disk, the old backbone stays active and the
  status is `kept_previous`.
- **Discriminators:** rebuilt on all samples with the same features the live tracker uses (808
  classic values, plus the deep embedding of the active backbone), so AI Track loads them. This
  fixes the old offline file of 2048 values that the live tracker never loaded. A classic (808)
  one is written too.
- **Device rule:** `--device auto` (default) uses CUDA only if at least 3 GiB are free, else the
  CPU. `cpu` and `cuda` force one.
- **Result:** `manifest.json` gets `last_retrain` (status `activated`, `kept_previous` or
  `error`, mode, variant, device, number of samples and videos, metric before and after, reason,
  log). The log is `<ROOT>/logs/retrain_<time>.log`. Exit code 0 = finished, activated or not.
- Other options: `--variant`, `--weights`, `--profile`, `--seed`, `--ai-models DIR`.

Next step: send the result to Hugging Face or restore a version, see [ai_tracker_models](ai_tracker_models.md).

## Notes

- Pose and detect layouts are both accepted (boxes parsed from YOLO label lines).
- Optional `ai_tracker_train.json` next to `data.yaml` records task/classes.
- CUDA used when available (`--device auto`); otherwise CPU.
