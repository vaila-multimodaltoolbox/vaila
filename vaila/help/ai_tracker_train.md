# Train AI Tracker

Version: 0.4.4  
Updated: 2026-09-18

Offline fine-tuning for *vailá* Track AI assets under `vaila/models/ai_tracker/`,
using YOLO datasets produced by **getpixelvideo → Save ML**.

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

- `vaila/models/ai_tracker/{variant}_finetuned_YYYYMMDD_HHMMSS.pth`
- `vaila/models/ai_tracker/discriminator_<profile>.npz`

Never overwrites `*_imagenet.pth` automatically. Load the finetuned `.pth` in
getpixelvideo Track AI (Deep / Cfg / browse). Discriminator profiles load via
the existing Track AI checkpoint path.

## Notes

- Pose and detect layouts are both accepted (boxes parsed from YOLO label lines).
- Optional `ai_tracker_train.json` next to `data.yaml` records task/classes.
- CUDA used when available (`--device auto`); otherwise CPU.
