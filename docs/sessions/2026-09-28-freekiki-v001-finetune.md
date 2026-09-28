# FreeKiki v001 fine-tune (handoff)

Date: 28 September 2026. Branch: `trainkiki`. Workspace is outside the git clone.

## Stopped after epoch 28 (safe to resume)

Stopped on 28 September 2026 once epoch 28 had written `weights/last.pt`
(optimizer state present; checkpoint `epoch` field 27, which is that completed
epoch in Ultralytics' index). The GPU was free afterward. `models/active.pt`
is still the previous checkpoint (sha256
`dc0464d1cce19b69ed95bc5267b5aeb8f542053dcb16ec8edcff1c4628023022`).
Images and labels under `datasets/kiki49` stay as they were. Ultralytics may
have rewritten the derived `labels/train.cache`.

Epoch 28 val: pose precision 0.968, mAP50-95 0.879.
Continue with:

```bash
cd /home/preto/data/vaila
uv run --no-sync vaila/freekiki.py resume -w /home/preto/data/FreeKiki --name kiki49_v001_ft
```

On this machine always use `uv run --no-sync`. A bare `uv run` replaces the
CUDA wheels with CPU ones.

## What is training

Repeat-factor sampling (LVIS), each keypoint as a category. Manifest
`/home/preto/data/FreeKiki/manifests/v001` (`t=0.05`, cap `4`, seed `0`).
19 174 train images become 19 899 list entries. Val and test stay the original
split. At train time the relative list is expanded to absolute paths in
`runs/<name>/sampling/`.

Command that is running:

```bash
uv run --no-sync vaila/freekiki.py train \
  -w /home/preto/data/FreeKiki \
  --manifest v001 --base active \
  --epochs 60 --imgsz 1280 --batch 8 --workers 16 --seed 0 \
  --name kiki49_v001_ft
```

`--base active` injects `ACTIVE_FINETUNE_ARGS` in `vaila/freekiki.py`:
AdamW, `lr0=1e-4`, `lrf=0.01`, `warmup_epochs=1`, `cos_lr`, `mosaic=0`,
backbone unfrozen, patience 30. A first train from `yolo26*-pose.pt` still
uses Ultralytics `optimizer=auto`.

An earlier attempt, `runs/kiki49_v001_ft_stopped_e1`, used `lr0=0.001` and
mosaic on. Epoch 1 dropped pose precision from 0.954 to 0.871 and pose
mAP50-95 from 0.857 to 0.601. That folder is archived. Do not resume it.

## Metrics (val), snapshot while epoch 26 was logged

Baseline is `runs/kiki49_20260925_222907`, epoch 150:
pose precision 0.954, recall 0.938, mAP50 0.949, mAP50-95 0.857.

| Epoch | Precision | Recall | mAP50 | mAP50-95 | val pose loss |
|---:|---:|---:|---:|---:|---:|
| 19 | 0.964 | 0.945 | 0.952 | 0.867 | |
| 22 (best mAP50-95 so far) | 0.963 | 0.946 | 0.959 | 0.875 | 0.818 |
| 25 | 0.967 | 0.954 | 0.962 | 0.872 | 0.809 |
| 26 | 0.967 | 0.949 | 0.960 | 0.874 | 0.798 |

Read the live table from
`/home/preto/data/FreeKiki/runs/kiki49_v001_ft/results.csv`.
Ultralytics `weights/best.pt` follows fitness, not the last epoch.
`active.pt` is replaced only at the end, by `register_run`, if the promotion
gate passes. Re-check the sha256 above before assuming it moved.

## After the run: videos for the user to judge

Four clips in
`/home/preto/Downloads/vaila_ytdownload_20260927_124226/aus_bra`
(`Australia_vs_Brazil_frame_3736_to_3781.mp4`, `..._4110_to_4155.mp4`,
`..._4202_to_4272.mp4`, `..._4312_to_4442.mp4`).
Do not start this while the 4090 is in the training run.

```bash
uv run --no-sync vaila/freekiki.py detect \
  -w /home/preto/data/FreeKiki \
  --model /home/preto/data/FreeKiki/runs/kiki49_v001_ft/weights/best.pt \
  --video /home/preto/Downloads/vaila_ytdownload_20260927_124226/aus_bra \
  --output-dir /home/preto/data/FreeKiki/outputs
```

GUI: Frame B, Soccer Tools, FreeKiki. Workspace
`/home/preto/data/FreeKiki`. Section 5: Folder = that `aus_bra` directory,
Output dir = `/home/preto/data/FreeKiki/outputs`, Model = the `best.pt` above,
Overlay MP4 on, Detect. The log prints the `>>` CLI line.
`quality.json` is label-free. The user judges the overlay.

## Code already changed (uncommitted)

`vaila/freekiki.py`, `vaila/help/freekiki.md`, `vaila/help/freekiki.html`,
`tests/test_freekiki.py` (fine-tune args + manifest tests). Other dirty files
on `trainkiki` (`vaila_ytdown`, yolotrain, README) are a separate change.
Do not revert them to clean the tree.
