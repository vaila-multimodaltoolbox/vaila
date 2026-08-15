# SAM3-Guided Sapiens2 Pose Pipeline (vailá)

Use when working on **SAM3 + Sapiens2 pose estimation**, running batch CLI video pipelines (`vaila/sam3sapiens2.py`), tuning batch sizes for NVIDIA GPUs (RTX 4090 / 3090 / 5050), configuring headless GPU mode via `gpumode`, or diagnosing VRAM/throughput bottlenecks.

## Overview

`vaila/sam3sapiens2.py` is a top-down markerless pose pipeline integrating:
- **SAM3 (Segment Anything Model 3)**: Serves as the authority for person detection, tracking, silhouette extraction, and persistent `obj_id`s.
- **Sapiens2 Pose (1B / 5B / 0.8b / 0.4b ViT)**: Runs top-down on SAM3-generated crops and masks (308 body/face/hand keypoints). DETR detector is disabled.

## CLI Execution Pattern

```bash
# Standard batch run on directory:
uv run python -u vaila/sam3sapiens2.py \
    -i /path/to/videos \
    -o /path/to/output_dir \
    -t person \
    --model 1b \
    --stride 1 \
    --device 0 \
    --kpt-thr 0.3 \
    --bbox-padding 0.12 \
    --contour-margin 8 \
    --max-persons 1 \
    --pose-batch-size 16

# With Test-Time Augmentation (Flip-Test) for highest joint accuracy:
uv run python -u vaila/sam3sapiens2.py \
    -i /path/to/videos \
    -o /path/to/output_dir \
    -t person \
    --model 1b \
    --pose-batch-size 16 \
    --flip-test
```

## GPU & Batch Sizing Benchmark (RTX 4090 24GB)

Measured on Sapiens2 1B (308 keypoints, 1024x768 crop resolution):

| Batch Size | Throughput (crops/s) | VRAM Allocated | VRAM Reserved (Peak) | Safety Margin |
| :--- | :--- | :--- | :--- | :--- |
| **4** | 2.45 crops/s | 7.93 GiB | 9.57 GiB | ~14.4 GiB free |
| **8** | **2.67 crops/s** | 10.33 GiB | 13.54 GiB | ~10.5 GiB free |
| **12** | 2.66 crops/s | 12.72 GiB | 17.40 GiB | ~6.6 GiB free |
| **16** | **2.67 crops/s** | 15.11 GiB | 21.22 GiB | ~2.8 GiB free |
| **20** | 2.61 crops/s | 17.51 GiB | 20.01 GiB | ~4.0 GiB free |
| **24** | 2.58 crops/s | 18.68 GiB | **22.73 GiB** | **~1.3 GiB free** |

### Key Rules & Recommendations:
1. **Saturation:** ViT 1B saturates GPU Tensor Cores at **Batch 8 to 16** (~2.67 crops/s). Higher batch sizes (e.g. 24) do **not** increase throughput and risk CUDA OOM during long multi-video runs.
2. **Optimal Batch:** Use `--pose-batch-size 16` for high throughput with sufficient headroom.
3. **Single Athlete Videos:** In videos with 1 athlete, batch size per frame is 1 crop (`~7.19 GiB` VRAM). `--max-persons 1` prevents spurious multi-crop overhead.
4. **`--flip-test` Impact:** Doubles forward passes (`2x` compute time), negligible VRAM change, provides higher anatomical accuracy for challenging sports movements.

## Headless / CLI GPU Management (`gpumode`)

To free all VRAM from desktop compositors (Xorg/Wayland) on dedicated NVIDIA setups:
```bash
# Switch to CLI mode (multi-user.target, frees 100% VRAM):
sudo gpumode --cuda

# Return to GUI desktop:
sudo gpumode --desktop
```
