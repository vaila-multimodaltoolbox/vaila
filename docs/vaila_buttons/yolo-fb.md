# YOLO + FB Chooser — Button B_r4_c1

## Overview

**Button Position:** B_r4_c1  
**Method Name:** `yolotrackerpose` (alias `yolo_and_sam`)  
**Button Text:** YOLO + FB  
**Dialog Title:** Select YOLO / Meta (FB) Tool

Opens a chooser for Ultralytics YOLOv26 tools and Meta (Facebook) video AI: SAM 3, Sapiens2 Pose, the combined SAM3-guided Sapiens2 2D pose pipeline, and its SAM 3D Body (DINOv3) markerless 3D counterpart.

## GUI → CLI mirror

Every chooser button prints a **launcher** command to the main vailá terminal (`>> Equivalent launch CLI`).  
Each tool's **Run** button prints the **full** copy-paste command with your paths and flags in that tool's terminal.

| Chooser button | Launcher CLI |
|----------------|--------------|
| Tracker (v26) | `uv run python -u -m vaila.yolov26track` |
| Pose (video) | `uv run python -u -m vaila.yolov26track` (in-process from main GUI; params after Run) |
| Pose (tracking) | `uv run python -u -m vaila.yolov26track` (step 1: `track` CLI; step 2: GUI) |
| Seg (v26) | `uv run python -u -m vaila.yolov26track` (pick `-seg.pt` + seg run mode) |
| SAM 3 video | `uv run python -u vaila/vaila_sam.py` |
| Sapiens2 Pose | `uv run python -u vaila/vaila_sapiens.py` |
| Train YOLOv26 | `uv run python -u -m vaila.yolotrain` |
| SAM3+Sapiens2 | `uv run python -u vaila/sam3sapiens2.py` |
| SAM3+Sapiens2 Visualize ID | `uv run python -u vaila/sam3sapiens2_visualize.py` |
| SAM3+DINOv3 3D | `uv run python -u vaila/sam3dinov3.py` |

### Full Run examples (printed automatically)

- **Tracker:** `uv run python -m vaila.yolov26track track --model ... --source VIDEO.mp4 --output OUT/ ...`
- **SAM 3:** `uv run vaila/vaila_sam.py -i ... -o ... -t person ...`
- **Sapiens2:** `uv run vaila/vaila_sapiens.py -i ... -o ... --model 1b ...` → one `processed_sapiens_<timestamp>/` (v0.3.76)
- **SAM3+Sapiens2 Visualize ID:** `uv run python -u vaila/sam3sapiens2_visualize.py --sam-results ... --video ... --id N --output ...` (`--id` optional → interactive prompt)
- **SAM3+DINOv3 3D:** `uv run python -u vaila/sam3dinov3.py -i ... -o ... -t person [--sam-results ...] [--focal-px F] [--save-mesh]` → markerless 3D MHR mesh + metric joints
- **Train:** `uv run python -m vaila.yolotrain --data data.yaml --task detect ...`

## Related help

- [vaila_sam.md](../../vaila/help/vaila_sam.md) — SAM 3 video
- [vaila_sapiens.md](../../vaila/help/vaila_sapiens.md) — Sapiens2 Pose
- [sam3sapiens2.md](../../vaila/help/sam3sapiens2.md) — SAM3+Sapiens2 pipeline
- [sam3sapiens2_visualize.md](../../vaila/help/sam3sapiens2_visualize.md) — selected-ID CPU rerenderer
- [sam3dinov3.md](../../vaila/help/sam3dinov3.md) — SAM3+DINOv3 3D markerless 3D pipeline
- [yolov26track.md](../../vaila/help/yolov26track.md) — YOLO tracking / pose
- [yolotrain.md](../../vaila/help/yolotrain.md) — YOLO training

---

**Last Updated:** 01 August 2026 (v0.3.92)  
**Part of vailá - Multimodal Toolbox**  
**License:** AGPLv3.0
