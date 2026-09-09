# Environment, Dependencies & AI Models Inspector (`vaila_env.py`)

## Module information

- **Category:** Environment & Tools
- **Version:** 0.3.131
- **Updated:** 2026-09-09
- **GUI:** Bottom Toolbar → **imagination!** (or Help menu)
- **CLI:** Yes (`uv run vaila/vaila_env.py` or `uv run vaila.py --env-info`)

## Purpose

The `vaila_env.py` module provides a comprehensive diagnostic, dependency audit, and workflow assistant:

1. **vailá Application Version & Git Status:** Displays global version (`v0.3.131`), update date (`09 September 2026`), git branch, commit hash, and working tree cleanliness.
2. **Hardware GPU Discovery & PyTorch CUDA:** Uses direct `nvidia-smi` hardware querying to reliably detect physical NVIDIA GPUs (e.g. RTX 4090, VRAM, driver version) even when PyTorch encounters library issues, while also validating the PyTorch CUDA runtime.
3. **Hugging Face Hub Integration:** Audits `huggingface_hub` package version, token configuration, authenticated username (`whoami()`), and local model cache directory.
4. **AI Tracking Models & Checkpoints Inventory:** Scans local weights in `vaila/models/` for:
   - **Meta SAM 3:** `sam3/sam3.pt`, `model.safetensors`, `config.json`.
   - **Meta SAM-3D-DINOv3:** `sam-3d-dinov3/model.ckpt`, `model_config.yaml` (FIFA Skeletal Tracking Light).
   - **Meta Sapiens2:** Pose checkpoints (`0.4B`, `1B`, `5B`) and DETR detector weights (`detr-resnet-101-dc5`).
   - **Google MediaPipe:** Pose heavy/lite tasks, face mesh, hand tracking, athlete face cropper, and selfie segmenters.
   - **Ultralytics YOLO:** Pose checkpoints (`yolo26*-pose.pt`), detection/segmentation weights (`yolo26*.pt`, `best.pt`), exported ONNX models, and TensorRT engines.
   - **Re-ID & Kinematics:** OSNet models (`osnet_x0_25_msmt17.pt/.onnx`) and gait models.
5. **Core Scientific Dependencies:** Audits 17 core libraries (including `ezc3d`, `c3d`, `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `torchvision`, `ultralytics`, `cv2`, `mediapipe`, `huggingface_hub`, `open3d`, `pyvista`, `dask`, `rich`, `tqdm`).
6. **Virtual Environment Detection:** Identifies `.venv` active status and root path, providing copyable activation commands for Linux, macOS, Windows PowerShell, and Windows CMD.
7. **IPython Quickstart & Cheatsheet:** Details how to launch interactive IPython sessions inside `.venv` for custom biomechanical calculations and C3D/CSV scripting.
8. **Direct Terminal Launcher:** Opens a system terminal in the project root with `.venv` pre-activated or launches IPython directly.

## GUI

Click the **imagination!** button on the bottom toolbar of `vaila.py`:
- **Tab 1: Installed Packages & System:** Package versions table with C3D status, OS, Python executable, and GPU hardware.
- **Tab 2: AI Models & Hugging Face:** Hugging Face authentication card and structured inventory of downloaded AI tracking checkpoints with sizes and status.
- **Tab 3: Terminal & IPython Guide:** Monospace guide with copyable terminal commands and interactive C3D snippets.
- **Action Buttons:** Copy Activation Command, Copy Full Report, Open Terminal (.venv), Launch IPython in Terminal.

## CLI

```bash
# Print complete environment, GPU, Hugging Face, and AI models report
uv run vaila/vaila_env.py

# Or via main application flag
uv run vaila.py --env-info
```

