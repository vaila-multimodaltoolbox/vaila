# Dynamic Hardware Optimization & GPU Diagnostic Guide

*vailá* is designed to adapt intelligently to the hardware it runs on. Whether you are using a high-end workstation with an RTX 4090 or a sleek laptop with limited graphic memory, the toolbox automatically configures itself for the best possible performance.

---

## 🔍 GPU & AI Stack Diagnostics (GPU Test)

vailá includes a built-in diagnostic test suite to verify whether PyTorch, CUDA, and model pipelines are operational.

### How to Run

1. **GUI**: Click the **GPU Test** button in the footer of `vaila.py` (next to **Help** and **Check for Updates**).
2. **CLI**: Run in terminal:
   ```bash
   uv run python vaila/gputest.py
   ```
3. **GUI Mode from Terminal**:
   ```bash
   uv run python vaila/gputest.py --gui
   ```

### Tested Scripts & AI Models

The diagnostic suite validates the entire AI and GPU stack across 5 target areas:

1. **PyTorch & CUDA Core**:
   - PyTorch version, CUDA runtime build version.
   - `torch.cuda.is_available()` check.
   - GPU device detection, Compute capability, and VRAM memory.
   - CUDA MatMul tensor allocation smoke test.
   - `nvidia-smi` presence and driver status.

2. **markerless2d_yolo26.py (2D Pose)**:
   - Ultralytics YOLO module and keypoint schema check.
   - Pose weights discovery (`yolo11n-pose.pt` / `yolo26*-pose.pt` in `vaila/models/`).
   - 1-frame dummy inference smoke test with device routing and latency reporting.

3. **yolov26track.py (Tracking & Markers)**:
   - Object detection and tracking runtime check.
   - Linear assignment solver check (`lap` / `lapx` / `scipy`).
   - FFmpeg availability for fast video transcode and overlay export.
   - 1-frame tracking smoke test.

4. **sam3sapiens2.py (SAM 3 + Sapiens2)**:
   - CUDA hardware availability (hard requirement).
   - SAM 3 package & checkpoint (`sam3.pt`).
   - Sapiens2 package & config (`configs/keypoints308/...`).
   - Sapiens2 1B pose weights (`sapiens2_1b_pose.safetensors`) & DETR detector.

5. **sam3dinov3.py (SAM 3 + DINOv3 3D)**:
   - CUDA hardware availability (hard requirement).
   - SAM 3 package & checkpoint.
   - SAM 3D Body checkout (`ensure_sam3d_importable()`).
   - SAM 3D DINOv3 model checkpoint (`model.ckpt`) & MHR rig (`assets/mhr_model.pt`).
   - PyTorch Lightning & Transformers stack.

---

## 🛠️ Common Fixes & Setup Commands

If the diagnostic report indicates errors or missing dependencies, apply the relevant command below:

### 1. Switch Hardware Template (Linux / Windows CUDA)
```bash
# Linux CUDA Workstation (RTX 4090 / 3090 / Ada / Blackwell)
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,fifa,sapiens --yes

# Windows CUDA Workstation
pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu,sam,fifa,sapiens -Yes
```

### 2. Sapiens2 Setup
```bash
# Clone Sapiens2 repo and download 1B weights:
bash bin/setup_sapiens2.sh
```

### 3. SAM 3D Body (FIFA) Setup
```bash
# Clone sam_3d_body repo and download DINOv3 weights:
bash bin/setup_fifa_sam3d.sh
```

### 4. Gated Hugging Face Weights Authentication
```bash
uv run hf auth login
uv run vaila/vaila_sam.py --download-weights
uv run vaila/vaila_sapiens.py --download-weights --model 1b
```

---

## How Hardware Profiles Work

The built-in **HardwareManager** scans your system at startup to detect:
1.  **Operating System**: Linux or Windows.
2.  **GPU Model**: Specifically checking for NVIDIA GPUs.
3.  **VRAM (Video Memory)**: The amount of memory available on your graphics card.

Based on this scan, it assigns one of three performance profiles:

| Profile | Hardware | Description | Optimization Settings |
| :--- | :--- | :--- | :--- |
| **ULTRA** | > 20GB VRAM | High-end cards (RTX 4090, RTX 3090, RTX 6000) | **FP16** Precision, **8GB** Workspace. Maximum speed. |
| **HIGH** | 7GB - 20GB | Mid-range & Laptops (RTX 5050, 4070, 3060) | **FP16** Precision, **2GB** Workspace. Balanced performance. |
| **LITE** | < 7GB or CPU | Entry-level GPUs or CPU-only systems | **FP32** (CPU) or minimal GPU usage. Compatibility mode. |

---

## Auto-Export & Cross-Platform Support

*vailá* uses **TensorRT** (`.engine` files) for extreme acceleration on NVIDIA GPUs. These files are hardware-specific—an engine built for an RTX 4090 will **not** work on an RTX 5050, and an engine built on Linux will **not** work on Windows.

To solve this, *vailá* implements **Auto-Export**:

1.  **Automatic Detection**: When you load a model (e.g., `yolo26x-pose`), the system checks if a `.engine` file exists for **your specific OS and GPU**.
2.  **Automatic Creation**: If the optimized file is missing, *vailá* automatically creates it from the source `.pt` model.
    *   *Example Filename*: `yolo26x-pose_NVIDIA_GeForce_RTX_4090.engine`
3.  **Coexistence**: You can keep the `vaila/models` folder on a shared drive or cloud sync. The system will simply create multiple `.engine` files side-by-side, one for each machine you use.

### First Run Notification ⚠️

**The first time** you run a new model on a new computer (or after updating drivers), the Auto-Export process will run.

*   **What happens**: The terminal will show `⚡ Auto-Exporting [model]...` and `⚙️ Building TensorRT Engine...`.
*   **Duration**: This process can take **2 to 10 minutes** depending on your hardware.
*   **Action**: **Do not close the window.** It is not frozen! It is compiling a high-performance binary for you.
*   **Subsequent Runs**: Loading will be instant (milliseconds).

---

## Windows vs. Linux

The experience is identical on both platforms.

*   **Linux**: Uses the `trtexec` binary provided by the `tensorrt` package.
*   **Windows**: Uses `trtexec.exe` provided by the installed Python package.

If you dual-boot the same machine, *vailá* will detect the OS change and generate separate engines for Windows and Linux automatically.

---

## Troubleshooting

### "GPU Not Detected"
If your report shows `Mode: LITE` or `CUDA Available: False` but you have an NVIDIA GPU:
1.  Run the **GPU Test** button in the `vaila.py` footer or run `uv run python vaila/gputest.py`.
2.  Ensure you have the latest **NVIDIA Drivers** installed (`nvidia-smi`).
3.  Ensure you activated the CUDA template (`bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,fifa,sapiens --yes`).

### "Out of Memory" (OOM)
If you get memory errors during Auto-Export on cards with limited VRAM:
*   The **HIGH** profile limits workspace to 2GB to prevent this, but other apps (browser, background processes) might be using VRAM.
*   **Solution**: Close other heavy applications before the first run of a large model.
