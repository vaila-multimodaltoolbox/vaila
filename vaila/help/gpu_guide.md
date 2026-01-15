# Dynamic Hardware Optimization Guide

*vailá* is designed to adapt intelligently to the hardware it runs on. Whether you are using a high-end workstation with an RTX 4090 or a sleek laptop with limited graphic memory, the toolbox automatically configures itself for the best possible performance.

## How it Works

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

## Troubleshooting

### "GPU Not Detected"
If your report shows `Mode: LITE` but you have an NVIDIA GPU:
1.  Ensure you have the latest **NVIDIA Drivers** installed.
2.  On Windows, ensure you selected "NVIDIA GPU" during installation or ran the installer with the GPU option.
3.  On Linux, ensure `nvidia-smi` works in your terminal.

### "Out of Memory" (OOM)
If you get memory errors during Auto-Export on cards with exactly 8GB VRAM (like RTX 3070/4060):
*   The **HIGH** profile limits workspace to 2GB to prevent this, but other apps (browser, game) might be using VRAM.
*   **Solution**: Close other heavy applications before the first run of a big model (like YOLOv8x/v26x).
