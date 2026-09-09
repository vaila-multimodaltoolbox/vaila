"""
===============================================================================
vaila_env.py
===============================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 September 2026
Update Date: 09 September 2026
Version: 0.3.131

Description:
Environment, dependencies, and AI tracking models inspector for vailá.
Audits vailá version and update date, hardware acceleration (NVIDIA GPU via
nvidia-smi and PyTorch CUDA), Hugging Face Hub authentication and cache status,
local AI tracking weights (SAM 3, SAM-3D-DINOv3, Sapiens2, MediaPipe, YOLO),
and core scientific packages (ezc3d, c3d, etc.). Provides platform-specific
virtual environment (.venv) activation commands and interactive IPython coding
guidance for biomechanics.

Features:
- Audits vailá application version, update date, and git repository status.
- Hardware GPU discovery: physical NVIDIA GPU (VRAM, driver) via nvidia-smi with
  PyTorch CUDA runtime validation.
- Hugging Face Hub diagnostic: package version, token configuration, authenticated
  user (whoami), and cache storage location.
- Comprehensive AI Models & Weights inventory:
  - Meta SAM 3 (sam3.pt, model.safetensors, config.json)
  - Meta SAM-3D-DINOv3 (FIFA Skeletal Tracking Light: model.ckpt, model_config.yaml)
  - Meta Sapiens2 (0.4B, 1B, 5B pose weights & DETR person detector)
  - Google MediaPipe (pose heavy/lite, face, hand, athlete face cropper, selfie segmenters)
  - Ultralytics YOLO (pose, detection, segmentation, ONNX, and TensorRT engines)
  - Re-ID & Gait Kinematics (OSNet and gait models)
- Inspects core and optional library versions: ezc3d, c3d, numpy, pandas, scipy,
  matplotlib, torch, torchvision, ultralytics, cv2, mediapipe, huggingface_hub, etc.
- Displays OS and platform details with virtual environment status (.venv).
- Provides IPython launch commands and interactive biomechanics math snippets.
- Multi-tab interactive GUI dialog (Packages, AI Models & HF, IPython Guide) + CLI.

Usage:
    CLI:
        uv run vaila/vaila_env.py
        python -m vaila.vaila_env

    GUI:
        Launched via the "imagination!" button in vaila.py or run_vaila_env_gui().
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

# Global application version and update date (matching vaila.py)
VAILA_VERSION = "0.3.131"
VAILA_UPDATE_DATE = "09 September 2026"

# Key packages tracked in vailá
CORE_PACKAGES: list[tuple[str, str]] = [
    ("ezc3d", "C3D Biomechanics I/O (C++/SWIG bindings, pyomeca)"),
    ("c3d", "C3D Legacy Reader (Pure-Python)"),
    ("numpy", "Core numerical computing"),
    ("pandas", "Tabular data & CSV processing"),
    ("scipy", "Scientific computing & signal filtering"),
    ("matplotlib", "Plotting & 2D visualization"),
    ("torch", "PyTorch deep learning framework"),
    ("torchvision", "Computer vision models & transforms"),
    ("ultralytics", "YOLO object & pose tracking"),
    ("cv2", "OpenCV video & image processing"),
    ("mediapipe", "MediaPipe markerless body/face tracking"),
    ("huggingface_hub", "Hugging Face Model Hub Client & Token Manager"),
    ("open3d", "Open3D interactive 3D point cloud & C3D viewer"),
    ("pyvista", "PyVista 3D mesh & surface rendering"),
    ("dask", "Parallel data processing for large files"),
    ("rich", "Terminal formatting & highlights"),
    ("tqdm", "Progress bars for batch processing"),
]


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable units (B, KB, MB, GB)."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.1f} {units[i]}"


def get_vaila_info() -> dict[str, Any]:
    """Retrieve vailá version, update date, and git repository status."""
    branch = "unknown"
    commit = "unknown"
    is_dirty = False
    root_dir = Path(__file__).resolve().parent.parent

    try:
        res_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res_branch.returncode == 0 and res_branch.stdout.strip():
            branch = res_branch.stdout.strip()

        res_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res_commit.returncode == 0 and res_commit.stdout.strip():
            commit = res_commit.stdout.strip()

        res_status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if res_status.returncode == 0:
            is_dirty = bool(res_status.stdout.strip())
    except Exception:
        pass

    return {
        "version": VAILA_VERSION,
        "update_date": VAILA_UPDATE_DATE,
        "branch": branch,
        "commit": commit,
        "is_dirty": is_dirty,
        "root_dir": str(root_dir),
    }


def get_hardware_info() -> dict[str, Any]:
    """Detect physical GPUs via nvidia-smi and check PyTorch CUDA runtime."""
    smi_available = bool(shutil.which("nvidia-smi"))
    smi_gpus: list[dict[str, str]] = []
    if smi_available:
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if res.returncode == 0:
                for line in res.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        smi_gpus.append(
                            {
                                "name": parts[0],
                                "memory": parts[1],
                                "driver": parts[2],
                            }
                        )
        except Exception:
            pass

    torch_cuda_available = False
    torch_cuda_device = "None"
    torch_cuda_error: str | None = None

    try:
        import torch

        if torch.cuda.is_available():
            torch_cuda_available = True
            torch_cuda_device = torch.cuda.get_device_name(0)
    except Exception as e:
        torch_cuda_error = str(e)

    # Physical GPU summary string
    if smi_gpus:
        gpu_summary = ", ".join(
            f"{g['name']} ({g['memory']}, Driver {g['driver']})" for g in smi_gpus
        )
    elif torch_cuda_available:
        gpu_summary = torch_cuda_device
    else:
        gpu_summary = "None (CPU only)"

    return {
        "nvidia_smi_available": smi_available,
        "smi_gpus": smi_gpus,
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_device": torch_cuda_device,
        "torch_cuda_error": torch_cuda_error,
        "gpu_summary": gpu_summary,
    }


def get_huggingface_info() -> dict[str, Any]:
    """Inspect Hugging Face Hub library, authentication token, and logged-in user."""
    hf_ver, is_installed = get_package_version("huggingface_hub")
    has_token = False
    username = "Not authenticated"
    cache_dir = "Default (~/.cache/huggingface)"
    error_msg: str | None = None

    if is_installed:
        try:
            import huggingface_hub

            token = huggingface_hub.get_token()
            has_token = bool(token)
            if hasattr(huggingface_hub, "constants") and hasattr(
                huggingface_hub.constants, "HF_HOME"
            ):
                cache_dir = str(huggingface_hub.constants.HF_HOME)

            if has_token:
                try:
                    user_info = huggingface_hub.whoami()
                    username = user_info.get("name") or user_info.get(
                        "username", "Authenticated user"
                    )
                except Exception as e:
                    username = f"Token present (offline/whoami: {e})"
        except Exception as e:
            error_msg = str(e)

    return {
        "installed": is_installed,
        "version": hf_ver,
        "has_token": has_token,
        "username": username,
        "cache_dir": cache_dir,
        "error": error_msg,
    }


def get_models_dir() -> Path:
    """Return the resolved path to vaila/models directory."""
    root_dir = Path(__file__).resolve().parent.parent
    candidate = root_dir / "vaila" / "models"
    if candidate.is_dir():
        return candidate
    candidate_root = root_dir / "models"
    if candidate_root.is_dir():
        return candidate_root
    return candidate


def get_ai_models_status() -> dict[str, Any]:
    """Scan and return status of all AI models in vaila/models/."""
    models_dir = get_models_dir()

    def check_file(rel_path: str, label: str, required: bool = True) -> dict[str, Any]:
        p = models_dir / rel_path
        exists = p.is_file()
        size = p.stat().st_size if exists else 0
        return {
            "name": label,
            "rel_path": rel_path,
            "abs_path": str(p),
            "exists": exists,
            "size": size,
            "size_str": format_size(size) if exists else "0 B",
            "status": "Ready" if exists else ("Missing" if required else "Optional / Absent"),
        }

    # 1. SAM 3
    sam3_items = [
        check_file("sam3/sam3.pt", "sam3.pt (SAM3 checkpoint)", required=True),
        check_file("sam3/model.safetensors", "model.safetensors (Weights)", required=False),
        check_file("sam3/config.json", "config.json (Model config)", required=False),
    ]

    # 2. SAM-3D-DINOv3
    sam3d_items = [
        check_file("sam-3d-dinov3/model.ckpt", "model.ckpt (SAM-3D-Body DINOv3)", required=True),
        check_file("sam-3d-dinov3/model_config.yaml", "model_config.yaml (Config)", required=False),
    ]

    # 3. Sapiens2
    detr_safetensors = check_file(
        "sapiens2/detector/detr-resnet-101-dc5/model.safetensors",
        "DETR detector (model.safetensors)",
        required=False,
    )
    detr_bin = check_file(
        "sapiens2/detector/detr-resnet-101-dc5/pytorch_model.bin",
        "DETR detector (pytorch_model.bin)",
        required=False,
    )
    sapiens_items = [
        check_file(
            "sapiens2/pose/sapiens2_0.4b_pose.safetensors",
            "sapiens2_0.4b_pose.safetensors (0.4B Pose)",
            required=False,
        ),
        check_file(
            "sapiens2/pose/sapiens2_1b_pose.safetensors",
            "sapiens2_1b_pose.safetensors (1B Pose Default)",
            required=True,
        ),
        check_file(
            "sapiens2/pose/sapiens2_5b_pose.safetensors",
            "sapiens2_5b_pose.safetensors (5B Pose High-Res)",
            required=False,
        ),
        detr_safetensors,
        detr_bin,
    ]

    # 4. MediaPipe
    face_det_rel = (
        "crop_face/face_detector.task"
        if (models_dir / "crop_face" / "face_detector.task").is_file()
        else "face_detector.task"
    )
    mediapipe_items = [
        check_file(
            "pose_landmarker_heavy.task", "pose_landmarker_heavy.task (Pose Heavy)", required=True
        ),
        check_file(
            "pose_landmarker_lite.task", "pose_landmarker_lite.task (Pose Lite)", required=False
        ),
        check_file("face_landmarker.task", "face_landmarker.task (Face Mesh)", required=False),
        check_file("hand_landmarker.task", "hand_landmarker.task (Hands)", required=False),
        check_file(face_det_rel, "face_detector.task (Athlete Face Cropper)", required=False),
        check_file(
            "image_segmenter_selfie_landscape.tflite",
            "selfie_landscape.tflite (Segmenter)",
            required=False,
        ),
        check_file(
            "image_segmenter_selfie_square.tflite",
            "selfie_square.tflite (Segmenter)",
            required=False,
        ),
    ]

    # 5. Ultralytics YOLO
    yolo_items: list[dict[str, Any]] = []
    known_yolo_pts = [
        ("yolo26x-pose.pt", "YOLOv26 Extra-Large Pose (Keypoints)"),
        ("yolo26l-pose.pt", "YOLOv26 Large Pose"),
        ("yolo26m-pose.pt", "YOLOv26 Medium Pose"),
        ("yolo26n-pose.pt", "YOLOv26 Nano Pose"),
        ("yolo26x.pt", "YOLOv26 Extra-Large Object Detection"),
        ("yolo26m.pt", "YOLOv26 Medium Object Detection"),
        ("yolo26n.pt", "YOLOv26 Nano Object Detection"),
        ("yolo11n.pt", "YOLO11 Nano Detection"),
        ("best.pt", "best.pt (Custom Fine-tuned Weights)"),
        ("yolo26x-seg.pt", "YOLOv26 Extra-Large Segmentation"),
    ]
    for filename, desc in known_yolo_pts:
        yolo_items.append(check_file(filename, f"{filename} - {desc}", required=False))

    if models_dir.is_dir():
        for onnx_f in sorted(models_dir.glob("*.onnx")):
            rel = onnx_f.relative_to(models_dir).as_posix()
            yolo_items.append(
                {
                    "name": f"{onnx_f.name} (ONNX Export)",
                    "rel_path": rel,
                    "abs_path": str(onnx_f),
                    "exists": True,
                    "size": onnx_f.stat().st_size,
                    "size_str": format_size(onnx_f.stat().st_size),
                    "status": "Ready",
                }
            )
        for eng_f in sorted(models_dir.glob("*.engine")):
            if "broken" in eng_f.name:
                continue
            rel = eng_f.relative_to(models_dir).as_posix()
            sz = eng_f.stat().st_size
            yolo_items.append(
                {
                    "name": f"{eng_f.name} (TensorRT Engine)",
                    "rel_path": rel,
                    "abs_path": str(eng_f),
                    "exists": True,
                    "size": sz,
                    "size_str": format_size(sz),
                    "status": "Ready",
                }
            )

    # 6. Re-ID & Kinematic Models
    reid_items = [
        check_file(
            "osnet_x0_25_msmt17.pt", "osnet_x0_25_msmt17.pt (OSNet Re-ID PyTorch)", required=False
        ),
        check_file(
            "osnet_x0_25_msmt17.onnx", "osnet_x0_25_msmt17.onnx (OSNet Re-ID ONNX)", required=False
        ),
    ]

    return {
        "models_dir": str(models_dir),
        "sam3": {
            "title": "SAM 3 - Meta Segment Anything Model 3",
            "items": sam3_items,
            "has_any": any(it["exists"] for it in sam3_items),
        },
        "sam_3d_dinov3": {
            "title": "SAM-3D-DINOv3 - FIFA Skeletal Tracking Light",
            "items": sam3d_items,
            "has_any": any(it["exists"] for it in sam3d_items),
        },
        "sapiens2": {
            "title": "Sapiens2 - Meta Pose & DETR Detector",
            "items": sapiens_items,
            "has_any": any(it["exists"] for it in sapiens_items),
        },
        "mediapipe": {
            "title": "Google MediaPipe Landmark & Segmentation Tasks",
            "items": mediapipe_items,
            "has_any": any(it["exists"] for it in mediapipe_items),
        },
        "yolo": {
            "title": "Ultralytics YOLO (Pose, Detection, ONNX & TensorRT)",
            "items": yolo_items,
            "has_any": any(it["exists"] for it in yolo_items),
        },
        "reid": {
            "title": "Re-ID & Walkway Gait Kinematic Models",
            "items": reid_items,
            "has_any": any(it["exists"] for it in reid_items),
        },
    }


def get_package_version(pkg_name: str) -> tuple[str, bool]:
    """Return (version_string, is_installed)."""
    # Special aliases
    dist_name = "opencv-python" if pkg_name == "cv2" else pkg_name
    try:
        ver = importlib.metadata.version(dist_name)
        return ver, True
    except importlib.metadata.PackageNotFoundError:
        try:
            mod = importlib.import_module(pkg_name)
            ver = getattr(mod, "__version__", "Installed (version unknown)")
            return str(ver), True
        except ImportError:
            return "Not installed", False


def get_system_info() -> dict[str, Any]:
    """Collect OS, Python, venv, and GPU details."""
    in_venv = sys.prefix != sys.base_prefix
    venv_path = sys.prefix if in_venv else None
    hw = get_hardware_info()

    has_cuda = hw["torch_cuda_available"] or bool(hw["smi_gpus"])
    gpu_name = hw["gpu_summary"]

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "python_path": sys.executable,
        "in_venv": in_venv,
        "venv_path": venv_path,
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
        "hardware": hw,
    }


def get_activation_command() -> str:
    """Return platform-specific command to activate .venv."""
    current_os = platform.system()
    if current_os == "Windows":
        return r".\.venv\Scripts\Activate.ps1"
    return "source .venv/bin/activate"


def get_ipython_guide() -> str:
    """Return helpful instructions and cheat sheet for IPython biomechanics coding."""
    cmd = get_activation_command()
    return f"""# 1. Activate vailá virtual environment:
# {cmd}

# 2. Launch interactive IPython session:
ipython
# or directly via uv without manual activation:
# uv run ipython

# 3. Quick Biomechanical Calculations in IPython:
import ezc3d, numpy as np, pandas as pd

# Load and inspect any C3D file:
c = ezc3d.c3d("tests/C3D_to_CSV/C3D_to_CSV_01.c3d")
pts = c["data"]["points"]  # shape: (4, num_markers, num_frames)
print(f"Loaded {{pts.shape[1]}} markers across {{pts.shape[2]}} frames")

# Calculate euclidean trajectory or speed of marker 0:
marker_xyz = pts[0:3, 0, :]  # (3, frames)
diffs = np.diff(marker_xyz, axis=1)
disp = np.linalg.norm(diffs, axis=0)
fps = c["parameters"]["POINT"]["RATE"]["value"][0]
velocity = disp * fps  # in mm/s or m/s
print(f"Mean marker velocity: {{np.nanmean(velocity):.2f}} units/s")
"""


def format_env_summary() -> str:
    """Return a formatted string report for terminal output."""
    vaila_info = get_vaila_info()
    sys_info = get_system_info()
    hw_info = sys_info.get("hardware", get_hardware_info())
    hf_info = get_huggingface_info()
    models_info = get_ai_models_status()

    # Hardware GPU details
    gpu_disp = hw_info["gpu_summary"]
    if hw_info["torch_cuda_available"]:
        cuda_runtime_str = f"Ready (Device: {hw_info['torch_cuda_device']})"
    elif hw_info["torch_cuda_error"]:
        # Extract first line of error
        err_first_line = hw_info["torch_cuda_error"].strip().split("\n")[0]
        cuda_runtime_str = f"Library error: {err_first_line} (CPU runtime fallback)"
    else:
        cuda_runtime_str = "CPU runtime active"

    git_str = (
        f"{vaila_info['branch']} ({vaila_info['commit']}{' *' if vaila_info['is_dirty'] else ''})"
    )

    lines = [
        "================================================================================",
        " vailá - Environment, Dependencies & AI Models Inspector",
        f" Version: {vaila_info['version']} | Update Date: {vaila_info['update_date']} | Git: {git_str}",
        "================================================================================",
        f"Operating System:     {sys_info['os']} {sys_info['os_release']} ({sys_info['arch']})",
        f"Python Version:       {sys_info['python_version']} ({sys_info['python_path']})",
        f"Virtual Env (.venv):  {'Active' if sys_info['in_venv'] else 'Not active'} ({sys_info['venv_path'] or 'None'})",
        f"Hardware GPU:         {gpu_disp}",
        f"PyTorch CUDA:         {cuda_runtime_str}",
        "--------------------------------------------------------------------------------",
        "Hugging Face Hub Status:",
        "--------------------------------------------------------------------------------",
        f"  * Package Version:  v{hf_info['version'] if hf_info['installed'] else '[Not installed]'}",
        f"  * Token Configured: {'Yes (Active)' if hf_info['has_token'] else 'No token detected'}",
        f"  * Authenticated As: {hf_info['username']}",
        f"  * Local Cache Dir:  {hf_info['cache_dir']}",
        "--------------------------------------------------------------------------------",
        "Installed Core Packages & C3D Library Status:",
        "--------------------------------------------------------------------------------",
    ]

    for pkg, desc in CORE_PACKAGES:
        ver, installed = get_package_version(pkg)
        status = f"v{ver}" if installed else "[Not Installed]"
        extra = ""
        if pkg == "ezc3d" and installed:
            extra = " (Latest version: 1.7.2)"
        lines.append(f"  * {pkg:<16} {status:<18} - {desc}{extra}")

    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            f"AI Tracking Models & Checkpoints Inventory ({models_info['models_dir']}):",
            "--------------------------------------------------------------------------------",
        ]
    )

    # Add each AI model family
    families = [
        ("sam3", models_info["sam3"]),
        ("sam_3d_dinov3", models_info["sam_3d_dinov3"]),
        ("sapiens2", models_info["sapiens2"]),
        ("mediapipe", models_info["mediapipe"]),
        ("yolo", models_info["yolo"]),
        ("reid", models_info["reid"]),
    ]

    for _, fam in families:
        items = fam.get("items", [])
        if not items:
            continue
        lines.append(f"  [{fam['title']}]")
        for it in items:
            status_tag = f"[{it['status']}]"
            lines.append(f"    * {it['name']:<46} {it['size_str']:<10} {status_tag}")
        lines.append("")

    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            "How to activate .venv on your system:",
            "--------------------------------------------------------------------------------",
            f"  Current OS ({sys_info['os']}):",
            f"    {get_activation_command()}",
            "",
            "  Windows PowerShell:    .\\.venv\\Scripts\\Activate.ps1",
            "  Windows CMD:           .\\.venv\\Scripts\\activate.bat",
            "  Linux / macOS:         source .venv/bin/activate",
            "--------------------------------------------------------------------------------",
            "How to run IPython for calculations and coding:",
            "--------------------------------------------------------------------------------",
            "  Command:  ipython  (or: uv run ipython)",
            "",
            "Example interactive snippet:",
            '  import ezc3d, numpy as np; c = ezc3d.c3d("file.c3d"); print(c["parameters"]["POINT"]["RATE"]["value"])',
            "================================================================================",
        ]
    )
    return "\n".join(lines)


class VailaEnvGUI(tk.Toplevel):
    """Interactive GUI dialog showing environment, libraries, and IPython guide."""

    def __init__(self, parent: tk.Tk | tk.Toplevel | None = None):
        super().__init__(parent)
        self.title("vailá - Imagination & Environment Info")
        self.geometry("920x720")
        self.minsize(780, 560)

        self._create_widgets()

    def _create_widgets(self) -> None:
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Arial", 12, "bold"))
        style.configure("SubHeader.TLabel", font=("Arial", 9))
        style.configure("Section.TLabel", font=("Arial", 10, "bold"))

        # Top banner
        top_frame = ttk.Frame(self, padding=10)
        top_frame.pack(fill=tk.X)

        v_info = get_vaila_info()
        sys_info = get_system_info()
        hw_info = sys_info.get("hardware", get_hardware_info())
        hf_info = get_huggingface_info()

        ttk.Label(
            top_frame,
            text=f"vailá Multimodal Toolbox  v{v_info['version']} ({v_info['update_date']})",
            style="Header.TLabel",
        ).pack(anchor="w")

        info_str = (
            f"Git: {v_info['branch']} ({v_info['commit']})  |  "
            f"OS: {sys_info['os']} {sys_info['os_release']} ({sys_info['arch']})  |  "
            f"Python: {sys_info['python_version']}  |  "
            f"GPU: {hw_info['gpu_summary']}"
        )
        ttk.Label(top_frame, text=info_str, style="SubHeader.TLabel", foreground="#555555").pack(
            anchor="w", pady=(2, 0)
        )

        hf_sub_str = (
            f"Hugging Face: {hf_info['username']} "
            f"({'Token Present' if hf_info['has_token'] else 'No Token'})  |  "
            f"Cache: {hf_info['cache_dir']}"
        )
        ttk.Label(top_frame, text=hf_sub_str, style="SubHeader.TLabel", foreground="#0066aa").pack(
            anchor="w", pady=(1, 0)
        )

        # Notebook with 3 tabs:
        # Tab 1: Packages & System
        # Tab 2: AI Models & Hugging Face
        # Tab 3: Terminal & IPython Guide
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tab_pkgs = ttk.Frame(notebook, padding=10)
        notebook.add(tab_pkgs, text="Installed Packages & System")
        self._build_packages_tab(tab_pkgs)

        tab_models = ttk.Frame(notebook, padding=10)
        notebook.add(tab_models, text="AI Models & Hugging Face")
        self._build_models_tab(tab_models)

        tab_guide = ttk.Frame(notebook, padding=10)
        notebook.add(tab_guide, text="Terminal & IPython Guide")
        self._build_guide_tab(tab_guide)

        # Bottom Frame: Action buttons
        bot_frame = ttk.Frame(self, padding=10)
        bot_frame.pack(fill=tk.X)

        btn_term = ttk.Button(bot_frame, text="Open Terminal (.venv)", command=self.open_terminal)
        btn_term.pack(side=tk.LEFT, padx=5)

        btn_ipython = ttk.Button(
            bot_frame, text="Launch IPython in Terminal", command=self.launch_ipython
        )
        btn_ipython.pack(side=tk.LEFT, padx=5)

        btn_copy = ttk.Button(
            bot_frame, text="Copy Activation Command", command=self.copy_activation
        )
        btn_copy.pack(side=tk.LEFT, padx=5)

        btn_copy_rep = ttk.Button(bot_frame, text="Copy Full Report", command=self.copy_summary)
        btn_copy_rep.pack(side=tk.LEFT, padx=5)

        btn_close = ttk.Button(bot_frame, text="Close", command=self.destroy)
        btn_close.pack(side=tk.RIGHT, padx=5)

    def _build_packages_tab(self, parent: ttk.Frame) -> None:
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("package", "version", "description")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=14)
        tree.heading("package", text="Package")
        tree.heading("version", text="Installed Version")
        tree.heading("description", text="Description / Purpose")

        tree.column("package", width=140, anchor="w")
        tree.column("version", width=140, anchor="center")
        tree.column("description", width=500, anchor="w")

        for pkg, desc in CORE_PACKAGES:
            ver, installed = get_package_version(pkg)
            v_text = f"v{ver}" if installed else "Not installed"
            if pkg == "ezc3d" and installed:
                v_text += " (Latest)"
            tree.insert("", tk.END, values=(pkg, v_text, desc))

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # System details bottom panel
        sys_info = get_system_info()
        hw_info = sys_info.get("hardware", get_hardware_info())
        cuda_err = (
            f" (CUDA error: {hw_info['torch_cuda_error']})" if hw_info["torch_cuda_error"] else ""
        )
        detail_txt = (
            f"Python: {sys_info['python_path']}\n"
            f"Virtualenv: {sys_info['venv_path'] or 'None'}\n"
            f"Hardware: {hw_info['gpu_summary']}{cuda_err}"
        )
        ttk.Label(parent, text=detail_txt, font=("Arial", 8), foreground="#555555").pack(
            anchor="w", pady=(8, 0)
        )

    def _build_models_tab(self, parent: ttk.Frame) -> None:
        # HF top frame
        hf_info = get_huggingface_info()
        hf_frame = ttk.LabelFrame(parent, text="Hugging Face Hub Status", padding=8)
        hf_frame.pack(fill=tk.X, pady=(0, 10))

        hf_text = (
            f"Authenticated User: {hf_info['username']}   |   "
            f"Token: {'Present & Active' if hf_info['has_token'] else 'Not Found'}   |   "
            f"Cache Dir: {hf_info['cache_dir']}"
        )
        ttk.Label(hf_frame, text=hf_text, font=("Arial", 9)).pack(anchor="w")

        # Models treeview
        models_frame = ttk.LabelFrame(parent, text="AI Tracking Models & Checkpoints", padding=8)
        models_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("category", "model", "status", "size", "rel_path")
        tree = ttk.Treeview(models_frame, columns=columns, show="headings", height=14)
        tree.heading("category", text="Architecture / Family")
        tree.heading("model", text="Model / Checkpoint")
        tree.heading("status", text="Status")
        tree.heading("size", text="Size")
        tree.heading("rel_path", text="Relative Path (vaila/models/)")

        tree.column("category", width=180, anchor="w")
        tree.column("model", width=260, anchor="w")
        tree.column("status", width=90, anchor="center")
        tree.column("size", width=80, anchor="center")
        tree.column("rel_path", width=220, anchor="w")

        models_info = get_ai_models_status()
        families = [
            ("SAM 3", models_info["sam3"]),
            ("SAM-3D-DINOv3", models_info["sam_3d_dinov3"]),
            ("Sapiens2", models_info["sapiens2"]),
            ("MediaPipe", models_info["mediapipe"]),
            ("Ultralytics YOLO", models_info["yolo"]),
            ("Re-ID / Gait", models_info["reid"]),
        ]

        for fam_name, fam in families:
            items = fam.get("items", [])
            for it in items:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        fam_name,
                        it["name"],
                        it["status"],
                        it["size_str"],
                        it.get("rel_path", ""),
                    ),
                )

        scroll = ttk.Scrollbar(models_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_guide_tab(self, parent: ttk.Frame) -> None:
        guide_text = tk.Text(parent, wrap="word", font=("Courier", 10))
        scroll = ttk.Scrollbar(parent, orient="vertical", command=guide_text.yview)
        guide_text.configure(yscrollcommand=scroll.set)
        guide_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        guide_text.insert(tk.END, get_ipython_guide())
        guide_text.configure(state="disabled")

    def copy_activation(self) -> None:
        cmd = get_activation_command()
        self.clipboard_clear()
        self.clipboard_append(cmd)
        messagebox.showinfo("Copied", f"Copied activation command to clipboard:\n\n{cmd}")

    def copy_summary(self) -> None:
        summary = format_env_summary()
        self.clipboard_clear()
        self.clipboard_append(summary)
        messagebox.showinfo("Copied", "Copied complete environment summary report to clipboard!")

    def open_terminal(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent
        cmd = get_activation_command()
        sys_os = platform.system()

        if sys_os == "Linux":
            terminals = ["gnome-terminal", "konsole", "xfce4-terminal", "x-terminal-emulator"]
            for term in terminals:
                if shutil.which(term):
                    subprocess.Popen(
                        [term, "--", "bash", "-c", f"cd '{root_dir}' && {cmd} && exec bash"],
                        cwd=root_dir,
                    )
                    return
        elif sys_os == "Darwin":
            script = f'tell application "Terminal" to do script "cd \'{root_dir}\' && {cmd}"'
            subprocess.Popen(["osascript", "-e", script])
            return
        elif sys_os == "Windows":
            subprocess.Popen(
                ["powershell", "-NoExit", "-Command", f"Set-Location '{root_dir}'; {cmd}"],
                cwd=root_dir,
            )
            return

        messagebox.showinfo(
            "Terminal",
            f"Open your system terminal and run:\n\n  cd {root_dir}\n  {cmd}",
        )

    def launch_ipython(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent
        cmd = get_activation_command()
        sys_os = platform.system()

        if sys_os == "Linux":
            terminals = ["gnome-terminal", "konsole", "xfce4-terminal", "x-terminal-emulator"]
            for term in terminals:
                if shutil.which(term):
                    subprocess.Popen(
                        [
                            term,
                            "--",
                            "bash",
                            "-c",
                            f"cd '{root_dir}' && {cmd} && echo 'Launching IPython...' && ipython",
                        ],
                        cwd=root_dir,
                    )
                    return
        elif sys_os == "Darwin":
            script = (
                f'tell application "Terminal" to do script "cd \'{root_dir}\' && {cmd} && ipython"'
            )
            subprocess.Popen(["osascript", "-e", script])
            return
        elif sys_os == "Windows":
            subprocess.Popen(
                [
                    "powershell",
                    "-NoExit",
                    "-Command",
                    f"Set-Location '{root_dir}'; {cmd}; ipython",
                ],
                cwd=root_dir,
            )
            return

        messagebox.showinfo(
            "IPython",
            f"Open terminal and execute:\n\n  cd {root_dir}\n  {cmd}\n  ipython",
        )


def run_vaila_env_gui(parent: tk.Tk | tk.Toplevel | None = None) -> None:
    """Launch the Environment & Imagination GUI dialog."""
    root = parent or getattr(tk, "_default_root", None)
    created = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created = True

    app = VailaEnvGUI(root)
    if created:
        app.protocol("WM_DELETE_WINDOW", root.destroy)
        root.mainloop()


def main() -> int:
    # Standalone CLI mode
    print(format_env_summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
