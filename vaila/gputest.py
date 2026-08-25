"""
===============================================================================
vailá Multimodal Toolbox
Script: gputest.py
===============================================================================
Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 07 October 2024
Update Date: 24 August 2026
Version: 0.3.113

Description:
------------
Comprehensive GPU, CUDA, PyTorch, and AI model diagnostic suite for vailá.
Validates the environment, runtime dependencies, hardware acceleration, and
model assets specifically for:
  1. PyTorch & CUDA Core (Hardware & Tensor allocation)
  2. markerless2d_yolo26.py (YOLO 2D Pose Estimation)
  3. yolov26track.py (YOLO Object Detection & Tracking)
  4. sam3sapiens2.py (SAM 3 + Sapiens2 2D Pose)
  5. sam3dinov3.py (SAM 3 + SAM 3D Body / DINOv3 3D)

Provides:
  - Rich formatted terminal tables and status summaries
  - Exact copy-pasteable CLI remediation commands for any errors/warnings
  - Interactive Tkinter GUI diagnostics window accessible from vaila.py footer

Usage:
------
# Run via CLI:
uv run python vaila/gputest.py

# Run with GUI window:
uv run python vaila/gputest.py --gui

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
===============================================================================
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time
import tkinter as tk
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class DiagnosticResult:
    """Diagnostic check result for a single component or script."""

    target: str
    script_path: str
    status: str  # "OK", "WARNING", "FAIL"
    summary: str
    details: list[str] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)


def _get_project_root() -> Path:
    """Resolve vailá repo / package root directory."""
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "vaila" and (script_dir.parent / "pyproject.toml").exists():
        return script_dir.parent
    if (script_dir / "pyproject.toml").exists():
        return script_dir
    return Path.cwd()


def _get_models_dir() -> Path:
    """Resolve models directory (vaila/models)."""
    return Path(__file__).resolve().parent / "models"


def check_pytorch_and_cuda() -> DiagnosticResult:
    """Check PyTorch installation, CUDA availability, NVIDIA GPU, and tensor allocation."""
    details: list[str] = []
    remediation: list[str] = []
    status = "OK"
    summary = "PyTorch with CUDA is operational"

    py_ver = sys.version.split()[0]
    os_sys = platform.system()
    os_release = platform.release()
    details.append(f"Python: {py_ver} on {os_sys} ({os_release})")

    # Check PyTorch import
    try:
        import torch
    except ImportError as exc:
        return DiagnosticResult(
            target="PyTorch & CUDA Core",
            script_path="Core Environment",
            status="FAIL",
            summary="PyTorch is not installed",
            details=[f"Import error: {exc}"],
            remediation=[
                "# 1. Setup PyTorch and CUDA dependencies:",
                "  bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,fifa,sapiens --yes",
            ],
        )

    torch_ver = torch.__version__
    cuda_compiled = getattr(torch.version, "cuda", None) or "None (CPU build)"
    details.append(f"PyTorch Version: {torch_ver} (Compiled with CUDA: {cuda_compiled})")

    # Check nvidia-smi presence
    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi_info = None
    if nvidia_smi_path:
        try:
            res = subprocess.run(
                [
                    nvidia_smi_path,
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().split("\n")
                nvidia_smi_info = lines[0].strip()
                details.append(f"NVIDIA-SMI: {nvidia_smi_info}")
        except Exception:
            pass

    # Check PyTorch CUDA availability
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:
        details.append(f"CUDA Init Error: {exc}")

    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            details.append(f"CUDA Devices: {device_count}")
            for i in range(device_count):
                dev_name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                total_mem_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                details.append(
                    f"Device {i}: {dev_name} (Compute {cap[0]}.{cap[1]}, {total_mem_gib:.1f} GiB VRAM)"
                )

            # Perform CUDA tensor computation smoke test
            t0 = time.perf_counter()
            x = torch.randn((1000, 1000), device="cuda", dtype=torch.float32)
            y = torch.matmul(x, x)
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000
            details.append(f"CUDA MatMul 1000x1000 Test: [OK] ({dt_ms:.2f} ms)")
            del x, y
            torch.cuda.empty_cache()
            status = "OK"
            summary = f"CUDA Active: {torch.cuda.get_device_name(0)}"
        except Exception as exc:
            status = "FAIL"
            summary = f"CUDA Error during computation: {exc}"
            details.append(f"Tensor Test Failed: {exc}")
            remediation.append(
                "CUDA tensor allocation failed. Check GPU memory or reboot if driver was updated."
            )
    else:
        if nvidia_smi_info:
            status = "FAIL"
            summary = "NVIDIA GPU found by OS, but PyTorch CUDA failed to initialize"
            details.append(
                "Diagnosis: PyTorch is CPU-only, or driver mismatch / uninitialized CUDA."
            )
            if os_sys == "Linux":
                remediation.extend(
                    [
                        "# Switch to Linux CUDA template and synchronize environment:",
                        "bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam,fifa,sapiens --yes",
                        "# If you recently updated NVIDIA drivers, reboot or reload the nvidia kernel module:",
                        "sudo modprobe nvidia",
                    ]
                )
            elif os_sys == "Windows":
                remediation.extend(
                    [
                        "# Switch to Windows CUDA template and synchronize environment:",
                        "pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu,sam,fifa,sapiens -Yes",
                    ]
                )
        elif os_sys == "Darwin":
            status = "WARNING"
            summary = "macOS Metal/MPS detected (CPU/MPS mode only)"
            details.append(
                "SAM 3 and Sapiens2 require NVIDIA CUDA; standard 2D and tools run on CPU/MPS."
            )
            remediation.append(
                "# For SAM3 / SAM 3D / Sapiens2, run on a Linux or Windows NVIDIA CUDA workstation."
            )
        else:
            status = "WARNING"
            summary = "No NVIDIA GPU detected (CPU mode active)"
            details.append("Running on CPU. Biomechanical analysis & 2D YOLO work on CPU.")
            if os_sys == "Linux":
                remediation.append(
                    "bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu --yes"
                )
            elif os_sys == "Windows":
                remediation.append("pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu -Yes")

    return DiagnosticResult(
        target="PyTorch & CUDA Core",
        script_path="vaila (Environment)",
        status=status,
        summary=summary,
        details=details,
        remediation=remediation,
    )


def check_markerless2d_yolo26() -> DiagnosticResult:
    """Check requirements and run inference test for vaila/markerless2d_yolo26.py."""
    target = "markerless2d_yolo26.py (2D Pose)"
    script_path = "vaila/markerless2d_yolo26.py"
    details: list[str] = []
    remediation: list[str] = []
    status = "OK"

    # Check ultralytics and cv2 imports
    try:
        import cv2  # noqa: F401
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        return DiagnosticResult(
            target=target,
            script_path=script_path,
            status="FAIL",
            summary=f"Missing dependencies: {exc}",
            details=[str(exc)],
            remediation=["uv sync"],
        )

    details.append(f"Ultralytics: v{ultralytics.__version__}")

    # Check pose weights in models directory
    models_dir = _get_models_dir()
    pose_weights = (
        list(models_dir.glob("yolo*pose*.pt"))
        + list(models_dir.glob("yolo11*-pose.pt"))
        + list(models_dir.glob("yolo26*-pose.pt"))
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if pose_weights:
        chosen_weight = pose_weights[0]
        details.append(f"Found Pose Weights: {chosen_weight.name} ({len(pose_weights)} available)")
    else:
        chosen_weight = models_dir / "yolo11n-pose.pt"
        details.append("Pose Weights: Not pre-downloaded (auto-downloads on first run)")

    # Run lightweight 1-frame inference test
    try:
        t0 = time.perf_counter()
        model = YOLO(str(chosen_weight) if chosen_weight.exists() else "yolo11n-pose.pt")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        raw_results = model(dummy_frame, device=device, verbose=False)
        results_list = list(raw_results) if raw_results is not None else []
        dt_ms = (time.perf_counter() - t0) * 1000

        has_kpts = False
        if results_list and getattr(results_list[0], "keypoints", None) is not None:
            has_kpts = True

        details.append(
            f"Inference Smoke Test: [OK] ({dt_ms:.1f} ms on {device.upper()}), Keypoints schema ready: {has_kpts}"
        )
        if device == "cuda":
            status = "OK"
            summary = f"Ready (GPU Inference {dt_ms:.1f} ms)"
        else:
            status = "WARNING"
            summary = f"Functional on CPU ({dt_ms:.1f} ms, GPU recommended for high FPS)"
    except Exception as exc:
        status = "FAIL"
        summary = f"Inference test failed: {exc}"
        details.append(f"Error: {exc}")
        remediation.extend(
            [
                "# Test or initialize YOLO pose weights:",
                "uv run python vaila/markerless2d_yolo26.py",
            ]
        )

    return DiagnosticResult(
        target=target,
        script_path=script_path,
        status=status,
        summary=summary,
        details=details,
        remediation=remediation,
    )


def check_yolov26track() -> DiagnosticResult:
    """Check requirements and run tracking smoke test for vaila/yolov26track.py."""
    target = "yolov26track.py (Tracking & Markers)"
    script_path = "vaila/yolov26track.py"
    details: list[str] = []
    remediation: list[str] = []
    status = "OK"

    # Check ultralytics and tracking dependencies
    try:
        import numpy as np
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        return DiagnosticResult(
            target=target,
            script_path=script_path,
            status="FAIL",
            summary=f"Missing dependencies: {exc}",
            details=[str(exc)],
            remediation=["uv sync"],
        )

    # Check linear assignment for ByteTrack / BoT-SORT
    if importlib.util.find_spec("lap") is not None or importlib.util.find_spec("lapx") is not None:
        details.append("Linear Assignment (lap/lapx): [OK] Installed")
    else:
        details.append("Linear Assignment (lap/lapx): [INFO] scipy fallback")

    # Check FFmpeg
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        details.append(f"FFmpeg: [OK] Found ({ffmpeg_bin})")
    else:
        status = "WARNING"
        details.append("FFmpeg: [WARNING] Missing from PATH (video re-encoding slower)")
        remediation.extend(
            [
                "# Install FFmpeg for video transcoding and overlay export:",
                "sudo apt update && sudo apt install -y ffmpeg  # Linux",
                "# or: winget install Gyan.FFmpeg             # Windows",
                "# or: brew install ffmpeg                   # macOS",
            ]
        )

    # Check model weights
    models_dir = _get_models_dir()
    det_weights = list(models_dir.glob("yolo*n.pt")) + list(models_dir.glob("yolo11*.pt"))
    if det_weights:
        chosen_weight = det_weights[0]
        details.append(f"Detection Weights: {chosen_weight.name}")
    else:
        chosen_weight = models_dir / "yolo11n.pt"
        details.append("Detection Weights: yolo11n.pt (auto-downloads on first run)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run 1-frame tracking test
    try:
        t0 = time.perf_counter()
        model = YOLO(str(chosen_weight) if chosen_weight.exists() else "yolo11n.pt")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        model.track(dummy_frame, persist=True, device=device, verbose=False)
        dt_ms = (time.perf_counter() - t0) * 1000
        details.append(f"Tracking Smoke Test: [OK] ({dt_ms:.1f} ms on {device.upper()})")
        if status == "OK":
            summary = (
                f"Ready (GPU Tracking {dt_ms:.1f} ms)"
                if device == "cuda"
                else f"Functional on CPU ({dt_ms:.1f} ms)"
            )
    except Exception as exc:
        status = "FAIL"
        summary = f"Tracking test failed: {exc}"
        details.append(f"Error: {exc}")
        remediation.extend(
            [
                "# Test yolov26track CLI help:",
                "uv run python -u -m vaila.yolov26track --help",
            ]
        )

    return DiagnosticResult(
        target=target,
        script_path=script_path,
        status=status,
        summary=summary,
        details=details,
        remediation=remediation,
    )


def check_sam3sapiens2() -> DiagnosticResult:
    """Check dependencies, CUDA, SAM3, and Sapiens2 pose assets for vaila/sam3sapiens2.py."""
    target = "sam3sapiens2.py (SAM 3 + Sapiens2)"
    script_path = "vaila/sam3sapiens2.py"
    details: list[str] = []
    remediation: list[str] = []
    missing_items: list[str] = []
    status = "OK"

    # Check CUDA requirement
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        status = "FAIL"
        missing_items.append("NVIDIA CUDA GPU (strictly required by SAM3 & Sapiens2)")
        details.append("CUDA: [FAIL] Not available (SAM 3 and Sapiens2 do not support CPU/MPS)")
    else:
        details.append("CUDA: [OK] Active")

    # Check SAM 3 package
    sam3_spec = importlib.util.find_spec("sam3")
    if sam3_spec is not None:
        details.append("SAM 3 package: [OK] Installed")
    else:
        status = "FAIL"
        missing_items.append("SAM 3 package (uv sync --extra sam)")
        details.append("SAM 3 package: [FAIL] Not found")

    # Check SAM 3 checkpoint
    models_dir = _get_models_dir()
    sam3_ckpt = models_dir / "sam3" / "sam3.pt"
    sam3_env = os.environ.get("SAM3_CHECKPOINT") or os.environ.get("VAILA_SAM3_CHECKPOINT")
    if sam3_env and Path(sam3_env).is_file():
        details.append(f"SAM 3 checkpoint: [OK] Found via env ({sam3_env})")
    elif sam3_ckpt.is_file():
        details.append(f"SAM 3 checkpoint: [OK] Found ({sam3_ckpt})")
    else:
        details.append(
            f"SAM 3 checkpoint: [INFO] Missing ({sam3_ckpt}) — download via vaila_sam.py"
        )
        missing_items.append("SAM 3 weights (facebook/sam3)")

    # Check Sapiens2 package & config & weights
    try:
        try:
            from .vaila_sapiens import resolve_model_spec
        except ImportError:
            from vaila_sapiens import resolve_model_spec  # ty: ignore[unresolved-import]

        spec = resolve_model_spec("1b")
        if spec.config_path.is_file():
            details.append("Sapiens2 Pose config (1B): [OK] Found")
        else:
            missing_items.append("Sapiens2 checkout (bash bin/setup_sapiens2.sh)")
            details.append(f"Sapiens2 config: [FAIL] Missing ({spec.config_path})")

        if spec.checkpoint_path.is_file():
            details.append(f"Sapiens2 Pose weights (1B): [OK] Found ({spec.checkpoint_path.name})")
        else:
            details.append(f"Sapiens2 Pose weights (1B): [INFO] Missing ({spec.checkpoint_path})")
            missing_items.append("Sapiens2 1B pose weights (facebook/sapiens2-pose-1b)")

        if spec.detector_path.is_dir():
            details.append("Sapiens2 DETR detector: [OK] Found")
        else:
            details.append(f"Sapiens2 DETR detector: [INFO] Missing ({spec.detector_path})")
            missing_items.append("DETR detector (facebook/detr-resnet-101-dc5)")
    except Exception as exc:
        status = "FAIL"
        missing_items.append(f"Sapiens2 setup error: {exc}")
        details.append(f"Sapiens2 check failed: {exc}")

    if missing_items:
        if status != "FAIL":
            status = "WARNING"
        summary = f"Setup needed ({len(missing_items)} item(s) pending)"
        remediation.extend(
            [
                "# Sapiens2 Setup & Weights:",
                "bash bin/setup_sapiens2.sh",
                "uv run hf auth login",
                "uv run vaila/vaila_sam.py --download-weights",
                "uv run vaila/vaila_sapiens.py --download-weights --model 1b",
            ]
        )
    else:
        summary = "Ready (SAM3 + Sapiens2 1B fully provisioned)"

    return DiagnosticResult(
        target=target,
        script_path=script_path,
        status=status,
        summary=summary,
        details=details,
        remediation=remediation,
    )


def check_sam3dinov3() -> DiagnosticResult:
    """Check dependencies, CUDA, SAM 3D Body, and DINOv3 assets for vaila/sam3dinov3.py."""
    target = "sam3dinov3.py (SAM 3 + DINOv3 3D)"
    script_path = "vaila/sam3dinov3.py"
    details: list[str] = []
    remediation: list[str] = []
    missing_items: list[str] = []
    status = "OK"

    # Check CUDA requirement
    try:
        import torch

        cuda_ok = torch.cuda.is_available()
    except Exception:
        cuda_ok = False

    if not cuda_ok:
        status = "FAIL"
        missing_items.append("NVIDIA CUDA GPU (strictly required by SAM 3D Body)")
        details.append("CUDA: [FAIL] Not available (SAM 3D Body unconditionally requires cuda)")
    else:
        details.append("CUDA: [OK] Active")

    # Check SAM 3 package
    sam3_spec = importlib.util.find_spec("sam3")
    if sam3_spec is not None:
        details.append("SAM 3 package: [OK] Installed")
    else:
        status = "FAIL"
        missing_items.append("SAM 3 package (uv sync --extra sam)")
        details.append("SAM 3 package: [FAIL] Not found")

    # Check SAM 3D Body checkout
    try:
        try:
            from .sam3dinov3 import ensure_sam3d_importable, resolve_sam3d_assets
        except ImportError:
            from sam3dinov3 import (  # ty: ignore[unresolved-import]
                ensure_sam3d_importable,
                resolve_sam3d_assets,
            )

        sam3d_checkout = ensure_sam3d_importable()
        if sam3d_checkout:
            details.append(f"sam_3d_body checkout: [OK] Found ({sam3d_checkout})")
        else:
            status = "FAIL"
            missing_items.append("sam_3d_body repository (bash bin/setup_fifa_sam3d.sh)")
            details.append("sam_3d_body checkout: [FAIL] Not found")

        # Check SAM 3D DINOv3 assets
        try:
            ckpt_path, mhr_path = resolve_sam3d_assets(None)
            details.append(f"SAM 3D checkpoint: [OK] Found ({ckpt_path.name})")
            details.append(f"MHR model: [OK] Found ({mhr_path.name})")
        except Exception as exc:
            missing_items.append("SAM 3D DINOv3 weights (facebook/sam-3d-body-dinov3)")
            details.append(f"SAM 3D assets: [INFO] Missing ({exc})")
    except Exception as exc:
        status = "FAIL"
        missing_items.append(f"SAM 3D import error: {exc}")
        details.append(f"SAM 3D check failed: {exc}")

    # Check Lightning stack
    pl_spec = importlib.util.find_spec("pytorch_lightning") or importlib.util.find_spec("lightning")
    if pl_spec is not None:
        details.append("PyTorch Lightning: [OK] Installed")
    else:
        details.append("PyTorch Lightning: [INFO] Optional stack (uv sync --extra fifa)")

    if missing_items:
        if status != "FAIL":
            status = "WARNING"
        summary = f"Setup needed ({len(missing_items)} item(s) pending)"
        remediation.extend(
            [
                "# SAM 3D Body (DINOv3) Setup & Weights:",
                "bash bin/setup_fifa_sam3d.sh",
                "uv run hf auth login",
            ]
        )
    else:
        summary = "Ready (SAM3 + SAM 3D Body / DINOv3 3D fully provisioned)"

    return DiagnosticResult(
        target=target,
        script_path=script_path,
        status=status,
        summary=summary,
        details=details,
        remediation=remediation,
    )


def run_gpu_diagnostics(verbose: bool = True) -> list[DiagnosticResult]:
    """Run all GPU, PyTorch, CUDA, and AI model diagnostic checks."""
    if verbose:
        console.print(
            Panel.fit(
                "[bold cyan]vailá — GPU & AI Stack Diagnostics[/bold cyan]\n"
                f"[dim]System: {platform.system()} | Python: {sys.version.split()[0]} | {time.strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
                border_style="cyan",
            )
        )

    results: list[DiagnosticResult] = []

    # 1. PyTorch & CUDA Core
    r_core = check_pytorch_and_cuda()
    results.append(r_core)

    # 2. markerless2d_yolo26.py
    r_yolo_pose = check_markerless2d_yolo26()
    results.append(r_yolo_pose)

    # 3. yolov26track.py
    r_yolo_track = check_yolov26track()
    results.append(r_yolo_track)

    # 4. sam3sapiens2.py
    r_sam3_sapiens = check_sam3sapiens2()
    results.append(r_sam3_sapiens)

    # 5. sam3dinov3.py
    r_sam3_dinov3 = check_sam3dinov3()
    results.append(r_sam3_dinov3)

    if verbose:
        print_diagnostics_report(results)

    return results


def print_diagnostics_report(results: list[DiagnosticResult]) -> None:
    """Print beautifully formatted Rich tables, summary, and remediation commands to terminal."""
    status_styles = {
        "OK": "[bold green][PASS][/bold green]",
        "WARNING": "[bold yellow][WARN][/bold yellow]",
        "FAIL": "[bold red][FAIL][/bold red]",
    }

    # Print individual section panels
    for r in results:
        t = Table(show_header=True, header_style="bold magenta", expand=True)
        t.add_column("Property / Check", style="cyan", ratio=1)
        t.add_column("Result / Value", style="white", ratio=2)

        for line in r.details:
            if ":" in line:
                k, v = line.split(":", 1)
                t.add_row(k.strip(), v.strip())
            else:
                t.add_row("Info", line)

        panel_color = (
            "green" if r.status == "OK" else ("yellow" if r.status == "WARNING" else "red")
        )
        console.print(
            Panel(
                t,
                title=f"[{panel_color}]{r.target} — {status_styles.get(r.status, r.status)}[/{panel_color}]",
                subtitle=f"[dim]{r.script_path}[/dim]",
                border_style=panel_color,
            )
        )

    # Print summary table
    summary_table = Table(
        title="vailá Diagnostics Summary",
        show_header=True,
        header_style="bold white on blue",
        expand=True,
    )
    summary_table.add_column("Target Script / Component", style="bold cyan")
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Summary Notes", style="white")

    has_errors = False
    all_remediations: list[str] = []

    for r in results:
        summary_table.add_row(r.target, status_styles.get(r.status, r.status), r.summary)
        if r.status == "FAIL":
            has_errors = True
        for cmd in r.remediation:
            if cmd not in all_remediations:
                all_remediations.append(cmd)

    console.print(summary_table)

    # Print action instructions if there are any issues
    if all_remediations:
        box_title = (
            "[bold red]>> ACTION REQUIRED — HOW TO FIX IN TERMINAL[/bold red]"
            if has_errors
            else "[bold yellow]>> RECOMMENDED COMMANDS / ACTIONS[/bold yellow]"
        )
        border_color = "red" if has_errors else "yellow"

        body = "\n".join(all_remediations)
        console.print(Panel(body, title=box_title, border_style=border_color, padding=(1, 2)))
    else:
        console.print(
            Panel(
                "[bold green]✓ All AI models, CUDA runtimes, and dependencies are fully operational![/bold green]",
                title="[bold green]System Health Check[/bold green]",
                border_style="green",
            )
        )


def _copy_text_to_clipboard(text: str, root_widget: tk.Misc) -> None:
    """Copy string to OS clipboard via Tkinter."""
    try:
        root_widget.clipboard_clear()
        root_widget.clipboard_append(text)
        root_widget.update()
        messagebox.showinfo(
            "Copied",
            "Fix commands copied to clipboard! You can paste them into your terminal.",
            parent=root_widget,
        )
    except Exception as exc:
        messagebox.showwarning(
            "Clipboard", f"Could not copy to clipboard: {exc}", parent=root_widget
        )


def show_gpu_diagnostics_dialog(
    parent: tk.Misc | None = None, results: list[DiagnosticResult] | None = None
) -> None:
    """Display an interactive Tkinter Toplevel window with the diagnostic report and fix instructions."""
    if results is None:
        results = run_gpu_diagnostics(verbose=True)

    # Create Toplevel or standalone Tk
    standalone = parent is None
    root = tk.Tk() if standalone else tk.Toplevel(parent)
    root.title("vailá — GPU & AI Stack Diagnostics")
    root.geometry("860x680")
    root.minsize(720, 520)

    # Status color mapping
    status_colors = {
        "OK": "#28a745",
        "WARNING": "#e67e22",
        "FAIL": "#dc3545",
    }
    status_badges = {
        "OK": " [PASS] ",
        "WARNING": " [WARN] ",
        "FAIL": " [FAIL] ",
    }

    has_fails = any(r.status == "FAIL" for r in results)
    has_warns = any(r.status == "WARNING" for r in results)
    overall_color = "#dc3545" if has_fails else ("#e67e22" if has_warns else "#28a745")
    overall_text = (
        "ERRORS DETECTED — Terminal Action Required"
        if has_fails
        else (
            "WARNINGS — Recommendations Available" if has_warns else "ALL CHECKS PASSED — GPU Ready"
        )
    )

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill=tk.BOTH, expand=True)

    # Header banner
    banner_frame = tk.Frame(main_frame, bg=overall_color, padx=10, pady=8)
    banner_frame.pack(fill=tk.X, pady=(0, 10))

    tk.Label(
        banner_frame,
        text="vailá GPU & AI Stack Diagnostic Report",
        font=("Helvetica", 13, "bold"),
        fg="white",
        bg=overall_color,
    ).pack(anchor="w")

    tk.Label(
        banner_frame,
        text=overall_text,
        font=("Helvetica", 10, "bold"),
        fg="#f8f9fa",
        bg=overall_color,
    ).pack(anchor="w")

    # Scrollable container for test results
    container = ttk.Frame(main_frame)
    container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_content = ttk.Frame(canvas)

    scrollable_content.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas_window = canvas.create_window((0, 0), window=scrollable_content, anchor="nw")

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)

    canvas.bind("<Configure>", _on_canvas_configure)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Populate cards for each diagnostic result
    all_remediations: list[str] = []
    for r in results:
        card = ttk.LabelFrame(scrollable_content, text=f" {r.target} ", padding=8)
        card.pack(fill=tk.X, padx=4, pady=4)

        # Header with badge
        hdr_row = ttk.Frame(card)
        hdr_row.pack(fill=tk.X, pady=(0, 4))

        color = status_colors.get(r.status, "#6c757d")
        badge = status_badges.get(r.status, f" [{r.status}] ")
        badge_lbl = tk.Label(
            hdr_row,
            text=badge,
            font=("Helvetica", 9, "bold"),
            bg=color,
            fg="white",
            padx=4,
            pady=1,
        )
        badge_lbl.pack(side=tk.LEFT, padx=(0, 6))

        summary_lbl = ttk.Label(hdr_row, text=r.summary, font=("Helvetica", 10, "bold"))
        summary_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Details list
        details_txt = "\n".join(f"• {d}" for d in r.details)
        det_lbl = ttk.Label(
            card, text=details_txt, font=("TkFixedFont", 9), justify=tk.LEFT, wraplength=760
        )
        det_lbl.pack(anchor="w", padx=8, pady=(2, 2))

        for cmd in r.remediation:
            if cmd not in all_remediations:
                all_remediations.append(cmd)

    # Remediation / Commands Box
    if all_remediations:
        fix_frame = ttk.LabelFrame(
            main_frame, text=" Actionable Fix Commands (Copy to Terminal) ", padding=8
        )
        fix_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        remediation_text_content = "\n".join(all_remediations)

        text_scroll = ttk.Scrollbar(fix_frame, orient="vertical")
        cmd_text = tk.Text(
            fix_frame,
            height=6,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg="#212529",
            fg="#f8f9fa",
            insertbackground="white",
            yscrollcommand=text_scroll.set,
        )
        text_scroll.config(command=cmd_text.yview)
        cmd_text.insert(tk.END, remediation_text_content)
        cmd_text.config(state=tk.DISABLED)

        cmd_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # Action buttons bar
    btn_bar = ttk.Frame(main_frame)
    btn_bar.pack(fill=tk.X)

    if all_remediations:
        copy_btn = ttk.Button(
            btn_bar,
            text="📋 Copy Fix Commands",
            command=lambda: _copy_text_to_clipboard("\n".join(all_remediations), root),
        )
        copy_btn.pack(side=tk.LEFT, padx=(0, 6))

    def _open_gpu_guide():
        guide_html = _get_project_root() / "vaila" / "help" / "gpu_guide.html"
        if guide_html.exists():
            webbrowser.open_new_tab(guide_html.resolve().as_uri())
        else:
            messagebox.showinfo(
                "GPU Guide", "GPU Guide documentation is under vaila/help/gpu_guide.html"
            )

    help_btn = ttk.Button(btn_bar, text="📖 Open GPU Guide", command=_open_gpu_guide)
    help_btn.pack(side=tk.LEFT, padx=(0, 6))

    def _rerun():
        root.destroy()
        show_gpu_diagnostics_dialog(parent=parent)

    rerun_btn = ttk.Button(btn_bar, text="🔄 Re-run Test", command=_rerun)
    rerun_btn.pack(side=tk.LEFT, padx=(0, 6))

    close_btn = ttk.Button(btn_bar, text="Close", command=root.destroy)
    close_btn.pack(side=tk.RIGHT)

    if standalone:
        root.mainloop()


def main() -> None:
    """CLI entry point for gputest."""
    parser = argparse.ArgumentParser(
        description="vailá GPU & AI Stack Diagnostics (CUDA, PyTorch, SAM 3, Sapiens2, YOLO)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Display the interactive Tkinter diagnostic window",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal printing and return results silently",
    )
    args = parser.parse_args()

    results = run_gpu_diagnostics(verbose=not args.quiet)

    if args.gui:
        show_gpu_diagnostics_dialog(results=results)


if __name__ == "__main__":
    main()
