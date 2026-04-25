"""
Project: vailá
Script: vaila_sam.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 16 April 2026
Update Date: 19 April 2026
Version: 0.0.4

Description:
    This script performs video segmentation using the SAM 3 model from Meta.
    It supports text-prompt masks via Hugging Face checkpoints.
    Requires: uv sync --extra sam (works on the universal CPU pyproject; inference still needs NVIDIA CUDA).
    Hybrid installs: see AGENTS.md (bin/use_pyproject_* to switch laptop CPU vs workstation CUDA).
    Auth: the repo facebook/sam3 is gated. Either:
        - Accept the model on Hugging Face, then: uv run hf auth login
        - Or download into the repo: uv run vaila/vaila_sam.py --download-weights
        - Or: uv run hf download facebook/sam3 sam3.pt --local-dir DIR
        - Or set env SAM3_CHECKPOINT=/path/to/sam3.pt
        - Default weights: vaila/models/sam3/sam3.pt (flat) or vaila/models/sam3/sam3_weights/sam3.pt (nested)
        - Legacy repo root: ./sam3_weights/sam3.pt — prefer flat layout under vaila/models/sam3/
        - VRAM: long clips load all frames onto GPU; auto-sized to GPU VRAM by default, or set SAM3_MAX_FRAMES / --max-frames
        - **Spatial:** very large frames (4K broadcast) can OOM even with ``--max-frames 1``; use ``--max-input-long-edge`` or ``SAM3_MAX_INPUT_LONG_EDGE`` (default 1920).
        - GPU: CUDA only (sam3 video predictor loads on .cuda()). No CPU or macOS MPS path; ``--frame-by-frame`` only lowers VRAM on CUDA.

Usage:
    uv run vaila/vaila_sam.py
    uv run vaila/vaila_sam.py --open-help          # SAM 3 setup (HTML in browser)
    uv run vaila/vaila_sam.py --download-weights   # HF_TOKEN or hf auth login

Requires: ``uv sync --extra sam`` (works on the universal CPU ``pyproject``; inference still needs NVIDIA CUDA).
Hybrid installs: see ``AGENTS.md`` (``bin/use_pyproject_*`` to switch laptop CPU vs workstation CUDA).
Auth: the repo facebook/sam3 is gated. Either:
  - Accept the model on Hugging Face, then: uv run hf auth login
  - Or download into the repo: uv run vaila/vaila_sam.py --download-weights
    (writes vaila/models/sam3/; *.pt is gitignored)
  - Or: uv run hf download facebook/sam3 sam3.pt --local-dir DIR
  - Or set env SAM3_CHECKPOINT=/path/to/sam3.pt
  - Default weights: vaila/models/sam3/sam3.pt (flat) or vaila/models/sam3/sam3_weights/sam3.pt (nested)
  - Legacy repo root: ./sam3_weights/sam3.pt — prefer flat layout under vaila/models/sam3/
  - VRAM: long clips load all frames onto GPU; auto-sized to GPU VRAM by default, or set SAM3_MAX_FRAMES / --max-frames
GPU: CUDA only (sam3 video predictor loads on .cuda()). No CPU or macOS MPS; ``--frame-by-frame`` is CUDA VRAM-only (not a CPU mode).

FIFA Skeletal Tracking Light (optional ``--extra fifa``): subcommand ``fifa`` delegates to
``vaila.fifa_skeletal_pipeline`` (prepare, boxes, preprocess, baseline, dlt-export, pack). Example::

    uv sync --extra fifa --extra sam --extra gpu   # CUDA template + deps
    uv run vaila/vaila_sam.py fifa prepare --video-source DIR --data-root data/
    uv run vaila/vaila_sam.py fifa boxes --data-root data/ --sequences data/sequences_val.txt
    uv run vaila/vaila_sam.py fifa preprocess --data-root data/ --sequences data/sequences_val.txt
    uv run vaila/vaila_sam.py fifa baseline --data-root data/ --sequences data/sequences_val.txt -o out/npz
    uv run vaila/vaila_sam.py fifa dlt-export --cameras-dir data/cameras --output-dir out/dlt
    uv run vaila/vaila_sam.py fifa pack --submission-full out/npz --data-root data/ --output-dir out/ --split val

License:
    This program is licensed under the GNU Affero General Public License v3.0.
    For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
    Visit the project repository: https://github.com/vaila-multimodaltoolbox
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import json
import os
import platform
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

SAM3_DEFAULT_CKPT_NAME = "sam3.pt"
SAM3_MULTIPLEX_CKPT_NAME = "sam3.1_multiplex.pt"
SAM3_HF_REPO_ID = "facebook/sam3"

# All recognised checkpoint file names (tried in order during auto-detection)
SAM3_CKPT_CANDIDATES: tuple[str, ...] = (SAM3_DEFAULT_CKPT_NAME, SAM3_MULTIPLEX_CKPT_NAME)

_SAM3_PERPETUAL_TRACKER_AUTOCAST_PATCHED = False
_SAM3_BACKBONE_FPN_FP32_PATCHED = False


def _patch_sam3_disable_perpetual_tracker_autocast() -> None:
    """Exit SAM3's permanent bfloat16 autocast on the video tracker.

    ``Sam3TrackerPredictor`` calls ``bf16_context.__enter__()`` in ``__init__`` and never
    balances it with ``__exit__``, so the whole video stack runs under autocast.  Some
    PyTorch/cuDNN builds then fail with::

        Input type (c10::BFloat16) and bias type (float) should be the same

    Exiting the context after the upstream ``__init__`` restores stable float32 for those
    ops.  Set ``SAM3_KEEP_TRACKER_BF16_AUTOCAST=1`` to keep upstream behavior.
    """
    global _SAM3_PERPETUAL_TRACKER_AUTOCAST_PATCHED
    if _SAM3_PERPETUAL_TRACKER_AUTOCAST_PATCHED:
        return
    if os.environ.get("SAM3_KEEP_TRACKER_BF16_AUTOCAST", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        _SAM3_PERPETUAL_TRACKER_AUTOCAST_PATCHED = True
        return
    try:
        from sam3.model import sam3_tracking_predictor as _stp  # type: ignore[reportMissingImports]
    except ImportError:
        return

    _orig_init = _stp.Sam3TrackerPredictor.__init__

    def _init_no_perpetual_autocast(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        ctx = getattr(self, "bf16_context", None)
        if ctx is not None:
            ctx.__exit__(None, None, None)
            self.bf16_context = contextlib.nullcontext()

    _stp.Sam3TrackerPredictor.__init__ = _init_no_perpetual_autocast  # ty: ignore[invalid-assignment]
    _SAM3_PERPETUAL_TRACKER_AUTOCAST_PATCHED = True


def _patch_sam3_force_fp32_tracker_backbone_features() -> None:
    """Coerce SAM3 tracker backbone FPN outputs to float32.

    On some CUDA/PyTorch builds, detector FPN features arrive as bfloat16 while
    ``sam_mask_decoder.conv_s0/conv_s1`` biases remain float32, producing:

        Input type (c10::BFloat16) and bias type (float) should be the same

    This patch normalizes ``tracker_backbone_fpn_{0,1,2}`` to float32 at the
    detector output boundary.
    """
    global _SAM3_BACKBONE_FPN_FP32_PATCHED
    if _SAM3_BACKBONE_FPN_FP32_PATCHED:
        return
    if os.environ.get("SAM3_ALLOW_BF16_BACKBONE_FPN", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        _SAM3_BACKBONE_FPN_FP32_PATCHED = True
        return
    try:
        from sam3.model.sam3_image import (  # type: ignore[reportMissingImports]
            Sam3ImageOnVideoMultiGPU as _Sam3ImageOnVideoMultiGPU,
        )
    except ImportError:
        return

    _orig_forward = _Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu

    def _forward_with_fp32_tracker_fpn(self, *args, **kwargs):
        out = _orig_forward(self, *args, **kwargs)
        if not (isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], dict)):
            return out

        sam3_image_out = out[0]
        for key in ("tracker_backbone_fpn_0", "tracker_backbone_fpn_1", "tracker_backbone_fpn_2"):
            t = sam3_image_out.get(key)
            if t is not None and hasattr(t, "dtype"):
                sam3_image_out[key] = t.float()
        return out

    _Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu = _forward_with_fp32_tracker_fpn  # ty: ignore[invalid-assignment]
    _SAM3_BACKBONE_FPN_FP32_PATCHED = True


def _repo_root() -> Path:
    """Project root (parent of the ``vaila`` package directory)."""
    return Path(__file__).resolve().parent.parent


def _package_sam3_dir() -> Path:
    return Path(__file__).resolve().parent / "models" / "sam3"


def _repo_models_sam3_dir() -> Path:
    """Alternate SAM3 dir under repository root (``<repo>/models/sam3``)."""
    return _repo_root() / "models" / "sam3"


def sam3_install_help_html_path() -> Path:
    """Browser help page: SAM 3 / optional extras (same content as README section)."""
    return Path(__file__).resolve().parent / "help" / "vaila_sam.html"


def open_sam3_install_help_in_browser() -> None:
    path = sam3_install_help_html_path()
    if path.is_file():
        webbrowser.open_new_tab(path.resolve().as_uri())
    else:
        webbrowser.open_new_tab(
            "https://raw.githubusercontent.com/vaila-multimodaltoolbox/vaila/main/vaila/help/vaila_sam.html"
        )


def _print_sam3_install_instructions() -> None:
    msg = (
        "SAM 3 is not installed.\n"
        "  Standard:  uv sync --extra sam\n"
        "  CUDA host: uv sync --extra gpu --extra sam   (after CUDA pyproject template)\n"
        "Opening setup instructions in your browser…\n"
        "See also: AGENTS.md (Hybrid CPU vs NVIDIA workstation).\n"
    )
    print(msg, file=sys.stderr)


_SAM3_CUDA_REQUIRED_BODY = (
    "SAM 3 video in vailá requires an NVIDIA GPU with CUDA "
    "(PyTorch must see torch.cuda.is_available()).\n\n"
    "Not supported for this script:\n"
    "  • CPU-only (e.g. Windows without an NVIDIA CUDA GPU)\n"
    "  • macOS Metal/MPS — the SAM 3 video stack targets CUDA\n\n"
    "The --frame-by-frame / GUI “frame-by-frame” option only reduces VRAM on CUDA; "
    "it is not a CPU fallback.\n\n"
    "Without CUDA, use other vailá tools (Markerless 2D, YOLO trackers, etc.) "
    "or run SAM 3 video on a Linux/Windows NVIDIA workstation or cloud GPU."
)


def _sam3_video_cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _sam3_guard_cuda_cli() -> None:
    if _sam3_video_cuda_available():
        return
    print("SAM 3 video requires CUDA (NVIDIA GPU).", file=sys.stderr)
    print(_SAM3_CUDA_REQUIRED_BODY, file=sys.stderr)
    raise SystemExit(2)


def _sam3_guard_cuda_gui(parent: tk.Tk) -> bool:
    if _sam3_video_cuda_available():
        return True
    messagebox.showerror("SAM 3 — CUDA required", _SAM3_CUDA_REQUIRED_BODY, parent=parent)
    return False


def _package_sam3_ckpt_path() -> Path:
    return _package_sam3_dir() / SAM3_DEFAULT_CKPT_NAME


def _package_sam3_nested_weights_ckpt_path() -> Path:
    """HF ``local-dir`` / manual ``mv sam3_weights`` often leaves ``.../sam3/sam3_weights/sam3.pt``."""
    return _package_sam3_dir() / "sam3_weights" / SAM3_DEFAULT_CKPT_NAME


def download_sam3_weights_to_vaila_models() -> Path:
    """Download gated SAM3 files into ``vaila/models/sam3/`` (gitignored *.pt).

    Uses ``HF_TOKEN`` or the token from ``hf auth login`` (Hugging Face cache).
    """
    from huggingface_hub import hf_hub_download

    dest = _package_sam3_dir()
    dest.mkdir(parents=True, exist_ok=True)
    try:
        for fname in ("config.json", SAM3_DEFAULT_CKPT_NAME):
            hf_hub_download(
                repo_id=SAM3_HF_REPO_ID,
                filename=fname,
                local_dir=str(dest),
            )
    except Exception as e:
        if _is_gated_repo_error(e):
            raise RuntimeError(
                _hf_access_help()
                + "\n\nFor --download-weights: set HF_TOKEN or run: uv run hf auth login"
            ) from e
        raise
    ckpt = _package_sam3_ckpt_path()
    if not ckpt.is_file():
        raise RuntimeError(f"Download finished but checkpoint missing: {ckpt}")
    return ckpt


def _is_gated_repo_error(exc: BaseException) -> bool:
    cur: BaseException | None = exc
    while cur is not None:
        if type(cur).__name__ == "GatedRepoError":
            return True
        msg = str(cur).lower()
        if "gated" in msg and "sam3" in msg:
            return True
        if "403" in str(cur) and "facebook/sam3" in str(cur):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _hf_access_help() -> str:
    return (
        "Hugging Face returned 403 for facebook/sam3 (gated model). The CLI message "
        '"not in the authorized list" means this Hugging Face *account* cannot '
        "download yet — not a vailá bug.\n\n"
        "Checklist:\n"
        "  1) While logged in on huggingface.co, open:\n"
        "       https://huggingface.co/facebook/sam3\n"
        "     Accept the license. If it says access pending, wait for approval.\n"
        "  2) Use the *same* account in this machine:\n"
        "       uv run hf auth login --force\n"
        "     Paste a token with Read (fine-grained: access to this model).\n"
        '     "User is already logged in" can be a different/old account — use --force.\n'
        "  3) Optional: export HF_TOKEN=... for that account, then retry download.\n\n"
        "Local weights (after a successful download once):\n"
        "       uv run vaila/vaila_sam.py --download-weights\n"
        "       # or: uv run hf download facebook/sam3 sam3.pt --local-dir vaila/models/sam3\n"
        "       uv run vaila/vaila_sam.py -i VIDEO -o OUT --checkpoint vaila/models/sam3/sam3.pt\n\n"
        "Environment variable: SAM3_CHECKPOINT=/path/to/sam3.pt"
    )


def _assert_sam3_video_checkpoint_not_3d_body_weights(path: Path) -> None:
    """SAM 3 *video* (facebook/sam3, ``sam3.pt``) is a different model than SAM 3D Body / FIFA."""
    lowered = [x.lower() for x in path.parts]
    if "sam-3d-dinov3" in lowered or "sam_3d_body" in lowered:
        raise ValueError(
            "This path is under SAM 3D Body / FIFA weights, not SAM 3 *video* (facebook/sam3).\n\n"
            'Clear "sam3.pt (optional)" to use vaila/models/sam3/sam3.pt, or run:\n'
            "  uv run vaila/vaila_sam.py --download-weights\n\n"
            f"Refused: {path}"
        )
    name = path.name.lower()
    if name == "mhr_model.pt" or (name == "model.ckpt" and path.suffix.lower() == ".ckpt"):
        raise ValueError(
            "That file is a SAM 3D Body / Lightning checkpoint, not SAM 3 video (sam3.pt).\n"
            "Use weights from https://huggingface.co/facebook/sam3\n\n"
            f"Refused: {path}"
        )


def _resolve_sam3_checkpoint_file(checkpoint: Path | None) -> Path | None:
    """
    Return path to sam3.pt / sam3.1_multiplex.pt, or None to let sam3 download from Hub.

    Order: argument → env SAM3_CHECKPOINT / VAILA_SAM3_CHECKPOINT →
    ``vaila/models/sam3/{sam3.pt, sam3.1_multiplex.pt}`` →
    ``vaila/models/sam3/sam3_weights/{sam3.pt, sam3.1_multiplex.pt}`` →
    ``models/sam3/{sam3.pt, sam3.1_multiplex.pt}`` (repo root) →
    ``models/sam3/sam3_weights/{sam3.pt, sam3.1_multiplex.pt}`` (repo root) →
    legacy ``<repo>/sam3_weights/sam3.pt``.
    """
    raw: str | None = None
    if checkpoint is not None:
        raw = str(checkpoint)
    if not raw:
        raw = os.environ.get("SAM3_CHECKPOINT") or os.environ.get("VAILA_SAM3_CHECKPOINT")
    if not raw or not raw.strip():
        # Auto-detect: try each candidate name in the package dir
        pkg_dir = _package_sam3_dir()
        for cname in SAM3_CKPT_CANDIDATES:
            p = pkg_dir / cname
            if p.is_file():
                return p
        # Nested weights dir (HF local-dir layout)
        for cname in SAM3_CKPT_CANDIDATES:
            p = pkg_dir / "sam3_weights" / cname
            if p.is_file():
                return p
        # Alternate repo-root models dir fallback
        repo_models_dir = _repo_models_sam3_dir()
        for cname in SAM3_CKPT_CANDIDATES:
            p = repo_models_dir / cname
            if p.is_file():
                return p
        for cname in SAM3_CKPT_CANDIDATES:
            p = repo_models_dir / "sam3_weights" / cname
            if p.is_file():
                return p
        # Legacy repo root fallback
        legacy_ckpt = _repo_root() / "sam3_weights" / SAM3_DEFAULT_CKPT_NAME
        if legacy_ckpt.is_file():
            return legacy_ckpt
        return None
    p = Path(raw).expanduser().resolve()
    if p.is_dir():
        # If a directory was given, look for any candidate inside it
        for cname in SAM3_CKPT_CANDIDATES:
            candidate = p / cname
            if candidate.is_file():
                _assert_sam3_video_checkpoint_not_3d_body_weights(candidate)
                return candidate
        raise FileNotFoundError(
            f"SAM3 checkpoint not found in directory: {p}\n"
            f"Expected one of: {', '.join(SAM3_CKPT_CANDIDATES)}"
        )
    if not p.is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {p}\n"
            f"Expected a file named {SAM3_DEFAULT_CKPT_NAME} or {SAM3_MULTIPLEX_CKPT_NAME} "
            f"(or pass the full path)."
        )
    _assert_sam3_video_checkpoint_not_3d_body_weights(p)
    return p


def _resolve_bpe_path() -> Path:
    """SAM3 PyPI wheel may omit site-packages/assets; fall back to boxmot's CLIP BPE."""
    try:
        import sam3  # type: ignore[reportMissingImports]
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Optional dependency 'sam3' is not installed. Install the SAM 3 stack with "
            "`uv sync --extra sam` (and on NVIDIA workstations also switch to the CUDA "
            "pyproject template + `uv sync --extra gpu`)."
        ) from e

    candidates = [
        Path(sam3.__file__).resolve().parent.parent / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        Path(sam3.__file__).resolve().parent / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    for p in candidates:
        if p.is_file():
            return p
    try:
        import boxmot

        fb = (
            Path(boxmot.__file__).resolve().parent
            / "reid"
            / "backbones"
            / "clip"
            / "clip"
            / "bpe_simple_vocab_16e6.txt.gz"
        )
        if fb.is_file():
            return fb
    except ImportError:
        pass
    raise FileNotFoundError(
        "Could not find bpe_simple_vocab_16e6.txt.gz. Install sam3 extra: uv sync --extra sam"
    )


def _auto_max_frames_for_vram() -> int:
    """Estimate a safe frame cap based on **currently free** GPU VRAM.

    Empirical data on RTX 5050 Laptop (7.53 GiB):
      - 256 frames → 7.23 GiB PyTorch allocated, OOM at +12 MiB
      - 125 frames → 6.76 GiB PyTorch allocated, OOM at +914 MiB (processing)
      - Per-frame cost: ~3.6 MiB  |  Model base: ~6.3 GiB
      - Processing peak: ~0.9 GiB contiguous allocation

    Budget: model_base + N*per_frame + processing_peak + headroom < free_vram
    With ``expandable_segments:True`` the 0.9 GiB processing allocation can
    reuse freed memory, so we only reserve 0.5 GiB processing overhead here.

    Using **free** VRAM (not total) is critical in batch mode: even after
    ``predictor.shutdown()`` the previous run leaves reserved memory behind.
    Sizing for ``total_memory`` then OOMs from video #2 onwards.
    """
    profile = _sam3_vram_profile()
    if profile is None:
        return 32
    safe_frames = int(profile["safe_frames"])
    total_gib = float(profile["total_gib"])
    free_gib = float(profile["free_gib"])
    budget_gib = float(profile["budget_gib"])
    print(
        f"[SAM3 VRAM] GPU {total_gib:.1f} GiB total, {free_gib:.1f} GiB free → "
        f"auto max_frames={safe_frames} "
        f"(model ~{profile['model_gib']:.1f} GiB, budget ~{budget_gib:.1f} GiB for frames)"
    )
    # #region agent log
    _agent_dbg_sam3(
        hypothesis_id="H2",
        location="vaila_sam.py:_auto_max_frames_for_vram",
        message="auto_max_frames",
        data={
            "total_gib": round(total_gib, 3),
            "free_gib": round(free_gib, 3),
            "budget_gib": round(budget_gib, 3),
            "safe_frames": safe_frames,
        },
    )
    # #endregion
    return safe_frames


def _sam3_vram_profile() -> dict[str, float] | None:
    """Collect non-throwing CUDA/VRAM stats used by SAM3 planning helpers."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        total_gib = props.total_memory / (1024**3)
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        free_gib = free_bytes / (1024**3)
    except Exception:
        return None

    model_gib = 6.3
    processing_gib = 0.5
    headroom_gib = 0.5
    per_frame_mib = 3.6
    budget_gib = max(0.0, free_gib - model_gib - processing_gib - headroom_gib)
    safe_frames = int(budget_gib * 1024 / per_frame_mib)
    safe_frames = max(16, min(safe_frames, 128))
    return {
        "total_gib": total_gib,
        "free_gib": free_gib,
        "budget_gib": budget_gib,
        "safe_frames": float(safe_frames),
        "model_gib": model_gib,
        "processing_gib": processing_gib,
        "headroom_gib": headroom_gib,
        "per_frame_mib": per_frame_mib,
    }


def _release_sam3_gpu_memory() -> None:
    """Best-effort GPU memory release between SAM3 batch items.

    SAM3's video predictor (and its shadow allocations) keeps memory reserved
    even after ``predictor.shutdown()``.  Without an explicit gc + empty_cache,
    the next ``build_sam3_video_predictor`` allocates on top of the previous
    state and OOMs even on 24 GiB cards.

    Note (debug session 42b4a5, Apr 2026): this helper clears only what Python's
    gc can reach.  A failed ``predictor.handle_request("start_session")`` leaves
    ~13 GiB of orphan tensors in SAM3's C++ workspace pools that only a process
    death can release — hence the subprocess-per-video isolation in ``main()``.
    See ``.claude/skills/sam3-video/SKILL.md`` § *Why subprocess-per-video*.
    """
    import gc

    try:
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch.cuda.ipc_collect()
        with contextlib.suppress(Exception):
            torch.cuda.reset_peak_memory_stats()


def _read_max_input_frames(cli_value: int | None) -> int:
    """Max frames fed to SAM3 on GPU (0 = no cap).

    Priority: CLI --max-frames → env SAM3_MAX_FRAMES → auto-detect from VRAM.
    """
    if cli_value is not None:
        return max(0, int(cli_value))
    return _resolve_max_input_frames_value(cli_value)


def _resolve_max_input_frames_value(
    cli_value: int | None, *, auto_profile: dict[str, float] | None = None
) -> int:
    """Resolve max_input_frames value from CLI/env without repeating VRAM probes."""
    if cli_value is not None:
        return max(0, int(cli_value))
    raw = os.environ.get("SAM3_MAX_FRAMES", "").strip().lower()
    if raw in ("0", "none", "unlimited", "off"):
        return 0
    if raw:
        return max(0, int(raw))
    profile = _sam3_vram_profile() if auto_profile is None else auto_profile
    if profile is None:
        return 32
    return int(profile["safe_frames"])


def _sam3_dry_run_report(
    input_path: Path,
    output_parent: Path,
    *,
    ckpt_opt: Path | None,
    text_prompt: str,
    frame_index: int,
    max_input_frames: int | None,
    save_overlay_mp4: bool,
    save_mask_png: bool,
    frame_by_frame_fallback: bool,
    per_video_output: bool,
    max_videos_to_show: int = 8,
) -> list[str]:
    """Build a dry-run plan for SAM3 without importing/training heavy dependencies."""
    lines: list[str] = []
    lines.append("[SAM3] Dry-run / smoke mode (no inference will run)")
    lines.append(f"input={input_path}")
    lines.append(f"prompt='{text_prompt}' | frame_index={frame_index}")
    lines.append(
        f"frame_by_frame_fallback={'enabled' if frame_by_frame_fallback else 'disabled'}"
        f" | save_overlay_mp4={save_overlay_mp4} | save_mask_png={save_mask_png}"
    )

    vram_profile = _sam3_vram_profile()
    if vram_profile is None:
        lines.append("VRAM profile: CUDA unavailable or not accessible; auto fallback uses 32.")
    else:
        lines.append(
            "VRAM profile: "
            f"total={vram_profile['total_gib']:.2f} GiB, "
            f"free={vram_profile['free_gib']:.2f} GiB, "
            f"safe_auto={int(vram_profile['safe_frames'])} frames"
        )

    try:
        ckpt_file = _resolve_sam3_checkpoint_file(ckpt_opt)
        lines.append(f"checkpoint: {ckpt_file}")
    except Exception as e:
        lines.append(f"checkpoint: unresolved ({e})")

    resolved_initial = _resolve_max_input_frames_value(max_input_frames, auto_profile=vram_profile)
    if max_input_frames is None:
        src = "env SAM3_MAX_FRAMES / auto"
    elif max_input_frames == 0:
        src = "explicit full-clip mode (0)"
    else:
        src = f"CLI value (--max-frames {max_input_frames})"
    lines.append(f"max_input_frames source: {src} (resolved initial={resolved_initial})")

    attempts = _sam3_build_oom_retry_attempts(max_input_frames)
    attempts_display = ", ".join(
        f"{'auto' if cap is None else cap}->"
        f"{_resolve_max_input_frames_value(cap, auto_profile=vram_profile)}"
        for cap in attempts
    )
    lines.append(f"OOM retry ladder (raw->resolved): {attempts_display}")

    video_files = _find_videos(input_path) if input_path.is_dir() else [input_path]
    if not video_files:
        lines.append(f"No video files found under {input_path}")
        return lines

    lines.append(f"Detected videos: {len(video_files)}")
    for idx, vf in enumerate(video_files, start=1):
        if idx > max_videos_to_show:
            lines.append(
                f"... {len(video_files) - max_videos_to_show} additional video(s) not expanded."
            )
            break
        n_frames = _video_frame_count(str(vf))
        if n_frames <= 0:
            lines.append(f"- {vf.name}: frame_count=unknown")
            continue
        retry_frames = [
            n_frames if resolved <= 0 else min(n_frames, resolved)
            for resolved in (
                _resolve_max_input_frames_value(cap, auto_profile=vram_profile) for cap in attempts
            )
        ]
        retry_text = ", ".join(
            f"{'auto' if cap is None else cap}=>{retry_frames[idx2]}"
            for idx2, cap in enumerate(attempts)
        )
        out_dir = output_parent / vf.stem if per_video_output else output_parent
        lines.append(f"- {vf.name}: frame_count={n_frames}, planned_out={out_dir}")
        lines.append(f"  sessions (by retry): {retry_text}")

    lines.append("Checks: no SAM3 initialization, no torch checkpoint load, no predictor start.")
    return lines


def _write_sam3_dry_run_report(lines: list[str], output_parent: Path) -> Path | None:
    """Persist dry-run plan and return report path when possible."""
    report_path = output_parent / "SAM3_DRY_RUN.txt"
    try:
        output_parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        return None
    return report_path


_SAM3_OOM_FALLBACK_FRAMES: tuple[int, ...] = (64, 32, 24, 16, 12, 8, 4, 2, 1)


def _sam3_next_oom_caps(max_input_frames: int | None) -> tuple[int, ...]:
    """Suggested lower frame caps to try after a CUDA OOM."""
    if max_input_frames is None:
        return _SAM3_OOM_FALLBACK_FRAMES
    if max_input_frames <= 0:
        return _SAM3_OOM_FALLBACK_FRAMES
    return tuple(frame for frame in _SAM3_OOM_FALLBACK_FRAMES if frame < max_input_frames)


def _sam3_next_oom_tip(max_input_frames: int | None, *, fallback_to_one: bool = True) -> int:
    """Smallest recommended cap after a failed attempt."""
    caps = _sam3_next_oom_caps(max_input_frames)
    if caps:
        return caps[0]
    return 1 if fallback_to_one else max_input_frames or 1


def _video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, n)


def _maybe_subsample_video_for_vram(
    video_path: Path,
    output_dir: Path,
    max_frames: int,
) -> tuple[Path, Path | None, int]:
    """If the clip has more than ``max_frames``, write a temp MP4 with evenly spaced frames.

    Returns ``(path_for_sam3, temp_path_or_none, num_frames_in_that_path)``.
    SAM3 loads the full tensor to GPU in ``init_state``; capping frames avoids OOM on 8GB cards.
    """
    vp = str(video_path.resolve())
    if max_frames <= 0:
        n = _video_frame_count(vp)
        return video_path.resolve(), None, n

    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {vp}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0:
        n = _video_frame_count(vp)
    if n <= max_frames:
        return video_path.resolve(), None, n

    indices = np.unique(np.linspace(0, n - 1, num=max_frames, dtype=np.int64))
    out_path = output_dir / "_sam3_subsample_input.mp4"
    cap = cv2.VideoCapture(vp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        cap.release()
        raise OSError("Could not open VideoWriter for SAM3 subsample (try another codec/OS path)")
    written = 0
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if ok and frame is not None:
            writer.write(frame)
            written += 1
    cap.release()
    writer.release()
    if written == 0:
        with contextlib.suppress(OSError):
            out_path.unlink(missing_ok=True)
        raise OSError("Subsampled zero frames; check video path/codec")
    return out_path.resolve(), out_path.resolve(), written


def _read_max_input_long_edge(cli_value: int | None) -> int:
    """Max long edge in pixels for frames fed to SAM3. 0 = do not downscale (native resolution).

    **VRAM:** Even 1920×1080 can OOM at ``--max-frames 1`` on 8 GiB cards (model + backbone
    features dominate). When env is unset, we auto-pick **1280** if total GPU VRAM < 10 GiB,
    else **1920**. 4K+ is always scaled down to that cap. Override with ``SAM3_MAX_INPUT_LONG_EDGE``
    or ``--max-input-long-edge``.
    """
    if cli_value is not None:
        return max(0, int(cli_value))
    raw = os.environ.get("SAM3_MAX_INPUT_LONG_EDGE", "").strip().lower()
    if raw in ("0", "none", "unlimited", "off"):
        return 0
    if raw:
        return max(0, int(raw))
    profile = _sam3_vram_profile()
    if profile is not None and profile["total_gib"] < 10.0:
        return 1280
    return 1920


def _maybe_downscale_video_long_edge(
    video_path: Path,
    output_dir: Path,
    max_long_edge: int,
) -> tuple[Path, Path | None, int, int, int]:
    """If max(width,height) > ``max_long_edge``, write a temp MP4 with all frames resized.

    Returns ``(path_for_sam3, temp_path_or_none, out_w, out_h, n_frames)``.
    """
    if max_long_edge <= 0:
        vp = str(video_path.resolve())
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise OSError(f"Could not open video: {vp}")
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n0 <= 0:
            n0 = _video_frame_count(vp)
        return video_path.resolve(), None, w0, h0, n0

    vp = str(video_path.resolve())
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {vp}")
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n0 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n0 <= 0:
        n0 = _video_frame_count(vp)

    m0 = max(w0, h0)
    if m0 <= max_long_edge:
        return video_path.resolve(), None, w0, h0, n0

    scale = max_long_edge / m0
    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    print(
        f"[SAM3] Downscaling input for VRAM: {w0}x{h0} -> {new_w}x{new_h} "
        f"(max long edge {max_long_edge}px; set SAM3_MAX_INPUT_LONG_EDGE=0 to disable)"
    )

    out_path = output_dir / "_sam3_spatial_downscale_input.mp4"
    cap = cv2.VideoCapture(vp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (new_w, new_h))
    if not writer.isOpened():
        cap.release()
        raise OSError("Could not open VideoWriter for SAM3 spatial downscale (try another codec)")

    written = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[1] != new_w or frame.shape[0] != new_h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()

    if written == 0:
        with contextlib.suppress(OSError):
            out_path.unlink(missing_ok=True)
        raise OSError("Spatial downscale wrote zero frames; check video path/codec")
    if written != n0 and n0 > 0:
        n0 = written
    return out_path.resolve(), out_path.resolve(), new_w, new_h, n0


def _split_video_into_chunks(
    video_path: Path,
    chunk_dir: Path,
    chunk_size: int,
) -> list[tuple[Path, int, int]]:
    """Split a video into temporal chunks of ``chunk_size`` frames.

    Returns a list of ``(chunk_mp4_path, start_frame_inclusive, end_frame_exclusive)``
    tuples.  Each chunk is a self-contained MP4 with contiguous frames from the
    original video.

    This is the *divide* half of the divide-and-conquer OOM strategy: when a
    video is too large for SAM3's all-at-once GPU loading, we split it into
    manageable pieces that each fit in VRAM.
    """
    vp = str(video_path.resolve())
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise OSError(f"Could not open video for chunking: {vp}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0:
        n = _video_frame_count(vp)
    if n <= 0:
        raise OSError(f"Video has 0 frames: {vp}")

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, int, int]] = []
    cap = cv2.VideoCapture(vp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]

    for chunk_idx in range(0, n, chunk_size):
        start = chunk_idx
        end = min(chunk_idx + chunk_size, n)
        chunk_path = chunk_dir / f"_chunk_{chunk_idx:06d}.mp4"
        writer = cv2.VideoWriter(str(chunk_path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            cap.release()
            raise OSError(f"Could not open VideoWriter for chunk {chunk_idx}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        written = 0
        for _ in range(start, end):
            ok, frame = cap.read()
            if ok and frame is not None:
                writer.write(frame)
                written += 1
            else:
                break
        writer.release()
        if written > 0:
            chunks.append((chunk_path.resolve(), start, start + written))
    cap.release()
    return chunks


def _merge_chunk_outputs(
    chunks: list[tuple[Path, int, int]],
    chunk_output_dirs: list[Path],
    final_output_dir: Path,
    video_path: Path,
    *,
    save_overlay_mp4: bool = True,
    save_mask_png: bool = True,
) -> None:
    """Merge SAM3 results from temporal chunks into a unified output directory.

    This is the *conquer* half of the divide-and-conquer OOM strategy.
    Mask PNGs are renumbered from chunk-local indices to global frame indices.
    The ``sam_frames_meta.csv`` rows are concatenated with corrected frame numbers.
    An overlay MP4 is stitched from the per-chunk overlays if requested.
    """
    import shutil

    final_masks_dir = final_output_dir / "masks"
    if save_mask_png:
        final_masks_dir.mkdir(parents=True, exist_ok=True)

    all_meta_rows: list[str] = []
    header_line: str = ""
    all_unique_oids: set[int] = set()

    # First pass: collect all object IDs across chunks for a unified header
    for chunk_out in chunk_output_dirs:
        meta_csv = chunk_out / "sam_frames_meta.csv"
        if meta_csv.is_file():
            lines = meta_csv.read_text(encoding="utf-8").strip().split("\n")
            if lines:
                hdr = lines[0]
                # Extract OIDs from header columns like box_x_1, box_y_1, ...
                for col in hdr.split(","):
                    col = col.strip()
                    if col.startswith("box_x_"):
                        with contextlib.suppress(ValueError):
                            all_unique_oids.add(int(col.split("_")[-1]))

    sorted_oids = sorted(all_unique_oids)
    if sorted_oids:
        header_cols = ["frame"]
        for oid in sorted_oids:
            header_cols.extend(
                [f"box_x_{oid}", f"box_y_{oid}", f"box_w_{oid}", f"box_h_{oid}", f"prob_{oid}"]
            )
        header_line = ",".join(header_cols)

    # Second pass: renumber frames and copy masks
    for ci, (chunk_info, chunk_out) in enumerate(zip(chunks, chunk_output_dirs, strict=True)):
        _, start_frame, end_frame = chunk_info
        meta_csv = chunk_out / "sam_frames_meta.csv"
        chunk_masks = chunk_out / "masks"

        if meta_csv.is_file():
            lines = meta_csv.read_text(encoding="utf-8").strip().split("\n")
            chunk_header = lines[0] if lines else ""
            chunk_oid_cols = {}
            # Build a map from chunk column positions to OIDs
            chunk_cols = chunk_header.split(",")
            for col_idx, col in enumerate(chunk_cols):
                col = col.strip()
                if col.startswith("box_x_"):
                    with contextlib.suppress(ValueError):
                        oid = int(col.split("_")[-1])
                        chunk_oid_cols[oid] = col_idx

            for row_line in lines[1:]:
                parts = row_line.split(",")
                if not parts:
                    continue
                try:
                    local_frame = int(parts[0])
                except ValueError:
                    continue
                global_frame = start_frame + local_frame

                # Build unified row
                row_parts = [str(global_frame)]
                for oid in sorted_oids:
                    if oid in chunk_oid_cols:
                        ci_start = chunk_oid_cols[oid]
                        # 5 columns per OID: box_x, box_y, box_w, box_h, prob
                        vals = parts[ci_start : ci_start + 5]
                        while len(vals) < 5:
                            vals.append("")
                        row_parts.extend(vals)
                    else:
                        row_parts.extend(["", "", "", "", ""])
                all_meta_rows.append(",".join(row_parts))

        # Copy and renumber mask PNGs
        if save_mask_png and chunk_masks.is_dir():
            for png in sorted(chunk_masks.glob("frame_*_obj_*.png")):
                # Parse chunk-local frame index: frame_000005_obj_1.png
                parts_name = png.stem.split("_")
                # Expected: frame, NNNNNN, obj, OID
                try:
                    local_idx = int(parts_name[1])
                    obj_suffix = "_".join(parts_name[2:])  # obj_1
                    global_idx = start_frame + local_idx
                    dest = final_masks_dir / f"frame_{global_idx:06d}_{obj_suffix}.png"
                    shutil.copy2(str(png), str(dest))
                except (ValueError, IndexError):
                    # Fallback: just copy with a prefix
                    dest = final_masks_dir / f"chunk{ci}_{png.name}"
                    shutil.copy2(str(png), str(dest))

    # Write unified meta CSV
    if header_line and all_meta_rows:
        meta_path = final_output_dir / "sam_frames_meta.csv"
        # Sort rows by global frame index
        all_meta_rows.sort(key=lambda r: int(r.split(",")[0]) if r.split(",")[0].isdigit() else 0)
        meta_path.write_text(
            header_line + "\n" + "\n".join(all_meta_rows) + "\n",
            encoding="utf-8",
        )

    # Stitch overlay MP4s in order
    if save_overlay_mp4:
        overlay_parts = []
        for chunk_out in chunk_output_dirs:
            overlays = list(chunk_out.glob("*_sam_overlay.mp4"))
            if overlays:
                overlay_parts.append(overlays[0])
        if overlay_parts:
            _stitch_overlay_mp4s(overlay_parts, final_output_dir, video_path)


def _stitch_overlay_mp4s(parts: list[Path], final_output_dir: Path, video_path: Path) -> None:
    """Concatenate multiple overlay MP4 segments into a single file."""
    out_path = final_output_dir / f"{video_path.stem}_sam_overlay.mp4"
    if len(parts) == 1:
        import shutil

        shutil.copy2(str(parts[0]), str(out_path))
        return
    # Use OpenCV to stitch — read all parts and write sequentially
    first_cap = cv2.VideoCapture(str(parts[0]))
    if not first_cap.isOpened():
        return
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        return
    for part in parts:
        cap = cv2.VideoCapture(str(part))
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
        cap.release()
    writer.release()


def _process_video_chunked(
    video_file: Path,
    output_dir: Path,
    *,
    text_prompt: str,
    frame_index: int,
    checkpoint: Path | None,
    max_input_frames: int | None,
    max_input_long_edge: int | None = None,
    save_overlay_mp4: bool,
    save_mask_png: bool,
    chunk_size: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Divide-and-conquer: split video into temporal chunks, process each in a
    subprocess, merge results.

    This is the fallback when the standard OOM retry ladder fails — typically
    because 1920×1080 frames are too large for SAM3 even at max_frames=1.
    Each chunk is a contiguous segment of ``chunk_size`` frames (default: auto
    from VRAM profile).  Processing in separate subprocesses guarantees a clean
    GPU state for each chunk, avoiding the OOM cascade.

    Returns ``(success, message)``.
    """
    import subprocess as _sp

    def _log(s: str) -> None:
        if log is not None:
            log(s)
        else:
            print(s)

    # Determine chunk size
    if chunk_size is None or chunk_size <= 0:
        profile = _sam3_vram_profile()
        chunk_size = max(16, int(profile["safe_frames"])) if profile is not None else 64
    # Safety: ensure chunk_size is at least 8 frames
    chunk_size = max(8, chunk_size)

    n_total = _video_frame_count(str(video_file))
    if n_total <= 0:
        return False, f"Could not read frame count from {video_file}"

    n_chunks = (n_total + chunk_size - 1) // chunk_size
    _log(
        f"  [SAM3-CHUNK] Divide and conquer: {n_total} frames → "
        f"{n_chunks} chunks of ≤{chunk_size} frames each"
    )

    # Create chunk working directory
    chunk_work_dir = output_dir / "_chunks"
    chunk_work_dir.mkdir(parents=True, exist_ok=True)

    # Split video into chunks
    _log("  [SAM3-CHUNK] Splitting video into chunks...")
    try:
        chunks = _split_video_into_chunks(video_file, chunk_work_dir, chunk_size)
    except Exception as e:
        return False, f"Failed to split video: {e}"
    _log(f"  [SAM3-CHUNK] Created {len(chunks)} chunk(s)")

    # Process each chunk in an isolated subprocess
    chunk_output_dirs: list[Path] = []
    failed_chunks: list[int] = []
    for ci, (chunk_path, start_frame, end_frame) in enumerate(chunks):
        chunk_out = chunk_work_dir / f"out_{start_frame:06d}"
        chunk_out.mkdir(parents=True, exist_ok=True)
        chunk_output_dirs.append(chunk_out)

        _log(
            f"  [SAM3-CHUNK] Processing chunk {ci + 1}/{len(chunks)}: "
            f"frames {start_frame}-{end_frame - 1} ({end_frame - start_frame} frames)"
        )

        # Use frame_index=0 for each chunk (prompt at start of chunk)
        # For the chunk that contains the original frame_index, use the offset
        chunk_fi = 0
        if start_frame <= frame_index < end_frame:
            chunk_fi = frame_index - start_frame

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--input",
            str(chunk_path),
            "--video-output-dir",
            str(chunk_out),
            "--text",
            text_prompt,
            "--frame",
            str(chunk_fi),
        ]
        if max_input_frames is not None and max_input_frames > 0:
            cmd += ["--max-frames", str(max_input_frames)]
        if max_input_long_edge is not None:
            cmd += ["--max-input-long-edge", str(max_input_long_edge)]
        if checkpoint is not None:
            cmd += ["--checkpoint", str(checkpoint)]
        if not save_overlay_mp4:
            cmd.append("--no-overlay")
        if not save_mask_png:
            cmd.append("--no-png")

        env = os.environ.copy()
        env.setdefault("TQDM_DISABLE", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            rc = _sp.call(cmd, env=env)
        except KeyboardInterrupt:
            _log(f"  [SAM3-CHUNK] Interrupted at chunk {ci + 1}")
            failed_chunks.append(ci)
            break
        except Exception as e:
            _log(f"  [SAM3-CHUNK] Exception on chunk {ci + 1}: {e}")
            failed_chunks.append(ci)
            continue

        if rc == 0:
            _log(f"  [SAM3-CHUNK] Chunk {ci + 1}/{len(chunks)} done")
        else:
            _log(f"  [SAM3-CHUNK] Chunk {ci + 1}/{len(chunks)} FAILED (exit={rc})")
            failed_chunks.append(ci)

        # Clean up chunk MP4 to save disk space
        with contextlib.suppress(OSError):
            chunk_path.unlink(missing_ok=True)

    # Merge results
    successful_chunks = len(chunks) - len(failed_chunks)
    if successful_chunks == 0:
        _write_failure_marker(output_dir, video_file, "All chunks failed (CUDA OOM)")
        return False, "All chunks failed"

    _log(f"  [SAM3-CHUNK] Merging {successful_chunks}/{len(chunks)} chunk results...")
    try:
        _merge_chunk_outputs(
            chunks,
            chunk_output_dirs,
            output_dir,
            video_file,
            save_overlay_mp4=save_overlay_mp4,
            save_mask_png=save_mask_png,
        )
    except Exception as e:
        _log(f"  [SAM3-CHUNK] Merge error: {e}")
        return False, f"Chunk merge failed: {e}"

    # Write README
    readme = output_dir / "README_sam.txt"
    readme.write_text(
        f"SAM 3 video export (chunked divide-and-conquer)\n"
        f"source={video_file.resolve()}\n"
        f"total_frames={n_total}\n"
        f"chunk_size={chunk_size}\n"
        f"total_chunks={len(chunks)}\n"
        f"successful_chunks={successful_chunks}\n"
        f"failed_chunks={len(failed_chunks)}\n"
        f"prompt={text_prompt!r}\n",
        encoding="utf-8",
    )

    # Clean up chunk work dir (keep outputs in final dir)
    import shutil

    with contextlib.suppress(OSError):
        shutil.rmtree(str(chunk_work_dir), ignore_errors=True)

    if failed_chunks:
        _log(
            f"  [SAM3-CHUNK] Warning: {len(failed_chunks)} chunk(s) failed, "
            f"but {successful_chunks} succeeded → partial result available"
        )
        return True, f"Partial: {successful_chunks}/{len(chunks)} chunks OK"
    _log(f"  [SAM3-CHUNK] All {len(chunks)} chunks merged successfully")
    return True, ""


def _palette(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return (rng.random((max(n, 1), 3)) * 255).astype(np.uint8)


def _composite_masks_bgr(
    frame_bgr: np.ndarray,
    binary_masks: np.ndarray,
    obj_ids: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """binary_masks: (N, H, W) bool; obj_ids: (N,) int."""
    out = frame_bgr.copy().astype(np.float32)
    if binary_masks.size == 0:
        return out.astype(np.uint8)
    colors = _palette(int(obj_ids.max()) + 1 if obj_ids.size else 1)
    h, w = frame_bgr.shape[:2]
    for i in range(binary_masks.shape[0]):
        m = binary_masks[i]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        oid = int(obj_ids[i]) if i < len(obj_ids) else i
        c = colors[oid % len(colors)].astype(np.float32)
        mo = m[..., None].astype(np.float32)
        out = out * (1 - alpha * mo) + c * (alpha * mo)
    return np.clip(out, 0, 255).astype(np.uint8)


def _sam3_cuda_oom_help(
    *,
    max_input_frames: int,
    session_frames: int,
    max_input_long_edge: int = 0,
) -> str:
    followups = _sam3_next_oom_caps(max_input_frames)
    if followups:
        followup_hint = ", ".join(map(str, followups[:4]))
        followup_text = (
            "Lower values use less VRAM. If this keeps failing, try in this order: "
            f"{followup_hint}..."
        )
    else:
        followup_text = "Lower values use less VRAM."

    if max_input_frames > 0:
        max_hint = f"or pass --max-frames with a smaller value than {max_input_frames}"
    elif max_input_frames == 0:
        max_hint = (
            "or pass an explicit --max-frames (for example 64, then 32, then 24...) "
            "instead of keeping the full clip in memory"
        )
    else:
        max_hint = "or pass a smaller --max-frames value"

    spatial = ""
    if session_frames <= 1 or max_input_frames <= 1:
        spatial = (
            "\n\nAt 1 (or very few) session frame(s), OOM is usually input resolution (4K/8K), "
            "not frame count. Try:\n"
            "  - --max-input-long-edge 1280 (or 960, 640) or env SAM3_MAX_INPUT_LONG_EDGE=1280\n"
            "  - or --frame-by-frame (per-frame subprocess with 640p cap)\n"
        )
    if max_input_long_edge > 0 and session_frames <= 1:
        spatial += (
            f"\nCurrent max long edge in effect: {max_input_long_edge}px (try a lower value).\n"
        )

    return (
        "CUDA out of memory while loading the video into SAM3. "
        "The library puts the whole session on GPU at once (~few GiB per long clip; "
        "very large frame sizes at 4K+ also cost a lot of VRAM per frame at 1008² features).\n\n"
        f"This run used session_frames={session_frames} (max_input_frames_cap={max_input_frames}; "
        "0 means no subsampling). "
        f"max_input_long_edge={max_input_long_edge} (0 = native / no downscale)."
        f"{spatial}\n"
        "Fix: unset SAM3_MAX_FRAMES (auto-sizes to GPU VRAM), "
        f"{max_hint}. "
        f"{followup_text} Keep weights under vaila/models/sam3/."
    )


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc)
    return type(exc).__name__ == "OutOfMemoryError" or "CUDA out of memory" in msg


def _setup_cuda_for_sam3() -> None:
    """Configure PyTorch CUDA allocator for SAM3's large contiguous allocations.

    ``expandable_segments`` avoids fragmentation that causes OOM even when the
    total free VRAM looks sufficient.  Without it, 8 GiB cards routinely fail
    at the ``start_session`` / ``add_prompt`` step where SAM3 allocates a
    ~900 MiB feature tensor.
    """
    import torch

    # PyTorch 2.9+ renamed the env var; set both for compat
    alloc_val = "expandable_segments:True"
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        cur = os.environ.get(key, "")
        if "expandable_segments" not in cur:
            os.environ[key] = f"{cur},{alloc_val}" if cur else alloc_val

    torch.cuda.empty_cache()


def run_sam3_on_video(
    video_path: Path,
    output_dir: Path,
    text_prompt: str,
    frame_index: int = 0,
    *,
    checkpoint: Path | None = None,
    max_input_frames: int | None = None,
    max_input_long_edge: int | None = None,
    save_overlay_mp4: bool = True,
    save_mask_png: bool = True,
    frame_by_frame_fallback: bool = False,
) -> None:
    import torch

    _patch_sam3_disable_perpetual_tracker_autocast()
    _patch_sam3_force_fp32_tracker_backbone_features()
    try:
        from sam3.model_builder import (  # type: ignore[reportMissingImports]
            build_sam3_video_predictor,
        )
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "Optional dependency 'sam3' is not installed. Install the SAM 3 stack with "
            "`uv sync --extra sam` (and on NVIDIA workstations also switch to the CUDA "
            "pyproject template + `uv sync --extra gpu`)."
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3 video predictor requires CUDA. No GPU available.")

    _setup_cuda_for_sam3()

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_spatial: Path | None = None
    mf = _read_max_input_frames(max_input_frames)
    session_path, temp_clip, n_sess = _maybe_subsample_video_for_vram(video_path, output_dir, mf)
    le_cap = _read_max_input_long_edge(max_input_long_edge)
    if max_input_long_edge is None and le_cap == 1280:
        _vprof = _sam3_vram_profile()
        if _vprof is not None:
            print(
                f"[SAM3] Default max_input_long_edge=1280 (GPU total {_vprof['total_gib']:.1f} GiB; "
                "set SAM3_MAX_INPUT_LONG_EDGE=1920 to try full HD input if you have VRAM headroom)"
            )
    session_path, temp_spatial, _w_sess, _h_sess, n_sess = _maybe_downscale_video_long_edge(
        session_path, output_dir, le_cap
    )
    vp = str(session_path)
    vp_orig = str(video_path.resolve())
    fi_run = min(max(0, int(frame_index)), max(0, n_sess - 1))

    bpe_path = _resolve_bpe_path()
    ckpt_file = _resolve_sam3_checkpoint_file(checkpoint)
    pred_kw: dict = {
        "bpe_path": str(bpe_path),
        "gpus_to_use": [torch.cuda.current_device()],
    }
    if ckpt_file is not None:
        pred_kw["checkpoint_path"] = str(ckpt_file)

    try:
        predictor: Any = None
        outputs_by_frame: dict[int, dict] = {}
        _autocast: contextlib.AbstractContextManager[object] = contextlib.nullcontext()

        if not frame_by_frame_fallback:
            try:
                predictor = build_sam3_video_predictor(**pred_kw)
            except Exception as e:
                if _is_gated_repo_error(e):
                    raise RuntimeError(_hf_access_help()) from e
                raise

            # On low-VRAM cards (< 10 GiB), FP32 model (~6.8 GiB) leaves almost no
            # room for inference tensors.  Use autocast to run matmuls in bfloat16
            # while keeping input tensors in float32 (avoids dtype mismatch errors).
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            _use_bf16 = props.total_memory / (1024**3) < 10.0
            if _use_bf16:
                print(
                    f"[SAM3] Low-VRAM GPU ({props.total_memory / (1024**3):.1f} GiB) — "
                    "using bfloat16 autocast for inference"
                )

            import gc

            gc.collect()
            torch.cuda.empty_cache()
            _autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=_use_bf16)

        if frame_by_frame_fallback:
            print(
                "[SAM3] Frame-by-frame fallback activated: processing each frame via isolated subprocesses."
            )
            import subprocess

            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                raise OSError(f"Could not open video: {vp}")
            nframes_sess = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_sess = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w_sess = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_sess = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            for fi_eval in range(nframes_sess):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi_eval)
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue

                scale = 1.0
                max_dim = 640
                if max(w_sess, h_sess) > max_dim:
                    scale = max_dim / max(w_sess, h_sess)
                    new_w = max(1, int(round(w_sess * scale)))
                    new_h = max(1, int(round(h_sess * scale)))
                    bgr_small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    new_w, new_h = w_sess, h_sess
                    bgr_small = bgr

                temp_dir = output_dir / f"_fallback_tmp_{fi_eval}"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_vid = temp_dir / f"frame_{fi_eval}.mp4"

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
                tmp_writer = cv2.VideoWriter(str(temp_vid), fourcc, float(fps_sess), (new_w, new_h))
                tmp_writer.write(bgr_small)
                tmp_writer.release()

                print(f"  [SAM3] Isolating frame {fi_eval + 1}/{nframes_sess} in fresh process...")
                cmd = [
                    sys.executable,
                    __file__,
                    "-i",
                    str(temp_vid),
                    "-o",
                    str(temp_dir),
                    "-t",
                    text_prompt,
                    "--max-frames",
                    "1",
                    "--no-overlay",
                ]
                if checkpoint is not None:
                    cmd.extend(["-w", str(checkpoint)])

                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"  [SAM3] Process failed for frame {fi_eval}:\n{e.stderr}")
                    continue

                # Parse output to rebuild the tracker dictionary
                found_masks = []
                found_oids = []
                found_probs = []
                found_boxes = []

                meta_csvs = list(temp_dir.rglob("sam_frames_meta.csv"))
                if meta_csvs:
                    meta_path = meta_csvs[0]
                    masks_dir = meta_path.parent / "masks"
                    try:
                        lines = meta_path.read_text(encoding="utf-8").strip().split("\n")
                        if len(lines) > 1:
                            header = lines[0].split(",")
                            data = lines[1].split(",")
                            for col_idx in range(1, len(header), 5):
                                if col_idx + 4 < len(data):
                                    col_name = header[col_idx]
                                    oid_str = col_name.split("_")[-1]
                                    try:
                                        oid = int(oid_str)
                                    except ValueError:
                                        continue

                                    if not data[col_idx]:
                                        continue

                                    bx, by, bw, bh = map(float, data[col_idx : col_idx + 4])
                                    prob = float(data[col_idx + 4])

                                    png_file = masks_dir / f"frame_{fi_eval:06d}_obj_{oid}.png"
                                    if png_file.is_file():
                                        mask_img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
                                        if mask_img is not None:
                                            if scale != 1.0:
                                                mask_img = cv2.resize(
                                                    mask_img,
                                                    (w_sess, h_sess),
                                                    interpolation=cv2.INTER_NEAREST,
                                                )
                                            found_masks.append(mask_img > 127)
                                            found_oids.append(oid)
                                            found_probs.append(prob)
                                            found_boxes.append([bx, by, bw, bh])

                        if found_masks:
                            outputs_by_frame[fi_eval] = {
                                "out_binary_masks": np.stack(found_masks),
                                "out_obj_ids": np.array(found_oids, dtype=np.int32),
                                "out_probs": np.array(found_probs, dtype=np.float32),
                                "out_boxes_xywh": np.array(found_boxes, dtype=np.float32),
                            }
                    except Exception as e:
                        print(f"  [SAM3] Failed to parse mask for frame {fi_eval}: {e}")

                with contextlib.suppress(OSError):
                    pass
                    # shutil.rmtree(temp_dir, ignore_errors=True)

            cap.release()

        else:
            if predictor is None:
                raise RuntimeError("SAM3 predictor failed to initialize.")
            try:
                with _autocast:
                    start = predictor.handle_request(
                        {"type": "start_session", "resource_path": vp},
                    )
            except Exception as e:
                if type(e).__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(e):
                    raise RuntimeError(
                        _sam3_cuda_oom_help(
                            max_input_frames=mf,
                            session_frames=n_sess,
                            max_input_long_edge=le_cap,
                        )
                    ) from e
                raise
            session_id = start["session_id"]

            gc.collect()
            torch.cuda.empty_cache()

            try:
                with _autocast:
                    prompt_resp = predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": fi_run,
                            "text": text_prompt,
                        }
                    )
                fi0 = int(prompt_resp["frame_index"])
                outputs_by_frame[fi0] = prompt_resp["outputs"]

                stream_req = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                    "start_frame_index": fi_run,
                    "max_frame_num_to_track": None,
                }
                with _autocast:
                    for chunk in predictor.handle_stream_request(stream_req):
                        outputs_by_frame[int(chunk["frame_index"])] = chunk["outputs"]
            except Exception as e:
                if _is_cuda_oom_error(e):
                    tip_n = _sam3_next_oom_tip(mf)
                    msg = (
                        _sam3_cuda_oom_help(
                            max_input_frames=mf,
                            session_frames=n_sess,
                            max_input_long_edge=le_cap,
                        )
                        + f"\n\nTip: rerun with --max-frames {tip_n} (or another lower value until it fits)."
                    )
                    raise RuntimeError(msg) from e
                raise

            with contextlib.suppress(Exception):
                predictor.handle_request({"type": "close_session", "session_id": session_id})

        if predictor is not None:
            with contextlib.suppress(Exception):
                predictor.shutdown()
            predictor = None
            _release_sam3_gpu_memory()

        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            raise OSError(f"Could not open video: {vp}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        writer = None
        overlay_path = output_dir / f"{video_path.stem}_sam_overlay.mp4"
        if save_overlay_mp4:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
            writer = cv2.VideoWriter(str(overlay_path), fourcc, float(fps), (w, h))
            if not writer.isOpened():
                writer = None

        masks_dir = output_dir / "masks"
        if save_mask_png:
            masks_dir.mkdir(parents=True, exist_ok=True)

        all_unique_oids = set()
        for idx_eval in outputs_by_frame:
            out = outputs_by_frame[idx_eval]
            oids = out.get("out_obj_ids")
            if oids is not None:
                for oid in oids:
                    all_unique_oids.add(int(oid))

        sorted_oids = sorted(all_unique_oids)
        header_cols = ["frame"]
        for oid in sorted_oids:
            header_cols.extend(
                [f"box_x_{oid}", f"box_y_{oid}", f"box_w_{oid}", f"box_h_{oid}", f"prob_{oid}"]
            )

        meta_rows_wide: list[str] = []

        cap = cv2.VideoCapture(vp)
        nframes_to_write = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_idx in range(nframes_to_write):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue

            out = outputs_by_frame.get(frame_idx)
            if out is None:
                if writer is not None:
                    writer.write(bgr)
                continue

            masks = out.get("out_binary_masks")
            oids = out.get("out_obj_ids")
            probs = out.get("out_probs")
            boxes = out.get("out_boxes_xywh")
            if masks is None or oids is None or masks.size == 0:
                if writer is not None:
                    writer.write(bgr)
                continue
            comp = _composite_masks_bgr(bgr, masks, oids)
            if writer is not None:
                writer.write(comp)
            frame_data = {}
            for i in range(masks.shape[0]):
                oid = int(oids[i])
                if save_mask_png:
                    png = masks_dir / f"frame_{frame_idx:06d}_obj_{oid}.png"
                    cv2.imwrite(str(png), (masks[i].astype(np.uint8)) * 255)
                pr = float(probs[i]) if probs is not None and i < len(probs) else float("nan")
                bx = by = bw = bh = float("nan")
                if boxes is not None and i < len(boxes):
                    bx, by, bw, bh = (float(x) for x in boxes[i])
                frame_data[oid] = (bx, by, bw, bh, pr)

            row_cols = [str(frame_idx)]
            for oid in sorted_oids:
                if oid in frame_data:
                    bx, by, bw, bh, pr = frame_data[oid]
                    row_cols.extend(
                        [f"{bx:.6f}", f"{by:.6f}", f"{bw:.6f}", f"{bh:.6f}", f"{pr:.6f}"]
                    )
                else:
                    row_cols.extend(["", "", "", "", ""])

            meta_rows_wide.append(",".join(row_cols))

        cap.release()
        if writer is not None:
            writer.release()

        meta_path = output_dir / "sam_frames_meta.csv"
        meta_path.write_text(
            ",".join(header_cols) + "\n" + "\n".join(meta_rows_wide) + "\n",
            encoding="utf-8",
        )

        readme = output_dir / "README_sam.txt"
        ckpt_note = (
            str(ckpt_file) if ckpt_file is not None else "Hugging Face facebook/sam3 (cached)"
        )
        readme.write_text(
            f"SAM 3 video export\n"
            f"source_original={vp_orig}\n"
            f"session_resource={vp}\n"
            f"subsampled_to_disk={temp_clip is not None} spatial_downscale={temp_spatial is not None} "
            f"max_input_long_edge_cap={le_cap} max_input_frames_cap={mf} session_frames={n_sess}\n"
            f"checkpoint={ckpt_note}\n"
            f"prompt={text_prompt!r}\n"
            f"prompt_frame_requested={frame_index} prompt_frame_used={fi_run}\n"
            f"frames_with_outputs={len(outputs_by_frame)} / {nframes}\n",
            encoding="utf-8",
        )
    finally:
        for _tmp in (temp_clip, temp_spatial):
            if _tmp is not None and Path(_tmp).is_file():
                with contextlib.suppress(OSError):
                    Path(_tmp).unlink(missing_ok=True)
        # Last-line GPU cleanup: even if the run raised, drop the predictor
        # and reset CUDA caches so the next batch item starts from a clean slate.
        with contextlib.suppress(Exception):
            if "predictor" in locals() and locals().get("predictor") is not None:
                predictor = None  # type: ignore[assignment]
        with contextlib.suppress(Exception):
            outputs_by_frame.clear()
        _release_sam3_gpu_memory()


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# SAM 3 is open-vocabulary — any free-text prompt is valid. These presets only
# populate the GUI combobox for convenience; ``state="normal"`` lets the user
# type anything.  Update this list when adding domain-specific scenarios.
SAM3_PROMPT_PRESETS: tuple[str, ...] = (
    "person",
    "player",
    "goalkeeper",
    "referee",
    "ball",
    "soccer ball",
    "basketball",
    "volleyball",
    "coach",
    "crowd",
    "car",
    "bike",
    "dog",
    "cat",
)


def _agent_dbg_sam3(*, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    try:
        line = (
            json.dumps(
                {
                    "sessionId": "42b4a5",
                    "hypothesisId": hypothesis_id,
                    "location": location,
                    "message": message,
                    "data": data,
                    "timestamp": int(time.time() * 1000),
                },
                default=str,
            )
            + "\n"
        )
        with Path("/home/preto/data/vaila/.cursor/debug-42b4a5.log").open(
            "a", encoding="utf-8"
        ) as _lf:
            _lf.write(line)
    except OSError:
        pass
    # #endregion


def _sam3_build_oom_retry_attempts(max_input_frames: int | None) -> list[int | None]:
    """Caps to try on CUDA OOM (high → low). Includes low caps like 8→4→2→1."""
    attempts: list[int | None] = [max_input_frames]
    for step in _SAM3_OOM_FALLBACK_FRAMES:
        if max_input_frames is None or max_input_frames <= 0 or max_input_frames > step:
            attempts.append(step)
    seen: set[int | None] = set()
    uniq: list[int | None] = []
    for a in attempts:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def _write_failure_marker(output_dir: Path, video_path: Path, reason: str) -> None:
    """Mark a video as failed inside its (otherwise empty) output dir."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        marker = output_dir / "FAILED_sam.txt"
        marker.write_text(
            f"SAM 3 video FAILED\n"
            f"video={video_path.resolve()}\n"
            f"timestamp={dt.datetime.now().isoformat(timespec='seconds')}\n"
            f"reason={reason}\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _process_one_video_with_oom_retry(
    video_file: Path,
    output_dir: Path,
    *,
    text_prompt: str,
    frame_index: int,
    checkpoint: Path | None,
    max_input_frames: int | None,
    max_input_long_edge: int | None = None,
    save_overlay_mp4: bool,
    save_mask_png: bool,
    frame_by_frame_fallback: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Run one video; on CUDA OOM retry with descending frame caps
    (e.g. auto→64→32→24→16→12→8→4→2→1).

    Returns ``(success, message)``.  ``message`` is empty on success.

    Note (debug session 42b4a5, Apr 2026): ``_release_sam3_gpu_memory()`` is
    called at the top of each attempt, NOT inside the ``except`` block.  Runtime
    logs proved the ``except``-block call is a no-op while ``e`` is alive — its
    traceback pins ~7 GiB of inner SAM3 tensors that Python's gc cannot reach
    until ``e`` is auto-deleted at the end of the except scope.  An additional
    ~13 GiB of C++-side orphan tensors on OOM is unrecoverable in-process, which
    is why the CLI batch loop spawns one subprocess per video.  See
    ``.claude/skills/sam3-video/SKILL.md`` § *Why subprocess-per-video*.
    """

    def _log(s: str) -> None:
        if log is not None:
            log(s)
        else:
            print(s)

    attempts = _sam3_build_oom_retry_attempts(max_input_frames)
    # #region agent log
    _agent_dbg_sam3(
        hypothesis_id="H1",
        location="vaila_sam.py:_process_one_video_with_oom_retry",
        message="oom_retry_attempts_built",
        data={"max_input_frames": max_input_frames, "attempts": attempts, "n": len(attempts)},
    )
    # #endregion

    last_err: str = ""
    mf_try: int | None = None
    for attempt_idx, mf_try in enumerate(attempts, start=1):
        _release_sam3_gpu_memory()
        mf_try_resolved = _read_max_input_frames(mf_try)
        try:
            run_sam3_on_video(
                video_file,
                output_dir,
                text_prompt,
                frame_index=frame_index,
                checkpoint=checkpoint,
                max_input_frames=mf_try,
                max_input_long_edge=max_input_long_edge,
                save_overlay_mp4=save_overlay_mp4,
                save_mask_png=save_mask_png,
                frame_by_frame_fallback=frame_by_frame_fallback,
            )
            return True, ""
        except Exception as e:
            last_err = str(e)
            if not _is_cuda_oom_error(e):
                _write_failure_marker(output_dir, video_file, last_err)
                return False, last_err
            if attempt_idx >= len(attempts):
                # Fall through to chunking fallback
                break
            next_mf = attempts[attempt_idx]
            next_tip = (
                next_mf
                if isinstance(next_mf, int) and next_mf > 0
                else _sam3_next_oom_tip(mf_try_resolved)
            )
            # #region agent log
            _agent_dbg_sam3(
                hypothesis_id="H1",
                location="vaila_sam.py:_process_one_video_with_oom_retry",
                message="cuda_oom_retrying",
                data={
                    "attempt_idx": attempt_idx,
                    "len_attempts": len(attempts),
                    "mf_try": mf_try,
                    "next_mf": next_mf,
                    "video": str(video_file),
                },
            )
            # #endregion
            _log(
                f"  [SAM3] CUDA OOM at max_frames={mf_try_resolved}; "
                f"retrying with max_frames={next_tip}..."
            )
    # Frame-cap ladder exhausted; 4K+ sources often OOM at session_frames=1 — lower long edge.
    smf = mf_try if mf_try is not None else (attempts[-1] if attempts else 1)
    user_le = _read_max_input_long_edge(max_input_long_edge)
    for le in (1280, 960, 640, 512):
        if user_le > 0 and le >= user_le:
            continue
        _release_sam3_gpu_memory()
        _log(
            f"  [SAM3] CUDA OOM persists for {video_file.name}; "
            f"retrying with max_input_long_edge={le}px (spatial downscale) "
            f"and max_input_frames={smf!r}..."
        )
        try:
            run_sam3_on_video(
                video_file,
                output_dir,
                text_prompt,
                frame_index=frame_index,
                checkpoint=checkpoint,
                max_input_frames=smf,
                max_input_long_edge=le,
                save_overlay_mp4=save_overlay_mp4,
                save_mask_png=save_mask_png,
                frame_by_frame_fallback=frame_by_frame_fallback,
            )
            return True, ""
        except Exception as e:
            last_err = str(e)
            if not _is_cuda_oom_error(e):
                _write_failure_marker(output_dir, video_file, last_err)
                return False, last_err
    # All retry attempts exhausted — fall back to divide-and-conquer chunking.
    # This splits the video into temporal segments small enough for VRAM,
    # processes each in an isolated subprocess, then merges the masks.
    _log(
        f"  [SAM3] All OOM retry attempts exhausted for {video_file.name}; "
        f"falling back to divide-and-conquer chunking..."
    )
    _release_sam3_gpu_memory()
    ok, chunk_msg = _process_video_chunked(
        video_file,
        output_dir,
        text_prompt=text_prompt,
        frame_index=frame_index,
        checkpoint=checkpoint,
        max_input_frames=max_input_frames,
        max_input_long_edge=max_input_long_edge,
        save_overlay_mp4=save_overlay_mp4,
        save_mask_png=save_mask_png,
        log=log,
    )
    if ok:
        _log(f"  [SAM3] Chunked processing succeeded for {video_file.name}")
        return True, chunk_msg
    _write_failure_marker(output_dir, video_file, f"Chunked fallback failed: {chunk_msg}")
    return False, f"Chunked fallback failed: {chunk_msg}"


def _find_videos(directory: Path) -> list[Path]:
    """Return sorted list of video files in *directory* (non-recursive)."""
    return sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()),
        key=lambda p: p.name.lower(),
    )


class SamVideoDialog(tk.Toplevel):
    """Configure SAM 3 segmentation — supports a directory of videos (batch) or a single file."""

    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.title("SAM 3 — video segmentation")
        self.result: tuple[Path, Path, Path | None, str, int, bool, bool, bool, bool] | None = None

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Label(
            frm,
            text=(
                "Select an input directory with videos (batch) or a single video file.\n"
                "Requires: uv sync --extra sam  |  NVIDIA CUDA  |  HF: accept facebook/sam3 + hf auth login\n"
                "Checkpoint: video weights only (facebook/sam3 → sam3.pt / sam3.1_multiplex.pt).\n"
                "Not FIFA / sam-3d-dinov3. CLI equivalent: -w / --weights / --checkpoint.\n"
                "Results folder: <output>/processed_sam_YYYYMMDD_HHMMSS/ (created under Output folder).\n"
                "VRAM: frame cap auto-sized to GPU; override with SAM3_MAX_FRAMES or --max-frames."
            ),
            wraplength=520,
            justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(frm, text="Input (dir or file):").grid(row=1, column=0, sticky="w", pady=4)
        self.input_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.input_var, width=52).grid(
            row=1, column=1, sticky="ew", pady=4
        )
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=1, column=2, padx=4)
        ttk.Button(btn_frame, text="Dir…", command=self._browse_dir).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="File…", command=self._browse_file).pack(side=tk.LEFT, padx=1)

        ttk.Label(frm, text="Output folder:").grid(row=2, column=0, sticky="w", pady=4)
        self.out_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.out_var, width=52).grid(
            row=2, column=1, sticky="ew", pady=4
        )
        ttk.Button(frm, text="Browse…", command=self._browse_out).grid(row=2, column=2, padx=4)

        ttk.Label(frm, text="sam3.pt / weights (-w):").grid(row=3, column=0, sticky="nw", pady=4)
        self.ckpt_var = tk.StringVar(value=os.environ.get("SAM3_CHECKPOINT", ""))
        ttk.Entry(frm, textvariable=self.ckpt_var, width=52).grid(
            row=3, column=1, sticky="ew", pady=4
        )
        ttk.Button(frm, text="Browse…", command=self._browse_ckpt).grid(row=3, column=2, padx=4)

        ttk.Label(frm, text="Text prompt:").grid(row=4, column=0, sticky="w", pady=4)
        self.prompt_var = tk.StringVar(value="person")
        # Open-vocabulary: these are *examples* — any free-text prompt is valid.
        self.prompt_combo = ttk.Combobox(
            frm,
            textvariable=self.prompt_var,
            values=SAM3_PROMPT_PRESETS,
            width=50,
            state="normal",  # editable — type your own prompt
        )
        self.prompt_combo.grid(row=4, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(frm, text="Prompt frame index:").grid(row=5, column=0, sticky="w", pady=4)
        self.frame_var = tk.StringVar(value="0")
        ttk.Entry(frm, textvariable=self.frame_var, width=12).grid(
            row=5, column=1, sticky="w", pady=4
        )

        self.overlay_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=True)
        self.fallback_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm,
            text="Save overlay MP4 (video with colored masks on top of frames)",
            variable=self.overlay_var,
        ).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Save mask PNGs (per object)", variable=self.png_var).grid(
            row=7, column=1, sticky="w"
        )
        ttk.Checkbutton(
            frm,
            text="Fallback: Frame-by-Frame (CUDA only; lower VRAM, slower, no temporal tracking)",
            variable=self.fallback_var,
        ).grid(row=8, column=1, sticky="w")
        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm,
            text="Dry-run / smoke (show plan only, do not run SAM3)",
            variable=self.dry_run_var,
        ).grid(row=9, column=1, sticky="w")

        btns = ttk.Frame(frm)
        btns.grid(row=10, column=0, columnspan=3, pady=12)
        ttk.Button(btns, text="Run", command=self._on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            btns,
            text="Help",
            command=open_sam3_install_help_in_browser,
        ).pack(side=tk.LEFT, padx=4)

        frm.columnconfigure(1, weight=1)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.transient(parent)  # ty: ignore[no-matching-overload]
        self.grab_set()
        self.after(50, self._bring_to_front)

    def _bring_to_front(self) -> None:
        try:
            self.lift()
            self.attributes("-topmost", True)
            self.focus_force()
            self.after(400, lambda: self.attributes("-topmost", False))
        except tk.TclError:
            pass

    def _browse_dir(self) -> None:
        p = filedialog.askdirectory(parent=self, title="Select directory with videos")
        if p:
            self.input_var.set(p)
            if not self.out_var.get().strip():
                self.out_var.set(p)

    def _browse_file(self) -> None:
        p = filedialog.askopenfilename(
            parent=self,
            title="Select a single video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")],
        )
        if p:
            self.input_var.set(p)
            if not self.out_var.get().strip():
                self.out_var.set(str(Path(p).parent))

    def _browse_out(self) -> None:
        p = filedialog.askdirectory(parent=self, title="Output folder")
        if p:
            self.out_var.set(p)

    def _browse_ckpt(self) -> None:
        p = filedialog.askopenfilename(
            parent=self,
            title="SAM3 checkpoint (sam3.pt)",
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")],
        )
        if p:
            self.ckpt_var.set(p)

    def _on_ok(self) -> None:
        v = self.input_var.get().strip()
        o = self.out_var.get().strip()
        if not v or not o:
            messagebox.showerror(
                "Error", "Select input (dir or file) and output folder.", parent=self
            )
            return
        inp = Path(v)
        if inp.is_dir():
            videos = _find_videos(inp)
            if not videos:
                messagebox.showerror(
                    "No videos found",
                    f"No video files ({', '.join(VIDEO_EXTENSIONS)}) in:\n{inp}",
                    parent=self,
                )
                return
        elif not inp.is_file():
            messagebox.showerror("Error", f"Path does not exist:\n{inp}", parent=self)
            return
        try:
            fi = int(self.frame_var.get().strip())
        except ValueError:
            messagebox.showerror("Error", "Frame index must be an integer.", parent=self)
            return
        ck = self.ckpt_var.get().strip()
        ckpt_path: Path | None = Path(ck) if ck else None
        self.result = (
            inp,
            Path(o),
            ckpt_path,
            self.prompt_var.get().strip() or "person",
            fi,
            self.overlay_var.get(),
            self.png_var.get(),
            self.fallback_var.get(),
            self.dry_run_var.get(),
        )
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        with contextlib.suppress(tk.TclError):
            self.grab_release()
        self.destroy()


class SamBatchProgress(tk.Toplevel):
    """Modal-ish progress window for SAM3 GUI batch (per-video status + log + cancel).

    The actual SAM3 work runs on a background thread; this window only mirrors
    log lines from the worker via ``schedule_log`` (thread-safe through ``after``).
    """

    def __init__(self, parent: tk.Misc, total: int, *, output_base: Path | None = None) -> None:
        super().__init__(parent)
        self.title("SAM 3 — batch progress")
        self.geometry("760x560")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.cancelled = False
        self._total = max(1, total)
        self._output_base = output_base
        self._parent = parent

        frm = ttk.Frame(self, padding=8)
        frm.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value=f"0 / {total} processed")
        ttk.Label(frm, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )

        self.progress = ttk.Progressbar(frm, maximum=self._total, value=0)
        self.progress.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(frm, text="Log:").pack(anchor="w")
        log_frame = ttk.Frame(frm)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_frame, height=18, wrap="none", state="disabled")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)

        post_frame = ttk.LabelFrame(frm, text="Post-processing (enabled after batch finishes)")
        post_frame.pack(fill=tk.X, pady=(6, 0))
        self.postproc_btn = ttk.Button(
            post_frame,
            text="Build sam_points.csv (foot+center+mask)",
            command=self._on_build_points,
            state="disabled",
        )
        self.postproc_btn.pack(side=tk.LEFT, padx=4, pady=4)
        self.calib_btn = ttk.Button(
            post_frame,
            text="Calibrate field (DLT2D)",
            command=self._on_calibrate,
            state="disabled",
        )
        self.calib_btn.pack(side=tk.LEFT, padx=4, pady=4)

        btns = ttk.Frame(frm)
        btns.pack(fill=tk.X, pady=(6, 0))
        self.cancel_btn = ttk.Button(btns, text="Request cancel", command=self._on_cancel)
        self.cancel_btn.pack(side=tk.LEFT)
        self.help_btn = ttk.Button(btns, text="Help", command=open_sam3_install_help_in_browser)
        self.help_btn.pack(side=tk.LEFT, padx=4)
        self.close_btn = ttk.Button(btns, text="Close", command=self._on_close, state="disabled")
        self.close_btn.pack(side=tk.RIGHT)

        self.transient(parent)  # ty: ignore[no-matching-overload]
        self.lift()

    def schedule_log(self, line: str) -> None:
        self.after(0, lambda s=line: self._append_log(s))

    def schedule_progress(self, done: int) -> None:
        self.after(0, lambda d=done: self._set_progress(d))

    def schedule_finish(self) -> None:
        self.after(0, self._finish)

    def _append_log(self, line: str) -> None:
        self.log_text.config(state="normal")
        self.log_text.insert("end", line.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_progress(self, done: int) -> None:
        self.progress["value"] = done
        self.status_var.set(f"{done} / {self._total} processed")

    def _on_cancel(self) -> None:
        self.cancelled = True
        self.cancel_btn.config(state="disabled")
        self._append_log("[GUI] Cancel requested — waiting for current video to end...")

    def _finish(self) -> None:
        def _done() -> None:
            self.cancel_btn.config(state="disabled")
            self.close_btn.config(state="normal")
            if self._output_base is not None and self._output_base.is_dir():
                self.postproc_btn.config(state="normal")
                self.calib_btn.config(state="normal")

        self.after(0, _done)

    def set_output_base(self, output_base: Path) -> None:
        self._output_base = output_base

    def _on_build_points(self) -> None:
        ob = self._output_base
        if ob is None or not ob.is_dir():
            messagebox.showwarning(
                "Post-processing", "Output directory not available.", parent=self
            )
            return
        self.postproc_btn.config(state="disabled")
        self._append_log("[postprocess] Building sam_points.csv (mode=all)...")

        def _worker() -> None:
            try:
                from vaila.sam_postprocess import extract_points_for_batch

                outs = extract_points_for_batch(ob, mode="all")
                msg = f"Wrote {len(outs)} sam_points.csv file(s) under {self._output_base}"
                self.after(
                    0,
                    lambda: (
                        self._append_log(f"[postprocess] {msg}"),
                        self.postproc_btn.config(state="normal"),
                        messagebox.showinfo("Post-processing done", msg, parent=self),
                    ),
                )
            except Exception as exc:
                err = str(exc)
                self.after(
                    0,
                    lambda: (
                        self._append_log(f"[postprocess] FAILED: {err}"),
                        self.postproc_btn.config(state="normal"),
                        messagebox.showerror("Post-processing failed", err, parent=self),
                    ),
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_calibrate(self) -> None:
        if self._output_base is None or not self._output_base.is_dir():
            messagebox.showwarning(
                "Calibrate field", "Output directory not available.", parent=self
            )
            return
        try:
            from vaila.soccerfield_calib import launch_calibrate_dialog

            launch_calibrate_dialog(self, self._output_base)
        except Exception as exc:
            messagebox.showerror("Calibrate field — error", str(exc), parent=self)

    def _on_close(self) -> None:
        if self.close_btn["state"] == "disabled":
            self._on_cancel()
            return
        with contextlib.suppress(tk.TclError):
            self.destroy()


def _run_sam_batch_in_thread(
    progress: SamBatchProgress,
    *,
    video_files: list[Path],
    output_base: Path,
    prompt: str,
    frame_idx: int,
    ckpt_opt: Path | None,
    save_ov: bool,
    save_png: bool,
    frame_fallback: bool,
    on_done: Callable[[int, int, list[str], Path], None],
) -> None:
    """Background worker for the GUI batch (started from a Thread)."""

    def log(line: str) -> None:
        progress.schedule_log(line)

    failed: list[str] = []
    succeeded = 0

    log(f"SAM 3 batch — {len(video_files)} video(s); output: {output_base}")
    for idx, video_file in enumerate(video_files, start=1):
        if progress.cancelled:
            log(f"[GUI] Cancelled before {video_file.name}; stopping.")
            failed.append(f"{video_file.name}: cancelled by user")
            break
        log(f"\n=== {idx}/{len(video_files)}: {video_file.name} ===")
        out_dir = output_base / video_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, err = _process_one_video_with_oom_retry(
            video_file,
            out_dir,
            text_prompt=prompt,
            frame_index=frame_idx,
            checkpoint=ckpt_opt,
            max_input_frames=None,
            save_overlay_mp4=save_ov,
            save_mask_png=save_png,
            frame_by_frame_fallback=frame_fallback,
            log=log,
        )
        if ok:
            succeeded += 1
            log(f"  Done: {out_dir}")
        else:
            failed.append(f"{video_file.name}: {err}")
            log(f"  ERROR on {video_file.name}: {err}")
        progress.schedule_progress(idx)
        _release_sam3_gpu_memory()

    progress.schedule_finish()
    on_done(succeeded, len(video_files), failed, output_base)


def _start_sam_batch_subprocess(
    *,
    progress: SamBatchProgress,
    input_path: Path,
    out_parent: Path,
    output_base: Path,
    prompt: str,
    frame_idx: int,
    ckpt_opt: Path | None,
    save_ov: bool,
    save_png: bool,
    frame_fallback: bool,
    on_done: Callable[[int, int, list[str], Path], None],
) -> None:
    """Run SAM3 batch in an isolated child process; reader threads stream stdout.

    Why a subprocess instead of ``threading.Thread``:
        Running the SAM3 video predictor (CUDA + torch.compile + Triton + OpenCV)
        in a worker thread of the same Python process as the Tk mainloop reliably
        triggers ``terminate called without an active exception`` (SIGABRT) — the
        Tcl event loop in MainThread interferes with the C++ destructors of CUDA /
        torch / Triton background workers spawned by the worker thread. Verified
        experimentally: same workload via the CLI (no Tk) completes cleanly, but
        the moment a Tcl mainloop runs in MainThread, the worker process aborts
        partway. Isolating the GPU work in a subprocess fully decouples it from
        the Tk interpreter and resolves the abort.
    """
    import re
    import shlex
    import subprocess
    import tempfile

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "-i",
        str(input_path.resolve()),
        "-o",
        str(out_parent.resolve()),
        "-t",
        prompt,
        "-f",
        str(frame_idx),
    ]
    if ckpt_opt is not None:
        cmd += ["-w", str(ckpt_opt.resolve())]
    if not save_ov:
        cmd.append("--no-overlay")
    if not save_png:
        cmd.append("--no-png")
    if frame_fallback:
        cmd.append("--frame-by-frame")

    progress.schedule_log(f"[GUI] launching subprocess: {shlex.join(cmd)}")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # tqdm bars use carriage returns and would clutter the GUI log; disable them.
    env["TQDM_DISABLE"] = "1"

    log_fd, log_path = tempfile.mkstemp(prefix="vaila_sam_batch_", suffix=".log", text=True)
    log_handle = os.fdopen(log_fd, "w", buffering=1)

    try:
        proc = subprocess.Popen(  # noqa: S603 — args list is built locally
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    except Exception as e:
        log_handle.close()
        progress.schedule_log(f"[GUI] FAILED to launch subprocess: {e!r}")
        progress.schedule_finish()
        on_done(0, 0, [f"subprocess launch failed: {e!r}"], output_base)
        return

    state: dict[str, Any] = {
        "succeeded": 0,
        "failed": [],
        "actual_output": output_base,
        "total_videos": 0,
        "current_idx": 0,
        "read_pos": 0,
        "buffer": "",
        "finished": False,
    }
    re_processing = re.compile(r"^Processing video\s+(\d+)\s*/\s*(\d+)\s*:")
    re_done_line = re.compile(r"^\s*Done:\s+(.+)$")
    re_error_line = re.compile(r"^\s*ERROR on\s+(.+):\s*(.*)$")
    re_output_dir = re.compile(r"^All done\.\s*Output:\s*(.+)$")

    def _process_line(line: str) -> None:
        progress._append_log(line)
        if (m := re_processing.match(line)) is not None:
            idx = int(m.group(1))
            total = int(m.group(2))
            state["current_idx"] = idx
            cur_total = state["total_videos"]
            state["total_videos"] = max(int(cur_total), total)
            progress._set_progress(idx - 1)
        elif (m := re_done_line.match(line)) is not None:
            state["succeeded"] = int(state["succeeded"]) + 1
            idx_done = int(state["current_idx"])
            if idx_done > 0:
                progress._set_progress(idx_done)
        elif (m := re_error_line.match(line)) is not None:
            fname = m.group(1).strip()
            err_text = m.group(2).strip()
            failed_list: list[str] = state["failed"]
            failed_list.append(f"{fname}: {err_text}")
        elif (m := re_output_dir.match(line)) is not None:
            actual_out_str = m.group(1).strip()
            state["actual_output"] = Path(actual_out_str)

    def _drain_log_file() -> None:
        try:
            with open(log_path, encoding="utf-8", errors="replace") as fh:
                fh.seek(int(state["read_pos"] or 0))
                chunk = fh.read()
                state["read_pos"] = fh.tell()
        except FileNotFoundError:
            chunk = ""
        if chunk:
            buf = str(state["buffer"] or "") + chunk
            *complete_lines, remainder = buf.split("\n")
            state["buffer"] = remainder
            for ln in complete_lines:
                if ln.strip():
                    with contextlib.suppress(tk.TclError):
                        _process_line(ln)

    def _poll() -> None:
        if progress.cancelled and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
        _drain_log_file()
        rc = proc.poll()
        if rc is None:
            with contextlib.suppress(tk.TclError):
                progress.after(150, _poll)
            return
        # Subprocess finished — final drain + finalize.
        _drain_log_file()
        remainder = str(state["buffer"] or "")
        if remainder.strip():
            with contextlib.suppress(tk.TclError):
                _process_line(remainder)
        state["buffer"] = ""
        if state["finished"]:
            return
        state["finished"] = True
        with contextlib.suppress(Exception):
            log_handle.close()
        with contextlib.suppress(tk.TclError):
            progress._append_log(f"[GUI] subprocess exited with code {rc}")
        if rc != 0:
            failed_list = state["failed"]
            if isinstance(failed_list, list) and not failed_list:
                failed_list.append(f"subprocess exit code {rc}")
        with contextlib.suppress(tk.TclError):
            progress._finish()
        try:
            actual_out = state["actual_output"]
            succ = int(state["succeeded"] or 0)
            total_default = len(_find_videos(input_path)) if input_path.is_dir() else 1
            total = int(state["total_videos"] or 0) or total_default
            failed_obj = state["failed"]
            failed_list = failed_obj if isinstance(failed_obj, list) else []
            out_path = actual_out if isinstance(actual_out, Path) else output_base
            on_done(succ, total, failed_list, out_path)
        except Exception as e:
            with contextlib.suppress(tk.TclError):
                progress._append_log(f"[GUI] on_done callback error: {e!r}")

    progress.after(50, _poll)


def run_sam_video(existing_root: tk.Tk | None = None) -> None:
    """GUI entry: configure and run SAM3 on a directory of videos (batch) or a single file."""
    root = existing_root
    owns_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True
    try:
        # Windows + Linux: a fully withdrawn root can prevent the Toplevel from mapping
        # reliably under some compositors (invisible / wrong workspace).
        if platform.system() in ("Windows", "Linux"):
            root.deiconify()
            root.geometry("1x1+100+100")
            root.update_idletasks()
    except Exception:
        pass

    dlg = SamVideoDialog(root)
    root.wait_window(dlg)

    if dlg.result is None:
        if owns_root:
            root.destroy()
        return

    (
        input_path,
        out_parent,
        ckpt_opt,
        prompt,
        frame_idx,
        save_ov,
        save_png,
        frame_fallback,
        dry_run,
    ) = dlg.result

    video_files = _find_videos(input_path) if input_path.is_dir() else [input_path]
    if not video_files:
        messagebox.showinfo(
            "SAM 3 — nothing to do", "No video files found at the input path.", parent=root
        )
        if owns_root:
            root.destroy()
        return

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = out_parent / f"processed_sam_{ts}"
    output_base.mkdir(parents=True, exist_ok=True)

    if dry_run:
        report = _sam3_dry_run_report(
            input_path,
            output_base,
            ckpt_opt=ckpt_opt,
            text_prompt=prompt,
            frame_index=frame_idx,
            max_input_frames=None,
            save_overlay_mp4=save_ov,
            save_mask_png=save_png,
            frame_by_frame_fallback=frame_fallback,
            per_video_output=True,
        )
        for line in report:
            print(line)
        dry_report = _write_sam3_dry_run_report(report, output_base)
        if dry_report is not None:
            report.append(f"Report written to: {dry_report}")
        summary = "\n".join(report[: min(18, len(report))])
        if len(report) > 18:
            summary += "\n… veja o arquivo SAM3_DRY_RUN.txt para o plano completo."
        root.after(
            0,
            lambda: messagebox.showinfo("SAM 3 — dry-run", summary, parent=root),
        )
        if owns_root:
            root.destroy()
        return

    if importlib.util.find_spec("sam3") is None:
        _print_sam3_install_instructions()
        open_sam3_install_help_in_browser()
        if owns_root:
            root.destroy()
        return

    if not _sam3_guard_cuda_gui(root):
        if owns_root:
            root.destroy()
        return

    progress = SamBatchProgress(root, total=len(video_files), output_base=output_base)

    def _on_done(succeeded: int, total: int, failed: list[str], out_base: Path) -> None:
        summary = f"Processed {succeeded}/{total} video(s).\nOutput: {out_base}"
        if failed:
            summary += f"\n\nFailed ({len(failed)}):\n" + "\n".join(failed[:20])
            if len(failed) > 20:
                summary += f"\n…and {len(failed) - 20} more (see FAILED_sam.txt in each folder)."
            root.after(
                0,
                lambda: messagebox.showwarning(
                    "SAM 3 — finished with errors", summary, parent=root
                ),
            )
        else:
            root.after(0, lambda: messagebox.showinfo("SAM 3 — done", summary, parent=root))

    _start_sam_batch_subprocess(
        progress=progress,
        input_path=input_path,
        out_parent=out_parent,
        output_base=output_base,
        prompt=prompt,
        frame_idx=frame_idx,
        ckpt_opt=ckpt_opt,
        save_ov=save_ov,
        save_png=save_png,
        frame_fallback=frame_fallback,
        on_done=_on_done,
    )

    root.wait_window(progress)

    if owns_root:
        root.destroy()


def main() -> None:
    # Must be set BEFORE the first ``import torch`` for the allocator to honor it
    _alloc_val = "expandable_segments:True"
    for _key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        _cur = os.environ.get(_key, "")
        if "expandable_segments" not in _cur:
            os.environ[_key] = f"{_cur},{_alloc_val}" if _cur else _alloc_val

    if len(sys.argv) > 1 and sys.argv[1] == "fifa":
        from vaila.fifa_skeletal_pipeline import main_fifa_cli

        main_fifa_cli(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="SAM 3 video segmentation (vailá)")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input video file OR directory containing videos (batch)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output base directory")
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="person",
        help=(
            "Open-vocabulary text prompt (SAM 3). Examples: 'person', 'player', "
            "'goalkeeper', 'referee', 'ball', 'soccer ball', 'basketball', "
            "'crowd', 'car', 'dog'. Any free-text description works."
        ),
    )
    parser.add_argument("-f", "--frame", type=int, default=0, help="Frame index for prompt")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay MP4")
    parser.add_argument("--no-png", action="store_true", help="Skip mask PNGs")
    parser.add_argument(
        "--frame-by-frame",
        action="store_true",
        help="CUDA only: process each frame individually to reduce VRAM (not CPU inference). "
        "Loses temporal tracking.",
    )
    parser.add_argument(
        "-w",
        "--weights",
        "--checkpoint",
        dest="checkpoint",
        type=Path,
        default=None,
        help="Path to SAM 3 video weights (sam3.pt or sam3.1_multiplex.pt), or a "
        "folder containing them. Skips Hub download. "
        "Default: env SAM3_CHECKPOINT / VAILA_SAM3_CHECKPOINT, else "
        "vaila/models/sam3/{sam3.pt,sam3.1_multiplex.pt} "
        "(legacy: repo sam3_weights/sam3.pt).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        metavar="N",
        help="Max frames passed to SAM3 on GPU (VRAM). Overrides SAM3_MAX_FRAMES; 0 = full clip.",
    )
    parser.add_argument(
        "--max-input-long-edge",
        type=int,
        default=None,
        metavar="PX",
        help="Max long edge in pixels (width/height) for each frame fed to SAM3. "
        "0 = native resolution (no downscale). Default 1920 or SAM3_MAX_INPUT_LONG_EDGE. "
        "4K+ broadcast needs this on many GPUs; try 1280 or 960 if OOM with --max-frames 1.",
    )
    parser.add_argument(
        "--dry-run",
        "--smoke",
        dest="dry_run",
        action="store_true",
        help="Print effective settings, effective caps, retries, and detected checkpoint; "
        "do not run SAM3.",
    )
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download facebook/sam3 (config.json + sam3.pt) into vaila/models/sam3/. "
        "Needs HF access to the repo; use HF_TOKEN or: uv run hf auth login",
    )
    parser.add_argument(
        "--open-help",
        action="store_true",
        help="Open SAM 3 setup instructions (HTML) in the default browser and exit.",
    )
    parser.add_argument(
        "--postprocess-points",
        choices=["none", "foot", "center", "mask", "all"],
        default="none",
        help="After the batch finishes, build vailá-format pixel CSVs (sam_points.csv) "
        "per video subdirectory. 'foot' = bottom-center of bbox (best for soccer-field "
        "homography rec2d); 'center' = bbox center; 'mask' = real centroid of the mask "
        "PNG; 'all' = canonical foot pair plus extra cx/cy/mx/my columns. Default: none.",
    )
    parser.add_argument(
        "--no-isolate-batch",
        action="store_true",
        help="Disable subprocess-per-video isolation in batch mode. "
        "Default: each video runs in its own subprocess so a CUDA OOM in one video "
        "cannot leak GPU state into the next (SAM3's start_session leaves ~13 GiB "
        "of orphan tensors on OOM that no in-process gc can free).",
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,  # internal: used by subprocess-per-video isolation
    )
    args = parser.parse_args()

    if args.open_help:
        open_sam3_install_help_in_browser()
        return

    if args.download_weights:
        out_ckpt = download_sam3_weights_to_vaila_models()
        print(f"SAM3 weights ready: {out_ckpt}")
        return

    # Internal isolated single-video mode: skip CUDA guard / processed_sam_TS wrapper
    # and write outputs directly to the parent-supplied dir.  Used by the subprocess-
    # per-video isolation in batch mode so a CUDA OOM in one video cannot leak ~13 GiB
    # of orphan SAM3 tensors into the next video (proven by debug-mode runtime logs:
    # H4+H7 — CUDA state corruption at start_session is irrecoverable in-process).
    if args.input and args.video_output_dir is not None:
        single = args.input.resolve()
        if not single.is_file():
            print(f"--video-output-dir requires --input to be a single file: {single}")
            raise SystemExit(2)
        out_dir = args.video_output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            report = _sam3_dry_run_report(
                single,
                out_dir,
                ckpt_opt=args.checkpoint,
                text_prompt=args.text,
                frame_index=args.frame,
                max_input_frames=args.max_frames,
                save_overlay_mp4=not args.no_overlay,
                save_mask_png=not args.no_png,
                frame_by_frame_fallback=args.frame_by_frame,
                per_video_output=False,
            )
            for line in report:
                print(line)
            dry_report = _write_sam3_dry_run_report(report, out_dir)
            if dry_report is not None:
                print(f"Dry-run report written to: {dry_report}")
            return
        if importlib.util.find_spec("sam3") is None:
            _print_sam3_install_instructions()
            open_sam3_install_help_in_browser()
            raise SystemExit(1)
        _sam3_guard_cuda_cli()
        ok, err = _process_one_video_with_oom_retry(
            single,
            out_dir,
            text_prompt=args.text,
            frame_index=args.frame,
            checkpoint=args.checkpoint,
            max_input_frames=args.max_frames,
            max_input_long_edge=args.max_input_long_edge,
            save_overlay_mp4=not args.no_overlay,
            save_mask_png=not args.no_png,
            frame_by_frame_fallback=args.frame_by_frame,
        )
        if ok:
            print(f"  Done: {out_dir}")
            raise SystemExit(0)
        print(f"  ERROR on {single.name}: {err}")
        raise SystemExit(3)

    if args.input and args.output:
        inp = args.input.resolve()
        if inp.is_dir():
            video_files = _find_videos(inp)
            if not video_files:
                print(f"No video files found in {inp}")
                return
        elif inp.is_file():
            video_files = [inp]
        else:
            print(f"Input path does not exist: {inp}")
            return

        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = args.output / f"processed_sam_{ts}"
        output_base.mkdir(parents=True, exist_ok=True)

        print(f"\nSAM 3 batch — {len(video_files)} video(s) to process")
        for idx, vf in enumerate(video_files, 1):
            print(f"  {idx}. {vf.name}")

        if args.dry_run:
            report = _sam3_dry_run_report(
                inp,
                output_base,
                ckpt_opt=args.checkpoint,
                text_prompt=args.text,
                frame_index=args.frame,
                max_input_frames=args.max_frames,
                save_overlay_mp4=not args.no_overlay,
                save_mask_png=not args.no_png,
                frame_by_frame_fallback=args.frame_by_frame,
                per_video_output=True,
            )
            for line in report:
                print(line)
            dry_report = _write_sam3_dry_run_report(report, output_base)
            if dry_report is not None:
                print(f"Dry-run report written to: {dry_report}")
            return

        if importlib.util.find_spec("sam3") is None:
            _print_sam3_install_instructions()
            open_sam3_install_help_in_browser()
            raise SystemExit(1)
        _sam3_guard_cuda_cli()

        # Subprocess-per-video isolation is the ONLY reliable fix for the SAM3
        # CUDA OOM cascade.  Runtime evidence (debug session 42b4a5):
        #   - Without OOM: post-cleanup alloc=0.009 GiB → batch works.
        #   - With OOM: ~13 GiB orphan C++ tensors persist; gc.collect / empty_cache
        #     cannot reach them → next video starts with leaked state → cascade.
        # Killing the Python process is the only way to release SAM3's internal
        # C++ workspace pools after a failed start_session.
        use_isolation = (not args.no_isolate_batch) and len(video_files) > 1
        failed_cli: list[str] = []

        if use_isolation:
            import subprocess as _sp

            print("[batch] subprocess-per-video isolation: ENABLED (each video in fresh process)")
            for idx, video_file in enumerate(video_files, 1):
                print(f"\n{'=' * 60}")
                print(f"Processing video {idx}/{len(video_files)}: {video_file.name} (isolated)")
                print(f"{'=' * 60}")
                out_dir = output_base / video_file.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--input",
                    str(video_file),
                    "--video-output-dir",
                    str(out_dir),
                    "--text",
                    args.text,
                    "--frame",
                    str(args.frame),
                ]
                if args.max_frames is not None:
                    cmd += ["--max-frames", str(args.max_frames)]
                if args.max_input_long_edge is not None:
                    cmd += ["--max-input-long-edge", str(args.max_input_long_edge)]
                if args.checkpoint is not None:
                    cmd += ["--checkpoint", str(args.checkpoint)]
                if args.no_overlay:
                    cmd.append("--no-overlay")
                if args.no_png:
                    cmd.append("--no-png")
                if args.frame_by_frame:
                    cmd.append("--frame-by-frame")
                env = os.environ.copy()
                env.setdefault("TQDM_DISABLE", "1")
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                try:
                    rc = _sp.call(cmd, env=env)
                except KeyboardInterrupt:
                    print(f"  INTERRUPTED on {video_file.name}")
                    failed_cli.append(f"{video_file.name}: interrupted")
                    raise
                if rc == 0:
                    print(f"  Done: {out_dir}")
                else:
                    err_msg = f"subprocess exit={rc}"
                    print(f"  ERROR on {video_file.name}: {err_msg}")
                    failed_cli.append(f"{video_file.name}: {err_msg}")
        else:
            for idx, video_file in enumerate(video_files, 1):
                print(f"\n{'=' * 60}")
                print(f"Processing video {idx}/{len(video_files)}: {video_file.name}")
                print(f"{'=' * 60}")
                out_dir = output_base / video_file.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                ok, err = _process_one_video_with_oom_retry(
                    video_file,
                    out_dir,
                    text_prompt=args.text,
                    frame_index=args.frame,
                    checkpoint=args.checkpoint,
                    max_input_frames=args.max_frames,
                    max_input_long_edge=args.max_input_long_edge,
                    save_overlay_mp4=not args.no_overlay,
                    save_mask_png=not args.no_png,
                    frame_by_frame_fallback=args.frame_by_frame,
                )
                if ok:
                    print(f"  Done: {out_dir}")
                else:
                    print(f"  ERROR on {video_file.name}: {err}")
                    failed_cli.append(f"{video_file.name}: {err}")
                _release_sam3_gpu_memory()
        print(f"\nAll done. Output: {output_base}")
        if failed_cli:
            print(f"Failed ({len(failed_cli)}/{len(video_files)}):")
            for line in failed_cli:
                print(f"  - {line}")

        if args.postprocess_points != "none":
            ob = output_base
            try:
                from vaila.sam_postprocess import extract_points_for_batch

                print(f"\n[postprocess] mode={args.postprocess_points}")
                outs = extract_points_for_batch(ob, mode=args.postprocess_points)
                print(f"[postprocess] wrote {len(outs)} sam_points.csv file(s).")
            except Exception as exc:
                print(f"[postprocess] FAILED: {exc}")
        return

    run_sam_video()


if __name__ == "__main__":
    main()
