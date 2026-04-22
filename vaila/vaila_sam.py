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
    try:
        import torch

        if not torch.cuda.is_available():
            return 32
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        total_gib = props.total_memory / (1024**3)
        free_bytes, _ = torch.cuda.mem_get_info(dev)
        free_gib = free_bytes / (1024**3)
    except Exception:
        return 32

    model_gib = 6.3
    processing_gib = 0.5
    headroom_gib = 0.5
    per_frame_mib = 3.6
    budget_gib = max(0.0, free_gib - model_gib - processing_gib - headroom_gib)
    safe_frames = int(budget_gib * 1024 / per_frame_mib)
    # Even with large VRAM, propagation memory grows with timeline length.
    # Cap the auto value conservatively to avoid late-stage OOM on long clips.
    safe_frames = max(16, min(safe_frames, 128))
    print(
        f"[SAM3 VRAM] GPU {total_gib:.1f} GiB total, {free_gib:.1f} GiB free → "
        f"auto max_frames={safe_frames} "
        f"(model ~6.3 GiB, budget ~{budget_gib:.1f} GiB for frames)"
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
    raw = os.environ.get("SAM3_MAX_FRAMES", "").strip().lower()
    if raw in ("0", "none", "unlimited", "off"):
        return 0
    if raw:
        return max(0, int(raw))
    return _auto_max_frames_for_vram()


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


def _sam3_cuda_oom_help(*, max_input_frames: int, session_frames: int) -> str:
    return (
        "CUDA out of memory while loading the video into SAM3. "
        "The library puts the whole session on GPU at once (~few GB per hundred frames at 1008²).\n\n"
        f"This run used session_frames={session_frames} (max_input_frames_cap={max_input_frames}; "
        "0 means no subsampling).\n\n"
        "Fix: unset SAM3_MAX_FRAMES (auto-sizes to GPU VRAM), or set SAM3_MAX_FRAMES=64, or pass "
        "--max-frames 64. Lower values use less VRAM. Keep weights under vaila/models/sam3/."
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
    mf = _read_max_input_frames(max_input_frames)
    session_path, temp_clip, n_sess = _maybe_subsample_video_for_vram(video_path, output_dir, mf)
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
        from typing import Any

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
                        _sam3_cuda_oom_help(max_input_frames=mf, session_frames=n_sess)
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
                    tip_n = max(8, mf // 2) if mf > 0 else 64
                    msg = (
                        _sam3_cuda_oom_help(max_input_frames=mf, session_frames=n_sess)
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
            f"subsampled_to_disk={temp_clip is not None} max_input_frames_cap={mf} session_frames={n_sess}\n"
            f"checkpoint={ckpt_note}\n"
            f"prompt={text_prompt!r}\n"
            f"prompt_frame_requested={frame_index} prompt_frame_used={fi_run}\n"
            f"frames_with_outputs={len(outputs_by_frame)} / {nframes}\n",
            encoding="utf-8",
        )
    finally:
        if temp_clip is not None and Path(temp_clip).is_file():
            with contextlib.suppress(OSError):
                Path(temp_clip).unlink(missing_ok=True)
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
    """Caps to try on CUDA OOM (high → low). Includes 24…8 when initial cap is 32 (bugfix)."""
    attempts: list[int | None] = [max_input_frames]
    if max_input_frames is None or max_input_frames > 64:
        attempts.append(64)
    if max_input_frames is None or max_input_frames > 32:
        attempts.append(32)
    positives = [x for x in attempts if isinstance(x, int) and x > 0]
    floor = min(positives) if positives else 2**30
    for step in (24, 16, 12, 8):
        if step < floor and step not in attempts:
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
    save_overlay_mp4: bool,
    save_mask_png: bool,
    frame_by_frame_fallback: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """Run one video; on CUDA OOM retry with descending frame caps (e.g. auto→64→32→24→8).

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
    for attempt_idx, mf_try in enumerate(attempts, start=1):
        _release_sam3_gpu_memory()
        try:
            run_sam3_on_video(
                video_file,
                output_dir,
                text_prompt,
                frame_index=frame_index,
                checkpoint=checkpoint,
                max_input_frames=mf_try,
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
                _write_failure_marker(output_dir, video_file, "CUDA OOM: " + last_err)
                return False, last_err
            next_mf = attempts[attempt_idx]
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
                f"  [SAM3] CUDA OOM at max_frames={mf_try!r}; retrying with max_frames={next_mf}..."
            )
    _write_failure_marker(output_dir, video_file, last_err or "unknown failure")
    return False, last_err or "unknown failure"


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
        self.result: tuple[Path, Path, Path | None, str, int, bool, bool, bool] | None = None

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

        btns = ttk.Frame(frm)
        btns.grid(row=9, column=0, columnspan=3, pady=12)
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
        self.cancel_btn.config(state="disabled")
        self.close_btn.config(state="normal")
        if self._output_base is not None and self._output_base.is_dir():
            self.postproc_btn.config(state="normal")
            self.calib_btn.config(state="normal")

    def set_output_base(self, output_base: Path) -> None:
        self._output_base = output_base

    def _on_build_points(self) -> None:
        if self._output_base is None or not self._output_base.is_dir():
            messagebox.showwarning(
                "Post-processing", "Output directory not available.", parent=self
            )
            return
        self.postproc_btn.config(state="disabled")
        self._append_log("[postprocess] Building sam_points.csv (mode=all)...")

        def _worker() -> None:
            try:
                from vaila.sam_postprocess import extract_points_for_batch

                outs = extract_points_for_batch(self._output_base, mode="all")
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

    state: dict[str, object] = {
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
            state["total_videos"] = max(int(state["total_videos"] or 0), total)
            progress._set_progress(idx - 1)
        elif (m := re_done_line.match(line)) is not None:
            state["succeeded"] = int(state["succeeded"] or 0) + 1
            idx = int(state["current_idx"] or 0)
            if idx > 0:
                progress._set_progress(idx)
        elif (m := re_error_line.match(line)) is not None:
            fname = m.group(1).strip()
            err_text = m.group(2).strip()
            failed_list = state["failed"]
            if isinstance(failed_list, list):
                failed_list.append(f"{fname}: {err_text}")
        elif (m := re_output_dir.match(line)) is not None:
            state["actual_output"] = Path(m.group(1).strip())

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
    if importlib.util.find_spec("sam3") is None:
        _print_sam3_install_instructions()
        open_sam3_install_help_in_browser()
        return

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

    if not _sam3_guard_cuda_gui(root):
        if owns_root:
            root.destroy()
        return

    input_path, out_parent, ckpt_opt, prompt, frame_idx, save_ov, save_png, frame_fallback = (
        dlg.result
    )

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
        if importlib.util.find_spec("sam3") is None:
            _print_sam3_install_instructions()
            open_sam3_install_help_in_browser()
            raise SystemExit(1)
        _sam3_guard_cuda_cli()
        single = args.input.resolve()
        if not single.is_file():
            print(f"--video-output-dir requires --input to be a single file: {single}")
            raise SystemExit(2)
        out_dir = args.video_output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ok, err = _process_one_video_with_oom_retry(
            single,
            out_dir,
            text_prompt=args.text,
            frame_index=args.frame,
            checkpoint=args.checkpoint,
            max_input_frames=args.max_frames,
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
        if importlib.util.find_spec("sam3") is None:
            _print_sam3_install_instructions()
            open_sam3_install_help_in_browser()
            raise SystemExit(1)
        _sam3_guard_cuda_cli()
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
            try:
                from vaila.sam_postprocess import extract_points_for_batch

                print(f"\n[postprocess] mode={args.postprocess_points}")
                outs = extract_points_for_batch(output_base, mode=args.postprocess_points)
                print(f"[postprocess] wrote {len(outs)} sam_points.csv file(s).")
            except Exception as exc:
                print(f"[postprocess] FAILED: {exc}")
        return

    run_sam_video()


if __name__ == "__main__":
    main()
