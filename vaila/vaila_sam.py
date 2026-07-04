"""
Project: vailá
Script: vaila_sam.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 16 April 2026
Update Date: 04 July 2026
Version: 0.3.69

Description:
    Video segmentation with Meta SAM 3 (text prompts, Hugging Face checkpoints).
    Install: ``uv sync --extra sam``; **inference requires NVIDIA CUDA** (see AGENTS.md / ``bin/use_pyproject_*`` for CPU laptop vs CUDA workstation).

Auth (gated ``facebook/sam3``):
    - Hugging Face: accept the model, then ``uv run hf auth login``
    - Or: ``uv run vaila/vaila_sam.py --download-weights``
    - Or: ``uv run hf download facebook/sam3 sam3.pt --local-dir DIR`` / ``SAM3_CHECKPOINT=/path/to/sam3.pt``
    - Defaults: ``vaila/models/sam3/sam3.pt`` or nested ``.../sam3_weights/sam3.pt``

VRAM / resolution:
    - Long clips: ``--max-frames`` / ``SAM3_MAX_FRAMES`` / GUI field; **empty = auto** from free VRAM (cap scales up on 16–24 GiB GPUs). **0 = full clip** (every frame to SAM — best overlay sync; needs enough VRAM). Heavy temporal subsample reuses masks across many original frames → motion looks “late”; fix is more frames or **0**.
    - Large frames (e.g. 4K): ``--max-input-long-edge`` / ``SAM3_MAX_INPUT_LONG_EDGE`` (defaults depend on GPU size).
    - Heavy OOM: ``--frame-by-frame`` (CUDA VRAM tradeoff only; not a CPU mode).

Host RAM (not VRAM) — **subprocess exit=-9 / SIGKILL**:
    - Long broadcast clips (e.g. 16k+ frames at 1080p) decoded into the temp
      ``_sam3_subsample_input.mp4`` and then loaded by SAM3 can easily peak
      well over **30 GiB of system RAM**. On smaller hosts the Linux OOM
      killer terminates the SAM subprocess with ``SIGKILL`` (exit code
      ``-9``). This is **not a VRAM problem** — lowering ``--max-frames`` to
      e.g. ``256`` / ``128`` fixes it. See ``--print-examples`` and
      ``vaila/help/vaila_sam.md`` § *Common errors*.

Usage (quick):
    uv run vaila/vaila_sam.py                      # Tkinter: pick video, prompt, output folder
    uv run vaila/vaila_sam.py --open-help          # SAM 3 setup in browser
    uv run vaila/vaila_sam.py --download-weights   # fetch weights (needs HF auth)

**Full match / broadcast clip (CLI)** — segmentation + ``*_sam_overlay.mp4`` + CSV/JSON exports.
Tweak ``--max-input-long-edge`` / ``--max-frames`` if CUDA OOMs:

    uv run vaila/vaila_sam.py \\
        -i /path/to/match.mp4 \\
        -o /path/to/processed_sam_YYYYMMDD_HHMMSS \\
        -t player \\
        --max-input-long-edge 1920

Shorter temporal SAM input (overlay still follows every frame of ``-i``):

    uv run vaila/vaila_sam.py -i /path/to/match.mp4 -o /path/to/out -t player --max-frames 400

More detail: ``vaila/help/vaila_sam.md`` / ``vaila/help/vaila_sam.html``.

FIFA Skeletal Tracking Light (optional ``--extra fifa``): subcommand ``fifa`` delegates to
``vaila.fifa_skeletal_pipeline`` (prepare, boxes, preprocess, baseline, dlt-export, pack). Example:

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
import csv
import datetime as dt
import importlib.util
import json
import os
import platform
import shutil
import sys
import time
import tkinter as tk
import webbrowser
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

try:
    from .geometric_reid import (
        GeometricFrameLinker,
        GeometricLinkerConfig,
        assignment_min_cost,
        bbox_iou_xywh,
        mask_iou_u8,
        write_reid_links_csv,
    )
except ImportError:
    from geometric_reid import (  # ty: ignore[unresolved-import]
        GeometricFrameLinker,
        GeometricLinkerConfig,
        assignment_min_cost,
        bbox_iou_xywh,
        mask_iou_u8,
        write_reid_links_csv,
    )

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

    _Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu = _forward_with_fp32_tracker_fpn
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
        "SAM 3 is not installed.\n\n"
        "Install the optional stack, then restart vailá:\n"
        "  uv sync --extra sam\n\n"
        "NVIDIA CUDA workstation:\n"
        "  bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam --yes\n"
        "  # or, after CUDA template is active:\n"
        "  uv sync --extra gpu --extra sam\n\n"
        "Windows NVIDIA CUDA workstation:\n"
        "  pwsh bin/setup_pyproject.ps1 -Target win-cuda -Extras gpu,sam -Yes\n\n"
        "After install, accept the gated Hugging Face model and authenticate:\n"
        "  uv run hf auth login\n"
        "  uv run vaila/vaila_sam.py --download-weights\n\n"
        "CLI help / examples:\n"
        "  uv run vaila/vaila_sam.py --open-help\n"
        "  uv run vaila/vaila_sam.py --print-examples\n\n"
        "Runtime note: SAM 3 video requires NVIDIA CUDA. CPU and macOS Metal/MPS are "
        "not supported for this integration.\n"
        "A GUI dialog/browser help may also open, but this terminal output is the "
        "copy/paste install path.\n"
        "See also: AGENTS.md - Hybrid CPU vs NVIDIA workstation.\n"
    )
    print("\n" + "=" * 72, file=sys.stderr)
    print(msg, file=sys.stderr)
    print("=" * 72 + "\n", file=sys.stderr)


SAM3_CLI_EXAMPLES = """\
SAM 3 — copy/paste CLI recipes
==============================

# 0. Open help / setup page in your default browser
uv run vaila/vaila_sam.py --open-help

# 1. Print these examples again (no GPU work)
uv run vaila/vaila_sam.py --print-examples

# 2. Download gated facebook/sam3 weights into vaila/models/sam3/
#    (needs 'uv run hf auth login' or HF_TOKEN with access to the model)
uv run vaila/vaila_sam.py --download-weights

# 3. Smoke / dry-run on a single video (prints effective settings, OOM ladder)
uv run vaila/vaila_sam.py \\
  -i tests/SAM/test1000.mp4 \\
  -o tests/SAM/ \\
  -t person \\
  --dry-run

# 4. Single video — short clip on a 24 GiB GPU (auto everything)
uv run vaila/vaila_sam.py \\
  -i path/to/video.mp4 \\
  -o path/to/output_parent/ \\
  -t player

# 5. LONG broadcast clip (~15k+ frames). The default auto cap can demand more
#    HOST RAM than the OS has, which results in subprocess exit=-9 (SIGKILL by
#    the Linux OOM killer). Force a conservative cap:
uv run vaila/vaila_sam.py \\
  -i path/to/long_match.mp4 \\
  -o path/to/output_parent/ \\
  -t player \\
  --max-frames 128 \\
  --max-input-long-edge 1280 \\
  --postprocess-points all

# 6. Batch over a directory of clips (subprocess-per-video isolation is ON by
#    default; each clip starts with a clean CUDA context).
uv run vaila/vaila_sam.py \\
  -i path/to/clips_dir/ \\
  -o path/to/output_parent/ \\
  -t player \\
  --max-frames 256 \\
  --postprocess-points foot

# 7. Low-VRAM GPU (e.g. RTX 5050 8 GiB) — disable temporal tracking but never OOM
uv run vaila/vaila_sam.py \\
  -i path/to/video.mp4 \\
  -o path/to/output_parent/ \\
  -t person \\
  --frame-by-frame --no-png --no-overlay

# 8. Use a specific checkpoint or SAM 3.1 multiplex weights
uv run vaila/vaila_sam.py \\
  -i path/to/video.mp4 \\
  -o path/to/output_parent/ \\
  -t player \\
  -w vaila/models/sam3/sam3.1_multiplex.pt

# 9. Preflight scan only — write SAM3_PREFLIGHT.csv (resolution/fps/duration)
uv run vaila/vaila_sam.py --input path/to/clips_dir/ --preflight \\
  --output path/to/output_parent/

# 10. FIFA Skeletal Tracking Light subcommand (needs --extra fifa, CUDA, SAM 3D Body weights)
uv run vaila/vaila_sam.py fifa --help
uv run vaila/vaila_sam.py fifa bootstrap  --videos-dir DIR --data-root data/
uv run vaila/vaila_sam.py fifa prepare    --video-source DIR --data-root data/
uv run vaila/vaila_sam.py fifa boxes      --data-root data/ --sequences data/sequences_val.txt
uv run vaila/vaila_sam.py fifa preprocess --data-root data/ --sequences data/sequences_val.txt
uv run vaila/vaila_sam.py fifa baseline   --data-root data/ --sequences data/sequences_val.txt \\
  --output outputs/submission_val.npz --export-camera
uv run vaila/vaila_sam.py fifa dlt-export --cameras-dir data/cameras --output-dir outputs/dlt
uv run vaila/vaila_sam.py fifa pack       --submission-full outputs/submission_val.npz \\
  --data-root data/ --output-dir outputs/ --split val

Tips
----
* GUI (no args)              : uv run vaila/vaila_sam.py
* Common error 'subprocess exit=-9' = host-RAM OOM killer; lower --max-frames.
* Common error 'CUDA out of memory' = drop --max-input-long-edge to 1280/960
  or add --frame-by-frame as a last resort.
* Full reference            : vaila/help/vaila_sam.md  (or --open-help).
"""


def _print_sam3_cli_examples() -> None:
    """Dump the copy-paste cheat sheet to stdout (used by ``--print-examples``)."""
    print(SAM3_CLI_EXAMPLES, flush=True)


# Exit codes for the per-video isolated subprocess (see batch CLI loop).
# When the per-video subprocess exhausts its OOM retry ladder while running
# under ``--no-chunked-fallback``, it exits with this code so the *coordinator*
# (the outer CLI process, which never loaded SAM3 and therefore has a clean
# CUDA context) can run the chunked fallback itself.  Keeping the chunked
# coordinator separate from the OOM victim is the only way to free the ~13 GiB
# of orphan SAM3 C++ workspace tensors that no in-process gc can reach.
EXIT_NEEDS_CHUNKING = 7


# Common POSIX signals returned as negative exit codes by ``subprocess.call`` /
# ``Popen.poll`` on Unix; the GUI / CLI batch monitors translate them to human
# advice (host-RAM OOM, segfault, manual kill, …) instead of just printing the
# raw ``subprocess exit=-9``.
_SUBPROCESS_SIGNAL_HINTS: dict[int, tuple[str, str]] = {
    -9: (
        "SIGKILL",
        "Likely the Linux OOM killer (SYSTEM RAM, not VRAM). "
        "Long broadcast clips load the temporal subsample into host RAM and "
        "easily peak above 20-30 GiB. Lower --max-frames (try 256, 128 or 64) "
        "and/or --max-input-long-edge 1280, or close other heavy apps. "
        "Run `dmesg | tail` or `journalctl -k -n 50` to confirm an oom-kill event.",
    ),
    -11: (
        "SIGSEGV",
        "Native segmentation fault in CUDA / Torch / OpenCV. Verify the GPU "
        "driver, try `nvidia-smi`, and rerun with --dry-run / --preflight to "
        "isolate the failing video.",
    ),
    -15: (
        "SIGTERM",
        "External termination (you, the OS, or a parent process killed the run).",
    ),
    -2: ("SIGINT", "Cancelled by user (Ctrl+C)."),
    -6: (
        "SIGABRT",
        "C++ abort — usually a torch / Triton / CUDA destructor running on the "
        "wrong thread. If reproducible, attach gdb to the child or run from "
        "the CLI to get the traceback.",
    ),
    -7: (
        "SIGBUS",
        "Bus error — typically corrupted video file or a broken shared mmap. "
        "Try re-encoding the input video with ffmpeg.",
    ),
}


def _format_subprocess_exit_diagnosis(rc: int) -> str:
    """Human-readable explanation for a subprocess return code.

    Negative codes on Unix correspond to terminating signals (``rc == -signum``);
    positive codes are application exit codes (we recognise ``EXIT_NEEDS_CHUNKING``).
    """
    if rc == 0:
        return "subprocess exited cleanly (exit=0)."
    if rc == EXIT_NEEDS_CHUNKING:
        return (
            f"subprocess exit={rc} (EXIT_NEEDS_CHUNKING) — per-video child "
            "exhausted its OOM retry ladder; coordinator will run the chunked "
            "divide-and-conquer fallback from a clean GPU."
        )
    if rc < 0:
        name, hint = _SUBPROCESS_SIGNAL_HINTS.get(
            rc,
            (f"SIG{-rc}", "Process killed by signal; check `dmesg` / system logs."),
        )
        return f"subprocess killed by {name} (exit={rc}). {hint}"
    return f"subprocess exit={rc} — non-zero exit, see logs above for the SAM3 traceback."


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


def _sam3_auto_max_frames_upper_bound(total_gib: float) -> int:
    """Max value for auto ``max_frames`` before budget math clamps lower.

    A hard cap of 128 was safe on 8 GiB laptops but forced 2.7k-frame clips down to
    ~128 SAM keyframes on 24 GiB cards — overlays looked permanently out of sync.
    """
    if total_gib >= 24.0:
        return 8192
    if total_gib >= 18.0:
        return 4096
    if total_gib >= 12.0:
        return 1024
    if total_gib >= 10.0:
        return 512
    return 128


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
    upper = _sam3_auto_max_frames_upper_bound(total_gib)
    safe_frames = max(16, min(safe_frames, upper))
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


def _host_ram_profile() -> dict[str, float] | None:
    """Best-effort host (system) RAM probe; falls back to ``/proc/meminfo``."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_gib": float(vm.total) / (1024**3),
            "available_gib": float(vm.available) / (1024**3),
            "used_pct": float(vm.percent),
        }
    except Exception:
        pass
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                value = rest.strip().split()
                if len(value) >= 1 and value[0].isdigit():
                    info[key.strip()] = int(value[0])  # KiB
        total = info.get("MemTotal", 0) / (1024**2)
        avail = info.get("MemAvailable", info.get("MemFree", 0)) / (1024**2)
        if total <= 0:
            return None
        used_pct = 100.0 * (1.0 - (avail / total))
        return {
            "total_gib": float(total),
            "available_gib": float(avail),
            "used_pct": float(used_pct),
        }
    except Exception:
        return None


def _estimate_subsample_host_ram_gib(
    n_frames: int, width: int, height: int, channels: int = 3
) -> float:
    """Worst-case host-RAM estimate for the SAM3 video tensor (CPU side).

    SAM3 / OpenCV typically materialises one **decoded** uint8 frame buffer per
    session frame on the CPU before the tensor is moved to GPU. ``float32``
    backbone work then transiently doubles peak usage. We report **2.5x** the
    raw uint8 byte count as a single-number budget that users can compare with
    available RAM before launching long clips.
    """
    raw_bytes = max(0, int(n_frames)) * max(1, int(width)) * max(1, int(height)) * int(channels)
    return 2.5 * (raw_bytes / (1024**3))


def _max_frames_was_explicit(cli_value: int | None) -> bool:
    """True when user/env explicitly set max frames; false means auto mode."""
    if cli_value is not None:
        return True
    raw = os.environ.get("SAM3_MAX_FRAMES", "").strip()
    return bool(raw)


def _scaled_dims_for_long_edge(width: int, height: int, long_edge_cap: int) -> tuple[int, int]:
    """Return dimensions after the SAM3 input long-edge cap is applied."""
    width = max(1, int(width))
    height = max(1, int(height))
    if long_edge_cap <= 0:
        return width, height
    long_edge = max(width, height)
    if long_edge <= long_edge_cap:
        return width, height
    scale = float(long_edge_cap) / float(long_edge)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _host_ram_adjusted_auto_max_frames(
    video_files: list[Path],
    *,
    max_input_frames: int | None,
    max_input_long_edge: int | None,
) -> int | None:
    """Lower auto max_frames when the host-RAM estimate already predicts SIGKILL.

    This only changes true auto mode. CLI ``--max-frames`` and ``SAM3_MAX_FRAMES``
    remain authoritative because those are explicit user choices.
    """
    if not video_files or _max_frames_was_explicit(max_input_frames):
        return None
    host = _host_ram_profile()
    if host is None:
        return None
    available_gib = float(host["available_gib"])
    if available_gib <= 0.0:
        return None

    current = _resolve_max_input_frames_value(max_input_frames)
    if current <= 0:
        return None
    long_edge_cap = _read_max_input_long_edge(max_input_long_edge)

    suggested = current
    worst_est = 0.0
    worst_name = ""
    for vf in video_files:
        try:
            meta = _video_basic_meta(vf)
        except Exception:
            continue
        n_total = int(meta.get("frames", meta.get("frame_count", 0)) or 0)
        w = int(meta.get("width", 0) or 0)
        h = int(meta.get("height", 0) or 0)
        if n_total <= 0 or w <= 0 or h <= 0:
            continue
        eff_w, eff_h = _scaled_dims_for_long_edge(w, h, long_edge_cap)
        n_session = min(n_total, current)
        ram_est = _estimate_subsample_host_ram_gib(n_session, eff_w, eff_h)
        if ram_est > worst_est:
            worst_est = ram_est
            worst_name = vf.name
        if ram_est <= 0.80 * available_gib:
            continue

        per_frame_gib = _estimate_subsample_host_ram_gib(1, eff_w, eff_h)
        ram_cap = int((0.45 * available_gib) / per_frame_gib) if per_frame_gib > 0 else current
        conservative_cap = max(32, min(512, current // 4))
        suggested = min(suggested, max(32, min(conservative_cap, ram_cap)))

    if suggested >= current:
        return None

    print("", flush=True)
    print("[SAM3 RAM] Auto max_frames adjusted to avoid host-RAM OOM:", flush=True)
    print(
        f"  max_frames auto {current} -> {suggested} "
        f"(worst estimate: {worst_name or 'video'} ~{worst_est:.1f} GiB "
        f"vs {available_gib:.1f} GiB available)",
        flush=True,
    )
    print(
        "  Override if needed with --max-frames N or SAM3_MAX_FRAMES=N. "
        "Use --max-frames 0 only when host RAM is enough for the full clip.",
        flush=True,
    )
    return suggested


def _warn_host_ram_for_videos(video_files: list[Path], *, max_input_frames: int | None) -> None:
    """Warn (without aborting) when the planned SAM3 session may exceed host RAM.

    Subprocess exit=-9 (SIGKILL) on long broadcast clips is almost always the
    Linux OOM killer firing on **host RAM**, not VRAM. We estimate the worst
    case from each clip's resolution and the resolved temporal cap, and print
    a clear hint with the recommended ``--max-frames`` reduction.
    """
    if not video_files:
        return
    host = _host_ram_profile()
    if host is None:
        return
    available_gib = float(host["available_gib"])
    resolved_cap = _resolve_max_input_frames_value(max_input_frames)
    cap_label = "FULL CLIP (no temporal cap)" if resolved_cap <= 0 else f"max_frames={resolved_cap}"

    warned = False
    for vf in video_files:
        try:
            meta = _video_basic_meta(vf)
        except Exception:
            continue
        n_total = int(meta.get("frames", meta.get("frame_count", 0)) or 0)
        w = int(meta.get("width", 0) or 0)
        h = int(meta.get("height", 0) or 0)
        if n_total <= 0 or w <= 0 or h <= 0:
            continue
        n_session = n_total if resolved_cap <= 0 else min(n_total, resolved_cap)
        ram_est = _estimate_subsample_host_ram_gib(n_session, w, h)
        if ram_est > 0.6 * available_gib or ram_est > 20.0:
            if not warned:
                print("", flush=True)
                print("[SAM3 RAM] Host-RAM heads-up (helps avoid SIGKILL / exit=-9):", flush=True)
                warned = True
            print(
                f"  - {vf.name}: {w}x{h} x {n_session} frames "
                f"(of {n_total}) -> est. peak ~{ram_est:.1f} GiB host RAM "
                f"vs {available_gib:.1f} GiB available  [{cap_label}]",
                flush=True,
            )
    if warned:
        suggested_cap = max(64, min(512, resolved_cap // 4)) if resolved_cap > 256 else 128
        print(
            "  Tip: long broadcast clips often need --max-frames "
            f"{suggested_cap} (or 64/32) and --max-input-long-edge 1280 "
            "to fit in host RAM. exit=-9 means the OS killed the SAM "
            "subprocess (oom-killer), not a CUDA error.",
            flush=True,
        )
        print("", flush=True)


def _format_runtime_banner(args: argparse.Namespace, video_files: list[Path] | None) -> str:
    """Compact, copy-paste-friendly summary printed at the start of every CLI run."""
    lines: list[str] = []
    lines.append("=" * 62)
    lines.append("vailá SAM 3 — runtime configuration")
    lines.append("=" * 62)
    lines.append(f"  input          : {args.input}")
    lines.append(f"  output (parent): {args.output}")
    if getattr(args, "output_base", None):
        lines.append(f"  output_base    : {args.output_base}")
    if getattr(args, "chunk_size", None):
        lines.append(f"  chunk_size     : {args.chunk_size}")
    lines.append(f"  text prompt    : {args.text!r}    (frame index={args.frame})")
    mf_disp = (
        "auto"
        if args.max_frames is None
        else ("FULL CLIP" if args.max_frames == 0 else str(args.max_frames))
    )
    le_disp = (
        "auto"
        if args.max_input_long_edge is None
        else ("native" if args.max_input_long_edge == 0 else f"{args.max_input_long_edge}px")
    )
    lines.append(f"  max_frames     : {mf_disp}   max_input_long_edge: {le_disp}")
    lines.append(
        f"  overlay_rich={bool(args.overlay_rich)}  draw(contour={bool(args.draw_contour)},"
        f" box={bool(args.draw_box)}, id={bool(args.draw_id)}, centroid={bool(args.draw_centroid)})"
    )
    lines.append(
        f"  save_overlay_mp4={not args.no_overlay}  save_mask_png={not args.no_png}"
        f"  save_contours={bool(args.save_contours)}  save_tracks_csv={bool(args.save_tracks_csv)}"
    )
    lines.append(
        f"  contours_format={args.contours_format}  contours_gzip={bool(args.contours_gzip)}"
        f"  postprocess_points={args.postprocess_points}"
    )
    if (
        getattr(args, "tracks_only", False)
        or getattr(args, "delete_mask_png", False)
        or getattr(args, "keep_mask_png", False)
        or getattr(args, "keep_masks", False)
        or getattr(args, "stabilize_ids", False)
    ):
        resolved_delete = (
            not bool(getattr(args, "keep_mask_png", False))
            and not bool(getattr(args, "keep_masks", False))
        ) or bool(getattr(args, "delete_mask_png", False))
        lines.append(
            f"  tracks_only={bool(getattr(args, 'tracks_only', False))}  "
            f"delete_mask_png={resolved_delete}  "
            f"stabilize_ids={bool(getattr(args, 'stabilize_ids', False))}"
        )
    if args.checkpoint is not None:
        lines.append(f"  checkpoint     : {args.checkpoint}")

    vram = _sam3_vram_profile()
    if vram is not None:
        lines.append(
            f"  GPU VRAM       : {vram['total_gib']:.1f} GiB total / "
            f"{vram['free_gib']:.1f} GiB free  (safe auto frames={int(vram['safe_frames'])})"
        )
    else:
        lines.append("  GPU VRAM       : CUDA unavailable (SAM3 video will refuse to run).")

    host = _host_ram_profile()
    if host is not None:
        lines.append(
            f"  Host RAM       : {host['total_gib']:.1f} GiB total / "
            f"{host['available_gib']:.1f} GiB available ({host['used_pct']:.0f}% used)"
        )
    else:
        lines.append("  Host RAM       : (psutil + /proc/meminfo unavailable)")

    if video_files:
        lines.append(f"  videos queued  : {len(video_files)}")
        if len(video_files) <= 5:
            for vf in video_files:
                lines.append(f"      - {vf.name}")
        else:
            for vf in video_files[:3]:
                lines.append(f"      - {vf.name}")
            lines.append(f"      … (+{len(video_files) - 3} more)")
    lines.append("=" * 62)
    return "\n".join(lines)


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
    max_input_long_edge: int | None = None,
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
    if max_input_long_edge is not None:
        lines.append(f"max_input_long_edge (CLI)={int(max_input_long_edge)}")
    else:
        lines.append("max_input_long_edge (CLI)=auto/env")

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


def _video_basic_meta(video_path: Path) -> dict[str, float | int | str]:
    """Best-effort metadata probe via OpenCV (no decode pass)."""
    vp = str(video_path.resolve())
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        return {
            "path": vp,
            "name": video_path.name,
            "ok": 0,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frames": 0,
            "duration_s": 0.0,
            "long_edge": 0,
        }
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n <= 0:
        n = _video_frame_count(vp)
    duration_s = float(n) / fps if fps > 0.0 and n > 0 else 0.0
    return {
        "path": vp,
        "name": video_path.name,
        "ok": 1,
        "width": w,
        "height": h,
        "fps": fps,
        "frames": int(max(0, n)),
        "duration_s": float(max(0.0, duration_s)),
        "long_edge": int(max(w, h)),
    }


def _sam3_preflight_scan(
    input_path: Path,
    *,
    output_base: Path,
    max_input_frames: int | None,
    max_input_long_edge: int | None,
) -> Path | None:
    """Scan videos and write a CSV summary + recommendations; no SAM3 inference."""
    video_files = _find_videos(input_path) if input_path.is_dir() else [input_path]
    if not video_files:
        print(f"[preflight] No video files found under {input_path}")
        return None

    vram_profile = _sam3_vram_profile()
    mf_resolved = _resolve_max_input_frames_value(max_input_frames, auto_profile=vram_profile)
    le_cap = _read_max_input_long_edge(max_input_long_edge)
    safe_auto = int(vram_profile["safe_frames"]) if vram_profile is not None else 32

    output_base.mkdir(parents=True, exist_ok=True)
    out_csv = output_base / "SAM3_PREFLIGHT.csv"
    import csv as _csv

    max_long_edge_seen = 0
    max_frames_seen = 0
    max_duration_seen = 0.0

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "path",
                "ok",
                "width",
                "height",
                "long_edge",
                "fps",
                "frames",
                "duration_s",
                "suggest_max_input_long_edge",
                "suggest_max_frames",
                "recommended_flags",
                "risk",
                "notes",
            ],
        )
        w.writeheader()
        for vf in video_files:
            m = _video_basic_meta(vf)
            long_edge = int(m["long_edge"]) if isinstance(m["long_edge"], int) else 0
            frames = int(m["frames"]) if isinstance(m["frames"], int) else 0
            duration_s = float(m["duration_s"]) if isinstance(m["duration_s"], float) else 0.0

            if long_edge > max_long_edge_seen:
                max_long_edge_seen = long_edge
            if frames > max_frames_seen:
                max_frames_seen = frames
            if duration_s > max_duration_seen:
                max_duration_seen = duration_s

            suggest_le = le_cap
            notes: list[str] = []
            if long_edge <= 0:
                notes.append("unreadable_meta")
            if le_cap > 0 and long_edge > 0 and long_edge > le_cap:
                notes.append(f"downscale_long_edge({long_edge}->{le_cap})")
            if le_cap == 0 and long_edge >= 3000:
                notes.append("high_res_no_cap(consider --max-input-long-edge)")

            suggest_mf = mf_resolved
            if frames > 0 and suggest_mf > 0 and frames > suggest_mf:
                notes.append(f"will_subsample_frames({frames}->{suggest_mf})")
            if frames > 0 and suggest_mf <= 0:
                notes.append("full_clip_mode(consider --max-frames)")

            risk = "low"
            recommended_flags: list[str] = []
            if long_edge > 0 and le_cap > 0 and long_edge > le_cap:
                risk = "medium"
                recommended_flags += ["--max-input-long-edge", str(le_cap)]
            if frames > 0 and frames > safe_auto:
                risk = "high"
                recommended_flags += ["--max-frames", str(safe_auto)]
            if frames > 0 and frames > (10 * safe_auto):
                notes.append("very_long_clip(consider --frame-by-frame if OOM)")

            w.writerow(
                {
                    **m,
                    "suggest_max_input_long_edge": int(suggest_le),
                    "suggest_max_frames": int(suggest_mf),
                    "recommended_flags": " ".join(recommended_flags),
                    "risk": risk,
                    "notes": ";".join(notes),
                }
            )

    print(f"[preflight] Videos: {len(video_files)}")
    if vram_profile is None:
        print("[preflight] VRAM profile: unavailable (CUDA not accessible)")
    else:
        print(
            "[preflight] VRAM profile: "
            f"total={vram_profile['total_gib']:.2f} GiB free={vram_profile['free_gib']:.2f} GiB "
            f"safe_auto={int(vram_profile['safe_frames'])} frames"
        )
    print(
        "[preflight] Effective caps (current CLI/env): "
        f"max_frames={mf_resolved} max_long_edge={le_cap} safe_auto_frames={safe_auto}"
    )
    print(f"[preflight] Wrote: {out_csv}")

    suggest_cmd: list[str] = [
        "uv",
        "run",
        "vaila/vaila_sam.py",
        "-i",
        str(input_path),
        "-o",
        str(output_base),
        "-t",
        "person",
        "--postprocess-points",
        "all",
    ]
    if max_long_edge_seen > 0 and le_cap > 0 and max_long_edge_seen > le_cap:
        suggest_cmd += ["--max-input-long-edge", str(le_cap)]
    if max_frames_seen > safe_auto:
        suggest_cmd += ["--max-frames", str(safe_auto)]
    print("[preflight] Suggested command (conservative caps based on scan):")
    print("  " + " ".join(suggest_cmd))
    if max_duration_seen > 0.0:
        print(
            f"[preflight] Largest clip: long_edge={max_long_edge_seen}px "
            f"frames={max_frames_seen} duration≈{max_duration_seen:.1f}s"
        )
    return out_csv


def _nearest_sess_idx_for_orig_frame(frame_idx: int, sess_to_orig: np.ndarray) -> int:
    """Map an original-timeline frame index to a SAM session frame index.

    VRAM subsampling feeds SAM a shortened clip; each session frame corresponds to one
    original frame index in ``sess_to_orig``. Using the latest keyframe with
    ``sess_to_orig[k] <= frame_idx`` lags overlays by up to ~half the subsample spacing
    (masks look ``behind`` motion). Nearest keyframe in time fixes that.
    """
    if sess_to_orig.size <= 0:
        return 0
    a = sess_to_orig
    ins = int(np.searchsorted(a, frame_idx, side="left"))
    candidates: list[int] = []
    if ins > 0:
        candidates.append(ins - 1)
    if ins < int(a.size):
        candidates.append(ins)
    if not candidates:
        return 0
    # Prefer smaller temporal error; on ties, prefer the larger original index (newer keyframe).
    return int(min(candidates, key=lambda j: (abs(int(a[j]) - frame_idx), -int(a[j]))))


def _maybe_subsample_video_for_vram(
    video_path: Path,
    output_dir: Path,
    max_frames: int,
) -> tuple[Path, Path | None, int, np.ndarray]:
    """If the clip has more than ``max_frames``, write a temp MP4 with evenly spaced frames.

    Returns ``(path_for_sam3, temp_path_or_none, num_frames_in_that_path, orig_frame_indices)``.
    SAM3 loads the full tensor to GPU in ``init_state``; capping frames avoids OOM on 8GB cards.
    """
    vp = str(video_path.resolve())
    if max_frames <= 0:
        n = _video_frame_count(vp)
        return video_path.resolve(), None, n, np.arange(max(0, n), dtype=np.int64)

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
        return video_path.resolve(), None, n, np.arange(max(0, n), dtype=np.int64)

    indices = np.unique(np.linspace(0, n - 1, num=max_frames, dtype=np.int64))
    print(
        f"[SAM3] Temporal subsample: {n} → {len(indices)} frames (max_frames={max_frames}). "
        "Overlay uses nearest SAM keyframe on each original frame — wide spacing looks like lag. "
        "Use Max frames = 0 / --max-frames 0 for full clip if VRAM allows.",
        flush=True,
    )
    wanted: set[int] = {int(x) for x in indices.tolist()}
    out_path = output_dir / "_sam3_subsample_input.mp4"
    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        raise OSError(f"Could not open video: {vp}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    # Preserve *duration* when subsampling: we keep fewer frames that represent the full clip,
    # so we must reduce the output FPS proportionally; otherwise the clip (and SAM overlay)
    # plays back faster than real time.
    fps_out = float(fps) * (float(len(indices)) / float(n)) if n > 0 else float(fps)
    fps_out = max(1e-3, fps_out)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps_out), (w, h))
    if not writer.isOpened():
        cap.release()
        raise OSError("Could not open VideoWriter for SAM3 subsample (try another codec/OS path)")
    written_orig: list[int] = []
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if fi in wanted:
            writer.write(frame)
            written_orig.append(fi)
        fi += 1
    cap.release()
    writer.release()
    written = len(written_orig)
    if written == 0:
        with contextlib.suppress(OSError):
            out_path.unlink(missing_ok=True)
        raise OSError("Subsampled zero frames; check video path/codec")
    orig_kept = np.asarray(written_orig, dtype=np.int64)
    return out_path.resolve(), out_path.resolve(), written, orig_kept


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
    overlap_frames: int = 2,
) -> list[tuple[Path, int, int]]:
    """Split a video into overlapping temporal chunks.

    Returns ``(chunk_mp4_path, start_frame_inclusive, end_frame_exclusive)``.
    Adjacent chunks share ``overlap_frames`` frames so the merge stage can link
    chunk-local SAM IDs by exact same-frame IoU/centroid association before
    duplicate overlap frames are dropped from the final output.
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

    chunk_size = max(1, int(chunk_size))
    overlap_frames = max(0, min(int(overlap_frames), chunk_size - 1))
    step = max(1, chunk_size - overlap_frames)

    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[tuple[Path, int, int]] = []
    cap = cv2.VideoCapture(vp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]

    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk_path = chunk_dir / f"_chunk_{start:06d}.mp4"
        writer = cv2.VideoWriter(str(chunk_path), fourcc, float(fps), (w, h))
        if not writer.isOpened():
            cap.release()
            raise OSError(f"Could not open VideoWriter for chunk {start}")
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
        if end >= n:
            break
        start += step
    cap.release()
    return chunks


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def _row_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, ""))
    except Exception:
        return default


def _track_row_detection(
    row: dict[str, str], global_frame: int, obj_id: int
) -> dict[str, Any] | None:
    try:
        x = float(row["x_px"])
        y = float(row["y_px"])
        w_box = float(row["w_px"])
        h_box = float(row["h_px"])
    except Exception:
        return None
    if not np.isfinite([x, y, w_box, h_box]).all():
        return None
    cx = _row_float(row, "cx_px", x + w_box * 0.5)
    cy = _row_float(row, "cy_px", y + h_box * 0.5)
    return {
        "frame": int(global_frame),
        "obj_id": int(obj_id),
        "local_obj_id": int(obj_id),
        "bbox": (x, y, w_box, h_box),
        "centroid": (float(cx), float(cy)),
    }


def _collect_chunk_local_ids(chunk_out: Path) -> set[int]:
    ids: set[int] = set()
    tracks_fields, tracks_rows = _csv_rows(chunk_out / "sam_tracks.csv")
    if {"obj_id"}.issubset(tracks_fields):
        for row in tracks_rows:
            with contextlib.suppress(Exception):
                ids.add(int(float(row["obj_id"])))
    meta = chunk_out / "sam_frames_meta.csv"
    if meta.is_file():
        header = meta.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        for col in header.split(","):
            col = col.strip()
            if col.startswith("box_x_"):
                with contextlib.suppress(ValueError):
                    ids.add(int(col.split("_")[-1]))
    return ids


def _load_chunk_mask_u8(chunk_out: Path, local_frame: int, obj_id: int) -> np.ndarray | None:
    """Load a chunk-local binary mask PNG when present."""
    mask_path = chunk_out / "masks" / f"frame_{local_frame:06d}_obj_{obj_id}.png"
    if not mask_path.is_file():
        return None
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    return img if img is not None else None


def _build_cross_chunk_id_maps(
    chunks: list[tuple[Path, int, int]],
    chunk_output_dirs: list[Path],
    *,
    max_centroid_dist_px: float = 180.0,
    min_iou: float = 0.05,
    mask_iou_weight: float = 0.25,
) -> list[dict[int, int]]:
    """Map each chunk-local object ID to a global ID using overlap frames.

    Implements the **Cross-Chunk Tracklet Linking with Overlap** spec:

    1. **Sliding window overlap** — adjacent chunks share 1-2 frames written by
       :func:`_split_video_into_chunks` (``overlap_frames=2`` default).
    2. **Feature caching** — for each chunk, the last overlap frame keeps a
       cache of ``(obj_id, bbox, centroid)`` from ``sam_tracks.csv``.
    3. **Graph-based association** — at chunk boundary N→N+1, a bipartite
       cost matrix combines spatial IoU and centroid distance on shared frames.
    4. **Optimal matching** — :func:`_assignment_min_cost` (Hungarian when
       SciPy present) finds the 1:1 mapping.
    5. **Linked-list merging** — matched chunk-local IDs of N+1 inherit the
       persistent global ID from N; unmatched IDs allocate a new global ID.
    """
    maps: list[dict[int, int]] = []
    history_by_frame: dict[int, list[dict[str, Any]]] = {}
    next_gid = 0

    for ci, (chunk_info, chunk_out) in enumerate(zip(chunks, chunk_output_dirs, strict=True)):
        _, start_frame, _end_frame = chunk_info
        write_start = start_frame if ci == 0 else max(start_frame, chunks[ci - 1][2])
        local_ids = _collect_chunk_local_ids(chunk_out)
        _fields, track_rows = _csv_rows(chunk_out / "sam_tracks.csv")

        detections_by_local: dict[int, list[dict[str, Any]]] = {oid: [] for oid in local_ids}
        for row in track_rows:
            with contextlib.suppress(Exception):
                local_frame = int(float(row["frame"]))
                oid = int(float(row["obj_id"]))
                global_frame = start_frame + local_frame
                det = _track_row_detection(row, global_frame, oid)
                if det is not None:
                    detections_by_local.setdefault(oid, []).append(det)

        mapping: dict[int, int] = {}
        if ci > 0 and detections_by_local:
            candidate_locals = sorted(
                oid
                for oid, dets in detections_by_local.items()
                if any(d["frame"] < write_start for d in dets)
            )
            candidate_globals = sorted(
                {
                    int(prev["obj_id"])
                    for gf, prevs in history_by_frame.items()
                    if start_frame <= gf < write_start
                    for prev in prevs
                }
            )
            if candidate_locals and candidate_globals:
                cost = np.full((len(candidate_locals), len(candidate_globals)), 1e6, dtype=float)
                ok_pair: dict[tuple[int, int], bool] = {}
                for li, local_oid in enumerate(candidate_locals):
                    dets = [
                        d
                        for d in detections_by_local.get(local_oid, [])
                        if d["frame"] < write_start
                    ]
                    for gi, global_oid in enumerate(candidate_globals):
                        vals: list[float] = []
                        ious: list[float] = []
                        dists: list[float] = []
                        for det in dets:
                            for prev in history_by_frame.get(int(det["frame"]), []):
                                if int(prev["obj_id"]) != global_oid:
                                    continue
                                iou = bbox_iou_xywh(det["bbox"], prev["bbox"])
                                dcx = float(det["centroid"][0]) - float(prev["centroid"][0])
                                dcy = float(det["centroid"][1]) - float(prev["centroid"][1])
                                dist = float(np.hypot(dcx, dcy))
                                cost_val = (1.0 - iou) + min(1.0, dist / max_centroid_dist_px)
                                if mask_iou_weight > 0.0:
                                    local_oid = int(det.get("local_obj_id", 0))
                                    local_frame = int(det["frame"]) - start_frame
                                    det_mask = _load_chunk_mask_u8(
                                        chunk_out, local_frame, local_oid
                                    )
                                    prev_local_frame = int(det["frame"]) - chunks[ci - 1][1]
                                    prev_local_oid = int(prev.get("local_obj_id", global_oid))
                                    prev_mask = _load_chunk_mask_u8(
                                        chunk_output_dirs[ci - 1],
                                        prev_local_frame,
                                        prev_local_oid,
                                    )
                                    if det_mask is not None and prev_mask is not None:
                                        miou = mask_iou_u8(det_mask, prev_mask)
                                        cost_val += float(mask_iou_weight) * (1.0 - miou)
                                vals.append(cost_val)
                                ious.append(iou)
                                dists.append(dist)
                        if vals:
                            mean_iou = float(np.mean(ious))
                            mean_dist = float(np.mean(dists))
                            ok_pair[(li, gi)] = (
                                mean_iou >= min_iou or mean_dist <= max_centroid_dist_px
                            )
                            cost[li, gi] = float(np.mean(vals))
                for li, gi in assignment_min_cost(cost):
                    if cost[li, gi] >= 1e5 or not ok_pair.get((li, gi), False):
                        continue
                    mapping[candidate_locals[li]] = candidate_globals[gi]

        for oid in sorted(local_ids):
            if oid not in mapping:
                mapping[oid] = next_gid
                next_gid += 1
        if mapping:
            next_gid = max(next_gid, max(mapping.values()) + 1)
        maps.append(mapping)

        for oid, dets in detections_by_local.items():
            gid = mapping.get(oid)
            if gid is None:
                continue
            for det in dets:
                det2 = dict(det)
                det2["obj_id"] = gid
                history_by_frame.setdefault(int(det2["frame"]), []).append(det2)

    return maps


def _render_overlay_from_merged_masks(
    final_output_dir: Path,
    video_path: Path,
    *,
    draw_contour: bool = True,
    draw_box: bool = True,
    draw_id: bool = True,
    draw_centroid: bool = False,
) -> bool:
    """Render final overlay from original frames and remapped mask filenames."""
    masks_dir = final_output_dir / "masks"
    tracks_path = final_output_dir / "sam_tracks.csv"
    if not masks_dir.is_dir() or not tracks_path.is_file():
        return False
    fields, rows = _csv_rows(tracks_path)
    if not rows or not {"frame", "obj_id", "x_px", "y_px", "w_px", "h_px"}.issubset(fields):
        return False

    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        with contextlib.suppress(Exception):
            by_frame.setdefault(int(float(row["frame"])), []).append(row)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = final_output_dir / f"{video_path.stem}_sam_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        cap.release()
        return False

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_rows = by_frame.get(frame_idx, [])
        masks: list[np.ndarray] = []
        oids: list[int] = []
        probs: list[float] = []
        boxes: list[list[float]] = []
        for row in frame_rows:
            try:
                oid = int(float(row["obj_id"]))
                mask_path = masks_dir / f"frame_{frame_idx:06d}_obj_{oid}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                masks.append(mask > 127)
                oids.append(oid)
                probs.append(_row_float(row, "score"))
                boxes.append(
                    [
                        _row_float(row, "x_px"),
                        _row_float(row, "y_px"),
                        _row_float(row, "w_px"),
                        _row_float(row, "h_px"),
                    ]
                )
            except Exception:
                continue
        if masks:
            comp = _composite_masks_bgr(
                frame,
                np.asarray(masks, dtype=bool),
                np.asarray(oids, dtype=np.int32),
                probs=np.asarray(probs, dtype=np.float32),
                boxes_xywh=np.asarray(boxes, dtype=np.float32),
                draw_box=draw_box,
                draw_id=draw_id,
                draw_contour=draw_contour,
                draw_centroid=draw_centroid,
            )
            writer.write(comp)
        else:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return True


def _merge_chunk_outputs(
    chunks: list[tuple[Path, int, int]],
    chunk_output_dirs: list[Path],
    final_output_dir: Path,
    video_path: Path,
    *,
    save_overlay_mp4: bool = True,
    save_mask_png: bool = True,
    draw_contour: bool = True,
    draw_box: bool = True,
    draw_id: bool = True,
    draw_centroid: bool = False,
) -> None:
    """Merge overlapping SAM3 chunks and link chunk-local IDs into global IDs."""
    import shutil

    final_masks_dir = final_output_dir / "masks"
    if save_mask_png:
        final_masks_dir.mkdir(parents=True, exist_ok=True)

    id_maps = _build_cross_chunk_id_maps(chunks, chunk_output_dirs)
    all_unique_oids: set[int] = set()
    for m in id_maps:
        all_unique_oids.update(m.values())
    sorted_oids = sorted(all_unique_oids)

    header_line = ""
    if sorted_oids:
        header_cols = ["frame"]
        for oid in sorted_oids:
            header_cols.extend(
                [f"box_x_{oid}", f"box_y_{oid}", f"box_w_{oid}", f"box_h_{oid}", f"prob_{oid}"]
            )
        header_line = ",".join(header_cols)

    all_meta_rows: list[str] = []
    merged_tracks_rows: list[str] = []
    merged_manifest_rows: list[str] = ["frame,obj_id,area_px,mask_png"]
    merged_contours_frames: list[dict[str, Any]] = []
    merged_contour_oids: set[int] = set()

    for ci, (chunk_info, chunk_out) in enumerate(zip(chunks, chunk_output_dirs, strict=True)):
        _, start_frame, _end_frame = chunk_info
        write_start = start_frame if ci == 0 else max(start_frame, chunks[ci - 1][2])
        id_map = id_maps[ci] if ci < len(id_maps) else {}

        meta_csv = chunk_out / "sam_frames_meta.csv"
        if meta_csv.is_file() and header_line:
            lines = meta_csv.read_text(encoding="utf-8").strip().split("\n")
            chunk_cols = lines[0].split(",") if lines else []
            chunk_oid_cols: dict[int, int] = {}
            for col_idx, col in enumerate(chunk_cols):
                col = col.strip()
                if col.startswith("box_x_"):
                    with contextlib.suppress(ValueError):
                        chunk_oid_cols[int(col.split("_")[-1])] = col_idx
            for row_line in lines[1:]:
                parts = row_line.split(",")
                if not parts:
                    continue
                try:
                    local_frame = int(parts[0])
                except ValueError:
                    continue
                global_frame = start_frame + local_frame
                if global_frame < write_start:
                    continue
                values_by_gid: dict[int, list[str]] = {}
                for local_oid, ci_start in chunk_oid_cols.items():
                    gid = id_map.get(local_oid, local_oid)
                    vals = parts[ci_start : ci_start + 5]
                    while len(vals) < 5:
                        vals.append("")
                    if any(v.strip() for v in vals):
                        values_by_gid[gid] = vals
                row_parts = [str(global_frame)]
                for oid in sorted_oids:
                    row_parts.extend(values_by_gid.get(oid, ["", "", "", "", ""]))
                all_meta_rows.append(",".join(row_parts))

        if save_mask_png and (chunk_out / "masks").is_dir():
            for png in sorted((chunk_out / "masks").glob("frame_*_obj_*.png")):
                parts_name = png.stem.split("_")
                try:
                    local_idx = int(parts_name[1])
                    local_oid = int(parts_name[3])
                except (ValueError, IndexError):
                    continue
                global_idx = start_frame + local_idx
                if global_idx < write_start:
                    continue
                gid = id_map.get(local_oid, local_oid)
                dest = final_masks_dir / f"frame_{global_idx:06d}_obj_{gid}.png"
                shutil.copy2(str(png), str(dest))

        tracks_csv = chunk_out / "sam_tracks.csv"
        if tracks_csv.is_file():
            lines = tracks_csv.read_text(encoding="utf-8").strip().split("\n")
            for ln in lines[1:]:
                if not ln.strip():
                    continue
                parts = ln.split(",")
                if len(parts) < 2:
                    continue
                try:
                    local_f = int(float(parts[0]))
                    local_oid = int(float(parts[1]))
                except ValueError:
                    continue
                global_f = start_frame + local_f
                if global_f < write_start:
                    continue
                parts[0] = str(global_f)
                parts[1] = str(id_map.get(local_oid, local_oid))
                if len(parts) == 10:
                    parts.extend(["", ""])
                merged_tracks_rows.append(",".join(parts))

        manifest_csv = chunk_out / "sam_masks_manifest.csv"
        if manifest_csv.is_file():
            lines = manifest_csv.read_text(encoding="utf-8").strip().split("\n")
            for ln in lines[1:]:
                if not ln.strip():
                    continue
                parts = ln.split(",")
                if len(parts) < 4:
                    continue
                try:
                    local_f = int(parts[0])
                    local_oid = int(parts[1])
                except ValueError:
                    continue
                global_f = start_frame + local_f
                if global_f < write_start:
                    continue
                gid = id_map.get(local_oid, local_oid)
                parts[0] = str(global_f)
                parts[1] = str(gid)
                parts[3] = f"masks/frame_{global_f:06d}_obj_{gid}.png"
                merged_manifest_rows.append(",".join(parts[:4]))

        contours_candidates = [
            ("json", chunk_out / "sam_contours.json"),
            ("json", chunk_out / "sam_contours.json.gz"),
            ("jsonl", chunk_out / "sam_contours.jsonl"),
            ("jsonl", chunk_out / "sam_contours.jsonl.gz"),
        ]
        found = next(((fmt, p) for fmt, p in contours_candidates if p.is_file()), None)
        if found is not None:
            fmt, cpath = found
            if cpath.suffix == ".gz":
                import gzip

                with gzip.open(cpath, "rt", encoding="utf-8") as fh:
                    raw = fh.read()
            else:
                raw = cpath.read_text(encoding="utf-8")
            if fmt == "json":
                payload = json.loads(raw)
                frames = payload.get("frames") or []
            else:
                lines = [ln for ln in raw.split("\n") if ln.strip()]
                frames = [json.loads(ln) for ln in lines[1:]]
            for fr in frames:
                try:
                    local_f = int(fr.get("frame"))
                except Exception:
                    continue
                global_f = start_frame + local_f
                if global_f < write_start:
                    continue
                fr2 = dict(fr)
                fr2["frame"] = global_f
                objs = []
                for obj in fr.get("objects") or []:
                    obj2 = dict(obj)
                    local_oid = obj2.get("obj_id")
                    if isinstance(local_oid, int):
                        gid = id_map.get(local_oid, local_oid)
                        obj2["obj_id"] = gid
                        merged_contour_oids.add(gid)
                        if save_mask_png and obj2.get("mask_png"):
                            obj2["mask_png"] = f"masks/frame_{global_f:06d}_obj_{gid}.png"
                    objs.append(obj2)
                fr2["objects"] = objs
                merged_contours_frames.append(fr2)

    if header_line and all_meta_rows:
        all_meta_rows.sort(key=lambda r: int(r.split(",")[0]) if r.split(",")[0].isdigit() else 0)
        (final_output_dir / "sam_frames_meta.csv").write_text(
            header_line + "\n" + "\n".join(all_meta_rows) + "\n",
            encoding="utf-8",
        )

    if merged_tracks_rows:
        merged_tracks_rows.sort(
            key=lambda r: (
                int(r.split(",")[0]) if r.split(",")[0].isdigit() else 0,
                int(r.split(",")[1]) if len(r.split(",")) > 1 and r.split(",")[1].isdigit() else 0,
            )
        )
        (final_output_dir / "sam_tracks.csv").write_text(
            "frame,obj_id,x_px,y_px,w_px,h_px,score,area_px,n_polygons,largest_polygon_pts,cx_px,cy_px\n"
            + "\n".join(merged_tracks_rows)
            + "\n",
            encoding="utf-8",
        )

    if len(merged_manifest_rows) > 1:
        (final_output_dir / "sam_masks_manifest.csv").write_text(
            "\n".join(merged_manifest_rows) + "\n",
            encoding="utf-8",
        )

    if merged_contours_frames:
        merged_contours_frames.sort(key=lambda d: int(d.get("frame", 0)))
        cap = cv2.VideoCapture(str(video_path))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) if cap.isOpened() else 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        with contextlib.suppress(Exception):
            cap.release()
        payload = {
            "schema": "vaila_sam_contours_v1",
            "video": video_path.name,
            "width": int(w),
            "height": int(h),
            "fps": float(fps),
            "n_frames": int(nframes),
            "object_ids": sorted(int(x) for x in merged_contour_oids),
            "frames": merged_contours_frames,
        }
        (final_output_dir / "sam_contours.json").write_text(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if save_overlay_mp4:
        rendered = False
        if save_mask_png:
            rendered = _render_overlay_from_merged_masks(
                final_output_dir,
                video_path,
                draw_contour=draw_contour,
                draw_box=draw_box,
                draw_id=draw_id,
                draw_centroid=draw_centroid,
            )
        if not rendered:
            overlay_parts = []
            for chunk_out in chunk_output_dirs:
                overlays = list(chunk_out.glob("*_sam_overlay.mp4"))
                if overlays:
                    overlay_parts.append(overlays[0])
            if overlay_parts:
                _stitch_overlay_mp4s(overlay_parts, final_output_dir, video_path, chunks=chunks)


def _stitch_overlay_mp4s(
    parts: list[Path],
    final_output_dir: Path,
    video_path: Path,
    *,
    chunks: list[tuple[Path, int, int]] | None = None,
) -> None:
    """Concatenate overlay MP4 segments, skipping duplicated overlap frames."""
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
    for pi, part in enumerate(parts):
        cap = cv2.VideoCapture(str(part))
        local_idx = 0
        start_frame = 0
        write_start = 0
        if chunks is not None and pi < len(chunks):
            _chunk_path, start_frame, _end_frame = chunks[pi]
            write_start = start_frame if pi == 0 else max(start_frame, chunks[pi - 1][2])
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            global_frame = start_frame + local_idx
            local_idx += 1
            if global_frame < write_start:
                continue
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
    overlay_rich: bool = True,
    draw_contour: bool = True,
    draw_box: bool = True,
    draw_id: bool = True,
    draw_centroid: bool = False,
    save_contours: bool = True,
    save_tracks_csv: bool = True,
    delete_mask_png: bool = False,
    stabilize_ids: bool = False,
    contours_format: str = "json",
    contours_gzip: bool = False,
    chunk_size: int | None = None,
    overlap_frames: int = 2,
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

    chunk_save_png = save_mask_png or save_overlay_mp4

    def _log(s: str) -> None:
        if log is not None:
            log(s)
        else:
            print(s)

    # Determine chunk size.  This is OOM-recovery territory: by definition we
    # are running because the in-process retry ladder failed for the source
    # resolution.  Pick a conservative value capped at 48 frames so each chunk
    # fits comfortably in 24 GiB even at 1080p×1280-long-edge (KV cache for
    # SAM3's transformer grows ~80–120 MiB per propagated frame, so 48 frames
    # ≈ 5 GiB KV + 6.3 GiB model + processing).  A larger chunk_size triggered
    # the same cascade we are trying to escape (debug 2026-04-30: chunk_size=128
    # OOMed at ~58% propagation of the first 128-frame chunk).
    if chunk_size is None or chunk_size <= 0:
        profile = _sam3_vram_profile()
        auto_safe = int(profile["safe_frames"]) if profile is not None else 64
        chunk_size = max(16, min(48, auto_safe))

    n_total = _video_frame_count(str(video_file))
    if n_total <= 0:
        return False, f"Could not read frame count from {video_file}"

    overlap_frames = max(0, int(overlap_frames))
    _log(
        f"  [SAM3-CHUNK] Divide and conquer: {n_total} frames → "
        f"chunks of ≤{chunk_size} frames with {overlap_frames}-frame overlap"
    )

    # Create chunk working directory
    chunk_work_dir = output_dir / "_chunks"
    chunk_work_dir.mkdir(parents=True, exist_ok=True)

    # Split video into chunks. Adjacent chunks share overlap frames (sliding-window
    # overlap) so cross-chunk ID linking can match identical spatial detections
    # at the boundary; see :func:`_build_cross_chunk_id_maps` for the
    # IoU+centroid Hungarian assignment that converts chunk-local SAM IDs into
    # persistent global IDs.
    _log(
        f"  [SAM3-CHUNK] Splitting video into chunks "
        f"(overlap_frames={overlap_frames} for tracklet linking)..."
    )
    try:
        chunks = _split_video_into_chunks(
            video_file, chunk_work_dir, chunk_size, overlap_frames=overlap_frames
        )
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
            # Recursion guard: a chunk that itself OOMs must NOT spawn another
            # chunked fallback (otherwise ``_chunks/out_*/_chunks/out_*/...``
            # explodes — the chunk_size=16 floor means a 16-frame chunk would
            # split into one 16-frame chunk == itself, looping forever).
            "--no-chunked-fallback",
        ]
        # Pin the chunk subprocess to a max-frames cap = chunk_size so it does
        # NOT use the optimistic auto value (e.g. auto=128 inside a 48-frame
        # chunk would still try to load 48 frames into one session, which is
        # what we want — but if the chunk happens to be shorter than auto, the
        # subprocess otherwise reports auto=128 and tries the full retry ladder
        # 128→64→32→… on EACH chunk in-process, recreating the cascade).
        effective_max_frames = chunk_size
        if max_input_frames is not None and max_input_frames > 0:
            effective_max_frames = min(chunk_size, int(max_input_frames))
        cmd += ["--max-frames", str(effective_max_frames)]
        # Force a tighter spatial cap for chunk subprocesses: the parent landed
        # in chunked fallback precisely because its frame-cap AND long-edge
        # ladders failed at the source resolution.  Re-running at the same
        # resolution per chunk would just OOM again.  Use the user's value if
        # it is already <=1280, else clamp to 1280.
        chunk_long_edge = (
            max_input_long_edge
            if isinstance(max_input_long_edge, int) and 0 < max_input_long_edge <= 1280
            else 1280
        )
        cmd += ["--max-input-long-edge", str(chunk_long_edge)]
        if checkpoint is not None:
            cmd += ["--checkpoint", str(checkpoint)]
        if not save_overlay_mp4:
            cmd.append("--no-overlay")
        if not chunk_save_png:
            cmd.append("--no-png")
        else:
            # We want the chunk subprocess to write mask PNGs so the coordinator
            # can remap/merge them. But we explicitly tell it to keep them so it
            # doesn't delete them before we can merge!
            cmd.append("--keep-mask-png")
        if not overlay_rich:
            cmd.append("--no-overlay-rich")
        if not draw_contour:
            cmd.append("--no-draw-contour")
        if not draw_box:
            cmd.append("--no-draw-box")
        if not draw_id:
            cmd.append("--no-draw-id")
        if draw_centroid:
            cmd.append("--draw-centroid")
        if not save_contours:
            cmd.append("--no-save-contours")
        if not save_tracks_csv:
            cmd.append("--no-save-tracks-csv")
        if contours_format:
            cmd += ["--contours-format", str(contours_format)]
        if contours_gzip:
            cmd.append("--contours-gzip")
        # Chunk subprocesses should NOT delete their masks (we keep them via --keep-mask-png
        # above, and the coordinator deletes them at the very end when it removes chunk_work_dir).
        # Do not run per-chunk stabilization: it rewrites chunk CSV IDs but not mask
        # filenames. Cross-chunk linking below remaps CSVs, contours, masks, and
        # the regenerated final overlay in one consistent pass.

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
            save_mask_png=chunk_save_png,
            draw_contour=draw_contour,
            draw_box=draw_box,
            draw_id=draw_id,
            draw_centroid=draw_centroid,
        )
        if delete_mask_png or not save_mask_png:
            _delete_mask_artifacts(output_dir)
    except Exception as e:
        _log(f"  [SAM3-CHUNK] Merge error: {e}")
        return False, f"Chunk merge failed: {e}"

    # Write verbose README + bbox-discoverability alias for sam_tracks.csv
    _write_sam_run_readme(
        output_dir,
        header=(
            "SAM 3 video export (chunked divide-and-conquer)\n"
            f"source={video_file.resolve()}\n"
            f"total_frames={n_total}\n"
            f"chunk_size={chunk_size}\n"
            f"total_chunks={len(chunks)}\n"
            f"successful_chunks={successful_chunks}\n"
            f"failed_chunks={len(failed_chunks)}\n"
            f"prompt={text_prompt!r}\n"
        ),
    )
    _make_sam_bbox_tracks_alias(output_dir)

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


def _sam_frame_outputs_to_numpy(outputs: dict[str, Any]) -> dict[str, Any]:
    """Make SAM3 ``add_prompt`` / ``propagate`` outputs overlay-safe (host numpy)."""
    try:
        import torch
    except ImportError:
        tensor_type: tuple[type, ...] = ()
    else:
        tensor_type = (torch.Tensor,)

    out = dict(outputs)
    for key in ("out_binary_masks", "out_obj_ids", "out_probs", "out_boxes_xywh"):
        v = out.get(key)
        if v is None:
            continue
        if tensor_type and isinstance(v, tensor_type):
            v = v.detach().cpu().numpy()
        else:
            v = np.asarray(v)
        if key == "out_binary_masks":
            out[key] = v.astype(bool, copy=False)
        elif key == "out_obj_ids":
            out[key] = v.astype(np.int32, copy=False)
        elif key in ("out_probs", "out_boxes_xywh"):
            out[key] = v.astype(np.float32, copy=False)
    return out


def _composite_masks_bgr(
    frame_bgr: np.ndarray,
    binary_masks: np.ndarray,
    obj_ids: np.ndarray,
    alpha: float = 0.45,
    *,
    probs: np.ndarray | None = None,
    boxes_xywh: np.ndarray | None = None,
    draw_box: bool = True,
    draw_id: bool = True,
    draw_contour: bool = True,
    draw_centroid: bool = False,
    contour_thickness: int = 2,
    label_scale: float = 0.6,
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
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    # Draw overlays on top of the blended masks.
    for i in range(binary_masks.shape[0]):
        m = binary_masks[i]
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        oid = int(obj_ids[i]) if i < len(obj_ids) else i
        color = tuple(int(x) for x in colors[oid % len(colors)][::-1])  # BGR

        if draw_contour:
            mu8 = (m.astype(np.uint8)) * 255
            contours, _hier = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            if contours:
                cv2.polylines(
                    out_u8,
                    contours,
                    isClosed=True,
                    color=color,
                    thickness=int(max(1, contour_thickness)),
                    lineType=cv2.LINE_AA,
                )

        bx = by = bw = bh = None
        if boxes_xywh is not None and i < len(boxes_xywh):
            try:
                bx, by, bw, bh = (float(x) for x in boxes_xywh[i])
            except Exception:
                bx = by = bw = bh = None

        if draw_box and bx is not None and by is not None and bw is not None and bh is not None:
            x1 = int(round(bx))
            y1 = int(round(by))
            x2 = int(round(bx + bw))
            y2 = int(round(by + bh))
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(out_u8, (x1, y1), (x2, y2), color=color, thickness=2)

        if draw_centroid:
            ys, xs = np.nonzero(m)
            if xs.size:
                cx = int(round(float(xs.mean())))
                cy = int(round(float(ys.mean())))
                cv2.circle(out_u8, (cx, cy), 3, color=(255, 255, 255), thickness=-1)
                cv2.circle(out_u8, (cx, cy), 3, color=color, thickness=1)

        if draw_id:
            score = None
            if probs is not None and i < len(probs):
                with contextlib.suppress(Exception):
                    score = float(probs[i])
            label = f"#{oid}" + (f" {score:.2f}" if score is not None else "")

            if bx is None or by is None:
                ys, xs = np.nonzero(m)
                if xs.size:
                    bx0 = int(xs.min())
                    by0 = int(ys.min())
                else:
                    bx0 = 0
                    by0 = 0
            else:
                bx0 = int(round(float(bx)))
                by0 = int(round(float(by)))
            tx = max(0, min(w - 1, bx0))
            ty = max(0, min(h - 1, by0 - 6))

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, float(label_scale), 1)
            pad = 2
            x1 = max(0, tx)
            y1 = max(0, ty - th - baseline - pad * 2)
            x2 = min(w - 1, tx + tw + pad * 2)
            y2 = min(h - 1, ty + pad * 2)
            cv2.rectangle(out_u8, (x1, y1), (x2, y2), color=(0, 0, 0), thickness=-1)
            cv2.putText(
                out_u8,
                label,
                (x1 + pad, y2 - pad - baseline),
                font,
                float(label_scale),
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                out_u8,
                label,
                (x1 + pad, y2 - pad - baseline),
                font,
                float(label_scale),
                color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    return out_u8


def _mask_centroid_xy(mask: np.ndarray) -> tuple[float, float] | None:
    """Centroid of a boolean/uint8 mask in pixel coordinates, or None for empty masks."""
    ys, xs = np.nonzero(mask)
    if xs.size <= 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _delete_mask_artifacts(output_dir: Path) -> None:
    """Remove bulky per-object mask PNG outputs after CSV exports are complete."""
    masks_dir = output_dir / "masks"
    if masks_dir.exists():
        shutil.rmtree(str(masks_dir), ignore_errors=True)
    with contextlib.suppress(OSError):
        (output_dir / "sam_masks_manifest.csv").unlink(missing_ok=True)


SAM_OUTPUT_FILE_GLOSSARY = """\
Output files reference (only files that were actually written exist in this dir):

  sam_tracks.csv            Long-format **bounding-box** table (also aliased as
                            ``sam_bbox_tracks.csv``). One row per
                            (frame, obj_id). Columns:
                              frame                  Frame index (0-based)
                              obj_id                 Persistent SAM object ID
                                                     (chunked runs: global ID
                                                     after cross-chunk
                                                     tracklet linking)
                              x_px,y_px,w_px,h_px    Bounding box in pixel
                                                     space (top-left + size)
                              score                  SAM confidence (0..1)
                              area_px                Mask area in pixels
                              n_polygons             # contours in the mask
                              largest_polygon_pts    Vertex count of the
                                                     largest contour
                              cx_px,cy_px            Mask centroid (pixels)
                            This is the file the vailá pixel tool
                            (getpixelvideo.py) consumes for the
                            ``Load Tracking CSV`` button.

  sam_bbox_tracks.csv       Discoverability alias (hardlink or copy) of
                            ``sam_tracks.csv`` — same contents, ``bbox`` in
                            the name so users can spot it quickly.

  sam_frames_meta.csv       Per-frame metadata with **normalised** bbox
                            (xc, yc, w, h in [0,1]) — useful for
                            resolution-independent downstream code.

  sam_points.csv            vailá pixel-marker format (wide). One row per
                            frame, columns frame, p0_x, p0_y, p1_x, p1_y, …
                            One column-pair per obj_id; ready for direct
                            loading in getpixelvideo or rec2d. Written by
                            default (``--postprocess-points all``).

  sam_id_map.csv            Maps SAM obj_id to the column slot (p{N}) used in
                            ``sam_points.csv``.

  sam_vaila_center.csv      Simple vailá-style ``frame,x1,y1,...,xN,yN`` —
                            one (x,y) per object using the **bbox center**
                            as anchor. Ready for rec2d / getpixelvideo.
  sam_vaila_bottom.csv      Same format — **bottom-center** (foot) anchor.
  sam_vaila_top.csv         Same format — **top-center** anchor.
  sam_vaila_left.csv        Same format — **left-center** anchor.
  sam_vaila_right.csv       Same format — **right-center** anchor.

  sam_masks_manifest.csv    Index of per-frame mask PNGs (when written).

  sam_contours.json[.gz]    Polygon vertices per object, per frame
                            (schema ``vaila_sam_contours_v1``). Suitable for
                            silhouette analysis / mesh fitting.

  <video>_sam_overlay.mp4   Coloured-mask overlay video (when written).

  masks/                    Per-frame, per-object binary mask PNGs (when
                            ``--save-mask-png`` was on). Named
                            ``frame_NNNNNN_obj_K.png``.

  FAILED_sam.txt            Only present if the run failed irrecoverably;
                            contains the reason (e.g. OOM exhaustion).
"""


def _write_sam_run_readme(output_dir: Path, *, header: str) -> None:
    """Write a verbose ``README_sam.txt`` describing every produced file.

    ``header`` is the run-specific block (source, chunk size, prompt, etc.).
    The shared glossary (:data:`SAM_OUTPUT_FILE_GLOSSARY`) is appended so any
    user opening the run dir can identify the role of each CSV / JSON / MP4
    without consulting the source code.
    """
    readme = output_dir / "README_sam.txt"
    with contextlib.suppress(OSError):
        readme.write_text(
            header.rstrip() + "\n\n" + SAM_OUTPUT_FILE_GLOSSARY,
            encoding="utf-8",
        )


def _make_sam_bbox_tracks_alias(output_dir: Path) -> None:
    """Create ``sam_bbox_tracks.csv`` next to ``sam_tracks.csv``.

    Prefers a POSIX hardlink (zero disk cost, both names stay in sync); falls
    back to a regular copy when hardlink is not allowed (different filesystem
    or Windows without dev-mode). Silently no-ops when ``sam_tracks.csv`` is
    missing (e.g. ``--no-save-tracks-csv``).
    """
    src = output_dir / "sam_tracks.csv"
    dst = output_dir / "sam_bbox_tracks.csv"
    if not src.is_file():
        return
    with contextlib.suppress(OSError):
        if dst.exists() or dst.is_symlink():
            dst.unlink()
    try:
        os.link(str(src), str(dst))
        return
    except OSError:
        pass
    with contextlib.suppress(OSError):
        shutil.copy2(str(src), str(dst))


def _bbox_iou_xywh(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    return inter / union if union > 0.0 else 0.0


def _stabilize_sam_track_ids(
    output_dir: Path,
    *,
    width: int,
    height: int,
    max_gap: int = 12,
    max_centroid_dist_px: float = 180.0,
    min_iou: float = 0.05,
    direction_weight: float = 0.5,
) -> None:
    """Rewrite SAM object IDs using :class:`GeometricFrameLinker` (Hungarian + velocity)."""
    tracks_path = output_dir / "sam_tracks.csv"
    if not tracks_path.is_file() or width <= 0 or height <= 0:
        return
    with tracks_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        return
    needed = {"frame", "obj_id", "x_px", "y_px", "w_px", "h_px"}
    if not needed.issubset(fieldnames):
        return
    if "cx_px" not in fieldnames:
        fieldnames.append("cx_px")
    if "cy_px" not in fieldnames:
        fieldnames.append("cy_px")

    by_frame: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        with contextlib.suppress(Exception):
            by_frame.setdefault(int(float(row["frame"])), []).append(row)

    linker = GeometricFrameLinker(
        enabled=True,
        config=GeometricLinkerConfig(
            max_gap=max_gap,
            max_centroid_dist_px=max_centroid_dist_px,
            min_iou=min_iou,
            direction_weight=direction_weight,
        ),
        start_stable_id=0,
    )
    remapped_rows: list[dict[str, str]] = []

    for frame in sorted(by_frame):
        detections = by_frame[frame]
        valid_rows: list[dict[str, str]] = []
        frame_dets: list[dict[str, Any]] = []
        for row in detections:
            try:
                x = float(row["x_px"])
                y = float(row["y_px"])
                w_box = float(row["w_px"])
                h_box = float(row["h_px"])
                old_oid = int(float(row["obj_id"]))
            except Exception:
                continue
            valid_rows.append(row)
            frame_dets.append(
                {
                    "raw_id": old_oid,
                    "tracker_id": old_oid,
                    "xyxy": (x, y, x + w_box, y + h_box),
                }
            )
        linked = linker.assign_frame(frame, frame_dets)
        for row, det in zip(valid_rows, linked, strict=True):
            x_min, y_min, x_max, y_max = det["xyxy"]
            cx = (float(x_min) + float(x_max)) * 0.5
            cy = (float(y_min) + float(y_max)) * 0.5
            row["obj_id"] = str(int(det["stable_id"]))
            row["cx_px"] = f"{cx:.3f}"
            row["cy_px"] = f"{cy:.3f}"
            remapped_rows.append(row)

    remapped_rows.sort(key=lambda r: (int(float(r["frame"])), int(float(r["obj_id"]))))
    with tracks_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(remapped_rows)
    write_reid_links_csv(
        str(output_dir / "sam_reid_links.csv"),
        linker.reid_links,
        ("frame", "old_obj_id", "obj_id"),
    )

    meta_frames: dict[int, dict[int, tuple[float, float, float, float, float]]] = {}
    stable_ids: set[int] = set()
    for row in remapped_rows:
        try:
            frame = int(float(row["frame"]))
            oid = int(float(row["obj_id"]))
            x = float(row["x_px"]) / float(width)
            y = float(row["y_px"]) / float(height)
            w_box = float(row["w_px"]) / float(width)
            h_box = float(row["h_px"]) / float(height)
            score = float(row.get("score") or "nan")
        except Exception:
            continue
        stable_ids.add(oid)
        meta_frames.setdefault(frame, {})[oid] = (x, y, w_box, h_box, score)

    existing_meta = output_dir / "sam_frames_meta.csv"
    frame_numbers = sorted(meta_frames)
    if existing_meta.is_file():
        with contextlib.suppress(Exception), existing_meta.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                frame_numbers.append(int(float(row["frame"])))
    frame_numbers = sorted(set(frame_numbers))
    ids_sorted = sorted(stable_ids)
    header = ["frame"]
    for oid in ids_sorted:
        header.extend(
            [f"box_x_{oid}", f"box_y_{oid}", f"box_w_{oid}", f"box_h_{oid}", f"prob_{oid}"]
        )
    lines = [",".join(header)]
    for frame in frame_numbers:
        parts = [str(frame)]
        frame_map = meta_frames.get(frame, {})
        for oid in ids_sorted:
            vals = frame_map.get(oid)
            if vals is None:
                parts.extend(["", "", "", "", ""])
            else:
                x, y, w_box, h_box, score = vals
                parts.extend(
                    [f"{x:.6f}", f"{y:.6f}", f"{w_box:.6f}", f"{h_box:.6f}", f"{score:.6f}"]
                )
        lines.append(",".join(parts))
    existing_meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[SAM3-ReID] geometric ID stabilization: {len(stable_ids)} track(s) -> sam_reid_links.csv"
    )


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
    overlay_rich: bool = True,
    draw_contour: bool = True,
    draw_box: bool = True,
    draw_id: bool = True,
    draw_centroid: bool = False,
    save_contours: bool = True,
    save_tracks_csv: bool = True,
    delete_mask_png: bool = False,
    stabilize_ids: bool = False,
    contours_format: str = "json",
    contours_gzip: bool = False,
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
    contours_format = (contours_format or "json").strip().lower()
    if contours_format not in ("json", "jsonl"):
        contours_format = "json"
    temp_spatial: Path | None = None
    mf = _read_max_input_frames(max_input_frames)
    session_path, temp_clip, n_sess, sess_to_orig = _maybe_subsample_video_for_vram(
        video_path, output_dir, mf
    )
    le_cap = _read_max_input_long_edge(max_input_long_edge)
    if max_input_long_edge is None and le_cap == 1280:
        _vprof = _sam3_vram_profile()
        if _vprof is not None:
            print(
                f"[SAM3] Default max_input_long_edge=1280 (GPU total {_vprof['total_gib']:.1f} GiB; "
                "set SAM3_MAX_INPUT_LONG_EDGE=1920 to try full HD input if you have VRAM headroom)"
            )
    session_path, temp_spatial, w_sess, h_sess, n_sess = _maybe_downscale_video_long_edge(
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
                outputs_by_frame[fi0] = _sam_frame_outputs_to_numpy(prompt_resp["outputs"])

                stream_req = {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both",
                    "start_frame_index": fi_run,
                    "max_frame_num_to_track": None,
                }
                with _autocast:
                    for chunk in predictor.handle_stream_request(stream_req):
                        raw_out = chunk.get("outputs")
                        if raw_out is None:
                            continue
                        outputs_by_frame[int(chunk["frame_index"])] = _sam_frame_outputs_to_numpy(
                            raw_out
                        )
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

        cap = cv2.VideoCapture(vp_orig)
        if not cap.isOpened():
            raise OSError(f"Could not open video: {vp_orig}")
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

        contours_frames: list[dict[str, Any]] = []
        tracks_rows: list[str] = []
        mask_manifest_rows: list[str] = ["frame,obj_id,area_px,mask_png"]

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

        # Write overlay at the ORIGINAL FPS and frame count.
        #
        # If we subsampled for VRAM, SAM3 only produced masks for the session clip frames.
        # For the final overlay, we repeat the closest available session mask between those
        # sampled frames so the output MP4 matches the original timeline.
        cap = cv2.VideoCapture(vp_orig) if writer is not None else None
        nframes_to_write = int(nframes)
        if cap is not None:
            nframes_to_write = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(nframes)

        sess_to_orig = np.asarray(sess_to_orig, dtype=np.int64)
        if sess_to_orig.size <= 0:
            sess_to_orig = np.arange(max(0, nframes_to_write), dtype=np.int64)

        n_unique_masks = len(outputs_by_frame)
        if sess_to_orig.size < nframes_to_write:
            print(
                f"[SAM3] Overlay: {nframes_to_write} original frames, "
                f"{n_unique_masks} unique mask keyframes (session={sess_to_orig.size} frames). "
                f"Each output frame uses temporally nearest SAM keyframe (VRAM subsample alignment)."
            )
        else:
            print(
                f"[SAM3] Overlay: {nframes_to_write} frames, "
                f"{n_unique_masks} frames with mask outputs."
            )
        export_label = (
            "rendering overlay + exporting CSV/JSON" if writer is not None else "exporting CSV/JSON"
        )
        print(f"[SAM3] {export_label} ({nframes_to_write} frames at {w}x{h})...")
        sys.stdout.flush()

        _overlay_log_interval = max(1, nframes_to_write // 20)  # ~5% increments
        last_sess_idx: int | None = None
        last_masks: np.ndarray | None = None
        last_oids: np.ndarray | None = None
        last_probs: np.ndarray | None = None
        last_boxes = None
        last_boxes_px: np.ndarray | None = None

        frame_idx = 0
        while frame_idx < nframes_to_write:
            bgr = None
            if cap is not None:
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    frame_idx += 1
                    continue

            if frame_idx % _overlay_log_interval == 0 or frame_idx == nframes_to_write - 1:
                pct = 100.0 * (frame_idx + 1) / nframes_to_write
                print(
                    f"  export: frame {frame_idx + 1}/{nframes_to_write} ({pct:.0f}%)", flush=True
                )

            sess_idx = _nearest_sess_idx_for_orig_frame(frame_idx, sess_to_orig)

            if last_sess_idx != sess_idx:
                out = outputs_by_frame.get(sess_idx)
                if out is None:
                    last_masks = last_oids = last_probs = None
                    last_boxes = None
                    last_boxes_px = None
                else:
                    last_masks = out.get("out_binary_masks")
                    last_oids = out.get("out_obj_ids")
                    last_probs = out.get("out_probs")
                    last_boxes = out.get("out_boxes_xywh")

                    if (
                        last_masks is not None
                        and last_oids is not None
                        and getattr(last_masks, "size", 0) != 0
                        and (int(w_sess), int(h_sess)) != (int(w), int(h))
                    ):
                        resized: list[np.ndarray] = []
                        for i in range(last_masks.shape[0]):
                            m = last_masks[i].astype(np.uint8)
                            m2 = cv2.resize(m, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
                            resized.append(m2.astype(bool))
                        last_masks = np.stack(resized, axis=0) if resized else last_masks

                    last_boxes_px = None
                    if last_boxes is not None and len(last_boxes):
                        with contextlib.suppress(Exception):
                            b = np.asarray(last_boxes, dtype=np.float32)
                            last_boxes_px = b.copy()
                            last_boxes_px[:, 0] *= float(w)
                            last_boxes_px[:, 2] *= float(w)
                            last_boxes_px[:, 1] *= float(h)
                            last_boxes_px[:, 3] *= float(h)

                last_sess_idx = sess_idx

            masks = last_masks
            oids = last_oids
            probs = last_probs
            boxes = last_boxes
            boxes_px = last_boxes_px
            if masks is None or oids is None or getattr(masks, "size", 0) == 0:
                if writer is not None and bgr is not None:
                    writer.write(bgr)
                frame_idx += 1
                continue

            if writer is not None and bgr is not None:
                if overlay_rich:
                    comp = _composite_masks_bgr(
                        bgr,
                        masks,
                        oids,
                        probs=probs,
                        boxes_xywh=boxes_px,
                        draw_box=draw_box,
                        draw_id=draw_id,
                        draw_contour=draw_contour,
                        draw_centroid=draw_centroid,
                    )
                else:
                    comp = _composite_masks_bgr(bgr, masks, oids)
            if writer is not None:
                writer.write(comp)

            objects_for_json: list[dict[str, Any]] = []
            frame_data = {}
            for i in range(masks.shape[0]):
                oid = int(oids[i])
                if save_mask_png and int(sess_to_orig[sess_idx]) == int(frame_idx):
                    png = masks_dir / f"frame_{frame_idx:06d}_obj_{oid}.png"
                    cv2.imwrite(str(png), (masks[i].astype(np.uint8)) * 255)
                pr = float(probs[i]) if probs is not None and i < len(probs) else float("nan")
                bx = by = bw = bh = float("nan")
                if boxes is not None and i < len(boxes):
                    bx, by, bw, bh = (float(x) for x in boxes[i])
                frame_data[oid] = (bx, by, bw, bh, pr)

                # Rich exports (pixels, polygons) for 3D reconstruction / SAM-3D-Body.
                if save_contours or save_tracks_csv:
                    m = masks[i]
                    area_px = int(m.sum())
                    centroid = _mask_centroid_xy(m)
                    cx_px = cy_px = float("nan")
                    if centroid is not None:
                        cx_px, cy_px = centroid
                    polys: list[list[list[int]]] = []
                    if save_contours and draw_contour:
                        mu8 = (m.astype(np.uint8)) * 255
                        contours, _hier = cv2.findContours(
                            mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
                        )
                        for c in contours:
                            pts = c.reshape(-1, 2)
                            if pts.shape[0] < 3:
                                continue
                            polys.append([[int(x), int(y)] for x, y in pts])

                    if boxes_px is not None and i < len(boxes_px):
                        bx_px, by_px, bw_px, bh_px = (float(x) for x in boxes_px[i])
                    else:
                        bx_px = by_px = bw_px = bh_px = float("nan")

                    mask_rel = (
                        f"masks/frame_{frame_idx:06d}_obj_{oid}.png"
                        if save_mask_png
                        and not delete_mask_png
                        and int(sess_to_orig[sess_idx]) == int(frame_idx)
                        else ""
                    )
                    if mask_rel:
                        mask_manifest_rows.append(f"{frame_idx},{oid},{area_px},{mask_rel}")
                    if save_tracks_csv:
                        n_polys = len(polys) if polys else 0
                        largest_pts = max((len(p) for p in polys), default=0)
                        tracks_rows.append(
                            f"{frame_idx},{oid},{bx_px:.3f},{by_px:.3f},{bw_px:.3f},{bh_px:.3f},"
                            f"{pr:.6f},{area_px},{n_polys},{largest_pts},{cx_px:.3f},{cy_px:.3f}"
                        )
                    if save_contours:
                        objects_for_json.append(
                            {
                                "obj_id": oid,
                                "score": None if not np.isfinite(pr) else float(pr),
                                "bbox_xywh_px": [
                                    None if not np.isfinite(bx_px) else int(round(bx_px)),
                                    None if not np.isfinite(by_px) else int(round(by_px)),
                                    None if not np.isfinite(bw_px) else int(round(bw_px)),
                                    None if not np.isfinite(bh_px) else int(round(bh_px)),
                                ],
                                "area_px": area_px,
                                "mask_png": mask_rel or None,
                                "polygons": polys,
                            }
                        )

            if save_contours:
                contours_frames.append(
                    {
                        "frame": int(frame_idx),
                        "session_frame": int(sess_idx),
                        "objects": objects_for_json,
                    }
                )

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

            frame_idx += 1

        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

        meta_path = output_dir / "sam_frames_meta.csv"
        meta_path.write_text(
            ",".join(header_cols) + "\n" + "\n".join(meta_rows_wide) + "\n",
            encoding="utf-8",
        )

        if save_tracks_csv:
            tracks_path = output_dir / "sam_tracks.csv"
            tracks_path.write_text(
                "frame,obj_id,x_px,y_px,w_px,h_px,score,area_px,n_polygons,largest_polygon_pts,cx_px,cy_px\n"
                + "\n".join(tracks_rows)
                + ("\n" if tracks_rows else ""),
                encoding="utf-8",
            )

        if save_mask_png:
            manifest_path = output_dir / "sam_masks_manifest.csv"
            manifest_path.write_text(
                "\n".join(mask_manifest_rows) + "\n",
                encoding="utf-8",
            )

        if stabilize_ids:
            _stabilize_sam_track_ids(output_dir, width=w, height=h)

        if delete_mask_png:
            _delete_mask_artifacts(output_dir)

        if save_contours:
            payload = {
                "schema": "vaila_sam_contours_v1",
                "video": video_path.name,
                "width": int(w),
                "height": int(h),
                "fps": float(fps),
                "n_frames": int(nframes_to_write),
                "object_ids": [int(x) for x in sorted_oids],
                "frames": contours_frames,
            }
            if contours_format == "jsonl":
                out_path = output_dir / "sam_contours.jsonl"
                lines = []
                header = dict(payload)
                header.pop("frames", None)
                lines.append(json.dumps(header, separators=(",", ":"), ensure_ascii=False))
                for fr in contours_frames:
                    lines.append(json.dumps(fr, separators=(",", ":"), ensure_ascii=False))
                text = "\n".join(lines) + "\n"
                if contours_gzip:
                    import gzip

                    gz_path = output_dir / "sam_contours.jsonl.gz"
                    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
                        fh.write(text)
                else:
                    out_path.write_text(text, encoding="utf-8")
            else:
                out_path = output_dir / "sam_contours.json"
                text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
                if contours_gzip:
                    import gzip

                    gz_path = output_dir / "sam_contours.json.gz"
                    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
                        fh.write(text)
                else:
                    out_path.write_text(text, encoding="utf-8")

        ckpt_note = (
            str(ckpt_file) if ckpt_file is not None else "Hugging Face facebook/sam3 (cached)"
        )
        _write_sam_run_readme(
            output_dir,
            header=(
                "SAM 3 video export\n"
                f"source_original={vp_orig}\n"
                f"session_resource={vp}\n"
                f"subsampled_to_disk={temp_clip is not None} "
                f"spatial_downscale={temp_spatial is not None} "
                f"max_input_long_edge_cap={le_cap} "
                f"max_input_frames_cap={mf} session_frames={n_sess}\n"
                f"checkpoint={ckpt_note}\n"
                f"prompt={text_prompt!r}\n"
                f"prompt_frame_requested={frame_index} "
                f"prompt_frame_used={fi_run}\n"
                f"frames_with_outputs={len(outputs_by_frame)} / {nframes}\n"
            ),
        )
        _make_sam_bbox_tracks_alias(output_dir)

        # Summary for the user.
        _written_files = [
            f
            for f in (
                overlay_path if save_overlay_mp4 and writer is not None else None,
                output_dir / "sam_frames_meta.csv",
                output_dir / "sam_tracks.csv" if save_tracks_csv else None,
                output_dir / "sam_contours.json" if save_contours else None,
            )
            if f is not None and Path(f).is_file()
        ]
        print(f"[SAM3] ✓ Done — {len(_written_files)} output files in {output_dir}")
        for f in _written_files:
            print(f"  • {Path(f).name} ({Path(f).stat().st_size / 1024:.0f} KB)")
        sys.stdout.flush()
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
        with (Path(__file__).resolve().parents[1] / ".cursor" / "debug-42b4a5.log").open(
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
    overlay_rich: bool = True,
    draw_contour: bool = True,
    draw_box: bool = True,
    draw_id: bool = True,
    draw_centroid: bool = False,
    save_contours: bool = True,
    save_tracks_csv: bool = True,
    delete_mask_png: bool = False,
    stabilize_ids: bool = False,
    contours_format: str = "json",
    contours_gzip: bool = False,
    no_chunked_fallback: bool = False,
    chunk_size: int | None = None,
    overlap_frames: int = 2,
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
                overlay_rich=overlay_rich,
                draw_contour=draw_contour,
                draw_box=draw_box,
                draw_id=draw_id,
                draw_centroid=draw_centroid,
                save_contours=save_contours,
                save_tracks_csv=save_tracks_csv,
                delete_mask_png=delete_mask_png,
                stabilize_ids=stabilize_ids,
                contours_format=contours_format,
                contours_gzip=contours_gzip,
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
                overlay_rich=overlay_rich,
                draw_contour=draw_contour,
                draw_box=draw_box,
                draw_id=draw_id,
                draw_centroid=draw_centroid,
                save_contours=save_contours,
                save_tracks_csv=save_tracks_csv,
                delete_mask_png=delete_mask_png,
                stabilize_ids=stabilize_ids,
                contours_format=contours_format,
                contours_gzip=contours_gzip,
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
    #
    # Recursion guard (debug session 2026-04-30, local Videos test):
    # When ``_process_video_chunked`` spawns a chunk subprocess, it passes
    # ``--no-chunked-fallback`` so the child never re-enters this branch.  Without
    # this guard a single OOMing chunk produced 10+ levels of
    # ``_chunks/out_000000/_chunks/...`` because ``safe_frames`` is clamped to
    # ``max(16, …)`` — chunking a 16-frame video yields exactly one chunk equal
    # to the input, OOMs, and recurses forever.
    if no_chunked_fallback:
        _log(
            f"  [SAM3] All OOM retries exhausted for {video_file.name} "
            "(--no-chunked-fallback set; refusing to recurse into another chunked split)"
        )
        _write_failure_marker(
            output_dir,
            video_file,
            last_err or "All OOM retries exhausted (chunked fallback disabled)",
        )
        return False, last_err or "All OOM retries exhausted (chunked fallback disabled)"
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
        overlay_rich=overlay_rich,
        draw_contour=draw_contour,
        draw_box=draw_box,
        draw_id=draw_id,
        draw_centroid=draw_centroid,
        save_contours=save_contours,
        save_tracks_csv=save_tracks_csv,
        delete_mask_png=delete_mask_png,
        stabilize_ids=stabilize_ids,
        contours_format=contours_format,
        contours_gzip=contours_gzip,
        chunk_size=chunk_size,
        overlap_frames=overlap_frames,
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
        self.result: (
            tuple[
                Path,  # input (dir or file)
                Path,  # output parent
                Path | None,  # checkpoint
                str,  # prompt
                int,  # prompt frame
                int | None,  # max_frames
                int | None,  # max_input_long_edge
                str,  # postprocess_points
                bool,  # save_overlay
                bool,  # save_png
                bool,  # overlay_rich
                bool,  # draw_contour
                bool,  # draw_box
                bool,  # draw_id
                bool,  # draw_centroid
                bool,  # save_contours
                bool,  # save_tracks_csv
                str,  # contours_format
                bool,  # contours_gzip
                bool,  # stabilize_ids
                bool,  # frame_fallback
                bool,  # dry_run
            ]
            | None
        ) = None

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
                "VRAM: optional Max frames — 0 = every frame to SAM (best mask sync); "
                "empty = auto from free VRAM (see log line [SAM3 VRAM])."
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
        ttk.Label(
            frm,
            text="Max frames (0=full clip; empty=auto from VRAM):",
        ).grid(row=6, column=0, sticky="w", pady=4)
        self.max_frames_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.max_frames_var, width=12).grid(
            row=6, column=1, sticky="w", pady=4
        )
        ttk.Label(frm, text="Max input long edge (px):").grid(row=7, column=0, sticky="w", pady=4)
        self.max_long_edge_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.max_long_edge_var, width=12).grid(
            row=7, column=1, sticky="w", pady=4
        )
        ttk.Label(frm, text="Post-process points:").grid(row=8, column=0, sticky="w", pady=4)
        self.postprocess_var = tk.StringVar(value="all")
        ttk.Combobox(
            frm,
            textvariable=self.postprocess_var,
            values=("none", "foot", "center", "mask", "all"),
            width=14,
            state="readonly",
        ).grid(row=8, column=1, sticky="w", pady=4)

        self.overlay_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=False)
        self.fallback_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm,
            text="Save overlay MP4 (video with colored masks on top of frames)",
            variable=self.overlay_var,
        ).grid(row=9, column=1, sticky="w")
        ttk.Checkbutton(frm, text="Save mask PNGs (per object)", variable=self.png_var).grid(
            row=10, column=1, sticky="w"
        )

        out_opts = ttk.LabelFrame(frm, text="Overlay & Output (rich)", padding=6)
        out_opts.grid(row=11, column=1, columnspan=2, sticky="ew", pady=(6, 0))
        self.overlay_rich_var = tk.BooleanVar(value=True)
        self.draw_contour_var = tk.BooleanVar(value=True)
        self.draw_box_var = tk.BooleanVar(value=True)
        self.draw_id_var = tk.BooleanVar(value=True)
        self.draw_centroid_var = tk.BooleanVar(value=False)
        self.save_contours_var = tk.BooleanVar(value=True)
        self.save_tracks_csv_var = tk.BooleanVar(value=True)
        self.contours_format_var = tk.StringVar(value="json")
        self.contours_gzip_var = tk.BooleanVar(value=False)
        self.stabilize_ids_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            out_opts,
            text="Rich overlay (bbox/ID/score/contours)",
            variable=self.overlay_rich_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(out_opts, text="Draw contours", variable=self.draw_contour_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Checkbutton(out_opts, text="Draw boxes", variable=self.draw_box_var).grid(
            row=1, column=1, sticky="w"
        )
        ttk.Checkbutton(out_opts, text="Draw IDs", variable=self.draw_id_var).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Checkbutton(out_opts, text="Draw centroid", variable=self.draw_centroid_var).grid(
            row=2, column=1, sticky="w"
        )
        ttk.Checkbutton(
            out_opts, text="Save sam_contours.json", variable=self.save_contours_var
        ).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(
            out_opts, text="Save sam_tracks.csv", variable=self.save_tracks_csv_var
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(out_opts, text="Contours format:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        ttk.Combobox(
            out_opts,
            textvariable=self.contours_format_var,
            values=("json", "jsonl"),
            width=10,
            state="readonly",
        ).grid(row=4, column=1, sticky="w", pady=(4, 0))
        ttk.Checkbutton(out_opts, text="Gzip contours", variable=self.contours_gzip_var).grid(
            row=5, column=0, sticky="w"
        )
        ttk.Checkbutton(
            out_opts,
            text="ReID/Stabilize SAM IDs + final CSVs",
            variable=self.stabilize_ids_var,
        ).grid(row=5, column=1, sticky="w")

        ttk.Checkbutton(
            frm,
            text="Fallback: Frame-by-Frame (CUDA only; lower VRAM, slower, no temporal tracking)",
            variable=self.fallback_var,
        ).grid(row=12, column=1, sticky="w")
        self.dry_run_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frm,
            text="Dry-run / smoke (show plan only, do not run SAM3)",
            variable=self.dry_run_var,
        ).grid(row=13, column=1, sticky="w")

        btns = ttk.Frame(frm)
        btns.grid(row=14, column=0, columnspan=3, pady=12)
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
        max_frames: int | None = None
        max_frames_txt = self.max_frames_var.get().strip()
        if max_frames_txt:
            try:
                max_frames = int(max_frames_txt)
            except ValueError:
                messagebox.showerror("Error", "Max frames must be an integer.", parent=self)
                return
            if max_frames < 0:
                messagebox.showerror("Error", "Max frames must be >= 0.", parent=self)
                return
        max_long_edge: int | None = None
        max_long_edge_txt = self.max_long_edge_var.get().strip()
        if max_long_edge_txt:
            try:
                max_long_edge = int(max_long_edge_txt)
            except ValueError:
                messagebox.showerror(
                    "Error", "Max input long edge must be an integer.", parent=self
                )
                return
            if max_long_edge < 0:
                messagebox.showerror("Error", "Max input long edge must be >= 0.", parent=self)
                return
        ck = self.ckpt_var.get().strip()
        ckpt_path: Path | None = Path(ck) if ck else None
        stabilize_ids = self.stabilize_ids_var.get()
        save_tracks_csv = self.save_tracks_csv_var.get()
        postprocess_points = self.postprocess_var.get().strip() or "all"
        save_png = self.png_var.get()
        if stabilize_ids:
            save_tracks_csv = True
        self.result = (
            inp,
            Path(o),
            ckpt_path,
            self.prompt_var.get().strip() or "person",
            fi,
            max_frames,
            max_long_edge,
            postprocess_points,
            self.overlay_var.get(),
            save_png,
            self.overlay_rich_var.get(),
            self.draw_contour_var.get(),
            self.draw_box_var.get(),
            self.draw_id_var.get(),
            self.draw_centroid_var.get(),
            self.save_contours_var.get(),
            save_tracks_csv,
            self.contours_format_var.get().strip() or "json",
            self.contours_gzip_var.get(),
            stabilize_ids,
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

        calib_frame = ttk.LabelFrame(frm, text="Field calibration (after batch finishes)")
        calib_frame.pack(fill=tk.X, pady=(6, 0))
        self.calib_btn = ttk.Button(
            calib_frame,
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
        clean = line.rstrip()
        if clean:
            sys.stdout.write(clean + "\n")
            sys.stdout.flush()
        self.log_text.config(state="normal")
        self.log_text.insert("end", clean + "\n")
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
                self.calib_btn.config(state="normal")

        self.after(0, _done)

    def set_output_base(self, output_base: Path) -> None:
        self._output_base = output_base

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
    overlay_rich: bool,
    draw_contour: bool,
    draw_box: bool,
    draw_id: bool,
    draw_centroid: bool,
    save_contours: bool,
    save_tracks_csv: bool,
    contours_format: str,
    contours_gzip: bool,
    stabilize_ids: bool,
    frame_fallback: bool,
    max_frames: int | None,
    max_input_long_edge: int | None,
    postprocess_points: str,
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
            max_input_frames=max_frames,
            max_input_long_edge=max_input_long_edge,
            save_overlay_mp4=save_ov,
            save_mask_png=save_png,
            frame_by_frame_fallback=frame_fallback,
            overlay_rich=overlay_rich,
            draw_contour=draw_contour,
            draw_box=draw_box,
            draw_id=draw_id,
            draw_centroid=draw_centroid,
            save_contours=save_contours,
            save_tracks_csv=save_tracks_csv,
            stabilize_ids=stabilize_ids,
            contours_format=contours_format,
            contours_gzip=contours_gzip,
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

    if postprocess_points and postprocess_points != "none" and succeeded > 0:
        try:
            from vaila.sam_postprocess import (
                extract_points_for_batch,
                write_vaila_anchor_csvs_for_batch,
            )

            log(f"[postprocess] mode={postprocess_points}")
            outs = extract_points_for_batch(output_base, mode=postprocess_points)
            vaila_outs = write_vaila_anchor_csvs_for_batch(output_base)
            log(
                f"[postprocess] wrote {len(outs)} sam_points.csv + "
                f"{len(vaila_outs)} vailá anchor CSV(s)."
            )
        except Exception as exc:
            failed.append(f"postprocess: {exc}")
            log(f"[postprocess] FAILED: {exc}")

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
    overlay_rich: bool,
    draw_contour: bool,
    draw_box: bool,
    draw_id: bool,
    draw_centroid: bool,
    save_contours: bool,
    save_tracks_csv: bool,
    contours_format: str,
    contours_gzip: bool,
    stabilize_ids: bool,
    frame_fallback: bool,
    max_frames: int | None,
    max_input_long_edge: int | None,
    postprocess_points: str,
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
        "--output-base",
        str(output_base.resolve()),
        "-t",
        prompt,
        "-f",
        str(frame_idx),
    ]
    if ckpt_opt is not None:
        cmd += ["-w", str(ckpt_opt.resolve())]
    if max_frames is not None:
        cmd += ["--max-frames", str(max_frames)]
    if max_input_long_edge is not None:
        cmd += ["--max-input-long-edge", str(max_input_long_edge)]
    if postprocess_points and postprocess_points != "none":
        cmd += ["--postprocess-points", postprocess_points]
    if not save_ov:
        cmd.append("--no-overlay")
    if not save_png:
        cmd.append("--no-png")
    else:
        cmd.append("--keep-mask-png")
    if frame_fallback:
        cmd.append("--frame-by-frame")
    if not overlay_rich:
        cmd.append("--no-overlay-rich")
    if not draw_contour:
        cmd.append("--no-draw-contour")
    if not draw_box:
        cmd.append("--no-draw-box")
    if not draw_id:
        cmd.append("--no-draw-id")
    if draw_centroid:
        cmd.append("--draw-centroid")
    if not save_contours:
        cmd.append("--no-save-contours")
    if not save_tracks_csv:
        cmd.append("--no-save-tracks-csv")
    if contours_format:
        cmd += ["--contours-format", str(contours_format)]
    if contours_gzip:
        cmd.append("--contours-gzip")
    if stabilize_ids:
        cmd.append("--stabilize-ids")

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
        diagnosis = _format_subprocess_exit_diagnosis(rc)
        with contextlib.suppress(tk.TclError):
            progress._append_log(f"[GUI] subprocess exited with code {rc}")
            if rc != 0:
                # Wrap long advice lines for readability inside the Tk log widget.
                for ln in diagnosis.splitlines():
                    progress._append_log(f"[GUI] {ln}")
                progress._append_log("[GUI] Full subprocess stdout/stderr captured at: " + log_path)
        if rc != 0:
            failed_list = state["failed"]
            if isinstance(failed_list, list) and not failed_list:
                failed_list.append(diagnosis)
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
        max_frames,
        max_input_long_edge,
        postprocess_points,
        save_ov,
        save_png,
        overlay_rich,
        draw_contour,
        draw_box,
        draw_id,
        draw_centroid,
        save_contours,
        save_tracks_csv,
        contours_format,
        contours_gzip,
        stabilize_ids,
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

    if dry_run:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = out_parent / f"processed_sam_{ts}"
        output_base.mkdir(parents=True, exist_ok=True)
        report = _sam3_dry_run_report(
            input_path,
            output_base,
            ckpt_opt=ckpt_opt,
            text_prompt=prompt,
            frame_index=frame_idx,
            max_input_frames=max_frames,
            max_input_long_edge=max_input_long_edge,
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

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = out_parent / f"processed_sam_{ts}"
    output_base.mkdir(parents=True, exist_ok=True)

    progress = SamBatchProgress(root, total=len(video_files), output_base=output_base)

    def _on_done(succeeded: int, total: int, failed: list[str], out_base: Path) -> None:
        summary = f"Processed {succeeded}/{total} video(s).\nOutput: {out_base}"
        print("\nSAM 3 GUI batch finished")
        print(summary)
        if failed:
            print(f"Failed ({len(failed)}):")
            for item in failed[:20]:
                print(f"  {item}")
            if len(failed) > 20:
                print(f"  ...and {len(failed) - 20} more")
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
        overlay_rich=bool(overlay_rich),
        draw_contour=bool(draw_contour),
        draw_box=bool(draw_box),
        draw_id=bool(draw_id),
        draw_centroid=bool(draw_centroid),
        save_contours=bool(save_contours),
        save_tracks_csv=bool(save_tracks_csv),
        contours_format=str(contours_format),
        contours_gzip=bool(contours_gzip),
        stabilize_ids=bool(stabilize_ids),
        frame_fallback=frame_fallback,
        max_frames=max_frames,
        max_input_long_edge=max_input_long_edge,
        postprocess_points=postprocess_points,
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

    parser = argparse.ArgumentParser(
        description=(
            "SAM 3 video segmentation (vailá). Pass --print-examples for a "
            "copy-paste recipe sheet, --open-help for the full HTML reference."
        ),
        epilog=SAM3_CLI_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input video file OR directory containing videos (batch)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output base directory")
    parser.add_argument(
        "--output-base",
        type=Path,
        help=argparse.SUPPRESS,  # internal: GUI passes an exact processed_sam_TS dir
    )
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
        "--tracks-only",
        action="store_true",
        help="Fast export profile: skip overlay MP4, mask PNGs and contours; write bbox/centroid CSV only.",
    )
    parser.add_argument(
        "--delete-mask-png",
        action="store_true",
        help="Delete masks/ and sam_masks_manifest.csv after CSV/JSON exports finish (this is now the default).",
    )
    parser.add_argument(
        "--keep-mask-png",
        action="store_true",
        help="Keep masks/ and sam_masks_manifest.csv after exports finish (by default they are deleted).",
    )
    parser.add_argument(
        "--keep-masks",
        action="store_true",
        help="Alias for --keep-mask-png.",
    )
    parser.add_argument(
        "--stabilize-ids",
        action="store_true",
        help="Rewrite SAM IDs with geometric continuity (IoU/centroid) after export; helps chunk ID resets.",
    )
    parser.add_argument(
        "--overlay-rich",
        dest="overlay_rich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay enrichment: draw bbox/ID/score/contours on top of colored masks (default: on).",
    )
    parser.add_argument(
        "--draw-contour",
        dest="draw_contour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw mask contours (default: on).",
    )
    parser.add_argument(
        "--draw-box",
        dest="draw_box",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw bounding boxes (default: on).",
    )
    parser.add_argument(
        "--draw-id",
        dest="draw_id",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw object ID and score label (default: on).",
    )
    parser.add_argument(
        "--draw-centroid",
        action="store_true",
        help="Draw mask centroid on the overlay (default: off).",
    )
    parser.add_argument(
        "--save-contours",
        dest="save_contours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write sam_contours.json with polygons per frame/object (default: on).",
    )
    parser.add_argument(
        "--save-tracks-csv",
        dest="save_tracks_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write sam_tracks.csv (long format, bbox in pixels + area + polygons stats) (default: on).",
    )
    parser.add_argument(
        "--contours-format",
        choices=["json", "jsonl"],
        default="json",
        help="Contours file format (default: json).",
    )
    parser.add_argument(
        "--contours-gzip",
        action="store_true",
        help="Gzip sam_contours output (json.gz or jsonl.gz).",
    )
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
        "--chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Chunk size for divide-and-conquer fallback. Larger values reduce model reloads but can OOM.",
    )
    parser.add_argument(
        "--overlap-frames",
        type=int,
        default=2,
        metavar="N",
        help="Shared frames between adjacent SAM chunks for cross-chunk Re-ID (default: 2).",
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
        "--preflight",
        action="store_true",
        help="Scan input file/dir and write SAM3_PREFLIGHT.csv (resolution/fps/frames/duration) "
        "plus suggested caps. Does not run SAM3.",
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
        "--print-examples",
        action="store_true",
        help="Print copy-paste CLI recipes (single video, batch, OOM tips, FIFA) and exit.",
    )
    parser.add_argument(
        "--postprocess-points",
        choices=["none", "foot", "center", "mask", "all"],
        default="all",
        help="After the batch finishes, build vailá-format pixel CSVs (sam_points.csv + "
        "five sam_vaila_*.csv anchor files) per video subdirectory. "
        "'foot' = bottom-center of bbox (best for soccer-field homography rec2d); "
        "'center' = bbox center; 'mask' = real centroid of the mask PNG; "
        "'all' = canonical foot pair plus extra cx/cy/mx/my columns. "
        "'none' = skip post-processing entirely. Default: all.",
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
    parser.add_argument(
        "--no-chunked-fallback",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: prevents recursive chunking when
        # this process is itself a chunk spawned by _process_video_chunked
    )
    args = parser.parse_args()

    keep_mask_png = bool(args.keep_mask_png) or bool(args.keep_masks)
    if args.delete_mask_png:
        keep_mask_png = False
    delete_mask_png = not keep_mask_png

    if args.tracks_only:
        args.no_overlay = True
        args.no_png = True
        args.overlay_rich = False
        args.draw_contour = False
        args.save_contours = False
        args.save_tracks_csv = True
    if args.stabilize_ids:
        if not args.save_tracks_csv:
            print(
                "[SAM3-ReID] --stabilize-ids requires sam_tracks.csv; enabling --save-tracks-csv."
            )
            args.save_tracks_csv = True
        if args.postprocess_points == "none":
            args.postprocess_points = "all"

    if args.open_help:
        open_sam3_install_help_in_browser()
        return

    if args.print_examples:
        _print_sam3_cli_examples()
        return

    if args.download_weights:
        out_ckpt = download_sam3_weights_to_vaila_models()
        print(f"SAM3 weights ready: {out_ckpt}")
        return

    if args.preflight:
        if args.input is None:
            print("--preflight requires --input")
            raise SystemExit(2)
        output_base = (args.output or Path.cwd() / "sam_preflight").resolve()
        _sam3_preflight_scan(
            args.input,
            output_base=output_base,
            max_input_frames=args.max_frames,
            max_input_long_edge=args.max_input_long_edge,
        )
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
        print(_format_runtime_banner(args, [single]), flush=True)
        _warn_host_ram_for_videos([single], max_input_frames=args.max_frames)
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
            overlay_rich=bool(args.overlay_rich),
            draw_contour=bool(args.draw_contour),
            draw_box=bool(args.draw_box),
            draw_id=bool(args.draw_id),
            draw_centroid=bool(args.draw_centroid),
            save_contours=bool(args.save_contours),
            save_tracks_csv=bool(args.save_tracks_csv),
            delete_mask_png=delete_mask_png,
            stabilize_ids=bool(args.stabilize_ids),
            contours_format=str(args.contours_format),
            contours_gzip=bool(args.contours_gzip),
            no_chunked_fallback=bool(args.no_chunked_fallback),
            chunk_size=args.chunk_size,
            overlap_frames=int(getattr(args, "overlap_frames", 2)),
        )
        if ok:
            print(f"  Done: {out_dir}")
            if args.postprocess_points != "none":
                try:
                    from vaila.sam_postprocess import (
                        extract_points_from_sam_run,
                        write_vaila_anchor_csvs,
                    )

                    print(f"[postprocess] mode={args.postprocess_points}")
                    out_csv = extract_points_from_sam_run(out_dir, mode=args.postprocess_points)
                    print(f"[postprocess] wrote {out_csv}")
                    vaila_outs = write_vaila_anchor_csvs(out_dir)
                    print(f"[postprocess] wrote {len(vaila_outs)} vailá anchor CSV(s)")
                except Exception as exc:
                    print(f"[postprocess] FAILED: {exc}")
                    raise SystemExit(3) from exc
            raise SystemExit(0)
        # If we are running under --no-chunked-fallback (i.e. either as a chunk
        # spawned by ``_process_video_chunked`` *or* as a per-video isolated
        # subprocess from the coordinator), and the failure was OOM, exit with
        # EXIT_NEEDS_CHUNKING so the *outer* coordinator can run the chunked
        # fallback from a clean CUDA context.  The current process is poisoned:
        # SAM3 leaves ~13 GiB of orphan C++ workspace tensors after a failed
        # start_session that nothing short of process death can release.
        if bool(args.no_chunked_fallback) and ("out of memory" in err.lower()):
            print(f"  OOM EXHAUSTED on {single.name}; exiting with EXIT_NEEDS_CHUNKING")
            raise SystemExit(EXIT_NEEDS_CHUNKING)
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

        if args.output_base is not None:
            output_base = args.output_base.resolve()
        else:
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = args.output / f"processed_sam_{ts}"
        output_base.mkdir(parents=True, exist_ok=True)

        adjusted_max_frames = _host_ram_adjusted_auto_max_frames(
            video_files,
            max_input_frames=args.max_frames,
            max_input_long_edge=args.max_input_long_edge,
        )
        if adjusted_max_frames is not None:
            args.max_frames = adjusted_max_frames

        print(_format_runtime_banner(args, video_files), flush=True)
        print(f"\nSAM 3 batch — {len(video_files)} video(s) to process")
        for idx, vf in enumerate(video_files, 1):
            print(f"  {idx}. {vf.name}")
        # Host-RAM heads-up for very long broadcast clips — the most common cause
        # of subprocess exit=-9 (SIGKILL by the Linux OOM killer).
        _warn_host_ram_for_videos(video_files, max_input_frames=args.max_frames)

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
        # CUDA OOM cascade.  Runtime evidence (debug session 42b4a5, plus a
        # follow-up on 2026-04-30 with a 1080p×1189 frame clip on a clean RTX
        # 4090):
        #   - Without OOM: post-cleanup alloc=0.009 GiB → batch works.
        #   - With OOM: ~13 GiB orphan C++ tensors persist; gc.collect / empty_cache
        #     cannot reach them → next video AND any chunked-fallback children
        #     spawned from the poisoned process see only ~3 GiB free → cascade.
        # Killing the Python process is the only way to release SAM3's internal
        # C++ workspace pools after a failed start_session.
        #
        # Coordinator pattern (2026-04-30): isolation now runs for *every* video,
        # including single-video CLI invocations.  This keeps the *outer*
        # coordinator process clean (it only runs ``_sam3_guard_cuda_cli`` ~
        # 100 MiB CUDA context) so when a per-video subprocess OOMs and exits
        # with EXIT_NEEDS_CHUNKING, we can run ``_process_video_chunked`` from
        # *here* (the coordinator), spawning chunk subprocesses against a
        # near-empty GPU instead of the 20 GiB-poisoned victim process.
        use_isolation = not args.no_isolate_batch
        failed_cli: list[str] = []

        if use_isolation:
            import subprocess as _sp

            scope = "single-video" if len(video_files) == 1 else "each video"
            print(f"[batch] subprocess-per-video isolation: ENABLED ({scope} in a fresh process)")

            def _build_isolated_cmd(video_file: Path, out_dir: Path) -> list[str]:
                cmd_local = [
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
                    # Per-video subprocess MUST NOT chunk in-process: any chunk
                    # subprocesses it would spawn would inherit a poisoned GPU
                    # (the OOM victim still holds ~13 GiB of orphan C++ pools
                    # until process exit).  When OOM-exhausted, this child
                    # exits with EXIT_NEEDS_CHUNKING and the coordinator runs
                    # the chunked fallback from a clean state.
                    "--no-chunked-fallback",
                ]
                if args.max_frames is not None:
                    cmd_local += ["--max-frames", str(args.max_frames)]
                if args.max_input_long_edge is not None:
                    cmd_local += ["--max-input-long-edge", str(args.max_input_long_edge)]
                if args.checkpoint is not None:
                    cmd_local += ["--checkpoint", str(args.checkpoint)]
                if args.no_overlay:
                    cmd_local.append("--no-overlay")
                if args.no_png:
                    cmd_local.append("--no-png")
                if args.frame_by_frame:
                    cmd_local.append("--frame-by-frame")
                if not args.overlay_rich:
                    cmd_local.append("--no-overlay-rich")
                if not args.draw_contour:
                    cmd_local.append("--no-draw-contour")
                if not args.draw_box:
                    cmd_local.append("--no-draw-box")
                if not args.draw_id:
                    cmd_local.append("--no-draw-id")
                if args.draw_centroid:
                    cmd_local.append("--draw-centroid")
                if not args.save_contours:
                    cmd_local.append("--no-save-contours")
                if not args.save_tracks_csv:
                    cmd_local.append("--no-save-tracks-csv")
                if args.contours_format:
                    cmd_local += ["--contours-format", str(args.contours_format)]
                if args.contours_gzip:
                    cmd_local.append("--contours-gzip")
                if args.tracks_only:
                    cmd_local.append("--tracks-only")
                if keep_mask_png:
                    cmd_local.append("--keep-mask-png")
                else:
                    cmd_local.append("--delete-mask-png")
                if args.stabilize_ids:
                    cmd_local.append("--stabilize-ids")
                return cmd_local

            for idx, video_file in enumerate(video_files, 1):
                print(f"\n{'=' * 60}")
                print(f"Processing video {idx}/{len(video_files)}: {video_file.name} (isolated)")
                print(f"{'=' * 60}")
                out_dir = output_base / video_file.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = _build_isolated_cmd(video_file, out_dir)
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
                    continue
                if rc == EXIT_NEEDS_CHUNKING:
                    print(
                        f"  [coordinator] {video_file.name}: per-video subprocess OOM-exhausted; "
                        "running chunked divide-and-conquer from coordinator (clean GPU)..."
                    )
                    import shutil as _shutil_local

                    # Wipe stale chunk artefacts from the failed in-process attempt
                    # so the coordinator-driven retry starts clean.
                    chunk_work_dir = out_dir / "_chunks"
                    if chunk_work_dir.exists():
                        with contextlib.suppress(OSError):
                            _shutil_local.rmtree(str(chunk_work_dir), ignore_errors=True)
                    failure_marker = out_dir / "FAILED_sam.txt"
                    with contextlib.suppress(OSError):
                        failure_marker.unlink(missing_ok=True)
                    chunk_ok, chunk_msg = _process_video_chunked(
                        video_file,
                        out_dir,
                        text_prompt=args.text,
                        frame_index=args.frame,
                        checkpoint=args.checkpoint,
                        max_input_frames=args.max_frames,
                        max_input_long_edge=args.max_input_long_edge,
                        save_overlay_mp4=not args.no_overlay,
                        save_mask_png=not args.no_png,
                        overlay_rich=bool(args.overlay_rich),
                        draw_contour=bool(args.draw_contour),
                        draw_box=bool(args.draw_box),
                        draw_id=bool(args.draw_id),
                        draw_centroid=bool(args.draw_centroid),
                        save_contours=bool(args.save_contours),
                        save_tracks_csv=bool(args.save_tracks_csv),
                        delete_mask_png=delete_mask_png,
                        stabilize_ids=bool(args.stabilize_ids),
                        contours_format=str(args.contours_format),
                        contours_gzip=bool(args.contours_gzip),
                        chunk_size=args.chunk_size,
                        overlap_frames=int(getattr(args, "overlap_frames", 2)),
                    )
                    if chunk_ok:
                        print(f"  Done (chunked): {out_dir} — {chunk_msg}")
                    else:
                        err_msg = f"chunked fallback failed: {chunk_msg}"
                        print(f"  ERROR on {video_file.name}: {err_msg}")
                        failed_cli.append(f"{video_file.name}: {err_msg}")
                else:
                    err_msg = _format_subprocess_exit_diagnosis(rc)
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
                    overlay_rich=bool(args.overlay_rich),
                    draw_contour=bool(args.draw_contour),
                    draw_box=bool(args.draw_box),
                    draw_id=bool(args.draw_id),
                    draw_centroid=bool(args.draw_centroid),
                    save_contours=bool(args.save_contours),
                    save_tracks_csv=bool(args.save_tracks_csv),
                    delete_mask_png=delete_mask_png,
                    stabilize_ids=bool(args.stabilize_ids),
                    contours_format=str(args.contours_format),
                    contours_gzip=bool(args.contours_gzip),
                    no_chunked_fallback=bool(args.no_chunked_fallback),
                    chunk_size=args.chunk_size,
                    overlap_frames=int(getattr(args, "overlap_frames", 2)),
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

        all_failed = len(failed_cli) >= len(video_files)
        if args.postprocess_points != "none" and not all_failed:
            ob = output_base
            try:
                from vaila.sam_postprocess import (
                    extract_points_for_batch,
                    write_vaila_anchor_csvs_for_batch,
                )

                print(f"\n[postprocess] mode={args.postprocess_points}")
                outs = extract_points_for_batch(ob, mode=args.postprocess_points)
                vaila_outs = write_vaila_anchor_csvs_for_batch(ob)
                print(
                    f"[postprocess] wrote {len(outs)} sam_points.csv + "
                    f"{len(vaila_outs)} vailá anchor CSV(s)."
                )
            except Exception as exc:
                print(f"[postprocess] FAILED: {exc}")
                if not failed_cli:
                    failed_cli.append(f"postprocess: {exc}")
        elif args.postprocess_points != "none" and all_failed:
            print("\n[postprocess] skipped because all SAM 3 video runs failed.")

        if failed_cli:
            raise SystemExit(3)
        return

    run_sam_video()


if __name__ == "__main__":
    main()
