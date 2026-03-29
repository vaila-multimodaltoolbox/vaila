"""
SAM 3 video segmentation (Meta) — text-prompt masks via Hugging Face checkpoints.

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
GPU: CUDA only (sam3 video predictor loads on .cuda()).

Usage:
    uv run vaila/vaila_sam.py
    uv run vaila/vaila_sam.py --download-weights   # HF_TOKEN or hf auth login
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import os
import platform
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

SAM3_DEFAULT_CKPT_NAME = "sam3.pt"
SAM3_HF_REPO_ID = "facebook/sam3"


def _repo_root() -> Path:
    """Project root (parent of the ``vaila`` package directory)."""
    return Path(__file__).resolve().parent.parent


def _package_sam3_dir() -> Path:
    return Path(__file__).resolve().parent / "models" / "sam3"


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


def _resolve_sam3_checkpoint_file(checkpoint: Path | None) -> Path | None:
    """
    Return path to sam3.pt, or None to let sam3 download from Hub (needs auth + access).

    Order: argument → env SAM3_CHECKPOINT / VAILA_SAM3_CHECKPOINT →
    ``vaila/models/sam3/sam3.pt`` → ``vaila/models/sam3/sam3_weights/sam3.pt`` →
    legacy ``<repo>/sam3_weights/sam3.pt``.
    """
    raw: str | None = None
    if checkpoint is not None:
        raw = str(checkpoint)
    if not raw:
        raw = os.environ.get("SAM3_CHECKPOINT") or os.environ.get("VAILA_SAM3_CHECKPOINT")
    if not raw or not raw.strip():
        pkg_ckpt = _package_sam3_ckpt_path()
        if pkg_ckpt.is_file():
            return pkg_ckpt
        nested_ckpt = _package_sam3_nested_weights_ckpt_path()
        if nested_ckpt.is_file():
            return nested_ckpt
        legacy_ckpt = _repo_root() / "sam3_weights" / SAM3_DEFAULT_CKPT_NAME
        if legacy_ckpt.is_file():
            return legacy_ckpt
        return None
    p = Path(raw).expanduser().resolve()
    if p.is_dir():
        p = p / SAM3_DEFAULT_CKPT_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {p}\n"
            f"Expected a file named {SAM3_DEFAULT_CKPT_NAME} (or pass the full path to sam3.pt)."
        )
    return p


def _resolve_bpe_path() -> Path:
    """SAM3 PyPI wheel may omit site-packages/assets; fall back to boxmot's CLIP BPE."""
    import sam3

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
    """Estimate a safe frame cap based on available GPU VRAM.

    Empirical data on RTX 5050 Laptop (7.53 GiB):
      - 256 frames → 7.23 GiB PyTorch allocated, OOM at +12 MiB
      - 125 frames → 6.76 GiB PyTorch allocated, OOM at +914 MiB (processing)
      - Per-frame cost: ~3.6 MiB  |  Model base: ~6.3 GiB
      - Processing peak: ~0.9 GiB contiguous allocation

    Budget: model_base + N*per_frame + processing_peak + headroom < total_vram
    With ``expandable_segments:True`` the 0.9 GiB processing allocation can
    reuse freed memory, so we only reserve 0.5 GiB processing overhead here
    (the allocator config is set in ``_setup_cuda_for_sam3``).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 32
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_gib = props.total_memory / (1024**3)
    except Exception:
        return 32

    model_gib = 6.3
    processing_gib = 0.5
    headroom_gib = 0.3
    per_frame_mib = 3.6
    available_gib = max(0.0, total_gib - model_gib - processing_gib - headroom_gib)
    safe_frames = int(available_gib * 1024 / per_frame_mib)
    safe_frames = max(16, min(safe_frames, 2048))
    print(
        f"[SAM3 VRAM] GPU {total_gib:.1f} GiB → auto max_frames={safe_frames} "
        f"(model ~6.3 GiB, available ~{available_gib:.1f} GiB for frames)"
    )
    return safe_frames


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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


def _setup_cuda_for_sam3() -> None:
    """Configure PyTorch CUDA allocator for SAM3's large contiguous allocations.

    ``expandable_segments`` avoids fragmentation that causes OOM even when the
    total free VRAM looks sufficient.  Without it, 8 GiB cards routinely fail
    at the ``start_session`` / ``add_prompt`` step where SAM3 allocates a
    ~900 MiB feature tensor.
    """
    import torch

    key = "PYTORCH_CUDA_ALLOC_CONF"
    cur = os.environ.get(key, "")
    if "expandable_segments" not in cur:
        new_val = f"{cur},expandable_segments:True" if cur else "expandable_segments:True"
        os.environ[key] = new_val

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
) -> None:
    import torch
    from sam3.model_builder import build_sam3_video_predictor

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
        try:
            predictor = build_sam3_video_predictor(**pred_kw)
        except Exception as e:
            if _is_gated_repo_error(e):
                raise RuntimeError(_hf_access_help()) from e
            raise

        torch.cuda.empty_cache()

        try:
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

        prompt_resp = predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": fi_run,
                "text": text_prompt,
            }
        )
        outputs_by_frame: dict[int, dict] = {}
        fi0 = int(prompt_resp["frame_index"])
        outputs_by_frame[fi0] = prompt_resp["outputs"]

        stream_req = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "both",
            "start_frame_index": fi_run,
            "max_frame_num_to_track": None,
        }
        for chunk in predictor.handle_stream_request(stream_req):
            outputs_by_frame[int(chunk["frame_index"])] = chunk["outputs"]

        with contextlib.suppress(Exception):
            predictor.handle_request({"type": "close_session", "session_id": session_id})
        with contextlib.suppress(Exception):
            predictor.shutdown()

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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(overlay_path), fourcc, float(fps), (w, h))
            if not writer.isOpened():
                writer = None

        masks_dir = output_dir / "masks"
        if save_mask_png:
            masks_dir.mkdir(parents=True, exist_ok=True)

        meta_rows: list[str] = []

        cap = cv2.VideoCapture(vp)
        for frame_idx in sorted(outputs_by_frame.keys()):
            out = outputs_by_frame[frame_idx]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
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
            for i in range(masks.shape[0]):
                oid = int(oids[i])
                if save_mask_png:
                    png = masks_dir / f"frame_{frame_idx:06d}_obj_{oid}.png"
                    cv2.imwrite(str(png), (masks[i].astype(np.uint8)) * 255)
                pr = float(probs[i]) if probs is not None and i < len(probs) else float("nan")
                bx = by = bw = bh = float("nan")
                if boxes is not None and i < len(boxes):
                    bx, by, bw, bh = (float(x) for x in boxes[i])
                meta_rows.append(f"{frame_idx},{oid},{pr:.6f},{bx:.6f},{by:.6f},{bw:.6f},{bh:.6f}")

        cap.release()
        if writer is not None:
            writer.release()

        meta_path = output_dir / "sam_frames_meta.csv"
        meta_path.write_text(
            "frame,obj_id,prob,box_x,box_y,box_w,box_h\n" + "\n".join(meta_rows) + "\n",
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


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


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
        self.result: tuple[Path, Path, Path | None, str, int, bool, bool] | None = None

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        ttk.Label(
            frm,
            text=(
                "Select an input directory with videos (batch) or a single video file.\n"
                "Requires: uv sync --extra sam  |  NVIDIA CUDA  |  HF: accept facebook/sam3 + hf auth login\n"
                "Checkpoint: auto-detected in vaila/models/sam3/ or browse below.\n"
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

        ttk.Label(frm, text="sam3.pt (optional):").grid(row=3, column=0, sticky="nw", pady=4)
        self.ckpt_var = tk.StringVar(value=os.environ.get("SAM3_CHECKPOINT", ""))
        ttk.Entry(frm, textvariable=self.ckpt_var, width=52).grid(
            row=3, column=1, sticky="ew", pady=4
        )
        ttk.Button(frm, text="Browse…", command=self._browse_ckpt).grid(row=3, column=2, padx=4)

        ttk.Label(frm, text="Text prompt:").grid(row=4, column=0, sticky="w", pady=4)
        self.prompt_var = tk.StringVar(value="person")
        ttk.Entry(frm, textvariable=self.prompt_var, width=52).grid(
            row=4, column=1, columnspan=2, sticky="ew", pady=4
        )

        ttk.Label(frm, text="Prompt frame index:").grid(row=5, column=0, sticky="w", pady=4)
        self.frame_var = tk.StringVar(value="0")
        ttk.Entry(frm, textvariable=self.frame_var, width=12).grid(
            row=5, column=1, sticky="w", pady=4
        )

        self.overlay_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Save overlay MP4", variable=self.overlay_var).grid(
            row=6, column=1, sticky="w"
        )
        ttk.Checkbutton(frm, text="Save mask PNGs (per object)", variable=self.png_var).grid(
            row=7, column=1, sticky="w"
        )

        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=3, pady=12)
        ttk.Button(btns, text="Run", command=self._on_ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, padx=4)

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
        )
        self.grab_release()
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        with contextlib.suppress(tk.TclError):
            self.grab_release()
        self.destroy()


def run_sam_video(existing_root: tk.Tk | None = None) -> None:
    """GUI entry: configure and run SAM3 on a directory of videos (batch) or a single file."""
    if importlib.util.find_spec("sam3") is None:
        root_e = tk.Tk()
        root_e.withdraw()
        messagebox.showerror(
            "SAM 3 not installed",
            "Install optional dependencies:\n  uv sync --extra sam",
            parent=root_e,
        )
        root_e.destroy()
        return

    root = existing_root
    owns_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True
    try:
        if platform.system() == "Windows":
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

    input_path, out_parent, ckpt_opt, prompt, frame_idx, save_ov, save_png = dlg.result

    video_files = _find_videos(input_path) if input_path.is_dir() else [input_path]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = out_parent / f"processed_sam_{ts}"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"\nSAM 3 batch — {len(video_files)} video(s) to process")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")

    failed: list[str] = []
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing video {i}/{len(video_files)}: {video_file.name}")
        print(f"{'=' * 60}")
        output_dir = output_base / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_sam3_on_video(
                video_file,
                output_dir,
                prompt,
                frame_index=frame_idx,
                checkpoint=ckpt_opt,
                save_overlay_mp4=save_ov,
                save_mask_png=save_png,
            )
            print(f"  Done: {output_dir}")
        except Exception as e:
            print(f"  ERROR on {video_file.name}: {e}")
            failed.append(f"{video_file.name}: {e}")

    summary = f"Processed {len(video_files) - len(failed)}/{len(video_files)} video(s).\nOutput: {output_base}"
    if failed:
        summary += f"\n\nFailed ({len(failed)}):\n" + "\n".join(failed)
        messagebox.showwarning("SAM 3 — finished with errors", summary, parent=root)
    else:
        messagebox.showinfo("SAM 3 — done", summary, parent=root)

    if owns_root:
        root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM 3 video segmentation (vailá)")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input video file OR directory containing videos (batch)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output base directory")
    parser.add_argument("-t", "--text", type=str, default="person", help="Text prompt")
    parser.add_argument("-f", "--frame", type=int, default=0, help="Frame index for prompt")
    parser.add_argument("--no-overlay", action="store_true", help="Skip overlay MP4")
    parser.add_argument("--no-png", action="store_true", help="Skip mask PNGs")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to sam3.pt (or folder containing it). Skips Hub download. "
        "Default: env SAM3_CHECKPOINT / VAILA_SAM3_CHECKPOINT, else vaila/models/sam3/sam3.pt "
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
    args = parser.parse_args()

    if args.download_weights:
        out_ckpt = download_sam3_weights_to_vaila_models()
        print(f"SAM3 weights ready: {out_ckpt}")
        return

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

        for idx, video_file in enumerate(video_files, 1):
            print(f"\n{'=' * 60}")
            print(f"Processing video {idx}/{len(video_files)}: {video_file.name}")
            print(f"{'=' * 60}")
            out_dir = output_base / video_file.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_sam3_on_video(
                    video_file,
                    out_dir,
                    args.text,
                    frame_index=args.frame,
                    checkpoint=args.checkpoint,
                    max_input_frames=args.max_frames,
                    save_overlay_mp4=not args.no_overlay,
                    save_mask_png=not args.no_png,
                )
                print(f"  Done: {out_dir}")
            except Exception as e:
                print(f"  ERROR on {video_file.name}: {e}")
        print(f"\nAll done. Output: {output_base}")
        return

    run_sam_video()


if __name__ == "__main__":
    main()
