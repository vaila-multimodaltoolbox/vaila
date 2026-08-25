"""
Project: vailá
Script: sam3dinov3.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila

Creation Date: 01 August 2026
Update Date: 24 August 2026
Version: 0.3.112

Description:
    Monocular markerless **3D** human mesh/skeleton recovery from video, using
    SAM 3 as the detection/identity authority and **SAM 3D Body (DINOv3
    backbone)** as the 3D lifter.

    The pipeline mirrors ``sam3sapiens2.py`` (SAM3 -> Sapiens2 2D pose), but
    swaps the second stage for ``facebook/sam-3d-body-dinov3``:

        1. SAM 3 segments and tracks people; it owns bbox, silhouette,
           score and the persistent ``obj_id``.
        2. Those SAM boxes and silhouettes are fed directly to SAM 3D Body via
           ``SAM3DBodyEstimator.process_one_image(bboxes=..., masks=...)`` — the
           upstream ViTDet detector and SAM2 segmentor are never loaded.
        3. SAM 3D Body predicts, per person, an MHR (Momentum Human Rig) mesh,
           3D keypoints in metres, 2D reprojections in pixels, the perspective
           camera translation, and the focal length.
        4. ``person_id`` is exactly the SAM ``obj_id``, so no second Re-ID stage
           can swap identities between frames.

    References:
        SAM 3            https://ai.meta.com/research/sam3/
        DINOv3           https://ai.meta.com/research/dinov3/
        SAM 3D Body      https://github.com/facebookresearch/sam-3d-body
        Weights          https://huggingface.co/facebook/sam-3d-body-dinov3
        Rerun viewer     https://github.com/rerun-io/sam3d-body-rerun

Setup:
    bash bin/setup_fifa_sam3d.sh     # clones sam-3d-body + gated DINOv3 weights
    uv sync --extra sam              # SAM 3 stack (CUDA)

Usage:
    uv run python -u vaila/sam3dinov3.py \
        -i /path/to/video_or_folder -o /path/to/output -t person

    # Reuse an existing processed_sam_* directory (no SAM rerun):
    uv run python -u vaila/sam3dinov3.py \
        -i /path/to/videos -o /path/to/output \
        --sam-results /path/to/processed_sam_YYYYMMDD_HHMMSS

    # Metric-accurate run with a known focal length (px) and mesh export:
    uv run python -u vaila/sam3dinov3.py \
        -i clip.mp4 -o /path/to/output --focal-px 1400 --save-mesh

    # GUI: omit arguments, or Frame B -> YOLO + FB -> SAM3+DINOv3 3D

Runtime:
    Requires an NVIDIA CUDA GPU. The upstream estimator moves its batch to
    ``cuda`` unconditionally, so CPU/MPS execution is not supported.

License:
    This program is licensed under the GNU Affero General Public License v3.0.
    SAM 3 and SAM 3D Body weights keep their respective Meta licenses.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import importlib
import io
import json
import os
import shlex
import sys
import tkinter as tk
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

try:
    from .cli_highlight import print_gui_cli_mirror
    from .gpu_subprocess import run_isolated_gpu_subprocess
    from .joint_kinematics import (
        MHR127_NUM_JOINTS,
        MHR127_PARENTS,
        infer_joint_names_from_positions,
        local_rotations_from_global,
        rotmat_to_euler_xyz_deg,
        rotmat_to_quat_wxyz,
    )
    from .sam3sapiens2 import (
        DEFAULT_BBOX_PADDING,
        DEFAULT_CONTOUR_MARGIN_PX,
        DEFAULT_MAX_PERSONS,
        DEFAULT_MIN_SAM_AREA,
        DEFAULT_MIN_SAM_SCORE,
        SamGuidance,
        _color_for_id,
        _contour_mask,
        _draw_sam_guidance,
        _find_videos,
        _guidance_for_frame,
        _pose_bbox_from_sam,
        _prepare_gui_root,
        _video_frame_count,
        find_local_sam_dir,
        load_sam_guidance,
        prepare_sam_rerun_dir,
        resolve_auto_resume_output_base,
        resolve_sam_results_dir,
        run_sam_stage,
        write_batch_input_marker,
    )
    from .vaila_sam import _open_sam3_video_writer
except ImportError:  # standalone execution
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    from gpu_subprocess import run_isolated_gpu_subprocess  # ty: ignore[unresolved-import]
    from joint_kinematics import (  # ty: ignore[unresolved-import]
        MHR127_NUM_JOINTS,
        MHR127_PARENTS,
        infer_joint_names_from_positions,
        local_rotations_from_global,
        rotmat_to_euler_xyz_deg,
        rotmat_to_quat_wxyz,
    )
    from sam3sapiens2 import (  # ty: ignore[unresolved-import]
        DEFAULT_BBOX_PADDING,
        DEFAULT_CONTOUR_MARGIN_PX,
        DEFAULT_MAX_PERSONS,
        DEFAULT_MIN_SAM_AREA,
        DEFAULT_MIN_SAM_SCORE,
        SamGuidance,
        _color_for_id,
        _contour_mask,
        _draw_sam_guidance,
        _find_videos,
        _guidance_for_frame,
        _pose_bbox_from_sam,
        _prepare_gui_root,
        _video_frame_count,
        find_local_sam_dir,
        load_sam_guidance,
        prepare_sam_rerun_dir,
        resolve_auto_resume_output_base,
        resolve_sam_results_dir,
        run_sam_stage,
        write_batch_input_marker,
    )
    from vaila_sam import _open_sam3_video_writer  # ty: ignore[unresolved-import]

DEFAULT_WEIGHTS_SUBDIR = Path("models") / "sam-3d-dinov3"
DEFAULT_HF_REPO_ID = "facebook/sam-3d-body-dinov3"
#: Optional upstream package, resolved at runtime rather than pip-installed.
SAM3D_PACKAGE = "sam_3d_body"
SAM3D_FOV_MODULE = "tools.build_fov_estimator"
INFERENCE_TYPES = ("full", "body", "hand")
DEFAULT_INFERENCE_TYPE = "full"
DEFAULT_STRIDE = 1

#: First 70 of the 308 MHR keypoints (the body/hand subset used for analysis).
#: Kept as a literal fallback so CSV headers stay stable even when the upstream
#: ``sam_3d_body`` package is not importable (dry-run, docs, tests).
MHR70_NAMES: tuple[str, ...] = (
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-big-toe-tip",
    "left-small-toe-tip",
    "left-heel",
    "right-big-toe-tip",
    "right-small-toe-tip",
    "right-heel",
    "right-thumb-tip",
    "right-thumb-first-joint",
    "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip",
    "right-index-first-joint",
    "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-tip",
    "right-middle-first-joint",
    "right-middle-second-joint",
    "right-middle-third-joint",
    "right-ring-tip",
    "right-ring-first-joint",
    "right-ring-second-joint",
    "right-ring-third-joint",
    "right-pinky-tip",
    "right-pinky-first-joint",
    "right-pinky-second-joint",
    "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip",
    "left-thumb-first-joint",
    "left-thumb-second-joint",
    "left-thumb-third-joint",
    "left-index-tip",
    "left-index-first-joint",
    "left-index-second-joint",
    "left-index-third-joint",
    "left-middle-tip",
    "left-middle-first-joint",
    "left-middle-second-joint",
    "left-middle-third-joint",
    "left-ring-tip",
    "left-ring-first-joint",
    "left-ring-second-joint",
    "left-ring-third-joint",
    "left-pinky-tip",
    "left-pinky-first-joint",
    "left-pinky-second-joint",
    "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon",
    "right-olecranon",
    "left-cubital-fossa",
    "right-cubital-fossa",
    "left-acromion",
    "right-acromion",
    "neck",
)

#: Body skeleton used for the 2D overlay, expressed by keypoint name so it stays
#: valid if upstream ever reorders the MHR indices.
SKELETON_EDGE_NAMES: tuple[tuple[str, str], ...] = (
    ("neck", "nose"),
    ("nose", "left-eye"),
    ("nose", "right-eye"),
    ("left-eye", "left-ear"),
    ("right-eye", "right-ear"),
    ("neck", "left-shoulder"),
    ("neck", "right-shoulder"),
    ("left-shoulder", "left-elbow"),
    ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"),
    ("right-elbow", "right-wrist"),
    ("left-shoulder", "left-hip"),
    ("right-shoulder", "right-hip"),
    ("left-hip", "right-hip"),
    ("left-hip", "left-knee"),
    ("left-knee", "left-ankle"),
    ("left-ankle", "left-heel"),
    ("left-ankle", "left-big-toe-tip"),
    ("left-ankle", "left-small-toe-tip"),
    ("left-heel", "left-big-toe-tip"),
    ("left-heel", "left-small-toe-tip"),
    ("left-big-toe-tip", "left-small-toe-tip"),
    ("right-hip", "right-knee"),
    ("right-knee", "right-ankle"),
    ("right-ankle", "right-heel"),
    ("right-ankle", "right-big-toe-tip"),
    ("right-ankle", "right-small-toe-tip"),
    ("right-heel", "right-big-toe-tip"),
    ("right-heel", "right-small-toe-tip"),
    ("right-big-toe-tip", "right-small-toe-tip"),
)

# MHR70 joint names carry an explicit "left-"/"right-" prefix, so the skeleton
# overlay can be colored by body side directly from the name — same palette
# used by sam3sapiens2_visualize / sam3dinov3_visualize for a consistent look
# across the whole SAM3 family: left=green, right=orange, center/spine=blue.
COLOR_LEFT_RGB = (0, 255, 0)
COLOR_RIGHT_RGB = (255, 128, 0)
COLOR_CENTER_RGB = (51, 153, 255)


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(color[2]), int(color[1]), int(color[0])


def _side_color_bgr(name: str) -> tuple[int, int, int]:
    label = name.lower()
    if label.startswith("left-") or label.startswith("left_"):
        return _rgb_to_bgr(COLOR_LEFT_RGB)
    if label.startswith("right-") or label.startswith("right_"):
        return _rgb_to_bgr(COLOR_RIGHT_RGB)
    return _rgb_to_bgr(COLOR_CENTER_RGB)


@dataclass
class Sam3dGuiSettings:
    """User choices collected by the Tkinter dialog."""

    input_path: Path
    output_parent: Path | None
    resume: Path | None
    sam_results: Path | None
    prompt: str
    device: int
    stride: int
    bbox_padding: float
    contour_margin: int
    max_persons: int
    inference_type: str
    use_mask: bool
    save_overlay: bool
    save_mesh: bool
    focal_px: float | None
    weights_dir: Path | None


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _log(message: str) -> None:
    # ``>>`` instead of ``[brackets]``: absl logging (pulled in by mediapipe /
    # opencv) silently swallows bracketed stdout prefixes.
    # Overnight/detached runs must not die on a dropped terminal (EIO/BrokenPipe).
    with contextlib.suppress(OSError, BrokenPipeError):
        print(f">> vaila/sam3dinov3: {message}", flush=True)


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _help_path() -> Path:
    return _module_dir() / "help" / "sam3dinov3.html"


def default_weights_dir() -> Path:
    return _module_dir() / DEFAULT_WEIGHTS_SUBDIR


def _format_cell(value: float | None) -> str:
    if value is None or not np.isfinite(float(value)):
        return ""
    return f"{float(value):.6f}"


def keypoint_names(n_keypoints: int) -> list[str]:
    """MHR names for the first 70 joints; generic ``kpt_NNN`` beyond that."""
    names = list(MHR70_NAMES[:n_keypoints])
    names.extend(f"kpt_{idx:03d}" for idx in range(len(names), n_keypoints))
    return names


def skeleton_edges(names: list[str]) -> list[tuple[int, int]]:
    index = {name: i for i, name in enumerate(names)}
    edges: list[tuple[int, int]] = []
    for a, b in SKELETON_EDGE_NAMES:
        if a in index and b in index:
            edges.append((index[a], index[b]))
    return edges


# --------------------------------------------------------------------------- #
# SAM 3D Body model
# --------------------------------------------------------------------------- #
def resolve_sam3d_assets(weights_dir: Path | None) -> tuple[Path, Path]:
    """Locate ``model.ckpt`` + ``assets/mhr_model.pt`` or explain how to get them."""
    root = (weights_dir or default_weights_dir()).expanduser().resolve()
    ckpt = root / "model.ckpt"
    mhr = root / "assets" / "mhr_model.pt"
    missing = [str(p) for p in (ckpt, mhr) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "SAM 3D Body weights are incomplete.\n"
            + "\n".join(f"  missing: {p}" for p in missing)
            + "\n\nAccept the license and download them:\n"
            f"  1) https://huggingface.co/{DEFAULT_HF_REPO_ID}\n"
            "  2) uv run hf auth login\n"
            "  3) bash bin/setup_fifa_sam3d.sh\n"
            f"     (or: uv run hf download {DEFAULT_HF_REPO_ID} --local-dir {root})"
        )
    return ckpt, mhr


def ensure_sam3d_assets(weights_dir: Path | None) -> tuple[Path, Path]:
    """Resolve SAM 3D Body (DINOv3) weights, downloading the gated repo if absent.

    Runs before the SAM3 stage starts, so a missing checkpoint fails immediately
    instead of after SAM3 has already segmented and exported the whole clip.
    """
    try:
        return resolve_sam3d_assets(weights_dir)
    except FileNotFoundError:
        pass

    root = (weights_dir or default_weights_dir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"[SAM3D] Weights not found under {root} - downloading {DEFAULT_HF_REPO_ID} "
        "(one-time, several GB)...",
        flush=True,
    )
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=DEFAULT_HF_REPO_ID, local_dir=str(root))
    except Exception as e:
        gated = type(e).__name__ == "GatedRepoError" or "403" in str(e)
        hint = (
            f"Accept the license at https://huggingface.co/{DEFAULT_HF_REPO_ID}, "
            "then: uv run hf auth login --force\n"
            if gated
            else ""
        )
        raise RuntimeError(
            f"Failed to download {DEFAULT_HF_REPO_ID}: {e}\n\n{hint}"
            "Or run the full setup: bash bin/setup_fifa_sam3d.sh"
        ) from e
    return resolve_sam3d_assets(weights_dir)


def _sam3d_import_error(exc: Exception) -> RuntimeError:
    searched = "\n".join(f"    {p}" for p in _sam3d_checkout_candidates())
    return RuntimeError(
        "Could not import the 'sam_3d_body' package.\n"
        f"  underlying error: {exc}\n\n"
        "Upstream 'sam-3d-body' ships no pyproject.toml/setup.py, so it cannot be\n"
        "pip-installed; its checkout root has to be on sys.path instead.\n"
        "Clone it with:\n"
        "  bash bin/setup_fifa_sam3d.sh\n"
        "or point VAILA_SAM3D_BODY_DIR at an existing checkout.\n\n"
        f"Searched:\n{searched}"
    )


def _sam3d_checkout_candidates() -> list[Path]:
    """Directories that may hold the ``facebookresearch/sam-3d-body`` checkout."""
    candidates: list[Path] = []
    env_dir = os.environ.get("VAILA_SAM3D_BODY_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    repo_root = _module_dir().parent
    candidates.extend(
        [
            repo_root / "sam_3d_body",
            repo_root / "sam-3d-body",
            Path.cwd() / "sam_3d_body",
            Path.cwd() / "sam-3d-body",
        ]
    )
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def ensure_sam3d_importable() -> Path | None:
    """Put the ``sam-3d-body`` checkout root on ``sys.path`` when needed.

    Upstream has no packaging metadata, so ``uv pip install -e`` cannot work and
    ``import sam_3d_body`` only resolves once the *checkout root* (the directory
    that contains the inner ``sam_3d_body/`` package) is importable. Returns the
    directory that was added, or ``None`` when the package already imported.
    """
    from importlib.util import find_spec

    with contextlib.suppress(Exception):
        spec = find_spec("sam_3d_body")
        # A namespace package (submodule_search_locations without an origin)
        # means we found the bare clone directory, not the real package.
        if spec is not None and spec.origin not in (None, "namespace"):
            return None

    for candidate in _sam3d_checkout_candidates():
        if (candidate / "sam_3d_body" / "__init__.py").is_file():
            path_entry = str(candidate.resolve())
            if path_entry not in sys.path:
                sys.path.insert(0, path_entry)
            # Drop a cached namespace-package entry so the real one is picked up.
            sys.modules.pop("sam_3d_body", None)
            _log(f"Using sam-3d-body checkout: {path_entry}")
            return candidate.resolve()
    return None


def load_sam3d_estimator(
    weights_dir: Path | None,
    *,
    device: str = "cuda",
    fov_estimator_name: str = "",
) -> Any:
    """Build a ``SAM3DBodyEstimator`` with no detector and no segmentor.

    SAM 3 already provides boxes and silhouettes, so ViTDet/SAM2 are skipped:
    that keeps VRAM free and guarantees SAM 3 stays the identity authority.
    """
    ckpt, mhr = resolve_sam3d_assets(weights_dir)
    ensure_sam3d_importable()
    try:
        # Imported dynamically: the package is an optional dependency whose
        # location is only resolved at runtime (see ensure_sam3d_importable),
        # so a static import would be unanalysable anyway.
        # getattr (not attribute syntax) because a bare `sam_3d_body/` clone
        # directory also resolves as an empty namespace package to static
        # analysers, which would then flag these very real attributes.
        sam3d = importlib.import_module(SAM3D_PACKAGE)
        SAM3DBodyEstimator = getattr(sam3d, "SAM3DBodyEstimator")  # noqa: B009
        load_sam_3d_body = getattr(sam3d, "load_sam_3d_body")  # noqa: B009
    except Exception as exc:  # noqa: BLE001 - re-raised with setup instructions
        raise _sam3d_import_error(exc) from exc

    _log(f"Loading SAM 3D Body (DINOv3 backbone) from {ckpt}")
    model, model_cfg = load_sam_3d_body(str(ckpt), device=device, mhr_path=str(mhr))

    fov_estimator = None
    if fov_estimator_name:
        # Same rationale as above: lives in the sam-3d-body checkout, not a wheel.
        build_fov = importlib.import_module(SAM3D_FOV_MODULE)
        _log(f"Loading FOV estimator: {fov_estimator_name}")
        fov_estimator = build_fov.FOVEstimator(name=fov_estimator_name, device=device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
    _log("SAM 3D Body ready (detector/segmentor disabled; SAM3 supplies boxes+masks)")
    return estimator


def build_cam_int(focal_px: float, width: int, height: int) -> Any:
    """3x3 pinhole intrinsics tensor for a known focal length, centred principal point."""
    import torch

    return torch.tensor(
        [
            [
                [float(focal_px), 0.0, width / 2.0],
                [0.0, float(focal_px), height / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------- #
# per-frame inference
# --------------------------------------------------------------------------- #
def _frame_batch_from_guidance(
    frame_guidance: list[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    frame_width: int,
    frame_height: int,
    bbox_padding: float,
    contour_margin: int,
    use_mask: bool,
) -> tuple[list[int], np.ndarray | None, np.ndarray | None]:
    """Turn SAM tracks/contours into ``(obj_ids, boxes Nx4, masks NxHxW)``."""
    obj_ids: list[int] = []
    boxes: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for track, contour_obj in frame_guidance:
        bbox = _pose_bbox_from_sam(
            track,
            contour_obj,
            frame_width=frame_width,
            frame_height=frame_height,
            padding_fraction=bbox_padding,
        )
        if bbox[2] - bbox[0] < 2.0 or bbox[3] - bbox[1] < 2.0:
            continue
        obj_ids.append(int(track["obj_id"]))
        boxes.append(np.asarray(bbox, dtype=np.float32))
        if use_mask:
            raw = _contour_mask(
                (frame_height, frame_width),
                track,
                contour_obj,
                margin_px=contour_margin,
            )
            # SAM 3D Body expects binary 0/1 masks (upstream feeds boolean SAM2
            # masks through ``.astype(np.uint8)``); ``_contour_mask`` yields 0/255.
            masks.append((raw > 0).astype(np.uint8))
    if not boxes:
        return [], None, None
    box_array = np.stack(boxes, axis=0).astype(np.float32)
    mask_array = np.stack(masks, axis=0).astype(np.uint8) if masks else None
    return obj_ids, box_array, mask_array


def _instances_from_outputs(
    obj_ids: list[int],
    outputs: list[dict[str, Any]],
    *,
    frame_idx: int,
) -> list[dict[str, Any]]:
    """Attach SAM identities to SAM 3D Body predictions (order is preserved)."""
    if len(outputs) != len(obj_ids):
        _log(
            f"WARNING frame {frame_idx}: SAM 3D Body returned {len(outputs)} people "
            f"for {len(obj_ids)} SAM boxes; pairing the common prefix only."
        )
    instances: list[dict[str, Any]] = []
    for obj_id, out in zip(obj_ids, outputs, strict=False):
        kp3d = np.asarray(out["pred_keypoints_3d"], dtype=np.float32).reshape(-1, 3)
        kp2d = np.asarray(out["pred_keypoints_2d"], dtype=np.float32).reshape(-1, 2)
        cam_t = np.asarray(out["pred_cam_t"], dtype=np.float32).reshape(3)
        # Per-joint GLOBAL rotation matrices for the model's own 127-joint MHR
        # rig (Meta's Momentum Human Rig) -- NOT the same indexing as the
        # 70-name MHR70_NAMES keypoint list above, which is position-only.
        # Small (127x3x3 floats), unlike vertices, so kept for the whole run
        # rather than dropped after the overlay is drawn. See
        # joint_kinematics.py for the joint-angle math built on this.
        global_rots = out.get("pred_global_rots")
        joint_coords = out.get("pred_joint_coords")
        instances.append(
            {
                "person_id": int(obj_id),
                "sam_obj_id": int(obj_id),
                "bbox": np.asarray(out["bbox"], dtype=np.float32).reshape(4),
                "keypoints_3d": kp3d,
                "keypoints_2d": kp2d,
                "cam_t": cam_t,
                # Camera-frame absolute coordinates: root-relative joints plus the
                # perspective translation (same convention the upstream renderer
                # uses for ``pred_vertices + pred_cam_t``).
                "keypoints_3d_cam": kp3d + cam_t[None, :],
                "focal_length": float(np.asarray(out["focal_length"]).reshape(-1)[0]),
                "vertices": out.get("pred_vertices"),
                "global_rots": (
                    np.asarray(global_rots, dtype=np.float64).reshape(-1, 3, 3)
                    if global_rots is not None
                    else None
                ),
                "joint_coords_3d": (
                    np.asarray(joint_coords, dtype=np.float64).reshape(-1, 3)
                    if joint_coords is not None
                    else None
                ),
            }
        )
    return instances


def _draw_pose_overlay(
    image: np.ndarray,
    instances: list[dict[str, Any]],
    edges: list[tuple[int, int]],
    names: list[str],
    *,
    draw_ids: bool,
) -> np.ndarray:
    """Draw the reprojected MHR skeleton, colored by joint side.

    Left/right/center coloring (rather than one solid color per person) makes
    the live overlay video readable the same way sam3sapiens2_visualize and
    sam3dinov3_visualize already are. ``_color_for_id`` is kept only for the
    ``ID nn`` text label, so multiple people stay distinguishable by that tag.
    """
    out = image
    side_colors = [_side_color_bgr(name) for name in names]
    center = _rgb_to_bgr(COLOR_CENTER_RGB)
    for inst in instances:
        id_color = _color_for_id(int(inst["person_id"]))
        kp2d = inst["keypoints_2d"]
        height, width = out.shape[:2]
        for a, b in edges:
            if a >= len(kp2d) or b >= len(kp2d):
                continue
            pa = kp2d[a]
            pb = kp2d[b]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            color_a = side_colors[a] if a < len(side_colors) else center
            color_b = side_colors[b] if b < len(side_colors) else center
            # Only take a side color when both endpoints agree; mixed edges
            # (e.g. neck -> shoulder) fall back to the center color.
            color = color_a if color_a == color_b else center
            cv2.line(
                out,
                (int(round(float(pa[0]))), int(round(float(pa[1])))),
                (int(round(float(pb[0]))), int(round(float(pb[1])))),
                color,
                2,
                cv2.LINE_AA,
            )
        n_draw = min(len(kp2d), len(names))
        for idx in range(n_draw):
            point = kp2d[idx]
            if not np.isfinite(point).all():
                continue
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < width and 0 <= y < height:
                joint_color = side_colors[idx] if idx < len(side_colors) else center
                cv2.circle(out, (x, y), 3, joint_color, -1, cv2.LINE_AA)
        if draw_ids:
            depth = float(inst["cam_t"][2])
            label = f"ID {int(inst['person_id'])}  z={depth:.2f} m"
            x1, y1 = inst["bbox"][0], inst["bbox"][1]
            cv2.putText(
                out,
                label,
                (max(0, int(x1)), max(34, int(y1) - 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                id_color,
                2,
                cv2.LINE_AA,
            )
    return out


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def _collect_person_ids(timeline: dict[int, list[dict[str, Any]]]) -> list[int]:
    ids: set[int] = set()
    for instances in timeline.values():
        ids.update(int(inst["person_id"]) for inst in instances)
    return sorted(ids)


def write_long_keypoints_csvs(
    output_dir: Path,
    stem: str,
    timeline: dict[int, list[dict[str, Any]]],
    names: list[str],
) -> list[Path]:
    """Long-format 3D (metres) and 2D (pixels) keypoint tables."""
    path3d = output_dir / f"{stem}_sam3dinov3_keypoints3d.csv"
    path2d = output_dir / f"{stem}_sam3dinov3_keypoints2d.csv"
    with (
        path3d.open("w", newline="", encoding="utf-8") as fh3,
        path2d.open("w", newline="", encoding="utf-8") as fh2,
    ):
        w3 = csv.writer(fh3)
        w2 = csv.writer(fh2)
        w3.writerow(
            [
                "frame",
                "person_id",
                "kpt_idx",
                "kpt_name",
                "x_m",
                "y_m",
                "z_m",
                "xcam_m",
                "ycam_m",
                "zcam_m",
            ]
        )
        w2.writerow(["frame", "person_id", "kpt_idx", "kpt_name", "x_px", "y_px"])
        for frame_idx in sorted(timeline):
            for inst in timeline[frame_idx]:
                pid = int(inst["person_id"])
                kp3d = inst["keypoints_3d"]
                kp3d_cam = inst["keypoints_3d_cam"]
                kp2d = inst["keypoints_2d"]
                for kpt_idx, name in enumerate(names):
                    if kpt_idx < len(kp3d):
                        rel = kp3d[kpt_idx]
                        cam = kp3d_cam[kpt_idx]
                        w3.writerow(
                            [
                                frame_idx,
                                pid,
                                kpt_idx,
                                name,
                                _format_cell(rel[0]),
                                _format_cell(rel[1]),
                                _format_cell(rel[2]),
                                _format_cell(cam[0]),
                                _format_cell(cam[1]),
                                _format_cell(cam[2]),
                            ]
                        )
                    if kpt_idx < len(kp2d):
                        pt = kp2d[kpt_idx]
                        w2.writerow(
                            [
                                frame_idx,
                                pid,
                                kpt_idx,
                                name,
                                _format_cell(pt[0]),
                                _format_cell(pt[1]),
                            ]
                        )
    return [path3d, path2d]


def write_camera_csv(
    output_dir: Path,
    stem: str,
    timeline: dict[int, list[dict[str, Any]]],
) -> Path:
    path = output_dir / f"{stem}_sam3dinov3_camera.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame",
                "person_id",
                "focal_length_px",
                "cam_t_x_m",
                "cam_t_y_m",
                "cam_t_z_m",
                "bbox_x1_px",
                "bbox_y1_px",
                "bbox_x2_px",
                "bbox_y2_px",
            ]
        )
        for frame_idx in sorted(timeline):
            for inst in timeline[frame_idx]:
                cam_t = inst["cam_t"]
                bbox = inst["bbox"]
                writer.writerow(
                    [
                        frame_idx,
                        int(inst["person_id"]),
                        _format_cell(inst["focal_length"]),
                        _format_cell(cam_t[0]),
                        _format_cell(cam_t[1]),
                        _format_cell(cam_t[2]),
                        _format_cell(bbox[0]),
                        _format_cell(bbox[1]),
                        _format_cell(bbox[2]),
                        _format_cell(bbox[3]),
                    ]
                )
    return path


def _joint_rig_names(
    timeline: dict[int, list[dict[str, Any]]], names: list[str]
) -> list[str] | None:
    """Infer names for the 127-joint MHR rig, once, from the first usable frame.

    The rig-joint <-> keypoint-name correspondence is a fixed model property
    (not something that changes frame to frame), so this only needs to run
    once per run, against whichever frame/person has both fields populated.
    Returns None if no instance in the whole timeline carries rig data (e.g.
    an older run, or if the upstream ``process_one_image`` output ever
    drops these fields).
    """
    for frame_idx in sorted(timeline):
        for inst in timeline[frame_idx]:
            rig = inst.get("joint_coords_3d")
            if rig is not None and inst.get("global_rots") is not None:
                return infer_joint_names_from_positions(rig, inst["keypoints_3d"], names)
    return None


def write_long_joint_angles_csv(
    output_dir: Path,
    stem: str,
    timeline: dict[int, list[dict[str, Any]]],
    names: list[str],
) -> Path | None:
    """Long-format joint-angle table: local (parent-relative) Euler + quaternion.

    One row per frame/person/joint of the model's own 127-joint MHR rig (see
    ``joint_kinematics.py`` for why this is preferred over reconstructing
    angles from keypoint positions, and for the Euler/quaternion
    conventions used). Returns None (writes nothing) if no instance in this
    run carries the rotation fields -- e.g. a run predating this feature.
    """
    rig_names = _joint_rig_names(timeline, names)
    if rig_names is None:
        _log(
            "No per-joint rotation data in this run's predictions "
            "(pred_global_rots); skipping joint-angle CSV."
        )
        return None

    path = output_dir / f"{stem}_sam3dinov3_joint_angles.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame",
                "person_id",
                "joint_idx",
                "joint_name",
                "parent_idx",
                "euler_x_deg",
                "euler_y_deg",
                "euler_z_deg",
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
            ]
        )
        for frame_idx in sorted(timeline):
            for inst in timeline[frame_idx]:
                global_rots = inst.get("global_rots")
                if global_rots is None:
                    continue
                pid = int(inst["person_id"])
                n = min(len(global_rots), MHR127_NUM_JOINTS)
                local_rots = local_rotations_from_global(global_rots[:n], MHR127_PARENTS[:n])
                euler = rotmat_to_euler_xyz_deg(local_rots)
                quat = rotmat_to_quat_wxyz(local_rots)
                for j in range(n):
                    writer.writerow(
                        [
                            frame_idx,
                            pid,
                            j,
                            rig_names[j] if j < len(rig_names) else f"joint_{j:03d}",
                            MHR127_PARENTS[j],
                            _format_cell(euler[j, 0]),
                            _format_cell(euler[j, 1]),
                            _format_cell(euler[j, 2]),
                            _format_cell(quat[j, 0]),
                            _format_cell(quat[j, 1]),
                            _format_cell(quat[j, 2]),
                            _format_cell(quat[j, 3]),
                        ]
                    )
    return path


def _instance_for_person(instances: list[dict[str, Any]], person_id: int) -> dict[str, Any] | None:
    for inst in instances:
        if int(inst["person_id"]) == person_id:
            return inst
    return None


def write_wide_person_csvs(
    output_dir: Path,
    stem: str,
    timeline: dict[int, list[dict[str, Any]]],
    names: list[str],
    *,
    n_frames: int,
) -> list[Path]:
    """Per-identity wide tables: named 3D, ``rec3d``-style 3D, and 2D markers.

    ``*_rec3d.csv`` uses vailá's ``frame,p1_x,p1_y,p1_z,...`` convention so the
    output drops straight into ``rec3d.py`` / ``viewc3d.py``; ``*_markers.csv``
    uses ``frame,p1_x,p1_y,...`` for REC2D and ``getpixelvideo.py``.
    """
    written: list[Path] = []
    person_ids = _collect_person_ids(timeline)
    n_kpts = len(names)

    named_header = ["frame"]
    for name in names:
        safe = name.replace("-", "_")
        named_header.extend([f"{safe}_x", f"{safe}_y", f"{safe}_z"])
    rec3d_header = ["frame"]
    markers_header = ["frame"]
    for i in range(1, n_kpts + 1):
        rec3d_header.extend([f"p{i}_x", f"p{i}_y", f"p{i}_z"])
        markers_header.extend([f"p{i}_x", f"p{i}_y"])

    for person_id in person_ids:
        named_rows: list[list[str]] = []
        rec3d_rows: list[list[str]] = []
        marker_rows: list[list[str]] = []
        for frame_idx in range(n_frames):
            inst = _instance_for_person(timeline.get(frame_idx, []), person_id)
            named_cells = [str(frame_idx)]
            rec3d_cells = [str(frame_idx)]
            marker_cells = [str(frame_idx)]
            for kpt_idx in range(n_kpts):
                if inst is None or kpt_idx >= len(inst["keypoints_3d"]):
                    named_cells.extend(["", "", ""])
                    rec3d_cells.extend(["", "", ""])
                else:
                    cam = inst["keypoints_3d_cam"][kpt_idx]
                    trio = [_format_cell(cam[0]), _format_cell(cam[1]), _format_cell(cam[2])]
                    named_cells.extend(trio)
                    rec3d_cells.extend(trio)
                if inst is None or kpt_idx >= len(inst["keypoints_2d"]):
                    marker_cells.extend(["", ""])
                else:
                    pt = inst["keypoints_2d"][kpt_idx]
                    marker_cells.extend([_format_cell(pt[0]), _format_cell(pt[1])])
            named_rows.append(named_cells)
            rec3d_rows.append(rec3d_cells)
            marker_rows.append(marker_cells)

        for header, rows, suffix in (
            (named_header, named_rows, "mhr70_3d"),
            (rec3d_header, rec3d_rows, "mhr70_rec3d"),
            (markers_header, marker_rows, "markers"),
        ):
            path = output_dir / f"{stem}_id_{person_id:02d}_{suffix}.csv"
            path.write_text(
                ",".join(header)
                + "\n"
                + "\n".join(",".join(row) for row in rows)
                + ("\n" if rows else ""),
                encoding="utf-8",
            )
            written.append(path)
    return written


def write_predictions_json(
    output_dir: Path,
    stem: str,
    timeline: dict[int, list[dict[str, Any]]],
    names: list[str],
    *,
    meta: dict[str, Any],
) -> Path:
    """Gzipped provenance + full per-instance predictions (mesh excluded)."""
    frames: list[dict[str, Any]] = []
    for frame_idx in sorted(timeline):
        instances = []
        for inst in timeline[frame_idx]:
            instances.append(
                {
                    "person_id": int(inst["person_id"]),
                    "sam_obj_id": int(inst["sam_obj_id"]),
                    "bbox_xyxy": [float(v) for v in inst["bbox"]],
                    "focal_length_px": float(inst["focal_length"]),
                    "cam_t_m": [float(v) for v in inst["cam_t"]],
                    "keypoints_3d_m": [[float(v) for v in p] for p in inst["keypoints_3d"]],
                    "keypoints_3d_cam_m": [[float(v) for v in p] for p in inst["keypoints_3d_cam"]],
                    "keypoints_2d_px": [[float(v) for v in p] for p in inst["keypoints_2d"]],
                }
            )
        frames.append({"frame": frame_idx, "instances": instances})

    payload = {
        "schema": "vaila_sam3dinov3_v1",
        "keypoint_names": names,
        **meta,
        "frames": frames,
    }
    path = output_dir / f"{stem}_sam3dinov3_predictions.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def _write_readme(
    output_dir: Path,
    *,
    video_path: Path,
    sam_dir: Path,
    weights_dir: Path,
    inference_type: str,
    stride: int,
    use_mask: bool,
    bbox_padding: float,
    contour_margin: int,
    focal_px: float | None,
    n_keypoints: int,
) -> Path:
    text = f"""vailá SAM3 + DINOv3 (SAM 3D Body) markerless 3D run
video={video_path}
sam_results={sam_dir}
sam3d_weights={weights_dir}
inference_type={inference_type}
stride={stride}
mask_conditioned={use_mask}
bbox_padding_fraction={bbox_padding}
contour_margin_px={contour_margin}
focal_px={"auto (default FOV)" if focal_px is None else focal_px}
n_keypoints={n_keypoints}
identity_authority=SAM3 obj_id
person_detector=SAM3 (ViTDet disabled)
segmentor=SAM3 contours (SAM2 disabled)

Pipeline
--------
1. SAM 3 segments/tracks people and defines bbox, silhouette, score and obj_id.
2. SAM 3D Body (DINOv3 ViT-H/16+ backbone) receives only those SAM boxes and
   silhouettes; the upstream ViTDet detector and SAM2 segmentor never load.
3. For each person it regresses an MHR (Momentum Human Rig) mesh, 3D joints in
   metres, their 2D reprojection in pixels, the camera translation and focal.
4. person_id is exactly the SAM obj_id, so no second Re-ID can swap identities.

Coordinate systems and units
----------------------------
x_m,y_m,z_m         Root-relative 3D joints, metres, camera axes (OpenCV: +x
                    right, +y down, +z forward/away from the camera).
xcam_m,ycam_m,zcam_m Camera-frame absolute joints = root-relative + cam_t.
                    Use these for inter-person distances and depth.
x_px,y_px           Perspective reprojection into the original full frame.
frames              Zero-based, matching the source video and the SAM outputs.

Scale caveat: monocular depth is metric only up to the assumed camera intrinsics.
Without --focal-px the model falls back to a default FOV (f = sqrt(W^2+H^2)),
so absolute depth carries that assumption. Supply the true focal length in
pixels (or a FOV estimator) whenever absolute distances matter.

Main outputs
------------
<video>_sam3dinov3_overlay.mp4          SAM contour/bbox/ID + reprojected 3D skeleton,
                                         colored by joint side (left=green, right=orange,
                                         center/spine=blue) — same palette as
                                         sam3sapiens2_visualize / sam3dinov3_visualize.
<video>_sam3dinov3_keypoints3d.csv      Long table, root-relative and camera-frame metres.
<video>_sam3dinov3_keypoints2d.csv      Long table, reprojected pixels.
<video>_sam3dinov3_camera.csv           Per-frame focal length, cam_t and bbox.
<video>_sam3dinov3_joint_angles.csv     Long table, local (parent-relative) joint angles
                                         for the model's own 127-joint MHR rig: Euler XYZ
                                         degrees + scalar-first (w,x,y,z) quaternion, from
                                         the model's own regressed rotations (not a
                                         position-only heuristic) -- see joint_kinematics.py.
<video>_id_NN_mhr70_3d.csv              Wide, named columns (nose_x, nose_y, nose_z, ...).
<video>_id_NN_mhr70_rec3d.csv           Wide, vailá rec3d convention (p1_x,p1_y,p1_z, ...).
<video>_id_NN_markers.csv               Wide 2D for REC2D / getpixelvideo.
<video>_sam3dinov3_predictions.json.gz  Full provenance and per-instance predictions.
meshes/frame_NNNNNN.npz                 Only with --save-mesh (vertices + obj_ids).
mesh_faces.npy                          Only with --save-mesh (shared MHR topology).
sam3dinov3_summary.json                 Machine-readable run summary.
README_sam3dinov3.txt                   This file.
FAILED_sam3dinov3.txt                   Exists only when this stage fails.

The original SAM artifacts remain under the path recorded above (or in ./sam3
when SAM 3 was run by this pipeline).

References
----------
SAM 3        https://ai.meta.com/research/sam3/
DINOv3       https://ai.meta.com/research/dinov3/
SAM 3D Body  https://github.com/facebookresearch/sam-3d-body
Weights      https://huggingface.co/{DEFAULT_HF_REPO_ID}
"""
    path = output_dir / "README_sam3dinov3.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _write_failure(
    output_dir: Path,
    video_path: Path,
    reason: str,
    traceback_str: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "SAM3+DINOv3 (SAM 3D Body) FAILED\n"
        f"video={video_path}\n"
        f"timestamp={dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"reason={reason}\n"
    )
    if traceback_str:
        body += f"\nTraceback:\n{traceback_str}\n"
    (output_dir / "FAILED_sam3dinov3.txt").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# core stage
# --------------------------------------------------------------------------- #
def run_sam3d_from_sam(
    video_path: Path,
    output_dir: Path,
    guidance: SamGuidance,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run SAM 3D Body over one video using SAM 3 boxes/silhouettes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    estimator = load_sam3d_estimator(
        args.weights_dir,
        device="cuda",
        fov_estimator_name=str(args.fov_estimator or ""),
    )
    names = keypoint_names(len(MHR70_NAMES))
    edges = skeleton_edges(names)

    cam_int = None
    if args.focal_px is not None:
        cam_int = build_cam_int(float(args.focal_px), width, height)
        _log(f"Using fixed intrinsics: f={float(args.focal_px):.1f} px, principal point centred")

    writer = None
    overlay_path = output_dir / f"{stem}_sam3dinov3_overlay.mp4"
    if not args.no_overlay:
        try:
            # Shared helper: mp4v -> MJPG/XVID -> ffmpeg pipe fallback. It may
            # return an .avi path when mp4v is unavailable on this build.
            writer, overlay_path = _open_sam3_video_writer(
                overlay_path,
                fps,
                (width, height),
                purpose="SAM3+DINOv3 overlay",
            )
        except OSError as exc:
            _log(f"WARNING: could not open overlay writer ({exc}); continuing without overlay")
            writer = None

    mesh_dir = output_dir / "meshes"
    if args.save_mesh:
        mesh_dir.mkdir(parents=True, exist_ok=True)
        faces = np.asarray(estimator.faces)
        np.save(output_dir / "mesh_faces.npy", faces)
        _log(f"Mesh export enabled: {faces.shape[0]} faces per person and frame")

    timeline: dict[int, list[dict[str, Any]]] = {}
    stride = max(1, int(args.stride))
    n_processed = 0
    n_people = 0
    frame_idx = 0
    n_keypoints = len(names)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            instances: list[dict[str, Any]] = []
            frame_guidance = _guidance_for_frame(
                guidance,
                frame_idx,
                min_score=float(args.min_sam_score),
                min_area=int(args.min_sam_area),
                max_persons=int(args.max_persons),
            )
            if frame_idx % stride == 0 and frame_guidance:
                obj_ids, boxes, masks = _frame_batch_from_guidance(
                    frame_guidance,
                    frame_width=width,
                    frame_height=height,
                    bbox_padding=float(args.bbox_padding),
                    contour_margin=int(args.contour_margin),
                    use_mask=not args.no_mask,
                )
                if boxes is not None:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    # The upstream estimator is chatty (one banner per person per
                    # frame); keep our own progress line readable.
                    sink = contextlib.nullcontext() if args.verbose_model else _muted_stdout()
                    with sink:
                        outputs = estimator.process_one_image(
                            rgb,
                            bboxes=boxes,
                            masks=masks,
                            cam_int=cam_int,
                            use_mask=masks is not None,
                            inference_type=str(args.inference_type),
                        )
                    instances = _instances_from_outputs(obj_ids, outputs, frame_idx=frame_idx)
                    if instances:
                        n_keypoints = max(n_keypoints, len(instances[0]["keypoints_3d"]))
                    n_processed += 1
                    n_people += len(instances)

                    if args.save_mesh and instances:
                        vertices = [
                            np.asarray(inst["vertices"], dtype=np.float32)
                            for inst in instances
                            if inst.get("vertices") is not None
                        ]
                        if vertices:
                            np.savez_compressed(
                                mesh_dir / f"frame_{frame_idx:06d}.npz",
                                obj_ids=np.asarray(
                                    [int(i["person_id"]) for i in instances], dtype=np.int32
                                ),
                                vertices=np.stack(vertices, axis=0),
                                cam_t=np.stack([inst["cam_t"] for inst in instances], axis=0),
                            )

            # Vertices are large; drop them once written to keep RAM bounded.
            for inst in instances:
                inst.pop("vertices", None)
            if instances:
                timeline[frame_idx] = instances

            if writer is not None:
                overlay = _draw_sam_guidance(
                    frame_bgr,
                    frame_guidance,
                    draw_ids=not args.no_draw_id,
                )
                overlay = _draw_pose_overlay(
                    overlay,
                    instances,
                    edges,
                    names,
                    draw_ids=not args.no_draw_id,
                )
                writer.write(overlay)

            if frame_idx % 50 == 0:
                _log(
                    f"frame {frame_idx}/{total_frames if total_frames > 0 else '?'} "
                    f"people={len(instances)}"
                )
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        _release_gpu_memory()

    if not timeline:
        raise RuntimeError(
            "SAM 3D Body produced no instances; check the SAM prompt, "
            "--min-sam-score/--min-sam-area filters, or the weights directory."
        )

    names = keypoint_names(n_keypoints)
    written = write_long_keypoints_csvs(output_dir, stem, timeline, names)
    written.append(write_camera_csv(output_dir, stem, timeline))
    joint_angles_path = write_long_joint_angles_csv(output_dir, stem, timeline, names)
    if joint_angles_path is not None:
        written.append(joint_angles_path)
    written.extend(
        write_wide_person_csvs(
            output_dir,
            stem,
            timeline,
            names,
            n_frames=max(frame_idx, max(timeline) + 1),
        )
    )
    meta = {
        "video": str(video_path),
        "sam_results": str(guidance.sam_dir),
        "width": width,
        "height": height,
        "fps": fps,
        "n_frames": frame_idx,
        "stride": stride,
        "inference_type": str(args.inference_type),
        "mask_conditioned": not args.no_mask,
        "focal_px": None if args.focal_px is None else float(args.focal_px),
    }
    written.append(write_predictions_json(output_dir, stem, timeline, names, meta=meta))
    written.append(
        _write_readme(
            output_dir,
            video_path=video_path,
            sam_dir=guidance.sam_dir,
            weights_dir=(args.weights_dir or default_weights_dir()),
            inference_type=str(args.inference_type),
            stride=stride,
            use_mask=not args.no_mask,
            bbox_padding=float(args.bbox_padding),
            contour_margin=int(args.contour_margin),
            focal_px=None if args.focal_px is None else float(args.focal_px),
            n_keypoints=len(names),
        )
    )

    summary = {
        "schema": "vaila_sam3dinov3_video_v1",
        "status": "complete",
        **meta,
        "n_inference_frames": n_processed,
        "n_person_frames": n_people,
        "person_ids": _collect_person_ids(timeline),
        "n_keypoints": len(names),
        "overlay": str(overlay_path) if writer is not None else None,
        "outputs": [str(p) for p in written],
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "sam3dinov3_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _log(
        f"Done {video_path.name}: {n_processed} inference frames, "
        f"{n_people} person-frames, ids={summary['person_ids']} -> {output_dir}"
    )
    return summary


@contextlib.contextmanager
def _muted_stdout():
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        yield


def _release_gpu_memory() -> None:
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# per-video orchestration
# --------------------------------------------------------------------------- #
def _resolve_or_run_sam(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    single_video: bool,
) -> Path:
    """Reuse SAM results when available, otherwise run the SAM 3 stage."""
    if args.worker_sam_dir is not None:
        sam_dir = args.worker_sam_dir.expanduser().resolve()
        _log(f"Reusing SAM3 results: {sam_dir}")
        return sam_dir
    if args.sam_results is not None:
        sam_dir = resolve_sam_results_dir(
            args.sam_results,
            video_path,
            single_video=single_video,
        )
        _log(f"Reusing SAM3 results: {sam_dir}")
        return sam_dir
    local = find_local_sam_dir(output_dir, expected_frames=_video_frame_count(video_path))
    if local is not None:
        _log(f"Reusing SAM3 results already present in the run dir: {local}")
        return local

    sam_dir = output_dir / "sam3"
    run_sam_stage(
        video_path,
        sam_dir,
        prompt=str(args.text),
        prompt_frame=int(args.sam_frame),
        checkpoint=args.sam_checkpoint,
        max_frames=args.sam_max_frames,
        max_input_long_edge=args.sam_max_input_long_edge,
        keep_masks=bool(args.keep_sam_masks),
    )
    return sam_dir


def _process_one_video(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    single_video: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sam_dir = _resolve_or_run_sam(
            video_path,
            output_dir,
            args,
            single_video=single_video,
        )
        guidance = load_sam_guidance(sam_dir)
        summary = run_sam3d_from_sam(video_path, output_dir, guidance, args)
    except Exception as exc:
        _write_failure(output_dir, video_path, str(exc), traceback.format_exc())
        raise
    with contextlib.suppress(OSError):
        (output_dir / "FAILED_sam3dinov3.txt").unlink(missing_ok=True)
    return summary


def load_completed_summary(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "sam3dinov3_summary.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("status") != "complete":
        return None
    return data


def _add_if_value(cmd: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def build_worker_command(
    video_path: Path,
    output_parent: Path,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    sam_dir: Path | None,
) -> list[str]:
    """One isolated subprocess per video keeps CUDA clean after an OOM."""
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "-i",
        str(video_path),
        "-o",
        str(output_parent),
        "--worker-output-dir",
        str(output_dir),
        "-t",
        str(args.text),
        "--device",
        str(int(args.device)),
        "--stride",
        str(int(args.stride)),
        "--inference-type",
        str(args.inference_type),
        "--bbox-padding",
        str(float(args.bbox_padding)),
        "--contour-margin",
        str(int(args.contour_margin)),
        "--max-persons",
        str(int(args.max_persons)),
        "--min-sam-score",
        str(float(args.min_sam_score)),
        "--min-sam-area",
        str(int(args.min_sam_area)),
        # Forwarded so a worker that still has to run SAM itself keeps the
        # user's prompt frame instead of silently falling back to frame 0.
        "--sam-frame",
        str(int(args.sam_frame)),
    ]
    _add_if_value(cmd, "--worker-sam-dir", sam_dir)
    _add_if_value(cmd, "--weights-dir", args.weights_dir)
    _add_if_value(cmd, "--focal-px", args.focal_px)
    _add_if_value(cmd, "--fov-estimator", args.fov_estimator or None)
    _add_if_value(cmd, "--sam-checkpoint", args.sam_checkpoint)
    _add_if_value(cmd, "--sam-max-frames", args.sam_max_frames)
    _add_if_value(cmd, "--sam-max-input-long-edge", args.sam_max_input_long_edge)
    if args.no_mask:
        cmd.append("--no-mask")
    if args.no_overlay:
        cmd.append("--no-overlay")
    if args.no_draw_id:
        cmd.append("--no-draw-id")
    if args.save_mesh:
        cmd.append("--save-mesh")
    if args.keep_sam_masks:
        cmd.append("--keep-sam-masks")
    if args.verbose_model:
        cmd.append("--verbose-model")
    return cmd


def _build_dry_run_report(
    videos: list[Path],
    output_base: Path,
    args: argparse.Namespace,
) -> list[str]:
    weights_dir = (args.weights_dir or default_weights_dir()).expanduser().resolve()
    try:
        ckpt, mhr = resolve_sam3d_assets(args.weights_dir)
        weights_state = f"OK ({ckpt.name}, {mhr.name})"
    except FileNotFoundError as exc:
        weights_state = f"MISSING — {str(exc).splitlines()[0]}"
    lines = [
        "vailá SAM3 + DINOv3 (SAM 3D Body) — dry run",
        f"timestamp        : {dt.datetime.now().isoformat(timespec='seconds')}",
        f"output_base      : {output_base}",
        f"videos           : {len(videos)}",
        f"sam3d_weights    : {weights_dir}",
        f"weights_state    : {weights_state}",
        f"sam_results      : {args.sam_results or '(run SAM 3 per video)'}",
        f"prompt           : {args.text}",
        f"inference_type   : {args.inference_type}",
        f"mask_conditioned : {not args.no_mask}",
        f"stride           : {args.stride}",
        f"bbox_padding     : {args.bbox_padding}",
        f"contour_margin   : {args.contour_margin}",
        f"max_persons      : {args.max_persons}",
        f"focal_px         : {args.focal_px if args.focal_px is not None else 'auto (default FOV)'}",
        f"save_mesh        : {args.save_mesh}",
        f"overlay          : {not args.no_overlay}",
        "",
        "Planned per-video work:",
    ]
    for video in videos:
        out_dir = output_base / video.stem
        frames = _video_frame_count(video)
        lines.append(f"  {video.name}  frames={frames if frames > 0 else '?'}  ->  {out_dir}")
    lines.append("")
    lines.append("No GPU inference was executed (--dry-run).")
    return lines


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def _format_gui_cli(settings: Sam3dGuiSettings) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "vaila/sam3dinov3.py",
        "-i",
        str(settings.input_path),
    ]
    if settings.resume is not None:
        cmd.extend(["--resume", str(settings.resume)])
    else:
        assert settings.output_parent is not None
        cmd.extend(["-o", str(settings.output_parent)])
    cmd.extend(
        [
            "-t",
            settings.prompt,
            "--device",
            str(settings.device),
            "--stride",
            str(settings.stride),
            "--inference-type",
            settings.inference_type,
            "--bbox-padding",
            str(settings.bbox_padding),
            "--contour-margin",
            str(settings.contour_margin),
            "--max-persons",
            str(settings.max_persons),
        ]
    )
    if settings.sam_results is not None:
        cmd.extend(["--sam-results", str(settings.sam_results)])
    if settings.weights_dir is not None:
        cmd.extend(["--weights-dir", str(settings.weights_dir)])
    if settings.focal_px is not None:
        cmd.extend(["--focal-px", str(settings.focal_px)])
    if not settings.use_mask:
        cmd.append("--no-mask")
    if not settings.save_overlay:
        cmd.append("--no-overlay")
    if settings.save_mesh:
        cmd.append("--save-mesh")
    return cmd


def run_sam3dinov3(existing_root: Any | None = None) -> None:
    """Open the SAM3+DINOv3 3D settings GUI and launch its reproducible CLI."""
    owns_root = existing_root is None
    root = existing_root if existing_root is not None else tk.Tk()
    if owns_root:
        root.withdraw()
    _prepare_gui_root(root, owns_root=owns_root)

    class Sam3dDialog(tk.Toplevel):
        def __init__(self, master: tk.Misc) -> None:
            super().__init__(master)
            self.title("vailá — SAM3 + DINOv3 (SAM 3D Body) markerless 3D")
            self.result: Sam3dGuiSettings | None = None
            self.transient(master)  # ty: ignore[no-matching-overload]
            self.resizable(False, False)

            self.input_var = tk.StringVar()
            self.output_var = tk.StringVar()
            self.sam_var = tk.StringVar()
            self.weights_var = tk.StringVar(value=str(default_weights_dir()))
            self.prompt_var = tk.StringVar(value="person")
            self.device_var = tk.StringVar(value="0")
            self.stride_var = tk.StringVar(value=str(DEFAULT_STRIDE))
            self.inference_var = tk.StringVar(value=DEFAULT_INFERENCE_TYPE)
            self.padding_var = tk.StringVar(value=str(DEFAULT_BBOX_PADDING))
            self.margin_var = tk.StringVar(value=str(DEFAULT_CONTOUR_MARGIN_PX))
            self.max_persons_var = tk.StringVar(value=str(DEFAULT_MAX_PERSONS))
            self.focal_var = tk.StringVar()
            self.mask_var = tk.BooleanVar(value=True)
            self.overlay_var = tk.BooleanVar(value=True)
            self.mesh_var = tk.BooleanVar(value=False)

            frame = ttk.Frame(self, padding=12)
            frame.grid(row=0, column=0, sticky="nsew")
            row = 0

            ttk.Label(
                frame,
                text=(
                    "SAM 3 detects, segments and keeps identities; "
                    "SAM 3D Body (DINOv3) lifts each person to 3D.\n"
                    "Requires an NVIDIA CUDA GPU and the gated "
                    f"{DEFAULT_HF_REPO_ID} weights."
                ),
                justify="left",
            ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 10))
            row += 1

            def _path_row(label: str, var: tk.StringVar, browse: Any) -> None:
                nonlocal row
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
                ttk.Entry(frame, textvariable=var, width=52).grid(
                    row=row, column=1, columnspan=2, sticky="we", pady=3
                )
                ttk.Button(frame, text="Browse…", command=browse).grid(
                    row=row, column=3, sticky="w", padx=(6, 0)
                )
                row += 1

            ttk.Label(frame, text="Input video/folder").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(frame, textvariable=self.input_var, width=52).grid(
                row=row, column=1, columnspan=2, sticky="we", pady=3
            )
            ttk.Button(frame, text="Dir…", command=self._browse_input_dir).grid(
                row=row, column=3, sticky="w", padx=(6, 0)
            )
            row += 1
            ttk.Button(frame, text="File…", command=self._browse_input_file).grid(
                row=row, column=3, sticky="w", padx=(6, 0), pady=(0, 3)
            )
            row += 1

            _path_row("Output parent", self.output_var, self._browse_output)
            _path_row("Existing SAM results (optional)", self.sam_var, self._browse_sam)
            _path_row("SAM 3D Body weights dir", self.weights_var, self._browse_weights)

            def _entry_row(label: str, var: tk.StringVar, width: int = 12) -> None:
                nonlocal row
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
                ttk.Entry(frame, textvariable=var, width=width).grid(
                    row=row, column=1, sticky="w", pady=3
                )
                row += 1

            _entry_row("SAM 3 text prompt", self.prompt_var, width=28)
            _entry_row("CUDA device index", self.device_var)
            _entry_row("Frame stride", self.stride_var)
            _entry_row("BBox padding fraction", self.padding_var)
            _entry_row("Contour margin (px)", self.margin_var)
            _entry_row("Max persons per frame", self.max_persons_var)
            _entry_row("Focal length px (blank = auto)", self.focal_var)

            ttk.Label(frame, text="Inference type").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Combobox(
                frame,
                textvariable=self.inference_var,
                values=list(INFERENCE_TYPES),
                state="readonly",
                width=10,
            ).grid(row=row, column=1, sticky="w", pady=3)
            row += 1

            ttk.Checkbutton(
                frame,
                text="Mask-conditioned (use SAM silhouettes)",
                variable=self.mask_var,
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            row += 1
            ttk.Checkbutton(frame, text="Write overlay video", variable=self.overlay_var).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=2
            )
            row += 1
            ttk.Checkbutton(
                frame,
                text="Save MHR meshes (large: ~220 KB per person and frame)",
                variable=self.mesh_var,
            ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
            row += 1

            buttons = ttk.Frame(frame)
            buttons.grid(row=row, column=0, columnspan=4, sticky="e", pady=(12, 0))
            ttk.Button(buttons, text="Help", command=self._open_help).pack(side="left", padx=4)
            ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=4)
            ttk.Button(buttons, text="Run", command=self._on_run).pack(side="left", padx=4)

        def _browse_input_dir(self) -> None:
            path = filedialog.askdirectory(parent=self, title="Select folder with videos")
            if path:
                self.input_var.set(path)

        def _browse_input_file(self) -> None:
            path = filedialog.askopenfilename(
                parent=self,
                title="Select a video file",
                filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"), ("All", "*.*")],
            )
            if path:
                self.input_var.set(path)

        def _browse_output(self) -> None:
            path = filedialog.askdirectory(parent=self, title="Select output parent")
            if path:
                self.output_var.set(path)

        def _browse_sam(self) -> None:
            path = filedialog.askdirectory(
                parent=self, title="Select processed_sam_* or per-video SAM directory"
            )
            if path:
                self.sam_var.set(path)

        def _browse_weights(self) -> None:
            path = filedialog.askdirectory(
                parent=self, title="Select the sam-3d-dinov3 weights directory"
            )
            if path:
                self.weights_var.set(path)

        def _open_help(self) -> None:
            if _help_path().is_file():
                webbrowser.open_new_tab(_help_path().as_uri())
            else:
                webbrowser.open_new_tab(f"https://huggingface.co/{DEFAULT_HF_REPO_ID}")

        def _on_run(self) -> None:
            try:
                input_path = Path(self.input_var.get().strip()).expanduser()
                if not self.input_var.get().strip() or not input_path.exists():
                    raise ValueError("Select an existing input video or folder (Dir… / File…).")
                videos = _find_videos(input_path)
                if not videos:
                    raise ValueError(f"No supported videos found under: {input_path}")
                output_raw = self.output_var.get().strip()
                output_parent = Path(output_raw).expanduser() if output_raw else None
                if output_parent is None:
                    raise ValueError("Select an output parent folder.")
                sam_raw = self.sam_var.get().strip()
                sam_results = Path(sam_raw).expanduser() if sam_raw else None
                if sam_results is not None and not sam_results.exists():
                    raise ValueError(f"Existing SAM results path not found: {sam_results}")
                weights_raw = self.weights_var.get().strip()
                weights_dir = Path(weights_raw).expanduser() if weights_raw else None
                focal_raw = self.focal_var.get().strip()
                focal_px = float(focal_raw) if focal_raw else None
                if focal_px is not None and focal_px <= 0:
                    raise ValueError("Focal length must be a positive number of pixels.")
                result = Sam3dGuiSettings(
                    input_path=input_path,
                    output_parent=output_parent,
                    resume=None,
                    sam_results=sam_results,
                    prompt=self.prompt_var.get().strip() or "person",
                    device=max(0, int(self.device_var.get())),
                    stride=max(1, int(self.stride_var.get())),
                    bbox_padding=max(0.0, float(self.padding_var.get())),
                    contour_margin=max(0, int(self.margin_var.get())),
                    max_persons=max(1, int(self.max_persons_var.get())),
                    inference_type=self.inference_var.get().strip() or DEFAULT_INFERENCE_TYPE,
                    use_mask=bool(self.mask_var.get()),
                    save_overlay=bool(self.overlay_var.get()),
                    save_mesh=bool(self.mesh_var.get()),
                    focal_px=focal_px,
                    weights_dir=weights_dir,
                )
            except ValueError as exc:
                messagebox.showerror("SAM3+DINOv3 3D", str(exc), parent=self)
                return
            _log(
                f"Queued {len(videos)} video(s): "
                + ", ".join(v.name for v in videos[:8])
                + ("…" if len(videos) > 8 else "")
            )
            self.result = result
            self.destroy()

    _log("Opening the settings window; bare CLI without -i/-o intentionally uses this GUI.")
    dialog = Sam3dDialog(root)
    root.wait_window(dialog)
    settings = dialog.result
    if owns_root:
        root.destroy()
    if settings is None:
        return
    cli = _format_gui_cli(settings)
    print_gui_cli_mirror("vaila/sam3dinov3", cli)
    launch = [sys.executable, "-u", str(Path(__file__).resolve())] + cli[5:]
    gui_result = run_isolated_gpu_subprocess(launch, device=int(settings.device))
    raise SystemExit(gui_result.returncode)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SAM3-guided SAM 3D Body (DINOv3): monocular markerless 3D human "
            "mesh and keypoints. SAM 3 owns bbox/contour/ID."
        )
    )
    parser.add_argument("-i", "--input", type=Path, help="Input video or non-recursive folder")
    parser.add_argument("-o", "--output", type=Path, help="Output parent directory")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Resume an existing processed_sam3dinov3_* directory: skip completed "
            "videos, retry failures, and reuse existing sam3/sam_tracks.csv."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Ignore any matching previous processed_sam3dinov3_* run under "
            "--output and start a new timestamped output directory. Default "
            "behavior auto-resumes a matching run without needing --resume."
        ),
    )
    parser.add_argument("-t", "--text", default="person", help="SAM3 text prompt")
    parser.add_argument(
        "--sam-results",
        type=Path,
        default=None,
        help="Reuse a processed_sam_* batch parent or a per-video SAM output dir",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help=f"SAM 3D Body weights dir (default: vaila/{DEFAULT_WEIGHTS_SUBDIR})",
    )
    parser.add_argument(
        "--inference-type",
        default=DEFAULT_INFERENCE_TYPE,
        choices=INFERENCE_TYPES,
        help="full = body+hand decoders, body = body only, hand = hand only",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--bbox-padding", type=float, default=DEFAULT_BBOX_PADDING)
    parser.add_argument("--contour-margin", type=int, default=DEFAULT_CONTOUR_MARGIN_PX)
    parser.add_argument("--min-sam-score", type=float, default=DEFAULT_MIN_SAM_SCORE)
    parser.add_argument("--min-sam-area", type=int, default=DEFAULT_MIN_SAM_AREA)
    parser.add_argument("--max-persons", type=int, default=DEFAULT_MAX_PERSONS)
    parser.add_argument(
        "--focal-px",
        type=float,
        default=None,
        help=(
            "Known focal length in pixels for metric depth. Without it the model "
            "assumes a default FOV, so absolute depth inherits that assumption."
        ),
    )
    parser.add_argument(
        "--fov-estimator",
        default="",
        help="Optional upstream FOV estimator name (e.g. moge2); needs extra weights",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Use SAM boxes/IDs but skip mask-conditioned inference",
    )
    parser.add_argument(
        "--save-mesh",
        action="store_true",
        help="Write per-frame MHR vertices to meshes/*.npz (large)",
    )
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-draw-id", action="store_true")
    parser.add_argument(
        "--verbose-model",
        action="store_true",
        help="Do not silence the upstream per-frame model chatter",
    )
    parser.add_argument("--sam-frame", type=int, default=0)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-max-frames", type=int, default=None)
    parser.add_argument("--sam-max-input-long-edge", type=int, default=None)
    parser.add_argument("--keep-sam-masks", action="store_true")
    parser.add_argument(
        "--no-isolate-batch",
        action="store_true",
        help="Debug: process videos in the coordinator process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and weights, print the plan, run no GPU inference",
    )
    parser.add_argument("--open-help", action="store_true")
    parser.add_argument("--worker-output-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sam-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.open_help:
        if _help_path().is_file():
            webbrowser.open_new_tab(_help_path().as_uri())
        else:
            webbrowser.open_new_tab(f"https://huggingface.co/{DEFAULT_HF_REPO_ID}")
        return
    if args.input is None and args.output is None and args.resume is None:
        run_sam3dinov3()
        return
    if args.input is None or (args.output is None and args.resume is None):
        parser.error("--input and either --output or --resume must be supplied")
    if not 0.0 <= args.min_sam_score <= 1.0:
        parser.error("--min-sam-score must be in [0,1]")
    if args.bbox_padding < 0:
        parser.error("--bbox-padding must be >= 0")
    if args.focal_px is not None and args.focal_px <= 0:
        parser.error("--focal-px must be > 0")
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    if args.fresh and args.resume is not None:
        parser.error("--fresh and --resume are mutually exclusive")

    if not args.dry_run:
        # SAM 3D Body / DINOv3 weights only; the SAM3 *video* checkpoint used by the
        # per-video subprocess is resolved/downloaded by vaila_sam.py's own preflight.
        ensure_sam3d_assets(args.weights_dir)

    input_path = args.input.expanduser().resolve()
    output_parent = (
        args.output.expanduser().resolve()
        if args.output is not None
        else args.resume.expanduser().resolve().parent
    )
    videos = _find_videos(input_path)
    if not videos:
        parser.error(f"no supported video found under {input_path}")

    if args.worker_output_dir is not None:
        if len(videos) != 1:
            parser.error("internal worker mode requires one input video")
        _process_one_video(
            videos[0],
            args.worker_output_dir.expanduser().resolve(),
            args,
            single_video=True,
        )
        return

    if args.resume is not None:
        output_base = args.resume.expanduser().resolve()
        if not output_base.is_dir():
            parser.error(f"resume directory does not exist: {output_base}")
        is_resume = True
        _log(f"Resume mode: reusing {output_base}")
    else:
        output_base, is_resume = resolve_auto_resume_output_base(
            output_parent, input_path, "sam3dinov3", fresh=args.fresh
        )
        if is_resume:
            _log(f"Auto-resume: found matching run, reusing {output_base}")
    resume_flag = args.resume is not None or is_resume
    output_base.mkdir(parents=True, exist_ok=True)
    write_batch_input_marker(output_base, input_path, "sam3dinov3")

    if args.dry_run:
        lines = _build_dry_run_report(videos, output_base, args)
        report = output_base / "SAM3DINOV3_DRY_RUN.txt"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        print(f"Dry-run report: {report}", flush=True)
        return

    completed_count = sum(
        1 for video in videos if load_completed_summary(output_base / video.stem) is not None
    )
    _log(
        f"Resume: {completed_count}/{len(videos)} videos already completed, "
        f"{len(videos) - completed_count} remaining"
    )

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    failed: list[str] = []
    summaries: list[dict[str, Any]] = []
    use_isolation = not args.no_isolate_batch
    single_video = len(videos) == 1

    for index, video in enumerate(videos, start=1):
        output_dir = output_base / video.stem
        if resume_flag:
            done = load_completed_summary(output_dir)
            if done is not None:
                summaries.append(done)
                _log(f"[SKIP] Already processed: {video.name}")
                continue

        sam_dir: Path | None = None
        if args.sam_results is not None:
            try:
                sam_dir = resolve_sam_results_dir(
                    args.sam_results, video, single_video=single_video
                )
            except FileNotFoundError as exc:
                failed.append(f"{video.name}: {exc}")
                _log(f"ERROR: {failed[-1]}")
                continue
        else:
            sam_dir = find_local_sam_dir(output_dir, expected_frames=_video_frame_count(video))
            if sam_dir is None:
                prepare_sam_rerun_dir(output_dir)

        _log(f"Video {index}/{len(videos)}: {video.name}")
        try:
            if use_isolation:
                cmd = build_worker_command(
                    video,
                    output_parent,
                    output_dir,
                    args,
                    sam_dir=sam_dir,
                )
                _log("Isolated worker CLI: " + shlex.join(cmd))
                gpu_result = run_isolated_gpu_subprocess(
                    cmd,
                    env=os.environ.copy(),
                    device=int(args.device),
                    log=lambda message: _log(f"Worker {message}"),
                )
                if gpu_result.returncode != 0:
                    raise RuntimeError(f"worker subprocess exit={gpu_result.returncode}")
                completed = load_completed_summary(output_dir)
                if completed is None:
                    raise RuntimeError(
                        "worker exited 0 without a complete summary: "
                        f"{output_dir / 'sam3dinov3_summary.json'}"
                    )
                summaries.append(completed)
            else:
                worker_args = argparse.Namespace(**vars(args))
                worker_args.worker_sam_dir = sam_dir
                summaries.append(
                    _process_one_video(
                        video,
                        output_dir,
                        worker_args,
                        single_video=single_video,
                    )
                )
        except Exception as exc:
            failed.append(f"{video.name}: {exc}")
            _log(f"ERROR: {failed[-1]}")

    batch_summary = {
        "schema": "vaila_sam3dinov3_batch_v1",
        "input": str(input_path),
        "output": str(output_base),
        "succeeded": len(summaries),
        "failed": failed,
        "videos": summaries,
    }
    (output_base / "sam3dinov3_batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"Batch done: {len(summaries)}/{len(videos)} succeeded -> {output_base}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
