"""
Project: vailá
Script: sapiens2_3d.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila

Creation Date: 06 August 2026
Update Date: 16 August 2026
Version: 0.3.106

Description:
    Monocular markerless **3D** human mesh/skeleton recovery, complementing
    ``sam3dinov3.py`` by using Meta Sapiens2's 308-keypoint 2D pose as extra
    guidance into the **same** 3D lifter (SAM 3D Body, DINOv3 backbone) --
    NOT a second, independent 3D pipeline.

    Scoping note (read before extending): Sapiens2 as vendored in this repo
    (``vaila_sapiens.py``) is a 2D-only top-down pose model -- x, y, score,
    no depth/normal/mesh head. It cannot produce 3D on its own. The vendored
    ``sam_3d_body_estimator.py``'s ``process_one_image()`` accepts only
    ``bboxes``/``masks``/``cam_int`` as external guidance at inference time;
    its internal ``keypoint_prompt_sampler`` machinery is a TRAINING-time
    self-refinement loop that reads ground-truth ``batch["keypoints_2d"]``,
    a field the inference-time batch builder never populates -- reusing it
    would mean patching unsupported internal state in vendored upstream
    code. So the guidance this module adds is **bbox tightening only**:
    Sapiens2's 308 keypoints (already computed by ``sam3sapiens2.py``) are
    used to compute a tighter, keypoint-derived person bbox when enough
    confident keypoints are available, replacing SAM3's mask-derived bbox
    for that frame/person; otherwise it falls back to the SAM3 bbox
    unchanged. The mesh regressor itself, and everything downstream of
    ``process_one_image()``, is identical to ``sam3dinov3.py``.

    Pipeline:
        1. Reuses an existing ``sam3sapiens2.py`` combined run (SAM3
           bbox/contour/ID authority + Sapiens2 308-keypoint 2D pose), or
           runs that stage itself from a raw SAM3 result if only
           ``--sam-results`` is given.
        2. Per frame/person, tightens the SAM3 bbox using Sapiens2 keypoints
           (``tighten_bbox_with_sapiens2``), with an IoU sanity check against
           the SAM3 bbox so a misassigned/garbage keypoint set cannot send
           the crop somewhere unrelated to the actual person.
        3. Feeds the resulting bbox + the SAM3 contour mask into
           ``SAM3DBodyEstimator.process_one_image(bboxes=..., masks=...)``,
           the exact call ``sam3dinov3.py`` already makes.
        4. Writes the same MHR70 long/wide CSV + camera CSV + optional mesh
           family ``sam3dinov3.py`` writes (by importing and calling its
           writer functions directly -- no duplication), plus one companion
           CSV (``*_sapiens2_3d_guidance.csv``) recording, per frame/person,
           whether that frame was actually guided by Sapiens2 keypoints.
        5. Optional: when ``--dlt3d`` is given, each person's monocular
           camera-frame output is automatically placed into the
           DLT3D-calibrated lab frame by calling
           ``monocular_dlt_align.align_monocular_to_world()`` once per
           person (imported, never duplicated) -> ``dlt_world/id_NN/``.
           Absent ``--dlt3d``, behavior is unchanged.

    This module never modifies ``sam3dinov3.py``'s, ``sam3sapiens2.py``'s, or
    ``monocular_dlt_align.py``'s shared functions or output schemas -- it
    only imports and calls them, so existing pipelines are unaffected.

    Input resolution (v0.3.101): pointing ``--sapiens2-results``/``-i`` at a
    plausible-but-wrong directory (e.g. a ``sam3sapiens2_visualize.py``
    single-ID rerender output instead of the combined run) auto-resolves
    when unambiguous -- see ``find_sapiens2_predictions_json()`` and
    ``_auto_locate_video_from_results()`` -- rather than failing with a bare
    "not found". A genuine video/predictions-JSON mismatch still raises
    (frame-by-frame guidance must never silently misalign against the wrong
    footage).

Scope (MVP, documented rather than hidden):
    - Processes videos sequentially in one process (no per-video subprocess
      isolation like ``sam3dinov3.py``'s batch coordinator); call
      ``_release_gpu_memory()`` between videos keeps VRAM bounded for the
      video counts this pipeline currently targets. Revisit if OOM appears
      across large batches.
    - Cannot start from a completely raw video with no existing SAM3 or
      SAM3+Sapiens2 run -- point ``--sapiens2-results`` at an existing
      ``sam3sapiens2.py`` output, or ``--sam-results`` at a raw SAM3 output
      (this module runs the Sapiens2 stage itself in that case).

Setup:
    bash bin/setup_fifa_sam3d.sh     # clones sam-3d-body + gated DINOv3 weights
    uv sync --extra sam               # SAM 3 stack (CUDA)
    uv sync --extra sapiens           # Sapiens2 stack (CUDA); bash bin/setup_sapiens2.sh after

Usage:
    # Reuse an existing SAM3+Sapiens2 combined run:
    uv run python -u vaila/sapiens2_3d.py \
        -i /path/to/video.mp4 -o /path/to/output \
        --sapiens2-results /path/to/processed_sam3sapiens2_YYYYMMDD_HHMMSS

    # Only a raw SAM3 run exists -- run the Sapiens2 stage first, then this:
    uv run python -u vaila/sapiens2_3d.py \
        -i /path/to/video.mp4 -o /path/to/output \
        --sam-results /path/to/processed_sam_YYYYMMDD_HHMMSS

    # Auto-chain into the DLT3D-calibrated lab frame (per person):
    uv run python -u vaila/sapiens2_3d.py \
        -i /path/to/video.mp4 -o /path/to/output \
        --sapiens2-results /path/to/processed_sam3sapiens2_YYYYMMDD_HHMMSS \
        --dlt3d /path/to/camera.dlt3d --ref3d /path/to/control_points.ref3d \
        --save-mesh --export-mesh obj

    # GUI: omit arguments, or Frame B -> Markerless 3D -> Sapiens2 3D Pose

Runtime:
    Requires an NVIDIA CUDA GPU (the upstream SAM 3D Body estimator moves its
    batch to ``cuda`` unconditionally, same as ``sam3dinov3.py``).

License:
    This program is licensed under the GNU Affero General Public License v3.0.
    SAM 3, SAM 3D Body, and Sapiens2 weights keep their respective Meta licenses.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import os
import re
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
    from .geometric_reid import bbox_iou_xyxy
    from .gpu_subprocess import run_isolated_gpu_subprocess
    from .monocular_dlt_align import (
        DEFAULT_EXPORT_MESH as DLT_DEFAULT_EXPORT_MESH,
    )
    from .monocular_dlt_align import (
        DEFAULT_PLACEMENT_ORIGIN_MARKERS as DLT_DEFAULT_ORIGIN_MARKERS,
    )
    from .monocular_dlt_align import (
        DEFAULT_SMOOTH_HZ as DLT_DEFAULT_SMOOTH_HZ,
    )
    from .monocular_dlt_align import (
        align_monocular_to_world,
    )
    from .sam3dinov3 import (
        DEFAULT_HF_REPO_ID,
        MHR70_NAMES,
        _collect_person_ids,
        _draw_pose_overlay,
        _instances_from_outputs,
        _muted_stdout,
        _open_sam3_video_writer,
        _release_gpu_memory,
        build_cam_int,
        default_weights_dir,
        keypoint_names,
        load_sam3d_estimator,
        skeleton_edges,
        write_camera_csv,
        write_long_joint_angles_csv,
        write_long_keypoints_csvs,
        write_predictions_json,
        write_wide_person_csvs,
    )
    from .sam3sapiens2 import (
        DEFAULT_BBOX_PADDING,
        DEFAULT_CONTOUR_MARGIN_PX,
        DEFAULT_MAX_PERSONS,
        DEFAULT_MIN_SAM_AREA,
        DEFAULT_MIN_SAM_SCORE,
        SamGuidance,
        _contour_mask,
        _draw_sam_guidance,
        _find_videos,
        _guidance_for_frame,
        _pose_bbox_from_sam,
        _prepare_gui_root,
        _video_frame_count,
        load_sam_guidance,
        resolve_sam_results_dir,
        run_sapiens_from_sam,
    )
except ImportError:  # standalone execution
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    from geometric_reid import bbox_iou_xyxy  # ty: ignore[unresolved-import]
    from gpu_subprocess import run_isolated_gpu_subprocess  # ty: ignore[unresolved-import]
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        DEFAULT_EXPORT_MESH as DLT_DEFAULT_EXPORT_MESH,
    )
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        DEFAULT_PLACEMENT_ORIGIN_MARKERS as DLT_DEFAULT_ORIGIN_MARKERS,
    )
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        DEFAULT_SMOOTH_HZ as DLT_DEFAULT_SMOOTH_HZ,
    )
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        align_monocular_to_world,
    )
    from sam3dinov3 import (  # ty: ignore[unresolved-import]
        DEFAULT_HF_REPO_ID,
        MHR70_NAMES,
        _collect_person_ids,
        _draw_pose_overlay,
        _instances_from_outputs,
        _muted_stdout,
        _open_sam3_video_writer,
        _release_gpu_memory,
        build_cam_int,
        default_weights_dir,
        keypoint_names,
        load_sam3d_estimator,
        skeleton_edges,
        write_camera_csv,
        write_long_joint_angles_csv,
        write_long_keypoints_csvs,
        write_predictions_json,
        write_wide_person_csvs,
    )
    from sam3sapiens2 import (  # ty: ignore[unresolved-import]
        DEFAULT_BBOX_PADDING,
        DEFAULT_CONTOUR_MARGIN_PX,
        DEFAULT_MAX_PERSONS,
        DEFAULT_MIN_SAM_AREA,
        DEFAULT_MIN_SAM_SCORE,
        SamGuidance,
        _contour_mask,
        _draw_sam_guidance,
        _find_videos,
        _guidance_for_frame,
        _pose_bbox_from_sam,
        _prepare_gui_root,
        _video_frame_count,
        load_sam_guidance,
        resolve_sam_results_dir,
        run_sapiens_from_sam,
    )

# --------------------------------------------------------------------------- #
# Sapiens2-guidance constants (this module only; do not confuse with SAM3's
# own DEFAULT_BBOX_PADDING above, which pads the *SAM* bbox before masking).
# --------------------------------------------------------------------------- #
#: Minimum number of confident Sapiens2 keypoints required before trusting a
#: keypoint-derived bbox over the SAM3 one. Below this, fall back to SAM3.
DEFAULT_MIN_GUIDANCE_KEYPOINTS = 4
#: Sapiens2 keypoint confidence threshold (its own [0,1] score column).
DEFAULT_KPT_SCORE_THRESH = 0.3
#: Fractional padding added around the tight keypoint bounding box.
DEFAULT_KPT_BBOX_PADDING_FRAC = 0.08
#: Minimum IoU the keypoint-derived bbox must share with the SAM3 bbox to be
#: trusted -- guards against a misassigned/garbage keypoint set (e.g. a
#: Sapiens2 detection that drifted onto a different, nearby person) silently
#: sending the 3D lifter's crop somewhere unrelated to the tracked person.
DEFAULT_MIN_SANITY_IOU = 0.05
DEFAULT_STRIDE = 1
DEFAULT_INFERENCE_TYPE = "full"
INFERENCE_TYPES = ("full", "body", "hand")


def _log(message: str) -> None:
    # ``>>`` instead of ``[brackets]``: absl logging (pulled in by mediapipe /
    # opencv) silently swallows bracketed stdout prefixes.
    # Overnight/detached runs must not die on a dropped terminal (EIO/BrokenPipe).
    with contextlib.suppress(OSError, BrokenPipeError):
        print(f">> vaila/sapiens2_3d: {message}", flush=True)


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _help_path() -> Path:
    return _module_dir() / "help" / "sapiens2_3d.html"


# --------------------------------------------------------------------------- #
# Bbox tightening (pure math -- unit-tested without any GPU/model dependency)
# --------------------------------------------------------------------------- #
def sapiens2_keypoint_bbox(
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    *,
    score_thresh: float,
    frame_width: int,
    frame_height: int,
    padding_frac: float,
) -> tuple[np.ndarray | None, int]:
    """Bounding box around confident Sapiens2 keypoints, padded and clipped.

    Returns ``(None, 0)`` when fewer than 1 keypoint clears ``score_thresh``
    or any are non-finite -- callers decide the minimum count to trust.
    """
    pts = np.asarray(keypoints_xy, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(scores, dtype=np.float64).reshape(-1)
    if pts.shape[0] != conf.shape[0]:
        raise ValueError(f"keypoints_xy has {pts.shape[0]} rows but scores has {conf.shape[0]}")
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(conf) & (conf >= score_thresh)
    num_used = int(valid.sum())
    if num_used == 0:
        return None, 0

    used = pts[valid]
    x0, y0 = used.min(axis=0)
    x1, y1 = used.max(axis=0)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    pad_x = width * padding_frac
    pad_y = height * padding_frac
    x0 = max(0.0, x0 - pad_x)
    y0 = max(0.0, y0 - pad_y)
    x1 = min(float(frame_width), x1 + pad_x)
    y1 = min(float(frame_height), y1 + pad_y)
    return np.array([x0, y0, x1, y1], dtype=np.float32), num_used


def tighten_bbox_with_sapiens2(
    sam_bbox_xyxy: np.ndarray,
    keypoints_xy: np.ndarray | None,
    scores: np.ndarray | None,
    *,
    frame_width: int,
    frame_height: int,
    min_guidance_keypoints: int = DEFAULT_MIN_GUIDANCE_KEYPOINTS,
    score_thresh: float = DEFAULT_KPT_SCORE_THRESH,
    padding_frac: float = DEFAULT_KPT_BBOX_PADDING_FRAC,
    min_sanity_iou: float = DEFAULT_MIN_SANITY_IOU,
) -> tuple[np.ndarray, bool, int]:
    """Replace ``sam_bbox_xyxy`` with a Sapiens2-keypoint-derived bbox when safe.

    Returns ``(bbox_xyxy, guided, num_keypoints_used)``. Falls back to the
    unmodified SAM3 bbox (``guided=False``) when: Sapiens2 keypoints are
    missing/None; fewer than ``min_guidance_keypoints`` clear
    ``score_thresh``; or the keypoint-derived bbox shares less than
    ``min_sanity_iou`` overlap with the SAM3 bbox (a sign the keypoints
    belong to a different person or the track is otherwise unreliable this
    frame -- never trust guidance that disagrees with SAM3 about *where the
    person roughly is*, only about how tightly to crop them).
    """
    sam_bbox = np.asarray(sam_bbox_xyxy, dtype=np.float32).reshape(4)
    if keypoints_xy is None or scores is None:
        return sam_bbox, False, 0

    kp_bbox, num_used = sapiens2_keypoint_bbox(
        keypoints_xy,
        scores,
        score_thresh=score_thresh,
        frame_width=frame_width,
        frame_height=frame_height,
        padding_frac=padding_frac,
    )
    if kp_bbox is None or num_used < min_guidance_keypoints:
        return sam_bbox, False, num_used

    iou = bbox_iou_xyxy(tuple(kp_bbox.tolist()), tuple(sam_bbox.tolist()))
    if iou < min_sanity_iou:
        _log(
            f"WARNING: Sapiens2 keypoint bbox IoU={iou:.3f} vs SAM3 bbox "
            f"(< {min_sanity_iou}); keeping SAM3 bbox for this person/frame."
        )
        return sam_bbox, False, num_used
    return kp_bbox, True, num_used


# --------------------------------------------------------------------------- #
# Reading an existing sam3sapiens2.py combined run
# --------------------------------------------------------------------------- #
# A sam3sapiens2_visualize.py / sam3dinov3_visualize.py single-ID rerender
# output dir is named "<base>_sam3sapiens2_visualized_id_NN" /
# "<base>_sam3dinov3_visualized_id_NN" -- its sibling combined-run dir is
# "<base>" next to it. Used to auto-resolve exactly the wrong-directory case
# reported 2026-08-07 (see loops/sapiens2-3d-usability-loop.md).
_VISUALIZED_ID_DIR_RE = re.compile(
    r"^(?P<base>.+)_(?:sam3sapiens2|sam3dinov3)_visualized_id_\d+$", re.IGNORECASE
)


def find_sapiens2_predictions_json(results_dir: Path, video_stem: str | None = None) -> Path:
    """Locate a ``*_sam3sapiens2_predictions.json`` file under ``results_dir``.

    Resolution order (first match wins):
      1. Exact ``video_stem`` match directly in ``results_dir`` or its
         ``video_stem/`` subdirectory.
      2. ``results_dir`` itself is such a file.
      3. Exactly one ``*_sam3sapiens2_predictions.json`` sitting directly in
         ``results_dir`` -- used even when its name doesn't match
         ``video_stem``, since a directory holding exactly one such file is
         unambiguous regardless of naming (e.g. a Visualize-ID rerender dir
         that also preserves a copy of the original combined run's JSON in
         its ``source_artifacts``).
      4. ``results_dir`` looks like a Visualize-ID rerender output
         (``<base>_sam3sapiens2_visualized_id_NN`` /
         ``<base>_sam3dinov3_visualized_id_NN``) -- resolved to its sibling
         combined-run directory.
      5. Recursive search under ``results_dir`` (last resort).
    """
    root = Path(results_dir).expanduser().resolve()

    if video_stem:
        direct = root / f"{video_stem}_sam3sapiens2_predictions.json"
        if direct.is_file():
            return direct
        nested = root / video_stem / f"{video_stem}_sam3sapiens2_predictions.json"
        if nested.is_file():
            return nested

    if root.is_file() and root.name.endswith("_sam3sapiens2_predictions.json"):
        return root

    if root.is_dir():
        direct_children = sorted(root.glob("*_sam3sapiens2_predictions.json"))
        if len(direct_children) == 1:
            found = direct_children[0]
            if video_stem and found.name != f"{video_stem}_sam3sapiens2_predictions.json":
                _log(
                    f"Using {found.name} found directly in {root} -- it does not match "
                    f"the expected video_stem {video_stem!r}, but it is the only "
                    "predictions JSON in that directory."
                )
            return found

        id_dir_match = _VISUALIZED_ID_DIR_RE.match(root.name)
        if id_dir_match:
            sibling = root.parent / id_dir_match.group("base")
            sibling_json = sibling / f"{id_dir_match.group('base')}_sam3sapiens2_predictions.json"
            if sibling_json.is_file():
                _log(
                    f"'{root.name}' looks like a Visualize-ID rerender output; "
                    f"resolved to its combined run at {sibling}"
                )
                return sibling_json

    matches = sorted(root.glob("**/*_sam3sapiens2_predictions.json"))
    if video_stem:
        matches = [m for m in matches if m.name == f"{video_stem}_sam3sapiens2_predictions.json"]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        hint = ""
        id_dir_match = _VISUALIZED_ID_DIR_RE.match(root.name)
        if id_dir_match:
            suggested = root.parent / id_dir_match.group("base")
            hint = (
                f" '{root.name}' looks like a Visualize-ID rerender output; "
                f"did you mean the combined run at {suggested}?"
            )
        raise FileNotFoundError(
            f"No *_sam3sapiens2_predictions.json found under {root} "
            f"(video_stem={video_stem!r}). Run sam3sapiens2.py first.{hint}"
        )
    raise FileNotFoundError(
        f"Multiple *_sam3sapiens2_predictions.json found under {root}; "
        f"pass a more specific --sapiens2-results path. Candidates: "
        + ", ".join(str(m) for m in matches)
    )


def load_sapiens2_predictions(json_path: Path) -> dict[str, Any]:
    """Load a ``sam3sapiens2.py`` combined-predictions payload."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "vaila_sam3sapiens2_v1":
        _log(
            f"WARNING: {json_path} has schema={payload.get('schema')!r}, "
            "expected 'vaila_sam3sapiens2_v1'; proceeding, but the layout may not match."
        )
    return payload


def sapiens2_keypoints_by_frame(
    payload: dict[str, Any],
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Index a combined-predictions payload as ``{frame: {sam_obj_id: (xy, scores)}}``."""
    lookup: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for record in payload.get("frames", []):
        frame_idx = int(record["frame_index"])
        per_person: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for inst in record.get("instances", []):
            obj_id = int(inst["sam_obj_id"])
            kpts = np.asarray(inst["keypoints"], dtype=np.float64).reshape(-1, 2)
            scores = np.asarray(inst["keypoint_scores"], dtype=np.float64).reshape(-1)
            per_person[obj_id] = (kpts, scores)
        lookup[frame_idx] = per_person
    return lookup


# --------------------------------------------------------------------------- #
# Per-frame SAM3D Body batch construction (mirrors sam3dinov3._frame_batch_from_guidance,
# substituting the bbox source; masks stay SAM3-authoritative, unchanged)
# --------------------------------------------------------------------------- #
@dataclass
class GuidanceRecord:
    frame: int
    person_id: int
    guided: bool
    num_keypoints_used: int


def _frame_batch_from_guidance_sapiens2(
    frame_guidance: list[tuple[dict[str, Any], dict[str, Any] | None]],
    sapiens2_frame_lookup: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    frame_idx: int,
    frame_width: int,
    frame_height: int,
    bbox_padding: float,
    contour_margin: int,
    use_mask: bool,
    min_guidance_keypoints: int,
    kpt_score_thresh: float,
    kpt_padding_frac: float,
    min_sanity_iou: float,
) -> tuple[list[int], np.ndarray | None, np.ndarray | None, list[GuidanceRecord]]:
    """Turn SAM tracks/contours + Sapiens2 keypoints into a SAM3D Body batch."""
    obj_ids: list[int] = []
    boxes: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    records: list[GuidanceRecord] = []
    for track, contour_obj in frame_guidance:
        obj_id = int(track["obj_id"])
        sam_bbox = _pose_bbox_from_sam(
            track,
            contour_obj,
            frame_width=frame_width,
            frame_height=frame_height,
            padding_fraction=bbox_padding,
        )
        keypoints_xy, kpt_scores = sapiens2_frame_lookup.get(obj_id, (None, None))
        bbox, guided, num_used = tighten_bbox_with_sapiens2(
            sam_bbox,
            keypoints_xy,
            kpt_scores,
            frame_width=frame_width,
            frame_height=frame_height,
            min_guidance_keypoints=min_guidance_keypoints,
            score_thresh=kpt_score_thresh,
            padding_frac=kpt_padding_frac,
            min_sanity_iou=min_sanity_iou,
        )
        if bbox[2] - bbox[0] < 2.0 or bbox[3] - bbox[1] < 2.0:
            continue
        obj_ids.append(obj_id)
        boxes.append(bbox)
        records.append(
            GuidanceRecord(
                frame=frame_idx, person_id=obj_id, guided=guided, num_keypoints_used=num_used
            )
        )
        if use_mask:
            raw = _contour_mask(
                (frame_height, frame_width),
                track,
                contour_obj,
                margin_px=contour_margin,
            )
            # Same convention as sam3dinov3.py: SAM 3D Body expects 0/1 masks,
            # _contour_mask yields 0/255.
            masks.append((raw > 0).astype(np.uint8))
    if not boxes:
        return [], None, None, records
    box_array = np.stack(boxes, axis=0).astype(np.float32)
    mask_array = np.stack(masks, axis=0).astype(np.uint8) if masks else None
    return obj_ids, box_array, mask_array, records


# --------------------------------------------------------------------------- #
# Companion CSV: which frames/people were actually Sapiens2-guided
# --------------------------------------------------------------------------- #
def write_guidance_csv(output_dir: Path, stem: str, records: list[GuidanceRecord]) -> Path:
    """Per frame/person record of whether Sapiens2 guidance was used.

    Written alongside (never merged into) sam3dinov3.py's shared MHR70 CSVs,
    so this module never has to change a schema another pipeline depends on.
    """
    path = output_dir / f"{stem}_sapiens2_3d_guidance.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "person_id", "guided", "num_keypoints_used"])
        for rec in records:
            writer.writerow([rec.frame, rec.person_id, int(rec.guided), rec.num_keypoints_used])
    return path


def _write_readme(
    output_dir: Path,
    *,
    video_path: Path,
    sam_dir: Path,
    sapiens2_predictions: Path,
    weights_dir: Path,
    inference_type: str,
    stride: int,
    use_mask: bool,
    min_guidance_keypoints: int,
    kpt_score_thresh: float,
    focal_px: float | None,
    n_keypoints: int,
    n_guided_frames: int,
    n_total_person_frames: int,
) -> Path:
    guided_pct = 100.0 * n_guided_frames / n_total_person_frames if n_total_person_frames else 0.0
    text = f"""vailá Sapiens2-guided SAM 3D Body markerless 3D run
video={video_path}
sam_results={sam_dir}
sapiens2_predictions={sapiens2_predictions}
sam3d_weights={weights_dir}
inference_type={inference_type}
stride={stride}
mask_conditioned={use_mask}
min_guidance_keypoints={min_guidance_keypoints}
kpt_score_thresh={kpt_score_thresh}
focal_px={"auto (default FOV)" if focal_px is None else focal_px}
n_keypoints={n_keypoints}
person_frames_guided_by_sapiens2={n_guided_frames}/{n_total_person_frames} ({guided_pct:.1f}%)
identity_authority=SAM3 obj_id

Pipeline
--------
1. SAM 3 segments/tracks people and defines bbox, silhouette, score and obj_id
   (via an existing sam3sapiens2.py run, or one this module ran on demand).
2. Sapiens2 (308-keypoint 2D pose, already computed by sam3sapiens2.py) is
   used ONLY to tighten the per-person bbox: when enough confident keypoints
   are available and they agree geometrically with the SAM3 bbox (IoU sanity
   check), the tighter keypoint-derived box replaces SAM3's; otherwise the
   SAM3 bbox is used unchanged. Sapiens2 does not otherwise influence the 3D
   lifter -- see the module docstring for why (no supported inference-time
   keypoint-prompt hook exists in the vendored SAM 3D Body estimator).
3. SAM 3D Body (DINOv3 ViT-H/16+ backbone) receives that bbox + the SAM3
   contour mask, through the SAME
   ``SAM3DBodyEstimator.process_one_image(bboxes=..., masks=...)`` call
   sam3dinov3.py already makes.
4. person_id is exactly the SAM obj_id, so no second Re-ID can swap identities.

Coordinate systems and units -- identical to sam3dinov3.py
------------------------------------------------------------
x_m,y_m,z_m          Root-relative 3D joints, metres, camera axes (OpenCV: +x
                     right, +y down, +z forward/away from the camera).
xcam_m,ycam_m,zcam_m Camera-frame absolute joints = root-relative + cam_t.
x_px,y_px            Perspective reprojection into the original full frame.
frames               Zero-based, matching the source video.

Scale caveat: monocular depth is metric only up to the assumed camera
intrinsics. Without --focal-px the model falls back to a default FOV
(f = sqrt(W^2+H^2)).

Main outputs
------------
<video>_sapiens2_3d_overlay.mp4          SAM contour/bbox/ID + reprojected 3D skeleton.
<video>_sam3dinov3_keypoints3d.csv       Long table (written by sam3dinov3.py's own
                                          writer; schema unchanged by this module).
<video>_sam3dinov3_keypoints2d.csv       Long table, reprojected pixels.
<video>_sam3dinov3_camera.csv            Per-frame focal length, cam_t and bbox.
<video>_sam3dinov3_joint_angles.csv      Long table, local joint angles (if available).
<video>_id_NN_mhr70_3d.csv               Wide, named columns.
<video>_id_NN_mhr70_rec3d.csv            Wide, vailá rec3d convention.
<video>_id_NN_markers.csv                Wide 2D for REC2D / getpixelvideo.
<video>_sam3dinov3_predictions.json.gz   Full provenance and per-instance predictions.
<video>_sapiens2_3d_guidance.csv         THIS module's own output: per frame/person,
                                          whether Sapiens2 keypoint guidance was used
                                          and how many keypoints cleared the score
                                          threshold. Absent from sam3dinov3.py runs.
meshes/frame_NNNNNN.npz                  Only with --save-mesh.
mesh_faces.npy                           Only with --save-mesh.
sapiens2_3d_summary.json                 Machine-readable run summary.
README_sapiens2_3d.txt                   This file.
FAILED_sapiens2_3d.txt                   Exists only when this stage fails.

References
----------
SAM 3        https://ai.meta.com/research/sam3/
Sapiens2     https://about.meta.com/realitylabs/codecavatars/sapiens/
SAM 3D Body  https://github.com/facebookresearch/sam-3d-body
Weights      https://huggingface.co/{DEFAULT_HF_REPO_ID}
"""
    path = output_dir / "README_sapiens2_3d.txt"
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
        "Sapiens2-guided SAM 3D Body FAILED\n"
        f"video={video_path}\n"
        f"timestamp={dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"reason={reason}\n"
    )
    if traceback_str:
        body += f"\nTraceback:\n{traceback_str}\n"
    (output_dir / "FAILED_sapiens2_3d.txt").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Optional DLT3D/ref3d auto-chain: place each person's monocular camera-frame
# output into the calibrated lab frame via monocular_dlt_align.py, unmodified
# and imported -- never duplicated. Only runs when --dlt3d is given; absent
# it, behavior is byte-identical to before this feature existed.
# --------------------------------------------------------------------------- #
def _run_dlt_chain(
    output_dir: Path,
    stem: str,
    person_ids: list[int],
    fps: float,
    args: argparse.Namespace,
) -> dict[int, str]:
    """Auto-chain each person's ``*_id_NN_mhr70_rec3d.csv`` into the
    DLT3D-calibrated lab frame, writing into ``dlt_world/id_NN/``.

    One person's placement failure is logged and skipped -- never fatal for
    the rest of the run, since the SAM3D outputs it would otherwise discard
    are already safely on disk.
    """
    results: dict[int, str] = {}
    for pid in person_ids:
        mono3d_path = output_dir / f"{stem}_id_{pid:02d}_mhr70_rec3d.csv"
        if not mono3d_path.is_file():
            continue
        pixels_path = output_dir / f"{stem}_id_{pid:02d}_markers.csv"
        dlt_output_dir = output_dir / "dlt_world" / f"id_{pid:02d}"
        _log(f"DLT-chaining person {pid}: placing into the calibrated lab frame")
        try:
            result = align_monocular_to_world(
                mono3d_path,
                args.dlt3d,
                dlt_output_dir,
                pixels_path=pixels_path if pixels_path.is_file() else None,
                ref3d_path=args.ref3d,
                point_rate=fps,
                smooth_hz=0.0 if args.no_smooth else float(args.smooth_hz),
                refine=not args.no_refine,
                origin_markers=tuple(args.origin_markers),
                skeleton_json_path=args.skeleton,
                export_mesh=args.export_mesh,
                gui=False,
            )
        except Exception as exc:  # noqa: BLE001 - one person's placement failure is not fatal
            _log(f"WARNING: DLT placement failed for person {pid}: {exc}")
            continue
        if result is None:
            _log(f"WARNING: DLT placement returned no result for person {pid}")
            continue
        results[pid] = str(result[0])
        _log(f"DLT-chained person {pid} -> {result[0]}")
    return results


# --------------------------------------------------------------------------- #
# Core per-video pipeline
# --------------------------------------------------------------------------- #
def run_sapiens2_guided_sam3d(
    video_path: Path,
    output_dir: Path,
    guidance: SamGuidance,
    sapiens2_lookup: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]],
    sapiens2_predictions_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run Sapiens2-guided SAM 3D Body over one video.

    Mirrors ``sam3dinov3.run_sam3d_from_sam`` closely -- same estimator call,
    same writers -- with only the bbox source and an extra guidance CSV
    differing. See the module docstring for exactly what "guided" means here.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    try:
        estimator = load_sam3d_estimator(args.weights_dir, device="cuda")
    except Exception as exc:  # noqa: BLE001 - surfaced with setup context
        cap.release()
        raise RuntimeError(f"Failed to load SAM 3D Body estimator: {exc}") from exc

    names = keypoint_names(len(MHR70_NAMES))
    edges = skeleton_edges(names)

    cam_int = None
    if args.focal_px is not None:
        cam_int = build_cam_int(float(args.focal_px), width, height)
        _log(f"Using fixed intrinsics: f={float(args.focal_px):.1f} px, principal point centred")

    writer = None
    overlay_path = output_dir / f"{stem}_sapiens2_3d_overlay.mp4"
    if not args.no_overlay:
        try:
            writer, overlay_path = _open_sam3_video_writer(
                overlay_path,
                fps,
                (width, height),
                purpose="Sapiens2-guided SAM3D overlay",
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
    guidance_records: list[GuidanceRecord] = []
    stride = max(1, int(args.stride))
    n_processed = 0
    n_people = 0
    frame_idx = 0

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
                sapiens2_frame_lookup = sapiens2_lookup.get(frame_idx, {})
                obj_ids, boxes, masks, frame_records = _frame_batch_from_guidance_sapiens2(
                    frame_guidance,
                    sapiens2_frame_lookup,
                    frame_idx=frame_idx,
                    frame_width=width,
                    frame_height=height,
                    bbox_padding=float(args.bbox_padding),
                    contour_margin=int(args.contour_margin),
                    use_mask=not args.no_mask,
                    min_guidance_keypoints=int(args.min_guidance_keypoints),
                    kpt_score_thresh=float(args.kpt_score_thresh),
                    kpt_padding_frac=float(args.kpt_bbox_padding_frac),
                    min_sanity_iou=float(args.min_sanity_iou),
                )
                guidance_records.extend(frame_records)
                if boxes is not None:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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

            for inst in instances:
                inst.pop("vertices", None)
            if instances:
                timeline[frame_idx] = instances

            if writer is not None:
                overlay = _draw_sam_guidance(
                    frame_bgr, frame_guidance, draw_ids=not args.no_draw_id
                )
                overlay = _draw_pose_overlay(
                    overlay, instances, edges, names, draw_ids=not args.no_draw_id
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
            "SAM 3D Body produced no instances; check the SAM/Sapiens2 guidance, "
            "--min-sam-score/--min-sam-area filters, or the weights directory."
        )

    written = write_long_keypoints_csvs(output_dir, stem, timeline, names)
    written.append(write_camera_csv(output_dir, stem, timeline))
    joint_angles_path = write_long_joint_angles_csv(output_dir, stem, timeline, names)
    if joint_angles_path is not None:
        written.append(joint_angles_path)
    written.extend(
        write_wide_person_csvs(
            output_dir, stem, timeline, names, n_frames=max(frame_idx, max(timeline) + 1)
        )
    )
    meta = {
        "video": str(video_path),
        "sam_results": str(guidance.sam_dir),
        "sapiens2_predictions": str(sapiens2_predictions_path),
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
    written.append(write_guidance_csv(output_dir, stem, guidance_records))

    dlt_world_outputs: dict[int, str] = {}
    if getattr(args, "dlt3d", None) is not None:
        dlt_world_outputs = _run_dlt_chain(
            output_dir, stem, _collect_person_ids(timeline), fps, args
        )

    n_guided = sum(1 for r in guidance_records if r.guided)
    written.append(
        _write_readme(
            output_dir,
            video_path=video_path,
            sam_dir=guidance.sam_dir,
            sapiens2_predictions=sapiens2_predictions_path,
            weights_dir=(args.weights_dir or default_weights_dir()),
            inference_type=str(args.inference_type),
            stride=stride,
            use_mask=not args.no_mask,
            min_guidance_keypoints=int(args.min_guidance_keypoints),
            kpt_score_thresh=float(args.kpt_score_thresh),
            focal_px=None if args.focal_px is None else float(args.focal_px),
            n_keypoints=len(names),
            n_guided_frames=n_guided,
            n_total_person_frames=len(guidance_records),
        )
    )

    summary = {
        "schema": "vaila_sapiens2_3d_video_v1",
        "status": "complete",
        **meta,
        "n_inference_frames": n_processed,
        "n_person_frames": n_people,
        "n_guidance_records": len(guidance_records),
        "n_sapiens2_guided_person_frames": n_guided,
        "person_ids": _collect_person_ids(timeline),
        "n_keypoints": len(names),
        "overlay": str(overlay_path) if writer is not None else None,
        "outputs": [str(p) for p in written],
        "dlt_world_outputs": dlt_world_outputs,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (output_dir / "sapiens2_3d_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _log(
        f"Done {video_path.name}: {n_processed} inference frames, {n_people} person-frames, "
        f"{n_guided}/{len(guidance_records)} person-frames Sapiens2-guided -> {output_dir}"
    )
    return summary


# --------------------------------------------------------------------------- #
# Front-end resolution: an existing sam3sapiens2 run, or run it from raw SAM3
# --------------------------------------------------------------------------- #
def _find_raw_video_by_name(
    start_dirs: list[Path], filename: str, *, max_levels: int = 5
) -> Path | None:
    """Walk upward from each of ``start_dirs`` looking for a file named exactly
    ``filename``. Returns it only when exactly one distinct match is found
    across all search roots/levels -- ambiguity is treated as "not found" so
    the caller falls back to asking for an explicit ``--input`` rather than
    silently guessing between candidates.
    """
    found: set[Path] = set()
    for start in start_dirs:
        current = Path(start).expanduser().resolve()
        if current.is_file():
            current = current.parent
        for _ in range(max_levels + 1):
            candidate = current / filename
            if candidate.is_file():
                found.add(candidate)
            parent = current.parent
            if parent == current:
                break
            current = parent
    if len(found) == 1:
        return next(iter(found))
    return None


def _auto_locate_video_from_results(
    *,
    sapiens2_results: Path | None,
    sam_results: Path | None,
    input_hint: Path | None,
) -> Path | None:
    """When ``-i`` resolves to no usable video, try to locate the raw video
    an existing ``--sapiens2-results`` run was actually computed from, using
    the filename its own predictions JSON remembers (``payload["video"]``).
    Scoped to ``--sapiens2-results`` only -- a raw ``--sam-results`` dir has
    no equivalent stored video name to key off of.
    """
    if sapiens2_results is None:
        return None
    try:
        json_path = find_sapiens2_predictions_json(sapiens2_results)
    except FileNotFoundError:
        return None
    payload = load_sapiens2_predictions(json_path)
    video_name = payload.get("video")
    if not video_name:
        return None
    search_roots = [sapiens2_results]
    if input_hint is not None:
        search_roots.append(input_hint)
    found = _find_raw_video_by_name(search_roots, video_name)
    if found is not None:
        _log(f"Auto-located raw video '{video_name}' at {found} (from {json_path.name})")
    return found


def _resolve_sapiens2_front_end(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[SamGuidance, dict[int, dict[int, tuple[np.ndarray, np.ndarray]]], Path]:
    """Return (SAM guidance, Sapiens2 keypoint lookup, predictions json path)."""
    if args.sapiens2_results is not None:
        json_path = find_sapiens2_predictions_json(args.sapiens2_results, video_path.stem)
    elif args.sam_results is not None:
        sam_dir = resolve_sam_results_dir(args.sam_results, video_path, single_video=True)
        sapiens2_dir = output_dir / "sam3sapiens2"
        _log(f"No --sapiens2-results given; running the Sapiens2 stage from {sam_dir}")
        run_sapiens_from_sam(video_path, sapiens2_dir, sam_dir)
        json_path = find_sapiens2_predictions_json(sapiens2_dir, video_path.stem)
    else:
        raise ValueError(
            "Provide --sapiens2-results (an existing sam3sapiens2.py run) or "
            "--sam-results (a raw SAM3 run; the Sapiens2 stage will be run first)."
        )

    payload = load_sapiens2_predictions(json_path)
    recorded_video = payload.get("video")
    if recorded_video and recorded_video != video_path.name:
        raise ValueError(
            f"{json_path.name} was built from '{recorded_video}', but you are "
            f"processing '{video_path.name}' -- these must be the same video "
            "(mismatched guidance would silently misalign every frame). If you "
            f"meant to process '{recorded_video}', point --input at it directly."
        )
    sam_dir = Path(payload["sam_results"])
    guidance = load_sam_guidance(sam_dir)
    lookup = sapiens2_keypoints_by_frame(payload)
    return guidance, lookup, json_path


def _process_one_video(
    video_path: Path, output_root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    output_dir = output_root / video_path.stem
    try:
        guidance, lookup, json_path = _resolve_sapiens2_front_end(video_path, output_dir, args)
        return run_sapiens2_guided_sam3d(video_path, output_dir, guidance, lookup, json_path, args)
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised for the caller to count
        _write_failure(output_dir, video_path, str(exc), traceback.format_exc())
        raise
    finally:
        _release_gpu_memory()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sapiens2-guided SAM 3D Body (DINOv3): monocular markerless 3D, "
            "SAM3 owns bbox/contour/ID, Sapiens2 keypoints tighten the bbox."
        )
    )
    parser.add_argument("-i", "--input", type=Path, help="Input video or non-recursive folder")
    parser.add_argument("-o", "--output", type=Path, help="Output parent directory")
    parser.add_argument(
        "--sapiens2-results",
        type=Path,
        default=None,
        help="Existing sam3sapiens2.py output dir (preferred front end)",
    )
    parser.add_argument(
        "--sam-results",
        type=Path,
        default=None,
        help="Raw SAM3 output dir; the Sapiens2 stage runs first if given instead",
    )
    parser.add_argument("--weights-dir", type=Path, default=None, help="SAM 3D Body weights dir")
    parser.add_argument("--inference-type", default=DEFAULT_INFERENCE_TYPE, choices=INFERENCE_TYPES)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--bbox-padding", type=float, default=DEFAULT_BBOX_PADDING)
    parser.add_argument("--contour-margin", type=int, default=DEFAULT_CONTOUR_MARGIN_PX)
    parser.add_argument("--min-sam-score", type=float, default=DEFAULT_MIN_SAM_SCORE)
    parser.add_argument("--min-sam-area", type=int, default=DEFAULT_MIN_SAM_AREA)
    parser.add_argument("--max-persons", type=int, default=DEFAULT_MAX_PERSONS)
    parser.add_argument(
        "--min-guidance-keypoints", type=int, default=DEFAULT_MIN_GUIDANCE_KEYPOINTS
    )
    parser.add_argument("--kpt-score-thresh", type=float, default=DEFAULT_KPT_SCORE_THRESH)
    parser.add_argument(
        "--kpt-bbox-padding-frac", type=float, default=DEFAULT_KPT_BBOX_PADDING_FRAC
    )
    parser.add_argument("--min-sanity-iou", type=float, default=DEFAULT_MIN_SANITY_IOU)
    parser.add_argument("--focal-px", type=float, default=None)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--save-mesh", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-draw-id", action="store_true")
    parser.add_argument("--verbose-model", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument(
        "--dlt3d",
        type=Path,
        default=None,
        help=(
            "This camera's DLT3D calibration. When given, each person's monocular "
            "output is automatically placed into the calibrated lab frame "
            "(monocular_dlt_align.py, called once per person) -> dlt_world/id_NN/. "
            "Absent, behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--ref3d", type=Path, default=None, help="DLT-chain: control points, for validation only"
    )
    parser.add_argument(
        "--smooth-hz",
        type=float,
        default=DLT_DEFAULT_SMOOTH_HZ,
        help=f"DLT-chain: Butterworth cutoff on the 6-DOF placement (default {DLT_DEFAULT_SMOOTH_HZ})",
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="DLT-chain: disable placement smoothing (raw)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="DLT-chain: translation only (skip the 6-DOF refinement); usually worse",
    )
    parser.add_argument(
        "--origin-markers",
        type=int,
        nargs="+",
        default=list(DLT_DEFAULT_ORIGIN_MARKERS),
        help=(
            "DLT-chain: 1-based MHR70 markers whose midpoint the placement rotates "
            f"about (default {' '.join(map(str, DLT_DEFAULT_ORIGIN_MARKERS))} = hips)"
        ),
    )
    parser.add_argument(
        "--skeleton",
        type=Path,
        default=None,
        help="DLT-chain: skeleton JSON for the Blender script",
    )
    parser.add_argument(
        "--export-mesh",
        choices=("none", "obj", "ply"),
        default=DLT_DEFAULT_EXPORT_MESH,
        help=(
            f"DLT-chain: aligned per-frame mesh format for Blender (default "
            f"{DLT_DEFAULT_EXPORT_MESH}); needs --save-mesh in this same run; "
            "silently skipped when no meshes/ source is found"
        ),
    )
    parser.add_argument("--open-help", action="store_true")
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
    if args.input is None and args.output is None:
        run_sapiens2_3d()
        return
    if args.input is None or args.output is None:
        parser.error("--input and --output must both be supplied")
    if args.sapiens2_results is None and args.sam_results is None:
        parser.error("provide --sapiens2-results or --sam-results")
    if args.focal_px is not None and args.focal_px <= 0:
        parser.error("--focal-px must be > 0")
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    input_path = args.input.expanduser().resolve()
    output_parent = args.output.expanduser().resolve()
    videos = _find_videos(input_path)
    if not videos and args.sapiens2_results is not None:
        auto = _auto_locate_video_from_results(
            sapiens2_results=args.sapiens2_results,
            sam_results=args.sam_results,
            input_hint=input_path,
        )
        if auto is not None:
            videos = [auto]
    if not videos:
        parser.error(f"no supported video found under {input_path}")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = output_parent / f"processed_sapiens2_3d_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    failed: list[str] = []
    summaries: list[dict[str, Any]] = []
    for video in videos:
        _log(f"Processing {video.name} ({_video_frame_count(video)} frames)")
        try:
            summaries.append(_process_one_video(video, output_base, args))
        except Exception as exc:  # noqa: BLE001 - one failure must not abort the batch
            _log(f"FAILED {video.name}: {exc}")
            failed.append(video.name)

    _log(f"Batch done: {len(summaries)} ok, {len(failed)} failed -> {output_base}")
    if failed:
        sys.exit(1)


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
@dataclass
class Sapiens23dGuiSettings:
    """User choices collected by the Tkinter dialog."""

    input_path: Path
    output_parent: Path
    sapiens2_results: Path | None
    sam_results: Path | None
    device: int
    stride: int
    min_guidance_keypoints: int
    kpt_score_thresh: float
    focal_px: float | None
    weights_dir: Path | None
    use_mask: bool
    save_overlay: bool
    save_mesh: bool
    dlt3d: Path | None
    ref3d: Path | None
    export_mesh: str


def _format_gui_cli(settings: Sapiens23dGuiSettings) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "vaila/sapiens2_3d.py",
        "-i",
        str(settings.input_path),
        "-o",
        str(settings.output_parent),
        "--device",
        str(settings.device),
        "--stride",
        str(settings.stride),
        "--min-guidance-keypoints",
        str(settings.min_guidance_keypoints),
        "--kpt-score-thresh",
        str(settings.kpt_score_thresh),
    ]
    if settings.sapiens2_results is not None:
        cmd.extend(["--sapiens2-results", str(settings.sapiens2_results)])
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
    if settings.dlt3d is not None:
        cmd.extend(["--dlt3d", str(settings.dlt3d)])
    if settings.ref3d is not None:
        cmd.extend(["--ref3d", str(settings.ref3d)])
    if settings.export_mesh != DLT_DEFAULT_EXPORT_MESH:
        cmd.extend(["--export-mesh", settings.export_mesh])
    return cmd


def run_sapiens2_3d(existing_root: Any | None = None) -> None:
    """Open the Sapiens2-guided SAM 3D Body settings GUI and launch its CLI."""
    owns_root = existing_root is None
    root = existing_root if existing_root is not None else tk.Tk()
    if owns_root:
        root.withdraw()
    _prepare_gui_root(root, owns_root=owns_root)

    class Sapiens23dDialog(tk.Toplevel):
        def __init__(self, master: tk.Misc) -> None:
            super().__init__(master)
            self.title("vailá — Sapiens2-guided SAM 3D Body markerless 3D")
            self.result: Sapiens23dGuiSettings | None = None
            self.transient(master)  # ty: ignore[no-matching-overload]
            self.resizable(False, False)

            self.input_var = tk.StringVar()
            self.output_var = tk.StringVar()
            self.sapiens2_var = tk.StringVar()
            self.sam_var = tk.StringVar()
            self.weights_var = tk.StringVar(value=str(default_weights_dir()))
            self.device_var = tk.StringVar(value="0")
            self.stride_var = tk.StringVar(value=str(DEFAULT_STRIDE))
            self.min_kpts_var = tk.StringVar(value=str(DEFAULT_MIN_GUIDANCE_KEYPOINTS))
            self.kpt_thresh_var = tk.StringVar(value=str(DEFAULT_KPT_SCORE_THRESH))
            self.focal_var = tk.StringVar()
            self.mask_var = tk.BooleanVar(value=True)
            self.overlay_var = tk.BooleanVar(value=True)
            self.mesh_var = tk.BooleanVar(value=False)
            self.dlt3d_var = tk.StringVar()
            self.ref3d_var = tk.StringVar()
            self.export_mesh_var = tk.StringVar(value=DLT_DEFAULT_EXPORT_MESH)

            frame = ttk.Frame(self, padding=12)
            frame.grid(row=0, column=0, sticky="nsew")
            row = 0

            ttk.Label(
                frame,
                text=(
                    "SAM 3 owns bbox/contour/ID; Sapiens2's 308 keypoints (from an\n"
                    "existing sam3sapiens2.py run) tighten the bbox before SAM 3D\n"
                    "Body (DINOv3) lifts each person to 3D. Requires CUDA."
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
            _path_row(
                "Existing sam3sapiens2 results (preferred)",
                self.sapiens2_var,
                self._browse_sapiens2,
            )
            _path_row(
                "Raw SAM3 results (used if no Sapiens2 results)", self.sam_var, self._browse_sam
            )
            _path_row("SAM 3D Body weights dir", self.weights_var, self._browse_weights)

            def _entry_row(label: str, var: tk.StringVar, width: int = 12) -> None:
                nonlocal row
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3)
                ttk.Entry(frame, textvariable=var, width=width).grid(
                    row=row, column=1, sticky="w", pady=3
                )
                row += 1

            _entry_row("CUDA device index", self.device_var)
            _entry_row("Frame stride", self.stride_var)
            _entry_row("Min. Sapiens2 keypoints to guide", self.min_kpts_var)
            _entry_row("Sapiens2 keypoint score threshold", self.kpt_thresh_var)
            _entry_row("Focal length px (blank = auto)", self.focal_var)

            ttk.Checkbutton(
                frame, text="Mask-conditioned (use SAM silhouettes)", variable=self.mask_var
            ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            row += 1
            ttk.Checkbutton(frame, text="Write overlay video", variable=self.overlay_var).grid(
                row=row, column=0, columnspan=2, sticky="w", pady=2
            )
            row += 1
            ttk.Checkbutton(frame, text="Save MHR meshes (large)", variable=self.mesh_var).grid(
                row=row, column=0, columnspan=3, sticky="w", pady=2
            )
            row += 1

            ttk.Separator(frame, orient="horizontal").grid(
                row=row, column=0, columnspan=4, sticky="we", pady=(10, 6)
            )
            row += 1
            ttk.Label(
                frame,
                text=(
                    "Calibrated lab frame (optional) — give a .dlt3d to auto-place\n"
                    "each person into the DLT-calibrated world frame after this run\n"
                    "(calls monocular_dlt_align.py; see its own help for the math)."
                ),
                justify="left",
            ).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 4))
            row += 1
            _path_row("DLT3D calibration (this camera)", self.dlt3d_var, self._browse_dlt3d)
            _path_row("ref3d control points (optional)", self.ref3d_var, self._browse_ref3d)
            ttk.Label(frame, text="Aligned mesh export").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Combobox(
                frame,
                textvariable=self.export_mesh_var,
                values=("none", "obj", "ply"),
                state="readonly",
                width=10,
            ).grid(row=row, column=1, sticky="w", pady=3)
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

        def _browse_sapiens2(self) -> None:
            path = filedialog.askdirectory(parent=self, title="Select processed_sam3sapiens2_* run")
            if path:
                self.sapiens2_var.set(path)

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

        def _browse_dlt3d(self) -> None:
            path = filedialog.askopenfilename(
                parent=self,
                title="Select this camera's .dlt3d calibration",
                filetypes=[("DLT3D", "*.dlt3d"), ("All", "*.*")],
            )
            if path:
                self.dlt3d_var.set(path)

        def _browse_ref3d(self) -> None:
            path = filedialog.askopenfilename(
                parent=self,
                title="Select control points (.ref3d), optional",
                filetypes=[("ref3d", "*.ref3d"), ("All", "*.*")],
            )
            if path:
                self.ref3d_var.set(path)

        def _open_help(self) -> None:
            if _help_path().is_file():
                webbrowser.open_new_tab(_help_path().as_uri())
            else:
                webbrowser.open_new_tab(f"https://huggingface.co/{DEFAULT_HF_REPO_ID}")

        def _on_run(self) -> None:
            try:
                input_raw = self.input_var.get().strip()
                if not input_raw:
                    raise ValueError("Select an existing input video or folder (Dir… / File…).")
                input_path = Path(input_raw).expanduser()
                if not input_path.exists():
                    raise ValueError(f"Input path not found: {input_path}")
                output_raw = self.output_var.get().strip()
                if not output_raw:
                    raise ValueError("Select an output parent folder.")
                output_parent = Path(output_raw).expanduser()
                sapiens2_raw = self.sapiens2_var.get().strip()
                sapiens2_results = Path(sapiens2_raw).expanduser() if sapiens2_raw else None
                sam_raw = self.sam_var.get().strip()
                sam_results = Path(sam_raw).expanduser() if sam_raw else None
                if sapiens2_results is None and sam_results is None:
                    raise ValueError(
                        "Select an existing sam3sapiens2 results dir, or a raw SAM3 results dir."
                    )
                if sapiens2_results is not None and not sapiens2_results.exists():
                    raise ValueError(f"Sapiens2 results path not found: {sapiens2_results}")
                if sam_results is not None and not sam_results.exists():
                    raise ValueError(f"SAM3 results path not found: {sam_results}")

                videos = _find_videos(input_path)
                if not videos and sapiens2_results is not None:
                    auto = _auto_locate_video_from_results(
                        sapiens2_results=sapiens2_results,
                        sam_results=sam_results,
                        input_hint=input_path,
                    )
                    if auto is not None:
                        videos = [auto]
                        input_path = auto
                if not videos:
                    raise ValueError(f"No supported videos found under: {input_path}")
                weights_raw = self.weights_var.get().strip()
                weights_dir = Path(weights_raw).expanduser() if weights_raw else None
                focal_raw = self.focal_var.get().strip()
                focal_px = float(focal_raw) if focal_raw else None
                if focal_px is not None and focal_px <= 0:
                    raise ValueError("Focal length must be a positive number of pixels.")
                dlt3d_raw = self.dlt3d_var.get().strip()
                dlt3d = Path(dlt3d_raw).expanduser() if dlt3d_raw else None
                if dlt3d is not None and not dlt3d.is_file():
                    raise ValueError(f"DLT3D file not found: {dlt3d}")
                ref3d_raw = self.ref3d_var.get().strip()
                ref3d = Path(ref3d_raw).expanduser() if ref3d_raw else None
                if ref3d is not None and not ref3d.is_file():
                    raise ValueError(f"ref3d file not found: {ref3d}")
                result = Sapiens23dGuiSettings(
                    input_path=input_path,
                    output_parent=output_parent,
                    sapiens2_results=sapiens2_results,
                    sam_results=sam_results,
                    device=max(0, int(self.device_var.get())),
                    stride=max(1, int(self.stride_var.get())),
                    min_guidance_keypoints=max(0, int(self.min_kpts_var.get())),
                    kpt_score_thresh=max(0.0, float(self.kpt_thresh_var.get())),
                    focal_px=focal_px,
                    weights_dir=weights_dir,
                    use_mask=bool(self.mask_var.get()),
                    save_overlay=bool(self.overlay_var.get()),
                    save_mesh=bool(self.mesh_var.get()),
                    dlt3d=dlt3d,
                    ref3d=ref3d,
                    export_mesh=self.export_mesh_var.get().strip() or DLT_DEFAULT_EXPORT_MESH,
                )
            except ValueError as exc:
                messagebox.showerror("Sapiens2 3D Pose", str(exc), parent=self)
                return
            _log(f"Queued {len(videos)} video(s): " + ", ".join(v.name for v in videos[:8]))
            self.result = result
            self.destroy()

    _log("Opening the settings window; bare CLI without -i/-o intentionally uses this GUI.")
    dialog = Sapiens23dDialog(root)
    root.wait_window(dialog)
    settings = dialog.result
    if owns_root:
        root.destroy()
    if settings is None:
        return
    cli = _format_gui_cli(settings)
    print_gui_cli_mirror("vaila/sapiens2_3d", cli)
    launch = [sys.executable, "-u", str(Path(__file__).resolve())] + cli[5:]
    gui_result = run_isolated_gpu_subprocess(launch, device=int(settings.device))
    raise SystemExit(gui_result.returncode)


if __name__ == "__main__":
    main()
