"""
Project: vailá
Script: sam3dinov3_visualize.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 01 August 2026
Update Date: 28 October 2026
Version: 0.4.12

Description:
    CPU-only rerenderer for an existing SAM3+DINOv3 3D (SAM 3D Body) run. It
    selects one or more people (SAM `obj_id`s), draws each contour, bbox, ID,
    and the reprojected MHR skeleton on the original video, and writes
    filtered tracking/keypoint/mesh artifacts for that ID set. SAM3/SAM 3D
    Body weights are never loaded, so this is safe to run on a CPU-only
    machine right after a GPU inference run.

    Supports overriding 2D keypoint coordinates with a manually corrected
    marker CSV (e.g. from getpixelvideo.py via --markers-csv or auto-detection).
    When an override is applied, all coordinate-dependent outputs (overlay
    skeleton video, wide marker CSV, keypoints2d CSV, predictions JSON) are
    regenerated using the corrected positions, while upstream SAM3 segmentation
    masks, bounding boxes, 3D body models, camera translation, and metadata
    are strictly preserved without re-running model inference.

    If the source run's *_sam3dinov3_joint_angles.csv exists (runs made after
    this feature was added -- see joint_kinematics.py), it is filtered to the
    selected ID like the other per-person CSVs: local (parent-relative) Euler
    XYZ degrees + scalar-first quaternion per joint of the model's own
    127-joint MHR rig.

    The MHR70 skeleton is colored by joint side (left=green, right=orange,
    center=blue) using the "left-"/"right-" name prefixes, and the SAM
    contour fill is anti-aliased for a cleaner silhouette edge.

    When the source run used --save-mesh, --export-mesh {obj,ply} writes the
    selected person's per-frame body mesh (vertices + shared MHR faces, with
    cam_t translation applied) as a Blender-importable sequence (built-in
    "Stop Motion OBJ" add-on: Import > Mesh Sequence).

Usage:
    uv run python -u vaila/sam3dinov3_visualize.py \
        --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
        --video /path/to/video.mp4 --id 2 --output /path/to/output \
        --export-mesh obj

    # Multiple IDs → one composite overlay + filtered artifacts for the set:
    uv run python -u vaila/sam3dinov3_visualize.py \
        --sam3d-results /path/to/.../video_stem --id 0 1 2 -o /path/to/out
    # Equivalent: --id 0,1,2

    # Regenerate using manually corrected marker coordinates from getpixelvideo
    # (single-ID only):
    uv run python -u vaila/sam3dinov3_visualize.py \
        --input-dir /path/to/..._visualized_id_00 \
        --markers-csv /path/to/..._overlay_markers.csv

    # Omit --id to be prompted interactively (comma/space-separated list OK).
    # GUI: omit all arguments, or Frame B → Markerless 3D → SAM3+DINOv3 Visualize ID

    Recursive batch over every processed_sam3dinov3_* run directory found under
    a root (CLI-only): omitting --id renders every available id separately per
    video; with ``--id 1 3`` one composite multi-ID output is written per video.
    uv run python -u vaila/sam3dinov3_visualize.py \
        --sam3d-results /path/to/root -r -d -1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import json
import os
import re
import shutil
import sys
import threading
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

try:
    from .sam3dinov3 import (
        COLOR_CENTER_RGB,
        MHR70_NAMES,
        _draw_box_label,
        _rgb_to_bgr,
        _side_color_bgr,
        keypoint_names,
        skeleton_edges,
    )
except ImportError:  # standalone execution
    from sam3dinov3 import (  # ty: ignore[unresolved-import]
        COLOR_CENTER_RGB,
        MHR70_NAMES,
        _draw_box_label,
        _rgb_to_bgr,
        _side_color_bgr,
        keypoint_names,
        skeleton_edges,
    )

try:
    from .dialogsuser import ask_output_directory
except ImportError:
    from dialogsuser import ask_output_directory  # ty: ignore[unresolved-import]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
# Match SAM3 composite alpha (~0.45) for selected-ID contour fills.
SAM_CONTOUR_FILL_ALPHA = 0.45
MESH_EXPORT_FORMATS = ("none", "obj", "ply")


def _log(message: str) -> None:
    # Detached/long-running renders must not die on a dropped terminal (EIO/BrokenPipe).
    with contextlib.suppress(OSError, BrokenPipeError):
        print(f">> vaila/sam3dinov3_visualize: {message}", flush=True)


def _try_import_tqdm() -> Any:
    """Return tqdm class or None."""
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _predictions_path(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*_sam3dinov3_predictions.json.gz"))
    if not candidates:
        raise FileNotFoundError(f"No *_sam3dinov3_predictions.json.gz in {run_dir}")
    return candidates[0]


def resolve_run_dir(path: Path, video_path: Path | None = None) -> Path:
    """Resolve a direct per-video output or a processed batch parent."""
    path = path.expanduser().resolve()
    if path.is_dir() and list(path.glob("*_sam3dinov3_predictions.json.gz")):
        return path
    if video_path is not None:
        stem_dir = path / video_path.stem
        if list(stem_dir.glob("*_sam3dinov3_predictions.json.gz")):
            return stem_dir
    candidates = (
        sorted(
            p
            for p in path.iterdir()
            if p.is_dir() and list(p.glob("*_sam3dinov3_predictions.json.gz"))
        )
        if path.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(p.name for p in candidates[:8])
    raise FileNotFoundError(
        f"Could not resolve a per-video SAM3+DINOv3 directory from {path}. "
        f"Candidates: {names or 'none'}"
    )


_RUN_DIR_RE = re.compile(r"^processed_sam3dinov3_\d{8}_\d{6}$", re.IGNORECASE)
_SKIP_DESCEND_RE = re.compile(r"_sam3dinov3_visualized_id_\d+$", re.IGNORECASE)


def find_run_directories(root: Path, max_depth: int) -> list[Path]:
    """Walk ``root`` and return every completed ``processed_sam3dinov3_*``
    run directory, for ``--recursive`` batch visualization across many runs.

    A matched run directory is never descended into further (no nested runs
    are ever written), and neither is a prior ``*_visualized_id_N`` output --
    the two rules that keep this from ever re-visiting its own output.
    ``max_depth`` matches ``compress_videos_h264.py``'s convention: -1 is
    unlimited, 0 is root only, 1..99 is that many levels below root.
    """
    root = root.expanduser().resolve()
    found: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        rel = current.relative_to(root)
        depth = 0 if rel == Path(".") else len(rel.parts)
        if _RUN_DIR_RE.match(current.name):
            found.append(current)
            dirnames[:] = []
            continue
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not _SKIP_DESCEND_RE.search(name.lower()) and (max_depth < 0 or depth < max_depth)
        )
    return sorted(found, key=lambda p: str(p).lower())


def _iter_video_run_dirs(run_dir: Path) -> list[Path]:
    """Per-video subdirectories of a processed_sam3dinov3_* run that have
    completed predictions (skips anything that failed or is unrelated)."""
    video_dirs = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        try:
            _predictions_path(child)
        except FileNotFoundError:
            continue
        video_dirs.append(child)
    return video_dirs


def load_predictions(run_dir: Path) -> dict[str, Any]:
    path = _predictions_path(run_dir)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    if payload.get("schema") != "vaila_sam3dinov3_v1":
        raise ValueError(f"Unsupported predictions schema in {path}: {payload.get('schema')!r}")
    return payload


def discover_source_video(run_dir: Path, payload: dict[str, Any] | None = None) -> Path | None:
    """Find the exact source video recorded by a SAM3+DINOv3 run.

    The summary's absolute path is authoritative. Relative fallbacks keep runs
    usable after the whole input/results tree has been moved to another disk.
    """
    run_dir = run_dir.expanduser().resolve()
    summary = run_dir / "sam3dinov3_summary.json"
    recorded: Path | None = None
    if summary.is_file():
        try:
            raw = json.loads(summary.read_text(encoding="utf-8")).get("video")
            if raw:
                recorded = Path(str(raw)).expanduser()
                if recorded.is_file():
                    return recorded.resolve()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    payload = payload or load_predictions(run_dir)
    video_name = Path(str(payload.get("video") or (recorded.name if recorded else ""))).name
    if not video_name:
        return None
    for parent in (run_dir, run_dir.parent, run_dir.parent.parent):
        candidate = parent / video_name
        if candidate.is_file():
            return candidate.resolve()
    return None


def validate_source_video(video_path: Path, payload: dict[str, Any]) -> dict[str, int | float]:
    """Prove that the selected video matches the prediction frame space."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    try:
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()

    expected_frames = _safe_int(payload.get("n_frames"))
    expected_width = _safe_int(payload.get("width"))
    expected_height = _safe_int(payload.get("height"))
    mismatches: list[str] = []
    if expected_frames and frames != expected_frames:
        mismatches.append(f"frames={frames}, expected {expected_frames}")
    if expected_width and expected_height and (width, height) != (expected_width, expected_height):
        mismatches.append(f"size={width}x{height}, expected {expected_width}x{expected_height}")
    if mismatches:
        raise ValueError(
            f"The selected video does not match this SAM3+DINOv3 run ({'; '.join(mismatches)}). "
            "Choose the synchronized/cropped source video recorded in sam3dinov3_summary.json."
        )
    return {"frames": frames, "width": width, "height": height, "fps": fps}


def normalize_selected_ids(raw_ids: Sequence[int] | Sequence[str] | int | str) -> list[int]:
    """Parse CLI/GUI ID tokens into a sorted, deduplicated list of ints.

    Accepts ``2``, ``[2, 9]``, ``["0", "1", "2"]``, or comma-bearing tokens
    such as ``["1,3", "7"]`` → ``[1, 3, 7]``.
    """
    if isinstance(raw_ids, (int, str)):
        tokens: list[str] = [str(raw_ids)]
    else:
        tokens = [str(item) for item in raw_ids]
    parsed: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        for part in token.replace(";", ",").split(","):
            piece = part.strip()
            if not piece:
                continue
            # Also split whitespace inside a single argv token ("1 2 3").
            for sub in piece.split():
                value = int(sub)
                if value not in seen:
                    seen.add(value)
                    parsed.append(value)
    if not parsed:
        raise ValueError("At least one person ID is required")
    return sorted(parsed)


def format_ids_tag(selected_ids: Sequence[int]) -> str:
    """Zero-padded ID tag for filenames: ``02`` or ``00_01_02``."""
    ids = normalize_selected_ids(selected_ids)
    return "_".join(f"{i:02d}" for i in ids)


def _unique_output_dir_for_ids(
    output_parent: Path, video_path: Path, selected_ids: Sequence[int]
) -> Path:
    """Return a new child directory for one or more selected IDs."""
    output_parent = output_parent.expanduser().resolve()
    ids = normalize_selected_ids(selected_ids)
    tag = format_ids_tag(ids)
    if len(ids) == 1:
        base = output_parent / f"{video_path.stem}_sam3dinov3_visualized_id_{tag}"
    else:
        base = output_parent / f"{video_path.stem}_sam3dinov3_visualized_ids_{tag}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate


def _unique_gui_output_dir(output_parent: Path, video_path: Path, selected_id: int) -> Path:
    """Return a new ID-specific child directory below a GUI-selected parent."""
    return _unique_output_dir_for_ids(output_parent, video_path, [selected_id])


def discover_ids(run_dir: Path, payload: dict[str, Any] | None = None) -> list[int]:
    payload = payload or load_predictions(run_dir)
    ids: set[int] = set()
    for frame in payload.get("frames", []):
        for instance in frame.get("instances", []):
            value = _safe_int(instance.get("person_id", instance.get("sam_obj_id")))
            if value is not None:
                ids.add(value)
    if not ids:
        summary = run_dir / "sam3dinov3_summary.json"
        if summary.is_file():
            with contextlib.suppress(OSError, ValueError, TypeError, json.JSONDecodeError):
                data = json.loads(summary.read_text(encoding="utf-8"))
                ids.update(int(v) for v in data.get("person_ids", []))
    if not ids:
        tracks = run_dir / "sam3" / "sam_tracks.csv"
        if tracks.exists():
            with tracks.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    value = _safe_int(row.get("obj_id"))
                    if value is not None:
                        ids.add(value)
    if not ids:
        raise ValueError(f"No person IDs found in {run_dir}")
    return sorted(ids)


def prompt_selected_ids(available: list[int]) -> list[int]:
    """Ask for one or more person IDs on stdin until a valid choice is entered."""
    if not available:
        raise ValueError("No person IDs available to prompt")
    avail_set = set(available)
    while True:
        try:
            raw = input(">> Enter SAM/person ID(s) to visualize (comma/space-separated): ").strip()
        except EOFError as exc:
            raise SystemExit("No ID provided (EOF while prompting for --id)") from exc
        if not raw:
            _log(f"ID is required; choose one or more of {available}")
            continue
        try:
            values = normalize_selected_ids(raw)
        except ValueError:
            _log(f"Invalid IDs '{raw}'; choose one or more of {available}")
            continue
        missing = [v for v in values if v not in avail_set]
        if missing:
            _log(f"ID(s) {missing} unavailable; choose from {available}")
            continue
        return values


def prompt_selected_id(available: list[int]) -> int:
    """Ask for a single person ID (compat wrapper around :func:`prompt_selected_ids`)."""
    return prompt_selected_ids(available)[0]


def _records_by_frame_multi(
    payload: dict[str, Any], selected_ids: Sequence[int]
) -> dict[int, list[dict[str, Any]]]:
    """Map frame index → list of instances whose person_id is in ``selected_ids``."""
    wanted = set(normalize_selected_ids(selected_ids))
    records: dict[int, list[dict[str, Any]]] = {}
    for frame in payload.get("frames", []):
        frame_idx = _safe_int(frame.get("frame"))
        if frame_idx is None:
            continue
        keep: list[dict[str, Any]] = []
        for instance in frame.get("instances", []):
            value = _safe_int(instance.get("person_id", instance.get("sam_obj_id")))
            if value is not None and value in wanted:
                keep.append(instance)
        if keep:
            records[frame_idx] = keep
    return records


def _records_by_frame(payload: dict[str, Any], selected_id: int) -> dict[int, dict[str, Any]]:
    """Single-ID map frame → instance (compat)."""
    multi = _records_by_frame_multi(payload, [selected_id])
    return {frame: instances[0] for frame, instances in multi.items() if instances}


def _contours_by_frame_multi(
    run_dir: Path, selected_ids: Sequence[int]
) -> dict[int, dict[int, dict[str, Any]]]:
    """Map frame → {obj_id → contour object} for the selected ID set."""
    wanted = set(normalize_selected_ids(selected_ids))
    path: Path | None = None
    for candidate in (
        run_dir / "sam3" / "sam_contours.json",
        run_dir / "sam_contours.json",
        run_dir / "source_artifacts" / "sam3" / "sam_contours.json",
        run_dir / "source_artifacts" / "sam_contours.json",
    ):
        if candidate.is_file():
            path = candidate
            break
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, dict[int, dict[str, Any]]] = {}
    for frame in payload.get("frames", []):
        frame_idx = _safe_int(frame.get("frame", frame.get("session_frame")))
        if frame_idx is None:
            continue
        by_id: dict[int, dict[str, Any]] = {}
        for obj in frame.get("objects", []):
            oid = _safe_int(obj.get("obj_id"))
            if oid is not None and oid in wanted:
                by_id[oid] = obj
        if by_id:
            result[frame_idx] = by_id
    return result


def _contours_by_frame(run_dir: Path, selected_id: int) -> dict[int, dict[str, Any]]:
    """Single-ID map frame → contour (compat)."""
    multi = _contours_by_frame_multi(run_dir, [selected_id])
    return {frame: by_id[selected_id] for frame, by_id in multi.items() if selected_id in by_id}


def _color_for_id(selected_id: int) -> tuple[int, int, int]:
    hsv = np.asarray([[[int(selected_id * 47) % 180, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _object_polygons(contour: dict[str, Any] | None) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not contour:
        return polygons
    for raw in contour.get("polygons") or []:
        arr = np.asarray(raw, dtype=np.int32).reshape(-1, 2)
        if len(arr) >= 3 and abs(float(cv2.contourArea(arr))) >= 2.0:
            polygons.append(arr.reshape(-1, 1, 2))
    return polygons


def _draw_sam_contour_fill(
    image: np.ndarray,
    contour: dict[str, Any] | None,
    *,
    selected_id: int,
) -> np.ndarray:
    """Semi-transparent SAM silhouette fill (SAM3-like alpha blend)."""
    polygons = _object_polygons(contour)
    out = image.copy()
    if not polygons:
        return out
    color = _color_for_id(selected_id)
    overlay = out.copy()
    cv2.fillPoly(overlay, polygons, color, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, SAM_CONTOUR_FILL_ALPHA, out, 1.0 - SAM_CONTOUR_FILL_ALPHA, 0.0)


def _draw_sam_contour_outline_and_id(
    image: np.ndarray,
    instance: dict[str, Any],
    contour: dict[str, Any] | None,
    *,
    selected_id: int,
) -> np.ndarray:
    """SAM contour outline + bbox + ID/depth label (matches sam3dinov3 guidance layer)."""
    out = image
    color = _color_for_id(selected_id)
    polygons = _object_polygons(contour)
    if polygons:
        cv2.polylines(out, polygons, True, color, 2, cv2.LINE_AA)
    bbox = instance.get("bbox_xyxy")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(_safe_float(v))) for v in bbox[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # the ID only: cam_t_m[2] is the camera translation of SAM 3D Body (unscaled), not a height
        label = f"ID {selected_id}"
        _draw_box_label(out, label, x1, y1, color)
    return out


def _draw_mhr_skeleton(
    image: np.ndarray,
    instance: dict[str, Any],
    edges: list[tuple[int, int]],
    names: list[str],
) -> np.ndarray:
    """Draw the reprojected MHR70 skeleton (2D pixels), colored left/right/center."""
    kp2d = np.asarray(instance.get("keypoints_2d_px") or [], dtype=np.float32).reshape(-1, 2)
    if len(kp2d) == 0:
        return image
    out = image
    height, width = out.shape[:2]
    side_colors = [_side_color_bgr(name) for name in names]
    center = _rgb_to_bgr(COLOR_CENTER_RGB)
    for a, b in edges:
        if a >= len(kp2d) or b >= len(kp2d):
            continue
        pa, pb = kp2d[a], kp2d[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        color_a = side_colors[a] if a < len(side_colors) else center
        color_b = side_colors[b] if b < len(side_colors) else center
        color = color_a if color_a == color_b else center
        cv2.line(
            out,
            (int(round(float(pa[0]))), int(round(float(pa[1])))),
            (int(round(float(pb[0]))), int(round(float(pb[1])))),
            color,
            2,
            cv2.LINE_AA,
        )
    n_draw = min(len(kp2d), len(MHR70_NAMES))
    for idx in range(n_draw):
        point = kp2d[idx]
        if not np.isfinite(point).all():
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        if 0 <= x < width and 0 <= y < height:
            color = side_colors[idx] if idx < len(side_colors) else center
            cv2.circle(out, (x, y), 3, color, -1, cv2.LINE_AA)
    return out


def _draw_instance(
    image: np.ndarray,
    instance: dict[str, Any],
    contour: dict[str, Any] | None,
    edges: list[tuple[int, int]],
    names: list[str],
    *,
    selected_id: int,
) -> np.ndarray:
    """Match SAM3+DINOv3 overlay look: contour fill, left/right MHR skeleton, then outline/ID."""
    out = _draw_sam_contour_fill(image, contour, selected_id=selected_id)
    out = _draw_mhr_skeleton(out, instance, edges, names)
    return _draw_sam_contour_outline_and_id(out, instance, contour, selected_id=selected_id)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[Any, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from .vaila_sam import _open_sam3_video_writer
    except ImportError:
        from vaila_sam import _open_sam3_video_writer  # ty: ignore[unresolved-import]
    return _open_sam3_video_writer(
        path,
        fps,
        size,
        purpose="SAM3+DINOv3 visualize overlay",
    )


def _normalize_frame_records(
    records: dict[int, dict[str, Any]] | dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Accept single-instance or multi-instance per-frame maps."""
    if not records:
        return {}
    sample = next(iter(records.values()))
    if isinstance(sample, list):
        return {int(k): list(v) for k, v in records.items()}  # type: ignore[arg-type]
    return {int(k): [v] for k, v in records.items()}  # type: ignore[misc]


def _normalize_frame_contours(
    contours: dict[int, dict[str, Any]] | dict[int, dict[int, dict[str, Any]]],
    *,
    selected_ids: Sequence[int],
) -> dict[int, dict[int, dict[str, Any]]]:
    """Accept single-contour or multi-contour-by-id per-frame maps."""
    if not contours:
        return {}
    sample = next(iter(contours.values()))
    # Multi form: values are {obj_id:int → contour dict}.
    if (
        isinstance(sample, dict)
        and sample
        and all(isinstance(k, int) for k in sample)
        and all(isinstance(v, dict) for v in sample.values())
    ):
        return {
            int(frame): {int(oid): contour for oid, contour in by_id.items()}
            for frame, by_id in contours.items()  # type: ignore[assignment]
        }
    # Single-ID form: values are contour dicts keyed by frame only.
    sid = int(selected_ids[0]) if selected_ids else 0
    return {int(k): {sid: v} for k, v in contours.items() if isinstance(v, dict)}  # type: ignore[misc]


def render_selected_video(
    video_path: Path,
    output_path: Path,
    records: dict[int, dict[str, Any]] | dict[int, list[dict[str, Any]]],
    contours: dict[int, dict[str, Any]] | dict[int, dict[int, dict[str, Any]]],
    edges: list[tuple[int, int]],
    names: list[str],
    *,
    selected_id: int | None = None,
    selected_ids: Sequence[int] | None = None,
) -> tuple[Path, int, int]:
    """Render overlay video for one or more selected person IDs."""
    if selected_ids is None:
        if selected_id is None:
            raise ValueError("selected_id or selected_ids is required")
        ids = [int(selected_id)]
    else:
        ids = normalize_selected_ids(selected_ids)
    ids_tag = format_ids_tag(ids)
    frame_records = _normalize_frame_records(records)
    frame_contours = _normalize_frame_contours(contours, selected_ids=ids)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer, actual_path = _open_writer(output_path, fps, (width, height))
    frame_count = 0
    drawn_count = 0

    tqdm_cls = _try_import_tqdm()
    pbar = None
    id_label = f"ID {ids[0]}" if len(ids) == 1 else f"IDs {ids_tag}"
    if tqdm_cls is not None:
        pbar = tqdm_cls(
            total=total_frames if total_frames > 0 else None,
            desc=f">> sam3dinov3_visualize render {id_label}",
            unit="frame",
            leave=True,
        )
    else:
        _log(f"Rendering video overlay for {id_label} ({total_frames} frames)...")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            instances = frame_records.get(frame_count) or []
            contour_by_id = frame_contours.get(frame_count) or {}
            if instances:
                for instance in instances:
                    pid = _safe_int(instance.get("person_id", instance.get("sam_obj_id")))
                    if pid is None:
                        continue
                    frame = _draw_instance(
                        frame,
                        instance,
                        contour_by_id.get(pid),
                        edges,
                        names,
                        selected_id=pid,
                    )
                drawn_count += 1
            writer.write(frame)
            frame_count += 1

            if pbar is not None:
                pbar.update(1)
            elif total_frames > 0:
                step = max(1, total_frames // 10)
                if frame_count % step == 0 or frame_count == total_frames:
                    pct = (frame_count / total_frames) * 100.0
                    _log(f"Rendering {id_label}: frame {frame_count}/{total_frames} ({pct:.1f}%)")
            elif frame_count % 100 == 0:
                _log(f"Rendering {id_label}: frame {frame_count}...")
    finally:
        if pbar is not None:
            pbar.close()
        cap.release()
        writer.release()
    return actual_path, frame_count, drawn_count


def _filter_rows(
    path: Path,
    output: Path,
    id_columns: tuple[str, ...],
    selected_ids: int | Sequence[int],
) -> bool:
    if not path.exists():
        return False
    wanted = set(normalize_selected_ids(selected_ids))
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return False
        rows = list(reader)
    column = next((name for name in id_columns if name in reader.fieldnames), None)
    if column is None:
        return False
    selected = [row for row in rows if _safe_int(row.get(column)) in wanted]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return True


def _write_filtered_predictions(
    payload: dict[str, Any], output: Path, selected_ids: int | Sequence[int]
) -> bool:
    import copy

    wanted = set(normalize_selected_ids(selected_ids))
    output.parent.mkdir(parents=True, exist_ok=True)
    payload_copy = copy.deepcopy(payload)
    for frame in payload_copy.get("frames", []):
        frame["instances"] = [
            inst
            for inst in frame.get("instances", [])
            if _safe_int(inst.get("person_id", inst.get("sam_obj_id"))) in wanted
        ]
    with gzip.open(output, "wt", encoding="utf-8") as fh:
        json.dump(payload_copy, fh)
    return True


def _write_filtered_contours(path: Path, output: Path, selected_ids: int | Sequence[int]) -> bool:
    if not path.exists():
        return False
    wanted = normalize_selected_ids(selected_ids)
    wanted_set = set(wanted)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["object_ids"] = wanted
    for frame in payload.get("frames", []):
        frame["objects"] = [
            o for o in frame.get("objects", []) if _safe_int(o.get("obj_id")) in wanted_set
        ]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def _write_filtered_meshes(
    run_dir: Path, output_dir: Path, selected_ids: int | Sequence[int]
) -> int:
    """Extract selected person(s) vertices/cam_t from each per-frame mesh npz."""
    wanted = normalize_selected_ids(selected_ids)
    wanted_set = set(wanted)
    mesh_dir: Path | None = None
    for cand in (run_dir / "meshes", run_dir / "source_artifacts" / "meshes"):
        if cand.is_dir():
            mesh_dir = cand
            break
    if mesh_dir is None:
        return 0
    out_mesh_dir = output_dir / "meshes"
    written = 0
    for source in sorted(mesh_dir.glob("frame_*.npz")):
        with np.load(source) as data:
            obj_ids = np.asarray(data["obj_ids"])
            keep_idx = [i for i, oid in enumerate(obj_ids.tolist()) if int(oid) in wanted_set]
            if not keep_idx:
                continue
            out_mesh_dir.mkdir(parents=True, exist_ok=True)
            idx = np.asarray(keep_idx, dtype=np.int64)
            np.savez_compressed(
                out_mesh_dir / source.name,
                obj_ids=obj_ids[idx].astype(np.int32),
                vertices=data["vertices"][idx],
                cam_t=data["cam_t"][idx],
            )
            written += 1
    faces_path: Path | None = None
    for cand in (run_dir / "mesh_faces.npy", run_dir / "source_artifacts" / "mesh_faces.npy"):
        if cand.is_file():
            faces_path = cand
            break
    if written and faces_path is not None:
        shutil.copy2(faces_path, output_dir / "mesh_faces.npy")
    return written


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write an ASCII Wavefront OBJ (universally readable by Blender)."""
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# vailá SAM3+DINOv3 mesh frame\n")
        np.savetxt(fh, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, faces + 1, fmt="f %d %d %d")


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a compact binary-little-endian PLY (smaller than OBJ for dense meshes)."""
    n_v, n_f = len(vertices), len(faces)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n_v}\nproperty float x\nproperty float y\nproperty float z\n"
        f"element face {n_f}\nproperty list uchar int vertex_indices\nend_header\n"
    ).encode("ascii")
    face_records = np.zeros(n_f, dtype=[("count", "u1"), ("idx", "<i4", (3,))])
    face_records["count"] = 3
    face_records["idx"] = faces.astype(np.int32)
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(vertices.astype("<f4").tobytes())
        fh.write(face_records.tobytes())


def export_mesh_sequence(
    mesh_dir: Path,
    faces_path: Path,
    output_dir: Path,
    *,
    person_id: int | None = None,
    fmt: str = "obj",
    include_translation: bool = True,
) -> list[Path]:
    """Export a per-frame MHR mesh (``meshes/frame_NNNNNN.npz``) as an OBJ/PLY

    sequence Blender can import as a mesh-cache animation (built-in "Stop
    Motion OBJ" add-on: Edit > Preferences > Add-ons > enable "Mesh: Stop
    Motion OBJ", then Import > Mesh Sequence and point it at ``output_dir``).
    ``include_translation`` adds each frame's ``cam_t`` so the body keeps its
    camera-frame position instead of resetting to the origin every frame.
    """
    if fmt not in ("obj", "ply"):
        raise ValueError(f"Unsupported mesh export format: {fmt!r}")
    faces = np.asarray(np.load(faces_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    writer = _write_obj if fmt == "obj" else _write_ply
    sources = sorted(mesh_dir.glob("frame_*.npz"))
    total_files = len(sources)

    tqdm_cls = _try_import_tqdm()
    pbar = None
    if tqdm_cls is not None and total_files > 0:
        pbar = tqdm_cls(
            sources,
            desc=f">> sam3dinov3_visualize mesh export ({fmt.upper()})",
            unit="frame",
            leave=True,
        )
        iterable = pbar
    else:
        if total_files > 0:
            _log(f"Exporting {total_files} mesh frames ({fmt.upper()})...")
        iterable = sources

    for idx_file, source in enumerate(iterable, start=1):
        with np.load(source) as data:
            obj_ids = np.asarray(data["obj_ids"])
            if obj_ids.size == 0:
                continue
            if person_id is None:
                idx = 0
                if obj_ids.size > 1:
                    _log(
                        f"WARNING {source.name}: {obj_ids.size} people present; "
                        f"exporting obj_ids[0]={int(obj_ids[0])}. Pass person_id to choose."
                    )
            else:
                matches = np.nonzero(obj_ids == person_id)[0]
                if matches.size == 0:
                    continue
                idx = int(matches[0])
            vertices = np.asarray(data["vertices"][idx], dtype=np.float64)
            if include_translation:
                vertices = vertices + np.asarray(data["cam_t"][idx], dtype=np.float64)
        frame_idx = _safe_int(source.stem.rsplit("_", 1)[-1]) or 0
        target = output_dir / f"frame_{frame_idx:06d}.{fmt}"
        writer(target, vertices, faces)
        written.append(target)

        if pbar is None and total_files > 0:
            step = max(1, total_files // 10)
            if idx_file % step == 0 or idx_file == total_files:
                pct = (idx_file / total_files) * 100.0
                _log(f"Exporting mesh sequence: {idx_file}/{total_files} ({pct:.1f}%)")

    return written


_P_COL_RE = re.compile(r"^p(\d+)_(.+)$")


def _rebase_wide_markers_to_p0(path: Path) -> None:
    """Shift a legacy wide marker CSV that starts at ``p1`` down to ``p0``.

    Files that already contain ``p0`` are unchanged. Named columns such as
    ``nose_x`` are unchanged.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    indexes = [int(match.group(1)) for field in fields if (match := _P_COL_RE.match(field))]
    if not indexes or 0 in indexes or min(indexes) != 1:
        return
    renamed = [
        f"p{int(match.group(1)) - 1}_{match.group(2)}"
        if (match := _P_COL_RE.match(field))
        else field
        for field in fields
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=renamed)
        writer.writeheader()
        for row in rows:
            writer.writerow({renamed[i]: row.get(fields[i], "") for i in range(len(fields))})


def _find_first_existing(candidates: Sequence[Path | None]) -> Path | None:
    """Return the first candidate path that exists on disk."""
    for c in candidates:
        if c is not None and c.exists():
            return c
    return None


def resolve_markers_csv(
    run_dir: Path,
    selected_id: int,
    explicit_markers_csv: Path | str | None = None,
    *,
    video_stem: str | None = None,
) -> tuple[Path | None, str]:
    """Resolve the marker CSV to use and its source type:
    'explicit', 'auto_detected', or 'original'.
    """
    if explicit_markers_csv:
        p = Path(explicit_markers_csv).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Specified markers CSV not found: {p}")
        return p, "explicit"

    candidates: list[Path] = []
    seen: set[Path] = set()

    search_dirs = [run_dir]
    if run_dir.is_dir() and "visualized" in run_dir.name:
        search_dirs.append(run_dir.parent)

    patterns = [
        f"*id_{selected_id:02d}*overlay_markers.csv",
        f"*id_{selected_id:02d}*markers_corrected.csv",
        f"*id_{selected_id:02d}*corrected_markers.csv",
    ]
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pat in patterns:
            for match in sorted(d.glob(pat)):
                if match.is_file() and match not in seen:
                    candidates.append(match)
                    seen.add(match)

    if not candidates and run_dir.is_dir():
        for match in sorted(run_dir.glob("*overlay_markers.csv")):
            if match.is_file() and match not in seen:
                candidates.append(match)
                seen.add(match)

    if len(candidates) == 1:
        _log(f"Auto-detected unambiguous corrected markers CSV: {candidates[0]}")
        return candidates[0], "auto_detected"
    if len(candidates) > 1:
        c_names = ", ".join(c.name for c in candidates)
        raise ValueError(
            f"Ambiguous corrected marker files found for ID {selected_id} ({c_names}). "
            "Please specify --markers-csv explicitly."
        )

    id_tag = f"{selected_id:02d}"
    orig_candidates = []
    if video_stem:
        orig_candidates.extend(
            [
                run_dir / f"{video_stem}_id_{id_tag}_markers.csv",
                run_dir / "source_artifacts" / f"{video_stem}_id_{id_tag}_markers.csv",
            ]
        )
    orig_candidates.extend(
        [
            match
            for match in sorted(run_dir.glob(f"*_id_{id_tag}_markers.csv"))
            if not match.name.endswith("_overlay_markers.csv")
            and not match.name.endswith("_markers_corrected.csv")
        ]
    )
    if (run_dir / "source_artifacts").is_dir():
        orig_candidates.extend(
            [
                match
                for match in sorted(
                    (run_dir / "source_artifacts").glob(f"*_id_{id_tag}_markers.csv")
                )
                if not match.name.endswith("_overlay_markers.csv")
                and not match.name.endswith("_markers_corrected.csv")
            ]
        )
    for orig in orig_candidates:
        if orig and orig.is_file():
            return orig, "original"

    return None, "model_predictions"


def load_marker_coordinates(
    markers_csv_path: Path,
    expected_kpts: int = 70,
    names: list[str] | None = None,
) -> dict[int, np.ndarray]:
    """Load marker coordinates from a wide or long CSV.

    Returns:
        dict mapping frame_idx -> (n_kpts, 2) numpy array of float32.
    """
    markers_csv_path = Path(markers_csv_path).expanduser().resolve()
    if not markers_csv_path.is_file():
        raise FileNotFoundError(f"Markers CSV not found: {markers_csv_path}")

    with markers_csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise ValueError(f"Markers CSV is empty: {markers_csv_path}")

    # Long format: frame, (kpt_idx or marker/kpt_name), (x_px or x), (y_px or y)
    is_long = (
        "kpt_idx" in fields or "kpt_name" in fields or "marker" in fields or "marker_name" in fields
    ) and ("x_px" in fields or "x" in fields)
    if is_long:
        x_col = "x_px" if "x_px" in fields else "x"
        y_col = "y_px" if "y_px" in fields else "y"
        name_to_idx: dict[str, int] = {}
        if names:
            for idx, n in enumerate(names):
                name_to_idx[n] = idx
                name_to_idx[n.replace("-", "_")] = idx
        long_coords: dict[int, np.ndarray] = {}
        for row in rows:
            f_idx = _safe_int(row.get("frame"))
            if f_idx is None:
                continue
            k_idx = _safe_int(row.get("kpt_idx"))
            if k_idx is None:
                m_name = row.get("marker") or row.get("marker_name") or row.get("kpt_name")
                if m_name and m_name in name_to_idx:
                    k_idx = name_to_idx[m_name]
            if k_idx is None or k_idx >= expected_kpts:
                continue
            if f_idx not in long_coords:
                long_coords[f_idx] = np.full((expected_kpts, 2), np.nan, dtype=np.float32)
            long_coords[f_idx][k_idx, 0] = _safe_float(row.get(x_col), np.nan)
            long_coords[f_idx][k_idx, 1] = _safe_float(row.get(y_col), np.nan)
        return long_coords

    # Wide format: frame, p0_x, p0_y, ...
    indexes = [int(match.group(1)) for field in fields if (match := _P_COL_RE.match(field))]
    is_legacy_p1 = bool(indexes and 0 not in indexes and min(indexes) == 1)

    coords = {}
    for row in rows:
        f_idx = _safe_int(row.get("frame"))
        if f_idx is None:
            continue
        frame_arr = np.full((expected_kpts, 2), np.nan, dtype=np.float32)
        for i in range(expected_kpts):
            p_idx = i + 1 if is_legacy_p1 else i
            x_val = row.get(f"p{p_idx}_x")
            y_val = row.get(f"p{p_idx}_y")
            if (x_val is None or y_val is None) and names and i < len(names):
                raw_name = names[i]
                safe_name = raw_name.replace("-", "_")
                if x_val is None:
                    x_val = row.get(f"{raw_name}_x")
                if x_val is None:
                    x_val = row.get(f"{safe_name}_x")
                if y_val is None:
                    y_val = row.get(f"{raw_name}_y")
                if y_val is None:
                    y_val = row.get(f"{safe_name}_y")
            if x_val is not None and str(x_val).strip() != "":
                frame_arr[i, 0] = _safe_float(x_val, np.nan)
            if y_val is not None and str(y_val).strip() != "":
                frame_arr[i, 1] = _safe_float(y_val, np.nan)
        coords[f_idx] = frame_arr

    return coords


def validate_marker_coordinates(
    coords: dict[int, np.ndarray],
    video_info: dict[str, int | float],
    names: list[str],
) -> list[str]:
    """Validate frame coverage, bounds, and keypoint counts."""
    warnings: list[str] = []
    if not coords:
        raise ValueError("No valid marker coordinates loaded from CSV")

    n_video_frames = _safe_int(video_info.get("frames")) or 0
    width = _safe_int(video_info.get("width")) or 0
    height = _safe_int(video_info.get("height")) or 0

    min_frame = min(coords.keys())
    max_frame = max(coords.keys())

    if min_frame == 1 and max_frame == n_video_frames and n_video_frames > 0:
        warnings.append(
            f"CSV frames appear to be 1-based (1..{n_video_frames}) while video is 0-based. "
            "Coordinates will be shifted down by 1 to align with 0-based frames."
        )

    out_of_bounds = 0
    for _f_idx, arr in coords.items():
        valid_x = arr[:, 0][np.isfinite(arr[:, 0])]
        valid_y = arr[:, 1][np.isfinite(arr[:, 1])]
        is_x_oob = width > 0 and (np.any(valid_x < 0) or np.any(valid_x >= width))
        is_y_oob = height > 0 and (np.any(valid_y < 0) or np.any(valid_y >= height))
        if is_x_oob or is_y_oob:
            out_of_bounds += 1

    if out_of_bounds > 0:
        warnings.append(
            f"{out_of_bounds} frames have keypoints out of video bounds ({width}x{height})"
        )

    return warnings


def _write_keypoints2d_csv(
    output_path: Path,
    records: dict[int, dict[str, Any]],
    names: list[str],
    selected_id: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "person_id", "kpt_idx", "kpt_name", "x_px", "y_px"])
        for frame_idx in sorted(records):
            inst = records[frame_idx]
            kp2d = inst.get("keypoints_2d_px") or []
            for kpt_idx, name in enumerate(names):
                if kpt_idx < len(kp2d):
                    pt = kp2d[kpt_idx]
                    x_val = f"{float(pt[0]):.6f}" if np.isfinite(pt[0]) else ""
                    y_val = f"{float(pt[1]):.6f}" if np.isfinite(pt[1]) else ""
                    writer.writerow([frame_idx, selected_id, kpt_idx, name, x_val, y_val])


def _write_wide_markers_csv(
    output_path: Path,
    records: dict[int, dict[str, Any]],
    names: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["frame"]
    for i in range(len(names)):
        header.extend([f"p{i}_x", f"p{i}_y"])
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for frame_idx in sorted(records):
            inst = records[frame_idx]
            kp2d = inst.get("keypoints_2d_px") or []
            row = [str(frame_idx)]
            for i in range(len(names)):
                if i < len(kp2d) and np.isfinite(kp2d[i][0]) and np.isfinite(kp2d[i][1]):
                    row.extend([f"{float(kp2d[i][0]):.6f}", f"{float(kp2d[i][1]):.6f}"])
                else:
                    row.extend(["", ""])
            writer.writerow(row)


def write_selected_artifacts(
    run_dir: Path,
    output_dir: Path,
    selected_id: int | Sequence[int],
    payload: dict[str, Any],
    *,
    records: dict[int, dict[str, Any]] | None = None,
    names: list[str] | None = None,
    is_corrected: bool = False,
) -> list[str]:
    """Write filtered artifacts and preserve all source artifacts for provenance."""
    selected_ids = normalize_selected_ids(selected_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_artifacts"
    source_dir.mkdir(exist_ok=True)
    written: list[str] = []
    stem = Path(str(payload.get("video", "video.mp4"))).stem

    mapping = (
        (
            [
                run_dir / "sam3" / "sam_tracks.csv",
                run_dir / "sam_tracks.csv",
                run_dir / "source_artifacts" / "sam3" / "sam_tracks.csv",
                run_dir / "source_artifacts" / "sam_tracks.csv",
            ],
            output_dir / "sam_tracks.csv",
            ("obj_id",),
        ),
        (
            [
                run_dir / "sam3" / "sam_bbox_tracks.csv",
                run_dir / "sam_bbox_tracks.csv",
                run_dir / "source_artifacts" / "sam3" / "sam_bbox_tracks.csv",
                run_dir / "source_artifacts" / "sam_bbox_tracks.csv",
            ],
            output_dir / "sam_bbox_tracks.csv",
            ("obj_id",),
        ),
        (
            [
                run_dir / f"{stem}_sam3dinov3_keypoints3d.csv",
                run_dir / "source_artifacts" / f"{stem}_sam3dinov3_keypoints3d.csv",
            ],
            output_dir / f"{stem}_sam3dinov3_keypoints3d.csv",
            ("person_id",),
        ),
        (
            [
                run_dir / f"{stem}_sam3dinov3_camera.csv",
                run_dir / "source_artifacts" / f"{stem}_sam3dinov3_camera.csv",
            ],
            output_dir / f"{stem}_sam3dinov3_camera.csv",
            ("person_id",),
        ),
        (
            [
                run_dir / f"{stem}_sam3dinov3_joint_angles.csv",
                run_dir / "source_artifacts" / f"{stem}_sam3dinov3_joint_angles.csv",
            ],
            output_dir / f"{stem}_sam3dinov3_joint_angles.csv",
            ("person_id",),
        ),
    )
    for sources, target, columns in mapping:
        src = _find_first_existing(sources)
        if src and _filter_rows(src, target, columns, selected_ids):
            written.append(str(target.name))

    # 2D Keypoints (long format): regenerate from records if corrected (single-ID), else filter
    kp2d_out = output_dir / f"{stem}_sam3dinov3_keypoints2d.csv"
    if is_corrected and records and names and len(selected_ids) == 1:
        _write_keypoints2d_csv(kp2d_out, records, names, selected_ids[0])
        written.append(kp2d_out.name)
    else:
        src_kp2d = _find_first_existing(
            [
                run_dir / f"{stem}_sam3dinov3_keypoints2d.csv",
                run_dir / "source_artifacts" / f"{stem}_sam3dinov3_keypoints2d.csv",
            ]
        )
        if src_kp2d and _filter_rows(src_kp2d, kp2d_out, ("person_id",), selected_ids):
            written.append(kp2d_out.name)

    # Per-ID wide CSVs (markers + 3D): corrected rewrite only for single ID
    for sid in selected_ids:
        id_tag = f"{sid:02d}"
        markers_out = output_dir / f"{stem}_id_{id_tag}_markers.csv"
        if is_corrected and records and names and len(selected_ids) == 1:
            _write_wide_markers_csv(markers_out, records, names)
            written.append(markers_out.name)
        else:
            src_markers = _find_first_existing(
                [
                    run_dir / f"{stem}_id_{id_tag}_markers.csv",
                    run_dir / "source_artifacts" / f"{stem}_id_{id_tag}_markers.csv",
                ]
            )
            if src_markers and src_markers.is_file():
                shutil.copy2(src_markers, markers_out)
                _rebase_wide_markers_to_p0(markers_out)
                written.append(markers_out.name)

        for suffix in ("mhr70_3d", "mhr70_rec3d"):
            source = _find_first_existing(
                [
                    run_dir / f"{stem}_id_{id_tag}_{suffix}.csv",
                    run_dir / "source_artifacts" / f"{stem}_id_{id_tag}_{suffix}.csv",
                ]
            )
            if source and source.is_file():
                target = output_dir / source.name
                shutil.copy2(source, target)
                if suffix == "mhr70_rec3d":
                    _rebase_wide_markers_to_p0(target)
                written.append(target.name)

    pred_out = output_dir / f"{stem}_sam3dinov3_predictions.json.gz"
    if _write_filtered_predictions(payload, pred_out, selected_ids):
        written.append(pred_out.name)

    contour_source = _find_first_existing(
        [
            run_dir / "sam3" / "sam_contours.json",
            run_dir / "sam_contours.json",
            run_dir / "source_artifacts" / "sam3" / "sam_contours.json",
            run_dir / "source_artifacts" / "sam_contours.json",
        ]
    )
    if contour_source and _write_filtered_contours(
        contour_source, output_dir / "sam_contours.json", selected_ids
    ):
        written.append("sam_contours.json")

    n_meshes = _write_filtered_meshes(run_dir, output_dir, selected_ids)
    if n_meshes:
        written.append(f"meshes/ ({n_meshes} frames)")
        written.append("mesh_faces.npy")

    # Keep every original non-video artifact in a namespaced folder.
    for source in run_dir.rglob("*"):
        if not source.is_file() or source == output_dir or output_dir in source.parents:
            continue
        if "source_artifacts" in source.parts:
            continue
        if source.name.endswith("_sam3dinov3_overlay.mp4") or source.name.endswith(
            "_sam3dinov3_overlay.avi"
        ):
            continue
        relative = source.relative_to(run_dir)
        target = source_dir / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return written


def visualize_selected_ids(
    run_dir: Path,
    video_path: Path,
    selected_ids: Sequence[int],
    output_dir: Path,
    *,
    markers_csv: Path | str | None = None,
    overwrite: bool = False,
    export_mesh: str = "none",
) -> dict[str, Any]:
    """Render one overlay video + filtered artifacts for one or more person IDs."""
    selected_ids = normalize_selected_ids(selected_ids)
    ids_tag = format_ids_tag(selected_ids)
    run_dir = resolve_run_dir(run_dir, video_path)
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FileNotFoundError(f"Video not found or unsupported: {video_path}")
    payload = load_predictions(run_dir)
    video_info = validate_source_video(video_path, payload)
    total_frames = int(video_info["frames"])
    available = discover_ids(run_dir, payload)
    missing = [sid for sid in selected_ids if sid not in available]
    if missing:
        raise ValueError(f"ID(s) {missing} unavailable; choose from {available}")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir} (use --overwrite)")
    output_dir.mkdir(parents=True, exist_ok=True)

    records_multi = _records_by_frame_multi(payload, selected_ids)
    contours_multi = _contours_by_frame_multi(run_dir, selected_ids)
    names = payload.get("keypoint_names")
    if not isinstance(names, list) or not names:
        sample_inst = next(iter(records_multi.values()), [{}])[0]
        n_kpts = len(sample_inst.get("keypoints_3d_m", MHR70_NAMES))
        names = keypoint_names(n_kpts)
    names = [str(n) for n in names]
    edges = skeleton_edges(names)

    # Marker CSV override is single-ID only.
    single_records: dict[int, dict[str, Any]] | None = None
    is_corrected = False
    marker_source_label = "automatic"
    marker_csv_used = "model_predictions"
    if len(selected_ids) == 1:
        sid0 = selected_ids[0]
        resolved_markers_csv, marker_source_type = resolve_markers_csv(
            run_dir, sid0, markers_csv, video_stem=video_path.stem
        )
        is_corrected = (
            marker_source_type in ("explicit", "auto_detected") and resolved_markers_csv is not None
        )
        marker_source_label = "manual_correction" if is_corrected else "automatic"
        marker_csv_used = str(resolved_markers_csv) if resolved_markers_csv else "model_predictions"
        single_records = {f: insts[0] for f, insts in records_multi.items() if insts}

        if is_corrected and resolved_markers_csv is not None:
            _log(f"Marker source: MANUALLY CORRECTED ({marker_source_type})")
            _log(f"Marker CSV: {resolved_markers_csv}")
            corrected_coords = load_marker_coordinates(
                resolved_markers_csv, expected_kpts=len(names), names=names
            )
            warns = validate_marker_coordinates(corrected_coords, video_info, names)
            for w in warns:
                _log(f"WARNING: {w}")

            if (
                total_frames > 0
                and len(corrected_coords) == total_frames
                and min(corrected_coords.keys()) == 1
                and max(corrected_coords.keys()) == total_frames
            ):
                _log("Adjusting 1-based CSV frame numbering (1..N) to 0-based to match video.")
                corrected_coords = {f - 1: arr for f, arr in corrected_coords.items()}

            assert single_records is not None
            for f_idx, kp_arr in corrected_coords.items():
                if f_idx in single_records:
                    single_records[f_idx]["keypoints_2d_px"] = kp_arr.tolist()
                else:
                    single_records[f_idx] = {
                        "person_id": sid0,
                        "sam_obj_id": sid0,
                        "keypoints_2d_px": kp_arr.tolist(),
                        "bbox_xyxy": None,
                        "cam_t_m": [0.0, 0.0, 0.0],
                    }
                records_multi[f_idx] = [single_records[f_idx]]

            for frame_obj in payload.get("frames", []):
                f_idx = _safe_int(frame_obj.get("frame"))
                if f_idx is not None and f_idx in corrected_coords:
                    for inst in frame_obj.get("instances", []):
                        pid = _safe_int(inst.get("person_id", inst.get("sam_obj_id")))
                        if pid == sid0:
                            inst["keypoints_2d_px"] = corrected_coords[f_idx].tolist()
        else:
            _log(f"Marker source: AUTOMATIC ({marker_csv_used})")
    elif markers_csv is not None:
        _log("WARNING: --markers-csv ignored for multi-ID visualization (single-ID only).")
        _log(f"Marker source: AUTOMATIC ({marker_csv_used})")
    else:
        _log(f"Marker source: AUTOMATIC ({marker_csv_used})")

    id_label = f"ID {selected_ids[0]}" if len(selected_ids) == 1 else f"IDs {ids_tag}"
    _log(f"Starting visualization for {id_label} on video: {video_path.name}")
    if len(selected_ids) == 1:
        overlay_name = f"{video_path.stem}_sam3dinov3_id_{ids_tag}_overlay.mp4"
    else:
        overlay_name = f"{video_path.stem}_sam3dinov3_ids_{ids_tag}_overlay.mp4"
    _log(f"Rendering overlay video for {id_label}...")
    overlay, frames, drawn = render_selected_video(
        video_path,
        output_dir / overlay_name,
        records_multi,
        contours_multi,
        edges,
        names,
        selected_ids=selected_ids,
    )
    _log(f"Filtering and writing artifacts for {id_label}...")
    written = write_selected_artifacts(
        run_dir,
        output_dir,
        selected_ids,
        payload,
        records=single_records,
        names=names,
        is_corrected=is_corrected,
    )

    mesh_export_dir: Path | None = None
    n_mesh_exported = 0
    if export_mesh != "none":
        mesh_dir = output_dir / "meshes"
        faces_path = output_dir / "mesh_faces.npy"
        if mesh_dir.is_dir() and faces_path.is_file():
            if len(selected_ids) == 1:
                mesh_export_dir = output_dir / f"meshes_{export_mesh}"
                exported = export_mesh_sequence(
                    mesh_dir,
                    faces_path,
                    mesh_export_dir,
                    person_id=selected_ids[0],
                    fmt=export_mesh,
                )
                n_mesh_exported = len(exported)
            else:
                mesh_export_dir = output_dir / f"meshes_{export_mesh}"
                for sid in selected_ids:
                    sub = mesh_export_dir / f"id_{sid:02d}"
                    exported = export_mesh_sequence(
                        mesh_dir, faces_path, sub, person_id=sid, fmt=export_mesh
                    )
                    n_mesh_exported += len(exported)
            _log(
                f"Exported {n_mesh_exported} {export_mesh.upper()} mesh frames -> {mesh_export_dir}"
            )
        else:
            _log(
                "No meshes/ found for this selection; rerun sam3dinov3.py with --save-mesh "
                "to get body-mesh vertices for Blender export."
            )

    manifest: dict[str, Any] = {
        "schema": "vaila_sam3dinov3_selected_id_v1",
        "source_run": str(run_dir),
        "source_video": str(video_path),
        "selected_ids": selected_ids,
        "available_ids": available,
        "marker_source": marker_source_label,
        "marker_csv": marker_csv_used,
        "corrected_markers_applied": is_corrected,
        "fps": payload.get("fps"),
        "width": payload.get("width"),
        "height": payload.get("height"),
        "n_video_frames": frames,
        "n_frames_with_selected_person": drawn,
        "overlay": str(overlay),
        "artifacts": written,
        "mesh_export_format": export_mesh,
        "mesh_export_dir": str(mesh_export_dir) if mesh_export_dir is not None else None,
        "n_mesh_frames_exported": n_mesh_exported,
        "created_at": dt.datetime.now().astimezone().isoformat(),
    }
    if len(selected_ids) == 1:
        manifest["selected_id"] = selected_ids[0]
    (output_dir / "sam3dinov3_selected_id_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    blender_note = (
        f"mesh_export={export_mesh} -> {mesh_export_dir} ({n_mesh_exported} frames)\n"
        "Blender: Edit > Preferences > Add-ons > enable 'Mesh: Stop Motion OBJ', then\n"
        "File > Import > Mesh Sequence, and point it at the meshes_obj/ (or meshes_ply/) folder.\n"
        if mesh_export_dir is not None
        else "mesh_export=none (no meshes/ in the source run; rerun sam3dinov3.py with "
        "--save-mesh, then re-visualize with --export-mesh obj|ply for a Blender-ready sequence)\n"
    )
    ids_line = (
        f"selected_id={selected_ids[0]}\n"
        if len(selected_ids) == 1
        else f"selected_ids={selected_ids}\n"
    )
    (output_dir / "README_sam3dinov3_selected_id.txt").write_text(
        "vailá SAM3+DINOv3 3D selected-ID visualization\n"
        f"{ids_line}source_run={run_dir}\nsource_video={video_path}\n"
        "identity_authority=SAM3 obj_id (person_id == sam_obj_id)\n"
        "coordinate_units=full-frame pixels for 2D; metres for 3D; frame_index=zero-based\n"
        f"marker_source={marker_source_label} ({marker_csv_used})\n"
        "The root contains filtered artifacts. source_artifacts/ preserves the original run.\n"
        "Overlay style: SAM3 contour fill/outline + MHR70 skeleton "
        "(left=green RGB 0,255,0; right=orange RGB 255,128,0; center=blue) + depth label.\n"
        "Wide marker CSVs start at p0 (a legacy p1..pN header is shifted down on copy).\n"
        "C3D/rec3d pN map (0-based MHR70): p5/p7/p9/p11/p13=left shoulder/elbow/hip/knee/ankle; "
        "p6/p8/p10/p12/p14=right; p0=nose; p69=neck. Camera-frame mono (+Y down) is not L/R swapped — "
        "use monocular_dlt_align for lab C3D; bare Y/Z swap mirrors in Blender.\n"
        f"{video_path.stem}_sam3dinov3_joint_angles.csv: local (parent-relative) joint angles for the\n"
        "127-joint MHR rig -- Euler XYZ degrees + scalar-first (w,x,y,z) quaternion, from the\n"
        "model's own regressed rotations (present only for runs made after this feature was\n"
        "added; older runs have no *_joint_angles.csv to filter).\n"
        f"{blender_note}",
        encoding="utf-8",
    )
    _log(f"{id_label}: rendered {drawn}/{frames} frames -> {output_dir}")
    return manifest


def visualize_selected_id(
    run_dir: Path,
    video_path: Path,
    selected_id: int,
    output_dir: Path,
    *,
    markers_csv: Path | str | None = None,
    overwrite: bool = False,
    export_mesh: str = "none",
) -> dict[str, Any]:
    """Compat wrapper: visualize a single person ID."""
    return visualize_selected_ids(
        run_dir,
        video_path,
        [selected_id],
        output_dir,
        markers_csv=markers_csv,
        overwrite=overwrite,
        export_mesh=export_mesh,
    )


def _show_dialog_in_front(dialog: tk.Toplevel) -> None:
    dialog.update_idletasks()
    width = max(dialog.winfo_reqwidth(), dialog.winfo_width())
    height = max(dialog.winfo_reqheight(), dialog.winfo_height())
    x = max(0, (dialog.winfo_screenwidth() - width) // 2)
    y = max(0, (dialog.winfo_screenheight() - height) // 2)
    dialog.geometry(f"+{x}+{y}")
    dialog.deiconify()
    dialog.lift()
    dialog.attributes("-topmost", True)
    dialog.focus_force()
    dialog.after(350, lambda: dialog.attributes("-topmost", False))


def run_visualizer_gui(existing_root: tk.Tk | tk.Toplevel | None = None) -> None:
    owns_root = existing_root is None
    root = existing_root or tk.Tk()
    if owns_root:
        root.withdraw()
    dialog = tk.Toplevel(root)
    dialog.title("SAM3+DINOv3 3D — Visualize person ID(s)")
    dialog.resizable(False, False)
    vars_: dict[str, tk.StringVar] = {
        "run": tk.StringVar(),
        "video": tk.StringVar(),
        "markers_csv": tk.StringVar(),
        "output": tk.StringVar(),
        "id": tk.StringVar(),
        "available": tk.StringVar(value=""),
        "status": tk.StringVar(value="Choose a processed_sam3dinov3_* directory first."),
    }
    export_mesh_var = tk.BooleanVar(value=False)
    frame = ttk.Frame(dialog, padding=14)
    frame.grid(sticky="nsew")
    ttk.Label(
        frame,
        text="SAM3+DINOv3 3D — selected-ID visualization (1+ IDs)",
        font=("TkDefaultFont", 11, "bold"),
    ).grid(column=0, row=0, columnspan=3, sticky="w", pady=(0, 10))

    def on_id_change(_event: Any = None) -> None:
        raw_id = vars_["id"].get().strip()
        if not raw_id:
            return
        try:
            ids = normalize_selected_ids(raw_id)
            if len(ids) != 1:
                return
            run_val = vars_["run"].get()
            if run_val:
                vid_val = vars_["video"].get()
                cand_csv, cand_type = resolve_markers_csv(
                    Path(run_val), ids[0], video_stem=Path(vid_val).stem if vid_val else None
                )
                if cand_type in ("explicit", "auto_detected") and cand_csv is not None:
                    vars_["markers_csv"].set(str(cand_csv))
        except Exception:
            pass

    def browse_run() -> None:
        chosen = filedialog.askdirectory(parent=dialog, title="Processed SAM3+DINOv3 directory")
        if not chosen:
            return
        vars_["run"].set(chosen)
        try:
            resolved = resolve_run_dir(
                Path(chosen), Path(vars_["video"].get()) if vars_["video"].get() else None
            )
            payload = load_predictions(resolved)
            ids = discover_ids(resolved, payload)
            vars_["available"].set(", ".join(str(v) for v in ids))
            if ids:
                vars_["id"].set(str(ids[0]))
            source_video = discover_source_video(resolved, payload)
            if source_video is not None:
                vars_["video"].set(str(source_video))
                vars_["status"].set(f"Available IDs: {ids}. Source video found automatically.")
            else:
                vars_["status"].set(f"Available IDs: {ids}. Choose the matching source video.")
            if ids:
                try:
                    cand_csv, cand_type = resolve_markers_csv(
                        resolved, ids[0], video_stem=source_video.stem if source_video else None
                    )
                    if cand_type in ("explicit", "auto_detected") and cand_csv is not None:
                        vars_["markers_csv"].set(str(cand_csv))
                except Exception:
                    pass
        except Exception as exc:
            vars_["available"].set("")
            vars_["id"].set("")
            vars_["status"].set(str(exc))

    def browse_video() -> None:
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Original video",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*")],
        )
        if chosen:
            vars_["video"].set(chosen)

    def browse_markers_csv() -> None:
        chosen = filedialog.askopenfilename(
            parent=dialog,
            title="Corrected markers CSV (from getpixelvideo.py)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if chosen:
            vars_["markers_csv"].set(chosen)

    def browse_output() -> None:
        chosen = ask_output_directory(
            vars_["run"].get(), title="Output parent directory", parent=dialog
        )
        if chosen:
            vars_["output"].set(chosen)

    for row, (label, key, command) in enumerate(
        (
            ("Run directory", "run", browse_run),
            ("Video", "video", browse_video),
            ("Markers CSV (optional)", "markers_csv", browse_markers_csv),
            ("Output parent", "output", browse_output),
        ),
        start=1,
    ):
        ttk.Label(frame, text=label).grid(column=0, row=row, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=vars_[key], width=66).grid(
            column=1, row=row, sticky="ew", padx=6
        )
        ttk.Button(frame, text="Browse…", command=command).grid(column=2, row=row)
    ttk.Label(frame, text="ID(s) e.g. 0,1,2").grid(column=0, row=5, sticky="w", pady=3)
    id_entry = ttk.Entry(frame, textvariable=vars_["id"], width=40)
    id_entry.grid(column=1, row=5, sticky="w", padx=6)
    id_entry.bind("<FocusOut>", on_id_change)
    id_entry.bind("<Return>", on_id_change)
    ttk.Label(frame, text="Available").grid(column=0, row=6, sticky="w", pady=3)
    ttk.Label(frame, textvariable=vars_["available"], foreground="#555555", wraplength=480).grid(
        column=1, row=6, columnspan=2, sticky="w", padx=6
    )
    ttk.Checkbutton(
        frame,
        text="Export mesh sequence (.obj, for Blender — needs --save-mesh in the source run)",
        variable=export_mesh_var,
    ).grid(column=0, row=7, columnspan=3, sticky="w", pady=2)
    ttk.Label(frame, textvariable=vars_["status"], foreground="#7a4f00", wraplength=560).grid(
        column=0, row=8, columnspan=3, sticky="w", pady=(8, 4)
    )

    def run() -> None:
        try:
            run_dir = Path(vars_["run"].get()).expanduser()
            video = Path(vars_["video"].get()).expanduser()
            selected = normalize_selected_ids(vars_["id"].get())
            markers_csv_text = vars_["markers_csv"].get().strip()
            markers_csv_path = Path(markers_csv_text).expanduser() if markers_csv_text else None
            output_text = vars_["output"].get().strip()
            output_parent = Path(output_text).expanduser() if output_text else run_dir
            output = _unique_output_dir_for_ids(output_parent, video, selected)
        except Exception as exc:
            messagebox.showerror(
                "SAM3+DINOv3 3D visualization", f"Invalid settings: {exc}", parent=dialog
            )
            return
        export_mesh = "obj" if export_mesh_var.get() else "none"
        id_args = [str(i) for i in selected]
        cli = [
            "uv run python -u vaila/sam3dinov3_visualize.py",
            "--sam3d-results",
            str(run_dir),
            "--video",
            str(video),
            "--id",
            *id_args,
            "--output",
            str(output),
        ]
        if markers_csv_path is not None:
            cli.extend(["--markers-csv", str(markers_csv_path)])
        if export_mesh != "none":
            cli.extend(["--export-mesh", export_mesh])
        _log("GUI equivalent CLI: " + " ".join(cli))
        vars_["status"].set("Rendering selected ID(s); the window remains responsive…")

        def worker() -> None:
            try:
                result = visualize_selected_ids(
                    run_dir,
                    video,
                    selected,
                    output,
                    markers_csv=markers_csv_path,
                    export_mesh=export_mesh,
                )
                label = (
                    f"ID {selected[0]}" if len(selected) == 1 else f"IDs {format_ids_tag(selected)}"
                )
                dialog.after(
                    0,
                    lambda: (
                        vars_["status"].set(f"Done: {result['overlay']}"),
                        messagebox.showinfo(
                            "SAM3+DINOv3 3D visualization",
                            f"Finished {label}.\n\n{output}",
                            parent=dialog,
                        ),
                    ),
                )
            except Exception as exc:
                error_text = str(exc)
                dialog.after(
                    0,
                    lambda: (
                        vars_["status"].set(error_text),
                        messagebox.showerror(
                            "SAM3+DINOv3 3D visualization", error_text, parent=dialog
                        ),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(frame, text="Run selected ID(s)", command=run).grid(
        column=1, row=9, sticky="w", padx=6, pady=(10, 0)
    )
    ttk.Button(frame, text="Cancel", command=dialog.destroy).grid(column=2, row=9, pady=(10, 0))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    _show_dialog_in_front(dialog)
    if owns_root:
        root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one or more SAM3+DINOv3 3D person IDs from an existing run."
    )
    parser.add_argument(
        "--sam3d-results",
        "--input-dir",
        dest="sam3d_results",
        required=False,
        type=Path,
        help="Per-video or processed_sam3dinov3_* directory.",
    )
    parser.add_argument("--video", "-i", type=Path, help="Original source video.")
    parser.add_argument(
        "--id",
        dest="selected_ids",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "SAM/person ID(s) to visualize (one or more). Accepts space-separated "
            "values and/or commas (e.g. --id 0 1 2 or --id 0,1,2). "
            "If omitted in CLI mode, prompt interactively."
        ),
    )
    parser.add_argument(
        "--markers-csv",
        type=Path,
        default=None,
        help=(
            "Override 2D keypoint coordinates with a manually corrected CSV "
            "(e.g. from getpixelvideo.py). If omitted, an unambiguous *overlay_markers.csv "
            "or *_markers_corrected.csv in the run directory is auto-detected."
        ),
    )
    parser.add_argument("--output", "-o", type=Path, help="New output directory for this ID.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow a non-empty output directory."
    )
    parser.add_argument(
        "--export-mesh",
        choices=MESH_EXPORT_FORMATS,
        default="none",
        help=(
            "Export the filtered per-frame body mesh as an OBJ/PLY sequence for "
            "Blender (needs --save-mesh in the source sam3dinov3.py run)."
        ),
    )
    parser.add_argument(
        "--list-ids", action="store_true", help="Print available IDs and exit without rendering."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without writing a video.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help=(
            "Batch-visualize every processed_sam3dinov3_* run directory found "
            "under --sam3d-results, instead of a single run. --video is not used "
            "in this mode (each run's source video is auto-discovered). If "
            "--id is omitted, every available id is rendered separately per "
            "video; with multiple --id values one composite output is written. "
            "Never re-renders an id/set that already has a visualized output. CLI-only."
        ),
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=-1,
        metavar="N",
        help="With --recursive: -1 unlimited (default), 0 root only, 1-99 levels below --sam3d-results.",
    )
    return parser


def _run_recursive_batch(args: argparse.Namespace) -> int:
    root = args.sam3d_results.expanduser().resolve()
    run_dirs = find_run_directories(root, args.depth)
    if not run_dirs:
        print(
            f"Error: no processed_sam3dinov3_* run directories found under {root} "
            f"(--recursive --depth {args.depth})",
            file=sys.stderr,
        )
        return 2
    explicit_output_parent = args.output.expanduser().resolve() if args.output is not None else None
    total_rendered = 0
    total_skipped = 0
    total_failed = 0
    for run_index, run_dir in enumerate(run_dirs, start=1):
        print(f"\n[Run {run_index}/{len(run_dirs)}] {run_dir}")
        for video_dir in _iter_video_run_dirs(run_dir):
            try:
                payload = load_predictions(video_dir)
                source_video = discover_source_video(video_dir, payload)
                if source_video is None:
                    raise FileNotFoundError("could not locate source video")
                available_ids = discover_ids(video_dir, payload)
            except Exception as exc:
                print(f"[WARN] {video_dir}: {exc}", file=sys.stderr)
                total_failed += 1
                continue
            # No --id → one output per available ID (legacy overnight batch).
            # With --id list → one composite (or single) output for that set.
            if args.selected_ids is None:
                job_sets: list[list[int]] = [[tid] for tid in available_ids]
            else:
                try:
                    requested = normalize_selected_ids(args.selected_ids)
                except ValueError as exc:
                    print(f"[WARN] {video_dir}: {exc}", file=sys.stderr)
                    total_failed += 1
                    continue
                missing = [tid for tid in requested if tid not in available_ids]
                if missing:
                    print(
                        f"[WARN] {source_video.name}: ID(s) {missing} unavailable "
                        f"(has {available_ids}), skipping",
                        file=sys.stderr,
                    )
                    total_failed += 1
                    continue
                job_sets = [requested]
            output_parent = (
                explicit_output_parent if explicit_output_parent is not None else video_dir.parent
            )
            for target_ids in job_sets:
                tag = format_ids_tag(target_ids)
                if len(target_ids) == 1:
                    pattern = f"{source_video.stem}_sam3dinov3_visualized_id_{tag}*"
                else:
                    pattern = f"{source_video.stem}_sam3dinov3_visualized_ids_{tag}*"
                existing = sorted(output_parent.glob(pattern))
                if existing:
                    print(f"[SKIP] Already visualized: {source_video.name} id(s) {tag}")
                    total_skipped += 1
                    continue
                output_dir = _unique_output_dir_for_ids(output_parent, source_video, target_ids)
                try:
                    visualize_selected_ids(
                        video_dir,
                        source_video,
                        target_ids,
                        output_dir,
                        markers_csv=args.markers_csv,
                        overwrite=args.overwrite,
                        export_mesh=args.export_mesh,
                    )
                    total_rendered += 1
                except Exception as exc:
                    print(f"[ERROR] {source_video.name} id(s) {tag}: {exc}", file=sys.stderr)
                    total_failed += 1
    print(
        f"\nRecursive batch done: {total_rendered} rendered, {total_skipped} skipped, "
        f"{total_failed} failed across {len(run_dirs)} run directories"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.recursive:
        if args.sam3d_results is None:
            parser.error("--recursive requires --sam3d-results/--input-dir as the root to walk")
        if args.depth < -1 or args.depth > 99:
            parser.error("--depth must be -1 (unlimited), 0 (root only), or 1-99.")
        return _run_recursive_batch(args)
    if not any(
        (
            args.sam3d_results,
            args.video,
            args.selected_ids is not None,
            args.output,
            args.markers_csv,
        )
    ):
        run_visualizer_gui()
        return 0
    if args.sam3d_results is None:
        parser.error("--sam3d-results/--input-dir is required in CLI mode")
    run_dir = resolve_run_dir(args.sam3d_results, args.video)
    payload = load_predictions(run_dir)
    video_path = args.video
    if video_path is None:
        video_path = discover_source_video(run_dir, payload)
        if video_path is None:
            parser.error(
                "--video/-i could not be auto-discovered; please specify --video explicitly"
            )
        print(f">> Source video auto-discovered: {video_path}")
    else:
        video_path = video_path.expanduser().resolve()

    ids = discover_ids(run_dir, payload)
    print(f">> Available person IDs: {ids}")
    if args.list_ids:
        return 0
    if args.selected_ids is None:
        if len(ids) == 1:
            selected_ids = [ids[0]]
            print(f">> Single person ID {selected_ids[0]} auto-selected.")
        else:
            selected_ids = prompt_selected_ids(ids)
    else:
        try:
            selected_ids = normalize_selected_ids(args.selected_ids)
        except ValueError as exc:
            parser.error(str(exc))
        missing = [sid for sid in selected_ids if sid not in ids]
        if missing:
            parser.error(f"ID(s) {missing} unavailable; choose from {ids}")

    ids_tag = format_ids_tag(selected_ids)
    output_dir = args.output
    if output_dir is None:
        if len(selected_ids) == 1 and run_dir.name.endswith(f"_id_{ids_tag}"):
            output_dir = run_dir.parent / f"{run_dir.name}_corrected"
        else:
            suffix = "_corrected" if args.markers_csv is not None and len(selected_ids) == 1 else ""
            output_dir = _unique_output_dir_for_ids(run_dir.parent, video_path, selected_ids)
            if suffix and not str(output_dir).endswith(suffix):
                output_dir = output_dir.parent / f"{output_dir.name}{suffix}"
        print(f">> Defaulting output directory to: {output_dir}")
    else:
        output_dir = output_dir.expanduser().resolve()

    if args.dry_run:
        validate_source_video(video_path, payload)
        if len(selected_ids) == 1:
            resolved_csv, marker_type = resolve_markers_csv(
                run_dir, selected_ids[0], args.markers_csv, video_stem=video_path.stem
            )
            marker_note = f"{marker_type} ({resolved_csv})"
        else:
            marker_note = "automatic (multi-ID; --markers-csv ignored)"
        print(
            f">> Dry-run OK: video={video_path} run_dir={run_dir} "
            f"ids={selected_ids} output={output_dir}"
        )
        print(f">> Marker source: {marker_note}")
        return 0

    visualize_selected_ids(
        run_dir,
        video_path,
        selected_ids,
        output_dir,
        markers_csv=args.markers_csv,
        overwrite=args.overwrite,
        export_mesh=args.export_mesh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
