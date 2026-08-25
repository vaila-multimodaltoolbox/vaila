"""
Project: vailá
Script: sam3sapiens2_visualize.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Creation Date: 31 July 2026
Update Date: 24 August 2026
Version: 0.3.112

Description:
    CPU-only rerenderer for an existing SAM3+Sapiens2 run. It selects one
    SAM object ID, draws its contour, bbox, ID, and Sapiens2 keypoints on the
    original video, and writes ID-specific tracking/pose/contour artifacts.

Usage:
    uv run python -u vaila/sam3sapiens2_visualize.py \
        --sam-results /path/to/processed_sam3sapiens2_.../video_stem \
        --video /path/to/video.mp4 --id 2 --output /path/to/output

    # Omit --id to be prompted interactively with the available SAM IDs.
    # GUI: omit all arguments, or use Frame B -> YOLO + FB -> SAM3+Sapiens2 Visualize ID
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_KPT_THR = 0.30
DEFAULT_SKELETON_RADIUS = 4
DEFAULT_SKELETON_THICKNESS = 2
# Match SAM3 composite alpha (~0.45) for selected-ID contour fills.
SAM_CONTOUR_FILL_ALPHA = 0.45

# Sapiens2 Goliath-308 topology: body joints 0..20, right hand 21..41, left hand 42..62,
# arms 63..68, neck 69.
# Fallback left/right colors match keypoints308.py (RGB; converted to BGR in OpenCV).
COLOR_LEFT_RGB = (0, 255, 0)
COLOR_RIGHT_RGB = (255, 128, 0)
COLOR_CENTER_RGB = (51, 153, 255)
LEFT_BODY_INDICES = frozenset(
    {
        1,
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        16,
        17,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        65,
        67,
    }
)
RIGHT_BODY_INDICES = frozenset(
    {
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        64,
        66,
        68,
    }
)
BODY_EDGES = (
    # Head
    (0, 1),  # nose - left_eye
    (0, 2),  # nose - right_eye
    (1, 3),  # left_eye - left_ear
    (2, 4),  # right_eye - right_ear
    (1, 2),  # left_eye - right_eye
    # Shoulders & Upper limbs
    (5, 6),  # left_shoulder - right_shoulder
    (5, 7),  # left_shoulder - left_elbow
    (7, 62),  # left_elbow - left_wrist
    (6, 8),  # right_shoulder - right_elbow
    (8, 41),  # right_elbow - right_wrist
    # Torso
    (5, 9),  # left_shoulder - left_hip
    (6, 10),  # right_shoulder - right_hip
    (9, 10),  # left_hip - right_hip
    # Left lower limb & foot
    (9, 11),  # left_hip - left_knee
    (11, 13),  # left_knee - left_ankle
    (13, 17),  # left_ankle - left_heel
    (13, 15),  # left_ankle - left_big_toe
    (13, 16),  # left_ankle - left_small_toe
    (17, 15),  # left_heel - left_big_toe
    (17, 16),  # left_heel - left_small_toe
    (15, 16),  # left_big_toe - left_small_toe
    # Right lower limb & foot
    (10, 12),  # right_hip - right_knee
    (12, 14),  # right_knee - right_ankle
    (14, 20),  # right_ankle - right_heel
    (14, 18),  # right_ankle - right_big_toe
    (14, 19),  # right_ankle - right_small_toe
    (20, 18),  # right_heel - right_big_toe
    (20, 19),  # right_heel - right_small_toe
    (18, 19),  # right_big_toe - right_small_toe
    # Neck & Spine
    (69, 0),  # neck - nose
    (69, 5),  # neck - left_shoulder
    (69, 6),  # neck - right_shoulder
)

_STYLE_UNSET = object()
_SAPIENS_STYLE_CACHE: Any = _STYLE_UNSET


def _log(message: str) -> None:
    # Detached/long-running renders must not die on a dropped terminal (EIO/BrokenPipe).
    with contextlib.suppress(OSError, BrokenPipeError):
        print(f">> vaila/sam3sapiens2_visualize: {message}", flush=True)


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


def _prediction_path(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*_sam3sapiens2_predictions.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_sam3sapiens2_predictions.json in {run_dir}")
    return candidates[0]


def resolve_run_dir(path: Path, video_path: Path | None = None) -> Path:
    """Resolve a direct per-video output or a processed batch parent."""
    path = path.expanduser().resolve()
    if path.is_dir() and (path / "sam3").is_dir():
        return path
    if video_path is not None:
        stem_dir = path / video_path.stem
        if (stem_dir / "sam3").is_dir() or list(stem_dir.glob("*_sam3sapiens2_predictions.json")):
            return stem_dir
    direct = list(path.glob("*_sam3sapiens2_predictions.json")) if path.is_dir() else []
    if direct:
        return path
    candidates = (
        sorted(
            p
            for p in path.iterdir()
            if p.is_dir()
            and (p / "sam3").is_dir()
            and list(p.glob("*_sam3sapiens2_predictions.json"))
        )
        if path.is_dir()
        else []
    )
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(p.name for p in candidates[:8])
    raise FileNotFoundError(
        f"Could not resolve a per-video SAM3+Sapiens2 directory from {path}. "
        f"Candidates: {names or 'none'}"
    )


def load_predictions(run_dir: Path) -> dict[str, Any]:
    path = _prediction_path(run_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "vaila_sam3sapiens2_v1":
        raise ValueError(f"Unsupported predictions schema in {path}: {payload.get('schema')!r}")
    return payload


def discover_source_video(run_dir: Path, payload: dict[str, Any] | None = None) -> Path | None:
    """Find the exact source video recorded by a SAM3+Sapiens2 run.

    The summary's absolute path is authoritative. Relative fallbacks keep runs
    usable after the whole input/results tree has been moved to another disk.
    """
    run_dir = run_dir.expanduser().resolve()
    summary = run_dir / "sam3sapiens2_summary.json"
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

    expected_frames = len(payload.get("frames", []))
    expected_size = payload.get("image_size") or []
    expected_height = _safe_int(expected_size[0]) if len(expected_size) >= 2 else None
    expected_width = _safe_int(expected_size[1]) if len(expected_size) >= 2 else None
    mismatches: list[str] = []
    if expected_frames and frames != expected_frames:
        mismatches.append(f"frames={frames}, expected {expected_frames}")
    if expected_width and expected_height and (width, height) != (expected_width, expected_height):
        mismatches.append(f"size={width}x{height}, expected {expected_width}x{expected_height}")
    if mismatches:
        raise ValueError(
            f"The selected video does not match this SAM3+Sapiens2 run ({'; '.join(mismatches)}). "
            "Choose the synchronized/cropped source video recorded in sam3sapiens2_summary.json."
        )
    return {"frames": frames, "width": width, "height": height, "fps": fps}


def _unique_gui_output_dir(output_parent: Path, video_path: Path, selected_id: int) -> Path:
    """Return a new ID-specific child directory below a GUI-selected parent."""
    output_parent = output_parent.expanduser().resolve()
    base = output_parent / f"{video_path.stem}_sam3sapiens2_visualized_id_{selected_id:02d}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate


def discover_ids(run_dir: Path, payload: dict[str, Any] | None = None) -> list[int]:
    payload = payload or load_predictions(run_dir)
    ids: set[int] = set()
    for frame in payload.get("frames", []):
        for instance in frame.get("instances", []):
            value = _safe_int(instance.get("stable_id", instance.get("sam_obj_id")))
            if value is not None:
                ids.add(value)
    if not ids:
        tracks = run_dir / "sam3" / "sam_tracks.csv"
        if tracks.exists():
            with tracks.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    value = _safe_int(row.get("obj_id"))
                    if value is not None:
                        ids.add(value)
    if not ids:
        raise ValueError(f"No SAM IDs found in {run_dir}")
    return sorted(ids)


def prompt_selected_id(available: list[int]) -> int:
    """Ask for a SAM/stable ID on stdin until a valid choice is entered."""
    if not available:
        raise ValueError("No SAM IDs available to prompt")
    while True:
        try:
            raw = input(">> Enter SAM/stable ID to visualize: ").strip()
        except EOFError as exc:
            raise SystemExit("No ID provided (EOF while prompting for --id)") from exc
        if not raw:
            _log(f"ID is required; choose one of {available}")
            continue
        try:
            value = int(raw)
        except ValueError:
            _log(f"Invalid integer '{raw}'; choose one of {available}")
            continue
        if value not in available:
            _log(f"ID {value} is unavailable; choose one of {available}")
            continue
        return value


def _records_by_frame(payload: dict[str, Any], selected_id: int) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for frame in payload.get("frames", []):
        frame_idx = _safe_int(frame.get("frame_index", frame.get("frame")))
        if frame_idx is None:
            continue
        for instance in frame.get("instances", []):
            value = _safe_int(instance.get("stable_id", instance.get("sam_obj_id")))
            if value == selected_id:
                records[frame_idx] = instance
                break
    return records


def _contours_by_frame(run_dir: Path, selected_id: int) -> dict[int, dict[str, Any]]:
    path = run_dir / "sam3" / "sam_contours.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing contour artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[int, dict[str, Any]] = {}
    for frame in payload.get("frames", []):
        frame_idx = _safe_int(frame.get("frame", frame.get("session_frame")))
        if frame_idx is None:
            continue
        for obj in frame.get("objects", []):
            if _safe_int(obj.get("obj_id")) == selected_id:
                result[frame_idx] = obj
                break
    return result


def _color_for_id(selected_id: int) -> tuple[int, int, int]:
    hsv = np.asarray([[[int(selected_id * 47) % 180, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    return int(color[2]), int(color[1]), int(color[0])


def _side_color_rgb(index: int, names: list[str] | None = None) -> tuple[int, int, int]:
    if names is not None and 0 <= index < len(names):
        label = names[index].lower()
        if "left" in label:
            return COLOR_LEFT_RGB
        if "right" in label:
            return COLOR_RIGHT_RGB
        return COLOR_CENTER_RGB
    if index in LEFT_BODY_INDICES:
        return COLOR_LEFT_RGB
    if index in RIGHT_BODY_INDICES:
        return COLOR_RIGHT_RGB
    return COLOR_CENTER_RGB


def _load_sapiens_overlay_style() -> dict[str, Any] | None:
    """Load Sapiens2 visualize_keypoints + 308 metainfo without pose weights."""
    global _SAPIENS_STYLE_CACHE
    if _SAPIENS_STYLE_CACHE is not _STYLE_UNSET:
        return None if _SAPIENS_STYLE_CACHE is None else dict(_SAPIENS_STYLE_CACHE)
    try:
        try:
            from .vaila_sapiens import (  # type: ignore[attr-defined]
                _load_visualize_keypoints,
                _require_sapiens_installed,
                _sapiens_pose_context,
            )
        except ImportError:
            from vaila_sapiens import (  # type: ignore[no-redef]  # ty: ignore[unresolved-import]
                _load_visualize_keypoints,
                _require_sapiens_installed,
                _sapiens_pose_context,
            )

        _require_sapiens_installed()
        visualize_fn = _load_visualize_keypoints()
        if visualize_fn is None:
            raise RuntimeError("pose_render_utils.visualize_keypoints unavailable")
        with _sapiens_pose_context():
            from sapiens.pose.datasets import (  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
                parse_pose_metainfo,
            )

            meta = parse_pose_metainfo({"from_file": "configs/_base_/keypoints308.py"})
        style = {
            "visualize": visualize_fn,
            "skeleton": list(meta["skeleton_links"]),
            "kpt_color": np.asarray(meta["keypoint_colors"]),
            "link_color": np.asarray(meta["skeleton_link_colors"]),
            "keypoint_names": [
                str(meta["keypoint_id2name"][i]) for i in range(int(meta["num_keypoints"]))
            ],
        }
        _SAPIENS_STYLE_CACHE = style
        _log("Using Sapiens2 pose style (left/right skeleton colors from keypoints308)")
        return dict(style)
    except Exception as exc:
        _SAPIENS_STYLE_CACHE = None
        _log(f"Sapiens2 pose style unavailable ({exc}); using left/right OpenCV fallback")
        return None


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
    """SAM contour outline + bbox + ID label (matches sam3sapiens2 guidance layer)."""
    out = image
    color = _color_for_id(selected_id)
    polygons = _object_polygons(contour)
    if polygons:
        cv2.polylines(out, polygons, True, color, 2, cv2.LINE_AA)
    bbox = instance.get("sam_bbox_xyxy") or instance.get("bbox")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(_safe_float(v))) for v in bbox[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        score = instance.get("sam_score")
        label = f"SAM #{selected_id}"
        if score is not None:
            label = f"{label} {_safe_float(score):.2f}"
        else:
            label = f"SAM/Sapiens2 ID {selected_id}"
        cv2.putText(
            out,
            label,
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _draw_sapiens_skeleton(
    image: np.ndarray,
    instance: dict[str, Any],
    *,
    kpt_thr: float,
    draw_all_keypoints: bool,
    style: dict[str, Any] | None,
    keypoint_names: list[str] | None = None,
) -> np.ndarray:
    """Draw Sapiens2 skeleton with left/right colors (official style when available)."""
    points = np.asarray(instance.get("keypoints") or [], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(instance.get("keypoint_scores") or [], dtype=np.float32).reshape(-1)
    if len(points) == 0:
        return image
    if len(scores) < len(points):
        padded = np.ones(len(points), dtype=np.float32)
        padded[: len(scores)] = scores
        scores = padded

    if style is not None:
        skeleton = list(style["skeleton"])
        kpt_color = style["kpt_color"]
        link_color = style["link_color"]
        if not draw_all_keypoints:
            skeleton = [(a, b) for a, b in skeleton if a < 21 and b < 21]
            n = min(21, len(points))
            points = points[:n]
            scores = scores[:n]
            if hasattr(kpt_color, "__len__") and len(kpt_color) > n:
                kpt_color = np.asarray(kpt_color)[:n]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_rgb = style["visualize"](
            image=image_rgb,
            keypoints=[points],
            keypoints_visible=[np.ones(len(points), dtype=bool)],
            keypoint_scores=[scores],
            radius=DEFAULT_SKELETON_RADIUS,
            thickness=DEFAULT_SKELETON_THICKNESS,
            kpt_thr=kpt_thr,
            skeleton=skeleton,
            kpt_color=kpt_color,
            link_color=link_color,
        )
        return cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    # OpenCV fallback: COCO-21 edges + left/right colors matching keypoints308.
    out = image.copy()
    names = keypoint_names
    n_draw = len(points) if draw_all_keypoints else min(21, len(points))
    valid = np.zeros(len(points), dtype=bool)
    for idx in range(len(points)):
        x, y = points[idx]
        score = float(scores[idx]) if idx < len(scores) else 1.0
        valid[idx] = score >= kpt_thr and np.isfinite(x) and np.isfinite(y)
    for left, right in BODY_EDGES:
        if left >= n_draw or right >= n_draw:
            continue
        if not (valid[left] and valid[right]):
            continue
        color_rgb = COLOR_CENTER_RGB
        if left in LEFT_BODY_INDICES and right in LEFT_BODY_INDICES:
            color_rgb = COLOR_LEFT_RGB
        elif left in RIGHT_BODY_INDICES and right in RIGHT_BODY_INDICES:
            color_rgb = COLOR_RIGHT_RGB
        elif names is not None:
            left_c = _side_color_rgb(left, names)
            right_c = _side_color_rgb(right, names)
            if left_c == right_c:
                color_rgb = left_c
        cv2.line(
            out,
            tuple(np.round(points[left]).astype(int)),
            tuple(np.round(points[right]).astype(int)),
            _rgb_to_bgr(color_rgb),
            DEFAULT_SKELETON_THICKNESS,
            cv2.LINE_AA,
        )
    for idx in range(n_draw):
        if not valid[idx]:
            continue
        radius = DEFAULT_SKELETON_RADIUS if idx < 21 else 2
        color = _rgb_to_bgr(_side_color_rgb(idx, names))
        pt = tuple(np.round(points[idx]).astype(int))
        cv2.circle(out, pt, radius, color, -1, cv2.LINE_AA)
    return out


def _draw_instance(
    image: np.ndarray,
    instance: dict[str, Any],
    contour: dict[str, Any] | None,
    *,
    selected_id: int,
    kpt_thr: float,
    draw_all_keypoints: bool,
    keypoint_names: list[str] | None = None,
) -> np.ndarray:
    """Match SAM3+Sapiens2 look: contour fill, left/right skeleton, then outline/ID."""
    style = _load_sapiens_overlay_style()
    names = keypoint_names
    if names is None and style is not None:
        names = style.get("keypoint_names")
    out = _draw_sam_contour_fill(image, contour, selected_id=selected_id)
    out = _draw_sapiens_skeleton(
        out,
        instance,
        kpt_thr=kpt_thr,
        draw_all_keypoints=draw_all_keypoints,
        style=style,
        keypoint_names=names,
    )
    return _draw_sam_contour_outline_and_id(out, instance, contour, selected_id=selected_id)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, codec in ((".mp4", "mp4v"), (".avi", "XVID")):
        candidate = path.with_suffix(suffix)
        writer = cv2.VideoWriter(
            str(candidate),
            cv2.VideoWriter_fourcc(*codec),  # ty: ignore[unresolved-attribute]
            fps,
            size,
        )
        if writer.isOpened():
            return writer, candidate
        writer.release()
    raise OSError(f"Could not open a video writer for {path}")


def render_selected_video(
    video_path: Path,
    output_path: Path,
    records: dict[int, dict[str, Any]],
    contours: dict[int, dict[str, Any]],
    *,
    selected_id: int,
    kpt_thr: float = DEFAULT_KPT_THR,
    draw_all_keypoints: bool = True,
    keypoint_names: list[str] | None = None,
) -> tuple[Path, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Warm the style cache once before the frame loop (logs once).
    _load_sapiens_overlay_style()
    writer, actual_path = _open_writer(output_path, fps, (width, height))
    frame_count = 0
    drawn_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            instance = records.get(frame_count)
            if instance is not None:
                frame = _draw_instance(
                    frame,
                    instance,
                    contours.get(frame_count),
                    selected_id=selected_id,
                    kpt_thr=kpt_thr,
                    draw_all_keypoints=draw_all_keypoints,
                    keypoint_names=keypoint_names,
                )
                drawn_count += 1
            writer.write(frame)
            frame_count += 1
    finally:
        cap.release()
        writer.release()
    return actual_path, frame_count, drawn_count


def _filter_rows(path: Path, output: Path, id_columns: tuple[str, ...], selected_id: int) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return False
        rows = list(reader)
    column = next((name for name in id_columns if name in reader.fieldnames), None)
    if column is None:
        return False
    selected = [row for row in rows if _safe_int(row.get(column)) == selected_id]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    return True


def _write_filtered_json(
    path: Path, output: Path, selected_id: int, *, contours: bool = False
) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if contours:
        payload["object_ids"] = [selected_id]
        for frame in payload.get("frames", []):
            frame["objects"] = [
                o for o in frame.get("objects", []) if _safe_int(o.get("obj_id")) == selected_id
            ]
    else:
        for frame in payload.get("frames", []):
            frame["instances"] = [
                i
                for i in frame.get("instances", [])
                if _safe_int(i.get("stable_id", i.get("sam_obj_id"))) == selected_id
            ]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def _id_slot(run_dir: Path, selected_id: int) -> int | None:
    path = run_dir / "sapiens_id_map.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if _safe_int(row.get("stable_id")) == selected_id:
                return _safe_int(row.get("pN"))
    return None


def _write_wide_selected(path: Path, output: Path, slot: int | None) -> bool:
    if not path.exists() or slot is None:
        return False
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        keep = [field for field in fields if field == "frame" or field.startswith(f"p{slot}_")]
        if not keep:
            return False
        rows = [{field: row.get(field, "") for field in keep} for row in reader]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keep)
        writer.writeheader()
        writer.writerows(rows)
    return True


def write_selected_artifacts(
    run_dir: Path, output_dir: Path, selected_id: int, payload: dict[str, Any]
) -> list[str]:
    """Write filtered artifacts and preserve all source artifacts for provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_artifacts"
    source_dir.mkdir(exist_ok=True)
    written: list[str] = []
    stem = Path(str(payload.get("video", "video.mp4"))).stem
    mapping = (
        (run_dir / "sam3" / "sam_tracks.csv", output_dir / "sam_tracks.csv", ("obj_id",)),
        (run_dir / "sam3" / "sam_bbox_tracks.csv", output_dir / "sam_bbox_tracks.csv", ("obj_id",)),
        (
            run_dir / "sapiens_tracks.csv",
            output_dir / "sapiens_tracks.csv",
            ("stable_id", "person_id"),
        ),
        (
            run_dir / "sam3sapiens2_id_audit.csv",
            output_dir / "sam3sapiens2_id_audit.csv",
            ("stable_id", "sam_obj_id"),
        ),
        (
            run_dir / f"{stem}_sam3sapiens2_vaila.csv",
            output_dir / f"{stem}_sam3sapiens2_vaila.csv",
            ("stable_id", "person_id", "sam_obj_id"),
        ),
    )
    for source, target, columns in mapping:
        if _filter_rows(source, target, columns, selected_id):
            written.append(str(target.name))
    if _write_filtered_json(
        _prediction_path(run_dir),
        output_dir / f"{stem}_sam3sapiens2_predictions.json",
        selected_id,
    ):
        written.append(f"{stem}_sam3sapiens2_predictions.json")
    if _write_filtered_json(
        run_dir / "sam3" / "sam_contours.json",
        output_dir / "sam_contours.json",
        selected_id,
        contours=True,
    ):
        written.append("sam_contours.json")
    slot = _id_slot(run_dir, selected_id)
    if slot is not None:
        with (output_dir / "sapiens_id_map.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["pN", "stable_id", "selected"])
            writer.writerow([slot, selected_id, True])
        written.append("sapiens_id_map.csv")
        for source_name in (
            "sapiens_points.csv",
            "sapiens_vaila_center.csv",
            "sapiens_vaila_bottom.csv",
            "sapiens_vaila_top.csv",
            "sapiens_vaila_left.csv",
            "sapiens_vaila_right.csv",
        ):
            if _write_wide_selected(run_dir / source_name, output_dir / source_name, slot):
                written.append(source_name)
    for source in run_dir.glob(f"*id_{selected_id:02d}*sapiens_pose.csv"):
        shutil.copy2(source, output_dir / source.name)
        written.append(source.name)

    # Keep every original non-video artifact in a namespaced folder. The root
    # remains ID-specific, while provenance is never silently discarded.
    for source in run_dir.rglob("*"):
        if not source.is_file() or source == output_dir or output_dir in source.parents:
            continue
        if source.name.endswith("_sam3sapiens2_overlay.mp4"):
            continue
        relative = source.relative_to(run_dir)
        target = source_dir / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return written


def visualize_selected_id(
    run_dir: Path,
    video_path: Path,
    selected_id: int,
    output_dir: Path,
    *,
    kpt_thr: float = DEFAULT_KPT_THR,
    draw_all_keypoints: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    run_dir = resolve_run_dir(run_dir, video_path)
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise FileNotFoundError(f"Video not found or unsupported: {video_path}")
    payload = load_predictions(run_dir)
    available = discover_ids(run_dir, payload)
    if selected_id not in available:
        raise ValueError(f"ID {selected_id} is unavailable; choose one of {available}")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir} (use --overwrite)")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _records_by_frame(payload, selected_id)
    contours = _contours_by_frame(run_dir, selected_id)
    keypoint_names = payload.get("keypoint_names")
    if isinstance(keypoint_names, list):
        keypoint_names = [str(name) for name in keypoint_names]
    else:
        keypoint_names = None
    overlay, frames, drawn = render_selected_video(
        video_path,
        output_dir / f"{video_path.stem}_sam3sapiens2_id_{selected_id:02d}_overlay.mp4",
        records,
        contours,
        selected_id=selected_id,
        kpt_thr=kpt_thr,
        draw_all_keypoints=draw_all_keypoints,
        keypoint_names=keypoint_names,
    )
    written = write_selected_artifacts(run_dir, output_dir, selected_id, payload)
    manifest = {
        "schema": "vaila_sam3sapiens2_selected_id_v1",
        "source_run": str(run_dir),
        "source_video": str(video_path),
        "selected_id": selected_id,
        "available_ids": available,
        "fps": payload.get("fps"),
        "image_size": payload.get("image_size"),
        "n_video_frames": frames,
        "n_frames_with_selected_pose": drawn,
        "kpt_threshold": kpt_thr,
        "overlay": str(overlay),
        "artifacts": written,
        "created_at": dt.datetime.now().astimezone().isoformat(),
    }
    (output_dir / "sam3sapiens2_selected_id_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README_sam3sapiens2_selected_id.txt").write_text(
        "vailá SAM3+Sapiens2 selected-ID visualization\n"
        f"selected_id={selected_id}\nsource_run={run_dir}\nsource_video={video_path}\n"
        "identity_authority=SAM3 obj_id / stable_id\n"
        "coordinate_units=full-frame pixels; frame_index=zero-based\n"
        "The root contains filtered artifacts. source_artifacts/ preserves the original run.\n"
        "Overlay style: Sapiens2 left/right skeleton colors + SAM3 contour fill/outline.\n",
        encoding="utf-8",
    )
    _log(f"ID {selected_id}: rendered {drawn}/{frames} frames -> {output_dir}")
    return manifest


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
    dialog.title("SAM3+Sapiens2 — Visualize one ID")
    dialog.resizable(False, False)
    vars_: dict[str, tk.StringVar] = {
        "run": tk.StringVar(),
        "video": tk.StringVar(),
        "output": tk.StringVar(),
        "id": tk.StringVar(),
        "thr": tk.StringVar(value=str(DEFAULT_KPT_THR)),
        "status": tk.StringVar(value="Choose a processed_sam3sapiens2_* directory first."),
    }
    frame = ttk.Frame(dialog, padding=14)
    frame.grid(sticky="nsew")
    ttk.Label(
        frame, text="SAM3+Sapiens2 — selected-ID visualization", font=("TkDefaultFont", 11, "bold")
    ).grid(column=0, row=0, columnspan=3, sticky="w", pady=(0, 10))

    def browse_run() -> None:
        chosen = filedialog.askdirectory(parent=dialog, title="Processed SAM3+Sapiens2 directory")
        if not chosen:
            return
        vars_["run"].set(chosen)
        try:
            resolved = resolve_run_dir(
                Path(chosen), Path(vars_["video"].get()) if vars_["video"].get() else None
            )
            payload = load_predictions(resolved)
            ids = discover_ids(resolved, payload)
            combo["values"] = [str(value) for value in ids]
            if ids:
                vars_["id"].set(str(ids[0]))
            source_video = discover_source_video(resolved, payload)
            if source_video is not None:
                vars_["video"].set(str(source_video))
                vars_["status"].set(f"Available IDs: {ids}. Source video found automatically.")
            else:
                vars_["status"].set(f"Available IDs: {ids}. Choose the matching source video.")
        except Exception as exc:
            combo["values"] = []
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

    def browse_output() -> None:
        chosen = filedialog.askdirectory(parent=dialog, title="Output parent directory")
        if chosen:
            vars_["output"].set(chosen)

    for row, (label, key, command) in enumerate(
        (
            ("Run directory", "run", browse_run),
            ("Video", "video", browse_video),
            ("Output parent", "output", browse_output),
        ),
        start=1,
    ):
        ttk.Label(frame, text=label).grid(column=0, row=row, sticky="w", pady=3)
        ttk.Entry(frame, textvariable=vars_[key], width=66).grid(
            column=1, row=row, sticky="ew", padx=6
        )
        ttk.Button(frame, text="Browse…", command=command).grid(column=2, row=row)
    ttk.Label(frame, text="Selected ID").grid(column=0, row=4, sticky="w", pady=3)
    combo = ttk.Combobox(frame, textvariable=vars_["id"], state="readonly", width=14)
    combo.grid(column=1, row=4, sticky="w", padx=6)
    ttk.Label(frame, text="Keypoint threshold").grid(column=0, row=5, sticky="w", pady=3)
    ttk.Entry(frame, textvariable=vars_["thr"], width=14).grid(column=1, row=5, sticky="w", padx=6)
    ttk.Label(frame, textvariable=vars_["status"], foreground="#7a4f00", wraplength=560).grid(
        column=0, row=6, columnspan=3, sticky="w", pady=(8, 4)
    )

    def run() -> None:
        try:
            run_dir = Path(vars_["run"].get()).expanduser()
            video = Path(vars_["video"].get()).expanduser()
            selected = int(vars_["id"].get())
            output_text = vars_["output"].get().strip()
            output_parent = Path(output_text).expanduser() if output_text else run_dir
            output = _unique_gui_output_dir(output_parent, video, selected)
            threshold = float(vars_["thr"].get())
        except Exception as exc:
            messagebox.showerror(
                "SAM3+Sapiens2 visualization", f"Invalid settings: {exc}", parent=dialog
            )
            return
        _log(
            "GUI equivalent CLI: "
            + " ".join(
                [
                    "uv run python -u vaila/sam3sapiens2_visualize.py",
                    "--sam-results",
                    str(run_dir),
                    "--video",
                    str(video),
                    "--id",
                    str(selected),
                    "--output",
                    str(output),
                ]
            )
        )
        vars_["status"].set("Rendering selected ID; the window remains responsive…")

        def worker() -> None:
            try:
                result = visualize_selected_id(run_dir, video, selected, output, kpt_thr=threshold)
                dialog.after(
                    0,
                    lambda: (
                        vars_["status"].set(f"Done: {result['overlay']}"),
                        messagebox.showinfo(
                            "SAM3+Sapiens2 visualization",
                            f"Finished ID {selected}.\n\n{output}",
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
                            "SAM3+Sapiens2 visualization", error_text, parent=dialog
                        ),
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    ttk.Button(frame, text="Run selected ID", command=run).grid(
        column=1, row=7, sticky="w", padx=6, pady=(10, 0)
    )
    ttk.Button(frame, text="Cancel", command=dialog.destroy).grid(column=2, row=7, pady=(10, 0))
    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    _show_dialog_in_front(dialog)
    if owns_root:
        root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one SAM3+Sapiens2 ID from an existing run."
    )
    parser.add_argument(
        "--sam-results",
        "--input-dir",
        required=False,
        type=Path,
        help="Per-video or processed_sam3sapiens2_* directory.",
    )
    parser.add_argument("--video", "-i", type=Path, help="Original source video.")
    parser.add_argument(
        "--id",
        dest="selected_id",
        type=int,
        help="SAM/stable ID to visualize. If omitted in CLI mode, prompt interactively.",
    )
    parser.add_argument("--output", "-o", type=Path, help="New output directory for this ID.")
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=DEFAULT_KPT_THR,
        help="Keypoint confidence threshold (default: 0.30).",
    )
    parser.add_argument(
        "--no-all-keypoints",
        action="store_true",
        help="Draw only the 21 main body points instead of all visible Sapiens2 points.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow a non-empty output directory."
    )
    parser.add_argument(
        "--list-ids", action="store_true", help="Print available IDs and exit without rendering."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without writing a video.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.sam_results, args.video, args.selected_id is not None, args.output)):
        run_visualizer_gui()
        return 0
    if args.sam_results is None or args.video is None:
        parser.error("--sam-results/--input-dir and --video/-i must be supplied together")
    run_dir = resolve_run_dir(args.sam_results, args.video)
    payload = load_predictions(run_dir)
    ids = discover_ids(run_dir, payload)
    print(f">> Available SAM IDs: {ids}")
    if args.list_ids:
        return 0
    selected_id = args.selected_id
    if selected_id is None:
        selected_id = prompt_selected_id(ids)
    elif selected_id not in ids:
        parser.error(f"ID {selected_id} is unavailable; choose one of {ids}")
    if args.output is None:
        parser.error("--output/-o is required for rendering")
    if args.dry_run:
        validate_source_video(args.video.expanduser().resolve(), payload)
        print(
            f">> Dry-run OK: video={args.video} run_dir={run_dir} id={selected_id} output={args.output}"
        )
        return 0
    visualize_selected_id(
        run_dir,
        args.video,
        selected_id,
        args.output,
        kpt_thr=args.kpt_thr,
        draw_all_keypoints=not args.no_all_keypoints,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
