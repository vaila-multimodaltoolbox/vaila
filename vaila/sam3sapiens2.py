"""
Project: vailá
Script: sam3sapiens2.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 30 July 2026
Update Date: 31 July 2026
Version: 0.3.86

Description:
    SAM3-guided Sapiens2 pose pipeline. SAM3 runs first and remains the
    authority for person detection, silhouettes, and persistent object IDs.
    Sapiens2 then runs only on padded SAM boxes using contour-focused images;
    DETR is not loaded. Outputs retain the SAM obj_id as stable_id/person_id.

Usage:
    uv run python -u vaila/sam3sapiens2.py \
        -i /path/to/video_or_folder -o /path/to/output -t person --model 1b

    # Reuse an existing processed_sam_* directory (no SAM rerun):
    uv run python -u vaila/sam3sapiens2.py \
        -i /path/to/videos -o /path/to/output \
        --sam-results /path/to/processed_sam_YYYYMMDD_HHMMSS

    # GUI: omit arguments, or Frame B -> YOLO + FB -> SAM3+Sapiens2

License:
    This program is licensed under the GNU Affero General Public License v3.0.
    Sapiens2 and SAM3 model weights keep their respective Meta licenses.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import os
import shlex
import subprocess
import sys
import tkinter as tk
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

try:
    from .vaila_sapiens import (
        DEFAULT_KPT_THR,
        DEFAULT_MODEL_KEY,
        PoseInferenceSession,
        _default_pose_batch_size,
        _expand_pose_timeline,
        _open_sam3_video_writer,
        _release_sapiens_gpu_memory,
        _resolve_sapiens_keypoint_names,
        flatten_instances_to_csv_rows,
        resolve_model_spec,
        write_sapiens_biomechanics_csvs,
        write_sapiens_getpixelvideo_pose_csvs,
        write_vaila_pose_csv,
    )
except ImportError:
    from vaila_sapiens import (  # ty: ignore[unresolved-import]
        DEFAULT_KPT_THR,
        DEFAULT_MODEL_KEY,
        PoseInferenceSession,
        _default_pose_batch_size,
        _expand_pose_timeline,
        _open_sam3_video_writer,
        _release_sapiens_gpu_memory,
        _resolve_sapiens_keypoint_names,
        flatten_instances_to_csv_rows,
        resolve_model_spec,
        write_sapiens_biomechanics_csvs,
        write_sapiens_getpixelvideo_pose_csvs,
        write_vaila_pose_csv,
    )

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_BBOX_PADDING = 0.12
DEFAULT_CONTOUR_MARGIN_PX = 8
DEFAULT_OUTSIDE_CONTOUR_FACTOR = 0.25
DEFAULT_MIN_SAM_SCORE = 0.0
DEFAULT_MIN_SAM_AREA = 64
DEFAULT_MAX_PERSONS = 32


@dataclass(frozen=True)
class SamGuidance:
    """SAM bbox/contour records indexed by original zero-based video frame."""

    sam_dir: Path
    tracks_by_frame: dict[int, list[dict[str, Any]]]
    contours_by_frame: dict[int, dict[int, dict[str, Any]]]
    width: int | None
    height: int | None
    fps: float | None
    n_frames: int | None
    contour_path: Path | None


@dataclass(frozen=True)
class CombinedGuiSettings:
    input_path: Path
    output_parent: Path
    sam_results: Path | None
    prompt: str
    model: str
    stride: int
    device: int
    kpt_thr: float
    pose_batch_size: int | None
    bbox_padding: float
    contour_margin: int
    max_persons: int
    flip_test: bool
    save_overlay: bool


def _log(message: str) -> None:
    print(f">> vaila/sam3sapiens2: {message}", flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _help_path() -> Path:
    return Path(__file__).resolve().parent / "help" / "sam3sapiens2.html"


def _prepare_gui_root(root: tk.Tk | tk.Toplevel, *, owns_root: bool) -> None:
    """Make a standalone Tk root manageable by Linux/Windows window managers."""
    if not owns_root or sys.platform == "darwin":
        return
    try:
        root.deiconify()
        root.geometry("1x1+100+100")
        root.update_idletasks()
    except tk.TclError:
        pass


def _show_dialog_in_front(dialog: tk.Toplevel) -> None:
    """Center, map, and briefly raise a subprocess-owned settings dialog."""
    try:
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

        def _restore_normal_stacking() -> None:
            try:
                dialog.attributes("-topmost", False)
                dialog.lift()
                dialog.focus_force()
            except tk.TclError:
                pass

        dialog.after(350, _restore_normal_stacking)
    except tk.TclError:
        pass


def _is_derived_video(path: Path) -> bool:
    name = path.name.lower()
    return any(tag in name for tag in ("_sam_overlay", "_sapiens_overlay", "_sam3sapiens2_overlay"))


def _find_videos(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in VIDEO_EXTENSIONS else []
    if not path.is_dir():
        return []
    return sorted(
        (
            p
            for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and not _is_derived_video(p)
        ),
        key=lambda p: p.name.lower(),
    )


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _open_json_text(path: Path) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


def _find_contour_path(sam_dir: Path) -> Path | None:
    for name in (
        "sam_contours.json",
        "sam_contours.json.gz",
        "sam_contours.jsonl",
        "sam_contours.jsonl.gz",
    ):
        candidate = sam_dir / name
        if candidate.is_file():
            return candidate
    return None


def _load_contours(path: Path | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if path is None:
        return {}, []
    text = _open_json_text(path)
    if ".jsonl" in path.name.lower():
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        if not records:
            return {}, []
        return dict(records[0]), [dict(record) for record in records[1:]]
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid SAM contours payload: {path}")
    frames = payload.get("frames") or []
    return payload, [dict(record) for record in frames if isinstance(record, dict)]


def load_sam_guidance(sam_dir: Path | str) -> SamGuidance:
    """Load and validate SAM3 bbox tracks plus optional polygon contours."""
    root = Path(sam_dir).expanduser().resolve()
    tracks_path = root / "sam_tracks.csv"
    if not tracks_path.is_file():
        alias = root / "sam_bbox_tracks.csv"
        if alias.is_file():
            tracks_path = alias
        else:
            raise FileNotFoundError(
                f"SAM tracks not found under {root}; expected sam_tracks.csv or sam_bbox_tracks.csv"
            )

    tracks_by_frame: dict[int, list[dict[str, Any]]] = {}
    with tracks_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"frame", "obj_id", "x_px", "y_px", "w_px", "h_px"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{tracks_path} is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                frame_idx = int(float(row["frame"]))
                obj_id = int(float(row["obj_id"]))
                x = float(row["x_px"])
                y = float(row["y_px"])
                w = float(row["w_px"])
                h = float(row["h_px"])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite([x, y, w, h]).all() or w <= 1.0 or h <= 1.0:
                continue
            record: dict[str, Any] = {
                "frame": frame_idx,
                "obj_id": obj_id,
                "x_px": x,
                "y_px": y,
                "w_px": w,
                "h_px": h,
                "score": _safe_float(row.get("score"), 1.0),
                "area_px": _safe_int(row.get("area_px"), int(round(w * h))),
                "cx_px": _safe_float(row.get("cx_px"), x + 0.5 * w),
                "cy_px": _safe_float(row.get("cy_px"), y + 0.5 * h),
            }
            tracks_by_frame.setdefault(frame_idx, []).append(record)

    if not tracks_by_frame:
        raise ValueError(f"No valid SAM bbox rows found in {tracks_path}")
    for rows in tracks_by_frame.values():
        rows.sort(key=lambda row: int(row["obj_id"]))

    contour_path = _find_contour_path(root)
    header, frame_records = _load_contours(contour_path)
    contours_by_frame: dict[int, dict[int, dict[str, Any]]] = {}
    for frame_record in frame_records:
        try:
            frame_idx = int(frame_record.get("frame", -1))
        except (TypeError, ValueError):
            continue
        objects: dict[int, dict[str, Any]] = {}
        for obj in frame_record.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            try:
                objects[int(obj["obj_id"])] = obj
            except (KeyError, TypeError, ValueError):
                continue
        contours_by_frame[frame_idx] = objects

    _log(
        f"SAM guidance loaded: {len(tracks_by_frame)} frames, "
        f"{sum(len(rows) for rows in tracks_by_frame.values())} boxes, "
        f"contours={'yes' if contour_path else 'no'} ({root})"
    )
    return SamGuidance(
        sam_dir=root,
        tracks_by_frame=tracks_by_frame,
        contours_by_frame=contours_by_frame,
        width=_optional_int(header.get("width")),
        height=_optional_int(header.get("height")),
        fps=_optional_float(header.get("fps")),
        n_frames=_optional_int(header.get("n_frames")),
        contour_path=contour_path,
    )


def resolve_sam_results_dir(
    sam_results: Path | str,
    video_path: Path | str,
    *,
    single_video: bool = False,
) -> Path:
    """Resolve a direct SAM run dir or a processed_sam_* batch parent."""
    root = Path(sam_results).expanduser().resolve()
    video = Path(video_path)
    direct_candidates = [root, root / video.stem, root / video.stem / "sam3", root / "sam3"]
    for candidate in direct_candidates:
        if (candidate / "sam_tracks.csv").is_file() or (
            candidate / "sam_bbox_tracks.csv"
        ).is_file():
            if candidate == root and not single_video and root.name != video.stem:
                continue
            return candidate
    matches = sorted(root.glob(f"*/{video.stem}/sam_tracks.csv"))
    if len(matches) == 1:
        return matches[0].parent
    raise FileNotFoundError(
        f"Could not match SAM3 outputs for {video.name} under {root}. "
        f"Expected {root / video.stem / 'sam_tracks.csv'}"
    )


def _object_polygons(obj: dict[str, Any] | None) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    if not obj:
        return polygons
    for raw in obj.get("polygons") or []:
        arr = np.asarray(raw, dtype=np.int32).reshape(-1, 2)
        if len(arr) >= 3 and abs(float(cv2.contourArea(arr))) >= 2.0:
            polygons.append(arr)
    return polygons


def _track_xyxy(track: dict[str, Any]) -> tuple[float, float, float, float]:
    x = float(track["x_px"])
    y = float(track["y_px"])
    return x, y, x + float(track["w_px"]), y + float(track["h_px"])


def _pose_bbox_from_sam(
    track: dict[str, Any],
    contour_obj: dict[str, Any] | None,
    *,
    frame_width: int,
    frame_height: int,
    padding_fraction: float,
) -> np.ndarray:
    """Create a tight contour box with context padding, clamped to the video."""
    x1, y1, x2, y2 = _track_xyxy(track)
    polygons = _object_polygons(contour_obj)
    if polygons:
        points = np.concatenate(polygons, axis=0)
        cx1, cy1 = np.min(points, axis=0).astype(float)
        cx2, cy2 = np.max(points, axis=0).astype(float)
        if cx2 > cx1 + 2 and cy2 > cy1 + 2:
            x1, y1, x2, y2 = cx1, cy1, cx2 + 1.0, cy2 + 1.0
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    pad = max(0.0, float(padding_fraction))
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    return np.asarray(
        [
            np.clip(x1, 0.0, max(0.0, frame_width - 1.0)),
            np.clip(y1, 0.0, max(0.0, frame_height - 1.0)),
            np.clip(x2, 1.0, float(frame_width)),
            np.clip(y2, 1.0, float(frame_height)),
        ],
        dtype=np.float32,
    )


def _contour_mask(
    frame_shape: tuple[int, ...],
    track: dict[str, Any],
    contour_obj: dict[str, Any] | None,
    *,
    margin_px: int,
) -> np.ndarray:
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    polygons = _object_polygons(contour_obj)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
    else:
        x1, y1, x2, y2 = _track_xyxy(track)
        cv2.rectangle(
            mask,
            (max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))),
            (
                min(mask.shape[1] - 1, int(np.ceil(x2))),
                min(mask.shape[0] - 1, int(np.ceil(y2))),
            ),
            255,
            -1,
        )
    radius = max(0, int(margin_px))
    if radius > 0:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel)
    return mask


def _contour_focused_image(
    frame: np.ndarray,
    pose_bbox: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blur background inside the pose crop while preserving SAM foreground."""
    x1, y1, x2, y2 = [int(round(float(v))) for v in pose_bbox]
    x1 = max(0, min(frame.shape[1] - 1, x1))
    x2 = max(x1 + 1, min(frame.shape[1], x2))
    y1 = max(0, min(frame.shape[0] - 1, y1))
    y2 = max(y1 + 1, min(frame.shape[0], y2))
    focused = frame.copy()
    patch = frame[y1:y2, x1:x2]
    local_mask = mask[y1:y2, x1:x2]
    if patch.size == 0 or not np.any(local_mask):
        return focused
    sigma = max(3.0, min(patch.shape[:2]) / 18.0)
    background = cv2.GaussianBlur(patch, (0, 0), sigmaX=sigma, sigmaY=sigma)
    keep = local_mask.astype(bool)
    background[keep] = patch[keep]
    focused[y1:y2, x1:x2] = background
    return focused


def _attenuate_scores_outside_contour(
    keypoints: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    *,
    outside_factor: float,
) -> tuple[np.ndarray, list[bool]]:
    adjusted = np.asarray(scores, dtype=np.float32).reshape(-1).copy()
    inside_flags: list[bool] = []
    height, width = mask.shape[:2]
    factor = float(np.clip(outside_factor, 0.0, 1.0))
    for idx, point in enumerate(np.asarray(keypoints).reshape(-1, 2)):
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        inside = 0 <= x < width and 0 <= y < height and bool(mask[y, x])
        inside_flags.append(inside)
        if idx < len(adjusted) and not inside:
            adjusted[idx] *= factor
    return adjusted, inside_flags


def _guidance_for_frame(
    guidance: SamGuidance,
    frame_idx: int,
    *,
    min_score: float,
    min_area: int,
    max_persons: int,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    rows = [
        row
        for row in guidance.tracks_by_frame.get(frame_idx, [])
        if float(row.get("score", 1.0)) >= float(min_score)
        and int(row.get("area_px", 0)) >= int(min_area)
    ]
    if len(rows) > max_persons:
        rows = sorted(
            rows,
            key=lambda row: (
                -float(row.get("score", 1.0)) * max(1.0, float(row.get("area_px", 1))),
                int(row["obj_id"]),
            ),
        )[:max_persons]
    rows.sort(key=lambda row: int(row["obj_id"]))
    contour_map = guidance.contours_by_frame.get(frame_idx, {})
    return [(row, contour_map.get(int(row["obj_id"]))) for row in rows]


def _color_for_id(obj_id: int) -> tuple[int, int, int]:
    hue = (int(obj_id) * 47) % 180
    hsv = np.asarray([[[hue, 220, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_sam_guidance(
    image: np.ndarray,
    frame_guidance: list[tuple[dict[str, Any], dict[str, Any] | None]],
    *,
    draw_ids: bool,
) -> np.ndarray:
    out = image.copy()
    for track, contour_obj in frame_guidance:
        obj_id = int(track["obj_id"])
        color = _color_for_id(obj_id)
        polygons = _object_polygons(contour_obj)
        if polygons:
            cv2.polylines(out, polygons, True, color, 2, cv2.LINE_AA)
        x1, y1, x2, y2 = _track_xyxy(track)
        cv2.rectangle(out, (round(x1), round(y1)), (round(x2), round(y2)), color, 2)
        if draw_ids:
            label = f"SAM #{obj_id} {float(track.get('score', 1.0)):.2f}"
            tx = max(0, round(x1))
            ty = max(18, round(y1) - 5)
            cv2.putText(
                out,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
    return out


def _transform_pose_to_current_sam_box(
    instance: dict[str, Any],
    track: dict[str, Any],
    contour_obj: dict[str, Any] | None,
    *,
    frame_width: int,
    frame_height: int,
    bbox_padding: float,
) -> dict[str, Any]:
    current_bbox = np.asarray(_track_xyxy(track), dtype=np.float32)
    source_bbox = np.asarray(instance.get("sam_bbox_xyxy") or instance["bbox"], dtype=np.float32)
    sw = max(1.0, float(source_bbox[2] - source_bbox[0]))
    sh = max(1.0, float(source_bbox[3] - source_bbox[1]))
    tw = max(1.0, float(current_bbox[2] - current_bbox[0]))
    th = max(1.0, float(current_bbox[3] - current_bbox[1]))
    kpts = np.asarray(instance.get("keypoints") or [], dtype=np.float32).reshape(-1, 2)
    if len(kpts):
        kpts[:, 0] = current_bbox[0] + (kpts[:, 0] - source_bbox[0]) * tw / sw
        kpts[:, 1] = current_bbox[1] + (kpts[:, 1] - source_bbox[1]) * th / sh
    transformed = dict(instance)
    transformed.update(
        {
            "bbox": current_bbox.tolist(),
            "sam_bbox_xyxy": current_bbox.tolist(),
            "pose_bbox_xyxy": _pose_bbox_from_sam(
                track,
                contour_obj,
                frame_width=frame_width,
                frame_height=frame_height,
                padding_fraction=bbox_padding,
            ).tolist(),
            "keypoints": kpts.tolist(),
            "sam_score": float(track.get("score", 1.0)),
            "sam_area_px": int(track.get("area_px", 0)),
        }
    )
    return transformed


def _expand_pose_with_sam_guidance(
    inferred: dict[int, list[dict[str, Any]]],
    guidance: SamGuidance,
    *,
    n_frames: int,
    frame_width: int,
    frame_height: int,
    bbox_padding: float,
    min_score: float,
    min_area: int,
    max_persons: int,
) -> dict[int, list[dict[str, Any]]]:
    """Fill stride gaps while updating every instance to that frame's SAM box."""
    if not inferred:
        return {frame_idx: [] for frame_idx in range(n_frames)}
    expanded_nearest = _expand_pose_timeline(inferred, n_frames)
    timeline: dict[int, list[dict[str, Any]]] = {}
    for frame_idx in range(n_frames):
        source_instances = {
            int(inst["stable_id"]): inst for inst in expanded_nearest.get(frame_idx, [])
        }
        current: list[dict[str, Any]] = []
        for track, contour_obj in _guidance_for_frame(
            guidance,
            frame_idx,
            min_score=min_score,
            min_area=min_area,
            max_persons=max_persons,
        ):
            obj_id = int(track["obj_id"])
            source = source_instances.get(obj_id)
            if source is None:
                continue
            current.append(
                _transform_pose_to_current_sam_box(
                    source,
                    track,
                    contour_obj,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    bbox_padding=bbox_padding,
                )
            )
        timeline[frame_idx] = current
    return timeline


def _write_combined_overlay(
    video_path: Path,
    output_path: Path,
    timeline: dict[int, list[dict[str, Any]]],
    guidance: SamGuidance,
    session: PoseInferenceSession,
    *,
    min_score: float,
    min_area: int,
    max_persons: int,
    draw_ids: bool,
) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not reopen video for overlay: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer, actual_path = _open_sam3_video_writer(
        output_path,
        fps,
        (width, height),
        purpose="SAM3+Sapiens2 overlay",
    )
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            instances = timeline.get(frame_idx, [])
            keypoints = [
                np.asarray(inst.get("keypoints") or [], dtype=np.float32) for inst in instances
            ]
            scores = [
                np.asarray(inst.get("keypoint_scores") or [], dtype=np.float32)
                for inst in instances
            ]
            rendered = session.render_overlay(
                frame,
                keypoints,
                scores,
                instances=None,
                draw_id=False,
            )
            rendered = _draw_sam_guidance(
                rendered,
                _guidance_for_frame(
                    guidance,
                    frame_idx,
                    min_score=min_score,
                    min_area=min_area,
                    max_persons=max_persons,
                ),
                draw_ids=draw_ids,
            )
            writer.write(rendered)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()
    return actual_path


def _write_identity_audit(
    path: Path,
    timeline: dict[int, list[dict[str, Any]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "frame",
                "sam_obj_id",
                "stable_id",
                "sam_score",
                "sam_area_px",
                "mean_kpt_score",
                "inside_contour_ratio",
            ]
        )
        for frame_idx in sorted(timeline):
            for inst in timeline[frame_idx]:
                scores = np.asarray(inst.get("keypoint_scores") or [], dtype=float)
                inside = inst.get("keypoints_inside_sam_contour") or []
                writer.writerow(
                    [
                        frame_idx,
                        int(inst["sam_obj_id"]),
                        int(inst["stable_id"]),
                        float(inst.get("sam_score", 1.0)),
                        int(inst.get("sam_area_px", 0)),
                        float(np.mean(scores)) if len(scores) else "",
                        float(np.mean(inside)) if inside else "",
                    ]
                )


def _write_readme(
    output_dir: Path,
    *,
    video_path: Path,
    sam_dir: Path,
    model: str,
    stride: int,
    contour_focus: bool,
    bbox_padding: float,
    contour_margin: int,
) -> None:
    text = f"""vailá SAM3+Sapiens2 run
video={video_path}
sam_results={sam_dir}
model={model}
stride={stride}
identity_authority=SAM3 obj_id
person_detector=SAM3 (DETR disabled)
contour_focus={contour_focus}
bbox_padding_fraction={bbox_padding}
contour_margin_px={contour_margin}

Pipeline
--------
1. SAM3 segments/tracks people and defines bbox, silhouette, score and obj_id.
2. Sapiens2 receives only those SAM boxes; DETR is never loaded.
3. Each Sapiens crop is focused with the corresponding SAM silhouette.
4. Keypoints outside the dilated silhouette have confidence attenuated.
5. stable_id/person_id is exactly SAM obj_id; no second Re-ID can swap it.

Main outputs
------------
<video>_sam3sapiens2_overlay.mp4  SAM contour+bbox+ID and Sapiens2 skeleton.
<video>_sam3sapiens2_predictions.json  Full provenance and 308-keypoint instances.
<video>_sam3sapiens2_vaila.csv  Long frame/person/keypoint table.
sam3sapiens2_id_audit.csv  Per-frame proof that sam_obj_id == stable_id.
<video>_markers.csv and sapiens_vaila_*.csv  REC2D/REC3D/getpixelvideo outputs.
sapiens_points.csv, sapiens_id_map.csv, sapiens_bbox_tracks.csv  Stable SAM-ID tables.
<video>_id_NN_sapiens_pose.csv  Wide 308-keypoint file per SAM identity.
README_sam3sapiens2.txt  This file.
FAILED_sam3sapiens2.txt  Exists only when this combined stage fails.

The original SAM artifacts remain under the path recorded above (or in ./sam3
when SAM3 was run by this pipeline). Coordinate units are full-frame pixels and
frames are zero-based.
"""
    (output_dir / "README_sam3sapiens2.txt").write_text(text, encoding="utf-8")


def run_sapiens_from_sam(
    video_path: Path,
    output_dir: Path,
    sam_dir: Path,
    *,
    model: str = DEFAULT_MODEL_KEY,
    stride: int = 1,
    device: int = 0,
    kpt_thr: float = DEFAULT_KPT_THR,
    pose_batch_size: int | None = None,
    bbox_padding: float = DEFAULT_BBOX_PADDING,
    contour_margin: int = DEFAULT_CONTOUR_MARGIN_PX,
    outside_contour_factor: float = DEFAULT_OUTSIDE_CONTOUR_FACTOR,
    min_sam_score: float = DEFAULT_MIN_SAM_SCORE,
    min_sam_area: int = DEFAULT_MIN_SAM_AREA,
    max_persons: int = DEFAULT_MAX_PERSONS,
    contour_focus: bool = True,
    flip_test: bool = False,
    save_overlay: bool = True,
    draw_ids: bool = True,
) -> dict[str, Any]:
    """Run DETR-free Sapiens2 pose using SAM bboxes, contours, and IDs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    guidance = load_sam_guidance(sam_dir)

    cap_probe = cv2.VideoCapture(str(video_path))
    if not cap_probe.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap_probe.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap_probe.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid video dimensions: {width}x{height}")
    if (
        guidance.width
        and guidance.height
        and (guidance.width, guidance.height)
        != (
            width,
            height,
        )
    ):
        raise ValueError(
            f"SAM contour size {guidance.width}x{guidance.height} does not match "
            f"video {width}x{height}; select the matching SAM run."
        )

    model_key = model.strip().lower()
    micro_batch = pose_batch_size or _default_pose_batch_size(model_key)
    spec = resolve_model_spec(model_key)
    session: PoseInferenceSession | None = None
    inferred: dict[int, list[dict[str, Any]]] = {}
    num_keypoints: int | None = None
    stride_eff = max(1, int(stride))
    _write_readme(
        output_dir,
        video_path=video_path,
        sam_dir=guidance.sam_dir,
        model=model_key,
        stride=stride_eff,
        contour_focus=contour_focus,
        bbox_padding=bbox_padding,
        contour_margin=contour_margin,
    )
    _log(
        f"Starting DETR-free pose: {video_path.name}, {width}x{height}, "
        f"{n_frames} frames, model={model_key}, stride={stride_eff}, "
        f"pose_batch={micro_batch}"
    )

    try:
        session = PoseInferenceSession(
            spec,
            device=f"cuda:{int(device)}",
            kpt_thr=float(kpt_thr),
            flip_test=bool(flip_test),
            max_persons=max(1, int(max_persons)),
            pose_batch_size=max(1, int(micro_batch)),
            use_detector=False,
        )
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Could not open video: {video_path}")
        frame_idx = 0
        heartbeat = max(1, n_frames // 20) if n_frames else 100
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_idx % stride_eff == 0:
                    frame_guidance = _guidance_for_frame(
                        guidance,
                        frame_idx,
                        min_score=min_sam_score,
                        min_area=min_sam_area,
                        max_persons=max_persons,
                    )
                    if frame_guidance:
                        pose_boxes: list[np.ndarray] = []
                        pose_images: list[np.ndarray] = []
                        masks: list[np.ndarray] = []
                        for track, contour_obj in frame_guidance:
                            pose_bbox = _pose_bbox_from_sam(
                                track,
                                contour_obj,
                                frame_width=width,
                                frame_height=height,
                                padding_fraction=bbox_padding,
                            )
                            mask = _contour_mask(
                                frame.shape,
                                track,
                                contour_obj,
                                margin_px=contour_margin,
                            )
                            pose_boxes.append(pose_bbox)
                            masks.append(mask)
                            if contour_focus:
                                pose_images.append(_contour_focused_image(frame, pose_bbox, mask))
                        keypoints, scores, _returned_boxes = session.process_frame_with_bboxes(
                            frame,
                            pose_boxes,
                            pose_images=pose_images if contour_focus else None,
                        )
                        instances: list[dict[str, Any]] = []
                        for (track, contour_obj), pose_bbox, mask, kpts, score_values in zip(
                            frame_guidance,
                            pose_boxes,
                            masks,
                            keypoints,
                            scores,
                            strict=True,
                        ):
                            adjusted, inside_flags = _attenuate_scores_outside_contour(
                                kpts,
                                score_values,
                                mask,
                                outside_factor=outside_contour_factor,
                            )
                            obj_id = int(track["obj_id"])
                            sam_bbox = list(_track_xyxy(track))
                            instances.append(
                                {
                                    "raw_id": obj_id,
                                    "temporal_id": obj_id,
                                    "stable_id": obj_id,
                                    "sam_obj_id": obj_id,
                                    "bbox": sam_bbox,
                                    "sam_bbox_xyxy": sam_bbox,
                                    "pose_bbox_xyxy": pose_bbox.tolist(),
                                    "sam_score": float(track.get("score", 1.0)),
                                    "sam_area_px": int(track.get("area_px", 0)),
                                    "sam_contour_points": sum(
                                        len(poly) for poly in _object_polygons(contour_obj)
                                    ),
                                    "keypoints": np.asarray(kpts, dtype=float).tolist(),
                                    "keypoint_scores": adjusted.astype(float).tolist(),
                                    "keypoints_inside_sam_contour": inside_flags,
                                }
                            )
                        inferred[frame_idx] = instances
                        if num_keypoints is None and instances:
                            num_keypoints = len(instances[0]["keypoints"])
                frame_idx += 1
                if frame_idx == 1 or frame_idx % heartbeat == 0 or frame_idx == n_frames:
                    _log(f"{video_path.stem}: pose {frame_idx}/{n_frames or '?'}")
        finally:
            cap.release()

        actual_frames = frame_idx
        timeline = _expand_pose_with_sam_guidance(
            inferred,
            guidance,
            n_frames=actual_frames,
            frame_width=width,
            frame_height=height,
            bbox_padding=bbox_padding,
            min_score=min_sam_score,
            min_area=min_sam_area,
            max_persons=max_persons,
        )
        keypoint_names = _resolve_sapiens_keypoint_names(session, n_kp=num_keypoints)
        stem = video_path.stem
        overlay_path: Path | None = None
        if save_overlay:
            _log("Writing combined contour + ID + skeleton overlay …")
            overlay_path = _write_combined_overlay(
                video_path,
                output_dir / f"{stem}_sam3sapiens2_overlay.mp4",
                timeline,
                guidance,
                session,
                min_score=min_sam_score,
                min_area=min_sam_area,
                max_persons=max_persons,
                draw_ids=draw_ids,
            )

        records: list[dict[str, Any]] = []
        csv_rows: list[tuple[int, int, int, float, float, float]] = []
        for frame_idx in sorted(timeline):
            instances = timeline[frame_idx]
            csv_rows.extend(flatten_instances_to_csv_rows(frame_idx, instances))
            records.append({"frame_index": frame_idx, "instances": instances})

        payload = {
            "schema": "vaila_sam3sapiens2_v1",
            "video": video_path.name,
            "image_size": [height, width],
            "fps": fps,
            "n_frames": actual_frames,
            "model": spec.arch,
            "stride": stride_eff,
            "num_keypoints": num_keypoints,
            "keypoint_names": keypoint_names or [],
            "detection_authority": "SAM3",
            "identity_authority": "SAM3 obj_id",
            "detr_loaded": False,
            "sam_results": str(guidance.sam_dir),
            "sam_contours": str(guidance.contour_path) if guidance.contour_path else None,
            "bbox_padding_fraction": float(bbox_padding),
            "contour_focus": bool(contour_focus),
            "contour_margin_px": int(contour_margin),
            "outside_contour_score_factor": float(outside_contour_factor),
            "frames": records,
        }
        json_path = output_dir / f"{stem}_sam3sapiens2_predictions.json"
        long_csv = output_dir / f"{stem}_sam3sapiens2_vaila.csv"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_vaila_pose_csv(long_csv, csv_rows)
        biomechanics = write_sapiens_biomechanics_csvs(
            output_dir,
            stem,
            timeline,
            kpt_thr=float(kpt_thr),
        )
        pose_csvs = write_sapiens_getpixelvideo_pose_csvs(
            output_dir,
            stem,
            timeline,
            kpt_thr=float(kpt_thr),
            keypoint_names=keypoint_names,
        )
        audit_path = output_dir / "sam3sapiens2_id_audit.csv"
        _write_identity_audit(audit_path, timeline)
        result = {
            "video": str(video_path),
            "sam_dir": str(guidance.sam_dir),
            "output_dir": str(output_dir),
            "overlay": str(overlay_path) if overlay_path else None,
            "predictions": str(json_path),
            "long_csv": str(long_csv),
            "identity_audit": str(audit_path),
            "n_frames": actual_frames,
            "n_inferred_frames": len(inferred),
            "n_stable_ids": len(
                {int(inst["stable_id"]) for instances in timeline.values() for inst in instances}
            ),
            "biomechanics_files": len(biomechanics),
            "pose_csv_files": len(pose_csvs),
        }
        (output_dir / "sam3sapiens2_summary.json").write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )
        _log(
            f"Done {video_path.name}: {result['n_stable_ids']} SAM IDs, "
            f"{len(inferred)} inferred frames -> {output_dir}"
        )
        return result
    finally:
        _release_sapiens_gpu_memory(session)


def _sam_script() -> Path:
    return Path(__file__).resolve().parent / "vaila_sam.py"


def build_sam_command(
    video_path: Path,
    sam_output_dir: Path,
    *,
    prompt: str,
    prompt_frame: int,
    checkpoint: Path | None,
    max_frames: int | None,
    max_input_long_edge: int | None,
    keep_masks: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(_sam_script()),
        "-i",
        str(video_path),
        "-o",
        str(sam_output_dir.parent),
        "--video-output-dir",
        str(sam_output_dir),
        "-t",
        prompt,
        "-f",
        str(max(0, int(prompt_frame))),
        "--no-overlay",
        "--save-contours",
        "--save-tracks-csv",
        "--postprocess-points",
        "none",
    ]
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(checkpoint)])
    if max_frames is not None:
        cmd.extend(["--max-frames", str(int(max_frames))])
    if max_input_long_edge is not None:
        cmd.extend(["--max-input-long-edge", str(int(max_input_long_edge))])
    if keep_masks:
        cmd.append("--keep-mask-png")
    else:
        cmd.append("--no-png")
    return cmd


def run_sam_stage(
    video_path: Path,
    sam_output_dir: Path,
    *,
    prompt: str,
    prompt_frame: int,
    checkpoint: Path | None,
    max_frames: int | None,
    max_input_long_edge: int | None,
    keep_masks: bool,
) -> None:
    sam_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_sam_command(
        video_path,
        sam_output_dir,
        prompt=prompt,
        prompt_frame=prompt_frame,
        checkpoint=checkpoint,
        max_frames=max_frames,
        max_input_long_edge=max_input_long_edge,
        keep_masks=keep_masks,
    )
    _log("SAM3 stage CLI: " + shlex.join(cmd))
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(cmd, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"SAM3 stage failed with subprocess exit={completed.returncode}")
    if not (sam_output_dir / "sam_tracks.csv").is_file():
        raise RuntimeError(f"SAM3 finished but sam_tracks.csv is missing: {sam_output_dir}")


def _write_failure(output_dir: Path, video_path: Path, reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "FAILED_sam3sapiens2.txt").write_text(
        "SAM3+Sapiens2 FAILED\n"
        f"video={video_path}\n"
        f"timestamp={dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"reason={reason}\n",
        encoding="utf-8",
    )


def _process_one_video(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.worker_sam_dir is not None:
            sam_dir = args.worker_sam_dir.expanduser().resolve()
            _log(f"Reusing SAM3 results: {sam_dir}")
        else:
            sam_dir = output_dir / "sam3"
            run_sam_stage(
                video_path,
                sam_dir,
                prompt=args.text,
                prompt_frame=args.sam_frame,
                checkpoint=args.sam_checkpoint,
                max_frames=args.sam_max_frames,
                max_input_long_edge=args.sam_max_input_long_edge,
                keep_masks=args.keep_sam_masks,
            )
        return run_sapiens_from_sam(
            video_path,
            output_dir,
            sam_dir,
            model=args.model,
            stride=args.stride,
            device=args.device,
            kpt_thr=args.kpt_thr,
            pose_batch_size=args.pose_batch_size,
            bbox_padding=args.bbox_padding,
            contour_margin=args.contour_margin,
            outside_contour_factor=args.outside_contour_factor,
            min_sam_score=args.min_sam_score,
            min_sam_area=args.min_sam_area,
            max_persons=args.max_persons,
            contour_focus=not args.no_contour_focus,
            flip_test=args.flip_test,
            save_overlay=not args.no_overlay,
            draw_ids=not args.no_draw_id,
        )
    except Exception as exc:
        _write_failure(output_dir, video_path, str(exc))
        raise


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
        args.text,
        "--model",
        args.model,
        "--stride",
        str(args.stride),
        "--device",
        str(args.device),
        "--kpt-thr",
        str(args.kpt_thr),
        "--bbox-padding",
        str(args.bbox_padding),
        "--contour-margin",
        str(args.contour_margin),
        "--outside-contour-factor",
        str(args.outside_contour_factor),
        "--min-sam-score",
        str(args.min_sam_score),
        "--min-sam-area",
        str(args.min_sam_area),
        "--max-persons",
        str(args.max_persons),
        "--sam-frame",
        str(args.sam_frame),
    ]
    if sam_dir is not None:
        cmd.extend(["--worker-sam-dir", str(sam_dir)])
    _add_if_value(cmd, "--pose-batch-size", args.pose_batch_size)
    _add_if_value(cmd, "--sam-checkpoint", args.sam_checkpoint)
    _add_if_value(cmd, "--sam-max-frames", args.sam_max_frames)
    _add_if_value(cmd, "--sam-max-input-long-edge", args.sam_max_input_long_edge)
    for enabled, flag in (
        (args.keep_sam_masks, "--keep-sam-masks"),
        (args.no_contour_focus, "--no-contour-focus"),
        (args.flip_test, "--flip-test"),
        (args.no_overlay, "--no-overlay"),
        (args.no_draw_id, "--no-draw-id"),
    ):
        if enabled:
            cmd.append(flag)
    return cmd


def _build_dry_run_report(
    videos: list[Path],
    output_base: Path,
    args: argparse.Namespace,
) -> list[str]:
    lines = [
        "SAM3+Sapiens2 dry-run (no GPU/model inference)",
        f"input={args.input}",
        f"output_base={output_base}",
        f"videos={len(videos)}",
        f"sam_results={args.sam_results or 'run SAM3 first'}",
        f"model={args.model} stride={args.stride} device=cuda:{args.device}",
        f"DETR=disabled bbox_padding={args.bbox_padding} contour_margin={args.contour_margin}",
        f"identity_authority=SAM3 obj_id contour_focus={not args.no_contour_focus}",
    ]
    for video in videos:
        lines.append(f"video={video}")
        if args.sam_results is not None:
            try:
                resolved = resolve_sam_results_dir(
                    args.sam_results,
                    video,
                    single_video=len(videos) == 1,
                )
                guidance = load_sam_guidance(resolved)
                lines.append(
                    f"  SAM OK: {resolved} frames={len(guidance.tracks_by_frame)} "
                    f"contours={bool(guidance.contour_path)}"
                )
            except Exception as exc:
                lines.append(f"  SAM ERROR: {exc}")
        else:
            sam_dir = output_base / video.stem / "sam3"
            lines.append(
                "  SAM CLI: "
                + shlex.join(
                    build_sam_command(
                        video,
                        sam_dir,
                        prompt=args.text,
                        prompt_frame=args.sam_frame,
                        checkpoint=args.sam_checkpoint,
                        max_frames=args.sam_max_frames,
                        max_input_long_edge=args.sam_max_input_long_edge,
                        keep_masks=args.keep_sam_masks,
                    )
                )
            )
    return lines


def _format_gui_cli(settings: CombinedGuiSettings) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "vaila/sam3sapiens2.py",
        "-i",
        str(settings.input_path),
        "-o",
        str(settings.output_parent),
        "-t",
        settings.prompt,
        "--model",
        settings.model,
        "--stride",
        str(settings.stride),
        "--device",
        str(settings.device),
        "--kpt-thr",
        str(settings.kpt_thr),
        "--bbox-padding",
        str(settings.bbox_padding),
        "--contour-margin",
        str(settings.contour_margin),
        "--max-persons",
        str(settings.max_persons),
    ]
    if settings.sam_results is not None:
        cmd.extend(["--sam-results", str(settings.sam_results)])
    if settings.pose_batch_size is not None:
        cmd.extend(["--pose-batch-size", str(settings.pose_batch_size)])
    if settings.flip_test:
        cmd.append("--flip-test")
    if not settings.save_overlay:
        cmd.append("--no-overlay")
    return cmd


def run_sam3sapiens2(existing_root: Any | None = None) -> None:
    """Open the combined pipeline GUI and launch its reproducible CLI."""
    owns_root = existing_root is None
    root = existing_root if existing_root is not None else tk.Tk()
    if owns_root:
        root.withdraw()
    _prepare_gui_root(root, owns_root=owns_root)

    class CombinedDialog(tk.Toplevel):
        def __init__(self, master: tk.Misc) -> None:
            super().__init__(master)
            self.title("SAM3+Sapiens2 — SAM-guided pose")
            self.resizable(False, False)
            self.result: CombinedGuiSettings | None = None
            frm = ttk.Frame(self, padding=14)
            frm.grid(sticky="nsew")
            self.input_var = tk.StringVar()
            self.output_var = tk.StringVar()
            self.sam_var = tk.StringVar()
            self.prompt_var = tk.StringVar(value="person")
            self.model_var = tk.StringVar(value="1b")
            self.stride_var = tk.StringVar(value="1")
            self.device_var = tk.StringVar(value="0")
            self.kpt_var = tk.StringVar(value=str(DEFAULT_KPT_THR))
            self.batch_var = tk.StringVar(value="2")
            self.padding_var = tk.StringVar(value=str(DEFAULT_BBOX_PADDING))
            self.margin_var = tk.StringVar(value=str(DEFAULT_CONTOUR_MARGIN_PX))
            self.max_persons_var = tk.StringVar(value=str(DEFAULT_MAX_PERSONS))
            self.flip_var = tk.BooleanVar(value=False)
            self.overlay_var = tk.BooleanVar(value=True)

            ttk.Label(
                frm,
                text="SAM3 + Sapiens2",
                font=("TkDefaultFont", 12, "bold"),
            ).grid(row=0, column=0, columnspan=5, sticky="w", pady=(0, 8))
            ttk.Label(
                frm,
                text=(
                    "SAM3 defines bbox, contour and ID; Sapiens2 pose runs without DETR. "
                    "Use Dir… for batch (all videos in a folder) or File… for one clip."
                ),
            ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(0, 10))
            self._input_path_row(frm, 2)
            self._path_row(frm, 3, "Output parent", self.output_var, self._browse_output)
            self._path_row(
                frm,
                4,
                "Existing SAM results (optional)",
                self.sam_var,
                self._browse_sam,
            )

            ttk.Label(frm, text="SAM prompt").grid(row=5, column=0, sticky="w", pady=(8, 0))
            ttk.Entry(frm, textvariable=self.prompt_var, width=18).grid(row=5, column=1, sticky="w")
            ttk.Label(frm, text="Sapiens model").grid(row=5, column=2, sticky="e", padx=(12, 4))
            ttk.Combobox(
                frm,
                textvariable=self.model_var,
                values=("0.4b", "0.8b", "1b", "5b"),
                state="readonly",
                width=8,
            ).grid(row=5, column=3, sticky="w")

            options = (
                ("Stride", self.stride_var),
                ("CUDA device", self.device_var),
                ("Kpt threshold", self.kpt_var),
                ("Pose batch", self.batch_var),
                ("BBox padding", self.padding_var),
                ("Contour margin px", self.margin_var),
                ("Max persons", self.max_persons_var),
            )
            for idx, (label, variable) in enumerate(options):
                row = 6 + idx // 2
                col = (idx % 2) * 2
                ttk.Label(frm, text=label).grid(row=row, column=col, sticky="w", pady=(6, 0))
                ttk.Entry(frm, textvariable=variable, width=10).grid(
                    row=row, column=col + 1, sticky="w", pady=(6, 0)
                )
            ttk.Checkbutton(frm, text="Flip test", variable=self.flip_var).grid(
                row=10, column=0, columnspan=2, sticky="w", pady=(8, 0)
            )
            ttk.Checkbutton(
                frm,
                text="Save combined overlay",
                variable=self.overlay_var,
            ).grid(row=10, column=2, columnspan=2, sticky="w", pady=(8, 0))
            buttons = ttk.Frame(frm)
            buttons.grid(row=11, column=0, columnspan=4, pady=(14, 0))
            ttk.Button(buttons, text="Help", command=self._open_help).pack(side="left", padx=4)
            ttk.Button(buttons, text="Run", command=self._on_run).pack(side="left", padx=4)
            ttk.Button(buttons, text="Cancel", command=self.destroy).pack(side="left", padx=4)
            self.transient(master)  # ty: ignore[no-matching-overload]
            self.grab_set()
            _show_dialog_in_front(self)

        def _path_row(
            self,
            parent: ttk.Frame,
            row: int,
            label: str,
            variable: tk.StringVar,
            command: Any,
        ) -> None:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(parent, textvariable=variable, width=58).grid(
                row=row, column=1, columnspan=2, sticky="ew", padx=4
            )
            ttk.Button(parent, text="Browse…", command=command).grid(row=row, column=3)

        def _input_path_row(self, parent: ttk.Frame, row: int) -> None:
            ttk.Label(parent, text="Input video/folder").grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(parent, textvariable=self.input_var, width=58).grid(
                row=row, column=1, columnspan=2, sticky="ew", padx=4
            )
            in_btns = ttk.Frame(parent)
            in_btns.grid(row=row, column=3, sticky="w")
            ttk.Button(in_btns, text="Dir…", command=self._browse_dir).pack(side="left", padx=1)
            ttk.Button(in_btns, text="File…", command=self._browse_file).pack(side="left", padx=1)

        def _browse_dir(self) -> None:
            path = filedialog.askdirectory(
                parent=self,
                title="Select folder with videos (batch)",
            )
            if path:
                self.input_var.set(path)

        def _browse_file(self) -> None:
            path = filedialog.askopenfilename(
                parent=self,
                title="Select one video file",
                filetypes=[
                    ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("All", "*.*"),
                ],
            )
            if path:
                self.input_var.set(path)

        def _browse_output(self) -> None:
            path = filedialog.askdirectory(parent=self, title="Select output parent")
            if path:
                self.output_var.set(path)

        def _browse_sam(self) -> None:
            path = filedialog.askdirectory(
                parent=self,
                title="Select processed_sam_* or per-video SAM directory",
            )
            if path:
                self.sam_var.set(path)

        def _open_help(self) -> None:
            if _help_path().is_file():
                webbrowser.open_new_tab(_help_path().as_uri())

        def _on_run(self) -> None:
            try:
                input_path = Path(self.input_var.get().strip()).expanduser()
                output_parent = Path(self.output_var.get().strip()).expanduser()
                if not self.input_var.get().strip() or not input_path.exists():
                    raise ValueError("Select an existing input video or folder (Dir… / File…).")
                videos = _find_videos(input_path)
                if not videos:
                    raise ValueError(
                        f"No supported videos found under: {input_path}\n"
                        "Use Dir… for a folder of .mp4/.avi/.mov/.mkv/.webm files, "
                        "or File… for a single clip."
                    )
                if not self.output_var.get().strip():
                    raise ValueError("Select an output parent folder.")
                sam_raw = self.sam_var.get().strip()
                sam_results = Path(sam_raw).expanduser() if sam_raw else None
                if sam_results is not None and not sam_results.exists():
                    raise ValueError(f"Existing SAM results path not found: {sam_results}")
                batch_raw = self.batch_var.get().strip()
                result = CombinedGuiSettings(
                    input_path=input_path,
                    output_parent=output_parent,
                    sam_results=sam_results,
                    prompt=self.prompt_var.get().strip() or "person",
                    model=self.model_var.get().strip() or "1b",
                    stride=max(1, int(self.stride_var.get())),
                    device=max(0, int(self.device_var.get())),
                    kpt_thr=float(self.kpt_var.get()),
                    pose_batch_size=max(1, int(batch_raw)) if batch_raw else None,
                    bbox_padding=max(0.0, float(self.padding_var.get())),
                    contour_margin=max(0, int(self.margin_var.get())),
                    max_persons=max(1, int(self.max_persons_var.get())),
                    flip_test=bool(self.flip_var.get()),
                    save_overlay=bool(self.overlay_var.get()),
                )
                if not 0.0 <= result.kpt_thr <= 1.0:
                    raise ValueError("Kpt threshold must be between 0 and 1.")
            except ValueError as exc:
                messagebox.showerror("SAM3+Sapiens2", str(exc), parent=self)
                return
            _log(
                f"Queued {len(videos)} video(s) from {'folder' if input_path.is_dir() else 'file'}: "
                + ", ".join(v.name for v in videos[:8])
                + ("…" if len(videos) > 8 else "")
            )
            self.result = result
            self.destroy()

    _log("Opening the settings window; bare CLI without -i/-o intentionally uses this GUI.")
    dialog = CombinedDialog(root)
    _log("Settings window opened and raised; waiting for Run or Cancel.")
    root.wait_window(dialog)
    settings = dialog.result
    if owns_root:
        root.destroy()
    if settings is None:
        return
    cli = _format_gui_cli(settings)
    print("\n>> vaila/sam3sapiens2: Equivalent CLI (copy/paste):", flush=True)
    print(">>   " + shlex.join(cli), flush=True)
    launch = [sys.executable, "-u", str(Path(__file__).resolve())] + cli[5:]
    raise SystemExit(subprocess.call(launch))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SAM3-guided Sapiens2 pose: SAM3 owns bbox/contour/ID; "
            "Sapiens2 runs top-down without DETR."
        )
    )
    parser.add_argument("-i", "--input", type=Path, help="Input video or non-recursive folder")
    parser.add_argument("-o", "--output", type=Path, help="Output parent directory")
    parser.add_argument("-t", "--text", default="person", help="SAM3 text prompt")
    parser.add_argument(
        "--sam-results",
        type=Path,
        default=None,
        help="Reuse processed_sam_* batch parent or a per-video SAM output dir",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        choices=("0.4b", "0.8b", "1b", "5b"),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--kpt-thr", type=float, default=DEFAULT_KPT_THR)
    parser.add_argument("--pose-batch-size", type=int, default=None)
    parser.add_argument("--bbox-padding", type=float, default=DEFAULT_BBOX_PADDING)
    parser.add_argument("--contour-margin", type=int, default=DEFAULT_CONTOUR_MARGIN_PX)
    parser.add_argument(
        "--outside-contour-factor",
        type=float,
        default=DEFAULT_OUTSIDE_CONTOUR_FACTOR,
        help="Confidence multiplier for keypoints outside the dilated SAM contour",
    )
    parser.add_argument("--min-sam-score", type=float, default=DEFAULT_MIN_SAM_SCORE)
    parser.add_argument("--min-sam-area", type=int, default=DEFAULT_MIN_SAM_AREA)
    parser.add_argument("--max-persons", type=int, default=DEFAULT_MAX_PERSONS)
    parser.add_argument(
        "--no-contour-focus",
        action="store_true",
        help="Use SAM boxes/IDs but do not blur crop background",
    )
    parser.add_argument("--flip-test", action="store_true")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-draw-id", action="store_true")
    parser.add_argument("--sam-frame", type=int, default=0)
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-max-frames", type=int, default=None)
    parser.add_argument("--sam-max-input-long-edge", type=int, default=None)
    parser.add_argument("--keep-sam-masks", action="store_true")
    parser.add_argument(
        "--no-isolate-batch",
        action="store_true",
        help="Debug: process videos in coordinator process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print plan without GPU inference",
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
        return
    if args.input is None and args.output is None:
        run_sam3sapiens2()
        return
    if args.input is None or args.output is None:
        parser.error("-i/--input and -o/--output must be supplied together")
    if not 0.0 <= args.kpt_thr <= 1.0:
        parser.error("--kpt-thr must be in [0,1]")
    if not 0.0 <= args.min_sam_score <= 1.0:
        parser.error("--min-sam-score must be in [0,1]")
    if not 0.0 <= args.outside_contour_factor <= 1.0:
        parser.error("--outside-contour-factor must be in [0,1]")
    if args.bbox_padding < 0:
        parser.error("--bbox-padding must be >= 0")

    input_path = args.input.expanduser().resolve()
    output_parent = args.output.expanduser().resolve()
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
        )
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = output_parent / f"processed_sam3sapiens2_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        lines = _build_dry_run_report(videos, output_base, args)
        report = output_base / "SAM3SAPIENS2_DRY_RUN.txt"
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        print(f"Dry-run report: {report}", flush=True)
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    failed: list[str] = []
    summaries: list[dict[str, Any]] = []
    use_isolation = not args.no_isolate_batch
    for index, video in enumerate(videos, start=1):
        output_dir = output_base / video.stem
        sam_dir: Path | None = None
        if args.sam_results is not None:
            sam_dir = resolve_sam_results_dir(
                args.sam_results,
                video,
                single_video=len(videos) == 1,
            )
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
                rc = subprocess.call(cmd, env=os.environ.copy())
                if rc != 0:
                    raise RuntimeError(f"combined worker subprocess exit={rc}")
                summary_path = output_dir / "sam3sapiens2_summary.json"
                if summary_path.is_file():
                    summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            else:
                worker_args = argparse.Namespace(**vars(args))
                worker_args.worker_sam_dir = sam_dir
                summaries.append(_process_one_video(video, output_dir, worker_args))
        except Exception as exc:
            failed.append(f"{video.name}: {exc}")
            _log(f"ERROR: {failed[-1]}")

    batch_summary = {
        "schema": "vaila_sam3sapiens2_batch_v1",
        "input": str(input_path),
        "output": str(output_base),
        "succeeded": len(summaries),
        "failed": failed,
        "videos": summaries,
    }
    (output_base / "sam3sapiens2_batch_summary.json").write_text(
        json.dumps(batch_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _log(f"Batch done: {len(summaries)}/{len(videos)} succeeded -> {output_base}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
