"""
Project: vailá
Script: sam3dinov3_visualize.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 01 August 2026
Update Date: 24 August 2026
Version: 0.3.112

Description:
    CPU-only rerenderer for an existing SAM3+DINOv3 3D (SAM 3D Body) run. It
    selects one person (SAM `obj_id`), draws its contour, bbox, ID, and the
    reprojected MHR skeleton on the original video, and writes ID-specific
    tracking/keypoint/mesh artifacts. SAM3/SAM 3D Body weights are never
    loaded, so this is safe to run on a CPU-only machine right after a GPU
    inference run.

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

    # Omit --id to be prompted interactively with the available person IDs.
    # GUI: omit all arguments, or use Frame B -> YOLO + FB -> SAM3+DINOv3 Visualize ID
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import gzip
import json
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
import numpy as np

try:
    from .sam3dinov3 import (
        COLOR_CENTER_RGB,
        MHR70_NAMES,
        _rgb_to_bgr,
        _side_color_bgr,
        keypoint_names,
        skeleton_edges,
    )
except ImportError:  # standalone execution
    from sam3dinov3 import (  # ty: ignore[unresolved-import]
        COLOR_CENTER_RGB,
        MHR70_NAMES,
        _rgb_to_bgr,
        _side_color_bgr,
        keypoint_names,
        skeleton_edges,
    )

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


def _unique_gui_output_dir(output_parent: Path, video_path: Path, selected_id: int) -> Path:
    """Return a new ID-specific child directory below a GUI-selected parent."""
    output_parent = output_parent.expanduser().resolve()
    base = output_parent / f"{video_path.stem}_sam3dinov3_visualized_id_{selected_id:02d}"
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


def prompt_selected_id(available: list[int]) -> int:
    """Ask for a person ID on stdin until a valid choice is entered."""
    if not available:
        raise ValueError("No person IDs available to prompt")
    while True:
        try:
            raw = input(">> Enter SAM/person ID to visualize: ").strip()
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
        frame_idx = _safe_int(frame.get("frame"))
        if frame_idx is None:
            continue
        for instance in frame.get("instances", []):
            value = _safe_int(instance.get("person_id", instance.get("sam_obj_id")))
            if value == selected_id:
                records[frame_idx] = instance
                break
    return records


def _contours_by_frame(run_dir: Path, selected_id: int) -> dict[int, dict[str, Any]]:
    path = run_dir / "sam3" / "sam_contours.json"
    if not path.exists():
        return {}
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
        cam_t = instance.get("cam_t_m") or [0.0, 0.0, 0.0]
        depth = _safe_float(cam_t[2] if len(cam_t) > 2 else None)
        label = f"ID {selected_id}  z={depth:.2f} m"
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
    edges: list[tuple[int, int]],
    names: list[str],
    *,
    selected_id: int,
) -> tuple[Path, int, int]:
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
    if tqdm_cls is not None:
        pbar = tqdm_cls(
            total=total_frames if total_frames > 0 else None,
            desc=f">> sam3dinov3_visualize render ID {selected_id}",
            unit="frame",
            leave=True,
        )
    else:
        _log(f"Rendering video overlay for ID {selected_id} ({total_frames} frames)...")

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
                    edges,
                    names,
                    selected_id=selected_id,
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
                    _log(
                        f"Rendering ID {selected_id}: frame {frame_count}/{total_frames} ({pct:.1f}%)"
                    )
            elif frame_count % 100 == 0:
                _log(f"Rendering ID {selected_id}: frame {frame_count}...")
    finally:
        if pbar is not None:
            pbar.close()
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


def _write_filtered_predictions(path: Path, output: Path, selected_id: int) -> bool:
    if not path.exists():
        return False
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    for frame in payload.get("frames", []):
        frame["instances"] = [
            inst
            for inst in frame.get("instances", [])
            if _safe_int(inst.get("person_id", inst.get("sam_obj_id"))) == selected_id
        ]
    with gzip.open(output, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return True


def _write_filtered_contours(path: Path, output: Path, selected_id: int) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["object_ids"] = [selected_id]
    for frame in payload.get("frames", []):
        frame["objects"] = [
            o for o in frame.get("objects", []) if _safe_int(o.get("obj_id")) == selected_id
        ]
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def _write_filtered_meshes(run_dir: Path, output_dir: Path, selected_id: int) -> int:
    """Extract the selected person's vertices/cam_t from each per-frame mesh npz."""
    mesh_dir = run_dir / "meshes"
    if not mesh_dir.is_dir():
        return 0
    out_mesh_dir = output_dir / "meshes"
    written = 0
    for source in sorted(mesh_dir.glob("frame_*.npz")):
        with np.load(source) as data:
            obj_ids = np.asarray(data["obj_ids"])
            matches = np.nonzero(obj_ids == selected_id)[0]
            if matches.size == 0:
                continue
            idx = int(matches[0])
            out_mesh_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_mesh_dir / source.name,
                obj_ids=np.asarray([selected_id], dtype=np.int32),
                vertices=data["vertices"][idx : idx + 1],
                cam_t=data["cam_t"][idx : idx + 1],
            )
            written += 1
    faces_path = run_dir / "mesh_faces.npy"
    if written and faces_path.is_file():
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


def write_selected_artifacts(
    run_dir: Path, output_dir: Path, selected_id: int, payload: dict[str, Any]
) -> list[str]:
    """Write filtered artifacts and preserve all source artifacts for provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_artifacts"
    source_dir.mkdir(exist_ok=True)
    written: list[str] = []
    stem = Path(str(payload.get("video", "video.mp4"))).stem
    id_tag = f"{selected_id:02d}"

    mapping = (
        (run_dir / "sam3" / "sam_tracks.csv", output_dir / "sam_tracks.csv", ("obj_id",)),
        (run_dir / "sam3" / "sam_bbox_tracks.csv", output_dir / "sam_bbox_tracks.csv", ("obj_id",)),
        (
            run_dir / f"{stem}_sam3dinov3_keypoints3d.csv",
            output_dir / f"{stem}_sam3dinov3_keypoints3d.csv",
            ("person_id",),
        ),
        (
            run_dir / f"{stem}_sam3dinov3_keypoints2d.csv",
            output_dir / f"{stem}_sam3dinov3_keypoints2d.csv",
            ("person_id",),
        ),
        (
            run_dir / f"{stem}_sam3dinov3_camera.csv",
            output_dir / f"{stem}_sam3dinov3_camera.csv",
            ("person_id",),
        ),
        (
            run_dir / f"{stem}_sam3dinov3_joint_angles.csv",
            output_dir / f"{stem}_sam3dinov3_joint_angles.csv",
            ("person_id",),
        ),
    )
    for source, target, columns in mapping:
        if _filter_rows(source, target, columns, selected_id):
            written.append(str(target.name))

    # Per-ID wide CSVs already exist in the source run (person_id == sam_obj_id).
    for suffix in ("mhr70_3d", "mhr70_rec3d", "markers"):
        source = run_dir / f"{stem}_id_{id_tag}_{suffix}.csv"
        if source.is_file():
            target = output_dir / source.name
            shutil.copy2(source, target)
            written.append(target.name)

    if _write_filtered_predictions(
        _predictions_path(run_dir),
        output_dir / f"{stem}_sam3dinov3_predictions.json.gz",
        selected_id,
    ):
        written.append(f"{stem}_sam3dinov3_predictions.json.gz")
    if _write_filtered_contours(
        run_dir / "sam3" / "sam_contours.json",
        output_dir / "sam_contours.json",
        selected_id,
    ):
        written.append("sam_contours.json")
    n_meshes = _write_filtered_meshes(run_dir, output_dir, selected_id)
    if n_meshes:
        written.append(f"meshes/ ({n_meshes} frames)")
        written.append("mesh_faces.npy")

    # Keep every original non-video artifact in a namespaced folder. The root
    # remains ID-specific, while provenance is never silently discarded.
    for source in run_dir.rglob("*"):
        if not source.is_file() or source == output_dir or output_dir in source.parents:
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


def visualize_selected_id(
    run_dir: Path,
    video_path: Path,
    selected_id: int,
    output_dir: Path,
    *,
    overwrite: bool = False,
    export_mesh: str = "none",
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
    names = payload.get("keypoint_names")
    if not isinstance(names, list) or not names:
        n_kpts = len(next(iter(records.values()), {}).get("keypoints_3d_m", MHR70_NAMES))
        names = keypoint_names(n_kpts)
    names = [str(n) for n in names]
    edges = skeleton_edges(names)

    _log(f"Starting visualization for ID {selected_id} on video: {video_path.name}")
    _log(f"Rendering overlay video for ID {selected_id}...")
    overlay, frames, drawn = render_selected_video(
        video_path,
        output_dir / f"{video_path.stem}_sam3dinov3_id_{selected_id:02d}_overlay.mp4",
        records,
        contours,
        edges,
        names,
        selected_id=selected_id,
    )
    _log(f"Filtering and writing artifacts for ID {selected_id}...")
    written = write_selected_artifacts(run_dir, output_dir, selected_id, payload)

    mesh_export_dir: Path | None = None
    n_mesh_exported = 0
    if export_mesh != "none":
        mesh_dir = output_dir / "meshes"
        faces_path = output_dir / "mesh_faces.npy"
        if mesh_dir.is_dir() and faces_path.is_file():
            mesh_export_dir = output_dir / f"meshes_{export_mesh}"
            exported = export_mesh_sequence(
                mesh_dir, faces_path, mesh_export_dir, person_id=selected_id, fmt=export_mesh
            )
            n_mesh_exported = len(exported)
            _log(
                f"Exported {n_mesh_exported} {export_mesh.upper()} mesh frames -> {mesh_export_dir}"
            )
        else:
            _log(
                "No meshes/ found for this ID; rerun sam3dinov3.py with --save-mesh "
                "to get body-mesh vertices for Blender export."
            )

    manifest = {
        "schema": "vaila_sam3dinov3_selected_id_v1",
        "source_run": str(run_dir),
        "source_video": str(video_path),
        "selected_id": selected_id,
        "available_ids": available,
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
    (output_dir / "README_sam3dinov3_selected_id.txt").write_text(
        "vailá SAM3+DINOv3 3D selected-ID visualization\n"
        f"selected_id={selected_id}\nsource_run={run_dir}\nsource_video={video_path}\n"
        "identity_authority=SAM3 obj_id (person_id == sam_obj_id)\n"
        "coordinate_units=full-frame pixels for 2D; metres for 3D; frame_index=zero-based\n"
        "The root contains filtered artifacts. source_artifacts/ preserves the original run.\n"
        "Overlay style: SAM3 contour fill/outline + left/right/center MHR70 skeleton + depth label.\n"
        f"{video_path.stem}_sam3dinov3_joint_angles.csv: local (parent-relative) joint angles for the\n"
        "127-joint MHR rig -- Euler XYZ degrees + scalar-first (w,x,y,z) quaternion, from the\n"
        "model's own regressed rotations (present only for runs made after this feature was\n"
        "added; older runs have no *_joint_angles.csv to filter).\n"
        f"{blender_note}",
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
    dialog.title("SAM3+DINOv3 3D — Visualize one person")
    dialog.resizable(False, False)
    vars_: dict[str, tk.StringVar] = {
        "run": tk.StringVar(),
        "video": tk.StringVar(),
        "output": tk.StringVar(),
        "id": tk.StringVar(),
        "status": tk.StringVar(value="Choose a processed_sam3dinov3_* directory first."),
    }
    export_mesh_var = tk.BooleanVar(value=False)
    frame = ttk.Frame(dialog, padding=14)
    frame.grid(sticky="nsew")
    ttk.Label(
        frame, text="SAM3+DINOv3 3D — selected-ID visualization", font=("TkDefaultFont", 11, "bold")
    ).grid(column=0, row=0, columnspan=3, sticky="w", pady=(0, 10))

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
    ttk.Checkbutton(
        frame,
        text="Export mesh sequence (.obj, for Blender — needs --save-mesh in the source run)",
        variable=export_mesh_var,
    ).grid(column=0, row=5, columnspan=3, sticky="w", pady=2)
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
        except Exception as exc:
            messagebox.showerror(
                "SAM3+DINOv3 3D visualization", f"Invalid settings: {exc}", parent=dialog
            )
            return
        export_mesh = "obj" if export_mesh_var.get() else "none"
        cli = [
            "uv run python -u vaila/sam3dinov3_visualize.py",
            "--sam3d-results",
            str(run_dir),
            "--video",
            str(video),
            "--id",
            str(selected),
            "--output",
            str(output),
        ]
        if export_mesh != "none":
            cli.extend(["--export-mesh", export_mesh])
        _log("GUI equivalent CLI: " + " ".join(cli))
        vars_["status"].set("Rendering selected ID; the window remains responsive…")

        def worker() -> None:
            try:
                result = visualize_selected_id(
                    run_dir, video, selected, output, export_mesh=export_mesh
                )
                dialog.after(
                    0,
                    lambda: (
                        vars_["status"].set(f"Done: {result['overlay']}"),
                        messagebox.showinfo(
                            "SAM3+DINOv3 3D visualization",
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
                            "SAM3+DINOv3 3D visualization", error_text, parent=dialog
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
        description="Render one SAM3+DINOv3 3D person ID from an existing run."
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
        dest="selected_id",
        type=int,
        help="SAM/person ID to visualize. If omitted in CLI mode, prompt interactively.",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not any((args.sam3d_results, args.video, args.selected_id is not None, args.output)):
        run_visualizer_gui()
        return 0
    if args.sam3d_results is None or args.video is None:
        parser.error("--sam3d-results/--input-dir and --video/-i must be supplied together")
    run_dir = resolve_run_dir(args.sam3d_results, args.video)
    payload = load_predictions(run_dir)
    ids = discover_ids(run_dir, payload)
    print(f">> Available person IDs: {ids}")
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
        overwrite=args.overwrite,
        export_mesh=args.export_mesh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
