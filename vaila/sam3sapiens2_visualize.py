"""
Project: vailá
Script: sam3sapiens2_visualize.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Creation Date: 31 July 2026
Update Date: 31 July 2026
Version: 0.3.86

Description:
    CPU-only rerenderer for an existing SAM3+Sapiens2 run. It selects one
    SAM object ID, draws its contour, bbox, ID, and Sapiens2 keypoints on the
    original video, and writes ID-specific tracking/pose/contour artifacts.

Usage:
    uv run python -u vaila/sam3sapiens2_visualize.py \
        --sam-results /path/to/processed_sam3sapiens2_.../video_stem \
        --video /path/to/video.mp4 --id 2 --output /path/to/output

    # GUI: omit all arguments, or use Frame B -> YOLO + FB -> SAM3+Sapiens2 Visualize ID
"""

from __future__ import annotations

import argparse
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

# Sapiens2's first 21 points are the COCO-style body/foot topology. The
# remaining 287 face/hand/ear points are rendered as dots when visible.
BODY_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (13, 17),
    (12, 14),
    (14, 18),
    (14, 20),
    (15, 16),
    (18, 19),
)


def _log(message: str) -> None:
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


def _draw_instance(
    image: np.ndarray,
    instance: dict[str, Any],
    contour: dict[str, Any] | None,
    *,
    selected_id: int,
    kpt_thr: float,
    draw_all_keypoints: bool,
) -> np.ndarray:
    out = image.copy()
    color = _color_for_id(selected_id)
    if contour:
        polygons = []
        for raw in contour.get("polygons", []):
            poly = np.asarray(raw, dtype=np.int32).reshape(-1, 1, 2)
            if len(poly) >= 3:
                polygons.append(poly)
        if polygons:
            cv2.polylines(out, polygons, True, color, 3, cv2.LINE_AA)
            overlay = out.copy()
            cv2.fillPoly(overlay, polygons, color)
            out = cv2.addWeighted(overlay, 0.10, out, 0.90, 0.0)
    bbox = instance.get("sam_bbox_xyxy") or instance.get("bbox")
    if bbox and len(bbox) >= 4:
        x1, y1, x2, y2 = [int(round(_safe_float(v))) for v in bbox[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
        label = f"SAM/Sapiens2 ID {selected_id}"
        cv2.putText(
            out,
            label,
            (max(0, x1), max(26, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    points = np.asarray(instance.get("keypoints") or [], dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(instance.get("keypoint_scores") or [], dtype=np.float32).reshape(-1)
    valid = np.zeros(len(points), dtype=bool)
    for idx, (x, y) in enumerate(points):
        score = float(scores[idx]) if idx < len(scores) else 1.0
        valid[idx] = score >= kpt_thr and np.isfinite(x) and np.isfinite(y)
    for left, right in BODY_EDGES:
        if left < len(points) and right < len(points) and valid[left] and valid[right]:
            cv2.line(
                out,
                tuple(np.round(points[left]).astype(int)),
                tuple(np.round(points[right]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    draw_indices = range(len(points)) if draw_all_keypoints else range(min(21, len(points)))
    for idx in draw_indices:
        if valid[idx]:
            radius = 3 if idx < 21 else 2
            cv2.circle(
                out,
                tuple(np.round(points[idx]).astype(int)),
                radius,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(out, tuple(np.round(points[idx]).astype(int)), radius, color, 1, cv2.LINE_AA)
    return out


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, codec in ((".mp4", "mp4v"), (".avi", "XVID")):
        candidate = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*codec), fps, size)
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
) -> tuple[Path, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    overlay, frames, drawn = render_selected_video(
        video_path,
        output_dir / f"{video_path.stem}_sam3sapiens2_id_{selected_id:02d}_overlay.mp4",
        records,
        contours,
        selected_id=selected_id,
        kpt_thr=kpt_thr,
        draw_all_keypoints=draw_all_keypoints,
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
        "The root contains filtered artifacts. source_artifacts/ preserves the original run.\n",
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
            ids = discover_ids(
                resolve_run_dir(
                    Path(chosen), Path(vars_["video"].get()) if vars_["video"].get() else None
                )
            )
            combo["values"] = [str(value) for value in ids]
            if ids:
                vars_["id"].set(str(ids[0]))
            vars_["status"].set(f"Available IDs: {ids}")
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
        chosen = filedialog.askdirectory(parent=dialog, title="Output directory for selected ID")
        if chosen:
            vars_["output"].set(chosen)

    for row, (label, key, command) in enumerate(
        (
            ("Run directory", "run", browse_run),
            ("Video", "video", browse_video),
            ("Output", "output", browse_output),
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
            output = (
                Path(output_text).expanduser()
                if output_text
                else run_dir / f"visualized_id_{selected:02d}"
            )
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
    parser.add_argument("--id", dest="selected_id", type=int, help="SAM/stable ID to visualize.")
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
    if args.selected_id is None:
        parser.error("--id is required unless --list-ids is used")
    if args.output is None:
        parser.error("--output/-o is required for rendering")
    if args.selected_id not in ids:
        parser.error(f"ID {args.selected_id} is unavailable; choose one of {ids}")
    if args.dry_run:
        print(
            f">> Dry-run OK: video={args.video} run_dir={run_dir} id={args.selected_id} output={args.output}"
        )
        return 0
    visualize_selected_id(
        run_dir,
        args.video,
        args.selected_id,
        args.output,
        kpt_thr=args.kpt_thr,
        draw_all_keypoints=not args.no_all_keypoints,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
