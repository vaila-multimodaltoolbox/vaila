"""
================================================================================
Script: freekiki_ball_train.py - dataset, metrics and training of the ball detector
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.11
Created: 26 October 2026
Update Date: 26 October 2026

Description:
    The ball has its own detector (one YOLO class ``ball``), kept apart from the
    49-point pitch network: ``kiki49`` has no ball annotation. Its labels come
    from the SAM 3 ``sports ball`` track of the match videos and are reviewed by a
    person (accept, reject or correct the box). This module turns the reviewed
    labels into a YOLO dataset, measures a detector against them, and trains it.

    * :func:`read_review_csv` - reviewed labels, one row per frame
      (``video,frame,status,x1,y1,x2,y2``, status ``ball`` or ``none``);
    * :func:`split_by_video` - train/val split by video, so no video is in both;
    * :func:`build_yolo_dataset` - frames as JPG and YOLO labels, with ``data.yaml``;
    * :func:`ball_metrics` - recall, precision and centre error against labels;
    * :func:`train_ball_detector` - Ultralytics detection training (needs CUDA).

    Ultralytics and OpenCV are imported only where they are used.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

REVIEW_COLUMNS = ["video", "frame", "status", "x1", "y1", "x2", "y2"]
BALL_CLASS = "ball"
DEFAULT_BASE = "yolo26m.pt"


def read_review_csv(path: Path | str) -> list[dict]:
    """Reviewed ball labels. ``status`` ``ball`` needs a box; ``none`` marks a frame without a ball."""
    rows: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            status = (row.get("status") or "").strip()
            if status not in ("ball", "none"):
                raise ValueError(f"review status must be 'ball' or 'none', got {status!r}")
            box = [float(row[k]) for k in ("x1", "y1", "x2", "y2")] if status == "ball" else None
            rows.append(
                {"video": row["video"], "frame": int(row["frame"]), "status": status, "box": box}
            )
    return rows


def split_by_video(
    videos: list[str], *, val_fraction: float = 0.2, seed: int = 0
) -> dict[str, str]:
    """Video name -> ``train`` or ``val``, decided by a stable hash so a rerun keeps the split.

    A video is never split between the two sets: the frames of one match stay together.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    names = sorted(set(videos))
    if len(names) < 2:
        raise ValueError("need at least two videos to split into train and val")
    out: dict[str, str] = {}
    for name in names:
        digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        out[name] = "val" if bucket < val_fraction else "train"
    if "val" not in out.values() or "train" not in out.values():
        raise ValueError("the split left one side empty; add more videos or change val_fraction")
    return out


def yolo_ball_line(box, width: int, height: int) -> str:
    """One YOLO label line ``0 cx cy w h`` (normalised, clipped to the image)."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = sorted((min(max(x1, 0.0), width), min(max(x2, 0.0), width)))
    y1, y2 = sorted((min(max(y1, 0.0), height), min(max(y2, 0.0), height)))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        raise ValueError(f"degenerate ball box {box}")
    return (
        f"0 {(x1 + w / 2) / width:.6f} {(y1 + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"
    )


def build_yolo_dataset(
    labels: list[dict],
    video_paths: dict[str, Path | str],
    out_dir: Path | str,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> dict:
    """Write frames and YOLO labels for the reviewed ``labels`` and return the counts.

    ``video_paths`` maps each video name in ``labels`` to its file. A frame with
    status ``none`` gets an empty label file (a negative example).
    """
    import cv2

    out_dir = Path(out_dir)
    split_of = split_by_video(
        [row["video"] for row in labels], val_fraction=val_fraction, seed=seed
    )
    counts = {"train": 0, "val": 0, "positive": 0, "negative": 0}
    by_video: dict[str, dict[int, dict]] = {}
    for row in labels:
        by_video.setdefault(row["video"], {})[row["frame"]] = row

    for video, frames in by_video.items():
        if video not in video_paths:
            raise KeyError(f"no video file for {video}")
        split = split_of[video]
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_paths[video]))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video_paths[video]}")
        try:
            wanted = sorted(frames)
            index = 0
            frame_no = 0
            while index < len(wanted):
                ok, bgr = cap.read()
                if not ok:
                    break
                if frame_no == wanted[index]:
                    row = frames[frame_no]
                    stem = f"{Path(video).stem}_f{frame_no:06d}"
                    height, width = bgr.shape[:2]
                    cv2.imwrite(str(img_dir / f"{stem}.jpg"), bgr)
                    line = yolo_ball_line(row["box"], width, height) if row["box"] else ""
                    (lbl_dir / f"{stem}.txt").write_text(
                        line + ("\n" if line else ""), encoding="utf-8"
                    )
                    counts[split] += 1
                    counts["positive" if line else "negative"] += 1
                    index += 1
                frame_no += 1
        finally:
            cap.release()

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.resolve()}\ntrain: images/train\nval: images/val\n"
        f"names:\n  0: {BALL_CLASS}\n",
        encoding="utf-8",
    )
    return {**counts, "data_yaml": str(data_yaml)}


def ball_metrics(
    predicted: dict[tuple[str, int], tuple[float, float] | None],
    reference: dict[tuple[str, int], tuple[float, float] | None],
    *,
    match_px: float = 10.0,
) -> dict:
    """Detector against reviewed labels, keyed by ``(video, frame)``.

    A frame with a reference ball is a hit when the predicted centre is within
    ``match_px``. A frame with no reference ball counts as a false positive when the
    detector still reports one. ``median_err_px`` is over the hits.
    """
    tp = fn = fp = tn = 0
    errors: list[float] = []
    for key, ref in reference.items():
        pred = predicted.get(key)
        if ref is None:
            if pred is None:
                tn += 1
            else:
                fp += 1
            continue
        if pred is None:
            fn += 1
            continue
        err = float(np.hypot(pred[0] - ref[0], pred[1] - ref[1]))
        if err <= match_px:
            tp += 1
            errors.append(err)
        else:
            fn += 1
            fp += 1
    recall = tp / (tp + fn) if tp + fn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return {
        "frames": len(reference),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": recall,
        "precision": precision,
        "median_err_px": float(np.median(errors)) if errors else float("nan"),
    }


def train_ball_detector(
    data_yaml: Path | str,
    out_dir: Path | str,
    *,
    base: str = DEFAULT_BASE,
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int | float = -1,
    device: str | None = None,
    patience: int = 30,
) -> Path:
    """Train the ball detector with Ultralytics and return the path of ``best.pt``."""
    from ultralytics import YOLO

    model = YOLO(base)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=str(out_dir),
        name="ball",
        exist_ok=True,
        single_cls=True,
    )
    return Path(out_dir) / "ball" / "weights" / "best.pt"
