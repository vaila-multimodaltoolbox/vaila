"""
Project: vailá Multimodal Toolbox
Script: sam_to_yolo.py - SAM 3 tracks -> YOLO detection dataset

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 June 2026
Update Date: 29 June 2026
Version: 0.3.67

Change History:
    - v0.3.65: Added --split-mode {temporal,random}. Default is now temporal
               (chronological train/val/test blocks) so near-duplicate
               consecutive video frames no longer leak between train and val
               and inflate mAP. random preserves the old shuffle behaviour.
               Added --frame-stride N (keep every Nth frame; 1=full) to build a
               smaller, less-redundant labelling set from the SAM3 tracks.
               Frame extraction now uses a single sequential grab/retrieve pass
               (far faster than per-frame random seeks) and prints a banner +
               tqdm progress bar so the terminal always shows what is happening.

Description:
    Convert a SAM 3 video tracking export (``sam_tracks.csv`` / its
    ``sam_bbox_tracks.csv`` alias, or ``sam_frames_meta.csv``) into a
    **YOLO detection** dataset (one bounding box per tracked instance per
    frame, a single ``person`` class by default).

    This is the CORRECT path when the goal is to detect/track N moving
    people/objects with a YOLO detector + BoT-SORT. The tracker (not the
    detector classes) assigns the per-instance IDs 1..N at inference time.

    Why this module exists
    ----------------------
    A common mistake is to load SAM bbox tracks into ``getpixelvideo``,
    convert each bbox to a single marker (anchor), then export with the
    **pose** writer (``export_pose_dataset``, F9). That collapses *all*
    instances of a frame into ONE object whose keypoints are the markers
    (``nc: 1, names: ['object'], kpt_shape: [N, 3]``). A model trained that
    way predicts ONE box + N keypoints, never N separate detections — so at
    tracking time everything shows up as a single ``object`` class.

    For detection/tracking you want, per frame::

        0 cx cy w h      # instance 1 (class 0 = person)
        0 cx cy w h      # instance 2
        ...              # ~N lines, one per SAM track present in the frame

Usage:
    # Build a detection dataset (CLI / headless)
    uv run python -m vaila.sam_to_yolo build \
        --sam-tracks /path/processed_sam_*/<video>/sam_tracks.csv \
        --video /path/<video>.mp4 \
        --class-name person \
        --reuse-images-dir /path/vaila_dataset_*/   # optional, reuse frames
        --output /path/out_dataset                  # optional
    # (a bare `--sam-tracks ...` with no subcommand still implies `build`)

    # Rename the class of an EXISTING dataset (metadata only; labels are
    # index-based so the thousands of .txt files are untouched)
    uv run python -m vaila.sam_to_yolo rename-dataset \
        --dataset /path/out_dataset --class-name person

    # Rename the class baked into a TRAINED .pt (no retraining)
    uv run python -m vaila.sam_to_yolo rename-model \
        --weights best.pt --class-name person --output best_person.pt

    # GUI (Tkinter file pickers, build mode)
    uv run python -m vaila.sam_to_yolo

Requirements:
    - Python 3.12
    - OpenCV (frame extraction fallback)
    - Tkinter (GUI mode only)

License:
    GNU Affero General Public License v3.0
    https://www.gnu.org/licenses/agpl-3.0.html
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    import cv2  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - cv2 always present in vailá runtime
    cv2 = None  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# Terminal feedback helpers
# ---------------------------------------------------------------------------
# NOTE: absl logging (pulled in by mediapipe/opencv) silently eats a leading
# "[bracketed]" prefix from stdout, so all human-facing lines use a ">>"
# prefix instead. tqdm writes to stderr and is unaffected.


def _try_import_tqdm():
    """Return tqdm if available, else None (soft dependency via ultralytics)."""
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        return tqdm
    except ImportError:
        return None


def _banner(title: str, *lines: str) -> None:
    """Print a boxed banner so long operations are visibly underway."""
    print("\n" + "=" * 70)
    print(f">> sam_to_yolo: {title}")
    for line in lines:
        print(f"   {line}")
    print("=" * 70, flush=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _Box:
    """A single normalized YOLO detection box (class id implicit = 0)."""

    cx: float
    cy: float
    w: float
    h: float

    def to_label_line(self, class_id: int = 0) -> str:
        return f"{class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class _BuildStats:
    frames_with_boxes: int = 0
    total_boxes: int = 0
    dropped_boxes: int = 0
    images_linked: int = 0
    images_copied: int = 0
    images_extracted: int = 0
    images_missing: int = 0
    splits: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing SAM exports
# ---------------------------------------------------------------------------

_FRAME_FILE_RE = re.compile(r"(?:^|[_\-])(\d{1,8})\.(?:jpg|jpeg|png|bmp)$", re.IGNORECASE)


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def parse_sam_tracks(
    csv_path: str | Path,
    frame_width: int,
    frame_height: int,
    *,
    min_box_px: float = 1.0,
    min_score: float = 0.0,
) -> dict[int, list[_Box]]:
    """Read ``sam_tracks.csv`` (long format) into ``{frame -> [boxes]}``.

    Expected columns (SAM 3 video export): ``frame,obj_id,x_px,y_px,w_px,h_px,
    score,...,cx_px,cy_px``. ``x_px/y_px`` is the top-left corner; ``cx_px/cy_px``
    the center. Degenerate boxes (``w<=0`` or ``h<=0``) and boxes below
    ``min_score`` are dropped.
    """
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    per_frame: dict[int, list[_Box]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        if "frame" not in cols:
            raise ValueError(f"{csv_path}: missing 'frame' column")
        has_center = "cx_px" in cols and "cy_px" in cols
        has_corner = all(k in cols for k in ("x_px", "y_px", "w_px", "h_px"))
        if not (has_center or has_corner):
            raise ValueError(f"{csv_path}: need cx_px/cy_px or x_px/y_px/w_px/h_px columns")
        for row in reader:
            try:
                frame = int(float(row[cols["frame"]]))
            except (KeyError, ValueError, TypeError):
                continue
            try:
                w_px = float(row[cols["w_px"]]) if "w_px" in cols else 0.0
                h_px = float(row[cols["h_px"]]) if "h_px" in cols else 0.0
            except (ValueError, TypeError):
                continue
            if "score" in cols and min_score > 0.0:
                try:
                    if float(row[cols["score"]]) < min_score:
                        continue
                except (ValueError, TypeError):
                    pass
            if w_px < min_box_px or h_px < min_box_px:
                per_frame.setdefault(frame, [])  # remember frame, drop box
                continue
            if has_center:
                cx_px = float(row[cols["cx_px"]])
                cy_px = float(row[cols["cy_px"]])
            else:
                x_px = float(row[cols["x_px"]])
                y_px = float(row[cols["y_px"]])
                cx_px = x_px + w_px / 2.0
                cy_px = y_px + h_px / 2.0
            box = _Box(
                cx=_clamp01(cx_px / frame_width),
                cy=_clamp01(cy_px / frame_height),
                w=_clamp01(w_px / frame_width),
                h=_clamp01(h_px / frame_height),
            )
            if box.w <= 0.0 or box.h <= 0.0:
                continue
            per_frame.setdefault(frame, []).append(box)
    return per_frame


def probe_video_size(video_path: str | Path) -> tuple[int, int, int]:
    """Return ``(width, height, n_frames)`` for a video via OpenCV."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required to read the video")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return w, h, n


def index_existing_frames(reuse_dir: str | Path) -> dict[int, Path]:
    """Map ``frame_index -> image path`` for ``frame_NNNNNN.*`` under ``reuse_dir``.

    Recursively scans for files whose name encodes a frame number (e.g.
    ``frame_000123.jpg``). Useful to reuse frames already extracted by a prior
    (pose) dataset export instead of re-decoding the video.
    """
    index: dict[int, Path] = {}
    root = Path(reuse_dir)
    if not root.is_dir():
        return index
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        m = _FRAME_FILE_RE.search(path.name)
        if not m:
            continue
        idx = int(m.group(1))
        index.setdefault(idx, path)
    return index


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


def _place_image(src: Path, dst: Path) -> str:
    """Place ``src`` at ``dst`` cheaply: hardlink -> copy -> symlink. Returns mode."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        pass
    try:
        shutil.copy2(src, dst)
        return "copy"
    except OSError:
        os.symlink(os.path.abspath(src), dst)
        return "symlink"


def build_detection_dataset_from_sam_tracks(
    sam_tracks_csv: str | Path,
    video_path: str | Path | None = None,
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    class_name: str = "person",
    output_dir: str | Path | None = None,
    reuse_images_dir: str | Path | None = None,
    split_ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    split_mode: str = "random",
    frame_stride: int = 1,
    image_format: str = "jpg",
    min_box_px: float = 1.0,
    min_score: float = 0.0,
    seed: int = 0,
) -> tuple[str, str, _BuildStats]:
    """Build a YOLO **detection** dataset from a SAM tracks CSV.

    Args:
        sam_tracks_csv: path to ``sam_tracks.csv`` (or ``sam_bbox_tracks.csv``).
        video_path: source video; needed for size auto-detection and frame
            extraction when an image is not found in ``reuse_images_dir``.
        frame_width/frame_height: override video size (skips probing).
        class_name: single detection class (default ``person``).
        output_dir: dataset root; default ``<video_dir>/vaila_dataset_detect_TS``.
        reuse_images_dir: dir with already-extracted ``frame_NNNNNN.*`` images to
            hardlink instead of decoding the video.
        split_ratios: train/val/test ratios (sum ~1.0).
        split_mode: ``random`` (shuffle frames) or ``temporal`` (chronological
            blocks: first frames train, middle val, last test). ``temporal`` is
            recommended for video so near-duplicate consecutive frames do not
            leak between train and val and inflate mAP.
        frame_stride: keep only every Nth frame (by absolute frame index, so
            ``10`` keeps frames 0, 10, 20, ...). ``1`` (default) keeps every
            annotated frame (full extract). Use a larger stride to build a
            smaller, less-redundant labelling set from the SAM3 tracks.
        image_format: ``jpg`` or ``png`` for newly extracted frames.
        min_box_px: drop boxes whose pixel w or h is below this.
        min_score: drop boxes whose ``score`` column is below this.
        seed: deterministic split shuffle seed.

    Returns:
        ``(dataset_dir, message, stats)``.
    """
    sam_tracks_csv = Path(sam_tracks_csv)
    if not sam_tracks_csv.is_file():
        raise FileNotFoundError(sam_tracks_csv)

    if frame_width is None or frame_height is None:
        if video_path is None:
            raise ValueError("Provide video_path or explicit frame_width/height")
        w, h, _ = probe_video_size(video_path)
        frame_width = frame_width or w
        frame_height = frame_height or h

    _banner(
        "Reading SAM tracks",
        f"csv: {sam_tracks_csv}",
        f"frame size: {frame_width}x{frame_height}",
        "Parsing rows and grouping boxes per frame (large CSVs take a few seconds)...",
    )
    per_frame = parse_sam_tracks(
        sam_tracks_csv,
        frame_width,
        frame_height,
        min_box_px=min_box_px,
        min_score=min_score,
    )
    frames_with_boxes = sorted(f for f, boxes in per_frame.items() if boxes)
    if not frames_with_boxes:
        raise ValueError("No valid bounding boxes found in SAM tracks CSV")
    print(
        f">> sam_to_yolo: parsed {len(frames_with_boxes)} frames with boxes from CSV.",
        flush=True,
    )

    frame_stride = max(1, int(frame_stride))
    if frame_stride > 1:
        kept = [f for f in frames_with_boxes if f % frame_stride == 0]
        if not kept:
            # No frame index is divisible by the stride; fall back to taking
            # every Nth annotated frame in order so the user still gets data.
            kept = frames_with_boxes[::frame_stride]
        print(
            f">> sam_to_yolo: frame-stride={frame_stride} -> keeping "
            f"{len(kept)}/{len(frames_with_boxes)} frames (every {frame_stride}th).",
            flush=True,
        )
        frames_with_boxes = kept

    if output_dir is None:
        anchor_dir = Path(video_path).parent if video_path else sam_tracks_csv.parent
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = anchor_dir / f"vaila_dataset_detect_{ts}"
    dataset_dir = Path(output_dir)
    for split in ("train", "val", "test"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    reuse_index = index_existing_frames(reuse_images_dir) if reuse_images_dir else {}

    if split_mode == "temporal":
        # Chronological blocks: first frames -> train, middle -> val, last ->
        # test. Avoids the data leakage of a random split on video, where
        # near-duplicate consecutive frames otherwise land in both train and
        # val and inflate mAP.
        ordered = sorted(frames_with_boxes)
    else:
        rng = random.Random(seed)
        ordered = frames_with_boxes[:]
        rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])
    split_of: dict[int, str] = {}
    for i, frame in enumerate(ordered):
        if i < n_train:
            split_of[frame] = "train"
        elif i < n_train + n_val:
            split_of[frame] = "val"
        else:
            split_of[frame] = "test"

    stats = _BuildStats()
    img_ext = image_format.lower().lstrip(".")
    total_boxes = sum(len(per_frame[f]) for f in frames_with_boxes)
    will_reuse = bool(reuse_index)
    _banner(
        "Building YOLO detection dataset",
        f"output: {dataset_dir}",
        f"frames kept: {len(frames_with_boxes)} | total boxes: {total_boxes}"
        + (f" | frame-stride={frame_stride}" if frame_stride > 1 else ""),
        f"split: {split_mode} (train={n_train} val={n_val} test={n - n_train - n_val})",
        f"class: '{class_name}' | image source: "
        + ("reuse extracted frames (hardlink, fast)" if will_reuse else "sequential video decode"),
        "Writing labels first, then images — watch the progress bars below.",
    )
    tqdm = _try_import_tqdm()

    # 1) Write every label file first (pure disk I/O, fast) and bucket each
    #    frame into "reuse a frame on disk" vs "decode from the video".
    label_iter = (
        tqdm(frames_with_boxes, desc=">> sam_to_yolo labels", unit="frame")
        if tqdm is not None
        else frames_with_boxes
    )
    to_link: list[tuple[int, Path]] = []
    to_extract: list[int] = []
    for frame in label_iter:
        boxes = per_frame[frame]
        split = split_of[frame]
        stem = f"frame_{frame:06d}"
        (dataset_dir / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(b.to_label_line(0) for b in boxes) + "\n",
            encoding="utf-8",
        )
        stats.total_boxes += len(boxes)
        stats.frames_with_boxes += 1
        src = reuse_index.get(frame)
        if src is not None and src.is_file():
            to_link.append((frame, src))
        else:
            to_extract.append(frame)

    # 2) Hardlink reusable frames (near-instant; zero extra disk).
    if to_link:
        link_iter = (
            tqdm(to_link, desc=">> sam_to_yolo link", unit="img") if tqdm is not None else to_link
        )
        for frame, src in link_iter:
            split = split_of[frame]
            stem = f"frame_{frame:06d}"
            dst = dataset_dir / split / "images" / f"{stem}{src.suffix.lower()}"
            mode = _place_image(src, dst)
            if mode == "link":
                stats.images_linked += 1
            elif mode in ("copy", "symlink"):
                stats.images_copied += 1

    # 3) Extract the rest from the video in ONE sequential pass (grab/retrieve),
    #    which is far faster than a per-frame random seek.
    if to_extract:
        if video_path is None or cv2 is None:
            stats.images_missing += len(to_extract)
            print(
                ">> sam_to_yolo: no video/cv2 available; "
                f"{len(to_extract)} frames could not be extracted.",
                flush=True,
            )
        else:
            _extract_frames_sequential(
                video_path,
                to_extract,
                dataset_dir,
                split_of,
                img_ext,
                stats,
                tqdm=tqdm,
            )

    for s in ("train", "val", "test"):
        stats.splits[s] = sum(1 for v in split_of.values() if v == s)

    (dataset_dir / "classes.txt").write_text(class_name + "\n", encoding="utf-8")
    data_yaml = dataset_dir / "data.yaml"
    train_abs = (dataset_dir / "train" / "images").resolve()
    val_abs = (dataset_dir / "val" / "images").resolve()
    test_abs = (dataset_dir / "test" / "images").resolve()
    data_yaml.write_text(
        "# YOLO DETECTION dataset - generated by vailá sam_to_yolo\n"
        "# One bounding box per SAM track per frame; the TRACKER assigns IDs at\n"
        "# inference. Do NOT add kpt_shape here (that would make it a pose set).\n"
        f"path: {dataset_dir.resolve().as_posix()}\n"
        f"train: {train_abs.as_posix()}\n"
        f"val: {val_abs.as_posix()}\n"
        f"test: {test_abs.as_posix()}\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n",
        encoding="utf-8",
    )

    _write_dataset_readme(dataset_dir, sam_tracks_csv, video_path, class_name, stats)

    msg = (
        f"Detection dataset: {stats.frames_with_boxes} frames, "
        f"{stats.total_boxes} boxes (train={stats.splits.get('train', 0)}, "
        f"val={stats.splits.get('val', 0)}, test={stats.splits.get('test', 0)}); "
        f"images linked={stats.images_linked} copied={stats.images_copied} "
        f"extracted={stats.images_extracted} missing={stats.images_missing}"
    )
    _banner(
        "DONE — dataset ready to train",
        msg,
        f"data.yaml: {data_yaml}",
        "Validate:  uv run python -m vaila.yolotrain --data "
        f"{data_yaml} --task auto --model auto --dry-run",
    )
    return str(dataset_dir), msg, stats


def _extract_frames_sequential(
    video_path: str | Path,
    frames: list[int],
    dataset_dir: Path,
    split_of: dict[int, str],
    img_ext: str,
    stats: _BuildStats,
    *,
    tqdm=None,
) -> None:
    """Decode the video once, sequentially, writing only the wanted frames.

    Uses ``cap.grab()`` to advance frame-by-frame cheaply and ``cap.retrieve()``
    only on the frames we actually need. This is dramatically faster than
    ``cap.set(CAP_PROP_POS_FRAMES, ...)`` random seeks (which re-seek to a
    keyframe and re-decode forward for every single frame).
    """
    need = set(frames)
    if not need:
        return
    max_frame = max(need)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        stats.images_missing += len(need)
        print(f">> sam_to_yolo: could not open video {video_path}", flush=True)
        return

    bar = (
        tqdm(total=len(need), desc=">> sam_to_yolo extract", unit="img")
        if tqdm is not None
        else None
    )
    progress_step = max(1, len(need) // 20)
    written = 0
    try:
        idx = 0
        while idx <= max_frame:
            if not cap.grab():
                break
            if idx in need:
                ok, img = cap.retrieve()
                if ok and img is not None:
                    split = split_of[idx]
                    dst = dataset_dir / split / "images" / f"frame_{idx:06d}.{img_ext}"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(dst), img):
                        stats.images_extracted += 1
                    else:
                        stats.images_missing += 1
                else:
                    stats.images_missing += 1
                written += 1
                if bar is not None:
                    bar.update(1)
                elif written % progress_step == 0:
                    pct = 100.0 * written / len(need)
                    print(
                        f">> sam_to_yolo: extracted {written}/{len(need)} frames ({pct:.0f}%)",
                        flush=True,
                    )
                if written >= len(need):
                    break
            idx += 1
        unreached = len(need) - written
        if unreached > 0:
            stats.images_missing += unreached
            print(
                f">> sam_to_yolo: {unreached} requested frames were past the "
                "end of the decodable video.",
                flush=True,
            )
    finally:
        cap.release()
        if bar is not None:
            bar.close()


def _extract_one_frame(
    video_path: str | Path | None,
    frame: int,
    dst: Path,
    cap_holder: list,
) -> bool:
    """Seek+write a single frame from the video. Lazily opens a shared capture."""
    if video_path is None or cv2 is None:
        return False
    cap = cap_holder[0]
    if cap is None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        cap_holder[0] = cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame))
    ok, img = cap.read()
    if not ok or img is None:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), img))


def _write_dataset_readme(
    dataset_dir: Path,
    sam_tracks_csv: Path,
    video_path: str | Path | None,
    class_name: str,
    stats: _BuildStats,
) -> None:
    (dataset_dir / "README_dataset.txt").write_text(
        "vailá SAM 3 tracks -> YOLO DETECTION dataset\n"
        "=============================================\n"
        f"source_tracks : {sam_tracks_csv}\n"
        f"source_video  : {video_path}\n"
        f"class         : {class_name} (nc=1)\n"
        f"frames        : {stats.frames_with_boxes}\n"
        f"boxes         : {stats.total_boxes}\n"
        f"split         : train={stats.splits.get('train', 0)} "
        f"val={stats.splits.get('val', 0)} test={stats.splits.get('test', 0)}\n"
        f"images        : linked={stats.images_linked} copied={stats.images_copied} "
        f"extracted={stats.images_extracted} missing={stats.images_missing}\n"
        "\n"
        "Label format (YOLO detect): `0 cx cy w h` (normalized), many lines/frame.\n"
        "The detector finds N boxes of one class; BoT-SORT/ByteTrack assigns the\n"
        "per-instance IDs 1..N at tracking time. This is the CORRECT layout for\n"
        "tracking 16 players; a pose dataset (kpt_shape) is NOT.\n"
        "\n"
        "Train (task auto-detects to `detect` because there is no kpt_shape):\n"
        f"    uv run python -m vaila.yolotrain --data {(dataset_dir / 'data.yaml').resolve()} \\\n"
        "        --task detect --model yolo26x.pt --epochs 100 --imgsz 1280\n"
        "or via Ultralytics directly:\n"
        f"    yolo detect train data={(dataset_dir / 'data.yaml').resolve()} "
        "model=yolo26x.pt epochs=100 imgsz=1280\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Class-name management (rename without touching index-based labels)
# ---------------------------------------------------------------------------
#
# In a YOLO *detection* dataset the label files store only the class INDEX
# (e.g. ``0``), never the class text. So renaming ``object`` -> ``person`` is a
# pure-metadata edit of ``data.yaml`` (``names:``) and ``classes.txt`` — the
# thousands of label files do not change. After training, the chosen names are
# baked into the checkpoint (``model.names``); renaming there is a small edit of
# the saved weights, so you can flip ``object`` -> ``person`` post-hoc too.


def normalize_class_names(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """Accept a single name or a list; return a clean list of class names."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if "," in text:
            return [part.strip() for part in text.split(",") if part.strip()]
        return [text]
    return [str(item).strip() for item in value if str(item).strip()]


def set_detection_class_names(
    dataset_dir: str | Path,
    names: str | list[str],
) -> str:
    """Rename the classes of an existing YOLO **detection** dataset in place.

    Only ``data.yaml`` (``names:`` / ``nc:``) and ``classes.txt`` are rewritten;
    the index-based label files are untouched. ``names`` may be a single string
    (``"person"``) or a list (multi-class). Refuses pose datasets (``kpt_shape``)
    to avoid masking a structural mistake.

    Returns a status message.
    """
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"No data.yaml in {dataset_dir}")
    name_list = normalize_class_names(names)
    if not name_list:
        raise ValueError("Provide at least one class name")

    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    if any(line.strip().startswith("kpt_shape:") for line in lines):
        raise ValueError(
            "data.yaml has kpt_shape (POSE dataset). Renaming the class will not "
            "make it a detection set — rebuild with sam_to_yolo build instead."
        )

    out: list[str] = []
    saw_nc = saw_names = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("nc:"):
            out.append(f"nc: {len(name_list)}")
            saw_nc = True
        elif stripped.startswith("names:"):
            out.append("names: [" + ", ".join(f"'{n}'" for n in name_list) + "]")
            saw_names = True
        else:
            out.append(line)
    if not saw_nc:
        out.append(f"nc: {len(name_list)}")
    if not saw_names:
        out.append("names: [" + ", ".join(f"'{n}'" for n in name_list) + "]")
    data_yaml.write_text("\n".join(out) + "\n", encoding="utf-8")

    (dataset_dir / "classes.txt").write_text("\n".join(name_list) + "\n", encoding="utf-8")
    return f"Renamed classes to {name_list} in {data_yaml}"


def rename_model_class_names(
    weights: str | Path,
    names: str | list[str],
    output: str | Path | None = None,
) -> str:
    """Rename the class names baked into a trained Ultralytics ``.pt``.

    Changes what the tracker/overlay displays (e.g. ``object`` -> ``person``)
    without retraining. Writes to ``output`` (default: overwrite ``weights``).
    The number of names must match the model's class count.

    Returns a status message.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - ultralytics is a runtime dep
        raise RuntimeError("Ultralytics is required to rename model classes") from exc

    name_list = normalize_class_names(names)
    if not name_list:
        raise ValueError("Provide at least one class name")

    model = YOLO(str(weights))
    current = getattr(model, "names", {}) or {}
    n_current = len(current) if current else 0
    if n_current and len(name_list) != n_current:
        raise ValueError(
            f"Model has {n_current} classes but {len(name_list)} names were given "
            f"({name_list}). Provide exactly {n_current}."
        )
    new_names = dict(enumerate(name_list))
    # Set on both the wrapper and the inner nn.Module so save() persists it.
    inner = getattr(model, "model", None)
    if inner is not None:
        inner.names = new_names
    with contextlib.suppress(Exception):
        model.names = new_names  # ty: ignore[invalid-assignment]

    out_path = Path(output) if output else Path(weights)
    model.save(str(out_path))
    return f"Renamed model classes to {name_list} -> {out_path}"


# ---------------------------------------------------------------------------
# CLI / GUI entry points
# ---------------------------------------------------------------------------

_SUBCOMMANDS = ("build", "rename-dataset", "rename-model")


def _run_cli(argv: list[str]) -> int:
    # Back-compat: a bare ``--sam-tracks ...`` (no subcommand) means ``build``.
    if argv and argv[0] not in _SUBCOMMANDS and not argv[0].startswith("-"):
        pass  # unknown positional; let argparse error out
    elif argv and argv[0].startswith("-"):
        argv = ["build", *argv]

    parser = argparse.ArgumentParser(
        prog="vaila.sam_to_yolo",
        description="SAM 3 tracks -> YOLO detection dataset, and class-name tools.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build a YOLO detection dataset from SAM tracks")
    p_build.add_argument("--sam-tracks", required=True, help="Path to sam_tracks.csv")
    p_build.add_argument("--video", default=None, help="Source video (size + frame fallback)")
    p_build.add_argument("--width", type=int, default=None, help="Frame width override")
    p_build.add_argument("--height", type=int, default=None, help="Frame height override")
    p_build.add_argument("--class-name", default="person", help="Single detection class")
    p_build.add_argument("--output", default=None, help="Output dataset dir")
    p_build.add_argument(
        "--reuse-images-dir",
        default=None,
        help="Dir with frame_NNNNNN.* images to hardlink instead of decoding the video",
    )
    p_build.add_argument(
        "--split-mode",
        default="temporal",
        choices=["temporal", "random"],
        help=(
            "temporal (default): chronological train/val/test blocks (no video "
            "frame leakage). random: shuffle frames (inflated mAP on video)."
        ),
    )
    p_build.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Keep only every Nth frame (by frame index: 10 keeps 0,10,20,...). "
            "1 (default) = full extract. Larger = smaller, less-redundant "
            "labelling set from the SAM3 tracks."
        ),
    )
    p_build.add_argument("--image-format", default="jpg", choices=["jpg", "png"])
    p_build.add_argument("--min-box-px", type=float, default=1.0)
    p_build.add_argument("--min-score", type=float, default=0.0)
    p_build.add_argument("--seed", type=int, default=0)

    p_rd = sub.add_parser(
        "rename-dataset", help="Rename classes of an existing detection dataset (metadata only)"
    )
    p_rd.add_argument("--dataset", required=True, help="Dataset dir containing data.yaml")
    p_rd.add_argument(
        "--class-name", required=True, help="New class name(s); comma-separated for multi-class"
    )

    p_rm = sub.add_parser(
        "rename-model", help="Rename class names baked into a trained .pt (no retrain)"
    )
    p_rm.add_argument("--weights", required=True, help="Trained .pt to relabel")
    p_rm.add_argument(
        "--class-name", required=True, help="New class name(s); comma-separated for multi-class"
    )
    p_rm.add_argument("--output", default=None, help="Output .pt (default: overwrite)")

    args = parser.parse_args(argv)

    if args.command == "rename-dataset":
        print(f">> sam_to_yolo: {set_detection_class_names(args.dataset, args.class_name)}")
        return 0
    if args.command == "rename-model":
        print(
            ">> sam_to_yolo: "
            + rename_model_class_names(args.weights, args.class_name, args.output)
        )
        return 0

    dataset_dir, msg, _stats = build_detection_dataset_from_sam_tracks(
        args.sam_tracks,
        video_path=args.video,
        frame_width=args.width,
        frame_height=args.height,
        class_name=args.class_name,
        output_dir=args.output,
        reuse_images_dir=args.reuse_images_dir,
        split_mode=args.split_mode,
        frame_stride=args.frame_stride,
        image_format=args.image_format,
        min_box_px=args.min_box_px,
        min_score=args.min_score,
        seed=args.seed,
    )
    print(f">> sam_to_yolo: {msg}")
    print(f">> sam_to_yolo: dataset -> {dataset_dir}")
    print(f">> sam_to_yolo: data.yaml -> {Path(dataset_dir) / 'data.yaml'}")
    print(
        ">> sam_to_yolo: to rename later: "
        "python -m vaila.sam_to_yolo rename-dataset --dataset "
        f"{dataset_dir} --class-name person"
    )
    return 0


def run_sam_to_yolo_gui() -> None:
    """Tkinter file pickers -> build detection dataset. GUI entry point."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

    root = tk.Tk()
    root.withdraw()

    sam_tracks = filedialog.askopenfilename(
        title="Select SAM tracks CSV (sam_tracks.csv / sam_bbox_tracks.csv)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not sam_tracks:
        return
    video = filedialog.askopenfilename(
        title="Select the source video (for size + frame extraction)",
        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
    )
    if not video:
        return
    stride = simpledialog.askinteger(
        "Frame stride",
        "Keep every Nth frame:\n"
        "  1  = full extract (all annotated frames)\n"
        " 10 = every 10th frame (smaller labelling set)\n"
        " 20 = every 20th frame, etc.",
        initialvalue=1,
        minvalue=1,
        parent=root,
    )
    if stride is None:
        return
    reuse = filedialog.askdirectory(
        title="(Optional) Dir with already-extracted frames to reuse (Cancel to skip)"
    )

    try:
        dataset_dir, msg, _stats = build_detection_dataset_from_sam_tracks(
            sam_tracks,
            video_path=video,
            class_name="person",
            reuse_images_dir=reuse or None,
            split_mode="temporal",
            frame_stride=stride,
        )
    except Exception as exc:  # noqa: BLE001 - surface to GUI
        messagebox.showerror("sam_to_yolo", f"Failed: {exc}")
        return
    messagebox.showinfo(
        "sam_to_yolo",
        f"{msg}\n\nDataset:\n{dataset_dir}\n\nTrain with yolotrain (task auto -> detect).",
    )


def main() -> int:
    argv = sys.argv[1:]
    if argv:
        return _run_cli(argv)
    run_sam_to_yolo_gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
