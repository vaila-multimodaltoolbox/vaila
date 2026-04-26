"""FIFA Dataset Builder — unified 32-pt soccer-pitch keypoint dataset.

Goal
----
Aggregate, normalise and merge multiple open-source soccer-pitch keypoint
datasets into a single YOLO Pose dataset compatible with the
``vaila.soccerfield_keypoints_ai`` model
(``models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt``) and with the
HF dataset ``martinjolif/football-pitch-detection`` (``kpt_shape: [32, 3]``).

The canonical 32-keypoint order matches the one used by the trained vailá
model — i.e. the Roboflow ``football-field-detection-f07vi`` schema.  The
semantic names below come from ``vaila/models/hf_datasets/football-pitch-detection/pitch_keypoints.png``
(labels 01..32 in the image are stored as ``kp_00..kp_31`` in the YOLO files,
so ``kp_i = label_{i+1}`` in the figure).

Output layout (``--out-root``)
-----------------------------
::

    dataset_vaila_fifa/
    ├── sources/                        # raw downloads, untouched
    │   ├── martinjolif_football-pitch-detection/
    │   ├── Adit-jain_Soccana_Keypoint_detection_v1/
    │   ├── soccernet_calibration_2023/  # via SoccerNet PyPI (calibration-2023)
    │   └── PiotrGrabysz_PitchGeometry/
    ├── staging/                        # per-source 32-pt YOLO Pose conversion
    │   └── <source_name>/{images,labels}/
    ├── unified/                        # final merged dataset
    │   ├── images/{train,val,test}/
    │   ├── labels/{train,val,test}/
    │   ├── data.yaml                   # ready for ``yolo pose train``
    │   └── manifest.csv                # provenance per image
    └── reports/build_<TIMESTAMP>.log

Run modes
---------
* CLI: ``uv run python -m vaila.fifa_dataset_builder --out-root /path``
* GUI: ``uv run python -m vaila.fifa_dataset_builder`` (Tkinter dialog)

Optional credentials are taken from the environment so they can stay out of
shell history:

* ``HF_TOKEN`` (or ``HUGGING_FACE_HUB_TOKEN``)
* ``ROBOFLOW_API_KEY``
* ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (or ``~/.kaggle/kaggle.json``)

SoccerNet calibration
---------------------
The ``soccernet_calibration_2023`` source uses the official ``SoccerNet`` PyPI
package — it does NOT require an NDA password (unlike the broadcast videos).
Install via ``uv add SoccerNet`` (or ``--dev``), then run with
``--include-soccernet`` (or via the GUI checkbox).  The downloader writes
~22 800 images plus their pitch-line JSONs into
``sources/soccernet_calibration_2023/calibration-2023/{train,valid,test,challenge}/``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical 32-keypoint schema (Roboflow ``football-field-detection-f07vi``)
# ---------------------------------------------------------------------------

# Index here is 0-based to match YOLO Pose ordering.  The numeric label in
# ``vaila/models/hf_datasets/football-pitch-detection/pitch_keypoints.png`` is
# 1-based (i.e. ``kp_00`` corresponds to label ``01`` on the image).
CANONICAL_KP_NAMES_32: tuple[str, ...] = (
    "top_left_corner",  # 01
    "left_penalty_box_top_left",  # 02
    "left_goal_area_top_left",  # 03
    "left_goal_area_bottom_left",  # 04
    "left_penalty_box_bottom_left",  # 05
    "bottom_left_corner",  # 06
    "left_goal_area_top_right",  # 07
    "left_goal_area_bottom_right",  # 08
    "left_penalty_spot",  # 09
    "left_penalty_box_top_right",  # 10
    "left_penalty_arc_top_intersection",  # 11
    "left_penalty_arc_bottom_intersection",  # 12
    "left_penalty_box_bottom_right",  # 13
    "center_circle_left",  # 14
    "midfield_top",  # 15
    "center_circle_top",  # 16
    "center_circle_bottom",  # 17
    "midfield_bottom",  # 18
    "center_circle_right",  # 19
    "right_penalty_box_top_left",  # 20
    "right_penalty_arc_top_intersection",  # 21
    "right_penalty_arc_bottom_intersection",  # 22
    "right_penalty_box_bottom_left",  # 23
    "right_penalty_spot",  # 24
    "right_goal_area_top_left",  # 25
    "right_goal_area_bottom_left",  # 26
    "top_right_corner",  # 27
    "right_penalty_box_top_right",  # 28
    "right_goal_area_top_right",  # 29
    "right_goal_area_bottom_right",  # 30
    "right_penalty_box_bottom_right",  # 31
    "bottom_right_corner",  # 32
)
NUM_KEYPOINTS = 32

# Horizontal-flip permutation, matching the ``flip_idx`` published by the
# canonical HF dataset.
CANONICAL_FLIP_IDX_32: tuple[int, ...] = (
    24,
    25,
    26,
    27,
    28,
    29,
    22,
    23,
    21,
    17,
    18,
    19,
    20,
    13,
    14,
    15,
    16,
    9,
    10,
    11,
    12,
    8,
    6,
    7,
    0,
    1,
    2,
    3,
    4,
    5,
    31,
    30,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _link_or_copy(src: Path, dst: Path, *, copy_fallback: bool = True) -> str:
    """Symlink ``dst`` -> ``src``; copy when symlinks fail.

    Returns ``"symlink"`` or ``"copy"`` depending on what was performed.
    """
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except IsADirectoryError:
            shutil.rmtree(dst)
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError):
        if not copy_fallback:
            raise
        shutil.copy2(src, dst)
        return "copy"


def _all_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    out: list[Path] = []
    for ext in IMAGE_EXTS:
        out.extend(folder.rglob(f"*{ext}"))
    out.sort()
    return out


def _read_yolo_pose_label(label_path: Path) -> list[list[float]] | None:
    """Read a YOLO Pose ``.txt`` file as a list of float vectors (one per row)."""
    if not label_path.exists():
        return None
    rows: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return rows


def _format_yolo_pose_row(
    cls: int,
    bbox_xywhn: tuple[float, float, float, float],
    keypoints_xyv: list[tuple[float, float, float]],
) -> str:
    """Serialise a (class, bbox, keypoints) tuple back to a YOLO Pose line."""
    cx, cy, w, h = bbox_xywhn
    kp_str = " ".join(f"{x:.6f} {y:.6f} {int(round(v))}" for x, y, v in keypoints_xyv)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp_str}"


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


@dataclass
class DatasetSource:
    """Description of one external dataset that we know how to ingest."""

    name: str  # short, filesystem-safe id
    kind: str  # "hf_dataset" | "github" | "roboflow_universe" | "kaggle"
    url: str  # repo id, GitHub URL, etc.
    converter: str  # name of the converter function below
    licence: str = ""
    notes: str = ""
    optional: bool = False  # True => only run when explicitly requested
    requires_env: tuple[str, ...] = ()  # required env vars for download


# A source is "available" if all required env vars are set OR no env vars are
# needed (e.g. public HF dataset).  Optional ones still need to be requested
# explicitly via ``--include`` / ``--include-optional``.
REGISTRY: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="martinjolif_football-pitch-detection",
        kind="hf_dataset",
        url="martinjolif/football-pitch-detection",
        converter="convert_yolo_pose_passthrough",
        licence="CC-BY 4.0",
        notes=(
            "Canonical 32-pt seed (Roboflow football-field-detection-f07vi-d0ele); "
            "already in YOLO Pose format with the right kp ordering."
        ),
    ),
    DatasetSource(
        name="Adit-jain_Soccana_Keypoint_detection_v1",
        kind="hf_dataset",
        url="Adit-jain/Soccana_Keypoint_detection_v1",
        converter="convert_soccana_29pt",
        licence="see Hugging Face page",
        notes=(
            "29-pt JSON+YOLO labels derived from SoccerNet calibration; the "
            "source images are NOT in this dataset and must be paired with "
            "the SoccerNet calibration-2023 download (see soccernet_calibration_2023)."
        ),
    ),
    DatasetSource(
        name="soccernet_calibration_2023",
        kind="soccernet",
        url="calibration-2023",
        converter="convert_soccernet_images_only",
        licence="SoccerNet (no NDA needed for this task)",
        notes=(
            "~22 800 broadcast frames with line-endpoint JSONs.  Used as the "
            "image source paired with the Soccana 29-pt label JSONs.  Requires "
            "the 'SoccerNet' PyPI package: `uv add SoccerNet`."
        ),
        optional=True,
        requires_env=(),
    ),
    DatasetSource(
        name="PiotrGrabysz_PitchGeometry",
        kind="github",
        url="https://github.com/PiotrGrabysz/PitchGeometry.git",
        converter="convert_pitchgeometry_csv",
        licence="see GitHub repository",
        notes=(
            "Repo ships test-fixture images only (~5 frames). Full dataset is "
            "not redistributed; converter falls back to the tests/dataset/ samples."
        ),
        optional=True,
    ),
    DatasetSource(
        name="hamzaboulahia_kaggle_landmarks",
        kind="kaggle",
        url="hamzaboulahia/football-field-keypoints-dataset",
        converter="convert_kaggle_bbox_landmarks",
        licence="see Kaggle page",
        notes=(
            "342 high-angle frames with 28 BBoxes per image; we extract centroids "
            "and map class names to canonical 32-pt slots."
        ),
        optional=True,
        requires_env=("KAGGLE_USERNAME", "KAGGLE_KEY"),
    ),
    DatasetSource(
        name="roboflow_woudenberg_keypoints-on-soccer-pitch",
        kind="roboflow_universe",
        url="wim-van-woudenberg-bbs75/keypoints-on-soccer-pitch",
        converter="convert_roboflow_yolo_pose",
        licence="see Roboflow workspace",
        notes="Public Roboflow Universe dataset; needs ROBOFLOW_API_KEY to download.",
        optional=True,
        requires_env=("ROBOFLOW_API_KEY",),
    ),
)


def _source_available(src: DatasetSource) -> tuple[bool, str]:
    """Decide whether a registry entry can be downloaded right now."""
    missing = [v for v in src.requires_env if not os.environ.get(v)]
    if missing:
        return False, f"missing env: {', '.join(missing)}"
    return True, ""


# ---------------------------------------------------------------------------
# Download backends
# ---------------------------------------------------------------------------


def _download_hf_dataset(repo_id: str, dst: Path, *, hf_token: str | None) -> None:
    """Snapshot a Hugging Face dataset repository into ``dst``.

    Uses :func:`huggingface_hub.snapshot_download` so that it benefits from
    the local HF cache and resumable transfers.
    """
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    _ensure_dir(dst)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dst),
        token=hf_token,
        # Keep symlinks so the cache is reused; the consumer code reads through
        # them transparently.
    )


def _download_github(url: str, dst: Path, *, branch: str | None = None) -> None:
    """Shallow-clone a GitHub repository into ``dst``."""
    if dst.exists() and (dst / ".git").exists():
        # already cloned -> just refresh
        with contextlib.suppress(subprocess.CalledProcessError):
            subprocess.run(
                ["git", "-C", str(dst), "fetch", "--depth", "1", "origin"],
                check=True,
                capture_output=True,
            )
        return
    _ensure_dir(dst.parent)
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, str(dst)]
    subprocess.run(cmd, check=True)


def _download_roboflow_universe(url: str, dst: Path, *, api_key: str, version: int = 1) -> None:
    """Download a Roboflow Universe project as YOLO v8 Pose.

    ``url`` is expected as ``<workspace>/<project>``; we pick the latest
    version when possible. We use the Roboflow Inference HTTP endpoint so this
    works without the ``roboflow`` PyPI package (which we don't add by
    default).
    """
    workspace, _, project = url.partition("/")
    if not workspace or not project:
        raise ValueError(f"invalid roboflow project '{url}'; expected 'workspace/project'")
    _ensure_dir(dst)
    fmt = "yolov8"
    api_url = (
        f"https://api.roboflow.com/{workspace}/{project}/{version}"
        f"/{fmt}?api_key={urllib.parse.quote(api_key)}"
    )
    # Use urllib so we don't add a hard requests dependency here.
    with urllib.request.urlopen(api_url, timeout=120) as resp:
        meta = json.loads(resp.read().decode("utf-8"))
    export_url = meta.get("export", {}).get("link")
    if not export_url:
        raise RuntimeError(f"Roboflow API did not return an export link: {meta}")
    archive = dst / "export.zip"
    urllib.request.urlretrieve(export_url, archive)
    # Extract using stdlib so we don't need 7z.
    import zipfile

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dst)
    archive.unlink(missing_ok=True)


def _download_kaggle_dataset(slug: str, dst: Path) -> None:
    """Download a Kaggle dataset using the ``kaggle`` CLI if available."""
    _ensure_dir(dst)
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise RuntimeError(
            "kaggle CLI not found. Install with `uv add --dev kaggle` or place "
            "kaggle.json in ~/.kaggle/, then run `kaggle datasets download` manually."
        )
    subprocess.run(
        [kaggle, "datasets", "download", "-d", slug, "-p", str(dst), "--unzip"],
        check=True,
    )


def _download_soccernet_calibration(
    dst: Path,
    *,
    splits: tuple[str, ...] = ("train", "valid", "test", "challenge"),
) -> None:
    """Download SoccerNet ``calibration-2023`` images + line JSONs.

    Uses the official ``SoccerNet`` PyPI package (no NDA password required for
    this task).  Pip-install with ``uv add SoccerNet`` if missing.
    """
    try:
        from SoccerNet.Downloader import (  # type: ignore[import-not-found]
            SoccerNetDownloader,
        )
    except ModuleNotFoundError as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "The 'SoccerNet' package is required for the soccernet_calibration_2023 "
            "source. Install with: uv add SoccerNet"
        ) from e
    _ensure_dir(dst)
    snd = SoccerNetDownloader(LocalDirectory=str(dst))
    snd.downloadDataTask(task="calibration-2023", split=list(splits))
    _extract_soccernet_zips(dst, splits=splits)


def _extract_soccernet_zips(
    soccernet_root: Path,
    *,
    splits: tuple[str, ...] = ("train", "valid", "test", "challenge"),
) -> None:
    """Extract any ``<split>.zip`` files inside ``calibration-2023/`` in place.

    The official SoccerNet downloader leaves zips untouched; the inner archive
    already contains a top-level ``<split>/`` directory, so we extract straight
    into ``calibration-2023/``.  Idempotent: skips a split whose extracted dir
    already contains files.
    """
    calib_dir = soccernet_root / "calibration-2023"
    if not calib_dir.is_dir():
        return
    for split in splits:
        zip_path = calib_dir / f"{split}.zip"
        out_dir = calib_dir / split
        if not zip_path.exists():
            continue
        if out_dir.exists() and any(out_dir.iterdir()):
            continue  # already extracted
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(calib_dir)


# ---------------------------------------------------------------------------
# Converters: each takes (downloaded_path, staging_dir) and returns stats dict
# ---------------------------------------------------------------------------


def _make_canonical_label_text(
    bbox_xywhn: tuple[float, float, float, float] | None,
    kps_xyv: list[tuple[float, float, float]],
    cls: int = 0,
) -> str:
    """Pad / clip a list of (x, y, v) keypoints to the canonical 32 size."""
    if len(kps_xyv) != NUM_KEYPOINTS:
        # pad / truncate
        out = list(kps_xyv[:NUM_KEYPOINTS])
        while len(out) < NUM_KEYPOINTS:
            out.append((0.0, 0.0, 0))
        kps_xyv = out
    if bbox_xywhn is None:
        # derive bbox from visible keypoints
        visible = [(x, y) for (x, y, v) in kps_xyv if v > 0 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0]
        if visible:
            xs = [x for (x, _) in visible]
            ys = [y for (_, y) in visible]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            w = max(1e-3, x1 - x0)
            h = max(1e-3, y1 - y0)
        else:
            cx, cy, w, h = 0.5, 0.5, 1.0, 1.0
        bbox_xywhn = (cx, cy, w, h)
    return _format_yolo_pose_row(cls, bbox_xywhn, kps_xyv)


def convert_yolo_pose_passthrough(src_root: Path, dst_root: Path) -> dict[str, int]:
    """Pass-through converter: source is *already* 32-pt YOLO Pose.

    Used for ``martinjolif/football-pitch-detection`` and any compatible
    Roboflow export.  The function looks for ``data/{train,valid,test}/{images,labels}``
    structure (the de-facto Roboflow layout) and replicates it under ``dst_root``.
    """
    stats = {"images": 0, "labels": 0, "skipped": 0}
    for split_in, split_out in (("train", "train"), ("valid", "val"), ("test", "test")):
        img_dir_candidates = [
            src_root / "data" / split_in / "images",
            src_root / split_in / "images",
        ]
        lbl_dir_candidates = [
            src_root / "data" / split_in / "labels",
            src_root / split_in / "labels",
        ]
        img_dir = next((p for p in img_dir_candidates if p.is_dir()), None)
        lbl_dir = next((p for p in lbl_dir_candidates if p.is_dir()), None)
        if img_dir is None or lbl_dir is None:
            continue
        for img_path in _all_image_files(img_dir):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            rows = _read_yolo_pose_label(label_path)
            if rows is None or not rows:
                stats["skipped"] += 1
                continue
            # Validate keypoint count: 1 cls + 4 bbox + 32 * 3 = 101 fields.
            row = rows[0]
            if len(row) != 1 + 4 + NUM_KEYPOINTS * 3:
                # try to coerce: derive bbox if shorter; pad keypoints.
                cls = int(row[0]) if row else 0
                if len(row) >= 5:
                    bbox = (row[1], row[2], row[3], row[4])
                    kp_floats = row[5:]
                else:
                    bbox = None
                    kp_floats = row[1:]
                triplets: list[tuple[float, float, float]] = []
                for i in range(0, len(kp_floats), 3):
                    triplet = kp_floats[i : i + 3]
                    if len(triplet) < 3:
                        triplet = triplet + [0.0] * (3 - len(triplet))
                    triplets.append((triplet[0], triplet[1], triplet[2]))
                line = _make_canonical_label_text(bbox, triplets, cls=cls)
            else:
                line = " ".join(
                    str(int(row[0]))
                    if i == 0
                    else (f"{row[i]:.6f}" if (i - 1) % 3 != 2 or i < 5 else str(int(round(row[i]))))
                    for i in range(len(row))
                )
            out_img_dir = _ensure_dir(dst_root / "images" / split_out)
            out_lbl_dir = _ensure_dir(dst_root / "labels" / split_out)
            out_img = out_img_dir / img_path.name
            out_lbl = out_lbl_dir / f"{img_path.stem}.txt"
            _link_or_copy(img_path, out_img)
            out_lbl.write_text(line + "\n", encoding="utf-8")
            stats["images"] += 1
            stats["labels"] += 1
    return stats


# ---------------------------------------------------------------------------
# Soccana 29-pt → canonical 32-pt mapping
# ---------------------------------------------------------------------------
#
# The Soccana JSON keypoint names follow ``<index>_<semantic_name>`` with
# ``<index> ∈ 0..28``. Convention from upstream LineIntersectionCalculator
# (verified empirically across train/valid/test):
#
#   * pt1 = OUTER endpoint (on the goal line)
#   * pt2 = INNER endpoint (toward the centre)
#
# So the LEFT penalty box (``big_rect_left``) has its outer-NW corner at
# ``1_big_rect_left_top_pt1`` and its inner-NE corner at
# ``2_big_rect_left_top_pt2``; same for bottom edges and small (5.5 m) box.
SOCCANA_INDEX_TO_CANONICAL: dict[int, int] = {
    0: 0,  # sideline_top_left            -> top_left_corner
    1: 1,  # big_rect_left_top_pt1   (out) -> left_penalty_box_top_left
    2: 9,  # big_rect_left_top_pt2   (in)  -> left_penalty_box_top_right
    3: 4,  # big_rect_left_bottom_pt1 (out)-> left_penalty_box_bottom_left
    4: 12,  # big_rect_left_bottom_pt2 (in) -> left_penalty_box_bottom_right
    5: 2,  # small_rect_left_top_pt1 (out) -> left_goal_area_top_left
    6: 6,  # small_rect_left_top_pt2 (in)  -> left_goal_area_top_right
    7: 3,  # small_rect_left_bottom_pt1 (out) -> left_goal_area_bottom_left
    8: 7,  # small_rect_left_bottom_pt2 (in)  -> left_goal_area_bottom_right
    9: 5,  # sideline_bottom_left         -> bottom_left_corner
    10: -1,  # left_semicircle_right       (apex; no canonical match)
    11: 14,  # center_line_top              -> midfield_top
    12: 17,  # center_line_bottom           -> midfield_bottom
    13: 15,  # center_circle_top            -> center_circle_top
    14: 16,  # center_circle_bottom         -> center_circle_bottom
    15: -1,  # field_center                 (no canonical match)
    16: 26,  # sideline_top_right           -> top_right_corner
    17: 27,  # big_rect_right_top_pt1  (out) -> right_penalty_box_top_right
    18: 19,  # big_rect_right_top_pt2  (in)  -> right_penalty_box_top_left
    19: 30,  # big_rect_right_bottom_pt1 (out) -> right_penalty_box_bottom_right
    20: 22,  # big_rect_right_bottom_pt2 (in)  -> right_penalty_box_bottom_left
    21: 28,  # small_rect_right_top_pt1 (out) -> right_goal_area_top_right
    22: 24,  # small_rect_right_top_pt2 (in)  -> right_goal_area_top_left
    23: 29,  # small_rect_right_bottom_pt1 (out) -> right_goal_area_bottom_right
    24: 25,  # small_rect_right_bottom_pt2 (in)  -> right_goal_area_bottom_left
    25: 31,  # sideline_bottom_right        -> bottom_right_corner
    26: -1,  # right_semicircle_left        (apex; no canonical match)
    27: 13,  # center_circle_left           -> center_circle_left
    28: 18,  # center_circle_right          -> center_circle_right
}

# Backwards-compatible name alias used in earlier API.  ``-1`` means "skip".
SOCCANA_29_TO_CANONICAL: dict[str, int] = {
    "sideline_top_left": 0,
    "big_rect_left_top_pt1": 1,
    "big_rect_left_top_pt2": 9,
    "big_rect_left_bottom_pt1": 4,
    "big_rect_left_bottom_pt2": 12,
    "small_rect_left_top_pt1": 2,
    "small_rect_left_top_pt2": 6,
    "small_rect_left_bottom_pt1": 3,
    "small_rect_left_bottom_pt2": 7,
    "sideline_bottom_left": 5,
    "center_line_top": 14,
    "center_line_bottom": 17,
    "center_circle_top": 15,
    "center_circle_bottom": 16,
    "sideline_top_right": 26,
    "big_rect_right_top_pt1": 27,
    "big_rect_right_top_pt2": 19,
    "big_rect_right_bottom_pt1": 30,
    "big_rect_right_bottom_pt2": 22,
    "small_rect_right_top_pt1": 28,
    "small_rect_right_top_pt2": 24,
    "small_rect_right_bottom_pt1": 29,
    "small_rect_right_bottom_pt2": 25,
    "sideline_bottom_right": 31,
    "center_circle_left": 13,
    "center_circle_right": 18,
}


def _strip_soccana_kp_name(raw: str) -> str:
    """``"17_big_rect_right_top_pt1"`` -> ``"big_rect_right_top_pt1"``."""
    if "_" in raw and raw.split("_", 1)[0].isdigit():
        return raw.split("_", 1)[1]
    return raw


def _extract_soccana_zip_if_needed(src_root: Path) -> Path:
    """Auto-extract ``dataset_labels.zip`` to ``src_root/extracted/`` once."""
    zip_path = src_root / "dataset_labels.zip"
    extracted = src_root / "extracted"
    if zip_path.exists() and not extracted.exists():
        import zipfile

        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)
    return extracted if extracted.exists() else src_root


def _index_soccernet_images(soccernet_root: Path | None) -> dict[str, Path]:
    """Map ``"00000"`` -> ``Path/.../00000.jpg`` for every SoccerNet calib image.

    Looks in the standard ``calibration-2023/{train,valid,test,challenge}/``
    layout and falls back to a recursive search for ``[0-9]+.jpg`` files.
    """
    out: dict[str, Path] = {}
    if soccernet_root is None or not soccernet_root.exists():
        return out
    for split in ("train", "valid", "test", "challenge"):
        for base in (
            soccernet_root / "calibration-2023" / split,
            soccernet_root / split,
            soccernet_root / "calibration" / split,
        ):
            if not base.is_dir():
                continue
            for ext in IMAGE_EXTS:
                for p in base.rglob(f"*{ext}"):
                    out.setdefault(p.stem, p)
    if not out:
        for ext in IMAGE_EXTS:
            for p in soccernet_root.rglob(f"*{ext}"):
                out.setdefault(p.stem, p)
    return out


def convert_soccana_29pt(
    src_root: Path,
    dst_root: Path,
    *,
    soccernet_images: dict[str, Path] | None = None,
) -> dict[str, int]:
    """Convert ``Adit-jain/Soccana_Keypoint_detection_v1`` to 32-pt YOLO Pose.

    The HF dataset only ships *labels* (29-pt YOLO + JSON with named pts).
    Source images live in SoccerNet Calibration; pass ``soccernet_images`` to
    pair them by file stem.  When neither images nor JSONs are available the
    function falls back to the bundled ``00*_annotated.jpg`` previews and
    reports them in ``stats["fallback_visualisations"]`` so the run is still
    visible end-to-end.
    """
    stats = {
        "images": 0,
        "labels": 0,
        "skipped_no_image": 0,
        "skipped_low_visible": 0,
        "kp_unmapped": 0,
        "fallback_visualisations": 0,
    }

    extracted = _extract_soccana_zip_if_needed(src_root)
    json_root = extracted / "annotations_json"
    yolo_root = extracted / "yolo_labels"

    # When no JSON is found, fall back to the legacy YOLO-only flow on the
    # raw download; nothing else to do then.
    if not json_root.exists():
        return stats

    img_index: dict[str, Path] = dict(soccernet_images or {})
    # Bundled preview JPGs are named e.g. ``00024_annotated.jpg`` -> stem ``00024``.
    for preview in src_root.glob("*_annotated.jpg"):
        stem = preview.stem.removesuffix("_annotated")
        img_index.setdefault(stem, preview)

    for split_in, split_out in (("train", "train"), ("valid", "val"), ("test", "test")):
        json_dir = json_root / split_in
        if not json_dir.is_dir():
            continue
        yolo_dir = yolo_root / split_in if yolo_root.exists() else None
        for j in sorted(json_dir.glob("*.json")):
            stem = j.stem
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                stats["skipped_no_image"] += 1
                continue
            kps = meta.get("keypoints") or {}
            if not kps:
                stats["skipped_low_visible"] += 1
                continue
            canonical: list[tuple[float, float, float]] = [(0.0, 0.0, 0)] * NUM_KEYPOINTS
            n_visible = 0
            for raw_name, xy in kps.items():
                stripped = _strip_soccana_kp_name(raw_name)
                target = SOCCANA_29_TO_CANONICAL.get(stripped, -1)
                if target < 0:
                    stats["kp_unmapped"] += 1
                    continue
                try:
                    x = float(xy[0])
                    y = float(xy[1])
                except (IndexError, TypeError, ValueError):
                    continue
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    continue
                canonical[target] = (x, y, 2)
                n_visible += 1
            if n_visible < 4:
                stats["skipped_low_visible"] += 1
                continue
            img_path = img_index.get(stem)
            from_preview = False
            if img_path is None and yolo_dir is not None and (yolo_dir / f"{stem}.txt").exists():
                # No real image available — keep label text, but skip the row;
                # we record the count so the user knows how many remained
                # un-paired.
                stats["skipped_no_image"] += 1
                continue
            if img_path is None:
                stats["skipped_no_image"] += 1
                continue
            from_preview = img_path.name.endswith("_annotated.jpg")
            if from_preview:
                stats["fallback_visualisations"] += 1
            # Optional: prefer the bbox computed by upstream when available.
            pitch = meta.get("pitch_object") or {}
            if {"center_x", "center_y", "width", "height"} <= pitch.keys():
                bbox = (
                    float(pitch["center_x"]),
                    float(pitch["center_y"]),
                    float(pitch["width"]),
                    float(pitch["height"]),
                )
            else:
                bbox = None
            line = _make_canonical_label_text(bbox, canonical, cls=0)
            out_img_dir = _ensure_dir(dst_root / "images" / split_out)
            out_lbl_dir = _ensure_dir(dst_root / "labels" / split_out)
            _link_or_copy(img_path, out_img_dir / f"{stem}{img_path.suffix.lower()}")
            (out_lbl_dir / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")
            stats["images"] += 1
            stats["labels"] += 1
    return stats


# Mapping kept for backwards compatibility with earlier callers / tests that
# imported the legacy positional constant.
_SOCCANA_LEGACY_POSITIONAL_29 = tuple(SOCCANA_INDEX_TO_CANONICAL.get(i, -1) for i in range(29))


# Mapping from the PitchGeometry CSV ``kid`` field (semantic id) to canonical
# 32-pt index.  Names are the ones used in
# https://github.com/PiotrGrabysz/PitchGeometry/blob/main/keypoints.md (paraphrased).
PITCHGEOMETRY_NAME_TO_CANONICAL: dict[str, int] = {
    "TOP_LEFT_CORNER": 0,
    "TOP_RIGHT_CORNER": 26,
    "BOTTOM_LEFT_CORNER": 5,
    "BOTTOM_RIGHT_CORNER": 31,
    "MIDFIELD_TOP": 14,
    "MIDFIELD_BOTTOM": 17,
    "CENTER_CIRCLE_TOP": 15,
    "CENTER_CIRCLE_BOTTOM": 16,
    "CENTER_CIRCLE_LEFT": 13,
    "CENTER_CIRCLE_RIGHT": 18,
    "LEFT_PENALTY_AREA_TOP_LEFT": 1,
    "LEFT_PENALTY_AREA_TOP_RIGHT": 9,
    "LEFT_PENALTY_AREA_BOTTOM_LEFT": 4,
    "LEFT_PENALTY_AREA_BOTTOM_RIGHT": 12,
    "LEFT_GOAL_AREA_TOP_LEFT": 2,
    "LEFT_GOAL_AREA_TOP_RIGHT": 6,
    "LEFT_GOAL_AREA_BOTTOM_LEFT": 3,
    "LEFT_GOAL_AREA_BOTTOM_RIGHT": 7,
    "LEFT_PENALTY_SPOT": 8,
    "LEFT_PENALTY_ARC_TOP": 10,
    "LEFT_PENALTY_ARC_BOTTOM": 11,
    "RIGHT_PENALTY_AREA_TOP_LEFT": 19,
    "RIGHT_PENALTY_AREA_TOP_RIGHT": 27,
    "RIGHT_PENALTY_AREA_BOTTOM_LEFT": 22,
    "RIGHT_PENALTY_AREA_BOTTOM_RIGHT": 30,
    "RIGHT_GOAL_AREA_TOP_LEFT": 24,
    "RIGHT_GOAL_AREA_TOP_RIGHT": 28,
    "RIGHT_GOAL_AREA_BOTTOM_LEFT": 25,
    "RIGHT_GOAL_AREA_BOTTOM_RIGHT": 29,
    "RIGHT_PENALTY_SPOT": 23,
    "RIGHT_PENALTY_ARC_TOP": 20,
    "RIGHT_PENALTY_ARC_BOTTOM": 21,
}


def convert_pitchgeometry_csv(src_root: Path, dst_root: Path) -> dict[str, int]:
    """Convert PiotrGrabysz/PitchGeometry CSV format to 32-pt YOLO Pose.

    The repository ships images in ``dataset/{train,val}/images`` and CSV
    annotations in ``dataset/{train,val}/labels.csv`` with columns
    ``frame, kid, x, y, vis`` (long format, normalised).  We aggregate by
    frame, slot each row into the canonical position, and write a YOLO Pose
    label.

    If the on-disk layout is different (the upstream README has been refactored
    multiple times), the function falls back to a best-effort search.
    """
    stats = {"images": 0, "labels": 0, "skipped": 0}
    # Try a few common layouts
    candidate_csvs: list[Path] = []
    for split in ("train", "val", "valid", "test"):
        for sub in ("dataset", "data", ""):
            base = src_root / sub / split if sub else src_root / split
            csv_path = base / "labels.csv"
            if csv_path.exists():
                candidate_csvs.append(csv_path)
            else:
                # Legacy layout: a single annotations.csv at root with a `split` column.
                for alt in (base / "annotations.csv", base.parent / "annotations.csv"):
                    if alt.exists() and alt not in candidate_csvs:
                        candidate_csvs.append(alt)
    if not candidate_csvs:
        # Most lenient search.
        candidate_csvs = list(src_root.rglob("annotations.csv")) + list(
            src_root.rglob("labels.csv")
        )

    for csv_path in candidate_csvs:
        # Determine split from path or column.
        split_hint = (
            "train"
            if "train" in csv_path.parts
            else ("val" if any(p in csv_path.parts for p in ("val", "valid")) else "test")
        )
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            grouped: dict[str, list[dict[str, str]]] = {}
            for row in reader:
                frame_key = row.get("frame") or row.get("filename") or row.get("image")
                if not frame_key:
                    continue
                grouped.setdefault(frame_key, []).append(row)
        # Build labels per frame
        for frame_key, rows in grouped.items():
            canonical: list[tuple[float, float, float]] = [(0.0, 0.0, 0)] * NUM_KEYPOINTS
            n_visible = 0
            for r in rows:
                kid = (r.get("kid") or r.get("name") or "").upper()
                target = PITCHGEOMETRY_NAME_TO_CANONICAL.get(kid)
                if target is None:
                    continue
                try:
                    x = float(r.get("x", "0"))
                    y = float(r.get("y", "0"))
                    vis = int(float(r.get("vis", "0")))
                except ValueError:
                    continue
                if vis <= 0 or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    continue
                canonical[target] = (x, y, 2)
                n_visible += 1
            if n_visible < 4:
                # Need at least 4 points for any usable homography sample.
                stats["skipped"] += 1
                continue
            split_value = (rows[0].get("split") or split_hint).lower()
            split_out = (
                "val"
                if split_value in {"val", "valid", "validation"}
                else ("test" if split_value == "test" else "train")
            )
            # Find image: the CSV references a frame key; look for an
            # image with that stem.
            stem = Path(frame_key).stem
            img_path = next(
                iter(p for p in src_root.rglob(f"{stem}.*") if p.suffix.lower() in IMAGE_EXTS),
                None,
            )
            if img_path is None:
                stats["skipped"] += 1
                continue
            line = _make_canonical_label_text(None, canonical, cls=0)
            out_img_dir = _ensure_dir(dst_root / "images" / split_out)
            out_lbl_dir = _ensure_dir(dst_root / "labels" / split_out)
            _link_or_copy(img_path, out_img_dir / img_path.name)
            (out_lbl_dir / f"{img_path.stem}.txt").write_text(line + "\n", encoding="utf-8")
            stats["images"] += 1
            stats["labels"] += 1
    return stats


# Kaggle bbox class -> canonical index
KAGGLE_BBOX_NAME_TO_CANONICAL: dict[str, int] = {
    "top_left_corner": 0,
    "top_right_corner": 26,
    "bottom_left_corner": 5,
    "bottom_right_corner": 31,
    "halfway_top": 14,
    "halfway_bottom": 17,
    "center_circle_top": 15,
    "center_circle_bottom": 16,
    "center_circle_left": 13,
    "center_circle_right": 18,
    "left_penalty_box_top_left": 1,
    "left_penalty_box_top_right": 9,
    "left_penalty_box_bottom_left": 4,
    "left_penalty_box_bottom_right": 12,
    "left_goal_area_top_left": 2,
    "left_goal_area_top_right": 6,
    "left_goal_area_bottom_left": 3,
    "left_goal_area_bottom_right": 7,
    "right_penalty_box_top_left": 19,
    "right_penalty_box_top_right": 27,
    "right_penalty_box_bottom_left": 22,
    "right_penalty_box_bottom_right": 30,
    "right_goal_area_top_left": 24,
    "right_goal_area_top_right": 28,
    "right_goal_area_bottom_left": 25,
    "right_goal_area_bottom_right": 29,
    "left_penalty_spot": 8,
    "right_penalty_spot": 23,
}


def convert_kaggle_bbox_landmarks(src_root: Path, dst_root: Path) -> dict[str, int]:
    """Convert hamzaboulahia/football-field-keypoints-dataset (Kaggle).

    The annotations are YOLO BBoxes with class names indicating the landmark.
    For each bbox we take its centre as the canonical keypoint position and
    set ``v=2``.  Images that match no canonical class are skipped.
    """
    stats = {"images": 0, "labels": 0, "skipped": 0, "unknown_classes": 0}
    # Look for a classes.txt or data.yaml to recover class id -> name.
    class_map: dict[int, str] = {}
    for cls_file in (src_root / "classes.txt", src_root / "obj.names"):
        if cls_file.exists():
            for i, line in enumerate(cls_file.read_text(encoding="utf-8").splitlines()):
                name = line.strip().lower()
                if name:
                    class_map[i] = name
            break
    if not class_map:
        # Try data.yaml
        data_yaml = next(src_root.rglob("data.yaml"), None)
        if data_yaml is not None:
            try:
                import yaml  # type: ignore[import-not-found]

                with data_yaml.open("r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                names = cfg.get("names") or []
                if isinstance(names, dict):
                    class_map = {int(k): str(v).lower() for k, v in names.items()}
                else:
                    class_map = dict(enumerate(str(n).lower() for n in names))
            except Exception:
                pass
    splits = (("train", "train"), ("valid", "val"), ("test", "test"))
    for split_in, split_out in splits:
        img_dir = next(
            iter(p for p in (src_root / split_in / "images", src_root / "images") if p.is_dir()),
            None,
        )
        lbl_dir = next(
            iter(p for p in (src_root / split_in / "labels", src_root / "labels") if p.is_dir()),
            None,
        )
        if img_dir is None or lbl_dir is None:
            continue
        for img_path in _all_image_files(img_dir):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                stats["skipped"] += 1
                continue
            canonical: list[tuple[float, float, float]] = [(0.0, 0.0, 0)] * NUM_KEYPOINTS
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                except ValueError:
                    continue
                cls_name = class_map.get(cls_id, "").lower().replace(" ", "_")
                target = KAGGLE_BBOX_NAME_TO_CANONICAL.get(cls_name)
                if target is None:
                    stats["unknown_classes"] += 1
                    continue
                if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
                    canonical[target] = (cx, cy, 2)
            if sum(1 for _, _, v in canonical if v > 0) < 4:
                stats["skipped"] += 1
                continue
            line_text = _make_canonical_label_text(None, canonical, cls=0)
            out_img_dir = _ensure_dir(dst_root / "images" / split_out)
            out_lbl_dir = _ensure_dir(dst_root / "labels" / split_out)
            _link_or_copy(img_path, out_img_dir / img_path.name)
            (out_lbl_dir / f"{img_path.stem}.txt").write_text(line_text + "\n", encoding="utf-8")
            stats["images"] += 1
            stats["labels"] += 1
    return stats


def convert_roboflow_yolo_pose(src_root: Path, dst_root: Path) -> dict[str, int]:
    """Convert any Roboflow YOLO Pose export.

    Most Roboflow Universe pitch projects already follow the 32-pt convention
    after the canonical f07vi project; for those we can reuse the pass-through
    converter.  When the kpt count differs, this falls back to truncation /
    zero-padding.
    """
    return convert_yolo_pose_passthrough(src_root, dst_root)


def convert_soccernet_images_only(src_root: Path, dst_root: Path) -> dict[str, int]:
    """No-op converter for ``soccernet_calibration_2023``.

    SoccerNet provides the *images* used to materialise the Soccana 29-pt
    JSON labels.  This function simply enumerates the available frames so the
    Soccana converter (run later in :func:`build_dataset`) can pick them up
    via ``soccernet_images``.  No labels are written from this source on its
    own.
    """
    n_images = sum(1 for _ in _index_soccernet_images(src_root).values())
    return {
        "images_indexed": n_images,
        "images": 0,  # No staging output yet — Soccana will consume them.
        "labels": 0,
    }


# Converters may accept extra kwargs (e.g. ``soccernet_images=`` for the
# Soccana converter); we use a generic ``Callable[..., ...]`` here.
CONVERTERS: dict[str, Callable[..., dict[str, int]]] = {
    "convert_yolo_pose_passthrough": convert_yolo_pose_passthrough,
    "convert_soccana_29pt": convert_soccana_29pt,
    "convert_pitchgeometry_csv": convert_pitchgeometry_csv,
    "convert_kaggle_bbox_landmarks": convert_kaggle_bbox_landmarks,
    "convert_roboflow_yolo_pose": convert_roboflow_yolo_pose,
    "convert_soccernet_images_only": convert_soccernet_images_only,
}


# ---------------------------------------------------------------------------
# Build orchestration
# ---------------------------------------------------------------------------


@dataclass
class SourceReport:
    name: str
    status: str
    download_path: Path | None = None
    staging_path: Path | None = None
    stats: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    notes: str = ""


@dataclass
class BuildReport:
    out_root: Path
    started_at: str
    finished_at: str
    sources: list[SourceReport]
    unified_image_counts: dict[str, int] = field(default_factory=dict)
    data_yaml: Path | None = None
    manifest_csv: Path | None = None

    def as_summary(self) -> str:
        lines = [
            "FIFA Dataset Builder report",
            "---------------------------",
            f"out_root:    {self.out_root}",
            f"started:     {self.started_at}",
            f"finished:    {self.finished_at}",
            f"sources:     {len(self.sources)}",
        ]
        for s in self.sources:
            lines.append(f"  - {s.name}: {s.status}")
            if s.error:
                lines.append(f"      error: {s.error}")
            elif s.stats:
                stat_str = ", ".join(f"{k}={v}" for k, v in s.stats.items())
                lines.append(f"      stats: {stat_str}")
            if s.notes:
                lines.append(f"      note:  {s.notes}")
        if self.unified_image_counts:
            lines.append("unified images:")
            for split, n in self.unified_image_counts.items():
                lines.append(f"  {split:>5s}: {n}")
        if self.data_yaml:
            lines.append(f"data.yaml:   {self.data_yaml}")
        if self.manifest_csv:
            lines.append(f"manifest:    {self.manifest_csv}")
        return "\n".join(lines)


def _select_sources(include: list[str] | None, include_optional: bool) -> list[DatasetSource]:
    if include:
        wanted = {s.lower() for s in include}
        return [s for s in REGISTRY if s.name.lower() in wanted or s.url.lower() in wanted]
    return [s for s in REGISTRY if include_optional or not s.optional]


def _split_assignment(
    image_paths: list[Path],
    seed: int,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
) -> dict[str, str]:
    rng = random.Random(seed)
    items = list(image_paths)
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, int(n * val_ratio)) if n > 1 else 0
    n_test = int(n * test_ratio)
    assignment: dict[str, str] = {}
    for i, p in enumerate(items):
        if i < n_test:
            assignment[p.name] = "test"
        elif i < n_test + n_val:
            assignment[p.name] = "val"
        else:
            assignment[p.name] = "train"
    return assignment


def _merge_staging_into_unified(
    staging_root: Path,
    unified_root: Path,
    *,
    keep_source_splits: bool = True,
    seed: int = 42,
) -> dict[str, int]:
    """Copy/symlink images and labels from ``staging_root/<source>`` into the
    unified dataset.  When ``keep_source_splits`` is true we honour the splits
    written by the converter; otherwise we shuffle deterministically.
    """
    counts = {"train": 0, "val": 0, "test": 0}
    for src_dir in sorted(staging_root.iterdir()):
        if not src_dir.is_dir():
            continue
        for split in ("train", "val", "test"):
            img_dir = src_dir / "images" / split
            lbl_dir = src_dir / "labels" / split
            if not img_dir.is_dir():
                continue
            for img_path in _all_image_files(img_dir):
                lbl_src = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_src.exists():
                    continue
                target_split = split
                if not keep_source_splits:
                    # Reassign deterministically using the file name as seed.
                    rng = random.Random((seed * 1_000_003) ^ hash(img_path.name))
                    target_split = rng.choices(["train", "val", "test"], weights=[0.85, 0.1, 0.05])[
                        0
                    ]
                # Prefix file name with source to avoid collisions.
                new_stem = f"{src_dir.name}__{img_path.stem}"
                out_img_dir = _ensure_dir(unified_root / "images" / target_split)
                out_lbl_dir = _ensure_dir(unified_root / "labels" / target_split)
                _link_or_copy(img_path, out_img_dir / f"{new_stem}{img_path.suffix.lower()}")
                shutil.copy2(lbl_src, out_lbl_dir / f"{new_stem}.txt")
                counts[target_split] += 1
    return counts


def _write_data_yaml(unified_root: Path) -> Path:
    yaml_path = unified_root / "data.yaml"
    flip_idx = ", ".join(str(i) for i in CANONICAL_FLIP_IDX_32)
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(("football_pitch",)))
    text = (
        f"# Auto-generated by vaila.fifa_dataset_builder on {_now()}\n"
        f"path: {unified_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"kpt_shape: [{NUM_KEYPOINTS}, 3]\n"
        f"flip_idx: [{flip_idx}]\n"
        f"\n"
        f"names:\n{names_yaml}\n"
    )
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def _write_manifest(unified_root: Path, sources: list[SourceReport]) -> Path:
    manifest = unified_root / "manifest.csv"
    rows = []
    for split in ("train", "val", "test"):
        img_dir = unified_root / "images" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            source_name = img.stem.split("__", 1)[0] if "__" in img.stem else "unknown"
            rows.append(
                {
                    "split": split,
                    "image": img.name,
                    "source": source_name,
                    "label": f"labels/{split}/{img.stem}.txt",
                }
            )
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image", "source", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


# ---------------------------------------------------------------------------
# Top-level build()
# ---------------------------------------------------------------------------


def build_dataset(
    out_root: Path,
    *,
    include: list[str] | None = None,
    include_optional: bool = False,
    keep_source_splits: bool = True,
    seed: int = 42,
    hf_token: str | None = None,
    roboflow_api_key: str | None = None,
    roboflow_version: int = 1,
    use_local_seed: Path | None = None,
    soccernet_dir: Path | None = None,
    soccernet_splits: tuple[str, ...] = ("train", "valid", "test", "challenge"),
    progress: Callable[[str], None] | None = None,
) -> BuildReport:
    """Download, convert and merge the configured datasets into ``out_root``.

    Args:
        out_root: Destination directory; created if missing.
        include: Optional subset of registry entries (by ``name`` or ``url``).
        include_optional: When True, include sources marked ``optional=True``
            even when they require credentials.
        keep_source_splits: Keep the train/val/test split written by the
            converter; when False the merger reshuffles deterministically.
        seed: RNG seed for split reshuffling.
        hf_token: Optional Hugging Face token (env var falls back automatically).
        roboflow_api_key: Optional Roboflow API key (env var fallback).
        roboflow_version: Default version to download from Roboflow.
        use_local_seed: Optional path to an existing copy of
            ``martinjolif/football-pitch-detection`` (e.g. the bundle that
            ships under ``vaila/models/hf_datasets/``).  When provided we
            skip the HF snapshot for that source and use the local copy.
        soccernet_dir: Optional path to an existing SoccerNet calibration-2023
            directory.  When provided we **skip** downloading SoccerNet and
            symlink it into ``sources/soccernet_calibration_2023/``.
        progress: Optional callback for human-readable progress lines.
    """

    out_root = out_root.resolve()
    sources_dir = _ensure_dir(out_root / "sources")
    staging_dir = _ensure_dir(out_root / "staging")
    unified_dir = _ensure_dir(out_root / "unified")
    reports_dir = _ensure_dir(out_root / "reports")
    selected = _select_sources(include, include_optional)
    started_at = _now()
    started = time.time()

    def log(msg: str) -> None:
        line = f"[{_now()}] {msg}"
        if progress:
            progress(line)
        print(line, flush=True)

    log(f"Build target: {out_root}")
    log(f"Selected {len(selected)} source(s): {[s.name for s in selected]}")

    hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    roboflow_api_key = roboflow_api_key or os.environ.get("ROBOFLOW_API_KEY")

    # SoccerNet calibration must be downloaded BEFORE Soccana so that we can
    # pair Soccana labels with the corresponding images.  Sort the selected
    # list so ``soccernet_calibration_2023`` runs before any soccana entry.
    soccernet_first: list[DatasetSource] = []
    others: list[DatasetSource] = []
    for s in selected:
        (soccernet_first if s.kind == "soccernet" else others).append(s)
    selected = soccernet_first + others

    soccernet_image_index: dict[str, Path] = {}
    reports: list[SourceReport] = []
    for src in selected:
        report = SourceReport(name=src.name, status="pending", notes=src.notes)
        ok, why = _source_available(src)
        if not ok:
            report.status = "skipped"
            report.error = why
            log(f"  - {src.name}: SKIP ({why})")
            reports.append(report)
            continue
        download_path = sources_dir / src.name
        try:
            log(f"  - {src.name}: downloading ({src.kind})")
            if src.kind == "hf_dataset":
                if (
                    use_local_seed is not None
                    and src.name.startswith("martinjolif")
                    and use_local_seed.exists()
                ):
                    log(f"      using local seed: {use_local_seed}")
                    if not download_path.exists():
                        os.symlink(use_local_seed.resolve(), download_path)
                else:
                    _download_hf_dataset(src.url, download_path, hf_token=hf_token)
            elif src.kind == "github":
                _download_github(src.url, download_path)
            elif src.kind == "roboflow_universe":
                if not roboflow_api_key:
                    raise RuntimeError("ROBOFLOW_API_KEY is required")
                _download_roboflow_universe(
                    src.url, download_path, api_key=roboflow_api_key, version=roboflow_version
                )
            elif src.kind == "kaggle":
                _download_kaggle_dataset(src.url, download_path)
            elif src.kind == "soccernet":
                if soccernet_dir is not None and soccernet_dir.exists():
                    log(f"      reusing existing SoccerNet dir: {soccernet_dir}")
                    if not download_path.exists():
                        os.symlink(soccernet_dir.resolve(), download_path)
                    _extract_soccernet_zips(
                        download_path, splits=tuple(soccernet_splits)
                    )
                else:
                    _download_soccernet_calibration(
                        download_path, splits=tuple(soccernet_splits)
                    )
            else:
                raise NotImplementedError(f"unknown source kind: {src.kind}")
        except Exception as e:  # pragma: no cover - network errors are runtime
            report.status = "download_failed"
            report.error = f"{type(e).__name__}: {e}"
            log(f"      download failed: {report.error}")
            reports.append(report)
            continue
        report.download_path = download_path
        log(f"      ok -> {download_path}")

        # Build/refresh the SoccerNet image index right after that download
        # finishes so it can be threaded into the Soccana converter.
        if src.kind == "soccernet":
            soccernet_image_index = _index_soccernet_images(download_path)
            log(f"      indexed {len(soccernet_image_index)} SoccerNet images")

        # Convert
        staging_path = staging_dir / src.name
        if staging_path.exists():
            shutil.rmtree(staging_path)
        _ensure_dir(staging_path)
        converter = CONVERTERS.get(src.converter)
        if converter is None:
            report.status = "no_converter"
            report.error = f"unknown converter '{src.converter}'"
            log(f"      no converter: {src.converter}")
            reports.append(report)
            continue
        try:
            kwargs: dict[str, object] = {}
            if src.converter == "convert_soccana_29pt" and soccernet_image_index:
                kwargs["soccernet_images"] = soccernet_image_index
            stats = converter(download_path, staging_path, **kwargs)
            report.stats = stats
            report.staging_path = staging_path
            report.status = "ok" if stats.get("images", 0) > 0 else "empty"
            log(f"      converted: {stats}")
        except Exception as e:
            report.status = "convert_failed"
            report.error = f"{type(e).__name__}: {e}"
            log(f"      convert failed: {report.error}")
        reports.append(report)

    # --- merge ---
    log("Merging staging -> unified")
    counts = _merge_staging_into_unified(
        staging_dir,
        unified_dir,
        keep_source_splits=keep_source_splits,
        seed=seed,
    )
    log(f"Unified counts: {counts}")
    data_yaml = _write_data_yaml(unified_dir)
    manifest = _write_manifest(unified_dir, reports)
    log(f"Wrote: {data_yaml}")
    log(f"Wrote: {manifest}")

    finished_at = _now()
    elapsed = time.time() - started
    log(f"Done in {elapsed:.1f}s")

    report_path = reports_dir / f"build_{_timestamp()}.log"
    full_report = BuildReport(
        out_root=out_root,
        started_at=started_at,
        finished_at=finished_at,
        sources=reports,
        unified_image_counts=counts,
        data_yaml=data_yaml,
        manifest_csv=manifest,
    )
    report_path.write_text(full_report.as_summary() + "\n", encoding="utf-8")
    log(f"Report saved: {report_path}")
    return full_report


# ---------------------------------------------------------------------------
# CLI / GUI entry points
# ---------------------------------------------------------------------------


def _vaila_local_seed_path() -> Path:
    """Return the bundled martinjolif copy if present, else a non-existent path."""
    here = Path(__file__).resolve().parent
    candidate = here / "models" / "hf_datasets" / "football-pitch-detection"
    return candidate


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vaila.fifa_dataset_builder",
        description=(
            "Download and unify open-source 32-pt soccer-pitch keypoint datasets "
            "into a single YOLO Pose dataset compatible with vaila.soccerfield_keypoints_ai."
        ),
    )
    p.add_argument(
        "--out-root",
        type=Path,
        required=False,
        help="Destination directory (default: current working directory).",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Restrict to specific source names or URLs.",
    )
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="Also try optional sources (Roboflow, Kaggle).",
    )
    p.add_argument(
        "--reshuffle-splits",
        action="store_true",
        help="Reassign train/val/test splits deterministically (else keep source splits).",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling (default: 42).")
    p.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (env: HF_TOKEN / HUGGING_FACE_HUB_TOKEN).",
    )
    p.add_argument(
        "--roboflow-api-key",
        default=None,
        help="Roboflow API key (env: ROBOFLOW_API_KEY).",
    )
    p.add_argument(
        "--roboflow-version",
        type=int,
        default=1,
        help="Roboflow project version to download (default: 1).",
    )
    p.add_argument(
        "--no-local-seed",
        action="store_true",
        help="Disable using the bundled martinjolif copy under vaila/models/hf_datasets/.",
    )
    p.add_argument(
        "--include-soccernet",
        action="store_true",
        help=(
            "Also download SoccerNet calibration-2023 images via the SoccerNet "
            "PyPI package (no NDA password needed for this task) and pair them "
            "with the Soccana 29-pt label JSONs."
        ),
    )
    p.add_argument(
        "--soccernet-dir",
        type=Path,
        default=None,
        help=(
            "Use an existing SoccerNet calibration-2023 directory (skips "
            "downloading). Expected layout: <dir>/calibration-2023/{train,valid,test,challenge}/."
        ),
    )
    p.add_argument(
        "--soccernet-splits",
        nargs="*",
        default=("train", "valid", "test", "challenge"),
        help="SoccerNet calibration splits to download (default: all 4).",
    )
    p.add_argument(
        "--preview",
        type=int,
        default=0,
        metavar="N",
        help=(
            "After (or instead of) building, render N random sample images "
            "from <out-root>/unified/ with keypoints overlaid into "
            "<out-root>/preview/. Set to 0 to disable (default)."
        ),
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Skip the build and only render --preview samples from an "
            "existing unified/ directory."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List the registered sources and exit.",
    )
    return p


def list_sources() -> None:
    print(f"{'name':45s} {'kind':22s} {'optional':9s} url")
    print("-" * 110)
    for s in REGISTRY:
        print(f"{s.name:45s} {s.kind:22s} {str(s.optional):9s} {s.url}")
    print()


def render_preview(unified_root: Path, n_samples: int = 12, *, seed: int = 0) -> Path:
    """Render ``n_samples`` random training images with their keypoints overlaid.

    Saves PNGs into ``<unified_root>/../preview/`` and returns the directory.
    Skips silently when ``cv2`` / ``numpy`` are unavailable.  Used as a quick
    QA check after a build.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # noqa: F401  (cv2 needs numpy at runtime)
    except ModuleNotFoundError:
        print("[preview] OpenCV not installed; skipping (uv add opencv-python).")
        return unified_root.parent / "preview"
    out_dir = unified_root.parent / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    candidates: list[Path] = []
    for split in ("train", "val", "test"):
        candidates.extend((unified_root / "labels" / split).glob("*.txt"))
    if not candidates:
        print(f"[preview] No labels found under {unified_root}; nothing to draw.")
        return out_dir
    rng.shuffle(candidates)
    drawn = 0
    for lbl_path in candidates:
        if drawn >= n_samples:
            break
        split = lbl_path.parent.name
        img_dir = unified_root / "images" / split
        # Try common extensions; staging uses the source extension verbatim.
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            cand = img_dir / f"{lbl_path.stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        line = lbl_path.read_text().strip().splitlines()[0].split()
        if len(line) < 5 + NUM_KEYPOINTS * 3:
            continue
        cls = int(line[0])
        cx, cy, bw, bh = (float(x) for x in line[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
        for i in range(NUM_KEYPOINTS):
            x = float(line[5 + i * 3]) * w
            y = float(line[5 + i * 3 + 1]) * h
            v = int(float(line[5 + i * 3 + 2]))
            if v <= 0:
                continue
            color = (0, 255, 0) if v == 2 else (0, 165, 255)
            cv2.circle(bgr, (int(x), int(y)), 4, color, -1)
            cv2.putText(
                bgr,
                str(i),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            bgr,
            f"{lbl_path.stem} [{split}] cls={cls}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_dir / f"{drawn:02d}_{split}_{lbl_path.stem}.jpg"), bgr)
        drawn += 1
    print(f"[preview] Wrote {drawn} sample(s) to {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.list:
        list_sources()
        return 0
    out_root = args.out_root if args.out_root else Path.cwd() / "dataset_vaila_fifa"
    if args.preview_only:
        unified = out_root / "unified"
        if not unified.exists():
            print(f"[preview-only] {unified} does not exist; build first.")
            return 1
        render_preview(unified, n_samples=max(1, args.preview), seed=args.seed)
        return 0
    use_local_seed = None if args.no_local_seed else _vaila_local_seed_path()
    if use_local_seed is not None and not (use_local_seed / "data" / "data.yaml").exists():
        use_local_seed = None

    # ``--include-soccernet`` is shorthand for adding the SoccerNet source name
    # to ``--include``.
    include_arg: list[str] | None = list(args.include) if args.include else None
    if args.include_soccernet:
        include_arg = (include_arg or [s.name for s in REGISTRY if not s.optional]) + [
            "soccernet_calibration_2023"
        ]
    report = build_dataset(
        out_root=out_root,
        include=include_arg,
        include_optional=args.include_optional,
        keep_source_splits=not args.reshuffle_splits,
        seed=args.seed,
        hf_token=args.hf_token,
        roboflow_api_key=args.roboflow_api_key,
        roboflow_version=args.roboflow_version,
        use_local_seed=use_local_seed,
        soccernet_dir=args.soccernet_dir,
        soccernet_splits=tuple(args.soccernet_splits),
    )
    print()
    print(report.as_summary())
    if args.preview > 0:
        render_preview(out_root / "unified", n_samples=args.preview, seed=args.seed)
    return 0


def run_gui_flow() -> int:
    """Tkinter dialog for use from ``vaila.py``."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    out_root_str = filedialog.askdirectory(
        title="FIFA Dataset Builder — choose target directory (will create dataset_vaila_fifa/)"
    )
    if not out_root_str:
        root.destroy()
        return 0
    out_root = Path(out_root_str)
    if out_root.name != "dataset_vaila_fifa":
        out_root = out_root / "dataset_vaila_fifa"

    include_optional = messagebox.askyesno(
        "Optional sources",
        "Include optional sources (Roboflow / Kaggle)?\n"
        "They require credentials in env vars (ROBOFLOW_API_KEY, KAGGLE_USERNAME/KAGGLE_KEY).",
    )

    include_soccernet = messagebox.askyesno(
        "SoccerNet calibration",
        "Also include SoccerNet calibration-2023 (~22 800 broadcast frames)?\n\n"
        "• No NDA password is required for this task.\n"
        "• Requires the 'SoccerNet' PyPI package (`uv add SoccerNet`).\n"
        "• Pairs the Soccana 29-pt label JSONs with the matching images.",
    )

    soccernet_dir: Path | None = None
    if include_soccernet:
        existing = filedialog.askdirectory(
            title=(
                "SoccerNet calibration-2023 already downloaded? "
                "Select its parent dir (cancel to download fresh)."
            )
        )
        if existing:
            soccernet_dir = Path(existing)

    sources_msg = [s.name for s in _select_sources(None, include_optional)]
    if include_soccernet and "soccernet_calibration_2023" not in sources_msg:
        sources_msg.append("soccernet_calibration_2023")

    msg = (
        f"Will build into:\n  {out_root}\n\n"
        f"Sources to attempt: {', '.join(sources_msg)}\n\n"
        "Continue?"
    )
    if not messagebox.askyesno("Confirm", msg):
        root.destroy()
        return 0

    use_local_seed = _vaila_local_seed_path()
    if not (use_local_seed / "data" / "data.yaml").exists():
        use_local_seed = None

    include_arg: list[str] | None = None
    if include_soccernet:
        include_arg = [s.name for s in REGISTRY if not s.optional] + ["soccernet_calibration_2023"]
    try:
        report = build_dataset(
            out_root=out_root,
            include=include_arg,
            include_optional=include_optional,
            use_local_seed=use_local_seed,
            soccernet_dir=soccernet_dir,
        )
        messagebox.showinfo(
            "FIFA Dataset Builder",
            f"Done.\n\n{report.as_summary()}\n\nSee logs in {out_root / 'reports'}.",
        )
    except Exception as e:
        messagebox.showerror("FIFA Dataset Builder", f"Build failed:\n{e}")
        root.destroy()
        return 1
    root.destroy()
    return 0


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        sys.exit(run_gui_flow())
    sys.exit(main())
