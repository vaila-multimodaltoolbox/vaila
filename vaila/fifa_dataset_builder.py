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
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

# ---------------------------------------------------------------------------
# Canonical 32-keypoint schema (Roboflow ``football-field-detection-f07vi``)
# ---------------------------------------------------------------------------
#
# The canonical schema is the one used by:
#   * ``martinjolif/football-pitch-detection`` (HF) — the model trained on this.
#   * Roboflow ``football-field-detection-f07vi`` and the ``roboflow/sports``
#     repo's ``SoccerPitchConfiguration`` (https://github.com/roboflow/sports).
#
# Reference field dimensions (cm), copied verbatim from
# ``sports/configs/soccer.py`` so the labels are bit-for-bit compatible with
# the HF model that vailá ships at
# ``models/runs/pose_fifa/pitch32_recipeA_400ep/weights/best.pt``::
#
#   width:               7000 cm  (along Y, "short" axis)
#   length:             12000 cm  (along X, "long" axis)
#   penalty_box_width:   4100 cm
#   penalty_box_length:  2015 cm
#   goal_box_width:      1832 cm
#   goal_box_length:      550 cm
#   centre_circle_radius: 915 cm
#   penalty_spot_distance: 1100 cm
#
# Coordinate system (Roboflow / canonical labels):
#   * origin = top-left corner of the field
#   * X grows to the right (toward the right endline)
#   * Y grows DOWN (toward the bottom sideline)  ← image convention
#
# vailá ``drawsportsfields.py`` uses a DIFFERENT convention (the user wants
# us to align to it):
#   * origin = field center  ([0, 0, 0])
#   * X grows to the right, Y grows UP
#   * dimensions ≈ 104.9 m × 67.9 m (FIFA standard) — see
#     ``vaila/models/soccerfield_ref3d_fifa.csv``.
#
# When we need the centered-metres convention we derive it from the Roboflow
# normalised coords; for homography projection we use the KpSFR template
# (114.83 × 74.37 yards = 105 × 68 m, Y-down, origin top-left).
ROBOFLOW_FIELD_LENGTH_CM = 12000
ROBOFLOW_FIELD_WIDTH_CM = 7000
ROBOFLOW_PEN_BOX_LEN_CM = 2015
ROBOFLOW_PEN_BOX_WID_CM = 4100
ROBOFLOW_GOAL_BOX_LEN_CM = 550
ROBOFLOW_GOAL_BOX_WID_CM = 1832
ROBOFLOW_CIRCLE_R_CM = 915
ROBOFLOW_PEN_SPOT_CM = 1100

# Drawsportsfields / FIFA centered model (length, width) in metres.
DRAWSPORTSFIELDS_LENGTH_M = 104.9
DRAWSPORTSFIELDS_WIDTH_M = 67.9

# KpSFR / WC14 template dimensions used by public homography datasets.
KPSFR_TEMPLATE_LENGTH_YARDS = 114.83
KPSFR_TEMPLATE_WIDTH_YARDS = 74.37


def _canonical_vertices_cm() -> list[tuple[float, float]]:
    """Return the 32 canonical keypoints in **Roboflow cm coordinates**.

    Same order/positions as :class:`sports.configs.soccer.SoccerPitchConfiguration`
    so that YOLO Pose label index ``i`` corresponds exactly to ``vertices[i]``.
    """
    L = ROBOFLOW_FIELD_LENGTH_CM
    W = ROBOFLOW_FIELD_WIDTH_CM
    PBL = ROBOFLOW_PEN_BOX_LEN_CM
    PBW = ROBOFLOW_PEN_BOX_WID_CM
    GBL = ROBOFLOW_GOAL_BOX_LEN_CM
    GBW = ROBOFLOW_GOAL_BOX_WID_CM
    R = ROBOFLOW_CIRCLE_R_CM
    PS = ROBOFLOW_PEN_SPOT_CM
    return [
        (0.0, 0.0),  # 0 top_left_corner
        (0.0, (W - PBW) / 2.0),  # 1 left_pen_box_top_outer
        (0.0, (W - GBW) / 2.0),  # 2 left_goal_area_top_outer
        (0.0, (W + GBW) / 2.0),  # 3 left_goal_area_bottom_outer
        (0.0, (W + PBW) / 2.0),  # 4 left_pen_box_bottom_outer
        (0.0, float(W)),  # 5 bottom_left_corner
        (float(GBL), (W - GBW) / 2.0),  # 6 left_goal_area_top_inner
        (float(GBL), (W + GBW) / 2.0),  # 7 left_goal_area_bottom_inner
        (float(PS), W / 2.0),  # 8 left_penalty_spot
        (float(PBL), (W - PBW) / 2.0),  # 9 left_pen_box_top_inner
        (float(PBL), (W - GBW) / 2.0),  # 10 left_pen_box_inner_top_at_goal_y
        (float(PBL), (W + GBW) / 2.0),  # 11 left_pen_box_inner_bottom_at_goal_y
        (float(PBL), (W + PBW) / 2.0),  # 12 left_pen_box_bottom_inner
        (L / 2.0, 0.0),  # 13 midfield_top
        (L / 2.0, W / 2.0 - R),  # 14 center_circle_top
        (L / 2.0, W / 2.0 + R),  # 15 center_circle_bottom
        (L / 2.0, float(W)),  # 16 midfield_bottom
        (float(L - PBL), (W - PBW) / 2.0),  # 17 right_pen_box_top_inner
        (float(L - PBL), (W - GBW) / 2.0),  # 18 right_pen_box_inner_top_at_goal_y
        (float(L - PBL), (W + GBW) / 2.0),  # 19 right_pen_box_inner_bottom_at_goal_y
        (float(L - PBL), (W + PBW) / 2.0),  # 20 right_pen_box_bottom_inner
        (float(L - PS), W / 2.0),  # 21 right_penalty_spot
        (float(L - GBL), (W - GBW) / 2.0),  # 22 right_goal_area_top_inner
        (float(L - GBL), (W + GBW) / 2.0),  # 23 right_goal_area_bottom_inner
        (float(L), 0.0),  # 24 top_right_corner
        (float(L), (W - PBW) / 2.0),  # 25 right_pen_box_top_outer
        (float(L), (W - GBW) / 2.0),  # 26 right_goal_area_top_outer
        (float(L), (W + GBW) / 2.0),  # 27 right_goal_area_bottom_outer
        (float(L), (W + PBW) / 2.0),  # 28 right_pen_box_bottom_outer
        (float(L), float(W)),  # 29 bottom_right_corner
        (L / 2.0 - R, W / 2.0),  # 30 center_circle_left
        (L / 2.0 + R, W / 2.0),  # 31 center_circle_right
    ]


# Index here is 0-based to match YOLO Pose ordering and the order of
# :func:`_canonical_vertices_cm`.  Names are derived from the Roboflow
# ``SoccerPitchConfiguration`` (and verified against the ``flip_idx`` below).
CANONICAL_KP_NAMES_32: tuple[str, ...] = (
    "top_left_corner",  # 0
    "left_pen_box_top_outer",  # 1
    "left_goal_area_top_outer",  # 2
    "left_goal_area_bottom_outer",  # 3
    "left_pen_box_bottom_outer",  # 4
    "bottom_left_corner",  # 5
    "left_goal_area_top_inner",  # 6
    "left_goal_area_bottom_inner",  # 7
    "left_penalty_spot",  # 8
    "left_pen_box_top_inner",  # 9
    "left_pen_box_inner_top_at_goal_y",  # 10
    "left_pen_box_inner_bottom_at_goal_y",  # 11
    "left_pen_box_bottom_inner",  # 12
    "midfield_top",  # 13
    "center_circle_top",  # 14
    "center_circle_bottom",  # 15
    "midfield_bottom",  # 16
    "right_pen_box_top_inner",  # 17
    "right_pen_box_inner_top_at_goal_y",  # 18
    "right_pen_box_inner_bottom_at_goal_y",  # 19
    "right_pen_box_bottom_inner",  # 20
    "right_penalty_spot",  # 21
    "right_goal_area_top_inner",  # 22
    "right_goal_area_bottom_inner",  # 23
    "top_right_corner",  # 24
    "right_pen_box_top_outer",  # 25
    "right_goal_area_top_outer",  # 26
    "right_goal_area_bottom_outer",  # 27
    "right_pen_box_bottom_outer",  # 28
    "bottom_right_corner",  # 29
    "center_circle_left",  # 30
    "center_circle_right",  # 31
)
NUM_KEYPOINTS = 32

# Horizontal-flip permutation, matching the ``flip_idx`` published by the
# canonical HF dataset and verified against :func:`_canonical_vertices_cm`
# (each pair (i, flip_idx[i]) maps to a horizontal mirror around X = L/2).
CANONICAL_FLIP_IDX_32: tuple[int, ...] = (
    24,  # 0  top_left_corner            <-> top_right_corner
    25,  # 1  left_pen_box_top_outer     <-> right_pen_box_top_outer
    26,  # 2  left_goal_area_top_outer   <-> right_goal_area_top_outer
    27,  # 3  left_goal_area_bottom_outer<-> right_goal_area_bottom_outer
    28,  # 4  left_pen_box_bottom_outer  <-> right_pen_box_bottom_outer
    29,  # 5  bottom_left_corner         <-> bottom_right_corner
    22,  # 6  left_goal_area_top_inner   <-> right_goal_area_top_inner
    23,  # 7  left_goal_area_bottom_inner<-> right_goal_area_bottom_inner
    21,  # 8  left_penalty_spot          <-> right_penalty_spot
    17,  # 9  left_pen_box_top_inner     <-> right_pen_box_top_inner
    18,  # 10 left_pen_box_inner_top_y   <-> right_pen_box_inner_top_y
    19,  # 11 left_pen_box_inner_bot_y   <-> right_pen_box_inner_bot_y
    20,  # 12 left_pen_box_bottom_inner  <-> right_pen_box_bottom_inner
    13,  # 13 midfield_top (self)
    14,  # 14 center_circle_top (self)
    15,  # 15 center_circle_bottom (self)
    16,  # 16 midfield_bottom (self)
    9,  # 17 -> 9
    10,  # 18 -> 10
    11,  # 19 -> 11
    12,  # 20 -> 12
    8,  # 21 -> 8
    6,  # 22 -> 6
    7,  # 23 -> 7
    0,  # 24 -> 0
    1,  # 25 -> 1
    2,  # 26 -> 2
    3,  # 27 -> 3
    4,  # 28 -> 4
    5,  # 29 -> 5
    31,  # 30 center_circle_left  <-> center_circle_right
    30,  # 31
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
    DatasetSource(
        name="worldcup2014_nhoma",
        kind="url_archive",
        url="https://nhoma.github.io/data/soccer_data.tar.gz",
        converter="convert_worldcup2014_homography",
        licence="see WorldCup 2014 dataset page/license terms",
        notes=(
            "WorldCup 2014 benchmark (209 train + 186 test images) with "
            "per-frame homography matrices. We project the canonical 32 points "
            "from the centered FIFA template into each image."
        ),
        optional=True,
    ),
    DatasetSource(
        name="ts_worldcup_kpsfr",
        kind="url_archive",
        url="https://cgv.cs.nthu.edu.tw/KpSFR_data/TS-WorldCup.zip",
        converter="convert_tsworldcup_homography",
        licence="see TS-WorldCup/KpSFR page",
        notes=(
            "TS-WorldCup (3,812 frames, 43 clips) with per-frame homography "
            "matrices under Annotations/. Converted to canonical 32-kp labels "
            "by projecting centered FIFA template points."
        ),
        optional=True,
    ),
    DatasetSource(
        name="wc14_tvcalib_additional_annotations",
        kind="url_archive",
        url="https://tib.eu/cloud/s/Jz4x2KsjinEEkwQ/download/wc14-test-additional_annotations_wacv23_theiner.tar",
        converter="convert_wc14_tvcalib_segments",
        licence="see TVCalib dataset page/license terms",
        notes=(
            "Additional WC14 segment annotations in SoccerNet-style format. "
            "Converted to canonical keypoints via geometric line intersections."
        ),
        optional=True,
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


def _download_url_archive(url: str, dst: Path) -> None:
    """Download a .zip / .tar(.gz) archive URL and extract into ``dst``."""
    _ensure_dir(dst)
    parsed = urllib.parse.urlparse(url)
    archive_name = Path(parsed.path).name or "dataset_archive"
    archive_path = dst / archive_name
    if not archive_path.exists():
        urllib.request.urlretrieve(url, archive_path)  # noqa: S310
    # Idempotency marker: if this exists, extraction has already run.
    extracted_marker = dst / ".extracted_ok"
    if extracted_marker.exists():
        return
    lower = archive_name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dst)
    elif lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dst)
    else:
        raise RuntimeError(f"Unsupported archive extension for URL source: {archive_name}")
    extracted_marker.write_text("ok\n", encoding="utf-8")


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


def _canonical_vertices_normalized() -> list[tuple[float, float]]:
    """Return the 32 canonical keypoints as ``(nx, ny)`` in ``[0, 1]^2``.

    Normalised against the Roboflow template (12000×7000 cm), Y-down,
    origin at top-left corner — the same convention used by the YOLO Pose
    label files.
    """
    L = float(ROBOFLOW_FIELD_LENGTH_CM)
    W = float(ROBOFLOW_FIELD_WIDTH_CM)
    return [(x / L, y / W) for (x, y) in _canonical_vertices_cm()]


def _load_centered_fifa_points_32() -> tuple[list[tuple[float, float, float]], float, float]:
    """Return canonical 32 keypoints in **centered** drawsportsfields metres.

    The output convention matches ``vaila/models/soccerfield_ref3d_fifa.csv``:

    * origin = field center ``(0, 0, 0)``
    * X grows to the right, Y grows UP
    * field dimensions = ``DRAWSPORTSFIELDS_LENGTH_M × DRAWSPORTSFIELDS_WIDTH_M``
      (FIFA standard: 104.9 × 67.9 m)

    The ``length_m`` and ``width_m`` values returned describe the FIFA field
    in metres (X span and Y span respectively).  Used by the homography
    converters and by ``_write_keypoint_reference``.
    """
    L_m = DRAWSPORTSFIELDS_LENGTH_M
    W_m = DRAWSPORTSFIELDS_WIDTH_M
    centered: list[tuple[float, float, float]] = []
    for nx, ny in _canonical_vertices_normalized():
        # Roboflow: origin top-left, Y-down, normalised in [0,1].
        # Centered drawsportsfields: origin centre, Y-up, scaled to FIFA dims.
        x_m = (nx - 0.5) * L_m
        y_m = (0.5 - ny) * W_m
        centered.append((x_m, y_m, 0.0))
    return centered, L_m, W_m


def _centered_meters_to_kpsfr_template(
    x_m: float,
    y_m: float,
    *,
    length_m: float,
    width_m: float,
) -> tuple[float, float]:
    """Map centered (Y-up) metres to KpSFR template (yards, Y-down, top-left origin).

    KpSFR / WC14 use a 114.83 × 74.37 yard template with origin at the
    top-left corner of the field and Y growing DOWN (image convention).
    The drawsportsfields convention is centered with Y growing UP, so we
    flip the Y axis when shifting.
    """
    # Centered (Y-up) -> normalised top-left (Y-down).
    nx = x_m / length_m + 0.5
    ny = 0.5 - y_m / width_m
    x_tpl = nx * KPSFR_TEMPLATE_LENGTH_YARDS
    y_tpl = ny * KPSFR_TEMPLATE_WIDTH_YARDS
    return x_tpl, y_tpl


def _project_template_points_to_image(
    H: np.ndarray,
    template_xy: list[tuple[float, float]],
    *,
    img_w: int,
    img_h: int,
) -> list[tuple[float, float, float]]:
    """Project template points to image using whichever H direction fits best.

    Public sports datasets differ in homography convention. We try both:
    - image -> template (KpSFR convention): image = template @ inv(H.T)
    - template -> image: image = template @ H.T
    and keep the one with more projected points inside image bounds.
    """
    src = np.array([[x, y, 1.0] for x, y in template_xy], dtype=np.float64)
    mats: list[np.ndarray] = []
    with contextlib.suppress(np.linalg.LinAlgError):
        mats.append(np.linalg.inv(H.T))
    mats.append(H.T)

    best: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * len(template_xy)
    best_inside = -1
    for M in mats:
        proj = src @ M
        z = proj[:, 2]
        valid_z = np.abs(z) > 1e-9
        xy = np.zeros((len(template_xy), 2), dtype=np.float64)
        xy[valid_z, 0] = proj[valid_z, 0] / z[valid_z]
        xy[valid_z, 1] = proj[valid_z, 1] / z[valid_z]

        candidate: list[tuple[float, float, float]] = []
        inside = 0
        for i in range(len(template_xy)):
            if not valid_z[i]:
                candidate.append((0.0, 0.0, 0.0))
                continue
            x = float(xy[i, 0])
            y = float(xy[i, 1])
            if 0.0 <= x < float(img_w) and 0.0 <= y < float(img_h):
                candidate.append((x / img_w, y / img_h, 2.0))
                inside += 1
            else:
                candidate.append((0.0, 0.0, 0.0))
        if inside > best_inside:
            best = candidate
            best_inside = inside
    return best


def convert_images_only(src_root: Path, dst_root: Path) -> dict[str, int]:
    """No-op converter used for metadata-only sources."""
    n_images = sum(1 for _ in _all_image_files(src_root))
    return {"images_indexed": n_images, "images": 0, "labels": 0}


def _fit_line_abc(points_xy: list[tuple[float, float]]) -> tuple[float, float, float] | None:
    """Fit line ax+by+c=0 through points in normalized image coordinates."""
    if len(points_xy) < 2:
        return None
    arr = np.array([[x, y, 1.0] for x, y in points_xy], dtype=np.float64)
    # Total least squares via homogeneous SVD.
    _, _, vh = np.linalg.svd(arr)
    a, b, c = vh[-1, :]
    norm = float(np.hypot(a, b))
    if norm <= 1e-12:
        return None
    return (a / norm, b / norm, c / norm)


def _line_intersection(
    l1: tuple[float, float, float] | None,
    l2: tuple[float, float, float] | None,
) -> tuple[float, float] | None:
    if l1 is None or l2 is None:
        return None
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-12:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (float(x), float(y))


def _as_xy_list(raw_points: object) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    if not isinstance(raw_points, list):
        return out
    for p in raw_points:
        if not isinstance(p, dict):
            continue
        p_dict = cast(dict[str, Any], p)
        x_raw = p_dict.get("x")
        y_raw = p_dict.get("y")
        if not isinstance(x_raw, int | float | str):
            continue
        if not isinstance(y_raw, int | float | str):
            continue
        try:
            x = float(x_raw)
            y = float(y_raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x) and np.isfinite(y):
            out.append((x, y))
    return out


def _put_if_valid(
    canonical: list[tuple[float, float, float]],
    idx: int,
    pt: tuple[float, float] | None,
) -> None:
    if pt is None:
        return
    x, y = pt
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        canonical[idx] = (x, y, 2.0)


def _approx_circle_extrema(points: list[tuple[float, float]]) -> dict[str, tuple[float, float]]:
    """Approximate left/right/top/bottom points from sampled circle/polyline points."""
    if not points:
        return {}
    left = min(points, key=lambda p: p[0])
    right = max(points, key=lambda p: p[0])
    top = min(points, key=lambda p: p[1])
    bottom = max(points, key=lambda p: p[1])
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def convert_wc14_tvcalib_segments(src_root: Path, dst_root: Path) -> dict[str, int]:
    """Convert WC14 TVCalib additional segment annotations to canonical 32-kp labels."""
    stats = {"images": 0, "labels": 0, "skipped": 0, "low_visible": 0}
    # Resolve images from the WC14 source downloaded by worldcup2014_nhoma.
    wc14_root = src_root.parent / "worldcup2014_nhoma"

    def _resolve_wc14_image(stem: str) -> Path | None:
        for candidate in (
            wc14_root / "raw" / "test" / f"{stem}.jpg",
            wc14_root / "raw" / "train_val" / f"{stem}.jpg",
            wc14_root / f"{stem}.jpg",
        ):
            if candidate.exists():
                return candidate
        matches = list(wc14_root.rglob(f"{stem}.jpg"))
        return matches[0] if matches else None

    for json_path in sorted(src_root.glob("*.json")):
        if json_path.name == "match_info_cam_gt.json":
            continue
        stem = json_path.stem
        img_path = _resolve_wc14_image(stem)
        if img_path is None:
            stats["skipped"] += 1
            continue
        try:
            ann = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            stats["skipped"] += 1
            continue

        lines: dict[str, tuple[float, float, float] | None] = {}
        for k, v in ann.items():
            pts = _as_xy_list(v)
            lines[k] = _fit_line_abc(pts)

        canonical: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * NUM_KEYPOINTS
        # Field corners and sidelines.
        _put_if_valid(
            canonical,
            0,
            _line_intersection(lines.get("Side line top"), lines.get("Side line left")),
        )
        _put_if_valid(
            canonical,
            24,
            _line_intersection(lines.get("Side line top"), lines.get("Side line right")),
        )
        _put_if_valid(
            canonical,
            5,
            _line_intersection(lines.get("Side line bottom"), lines.get("Side line left")),
        )
        _put_if_valid(
            canonical,
            29,
            _line_intersection(lines.get("Side line bottom"), lines.get("Side line right")),
        )
        # Midfield (intersections of the middle line with sidelines).
        _put_if_valid(
            canonical, 13, _line_intersection(lines.get("Middle line"), lines.get("Side line top"))
        )
        _put_if_valid(
            canonical,
            16,
            _line_intersection(lines.get("Middle line"), lines.get("Side line bottom")),
        )
        # Left big rect (penalty box):
        # 1 = outer top (on side line left), 9 = inner top (on big rect main),
        # 4 = outer bottom, 12 = inner bottom.
        _put_if_valid(
            canonical,
            1,
            _line_intersection(lines.get("Big rect. left top"), lines.get("Side line left")),
        )
        _put_if_valid(
            canonical,
            9,
            _line_intersection(lines.get("Big rect. left top"), lines.get("Big rect. left main")),
        )
        _put_if_valid(
            canonical,
            4,
            _line_intersection(lines.get("Big rect. left bottom"), lines.get("Side line left")),
        )
        _put_if_valid(
            canonical,
            12,
            _line_intersection(
                lines.get("Big rect. left bottom"), lines.get("Big rect. left main")
            ),
        )
        # Left small rect (goal area):
        # 2 = outer top, 6 = inner top, 3 = outer bottom, 7 = inner bottom.
        _put_if_valid(
            canonical,
            2,
            _line_intersection(lines.get("Small rect. left top"), lines.get("Side line left")),
        )
        _put_if_valid(
            canonical,
            6,
            _line_intersection(
                lines.get("Small rect. left top"), lines.get("Small rect. left main")
            ),
        )
        _put_if_valid(
            canonical,
            3,
            _line_intersection(lines.get("Small rect. left bottom"), lines.get("Side line left")),
        )
        _put_if_valid(
            canonical,
            7,
            _line_intersection(
                lines.get("Small rect. left bottom"), lines.get("Small rect. left main")
            ),
        )
        # Right big rect:
        # 25 = outer top (on side line right), 17 = inner top,
        # 28 = outer bottom, 20 = inner bottom.
        _put_if_valid(
            canonical,
            25,
            _line_intersection(lines.get("Big rect. right top"), lines.get("Side line right")),
        )
        _put_if_valid(
            canonical,
            17,
            _line_intersection(lines.get("Big rect. right top"), lines.get("Big rect. right main")),
        )
        _put_if_valid(
            canonical,
            28,
            _line_intersection(lines.get("Big rect. right bottom"), lines.get("Side line right")),
        )
        _put_if_valid(
            canonical,
            20,
            _line_intersection(
                lines.get("Big rect. right bottom"), lines.get("Big rect. right main")
            ),
        )
        # Right small rect:
        # 26 = outer top, 22 = inner top, 27 = outer bottom, 23 = inner bottom.
        _put_if_valid(
            canonical,
            26,
            _line_intersection(lines.get("Small rect. right top"), lines.get("Side line right")),
        )
        _put_if_valid(
            canonical,
            22,
            _line_intersection(
                lines.get("Small rect. right top"), lines.get("Small rect. right main")
            ),
        )
        _put_if_valid(
            canonical,
            27,
            _line_intersection(lines.get("Small rect. right bottom"), lines.get("Side line right")),
        )
        _put_if_valid(
            canonical,
            23,
            _line_intersection(
                lines.get("Small rect. right bottom"), lines.get("Small rect. right main")
            ),
        )

        # Penalty spots: take the centre of the small left/right circles
        # (approx. by the bounding box centroid of sampled points).
        for ann_name, idx in (("Circle left", 8), ("Circle right", 21)):
            circ_pts = _as_xy_list(ann.get(ann_name))
            if circ_pts:
                ex = _approx_circle_extrema(circ_pts)
                cx = 0.5 * (ex["left"][0] + ex["right"][0])
                cy = 0.5 * (ex["top"][1] + ex["bottom"][1])
                _put_if_valid(canonical, idx, (cx, cy))
        # Center circle: top/bottom intersections (14, 15) and the left/right
        # extremes (30, 31).
        center_circle_pts = _as_xy_list(ann.get("Circle central"))
        if center_circle_pts:
            ex = _approx_circle_extrema(center_circle_pts)
            _put_if_valid(canonical, 30, ex.get("left"))
            _put_if_valid(canonical, 31, ex.get("right"))
            _put_if_valid(canonical, 14, ex.get("top"))
            _put_if_valid(canonical, 15, ex.get("bottom"))

        if sum(1 for _, _, v in canonical if v > 0) < 4:
            stats["low_visible"] += 1
            continue
        line_text = _make_canonical_label_text(None, canonical, cls=0)
        out_img_dir = _ensure_dir(dst_root / "images" / "test")
        out_lbl_dir = _ensure_dir(dst_root / "labels" / "test")
        _link_or_copy(img_path, out_img_dir / img_path.name)
        (out_lbl_dir / f"{stem}.txt").write_text(line_text + "\n", encoding="utf-8")
        stats["images"] += 1
        stats["labels"] += 1
    return stats


def _parse_split_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().strip("/")
        if line:
            out.add(line)
    return out


def convert_worldcup2014_homography(src_root: Path, dst_root: Path) -> dict[str, int]:
    """Convert WC14 (.homographyMatrix) to canonical 32-pt YOLO Pose labels."""
    import numpy as np
    from PIL import Image

    stats = {"images": 0, "labels": 0, "skipped": 0, "low_visible": 0}
    world_pts_m, length_m, width_m = _load_centered_fifa_points_32()
    template_xy = [
        _centered_meters_to_kpsfr_template(x, y, length_m=length_m, width_m=width_m)
        for x, y, _ in world_pts_m
    ]

    for hom in sorted(src_root.rglob("*.homographyMatrix")):
        stem = hom.stem
        img_path = next(
            (p for p in (hom.with_suffix(".jpg"), hom.with_suffix(".png")) if p.exists()),
            None,
        )
        if img_path is None:
            stats["skipped"] += 1
            continue
        split_out = "test" if "test" in hom.parts else "train"
        try:
            H = np.loadtxt(hom)
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            stats["skipped"] += 1
            continue
        canonical = _project_template_points_to_image(H, template_xy, img_w=img_w, img_h=img_h)
        if sum(1 for _, _, v in canonical if v > 0) < 4:
            stats["low_visible"] += 1
            continue
        line_text = _make_canonical_label_text(None, canonical, cls=0)
        out_img_dir = _ensure_dir(dst_root / "images" / split_out)
        out_lbl_dir = _ensure_dir(dst_root / "labels" / split_out)
        _link_or_copy(img_path, out_img_dir / img_path.name)
        (out_lbl_dir / f"{stem}.txt").write_text(line_text + "\n", encoding="utf-8")
        stats["images"] += 1
        stats["labels"] += 1
    return stats


def convert_tsworldcup_homography(src_root: Path, dst_root: Path) -> dict[str, int]:
    """Convert TS-WorldCup homography annotations to canonical 32-pt YOLO Pose."""
    import numpy as np
    from PIL import Image

    stats = {"images": 0, "labels": 0, "skipped": 0, "low_visible": 0}
    world_pts_m, length_m, width_m = _load_centered_fifa_points_32()
    template_xy = [
        _centered_meters_to_kpsfr_template(x, y, length_m=length_m, width_m=width_m)
        for x, y, _ in world_pts_m
    ]

    train_set = _parse_split_file(src_root / "TS-WorldCup" / "train.txt")
    test_set = _parse_split_file(src_root / "TS-WorldCup" / "test.txt")
    ann_root = src_root / "TS-WorldCup" / "Annotations"
    img_root = src_root / "TS-WorldCup" / "Dataset"

    for hom in sorted(ann_root.rglob("*_homography.npy")):
        rel_clip = hom.parent.relative_to(ann_root)
        # The split files omit the score-range directory (e.g. "80_95/").
        clip_key = "/".join(rel_clip.parts[1:]) if len(rel_clip.parts) > 1 else rel_clip.as_posix()
        if clip_key in test_set:
            split_out = "test"
        elif clip_key in train_set:
            split_out = "train"
        else:
            split_out = "train"

        img_name = hom.name.replace("_homography.npy", ".jpg")
        img_path = img_root / rel_clip / img_name
        if not img_path.exists():
            img_path = img_root / rel_clip / img_name.replace(".jpg", ".png")
        if not img_path.exists():
            stats["skipped"] += 1
            continue
        try:
            H = np.load(hom)
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            stats["skipped"] += 1
            continue
        canonical = _project_template_points_to_image(H, template_xy, img_w=img_w, img_h=img_h)
        if sum(1 for _, _, v in canonical if v > 0) < 4:
            stats["low_visible"] += 1
            continue
        line_text = _make_canonical_label_text(None, canonical, cls=0)
        stem = f"{rel_clip.as_posix().replace('/', '__')}__{img_path.stem}"
        out_img_dir = _ensure_dir(dst_root / "images" / split_out)
        out_lbl_dir = _ensure_dir(dst_root / "labels" / split_out)
        _link_or_copy(img_path, out_img_dir / f"{stem}{img_path.suffix.lower()}")
        (out_lbl_dir / f"{stem}.txt").write_text(line_text + "\n", encoding="utf-8")
        stats["images"] += 1
        stats["labels"] += 1
    return stats


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
                # Layout: [cls, cx, cy, bw, bh, x1, y1, v1, x2, y2, v2, ...].
                # Visibility flags live at indices 7, 10, 13, ... i.e.
                # ``i >= 5 and (i - 5) % 3 == 2``.  Everything else is a float.
                parts: list[str] = []
                for i, val in enumerate(row):
                    if i == 0:
                        parts.append(str(int(val)))
                    elif i >= 5 and (i - 5) % 3 == 2:
                        parts.append(str(int(round(val))))
                    else:
                        parts.append(f"{val:.6f}")
                line = " ".join(parts)
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
    0: 0,  # sideline_top_left              -> top_left_corner (0)
    1: 1,  # big_rect_left_top_pt1   (out)  -> left_pen_box_top_outer (1)
    2: 9,  # big_rect_left_top_pt2   (in)   -> left_pen_box_top_inner (9)
    3: 4,  # big_rect_left_bottom_pt1 (out) -> left_pen_box_bottom_outer (4)
    4: 12,  # big_rect_left_bottom_pt2 (in)  -> left_pen_box_bottom_inner (12)
    5: 2,  # small_rect_left_top_pt1 (out)  -> left_goal_area_top_outer (2)
    6: 6,  # small_rect_left_top_pt2 (in)   -> left_goal_area_top_inner (6)
    7: 3,  # small_rect_left_bottom_pt1 (out) -> left_goal_area_bottom_outer (3)
    8: 7,  # small_rect_left_bottom_pt2 (in)  -> left_goal_area_bottom_inner (7)
    9: 5,  # sideline_bottom_left           -> bottom_left_corner (5)
    10: -1,  # left_semicircle_right          (apex; no canonical match)
    11: 13,  # center_line_top                -> midfield_top (13)
    12: 16,  # center_line_bottom             -> midfield_bottom (16)
    13: 14,  # center_circle_top              -> center_circle_top (14)
    14: 15,  # center_circle_bottom           -> center_circle_bottom (15)
    15: -1,  # field_center                   (no canonical match)
    16: 24,  # sideline_top_right             -> top_right_corner (24)
    17: 25,  # big_rect_right_top_pt1   (out) -> right_pen_box_top_outer (25)
    18: 17,  # big_rect_right_top_pt2   (in)  -> right_pen_box_top_inner (17)
    19: 28,  # big_rect_right_bottom_pt1 (out)-> right_pen_box_bottom_outer (28)
    20: 20,  # big_rect_right_bottom_pt2 (in) -> right_pen_box_bottom_inner (20)
    21: 26,  # small_rect_right_top_pt1 (out) -> right_goal_area_top_outer (26)
    22: 22,  # small_rect_right_top_pt2 (in)  -> right_goal_area_top_inner (22)
    23: 27,  # small_rect_right_bottom_pt1 (out) -> right_goal_area_bottom_outer (27)
    24: 23,  # small_rect_right_bottom_pt2 (in)  -> right_goal_area_bottom_inner (23)
    25: 29,  # sideline_bottom_right          -> bottom_right_corner (29)
    26: -1,  # right_semicircle_left          (apex; no canonical match)
    27: 30,  # center_circle_left             -> center_circle_left (30)
    28: 31,  # center_circle_right            -> center_circle_right (31)
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
    "center_line_top": 13,
    "center_line_bottom": 16,
    "center_circle_top": 14,
    "center_circle_bottom": 15,
    "sideline_top_right": 24,
    "big_rect_right_top_pt1": 25,
    "big_rect_right_top_pt2": 17,
    "big_rect_right_bottom_pt1": 28,
    "big_rect_right_bottom_pt2": 20,
    "small_rect_right_top_pt1": 26,
    "small_rect_right_top_pt2": 22,
    "small_rect_right_bottom_pt1": 27,
    "small_rect_right_bottom_pt2": 23,
    "sideline_bottom_right": 29,
    "center_circle_left": 30,
    "center_circle_right": 31,
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
    """Map ``"<split>/<stem>"`` -> ``Path/.../calibration-2023/<split>/<stem>.jpg``.

    SoccerNet calibration-2023 reuses the same stems (``00000.jpg``, …) across
    ``train/``, ``valid/``, ``test/`` and ``challenge/``, so we need a
    split-aware key.  Plain stem lookups also still work via a fallback that
    points at the *first* split where the stem is found, but Soccana 29-pt
    labels are split-tagged and **always** prefer the ``"<split>/<stem>"``
    form (see :func:`convert_soccana_29pt`).
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
                    key = f"{split}/{p.stem}"
                    out.setdefault(key, p)
                    # Stem-only fallback (used by tests / legacy callers).
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
            # Prefer the split-aware lookup (e.g. "valid/00000") so that
            # Soccana ``valid/00000.json`` pairs with SoccerNet
            # ``calibration-2023/valid/00000.jpg`` and not with the train one
            # — they share the same stem but are different physical frames.
            soccernet_split = "valid" if split_in == "valid" else split_in
            img_path = img_index.get(f"{soccernet_split}/{stem}") or img_index.get(stem)
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
    "TOP_RIGHT_CORNER": 24,
    "BOTTOM_LEFT_CORNER": 5,
    "BOTTOM_RIGHT_CORNER": 29,
    "MIDFIELD_TOP": 13,
    "MIDFIELD_BOTTOM": 16,
    "CENTER_CIRCLE_TOP": 14,
    "CENTER_CIRCLE_BOTTOM": 15,
    "CENTER_CIRCLE_LEFT": 30,
    "CENTER_CIRCLE_RIGHT": 31,
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
    "RIGHT_PENALTY_AREA_TOP_LEFT": 17,
    "RIGHT_PENALTY_AREA_TOP_RIGHT": 25,
    "RIGHT_PENALTY_AREA_BOTTOM_LEFT": 20,
    "RIGHT_PENALTY_AREA_BOTTOM_RIGHT": 28,
    "RIGHT_GOAL_AREA_TOP_LEFT": 22,
    "RIGHT_GOAL_AREA_TOP_RIGHT": 26,
    "RIGHT_GOAL_AREA_BOTTOM_LEFT": 23,
    "RIGHT_GOAL_AREA_BOTTOM_RIGHT": 27,
    "RIGHT_PENALTY_SPOT": 21,
    "RIGHT_PENALTY_ARC_TOP": 18,
    "RIGHT_PENALTY_ARC_BOTTOM": 19,
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
    "top_right_corner": 24,
    "bottom_left_corner": 5,
    "bottom_right_corner": 29,
    "halfway_top": 13,
    "halfway_bottom": 16,
    "center_circle_top": 14,
    "center_circle_bottom": 15,
    "center_circle_left": 30,
    "center_circle_right": 31,
    "left_penalty_box_top_left": 1,
    "left_penalty_box_top_right": 9,
    "left_penalty_box_bottom_left": 4,
    "left_penalty_box_bottom_right": 12,
    "left_goal_area_top_left": 2,
    "left_goal_area_top_right": 6,
    "left_goal_area_bottom_left": 3,
    "left_goal_area_bottom_right": 7,
    "right_penalty_box_top_left": 17,
    "right_penalty_box_top_right": 25,
    "right_penalty_box_bottom_left": 20,
    "right_penalty_box_bottom_right": 28,
    "right_goal_area_top_left": 22,
    "right_goal_area_top_right": 26,
    "right_goal_area_bottom_left": 23,
    "right_goal_area_bottom_right": 27,
    "left_penalty_spot": 8,
    "right_penalty_spot": 21,
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
    # ``_index_soccernet_images`` writes both ``"<split>/<stem>"`` and the
    # bare ``stem`` keys.  Count unique physical paths so the human-readable
    # "indexed N" reflects the actual number of frames.
    n_images = len({str(p) for p in _index_soccernet_images(src_root).values()})
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
    "convert_worldcup2014_homography": convert_worldcup2014_homography,
    "convert_tsworldcup_homography": convert_tsworldcup_homography,
    "convert_images_only": convert_images_only,
    "convert_wc14_tvcalib_segments": convert_wc14_tvcalib_segments,
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


# Fusion priority: when the same physical image (matched by stem + dimensions)
# is annotated by multiple sources, keep the label from the highest-priority
# source.  Higher value = higher priority.
SOURCE_FUSION_PRIORITY: dict[str, int] = {
    "martinjolif_football-pitch-detection": 100,  # ground truth (HF)
    "wc14_tvcalib_additional_annotations": 90,  # TVCalib geometric labels
    "Adit-jain_Soccana_Keypoint_detection_v1": 80,  # Soccana JSON
    "PiotrGrabysz_PitchGeometry": 70,
    "ts_worldcup_kpsfr": 60,  # homography-derived
    "worldcup2014_nhoma": 50,  # homography-derived
    "soccernet_calibration_2023": 40,
}


def _label_quality(label_text: str) -> tuple[int, int]:
    """Return ``(n_visible, n_valid_in_bounds)`` for a YOLO Pose label line."""
    parts = label_text.split()
    n_kp_floats = len(parts) - 5
    if n_kp_floats <= 0 or n_kp_floats % 3 != 0:
        return 0, 0
    n_visible = 0
    n_in_bounds = 0
    for k in range(NUM_KEYPOINTS):
        try:
            x = float(parts[5 + 3 * k])
            y = float(parts[5 + 3 * k + 1])
            v = float(parts[5 + 3 * k + 2])
        except (ValueError, IndexError):
            continue
        if v > 0.5:
            n_visible += 1
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                n_in_bounds += 1
    return n_visible, n_in_bounds


def _merge_staging_into_unified(
    staging_root: Path,
    unified_root: Path,
    *,
    keep_source_splits: bool = True,
    seed: int = 42,
    min_visible_kp: int = 4,
) -> dict[str, int]:
    """Merge ``staging_root/<source>`` into the unified dataset.

    Adds two features over a naive copy:

    * **Quality filter** — drops labels with fewer than ``min_visible_kp``
      keypoints whose visibility flag is set and whose ``(x, y)`` falls
      inside ``[0, 1]^2``.
    * **Source fusion** — when the same image stem is annotated by multiple
      sources we keep the label from the source with the highest
      :data:`SOURCE_FUSION_PRIORITY`; the others are dropped silently.
      Images keep the prefix ``<best_source>__`` in the merged dataset.
    """
    counts = {"train": 0, "val": 0, "test": 0}
    skipped_low_quality = 0
    fused_drops = 0

    def _physical_key(p: Path) -> str:
        """Identity of the *underlying* image file.

        We use the symlink target (or absolute path for real files); two
        staging entries pointing at the same physical jpg are deduplicated.
        Sources that stage independent files keep distinct keys even when
        the basename collides (e.g. Soccana ``00000.jpg`` vs WC14 ``1.jpg``).
        """
        try:
            return str(p.resolve())
        except OSError:
            return str(p.absolute())

    # First pass: collect all candidates with quality + priority.
    candidates: dict[str, dict[str, object]] = {}
    for src_dir in sorted(staging_root.iterdir()):
        if not src_dir.is_dir():
            continue
        prio = SOURCE_FUSION_PRIORITY.get(src_dir.name, 10)
        for split in ("train", "val", "test"):
            img_dir = src_dir / "images" / split
            lbl_dir = src_dir / "labels" / split
            if not img_dir.is_dir():
                continue
            for img_path in _all_image_files(img_dir):
                lbl_src = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_src.exists():
                    continue
                try:
                    label_text = lbl_src.read_text(encoding="utf-8").strip()
                except OSError:
                    continue
                if not label_text:
                    continue
                first_line = label_text.splitlines()[0]
                n_visible, n_in_bounds = _label_quality(first_line)
                if n_visible < min_visible_kp or n_in_bounds < min_visible_kp:
                    skipped_low_quality += 1
                    continue
                fusion_key = _physical_key(img_path)
                cand = candidates.get(fusion_key)
                if cand is None or int(cast(int, cand["priority"])) < prio:
                    if cand is not None:
                        fused_drops += 1
                    candidates[fusion_key] = {
                        "img_path": img_path,
                        "lbl_src": lbl_src,
                        "src_name": src_dir.name,
                        "priority": prio,
                        "split": split,
                        "n_visible": n_visible,
                    }
                else:
                    fused_drops += 1

    # Second pass: write the survivors.
    for cand in candidates.values():
        img_path = cast(Path, cand["img_path"])
        lbl_src = cast(Path, cand["lbl_src"])
        src_name = cast(str, cand["src_name"])
        split = cast(str, cand["split"])
        target_split = split
        if not keep_source_splits:
            rng = random.Random((seed * 1_000_003) ^ hash(img_path.name))
            target_split = rng.choices(["train", "val", "test"], weights=[0.85, 0.1, 0.05])[0]
        new_stem = f"{src_name}__{img_path.stem}"
        out_img_dir = _ensure_dir(unified_root / "images" / target_split)
        out_lbl_dir = _ensure_dir(unified_root / "labels" / target_split)
        _link_or_copy(img_path, out_img_dir / f"{new_stem}{img_path.suffix.lower()}")
        shutil.copy2(lbl_src, out_lbl_dir / f"{new_stem}.txt")
        counts[target_split] += 1
    counts["_skipped_low_quality"] = skipped_low_quality
    counts["_fused_drops"] = fused_drops
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


def _write_keypoint_reference(unified_root: Path) -> Path:
    """Write canonical 32-kp reference in **all** the relevant frames.

    Columns:

    * ``idx``, ``canonical_name``, ``flip_idx`` — from the canonical schema.
    * ``x_roboflow_cm``, ``y_roboflow_cm`` — Roboflow truth (origin top-left,
      Y-down, 12000×7000 cm).
    * ``x_norm``, ``y_norm`` — normalised in ``[0,1]`` (Y-down).
    * ``x_center_m``, ``y_center_m``, ``z_center_m`` — drawsportsfields convention
      (origin field center, Y-up, 104.9×67.9 m).
    * ``x_template_yard``, ``y_template_yard`` — KpSFR template (yards, Y-down,
      origin top-left, 114.83×74.37 yd).
    """
    ref_path = unified_root / "keypoints_reference_drawsportsfields.csv"
    cm_pts = _canonical_vertices_cm()
    norm_pts = _canonical_vertices_normalized()
    centered_pts, length_m, width_m = _load_centered_fifa_points_32()
    rows: list[dict[str, object]] = []
    for i, ((x_cm, y_cm), (nx, ny), (x_m, y_m, z_m)) in enumerate(
        zip(cm_pts, norm_pts, centered_pts, strict=True)
    ):
        x_tpl, y_tpl = _centered_meters_to_kpsfr_template(
            x_m, y_m, length_m=length_m, width_m=width_m
        )
        rows.append(
            {
                "idx": i,
                "canonical_name": CANONICAL_KP_NAMES_32[i],
                "flip_idx": CANONICAL_FLIP_IDX_32[i],
                "x_roboflow_cm": f"{x_cm:.2f}",
                "y_roboflow_cm": f"{y_cm:.2f}",
                "x_norm": f"{nx:.6f}",
                "y_norm": f"{ny:.6f}",
                "x_center_m": f"{x_m:.6f}",
                "y_center_m": f"{y_m:.6f}",
                "z_center_m": f"{z_m:.6f}",
                "x_template_yard": f"{x_tpl:.6f}",
                "y_template_yard": f"{y_tpl:.6f}",
            }
        )
    with ref_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "canonical_name",
                "flip_idx",
                "x_roboflow_cm",
                "y_roboflow_cm",
                "x_norm",
                "y_norm",
                "x_center_m",
                "y_center_m",
                "z_center_m",
                "x_template_yard",
                "y_template_yard",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return ref_path


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
            elif src.kind == "url_archive":
                _download_url_archive(src.url, download_path)
            elif src.kind == "soccernet":
                if soccernet_dir is not None and soccernet_dir.exists():
                    log(f"      reusing existing SoccerNet dir: {soccernet_dir}")
                    if not download_path.exists():
                        os.symlink(soccernet_dir.resolve(), download_path)
                    _extract_soccernet_zips(download_path, splits=tuple(soccernet_splits))
                else:
                    _download_soccernet_calibration(download_path, splits=tuple(soccernet_splits))
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
    if unified_dir.exists():
        shutil.rmtree(unified_dir)
    _ensure_dir(unified_dir)
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
    kp_ref = _write_keypoint_reference(unified_dir)
    log(f"Wrote: {data_yaml}")
    log(f"Wrote: {manifest}")
    log(f"Wrote: {kp_ref}")

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
            "Skip the build and only render --preview samples from an existing unified/ directory."
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

    # Group by source so the QA preview always covers every dataset, not just
    # the largest one (Soccana otherwise dominates random sampling).
    by_source: dict[str, list[Path]] = {}
    for p in candidates:
        src = p.stem.split("__", 1)[0] if "__" in p.stem else "_other"
        by_source.setdefault(src, []).append(p)
    for lst in by_source.values():
        rng.shuffle(lst)
    sources = sorted(by_source)
    # Round-robin pick: at least one per source, then top up.
    ordered: list[Path] = []
    while len(ordered) < n_samples and any(by_source[s] for s in sources):
        for s in sources:
            if by_source[s] and len(ordered) < n_samples:
                ordered.append(by_source[s].pop())
    drawn = 0
    for lbl_path in ordered:
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
