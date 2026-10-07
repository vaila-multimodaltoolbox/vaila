"""
================================================================================
Script: soccernet.py - kiki49_build SoccerNet line annotations to samples
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 06 October 2026
Update Date: 06 October 2026

Ported from mkvis3d ``openbiomech/soccer_field`` (the builder of the kiki49
dataset) so FreeKiki can convert external field datasets by itself.

Description:
    SoccerNet calibration-2023: human line annotations -> kiki49.

    Official train/valid/test splits; the group is the match (``match_info.json``).
    Planar points come from the annotation intersections and the fitted metric
    homography; posts tops are annotated where the crossbar is. The z > 0 / net
    points need a camera, estimated from the homography; images with annotated
    post tops measure how far that camera is from the truth (``post_top_err``).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import DATASET_ROOT, Options
from .camera import project
from .kiki49 import load_kiki49
from .label_io import MIN_VISIBLE
from .sample import Reject, Sample, camera_from_plane, complete_keypoints, px_scale
from .soccernet_lines import parse_annotation

SOURCE = "soccernet"
ROOT = DATASET_ROOT / "soccernet_calibration_2023" / "calibration-2023"
SPLITS = {"train": "train", "valid": "val", "test": "test"}
POST_TOPS = (34, 35, 42, 43)

MAX_MEDIAN_RES = 3.0  # px at 1280: annotation vs fitted homography
MAX_P90_RES = 8.0
MIN_REDUNDANCY = 4  # residual scalars beyond the 8 homography unknowns
MAX_CAMERA_DISAGREEMENT = 2.0  # px at 1280: pinhole camera vs homography
MAX_POST_TOP_ERR = 8.0  # px at 1280: camera vs annotated post tops


def tasks(limit: int | None) -> list[tuple[str, str, str, str]]:
    out = []
    for folder, split in SPLITS.items():
        d = ROOT / folder
        info = json.loads((d / "match_info.json").read_text())
        n = 0
        for j in sorted(d.glob("*.json")):
            if not j.stem.isdigit() or not j.with_suffix(".jpg").exists():
                continue
            m = info.get(f"{j.stem}.jpg", {})
            group = "|".join(
                str(m.get(k, "")).strip() for k in ("league", "season", "match", "date")
            )
            out.append((str(j), split, group or f"unknown:{folder}:{j.stem}", folder))
            n += 1
            if limit is not None and n >= limit:
                break
    return out


def sample_from_lines(
    annotation: dict,
    width: int,
    height: int,
    *,
    source: str,
    uid: str,
    split: str,
    group: str,
    image: Path | None,
    opts: Options,
    seed_points: dict | None = None,
) -> Sample | Reject:
    """SoccerNet line annotation (calibration / game-state format) -> kiki49 sample.

    ``seed_points`` (network keypoints) only anchor the plane; see
    ``soccernet_lines.parse_annotation``.
    """
    lab = parse_annotation(annotation, width, height, seed_points)
    stats = dict(lab.stats)
    if lab.H is None:
        return Reject(source, uid, "no_homography", stats)
    s = px_scale(width)
    res = lab.residual_px / s
    stats["res_med"] = float(np.median(res))
    stats["res_p90"] = float(np.percentile(res, 90))
    if len(res) < 8 + MIN_REDUNDANCY:
        return Reject(source, uid, "underdetermined", stats)
    if stats["res_med"] > MAX_MEDIAN_RES or stats["res_p90"] > MAX_P90_RES:
        return Reject(source, uid, "residual", stats)
    cam, disagreement = camera_from_plane(lab.H, width, height)
    if cam is None:
        return Reject(source, uid, "no_camera", stats)
    stats["cam_disagreement"] = disagreement

    aux3d = opts.aux3d and disagreement <= MAX_CAMERA_DISAGREEMENT
    tops = [k for k in POST_TOPS if lab.annotated[k, 2] > 0]
    if tops:
        uv, _ = project(load_kiki49().xyz[tops], cam)
        err = np.hypot(*(uv - lab.annotated[tops, :2]).T) / s
        stats["post_top_err"] = float(np.median(err))
        aux3d = aux3d and stats["post_top_err"] <= MAX_POST_TOP_ERR
    kps, origin = complete_keypoints(lab.annotated, width, height, lab.H, cam, aux3d)
    if int((kps[:, 2] > 0).sum()) < MIN_VISIBLE:
        return Reject(source, uid, "few_visible", stats)
    return Sample(
        source=source,
        uid=uid,
        split=split,
        group=group,
        width=width,
        height=height,
        kps=kps,
        origin=origin,
        image=image,
        H=lab.H,
        camera=cam,
        aux3d=aux3d,
        qa=stats["res_med"],
        stats=stats,
    )


def process(task: tuple[str, str, str, str], opts: Options) -> list[Sample | Reject]:
    path, split, group, folder = task
    j = Path(path)
    uid = f"{folder}_{j.stem}"
    img = cv2.imread(str(j.with_suffix(".jpg")), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None:
        return [Reject(SOURCE, uid, "unreadable_image")]
    height, width = img.shape[0] * 2, img.shape[1] * 2
    return [
        sample_from_lines(
            json.loads(j.read_text()),
            width,
            height,
            source=SOURCE,
            uid=uid,
            split=split,
            group=group,
            image=j.with_suffix(".jpg"),
            opts=opts,
        )
    ]
