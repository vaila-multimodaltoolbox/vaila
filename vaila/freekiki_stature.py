"""
================================================================================
Script: freekiki_stature.py - vertical scale of the players from a stature prior
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 27 October 2026
Update Date: 27 October 2026

Description:
    The SAM 3D Body skeleton keeps its own metric size, which comes out short (about 1.2 to 1.5 m
    for a standing adult). Each track is therefore scaled on its vertical axis so that the nose,
    measured as the median over the track, sits at the stature prior minus the nose-to-vertex
    offset. The scale is one number per track, so the body keeps its shape; the feet stay on
    z = 0 and the x-y positions do not change.

    The prior is an operational choice (the average stature given for the pitch players), not a
    measurement of each player. ``NOSE_BELOW_TOP_M`` is an assumption, not a validated value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STATURE_M = 1.83  # average stature of the players (operational prior, not measured per player)
NOSE_BELOW_TOP_M = 0.10  # assumed distance from the nose to the top of the head
MIN_NOSE_M = 0.2  # a median nose height below this cannot be scaled (the track is not standing)
PLAUSIBLE_SCALE = (
    0.8,
    1.6,
)  # a factor outside this range means the raw skeleton is far off: flagged


@dataclass(frozen=True)
class StatureScale:
    """The scale of one track and the evidence it was computed from."""

    person_id: int
    frames: int
    nose_median_raw_m: float
    scale: float | None
    stature_before_m: float
    stature_after_m: float | None
    flag: str


def nose_heights(pose_rows: list[dict]) -> dict[int, list[float]]:
    """Raw nose heights (z, metres) of every person, one value per frame."""
    heights: dict[int, list[float]] = {}
    for row in pose_rows:
        if row["kpt_name"] == "nose" and np.isfinite(row["z_m"]):
            heights.setdefault(int(row["person_id"]), []).append(float(row["z_m"]))
    return heights


def apply_stature_scale(
    pose_rows: list[dict],
    *,
    stature_m: float = STATURE_M,
    nose_below_top_m: float = NOSE_BELOW_TOP_M,
) -> list[StatureScale]:
    """Scale ``z_m`` of every pose row in place, one factor per person; returns the evidence.

    A track whose median nose height is below :data:`MIN_NOSE_M` (never standing up in the
    record) is left as it is and flagged. Rows of any keypoint, the ground marker included, are
    scaled the same way (the ground marker is at z = 0, so it stays there).
    """
    target = stature_m - nose_below_top_m
    heights = nose_heights(pose_rows)
    scales: dict[int, StatureScale] = {}
    for pid, values in sorted(heights.items()):
        median = float(np.median(values))
        if median < MIN_NOSE_M:
            scales[pid] = StatureScale(
                person_id=pid,
                frames=len(values),
                nose_median_raw_m=median,
                scale=None,
                stature_before_m=median + nose_below_top_m,
                stature_after_m=None,
                flag="not standing: no scale applied",
            )
            continue
        factor = target / median
        after = (median * factor) + nose_below_top_m
        low, high = PLAUSIBLE_SCALE
        flag = "ok" if low <= factor <= high else "check the track"
        scales[pid] = StatureScale(
            person_id=pid,
            frames=len(values),
            nose_median_raw_m=median,
            scale=factor,
            stature_before_m=median + nose_below_top_m,
            stature_after_m=after,
            flag=flag,
        )
    for row in pose_rows:
        scale = scales.get(int(row["person_id"]))
        if scale is not None and scale.scale is not None and np.isfinite(row["z_m"]):
            row["z_m"] = float(row["z_m"]) * scale.scale
    return list(scales.values())
