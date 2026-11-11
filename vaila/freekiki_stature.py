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
Version: 0.4.16
Created: 27 October 2026
Update Date: 11 November 2026

Description:
    The SAM 3D Body skeleton keeps its own metric size, which comes out short (about 1.2 to 1.5 m
    for a standing adult). Each track is therefore scaled on its vertical axis so that the nose,
    measured as the median over the track, sits at the stature prior minus the nose-to-vertex
    offset. The scale is one number per track, so the body keeps its shape; the feet stay on
    z = 0 and the x-y positions do not change.

    The prior is an operational choice (the average stature given for the pitch players), not a
    measurement of each player. ``NOSE_BELOW_TOP_M`` is an assumption, not a validated value.

    Known statures (``--stature keeper=2.04,kicker=1.82,sam:39=2.04,default=1.83``): a person
    whose stature is given is scaled as a whole body about its ground point (x, y and z by the same
    factor), so the arms grow sideways too; the raw height is then a high quantile of the nose (a
    keeper crouched in the ready stance most of the clip would make the median too low). People
    without a given stature keep the prior and the vertical-only scale above.
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


ROLES = ("keeper", "kicker")
STATURE_RANGE_M = (1.4, 2.3)  # a given stature outside this is a typing error


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
    stature_target_m: float = STATURE_M
    stature_source: str = "prior"  # "prior", a role ("keeper", "kicker") or "sam:<id>"
    isotropic: bool = False  # True: x, y and z scaled about the ground point


def parse_stature_spec(text: str | None) -> dict[str, float]:
    """``"keeper=2.04,kicker=1.82,sam:39=2.04,default=1.83"`` -> ``{key: metres}``.

    Keys: a role (``keeper``, ``kicker``), ``sam:<id>`` (a SAM person id) or ``default`` (the prior
    of everyone else). Raises ``ValueError`` on an unknown key, a missing ``=`` or a stature outside
    :data:`STATURE_RANGE_M`.
    """
    out: dict[str, float] = {}
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key = key.strip().lower()
        if not sep:
            raise ValueError(f"stature '{item}': use key=metres (e.g. keeper=2.04)")
        valid = key in ROLES or key == "default"
        if key.startswith("sam:"):
            valid = key[4:].isdigit()
        if not valid:
            raise ValueError(f"stature key '{key}': use keeper, kicker, sam:<id> or default")
        try:
            metres = float(value)
        except ValueError:
            raise ValueError(f"stature '{item}': '{value}' is not a number of metres") from None
        low, high = STATURE_RANGE_M
        if not low <= metres <= high:
            raise ValueError(f"stature '{item}': {metres} m is outside {low}-{high} m")
        out[key] = metres
    return out


def resolve_statures(
    spec: dict[str, float], roles: dict[str, int] | None
) -> tuple[dict[int, tuple[float, str]], float]:
    """The given statures by SAM person id, and the prior of everyone else.

    ``roles`` maps a role to its SAM id (from ``player_map.csv``; empty before the roles exist).
    An explicit ``sam:<id>`` wins over a role on the same person.
    """
    by_person: dict[int, tuple[float, str]] = {}
    for role in ROLES:
        if role in spec and roles and roles.get(role) is not None:
            by_person[int(roles[role])] = (float(spec[role]), role)
    for key, metres in spec.items():
        if key.startswith("sam:"):
            by_person[int(key[4:])] = (float(metres), key)
    return by_person, float(spec.get("default", STATURE_M))


def nose_heights(pose_rows: list[dict]) -> dict[int, list[float]]:
    """Raw nose heights (z, metres) of every person, one value per frame."""
    heights: dict[int, list[float]] = {}
    for row in pose_rows:
        if row["kpt_name"] == "nose" and np.isfinite(row["z_m"]):
            heights.setdefault(int(row["person_id"]), []).append(float(row["z_m"]))
    return heights


CHAIN_PARTS = (
    "left-ankle",
    "right-ankle",
    "left-knee",
    "right-knee",
    "left-hip",
    "right-hip",
    "left-shoulder",
    "right-shoulder",
    "nose",
)
ANKLE_ABOVE_SOLE_M = 0.08


def _skeletal_chain_heights(pose_rows: list[dict]) -> dict[int, list[float]]:
    """Geodesic leg+trunk+head length (ankle -> knee -> hip -> shoulder -> nose) per person/frame.

    A goalkeeper who stays crouched in the ready stance for an entire short penalty clip has a low
    vertical ``nose`` height (~1.05 m) even though his 3D segment lengths are full-sized (~1.9 m).
    """
    by_pf: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for row in pose_rows:
        name = str(row["kpt_name"])
        if name not in CHAIN_PARTS:
            continue
        xyz = np.array([row["x_m"], row["y_m"], row["z_m"]], dtype=float)
        if np.isfinite(xyz).all():
            by_pf.setdefault((int(row["person_id"]), int(row["frame"])), {})[name] = xyz
    out: dict[int, list[float]] = {}
    for (pid, _), pts in sorted(by_pf.items()):
        if not all(k in pts for k in CHAIN_PARTS):
            continue
        shank = 0.5 * (
            float(np.linalg.norm(pts["left-knee"] - pts["left-ankle"]))
            + float(np.linalg.norm(pts["right-knee"] - pts["right-ankle"]))
        )
        thigh = 0.5 * (
            float(np.linalg.norm(pts["left-hip"] - pts["left-knee"]))
            + float(np.linalg.norm(pts["right-hip"] - pts["right-knee"]))
        )
        hip_mid = 0.5 * (pts["left-hip"] + pts["right-hip"])
        sh_mid = 0.5 * (pts["left-shoulder"] + pts["right-shoulder"])
        trunk = float(np.linalg.norm(sh_mid - hip_mid))
        neck_head = float(np.linalg.norm(pts["nose"] - sh_mid))
        out.setdefault(pid, []).append(ANKLE_ABOVE_SOLE_M + shank + thigh + trunk + neck_head)
    return out


def apply_stature_scale(
    pose_rows: list[dict],
    *,
    stature_m: float = STATURE_M,
    nose_below_top_m: float = NOSE_BELOW_TOP_M,
    by_person: dict[int, tuple[float, str]] | None = None,
    ground: dict[tuple[int, int], tuple[float, float]] | None = None,
    height_quantile: float = 0.5,
) -> list[StatureScale]:
    """Scale every pose row in place, one factor per person; returns the evidence.

    Without ``by_person`` (the default) each person's ``z_m`` is scaled so that the median nose sits
    at ``stature_m`` minus :data:`NOSE_BELOW_TOP_M`, as before. ``by_person`` maps a SAM id to its
    known ``(stature_m, source)``: those people use ``height_quantile`` of the nose for the raw
    height and, when ``ground`` (``(frame, person_id) -> (x, y)``, the person's ground point) is
    given, x and y are scaled about that point too (the whole body keeps its shape).

    A track whose nose height is below :data:`MIN_NOSE_M` (never standing up in the record) is
    left as it is and flagged. Rows of any keypoint, the ground marker included, are scaled the
    same way (the ground marker is at z = 0, so it stays there).
    """
    known = by_person or {}
    heights = nose_heights(pose_rows)
    chains = _skeletal_chain_heights(pose_rows) if known else {}
    scales: dict[int, StatureScale] = {}
    for pid, values in sorted(heights.items()):
        target_stature, source = known.get(pid, (stature_m, "prior"))
        quantile = height_quantile if pid in known else 0.5
        raw = float(np.quantile(values, quantile))
        if pid in known and pid in chains and chains[pid]:
            raw = max(raw, float(np.median(chains[pid])))
        isotropic = pid in known and ground is not None
        target = target_stature - nose_below_top_m
        if raw < MIN_NOSE_M:
            scales[pid] = StatureScale(
                person_id=pid,
                frames=len(values),
                nose_median_raw_m=raw,
                scale=None,
                stature_before_m=raw + nose_below_top_m,
                stature_after_m=None,
                flag="not standing: no scale applied",
                stature_target_m=target_stature,
                stature_source=source,
                isotropic=False,
            )
            continue
        factor = target / raw
        after = (raw * factor) + nose_below_top_m
        low, high = PLAUSIBLE_SCALE
        flag = "ok" if low <= factor <= high else "check the track"
        scales[pid] = StatureScale(
            person_id=pid,
            frames=len(values),
            nose_median_raw_m=raw,
            scale=factor,
            stature_before_m=raw + nose_below_top_m,
            stature_after_m=after,
            flag=flag,
            stature_target_m=target_stature,
            stature_source=source,
            isotropic=isotropic,
        )
    for row in pose_rows:
        pid = int(row["person_id"])
        scale = scales.get(pid)
        if scale is None or scale.scale is None or not np.isfinite(row["z_m"]):
            continue
        if scale.isotropic and ground is not None:
            centre = ground.get((int(row["frame"]), pid))
            if centre is not None:
                row["x_m"] = centre[0] + (float(row["x_m"]) - centre[0]) * scale.scale
                row["y_m"] = centre[1] + (float(row["y_m"]) - centre[1]) * scale.scale
        row["z_m"] = float(row["z_m"]) * scale.scale
    return list(scales.values())
