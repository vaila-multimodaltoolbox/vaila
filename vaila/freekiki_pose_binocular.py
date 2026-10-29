"""
================================================================================
Script: freekiki_pose_binocular.py - people's keypoints triangulated from two cameras
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 28 October 2026
Update Date: 28 October 2026

Description:
    Triangulates every keypoint of every paired person (the ``player_id`` of ``player_map.csv``)
    from the two cameras, frame by frame, with each camera's DLT3D. The frames of the reference
    camera are the output frames; the other camera is read at the same instant (``capture_time``
    and ``cut_frame_at`` of :mod:`freekiki_stereo`). On a frame that the Vitruvian refit covers
    (``vitruvian.dlt3d``) that DLT is used, otherwise the run's own ``.dlt3d``.

    The 2D keypoints are the reprojections that SAM 3D Body makes of each camera's own mesh. A
    triangulated point therefore fuses two models; it is not an independent measurement. The
    residual of each view is the quality control: a point is valid only when both residuals are
    within :data:`MAX_KP_RESIDUAL_PX` and the two rays are at least ``MIN_ANGLE_DEG`` apart.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .freekiki_field3d import dlt_for_frames
    from .freekiki_stature import NOSE_BELOW_TOP_M
    from .freekiki_stereo import MIN_ANGLE_DEG, CamView, capture_time, cut_frame_at, triangulate
except ImportError:
    from freekiki_field3d import dlt_for_frames  # ty: ignore[unresolved-import]
    from freekiki_stature import NOSE_BELOW_TOP_M  # ty: ignore[unresolved-import]
    from freekiki_stereo import (  # ty: ignore[unresolved-import]
        MIN_ANGLE_DEG,
        CamView,
        capture_time,
        cut_frame_at,
        triangulate,
    )

KPT_COUNT = 70  # the MHR70 keypoints of SAM 3D Body (indices 0-69)
MAX_KP_RESIDUAL_PX = (
    10.0  # operational choice: a keypoint off by more than this in a view is not used
)
POSE_COLUMNS = [
    "frame",
    "time_s",
    "player_id",
    "kpt_idx",
    "kpt_name",
    "x_m",
    "y_m",
    "z_m",
    "resid_ref_px",
    "resid_other_px",
    "angle_deg",
    "dlt_ref",
    "dlt_other",
    "valid",
    "reason",
]


@dataclass(frozen=True)
class TrackSpan:
    """One track of a camera's ``player_map.csv``: the SAM id that is ``player_id`` on these frames."""

    player_id: int
    sam_id: int
    first: int
    last: int


@dataclass
class Keypoints:
    """2D keypoints of one camera: ``names[k]`` is the name of index ``k``; ``by_person[sam]`` is
    ``(n_frames, KPT_COUNT, 2)`` in pixels, NaN where the keypoint is missing."""

    names: list[str]
    by_person: dict[int, np.ndarray]


def read_tracks(path: Path | str) -> list[TrackSpan]:
    """The tracks of one camera (one row per track of ``player_map.csv``)."""
    out: list[TrackSpan] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(
                TrackSpan(
                    player_id=int(row["player_id"]),
                    sam_id=int(row["sam_id"]),
                    first=int(row["first_frame"]),
                    last=int(row["last_frame"]),
                )
            )
    return out


def read_keypoints2d(path: Path | str, n_frames: int) -> Keypoints:
    """The ``*_sam3dinov3_keypoints2d.csv`` of one camera, the MHR70 keypoints only."""
    names = [""] * KPT_COUNT
    by_person: dict[int, np.ndarray] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            k = int(row["kpt_idx"])
            f = int(row["frame"])
            if k >= KPT_COUNT or not 0 <= f < n_frames:
                continue
            names[k] = row["kpt_name"]
            arr = by_person.setdefault(
                int(row["person_id"]), np.full((n_frames, KPT_COUNT, 2), np.nan)
            )
            arr[f, k] = (float(row["x_px"]), float(row["y_px"]))
    return Keypoints(names=names, by_person=by_person)


def _pixels_at(arr: np.ndarray, frame: float) -> np.ndarray:
    """Keypoints of a fractional frame, linear in time; NaN outside the record."""
    lo = int(np.floor(frame))
    w = frame - lo
    if lo < 0 or lo >= len(arr):
        return np.full(arr.shape[1:], np.nan)
    if abs(w) < 1e-9:
        return arr[lo]
    if lo + 1 >= len(arr):
        return np.full(arr.shape[1:], np.nan)
    return (1.0 - w) * arr[lo] + w * arr[lo + 1]


def _covering(tracks: list[TrackSpan], player_id: int, frame: float) -> TrackSpan | None:
    for t in tracks:
        if t.player_id == player_id and t.first - 0.5 <= frame <= t.last + 0.5:
            return t
    return None


def _dlt_at(dlt: dict[int, np.ndarray], frame: float) -> np.ndarray | None:
    return dlt.get(int(round(frame)))


def triangulate_people(
    ref: CamView,
    other: CamView,
    *,
    ref_tracks: list[TrackSpan],
    other_tracks: list[TrackSpan],
    ref_kps: Keypoints,
    other_kps: Keypoints,
    dlt_source_ref: dict[int, str] | None = None,
    dlt_source_other: dict[int, str] | None = None,
    min_angle_deg: float = MIN_ANGLE_DEG,
    max_residual_px: float = MAX_KP_RESIDUAL_PX,
) -> list[dict]:
    """One row per (reference frame, paired person, keypoint) with its triangulation and quality.

    ``ref.dlt`` and ``other.dlt`` must already hold the DLT of every frame to use (the Vitruvian
    frames replace the run's ones; ``dlt_source_*`` records which was used). A person is taken on a
    reference frame when one of its ``ref_tracks`` covers that frame; the other camera is read at
    the same instant, and the other track of the same ``player_id`` must cover that instant too.
    Rows that cannot be triangulated keep the reason; nothing is filled in.
    """
    rows: list[dict] = []
    n_ref = int(ref.pts_cut.size)
    source_ref = dlt_source_ref or {}
    source_other = dlt_source_other or {}
    player_ids = sorted({t.player_id for t in ref_tracks})
    for frame in range(n_ref):
        time_s = capture_time(ref, frame)
        frame_other = cut_frame_at(other, time_s)
        for pid in player_ids:
            track_ref = _covering(ref_tracks, pid, frame)
            if track_ref is None:
                continue
            sam_ref = track_ref.sam_id
            arr_ref = ref_kps.by_person.get(sam_ref)
            base = {
                "frame": frame,
                "time_s": round(time_s, 6),
                "player_id": pid,
                "dlt_ref": source_ref.get(frame, "run"),
            }
            if frame_other is None:
                rows += _missing_rows(base, "outside the other camera")
                continue
            track_other = _covering(other_tracks, pid, frame_other)
            if track_other is None:
                rows += _missing_rows(base, "not paired on this instant")
                continue
            arr_other = other_kps.by_person.get(track_other.sam_id)
            L_ref = _dlt_at(ref.dlt, frame)
            L_other = _dlt_at(other.dlt, frame_other)
            other_src = source_other.get(int(round(frame_other)), "run")
            base["dlt_other"] = other_src
            if arr_ref is None or arr_other is None:
                rows += _missing_rows(base, "no keypoints")
                continue
            if L_ref is None or L_other is None:
                rows += _missing_rows(base, "no DLT on this frame")
                continue
            uv_other_all = _pixels_at(arr_other, frame_other)
            uv_ref_all = arr_ref[frame]
            for k in range(KPT_COUNT):
                name = ref_kps.names[k] or other_kps.names[k]
                row = dict(base, kpt_idx=k, kpt_name=name)
                uv_r, uv_o = uv_ref_all[k], uv_other_all[k]
                if not (np.isfinite(uv_r).all() and np.isfinite(uv_o).all()):
                    row.update(_blank(), valid=False, reason="no keypoint in a view")
                    rows.append(row)
                    continue
                xyz, res_r, res_o, angle = triangulate(L_ref, L_other, uv_r, uv_o)
                reason = _reason(res_r, res_o, angle, min_angle_deg, max_residual_px)
                row.update(
                    x_m=float(xyz[0]),
                    y_m=float(xyz[1]),
                    z_m=float(xyz[2]),
                    resid_ref_px=round(res_r, 3),
                    resid_other_px=round(res_o, 3),
                    angle_deg=round(angle, 3),
                    valid=reason == "",
                    reason=reason,
                )
                rows.append(row)
    return rows


def _reason(res_r: float, res_o: float, angle: float, min_angle: float, max_px: float) -> str:
    if not np.isfinite([res_r, res_o, angle]).all():
        return "triangulation failed"
    if max(res_r, res_o) > max_px:
        return f"residual above {max_px:g} px"
    if angle < min_angle:
        return f"rays {angle:.1f} deg apart (below {min_angle:g})"
    return ""


def _blank() -> dict:
    return {
        "x_m": np.nan,
        "y_m": np.nan,
        "z_m": np.nan,
        "resid_ref_px": np.nan,
        "resid_other_px": np.nan,
        "angle_deg": np.nan,
    }


def _missing_rows(base: dict, reason: str) -> list[dict]:
    """The one row that says why a person has no keypoints on a frame (no fake 0 points)."""
    row = dict(base, kpt_idx=-1, kpt_name="", valid=False, reason=reason, **_blank())
    row.setdefault("dlt_other", "")
    return [row]


def write_poses(rows: list[dict], path: Path | str) -> Path:
    """``poses_3d_binocular.csv``: the metres of the valid points and the reason of the others."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=POSE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in POSE_COLUMNS}
            for key in ("x_m", "y_m", "z_m", "resid_ref_px", "resid_other_px", "angle_deg"):
                value = out[key]
                out[key] = (
                    "" if value == "" or not np.isfinite(float(value)) else round(float(value), 6)
                )
            writer.writerow(out)
    return path


def summarize(rows: list[dict]) -> dict:
    """Counts and medians of the residuals, per person, for the README and the summary."""
    per: dict[int, dict] = {}
    for row in rows:
        if row.get("kpt_idx", -1) == -1:
            continue
        item = per.setdefault(int(row["player_id"]), {"points": 0, "valid": 0, "resid": []})
        item["points"] += 1
        if row["valid"]:
            item["valid"] += 1
            item["resid"].append(max(float(row["resid_ref_px"]), float(row["resid_other_px"])))
    out = {}
    for pid, item in sorted(per.items()):
        resid = item["resid"]
        out[str(pid)] = {
            "points": item["points"],
            "valid": item["valid"],
            "valid_share": round(item["valid"] / item["points"], 3) if item["points"] else None,
            "median_worst_residual_px": round(float(np.median(resid)), 3) if resid else None,
        }
    return out


def height_check(rows: list[dict]) -> dict[str, float]:
    """A control of the stature from the triangulated points: median (nose height above the mean
    ankle, plus the nose-to-top prior of :data:`freekiki_stature.NOSE_BELOW_TOP_M`). Not used to
    scale anything; it is reported next to the 1,83 m prior."""
    by: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        if not row.get("valid") or row.get("kpt_name") not in ("nose", "left-ankle", "right-ankle"):
            continue
        by.setdefault((int(row["player_id"]), int(row["frame"])), {})[row["kpt_name"]] = float(
            row["z_m"]
        )
    heights: dict[str, list[float]] = {}
    for (pid, _), vals in by.items():
        if {"nose", "left-ankle", "right-ankle"} <= vals.keys():
            ankle = 0.5 * (vals["left-ankle"] + vals["right-ankle"])
            heights.setdefault(str(pid), []).append(vals["nose"] - ankle + NOSE_BELOW_TOP_M)
    return {pid: round(float(np.median(v)), 3) for pid, v in heights.items() if v}


def load_vitruvian(path: Path | str) -> dict[int, np.ndarray]:
    """Frame -> DLT of the Vitruvian refit (``vitruvian.dlt3d``); {} when the file is missing."""
    p = Path(path)
    return dlt_for_frames(p) if p.is_file() else {}
