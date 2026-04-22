"""Pure-NumPy primitives for football shot orientation metrics.

Inspired by the xGHub paper (CVIU):
    "Once Upon a Goal: Towards orientation-based shot metrics in football"
    https://www.sciencedirect.com/science/article/pii/S1077314226000998
    https://github.com/mguti97/xGHub

This module exposes four side-effect free helpers that other vaila modules
(``xghub_loader``, ``xghub_benchmark``, GUI overlays) can compose:

    * :func:`body_orientation_2d`     - 2D body azimuth from image-space joints.
    * :func:`body_orientation_3d`     - 3D body azimuth + elevation in a world frame.
    * :func:`goal_geometry`           - angle/aperture/distance to a goal segment.
    * :func:`relative_orientation_to_goal` - signed angle of body vs goal direction.

The xGHub ``tabular_data.json`` schema we mirror:

    angle_to_goal           goal_geometry(...)['angle_to_goal_deg']
    angle_to_posts          goal_geometry(...)['angle_to_posts_deg']
    distance_from_goal      goal_geometry(...)['distance_m']
    2d_orient_angle         body_orientation_2d(...)
    2d_orient_rel_angle     relative_orientation_to_goal(2d_orient, ...)
    3d_orient_angle         body_orientation_3d(...)['azimuth_deg']
    3d_orient_elev_angle    body_orientation_3d(...)['elevation_deg']
    3d_orient_rel_angle     relative_orientation_to_goal(3d_azimuth, ...)

Author: Paulo R. P. Santiago - vaila project
Created: 19 April 2026
License: AGPL-3.0-or-later
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "FifaJointIndices",
    "body_orientation_2d",
    "body_orientation_3d",
    "goal_geometry",
    "relative_orientation_to_goal",
    "wrap_angle_deg",
]


@dataclass(frozen=True)
class FifaJointIndices:
    """Joint indices for the 15-joint FIFA Skeletal Tracking Light layout.

    Matches ``OPENPOSE_TO_OURS`` in ``vaila.fifa_skeletal_pipeline``:
    nose=0, right_shoulder=1, left_shoulder=2, right_hip=7, left_hip=8.
    """

    nose: int = 0
    right_shoulder: int = 1
    left_shoulder: int = 2
    right_hip: int = 7
    left_hip: int = 8


_FIFA = FifaJointIndices()


def wrap_angle_deg(angle_deg: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle (degrees) to the half-open interval ``(-180, 180]``."""
    a = np.asarray(angle_deg, dtype=float)
    out = ((a + 180.0) % 360.0) - 180.0
    out = np.where(out <= -180.0, out + 360.0, out)
    if np.ndim(angle_deg) == 0:
        return float(out)
    return out


def _coerce_skel(skel: np.ndarray, expected_dim: int) -> np.ndarray:
    arr = np.asarray(skel, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise ValueError(f"skel must have shape (J, {expected_dim}); got {arr.shape}")
    return arr


def body_orientation_2d(
    skel_2d: np.ndarray,
    joints: FifaJointIndices = _FIFA,
    *,
    image_y_down: bool = True,
) -> float:
    """Estimate the 2-D body azimuth (degrees) from image-space joints.

    Parameters
    ----------
    skel_2d:
        Array of shape ``(J, 2)`` with image coordinates ``(x, y)`` for at
        least the four indices stored in ``joints`` (shoulders + hips).
    joints:
        Joint index layout. Defaults to the 15-joint FIFA subset.
    image_y_down:
        If ``True`` (OpenCV/NumPy default), the y axis grows downward and we
        flip its sign so that the returned angle follows the standard math
        convention (counter-clockwise from +x). If ``False``, coordinates are
        already in math orientation.

    Returns
    -------
    float
        Body facing angle in degrees, wrapped to ``(-180, 180]``. The angle
        is the math-frame direction of the shoulder-hip lateral vector
        rotated ``-90`` degrees (i.e., the direction reached by turning the
        lateral 90 degrees clockwise in math axes).

    Notes
    -----
    Pure 2-D pose is fundamentally ambiguous about whether the player is
    facing into or out of the image, so we adopt a stable convention:

    * facing the camera with shoulders horizontal returns ``-90`` when
      ``image_y_down`` is ``True`` (canonical OpenCV pixel layout) and
      ``+90`` when ``image_y_down`` is ``False``;
    * a player whose lateral runs along the image y axis (e.g., turned 90
      degrees CCW relative to the camera) returns ``0``.

    NaNs in any of the four required joints raise ``ValueError``.
    """
    arr = _coerce_skel(skel_2d, 2)

    rs = arr[joints.right_shoulder]
    ls = arr[joints.left_shoulder]
    rh = arr[joints.right_hip]
    lh = arr[joints.left_hip]
    if not np.all(np.isfinite([rs, ls, rh, lh])):
        raise ValueError("body_orientation_2d: shoulder/hip joints contain NaN")

    if image_y_down:
        rs = np.array([rs[0], -rs[1]])
        ls = np.array([ls[0], -ls[1]])
        rh = np.array([rh[0], -rh[1]])
        lh = np.array([lh[0], -lh[1]])

    shoulder_lat = ls - rs
    hip_lat = lh - rh
    lateral = (shoulder_lat + hip_lat) * 0.5
    if float(np.linalg.norm(lateral)) < 1e-9:
        raise ValueError("body_orientation_2d: degenerate shoulder/hip span")

    # Facing direction = lateral rotated -90° (so right→front when facing camera).
    facing = np.array([lateral[1], -lateral[0]])
    angle = float(np.degrees(np.arctan2(facing[1], facing[0])))
    return float(wrap_angle_deg(angle))


def body_orientation_3d(
    skel_3d: np.ndarray,
    joints: FifaJointIndices = _FIFA,
    *,
    up_axis: Literal["x", "y", "z"] = "z",
) -> dict[str, float]:
    """Estimate 3-D body azimuth and elevation (degrees) from world joints.

    Parameters
    ----------
    skel_3d:
        Array of shape ``(J, 3)`` with world coordinates of the joints.
    joints:
        Joint layout (defaults to the FIFA 15-joint subset).
    up_axis:
        Which world axis is "up". Determines the horizontal plane used for
        the azimuth and the vertical reference for the elevation.

    Returns
    -------
    dict with keys ``azimuth_deg`` (yaw of the body in the horizontal plane,
    in ``(-180, 180]``) and ``elevation_deg`` (signed body lean from the
    vertical, in ``(-90, 90]``; positive = leaning forward in the facing
    direction).
    """
    arr = _coerce_skel(skel_3d, 3)
    rs = arr[joints.right_shoulder]
    ls = arr[joints.left_shoulder]
    rh = arr[joints.right_hip]
    lh = arr[joints.left_hip]
    if not np.all(np.isfinite([rs, ls, rh, lh])):
        raise ValueError("body_orientation_3d: shoulder/hip joints contain NaN")

    axis_idx = {"x": 0, "y": 1, "z": 2}[up_axis]
    plane = [i for i in (0, 1, 2) if i != axis_idx]

    mid_shoulder = 0.5 * (rs + ls)
    mid_hip = 0.5 * (rh + lh)
    trunk = mid_shoulder - mid_hip
    if float(np.linalg.norm(trunk)) < 1e-9:
        raise ValueError("body_orientation_3d: degenerate trunk vector")

    lateral = (ls - rs) + (lh - rh)
    horizontal_lat = lateral.copy()
    horizontal_lat[axis_idx] = 0.0
    if float(np.linalg.norm(horizontal_lat)) < 1e-9:
        raise ValueError("body_orientation_3d: degenerate horizontal lateral")

    up = np.zeros(3)
    up[axis_idx] = 1.0
    facing = np.cross(horizontal_lat, up)
    if float(np.linalg.norm(facing)) < 1e-9:
        raise ValueError("body_orientation_3d: cannot derive facing direction")

    azimuth_deg = float(np.degrees(np.arctan2(facing[plane[1]], facing[plane[0]])))

    trunk_norm = trunk / np.linalg.norm(trunk)
    cos_lean = float(np.clip(np.dot(trunk_norm, up), -1.0, 1.0))
    lean_from_vertical = float(np.degrees(np.arccos(cos_lean)))
    facing_norm = facing / np.linalg.norm(facing)
    sign = 1.0 if float(np.dot(trunk_norm, facing_norm)) >= 0.0 else -1.0
    elevation_deg = float(sign * lean_from_vertical)

    return {
        "azimuth_deg": float(wrap_angle_deg(azimuth_deg)),
        "elevation_deg": float(np.clip(elevation_deg, -90.0, 90.0)),
    }


def goal_geometry(
    shooter_xy: np.ndarray,
    goal_left_post_xy: np.ndarray,
    goal_right_post_xy: np.ndarray,
) -> dict[str, float]:
    """Compute the xGHub goal-geometry triplet for a shot.

    All inputs are 2-D positions on the pitch plane (in metres, any axes
    convention, as long as they are consistent).

    Returns
    -------
    dict with:
        ``angle_to_goal_deg``  : direction from shooter to the goal centre,
                                 measured CCW from the +x axis, in ``(-180, 180]``.
        ``angle_to_posts_deg`` : angular aperture (>= 0) of the goal as seen
                                 from the shooter (``angle(post_left, post_right)``).
        ``distance_m``         : Euclidean distance to the goal centre.
    """
    s = np.asarray(shooter_xy, dtype=float).reshape(2)
    pl = np.asarray(goal_left_post_xy, dtype=float).reshape(2)
    pr = np.asarray(goal_right_post_xy, dtype=float).reshape(2)

    centre = 0.5 * (pl + pr)
    to_centre = centre - s
    distance = float(np.linalg.norm(to_centre))
    if distance < 1e-9:
        raise ValueError("goal_geometry: shooter coincides with goal centre")
    angle_to_goal = float(np.degrees(np.arctan2(to_centre[1], to_centre[0])))

    v_left = pl - s
    v_right = pr - s
    nl = float(np.linalg.norm(v_left))
    nr = float(np.linalg.norm(v_right))
    if nl < 1e-9 or nr < 1e-9:
        raise ValueError("goal_geometry: shooter on a goal post")
    cos_ap = float(np.clip(np.dot(v_left, v_right) / (nl * nr), -1.0, 1.0))
    angle_to_posts = float(np.degrees(np.arccos(cos_ap)))

    return {
        "angle_to_goal_deg": float(wrap_angle_deg(angle_to_goal)),
        "angle_to_posts_deg": angle_to_posts,
        "distance_m": distance,
    }


def relative_orientation_to_goal(
    body_angle_deg: float,
    shooter_xy: np.ndarray,
    goal_centre_xy: np.ndarray,
) -> float:
    """Body orientation expressed relative to the shooter→goal direction.

    Parameters
    ----------
    body_angle_deg:
        Absolute body azimuth (degrees, math convention) as returned by
        :func:`body_orientation_2d` or
        :func:`body_orientation_3d` (``azimuth_deg`` field).
    shooter_xy:
        2-D shooter position (same plane/units as ``goal_centre_xy``).
    goal_centre_xy:
        2-D goal centre position.

    Returns
    -------
    float
        Signed angle (degrees, ``(-180, 180]``) of the body direction
        relative to the shooter→goal line. ``0`` means the player is facing
        the goal; ``+90`` means the goal is on the player's right side;
        ``±180`` means the player has their back to the goal.
    """
    s = np.asarray(shooter_xy, dtype=float).reshape(2)
    g = np.asarray(goal_centre_xy, dtype=float).reshape(2)
    to_goal = g - s
    if float(np.linalg.norm(to_goal)) < 1e-9:
        raise ValueError("relative_orientation_to_goal: shooter at the goal")
    goal_angle = float(np.degrees(np.arctan2(to_goal[1], to_goal[0])))
    return float(wrap_angle_deg(float(body_angle_deg) - goal_angle))
