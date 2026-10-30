"""
================================================================================
Script: freekiki_com.py - the centre of mass of a person from the MHR70 keypoints
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 29 October 2026
Update Date: 29 October 2026

Description:
    The centre of mass (CoM) of one person per frame, from the 3D keypoints in pitch metres that
    the freekiki analysis already has (``freekiki_kicker.person_points``). It is the segment
    model of De Leva (1996) with the mass fractions the repository already uses for the 2D CoM of
    the deadlift (``vaila_deadlift._SEGMENT_MASS``), here with the MHR70 keypoint names:

    * head: the middle of the two ears;
    * trunk: from the middle of the shoulders to the middle of the hips;
    * each upper arm (shoulder to elbow), each forearm with the hand (elbow to wrist), each thigh
      (hip to knee), each shank (knee to ankle) and each foot (ankle to big toe).

    Each segment is taken at the middle of its two ends and weighted by its mass fraction. This is
    an operational definition: De Leva's own segment centres are not at 50% of the segment, and
    the hands are the wrists. A segment with a missing end is left out and the others are
    renormalised by the mass that is present; the CoM is NaN when the trunk is missing or when
    less than ``MIN_MASS`` of the body mass is present. The values carry the error of the keypoints.
"""

from __future__ import annotations

import numpy as np

# (segment, De Leva mass fraction, proximal keypoints, distal keypoints); a point made of several
# keypoints is their mean (the middle of the shoulders, of the hips)
SEGMENTS: tuple[tuple[str, float, tuple[str, ...], tuple[str, ...]], ...] = (
    ("head", 0.081, ("left-ear",), ("right-ear",)),
    ("trunk", 0.430, ("left-shoulder", "right-shoulder"), ("left-hip", "right-hip")),
    ("upper_arm_left", 0.027, ("left-shoulder",), ("left-elbow",)),
    ("upper_arm_right", 0.027, ("right-shoulder",), ("right-elbow",)),
    ("forearm_hand_left", 0.022, ("left-elbow",), ("left-wrist",)),
    ("forearm_hand_right", 0.022, ("right-elbow",), ("right-wrist",)),
    ("thigh_left", 0.142, ("left-hip",), ("left-knee",)),
    ("thigh_right", 0.142, ("right-hip",), ("right-knee",)),
    ("shank_left", 0.043, ("left-knee",), ("left-ankle",)),
    ("shank_right", 0.043, ("right-knee",), ("right-ankle",)),
    ("foot_left", 0.014, ("left-ankle",), ("left-big-toe-tip",)),
    ("foot_right", 0.014, ("right-ankle",), ("right-big-toe-tip",)),
)
TOTAL_MASS = sum(frac for _, frac, _, _ in SEGMENTS)
MIN_MASS = 0.6  # the CoM is not given when less than this share of the body mass is present


def _end(points: dict[str, np.ndarray], names: tuple[str, ...], n: int) -> np.ndarray:
    """The mean of the named keypoints per frame, ``(n, 3)``; NaN where any of them is missing."""
    arrays = [points.get(name) for name in names]
    if any(a is None for a in arrays):
        return np.full((n, 3), np.nan)
    return np.mean(np.stack([np.asarray(a, dtype=float) for a in arrays]), axis=0)


def com_track(points: dict[str, np.ndarray], *, min_mass: float = MIN_MASS) -> np.ndarray:
    """The CoM of one person per frame, ``(n, 3)`` in metres; NaN where it cannot be given.

    ``points`` is ``keypoint name -> (n, 3)`` (NaN where the point is not valid), as
    ``freekiki_kicker.person_points`` returns it.
    """
    if not points:
        return np.empty((0, 3))
    n = len(next(iter(points.values())))
    weighted = np.zeros((n, 3))
    present = np.zeros(n)
    trunk_ok = np.zeros(n, dtype=bool)
    for name, frac, proximal, distal in SEGMENTS:
        middle = (_end(points, proximal, n) + _end(points, distal, n)) / 2.0
        ok = np.isfinite(middle).all(axis=1)
        weighted[ok] += frac * middle[ok]
        present[ok] += frac
        if name == "trunk":
            trunk_ok = ok
    out = np.full((n, 3), np.nan)
    good = trunk_ok & (present / TOTAL_MASS >= min_mass)
    out[good] = weighted[good] / present[good, None]
    return out
