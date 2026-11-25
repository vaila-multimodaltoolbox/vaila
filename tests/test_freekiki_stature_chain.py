"""Stature from the 2D chain of the SAM keypoints (vaila/freekiki_teams.py), on a synthetic skeleton.

A skeleton of known stature S, with the segment proportions of the chain model (Winter 2009), is
posed (walking, running, crouching, a keeper's ready stance), projected through a known broadcast-like
camera (90 m from the person, 16 m high, f = 5000 px: a person ~100 px tall) and measured again.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaila import freekiki_teams as T  # noqa: N812 (the short name keeps the tests readable)
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt

NAMES = [
    "nose", "left-shoulder", "right-shoulder", "left-hip", "right-hip", "left-knee", "right-knee",
    "left-ankle", "right-ankle", "left-heel", "right-heel",
]  # fmt: skip
W, H_IMG = 1920.0, 1080.0
UP = np.array([0.0, 0.0, 1.0])


def _look_at(centre, target, focal: float = 5000.0) -> tuple[np.ndarray, tuple]:
    """``(P 3x4, K R t decomposed as (K, R, t))`` of a camera at ``centre`` looking at ``target``."""
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, UP)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    K = np.array([[focal, 0.0, W / 2], [0.0, focal, H_IMG / 2], [0.0, 0.0, 1.0]])
    t = -R @ centre
    return K @ np.hstack([R, t[:, None]]), (K, R, t)


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


def _dir(heading: float, angle: float, sign: float = 1.0) -> np.ndarray:
    """Unit vector in the sagittal plane, ``angle`` radians forward of the vertical (``sign`` -1: down)."""
    forward = np.array([math.cos(heading), math.sin(heading), 0.0])
    return sign * math.cos(angle) * UP + math.sin(angle) * forward


POSTURES = {  # (hip flexion, knee flexion) of the right leg and of the left leg, trunk lean (rad)
    "stand": ((0.0, 0.0), (0.0, 0.0), 0.0),
    "crouch": ((1.2, 2.0), (1.2, 2.0), 0.7),  # sitting back, trunk over the knees
    "ready": ((0.6, 1.0), (0.6, 1.0), 0.45),  # a keeper's ready stance
}


def skeleton(S: float, foot, heading: float, posture: str, phase: float = 0.0) -> dict:
    """3D joints (m) of a person of stature ``S``: segments 0.245/0.246/0.288 S, nose 0.11 S."""
    thigh, shank, trunk = 0.245 * S, 0.246 * S, 0.288 * S
    fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
    side = np.cross(UP, fwd)
    if posture == "run":  # legs swing in opposite phase, the swing knee folds, the trunk leans
        s = math.sin(phase)
        right, left, lean = (0.9 * s, 1.4 * max(s, 0.0)), (-0.9 * s, 1.4 * max(-s, 0.0)), 0.25
    else:
        right, left, lean = POSTURES[posture]
    pts: dict[str, np.ndarray] = {}
    for name, sign, (flexion, knee) in (("right", -1.0, right), ("left", 1.0, left)):
        hip = 0.09 * S * sign * side  # the pelvis is the origin
        k = hip + thigh * _dir(heading, flexion, -1.0)
        a = k + shank * _dir(heading, flexion - knee, -1.0)
        pts[f"{name}-hip"], pts[f"{name}-knee"], pts[f"{name}-ankle"] = hip, k, a
    mid = (pts["right-hip"] + pts["left-hip"]) / 2
    td = _dir(heading, lean)
    shoulder = mid + trunk * td
    pts["right-shoulder"] = shoulder - 0.095 * S * side
    pts["left-shoulder"] = shoulder + 0.095 * S * side
    face = math.cos(lean) * fwd - math.sin(lean) * UP  # forward, perpendicular to the trunk axis
    pts["nose"] = shoulder + 0.095 * S * td + 0.055 * S * face
    for name in ("right", "left"):
        pts[f"{name}-heel"] = pts[f"{name}-ankle"] - np.array([0, 0, 0.03 * S]) - 0.05 * fwd
    low = min(pts["right-ankle"][2], pts["left-ankle"][2])  # the lower foot stands on the ground
    shift = np.array([foot[0], foot[1], 0.039 * S - low])
    return {k: v + shift for k, v in pts.items()}


def instance(P: np.ndarray, sam_id: int, joints: dict) -> dict:
    """A SAM-like prediction: ``keypoints_2d_px`` in MHR70 order (here ``NAMES``) and a padded box."""
    kp = np.array([_project(P, joints[n]) for n in NAMES])
    x1, y1 = kp.min(axis=0)
    x2, y2 = kp.max(axis=0)
    pad = 0.15 * (y2 - y1)
    return {
        "sam_obj_id": sam_id,
        "keypoints_2d_px": kp.tolist(),
        "bbox_xyxy": [x1 - pad, y1 - pad, x2 + pad, y2 + pad],
    }


def _track(
    S: float, posture: str, headings, P, foot=(0.0, 0.0, 0.0), noise_px: float = 0.0
) -> dict:
    rng = np.random.default_rng(3)
    frames: dict[int, list[dict]] = {}
    for i, heading in enumerate(headings):
        joints = skeleton(S, foot, float(heading), posture, phase=0.5 * i)
        inst = instance(P, 7, joints)
        if noise_px:
            inst["keypoints_2d_px"] = (
                np.asarray(inst["keypoints_2d_px"]) + rng.normal(0, noise_px, (len(NAMES), 2))
            ).tolist()
        frames[i] = [inst]
    return frames


CAMERA = _look_at([2.0, -90.0, 16.0], [0.0, 0.0, 0.0])[0]
SIDE_ON = np.linspace(-0.6, 0.6, 40)  # heading of the person: nearly across the view (x axis)
ALL_HEADINGS = np.linspace(0.0, 2 * math.pi, 41)[:-1]  # the person turns all the way round


@pytest.mark.parametrize("S", [1.62, 1.78, 1.94])
@pytest.mark.parametrize("posture", ["stand", "run", "crouch", "ready"])
def test_the_stature_is_recovered_within_3_percent_in_any_posture(S: float, posture: str) -> None:
    frames = _track(S, posture, SIDE_ON, CAMERA)
    result = T.stature_from_chain(dict.fromkeys(frames, CAMERA), frames, 7, NAMES)
    assert result["method"] == "chain-p80" and result["frames"] == len(frames)
    assert abs(result["height_m"] - S) / S < 0.03, (posture, result)


def test_a_person_turning_all_the_way_round_is_still_within_3_percent() -> None:
    frames = _track(1.80, "run", ALL_HEADINGS, CAMERA)
    result = T.stature_from_chain(dict.fromkeys(frames, CAMERA), frames, 7, NAMES)
    assert abs(result["height_m"] - 1.80) / 1.80 < 0.03, result


def test_the_chain_does_not_change_with_the_posture_the_vertical_span_does() -> None:
    standing = _track(1.80, "stand", [0.0], CAMERA)[0][0]
    crouched = _track(1.80, "crouch", [0.0], CAMERA)[0][0]
    idx = {n: i for i, n in enumerate(NAMES)}
    a = T.chain_length_px(np.asarray(standing["keypoints_2d_px"]), idx)
    b = T.chain_length_px(np.asarray(crouched["keypoints_2d_px"]), idx)
    assert a is not None and b is not None and abs(a[0] - b[0]) / a[0] < 0.02
    span = lambda inst: np.ptp(np.asarray(inst["keypoints_2d_px"])[:, 1])  # noqa: E731
    assert span(crouched) < 0.75 * span(standing)  # the old nose-to-heel span shrinks 25 %+


def test_a_noisy_small_person_100_px_tall_stays_within_3_percent() -> None:
    frames = _track(1.80, "run", SIDE_ON, CAMERA, noise_px=0.7)
    box = frames[0][0]["bbox_xyxy"]
    assert 90 <= box[3] - box[1] <= 130  # a broadcast-size person, padded box included
    result = T.stature_from_chain(dict.fromkeys(frames, CAMERA), frames, 7, NAMES)
    assert abs(result["height_m"] - 1.80) / 1.80 < 0.03, result


def test_the_camera_may_be_given_as_11_dlt_coefficients() -> None:
    _, (K, R, t) = _look_at([2.0, -90.0, 16.0], [0.0, 0.0, 0.0])
    L = compute_dlt3d_from_KRt(K, R, t)
    frames = _track(1.80, "stand", SIDE_ON, CAMERA)
    by_matrix = T.stature_from_chain(dict.fromkeys(frames, CAMERA), frames, 7, NAMES)
    by_dlt = T.stature_from_chain(dict.fromkeys(frames, L), frames, 7, NAMES)
    assert by_dlt["height_m"] == by_matrix["height_m"]


def test_the_vertical_factor_comes_from_the_camera_not_from_a_constant() -> None:
    low = _look_at([0.0, -90.0, 2.0], [0.0, 0.0, 0.0])[0]  # a camera 2 m high: ~1.0
    high = _look_at([0.0, -40.0, 40.0], [0.0, 0.0, 0.0])[0]  # 45 degrees down: ~0.7
    a = T.ground_scale(low, _project(low, np.zeros(3)))
    b = T.ground_scale(high, _project(high, np.zeros(3)))
    assert a is not None and b is not None
    assert a[1] == pytest.approx(math.cos(math.atan2(2.0, 90.0)), abs=0.01)
    assert b[1] == pytest.approx(math.cos(math.radians(45.0)), abs=0.02)
    # metres per pixel is depth / focal length
    assert b[0] == pytest.approx(math.hypot(40.0, 40.0) / 5000.0, rel=0.02)


@pytest.mark.parametrize("posture", ["stand", "run", "crouch"])
def test_the_stature_recovered_with_a_higher_camera_is_still_right(posture: str) -> None:
    """15 degrees down (a stand camera, twice the angle of the broadcast one); the vertical factor
    is 0.966 there. Above ~25 degrees the error of a running person passes 3 % (documented limit)."""
    el = math.radians(15.0)
    P = _look_at([0.0, -90.0 * math.cos(el), 90.0 * math.sin(el)], [0.0, 0.0, 0.0])[0]
    frames = _track(1.80, posture, SIDE_ON, P)
    result = T.stature_from_chain(dict.fromkeys(frames, P), frames, 7, NAMES)
    assert abs(result["height_m"] - 1.80) / 1.80 < 0.03, result


def test_frames_without_a_camera_and_wrong_boxes_do_not_count() -> None:
    frames = _track(1.80, "stand", SIDE_ON, CAMERA)
    cameras = dict.fromkeys(list(frames)[:4], CAMERA)  # four frames: fewer than MIN_CHAIN_FRAMES
    result = T.stature_from_chain(cameras, frames, 7, NAMES)
    assert result["height_m"] is None and result["frames"] == 4 and result["reason"]
    for inst_list in frames.values():  # keypoints that do not fit the box (a lost person)
        inst_list[0]["bbox_xyxy"] = [0.0, 0.0, 6.0, 30.0]
    result = T.stature_from_chain(dict.fromkeys(frames, CAMERA), frames, 7, NAMES)
    assert result["height_m"] is None and result["frames"] == 0


def test_the_spread_and_the_range_gate_are_reported() -> None:
    frames = _track(1.80, "run", ALL_HEADINGS, CAMERA)
    result = T.stature_from_chain(dict.fromkeys(frames, CAMERA), frames, 7, NAMES)
    assert result["iqr_m"] is not None and result["iqr_m"] > 0
    giant = _track(2.6, "stand", SIDE_ON, CAMERA)  # outside STATURE_RANGE_M: not reported
    result = T.stature_from_chain(dict.fromkeys(giant, CAMERA), giant, 7, NAMES)
    assert result["height_m"] is None and "outside" in result["reason"]
    assert result["iqr_m"] is not None


def test_a_model_without_the_chain_keypoints_reports_a_reason() -> None:
    result = T.stature_from_chain({0: CAMERA}, {0: []}, 7, ["nose"])
    assert result["height_m"] is None and "missing" in result["reason"]
