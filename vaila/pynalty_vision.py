"""
Project: vailá Multimodal Toolbox
Script: pynalty_vision.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 07 September 2026
Update Date: 07 September 2026
Version: 0.3.129

Description:
    Computer-vision helpers for the Pynalty penalty analysis: YOLO ball
    detection across the flight window, MediaPipe pose estimation inside a
    user-drawn bounding box, derived joint kinematics, and the overlay video /
    composite image writers.

    Both ultralytics and mediapipe are optional. When a backend is missing the
    corresponding function returns an empty result and prints a `>>` notice, so
    the rest of the Pynalty pipeline still completes.

License:
    This project is licensed under the terms of AGPLv3.0.
"""

from __future__ import annotations

import contextlib
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from .pynalty_analysis import BallPath, BallPathPoint, GoalGeometry, pixel_to_goal_plane
except ImportError:
    from pynalty_analysis import BallPath, BallPathPoint, GoalGeometry, pixel_to_goal_plane

SPORTS_BALL_CLASS = 32
PERSON_CLASS = 0

LANDMARK_NAMES = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)  # fmt: skip

L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

_POSE_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}
_POSE_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}


def _notice(msg: str) -> None:
    """Print a progress/warning line. The ``>>`` prefix survives absl logging."""
    print(f">> vaila/pynalty_vision: {msg}")


def _try_tqdm(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


# ==============================================================================
# Optional backends
# ==============================================================================


def models_dir() -> Path:
    d = Path(__file__).parent / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_yolo_weights(name: str = "yolo11n.pt") -> str | None:
    """Path to a YOLO detection checkpoint, or None when it cannot be found.

    Ultralytics downloads a missing checkpoint into the working directory on
    first use, so a bare name is returned as a last resort.
    """
    local = models_dir() / name
    if local.exists():
        return str(local)
    for alt in ("yolo11n.pt", "yolov8n.pt", "yolo26n.pt"):
        cand = models_dir() / alt
        if cand.exists():
            return str(cand)
    return name


def resolve_pose_model(complexity: int = 2) -> str | None:
    """Path to a MediaPipe Tasks pose model, downloading it if necessary."""
    name = _POSE_MODEL_NAMES.get(complexity, _POSE_MODEL_NAMES[2])
    path = models_dir() / name
    if path.exists():
        return str(path)
    url = _POSE_MODEL_URLS.get(complexity, _POSE_MODEL_URLS[2])
    try:
        _notice(f"downloading {name} ...")
        urllib.request.urlretrieve(url, str(path))
        _notice("download complete")
        return str(path)
    except Exception as exc:
        _notice(f"could not download {name}: {exc}")
        return None


def _load_yolo(weights: str | None = None):
    try:
        from ultralytics import YOLO
    except ImportError:
        _notice("ultralytics is not installed - automatic detection is unavailable")
        return None
    try:
        return YOLO(weights or resolve_yolo_weights())
    except Exception as exc:
        _notice(f"could not load YOLO weights: {exc}")
        return None


def _create_pose_landmarker(complexity: int = 2, num_poses: int = 1):
    try:
        import mediapipe as mp
    except ImportError:
        _notice("mediapipe is not installed - pose estimation is unavailable")
        return None, None
    model_path = resolve_pose_model(complexity)
    if not model_path:
        return None, None
    try:
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=0.1,
            min_pose_presence_confidence=0.1,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options), mp
    except Exception as exc:
        _notice(f"could not create PoseLandmarker: {exc}")
        return None, None


# ==============================================================================
# Ball detection
# ==============================================================================


def detect_ball_path(
    video_path: str,
    first_frame: int,
    last_frame: int,
    *,
    seed_px: tuple[float, float] | None = None,
    seed_frame: int | None = None,
    weights: str | None = None,
    conf: float = 0.10,
    max_jump_px: float = 220.0,
    imgsz: int = 1280,
) -> BallPath:
    """Track the ball through the flight window with YOLO, returning pixel points.

    Every candidate `sports ball` detection is collected first, then a single
    trajectory is walked out from a seed using constant-velocity prediction, so
    a stray detection on a distant ball or a white sock cannot capture the
    track. Frames with no acceptable candidate are simply left out; gap filling
    is the caller's job via ``pynalty_analysis.interpolate_ball_path``.
    """
    first_frame, last_frame = int(min(first_frame, last_frame)), int(max(first_frame, last_frame))
    model = _load_yolo(weights)
    if model is None:
        return BallPath()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _notice(f"could not open video: {video_path}")
        return BallPath()

    candidates: dict[int, list[tuple[float, float, float]]] = {}
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        frames = range(first_frame, last_frame + 1)
        for f in _try_tqdm(frames, desc="pynalty ball detect", unit="frame"):
            ok, frame = cap.read()
            if not ok:
                break
            try:
                res = model.predict(
                    frame,
                    classes=[SPORTS_BALL_CLASS],
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False,
                )
            except Exception as exc:
                _notice(f"detection failed on frame {f}: {exc}")
                break
            found: list[tuple[float, float, float]] = []
            for r in res:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue
                for b in boxes:
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                    found.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, float(b.conf[0])))
            if found:
                candidates[f] = found
    finally:
        cap.release()

    if not candidates:
        _notice("no ball detections in the flight window")
        return BallPath()

    picked = _walk_best_track(candidates, seed_px, seed_frame, max_jump_px)
    _notice(f"ball detected on {len(picked)} of {last_frame - first_frame + 1} frames")
    return BallPath(
        points=[
            BallPathPoint(frame=f, x_px=x, y_px=y, source="auto")
            for f, (x, y) in sorted(picked.items())
        ]
    )


def _walk_best_track(
    candidates: dict[int, list[tuple[float, float, float]]],
    seed_px: tuple[float, float] | None,
    seed_frame: int | None,
    max_jump_px: float,
) -> dict[int, tuple[float, float]]:
    """Pick one detection per frame, growing outward from the most reliable seed."""
    frames = sorted(candidates)

    if seed_frame is not None and seed_px is not None and seed_frame in candidates:
        start = seed_frame
        best = min(
            candidates[start],
            key=lambda c: math.hypot(c[0] - seed_px[0], c[1] - seed_px[1]),
        )
    else:
        start = max(frames, key=lambda f: max(c[2] for c in candidates[f]))
        best = max(candidates[start], key=lambda c: c[2])

    picked: dict[int, tuple[float, float]] = {start: (best[0], best[1])}

    def grow(order: list[int]) -> None:
        prev_pos: tuple[float, float] | None = picked[start]
        prev_frame = start
        velocity = (0.0, 0.0)
        for f in order:
            if f not in candidates or prev_pos is None:
                continue
            span = f - prev_frame
            pred = (prev_pos[0] + velocity[0] * span, prev_pos[1] + velocity[1] * span)
            cand = min(candidates[f], key=lambda c: math.hypot(c[0] - pred[0], c[1] - pred[1]))
            if math.hypot(cand[0] - pred[0], cand[1] - pred[1]) > max_jump_px * max(1, abs(span)):
                continue
            picked[f] = (cand[0], cand[1])
            if span != 0:
                velocity = ((cand[0] - prev_pos[0]) / span, (cand[1] - prev_pos[1]) / span)
            prev_pos, prev_frame = (cand[0], cand[1]), f

    grow([f for f in frames if f > start])
    grow(sorted((f for f in frames if f < start), reverse=True))
    return picked


# ==============================================================================
# Pose estimation inside a bounding box
# ==============================================================================


@dataclass
class PoseSequence:
    """Per-frame 33-landmark pixel coordinates for one tracked athlete."""

    label: str
    frames: list[int]
    landmarks: dict[int, np.ndarray]  # frame -> (33, 3) as x_px, y_px, visibility
    boxes: dict[int, tuple[int, int, int, int]]  # frame -> x1, y1, x2, y2

    def is_empty(self) -> bool:
        return not self.landmarks


def _expand_box(
    box: tuple[float, float, float, float],
    margin: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    return (
        int(max(0, min(x1, width - 1))),
        int(max(0, min(y1, height - 1))),
        int(max(1, min(x2, width))),
        int(max(1, min(y2, height))),
    )


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def pose_from_bbox(
    video_path: str,
    bbox: tuple[float, float, float, float],
    first_frame: int,
    last_frame: int,
    *,
    label: str = "athlete",
    complexity: int = 2,
    scale_factor: float = 4.0,
    safety_margin: float = 0.25,
    track_box: bool = True,
    yolo_weights: str | None = None,
) -> PoseSequence:
    """Run MediaPipe Pose on a cropped, upscaled bounding box across a frame range.

    The crop-upscale-detect-remap chain matches the one proven in
    ``markerless2d_mpyolo.py``: cropping to the athlete and upscaling before
    inference recovers landmarks that full-frame detection misses when the
    subject occupies a small part of a broadcast frame.

    When ``track_box`` is set and ultralytics is available, the box is
    re-centred each frame on the person detection that best overlaps the
    previous box, so a diving keeper does not walk out of a static crop.
    """
    first_frame, last_frame = int(min(first_frame, last_frame)), int(max(first_frame, last_frame))
    landmarker, mp = _create_pose_landmarker(complexity)
    if landmarker is None:
        return PoseSequence(label=label, frames=[], landmarks={}, boxes={})

    detector = _load_yolo(yolo_weights) if track_box else None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _notice(f"could not open video: {video_path}")
        return PoseSequence(label=label, frames=[], landmarks={}, boxes={})

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current = tuple(float(v) for v in bbox)

    landmarks: dict[int, np.ndarray] = {}
    boxes: dict[int, tuple[int, int, int, int]] = {}

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        for f in _try_tqdm(
            range(first_frame, last_frame + 1), desc=f"pynalty pose [{label}]", unit="frame"
        ):
            ok, frame = cap.read()
            if not ok:
                break

            if detector is not None:
                current = _retrack_box(detector, frame, current, width, height) or current

            x1, y1, x2, y2 = _expand_box(current, safety_margin, width, height)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            boxes[f] = (x1, y1, x2, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ch, cw = crop.shape[:2]
            new_w, new_h = int(cw * scale_factor), int(ch * scale_factor)
            scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB),
            )
            try:
                result = landmarker.detect(mp_image)
            except Exception as exc:
                _notice(f"pose detection failed on frame {f}: {exc}")
                continue
            if not getattr(result, "pose_landmarks", None):
                continue

            pts = np.full((len(LANDMARK_NAMES), 3), np.nan, dtype=float)
            for i, lm in enumerate(result.pose_landmarks[0]):
                if i >= pts.shape[0]:
                    break
                pts[i, 0] = (lm.x * new_w) / scale_factor + x1
                pts[i, 1] = (lm.y * new_h) / scale_factor + y1
                pts[i, 2] = float(getattr(lm, "visibility", np.nan))
            landmarks[f] = pts

            # Follow the athlete with the detected torso when YOLO is absent.
            if detector is None:
                current = _box_from_landmarks(pts, current)
    finally:
        cap.release()
        with contextlib.suppress(Exception):
            landmarker.close()

    _notice(f"pose [{label}]: landmarks on {len(landmarks)} frames")
    return PoseSequence(label=label, frames=sorted(landmarks), landmarks=landmarks, boxes=boxes)


def _retrack_box(detector, frame, prev_box, width, height):
    """Re-centre the crop on the person detection overlapping the previous box."""
    roi = _expand_box(prev_box, 0.6, width, height)
    x1, y1, x2, y2 = roi
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    try:
        res = detector.predict(sub, classes=[PERSON_CLASS], conf=0.25, verbose=False)
    except Exception:
        return None

    best, best_iou = None, 0.0
    shifted_prev = (prev_box[0], prev_box[1], prev_box[2], prev_box[3])
    for r in res:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for b in boxes:
            bx1, by1, bx2, by2 = (float(v) for v in b.xyxy[0].tolist())
            cand = (bx1 + x1, by1 + y1, bx2 + x1, by2 + y1)
            iou = _bbox_iou(cand, shifted_prev)
            if iou > best_iou:
                best, best_iou = cand, iou
    return best if best_iou > 0.05 else None


def _box_from_landmarks(pts: np.ndarray, fallback) -> tuple[float, float, float, float]:
    xy = pts[:, :2]
    valid = xy[~np.isnan(xy).any(axis=1)]
    if valid.shape[0] < 4:
        return fallback
    return (
        float(valid[:, 0].min()),
        float(valid[:, 1].min()),
        float(valid[:, 0].max()),
        float(valid[:, 1].max()),
    )


# ==============================================================================
# Derived kinematics
# ==============================================================================


def _angle_deg(a, b, c) -> float:
    """Interior angle at ``b`` formed by points ``a-b-c``, in degrees."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return float("nan")
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def kicker_kinematics(seq: PoseSequence, kick_frame: int, fps: float) -> dict:
    """Support-leg, trunk and foot-speed descriptors for the kicker at contact.

    The kicking leg is identified as the ankle with the higher pixel speed in
    the frames just before contact; the other leg is taken as the support leg.
    Angles are planar image-space angles, so they are affected by the camera
    viewpoint and should be compared within a session, not across venues.
    """
    out: dict = {}
    if seq.is_empty():
        return out

    frame = _nearest_frame(seq.frames, kick_frame)
    if frame is None:
        return out
    pts = seq.landmarks[frame]
    out["kicker_pose_frame"] = frame

    speeds = _ankle_speeds(seq, frame, fps, window=5)
    out["kicker_left_ankle_speed_px_s"] = speeds.get("left", float("nan"))
    out["kicker_right_ankle_speed_px_s"] = speeds.get("right", float("nan"))

    left_fast = speeds.get("left", 0.0) >= speeds.get("right", 0.0)
    kick_side = "left" if left_fast else "right"
    out["kicker_kicking_leg"] = kick_side
    out["kicker_foot_speed_px_s"] = speeds.get(kick_side, float("nan"))

    if kick_side == "left":
        s_hip, s_knee, s_ankle = R_HIP, R_KNEE, R_ANKLE
    else:
        s_hip, s_knee, s_ankle = L_HIP, L_KNEE, L_ANKLE

    out["kicker_support_knee_angle_deg"] = _angle_deg(
        pts[s_hip, :2], pts[s_knee, :2], pts[s_ankle, :2]
    )
    shoulder_mid = np.nanmean(pts[[L_SHOULDER, R_SHOULDER], :2], axis=0)
    hip_mid = np.nanmean(pts[[L_HIP, R_HIP], :2], axis=0)
    if not np.isnan(shoulder_mid).any() and not np.isnan(hip_mid).any():
        dx = shoulder_mid[0] - hip_mid[0]
        dy = hip_mid[1] - shoulder_mid[1]  # image Y grows downward
        out["kicker_trunk_lean_deg"] = float(math.degrees(math.atan2(dx, max(dy, 1e-6))))
    out["kicker_support_hip_angle_deg"] = _angle_deg(shoulder_mid, pts[s_hip, :2], pts[s_knee, :2])
    return out


def gk_kinematics(
    seq: PoseSequence,
    kick_frame: int,
    goal_frame: int,
    fps: float,
    calib_pixels=None,
    goal: GoalGeometry | None = None,
) -> dict:
    """Dive descriptors for the goalkeeper across the flight window.

    Hand spread is converted to metres with the goal-plane homography, which is
    exact only for points on the goal plane. The keeper is close to that plane
    during a save, so the value is a good approximation there and is labelled
    as such in the report.
    """
    out: dict = {}
    if seq.is_empty():
        return out

    kf = _nearest_frame(seq.frames, kick_frame)
    gf = _nearest_frame(seq.frames, goal_frame)
    if kf is None or gf is None:
        return out

    start, end = seq.landmarks[kf], seq.landmarks[gf]
    out["gk_pose_kick_frame"] = kf
    out["gk_pose_goal_frame"] = gf

    hip_start = np.nanmean(start[[L_HIP, R_HIP], :2], axis=0)
    hip_end = np.nanmean(end[[L_HIP, R_HIP], :2], axis=0)
    if not np.isnan(hip_start).any() and not np.isnan(hip_end).any():
        out["gk_hip_travel_px"] = float(np.linalg.norm(hip_end - hip_start))
        out["gk_dive_direction_px"] = float(hip_end[0] - hip_start[0])

    # Widest hand separation over the window: the effective span the keeper made.
    best_span, best_frame = float("nan"), None
    for f in seq.frames:
        if not (min(kf, gf) <= f <= max(kf, gf)):
            continue
        p = seq.landmarks[f]
        lw, rw = p[L_WRIST, :2], p[R_WRIST, :2]
        if np.isnan(lw).any() or np.isnan(rw).any():
            continue
        span = float(np.linalg.norm(rw - lw))
        if not np.isfinite(best_span) or span > best_span:
            best_span, best_frame = span, f
    out["gk_max_hand_span_px"] = best_span
    if best_frame is not None:
        out["gk_max_hand_span_frame"] = best_frame

    if calib_pixels is not None and best_frame is not None and np.isfinite(best_span):
        try:
            p = seq.landmarks[best_frame]
            real = pixel_to_goal_plane(
                calib_pixels, [p[L_WRIST, :2], p[R_WRIST, :2]], goal or GoalGeometry()
            )
            out["gk_max_hand_span_m"] = float(np.linalg.norm(real[1] - real[0]))
        except Exception:
            pass

    # First frame at which either wrist moves clearly away from its set position.
    move_frame = _first_movement_frame(seq, (L_WRIST, R_WRIST), kf, threshold_px=12.0)
    if move_frame is not None:
        out["gk_first_hand_move_frame"] = move_frame
        if fps > 0:
            out["gk_first_hand_move_rel_kick_s"] = (move_frame - kick_frame) / fps
    return out


def nearest_landmark_to_ball(
    seq: PoseSequence,
    frame: int,
    ball_px,
    *,
    prefer_hands: bool = True,
) -> dict:
    """Landmark of the athlete closest to the ball in image space.

    Prefers wrists/hands when ``prefer_hands`` is True and they are visible;
    otherwise returns the globally nearest landmark. Empty dict when pose is
    missing on that frame.
    """
    if seq.is_empty() or ball_px is None:
        return {}
    f = _nearest_frame(seq.frames, frame)
    if f is None or f not in seq.landmarks:
        return {}
    pts = seq.landmarks[f]
    ball = np.asarray(ball_px, dtype=float).reshape(2)

    hand_idx = (L_WRIST, R_WRIST, 17, 18, 19, 20, 21, 22)  # wrists + fingers
    candidates = hand_idx if prefer_hands else range(min(len(pts), len(LANDMARK_NAMES)))

    best_i, best_d = None, float("inf")
    for i in candidates:
        if i >= len(pts):
            continue
        xy = pts[i, :2]
        if np.isnan(xy).any():
            continue
        d = float(np.linalg.norm(xy - ball))
        if d < best_d:
            best_i, best_d = i, d

    if best_i is None and prefer_hands:
        return nearest_landmark_to_ball(seq, frame, ball_px, prefer_hands=False)
    if best_i is None:
        return {}

    name = LANDMARK_NAMES[best_i] if best_i < len(LANDMARK_NAMES) else str(best_i)
    return {
        "frame": f,
        "landmark_index": best_i,
        "landmark_name": name,
        "x_px": float(pts[best_i, 0]),
        "y_px": float(pts[best_i, 1]),
        "distance_px": best_d,
        "all_xy": pts[:, :2].copy(),
    }


def _nearest_frame(frames: list[int], target: int) -> int | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f - target))


def _ankle_speeds(seq: PoseSequence, frame: int, fps: float, window: int = 5) -> dict:
    """Mean pixel speed of each ankle over the frames leading up to ``frame``."""
    if fps <= 0:
        return {}
    usable = [f for f in seq.frames if frame - window <= f <= frame]
    if len(usable) < 2:
        return {}
    out: dict[str, float] = {}
    for side, idx in (("left", L_ANKLE), ("right", R_ANKLE)):
        total, dt = 0.0, 0.0
        for a, b in zip(usable, usable[1:], strict=False):
            pa, pb = seq.landmarks[a][idx, :2], seq.landmarks[b][idx, :2]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            total += float(np.linalg.norm(pb - pa))
            dt += (b - a) / fps
        out[side] = total / dt if dt > 0 else float("nan")
    return out


def _first_movement_frame(
    seq: PoseSequence,
    indices: tuple[int, ...],
    from_frame: int,
    threshold_px: float,
) -> int | None:
    frames = [f for f in seq.frames if f >= from_frame]
    if len(frames) < 2:
        return None
    base = seq.landmarks[frames[0]]
    for f in frames[1:]:
        pts = seq.landmarks[f]
        for idx in indices:
            a, b = base[idx, :2], pts[idx, :2]
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            if float(np.linalg.norm(b - a)) >= threshold_px:
                return f
    return None


# ==============================================================================
# Overlay rendering
# ==============================================================================

_TRAIL_COLOR = (0, 255, 255)
_BALL_COLOR = (0, 200, 255)


def draw_pose(image, pts: np.ndarray, color=(0, 255, 0), line_color=(255, 255, 255)) -> None:
    """Draw a 33-landmark skeleton in place using OpenCV primitives."""
    for i, j in POSE_CONNECTIONS:
        if i >= pts.shape[0] or j >= pts.shape[0]:
            continue
        a, b = pts[i, :2], pts[j, :2]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        cv2.line(image, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), line_color, 2)
    for k in range(pts.shape[0]):
        p = pts[k, :2]
        if np.isnan(p).any():
            continue
        cv2.circle(image, (int(p[0]), int(p[1])), 3, color, -1)


def _draw_trail(image, trail: list[tuple[float, float]], max_len: int = 40) -> None:
    """Draw a fading polyline through the recent ball positions."""
    pts = trail[-max_len:]
    n = len(pts)
    for i in range(1, n):
        weight = i / n
        color = tuple(int(c * (0.25 + 0.75 * weight)) for c in _TRAIL_COLOR)
        cv2.line(
            image,
            (int(pts[i - 1][0]), int(pts[i - 1][1])),
            (int(pts[i][0]), int(pts[i][1])),
            color,
            max(1, int(1 + 3 * weight)),
        )
    if pts:
        cv2.circle(image, (int(pts[-1][0]), int(pts[-1][1])), 7, _BALL_COLOR, 2)


def _draw_region(image, box, label, color) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(14, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def write_overlay_video(
    video_path: str,
    out_path: str,
    first_frame: int,
    last_frame: int,
    *,
    ball_rows: list[dict] | None = None,
    poses: list[PoseSequence] | None = None,
    regions: list[tuple[tuple[float, float, float, float], str, tuple[int, int, int]]]
    | None = None,
    fps: float | None = None,
) -> str | None:
    """Render an annotated clip of the flight window.

    Draws, when supplied: the fading ball trail with its instantaneous speed,
    one skeleton per pose sequence with its tracked box, and static labelled
    regions such as the goal mouth and the kick area.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _notice(f"could not open video: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    if not writer.isOpened():
        _notice(f"could not open the video writer for {out_path}")
        cap.release()
        return None

    ball_by_frame = {int(r["frame"]): r for r in (ball_rows or [])}
    pose_colors = [(0, 255, 0), (255, 128, 0), (255, 0, 255)]
    trail: list[tuple[float, float]] = []

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        for f in _try_tqdm(
            range(first_frame, last_frame + 1), desc="pynalty overlay", unit="frame"
        ):
            ok, frame = cap.read()
            if not ok:
                break

            for box, label, color in regions or []:
                _draw_region(frame, box, label, color)

            for i, seq in enumerate(poses or []):
                color = pose_colors[i % len(pose_colors)]
                if f in seq.boxes:
                    _draw_region(frame, seq.boxes[f], seq.label, color)
                if f in seq.landmarks:
                    draw_pose(frame, seq.landmarks[f], color=color)

            row = ball_by_frame.get(f)
            if row is not None:
                trail.append((row["x_px"], row["y_px"]))
            if trail:
                _draw_trail(frame, trail)
            if row is not None:
                speed = row.get("plane_speed_ms")
                if speed is not None and np.isfinite(speed):
                    cv2.putText(
                        frame,
                        f"{speed:.1f} m/s",
                        (int(row["x_px"]) + 12, int(row["y_px"]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        _BALL_COLOR,
                        2,
                    )

            cv2.putText(
                frame,
                f"frame {f}",
                (16, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(frame)
    finally:
        writer.release()
        cap.release()

    _notice(f"wrote {out_path}")
    return out_path


def write_ball_path_composite(
    video_path: str,
    out_path: str,
    base_frame: int,
    ball_rows: list[dict],
    *,
    regions=None,
) -> str | None:
    """Stamp every tracked ball position onto one frame, as a single still."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, base_frame))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        return None

    for box, label, color in regions or []:
        _draw_region(frame, box, label, color)

    pts = [(r["x_px"], r["y_px"]) for r in ball_rows]
    for i in range(1, len(pts)):
        cv2.line(
            frame,
            (int(pts[i - 1][0]), int(pts[i - 1][1])),
            (int(pts[i][0]), int(pts[i][1])),
            _TRAIL_COLOR,
            2,
        )
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 5, _BALL_COLOR, -1)
        if i % 5 == 0:
            cv2.putText(
                frame,
                str(int(ball_rows[i]["frame"])),
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, frame)
    _notice(f"wrote {out_path}")
    return out_path
