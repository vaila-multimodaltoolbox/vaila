"""
================================================================================
Script: freekiki_posegate.py - FreeKiki close-up gate (MediaPipe pose size)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.8
Created: 08 October 2026
Update Date: 08 October 2026

Description:
    The FreeKiki network is trained on wide broadcast cameras. Cameras behind a
    goal, on a celebrating player or zoomed on a face make it put keypoints in
    wrong places. A wide shot shows small players; a close-up shows a large one.
    This gate runs MediaPipe Pose (the PoseLandmarker of
    ``markerless_2d_analysis.py``) on frames of a candidate clip and calls a
    frame a close-up when a pose is taller than ``max_pose`` of the image
    height (default 1/10). No pose found means the frame is apparently fine.

    * :func:`pose_fraction` - height of the largest pose, from a MediaPipe result;
    * :class:`PoseGate` - the lazy MediaPipe landmarker (CPU, image mode);
    * :func:`check_window` - sample the frames of a video window;
    * :func:`split_good` - keep the stretches of a window without close-ups.

    MediaPipe is imported only when a gate is created, so the pure functions
    run (and are tested) without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

MAX_POSE = 0.10  # pose height / image height above which a frame is a close-up
MAX_SIDE = 960  # frames are shrunk to this width before MediaPipe (speed)


def pose_fraction(result, *, min_visibility: float = 0.3) -> float:
    """Height of the largest pose of a PoseLandmarker result, as a fraction of the image.

    ``result.pose_landmarks`` is one list of normalised landmarks per person;
    only landmarks with ``visibility >= min_visibility`` count (a hidden
    landmark is extrapolated by MediaPipe and would inflate the box).
    Returns 0.0 when there is no pose.
    """
    best = 0.0
    for pose in getattr(result, "pose_landmarks", None) or []:
        ys = [
            float(lm.y)
            for lm in pose
            if (getattr(lm, "visibility", None) is None or lm.visibility >= min_visibility)
        ]
        if len(ys) >= 2:
            best = max(best, max(ys) - min(ys))
    return best


class PoseGate:
    """MediaPipe pose size of a frame (CPU, image mode, up to ``num_poses`` people)."""

    def __init__(self, complexity: int = 0, max_pose: float = MAX_POSE, num_poses: int = 5) -> None:
        import mediapipe as mp

        tasks: Any = mp.tasks  # the stubs lack the lazily loaded submodules
        vision = tasks.vision
        try:
            from .markerless_2d_analysis import get_model_path
        except ImportError:
            from markerless_2d_analysis import get_model_path  # ty: ignore[unresolved-import]

        self.max_pose = float(max_pose)
        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=tasks.BaseOptions(model_asset_path=get_model_path(complexity)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def result(self, frame_bgr):
        """The raw PoseLandmarker result of a BGR frame."""
        import cv2

        h, w = frame_bgr.shape[:2]
        if w > MAX_SIDE:
            frame_bgr = cv2.resize(frame_bgr, (MAX_SIDE, round(h * MAX_SIDE / w)))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(image)

    def fraction(self, frame_bgr) -> float:
        """Height of the largest pose, as a fraction of the image height."""
        return pose_fraction(self.result(frame_bgr))

    def big(self, frame_bgr) -> bool:
        return self.fraction(frame_bgr) > self.max_pose

    def close(self) -> None:
        self._landmarker.close()


def check_window(
    video, start: int, end: int, gate, *, every: int = 10, on_big=None
) -> list[tuple[int, float]]:
    """``(frame, largest pose fraction)`` of every ``every``-th frame of a window.

    ``start`` / ``end`` are 0-based inclusive frame indices. The window is read
    sequentially (frame-accurate, unlike seeking inside a stream). ``on_big``
    (optional) is called as ``on_big(frame, image, fraction)`` for every
    sampled frame above ``gate.max_pose``.
    """
    import cv2

    cap = cv2.VideoCapture(str(Path(video)))
    out: list[tuple[int, float]] = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for frame in range(start, end + 1):
            if (frame - start) % every:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            frac = float(gate.fraction(img))
            out.append((frame, frac))
            if on_big is not None and frac > gate.max_pose:
                on_big(frame, img, frac)
    finally:
        cap.release()
    return out


def split_good(
    start: int,
    end: int,
    samples: list[tuple[int, float]],
    *,
    max_pose: float = MAX_POSE,
    every: int = 10,
    min_len: int = 50,
) -> list[tuple[int, int]]:
    """Stretches of ``[start, end]`` (0-based, inclusive) without close-ups.

    A sampled frame whose pose fraction is above ``max_pose`` removes the
    stretch of ``every`` frames centred on it. The window is split there and
    the pieces of at least ``min_len`` frames are returned in time order.
    """
    half = max(1, every // 2)
    bad = sorted(f for f, frac in samples if frac > max_pose)
    pieces, cur = [], start
    for f in bad:
        if f - half - 1 >= cur:
            pieces.append((cur, f - half - 1))
        cur = max(cur, f + half)
    if cur <= end:
        pieces.append((cur, end))
    return [(a, b) for a, b in pieces if b - a + 1 >= min_len]
