"""
================================================================================
MediaPipe pose, Tasks API helper - mp_pose_tasks.py
================================================================================
*vailá* – Multimodal Toolbox
Authors: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
https://github.com/vaila-multimodaltoolbox/vaila
Date: 16 November 2026
Update: 16 November 2026
Version: 0.4.19
Python Version: 3.12.16

Description:
------------
mediapipe 1.x removed the legacy ``mp.solutions`` API (``mp.solutions.pose.Pose``):
the pose model is now the Tasks ``PoseLandmarker``. This module gives the modules that
still ran the legacy API what they need, with the same defaults:

* ``get_pose_model_path(complexity)`` - the ``.task`` model file in ``vaila/models``
  (0 lite, 1 full, 2 heavy; downloaded once, the same files the other markerless
  modules use);
* ``PoseVideoLandmarker`` - a video-mode landmarker (it tracks and smooths between
  frames, like ``static_image_mode=False``): ``detect(bgr_frame)`` returns the 33
  landmarks of the first person, or ``None``;
* ``landmark_visibility`` - the visibility of a landmark as a float.

Landmark order and names are the BlazePose ones (``mp.tasks.vision.PoseLandmark``,
an ``IntEnum``, so it indexes the list directly).

Usage:
------
    from vaila.mp_pose_tasks import PoseVideoLandmarker

    with PoseVideoLandmarker(complexity=2) as landmarker:
        landmarks = landmarker.detect(frame_bgr, timestamp_ms=1000.0 * i / fps)
        if landmarks is not None:
            x, y = landmarks[0].x * width, landmarks[0].y * height  # the nose

Requirements:
-------------
- Python 3.12, mediapipe >= 1.0 (Tasks API), OpenCV, numpy

License:
--------
AGPL-3.0, see LICENSE.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

POSE_MODEL_NAMES = {
    0: "pose_landmarker_lite.task",
    1: "pose_landmarker_full.task",
    2: "pose_landmarker_heavy.task",
}
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/{stem}/float16/1/{name}"
)


def get_pose_model_path(
    complexity: int = 1, *, models_dir: str | Path | None = None, download: bool = True
) -> str:
    """Path of the pose ``.task`` model (0 lite, 1 full, 2 heavy), downloaded if missing.

    Raises ``RuntimeError`` when the file is missing and ``download`` is False or fails.
    """
    folder = (
        Path(models_dir) if models_dir is not None else Path(__file__).resolve().parent / "models"
    )
    name = POSE_MODEL_NAMES.get(int(complexity), POSE_MODEL_NAMES[1])
    path = folder / name
    if path.is_file():
        return str(path)
    if not download:
        raise RuntimeError(f"MediaPipe pose model not found: {path}")
    folder.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe pose model ({name})... please wait.")
    try:
        urllib.request.urlretrieve(
            _POSE_MODEL_URL.format(stem=name.removesuffix(".task"), name=name), str(path)
        )
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download the MediaPipe pose model {name}: {exc}") from exc
    print("Download completed!")
    return str(path)


def landmark_visibility(landmark: Any) -> float:
    """Visibility of a pose landmark (0.0 when the model gives none)."""
    value = getattr(landmark, "visibility", None)
    return 0.0 if value is None else float(value)


class PoseVideoLandmarker:
    """Tasks ``PoseLandmarker`` in video mode: one person, tracked between frames."""

    def __init__(
        self,
        complexity: int = 1,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:  # pragma: no cover - mediapipe is a core dependency
            raise ImportError("mediapipe is required for pose estimation") from exc
        self._mp = mp
        path = str(model_path) if model_path is not None else get_pose_model_path(complexity)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=float(min_detection_confidence),
            min_pose_presence_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: float | None = None) -> list | None:
        """The 33 landmarks (``.x .y .z .visibility``, x/y in 0-1) of the first person, or None.

        ``timestamp_ms`` is the time of the frame (video: ``1000 * i / fps``); left out, the
        clock is used (webcam). Timestamps must grow, so a repeated one is moved up 1 ms.
        """
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if timestamp_ms is None:
            timestamp_ms = time.monotonic() * 1000.0
        self._last_ms = max(int(timestamp_ms), self._last_ms + 1)
        result = self._landmarker.detect_for_video(image, self._last_ms)
        return list(result.pose_landmarks[0]) if result.pose_landmarks else None

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> PoseVideoLandmarker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
