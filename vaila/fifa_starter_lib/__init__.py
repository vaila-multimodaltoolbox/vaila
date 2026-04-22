"""FIFA Skeletal Tracking Starter Kit 2026 — MIT-vendored utilities.

This package is a light-weight vendoring of the ``lib/`` directory of the
official FIFA Skeletal Tracking Light 2026 starter kit (MIT-licensed, see
``LICENSE_MIT``).  It exposes the public entry points consumed by
``vaila.fifa_skeletal_pipeline``:

- ``CameraTracker``, ``CameraTrackerOptions``, ``CameraState`` — broadcast
  camera extrinsic tracking from optical flow + field-line refinement.
- ``smoothen``, ``smoothen_traj`` — Gaussian smoothing of 3D skeleton
  trajectories.

Upstream: https://github.com/FIFA-Skeletal-Light-Tracking-Challenge/FIFA-Skeletal-Tracking-Starter-Kit-2026
"""

from .camera_tracker import (
    CameraState,
    CameraTracker,
    CameraTrackerOptions,
    Debugger,
    extract_lane_lines_mask,
    optical_flow_pyrlk,
)
from .postprocess import interpolate_with_gap, smoothen, smoothen_traj

__all__ = [
    "CameraState",
    "CameraTracker",
    "CameraTrackerOptions",
    "Debugger",
    "extract_lane_lines_mask",
    "interpolate_with_gap",
    "optical_flow_pyrlk",
    "smoothen",
    "smoothen_traj",
]
