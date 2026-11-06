"""Regression: ``run_sam3d_from_sam`` with ``--save-mesh`` and no ``--cam-int-npy``.

A local ``import numpy as np`` inside the ``--cam-int-npy`` branch made ``np`` a local name
for the whole function, so the mesh export (and every later ``np`` use) raised
``UnboundLocalError: cannot access local variable 'np'`` whenever per-frame intrinsics were
not given (5 November 2026, batch ``processed_sam3dinov3_20261105_162103``).

No GPU, weights or video are needed: the capture and the SAM 3D Body estimator are stubbed,
and the run stops at the expected "no instances" error after the mesh faces are saved.
"""

from __future__ import annotations

import argparse
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import vaila.sam3dinov3 as s3d


class _EmptyCapture:
    """cv2.VideoCapture stand-in: opens, reports a 64x48 30 fps clip, yields no frames."""

    _props = {3: 64.0, 4: 48.0, 5: 30.0, 7: 0.0}

    def __init__(self, *_args, **_kwargs):
        pass

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV API
        return True

    def get(self, prop):
        return self._props.get(int(prop), 0.0)

    def read(self):
        return False, None

    def release(self):
        pass


def _args(**overrides) -> argparse.Namespace:
    base = {
        "weights_dir": None,
        "fov_estimator": "",
        "focal_px": None,
        "cam_int_npy": None,
        "no_overlay": True,
        "save_mesh": True,
        "stride": 1,
        "min_sam_score": 0.0,
        "min_sam_area": 0,
        "max_persons": 10,
        "bbox_padding": 0.0,
        "contour_margin": 0,
        "no_mask": False,
        "verbose_model": False,
        "inference_type": "full",
        "no_draw_id": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_save_mesh_without_cam_int_npy_writes_faces(tmp_path, monkeypatch):
    faces = np.arange(12, dtype=np.int64).reshape(4, 3)
    monkeypatch.setattr(s3d.cv2, "VideoCapture", _EmptyCapture)
    monkeypatch.setattr(s3d, "load_sam3d_estimator", lambda *a, **k: SimpleNamespace(faces=faces))
    out = tmp_path / "out"
    video = tmp_path / "clip.mp4"
    with pytest.raises(RuntimeError, match="produced no instances"):
        s3d.run_sam3d_from_sam(video, out, guidance=None, args=_args())
    saved = np.load(out / "mesh_faces.npy")
    np.testing.assert_array_equal(saved, faces)


def test_no_function_shadows_module_numpy():
    """A local ``np`` in any function of this module would hide the module import."""
    shadowing = [
        name
        for name, fn in inspect.getmembers(s3d, inspect.isfunction)
        if fn.__module__ == s3d.__name__ and "np" in fn.__code__.co_varnames
    ]
    assert shadowing == []
