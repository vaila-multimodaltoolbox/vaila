"""Light tests for vaila_sam helpers (no GPU / no HF download)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

sam3_spec = importlib.util.find_spec("sam3")


@pytest.mark.skipif(sam3_spec is None, reason="sam3 optional extra not installed")
def test_resolve_bpe_path_exists() -> None:
    from vaila.vaila_sam import _resolve_bpe_path

    p = _resolve_bpe_path()
    assert p.is_file()


def test_resolve_checkpoint_rejects_sam3d_body_path(tmp_path: Path) -> None:
    from vaila.vaila_sam import _resolve_sam3_checkpoint_file

    p = tmp_path / "vaila" / "models" / "sam-3d-dinov3" / "assets" / "mhr_model.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="SAM 3D Body"):
        _resolve_sam3_checkpoint_file(p)


def test_composite_masks_bgr_empty() -> None:
    from vaila.vaila_sam import _composite_masks_bgr

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    out = _composite_masks_bgr(
        frame, np.zeros((0, 64, 64), dtype=bool), np.array([], dtype=np.int64)
    )
    assert out.shape == frame.shape


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


SAM_SAMPLE_VIDEO = Path(__file__).resolve().parent / "SAM" / "test1000.mp4"


@pytest.mark.skipif(sam3_spec is None, reason="sam3 optional extra not installed")
@pytest.mark.skipif(not _cuda_available(), reason="SAM 3 video path requires CUDA")
@pytest.mark.skipif(
    os.environ.get("VAILA_TEST_SAM_GPU") != "1", reason="set VAILA_TEST_SAM_GPU=1 to run"
)
def test_sam3_smoke_sample_video(tmp_path: Path) -> None:
    """End-to-end SAM3 on tests/SAM/test1000.mp4 (slow; needs HF access or local sam3.pt)."""
    if not SAM_SAMPLE_VIDEO.is_file():
        pytest.skip(f"Add sample video at {SAM_SAMPLE_VIDEO} (see tests/SAM/README.md)")

    from vaila.vaila_sam import run_sam3_on_video

    out_dir = tmp_path / "sam_smoke"
    run_sam3_on_video(
        SAM_SAMPLE_VIDEO,
        out_dir,
        "person",
        frame_index=0,
        save_overlay_mp4=False,
        save_mask_png=False,
    )
    assert (out_dir / "README_sam.txt").is_file()
    assert (out_dir / "sam_frames_meta.csv").is_file()
