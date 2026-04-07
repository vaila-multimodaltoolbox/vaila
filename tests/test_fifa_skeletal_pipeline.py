"""Unit tests for FIFA skeletal pipeline (no GPU / no weights)."""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vaila.fifa_skeletal_pipeline import (
    OPENPOSE_TO_OURS,
    fifa_pack,
    load_sequences,
    write_sequences_from_videos,
)


def test_openpose_to_ours_length() -> None:
    assert len(OPENPOSE_TO_OURS) == 15


def test_load_sequences_strips_comments() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("# c\nSEQ_A\n\nSEQ_B\n")
        p = Path(f.name)
    try:
        assert load_sequences(p) == ["SEQ_A", "SEQ_B"]
    finally:
        p.unlink(missing_ok=True)


def test_fifa_pack_creates_zip() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dr = root / "data"
        dr.mkdir()
        (dr / "sequences_val.txt").write_text("S1\n", encoding="utf-8")
        full = root / "submission_full.npz"
        arr = np.zeros((2, 3, 15, 3), dtype=np.float32)
        np.savez_compressed(full, S1=arr)
        outd = root / "zips"
        zpath = fifa_pack(full, dr, outd, "val")
        assert zpath.is_file()
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            assert "submission.npz" in names
            inner = np.load(zf.open("submission.npz"))
            assert "S1" in inner.files


def test_write_sequences_from_videos_empty_dir(tmp_path: Path) -> None:
    vdir = tmp_path / "v"
    vdir.mkdir()
    out = tmp_path / "seq.txt"
    assert write_sequences_from_videos(vdir, out) == []
    assert out.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("shape25", "expected15"),
    [
        ((25, 2), 15),
        ((25, 3), 15),
    ],
)
def test_joint_subset_shape(shape25: tuple[int, int], expected15: int) -> None:
    x = np.arange(np.prod(shape25)).reshape(shape25).astype(np.float32)
    y = x[OPENPOSE_TO_OURS]
    assert y.shape == (expected15, shape25[1])
