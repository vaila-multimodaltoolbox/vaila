from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vaila import remove_frames2sync


def _write_png(path: Path, value: int) -> None:
    arr = np.full((24, 32, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _write_sequence(folder: Path, values: list[int]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for i, v in enumerate(values, start=1):
        _write_png(folder / f"{i:09d}.png", v)


def test_exact_dedupe_non_destructive_writes_outputs(tmp_path: Path) -> None:
    seq = tmp_path / "seq_cam"
    _write_sequence(seq, [0, 0, 10, 20, 20])

    out_dir = tmp_path / "processed"
    result = remove_frames2sync.process_folder(
        seq,
        dry_run=False,
        in_place=False,
        out_dir=out_dir,
        mode="exact",
        phash_threshold=2,
        verbose=False,
    )
    assert result is not None
    assert result["total"] == 5
    assert result["removed"] == 2
    assert result["remaining"] == 3

    # Original sequence untouched
    assert len(list(seq.glob("*.png"))) == 5

    kept = out_dir / "kept_frames" / "seq_cam"
    removed = out_dir / "removed_frames" / "seq_cam"
    assert kept.exists()
    assert removed.exists()
    assert len(list(kept.glob("*.png"))) == 3
    assert (kept / "000000001.png").exists()
    assert (kept / "000000003.png").exists()
    assert len(list(removed.glob("*.png"))) == 2


def test_dry_run_does_not_modify_or_write_outputs(tmp_path: Path) -> None:
    seq = tmp_path / "seq_cam"
    _write_sequence(seq, [1, 1, 2])

    before = sorted(p.name for p in seq.glob("*.png"))
    result = remove_frames2sync.process_folder(
        seq,
        dry_run=True,
        in_place=False,
        out_dir=None,
        mode="exact",
        verbose=False,
    )
    after = sorted(p.name for p in seq.glob("*.png"))

    assert before == after
    assert result is not None
    assert result["removed"] == 1
    assert result["remaining"] == 2

    assert not (seq / "removed_frames").exists()


def test_anchor_fps_estimation_math() -> None:
    results = [
        {
            "folder": "anchor",
            "remaining": 100,
            "total": 100,
            "removed": 0,
            "estimated_fps": 60.0,
            "ratio": 1.0,
        },
        {
            "folder": "camB",
            "remaining": 50,
            "total": 60,
            "removed": 10,
            "estimated_fps": 50.0,
            "ratio": 0.8333,
        },
    ]
    remove_frames2sync._compute_anchor_fps(results, anchor_folder="anchor", anchor_fps=50.0)
    assert results[0]["estimated_fps_from_anchor"] == pytest.approx(50.0)
    assert results[1]["estimated_fps_from_anchor"] == pytest.approx(25.0)


def test_non_numeric_filenames_raise(tmp_path: Path) -> None:
    seq = tmp_path / "seq_bad"
    seq.mkdir()
    _write_png(seq / "frame_a.png", 0)
    _write_png(seq / "000000002.png", 0)

    with pytest.raises(ValueError, match="numeric"):
        remove_frames2sync.process_folder(seq, dry_run=True, verbose=False)
