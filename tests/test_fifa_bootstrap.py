"""Tests for vaila.fifa_bootstrap (FIFA Skeletal Tracking Light data layout)."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila.fifa_bootstrap import prepare_fifa_data_layout


def _touch_mp4(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake mp4 content")


def test_prepare_fifa_data_layout_creates_expected_files(tmp_path: Path) -> None:
    videos_src = tmp_path / "src_videos"
    _touch_mp4(videos_src / "ARG_CRO_000001.mp4")
    _touch_mp4(videos_src / "GER_FRA_000002.mp4")
    data_root = tmp_path / "data"

    with pytest.warns(UserWarning):
        res = prepare_fifa_data_layout(videos_dir=videos_src, data_root=data_root)

    # videos are symlinked (or copied as fallback) with the same stems
    stems = sorted(p.stem for p in (data_root / "videos").iterdir())
    assert stems == ["ARG_CRO_000001", "GER_FRA_000002"]
    assert len(res.videos_linked) == 2

    # sequences_full/val/test exist; val defaults to all (with a warning);
    # test is empty.
    full = res.sequences_full.read_text().splitlines()
    val = res.sequences_val.read_text().splitlines()
    test = res.sequences_test.read_text().splitlines() if res.sequences_test.read_bytes() else []
    assert sorted(full) == ["ARG_CRO_000001", "GER_FRA_000002"]
    assert sorted(val) == sorted(full)
    assert test == []

    # pitch_points.txt was copied from the vendored starter kit
    pp = res.pitch_points.read_text().splitlines()
    assert len(pp) > 100  # the vendored file has ~714 pitch points


def test_prepare_fifa_data_layout_with_explicit_splits(tmp_path: Path) -> None:
    videos_src = tmp_path / "src_videos"
    for name in ("A.mp4", "B.mp4", "C.mp4"):
        _touch_mp4(videos_src / name)
    data_root = tmp_path / "data"

    val_file = tmp_path / "val.txt"
    val_file.write_text("A\nB\n")
    test_file = tmp_path / "test.txt"
    test_file.write_text("C\n")

    res = prepare_fifa_data_layout(
        videos_dir=videos_src,
        data_root=data_root,
        val_sequences=val_file,
        test_sequences=test_file,
    )

    assert res.sequences_val.read_text().splitlines() == ["A", "B"]
    assert res.sequences_test.read_text().splitlines() == ["C"]
    assert res.sequences_full.read_text().splitlines() == ["A", "B", "C"]
