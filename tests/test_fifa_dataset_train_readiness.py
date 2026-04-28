"""Tests for :mod:`vaila.fifa_dataset_train_readiness`."""

from __future__ import annotations

from pathlib import Path

from vaila import fifa_dataset_train_readiness as fr


def _write_min_data_yaml(unified: Path) -> None:
    unified.mkdir(parents=True, exist_ok=True)
    (unified / "data.yaml").write_text(
        f"path: {unified}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "kpt_shape: [32, 3]\n"
        "flip_idx: []\n"
        "names:\n  0: football_pitch\n",
        encoding="utf-8",
    )


def _one_valid_label_line() -> str:
    parts = ["0", "0.5", "0.5", "1.0", "1.0"]
    for _ in range(32):
        parts += ["0.0", "0.0", "0"]
    assert len(parts) == 101
    return " ".join(parts) + "\n"


def test_verify_unified_ok(tmp_path: Path) -> None:
    unified = tmp_path / "unified"
    _write_min_data_yaml(unified)
    for split in ("train", "val", "test"):
        (unified / "images" / split).mkdir(parents=True, exist_ok=True)
        (unified / "labels" / split).mkdir(parents=True, exist_ok=True)
    split = "train"
    stem = "src__frame0"
    (unified / "images" / split / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpg")
    (unified / "labels" / split / f"{stem}.txt").write_text(
        _one_valid_label_line(), encoding="utf-8"
    )

    issues, counts = fr.verify_unified_dataset(unified)
    assert not issues
    assert counts["labels"] == 1
    assert counts["images_missing"] == 0


def test_prune_removes_unified_row_not_in_flat(tmp_path: Path) -> None:
    unified = tmp_path / "unified"
    flat = tmp_path / "check_all_labels"
    _write_min_data_yaml(unified)
    for split, stem in [("train", "keep__a"), ("train", "drop__b")]:
        (unified / "images" / split).mkdir(parents=True, exist_ok=True)
        (unified / "labels" / split).mkdir(parents=True, exist_ok=True)
        (unified / "images" / split / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpg")
        (unified / "labels" / split / f"{stem}.txt").write_text(
            _one_valid_label_line(), encoding="utf-8"
        )

    (flat / "images").mkdir(parents=True)
    (flat / "images" / "train__keep__a.jpg").write_bytes(b"x")

    fr.prune_unified_to_match_flat(unified, flat, dry_run=False)
    assert (unified / "labels" / "train" / "keep__a.txt").exists()
    assert not (unified / "labels" / "train" / "drop__b.txt").exists()


def test_collect_flat_image_stems(tmp_path: Path) -> None:
    d = tmp_path / "images"
    d.mkdir()
    (d / "train__x.jpg").write_text("a")
    (d / "val__y.PNG").write_text("b")
    assert fr.collect_flat_image_stems(d) == {"train__x", "val__y"}
