"""Tests for :mod:`vaila.fifa_manual_merge`."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from vaila import fifa_manual_merge as fmm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _one_valid_label_line() -> str:
    parts = ["0", "0.5", "0.5", "1.0", "1.0"]
    for _ in range(32):
        parts += ["0.0", "0.0", "0"]
    assert len(parts) == fmm.EXPECTED_LINE_FIELDS
    return " ".join(parts) + "\n"


def _write_min_unified(unified: Path) -> None:
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
    for split in fmm.SPLITS:
        (unified / "images" / split).mkdir(parents=True, exist_ok=True)
        (unified / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_manual_pair(
    src_root: Path,
    *,
    annotator: str,
    sequence: str,
    dataset: str,
    split: str,
    stem: str,
    label_text: str,
    image_ext: str = ".jpg",
) -> tuple[Path, Path]:
    base = src_root / annotator / sequence / dataset
    (base / "images" / split).mkdir(parents=True, exist_ok=True)
    (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    img = base / "images" / split / f"{stem}{image_ext}"
    lbl = base / "labels" / split / f"{stem}.txt"
    img.write_bytes(b"\xff\xd8\xff\xe0fakejpg")
    lbl.write_text(label_text, encoding="utf-8")
    return img, lbl


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validate_label_text_accepts_canonical_line() -> None:
    ok, reason = fmm.validate_label_text(_one_valid_label_line())
    assert ok, reason
    assert reason == ""


def test_validate_label_text_rejects_empty() -> None:
    ok, reason = fmm.validate_label_text("")
    assert not ok
    assert reason == "empty_label"


def test_validate_label_text_rejects_wrong_field_count() -> None:
    short_line = " ".join(["0"] * (fmm.EXPECTED_LINE_FIELDS - 1)) + "\n"
    ok, reason = fmm.validate_label_text(short_line)
    assert not ok
    assert "field_count" in reason


def test_validate_label_text_rejects_non_numeric() -> None:
    parts = ["0", "0.5", "0.5", "1.0", "1.0"]
    for _ in range(32):
        parts += ["NaN_token", "0.0", "0"]
    ok, reason = fmm.validate_label_text(" ".join(parts))
    assert not ok
    assert "non_numeric" in reason


# ---------------------------------------------------------------------------
# Discovery + name mangling
# ---------------------------------------------------------------------------


def test_discover_finds_pose_dataset_pairs(tmp_path: Path) -> None:
    src = tmp_path / "vaila_dataset"
    _write_manual_pair(
        src,
        annotator="sergio",
        sequence="CRO_MORp2",
        dataset="pose_dataset_20260430_110714",
        split="train",
        stem="CRO_MOR_190500_frame_001012",
        label_text=_one_valid_label_line(),
    )
    cands = fmm.discover_candidates(src)
    assert len(cands) == 1
    c = cands[0]
    assert c.annotator == "sergio"
    assert c.sequence == "CRO_MORp2"
    assert c.dataset == "pose_dataset_20260430_110714"
    assert c.split == "train"
    assert c.orig_stem == "CRO_MOR_190500_frame_001012"
    assert c.source_name == ("vaila_manual__sergio__CRO_MORp2__pose_dataset_20260430_110714")
    assert c.new_stem == (
        "vaila_manual__sergio__CRO_MORp2__pose_dataset_20260430_110714__CRO_MOR_190500_frame_001012"
    )


def test_discover_handles_nested_sequence(tmp_path: Path) -> None:
    """Layouts like lennin/BRA_KOR/BRA_KOR_p1/<dataset>/ are handled."""
    src = tmp_path / "vaila_dataset"
    _write_manual_pair(
        src,
        annotator="lennin",
        sequence="BRA_KOR/BRA_KOR_p1",
        dataset="fifa_dataset_template",
        split="val",
        stem="frame_000001",
        label_text=_one_valid_label_line(),
    )
    cands = fmm.discover_candidates(src)
    assert len(cands) == 1
    assert cands[0].sequence == "BRA_KOR__BRA_KOR_p1"


# ---------------------------------------------------------------------------
# End-to-end merge
# ---------------------------------------------------------------------------


def test_merge_happy_path(tmp_path: Path) -> None:
    src = tmp_path / "vaila_dataset"
    dst = tmp_path / "dataset_vaila_fifa"
    _write_min_unified(dst / "unified")

    _write_manual_pair(
        src,
        annotator="sergio",
        sequence="CRO_MORp2",
        dataset="pose_dataset_20260430_110714",
        split="train",
        stem="frame_a",
        label_text=_one_valid_label_line(),
    )
    _write_manual_pair(
        src,
        annotator="sergio",
        sequence="CRO_MORp2",
        dataset="pose_dataset_20260430_110714",
        split="val",
        stem="frame_b",
        label_text=_one_valid_label_line(),
        image_ext=".png",
    )
    _write_manual_pair(
        src,
        annotator="sergio",
        sequence="ARG_CROp1",
        dataset="pose_dataset_20260430_100433",
        split="test",
        stem="frame_empty",
        label_text="",
    )

    result = fmm.merge_manual_dataset(src, dst, verify_after=False)

    assert result.counts["candidates_total"] == 3
    assert result.counts.get("added", 0) == 2
    assert result.counts.get("skipped_empty_label", 0) == 1
    assert result.counts["manifest_rows_written"] == 2

    unified = dst / "unified"
    img_train = (
        unified
        / "images"
        / "train"
        / "vaila_manual__sergio__CRO_MORp2__pose_dataset_20260430_110714__frame_a.jpg"
    )
    lbl_train = (
        unified
        / "labels"
        / "train"
        / "vaila_manual__sergio__CRO_MORp2__pose_dataset_20260430_110714__frame_a.txt"
    )
    img_val = (
        unified
        / "images"
        / "val"
        / "vaila_manual__sergio__CRO_MORp2__pose_dataset_20260430_110714__frame_b.png"
    )
    assert img_train.is_symlink()
    assert lbl_train.is_symlink()
    assert img_val.is_symlink()
    assert img_train.resolve().is_file()
    assert lbl_train.resolve().is_file()
    assert img_val.resolve().is_file()

    manifest = unified / "manifest.csv"
    assert manifest.is_file()
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    splits = {r["split"] for r in rows}
    assert splits == {"train", "val"}
    sources = {r["source"] for r in rows}
    assert sources == {"vaila_manual__sergio__CRO_MORp2__pose_dataset_20260430_110714"}

    assert result.report_csv is not None and result.report_csv.is_file()
    assert result.report_log is not None and result.report_log.is_file()


def test_merge_idempotent(tmp_path: Path) -> None:
    src = tmp_path / "vaila_dataset"
    dst = tmp_path / "dataset_vaila_fifa"
    _write_min_unified(dst / "unified")
    _write_manual_pair(
        src,
        annotator="sergio",
        sequence="CRO_MORp2",
        dataset="pose_dataset_20260430_110714",
        split="train",
        stem="frame_a",
        label_text=_one_valid_label_line(),
    )

    first = fmm.merge_manual_dataset(src, dst, verify_after=False)
    second = fmm.merge_manual_dataset(src, dst, verify_after=False)

    assert first.counts.get("added", 0) == 1
    assert first.counts["manifest_rows_written"] == 1
    assert second.counts.get("added", 0) == 0
    assert second.counts.get("already_present", 0) == 1
    assert second.counts["manifest_rows_written"] == 0

    manifest = dst / "unified" / "manifest.csv"
    with manifest.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1


def test_merge_dry_run_writes_nothing(tmp_path: Path) -> None:
    src = tmp_path / "vaila_dataset"
    dst = tmp_path / "dataset_vaila_fifa"
    _write_min_unified(dst / "unified")
    _write_manual_pair(
        src,
        annotator="sergio",
        sequence="CRO_MORp2",
        dataset="pose_dataset_20260430_110714",
        split="train",
        stem="frame_a",
        label_text=_one_valid_label_line(),
    )

    result = fmm.merge_manual_dataset(src, dst, dry_run=True, verify_after=False)

    assert result.counts.get("dry_run", 0) == 1
    assert result.counts["manifest_rows_written"] == 0
    assert not (dst / "unified" / "manifest.csv").exists()
    assert not (dst / "staging").exists()


def test_merge_skips_invalid_field_count(tmp_path: Path) -> None:
    src = tmp_path / "vaila_dataset"
    dst = tmp_path / "dataset_vaila_fifa"
    _write_min_unified(dst / "unified")
    short_line = " ".join(["0.0"] * (fmm.EXPECTED_LINE_FIELDS - 1)) + "\n"
    _write_manual_pair(
        src,
        annotator="sergio",
        sequence="CRO_MORp2",
        dataset="pose_dataset_20260430_110714",
        split="train",
        stem="frame_short",
        label_text=short_line,
    )

    result = fmm.merge_manual_dataset(src, dst, verify_after=False)

    assert result.counts.get("added", 0) == 0
    skipped = sum(v for k, v in result.counts.items() if k.startswith("skipped_invalid_label"))
    assert skipped == 1


def test_merge_requires_existing_unified(tmp_path: Path) -> None:
    src = tmp_path / "vaila_dataset"
    dst = tmp_path / "dataset_vaila_fifa"
    src.mkdir()
    dst.mkdir()
    with pytest.raises(FileNotFoundError):
        fmm.merge_manual_dataset(src, dst, verify_after=False)
