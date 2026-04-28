"""Tests for :mod:`vaila.fifa_check_labels_dedupe`."""

from __future__ import annotations

from pathlib import Path

import pytest

from vaila import fifa_check_labels_dedupe as dedupe


def _write_rgb_jpg(path: Path, rgb: tuple[int, int, int] = (40, 80, 120)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("Pillow required") from exc
    Image.new("RGB", (160, 90), rgb).save(path, quality=95)


def test_pixel_similarity_identical() -> None:
    pytest.importorskip("cv2")
    import numpy as np

    a = np.zeros((20, 30, 3), dtype=np.uint8)
    b = a.copy()
    assert dedupe.pixel_similarity_bgr(a, b, channel_tol=0) == 1.0


def test_pixel_similarity_noise_within_tol() -> None:
    pytest.importorskip("cv2")
    import numpy as np

    a = np.zeros((10, 10, 3), dtype=np.uint8) + 100
    b = a.copy()
    b[0, 0, 0] += 10  # one channel fails if tol < 10
    assert dedupe.pixel_similarity_bgr(a, b, channel_tol=4) == 0.99


def test_find_duplicate_clusters_samefile_symlinks(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    bundle = tmp_path / "check_all_labels"
    img_dir = bundle / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    src = tmp_path / "master.jpg"
    _write_rgb_jpg(src)
    (img_dir / "train__a.jpg").symlink_to(src)
    (img_dir / "val__b.jpg").symlink_to(src)
    clusters = dedupe.find_duplicate_clusters(
        img_dir,
        similarity=0.99,
        channel_tol=4,
        max_ahash_hamming=0,
    )
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_dedupe_removes_extra_triplets(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    bundle = tmp_path / "check_all_labels"
    img_dir = bundle / "images"
    lbl_dir = bundle / "labels"
    ovl_dir = bundle / "images_with_labels"
    for d in (img_dir, lbl_dir, ovl_dir):
        d.mkdir(parents=True, exist_ok=True)

    src = tmp_path / "shared.jpg"
    _write_rgb_jpg(src)
    (img_dir / "train__dup_a.jpg").symlink_to(src)
    (img_dir / "train__dup_b.jpg").symlink_to(src)
    (lbl_dir / "train__dup_a.txt").write_text("0\n")
    (lbl_dir / "train__dup_b.txt").write_text("0\n")
    (ovl_dir / "train__dup_a.jpg").write_bytes(b"x")
    (ovl_dir / "train__dup_b.jpg").write_bytes(b"y")

    dedupe.dedupe_check_all_labels(bundle, dry_run=False)
    assert (img_dir / "train__dup_a.jpg").exists()
    assert not (img_dir / "train__dup_b.jpg").exists()
    assert not (lbl_dir / "train__dup_b.txt").exists()
    assert not (ovl_dir / "train__dup_b.jpg").exists()
