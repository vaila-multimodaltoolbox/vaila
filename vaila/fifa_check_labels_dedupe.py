"""Deduplicate ``check_all_labels`` bundles by near-identical image content.

Pipeline:

1. **Same file on disk** — resolved paths equal (e.g. multiple symlinks to one
   JPEG) → one cluster without decoding.
2. **Fingerprint** — Pillow reads each file only far enough to build an 8×8
   average-hash plus true ``(width, height)`` (fast on large files).
3. **Exact buckets** — images sharing identical ``(h, w, aHash)`` are compared
   with OpenCV on **full resolution**: at least ``--similarity`` of pixels must
   match in **all** BGR channels within ``--channel-tol`` (JPEG noise).  Within
   a bucket, each image is compared only to the first element (linear cost).

Optional ``--max-ahash-hamming N`` with ``N > 0`` runs an extra slow pass for
near-collisions on aHash (avoid on very large trees unless needed).

For each duplicate cluster, keeps the lexicographically smallest basename and
removes the other **triplets** (``images/``, ``labels/``, ``images_with_labels/``).

CLI::

    uv run python -m vaila.fifa_check_labels_dedupe \\
        --bundle /path/to/check_all_labels \\
        --similarity 0.99 --channel-tol 4

Use ``--dry-run`` to preview removals without deleting.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .fifa_dataset_builder import _IMAGE_EXTS_ORDERED
except ImportError:
    _IMAGE_EXTS_ORDERED = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _iter_bundle_images(images_dir: Path) -> Iterator[Path]:
    for ext in _IMAGE_EXTS_ORDERED:
        yield from sorted(images_dir.glob(f"*{ext}"))


def _real_key(path: Path) -> str:
    """Stable key for exact same file on disk (symlink → same target)."""
    try:
        return str(path.resolve(strict=True))
    except OSError:
        return str(path.resolve())


def _ahash_u64_gray_u8(gray8: Any) -> int:
    """8×8 average-hash from a uint8 grayscale 8×8 array."""
    m = float(gray8.mean())
    flat = (gray8 >= m).reshape(64).astype(np.uint8)
    packed = np.packbits(flat, bitorder="big")
    return int.from_bytes(packed.tobytes(), "big")


def _read_size_and_ahash_pil(path: Path) -> tuple[int, int, int]:
    """Image (h, w) + aHash without full-resolution decode (fast for large JPEGs)."""
    from PIL import Image

    try:
        with Image.open(path) as im:
            w, h = im.size
            gray = im.convert("L").resize((8, 8), Image.Resampling.BOX)
            arr = np.asarray(gray, dtype=np.uint8)
    except OSError:
        return (0, 0, 0)
    return (h, w, _ahash_u64_gray_u8(arr))


def _hamming_u64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def pixel_similarity_bgr(
    a: Any,
    b: Any,
    *,
    channel_tol: int,
) -> float:
    """Fraction of pixels where all BGR channels differ by at most ``channel_tol``."""
    if a.shape != b.shape:
        return 0.0
    da = a.astype(np.int16)
    db = b.astype(np.int16)
    diff = np.abs(da - db)
    good = np.all(diff <= channel_tol, axis=-1)
    return float(good.mean())


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[rj] = ri


def find_duplicate_clusters(
    images_dir: Path,
    *,
    similarity: float,
    channel_tol: int,
    max_ahash_hamming: int,
) -> list[list[Path]]:
    """Return lists of paths; each list is one cluster (length ≥ 2 = duplicates)."""
    import cv2  # type: ignore[import-not-found]

    paths = list(_iter_bundle_images(images_dir))
    if len(paths) < 2:
        return []

    # Pass 1: exact same inode / resolved path → union without decoding.
    by_real: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(paths):
        by_real[_real_key(p)].append(i)
    uf = _UnionFind(len(paths))
    for group in by_real.values():
        head = group[0]
        for j in group[1:]:
            uf.union(head, j)

    # Lightweight metadata (no full-image cache — avoids RAM blow-up on ~20k HD frames).
    meta: list[tuple[int, int, int]] = []  # h, w, ahash_u64
    for i, p in enumerate(paths):
        if (i + 1) % 5000 == 0:
            print(f"[fifa_check_labels_dedupe] fingerprint {i + 1}/{len(paths)} images…")
        h, w, ah = _read_size_and_ahash_pil(p)
        meta.append((h, w, ah))

    # Pass 2a: exact (h, w, aHash) buckets — O(n) buckets, tiny pairwise work per bucket.
    by_exact: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for k, (h, w, ah) in enumerate(meta):
        if h == 0:
            continue
        by_exact[(h, w, ah)].append(k)

    for idxs in by_exact.values():
        if len(idxs) < 2:
            continue
        # Same (h, w, aHash) almost always means identical or duplicate frames.
        # For large buckets (rare aHash collisions), compare each image only to
        # the first entry — O(n) loads instead of O(n²).
        root_i = idxs[0]
        arr_root = cv2.imread(str(paths[root_i]), cv2.IMREAD_COLOR)
        if arr_root is None:
            continue
        try:
            for ib in idxs[1:]:
                if uf.find(ib) == uf.find(root_i):
                    continue
                arr_b = cv2.imread(str(paths[ib]), cv2.IMREAD_COLOR)
                if arr_b is None:
                    continue
                try:
                    if pixel_similarity_bgr(arr_root, arr_b, channel_tol=channel_tol) >= similarity:
                        uf.union(root_i, ib)
                finally:
                    del arr_b
        finally:
            del arr_root

    # Pass 2b (optional): loose aHash within same (h, w) — can be slow on huge buckets.
    if max_ahash_hamming > 0:
        by_hw: dict[tuple[int, int], list[int]] = defaultdict(list)
        for k, (h, w, _ah) in enumerate(meta):
            if h == 0:
                continue
            by_hw[(h, w)].append(k)
        for (hw_h, hw_w), idxs in by_hw.items():
            if len(idxs) < 2:
                continue
            if len(idxs) > 2500:
                print(
                    f"[fifa_check_labels_dedupe] warning: loose aHash on large bucket "
                    f"{hw_w}x{hw_h} (n={len(idxs)}); expect long runtime."
                )
            for a_pos in range(len(idxs)):
                ia = idxs[a_pos]
                ha = meta[ia][2]
                for b_pos in range(a_pos + 1, len(idxs)):
                    ib = idxs[b_pos]
                    if uf.find(ia) == uf.find(ib):
                        continue
                    hb = meta[ib][2]
                    if _hamming_u64(ha, hb) > max_ahash_hamming:
                        continue
                    arr_a = cv2.imread(str(paths[ia]), cv2.IMREAD_COLOR)
                    arr_b = cv2.imread(str(paths[ib]), cv2.IMREAD_COLOR)
                    if arr_a is None or arr_b is None:
                        continue
                    try:
                        if (
                            pixel_similarity_bgr(arr_a, arr_b, channel_tol=channel_tol)
                            >= similarity
                        ):
                            uf.union(ia, ib)
                    finally:
                        del arr_a
                        del arr_b

    # Collect components with size > 1
    comp: dict[int, list[int]] = defaultdict(list)
    for i in range(len(paths)):
        comp[uf.find(i)].append(i)

    clusters: list[list[Path]] = []
    for members in comp.values():
        if len(members) < 2:
            continue
        clusters.append(sorted([paths[i] for i in members], key=lambda x: str(x)))
    return clusters


def _flat_base_from_image_path(img_path: Path) -> str:
    """``train__source__stem.jpg`` → ``train__source__stem``."""
    return img_path.stem


def dedupe_check_all_labels(
    bundle_root: Path,
    *,
    similarity: float = 0.99,
    channel_tol: int = 4,
    max_ahash_hamming: int = 0,
    dry_run: bool = False,
) -> dict[str, int]:
    """Remove duplicate triplets; keep lexicographically smallest basename per cluster."""
    bundle_root = bundle_root.resolve()
    images_d = bundle_root / "images"
    labels_d = bundle_root / "labels"
    overlay_d = bundle_root / "images_with_labels"
    if not images_d.is_dir():
        raise FileNotFoundError(f"missing images dir: {images_d}")

    clusters = find_duplicate_clusters(
        images_d,
        similarity=similarity,
        channel_tol=channel_tol,
        max_ahash_hamming=max_ahash_hamming,
    )
    removed = 0
    dry_run_lines = 0
    dry_run_cap = 40
    for cl in clusters:
        keep = min(cl, key=lambda p: _flat_base_from_image_path(p))
        keep_base = _flat_base_from_image_path(keep)
        for p in cl:
            if p == keep:
                continue
            base = _flat_base_from_image_path(p)
            if dry_run:
                if dry_run_lines < dry_run_cap:
                    print(f"[dry-run] would remove duplicate base={base!r} (keep {keep_base!r})")
                    dry_run_lines += 1
                elif dry_run_lines == dry_run_cap:
                    print("[dry-run] … (further lines suppressed; summary follows)")
                    dry_run_lines += 1
                removed += 1
                continue
            p.unlink(missing_ok=True)
            (labels_d / f"{base}.txt").unlink(missing_ok=True)
            (overlay_d / f"{base}.jpg").unlink(missing_ok=True)
            removed += 1

    summary = {
        "duplicate_clusters": len(clusters),
        "duplicate_samples_removed": removed,
        "dry_run": int(dry_run),
    }
    mode = "dry-run" if dry_run else "applied"
    print(
        f"[fifa_check_labels_dedupe] {mode}: clusters={summary['duplicate_clusters']}, "
        f"removed_samples={summary['duplicate_samples_removed']} "
        f"(each sample drops image + label + overlay when not dry-run)"
    )
    return summary


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vaila.fifa_check_labels_dedupe",
        description="Remove near-duplicate images from a check_all_labels bundle.",
    )
    p.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to check_all_labels (contains images/, labels/, images_with_labels/).",
    )
    p.add_argument(
        "--similarity",
        type=float,
        default=0.99,
        help="Minimum fraction of pixels matching in all channels (default: 0.99).",
    )
    p.add_argument(
        "--channel-tol",
        type=int,
        default=4,
        help="Max per-channel absolute difference for a pixel to count as match (default: 4).",
    )
    p.add_argument(
        "--max-ahash-hamming",
        type=int,
        default=0,
        help=(
            "Extra pass: max Hamming distance on 8×8 aHash within same resolution "
            "(default: 0 = disabled; only exact aHash buckets + same-file union). "
            "Values > 0 can be very slow on large datasets."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without deleting files.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    dedupe_check_all_labels(
        args.bundle,
        similarity=float(args.similarity),
        channel_tol=int(args.channel_tol),
        max_ahash_hamming=int(args.max_ahash_hamming),
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
