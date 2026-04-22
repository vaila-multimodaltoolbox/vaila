"""Post-process SAM 3 batch output to produce vailá-format pixel CSVs.

Reads the artifacts written by :mod:`vaila.vaila_sam` for each video:

    {sam_dir}/sam_frames_meta.csv      # wide table with normalized bbox + prob per obj_id
    {sam_dir}/masks/frame_{f:06d}_obj_{oid}.png   # binary mask in original pixel space
    {sam_dir}/{stem}_sam_overlay.mp4   # used only to read frame width/height

and writes:

    {sam_dir}/sam_points.csv   # 'frame, p1_x, p1_y, p1_cx, p1_cy, p1_mx, p1_my, ...'
                               # where pN follows the sorted list of obj_ids that
                               # appeared in the video. The canonical (x, y) pair is
                               # the **bottom-center** of the bbox in PIXELS so it
                               # can be loaded directly by ``getpixelvideo`` and
                               # ``rec2d`` (which both expect pN_x/pN_y in pixels).
                               # Extra columns (cx/cy = bbox center, mx/my = mask
                               # centroid) are ignored by getpixelvideo / rec2d but
                               # are consumed by ``vaila/sam_validate.py``.

    {sam_dir}/sam_id_map.csv   # 'pN, obj_id, n_frames, first_frame, last_frame'

The bbox in ``sam_frames_meta.csv`` is normalized (fractions of W and H, see
:mod:`vaila.vaila_sam`). All exported coordinates are in **pixels** at the
original video resolution.

Author: Paulo R. P. Santiago - vaila project
Created: 19 April 2026
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pandas as pd

__all__ = [
    "SamRunArtifacts",
    "discover_sam_run",
    "read_sam_meta",
    "frame_size",
    "mask_centroid",
    "extract_points_from_sam_run",
    "extract_points_for_batch",
]

PointMode = Literal["foot", "center", "mask", "all"]


@dataclass(frozen=True)
class SamRunArtifacts:
    sam_dir: Path
    meta_csv: Path
    masks_dir: Path
    overlay_mp4: Path | None
    stem: str


def discover_sam_run(sam_dir: Path) -> SamRunArtifacts:
    """Locate canonical SAM3 artifacts inside a per-video output directory."""
    sam_dir = Path(sam_dir)
    if not sam_dir.is_dir():
        raise FileNotFoundError(f"SAM run directory not found: {sam_dir}")

    meta = sam_dir / "sam_frames_meta.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"sam_frames_meta.csv missing in {sam_dir}")

    masks_dir = sam_dir / "masks"
    if not masks_dir.is_dir():
        masks_dir = sam_dir

    overlays = sorted(sam_dir.glob("*_sam_overlay.mp4"))
    overlay = overlays[0] if overlays else None
    stem = overlay.stem.removesuffix("_sam_overlay") if overlay else sam_dir.name

    return SamRunArtifacts(
        sam_dir=sam_dir,
        meta_csv=meta,
        masks_dir=masks_dir,
        overlay_mp4=overlay,
        stem=stem,
    )


def read_sam_meta(sam_dir: Path) -> tuple[pd.DataFrame, list[int]]:
    """Read ``sam_frames_meta.csv`` and return (df, sorted_obj_ids).

    Cells without observation are read as NaN. The returned list of obj_ids is
    parsed from the ``box_x_{oid}`` column names, sorted ascending.
    """
    art = discover_sam_run(sam_dir)
    df = pd.read_csv(art.meta_csv)
    pat = re.compile(r"^box_x_(\d+)$")
    oids = sorted(int(m.group(1)) for c in df.columns for m in [pat.match(c)] if m)
    if "frame" not in df.columns:
        raise ValueError(f"sam_frames_meta.csv missing 'frame' column: {art.meta_csv}")
    return df, oids


def frame_size(art: SamRunArtifacts) -> tuple[int, int]:
    """Return (width, height) of the original video in pixels.

    Tries the overlay MP4 first, then falls back to any mask PNG.
    """
    if art.overlay_mp4 is not None and art.overlay_mp4.is_file():
        cap = cv2.VideoCapture(str(art.overlay_mp4))
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()
        if w > 0 and h > 0:
            return w, h

    pngs = sorted(art.masks_dir.glob("frame_*_obj_*.png"))
    if pngs:
        m = cv2.imread(str(pngs[0]), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return int(m.shape[1]), int(m.shape[0])

    raise RuntimeError(
        f"Cannot determine frame size for SAM run at {art.sam_dir}: "
        "no readable overlay MP4 nor mask PNG."
    )


def mask_centroid(png_path: Path) -> tuple[float, float] | None:
    """Return (cx, cy) in pixels for a binary mask PNG, or ``None`` if empty."""
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    m = cv2.moments(binary, binaryImage=True)
    area = m.get("m00", 0.0)
    if area <= 0.0:
        return None
    return float(m["m10"] / area), float(m["m01"] / area)


def _canonical_pair(
    foot: tuple[float, float] | None,
    center: tuple[float, float] | None,
    mask: tuple[float, float] | None,
    canonical: Literal["foot", "center", "mask"],
) -> tuple[float, float] | None:
    if canonical == "foot":
        return foot
    if canonical == "center":
        return center
    return mask


def _format_cell(v: float | None) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.4f}"


def extract_points_from_sam_run(
    sam_dir: Path,
    *,
    mode: PointMode = "all",
    canonical: Literal["foot", "center", "mask"] = "foot",
    out_csv: Path | None = None,
    out_id_map: Path | None = None,
) -> Path:
    """Build ``sam_points.csv`` and ``sam_id_map.csv`` for one SAM run.

    Parameters
    ----------
    sam_dir:
        Path of a per-video SAM3 output directory (the one that contains
        ``sam_frames_meta.csv`` and ``masks/``).
    mode:
        Which point columns to write per ``pN``:

        * ``"foot"``   - only ``pN_x, pN_y`` (canonical bottom-center).
        * ``"center"`` - only ``pN_x, pN_y`` (canonical bbox center).
        * ``"mask"``   - only ``pN_x, pN_y`` (canonical mask centroid).
        * ``"all"``    - canonical pair plus ``pN_cx, pN_cy, pN_mx, pN_my``.
    canonical:
        Which point becomes the canonical ``pN_x, pN_y`` pair (consumed by
        ``getpixelvideo`` and ``rec2d``).
    out_csv, out_id_map:
        Override default output paths (defaults: ``sam_dir/sam_points.csv``
        and ``sam_dir/sam_id_map.csv``).
    """
    art = discover_sam_run(sam_dir)
    df, oids = read_sam_meta(sam_dir)
    width, height = frame_size(art)

    out_csv = out_csv or (art.sam_dir / "sam_points.csv")
    out_id_map = out_id_map or (art.sam_dir / "sam_id_map.csv")

    n_frames = len(df)
    id_to_pn = {oid: pn for pn, oid in enumerate(oids, start=1)}

    header = ["frame"]
    for pn in range(1, len(oids) + 1):
        header.extend([f"p{pn}_x", f"p{pn}_y"])
        if mode == "all":
            header.extend([f"p{pn}_cx", f"p{pn}_cy", f"p{pn}_mx", f"p{pn}_my"])

    rows: list[str] = []
    id_stats: dict[int, dict[str, int]] = {oid: {"n": 0, "first": -1, "last": -1} for oid in oids}

    for _, row in df.iterrows():
        frame_idx = int(row["frame"])
        cells: list[str] = [str(frame_idx)]
        for oid in oids:
            bx = row.get(f"box_x_{oid}", float("nan"))
            by = row.get(f"box_y_{oid}", float("nan"))
            bw = row.get(f"box_w_{oid}", float("nan"))
            bh = row.get(f"box_h_{oid}", float("nan"))
            present = all(np.isfinite([bx, by, bw, bh]))

            foot: tuple[float, float] | None
            center: tuple[float, float] | None
            mask_c: tuple[float, float] | None
            if present:
                px = float(bx) * width
                py = float(by) * height
                pw = float(bw) * width
                ph = float(bh) * height
                center = (px + pw * 0.5, py + ph * 0.5)
                foot = (px + pw * 0.5, py + ph)
                stats = id_stats[oid]
                stats["n"] += 1
                if stats["first"] < 0:
                    stats["first"] = frame_idx
                stats["last"] = frame_idx
            else:
                foot = center = None

            if present and mode in ("mask", "all"):
                png = art.masks_dir / f"frame_{frame_idx:06d}_obj_{oid}.png"
                mask_c = mask_centroid(png) if png.is_file() else None
            else:
                mask_c = None

            canonical_pt = _canonical_pair(foot, center, mask_c, canonical)
            cells.append(_format_cell(canonical_pt[0] if canonical_pt else None))
            cells.append(_format_cell(canonical_pt[1] if canonical_pt else None))
            if mode == "all":
                cells.append(_format_cell(center[0] if center else None))
                cells.append(_format_cell(center[1] if center else None))
                cells.append(_format_cell(mask_c[0] if mask_c else None))
                cells.append(_format_cell(mask_c[1] if mask_c else None))
        rows.append(",".join(cells))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text(
        ",".join(header) + "\n" + "\n".join(rows) + ("\n" if rows else ""), encoding="utf-8"
    )

    id_rows = ["pN,obj_id,n_frames,first_frame,last_frame"]
    for oid in oids:
        s = id_stats[oid]
        id_rows.append(
            f"{id_to_pn[oid]},{oid},{s['n']},{s['first'] if s['n'] else ''},{s['last'] if s['n'] else ''}"
        )
    out_id_map.write_text("\n".join(id_rows) + "\n", encoding="utf-8")

    print(
        f"[sam_postprocess] {art.sam_dir.name}: {len(oids)} ids, {n_frames} frames, "
        f"mode={mode}, canonical={canonical}, W={width}, H={height} -> {out_csv.name}"
    )
    return out_csv


def extract_points_for_batch(batch_root: Path, **kwargs) -> list[Path]:
    """Run :func:`extract_points_from_sam_run` on every per-video subdir.

    A subdir is recognized as a SAM run if it contains ``sam_frames_meta.csv``.
    """
    batch_root = Path(batch_root)
    if not batch_root.is_dir():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")
    out: list[Path] = []
    for child in sorted(batch_root.iterdir()):
        if child.is_dir() and (child / "sam_frames_meta.csv").is_file():
            try:
                out.append(extract_points_from_sam_run(child, **kwargs))
            except Exception as exc:
                print(f"[sam_postprocess] FAILED {child.name}: {exc}")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build vailá pixel CSVs from a SAM3 batch output")
    p.add_argument(
        "path",
        type=Path,
        help="Either a single per-video SAM dir or a batch root containing many.",
    )
    p.add_argument(
        "--mode",
        choices=["foot", "center", "mask", "all"],
        default="all",
        help="Which point columns to emit (default: all).",
    )
    p.add_argument(
        "--canonical",
        choices=["foot", "center", "mask"],
        default="foot",
        help="Which point becomes the canonical pN_x/pN_y (default: foot).",
    )
    args = p.parse_args(argv)

    target = Path(args.path).resolve()
    if (target / "sam_frames_meta.csv").is_file():
        extract_points_from_sam_run(target, mode=args.mode, canonical=args.canonical)
    else:
        outs = extract_points_for_batch(target, mode=args.mode, canonical=args.canonical)
        if not outs:
            print(f"[sam_postprocess] No SAM runs found under {target}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
