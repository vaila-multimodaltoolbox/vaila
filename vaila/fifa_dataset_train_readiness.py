"""Verify (and optionally align) a ``unified/`` YOLO-Pose tree for Ultralytics training.

``check_all_labels/`` is a **flat QA export** (``train__<stem>.jpg``). Training
reads ``unified/data.yaml`` → ``unified/images/{train,val,test}/``. Removing
duplicates only under ``check_all_labels/`` does **not** change ``unified/``;
use ``--prune-unified-to-flat`` once the flat tree is your source of truth.

This dataset is the **32 soccer-pitch keypoints** layout (Roboflow / vailá
``soccerfield_keypoints_ai``). The FIFA **Skeletal Tracking** body challenge
uses a different joint set — if you retargeted keypoint semantics for that
task, ``kpt_shape`` and the model head must match your new label format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .fifa_dataset_builder import (
        NUM_KEYPOINTS,
        find_unified_image_for_label,
    )
except ImportError:
    from fifa_dataset_builder import (  # ty: ignore[unresolved-import]
        NUM_KEYPOINTS,
        find_unified_image_for_label,
    )

_EXPECTED_LINE_FIELDS = 1 + 4 + NUM_KEYPOINTS * 3


def _parse_data_yaml_paths(data_yaml: Path) -> tuple[Path, dict[str, str]]:
    """Return (unified_root, {split: relative_dir}) from a vailá ``data.yaml``."""
    text = data_yaml.read_text(encoding="utf-8")
    unified_root: Path | None = None
    rel: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if key == "path":
            unified_root = Path(val)
        elif key in ("train", "val", "test"):
            rel[key] = val
    if unified_root is None:
        raise ValueError(f"no path: key in {data_yaml}")
    return unified_root.resolve(), rel


def collect_flat_image_stems(flat_images_dir: Path) -> set[str]:
    """Stems of ``train__foo`` style files under ``check_all_labels/images/``."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp", ".WEBP"}
    out: set[str] = set()
    for p in flat_images_dir.iterdir():
        if p.is_file() and p.suffix in exts:
            out.add(p.stem)
    return out


def verify_unified_dataset(
    unified_root: Path,
    *,
    max_label_errors_printed: int = 30,
) -> tuple[list[str], dict[str, int]]:
    """Check images ↔ labels and YOLO-Pose line width. Returns (issues, counts)."""
    unified_root = unified_root.resolve()
    counts: dict[str, int] = {"labels": 0, "images_found": 0, "images_missing": 0}
    issues: list[str] = []
    data_yaml = unified_root / "data.yaml"
    if not data_yaml.is_file():
        issues.append(f"missing data.yaml: {data_yaml}")
        return issues, counts

    try:
        yaml_root, _rel = _parse_data_yaml_paths(data_yaml)
    except ValueError as e:
        issues.append(str(e))
        return issues, counts
    if yaml_root != unified_root:
        issues.append(
            f"data.yaml path: ({yaml_root}) != unified_root ({unified_root}) — "
            "fix path: in data.yaml or pass the directory that contains it."
        )

    text = data_yaml.read_text(encoding="utf-8")
    if f"kpt_shape: [{NUM_KEYPOINTS}, 3]" not in text and "kpt_shape:" in text:
        issues.append(
            f"data.yaml kpt_shape should be [{NUM_KEYPOINTS}, 3] for this pitch schema "
            "(edit labels + yaml if you changed keypoint count for another task)."
        )
    elif "kpt_shape:" not in text:
        issues.append("data.yaml missing kpt_shape")

    for split in ("train", "val", "test"):
        lbl_dir = unified_root / "labels" / split
        if not lbl_dir.is_dir():
            issues.append(f"missing labels dir: {lbl_dir}")
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            counts["labels"] += 1
            stem = lbl.stem
            img = find_unified_image_for_label(unified_root, split, stem)
            if img is None:
                counts["images_missing"] += 1
                msg = f"[{split}] label without image: {lbl.name}"
                if len(issues) < max_label_errors_printed:
                    issues.append(msg)
                elif len(issues) == max_label_errors_printed:
                    issues.append("… (further label/image mismatches omitted)")
            else:
                counts["images_found"] += 1

            lines = lbl.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                issues.append(f"[{split}] empty label: {lbl.name}")
                continue
            parts = lines[0].split()
            if len(parts) != _EXPECTED_LINE_FIELDS:
                issues.append(
                    f"[{split}] {lbl.name}: expected {_EXPECTED_LINE_FIELDS} fields, got {len(parts)}"
                )

    return issues, counts


def prune_unified_to_match_flat(
    unified_root: Path,
    flat_root: Path,
    *,
    dry_run: bool = True,
) -> dict[str, int]:
    """Delete ``unified`` triplets whose ``{split}__{stem}`` key is absent from flat ``images/``."""
    unified_root = unified_root.resolve()
    flat_root = flat_root.resolve()
    flat_keys = collect_flat_image_stems(flat_root / "images")
    stats = {"would_delete": 0, "deleted": 0, "flat_keys": len(flat_keys)}
    for split in ("train", "val", "test"):
        lbl_dir = unified_root / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for lbl in list(lbl_dir.glob("*.txt")):
            key = f"{split}__{lbl.stem}"
            if key in flat_keys:
                continue
            img = find_unified_image_for_label(unified_root, split, lbl.stem)
            if dry_run:
                stats["would_delete"] += 1
                continue
            lbl.unlink(missing_ok=True)
            if img is not None:
                img.unlink(missing_ok=True)
            stats["deleted"] += 1
    mode = "dry-run" if dry_run else "applied"
    n = stats["would_delete"] if dry_run else stats["deleted"]
    print(
        f"[prune-unified-to-flat] {mode}: removed_triplets={n}, "
        f"flat_reference_images={stats['flat_keys']}"
    )
    if not dry_run and n:
        print(
            "[prune-unified-to-flat] note: manifest.csv (if present) is not auto-updated; "
            "re-run fifa_dataset_builder merge or edit it if you rely on provenance."
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Verify unified/ YOLO-Pose dataset for training; optional prune vs check_all_labels.",
    )
    p.add_argument(
        "--unified",
        type=Path,
        default=None,
        help="Path to unified/ (directory containing data.yaml).",
    )
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Explicit data.yaml (parent is used as unified root if --unified omitted).",
    )
    p.add_argument(
        "--compare-flat",
        type=Path,
        default=None,
        metavar="DIR",
        help="check_all_labels dir: warn if image stem count ≠ unified label count.",
    )
    p.add_argument(
        "--prune-unified-to-flat",
        type=Path,
        default=None,
        metavar="DIR",
        help="Remove unified samples missing from this flat check_all_labels/images/.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="With --prune-unified-to-flat: only print counts (default for prune).",
    )
    p.add_argument(
        "--apply-prune",
        action="store_true",
        help="With --prune-unified-to-flat: actually delete files (otherwise dry-run).",
    )
    args = p.parse_args(argv)
    exit_code = 0

    if args.data_yaml is not None:
        dy = args.data_yaml.expanduser().resolve()
        unified = dy.parent
    elif args.unified is not None:
        unified = args.unified.expanduser().resolve()
    else:
        p.error("pass --unified DIR or --data-yaml path/to/data.yaml")

    issues, counts = verify_unified_dataset(unified)
    print(
        f"[verify] unified={unified}\n"
        f"  labels_scanned={counts['labels']}, images_found={counts['images_found']}, "
        f"images_missing={counts['images_missing']}"
    )
    if issues:
        exit_code = 1
        print("[verify] issues:")
        for msg in issues:
            print(f"  - {msg}")
    else:
        print("[verify] no structural issues reported.")

    if args.compare_flat is not None:
        flat = args.compare_flat.expanduser().resolve()
        flat_n = len(collect_flat_image_stems(flat / "images"))
        u_n = counts["labels"]
        if flat_n != u_n:
            exit_code = 1
            print(
                f"[compare-flat] WARNING: flat_images={flat_n} vs unified_labels={u_n} "
                f"(diff={u_n - flat_n}). Training uses unified/ — align with "
                f"--prune-unified-to-flat after deduping check_all_labels."
            )
        else:
            print(f"[compare-flat] flat_images={flat_n} matches unified_labels={u_n}.")

    if args.prune_unified_to_flat is not None:
        dry = not bool(args.apply_prune)
        if args.apply_prune and args.dry_run:
            print("[prune] --apply-prune wins over --dry-run", file=sys.stderr)
            dry = False
        prune_unified_to_match_flat(
            unified,
            args.prune_unified_to_flat.expanduser().resolve(),
            dry_run=dry,
        )
        if not dry:
            issues2, counts2 = verify_unified_dataset(unified)
            print(
                f"[verify after prune] labels={counts2['labels']}, "
                f"images_missing={counts2['images_missing']}"
            )
            if issues2:
                for msg in issues2[:20]:
                    print(f"  - {msg}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
