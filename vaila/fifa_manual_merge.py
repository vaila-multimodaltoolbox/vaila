"""Merge manually-labeled YOLO Pose data into the FIFA `unified/` dataset.

Update Date: 14 May 2026
Version: 0.3.44

This module reads a folder tree authored by human annotators (vailá
``getpixelvideo`` FIFA template / ``pose_dataset_<TS>`` / ``vaila_dataset_<TS>``)
and integrates the
valid (image + non-empty 32-kp label) pairs into the existing
``dataset_vaila_fifa/unified/`` YOLO Pose tree built by
:mod:`vaila.fifa_dataset_builder`, using the same staging-then-symlink
convention and appending provenance rows to ``unified/manifest.csv``.

Source layout it understands::

    <src>/<annotator>/<sequence>/(fifa_dataset_template|pose_dataset_<TS>|vaila_dataset_<TS>)/
        images/{train,val,test}/<stem>.<ext>
        labels/{train,val,test}/<stem>.txt    # 1 + 4 + 32*3 = 101 fields

Destination layout it writes to::

    <dst>/staging/vaila_manual__<annotator>__<sequence>__<dataset>/
        images/{train,val,test}/<new_stem>.<ext>
        labels/{train,val,test}/<new_stem>.txt
    <dst>/unified/{images,labels}/{train,val,test}/<new_stem>.<ext|txt>
        # absolute symlinks pointing at the staged files
    <dst>/unified/manifest.csv                     # append-only, schema:
                                                   # split,image,source,label
    <dst>/reports/manual_merge_<TS>.{log,csv}      # per-file decisions

The new stem is::

    <new_stem> = vaila_manual__<annotator>__<sequence>__<dataset>__<orig_stem>

so the manual labels never collide with names from
:mod:`vaila.fifa_dataset_builder`. Re-running the merge is idempotent: if the
target symlink already points at the same staged absolute path, the file is
counted as ``already_present`` and the manifest is not re-appended.

CLI::

    uv run python -m vaila.fifa_manual_merge \\
        --src /home/preto/data/FIFA/vaila_dataset \\
        --dst /home/preto/data/FIFA/dataset_vaila_fifa

GUI::

    uv run python -m vaila.fifa_manual_merge   # Tkinter directory dialogs
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .fifa_dataset_builder import (
        _IMAGE_EXTS_ORDERED,
        NUM_KEYPOINTS,
    )
except ImportError:
    from fifa_dataset_builder import (  # ty: ignore[unresolved-import]
        _IMAGE_EXTS_ORDERED,
        NUM_KEYPOINTS,
    )

EXPECTED_LINE_FIELDS: int = 1 + 4 + NUM_KEYPOINTS * 3  # 1 cls + 4 bbox + 32*(x,y,v) = 101
SPLITS: tuple[str, ...] = ("train", "val", "test")
SOURCE_PREFIX: str = "vaila_manual"
MANIFEST_HEADER: tuple[str, ...] = ("split", "image", "source", "label")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A single (image, label) pair detected under the source tree."""

    annotator: str
    sequence: str
    dataset: str  # e.g. ``pose_dataset_20260430_110714`` or ``fifa_dataset_template``
    split: str
    src_label: Path
    src_image: Path | None  # None when no matching image was found
    orig_stem: str

    @property
    def source_name(self) -> str:
        return f"{SOURCE_PREFIX}__{self.annotator}__{self.sequence}__{self.dataset}"

    @property
    def new_stem(self) -> str:
        return f"{self.source_name}__{self.orig_stem}"


@dataclass
class Decision:
    """Outcome of evaluating + processing a single candidate."""

    candidate: Candidate
    status: str  # added | already_present | skipped_* | dry_run | error
    reason: str = ""
    staged_image: Path | None = None
    staged_label: Path | None = None
    unified_image: Path | None = None
    unified_label: Path | None = None


@dataclass
class MergeResult:
    """Aggregate result returned by :func:`merge_manual_dataset`."""

    decisions: list[Decision] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    report_csv: Path | None = None
    report_log: Path | None = None
    manifest_csv: Path | None = None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _is_dataset_dir(name: str) -> bool:
    return (
        name == "fifa_dataset_template"
        or name.startswith("pose_dataset_")
        or name.startswith("vaila_dataset_")
    )


def _find_image_for_label(label_path: Path, dataset_dir: Path, split: str) -> Path | None:
    """Locate ``<dataset_dir>/images/<split>/<stem>.<ext>`` for any known image ext."""
    images_dir = dataset_dir / "images" / split
    stem = label_path.stem
    for ext in _IMAGE_EXTS_ORDERED:
        cand = images_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def discover_candidates(src_root: Path) -> list[Candidate]:
    """Walk ``<src_root>/<annotator>/<sequence>/<dataset>/labels/<split>/*.txt``."""
    src_root = src_root.expanduser().resolve()
    if not src_root.is_dir():
        raise NotADirectoryError(f"--src is not a directory: {src_root}")

    candidates: list[Candidate] = []
    for annotator_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        annotator = annotator_dir.name
        for sequence_dir in sorted(p for p in annotator_dir.iterdir() if p.is_dir()):
            sequence = sequence_dir.name
            dataset_dirs: list[Path] = []
            for sub in sorted(sequence_dir.iterdir()):
                if sub.is_dir() and _is_dataset_dir(sub.name):
                    dataset_dirs.append(sub)
                elif sub.is_dir():
                    # Some annotators nest one extra level (e.g. lennin/BRA_KOR/BRA_KOR_p1/)
                    for sub2 in sorted(sub.iterdir()) if sub.is_dir() else []:
                        if sub2.is_dir() and _is_dataset_dir(sub2.name):
                            dataset_dirs.append(sub2)
            for dataset_dir in dataset_dirs:
                # Use the parent chain back to the sequence root as the sequence id, so
                # nested layouts like sequence/sub_sequence/dataset are unambiguous.
                rel = dataset_dir.relative_to(sequence_dir).parent
                seq_id = sequence if rel == Path(".") else f"{sequence}__{'_'.join(rel.parts)}"
                for split in SPLITS:
                    labels_split = dataset_dir / "labels" / split
                    if not labels_split.is_dir():
                        continue
                    for lbl in sorted(labels_split.glob("*.txt")):
                        img = _find_image_for_label(lbl, dataset_dir, split)
                        candidates.append(
                            Candidate(
                                annotator=annotator,
                                sequence=seq_id,
                                dataset=dataset_dir.name,
                                split=split,
                                src_label=lbl,
                                src_image=img,
                                orig_stem=lbl.stem,
                            )
                        )
    return candidates


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_label_text(text: str) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for a YOLO-Pose label string (32 kp).

    ``ok`` is True only when there is at least one non-empty line and **every**
    non-empty line has exactly :data:`EXPECTED_LINE_FIELDS` whitespace-separated
    tokens that all parse as floats. ``reason`` is empty when ``ok`` is True.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False, "empty_label"
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != EXPECTED_LINE_FIELDS:
            return (
                False,
                f"line_{idx}_field_count_{len(parts)}_expected_{EXPECTED_LINE_FIELDS}",
            )
        try:
            [float(p) for p in parts]
        except ValueError:
            return False, f"line_{idx}_non_numeric_token"
    return True, ""


def _validate_candidate(c: Candidate) -> tuple[bool, str]:
    if not c.src_label.is_file():
        return False, "missing_label_file"
    if c.src_image is None:
        return False, "no_matching_image"
    try:
        text = c.src_label.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"read_error_{e.__class__.__name__}"
    return validate_label_text(text)


# ---------------------------------------------------------------------------
# Staging + linking
# ---------------------------------------------------------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stage_pair(
    c: Candidate,
    *,
    staging_root: Path,
    move: bool,
    dry_run: bool,
) -> tuple[Path, Path]:
    """Copy/move (image, label) into ``staging_root`` and return the new abs paths."""
    assert c.src_image is not None  # validated upstream
    img_ext = c.src_image.suffix
    out_img = staging_root / "images" / c.split / f"{c.new_stem}{img_ext}"
    out_lbl = staging_root / "labels" / c.split / f"{c.new_stem}.txt"

    if dry_run:
        return out_img.resolve(), out_lbl.resolve()

    _ensure_dir(out_img.parent)
    _ensure_dir(out_lbl.parent)

    op = shutil.move if move else shutil.copy2
    if not out_img.exists():
        op(str(c.src_image), str(out_img))
    if not out_lbl.exists():
        op(str(c.src_label), str(out_lbl))
    return out_img.resolve(), out_lbl.resolve()


def _existing_symlink_target(p: Path) -> Path | None:
    """Return the absolute symlink target of ``p`` if it is a symlink, else None."""
    if not p.is_symlink():
        return None
    try:
        target = os.readlink(p)
    except OSError:
        return None
    target_path = Path(target)
    if not target_path.is_absolute():
        target_path = (p.parent / target_path).resolve()
    return target_path.resolve()


def _link_unified(
    staged: Path,
    unified: Path,
    *,
    dry_run: bool,
) -> tuple[str, str]:
    """Create an absolute symlink ``unified -> staged`` (idempotent).

    Returns ``(status, reason)`` where ``status`` is one of
    ``"linked"``, ``"already_present"``, ``"skipped_collision"`` or
    ``"dry_run"``.
    """
    staged_abs = staged.resolve()
    if dry_run:
        if unified.is_symlink():
            cur = _existing_symlink_target(unified)
            if cur == staged_abs:
                return "already_present", ""
            return "skipped_collision", f"symlink->{cur}"
        if unified.exists():
            return "skipped_collision", "non_symlink_file_present"
        return "dry_run", ""

    _ensure_dir(unified.parent)
    if unified.is_symlink():
        cur = _existing_symlink_target(unified)
        if cur == staged_abs:
            return "already_present", ""
        return "skipped_collision", f"symlink->{cur}"
    if unified.exists():
        return "skipped_collision", "non_symlink_file_present"

    os.symlink(str(staged_abs), str(unified))
    return "linked", ""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _existing_manifest_keys(manifest_csv: Path) -> set[tuple[str, str]]:
    """Return ``(split, image)`` pairs already present in ``manifest_csv``.

    The vailá unified manifest convention allows the same image basename to
    appear in multiple splits (one row per ``(split, image)``); dedup must use
    that pair as the key.
    """
    if not manifest_csv.is_file():
        return set()
    seen: set[tuple[str, str]] = set()
    with manifest_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "").strip()
            img = row.get("image", "").strip()
            if split and img:
                seen.add((split, img))
    return seen


def _append_manifest_rows(
    manifest_csv: Path,
    rows: Iterable[tuple[str, str, str, str]],
    *,
    dry_run: bool,
) -> int:
    """Append ``(split, image, source, label)`` rows to ``manifest_csv``.

    Writes the header if the file does not yet exist. Returns number of rows
    actually written (0 in dry-run).
    """
    rows_list = list(rows)
    if dry_run or not rows_list:
        return 0
    write_header = not manifest_csv.is_file()
    _ensure_dir(manifest_csv.parent)
    with manifest_csv.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(MANIFEST_HEADER)
        for r in rows_list:
            w.writerow(r)
    return len(rows_list)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_report(
    decisions: list[Decision],
    *,
    reports_root: Path,
    timestamp: str,
    dry_run: bool,
) -> tuple[Path, Path]:
    _ensure_dir(reports_root)
    csv_path = reports_root / f"manual_merge_{timestamp}.csv"
    log_path = reports_root / f"manual_merge_{timestamp}.log"

    if dry_run:
        return csv_path, log_path

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            (
                "annotator",
                "sequence",
                "dataset",
                "split",
                "src_label",
                "src_image",
                "orig_stem",
                "new_stem",
                "status",
                "reason",
                "unified_image",
                "unified_label",
            )
        )
        for d in decisions:
            c = d.candidate
            w.writerow(
                (
                    c.annotator,
                    c.sequence,
                    c.dataset,
                    c.split,
                    str(c.src_label),
                    str(c.src_image) if c.src_image is not None else "",
                    c.orig_stem,
                    c.new_stem,
                    d.status,
                    d.reason,
                    str(d.unified_image) if d.unified_image is not None else "",
                    str(d.unified_label) if d.unified_label is not None else "",
                )
            )

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.status] = counts.get(d.status, 0) + 1
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# vaila.fifa_manual_merge report ({timestamp})\n")
        f.write(f"# total candidates: {len(decisions)}\n")
        for status in sorted(counts):
            f.write(f"  {status}: {counts[status]}\n")

    return csv_path, log_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def merge_manual_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    move: bool = False,
    dry_run: bool = False,
    verify_after: bool = True,
) -> MergeResult:
    """Merge manually-labeled data under ``src_root`` into ``<dst_root>/unified/``.

    Parameters
    ----------
    src_root:
        Path to the manually-labeled tree (e.g. ``/home/preto/data/FIFA/vaila_dataset``).
    dst_root:
        Path to the FIFA dataset root that contains ``unified/`` (e.g.
        ``/home/preto/data/FIFA/dataset_vaila_fifa``).
    move:
        If ``True``, move source files into ``staging/`` instead of copying.
    dry_run:
        If ``True``, no files are written; the report still summarizes what
        WOULD happen.
    verify_after:
        If ``True`` (and not ``dry_run``), call
        :func:`vaila.fifa_dataset_train_readiness.verify_unified_dataset` and
        attach its issues to the result log.
    """
    src_root = src_root.expanduser().resolve()
    dst_root = dst_root.expanduser().resolve()
    unified_root = dst_root / "unified"
    staging_parent = dst_root / "staging"
    reports_root = dst_root / "reports"
    manifest_csv = unified_root / "manifest.csv"

    if not unified_root.is_dir():
        raise FileNotFoundError(
            f"unified/ not found under --dst={dst_root}: expected {unified_root}"
        )
    data_yaml = unified_root / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data.yaml missing under unified/: {data_yaml}. Run vaila.fifa_dataset_builder first."
        )

    candidates = discover_candidates(src_root)
    seen_manifest = _existing_manifest_keys(manifest_csv)
    decisions: list[Decision] = []
    manifest_rows: list[tuple[str, str, str, str]] = []

    for c in candidates:
        ok, reason = _validate_candidate(c)
        if not ok:
            status = (
                "skipped_no_image"
                if reason == "no_matching_image"
                else f"skipped_invalid_label_{reason}"
                if reason not in {"empty_label", "missing_label_file"}
                else f"skipped_{reason}"
            )
            decisions.append(Decision(candidate=c, status=status, reason=reason))
            continue

        staging_root = staging_parent / c.source_name
        try:
            staged_img, staged_lbl = _stage_pair(
                c, staging_root=staging_root, move=move, dry_run=dry_run
            )
        except OSError as e:
            decisions.append(
                Decision(
                    candidate=c,
                    status="error",
                    reason=f"stage_{e.__class__.__name__}: {e}",
                )
            )
            continue

        ext = staged_img.suffix
        unified_img = unified_root / "images" / c.split / f"{c.new_stem}{ext}"
        unified_lbl = unified_root / "labels" / c.split / f"{c.new_stem}.txt"
        img_status, img_reason = _link_unified(staged_img, unified_img, dry_run=dry_run)
        if img_status == "skipped_collision":
            decisions.append(
                Decision(
                    candidate=c,
                    status="skipped_collision_image",
                    reason=img_reason,
                    staged_image=staged_img,
                    staged_label=staged_lbl,
                )
            )
            continue
        lbl_status, lbl_reason = _link_unified(staged_lbl, unified_lbl, dry_run=dry_run)
        if lbl_status == "skipped_collision":
            decisions.append(
                Decision(
                    candidate=c,
                    status="skipped_collision_label",
                    reason=lbl_reason,
                    staged_image=staged_img,
                    staged_label=staged_lbl,
                    unified_image=unified_img,
                )
            )
            continue

        if img_status == "linked" or lbl_status == "linked":
            final_status = "added"
        elif dry_run and (img_status, lbl_status) == ("dry_run", "dry_run"):
            final_status = "dry_run"
        else:
            final_status = "already_present"

        rel_lbl = unified_lbl.relative_to(unified_root).as_posix()
        # Append manifest rows for both freshly added pairs AND back-fill
        # any already-linked pair whose ``(split, image)`` is missing from
        # manifest.csv (idempotent).
        if final_status in ("added", "already_present"):
            image_basename = unified_img.name
            key = (c.split, image_basename)
            if key not in seen_manifest:
                manifest_rows.append((c.split, image_basename, c.source_name, rel_lbl))
                seen_manifest.add(key)

        decisions.append(
            Decision(
                candidate=c,
                status=final_status,
                reason="",
                staged_image=staged_img,
                staged_label=staged_lbl,
                unified_image=unified_img,
                unified_label=unified_lbl,
            )
        )

    written_rows = _append_manifest_rows(manifest_csv, manifest_rows, dry_run=dry_run)

    timestamp = _timestamp()
    report_csv, report_log = _write_report(
        decisions, reports_root=reports_root, timestamp=timestamp, dry_run=dry_run
    )

    counts: dict[str, int] = {"manifest_rows_written": written_rows}
    for d in decisions:
        counts[d.status] = counts.get(d.status, 0) + 1
    counts.setdefault("added", 0)
    counts["candidates_total"] = len(candidates)

    if verify_after and not dry_run:
        try:
            from .fifa_dataset_train_readiness import verify_unified_dataset
        except ImportError:
            from fifa_dataset_train_readiness import (  # ty: ignore[unresolved-import]
                verify_unified_dataset,
            )
        issues, vc = verify_unified_dataset(unified_root)
        with report_log.open("a", encoding="utf-8") as f:
            f.write("\n# verify_unified_dataset:\n")
            f.write(f"  labels_scanned={vc.get('labels', 0)}\n")
            f.write(f"  images_found={vc.get('images_found', 0)}\n")
            f.write(f"  images_missing={vc.get('images_missing', 0)}\n")
            if issues:
                f.write(f"  issues ({len(issues)}):\n")
                for msg in issues:
                    f.write(f"    - {msg}\n")
            else:
                f.write("  no issues reported.\n")
        counts["verify_issues"] = len(issues)
        counts["verify_labels"] = vc.get("labels", 0)
        counts["verify_images_found"] = vc.get("images_found", 0)
        counts["verify_images_missing"] = vc.get("images_missing", 0)

    return MergeResult(
        decisions=decisions,
        counts=counts,
        report_csv=report_csv,
        report_log=report_log,
        manifest_csv=manifest_csv,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vaila.fifa_manual_merge",
        description=(
            "Merge manually-labeled YOLO Pose data into <dst>/unified/ "
            "(staging+symlink+manifest, idempotent)."
        ),
    )
    p.add_argument(
        "--src",
        type=Path,
        default=None,
        help=(
            "Source root with <annotator>/<sequence>/<dataset>/{images,labels}/<split>/ . "
            "Default: ask via Tkinter dialog."
        ),
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=None,
        help=(
            "Destination root containing unified/ (and where staging/ + reports/ "
            "are written). Default: ask via Tkinter dialog."
        ),
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="Move source files into staging/ instead of copying (default: copy).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and validate but do not change any files.",
    )
    p.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-merge fifa_dataset_train_readiness verify step.",
    )
    return p


def _print_summary(result: MergeResult) -> None:
    print(f"[fifa_manual_merge] candidates={result.counts.get('candidates_total', 0)}")
    for k in sorted(result.counts):
        if k == "candidates_total":
            continue
        print(f"  {k}: {result.counts[k]}")
    if result.report_csv is not None:
        print(f"[fifa_manual_merge] report csv: {result.report_csv}")
    if result.report_log is not None:
        print(f"[fifa_manual_merge] report log: {result.report_log}")
    if result.manifest_csv is not None:
        print(f"[fifa_manual_merge] manifest:   {result.manifest_csv}")


def _gui_pick_dirs() -> tuple[Path | None, Path | None, bool, bool]:
    """Tkinter directory dialogs. Returns ``(src, dst, move, dry_run)``."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    src_str = filedialog.askdirectory(
        title="FIFA manual merge: select SOURCE (manually-labeled root)"
    )
    if not src_str:
        root.destroy()
        return None, None, False, False
    dst_str = filedialog.askdirectory(
        title="FIFA manual merge: select DESTINATION (contains unified/)"
    )
    if not dst_str:
        root.destroy()
        return None, None, False, False
    move = bool(
        messagebox.askyesno(
            "Move or copy?",
            "Yes = MOVE source files into staging/ (frees disk).\n"
            "No  = COPY source files into staging/ (keeps source intact, default).",
            default=messagebox.NO,
        )
    )
    dry_run = bool(
        messagebox.askyesno(
            "Dry run?",
            "Yes = report only, do not change any files.\nNo  = perform the merge (default).",
            default=messagebox.NO,
        )
    )
    root.destroy()
    return Path(src_str), Path(dst_str), move, dry_run


def run_manual_merge() -> int:
    """GUI entry point used by ``vaila.py``. Returns shell-style exit code."""
    src, dst, move, dry_run = _gui_pick_dirs()
    if src is None or dst is None:
        print("[fifa_manual_merge] cancelled by user.")
        return 1
    print("\n" + "=" * 60)
    print(f"[fifa_manual_merge] src   = {src}")
    print(f"[fifa_manual_merge] dst   = {dst}")
    print(
        f"[fifa_manual_merge] mode  = {'move' if move else 'copy'}{' (dry-run)' if dry_run else ''}"
    )
    print("=" * 60 + "\n")
    try:
        result = merge_manual_dataset(
            src, dst, move=move, dry_run=dry_run, verify_after=not dry_run
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[fifa_manual_merge] ERROR: {e}")
        return 2
    _print_summary(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.src is None or args.dst is None:
        return run_manual_merge()
    try:
        result = merge_manual_dataset(
            args.src,
            args.dst,
            move=args.move,
            dry_run=args.dry_run,
            verify_after=not args.no_verify,
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"[fifa_manual_merge] ERROR: {e}", file=sys.stderr)
        return 2
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
