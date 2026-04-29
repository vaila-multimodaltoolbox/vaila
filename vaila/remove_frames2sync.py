"""Remove duplicate PNG frames and estimate per-camera FPS.

This module removes duplicated consecutive frames in PNG sequences (commonly
introduced when platforms upsample lower-fps camera clips into a higher-fps
container video). It can:

- Detect duplicates by exact pixel hash ("exact") or perceptual hash ("phash")
- Produce cleaned sequences (kept frames) **without modifying originals** (default)
- Optionally apply changes **in-place** (move/delete duplicates + renumber)
- Estimate each camera's FPS using an **anchor** sequence with known FPS

CLI examples:
    # Process a parent folder containing multiple sequence subfolders (safe default):
    uv run python vaila/remove_frames2sync.py /path/to/root --anchor seq1 --anchor-fps 50

    # Dry run (scan + reports only, do not write cleaned frames):
    uv run python vaila/remove_frames2sync.py /path/to/root --anchor seq1 --anchor-fps 50 --dry-run

    # Apply destructive changes in-place (opt-in):
    uv run python vaila/remove_frames2sync.py /path/to/root --anchor seq1 --anchor-fps 50 --in-place
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False


# ─── Image hashing ────────────────────────────────────────────────────────────


def md5_hash(path: Path) -> str:
    """Exact pixel MD5 hash. Fast and reliable for lossless PNGs."""
    with Image.open(path) as img:
        arr = np.array(img)
    return hashlib.md5(arr.tobytes()).hexdigest()


def perceptual_hash(path: Path) -> int:
    """
    8x8 average perceptual hash (no external library needed).
    Returns a 64-bit integer. Use hamming_distance to compare.
    """
    with Image.open(path) as img:
        # Pillow >= 10 prefers Image.Resampling.LANCZOS; keep compatibility.
        resampling = getattr(Image, "Resampling", None)
        resample_filter = getattr(Image, "LANCZOS", 1) if resampling is None else resampling.LANCZOS
        small = img.convert("L").resize((8, 8), resample_filter)
    arr = np.array(small, dtype=np.float32)
    mean = arr.mean()
    bits = (arr > mean).flatten()
    val = 0
    for bit in bits:
        val = (val << 1) | int(bit)
    return val


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ─── Core processing ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DedupeItem:
    orig_name: str
    status: str  # "kept" | "removed"
    dup_of_orig_name: str | None
    hash_method: str
    hash_value: str
    hamming: int | None


def _list_png_files(folder: Path) -> list[Path]:
    pngs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    if not pngs:
        return []

    non_numeric = [p.name for p in pngs if not p.stem.isdigit()]
    if non_numeric:
        example = ", ".join(non_numeric[:5])
        raise ValueError(
            f"PNG filenames must be numeric (e.g. 000000001.png). "
            f"Found non-numeric stems in {folder.name}: {example}"
        )

    return sorted(pngs, key=lambda p: int(p.stem))


def detect_duplicates_with_manifest(
    png_files: list[Path],
    *,
    mode: str = "exact",
    phash_threshold: int = 2,
    show_progress: bool = True,
) -> tuple[list[Path], list[Path], list[DedupeItem]]:
    """Detect consecutive duplicates and produce a per-frame manifest."""
    duplicates: list[Path] = []
    remaining: list[Path] = []
    manifest: list[DedupeItem] = []

    prev_val: str | int | None = None
    prev_path: Path | None = None

    total = len(png_files)
    for i, path in enumerate(png_files):
        if mode == "exact":
            curr_val = md5_hash(path)
            curr_val_str = curr_val
            ham = None
            is_dup = prev_val is not None and curr_val == prev_val
        else:
            curr_val_int = perceptual_hash(path)
            curr_val = curr_val_int
            curr_val_str = f"{curr_val_int:016x}"
            ham = None
            is_dup = False
            if prev_val is not None and isinstance(prev_val, int):
                ham = hamming_distance(curr_val_int, prev_val)
                is_dup = ham <= phash_threshold

        if is_dup:
            duplicates.append(path)
            manifest.append(
                DedupeItem(
                    orig_name=path.name,
                    status="removed",
                    dup_of_orig_name=prev_path.name if prev_path else None,
                    hash_method=mode,
                    hash_value=curr_val_str,
                    hamming=ham,
                )
            )
        else:
            remaining.append(path)
            manifest.append(
                DedupeItem(
                    orig_name=path.name,
                    status="kept",
                    dup_of_orig_name=None,
                    hash_method=mode,
                    hash_value=curr_val_str,
                    hamming=ham,
                )
            )

        prev_val = curr_val
        prev_path = path

        if show_progress and ((i + 1) % 200 == 0 or (i + 1) == total):
            print(f"    Scanned {i + 1:>5}/{total}...", end="\r", flush=True)

    if show_progress:
        print()
    return duplicates, remaining, manifest


def process_folder(
    folder: Path,
    dry_run: bool = False,
    delete_duplicates: bool = False,
    source_fps: float = 60.0,
    *,
    out_dir: Path | None = None,
    in_place: bool = False,
    mode: str = "exact",
    phash_threshold: int = 2,
    verbose: bool = True,
) -> dict | None:
    """
    Process one sequence folder:
      1. Collect PNGs sorted numerically
      2. Detect duplicate consecutive frames
      3. Move/delete duplicates
      4. Renumber remaining frames sequentially
      5. Return stats dict
    """
    png_files = _list_png_files(folder)

    if not png_files:
        if verbose:
            print(f"  [SKIP] No PNG files in: {folder.name}")
        return None

    total = len(png_files)
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Folder  : {folder.name}")
        print(f"  PNGs    : {total}")
        print(
            f"  Mode    : {mode}" + (f" (threshold={phash_threshold})" if mode == "phash" else "")
        )

    duplicates, remaining, manifest = detect_duplicates_with_manifest(
        png_files,
        mode=mode,
        phash_threshold=phash_threshold,
        show_progress=bool(verbose),
    )

    n_removed = len(duplicates)
    n_remaining = len(remaining)
    ratio = n_remaining / total if total > 0 else 1.0
    estimated_fps = source_fps * ratio

    if verbose:
        print(f"  Duplicates found : {n_removed}")
        print(f"  Frames remaining : {n_remaining}")
        print(f"  FPS ratio        : {n_remaining}/{total} = {ratio:.4f}")
        print(f"  Estimated cam fps: {estimated_fps:.2f} fps  (source: {source_fps} fps)")

    if dry_run:
        if verbose:
            print("  [DRY RUN] No files modified.")
        return _make_result(
            folder,
            total,
            n_removed,
            n_remaining,
            estimated_fps,
            ratio,
            manifest=manifest,
        )

    if in_place:
        # ── Move or delete duplicates ────────────────────────────────────────
        if n_removed > 0:
            if delete_duplicates:
                for dup in duplicates:
                    dup.unlink()
                if verbose:
                    print(f"  Deleted {n_removed} duplicate frames.")
            else:
                removed_dir = folder / "removed_frames"
                removed_dir.mkdir(exist_ok=True)
                for dup in duplicates:
                    shutil.move(str(dup), str(removed_dir / dup.name))
                if verbose:
                    print(f"  Moved {n_removed} duplicates → removed_frames/")

        # ── Renumber remaining frames sequentially ───────────────────────────
        temp_paths: list[Path] = []
        for i, src in enumerate(remaining):
            tmp = folder / f"__tmp_{i:010d}.png"
            src.rename(tmp)
            temp_paths.append(tmp)

        for i, tmp in enumerate(temp_paths):
            final = folder / f"{i + 1:09d}.png"
            tmp.rename(final)

        if verbose:
            print(f"  Renumbered 1 → {n_remaining} (000000001.png … {n_remaining:09d}.png)")
    else:
        if out_dir is None:
            raise ValueError("out_dir is required when in_place=False")

        kept_dir = out_dir / "kept_frames" / folder.name
        removed_dir = out_dir / "removed_frames" / folder.name
        kept_dir.mkdir(parents=True, exist_ok=True)
        removed_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(remaining):
            dst = kept_dir / f"{i + 1:09d}.png"
            shutil.copy2(src, dst)

        for src in duplicates:
            shutil.copy2(src, removed_dir / src.name)

        if verbose:
            print(f"  Wrote kept frames   → {kept_dir}")
            print(f"  Wrote removed frames→ {removed_dir}")

    return _make_result(
        folder,
        total,
        n_removed,
        n_remaining,
        estimated_fps,
        ratio,
        manifest=manifest,
    )


def _make_result(folder, total, n_removed, n_remaining, estimated_fps, ratio, *, manifest):
    return {
        "folder": folder.name,
        "total": total,
        "removed": n_removed,
        "remaining": n_remaining,
        "estimated_fps": round(estimated_fps, 4),
        "ratio": round(ratio, 6),
        "manifest": manifest,
    }


# ─── Report ───────────────────────────────────────────────────────────────────


def print_summary(results: list[dict], *, anchor_folder: str | None, anchor_fps: float | None):
    W = 70
    print(f"\n{'=' * W}")
    print("  SYNCHRONIZATION SUMMARY")
    print(f"{'=' * W}")
    print(
        f"  {'Folder':<38} {'Total':>6} {'Removed':>8} {'Left':>6} "
        f"{'FPS(src)':>9} {'FPS(anchor)':>11}"
    )
    print(
        f"  {'-' * 38} {'------':>6} {'-------':>8} {'------':>6} {'--------':>9} {'----------':>11}"
    )
    for r in results:
        is_anchor = anchor_folder is not None and r["folder"] == anchor_folder
        flag = "  ← anchor" if is_anchor else ""
        fps_anchor = r.get("estimated_fps_from_anchor")
        fps_anchor_str = (
            f"{fps_anchor:>11.2f}" if isinstance(fps_anchor, (int, float)) else f"{'':>11}"
        )
        print(
            f"  {r['folder'][:38]:<38} {r['total']:>6} {r['removed']:>8} "
            f"{r['remaining']:>6} {r['estimated_fps']:>9.2f} {fps_anchor_str}{flag}"
        )
    print(f"{'=' * W}")


def save_reports(report_dir: Path, results: list[dict], *, dry_run: bool) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "DRYRUN_" if dry_run else ""

    # TXT
    txt_path = report_dir / f"{prefix}sync_report_{ts}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SYNCHRONIZATION REPORT\n")
        f.write(f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dry run   : {dry_run}\n\n")
        f.write(
            f"{'Folder':<42} {'Total':>6} {'Removed':>8} {'Left':>6} "
            f"{'FPS_src':>10} {'Ratio':>8} {'FPS_anchor':>12}\n"
        )
        f.write("-" * 98 + "\n")
        for r in results:
            fps_anchor = r.get("estimated_fps_from_anchor")
            fps_anchor_str = f"{float(fps_anchor):.4f}" if fps_anchor is not None else ""
            f.write(
                f"{r['folder'][:42]:<42} {r['total']:>6} {r['removed']:>8} "
                f"{r['remaining']:>6} {r['estimated_fps']:>10.4f} {r['ratio']:>8.6f} "
                f"{fps_anchor_str:>12}\n"
            )

    # CSV
    csv_path = report_dir / f"{prefix}sync_report_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "total",
                "removed",
                "remaining",
                "estimated_fps",
                "ratio",
                "anchor_folder",
                "anchor_fps",
                "anchor_remaining",
                "estimated_fps_from_anchor",
            ],
        )
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in writer.fieldnames}
            writer.writerow(row)

    return txt_path, csv_path


def _write_manifest(manifest_dir: Path, folder_name: str, items: list[DedupeItem]) -> Path:
    manifest_path = manifest_dir / f"{folder_name}.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "orig_name",
                "status",
                "dup_of_orig_name",
                "hash_method",
                "hash_value",
                "hamming",
            ],
        )
        writer.writeheader()
        for it in items:
            writer.writerow(
                {
                    "orig_name": it.orig_name,
                    "status": it.status,
                    "dup_of_orig_name": it.dup_of_orig_name or "",
                    "hash_method": it.hash_method,
                    "hash_value": it.hash_value,
                    "hamming": "" if it.hamming is None else it.hamming,
                }
            )
    return manifest_path


def _compute_anchor_fps(results: list[dict], *, anchor_folder: str, anchor_fps: float) -> None:
    anchor = next((r for r in results if r["folder"] == anchor_folder), None)
    if anchor is None:
        raise ValueError(f"Anchor folder not found among processed folders: {anchor_folder}")
    if anchor["remaining"] <= 0:
        raise ValueError(
            "Anchor folder has 0 remaining frames after dedupe; cannot estimate duration."
        )

    anchor_remaining = int(anchor["remaining"])
    for r in results:
        r["anchor_folder"] = anchor_folder
        r["anchor_fps"] = float(anchor_fps)
        r["anchor_remaining"] = anchor_remaining
        r["estimated_fps_from_anchor"] = round(
            float(anchor_fps) * (r["remaining"] / anchor_remaining), 6
        )


def _default_processed_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"processed_frames2sync_{ts}"


def run_remove_frames2sync() -> None:
    """Tkinter GUI entry point (used by vailá GUI)."""
    if not _TK_AVAILABLE:
        raise RuntimeError("Tkinter is not available on this system.")

    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(title="Select root directory (contains PNG subfolders)")
    if not input_dir:
        return
    input_root = Path(input_dir).resolve()

    # List candidate subfolders with PNGs
    candidates = sorted(
        [
            d.name
            for d in input_root.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and any(p.is_file() and p.suffix.lower() == ".png" for p in d.iterdir())
        ]
    )
    if not candidates:
        messagebox.showerror("remove_frames2sync", "No PNG subfolders found in selected directory.")
        return

    dialog = tk.Toplevel(root)
    dialog.title("remove_frames2sync (PNG)")
    dialog.geometry("520x260")

    anchor_var = tk.StringVar(value=candidates[0])
    fps_var = tk.StringVar(value="50")
    mode_var = tk.StringVar(value="exact")
    thr_var = tk.StringVar(value="2")
    inplace_var = tk.BooleanVar(value=False)
    dryrun_var = tk.BooleanVar(value=True)

    ttk.Label(dialog, text="Anchor folder (reference camera):").pack(
        anchor="w", padx=12, pady=(12, 2)
    )
    anchor_cb = ttk.Combobox(dialog, textvariable=anchor_var, values=candidates, state="readonly")
    anchor_cb.pack(fill="x", padx=12)

    ttk.Label(dialog, text="Anchor FPS:").pack(anchor="w", padx=12, pady=(10, 2))
    ttk.Entry(dialog, textvariable=fps_var).pack(fill="x", padx=12)

    row = ttk.Frame(dialog)
    row.pack(fill="x", padx=12, pady=(10, 0))
    ttk.Label(row, text="Mode:").pack(side="left")
    ttk.Combobox(
        row, textvariable=mode_var, values=["exact", "phash"], state="readonly", width=10
    ).pack(side="left", padx=(8, 16))
    ttk.Label(row, text="pHash threshold:").pack(side="left")
    ttk.Entry(row, textvariable=thr_var, width=6).pack(side="left", padx=(8, 0))

    opts = ttk.Frame(dialog)
    opts.pack(fill="x", padx=12, pady=(10, 0))
    ttk.Checkbutton(opts, text="Dry run (reports only)", variable=dryrun_var).pack(anchor="w")
    ttk.Checkbutton(opts, text="Apply in-place (destructive)", variable=inplace_var).pack(
        anchor="w"
    )

    def _run():
        try:
            anchor_fps = float(fps_var.get())
            phash_thr = int(thr_var.get())
        except Exception:
            messagebox.showerror("remove_frames2sync", "Invalid FPS or threshold.")
            return

        try:
            out_dir = _default_processed_dir(input_root)
            _run_pipeline(
                input_root,
                anchor_folder=anchor_var.get(),
                anchor_fps=anchor_fps,
                dry_run=bool(dryrun_var.get()),
                in_place=bool(inplace_var.get()),
                delete_duplicates=False,
                source_fps=60.0,
                mode=mode_var.get(),
                phash_threshold=phash_thr,
                output_dir=out_dir,
                quiet=False,
                single=False,
            )
            messagebox.showinfo(
                "remove_frames2sync",
                f"Done.\n\nReports saved in:\n{out_dir / 'reports'}",
            )
            dialog.destroy()
            root.destroy()
        except Exception as e:
            messagebox.showerror("remove_frames2sync", str(e))

    ttk.Button(dialog, text="Run", command=_run).pack(pady=12)
    dialog.transient(root)
    dialog.grab_set()
    root.mainloop()


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _collect_folders(root: Path, *, single: bool) -> list[Path]:
    if single:
        return [root]

    folders = sorted(
        [
            d
            for d in root.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")
            and d.name not in {"removed_frames", "kept_frames", "reports", "manifests"}
            and any(p.is_file() and p.suffix.lower() == ".png" for p in d.iterdir())
        ]
    )
    if folders:
        return folders

    if any(p.is_file() and p.suffix.lower() == ".png" for p in root.iterdir()):
        return [root]

    return []


def _run_pipeline(
    root: Path,
    *,
    anchor_folder: str,
    anchor_fps: float,
    dry_run: bool,
    in_place: bool,
    delete_duplicates: bool,
    source_fps: float,
    mode: str,
    phash_threshold: int,
    output_dir: Path,
    quiet: bool,
    single: bool,
) -> tuple[list[dict], Path, Path | None, Path | None]:
    folders = _collect_folders(root, single=single)
    if not folders:
        raise ValueError(f"No PNG-containing folders found in: {root}")

    if not in_place:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir = output_dir / "reports"
        manifest_dir = output_dir / "manifests"
        report_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
    else:
        report_dir = root
        manifest_dir = None

    results: list[dict] = []
    for folder in folders:
        result = process_folder(
            folder,
            dry_run=dry_run,
            delete_duplicates=delete_duplicates,
            source_fps=source_fps,
            out_dir=output_dir if not in_place else None,
            in_place=in_place,
            mode=mode,
            phash_threshold=phash_threshold,
            verbose=not quiet,
        )
        if not result:
            continue

        # Persist per-folder manifest (only in non-destructive mode)
        if manifest_dir is not None:
            _write_manifest(manifest_dir, folder.name, result["manifest"])

        # Drop manifest objects from report rows (keep file paths only)
        result = dict(result)
        result.pop("manifest", None)
        results.append(result)

    if not results:
        raise ValueError("No folders processed.")

    _compute_anchor_fps(results, anchor_folder=anchor_folder, anchor_fps=anchor_fps)
    print_summary(results, anchor_folder=anchor_folder, anchor_fps=anchor_fps)

    txt_path = csv_path = None
    if not in_place:
        txt_path, csv_path = save_reports(report_dir, results, dry_run=dry_run)

        run_config = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "root": str(root),
            "output_dir": str(output_dir),
            "anchor_folder": anchor_folder,
            "anchor_fps": float(anchor_fps),
            "dry_run": bool(dry_run),
            "in_place": bool(in_place),
            "delete_duplicates": bool(delete_duplicates),
            "source_fps_for_ratio": float(source_fps),
            "mode": mode,
            "phash_threshold": int(phash_threshold),
            "folders": [r["folder"] for r in results],
        }
        with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2)

    return results, output_dir, txt_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove duplicate frames to synchronize multi-camera PNG sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "root_dir",
        help="Root directory (parent of sequence folders, or a single sequence folder with --single)",
    )
    parser.add_argument(
        "--single", action="store_true", help="Treat root_dir as a single sequence folder"
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan only — no files modified")
    parser.add_argument(
        "--delete", action="store_true", help="Delete duplicates (default: move to removed_frames/)"
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=60.0,
        metavar="FPS",
        help="FPS of the source video (default: 60)",
    )
    parser.add_argument(
        "--anchor", help="Anchor folder name (inside root_dir) used as reference camera"
    )
    parser.add_argument(
        "--anchor-fps", type=float, metavar="FPS", help="Known FPS of the anchor folder"
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Apply changes in-place (destructive): move/delete duplicates + renumber",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: processed_frames2sync_TIMESTAMP under root_dir)",
    )
    parser.add_argument(
        "--mode",
        choices=["exact", "phash"],
        default="exact",
        help="Duplicate detection: 'exact' (MD5, default) or 'phash' (perceptual hash for lossy)",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=2,
        metavar="N",
        help="Max hamming distance for phash mode (0-64, default: 2)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-folder verbose output")
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.exists():
        print(f"ERROR: Directory not found: {root}")
        sys.exit(1)

    if args.anchor_fps is None or not args.anchor:
        print("ERROR: --anchor and --anchor-fps are required to estimate FPS from an anchor.")
        sys.exit(2)

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else _default_processed_dir(root)
    )

    print("\nremove_frames2sync.py")
    print(f"{'─' * 60}")
    print(f"  Root        : {root}")
    print(f"  Anchor      : {args.anchor} @ {args.anchor_fps} fps")
    print(f"  Output dir  : {output_dir if not args.in_place else '(in-place)'}")
    print(f"  Source FPS  : {args.source_fps} (ratio-based secondary estimate)")
    print(
        f"  Mode        : {'DRY RUN — ' if args.dry_run else ''}{'DELETE' if args.delete else 'MOVE duplicates' if args.in_place else 'COPY duplicates'}"
    )
    print(f"  Hash method : {args.mode}")

    try:
        _run_pipeline(
            root,
            anchor_folder=args.anchor,
            anchor_fps=float(args.anchor_fps),
            dry_run=bool(args.dry_run),
            in_place=bool(args.in_place),
            delete_duplicates=bool(args.delete),
            source_fps=float(args.source_fps),
            mode=args.mode,
            phash_threshold=int(args.phash_threshold),
            output_dir=output_dir,
            quiet=bool(args.quiet),
            single=bool(args.single),
        )
        if not args.in_place:
            print(f"\n  Saved in   : {output_dir}\n")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
