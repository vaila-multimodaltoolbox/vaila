"""QA Roboflow / YOLOv5-style pitch datasets (train / valid / test).

Update Date: 19 May 2026
Version: 0.3.44

Unlike :mod:`vaila.fifa_dataset_builder`, this module does **not** download or
merge sources. It inspects an existing tree such as::

    <root>/
        data.yaml
        train/{images,labels}/
        valid/{images,labels}/   # Roboflow name; vailá unified uses ``val``
        test/{images,labels}/

When labels match the vailá **32 soccer-pitch keypoints** schema, it can render
overlays under ``<root>/images_with_labels/{train,valid,test}/`` for manual
review (same drawing style as ``fifa_dataset_builder.export_label_check_bundle``).

CLI::

    uv run python -m vaila.fifa_yolo_dataset_qa \\
        --dataset /path/to/football-field-detection_v1i_yolov5pytorch

GUI (``vaila.py`` → Soccer Tools → YOLO Dataset QA, or no CLI args)::

    uv run python -m vaila.fifa_yolo_dataset_qa
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .fifa_dataset_builder import (
        CANONICAL_FLIP_IDX_32,
        NUM_KEYPOINTS,
        draw_yolo_pose_overlay,
    )
    from .fifa_manual_merge import validate_label_text
except ImportError:
    from fifa_dataset_builder import (  # ty: ignore[unresolved-import]
        CANONICAL_FLIP_IDX_32,
        NUM_KEYPOINTS,
        draw_yolo_pose_overlay,
    )
    from fifa_manual_merge import validate_label_text  # ty: ignore[unresolved-import]

EXPECTED_LINE_FIELDS = 1 + 4 + NUM_KEYPOINTS * 3
_IMAGE_EXTS_ORDERED = (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP")


@dataclass(frozen=True)
class SplitPaths:
    """One split's ``images/`` and ``labels/`` directories."""

    roboflow_name: str  # folder under dataset root, e.g. ``valid``
    vaila_name: str  # unified name, e.g. ``val``
    images_dir: Path
    labels_dir: Path


@dataclass
class RoboflowLayout:
    """Detected Roboflow YOLO-Pose export under ``root``."""

    root: Path
    data_yaml: Path | None
    splits: tuple[SplitPaths, ...] = field(default_factory=tuple)


@dataclass
class SampleIssue:
    split: str
    stem: str
    reason: str


@dataclass
class QaReport:
    layout: RoboflowLayout
    issues: list[SampleIssue] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    data_yaml_issues: list[str] = field(default_factory=list)
    schema_ok: bool = False

    def summary_lines(self) -> list[str]:
        lines = [
            f"Dataset: {self.layout.root}",
            f"Splits: {', '.join(s.roboflow_name for s in self.layout.splits)}",
        ]
        for key in ("labels", "images", "pairs_ok", "pairs_bad"):
            if key in self.counts:
                lines.append(f"  {key}: {self.counts[key]}")
        if self.data_yaml_issues:
            lines.append("data.yaml:")
            lines.extend(f"  - {m}" for m in self.data_yaml_issues)
        lines.append(f"schema_ok: {self.schema_ok}")
        if self.issues:
            lines.append(f"sample_issues: {len(self.issues)}")
        return lines


def _find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in _IMAGE_EXTS_ORDERED:
        cand = images_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    return None


def detect_roboflow_layout(root: Path) -> RoboflowLayout | None:
    """Return layout when ``root`` has ``{train,valid|val,test}/{images,labels}``."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        return None

    data_yaml = root / "data.yaml"
    yaml_path = data_yaml if data_yaml.is_file() else None

    splits: list[SplitPaths] = []
    for roboflow_name, vaila_name in (
        ("train", "train"),
        ("valid", "val"),
        ("val", "val"),
        ("test", "test"),
    ):
        images_dir = root / roboflow_name / "images"
        labels_dir = root / roboflow_name / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        if any(s.roboflow_name == roboflow_name for s in splits):
            continue
        splits.append(
            SplitPaths(
                roboflow_name=roboflow_name,
                vaila_name=vaila_name,
                images_dir=images_dir,
                labels_dir=labels_dir,
            )
        )

    if not splits:
        return None
    return RoboflowLayout(root=root, data_yaml=yaml_path, splits=tuple(splits))


def _parse_flip_idx_from_yaml(text: str) -> list[int] | None:
    m = re.search(r"flip_idx:\s*\[([^\]]+)\]", text)
    if not m:
        return None
    try:
        return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
    except ValueError:
        return None


def _check_data_yaml(layout: RoboflowLayout) -> list[str]:
    issues: list[str] = []
    if layout.data_yaml is None:
        issues.append("missing data.yaml at dataset root")
        return issues

    text = layout.data_yaml.read_text(encoding="utf-8")
    if f"kpt_shape: [{NUM_KEYPOINTS}, 3]" not in text:
        if "kpt_shape:" in text:
            issues.append(f"kpt_shape should be [{NUM_KEYPOINTS}, 3] for vailá pitch schema")
        else:
            issues.append("data.yaml missing kpt_shape")

    flip = _parse_flip_idx_from_yaml(text)
    if flip is None:
        issues.append("data.yaml missing flip_idx")
    elif tuple(flip) != CANONICAL_FLIP_IDX_32:
        issues.append(
            "flip_idx differs from vailá canonical Roboflow 32-pt schema "
            f"(expected {len(CANONICAL_FLIP_IDX_32)} indices)"
        )

    if "nc:" not in text:
        issues.append("data.yaml missing nc")
    if "names:" not in text:
        issues.append("data.yaml missing names")

    required_in_yaml = {"train"}
    for split in layout.splits:
        if split.roboflow_name in ("valid", "val"):
            required_in_yaml.add("val")
            required_in_yaml.add("valid")
    for key in required_in_yaml:
        if key not in text:
            issues.append(f"data.yaml missing entry for split {key!r}")

    return issues


def audit_roboflow_dataset(root: Path) -> QaReport:
    """Validate labels and ``data.yaml``; set ``schema_ok`` when ready for overlays."""
    layout = detect_roboflow_layout(root)
    if layout is None:
        raise FileNotFoundError(
            f"Not a Roboflow YOLO tree (expected train|valid|test with images/ + labels/): {root}"
        )

    report = QaReport(layout=layout)
    report.data_yaml_issues = _check_data_yaml(layout)
    counts: dict[str, int] = {
        "labels": 0,
        "images": 0,
        "pairs_ok": 0,
        "pairs_bad": 0,
        "orphan_images": 0,
    }

    label_stems: set[tuple[str, str]] = set()
    for sp in layout.splits:
        for lbl_path in sorted(sp.labels_dir.glob("*.txt")):
            counts["labels"] += 1
            stem = lbl_path.stem
            label_stems.add((sp.roboflow_name, stem))
            img_path = _find_image_for_stem(sp.images_dir, stem)
            if img_path is None:
                counts["pairs_bad"] += 1
                report.issues.append(
                    SampleIssue(sp.roboflow_name, stem, "label_without_matching_image")
                )
                continue
            try:
                text = lbl_path.read_text(encoding="utf-8")
            except OSError as exc:
                counts["pairs_bad"] += 1
                report.issues.append(
                    SampleIssue(sp.roboflow_name, stem, f"read_error_{exc.__class__.__name__}")
                )
                continue
            ok, reason = validate_label_text(text)
            if not ok:
                counts["pairs_bad"] += 1
                report.issues.append(SampleIssue(sp.roboflow_name, stem, reason))
                continue
            counts["pairs_ok"] += 1

        for ext in _IMAGE_EXTS_ORDERED:
            for img_path in sp.images_dir.glob(f"*{ext}"):
                counts["images"] += 1
                if (sp.roboflow_name, img_path.stem) not in label_stems:
                    counts["orphan_images"] += 1
                    report.issues.append(
                        SampleIssue(sp.roboflow_name, img_path.stem, "image_without_label")
                    )

    report.counts = counts
    report.schema_ok = (
        not report.data_yaml_issues and counts["pairs_bad"] == 0 and counts["pairs_ok"] > 0
    )
    return report


def export_images_with_labels(
    root: Path,
    *,
    clean: bool = False,
    max_per_split: int | None = None,
    jpeg_quality: int = 92,
) -> dict[str, int]:
    """Draw bbox + 32 keypoints into ``<root>/images_with_labels/<split>/``.

    Requires :func:`audit_roboflow_dataset` to pass (``schema_ok``). Returns
    counters: ``written``, ``draw_fail``, ``skipped``.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("export_images_with_labels requires opencv-python (cv2).") from exc

    report = audit_roboflow_dataset(root)
    if not report.schema_ok:
        detail = report.data_yaml_issues[:5] + [i.reason for i in report.issues[:10]]
        raise ValueError(
            "Dataset failed QA; fix issues before exporting overlays. " + "; ".join(detail[:12])
        )

    layout = report.layout
    out_root = layout.root / "images_with_labels"
    if clean and out_root.exists():
        shutil.rmtree(out_root)

    stats = {"written": 0, "draw_fail": 0, "skipped": 0}
    for sp in layout.splits:
        out_split = out_root / sp.roboflow_name
        out_split.mkdir(parents=True, exist_ok=True)
        lbl_files = sorted(sp.labels_dir.glob("*.txt"))
        if max_per_split is not None:
            lbl_files = lbl_files[: max(0, max_per_split)]

        for lbl_path in lbl_files:
            stem = lbl_path.stem
            img_path = _find_image_for_stem(sp.images_dir, stem)
            if img_path is None:
                stats["skipped"] += 1
                continue
            line = lbl_path.read_text(encoding="utf-8").strip().splitlines()
            if not line:
                stats["skipped"] += 1
                continue
            parts = line[0].split()
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                stats["draw_fail"] += 1
                continue
            cap = f"{stem} [{sp.roboflow_name}→{sp.vaila_name}]"
            if not draw_yolo_pose_overlay(bgr, parts, caption=cap):
                stats["draw_fail"] += 1
                continue
            out_name = f"{stem}.jpg"
            cv2.imwrite(
                str(out_split / out_name),
                bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            stats["written"] += 1

    return stats


def write_qa_report_csv(report: QaReport, csv_path: Path) -> None:
    """Write per-sample issues to ``csv_path``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "stem", "reason"])
        for issue in report.issues:
            writer.writerow([issue.split, issue.stem, issue.reason])


def emit_vaila_data_yaml(layout: RoboflowLayout, out_path: Path | None = None) -> Path:
    """Write vailá-style ``data.yaml`` (``val`` split name, absolute ``path``)."""
    out_path = (out_path or layout.root / "data_vaila.yaml").resolve()
    flip_s = ", ".join(str(i) for i in CANONICAL_FLIP_IDX_32)
    lines = [
        f"path: {layout.root}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"kpt_shape: [{NUM_KEYPOINTS}, 3]",
        f"flip_idx: [{flip_s}]",
        "",
        "nc: 1",
        "names: ['pitch']",
        "",
        "# Generated by vaila.fifa_yolo_dataset_qa",
    ]
    # If this export uses ``val/`` instead of ``valid/``, point val: there
    if (layout.root / "val").is_dir() and not (layout.root / "valid").is_dir():
        lines[2] = "val: val/images"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QA Roboflow YOLO pitch datasets and export keypoint overlays.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        help="Dataset root (train/valid/test with images/ + labels/). GUI if omitted.",
    )
    p.add_argument(
        "--export-overlays",
        action="store_true",
        help="Write images_with_labels/{train,valid,test}/ when schema_ok.",
    )
    p.add_argument(
        "--clean-overlays",
        action="store_true",
        help="Remove existing images_with_labels/ before export.",
    )
    p.add_argument(
        "--max-per-split",
        type=int,
        default=0,
        help="Limit overlay count per split (0 = all).",
    )
    p.add_argument(
        "--issues-csv",
        type=Path,
        help="Write sample issues to this CSV path.",
    )
    p.add_argument(
        "--emit-vaila-yaml",
        action="store_true",
        help="Write data_vaila.yaml (vailá split names + canonical flip_idx).",
    )
    return p


def run_qa_job(
    root: Path,
    *,
    export_overlays: bool = False,
    clean_overlays: bool = False,
    emit_vaila_yaml: bool = False,
    max_per_split: int | None = None,
    issues_csv: Path | None = None,
) -> tuple[int, QaReport, dict[str, int] | None]:
    """Run audit and optional export. Returns ``(exit_code, report, overlay_stats)``."""
    report = audit_roboflow_dataset(root)
    overlay_stats: dict[str, int] | None = None

    if issues_csv is not None:
        write_qa_report_csv(report, issues_csv)

    if emit_vaila_yaml and report.schema_ok:
        yaml_out = emit_vaila_data_yaml(report.layout)
        print(f"vaila data.yaml -> {yaml_out}")

    if export_overlays:
        if not report.schema_ok:
            return 1, report, None
        overlay_stats = export_images_with_labels(
            root,
            clean=clean_overlays,
            max_per_split=max_per_split,
        )

    code = 0 if report.schema_ok else 1
    return code, report, overlay_stats


def run_fifa_yolo_dataset_qa() -> int:
    """Tkinter GUI entry point (used from ``vaila.py`` Soccer Tools)."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError:
        print("Tkinter not available; pass --dataset on the CLI.")
        return 1

    win = tk.Tk()
    win.title("YOLO Pitch Dataset QA")
    win.resizable(False, False)

    dataset_var = tk.StringVar()
    export_var = tk.BooleanVar(value=True)
    clean_var = tk.BooleanVar(value=True)
    yaml_var = tk.BooleanVar(value=True)
    max_var = tk.StringVar(value="0")

    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(
        frm,
        text="Validate Roboflow YOLO-Pose exports (32 pitch keypoints)",
        font=("default", 11, "bold"),
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

    ttk.Label(frm, text="Dataset root:").grid(row=1, column=0, sticky="w")
    ent = ttk.Entry(frm, textvariable=dataset_var, width=52)
    ent.grid(row=1, column=1, padx=4, sticky="we")

    def browse() -> None:
        picked = filedialog.askdirectory(
            title="Select Roboflow YOLO pitch dataset root (train/valid/test)"
        )
        if picked:
            dataset_var.set(picked)

    ttk.Button(frm, text="Browse…", command=browse).grid(row=1, column=2, sticky="e")

    opts = ttk.LabelFrame(frm, text="Actions", padding=8)
    opts.grid(row=2, column=0, columnspan=3, sticky="we", pady=10)
    ttk.Checkbutton(
        opts,
        text="Export keypoint overlays to images_with_labels/",
        variable=export_var,
    ).pack(anchor="w")
    ttk.Checkbutton(
        opts,
        text="Remove existing images_with_labels/ before export",
        variable=clean_var,
    ).pack(anchor="w", padx=(18, 0))
    ttk.Checkbutton(
        opts,
        text="Write data_vaila.yaml (vaila split names)",
        variable=yaml_var,
    ).pack(anchor="w")

    lim = ttk.Frame(opts)
    lim.pack(anchor="w", pady=(6, 0))
    ttk.Label(lim, text="Max overlays per split (0 = all):").pack(side="left")
    ttk.Entry(lim, textvariable=max_var, width=8).pack(side="left", padx=6)

    ttk.Label(
        frm,
        text="Expected layout: train|valid|test each with images/ and labels/.\n"
        "Keypoints are drawn in orange on the preview JPEGs.",
        justify="left",
    ).grid(row=3, column=0, columnspan=3, sticky="w")

    btn_row = ttk.Frame(frm)
    btn_row.grid(row=4, column=0, columnspan=3, pady=(12, 0), sticky="e")

    def on_run() -> None:
        raw = dataset_var.get().strip()
        if not raw:
            messagebox.showwarning("YOLO Pitch Dataset QA", "Select a dataset root folder.")
            return
        try:
            max_n = int(max_var.get().strip() or "0")
        except ValueError:
            messagebox.showwarning("YOLO Pitch Dataset QA", "Max per split must be an integer.")
            return
        if max_n < 0:
            messagebox.showwarning("YOLO Pitch Dataset QA", "Max per split must be >= 0.")
            return

        dataset_path = Path(raw).expanduser().resolve()
        run_btn.state(["disabled"])
        win.update_idletasks()
        try:
            code, report, stats = run_qa_job(
                dataset_path,
                export_overlays=bool(export_var.get()),
                clean_overlays=bool(clean_var.get()),
                emit_vaila_yaml=bool(yaml_var.get()),
                max_per_split=max_n if max_n > 0 else None,
            )
        except FileNotFoundError as exc:
            messagebox.showerror("YOLO Pitch Dataset QA", str(exc))
            run_btn.state(["!disabled"])
            return
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("YOLO Pitch Dataset QA", str(exc))
            run_btn.state(["!disabled"])
            return

        summary = "\n".join(report.summary_lines())
        if stats is not None:
            summary += (
                f"\n\nOverlays: {stats['written']} written\n"
                f"  -> {dataset_path / 'images_with_labels'}/"
            )
        if report.issues:
            summary += "\n\nFirst issues:\n"
            for issue in report.issues[:8]:
                summary += f"  [{issue.split}] {issue.stem}: {issue.reason}\n"

        if code == 0:
            messagebox.showinfo("YOLO Pitch Dataset QA", summary)
            win.destroy()
        else:
            messagebox.showwarning("YOLO Pitch Dataset QA", summary)
            run_btn.state(["!disabled"])

    def on_cancel() -> None:
        win.destroy()

    run_btn = ttk.Button(btn_row, text="Run QA", command=on_run)
    run_btn.pack(side="right", padx=4)
    ttk.Button(btn_row, text="Cancel", command=on_cancel).pack(side="right")

    frm.columnconfigure(1, weight=1)
    win.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.dataset is None:
        return run_fifa_yolo_dataset_qa()

    root = args.dataset.expanduser().resolve()
    max_n = args.max_per_split if args.max_per_split > 0 else None
    try:
        code, report, stats = run_qa_job(
            root,
            export_overlays=bool(args.export_overlays),
            clean_overlays=bool(args.clean_overlays),
            emit_vaila_yaml=bool(args.emit_vaila_yaml),
            max_per_split=max_n,
            issues_csv=args.issues_csv.expanduser().resolve() if args.issues_csv else None,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    for line in report.summary_lines():
        print(line)
    if report.issues:
        print(f"First issues ({min(15, len(report.issues))}):")
        for issue in report.issues[:15]:
            print(f"  [{issue.split}] {issue.stem}: {issue.reason}")

    if args.issues_csv is not None:
        print(f"Issues CSV -> {args.issues_csv}")

    if stats is not None:
        print(
            f"Overlays -> {root / 'images_with_labels'} "
            f"(written={stats['written']}, draw_fail={stats['draw_fail']}, "
            f"skipped={stats['skipped']})"
        )

    return code


if __name__ == "__main__":
    raise SystemExit(main())
