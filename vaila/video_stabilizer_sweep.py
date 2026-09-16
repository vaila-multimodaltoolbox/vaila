"""Two-phase method sweep for fixed-scene video stabilization.

Version: 0.4.3
Update Date: 15 September 2026
Author: Paulo R. P. Santiago
License: AGPL-3.0-or-later

Every candidate is first evaluated from marker data without decoding or
encoding video frames. Successful candidates are ranked by their worst
marker-covered frame region, and only the selected top candidates are rendered.
"""

from __future__ import annotations

import html
import itertools
import math
import os
import shlex
import shutil
import subprocess
import time
import tomllib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__:
    from .video_stabilizer import (
        _check_cancel,
        _method_slug,
        _next_available_output_dir,
        build_parser,
        encode_frames_h264,
        print_gui_cli_mirror,
        run_video_stabilizer,
    )
else:
    from video_stabilizer import (  # ty: ignore[unresolved-import]
        _check_cancel,
        _method_slug,
        _next_available_output_dir,
        build_parser,
        encode_frames_h264,
        print_gui_cli_mirror,
        run_video_stabilizer,
    )


VERSION = "0.4.3"

DEFAULT_GRID = {
    "modes": ["visual", "hybrid", "floor-lock"],
    "models": ["similarity", "affine", "homography"],
    "estimators": ["robust-lsq", "ransac"],
    "smooth": ["none", "savgol", "lowpass"],
    "references": ["auto", "first"],
    "stabilization_markers": ["all", "0-7", "8-13"],
    "anchors": ["none", "8-13@2.0", "8-13@4.0", "0-7@2.0"],
    "hybrid_imputed_weights": [0.25],
    "floor_estimators": ["robust-all", "ransac"],
}

GRID_ALIASES = {
    "mode": "modes",
    "model": "models",
    "estimator": "estimators",
    "smoothing": "smooth",
    "reference": "references",
    "markers": "stabilization_markers",
    "anchor": "anchors",
    "hybrid_imputed_weight": "hybrid_imputed_weights",
    "floor_estimator": "floor_estimators",
}


def _log(message, callback=None):
    print(f">> vaila/video_stabilizer_sweep: {message}", flush=True)
    if callback:
        callback(str(message))


def _read_grid(path):
    grid = {key: list(values) for key, values in DEFAULT_GRID.items()}
    if path is None:
        return grid
    with Path(path).expanduser().open("rb") as stream:
        document = tomllib.load(stream)
    overrides = document.get("sweep", document)
    unknown = set(overrides) - set(DEFAULT_GRID) - set(GRID_ALIASES)
    if unknown:
        raise ValueError(f"Unknown sweep grid axes: {sorted(unknown)}")
    for raw_key, values in overrides.items():
        key = GRID_ALIASES.get(raw_key, raw_key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep grid axis {raw_key!r} must be a non-empty TOML array.")
        grid[key] = values
    return grid


def _parse_anchor(value):
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None, 2.0
    marker_spec, separator, weight_text = text.rpartition("@")
    if not separator or not marker_spec:
        raise ValueError(
            f"Invalid anchor sweep value {value!r}; use none or MARKERS@WEIGHT, e.g. 8-13@4.0."
        )
    weight = float(weight_text)
    if not np.isfinite(weight) or weight <= 0:
        raise ValueError(f"Anchor weight must be positive and finite: {value!r}")
    return marker_spec, weight


def build_sweep_combinations(args, grid):
    """Expand the useful axes and collapse floor-lock to its two estimators."""
    combinations = []
    modes = [str(value) for value in grid["modes"]]
    if args.geometry_config is None:
        modes = [mode for mode in modes if mode == "visual"]
    for mode in modes:
        if mode == "floor-lock":
            for floor_estimator in grid["floor_estimators"]:
                combinations.append(
                    {
                        "mode": mode,
                        "model": "similarity",
                        "estimator": "robust-lsq",
                        "smooth": "savgol",
                        "reference": "auto",
                        "stabilization_markers": "all",
                        "anchor_markers": None,
                        "anchor_weight": 2.0,
                        "hybrid_imputed_weight": 0.25,
                        "floor_estimator": str(floor_estimator),
                    }
                )
            continue
        hybrid_weights = (
            grid["hybrid_imputed_weights"] if mode == "hybrid" else [args.hybrid_imputed_weight]
        )
        for values in itertools.product(
            grid["models"],
            grid["estimators"],
            grid["smooth"],
            grid["references"],
            grid["stabilization_markers"],
            grid["anchors"],
            hybrid_weights,
        ):
            model, estimator, smooth, reference, markers, anchor, imputed_weight = values
            anchor_markers, anchor_weight = _parse_anchor(anchor)
            combinations.append(
                {
                    "mode": mode,
                    "model": str(model),
                    "estimator": str(estimator),
                    "smooth": str(smooth),
                    "reference": str(reference),
                    "stabilization_markers": str(markers),
                    "anchor_markers": anchor_markers,
                    "anchor_weight": anchor_weight,
                    "hybrid_imputed_weight": float(imputed_weight),
                    "floor_estimator": str(grid["floor_estimators"][0]),
                }
            )
    unique = []
    seen = set()
    for options in combinations:
        key = tuple(sorted(options.items()))
        if key not in seen:
            unique.append(options)
            seen.add(key)
    return unique


def _command_argv(args, options):
    argv = [
        "--video",
        str(args.video),
        "--markers",
        str(args.markers),
        "--mode",
        options["mode"],
        "--model",
        options["model"],
        "--estimator",
        options["estimator"],
        "--smooth",
        options["smooth"],
        "--reference",
        options["reference"],
        "--stabilization-markers",
        options["stabilization_markers"],
        "--anchor-weight",
        f"{options['anchor_weight']:g}",
        "--hybrid-imputed-weight",
        f"{options['hybrid_imputed_weight']:g}",
        "--floor-estimator",
        options["floor_estimator"],
        "--canvas",
        args.canvas,
        "--border",
        args.border,
    ]
    if options["anchor_markers"]:
        argv.extend(["--anchor-markers", options["anchor_markers"]])
    if args.geometry_config:
        argv.extend(["--geometry-config", str(args.geometry_config)])
    if args.metric_markers:
        argv.extend(["--metric-markers", args.metric_markers])
    if args.debug_overlay:
        argv.append("--debug-overlay")
    if args.no_audio:
        argv.append("--no-audio")
    build_parser().parse_args(argv)
    return ["uv", "run", "python", "-m", "vaila.video_stabilizer", *argv]


def _command_text(argv):
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _base_row(index, slug, args, options):
    return {
        "rank": "",
        "combination": index,
        "status": "error",
        "method_slug": slug,
        **options,
        "anchor_markers": options["anchor_markers"] or "none",
        "canvas": args.canvas,
        "rms_after_region_top_px": float("nan"),
        "rms_after_region_middle_px": float("nan"),
        "rms_after_region_bottom_px": float("nan"),
        "rms_after_region_mean_px": float("nan"),
        "worst_region": "unavailable",
        "rms_after_worst_region_px": float("nan"),
        "direct_percent": float("nan"),
        "interpolated_percent": float("nan"),
        "max_scale_deviation": float("nan"),
        "error": "",
        "command": _command_text(_command_argv(args, options)),
        "triage_report": "",
        "rendered_video": "",
        "render_report": "",
    }


def _run_options(args, options, output_dir, *, render_video, progress_callback, cancel_event):
    return run_video_stabilizer(
        args.video,
        args.markers,
        output_dir,
        geometry_config=args.geometry_config,
        metric_markers=args.metric_markers,
        canvas=args.canvas,
        border=args.border,
        preserve_audio=not args.no_audio,
        debug_overlay=args.debug_overlay if render_video else False,
        render_video=render_video,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        **options,
    )


def _rank_rows(rows):
    successful = [row for row in rows if row["status"] == "ok"]

    def key(row):
        return tuple(
            value if np.isfinite(value) else math.inf
            for value in (
                _finite_float(row["rms_after_worst_region_px"]),
                _finite_float(row["rms_after_region_mean_px"]),
                _finite_float(row["max_scale_deviation"]),
            )
        )

    successful.sort(key=key)
    for rank, row in enumerate(successful, 1):
        row["rank"] = rank
    failed = [row for row in rows if row["status"] != "ok"]
    return successful + failed


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _letterbox(frame, width, height, label):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    cell = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    cell[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(cell, (0, 0), (width, 42), (0, 0, 0), -1)
    text = label if len(label) <= 70 else label[:67] + "..."
    cv2.putText(cell, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return cell


def _write_montage(source, rendered_rows, root, fps, callback=None):
    paths = [Path(source)] + [root / row["rendered_video"] for row in rendered_rows]
    labels = ["Original"] + [f"#{row['rank']} {row['method_slug']}" for row in rendered_rows]
    columns = max(1, math.ceil(math.sqrt(len(paths))))
    rows = math.ceil(len(paths) / columns)
    probes = [cv2.VideoCapture(str(path)) for path in paths]
    try:
        if not all(probe.isOpened() for probe in probes):
            raise RuntimeError("Could not inspect every video selected for the sweep montage.")
        natural_w = max(int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) for probe in probes)
        natural_h = max(int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) for probe in probes)
    finally:
        for probe in probes:
            probe.release()
    cell_w = max(2, min(natural_w, 2560 // columns) // 2 * 2)
    cell_h = max(2, min(natural_h, 1440 // rows) // 2 * 2)
    size = (cell_w * columns, cell_h * rows)

    def frames():
        captures = [cv2.VideoCapture(str(path)) for path in paths]
        try:
            if not all(capture.isOpened() for capture in captures):
                raise RuntimeError("Could not open every video selected for the sweep montage.")
            while True:
                decoded = [capture.read() for capture in captures]
                if not decoded[0][0]:
                    break
                if not all(ok for ok, _frame in decoded):
                    raise RuntimeError(
                        "A rendered candidate ended before the original montage video."
                    )
                cells = [
                    _letterbox(frame, cell_w, cell_h, label)
                    for (_ok, frame), label in zip(decoded, labels, strict=True)
                ]
                while len(cells) < rows * columns:
                    cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
                yield np.vstack(
                    [
                        np.hstack(cells[start : start + columns])
                        for start in range(0, len(cells), columns)
                    ]
                )
        finally:
            for capture in captures:
                capture.release()

    output = root / "sweep_montage.mp4"
    encode_frames_h264(output, size, fps, frames, callback)
    return output


def _write_ranking(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_report(path, rows, best_command, montage_path=None):
    columns = (
        "rank",
        "status",
        "method_slug",
        "mode",
        "model",
        "estimator",
        "smooth",
        "reference",
        "stabilization_markers",
        "anchor_markers",
        "rms_after_region_top_px",
        "rms_after_region_middle_px",
        "rms_after_region_bottom_px",
        "worst_region",
        "rms_after_worst_region_px",
        "max_scale_deviation",
        "error",
        "artifacts",
    )
    body = []
    for row in rows:
        cells = []
        for column in columns:
            if column == "artifacts":
                links = []
                if row["triage_report"]:
                    links.append(
                        f'<a href="{html.escape(row["triage_report"], quote=True)}">triage</a>'
                    )
                if row["rendered_video"]:
                    links.append(
                        f'<a href="{html.escape(row["rendered_video"], quote=True)}">video</a>'
                    )
                    links.append(
                        f'<a href="{html.escape(row["render_report"], quote=True)}">render report</a>'
                    )
                value = " · ".join(links)
            else:
                value = html.escape(str(row.get(column, "")))
            cells.append(f"<td>{value}</td>")
        class_name = ' class="best"' if row.get("rank") == 1 else ""
        body.append(f"<tr{class_name}>{''.join(cells)}</tr>")
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    montage_link = (
        f'<p><a href="{html.escape(str(montage_path), quote=True)}">Comparison montage</a></p>'
        if montage_path
        else ""
    )
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>vailá stabilizer method sweep</title><style>"
        "body{font:15px sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px;vertical-align:top}"
        "th{position:sticky;top:0;background:#eee}.best{background:#dcfce7}"
        "code{white-space:pre-wrap}</style><h1>Video stabilizer method sweep</h1>"
        "<p>Candidates are ranked by the largest marker residual among the top, middle and bottom thirds; ties use their mean and then scale deviation. A candidate missing any region ranks after fully covered candidates.</p>"
        f"<h2>Best reproducible command</h2><code>{html.escape(best_command)}</code>"
        f"{montage_link}"
        f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></html>",
        encoding="utf-8",
    )


def _render_count(value, available):
    if str(value).lower() == "all":
        return available
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("--sweep-render-top must be a positive integer or all.") from exc
    if count <= 0:
        raise ValueError("--sweep-render-top must be a positive integer or all.")
    return min(count, available)


def run_stabilizer_sweep(args, progress_callback=None, cancel_event=None):
    """Triage the configured grid, render its winners, and write comparison artifacts."""
    grid = _read_grid(args.sweep_grid)
    combinations = build_sweep_combinations(args, grid)
    if not combinations:
        raise ValueError("The sweep grid has no applicable modes for the supplied inputs.")
    root = (
        Path(_next_available_output_dir(args.output_dir, "sweep"))
        if args.output_dir
        else Path(args.video).expanduser().resolve().parent
        / f"vaila_stabilized_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    ).resolve()
    triage_dir, renders_dir, videos_dir = root / "triage", root / "renders", root / "videos"
    for directory in (triage_dir, renders_dir, videos_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _log(
        f"Triage: {len(combinations)} combinations; video encoding is disabled.", progress_callback
    )
    rows = []
    started = time.monotonic()
    for index, options in enumerate(combinations, 1):
        _check_cancel(cancel_event)
        slug = _method_slug(**options)
        row = _base_row(index, slug, args, options)
        _log(f"Triage {index}/{len(combinations)}: {slug}", progress_callback)
        try:
            outputs = _run_options(
                args,
                options,
                triage_dir / slug,
                render_video=False,
                progress_callback=None,
                cancel_event=cancel_event,
            )
            summary = pd.read_json(outputs["summary"], typ="series")
            regions = [
                _finite_float(summary.get(f"rms_after_region_{name}_px"))
                for name in ("top", "middle", "bottom")
            ]
            row.update(
                status="ok",
                triage_report=outputs["report"].relative_to(root).as_posix(),
                rms_after_region_top_px=regions[0],
                rms_after_region_middle_px=regions[1],
                rms_after_region_bottom_px=regions[2],
                rms_after_region_mean_px=(
                    float(np.mean(regions)) if np.isfinite(regions).all() else float("nan")
                ),
                worst_region=str(summary.get("worst_region", "unavailable")),
                rms_after_worst_region_px=_finite_float(summary.get("rms_after_worst_region_px")),
                direct_percent=_finite_float(summary.get("direct_percent")),
                interpolated_percent=_finite_float(summary.get("interpolated_percent")),
                max_scale_deviation=_finite_float(summary.get("max_scale_deviation")),
            )
        except InterruptedError:
            raise
        except Exception as exc:
            row["error"] = str(exc)
            _log(f"Combination failed: {exc}", progress_callback)
        rows.append(row)
        if index == min(10, len(combinations)):
            elapsed = time.monotonic() - started
            seconds_each = elapsed / index
            remaining = seconds_each * (len(combinations) - index)
            _log(
                f"Measured triage: {seconds_each:.3f} s/combination; estimated {remaining / 60:.1f} min remaining.",
                progress_callback,
            )
    ranked = _rank_rows(rows)
    successful = [row for row in ranked if row["status"] == "ok"]
    render_count = _render_count(args.sweep_render_top, len(successful))
    selected = successful[:render_count]
    rank_width = max(2, len(str(max(1, len(successful)))))
    for position, row in enumerate(selected, 1):
        _check_cancel(cancel_event)
        _log(
            f"Render {position}/{render_count}: #{row['rank']} {row['method_slug']}",
            progress_callback,
        )
        options = {
            key: row[key]
            for key in (
                "mode",
                "model",
                "estimator",
                "smooth",
                "reference",
                "stabilization_markers",
                "anchor_weight",
                "hybrid_imputed_weight",
                "floor_estimator",
            )
        }
        options["anchor_markers"] = (
            None if row["anchor_markers"] == "none" else row["anchor_markers"]
        )
        try:
            outputs = _run_options(
                args,
                options,
                renders_dir / row["method_slug"],
                render_video=True,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
            )
            flat = videos_dir / (
                f"{int(row['rank']):0{rank_width}d}_{row['method_slug']}{outputs['video'].suffix}"
            )
            _link_or_copy(outputs["video"], flat)
            row["rendered_video"] = flat.relative_to(root).as_posix()
            row["render_report"] = outputs["report"].relative_to(root).as_posix()
        except InterruptedError:
            raise
        except Exception as exc:
            row["error"] = f"Render failed: {exc}"
            _log(row["error"], progress_callback)
    ranking_path = root / "sweep_ranking.csv"
    _write_ranking(ranking_path, ranked)
    best_command = successful[0]["command"] if successful else "No successful combination."
    best_path = root / "sweep_best_command.txt"
    best_path.write_text(best_command + "\n", encoding="utf-8")
    outputs = {
        "output_dir": root,
        "ranking": ranking_path,
        "best_command": best_path,
        "videos": videos_dir,
    }
    rendered = [row for row in selected if row["rendered_video"]]
    montage_path = None
    if rendered:
        first_summary = pd.read_json(
            renders_dir / rendered[0]["method_slug"] / "stabilization_summary.json",
            typ="series",
        )
        montage_path = _write_montage(
            args.video, rendered, root, float(first_summary["fps"]), progress_callback
        )
        outputs["montage"] = montage_path
    report_path = root / "sweep_report.html"
    _write_report(
        report_path,
        ranked,
        best_command,
        montage_path.relative_to(root).as_posix() if montage_path else None,
    )
    outputs["report"] = report_path
    print_gui_cli_mirror(
        "vaila/video_stabilizer_sweep",
        best_command,
        note="Best ranked CLI (copy/paste to render again):",
    )
    _log(f"Complete. Ranked report: {report_path}", progress_callback)
    return outputs


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.video or not args.markers:
        parser.error("--video and --markers are required for a method sweep")
    args.sweep = True
    try:
        run_stabilizer_sweep(args)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        _log(str(exc) or "Cancelled")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
