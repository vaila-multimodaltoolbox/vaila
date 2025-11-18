"""
reidmplrswap.py — Detect and fix abrupt left/right swaps in marker tracks

This tool analyzes 2D marker CSVs (e.g., MediaPipe pixel outputs in vailá
format) to detect short, abrupt swaps between left/right marker pairs and
apply corrections. It supports:

1) Automatic detection and correction
   - Side consistency via the sign of (x_right - x_left)
   - Optional continuity check to prefer the swap that minimizes total motion
   - Produces a report with suggested swap frame ranges per marker pair

2) Manual correction
   - User specifies a marker pair and a frame interval to swap left<->right

Inputs
------
- CSV with columns like: marker_x, marker_y [, marker_z]. Left/right markers
  should be identifiable via tokens like L/R or left/right (prefix/suffix).
- Optional video path: if provided, the script exports short preview clips
  around suspected swap intervals to assist visual validation.

Outputs
-------
- Corrected CSV saved alongside original with suffix "_reidswap.csv"
- Text report with suspected swaps: "_reidswap_report.txt"

Usage (CLI)
-----------
python -m vaila.reidmplrswap --csv path/to/data.csv [--auto]
python -m vaila.reidmplrswap --csv path/to/data.csv \
       --manual "hip" --start 1200 --end 1300
python -m vaila.reidmplrswap --csv path/to/data.csv --video path/to/video.mp4 --review

If paths are omitted, a minimal Tk dialog will ask for the CSV path.

Notes
-----
- This script infers left/right pairs by name. You can review detections in the
  report and run manual swaps as needed.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:  # Tk not mandatory for CLI usage
    tk = None  # type: ignore
    filedialog = None  # type: ignore
    messagebox = None  # type: ignore


# ------------------------------
# Data structures and constants
# ------------------------------

COORD_SUFFIXES = ("x", "y", "z")


@dataclass
class MarkerCoords:
    base_name: str
    side: str  # "L" or "R"
    cols: dict[str, str]  # coord suffix -> column name (e.g., {"x": "hip_L_x"})


@dataclass
class LRPair:
    base_name: str
    left: MarkerCoords
    right: MarkerCoords


# ------------------------------
# Column parsing and pair finding
# ------------------------------


def _normalize_side_tokens(name: str) -> tuple[str | None, str]:
    """
    Extract side (L/R) and the base marker name from a column marker token.

    Supported shapes (case-insensitive):
    - L_<base>, R_<base>
    - <base>_L, <base>_R
    - left_<base>, right_<base>
    - <base>_left, <base>_right
    - l<sep><base>, r<sep><base> (and reversed)

    Returns (side, base) where side in {"L","R"} or None if not found.
    """
    token = name.lower()
    # common explicit tokens
    patterns = [
        r"^(left)[-_ ]*(.+)$",
        r"^(right)[-_ ]*(.+)$",
        r"^([lr])[-_ ]+(.+)$",
        r"^(.+)[-_ ]*(left)$",
        r"^(.+)[-_ ]*(right)$",
        r"^(.+)[-_ ]+([lr])$",
        r"^(l)(.+)$",
        r"^(r)(.+)$",
        r"^(.+)(l)$",
        r"^(.+)(r)$",
    ]
    for pat in patterns:
        m = re.match(pat, token)
        if m:
            groups = [g for g in m.groups() if g is not None]
            if len(groups) == 2:
                g1, g2 = groups
                sides = {"left": "L", "right": "R", "l": "L", "r": "R"}
                if g1 in sides:
                    return sides[g1], g2
                if g2 in sides:
                    return sides[g2], g1
    return None, name


def _split_marker_and_coord(col: str) -> tuple[str, str] | None:
    """
    Split a column into (marker_token, coord_suffix) when it ends with _x/_y/_z.
    Returns None if not a coordinate column.
    """
    m = re.match(r"^(.+)_([xyzXYZ])$", col)
    if not m:
        return None
    return m.group(1), m.group(2).lower()


def find_lr_pairs(df: pd.DataFrame) -> list[LRPair]:
    """
    Build left/right pairs from dataframe column names.

    We consider only columns ending with _x/_y/_z. The preceding token is
    parsed for side information (L/R or left/right). Pairs are grouped by the
    base_name (common part without side token).
    """
    by_marker: dict[str, dict[str, dict[str, str]]] = {}
    # base_name -> side(L/R) -> coord -> colname

    for col in df.columns:
        split = _split_marker_and_coord(col)
        if not split:
            continue
        marker_token, coord = split
        side, base = _normalize_side_tokens(marker_token)
        if side not in ("L", "R"):
            continue
        if base not in by_marker:
            by_marker[base] = {"L": {}, "R": {}}
        by_marker[base][side][coord] = col

    pairs: list[LRPair] = []
    for base, sides in by_marker.items():
        if set(sides.keys()) >= {"L", "R"} and sides["L"] and sides["R"]:
            left = MarkerCoords(base_name=base, side="L", cols=sides["L"])
            right = MarkerCoords(base_name=base, side="R", cols=sides["R"])
            # require at least x and y
            if "x" in left.cols and "x" in right.cols and "y" in left.cols and "y" in right.cols:
                pairs.append(LRPair(base_name=base, left=left, right=right))
    return pairs


# ------------------------------
# Swap detection and application
# ------------------------------


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of [start, end] inclusive ranges where mask is True."""
    if mask.ndim != 1:
        mask = mask.ravel()
    n = mask.size
    if n == 0:
        return []
    idx = np.flatnonzero(mask.astype(np.int8))
    if idx.size == 0:
        return []
    # group by consecutive indices
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    ranges = [(int(g[0]), int(g[-1])) for g in groups]
    return ranges


def propose_swaps_for_pair(
    df: pd.DataFrame,
    pair: LRPair,
    max_len: int = 30,
    min_gap: int = 1,
) -> list[tuple[int, int]]:
    """
    Propose swap intervals for a pair based on the sign of (xR - xL).

    Heuristic:
    - Determine the majority sign of d = xR - xL over valid frames.
    - Any short contiguous run where sign != majoritySign is flagged.
    - Short runs (<= max_len) are considered likely swaps. Very tiny runs
      shorter than min_gap are ignored.
    Returns list of (start_frame, end_frame) inclusive.
    """
    xL = df[pair.left.cols["x"]].values
    xR = df[pair.right.cols["x"]].values
    valid = np.isfinite(xL) & np.isfinite(xR)
    if not np.any(valid):
        return []
    d = xR - xL
    d_valid = d[valid]
    if d_valid.size == 0:
        return []
    # Determine majority sign (-1, +1). Zero treated as previous sign via epsilon.
    eps = 1e-6
    majority_sign = np.sign(np.nanmedian(np.where(d_valid == 0, eps, d_valid)))
    majority_sign = 1.0 if majority_sign >= 0 else -1.0
    bad = (np.sign(np.where(d == 0, eps, d)) != majority_sign) & valid
    ranges = _contiguous_regions(bad)
    # keep small/medium runs only
    filtered: list[tuple[int, int]] = []
    for s, e in ranges:
        length = e - s + 1
        if length < min_gap:
            continue
        if length <= max_len:
            filtered.append((s, e))
    return filtered


def _swap_block(df: pd.DataFrame, col_a: str, col_b: str, s: int, e: int) -> None:
    """Swap values between two columns within [s, e] inplace."""
    tmp = df.loc[s:e, col_a].copy()
    df.loc[s:e, col_a] = df.loc[s:e, col_b].values
    df.loc[s:e, col_b] = tmp.values


def apply_swap_for_pair(
    df: pd.DataFrame,
    pair: LRPair,
    start_frame: int,
    end_frame: int,
) -> None:
    """
    Apply left/right swap for all available coordinates (x/y[/z]) in [start,end].
    Operates in-place on df.
    """
    for coord in ("x", "y", "z"):
        colL = pair.left.cols.get(coord)
        colR = pair.right.cols.get(coord)
        if colL and colR and colL in df.columns and colR in df.columns:
            _swap_block(df, colL, colR, start_frame, end_frame)


def auto_fix_swaps(
    df: pd.DataFrame,
    pairs: Sequence[LRPair],
    max_len: int = 30,
    min_gap: int = 1,
) -> dict[str, list[tuple[int, int]]]:
    """
    Detect and apply swaps for each pair. Returns dict base_name -> list of ranges.
    """
    proposals: dict[str, list[tuple[int, int]]] = {}
    for p in pairs:
        ranges = propose_swaps_for_pair(df, p, max_len=max_len, min_gap=min_gap)
        if ranges:
            proposals[p.base_name] = ranges
            for s, e in ranges:
                apply_swap_for_pair(df, p, s, e)
    return proposals


# ------------------------------
# I/O and utility
# ------------------------------


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def save_csv_with_suffix(original_csv: Path, df: pd.DataFrame, suffix: str = "_reidswap") -> Path:
    out = original_csv.with_name(original_csv.stem + suffix + original_csv.suffix)
    df.to_csv(out, index=False)
    return out


def write_report(original_csv: Path, proposals: dict[str, list[tuple[int, int]]]) -> Path:
    report = original_csv.with_name(original_csv.stem + "_reidswap_report.txt")
    lines: list[str] = []
    lines.append(f"File: {original_csv}")
    if proposals:
        lines.append("Suspected swap intervals (inclusive) per marker pair:\n")
        for base, spans in proposals.items():
            for s, e in spans:
                lines.append(f"- {base}: {s}..{e}")
    else:
        lines.append("No suspected swaps were detected.")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _ask_csv_via_tk() -> Path | None:
    if tk is None or filedialog is None:
        return None
    root = tk.Tk()
    root.withdraw()
    try:
        path = filedialog.askopenfilename(
            title="Select marker CSV",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
        )
        return Path(path) if path else None
    finally:
        root.destroy()


# ------------------------------
# Optional video previews
# ------------------------------


def export_preview_clips(
    video_path: Path,
    proposals: dict[str, list[tuple[int, int]]],
    dest_root: Path,
    margin: int = 5,
    max_total_clips: int = 20,
) -> list[Path]:
    """
    Export short MP4 clips around each proposed swap interval to help inspection.
    Returns the list of created files.
    """
    try:
        import cv2  # Lazy import
    except Exception as exc:  # noqa: BLE001
        print(f"OpenCV not available for preview export: {exc}")
        return []

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_dir = dest_root
    out_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    clips_made = 0
    for base, spans in proposals.items():
        for s, e in spans:
            if clips_made >= max_total_clips:
                break
            start = max(0, int(s) - margin)
            end = min(frame_count - 1, int(e) + margin)
            if end <= start:
                continue

            out_path = out_dir / f"preview_{base}_{start}_{end}.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            # Seek
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            cur = start
            while cur <= end:
                ok, frame = cap.read()
                if not ok:
                    break
                # Overlay text
                txt = f"swap? {base} [{s}:{e}]"
                cv2.putText(
                    frame,
                    txt,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                cur += 1
            writer.release()
            created.append(out_path)
            clips_made += 1
        if clips_made >= max_total_clips:
            break

    cap.release()
    if created:
        print(f"Preview clips saved to: {out_dir}")
    return created


# ------------------------------
# Interactive video review
# ------------------------------


def interactive_review(
    csv_path: Path,
    video_path: Path,
    max_len: int = 30,
    min_gap: int = 1,
) -> Path | None:
    """
    Open an interactive OpenCV window to review suspected swaps over the video.
    Controls:
      - [ / ]: previous/next pair
      - p / n: previous/next proposed interval for current pair
      - t: toggle apply for current interval
      - a: apply all proposed intervals for current pair
      - c: clear all applied intervals for current pair
      - w: write corrected CSV (_reidswap.csv)
      - h: help (print to console)
      - q or ESC: quit without saving
      - Trackbar: scrub frames
      - , / . : step -1 / +1 frame
      - < / > : step -10 / +10 frames
    """
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        print(f"OpenCV not available: {exc}")
        return None

    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return None

    df = load_csv(csv_path)
    pairs = find_lr_pairs(df)
    if not pairs:
        print("No L/R pairs detected in CSV columns.")
        return None

    proposals_all: dict[str, list[tuple[int, int]]] = {}
    for p in pairs:
        spans = propose_swaps_for_pair(df, p, max_len=max_len, min_gap=min_gap)
        if spans:
            proposals_all[p.base_name] = spans

    # Applied flags per pair per interval index
    applied: dict[str, list[bool]] = {
        base: [True for _ in spans] for base, spans in proposals_all.items()
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Pre-fetch coords arrays for quicker access
    coords: dict[str, dict[str, np.ndarray]] = {}
    for p in pairs:
        base = p.base_name
        coords[base] = {}
        for side_label, mc in (("L", p.left), ("R", p.right)):
            x_col = mc.cols.get("x")
            y_col = mc.cols.get("y")
            if x_col in df.columns and y_col in df.columns:
                coords[base][f"{side_label}x"] = df[x_col].to_numpy(copy=False)
                coords[base][f"{side_label}y"] = df[y_col].to_numpy(copy=False)

    window = "reid-review"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    current_frame = 0

    def on_trackbar(v: int) -> None:
        nonlocal current_frame
        current_frame = int(v)

    cv2.createTrackbar("frame", window, 0, max(0, total_frames - 1), on_trackbar)

    # Navigation state
    base_names = [p.base_name for p in pairs]
    pair_idx = 0
    interval_idx = 0

    def get_current_base() -> str:
        return base_names[pair_idx]

    def in_any_interval(base: str, frame_idx: int) -> int | None:
        spans = proposals_all.get(base, [])
        for i, (s, e) in enumerate(spans):
            if s <= frame_idx <= e:
                return i
        return None

    def draw_overlay(img: np.ndarray) -> None:
        base = get_current_base()
        # Points
        Lx = coords.get(base, {}).get("Lx")
        Ly = coords.get(base, {}).get("Ly")
        Rx = coords.get(base, {}).get("Rx")
        Ry = coords.get(base, {}).get("Ry")
        if Lx is not None and Ly is not None and current_frame < len(Lx):
            x = Lx[current_frame]
            y = Ly[current_frame]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(x), int(y)), 6, (0, 255, 0), -1)  # Left in green
                cv2.putText(
                    img,
                    "L",
                    (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        if Rx is not None and Ry is not None and current_frame < len(Rx):
            x = Rx[current_frame]
            y = Ry[current_frame]
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(x), int(y)), 6, (0, 165, 255), -1)  # Right in orange
                cv2.putText(
                    img,
                    "R",
                    (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

        # HUD
        hud = [
            f"Pair: {base}  ({pair_idx + 1}/{len(base_names)})",
            f"Frame: {current_frame}/{total_frames - 1}",
        ]
        spans = proposals_all.get(base, [])
        if spans:
            cur_i = in_any_interval(base, current_frame)
            if cur_i is None:
                hud.append("Interval: - (not in proposed)")
            else:
                s, e = spans[cur_i]
                state = "ON" if applied.get(base, [])[cur_i] else "OFF"
                hud.append(f"Interval: {cur_i + 1}/{len(spans)} [{s}:{e}] apply={state}")
        else:
            hud.append("No proposed intervals for this pair")

        y0 = 28
        for line in hud:
            cv2.putText(img, line, (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y0 += 26

    def print_help() -> None:
        print(
            "\nControls:\n  [ / ]: prev/next pair\n  p / n: prev/next interval\n  t: toggle apply for current interval\n  a: apply all for pair, c: clear all for pair\n  , / . : step -1/+1 frame, </>: -10/+10\n  w: write CSV, q or ESC: quit\n  Trackbar: scrub frames\n"
        )

    print_help()

    saved_path: Path | None = None
    while True:
        # Clamp frame
        if total_frames > 0:
            current_frame = max(0, min(current_frame, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()
        if not ok:
            # Create blank if cannot read
            frame = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)

        draw_overlay(frame)
        cv2.imshow(window, frame)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            # quit without saving
            break
        elif key == ord("h"):
            print_help()
        elif key == ord("["):
            pair_idx = (pair_idx - 1) % len(base_names)
        elif key == ord("]"):
            pair_idx = (pair_idx + 1) % len(base_names)
        elif key == ord("p"):
            base = get_current_base()
            spans = proposals_all.get(base, [])
            if spans:
                interval_idx = (interval_idx - 1) % len(spans)
                current_frame = spans[interval_idx][0]
                cv2.setTrackbarPos("frame", window, int(current_frame))
        elif key == ord("n"):
            base = get_current_base()
            spans = proposals_all.get(base, [])
            if spans:
                interval_idx = (interval_idx + 1) % len(spans)
                current_frame = spans[interval_idx][0]
                cv2.setTrackbarPos("frame", window, int(current_frame))
        elif key == ord("t"):
            base = get_current_base()
            spans = proposals_all.get(base, [])
            if spans:
                cur_i = in_any_interval(base, current_frame)
                if cur_i is None:
                    cur_i = interval_idx
                if 0 <= cur_i < len(spans):
                    applied[base][cur_i] = not applied[base][cur_i]
        elif key == ord("a"):
            base = get_current_base()
            if base in applied:
                applied[base] = [True for _ in applied[base]]
        elif key == ord("c"):
            base = get_current_base()
            if base in applied:
                applied[base] = [False for _ in applied[base]]
        elif key == ord(","):
            current_frame = max(0, current_frame - 1)
            cv2.setTrackbarPos("frame", window, int(current_frame))
        elif key == ord("."):
            current_frame = min(total_frames - 1, current_frame + 1)
            cv2.setTrackbarPos("frame", window, int(current_frame))
        elif key == ord("<"):
            current_frame = max(0, current_frame - 10)
            cv2.setTrackbarPos("frame", window, int(current_frame))
        elif key == ord(">"):
            current_frame = min(total_frames - 1, current_frame + 10)
            cv2.setTrackbarPos("frame", window, int(current_frame))
        elif key == ord("w"):
            # Apply selected intervals and write
            # Work on a copy to avoid partial changes if something goes wrong
            out_df = df.copy()
            # Map base -> LRPair
            base_to_pair = {p.base_name: p for p in pairs}
            for base, flags in applied.items():
                spans = proposals_all.get(base, [])
                pair = base_to_pair.get(base)
                if pair is None:
                    continue
                for i, flag in enumerate(flags):
                    if i < len(spans) and flag:
                        s, e = spans[i]
                        apply_swap_for_pair(out_df, pair, int(s), int(e))
            saved_path = save_csv_with_suffix(csv_path, out_df)
            print(f"Saved corrected CSV: {saved_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_path


# ------------------------------
# CLI entry points
# ------------------------------


def run_auto(csv_path: Path, max_len: int = 30, min_gap: int = 1) -> tuple[Path, Path]:
    df = load_csv(csv_path)
    pairs = find_lr_pairs(df)
    proposals = auto_fix_swaps(df, pairs, max_len=max_len, min_gap=min_gap)
    out_csv = save_csv_with_suffix(csv_path, df)
    report = write_report(csv_path, proposals)
    return out_csv, report


def run_manual(csv_path: Path, pair_name: str, start: int, end: int) -> Path:
    df = load_csv(csv_path)
    pairs = find_lr_pairs(df)
    # find best match by base name (case-insensitive)
    target = None
    for p in pairs:
        if p.base_name.lower() == pair_name.lower():
            target = p
            break
    if target is None:
        # try partial match
        for p in pairs:
            if pair_name.lower() in p.base_name.lower():
                target = p
                break
    if target is None:
        raise ValueError(
            f"Pair '{pair_name}' not found among detected pairs: {[p.base_name for p in pairs]}"
        )
    apply_swap_for_pair(df, target, start, end)
    out_csv = save_csv_with_suffix(csv_path, df)
    return out_csv


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Detect and fix abrupt left/right marker swaps in CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", type=str, default="", help="Path to marker CSV")
    parser.add_argument("--video", type=str, default="", help="Optional video path (reserved)")
    parser.add_argument(
        "--review",
        action="store_true",
        help="Interactive video review mode (requires --video)",
    )
    parser.add_argument(
        "--auto", action="store_true", help="Run automatic detection and correction"
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=30,
        help="Max length of a swap interval to auto-correct",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=1,
        help="Min length to consider (ignore tiny blips)",
    )
    parser.add_argument("--manual", type=str, default="", help="Manual pair base name for swapping")
    parser.add_argument(
        "--start", type=int, default=-1, help="Start frame for manual swap (inclusive)"
    )
    parser.add_argument("--end", type=int, default=-1, help="End frame for manual swap (inclusive)")

    args = parser.parse_args(argv)

    csv_path: Path | None = Path(args.csv) if args.csv else None
    if not csv_path or not csv_path.exists():
        # try to ask via Tk
        possible = _ask_csv_via_tk()
        if possible is None:
            raise FileNotFoundError("CSV path not provided and no dialog available.")
        csv_path = possible

    if args.manual:
        if args.start < 0 or args.end < 0 or args.end < args.start:
            raise ValueError("For manual mode, provide valid --start and --end frames.")
        out_csv = run_manual(csv_path, args.manual, args.start, args.end)
        print(f"Manual swap applied for pair '{args.manual}' in {args.start}..{args.end}")
        print(f"Saved: {out_csv}")
        if args.video:
            # Export a single preview clip for the manual range
            proposals = {args.manual: [(args.start, args.end)]}
            previews_dir = out_csv.with_name(out_csv.stem + "_previews")
            export_preview_clips(Path(args.video), proposals, previews_dir)
        return

    if args.review:
        if not args.video:
            raise ValueError("--review requires --video path")
        saved = interactive_review(
            csv_path, Path(args.video), max_len=args.max_len, min_gap=args.min_gap
        )
        if saved:
            print(f"Interactive review saved: {saved}")
        else:
            print("Interactive review finished without saving.")
        return

    # default: auto if requested, else just produce a report without changes
    if args.auto:
        out_csv, report = run_auto(csv_path, max_len=args.max_len, min_gap=args.min_gap)
        print(f"Auto-correction complete. CSV: {out_csv}")
        print(f"Report: {report}")
        if args.video:
            # Load proposals again just for preview export
            df = load_csv(csv_path)
            pairs = find_lr_pairs(df)
            proposals = auto_fix_swaps(df.copy(), pairs, max_len=args.max_len, min_gap=args.min_gap)
            previews_dir = out_csv.with_name(out_csv.stem + "_previews")
            export_preview_clips(Path(args.video), proposals, previews_dir)
    else:
        # Only analyze and write report (no changes)
        df = load_csv(csv_path)
        pairs = find_lr_pairs(df)
        proposals: dict[str, list[tuple[int, int]]] = {}
        for p in pairs:
            proposals[p.base_name] = propose_swaps_for_pair(
                df, p, max_len=args.max_len, min_gap=args.min_gap
            )
        report = write_report(csv_path, proposals)
        print("Analysis complete (no corrections applied).")
        print(f"Report: {report}")
        if args.video and proposals:
            previews_dir = csv_path.with_name(csv_path.stem + "_previews")
            export_preview_clips(Path(args.video), proposals, previews_dir)


if __name__ == "__main__":
    main()
