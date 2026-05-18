"""Soccer-field 2D calibration (DLT2D) for vailá.

Project: vailá — vaila/soccerfield_calib.py
Update Date: 14 May 2026
Version: 0.3.44

Pipeline (per video):

1. Load the FIFA 105 × 68 m reference from ``models/soccerfield_ref3d.csv`` —
   29 named keypoints with world coordinates (X, Y, Z≈0).
2. Let the user click at least 6 visible keypoints on a single video frame
   (via :mod:`vaila.getpixelvideo` or any pre-existing pixel CSV).
3. Estimate the homography by fitting DLT2D (:func:`vaila.dlt2d.dlt2d`).
4. Report reprojection error per point and save:
   - ``<stem>_ref2d.csv`` — the X/Y pairs used for fitting.
   - ``<stem>.dlt2d`` — the 8 DLT parameters (as the existing CLI does).
   - ``<stem>_homography_report.txt`` — per-point reprojection error.
5. Optional: when ``--data-root`` is given, also drop a
   ``cameras/<stem>_homography.npz`` (keys ``H``, ``dlt2d``, ``ref_points``,
   ``pixel_points``) that the FIFA pipeline can consume as a fallback for
   sequences that do not ship an official ``cameras/<stem>.npz``.

Batch reconstruction of player pixel trajectories → field metres is available
through :func:`reconstruct_world_coords`, which wraps :func:`vaila.rec2d.rec2d`.

Z vertical (vertical anchors — goalposts, sideboards, anthropometry)::

    # TODO: Z vertical (DLT3D future work, deferred)
    #   - left_goal_top_post, right_goal_top_post (Z = 2.44 m)
    #   - stadium sideboards / crossbar endpoints
    #   - average player height (1.80 m) as a reprojection prior
    # This script only handles the Z = 0 ground plane today.

Author: vailá team (17 April 2026).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .dlt2d import dlt2d
    from .rec2d import rec2d
except ImportError:
    from dlt2d import dlt2d  # ty: ignore[unresolved-import]
    from rec2d import rec2d  # ty: ignore[unresolved-import]

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_REF3D = _MODULE_DIR / "models" / "soccerfield_ref3d.csv"
# 32-kp order matches ``soccerfield_keypoints_ai`` outputs (p1..p32) and Roboflow F07VI / pitch32 model.
DEFAULT_REF3D_FIFA_DATASET = _MODULE_DIR / "models" / "soccerfield_ref3d_fifa_dataset.csv"

# Keypoint presets (name → priority). Users are free to use *any* point_name
# from the CSV; this just drives the default GUI suggestions. The top-6 are
# chosen so that even zoomed broadcast crops can cover them.
DEFAULT_SUGGESTED_POINTS: tuple[str, ...] = (
    "bottom_left_corner",
    "bottom_right_corner",
    "top_left_corner",
    "top_right_corner",
    "midfield_left",
    "midfield_right",
    "center_field",
    "left_penalty_spot",
    "right_penalty_spot",
    "left_penalty_arc_top",
    "right_penalty_arc_top",
)


@dataclass
class FieldKeypoint:
    name: str
    number: int
    world_xy: np.ndarray  # shape (2,)

    @property
    def world_xyz(self) -> np.ndarray:
        return np.array([self.world_xy[0], self.world_xy[1], 0.0])


def load_field_reference(csv_path: Path | str = DEFAULT_REF3D) -> list[FieldKeypoint]:
    """Load the soccer-field FIFA reference CSV (29 named keypoints).

    Accepts the canonical layout ``point_name,point_number,x,y,z`` (produced
    by ``models/soccerfield_ref3d.csv``). Only X,Y are retained; Z is assumed 0.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"soccer-field reference not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"point_name", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
    out: list[FieldKeypoint] = []
    for _, row in df.iterrows():
        out.append(
            FieldKeypoint(
                name=str(row["point_name"]).strip(),
                number=int(row.get("point_number", len(out) + 1)),
                world_xy=np.array([float(row["x"]), float(row["y"])]),
            )
        )
    return out


def _filter_reference(keypoints: list[FieldKeypoint], names: list[str]) -> list[FieldKeypoint]:
    by_name = {kp.name: kp for kp in keypoints}
    out: list[FieldKeypoint] = []
    for n in names:
        if n not in by_name:
            raise KeyError(f"unknown soccer-field keypoint name: {n!r}")
        out.append(by_name[n])
    return out


def _pitch_index_from_header(name: str) -> int | None:
    n = name.strip()
    if n.startswith("p") and n[1:].isdigit():
        return int(n[1:])
    return None


def names_are_pitch32_style(names: list[str]) -> bool:
    """True if all names look like p1, p2, … (``soccerfield_keypoints_ai`` wide CSV)."""
    if not names:
        return False
    return all(_pitch_index_from_header(n) is not None for n in names)


def extract_points_from_dataframe_row(
    row: pd.Series,
    cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract finite (x,y) pairs from one CSV row (paired p*_x / p*_y columns)."""
    cols = [c.strip() for c in cols]
    if {"name", "x", "y"}.issubset(set(cols)):
        raise ValueError("extract_points_from_dataframe_row expects paired p*_x/p*_y columns")

    coord_cols = [c for c in cols if c != "frame"]
    if len(coord_cols) % 2 != 0:
        raise ValueError(
            f"pixel row has an odd number of coordinate columns after removing 'frame': {coord_cols}"
        )

    pts_list: list[list[float]] = []
    names: list[str] = []
    for i in range(0, len(coord_cols), 2):
        cx, cy = coord_cols[i], coord_cols[i + 1]
        name = cx
        for suffix in ("_x", ".x", "X"):
            if name.lower().endswith(suffix.lower()):
                name = name[: -len(suffix)]
                break
        try:
            x = float(row[cx])
            y = float(row[cy])
        except (TypeError, ValueError):
            continue
        if np.isnan(x) or np.isnan(y):
            continue
        pts_list.append([x, y])
        names.append(name)

    return np.asarray(pts_list, dtype=float), names


def align_pitch32_pixels_to_world(
    pixel_points: np.ndarray,
    names: list[str],
    ref_ordered: list[FieldKeypoint],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Map ``p1``…``p32`` pixels to ``ref_ordered`` rows by index (pitch32 / F07VI order)."""
    if len(pixel_points) != len(names):
        raise ValueError(f"pixel ({len(pixel_points)}) vs name ({len(names)}) count mismatch")
    order = sorted(
        range(len(names)), key=lambda i: int(names[i][1:]) if names[i][1:].isdigit() else 9999
    )
    pix_list: list[np.ndarray] = []
    world_list: list[np.ndarray] = []
    name_list: list[str] = []
    for i in order:
        nm = names[i]
        k = _pitch_index_from_header(nm)
        if k is None or k < 1 or k > len(ref_ordered):
            continue
        pix_list.append(pixel_points[i])
        world_list.append(ref_ordered[k - 1].world_xy.copy())
        name_list.append(ref_ordered[k - 1].name)
    return (
        np.asarray(pix_list, dtype=float),
        np.asarray(world_list, dtype=float),
        name_list,
    )


def load_pixel_points(
    pixel_csv: Path | str,
    *,
    frame_index: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load pixel points from a CSV.

    Supports two layouts:

    - **Paired columns (vailá default)**: ``frame,p1_x,p1_y,p2_x,p2_y,...``.
      Returns the pixels for a single frame (``frame_index`` or first row).
    - **Named rows**: ``name,x,y`` (one row per keypoint).

    Returns the (N, 2) pixel array and the list of point names that identify
    each row. For paired layouts the names default to ``point_name``s loaded
    from the column headers, or ``p1``, ``p2``, … when headers are purely
    numeric.
    """
    pixel_csv = Path(pixel_csv)
    df = pd.read_csv(pixel_csv)
    cols = [c.strip() for c in df.columns]

    if {"name", "x", "y"}.issubset(set(cols)):
        names = [str(n).strip() for n in df["name"].tolist()]
        pts = df[["x", "y"]].to_numpy(dtype=float)
        return pts, names

    if "frame" in cols:
        if frame_index is None:
            row = df.iloc[0]
        else:
            match = df[df["frame"] == frame_index]
            if match.empty:
                raise ValueError(f"frame {frame_index} not found in {pixel_csv}")
            row = match.iloc[0]
    else:
        if frame_index is not None and frame_index < len(df):
            row = df.iloc[frame_index]
        else:
            row = df.iloc[0]

    return extract_points_from_dataframe_row(row, cols)


def compute_dlt2d(
    pixel_points: np.ndarray, world_points: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit DLT2D parameters and compute the reprojection error.

    Args:
        pixel_points: (N, 2) array of pixel coordinates.
        world_points: (N, 2) array of world (metre) coordinates on Z = 0.

    Returns:
        (params, rms_error, per_point_error). ``params`` is an 8-vector
        compatible with :func:`vaila.rec2d.rec2d`. Errors are in pixels.
    """
    pixel_points = np.asarray(pixel_points, dtype=float)
    world_points = np.asarray(world_points, dtype=float)
    if pixel_points.shape[0] < 6:
        raise ValueError(
            f"DLT2D needs at least 6 point correspondences, got {pixel_points.shape[0]}"
        )
    if pixel_points.shape != world_points.shape:
        raise ValueError(
            f"pixel and world arrays must have the same shape, "
            f"got {pixel_points.shape} vs {world_points.shape}"
        )

    params = dlt2d(world_points, pixel_points)
    reproj = rec2d(params, pixel_points)
    per_point = np.linalg.norm(reproj - world_points, axis=1)
    rms = float(np.sqrt(np.mean(per_point**2)))
    return params, rms, per_point


def save_dlt2d_params(out_path: Path, params: np.ndarray, *, frame: int = 0) -> None:
    """Write the 8 DLT2D parameters to ``*.dlt2d`` (one row: frame + 8 columns)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [[int(frame), *params.tolist()]],
        columns=["frame", *[f"p{i + 1}" for i in range(8)]],  # ty: ignore[invalid-argument-type]
    )
    df.to_csv(out_path, index=False)


def save_dlt2d_params_per_frame(
    out_path: Path, frames_params: list[tuple[int, np.ndarray]]
) -> None:
    """Write multi-row ``*.dlt2d`` (one DLT row per frame) for ``rec2d`` / FIFA."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[int(fr), *np.asarray(p, dtype=float).tolist()] for fr, p in frames_params]
    df = pd.DataFrame(rows, columns=["frame", *[f"p{i + 1}" for i in range(8)]])
    df.to_csv(out_path, index=False)


def save_homography_report(
    out_path: Path,
    names: list[str],
    world_points: np.ndarray,
    pixel_points: np.ndarray,
    per_point_error: np.ndarray,
    rms_error: float,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Soccer-field DLT2D calibration report"]
    lines.append(f"# points: {len(names)}    RMS(m): {rms_error:.4f}")
    lines.append("#")
    lines.append(
        "# idx  name                               world_x   world_y   pixel_x   pixel_y   err(m)"
    )
    for i, (name, wp, px, err) in enumerate(
        zip(names, world_points, pixel_points, per_point_error, strict=False)
    ):
        lines.append(
            f"{i:3d}  {name:<34}  "
            f"{wp[0]:7.3f}  {wp[1]:7.3f}  "
            f"{px[0]:8.2f}  {px[1]:8.2f}  {err:6.3f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_ref2d_csv(out_path: Path, names: list[str], world_points: np.ndarray) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "point_name": names,
            "x": world_points[:, 0],
            "y": world_points[:, 1],
        }
    )
    df.to_csv(out_path, index=False)


def reconstruct_world_coords(
    dlt2d_params: np.ndarray,
    pixel_points: np.ndarray,
) -> np.ndarray:
    """Convert an (N, 2) array of pixel points to world metres using DLT2D."""
    return rec2d(np.asarray(dlt2d_params, dtype=float), np.asarray(pixel_points, dtype=float))


# ---------------------------------------------------------------------------
# GUI integration (optional)
# ---------------------------------------------------------------------------
def _run_getpixelvideo(video: Path, output_dir: Path) -> Path:
    """Invoke ``vaila.getpixelvideo`` as a subprocess and return the saved CSV.

    ``getpixelvideo`` only exposes ``-f <video>`` (no ``--video/--output``);
    it writes the CSV next to the video when the user clicks "Save". After
    the GUI closes we collect any CSV under the video's parent or the
    requested ``output_dir`` whose ``mtime`` is newer than the launch time.
    """
    import shutil
    import subprocess
    import time

    video = Path(video).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    uv = shutil.which("uv") or sys.executable
    script = Path(__file__).resolve().parent / "getpixelvideo.py"
    if uv == sys.executable:
        cmd = [sys.executable, str(script), "-f", str(video)]
    else:
        cmd = [uv, "run", str(script), "-f", str(video)]
    print(f">> launching getpixelvideo: {' '.join(str(c) for c in cmd)}")

    t0 = time.time() - 2.0  # 2s slack to absorb clock skew
    subprocess.run(cmd, check=False)

    search_dirs = {output_dir, video.parent}
    candidates: list[Path] = []
    for d in search_dirs:
        for pattern in ("*.csv", "*.dat"):
            for p in d.glob(pattern):
                try:
                    if p.stat().st_mtime >= t0:
                        candidates.append(p)
                except OSError:
                    continue
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise FileNotFoundError(
            "getpixelvideo did not produce a CSV. Click 'Save' in the GUI before closing. "
            f"Searched {sorted(str(d) for d in search_dirs)}."
        )

    chosen = candidates[0]
    if chosen.parent != output_dir:
        target = output_dir / chosen.name
        try:
            shutil.copy2(chosen, target)
            chosen = target
        except OSError:
            pass
    return chosen


def run_soccerfield_calib(
    *,
    video: Path | None = None,
    ref3d_csv: Path | None = None,
    pixel_csv: Path | None = None,
    output_dir: Path | None = None,
    frame_index: int | None = None,
    data_root: Path | None = None,
    keypoint_names: list[str] | None = None,
    all_frames: bool = False,
    pitch32: bool | None = None,
    min_points: int = 6,
) -> dict:
    """High-level entry point used by the CLI and by the ``vaila.py`` GUI.

    When ``pixel_csv`` is ``None`` and ``video`` is given, launches
    ``getpixelvideo`` so the user can click pixels interactively. Returns a
    dict summarising the artifacts that were produced.

    ``pitch32``: when True (or auto-detected from ``p1``… column names), pixel
    columns are aligned to rows of ``DEFAULT_REF3D_FIFA_DATASET`` (same order
    as ``soccerfield_keypoints_ai`` / Roboflow F07VI). ``all_frames``: fit one
    DLT per row in a multi-row ``field_keypoints_getpixelvideo.csv``.
    """
    if video is None and pixel_csv is None:
        raise ValueError("either --video or --pixels is required")

    video_path = Path(video) if video is not None else None
    pixel_path = Path(pixel_csv) if pixel_csv is not None else None

    if pixel_path is not None:
        video_stem = pixel_path.stem
    else:
        assert video_path is not None
        video_stem = video_path.stem

    if output_dir is None:
        if pixel_path is not None:
            base = pixel_path.parent
        else:
            assert video_path is not None
            base = video_path.parent
        output_dir = base / f"{video_stem}_soccerfield_calib"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pixel_path is None:
        assert video_path is not None
        pixel_path = _run_getpixelvideo(video_path, output_dir)
    pixel_csv_path = pixel_path

    df_pix = pd.read_csv(pixel_csv_path)
    cols = [c.strip() for c in df_pix.columns]
    name_xy_layout = {"name", "x", "y"}.issubset(set(cols))

    if not name_xy_layout:
        _, preview_names = extract_points_from_dataframe_row(df_pix.iloc[0], cols)
        use_pitch32 = pitch32 if pitch32 is not None else names_are_pitch32_style(preview_names)
    else:
        use_pitch32 = False

    if ref3d_csv is None:
        ref_path = DEFAULT_REF3D_FIFA_DATASET if use_pitch32 else DEFAULT_REF3D
    else:
        ref_path = Path(ref3d_csv)
        if use_pitch32 and ref_path.resolve() == DEFAULT_REF3D.resolve():
            ref_path = DEFAULT_REF3D_FIFA_DATASET

    ref_all = load_field_reference(ref_path)
    if use_pitch32 and len(ref_all) < 32:
        raise ValueError(
            f"--pitch32 expects a 32-kp reference (e.g. {DEFAULT_REF3D_FIFA_DATASET.name}), "
            f"got {len(ref_all)} points in {ref_path}"
        )

    dlt2d_path = output_dir / f"{video_stem}.dlt2d"
    ref2d_path = output_dir / f"{video_stem}_ref2d.csv"
    report_path = output_dir / f"{video_stem}_homography_report.txt"

    multi = bool(all_frames) and "frame" in cols and len(df_pix) > 1 and not name_xy_layout

    def _one_frame(
        row: pd.Series,
    ) -> tuple[int, np.ndarray, list[str], np.ndarray, np.ndarray, float, np.ndarray] | None:
        xy, names = extract_points_from_dataframe_row(row, cols)
        if len(xy) < min_points:
            return None
        if use_pitch32:
            xy_a, world_a, knames = align_pitch32_pixels_to_world(xy, names, ref_all)
        else:
            kn = keypoint_names or names
            if len(kn) != len(xy):
                raise ValueError(
                    f"pixel count ({len(xy)}) must match keypoint name count ({len(kn)}). "
                    "Rename columns or pass --keypoints."
                )
            selected = _filter_reference(ref_all, kn)
            world_a = np.asarray([kp.world_xy for kp in selected])
            knames = kn
            xy_a = xy
        if len(xy_a) < min_points:
            return None
        params, rms, per_point = compute_dlt2d(xy_a, world_a)
        fr_raw = row["frame"] if "frame" in row.index else 0
        fr = int(fr_raw) if not pd.isna(fr_raw) else 0
        return fr, params, knames, world_a, xy_a, rms, per_point

    if multi:
        dlt_rows: list[tuple[int, np.ndarray]] = []
        first_bundle: tuple[list[str], np.ndarray, np.ndarray, np.ndarray, float] | None = None
        for j in range(len(df_pix)):
            got = _one_frame(df_pix.iloc[j])
            if got is None:
                continue
            fr, params, knames, world_a, xy_a, rms, per_point = got
            dlt_rows.append((fr, params))
            if first_bundle is None:
                first_bundle = (knames, world_a, xy_a, per_point, rms)
        if not dlt_rows:
            raise ValueError(
                f"No frame had at least {min_points} correspondences for DLT2D ({pixel_csv_path})."
            )
        assert first_bundle is not None
        knames, world_a, xy_a, per_point, rms = first_bundle
        save_dlt2d_params_per_frame(dlt2d_path, dlt_rows)
        save_ref2d_csv(ref2d_path, knames, world_a)
        save_homography_report(report_path, knames, world_a, xy_a, per_point, rms)
        params_last = dlt_rows[-1][1]
        npz_path: Path | None = None
        if data_root is not None:
            data_root = Path(data_root).resolve()
            cam_dir = data_root / "cameras"
            cam_dir.mkdir(parents=True, exist_ok=True)
            npz_path = cam_dir / f"{video_stem}_homography.npz"
            np.savez(
                npz_path,
                dlt2d=params_last,
                ref_points=world_a,
                pixel_points=xy_a,
                keypoint_names=np.array(knames, dtype=object),
                rms=np.array(rms),
            )
        return {
            "video": str(video) if video else None,
            "pixel_csv": str(pixel_csv_path),
            "dlt2d": str(dlt2d_path),
            "ref2d": str(ref2d_path),
            "homography_report": str(report_path),
            "cameras_npz": str(npz_path) if npz_path else None,
            "rms_pixels_or_metres": float(rms),
            "n_points": int(len(knames)),
            "n_dlt_frames": int(len(dlt_rows)),
        }

    pixel_points_, names_ = load_pixel_points(pixel_csv_path, frame_index=frame_index)
    if use_pitch32:
        pixel_points, world_points, keypoint_names_out = align_pitch32_pixels_to_world(
            pixel_points_, names_, ref_all
        )
    else:
        keypoint_names_out = names_ if keypoint_names is None else keypoint_names
        if len(keypoint_names_out) != len(pixel_points_):
            raise ValueError(
                f"pixel count ({len(pixel_points_)}) must match keypoint name count "
                f"({len(keypoint_names_out)}). Rename columns or pass --keypoints."
            )
        selected = _filter_reference(ref_all, keypoint_names_out)
        world_points = np.asarray([kp.world_xy for kp in selected])
        pixel_points = pixel_points_

    params, rms, per_point = compute_dlt2d(pixel_points, world_points)

    frame_for_file = 0
    if "frame" in cols and frame_index is None:
        frame_for_file = int(df_pix.iloc[0]["frame"])
    elif frame_index is not None:
        frame_for_file = int(frame_index)

    save_dlt2d_params(dlt2d_path, params, frame=frame_for_file)
    save_ref2d_csv(ref2d_path, keypoint_names_out, world_points)
    save_homography_report(
        report_path, keypoint_names_out, world_points, pixel_points, per_point, rms
    )

    npz_path_single: Path | None = None
    if data_root is not None:
        data_root = Path(data_root).resolve()
        cam_dir = data_root / "cameras"
        cam_dir.mkdir(parents=True, exist_ok=True)
        npz_path_single = cam_dir / f"{video_stem}_homography.npz"
        np.savez(
            npz_path_single,
            dlt2d=params,
            ref_points=world_points,
            pixel_points=pixel_points,
            keypoint_names=np.array(keypoint_names_out, dtype=object),
            rms=np.array(rms),
        )

    return {
        "video": str(video) if video else None,
        "pixel_csv": str(pixel_csv_path),
        "dlt2d": str(dlt2d_path),
        "ref2d": str(ref2d_path),
        "homography_report": str(report_path),
        "cameras_npz": str(npz_path_single) if npz_path_single else None,
        "rms_pixels_or_metres": float(rms),
        "n_points": int(len(keypoint_names_out)),
        "n_dlt_frames": 1,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Soccer-field 2D calibration via DLT2D — estimate a 105×68 m homography "
            "from pixel keypoints and project player tracks to world metres."
        )
    )
    p.add_argument("-v", "--video", type=Path, default=None, help="Input video (triggers GUI)")
    p.add_argument("-p", "--pixels", type=Path, default=None, help="Pre-picked pixel CSV")
    p.add_argument(
        "-r",
        "--ref3d",
        type=Path,
        default=None,
        help=(
            "World reference CSV. Default: soccerfield_ref3d.csv (legacy named points), "
            "or soccerfield_ref3d_fifa_dataset.csv when --pitch32 / field_keypoints CSV."
        ),
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    p.add_argument("--frame", type=int, default=None, help="Frame index for paired-column CSVs")
    p.add_argument(
        "--all-frames",
        action="store_true",
        help=(
            "Fit one DLT2D per row (uses frame column). For field_keypoints_getpixelvideo.csv "
            "from soccerfield_keypoints_ai video mode."
        ),
    )
    p.add_argument(
        "--pitch32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="p1..pN columns map to 32-kp FIFA-dataset reference order (auto if filename contains "
        "'field_keypoints').",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="FIFA data root; also drops cameras/<stem>_homography.npz",
    )
    p.add_argument(
        "--keypoints",
        type=str,
        default=None,
        help="Comma-separated soccer-field keypoint names overriding CSV column names",
    )
    p.add_argument(
        "--list-keypoints",
        action="store_true",
        help="Print the list of valid soccer-field keypoint names and exit",
    )
    p.add_argument(
        "--from-sam",
        type=Path,
        default=None,
        metavar="SAM_VIDEO_DIR",
        help="Per-video SAM3 output directory (e.g. processed_sam_*/<video>/). "
        "When given, defaults --video to its overlay MP4 and --output to "
        "<sam_dir>/calib/. Combine with --pixels to skip the GUI.",
    )
    return p


def _resolve_sam_video_overlay(sam_dir: Path) -> Path:
    overlays = sorted(sam_dir.glob("*_sam_overlay.mp4"))
    if not overlays:
        raise FileNotFoundError(
            f"No *_sam_overlay.mp4 found in {sam_dir} (expected from a SAM3 batch)."
        )
    return overlays[0]


def run_gui() -> None:
    """Tkinter flow when ``soccerfield_calib`` is launched with no CLI args."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.withdraw()

    pix = filedialog.askopenfilename(
        parent=root,
        title="Pixel CSV (e.g. field_keypoints_getpixelvideo.csv)",
        filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
    )
    if not pix:
        root.destroy()
        return
    out_dir = filedialog.askdirectory(parent=root, title="Output folder")
    if not out_dir:
        root.destroy()
        return

    root.title("Soccer-field calib (DLT2D)")
    root.minsize(520, 320)
    root.geometry("+120+120")
    root.deiconify()
    frm = ttk.Frame(root, padding=10)
    frm.pack(fill=tk.BOTH, expand=True)

    pix_path = Path(pix)
    ref_var = tk.StringVar(value=str(DEFAULT_REF3D_FIFA_DATASET))
    vf = tk.StringVar(value="")
    pitch_var = tk.BooleanVar(value="field_keypoints" in pix_path.name.lower())
    allf_var = tk.BooleanVar(value=True)
    minpt_var = tk.StringVar(value="6")
    dr_var = tk.StringVar(value="")

    ttk.Label(frm, text=f"Pixels:\n{pix_path}", wraplength=520).pack(anchor="w")
    ttk.Label(frm, text="Reference 3D CSV (world metres, Z=0)").pack(anchor="w", pady=(10, 0))
    ttk.Entry(frm, textvariable=ref_var, width=76).pack(fill=tk.X, pady=4)

    def browse_ref() -> None:
        p = filedialog.askopenfilename(
            parent=root, filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if p:
            ref_var.set(p)

    ttk.Button(frm, text="Browse ref3d…", command=browse_ref).pack(anchor="w")
    ttk.Checkbutton(
        frm,
        text="Pitch32 mode (p1…p32 ↔ vailá FIFA-dataset 32-kp order)",
        variable=pitch_var,
    ).pack(anchor="w", pady=(10, 0))
    ttk.Checkbutton(
        frm,
        text="All frames — one DLT row per CSV row (video-mode keypoints AI)",
        variable=allf_var,
    ).pack(anchor="w")
    rowm = ttk.Frame(frm)
    rowm.pack(anchor="w", pady=(6, 0))
    ttk.Label(rowm, text="Min correspondences per frame:").pack(side=tk.LEFT)
    ttk.Entry(rowm, textvariable=minpt_var, width=6).pack(side=tk.LEFT, padx=6)

    ttk.Label(frm, text="Optional FIFA data-root (writes cameras/*_homography.npz):").pack(
        anchor="w", pady=(10, 0)
    )
    ttk.Entry(frm, textvariable=dr_var, width=76).pack(fill=tk.X)
    ttk.Label(
        frm, text="Optional video path (only if you need getpixelvideo; leave empty if CSV ready):"
    ).pack(anchor="w", pady=(10, 0))
    ttk.Entry(frm, textvariable=vf, width=76).pack(fill=tk.X)

    def browse_v() -> None:
        p = filedialog.askopenfilename(
            parent=root,
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if p:
            vf.set(p)

    ttk.Button(frm, text="Browse video…", command=browse_v).pack(anchor="w")

    def do_run() -> None:
        try:
            ref_s = ref_var.get().strip()
            res = run_soccerfield_calib(
                video=Path(vf.get()) if vf.get().strip() else None,
                ref3d_csv=Path(ref_s) if ref_s else None,
                pixel_csv=pix_path,
                output_dir=Path(out_dir),
                data_root=Path(dr_var.get().strip()) if dr_var.get().strip() else None,
                pitch32=pitch_var.get(),
                all_frames=allf_var.get(),
                min_points=int(minpt_var.get().strip() or "6"),
            )
            messagebox.showinfo(
                "Calibration done",
                f"dlt2d: {res['dlt2d']}\nref2d: {res['ref2d']}\n"
                f"DLT rows: {res.get('n_dlt_frames', 1)}",
                parent=root,
            )
            root.destroy()
        except Exception as e:
            messagebox.showerror("Calibration failed", str(e), parent=root)

    bf = ttk.Frame(frm)
    bf.pack(fill=tk.X, pady=14)
    ttk.Button(bf, text="Run", command=do_run).pack(side=tk.LEFT)
    ttk.Button(bf, text="Close", command=root.destroy).pack(side=tk.LEFT, padx=8)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    try:
        root.lift()
        root.attributes("-topmost", True)
        root.after(50, lambda: root.attributes("-topmost", False))
    except tk.TclError:
        pass
    root.focus_force()
    root.mainloop()


def main(argv: list[str] | None = None) -> None:
    print(f"--- {os.path.basename(__file__)} ---")
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.list_keypoints:
        ref_l = args.ref3d if args.ref3d is not None else DEFAULT_REF3D
        for kp in load_field_reference(ref_l):
            print(f"{kp.number:3d}  {kp.name}  X={kp.world_xy[0]:7.3f}  Y={kp.world_xy[1]:7.3f}")
        return

    if args.from_sam is None and args.video is None and args.pixels is None:
        run_gui()
        return

    if args.from_sam is not None:
        sam_dir = Path(args.from_sam).resolve()
        if args.video is None:
            args.video = _resolve_sam_video_overlay(sam_dir)
        if args.output is None:
            args.output = sam_dir / "calib"

    keypoint_names: list[str] | None = None
    if args.keypoints:
        keypoint_names = [n.strip() for n in args.keypoints.split(",") if n.strip()]

    pitch32_flag = args.pitch32
    if pitch32_flag is None and args.pixels is not None:
        pitch32_flag = "field_keypoints" in str(args.pixels).lower()
    if pitch32_flag is None:
        pitch32_flag = False

    ref_eff = args.ref3d
    if ref_eff is None:
        ref_eff = DEFAULT_REF3D_FIFA_DATASET if pitch32_flag else DEFAULT_REF3D

    res = run_soccerfield_calib(
        video=args.video,
        ref3d_csv=ref_eff,
        pixel_csv=args.pixels,
        output_dir=args.output,
        frame_index=args.frame,
        data_root=args.data_root,
        keypoint_names=keypoint_names,
        all_frames=bool(args.all_frames),
        pitch32=pitch32_flag,
    )
    print(">> Soccer-field DLT2D calibration complete")
    for k, v in res.items():
        print(f"   {k}: {v}")


def launch_calibrate_dialog(parent, batch_root: Path) -> None:
    """Tk dialog launched from `vaila_sam.py` after a SAM3 batch finishes.

    Lets the user pick one per-video SAM directory and either supply a
    pre-existing pixel CSV (skip GUI) or open ``getpixelvideo`` on the
    overlay MP4 to mark the soccer-field keypoints. Runs the calibration
    in a background thread and shows the resulting RMS in a messagebox.
    """
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    batch_root = Path(batch_root).resolve()
    sam_subdirs = sorted(
        d for d in batch_root.iterdir() if d.is_dir() and (d / "sam_frames_meta.csv").is_file()
    )
    if not sam_subdirs:
        messagebox.showwarning(
            "Calibrate field",
            f"No per-video SAM runs found under {batch_root}.\n"
            "Run a SAM3 batch first or pick the right base directory.",
            parent=parent,
        )
        return

    win = tk.Toplevel(parent)
    win.title("Calibrate field (DLT2D)")
    win.geometry("640x420")
    win.transient(parent)  # ty: ignore[no-matching-overload]

    frm = ttk.Frame(win, padding=10)
    frm.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frm, text="Pick a SAM video directory:").pack(anchor="w")
    listbox = tk.Listbox(frm, height=10)
    for d in sam_subdirs:
        listbox.insert(tk.END, d.name)
    listbox.pack(fill=tk.BOTH, expand=True, pady=(2, 8))
    listbox.selection_set(0)

    pixels_var = tk.StringVar(value="")
    pix_row = ttk.Frame(frm)
    pix_row.pack(fill=tk.X)
    ttk.Label(pix_row, text="Pixel CSV (optional, skip GUI):").pack(side=tk.LEFT)
    ttk.Entry(pix_row, textvariable=pixels_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

    def _browse() -> None:
        p = filedialog.askopenfilename(
            parent=win,
            title="Select pre-picked pixel CSV",
            filetypes=[("CSV", "*.csv"), ("DAT", "*.dat"), ("All", "*.*")],
        )
        if p:
            pixels_var.set(p)

    ttk.Button(pix_row, text="Browse…", command=_browse).pack(side=tk.LEFT)

    log_text = tk.Text(frm, height=8, state="disabled", wrap="none")
    log_text.pack(fill=tk.BOTH, expand=True, pady=(8, 4))

    def _log(msg: str) -> None:
        log_text.config(state="normal")
        log_text.insert("end", msg.rstrip() + "\n")
        log_text.see("end")
        log_text.config(state="disabled")

    btns = ttk.Frame(frm)
    btns.pack(fill=tk.X, pady=(4, 0))
    run_btn = ttk.Button(btns, text="Run calibration")
    run_btn.pack(side=tk.LEFT)

    def _open_help() -> None:
        import webbrowser

        help_html = Path(__file__).resolve().parent / "help" / "soccerfield_calib.html"
        if help_html.exists():
            webbrowser.open(help_html.as_uri())
        else:
            messagebox.showinfo(
                "Help",
                "Help file not found. See docs/fifa_workflow.md or run with --help.",
                parent=win,
            )

    ttk.Button(btns, text="Help", command=_open_help).pack(side=tk.LEFT, padx=4)
    ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

    def _on_run() -> None:
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("Calibrate field", "Pick a directory first.", parent=win)
            return
        sam_dir = sam_subdirs[sel[0]]
        try:
            video = _resolve_sam_video_overlay(sam_dir)
        except FileNotFoundError as exc:
            messagebox.showerror("Calibrate field", str(exc), parent=win)
            return
        out_dir = sam_dir / "calib"
        pixel_csv_str = pixels_var.get().strip()
        pixel_csv = Path(pixel_csv_str) if pixel_csv_str else None
        run_btn.config(state="disabled")
        _log(f"Running calibration for {sam_dir.name}…")

        def _worker() -> None:
            try:
                res = run_soccerfield_calib(
                    video=None if pixel_csv else video,
                    pixel_csv=pixel_csv,
                    output_dir=out_dir,
                )
                rms = res["rms_pixels_or_metres"]
                msg = (
                    f"OK — {sam_dir.name}: rms={rms:.4f} m, "
                    f"n_points={res['n_points']}\n"
                    f"dlt2d: {res['dlt2d']}"
                )
                win.after(
                    0,
                    lambda: (
                        _log(msg),
                        run_btn.config(state="normal"),
                        messagebox.showinfo("Calibration done", msg, parent=win),
                    ),
                )
            except Exception as exc:
                err = str(exc)
                win.after(
                    0,
                    lambda: (
                        _log(f"FAILED: {err}"),
                        run_btn.config(state="normal"),
                        messagebox.showerror("Calibration failed", err, parent=win),
                    ),
                )

        threading.Thread(target=_worker, daemon=True).start()

    run_btn.config(command=_on_run)


if __name__ == "__main__":
    main()
