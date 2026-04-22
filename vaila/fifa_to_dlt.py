"""
FIFA camera NPZ (K, R, t, k per frame) -> vailá DLT2D / DLT3D CSV files.

Used after ``fifa baseline`` (or any ``cameras/<stem>.npz`` with the same keys)
so that :mod:`vaila.rec2d` / :mod:`vaila.rec3d` can reconstruct with **per-frame**
DLT parameters (broadcast / moving camera).

© vailá contributors — see repository AUTHORS.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from rich import print

_DIST_WARN_THRESHOLD = 1e-3


def _as_float64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def compute_dlt2d_from_KRt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: N802
    """Homography world plane Z=0 -> image (matches :func:`vaila.rec2d.rec2d` convention).

    ``H = K @ [r0 | r1 | t]`` (3×3), then normalize so ``H[2,2] == 1``.
    Returns 8 coefficients ``[H00,H01,H02,H10,H11,H12,H20,H21]``.
    """
    K = _as_float64(K).reshape(3, 3)
    R = _as_float64(R).reshape(3, 3)
    t = _as_float64(t).reshape(3)
    r0 = R[:, 0]
    r1 = R[:, 1]
    H = (K @ np.column_stack((r0, r1, t))).astype(np.float64)
    scale = H[2, 2]
    if abs(scale) < 1e-12:
        raise ValueError("Homography H[2,2] is ~0; invalid camera pose for Z=0 plane.")
    H /= scale
    return np.array(
        [H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]],
        dtype=np.float64,
    )


def compute_dlt3d_from_KRt(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:  # noqa: N802
    """Full pinhole ``P = K [R|t]``, normalized so ``P[2,3] == 1`` (vailá DLT3D layout).

    Order matches :func:`vaila.dlt3d.calculate_dlt3d_params` / :func:`vaila.rec3d.rec3d_multicam`:
    ``L1..L4`` (row 0), ``L5..L8`` (row 1), ``L9..L11`` (row 2, first three cols).
    """
    K = _as_float64(K).reshape(3, 3)
    R = _as_float64(R).reshape(3, 3)
    t = _as_float64(t).reshape(3, 1)
    Rt = np.hstack((R, t))
    P = (K @ Rt).astype(np.float64)
    scale = P[2, 3]
    if abs(scale) < 1e-12:
        raise ValueError("Projection P[2,3] is ~0; invalid camera matrix / pose.")
    P /= scale
    return np.array(
        [
            P[0, 0],
            P[0, 1],
            P[0, 2],
            P[0, 3],
            P[1, 0],
            P[1, 1],
            P[1, 2],
            P[1, 3],
            P[2, 0],
            P[2, 1],
            P[2, 2],
        ],
        dtype=np.float64,
    )


def _load_camera_npz(path: Path) -> dict[str, np.ndarray]:
    data = dict(np.load(path, allow_pickle=False))
    required = ("K", "R", "t")
    for key in required:
        if key not in data:
            raise KeyError(f"{path}: missing key {key!r}")
    if "k" not in data:
        K0 = np.asarray(data["K"])
        n0 = K0.shape[0] if K0.ndim == 3 else 1
        data["k"] = np.zeros((n0, 2), dtype=np.float64)
    return data


def _broadcast_k(k: np.ndarray, n: int) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    if k.ndim == 1:
        k = np.tile(k.reshape(1, -1), (n, 1))
    if k.shape[0] == 1 and n > 1:
        k = np.tile(k, (n, 1))
    return k


def _maybe_warn_distortion(k: np.ndarray, source: Path) -> None:
    k = np.asarray(k, dtype=np.float64)
    if k.size and float(np.max(np.abs(k))) > _DIST_WARN_THRESHOLD:
        warnings.warn(
            f"{source}: |k|_max > {_DIST_WARN_THRESHOLD}; undistort pixel CSVs before "
            f"running rec2d/rec3d, e.g. `python -m vaila.fifa_to_dlt --undistort-pixels-dir ...`.",
            stacklevel=2,
        )


def convert_cameras_npz_to_dlt(
    npz_path: Path,
    out_dir: Path,
    *,
    mode: str = "both",
) -> dict[str, Path | None]:
    """Write ``<stem>.dlt2d`` and/or ``<stem>.dlt3d`` next to ``out_dir`` (flat).

    Returns mapping with keys ``dlt2d``, ``dlt3d`` and paths or None if skipped.
    """
    npz_path = Path(npz_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = npz_path.stem
    cam = _load_camera_npz(npz_path)
    K = np.asarray(cam["K"], dtype=np.float64)
    R = np.asarray(cam["R"], dtype=np.float64)
    t = np.asarray(cam["t"], dtype=np.float64)

    if K.ndim == 2:
        K = K.reshape(1, 3, 3)
        R = R.reshape(1, 3, 3)
        t = t.reshape(1, 3)

    n = K.shape[0]
    k = _broadcast_k(cam["k"], n)
    if R.shape[0] != n or t.shape[0] != n:
        raise ValueError(f"{npz_path}: K/R/t frame count mismatch")

    _maybe_warn_distortion(k, npz_path)

    result: dict[str, Path | None] = {"dlt2d": None, "dlt3d": None}

    if mode in ("2d", "both"):
        rows2d = []
        for i in range(n):
            d2 = compute_dlt2d_from_KRt(K[i], R[i], t[i])
            rows2d.append([i, *d2.tolist()])
        cols = ["frame"] + [f"p{j + 1}" for j in range(8)]
        dlt2d_path = out_dir / f"{stem}.dlt2d"
        pd.DataFrame(rows2d, columns=cast(Any, cols)).to_csv(dlt2d_path, index=False)
        result["dlt2d"] = dlt2d_path

    if mode in ("3d", "both"):
        rows3d = []
        for i in range(n):
            d3 = compute_dlt3d_from_KRt(K[i], R[i], t[i])
            rows3d.append([i, *d3.tolist()])
        cols = ["frame"] + [f"p{j + 1}" for j in range(11)]
        dlt3d_path = out_dir / f"{stem}.dlt3d"
        pd.DataFrame(rows3d, columns=cast(Any, cols)).to_csv(dlt3d_path, index=False)
        result["dlt3d"] = dlt3d_path

    return result


def _resolve_npz_stem_for_csv(cameras_dir: Path, csv_path: Path) -> str | None:
    """Match ``SEQ.npz`` to ``SEQ.csv`` or ``SEQ_sam_points.csv``."""
    stem = csv_path.stem
    if (cameras_dir / f"{stem}.npz").is_file():
        return stem
    suffix = "_sam_points"
    if stem.endswith(suffix):
        base = stem[: -len(suffix)]
        if (cameras_dir / f"{base}.npz").is_file():
            return base
    return None


def undistort_pixel_csv(
    pixel_csv: Path,
    cameras_npz: Path,
    out_csv: Path,
) -> None:
    """Undistort ``p*_x`` / ``p*_y`` columns using per-frame ``K`` and ``k``."""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("undistort_pixel_csv requires OpenCV (cv2).") from e

    cam = _load_camera_npz(cameras_npz)
    K = np.asarray(cam["K"], dtype=np.float64)
    k = np.asarray(cam["k"], dtype=np.float64)

    if K.ndim == 2:
        K = K.reshape(1, 3, 3)
    n_fr = K.shape[0]
    k = _broadcast_k(k, n_fr)

    df = pd.read_csv(pixel_csv)
    frame_col = "frame" if "frame" in df.columns else "Frame"
    if frame_col not in df.columns:
        raise ValueError(f"{pixel_csv}: need a 'frame' or 'Frame' column")

    out = df.copy()
    xy_cols = [
        (c, c.replace("_x", "_y"))
        for c in df.columns
        if c.startswith("p") and c.endswith("_x") and c.replace("_x", "_y") in df.columns
    ]

    for idx, row in df.iterrows():
        fi = int(row[frame_col])
        if fi < 0 or fi >= K.shape[0]:
            continue
        Ki = K[fi]
        dist = k[fi].reshape(-1, 1).astype(np.float64)

        for cx, cy in xy_cols:
            xv, yv = row[cx], row[cy]
            if pd.isna(xv) or pd.isna(yv):
                continue
            pts = np.array([[[float(xv), float(yv)]]], dtype=np.float64)
            und = cv2.undistortPoints(pts, Ki, dist, P=Ki)
            out.loc[idx, cx] = float(und[0, 0, 0])
            out.loc[idx, cy] = float(und[0, 0, 1])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def _collect_npz_paths(inp: Path) -> list[Path]:
    inp = inp.resolve()
    if inp.is_file():
        if inp.suffix.lower() != ".npz":
            raise ValueError(f"Expected .npz file, got {inp}")
        return [inp]
    if inp.is_dir():
        paths = sorted(inp.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz files under {inp}")
        return paths
    raise FileNotFoundError(inp)


def run_cli(
    *,
    input_path: Path,
    output_dir: Path,
    mode: str = "both",
    undistort_pixels_dir: Path | None = None,
) -> int:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for npz in _collect_npz_paths(input_path):
        paths = convert_cameras_npz_to_dlt(npz, output_dir, mode=mode)
        p2 = paths["dlt2d"].name if paths["dlt2d"] else "-"
        p3 = paths["dlt3d"].name if paths["dlt3d"] else "-"
        print(f"[green]OK[/green] {npz.name} -> dlt2d={p2} dlt3d={p3}")

    if undistort_pixels_dir is not None:
        try:
            import cv2  # noqa: F401, PLC0415 — optional dependency
        except ImportError:
            print("[red]OpenCV (cv2) required for --undistort-pixels-dir[/red]")
            return 2
        pix_dir = Path(undistort_pixels_dir).resolve()
        cam_dir = (
            Path(input_path).resolve() if Path(input_path).is_dir() else Path(input_path).parent
        )
        if not cam_dir.is_dir():
            cam_dir = pix_dir
        out_ud = output_dir / "pixels_undistorted"
        out_ud.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(pix_dir.glob("*.csv")):
            stem = _resolve_npz_stem_for_csv(cam_dir, csv_path)
            if stem is None:
                print(f"[yellow]skip[/yellow] {csv_path.name}: no matching {cam_dir}/*.npz")
                continue
            npz_path = cam_dir / f"{stem}.npz"
            target = out_ud / csv_path.name
            undistort_pixel_csv(csv_path, npz_path, target)
            print(f"[green]undistort[/green] {csv_path.name} -> {target}")
    return 0


def run_gui_flow() -> None:
    """Tk dialogs: cameras folder -> output folder -> optional pixels folder."""
    from tkinter import Tk, filedialog, messagebox

    root = Tk()
    root.withdraw()
    cam_dir = filedialog.askdirectory(title="Select folder with cameras/*.npz")
    if not cam_dir:
        root.destroy()
        return
    out_dir = filedialog.askdirectory(title="Select output folder for .dlt2d / .dlt3d")
    if not out_dir:
        root.destroy()
        return
    ud = messagebox.askyesno(
        "Undistort pixels?",
        "Also undistort all *.csv in a folder (optional)?\n\nIf No, only DLT export runs.",
    )
    pix_dir = None
    if ud:
        pix_dir = filedialog.askdirectory(title="Select folder with pixel CSVs (*.csv)")
        if not pix_dir:
            pix_dir = None
    root.destroy()

    try:
        rc = run_cli(
            input_path=Path(cam_dir),
            output_dir=Path(out_dir),
            mode="both",
            undistort_pixels_dir=Path(pix_dir) if pix_dir else None,
        )
    except Exception as e:
        messagebox.showerror("fifa_to_dlt", str(e))
        return
    if rc == 0:
        messagebox.showinfo("fifa_to_dlt", f"Done.\nOutput: {out_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export per-frame .dlt2d / .dlt3d from FIFA cameras/*.npz (K,R,t,k)."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="One cameras_<seq>.npz or a directory of *.npz",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for DLT CSV files",
    )
    p.add_argument(
        "--mode",
        choices=("2d", "3d", "both"),
        default="both",
        help="Which DLT files to write (default: both)",
    )
    p.add_argument(
        "--undistort-pixels-dir",
        type=Path,
        default=None,
        help="Optional folder of *.csv; writes pixels_undistorted/ under --output",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    try:
        raise SystemExit(
            run_cli(
                input_path=args.input,
                output_dir=args.output,
                mode=args.mode,
                undistort_pixels_dir=args.undistort_pixels_dir,
            )
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[red]{e}[/red]")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
