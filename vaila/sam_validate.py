"""Validate SAM 3 batch outputs against FIFA Skeletal Tracking Light GT.

Three metrics (all reported by player and overall):

    (a) BBox IoU in pixel space (Hungarian per-frame matching).
    (b) RMSE (m) between SAM foot world-coords (via DLT2D rec2d) and
        FIFA ``skel_2d`` mid-hip world-coords (via the same DLT2D).
    (c) RMSE (m) between SAM foot world-coords and FIFA ``skel_3d``
        mid-hip in world meters (drop Z).

Inputs
------
- ``--sam-dir``   per-video SAM3 output directory (must contain
                  ``sam_frames_meta.csv`` and ``sam_points.csv``; the
                  latter is built by :mod:`vaila.sam_postprocess`).
- ``--fifa-data`` FIFA dataset root (with ``boxes/``, ``skel_2d/``,
                  ``skel_3d/``, ``cameras/``).
- ``--match-stem`` Stem identifying both the FIFA arrays and the SAM
                  output directory (e.g. ``ARG_CRO_000737``).
- ``--dlt2d``     ``.dlt2d`` calibration file from
                  :mod:`vaila.soccerfield_calib`.
- ``--out``       Output directory (defaults to ``sam_dir/validate``).

Outputs
-------
- ``validation_summary.csv``  flat metric/scope/value table.
- ``validation_report.html``  human-friendly summary with embedded plots.
- ``iou_per_frame.png`` and ``rmse_hist.png`` (also embedded inline).

Author: Paulo R. P. Santiago - vaila project
Created: 19 April 2026
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import base64
import io
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .rec2d import rec2d
    from .sam_postprocess import discover_sam_run, frame_size, read_sam_meta
except ImportError:
    from rec2d import rec2d  # ty: ignore[unresolved-import]
    from sam_postprocess import (  # ty: ignore[unresolved-import]
        discover_sam_run,
        frame_size,
        read_sam_meta,
    )

__all__ = [
    "ValidationResult",
    "load_dlt2d_params",
    "iou_xyxy",
    "associate_ids_hungarian",
    "validate_sam_run",
]

BODY25_MIDHIP_IDX = 8


@dataclass(frozen=True)
class ValidationResult:
    summary_csv: Path
    report_html: Path
    overall_iou: float
    overall_rmse_skel2d_m: float
    overall_rmse_skel3d_m: float
    n_matched_frames: int
    n_id_switches: int


def load_dlt2d_params(dlt_path: Path) -> np.ndarray:
    """Load the 8 DLT2D parameters from a ``*.dlt2d`` file."""
    df = pd.read_csv(dlt_path)
    cols = [c for c in df.columns if c != "frame"]
    if len(cols) != 8:
        raise ValueError(f"{dlt_path} should have 8 parameter columns; got {len(cols)}.")
    return df.iloc[0][cols].to_numpy(dtype=float)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two ``[x_min, y_min, x_max, y_max]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _box_iou_matrix(sam_boxes: np.ndarray, fifa_boxes: np.ndarray) -> np.ndarray:
    """IoU matrix of shape (N_sam, N_fifa)."""
    if sam_boxes.size == 0 or fifa_boxes.size == 0:
        return np.zeros((sam_boxes.shape[0], fifa_boxes.shape[0]), dtype=float)
    n_s = sam_boxes.shape[0]
    n_f = fifa_boxes.shape[0]
    out = np.zeros((n_s, n_f), dtype=float)
    for i in range(n_s):
        for j in range(n_f):
            out[i, j] = iou_xyxy(sam_boxes[i], fifa_boxes[j])
    return out


def associate_ids_hungarian(
    iou_per_frame: dict[int, np.ndarray],
    sam_ids_per_frame: dict[int, list[int]],
    fifa_n_players: int,
    iou_threshold: float = 0.1,
) -> tuple[dict[int, int], int, list[float]]:
    """Run Hungarian matching per frame and return (mapping, switches, ious).

    ``mapping`` is the **majority** id_sam -> id_fifa mapping over the clip.
    ``switches`` counts frames whose per-frame match disagrees with majority.
    ``ious`` is a flat list of all matched IoU values (one per matched pair).
    """
    from scipy.optimize import linear_sum_assignment

    votes: dict[int, Counter[int]] = defaultdict(Counter)
    per_frame_match: dict[int, dict[int, int]] = {}
    matched_ious: list[float] = []

    for frame_idx, iou in iou_per_frame.items():
        sam_ids = sam_ids_per_frame[frame_idx]
        if iou.size == 0 or len(sam_ids) == 0:
            per_frame_match[frame_idx] = {}
            continue
        cost = -iou
        row_ind, col_ind = linear_sum_assignment(cost)
        frame_match: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind, strict=False):
            if iou[r, c] >= iou_threshold:
                sid = sam_ids[r]
                frame_match[sid] = int(c)
                votes[sid][int(c)] += 1
                matched_ious.append(float(iou[r, c]))
        per_frame_match[frame_idx] = frame_match

    mapping: dict[int, int] = {}
    for sid, vote in votes.items():
        majority_fid = vote.most_common(1)[0][0]
        if majority_fid >= fifa_n_players:
            continue
        mapping[sid] = majority_fid

    switches = 0
    for _frame_idx, frame_match in per_frame_match.items():
        for sid, fid in frame_match.items():
            if sid in mapping and mapping[sid] != fid:
                switches += 1

    return mapping, switches, matched_ious


def _sam_boxes_xyxy(
    df: pd.DataFrame, oids: list[int], width: int, height: int
) -> dict[int, dict[int, np.ndarray]]:
    """Return {frame_idx: {obj_id: [x1,y1,x2,y2]}} in pixel space."""
    out: dict[int, dict[int, np.ndarray]] = {}
    for _, row in df.iterrows():
        f = int(row["frame"])
        per_id: dict[int, np.ndarray] = {}
        for oid in oids:
            bx = row.get(f"box_x_{oid}", float("nan"))
            by = row.get(f"box_y_{oid}", float("nan"))
            bw = row.get(f"box_w_{oid}", float("nan"))
            bh = row.get(f"box_h_{oid}", float("nan"))
            if not all(np.isfinite([bx, by, bw, bh])):
                continue
            x1 = float(bx) * width
            y1 = float(by) * height
            x2 = (float(bx) + float(bw)) * width
            y2 = (float(by) + float(bh)) * height
            per_id[oid] = np.array([x1, y1, x2, y2], dtype=float)
        out[f] = per_id
    return out


def _read_sam_points(sam_dir: Path) -> tuple[pd.DataFrame, dict[int, int]]:
    points_csv = sam_dir / "sam_points.csv"
    id_map_csv = sam_dir / "sam_id_map.csv"
    if not points_csv.is_file():
        raise FileNotFoundError(
            f"sam_points.csv missing in {sam_dir}. Run vaila/sam_postprocess.py first."
        )
    if not id_map_csv.is_file():
        raise FileNotFoundError(f"sam_id_map.csv missing in {sam_dir}.")
    pts = pd.read_csv(points_csv)
    id_map_df = pd.read_csv(id_map_csv)
    pn_to_oid = {int(r["pN"]): int(r["obj_id"]) for _, r in id_map_df.iterrows()}
    return pts, pn_to_oid


def _foot_array_for_pn(points_df: pd.DataFrame, pn: int) -> np.ndarray:
    cols = [f"p{pn}_x", f"p{pn}_y"]
    if not all(c in points_df.columns for c in cols):
        raise KeyError(f"sam_points.csv missing {cols}")
    arr = points_df[cols].to_numpy(dtype=float)
    return arr


def _save_plot(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_html(
    *,
    stem: str,
    n_frames: int,
    overall_iou: float,
    rmse_skel2d_m: float,
    rmse_skel3d_m: float,
    switches: int,
    per_player_rows: list[dict],
    iou_png_b64: str,
    rmse_png_b64: str,
    summary_csv_name: str,
) -> str:
    rows_html = "".join(
        f"<tr><td>{r['pN']}</td><td>{r['fifa_id']}</td>"
        f"<td>{r['n_frames']}</td><td>{r['mIoU']:.4f}</td>"
        f"<td>{r['rmse_skel2d_m']:.4f}</td><td>{r['rmse_skel3d_m']:.4f}</td></tr>"
        for r in per_player_rows
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SAM3 validation - {stem}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #eef; }}
img {{ max-width: 900px; }}
</style></head>
<body>
<h1>SAM3 validation - {stem}</h1>
<p>Frames analyzed: <b>{n_frames}</b> &nbsp; Matched players: <b>{len(per_player_rows)}</b>
&nbsp; ID switches: <b>{switches}</b></p>
<table>
<tr><th colspan="2">Overall</th></tr>
<tr><td>Mean IoU (pixel)</td><td>{overall_iou:.4f}</td></tr>
<tr><td>RMSE rec2d vs skel_2d (m)</td><td>{rmse_skel2d_m:.4f}</td></tr>
<tr><td>RMSE rec2d vs skel_3d (m)</td><td>{rmse_skel3d_m:.4f}</td></tr>
</table>
<h2>Per player</h2>
<table>
<tr><th>pN</th><th>fifa_id</th><th>n_frames</th><th>mIoU</th>
<th>RMSE 2D (m)</th><th>RMSE 3D (m)</th></tr>
{rows_html}
</table>
<h2>IoU per frame</h2>
<img src="data:image/png;base64,{iou_png_b64}"/>
<h2>RMSE histograms</h2>
<img src="data:image/png;base64,{rmse_png_b64}"/>
<p>See <a href="{summary_csv_name}">validation_summary.csv</a> for the
flat metric/scope table.</p>
</body></html>
"""


def validate_sam_run(
    *,
    sam_dir: Path,
    fifa_data: Path,
    match_stem: str,
    dlt2d: Path,
    out_dir: Path | None = None,
    iou_threshold: float = 0.1,
) -> ValidationResult:
    sam_dir = Path(sam_dir).resolve()
    fifa_data = Path(fifa_data).resolve()
    out_dir = Path(out_dir).resolve() if out_dir is not None else sam_dir / "validate"
    out_dir.mkdir(parents=True, exist_ok=True)

    art = discover_sam_run(sam_dir)
    width, height = frame_size(art)
    sam_meta_df, oids = read_sam_meta(sam_dir)
    sam_boxes_per_frame = _sam_boxes_xyxy(sam_meta_df, oids, width, height)

    points_df, pn_to_oid = _read_sam_points(sam_dir)
    oid_to_pn = {oid: pn for pn, oid in pn_to_oid.items()}

    fifa_boxes = np.load(fifa_data / "boxes" / f"{match_stem}.npy")
    fifa_skel2d = np.load(fifa_data / "skel_2d" / f"{match_stem}.npy")
    fifa_skel3d = np.load(fifa_data / "skel_3d" / f"{match_stem}.npy")

    if fifa_boxes.ndim != 3 or fifa_boxes.shape[2] != 4:
        raise ValueError(f"Unexpected boxes shape: {fifa_boxes.shape} (want (T,P,4))")
    if fifa_skel2d.ndim != 4 or fifa_skel2d.shape[3] != 2:
        raise ValueError(f"Unexpected skel_2d shape: {fifa_skel2d.shape} (want (T,P,J,2))")
    if fifa_skel3d.ndim != 4 or fifa_skel3d.shape[3] != 3:
        raise ValueError(f"Unexpected skel_3d shape: {fifa_skel3d.shape} (want (T,P,J,3))")

    T_fifa, P_fifa, _ = fifa_boxes.shape
    T_sam = int(sam_meta_df["frame"].max()) + 1 if len(sam_meta_df) else 0
    T = min(T_sam, T_fifa)
    if T == 0:
        raise ValueError("No frames available to validate.")

    iou_per_frame: dict[int, np.ndarray] = {}
    sam_ids_per_frame: dict[int, list[int]] = {}
    for f in range(T):
        per_id = sam_boxes_per_frame.get(f, {})
        sam_ids = sorted(per_id.keys())
        if not sam_ids:
            iou_per_frame[f] = np.zeros((0, P_fifa), dtype=float)
            sam_ids_per_frame[f] = []
            continue
        sam_arr = np.stack([per_id[oid] for oid in sam_ids], axis=0)
        iou_mat = _box_iou_matrix(sam_arr, fifa_boxes[f])
        iou_per_frame[f] = iou_mat
        sam_ids_per_frame[f] = sam_ids

    mapping, switches, matched_ious = associate_ids_hungarian(
        iou_per_frame, sam_ids_per_frame, P_fifa, iou_threshold=iou_threshold
    )
    overall_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    params = load_dlt2d_params(dlt2d)
    pn_for_oid = oid_to_pn

    sq_err_2d_per_pn: dict[int, list[float]] = defaultdict(list)
    sq_err_3d_per_pn: dict[int, list[float]] = defaultdict(list)
    n_frames_per_pn: dict[int, int] = defaultdict(int)
    iou_per_pn: dict[int, list[float]] = defaultdict(list)

    iou_per_frame_overall: list[float] = []

    for f in range(T):
        iou_mat = iou_per_frame[f]
        sam_ids = sam_ids_per_frame[f]
        sam_boxes_f = sam_boxes_per_frame.get(f, {})
        if not sam_ids:
            iou_per_frame_overall.append(0.0)
            continue
        frame_ious: list[float] = []
        sam_pixels = []
        fifa_2d_pixels = []
        fifa_3d_xy = []
        keys = []
        for sid in sam_ids:
            if sid not in mapping:
                continue
            fid = mapping[sid]
            row_idx = sam_ids.index(sid)
            iou_val = float(iou_mat[row_idx, fid]) if fid < iou_mat.shape[1] else 0.0
            if iou_val < iou_threshold:
                continue
            pn = pn_for_oid.get(sid)
            if pn is None:
                continue
            row_pts = points_df[points_df["frame"] == f]
            if row_pts.empty:
                continue
            fx = row_pts.iloc[0].get(f"p{pn}_x", float("nan"))
            fy = row_pts.iloc[0].get(f"p{pn}_y", float("nan"))
            if not (np.isfinite(fx) and np.isfinite(fy)):
                continue
            box = sam_boxes_f.get(sid)
            if box is None:
                continue
            sam_pixels.append([float(fx), float(fy)])

            mid_2d = fifa_skel2d[f, fid, BODY25_MIDHIP_IDX, :]
            mid_3d = fifa_skel3d[f, fid, BODY25_MIDHIP_IDX, :2]
            fifa_2d_pixels.append(mid_2d.astype(float))
            fifa_3d_xy.append(mid_3d.astype(float))
            keys.append((sid, pn))
            iou_per_pn[pn].append(iou_val)
            frame_ious.append(iou_val)
            n_frames_per_pn[pn] += 1

        iou_per_frame_overall.append(float(np.mean(frame_ious)) if frame_ious else 0.0)

        if not keys:
            continue
        sam_world = rec2d(params, np.asarray(sam_pixels, dtype=float))
        fifa_2d_world = rec2d(params, np.asarray(fifa_2d_pixels, dtype=float))
        fifa_3d_world = np.asarray(fifa_3d_xy, dtype=float)

        for k_idx, (_, pn) in enumerate(keys):
            d2 = sam_world[k_idx] - fifa_2d_world[k_idx]
            d3 = sam_world[k_idx] - fifa_3d_world[k_idx]
            sq_err_2d_per_pn[pn].append(float(d2 @ d2))
            sq_err_3d_per_pn[pn].append(float(d3 @ d3))

    def _rmse(d: dict[int, list[float]]) -> tuple[float, dict[int, float]]:
        per = {pn: float(np.sqrt(np.mean(v))) for pn, v in d.items() if v}
        flat = [x for v in d.values() for x in v]
        overall = float(np.sqrt(np.mean(flat))) if flat else float("nan")
        return overall, per

    overall_rmse_2d, per_pn_rmse_2d = _rmse(sq_err_2d_per_pn)
    overall_rmse_3d, per_pn_rmse_3d = _rmse(sq_err_3d_per_pn)

    summary_rows: list[dict] = [
        {"metric": "overall_iou", "scope": "all", "value": overall_iou},
        {"metric": "rmse_skel2d_m", "scope": "all", "value": overall_rmse_2d},
        {"metric": "rmse_skel3d_m", "scope": "all", "value": overall_rmse_3d},
        {"metric": "id_switches", "scope": "all", "value": float(switches)},
        {"metric": "n_matched_frames", "scope": "all", "value": float(len(matched_ious))},
    ]
    per_player_rows: list[dict] = []
    for pn in sorted(
        set(list(per_pn_rmse_2d.keys()) + list(per_pn_rmse_3d.keys()) + list(iou_per_pn.keys()))
    ):
        miou = float(np.mean(iou_per_pn[pn])) if iou_per_pn[pn] else float("nan")
        r2 = per_pn_rmse_2d.get(pn, float("nan"))
        r3 = per_pn_rmse_3d.get(pn, float("nan"))
        oid = pn_to_oid.get(pn, -1)
        fifa_id = mapping.get(oid, -1)
        per_player_rows.append(
            {
                "pN": pn,
                "obj_id": oid,
                "fifa_id": fifa_id,
                "n_frames": n_frames_per_pn[pn],
                "mIoU": miou,
                "rmse_skel2d_m": r2,
                "rmse_skel3d_m": r3,
            }
        )
        summary_rows.append({"metric": "mIoU", "scope": f"p{pn}", "value": miou})
        summary_rows.append({"metric": "rmse_skel2d_m", "scope": f"p{pn}", "value": r2})
        summary_rows.append({"metric": "rmse_skel3d_m", "scope": f"p{pn}", "value": r3})

    summary_csv = out_dir / "validation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(iou_per_frame_overall, lw=0.8)
    ax.axhline(overall_iou, color="r", lw=0.5, ls="--", label=f"mean = {overall_iou:.3f}")
    ax.set_xlabel("frame")
    ax.set_ylabel("mean IoU (matched)")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    iou_png_path = out_dir / "iou_per_frame.png"
    fig.savefig(iou_png_path, dpi=120, bbox_inches="tight")
    iou_b64 = _save_plot(fig)
    plt.close(fig)

    flat_2d = [x for v in sq_err_2d_per_pn.values() for x in v]
    flat_3d = [x for v in sq_err_3d_per_pn.values() for x in v]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    if flat_2d:
        axes[0].hist(np.sqrt(flat_2d), bins=30)
        axes[0].set_title(f"per-frame error vs skel_2d (m); RMSE={overall_rmse_2d:.3f}")
    if flat_3d:
        axes[1].hist(np.sqrt(flat_3d), bins=30)
        axes[1].set_title(f"per-frame error vs skel_3d (m); RMSE={overall_rmse_3d:.3f}")
    for ax in axes:
        ax.set_xlabel("error (m)")
        ax.grid(alpha=0.3)
    rmse_png_path = out_dir / "rmse_hist.png"
    fig.savefig(rmse_png_path, dpi=120, bbox_inches="tight")
    rmse_b64 = _save_plot(fig)
    plt.close(fig)

    html = _render_html(
        stem=match_stem,
        n_frames=T,
        overall_iou=overall_iou,
        rmse_skel2d_m=overall_rmse_2d,
        rmse_skel3d_m=overall_rmse_3d,
        switches=switches,
        per_player_rows=per_player_rows,
        iou_png_b64=iou_b64,
        rmse_png_b64=rmse_b64,
        summary_csv_name=summary_csv.name,
    )
    report_html = out_dir / "validation_report.html"
    report_html.write_text(html, encoding="utf-8")

    print(
        f"[sam_validate] {match_stem}: T={T}, mIoU={overall_iou:.4f}, "
        f"RMSE_2D={overall_rmse_2d:.4f} m, RMSE_3D={overall_rmse_3d:.4f} m, "
        f"switches={switches}, report={report_html}"
    )

    return ValidationResult(
        summary_csv=summary_csv,
        report_html=report_html,
        overall_iou=overall_iou,
        overall_rmse_skel2d_m=overall_rmse_2d,
        overall_rmse_skel3d_m=overall_rmse_3d,
        n_matched_frames=len(matched_ious),
        n_id_switches=switches,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate SAM3 outputs vs FIFA GT.")
    p.add_argument("--sam-dir", type=Path, required=True)
    p.add_argument("--fifa-data", type=Path, required=True)
    p.add_argument("--match-stem", type=str, required=True)
    p.add_argument("--dlt2d", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--iou-threshold", type=float, default=0.1)
    args = p.parse_args(argv)

    validate_sam_run(
        sam_dir=args.sam_dir,
        fifa_data=args.fifa_data,
        match_stem=args.match_stem,
        dlt2d=args.dlt2d,
        out_dir=args.out,
        iou_threshold=args.iou_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
