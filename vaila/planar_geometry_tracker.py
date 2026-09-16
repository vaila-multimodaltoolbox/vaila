"""
================================================================================
Script: planar_geometry_tracker.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.4.3
Created: 14 September 2026
Last Updated: 16 September 2026
================================================================================
Description:
    Standalone planar-geometry tracker / gap-filler / extrapolator.

    Given a 2D pixel-space marker CSV from ``getpixelvideo.py`` and a metric
    target-geometry TOML (EVA tatame, soccer pitch, …), each frame:

    1. **Locks measured pixels** — digitized markers are never overwritten.
    2. **Topology fill** — missing midpoints / corners from line intersections
       and midpoints on ``[topology].lines``.
    3. **Temporal gap-fill** — linear interpolation between measured anchors
       for still-missing markers (avoids DLT slam-to-interior jumps).
    4. **DLT only for leftovers** — per-frame DLT2D projects remaining gaps;
       never replaces measured / topology / gap-fill points.

    Imputed wide CSV is always a *new* file (``*_imputed.csv``); the input
    measurements CSV is never overwritten. Optional HTML shows pixel +
    world (rec2d) geometry frame by frame.

    Session helpers for the Geo Homog wizard: rectangle TOML, scale profile
    bounding box, remap marker CSV columns, write session TOML.

    100% standalone: never mutates ``getpixelvideo.py`` runtime state. The
    getpixelvideo "Geo Homog" button shells out via subprocess. A dedicated
    **Planar Geo** button also lives under Frame C → Video and Image.

Usage:
    # GUI (Video and Image → Planar Geo, or no CLI args)
    uv run python -m vaila.planar_geometry_tracker

    uv run python -m vaila.planar_geometry_tracker \\
        --config vaila/models/planar_targets/tatame_1x1m.toml \\
        --measurements-csv path/to/markers.csv \\
        --video-path path/to/video.mp4 \\
        --output-dir ./output/tatame_run_01 \\
        --ransac-thresh 3.0 \\
        --extended-canvas \\
        --debug-viz

Outputs (written to --output-dir):
    <stem>_imputed.csv           - wide CSV (frame,p0_x,p0_y,...), NaN-free,
                                    loadable straight back via getpixelvideo.py
    dense_projected_points_long.csv
                                  - long/relational per-point-per-frame table
    homographies.npz              - H, H_inv, frame_ids arrays
    target_calibration.ref3d      - target metric points, repo .ref3d CSV
                                     convention (see vaila/drawsportsfields.py)
    debug_projected_wireframe.mp4 - only with --debug-viz
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]

# ------------------------------------------------------------------------- #
# Geometry configuration model
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TargetPoint:
    """One metric control point of the target geometry (Z = 0 plane)."""

    point_id: int
    name: str
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class TargetCircle:
    name: str
    center_point: int
    radius: float
    samples: int = 64


@dataclass(frozen=True)
class TargetArc:
    name: str
    center_point: int
    radius: float
    angle_start_deg: float
    angle_end_deg: float
    samples: int = 32


@dataclass(frozen=True)
class TargetGeometry:
    """Parsed ``[target]``/``[points]``/``[topology]``/circles/arcs TOML profile."""

    name: str
    description: str
    target_type: str
    points: dict[int, TargetPoint]
    perimeter: list[int] = field(default_factory=list)
    lines: list[tuple[int, int]] = field(default_factory=list)
    circles: list[TargetCircle] = field(default_factory=list)
    arcs: list[TargetArc] = field(default_factory=list)

    def world_xy(self, point_id: int) -> NDArray[np.float64]:
        p = self.points[point_id]
        return np.array([p.x, p.y], dtype=np.float64)

    @property
    def sorted_ids(self) -> list[int]:
        return sorted(self.points)


def load_target_geometry(config_path: str | Path) -> TargetGeometry:
    """Parse a target-geometry TOML profile (see ``vaila/models/planar_targets/``)."""
    config_path = Path(config_path)
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)

    target = data.get("target", {})
    raw_points = data.get("points", {})
    points: dict[int, TargetPoint] = {}
    for key, val in raw_points.items():
        pid = int(key)
        points[pid] = TargetPoint(
            point_id=pid,
            name=str(val.get("name", f"p{pid}")),
            x=float(val["x"]),
            y=float(val["y"]),
            z=float(val.get("z", 0.0)),
        )

    topology = data.get("topology", {})
    perimeter = [int(i) for i in topology.get("perimeter", [])]
    lines = [(int(a), int(b)) for a, b in topology.get("lines", [])]

    circles = [
        TargetCircle(
            name=str(c.get("name", "circle")),
            center_point=int(c["center_point"]),
            radius=float(c["radius"]),
            samples=int(c.get("samples", 64)),
        )
        for c in data.get("circles", [])
    ]
    arcs = [
        TargetArc(
            name=str(a.get("name", "arc")),
            center_point=int(a["center_point"]),
            radius=float(a["radius"]),
            angle_start_deg=float(a["angle_start_deg"]),
            angle_end_deg=float(a["angle_end_deg"]),
            samples=int(a.get("samples", 32)),
        )
        for a in data.get("arcs", [])
    ]

    if not points:
        raise ValueError(f"Target geometry profile has no [points]: {config_path}")

    return TargetGeometry(
        name=str(target.get("name", config_path.stem)),
        description=str(target.get("description", "")),
        target_type=str(target.get("type", "polygon")),
        points=points,
        perimeter=perimeter,
        lines=lines,
        circles=circles,
        arcs=arcs,
    )


def geometry_bounding_box(geometry: TargetGeometry) -> tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` of the target control points."""
    xs = [p.x for p in geometry.points.values()]
    ys = [p.y for p in geometry.points.values()]
    return min(xs), min(ys), max(xs), max(ys)


def make_rectangle_geometry(
    width: float,
    height: float,
    *,
    name: str = "rectangle",
    description: str = "Axis-aligned rectangle on Z=0",
) -> TargetGeometry:
    """Build a 4-corner rectangle (SW, SE, NE, NW) with perimeter topology."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Rectangle width/height must be positive, got {width} x {height}")
    corners = {
        0: TargetPoint(0, "corner_sw", 0.0, 0.0, 0.0),
        1: TargetPoint(1, "corner_se", float(width), 0.0, 0.0),
        2: TargetPoint(2, "corner_ne", float(width), float(height), 0.0),
        3: TargetPoint(3, "corner_nw", 0.0, float(height), 0.0),
    }
    return TargetGeometry(
        name=name,
        description=description,
        target_type="polygon",
        points=corners,
        perimeter=[0, 1, 2, 3],
        lines=[(0, 1), (1, 2), (2, 3), (3, 0)],
    )


def scale_target_geometry(
    geometry: TargetGeometry,
    new_width: float,
    new_height: float,
    *,
    name: str | None = None,
    description: str | None = None,
) -> TargetGeometry:
    """Uniformly scale a profile's bounding box to ``new_width`` × ``new_height``.

    Origin (min_x, min_y) is preserved; relative layout of points/circles/arcs
    is kept. Circles/arcs radii scale by the mean of the two axis scales.
    """
    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"Scale width/height must be positive, got {new_width} x {new_height}")
    min_x, min_y, max_x, max_y = geometry_bounding_box(geometry)
    old_w = max_x - min_x
    old_h = max_y - min_y
    if old_w <= 1e-12 or old_h <= 1e-12:
        raise ValueError("Cannot scale a degenerate geometry bounding box")
    sx = new_width / old_w
    sy = new_height / old_h
    s_r = 0.5 * (sx + sy)

    points = {
        pid: TargetPoint(
            point_id=pid,
            name=p.name,
            x=min_x + (p.x - min_x) * sx,
            y=min_y + (p.y - min_y) * sy,
            z=p.z,
        )
        for pid, p in geometry.points.items()
    }
    circles = [
        TargetCircle(
            name=c.name,
            center_point=c.center_point,
            radius=c.radius * s_r,
            samples=c.samples,
        )
        for c in geometry.circles
    ]
    arcs = [
        TargetArc(
            name=a.name,
            center_point=a.center_point,
            radius=a.radius * s_r,
            angle_start_deg=a.angle_start_deg,
            angle_end_deg=a.angle_end_deg,
            samples=a.samples,
        )
        for a in geometry.arcs
    ]
    return TargetGeometry(
        name=name if name is not None else geometry.name,
        description=(
            description
            if description is not None
            else f"{geometry.description} (scaled to {new_width:g}x{new_height:g})"
        ),
        target_type=geometry.target_type,
        points=points,
        perimeter=list(geometry.perimeter),
        lines=list(geometry.lines),
        circles=circles,
        arcs=arcs,
    )


def write_target_geometry_toml(geometry: TargetGeometry, path: str | Path) -> Path:
    """Serialize a :class:`TargetGeometry` to a TOML profile on disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines_out: list[str] = [
        "[target]",
        f'name = "{geometry.name}"',
        f'description = "{geometry.description}"',
        f'type = "{geometry.target_type}"',
        "",
        "[points]",
    ]
    for pid in geometry.sorted_ids:
        p = geometry.points[pid]
        lines_out.append(
            f'{pid} = {{ name = "{p.name}", x = {p.x:.6f}, y = {p.y:.6f}, z = {p.z:.6f} }}'
        )
    lines_out.append("")
    lines_out.append("[topology]")
    if geometry.perimeter:
        peri = ", ".join(str(i) for i in geometry.perimeter)
        lines_out.append(f"perimeter = [{peri}]")
    if geometry.lines:
        pairs = ", ".join(f"[{a}, {b}]" for a, b in geometry.lines)
        lines_out.append(f"lines = [{pairs}]")
    for circle in geometry.circles:
        lines_out.extend(
            [
                "",
                "[[circles]]",
                f'name = "{circle.name}"',
                f"center_point = {circle.center_point}",
                f"radius = {circle.radius:.6f}",
                f"samples = {circle.samples}",
            ]
        )
    for arc in geometry.arcs:
        lines_out.extend(
            [
                "",
                "[[arcs]]",
                f'name = "{arc.name}"',
                f"center_point = {arc.center_point}",
                f"radius = {arc.radius:.6f}",
                f"angle_start_deg = {arc.angle_start_deg:.6f}",
                f"angle_end_deg = {arc.angle_end_deg:.6f}",
                f"samples = {arc.samples}",
            ]
        )
    path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    return path


def write_rectangle_toml(
    path: str | Path,
    width: float,
    height: float,
    *,
    name: str = "rectangle",
) -> TargetGeometry:
    """Write a 4-corner rectangle profile and return the parsed geometry."""
    geometry = make_rectangle_geometry(width, height, name=name)
    write_target_geometry_toml(geometry, path)
    return geometry


def parse_marker_geom_mapping(text: str) -> dict[int, int]:
    """Parse ``marker_id:geom_id`` pairs from ``0:0,1:2,3:4`` (or whitespace)."""
    mapping: dict[int, int] = {}
    cleaned = text.strip()
    if not cleaned:
        return mapping
    for part in cleaned.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Mapping entry must be marker:geom, got {token!r}")
        left, right = token.split(":", 1)
        mapping[int(left.strip())] = int(right.strip())
    return mapping


def default_identity_mapping(geometry: TargetGeometry, marker_ids: set[int]) -> dict[int, int]:
    """Identity map for geometry point ids that also exist as marker slots."""
    return {pid: pid for pid in geometry.sorted_ids if pid in marker_ids}


def remap_measurements_csv(
    src_csv: str | Path,
    mapping: dict[int, int],
    dst_csv: str | Path,
) -> Path:
    """Copy mapped marker columns into ``p{geom_id}_x/y`` on a new wide CSV.

    ``mapping`` keys are source marker column indices; values are destination
    geometry point ids (must match the session TOML ``[points]`` keys).
    """
    if len(mapping) < 4:
        raise ValueError(f"Need at least 4 mapped marker pairs, got {len(mapping)}")
    src_csv = Path(src_csv)
    dst_csv = Path(dst_csv)
    df = pd.read_csv(src_csv)
    if "frame" not in df.columns:
        raise ValueError(f"Source measurements CSV lacks 'frame' column: {src_csv}")

    geom_ids = sorted(set(mapping.values()))
    out: dict[str, Any] = {"frame": df["frame"].to_numpy()}
    for marker_id, geom_id in mapping.items():
        sx, sy = f"p{marker_id}_x", f"p{marker_id}_y"
        dx, dy = f"p{geom_id}_x", f"p{geom_id}_y"
        if sx not in df.columns or sy not in df.columns:
            raise ValueError(f"Source CSV missing marker columns {sx}/{sy}")
        out[dx] = df[sx].to_numpy()
        out[dy] = df[sy].to_numpy()

    # Stable column order: frame, then ascending geom ids.
    ordered_cols = ["frame"]
    for gid in geom_ids:
        ordered_cols.extend([f"p{gid}_x", f"p{gid}_y"])
    out_df = pd.DataFrame({c: out[c] for c in ordered_cols})
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst_csv, index=False)
    return dst_csv


def load_homographies_npz(npz_path: str | Path) -> dict[int, NDArray[np.float64]]:
    """Load ``homographies.npz`` into ``{frame_id: 3x3 H}``."""
    data = np.load(Path(npz_path))
    frame_ids = np.asarray(data["frame_ids"]).astype(np.int64)
    h_stack = np.asarray(data["H"], dtype=np.float64)
    return {int(fid): h_stack[i] for i, fid in enumerate(frame_ids)}


def project_wireframe_segments(
    geometry: TargetGeometry,
    homography: NDArray[np.float64],
) -> list[list[tuple[float, float]]]:
    """Project topology lines/circles/arcs through H (legacy overlay helper)."""
    segments: list[list[tuple[float, float]]] = []
    for a, b in geometry.lines:
        pts = project_points(homography, np.array([geometry.world_xy(a), geometry.world_xy(b)]))
        segments.append(
            [(float(pts[0, 0]), float(pts[0, 1])), (float(pts[1, 0]), float(pts[1, 1]))]
        )
    for circle in geometry.circles:
        world = sample_circle_world(circle, geometry)
        pts = project_points(homography, world)
        segments.append([(float(x), float(y)) for x, y in pts])
    for arc in geometry.arcs:
        world = sample_arc_world(arc, geometry)
        pts = project_points(homography, world)
        segments.append([(float(x), float(y)) for x, y in pts])
    return segments


def wireframe_segments_from_pixels(
    geometry: TargetGeometry,
    resolved: dict[int, tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Build wireframe polylines from resolved pixel coordinates (preferred)."""
    segments: list[list[tuple[float, float]]] = []
    for a, b in geometry.lines:
        if a not in resolved or b not in resolved:
            continue
        ua, va = resolved[a]
        ub, vb = resolved[b]
        if not (np.isfinite(ua) and np.isfinite(va) and np.isfinite(ub) and np.isfinite(vb)):
            continue
        segments.append([(float(ua), float(va)), (float(ub), float(vb))])
    return segments


# ------------------------------------------------------------------------- #
# Measurements CSV ingestion (wide primary; long also supported)
# ------------------------------------------------------------------------- #


def load_measurements_csv(
    csv_path: str | Path,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Load a marker CSV into ``{frame: {point_id: (u, v)}}``.

    Supports the wide format actually written by ``getpixelvideo.py``
    (``frame, p0_x, p0_y, p1_x, p1_y, ...``) as the primary path, and the
    long/relational format (``frame, point_id, u, v``) as a secondary path.
    Missing/NaN coordinates are simply omitted from the per-frame dict.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    columns = set(df.columns)

    out: dict[int, dict[int, tuple[float, float]]] = {}

    if {"point_id", "u", "v"}.issubset(columns):
        # Long format.
        for row_d in df.to_dict("records"):
            frame = int(row_d["frame"])
            pid = int(row_d["point_id"])
            u = float(row_d["u"])
            v = float(row_d["v"])
            if np.isnan(u) or np.isnan(v):
                continue
            out.setdefault(frame, {})[pid] = (u, v)
        return out

    # Wide format: frame, p{i}_x, p{i}_y, ...
    if "frame" not in columns:
        raise ValueError(
            f"Measurements CSV {csv_path} has neither long (point_id,u,v) nor "
            "wide (frame,pN_x,pN_y,...) columns."
        )
    point_ids: set[int] = set()
    for col in df.columns:
        if col.startswith("p") and col.endswith("_x"):
            try:
                point_ids.add(int(col[1:-2]))
            except ValueError:
                continue

    for row_d in df.to_dict("records"):
        frame = int(row_d["frame"])
        per_frame: dict[int, tuple[float, float]] = {}
        for pid in point_ids:
            xk, yk = f"p{pid}_x", f"p{pid}_y"
            if xk not in row_d or yk not in row_d:
                continue
            u, v = row_d[xk], row_d[yk]
            if u is None or v is None:
                continue
            try:
                uf, vf = float(u), float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(uf) or np.isnan(vf):
                continue
            per_frame[pid] = (uf, vf)
        out[frame] = per_frame
    return out


# ------------------------------------------------------------------------- #
# Step 1: Non-collinearity checks
# ------------------------------------------------------------------------- #


def is_collinear_triplet(
    p1: NDArray[np.float64], p2: NDArray[np.float64], p3: NDArray[np.float64], tol: float = 1e-4
) -> bool:
    """Computes twice the triangle area to detect collinear points."""
    return (
        0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) < tol
    )


def has_non_collinear_quad(
    valid_indices: list[int], world_points: dict[int, NDArray[np.float64]]
) -> bool:
    """Returns True if there is at least one subset of 4 points containing no collinear triplets."""
    if len(valid_indices) < 4:
        return False
    for c in combinations(valid_indices, 4):
        p0, p1, p2, p3 = (world_points[i] for i in c)
        if not (
            is_collinear_triplet(p0, p1, p2)
            or is_collinear_triplet(p0, p1, p3)
            or is_collinear_triplet(p0, p2, p3)
            or is_collinear_triplet(p1, p2, p3)
        ):
            return True
    return False


# ------------------------------------------------------------------------- #
# Topology index (midpoints, collinear sets, vertex incident edges)
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TopologyIndex:
    """Precomputed metric topology helpers for image-space imputation."""

    # mid_id -> (endpoint_a, endpoint_b) when mid is world-midpoint of AB
    metric_midpoints: dict[int, tuple[int, int]]
    # topology edge (a,b) -> all point ids collinear on that world segment
    line_point_sets: dict[tuple[int, int], list[int]]
    # vertex_id -> list of (other_endpoint_on_incident_edge,)
    # used with line_point_sets to find two image lines to intersect
    vertex_incident: dict[int, list[tuple[int, int]]]


def _point_on_segment_param(
    p: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64], tol: float = 1e-6
) -> float | None:
    """Return t in [0,1] if p lies on segment AB in world space, else None."""
    ab = b - a
    length2 = float(ab @ ab)
    if length2 < tol * tol:
        return None
    t = float((p - a) @ ab) / length2
    if t < -tol or t > 1.0 + tol:
        return None
    closest = a + t * ab
    if float(np.linalg.norm(p - closest)) > tol:
        return None
    return float(np.clip(t, 0.0, 1.0))


def build_topology_index(geometry: TargetGeometry, *, mid_tol: float = 1e-4) -> TopologyIndex:
    """Derive midpoint / collinear / incidence tables from metric points + lines."""
    world = {pid: geometry.world_xy(pid) for pid in geometry.sorted_ids}
    metric_midpoints: dict[int, tuple[int, int]] = {}
    line_point_sets: dict[tuple[int, int], list[int]] = {}
    vertex_incident: dict[int, list[tuple[int, int]]] = {pid: [] for pid in geometry.sorted_ids}

    for a, b in geometry.lines:
        key = (a, b)
        pts_on: list[tuple[float, int]] = [(0.0, a), (1.0, b)]
        wa, wb = world[a], world[b]
        for pid, wp in world.items():
            if pid in (a, b):
                continue
            t = _point_on_segment_param(wp, wa, wb)
            if t is None:
                continue
            pts_on.append((t, pid))
            # Exact midpoint?
            mid = 0.5 * (wa + wb)
            if float(np.linalg.norm(wp - mid)) <= mid_tol:
                metric_midpoints[pid] = (a, b)

        pts_on.sort(key=lambda item: item[0])
        line_point_sets[key] = [pid for _, pid in pts_on]
        vertex_incident.setdefault(a, []).append((a, b))
        vertex_incident.setdefault(b, []).append((a, b))

    # Also detect midpoints between any pair of topology endpoints (not only
    # direct line endpoints) — e.g. north mid between NE and NW corners when
    # topology lists [4,5] and [5,6] separately.
    endpoint_ids = sorted({i for pair in geometry.lines for i in pair})
    for a, b in combinations(endpoint_ids, 2):
        wa, wb = world[a], world[b]
        mid = 0.5 * (wa + wb)
        for pid, wp in world.items():
            if pid in (a, b):
                continue
            if float(np.linalg.norm(wp - mid)) <= mid_tol and pid not in metric_midpoints:
                metric_midpoints[pid] = (a, b)

    return TopologyIndex(
        metric_midpoints=metric_midpoints,
        line_point_sets=line_point_sets,
        vertex_incident=vertex_incident,
    )


def line_intersection_2d(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    p3: NDArray[np.float64],
    p4: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Intersection of infinite lines p1–p2 and p3–p4, or None if parallel."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return px, py


def _import_dlt2d():
    try:
        from .dlt2d import dlt2d as dlt2d_solve
    except ImportError:
        from dlt2d import dlt2d as dlt2d_solve  # ty: ignore[unresolved-import]
    return dlt2d_solve


def _import_rec2d():
    try:
        from .rec2d_one_dlt2d import rec2d as rec2d_apply
    except ImportError:
        from rec2d_one_dlt2d import rec2d as rec2d_apply  # ty: ignore[unresolved-import]
    return rec2d_apply


def dlt_params_to_homography(params: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert 8 DLT2D coefficients (world→pixel) into a 3×3 homography.

    DLT maps (X,Y) → (u,v) via::

        u = (L1 X + L2 Y + L3) / (L7 X + L8 Y + 1)
        v = (L4 X + L5 Y + L6) / (L7 X + L8 Y + 1)

    which is the projective matrix [[L1,L2,L3],[L4,L5,L6],[L7,L8,1]].
    """
    p = np.asarray(params, dtype=np.float64).ravel()
    if p.size != 8:
        raise ValueError(f"Expected 8 DLT params, got {p.size}")
    return np.array(
        [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]],
        dtype=np.float64,
    )


def fit_dlt2d_visible(
    geometry: TargetGeometry,
    obs: dict[int, tuple[float, float]],
) -> NDArray[np.float64] | None:
    """Fit DLT2D from currently resolved/measured visible points (≥4 non-collinear)."""
    world_points = {pid: geometry.world_xy(pid) for pid in geometry.sorted_ids}
    valid_ids = [pid for pid in obs if pid in world_points]
    if len(valid_ids) < 4 or not has_non_collinear_quad(valid_ids, world_points):
        return None
    reals = np.array([world_points[i] for i in valid_ids], dtype=np.float64)
    pixels = np.array([obs[i] for i in valid_ids], dtype=np.float64)
    try:
        params = np.asarray(_import_dlt2d()(reals, pixels), dtype=np.float64).ravel()
    except Exception:
        return None
    if params.size != 8 or not np.isfinite(params).all():
        return None
    return params


def project_world_via_dlt(
    params: NDArray[np.float64], world_xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project Nx2 world points through DLT2D params (world→pixel via rec2d)."""
    # rec2d expects A = DLT params and cc2d = pixel coords for the *inverse*
    # (pixel→world). For world→pixel we use the homography form.
    h = dlt_params_to_homography(params)
    return project_points(h, world_xy)


def _two_known_pixels_on_world_line(
    line_a: int,
    line_b: int,
    geometry: TargetGeometry,
    resolved: dict[int, tuple[float, float]],
    *,
    tol: float = 1e-4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return two resolved pixels whose world points lie on the infinite line AB."""
    wa = geometry.world_xy(line_a)
    wb = geometry.world_xy(line_b)
    known: list[int] = []
    for pid, uv in resolved.items():
        if pid not in geometry.points:
            continue
        if not (np.isfinite(uv[0]) and np.isfinite(uv[1])):
            continue
        t = _point_on_segment_param(geometry.world_xy(pid), wa, wb, tol=tol)
        # Also accept points on the infinite line (t outside [0,1])
        if t is not None:
            known.append(pid)
            continue
        # Infinite-line collinearity check
        ab = wb - wa
        ap = geometry.world_xy(pid) - wa
        cross = abs(ab[0] * ap[1] - ab[1] * ap[0])
        if float(np.linalg.norm(ab)) > 1e-12 and cross / float(np.linalg.norm(ab)) <= tol:
            known.append(pid)
    if len(known) < 2:
        return None
    p0 = np.array(resolved[known[0]], dtype=np.float64)
    p1 = np.array(resolved[known[1]], dtype=np.float64)
    if float(np.linalg.norm(p0 - p1)) < 1e-9:
        return None
    return p0, p1


def _two_known_pixels_on_edge(
    edge: tuple[int, int],
    resolved: dict[int, tuple[float, float]],
    topo: TopologyIndex,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return two distinct known pixel points that lie on a topology edge."""
    known = [pid for pid in topo.line_point_sets.get(edge, list(edge)) if pid in resolved]
    if len(known) < 2:
        known = [pid for pid in edge if pid in resolved]
    if len(known) < 2:
        return None
    p_a = np.array(resolved[known[0]], dtype=np.float64)
    p_b = np.array(resolved[known[1]], dtype=np.float64)
    if float(np.linalg.norm(p_a - p_b)) < 1e-9:
        return None
    return p_a, p_b


def build_temporal_gap_fills(
    geometry: TargetGeometry,
    measurements: dict[int, dict[int, tuple[float, float]]],
) -> dict[int, dict[int, tuple[float, float]]]:
    """Temporal priors for missing markers (ideal for stabilized floor targets).

    For each geometry point:

    * **Interior gaps** — linear interpolation between measured anchors.
    * **Before first / after last** — constant hold of the nearest anchor
      (shape stays put when a marker is occluded on a stabilized video).
    """
    geom_ids = set(geometry.sorted_ids)
    frames_sorted = sorted(measurements)
    fills: dict[int, dict[int, tuple[float, float]]] = {f: {} for f in frames_sorted}
    if not frames_sorted:
        return fills

    for pid in geometry.sorted_ids:
        anchors: list[tuple[int, tuple[float, float]]] = []
        for frame in frames_sorted:
            obs = measurements.get(frame, {})
            if pid in obs and pid in geom_ids:
                anchors.append((frame, obs[pid]))
        if not anchors:
            continue

        f_first, p_first = anchors[0]
        f_last, p_last = anchors[-1]
        for frame in frames_sorted:
            if frame < f_first:
                fills.setdefault(frame, {})[pid] = p_first
            elif frame > f_last:
                fills.setdefault(frame, {})[pid] = p_last

        for i in range(len(anchors) - 1):
            fa, pa = anchors[i]
            fb, pb = anchors[i + 1]
            if fb <= fa + 1:
                continue
            span = float(fb - fa)
            for frame in range(fa + 1, fb):
                if pid in measurements.get(frame, {}):
                    continue
                t = (frame - fa) / span
                fills.setdefault(frame, {})[pid] = (
                    float((1.0 - t) * pa[0] + t * pb[0]),
                    float((1.0 - t) * pa[1] + t * pb[1]),
                )
    return fills


def project_point_onto_line(
    pt: NDArray[np.float64],
    p_a: NDArray[np.float64],
    p_b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Orthogonally project 2D point pt onto the infinite line through p_a and p_b."""
    v = p_b - p_a
    l2 = float(v @ v)
    if l2 < 1e-12:
        return pt
    t = float((pt - p_a) @ v) / l2
    return p_a + t * v


def _enforce_line_collinearity(
    geometry: TargetGeometry,
    pid: int,
    pt: NDArray[np.float64],
    resolved: dict[int, tuple[float, float]],
    topo: TopologyIndex,
) -> NDArray[np.float64]:
    """Ensure pt is collinear with known resolved points on its incident lines."""
    edges = topo.vertex_incident.get(pid, [])
    for edge in edges:
        pair = _two_known_pixels_on_edge(edge, resolved, topo)
        if pair is None:
            pair = _two_known_pixels_on_world_line(edge[0], edge[1], geometry, resolved)
        if pair is not None:
            pt = project_point_onto_line(pt, pair[0], pair[1])
            break
    return pt


def _topology_fill_missing(
    geometry: TargetGeometry,
    seed: dict[int, tuple[float, float]],
    topo: TopologyIndex,
    *,
    use_image_midpoints: bool = False,
    temporal_priors: dict[int, tuple[float, float]] | None = None,
) -> dict[int, tuple[float, float]]:
    """Bootstrap missing pixels via line intersections (and optional midpoints).

    Image-space midpoints are **off** under perspective (world midpoint ≠
    midpoint of image endpoints). Prefer temporal priors or DLT for mids;
    keep line–line intersections (projectively valid).
    """
    resolved: dict[int, tuple[float, float]] = dict(seed)
    world = {pid: geometry.world_xy(pid) for pid in geometry.sorted_ids}

    def _still_missing() -> list[int]:
        return [pid for pid in geometry.sorted_ids if pid not in resolved]

    if use_image_midpoints:
        for mid_id, (a_id, b_id) in topo.metric_midpoints.items():
            if mid_id in resolved:
                continue
            if a_id in resolved and b_id in resolved:
                ua, va = resolved[a_id]
                ub, vb = resolved[b_id]
                resolved[mid_id] = (0.5 * (ua + ub), 0.5 * (va + vb))

    for (a, b), pids in topo.line_point_sets.items():
        if a not in resolved or b not in resolved:
            continue
        wa, wb = world[a], world[b]
        pa = np.array(resolved[a], dtype=np.float64)
        pb = np.array(resolved[b], dtype=np.float64)
        ab_len2 = float((wb - wa) @ (wb - wa))
        if ab_len2 < 1e-12:
            continue
        for pid in pids:
            if pid in resolved or pid in (a, b):
                continue
            if not use_image_midpoints:
                continue
            t = float((world[pid] - wa) @ (wb - wa)) / ab_len2
            pix = pa + t * (pb - pa)
            resolved[pid] = (float(pix[0]), float(pix[1]))

    # Iterate line-intersection solves until convergence
    changed = True
    while changed:
        changed = False
        for vid in list(_still_missing()):
            edges = topo.vertex_incident.get(vid, [])
            if len(edges) < 2:
                continue
            line_pts: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
            for edge in edges:
                pair = _two_known_pixels_on_edge(edge, resolved, topo)
                if pair is None:
                    pair = _two_known_pixels_on_world_line(edge[0], edge[1], geometry, resolved)
                if pair is not None:
                    line_pts.append(pair)
                if len(line_pts) >= 2:
                    break
            if len(line_pts) < 2:
                continue
            hit = line_intersection_2d(
                line_pts[0][0], line_pts[0][1], line_pts[1][0], line_pts[1][1]
            )
            if hit is not None and np.isfinite(hit[0]) and np.isfinite(hit[1]):
                t_prior = (
                    np.array(temporal_priors[vid], dtype=np.float64)
                    if temporal_priors and vid in temporal_priors
                    else None
                )
                if t_prior is not None:
                    if float(np.linalg.norm(hit - t_prior)) <= 12.0:
                        resolved[vid] = (float(hit[0]), float(hit[1]))
                        changed = True
                    else:
                        proj_pt = project_point_onto_line(t_prior, line_pts[0][0], line_pts[0][1])
                        resolved[vid] = (float(proj_pt[0]), float(proj_pt[1]))
                        changed = True
                else:
                    resolved[vid] = hit
                    changed = True

    if temporal_priors:
        for vid in list(_still_missing()):
            if vid in temporal_priors:
                t_prior = np.array(temporal_priors[vid], dtype=np.float64)
                edges = topo.vertex_incident.get(vid, [])
                proj_done = False
                for edge in edges:
                    pair = _two_known_pixels_on_edge(edge, resolved, topo)
                    if pair is None:
                        pair = _two_known_pixels_on_world_line(edge[0], edge[1], geometry, resolved)
                    if pair is not None:
                        proj_pt = project_point_onto_line(t_prior, pair[0], pair[1])
                        resolved[vid] = (float(proj_pt[0]), float(proj_pt[1]))
                        proj_done = True
                        break
                if not proj_done:
                    resolved[vid] = (float(t_prior[0]), float(t_prior[1]))

    if use_image_midpoints:
        for mid_id, (a_id, b_id) in topo.metric_midpoints.items():
            if mid_id in resolved:
                continue
            if a_id in resolved and b_id in resolved:
                ua, va = resolved[a_id]
                ub, vb = resolved[b_id]
                resolved[mid_id] = (0.5 * (ua + ub), 0.5 * (va + vb))

    return resolved


def resolve_frame_pixels(
    geometry: TargetGeometry,
    obs: dict[int, tuple[float, float]],
    *,
    topo: TopologyIndex | None = None,
    prev_dlt: NDArray[np.float64] | None = None,
    temporal_guess: dict[int, tuple[float, float]] | None = None,
) -> tuple[dict[int, tuple[float, float]], NDArray[np.float64] | None, str]:
    """Resolve target points for a single frame.

    Priority:
    1. **Locked measurements** — digitized markers are preserved exactly.
    2. **Topology intersections** for missing corners via line–line crossing.
    3. **Direct DLT2D** on measured + topology-resolved points (>=4 non-collinear).
    4. **DLT projection** with line collinearity enforcement for remaining points.
    5. **Fallback to prev_dlt** if available.

    Returns ``(resolved_pixels, dlt_params_or_None, method)``.
    """
    topo = topo or build_topology_index(geometry)
    geom_ids = set(geometry.sorted_ids)
    obs = {pid: uv for pid, uv in obs.items() if pid in geom_ids}
    world = {pid: geometry.world_xy(pid) for pid in geometry.sorted_ids}

    resolved: dict[int, tuple[float, float]] = dict(obs)
    method = "measured_only"

    # 1. If explicit temporal_guess was passed, honor it
    if temporal_guess:
        for pid, uv in temporal_guess.items():
            if pid in geom_ids and pid not in resolved:
                resolved[pid] = uv
                method = "temporal_gap"

    # 2. Topology intersections for missing corners
    topo_filled = _topology_fill_missing(
        geometry, resolved, topo, use_image_midpoints=False, temporal_priors=temporal_guess
    )
    for pid, uv in topo_filled.items():
        if pid not in resolved:
            resolved[pid] = uv
            if method == "measured_only":
                method = "topology"

    # 3. Direct DLT on measured + topology-resolved points
    dlt_params = fit_dlt2d_visible(geometry, resolved)
    if dlt_params is not None:
        still_missing = [pid for pid in geometry.sorted_ids if pid not in resolved]
        if still_missing:
            world_miss = np.array([world[pid] for pid in still_missing], dtype=np.float64)
            projected = project_world_via_dlt(dlt_params, world_miss)
            for i, pid in enumerate(still_missing):
                pt_proj = _enforce_line_collinearity(geometry, pid, projected[i], resolved, topo)
                resolved[pid] = (float(pt_proj[0]), float(pt_proj[1]))
        return resolved, dlt_params, "dlt" if method == "measured_only" else method

    # 4. Fallback to prev_dlt if available
    if prev_dlt is not None:
        dlt_params = prev_dlt
        dlt_label = "temporal_dlt"
    else:
        dlt_label = "dlt"

    still_missing = [pid for pid in geometry.sorted_ids if pid not in resolved]
    if dlt_params is not None and still_missing:
        world_miss = np.array([world[pid] for pid in still_missing], dtype=np.float64)
        projected = project_world_via_dlt(dlt_params, world_miss)
        for i, pid in enumerate(still_missing):
            pt_proj = _enforce_line_collinearity(geometry, pid, projected[i], resolved, topo)
            resolved[pid] = (float(pt_proj[0]), float(pt_proj[1]))
        method = dlt_label if method == "measured_only" else method
    elif dlt_params is not None and method == "measured_only":
        method = "dlt"

    return resolved, dlt_params, method


# ------------------------------------------------------------------------- #
# Steps 2-3: Per-frame DLT / topology solve (+ legacy RANSAC helpers)
# ------------------------------------------------------------------------- #


@dataclass
class FrameHomography:
    frame: int
    homography: NDArray[np.float64]
    method: str  # "dlt" | "topology" | "temporal_gap" | "temporal_dlt" | ...
    n_correspondences: int


def _normalize_h(homography: NDArray[np.float64]) -> NDArray[np.float64]:
    if abs(homography[2, 2]) > 1e-12:
        return homography / homography[2, 2]
    return homography


def _is_direct_dlt_reliable(
    geometry: TargetGeometry,
    resolved: dict[int, tuple[float, float]],
    dlt: NDArray[np.float64],
    topo: TopologyIndex | None = None,
    *,
    max_rmse: float = 25.0,
) -> bool:
    """Verify DLT homography is well-conditioned and adequately covers the target."""
    h = dlt_params_to_homography(dlt)
    # 1. Positive projective depth for all target world points
    for pid in geometry.sorted_ids:
        wx, wy = geometry.world_xy(pid)
        w_prime = float(h[2, 0] * wx + h[2, 1] * wy + h[2, 2])
        if w_prime <= 0.05:
            return False

    # 2. Reprojection RMSE on resolved target points
    target_pts = [p for p in resolved if p in geometry.points]
    if len(target_pts) < 4:
        return False
    reals = np.array([geometry.world_xy(p) for p in target_pts], dtype=np.float64)
    pixs = np.array([resolved[p] for p in target_pts], dtype=np.float64)
    projs = project_world_via_dlt(dlt, reals)
    rmse = float(np.sqrt(np.mean(np.sum((pixs - projs) ** 2, axis=1))))
    if rmse > max_rmse:
        return False

    # 3. Geometric coverage: perimeter corners or sides
    if topo is not None:
        corners = [p for p in geometry.perimeter if p not in topo.metric_midpoints]
    else:
        corners = list(geometry.perimeter)

    if len(corners) >= 4:
        c_resolved = [c for c in corners if c in resolved]
        if len(c_resolved) >= 3:
            return True
        # If only 2 corners, check that perimeter sides have at least 2 points
        n_c = len(corners)
        all_sides_ok = True
        for i in range(n_c):
            c1 = corners[i]
            c2 = corners[(i + 1) % n_c]
            pts_on_side = [p for p in resolved if p in (c1, c2)]
            if topo is not None:
                for mid, (ma, mb) in topo.metric_midpoints.items():
                    if {ma, mb} == {c1, c2} and mid in resolved:
                        pts_on_side.append(mid)
            if len(pts_on_side) < 2:
                all_sides_ok = False
                break
        return all_sides_ok

    # Fallback for arbitrary point sets: at least 3 quadrants covered
    world = {pid: geometry.world_xy(pid) for pid in geometry.sorted_ids}
    cx = 0.5 * (min(w[0] for w in world.values()) + max(w[0] for w in world.values()))
    cy = 0.5 * (min(w[1] for w in world.values()) + max(w[1] for w in world.values()))
    quads = set()
    for p in target_pts:
        wx, wy = world[p]
        qx = 1 if wx >= cx else 0
        qy = 1 if wy >= cy else 0
        quads.add((qx, qy))
    return len(quads) >= 3


def resolve_all_frames(
    geometry: TargetGeometry,
    measurements: dict[int, dict[int, tuple[float, float]]],
) -> tuple[dict[int, dict[int, tuple[float, float]]], dict[int, FrameHomography]]:
    """Resolve every frame: pure measured locked, projective DLT gap-fill.

    Multi-pass projective tracking:
    1. Direct DLT2D fitted on measured + topology-resolved corner intersections.
    2. Occlusion gaps smoothly interpolate DLT homographies between boundary
       frames and align translations against any visible markers.
    3. Missing markers are reprojected from the physical target geometry
       via the per-frame homography and clamped to collinear incident lines.

    Returns ``(resolved_by_frame, solved_homographies)``.
    """
    topo = build_topology_index(geometry)
    geom_ids = set(geometry.sorted_ids)
    measurements = {
        f: {p: uv for p, uv in obs.items() if p in geom_ids} for f, obs in measurements.items()
    }
    frames_sorted = sorted(measurements)
    if not frames_sorted:
        raise ValueError("No measurement frames to resolve.")
    world = {pid: geometry.world_xy(pid) for pid in geometry.sorted_ids}
    temporal_fills = build_temporal_gap_fills(geometry, measurements)

    # Pass 1: Direct DLT on clean measured observations (+ topology intersections)
    direct_dlts: dict[int, NDArray[np.float64]] = {}
    topo_resolved_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    candidate_dlts: dict[int, NDArray[np.float64]] = {}
    for frame in frames_sorted:
        obs = measurements.get(frame, {})
        topo_filled = _topology_fill_missing(
            geometry, obs, topo, use_image_midpoints=False, temporal_priors=temporal_fills.get(frame)
        )
        topo_resolved_by_frame[frame] = topo_filled
        dlt = fit_dlt2d_visible(geometry, topo_filled)
        if dlt is not None:
            candidate_dlts[frame] = dlt
            if _is_direct_dlt_reliable(geometry, topo_filled, dlt, topo, max_rmse=25.0):
                direct_dlts[frame] = dlt

    if not direct_dlts and candidate_dlts:
        direct_dlts = dict(candidate_dlts)

    valid_frames = sorted(direct_dlts)
    all_dlts: dict[int, NDArray[np.float64] | None] = {}
    methods: dict[int, str] = {}

    # Pass 2: Fill gaps by homography interpolation / propagation + visible alignment
    for frame in frames_sorted:
        obs = measurements.get(frame, {})
        topo_obs = topo_resolved_by_frame.get(frame, obs)
        if frame in direct_dlts:
            all_dlts[frame] = direct_dlts[frame]
            methods[frame] = "dlt"
        elif valid_frames:
            left = [v for v in valid_frames if v < frame]
            right = [v for v in valid_frames if v > frame]
            vis = [p for p in topo_obs if p in geom_ids]
            if left and right:
                f0, f1 = left[-1], right[0]
                alpha = float(frame - f0) / float(f1 - f0)
                dlt_cur = (1.0 - alpha) * direct_dlts[f0] + alpha * direct_dlts[f1]
                m = "interpolated_dlt"
            elif left:
                dlt_cur = direct_dlts[left[-1]].copy()
                m = "propagated_dlt"
            else:
                dlt_cur = direct_dlts[right[0]].copy()
                m = "propagated_dlt"

            if vis:
                proj_vis = project_world_via_dlt(
                    dlt_cur, np.array([world[p] for p in vis], dtype=np.float64)
                )
                obs_vis = np.array([topo_obs[p] for p in vis], dtype=np.float64)
                shift = np.mean(obs_vis - proj_vis, axis=0)
                H = dlt_params_to_homography(dlt_cur)
                T = np.eye(3, dtype=np.float64)
                T[0, 2] = float(shift[0])
                T[1, 2] = float(shift[1])
                H_shifted = T @ H
                H_norm = H_shifted / H_shifted[2, 2]
                dlt_cur = np.array(
                    [
                        H_norm[0, 0],
                        H_norm[0, 1],
                        H_norm[0, 2],
                        H_norm[1, 0],
                        H_norm[1, 1],
                        H_norm[1, 2],
                        H_norm[2, 0],
                        H_norm[2, 1],
                    ],
                    dtype=np.float64,
                )
                m += "_aligned"
            all_dlts[frame] = dlt_cur
            methods[frame] = m
        else:
            # Complete absence of direct DLT: fallback to single-frame resolve
            resolved_f, dlt_f, m_f = resolve_frame_pixels(
                geometry, obs, topo=topo, temporal_guess=temporal_fills.get(frame)
            )
            all_dlts[frame] = dlt_f
            methods[frame] = m_f

    # Pass 3: Assemble resolved pixels (measured locked, missing projected) & FrameHomography
    resolved_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    solved: dict[int, FrameHomography] = {}

    for frame in frames_sorted:
        obs = measurements.get(frame, {})
        topo_filled = topo_resolved_by_frame.get(frame, {})
        resolved = dict(topo_filled)
        # Always lock original measured observations
        for p, uv in obs.items():
            resolved[p] = uv

        # Enforce line collinearity for any imputed (non-measured) points
        for p in geometry.sorted_ids:
            if p not in obs and p in resolved:
                pt_p = _enforce_line_collinearity(
                    geometry, p, np.array(resolved[p], dtype=np.float64), resolved, topo
                )
                resolved[p] = (float(pt_p[0]), float(pt_p[1]))

        dlt = all_dlts.get(frame)
        if dlt is not None:
            h = _normalize_h(dlt_params_to_homography(dlt))
            still_missing = [p for p in geometry.sorted_ids if p not in resolved]
            if still_missing:
                world_miss = np.array([world[p] for p in still_missing], dtype=np.float64)
                projected = project_world_via_dlt(dlt, world_miss)
                for i, p in enumerate(still_missing):
                    pt_proj = _enforce_line_collinearity(geometry, p, projected[i], resolved, topo)
                    resolved[p] = (float(pt_proj[0]), float(pt_proj[1]))
        else:
            h = np.eye(3, dtype=np.float64)

        resolved_by_frame[frame] = resolved
        solved[frame] = FrameHomography(
            frame=frame,
            homography=h,
            method=methods[frame],
            n_correspondences=len(obs),
        )

    if not any(
        s.method.startswith("dlt")
        or s.method
        in (
            "dlt",
            "interpolated_dlt",
            "propagated_dlt",
            "topology",
            "temporal_dlt",
            "ransac",
        )
        for s in solved.values()
    ):
        any_full = any(len(r) >= 4 for r in resolved_by_frame.values())
        if not any_full:
            raise ValueError(
                "No frame yielded enough correspondences or topology to resolve the geometry."
            )

    return resolved_by_frame, solved



def solve_frame_homographies(
    geometry: TargetGeometry,
    measurements: dict[int, dict[int, tuple[float, float]]],
    ransac_thresh: float = 3.0,  # kept for API compat; unused in DLT path
) -> dict[int, FrameHomography]:
    """Per-frame DLT2D (+ topology) solve. No PCHIP smoothing.

    ``ransac_thresh`` is accepted for backward compatibility with callers /
    CLI but the primary path uses least-squares DLT2D on visible points.
    """
    del ransac_thresh  # unused — DLT path does not RANSAC
    _resolved, solved = resolve_all_frames(geometry, measurements)
    return solved


def _nearest_solved_frame(frame: int, solved: dict[int, FrameHomography]) -> int | None:
    if not solved:
        return None
    candidates = [f for f in solved if f < frame]
    if candidates:
        return max(candidates)
    candidates = [f for f in solved if f > frame]
    if candidates:
        return min(candidates)
    return None


def cv2_find_homography(
    world_pts: NDArray[np.float64], pixel_pts: NDArray[np.float64], ransac_thresh: float
) -> tuple[NDArray[np.float64] | None, NDArray[Any] | None]:
    import cv2

    homography, inliers = cv2.findHomography(world_pts, pixel_pts, cv2.RANSAC, ransac_thresh)
    if homography is None:
        return None, None
    return homography.astype(np.float64), inliers


def cv2_estimate_affine_partial(
    src_pts: NDArray[np.float64], dst_pts: NDArray[np.float64]
) -> tuple[NDArray[np.float64] | None, NDArray[Any] | None]:
    import cv2

    transform, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if transform is None:
        return None, None
    return transform.astype(np.float64), inliers


# ------------------------------------------------------------------------- #
# Projection helpers (circles/arcs + H @ world)
# ------------------------------------------------------------------------- #


def project_points(
    homography: NDArray[np.float64], world_xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project Nx2 world (X, Y) points through H (unclamped, full-projective)."""
    n = world_xy.shape[0]
    homogeneous = np.hstack([world_xy, np.ones((n, 1), dtype=np.float64)])
    projected = (homography @ homogeneous.T).T
    w = projected[:, 2]
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return projected[:, :2] / w_safe[:, None]


def sample_circle_world(circle: TargetCircle, geometry: TargetGeometry) -> NDArray[np.float64]:
    center = geometry.world_xy(circle.center_point)
    theta = np.linspace(0.0, 2.0 * np.pi, circle.samples, endpoint=True)
    x = center[0] + circle.radius * np.cos(theta)
    y = center[1] + circle.radius * np.sin(theta)
    return np.column_stack([x, y])


def sample_arc_world(arc: TargetArc, geometry: TargetGeometry) -> NDArray[np.float64]:
    center = geometry.world_xy(arc.center_point)
    theta = np.radians(np.linspace(arc.angle_start_deg, arc.angle_end_deg, arc.samples))
    x = center[0] + arc.radius * np.cos(theta)
    y = center[1] + arc.radius * np.sin(theta)
    return np.column_stack([x, y])


# ------------------------------------------------------------------------- #
# Dense per-frame diagnostic table + imputed wide CSV
# ------------------------------------------------------------------------- #


@dataclass
class DenseRow:
    frame: int
    point_id: int
    point_name: str
    u: float
    v: float
    is_measured: bool
    in_fov: bool
    reproj_error_px: float
    frame_rmse_px: float


def build_dense_table(
    geometry: TargetGeometry,
    measurements: dict[int, dict[int, tuple[float, float]]],
    solved: dict[int, FrameHomography],
    frame_size: tuple[int, int] | None,
    *,
    resolved_by_frame: dict[int, dict[int, tuple[float, float]]] | None = None,
) -> list[DenseRow]:
    """Build long diagnostic rows from geometry-consistent resolved pixels.

    ``u``/``v`` always come from the resolved (DLT-reprojected) geometry.
    ``is_measured`` flags which points had an observation; ``reproj_error_px``
    is |measured − resolved| for those points.
    """
    rows: list[DenseRow] = []
    point_ids = geometry.sorted_ids
    w_bound, h_bound = frame_size if frame_size is not None else (None, None)

    for frame in sorted(solved):
        fh = solved[frame]
        obs = measurements.get(frame, {})
        resolved = resolved_by_frame.get(frame, {}) if resolved_by_frame is not None else {}
        if not resolved:
            # Legacy fallback: project all world points through H
            world_xy = np.array([geometry.world_xy(pid) for pid in point_ids], dtype=np.float64)
            projected = project_points(fh.homography, world_xy)
            resolved = {
                pid: (float(projected[i, 0]), float(projected[i, 1]))
                for i, pid in enumerate(point_ids)
            }

        frame_errors: list[float] = []
        per_point: list[DenseRow] = []
        for pid in point_ids:
            measured = obs.get(pid)
            u_res, v_res = resolved.get(pid, (float("nan"), float("nan")))
            u_out, v_out = float(u_res), float(v_res)
            if measured is not None:
                u_meas, v_meas = measured
                if np.isfinite(u_res) and np.isfinite(v_res):
                    err = float(np.hypot(u_meas - u_res, v_meas - v_res))
                    frame_errors.append(err)
                    reproj_error = err
                else:
                    reproj_error = float("nan")
                is_measured = True
            else:
                is_measured = False
                reproj_error = float("nan")

            if w_bound is None or h_bound is None:
                in_fov = True
            else:
                in_fov = bool(
                    np.isfinite(u_out)
                    and np.isfinite(v_out)
                    and 0 <= u_out <= w_bound
                    and 0 <= v_out <= h_bound
                )

            per_point.append(
                DenseRow(
                    frame=frame,
                    point_id=pid,
                    point_name=geometry.points[pid].name,
                    u=u_out,
                    v=v_out,
                    is_measured=is_measured,
                    in_fov=in_fov,
                    reproj_error_px=reproj_error,
                    frame_rmse_px=float("nan"),
                )
            )

        frame_rmse = (
            float(np.sqrt(np.mean(np.square(frame_errors)))) if frame_errors else float("nan")
        )
        for row in per_point:
            row.frame_rmse_px = frame_rmse
        rows.extend(per_point)

    return rows


def dense_rows_to_wide_csv(rows: list[DenseRow], point_ids: list[int]) -> pd.DataFrame:
    """Reshape the long diagnostic rows into getpixelvideo.py's wide convention."""
    by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    for row in rows:
        by_frame.setdefault(row.frame, {})[row.point_id] = (row.u, row.v)

    records = []
    for frame in sorted(by_frame):
        rec: dict[str, float | int] = {"frame": frame}
        for pid in point_ids:
            u, v = by_frame[frame].get(pid, (float("nan"), float("nan")))
            rec[f"p{pid}_x"] = u
            rec[f"p{pid}_y"] = v
        records.append(rec)
    return pd.DataFrame(records)


# ------------------------------------------------------------------------- #
# Outputs
# ------------------------------------------------------------------------- #


def write_geometry_animation_html(
    output_path: Path,
    geometry: TargetGeometry,
    resolved_by_frame: dict[int, dict[int, tuple[float, float]]],
    solved: dict[int, FrameHomography],
    *,
    title: str = "Planar geometry (pixel + world)",
) -> Path:
    """Write a self-contained HTML slider: pixel wireframe + rec2d world."""
    import json

    rec2d_apply = _import_rec2d()
    frames_payload: list[dict[str, object]] = []
    lines = [[int(a), int(b)] for a, b in geometry.lines]
    ideal_world = {
        str(pid): [float(geometry.world_xy(pid)[0]), float(geometry.world_xy(pid)[1])]
        for pid in geometry.sorted_ids
    }

    for frame in sorted(resolved_by_frame):
        pix = resolved_by_frame[frame]
        pix_out = {
            str(pid): [float(pix[pid][0]), float(pix[pid][1])]
            for pid in geometry.sorted_ids
            if pid in pix
        }
        world_rec: dict[str, list[float]] = {}
        fh = solved.get(frame)
        if fh is not None and pix_out:
            try:
                # H maps world→pixel; invert via DLT params for rec2d (pixel→world)
                h = fh.homography
                params = np.array(
                    [h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1]],
                    dtype=np.float64,
                )
                ids = [int(k) for k in pix_out]
                cc = np.array([pix_out[str(i)] for i in ids], dtype=np.float64)
                wr = np.asarray(rec2d_apply(params, cc), dtype=np.float64)
                world_rec = {str(i): [float(wr[j, 0]), float(wr[j, 1])] for j, i in enumerate(ids)}
            except Exception:
                world_rec = {}
        frames_payload.append({"frame": int(frame), "pixel": pix_out, "world": world_rec})

    data = {
        "title": title,
        "lines": lines,
        "ideal_world": ideal_world,
        "frames": frames_payload,
    }
    payload = json.dumps(data, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 16px; background: #111; color: #eee; }}
.row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
canvas {{ background: #1a1a1a; border: 1px solid #444; }}
#meta {{ margin: 8px 0; font-size: 14px; }}
input[type=range] {{ width: min(640px, 100%); }}
</style>
</head>
<body>
<h1>{title}</h1>
<div id="meta"></div>
<input id="slider" type="range" min="0" max="0" value="0"/>
<div class="row">
  <div><h3>Pixel</h3><canvas id="pix" width="600" height="600"></canvas></div>
  <div><h3>World (rec2d) vs TOML</h3><canvas id="world" width="600" height="600"></canvas></div>
</div>
<script>
const DATA = {payload};
const slider = document.getElementById('slider');
const meta = document.getElementById('meta');
const pixC = document.getElementById('pix');
const worldC = document.getElementById('world');
slider.max = Math.max(0, DATA.frames.length - 1);

function bounds(pts) {{
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for (const p of Object.values(pts)) {{
    if (!p) continue;
    minX=Math.min(minX,p[0]); minY=Math.min(minY,p[1]);
    maxX=Math.max(maxX,p[0]); maxY=Math.max(maxY,p[1]);
  }}
  if (!isFinite(minX)) return {{minX:0,minY:0,maxX:1,maxY:1,spanX:1,spanY:1,cx:0.5,cy:0.5}};
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  return {{minX, minY, maxX, maxY, spanX, spanY, cx: 0.5 * (minX + maxX), cy: 0.5 * (minY + maxY)}};
}}

function bounds_all(dictList) {{
  let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  for (const pts of dictList) {{
    if (!pts) continue;
    for (const p of Object.values(pts)) {{
      if (!p) continue;
      minX=Math.min(minX,p[0]); minY=Math.min(minY,p[1]);
      maxX=Math.max(maxX,p[0]); maxY=Math.max(maxY,p[1]);
    }}
  }}
  if (!isFinite(minX)) return {{minX:0,minY:0,maxX:1,maxY:1,spanX:1,spanY:1,cx:0.5,cy:0.5}};
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  return {{minX, minY, maxX, maxY, spanX, spanY, cx: 0.5 * (minX + maxX), cy: 0.5 * (minY + maxY)}};
}}

const pixBounds = bounds_all(DATA.frames.map(f => f.pixel));
const worldBounds = bounds(DATA.ideal_world);

function mapPt(p, b, w, h, flipY) {{
  const pad = 40;
  const availW = Math.max(10, w - 2 * pad);
  const availH = Math.max(10, h - 2 * pad);
  const scale = Math.min(availW / b.spanX, availH / b.spanY);
  const x = w * 0.5 + (p[0] - b.cx) * scale;
  const y = flipY ? (h * 0.5 - (p[1] - b.cy) * scale) : (h * 0.5 + (p[1] - b.cy) * scale);
  return [x, y];
}}

function draw(canvas, pts, lines, ideal, flipY, b) {{
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.strokeStyle = '#666'; ctx.setLineDash([4,4]);
  if (ideal) {{
    for (const [a,c] of lines) {{
      const pa = ideal[String(a)], pc = ideal[String(c)];
      if (!pa||!pc) continue;
      const A=mapPt(pa,b,w,h,flipY), C=mapPt(pc,b,w,h,flipY);
      ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(C[0],C[1]); ctx.stroke();
    }}
  }}
  ctx.setLineDash([]); ctx.strokeStyle = '#4fc3f7'; ctx.lineWidth = 2;
  for (const [a,c] of lines) {{
    const pa = pts[String(a)], pc = pts[String(c)];
    if (!pa||!pc) continue;
    const A=mapPt(pa,b,w,h,flipY), C=mapPt(pc,b,w,h,flipY);
    ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(C[0],C[1]); ctx.stroke();
  }}
  ctx.fillStyle = '#ffca28';
  for (const [id,p] of Object.entries(pts)) {{
    const P=mapPt(p,b,w,h,flipY);
    ctx.beginPath(); ctx.arc(P[0],P[1],4,0,Math.PI*2); ctx.fill();
    ctx.fillText('p'+id, P[0]+6, P[1]-6);
  }}
}}

function render(i) {{
  const fr = DATA.frames[i];
  if (!fr) return;
  const maxIdx = Math.max(0, DATA.frames.length - 1);
  meta.textContent = 'Frame: ' + fr.frame + ' (' + fr.frame + '/' + maxIdx + ')  •  Total: ' + DATA.frames.length + ' frames';
  draw(pixC, fr.pixel, DATA.lines, null, false, pixBounds);
  draw(worldC, fr.world, DATA.lines, DATA.ideal_world, true, worldBounds);
}}
slider.addEventListener('input', () => render(+slider.value));
window.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {{
    slider.value = Math.max(0, +slider.value - 1);
    render(+slider.value);
  }} else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {{
    slider.value = Math.min(DATA.frames.length - 1, +slider.value + 1);
    render(+slider.value);
  }}
}});
render(0);
</script>
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_outputs(
    output_dir: Path,
    stem: str,
    geometry: TargetGeometry,
    rows: list[DenseRow],
    solved: dict[int, FrameHomography],
    *,
    resolved_by_frame: dict[int, dict[int, tuple[float, float]]] | None = None,
) -> dict[str, Path]:
    # Geometry parsing/projection is also used by headless video stabilization.
    # Load the GUI-facing sports-field export layer only when exporting REF3D.
    if __package__:
        from .drawsportsfields import (
            FieldControlPoint,
            build_ref3d_dataframe,
            build_ref3d_map_dataframe,
        )
    else:
        from drawsportsfields import (
            FieldControlPoint,
            build_ref3d_dataframe,
            build_ref3d_map_dataframe,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    point_ids = geometry.sorted_ids
    imputed_df = dense_rows_to_wide_csv(rows, point_ids)
    imputed_path = output_dir / f"{stem}_imputed.csv"
    imputed_df.to_csv(imputed_path, index=False)
    written["imputed_csv"] = imputed_path

    long_df = pd.DataFrame(
        [
            {
                "frame": r.frame,
                "point_id": r.point_id,
                "point_name": r.point_name,
                "u": r.u,
                "v": r.v,
                "is_measured": r.is_measured,
                "in_fov": r.in_fov,
                "reproj_error_px": r.reproj_error_px,
                "frame_rmse_px": r.frame_rmse_px,
            }
            for r in rows
        ]
    )
    long_path = output_dir / "dense_projected_points_long.csv"
    long_df.to_csv(long_path, index=False)
    written["long_csv"] = long_path

    frames_sorted = sorted(solved)
    h_stack = np.stack([solved[f].homography for f in frames_sorted])
    h_inv_stack = np.stack([np.linalg.inv(solved[f].homography) for f in frames_sorted])
    npz_path = output_dir / "homographies.npz"
    np.savez_compressed(
        npz_path,
        H=h_stack,
        H_inv=h_inv_stack,
        frame_ids=np.array(frames_sorted, dtype=np.int64),
    )
    written["homographies_npz"] = npz_path

    # target_calibration.ref3d: repo's real CSV .ref3d convention (see
    # vaila/drawsportsfields.py build_ref3d_dataframe/build_ref3d_map_dataframe),
    # not the spec's incorrect "Kinovea XML" claim -- kept consistent with
    # every other .ref3d producer/consumer in the repo.
    control_points = [
        FieldControlPoint(
            key=f"target:{pid}:{geometry.points[pid].name}",
            label=f"{pid:02d}  {geometry.points[pid].name}",
            x=geometry.points[pid].x,
            y=geometry.points[pid].y,
            z=geometry.points[pid].z,
            source_index=pid,
            source_name=geometry.points[pid].name,
        )
        for pid in point_ids
    ]
    ref3d_path = output_dir / "target_calibration.ref3d"
    build_ref3d_dataframe(control_points, index_base=1).to_csv(ref3d_path, index=False)
    map_path = output_dir / "target_calibration.ref3d_map.csv"
    build_ref3d_map_dataframe(control_points, index_base=1).to_csv(map_path, index=False)
    written["ref3d"] = ref3d_path
    written["ref3d_map"] = map_path

    if resolved_by_frame is not None:
        html_path = output_dir / "geometry_animation.html"
        write_geometry_animation_html(
            html_path,
            geometry,
            resolved_by_frame,
            solved,
            title=f"{stem} — planar geometry",
        )
        written["animation_html"] = html_path

    return written


# ------------------------------------------------------------------------- #
# Extended-canvas / debug-viz renderer
# ------------------------------------------------------------------------- #

_COLOR_MEASURED = (0, 200, 0)  # solid green circle
_COLOR_IMPUTED_IN_FOV = (255, 255, 0)  # cyan triangle (BGR: cyan == (255,255,0))
_COLOR_EXTRAPOLATED = (255, 0, 255)  # magenta square
_COLOR_WIREFRAME = (0, 255, 255)  # yellow


def render_debug_video(
    video_path: Path,
    output_path: Path,
    geometry: TargetGeometry,
    solved: dict[int, FrameHomography],
    rows: list[DenseRow],
    *,
    extended_canvas: bool,
    canvas_scale: float,
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if extended_canvas:
        canvas_w = int(round(canvas_scale * frame_w))
        canvas_h = int(round(canvas_scale * frame_h))
        x_offset = (canvas_scale - 1.0) * frame_w / 2.0
        y_offset = (canvas_scale - 1.0) * frame_h / 2.0
        t_canvas = np.array(
            [[1.0, 0.0, x_offset], [0.0, 1.0, y_offset], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    else:
        canvas_w, canvas_h = frame_w, frame_h
        t_canvas = np.eye(3, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    rows_by_frame: dict[int, list[DenseRow]] = {}
    for r in rows:
        rows_by_frame.setdefault(r.frame, []).append(r)

    try:
        frame_idx = 0
        while True:
            ok, frame_img = cap.read()
            if not ok:
                break

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            if extended_canvas:
                x0, y0 = int(round(x_offset)), int(round(y_offset))
                canvas[y0 : y0 + frame_h, x0 : x0 + frame_w] = frame_img
            else:
                canvas[:, :] = frame_img

            frame_rows = rows_by_frame.get(frame_idx, [])
            if frame_idx in solved and frame_rows:
                h_canvas = t_canvas @ solved[frame_idx].homography
                _draw_wireframe(canvas, geometry, h_canvas)
                for r in frame_rows:
                    pt_canvas = project_points(t_canvas, np.array([[r.u, r.v]]))[0]
                    center = (int(round(pt_canvas[0])), int(round(pt_canvas[1])))
                    if r.is_measured:
                        cv2.circle(canvas, center, 5, _COLOR_MEASURED, -1)
                    elif r.in_fov:
                        _draw_triangle(canvas, center, _COLOR_IMPUTED_IN_FOV)
                    else:
                        _draw_square(canvas, center, _COLOR_EXTRAPOLATED)

            writer.write(canvas)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def _draw_wireframe_from_pixels(
    canvas: NDArray[np.uint8],
    geometry: TargetGeometry,
    resolved: dict[int, tuple[float, float]],
    t_canvas: NDArray[np.float64],
) -> None:
    """Draw topology edges by connecting resolved pixel coordinates."""
    import cv2

    for a, b in geometry.lines:
        if a not in resolved or b not in resolved:
            continue
        pts = project_points(t_canvas, np.array([resolved[a], resolved[b]], dtype=np.float64))
        p1 = (int(round(pts[0, 0])), int(round(pts[0, 1])))
        p2 = (int(round(pts[1, 0])), int(round(pts[1, 1])))
        cv2.line(canvas, p1, p2, _COLOR_WIREFRAME, 1, cv2.LINE_AA)


def _draw_circles_arcs(
    canvas: NDArray[np.uint8], geometry: TargetGeometry, homography: NDArray[np.float64]
) -> None:
    import cv2

    for circle in geometry.circles:
        world = sample_circle_world(circle, geometry)
        pts = project_points(homography, world)
        poly = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, _COLOR_WIREFRAME, 1, cv2.LINE_AA)

    for arc in geometry.arcs:
        world = sample_arc_world(arc, geometry)
        pts = project_points(homography, world)
        poly = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, _COLOR_WIREFRAME, 1, cv2.LINE_AA)


def _draw_wireframe(
    canvas: NDArray[np.uint8], geometry: TargetGeometry, homography: NDArray[np.float64]
) -> None:
    """Legacy: project topology through H (kept for callers / circles)."""
    import cv2

    for a, b in geometry.lines:
        pts = project_points(homography, np.array([geometry.world_xy(a), geometry.world_xy(b)]))
        p1 = tuple(int(round(v)) for v in pts[0])
        p2 = tuple(int(round(v)) for v in pts[1])
        cv2.line(canvas, p1, p2, _COLOR_WIREFRAME, 1, cv2.LINE_AA)
    _draw_circles_arcs(canvas, geometry, homography)


def _draw_triangle(
    canvas: NDArray[np.uint8], center: tuple[int, int], color: tuple[int, int, int]
) -> None:
    import cv2

    cx, cy = center
    pts = np.array([[cx, cy - 6], [cx - 6, cy + 5], [cx + 6, cy + 5]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color)


def _draw_square(
    canvas: NDArray[np.uint8], center: tuple[int, int], color: tuple[int, int, int]
) -> None:
    import cv2

    cx, cy = center
    cv2.rectangle(canvas, (cx - 5, cy - 5), (cx + 5, cy + 5), color, -1)


# ------------------------------------------------------------------------- #
# Video frame size probe (optional)
# ------------------------------------------------------------------------- #


def probe_video_frame_size(video_path: Path | None) -> tuple[int, int] | None:
    if video_path is None or not Path(video_path).is_file():
        return None
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        return None
    return w, h


# ------------------------------------------------------------------------- #
# Pipeline entry point
# ------------------------------------------------------------------------- #


def run_planar_geometry_tracker(
    config_path: Path,
    measurements_csv: Path,
    output_dir: Path,
    *,
    video_path: Path | None = None,
    ransac_thresh: float = 3.0,
    canvas_scale: float = 1.8,
    extended_canvas: bool = False,
    debug_viz: bool = False,
) -> dict[str, Path]:
    del ransac_thresh  # retained for CLI/API compat; primary path is DLT2D
    output_dir = Path(output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_parent = Path(measurements_csv).resolve().parent
    vid_parent = Path(video_path).resolve().parent if video_path else None
    if output_dir.resolve() in (csv_parent, vid_parent):
        output_dir = output_dir / f"processed_planar_geom_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = load_target_geometry(config_path)
    measurements = load_measurements_csv(measurements_csv)
    resolved_by_frame, solved = resolve_all_frames(geometry, measurements)
    frame_size = probe_video_frame_size(video_path)
    rows = build_dense_table(
        geometry,
        measurements,
        solved,
        frame_size,
        resolved_by_frame=resolved_by_frame,
    )

    stem = Path(measurements_csv).stem
    written = write_outputs(
        output_dir,
        stem,
        geometry,
        rows,
        solved,
        resolved_by_frame=resolved_by_frame,
    )

    if debug_viz and video_path is not None:
        debug_path = output_dir / "debug_projected_wireframe.mp4"
        render_debug_video(
            Path(video_path),
            debug_path,
            geometry,
            solved,
            rows,
            extended_canvas=extended_canvas,
            canvas_scale=canvas_scale,
        )
        written["debug_video"] = debug_path
    elif debug_viz and video_path is None:
        print(">> --debug-viz requested but no --video-path given; skipping video render.")

    return written


# ------------------------------------------------------------------------- #
# GUI entry point (Frame C → Video and Image → Planar Geo)
# ------------------------------------------------------------------------- #


def _default_planar_targets_dir() -> Path:
    return Path(__file__).resolve().parent / "models" / "planar_targets"


def run_planar_geometry_tracker_gui(parent: Any | None = None) -> None:
    """Tkinter file-dialog flow for the standalone planar-geometry tracker.

    Prompts for measurements CSV, target-geometry TOML, optional video, and an
    output directory (defaults to a timestamped folder next to the CSV). Also
    reachable from getpixelvideo's **Geo Homog** toolbar button (wizard path).
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    owns_root = parent is None
    root = tk.Tk() if owns_root else tk.Toplevel(parent)
    root.withdraw()
    if parent is not None:
        root.transient(parent)

    try:
        csv_path = filedialog.askopenfilename(
            parent=root,
            title="Select marker measurements CSV (getpixelvideo)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not csv_path:
            return
        measurements_csv = Path(csv_path)

        initial_dir = _default_planar_targets_dir()
        if not initial_dir.is_dir():
            initial_dir = measurements_csv.parent
        config_path_str = filedialog.askopenfilename(
            parent=root,
            title="Select target-geometry TOML",
            initialdir=str(initial_dir),
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not config_path_str:
            return
        config_path = Path(config_path_str)

        video_path_str = filedialog.askopenfilename(
            parent=root,
            title="Optional reference video (Cancel to skip)",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
                ("All files", "*.*"),
            ],
        )
        video_path = Path(video_path_str) if video_path_str else None

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = measurements_csv.parent / f"processed_planar_geom_{stamp}"
        output_dir_str = filedialog.askdirectory(
            parent=root,
            title="Select output directory (Cancel = auto timestamped)",
            initialdir=str(measurements_csv.parent),
        )
        if not output_dir_str:
            output_dir = default_out
        else:
            chosen = Path(output_dir_str)
            if chosen.resolve() == measurements_csv.parent.resolve() or not chosen.name.startswith("processed_"):
                output_dir = chosen / f"processed_planar_geom_{stamp}"
            else:
                output_dir = chosen
        output_dir.mkdir(parents=True, exist_ok=True)

        cli_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "vaila.planar_geometry_tracker",
            "--config",
            str(config_path),
            "--measurements-csv",
            str(measurements_csv),
            "--output-dir",
            str(output_dir),
        ]
        if video_path is not None:
            cli_cmd.extend(["--video-path", str(video_path), "--debug-viz"])
        print_gui_cli_mirror("vaila/planar_geometry_tracker", cli_cmd)

        try:
            written = run_planar_geometry_tracker(
                config_path,
                measurements_csv,
                output_dir,
                video_path=video_path,
                debug_viz=video_path is not None,
            )
        except Exception as exc:  # noqa: BLE001 - surface to user, keep GUI alive
            messagebox.showerror("Planar Geo", f"Run failed:\n{exc}", parent=root)
            return

        lines = [f"{key}: {path}" for key, path in written.items()]
        messagebox.showinfo(
            "Planar Geo",
            "Finished.\n\n" + "\n".join(lines),
            parent=root,
        )
    finally:
        root.destroy()


# ------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaila.planar_geometry_tracker",
        description=(
            "Planar-geometry homography tracker/extrapolator/gap-filler: fits "
            "per-frame homographies against a metric target profile (TOML), "
            "imputes occluded markers and extrapolates points beyond the "
            "camera FOV without clamping."
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="Target-geometry TOML profile.")
    parser.add_argument(
        "--measurements-csv",
        required=True,
        type=Path,
        help="Marker CSV from getpixelvideo.py (wide p{i}_x/p{i}_y or long point_id/u/v).",
    )
    parser.add_argument("--video-path", type=Path, default=None, help="Reference video (optional).")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./vaila_tracker_output"), help="Output directory."
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=3.0,
        help="RANSAC inlier reprojection threshold in pixels (default 3.0).",
    )
    parser.add_argument(
        "--canvas-scale",
        type=float,
        default=1.8,
        help="Extended-canvas scale factor (default 1.8).",
    )
    parser.add_argument(
        "--extended-canvas",
        action="store_true",
        help="Render the debug video on an expanded virtual canvas.",
    )
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help="Write debug_projected_wireframe.mp4 (requires --video-path).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        run_planar_geometry_tracker_gui()
        return 0

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        written = run_planar_geometry_tracker(
            args.config,
            args.measurements_csv,
            args.output_dir,
            video_path=args.video_path,
            ransac_thresh=args.ransac_thresh,
            canvas_scale=args.canvas_scale,
            extended_canvas=args.extended_canvas,
            debug_viz=args.debug_viz,
        )
    except Exception as exc:  # noqa: BLE001 - CLI top-level error reporting
        print(f">> planar_geometry_tracker error: {exc}", file=sys.stderr)
        return 1

    print(">> planar_geometry_tracker finished. Outputs:")
    for key, path in written.items():
        print(f"   {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
