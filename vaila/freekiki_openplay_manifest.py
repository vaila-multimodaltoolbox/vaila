"""Two-camera original-frame preparation and dribble analysis for FreeKiki.

Version: 0.4.14
Created: 02 November 2026
Update Date: 02 November 2026

Input files are read only. Equalized imagery supplies seeds, never a measurement
clock. Image registration transfers the supplied calibration to originals; its
height/scale assumptions remain inherited and are reported explicitly.

Optional manifest ``turns``: the changes of direction of the dribbler, chosen on the video in
original pairs (0-based). ``before``/``after`` are the straight stretches around the turn (path
angle, entry/exit speed); ``body_frames`` measures the body turn from the SAM 3D Body hip line in
each camera. ``primary`` (``path`` or ``body``) says which number the TV card shows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.signal import savgol_filter

from .freekiki_camera3d import cam_int_sequence, focal_from_dlt3d
from .freekiki_diag import load_field_points
from .freekiki_field3d import dlt_for_frames
from .freekiki_openplay_kinematics import (
    body_rotation,
    body_yaw_deg,
    pair_approach,
    track_metrics,
    turn_between,
)
from .freekiki_players import homography_from_dlt2d
from .freekiki_stereo import triangulate
from .monocular_dlt_align import dlt_projection_matrix


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(r) for r in rows)


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "freekiki-dribble-tv-v1":
        raise ValueError("manifest schema must be freekiki-dribble-tv-v1")
    cams = data.get("cameras", [])
    people = data.get("participants", [])
    if len(cams) != 2 or [p.get("role") for p in people].count("dribbler") != 1:
        raise ValueError("manifest needs two cameras and one unique dribbler")
    if not 2 <= len(people) <= 4:
        raise ValueError("manifest needs a dribbler and one to three defenders")
    if len({p["id"] for p in people}) != len(people):
        raise ValueError("participant identities must be unique across cameras")
    if len({c["id"] for c in cams}) != 2 or sum(bool(c.get("main")) for c in cams) != 1:
        raise ValueError("camera IDs must be unique and one camera must be main")
    counts = []
    for cam in cams:
        for key in (
            "video",
            "seed_video",
            "frame_map",
            "dlt2d",
            "dlt3d",
            "ball_markers",
            "field_points",
            "seed_camera",
            "seed_keypoints",
        ):
            p = Path(cam[key]).expanduser()
            cam[key] = str((path.parent / p).resolve() if not p.is_absolute() else p)
            if not Path(cam[key]).is_file():
                raise FileNotFoundError(cam[key])
        start, end = cam["range_1based"]
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            raise ValueError("camera range must be an inclusive 1-based integer pair")
        counts.append(end - start + 1)
        ids = [p["camera_ids"][cam["id"]] for p in people]
        if len(ids) != len(set(ids)):
            raise ValueError("one camera ID cannot be both dribbler and defender")
    if counts[0] != counts[1] or counts[0] < 3:
        raise ValueError("original frame ranges must contain the same number of pairs (>=3)")
    validate_turns(data.get("turns", []), counts[0])
    data["n_frames"] = counts[0]
    data["manifest_path"] = str(path)
    return data


TURN_PRIMARY = ("path", "body")
TURN_ANGLE_SPREAD_DEG = 15.0  # operational: path angle across the 5/7/9 filters for a stable turn
TURN_BODY_AGREEMENT_DEG = 20.0  # operational: body turn of the two cameras for a stable turn


def validate_turns(turns, n):
    """Check the manifest ``turns`` (0-based original pairs, inclusive)."""
    if not isinstance(turns, list):
        raise ValueError("turns must be a list")
    ids = set()
    for turn in turns:
        tid = turn.get("id")
        if not tid or tid in ids:
            raise ValueError("each turn needs a unique id")
        ids.add(tid)
        primary = turn.get("primary", "path")
        if primary not in TURN_PRIMARY:
            raise ValueError(f"turn {tid}: primary must be one of {TURN_PRIMARY}")
        spans = [turn[k] for k in ("before", "after", "body_frames") if k in turn]
        if "before" not in turn or "after" not in turn:
            raise ValueError(f"turn {tid}: before and after stretches are required")
        if primary == "body" and "body_frames" not in turn:
            raise ValueError(f"turn {tid}: a body turn needs body_frames")
        for a, b in spans:
            if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b < n):
                raise ValueError(f"turn {tid}: frames must be 0-based pairs with start < end < {n}")
        if turn["before"][1] > turn["after"][0]:
            raise ValueError(f"turn {tid}: before must end at or before the start of after")


def body_yaw_series(folder, cam, sam_id, n):
    """Hip-line yaw (deg, unwrapped) of one SAM 3D Body track per original pair, or None."""
    files = sorted((Path(folder) / "sam3d").glob(f"*_id_{int(sam_id):02d}_mhr70_3d.csv"))
    if not files:
        return None
    offset = cam["range_1based"][0] - 1
    left = np.full((n, 2), np.nan)
    right = np.full((n, 2), np.nan)
    for r in read_csv(files[0]):
        pair = int(float(r["frame"])) - offset
        if not 0 <= pair < n:
            continue
        try:
            left[pair] = float(r["left_hip_x"]), float(r["left_hip_z"])
            right[pair] = float(r["right_hip_x"]), float(r["right_hip_z"])
        except (KeyError, ValueError):
            continue
    return body_yaw_deg(left, right)


def analyze_turns(turns, paths_by_window, body_yaws, rate):
    """Each manifest turn measured with the three filters and, for the body, in each camera."""
    results = []
    for turn in turns:
        by_window = {
            w: turn_between(xy, rate, tuple(turn["before"]), tuple(turn["after"]))
            for w, xy in paths_by_window.items()
        }
        found = [r for r in by_window.values() if r is not None]
        body = {}
        if "body_frames" in turn:
            start, end = turn["body_frames"]
            for cam_id, yaw in body_yaws.items():
                body[cam_id] = body_rotation(yaw, rate, start, end) if yaw is not None else None
        angles = [r["abs_deg"] for r in found]
        speeds = [v for r in found for v in (r["v_in_mps"], r["v_out_mps"])]
        v_spread = (
            max(
                max(abs(a["v_in_mps"] - b["v_in_mps"]), abs(a["v_out_mps"] - b["v_out_mps"]))
                for a in found
                for b in found
            )
            if found
            else None
        )
        path_stable = (
            len(found) == len(by_window)
            and max(angles) - min(angles) <= TURN_ANGLE_SPREAD_DEG
            and v_spread is not None
            and v_spread <= max(0.7, 0.25 * float(np.mean(speeds)))
        )
        body_values = [b["abs_deg"] for b in body.values() if b is not None]
        body_stable = len(body_values) >= 2 and (
            max(body_values) - min(body_values) <= TURN_BODY_AGREEMENT_DEG
        )
        primary = turn.get("primary", "path")
        results.append(
            {
                **{k: turn[k] for k in ("id", "name", "before", "after") if k in turn},
                "primary": primary,
                "break_frame": turn.get("break_frame"),
                "body_frames": turn.get("body_frames"),
                "path": by_window,
                "path_stable": bool(path_stable),
                "body": body,
                "body_stable": bool(body_stable),
                "stable": bool(body_stable if primary == "body" else path_stable),
            }
        )
    return results


def video_info(path):
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,width,height,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    rate = Fraction(stream["avg_frame_rate"])
    if rate <= 0:
        raise ValueError(f"invalid FPS: {path}")
    return rate, int(stream["width"]), int(stream["height"])


def frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def transform(matrix, points):
    points = np.asarray(points, dtype=float)
    hom = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return hom[:, :2] / hom[:, 2, None]


def register_original(seed, original):
    """RANSAC image registration, checked by forward/backward optical flow."""
    a, b = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (seed, original))
    points = cv2.goodFeaturesToTrack(a, maxCorners=1200, qualityLevel=0.02, minDistance=18)
    if points is None or len(points) < 12:
        raise ValueError("too few image features to revalidate calibration")
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, points.copy(), winSize=(25, 25), maxLevel=4)
    back, good, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, nxt.copy(), winSize=(25, 25), maxLevel=4)
    keep = (
        (ok[:, 0] > 0)
        & (good[:, 0] > 0)
        & (np.linalg.norm(back[:, 0] - points[:, 0], axis=1) < 1.0)
    )
    src, dst = points[keep, 0], nxt[keep, 0]
    if len(src) < 12:
        raise ValueError("optical-flow round-trip check failed")
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if matrix is None or int(inliers.sum()) < 12:
        raise ValueError("original/seed image registration failed")
    residual = np.linalg.norm(transform(matrix, src) - dst, axis=1)[inliers[:, 0] > 0]
    return matrix, {
        "registration_inliers": int(inliers.sum()),
        "registration_rmse_px": float(np.sqrt(np.mean(residual**2))),
    }


def ball_on_original(image, seed_uv):
    """Yellow-ball component close to the supplied seed; occlusion stays absent."""
    u, v = np.rint(seed_uv).astype(int)
    h, w = image.shape[:2]
    x, y = max(0, u - 28), max(0, v - 28)
    patch = image[y : min(h, v + 29), x : min(w, u + 29)]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 100, 115]), np.array([32, 255, 255]))
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        point = centres[i] + [x, y]
        distance = np.linalg.norm(point - seed_uv)
        aspect = stats[i, cv2.CC_STAT_WIDTH] / max(1, stats[i, cv2.CC_STAT_HEIGHT])
        if 6 <= area <= 600 and distance <= 26 and 0.4 <= aspect <= 2.5:
            candidates.append((distance, point, area))
    if not candidates:
        return None
    _, point, area = max(candidates, key=lambda c: (c[2], -c[0]))
    return point, area


def smooth_segments(points, window):
    """Smooth valid contiguous pieces, without filling gaps or mixing identities."""
    result = np.asarray(points, dtype=float).copy()
    valid = np.isfinite(result).all(axis=1)
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if stop - start >= window:
            result[start:stop] = savgol_filter(result[start:stop], window, 2, axis=0)
    return result


def _read_coeffs(path, n_coeffs):
    return {
        int(r["frame"]): np.array([float(v) for k, v in r.items() if k != "frame"])[:n_coeffs]
        for r in read_csv(path)
    }


def prepare_camera(cam, participants, out):
    """Revalidate seeds on the original images; retain every provenance row."""
    out.mkdir(parents=True, exist_ok=True)
    rate, width, height = video_info(cam["video"])
    seed_frames = list(frames(cam["seed_video"]))
    mapping = read_csv(cam["frame_map"])
    if len(mapping) != len(seed_frames):
        raise ValueError("seed frame map length differs from video")
    positions = np.array([float(Fraction(r["source_position"])) - 1 for r in mapping])
    dlt3 = dlt_for_frames(cam["dlt3d"])
    dlt2 = _read_coeffs(cam["dlt2d"], 8)
    ball = {
        int(r["frame"]): np.array([float(r["p0_x"]), float(r["p0_y"])])
        for r in read_csv(cam["ball_markers"])
        if r.get("p0_x") and r.get("p0_y")
    }
    field = {int(r["frame"]): r for r in read_csv(cam["field_points"])}
    boxes = {}
    selected = {int(p["camera_ids"][cam["id"]]) for p in participants}
    for row in read_csv(cam["seed_camera"]):
        if int(row["person_id"]) in selected:
            boxes.setdefault(int(row["frame"]), []).append(row)
    names, _, xyz = load_field_points()
    rows3, rows2, tracks, balls, audit, checks = [], [], [], [], [], []
    focals = []
    start, end = cam["range_1based"]
    count = 0
    for f, image in enumerate(frames(cam["video"])):
        if f < start - 1 or f >= end:
            continue
        pair = f - start + 1
        seed = int(np.argmin(np.abs(positions - f)))
        matrix, quality = register_original(seed_frames[seed], image)
        if seed not in dlt3 or seed not in dlt2:
            raise ValueError(f"missing calibration at seed {seed}")
        P = matrix @ dlt_projection_matrix(dlt3[seed])
        P /= P[2, 3]
        H = matrix @ homography_from_dlt2d(dlt2[seed])
        H /= H[2, 2]
        rows3.append(
            dict(
                zip(["frame", *[f"L{i}" for i in range(1, 12)]], [f, *P.ravel()[:11]], strict=True)
            )
        )
        rows2.append(
            dict(zip(["frame", *[f"L{i}" for i in range(1, 9)]], [f, *H.ravel()[:8]], strict=True))
        )
        focals.append((f, focal_from_dlt3d(P.ravel()[:11], width)))
        # Independently track accepted field pixels from seed to original.
        known = field[seed]
        indices, pixels = [], []
        for k in range(len(names)):
            u, v = known.get(f"p{k}_x"), known.get(f"p{k}_y")
            if u and v and np.isfinite([float(u), float(v)]).all():
                indices.append(k)
                pixels.append([float(u), float(v)])
        errors = []
        if pixels:
            a, b = (cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (seed_frames[seed], image))
            pts = np.asarray(pixels, np.float32).reshape(-1, 1, 2)
            moved, status, _ = cv2.calcOpticalFlowPyrLK(
                a, b, pts, pts.copy(), winSize=(31, 31), maxLevel=4
            )
            back, valid, _ = cv2.calcOpticalFlowPyrLK(
                b, a, moved, moved.copy(), winSize=(31, 31), maxLevel=4
            )
            reproj = transform(P, xyz)
            for j, k in enumerate(indices):
                good = bool(
                    status[j, 0] and valid[j, 0] and np.linalg.norm(back[j, 0] - pts[j, 0]) <= 1.5
                )
                error = float(np.linalg.norm(reproj[k] - moved[j, 0])) if good else None
                checks.append(
                    {
                        "frame": f,
                        "pair": pair,
                        "point": k,
                        "name": names[k],
                        "u_px": float(moved[j, 0, 0]) if good else None,
                        "v_px": float(moved[j, 0, 1]) if good else None,
                        "validated_flow": good,
                        "reprojection_px": error,
                    }
                )
                if error is not None:
                    errors.append(error)
        uv = ball_on_original(image, transform(matrix, [ball[seed]])[0]) if seed in ball else None
        balls.append(
            {
                "frame": f,
                "p0_x": uv[0][0] if uv else None,
                "p0_y": uv[0][1] if uv else None,
                "method": "yellow_component_original" if uv else "absent",
                "area_px": uv[1] if uv else None,
            }
        )
        for row in boxes.get(seed, []):
            x1, y1, x2, y2 = [
                float(row[k]) for k in ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")
            ]
            corners = transform(matrix, [[x1, y1], [x2, y2]])
            tracks.append(
                {
                    "frame": f,
                    "obj_id": int(row["person_id"]),
                    "x1": corners[0, 0],
                    "y1": corners[0, 1],
                    "x2": corners[1, 0],
                    "y2": corners[1, 1],
                    "area_px": max(0, int(np.prod(corners[1] - corners[0]))),
                    "score": 1.0,
                }
            )
        audit.append(
            {
                "pair": pair,
                "original_frame_1based": f + 1,
                "original_time_rational": str(Fraction(pair, 1) / rate),
                "seed_frame_0based": seed,
                "seed_kind": mapping[seed]["kind"],
                "seed_source_position": mapping[seed]["source_position"],
                "measurement_image": "original",
                "field_points_checked": len(errors),
                "field_rmse_px": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                **quality,
            }
        )
        if len(errors) < 4 or np.sqrt(np.mean(np.square(errors))) > 10:
            raise ValueError(f"field revalidation failed at original frame {f + 1} of {cam['id']}")
        count += 1
        if pair % 20 == 0:
            print(f">> original {cam['id']}: {pair + 1}/{end - start + 1}", flush=True)
    if count != end - start + 1:
        raise ValueError("original video is shorter than the specified range")
    for name, records in (
        ("camera.dlt3d", rows3),
        ("camera.dlt2d", rows2),
        ("sam_tracks.csv", tracks),
        ("ball_markers.csv", balls),
        ("original_frames.csv", audit),
        ("field_validation.csv", checks),
    ):
        write_csv(out / name, records)
    full_focals = np.full(end, np.nan)
    for f, focal in focals:
        if focal is not None:
            full_focals[f] = focal
    np.save(out / "cam_int.npy", cam_int_sequence(full_focals, width, height))
    return audit


def rerun_body(cam, out):
    """Invoke the existing lifter in a disposable process, preserving selected SAM IDs."""
    command = [
        sys.executable,
        "-m",
        "vaila.freekiki_openplay_manifest",
        "sam-worker",
        str(out / "worker.json"),
    ]
    write_json(out / "worker.json", {"camera": cam, "out": str(out)})
    print(">> Equivalent CLI: " + __import__("shlex").join(command), flush=True)
    with (out / "sam_worker.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def sam_worker(path):
    # Reuse the established GPU environment and SAM3D input/output contracts.
    from .gpu_subprocess import reexec_self_if_nvrtc_env_missing

    reexec_self_if_nvrtc_env_missing()
    from .sam3dinov3 import _build_parser, run_sam3d_from_sam
    from .sam3sapiens2 import SamGuidance

    spec = json.loads(Path(path).read_text())
    cam, out = spec["camera"], Path(spec["out"])
    rate, width, height = video_info(cam["video"])
    tracks = {}
    for r in read_csv(out / "sam_tracks.csv"):
        row = {k: float(v) for k, v in r.items()}
        row["obj_id"], row["area_px"] = int(row["obj_id"]), int(row["area_px"])
        row.update(
            x_px=row["x1"], y_px=row["y1"], w_px=row["x2"] - row["x1"], h_px=row["y2"] - row["y1"]
        )
        tracks.setdefault(int(row["frame"]), []).append(row)
    guidance = SamGuidance(
        out, tracks, {}, width, height, float(rate), cam["range_1based"][1], None
    )
    args = _build_parser().parse_args(
        [
            "--no-mask",
            "--no-overlay",
            "--max-persons",
            "4",
            "--min-sam-area",
            "0",
            "--cam-int-npy",
            str(out / "cam_int.npy"),
        ]
    )
    run_sam3d_from_sam(Path(cam["video"]), out / "sam3d", guidance, args)


def load_poses(path):
    result = {}
    for r in read_csv(path):
        point = np.array([float(r["x_px"]), float(r["y_px"])])
        if np.isfinite(point).all():
            result.setdefault((int(r["frame"]), int(r["person_id"])), {})[r["kpt_name"]] = point
    return result


def analyze_originals(manifest, out):
    """Triangulate the ball and estimate player ground positions from original poses."""
    cams, people, n = manifest["cameras"], manifest["participants"], manifest["n_frames"]
    rate = video_info(cams[0]["video"])[0]
    if video_info(cams[1]["video"])[0] != rate:
        raise ValueError("paired-original mode requires equal original FPS")
    dlts, homographies, balls, poses = [], [], [], []
    for cam in cams:
        folder = out / "cameras" / cam["id"]
        dlts.append(dlt_for_frames(folder / "camera.dlt3d"))
        homographies.append(
            {
                f: homography_from_dlt2d(c)
                for f, c in _read_coeffs(folder / "camera.dlt2d", 8).items()
            }
        )
        balls.append(
            {
                int(r["frame"]): np.array([float(r["p0_x"]), float(r["p0_y"])])
                for r in read_csv(folder / "ball_markers.csv")
                if r["p0_x"] and r["p0_y"]
            }
        )
        poses.append(
            load_poses(folder / "sam3d" / f"{Path(cam['video']).stem}_sam3dinov3_keypoints2d.csv")
        )
    raw = {p["id"]: np.full((n, 2), np.nan) for p in people}
    ball_xyz = np.full((n, 3), np.nan)
    quality, ground_rows = [], []
    for pair in range(n):
        fs = [cam["range_1based"][0] - 1 + pair for cam in cams]
        row = {
            "pair": pair,
            "frame_open_1based": fs[0] + 1,
            "frame_main_1based": fs[1] + 1,
            "t_s": pair / float(rate),
            "reprojection_open_px": None,
            "reprojection_main_px": None,
            "ray_angle_deg": None,
            "accepted": False,
        }
        if all(f in b and f in d for f, b, d in zip(fs, balls, dlts, strict=True)):
            try:
                xyz, ra, rb, angle = triangulate(
                    dlts[0][fs[0]], dlts[1][fs[1]], balls[0][fs[0]], balls[1][fs[1]]
                )
                row.update(
                    x_raw_m=xyz[0],
                    y_raw_m=xyz[1],
                    z_raw_m=xyz[2],
                    reprojection_open_px=ra,
                    reprojection_main_px=rb,
                    ray_angle_deg=angle,
                )
                accepted = bool(
                    np.isfinite(xyz).all()
                    and max(ra, rb) <= 10
                    and angle >= 1
                    and -0.1 <= xyz[2] <= 4
                )
                row["accepted"] = accepted
                if accepted:
                    ball_xyz[pair] = xyz
            except (ValueError, np.linalg.LinAlgError):
                pass
        quality.append(row)
        for p in people:
            observations = []
            for ci, (f, cam) in enumerate(zip(fs, cams, strict=True)):
                kp = poses[ci].get((f, int(p["camera_ids"][cam["id"]])), {})
                feet = [kp[name] for name in ("left-ankle", "right-ankle") if name in kp]
                xy = None
                if len(feet) == 2 and f in homographies[ci]:
                    xy = transform(np.linalg.inv(homographies[ci][f]), [np.mean(feet, axis=0)])[0]
                    if np.isfinite(xy).all():
                        observations.append((ci, xy))
                ground_rows.append(
                    {
                        "pair": pair,
                        "camera": cam["id"],
                        "original_frame_1based": f + 1,
                        "participant": p["id"],
                        "sam_id": p["camera_ids"][cam["id"]],
                        "x_m": xy[0] if xy is not None else None,
                        "y_m": xy[1] if xy is not None else None,
                        "method": "SAM3D_ankle_mean_on_DLT2D_plane",
                    }
                )
            # Use the main camera consistently, rather than switching/fusing noisy views.
            main = next((xy for ci, xy in observations if cams[ci].get("main")), None)
            if main is not None:
                raw[p["id"]][pair] = main
    write_csv(
        out / "data" / "ball_reprojection.csv",
        quality,
        list(dict.fromkeys(k for r in quality for k in r)),
    )
    write_csv(out / "data" / "ground_observations.csv", ground_rows)
    write_csv(
        out / "data" / "raw_tracks.csv",
        [
            {
                "pair": f,
                "t_s": f / float(rate),
                "participant": p["id"],
                "x_m": raw[p["id"]][f, 0],
                "y_m": raw[p["id"]][f, 1],
            }
            for p in people
            for f in range(n)
        ],
    )
    dribbler = next(p for p in people if p["role"] == "dribbler")
    packs: dict[str, dict[str, Any]] = {}
    dribbler_paths: dict[str, np.ndarray] = {}
    for window in (5, 7, 9):
        paths = {pid: smooth_segments(xy, window) for pid, xy in raw.items()}
        ball_smooth = smooth_segments(ball_xyz, window)
        drib = paths[dribbler["id"]]
        dribbler_paths[str(window)] = drib
        pack = {
            "kind": "dribble",
            "fps": float(rate),
            "fps_rational": str(rate),
            "n_frames": n,
            "dribbler_id": dribbler["id"],
            "ball": track_metrics(ball_smooth[:, :2], float(rate)),
            "dribbler": track_metrics(drib, float(rate)),
            "markers": [],
            "ball_xyz_m": ball_smooth,
            "ball_raw_xyz_m": ball_xyz,
            "window": window,
        }
        pack["ball"]["distance_to_dribbler_m"] = np.linalg.norm(ball_smooth[:, :2] - drib, axis=1)
        spatial_speed = np.r_[
            np.nan, np.linalg.norm(np.diff(ball_smooth, axis=0), axis=1) * float(rate)
        ]
        pack["ball"]["speed_3d_mps"] = spatial_speed
        pack["ball"]["accel_3d_mps2"] = np.r_[np.nan, np.diff(spatial_speed) * float(rate)]
        pack["ball"]["distance_to_ground_anchor_3d_m"] = np.linalg.norm(
            ball_smooth - np.column_stack((drib, np.zeros(n))), axis=1
        )
        for p in people:
            if p["role"] != "dribbler":
                pack["markers"].append(
                    {
                        "person_id": p["id"],
                        "name": p["name"],
                        "metrics": track_metrics(paths[p["id"]], float(rate)),
                        "vs_dribbler": pair_approach(drib, paths[p["id"]], float(rate)),
                    }
                )
        packs[str(window)] = pack
    analysis = packs["7"]
    analysis["filter_sensitivity"] = {
        w: {"dribbler": p["dribbler"], "ball": p["ball"]} for w, p in packs.items()
    }
    analysis["participants"] = people
    if manifest.get("turns"):
        body_yaws = {
            cam["id"]: body_yaw_series(
                out / "cameras" / cam["id"], cam, dribbler["camera_ids"][cam["id"]], n
            )
            for cam in cams
        }
        analysis["turns"] = analyze_turns(manifest["turns"], dribbler_paths, body_yaws, float(rate))
    analysis["quality"] = {
        "ball_accepted_pairs": int(np.isfinite(ball_xyz).all(axis=1).sum()),
        "total_pairs": n,
        "calibration": "supplied DLT transferred by original-image registration; inherited metric scale and height calibration",
        "ball_height_validated": False,
        "ground_method": "main-camera SAM3D ankle midpoint projected on the pitch, model estimate",
        "measurement_clock": "paired original frames; RIFE seeds are presentation/reference only",
    }
    write_json(out / "data" / "dribble_analysis.json", analysis)
    for w, pack in packs.items():
        entities = [
            (dribbler["id"], pack["dribbler"]),
            ("ball", pack["ball"]),
            *[(r["person_id"], r["metrics"]) for r in pack["markers"]],
        ]
        series = []
        for pid, m in entities:
            for f in range(n):
                series.append(
                    {
                        "pair": f,
                        "t_s": f / float(rate),
                        "participant": pid,
                        "window_frames": w,
                        "x_m": m["xy_m"][f][0],
                        "y_m": m["xy_m"][f][1],
                        **{
                            k: m[k][f]
                            for k in (
                                "speed_mps",
                                "accel_mps2",
                                "heading_rad",
                                "curvature_rad_per_m",
                            )
                        },
                    }
                )
        write_csv(out / "data" / f"series_window_{w}.csv", series)
        interactions = []
        for f in range(n):
            interactions.append(
                {
                    "pair": f,
                    "t_s": f / float(rate),
                    "participant": "ball",
                    "distance_to_dribbler_xy_m": pack["ball"]["distance_to_dribbler_m"][f],
                    "distance_to_ground_anchor_3d_m": pack["ball"][
                        "distance_to_ground_anchor_3d_m"
                    ][f],
                    "speed_3d_mps": pack["ball"]["speed_3d_mps"][f],
                    "accel_3d_mps2": pack["ball"]["accel_3d_mps2"][f],
                }
            )
            for row in pack["markers"]:
                interactions.append(
                    {
                        "pair": f,
                        "t_s": f / float(rate),
                        "participant": row["person_id"],
                        "distance_to_dribbler_xy_m": row["vs_dribbler"]["distance_m"][f],
                        "separation_rate_mps": row["vs_dribbler"]["rel_speed_mps"][f],
                    }
                )
        write_csv(
            out / "data" / f"interactions_window_{w}.csv",
            interactions,
            list(dict.fromkeys(k for r in interactions for k in r)),
        )
    return clean(analysis), poses


def fingerprint_inputs(manifest):
    paths = {manifest["manifest_path"]}
    for cam in manifest["cameras"]:
        paths.update(
            cam[k]
            for k in (
                "video",
                "seed_video",
                "frame_map",
                "dlt2d",
                "dlt3d",
                "ball_markers",
                "field_points",
                "seed_camera",
                "seed_keypoints",
            )
        )
    result = []
    for name in sorted(paths):
        path = Path(name)
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(block)
        result.append({"path": name, "sha256": digest.hexdigest(), "size": path.stat().st_size})
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "sam-worker":
        sam_worker(sys.argv[2])
    else:
        raise SystemExit("Internal worker; use freekiki.py openplay-report --manifest …")
