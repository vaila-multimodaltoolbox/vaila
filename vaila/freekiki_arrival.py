"""Observed monocular arrivals and goal-plane calibration diagnostics.

Version: 0.4.14
Update Date: 02 November 2026

An entry is a manually reviewed ball centre on the goal plane, with the four goal
corners from that same frame. It is not inferred from the last frame or a save.
All world coordinates come from the unchanged kiki49 reference.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .freekiki_diag import load_field_points
    from .monocular_dlt_align import decompose_dlt3d, dlt_projection_matrix
    from .pynalty_analysis import dlt2d, rec2d
except ImportError:
    from freekiki_diag import load_field_points  # ty: ignore[unresolved-import]
    from monocular_dlt_align import decompose_dlt3d, dlt_projection_matrix  # ty: ignore[unresolved-import]
    from pynalty_analysis import dlt2d, rec2d  # ty: ignore[unresolved-import]

CALIBRATION_MODEL = "confirmed-goal-v1"
GOAL_IDS = {-1: (32, 33, 34, 35), 1: (40, 41, 42, 43)}


def validate_observation(data: dict, n_frames: int | None = None) -> dict:
    """Validate and normalize an observation without silently rounding frame numbers."""
    if not isinstance(data, dict) or data.get("type") not in ("goal_entry", "keeper_hand"):
        raise ValueError("arrival type must be goal_entry or keeper_hand")
    frame = data.get("frame")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError("arrival frame must be a 0-based integer >= 0")
    if n_frames is not None and frame >= n_frames:
        raise ValueError(f"arrival frame {frame} is outside the video ({n_frames} frames)")
    sigma = float(data.get("sigma_m", 0.2 if data["type"] == "goal_entry" else 0.3))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("arrival sigma_m must be finite and > 0")
    out = {"type": data["type"], "frame": frame, "sigma_m": sigma, "provenance": "manual"}
    if data["type"] == "keeper_hand":
        if data.get("confirmed") is not True:
            raise ValueError("a keeper-hand arrival requires confirmed: true")
        xyz = np.asarray(data.get("xyz"), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("confirmed keeper-hand arrival needs xyz in pitch metres")
        out.update(xyz=xyz.tolist(), confirmed=True)
        if "person_id" in data:
            out["person_id"] = int(data["person_id"])
        return out
    side = data.get("goal_side")
    if isinstance(side, bool) or side not in GOAL_IDS:
        raise ValueError("goal_side must be -1 (p32..p35) or +1 (p40..p43)")
    uv = np.asarray(data.get("pixel"), dtype=float)
    corners = data.get("corners_px")
    if not isinstance(corners, dict) or set(corners) != {f"p{i}" for i in GOAL_IDS[side]}:
        raise ValueError(f"corners_px needs exactly {[f'p{i}' for i in GOAL_IDS[side]]}")
    xy = np.asarray([corners[f"p{i}"] for i in GOAL_IDS[side]], dtype=float)
    if uv.shape != (2,) or xy.shape != (4, 2) or not np.isfinite(np.r_[uv, xy.ravel()]).all():
        raise ValueError("ball pixel and the four corners must be finite pixel pairs")
    # A crossed or collapsed quadrilateral cannot represent the goal rectangle.
    polygon = xy[[0, 1, 3, 2]]
    edges = np.roll(polygon, -1, axis=0) - polygon
    cross = (
        edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1]
        - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    )
    if not (np.all(cross > 1e-3) or np.all(cross < -1e-3)):
        raise ValueError("goal corners are collapsed or crossed; review their kiki49 identities")
    pixel_sigma = float(data.get("pixel_sigma", 2.0))
    if not np.isfinite(pixel_sigma) or pixel_sigma <= 0:
        raise ValueError("pixel_sigma must be finite and > 0")
    out.update(
        goal_side=int(side),
        pixel=uv.tolist(),
        corners_px={f"p{i}": p.tolist() for i, p in zip(GOAL_IDS[side], xy, strict=True)},
        pixel_sigma=pixel_sigma,
    )
    return out


def load_observation(value, n_frames: int | None = None) -> dict | None:
    if value is None or value == "none":
        return None
    data = (
        json.loads(Path(value).expanduser().read_text(encoding="utf-8"))
        if isinstance(value, (str, Path))
        else value
    )
    return validate_observation(data, n_frames)


def save_observation(path: Path, data: dict) -> Path:
    """Keep every replaced manual observation as a sibling backup in corrections/."""
    normalized = validate_observation(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        backup = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
        shutil.copy2(path, backup)
    path.write_text(json.dumps(normalized, indent=2, allow_nan=False), encoding="utf-8")
    return path


def calibration_key(observation: dict | None) -> dict | None:
    """Only the confirmed corners change the camera; moving the ball alone reuses SAM."""
    if observation is None or observation["type"] != "goal_entry":
        return None
    return {k: observation[k] for k in ("frame", "goal_side", "corners_px")} | {
        "model": CALIBRATION_MODEL
    }


def goal_geometry(observation: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, xyz = load_field_points()
    world = np.asarray(xyz, dtype=float)[list(GOAL_IDS[observation["goal_side"]])]
    xy = np.asarray(
        [observation["corners_px"][f"p{i}"] for i in GOAL_IDS[observation["goal_side"]]],
        dtype=float,
    )
    H = dlt2d(world[:, 1:], xy)
    return world, xy, H


def entry_anchor(observation: dict, projection: np.ndarray) -> dict:
    """Local goal homography, propagated ball-pixel uncertainty and global-camera residuals."""
    world, xy, H = goal_geometry(observation)
    uv = np.asarray(observation["pixel"], dtype=float)
    yz = rec2d(H, uv.reshape(1, 2))[0]
    if not np.isfinite(yz).all():
        raise ValueError("goal homography is singular; review the four corners")
    # Linear propagation of the observed centre's pixel uncertainty. Corner/calibration and
    # frame uncertainty are represented by the separate sigma_m floor, not claimed measured.
    J = np.column_stack(
        [
            (rec2d(H, (uv + delta).reshape(1, 2))[0] - rec2d(H, (uv - delta).reshape(1, 2))[0]) / 2
            for delta in np.eye(2)
        ]
    )
    sigma = max(
        observation["sigma_m"], float(np.linalg.norm(J, ord=2)) * observation["pixel_sigma"]
    )
    h = np.c_[world, np.ones(4)] @ projection.T
    projected = h[:, :2] / h[:, 2:3]
    errors = np.linalg.norm(projected - xy, axis=1)
    # Inverse of the actual global camera restricted to x = x_goal.
    matrix = np.column_stack(
        [projection[:, 1], projection[:, 2], projection[:, 0] * world[0, 0] + projection[:, 3]]
    )
    global_h = np.linalg.solve(matrix, np.r_[uv, 1.0])
    global_yz = global_h[:2] / global_h[2]
    return {
        "type": "entry",
        "frame": observation["frame"],
        "xyz": [float(world[0, 0]), *yz.tolist()],
        "sigma": sigma,
        "label": "chegada: entrada no gol (observada; plano da baliza)",
        "provenance": "manual",
        "observation": observation,
        "diagnostics": {
            "goal_corner_errors_px": {
                f"p{i}": float(e)
                for i, e in zip(GOAL_IDS[observation["goal_side"]], errors, strict=True)
            },
            "global_yz_m": global_yz.tolist(),
            "local_yz_m": yz.tolist(),
            "global_local_gap_m": float(np.linalg.norm(global_yz - yz)),
        },
    }


def refit_goal_calibration(run_dir: Path, observation: dict, width: int, height: int) -> dict:
    """Refit the pinhole with confirmed corners plus valid field points, retaining a point audit.

    The reference coordinates are never changed. The existing fit supplies the initial pose
    and a 15 px gate for field points. All four confirmed goal corners participate; none can be
    pruned. Their uncertainty is 1 px, other accepted field points 3 px. Residuals are saved even
    when the confirmed frame cannot support a consistent camera. Only then are DLTs replaced.
    """
    import cv2
    from scipy.optimize import least_squares

    try:
        from .freekiki_detect3d import _read_field_pixels, dlt_for_frames
    except ImportError:
        from freekiki_detect3d import _read_field_pixels, dlt_for_frames  # ty: ignore[unresolved-import]

    run_dir = Path(run_dir)
    d3 = next(run_dir.glob("*.dlt3d"))
    cameras = dlt_for_frames(d3)
    pixels = _read_field_pixels(run_dir / "field_kps_getpixelvideo.csv")
    _, _, xyz = load_field_points()
    xyz = np.asarray(xyz, dtype=float)
    ids_goal = GOAL_IDS[observation["goal_side"]]
    fitted, audit, frames = {}, [], []
    for frame, L in sorted(cameras.items()):
        xy = pixels.get(frame)
        if xy is None:
            continue
        xy = xy.copy()
        if frame == observation["frame"]:
            xy[list(ids_goal)] = np.asarray([observation["corners_px"][f"p{i}"] for i in ids_goal])
        finite = np.isfinite(xy).all(axis=1)
        P0 = dlt_projection_matrix(L)
        h0 = np.c_[xyz, np.ones(len(xyz))] @ P0.T
        uv0 = h0[:, :2] / h0[:, 2:3]
        before = np.linalg.norm(uv0 - xy, axis=1)
        use = finite & (xyz[:, 2] == 0) & (before <= 15.0 * width / 1920)
        use[list(ids_goal)] = finite[list(ids_goal)]
        if use.sum() < 7 or not finite[list(ids_goal)].all():
            fitted[frame] = P0
            frames.append({"frame": frame, "status": "insufficient goal/field points"})
            continue
        cam = decompose_dlt3d(L)
        rv, _ = cv2.Rodrigues(cam.R)
        start = np.r_[rv.ravel(), -cam.R @ cam.C, np.log(0.5 * (cam.K[0, 0] + cam.K[1, 1]))]
        used_ids = np.flatnonzero(use)
        sigma = np.asarray([1.0 if i in ids_goal else 3.0 for i in used_ids]) * width / 1920

        def projection(params):
            R, _ = cv2.Rodrigues(params[:3])
            f = np.exp(params[6])
            K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]])
            return K @ np.c_[R, params[3:6]]

        def residuals(params, use=use, xy=xy, sigma=sigma):
            h = np.c_[xyz[use], np.ones(use.sum())] @ projection(params).T
            return ((h[:, :2] / h[:, 2:3] - xy[use]) / sigma[:, None]).ravel()

        lo = np.r_[np.full(6, -np.inf), np.log(0.3 * width)]
        hi = np.r_[np.full(6, np.inf), np.log(10 * width)]
        sol = least_squares(residuals, np.clip(start, lo, hi), bounds=(lo, hi), max_nfev=400)
        P = projection(sol.x)
        h = np.c_[xyz, np.ones(len(xyz))] @ P.T
        uv = h[:, :2] / h[:, 2:3]
        errors = np.linalg.norm(uv - xy, axis=1)
        consistent = bool(
            sol.success
            and np.all(h[use, 2] > 0)
            and np.sqrt(np.mean(errors[use] ** 2)) <= 8 * width / 1920
            and errors[list(ids_goal)].max() <= 3 * width / 1920
        )
        frames.append(
            {
                "frame": frame,
                "status": "ok" if consistent else "review points",
                "rms_px": float(np.sqrt(np.mean(errors[use] ** 2))),
                "goal_max_px": float(errors[list(ids_goal)].max()),
            }
        )
        fitted[frame] = P
        for i in np.flatnonzero(finite):
            audit.append(
                {
                    "frame": frame,
                    "point": f"p{i}",
                    "used": bool(use[i]),
                    "confirmed": frame == observation["frame"] and i in ids_goal,
                    "x_m": xyz[i, 0],
                    "y_m": xyz[i, 1],
                    "z_m": xyz[i, 2],
                    "observed_u": xy[i, 0],
                    "observed_v": xy[i, 1],
                    "before_px": before[i],
                    "after_px": errors[i],
                }
            )
    with (run_dir / "goal_calibration_residuals.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(audit[0]) if audit else ["frame", "point", "used"]
        )
        writer.writeheader()
        writer.writerows(audit)
    summary = {
        "model": CALIBRATION_MODEL,
        "observation": calibration_key(observation),
        "frames": frames,
        "reference": "soccerfield_kiki.csv (unchanged)",
        "field_sigma_px_at_1920": 3,
        "goal_sigma_px_at_1920": 1,
    }
    (run_dir / "goal_calibration.json").write_text(json.dumps(summary, indent=2))
    marked = next((r for r in frames if r["frame"] == observation["frame"]), {})
    if marked.get("status") != "ok":
        raise ValueError(
            f"Confirmed goal frame needs point review: {marked}. See {run_dir / 'goal_calibration_residuals.csv'}"
        )
    # Keep the detector's camera files within this new detect run for audit/recovery.
    backup = run_dir / "before_goal_refit"
    backup.mkdir(exist_ok=True)
    accepted = run_dir / "field_kps_getpixelvideo.csv"
    shutil.copy2(accepted, backup / accepted.name)
    marked_pixels = pixels[observation["frame"]].copy()
    marked_pixels[list(ids_goal)] = np.asarray(
        [observation["corners_px"][f"p{i}"] for i in ids_goal]
    )
    pixels[observation["frame"]] = marked_pixels
    with accepted.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame"] + [f"p{i}_{c}" for i in range(len(xyz)) for c in "xy"])
        for frame, xy in sorted(pixels.items()):
            writer.writerow([frame, *xy.ravel()])
    for path, dims in ((d3, 3), (d3.with_suffix(".dlt2d"), 2)):
        if path.exists():
            shutil.copy2(path, backup / path.name)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["frame"] + [f"L{i}" for i in range(1, 12 if dims == 3 else 9)])
            for frame, P in sorted(fitted.items()):
                matrix = P if dims == 3 else P[:, [0, 1, 3]]
                coeffs = (matrix / matrix[-1, -1]).ravel()[:-1]
                writer.writerow([frame, *coeffs])
    return summary


def show_arrival_dialog(parent, camera: dict, *, on_saved=None):
    """Review the actual crossing frame, ball and corners, then persist one manual observation.

    Clicking the preview moves the selected point in original video pixels. Existing marks
    are drafts until Save is clicked. A save requires a separate confirmation checkbox.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    import cv2
    from PIL import Image, ImageTk

    try:
        from .freekiki_detect3d import _read_field_pixels, read_corrected_ball
    except ImportError:
        from freekiki_detect3d import _read_field_pixels, read_corrected_ball  # ty: ignore[unresolved-import]

    video = Path(camera["video"])
    target = Path(camera["run"]) / "corrections" / video.stem / "arrival_observation.json"
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    field_path = video.with_name(f"{video.stem}_kiki49.csv")
    if not field_path.is_file() and camera.get("predict"):
        field_path = Path(camera["predict"]) / "field_kps_getpixelvideo.csv"
    field = _read_field_pixels(field_path) if field_path.is_file() else {}
    ball_path = video.with_name(f"{video.stem}_markers.csv")
    ball = {
        int(r["frame"]): (r["cx"], r["cy"])
        for r in read_corrected_ball(ball_path, n)
        if r["cx"] != ""
    }
    previous = load_observation(target, n) if target.is_file() else None
    win = tk.Toplevel(parent)
    win.title("Observed arrival — review the crossing frame")
    frame_var = tk.IntVar(win, previous["frame"] if previous else max(1, n - 2))
    kind = tk.StringVar(win, previous["type"] if previous else "goal_entry")
    side = tk.IntVar(win, previous.get("goal_side", -1) if previous else -1)
    selected = tk.StringVar(win, "ball")
    sigma = tk.StringVar(win, str(previous["sigma_m"] if previous else 0.2))
    confirmed = tk.BooleanVar(win, False)
    zoom = tk.BooleanVar(win, True)
    coords = {
        name: [tk.StringVar(win, ""), tk.StringVar(win, "")]
        for name in ("ball", "base 1", "base 2", "top 1", "top 2")
    }
    hand_xyz = [tk.StringVar(win, str(v)) for v in ((previous or {}).get("xyz") or [0, 0, 0])]
    info = tk.StringVar(win, "")
    panel = ttk.Frame(win, padding=8)
    panel.pack(fill="both", expand=True)
    toolbar = ttk.Frame(panel)
    toolbar.pack(fill="x")
    ttk.Label(toolbar, text="Frame (0-based)").pack(side="left")
    spin = ttk.Spinbox(toolbar, from_=0, to=max(0, n - 1), textvariable=frame_var, width=6)
    spin.pack(side="left", padx=4)
    ttk.Combobox(
        toolbar, textvariable=kind, values=("goal_entry", "keeper_hand"), state="readonly", width=14
    ).pack(side="left", padx=6)
    for label, value in (("Left goal (32–35)", -1), ("Right goal (40–43)", 1)):
        ttk.Radiobutton(
            toolbar, text=label, variable=side, value=value, command=lambda: load_frame()
        ).pack(side="left")
    canvas = tk.Canvas(panel, width=960, height=480, background="#111")
    canvas.pack()
    rows = ttk.Frame(panel)
    rows.pack(fill="x")
    for name, values in coords.items():
        cell = ttk.Frame(rows)
        cell.pack(side="left", padx=5)
        ttk.Radiobutton(cell, text=name, variable=selected, value=name).pack()
        for var in values:
            ttk.Entry(cell, textvariable=var, width=8).pack(side="left")
    options = ttk.Frame(panel)
    options.pack(fill="x", pady=6)
    ttk.Checkbutton(options, text="Zoom goal", variable=zoom, command=lambda: draw()).pack(
        side="left"
    )
    ttk.Label(options, text="Uncertainty floor (m)").pack(side="left", padx=(15, 4))
    ttk.Entry(options, textvariable=sigma, width=6).pack(side="left")
    ttk.Label(options, text="Confirmed hand xyz (m), only for keeper_hand:").pack(
        side="left", padx=8
    )
    for var in hand_xyz:
        ttk.Entry(options, textvariable=var, width=8).pack(side="left")
    ttk.Checkbutton(
        panel, text="I confirm ball–hand contact (keeper_hand only)", variable=confirmed
    ).pack(anchor="w")
    ttk.Label(panel, textvariable=info, wraplength=940).pack(anchor="w")
    state = {}

    def draw():
        if "image" not in state:
            return
        image = state["image"].copy()
        pts = {}
        for name, variables in coords.items():
            try:
                pts[name] = np.array([float(v.get()) for v in variables])
            except ValueError:
                continue
        x0 = y0 = 0
        x1, y1 = image.shape[1], image.shape[0]
        if zoom.get() and pts:
            arr = np.array(list(pts.values()))
            x0, y0 = np.maximum(0, np.floor(arr.min(axis=0) - 70)).astype(int)
            x1, y1 = np.minimum([x1, y1], np.ceil(arr.max(axis=0) + 70)).astype(int)
        if x1 <= x0 or y1 <= y0:
            return
        crop = image[y0:y1, x0:x1]
        scale = min(960 / crop.shape[1], 480 / crop.shape[0])
        display = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)))
        for name, p in pts.items():
            x, y = np.rint((p - [x0, y0]) * scale).astype(int)
            color = (0, 255, 255) if name != "ball" else (0, 60, 255)
            cv2.circle(display, (x, y), 5, color, 1)
            cv2.putText(display, name, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        state["transform"] = (x0, y0, scale)
        state["photo"] = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB)), master=win
        )
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])

    def load_frame(*_):
        try:
            f = int(frame_var.get())
            if not 0 <= f < n:
                raise ValueError(f"Frame must be within 0..{n - 1}")
            cap = cv2.VideoCapture(str(video))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cap.read()
            finally:
                cap.release()
            if not ok:
                raise ValueError(f"Cannot read frame {f}")
            state["image"] = im
            state["loaded_frame"] = f
            points = [ball.get(f)]
            field_row = field.get(f)
            points += [
                field_row[i] if field_row is not None else None for i in GOAL_IDS[side.get()]
            ]
            for (_name, variables), p in zip(coords.items(), points, strict=True):
                for c, var in enumerate(variables):
                    var.set(
                        f"{float(p[c]):.2f}" if p is not None and np.isfinite(float(p[c])) else ""
                    )
            confirmed.set(False)
            info.set(
                f"Frame {f}. Draft corners: {list(GOAL_IDS[side.get()])}. Select ball/base/top, then click to adjust. Confirm the crossing frame, not a later position in the net. Save confirms these goal marks; keeper contact requires its checkbox."
            )
            draw()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Observed arrival", str(exc), parent=win)

    def clicked(event):
        if "transform" not in state:
            return
        x0, y0, scale = state["transform"]
        x, y = x0 + event.x / scale, y0 + event.y / scale
        h, w = state["image"].shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        for var, value in zip(coords[selected.get()], (x, y), strict=True):
            var.set(f"{value:.2f}")
        draw()

    def save():
        try:
            f = int(frame_var.get())
            if f != state.get("loaded_frame"):
                raise ValueError("Load the chosen frame before saving its marks")
            data = {"type": kind.get(), "frame": f, "sigma_m": float(sigma.get())}
            if kind.get() == "goal_entry":
                data.update(
                    goal_side=side.get(),
                    pixel=[float(v.get()) for v in coords["ball"]],
                    corners_px={
                        f"p{i}": [float(v.get()) for v in coords[name]]
                        for i, name in zip(GOAL_IDS[side.get()], list(coords)[1:], strict=True)
                    },
                )
                validated = validate_observation(data, n)
                world, _xy, H = goal_geometry(validated)
                yz = rec2d(H, np.array([validated["pixel"]]))[0]
                if abs(yz[0]) > 3.86 or not -0.2 <= yz[1] <= 2.64:
                    raise ValueError(
                        f"Marked entry outside the goal: x={world[0, 0]:.2f}, y={yz[0]:.2f}, z={yz[1]:.2f} m. Review frame/ball/corners."
                    )
            else:
                data.update(xyz=[float(v.get()) for v in hand_xyz], confirmed=confirmed.get())
                validate_observation(data, n)
            save_observation(target, data)
            print(f">> Arrival observation saved: {target}", flush=True)
            if on_saved:
                on_saved(target)
            win.destroy()
        except (ValueError, OSError, tk.TclError) as exc:
            messagebox.showerror("Observed arrival", str(exc), parent=win)

    spin.configure(command=load_frame)
    spin.bind("<Return>", load_frame)
    canvas.bind("<Button-1>", clicked)
    actions = ttk.Frame(panel)
    actions.pack(fill="x", pady=4)
    ttk.Button(actions, text="Load frame / refresh marks", command=load_frame).pack(side="left")
    ttk.Button(actions, text="Preview edits", command=draw).pack(side="left", padx=8)
    ttk.Button(actions, text="Save observed arrival", command=save).pack(side="right")
    load_frame()
    if previous and previous["type"] == "goal_entry":
        for var, v in zip(coords["ball"], previous["pixel"], strict=True):
            var.set(str(v))
        for i, name in zip(GOAL_IDS[previous["goal_side"]], list(coords)[1:], strict=True):
            for var, v in zip(coords[name], previous["corners_px"][f"p{i}"], strict=True):
                var.set(str(v))
        draw()
    return win
