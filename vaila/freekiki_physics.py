"""
================================================================================
Script: freekiki_physics.py - physically consistent tracks of the ball and the players
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 26 October 2026
Update Date: 30 October 2026

Description:
    A track of pitch positions (metres, one point per frame) cannot jump: a ball does
    not cross 30 m in a frame and a body does not change speed by 100 m/s in 1/30 s.
    The raw points come from a box base or from the best candidate of each frame, so
    they do jump. This module gives every track a constant-velocity Kalman filter
    with random acceleration (continuous white noise), followed by a Rauch-Tung-
    Striebel smoother, so the result is continuous in position, velocity and
    acceleration.

    * :func:`smooth_track` - gated RTS smoothing of one track (pure NumPy);
    * :data:`BALL_MODEL` / :data:`PLAYER_MODEL` - the acceleration noise of a kick and of a run.

    A measurement whose Mahalanobis distance from the prediction is above ``gate``
    is rejected; after ``max_miss`` rejections in a row the filter restarts on the
    next measurement (the track was really somewhere else). A frame farther than
    ``max_gap`` frames from any accepted measurement stays empty: nothing is invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Version of the flight fit (anchors, kick time, weighting). Written to every ball_flight.json and
# part of the inputs of the one-click 3D stage: change it with the fit, so that an older fit is
# recomputed (stage) or reported (kick-figures) instead of being reused. A file without the tag is
# older than v2 (the v1 fit could send a one-camera penalty to the side: 28 px RMS on a real clip).
FLIGHT_MODEL = "flight-v2"


@dataclass(frozen=True)
class TrackModel:
    accel_sd: float  # m/s^2, random acceleration (the process noise)
    meas_sd: float  # m, noise of one measurement
    gate: float  # squared Mahalanobis distance above which a measurement is rejected
    max_miss: int  # rejections in a row before the filter restarts on a measurement
    max_gap: int  # frames without an accepted measurement before a frame stays empty
    max_speed_mps: float = 0.0  # physical ceiling of the smoothed speed; 0 = no ceiling


# A kick accelerates a ball by hundreds of m/s^2 for a few frames; rolling and bouncing are smaller.
# The ceiling is the speed of the fastest professional shots (about 40 m/s).
BALL_MODEL = TrackModel(
    accel_sd=40.0, meas_sd=0.3, gate=9.21, max_miss=8, max_gap=6, max_speed_mps=40.0
)
# A person's acceleration stays around 2 m/s^2 in play (peaks of a sprint start are about 8 to 10).
# The box base is noisy, about 0.6 m; with these values the largest acceleration of the
# clip was 9.5 m/s^2 and the median deviation from the raw box base 0.24 m (grid test on the
# Paqueta run: accel 5.0 gave 28 m/s^2 spikes; 1.0 gave an over-flat track).
PLAYER_MODEL = TrackModel(accel_sd=2.0, meas_sd=0.6, gate=13.8, max_miss=10, max_gap=15)


def _transition(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """State [x, y, vx, vy] transition and the process noise of a random acceleration."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    qa = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * q
    Q = np.zeros((4, 4))
    for axis in range(2):
        idx = [axis, axis + 2]
        Q[np.ix_(idx, idx)] = qa
    return F, Q


def smooth_track(
    measurements: dict[int, tuple[float, float]],
    n_frames: int,
    fps: float,
    model: TrackModel,
) -> dict[int, tuple[float, float]]:
    """The physically consistent position of a track at every frame of ``range(n_frames)``.

    ``measurements`` maps frame -> ``(x, y)`` in metres. Frames within ``model.max_gap``
    of an accepted measurement get a smoothed position; the others are left out.
    """
    if not measurements or n_frames <= 0:
        return {}
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    R = np.eye(2) * model.meas_sd**2
    q = model.accel_sd**2

    n = n_frames
    times = np.arange(n) / fps
    x_pred = np.zeros((n, 4))
    P_pred = np.zeros((n, 4, 4))
    x_filt = np.zeros((n, 4))
    P_filt = np.zeros((n, 4, 4))
    F_of = np.zeros((n, 4, 4))
    has_state = np.zeros(n, dtype=bool)
    accepted = np.zeros(n, dtype=bool)

    first = min(measurements)
    x = np.array([*measurements[first], 0.0, 0.0])
    P = np.diag([model.meas_sd**2, model.meas_sd**2, 10.0**2, 10.0**2])
    misses = 0
    for k in range(first, n):
        if k == first:
            x_pred[k], P_pred[k] = x, P
        else:
            F, Q = _transition(times[k] - times[k - 1], q)
            F_of[k] = F
            x_pred[k] = F @ x
            P_pred[k] = F @ P @ F.T + Q
        if k in measurements:
            z = np.asarray(measurements[k], dtype=np.float64)
            innov = z - H @ x_pred[k]
            S = H @ P_pred[k] @ H.T + R
            d2 = float(innov @ np.linalg.solve(S, innov))
            if d2 <= model.gate or k == first:
                K = P_pred[k] @ H.T @ np.linalg.inv(S)
                x = x_pred[k] + K @ innov
                P = (np.eye(4) - K @ H) @ P_pred[k]
                accepted[k] = True
                misses = 0
            else:
                misses += 1
                x, P = x_pred[k], P_pred[k]
                if misses > model.max_miss:  # the track was elsewhere: restart on this point
                    x = np.array([*z, 0.0, 0.0])
                    P = np.diag([model.meas_sd**2, model.meas_sd**2, 10.0**2, 10.0**2])
                    accepted[k] = True
                    misses = 0
        else:
            x, P = x_pred[k], P_pred[k]
        x_filt[k], P_filt[k] = x, P
        has_state[k] = True

    # Rauch-Tung-Striebel backward pass.
    x_s = x_filt.copy()
    for k in range(n - 2, first - 1, -1):
        Fn = F_of[k + 1]
        try:
            C = P_filt[k] @ Fn.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        x_s[k] = x_filt[k] + C @ (x_s[k + 1] - x_pred[k + 1])

    # A frame is kept only near an accepted measurement.
    acc_frames = np.flatnonzero(accepted)
    kept = [
        k
        for k in range(first, n)
        if has_state[k] and np.min(np.abs(acc_frames - k)) <= model.max_gap
    ]
    pos = np.array([x_s[k, :2] for k in kept]).reshape(-1, 2)
    if model.max_speed_mps > 0 and len(kept) > 1:  # the ceiling: no step faster than it allows
        for i in range(1, len(kept)):
            delta = pos[i] - pos[i - 1]
            limit = model.max_speed_mps * (kept[i] - kept[i - 1]) / fps
            norm = float(np.hypot(*delta))
            if norm > limit > 0:
                pos[i] = pos[i - 1] + delta * (limit / norm)
    return {k: (float(x), float(y)) for k, (x, y) in zip(kept, pos, strict=True)}


# --------------------------------------------------------------------------- #
# Flight of a hand-corrected ball: drag Cd(Re), Magnus lift, spin inferred from the pixels
# --------------------------------------------------------------------------- #
# Sources: Kiratidis & Leinweber, arXiv:1710.02784 (2018), Eqs. 5, 6, 8 and Table 1 (Brazuca);
# Asai & Seo, SpringerPlus 2 (2013) (critical Reynolds, Cd about 0.47 below the crisis and
# 0.12-0.15 above it); Aguiar, "Aerodinamica da bola de futebol" (UFRJ), Magnus and the
# "gol que Pele nao fez" case used as the validation of the flight.

GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.2  # kg/m^3
AIR_VISCOSITY = 1.8e-5  # Pa s (dynamic)
BALL_DIAMETER = 0.22  # m
BALL_MASS = 0.43  # kg
BALL_RADIUS = BALL_DIAMETER / 2.0  # m
BALL_AREA = np.pi * BALL_RADIUS**2  # m^2, cross section
PIXEL_SIGMA = 2.0  # px, the noise of a hand-placed point
ANCHOR_SIGMA = {"rest": 0.3, "barrier_xy": 0.1, "barrier_z": 0.1, "keeper": 0.2}  # m
SPIN_DRAG_B = 0.514  # rad^-1, Cd += b*Sp (Kiratidis Eq. 6; the Teamgeist value)
SPIN_MAX_RPS = 12.0  # bounds the spin search, about 6-8 rps is a strong kick
# Brazuca drag fit of Kiratidis & Leinweber Table 1 (v in m/s). Fitted up to v_max = 35 m/s.
BRAZUCA_CD = {
    "a": 0.4740,
    "v_c": 12.92,
    "v_s": 1.000,
    "b_min": 0.1657,
    "b_max": 0.2112,
    "v_min": 14.61,
    "v_max": 35.00,
}


def reynolds(speed: float) -> float:
    """Re = v D / nu, with nu = mu / rho."""
    return float(speed) * BALL_DIAMETER * AIR_DENSITY / AIR_VISCOSITY


def spin_parameter(speed: float, omega_norm: float) -> float:
    """Sp = r |omega| / v (Aguiar; Kiratidis Eq. 4)."""
    return BALL_RADIUS * float(omega_norm) / max(float(speed), 1e-6)


def drag_coefficient(speed: float, sp: float = 0.0) -> float:
    """Cd of the Brazuca at speed ``speed`` (m/s) and spin parameter ``sp``.

    Kiratidis Eq. 5 (a sigmoid blend of the subcritical and supercritical values) plus
    the spin term of Eq. 6. Above ``v_max`` the fit is held at its value at ``v_max``.
    """
    c = BRAZUCA_CD
    v = min(float(speed), c["v_max"])
    e1 = np.clip((v - c["v_c"]) / c["v_s"], -60.0, 60.0)
    e2 = np.clip(-(v - c["v_min"]) / c["v_s"], -60.0, 60.0)
    cd = (c["a"] - c["b_min"]) / (1.0 + np.exp(e1)) + c["b_min"]
    cd += (
        (v - c["v_min"])
        / (1.0 + np.exp(e2))
        * (c["b_max"] - c["b_min"])
        / (c["v_max"] - c["v_min"])
    )
    return float(cd + SPIN_DRAG_B * sp)


def lift_coefficient(sp: float) -> float:
    """C_L = 1.15 Sp^0.83 (Kiratidis Eq. 8, without the drag factor of Eq. 9)."""
    return 1.15 * max(float(sp), 0.0) ** 0.83


def simulate_flight(
    theta: np.ndarray,
    times: np.ndarray,
    rest_height: float = BALL_RADIUS,
) -> np.ndarray:
    """Ball centre (x, y, z) at each time for ``theta = [x0, y0, vx, vy, vz, tk, wx, wy, wz]``.

    Before the kick (``t < tk``) the ball rests at ``(x0, y0, rest_height)``. From ``tk`` it
    flies under gravity, quadratic drag ``-(rho A / 2m) Cd |v| v`` and Magnus lift
    ``(rho A / 2m) C_L v^2 (w_hat x v_hat)``, with the spin ``omega = (wx, wy, wz)`` in rad/s.
    """
    from scipy.integrate import solve_ivp

    x0, y0, vx, vy, vz, tk, wx, wy, wz = (float(v) for v in theta[:9])
    omega = np.array([wx, wy, wz])
    omega_norm = float(np.linalg.norm(omega))
    times = np.asarray(times, dtype=np.float64)
    out = np.column_stack([np.full(times.shape, x0), np.full(times.shape, y0),
                           np.full(times.shape, rest_height)])  # fmt: skip
    flying = times >= tk
    if not flying.any():
        return out
    k = 0.5 * AIR_DENSITY * BALL_AREA / BALL_MASS

    def derivative(_t, s):
        v = s[3:]
        speed = float(np.linalg.norm(v))
        if speed < 1e-9:
            return np.concatenate([v, [0.0, 0.0, -GRAVITY]])
        sp = spin_parameter(speed, omega_norm)
        cd = drag_coefficient(speed, sp)
        cl = lift_coefficient(sp)
        spin_dir = np.cross(omega / max(omega_norm, 1e-12), v / speed) if omega_norm > 0 else 0.0
        a = -k * cd * speed * v + k * cl * speed * speed * spin_dir
        return np.concatenate([v, a + np.array([0.0, 0.0, -GRAVITY])])

    s0 = np.array([x0, y0, rest_height, vx, vy, vz])
    t_eval = times[flying] - tk
    if t_eval.max() <= 0.0:
        out[flying] = s0[:3]
        return out
    sol = solve_ivp(derivative, (0.0, float(t_eval.max())), s0, t_eval=t_eval,
                    rtol=1e-6, atol=1e-8)  # fmt: skip
    if not sol.success or sol.y.shape[1] != t_eval.size:
        out[flying] = np.nan
        return out
    out[flying] = sol.y[:3].T
    return out


def seed_flight(
    frames: np.ndarray,
    xyz: np.ndarray,
    fps: float,
    *,
    start_frame: int | None = None,
    launch_height_m: float = 0.3,
) -> np.ndarray:
    """A starting guess of the flight ``theta`` from a rough track (metres, one row per frame).

    The rest point is the track at ``start_frame`` (default: the first frame). The launch is the
    first frame above ``launch_height_m``; when the track never leaves the ground (a single camera
    lifted to the plane), the launch is the start and the velocity is the horizontal motion.
    """
    frames = np.asarray(frames, dtype=float)
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    start = int(frames[0]) if start_frame is None else int(start_frame)
    idx0 = int(np.argmin(np.abs(frames - start)))
    rest = xyz[idx0]
    lifted = np.flatnonzero((frames >= start) & (xyz[:, 2] > launch_height_m))
    k = int(lifted[0]) if lifted.size else idx0
    v = np.zeros(3)
    after = np.flatnonzero(frames >= frames[k])
    if after.size >= 4:
        j = after[3]
        dt = (frames[j] - frames[k]) / fps
        if dt > 0:
            v = (xyz[j] - xyz[k]) / dt
    if not lifted.size:  # no lift in the track: keep only the horizontal motion
        v[2] = 0.0
    tk = max(frames[k] - 1, start) / fps
    return np.array([rest[0], rest[1], v[0], v[1], v[2], tk, 0.0, 0.0, 0.0])


def fit_flight_views(
    views: list[tuple[dict[int, tuple[float, float]], dict[int, np.ndarray]]],
    fps: float,
    n_frames: int,
    *,
    anchors: dict,
    rest_height: float = BALL_RADIUS,
    spin_starts: int = 3,
    theta0: np.ndarray | None = None,
    kick_window: tuple[float, float] | None = None,
) -> dict:
    """Fit one flight to the pixels of one or more calibrated cameras (see :func:`fit_flight`).

    ``kick_window`` (seconds, optional) bounds the kick time: the flight cannot start before the
    ball was seen to move, so a fit on the frames after the contact keeps its start there.

    ``views`` is a list of ``(pixels, cameras)``, one pair per camera: ``pixels`` maps frame ->
    ``(u, v)`` and ``cameras`` maps frame -> that camera's 3x4 projection on the same frame clock.
    Each camera adds its own reprojection residuals, so a second camera fixes the depth of the
    ball. ``theta0`` (optional) is one extra start for the search: the parameters of a rough
    track, for example the triangulated one.
    """
    from scipy.optimize import least_squares

    obs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for pixels, cameras in views:
        frames = np.array(sorted(f for f in pixels if f in cameras), dtype=int)
        uv = np.array([pixels[f] for f in frames], dtype=np.float64).reshape(-1, 2)
        P = np.array([cameras[f] for f in frames], dtype=np.float64).reshape(-1, 3, 4)
        obs.append((frames, uv, P))
    n_obs = sum(2 * len(frames) for frames, _, _ in obs)
    times = np.arange(n_frames) / fps
    rest_xy = anchors.get("rest")
    barrier = anchors.get("barrier")
    keeper = anchors.get("keeper")
    point_list = anchors.get("points", [])
    point_frames = [int(p["frame"]) for p in point_list]
    point_xyz = [np.asarray(p["xyz"], dtype=float) for p in point_list]
    point_sigma = [float(p["sigma"]) for p in point_list]
    tail_size = (
        (2 if rest_xy is not None else 0)
        + (2 if barrier is not None else 0)
        + (3 if keeper is not None else 0)
        + 3 * len(point_frames)
    )

    def residuals(theta: np.ndarray) -> np.ndarray:
        pos = simulate_flight(theta, times, rest_height)
        if not np.isfinite(pos).all():
            return np.full(n_obs + tail_size, 1e3)
        parts = []
        for frames, uv, P in obs:
            X = np.column_stack([pos[frames], np.ones(len(frames))])
            h = np.einsum("nij,nj->ni", P, X)
            ok = h[:, 2] > 1e-6
            proj = np.where(ok[:, None], h[:, :2] / np.where(ok, h[:, 2], 1.0)[:, None], 1e4)
            parts.append(((proj - uv) / PIXEL_SIGMA).ravel())
        if rest_xy is not None:
            parts.append((theta[:2] - np.asarray(rest_xy, dtype=float)) / ANCHOR_SIGMA["rest"])
        if barrier is not None:
            b = int(barrier["frame"])
            d = float(np.hypot(*(pos[b, :2] - np.asarray(barrier["xy"], dtype=float))))
            parts.append(np.array([(d - float(barrier["offset"])) / ANCHOR_SIGMA["barrier_xy"]]))
            parts.append(np.array([(pos[b, 2] - float(barrier["z"])) / ANCHOR_SIGMA["barrier_z"]]))
        if keeper is not None:
            k = int(keeper["frame"])
            parts.append((pos[k] - np.asarray(keeper["xyz"], dtype=float)) / ANCHOR_SIGMA["keeper"])
        for f_p, xyz_p, s_p in zip(point_frames, point_xyz, point_sigma, strict=True):
            parts.append((pos[f_p] - xyz_p) / s_p)
        return np.concatenate(parts)

    start_xy = rest_xy
    if start_xy is None and point_frames:  # the start of the guess only; residuals unchanged
        start_xy = point_xyz[int(np.argmin(point_frames))][:2]
    x0_guess = np.asarray(start_xy, dtype=float) if start_xy is not None else np.zeros(2)
    target: dict | None = keeper
    if target is None and point_frames and max(point_frames) > 0:
        last_i = int(np.argmax(point_frames))
        target = {"frame": point_frames[last_i], "xyz": point_xyz[last_i]}
    spin_guesses = [
        np.zeros(3),
        np.array([0.0, 0.0, 2 * np.pi * 6.0]),
        np.array([0.0, 0.0, -2 * np.pi * 6.0]),
    ][:spin_starts]
    omega_max = 2 * np.pi * SPIN_MAX_RPS
    lo = [-80, -60, -80, -80, -80, 0.0, -omega_max, -omega_max, -omega_max]
    hi = [80, 60, 80, 80, 80, (n_frames - 1) / fps, omega_max, omega_max, omega_max]
    if kick_window is not None:
        lo[5] = max(lo[5], float(kick_window[0]))
        hi[5] = min(hi[5], float(kick_window[1]))
    starts: list[np.ndarray] = []
    if theta0 is not None:
        starts.append(np.asarray(theta0, dtype=float))
    tk_guesses = [0.0, 0.5 / fps, 1.0 / fps]
    if kick_window is not None:
        tk_guesses.insert(0, 0.5 * (float(kick_window[0]) + float(kick_window[1])))
    for tk0 in tk_guesses:
        v = np.zeros(3)
        if target is not None:
            kf = int(target["frame"])
            T = max(kf / fps - tk0, 1e-3)
            kx, ky, kz = (float(c) for c in np.asarray(target["xyz"], dtype=float))
            v = np.array(
                [
                    (kx - x0_guess[0]) / T,
                    (ky - x0_guess[1]) / T,
                    (kz - rest_height + 0.5 * GRAVITY * T * T) / T,
                ]
            )
        for w0 in spin_guesses:
            starts.append(np.concatenate([x0_guess, v, [tk0], w0]))
    best = None
    for guess in starts:
        sol = least_squares(
            residuals,
            np.clip(guess, lo, hi),
            bounds=(lo, hi),
            method="trf",
            max_nfev=400,
        )
        if best is None or sol.cost < best.cost:
            best = sol

    if best is None:  # the loops above always run, this only narrows the type
        raise RuntimeError("no flight was fitted")
    theta = best.x
    res = residuals(theta)
    per_frame_acc: dict[int, list[float]] = {}
    px_all: list[np.ndarray] = []
    offset = 0
    for frames, _uv, _P in obs:
        n = 2 * len(frames)
        px = res[offset : offset + n].reshape(-1, 2) * PIXEL_SIGMA
        offset += n
        for f, e in zip(frames, px, strict=True):
            per_frame_acc.setdefault(int(f), []).append(float(np.hypot(*e)))
            px_all.append(e)
    px_arr = np.array(px_all) if px_all else np.zeros((0, 2))
    per_frame_px = {f: float(np.mean(errs)) for f, errs in sorted(per_frame_acc.items())}
    pos = simulate_flight(theta, times, rest_height)
    speed = np.zeros(n_frames)
    if n_frames > 1:
        speed[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    omega = theta[6:9]
    omega_norm = float(np.linalg.norm(omega))
    tail = res[n_obs:]
    anchor_res: dict[str, float | None] = {
        "rest_m": None,
        "barrier_distance_m": None,
        "barrier_height_m": None,
        "keeper_m": None,
    }
    i = 0
    if rest_xy is not None:
        anchor_res["rest_m"] = round(float(np.hypot(*tail[i : i + 2]) * ANCHOR_SIGMA["rest"]), 3)
        i += 2
    if barrier is not None:
        anchor_res["barrier_distance_m"] = round(float(tail[i] * ANCHOR_SIGMA["barrier_xy"]), 3)
        anchor_res["barrier_height_m"] = round(float(tail[i + 1] * ANCHOR_SIGMA["barrier_z"]), 3)
        i += 2
    if keeper is not None:
        anchor_res["keeper_m"] = round(
            float(np.linalg.norm(tail[i : i + 3]) * ANCHOR_SIGMA["keeper"]), 3
        )
        i += 3
    for j, s_p in enumerate(point_sigma):
        anchor_res[f"p{j}_m"] = round(float(np.linalg.norm(tail[i : i + 3]) * s_p), 3)
        i += 3
    return {
        "flight_model": FLIGHT_MODEL,
        "theta": [round(float(v), 4) for v in theta],
        "kick_time_s": round(float(theta[5]), 4),
        "spin_rps": round(omega_norm / (2 * np.pi), 2),
        "spin_axis": [
            round(float(v), 4) for v in (omega / omega_norm if omega_norm > 0 else omega)
        ],
        "per_frame_px": per_frame_px,
        "rms_px": round(float(np.sqrt(np.mean(px_arr**2))), 3) if px_arr.size else None,
        "max_px": round(float(np.max(np.hypot(px_arr[:, 0], px_arr[:, 1]))), 3)
        if px_arr.size
        else None,
        "anchor_residuals": anchor_res,
        "positions": {int(f): (float(x), float(y), float(z)) for f, (x, y, z) in enumerate(pos)},
        "speed_mps": [round(float(s), 2) for s in speed],
        "peak_speed_mps": round(float(speed.max()), 2),
        "above_fit_range": bool(speed.max() > BRAZUCA_CD["v_max"]),
        "success": bool(best.success),
        "views": len(obs),
    }


def fit_flight(
    pixels: dict[int, tuple[float, float]],
    cameras: dict[int, np.ndarray],
    fps: float,
    n_frames: int,
    *,
    anchors: dict,
    rest_height: float = BALL_RADIUS,
    spin_starts: int = 3,
    theta0: np.ndarray | None = None,
    kick_window: tuple[float, float] | None = None,
) -> dict:
    """Fit one flight (velocity, spin and kick time) to a hand-corrected ball.

    ``pixels`` maps frame -> ``(u, v)`` of the ball centre; ``cameras`` maps frame -> the 3x4
    projection of that frame (pitch metres -> pixels). ``anchors`` may hold ``"rest"`` (x, y at
    the kick, the kicker's foot), ``"barrier"`` (``{"frame", "xy", "z", "offset"}``: the ball is
    ``z`` high and ``offset`` horizontally from ``xy``), ``"keeper"`` (``{"frame", "xyz"}``) and
    ``"points"`` (``{"frame", "xyz", "sigma"}`` each, at least three).

    Residuals: reprojection in px (``PIXEL_SIGMA``) and the anchors in metres
    (``ANCHOR_SIGMA``). Several kick times and spin starts are tried; the best is kept.
    One camera only: :func:`fit_flight_views` takes several. ``theta0`` is an optional extra
    start (see :func:`seed_flight`).
    """
    return fit_flight_views(
        [(pixels, cameras)],
        fps,
        n_frames,
        anchors=anchors,
        rest_height=rest_height,
        spin_starts=spin_starts,
        theta0=theta0,
        kick_window=kick_window,
    )
