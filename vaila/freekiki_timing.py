"""
Project: vailá
Script: freekiki_timing.py - camera clocks: frames to seconds, and one interval on several cameras

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 November 2026
Update Date: 11 November 2026
Version: 0.4.16

Description:
    A broadcast shows the same penalty from several cameras, and the frame counts differ because the
    cameras do not capture at the same rate: 11 frames of the live camera (29.97 fps), 19 of a replay
    captured at 50 fps and 42 of a slow-motion camera captured at 119.88 fps are the same ~0.36 s. A
    count is seconds only at the rate the frames were captured at, not at the rate the file plays.

    This module turns a count into an interval with its tolerance (``CameraSpan``), compares the
    intervals of several cameras (``reconcile``) and estimates the effective capture rate of a replay
    from the events both cameras show (``fit_clock``: t_main = a * frame + b).

    The capture rate of a replay is not in the file (the file says 29.97 fps whatever was captured), so
    each span carries where its rate came from: ``measured`` (the file of a live camera, or fitted
    from events), ``declared`` (told by whoever knows the production) or ``estimated``.

Usage:
    from vaila.freekiki_timing import CameraSpan, reconcile, fit_clock

License: AGPLv3
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SOURCES = ("measured", "declared", "estimated")


@dataclass(frozen=True)
class CameraSpan:
    """Frames counted between two events on one camera.

    ``file_fps`` is the rate the file plays at, ``capture_fps`` the rate the frames were captured at
    (the same for a live camera, higher for a slow-motion replay). ``frames_sigma`` is the counting
    tolerance in frames (one frame by default: each end of the interval is placed to half a frame).
    """

    name: str
    file_fps: float
    capture_fps: float
    frames: float
    source: str = "declared"
    frames_sigma: float = 1.0

    def _rate(self) -> float:
        if not (self.capture_fps > 0 and math.isfinite(self.capture_fps)):
            raise ValueError(
                f"{self.name}: the capture rate must be positive, got {self.capture_fps}"
            )
        return float(self.capture_fps)

    def seconds(self) -> float:
        """The real time of the counted frames."""
        return self.frames / self._rate()

    def tolerance(self) -> float:
        """The counting tolerance in seconds."""
        return self.frames_sigma / self._rate()

    def shown_seconds(self) -> float:
        """How long the file takes to play these frames (not the real time)."""
        return self.frames / self.file_fps

    def slowdown(self) -> float:
        """How many times slower than real time the file plays this camera."""
        return self._rate() / self.file_fps

    def row(self) -> dict:
        """The span as a table row: real time, tolerance and interval, shown time and slow-down."""
        if not self.frames > 0:
            raise ValueError(f"{self.name}: frames must be positive, got {self.frames}")
        dt, tol = self.seconds(), self.tolerance()
        return {
            "name": self.name,
            "source": self.source,
            "file_fps": round(self.file_fps, 3),
            "capture_fps": round(self.capture_fps, 3),
            "slowdown": round(self.slowdown(), 3),
            "frames": self.frames,
            "shown_s": round(self.shown_seconds(), 4),
            "dt_s": round(dt, 4),
            "tol_s": round(tol, 4),
            "lo_s": round(dt - tol, 4),
            "hi_s": round(dt + tol, 4),
        }


def slowdown(capture_hz: float, file_fps: float) -> float:
    """The slow-motion factor of a replay: the capture rate over the rate the file plays at."""
    return float(capture_hz) / float(file_fps)


def reconcile(spans: list[CameraSpan]) -> dict:
    """The intervals of several cameras side by side, and whether they agree within their tolerances.

    Returns ``cameras`` (one row each: real time, tolerance, interval, shown time, slow-down),
    ``pairs`` (difference, allowed difference, ``consistent`` and the ``gap_ms`` by which a pair
    misses), ``common_interval_s`` (the interval all cameras share, None when they do not), the
    largest ``max_gap_ms``, the ``span_s`` of the central values and the inverse-variance
    ``weighted_mean_s`` (each tolerance read as a uniform half-width, so the finest clock weighs most).
    """
    if len(spans) < 2:
        raise ValueError("reconciling needs at least two cameras")
    rows = [sp.row() for sp in spans]
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            diff = abs(a["dt_s"] - b["dt_s"])
            allowed = a["tol_s"] + b["tol_s"]
            pairs.append(
                {
                    "a": a["name"],
                    "b": b["name"],
                    "diff_ms": round(diff * 1000.0, 1),
                    "allowed_ms": round(allowed * 1000.0, 1),
                    "consistent": bool(diff <= allowed + 1e-9),
                    "gap_ms": round(max(0.0, diff - allowed) * 1000.0, 1),
                }
            )
    lo = max(r["lo_s"] for r in rows)
    hi = min(r["hi_s"] for r in rows)
    weights = np.array([3.0 / r["tol_s"] ** 2 for r in rows])
    values = np.array([r["dt_s"] for r in rows])
    return {
        "cameras": rows,
        "pairs": pairs,
        "common_interval_s": [lo, hi] if lo <= hi else None,
        "max_gap_ms": max(p["gap_ms"] for p in pairs),
        "span_s": round(float(values.max() - values.min()), 4),
        "weighted_mean_s": round(float((weights * values).sum() / weights.sum()), 4),
        "weighted_sigma_s": round(float(1.0 / math.sqrt(weights.sum())), 4),
    }


def fit_clock(ref_times_s, other_frames) -> dict:
    """The effective capture rate of a camera from events seen on both: ``t_ref = a * frame + b``.

    ``ref_times_s`` are the event times on the reference camera's clock, ``other_frames`` the frame of
    the same events on the other camera. ``rate_hz`` is ``1 / a``: the captured frames per real
    second, taken as the rate if every captured frame appears once in the file. ``residual_ms`` shows
    an event placed a frame off instead of letting it bend the rate silently.
    """
    t = np.asarray(ref_times_s, dtype=float)
    k = np.asarray(other_frames, dtype=float)
    if t.size != k.size:
        raise ValueError(
            "every event needs a time on the reference and a frame on the other camera"
        )
    if t.size < 2:
        raise ValueError("a clock needs at least two events")
    a, b = np.polyfit(k, t, 1)
    resid = (t - (a * k + b)) * 1000.0
    return {
        "a_s_per_frame": float(a),
        "rate_hz": float(1.0 / a),
        "offset_s": float(b),
        "residual_ms": [round(float(r), 2) for r in resid],
        "rms_ms": round(float(np.sqrt(np.mean(resid**2))), 2),
        "n": int(t.size),
    }
