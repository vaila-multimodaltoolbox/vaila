"""Shared numerical core for interp_smooth_split (GUI and CLI).

Update Date: 24 August 2026
Version: 0.3.112

Pure processing helpers used by both the Tkinter dialog and the headless CLI
so GUI and CLI stay in parity. Visualization stays in ``interp_smooth_split``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

try:
    from .filter_utils import butter_filter
except ImportError:
    from filter_utils import butter_filter  # ty: ignore[unresolved-import]

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
except ImportError:  # pragma: no cover - optional at import time
    _lowess = None


# ---------------------------------------------------------------------------
# Smoothing primitives
# ---------------------------------------------------------------------------


def savgol_smooth(data: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky-Golay filter to 1-D or column-wise 2-D data."""
    data = np.asarray(data, dtype=float)
    window_length = int(window_length)
    polyorder = int(polyorder)
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 1 + (polyorder % 2 == 0)
    if data.ndim == 1:
        n = len(data)
        wl = min(window_length, n if n % 2 == 1 else n - 1)
        if wl <= polyorder:
            return data.copy()
        return savgol_filter(data, wl, polyorder, axis=0)
    out = np.empty_like(data, dtype=float)
    for j in range(data.shape[1]):
        out[:, j] = savgol_smooth(data[:, j], window_length, polyorder)
    return out


def lowess_smooth(data: np.ndarray, frac: float = 0.3, it: int = 3) -> np.ndarray:
    """Apply LOWESS smoothing (1-D or column-wise 2-D)."""
    if _lowess is None:
        raise ImportError("statsmodels is required for LOWESS smoothing")
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        x = np.arange(len(data), dtype=float)
        return np.asarray(
            _lowess(endog=data, exog=x, frac=frac, it=it, return_sorted=False, is_sorted=True),
            dtype=float,
        )
    out = np.empty_like(data, dtype=float)
    x = np.arange(data.shape[0], dtype=float)
    for j in range(data.shape[1]):
        out[:, j] = _lowess(
            endog=data[:, j], exog=x, frac=frac, it=it, return_sorted=False, is_sorted=True
        )
    return out


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------


@dataclass
class TimeAxisInfo:
    status: str  # regular | irregular | invalid | none
    time: np.ndarray | None = None
    estimated_rate_hz: float | None = None
    median_dt: float | None = None
    irregular: bool = False
    warnings: list[str] = field(default_factory=list)


def is_time_column_name(name: str) -> bool:
    return str(name).strip().lower() in {"time", "t", "tempo"}


def is_frame_column_name(name: str) -> bool:
    return str(name).strip().lower() in {"frame", "frames", "frame_index", "frameindex"}


def is_index_column(name: str) -> bool:
    """True for Time/tempo or frame index columns (not interpolated as signals)."""
    return is_time_column_name(name) or is_frame_column_name(name)


def rebuild_frame_index(n_samples: int, *, start: int = 0) -> np.ndarray:
    """Return integer frame indices ``start .. start + n_samples - 1``."""
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    return np.arange(start, start + n_samples, dtype=np.int64)


def first_signal_column(df: pd.DataFrame) -> str | None:
    """First numeric column that is not a Time/frame index."""
    numeric = df.select_dtypes(include=[np.number]).columns
    for col in numeric:
        if not is_index_column(str(col)):
            return str(col)
    if len(numeric):
        return str(numeric[0])
    return None


def align_signals_for_comparison(
    original: np.ndarray | pd.Series,
    processed: np.ndarray | pd.Series,
    *,
    original_rate: float | None = None,
    final_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Align two 1-D signals to the same length for verification statistics."""
    orig = np.asarray(original, dtype=float)
    proc = np.asarray(processed, dtype=float)
    if len(orig) == len(proc):
        return orig, proc, "same_length"
    if (
        original_rate is not None
        and final_rate is not None
        and original_rate > 0
        and final_rate > 0
    ):
        t_proc = np.arange(len(proc), dtype=float) / float(final_rate)
        t_orig = np.arange(len(orig), dtype=float) / float(original_rate)
        f = interp1d(
            t_orig,
            orig,
            kind="linear",
            bounds_error=False,
            fill_value=(float(orig[0]), float(orig[-1])),
            assume_sorted=True,
        )
        orig_aligned = np.asarray(f(t_proc), dtype=float)
        note = (
            f"resample_aligned (original {len(orig)} @ {original_rate:g} Hz → "
            f"processed {len(proc)} @ {final_rate:g} Hz)"
        )
        return orig_aligned, proc, note
    if len(orig) > len(proc) and len(proc) >= 2:
        x_orig = np.linspace(0.0, 1.0, len(orig))
        x_proc = np.linspace(0.0, 1.0, len(proc))
        f = interp1d(
            x_orig,
            orig,
            kind="linear",
            bounds_error=False,
            fill_value=(float(orig[0]), float(orig[-1])),
            assume_sorted=True,
        )
        return (
            np.asarray(f(x_proc), dtype=float),
            proc,
            f"duration_aligned ({len(orig)}→{len(proc)})",
        )
    n = min(len(orig), len(proc))
    return orig[:n], proc[:n], f"truncated ({len(orig)} vs {len(proc)})"


def validate_time_axis(values: np.ndarray | pd.Series | None) -> TimeAxisInfo:
    """Classify a candidate Time column."""
    if values is None:
        return TimeAxisInfo(status="none", warnings=["No Time axis provided."])

    arr = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    if arr.size < 2:
        return TimeAxisInfo(status="invalid", time=arr, warnings=["Time axis too short."])
    if not np.isfinite(arr).all():
        return TimeAxisInfo(
            status="invalid",
            time=arr,
            warnings=["Time axis contains NaN/Inf."],
        )
    diffs = np.diff(arr)
    if np.any(diffs <= 0):
        return TimeAxisInfo(
            status="invalid",
            time=arr,
            warnings=["Time axis must be strictly monotonic increasing (no duplicates)."],
        )
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        return TimeAxisInfo(status="invalid", time=arr, warnings=["Non-positive median dt."])
    rate = 1.0 / median_dt
    rel_spread = float(np.std(diffs) / median_dt) if median_dt > 0 else np.inf
    irregular = rel_spread > 0.05
    status = "irregular" if irregular else "regular"
    warnings: list[str] = []
    if irregular:
        warnings.append(
            f"Irregular Time axis (dt relative spread={rel_spread:.3f}); "
            "using actual timestamps, not a constant-rate assumption."
        )
    return TimeAxisInfo(
        status=status,
        time=arr,
        estimated_rate_hz=rate,
        median_dt=median_dt,
        irregular=irregular,
        warnings=warnings,
    )


def estimate_sampling_rate(
    time_values: np.ndarray | pd.Series | None = None,
    *,
    configured_rate: float | None = None,
    require: bool = False,
) -> tuple[float | None, str, list[str]]:
    """Return (rate_hz, source, warnings).

    Precedence: valid Time median-dt → configured_rate → None.
    Never silently invents a rate.
    """
    warnings: list[str] = []
    info = validate_time_axis(time_values) if time_values is not None else TimeAxisInfo("none")
    if info.status in {"regular", "irregular"} and info.estimated_rate_hz:
        warnings.extend(info.warnings)
        return float(info.estimated_rate_hz), "time_column", warnings
    if configured_rate is not None and float(configured_rate) > 0:
        return float(configured_rate), "configured", warnings
    if require:
        warnings.append("Sampling rate required but neither Time nor configured rate is available.")
    return None, "none", warnings


def rebuild_time_from_rate(
    n_samples: int, sample_rate_hz: float, origin: float = 0.0
) -> np.ndarray:
    """``time = origin + frame / sample_rate``."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    return origin + np.arange(n_samples, dtype=float) / float(sample_rate_hz)


# ---------------------------------------------------------------------------
# Derivatives and residuals
# ---------------------------------------------------------------------------


@dataclass
class DerivativeResult:
    first: np.ndarray
    second: np.ndarray
    units_first: str
    units_second: str
    mode: str  # physical_time | constant_rate | per_sample
    warnings: list[str] = field(default_factory=list)


def compute_derivatives(
    position: np.ndarray,
    *,
    time: np.ndarray | None = None,
    fs: float | None = None,
) -> DerivativeResult:
    """Time-aware first and second derivatives.

    Prefer ``time`` when valid; else constant ``fs``; else per-sample.
    """
    y = np.asarray(position, dtype=float)
    warnings: list[str] = []
    info = validate_time_axis(time) if time is not None else TimeAxisInfo("none")
    if info.status in {"regular", "irregular"} and info.time is not None:
        t = info.time
        first = np.gradient(y, t)
        second = np.gradient(first, t)
        warnings.extend(info.warnings)
        return DerivativeResult(
            first=first,
            second=second,
            units_first="unit/s",
            units_second="unit/s²",
            mode="physical_time",
            warnings=warnings,
        )
    if fs is not None and float(fs) > 0:
        dt = 1.0 / float(fs)
        first = np.gradient(y, dt)
        second = np.gradient(first, dt)
        return DerivativeResult(
            first=first,
            second=second,
            units_first="unit/s",
            units_second="unit/s²",
            mode="constant_rate",
            warnings=warnings,
        )
    warnings.append(
        "No usable Time or sampling rate; derivatives are per sample (not physical velocity)."
    )
    first = np.gradient(y)
    second = np.gradient(first)
    return DerivativeResult(
        first=first,
        second=second,
        units_first="unit/sample",
        units_second="unit/sample²",
        mode="per_sample",
        warnings=warnings,
    )


def compute_residual_metrics(original: np.ndarray, smoothed: np.ndarray) -> dict[str, float]:
    """Finite-sample residual metrics between original and smoothed signals."""
    a = np.asarray(original, dtype=float)
    b = np.asarray(smoothed, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return {"rms": float("nan"), "std": float("nan"), "n": 0.0}
    resid = a[mask] - b[mask]
    return {
        "rms": float(np.sqrt(np.mean(resid**2))),
        "std": float(np.std(resid)),
        "n": float(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Butterworth validation and recommendation
# ---------------------------------------------------------------------------


def validate_butterworth_params(fs: float, cutoff: float, order: int = 4) -> list[str]:
    """Return validation errors (empty list means OK). Never auto-clamps cutoff."""
    errors: list[str] = []
    if fs is None or not np.isfinite(fs) or fs <= 0:
        errors.append("Butterworth fs must be > 0.")
        return errors
    nyq = fs / 2.0
    if cutoff is None or not np.isfinite(cutoff) or cutoff <= 0:
        errors.append("Butterworth cutoff must be > 0.")
    elif cutoff >= nyq:
        errors.append(f"Butterworth cutoff ({cutoff} Hz) must be < Nyquist ({nyq} Hz = fs/2).")
    if order is None or int(order) < 1:
        errors.append("Butterworth order must be >= 1.")
    return errors


def recommend_butterworth(
    data: np.ndarray,
    fs: float,
    *,
    fc_min: float = 1.0,
    fc_max: float | None = None,
    n_fc: int = 29,
    order: int = 4,
) -> dict[str, Any]:
    """Winter-style residual analysis recommendation for Butterworth cutoff.

    Criterion: among candidates with ``0 < fc < fs/2``, pick the cutoff that
    minimizes residual RMS of (signal - filtered). This is a documented heuristic,
    not a claim of objective optimality.
    """
    y = np.asarray(data, dtype=float)
    if not np.isfinite(y).all():
        y = pd.Series(y).interpolate(method="linear", limit_direction="both").to_numpy()
    if fs <= 0:
        raise ValueError("fs must be > 0")
    nyq = fs / 2.0
    if fc_max is None:
        fc_max = max(fc_min, min(15.0, 0.9 * nyq))
    fc_max = min(float(fc_max), 0.95 * nyq)
    if fc_max <= fc_min:
        fc_min = max(0.1, 0.05 * nyq)
        fc_max = 0.9 * nyq
    candidates = np.linspace(fc_min, fc_max, n_fc)
    rms_list: list[float] = []
    for fc in candidates:
        filtered = butter_filter(y, fs=fs, filter_type="low", cutoff=float(fc), order=order)
        rms_list.append(compute_residual_metrics(y, filtered)["rms"])
    rms_arr = np.asarray(rms_list, dtype=float)
    best_i = int(np.nanargmin(rms_arr))
    best_fc = float(candidates[best_i])
    filtered_best = butter_filter(y, fs=fs, filter_type="low", cutoff=best_fc, order=order)
    deriv = compute_derivatives(filtered_best, fs=fs)
    return {
        "selected_method": "butterworth",
        "recommended_cutoff": best_fc,
        "filter_order": order,
        "fs": fs,
        "nyquist": nyq,
        "residual_rms": float(rms_arr[best_i]),
        "candidates_fc": candidates,
        "candidates_rms": rms_arr,
        "derivative_rms": float(np.sqrt(np.nanmean(deriv.first**2))),
        "criterion": (
            "Minimize residual RMS of (signal - Butterworth(fc)) over fc in "
            f"[{fc_min:.3g}, {fc_max:.3g}] Hz (Winter-style heuristic)."
        ),
        "warnings": [
            "Recommended configuration is best candidate under the documented residual-RMS criterion, "
            "not an objectively unique optimum."
        ],
    }


# ---------------------------------------------------------------------------
# Interpolation / smoothing dispatch
# ---------------------------------------------------------------------------


def apply_interpolation_1d(
    values: np.ndarray,
    method: str,
    *,
    max_gap: int = 0,
) -> np.ndarray:
    """Interpolate a 1-D series; ``none``/``skip`` leave gaps (skip = unchanged)."""
    method = (method or "none").lower()
    series = pd.Series(np.asarray(values, dtype=float))
    if method in {"none", "skip"}:
        return series.to_numpy()

    nan_mask = series.isna()
    if not nan_mask.any():
        return series.to_numpy()

    large_gap_mask = np.zeros(len(series), dtype=bool)
    if max_gap and max_gap > 0:
        in_gap = False
        start = 0
        for i, is_nan in enumerate(nan_mask.to_numpy()):
            if is_nan and not in_gap:
                in_gap = True
                start = i
            elif not is_nan and in_gap:
                in_gap = False
                if (i - start) > max_gap:
                    large_gap_mask[start:i] = True
        if in_gap and (len(series) - start) > max_gap:
            large_gap_mask[start:] = True

    if method in {"linear", "cubic", "nearest"}:
        filled = series.interpolate(method=method, limit_direction="both")
    elif method == "kalman":
        # Lightweight fallback: linear fill; full Kalman remains in process_file path.
        filled = series.interpolate(method="linear", limit_direction="both")
    else:
        filled = series.interpolate(method="linear", limit_direction="both")

    out = filled.to_numpy(dtype=float)
    if large_gap_mask.any():
        out[large_gap_mask] = np.nan
    return out


def apply_smoothing_1d(
    values: np.ndarray,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    helpers: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply one smoothing method to a 1-D float array.

    ``helpers`` may supply heavier functions already defined in
    ``interp_smooth_split`` (kalman_smooth, spline_smooth, arima_smooth,
    median_filter_smooth, hampel_filter) so this core stays dependency-light.
    """
    method = (method or "none").lower()
    params = dict(params or {})
    helpers = helpers or {}
    y = np.asarray(values, dtype=float)

    if method in {"none", ""}:
        return y.copy()
    if method == "savgol":
        return savgol_smooth(
            y,
            int(params.get("window_length", 7)),
            int(params.get("polyorder", 3)),
        )
    if method == "lowess":
        return lowess_smooth(y, float(params.get("frac", 0.3)), int(params.get("it", 3)))
    if method == "butterworth":
        fs = float(params["fs"])
        cutoff = float(params["cutoff"])
        order = int(params.get("order", 4))
        errors = validate_butterworth_params(fs, cutoff, order)
        if errors:
            raise ValueError("; ".join(errors))
        return np.asarray(
            butter_filter(y, fs=fs, filter_type="low", cutoff=cutoff, order=order, padding=True),
            dtype=float,
        )
    if method == "kalman":
        fn = helpers.get("kalman_smooth")
        if fn is None:
            raise ValueError("kalman_smooth helper not provided")
        return np.asarray(
            fn(y, int(params.get("n_iter", 5)), int(params.get("mode", 1))), dtype=float
        ).reshape(-1)
    if method == "splines":
        fn = helpers.get("spline_smooth")
        if fn is None:
            raise ValueError("spline_smooth helper not provided")
        return np.asarray(fn(y, s=float(params.get("smoothing_factor", 1.0))), dtype=float)
    if method == "arima":
        fn = helpers.get("arima_smooth")
        if fn is None:
            raise ValueError("arima_smooth helper not provided")
        order = (
            int(params.get("p", 1)),
            int(params.get("d", 0)),
            int(params.get("q", 0)),
        )
        return np.asarray(fn(y, order=order), dtype=float)
    if method == "median":
        fn = helpers.get("median_filter_smooth")
        if fn is None:
            raise ValueError("median_filter_smooth helper not provided")
        return np.asarray(fn(y, kernel_size=int(params.get("kernel_size", 5))), dtype=float)
    if method == "hampel":
        fn = helpers.get("hampel_filter")
        if fn is None:
            raise ValueError("hampel_filter helper not provided")
        return np.asarray(
            fn(
                y,
                window_size=int(params.get("window_size", 7)),
                n_sigmas=float(params.get("n_sigmas", 3)),
            ),
            dtype=float,
        )
    raise ValueError(f"Unknown smooth method: {method}")


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def build_target_time_grid(
    t0: float,
    duration: float,
    final_rate: float,
) -> np.ndarray:
    """Inclusive-ish grid: ``n = round(duration * final_rate) + 1``, clamped to duration."""
    if final_rate <= 0:
        raise ValueError("final_rate must be > 0")
    if duration < 0:
        raise ValueError("duration must be >= 0")
    n = int(round(duration * final_rate)) + 1
    n = max(n, 1)
    grid = t0 + np.arange(n, dtype=float) / float(final_rate)
    # Keep duration: last sample at t0 + duration when possible
    if n > 1:
        grid[-1] = t0 + duration
    return grid


def resample_column(
    time: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
    *,
    kind: str = "linear",
) -> np.ndarray:
    """Interpolate one numeric column onto ``target_time``."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    if mask.sum() < 2:
        out = np.full(len(target_time), np.nan, dtype=float)
        return out
    f = interp1d(
        t[mask],
        y[mask],
        kind=kind,
        bounds_error=False,
        fill_value=(y[mask][0], y[mask][-1]),
        assume_sorted=True,
    )
    return np.asarray(f(target_time), dtype=float)


def apply_antialias_if_needed(
    values: np.ndarray,
    *,
    original_rate: float,
    final_rate: float,
    enabled: bool,
    cutoff: float | None = None,
    order: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Optional explicit anti-alias low-pass before downsampling."""
    warnings: list[str] = []
    y = np.asarray(values, dtype=float)
    if not enabled or final_rate >= original_rate:
        return y, warnings
    nyq_final = final_rate / 2.0
    aa_cutoff = float(cutoff) if cutoff is not None else 0.9 * nyq_final
    errors = validate_butterworth_params(original_rate, aa_cutoff, order)
    if errors:
        warnings.extend(errors)
        return y, warnings
    filtered = butter_filter(
        y, fs=original_rate, filter_type="low", cutoff=aa_cutoff, order=order, padding=True
    )
    warnings.append(
        f"Anti-alias Butterworth applied before downsample "
        f"(cutoff={aa_cutoff:.4g} Hz at fs={original_rate:.4g} Hz)."
    )
    return np.asarray(filtered, dtype=float), warnings


def resample_dataframe(
    df: pd.DataFrame,
    *,
    time_col: str,
    original_rate: float,
    final_rate: float,
    numeric_cols: list[str],
    antialias: bool = True,
    antialias_cutoff: float | None = None,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Resample all numeric signal columns onto one shared target time grid."""
    warnings: list[str] = []
    if final_rate <= 0 or original_rate <= 0:
        raise ValueError("original_rate and final_rate must be > 0")
    if abs(final_rate - original_rate) < 1e-12:
        warnings.append("Final rate equals original rate; resampling skipped.")
        return df.copy(), warnings

    skip_cols = set(exclude_cols or [])
    skip_cols.add(time_col)

    time = np.asarray(df[time_col], dtype=float)
    t0 = float(time[0]) if len(time) else 0.0
    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    target = build_target_time_grid(t0, duration, final_rate)

    out = {time_col: target}
    for col in numeric_cols:
        if col in skip_cols:
            continue
        series = np.asarray(df[col], dtype=float)
        series, aa_warn = apply_antialias_if_needed(
            series,
            original_rate=original_rate,
            final_rate=final_rate,
            enabled=antialias and final_rate < original_rate,
            cutoff=antialias_cutoff,
        )
        warnings.extend(aa_warn)
        out[col] = resample_column(time, series, target)

    # Preserve non-numeric / categorical columns by nearest-index lookup
    for col in df.columns:
        if col in out or col in skip_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            out[col] = resample_column(time, np.asarray(df[col], dtype=float), target)
        else:
            idx = np.clip(
                np.searchsorted(time, target, side="left"),
                0,
                len(df) - 1,
            )
            out[col] = df[col].to_numpy()[idx]

    result = pd.DataFrame(out)
    # Keep original column order where possible
    ordered = [c for c in df.columns if c in result.columns]
    extra = [c for c in result.columns if c not in ordered]
    result = result[ordered + extra]
    return result, warnings


def default_processing_config() -> dict[str, Any]:
    """Documented defaults for CLI/GUI when no TOML is present."""
    return {
        "interp_method": "linear",
        "smooth_method": "none",
        "smooth_params": {},
        "padding": 0.0,
        "max_gap": 0,
        "do_split": False,
        "sample_rate": None,
        "resample": False,
        "original_rate": None,
        "final_rate": None,
        "antialias": True,
        "antialias_cutoff": None,
    }
