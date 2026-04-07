# SPDX-License-Identifier: MIT
# From FIFA-Skeletal-Tracking-Starter-Kit-2026 (lib/postprocess.py); see NOTICE.md in this package.
import numpy as np
import pandas as pd


def interpolate_with_gap(data: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """
    Linearly interpolates missing data (NaNs), but ONLY if the gap is small.

    Args:
        data: (N, D) numpy array (time, dimensions)
        max_gap: Maximum number of consecutive NaNs to fill.
    """
    df = pd.DataFrame(data)

    def _gap_aware_fill(series):
        is_nan = series.isna()
        groups = is_nan.ne(is_nan.shift()).cumsum()
        gap_sizes = groups.map(groups.value_counts())
        interp_series = series.interpolate(method="linear", limit_direction="both")
        mask_bad_gaps = is_nan & (gap_sizes > max_gap)
        interp_series[mask_bad_gaps] = np.nan
        return interp_series

    df_clean = df.apply(_gap_aware_fill, axis=0)
    return df_clean.values


def smoothen_traj(traj: np.ndarray, window_size: int = 11, sigma: float = 2.0) -> np.ndarray:
    """
    Smoothens a trajectory using a Gaussian-weighted moving window.

    Args:
        traj: (N, D) numpy array (e.g., a single joint's xyz or a mid-hip path)
        window_size: Size of the window (should be odd).
        sigma: Spread of the gaussian curve.
    """
    traj_filled = interpolate_with_gap(traj, max_gap=3)
    df = pd.DataFrame(traj_filled)
    smoothed = df.rolling(window=window_size, center=True, win_type="gaussian", min_periods=1).mean(
        std=sigma
    )
    return smoothed.values


def smoothen(skels_3d: np.ndarray, window_size: int = 11, sigma: float = 2.0) -> np.ndarray:
    """Smoothens the entire skeleton structure."""
    mid_hip = skels_3d[..., [7, 8], :].mean(axis=-2, keepdims=False)
    mid_hip_filtered = smoothen_traj(mid_hip, window_size=window_size, sigma=sigma)
    return skels_3d - mid_hip[..., None, :] + mid_hip_filtered[..., None, :]
