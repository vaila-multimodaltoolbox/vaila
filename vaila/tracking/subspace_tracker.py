"""
Bidirectional Linear Subspace Appearance Tracker with Huber ICLK and RTS Smoother.

Combines multi-anchor linear subspace appearance modelling (Eigen-templates via SVD),
an M-estimator Inverse Compositional Lucas-Kanade (ICLK) solver with Huber loss via IRLS,
and Rauch-Tung-Striebel (RTS) backward smoothing to eliminate tracking drift and achieve
zero-phase distortion (Δϕ = 0).

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 10 September 2026
Version: 0.3.131
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .rts_smoother import RTSSmoother


@dataclass
class AnchorBlock:
    """Temporal block of consecutive manual anchor annotations."""

    frame_indices: list[int]  # Sorted frame indices (e.g. 10 frames)
    points: list[tuple[float, float]]  # Manual (x, y) coordinates for each frame


@dataclass
class TrackingResult:
    """Kinematic tracking and smoothing results for a sequence of T frames."""

    trajectory: np.ndarray  # Shape (T, 2) smoothed [x, y] coordinates
    velocities: np.ndarray  # Shape (T, 2) smoothed [vx, vy] in px/s
    accelerations: np.ndarray  # Shape (T, 2) smoothed [ax, ay] in px/s^2
    covariances: np.ndarray  # Shape (T, 6, 6) smoothed state covariances
    confidence_scores: np.ndarray  # Shape (T,) normalized correlation / fit metric


def extract_normalized_patch(
    image: np.ndarray,
    center_xy: tuple[float, float],
    patch_size: tuple[int, int] = (31, 31),
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts sub-pixel bicubic patch and returns (normalized_vector, raw_patch).

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image.
    center_xy : tuple[float, float]
        Sub-pixel center (x, y) coordinate.
    patch_size : tuple[int, int]
        (width, height) of patch.
    eps : float
        Small regularizer for Z-score denominator.

    Returns
    -------
    p_norm : np.ndarray
        1D float64 array of length W*H, zero-mean and unit-variance normalized.
    patch_2d : np.ndarray
        2D float64 array of shape (H, W).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray_f32 = gray.astype(np.float32)

    # Sub-pixel bicubic patch sampling (getRectSubPix requires float32 or uint8)
    pw, ph = int(patch_size[0]), int(patch_size[1])
    cx, cy = float(center_xy[0]), float(center_xy[1])
    patch = cv2.getRectSubPix(gray_f32, (pw, ph), (cx, cy)).astype(np.float64)

    # Per-patch Z-score normalization: p_norm = (p - mu) / (sigma + eps)
    mu = float(np.mean(patch))
    sigma = float(np.std(patch))
    patch_norm = (patch - mu) / (sigma + eps)

    return patch_norm.reshape(-1), patch


class BidirectionalSubspaceTracker:
    """Linear Subspace Appearance Tracker with Huber ICLK and RTS Smoother."""

    def __init__(
        self,
        patch_size: tuple[int, int] = (31, 31),
        n_components: int = 4,
        energy_threshold: float = 0.95,
        huber_delta: float = 1.345,
        max_iterations: int = 15,
        convergence_threshold: float = 1e-2,
        fps: float = 60.0,
        sigma_a: float = 100.0,
        sigma_manual: float = 0.8,
        sigma_track_base: float = 1.5,
    ) -> None:
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        self.n_components = int(n_components)
        self.energy_threshold = float(energy_threshold)
        self.huber_delta = float(huber_delta)
        self.max_iterations = int(max_iterations)
        self.convergence_threshold = float(convergence_threshold)
        self.fps = float(fps)
        self.sigma_a = float(sigma_a)
        self.sigma_manual = float(sigma_manual)
        self.sigma_track_base = float(sigma_track_base)

        # Calibrated model parameters
        self.mean_template: np.ndarray | None = None  # Shape (N,)
        self.eigenbasis: np.ndarray | None = None  # Shape (N, d)
        self.singular_values: np.ndarray | None = None
        self.spatial_gradients: np.ndarray | None = None  # Shape (N, 2)
        self.is_calibrated: bool = False

        # RTS Smoother instance
        self.smoother = RTSSmoother(
            fps=self.fps,
            sigma_a=self.sigma_a,
            sigma_manual=self.sigma_manual,
            sigma_track_base=self.sigma_track_base,
        )

    def calibrate(
        self,
        video_source: Any,
        anchors: list[AnchorBlock],
    ) -> None:
        """Calibrates appearance eigenbasis from 3 temporal anchor blocks (30 frames).

        Parameters
        ----------
        video_source : Any
            List of frames or VideoCapture-like source.
        anchors : list[AnchorBlock]
            List of 3 AnchorBlock objects (Start, Mid, End), each containing 10 frames.
        """
        if not anchors:
            raise ValueError("At least one AnchorBlock is required for calibration.")

        patch_list: list[np.ndarray] = []

        for block in anchors:
            for f_idx, pt in zip(block.frame_indices, block.points, strict=False):
                frame = self._get_frame(video_source, f_idx)
                if frame is None:
                    raise ValueError(f"Could not retrieve frame {f_idx} from video source.")
                p_norm, _ = extract_normalized_patch(frame, pt, self.patch_size)
                patch_list.append(p_norm)

        total_exemplars = len(patch_list)
        if total_exemplars < 3:
            raise ValueError(
                f"Expected at least 3 anchor frames for calibration, got {total_exemplars}"
            )

        # Construct data matrix X in R^(N x M)
        x_mat = np.column_stack(patch_list)  # (N, M)

        # Compute mean template t_bar
        self.mean_template = np.mean(x_mat, axis=1)  # (N,)

        # Zero-mean data matrix: X - t_bar
        x_centered = x_mat - self.mean_template[:, np.newaxis]

        # Thin SVD: X_centered = U * Sigma * V^T
        u_mat, s_vals, _ = np.linalg.svd(x_centered, full_matrices=False)
        self.singular_values = s_vals

        # Determine number of components d based on n_components and energy threshold
        s2 = s_vals**2
        total_energy = float(np.sum(s2))
        cum_energy = np.cumsum(s2) / (total_energy + 1e-12)

        # Find minimum components meeting energy_threshold, bounded by n_components and available rank
        energy_d = int(np.searchsorted(cum_energy, self.energy_threshold)) + 1
        d_chosen = max(1, min(max(self.n_components, energy_d), u_mat.shape[1]))
        self.eigenbasis = u_mat[:, :d_chosen]  # Shape (N, d)

        # Precompute spatial gradients of the mean template t_bar
        # Scale = 1.0 / 8.0 normalizes Sobel 3x3 kernel sum for exact pixel derivatives
        pw, ph = self.patch_size
        t_mean_2d = self.mean_template.reshape((ph, pw))

        grad_x = cv2.Sobel(t_mean_2d, cv2.CV_64F, 1, 0, ksize=3, scale=1.0 / 8.0)
        grad_y = cv2.Sobel(t_mean_2d, cv2.CV_64F, 0, 1, ksize=3, scale=1.0 / 8.0)

        # Spatial Jacobian J in R^(N x 2)
        j_mat = np.column_stack([grad_x.reshape(-1), grad_y.reshape(-1)])
        self.spatial_gradients = j_mat

        self.is_calibrated = True

    def align_patch(
        self,
        image: np.ndarray,
        p_init: tuple[float, float],
    ) -> tuple[tuple[float, float], float, np.ndarray]:
        """Runs robust ICLK optimization with Huber M-estimator via IRLS.

        Parameters
        ----------
        image : np.ndarray
            Current video frame.
        p_init : tuple[float, float]
            Initial position (x, y) estimate.

        Returns
        -------
        p_opt : tuple[float, float]
            Optimized sub-pixel position (x, y).
        confidence : float
            Normalized cross-correlation score in [0.0, 1.0].
        c_coeffs : np.ndarray
            Subspace appearance coefficients c in R^d.
        """
        if (
            not self.is_calibrated
            or self.mean_template is None
            or self.eigenbasis is None
            or self.spatial_gradients is None
        ):
            raise RuntimeError("Tracker must be calibrated before calling align_patch.")

        t_bar = self.mean_template
        u_d = self.eigenbasis  # (N, d)
        j_mat = self.spatial_gradients  # (N, 2)
        d = u_d.shape[1]

        # Linearized design matrix A = [J, U_d] in R^(N x (2 + d))
        a_mat = np.column_stack([j_mat, u_d])

        curr_x, curr_y = float(p_init[0]), float(p_init[1])
        c_coeffs = np.zeros(d, dtype=np.float64)

        img_h, img_w = image.shape[:2]
        pw, ph = self.patch_size
        half_w, half_h = pw / 2.0, ph / 2.0

        for _ in range(self.max_iterations):
            # Clamp to prevent drift outside frame boundaries
            curr_x = float(np.clip(curr_x, half_w, img_w - half_w))
            curr_y = float(np.clip(curr_y, half_h, img_h - half_h))

            p_norm, _ = extract_normalized_patch(image, (curr_x, curr_y), self.patch_size)
            r0 = p_norm.reshape(-1) - t_bar

            # Iteratively Reweighted Least Squares (IRLS) with Huber loss
            z = np.concatenate([np.zeros(2, dtype=np.float64), c_coeffs])
            for _irls_it in range(4):
                residual = r0 - (a_mat @ z)

                # Dynamic Huber threshold via Median Absolute Deviation (MAD)
                med = float(np.median(residual))
                mad = float(np.median(np.abs(residual - med)))
                delta = float(self.huber_delta * max(mad, 1e-4))

                abs_res = np.abs(residual)
                weights = np.ones_like(abs_res)
                outlier_mask = abs_res > delta
                weights[outlier_mask] = delta / (abs_res[outlier_mask] + 1e-8)

                sqrt_w = np.sqrt(weights)[:, np.newaxis]
                a_w = sqrt_w * a_mat
                rhs = a_mat.T @ (weights * r0)

                hessian = a_w.T @ a_w + 1e-6 * np.eye(2 + d, dtype=np.float64)
                try:
                    z = np.linalg.solve(hessian, rhs)
                except np.linalg.LinAlgError:
                    z = np.linalg.pinv(hessian) @ rhs

            delta_p = z[:2]
            c_coeffs = z[2:]

            # Inverse compositional warp update: p <- p - delta_p
            curr_x -= float(delta_p[0])
            curr_y -= float(delta_p[1])

            # Check convergence threshold
            step_norm = float(np.hypot(delta_p[0], delta_p[1]))
            if step_norm < self.convergence_threshold:
                break

        # Final patch extraction for confidence metric
        curr_x = float(np.clip(curr_x, half_w, img_w - half_w))
        curr_y = float(np.clip(curr_y, half_h, img_h - half_h))
        p_final, _ = extract_normalized_patch(image, (curr_x, curr_y), self.patch_size)
        t_final = t_bar + (u_d @ c_coeffs)

        # Normalized cross-correlation score
        norm_p = float(np.linalg.norm(p_final))
        norm_t = float(np.linalg.norm(t_final))
        if norm_p > 1e-6 and norm_t > 1e-6:
            confidence = float(np.dot(p_final, t_final) / (norm_p * norm_t))
        else:
            confidence = 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return (curr_x, curr_y), confidence, c_coeffs

    def process(
        self,
        video_source: Any,
        total_frames: int | None = None,
        anchors: list[AnchorBlock] | None = None,
    ) -> TrackingResult:
        """Executes forward ICLK tracking and backward RTS zero-phase smoothing.

        Parameters
        ----------
        video_source : Any
            List of frames or VideoCapture-like source.
        total_frames : int, optional
            Total frames in sequence. Inferred if video_source is a list or has __len__.
        anchors : list[AnchorBlock], optional
            Anchor blocks to fix ground-truth points and high confidence on anchor frames.

        Returns
        -------
        TrackingResult
            Smoothed trajectories, velocities, accelerations, covariances, and scores.
        """
        if not self.is_calibrated:
            raise RuntimeError("Tracker must be calibrated with calibrate() before process().")

        if total_frames is None:
            if hasattr(video_source, "__len__"):
                total_frames = len(video_source)
            elif hasattr(video_source, "get"):
                total_frames = int(video_source.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                raise ValueError("total_frames must be provided when video_source has no length.")

        # Map anchor points to frame index
        anchor_map: dict[int, tuple[float, float]] = {}
        if anchors:
            for blk in anchors:
                for f_i, pt in zip(blk.frame_indices, blk.points, strict=False):
                    anchor_map[int(f_i)] = (float(pt[0]), float(pt[1]))

        measurements = np.full((total_frames, 2), np.nan, dtype=np.float64)
        is_anchor_arr = np.zeros(total_frames, dtype=bool)
        confidences = np.zeros(total_frames, dtype=np.float64)

        if not anchor_map:
            raise ValueError("No anchor points found to initialize tracking.")

        first_anchor_frame = min(anchor_map.keys())
        last_pos = anchor_map[first_anchor_frame]

        # Track forward through all frames
        for t in range(total_frames):
            frame = self._get_frame(video_source, t)
            if frame is None:
                continue

            if t in anchor_map:
                pt = anchor_map[t]
                measurements[t] = pt
                is_anchor_arr[t] = True
                confidences[t] = 1.0
                last_pos = pt
            else:
                # ICLK alignment from previous position
                opt_pos, conf, _ = self.align_patch(frame, last_pos)
                measurements[t] = opt_pos
                is_anchor_arr[t] = False
                confidences[t] = conf
                last_pos = opt_pos

        # Apply Rauch-Tung-Striebel (RTS) backward smoother
        smoothed_states, smoothed_covs = self.smoother.smooth(
            measurements=measurements,
            is_anchor=is_anchor_arr,
            confidences=confidences,
        )

        return TrackingResult(
            trajectory=smoothed_states[:, 0:2],
            velocities=smoothed_states[:, 2:4],
            accelerations=smoothed_states[:, 4:6],
            covariances=smoothed_covs,
            confidence_scores=confidences,
        )

    @staticmethod
    def _get_frame(source: Any, frame_idx: int) -> np.ndarray | None:
        """Helper to fetch frame_idx from various video source types."""
        if isinstance(source, (list, tuple)):
            if 0 <= frame_idx < len(source):
                return source[frame_idx]
            return None
        if hasattr(source, "read_frame"):
            return source.read_frame(frame_idx)
        if hasattr(source, "set") and hasattr(source, "read"):
            source.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = source.read()
            if ret:
                return frame
            return None
        if hasattr(source, "__getitem__"):
            try:
                return source[frame_idx]
            except (IndexError, KeyError):
                return None
        return None
