"""
Rauch-Tung-Striebel (RTS) Zero-Phase Kinematic Smoother for vailá.

Implements forward Kalman filtering and backward Rauch-Tung-Striebel (RTS)
smoothing under a Continuous White Noise Acceleration (CWNA) kinematic model.
Guarantees zero phase distortion (Δϕ = 0), producing accurate kinematic
trajectories, velocities, and accelerations for biomechanical analysis.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 10 September 2026
Version: 0.3.131
"""

from __future__ import annotations

import numpy as np


class CWNAStateSpace:
    """Continuous White Noise Acceleration (CWNA) 6D Kinematic State-Space Model.

    State vector:
        x = [x, y, vx, vy, ax, ay]^T in R^6

    Continuous jerk noise with spectral density sigma_a^2 yields discrete process
    covariance matrix Q.
    """

    def __init__(self, fps: float = 60.0, sigma_a: float = 100.0) -> None:
        if fps <= 0.0:
            raise ValueError(f"FPS must be positive, got {fps}")
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.sigma_a = float(sigma_a)

        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        # State transition matrix F in R^(6x6)
        # x(t+dt) = x(t) + vx*dt + 0.5*ax*dt^2
        # vx(t+dt) = vx(t) + ax*dt
        # ax(t+dt) = ax(t)
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Discrete process noise covariance Q in R^(6x6)
        # Q_block = sigma_a^2 * [[dt^4/4, dt^3/2, dt^2/2],
        #                        [dt^3/2, dt^2,   dt],
        #                        [dt^2/2, dt,     1]]
        s2 = self.sigma_a**2
        q_block = s2 * np.array(
            [
                [0.25 * dt4, 0.5 * dt3, 0.5 * dt2],
                [0.5 * dt3, dt2, dt],
                [0.5 * dt2, dt, 1.0],
            ],
            dtype=np.float64,
        )

        self.Q = np.zeros((6, 6), dtype=np.float64)
        # Map X kinematics to indices (0, 2, 4)
        ix = [0, 2, 4]
        for r_idx, r in enumerate(ix):
            for c_idx, c in enumerate(ix):
                self.Q[r, c] = q_block[r_idx, c_idx]

        # Map Y kinematics to indices (1, 3, 5)
        iy = [1, 3, 5]
        for r_idx, r in enumerate(iy):
            for c_idx, c in enumerate(iy):
                self.Q[r, c] = q_block[r_idx, c_idx]

        # Measurement matrix H in R^(2x6) (observes position [x, y]^T)
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )


class RTSSmoother:
    """Forward Kalman Filter and Backward Rauch-Tung-Striebel (RTS) Smoother.

    Guarantees zero-phase distortion (Δϕ = 0), reconstructing smoothed position,
    velocity, and acceleration profiles with optimal mean squared error.
    """

    def __init__(
        self,
        fps: float = 60.0,
        sigma_a: float = 100.0,
        sigma_manual: float = 0.8,
        sigma_track_base: float = 1.5,
    ) -> None:
        self.fps = float(fps)
        self.model = CWNAStateSpace(fps=fps, sigma_a=sigma_a)
        self.sigma_manual = float(sigma_manual)
        self.sigma_track_base = float(sigma_track_base)

    def smooth(
        self,
        measurements: np.ndarray,
        is_anchor: np.ndarray | None = None,
        confidences: np.ndarray | None = None,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Executes forward Kalman filter followed by backward RTS smoother.

        Parameters
        ----------
        measurements : np.ndarray
            Shape (T, 2) array of [x, y] coordinates. Missing or unobserved frames
            may contain NaNs.
        is_anchor : np.ndarray, optional
            Boolean array of shape (T,) indicating whether each frame is a manual
            anchor. Anchor frames receive R_manual = diag(sigma_manual^2, sigma_manual^2).
        confidences : np.ndarray, optional
            Confidence scores in [0.0, 1.0] for each frame of shape (T,). Used to scale
            measurement covariance for tracked frames.
        initial_state : np.ndarray, optional
            Initial state vector x_0 in R^6. If None, initialized from first valid measurement.
        initial_covariance : np.ndarray, optional
            Initial error covariance P_0 in R^(6x6).

        Returns
        -------
        smoothed_states : np.ndarray
            Shape (T, 6) array containing smoothed [x, y, vx, vy, ax, ay].
        smoothed_covariances : np.ndarray
            Shape (T, 6, 6) array containing smoothed error covariances P_(t|T).
        """
        t_len = len(measurements)
        if t_len == 0:
            return np.zeros((0, 6), dtype=np.float64), np.zeros((0, 6, 6), dtype=np.float64)

        if is_anchor is None:
            is_anchor = np.zeros(t_len, dtype=bool)
        else:
            is_anchor = np.asarray(is_anchor, dtype=bool)

        if confidences is None:
            confidences = np.ones(t_len, dtype=np.float64)
        else:
            confidences = np.clip(np.asarray(confidences, dtype=np.float64), 0.01, 1.0)

        # Preallocate forward filter arrays
        x_pred = np.zeros((t_len, 6), dtype=np.float64)
        p_pred = np.zeros((t_len, 6, 6), dtype=np.float64)
        x_filt = np.zeros((t_len, 6), dtype=np.float64)
        p_filt = np.zeros((t_len, 6, 6), dtype=np.float64)

        f_mat = self.model.F
        q_mat = self.model.Q
        h_mat = self.model.H
        i_mat = np.eye(6, dtype=np.float64)

        # Find first valid measurement to initialize
        valid_mask = ~np.isnan(measurements[:, 0]) & ~np.isnan(measurements[:, 1])
        if not np.any(valid_mask):
            # All measurements are NaN
            return np.zeros((t_len, 6), dtype=np.float64), np.zeros((t_len, 6, 6), dtype=np.float64)

        first_valid_idx = int(np.argmax(valid_mask))
        init_x, init_y = measurements[first_valid_idx]

        # Estimate rough initial velocity if next valid measurement exists
        init_vx, init_vy = 0.0, 0.0
        for next_idx in range(first_valid_idx + 1, t_len):
            if valid_mask[next_idx]:
                n_dt = (next_idx - first_valid_idx) * self.model.dt
                if n_dt > 0:
                    init_vx = (measurements[next_idx, 0] - init_x) / n_dt
                    init_vy = (measurements[next_idx, 1] - init_y) / n_dt
                break

        if initial_state is not None:
            x_curr = np.asarray(initial_state, dtype=np.float64).copy()
        else:
            x_curr = np.array([init_x, init_y, init_vx, init_vy, 0.0, 0.0], dtype=np.float64)

        if initial_covariance is not None:
            p_curr = np.asarray(initial_covariance, dtype=np.float64).copy()
        else:
            p_curr = np.diag([1.0, 1.0, 25.0, 25.0, 100.0, 100.0]).astype(np.float64)

        # Forward Kalman Filter pass
        for t in range(t_len):
            if t == 0:
                x_prior = x_curr.copy()
                p_prior = p_curr.copy()
            else:
                x_prior = f_mat @ x_curr
                p_prior = f_mat @ p_curr @ f_mat.T + q_mat
                p_prior = 0.5 * (p_prior + p_prior.T)

            x_pred[t] = x_prior
            p_pred[t] = p_prior

            # Check if measurement is available at frame t
            z_t = measurements[t]
            if not np.isnan(z_t[0]) and not np.isnan(z_t[1]):
                # Determine measurement noise covariance R
                if is_anchor[t]:
                    r_val = self.sigma_manual**2
                else:
                    conf = float(confidences[t])
                    r_val = (self.sigma_track_base / np.sqrt(max(conf, 0.05))) ** 2

                r_mat = np.diag([r_val, r_val]).astype(np.float64)

                # Innovation
                y_innov = z_t - (h_mat @ x_prior)
                s_mat = h_mat @ p_prior @ h_mat.T + r_mat
                # Kalman gain
                k_gain = p_prior @ h_mat.T @ np.linalg.inv(s_mat)

                # State update
                x_curr = x_prior + (k_gain @ y_innov)

                # Joseph form covariance update for numerical stability
                i_kh = i_mat - (k_gain @ h_mat)
                p_curr = (i_kh @ p_prior @ i_kh.T) + (k_gain @ r_mat @ k_gain.T)
                p_curr = 0.5 * (p_curr + p_curr.T)
            else:
                # No measurement: state and covariance remain the predicted ones
                x_curr = x_prior
                p_curr = p_prior

            x_filt[t] = x_curr
            p_filt[t] = p_curr

        # Backward Rauch-Tung-Striebel (RTS) Smoother pass
        smoothed_states = np.zeros((t_len, 6), dtype=np.float64)
        smoothed_covariances = np.zeros((t_len, 6, 6), dtype=np.float64)

        smoothed_states[-1] = x_filt[-1]
        smoothed_covariances[-1] = p_filt[-1]

        for t in range(t_len - 2, -1, -1):
            p_f = p_filt[t]
            p_p_next = p_pred[t + 1]

            # Smoother gain C_t = P_(t|t) * F^T * [P_(t+1|t)]^(-1)
            # Regularize P_(t+1|t) slightly if nearly singular
            reg = 1e-9 * np.eye(6, dtype=np.float64)
            try:
                c_gain = np.linalg.solve(p_p_next.T + reg, (p_f @ f_mat.T).T).T
            except np.linalg.LinAlgError:
                c_gain = p_f @ f_mat.T @ np.linalg.pinv(p_p_next)

            x_next_diff = smoothed_states[t + 1] - x_pred[t + 1]
            p_next_diff = smoothed_covariances[t + 1] - p_p_next

            smoothed_states[t] = x_filt[t] + (c_gain @ x_next_diff)
            p_smooth = p_f + (c_gain @ p_next_diff @ c_gain.T)
            smoothed_covariances[t] = 0.5 * (p_smooth + p_smooth.T)

        return smoothed_states, smoothed_covariances
