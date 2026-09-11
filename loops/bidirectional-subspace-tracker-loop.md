---
name: bidirectional-subspace-tracker-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---
# vailá Bidirectional Subspace Appearance Tracking & RTS Zero-Phase Smoother Loop

## Description
Drift-resistant kinematic tracking module for `vailá`: multi-anchor linear appearance subspace model (Eigen-templates via SVD), M-estimator w/ Huber loss via IRLS, bidirectional temporal fusion via Rauch-Tung-Striebel (RTS) smoother for zero-phase distortion (Delta phi = 0). Integrates into `vaila/getpixelvideo.py` w/ dedicated "Track RTS" button + hotkey (T).

## Use When
- Tracking high-speed, non-rigid, or rotating biomechanical targets (anatomical landmarks, athletes, balls, joints) across video where standard Normalized Cross-Correlation (Kinovea / Block Matching) drifts or fails.
- Need zero-phase distortion (Delta phi = 0) so velocity/acceleration time series avoid causal filtering delay.
- Calibrating appearance variation from 3 temporal anchor blocks (30 frames total: 10 start, 10 mid, 10 end).

## Inputs
1. `subspace_tracker` — `vaila/tracking/subspace_tracker.py` (SVD eigenbasis, Huber ICLK solver, AnchorBlock, TrackingResult).
2. `rts_smoother` — `vaila/tracking/rts_smoother.py` (CWNA state space, forward Kalman, backward RTS smoother).
3. `gui` — `vaila/getpixelvideo.py` ("Track RTS" toolbar button, hotkey T, coordinates population, kinematics CSV export).
4. `tests` — `tests/test_subspace_tracker.py`, `tests/test_getpixelvideo_media_classify.py`.
5. `help` — `vaila/help/getpixelvideo.md`, `vaila/help/getpixelvideo.html`, `vaila/help/index.md`, `vaila/help/index.html`.

## Goal
Objectively verifiable tracking pipeline:
1. SVD eigenbasis decomposition captures >= 95% appearance variation across 30 anchor patches.
2. Robust ICLK Huber solver converges sub-pixel accuracy (RMSE < 0.08 px), downweights outlier occlusions.
3. Backward RTS smoother achieves zero-phase distortion (cross-correlation peak at lag = 0).
4. "Track RTS" button + hotkey T in `getpixelvideo.py` track markers all frames, export kinematics CSV.
5. All automated unit + integration tests pass clean.