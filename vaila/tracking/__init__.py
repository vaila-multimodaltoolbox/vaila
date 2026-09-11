"""
vailá Tracking Module.

Robust kinematic tracking with linear appearance subspace modeling (Eigen-templates via SVD),
M-estimator ICLK solver with Huber loss, and Rauch-Tung-Striebel (RTS) zero-phase smoothing.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 10 September 2026
Version: 0.3.131
"""

from .ai_tracker import (
    AITracker,
    AITrackerParameters,
    DeepFeatureExtractor,
    KinoveaTracker,
    KinoveaTrackerParameters,
    TemplateMatchResult,
    infill_and_smooth,
    refine_location_parabola,
)
from .rts_smoother import CWNAStateSpace, RTSSmoother
from .subspace_tracker import (
    AnchorBlock,
    BidirectionalSubspaceTracker,
    TrackingResult,
    extract_normalized_patch,
)

__all__ = [
    "AITracker",
    "AITrackerParameters",
    "AnchorBlock",
    "BidirectionalSubspaceTracker",
    "CWNAStateSpace",
    "DeepFeatureExtractor",
    "KinoveaTracker",
    "KinoveaTrackerParameters",
    "RTSSmoother",
    "TemplateMatchResult",
    "TrackingResult",
    "extract_normalized_patch",
    "infill_and_smooth",
    "refine_location_parabola",
]

