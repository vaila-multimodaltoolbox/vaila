"""
vailá Tracking Module.

Robust kinematic tracking with linear appearance subspace modeling (Eigen-templates via SVD),
M-estimator ICLK solver with Huber loss, and Rauch-Tung-Striebel (RTS) zero-phase smoothing.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 18 September 2026
Version: 0.4.4
"""

from .ai_tracker import (
    AITracker,
    AITrackerParameters,
    DeepFeatureExtractor,
    KinoveaTracker,
    KinoveaTrackerParameters,
    TemplateMatchResult,
    default_checkpoint_path,
    download_backbone_weights,
    ensure_backbone_weights,
    get_available_resnet50_checkpoints,
    get_available_resnet_checkpoints,
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
    "default_checkpoint_path",
    "download_backbone_weights",
    "ensure_backbone_weights",
    "extract_normalized_patch",
    "get_available_resnet50_checkpoints",
    "get_available_resnet_checkpoints",
    "infill_and_smooth",
    "refine_location_parabola",
]
