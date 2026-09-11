"""
Compatibility module forwarding to vaila.tracking.ai_tracker.

Maintained for backward compatibility with existing tests and scripts.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 10 September 2026
Version: 0.3.134
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

__all__ = [
    "AITracker",
    "AITrackerParameters",
    "DeepFeatureExtractor",
    "KinoveaTracker",
    "KinoveaTrackerParameters",
    "TemplateMatchResult",
    "infill_and_smooth",
    "refine_location_parabola",
]
