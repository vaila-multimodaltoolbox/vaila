"""
================================================================================
Script: kiki49_build/__init__.py - external field datasets to kiki49 labels
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 06 October 2026
Update Date: 06 October 2026

Description:
    Converters that turn the native annotations of an external soccer-field
    dataset (human clicks in another keypoint layout, SoccerNet line
    annotations) into complete 49-point FreeKiki labels: annotated points are
    kept, every other point in view is projected through a fitted homography or
    camera, and projected labels must sit on painted lines (line support).
    Ported from mkvis3d ``openbiomech/soccer_field``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

FIFA_ROOT = Path("/media/preto/Expansion/FIFA")
DATASET_ROOT = FIFA_ROOT / "dataset_vaila_fifa" / "sources"


@dataclass(frozen=True)
class Options:
    aux3d: bool = True  # label posts tops / net ground / flag tops when the camera is trusted
    tau: float = 0.35  # minimum line support for projected (not human-annotated) labels


def hashed_split(group: str, val: float = 0.15, test: float = 0.15) -> str:
    """Deterministic split for sources without an official one (by group)."""
    u = int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else "val" if u < test + val else "train"
