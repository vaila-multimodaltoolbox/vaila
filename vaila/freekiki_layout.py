"""
================================================================================
Script: freekiki_layout.py - where the files of an analysis folder live
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 28 October 2026
Update Date: 28 October 2026

Description:
    An analysis folder (the binocular ``processed_rec3d_binocular_<time>/`` or the 3D folder of one
    camera) keeps its tables and its fit in ``data/``. Its folders are the reports (``report_tv/``) and
    the teams (``team_A/``, ``team_B/``, ``referees/``). :func:`resolve` reads a file from ``data/`` when
    it is there and from the folder itself otherwise, so the older runs still open.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = "data"
TEAM_DIRS = ("team_A", "team_B", "referees")


def data_dir(folder: Path | str) -> Path:
    """The ``data/`` folder of an analysis (created when missing)."""
    path = Path(folder) / DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(folder: Path | str, name: str) -> Path:
    """``data/<name>`` when it exists, else ``<folder>/<name>`` (which may not exist either)."""
    candidate = Path(folder) / DATA_DIR / name
    if candidate.exists():
        return candidate
    return Path(folder) / name
