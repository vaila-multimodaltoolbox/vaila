"""Where the Practical-RIFE code and weights live: portable and chosen by the user.

Update Date: 30 October 2026
Version: 0.4.13

The RIFE assets are ~50 MB. They are not forced into one place: a RIFE *home* is any
folder holding ``Practical-RIFE/`` (the pinned upstream code) and ``train_log/`` (the
matching ``flownet.pkl`` and model files). Copy that folder to another disk or computer
and point vailá at it. The home is chosen, in this order, by

1. ``--rife-home DIR`` (CLI) or the RIFE fields of the equalizer window;
2. the environment variable ``VAILA_RIFE_HOME``;
3. the legacy folders inside the vailá checkout (``.local/third_party/Practical-RIFE`` and
   ``vaila/models/rife/train_log``), kept so existing installs keep working.

``--rife-dir`` / ``--weights`` override just one of the two folders. An explicitly chosen
home that does not hold RIFE is an error; it never silently falls back to another copy.

Install or copy a home with ``bin/setup_rife.py --home DIR`` / ``--export DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

ENV_HOME = "VAILA_RIFE_HOME"
CHECKOUT_DIR = "Practical-RIFE"
WEIGHTS_DIR = "train_log"
MANIFEST_NAME = "setup_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CHECKOUT = REPO_ROOT / ".local/third_party/Practical-RIFE"
LEGACY_WEIGHTS = REPO_ROOT / "vaila/models/rife/train_log"


class RifePaths(NamedTuple):
    checkout: Path
    weights: Path
    source: str  # where the choice came from, for messages


def home_layout(home: Path | str) -> tuple[Path, Path]:
    """(checkout, weights) of a portable RIFE home."""
    home = Path(home).expanduser()
    return home / CHECKOUT_DIR, home / WEIGHTS_DIR


def env_home() -> Path | None:
    """The home named by VAILA_RIFE_HOME, if set and not blank."""
    value = os.environ.get(ENV_HOME, "").strip()
    return Path(value).expanduser() if value else None


def resolve_paths(
    rife_dir: Path | str | None = None,
    weights: Path | str | None = None,
    home: Path | str | None = None,
) -> RifePaths:
    """Resolves the code and weight folders; see the module docstring for the order."""
    chosen, source = (Path(home).expanduser(), "--rife-home") if home else (None, "")
    if chosen is None:
        chosen = env_home()
        source = f"${ENV_HOME}"
    if chosen is not None:
        checkout, model = home_layout(chosen)
    else:
        checkout, model, source = LEGACY_CHECKOUT, LEGACY_WEIGHTS, "the vailá checkout"
    if rife_dir:
        checkout, source = Path(rife_dir).expanduser(), f"{source} + --rife-dir"
    if weights:
        model, source = Path(weights).expanduser(), f"{source} + --weights"
    return RifePaths(checkout, model, source)


def missing_message(paths: RifePaths) -> str:
    """How to fix a RIFE location that does not hold the code or the weights."""
    return (
        f"RIFE not found (from {paths.source}): code {paths.checkout}, weights {paths.weights}. "
        f"Choose the folder with --rife-home DIR or ${ENV_HOME}, or install/copy one with "
        "'bash bin/setup_rife.sh --home DIR' / '--export DIR' (see Syncvid Help → RIFE setup)."
    )
