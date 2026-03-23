"""Unified launcher for sports fields/courts visualization."""
from __future__ import annotations

from pathlib import Path

try:
    from . import soccerfield, tennis_court
except ImportError:
    import soccerfield
    import tennis_court


_SPORT_LABELS: dict[str, str] = {
    "soccer": "Soccer Field Visualization",
    "basketball": "Basketball Court Visualization",
    "volleyball": "Volleyball Court Visualization",
    "futsal": "Futsal Court Visualization",
    "handball": "Handball Court Visualization",
}

_SPORT_MODEL_FILES: dict[str, str] = {
    "basketball": "basketballcourt_ref3d.csv",
    "volleyball": "volleyball_ref3d.csv",
    "futsal": "futsal_ref3d.csv",
    "handball": "handball_ref3d.csv",
}


def _model_csv_path(surface: str) -> Path:
    """Return absolute CSV model path for non-soccer/tennis surfaces."""
    models_dir = Path(__file__).resolve().parent / "models"
    return models_dir / _SPORT_MODEL_FILES[surface]


def run_drawsportsfields(surface: str) -> None:
    """Open selected sports surface in the appropriate visualizer."""
    surface_key = (surface or "").strip().lower()

    if surface_key == "tennis":
        tennis_court.run_tenniscourt()
        return

    if surface_key == "soccer":
        soccerfield.run_soccerfield(
            window_title=_SPORT_LABELS["soccer"],
            help_html="sports_fields_courts.html",
        )
        return

    if surface_key in _SPORT_MODEL_FILES:
        model_csv = _model_csv_path(surface_key)
        soccerfield.run_soccerfield(
            initial_field_csv=str(model_csv),
            default_field_csv=str(model_csv),
            window_title=_SPORT_LABELS[surface_key],
            help_html="sports_fields_courts.html",
        )
        return

    raise ValueError(f"Unknown sports surface: {surface!r}")


if __name__ == "__main__":
    run_drawsportsfields("soccer")
