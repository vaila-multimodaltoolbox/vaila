"""Handball court launcher using the sports field viewer."""
from __future__ import annotations

from pathlib import Path

try:
    from . import soccerfield
except ImportError:
    import soccerfield


def run_handball_court() -> None:
    """Open the handball court model in the field viewer."""
    models_dir = Path(__file__).resolve().parent / "models"
    model_csv = models_dir / "handball_ref3d.csv"
    soccerfield.run_soccerfield(
        initial_field_csv=str(model_csv),
        default_field_csv=str(model_csv),
        window_title="Handball Court Visualization",
        help_html="sports_fields_courts.html",
    )


if __name__ == "__main__":
    run_handball_court()
