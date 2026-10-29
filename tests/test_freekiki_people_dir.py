"""The report draws the people of the reference camera when the flight folder has no poses.

A binocular folder (``rec3d-binocular``) holds ``ball_flight.json`` but not the poses or the
player map; its ``summary.json`` names the reference camera's 3D folder (0.4.12).
"""

from __future__ import annotations

import json
from pathlib import Path

from vaila.freekiki_report import _people_dir


def _touch(folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("x\n", encoding="utf-8")


def test_a_folder_with_its_own_poses_is_its_own_people_folder(tmp_path: Path) -> None:
    _touch(tmp_path, "poses_3d_on_field.csv")
    assert _people_dir(tmp_path) == tmp_path


def test_a_binocular_folder_uses_the_reference_run_of_its_summary(tmp_path: Path) -> None:
    ref = tmp_path / "c1_3d"
    _touch(ref, "poses_3d_on_field.csv")
    binocular = tmp_path / "processed_rec3d_binocular"
    binocular.mkdir()
    (binocular / "summary.json").write_text(json.dumps({"reference_run": str(ref)}), "utf-8")
    assert _people_dir(binocular) == ref


def test_a_missing_reference_keeps_the_folder_itself(tmp_path: Path) -> None:
    binocular = tmp_path / "processed_rec3d_binocular"
    binocular.mkdir()
    (binocular / "summary.json").write_text(
        json.dumps({"reference_run": str(tmp_path / "moved_away")}), "utf-8"
    )
    assert _people_dir(binocular) == binocular
    assert _people_dir(tmp_path / "no_summary") == tmp_path / "no_summary"
