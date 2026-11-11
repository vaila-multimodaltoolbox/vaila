"""Synthetic FreeKiki → Blender export (no Blender process)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from vaila.freekiki_blender import (
    _BODY_BONES,
    _companion_script,
    export_penalty_blender,
    write_bvh,
)


def _write_person_csv(path: Path, sam_id: int, n: int = 8, fps: float = 30.0) -> None:
    labels = [
        "nose",
        "left-shoulder",
        "right-shoulder",
        "left-elbow",
        "right-elbow",
        "left-wrist",
        "right-wrist",
        "left-hip",
        "right-hip",
        "left-knee",
        "right-knee",
        "left-ankle",
        "right-ankle",
        "neck",
    ]
    header = ["frame", "time_s"]
    for lab in labels:
        header += [f"P{sam_id}_{lab}_X", f"P{sam_id}_{lab}_Y", f"P{sam_id}_{lab}_Z"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for f in range(n):
            row: list[object] = [f, f / fps]
            for j, _lab in enumerate(labels):
                row += [40.0 + f * 0.05, float(j) * 0.1, 1.0 + 0.01 * j]
            writer.writerow(row)


def _write_ball_csv(path: Path, n: int = 8, fps: float = 30.0) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "time_s", "BALL_X", "BALL_Y", "BALL_Z"])
        for f in range(n):
            writer.writerow([f, f / fps, 41.0 + f * 1.2, 0.1, 0.2 + f * 0.05])


def test_write_bvh_swap_yz(tmp_path: Path) -> None:
    labels = ["nose", "left-hip"]
    xyz = np.zeros((3, 2, 3), dtype=float)
    xyz[:, 0] = [1.0, 2.0, 3.0]
    xyz[:, 1] = [4.0, 5.0, 6.0]
    path = write_bvh(tmp_path / "t.bvh", labels, xyz, fps=30.0, swap_yz=True)
    text = path.read_text(encoding="utf-8")
    assert "ROOT nose" in text
    assert "Frames: 3" in text
    # (x,y,z)=(1,2,3) → (x,z,-y)=(1,3,-2)
    assert "1.000000 3.000000 -2.000000" in text


def test_companion_mentions_bvh_goal_and_events() -> None:
    src = _companion_script(
        Path("/tmp/blender_out"),
        keeper_bvh="keeper.bvh",
        kicker_bvh="kicker.bvh",
        ball_json="ball_path.json",
        fps=29.97,
        n_frames=91,
        events={"kick_frame": 80, "gk_move_frame": 74, "save_frame": 90},
        goal_x=52.45,
    )
    assert "keeper.bvh" in src
    assert "kicker.bvh" in src
    assert "goal_mouth" in src
    assert "KICK_F = 80" in src
    assert "ONSET_F = 74" in src
    assert "SAVE_F = 90" in src
    assert "import_anim.bvh" in src
    # Goal corners must stay in Blender Z-up pitch coordinates [x, y, z]
    assert "[52.45, -3.66, 2.44]" in src
    assert "plane.rotation_euler" not in src
    for a, _b in _BODY_BONES[:3]:
        assert a in json.dumps(_BODY_BONES)


def test_export_penalty_blender_writes_package(tmp_path: Path) -> None:
    cam = tmp_path / "c1_main_3d_test"
    cam.mkdir()
    (cam / "data").mkdir()
    with (cam / "player_map.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["player_id", "role", "sam_id"])
        writer.writerow([0, "keeper", 39])
        writer.writerow([1, "kicker", 15])
    _write_person_csv(cam / "person_039_c3d.csv", 39)
    _write_person_csv(cam / "person_015_c3d.csv", 15)
    _write_ball_csv(cam / "ball_c3d.csv")
    (cam / "data" / "penalty_metrics.json").write_text(
        json.dumps(
            {
                "goal_x_m": 52.45,
                "kick_frame": 2,
                "gk_move_frame": 1,
                "save_frame": 7,
                "event": "save",
            }
        ),
        encoding="utf-8",
    )
    written = export_penalty_blender(cam, render=False)
    assert written["keeper_bvh"].is_file()
    assert written["kicker_bvh"].is_file()
    assert written["ball_json"].is_file()
    script = written["script"].read_text(encoding="utf-8")
    assert "keeper.bvh" in script
    assert "KICK_F = 2" in script
    assert "SAVE_F = 7" in script
    ball = json.loads(written["ball_json"].read_text(encoding="utf-8"))
    assert len(ball["xyz"]) == 8
