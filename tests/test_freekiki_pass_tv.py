"""Smoke test for the FreeKiki pass TV package (no video encoding)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import freekiki_pass as pa
from vaila import freekiki_pass_tv as ptv


def _run(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    video = folder.parent / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (160, 90))
    for _ in range(25):
        writer.write(np.full((90, 160, 3), 60, dtype=np.uint8))
    writer.release()
    (folder / "README.txt").write_text(f"FreeKiki 3D output\nvideo: {video}\n", encoding="utf-8")
    ball = ["frame,x_m,y_m,z_m,on_field\n"]
    pix = ["frame,cx,cy,score\n"]
    ground = ["frame,person_id,x_m,y_m\n"]
    for f in range(25):
        t = f / 25.0
        ball.append(f"{f},{0.5 + 14.0 * t},0.0,0.11,True\n")
        pix.append(f"{f},{20 + 4 * f},45,0.9\n")
        ground.append(f"{f},21,0.0,-0.5\n")
        ground.append(f"{f},51,{12.0 + 2.0 * t},{4.0 - 4.0 * t}\n")
    (folder / "ball_3d_on_field.csv").write_text("".join(ball), encoding="utf-8")
    (folder / "ball.csv").write_text("".join(pix), encoding="utf-8")
    (folder / "ground_tracks.csv").write_text("".join(ground), encoding="utf-8")
    (folder / "ball_flight.json").write_text(
        json.dumps({"kick_time_s": -0.03, "fps": 25.0}), encoding="utf-8"
    )
    return folder


def test_scenes_cover_sixty_seconds_without_shot(tmp_path: Path) -> None:
    folder = _run(tmp_path / "pass_3d")
    analysis = pa.run_pass(folder, 21, 51, arrival_frame=24)
    scenes = ptv.build_scenes(analysis)
    assert scenes[0][0] == 0 and scenes[-1][1] == ptv.TV_SECONDS
    assert all(a[1] == b[0] for a, b in zip(scenes[:-1], scenes[1:], strict=True))
    assert "shot" not in {s[5] for s in scenes}
    assert any("já corria" in s[4] for s in scenes)


def test_package_without_animation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ptv, "print_pdf", lambda html, pdf: None)
    folder = _run(tmp_path / "pass_3d")
    pa.run_pass(folder, 21, 51, arrival_frame=24)
    out = tmp_path / "pacote"
    result = ptv.build_pass_tv_package(folder, out, animation=False)
    assert result["dir"] == str(out.resolve())
    for name in (
        "conferencia_saida.jpg",
        "conferencia_primeiro_toque.jpg",
        "roteiro_tv.md",
        "explicacao_tv.srt",
        "roteiro_legendas.csv",
        "README_reproducao.md",
        "inputs_sha256.json",
        "outputs_sha256.json",
        "validacao_entrega.json",
    ):
        assert (out / name).is_file(), name
    assert (out / "data" / "pass_timeseries.csv").is_file()
    assert result["checks"]["csv_rows_match_frames"] is True
    with pytest.raises(ValueError, match="separate folder"):
        ptv.build_pass_tv_package(folder, folder, animation=False)


def _rolling_ball(t: np.ndarray) -> np.ndarray:
    """A decelerating ground pass along a diagonal (pitch metres) at times ``t`` (frames)."""
    s = 0.9 * t - 0.012 * t**2
    return np.column_stack([-30.0 - 0.6 * s, -22.0 + 0.8 * s])


def test_sync_by_ball_recovers_a_ramped_slow_replay() -> None:
    ref_t = np.arange(26, dtype=float)
    ref = _rolling_ball(ref_t)
    true_t = np.cumsum(np.r_[0.0, np.linspace(0.85, 0.6, 38)])  # 39 frames, slower and ramped
    other = _rolling_ball(true_t)
    sync = ptv.sync_by_ball(ref, other)
    ok = np.isfinite(sync["t_ref"])
    assert np.array_equal(ok, true_t <= 25.0)  # beyond the reference clip: unmapped
    assert np.max(np.abs(sync["t_ref"][ok] - true_t[ok])) < 0.05
    assert sync["median_m"] < 0.05
    lo, hi = sync["slowmo_range"]
    assert 1.1 < lo < hi < 1.8


def test_match_ids_takes_the_right_track_and_refuses_a_far_or_ambiguous_one() -> None:
    t_ref = np.arange(20, dtype=float)
    ref = {51: np.column_stack([np.linspace(-34, -38, 20), np.full(20, -10.0)])}
    exact = {j: (float(ref[51][j, 0]) + 0.2, -10.0) for j in range(20)}
    far = {j: (float(ref[51][j, 0]), -13.0) for j in range(20)}
    got = ptv.match_ids(ref, {13: exact, 19: far}, t_ref)
    assert got[51]["other_id"] == 13 and got[51]["accepted"] is True
    only_far = ptv.match_ids(ref, {19: far}, t_ref)
    assert only_far[51]["accepted"] is False
    twin = {j: (float(ref[51][j, 0]) + 0.25, -10.0) for j in range(20)}
    ambiguous = ptv.match_ids(ref, {13: exact, 19: twin}, t_ref)
    assert ambiguous[51]["accepted"] is False
