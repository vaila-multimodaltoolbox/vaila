"""Binocular ball reconstruction (vaila/freekiki_stereo.py) on synthetic cameras with known truth.

Two static cameras look at a known ball track; their pixels are generated from that track, so the
reconstruction must return it. Clock and sync cases are built on purpose: a wrong clock must show as
larger reprojection errors, and missing or far-away data must be reported, not filled in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import freekiki_stereo as st
from vaila.fifa_to_dlt import compute_dlt3d_from_KRt
from vaila.monocular_dlt_align import dlt_projection_matrix

W, H = 1920, 1080
FPS = 60.0
SYNC_CUT = 13  # the sync frame in the cut of both cameras
K = np.array([[2400.0, 0.0, W / 2.0], [0.0, 2400.0, H / 2.0], [0.0, 0.0, 1.0]])


def _camera(centre, target) -> np.ndarray:
    """DLT coefficients of a camera at ``centre`` looking at ``target`` (pitch metres -> pixels)."""
    centre, target = np.asarray(centre, float), np.asarray(target, float)
    z = target - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    return compute_dlt3d_from_KRt(K, R, -R @ centre)


def _truth(tau: np.ndarray) -> np.ndarray:
    """A kick-like track: rises to about 3 m and falls, in pitch metres. tau in seconds."""
    x = -30.0 + 20.0 * tau
    y = -20.0 + 8.0 * tau
    z = 0.11 + 6.0 * tau - 4.9 * tau**2
    return np.column_stack([x, y, z])


def _pix(L: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    P = dlt_projection_matrix(L)
    h = P @ np.append(xyz, 1.0)
    return h[:2] / h[2]


L_A = _camera([-60.0, -45.0, 25.0], [-30.0, -10.0, 0.0])
L_B = _camera([-75.0, 20.0, 22.0], [-30.0, -10.0, 0.0])


def _view(name: str, L: np.ndarray, markers: dict, pts: np.ndarray, cut_start: int) -> st.CamView:
    return st.CamView(
        name=name,
        dlt=dict.fromkeys(range(pts.size), L),
        markers=markers,
        cut_start=cut_start,
        sync_cut=SYNC_CUT,
        pts_cut=pts,
    )


def _markers_from_truth(L: np.ndarray, frames, pts: np.ndarray, offset_s: float = 0.0) -> dict:
    """Ball pixels of a camera at its cut frames, for the truth at the instant of each frame.

    The truth time of a frame is its clock time relative to the sync instant, plus ``offset_s`` on
    the camera's own clock: a nonzero offset makes the markers come from the wrong instant.
    """
    out = {}
    for f in frames:
        t_rel = pts[f] - pts[SYNC_CUT]
        tau = 0.6 + t_rel + offset_s
        out[f] = _pix(L, _truth(np.array([tau]))[0])
    return out


@pytest.fixture
def pair() -> tuple[st.CamView, st.CamView, np.ndarray]:
    pts = np.arange(78) / FPS
    frames = [f for f in range(78) if 0.25 <= 0.6 + (pts[f] - pts[SYNC_CUT]) <= 1.15]
    view_a = _view("a", L_A, _markers_from_truth(L_A, frames, pts), pts, cut_start=2)
    view_b = _view("b", L_B, _markers_from_truth(L_B, frames, pts), pts, cut_start=1)
    return view_a, view_b, np.array(frames)


def test_parse_sync_name_reads_sync_and_cut_start() -> None:
    assert st.parse_sync_name("c1_main_sync_15_frames_2_to_79") == (15, 2)
    assert st.parse_sync_name("c2_16mm_sync_14_frames_1_to_78") == (14, 1)
    assert st.parse_sync_name("plain_clip") is None


def _equalizer_folder(folder: Path, *, contact: int = 33, frame_base: int = 1) -> list[Path]:
    """Two videos and the ``event_equalizer.json`` that video_event_equalizer writes next to them."""
    names = {"reference": "reference_c2_equalized.mp4", "source": "source_c1_equalized.mp4"}
    for name in names.values():
        (folder / name).write_bytes(b"")
    manifest = {
        "vaila_event_equalizer": 1,
        "frame_base": frame_base,
        "output_event_frames": [frame_base, contact, 85],
        "cameras": {role: {"video": name} for role, name in names.items()},
    }
    (folder / st.EQUALIZER_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return [folder / names["reference"], folder / names["source"]]


def test_the_equalizer_videos_are_the_reference_and_the_source_in_that_order(
    tmp_path: Path,
) -> None:
    reference, source = _equalizer_folder(tmp_path)
    assert st.equalizer_videos(tmp_path) == [reference, source]


def test_the_equalizer_sync_is_the_contact_frame_counted_from_zero(tmp_path: Path) -> None:
    reference, source = _equalizer_folder(tmp_path, contact=33)
    assert st.equalizer_sync(reference) == st.equalizer_sync(source) == (32, 0)
    assert st.video_sync(reference) == (32, 0)
    zero_based = tmp_path / "zero"
    zero_based.mkdir()
    (first, _) = _equalizer_folder(zero_based, contact=32, frame_base=0)
    assert st.equalizer_sync(first) == (32, 0)


def test_the_sync_in_the_name_wins_over_the_equalizer(tmp_path: Path) -> None:
    _equalizer_folder(tmp_path)
    named = tmp_path / "c1_main_sync_15_frames_2_to_79.mp4"
    named.write_bytes(b"")
    assert st.video_sync(named) == (15, 2)


def test_load_view_reads_the_sync_of_an_equalized_video_from_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference, _ = _equalizer_folder(tmp_path)
    run_dir = tmp_path / "reference_c2_equalized_3d_20261029_000000"
    run_dir.mkdir()
    (run_dir / "README.txt").write_text(f"video: {reference}\n", encoding="utf-8")
    (tmp_path / "reference_c2_equalized_markers.csv").write_text(
        "frame,p0_x,p0_y\n0,1,1\n", encoding="utf-8"
    )
    monkeypatch.setattr(st, "nominal_clock", lambda path: np.arange(85) / 120.0)
    monkeypatch.setattr(st, "find_dlt3d", lambda folder, video: {})
    view, video = st.load_view(
        run_dir, markers=None, timing=None, sync_frame=None, cut_start=None, name="ref"
    )
    assert video == reference
    assert (view.cut_start, view.sync_cut) == (0, 32)  # no ValueError: the name has no sync


def test_a_video_the_manifest_does_not_name_has_no_equalizer_sync(tmp_path: Path) -> None:
    _equalizer_folder(tmp_path)
    other = tmp_path / "another.mp4"
    other.write_bytes(b"")
    assert st.video_sync(other) is None


def test_a_broken_or_foreign_manifest_gives_no_sync(tmp_path: Path) -> None:
    videos = _equalizer_folder(tmp_path)
    manifest = tmp_path / st.EQUALIZER_MANIFEST
    manifest.write_text("{not json", encoding="utf-8")
    assert st.video_sync(videos[0]) is None
    manifest.write_text(json.dumps({"output_event_frames": [1, 33, 85]}), encoding="utf-8")
    assert st.video_sync(videos[0]) is None  # not written by the equalizer
    assert st.equalizer_videos(tmp_path) == []


def test_a_manifest_naming_a_missing_video_gives_no_pair(tmp_path: Path) -> None:
    videos = _equalizer_folder(tmp_path)
    videos[1].unlink()
    assert st.equalizer_videos(tmp_path) == []
    assert st.video_sync(videos[0]) is None


def test_triangulate_recovers_a_known_point() -> None:
    X = np.array([-40.0, -3.0, 1.7])
    uv_a, uv_b = _pix(L_A, X), _pix(L_B, X)
    xyz, res_a, res_b, angle = st.triangulate(L_A, L_B, uv_a, uv_b)
    assert np.allclose(xyz, X, atol=1e-6)
    assert res_a < 1e-6 and res_b < 1e-6
    centre_a = st.projection_centre(dlt_projection_matrix(L_A))
    centre_b = st.projection_centre(dlt_projection_matrix(L_B))
    ra, rb = centre_a - X, centre_b - X
    expected = np.degrees(np.arccos(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb))))
    assert angle == pytest.approx(expected, abs=1e-6)  # the angle the two rays make at the ball


def test_reconstruction_recovers_the_track_when_clocks_agree(pair) -> None:
    view_a, view_b, frames = pair
    rows = st.reconstruct_ball(view_a, view_b)
    valid = [r for r in rows if r["valid"]]
    assert len(valid) == len(frames)
    for r in valid:
        f = r["frame"]
        t_rel = view_a.pts_cut[f] - view_a.pts_cut[SYNC_CUT]
        truth = _truth(np.array([0.6 + t_rel]))[0]
        assert np.allclose([r["x_m"], r["y_m"], r["z_m"]], truth, atol=1e-4)


def test_a_wrong_clock_shows_as_larger_reprojection_error(pair) -> None:
    view_a, view_b, frames = pair
    good = st.summarize(st.reconstruct_ball(view_a, view_b))
    # the second camera's clock is 50 ms late: its pixels are from another instant
    pts_late = view_b.pts_cut + 0.05
    late_b = st.CamView(
        name="b_late",
        dlt=view_b.dlt,
        markers=_markers_from_truth(L_B, frames, pts_late - 0.05, offset_s=0.05),
        cut_start=view_b.cut_start,
        sync_cut=SYNC_CUT,
        pts_cut=pts_late,
    )
    bad = st.summarize(st.reconstruct_ball(view_a, late_b))
    assert good["median_resid_other_px"] < 0.01
    assert bad["median_resid_other_px"] > good["median_resid_other_px"] + 0.5


def test_instant_outside_the_other_clip_is_reported(pair) -> None:
    view_a, view_b, _ = pair
    short_b = st.CamView(
        name="short",
        dlt=view_b.dlt,
        markers=view_b.markers,
        cut_start=view_b.cut_start,
        sync_cut=SYNC_CUT,
        pts_cut=view_b.pts_cut[:20],
    )
    rows = st.reconstruct_ball(view_a, short_b)
    reasons = {r["reason"] for r in rows if not r["valid"]}
    assert "instant outside the other camera's clip" in reasons
    assert all(np.isnan(r["z_m"]) for r in rows if not r["valid"])


def test_missing_ball_pixel_is_not_filled_in(pair) -> None:
    view_a, view_b, frames = pair
    gap = int(frames[len(frames) // 2])
    markers_b = {f: uv for f, uv in view_b.markers.items() if f != gap}
    b = st.CamView("b", view_b.dlt, markers_b, view_b.cut_start, SYNC_CUT, view_b.pts_cut)
    row = next(r for r in st.reconstruct_ball(view_a, b) if r["frame"] == gap)
    assert not row["valid"]
    assert row["reason"] == "no ball pixel or DLT in the other camera"
    assert np.isnan(row["x_m"])


def test_nearly_parallel_rays_are_not_trusted(pair) -> None:
    view_a, _, frames = pair
    L_near = _camera([-60.5, -45.0, 25.0], [-30.0, -10.0, 0.0])  # almost the same viewpoint
    pts = view_a.pts_cut
    near_b = _view("near", L_near, _markers_from_truth(L_near, frames, pts), pts, cut_start=1)
    rows = st.reconstruct_ball(view_a, near_b)
    assert any(r["reason"] == "rays nearly parallel: depth not trusted" for r in rows)


def test_large_reprojection_error_is_rejected(pair) -> None:
    view_a, view_b, frames = pair
    f = int(frames[5])
    markers_b = dict(view_b.markers)
    markers_b[f] = markers_b[f] + np.array([25.0, -18.0])  # a bad hand-placed marker
    b = st.CamView("b", view_b.dlt, markers_b, view_b.cut_start, SYNC_CUT, view_b.pts_cut)
    row = next(r for r in st.reconstruct_ball(view_a, b) if r["frame"] == f)
    assert not row["valid"]
    assert row["reason"] == "reprojection error above the limit"


def test_load_markers_requires_the_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad_markers.csv"
    bad.write_text("frame,x,y\n0,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frame,p0_x,p0_y"):
        st.load_markers(bad)


def test_load_markers_skips_empty_rows(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    path.write_text("frame,p0_x,p0_y\n0,10,20\n1,,\n2,30,40\n", encoding="utf-8")
    markers = st.load_markers(path)
    assert sorted(markers) == [0, 2]
    assert np.allclose(markers[2], [30.0, 40.0])


def test_nominal_clock_is_frame_over_fps(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 90))
    for _ in range(12):
        writer.write(np.zeros((90, 160, 3), np.uint8))
    writer.release()
    clock = st.nominal_clock(video)
    assert clock.size == 12
    assert np.allclose(np.diff(clock), 1.0 / 30.0)


def test_monocular_comparison_reads_the_ground_track(tmp_path: Path, pair) -> None:
    view_a, view_b, _ = pair
    rows = st.reconstruct_ball(view_a, view_b)
    mono = tmp_path / "ball_3d_on_field.csv"
    with mono.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "x_m", "y_m", "z_m"])
        for r in rows:
            writer.writerow([r["frame"], r["x_m"] + 1.0, r["y_m"], 0.11])
    diff = st.monocular_comparison(rows, mono)
    assert diff is not None and diff["frames_compared"] == sum(r["valid"] for r in rows)
    assert diff["median_horizontal_diff_m"] == pytest.approx(1.0, abs=1e-6)


def test_run_binocular_end_to_end(tmp_path: Path, pair) -> None:
    """Two 3D folders written the way the pipeline writes them, reconstructed by run_binocular."""
    view_a, view_b, frames = pair
    stems = {"a": "camA_sync_15_frames_2_to_79", "b": "camB_sync_14_frames_1_to_78"}
    work = tmp_path / "work"
    runs = {}
    for key, view in (("a", view_a), ("b", view_b)):
        stem = stems[key]
        cut = tmp_path / f"{stem}.mp4"
        writer = cv2.VideoWriter(str(cut), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for _ in range(78):
            writer.write(np.zeros((H, W, 3), np.uint8))
        writer.release()
        det = work / f"freekiki_predict_{stem}"
        det.mkdir(parents=True)
        L = L_A if key == "a" else L_B
        (det / f"{stem}.dlt3d").write_text(
            "frame," + ",".join(f"L{i}" for i in range(1, 12)) + "\n"
            + "".join(f"{f}," + ",".join(f"{v:.12g}" for v in L) + "\n" for f in range(78)),
            encoding="utf-8",
        )  # fmt: skip
        with (tmp_path / f"{stem}_markers.csv").open("w", newline="", encoding="utf-8") as fh:
            writer_csv = csv.writer(fh)
            writer_csv.writerow(["frame", "p0_x", "p0_y"])
            for f, uv in sorted(view.markers.items()):
                writer_csv.writerow([f, f"{uv[0]:.3f}", f"{uv[1]:.3f}"])
        run = tmp_path / f"{stem}_3d_20260101_000000"
        run.mkdir()
        (run / "README.txt").write_text(
            f"FreeKiki 3D output (vailá)\nvideo: {cut}\nwork from: {tmp_path}\n", encoding="utf-8"
        )
        runs[key] = run
    summary = st.run_binocular(runs["a"], runs["b"], out_dir=tmp_path / "out")
    assert summary["sync"] == {
        "reference_sync_frame": SYNC_CUT,
        "reference_cut_start": 2,
        "other_sync_frame": SYNC_CUT,
        "other_cut_start": 1,
    }
    assert summary["counts"]["valid"] == len(frames)
    written = list((tmp_path / "out").glob("ball_binocular.csv"))
    assert written and json.loads((tmp_path / "out" / "summary.json").read_text())["counts"]


def test_cli_accepts_the_rec3d_binocular_arguments() -> None:
    from vaila import freekiki

    args = freekiki.build_parser().parse_args(
        ["rec3d-binocular", "--run1", "a", "--run2", "b", "--sync2", "14", "--max-resid", "2.5"]
    )
    assert args.command == "rec3d-binocular"
    assert args.sync2 == 14 and args.max_resid == 2.5 and args.min_angle == 5.0
