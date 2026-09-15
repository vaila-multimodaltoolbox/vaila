"""Fixed-scene stabilization regression tests (no external downloads).

Version: 0.4.1
Update Date: 15 September 2026
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vaila import video_stabilizer as vs

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "vaila/models/planar_targets/tatame_1x1m.toml"


def test_marker_ranges_and_noncontiguous_ids():
    assert vs.parse_id_spec("0-2,8,12-14", [0, 1, 2, 8, 12, 13, 14]) == [0, 1, 2, 8, 12, 13, 14]
    assert vs.parse_id_spec("all", [8, 2]) == [2, 8]
    for spec in ("-1", "3-1", "9", "1,a"):
        with pytest.raises(ValueError):
            vs.parse_id_spec(spec, [0, 1, 2, 3])


def test_csv_aligns_sparse_frame_numbers_and_keeps_missing_ids(tmp_path):
    path = tmp_path / "markers.csv"
    path.write_text("frame,p2_x,p2_y,p8_x,p8_y\n3,2,4,,\n0,inf,2,4,5\n")
    table = vs.load_marker_csv(path, 6)
    assert table.ids == [2, 8]
    assert table.xy.shape == (6, 2, 2)
    assert np.isnan(table.xy[0, 0]).all()
    assert np.isnan(table.xy[1]).all()
    np.testing.assert_equal(table.xy[3, 0], [2, 4])
    for contents in (
        "frame,p0_x,p0_y\n0,1,2\n0,3,4",
        "frame,p0_x,p0_y\n0.5,1,2",
        "frame,p0_x,p0_y\n6,1,2",
    ):
        path.write_text(contents)
        with pytest.raises(ValueError):
            vs.load_marker_csv(path, 6)


def test_reference_is_a_real_central_frame_with_priority_anchors():
    base = np.array([[0, 0], [10, 0], [0, 10], [20, 20]], dtype=float)
    xy = np.array([base + [i * 5, 0] for i in range(5)])
    xy[0, 3] = np.nan
    table = vs.MarkerTable(np.arange(5), [0, 2, 8, 10], xy)
    ref = vs.choose_reference_frame(table, table.ids, [8, 10])
    assert ref in (2, 3)
    np.testing.assert_equal(table.xy[ref, :3], base[:3] + [ref * 5, 0])
    assert vs.choose_reference_frame(table, table.ids, [8, 10], "frame:4") == 4
    table.xy[0] = np.nan
    with pytest.raises(ValueError, match="Reference frame"):
        vs.choose_reference_frame(table, table.ids, [], "first")


def test_exact_similarity_recovery_and_shape_invariant():
    src = np.random.default_rng(2).normal(size=(20, 2)) * 200 + [10000, 20000]
    matrix = vs.params_to_matrix([50, -22, 0.23, np.log(1.04)])
    dst = vs.project_points(matrix, src)
    fitted = vs.fit_weighted_transform(src, dst)
    np.testing.assert_allclose(fitted, matrix, atol=1e-9)
    np.testing.assert_allclose(fitted[:2, :2].T @ fitted[:2, :2], np.eye(2) * 1.04**2, atol=1e-10)
    np.testing.assert_array_equal(fitted[2], [0, 0, 1])
    assert vs.fit_weighted_transform(np.ones((3, 2)), np.ones((3, 2))) is None


def test_weighted_fit_favors_reliable_observations_and_density_balances():
    src = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=float)
    dst = src + [10, 5]
    dst[-1] += [15, 0]
    weighted = vs.fit_weighted_transform(src, dst, [1, 1, 1, 0.001])
    plain = vs.fit_weighted_transform(src, dst)
    truth = src[:3] + [10, 5]
    assert (
        np.linalg.norm(vs.project_points(weighted, src[:3]) - truth)
        < np.linalg.norm(vs.project_points(plain, src[:3]) - truth) * 0.02
    )
    density = vs.spatial_weights(np.array([[0, 0], [1, 1], [2, 2], [100, 100]]), 100, 100)
    assert density[-1] > 2 * density[0]


def test_robust_fit_downweights_gross_digitization_error():
    src = np.random.default_rng(6).uniform(0, 500, (15, 2))
    truth = vs.params_to_matrix([20, -10, 0.01, 0])
    dst = vs.project_points(truth, src)
    dst[-1] += [400, -300]
    fitted = vs.estimate_transform(src, dst, np.ones(len(src)))
    errors = np.linalg.norm(
        vs.project_points(fitted, src[:-1]) - vs.project_points(truth, src[:-1]), axis=1
    )
    assert errors.max() < 2


@pytest.mark.parametrize("smooth", ["none", "savgol", "lowpass"])
def test_unwrap_interpolation_and_endpoint_propagation(smooth):
    matrices = [None] * 15
    matrices[2] = vs.params_to_matrix([2, 3, np.deg2rad(179), 0])
    matrices[12] = vs.params_to_matrix([12, 3, np.deg2rad(-179), np.log(1.1)])
    raw, final, methods = vs.regularize_transforms(matrices, 60, smooth)
    assert np.degrees(raw[7, 2]) == pytest.approx(180)
    np.testing.assert_allclose(raw[0], raw[2])
    assert methods[0] == "propagated" and methods[7] == "interpolated"
    assert np.isfinite(final).all()
    with pytest.raises(ValueError, match="No trustworthy"):
        vs.regularize_transforms([None, None], 60)


@pytest.mark.parametrize("model", ["similarity", "affine"])
def test_parameter_roundtrip(model):
    params = [10, 20, 0.7, np.log(1.2)] + ([np.log(0.8), 0.2] if model == "affine" else [])
    matrix = vs.params_to_matrix(params, model)
    np.testing.assert_allclose(vs.matrix_to_params(matrix, model), params, atol=1e-10)


def test_union_contains_every_transformed_corner_and_crop_is_inside():
    matrices = np.array([vs.params_to_matrix([i * 4, 0, i * 0.01, 0]) for i in range(4)])
    translation, width, height = vs.compute_canvas(matrices, 100, 80)
    corners = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=float)
    for matrix in matrices:
        points = vs.project_points(translation @ matrix, corners)
        assert (points >= -1e-8).all()
        assert (points <= [width + 1e-8, height + 1e-8]).all()
    assert width % 2 == height % 2 == 0
    translation, width, height = vs.compute_canvas(matrices, 100, 80, "crop")
    crop = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    for matrix in matrices:
        back = vs.project_points(np.linalg.solve(translation @ matrix, np.eye(3)), crop)
        assert (back >= -1e-4).all() and (back <= [100.0001, 80.0001]).all()


def test_exact_marker_output_uses_canvas_transform():
    table = vs.MarkerTable(
        np.arange(2), [8, 10], np.array([[[1.0, 2], [3, 4]], [[np.nan, np.nan], [5, 6]]])
    )
    matrices = np.array([vs.params_to_matrix([100, 200, 0, 0])] * 2)
    _, wide = vs.transform_marker_table(table, matrices)
    assert list(wide.columns) == ["frame", "p8_x", "p8_y", "p10_x", "p10_y"]
    assert wide.p10_x.tolist() == [103, 105]
    assert np.isnan(wide.p8_x.iloc[1])


def test_floor_rejects_four_points_with_three_collinear():
    geometry = vs.load_target_geometry(GEOMETRY)
    ids = [2, 3, 4, 7]
    pixels = np.array([geometry.world_xy(i) * 100 + [30, 40] for i in ids])
    table = vs.MarkerTable(np.array([0]), ids, pixels[None])
    matrices, diagnostics = vs.solve_floor(table, geometry, ids)
    assert np.isnan(matrices).all()
    assert diagnostics.method.iloc[0] == "degenerate"


def _sample(tmp_path, audio=False):
    source = tmp_path / "input with spaces.mp4"
    n, width, height = 20, 128, 96
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    assert writer.isOpened()
    data = {"frame": list(range(n))}
    geometry = vs.load_target_geometry(GEOMETRY)
    points = np.array(
        [geometry.world_xy(i) * 45 + [35, 30] for i in range(8)] + [[10, 8], [110, 8], [65, 12]]
    )
    observations = np.array([points + [np.sin(i / 4) * 8, i / 3] for i in range(n)])
    for frame_id in range(n):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for pt in observations[frame_id]:
            cv2.circle(frame, tuple(np.round(pt).astype(int)), 2, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    for col in range(11):
        data[f"p{col}_x"] = observations[:, col, 0]
        data[f"p{col}_y"] = observations[:, col, 1]
    csv = tmp_path / "markers with spaces.csv"
    pd.DataFrame(data).to_csv(csv, index=False)
    if audio:
        silent = tmp_path / "silent.mp4"
        source.rename(silent)
        subprocess.run(
            [
                vs.get_ffmpeg_path(),
                "-v",
                "error",
                "-i",
                str(silent),
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(source),
            ],
            check=True,
        )
    return source, csv


@pytest.mark.parametrize("mode", ["visual", "hybrid", "floor-lock"])
def test_pipeline_outputs_timeline_audio_and_metric_separation(tmp_path, mode):
    source, csv = _sample(tmp_path, audio=True)
    options = {"geometry_config": GEOMETRY, "metric_markers": "0-7"} if mode != "visual" else {}
    output = vs.run_video_stabilizer(
        source,
        csv,
        tmp_path / "out",
        mode=mode,
        anchor_markers="8-10",
        smooth="none",
        debug_overlay=mode == "hybrid",
        **options,
    )
    cap = cv2.VideoCapture(str(output["video"]))
    assert cap.isOpened()
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 20
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(20, abs=0.001)
    count = 0
    shapes = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shapes.add(frame.shape)
        count += 1
    cap.release()
    assert count == 20 and len(shapes) == 1
    info = vs.get_precise_video_metadata(output["video"])
    assert any(s["codec_type"] == "audio" for s in info["_raw_json"]["streams"])
    transforms = pd.read_csv(output["transforms"])
    matrix_cols = [f"matrix_{i}{j}" for i in range(3) for j in range(3)]
    assert np.isfinite(transforms[matrix_cols]).all().all()
    if mode != "floor-lock":
        np.testing.assert_allclose(
            transforms[["matrix_20", "matrix_21", "matrix_22"]], np.tile([0, 0, 1], (20, 1))
        )
    metrics = pd.read_csv(output["diagnostics"]).set_index("marker")
    assert metrics.loc["anchors", "rms_after_px"] < metrics.loc["anchors", "rms_before_px"] * 0.05
    if mode != "visual":
        floor = np.load(output["floor_homographies"])
        assert floor["metric_ids"].tolist() == list(range(8))
        assert pd.read_csv(output["floor_diagnostics"]).n_floor_correspondences.max() == 8
    if mode == "hybrid":
        assert output["overlay"].is_file()
    with pytest.raises(FileExistsError):
        vs.run_video_stabilizer(source, csv, tmp_path / "out")


def test_missing_frames_use_fallback_without_truncating_video(tmp_path):
    source, csv = _sample(tmp_path)
    df = pd.read_csv(csv)
    df = df[~df.frame.isin([0, 5, 6, 19])]
    df.to_csv(csv, index=False)
    output = vs.run_video_stabilizer(source, csv, tmp_path / "out", smooth="none")
    transforms = pd.read_csv(output["transforms"])
    assert len(transforms) == 20
    assert transforms.method.iloc[0] == transforms.method.iloc[-1] == "propagated"
    assert transforms.method.iloc[5] == "interpolated"


def test_cli_headless_and_quoted_command(tmp_path):
    source, csv = _sample(tmp_path)
    argv = [
        "--video",
        str(source),
        "--markers",
        str(csv),
        "--output-dir",
        str(tmp_path / "CLI output"),
        "--no-audio",
    ]
    command = vs.format_cli_command(argv)
    assert shlex.split(command)[3:] == argv
    env = dict(os.environ)
    env.pop("DISPLAY", None)
    result = subprocess.run(
        shlex.split(command), env=env, capture_output=True, text=True, timeout=40
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "CLI output/stabilization_report.html").is_file()
    result = subprocess.run(
        [sys.executable, "-m", "vaila.video_stabilizer", "--video", str(source)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_no_display_dependency_in_import():
    script = "import sys; import vaila.video_stabilizer; assert 'tkinter' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cancel_before_render_has_no_output(tmp_path):
    import threading

    source, csv = _sample(tmp_path)
    event = threading.Event()
    event.set()
    with pytest.raises(InterruptedError):
        vs.run_video_stabilizer(source, csv, tmp_path / "out", cancel_event=event)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "model,estimator,canvas",
    [
        ("affine", "robust-lsq", "crop"),
        ("similarity", "ransac", "original"),
        ("affine", "ransac", "union"),
    ],
)
def test_explicit_alternatives(tmp_path, model, estimator, canvas):
    source, csv = _sample(tmp_path)
    out = vs.run_video_stabilizer(
        source, csv, tmp_path / "out", model=model, estimator=estimator, canvas=canvas
    )
    assert out["video"].is_file()
    data = pd.read_csv(out["transforms"])
    assert np.isfinite(data.matrix_00).all()


def test_real_tatame_visual_and_hybrid_regression():
    """Validate existing real CLI artifacts; no external data or repeated rendering.

    Inspection found 77.66→9.48 px total and 90.24→10.11 px anchors.
    A 75% aggregate reduction gate leaves margin for annotation/parallax noise.
    """
    base = ROOT / "tests/interp_geometry"
    result_dir = base / "stabilizer_results"
    if not (result_dir / "hybrid/stabilization_report.html").exists():
        pytest.skip(
            "Generate the real visual/hybrid example artifacts to run this acceptance test."
        )
    original = vs.get_precise_video_metadata(base / "tatame.mp4")
    table = vs.load_marker_csv(base / "tatame_markers.csv", original["nb_frames"])
    output_transforms = []
    for mode in ("visual", "hybrid"):
        directory = result_dir / mode
        video = directory / "tatame_stabilized.mp4"
        info = vs.get_precise_video_metadata(video)
        assert info["nb_frames"] == original["nb_frames"] == 331
        assert info["fps"] == pytest.approx(original["fps"], abs=0.001)
        assert abs(info["duration"] - original["duration"]) < 0.08
        assert any(s["codec_type"] == "audio" for s in info["_raw_json"]["streams"])
        capture = cv2.VideoCapture(str(video))
        shapes = set()
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            shapes.add(frame.shape)
            count += 1
        capture.release()
        assert len(shapes) == 1 and count == 331
        data = pd.read_csv(directory / "stabilization_transforms.csv")
        matrices = (
            data[[f"matrix_{i}{j}" for i in range(3) for j in range(3)]]
            .to_numpy()
            .reshape(-1, 3, 3)
        )
        assert np.isfinite(matrices).all()
        np.testing.assert_allclose(matrices[:, 2, :], np.tile([0, 0, 1], (331, 1)))
        np.testing.assert_allclose(matrices[:, 0, 0], matrices[:, 1, 1])
        np.testing.assert_allclose(matrices[:, 0, 1], -matrices[:, 1, 0])
        _, expected = vs.transform_marker_table(table, matrices)
        actual = pd.read_csv(directory / "stabilized_markers.csv")
        np.testing.assert_allclose(
            expected.to_numpy(), actual.to_numpy(), atol=1e-8, equal_nan=True
        )
        metrics = pd.read_csv(directory / "stabilization_diagnostics.csv").set_index("marker")
        for group in ("all", "anchors", "p8", "p9", "p10"):
            assert metrics.loc[group, "rms_after_px"] < 0.25 * metrics.loc[group, "rms_before_px"]
        # Scale deviation must reflect observed source expansion/contraction.
        reference = int(data.reference_frame.iloc[0])
        anchor_xy = table.xy[:, [table.ids.index(i) for i in (8, 9, 10)]]
        source_distances = np.linalg.norm(anchor_xy[:, 0] - anchor_xy[:, 2], axis=1)
        distance_ratio = source_distances[reference] / source_distances
        assert np.max(abs(data.scale.to_numpy() - distance_ratio)) < 0.04
        summary = json.loads((directory / "stabilization_summary.json").read_text())
        assert summary["reference_frame"] == reference
        if mode == "hybrid":
            floor = np.load(directory / "floor_homographies.npz")
            assert floor["metric_ids"].tolist() == list(range(8))
            floor_df = pd.read_csv(directory / "floor_diagnostics.csv")
            assert floor_df.n_floor_correspondences.max() <= 8
            assert len(floor_df) == 331
            assert np.isnan(floor["H"][floor_df.method.ne("ransac")]).all()
            assert {245, 317}.issubset(set(floor_df.frame[floor_df.method.eq("degenerate")]))
        output_transforms.append(matrices)
    # Metric calibration has no effect on the visual warp in hybrid mode.
    np.testing.assert_allclose(*output_transforms, atol=1e-10)


def test_gui_worker_guards_and_main_thread_events(monkeypatch):
    """Exercise lifecycle without Tk; real widgets are covered by the GUI smoke."""
    import queue
    import threading
    import time

    class Widget:
        def __init__(self, value=""):
            self.value = value

        def get(self):
            return self.value

        def set(self, value):
            assert threading.current_thread() is threading.main_thread()
            self.value = value

        def configure(self, **kw):
            assert threading.current_thread() is threading.main_thread()

        def insert(self, *args):
            pass

        def see(self, *args):
            pass

        def after(self, *args):
            pass

    app = vs.StabilizerGUI.__new__(vs.StabilizerGUI)
    app.vars = {"video": Widget("input.mp4"), "markers": Widget("markers.csv")}
    app.events = queue.Queue()
    app.cancel = threading.Event()
    app.worker = None
    app.closing = False
    app.root = app.run_button = app.cancel_button = app.report_button = app.text = Widget()
    app.status = Widget()
    calls = []

    def run(args, progress_callback, cancel_event):
        calls.append(args)
        progress_callback("working")
        while not cancel_event.is_set():
            time.sleep(0.005)
        raise InterruptedError("cancelled")

    monkeypatch.setattr(vs, "_run_args", run)
    app.start()
    first_worker = app.worker
    app.start()
    assert app.worker is first_worker
    app.request_cancel()
    app.worker.join(timeout=2)
    app.drain()
    assert len(calls) == 1 and app.status.value == "cancelled"


@pytest.mark.skipif(
    os.environ.get("VAILA_GUI_SMOKE") != "1", reason="Opt-in real desktop smoke: VAILA_GUI_SMOKE=1"
)
def test_gui_display_smoke(tmp_path, monkeypatch):
    import importlib.util
    import time

    from PIL import ImageGrab

    spec = importlib.util.spec_from_file_location("vaila_desktop_for_stabilizer", ROOT / "vaila.py")
    desktop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(desktop)
    monkeypatch.setattr(desktop.Vaila, "_start_update_check", lambda *a, **kw: None)
    opened = []
    monkeypatch.setattr(vs.webbrowser, "open_new_tab", lambda uri: opened.append(uri))
    pictures = ROOT / "tests/interp_geometry/stabilizer_results"
    pictures.mkdir(parents=True, exist_ok=True)

    def screenshot(window, name):
        window.update()
        window.lift()
        window.update()
        time.sleep(0.2)
        x, y = window.winfo_rootx(), window.winfo_rooty()
        ImageGrab.grab(bbox=(x, y, x + window.winfo_width(), y + window.winfo_height())).save(
            pictures / name
        )

    root = desktop.Vaila()
    root.update()

    def children(widget):
        for item in widget.winfo_children():
            yield item
            yield from children(item)

    buttons = [
        w
        for w in children(root)
        if w.winfo_class() == "Button" and w.cget("text") == "Video Stabilizer"
    ]
    assert len(buttons) == 1
    assert int(buttons[0].grid_info()["column"]) == 1
    screenshot(root, "main_button.png")
    buttons[0].invoke()
    app = root._video_stabilizer_app
    root.update()
    assert str(app.root.transient()) == str(root)
    buttons[0].invoke()
    assert root._video_stabilizer_app is app
    app.open_help()
    assert opened[-1].endswith("/help/video_stabilizer.html")
    screenshot(app.root, "stabilizer_integrated.png")
    source, csv = _sample(tmp_path)
    app.vars["video"].set(str(source))
    app.vars["markers"].set(str(csv))
    app.vars["output-dir"].set(str(tmp_path / "gui_result"))
    app.start()
    worker = app.worker
    app.start()
    assert worker is app.worker
    ticks = 0
    deadline = time.monotonic() + 20
    while worker.is_alive() and time.monotonic() < deadline:
        root.update()
        ticks += 1
        time.sleep(0.02)
    assert not worker.is_alive()
    app.drain()
    assert ticks > 1
    assert app.outputs["video"].exists()
    app.open_report()
    assert opened[-1].endswith("stabilization_report.html")
    app.root.geometry("720x640")
    root.update()
    screenshot(app.root, "stabilizer_result.png")
    app.close()
    root.update()
    assert root.winfo_exists()
    root.destroy()
    standalone = vs.StabilizerGUI()
    standalone.root.update()
    assert standalone.owns_root
    screenshot(standalone.root, "stabilizer_standalone.png")
    standalone.close()
