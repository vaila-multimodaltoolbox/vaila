"""REF3D builder: control points from the measures of a box, pole or triad.

Author: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
Update Date: 16 November 2026
Version: 0.4.19
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vaila import ref3d_builder as rb
from vaila.dlt3d import normalize_ref3d_to_format1
from vaila.monocular_dlt_align import dlt_project

# The walkway calibration of 13 November 2026: a 1 m pole at the 4 corners of a
# rectangle, as typed by hand in ref3d.txt (X 0.55, Y 5).
WALKWAY_REF3D_TXT = [
    (0, 0, 0),
    (0, 0, 1),
    (0.55, 0, 0),
    (0.55, 0, 1),
    (0, 5, 0),
    (0, 5, 1),
    (0.55, 5, 0),
    (0.55, 5, 1),
]


def test_pole_on_a_rectangle_gives_the_hand_typed_walkway_ref3d() -> None:
    points = rb.pole_points((0, 1), rb.parse_positions("rect 0.55 x 5"))
    assert [p.xyz for p in points] == WALKWAY_REF3D_TXT
    assert [p.index for p in points] == list(range(8))
    assert points[3].label == "pos2 · mark 1 m"


def test_triad_order_is_origin_then_x_y_z_for_each_position() -> None:
    points = rb.triad_points((0.5,), (0.5, 1.0), (1.0,), ((0, 0), (2, 0)))
    assert [p.xyz for p in points[:5]] == [
        (0, 0, 0),
        (0.5, 0, 0),
        (0, 0.5, 0),
        (0, 1.0, 0),
        (0, 0, 1.0),
    ]
    assert points[5].xyz == (2, 0, 0) and points[5].label == "pos2 · origin"
    assert len(points) == 10


def test_box_corners_and_edge_marks() -> None:
    corners = rb.box_points(1, 2, 3)
    assert len(corners) == 8
    assert corners[0].xyz == (0, 0, 0) and corners[-1].xyz == (1, 2, 3)
    marked = rb.box_points(1, 2, 3, div_y=(1.0,))
    # a mark on Y adds one point on each of the 4 edges parallel to Y
    assert len(marked) == 12
    assert sum(p.y == 1.0 for p in marked) == 4


def test_bad_measures_say_what_is_wrong() -> None:
    with pytest.raises(rb.Ref3dError, match="outside"):
        rb.box_points(1, 1, 1, div_x=(2.0,))
    with pytest.raises(rb.Ref3dError, match="numbers"):
        rb.parse_lengths("0.5, um")
    with pytest.raises(rb.Ref3dError, match="two numbers"):
        rb.parse_positions("0,0; 1")
    with pytest.raises(rb.Ref3dError, match="marks"):
        rb.pole_points((), ((0, 0),))


def test_parsing_of_sizes_and_positions() -> None:
    assert rb.parse_lengths("0.25, 0.5; 1") == (0.25, 0.5, 1.0)
    assert rb.parse_size("0.55 x 5 x 1") == (0.55, 5.0, 1.0)
    assert rb.parse_positions("0,0; 0.55,0") == ((0.0, 0.0), (0.55, 0.0))
    assert rb.parse_positions("") == ((0.0, 0.0),)
    assert rb.parse_positions("1.38x5")[-1] == (1.38, 5.0)


def test_flat_templates_are_flagged() -> None:
    assert rb.points_are_coplanar(rb.pole_points((0,), rb.parse_positions("1 x 1")))
    assert not rb.points_are_coplanar(rb.pole_points((0, 1), rb.parse_positions("1 x 1")))


def test_written_ref3d_is_read_by_dlt3d_and_the_template_round_trips(tmp_path: Path) -> None:
    template = rb.Ref3dTemplate("pole", marks=(0, 1), positions=rb.parse_positions("0.55 x 5"))
    path = rb.write_ref3d(template.points(), tmp_path / "calib.ref3d")

    df = normalize_ref3d_to_format1(str(path))
    assert df is not None
    assert df.iloc[0, 1:].to_numpy(dtype=float).reshape(-1, 3).tolist() == [
        list(map(float, p)) for p in WALKWAY_REF3D_TXT
    ]
    assert [p.xyz for p in rb.read_ref3d_points(path)] == WALKWAY_REF3D_TXT

    rb.write_template(template, tmp_path / "calib.template.toml")
    again = rb.read_template(tmp_path / "calib.template.toml")
    assert again.points() == template.points()


def _camera(focal: float = 1350.0) -> np.ndarray:
    """A real pinhole camera 9 m behind the walkway, 1.2 m high, as 11 DLT coefficients."""
    K = np.array([[focal, 0, 960], [0, focal, 540], [0, 0, 1.0]])
    look = np.array([0.0, 1.0, -0.12])
    look /= np.linalg.norm(look)
    right = np.array([1.0, 0, 0])
    down = np.cross(look, right)
    R = np.vstack([right, down, look])
    C = np.array([0.7, -9.3, 1.2])
    P = K @ np.hstack([R, (-R @ C)[:, None]])
    P /= P[2, 3]
    return np.r_[P[0], P[1], P[2, :3]]


def _clicks(points, L, noise=0.0, seed=0) -> dict[int, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    uv = dlt_project(L, np.array([p.xyz for p in points])) + rng.normal(0, noise, (len(points), 2))
    return {p.index: (float(u), float(v)) for p, (u, v) in zip(points, uv, strict=True)}


def test_true_measures_give_a_real_camera() -> None:
    L = _camera()
    true_points = rb.pole_points((0, 1), rb.parse_positions("1.38 x 5"))

    check = rb.fit_and_check(true_points, _clicks(true_points, L, noise=0.5))

    assert isinstance(check, rb.DltCheck)
    assert check.ok, check.problems
    assert check.fx == pytest.approx(1350, rel=0.05)
    assert check.fy == pytest.approx(check.fx, rel=0.05)
    assert check.max_px < 3


def test_a_wrong_width_still_fits_the_clicks_but_is_not_a_real_camera() -> None:
    """The case of 13 November 2026: the floor width typed 0.55 m, really 1.38 m."""
    L = _camera()
    clicks = _clicks(rb.pole_points((0, 1), rb.parse_positions("1.38 x 5")), L)
    typed = rb.pole_points((0, 1), rb.parse_positions("0.55 x 5"))

    check = rb.fit_and_check(typed, clicks)

    assert isinstance(check, rb.DltCheck)
    assert check.max_px < 1  # the 11 parameters absorb it: the error alone hides it
    assert not check.ok
    assert "non-square pixels" in check.problems[0]


def test_projection_covers_points_not_clicked_yet() -> None:
    L = _camera()
    points = rb.pole_points((0, 0.5, 1), rb.parse_positions("1.38 x 5"))
    clicks = {i: uv for i, uv in _clicks(points, L).items() if i < 9}

    check = rb.fit_and_check(points, clicks)

    assert isinstance(check, rb.DltCheck)
    assert set(check.projected) == {p.index for p in points}
    expected = dlt_project(L, np.array([points[11].xyz]))[0]
    assert np.allclose(check.projected[11], expected, atol=1e-3)


def test_too_few_or_flat_points_explain_why() -> None:
    L = _camera()
    points = rb.pole_points((0, 1), rb.parse_positions("1.38 x 5"))
    clicks = _clicks(points, L)
    assert "needs 6" in str(rb.fit_and_check(points, {i: clicks[i] for i in range(5)}))
    floor = {i: clicks[i] for i in (0, 2, 4, 6)}
    floor_points = rb.box_points(1.38, 5, 1)  # any template; only clicks on z = 0 are used
    assert "6 points" in str(rb.fit_and_check(floor_points, floor))


def test_cli_builds_a_ref3d_and_checks_a_pixel_file(tmp_path: Path) -> None:
    out = tmp_path / "pole.ref3d"
    assert (
        rb.main(["--template", "pole", "--marks", "0,1", "--positions", "1.38 x 5", "-o", str(out)])
        == 0
    )
    assert (tmp_path / "pole.template.toml").is_file()
    # refuses to overwrite
    assert rb.main(["--template", "pole", "--marks", "0,1", "-o", str(out)]) == 1

    clicks = _clicks(rb.read_ref3d_points(out), _camera())
    pixels = tmp_path / "pole_markers_1_line.csv"
    header = "frame," + ",".join(f"p{i}_x,p{i}_y" for i in range(8))
    row = "0," + ",".join(f"{u:.2f},{v:.2f}" for u, v in (clicks[i] for i in range(8)))
    pixels.write_text(header + "\n" + row + "\n")

    assert rb.main(["--ref3d", str(out), "--pixels", str(pixels)]) == 0
    assert (tmp_path / "pole_markers_1_line.dlt3d").is_file()


def test_sketch_is_right_handed_x_cross_y_is_z_up() -> None:
    """The panel sketch draws X x Y = Z, with Z up (screen y grows downward)."""
    origin, ex, ey, ez = rb.sketch_projection(np.eye(4, 3)[[3, 0, 1, 2]])
    assert origin.tolist() == [0.0, 0.0]
    assert ez[1] < 0 and ez[0] == 0  # Z up
    assert ex[0] > 0 and ex[1] == 0  # X to the right
    assert ey[1] < 0  # Y goes away from the viewer (up on the screen)
    # Seen from the front, X x Y must point up (= Z): with screen y flipped to
    # "up", the turn from X to Y is counter-clockwise.
    cross = ex[0] * (-ey[1]) - (-ex[1]) * ey[0]
    assert cross > 0


# --- the illustrated guide (vaila/help/ref3d_guide*.html) ----------------------------

HELP = Path(__file__).resolve().parents[1] / "vaila" / "help"


def _svg_labels(svg: str) -> list[tuple[str, float, float]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg)  # valid XML, or this raises
    ns = "{http://www.w3.org/2000/svg}"
    return [
        (t.text or "", float(t.get("x", 0)), float(t.get("y", 0))) for t in root.iter(f"{ns}text")
    ]


def test_template_svg_numbers_every_point_in_click_order() -> None:
    template = rb.Ref3dTemplate("pole", marks=(0.0, 1.0), positions=rb.parse_positions("1.38 x 5"))
    labels = [t for t, _, _ in _svg_labels(rb.template_svg(template)) if t[:1] == "p"]
    names = [t for t in labels if t[1:].isdigit()]
    assert names == [f"p{p.index}" for p in template.points()]
    assert [t for t in labels if t.startswith("pos")] == ["pos1", "pos2", "pos3", "pos4"]


def test_template_svg_draws_z_up_and_x_to_the_right() -> None:
    template = rb.Ref3dTemplate("box", size=(1.0, 1.0, 1.0))
    where = {t: (x, y) for t, x, y in _svg_labels(rb.template_svg(template, coords=False))}
    # p0 = (0,0,0), p1 = (1,0,0), p4 = (0,0,1): X goes right, Z goes up on the screen
    assert where["p1"][0] > where["p0"][0]
    assert where["p4"][1] < where["p0"][1]


def test_guide_figures_are_up_to_date_with_the_code() -> None:
    """bin/make_ref3d_guide_svgs.py --check: the SVGs in vaila/help/images/ref3d are
    what the current ref3d_builder draws."""
    import subprocess
    import sys

    root = HELP.parents[1]
    done = subprocess.run(
        [sys.executable, str(root / "bin" / "make_ref3d_guide_svgs.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_guide_pages_point_to_existing_figures_in_both_languages() -> None:
    import re

    found = {}
    for name in ("ref3d_guide.html", "ref3d_guide_pt.html", "ref3d_guide.md", "ref3d_guide_pt.md"):
        text = (HELP / name).read_text(encoding="utf-8")
        srcs = re.findall(r'src="(images/[^"]+)"', text) + re.findall(r"\]\((images/[^)]+)\)", text)
        assert srcs, name
        for src in srcs:
            assert (HELP / src).is_file(), f"{name}: {src}"
        found[name] = sorted(s.replace("_pt.svg", ".svg").replace("_en.svg", ".svg") for s in srcs)
    assert found["ref3d_guide.html"] == found["ref3d_guide_pt.html"]
    assert found["ref3d_guide.md"] == found["ref3d_guide_pt.md"]
    assert set(found["ref3d_guide.html"]) == set(found["ref3d_guide.md"])


# --- template TOML filled in by hand ------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        rb.Ref3dTemplate("pole", marks=(0.0, 1.0), positions=rb.parse_positions("1.38 x 5")),
        rb.Ref3dTemplate("pole", marks=(0.0, 0.5, 1.0), positions=((0.0, 0.0), (2.0, 1.0))),
        rb.Ref3dTemplate("box", size=(1.38, 5.0, 1.0), div_y=(2.5,)),
        rb.Ref3dTemplate("triad", arms_x=(0.5,), arms_y=(0.5, 1.0), arms_z=(1.0,)),
        rb.Ref3dTemplate(
            "triad", arms_z=(0.5, 1.0), arms_x=(0.5,), positions=rb.parse_positions("1 x 2")
        ),
    ],
)
def test_template_file_round_trips_every_kind(tmp_path: Path, template) -> None:
    blank = rb.write_blank_template(tmp_path / rb.TEMPLATE_FILE_NAME, template)
    saved = rb.write_template(template, tmp_path / "clip_markers_1_line.template.toml")

    for path in (blank, saved):
        assert rb.read_template(path).points() == template.points()
    text = blank.read_text(encoding="utf-8")
    assert all(f"[{kind}]" in text for kind in rb.TEMPLATES)  # all three, to switch kind


def test_switching_kind_in_the_blank_template_uses_that_section(tmp_path: Path) -> None:
    path = rb.write_blank_template(tmp_path / "t.toml")
    path.write_text(
        path.read_text(encoding="utf-8").replace('kind = "pole"', 'kind = "box"'), "utf-8"
    )

    assert rb.read_template(path).points() == rb.box_points(1.0, 5.0, 1.0)


def test_older_flat_template_files_are_still_read(tmp_path: Path) -> None:
    path = tmp_path / "old.template.toml"
    path.write_text(
        'kind = "pole"\nsize = [0.0, 0.0, 0.0]\ndiv_x = []\ndiv_y = []\ndiv_z = []\n'
        "marks = [0.0, 1.0]\narms_x = []\narms_y = []\narms_z = []\n"
        "positions = [[0.0, 0.0], [0.55, 0.0], [0.0, 5.0], [0.55, 5.0]]\n",
        encoding="utf-8",
    )

    template = rb.read_template(path)

    assert template.points() == rb.pole_points((0.0, 1.0), rb.parse_positions("0.55 x 5"))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ('kind = "pole"\n[pole]\nmark = [0, 1]\n', "unknown key 'mark'"),
        (
            'kind = "pole"\n[pole]\nmarks = [0, 1]\nrect = [1, 5]\npositions = [[0, 0]]\n',
            "not both",
        ),
        ('kind = "stick"\n', "kind"),
        ('kind = "box"\n[box]\nsize = [1, 5]\n', "size"),
        ('kind = "pole"\n[pole]\nmarks = [0, 1]\nrect = [1.38]\n', "rect"),
        ('kind = "pole"\n[pole]\nmarks = ["a"]\n', "marks"),
        ('kind = "pole"\n[pole]\nmarks = [0, 1\n', "not valid TOML"),
    ],
)
def test_template_file_mistakes_name_the_key(tmp_path: Path, body: str, message: str) -> None:
    path = tmp_path / "t.toml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(rb.Ref3dError, match=message):
        rb.read_template(path)


def test_blank_template_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / rb.TEMPLATE_FILE_NAME
    path.write_text("mine\n", encoding="utf-8")

    with pytest.raises(rb.Ref3dError, match="exists"):
        rb.write_blank_template(path)
    assert path.read_text(encoding="utf-8") == "mine\n"


def test_cli_new_template_then_template_file(tmp_path: Path, capsys) -> None:
    path = tmp_path / rb.TEMPLATE_FILE_NAME
    assert rb.main(["--new-template", str(path), "--template", "triad"]) == 0
    assert rb.read_template(path).kind == "triad"
    assert rb.main(["--new-template", str(path)]) == 1  # exists: not written again

    out = tmp_path / "calib.ref3d"
    assert rb.main(["--template-file", str(path), "-o", str(out)]) == 0

    assert rb.read_ref3d_points(out) == [
        rb.Ref3dPoint(p.index, p.x, p.y, p.z, f"p{p.index}")
        for p in rb.read_template(path).points()
    ]
    assert "points (triad" in capsys.readouterr().out


def test_videos_table_maps_each_video_to_its_floor_position(tmp_path: Path) -> None:
    """[videos] names the floor position of each video; it round-trips and is checked."""
    path = tmp_path / "t.toml"
    path.write_text(
        'kind = "pole"\n[pole]\nmarks = [0, 1]\nrect = [1.38, 5]\n'
        '[videos]\n"pos1.mp4" = [0, 0]\n"pos2.mp4" = [1.38, 0]\n',
        encoding="utf-8",
    )
    template = rb.read_template(path)
    assert dict(template.video_positions) == {"pos1.mp4": (0.0, 0.0), "pos2.mp4": (1.38, 0.0)}
    assert template.describe().endswith("2 videos")
    copy = tmp_path / "copy.toml"
    rb.write_template(template, copy)
    assert dict(rb.read_template(copy).video_positions) == dict(template.video_positions)


def test_videos_table_rejects_a_position_the_object_does_not_have(tmp_path: Path) -> None:
    path = tmp_path / "t.toml"
    path.write_text(
        'kind = "pole"\n[pole]\nmarks = [0, 1]\nrect = [1.38, 5]\n[videos]\n"pos2.mp4" = [1.4, 0]\n',
        encoding="utf-8",
    )
    with pytest.raises(rb.Ref3dError, match="not one of the floor positions"):
        rb.read_template(path)


def test_videos_table_is_not_for_a_box(tmp_path: Path) -> None:
    path = tmp_path / "t.toml"
    path.write_text(
        'kind = "box"\n[box]\nsize = [1, 5, 1]\n[videos]\n"pos1.mp4" = [0, 0]\n',
        encoding="utf-8",
    )
    with pytest.raises(rb.Ref3dError, match="only for pole and triad"):
        rb.read_template(path)
