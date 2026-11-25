"""Tests for vaila.kick_panel (step panel of the kick_analysis loader), headless pygame."""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from vaila import kick_panel as kp

SENTINEL = (1, 2, 3)
SIZES = ((1366, 768, 600), (1920, 1080, 950))


@pytest.fixture(scope="module", autouse=True)
def _pygame():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture()
def font():
    return pygame.font.Font(None, 18)


def make_people(n=25):
    roles = ["kicker", "keeper", "other", "referee"]
    return [
        kp.PanelPerson(
            id=i + 1,
            role=roles[i % 4],
            team="AB"[i % 2] if i % 4 != 3 else "",
            first=i,
            last=95 + i,
            n=90,
            has_pose=i % 3 != 0,
        )
        for i in range(n)
    ]


def make_panel(**kw):
    counts = {"dlt3d": 60, "solved": 64, "approximate": 45}
    return kp.KickPanel(make_people(), counts, 169, **kw)


def render(panel, font, size=(1366, 768, 600), **kw):
    w, h, vh = size
    scr = pygame.Surface((w, h))
    scr.fill(SENTINEL)
    rect = panel.panel_rect(w, vh)
    panel.draw(
        scr,
        rect,
        font,
        frame=42,
        frame_source=kw.get("frame_source", "solved"),
        metric_text=kw.get("metric_text", {}),
    )
    return scr, rect


def click(panel, key):
    return panel.handle_click(panel.hit_rects[key].center)


# --------------------------------------------------------------- constants
def test_constants_match_design():
    assert kp.PANEL_WIDTH == 320
    assert kp.STEPS == ("calibration", "layers", "metrics", "export")
    assert kp.LAYERS == ("field", "boxes", "pose", "ball", "labels")
    assert kp.METRIC_KINDS == (
        "speed",
        "distance_ball",
        "distance_person",
        "height_kpt",
        "jump",
        "kpt_speed",
    )
    assert "pelvis" in kp.BODY_POINTS and len(kp.BODY_POINTS) == 7
    assert [k for k, _ in kp.MANUAL_MODES] == ["1", "3", "4", "7", "8", "9", "0", ",", "."]
    assert kp.ROLE_FILTERS == ("all", "kicker", "keeper", "team A", "team B", "referee", "selected")


def test_defaults():
    panel = make_panel()
    assert panel.is_open and panel.step == 1
    assert panel.layers == {
        "field": True,
        "boxes": False,
        "pose": False,
        "ball": False,
        "labels": False,
    }
    assert panel.role_filter == "all"
    assert panel.selected == set() and panel.active_person is None
    assert panel.metrics == [] and panel.last_export == "" and panel.message == ""


def test_metric_request_key_stable():
    assert kp.MetricRequest("speed", 21).key() == "speed:21"
    assert (
        kp.MetricRequest("height_kpt", 21, point="right_wrist").key() == "height_kpt:21:right_wrist"
    )
    assert kp.MetricRequest("distance_person", 21, 23).key() == "distance_person:21:23"
    assert kp.MetricRequest("jump", 5, point="pelvis").key() == "jump:5:pelvis"
    assert kp.MetricRequest("speed", 21).key() == kp.MetricRequest("speed", 21).key()
    assert hash(kp.MetricRequest("speed", 21)) == hash(kp.MetricRequest("speed", 21))


# ----------------------------------------------------------------- drawing
@pytest.mark.parametrize("lang", ["en", "pt"])
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("step", [1, 2, 3, 4])
def test_draw_every_step_stays_inside_panel(font, lang, size, step):
    panel = make_panel(lang=lang)
    for pid in (21, 3, 5):
        panel.toggle_person(pid)
    panel.step = step
    panel.pending_kind = "height_kpt"
    panel.metrics = [kp.MetricRequest("speed", 21), kp.MetricRequest("distance_person", 21, 3)]
    panel.last_export = (
        "/some/very/long/path/" + "x" * 200 + "/processed_kickmeasure_20261123_101500"
    )
    panel.message = "Select a second person."
    scr, rect = render(panel, font, size, metric_text={"speed:21": "now 12.3 | max 20.1 m/s"})
    arr = pygame.surfarray.array3d(scr)  # (w, h, 3)
    inside = np.zeros(arr.shape[:2], dtype=bool)
    inside[rect.x : rect.right, rect.y : rect.bottom] = True
    outside = arr[~inside]
    assert (outside == np.array(SENTINEL)).all()
    assert (arr[inside] != np.array(SENTINEL)).any()  # something was drawn inside
    assert rect.width == kp.PANEL_WIDTH and rect.right == size[0] and rect.height == size[2]


def test_draw_restores_clip(font):
    panel = make_panel()
    scr = pygame.Surface((800, 600))
    clip = pygame.Rect(10, 10, 700, 500)
    scr.set_clip(clip)
    panel.draw(scr, panel.panel_rect(800, 600), font, frame=0, frame_source="dlt3d", metric_text={})
    assert scr.get_clip() == clip


def test_draw_without_calib_data_and_unknown_source(font):
    panel = kp.KickPanel([], {}, 0)
    render(panel, font, frame_source=None)
    panel.step = 2
    render(panel, font)
    panel.step = 3
    render(panel, font)


def test_draw_many_frames_does_not_grow_caches(font):
    panel = make_panel()
    panel.step = 2
    for _ in range(5):
        render(panel, font)
    sizes = (len(panel._text_cache), len(panel._fit_cache), len(panel._wrap_cache))
    for _ in range(20):
        render(panel, font)
    assert (len(panel._text_cache), len(panel._fit_cache), len(panel._wrap_cache)) == sizes


# ----------------------------------------------------------- closed / open
def test_collapse_and_reopen(font):
    panel = make_panel()
    render(panel, font)
    assert click(panel, "collapse") == ("consumed",)
    assert not panel.is_open
    scr, rect = render(panel, font)
    assert rect.width == 28 and rect.right == 1366
    assert panel.handle_click((100, 100)) is None  # video area
    assert click(panel, "reopen") == ("consumed",)
    assert panel.is_open
    _, rect = render(panel, font)
    assert rect.width == kp.PANEL_WIDTH


def test_closed_tab_draws_only_inside_tab(font):
    panel = make_panel()
    panel.is_open = False
    scr, rect = render(panel, font)
    arr = pygame.surfarray.array3d(scr)
    inside = np.zeros(arr.shape[:2], dtype=bool)
    inside[rect.x : rect.right, rect.y : rect.bottom] = True
    assert (arr[~inside] == np.array(SENTINEL)).all()


def test_click_outside_panel_returns_none(font):
    panel = make_panel()
    assert panel.handle_click((10, 10)) is None  # before any draw
    render(panel, font)
    assert panel.handle_click((10, 10)) is None
    assert panel.handle_click((1366 - 5, 700)) is None  # below the panel (video_h = 600)


def test_click_on_empty_panel_area_is_consumed(font):
    panel = make_panel()
    render(panel, font)  # step 1 has a lot of empty space at the bottom
    assert panel.handle_click((1366 - 20, 580)) == ("consumed",)


# -------------------------------------------------------------------- tabs
def test_tabs_change_step(font):
    panel = make_panel()
    render(panel, font)
    for n in (2, 3, 4, 1):
        assert click(panel, f"tab:{n}") == ("consumed",)
        assert panel.step == n
        render(panel, font)


# ----------------------------------------------------------------- step 1
def test_step1_field_toggle(font):
    panel = make_panel()
    render(panel, font)
    assert click(panel, "layer:field") == ("layers",)
    assert panel.layers["field"] is False
    render(panel, font)
    assert click(panel, "layer:field") == ("layers",)
    assert panel.layers["field"] is True


# ----------------------------------------------------------------- step 2
def test_step2_layer_toggles(font):
    panel = make_panel()
    panel.step = 2
    render(panel, font)
    for name in ("boxes", "pose", "ball", "labels"):
        assert click(panel, f"layer:{name}") == ("layers",)
        assert panel.layers[name] is True


def test_step2_disabled_controls(font):
    panel = make_panel(has_pose=False, has_ball=False)
    panel.step = 2
    render(panel, font)
    assert click(panel, "layer:pose") == ("consumed",)
    assert click(panel, "layer:ball") == ("consumed",)
    assert click(panel, "ball_to_p0") == ("consumed",)
    assert panel.layers["pose"] is False and panel.layers["ball"] is False
    assert click(panel, "layer:boxes") == ("layers",)


def test_step2_ball_to_p0(font):
    panel = make_panel()
    panel.step = 2
    render(panel, font)
    assert click(panel, "ball_to_p0") == ("ball_to_p0",)


def test_person_toggle_and_active():
    panel = make_panel()
    panel.toggle_person(5)
    panel.toggle_person(2)
    assert panel.selected == {2, 5} and panel.active_person == 5
    panel.toggle_person(5)
    assert panel.selected == {2} and panel.active_person == 2
    panel.toggle_person(2)
    assert panel.selected == set() and panel.active_person is None


def test_role_filter_and_person_visible(font):
    panel = make_panel()
    assert all(panel.person_visible(p.id) for p in panel.people)
    assert not panel.person_visible(9999)
    panel.role_filter = "kicker"
    assert [p.id for p in panel.people if panel.person_visible(p.id)] == [1, 5, 9, 13, 17, 21, 25]
    panel.role_filter = "keeper"
    assert all(
        panel.people[i - 1].role == "keeper" for i in range(1, 26) if panel.person_visible(i)
    )
    panel.role_filter = "referee"
    assert {p.role for p in panel.people if panel.person_visible(p.id)} == {"referee"}
    panel.role_filter = "team A"
    assert {p.team for p in panel.people if panel.person_visible(p.id)} == {"A"}
    panel.role_filter = "team B"
    assert {p.team for p in panel.people if panel.person_visible(p.id)} == {"B"}
    panel.role_filter = "selected"
    assert not any(panel.person_visible(p.id) for p in panel.people)
    panel.toggle_person(7)
    assert [p.id for p in panel.people if panel.person_visible(p.id)] == [7]


def test_step2_role_chip_and_person_click(font):
    panel = make_panel()
    panel.step = 2
    render(panel, font)
    assert click(panel, "role:kicker") == ("selection",)
    assert panel.role_filter == "kicker"
    render(panel, font)
    assert "person:1" in panel.hit_rects and "person:2" not in panel.hit_rects
    assert click(panel, "person:1") == ("selection",)
    assert panel.selected == {1} and panel.active_person == 1
    assert click(panel, "person:1") == ("selection",)
    assert panel.selected == set()


def test_step2_person_list_scrolls_with_wheel(font):
    panel = make_panel()
    panel.step = 2
    render(panel, font)
    assert "person:1" in panel.hit_rects and "person:25" not in panel.hit_rects
    area = panel.scroll_areas["people"]
    assert panel.scroll["people"] == 0
    assert panel.handle_wheel(area.center, -3) is True  # wheel down
    assert panel.scroll["people"] > 0
    for _ in range(40):
        panel.handle_wheel(area.center, -3)
    render(panel, font)
    assert "person:25" in panel.hit_rects and "person:1" not in panel.hit_rects
    top = panel.scroll["people"]
    panel.handle_wheel(area.center, 1)
    assert panel.scroll["people"] < top
    for _ in range(40):
        panel.handle_wheel(area.center, 3)
    assert panel.scroll["people"] == 0


def test_wheel_outside_panel_not_handled(font):
    panel = make_panel()
    render(panel, font)
    assert panel.handle_wheel((10, 10), 1) is False
    panel.step = 3
    render(panel, font)
    assert panel.handle_wheel((1366 - 10, 10), 1) is True  # inside, over the tabs: consumed


def test_clipped_person_rows_hit_only_visible_part(font):
    panel = make_panel()
    panel.step = 2
    render(panel, font)
    area = panel.scroll_areas["people"]
    panel.handle_wheel(area.center, -1)
    render(panel, font)
    for key, r in panel.hit_rects.items():
        if key.startswith("person:"):
            assert area.contains(r)


# ----------------------------------------------------------------- step 3
def test_add_metric_needs_a_person():
    panel = make_panel()
    with pytest.raises(ValueError):
        panel.add_metric()


def test_add_remove_metric():
    panel = make_panel()
    panel.toggle_person(21)
    req = panel.add_metric()
    assert req == kp.MetricRequest("speed", 21) and panel.metrics == [req]
    assert panel.add_metric() == req and len(panel.metrics) == 1  # no duplicate
    panel.pending_kind = "height_kpt"
    panel.pending_point = "right_wrist"
    assert panel.add_metric().key() == "height_kpt:21:right_wrist"
    panel.pending_kind = "jump"
    panel.pending_point = "head"
    assert panel.add_metric().key() == "jump:21:pelvis"  # pelvis fixed
    panel.pending_kind = "distance_ball"
    assert panel.add_metric().key() == "distance_ball:21"
    panel.remove_metric("speed:21")
    assert [m.key() for m in panel.metrics] == [
        "height_kpt:21:right_wrist",
        "jump:21:pelvis",
        "distance_ball:21",
    ]
    panel.remove_metric("nope")
    assert len(panel.metrics) == 3


def test_distance_person_uses_other_selected_person():
    panel = make_panel()
    panel.toggle_person(21)
    panel.pending_kind = "distance_person"
    with pytest.raises(ValueError, match="second person"):
        panel.add_metric()
    panel.toggle_person(23)
    req = panel.add_metric()
    assert req.key() == "distance_person:21:23"


def test_step3_pickers_and_add_click(font):
    panel = make_panel()
    panel.toggle_person(21)
    panel.step = 3
    render(panel, font, size=(1920, 1080, 950))
    assert click(panel, "kind:height_kpt") == ("consumed",)
    assert panel.pending_kind == "height_kpt"
    render(panel, font, size=(1920, 1080, 950))
    assert click(panel, "point:head") == ("consumed",)
    assert panel.pending_point == "head"
    assert click(panel, "add_metric") == ("metrics",)
    assert [m.key() for m in panel.metrics] == ["height_kpt:21:head"]
    assert panel.message == ""
    render(panel, font, size=(1920, 1080, 950), metric_text={"height_kpt:21:head": "now 1.5"})
    assert click(panel, "remove:height_kpt:21:head") == ("metrics",)
    assert panel.metrics == []


def test_step3_failed_add_stores_message_not_exception(font):
    panel = make_panel()
    panel.step = 3
    render(panel, font)
    assert click(panel, "add_metric") == ("consumed",)  # no person selected
    assert panel.message and panel.metrics == []
    panel.toggle_person(21)
    panel.pending_kind = "distance_person"
    render(panel, font)
    assert click(panel, "add_metric") == ("consumed",)
    assert "second" in panel.message
    scr, rect = render(panel, font)  # message drawn without leaving the panel
    arr = pygame.surfarray.array3d(scr)
    assert (arr[: rect.x] == np.array(SENTINEL)).all()
    panel.toggle_person(23)
    assert click(panel, "add_metric") == ("metrics",)
    assert panel.message == ""


def test_step3_actor_cycle(font):
    panel = make_panel()
    for pid in (3, 8, 21):
        panel.toggle_person(pid)
    assert panel.active_person == 3
    panel.step = 3
    render(panel, font)
    assert click(panel, "actor_next") == ("consumed",)
    assert panel.active_person == 8
    click(panel, "actor_next")
    click(panel, "actor_next")
    assert panel.active_person == 3  # wraps
    click(panel, "actor_prev")
    assert panel.active_person == 21


def test_step3_actor_buttons_disabled_with_single_person(font):
    panel = make_panel()
    panel.toggle_person(3)
    panel.step = 3
    render(panel, font)
    assert click(panel, "actor_next") == ("consumed",)
    assert panel.active_person == 3


def test_step3_manual_measure_buttons(font):
    panel = make_panel()
    panel.step = 3
    render(panel, font, size=(1920, 1080, 950))
    for key, _ in kp.MANUAL_MODES:
        assert click(panel, f"measure:{key}") == ("measure", key)


def test_step3_manual_buttons_reachable_on_short_video(font):
    panel = make_panel()
    panel.step = 3
    panel.pending_kind = "height_kpt"
    panel.toggle_person(21)
    panel.metrics = [kp.MetricRequest("speed", 21)] * 3
    _, rect = render(panel, font)
    for key, _ in kp.MANUAL_MODES:
        assert rect.contains(panel.hit_rects[f"measure:{key}"])


def test_step3_metric_list_scrolls(font):
    panel = make_panel()
    panel.toggle_person(21)
    for p in kp.BODY_POINTS:
        panel.pending_kind = "kpt_speed"
        panel.pending_point = p
        panel.add_metric()
        panel.pending_kind = "height_kpt"
        panel.add_metric()
    panel.step = 3
    render(panel, font)
    area = panel.scroll_areas["metrics"]
    before = panel.scroll["metrics"]
    panel.handle_wheel(area.center, -2)
    assert panel.scroll["metrics"] > before


def test_kind_list_scrolls_and_pending_kind_visible(font):
    panel = make_panel()
    panel.toggle_person(21)
    panel.step = 3
    panel.pending_kind = "kpt_speed"
    panel._ensure_kind = True
    render(panel, font, size=(1366, 768, 480))
    assert "kind:kpt_speed" in panel.hit_rects
    area = panel.scroll_areas["kinds"]
    panel.handle_wheel(area.center, 5)
    assert panel.scroll["kinds"] >= 0


# ----------------------------------------------------------------- step 4
def test_step4_export_buttons(font):
    panel = make_panel()
    panel.step = 4
    render(panel, font)
    for what in ("mkvis3d", "blender", "metrics", "all"):
        assert click(panel, f"export:{what}") == ("export", what)


def test_step4_open_disabled_until_export(font):
    panel = make_panel()
    panel.step = 4
    render(panel, font)
    assert click(panel, "open:mkvis3d") == ("consumed",)
    assert click(panel, "open:blender") == ("consumed",)
    panel.last_export = "/tmp/out/processed_kickmeasure_20261123_101500"
    render(panel, font)
    assert click(panel, "open:mkvis3d") == ("open", "mkvis3d")
    assert click(panel, "open:blender") == ("open", "blender")


# --------------------------------------------------------------- languages
def test_portuguese_texts_have_accents(font):
    panel = make_panel(lang="pt")
    assert panel._t("tab3") == "3 Métr."
    assert "calibração" in panel._t("hint1")
    assert panel._t("layer_labels") == "Rótulos"
    assert kp._ROLE_LABELS["pt"]["referee"] == "árbitro"
    assert kp._POINT_LABELS["pt"]["head"] == "cabeça"


def test_all_texts_defined_in_both_languages():
    assert kp._TEXTS["en"].keys() == kp._TEXTS["pt"].keys()
    for lang in ("en", "pt"):
        assert set(kp._KIND_LABELS[lang]) == set(kp.METRIC_KINDS)
        assert set(kp._POINT_LABELS[lang]) == set(kp.BODY_POINTS)
        assert set(kp._ROLE_LABELS[lang]) >= set(kp.ROLE_FILTERS)
        assert {n for _, n in kp.MANUAL_MODES} <= set(kp._MANUAL_LABELS[lang])


def test_unknown_lang_falls_back_to_english():
    assert make_panel(lang="xx").lang == "en"


# ------------------------------------------------- audit fixes (labels, rows, PT, export)
def test_metric_label_en_and_pt():
    en, pt = make_panel(), make_panel(lang="pt")
    cases = [
        (kp.MetricRequest("speed", 21), "Speed · ID 21", "Velocidade · ID 21"),
        (
            kp.MetricRequest("height_kpt", 21, point="right_wrist"),
            "Height · R wrist · ID 21",
            "Altura · punho dir. · ID 21",
        ),
        (
            kp.MetricRequest("distance_person", 21, 23),
            "Distance ID 21 <-> ID 23",
            "Distância ID 21 <-> ID 23",
        ),
        (kp.MetricRequest("jump", 21, point="pelvis"), "Jump · ID 21", "Salto · ID 21"),
        (
            kp.MetricRequest("kpt_speed", 21, point="right_ankle"),
            "Point speed · R ankle · ID 21",
            "Velocidade do ponto · tornozelo dir. · ID 21",
        ),
        (
            kp.MetricRequest("distance_ball", 5),
            "Distance to ball · ID 5",
            "Distância até a bola · ID 5",
        ),
    ]
    for req, e, p in cases:
        assert en.metric_label(req) == e
        assert pt.metric_label(req) == p


def test_step3_rows_draw_label_not_key(font):
    panel = make_panel()
    panel.toggle_person(21)
    panel.step = 3
    panel.metrics = [kp.MetricRequest("speed", 21)]
    panel._text_cache.clear()
    render(panel, font, size=(1920, 1080, 950))
    texts = {k[1] for k in panel._text_cache}
    assert "Speed · ID 21" in texts and "speed:21" not in texts


@pytest.mark.parametrize("name,page", [("people", 2), ("kinds", 3), ("metrics", 3)])
def test_scroll_lists_move_by_whole_rows(font, name, page):
    panel = make_panel(lang="pt")
    panel.toggle_person(21)
    panel.step = page
    panel.pending_kind = "height_kpt"
    for p in kp.BODY_POINTS:
        panel.pending_point = p
        panel.add_metric()
    panel._ensure_kind = True
    render(panel, font, size=(1366, 768, 480))
    area = panel.scroll_areas[name]
    row_h = panel._scroll_info[name][1]
    assert panel._scroll_info[name][0] > area.h  # really scrolls
    for _ in range(12):
        panel.handle_wheel(area.center, -1)
        assert panel.scroll[name] % row_h == 0
        render(panel, font, size=(1366, 768, 480))
        assert panel.scroll[name] % row_h == 0
        assert area.h % row_h == 0
        prefix = {"people": "person:", "kinds": "kind:", "metrics": "remove:"}[name]
        rows = [r for k, r in panel.hit_rects.items() if k.startswith(prefix)]
        assert rows and min(r.y for r in rows) >= area.y - 1
        assert all(area.y <= r.y and r.bottom <= area.bottom + row_h for r in rows)


def test_kind_rows_not_cut_at_top_with_pending_kind_visible(font):
    panel = make_panel()
    panel.toggle_person(21)
    panel.step = 3
    panel.pending_kind = "kpt_speed"
    panel._ensure_kind = True
    render(panel, font, size=(1366, 768, 480))
    area = panel.scroll_areas["kinds"]
    r = panel.hit_rects["kind:kpt_speed"]
    assert area.contains(r)
    assert panel.scroll["kinds"] % panel._scroll_info["kinds"][1] == 0


def test_portuguese_step1_uses_resolvida_aproximada(font):
    panel = make_panel(lang="pt")
    assert panel._t("source_help").startswith("resolvida")
    assert "aproximada" in panel._t("source_help")
    assert "solved" not in panel._t("source_help") and "approx" not in panel._t("source_help")
    panel._text_cache.clear()
    render(panel, font, frame_source="approximate")
    texts = " | ".join(k[1] for k in panel._text_cache)
    assert (
        "resolvida: 64/169" in texts and "aproximada: 45/169" in texts and "DLT3D: 60/169" in texts
    )
    assert "aproximada" in {k[1] for k in panel._text_cache}
    assert "solved" not in texts and "approx" not in texts.replace("aproximada", "")
    en = make_panel()
    en._text_cache.clear()
    render(en, font)
    assert "solved: 64/169" in " | ".join(k[1] for k in en._text_cache)


@pytest.mark.parametrize(
    "lang,msg",
    [
        ("en", "No one selected: exports every person with a 3D pose"),
        ("pt", "Ninguém marcado: exporta todas as pessoas com pose 3D"),
    ],
)
def test_step4_note_when_nobody_selected(font, lang, msg):
    panel = make_panel(lang=lang)
    panel.step = 4
    assert panel._t("export_all_people") == msg
    panel._wrap_cache.clear()
    render(panel, font, size=(1920, 1080, 950))
    shown = " ".join(" ".join(v) for v in panel._wrap_cache.values())
    assert msg.split(":")[0] in shown
    panel._wrap_cache.clear()
    panel.toggle_person(3)
    render(panel, font, size=(1920, 1080, 950))
    assert msg.split(":")[0] not in " ".join(" ".join(v) for v in panel._wrap_cache.values())


def test_step1_counts_the_tracked_frames(font):
    """Frames whose camera was carried by the image motion (kick_measure "tracked") are listed."""
    for lang, name in (("en", "tracked"), ("pt", "rastreada")):
        panel = kp.KickPanel(
            make_people(),
            {"dlt3d": 61, "solved": 64, "tracked": 42, "approximate": 2},
            169,
            lang=lang,
        )
        panel._text_cache.clear()
        render(panel, font, frame_source="tracked")
        texts = " | ".join(k[1] for k in panel._text_cache)
        assert f"{name}: 42/169" in texts
        assert name in panel._t("source_help")
