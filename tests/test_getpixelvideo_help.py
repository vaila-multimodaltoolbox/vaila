"""Content of the H help screen and of the toolbar hints of getpixelvideo.

Author: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
Update Date: 15 November 2026
Version: 0.4.18
"""

from __future__ import annotations

import inspect

import vaila.getpixelvideo as gpv
from vaila import getpixelvideo_help as gh


def test_every_entry_has_both_languages_and_a_way_to_do_it() -> None:
    for entry in gh.help_entries():
        assert entry.en.strip() and entry.pt.strip(), entry
        assert entry.keys or entry.mouse or entry.button, entry
        assert entry.section in {s.id for s in gh.SECTIONS}, entry
        if entry.mouse:
            assert entry.mouse in gh.MOUSE_PT, entry


def test_no_section_is_empty_and_digits_reach_them_all() -> None:
    assert len(gh.SECTIONS) <= 10  # keys 1..9, 0
    for section in gh.SECTIONS:
        assert gh.help_entries(section.id), section.id
        assert section.title("en") and section.title("pt")


def test_f8_is_the_open_button() -> None:
    f8 = [e for e in gh.help_entries() if "F8" in e.keys]
    assert len(f8) == 1
    assert f8[0].button == "Open"


def test_esc_says_that_it_saves_and_closes() -> None:
    esc = next(e for e in gh.help_entries() if e.keys == ("ESC",))
    assert "save" in esc.en.lower() and "close" in esc.en.lower()
    assert "salva" in esc.pt.lower() and "fecha" in esc.pt.lower()


def test_one_line_steps_start_with_insert_only_in_visual() -> None:
    visual = gh.now_steps(gh.HelpContext(mode="1-line", insert=False, n_points=2, next_point=2))
    insert = gh.now_steps(gh.HelpContext(mode="1-line", insert=True, n_points=2, next_point=2))

    assert visual[0].keys == ("Ctrl+I",)
    assert all("Ctrl+I" not in step.keys for step in insert)
    assert insert[0].mouse == "Left click" and "p2" in insert[0].en
    next_video = next(step for step in insert if "F8" in step.keys)
    assert next_video.button == "Add video"
    assert "2 points" in next_video.en
    assert any(step.button == "Save" for step in insert)


def test_one_line_steps_with_no_points_offer_open() -> None:
    steps = gh.now_steps(gh.HelpContext(mode="1-line", insert=True))
    assert next(step for step in steps if "F8" in step.keys).button == "Open"


def test_now_title_names_the_mode_in_each_language() -> None:
    ctx = gh.HelpContext(mode="1-line", insert=True, n_points=1)
    assert gh.now_title(ctx, "en") == "You are now in: INSERT · 1-line · 1 point"
    assert gh.now_title(ctx, "pt") == "Agora você está em: INSERT · 1-line · 1 ponto"


def test_every_mode_has_steps() -> None:
    for mode in ("mark", "seq", "1-line", "labeling", "measure", "calib", "freekiki", "ref3d"):
        steps = gh.now_steps(gh.HelpContext(mode=mode, insert=True))
        assert steps, mode
        assert all(step.en and step.pt for step in steps), mode


def test_every_toolbar_button_has_a_hint() -> None:
    src = inspect.getsource(gpv.play_video_with_controls)
    start = src.index("TOOLBAR_HIT_RECTS.update(")
    block = src[start : src.index("return (", start)]
    ids = {
        line.split('"')[1]
        for line in block.splitlines()
        if line.strip().startswith('("') and "_rect" in line
    }
    ids |= {"open", "open_next"}
    assert ids <= set(gh.TOOLBAR_HINTS), ids - set(gh.TOOLBAR_HINTS)
    assert gh.toolbar_hint("open", "en") == (("F8",), "Open another video or image")
    assert gh.toolbar_hint("open", "pt") == (("F8",), "Abre outro vídeo ou imagem")
    assert gh.toolbar_hint("nope", "en") is None


def test_wrap_text_keeps_every_word_within_the_width() -> None:
    lines = gh.wrap_text("one two three four five", 9, len)
    assert lines == ["one two", "three", "four five"]
    assert " ".join(lines) == "one two three four five"
