"""Tests for vailá CLI menu registry, search, and headless CLI hints."""

from __future__ import annotations

import importlib.util
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from vaila.vaila_cli_hints import get_cli_hint
from vaila.vaila_cli_menu import (
    _NUMBER_BY_CODE,
    VAILA_MENU_ENTRIES,
    _entry_by_code,
    _entry_by_number,
    _search_entries,
    resolve_handler,
    run_cli_menu,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_vaila_app_class():
    root_vaila = _REPO_ROOT / "vaila.py"
    spec = importlib.util.spec_from_file_location("vaila_main", root_vaila)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["vaila_main"] = module
    spec.loader.exec_module(module)
    return module.Vaila


def test_menu_registry_has_unique_codes() -> None:
    codes = [entry.code for entry in VAILA_MENU_ENTRIES]
    assert len(codes) == len(set(codes))


def test_entry_by_code_case_insensitive() -> None:
    entry = _entry_by_code("b1_r1_c4")
    assert entry is not None
    assert entry.label == "Markerless 2D"


def test_search_markerless_finds_2d_and_3d_choosers() -> None:
    matches = _search_entries("markerless")
    codes = {m.code for m in matches}
    assert "B1_r1_c4" in codes
    assert "B1_r1_c5" in codes


def test_search_dlt_finds_toolkit_chooser() -> None:
    matches = _search_entries("dlt")
    codes = {m.code for m in matches}
    assert "C_A_r2_c1" in codes


def test_markerless_2d_hint_lists_absorbed_launchers() -> None:
    entry = _entry_by_code("B1_r1_c4")
    assert entry is not None
    hint = get_cli_hint(handler=entry.handler, code=entry.code, label=entry.label)
    assert not hint.invoke_handler
    assert any("yolov26track" in cmd for cmd in hint.commands)
    assert any("vaila_sam.py" in cmd for cmd in hint.commands)
    assert any("markerless2d_mpyolo.py" in cmd for cmd in hint.commands)
    assert any("mphands.py" in cmd for cmd in hint.commands)
    assert any("mpangles.py" in cmd for cmd in hint.commands)
    assert any("mp_facemesh.py" in cmd for cmd in hint.commands)
    assert any("markerless_live.py" in cmd for cmd in hint.commands)


def test_retired_standalone_codes_are_vaila_placeholders() -> None:
    for code in ("B4_r4_c3", "B4_r4_c4", "B4_r4_c5", "B5_r6_c2"):
        entry = _entry_by_code(code)
        assert entry is not None
        assert entry.label == "vailá"
        assert entry.handler == "show_vaila_message"


def test_markerless_3d_hint_lists_sam3dinov3_launchers() -> None:
    entry = _entry_by_code("B1_r1_c5")
    assert entry is not None
    hint = get_cli_hint(handler=entry.handler, code=entry.code, label=entry.label)
    assert not hint.invoke_handler
    assert any("sam3dinov3.py" in cmd for cmd in hint.commands)
    assert any("sam3dinov3_visualize.py" in cmd for cmd in hint.commands)


def test_dlt_rec_toolkit_hint_lists_all_six_scripts() -> None:
    entry = _entry_by_code("C_A_r2_c1")
    assert entry is not None
    assert entry.label == "DLT/REC 2D-3D"
    hint = get_cli_hint(handler=entry.handler, code=entry.code, label=entry.label)
    assert not hint.invoke_handler
    for script in (
        "dlt2d.py",
        "rec2d_one_dlt2d.py",
        "rec2d.py",
        "dlt3d.py",
        "rec3d_one_dlt3d.py",
        "rec3d.py",
    ):
        assert any(script in cmd for cmd in hint.commands), script


def test_resolve_handler_method() -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    entry = _entry_by_code("A_r1_c1")
    assert entry is not None
    handler = resolve_handler(app, entry)
    assert callable(handler)
    assert handler.__name__ == "rename_files"
    app.destroy()


def test_resolve_handler_dlt_rec_toolkit() -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    entry = _entry_by_code("C_A_r2_c1")
    assert entry is not None
    handler = resolve_handler(app, entry)
    assert callable(handler)
    assert handler.__name__ == "dlt_rec_toolkit"
    app.destroy()


def test_vaila_gui_false_does_not_show_window() -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    try:
        assert app.state() == "withdrawn"
    finally:
        app.destroy()


def test_headless_direct_code_prints_cli_not_gui() -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True, width=120, highlight=False)
    try:
        entry = _entry_by_code("B1_r1_c4")
        assert entry is not None
        from vaila.vaila_cli_menu import _run_entry

        _run_entry(app, entry, console, headless=True)
        out = buffer.getvalue()
        assert "yolov26track" in out
        assert "vaila_sam.py" in out
        assert "Equivalent CLI" in out
    finally:
        app.destroy()


def test_run_cli_menu_one_shot_markerless_2d(capsys: pytest.CaptureFixture[str]) -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    try:
        run_cli_menu(app, initial_code="B1_r1_c4", headless=True)
    finally:
        app.destroy()
    captured = capsys.readouterr().out
    assert "yolov26track" in captured
    assert "vaila_sam.py" in captured


def test_tools_grid_aligns_three_columns() -> None:
    from io import StringIO

    from rich.console import Console

    from vaila.vaila_cli_menu import _render_tools_grid

    console = Console(file=StringIO(), force_terminal=True, width=120, highlight=False)
    panel = _render_tools_grid(console)
    console.print(panel)
    out = console.file.getvalue()

    assert "C_A Data Files" in out
    assert "C_B Video/Image" in out
    assert "C_C Visualiz." in out
    assert "45.r1_c1 Edit CSV" in out
    assert "60.r1_c1 Video" in out
    assert "75.r1_c1 Show C3D" in out
    # Old glued multi-cell rows must not appear
    assert "Draw Box75" not in out
    assert "Fi…60" not in out


def test_entry_by_number_resolves_markerless_2d() -> None:
    entry = _entry_by_code("B1_r1_c4")
    assert entry is not None
    num = _NUMBER_BY_CODE[entry.code.upper()]
    assert _entry_by_number(num) == entry


def test_run_cli_menu_one_shot_by_number(capsys: pytest.CaptureFixture[str]) -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    entry = _entry_by_code("B1_r1_c4")
    assert entry is not None
    num = _NUMBER_BY_CODE[entry.code.upper()]
    try:
        run_cli_menu(app, initial_code=str(num), headless=True)
    finally:
        app.destroy()
    captured = capsys.readouterr().out
    assert "yolov26track" in captured


def test_run_cli_menu_slash_search_lists_matches(capsys: pytest.CaptureFixture[str]) -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    try:
        run_cli_menu(app, initial_code="/markerless", headless=True)
    finally:
        app.destroy()
    captured = capsys.readouterr().out
    assert "B1_r1_c4" in captured
    assert "Multiple matches" in captured


@pytest.mark.parametrize(
    ("code", "label"),
    [
        ("HELP", "Help"),
        ("EXIT", "Exit"),
        ("C_B_r3_c1", "GetPixelCoord"),
    ],
)
def test_known_menu_entries(code: str, label: str) -> None:
    entry = _entry_by_code(code)
    assert entry is not None
    assert entry.label == label
