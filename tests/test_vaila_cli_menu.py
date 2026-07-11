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
    entry = _entry_by_code("b4_r4_c1")
    assert entry is not None
    assert entry.label == "YOLO + FB"


def test_search_yolo_finds_yolo_fb() -> None:
    matches = _search_entries("yolo")
    codes = {m.code for m in matches}
    assert "B4_r4_c1" in codes


def test_yolo_fb_hint_lists_launchers() -> None:
    entry = _entry_by_code("B4_r4_c1")
    assert entry is not None
    hint = get_cli_hint(handler=entry.handler, code=entry.code, label=entry.label)
    assert not hint.invoke_handler
    assert any("yolov26track" in cmd for cmd in hint.commands)
    assert any("vaila_sam.py" in cmd for cmd in hint.commands)


def test_resolve_handler_method() -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    entry = _entry_by_code("A_r1_c1")
    assert entry is not None
    handler = resolve_handler(app, entry)
    assert callable(handler)
    assert handler.__name__ == "rename_files"
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
        entry = _entry_by_code("B4_r4_c1")
        assert entry is not None
        from vaila.vaila_cli_menu import _run_entry

        _run_entry(app, entry, console, headless=True)
        out = buffer.getvalue()
        assert "yolov26track" in out
        assert "vaila_sam.py" in out
        assert "Equivalent CLI" in out
    finally:
        app.destroy()


def test_run_cli_menu_one_shot_yolo_fb(capsys: pytest.CaptureFixture[str]) -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    try:
        run_cli_menu(app, initial_code="B4_r4_c1", headless=True)
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


def test_entry_by_number_resolves_yolo_fb() -> None:
    entry = _entry_by_code("B4_r4_c1")
    assert entry is not None
    num = _NUMBER_BY_CODE[entry.code.upper()]
    assert _entry_by_number(num) == entry


def test_run_cli_menu_one_shot_by_number(capsys: pytest.CaptureFixture[str]) -> None:
    Vaila = _load_vaila_app_class()
    app = Vaila(gui=False)
    entry = _entry_by_code("B4_r4_c1")
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
        run_cli_menu(app, initial_code="/yolo", headless=True)
    finally:
        app.destroy()
    captured = capsys.readouterr().out
    assert "B4_r4_c1" in captured
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
