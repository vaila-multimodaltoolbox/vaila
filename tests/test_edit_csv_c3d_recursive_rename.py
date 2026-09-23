"""Tests for the recursive depth-limited file scan added to
vaila/edit_csv_c3d.py (find_edit_csv_c3d_files / _headless_process max_depth)
and the bulk column rename/renumber helpers added to
vaila/rearrange_data.py (_build_renumber_map / _validate_rename_map).

Update Date: 23 September 2026
Version: 0.4.5
"""

import os

import pandas as pd
import pytest

try:
    from vaila.edit_csv_c3d import (
        _headless_process,
        _newest_edited_csv,
        _prompt_file_selection,
        find_edit_csv_c3d_files,
    )
    from vaila.rearrange_data import ColumnReorderGUI, _build_renumber_map, _validate_rename_map
except ImportError:  # standalone execution
    from edit_csv_c3d import (  # ty: ignore[unresolved-import]
        _headless_process,
        _newest_edited_csv,
        _prompt_file_selection,
        find_edit_csv_c3d_files,
    )
    from rearrange_data import (  # ty: ignore[unresolved-import]
        ColumnReorderGUI,
        _build_renumber_map,
        _validate_rename_map,
    )


def _touch_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame": [0, 1], "a": [1.0, 2.0]}).to_csv(path, index=False)


def test_find_edit_csv_c3d_files_depth_zero_is_top_level_only(tmp_path):
    _touch_csv(tmp_path / "root.csv")
    _touch_csv(tmp_path / "sub" / "nested.csv")

    found = find_edit_csv_c3d_files(str(tmp_path), max_depth=0)

    assert found == ["root.csv"]


def test_find_edit_csv_c3d_files_depth_one_descends_one_level(tmp_path):
    _touch_csv(tmp_path / "root.csv")
    _touch_csv(tmp_path / "a" / "level1.csv")
    _touch_csv(tmp_path / "a" / "b" / "level2.csv")

    found = find_edit_csv_c3d_files(str(tmp_path), max_depth=1)

    assert found == ["a/level1.csv", "root.csv"]


def test_find_edit_csv_c3d_files_depth_unlimited_finds_everything(tmp_path):
    _touch_csv(tmp_path / "root.csv")
    _touch_csv(tmp_path / "a" / "level1.csv")
    _touch_csv(tmp_path / "a" / "b" / "level2.csv")

    found = find_edit_csv_c3d_files(str(tmp_path), max_depth=-1)

    assert found == ["a/b/level2.csv", "a/level1.csv", "root.csv"]


def test_find_edit_csv_c3d_files_skips_own_processed_output(tmp_path):
    _touch_csv(tmp_path / "root.csv")
    _touch_csv(tmp_path / "processed_edit_csv_c3d_20260923_120000" / "root_final.csv")
    _touch_csv(tmp_path / "sub" / "_staging" / "staged.csv")
    _touch_csv(tmp_path / "sub" / "_staging" / "data_rearranged" / "staged_x.csv")

    found = find_edit_csv_c3d_files(str(tmp_path), max_depth=-1)

    assert found == ["root.csv"]


def test_headless_process_max_depth_preserves_subdirectory_structure(tmp_path):
    input_dir = tmp_path / "in"
    _touch_csv(input_dir / "root.csv")
    _touch_csv(input_dir / "a" / "nested.csv")
    _touch_csv(input_dir / "a" / "b" / "deep.csv")

    output_dir = tmp_path / "out"
    written = _headless_process(str(input_dir), str(output_dir), columns=None, max_depth=-1)

    written_set = {str(p) for p in written}
    assert len(written) == 3
    assert str(output_dir / "root_final.csv") in written_set
    assert str(output_dir / "a" / "nested_final.csv") in written_set
    assert str(output_dir / "a" / "b" / "deep_final.csv") in written_set


def test_build_renumber_map_shifts_marker_indices():
    headers = ["frame"]
    for i in range(1, 71):
        headers.append(f"p{i}_x")
        headers.append(f"p{i}_y")

    rename_map = _build_renumber_map(r"p(\d+)_", -1, headers)

    assert len(rename_map) == 140
    assert rename_map["p1_x"] == "p0_x"
    assert rename_map["p1_y"] == "p0_y"
    assert rename_map["p70_x"] == "p69_x"
    assert rename_map["p70_y"] == "p69_y"
    assert "frame" not in rename_map


def test_build_renumber_map_rejects_pattern_without_single_group():
    with pytest.raises(ValueError):
        _build_renumber_map(r"p(\d+)_(\w)", -1, ["p1_x"])


def test_validate_rename_map_rejects_duplicate_resulting_headers():
    headers = ["p0_x", "p1_x"]
    rename_map = {"p0_x": "p1_x"}

    problems = _validate_rename_map(rename_map, headers)

    assert problems
    assert "Duplicate" in problems[0]


def test_validate_rename_map_rejects_split_marker_triple():
    headers = ["p1_x", "p1_y", "p1_z"]
    rename_map = {"p1_x": "p0_x"}

    problems = _validate_rename_map(rename_map, headers)

    assert problems
    assert "split" in problems[0].lower()


def test_validate_rename_map_accepts_full_triple_rename():
    headers = ["p1_x", "p1_y", "p1_z"]
    rename_map = {"p1_x": "p0_x", "p1_y": "p0_y", "p1_z": "p0_z"}

    assert _validate_rename_map(rename_map, headers) == []


def test_prompt_file_selection_dialog_becomes_visible(tmp_path):
    """Regression for the "button does nothing" bug: `_prompt_file_selection`'s
    Toplevel used to inherit the withdrawn state of its `tk.Tk()` parent
    whenever `.transient(parent)` was set on that withdrawn parent (X11
    quirk), leaving the dialog invisible while `wait_window()` blocked
    forever. Skipped when no display is available (matches the guard used
    by `tests/test_extractpng.py::test_extractpng_gui_builds`)."""
    import tkinter as tk

    try:
        root = tk.Tk()
        root.withdraw()
    except tk.TclError:
        pytest.skip("No display available for Tkinter GUI test")

    _touch_csv(tmp_path / "a.csv")
    state: dict = {}

    def inspect_and_cancel():
        for child in root.winfo_children():
            if isinstance(child, tk.Toplevel):
                state["viewable"] = child.winfo_viewable()
                state["wm_state"] = child.wm_state()
                child.destroy()

    root.after(200, inspect_and_cancel)
    try:
        _prompt_file_selection(root, str(tmp_path))
    finally:
        root.destroy()

    assert state.get("wm_state") == "normal"
    assert state.get("viewable") == 1


def test_newest_edited_csv_prefers_final_suffix_over_newer_plain_save(tmp_path):
    """Regression: a bulk rename applied AFTER an earlier Ctrl+S save must not
    be discarded by picking the plain (non-final) file just because it has a
    newer mtime than a `_final` save written earlier in the same run."""
    rearranged_dir = tmp_path / "data_rearranged"
    rearranged_dir.mkdir()
    stem = "s10_markers"

    final_path = rearranged_dir / f"{stem}_20260923_110300_final.csv"
    final_path.write_text("frame,p0_x\n0,1.0\n")
    older_time = final_path.stat().st_mtime - 100
    os.utime(final_path, (older_time, older_time))

    plain_path = rearranged_dir / f"{stem}_20260923_110306.csv"
    plain_path.write_text("frame,p1_x\n0,1.0\n")

    result = _newest_edited_csv(str(rearranged_dir), stem)

    assert result == str(final_path)


def test_newest_edited_csv_excludes_final_from_fallback_only_when_absent(tmp_path):
    """No `_final` file exists yet -> fall back to newest match, same as before."""
    rearranged_dir = tmp_path / "data_rearranged"
    rearranged_dir.mkdir()
    stem = "s10_markers"

    resetidx_path = rearranged_dir / f"{stem}_20260923_110253_resetidx.csv"
    resetidx_path.write_text("frame,p1_x\n0,1.0\n")

    plain_path = rearranged_dir / f"{stem}_20260923_110306.csv"
    plain_path.write_text("frame,p1_x\n0,1.0\n")

    result = _newest_edited_csv(str(rearranged_dir), stem)

    assert result == str(plain_path)


def test_on_window_close_prompts_even_after_a_prior_save(tmp_path):
    """Regression for the sticky `self.saved` flag bug: once `self.saved`
    becomes True (any prior Ctrl+S / Save & Exit this session), a LATER edit
    (e.g. bulk column rename) must still trigger the unsaved-changes
    confirmation on window close instead of silently discarding it because
    the old guard was `has_unsaved_changes and not self.saved`."""
    import tkinter as tk
    from tkinter import messagebox
    from unittest.mock import patch

    try:
        probe = tk.Tk()
        probe.withdraw()
        probe.destroy()
    except tk.TclError:
        pytest.skip("No display available for Tkinter GUI test")

    csv_path = tmp_path / "a.csv"
    _touch_csv(csv_path)

    app = ColumnReorderGUI(["frame", "a"], ["a.csv"], str(tmp_path))
    try:
        app.saved = True  # a prior save already happened this session
        app.has_unsaved_changes = True  # a later edit (e.g. rename) followed it

        with patch.object(messagebox, "askyesnocancel", return_value=None) as mock_ask:
            app.on_window_close()

        mock_ask.assert_called_once()
        assert app.winfo_exists()  # Cancel path: window must stay open
    finally:
        if app.winfo_exists():
            app.destroy()
