"""Tests for the recursive depth-limited file scan added to
vaila/edit_csv_c3d.py (find_edit_csv_c3d_files / _headless_process max_depth)
and the bulk column rename/renumber helpers added to
vaila/rearrange_data.py (_build_renumber_map / _validate_rename_map).

Update Date: 23 September 2026
Version: 0.4.5
"""

import pandas as pd
import pytest

try:
    from vaila.edit_csv_c3d import _headless_process, find_edit_csv_c3d_files
    from vaila.rearrange_data import _build_renumber_map, _validate_rename_map
except ImportError:  # standalone execution
    from edit_csv_c3d import (  # ty: ignore[unresolved-import]
        _headless_process,
        find_edit_csv_c3d_files,
    )
    from rearrange_data import (  # ty: ignore[unresolved-import]
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
