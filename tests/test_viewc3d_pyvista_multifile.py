"""
================================================================================
Test: test_viewc3d_pyvista_multifile.py
================================================================================
vailá - Multimodal Toolbox
Pure helpers for viewc3d_pyvista multi-C3D CLI: path merge, CLI argv builder,
and per-file color assignment (reuses viewc3d.assign_file_color when available).
================================================================================
"""

from __future__ import annotations

from vaila.viewc3d_pyvista import (
    AVAILABLE_COLORS,
    assign_file_color,
    build_arg_parser,
    build_viewc3d_pyvista_cli,
    merge_c3d_input_paths,
)


def test_merge_c3d_input_paths_inputs_only():
    assert merge_c3d_input_paths(["a.c3d", "b.c3d"], None) == ["a.c3d", "b.c3d"]


def test_merge_c3d_input_paths_positional_only():
    assert merge_c3d_input_paths(None, "solo.c3d") == ["solo.c3d"]


def test_merge_c3d_input_paths_merge_and_dedupe():
    assert merge_c3d_input_paths(["a.c3d", "b.c3d"], "a.c3d") == ["a.c3d", "b.c3d"]
    assert merge_c3d_input_paths(["a.c3d"], "b.c3d") == ["a.c3d", "b.c3d"]


def test_merge_c3d_input_paths_empty():
    assert merge_c3d_input_paths(None, None) == []
    assert merge_c3d_input_paths([], "") == []


def test_build_viewc3d_pyvista_cli_no_paths():
    assert build_viewc3d_pyvista_cli() == ["uv", "run", "vaila/viewc3d_pyvista.py"]
    assert build_viewc3d_pyvista_cli([]) == ["uv", "run", "vaila/viewc3d_pyvista.py"]


def test_build_viewc3d_pyvista_cli_with_paths():
    cmd = build_viewc3d_pyvista_cli(["a.c3d", "b.c3d"])
    assert cmd == ["uv", "run", "vaila/viewc3d_pyvista.py", "-i", "a.c3d", "b.c3d"]


def test_argparse_i_flag_and_positional():
    parser = build_arg_parser()
    args = parser.parse_args(["-i", "a.c3d", "b.c3d"])
    paths = merge_c3d_input_paths(args.inputs, args.c3d_path)
    assert paths == ["a.c3d", "b.c3d"]

    args2 = parser.parse_args(["solo.c3d"])
    assert merge_c3d_input_paths(args2.inputs, args2.c3d_path) == ["solo.c3d"]

    args3 = parser.parse_args(["-i", "a.c3d", "extra.c3d"])
    assert merge_c3d_input_paths(args3.inputs, args3.c3d_path) == ["a.c3d", "extra.c3d"]


def test_assign_file_color_wraparound_matches_palette():
    c0, n0 = assign_file_color(0, AVAILABLE_COLORS)
    assert n0 == "Orange"
    assert c0 == AVAILABLE_COLORS[0][0]

    c1, n1 = assign_file_color(1, AVAILABLE_COLORS)
    assert n1 == "Blue"

    wrap_idx = len(AVAILABLE_COLORS)
    c_wrap, n_wrap = assign_file_color(wrap_idx, AVAILABLE_COLORS)
    assert n_wrap == n0
    assert c_wrap == c0
