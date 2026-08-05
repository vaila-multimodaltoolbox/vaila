"""
================================================================================
Script: generate_skeleton_jsons.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.3.99
Created: 04 August 2026
Last Updated: 05 August 2026

Description:
    Maintenance/dev tool (not wired into the vailá GUI/CLI dispatch) that
    regenerates the two model-derived skeleton-connection JSON presets in
    this directory:

      - sam3dinov3_mhr70.json    <- vaila/sam3dinov3.py's MHR70_NAMES +
                                    SKELETON_EDGE_NAMES (24 edges)
      - sapiens2_goliath308.json <- the vendored Sapiens2 checkout's
                                    sapiens/pose/configs/_base_/keypoints308.py
                                    dataset_info['keypoint_info'] (308 kp,
                                    0-indexed name order) + ['skeleton_info']
                                    (65 edges)

    Both source-of-truth definitions live elsewhere (sam3dinov3.py owns the
    MHR70 order; the Sapiens2 repo owns the 308-point order) — re-run this
    script instead of hand-editing the JSON if either upstream definition
    changes, so the two never silently drift out of sync.

    "pN" in every emitted connection is the 1-based column index of that
    keypoint in the wide CSV rec3d_one_dlt3d.py/rec3d.py write (frame,
    p1_x,p1_y,p1_z,...), i.e. keypoint index 0 (0-based) in the source
    definition becomes p1, index 1 becomes p2, etc. — the same convention
    documented in vaila/mesh_alignment.py's ALIGNMENT_MARKER_SPEC.

Usage:
    uv run python vaila/skeletons/generate_skeleton_jsons.py
    (requires bin/setup_sapiens2.sh to have been run at least once, so
    .local/third_party/sapiens2/ exists; sam3dinov3.py's constants are read
    directly from source, no import/CUDA required for either.)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


def _generate_dinov3_mhr70() -> dict:
    src = (REPO_ROOT / "vaila" / "sam3dinov3.py").read_text(encoding="utf-8")

    names_match = re.search(r"MHR70_NAMES: tuple\[str, \.\.\.\] = \((.*?)\)\n", src, re.S)
    if not names_match:
        raise RuntimeError("MHR70_NAMES not found in sam3dinov3.py")
    names = re.findall(r'"([^"]+)"', names_match.group(1))
    if len(names) != 70:
        raise RuntimeError(f"expected 70 MHR70 names, found {len(names)}")

    edges_match = re.search(
        r"SKELETON_EDGE_NAMES: tuple\[tuple\[str, str\], \.\.\.\] = \((.*?)\n\)\n", src, re.S
    )
    if not edges_match:
        raise RuntimeError("SKELETON_EDGE_NAMES not found in sam3dinov3.py")
    edge_pairs = re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', edges_match.group(1))

    name_to_pidx = {n: i + 1 for i, n in enumerate(names)}
    connections = []
    for a, b in edge_pairs:
        if a not in name_to_pidx or b not in name_to_pidx:
            raise RuntimeError(f"skeleton edge references unknown MHR70 name: {a!r}/{b!r}")
        connections.append([f"p{name_to_pidx[a]}", f"p{name_to_pidx[b]}"])

    return {
        "schema": "sam3dinov3_mhr70",
        "note": (
            "pN is 1-based index into the 70-keypoint MHR order used by "
            "sam3dinov3.py/sam3dinov3_visualize.py's *_mhr70_rec3d.csv / "
            "*_markers.csv (p1=nose, p6=left-shoulder, p7=right-shoulder, "
            "p10=left-hip, p11=right-hip, ..., p70=neck). Derived "
            "programmatically from vaila/sam3dinov3.py's MHR70_NAMES + "
            "SKELETON_EDGE_NAMES (2026-08-04) -- regenerate with "
            "generate_skeleton_jsons.py if those change, do not hand-edit."
        ),
        "connections": connections,
    }


def _generate_sapiens2_goliath308() -> dict:
    config_path = (
        REPO_ROOT
        / ".local"
        / "third_party"
        / "sapiens2"
        / "sapiens"
        / "pose"
        / "configs"
        / "_base_"
        / "keypoints308.py"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found — run bash bin/setup_sapiens2.sh first "
            "(it clones the Sapiens2 checkout this generator reads from)."
        )

    ns: dict = {}
    code = config_path.read_text(encoding="utf-8")
    exec(compile(code, str(config_path), "exec"), ns)  # noqa: S102 -- trusted vendored data file, no imports

    dataset_info = ns["dataset_info"]
    kp_info = dataset_info["keypoint_info"]
    skel_info = dataset_info["skeleton_info"]
    if len(kp_info) != 308:
        raise RuntimeError(f"expected 308 Sapiens2 keypoints, found {len(kp_info)}")

    name_to_pidx = {v["name"]: idx + 1 for idx, v in kp_info.items()}
    connections = []
    for i in sorted(skel_info.keys()):
        a, b = skel_info[i]["link"]
        if a not in name_to_pidx or b not in name_to_pidx:
            raise RuntimeError(f"skeleton_info link references unknown keypoint: {a!r}/{b!r}")
        connections.append([f"p{name_to_pidx[a]}", f"p{name_to_pidx[b]}"])

    extra_foot_pairs = [
        ["p18", "p16"],  # left_heel - left_big_toe
        ["p18", "p17"],  # left_heel - left_small_toe
        ["p16", "p17"],  # left_big_toe - left_small_toe
        ["p21", "p19"],  # right_heel - right_big_toe
        ["p21", "p20"],  # right_heel - right_small_toe
        ["p19", "p20"],  # right_big_toe - right_small_toe
    ]
    for pair in extra_foot_pairs:
        if pair not in connections and [pair[1], pair[0]] not in connections:
            connections.append(pair)

    return {
        "schema": "sapiens2_goliath_308",
        "note": (
            "pN is 1-based index into the 308-keypoint Sociopticon/Goliath "
            "order used by vaila_sapiens.py/sam3sapiens2.py (p1=nose, "
            "p6=left_shoulder, p7=right_shoulder, p10=left_hip, "
            "p11=right_hip, ..., p308=last face iris-border point). Derived "
            "programmatically from the vendored "
            "sapiens2/sapiens/pose/configs/_base_/keypoints308.py "
            "dataset_info['keypoint_info']/['skeleton_info'] (2026-08-04) -- "
            "regenerate with generate_skeleton_jsons.py if Sapiens2 changes, "
            "do not hand-edit."
        ),
        "connections": connections,
    }


def _write_compact(payload: dict, out_path: Path) -> None:
    """Write with one ["pA","pB"] pair per line — matches the existing hand-
    written skeleton_pose_*.json fixtures' style, easier to skim/diff than
    json.dumps' default one-token-per-line nested-list formatting."""
    lines = ["{"]
    lines.append(f'  "schema": {json.dumps(payload["schema"])},')
    lines.append(f'  "note": {json.dumps(payload["note"])},')
    lines.append('  "connections": [')
    conn_lines = [f"    {json.dumps(pair)}" for pair in payload["connections"]]
    lines.append(",\n".join(conn_lines))
    lines.append("  ]")
    lines.append("}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    dinov3 = _generate_dinov3_mhr70()
    out_path = OUT_DIR / "sam3dinov3_mhr70.json"
    _write_compact(dinov3, out_path)
    print(f"wrote {out_path} ({len(dinov3['connections'])} connections)")

    try:
        sapiens2 = _generate_sapiens2_goliath308()
    except FileNotFoundError as e:
        print(f"skipped sapiens2_goliath308.json: {e}")
        return
    out_path = OUT_DIR / "sapiens2_goliath308.json"
    _write_compact(sapiens2, out_path)
    print(f"wrote {out_path} ({len(sapiens2['connections'])} connections)")


if __name__ == "__main__":
    main()
