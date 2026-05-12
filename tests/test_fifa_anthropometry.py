from __future__ import annotations

from pathlib import Path

import numpy as np

from vaila.fifa_anthropometry import (
    AnthropometricProfile,
    load_sequence_profiles,
    resolve_sequence_csv,
    scale_skeleton_to_profile,
)
from vaila.fifa_skeletal_pipeline import build_fifa_argparser


def test_resolve_sequence_csv_matches_portuguese_team_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "argentina_holanda_full.csv"
    csv_path.write_text(
        "\n".join(
            [
                "nome,altura_m,numero,pais",
                "A,1.90,1,Argentina",
                "B,1.80,2,Holanda",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resolved = resolve_sequence_csv("NET_ARG_003203", tmp_path)
    assert resolved == csv_path


def test_load_sequence_profiles_assigns_tallest_player_to_tallest_slot(tmp_path: Path) -> None:
    csv_path = tmp_path / "croacia_marrocos_full.csv"
    csv_path.write_text(
        "\n".join(
            [
                "nome,altura_m,coxa_m,perna_m,tronco_m,braco_m,antebraco_m,numero,pais",
                "Tall,1.95,0.50,0.50,0.60,0.35,0.30,1,Croacia",
                "Mid,1.82,0.45,0.46,0.55,0.33,0.27,2,Croacia",
                "Short,1.70,0.41,0.43,0.50,0.30,0.25,3,Marrocos",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    boxes = np.array(
        [
            [[0, 0, 20, 80], [0, 0, 20, 120], [0, 0, 20, 60]],
            [[0, 0, 20, 78], [0, 0, 20, 118], [0, 0, 20, 61]],
            [[0, 0, 20, 79], [0, 0, 20, 119], [0, 0, 20, 59]],
        ],
        dtype=np.float32,
    )

    profiles, resolved = load_sequence_profiles("CRO_MOR_190500", boxes, tmp_path)

    assert resolved == csv_path
    assert profiles is not None
    assert profiles[1] is not None and profiles[1].player_name == "Tall"
    assert profiles[0] is not None and profiles[0].player_name == "Mid"
    assert profiles[2] is not None and profiles[2].player_name == "Short"


def test_scale_skeleton_to_profile_scales_uniformly_about_hip_center() -> None:
    skel = np.full((15, 3), np.nan, dtype=np.float32)
    skel[1] = [0.0, 1.0, 0.0]
    skel[2] = [1.0, 1.0, 0.0]
    skel[3] = [0.0, 2.0, 0.0]
    skel[4] = [1.0, 2.0, 0.0]
    skel[5] = [0.0, 3.0, 0.0]
    skel[6] = [1.0, 3.0, 0.0]
    skel[7] = [0.0, 0.0, 0.0]
    skel[8] = [1.0, 0.0, 0.0]
    skel[9] = [0.0, -1.0, 0.0]
    skel[10] = [1.0, -1.0, 0.0]
    skel[11] = [0.0, -2.0, 0.0]
    skel[12] = [1.0, -2.0, 0.0]

    profile = AnthropometricProfile(
        slot_index=0,
        player_name="Scaled",
        team="Croacia",
        number="1",
        source_csv=Path("/tmp/croacia_marrocos_full.csv"),
        thigh_m=2.0,
        shank_m=2.0,
        trunk_m=2.0,
        upper_arm_m=2.0,
        forearm_m=2.0,
    )

    scaled, scale = scale_skeleton_to_profile(skel, profile, min_scale=0.5, max_scale=2.5)

    assert np.isclose(scale, 2.0)
    assert np.allclose((scaled[7] + scaled[8]) / 2.0, np.array([0.5, 0.0, 0.0]))
    assert np.isclose(np.linalg.norm(scaled[7] - scaled[9]), 2.0)
    assert np.isclose(np.linalg.norm(scaled[9] - scaled[11]), 2.0)
    assert np.isclose(np.linalg.norm(scaled[1] - scaled[3]), 2.0)


def test_build_fifa_argparser_accepts_anthropometry_flags(tmp_path: Path) -> None:
    parser = build_fifa_argparser()
    args = parser.parse_args(
        [
            "baseline",
            "--data-root",
            str(tmp_path / "data"),
            "--sequences",
            str(tmp_path / "seq.txt"),
            "--output",
            str(tmp_path / "out.npz"),
            "--anthropometrics-dir",
            str(tmp_path / "anthro"),
            "--anthropometric-min-scale",
            "0.8",
            "--anthropometric-max-scale",
            "1.2",
        ]
    )

    assert args.anthropometrics_dir == tmp_path / "anthro"
    assert float(args.anthropometric_min_scale) == 0.8
    assert float(args.anthropometric_max_scale) == 1.2
