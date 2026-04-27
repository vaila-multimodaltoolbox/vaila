"""
Integration / functional / regression / E2E tests for ``vaila.tugturn``.

Levels covered (per the user-requested test taxonomy):

* Integration tests       — Several modules combined: dataframe load → analyzer
                            → segmentation → spatiotemporal → reports.
* Functional tests        — Validate domain-specific TUG semantics: phase
                            ordering, contiguity, gait counts, turn direction,
                            VC outputs.
* End-to-end (E2E) tests  — Run ``vaila/tugturn.py`` as a subprocess, exactly
                            the way ``uv run`` invokes it from the CLI / GUI.
* Regression tests        — Lock down numeric outputs of the reference trial
                            (``s26_m1_t1``) to detect silent algorithmic drift.

Reference (Chinaglia, Cesar & Santiago, 2026): https://arxiv.org/abs/2602.21425
Software repos:
    - vailá  : https://github.com/vaila-multimodaltoolbox/vaila
    - tugturn: https://github.com/paulopreto/tugturn_GP
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from vaila.tugturn import process_tug_file

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = REPO_ROOT / "tests" / "tugturn"
SAMPLE_CSV = TEST_DATA_DIR / "s26_m1_t1.csv"
SAMPLE_TOML = TEST_DATA_DIR / "s26_m1_t1.toml"
SAMPLE_STEM = "s26_m1_t1"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _require_sample():
    if not SAMPLE_CSV.exists() or not SAMPLE_TOML.exists():
        msg = f"Sample TUG trial not found in {TEST_DATA_DIR}"
        pytest.skip(msg)  # ty: ignore[too-many-positional-arguments]


@pytest.fixture
def sample_data(tmp_path):
    """Copy the reference TUG trial into an isolated tmp directory."""
    _require_sample()
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    csv_dest = work_dir / SAMPLE_CSV.name
    toml_dest = work_dir / SAMPLE_TOML.name
    shutil.copy(SAMPLE_CSV, csv_dest)
    shutil.copy(SAMPLE_TOML, toml_dest)
    out_dir = work_dir / "results"
    out_dir.mkdir()
    return {"work_dir": work_dir, "csv": csv_dest, "toml": toml_dest, "out_dir": out_dir}


@pytest.fixture(scope="module")
def reference_report(tmp_path_factory):
    """Run the full ``process_tug_file`` once for the whole test module so
    that regression / functional checks can share the result. Marked
    module-scope because the run is dominated by GIF rendering (~15-20 s)."""
    _require_sample()
    work_dir = tmp_path_factory.mktemp("tugturn_ref")
    csv_dest = work_dir / SAMPLE_CSV.name
    toml_dest = work_dir / SAMPLE_TOML.name
    shutil.copy(SAMPLE_CSV, csv_dest)
    shutil.copy(SAMPLE_TOML, toml_dest)
    out_dir = work_dir / "results"
    out_dir.mkdir()
    report = process_tug_file(csv_dest, out_dir, config_file=toml_dest)
    return {"report": report, "out_dir": out_dir, "csv": csv_dest, "toml": toml_dest}


# ---------------------------------------------------------------------------
# A. Integration tests — full pipeline returns the expected artifacts
# ---------------------------------------------------------------------------


def test_process_tug_file_integration(sample_data):
    """End-to-End test of the process_tug_file main function. Verifies that
    the correct output files (HTML, JSON, DB CSVs) are produced."""
    csv_path = sample_data["csv"]
    out_dir = sample_data["out_dir"]
    toml_path = sample_data["toml"]

    report = process_tug_file(csv_path, out_dir, config_file=toml_path)

    assert report is not None
    assert isinstance(report, dict)
    for key in ("Metadata", "Spatiotemporal", "Phases_Seconds", "Steps_Timeseries"):
        assert key in report

    expected_files = [
        f"{SAMPLE_STEM}_tugturn_report_interactive.html",
        f"{SAMPLE_STEM}_tugturn_report.html",
        f"{SAMPLE_STEM}_tugturn_data.json",
        f"{SAMPLE_STEM}_bd_results.csv",
        f"{SAMPLE_STEM}_bd_kinematics.csv",
        f"{SAMPLE_STEM}_bd_steps.csv",
        f"{SAMPLE_STEM}_bd_participants.csv",
        f"{SAMPLE_STEM}_bd_vector_coding.csv",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"Missing expected output: {fname}"

    db_df = pd.read_csv(out_dir / f"{SAMPLE_STEM}_bd_results.csv")
    assert not db_df.empty
    assert db_df.iloc[0]["File_ID"] == SAMPLE_STEM


def test_phase_skeleton_gifs_generated(sample_data):
    """All canonical phases that exist in the segmentation must yield a GIF
    next to the CSV outputs (one per phase)."""
    out_dir = sample_data["out_dir"]
    process_tug_file(sample_data["csv"], out_dir, config_file=sample_data["toml"])

    for phase in ("stand", "first_gait", "stop_5s", "turn180", "second_gait", "sit"):
        gif = out_dir / f"{SAMPLE_STEM}_{phase}.gif"
        assert gif.exists() and gif.stat().st_size > 0, f"Missing GIF: {gif.name}"


# ---------------------------------------------------------------------------
# B. Functional tests — domain semantics
# ---------------------------------------------------------------------------


def test_phases_are_ordered_and_non_overlapping(reference_report):
    """Phases must come in canonical order and must not overlap.

    Note: ``stand`` and ``first_gait`` may be separated by a small "ready"
    interval (subject upright but not yet stepping past ``y_chair``). All
    other adjacent phases must touch (end_a == start_b) by construction.
    """
    phases = reference_report["report"]["Phases_Seconds"]
    seq = ["stand", "first_gait", "stop_5s", "turn180", "second_gait", "sit"]
    starts_ends = [tuple(phases[k]) for k in seq]
    for s, e in starts_ends:
        assert s >= 0.0 and e >= s, f"Bad phase boundary: ({s}, {e})"
    pair_names = list(zip(seq, seq[1:], strict=False))
    pair_a = starts_ends[:-1]
    pair_b = starts_ends[1:]
    for (name_a, name_b), (s_a, e_a), (s_b, _e_b) in zip(pair_names, pair_a, pair_b, strict=True):
        assert e_a <= s_b + 1e-6, (
            f"Overlap between {name_a} ({s_a},{e_a}) and {name_b} starting at {s_b}"
        )


def test_total_time_matches_phase_span(reference_report):
    phases = reference_report["report"]["Phases_Seconds"]
    total = phases["Total_TUG_Time"]
    sit_end = phases["sit"][1]
    assert total >= sit_end
    assert total - sit_end < 1.0


def test_turn_direction_matches_toml_metadata(reference_report):
    """The TOML metadata for s26_m1_t1 says TURNSIDE = "R". The detector
    should agree (Right). This is a functional sanity check, not a strict
    requirement — a mismatch would deserve a careful review."""
    phases = reference_report["report"]["Phases_Seconds"]
    metadata = reference_report["report"]["Metadata"]
    expected = "Right" if metadata.get("TURNSIDE", "").upper() == "R" else "Left"
    assert phases.get("Turn_Direction") == expected


def test_steps_split_between_first_and_second_gait(reference_report):
    steps = reference_report["report"]["Steps_Timeseries"]
    first = [s for s in steps if s["Phase"] == "first_gait"]
    second = [s for s in steps if s["Phase"] == "second_gait"]

    assert len(first) >= 6, f"Too few first-gait steps: {len(first)}"
    assert len(second) >= 6, f"Too few second-gait steps: {len(second)}"

    first_y = [s["Y_m"] for s in first]
    assert first_y[-1] > first_y[0], "First-gait Y did not increase"

    second_y = [s["Y_m"] for s in second]
    assert second_y[-1] < second_y[0], "Second-gait Y did not decrease"


def test_alternating_step_sides(reference_report):
    """For a healthy gait the heel strikes alternate L/R most of the time.
    Allow up to 2 same-side back-to-back transitions per gait phase to keep
    the test tolerant of detector glitches at slow turn-around moments."""
    steps = reference_report["report"]["Steps_Timeseries"]
    for phase in ("first_gait", "second_gait"):
        sides = [s["Side"][0] for s in steps if s["Phase"] == phase]
        if len(sides) < 2:
            continue
        same = sum(1 for a, b in zip(sides, sides[1:], strict=False) if a == b)
        assert same <= 2, f"Too many consecutive same-side strikes in {phase}: {sides}"


def test_results_csv_has_known_columns(sample_data):
    out_dir = sample_data["out_dir"]
    process_tug_file(sample_data["csv"], out_dir, config_file=sample_data["toml"])
    df = pd.read_csv(out_dir / f"{SAMPLE_STEM}_bd_results.csv")
    expected = {
        "File_ID",
        "STS_Time_s",
        "First_Gait_Time_s",
        "Freeze_Before_Turn_s",
        "Turn_Time_s",
        "Second_Gait_Time_s",
        "Stand_to_Sit_Time_s",
        "Total_Time_s",
        "Turn_Direction",
        "Velocity_m_s",
        "Cadence",
        "R_Step_Length",
        "L_Step_Length",
        "Steps_First_Gait",
        "Steps_Second_Gait",
        "XcoM_Dev_First_Gait_m",
        "XcoM_Dev_Second_Gait_m",
        "VCTurn_Dominant",
        "VCStand_Dominant",
        "VCFirstGait_Dominant",
        "VCSecondGait_Dominant",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Missing columns in bd_results.csv: {missing}"


def test_metadata_propagates_into_report(reference_report):
    md = reference_report["report"]["Metadata"]
    assert md["SEX"] == "M"
    assert md["AGE"] == 63.0
    assert md["TURNSIDE"] == "R"
    assert md["FPS"] == pytest.approx(59.94005994005994, rel=1e-9)


# ---------------------------------------------------------------------------
# C. Regression tests — lock down reference numeric output
# ---------------------------------------------------------------------------


# Reference values captured from a known-good run on `s26_m1_t1` (commit
# preceding this test introduction). Tolerances are deliberately loose to
# absorb negligible numerical drift but tight enough to flag algorithmic
# regressions.
REGRESSION_PHASES = {
    "stand": (0.0, 1.4848, 0.05),
    "first_gait": (2.1688, 7.9746, 0.10),
    "stop_5s": (7.9746, 14.5145, 0.10),
    "turn180": (14.5145, 17.3006, 0.10),
    "second_gait": (17.3006, 23.0564, 0.10),
    "sit": (23.0564, 26.9102, 0.10),
}

REGRESSION_GLOBAL = {
    "Cadence_steps_per_min": (67.91, 1.0),
    "Velocity_m_s": (0.347, 0.05),
    "Steps_First_Gait": (12, 1),
    "Steps_Second_Gait": (11, 1),
}


def test_regression_phase_boundaries(reference_report):
    phases = reference_report["report"]["Phases_Seconds"]
    for name, (s_ref, e_ref, tol) in REGRESSION_PHASES.items():
        s, e = phases[name]
        assert abs(s - s_ref) <= tol, f"{name} start drifted: {s} vs {s_ref}"
        assert abs(e - e_ref) <= tol, f"{name} end drifted: {e} vs {e_ref}"


def test_regression_global_metrics(reference_report):
    g = reference_report["report"]["Spatiotemporal"]["Global"]
    for key, (ref, tol) in REGRESSION_GLOBAL.items():
        actual = g[key]
        assert abs(actual - ref) <= tol, f"{key} drifted: {actual} vs {ref}±{tol}"


def test_regression_total_tug_time(reference_report):
    total = reference_report["report"]["Phases_Seconds"]["Total_TUG_Time"]
    assert abs(total - 26.9269) < 0.10


def test_regression_step_lengths_per_side(reference_report):
    spt = reference_report["report"]["Spatiotemporal"]
    assert abs(spt["Right"]["Step_Length_m"] - 0.368) < 0.05
    assert abs(spt["Left"]["Step_Length_m"] - 0.321) < 0.05


def test_regression_vector_coding_summary(reference_report):
    vcs = reference_report["report"]["VC_Summary"]
    assert vcs["Axial_Turn"]["Dominant_Pattern"] in {"In_Phase", "Anti_Phase"}
    assert vcs["Limb_R_FirstGait"]["Dominant_Pattern"] == "Distal_Phase"
    assert vcs["Limb_R_SecondGait"]["Dominant_Pattern"] == "Distal_Phase"
    for k in ("Axial_Turn", "Axial_Stand", "Limb_R_FirstGait", "Limb_R_SecondGait"):
        cav = vcs[k]["CAV_deg"]
        assert 0.0 <= cav < 360.0, f"{k} CAV out of range: {cav}"


# ---------------------------------------------------------------------------
# D. Spatial-override functional test (TOML [spatial] block)
# ---------------------------------------------------------------------------


def test_toml_spatial_block_overrides_defaults(tmp_path):
    """When the TOML provides a [spatial] block, those values must reach the
    analyzer as ``_meta_y_chair`` / ``_meta_y_turn`` / ``_meta_y_tol`` and
    affect segmentation. We use overridden values, run the pipeline, and
    verify the run still produces a JSON with valid phases."""
    _require_sample()
    work = tmp_path / "wk"
    work.mkdir()
    csv_dest = work / SAMPLE_CSV.name
    shutil.copy(SAMPLE_CSV, csv_dest)

    custom_toml = work / SAMPLE_TOML.name
    base = SAMPLE_TOML.read_text(encoding="utf-8")
    custom_toml.write_text(
        base + "\n[spatial]\ny_chair = 1.10\ny_turn = 4.45\ny_tol = 0.50\n",
        encoding="utf-8",
    )

    out = work / "results"
    out.mkdir()
    report = process_tug_file(csv_dest, out, config_file=custom_toml)
    assert report is not None
    json_file = out / f"{SAMPLE_STEM}_tugturn_data.json"
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    for k in ("stand", "first_gait", "turn180", "second_gait", "sit"):
        s, e = data["Phases_Seconds"][k]
        assert e >= s


# ---------------------------------------------------------------------------
# E. End-to-End — invoke the CLI as a subprocess
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cli_end_to_end(tmp_path):
    """Reproduces the user-level invocation:

        uv run vaila/tugturn.py -i <csv> -c <toml> -o <out>

    The subprocess form catches issues that direct ``import`` tests cannot:
    argparse defaults, output-dir handling when ``-o`` is provided, and the
    ``if __name__ == "__main__":`` entry point."""
    _require_sample()
    csv_dest = tmp_path / SAMPLE_CSV.name
    toml_dest = tmp_path / SAMPLE_TOML.name
    shutil.copy(SAMPLE_CSV, csv_dest)
    shutil.copy(SAMPLE_TOML, toml_dest)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    script = REPO_ROOT / "vaila" / "tugturn.py"
    cmd = [
        sys.executable,
        str(script),
        "-i",
        str(csv_dest),
        "-c",
        str(toml_dest),
        "-o",
        str(out_dir),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=REPO_ROOT, timeout=180, check=False
    )
    assert proc.returncode == 0, (
        f"CLI exited with code {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    for fname in (
        f"{SAMPLE_STEM}_bd_results.csv",
        f"{SAMPLE_STEM}_bd_steps.csv",
        f"{SAMPLE_STEM}_tugturn_data.json",
        f"{SAMPLE_STEM}_tugturn_report.html",
    ):
        assert (out_dir / fname).exists(), f"CLI did not produce {fname}"
