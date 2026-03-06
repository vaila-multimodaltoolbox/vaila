import shutil
from pathlib import Path

import pytest

from vaila.tugturn import process_tug_file

TEST_DATA_DIR = Path(__file__).parent / "tugturn"


@pytest.fixture
def sample_data(tmp_path):
    """
    Sets up a temporary directory with copies of the sample data for testing.
    """
    if not TEST_DATA_DIR.exists():
        pytest.skip(f"Test data directory {TEST_DATA_DIR} not found.")

    csv_source = TEST_DATA_DIR / "s26_m1_t1.csv"
    toml_source = TEST_DATA_DIR / "s26_m1_t1.toml"

    if not csv_source.exists() or not toml_source.exists():
        pytest.skip("Required sample data files (s26_m1_t1.csv, s26_m1_t1.toml) not found.")

    work_dir = tmp_path / "work"
    work_dir.mkdir()

    csv_dest = work_dir / csv_source.name
    toml_dest = work_dir / toml_source.name

    shutil.copy(csv_source, csv_dest)
    shutil.copy(toml_source, toml_dest)

    out_dir = work_dir / "results"
    out_dir.mkdir()

    return {"work_dir": work_dir, "csv": csv_dest, "toml": toml_dest, "out_dir": out_dir}


def test_process_tug_file_integration(sample_data):
    """
    End-to-End test of the process_tug_file main function.
    Verifies that the correct output files (HTML, JSON, DB CSVs) are produced.
    """
    csv_path = sample_data["csv"]
    out_dir = sample_data["out_dir"]
    toml_path = sample_data["toml"]

    # Run the core processing function directly
    report = process_tug_file(csv_path, out_dir, config_file=toml_path)

    # Verify the report object is returned and populated
    assert report is not None
    assert isinstance(report, dict)
    assert "Metadata" in report
    assert "Spatiotemporal" in report

    # Check that output files were generated
    matplotlib_html = out_dir / "s26_m1_t1_tugturn_report_interactive.html"
    plotly_html = out_dir / "s26_m1_t1_tugturn_report.html"
    json_export = out_dir / "s26_m1_t1_tugturn_data.json"
    results_csv = out_dir / "s26_m1_t1_bd_results.csv"
    kinematics_csv = out_dir / "s26_m1_t1_bd_kinematics.csv"

    # Asserting all required outputs exist
    assert matplotlib_html.exists(), f"Matplotlib HTML missing: {matplotlib_html}"
    assert plotly_html.exists(), f"Plotly HTML missing: {plotly_html}"
    assert json_export.exists(), f"JSON export missing: {json_export}"
    assert results_csv.exists(), f"Results CSV missing: {results_csv}"
    assert kinematics_csv.exists(), f"Kinematics CSV missing: {kinematics_csv}"

    # Verify that the generated database has content
    import pandas as pd

    db_df = pd.read_csv(results_csv)
    assert not db_df.empty
    assert "File_ID" in db_df.columns
    assert db_df.iloc[0]["File_ID"] == "s26_m1_t1"
