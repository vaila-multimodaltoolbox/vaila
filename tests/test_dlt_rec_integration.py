import os
import shutil
import subprocess
from pathlib import Path
import pytest

@pytest.fixture
def test_data(tmp_path):
    # Setup paths
    base_dir = Path(__file__).parent.parent
    dlt3d_dir = base_dir / "tests" / "DLT3D_and_Rec3d"
    rec3d_one_dir = base_dir / "tests" / "rec3d_one_dlt3d"
    animal_dir = base_dir / "tests" / "Animal_Open_Field"
    
    # Create temp workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    # Copy relevant files for testing
    shutil.copy(animal_dir / "openfield_example1.csv", workspace / "pixels2d.csv")
    shutil.copy(animal_dir / "ref_real_openfield.ref2d", workspace / "real2d.ref2d")
    
    shutil.copy(dlt3d_dir / "pixelcorrds" / "c01_markers_1_line.csv", workspace / "pixels3d.csv")
    shutil.copy(dlt3d_dir / "ref3d_realworld" / "ref3d_realworld.ref3d", workspace / "real3d.ref3d")
    
    # DLT files for rec3d
    shutil.copy(dlt3d_dir / "dlt3d" / "c01_markers_1_line.dlt3d", workspace / "cam1.dlt3d")
    
    # Files for rec3d_one_dlt3d
    shutil.copy(rec3d_one_dir / "cam1_dlt_calib.dlt3d", workspace / "calib1.dlt3d")
    shutil.copy(rec3d_one_dir / "cam2_dlt_calib.dlt3d", workspace / "calib2.dlt3d")
    shutil.copy(rec3d_one_dir / "cam01_makerless.csv", workspace / "cam1_pix.csv")
    shutil.copy(rec3d_one_dir / "cam02_markerless.csv", workspace / "cam2_pix.csv")
    
    return workspace

def test_dlt2d_integration(test_data):
    pixel_file = str(test_data / "pixels2d.csv")
    real_file = str(test_data / "real2d.ref2d")
    
    # Use subprocess to test CLI parsing
    result = subprocess.run([
        "python3", "vaila/dlt2d.py",
        "--pixel", pixel_file,
        "--real", real_file
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    dlt_output = test_data / "pixels2d.dlt2d"
    assert dlt_output.exists()

def test_dlt3d_integration(test_data):
    pixel_file = str(test_data / "pixels3d.csv")
    real_file = str(test_data / "real3d.ref3d")
    
    result = subprocess.run([
        "python3", "vaila/dlt3d.py",
        "--pixel", pixel_file,
        "--real", real_file
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    dlt_output = test_data / "pixels3d.dlt3d"
    assert dlt_output.exists()

def test_rec2d_integration(test_data):
    # First generate the .dlt2d
    subprocess.run([
        "python3", "vaila/dlt2d.py",
        "--pixel", str(test_data / "pixels2d.csv"),
        "--real", str(test_data / "real2d.ref2d")
    ])
    
    dlt_file = str(test_data / "pixels2d.dlt2d")
    # Place pixels in a subdir for batch processing
    input_dir = test_data / "input2d"
    input_dir.mkdir()
    shutil.copy(test_data / "pixels2d.csv", input_dir / "data.csv")
    
    output_dir = test_data / "output2d"
    output_dir.mkdir()
    
    result = subprocess.run([
        "python3", "vaila/rec2d_one_dlt2d.py",
        "--dlt-file", dlt_file,
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--rate", "100"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # Check if a reconstructed folder was created
    subfolders = [f for f in output_dir.iterdir() if f.is_dir() and "vaila_rec" in f.name]
    assert len(subfolders) >= 1
    assert any(f.suffix == ".2d" for f in subfolders[0].iterdir())

def test_rec3d_integration(test_data):
    dlt_file = str(test_data / "cam1.dlt3d")
    input_dir = test_data / "input3d"
    input_dir.mkdir()
    shutil.copy(test_data / "pixels3d.csv", input_dir / "data.csv")
    
    output_dir = test_data / "output3d"
    output_dir.mkdir()
    
    result = subprocess.run([
        "python3", "vaila/rec3d.py",
        "--dlt-files", dlt_file,
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--rate", "100"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # Check if a reconstructed folder was created
    subfolders = [f for f in output_dir.iterdir() if f.is_dir() and "vaila_rec" in f.name]
    assert len(subfolders) >= 1
    assert any(f.suffix == ".3d" for f in subfolders[0].iterdir())

def test_rec3d_one_dlt3d_integration(test_data):
    # rec3d_one_dlt3d.py already had CLI arguments: --dlt3d, --pixels, --fps, --output
    dlt1 = str(test_data / "calib1.dlt3d")
    dlt2 = str(test_data / "calib2.dlt3d")
    pix1 = str(test_data / "cam1_pix.csv")
    pix2 = str(test_data / "cam2_pix.csv")
    
    output_dir = test_data / "output3d_one"
    output_dir.mkdir()
    
    result = subprocess.run([
        "python3", "vaila/rec3d_one_dlt3d.py",
        "--dlt3d", dlt1, dlt2,
        "--pixels", pix1, pix2,
        "--fps", "100",
        "--output", str(output_dir)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    # Check if reconstruction files were created
    subfolders = [f for f in output_dir.iterdir() if f.is_dir() and "vaila_rec3d" in f.name]
    assert len(subfolders) == 1
    # It creates .csv and .3d files
    files = list(subfolders[0].iterdir())
    assert any(f.suffix == ".csv" for f in files)
    assert any(f.suffix == ".3d" for f in files)
