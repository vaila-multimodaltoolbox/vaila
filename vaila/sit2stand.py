"""
================================================================================
sit2stand.py - Sit to Stand Analysis Module
================================================================================
Author: Prof. Paulo Santiago
Create: 10 October 2025
Update: 03 February 2026
Version: 0.0.7

Description:
------------
This module provides comprehensive functionality for analyzing sit-to-stand movements using force plate data.
It supports batch processing of C3D and CSV files with TOML configuration files and Butterworth filtering.

Key Features:
-------------
1. Batch Processing: Analyze multiple files simultaneously
2. File Format Support: C3D (with ezc3d library) and CSV file formats
3. TOML Configuration: All parameters can be defined in TOML configuration files
4. Butterworth Filtering: Configurable low-pass filtering with user-defined parameters
5. Column Selection: Interactive column selection with detailed file information
6. C3D File Analysis: Full support for C3D files with analog channel extraction
7. Force Plate Data Analysis: Focus on vertical force (Fz) data for sit-to-stand detection
8. Advanced Peak Detection: Uses scipy.signal.find_peaks with configurable parameters
9. Stability Analysis: Index of stability measuring deviation from horizontal baseline
10. Time Vector Generation: Configurable FPS for proper time axis generation
11. Energy Expenditure Analysis: Calculates mechanical work and metabolic energy based on body weight

Analysis Capabilities:
----------------------
- Sit-to-stand phase detection with configurable thresholds
- Force impulse calculation with filtered data
- Peak force identification and timing analysis using scipy.signal.find_peaks
- Movement timing analysis with onset detection
- Balance assessment during transitions
- Butterworth low-pass filtering for noise reduction
- Stability index calculation measuring deviation from horizontal baseline
- Noise and oscillation analysis during standing phase
- Configurable FPS for proper time vector generation

Configuration:
--------------
Parameters can be configured via:
1. TOML configuration files (recommended for reproducibility)
2. Interactive GUI dialogs (for quick testing)

TOML Configuration File Format:
-------------------------------
[analysis]
# Column containing vertical force data
force_column = "Force.Fz3"
fps = 2000.0  # Frames per second for time vector generation

[filtering]
# Butterworth filter parameters
enabled = false
cutoff_frequency = 100.0  # Hz
sampling_frequency = 2000.0  # Hz
order = 4

[detection]
# Sit-to-stand detection parameters
force_threshold = 10.0  # N
min_duration = 0.5  # seconds
onset_threshold = 5.0  # N above baseline

[detection.peak_detection]
# scipy.signal.find_peaks parameters
height = 139.0  # Minimum height of peaks (N) - adjust based on baseline
distance = 200  # Minimum distance between peaks (samples) - 100ms at 2000Hz
prominence = 10.0  # Minimum prominence of peaks (N)
rel_height = 0.5  # Relative height for width calculation

[stability]
# Stability analysis parameters
enabled = true
baseline_window = 0.5  # Seconds after first peak to consider as baseline
stability_threshold = 2.0  # Maximum deviation for stable standing (N)
noise_analysis = true  # Enable noise/oscillation analysis
rolling_window = 0.1  # Rolling window for noise analysis (seconds)

Usage:
------
This module is called by forceplate_analysis.py and provides a GUI for:
1. Loading TOML configuration files or using interactive setup
2. Selecting multiple C3D/CSV files for batch processing
3. Choosing the column containing vertical force data (if not in TOML)
4. Configuring analysis parameters via GUI or TOML
5. Running the analysis with Butterworth filtering and exporting results

Dependencies:
-------------
- tkinter: For GUI components
- pandas: For data manipulation
- numpy: For numerical computations
- scipy: For Butterworth filtering
- toml: For TOML configuration file parsing
- c3d: For C3D file reading (if available)

License:
--------
This module is part of the VAILA toolbox and follows the same MIT License.
================================================================================
"""

import os
import sys
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Set UTF-8 encoding for stdout/stderr to avoid encoding errors
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7 - set environment variable instead
        os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
from rich import print

try:
    import toml

    TOML_SUPPORT = True
except ImportError:
    try:
        import tomli as toml  # pyright: ignore[reportMissingImports]  # Fallback for older Python

        TOML_SUPPORT = True
    except ImportError:
        try:
            import tomllib as toml  # Python 3.11+ built-in

            TOML_SUPPORT = True
        except ImportError:
            TOML_SUPPORT = False
            print("Warning: No TOML library available. Install tomli: pip install tomli")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Try to import ezc3d for C3D file support
try:
    from ezc3d import c3d

    C3D_SUPPORT = True
except ImportError:
    C3D_SUPPORT = False
    print("Warning: ezc3d not found. C3D file support will be limited.")

# Import CoP calculation logic
try:
    from vaila.cop_calculate import calc_cop
    COP_SUPPORT = True
except ImportError:
    COP_SUPPORT = False
    print("Warning: specific vaila modules not found. CoP calcs may fail.")

# Optional: ellipse.py for PCA-based confidence ellipse and variance ratios
try:
    from vaila.ellipse import plot_ellipse_pca
    ELLIPSE_SUPPORT = True
except ImportError:
    ELLIPSE_SUPPORT = False


def main(cli_args=None):
    """
    Main function to run the sit-to-stand analysis in batch mode.
    Processes multiple C3D and CSV files using the same TOML configuration.

    Parameters:
    -----------
    cli_args : list, optional
        Command line arguments for CLI mode. If None, runs in GUI mode.
    """
    print("Starting Sit-to-Stand Analysis...")
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")

    # Parse command line arguments
    import sys

    if cli_args is None:
        cli_args = sys.argv[1:]

    # Check if we have CLI arguments
    if len(cli_args) >= 2:
        return run_cli_mode(cli_args)

    # Otherwise, run in GUI mode
    return run_gui_mode()


def run_cli_mode(cli_args):
    """
    Runs the analysis in CLI mode with command line arguments for batch processing.
    Uses the same TOML configuration for all files.

    Parameters:
    -----------
    cli_args : list
        Command line arguments [config_file, input_directory, output_directory, file_format]
    """
    if len(cli_args) < 2:
        print(
            "Usage: python sit2stand.py <config.toml> <input_directory> [output_directory]"
        )
        print("Example: python sit2stand.py config.toml /path/to/files /path/to/output")
        print("Note: Automatically processes all .c3d and .csv files in the input directory")
        return

    config_file = cli_args[0]
    input_dir = cli_args[1]
    output_dir = cli_args[2] if len(cli_args) > 2 else None
    # Always process both C3D and CSV files automatically
    file_format = "auto"

    try:
        # Handle empty config file (use defaults)
        if not config_file or config_file == "":
            config = get_default_config()
            print("Using default configuration")
        else:
            # Step 1: Load TOML configuration (same for all files)
            config = load_toml_config(config_file)
            if not config:
                print(f"Failed to load configuration from {config_file}")
                return

        # Step 2: Set up output directory
        if not output_dir:
            output_dir = os.path.join(input_dir, "sit2stand_analysis")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Step 3: Find all C3D and CSV files automatically
        files = find_analysis_files(input_dir, file_format)
        if not files:
            print(f"No .c3d or .csv files found in {input_dir}")
            return

        print(f"Found {len(files)} files to process:")
        for file_path in files:
            print(f"  - {os.path.basename(file_path)}")

        # Step 4: Run batch analysis with same config for all files
        results = run_batch_analysis(files, config, output_dir)

        # Step 5: Generate comprehensive report
        generate_batch_report(results, config, output_dir)

        print("Sit-to-Stand batch analysis completed successfully!")

    except Exception as e:
        print(f"Error in batch analysis: {e}")
        import traceback

        traceback.print_exc()


def get_default_config():
    """Returns default configuration for sit-to-stand analysis."""
    return {
        "analysis": {
            "force_column": "Force.Fz3",
            "fps": 2000.0,  # Frames per second for time vector generation
            "body_weight": 70.0,  # Body weight in kg for energy calculations
        },
        "force_plate": {
             "width_mm": 464.0,  # X dimension
             "height_mm": 508.0, # Y dimension
             "moment_unit": "N.mm" # Default unit for C3D moments
        },
        "filtering": {
            "enabled": False,
            "cutoff_frequency": 100.0,
            "sampling_frequency": 2000.0,
            "order": 4,
        },
        "detection": {
            "force_threshold": 1.0,  # Very low threshold for better detection
            "min_duration": 0.05,  # Very short minimum duration
            "onset_threshold": 1.0,  # Very low onset threshold
            "peak_detection": {
                "height": 139.0,  # Minimum height of peaks (N) - adjust based on baseline
                "distance": 200,  # Minimum distance between peaks (samples) - 100ms at 2000Hz
                "prominence": 10.0,  # Minimum prominence of peaks (N)
                "rel_height": 0.5,  # Relative height for width calculation
            },
        },
        "stability": {
            "enabled": True,
            "baseline_window": 0.5,  # Seconds after first peak to consider as baseline
            "stability_threshold": 10.0,  # Maximum deviation for stable standing (N) - adjusted for higher force values
            "noise_analysis": True,  # Enable noise/oscillation analysis
            "rolling_window": 0.1,  # Rolling window for noise analysis (seconds)
        },
    }


def run_gui_mode():
    """
    Runs the analysis in GUI mode for interactive file and configuration selection.
    """
    try:
        # Create GUI for file selection and configuration
        gui = SitToStandGUI()
        gui.run()

    except Exception as e:
        print(f"Error in GUI mode: {e}")
        import traceback

        traceback.print_exc()


def load_toml_config(config_file):
    """
    Loads TOML configuration file with support for multiple TOML libraries.

    Parameters:
    -----------
    config_file : str
        Path to TOML configuration file

    Returns:
    --------
    dict or None
        Configuration dictionary or None if failed
    """
    try:
        if not TOML_SUPPORT:
            print("TOML support not available. Install toml or tomli library.")
            return None

        if not os.path.exists(config_file):
            print(f"Configuration file not found: {config_file}")
            return None

        # Load TOML file - handle different library APIs and encoding
        try:
            # Try toml library (most common)
            with open(config_file, encoding="utf-8") as f:
                config = toml.load(f)
        except AttributeError:
            # Try tomllib (Python 3.11+ built-in, binary mode)
            with open(config_file, "rb") as f:
                config = toml.load(f)
        except UnicodeDecodeError:
            # Fallback: try with latin-1 encoding
            with open(config_file, encoding="latin-1") as f:
                config = toml.load(f)

        print(f"Loaded configuration from: {config_file}")

        # Validate required sections
        required_sections = ["analysis", "filtering", "detection"]
        for section in required_sections:
            if section not in config:
                print(f"Warning: Missing '{section}' section in configuration")
                config[section] = {}

        return config

    except Exception as e:
        print(f"Error loading TOML configuration: {e}")
        import traceback

        traceback.print_exc()
        return None


def find_analysis_files(input_dir, file_format="auto"):
    """
    Finds all C3D and CSV files in the input directory automatically.
    Processes both formats in batch.

    Parameters:
    -----------
    input_dir : str
        Input directory path
    file_format : str
        Ignored - always processes both .c3d and .csv files

    Returns:
    --------
    list
        List of file paths to process
    """
    import glob

    files = []

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return files

    # Always search for both C3D and CSV files
    extensions = [".c3d", ".csv"]

    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        found_files = glob.glob(pattern)
        files.extend(found_files)
        print(f"Found {len(found_files)} {ext} files")

    # Sort files for consistent processing
    files.sort()

    return files


# select_files function removed - now using find_analysis_files for batch processing


def select_or_confirm_column(sample_file, config):
    """
    Selects or confirms the column for analysis based on configuration.

    Parameters:
    -----------
    sample_file : str
        Path to a sample file to analyze column structure
    config : dict
        Configuration dictionary that may contain column information

    Returns:
    --------
    str or None
        Selected column name or None if cancelled
    """
    # Check if column is already specified in config
    if config.get("analysis", {}).get("force_column"):
        column_from_config = config["analysis"]["force_column"]
        # Verify column exists in file
        if verify_column_exists(sample_file, column_from_config):
            print(f"Using column from configuration: {column_from_config}")
            return column_from_config
        else:
            print(f"Column '{column_from_config}' from configuration not found in file")
            print("Attempting auto-detection...")

    # Try auto-detection
    return auto_detect_force_column(sample_file)


# select_column_interactive function removed - not needed in batch mode


def auto_detect_force_column(sample_file):
    """
    Automatically detects the most appropriate force column in a CSV file.

    Parameters:
    -----------
    sample_file : str
        Path to the CSV file to analyze

    Returns:
    --------
    str or None
        Name of the detected force column or None if not found
    """
    try:
        # Read CSV file header to check available columns
        try:
            df_header = pd.read_csv(sample_file, nrows=0, encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 if UTF-8 fails
            df_header = pd.read_csv(sample_file, nrows=0, encoding="latin-1")
        columns = list(df_header.columns)

        print(f"Auto-detecting force column from: {columns}")

        # Priority order for column detection
        force_patterns = [
            # Vertical force columns (most common for sit-to-stand)
            "Force.Fz1",
            "Force.Fz2",
            "Force.Fz3",
            "Force.Fz4",  # Force plates
            "Fz",
            "FZ",
            "Force_Z",
            "Vertical_Force",  # Generic names
            "Force.Fz",
            "force_z",  # Alternative patterns
            # If no vertical force found, try any force column
            "Force.Fx1",
            "Force.Fy1",
            "Force.Fx2",
            "Force.Fy2",
            "Fx",
            "Fy",
            "Force_X",
            "Force_Y",
        ]

        for pattern in force_patterns:
            for col in columns:
                if pattern.lower() in col.lower():
                    print(f"Auto-detected force column: {col}")
                    return col

        # If no standard pattern found, suggest the first numeric column that's not Time
        for col in columns:
            if col.lower() not in ["time", "timestamp", "frame", "t"]:
                try:
                    # Try to read a small sample to see if it's numeric
                    sample_data = pd.read_csv(sample_file, usecols=[col], nrows=10)
                    if pd.api.types.is_numeric_dtype(sample_data[col]):
                        print(f"Auto-detected numeric column: {col}")
                        return col
                except Exception:
                    continue

        print("No suitable force column found")
        return None

    except Exception as e:
        print(f"Error in auto-detection: {e}")
        return None


def verify_column_exists(sample_file, column_name):
    """
    Verifies if a column exists in the given file.

    Parameters:
    -----------
    sample_file : str
        Path to the file to check
    column_name : str
        Name of the column to verify

    Returns:
    --------
    bool
        True if column exists, False otherwise
    """
    try:
        if sample_file.lower().endswith(".c3d"):
            # For C3D files, we can't easily verify without proper library
            # Assume it exists if it's in our standard list
            standard_columns = ["Fz", "Force_Z", "Vertical_Force", "FZ"]
            return column_name in standard_columns
        else:
            # Check CSV file headers
            df = pd.read_csv(sample_file, nrows=0)
            return column_name in df.columns
    except Exception:
        return False


def butterworth_filter(data, fs, cutoff, order=4):
    """
    Applies Butterworth low-pass filter to the data.

    Parameters:
    -----------
    data : array-like
        Input data to filter
    fs : float
        Sampling frequency in Hz
    cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order (default: 4)

    Returns:
    --------
    array-like
        Filtered data
    """
    try:
        # Normalize the frequency
        nyquist = fs / 2
        normal_cutoff = cutoff / nyquist

        # Design the Butterworth filter
        b, a = butter(order, normal_cutoff, btype="low", analog=False)

        # Apply the filter
        filtered_data = filtfilt(b, a, data)

        return filtered_data

    except Exception as e:
        print(f"Error applying Butterworth filter: {str(e)}")
        return data  # Return original data if filtering fails


# configure_filtering_parameters and configure_detection_parameters functions removed - not needed in batch mode


def process_single_file(file_path, config, output_dir):
    """
    Processes a single file for sit-to-stand analysis.

    Parameters:
    -----------
    file_path : str
        Path to the file to analyze
    config : dict
        Complete configuration dictionary
    output_dir : str
        Output directory for results

    Returns:
    --------
    dict
        Analysis result for the file
    """
    try:
        print(f"Processing file: {os.path.basename(file_path)}")

        # Auto-detect or use configured column
        column_name = config.get("analysis", {}).get("force_column", "Force.Fz3")
        
        # For C3D files, column detection is handled in read_c3d_file
        # For CSV files, verify and auto-detect if needed
        if file_path.lower().endswith(".csv"):
            if not column_name or not verify_column_exists(file_path, column_name):
                print(f"Column '{column_name}' not found, attempting auto-detection...")
                column_name = select_or_confirm_column(file_path, config)
                if not column_name:
                    return {
                        "file": file_path,
                        "filename": os.path.basename(file_path),
                        "error": "Could not determine force column",
                    }
        # For C3D files, column_name will be handled/validated in read_c3d_file

        # Read file data
        if file_path.lower().endswith(".c3d"):
            data = read_c3d_file(file_path, column_name)
        else:
            data = read_csv_file(file_path, column_name, config)

        if data is None:
            print(f"  Skipping {file_path} - could not read data")
            return {
                "file": file_path,
                "filename": os.path.basename(file_path),
                "error": "Could not read data",
            }

        # Apply Butterworth filtering if enabled
        if config.get("filtering", {}).get("enabled", False):
            print("  Applying Butterworth filter...")
            filtered_force = butterworth_filter(
                data["Force"].values,
                fs=config["filtering"]["sampling_frequency"],
                cutoff=config["filtering"]["cutoff_frequency"],
                order=config["filtering"]["order"],
            )
            data = data.copy()
            data["Force"] = filtered_force
            print(f"  Filtered with cutoff {config['filtering']['cutoff_frequency']} Hz")

        # Analyze sit-to-stand movement
        analysis_result = analyze_sit_to_stand(data, config)

        # Create individual file directory inside sit2stand_results
        file_base_name = Path(file_path).stem
        main_results_dir = Path(output_dir) / "sit2stand_results"
        file_results_dir = main_results_dir / file_base_name
        file_results_dir.mkdir(parents=True, exist_ok=True)

        # Save force plot as PNG in the individual file directory
        plot_filename = f"{file_base_name}_force_plot.png"
        plot_path = file_results_dir / plot_filename
        save_force_plot_png(
            data,
            analysis_result["sit_to_stand_phases"],
            str(plot_path),
            config,
            analysis_result.get("stability_metrics", {}),
            analysis_result.get("all_peaks_global", []),
        )

        # Store results with file information
        result = {
            "file": file_path,
            "filename": os.path.basename(file_path),
            "analysis": analysis_result,
            "configuration": config.copy(),
            "plot_path": str(plot_path),
            "results_dir": str(file_results_dir),
        }

        # Generate individual report
        generate_individual_report(result, config, output_dir)

        # Generate interactive HTML report
        html_path = file_results_dir / f"{file_base_name}_interactive_report.html"
        generate_interactive_html_report(data, analysis_result, config, str(html_path), result)

        print(f"  [OK] Completed: {os.path.basename(file_path)}")
        return result

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "file": file_path,
            "filename": os.path.basename(file_path),
            "error": str(e),
        }


def run_batch_analysis(files, config, output_dir):
    """
    Runs the sit-to-stand analysis on multiple files using the same TOML configuration.
    All files are processed with identical analysis parameters.

    Parameters:
    -----------
    files : list
        List of file paths to analyze
    config : dict
        Complete configuration dictionary (same for all files)
    output_dir : str
        Output directory for results

    Returns:
    --------
    list
        List of analysis results for each file
    """
    results = []

    print(f"\nUsing the same TOML configuration for all {len(files)} files:")
    print(f"  Force column: {config.get('analysis', {}).get('force_column', 'Force.Fz3')}")
    print(f"  FPS: {config.get('analysis', {}).get('fps', 2000.0)}")
    print(f"  Body weight: {config.get('analysis', {}).get('body_weight', 70.0)} kg")
    print(f"  Filter enabled: {config.get('filtering', {}).get('enabled', False)}")
    print()

    for i, file_path in enumerate(files):
        try:
            print(f"Processing file {i + 1}/{len(files)}: {os.path.basename(file_path)}")
            
            # Use process_single_file to maintain consistency
            # All files use the same config
            result = process_single_file(file_path, config, output_dir)
            
            if result:
                results.append(result)
            else:
                results.append({
                    "file": file_path,
                    "filename": os.path.basename(file_path),
                    "error": "Processing returned no result",
                })

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append(
                {
                    "file": file_path,
                    "filename": os.path.basename(file_path),
                    "error": str(e),
                }
            )

    return results


def generate_individual_report(result, config, output_dir):
    """
    Generates individual report for a single file analysis.

    Parameters:
    -----------
    result : dict
        Analysis result for a single file
    config : dict
        Configuration used for analysis
    output_dir : str
        Output directory path
    """
    try:
        filename = result["filename"]
        base_name = Path(filename).stem
        analysis = result["analysis"]

        # Use the results directory from the result dict (created in process_single_file)
        if "results_dir" in result:
            file_results_dir = Path(result["results_dir"])
        else:
            # Fallback: create individual file directory path
            file_results_dir = Path(output_dir) / base_name
        
        # Ensure directory exists
        file_results_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual text report
        txt_path = file_results_dir / f"{base_name}_analysis_report.txt"

        with open(txt_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(f"Sit-to-Stand Analysis Report: {filename}\n")
            f.write("=" * 60 + "\n\n")

            # Configuration summary
            f.write("Configuration Used:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Force Column: {config['analysis']['force_column']}\n")
            if "body_weight" in config["analysis"]:
                f.write(f"Body Weight: {config['analysis']['body_weight']} kg\n")
            if "fps" in config["analysis"]:
                f.write(f"FPS: {config['analysis']['fps']} Hz\n")
            f.write(
                f"Butterworth Filter: {'Enabled' if config['filtering']['enabled'] else 'Disabled'}\n"
            )

            if config["filtering"]["enabled"]:
                f.write(f"  Cutoff Frequency: {config['filtering']['cutoff_frequency']} Hz\n")
                f.write(f"  Sampling Frequency: {config['filtering']['sampling_frequency']} Hz\n")
                f.write(f"  Filter Order: {config['filtering']['order']}\n")

            f.write("Detection Parameters:\n")
            f.write(f"  Force Threshold: {config['detection']['force_threshold']} N\n")
            f.write(f"  Min Duration: {config['detection']['min_duration']} s\n")
            f.write(f"  Onset Threshold: {config['detection']['onset_threshold']} N\n")
            f.write("\n" + "=" * 60 + "\n\n")

            # Basic metrics
            f.write("BASIC FORCE METRICS:\n")
            f.write(f"  Duration: {analysis['duration']:.2f} s\n")
            f.write(f"  Mean Force: {analysis['mean_force']:.2f} N\n")
            f.write(f"  Max Force: {analysis['max_force']:.2f} N\n")
            f.write(f"  Min Force: {analysis['min_force']:.2f} N\n\n")

            # Movement detection
            movement = analysis["movement_metrics"]
            f.write("MOVEMENT DETECTION:\n")
            f.write(f"  Phases Detected: {movement['num_phases']}\n")
            f.write(f"  Total Movement Time: {movement['total_movement_time']:.2f} s\n")
            f.write(f"  Average Phase Duration: {movement['average_phase_duration']:.2f} s\n")
            if movement["phases_per_minute"] > 0:
                f.write(f"  Phases per Minute: {movement['phases_per_minute']:.1f}\n\n")

            # Energy expenditure analysis
            energy = analysis.get("energy_metrics", {})
            if energy:
                f.write("ENERGY EXPENDITURE ANALYSIS:\n")
                f.write(
                    f"  Body Weight: {energy.get('body_weight_kg', 0):.1f} kg ({energy.get('body_weight_N', 0):.1f} N)\n"
                )
                f.write(f"  Total Movements: {energy.get('total_movements', 0)}\n")
                f.write(
                    f"  Total Mechanical Work: {energy.get('total_mechanical_work_J', 0):.2f} J ({energy.get('total_mechanical_work_J', 0) / 4184:.3f} kcal)\n"
                )
                f.write(
                    f"  Total Metabolic Energy: {energy.get('total_metabolic_energy_kcal', 0):.3f} kcal\n"
                )
                f.write(
                    f"  Average Energy per Movement: {energy.get('average_energy_per_movement_kcal', 0):.3f} kcal\n"
                )
                f.write(f"  Energy Efficiency: {energy.get('energy_efficiency', 0):.1f}%\n\n")

            # Stability analysis
            stability = analysis.get("stability_metrics", {})
            if stability:
                f.write("STABILITY ANALYSIS (Oscillation around Reference Peak):\n")
                f.write(f"  Reference Peak Force: {stability.get('first_peak_force', 0):.2f} N\n")
                f.write(
                    f"  Stability Index: {stability.get('stability_index', 0):.3f} (0-1 scale)\n"
                )
                f.write(
                    f"  Mean Deviation from Reference Peak: {stability.get('mean_deviation', 0):.2f} N\n"
                )
                f.write(
                    f"  Max Deviation from Reference Peak: {stability.get('max_deviation', 0):.2f} N\n"
                )
                f.write(
                    f"  Points Above Reference Peak: {stability.get('points_above', 0)} ({stability.get('percent_above', 0):.1f}%)\n"
                )
                f.write(
                    f"  Points Below Reference Peak: {stability.get('points_below', 0)} ({stability.get('percent_below', 0):.1f}%)\n"
                )
                f.write(
                    f"  Total Crossings: {stability.get('total_crossings', 0)} (↑{stability.get('crossings_above', 0)}, ↓{stability.get('crossings_below', 0)})\n"
                )
                f.write(
                    f"  Is Stable Standing: {'Yes' if stability.get('is_stable', False) else 'No'}\n"
                )

                # Standing baseline peaks
                standing_peaks = stability.get("standing_peaks", [])
                if standing_peaks:
                    f.write(f"\n  STANDING BASELINE PEAKS ({len(standing_peaks)} detected):\n")
                    for i, peak in enumerate(standing_peaks, 1):
                        f.write(
                            f"    Peak {i}: {peak['force']:.2f} N at {peak['time']:.3f} s (prominence: {peak['prominence']:.2f} N)\n"
                        )
                f.write("\n")

            # Center of Pressure (CoP) analysis
            cop_results = analysis.get("cop_results", {})
            if cop_results and "cop_path_length" in cop_results:
                f.write("CENTER OF PRESSURE (CoP) ANALYSIS:\n")
                f.write(f"  CoP Path Length: {cop_results.get('cop_path_length', 0):.2f} mm\n")
                f.write(f"  Ellipse Area (95% confidence): {cop_results.get('ellipse_area_95', 0):.2f} mm²\n")
                f.write(f"  Ellipse Angle: {cop_results.get('ellipse_angle_deg', 0):.2f}°\n")
                f.write(
                    f"  Root Mean Square of Total Sway (CoP): {cop_results.get('rms_sway_total_mm', 0):.2f} mm\n"
                )
                f.write(
                    f"  RMS Sway ML (Medio-Lateral, X): {cop_results.get('rms_sway_ml_mm', 0):.2f} mm\n"
                )
                f.write(
                    f"  RMS Sway AP (Antero-Posterior, Y): {cop_results.get('rms_sway_ap_mm', 0):.2f} mm\n"
                )
                pca1 = cop_results.get("pca_pc1_variance_ratio", 0) * 100
                pca2 = cop_results.get("pca_pc2_variance_ratio", 0) * 100
                f.write(f"  PCA PC1 Explained Variance: {pca1:.2f}%\n")
                f.write(f"  PCA PC2 Explained Variance: {pca2:.2f}%\n")
                f.write("\n")

            # Plot information
            if "plot_path" in result:
                f.write(f"Plot saved: {Path(result['plot_path']).name}\n")

        # Generate individual CSV report
        csv_path = file_results_dir / f"{base_name}_analysis_data.csv"

        # Create comprehensive CSV with all metrics
        csv_data = {"metric": [], "value": [], "unit": []}

        # Basic metrics
        csv_data["metric"].extend(
            ["Duration", "Mean Force", "Max Force", "Min Force", "Total Samples"]
        )
        csv_data["value"].extend(
            [
                analysis["duration"],
                analysis["mean_force"],
                analysis["max_force"],
                analysis["min_force"],
                analysis["total_samples"],
            ]
        )
        csv_data["unit"].extend(["s", "N", "N", "N", "samples"])

        # Movement metrics
        csv_data["metric"].extend(
            [
                "Phases Detected",
                "Total Movement Time",
                "Average Phase Duration",
                "Phases per Minute",
            ]
        )
        csv_data["value"].extend(
            [
                movement["num_phases"],
                movement["total_movement_time"],
                movement["average_phase_duration"],
                movement["phases_per_minute"],
            ]
        )
        csv_data["unit"].extend(["count", "s", "s", "phases/min"])

        # Energy metrics
        if energy:
            csv_data["metric"].extend(
                [
                    "Body Weight",
                    "Total Mechanical Work",
                    "Total Metabolic Energy",
                    "Average Energy per Movement",
                    "Energy Efficiency",
                ]
            )
            csv_data["value"].extend(
                [
                    energy.get("body_weight_kg", 0),
                    energy.get("total_mechanical_work_J", 0),
                    energy.get("total_metabolic_energy_kcal", 0),
                    energy.get("average_energy_per_movement_kcal", 0),
                    energy.get("energy_efficiency", 0),
                ]
            )
            csv_data["unit"].extend(["kg", "J", "kcal", "kcal", "%"])

        # Stability metrics
        if stability:
            csv_data["metric"].extend(
                [
                    "Reference Peak Force",
                    "Stability Index",
                    "Mean Deviation from Reference Peak",
                    "Max Deviation from Reference Peak",
                    "Points Above Reference Peak",
                    "Points Below Reference Peak",
                    "Percent Above",
                    "Percent Below",
                    "Total Crossings",
                    "Crossings Above",
                    "Crossings Below",
                    "Is Stable",
                    "Num Standing Peaks",
                ]
            )
            csv_data["value"].extend(
                [
                    stability.get("first_peak_force", 0),
                    stability.get("stability_index", 0),
                    stability.get("mean_deviation", 0),
                    stability.get("max_deviation", 0),
                    stability.get("points_above", 0),
                    stability.get("points_below", 0),
                    stability.get("percent_above", 0),
                    stability.get("percent_below", 0),
                    stability.get("total_crossings", 0),
                    stability.get("crossings_above", 0),
                    stability.get("crossings_below", 0),
                    stability.get("is_stable", False),
                    stability.get("num_standing_peaks", 0),
                ]
            )
            csv_data["unit"].extend(
                [
                    "N",
                    "0-1 scale",
                    "N",
                    "N",
                    "count",
                    "count",
                    "%",
                    "%",
                    "count",
                    "count",
                    "count",
                    "boolean",
                    "count",
                ]
            )

            # Add individual standing peaks
            standing_peaks = stability.get("standing_peaks", [])
            for i, peak in enumerate(standing_peaks, 1):
                csv_data["metric"].append(f"Standing Peak {i} Time")
                csv_data["value"].append(peak["time"])
                csv_data["unit"].append("s")
                csv_data["metric"].append(f"Standing Peak {i} Force")
                csv_data["value"].append(peak["force"])
                csv_data["unit"].append("N")
                csv_data["metric"].append(f"Standing Peak {i} Prominence")
                csv_data["value"].append(peak["prominence"])
                csv_data["unit"].append("N")

        # Center of Pressure (CoP) metrics
        cop_results = analysis.get("cop_results", {})
        if cop_results and "cop_path_length" in cop_results:
            csv_data["metric"].extend(
                [
                    "CoP Path Length",
                    "CoP Ellipse Area (95%)",
                    "CoP Ellipse Angle",
                    "RMS Sway Total (CoP)",
                    "RMS Sway ML (Medio-Lateral)",
                    "RMS Sway AP (Antero-Posterior)",
                    "PCA PC1 Explained Variance",
                    "PCA PC2 Explained Variance",
                ]
            )
            pca1 = cop_results.get("pca_pc1_variance_ratio", 0) * 100
            pca2 = cop_results.get("pca_pc2_variance_ratio", 0) * 100
            csv_data["value"].extend(
                [
                    cop_results.get("cop_path_length", 0),
                    cop_results.get("ellipse_area_95", 0),
                    cop_results.get("ellipse_angle_deg", 0),
                    cop_results.get("rms_sway_total_mm", 0),
                    cop_results.get("rms_sway_ml_mm", 0),
                    cop_results.get("rms_sway_ap_mm", 0),
                    pca1,
                    pca2,
                ]
            )
            csv_data["unit"].extend(
                ["mm", "mm²", "deg", "mm", "mm", "mm", "%", "%"]
            )

        # Save CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding="utf-8")

        # print(f"  Individual reports saved: {os.path.basename(txt_path)}, {os.path.basename(csv_path)}")

    except Exception as e:
        print(f"Error generating individual report for {result['filename']}: {e}")
        import traceback

        traceback.print_exc()


def generate_batch_report(results, config, output_dir):
    """
    Generates a comprehensive batch analysis report and individual file reports.

    Parameters:
    -----------
    results : list
        List of analysis results
    config : dict
        Configuration used for analysis
    output_dir : str
        Output directory path
    """
    try:
        # Generate individual reports for each file first
        for result in results:
            if "error" not in result:
                generate_individual_report(result, config, output_dir)

        # Generate summary statistics
        successful_analyses = sum(1 for r in results if "error" not in r)
        total_files = len(results)

        # Generate batch summary text report in main results directory
        # Use the first result's directory structure to determine main results dir
        if results and "results_dir" in results[0]:
            # Extract main results dir from first result
            first_result_dir = Path(results[0]["results_dir"])
            main_results_dir = first_result_dir.parent
        else:
            main_results_dir = Path(output_dir) / "sit2stand_results"
        
        main_results_dir.mkdir(parents=True, exist_ok=True)
        report_path = main_results_dir / "batch_analysis_summary.txt"

        with open(report_path, "w", encoding="utf-8", errors="replace") as f:
            f.write("Sit-to-Stand Batch Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            # Configuration summary
            f.write("Configuration Used:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Force Column: {config['analysis']['force_column']}\n")
            f.write(
                f"Butterworth Filter: {'Enabled' if config['filtering']['enabled'] else 'Disabled'}\n"
            )

            if config["filtering"]["enabled"]:
                f.write(f"  Cutoff Frequency: {config['filtering']['cutoff_frequency']} Hz\n")
                f.write(f"  Sampling Frequency: {config['filtering']['sampling_frequency']} Hz\n")
                f.write(f"  Filter Order: {config['filtering']['order']}\n")

            f.write("Detection Parameters:\n")
            f.write(f"  Force Threshold: {config['detection']['force_threshold']} N\n")
            f.write(f"  Min Duration: {config['detection']['min_duration']} s\n")
            f.write(f"  Onset Threshold: {config['detection']['onset_threshold']} N\n")
            f.write("\n" + "=" * 60 + "\n\n")

            # Results summary
            f.write("Analysis Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total files processed: {total_files}\n")
            f.write(f"Successful analyses: {successful_analyses}\n")
            f.write(f"Failed analyses: {total_files - successful_analyses}\n")
            f.write(f"Success rate: {(successful_analyses / total_files * 100):.1f}%\n\n")

            # Detailed results with comprehensive clinical metrics
            total_phases = 0
            for result in results:
                if "error" not in result:
                    f.write(f"\nFile: {result['filename']}\n")
                    f.write("=" * 60 + "\n")
                    analysis = result["analysis"]

                    # === BASIC METRICS ===
                    f.write("BASIC FORCE METRICS:\n")
                    f.write(f"  Duration: {analysis['duration']:.2f} s\n")
                    f.write(f"  Mean Force: {analysis['mean_force']:.2f} N\n")
                    f.write(f"  Max Force: {analysis['max_force']:.2f} N\n")
                    f.write(f"  Min Force: {analysis['min_force']:.2f} N\n\n")

                    # === MOVEMENT DETECTION ===
                    movement = analysis["movement_metrics"]
                    f.write("MOVEMENT DETECTION:\n")
                    f.write(f"  Phases Detected: {movement['num_phases']}\n")
                    f.write(f"  Total Movement Time: {movement['total_movement_time']:.2f} s\n")
                    f.write(
                        f"  Average Phase Duration: {movement['average_phase_duration']:.2f} s\n"
                    )

                    if movement["phases_per_minute"] > 0:
                        f.write(f"  Phases per Minute: {movement['phases_per_minute']:.1f}\n\n")

                    # === TIME TO PEAK METRICS (CRITICAL FOR CP) ===
                    time_to_peak = analysis["time_to_peak_metrics"]
                    f.write("TIME TO PEAK METRICS:\n")
                    if time_to_peak.get("time_to_first_peak") is not None:
                        f.write(
                            f"  Time to First Peak: {time_to_peak['time_to_first_peak']:.3f} s\n"
                        )
                    if time_to_peak.get("time_to_max_force") is not None:
                        f.write(f"  Time to Max Force: {time_to_peak['time_to_max_force']:.3f} s\n")
                    if time_to_peak.get("average_time_to_peak") > 0:
                        f.write(
                            f"  Average Time to Peak: {time_to_peak['average_time_to_peak']:.3f} s\n"
                        )
                    if time_to_peak.get("time_to_peak_variation") > 0:
                        f.write(
                            f"  Time to Peak Variation: {time_to_peak['time_to_peak_variation']:.3f} s\n\n"
                        )

                    # === RATE OF FORCE DEVELOPMENT (CRITICAL FOR CP) ===
                    f.write("RATE OF FORCE DEVELOPMENT (RFD):\n")
                    phases_data = analysis.get("sit_to_stand_phases", [])
                    if phases_data:
                        first_phase = phases_data[0]
                        f.write(f"  Overall RFD: {first_phase.get('overall_rfd', 0):.2f} N/s\n")
                        f.write(
                            f"  Early RFD (first 100ms): {first_phase.get('early_rfd', 0):.2f} N/s\n"
                        )
                        f.write(f"  Peak RFD: {first_phase.get('peak_rfd', 0):.2f} N/s\n")
                        if first_phase.get("weight_transfer_time"):
                            f.write(
                                f"  Weight Transfer Time: {first_phase['weight_transfer_time']:.3f} s\n\n"
                            )

                    # === IMPULSE METRICS ===
                    impulse = analysis["impulse_metrics"]
                    f.write("IMPULSE & POWER METRICS:\n")
                    f.write(f"  Total Impulse: {impulse['total_impulse']:.2f} N⋅s\n")
                    f.write(f"  Average Impulse: {impulse['average_impulse']:.2f} N⋅s\n")
                    f.write(f"  Peak Power: {impulse['peak_power']:.2f} W\n")
                    f.write(f"  Average Power: {impulse['average_power']:.2f} W\n")
                    f.write(
                        f"  Force Rate of Change: {impulse['force_rate_of_change']:.2f} N/s\n\n"
                    )

                    # === CLINICAL QUALITY METRICS ===
                    if phases_data:
                        first_phase = phases_data[0]
                        f.write("MOVEMENT QUALITY METRICS:\n")
                        if "force_cv" in first_phase:
                            f.write(
                                f"  Force Coefficient of Variation: {first_phase['force_cv']:.2f}%\n"
                            )
                        if "force_jerk" in first_phase:
                            f.write(f"  Force Smoothness (Jerk): {first_phase['force_jerk']:.2f}\n")
                        if "bilateral_index" in first_phase:
                            f.write(
                                f"  Bilateral Symmetry Index: {first_phase['peak_symmetry']:.3f}\n"
                            )
                        if "consistency_score" in first_phase:
                            f.write(
                                f"  Movement Consistency: {first_phase['consistency_score']:.3f}\n"
                            )
                        if "num_peaks" in first_phase:
                            f.write(f"  Number of Peaks: {first_phase['num_peaks']}\n")

                    # === STABILITY METRICS ===
                    stability = analysis.get("stability_metrics", {})
                    if stability:
                        f.write("\nSTABILITY ANALYSIS (Oscillation around Reference Peak):\n")
                        f.write(
                            f"  Reference Peak Force: {stability.get('first_peak_force', 0):.2f} N\n"
                        )
                        f.write(
                            f"  Stability Index: {stability.get('stability_index', 0):.3f} (0-1 scale)\n"
                        )
                        f.write(
                            f"  Mean Deviation from Reference Peak: {stability.get('mean_deviation', 0):.2f} N\n"
                        )
                        f.write(
                            f"  Max Deviation from Reference Peak: {stability.get('max_deviation', 0):.2f} N\n"
                        )
                        f.write(
                            f"  Points Above Reference Peak: {stability.get('points_above', 0)} ({stability.get('percent_above', 0):.1f}%)\n"
                        )
                        f.write(
                            f"  Points Below Reference Peak: {stability.get('points_below', 0)} ({stability.get('percent_below', 0):.1f}%)\n"
                        )
                        f.write(
                            f"  Total Crossings: {stability.get('total_crossings', 0)} (Up: {stability.get('crossings_above', 0)}, Down: {stability.get('crossings_below', 0)})\n"
                        )
                        f.write(f"  Noise Level: {stability.get('noise_level', 0):.2f} N\n")
                        f.write(
                            f"  Oscillation Frequency: {stability.get('oscillation_frequency', 0):.2f} Hz\n"
                        )
                        f.write(
                            f"  Stability Duration: {stability.get('stability_duration', 0):.2f} s\n"
                        )
                        f.write(
                            f"  Is Stable Standing: {'Yes' if stability.get('is_stable', False) else 'No'}\n"
                        )

                        # Standing baseline peaks
                        standing_peaks = stability.get("standing_peaks", [])
                        if standing_peaks:
                            f.write(
                                f"\n  STANDING BASELINE PEAKS ({len(standing_peaks)} detected):\n"
                            )
                            for i, peak in enumerate(standing_peaks, 1):
                                f.write(
                                    f"    Peak {i}: {peak['force']:.2f} N at {peak['time']:.3f} s (prominence: {peak['prominence']:.2f} N)\n"
                                )

                    # === ENERGY EXPENDITURE METRICS ===
                    energy = analysis.get("energy_metrics", {})
                    if energy:
                        f.write("\nENERGY EXPENDITURE ANALYSIS:\n")
                        f.write(
                            f"  Body Weight: {energy.get('body_weight_kg', 0):.1f} kg ({energy.get('body_weight_N', 0):.1f} N)\n"
                        )
                        f.write(f"  Total Movements: {energy.get('total_movements', 0)}\n")
                        f.write(
                            f"  Total Mechanical Work: {energy.get('total_mechanical_work_J', 0):.2f} J ({energy.get('total_mechanical_work_J', 0) / 4184:.3f} kcal)\n"
                        )
                        f.write(
                            f"  Total Metabolic Energy: {energy.get('total_metabolic_energy_kcal', 0):.3f} kcal\n"
                        )
                        f.write(
                            f"  Average Energy per Movement: {energy.get('average_energy_per_movement_kcal', 0):.3f} kcal\n"
                        )
                        f.write(f"  Energy Efficiency: {energy.get('energy_efficiency', 0):.1f}%\n")
                        f.write(
                            f"  Chair Height (reference): {energy.get('chair_height_m', 0):.3f} m\n"
                        )
                        f.write(f"  Reference Study: {energy.get('reference_study', 'N/A')}\n\n")

                        # Detailed phase energy data
                        phases_energy = energy.get("phases_energy", [])
                        if phases_energy:
                            f.write("  MOVEMENT-BY-MOVEMENT ENERGY:\n")
                            for phase_energy in phases_energy:
                                f.write(
                                    f"    Phase {phase_energy['phase_number']} ({phase_energy['movement_type']}):\n"
                                )
                                f.write(f"      Duration: {phase_energy['duration_s']:.2f} s\n")
                                f.write(
                                    f"      Mechanical Work: {phase_energy['mechanical_work_J']:.2f} J\n"
                                )
                                f.write(
                                    f"      Metabolic Energy: {phase_energy['metabolic_energy_kcal']:.3f} kcal\n"
                                )
                                f.write(
                                    f"      Average Force: {phase_energy['average_force_N']:.1f} N\n"
                                )
                                f.write(
                                    f"      Force above Body Weight: {phase_energy['force_above_body_weight_N']:.1f} N\n"
                                )
                                f.write(
                                    f"      Energy Efficiency: {phase_energy['energy_efficiency_percent']:.1f}%\n"
                                )
                            f.write("\n")

                    # Plot information
                    if "plot_path" in result:
                        f.write(f"Plot saved: {os.path.basename(result['plot_path'])}\n")

                    f.write("=" * 60 + "\n")
                    total_phases += movement["num_phases"]
                else:
                    f.write(f"\nERROR - {result['filename']}: {result['error']}\n")
                    f.write("=" * 60 + "\n")

            f.write("\nOverall Summary:\n")
            f.write(f"Total phases detected across all files: {total_phases}\n")
            f.write(
                f"Average phases per successful file: {total_phases / successful_analyses:.1f}\n"
            )

        # Generate comprehensive CSV summary with all clinical metrics
        csv_data = []
        for result in results:
            if "error" not in result:
                analysis = result["analysis"]
                movement = analysis["movement_metrics"]
                impulse = analysis["impulse_metrics"]
                time_to_peak = analysis["time_to_peak_metrics"]
                phases_data = analysis.get("sit_to_stand_phases", [])

                # Base metrics
                row = {
                    "filename": result["filename"],
                    "duration_s": analysis["duration"],
                    "mean_force_N": analysis["mean_force"],
                    "max_force_N": analysis["max_force"],
                    "min_force_N": analysis["min_force"],
                    # Movement detection
                    "num_phases": movement["num_phases"],
                    "total_movement_time_s": movement["total_movement_time"],
                    "average_phase_duration_s": movement["average_phase_duration"],
                    "phases_per_minute": movement["phases_per_minute"],
                    # Time to peak metrics (CRITICAL)
                    "time_to_first_peak_s": time_to_peak.get("time_to_first_peak"),
                    "time_to_max_force_s": time_to_peak.get("time_to_max_force"),
                    "average_time_to_peak_s": time_to_peak.get("average_time_to_peak", 0),
                    "time_to_peak_variation_s": time_to_peak.get("time_to_peak_variation", 0),
                    # Impulse metrics
                    "total_impulse_Ns": impulse["total_impulse"],
                    "average_impulse_Ns": impulse["average_impulse"],
                    "peak_power_W": impulse["peak_power"],
                    "average_power_W": impulse["average_power"],
                    "force_rate_of_change_Ns": impulse["force_rate_of_change"],
                    # Symmetry metrics
                    "symmetry_index": movement.get("symmetry_index", 0),
                    # Stability metrics (oscillation around first peak)
                    "first_peak_force_N": analysis.get("stability_metrics", {}).get(
                        "first_peak_force", 0
                    ),
                    "stability_index": analysis.get("stability_metrics", {}).get(
                        "stability_index", 0
                    ),
                    "mean_deviation_from_first_peak_N": analysis.get("stability_metrics", {}).get(
                        "mean_deviation", 0
                    ),
                    "max_deviation_from_first_peak_N": analysis.get("stability_metrics", {}).get(
                        "max_deviation", 0
                    ),
                    "points_above_first_peak": analysis.get("stability_metrics", {}).get(
                        "points_above", 0
                    ),
                    "points_below_first_peak": analysis.get("stability_metrics", {}).get(
                        "points_below", 0
                    ),
                    "percent_above_first_peak": analysis.get("stability_metrics", {}).get(
                        "percent_above", 0
                    ),
                    "percent_below_first_peak": analysis.get("stability_metrics", {}).get(
                        "percent_below", 0
                    ),
                    "total_crossings": analysis.get("stability_metrics", {}).get(
                        "total_crossings", 0
                    ),
                    "crossings_above": analysis.get("stability_metrics", {}).get(
                        "crossings_above", 0
                    ),
                    "crossings_below": analysis.get("stability_metrics", {}).get(
                        "crossings_below", 0
                    ),
                    "noise_level_N": analysis.get("stability_metrics", {}).get("noise_level", 0),
                    "oscillation_frequency_Hz": analysis.get("stability_metrics", {}).get(
                        "oscillation_frequency", 0
                    ),
                    "stability_duration_s": analysis.get("stability_metrics", {}).get(
                        "stability_duration", 0
                    ),
                    "is_stable": analysis.get("stability_metrics", {}).get("is_stable", False),
                    "num_standing_peaks": analysis.get("stability_metrics", {}).get(
                        "num_standing_peaks", 0
                    ),
                    # Energy expenditure metrics
                    "body_weight_kg": analysis.get("energy_metrics", {}).get("body_weight_kg", 0),
                    "body_weight_N": analysis.get("energy_metrics", {}).get("body_weight_N", 0),
                    "total_mechanical_work_J": analysis.get("energy_metrics", {}).get(
                        "total_mechanical_work_J", 0
                    ),
                    "total_mechanical_work_kcal": analysis.get("energy_metrics", {}).get(
                        "total_mechanical_work_J", 0
                    )
                    / 4184
                    if analysis.get("energy_metrics", {}).get("total_mechanical_work_J", 0) > 0
                    else 0,
                    "total_metabolic_energy_kcal": analysis.get("energy_metrics", {}).get(
                        "total_metabolic_energy_kcal", 0
                    ),
                    "average_energy_per_movement_kcal": analysis.get("energy_metrics", {}).get(
                        "average_energy_per_movement_kcal", 0
                    ),
                    "energy_per_movement_J": analysis.get("energy_metrics", {}).get(
                        "energy_per_movement_J", 0
                    ),
                    "energy_efficiency_percent": analysis.get("energy_metrics", {}).get(
                        "energy_efficiency", 0
                    ),
                    "total_movements": analysis.get("energy_metrics", {}).get("total_movements", 0),
                }

                # Add RFD metrics from first phase if available
                if phases_data:
                    first_phase = phases_data[0]
                    row.update(
                        {
                            "overall_rfd_Ns": first_phase.get("overall_rfd", 0),
                            "early_rfd_Ns": first_phase.get("early_rfd", 0),
                            "peak_rfd_Ns": first_phase.get("peak_rfd", 0),
                            "weight_transfer_time_s": first_phase.get("weight_transfer_time"),
                            "first_peak_force_N": first_phase.get("first_peak_force", 0),
                            "max_peak_force_N": first_phase.get("max_peak_force", 0),
                            "num_peaks_detected": first_phase.get("num_peaks", 0),
                            "force_cv_percent": first_phase.get("force_cv", 0),
                            "force_jerk": first_phase.get("force_jerk", 0),
                            "peak_symmetry": first_phase.get("peak_symmetry", 0),
                            "temporal_symmetry": first_phase.get("temporal_symmetry", 0),
                            "bilateral_index": first_phase.get("bilateral_index", 0),
                            "consistency_score": first_phase.get("consistency_score", 0),
                            "has_unloading_phase": first_phase.get("has_unloading_phase", False),
                            "force_range_N": first_phase.get("force_range", 0),
                            "force_excursion_N": first_phase.get("force_excursion", 0),
                            "mean_force_during_movement_N": first_phase.get(
                                "mean_force_during_movement", 0
                            ),
                            "sampling_frequency_Hz": first_phase.get("sampling_frequency", 0),
                        }
                    )

                csv_data.append(row)

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = main_results_dir / "batch_analysis_summary.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")
        else:
            csv_path = None

        print("\nBatch report generated:")
        print(f"  Batch summary report: {report_path}")
        print(f"  Batch CSV summary: {csv_path}")
        print(f"  Individual reports and plots saved in: {main_results_dir}")

    except Exception as e:
        print(f"Error generating batch report: {e}")


def read_c3d_file(file_path, column_name):
    """
    Reads C3D file and extracts the specified analog channel.
    Uses ezc3d library for comprehensive C3D file support.
    """
    if not C3D_SUPPORT:
        print("C3D support not available - ezc3d library not installed")
        return None

    try:
        print(f"Reading C3D file: {file_path}")

        # Load the C3D file using ezc3d
        datac3d = c3d(file_path, extract_forceplat_data=True)

        # Print detailed information about the C3D file
        print_c3d_info(datac3d)

        # Get analog data and labels
        analogs = datac3d["data"]["analogs"]
        analog_labels_raw = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
        
        # Handle different label formats
        if analog_labels_raw:
            if isinstance(analog_labels_raw[0], list):
                analog_labels = [label[0] if isinstance(label, list) else str(label) for label in analog_labels_raw]
            else:
                analog_labels = [str(label) for label in analog_labels_raw]
        else:
            analog_labels = []

        # Identify the target plate based on the requested column name (e.g. Force.Fz3 -> Plate 3)
        # We need to extract all 6 channels: Fx, Fy, Fz, Mx, My, Mz
        
        # 1. Determine suffix/index from the requested column
        # E.g., if column_name is "Force.Fz3", we look for "3"
        # If no number, we assume 1 or look for corresponding channels
        
        target_index = ""
        match = re.search(r"(\d+)$", column_name)
        if match:
            target_index = match.group(1)
            
        print(f"Targeting Force Plate Index: {target_index if target_index else 'Auto'}")

        # Define channel patterns to look for
        components = {
            "Fx": [f"Force.Fx{target_index}", f"Fx{target_index}", f"FX{target_index}", f"Force.FX{target_index}"],
            "Fy": [f"Force.Fy{target_index}", f"Fy{target_index}", f"FY{target_index}", f"Force.FY{target_index}"],
            "Fz": [f"Force.Fz{target_index}", f"Fz{target_index}", f"FZ{target_index}", f"Force.FZ{target_index}"],
            "Mx": [f"Moment.Mx{target_index}", f"Mx{target_index}", f"MX{target_index}", f"Moment.MX{target_index}"],
            "My": [f"Moment.My{target_index}", f"My{target_index}", f"MY{target_index}", f"Moment.MY{target_index}"],
            "Mz": [f"Moment.Mz{target_index}", f"Mz{target_index}", f"MZ{target_index}", f"Moment.MZ{target_index}"],
        }

        # If strict column name was passed and found, use it as Fz primary
        # But we also try to find the neighbors
        
        found_channels = {}
        
        for comp, patterns in components.items():
            for pattern in patterns:
                # Case insensitive search
                matching = [l for l in analog_labels if l.lower() == pattern.lower()]
                if matching:
                    found_channels[comp] = matching[0]
                    break
        
        # Check if we at least have Fz
        if "Fz" not in found_channels:
             # Fallback: if we were given a specific column_name that corresponds to vertical force
             # and we couldn't match the pattern logic above, verify if column_name itself exists
             matching = [l for l in analog_labels if l.lower() == column_name.lower()]
             if matching:
                 found_channels["Fz"] = matching[0]
                 print(f"Using requested column '{matching[0]}' as vertical force (Fz)")
             else:
                 print(f"Could not find vertical force channel (Fz). Requested: {column_name}")
                 return None

        print("Found channels:")
        for k, v in found_channels.items():
            print(f"  {k}: {v}")

        # Extract data for all found channels
        extracted_data = {}
        
        for comp, label in found_channels.items():
            idx = analog_labels.index(label)
            
            # shape handling (1, n_ch, n_frames) or (n_ch, n_frames)
            if analogs.ndim == 3:
                vals = analogs[0, idx, :]
            else:
                 # assume (n_ch, n_frames)
                 vals = analogs[idx, :]
            
            extracted_data[comp] = vals

        # Ensure Fz is non-negative (reaction force convention), same as forceplate_analysis calc_cop
        if np.any(extracted_data["Fz"] < 0):
            extracted_data["Fz"] = -np.asarray(extracted_data["Fz"])

        # Get timing
        analog_freq = datac3d["header"]["analogs"]["frame_rate"]
        num_frames = len(extracted_data["Fz"])
        time_values = np.arange(num_frames) / analog_freq

        # Construct DataFrame
        # Always ensure we have 'Time' and 'Force' (Fz alias) for backward compatibility
        df_dict = {
            "Time": time_values,
            "Force": extracted_data["Fz"]
        }
        
        # Add all components with standard names
        for comp in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
            if comp in extracted_data:
                df_dict[f"Force.{comp}"] = extracted_data[comp]
                
        force_data = pd.DataFrame(df_dict)
        
        print(f"Analog frequency: {analog_freq} Hz")
        print(f"Data points: {num_frames}")
        print(f"Time range: {time_values[0]:.3f} - {time_values[-1]:.3f} s")

        return force_data

    except Exception as e:
        print(f"Error reading C3D file: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_c3d_info(datac3d):
    """
    Print detailed information about the C3D file structure.
    """
    try:
        print("\n" + "=" * 50)
        print("C3D FILE INFORMATION")
        print("=" * 50)

        # Basic header info
        print(f"Points frame rate: {datac3d['header']['points']['frame_rate']} Hz")
        print(f"Analog frame rate: {datac3d['header']['analogs']['frame_rate']} Hz")
        print(f"Number of points: {datac3d['parameters']['POINT']['USED']['value'][0]}")
        print(f"Number of analog channels: {datac3d['parameters']['ANALOG']['USED']['value'][0]}")

        # Point/marker information
        if datac3d["parameters"]["POINT"]["USED"]["value"][0] > 0:
            marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
            print(f"Marker labels: {marker_labels}")

        # Analog information
        analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
        analog_units = (
            datac3d["parameters"]["ANALOG"]
            .get("UNITS", {})
            .get("value", ["Unknown"] * len(analog_labels))
        )

        print("\nAnalog channels and units:")
        for label, unit in zip(analog_labels, analog_units, strict=True):
            try:
                # Handle encoding issues with special characters
                label_str = str(label).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                unit_str = str(unit).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                print(f"  {label_str}: {unit_str}")
            except Exception:
                print(f"  [Label {analog_labels.index(label)}]: {unit}")

        # Force platform information
        if "platform" in datac3d["data"] and datac3d["data"]["platform"]:
            print(f"\nForce platforms: {len(datac3d['data']['platform'])}")
            for i, platform in enumerate(datac3d["data"]["platform"]):
                print(f"  Platform {i}:")
                if "center_of_pressure" in platform:
                    cop_shape = platform["center_of_pressure"].shape
                    print(f"    COP data shape: {cop_shape}")
                if "force" in platform:
                    force_shape = platform["force"].shape
                    print(f"    Force data shape: {force_shape}")

        print("=" * 50)

    except Exception as e:
        print(f"Error printing C3D info: {e}")


def suggest_force_column(analog_labels):
    """
    Suggests the most appropriate force column from available analog channels.
    """
    # Priority order for force column detection
    force_patterns = [
        # Force plate vertical forces (most common for sit-to-stand)
        "FZ1",
        "Fz1",
        "Force.Fz1",
        "force_z_1",
        "FZ2",
        "Fz2",
        "Force.Fz2",
        "force_z_2",
        "FZ3",
        "Fz3",
        "Force.Fz3",
        "force_z_3",
        "FZ4",
        "Fz4",
        "Force.Fz4",
        "force_z_4",
        # Generic force patterns
        "FZ",
        "Fz",
        "Force_Z",
        "force_z",
        # Any force component
        "FX",
        "FY",
        "Fx",
        "Fy",
        "Force_X",
        "Force_Y",
    ]

    for pattern in force_patterns:
        for label in analog_labels:
            if pattern.lower() in label.lower():
                return label

    return None


def read_csv_file(file_path, column_name, config=None):
    """
    Reads CSV file and extracts the specified column using pandas.

    Parameters:
    -----------
    file_path : str
        Path to CSV file
    column_name : str
        Name of column to extract
    config : dict, optional
        Configuration dictionary containing FPS and other parameters

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with timestamp and force data, or None if error
    """
    try:
        # Read CSV file header to check available columns
        print(f"Reading CSV file: {file_path}")
        df_header = pd.read_csv(file_path, nrows=0)
        available_columns = list(df_header.columns)
        print(f"Available columns: {available_columns}")

        # Check if column exists
        if column_name not in df_header.columns:
            print(f"Column '{column_name}' not found in {file_path}")
            print(f"Available columns: {available_columns}")

            # Try to auto-detect column
            detected_column = auto_detect_force_column(file_path)
            if detected_column:
                print(f"Using auto-detected column: {detected_column}")
                column_name = detected_column
            else:
                print("No suitable column found")
                return None

        # Read full CSV file
        df = pd.read_csv(file_path)

        # Try to identify time column (first numeric column or column with 'time' in name)
        time_col = None
        for col in df.columns:
            if col.lower() in ["time", "timestamp", "frame", "t"]:
                time_col = col
                break

        # If no time column found, use first column
        if time_col is None:
            time_col = df.columns[0]

        print(f"Using time column: {time_col}")
        print(f"Using force column: {column_name}")

        # Check if Time column is problematic (only 0s and 1s or not increasing properly)
        if time_col.lower() in ["time", "tempo", "frame", "t"]:
            unique_time_values = sorted(df[time_col].unique())
            if len(unique_time_values) <= 2 or not df[time_col].is_monotonic_increasing:
                print(f"[WARNING] Time column has issues: {unique_time_values}")
                print("[DEBUG] Generating time vector from FPS configuration")
                # Generate time vector using FPS from config
                fps = 100.0  # Default FPS
                if config and "analysis" in config:
                    fps = config["analysis"].get("fps", 100.0)

                # Create time vector based on FPS
                df = df.copy()
                time_vector = np.arange(len(df)) / fps
                df.insert(0, "Time_Generated", time_vector)
                time_col = "Time_Generated"
                print(f"Generated time vector using FPS: {fps} Hz")

        # Extract relevant data
        force_data = df[[time_col, column_name]].copy()

        # Rename columns for consistency
        force_data.columns = ["Time", "Force"]

        # Ensure Time column is numeric
        force_data["Time"] = pd.to_numeric(force_data["Time"], errors="coerce")

        # Ensure Force column is numeric
        force_data["Force"] = pd.to_numeric(force_data["Force"], errors="coerce")

        # Ensure Fz is non-negative (reaction force convention), same as C3D and forceplate_analysis
        if np.any(force_data["Force"] < 0):
            force_data["Force"] = -force_data["Force"]

        # If force_column suggests a plate index (e.g. Force.Fz3), try to add 6 components for CoP
        plate_idx = None
        if "Force.Fz" in column_name or "Fz" in column_name:
            m = re.search(r"Fz(\d+)|Force\.Fz(\d+)", column_name, re.IGNORECASE)
            if m:
                plate_idx = int(m.group(1) or m.group(2))
        if plate_idx is not None:
            cand = {
                "Force.Fx": f"Force.Fx{plate_idx}",
                "Force.Fy": f"Force.Fy{plate_idx}",
                "Force.Fz": f"Force.Fz{plate_idx}",
                "Force.Mx": f"Moment.Mx{plate_idx}",
                "Force.My": f"Moment.My{plate_idx}",
                "Force.Mz": f"Moment.Mz{plate_idx}",
            }
            if all(c in df.columns for c in cand.values()):
                for std_name, csv_name in cand.items():
                    force_data[std_name] = pd.to_numeric(df[csv_name], errors="coerce")
                # Align Fz sign: use same non-negative convention for Force.Fz
                if np.any(force_data["Force.Fz"] < 0):
                    force_data["Force.Fz"] = -force_data["Force.Fz"]
                print(f"Added 6 components for plate {plate_idx} for CoP calculation")

        # Remove any rows with NaN values (only in Time/Force for core; NaN in optional columns left as-is for CoP mask)
        force_data = force_data.dropna(subset=["Time", "Force"])

        if len(force_data) == 0:
            print("No valid data after cleaning")
            return None

        print(f"Successfully loaded {len(force_data)} data points")
        print(f"Time range: {force_data['Time'].min():.2f} - {force_data['Time'].max():.2f} s")
        print(f"Force range: {force_data['Force'].min():.2f} - {force_data['Force'].max():.2f} N")

        return force_data

    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None


def analyze_sit_to_stand(data, config):
    """
    Analyzes sit-to-stand movement in force data with advanced detection algorithms.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with 'Time' and 'Force' columns
    config : dict
        Configuration dictionary with detection parameters

    Returns:
    --------
    dict
        Analysis results with detailed movement metrics
    """
    try:
        # Extract parameters from config
        force_threshold = config["detection"]["force_threshold"]
        min_duration = config["detection"]["min_duration"]
        onset_threshold = config["detection"]["onset_threshold"]

        # Basic statistics
        total_samples = len(data)
        duration = data["Time"].iloc[-1] - data["Time"].iloc[0] if len(data) > 0 else 0
        mean_force = data["Force"].mean()
        max_force = data["Force"].max()
        min_force = data["Force"].min()

        # Detect ALL peaks in the entire signal using scipy if enabled
        all_peaks = []
        all_peaks_global = []  # All peaks in entire signal
        if config.get("detection", {}).get("peak_detection"):
            print("Using scipy.find_peaks for peak detection on entire signal")
            all_peaks_global = detect_peaks_scipy(data["Force"].values, data["Time"].values, config)
            all_peaks = all_peaks_global  # For compatibility with existing code

            print(f"Total peaks detected in signal: {len(all_peaks_global)}")
            if len(all_peaks_global) > 0:
                print(f"  Peak times: {[f'{p["time"]:.3f}s' for p in all_peaks_global[:10]]}")
                print(f"  Peak forces: {[f'{p["force"]:.2f}N' for p in all_peaks_global[:10]]}")

        # Detect sit-to-stand phases
        sit_to_stand_phases = detect_sit_to_stand_phases(
            data["Force"].values,
            data["Time"].values,
            force_threshold,
            min_duration,
            onset_threshold,
        )

        # Calculate stability metrics if enabled
        stability_metrics = {}
        if config.get("stability", {}).get("enabled", False):
            print("Calculating stability index...")
            # Use the maximum force peak as reference (not just first peak)
            if all_peaks:
                # Find the peak with maximum force
                max_peak = max(all_peaks, key=lambda p: p["force"])
                reference_peak_time = max_peak["time"]
                reference_peak_force = max_peak["force"]
                print(
                    f"Using maximum peak as reference: {reference_peak_force:.2f} N at {reference_peak_time:.3f} s"
                )
            else:
                # Fallback: use the maximum force in the data
                max_force_idx = np.argmax(data["Force"].values)
                reference_peak_time = data["Time"].iloc[max_force_idx]
                reference_peak_force = data["Force"].iloc[max_force_idx]
                print(
                    f"Using maximum force as reference: {reference_peak_force:.2f} N at {reference_peak_time:.3f} s"
                )

            stability_metrics = calculate_stability_index(
                data["Force"].values, data["Time"].values, reference_peak_time, config
            )

        # Calculate movement metrics
        movement_metrics = calculate_movement_metrics(data, sit_to_stand_phases)

        # Calculate impulse and power metrics
        impulse_metrics = calculate_impulse_metrics(data, sit_to_stand_phases)

        # Calculate time to peak metrics
        time_to_peak_metrics = calculate_time_to_peak_metrics(data, sit_to_stand_phases)

        # Calculate energy expenditure if body weight is configured
        energy_metrics = {}
        if config.get("analysis", {}).get("body_weight"):
            print("Calculating energy expenditure...")
            energy_metrics = calculate_energy_expenditure(
                data["Force"].values, data["Time"].values, sit_to_stand_phases, config
            )

        # Calculate CoP if 6-component data is available
        cop_results = {}
        if config.get("detection", {}).get("calculate_cop", True) and COP_SUPPORT:
            # Check if we have all necessary components
            # We look for columns starting with "Force." and containing the component names
            components_present = True
            component_cols = {}
            for comp in ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]:
                col = f"Force.{comp}"
                if col not in data.columns:
                    components_present = False
                    break
                component_cols[comp] = col
            
            if components_present:
                print("Calculating Center of Pressure (CoP)...")
                try:
                    # Get force plate dimensions from config
                    fp_config = config.get("force_plate", {})
                    width_mm = fp_config.get("width_mm", 464.0)
                    height_mm = fp_config.get("height_mm", 508.0)
                    
                    # Extract raw components (ensure numpy arrays)
                    Fx = data[component_cols["Fx"]].values
                    Fy = data[component_cols["Fy"]].values
                    Fz = data[component_cols["Fz"]].values
                    Mx = data[component_cols["Mx"]].values
                    My = data[component_cols["My"]].values
                    Mz = data[component_cols["Mz"]].values
                    
                    # Prepare arguments for calc_cop
                    # calc_cop signature: (forces_moments, dimensions)
                    # forces_moments = (Fx, Fy, Fz, Mx, My, Mz)
                    # dimensions = (width_mm, height_mm) -- expecting half-dimensions or full?
                    # cop_calculate.py usually standardizes on full dimensions if it does the offset math,
                    # OR it might be simpler. Let's look at the implementation of Shimba in cop_calculate.py
                    # To be safe, I will use my direct implementation which I know is robust for this context.
                    # AND it avoids dependency issues if cop_calculate changes.
                    
                    # Shimba Method (1984) - Simplified for Type 2 Plate (Origin at center of top surface)
                    # CoPx = -My / Fz
                    # CoPy =  Mx / Fz
                    
                    # Check moment unit
                    moment_unit = fp_config.get("moment_unit", "N.mm")
                    
                    # If moments are in N.mm and Force in N, Result is in mm.
                    # If moments are in N.m, Result is in m.
                    
                    # Filter Fz to avoid division by zero
                    # Use a threshold (e.g., 10 N)
                    valid_mask = np.abs(Fz) > 10.0
                    
                    cop_x = np.full_like(Fz, np.nan)
                    cop_y = np.full_like(Fz, np.nan)
                    
                    # Calculate CoP
                    # Note: Mx is moment about X-axis (controls Y-coord), My is moment about Y-axis (controls X-coord)
                    # Check sign convention. 
                    # Standard biomechanics (Right-Hand Rule):
                    # CoP_x = -My / Fz
                    # CoP_y =  Mx / Fz
                    
                    cop_x[valid_mask] = -My[valid_mask] / Fz[valid_mask]
                    cop_y[valid_mask] =  Mx[valid_mask] / Fz[valid_mask]
                    
                    # Unit conversion if needed (target: mm)
                    if "mm" not in moment_unit.lower(): # e.g. "N.m"
                         cop_x *= 1000.0
                         cop_y *= 1000.0
                    
                    # Root Mean Square of CoP sway (total and separated into ML/AP)
                    n_valid = np.sum(valid_mask)
                    rms_sway_total_mm = 0.0
                    rms_sway_ml_mm = 0.0
                    rms_sway_ap_mm = 0.0
                    if n_valid > 0:
                        cx = cop_x[valid_mask]
                        cy = cop_y[valid_mask]
                        mx = np.nanmean(cx)
                        my = np.nanmean(cy)
                        rms_sway_ml_mm = float(np.sqrt(np.nanmean((cx - mx) ** 2)))
                        rms_sway_ap_mm = float(np.sqrt(np.nanmean((cy - my) ** 2)))
                        rms_sway_total_mm = float(np.sqrt(np.nanmean((cx - mx) ** 2 + (cy - my) ** 2)))

                    # 95% Confidence Ellipse (PCA-based via ellipse.py, or covariance fallback)
                    ellipse_area_95 = 0.0
                    ellipse_angle_deg = 0.0
                    pca_pc1_variance_ratio = 0.0
                    pca_pc2_variance_ratio = 0.0
                    ellipse_x_path = None
                    ellipse_y_path = None
                    if n_valid > 10:
                        cop_data_2d = np.column_stack((cop_x[valid_mask], cop_y[valid_mask]))
                        if ELLIPSE_SUPPORT:
                            try:
                                area, angle, _bounds, ellipse_data = plot_ellipse_pca(
                                    cop_data_2d, confidence=0.95
                                )
                                ellipse_area_95 = float(area)
                                ellipse_angle_deg = float(angle)
                                if len(ellipse_data) > 5:
                                    evr = ellipse_data[5]
                                    pca_pc1_variance_ratio = float(evr[0]) if len(evr) > 0 else 0.0
                                    pca_pc2_variance_ratio = float(evr[1]) if len(evr) > 1 else 0.0
                                ellipse_x_path = ellipse_data[0].tolist()
                                ellipse_y_path = ellipse_data[1].tolist()
                            except Exception as e_ellipse:
                                print(f"Error in ellipse.py: {e_ellipse}")
                        if ellipse_area_95 == 0.0:
                            try:
                                cov_matrix = np.cov([cop_x[valid_mask], cop_y[valid_mask]])
                                eigenvalues = np.linalg.eigvals(cov_matrix)
                                ellipse_area_95 = float(
                                    np.pi * np.prod(np.sqrt(np.abs(eigenvalues))) * 5.991
                                )
                            except Exception as e_ellipse:
                                print(f"Error calculating Ellipse Area: {e_ellipse}")

                    cop_results = {
                        "cop_x": cop_x,
                        "cop_y": cop_y,
                        "valid_mask": valid_mask,
                        "mean_cop_x": np.nanmean(cop_x),
                        "mean_cop_y": np.nanmean(cop_y),
                        "cop_path_length": np.nansum(np.sqrt(np.diff(cop_x[valid_mask])**2 + np.diff(cop_y[valid_mask])**2)) if np.any(valid_mask) else 0.0,
                        "ellipse_area_95": ellipse_area_95,
                        "ellipse_angle_deg": ellipse_angle_deg,
                        "pca_pc1_variance_ratio": pca_pc1_variance_ratio,
                        "pca_pc2_variance_ratio": pca_pc2_variance_ratio,
                        "rms_sway_total_mm": rms_sway_total_mm,
                        "rms_sway_ml_mm": rms_sway_ml_mm,
                        "rms_sway_ap_mm": rms_sway_ap_mm,
                        "ellipse_x_path": ellipse_x_path,
                        "ellipse_y_path": ellipse_y_path,
                    }
                    print(f"CoP Path Length: {cop_results['cop_path_length']:.2f} mm")
                    print(f"CoP Ellipse Area (95%): {cop_results['ellipse_area_95']:.2f} mm^2")
                    print(f"CoP RMS Sway Total: {cop_results['rms_sway_total_mm']:.2f} mm (ML: {cop_results['rms_sway_ml_mm']:.2f}, AP: {cop_results['rms_sway_ap_mm']:.2f})")
                    
                except Exception as e:
                    print(f"Error calculating CoP: {e}")

        results = {
            "total_samples": total_samples,
            "duration": duration,
            "mean_force": mean_force,
            "max_force": max_force,
            "min_force": min_force,
            "all_peaks": all_peaks,
            "all_peaks_global": all_peaks_global,  # All peaks in entire signal
            "sit_to_stand_phases": sit_to_stand_phases,
            "movement_metrics": movement_metrics,
            "impulse_metrics": impulse_metrics,
            "time_to_peak_metrics": time_to_peak_metrics,
            "stability_metrics": stability_metrics,
            "energy_metrics": energy_metrics,
            "cop_results": cop_results, 
            "detection_threshold": force_threshold,
            "status": "analyzed",
        }

        return results

    except Exception as e:
        print(f"Error in sit-to-stand analysis: {str(e)}")
        return {
            "total_samples": len(data),
            "duration": data["Time"].iloc[-1] - data["Time"].iloc[0] if len(data) > 0 else 0,
            "mean_force": data["Force"].mean(),
            "max_force": data["Force"].max(),
            "min_force": data["Force"].min(),
            "error": str(e),
            "status": "error",
        }


def detect_sit_to_stand_phases(
    force_data,
    time_data,
    force_threshold,
    min_duration,
    onset_threshold,
    auto_threshold=True,
):
    """
    Detects sit-to-stand phases in force data with enhanced clinical biomechanical analysis.
    Specifically designed for pediatric cerebral palsy assessment with comprehensive metrics.

    Parameters:
    -----------
    force_data : array-like
        Vertical force values (Fz) from force plate
    time_data : array-like
        Time values corresponding to force measurements
    force_threshold : float
        Minimum force threshold for movement detection (N)
    min_duration : float
        Minimum duration for a valid sit-to-stand phase (seconds)
    onset_threshold : float
        Threshold for movement onset detection above baseline (N)
    auto_threshold : bool
        Whether to use automatic threshold detection based on force profile

    Returns:
    --------
    list
        List of detected phases with comprehensive biomechanical metrics including:
        - Time to first peak, time to max force
        - Rate of force development (RFD) - critical for CP assessment
        - Impulse and power metrics
        - Peak detection and timing analysis
        - Symmetry and consistency measures
    """
    phases = []

    try:
        # Auto-detect threshold if enabled (based on force gradient analysis)
        if auto_threshold:
            detected_threshold = detect_ascending_threshold(force_data, time_data)
            if detected_threshold:
                # detected_threshold is already relative to baseline, so use it directly
                # But validate it's reasonable (should be small relative to max force)
                baseline_force_prelim = np.percentile(force_data, 10)
                max_force = np.max(force_data)
                if detected_threshold < (max_force - baseline_force_prelim) * 0.5:
                    onset_threshold = detected_threshold
                    print(f"Auto-detected onset threshold: {onset_threshold:.2f} N")
                else:
                    print(
                        f"Auto-detected threshold too high ({detected_threshold:.2f} N), using config value ({onset_threshold:.2f} N)"
                    )
            else:
                print(
                    f"Auto-detection failed, using config onset threshold: {onset_threshold:.2f} N"
                )
        else:
            print(f"Using config onset threshold: {onset_threshold:.2f} N")

        # Find baseline (seated) force level - use 10th percentile for robustness
        baseline_force = np.percentile(force_data, 10)
        print(f"Baseline force (seated): {baseline_force:.2f} N")
        print(f"Force data range: {np.min(force_data):.2f} - {np.max(force_data):.2f} N")
        print(f"Onset threshold: {onset_threshold:.2f} N")
        print(f"Detection threshold (baseline + onset): {baseline_force + onset_threshold:.2f} N")

        # Detect movement onset (when force exceeds baseline + onset_threshold)
        onset_indices = np.where(force_data > baseline_force + onset_threshold)[0]
        print(f"Found {len(onset_indices)} onset indices")

        if len(onset_indices) == 0:
            print("No movement onset detected - check threshold settings")
            print("Try lowering onset_threshold or check if force data is correct")
            return phases

        # Find continuous segments representing distinct sit-to-stand movements
        movement_segments = []
        current_segment = [onset_indices[0]]

        # Estimate sampling frequency
        if len(time_data) > 1:
            dt = np.mean(np.diff(time_data))
            sampling_freq = 1.0 / dt if dt > 0 else 100.0
        else:
            sampling_freq = 100.0

        min_samples = int(min_duration * sampling_freq)
        print(f"Minimum samples for valid phase: {min_samples} (at {sampling_freq:.1f} Hz)")

        for i in range(1, len(onset_indices)):
            if onset_indices[i] == onset_indices[i - 1] + 1:
                current_segment.append(onset_indices[i])
            else:
                if len(current_segment) >= min_samples:
                    movement_segments.append(current_segment)
                current_segment = [onset_indices[i]]

        # Add the last segment if valid
        if len(current_segment) >= min_samples:
            movement_segments.append(current_segment)

        print(f"Detected {len(movement_segments)} potential sit-to-stand movement(s)")

        # Convert segments to phase information with comprehensive biomechanical metrics
        for phase_num, segment in enumerate(movement_segments):
            start_idx = segment[0]
            end_idx = segment[-1]

            # Extract segment data
            segment_forces = force_data[start_idx : end_idx + 1]
            segment_times = time_data[start_idx : end_idx + 1]

            # === PEAK DETECTION ===
            # Find ALL peaks in this segment (important for multi-peak movements in CP)
            all_peaks = detect_all_peaks_in_segment(segment_forces, segment_times, baseline_force)

            # Identify first peak (clinically important for initial force generation)
            first_peak = all_peaks[0] if all_peaks else None

            # Identify maximum peak
            max_peak = max(all_peaks, key=lambda p: p["force"]) if all_peaks else None

            # Global peak in segment (for compatibility)
            peak_idx = start_idx + np.argmax(segment_forces)
            peak_force = force_data[peak_idx]
            peak_time = time_data[peak_idx]

            # === TEMPORAL METRICS ===
            phase_duration = time_data[end_idx] - time_data[start_idx]

            # Time to first peak (onset to first peak) - Critical for CP assessment
            time_to_first_peak = first_peak["time"] - segment_times[0] if first_peak else None

            # Time to maximum force
            time_to_max_force = peak_time - segment_times[0]

            # === FORCE DEVELOPMENT METRICS ===
            # Rate of Force Development (RFD) - multiple methods

            # 1. Overall RFD (onset to peak)
            force_at_onset = segment_forces[0]
            overall_rfd = (
                (peak_force - force_at_onset) / time_to_max_force if time_to_max_force > 0 else 0
            )

            # 2. Early RFD (first 100ms or 10% of movement, whichever is shorter)
            early_window_time = min(0.1, phase_duration * 0.1)  # 100ms or 10%
            early_window_samples = int(early_window_time * sampling_freq)
            if early_window_samples > 1:
                early_force_change = segment_forces[early_window_samples] - segment_forces[0]
                early_rfd = early_force_change / early_window_time
            else:
                early_rfd = 0

            # 3. Peak RFD (maximum instantaneous rate)
            force_gradient = np.gradient(segment_forces, segment_times)
            peak_rfd = np.max(force_gradient)
            peak_rfd_time = segment_times[np.argmax(force_gradient)]

            # === IMPULSE METRICS ===
            # Total impulse (force-time integral)
            total_impulse = np.trapz(segment_forces, segment_times)

            # Impulse above baseline (more clinically relevant)
            impulse_above_baseline = np.trapz(segment_forces - baseline_force, segment_times)

            # Normalized impulse (per unit time)
            normalized_impulse = total_impulse / phase_duration if phase_duration > 0 else 0

            # === POWER METRICS ===
            # Average power
            average_power = total_impulse / phase_duration if phase_duration > 0 else 0

            # Peak power (at point of maximum force)
            peak_power = peak_force * peak_rfd if peak_rfd > 0 else 0

            # === FORCE VARIABILITY ===
            # Coefficient of variation of force during movement
            force_cv = (
                (np.std(segment_forces) / np.mean(segment_forces)) * 100
                if np.mean(segment_forces) > 0
                else 0
            )

            # Force smoothness (using jerk-like metric)
            force_jerk = np.mean(np.abs(np.gradient(force_gradient, segment_times)))

            # === SYMMETRY AND CONSISTENCY ===
            # Multiple peak analysis for bilateral assessment
            symmetry_metrics = calculate_detailed_symmetry(all_peaks, segment_forces, segment_times)

            # === CLINICAL METRICS ===
            # Weight transfer efficiency (how quickly force reaches threshold)
            weight_transfer_time = None
            threshold_idx = np.where(segment_forces >= (baseline_force + onset_threshold))[0]
            if len(threshold_idx) > 0:
                weight_transfer_time = segment_times[threshold_idx[0]] - segment_times[0]

            # Unloading phase (before standing) - negative force development
            unloading_indices = np.where(force_gradient < 0)[0]
            has_unloading = len(unloading_indices) > 0

            # Store comprehensive phase metrics
            phases.append(
                {
                    # Basic temporal metrics
                    "phase_number": phase_num + 1,
                    "start_time": time_data[start_idx],
                    "end_time": time_data[end_idx],
                    "duration": phase_duration,
                    "start_index": start_idx,
                    "end_index": end_idx,
                    # Peak force metrics
                    "peak_force": peak_force,
                    "peak_time": peak_time,
                    "first_peak_force": first_peak["force"] if first_peak else peak_force,
                    "first_peak_time": first_peak["time"] if first_peak else peak_time,
                    "max_peak_force": max_peak["force"] if max_peak else peak_force,
                    "max_peak_time": max_peak["time"] if max_peak else peak_time,
                    "num_peaks": len(all_peaks),
                    "all_peaks": all_peaks,
                    # Time to peak metrics (CRITICAL for CP assessment)
                    "time_to_first_peak": time_to_first_peak,
                    "time_to_max_force": time_to_max_force,
                    "weight_transfer_time": weight_transfer_time,
                    # Rate of Force Development (RFD) - CRITICAL for CP assessment
                    "overall_rfd": overall_rfd,
                    "early_rfd": early_rfd,
                    "peak_rfd": peak_rfd,
                    "peak_rfd_time": peak_rfd_time,
                    "rate_of_force_development": overall_rfd,  # Legacy compatibility
                    # Impulse metrics
                    "impulse": total_impulse,
                    "impulse_above_baseline": impulse_above_baseline,
                    "normalized_impulse": normalized_impulse,
                    # Power metrics
                    "average_power": average_power,
                    "peak_power": peak_power,
                    "power": average_power,  # Legacy compatibility
                    # Force variability and smoothness
                    "force_cv": force_cv,
                    "force_jerk": force_jerk,
                    # Symmetry metrics
                    "symmetry": symmetry_metrics["overall_symmetry"],
                    "peak_symmetry": symmetry_metrics["peak_symmetry"],
                    "temporal_symmetry": symmetry_metrics["temporal_symmetry"],
                    # Clinical indicators
                    "baseline_force": baseline_force,
                    "onset_threshold": onset_threshold,
                    "force_at_onset": force_at_onset,
                    "has_unloading_phase": has_unloading,
                    # Additional biomechanical metrics
                    "force_range": peak_force - force_at_onset,
                    "force_excursion": np.max(segment_forces) - np.min(segment_forces),
                    "mean_force_during_movement": np.mean(segment_forces),
                    "sampling_frequency": sampling_freq,
                }
            )

    except Exception as e:
        print(f"Error detecting sit-to-stand phases: {str(e)}")
        import traceback

        traceback.print_exc()

    return phases


def detect_ascending_threshold(force_data, time_data):
    """
    Automatically detects threshold based on ascending phase of sit-to-stand movement.

    Parameters:
    -----------
    force_data : array-like
        Force values
    time_data : array-like
        Time values

    Returns:
    --------
    float or None
        Detected threshold value or None if not detected
    """
    try:
        # Find baseline (first 10% of data)
        baseline_force = np.percentile(force_data, 10)

        # Find the ascending phase (where force starts increasing significantly)
        # Look for the point where force derivative is maximum
        force_derivative = np.gradient(force_data, time_data)

        # Find the maximum derivative in the first half of the data
        half_point = len(force_derivative) // 2
        max_derivative_idx = np.argmax(force_derivative[:half_point])

        if max_derivative_idx > 0:
            # Set threshold as baseline + 20% of the force increase at max derivative point
            force_at_max_deriv = force_data[max_derivative_idx]
            threshold = baseline_force + 0.2 * (force_at_max_deriv - baseline_force)
            return threshold

    except Exception as e:
        print(f"Error in automatic threshold detection: {str(e)}")

    return None


def detect_peaks_scipy(force_data, time_data, config):
    """
    Detects peaks using scipy.signal.find_peaks with configurable parameters.

    Parameters:
    -----------
    force_data : array-like
        Force values
    time_data : array-like
        Time values
    config : dict
        Configuration dictionary with peak detection parameters

    Returns:
    --------
    list
        List of detected peaks with time, force, and index information
    """
    try:
        peak_params = config.get("detection", {}).get("peak_detection", {})

        # Extract peak detection parameters
        height = peak_params.get("height", None)
        threshold = peak_params.get("threshold", None)
        distance = peak_params.get("distance", 10)
        prominence = peak_params.get("prominence", 5.0)
        width = peak_params.get("width", None)
        rel_height = peak_params.get("rel_height", 0.5)

        # Find peaks using scipy
        peaks, properties = find_peaks(
            force_data,
            height=height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
            width=width,
            rel_height=rel_height,
        )

        # Convert to list of dictionaries
        peak_list = []
        for i, peak_idx in enumerate(peaks):
            peak_info = {
                "index": int(peak_idx),
                "time": float(time_data[peak_idx]),
                "force": float(force_data[peak_idx]),
                "prominence": float(properties["prominences"][i])
                if "prominences" in properties
                else 0.0,
                "width": float(properties["widths"][i]) if "widths" in properties else 0.0,
                "height": float(properties["peak_heights"][i])
                if "peak_heights" in properties
                else force_data[peak_idx],
            }
            peak_list.append(peak_info)

        print(f"Detected {len(peak_list)} peaks using scipy.find_peaks")
        return peak_list

    except Exception as e:
        print(f"Error in scipy peak detection: {str(e)}")
        # Fallback to simple peak detection
        return detect_all_peaks_in_segment(force_data, time_data, np.percentile(force_data, 10))


def detect_all_peaks_in_segment(forces, times, baseline_force, min_prominence=5.0):
    """
    Detects all significant peaks in a sit-to-stand segment with enhanced filtering.
    Critical for identifying multiple force peaks in cerebral palsy patients.

    Parameters:
    -----------
    forces : array-like
        Force values in the segment
    times : array-like
        Time values in the segment
    baseline_force : float
        Baseline force level for reference
    min_prominence : float
        Minimum peak prominence (height above surrounding valleys)

    Returns:
    --------
    list
        List of detected peaks with time, force, and index information
    """
    try:
        peaks = []

        # Method 1: Local maxima detection with prominence check
        for i in range(1, len(forces) - 1):
            is_local_max = forces[i] > forces[i - 1] and forces[i] > forces[i + 1]

            if is_local_max:
                # Calculate prominence (height above nearby valleys)
                left_valley = np.min(forces[max(0, i - 10) : i]) if i > 0 else forces[i]
                right_valley = (
                    np.min(forces[i : min(len(forces), i + 10)])
                    if i < len(forces) - 1
                    else forces[i]
                )
                prominence = forces[i] - max(left_valley, right_valley)

                # Only include peaks with sufficient prominence and above baseline
                if prominence >= min_prominence and forces[i] > baseline_force + min_prominence:
                    peaks.append(
                        {
                            "time": times[i],
                            "force": forces[i],
                            "index": i,
                            "prominence": prominence,
                            "above_baseline": forces[i] - baseline_force,
                        }
                    )

        # If no peaks found with prominence filter, find at least the maximum
        if not peaks:
            max_idx = np.argmax(forces)
            peaks.append(
                {
                    "time": times[max_idx],
                    "force": forces[max_idx],
                    "index": max_idx,
                    "prominence": forces[max_idx] - np.min(forces),
                    "above_baseline": forces[max_idx] - baseline_force,
                }
            )

        # Sort peaks by time (chronological order)
        peaks = sorted(peaks, key=lambda x: x["time"])

        return peaks

    except Exception as e:
        print(f"Error detecting peaks in segment: {str(e)}")
        # Return at least the maximum as fallback
        max_idx = np.argmax(forces)
        return [
            {
                "time": times[max_idx],
                "force": forces[max_idx],
                "index": max_idx,
                "prominence": 0,
                "above_baseline": forces[max_idx] - baseline_force,
            }
        ]


def find_peaks_in_segment(forces, times):
    """Legacy function - kept for compatibility. Use detect_all_peaks_in_segment for new code."""
    try:
        # Simple peak detection - find local maxima
        peaks = []
        for i in range(1, len(forces) - 1):
            if forces[i] > forces[i - 1] and forces[i] > forces[i + 1]:
                peaks.append({"time": times[i], "force": forces[i], "index": i})

        return sorted(peaks, key=lambda x: x["force"], reverse=True)

    except Exception as e:
        print(f"Error finding peaks in segment: {str(e)}")
        return []


def calculate_energy_expenditure(force_data, time_data, phases, config):
    """
    Calculates energy expenditure for sit-to-stand movements based on force data and body weight.
    Uses both mechanical work and metabolic energy calculations.

    Parameters:
    -----------
    force_data : array-like
        Force values in Newtons
    time_data : array-like
        Time values in seconds
    phases : list
        List of detected sit-to-stand phases
    config : dict
        Configuration dictionary with body weight and other parameters

    Returns:
    --------
    dict
        Energy expenditure metrics including mechanical work and metabolic energy
    """
    try:
        body_weight = config.get("analysis", {}).get("body_weight", 70.0)  # kg
        gravity = 9.81  # m/s²

        # Calculate body weight in Newtons
        body_weight_N = body_weight * gravity  # noqa: N806 - Scientific notation convention

        energy_metrics = {
            "body_weight_kg": body_weight,
            "body_weight_N": body_weight_N,
            "total_mechanical_work_J": 0.0,
            "total_metabolic_energy_kcal": 0.0,
            "average_energy_per_movement_kcal": 0.0,
            "energy_per_movement_J": 0.0,
            "energy_efficiency": 0.0,
            "phases_energy": [],
        }

        if not phases:
            return energy_metrics

        total_mechanical_work = 0.0
        total_metabolic_energy = 0.0

        # Typical chair height for sit-to-stand (0.45-0.50m)
        # Using average of 0.475m as reference
        chair_height = 0.475  # meters

        for i, phase in enumerate(phases):
            # Extract phase data
            start_idx = phase["start_index"]
            end_idx = phase["end_index"]
            phase_forces = force_data[start_idx : end_idx + 1]
            phase_times = time_data[start_idx : end_idx + 1]
            phase_duration = phase["duration"]

            # === MECHANICAL WORK CALCULATION ===
            # Method 1: Force-time integral (impulse)
            impulse = np.trapz(phase_forces, phase_times)  # N⋅s

            # Method 2: Work = Force × Distance (assuming vertical movement)
            # Average force during movement
            avg_force = np.mean(phase_forces)
            # Work = average_force × chair_height
            mechanical_work = avg_force * chair_height  # Joules

            # Method 3: Net work above body weight
            net_force_above_weight = avg_force - body_weight_N
            net_work = max(0, net_force_above_weight) * chair_height  # Only positive work

            # === METABOLIC ENERGY CALCULATION ===
            # Based on Nakagata et al. (2019) study:
            # - STS-slow: 0.37 ± 0.12 kcal per movement
            # - STS-normal: 0.26 ± 0.06 kcal per movement

            # Determine movement speed based on duration
            if phase_duration > 2.0:  # Slow movement (>2s)
                metabolic_energy_per_movement = 0.37  # kcal
                movement_type = "slow"
            else:  # Normal movement (≤2s)
                metabolic_energy_per_movement = 0.26  # kcal
                movement_type = "normal"

            # Adjust for body weight (study was with ~64kg subjects)
            reference_weight = 64.0  # kg from the study
            weight_factor = body_weight / reference_weight
            adjusted_metabolic_energy = metabolic_energy_per_movement * weight_factor

            # === ENERGY EFFICIENCY ===
            # Efficiency = Mechanical work / Metabolic energy
            mechanical_work_kcal = mechanical_work / 4184  # Convert J to kcal
            efficiency = (
                (mechanical_work_kcal / adjusted_metabolic_energy) * 100
                if adjusted_metabolic_energy > 0
                else 0
            )

            # Store phase energy data
            phase_energy = {
                "phase_number": i + 1,
                "movement_type": movement_type,
                "duration_s": phase_duration,
                "mechanical_work_J": mechanical_work,
                "mechanical_work_kcal": mechanical_work_kcal,
                "net_work_J": net_work,
                "impulse_Ns": impulse,
                "average_force_N": avg_force,
                "metabolic_energy_kcal": adjusted_metabolic_energy,
                "energy_efficiency_percent": efficiency,
                "force_above_body_weight_N": net_force_above_weight,
            }

            energy_metrics["phases_energy"].append(phase_energy)
            total_mechanical_work += mechanical_work
            total_metabolic_energy += adjusted_metabolic_energy

        # Calculate total energy metrics
        energy_metrics["total_mechanical_work_J"] = total_mechanical_work
        energy_metrics["total_metabolic_energy_kcal"] = total_metabolic_energy
        energy_metrics["average_energy_per_movement_kcal"] = total_metabolic_energy / len(phases)
        energy_metrics["energy_per_movement_J"] = total_mechanical_work / len(phases)

        # Overall energy efficiency
        total_mechanical_work_kcal = total_mechanical_work / 4184
        energy_metrics["energy_efficiency"] = (
            (total_mechanical_work_kcal / total_metabolic_energy) * 100
            if total_metabolic_energy > 0
            else 0
        )

        # Additional metrics
        energy_metrics["total_movements"] = len(phases)
        energy_metrics["chair_height_m"] = chair_height
        energy_metrics["reference_study"] = "Nakagata et al. (2019) - PMC6473689"

        print("Energy Expenditure Analysis:")
        print(f"  Body Weight: {body_weight:.1f} kg ({body_weight_N:.1f} N)")
        print(f"  Total Movements: {len(phases)}")
        print(
            f"  Total Mechanical Work: {total_mechanical_work:.2f} J ({total_mechanical_work_kcal:.3f} kcal)"
        )
        print(f"  Total Metabolic Energy: {total_metabolic_energy:.3f} kcal")
        print(
            f"  Average per Movement: {energy_metrics['average_energy_per_movement_kcal']:.3f} kcal"
        )
        print(f"  Energy Efficiency: {energy_metrics['energy_efficiency']:.1f}%")

        return energy_metrics

    except Exception as e:
        print(f"Error calculating energy expenditure: {str(e)}")
        return {
            "body_weight_kg": 0.0,
            "body_weight_N": 0.0,
            "total_mechanical_work_J": 0.0,
            "total_metabolic_energy_kcal": 0.0,
            "average_energy_per_movement_kcal": 0.0,
            "energy_per_movement_J": 0.0,
            "energy_efficiency": 0.0,
            "error": str(e),
        }


def calculate_stability_index(force_data, time_data, reference_peak_time, config):
    """
    Calculates stability index based on oscillation around the reference peak value.
    Uses the reference peak force (typically the maximum peak) as reference and measures how the signal oscillates around it.

    Parameters:
    -----------
    force_data : array-like
        Force values
    time_data : array-like
        Time values
    reference_peak_time : float
        Time of the reference peak (typically maximum peak)
    config : dict
        Configuration dictionary with stability parameters

    Returns:
    --------
    dict
        Stability metrics including oscillation counts and deviation from reference peak
    """
    try:
        stability_config = config.get("stability", {})
        baseline_window = stability_config.get("baseline_window", 0.5)
        stability_threshold = stability_config.get("stability_threshold", 2.0)
        noise_analysis = stability_config.get("noise_analysis", True)
        rolling_window = stability_config.get("rolling_window", 0.1)

        # Find the index corresponding to reference peak time
        reference_peak_idx = np.argmin(np.abs(time_data - reference_peak_time))

        # Get the reference peak force value (this is our reference)
        reference_peak_force = force_data[reference_peak_idx]

        # Define the stability period (from reference peak to end)
        stability_start_idx = reference_peak_idx
        stability_end_idx = len(force_data)

        if stability_end_idx <= stability_start_idx:
            return {
                "stability_index": 0.0,
                "mean_deviation": 0.0,
                "max_deviation": 0.0,
                "noise_level": 0.0,
                "oscillation_frequency": 0.0,
                "stability_duration": 0.0,
                "is_stable": False,
                "first_peak_force": 0.0,
                "crossings_above": 0,
                "crossings_below": 0,
                "total_crossings": 0,
                "percent_above": 0.0,
                "percent_below": 0.0,
            }

        # Extract stability period data (from reference peak to end)
        stability_forces = force_data[stability_start_idx:stability_end_idx]
        stability_times = time_data[stability_start_idx:stability_end_idx]

        # Calculate RFD to find when standing phase actually starts
        # Standing phase = when RFD approaches zero (movement finished)
        # Note: force_rate calculated but not used in current implementation
        _force_rate = np.gradient(stability_forces, stability_times)

        # Find where RFD stabilizes (approaches zero)
        # Use baseline_window to skip initial transition period
        baseline_samples = int(
            baseline_window
            * len(stability_forces)
            / (stability_times[-1] - stability_times[0] + 1e-6)
        )
        baseline_samples = max(baseline_samples, 100)  # At least 100 samples
        baseline_samples = min(
            baseline_samples, len(stability_forces) // 2
        )  # At most half the data

        # Standing phase starts after baseline_window
        standing_start_idx = baseline_samples
        standing_forces = stability_forces[standing_start_idx:]
        standing_times = stability_times[standing_start_idx:]

        print(
            f"Standing phase analysis: from {standing_times[0]:.3f}s to {standing_times[-1]:.3f}s"
        )
        print(f"  Standing phase duration: {standing_times[-1] - standing_times[0]:.3f}s")
        print(f"  Standing phase samples: {len(standing_forces)}")

        # Detect peaks ONLY in the standing baseline (after movement stabilized)
        standing_peaks = []
        try:
            from scipy.signal import find_peaks

            peak_params = config.get("detection", {}).get("peak_detection", {})

            # Use same parameters from config but applied to standing phase only
            height = peak_params.get("height", None)
            distance = peak_params.get("distance", 10)
            prominence = peak_params.get("prominence", 5.0)

            # Find peaks in standing phase ONLY (after stabilization)
            peak_indices, properties = find_peaks(
                standing_forces, height=height, distance=distance, prominence=prominence
            )

            # Store peak information with absolute time and force values
            for i, peak_idx in enumerate(peak_indices):
                abs_idx = stability_start_idx + standing_start_idx + peak_idx
                standing_peaks.append(
                    {
                        "time": float(standing_times[peak_idx]),
                        "force": float(standing_forces[peak_idx]),
                        "index": int(abs_idx),  # Absolute index in original data
                        "prominence": float(properties["prominences"][i])
                        if "prominences" in properties
                        else 0.0,
                    }
                )

            print(
                f"Detected {len(standing_peaks)} peaks in standing baseline (after stabilization)"
            )
            if len(standing_peaks) > 0:
                print(f"  Standing peaks times: {[f'{p["time"]:.3f}s' for p in standing_peaks]}")
                print(f"  Standing peaks forces: {[f'{p["force"]:.2f}N' for p in standing_peaks]}")
        except Exception as e:
            print(f"Error detecting standing peaks: {str(e)}")

        # Calculate deviations from reference peak force (NOT from mean)
        deviations = stability_forces - reference_peak_force
        abs_deviations = np.abs(deviations)
        mean_deviation = np.mean(abs_deviations)
        max_deviation = np.max(abs_deviations)

        # Count how many times signal is above or below reference peak
        points_above = np.sum(stability_forces > reference_peak_force)
        points_below = np.sum(stability_forces < reference_peak_force)
        total_points = len(stability_forces)

        percent_above = (points_above / total_points) * 100 if total_points > 0 else 0
        percent_below = (points_below / total_points) * 100 if total_points > 0 else 0

        # Count crossings (transitions above/below the reference peak value)
        crossings = np.where(np.diff(np.signbit(deviations)))[0]
        total_crossings = len(crossings)

        # Separate upward and downward crossings
        crossings_above = 0  # Crosses from below to above
        crossings_below = 0  # Crosses from above to below

        for crossing_idx in crossings:
            if crossing_idx + 1 < len(deviations):
                if deviations[crossing_idx] < 0 and deviations[crossing_idx + 1] >= 0:
                    crossings_above += 1
                elif deviations[crossing_idx] >= 0 and deviations[crossing_idx + 1] < 0:
                    crossings_below += 1

        # Calculate stability index (0-1 scale, higher = more stable)
        # Based on how much the signal deviates from the reference peak value
        stability_index = max(0.0, 1.0 - (mean_deviation / stability_threshold))

        # Determine if standing is stable (small deviations from reference peak)
        is_stable = mean_deviation <= stability_threshold

        # Noise analysis if enabled
        noise_level = 0.0
        oscillation_frequency = 0.0

        if noise_analysis and len(stability_forces) > 1:
            # Calculate rolling standard deviation as noise measure
            window_samples = int(
                rolling_window * len(stability_forces) / (stability_times[-1] - stability_times[0])
            )
            if window_samples > 1:
                rolling_std = []
                for i in range(len(stability_forces) - window_samples + 1):
                    window_data = stability_forces[i : i + window_samples]
                    rolling_std.append(np.std(window_data))

                noise_level = np.mean(rolling_std) if rolling_std else 0.0

                # Estimate oscillation frequency from crossings
                if len(crossings) > 1 and (stability_times[-1] - stability_times[0]) > 0:
                    oscillation_frequency = len(crossings) / (
                        2 * (stability_times[-1] - stability_times[0])
                    )

        stability_duration = stability_times[-1] - stability_times[0]

        print(f"Stability Analysis (Reference: Peak = {reference_peak_force:.2f} N):")
        print(f"  Duration: {stability_duration:.2f} s")
        print(f"  Mean Deviation from Reference Peak: {mean_deviation:.2f} N")
        print(f"  Max Deviation from Reference Peak: {max_deviation:.2f} N")
        print(f"  Points Above Reference Peak: {points_above} ({percent_above:.1f}%)")
        print(f"  Points Below Reference Peak: {points_below} ({percent_below:.1f}%)")
        print(f"  Total Crossings: {total_crossings} (↑{crossings_above}, ↓{crossings_below})")
        print(f"  Stability Index: {stability_index:.3f}")
        print(f"  Is Stable: {is_stable}")
        print(f"  Noise Level: {noise_level:.2f} N")
        print(f"  Oscillation Frequency: {oscillation_frequency:.2f} Hz")

        return {
            "stability_index": float(stability_index),
            "mean_deviation": float(mean_deviation),
            "max_deviation": float(max_deviation),
            "noise_level": float(noise_level),
            "oscillation_frequency": float(oscillation_frequency),
            "stability_duration": float(stability_duration),
            "is_stable": bool(is_stable),
            "first_peak_force": float(reference_peak_force),
            "stability_threshold": float(stability_threshold),
            "stability_start_time": float(stability_times[0]),
            "stability_end_time": float(stability_times[-1]),
            "crossings_above": int(crossings_above),
            "crossings_below": int(crossings_below),
            "total_crossings": int(total_crossings),
            "percent_above": float(percent_above),
            "percent_below": float(percent_below),
            "points_above": int(points_above),
            "points_below": int(points_below),
            "standing_peaks": standing_peaks,  # List of peaks detected in standing baseline
            "num_standing_peaks": len(standing_peaks),
        }

    except Exception as e:
        print(f"Error calculating stability index: {str(e)}")
        return {
            "stability_index": 0.0,
            "mean_deviation": 0.0,
            "max_deviation": 0.0,
            "noise_level": 0.0,
            "oscillation_frequency": 0.0,
            "stability_duration": 0.0,
            "is_stable": False,
            "first_peak_force": 0.0,
            "crossings_above": 0,
            "crossings_below": 0,
            "total_crossings": 0,
            "percent_above": 0.0,
            "percent_below": 0.0,
            "standing_peaks": [],
            "num_standing_peaks": 0,
            "error": str(e),
        }


def calculate_detailed_symmetry(all_peaks, forces, times):
    """
    Calculates comprehensive symmetry metrics for bilateral movement assessment.
    Important for evaluating compensatory strategies in cerebral palsy.

    Parameters:
    -----------
    all_peaks : list
        List of all detected peaks with timing and force information
    forces : array-like
        Force values for the entire segment
    times : array-like
        Time values for the entire segment

    Returns:
    --------
    dict
        Comprehensive symmetry metrics
    """
    metrics = {
        "overall_symmetry": 1.0,
        "peak_symmetry": 1.0,
        "temporal_symmetry": 1.0,
        "bilateral_index": 0.0,
        "consistency_score": 1.0,
    }

    if len(all_peaks) < 2:
        return metrics  # Perfect symmetry if only one peak

    try:
        # Extract peak characteristics
        peak_forces = [peak["force"] for peak in all_peaks]
        peak_times = [peak["time"] for peak in all_peaks]

        # === PEAK FORCE SYMMETRY ===
        # Coefficient of variation of peak forces (lower = more symmetric)
        peak_force_cv = (
            (np.std(peak_forces) / np.mean(peak_forces)) * 100 if np.mean(peak_forces) > 0 else 0
        )
        peak_symmetry = 1.0 - min(peak_force_cv / 100.0, 1.0)  # Convert to 0-1 scale

        # === TEMPORAL SYMMETRY ===
        # Time distribution of peaks
        time_range = max(peak_times) - min(peak_times)
        if time_range > 0 and len(peak_times) > 1:
            actual_spacings = [
                peak_times[i + 1] - peak_times[i] for i in range(len(peak_times) - 1)
            ]
            time_cv = (
                (np.std(actual_spacings) / np.mean(actual_spacings)) * 100
                if np.mean(actual_spacings) > 0
                else 0
            )
            temporal_symmetry = 1.0 - min(time_cv / 100.0, 1.0)
        else:
            temporal_symmetry = 1.0

        # === BILATERAL INDEX ===
        # For sit-to-stand, assess if there are two peaks (bilateral loading pattern)
        if len(all_peaks) == 2:
            # Ideal bilateral pattern
            force_ratio = min(peak_forces) / max(peak_forces) if max(peak_forces) > 0 else 0
            bilateral_index = force_ratio  # 1.0 = perfect bilateral symmetry
        elif len(all_peaks) == 1:
            bilateral_index = 0.5  # Single peak might indicate asymmetry
        else:
            bilateral_index = 0.3  # Multiple peaks might indicate instability

        # === OVERALL SYMMETRY ===
        # Combined score weighting clinical relevance
        overall_symmetry = peak_symmetry * 0.4 + temporal_symmetry * 0.3 + bilateral_index * 0.3

        # === CONSISTENCY SCORE ===
        # How smooth and consistent is the force profile
        force_gradient = np.gradient(forces, times)
        gradient_variability = np.std(force_gradient) / (np.mean(np.abs(force_gradient)) + 1e-6)
        consistency_score = 1.0 / (1.0 + gradient_variability)  # Higher = more consistent

        metrics["overall_symmetry"] = overall_symmetry
        metrics["peak_symmetry"] = peak_symmetry
        metrics["temporal_symmetry"] = temporal_symmetry
        metrics["bilateral_index"] = bilateral_index
        metrics["consistency_score"] = consistency_score

    except Exception as e:
        print(f"Error calculating detailed symmetry: {str(e)}")

    return metrics


def calculate_symmetry(peaks):
    """
    Legacy symmetry calculation - kept for backward compatibility.
    For new code, use calculate_detailed_symmetry which provides more clinical metrics.
    """
    if len(peaks) < 2:
        return 1.0  # Perfect symmetry if only one peak

    try:
        # Calculate symmetry based on timing and force distribution
        times = [peak["time"] for peak in peaks]
        forces = [peak["force"] for peak in peaks]

        # Time-based symmetry (how evenly distributed in time)
        time_range = max(times) - min(times)
        if time_range > 0:
            expected_spacing = time_range / (len(times) - 1)
            actual_spacing = [times[i + 1] - times[i] for i in range(len(times) - 1)]
            time_variance = (
                np.var(actual_spacing) / (expected_spacing**2) if expected_spacing > 0 else 0
            )
        else:
            time_variance = 0

        # Force-based symmetry (how similar are the peak forces)
        force_variance = np.var(forces) / (np.mean(forces) ** 2) if np.mean(forces) > 0 else 0

        # Combined symmetry index (lower variance = higher symmetry)
        symmetry = 1.0 - (time_variance + force_variance) / 2.0

        return max(0.0, min(1.0, symmetry))  # Clamp between 0 and 1

    except Exception as e:
        print(f"Error calculating symmetry: {str(e)}")
        return 0.5


def calculate_movement_metrics(data, phases):
    """
    Calculates comprehensive movement-specific metrics from detected phases.
    Enhanced for pediatric cerebral palsy clinical assessment.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns
    phases : list
        List of detected sit-to-stand phases with enhanced metrics

    Returns:
    --------
    dict
        Movement metrics including time to peak, RFD, and clinical quality indicators
    """
    metrics = {
        "num_phases": len(phases),
        "total_movement_time": 0,
        "average_phase_duration": 0,
        "phases_per_minute": 0,
        "symmetry_index": None,
        "average_time_to_peak": 0,
        "time_to_peak_variation": 0,
        "average_rate_of_force_development": 0,
        "peak_force_consistency": 0,
        "movement_efficiency": 0,
        "average_overall_rfd": 0,
        "average_early_rfd": 0,
        "average_peak_rfd": 0,
        "average_bilateral_index": 0,
        "average_consistency_score": 0,
    }

    if not phases:
        return metrics

    try:
        # Calculate basic temporal metrics
        durations = [phase["duration"] for phase in phases]

        # Time to peak metrics (handle both new and legacy format)
        time_to_max = [
            phase.get("time_to_max_force", phase.get("time_to_peak", 0)) for phase in phases
        ]

        peak_forces = [phase["peak_force"] for phase in phases]
        rates_of_force_dev = [
            phase.get("overall_rfd", phase.get("rate_of_force_development", 0)) for phase in phases
        ]

        metrics["total_movement_time"] = sum(durations)
        metrics["average_phase_duration"] = np.mean(durations) if durations else 0
        metrics["average_time_to_peak"] = np.mean(time_to_max) if time_to_max else 0
        metrics["time_to_peak_variation"] = np.std(time_to_max) if len(time_to_max) > 1 else 0
        metrics["average_rate_of_force_development"] = (
            np.mean(rates_of_force_dev) if rates_of_force_dev else 0
        )

        # Enhanced RFD metrics
        overall_rfds = [phase.get("overall_rfd", 0) for phase in phases]
        early_rfds = [phase.get("early_rfd", 0) for phase in phases]
        peak_rfds = [phase.get("peak_rfd", 0) for phase in phases]

        metrics["average_overall_rfd"] = np.mean(overall_rfds) if overall_rfds else 0
        metrics["average_early_rfd"] = np.mean(early_rfds) if early_rfds else 0
        metrics["average_peak_rfd"] = np.mean(peak_rfds) if peak_rfds else 0

        # Calculate phases per minute if duration > 0
        total_duration = data["Time"].iloc[-1] - data["Time"].iloc[0]
        if total_duration > 0:
            metrics["phases_per_minute"] = (len(phases) / total_duration) * 60

        # Calculate peak force consistency
        if len(peak_forces) > 1:
            metrics["peak_force_consistency"] = 1 - (
                np.std(peak_forces) / (np.mean(peak_forces) + 1e-6)
            )

        # Calculate symmetry using enhanced method
        symmetry_values = [
            phase.get("symmetry", phase.get("overall_symmetry", 1.0)) for phase in phases
        ]
        bilateral_indices = [phase.get("bilateral_index", 0.5) for phase in phases]
        consistency_scores = [phase.get("consistency_score", 1.0) for phase in phases]

        metrics["symmetry_index"] = np.mean(symmetry_values) if symmetry_values else None
        metrics["average_bilateral_index"] = np.mean(bilateral_indices) if bilateral_indices else 0
        metrics["average_consistency_score"] = (
            np.mean(consistency_scores) if consistency_scores else 0
        )

        # Calculate movement efficiency (time to peak / total duration ratio)
        if metrics["total_movement_time"] > 0:
            metrics["movement_efficiency"] = (
                metrics["average_time_to_peak"] / metrics["total_movement_time"]
            )

    except Exception as e:
        print(f"Error calculating movement metrics: {str(e)}")
        import traceback

        traceback.print_exc()

    return metrics


def calculate_impulse_metrics(data, phases):
    """
    Calculates impulse and power-related metrics.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns
    phases : list
        List of detected sit-to-stand phases

    Returns:
    --------
    dict
        Impulse and power metrics
    """
    metrics = {
        "total_impulse": 0,
        "average_impulse": 0,
        "peak_power": 0,
        "average_power": 0,
        "force_rate_of_change": 0,
    }

    if not phases:
        return metrics

    # Calculate impulse metrics
    impulses = [phase["impulse"] for phase in phases]
    metrics["total_impulse"] = sum(impulses)
    metrics["average_impulse"] = np.mean(impulses)

    # Calculate power metrics (simplified)
    for phase in phases:
        duration = phase["duration"]
        if duration > 0:
            power = phase["impulse"] / duration
            metrics["peak_power"] = max(metrics["peak_power"], power)

    metrics["average_power"] = metrics["total_impulse"] / sum([p["duration"] for p in phases])

    # Calculate rate of force development (first phase only)
    if phases:
        first_phase = phases[0]
        force_change = first_phase["peak_force"] - data["Force"].iloc[first_phase["start_index"]]
        time_change = first_phase["peak_time"] - data["Time"].iloc[first_phase["start_index"]]
        if time_change > 0:
            metrics["force_rate_of_change"] = force_change / time_change

    return metrics


def calculate_time_to_peak_metrics(data, phases):
    """
    Calculates time to peak metrics for sit-to-stand phases.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns
    phases : list
        List of detected sit-to-stand phases

    Returns:
    --------
    dict
        Time to peak metrics
    """
    metrics = {
        "time_to_first_peak": None,
        "time_to_max_force": None,
        "average_time_to_peak": 0,
        "time_to_peak_variation": 0,
        "peak_timing_consistency": 0,
    }

    if not phases:
        return metrics

    try:
        # Calculate time to first peak (from movement onset)
        time_to_peaks = []
        for phase in phases:
            # Find the peak in this phase
            phase_data = data[
                (data["Time"] >= phase["start_time"]) & (data["Time"] <= phase["end_time"])
            ]
            if len(phase_data) > 0:
                peak_idx = phase_data["Force"].idxmax()
                peak_time = data.loc[peak_idx, "Time"]
                time_to_peak = peak_time - phase["start_time"]
                time_to_peaks.append(time_to_peak)

        if time_to_peaks:
            metrics["time_to_first_peak"] = time_to_peaks[0] if time_to_peaks else None
            metrics["average_time_to_peak"] = np.mean(time_to_peaks)
            metrics["time_to_peak_variation"] = np.std(time_to_peaks)

            # Overall time to maximum force
            overall_peak_idx = data["Force"].idxmax()
            overall_peak_time = data.loc[overall_peak_idx, "Time"]
            metrics["time_to_max_force"] = overall_peak_time - data["Time"].iloc[0]

            # Consistency measure (lower variation = higher consistency)
            if metrics["average_time_to_peak"] > 0:
                metrics["peak_timing_consistency"] = 1 - (
                    metrics["time_to_peak_variation"] / metrics["average_time_to_peak"]
                )

    except Exception as e:
        print(f"Error calculating time to peak metrics: {str(e)}")

    return metrics


def save_force_plot_png(
    data, phases, output_path, config, stability_metrics=None, all_peaks_global=None
):
    """
    Saves a clean PNG plot of the raw force data with highlighted points of interest.
    Shows the original signal with minimal processing and clear annotations.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns (RAW DATA)
    phases : list
        List of detected sit-to-stand phases with comprehensive metrics
    output_path : str
        Path where to save the PNG file
    config : dict
        Configuration dictionary
    stability_metrics : dict, optional
        Stability metrics including standing baseline peaks
    all_peaks_global : list, optional
        All peaks detected in the entire signal
    """
    try:
        # Create figure with subplots - make room for legend on the right
        plt.figure(figsize=(22, 12))

        # Main plot (force vs time) - RAW SIGNAL
        ax1 = plt.subplot(2, 1, 1)

        # Plot the COMPLETE RAW force data as the main signal
        ax1.plot(
            data["Time"],
            data["Force"],
            "b-",
            linewidth=1.5,
            alpha=0.9,
            label=f"Raw Force Signal (Min: {data['Force'].min():.1f}N, Max: {data['Force'].max():.1f}N)",
        )

        # Highlight points of interest on the RAW signal
        colors = [
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "cyan",
            "magenta",
            "yellow",
        ]

        for i, phase in enumerate(phases[:8]):  # Show up to 8 phases
            color = colors[i % len(colors)]

            # Mark movement onset (start of phase)
            onset_time = phase["start_time"]
            onset_force = data.loc[phase["start_index"], "Force"]
            ax1.plot(
                onset_time,
                onset_force,
                "^",
                color=color,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=2,
                zorder=10,
                label=f"Movement Onset {i + 1}",
            )

            # Mark ALL peaks in this phase on the RAW signal
            if "all_peaks" in phase:
                for j, peak in enumerate(phase["all_peaks"]):
                    peak_time = peak["time"]
                    peak_force = peak["force"]
                    ax1.plot(
                        peak_time,
                        peak_force,
                        "o",
                        color=color,
                        markersize=10,
                        markeredgecolor="black",
                        markeredgewidth=2,
                        zorder=10,
                    )

                    # Annotate first peak with force value
                    if j == 0:
                        ax1.annotate(
                            f"Peak {j + 1}\n{peak_force:.1f}N",
                            xy=(peak_time, peak_force),
                            xytext=(15, 15),
                            textcoords="offset points",
                            fontsize=9,
                            fontweight="bold",
                            bbox={"boxstyle": "round,pad=0.3", "facecolor": color, "alpha": 0.8},
                        )

            # Mark phase end
            end_time = phase["end_time"]
            end_force = data.loc[phase["end_index"], "Force"]
            ax1.plot(
                end_time,
                end_force,
                "s",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=2,
                zorder=10,
                label=f"Phase End {i + 1}",
            )

        # Mark ALL peaks detected in the entire signal
        if all_peaks_global and len(all_peaks_global) > 0:
            # Extract times and forces for all global peaks
            all_peak_times = [p["time"] for p in all_peaks_global]
            all_peak_forces = [p["force"] for p in all_peaks_global]

            # Plot all global peaks with star markers
            ax1.plot(
                all_peak_times,
                all_peak_forces,
                "*",
                color="gold",
                markersize=15,
                markeredgecolor="black",
                markeredgewidth=1.5,
                zorder=12,
                label=f"All Peaks Detected ({len(all_peaks_global)})",
            )

            # Annotate first few global peaks
            for i, peak in enumerate(all_peaks_global[:5]):  # Show first 5
                ax1.annotate(
                    f"P{i + 1}\n{peak['force']:.1f}N",
                    xy=(peak["time"], peak["force"]),
                    xytext=(0, -25),
                    textcoords="offset points",
                    fontsize=7,
                    fontweight="bold",
                    ha="center",
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "facecolor": "yellow",
                        "edgecolor": "orange",
                        "alpha": 0.9,
                    },
                    arrowprops={
                        "arrowstyle": "->",
                        "connectionstyle": "arc3,rad=0",
                        "color": "orange",
                        "lw": 1,
                    },
                )

        # Mark standing baseline peaks if available (these are AFTER stabilization)
        if stability_metrics and "standing_peaks" in stability_metrics:
            standing_peaks = stability_metrics["standing_peaks"]
            if len(standing_peaks) > 0:
                # Extract times and forces
                peak_times = [p["time"] for p in standing_peaks]
                peak_forces = [p["force"] for p in standing_peaks]

                # Plot standing peaks with distinct markers
                ax1.plot(
                    peak_times,
                    peak_forces,
                    "D",
                    color="darkblue",
                    markersize=12,
                    markeredgecolor="white",
                    markeredgewidth=2,
                    zorder=15,
                    label=f"Standing Baseline Peaks ({len(standing_peaks)})",
                )

                # Annotate each standing peak
                for i, peak in enumerate(standing_peaks):
                    ax1.annotate(
                        f"S{i + 1}\n{peak['force']:.1f}N",
                        xy=(peak["time"], peak["force"]),
                        xytext=(0, 20),
                        textcoords="offset points",
                        fontsize=8,
                        fontweight="bold",
                        ha="center",
                        bbox={
                            "boxstyle": "round,pad=0.3",
                            "facecolor": "lightblue",
                            "edgecolor": "darkblue",
                            "alpha": 0.8,
                        },
                        arrowprops={
                            "arrowstyle": "->",
                            "connectionstyle": "arc3,rad=0",
                            "color": "darkblue",
                            "lw": 1.5,
                        },
                    )

        # Add reference lines for context
        baseline = np.percentile(data["Force"], 10)
        threshold = config["detection"]["force_threshold"]
        max_force = data["Force"].max()

        ax1.axhline(
            y=baseline,
            color="gray",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label=f"Baseline (10th percentile: {baseline:.1f}N)",
        )
        ax1.axhline(
            y=threshold,
            color="red",
            linestyle=":",
            linewidth=2,
            alpha=0.7,
            label=f"Detection Threshold: {threshold:.1f}N",
        )
        ax1.axhline(
            y=max_force,
            color="green",
            linestyle="-.",
            linewidth=2,
            alpha=0.7,
            label=f"Maximum Force: {max_force:.1f}N",
        )

        # Customize main plot
        ax1.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Vertical Force - Fz (N)", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Sit-to-Stand Force Analysis - Raw Signal with Points of Interest",
            fontsize=14,
            fontweight="bold",
        )

        # Place legend in the bottom right corner inside the plot area
        ax1.legend(
            bbox_to_anchor=(0.98, 0.02),
            loc="lower right",
            fontsize=8,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        ax1.grid(True, alpha=0.3, linestyle="--")

        # No text box - keeping only the legend

        # === SUBPLOT 2: Force Rate (RFD visualization) ===
        ax2 = plt.subplot(2, 1, 2)

        # Calculate and plot force rate of change from RAW data
        force_rate = np.gradient(data["Force"].values, data["Time"].values)
        ax2.plot(
            data["Time"],
            force_rate,
            "g-",
            linewidth=1.5,
            alpha=0.8,
            label=f"Force Rate (RFD) - Max: {force_rate.max():.1f} N/s",
        )

        # Mark RFD peaks and phase boundaries
        for i, phase in enumerate(phases[:8]):
            color = colors[i % len(colors)]
            phase_data_idx = (data["Time"] >= phase["start_time"]) & (
                data["Time"] <= phase["end_time"]
            )
            phase_time = data.loc[phase_data_idx, "Time"]
            phase_rate = force_rate[phase_data_idx]

            # Highlight RFD during phases
            ax2.plot(phase_time, phase_rate, color=color, linewidth=2, alpha=0.8)

            # Mark peak RFD
            if "peak_rfd_time" in phase:
                peak_rfd_idx = np.argmin(np.abs(data["Time"] - phase["peak_rfd_time"]))
                peak_rfd_value = force_rate[peak_rfd_idx]
                ax2.plot(
                    data["Time"].iloc[peak_rfd_idx],
                    peak_rfd_value,
                    "*",
                    color=color,
                    markersize=15,
                    markeredgecolor="black",
                    markeredgewidth=2,
                    zorder=10,
                    label=f"Peak RFD {i + 1}: {peak_rfd_value:.1f} N/s",
                )

        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
        ax2.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Rate of Force Development (N/s)", fontsize=12, fontweight="bold")
        ax2.set_title(
            "Force Development Rate (RFD) - Derived from Raw Signal",
            fontsize=12,
            fontweight="bold",
        )

        # Place legend in the bottom right corner inside the plot area for RFD plot too
        ax2.legend(
            bbox_to_anchor=(0.98, 0.02),
            loc="lower right",
            fontsize=8,
            frameon=True,
            fancybox=True,
            shadow=True,
        )
        ax2.grid(True, alpha=0.3, linestyle="--")

        # Adjust layout - no need for extra space since legends are inside
        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Raw force signal plot saved to: {output_path}")

    except Exception as e:
        print(f"Error saving force plot: {str(e)}")
        import traceback

        traceback.print_exc()


# display_results function removed - not needed in batch mode, results are saved automatically


def generate_interactive_html_report(data, analysis_result, config, output_path, result):
    """
    Generates an interactive HTML report with static charts using Plotly.js.
    Uses Plotly for interactive zooming/panning/hovering without slow animations.
    Includes Center of Pressure (CoP) analysis visualizations.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns
    analysis_result : dict
        Analysis results dictionary
    config : dict
        Configuration dictionary
    output_path : str
        Path to save the HTML file
    result : dict
        Result dictionary with file information
    """
    try:
        import base64
        import json
        from datetime import datetime
        
        # Try to load VAILA logo
        logo_b64 = ""
        try:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            logo_path = project_root / "docs" / "images" / "vaila_logo.png"
            
            if not logo_path.exists():
                logo_path = project_root / "docs" / "images" / "vaila.png"
            
            if logo_path.exists():
                with open(logo_path, "rb") as img_file:
                    logo_data = img_file.read()
                    logo_b64 = base64.b64encode(logo_data).decode("utf-8")
        except Exception as e:
            print(f"Warning: Could not load logo: {e}")

        # Embed force plot PNG in report if available
        force_plot_b64 = ""
        plot_path = result.get("plot_path")
        if plot_path and Path(plot_path).exists():
            try:
                with open(plot_path, "rb") as img_file:
                    force_plot_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as e:
                print(f"Warning: Could not load force plot image: {e}")
        
        # Extract data for JavaScript - convert to JSON-safe format
        time_data = [float(x) for x in data["Time"].tolist()]
        force_data = [float(x) for x in data["Force"].tolist()]
        
        # Extract phases information
        phases = analysis_result.get("sit_to_stand_phases", [])
        phases_data = []
        for phase in phases:
            phases_data.append({
                "start_time": float(phase.get("start_time", 0)),
                "end_time": float(phase.get("end_time", 0)),
                "peak_time": float(phase.get("peak_time", 0)),
                "peak_force": float(phase.get("peak_force", 0)),
                "duration": float(phase.get("duration", 0)),
            })
        
        # Extract peaks
        all_peaks = analysis_result.get("all_peaks_global", [])
        peaks_data = [{"time": float(p.get("time", 0)), "force": float(p.get("force", 0))} for p in all_peaks]
        
        # Extract CoP data if available
        cop_data = analysis_result.get("cop_results", {})
        has_cop = False
        cop_x = []
        cop_y = []
        if cop_data and "cop_x" in cop_data and "cop_y" in cop_data:
            has_cop = True
            # Convert numpy arrays to lists of floats, handling NaNs
            cx = cop_data["cop_x"]
            cy = cop_data["cop_y"]
            # Replace NaNs with None for JSON
            cop_x = [float(x) if not np.isnan(x) else None for x in cx]
            cop_y = [float(y) if not np.isnan(y) else None for y in cy]
        cop_ellipse_x = list(cop_data.get("ellipse_x_path") or []) if cop_data else []
        cop_ellipse_y = list(cop_data.get("ellipse_y_path") or []) if cop_data else []

        # Extract stability metrics
        stability = analysis_result.get("stability_metrics", {})
        stability_data = {
            "reference_peak_force": float(stability.get("first_peak_force", 0)),
            "stability_index": float(stability.get("stability_index", 0)),
            "mean_deviation": float(stability.get("mean_deviation", 0)),
        }
        
        # Extract movement metrics
        movement = analysis_result.get("movement_metrics", {})
        energy = analysis_result.get("energy_metrics", {})
        time_to_peak = analysis_result.get("time_to_peak_metrics", {})
        impulse = analysis_result.get("impulse_metrics", {})
        
        # Prepare metrics grid data (align with old animated report: Duration, Mean/Max/Min Force, movement, time-to-peak, impulse, stability, energy)
        metrics_html = f'''
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Duration</div>
                    <div class="metric-value">{analysis_result.get('duration', 0):.2f}<span class="metric-unit">s</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Force</div>
                    <div class="metric-value">{analysis_result.get('mean_force', 0):.2f}<span class="metric-unit">N</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Force</div>
                    <div class="metric-value">{analysis_result.get('max_force', 0):.1f}<span class="metric-unit">N</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Min Force</div>
                    <div class="metric-value">{analysis_result.get('min_force', 0):.1f}<span class="metric-unit">N</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Phases Detected</div>
                    <div class="metric-value">{movement.get('num_phases', 0)}<span class="metric-unit">count</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Movement Time</div>
                    <div class="metric-value">{movement.get('total_movement_time', 0):.2f}<span class="metric-unit">s</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Phase Duration</div>
                    <div class="metric-value">{movement.get('average_phase_duration', 0):.2f}<span class="metric-unit">s</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Stability Index</div>
                    <div class="metric-value">{stability_data.get('stability_index', 0):.3f}<span class="metric-unit">(0-1)</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Mean Deviation from Peak</div>
                    <div class="metric-value">{stability_data.get('mean_deviation', 0):.2f}<span class="metric-unit">N</span></div>
                </div>
        '''
        if time_to_peak.get("time_to_first_peak") is not None:
            metrics_html += f'''
                <div class="metric-card">
                    <div class="metric-label">Time to First Peak</div>
                    <div class="metric-value">{time_to_peak.get('time_to_first_peak', 0):.3f}<span class="metric-unit">s</span></div>
                </div>
            '''
        if time_to_peak.get("time_to_max_force") is not None:
            metrics_html += f'''
                <div class="metric-card">
                    <div class="metric-label">Time to Max Force</div>
                    <div class="metric-value">{time_to_peak.get('time_to_max_force', 0):.3f}<span class="metric-unit">s</span></div>
                </div>
            '''
        if impulse:
            metrics_html += f'''
                <div class="metric-card">
                    <div class="metric-label">Total Impulse</div>
                    <div class="metric-value">{impulse.get('total_impulse', 0):.2f}<span class="metric-unit">N⋅s</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Peak Power</div>
                    <div class="metric-value">{impulse.get('peak_power', 0):.2f}<span class="metric-unit">W</span></div>
                </div>
            '''
        if energy:
            metrics_html += f'''
                <div class="metric-card">
                    <div class="metric-label">Body Weight</div>
                    <div class="metric-value">{energy.get('body_weight_kg', 0):.1f}<span class="metric-unit">kg</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Metabolic Energy</div>
                    <div class="metric-value">{energy.get('total_metabolic_energy_kcal', 0):.3f}<span class="metric-unit">kcal</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Energy Efficiency</div>
                    <div class="metric-value">{energy.get('energy_efficiency', 0):.1f}<span class="metric-unit">%</span></div>
                </div>
            '''
        if has_cop:
            pca1_pct = cop_data.get("pca_pc1_variance_ratio", 0) * 100
            pca2_pct = cop_data.get("pca_pc2_variance_ratio", 0) * 100
            metrics_html += f'''
                <div class="metric-card">
                    <div class="metric-label">CoP Path Length</div>
                    <div class="metric-value">{cop_data.get('cop_path_length', 0):.1f}<span class="metric-unit">mm</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ellipse Area (95%)</div>
                    <div class="metric-value">{cop_data.get('ellipse_area_95', 0):.1f}<span class="metric-unit">mm²</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ellipse Angle</div>
                    <div class="metric-value">{cop_data.get('ellipse_angle_deg', 0):.1f}<span class="metric-unit">°</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RMS Sway Total (CoP)</div>
                    <div class="metric-value">{cop_data.get('rms_sway_total_mm', 0):.2f}<span class="metric-unit">mm</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RMS Sway ML (X)</div>
                    <div class="metric-value">{cop_data.get('rms_sway_ml_mm', 0):.2f}<span class="metric-unit">mm</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RMS Sway AP (Y)</div>
                    <div class="metric-value">{cop_data.get('rms_sway_ap_mm', 0):.2f}<span class="metric-unit">mm</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">PCA PC1 Variance</div>
                    <div class="metric-value">{pca1_pct:.1f}<span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">PCA PC2 Variance</div>
                    <div class="metric-value">{pca2_pct:.1f}<span class="metric-unit">%</span></div>
                </div>
            '''
            
        metrics_html += '</div>'
        
        # Create HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sit-to-Stand Interactive Report - {result.get('filename', 'Analysis')}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            overflow: hidden;
        }}
        .header {{
            background: white;
            padding: 20px 30px;
            border-bottom: 1px solid #eee;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .header-content h1 {{ font-size: 1.8em; color: #1a202c; margin-bottom: 5px; }}
        .header-content p {{ color: #718096; }}
        .content {{ padding: 30px; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            text-align: center;
        }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; color: #2d3748; }}
        .metric-unit {{ font-size: 0.5em; color: #718096; margin-left: 5px; font-weight: normal; vertical-align: middle; }}
        .metric-label {{ font-size: 0.9em; color: #718096; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.05em; }}
        
        .chart-section {{
            margin-bottom: 30px;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            background: white;
        }}
        .chart-title {{ font-size: 1.2em; font-weight: 600; margin-bottom: 15px; color: #2d3748; display: flex; align-items: center; gap: 10px; }}
        .chart-container {{ width: 100%; height: 500px; }}
        
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        
        @media (max-width: 1000px) {{
            .two-col {{ grid-template-columns: 1fr; }}
        }}
        
        .footer {{
            text-align: center;
            padding: 20px;
            color: #718096;
            font-size: 0.9em;
            border-top: 1px solid #eee;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
            background: #edf2f7;
            color: #4a5568;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>Sit-to-Stand Analysis Report</h1>
                <p>{result.get('filename', 'Analysis File')} • {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            </div>
            {f'<img src="data:image/png;base64,{logo_b64}" style="height: 50px;" alt="vailá">' if logo_b64 else ''}
        </div>
        
        <div class="content">
            {metrics_html}
            
            {f'<div class="chart-section"><div class="chart-title">📊 Force plot (PNG)</div><img src="data:image/png;base64,{force_plot_b64}" alt="Force plot" style="max-width:100%; height:auto; border-radius:8px;" /></div>' if force_plot_b64 else ''}
            
            <div class="chart-section">
                <div class="chart-title">📊 Vertical Force & RFD Analysis</div>
                <div id="forceChart" class="chart-container" style="height: 600px;"></div>
            </div>
            
            {f'''
            <div class="two-col">
                <div class="chart-section">
                    <div class="chart-title">👣 CoP Path (Top View)</div>
                    <div id="copPathChart" class="chart-container" style="height: 450px;"></div>
                </div>
                <div class="chart-section">
                    <div class="chart-title">📈 CoP Displacement vs Time</div>
                    <div id="copTimeChart" class="chart-container" style="height: 450px;"></div>
                </div>
            </div>
            ''' if has_cop else ''}
            
        </div>
        
        <div class="footer">
            Generated by vailá Multimodal Toolbox
        </div>
    </div>
    
    <script>
        // Data
        const timeData = {json.dumps(time_data)};
        const forceData = {json.dumps(force_data)};
        const phases = {json.dumps(phases_data)};
        const peaks = {json.dumps(peaks_data)};
        
        // --- FORCE CHART ---
        
        // Calculate RFD (Force Rate) for subplot
        const rfdData = [];
        for(let i=1; i<forceData.length; i++) {{
            const dt = timeData[i] - timeData[i-1];
            if(dt > 0) {{
                rfdData.push((forceData[i] - forceData[i-1])/dt);
            }} else {{
                rfdData.push(0);
            }}
        }}
        rfdData.push(0); // Pad last
        
        const forceTrace = {{
            x: timeData,
            y: forceData,
            name: 'Vertical Force (Fz)',
            type: 'scatter',
            line: {{ color: '#4299e1', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(66, 153, 225, 0.1)'
        }};
        
        const rfdTrace = {{
            x: timeData,
            y: rfdData,
            name: 'RFD (N/s)',
            type: 'scatter',
            yaxis: 'y2',
            line: {{ color: '#ed8936', width: 1.5, dash: 'dot' }},
            opacity: 0.7
        }};
        
        // Phases Backgrounds
        const shapes = phases.map((p, i) => ({{
            type: 'rect',
            xref: 'x', yref: 'paper',
            x0: p.start_time, x1: p.end_time,
            y0: 0, y1: 1,
            fillcolor: i%2===0 ? 'rgba(72, 187, 120, 0.15)' : 'rgba(72, 187, 120, 0.05)',
            line: {{ width: 0 }},
            layer: 'below'
        }}));
        
        const layoutForce = {{
            title: {{ text: 'Vertical Ground Reaction Force & RFD', font: {{size: 14}} }},
            font: {{ family: 'Segoe UI' }},
            xaxis: {{ title: 'Time (s)', gridcolor: '#f7fafc' }},
            yaxis: {{ title: 'Force (N)', gridcolor: '#edf2f7' }},
            yaxis2: {{
                title: 'RFD (N/s)',
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                zeroline: false
            }},
            shapes: shapes,
            hovermode: 'closest',
            legend: {{ orientation: 'h', y: 1.1 }},
            margin: {{ l: 50, r: 50, t: 50, b: 50 }}
        }};
        
        Plotly.newPlot('forceChart', [forceTrace, rfdTrace], layoutForce, {{responsive: true}});
        
        // --- CoP CHARTS ---
        {f'''
        const copX = {json.dumps(cop_x)};
        const copY = {json.dumps(cop_y)};
        const copEllipseX = {json.dumps(cop_ellipse_x)};
        const copEllipseY = {json.dumps(cop_ellipse_y)};
        
        // CoP Path
        const copPathTrace = {{
            x: copX,
            y: copY,
            mode: 'markers+lines',
            type: 'scatter',
            name: 'CoP Path',
            line: {{ color: '#805ad5', width: 1 }},
            marker: {{ size: 3, color: timeData, colorscale: 'Viridis', showscale: false }}
        }};
        const copPathTraces = [copPathTrace];
        if (copEllipseX.length > 0 && copEllipseY.length > 0) {{
            copPathTraces.push({{
                x: copEllipseX,
                y: copEllipseY,
                mode: 'lines',
                type: 'scatter',
                name: '95% Confidence Ellipse',
                line: {{ color: 'gray', width: 2, dash: 'dash' }}
            }});
        }}
        
        const layoutCopPath = {{
            title: {{ text: 'Center of Pressure Path (mm)', font: {{size: 14}} }},
            xaxis: {{ title: 'CoP X (mm)', zeroline: true, zerolinecolor: '#cbd5e0' }},
            yaxis: {{ title: 'CoP Y (mm)', zeroline: true, zerolinecolor: '#cbd5e0', scaleanchor: 'x', scaleratio: 1 }},
            hovermode: 'closest',
            margin: {{ l: 40, r: 40, t: 40, b: 40 }}
        }};
        
        Plotly.newPlot('copPathChart', copPathTraces, layoutCopPath, {{responsive: true}});
        
        // CoP Time Series
        const copXTrace = {{
            x: timeData,
            y: copX,
            name: 'CoP X (ML)',
            line: {{ color: '#e53e3e' }}
        }};
        const copYTrace = {{
            x: timeData,
            y: copY,
            name: 'CoP Y (AP)',
            line: {{ color: '#38a169' }}
        }};
        
        const layoutCopTime = {{
            title: {{ text: 'CoP Components over Time', font: {{size: 14}} }},
            xaxis: {{ title: 'Time (s)' }},
            yaxis: {{ title: 'Displacement (mm)' }},
            legend: {{ orientation: 'h', y: 1.1 }},
            margin: {{ l: 40, r: 40, t: 40, b: 40 }}
        }};
        
        Plotly.newPlot('copTimeChart', [copXTrace, copYTrace], layoutCopTime, {{responsive: true}});
        ''' if has_cop else ''}
        
    </script>
</body>
</html>"""
        
        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"  Interactive HTML report saved: {Path(output_path).name}")
        
    except Exception as e:
        print(f"Error generating interactive HTML report: {str(e)}")
        import traceback
        traceback.print_exc()


class SitToStandGUI:
    """
    Simple GUI for sit-to-stand analysis configuration using basic tkinter.
    Cross-platform compatible and minimalistic.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sit-to-Stand Analysis")
        self.root.geometry("550x350")
        self.root.resizable(True, True)

        # Variables for storing user selections
        self.config_file = ""
        self.input_dir = ""
        self.output_dir = ""

        # Use defaults by default
        self.use_defaults = True

        self.create_widgets()

    def create_widgets(self):
        """Creates the simple GUI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(
            main_frame, text="Sit-to-Stand Analysis Setup", font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Config file row
        config_frame = tk.Frame(main_frame)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(config_frame, text="Config file:").pack(side=tk.LEFT)
        self.config_entry = tk.Entry(config_frame, width=40)
        self.config_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        config_btn = tk.Button(
            config_frame, text="Browse", command=self.browse_config_file, width=8
        )
        config_btn.pack(side=tk.LEFT)

        # Input directory row
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(input_frame, text="Input dir:").pack(side=tk.LEFT)
        self.input_entry = tk.Entry(input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        input_btn = tk.Button(input_frame, text="Browse", command=self.browse_input_dir, width=8)
        input_btn.pack(side=tk.LEFT)
        
        # Info label about automatic file detection
        info_label = tk.Label(
            main_frame, 
            text="Note: Automatically processes all .c3d and .csv files in the directory",
            font=("Arial", 8),
            fg="gray"
        )
        info_label.pack(pady=(0, 10))

        # Output directory row
        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(output_frame, text="Output dir:").pack(side=tk.LEFT)
        self.output_entry = tk.Entry(output_frame, width=40)
        self.output_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        output_btn = tk.Button(output_frame, text="Browse", command=self.browse_output_dir, width=8)
        output_btn.pack(side=tk.LEFT)

        # Options frame
        options_frame = tk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(10, 20))

        self.defaults_var = tk.BooleanVar(value=True)
        defaults_cb = tk.Checkbutton(
            options_frame,
            text="Use default config",
            variable=self.defaults_var,
            command=self.toggle_defaults,
        )
        defaults_cb.pack(side=tk.LEFT)

        # Create default config file button
        create_config_btn = tk.Button(
            options_frame,
            text="Create Default Config",
            command=self.create_default_config_file,
            width=20,
        )
        create_config_btn.pack(side=tk.LEFT, padx=(20, 0))

        # Buttons frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        run_btn = tk.Button(
            btn_frame,
            text="Run Analysis",
            command=self.run_analysis,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
        )
        run_btn.pack(side=tk.RIGHT, padx=(10, 0))

        cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.root.quit, padx=20)
        cancel_btn.pack(side=tk.RIGHT)

        # Status label
        self.status_label = tk.Label(main_frame, text="Ready", font=("Arial", 9), fg="blue")
        self.status_label.pack(pady=(10, 0))

    def toggle_defaults(self):
        """Toggles between default and custom config."""
        self.use_defaults = self.defaults_var.get()
        if self.use_defaults:
            self.config_entry.delete(0, tk.END)
            self.config_entry.config(state=tk.DISABLED)
        else:
            self.config_entry.config(state=tk.NORMAL)

    def browse_config_file(self):
        """Opens file dialog for TOML configuration file selection."""
        # If using defaults, automatically switch to custom config mode
        if self.use_defaults:
            self.use_defaults = False
            self.defaults_var.set(False)
            self.config_entry.config(state=tk.NORMAL)

        filename = filedialog.askopenfilename(
            title="Select TOML Configuration File",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if filename:
            self.config_file = filename
            self.config_entry.delete(0, tk.END)
            self.config_entry.insert(0, filename)

    def browse_input_dir(self):
        """Opens directory dialog for input files selection."""
        dirname = filedialog.askdirectory(title="Select Input Directory")
        if dirname:
            self.input_dir = dirname
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, dirname)
            # Auto-set output directory if not set
            if not self.output_entry.get():
                default_output = os.path.join(dirname, "sit2stand_analysis")
                self.output_entry.insert(0, default_output)

    def browse_output_dir(self):
        """Opens directory dialog for output directory selection."""
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir = dirname
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, dirname)

    def create_default_config_file(self):
        """Creates a default TOML configuration file."""
        default_config = get_default_config()

        # Ask user where to save the config file
        filename = filedialog.asksaveasfilename(
            title="Save Default Configuration File",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )

        if filename:
            try:
                import toml

                with open(filename, "w", encoding="utf-8") as f:
                    toml.dump(default_config, f)

                self.status_label.config(text=f"Default config saved to: {filename}", fg="green")
                print(f"Default configuration file created: {filename}")

                # Optionally load the created file
                if messagebox.askyesno("Load Config", "Load the created configuration file?"):
                    self.config_file = filename
                    self.config_entry.delete(0, tk.END)
                    self.config_entry.insert(0, filename)
                    self.use_defaults = False
                    self.defaults_var.set(False)
                    self.toggle_defaults()

            except Exception as e:
                self.status_label.config(text=f"Error creating config: {str(e)}", fg="red")
                messagebox.showerror("Error", f"Failed to create config file: {str(e)}")

    def run_analysis(self):
        """Validates inputs and runs the analysis."""
        # Get current values from entries
        config_file = self.config_entry.get().strip() if not self.use_defaults else ""
        input_dir = self.input_entry.get().strip()
        output_dir = self.output_entry.get().strip()

        # Validation
        if not input_dir:
            self.status_label.config(text="Error: Please select input directory", fg="red")
            return

        if not os.path.exists(input_dir):
            self.status_label.config(text="Error: Input directory does not exist", fg="red")
            return

        if not self.use_defaults and not config_file:
            self.status_label.config(
                text="Error: Please select config file or use defaults", fg="red"
            )
            return

        # Prepare arguments for CLI mode
        cli_args = []

        # Add config file or empty for defaults
        if self.use_defaults:
            cli_args.append("")  # Will be handled as defaults in CLI mode
        else:
            cli_args.append(config_file)

        # Add input directory
        cli_args.append(input_dir)

        # Add output directory if specified
        if output_dir:
            cli_args.append(output_dir)

        # Update status
        self.status_label.config(text="Running analysis...", fg="orange")
        self.root.update()

        try:
            # Run in CLI mode with the prepared arguments
            result = run_cli_mode(cli_args)

            if result is None:  # Success
                self.status_label.config(text="Analysis completed successfully!", fg="green")
                messagebox.showinfo("Success", "Sit-to-Stand batch analysis completed successfully!")
            else:
                self.status_label.config(text="Analysis failed", fg="red")
                messagebox.showerror("Error", "Analysis failed. Check console for details.")

        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", fg="red")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def run(self):
        """Runs the GUI main loop."""
        self.root.mainloop()


if __name__ == "__main__":
    main()
