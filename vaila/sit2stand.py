"""
================================================================================
sit2stand.py - Sit to Stand Analysis Module
================================================================================
Author: Prof. Paulo Santiago
Create: 10 October 2025
Update: 14 October 2025
Version: 0.0.3

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

Analysis Capabilities:
----------------------
- Sit-to-stand phase detection with configurable thresholds
- Force impulse calculation with filtered data
- Peak force identification and timing analysis
- Movement timing analysis with onset detection
- Balance assessment during transitions
- Butterworth low-pass filtering for noise reduction

Configuration:
--------------
Parameters can be configured via:
1. TOML configuration files (recommended for reproducibility)
2. Interactive GUI dialogs (for quick testing)

TOML Configuration File Format:
-------------------------------
[analysis]
# Column containing vertical force data
force_column = "Fz"

[filtering]
# Butterworth filter parameters
enabled = true
cutoff_frequency = 10.0  # Hz
sampling_frequency = 100.0  # Hz
order = 4

[detection]
# Sit-to-stand detection parameters
force_threshold = 10.0  # N
min_duration = 0.5  # seconds
onset_threshold = 5.0  # N above baseline

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
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import pandas as pd
import numpy as np
try:
    import toml
    TOML_SUPPORT = True
except ImportError:
    try:
        import tomli as toml  # Fallback
        TOML_SUPPORT = True
    except ImportError:
        try:
            import tomllib as toml  # Python 3.11+ built-in
            TOML_SUPPORT = True
        except ImportError:
            TOML_SUPPORT = False
            print("Warning: No TOML library available. Config files won't work.")
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path

# Try to import ezc3d for C3D file support
try:
    from ezc3d import c3d
    C3D_SUPPORT = True
except ImportError:
    C3D_SUPPORT = False
    print("Warning: ezc3d not found. C3D file support will be limited.")


def main(cli_args=None):
    """
    Main function to run the sit-to-stand analysis in batch mode or GUI mode.
    Processes multiple C3D and CSV files automatically using TOML configuration.

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
    Runs the analysis in CLI mode with command line arguments.

    Parameters:
    -----------
    cli_args : list
        Command line arguments [config_file, input_directory, output_directory, file_format]
    """
    if len(cli_args) < 3:
        print("Usage: python sit2stand.py <config.toml> <input_directory> [output_directory] [file_format]")
        print("Example: python sit2stand.py config.toml /path/to/files /path/to/output auto")
        print("File format: auto, c3d, csv")
        return

    config_file = cli_args[0]
    input_dir = cli_args[1]
    output_dir = cli_args[2] if len(cli_args) > 2 else None
    file_format = cli_args[3] if len(cli_args) > 3 else "auto"

    try:
        # Handle empty config file (use defaults)
        if not config_file or config_file == "":
            config = get_default_config()
            print("Using default configuration")
        else:
            # Step 1: Load TOML configuration
            config = load_toml_config(config_file)
            if not config:
                print(f"Failed to load configuration from {config_file}")
                return

        # Step 2: Set up output directory
        if not output_dir:
            output_dir = os.path.join(input_dir, "sit2stand_analysis")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Step 3: Find files based on format
        files = find_analysis_files(input_dir, file_format)
        if not files:
            print(f"No files found in {input_dir} with format {file_format}")
            return

        print(f"Found {len(files)} files to process:")
        for file_path in files:
            print(f"  - {os.path.basename(file_path)}")

        # Step 4: Run batch analysis
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
        'analysis': {
            'force_column': 'Fz'
        },
        'filtering': {
            'enabled': True,
            'cutoff_frequency': 10.0,
            'sampling_frequency': 100.0,
            'order': 4
        },
        'detection': {
            'force_threshold': 10.0,
            'min_duration': 0.5,
            'onset_threshold': 5.0
        }
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

        # Load TOML file - handle different library APIs
        try:
            # Try toml library (most common)
            with open(config_file, 'r') as f:
                config = toml.load(f)
        except AttributeError:
            # Try tomllib (Python 3.11+ built-in, binary mode)
            with open(config_file, 'rb') as f:
                config = toml.load(f)
                
        print(f"Loaded configuration from: {config_file}")

        # Validate required sections
        required_sections = ['analysis', 'filtering', 'detection']
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
    Finds files in the input directory based on specified format.

    Parameters:
    -----------
    input_dir : str
        Input directory path
    file_format : str
        File format to search for: 'auto', 'c3d', 'csv'

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

    # Determine extensions based on format
    if file_format == "auto":
        extensions = ['.c3d', '.csv']
    elif file_format == "c3d":
        extensions = ['.c3d']
    elif file_format == "csv":
        extensions = ['.csv']
    else:
        print(f"Unknown file format: {file_format}")
        return files

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
    if (config.get('analysis', {}).get('force_column')):
        column_from_config = config['analysis']['force_column']
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
        df_header = pd.read_csv(sample_file, nrows=0)
        columns = list(df_header.columns)

        print(f"Auto-detecting force column from: {columns}")

        # Priority order for column detection
        force_patterns = [
            # Vertical force columns (most common for sit-to-stand)
            'Force.Fz1', 'Force.Fz2', 'Force.Fz3', 'Force.Fz4',  # Force plates
            'Fz', 'FZ', 'Force_Z', 'Vertical_Force',  # Generic names
            'Force.Fz', 'force_z',  # Alternative patterns
            # If no vertical force found, try any force column
            'Force.Fx1', 'Force.Fy1', 'Force.Fx2', 'Force.Fy2',
            'Fx', 'Fy', 'Force_X', 'Force_Y'
        ]

        for pattern in force_patterns:
            for col in columns:
                if pattern.lower() in col.lower():
                    print(f"Auto-detected force column: {col}")
                    return col

        # If no standard pattern found, suggest the first numeric column that's not Time
        for col in columns:
            if col.lower() not in ['time', 'timestamp', 'frame', 't']:
                try:
                    # Try to read a small sample to see if it's numeric
                    sample_data = pd.read_csv(sample_file, usecols=[col], nrows=10)
                    if pd.api.types.is_numeric_dtype(sample_data[col]):
                        print(f"Auto-detected numeric column: {col}")
                        return col
                except:
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
        if sample_file.lower().endswith('.c3d'):
            # For C3D files, we can't easily verify without proper library
            # Assume it exists if it's in our standard list
            standard_columns = ['Fz', 'Force_Z', 'Vertical_Force', 'FZ']
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
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # Apply the filter
        filtered_data = filtfilt(b, a, data)

        return filtered_data

    except Exception as e:
        print(f"Error applying Butterworth filter: {str(e)}")
        return data  # Return original data if filtering fails


# configure_filtering_parameters and configure_detection_parameters functions removed - not needed in batch mode


def run_batch_analysis(files, config, output_dir):
    """
    Runs the sit-to-stand analysis on multiple files using configuration.

    Parameters:
    -----------
    files : list
        List of file paths to analyze
    config : dict
        Complete configuration dictionary
    output_dir : str
        Output directory for results

    Returns:
    --------
    list
        List of analysis results for each file
    """
    results = []
    column_name = config['analysis']['force_column']

    # Create output subdirectories
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for i, file_path in enumerate(files):
        try:
            print(f"Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}")

            # Read file data
            if file_path.lower().endswith('.c3d'):
                data = read_c3d_file(file_path, column_name)
            else:
                data = read_csv_file(file_path, column_name)

            if data is None:
                print(f"  Skipping {file_path} - could not read data")
                results.append({
                    'file': file_path,
                    'filename': os.path.basename(file_path),
                    'error': 'Could not read data'
                })
                continue

            # Apply Butterworth filtering if enabled
            if config.get('filtering', {}).get('enabled', False):
                print("  Applying Butterworth filter...")
                filtered_force = butterworth_filter(
                    data['Force'].values,
                    fs=config['filtering']['sampling_frequency'],
                    cutoff=config['filtering']['cutoff_frequency'],
                    order=config['filtering']['order']
                )
                data = data.copy()
                data['Force'] = filtered_force
                print(f"  Filtered with cutoff {config['filtering']['cutoff_frequency']} Hz")

            # Analyze sit-to-stand movement
            analysis_result = analyze_sit_to_stand(data, config)

            # Save force plot as PNG
            plot_filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_force_plot.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            save_force_plot_png(data, analysis_result['sit_to_stand_phases'], plot_path, config)

            # Store results with file information
            result = {
                'file': file_path,
                'filename': os.path.basename(file_path),
                'analysis': analysis_result,
                'configuration': config.copy(),
                'plot_path': plot_path
            }

            results.append(result)
            print(f"  ✓ Completed: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            results.append({
                'file': file_path,
                'filename': os.path.basename(file_path),
                'error': str(e)
            })

    return results


def generate_batch_report(results, config, output_dir):
    """
    Generates a comprehensive batch analysis report.

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
        # Create reports directory
        reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)

        # Generate summary statistics
        successful_analyses = sum(1 for r in results if 'error' not in r)
        total_files = len(results)

        # Generate text report
        report_path = os.path.join(reports_dir, "batch_analysis_report.txt")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Sit-to-Stand Batch Analysis Report\n")
            f.write("=" * 60 + "\n\n")

            # Configuration summary
            f.write("Configuration Used:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Force Column: {config['analysis']['force_column']}\n")
            f.write(f"Butterworth Filter: {'Enabled' if config['filtering']['enabled'] else 'Disabled'}\n")

            if config['filtering']['enabled']:
                f.write(f"  Cutoff Frequency: {config['filtering']['cutoff_frequency']} Hz\n")
                f.write(f"  Sampling Frequency: {config['filtering']['sampling_frequency']} Hz\n")
                f.write(f"  Filter Order: {config['filtering']['order']}\n")

            f.write(f"Detection Parameters:\n")
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
            f.write(f"Success rate: {(successful_analyses/total_files*100):.1f}%\n\n")

            # Detailed results with comprehensive clinical metrics
            total_phases = 0
            for result in results:
                if 'error' not in result:
                    f.write(f"\nFile: {result['filename']}\n")
                    f.write("=" * 60 + "\n")
                    analysis = result['analysis']

                    # === BASIC METRICS ===
                    f.write("BASIC FORCE METRICS:\n")
                    f.write(f"  Duration: {analysis['duration']:.2f} s\n")
                    f.write(f"  Mean Force: {analysis['mean_force']:.2f} N\n")
                    f.write(f"  Max Force: {analysis['max_force']:.2f} N\n")
                    f.write(f"  Min Force: {analysis['min_force']:.2f} N\n\n")

                    # === MOVEMENT DETECTION ===
                    movement = analysis['movement_metrics']
                    f.write("MOVEMENT DETECTION:\n")
                    f.write(f"  Phases Detected: {movement['num_phases']}\n")
                    f.write(f"  Total Movement Time: {movement['total_movement_time']:.2f} s\n")
                    f.write(f"  Average Phase Duration: {movement['average_phase_duration']:.2f} s\n")

                    if movement['phases_per_minute'] > 0:
                        f.write(f"  Phases per Minute: {movement['phases_per_minute']:.1f}\n\n")

                    # === TIME TO PEAK METRICS (CRITICAL FOR CP) ===
                    time_to_peak = analysis['time_to_peak_metrics']
                    f.write("TIME TO PEAK METRICS:\n")
                    if time_to_peak.get('time_to_first_peak') is not None:
                        f.write(f"  Time to First Peak: {time_to_peak['time_to_first_peak']:.3f} s\n")
                    if time_to_peak.get('time_to_max_force') is not None:
                        f.write(f"  Time to Max Force: {time_to_peak['time_to_max_force']:.3f} s\n")
                    if time_to_peak.get('average_time_to_peak') > 0:
                        f.write(f"  Average Time to Peak: {time_to_peak['average_time_to_peak']:.3f} s\n")
                    if time_to_peak.get('time_to_peak_variation') > 0:
                        f.write(f"  Time to Peak Variation: {time_to_peak['time_to_peak_variation']:.3f} s\n\n")

                    # === RATE OF FORCE DEVELOPMENT (CRITICAL FOR CP) ===
                    f.write("RATE OF FORCE DEVELOPMENT (RFD):\n")
                    phases_data = analysis.get('sit_to_stand_phases', [])
                    if phases_data:
                        first_phase = phases_data[0]
                        f.write(f"  Overall RFD: {first_phase.get('overall_rfd', 0):.2f} N/s\n")
                        f.write(f"  Early RFD (first 100ms): {first_phase.get('early_rfd', 0):.2f} N/s\n")
                        f.write(f"  Peak RFD: {first_phase.get('peak_rfd', 0):.2f} N/s\n")
                        if first_phase.get('weight_transfer_time'):
                            f.write(f"  Weight Transfer Time: {first_phase['weight_transfer_time']:.3f} s\n\n")

                    # === IMPULSE METRICS ===
                    impulse = analysis['impulse_metrics']
                    f.write("IMPULSE & POWER METRICS:\n")
                    f.write(f"  Total Impulse: {impulse['total_impulse']:.2f} N⋅s\n")
                    f.write(f"  Average Impulse: {impulse['average_impulse']:.2f} N⋅s\n")
                    f.write(f"  Peak Power: {impulse['peak_power']:.2f} W\n")
                    f.write(f"  Average Power: {impulse['average_power']:.2f} W\n")
                    f.write(f"  Force Rate of Change: {impulse['force_rate_of_change']:.2f} N/s\n\n")

                    # === CLINICAL QUALITY METRICS ===
                    if phases_data:
                        first_phase = phases_data[0]
                        f.write("MOVEMENT QUALITY METRICS:\n")
                        if 'force_cv' in first_phase:
                            f.write(f"  Force Coefficient of Variation: {first_phase['force_cv']:.2f}%\n")
                        if 'force_jerk' in first_phase:
                            f.write(f"  Force Smoothness (Jerk): {first_phase['force_jerk']:.2f}\n")
                        if 'bilateral_index' in first_phase:
                            f.write(f"  Bilateral Symmetry Index: {first_phase['peak_symmetry']:.3f}\n")
                        if 'consistency_score' in first_phase:
                            f.write(f"  Movement Consistency: {first_phase['consistency_score']:.3f}\n")
                        if 'num_peaks' in first_phase:
                            f.write(f"  Number of Peaks: {first_phase['num_peaks']}\n\n")

                    # Plot information
                    if 'plot_path' in result:
                        f.write(f"Plot saved: {os.path.basename(result['plot_path'])}\n")

                    f.write("=" * 60 + "\n")
                    total_phases += movement['num_phases']
                else:
                    f.write(f"\nERROR - {result['filename']}: {result['error']}\n")
                    f.write("=" * 60 + "\n")

            f.write(f"\nOverall Summary:\n")
            f.write(f"Total phases detected across all files: {total_phases}\n")
            f.write(f"Average phases per successful file: {total_phases/successful_analyses:.1f}\n")

        # Generate comprehensive CSV summary with all clinical metrics
        csv_data = []
        for result in results:
            if 'error' not in result:
                analysis = result['analysis']
                movement = analysis['movement_metrics']
                impulse = analysis['impulse_metrics']
                time_to_peak = analysis['time_to_peak_metrics']
                phases_data = analysis.get('sit_to_stand_phases', [])

                # Base metrics
                row = {
                    'filename': result['filename'],
                    'duration_s': analysis['duration'],
                    'mean_force_N': analysis['mean_force'],
                    'max_force_N': analysis['max_force'],
                    'min_force_N': analysis['min_force'],
                    
                    # Movement detection
                    'num_phases': movement['num_phases'],
                    'total_movement_time_s': movement['total_movement_time'],
                    'average_phase_duration_s': movement['average_phase_duration'],
                    'phases_per_minute': movement['phases_per_minute'],
                    
                    # Time to peak metrics (CRITICAL)
                    'time_to_first_peak_s': time_to_peak.get('time_to_first_peak'),
                    'time_to_max_force_s': time_to_peak.get('time_to_max_force'),
                    'average_time_to_peak_s': time_to_peak.get('average_time_to_peak', 0),
                    'time_to_peak_variation_s': time_to_peak.get('time_to_peak_variation', 0),
                    
                    # Impulse metrics
                    'total_impulse_Ns': impulse['total_impulse'],
                    'average_impulse_Ns': impulse['average_impulse'],
                    'peak_power_W': impulse['peak_power'],
                    'average_power_W': impulse['average_power'],
                    'force_rate_of_change_Ns': impulse['force_rate_of_change'],
                    
                    # Symmetry metrics
                    'symmetry_index': movement.get('symmetry_index', 0),
                }
                
                # Add RFD metrics from first phase if available
                if phases_data:
                    first_phase = phases_data[0]
                    row.update({
                        'overall_rfd_Ns': first_phase.get('overall_rfd', 0),
                        'early_rfd_Ns': first_phase.get('early_rfd', 0),
                        'peak_rfd_Ns': first_phase.get('peak_rfd', 0),
                        'weight_transfer_time_s': first_phase.get('weight_transfer_time'),
                        'first_peak_force_N': first_phase.get('first_peak_force', 0),
                        'max_peak_force_N': first_phase.get('max_peak_force', 0),
                        'num_peaks_detected': first_phase.get('num_peaks', 0),
                        'force_cv_percent': first_phase.get('force_cv', 0),
                        'force_jerk': first_phase.get('force_jerk', 0),
                        'peak_symmetry': first_phase.get('peak_symmetry', 0),
                        'temporal_symmetry': first_phase.get('temporal_symmetry', 0),
                        'bilateral_index': first_phase.get('bilateral_index', 0),
                        'consistency_score': first_phase.get('consistency_score', 0),
                        'has_unloading_phase': first_phase.get('has_unloading_phase', False),
                        'force_range_N': first_phase.get('force_range', 0),
                        'force_excursion_N': first_phase.get('force_excursion', 0),
                        'mean_force_during_movement_N': first_phase.get('mean_force_during_movement', 0),
                        'sampling_frequency_Hz': first_phase.get('sampling_frequency', 0)
                    })
                
                csv_data.append(row)

        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = os.path.join(reports_dir, "batch_analysis_summary.csv")
            df.to_csv(csv_path, index=False)

        print(f"\nBatch report generated:")
        print(f"  Text report: {report_path}")
        print(f"  CSV summary: {csv_path}")
        print(f"  Plots directory: {os.path.join(output_dir, 'plots')}")

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
        analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
        analog_units = datac3d["parameters"]["ANALOG"].get("UNITS", {}).get("value", ["Unknown"] * len(analog_labels))

        print(f"Available analog channels: {analog_labels}")

        # Check if requested column exists
        if column_name not in analog_labels:
            print(f"Column '{column_name}' not found in analog channels")
            print(f"Available analog channels: {analog_labels}")

            # Try to find a suitable force column automatically
            suggested_column = suggest_force_column(analog_labels)
            if suggested_column:
                print(f"Suggested column: {suggested_column}")
                column_name = suggested_column
            else:
                print("No suitable force column found")
                return None

        # Find the index of the requested column
        column_index = analog_labels.index(column_name)

        # Extract the analog data for this channel
        # analogs shape is typically (1, num_channels, num_frames) for single analog frame
        # or (num_frames, num_channels) depending on the data structure
        analog_data = analogs.squeeze(axis=0)  # Remove first dimension if it's 1

        if analog_data.ndim == 2:
            # Shape is (num_channels, num_frames)
            force_values = analog_data[column_index, :]
        else:
            print(f"Unexpected analog data shape: {analog_data.shape}")
            return None

        # Get timing information
        analog_freq = datac3d["header"]["analogs"]["frame_rate"]
        num_frames = len(force_values)
        time_values = np.arange(num_frames) / analog_freq

        print(f"Using analog channel: {column_name}")
        print(f"Analog frequency: {analog_freq} Hz")
        print(f"Data points: {num_frames}")
        print(f"Time range: {time_values[0]:.3f} - {time_values[-1]:.3f} s")
        print(f"Force range: {force_values.min():.3f} - {force_values.max():.3f}")

        # Create DataFrame
        force_data = pd.DataFrame({
            'Time': time_values,
            'Force': force_values
        })

        return force_data

    except Exception as e:
        print(f"Error reading C3D file: {e}")
        return None


def print_c3d_info(datac3d):
    """
    Print detailed information about the C3D file structure.
    """
    try:
        print("\n" + "="*50)
        print("C3D FILE INFORMATION")
        print("="*50)

        # Basic header info
        print(f"Points frame rate: {datac3d['header']['points']['frame_rate']} Hz")
        print(f"Analog frame rate: {datac3d['header']['analogs']['frame_rate']} Hz")
        print(f"Number of points: {datac3d['parameters']['POINT']['USED']['value'][0]}")
        print(f"Number of analog channels: {datac3d['parameters']['ANALOG']['USED']['value'][0]}")

        # Point/marker information
        if datac3d['parameters']['POINT']['USED']['value'][0] > 0:
            marker_labels = datac3d["parameters"]["POINT"]["LABELS"]["value"]
            print(f"Marker labels: {marker_labels}")

        # Analog information
        analog_labels = datac3d["parameters"]["ANALOG"]["LABELS"]["value"]
        analog_units = datac3d["parameters"]["ANALOG"].get("UNITS", {}).get("value", ["Unknown"] * len(analog_labels))

        print("\nAnalog channels and units:")
        for label, unit in zip(analog_labels, analog_units):
            print(f"  {label}: {unit}")

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

        print("="*50)

    except Exception as e:
        print(f"Error printing C3D info: {e}")


def suggest_force_column(analog_labels):
    """
    Suggests the most appropriate force column from available analog channels.
    """
    # Priority order for force column detection
    force_patterns = [
        # Force plate vertical forces (most common for sit-to-stand)
        'FZ1', 'Fz1', 'Force.Fz1', 'force_z_1',
        'FZ2', 'Fz2', 'Force.Fz2', 'force_z_2',
        'FZ3', 'Fz3', 'Force.Fz3', 'force_z_3',
        'FZ4', 'Fz4', 'Force.Fz4', 'force_z_4',
        # Generic force patterns
        'FZ', 'Fz', 'Force_Z', 'force_z',
        # Any force component
        'FX', 'FY', 'Fx', 'Fy', 'Force_X', 'Force_Y'
    ]

    for pattern in force_patterns:
        for label in analog_labels:
            if pattern.lower() in label.lower():
                return label

    return None


def read_csv_file(file_path, column_name):
    """
    Reads CSV file and extracts the specified column using pandas.

    Parameters:
    -----------
    file_path : str
        Path to CSV file
    column_name : str
        Name of column to extract

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
            if col.lower() in ['time', 'timestamp', 'frame', 't']:
                time_col = col
                break

        # If no time column found, use first column
        if time_col is None:
            time_col = df.columns[0]

        print(f"Using time column: {time_col}")
        print(f"Using force column: {column_name}")

        # Extract relevant data
        force_data = df[[time_col, column_name]].copy()

        # Rename columns for consistency
        force_data.columns = ['Time', 'Force']

        # Ensure Time column is numeric
        force_data['Time'] = pd.to_numeric(force_data['Time'], errors='coerce')

        # Ensure Force column is numeric
        force_data['Force'] = pd.to_numeric(force_data['Force'], errors='coerce')

        # Remove any rows with NaN values
        force_data = force_data.dropna()

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
        force_threshold = config['detection']['force_threshold']
        min_duration = config['detection']['min_duration']
        onset_threshold = config['detection']['onset_threshold']

        # Basic statistics
        total_samples = len(data)
        duration = data['Time'].iloc[-1] - data['Time'].iloc[0] if len(data) > 0 else 0
        mean_force = data['Force'].mean()
        max_force = data['Force'].max()
        min_force = data['Force'].min()

        # Detect sit-to-stand phases
        sit_to_stand_phases = detect_sit_to_stand_phases(
            data['Force'].values,
            data['Time'].values,
            force_threshold,
            min_duration,
            onset_threshold
        )

        # Calculate movement metrics
        movement_metrics = calculate_movement_metrics(data, sit_to_stand_phases)

        # Calculate impulse and power metrics
        impulse_metrics = calculate_impulse_metrics(data, sit_to_stand_phases)

        # Calculate time to peak metrics
        time_to_peak_metrics = calculate_time_to_peak_metrics(data, sit_to_stand_phases)

        results = {
            'total_samples': total_samples,
            'duration': duration,
            'mean_force': mean_force,
            'max_force': max_force,
            'min_force': min_force,
            'sit_to_stand_phases': sit_to_stand_phases,
            'movement_metrics': movement_metrics,
            'impulse_metrics': impulse_metrics,
            'time_to_peak_metrics': time_to_peak_metrics,
            'detection_threshold': force_threshold,
            'status': 'analyzed'
        }

        return results

    except Exception as e:
        print(f"Error in sit-to-stand analysis: {str(e)}")
        return {
            'total_samples': len(data),
            'duration': data['Time'].iloc[-1] - data['Time'].iloc[0] if len(data) > 0 else 0,
            'mean_force': data['Force'].mean(),
            'max_force': data['Force'].max(),
            'min_force': data['Force'].min(),
            'error': str(e),
            'status': 'error'
        }


def detect_sit_to_stand_phases(force_data, time_data, force_threshold, min_duration, onset_threshold, auto_threshold=True):
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
                onset_threshold = detected_threshold
                print(f"Auto-detected onset threshold: {onset_threshold:.2f} N")

        # Find baseline (seated) force level - use 10th percentile for robustness
        baseline_force = np.percentile(force_data, 10)
        print(f"Baseline force (seated): {baseline_force:.2f} N")

        # Detect movement onset (when force exceeds baseline + onset_threshold)
        onset_indices = np.where(force_data > baseline_force + onset_threshold)[0]

        if len(onset_indices) == 0:
            print("No movement onset detected - check threshold settings")
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
            if onset_indices[i] == onset_indices[i-1] + 1:
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
            segment_forces = force_data[start_idx:end_idx+1]
            segment_times = time_data[start_idx:end_idx+1]

            # === PEAK DETECTION ===
            # Find ALL peaks in this segment (important for multi-peak movements in CP)
            all_peaks = detect_all_peaks_in_segment(segment_forces, segment_times, baseline_force)
            
            # Identify first peak (clinically important for initial force generation)
            first_peak = all_peaks[0] if all_peaks else None
            
            # Identify maximum peak
            max_peak = max(all_peaks, key=lambda p: p['force']) if all_peaks else None
            
            # Global peak in segment (for compatibility)
            peak_idx = start_idx + np.argmax(segment_forces)
            peak_force = force_data[peak_idx]
            peak_time = time_data[peak_idx]

            # === TEMPORAL METRICS ===
            phase_duration = time_data[end_idx] - time_data[start_idx]
            
            # Time to first peak (onset to first peak) - Critical for CP assessment
            time_to_first_peak = first_peak['time'] - segment_times[0] if first_peak else None
            
            # Time to maximum force
            time_to_max_force = peak_time - segment_times[0]
            
            # === FORCE DEVELOPMENT METRICS ===
            # Rate of Force Development (RFD) - multiple methods
            
            # 1. Overall RFD (onset to peak)
            force_at_onset = segment_forces[0]
            overall_rfd = (peak_force - force_at_onset) / time_to_max_force if time_to_max_force > 0 else 0
            
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
            force_cv = (np.std(segment_forces) / np.mean(segment_forces)) * 100 if np.mean(segment_forces) > 0 else 0
            
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
            phases.append({
                # Basic temporal metrics
                'phase_number': phase_num + 1,
                'start_time': time_data[start_idx],
                'end_time': time_data[end_idx],
                'duration': phase_duration,
                'start_index': start_idx,
                'end_index': end_idx,
                
                # Peak force metrics
                'peak_force': peak_force,
                'peak_time': peak_time,
                'first_peak_force': first_peak['force'] if first_peak else peak_force,
                'first_peak_time': first_peak['time'] if first_peak else peak_time,
                'max_peak_force': max_peak['force'] if max_peak else peak_force,
                'max_peak_time': max_peak['time'] if max_peak else peak_time,
                'num_peaks': len(all_peaks),
                'all_peaks': all_peaks,
                
                # Time to peak metrics (CRITICAL for CP assessment)
                'time_to_first_peak': time_to_first_peak,
                'time_to_max_force': time_to_max_force,
                'weight_transfer_time': weight_transfer_time,
                
                # Rate of Force Development (RFD) - CRITICAL for CP assessment
                'overall_rfd': overall_rfd,
                'early_rfd': early_rfd,
                'peak_rfd': peak_rfd,
                'peak_rfd_time': peak_rfd_time,
                'rate_of_force_development': overall_rfd,  # Legacy compatibility
                
                # Impulse metrics
                'impulse': total_impulse,
                'impulse_above_baseline': impulse_above_baseline,
                'normalized_impulse': normalized_impulse,
                
                # Power metrics
                'average_power': average_power,
                'peak_power': peak_power,
                'power': average_power,  # Legacy compatibility
                
                # Force variability and smoothness
                'force_cv': force_cv,
                'force_jerk': force_jerk,
                
                # Symmetry metrics
                'symmetry': symmetry_metrics['overall_symmetry'],
                'peak_symmetry': symmetry_metrics['peak_symmetry'],
                'temporal_symmetry': symmetry_metrics['temporal_symmetry'],
                
                # Clinical indicators
                'baseline_force': baseline_force,
                'onset_threshold': onset_threshold,
                'force_at_onset': force_at_onset,
                'has_unloading_phase': has_unloading,
                
                # Additional biomechanical metrics
                'force_range': peak_force - force_at_onset,
                'force_excursion': np.max(segment_forces) - np.min(segment_forces),
                'mean_force_during_movement': np.mean(segment_forces),
                'sampling_frequency': sampling_freq
            })

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
            is_local_max = forces[i] > forces[i-1] and forces[i] > forces[i+1]
            
            if is_local_max:
                # Calculate prominence (height above nearby valleys)
                left_valley = np.min(forces[max(0, i-10):i]) if i > 0 else forces[i]
                right_valley = np.min(forces[i:min(len(forces), i+10)]) if i < len(forces)-1 else forces[i]
                prominence = forces[i] - max(left_valley, right_valley)
                
                # Only include peaks with sufficient prominence and above baseline
                if prominence >= min_prominence and forces[i] > baseline_force + min_prominence:
                    peaks.append({
                        'time': times[i],
                        'force': forces[i],
                        'index': i,
                        'prominence': prominence,
                        'above_baseline': forces[i] - baseline_force
                    })
        
        # If no peaks found with prominence filter, find at least the maximum
        if not peaks:
            max_idx = np.argmax(forces)
            peaks.append({
                'time': times[max_idx],
                'force': forces[max_idx],
                'index': max_idx,
                'prominence': forces[max_idx] - np.min(forces),
                'above_baseline': forces[max_idx] - baseline_force
            })
        
        # Sort peaks by time (chronological order)
        peaks = sorted(peaks, key=lambda x: x['time'])
        
        return peaks

    except Exception as e:
        print(f"Error detecting peaks in segment: {str(e)}")
        # Return at least the maximum as fallback
        max_idx = np.argmax(forces)
        return [{
            'time': times[max_idx],
            'force': forces[max_idx],
            'index': max_idx,
            'prominence': 0,
            'above_baseline': forces[max_idx] - baseline_force
        }]


def find_peaks_in_segment(forces, times):
    """Legacy function - kept for compatibility. Use detect_all_peaks_in_segment for new code."""
    try:
        # Simple peak detection - find local maxima
        peaks = []
        for i in range(1, len(forces) - 1):
            if forces[i] > forces[i-1] and forces[i] > forces[i+1]:
                peaks.append({
                    'time': times[i],
                    'force': forces[i],
                    'index': i
                })

        return sorted(peaks, key=lambda x: x['force'], reverse=True)

    except Exception as e:
        print(f"Error finding peaks in segment: {str(e)}")
        return []


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
        'overall_symmetry': 1.0,
        'peak_symmetry': 1.0,
        'temporal_symmetry': 1.0,
        'bilateral_index': 0.0,
        'consistency_score': 1.0
    }
    
    if len(all_peaks) < 2:
        return metrics  # Perfect symmetry if only one peak
    
    try:
        # Extract peak characteristics
        peak_forces = [peak['force'] for peak in all_peaks]
        peak_times = [peak['time'] for peak in all_peaks]
        
        # === PEAK FORCE SYMMETRY ===
        # Coefficient of variation of peak forces (lower = more symmetric)
        peak_force_cv = (np.std(peak_forces) / np.mean(peak_forces)) * 100 if np.mean(peak_forces) > 0 else 0
        peak_symmetry = 1.0 - min(peak_force_cv / 100.0, 1.0)  # Convert to 0-1 scale
        
        # === TEMPORAL SYMMETRY ===
        # Time distribution of peaks
        time_range = max(peak_times) - min(peak_times)
        if time_range > 0 and len(peak_times) > 1:
            expected_spacing = time_range / (len(peak_times) - 1)
            actual_spacings = [peak_times[i+1] - peak_times[i] for i in range(len(peak_times)-1)]
            time_cv = (np.std(actual_spacings) / np.mean(actual_spacings)) * 100 if np.mean(actual_spacings) > 0 else 0
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
        overall_symmetry = (peak_symmetry * 0.4 + temporal_symmetry * 0.3 + bilateral_index * 0.3)
        
        # === CONSISTENCY SCORE ===
        # How smooth and consistent is the force profile
        force_gradient = np.gradient(forces, times)
        gradient_variability = np.std(force_gradient) / (np.mean(np.abs(force_gradient)) + 1e-6)
        consistency_score = 1.0 / (1.0 + gradient_variability)  # Higher = more consistent
        
        metrics['overall_symmetry'] = overall_symmetry
        metrics['peak_symmetry'] = peak_symmetry
        metrics['temporal_symmetry'] = temporal_symmetry
        metrics['bilateral_index'] = bilateral_index
        metrics['consistency_score'] = consistency_score
        
    except Exception as e:
        print(f"Error calculating detailed symmetry: {str(e)}")
    
    return metrics


def find_peaks_in_segment(forces, times):
    """Legacy function - kept for compatibility. Use detect_all_peaks_in_segment for new code."""
    try:
        # Simple peak detection - find local maxima
        peaks = []
        for i in range(1, len(forces) - 1):
            if forces[i] > forces[i-1] and forces[i] > forces[i+1]:
                peaks.append({
                    'time': times[i],
                    'force': forces[i],
                    'index': i
                })

        return sorted(peaks, key=lambda x: x['force'], reverse=True)

    except Exception as e:
        print(f"Error finding peaks in segment: {str(e)}")
        return []


def calculate_symmetry(peaks):
    """
    Legacy symmetry calculation - kept for backward compatibility.
    For new code, use calculate_detailed_symmetry which provides more clinical metrics.
    """
    if len(peaks) < 2:
        return 1.0  # Perfect symmetry if only one peak

    try:
        # Calculate symmetry based on timing and force distribution
        times = [peak['time'] for peak in peaks]
        forces = [peak['force'] for peak in peaks]

        # Time-based symmetry (how evenly distributed in time)
        time_range = max(times) - min(times)
        if time_range > 0:
            expected_spacing = time_range / (len(times) - 1)
            actual_spacing = [times[i+1] - times[i] for i in range(len(times)-1)]
            time_variance = np.var(actual_spacing) / (expected_spacing ** 2) if expected_spacing > 0 else 0
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
        'num_phases': len(phases),
        'total_movement_time': 0,
        'average_phase_duration': 0,
        'phases_per_minute': 0,
        'symmetry_index': None,
        'average_time_to_peak': 0,
        'time_to_peak_variation': 0,
        'average_rate_of_force_development': 0,
        'peak_force_consistency': 0,
        'movement_efficiency': 0,
        'average_overall_rfd': 0,
        'average_early_rfd': 0,
        'average_peak_rfd': 0,
        'average_bilateral_index': 0,
        'average_consistency_score': 0
    }

    if not phases:
        return metrics

    try:
        # Calculate basic temporal metrics
        durations = [phase['duration'] for phase in phases]
        
        # Time to peak metrics (handle both new and legacy format)
        time_to_max = [phase.get('time_to_max_force', phase.get('time_to_peak', 0)) for phase in phases]
        time_to_first = [phase.get('time_to_first_peak', phase.get('time_to_peak', 0)) for phase in phases]
        
        peak_forces = [phase['peak_force'] for phase in phases]
        rates_of_force_dev = [phase.get('overall_rfd', phase.get('rate_of_force_development', 0)) for phase in phases]

        metrics['total_movement_time'] = sum(durations)
        metrics['average_phase_duration'] = np.mean(durations) if durations else 0
        metrics['average_time_to_peak'] = np.mean(time_to_max) if time_to_max else 0
        metrics['time_to_peak_variation'] = np.std(time_to_max) if len(time_to_max) > 1 else 0
        metrics['average_rate_of_force_development'] = np.mean(rates_of_force_dev) if rates_of_force_dev else 0

        # Enhanced RFD metrics
        overall_rfds = [phase.get('overall_rfd', 0) for phase in phases]
        early_rfds = [phase.get('early_rfd', 0) for phase in phases]
        peak_rfds = [phase.get('peak_rfd', 0) for phase in phases]
        
        metrics['average_overall_rfd'] = np.mean(overall_rfds) if overall_rfds else 0
        metrics['average_early_rfd'] = np.mean(early_rfds) if early_rfds else 0
        metrics['average_peak_rfd'] = np.mean(peak_rfds) if peak_rfds else 0

        # Calculate phases per minute if duration > 0
        total_duration = data['Time'].iloc[-1] - data['Time'].iloc[0]
        if total_duration > 0:
            metrics['phases_per_minute'] = (len(phases) / total_duration) * 60

        # Calculate peak force consistency
        if len(peak_forces) > 1:
            metrics['peak_force_consistency'] = 1 - (np.std(peak_forces) / (np.mean(peak_forces) + 1e-6))

        # Calculate symmetry using enhanced method
        symmetry_values = [phase.get('symmetry', phase.get('overall_symmetry', 1.0)) for phase in phases]
        bilateral_indices = [phase.get('bilateral_index', 0.5) for phase in phases]
        consistency_scores = [phase.get('consistency_score', 1.0) for phase in phases]
        
        metrics['symmetry_index'] = np.mean(symmetry_values) if symmetry_values else None
        metrics['average_bilateral_index'] = np.mean(bilateral_indices) if bilateral_indices else 0
        metrics['average_consistency_score'] = np.mean(consistency_scores) if consistency_scores else 0

        # Calculate movement efficiency (time to peak / total duration ratio)
        if metrics['total_movement_time'] > 0:
            metrics['movement_efficiency'] = metrics['average_time_to_peak'] / metrics['total_movement_time']

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
        'total_impulse': 0,
        'average_impulse': 0,
        'peak_power': 0,
        'average_power': 0,
        'force_rate_of_change': 0
    }

    if not phases:
        return metrics

    # Calculate impulse metrics
    impulses = [phase['impulse'] for phase in phases]
    metrics['total_impulse'] = sum(impulses)
    metrics['average_impulse'] = np.mean(impulses)

    # Calculate power metrics (simplified)
    for phase in phases:
        duration = phase['duration']
        if duration > 0:
            power = phase['impulse'] / duration
            metrics['peak_power'] = max(metrics['peak_power'], power)

    metrics['average_power'] = metrics['total_impulse'] / sum([p['duration'] for p in phases])

    # Calculate rate of force development (first phase only)
    if phases:
        first_phase = phases[0]
        force_change = first_phase['peak_force'] - data['Force'].iloc[first_phase['start_index']]
        time_change = first_phase['peak_time'] - data['Time'].iloc[first_phase['start_index']]
        if time_change > 0:
            metrics['force_rate_of_change'] = force_change / time_change

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
        'time_to_first_peak': None,
        'time_to_max_force': None,
        'average_time_to_peak': 0,
        'time_to_peak_variation': 0,
        'peak_timing_consistency': 0
    }

    if not phases:
        return metrics

    try:
        # Calculate time to first peak (from movement onset)
        time_to_peaks = []
        for phase in phases:
            # Find the peak in this phase
            phase_data = data[(data['Time'] >= phase['start_time']) &
                             (data['Time'] <= phase['end_time'])]
            if len(phase_data) > 0:
                peak_idx = phase_data['Force'].idxmax()
                peak_time = data.loc[peak_idx, 'Time']
                time_to_peak = peak_time - phase['start_time']
                time_to_peaks.append(time_to_peak)

        if time_to_peaks:
            metrics['time_to_first_peak'] = time_to_peaks[0] if time_to_peaks else None
            metrics['average_time_to_peak'] = np.mean(time_to_peaks)
            metrics['time_to_peak_variation'] = np.std(time_to_peaks)

            # Overall time to maximum force
            overall_peak_idx = data['Force'].idxmax()
            overall_peak_time = data.loc[overall_peak_idx, 'Time']
            metrics['time_to_max_force'] = overall_peak_time - data['Time'].iloc[0]

            # Consistency measure (lower variation = higher consistency)
            if metrics['average_time_to_peak'] > 0:
                metrics['peak_timing_consistency'] = 1 - (metrics['time_to_peak_variation'] / metrics['average_time_to_peak'])

    except Exception as e:
        print(f"Error calculating time to peak metrics: {str(e)}")

    return metrics


def save_force_plot_png(data, phases, output_path, config):
    """
    Saves a comprehensive PNG plot of the force data with all clinical metrics.
    Enhanced visualization for pediatric cerebral palsy assessment.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns
    phases : list
        List of detected sit-to-stand phases with comprehensive metrics
    output_path : str
        Path where to save the PNG file
    config : dict
        Configuration dictionary
    """
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Main plot (force vs time)
        ax1 = plt.subplot(2, 1, 1)
        
        # Plot force data
        ax1.plot(data['Time'], data['Force'], 'b-', linewidth=1.5, alpha=0.7, label='Force Data')

        # Plot detected phases with enhanced visualization
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'yellow']
        for i, phase in enumerate(phases[:8]):  # Show up to 8 phases
            color = colors[i % len(colors)]
            phase_data = data[(data['Time'] >= phase['start_time']) &
                             (data['Time'] <= phase['end_time'])]

            # Plot phase with enhanced info
            rfd = phase.get('overall_rfd', 0)
            plt.plot(phase_data['Time'], phase_data['Force'],
                    color=color, linewidth=2.5, alpha=0.8,
                    label=f'Phase {i+1} (Peak: {phase["peak_force"]:.1f}N, RFD: {rfd:.0f}N/s)')

            # Mark ALL peaks in this phase
            if 'all_peaks' in phase:
                for peak in phase['all_peaks']:
                    peak_time = peak['time']
                    peak_force = peak['force']
                    ax1.plot(peak_time, peak_force, 'o', color=color, markersize=10, 
                            markeredgecolor='black', markeredgewidth=1.5)
                    # Annotate first peak
                    if peak == phase['all_peaks'][0]:
                        ax1.annotate(f'1st Peak\n{peak_force:.1f}N', 
                                    xy=(peak_time, peak_force),
                                    xytext=(10, 10), textcoords='offset points',
                                    fontsize=8, fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

            # Mark movement onset
            ax1.plot(phase['start_time'], data.loc[phase['start_index'], 'Force'], 
                    '^', color=color, markersize=8, label=f'Onset {i+1}')

        # Add baseline and threshold lines
        baseline = np.percentile(data['Force'], 10)
        threshold = config['detection']['force_threshold']

        ax1.axhline(y=baseline, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (seated)')
        ax1.axhline(y=threshold, color='red', linestyle=':', linewidth=2, alpha=0.7, label='Detection Threshold')

        # Customize main plot
        ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Vertical Force - Fz (N)', fontsize=12, fontweight='bold')
        ax1.set_title('Sit-to-Stand Force Analysis - Comprehensive Clinical Assessment', 
                     fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Add comprehensive statistics text box
        if phases:
            first_phase = phases[0]
            stats_text = f"""CLINICAL METRICS:
Duration: {data['Time'].iloc[-1] - data['Time'].iloc[0]:.2f} s
Phases: {len(phases)}

FIRST PHASE:
Time to 1st Peak: {first_phase.get('time_to_first_peak', 0):.3f} s
Time to Max Force: {first_phase.get('time_to_max_force', 0):.3f} s
Overall RFD: {first_phase.get('overall_rfd', 0):.1f} N/s
Early RFD: {first_phase.get('early_rfd', 0):.1f} N/s
Peak RFD: {first_phase.get('peak_rfd', 0):.1f} N/s

FORCE:
Max: {data['Force'].max():.1f} N
Mean: {data['Force'].mean():.1f} N
Range: {first_phase.get('force_range', 0):.1f} N

QUALITY:
Force CV: {first_phase.get('force_cv', 0):.1f}%
Bilateral Index: {first_phase.get('bilateral_index', 0):.2f}
Consistency: {first_phase.get('consistency_score', 0):.2f}
Peaks: {first_phase.get('num_peaks', 0)}
"""
        else:
            stats_text = "No sit-to-stand phases detected"

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))

        # === SUBPLOT 2: Force Rate (RFD visualization) ===
        ax2 = plt.subplot(2, 1, 2)
        
        # Calculate and plot force rate of change
        force_rate = np.gradient(data['Force'].values, data['Time'].values)
        ax2.plot(data['Time'], force_rate, 'g-', linewidth=1, alpha=0.6, label='Force Rate (RFD)')
        
        # Mark phases
        for i, phase in enumerate(phases[:8]):
            color = colors[i % len(colors)]
            phase_data_idx = (data['Time'] >= phase['start_time']) & (data['Time'] <= phase['end_time'])
            phase_time = data.loc[phase_data_idx, 'Time']
            phase_rate = force_rate[phase_data_idx]
            
            ax2.plot(phase_time, phase_rate, color=color, linewidth=2, alpha=0.7)
            
            # Mark peak RFD
            if 'peak_rfd_time' in phase:
                peak_rfd_idx = np.argmin(np.abs(data['Time'] - phase['peak_rfd_time']))
                ax2.plot(data['Time'].iloc[peak_rfd_idx], force_rate[peak_rfd_idx], 
                        '*', color=color, markersize=15, markeredgecolor='black', markeredgewidth=1)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Rate of Force Development (N/s)', fontsize=12, fontweight='bold')
        ax2.set_title('Force Development Rate (RFD) - Critical for CP Motor Control Assessment', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comprehensive force plot saved to: {output_path}")

    except Exception as e:
        print(f"Error saving force plot: {str(e)}")
        import traceback
        traceback.print_exc()


# display_results function removed - not needed in batch mode, results are saved automatically


class SitToStandGUI:
    """
    Simple GUI for sit-to-stand analysis configuration using basic tkinter.
    Cross-platform compatible and minimalistic.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sit-to-Stand Analysis")
        self.root.geometry("500x300")
        self.root.resizable(True, True)

        # Variables for storing user selections
        self.config_file = ""
        self.input_dir = ""
        self.output_dir = ""
        self.file_format = tk.StringVar(value="auto")  # auto, c3d, csv

        # Use defaults by default
        self.use_defaults = True

        self.create_widgets()

    def create_widgets(self):
        """Creates the simple GUI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(main_frame, text="Sit-to-Stand Analysis Setup",
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))

        # Config file row
        config_frame = tk.Frame(main_frame)
        config_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(config_frame, text="Config file:").pack(side=tk.LEFT)
        self.config_entry = tk.Entry(config_frame, width=40)
        self.config_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        config_btn = tk.Button(config_frame, text="Browse",
                              command=self.browse_config_file, width=8)
        config_btn.pack(side=tk.LEFT)

        # Input directory row
        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(input_frame, text="Input dir:").pack(side=tk.LEFT)
        self.input_entry = tk.Entry(input_frame, width=40)
        self.input_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        input_btn = tk.Button(input_frame, text="Browse",
                             command=self.browse_input_dir, width=8)
        input_btn.pack(side=tk.LEFT)

        # Output directory row
        output_frame = tk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(output_frame, text="Output dir:").pack(side=tk.LEFT)
        self.output_entry = tk.Entry(output_frame, width=40)
        self.output_entry.pack(side=tk.LEFT, padx=(10, 5), fill=tk.X, expand=True)

        output_btn = tk.Button(output_frame, text="Browse",
                              command=self.browse_output_dir, width=8)
        output_btn.pack(side=tk.LEFT)

        # File format selection
        format_frame = tk.Frame(main_frame)
        format_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(format_frame, text="File format:").pack(side=tk.LEFT)
        format_combo = ttk.Combobox(format_frame, textvariable=self.file_format,
                                   values=["auto", "c3d", "csv"], state="readonly", width=10)
        format_combo.pack(side=tk.LEFT, padx=(10, 5))
        format_combo.set("auto")

        # Options frame
        options_frame = tk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(10, 20))

        self.defaults_var = tk.BooleanVar(value=True)
        defaults_cb = tk.Checkbutton(options_frame, text="Use default config",
                                    variable=self.defaults_var,
                                    command=self.toggle_defaults)
        defaults_cb.pack(side=tk.LEFT)

        # Create default config file button
        create_config_btn = tk.Button(options_frame, text="Create Default Config",
                                     command=self.create_default_config_file, width=20)
        create_config_btn.pack(side=tk.LEFT, padx=(20, 0))

        # Buttons frame
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        run_btn = tk.Button(btn_frame, text="Run Analysis",
                           command=self.run_analysis, bg="#4CAF50", fg="white",
                           font=("Arial", 10, "bold"), padx=20)
        run_btn.pack(side=tk.RIGHT, padx=(10, 0))

        cancel_btn = tk.Button(btn_frame, text="Cancel",
                              command=self.root.quit, padx=20)
        cancel_btn.pack(side=tk.RIGHT)

        # Status label
        self.status_label = tk.Label(main_frame, text="Ready",
                                    font=("Arial", 9), fg="blue")
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
        if self.use_defaults:
            return

        filename = filedialog.askopenfilename(
            title="Select TOML Configuration File",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")]
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
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")]
        )

        if filename:
            try:
                import toml
                with open(filename, 'w') as f:
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
        file_format = self.file_format.get()

        # Validation
        if not input_dir:
            self.status_label.config(text="Error: Please select input directory", fg="red")
            return

        if not self.use_defaults and not config_file:
            self.status_label.config(text="Error: Please select config file or use defaults", fg="red")
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

        # Add file format
        cli_args.append(file_format)

        # Update status
        self.status_label.config(text="Running analysis...", fg="orange")
        self.root.update()

        try:
            # Run in CLI mode with the prepared arguments
            result = run_cli_mode(cli_args)

            if result is None:  # Success
                self.status_label.config(text="Analysis completed successfully!", fg="green")
                messagebox.showinfo("Success", "Sit-to-Stand analysis completed successfully!")
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
