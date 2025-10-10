"""
================================================================================
sit2stand.py - Sit to Stand Analysis Module
================================================================================
Author: Prof. Paulo Santiago
Create: 10 October 2025
Update: 10 October 2025
Version: 0.2

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
import toml
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


def main():
    """
    Main function to run the sit-to-stand analysis in batch mode.
    Processes multiple C3D and CSV files automatically using TOML configuration.
    """
    print("Starting Sit-to-Stand Batch Analysis...")
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")

    # Parse command line arguments
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sit2stand.py <config.toml> <input_directory> [output_directory]")
        print("Example: python sit2stand.py config.toml /path/to/c3d/files /path/to/output")
        return

    config_file = sys.argv[1]
    input_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None

    try:
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

        # Step 3: Find all C3D and CSV files
        files = find_analysis_files(input_dir)
        if not files:
            print(f"No C3D or CSV files found in {input_dir}")
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


def load_toml_config(config_file):
    """
    Loads TOML configuration file.

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
        if not os.path.exists(config_file):
            print(f"Configuration file not found: {config_file}")
            return None

        config = toml.load(config_file)
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
        return None


def find_analysis_files(input_dir):
    """
    Finds all C3D and CSV files in the input directory.

    Parameters:
    -----------
    input_dir : str
        Input directory path

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

    # Supported extensions
    extensions = ['.c3d', '.csv']

    for ext in extensions:
        pattern = os.path.join(input_dir, f"*{ext}")
        files.extend(glob.glob(pattern))

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

            # Detailed results
            total_phases = 0
            for result in results:
                if 'error' not in result:
                    f.write(f"File: {result['filename']}\n")
                    analysis = result['analysis']

                    # Basic metrics
                    f.write(f"  Duration: {analysis['duration']:.2f} s\n")
                    f.write(f"  Mean Force: {analysis['mean_force']:.2f} N\n")
                    f.write(f"  Max Force: {analysis['max_force']:.2f} N\n")

                    # Movement metrics
                    movement = analysis['movement_metrics']
                    f.write(f"  Phases Detected: {movement['num_phases']}\n")
                    f.write(f"  Total Movement Time: {movement['total_movement_time']:.2f} s\n")
                    f.write(f"  Average Phase Duration: {movement['average_phase_duration']:.2f} s\n")

                    if movement['phases_per_minute'] > 0:
                        f.write(f"  Phases per Minute: {movement['phases_per_minute']:.1f}\n")

                    # Impulse metrics
                    impulse = analysis['impulse_metrics']
                    f.write(f"  Total Impulse: {impulse['total_impulse']:.2f} N⋅s\n")
                    f.write(f"  Average Impulse: {impulse['average_impulse']:.2f} N⋅s\n")

                    # Time to peak metrics
                    time_to_peak = analysis['time_to_peak_metrics']
                    if time_to_peak['time_to_first_peak'] is not None:
                        f.write(f"  Time to First Peak: {time_to_peak['time_to_first_peak']:.3f} s\n")
                    if time_to_peak['time_to_max_force'] is not None:
                        f.write(f"  Time to Max Force: {time_to_peak['time_to_max_force']:.3f} s\n")

                    # Plot information
                    if 'plot_path' in result:
                        f.write(f"  Plot saved: {os.path.basename(result['plot_path'])}\n")

                    f.write("-" * 30 + "\n")
                    total_phases += movement['num_phases']
                else:
                    f.write(f"ERROR - {result['filename']}: {result['error']}\n")
                    f.write("-" * 30 + "\n")

            f.write(f"\nOverall Summary:\n")
            f.write(f"Total phases detected across all files: {total_phases}\n")
            f.write(f"Average phases per successful file: {total_phases/successful_analyses:.1f}\n")

        # Generate CSV summary
        csv_data = []
        for result in results:
            if 'error' not in result:
                analysis = result['analysis']
                movement = analysis['movement_metrics']
                impulse = analysis['impulse_metrics']
                time_to_peak = analysis['time_to_peak_metrics']

                row = {
                    'filename': result['filename'],
                    'duration': analysis['duration'],
                    'mean_force': analysis['mean_force'],
                    'max_force': analysis['max_force'],
                    'min_force': analysis['min_force'],
                    'num_phases': movement['num_phases'],
                    'total_movement_time': movement['total_movement_time'],
                    'average_phase_duration': movement['average_phase_duration'],
                    'phases_per_minute': movement['phases_per_minute'],
                    'total_impulse': impulse['total_impulse'],
                    'average_impulse': impulse['average_impulse'],
                    'peak_power': impulse['peak_power'],
                    'average_power': impulse['average_power'],
                    'force_rate_of_change': impulse['force_rate_of_change'],
                    'time_to_first_peak': time_to_peak['time_to_first_peak'],
                    'time_to_max_force': time_to_peak['time_to_max_force'],
                    'symmetry_index': movement['symmetry_index']
                }
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


def detect_sit_to_stand_phases(force_data, time_data, force_threshold, min_duration, onset_threshold):
    """
    Detects sit-to-stand phases in force data.

    Parameters:
    -----------
    force_data : array-like
        Force values
    time_data : array-like
        Time values
    force_threshold : float
        Minimum force threshold for movement detection
    min_duration : float
        Minimum duration for a valid phase
    onset_threshold : float
        Threshold for movement onset detection

    Returns:
    --------
    list
        List of detected phases with start/end times and metrics
    """
    phases = []

    try:
        # Find baseline (seated) force level
        baseline_force = np.percentile(force_data, 10)  # Use 10th percentile as baseline

        # Detect movement onset (when force exceeds baseline + onset_threshold)
        onset_indices = np.where(force_data > baseline_force + onset_threshold)[0]

        if len(onset_indices) == 0:
            return phases

        # Find continuous segments
        movement_segments = []
        current_segment = [onset_indices[0]]

        for i in range(1, len(onset_indices)):
            if onset_indices[i] == onset_indices[i-1] + 1:
                current_segment.append(onset_indices[i])
            else:
                if len(current_segment) >= min_duration * (len(time_data) / time_data[-1]):
                    movement_segments.append(current_segment)
                current_segment = [onset_indices[i]]

        # Add the last segment if valid
        if len(current_segment) >= min_duration * (len(time_data) / time_data[-1]):
            movement_segments.append(current_segment)

        # Convert segments to phase information
        for segment in movement_segments:
            start_idx = segment[0]
            end_idx = segment[-1]

            # Find peak force in this segment
            segment_forces = force_data[start_idx:end_idx+1]
            peak_idx = start_idx + np.argmax(segment_forces)
            peak_force = force_data[peak_idx]
            peak_time = time_data[peak_idx]

            # Calculate phase duration
            phase_duration = time_data[end_idx] - time_data[start_idx]

            # Calculate force integral (impulse)
            phase_impulse = np.trapz(segment_forces, time_data[start_idx:end_idx+1])

            phases.append({
                'start_time': time_data[start_idx],
                'end_time': time_data[end_idx],
                'duration': phase_duration,
                'peak_force': peak_force,
                'peak_time': peak_time,
                'impulse': phase_impulse,
                'start_index': start_idx,
                'end_index': end_idx
            })

    except Exception as e:
        print(f"Error detecting sit-to-stand phases: {str(e)}")

    return phases


def calculate_movement_metrics(data, phases):
    """
    Calculates movement-specific metrics from detected phases.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns
    phases : list
        List of detected sit-to-stand phases

    Returns:
    --------
    dict
        Movement metrics
    """
    metrics = {
        'num_phases': len(phases),
        'total_movement_time': 0,
        'average_phase_duration': 0,
        'phases_per_minute': 0,
        'symmetry_index': None
    }

    if not phases:
        return metrics

    # Calculate basic metrics
    durations = [phase['duration'] for phase in phases]
    metrics['total_movement_time'] = sum(durations)
    metrics['average_phase_duration'] = np.mean(durations)

    # Calculate phases per minute if duration > 0
    total_duration = data['Time'].iloc[-1] - data['Time'].iloc[0]
    if total_duration > 0:
        metrics['phases_per_minute'] = (len(phases) / total_duration) * 60

    # Calculate symmetry if multiple phases
    if len(phases) > 1:
        # Simple symmetry measure based on peak force variation
        peak_forces = [phase['peak_force'] for phase in phases]
        if len(peak_forces) > 1:
            metrics['symmetry_index'] = 1 - (np.std(peak_forces) / np.mean(peak_forces))

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
    Saves a PNG plot of the force data with detected phases.

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with Time and Force columns
    phases : list
        List of detected sit-to-stand phases
    output_path : str
        Path where to save the PNG file
    config : dict
        Configuration dictionary
    """
    try:
        plt.figure(figsize=(12, 8))

        # Plot force data
        plt.plot(data['Time'], data['Force'], 'b-', linewidth=1.5, alpha=0.7, label='Force Data')

        # Plot detected phases
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, phase in enumerate(phases[:5]):  # Show first 5 phases
            color = colors[i % len(colors)]
            phase_data = data[(data['Time'] >= phase['start_time']) &
                             (data['Time'] <= phase['end_time'])]

            plt.plot(phase_data['Time'], phase_data['Force'],
                    color=color, linewidth=2.5, alpha=0.8,
                    label=f'Phase {i+1} (Peak: {phase["peak_force"]:.1f}N)')

            # Mark peak
            peak_data = phase_data[phase_data['Force'] == phase['peak_force']]
            if not peak_data.empty:
                plt.plot(peak_data['Time'].iloc[0], peak_data['Force'].iloc[0],
                        'o', color=color, markersize=8)

        # Add baseline and threshold lines
        baseline = np.percentile(data['Force'], 10)
        threshold = config['detection']['force_threshold']

        plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7, label='Baseline')
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label='Threshold')

        # Customize plot
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Force (N)', fontsize=12)
        plt.title('Sit-to-Stand Force Analysis', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f"""
        Total Duration: {data['Time'].iloc[-1] - data['Time'].iloc[0]:.2f} s
        Mean Force: {data['Force'].mean():.2f} N
        Max Force: {data['Force'].max():.2f} N
        Phases Detected: {len(phases)}
        """

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Force plot saved to: {output_path}")

    except Exception as e:
        print(f"Error saving force plot: {str(e)}")


# display_results function removed - not needed in batch mode, results are saved automatically


if __name__ == "__main__":
    main()
