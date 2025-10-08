# Interpolation and Smoothing Tool - Help Guide

## Overview

The **Interpolation and Smoothing Tool** (`interp_smooth_split.py`) is a comprehensive data processing module designed for biomechanical and time-series data analysis. It provides advanced gap filling and smoothing capabilities to clean and process your CSV data files.

## Key Features

- **Gap Filling**: Fill missing data points using various interpolation methods
- **Data Smoothing**: Apply advanced smoothing filters to reduce noise
- **Quality Analysis**: Visualize data quality with comprehensive plots
- **Batch Processing**: Process multiple CSV files simultaneously
- **Configuration Management**: Save and load processing configurations

## Getting Started

### 1. Launching the Tool

The tool can be launched in two ways:

**From VAILA Main Interface:**
- Click on "Smooth_Fill_Split" button in the main VAILA interface

**Direct Execution:**
```bash
python vaila/interp_smooth_split.py
```

### 2. Basic Workflow

1. **Configure Parameters**: Set interpolation and smoothing methods
2. **Load Test Data** (Optional): Test your configuration with sample data
3. **Analyze Quality** (Optional): Visualize the effects of your settings
4. **Select Source Directory**: Choose folder containing CSV files to process
5. **Process Files**: Apply settings to all CSV files in the directory

## Configuration Options

### Gap Filling Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **1 - Linear Interpolation** | Straight lines between points | Most general cases, smooth data |
| **2 - Cubic Spline** | Smooth curves between points | Natural-looking transitions |
| **3 - Nearest Value** | Copy closest available value | Categorical or discrete data |
| **4 - Kalman Filter** | Predictive filling with physics modeling | Movement data, tracking |
| **5 - None** | Leave gaps as NaN | When gaps should remain |
| **6 - Skip** | Keep original data, apply only smoothing | When interpolation is not needed |

### Smoothing Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **1 - None** | No smoothing applied | When data is already clean |
| **2 - Savitzky-Golay** | Preserves peaks and valleys | Biomechanical data, preserving features |
| **3 - LOWESS** | Local regression, adapts to trends | Noisy data, non-linear trends |
| **4 - Kalman Filter** | State estimation with noise reduction | Tracking data, real-time processing |
| **5 - Butterworth** | Frequency domain filtering | Biomechanics standard, remove high frequencies |
| **6 - Spline Smoothing** | Flexible curve fitting | Very smooth curves, artistic smoothing |
| **7 - ARIMA** | Time series modeling | Complex temporal patterns |

## Parameter Reference Guide

### Savitzky-Golay Filter

**Window Length (must be odd):**
- **Range**: 5-31
- **Smooth data**: 5-15
- **Noisy data**: 15-31
- **Tip**: Must be odd number, larger = more smoothing

**Polynomial Order:**
- **Range**: 2-5
- **Most cases**: 2-3
- **Tip**: Must be < window length, higher = more flexible

### LOWESS (Locally Weighted Scatterplot Smoothing)

**Fraction (0-1):**
- **Smooth data**: 0.1-0.3
- **Noisy data**: 0.3-0.5
- **Tip**: Higher values = smoother results

**Iterations:**
- **Range**: 2-4
- **Most cases**: 3
- **Tip**: More iterations = more robust to outliers

### Butterworth Filter

**Cutoff Frequency (Hz):**
- **Biomechanics**: 4-10 Hz
- **Slow movements**: 1-5 Hz
- **Tip**: Must be < sampling frequency/2 (Nyquist frequency)

**Sampling Frequency (Hz):**
- **Video data**: 30 Hz
- **Motion capture**: 100-1000 Hz
- **Tip**: Must be > 2×cutoff frequency

### Kalman Filter

**EM Iterations:**
- **Range**: 3-10
- **Most cases**: 5
- **Tip**: More iterations = better parameter estimation but slower

**Processing Mode:**
- **1**: Process each column independently
- **2**: Process x,y pairs together
- **Tip**: Use mode 2 for coordinate data

### Spline Smoothing

**Smoothing Factor (s):**
- **Light smoothing**: 0.1-1.0
- **Moderate smoothing**: 1.0-10
- **Strong smoothing**: 10+
- **Tip**: Higher values = more smoothing

### ARIMA (AutoRegressive Integrated Moving Average)

**AR Order (p):**
- **Most cases**: 1-3
- **Complex patterns**: Higher values
- **Tip**: Autoregressive terms

**Difference Order (d):**
- **Stationary data**: 0
- **Trending data**: 1-2
- **Tip**: Makes data stationary

**MA Order (q):**
- **Most cases**: 0-2
- **Complex noise**: Higher values
- **Tip**: Moving average terms

## General Parameters

### Padding Configuration

**Padding Length (% of data):**
- **Range**: 0-100%
- **Recommended**: 10%
- **Purpose**: Avoid edge effects in filtering
- **Tip**: Higher padding = better edge handling

### Gap Configuration

**Maximum Gap Size (frames):**
- **Range**: 0 (no limit) to any positive number
- **Recommended**: 60 frames (2 seconds at 30fps)
- **Purpose**: Only fill gaps smaller than this value
- **Tip**: 0 = fill all gaps, larger values = more conservative

### Split Configuration

**Split Data:**
- **Enabled**: Split data into two equal parts
- **Purpose**: Create separate files for analysis
- **Output**: Two files with "_part1" and "_part2" suffixes

## Quality Analysis Features

The tool includes a comprehensive quality analysis system:

### 1. Load Test Data
- Load a CSV file to test your configuration
- Preview how your settings will affect the data

### 2. Analyze Quality
- **Original vs Processed**: Compare before and after
- **Residuals Analysis**: Check for remaining patterns
- **Derivatives**: First and second derivatives
- **Distribution**: Histogram of residuals
- **Frequency Spectrum**: FFT analysis

### 3. Residual Filtering
- Apply the same smoothing method to residuals
- Detect if signal remains in the residuals
- Verify filtering effectiveness

## File Processing

### Input Requirements
- **Format**: CSV files
- **Structure**: First column should be frame numbers or time
- **Data**: Numeric columns for processing

### Output Structure
- **Directory**: Created with descriptive name including methods and timestamp
- **Files**: Processed files with method suffix
- **Report**: Detailed processing report (processing_report.txt)

### Output Naming Convention
```
original_filename_method_timestamp.csv
```

Examples:
- `data_savgol_20241216_143022.csv`
- `motion_butterworth_cut4_20241216_143022.csv`
- `tracking_lowess_frac30_it3_20241216_143022.csv`

## Configuration Management

### TOML Configuration Files
- **Save Configuration**: Export current settings to TOML file
- **Load Configuration**: Import previously saved settings
- **Template Creation**: Generate template files for future use

### Configuration File Structure
```toml
[interpolation]
method = "linear"
max_gap = 60

[smoothing]
method = "savgol"
window_length = 7
polyorder = 3

[padding]
percent = 10.0

[split]
enabled = false
```

## Best Practices

### 1. Data Preparation
- Ensure first column contains frame numbers or time
- Check for consistent sampling rates
- Identify expected data ranges

### 2. Method Selection
- **Start with Linear Interpolation** for most cases
- **Use Savitzky-Golay** for biomechanical data
- **Apply Butterworth** for standard biomechanics filtering
- **Try LOWESS** for noisy data

### 3. Parameter Tuning
- **Test with Quality Analysis** before batch processing
- **Start with default values** and adjust gradually
- **Use padding** to avoid edge effects
- **Set reasonable gap limits** to avoid over-interpolation

### 4. Quality Control
- **Always use Quality Analysis** for new data types
- **Check residuals** for remaining patterns
- **Verify frequency content** with FFT
- **Compare derivatives** for smoothness

## Troubleshooting

### Common Issues

**1. GUI Not Opening**
- Ensure VAILA environment is activated
- Check for Tkinter conflicts
- Try running directly: `python vaila/interp_smooth_split.py`

**2. Processing Errors**
- Check CSV file format
- Verify numeric columns
- Ensure sufficient data points

**3. Poor Results**
- Adjust smoothing parameters
- Try different interpolation methods
- Use Quality Analysis to diagnose issues

**4. Memory Issues**
- Process smaller batches
- Reduce padding percentage
- Use simpler smoothing methods

### Error Messages

**"Window length must be odd"**
- Use odd numbers: 5, 7, 9, 11, 13, etc.

**"Cutoff frequency must be less than half of sampling frequency"**
- Reduce cutoff frequency or increase sampling frequency

**"Polynomial order must be less than window length"**
- Reduce polynomial order or increase window length

## Advanced Features

### Batch Processing
- Process multiple files simultaneously
- Consistent settings across all files
- Detailed processing reports

### Method Combinations
- Combine interpolation and smoothing
- Apply different methods to different columns
- Customize processing per data type

### Quality Metrics
- RMS errorrr calculation
- Residual analysis
- Frequency domain analysis
- Statistical distribution analysis

## Support and Contact

For questions, issues, or feature requests:
- **Email**: paulosantiago@usp.br
- **GitHub**: https://github.com/vaila-multimodaltoolbox/vaila
- **Documentation**: Check VAILA documentation for updates

## Version Information

- **Version**: 0.0.7
- **Python**: 3.12.9
- **Last Updated**: 16 September 2025
- **Author**: Paulo R. P. Santiago

---

*This tool is part of the VAILA (Video Analysis and Interactive Learning Application) multimodal toolbox for biomechanical data analysis.*
