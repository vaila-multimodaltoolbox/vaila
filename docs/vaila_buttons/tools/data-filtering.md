# Data Filtering and Processing Tools

## Overview

The Data Filtering and Processing Tools module provides comprehensive data preprocessing capabilities for biomechanical time-series data, including gap filling, smoothing, filtering, and data splitting operations.

## Features

- **Gap Filling**: Multiple algorithms for filling missing data points
- **Smoothing Algorithms**: Advanced filtering techniques for noise reduction
- **Data Splitting**: Intelligent data segmentation for analysis
- **Batch Processing**: Process multiple CSV files simultaneously
- **TOML Configuration**: Comprehensive configuration management
- **GUI Interface**: Interactive parameter configuration
- **Quality Assurance**: Data validation and processing reports

## Gap Filling Methods

### Linear Interpolation
- **Method**: Linear interpolation between known data points
- **Use Case**: Regular sampling with missing values
- **Parameters**: None required (automatic)
- **Limitations**: May not capture nonlinear trends

### Kalman Filter
- **Method**: State estimation using Kalman filtering
- **Use Case**: Noisy data with underlying smooth trends
- **Parameters**:
  - `n_iter`: Number of filter iterations (default: 5)
  - `mode`: Filter mode (1: standard, 2: adaptive)
- **Advantages**: Handles varying noise levels

### Savitzky-Golay Filter
- **Method**: Polynomial smoothing with derivative preservation
- **Use Case**: Preserving signal shape while reducing noise
- **Parameters**:
  - `window_length`: Filter window size (must be odd)
  - `polyorder`: Polynomial order (2-4 recommended)
- **Advantages**: Maintains signal characteristics

### LOWESS (Locally Weighted Scatterplot Smoothing)
- **Method**: Locally weighted regression smoothing
- **Use Case**: Complex nonlinear trends
- **Parameters**:
  - `frac`: Fraction of data used for each local fit (0.1-0.9)
  - `it`: Number of iterations (default: 3)
- **Advantages**: Captures complex patterns

### Spline Interpolation
- **Method**: Cubic spline interpolation
- **Use Case**: Smooth interpolation with natural boundaries
- **Parameters**:
  - `s`: Smoothing parameter (0: exact interpolation, >0: smoothing)
- **Advantages**: Smooth and continuous results

### ARIMA Modeling
- **Method**: Autoregressive Integrated Moving Average modeling
- **Use Case**: Time series with trends and seasonality
- **Parameters**:
  - `order`: (p, d, q) model parameters
- **Advantages**: Handles complex temporal patterns

## Smoothing and Filtering

### Butterworth Filter
- **Method**: Infinite impulse response (IIR) low-pass filter
- **Use Case**: Remove high-frequency noise
- **Parameters**:
  - `cutoff`: Cutoff frequency in Hz
  - `order`: Filter order (2-8 recommended)
  - `filter_type`: 'low' or 'band'
- **Advantages**: Sharp frequency cutoff

### Edge Effect Mitigation
- **Padding**: Add data at boundaries to reduce edge artifacts
- **Reflection**: Use reflected data for boundary conditions
- **Windowing**: Apply window functions to reduce artifacts

## Data Splitting

### Time-based Splitting
- **Method**: Split data at specified time points
- **Use Case**: Separate different trial phases or conditions
- **Parameters**: Split times in seconds or frames

### Half Splitting
- **Method**: Split data into first and second halves
- **Use Case**: Compare early vs. late performance
- **Parameters**: None (automatic midpoint)

### Custom Segmentation
- **Method**: User-defined split points and segments
- **Use Case**: Complex analysis requiring specific segments
- **Parameters**: Custom split definitions

## Configuration Management

### TOML Configuration Files
```toml
# Interpolation and Smoothing Configuration
[gap_filling]
method = "kalman"                    # linear, kalman, savgol, lowess, spline, arima
kalman_iterations = 5
kalman_mode = 1

[smoothing]
butterworth_cutoff = 6.0             # Hz
butterworth_order = 4
savgol_window = 11
savgol_polyorder = 3

[processing]
padding_method = "reflection"        # none, reflection, constant
split_method = "half"                # half, time, custom
output_format = "csv"                # csv, excel
```

### GUI Configuration
- **Interactive Parameter Setting**: Visual parameter adjustment
- **Preset Management**: Save and load configuration presets
- **Real-time Preview**: Preview processing results
- **Batch Configuration**: Apply settings to multiple files

## Usage

### GUI Mode (Recommended)

```python
from vaila.interp_smooth_split import run_fill_split_dialog

# Launch interactive interface
run_fill_split_dialog()
```

### Programmatic Usage

```python
import pandas as pd
from vaila.interp_smooth_split import process_file

# Load data
data = pd.read_csv('biomechanical_data.csv')

# Configure processing
config = {
    'gap_filling': {
        'method': 'kalman',
        'kalman_iterations': 5
    },
    'smoothing': {
        'butterworth_cutoff': 6.0,
        'butterworth_order': 4
    }
}

# Process file
process_file(
    file_path='input_data.csv',
    dest_dir='/path/to/output',
    config=config
)
```

### Batch Processing

```python
import glob
from vaila.interp_smooth_split import process_file

# Process multiple files
data_files = glob.glob('/path/to/data/*.csv')

for file_path in data_files:
    process_file(
        file_path=file_path,
        dest_dir='/path/to/processed',
        config=processing_config
    )
```

## Advanced Features

### Custom Filter Chains

```python
# Create custom processing pipeline
def custom_filter_chain(data, config):
    # Step 1: Gap filling
    if config['gap_filling']['method'] == 'kalman':
        data = kalman_smooth(data, **config['gap_filling']['kalman_params'])

    # Step 2: Butterworth filtering
    data = butter_filter(data, **config['smoothing']['butterworth_params'])

    # Step 3: Savitzky-Golay smoothing
    data = savgol_filter(data, **config['smoothing']['savgol_params'])

    return data
```

### Quality Metrics

```python
# Calculate processing quality metrics
def calculate_quality_metrics(original, processed):
    metrics = {
        'rmse': np.sqrt(np.mean((original - processed)**2)),
        'mae': np.mean(np.abs(original - processed)),
        'correlation': np.corrcoef(original, processed)[0, 1]
    }
    return metrics
```

## Performance Considerations

### Processing Speed
- **Linear Interpolation**: Fastest method (~1ms per 1000 points)
- **Kalman Filter**: Moderate speed (~10ms per 1000 points)
- **Savitzky-Golay**: Moderate speed (~5ms per 1000 points)
- **Butterworth Filter**: Slowest for high orders (~50ms per 1000 points)

### Memory Usage
- **In-place Processing**: Minimize memory usage by modifying data in place
- **Chunked Processing**: Process large files in chunks
- **Temporary Files**: Use temporary storage for very large datasets

### Parallel Processing
- **Multi-threading**: Process multiple files simultaneously
- **Vectorization**: Use NumPy vectorized operations
- **GPU Acceleration**: CUDA support for intensive computations

## Quality Assurance

### Data Validation
- **Range Checks**: Verify processed data stays within reasonable bounds
- **Gap Detection**: Identify and report unfilled gaps
- **Statistical Validation**: Compare statistical properties before/after processing

### Processing Reports
- **Detailed Logs**: Complete processing history and parameters
- **Quality Metrics**: Statistical comparison of original vs. processed data
- **Error Reporting**: Comprehensive errorrrr messages and warnings

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Data Import**: Use with C3D/CSV conversion tools
- **Visualization**: Process data for plotting and analysis
- **Motion Capture**: Prepare motion capture data for analysis
- **Statistical Analysis**: Provide clean data for statistical processing

## Troubleshooting

### Common Issues

1. **Over-smoothing**: Reduce filter parameters or use lighter filtering
2. **Edge Artifacts**: Increase padding or use reflection padding
3. **Memory Errors**: Process large files in smaller chunks
4. **Quality Degradation**: Check parameter settings and validation metrics

### Parameter Optimization

- **Start Conservative**: Begin with lighter filtering and increase as needed
- **Validate Results**: Always compare processed vs. original data
- **Use Multiple Methods**: Compare results from different algorithms
- **Domain Knowledge**: Consider biomechanical signal characteristics

## Version History

- **v0.0.7**: Added TOML configuration and advanced smoothing algorithms
- **v0.0.6**: Added GUI interface and batch processing
- **v0.0.5**: Added Kalman and Savitzky-Golay filtering
- **v0.0.4**: Added linear interpolation and basic smoothing
- **v0.0.3**: Initial implementation with gap filling
- **v0.0.2**: Added data splitting functionality
- **v0.0.1**: Basic interpolation capabilities

## Requirements

### Core Dependencies
- **Python 3.8+**: Modern Python features
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing
- **scikit-learn**: Machine learning utilities
- **statsmodels**: Statistical modeling
- **pykalman**: Kalman filtering

### Optional Dependencies
- **TOML**: Configuration file support
- **Matplotlib**: Visualization for GUI
- **Tkinter**: GUI framework (usually included)

## References

- **Signal Processing**: Digital signal processing fundamentals
- **Biomechanical Data**: Characteristics of human movement data
- **Filtering Theory**: Filter design and implementation
- **Time Series Analysis**: Methods for temporal data processing
