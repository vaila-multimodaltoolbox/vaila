# 2D Plotting Tools

## Overview

The 2D Plotting Tools module provides comprehensive data visualization capabilities for biomechanical data analysis, featuring multiple plot types, interactive GUI controls, and advanced statistical plotting options.

## Features

- **Multiple Plot Types**: Time series, angle-angle diagrams, confidence intervals, box plots, and SPM analysis
- **Interactive GUI**: User-friendly interface for file selection and plot configuration
- **Memory Management**: Efficient handling of multiple plots and datasets
- **Statistical Analysis**: Integration with SPM (Statistical Parametric Mapping) for advanced analysis
- **Export Capabilities**: Save plots in multiple formats
- **Batch Processing**: Handle multiple CSV files simultaneously
- **Custom Color Schemes**: Extensive color palette options

## Supported Plot Types

### 1. Time Scatter Plot
Display time-series data across multiple variables:

- **Multiple Headers**: Plot several data columns simultaneously
- **Time Synchronization**: Align data by time or frame number
- **Color Coding**: Automatic color assignment for different variables
- **Legend Management**: Clear labeling of plotted variables

### 2. Angle-Angle Plot
Visualize relationships between two angular measurements:

- **Coupling Analysis**: Examine coordination between joint angles
- **Phase Relationships**: Analyze timing relationships between movements
- **Correlation Display**: Show linear and nonlinear relationships
- **Custom Axes**: Configurable axis ranges and labels

### 3. Confidence Interval Plot
Display data with statistical confidence bands:

- **Statistical Significance**: Visualize confidence intervals and error bands
- **Multiple Comparisons**: Compare multiple datasets with error bars
- **Custom Confidence Levels**: Adjustable confidence intervals (95%, 99%, etc.)
- **Error Bar Styling**: Configurable error bar appearance

### 4. Box Plot
Generate statistical summary plots:

- **Distribution Analysis**: Display quartiles, medians, and outliers
- **Multiple Groups**: Compare distributions across conditions
- **Outlier Detection**: Automatic identification of statistical outliers
- **Custom Styling**: Configurable box plot appearance

### 5. SPM (Statistical Parametric Mapping)
Advanced statistical analysis and visualization:

- **1D Statistical Tests**: Perform statistical tests along time series
- **Cluster Analysis**: Identify significant temporal clusters
- **Threshold Visualization**: Display statistical thresholds and p-values
- **Publication Ready**: Generate figures suitable for scientific publication

## Data Input

### Supported File Formats
- **CSV Files**: Comma-separated values with headers
- **Excel Files**: Microsoft Excel format (.xlsx, .xls)
- **Text Files**: Tab-separated or custom delimited files

### Data Structure Requirements
- **Headers**: Column headers for variable identification
- **Numeric Data**: Numerical values for plotting
- **Time/Frame Column**: Optional time or frame reference column
- **Missing Data**: Automatic handling of missing values

## GUI Interface

### File Selection
- **Directory Selection**: Choose folders containing data files
- **File Browser**: Interactive file selection dialog
- **Multiple Files**: Select and load multiple CSV files
- **Header Preview**: Preview available column headers

### Plot Configuration
- **Plot Type Selection**: Choose from available plot types
- **Header Selection**: Select specific columns for plotting
- **Parameter Settings**: Configure plot-specific parameters
- **Preview Options**: Preview plots before final generation

### Memory Management
- **Clear Plots**: Remove all plots from memory
- **Clear Data**: Clear loaded datasets from memory
- **New Figure**: Create new figure windows
- **Garbage Collection**: Automatic memory cleanup

## Plot Customization

### Visual Styling
- **Color Palettes**: Extensive color scheme options
- **Line Styles**: Solid, dashed, dotted, and custom line styles
- **Marker Options**: Various marker shapes and sizes
- **Font Settings**: Configurable text size and font family

### Axis Configuration
- **Axis Labels**: Custom axis titles and labels
- **Axis Limits**: Manual or automatic axis scaling
- **Grid Options**: Toggle grid lines and styling
- **Legend Placement**: Configurable legend position and styling

### Figure Layout
- **Subplots**: Multiple plots in single figure
- **Figure Size**: Configurable figure dimensions
- **DPI Settings**: Resolution control for publication-quality output
- **Aspect Ratio**: Control plot aspect ratios

## Statistical Features

### SPM Integration
- **1D Statistical Tests**: T-tests, ANOVA along time series
- **Multiple Comparison Correction**: FDR, Bonferroni corrections
- **Cluster Analysis**: Identify significant temporal clusters
- **Threshold Mapping**: Visualize statistical significance

### Confidence Intervals
- **Bootstrap Methods**: Bootstrap confidence interval calculation
- **Parametric Methods**: Standard error and t-distribution based intervals
- **Custom Alpha Levels**: Adjustable significance levels

## Export and Saving

### Image Formats
- **PNG**: Portable Network Graphics (raster)
- **SVG**: Scalable Vector Graphics (vector)
- **PDF**: Portable Document Format (vector)
- **EPS**: Encapsulated PostScript (vector)

### Data Export
- **Processed Data**: Export processed datasets
- **Statistical Results**: Save statistical analysis results
- **Configuration Files**: Save plot settings for reproducibility

## Performance Features

### Memory Efficiency
- **Lazy Loading**: Load data only when needed
- **Caching**: Cache loaded datasets for repeated use
- **Figure Management**: Track and manage multiple figures
- **Garbage Collection**: Automatic cleanup of unused objects

### Processing Speed
- **Batch Operations**: Efficient processing of multiple files
- **Parallel Processing**: Utilize multiple CPU cores when available
- **Progress Tracking**: Real-time progress indicators

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Data Processing**: Use with data filtering and processing tools
- **Motion Capture**: Visualize motion capture data and angles
- **Statistical Analysis**: Combine with statistical processing modules
- **Report Generation**: Create publication-ready figures

## Usage Examples

### Basic Plotting

```python
from vaila.vailaplot2d import run_plot_2d

# Launch GUI for interactive plotting
run_plot_2d()
```

### Programmatic Plotting

```python
import pandas as pd
import matplotlib.pyplot as plt
from vaila.vailaplot2d import plot_time_scatter

# Load data
data = pd.read_csv('biomechanical_data.csv')

# Create time scatter plot
plot_time_scatter(
    data=data,
    headers=['knee_angle', 'hip_angle', 'ankle_angle'],
    title='Joint Angles Over Time'
)

plt.show()
```

### Statistical Analysis

```python
from vaila.vailaplot2d import plot_spm

# Perform SPM analysis
plot_spm(
    data=data,
    headers=['condition1', 'condition2'],
    alpha=0.05,
    test_type='ttest'
)
```

## Advanced Configuration

### Custom Color Schemes

```python
# Define custom color palette
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Apply to plots
plot_time_scatter(
    data=data,
    headers=headers,
    colors=custom_colors
)
```

### Statistical Parameters

```python
# Configure SPM analysis
plot_spm(
    data=data,
    headers=headers,
    alpha=0.01,  # Significance level
    iterations=1000,  # Bootstrap iterations
    cluster_threshold=0.05  # Cluster correction threshold
)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Clear plots and data regularly when working with large datasets
2. **File Loading Errors**: Check file format and column headers
3. **Plot Display Issues**: Ensure matplotlib backend is properly configured
4. **Statistical Errors**: Verify data format and parameter settings

### Performance Tips

- **Batch Size**: Process files in smaller batches for large datasets
- **Memory Management**: Use clear functions regularly to free memory
- **File Organization**: Organize data files for efficient loading
- **Plot Optimization**: Reduce plot complexity for better performance

## Version History

- **v0.0.2**: Added SPM integration and improved memory management
- **v0.0.1**: Initial implementation with basic plotting capabilities

## Requirements

### Core Dependencies
- **Matplotlib**: For plotting functionality
- **Pandas**: For data handling
- **NumPy**: For numerical computations
- **Tkinter**: For GUI components

### Statistical Dependencies
- **SPM1D**: For statistical parametric mapping
- **SciPy**: For additional statistical functions

### Optional Dependencies
- **Seaborn**: For enhanced statistical plotting (optional)
- **Plotly**: For interactive web-based plots (optional)

## References

- **Matplotlib Documentation**: Official matplotlib plotting guide
- **SPM1D**: Statistical Parametric Mapping for 1D data
- **Data Visualization**: Best practices for scientific data visualization
