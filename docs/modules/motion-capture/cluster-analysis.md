# Cluster Analysis

## Overview

The Cluster Analysis module provides comprehensive analysis of motion capture data using anatomical marker clusters. This module processes 3D marker positions to compute trunk and pelvis rotations, generating Euler angles and 3D visualizations for biomechanical analysis.

## Features

- **3D Marker Processing**: Read and process 3D motion capture data from CSV files
- **Butterworth Filtering**: Apply low-pass filtering to reduce noise in marker trajectories
- **Orthonormal Basis Calculation**: Compute orthonormal bases for anatomical marker clusters
- **Euler Angle Computation**: Calculate rotation angles in three anatomical planes
- **Anatomical Reference Integration**: Optional comparison with anatomical reference data
- **3D Visualization**: Generate 3D plots of cluster orientations and Euler angles over time
- **Batch Processing**: Process multiple CSV files simultaneously
- **CSV Export**: Export processed results in CSV format for further analysis

## Supported Anatomical Clusters

### Trunk Cluster
- **Primary Markers**: Typically 3-4 markers placed on the thorax
- **Secondary Markers**: Additional markers for improved orientation accuracy
- **Coordinate System**: Local coordinate system aligned with anatomical axes

### Pelvis Cluster
- **Primary Markers**: Markers placed on the pelvis (ASIS, PSIS, etc.)
- **Secondary Markers**: Additional markers for robust orientation tracking
- **Coordinate System**: Anatomically aligned local coordinate system

## Input Data Format

### CSV File Structure
The module expects CSV files containing 3D marker coordinates with the following structure:

```csv
Frame,X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,...
1,x11,y11,z11,x12,y12,z12,x13,y13,z13,...
2,x21,y21,z21,x22,y22,z22,x23,y23,z23,...
...
```

### Column Naming Convention
- **Trunk Markers**: Headers like `trunk_X1`, `trunk_Y1`, `trunk_Z1`, etc.
- **Pelvis Markers**: Headers like `pelvis_X1`, `pelvis_Y1`, `pelvis_Z1`, etc.
- **Frame Column**: Typically the first column containing frame numbers

## Processing Pipeline

### 1. Data Loading and Validation
- Load CSV files from specified directory
- Validate marker coordinate data
- Check for missing or corrupted frames

### 2. Filtering and Preprocessing
- Apply Butterworth low-pass filter to reduce noise
- Configurable filter parameters (cutoff frequency, order)
- Handle missing data points through interpolation

### 3. Cluster Definition
- Interactive selection of markers for each cluster
- Define primary and secondary markers for each anatomical segment
- Validate cluster configurations

### 4. Orthonormal Basis Calculation
- Compute local coordinate systems for each cluster
- Calculate rotation matrices for each frame
- Ensure orthonormal properties of basis vectors

### 5. Euler Angle Computation
- Convert rotation matrices to Euler angles
- Choose appropriate rotation sequence (e.g., XYZ, ZYX)
- Handle gimbal lock situations

### 6. Anatomical Reference Integration (Optional)
- Load anatomical reference angles from separate CSV file
- Compare computed angles with reference data
- Calculate differences and statistics

### 7. Visualization Generation
- Create 3D plots of cluster orientations
- Generate time-series plots of Euler angles
- Save visualizations as PNG files

## Output Files

For each processed dataset, the module generates:

1. **Euler Angles CSV** (`*_cluster_angles.csv`):
   - Frame-by-frame Euler angles for trunk and pelvis
   - Angles in degrees for X, Y, Z rotations
   - Optional reference angles and differences

2. **Cluster Visualization** (`*_cluster_3d.png`):
   - 3D visualization of cluster orientations
   - Orthogonal basis vectors displayed
   - Multiple frames shown for motion analysis

3. **Angle Time Series** (`*_angles_plot.png`):
   - Time-series plots of Euler angles
   - Separate plots for each rotation axis
   - Trunk and pelvis angles shown together

4. **Processing Log** (`*_log.txt`):
   - Detailed processing parameters
   - Statistical summaries of computed angles
   - Warnings and error messages

## Usage

### GUI Mode (Recommended)

```python
from vaila.cluster_analysis import analyze_cluster_data

# Launch GUI for configuration and processing
analyze_cluster_data()
```

### Programmatic Usage

```python
import pandas as pd
from vaila.cluster_analysis import process_cluster_data

# Load marker data
marker_data = pd.read_csv('motion_capture_data.csv')

# Define cluster configurations
trunk_markers = ['trunk_X1', 'trunk_Y1', 'trunk_Z1', 'trunk_X2', 'trunk_Y2', 'trunk_Z2']
pelvis_markers = ['pelvis_X1', 'pelvis_Y1', 'pelvis_Z1', 'pelvis_X2', 'pelvis_Y2', 'pelvis_Z2']

# Process data
results = process_cluster_data(
    marker_data,
    trunk_markers=trunk_markers,
    pelvis_markers=pelvis_markers,
    filter_cutoff=6.0,  # Hz
    filter_order=4
)

# Save results
results['euler_angles'].to_csv('euler_angles.csv', index=False)
```

## Configuration Parameters

### Filtering Parameters
- **Cutoff Frequency**: Low-pass filter cutoff frequency (default: 6.0 Hz)
- **Filter Order**: Butterworth filter order (default: 4)
- **Sampling Rate**: Data sampling rate in Hz (required for filtering)

### Cluster Parameters
- **Primary Markers**: Minimum markers required for cluster definition
- **Secondary Markers**: Additional markers for improved accuracy
- **Coordinate System**: Definition of local anatomical coordinate system

### Visualization Parameters
- **3D Plot Settings**: Camera angles, lighting, marker sizes
- **Time Series Settings**: Line styles, colors, axis labels
- **Output Resolution**: DPI and figure size settings

## Accuracy and Validation

### Sources of Error
- **Marker Placement**: Incorrect anatomical marker placement
- **Soft Tissue Artifact**: Movement of markers relative to underlying bone
- **Calibration Errors**: Inaccurate camera calibration
- **Synchronization**: Temporal misalignment between cameras

### Validation Methods
- **Repeatability Analysis**: Multiple trials under same conditions
- **Comparison with References**: Validation against known anatomical angles
- **Cross-Validation**: Comparison between different cluster configurations

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Motion Capture**: Use with full-body motion capture analysis
- **Visualization**: Compatible with 3D plotting and C3D viewing tools
- **Data Processing**: Output can be processed with filtering and smoothing tools
- **Statistical Analysis**: Results can be used for statistical comparisons

## Performance Considerations

### Processing Speed
- **Filtering**: ~1-2 seconds per 1000 frames
- **Basis Calculation**: ~0.5-1 second per 1000 frames
- **Visualization**: ~2-5 seconds per figure

### Memory Usage
- **Large Datasets**: Consider processing in chunks for datasets > 100,000 frames
- **Multiple Files**: Batch processing may require significant RAM

## Troubleshooting

### Common Issues

1. **Marker Identification**: Ensure correct marker labeling in CSV files
2. **Missing Data**: Check for gaps in marker trajectories
3. **Filter Artifacts**: Adjust filter parameters if over-filtering occurs
4. **Coordinate System**: Verify anatomical coordinate system definition

### Data Quality Checks

- **Gap Detection**: Identify and handle missing marker data
- **Trajectory Validation**: Check for unrealistic marker movements
- **Statistical Outliers**: Identify and handle measurement errors

## Version History

- **v1.0**: Initial release with core cluster analysis functionality
- **v0.9**: Added anatomical reference integration
- **v0.8**: Improved 3D visualization capabilities
- **v0.7**: Enhanced filtering algorithms
- **v0.6**: Added batch processing support
- **v0.5**: Initial implementation with basic Euler angle computation

## References

- **Biomechanical Conventions**: ISB recommendations for joint coordinate systems
- **Euler Angles**: Proper handling of rotation sequences and gimbal lock
- **Filtering**: Digital signal processing techniques for motion data
