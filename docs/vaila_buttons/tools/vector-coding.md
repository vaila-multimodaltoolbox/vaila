# Vector Coding Analysis Tools

## Overview

The Vector Coding Analysis Tools module provides comprehensive analysis of joint coupling and coordination patterns using vector coding techniques. This method quantifies the relationship between two joint angles over time, identifying coordination patterns and phase relationships in human movement.

## Features

- **Coupling Angle Calculation**: Quantify coordination between joint pairs
- **Phase Analysis**: Identify in-phase and anti-phase coordination patterns
- **Time Normalization**: Standardize movement cycles for comparison
- **Visualization Suite**: Multiple visualization types for coordination analysis
- **Statistical Analysis**: Quantification of coordination variability
- **Batch Processing**: Analyze multiple movement trials simultaneously
- **Export Capabilities**: Save results in multiple formats

## Vector Coding Methodology

### Core Concept
Vector coding analyzes the relationship between two angular displacements by:
1. **Angle Vector Creation**: Represent each joint angle as a vector in polar coordinates
2. **Coupling Angle Calculation**: Determine the angle between joint angle vectors
3. **Phase Classification**: Categorize coordination patterns into discrete phases

### Mathematical Foundation

For two joint angles θ₁ and θ₂ at each time point:

1. **Vector Representation**:
   ```
   V₁ = (cos(θ₁), sin(θ₁))
   V₂ = (cos(θ₂), sin(θ₂))
   ```

2. **Coupling Angle**:
   ```
   γ = atan2(V₁×V₂, V₁·V₂)
   ```

3. **Phase Classification**:
   - **In-Phase**: γ ≈ 0° (coordinated movement)
   - **Anti-Phase**: γ ≈ 180° (opposing movement)
   - **Transitions**: Intermediate phases

## Supported Data Formats

### Input Data Requirements
- **CSV Files**: Time-series joint angle data
- **Column Structure**: Frame/Time, Joint1_Angle, Joint2_Angle
- **Angular Units**: Degrees or radians (configurable)
- **Sampling Rate**: Consistent sampling frequency

### Example Data Format
```csv
frame,joint1_angle,joint2_angle
1,45.2,23.8
2,46.1,24.2
3,47.3,24.8
4,48.2,25.1
...
```

## Analysis Pipeline

### 1. Data Preprocessing
- **Loading**: Import CSV files with automatic format detection
- **Validation**: Check data integrity and sampling consistency
- **Filtering**: Optional smoothing for noise reduction
- **Normalization**: Standardize data ranges if needed

### 2. Time Normalization
- **Cycle Detection**: Identify movement cycles
- **Resampling**: Normalize to fixed number of points (default: 101)
- **Interpolation**: Smooth interpolation between data points
- **Boundary Handling**: Proper handling of cycle boundaries

### 3. Vector Calculation
- **Angle Vectorization**: Convert angles to unit vectors
- **Coupling Computation**: Calculate angle between vector pairs
- **Phase Determination**: Classify coordination patterns
- **Variability Assessment**: Quantify coordination consistency

### 4. Statistical Analysis
- **Phase Percentages**: Time spent in each coordination pattern
- **Variability Metrics**: Standard deviation of coupling angles
- **Trend Analysis**: Changes in coordination over time
- **Comparative Statistics**: Between-condition comparisons

## Visualization Types

### Coupling Angle Plots
- **Circular Plot**: Polar plot of coupling angle distribution
- **Time Series**: Coupling angle over normalized time
- **Phase Regions**: Visual separation of coordination phases
- **Variability Bands**: Confidence intervals around mean patterns

### Angle-Angle Diagrams
- **Phase Portraits**: Joint angle relationships in phase space
- **Coordination Patterns**: Visual representation of coupling
- **Movement Trajectories**: Path of joint coordination over time
- **Comparative Overlays**: Multiple conditions on same plot

### Statistical Visualizations
- **Phase Distribution**: Pie/bar charts of phase percentages
- **Variability Plots**: Box plots and violin plots of coordination
- **Trend Analysis**: Line plots of coordination changes
- **Heat Maps**: Temporal coordination patterns

## Usage

### GUI Mode (Recommended)

```python
from vaila.run_vector_coding import run_vector_coding

# Launch interactive interface
run_vector_coding()
```

### Programmatic Usage

```python
import pandas as pd
from vaila.run_vector_coding import get_coupling_angle

# Load joint angle data
data = pd.read_csv('joint_angles.csv')

# Extract angle columns
joint1_angles = data.iloc[:, 1].values  # First joint angles
joint2_angles = data.iloc[:, 2].values  # Second joint angles

# Calculate coupling angles
coupling_results = get_coupling_angle(
    file='joint_angles.csv',
    freq=100.0,  # Sampling frequency in Hz
    joint1_name='Knee',
    joint2_name='Ankle',
    save=True,
    savedir='/path/to/results'
)
```

### Batch Processing

```python
import glob
from vaila.run_vector_coding import get_coupling_angle

# Process multiple files
data_files = glob.glob('/path/to/angle_data/*.csv')

for file_path in data_files:
    get_coupling_angle(
        file=file_path,
        freq=100.0,
        joint1_name='Proximal_Joint',
        joint2_name='Distal_Joint',
        save=True,
        savedir='/path/to/vector_coding_results'
    )
```

## Advanced Configuration

### Custom Normalization

```python
# Custom time normalization parameters
normalization_config = {
    'n_points': 101,          # Number of normalized time points
    'interpolation_kind': 'cubic',  # Interpolation method
    'boundary_handling': 'periodic'  # How to handle cycle boundaries
}

get_coupling_angle(
    file='joint_data.csv',
    normalization_config=normalization_config
)
```

### Advanced Statistical Analysis

```python
# Configure statistical analysis
stats_config = {
    'phase_bins': 8,           # Number of phase bins for analysis
    'variability_metric': 'std',  # Variability measurement method
    'confidence_interval': 95,   # Confidence interval percentage
    'outlier_detection': True    # Remove statistical outliers
}

coupling_results = get_coupling_angle(
    file='joint_data.csv',
    stats_config=stats_config
)
```

## Applications

### Biomechanical Research
- **Gait Analysis**: Examine coordination between lower limb joints
- **Sports Performance**: Analyze technique and coordination patterns
- **Rehabilitation**: Monitor recovery of normal movement patterns
- **Comparative Studies**: Compare coordination between groups

### Clinical Assessment
- **Neurological Disorders**: Assess coordination deficits in neurological conditions
- **Orthopedic Assessment**: Evaluate joint coordination post-injury/surgery
- **Developmental Studies**: Track coordination development in children
- **Geriatric Assessment**: Monitor age-related coordination changes

### Sports Science
- **Technique Analysis**: Evaluate coordination in athletic movements
- **Training Effects**: Monitor changes in coordination with training
- **Equipment Effects**: Assess how equipment affects movement coordination
- **Injury Prevention**: Identify coordination patterns associated with injury risk

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Motion Capture**: Use with cluster and full-body motion capture analysis
- **Visualization**: Compatible with 2D/3D plotting modules for enhanced visualization
- **Data Processing**: Combine with filtering and smoothing tools
- **Statistical Analysis**: Export data for advanced statistical processing

## Quality Assurance

### Data Quality Checks
- **Signal Integrity**: Verify angle data continuity and range
- **Sampling Consistency**: Ensure consistent sampling rates
- **Outlier Detection**: Identify and handle anomalous data points
- **Cycle Validation**: Verify movement cycle identification

### Analysis Validation
- **Method Verification**: Compare with established vector coding literature
- **Repeatability**: Assess consistency across multiple analyses
- **Sensitivity Analysis**: Test robustness to parameter changes
- **Cross-Validation**: Compare results with alternative analysis methods

## Troubleshooting

### Common Issues

1. **Phase Classification Errors**: Check angle data quality and range
2. **Normalization Problems**: Verify cycle detection and boundary handling
3. **Statistical Artifacts**: Check for sufficient data points per cycle
4. **Visualization Issues**: Ensure proper data formatting for plotting

### Parameter Optimization

- **Normalization Points**: Use 101 points for standard gait cycle analysis
- **Phase Bin Number**: 8-12 bins typically provide good resolution
- **Smoothing**: Apply light smoothing if data is very noisy
- **Threshold Values**: Adjust based on specific movement characteristics

## Version History

- **v2.0**: Added advanced statistical analysis and enhanced visualization
- **v1.5**: Added GUI interface and batch processing capabilities
- **v1.0**: Initial implementation with basic vector coding analysis

## Requirements

### Core Dependencies
- **Python 3.8+**: Modern Python features
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **SciPy**: Scientific computing
- **Matplotlib**: Visualization

### Optional Dependencies
- **Tkinter**: GUI interface (usually included)
- **Seaborn**: Enhanced statistical plotting

## References

- **Vector Coding**: Original methodology for joint coordination analysis
- **Biomechanical Coordination**: Principles of inter-joint coordination
- **Gait Analysis**: Application to human locomotion studies
- **Motor Control**: Neural control of coordinated movements
