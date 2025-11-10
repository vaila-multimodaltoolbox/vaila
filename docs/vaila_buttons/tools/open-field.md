# Open Field Analysis Tools

## Overview

The Open Field Analysis Tools module provides comprehensive behavioral analysis for animal studies in open field environments. This module processes movement trajectories to quantify locomotion patterns, spatial preferences, and temporal behaviors in controlled experimental settings.

## Features

- **Trajectory Analysis**: Complete kinematic analysis of movement paths
- **Spatial Zoning**: Automatic detection of spatial preferences and zone occupancy
- **Speed Analysis**: Detailed speed profiling with multiple speed ranges
- **Temporal Analysis**: Time-based behavioral quantification
- **Visualization Suite**: Multiple visualization types for comprehensive analysis
- **Batch Processing**: Process multiple experimental sessions simultaneously
- **Statistical Reporting**: Detailed statistical summaries and metrics

## Behavioral Metrics

### Locomotion Parameters
- **Total Distance Traveled**: Cumulative path length in meters
- **Average Speed**: Mean locomotion speed (m/s)
- **Maximum Speed**: Peak locomotion speed
- **Time Stationary**: Duration with minimal movement (< 0.05 m/s)
- **Movement Episodes**: Number and duration of movement bouts

### Speed Range Analysis
- **Speed Distribution**: Time spent in different speed ranges
- **Speed Categories**:
  - Stationary (0 m/s)
  - Slow movement (0-15 m/min)
  - Moderate movement (15-30 m/min)
  - Fast movement (30-45 m/min)
  - High-speed locomotion (>45 m/min)

### Spatial Analysis
- **Zone Occupancy**: Time spent in each spatial zone
- **Center vs. Border Preference**: Thigmotaxis quantification
- **Spatial Distribution**: Heat map analysis of positional preferences
- **Zone Transitions**: Frequency of movement between zones

## Spatial Configuration

### Open Field Layout
- **Standard Size**: 60cm × 60cm experimental arena
- **Grid Division**: 3×3 grid creating 20cm × 20cm zones
- **Center Zone**: Central 20cm × 20cm area
- **Border Zones**: Peripheral areas around the center

### Zone Definitions
```
+---+---+---+
| 1 | 2 | 3 |
+---+---+---+
| 4 | 5 | 6 |  ← Center Zone (5)
+---+---+---+
| 7 | 8 | 9 |
+---+---+---+
```

## Data Requirements

### Input Data Format
- **CSV Files**: Comma-separated values with movement data
- **Required Columns**:
  - `time(s)`: Time in seconds
  - `position_x(m)`: X-coordinate in meters
  - `position_y(m)`: Y-coordinate in meters

### Data Validation
- **Boundary Checking**: Automatic detection of out-of-bounds positions
- **Coordinate Correction**: Clipping positions to valid arena boundaries
- **Sampling Rate**: Configurable sampling frequency (Hz)
- **Data Smoothing**: Optional filtering for noise reduction

## Analysis Pipeline

### 1. Data Preprocessing
- **Loading**: Import CSV files with automatic format detection
- **Validation**: Check data integrity and coordinate ranges
- **Filtering**: Optional smoothing of position data
- **Boundary Correction**: Ensure all positions are within arena bounds

### 2. Kinematic Calculations
- **Distance Calculation**: Frame-to-frame Euclidean distances
- **Speed Computation**: Instantaneous and average speeds
- **Acceleration Analysis**: Rate of speed changes
- **Path Efficiency**: Directness of movement paths

### 3. Spatial Analysis
- **Zone Assignment**: Assign positions to spatial zones
- **Occupancy Calculation**: Time spent in each zone
- **Center-Border Analysis**: Thigmotaxis quantification
- **Heat Map Generation**: Spatial density analysis

### 4. Temporal Analysis
- **Speed Profiling**: Speed distribution over time
- **Movement Episodes**: Identification of movement bouts
- **Stationary Periods**: Detection of resting periods
- **Behavioral States**: Classification of behavioral states

### 5. Statistical Analysis
- **Descriptive Statistics**: Mean, median, standard deviation for all metrics
- **Distribution Analysis**: Frequency distributions of behavioral measures
- **Comparative Analysis**: Statistical comparison between conditions
- **Trend Analysis**: Temporal trends in behavioral measures

## Visualization Suite

### Pathway Plots
- **Trajectory Visualization**: Complete movement path with time coloring
- **Color Gradient**: Time-based color coding (early=blue, late=red)
- **Zone Overlays**: Spatial zone boundaries and labels
- **Movement Indicators**: Visual representation of movement patterns

### Heat Maps
- **Positional Density**: Heat map of time spent at each location
- **Zone-Specific Maps**: Separate heat maps for center and border areas
- **Gaussian Smoothing**: Smooth interpolation for density visualization
- **Color Scaling**: Configurable color schemes and intensity scaling

### Speed Analysis Plots
- **Speed Over Time**: Temporal speed profile with smoothing
- **Speed Range Distribution**: Bar chart of time in speed categories
- **Speed Categories**: Visual representation of behavioral speed states
- **Smoothed Curves**: Moving average smoothing for trend visualization

### Statistical Plots
- **Zone Occupancy Charts**: Bar charts of time spent in each zone
- **Behavioral Metrics**: Summary statistics visualization
- **Comparative Plots**: Multi-condition comparison charts
- **Distribution Plots**: Histograms and density plots

## Usage

### GUI Mode (Recommended)

```python
from vaila.animal_open_field import run_animal_open_field

# Launch interactive interface
run_animal_open_field()
```

### Programmatic Usage

```python
import pandas as pd
from vaila.animal_open_field import calculate_kinematics, calculate_zone_occupancy

# Load movement data
data = pd.read_csv('animal_trajectory.csv')

# Extract position data
time = data['time(s)'].values
x = data['position_x(m)'].values
y = data['position_y(m)'].values

# Calculate kinematics
kinematics = calculate_kinematics(x, y, fs=30.0)  # 30 Hz sampling rate

# Calculate zone occupancy
zone_results = calculate_zone_occupancy(x, y, distance_threshold=0.02)  # 2cm zones

# Generate visualizations
plot_pathway(x, y, time, kinematics['total_distance'], output_dir, 'experiment_1')
plot_heatmap(x, y, output_dir, 'experiment_1', zone_results)
```

### Batch Processing

```python
import glob
from vaila.animal_open_field import process_experiment

# Process multiple experiments
experiment_files = glob.glob('/path/to/experiments/*.csv')

for exp_file in experiment_files:
    process_experiment(
        input_file=exp_file,
        output_dir='/path/to/results',
        sampling_rate=30.0,
        smoothing_window=2.0  # 2 second smoothing
    )
```

## Advanced Configuration

### Custom Arena Dimensions

```python
# Define custom arena size and zones
arena_config = {
    'width': 0.8,      # 80cm width
    'height': 0.8,     # 80cm height
    'grid_size': 4,    # 4x4 grid instead of 3x3
    'center_radius': 0.2  # 20cm center zone radius
}

process_experiment(
    input_file='custom_arena.csv',
    arena_config=arena_config
)
```

### Custom Speed Categories

```python
# Define custom speed ranges for analysis
speed_config = {
    'stationary_threshold': 0.02,  # m/s
    'slow_range': (0.02, 0.1),
    'moderate_range': (0.1, 0.2),
    'fast_range': (0.2, 0.3),
    'very_fast_range': (0.3, float('inf'))
}

calculate_kinematics(x, y, fs=30.0, speed_config=speed_config)
```

## Applications

### Behavioral Neuroscience
- **Anxiety Assessment**: Measure thigmotaxis and center avoidance
- **Locomotor Activity**: Quantify overall movement and exploration
- **Habituation Studies**: Track changes in behavior over time
- **Drug Effects**: Evaluate pharmacological effects on locomotion

### Pharmacology Research
- **Dose-Response Studies**: Compare behavioral effects across doses
- **Time Course Analysis**: Track behavioral changes over drug action time
- **Withdrawal Studies**: Assess behavioral changes during withdrawal
- **Comparative Studies**: Compare effects between drug classes

### Toxicology Studies
- **Toxicity Assessment**: Identify behavioral signs of toxicity
- **Recovery Monitoring**: Track behavioral recovery post-exposure
- **Dose Optimization**: Determine optimal dosing for therapeutic effects
- **Safety Pharmacology**: Comprehensive behavioral safety assessment

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Data Processing**: Use with filtering and smoothing tools
- **Visualization**: Compatible with 2D/3D plotting modules
- **Statistical Analysis**: Combine with statistical processing tools
- **Video Analysis**: Correlate with video-based behavioral analysis

## Quality Assurance

### Data Quality Checks
- **Coordinate Validation**: Verify position data integrity
- **Sampling Consistency**: Check for consistent sampling rates
- **Boundary Compliance**: Ensure all positions are within arena bounds
- **Gap Detection**: Identify and handle missing data points

### Analysis Validation
- **Metric Consistency**: Verify kinematic calculations
- **Zone Assignment**: Validate spatial zone classifications
- **Statistical Accuracy**: Confirm statistical calculations
- **Visualization Accuracy**: Verify plot data representation

## Troubleshooting

### Common Issues

1. **Boundary Violations**: Check coordinate scaling and arena dimensions
2. **Speed Artifacts**: Adjust smoothing parameters for noisy data
3. **Zone Assignment Errors**: Verify coordinate system and zone definitions
4. **Memory Issues**: Process large datasets in smaller batches

### Parameter Optimization

- **Smoothing Window**: Start with 1-2 seconds and adjust based on data
- **Sampling Rate**: Ensure sampling rate matches data acquisition
- **Arena Dimensions**: Verify arena size matches experimental setup
- **Speed Thresholds**: Adjust based on species and experimental conditions

## Version History

- **v2.1.0**: Added speed range analysis and enhanced visualization
- **v2.0.0**: Major rewrite with comprehensive spatial analysis
- **v1.5.0**: Added zone-based analysis and heat maps
- **v1.0.0**: Initial implementation with basic kinematic analysis

## Requirements

### Core Dependencies
- **Python 3.8+**: Modern Python features
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **Seaborn**: Statistical plotting
- **SciPy**: Scientific computing

### Optional Dependencies
- **Tkinter**: GUI interface (usually included)
- **Pillow**: Image processing for enhanced visualizations

## References

- **Open Field Test**: Standard behavioral testing methodology
- **Animal Behavior**: Rodent behavior in open field environments
- **Kinematic Analysis**: Movement analysis techniques
- **Spatial Analysis**: Methods for spatial preference quantification
