# Vertical Jump Analysis Tools

## Overview

The Vertical Jump Analysis Tools module provides comprehensive biomechanical analysis of vertical jump performance, supporting multiple data input formats including time-of-flight measurements, jump height data, and MediaPipe pose estimation coordinates.

## Features

- **Multiple Input Formats**: Support for time-of-flight, jump height, and MediaPipe data
- **Biomechanical Calculations**: Complete kinematic and kinetic analysis
- **MediaPipe Integration**: Automatic coordinate transformation for pose estimation data
- **Performance Metrics**: Comprehensive jump performance quantification
- **Visualization Suite**: Multiple visualization types for jump analysis
- **Batch Processing**: Process multiple jump trials simultaneously
- **HTML Reporting**: Interactive web-based reports with visualizations

## Supported Data Formats

### Time-of-Flight Format
```csv
mass_kg,time_of_flight_s,contact_time_s
75.0,0.45,0.22
80.2,0.42,0.25
65.5,0.48,0.20
```

### Jump Height Format
```csv
mass_kg,height_m,contact_time_s
75.0,0.25,0.22
80.2,0.22,0.25
65.5,0.28,0.20
```

### MediaPipe Pose Format
- **CSV Files**: MediaPipe landmark coordinates (frame_index, landmark_x, landmark_y, landmark_z)
- **Coordinate System**: Automatic inversion of y-coordinates for biomechanical analysis
- **Reference Scaling**: Uses shank length for coordinate scaling to meters

## Biomechanical Calculations

### Core Kinematic Variables
- **Jump Height**: Vertical displacement during flight phase
- **Takeoff Velocity**: Vertical velocity at liftoff
- **Flight Time**: Duration of flight phase
- **Contact Time**: Ground contact duration

### Kinetic Variables
- **Force**: Ground reaction force during contact
- **Liftoff Force**: Force at moment of takeoff
- **Power**: Mechanical power output
- **Energy**: Kinetic and potential energy calculations

### Performance Indices
- **Jump Performance Index (JPI)**: Comprehensive performance metric
- **Relative Power**: Power normalized by body mass (W/kg)
- **Efficiency Metrics**: Energy utilization assessment

## MediaPipe Integration

### Coordinate Transformation
- **Y-Coordinate Inversion**: Transform screen coordinates (y downward) to biomechanical coordinates (y upward)
- **Scaling**: Convert normalized coordinates to meters using anatomical references
- **Center of Gravity**: Calculate CG position from pose landmarks

### Pose Landmark Usage
- **Lower Body Landmarks**: Hip, knee, ankle positions for jump analysis
- **Reference Length**: Shank length for coordinate scaling
- **Joint Angles**: Optional joint angle calculations

## Calculation Methods

### Time-of-Flight Method
1. **Flight Time Measurement**: Duration of flight phase
2. **Height Calculation**: `h = (1/2) × g × t²` (where g = 9.81 m/s²)
3. **Velocity Calculation**: `v = g × t/2` (takeoff velocity)

### Jump Height Method
1. **Direct Height Measurement**: Use measured jump height
2. **Velocity Calculation**: `v = √(2 × g × h)` (takeoff velocity)
3. **Flight Time**: `t = (2 × v)/g`

### MediaPipe Method
1. **Pose Detection**: Extract body landmarks from video frames
2. **Coordinate Scaling**: Convert to metric units using reference lengths
3. **CG Tracking**: Track center of gravity throughout jump
4. **Phase Detection**: Automatic identification of jump phases

## Performance Metrics

### Basic Metrics
- **Jump Height (m)**: Vertical displacement during flight
- **Takeoff Velocity (m/s)**: Vertical velocity at liftoff
- **Flight Time (s)**: Duration of flight phase
- **Contact Time (s)**: Ground contact duration

### Advanced Metrics
- **Average Power (W)**: Mechanical power output during contact
- **Peak Power (W)**: Maximum instantaneous power
- **Relative Power (W/kg)**: Power normalized by body mass
- **Force (N)**: Ground reaction force
- **Impulse (Ns)**: Force-time integral

### Energy Calculations
- **Potential Energy (J)**: Gravitational potential energy at peak height
- **Kinetic Energy (J)**: Kinetic energy at takeoff
- **Total Energy (J)**: Sum of potential and kinetic energy

## Visualization Suite

### Jump Trajectory Plots
- **Height vs. Time**: Vertical displacement over time
- **Velocity Profile**: Velocity changes throughout jump
- **Force-Time Curve**: Ground reaction force during contact
- **Power-Time Curve**: Power output during contact

### Comparative Analysis
- **Multi-Jump Comparison**: Compare multiple jump attempts
- **Normative Data**: Comparison with reference populations
- **Progress Tracking**: Longitudinal performance monitoring
- **Intervention Effects**: Training or treatment effect analysis

### Interactive Reports
- **HTML Reports**: Web-based interactive visualizations
- **Data Export**: CSV files with all calculated metrics
- **Statistical Summaries**: Descriptive statistics for all metrics
- **Graphical Reports**: Publication-ready figures

## Usage

### GUI Mode (Recommended)

```python
from vaila.vaila_and_jump import vaila_and_jump

# Launch interactive interface
vaila_and_jump()
```

### Programmatic Usage

```python
import pandas as pd
from vaila.vaila_and_jump import calculate_jump_height, calculate_power

# Time-of-flight method
flight_data = pd.read_csv('flight_times.csv')
jump_height = calculate_jump_height(
    time_of_flight=flight_data['time_of_flight_s'],
    gravity=9.81
)

# Power calculation
power_metrics = calculate_power(
    force=flight_data['force_n'],
    height=jump_height,
    contact_time=flight_data['contact_time_s']
)
```

### Batch Processing

```python
import glob
from vaila.vaila_and_jump import process_jump_data

# Process multiple jump files
jump_files = glob.glob('/path/to/jumps/*.csv')

for jump_file in jump_files:
    process_jump_data(
        input_file=jump_file,
        output_dir='/path/to/results',
        calculation_method='flight_time',  # or 'jump_height' or 'mediapipe'
        gravity=9.81,
        mass_kg=75.0  # if not in data file
    )
```

## Advanced Configuration

### Custom Calculation Parameters

```python
# Configure calculation parameters
config = {
    'gravity': 9.81,              # m/s²
    'air_resistance': False,      # Account for air resistance
    'wind_factor': 0.0,           # Wind effect (m/s)
    'temperature': 20,            # Air temperature (°C)
    'altitude': 0,               # Altitude above sea level (m)
    'body_mass_kg': 75.0         # Participant body mass
}

process_jump_data(
    input_file='jump_data.csv',
    calculation_config=config
)
```

### MediaPipe Configuration

```python
# Configure MediaPipe processing
mediapipe_config = {
    'shank_length_m': 0.42,       # Reference length for scaling
    'coordinate_inversion': True, # Invert y-coordinates
    'smoothing_window': 5,        # Frame smoothing window
    'confidence_threshold': 0.5,  # Pose detection confidence
    'cg_calculation_method': 'weighted'  # CG calculation method
}

process_jump_data(
    input_file='mediapipe_data.csv',
    mediapipe_config=mediapipe_config
)
```

## Applications

### Sports Performance
- **Athletic Assessment**: Evaluate jump performance in athletes
- **Training Monitoring**: Track improvements over training periods
- **Talent Identification**: Compare performance across individuals
- **Injury Prevention**: Monitor performance changes post-injury

### Clinical Assessment
- **Rehabilitation Progress**: Track recovery of jumping ability
- **Neurological Assessment**: Evaluate motor function through jump performance
- **Geriatric Assessment**: Assess functional mobility in older adults
- **Pediatric Development**: Monitor motor development milestones

### Research Applications
- **Biomechanical Studies**: Investigate jump mechanics and energetics
- **Comparative Physiology**: Compare jump performance across species
- **Equipment Testing**: Evaluate athletic equipment effectiveness
- **Training Methodology**: Assess different training interventions

## Integration with vailá Ecosystem

This module integrates with other vailá tools:

- **Data Processing**: Use with filtering and smoothing tools for raw data
- **Visualization**: Compatible with 2D/3D plotting modules
- **Video Analysis**: Combine with markerless analysis for video-based jumps
- **Statistical Analysis**: Export data for statistical processing

## Quality Assurance

### Data Validation
- **Input Validation**: Check data format and completeness
- **Range Checking**: Verify calculated values are within physiological ranges
- **Consistency Checks**: Ensure calculations are internally consistent
- **Error Detection**: Identify and report calculation errors

### Performance Validation
- **Ground Truth Comparison**: Validate against known jump heights
- **Repeatability Analysis**: Assess consistency across multiple trials
- **Inter-observer Reliability**: Check consistency between different analysts
- **Equipment Calibration**: Verify measurement equipment accuracy

## Troubleshooting

### Common Issues

1. **Coordinate System Errors**: Ensure proper y-coordinate inversion for MediaPipe data
2. **Scaling Issues**: Verify reference length measurements for MediaPipe scaling
3. **Flight Time Detection**: Check for accurate flight phase identification
4. **Force Plate Synchronization**: Ensure proper synchronization with motion data

### Parameter Optimization

- **Gravity Value**: Use appropriate gravity for location/altitude
- **Reference Lengths**: Measure anatomical references accurately
- **Smoothing Parameters**: Adjust based on data noise levels
- **Threshold Values**: Tune detection thresholds for jump phases

## Version History

- **v0.1.0**: Initial implementation with time-of-flight and jump height methods
- **v0.0.9**: Added MediaPipe integration and coordinate transformation
- **v0.0.8**: Enhanced calculation algorithms and error handling
- **v0.0.7**: Added comprehensive performance metrics
- **v0.0.6**: Added HTML reporting and visualization suite
- **v0.0.5**: Added batch processing capabilities

## Requirements

### Core Dependencies
- **Python 3.8+**: Modern Python features
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib**: Visualization
- **SciPy**: Scientific computing

### Optional Dependencies
- **Tkinter**: GUI interface (usually included)
- **MediaPipe**: For pose estimation data processing
- **OpenCV**: For video frame processing

## References

- **Vertical Jump Biomechanics**: Physics and physiology of vertical jumping
- **Force Plate Analysis**: Ground reaction force measurement techniques
- **MediaPipe Pose**: Human pose estimation methodology
- **Sports Performance**: Athletic performance assessment methods
