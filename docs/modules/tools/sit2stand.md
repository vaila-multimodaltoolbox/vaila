# Sit-to-Stand Analysis Tool

## Overview

The **Sit-to-Stand Analysis Tool** provides comprehensive biomechanical analysis of sit-to-stand movements using force plate data. This module is specifically designed for clinical assessment of movement quality, stability, and functional capacity.

## Key Features

### 1. **Advanced Peak Detection**
- Utilizes `scipy.signal.find_peaks` with configurable parameters
- Detects peaks during movement phases and standing baseline
- Configurable height, distance, prominence, and width parameters
- Automatic time vector generation based on sampling frequency

### 2. **Stability Analysis**
- Measures oscillation around reference peak force
- Detects peaks in standing baseline for stability assessment
- Calculates crossings, deviations, and consistency metrics
- Noise and oscillation frequency analysis

### 3. **Energy Expenditure Calculation**
- Mechanical work based on force-displacement
- Metabolic energy estimation using body weight
- Movement efficiency metrics
- Per-movement energy breakdown

### 4. **Comprehensive Reporting**
- Individual file reports (TXT + CSV)
- Batch summary reports
- High-resolution plots (300 DPI PNG)
- Organized output directory structure

### 5. **Flexible Configuration**
- TOML configuration files for reproducibility
- Interactive GUI for quick analysis
- Command-line interface for batch processing
- Default configurations included

## Installation

The Sit-to-Stand tool is included in the VAILA package. Ensure you have the following dependencies:

```bash
pip install pandas numpy scipy matplotlib toml
```

For C3D file support:
```bash
pip install ezc3d
```

## Usage

### GUI Mode (Recommended)

1. Launch VAILA
2. Select "Sit to Stand" from the main menu
3. Configure analysis:
   - Select input directory with CSV/C3D files
   - Choose TOML configuration file (or use defaults)
   - Select output directory
4. Click "Run Analysis"

### Command-Line Mode

```bash
python sit2stand.py <config.toml> <input_directory> [output_directory] [file_format]
```

**Example:**
```bash
python sit2stand.py config.toml ./data ./results auto
```

**Parameters:**
- `config.toml`: Configuration file (use empty string "" for defaults)
- `input_directory`: Directory containing CSV/C3D files
- `output_directory`: Where to save results (optional)
- `file_format`: `auto`, `c3d`, or `csv` (default: auto)

## Configuration File (TOML)

### Complete Example

```toml
[analysis]
# Column containing vertical force data
force_column = "Force.Fz3"
fps = 2000.0  # Frames per second for time vector generation
body_weight = 18.43  # Body weight in kg for energy calculations

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
```

### Parameter Descriptions

#### `[analysis]`
- **`force_column`**: Name of the column containing vertical force data (e.g., "Force.Fz3")
- **`fps`**: Sampling frequency in Hz (used to generate time vector if Time column is invalid)
- **`body_weight`**: Subject's body weight in kilograms (for energy calculations)

#### `[filtering]`
- **`enabled`**: Enable/disable Butterworth low-pass filtering
- **`cutoff_frequency`**: Filter cutoff frequency in Hz
- **`sampling_frequency`**: Data sampling frequency in Hz
- **`order`**: Butterworth filter order (typically 4)

#### `[detection]`
- **`force_threshold`**: Minimum force to consider as movement (N)
- **`min_duration`**: Minimum duration for valid sit-to-stand phase (seconds)
- **`onset_threshold`**: Force threshold above baseline for movement onset (N)

#### `[detection.peak_detection]`
- **`height`**: Minimum absolute height of peaks in Newtons (optional)
- **`distance`**: Minimum distance between peaks in samples
- **`prominence`**: Minimum prominence of peaks in Newtons
- **`rel_height`**: Relative height for width calculation (0-1 scale)

#### `[stability]`
- **`enabled`**: Enable/disable stability analysis
- **`baseline_window`**: Time window after first peak for baseline (seconds)
- **`stability_threshold`**: Maximum deviation for stable standing (N)
- **`noise_analysis`**: Enable noise/oscillation frequency analysis
- **`rolling_window`**: Rolling window size for noise analysis (seconds)

## Output Structure

```
sit2stand_results/
├── filename1/
│   ├── filename1_force_plot.png
│   ├── filename1_analysis_report.txt
│   └── filename1_analysis_data.csv
├── filename2/
│   ├── filename2_force_plot.png
│   ├── filename2_analysis_report.txt
│   └── filename2_analysis_data.csv
├── batch_analysis_summary.txt
└── batch_analysis_summary.csv
```

## Analysis Metrics

### Basic Force Metrics
- Duration, Mean Force, Max Force, Min Force
- Total samples processed

### Movement Detection
- Number of phases detected
- Total movement time
- Average phase duration
- Phases per minute

### Time to Peak Metrics
- Time to first peak
- Time to maximum force
- Average time to peak across phases
- Time to peak variation (consistency)

### Rate of Force Development (RFD)
- Overall RFD (onset to peak)
- Early RFD (first 100ms)
- Peak RFD (maximum instantaneous rate)
- Weight transfer time

### Impulse & Power
- Total impulse (force-time integral)
- Average impulse per phase
- Peak power
- Average power
- Force rate of change

### Movement Quality
- Force coefficient of variation
- Force smoothness (jerk metric)
- Bilateral symmetry index
- Movement consistency score
- Number of peaks detected

### Stability Analysis
- **Reference Peak Force**: Maximum force peak used as reference
- **Stability Index**: 0-1 scale (higher = more stable)
- **Mean/Max Deviation**: From reference peak force
- **Points Above/Below**: Percentage of time above/below reference
- **Total Crossings**: Number of oscillations around reference
- **Standing Baseline Peaks**: Peaks detected during standing phase
  - Time of each peak (seconds)
  - Force value (Newtons)
  - Prominence (Newtons)
- **Noise Level**: Standard deviation of oscillations
- **Oscillation Frequency**: Frequency of force oscillations (Hz)
- **Stability Duration**: Duration of standing phase

### Energy Expenditure
- **Body Weight**: In kg and Newtons
- **Total Mechanical Work**: Force × displacement (Joules)
- **Total Metabolic Energy**: Based on METs and body weight (kcal)
- **Average Energy per Movement**: Energy cost per sit-to-stand (kcal)
- **Energy Efficiency**: Mechanical work / metabolic energy (%)
- **Movement Type**: Slow (>2s) or Normal (≤2s)

## Visualization

### Force Plot Features

The generated plot includes two subplots:

#### **Main Plot: Raw Force Signal**
- Complete raw force signal in blue
- Movement onset markers (triangles)
- Peak force markers (circles) with annotations
- Phase end markers (squares)
- **All peaks detected (stars)** in gold with labels
- **Standing baseline peaks (diamonds)** in dark blue with labels
- Reference lines:
  - Baseline (10th percentile)
  - Detection threshold
  - Maximum force
- Grid and legend

#### **RFD Plot: Rate of Force Development**
- Force derivative over time
- Highlighted RFD during movement phases
- Peak RFD markers
- Zero reference line

### Peak Detection Visualization

The tool detects and displays two types of peaks:

#### **All Peaks (Global Detection)**
**Markers:**
- ⭐ Star shape in **gold**
- Black border (1.5px) for contrast
- Size: 15 points

**Annotations:**
- Label: `P1`, `P2`, `P3`, etc. (first 5 peaks)
- Shows force value in Newtons
- Yellow box with orange border
- Positioned below the peak

**Purpose:**
- Complete analysis of all peaks in the signal
- Includes peaks during movement and standing
- Useful for identifying compensatory strategies
- Shows all significant force events

#### **Standing Baseline Peaks**
**Markers:**
- ◆ Diamond shape in **dark blue**
- White border (2px) for contrast
- Size: 12 points

**Annotations:**
- Label: `S1`, `S2`, `S3`, etc.
- Shows force value in Newtons
- Light blue box with dark blue border
- Positioned above the peak

**Purpose:**
- Detected **after stabilization period** (baseline_window)
- Analyzes postural stability in standing
- Excludes transitional movement phase
- Quantifies standing tremor/oscillation

## Clinical Applications

### 1. Pediatric Cerebral Palsy Assessment
- Time to peak metrics for motor control evaluation
- RFD for strength assessment
- Stability analysis for balance assessment
- Multiple peak detection for compensatory strategies

### 2. Elderly Fall Risk
- Stability index for balance assessment
- Standing baseline peaks for postural control
- Energy expenditure for functional capacity
- Movement consistency for reliability

### 3. Rehabilitation Monitoring
- Track improvements in RFD over time
- Monitor reduction in standing oscillations
- Assess energy efficiency improvements
- Compare pre/post intervention metrics

### 4. Athletic Performance
- Power output analysis
- Movement efficiency optimization
- Bilateral symmetry assessment
- Consistency training

## Troubleshooting

### Issue: No peaks detected

**Solutions:**
1. Lower `height` parameter in `[detection.peak_detection]`
2. Reduce `prominence` parameter
3. Check that `force_column` is correct
4. Verify force values are in Newtons (positive values)

### Issue: Too many peaks detected

**Solutions:**
1. Increase `height` parameter
2. Increase `distance` parameter
3. Increase `prominence` parameter
4. Enable filtering to smooth signal

### Issue: Incorrect time axis

**Solutions:**
1. Verify `fps` parameter matches your data
2. Check Time column in CSV has proper values
3. Script auto-generates time vector if Time column is invalid

### Issue: No standing peaks detected

**Solutions:**
1. Check that stability analysis is `enabled = true`
2. Lower `height` in peak detection parameters
3. Verify data extends beyond movement phase
4. Check baseline force values

## Best Practices

### 1. Configuration
- Start with default configuration
- Adjust `height` based on baseline + 15-25N
- Set `distance` based on sampling frequency (100ms recommended)
- Calibrate `prominence` to detect significant peaks only

### 2. Data Preparation
- Ensure force data is in Newtons
- Check for valid time column or set correct FPS
- Remove any header/footer rows in CSV
- Verify force plate is properly calibrated

### 3. Analysis
- Process multiple trials for reliability
- Compare metrics across sessions
- Use batch mode for consistency
- Save TOML configurations for reproducibility

### 4. Interpretation
- Focus on metrics relevant to your clinical question
- Compare to normative data when available
- Consider individual variability
- Integrate with other assessments

## Advanced Features

### Custom Peak Detection

Adjust parameters based on signal characteristics:

```toml
[detection.peak_detection]
# For high-frequency data (>1000 Hz)
distance = 200  # 200 samples = 100ms at 2000Hz

# For noisy signals
prominence = 15.0  # Higher prominence

# For weak signals
height = 100.0  # Lower height threshold
```

### Filtering Guidelines

```toml
[filtering]
enabled = true

# For force plate data
cutoff_frequency = 10.0  # 10 Hz typical for biomechanics
sampling_frequency = 2000.0  # Match your data

# For very noisy data
cutoff_frequency = 5.0  # Lower cutoff
order = 6  # Higher order (sharper cutoff)
```

### Peak Detection Strategy

The tool uses a two-stage peak detection approach:

#### **Stage 1: Global Peak Detection**
Detects **all peaks** in the entire force signal:
- Applied from start to end of signal
- Uses parameters from `[detection.peak_detection]`
- Identifies all significant force events
- Stored in `all_peaks_global` in results

**Configuration:**
```toml
[detection.peak_detection]
height = 139.0      # Minimum peak height (N)
distance = 200      # Min samples between peaks (100ms @ 2000Hz)
prominence = 10.0   # Minimum prominence (N)
```

#### **Stage 2: Standing Baseline Peak Detection**
Detects peaks **only after stabilization**:
- Starts at: `reference_peak + baseline_window`
- Excludes transitional movement phase
- Focuses on postural stability analysis
- Stored in `stability_metrics['standing_peaks']`

**Timing Logic:**
```
Signal Timeline:
├─────Movement Phase─────┤├─Stabil.Window─┤├──Standing Phase──┤
│                        │                 │                   │
Onset                   Peak              Start               End
                      (reference)      Standing Peak
                                       Detection

baseline_window (0.5s) →│←────────────────┤
```

**Key Differences:**
- **Global Peaks**: ALL peaks (movement + standing)
- **Standing Peaks**: ONLY peaks after stabilization
- Standing peaks exclude movement-related oscillations
- Both use same detection parameters but different signal regions

## References

1. **Energy Expenditure**: Nakagata et al. (2019) - PMC6473689
   - STS energy cost validation
   - MET values for different speeds
   
2. **Peak Detection**: SciPy signal processing documentation
   - `scipy.signal.find_peaks` implementation
   - Parameter optimization guidelines

3. **Clinical Metrics**: Biomechanical literature on sit-to-stand
   - RFD importance in motor control
   - Stability indices for balance assessment

## Version History

- **v0.0.4** (2025-10-16): 
  - Added standing baseline peak detection
  - Enhanced visualization with diamond markers
  - Improved energy expenditure calculations
  - Added comprehensive stability metrics
  
- **v0.0.3** (2025-10-14):
  - Added peak detection with scipy
  - Implemented stability analysis
  - Added energy expenditure calculations
  
- **v0.0.2**: 
  - GUI implementation
  - TOML configuration support
  
- **v0.0.1**: 
  - Initial release
  - Basic sit-to-stand detection

## Support

For questions, issues, or feature requests:
- Check the VAILA documentation
- Review example configurations in `tests/sit2stand/`
- Consult with the development team

---

**Last Updated**: October 16, 2025  
**Author**: Prof. Paulo Santiago  
**Module**: `vaila/sit2stand.py`

