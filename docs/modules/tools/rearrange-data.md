# Rearrange Data - CSV Data Processing and Manipulation Tool

## Overview

The **Rearrange Data** module provides comprehensive tools for manipulating, reorganizing, and processing CSV data files. It offers an interactive GUI for reordering columns, applying mathematical operations, converting units, and transforming data formats from various motion capture and video tracking systems.

## Features

### Core Functionality
- **Column Reordering**: Interactive drag-and-drop style column reorganization
- **Custom Math Operations**: Apply any NumPy, Pandas, or SciPy operation to columns
- **Unit Conversion**: Convert between metric, imperial, and specialized units
- **Batch Processing**: Process multiple CSV files simultaneously
- **Precision Control**: Maintain or specify decimal precision for output files
- **Lab Reference System Modification**: Rotate and transform coordinate systems

### Data Format Conversions
- **MediaPipe to vailá**: Convert MediaPipe pose estimation data
- **Kinovea to vailá**: Convert Kinovea tracking data
- **YOLO Tracker to vailá**: Convert YOLO object tracking data
- **DLC to vailá**: Convert DeepLabCut tracking data
- **Dvideo to vailá**: Convert Dvideo tracking data

### Advanced Operations
- **Merge CSV Files**: Combine multiple CSV files horizontally (by columns)
- **Stack CSV Files**: Append CSV files vertically (by rows)
- **Split Data**: Save specific row ranges or second half of data
- **Index Reset**: Reset first column to sequential numbering

## Custom Math Operations

### Available Functions

The Custom Math Operation feature provides access to the complete NumPy, Pandas, and SciPy ecosystems:

#### Basic Operators
- **Arithmetic**: `+`, `-`, `*`, `/`, `**` (power)
- **Comparison**: `<`, `>`, `<=`, `>=`, `==`, `!=`

#### NumPy Functions
- **Mathematical**: `np.sqrt()`, `np.abs()`, `np.power()`, `np.exp()`
- **Logarithms**: `np.log()`, `np.log10()`, `np.log2()`
- **Trigonometric**: `np.sin()`, `np.cos()`, `np.tan()`, `np.arcsin()`, `np.arccos()`, `np.arctan()`
- **Rounding**: `np.round()`, `np.floor()`, `np.ceil()`
- **Statistics**: `np.mean()`, `np.median()`, `np.std()`, `np.var()`
- **Conversions**: `np.deg2rad()`, `np.rad2deg()`
- **Constants**: `np.pi`, `np.e`

#### Pandas Operations
- **Rolling Statistics**: `pd.Series(x).rolling(window).mean()`, `.median()`, `.std()`
- **Cumulative**: `pd.Series(x).cumsum()`, `.cumprod()`
- **Shifting**: `pd.Series(x).shift(n)`
- **Interpolation**: `pd.Series(x).interpolate()`

#### SciPy Signal Processing
- **Filtering**: `scipy.signal.medfilt(x, kernel_size)`
- **Smoothing**: `scipy.signal.savgol_filter(x, window, polyorder)`
- **FFT**: `scipy.fft.fft(x)`, `scipy.fft.ifft(x)`

### Usage Examples

| Operation | Expression | Description |
|-----------|-----------|-------------|
| **Scale Data** | `x * 2.5` | Multiply all values by 2.5 |
| **Unit Conversion** | `x / 1000` | Convert millimeters to meters |
| **Square Root** | `np.sqrt(x)` | Calculate square root |
| **Square Values** | `x ** 2` | Square all values |
| **Absolute Value** | `np.abs(x)` | Remove negative signs |
| **Natural Log** | `np.log(x)` | Natural logarithm |
| **Degrees to Radians** | `np.deg2rad(x)` | Convert angle units |
| **Moving Average** | `pd.Series(x).rolling(5).mean()` | 5-point moving average |
| **Median Filter** | `scipy.signal.medfilt(x, 5)` | 5-point median filter |
| **Complex Expression** | `np.sqrt(np.abs(x)) * 2.5 + 10` | Multiple operations combined |
| **Normalize** | `(x - np.mean(x)) / np.std(x)` | Z-score normalization |
| **Clamp Values** | `np.clip(x, 0, 100)` | Limit values to range [0, 100] |

### How to Use Custom Math Operations

1. **Open Rearrange Data**: Launch the tool from vailá main menu
2. **Select Directory**: Choose directory containing CSV files
3. **Click "Custom Math Operation"** button
4. **Select Columns**: Choose one or multiple columns to process
5. **Enter Expression**: Type mathematical expression using `x` for column values
6. **Test Expression**: Click "Test Expression" to preview results on first 5 rows
7. **Apply Operation**: Click "Apply Operation" to process all files
8. **Review Output**: Files saved in `data_rearranged/` subdirectory

### Output File Naming

Files are saved with descriptive names including:
- Original filename
- Timestamp (`YYYYMMDD_HHMMSS`)
- Operation identifier (`mathop_`)
- Cleaned expression (special characters replaced)

**Example**: `force_data_20251015_143025_mathop_xdiv1000.csv`

## Unit Conversion

### Supported Unit Systems

#### Length Units
- Meters (m)
- Centimeters (cm)
- Millimeters (mm)
- Kilometers (km)
- Inches (in)
- Feet (ft)
- Yards (yd)
- Miles (mi)

#### Time Units
- Seconds (s)
- Minutes (min)
- Hours (hr)
- Days (day)

#### Angle Units
- Degrees (deg)
- Radians (rad)

#### Electrical Units
- Volts (V)
- Millivolts (mV)
- Microvolts (µV)

#### Velocity Units
- Kilometers per hour (km/h)
- Meters per second (m/s)
- Miles per hour (mph)

#### Force/Mass Units
- Kilograms (kg)
- Newtons (N)

#### Acceleration Units
- Meters per second squared (m/s²)
- Gravitational force (g)

#### Angular Velocity
- Rotations per second (rps)
- Revolutions per minute (rpm)
- Radians per second (rad/s)

### Conversion Options

**Convert All Columns**: Apply conversion to entire dataset
**Ignore First Column**: Skip time/frame column (recommended for temporal data)

## Lab Reference System Modification

Transform coordinate systems using predefined or custom rotations:

### Predefined Rotations
- **(A) Rotate 180° in Z axis**: Flip reference frame
- **(B) Rotate 90° clockwise in Z**: Quarter turn clockwise
- **(C) Rotate 90° counter-clockwise in Z**: Quarter turn counter-clockwise

### Custom Rotations
Enter custom Euler angles in format: `[x, y, z]` or `[x, y, z], order`

**Examples**:
- `[0, -45, 0]`: 45° rotation around Y axis
- `[90, 0, 180], zyx`: Complex rotation with ZYX order

## File Format Conversions

### MediaPipe to vailá Format
Converts MediaPipe pose estimation output to vailá pixel format:
- Extracts X, Y coordinates for each landmark
- Organizes as `frame, p1_x, p1_y, p2_x, p2_y, ...`
- Batch processes entire directories

### Kinovea to vailá Format
Converts Kinovea video analysis output:
- Standardizes header format
- Adjusts coordinate precision to 1 decimal place
- Creates vailá-compatible pixel format

### YOLO Tracker to vailá Format
Converts YOLO object tracking data:
- Handles multiple tracked objects
- Processes large files efficiently with chunking
- Organizes by frame with person IDs converted to point numbers

### DeepLabCut (DLC) to vailá Format
Converts DLC pose estimation output:
- Extracts likelihood scores
- Reorganizes multi-animal tracking data
- Maintains bodypart labels

### Dvideo to vailá Format
Converts Dvideo tracking files (.dat):
- Reads space-separated format
- Converts to standard CSV structure
- Preserves coordinate precision

## Data Manipulation Operations

### Row Operations

#### Save Second Half
Splits each CSV file and saves only the second half of the data:
- Automatically detects midpoint
- Resets first column (frame/time) to start from 0
- Useful for trial segmentation

#### Save/Delete Row Range
Select specific row ranges to save or remove:
- Interactive row range selection (`start:end`)
- Processes all files in batch
- Preserves column structure

### Column Operations

#### Reorder Columns
Interactive column reordering:
1. Select column(s) in listbox
2. Press Enter to move to new position
3. Enter target position number
4. Columns moved maintaining data integrity

#### Delete Columns
Remove unwanted columns:
1. Select column(s) to delete
2. Press 'd' key
3. Columns removed from all files in batch

#### Manual Range Selection
Select column ranges efficiently:
1. Press 'm' key
2. Enter range format: `start:end`
3. Move entire range to new position

### Merge and Stack Operations

#### Merge CSV Files (Horizontal)
Combine files by adding columns:
- Select base file and merge file
- Choose insertion position
- Columns from merge file inserted at specified position
- Maintains row count from base file

#### Stack CSV Files (Vertical)
Append files by adding rows:
- Select base file and stack file
- Choose stack position (start or end)
- Rows from stack file appended to base file
- Maintains column structure

## Index Reset

### Reset Index Column 0
Resets the first column to sequential numbering:
- Starts from 0
- Increments by 1 for each row
- Useful after row deletion or data combination
- Ensures proper frame/time indexing

## GUI Controls

### Keyboard Shortcuts
- **Enter**: Reorder selected columns
- **d**: Delete selected columns
- **m**: Manual range selection
- **l**: Edit rows (save/delete range)
- **Ctrl+S**: Save intermediate version
- **Ctrl+Z**: Undo last operation
- **Esc**: Save final version and exit

### Mouse Operations
- **Click**: Select single column
- **Ctrl+Click**: Select multiple columns
- **Shift+Click**: Select range of columns

## Output Organization

All processed files are saved in timestamped subdirectories:

```
original_directory/
├── data_rearranged/
│   ├── file1_20251015_120000_mathop_xdiv1000.csv
│   ├── file1_20251015_120030_unit_mm_to_m_FIRST_IGNORED.csv
│   ├── file2_20251015_120100_resetidx.csv
│   └── ...
├── Convert_MediaPipe_to_vaila_20251015_120200/
│   └── converted_files...
└── rotated_files/
    └── rotated_files...
```

## Performance Considerations

### Large File Handling
- **Files > 100 MB**: Automatic simplified mode with limited preview
- **Chunk Processing**: YOLO tracker conversion uses configurable chunk sizes
- **Memory Management**: Automatic garbage collection for large operations

### Processing Speed
- **Column Reordering**: Instantaneous for most files
- **Math Operations**: ~1-10 seconds per file depending on size and complexity
- **Unit Conversion**: ~1-5 seconds per file
- **Format Conversion**: Variable (5 seconds to several minutes for large files)

## Best Practices

### Data Safety
1. **Original Files Preserved**: Original files never modified
2. **Timestamped Output**: All outputs include timestamps
3. **Test First**: Use "Test Expression" before applying math operations
4. **Backup Important Data**: Keep backups before batch operations

### Efficient Workflows
1. **Check Precision**: Review decimal precision before saving
2. **Use Batch Operations**: Process all files in directory simultaneously
3. **Organize Output**: Use timestamped directories for organization
4. **Verify Results**: Check output files after complex operations

### Expression Writing Tips
1. **Use NumPy Prefix**: Always use `np.` for NumPy functions (e.g., `np.sqrt(x)`)
2. **Test on Sample**: Use "Test Expression" button to verify before applying
3. **Handle Edge Cases**: Consider zeros, negatives, NaN values
4. **Vectorize When Possible**: Pandas operations are more efficient than loops

## Integration with vailá Ecosystem

### Upstream Modules
Data sources that work with Rearrange Data:
- **readc3d_export**: C3D file exports
- **markerless_2d_analysis**: MediaPipe outputs
- **yolo_tracking**: YOLO tracker outputs
- **DeepLabCut**: DLC tracking data

### Downstream Modules
Processed data can be used with:
- **vailaplot2d**: 2D plotting and visualization
- **dlt3d**: 3D reconstruction
- **kinematics analysis**: Motion analysis tools
- **ML modules**: Machine learning model training

## Troubleshooting

### Common Issues

**1. Expression Errors**
- Ensure correct NumPy prefix: `np.sqrt(x)` not `sqrt(x)`
- Check for division by zero
- Verify column contains numeric data

**2. File Not Found**
- Check directory permissions
- Ensure CSV files are in selected directory
- Verify file extensions are `.csv`

**3. Precision Issues**
- Use "Detect Precision" feature before saving
- Adjust decimal places in save dialog
- Check for scientific notation in original data

**4. Large File Performance**
- Close other applications to free memory
- Use simplified mode for files > 100 MB
- Process files individually if batch fails

### Data Quality Issues

**Missing Values (NaN)**
- Math operations may produce NaN for invalid inputs (e.g., `log(negative)`)
- Check original data for completeness
- Use `np.nan_to_num()` to replace NaN values

**Coordinate System Issues**
- Verify correct rotation option selected
- Test with sample file before batch processing
- Check coordinate ranges after transformation

## Usage Examples

### Example 1: Convert Millimeters to Meters

```python
# GUI: Select "Custom Math Operation"
# Select all coordinate columns (X, Y, Z)
# Expression: x / 1000
# Apply to all files
```

**Result**: All coordinate values divided by 1000

### Example 2: Apply Moving Average Filter

```python
# Select force/acceleration columns
# Expression: pd.Series(x).rolling(5).mean()
# Apply operation
```

**Result**: 5-point moving average applied to selected columns

### Example 3: Convert Degrees to Radians

```python
# Select angle columns
# Expression: np.deg2rad(x)
# Apply to selected columns
```

**Result**: All angles converted from degrees to radians

### Example 4: Normalize Data

```python
# Select data columns
# Expression: (x - np.mean(x)) / np.std(x)
# Note: This applies per-value, not per-column
# For column-wise normalization, use preprocessing tools
```

### Example 5: Apply Median Filter

```python
# Select noisy signal columns
# Expression: scipy.signal.medfilt(x, 5)
# Apply to reduce noise
```

**Result**: Median filter applied to remove outliers

## Batch Processing Workflow

### Typical Workflow
1. **Organize Files**: Place all CSV files in single directory
2. **Launch Tool**: Run from vailá main menu → "Rearrange Data"
3. **Select Directory**: Choose directory containing files
4. **Review Structure**: Check columns in preview
5. **Apply Operations**:
   - Reorder columns as needed
   - Apply math operations
   - Convert units if necessary
6. **Save Results**: Use Ctrl+S for intermediate saves, Esc for final save
7. **Verify Output**: Check `data_rearranged/` subdirectory

### Multi-Step Processing Example

**Scenario**: Convert MediaPipe data, apply unit conversion, and filter

1. **Step 1**: Click "Convert MediaPipe to vailá"
   - Converts pose estimation to pixel format
   
2. **Step 2**: Select converted files directory
   - Apply unit conversion if needed
   
3. **Step 3**: Click "Custom Math Operation"
   - Apply smoothing filter: `pd.Series(x).rolling(3).mean()`
   
4. **Step 4**: Save final results
   - Files ready for 3D reconstruction or analysis

## Configuration and Settings

### Precision Control
- **Automatic Detection**: Preserves original decimal precision
- **Manual Override**: Specify decimal places during save (Ctrl+S or Esc)
- **Per-Column Precision**: Different precision for different columns

### File Naming Conventions
All output files include:
- Original base filename
- Timestamp: `YYYYMMDD_HHMMSS`
- Operation identifier: `_mathop_`, `_unit_`, `_resetidx_`, etc.
- Safe filename: Special characters replaced with alphanumeric

**Examples**:
- `data_20251015_143025_mathop_xdiv1000.csv`
- `trial_20251015_143100_unit_mm_to_m_FIRST_IGNORED.csv`
- `markers_20251015_143200_resetidx.csv`

## API Reference

### Main Function

```python
from vaila.rearrange_data import rearrange_data_in_directory

# Launch GUI
rearrange_data_in_directory()
```

### Programmatic Usage

```python
from vaila.rearrange_data import (
    reshapedata,
    convert_mediapipe_to_pixel_format,
    batch_convert_kinovea,
    batch_convert_yolo_tracker
)

# Reorder columns programmatically
reshapedata(
    file_path='data.csv',
    new_order=['Time', 'X', 'Y', 'Z'],
    save_directory='output/',
    suffix='_reordered'
)

# Convert MediaPipe data
convert_mediapipe_to_pixel_format(
    file_path='mediapipe_output.csv',
    save_directory='converted/'
)

# Batch convert Kinovea files
batch_convert_kinovea(directory_path='kinovea_data/')
```

## Version History

- **v0.0.6** (Oct 2025): Added custom math operations feature with NumPy/Pandas/SciPy support
- **v0.0.5** (Jul 2025): Improved filename sanitization and batch processing
- **v0.0.4**: Added second half save functionality
- **v0.0.3**: Added batch MediaPipe conversion
- **v0.0.2**: Added automatic directory creation
- **v0.0.1**: Initial version with core CSV reordering

## Dependencies

### Required
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `tkinter`: GUI interface

### Optional (for specific features)
- `scipy`: Signal processing operations
- `rich`: Enhanced console output

## Technical Notes

### Memory Management
- Large files (> 100 MB) use simplified preview mode
- Only 5 rows loaded for display
- Full file processed during save operations
- Automatic garbage collection after batch operations

### Data Type Handling
- Automatic numeric type detection
- NaN value preservation
- Scientific notation support
- Mixed type column handling

### Error Recovery
- Graceful handling of malformed CSV files
- Expression validation before batch application
- Rollback capability with Ctrl+Z
- Detailed error messages for debugging

## Support and Resources

### Related Modules
- **[Data Filtering](data-filtering.md)**: Advanced filtering operations
- **[2D Plotting](../visualization/plot-2d.md)**: Visualize processed data
- **[C3D ↔ CSV Conversion](c3d-csv-conversion.md)**: Motion capture file conversion

### Additional Resources
- **NumPy Documentation**: https://numpy.org/doc/
- **Pandas Documentation**: https://pandas.pydata.org/docs/
- **SciPy Documentation**: https://docs.scipy.org/

## Examples Gallery

### Scientific Data Processing

**Convert voltage to microvolts and apply noise filter**:
```python
Expression: scipy.signal.medfilt(x * 1e6, 5)
```

**Calculate velocity from position data** (finite differences):
```python
Expression: pd.Series(x).diff() * sampling_rate
```

**Apply baseline correction**:
```python
Expression: x - np.median(x)
```

### Biomechanical Applications

**Convert body weight percentage to actual force**:
```python
Expression: x * body_weight_in_newtons / 100
```

**Calculate joint angles from radians to degrees**:
```python
Expression: np.rad2deg(x)
```

**Normalize force data by body weight**:
```python
Expression: x / body_weight
```

### Signal Processing

**Low-pass filter effect using moving average**:
```python
Expression: pd.Series(x).rolling(10, center=True).mean()
```

**Remove DC offset**:
```python
Expression: x - np.mean(x)
```

**Scale to 0-1 range**:
```python
Expression: (x - np.min(x)) / (np.max(x) - np.min(x))
```

## Conclusion

The Rearrange Data module is a powerful and flexible tool for CSV data manipulation within the vailá ecosystem. Its combination of interactive GUI, batch processing capabilities, and support for complex mathematical operations makes it essential for preprocessing biomechanical, motion capture, and tracking data.

For questions or feature requests, please refer to the main vailá documentation or GitHub repository.

