# C3D ↔ CSV Conversion

## Overview

The C3D ↔ CSV Conversion tools provide comprehensive conversion between C3D motion capture files and CSV format, enabling seamless data exchange between different motion capture software and analysis tools.

## Features

- **C3D to CSV Conversion**: Extract all data types from C3D files into CSV format
- **CSV to C3D Conversion**: Convert CSV data back to C3D format
- **Batch Processing**: Process multiple files simultaneously
- **Complete Data Extraction**: Extract markers, analogs, events, force platforms, and metadata
- **Metadata Preservation**: Maintain all C3D file metadata and parameters
- **Excel Export Option**: Optional export to Excel format for compatibility
- **Error Handling**: Robust errorrrr handling for corrupted or incomplete files

## Supported Data Types

### From C3D Files
- **Marker Data**: 3D coordinates of anatomical markers
- **Analog Data**: Analog signals (EMG, force, acceleration, etc.)
- **Events**: Event markers with labels and timestamps
- **Force Platform Data**: Ground reaction force and center of pressure data
- **Subject Information**: Participant demographics and trial information
- **Trial Parameters**: Sampling rates, units, and calibration data

### To C3D Files
- **Marker Trajectories**: 3D marker positions over time
- **Analog Signals**: EMG, force, and other analog data
- **Events**: Event markers and labels
- **Subject Parameters**: Participant and trial metadata

## C3D to CSV Conversion

### Input Requirements
- **C3D Files**: Standard C3D format motion capture files
- **Directory Structure**: Organized file structure for batch processing

### Output Files

For each C3D file processed, the tool generates:

1. **Marker Data CSV** (`*_markers.csv`):
   - 3D coordinates (X, Y, Z) for each marker
   - Frame numbers and timestamps
   - Units preserved from original C3D file

2. **Analog Data CSV** (`*_analogs.csv`):
   - All analog channels with proper labeling
   - Sampling rates and units
   - Time-synchronized with marker data

3. **Events CSV** (`*_events.csv`):
   - Event labels and timestamps
   - Event descriptions and contexts

4. **Force Platform CSV** (`*_platforms.csv`):
   - Ground reaction forces (Fx, Fy, Fz)
   - Center of pressure coordinates (CoPx, CoPy)
   - Moment data (Mx, My, Mz)

5. **Metadata Files**:
   - **Info File** (`*_info.txt`): Complete C3D file metadata
   - **Short Info** (`*_short_info.txt`): Key parameters summary
   - **Parameter Groups** (`*_parameters.txt`): Organized parameter listing

6. **Summary Files**:
   - **Data Statistics** (`*_statistics.txt`): Statistical summary of all data
   - **Header Summary** (`*_headers.txt`): Column headers and data structure

### Processing Options

#### Data Extraction Options
- **Marker Data**: Extract 3D marker trajectories
- **Analog Data**: Extract analog signals and units
- **Events**: Extract event markers and labels
- **Force Platforms**: Extract ground reaction force data
- **Points Residuals**: Extract marker tracking residuals

#### Output Format Options
- **CSV Format**: Standard comma-separated values
- **Excel Format**: Optional Excel export (slower for large files)
- **Custom Delimiters**: Configurable field separators

#### Filtering Options
- **Gap Handling**: Options for handling missing data
- **Smoothing**: Optional data smoothing during conversion
- **Units Conversion**: Automatic unit conversion if needed

## CSV to C3D Conversion

### Input Requirements
- **CSV Files**: Properly formatted CSV files with marker trajectories
- **Metadata**: Subject information and trial parameters
- **Configuration**: C3D file structure specifications

### Output Files
- **C3D Files**: Standard C3D format motion capture files
- **Processing Log**: Conversion details and warnings

## Usage

### GUI Mode (Recommended)

```python
from vaila.readc3d_export import convert_c3d_to_csv
from vaila.readcsv_export import convert_csv_to_c3d

# Convert C3D to CSV
convert_c3d_to_csv()

# Convert CSV to C3D
convert_csv_to_c3d()
```

### Programmatic Usage

```python
import pandas as pd
from vaila.readc3d_export import importc3d

# Load C3D file programmatically
data = importc3d('motion_capture_file.c3d')

# Access different data types
markers = data['markers']  # 3D marker coordinates
analogs = data['analogs']  # Analog signals
events = data['events']    # Event data
platforms = data['platforms']  # Force platform data

# Save as CSV
markers.to_csv('markers.csv', index=False)
analogs.to_csv('analogs.csv', index=False)
```

### Batch Processing

```python
from vaila.readc3d_export import batch_convert_c3d_to_csv

# Process entire directory
batch_convert_c3d_to_csv(
    input_directory='/path/to/c3d/files',
    output_directory='/path/to/csv/output',
    save_excel=False  # Set to True for Excel output
)
```

## Data Structure

### C3D File Structure
C3D files contain multiple data types organized in groups:

```
C3D File
├── Header Information
│   ├── File signature and format
│   ├── Number of frames, markers, analogs
│   └── Sampling rates and units
├── Point Data (Markers)
│   ├── 3D coordinates (X, Y, Z)
│   ├── Residual values
│   └── Camera masks
├── Analog Data
│   ├── Channel data and labels
│   ├── Units and scaling factors
│   └── Sampling rates
├── Events
│   ├── Event labels and descriptions
│   └── Timestamps
└── Force Platform Data
    ├── Forces and moments
    ├── Center of pressure
    └── Calibration matrices
```

### CSV Output Structure
The CSV files maintain the temporal structure of the original data:

```csv
# Markers CSV
Frame,Time,Marker1_X,Marker1_Y,Marker1_Z,Marker2_X,Marker2_Y,Marker2_Z,...

# Analogs CSV
Frame,Time,EMG1,EMG2,Force_X,Force_Y,Force_Z,...

# Events CSV
Frame,Time,Event_Label,Event_Description

# Platforms CSV
Frame,Time,Platform1_Fx,Platform1_Fy,Platform1_Fz,Platform1_CoPx,Platform1_CoPy,...
```

## Performance Considerations

### Processing Speed
- **Small Files** (< 1000 frames): ~1-5 seconds per file
- **Medium Files** (1000-10000 frames): ~5-30 seconds per file
- **Large Files** (> 10000 frames): ~30 seconds to several minutes per file

### Memory Usage
- **Memory Efficient**: Processes data in chunks to minimize memory usage
- **Large Files**: May require 2-8 GB RAM for very large C3D files
- **Batch Processing**: Consider processing files individually for limited RAM

### Storage Requirements
- **Output Size**: CSV files typically 3-5x larger than original C3D files
- **Temporary Files**: Temporary processing files may be created
- **Excel Export**: Significantly larger file sizes and longer processing times

## Quality Assurance

### Data Integrity Checks
- **Frame Count Validation**: Verify all frames are processed
- **Coordinate Validation**: Check for unrealistic coordinate values
- **Unit Consistency**: Ensure units are properly preserved
- **Metadata Completeness**: Verify all metadata is extracted

### Error Handling
- **Corrupted Files**: Graceful handling of corrupted C3D files
- **Missing Data**: Proper handling of gaps in data
- **Format Variations**: Support for different C3D format variations

## Integration with vailá Ecosystem

This conversion tool integrates with other vailá modules:

- **Motion Capture Analysis**: Use with cluster and full-body analysis tools
- **Visualization**: Compatible with 2D/3D plotting modules
- **Data Processing**: Output can be processed with filtering tools
- **Machine Learning**: Converted data can be used for ML model training

## Troubleshooting

### Common Issues

1. **C3D File Not Recognized**: Check file format and integrity
2. **Missing Markers**: Verify marker labeling in original C3D file
3. **Analog Channel Issues**: Check analog channel configuration
4. **Memory Errors**: Process large files individually or increase system memory

### Data Quality Issues

- **Coordinate Spikes**: May indicate tracking errorrrrs in original capture
- **Missing Frames**: Check for data acquisition issues
- **Unit Inconsistencies**: Verify calibration in original C3D file

## Version History

- **v0.0.3**: Added force platform data extraction and improved errorrrr handling
- **v0.0.2**: Added batch processing and Excel export options
- **v0.0.1**: Initial implementation with basic C3D to CSV conversion

## References

- **C3D Format Specification**: Official C3D file format documentation
- **Motion Capture Standards**: ISB recommendations for motion capture data
- **Data Exchange**: Standards for biomechanical data formats
