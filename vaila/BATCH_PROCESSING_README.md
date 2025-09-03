# vailá CSV to C3D Batch Processing

## Overview

The `readcsv_export.py` script now includes **batch processing capabilities** that allow you to convert multiple CSV files to C3D format automatically, without having to process each file individually.

## New Features

### Batch Processing
- **Process all CSV files** in a directory at once
- **Automatic parameter application** to all files
- **Timestamped output directories** for organized results
- **Progress tracking** with success/failure counts
- **Error handling** for individual files

### Enhanced Automation
- **No user prompts** during file processing
- **Consistent parameters** across all files
- **Automatic file naming** and organization
- **Batch summary reports** upon completion

## How to Use

### Option 1: Interactive Mode (Recommended for beginners)

1. **Run the script:**
   ```bash
   python readcsv_export.py
   ```

2. **Choose processing mode:**
   - **Yes** = Batch processing (all files in directory)
   - **No** = Single file processing (one file at a time)

3. **For batch processing:**
   - Select **input directory** containing CSV files
   - Select **output directory** for C3D files
   - Enter **point data rate** (Hz)
   - Choose if you have **analog data**
   - Enter **analog data rate** if applicable
   - Select **unit conversion factor**
   - Choose if you want **markers sorted alphabetically**

4. **Wait for completion:**
   - Script processes all files automatically
   - Shows progress for each file
   - Displays final summary

### Option 2: Direct Function Call (For advanced users)

```python
from readcsv_export import batch_convert_csv_to_c3d

# The function will prompt for directories and parameters
batch_convert_csv_to_c3d()
```

### Option 3: Programmatic Usage (For developers)

```python
from readcsv_export import auto_create_c3d_from_csv
import pandas as pd

# Load your CSV data
df = pd.read_csv("your_file.csv")

# Convert to C3D with specific parameters
auto_create_c3d_from_csv(
    points_df=df,
    output_path="output.c3d",
    point_rate=100,
    analog_rate=1000,
    conversion_factor=1.0,
    sort_markers=False
)
```

## File Structure Requirements

### Input Directory
```
input_directory/
├── file1.csv
├── file2.csv
├── file3.csv
└── ...
```

### Output Directory Structure
```
output_directory/
└── csv2c3d_20250903_075550/
    ├── file1.c3d
    ├── file2.c3d
    ├── file3.c3d
    └── ...
```

## CSV File Format Requirements

### Point Data CSV
- **First column**: Frame number or timestamp
- **Subsequent columns**: Marker coordinates in groups of 3 (X, Y, Z)
- **Example:**
  ```
  frame,P1_X,P1_Y,P1_Z,P2_X,P2_Y,P2_Z,P3_X,P3_Y,P3_Z
  0,1.23,4.56,7.89,2.34,5.67,8.90,3.45,6.78,9.01
  1,1.24,4.57,7.90,2.35,5.68,8.91,3.46,6.79,9.02
  ```

### Analog Data CSV (Optional)
- **First column**: Frame number or timestamp
- **Subsequent columns**: Analog signal values
- **Naming convention**: `filename_analog.csv` (same base name as point data)

## Parameters

### Common Parameters (Applied to all files)
- **Point Rate**: Sampling frequency for motion data (Hz)
- **Analog Rate**: Sampling frequency for analog data (Hz)
- **Conversion Factor**: Unit conversion (e.g., cm to meters)
- **Sort Markers**: Alphabetical ordering of marker labels

### Automatic Processing
- **Header Sanitization**: Removes invalid characters
- **Coordinate Validation**: Ensures proper X, Y, Z structure
- **Error Handling**: Continues processing even if individual files fail

## Example Workflow

### 1. Prepare Your Data
```
project_data/
├── session1/
│   ├── trial1.csv
│   ├── trial1_analog.csv
│   ├── trial2.csv
│   └── trial2_analog.csv
└── session2/
    ├── trial3.csv
    └── trial3_analog.csv
```

### 2. Run Batch Processing
```bash
python readcsv_export.py
# Choose: Yes (Batch processing)
# Input: project_data/session1/
# Output: project_results/
# Point Rate: 100 Hz
# Analog Data: Yes
# Analog Rate: 1000 Hz
# Conversion: cm to m (100)
# Sort Markers: Yes
```

### 3. Get Results
```
project_results/
└── csv2c3d_20250903_075550/
    ├── trial1.c3d
    ├── trial2.c3d
    └── batch_summary.txt
```

## Error Handling

### Individual File Failures
- **Continues processing** other files
- **Logs errors** for failed conversions
- **Shows summary** of successes and failures

### Common Issues
- **Invalid CSV format**: Check column structure
- **Missing coordinates**: Ensure X, Y, Z for each marker
- **File permissions**: Verify read/write access
- **Memory issues**: Process smaller batches if needed

## Performance Tips

### For Large Datasets
- **Close other applications** to free memory
- **Process in smaller batches** if memory is limited
- **Use SSD storage** for faster I/O operations

### For Many Files
- **Batch by session** rather than all at once
- **Monitor system resources** during processing
- **Use timestamped outputs** to avoid overwrites

## Troubleshooting

### Script Won't Start
- Check Python version (3.7+ required)
- Verify all dependencies are installed
- Ensure tkinter is available

### No Files Found
- Check file extensions (.csv)
- Verify directory permissions
- Ensure files are not hidden

### Conversion Errors
- Validate CSV format
- Check coordinate structure
- Verify marker naming conventions

## Dependencies

- **Python 3.7+**
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **ezc3d**: C3D file creation
- **tkinter**: GUI elements
- **pathlib**: Path operations

## Support

For issues or questions:
- **Email**: paulosantiago@usp.br
- **GitHub**: [vaila-multimodaltoolbox/vaila](https://github.com/vaila-multimodaltoolbox/vaila)
- **Documentation**: See help files in the vaila/help directory

---

**Version**: 0.1.0  
**Last Updated**: September 3, 2025  
**Author**: Prof. Paulo R. P. Santiago
