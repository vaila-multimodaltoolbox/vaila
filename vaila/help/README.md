# vailá Script Help Documentation

This directory contains help documentation for all Python scripts in the `vaila/` directory.

## Structure

Each script in `vaila/` has its own help documentation:
- **Markdown format** (`.md`) - For easy editing and version control
- **HTML format** (`.html`) - For web viewing

## Directory Organization

Help files are organized by category/subdirectory:

- **analysis/** - Analysis scripts (IMU, EMG, force plate, etc.)
- **ml/** - Machine Learning scripts (YOLO, ML walkway, etc.)
- **processing/** - Data processing scripts (DLT, filtering, etc.)
- **tools/** - Utility tools
- **utils/** - Utility functions
- **visualization/** - Visualization scripts
- **sit2stand/** - Sit-to-stand analysis
- Root level - General help files

## Script Help Format

Each script help should include:

1. **Module Information**
   - Category
   - File path
   - Lines of code
   - Version
   - Author
   - GUI Interface (Yes/No)

2. **Description**
   - What the script does
   - Key features
   - Use cases

3. **Main Functions**
   - List of main functions
   - Function descriptions

4. **Configuration Parameters**
   - All configurable parameters
   - Default values
   - Parameter ranges

5. **Output Files**
   - File formats
   - File naming conventions
   - Output structure

6. **Usage**
   - GUI mode instructions
   - Programmatic usage examples
   - Command-line usage (if applicable)

7. **Requirements**
   - System requirements
   - Python dependencies
   - Hardware requirements

8. **Performance Characteristics**
   - Processing speed
   - Memory usage
   - Best use cases

9. **Troubleshooting**
   - Common issues
   - Solutions
   - Performance tips

10. **Integration**
    - Compatible modules
    - Data flow
    - Integration examples

## Current Help Files

### Analysis Scripts
- `markerless_2d_analysis.md/html` - Standard MediaPipe pose estimation
- `markerless_3d_analysis.md/html` - 3D pose estimation
- `markerless_live.md/html` - Live pose estimation
- `imu_analysis.md/html` - IMU data analysis
- `emg_labiocom.md/html` - EMG analysis
- `forceplate_analysis.md/html` - Force plate analysis
- `gnss_analysis.md/html` - GNSS/GPS analysis
- `mocap_analysis.md/html` - Motion capture analysis
- `cluster_analysis.md/html` - Cluster analysis
- `cube2d_kinematics.md/html` - 2D kinematics
- `run_vector_coding.md/html` - Vector coding analysis
- `vaila_and_jump.md/html` - Jump analysis
- `animal_open_field.md/html` - Animal open field analysis

### Machine Learning Scripts
- `markerless2d_analysis_v2.md/html` - Advanced YOLO+MediaPipe pose estimation
- `markerless2d_mpyolo.md/html` - MediaPipe+YOLO integration
- `markerless3d_analysis_v2.md/html` - Advanced 3D pose estimation
- `vaila_mlwalkway.md/html` - ML walkway analysis
- `ml_models_training.md/html` - ML model training
- `ml_valid_models.md/html` - ML model validation
- `walkway_ml_prediction.md/html` - Walkway ML prediction
- `yolotrain.md/html` - YOLO training
- `yolov11track.md/html` - YOLOv11 tracking
- `yolov12track.md/html` - YOLOv12 tracking

### Processing Scripts
- `dlt2d.md/html` - 2D DLT calibration
- `dlt3d.md/html` - 3D DLT calibration
- `rec2d.md/html` - 2D reconstruction
- `rec3d.md/html` - 3D reconstruction
- `filtering.md/html` - Data filtering
- `filter_utils.md/html` - Filter utilities
- `rearrange_data.md/html` - Data rearrangement
- `reid_markers.md/html` - Marker re-identification
- And more...

## Adding New Script Help

When adding help for a new script:

1. Determine the appropriate category/subdirectory
2. Create both `.md` and `.html` files with the script name
3. Follow the standard format outlined above
4. Include all relevant technical details
5. Update this README with the new script

## Related Documentation

- Button documentation: `docs/vaila_buttons/` - Contains documentation for GUI buttons
- Module documentation: `docs/modules/` - Contains module-level documentation

---

**Last Updated:** November 2025
