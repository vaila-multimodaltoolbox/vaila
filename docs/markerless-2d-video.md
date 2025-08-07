# Markerless 2D Analysis

## Overview

The Markerless 2D Analysis module provides tools for analyzing human movement using computer vision techniques.

## Features

- **Single-person tracking**: Fast analysis for individual subjects
- **Multi-person tracking**: Advanced analysis with YOLO integration
- **Joint angle calculation**: Automatic calculation of joint angles
- **Movement analysis**: Comprehensive movement pattern analysis

## Usage

### Standard Mode (Faster)

```python
from vaila.markerless_2D_analysis import process_videos_in_directory
process_videos_in_directory()
```

### Advanced Mode (Multi-person)

```python
from vaila.markerless2d_analysis_v2 import process_videos_in_directory
process_videos_in_directory()
```

## Input Requirements

- **Video files**: MP4, AVI, MOV formats supported
- **Camera setup**: Single or multiple cameras
- **Lighting**: Good lighting conditions recommended

## Output

- **CSV files**: Joint coordinates and angles
- **Plots**: Movement visualization
- **Reports**: Analysis summaries

## Configuration

The module uses MediaPipe for pose estimation and can be configured via TOML files.
