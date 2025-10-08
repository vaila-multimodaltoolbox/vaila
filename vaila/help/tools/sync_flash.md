# sync_flash

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/sync_flash.py`
- **Lines:** 87
- **Size:** 2732 characters

- **Author:** [Your Name]
- **GUI Interface:** ❌ No

## 📖 Description


sync_flash.py

This script provides a function to automatically detect brightness levels in a specific
region of a video file by calculating the median of the R, G, and B values in that region.
This functionality can be used to help synchronize videos based on flashes or sudden
changes in brightness.

Features:
- Extracts the median R, G, and B values from a specified region of each frame in a video.
- The region for analysis can be customized by specifying coordinates and dimensions.
- Can be used as a standalone tool or imported into another script for video synchronization.

Dependencies:
- cv2 (OpenCV): For video capture and processing.
- numpy: For efficient numerical operations and median calculation.

Usage:
- Import the `get_median_brightness` function into another script or use it directly
  in this script's `__main__` block for testing or standalone operation.

Example:
- To calculate the median brightness in a region (x=50, y=50, width=100, height=100)
  of a video:

    ``...

## 🔧 Main Functions

**Total functions found:** 1

- `get_median_brightness`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
