"""
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
  
    ```python
    from sync_flash import get_median_brightness
    median_brightness = get_median_brightness("path/to/video.mp4", (50, 50, 100, 100))
    print(median_brightness)
    ```

Author: [Your Name]
Date: [Current Date]

"""

import cv2
import numpy as np


def get_median_brightness(video_file, region=None):
    """
    Extracts the median of the R, G, B values from a specified region of the video.

    Parameters:
    - video_file: path to the video file.
    - region: a tuple (x, y, width, height) defining the rectangular region.
              If None, the entire frame is considered.

    Returns:
    - median_rgb: a tuple containing the median R, G, B values.
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_file}")

    all_pixels = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if region:
            x, y, w, h = region
            frame = frame[y : y + h, x : x + w]

        # Convert frame to RGB and reshape it to a 2D array of pixels
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pixels = frame_rgb.reshape(-1, 3)
        all_pixels.append(pixels)

    cap.release()

    # Stack all pixels and calculate the median
    all_pixels = np.vstack(all_pixels)
    median_rgb = np.median(all_pixels, axis=0)

    return tuple(median_rgb)


if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    region = (50, 50, 100, 100)  # Example region (x, y, width, height)
    median_brightness = get_median_brightness(video_path, region)
    print(f"Median RGB values in the specified region: {median_brightness}")
