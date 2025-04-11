"""
Script: markerless_live.py
Author: Moser José (https://moserjose.com/),  Prof. Dr. Paulo Santiago
Version: 0.0.2
Created: April 9, 2025
Last Updated: April 9, 2025

Description:
This script performs real-time pose estimation and angle calculation using either YOLO or
MediaPipe's Pose model. It provides a graphical interface for selecting the pose detection
engine and configuring its parameters, allowing users to analyze movement in real-time
through their webcam.

Features:
- Dual engine support:
    - YOLO (better for multiple people detection)
    - MediaPipe (faster, optimized for single person)
- Real-time angle calculations for various body joints
- Configurable parameters for each engine:
    - YOLO: confidence threshold, model selection
    - MediaPipe: model complexity, detection confidence
- Visual feedback:
    - Skeleton overlay
    - Joint angles display
    - Person detection bounding box
- Data export:
    - CSV files with angle measurements
    - Angle plots over time
    - Automatic file naming with timestamps

Usage:
Run the script to open a graphical interface for:
1. Selecting the pose detection engine (YOLO or MediaPipe)
2. Configuring engine-specific parameters
3. Choosing the output directory for data saving
4. Starting real-time pose detection and angle calculation

How to Execute:
1. Ensure all dependencies are installed:
   - OpenCV: `pip install opencv-python`
   - MediaPipe: `pip install mediapipe`
   - Ultralytics YOLO: `pip install ultralytics`
   - Other dependencies: numpy, matplotlib, pandas, Tkinter
2. Run the script:
   python markerless_live.py

Requirements:
- Python 3.12.9
- OpenCV
- MediaPipe
- Ultralytics YOLO
- NumPy
- Matplotlib
- Tkinter

Output:
The script generates:
1. Real-time display with:
   - Pose skeleton overlay
   - Joint angle measurements
   - Detection confidence
2. Data files (when closing):
   - CSV with angle measurements
   - Plot of angles over time
   - Automatic file naming with timestamp

License:
This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU GPLv3 (General Public License Version 3) along with this program.
If not, see <https://www.gnu.org/licenses/>.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import mediapipe as mp
import os
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog


def list_available_cameras():
    """
    Lists all available camera devices and their capabilities.
    Returns a list of dictionaries containing camera information.
    """
    available_cameras = []

    # Common resolutions to test
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]  # VGA  # HD  # Full HD

    # Common FPS values to test
    fps_values = [15, 30, 60]

    # Test first 10 camera indexes
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera name
            camera_name = f"Camera {i}"

            # Test resolutions
            supported_resolutions = []
            for width, height in resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (actual_width, actual_height) not in supported_resolutions:
                    supported_resolutions.append((actual_width, actual_height))

            # Test FPS
            supported_fps = []
            for fps in fps_values:
                cap.set(cv2.CAP_PROP_FPS, fps)
                actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
                if actual_fps not in supported_fps:
                    supported_fps.append(actual_fps)

            # Sort resolutions and FPS values
            supported_resolutions.sort(key=lambda x: x[0] * x[1])
            supported_fps.sort()

            available_cameras.append(
                {
                    "index": i,
                    "name": camera_name,
                    "resolutions": supported_resolutions,
                    "fps_values": supported_fps,
                }
            )

            cap.release()

    return available_cameras


# Detection settings
BUFFER_SIZE = 100  # Number of frames to keep in the angle buffer

# Output settings
SAVE_DATA = True


# ===== CODE FROM ANGLE_CALCULATOR.PY =====
class AngleCalculator:
    """Base class for angle calculators."""

    def calculate_angle(self, p1, p2, p3):
        """Calculate the angle between three points."""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)

        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # Handle numerical precision errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)


class YOLOAngleCalculator(AngleCalculator):
    """Calculador de ângulos usando pontos-chave do YOLO."""

    def __init__(self):
        super().__init__()
        # Mapping based on the output that shows 7 valid points (0-6)
        # Based on the output that shows 7 valid points (0-6)
        self.joint_map = {
            "neck": [1, 0, 2],  # nose-neck-left_shoulder
            "right_shoulder": [0, 1, 5],  # neck-right_shoulder-right_arm
            "left_shoulder": [0, 2, 6],  # neck-left_shoulder-left_arm
            "right_arm": [1, 5, 3],  # right_shoulder-right_arm-right_elbow
            "left_arm": [2, 6, 4],  # left_shoulder-left_arm-left_elbow
        }

        # Default keypoint structure (YOLOv11n-pose)
        self.num_keypoints = 17

    def adapt_to_keypoint_structure(self, num_keypoints):
        """Adapt the joint map to the detected keypoint structure."""
        self.num_keypoints = num_keypoints
        print(f"Adapting to keypoint structure with {num_keypoints} keypoints")

        # For YOLOv11n-pose, we will keep the default mapping
        # since it is already optimized for the keypoint structure of this model
        print("Using YOLOv11n-pose keypoint structure")

    def process_keypoints(self, keypoints):
        """Process the YOLO keypoints and calculate the angles."""
        angles = {}

        # Debug: Print keypoints shape and content
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Processing keypoints...")

        # Check if the keypoints have enough confidence (third column)
        confidence_threshold = 0.3  # Reduced to 0.3 to capture more points
        valid_keypoints = keypoints[:, 2] > confidence_threshold

        # Create mask for valid keypoints (non-zero coordinates and high confidence)
        valid_mask = np.logical_and(
            valid_keypoints,
            np.logical_and(
                keypoints[:, 0] != 0,  # x is not zero
                keypoints[:, 1] != 0,  # y is not zero
            ),
        )

        print(f"Valid keypoints mask: {valid_mask}")
        print(f"Valid keypoints coordinates:")
        for i, valid in enumerate(valid_mask):
            if valid:
                print(
                    f"Point {i}: ({keypoints[i][0]:.1f}, {keypoints[i][1]:.1f}), conf: {keypoints[i][2]:.3f}"
                )

        for joint_name, indices in self.joint_map.items():
            p1_idx, p2_idx, p3_idx = indices

            # Check if all necessary points are valid
            if valid_mask[p1_idx] and valid_mask[p2_idx] and valid_mask[p3_idx]:
                p1 = keypoints[p1_idx][:2]  # Use only x,y
                p2 = keypoints[p2_idx][:2]
                p3 = keypoints[p3_idx][:2]

                angle = self.calculate_angle(p1, p2, p3)
                angles[joint_name] = angle
                print(f"Calculated {joint_name}: {angle:.4f}°")
                print(f"  Points used: {p1_idx}({p1}), {p2_idx}({p2}), {p3_idx}({p3})")

        print(f"Final angles: {angles}")
        return angles


class MediaPipeAngleCalculator(AngleCalculator):
    """Angle calculator using MediaPipe keypoints."""

    def __init__(self):
        super().__init__()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

        # MediaPipe joint mapping
        self.joint_map = {
            "right_elbow": [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                self.mp_pose.PoseLandmark.RIGHT_WRIST,
            ],
            "left_elbow": [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
                self.mp_pose.PoseLandmark.LEFT_WRIST,
            ],
            "right_shoulder": [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            ],
            "left_shoulder": [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_ELBOW,
            ],
            "right_hip": [
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
            ],
            "left_hip": [
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
            ],
            "right_knee": [
                self.mp_pose.PoseLandmark.RIGHT_HIP,
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            ],
            "left_knee": [
                self.mp_pose.PoseLandmark.LEFT_HIP,
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
            ],
            "right_ankle": [
                self.mp_pose.PoseLandmark.RIGHT_KNEE,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
            ],
            "left_ankle": [
                self.mp_pose.PoseLandmark.LEFT_KNEE,
                self.mp_pose.PoseLandmark.LEFT_ANKLE,
                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            ],
        }

    def process_frame(self, frame):
        """Process a frame with MediaPipe and calculate angles."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        angles = {}
        skeleton_connections = []
        visible_landmarks = []

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Calculate angles for each defined joint
            for joint_name, indices in self.joint_map.items():
                p1_idx, p2_idx, p3_idx = indices

                # Check if necessary landmarks are visible
                if (
                    landmarks[p1_idx].visibility > 0.5
                    and landmarks[p2_idx].visibility > 0.5
                    and landmarks[p3_idx].visibility > 0.5
                ):

                    p1 = [
                        landmarks[p1_idx].x * frame.shape[1],
                        landmarks[p1_idx].y * frame.shape[0],
                    ]
                    p2 = [
                        landmarks[p2_idx].x * frame.shape[1],
                        landmarks[p2_idx].y * frame.shape[0],
                    ]
                    p3 = [
                        landmarks[p3_idx].x * frame.shape[1],
                        landmarks[p3_idx].y * frame.shape[0],
                    ]

                    angle = self.calculate_angle(p1, p2, p3)
                    angles[joint_name] = angle

                    # Add connections for skeleton drawing
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    pt3 = (int(p3[0]), int(p3[1]))

                    skeleton_connections.append((pt1, pt2))
                    skeleton_connections.append((pt2, pt3))

            # Get coordinates of all visible landmarks for drawing
            visible_landmarks = []
            for i, landmark in enumerate(landmarks):
                if landmark.visibility > 0.5:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    visible_landmarks.append((x, y, i))

        return angles, skeleton_connections, visible_landmarks


# ===== END OF CODE FROM ANGLE_CALCULATOR.PY =====


def download_model(model_name):
    """
    Download a specific YOLO model to the vaila/vaila/models directory.

    Args:
        model_name: Name of the model to download (e.g., "yolov11n.pt")

    Returns:
        Path to the downloaded model
    """
    # Correct path to vaila/vaila/models
    script_dir = os.path.dirname(os.path.abspath(__file__))  # vaila/
    vaila_dir = os.path.dirname(script_dir)  # root directory
    models_dir = os.path.join(vaila_dir, "vaila", "models")  # vaila/vaila/models

    # Create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be downloaded to: {models_dir}")

    model_path = os.path.join(models_dir, model_name)

    # Check if model already exists
    if os.path.exists(model_path):
        print(
            f"Model {model_name} already exists at {model_path}, using existing file."
        )
        return model_path

    print(f"Downloading {model_name} to {model_path}...")
    try:
        # Create a temporary YOLO model instance that will download the weights
        model = YOLO(model_name)

        # Get the path where YOLO downloaded the model
        source_path = model.ckpt_path

        if os.path.exists(source_path):
            # Copy the downloaded model to our models directory
            import shutil

            shutil.copy2(source_path, model_path)
            print(f"Successfully saved {model_name} to {model_path}")
        else:
            print(f"YOLO downloaded the model but couldn't find it at {source_path}")

    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        print("Trying alternative download method...")

        try:
            # Alternative download method using requests
            import requests

            # URL for the model - updated to use the correct model name and version
            if model_name.lower().startswith("yolo11"):
                version_tag = "v11.0.0"
            else:
                version_tag = "v0.0.0"
            url = f"https://github.com/ultralytics/assets/releases/download/{version_tag}/{model_name}"

            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Save the file
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Successfully downloaded {model_name} using requests")
        except Exception as e2:
            print(f"All download methods failed for {model_name}: {e2}")
            print("Trying to find the model in the local directory...")

            # Try to find the model in the local directory
            local_model_path = os.path.join(script_dir, model_name)
            if os.path.exists(local_model_path):
                print(f"Found model at {local_model_path}, copying to {model_path}")
                import shutil

                shutil.copy2(local_model_path, model_path)
                return model_path
            else:
                print(f"Could not find model {model_name} locally or download it.")
                print(
                    "Please manually download the model and place it in the models directory."
                )

    return model_path


class MovementAnalyzer:
    def __init__(
        self,
        engine="yolo",
        model_name=None,
        conf_threshold=0.3,
        model_complexity=1,
        min_detection_confidence=0.5,
        camera_device=0,
        camera_fps=30,
        camera_width=640,
        camera_height=480,
        output_dir="output",
    ):
        """
        Initialize the MovementAnalyzer with the specified parameters.

        Args:
            engine (str): Detection engine to use ('yolo' or 'mediapipe')
            model_name (str): Name of the YOLO model file (only for YOLO engine)
            conf_threshold (float): Confidence threshold for YOLO detection
            model_complexity (int): Model complexity for MediaPipe (0, 1, or 2)
            min_detection_confidence (float): Minimum detection confidence for MediaPipe
            camera_device (int): Camera device index
            camera_fps (int): Target camera FPS
            camera_width (int): Target camera width
            camera_height (int): Target camera height
            output_dir (str): Directory to save output files
        """
        self.engine = engine.lower()

        # Save camera settings
        self.camera_device = camera_device
        self.camera_fps = camera_fps
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.output_dir = output_dir

        # Initialize YOLO model for detection
        if self.engine == "yolo":
            # Use specified model or default
            self.model_name = model_name if model_name else "yolo11n-pose.pt"
            self.model = YOLO(self.get_model_path(self.model_name), verbose=False)
            self.angle_calculator = YOLOAngleCalculator()
            # Set confidence threshold
            self.conf_threshold = conf_threshold

            # Detect YOLO model type and adapt keypoint processing
            self.detect_yolo_model_type()
        elif self.engine == "mediapipe":
            self.model = None  # We don't need YOLO model for MediaPipe
            self.angle_calculator = MediaPipeAngleCalculator()
            # Configure MediaPipe with provided parameters
            self.angle_calculator.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=min_detection_confidence,
            )
            # Define skeleton connections for MediaPipe
            self.mp_pose = self.angle_calculator.mp_pose
            self.mediapipe_connections = [
                # Right arm
                (
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                    self.mp_pose.PoseLandmark.RIGHT_WRIST,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_WRIST,
                    self.mp_pose.PoseLandmark.RIGHT_PINKY,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_WRIST,
                    self.mp_pose.PoseLandmark.RIGHT_INDEX,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_WRIST,
                    self.mp_pose.PoseLandmark.RIGHT_THUMB,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_PINKY,
                    self.mp_pose.PoseLandmark.RIGHT_INDEX,
                ),
                # Left arm
                (
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_ELBOW,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_ELBOW,
                    self.mp_pose.PoseLandmark.LEFT_WRIST,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_WRIST,
                    self.mp_pose.PoseLandmark.LEFT_PINKY,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_WRIST,
                    self.mp_pose.PoseLandmark.LEFT_INDEX,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_WRIST,
                    self.mp_pose.PoseLandmark.LEFT_THUMB,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_PINKY,
                    self.mp_pose.PoseLandmark.LEFT_INDEX,
                ),
                # Torso
                (
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                ),
                # Right leg
                (
                    self.mp_pose.PoseLandmark.RIGHT_HIP,
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_KNEE,
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                    self.mp_pose.PoseLandmark.RIGHT_HEEL,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_HEEL,
                    self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE,
                    self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
                ),
                # Left leg
                (
                    self.mp_pose.PoseLandmark.LEFT_HIP,
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_KNEE,
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                    self.mp_pose.PoseLandmark.LEFT_HEEL,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_HEEL,
                    self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_ANKLE,
                    self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                ),
                # Face (optional, but adds completeness to the skeleton)
                (
                    self.mp_pose.PoseLandmark.NOSE,
                    self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_EYE_INNER,
                    self.mp_pose.PoseLandmark.LEFT_EYE,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_EYE,
                    self.mp_pose.PoseLandmark.LEFT_EYE_OUTER,
                ),
                (
                    self.mp_pose.PoseLandmark.NOSE,
                    self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_EYE_INNER,
                    self.mp_pose.PoseLandmark.RIGHT_EYE,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_EYE,
                    self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                ),
                (
                    self.mp_pose.PoseLandmark.MOUTH_LEFT,
                    self.mp_pose.PoseLandmark.MOUTH_RIGHT,
                ),
                (
                    self.mp_pose.PoseLandmark.LEFT_EYE_OUTER,
                    self.mp_pose.PoseLandmark.LEFT_EAR,
                ),
                (
                    self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
                    self.mp_pose.PoseLandmark.RIGHT_EAR,
                ),
            ]
        else:
            raise ValueError(
                f"Unrecognized engine: {engine}. Use 'yolo' or 'mediapipe'."
            )

        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_device)

        # Configure camera settings
        if self.camera_fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        if self.camera_width is not None and self.camera_height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        # Verify camera settings
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Camera initialized with settings:")
        print(f"FPS: {actual_fps}")
        print(f"Resolution: {actual_width}x{actual_height}")

        self.angle_buffer = []
        self.start_time = time.time()

    def get_model_path(self, model_name="yolo11n-pose.pt"):
        """Get the path to the YOLO model, downloading it if necessary."""
        # Get the correct path relative to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vaila_dir = os.path.dirname(script_dir)
        models_dir = os.path.join(vaila_dir, "vaila", "models")
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, model_name)

        # Download if not exists
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, downloading...")
            model_path = download_model(model_name)
        else:
            print(f"Using existing model at: {model_path}")

        return model_path

    def detect_yolo_model_type(self):
        """Detect the YOLO model type and adapt keypoint processing accordingly."""
        # Create a small test frame
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Run inference on the test frame
        results = self.model(test_frame, conf=self.conf_threshold, verbose=False)

        # Check if keypoints are available
        if len(results) > 0:
            print(f"YOLO results type: {type(results)}")
            print(f"First result attributes: {dir(results[0])}")

            # Try to find keypoints in different possible locations
            keypoints = None
            if hasattr(results[0], "keypoints") and results[0].keypoints is not None:
                print("Found keypoints in results[0].keypoints")
                if len(results[0].keypoints.data) > 0:
                    keypoints = results[0].keypoints.data[0].cpu().numpy()
                    print(f"Keypoints from results[0].keypoints: {keypoints.shape}")
            elif hasattr(results[0], "poses") and results[0].poses is not None:
                print("Found poses in results[0].poses")
                if len(results[0].poses) > 0:
                    keypoints = results[0].poses[0].cpu().numpy()
                    print(f"Keypoints from results[0].poses: {keypoints.shape}")

            if keypoints is not None:
                # Get the keypoints shape
                keypoints_shape = keypoints.shape
                print(f"Detected YOLO model with keypoints shape: {keypoints_shape}")

                # Adapt the angle calculator based on the keypoints shape
                if len(keypoints_shape) >= 2:
                    num_keypoints = (
                        keypoints_shape[1]
                        if len(keypoints_shape) > 1
                        else keypoints_shape[0]
                    )
                    print(f"Number of keypoints: {num_keypoints}")

                    # Update the angle calculator with the detected keypoint structure
                    self.angle_calculator.adapt_to_keypoint_structure(num_keypoints)
            else:
                print(
                    "Warning: Could not detect keypoints in the YOLO model. Using default keypoint structure."
                )
                print("Available attributes in results[0]:", dir(results[0]))

                # Try to find pose-related attributes
                for attr in dir(results[0]):
                    if (
                        "pose" in attr.lower()
                        or "keypoint" in attr.lower()
                        or "landmark" in attr.lower()
                    ):
                        print(f"Found potential pose-related attribute: {attr}")
                        try:
                            value = getattr(results[0], attr)
                            print(f"Value type: {type(value)}")
                            if hasattr(value, "shape"):
                                print(f"Shape: {value.shape}")
                        except Exception as e:
                            print(f"Error accessing attribute {attr}: {e}")
        else:
            print("No results from YOLO model on test frame")

    def process_frame(self, frame):
        """Process a single frame and detect poses."""
        processed_frame = frame.copy()

        if self.engine == "yolo":
            # Using YOLO for detection and angle calculation
            # Use a lower confidence threshold for better detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)

            # Debug: Print the structure of results to understand what's available
            if len(results) > 0:
                print(f"YOLO results type: {type(results)}")
                print(f"First result attributes: {dir(results[0])}")

                # Check for pose-related attributes
                pose_attrs = [
                    attr
                    for attr in dir(results[0])
                    if "pose" in attr.lower()
                    or "keypoint" in attr.lower()
                    or "landmark" in attr.lower()
                ]
                if pose_attrs:
                    print(f"Found pose-related attributes: {pose_attrs}")

                # Try to find keypoints in different possible locations
                keypoints = None
                if (
                    hasattr(results[0], "keypoints")
                    and results[0].keypoints is not None
                ):
                    print("Found keypoints in results[0].keypoints")
                    if len(results[0].keypoints.data) > 0:
                        keypoints = results[0].keypoints.data[0].cpu().numpy()
                        print(f"Keypoints from results[0].keypoints: {keypoints.shape}")
                elif hasattr(results[0], "poses") and results[0].poses is not None:
                    print("Found poses in results[0].poses")
                    if len(results[0].poses) > 0:
                        keypoints = results[0].poses[0].cpu().numpy()
                        print(f"Keypoints from results[0].poses: {keypoints.shape}")
                elif (
                    hasattr(results[0], "landmarks")
                    and results[0].landmarks is not None
                ):
                    print("Found landmarks in results[0].landmarks")
                    if len(results[0].landmarks) > 0:
                        keypoints = results[0].landmarks[0].cpu().numpy()
                        print(f"Keypoints from results[0].landmarks: {keypoints.shape}")

                if keypoints is not None:
                    print(f"Keypoints shape: {keypoints.shape}")
                    print(f"Keypoints content: {keypoints}")

                    # Draw bounding box (bounding box) around the person
                    if hasattr(results[0], "boxes") and results[0].boxes is not None:
                        boxes = results[0].boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        # Draw rectangle around the person
                        cv2.rectangle(
                            processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2
                        )

                        # Add label "Person" with confidence
                        label = f"Person {conf:.2f}"
                        cv2.putText(
                            processed_frame,
                            label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                    # Define skeleton connections (pairs of keypoint indices)
                    # Adapt to YOLO12n keypoint format
                    skeleton = [
                        # Upper body
                        (5, 7),
                        (7, 9),
                        (6, 8),
                        (8, 10),
                        # Torso
                        (5, 6),
                        (5, 11),
                        (6, 12),
                        (11, 12),
                        # Lower body
                        (11, 13),
                        (13, 15),
                        (12, 14),
                        (14, 16),
                    ]

                    # Draw skeleton lines
                    for connection in skeleton:
                        # Check if indices are within the limits of the keypoints array
                        if (
                            connection[0] < len(keypoints)
                            and connection[1] < len(keypoints)
                            and all(keypoints[connection[0]] != 0)
                            and all(keypoints[connection[1]] != 0)
                        ):
                            pt1 = (
                                int(keypoints[connection[0]][0]),
                                int(keypoints[connection[0]][1]),
                            )
                            pt2 = (
                                int(keypoints[connection[1]][0]),
                                int(keypoints[connection[1]][1]),
                            )
                            cv2.line(processed_frame, pt1, pt2, (0, 255, 0), 2)

                    # Calculate angles using YOLO calculator
                    angles = self.angle_calculator.process_keypoints(keypoints)
                    print(f"Calculated angles: {angles}")

                    # Draw keypoints (excluding face keypoints)
                    for i, kp in enumerate(keypoints):
                        if (
                            i < len(keypoints) and all(kp != 0) and i >= 5
                        ):  # Ignore face keypoints (0-4)
                            cv2.circle(
                                processed_frame,
                                (int(kp[0]), int(kp[1])),
                                4,
                                (0, 0, 255),
                                -1,
                            )

                    # Store angles in buffer if available
                    if angles:
                        self.angle_buffer.append(
                            {
                                "timestamp": time.time() - self.start_time,
                                "angles": angles,
                            }
                        )

                        # Keep only the last measurements (BUFFER_SIZE)
                        if len(self.angle_buffer) > BUFFER_SIZE:
                            self.angle_buffer.pop(0)
                else:
                    print("No keypoints detected in this frame")
                    # Try to find any pose-related data
                    for attr in dir(results[0]):
                        if (
                            "pose" in attr.lower()
                            or "keypoint" in attr.lower()
                            or "landmark" in attr.lower()
                        ):
                            try:
                                value = getattr(results[0], attr)
                                print(f"Attribute {attr} type: {type(value)}")
                                if hasattr(value, "shape"):
                                    print(f"Attribute {attr} shape: {value.shape}")
                            except Exception as e:
                                print(f"Error accessing attribute {attr}: {e}")

        elif self.engine == "mediapipe":
            # Use MediaPipe for detection and angle calculation
            angles, _, landmarks = self.angle_calculator.process_frame(frame)

            # Process the frame with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.angle_calculator.pose.process(rgb_frame)

            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark

                # Create bounding box around the person
                x_coords = []
                y_coords = []
                for landmark in pose_landmarks:
                    if landmark.visibility > 0.5:
                        x_coords.append(landmark.x * frame.shape[1])
                        y_coords.append(landmark.y * frame.shape[0])

                if x_coords and y_coords:
                    # Calculate the bounding box coordinates
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))

                    # Add some padding to the bounding box
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)

                    # Draw rectangle around the person
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    # Add label "Person"
                    cv2.putText(
                        processed_frame,
                        "Person",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

                # Draw skeleton connections using custom definitions
                for connection in self.mediapipe_connections:
                    start_idx, end_idx = connection

                    if (
                        pose_landmarks[start_idx].visibility > 0.5
                        and pose_landmarks[end_idx].visibility > 0.5
                    ):

                        start_point = (
                            int(pose_landmarks[start_idx].x * frame.shape[1]),
                            int(pose_landmarks[start_idx].y * frame.shape[0]),
                        )

                        end_point = (
                            int(pose_landmarks[end_idx].x * frame.shape[1]),
                            int(pose_landmarks[end_idx].y * frame.shape[0]),
                        )

                        cv2.line(
                            processed_frame, start_point, end_point, (0, 255, 0), 2
                        )

                # Draw visible landmarks
                for i, landmark in enumerate(pose_landmarks):
                    if landmark.visibility > 0.5:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(processed_frame, (x, y), 4, (0, 0, 255), -1)

            # Store angles in buffer if available
            if angles:
                self.angle_buffer.append(
                    {"timestamp": time.time() - self.start_time, "angles": angles}
                )

                # Keep only the last measurements (BUFFER_SIZE)
                if len(self.angle_buffer) > BUFFER_SIZE:
                    self.angle_buffer.pop(0)

        # Draw angles on the frame (independent of the engine)
        if self.angle_buffer and "angles" in self.angle_buffer[-1]:
            angles = self.angle_buffer[-1]["angles"]

            # Text configurations
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            font_color = (255, 255, 255)
            bg_color = (0, 0, 0)

            # Calculate text size for background
            max_text_width = 0
            for joint, angle in angles.items():
                # Ensure that the angle is a number and use appropriate formatting
                if isinstance(angle, (int, float)):
                    text = (
                        f"{joint}: {angle:0.1f}°"  # Fixed format with a decimal place
                    )
                else:
                    text = f"{joint}: N/A"
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                max_text_width = max(max_text_width, text_width)

            # Create semi-transparent background
            padding = 10
            rows = (len(angles) + 1) // 2  # Divide into two columns
            overlay = processed_frame.copy()
            cv2.rectangle(
                overlay,
                (10, 10),
                (max_text_width + 2 * padding + 10, (rows * 35) + 2 * padding + 10),
                bg_color,
                -1,
            )
            cv2.addWeighted(overlay, 0.5, processed_frame, 0.5, 0, processed_frame)

            # Draw angles in two columns
            y_offset = 35
            x_offset = 15
            col = 0
            for joint, angle in angles.items():
                text = f"{joint}: {angle:0.1f}°"
                # Draw text with outline for better visibility
                cv2.putText(
                    processed_frame,
                    text,
                    (x_offset, y_offset),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness + 1,
                )  # Outline
                cv2.putText(
                    processed_frame,
                    text,
                    (x_offset, y_offset),
                    font,
                    font_scale,
                    font_color,
                    font_thickness,
                )  # Texto

                col += 1
                if col % 2 == 0:
                    y_offset += 35
                    x_offset = 15
                else:
                    x_offset = max_text_width + 30

        # Show which engine is being used
        cv2.putText(
            processed_frame,
            f"Engine: {self.engine.upper()}",
            (processed_frame.shape[1] - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        return processed_frame

    def save_data(self):
        """Save collected data to CSV file."""
        if SAVE_DATA and self.angle_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            print(f"Saving data to directory: {self.output_dir}")

            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Save as CSV with full precision (4 decimal places)
            csv_filename = os.path.join(
                self.output_dir, f"movement_data_{self.engine}_{timestamp}.csv"
            )
            plot_filename = os.path.join(
                self.output_dir, f"joint_angles_{self.engine}_{timestamp}.png"
            )

            try:
                if self.angle_buffer:
                    joint_names = set()
                    for data in self.angle_buffer:
                        joint_names.update(data["angles"].keys())
                    joint_names = sorted(list(joint_names))

                    with open(csv_filename, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["timestamp"] + joint_names)

                        # Write data rows with 4 decimal places
                        for data in self.angle_buffer:
                            row = [f"{data['timestamp']:.4f}"]
                            for joint in joint_names:
                                if joint in data["angles"]:
                                    row.append(f"{data['angles'][joint]:.4f}")
                                else:
                                    row.append("")
                            writer.writerow(row)

                    # Create plot
                    plt.figure(figsize=(10, 6))
                    times = [d["timestamp"] for d in self.angle_buffer]

                    for joint in joint_names:
                        angles = []
                        for d in self.angle_buffer:
                            if joint in d["angles"]:
                                angles.append(d["angles"][joint])
                            else:
                                angles.append(None)

                        valid_times = []
                        valid_angles = []
                        for t, a in zip(times, angles):
                            if a is not None:
                                valid_times.append(t)
                                valid_angles.append(a)

                        if valid_times and valid_angles:
                            plt.plot(valid_times, valid_angles, label=joint)

                    plt.xlabel("Time (s)")
                    plt.ylabel("Angle (degrees)")
                    plt.title(f"Joint Angles Over Time - {self.engine.upper()}")
                    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
                    plt.legend()
                    plt.grid(True)

                    # Save plot
                    plt.savefig(plot_filename)
                    plt.close()

                    print(
                        f"Data saved successfully to:\n{csv_filename}\n{plot_filename}"
                    )

            except Exception as e:
                print(f"Error saving data: {str(e)}")
                print(f"Attempted to save to: {self.output_dir}")
                backup_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "backup_output"
                )
                os.makedirs(backup_dir, exist_ok=True)
                backup_file = os.path.join(
                    backup_dir, f"movement_data_{self.engine}_{timestamp}.csv"
                )
                print(f"Attempting to save to backup location: {backup_file}")
                try:
                    with open(backup_file, "w", newline="") as f:
                        # ... same saving code as above ...
                        pass
                except Exception as e2:
                    print(f"Failed to save to backup location: {str(e2)}")

    def run(self):
        """Main loop for the movement analyzer."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)

                # Display the frame
                cv2.imshow("Movement Analyzer", processed_frame)
                # Break loop with 'q' pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.save_data()
            self.cap.release()
            cv2.destroyAllWindows()


def run_markerless_live():
    """
    Run the markerless movement analysis with user-selected parameters.
    """
    # Create a root window and keep it as the main window
    root = tk.Tk()
    root.withdraw()

    # Force the root window to be in front
    root.attributes("-topmost", True)

    # List available cameras
    available_cameras = list_available_cameras()
    if not available_cameras:
        messagebox.showerror("Error", "No cameras found!", parent=root)
        root.destroy()
        return

    # Create camera selection message
    camera_message = "Available cameras:\n\n"
    for cam in available_cameras:
        camera_message += f"{cam['index']}: {cam['name']}\n"
        camera_message += f"   Supported resolutions: {cam['resolutions']}\n"
        camera_message += f"   Supported FPS: {cam['fps_values']}\n\n"

    # Ask user to select camera
    camera_id = simpledialog.askinteger(
        "Select Camera",
        camera_message + "\nEnter camera ID:",
        initialvalue=0,
        minvalue=0,
        maxvalue=len(available_cameras) - 1,
        parent=root,
    )

    if camera_id is None:
        messagebox.showerror("Error", "No camera selected!", parent=root)
        root.destroy()
        return

    selected_camera = next(
        (cam for cam in available_cameras if cam["index"] == camera_id), None
    )
    if not selected_camera:
        messagebox.showerror("Error", "Invalid camera selection!", parent=root)
        root.destroy()
        return

    # Ask user to select resolution
    resolution_options = [
        f"{width}x{height}" for width, height in selected_camera["resolutions"]
    ]
    resolution_message = "Select resolution:\n\n" + "\n".join(
        f"{i}: {res}" for i, res in enumerate(resolution_options)
    )

    resolution_choice = simpledialog.askinteger(
        "Select Resolution",
        resolution_message + "\n\nEnter resolution number:",
        initialvalue=0,
        minvalue=0,
        maxvalue=len(resolution_options) - 1,
        parent=root,
    )

    if resolution_choice is None or resolution_choice >= len(
        selected_camera["resolutions"]
    ):
        messagebox.showinfo(
            "Default Resolution", "Using default resolution (640x480)", parent=root
        )
        camera_width, camera_height = 640, 480
    else:
        camera_width, camera_height = selected_camera["resolutions"][resolution_choice]

    # Ask user to select FPS
    fps_message = "Select FPS:\n\n" + "\n".join(
        f"{i}: {fps} FPS" for i, fps in enumerate(selected_camera["fps_values"])
    )

    fps_choice = simpledialog.askinteger(
        "Select FPS",
        fps_message + "\n\nEnter FPS number:",
        initialvalue=0,
        minvalue=0,
        maxvalue=len(selected_camera["fps_values"]) - 1,
        parent=root,
    )

    if fps_choice is None or fps_choice >= len(selected_camera["fps_values"]):
        messagebox.showinfo("Default FPS", "Using default FPS (30)", parent=root)
        camera_fps = 30
    else:
        camera_fps = selected_camera["fps_values"][fps_choice]

    # Show camera configuration
    camera_config = f"""
    Camera Configuration:
    --------------------
    Camera: {selected_camera['name']}
    Resolution: {camera_width}x{camera_height}
    FPS: {camera_fps}
    """
    messagebox.showinfo("Camera Configuration", camera_config, parent=root)

    # Continue with engine selection
    # Ask user to select the engine
    engine_choice = simpledialog.askstring(
        "Select Engine",
        "Choose the analysis engine:\n\n1: YOLO (Better for multiple people)\n2: MediaPipe (Faster, single person)",
        initialvalue="1",
        parent=root,
    )

    # Map the choice to the engine name and get parameters
    if engine_choice == "1":
        selected_engine = "yolo"
        # Configure YOLO specific parameters
        conf_threshold = simpledialog.askfloat(
            "YOLO Parameters",
            "Enter confidence threshold (0.0 to 1.0):",
            initialvalue=0.3,
            minvalue=0.0,
            maxvalue=1.0,
            parent=root,
        )
        if conf_threshold is None:
            conf_threshold = 0.3

        model_choice = simpledialog.askstring(
            "YOLO Model",
            "Choose the YOLO model:\n\n1: YOLOv11n-pose\n2: YOLOv8n-pose",
            initialvalue="1",
            parent=root,
        )
        if model_choice == "1":
            model_name = "yolo11n-pose.pt"
        else:
            model_name = "yolov8n-pose.pt"

        # Initialize MediaPipe parameters with default values
        model_complexity = 1
        min_detection_confidence = 0.5

    elif engine_choice == "2":
        selected_engine = "mediapipe"
        # Configure MediaPipe specific parameters
        model_complexity = simpledialog.askinteger(
            "MediaPipe Parameters",
            "Enter model complexity (0, 1, or 2):",
            initialvalue=1,
            minvalue=0,
            maxvalue=2,
            parent=root,
        )
        if model_complexity is None:
            model_complexity = 1  # default value

        min_detection_confidence = simpledialog.askfloat(
            "MediaPipe Parameters",
            "Enter minimum detection confidence (0.0 to 1.0):",
            initialvalue=0.5,
            minvalue=0.0,
            maxvalue=1.0,
            parent=root,
        )
        if min_detection_confidence is None:
            min_detection_confidence = 0.5  # default value

        # Initialize YOLO parameters with default values
        model_name = None
        conf_threshold = 0.3
    else:
        messagebox.showerror("Error", "Invalid selection. Using default engine (YOLO).")
        selected_engine = "yolo"
        model_name = "yolo11n-pose.pt"
        conf_threshold = 0.3
        model_complexity = 1
        min_detection_confidence = 0.5

    # Configure output directory using directory selection dialog
    default_output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output"
    )
    output_dir = filedialog.askdirectory(
        title="Select Output Directory",
        initialdir=default_output_dir,
        mustexist=False,
        parent=root,
    )

    if not output_dir:
        output_dir = default_output_dir
        messagebox.showinfo(
            "Default Directory",
            f"No directory selected. Using default directory:\n{default_output_dir}",
            parent=root,
        )

    # Ensure the selected directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Show selected configuration
    config_message = f"""
    Selected Configuration:
    ----------------------
    Engine: {selected_engine.upper()}
    Output Directory: {output_dir}
    """

    if selected_engine == "yolo":
        config_message += f"""
    YOLO Parameters:
    - Confidence Threshold: {conf_threshold}
    - Model: {model_name}
    """
    else:
        config_message += f"""
    MediaPipe Parameters:
    - Model Complexity: {model_complexity}
    - Min Detection Confidence: {min_detection_confidence}
    """

    messagebox.showinfo("Configuration", config_message, parent=root)

    # Destroy the root window before starting the analyzer
    root.destroy()

    # Initialize and run the analyzer with the selected parameters
    analyzer = MovementAnalyzer(
        engine=selected_engine,
        model_name=model_name,
        conf_threshold=conf_threshold,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        camera_device=selected_camera["index"],
        camera_fps=camera_fps,
        camera_width=camera_width,
        camera_height=camera_height,
        output_dir=output_dir,
    )

    # Run the analyzer
    analyzer.run()


if __name__ == "__main__":
    run_markerless_live()
