"""
================================================================================
Rotation Tools - 3D Rotation and Transformation Toolkit
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-07-19
Version: 1.0

Overview:

This Python script provides a comprehensive set of tools for performing 3D rotation and transformation operations using numpy and scipy. It includes methods for creating orthonormal bases from sets of points, calculating rotation matrices, converting rotations to Euler angles and quaternions, and rotating datasets.

Main Features:

    Orthonormal Base Creation:
        - `createortbase`: Generates an orthonormal base from three 3D points, supporting different configurations ('A', 'B', 'C', 'D').
        - `createortbase_4points`: Constructs an orthonormal base using four 3D points, tailored for both trunk and pelvis configurations.

    Rotation Matrix Calculation:
        - `calcmatrot`: Computes the rotation matrix between two bases or from one base to the canonical basis.

    Euler Angles and Quaternions:
        - `rotmat2euler`: Converts a rotation matrix to Euler angles in degrees.
        - `rotmat2quat`: Converts a rotation matrix to quaternions.

    Data Rotation:
        - `rotdata`: Rotates a set of data points using specified rotation angles around the x, y, and z axes, with customizable order of rotations (e.g., 'xyz', 'zyx').

Key Functions and Their Functionality:

    createortbase():
        Creates an orthonormal base from three points in space (p1, p2, p3), following different configurations ('A', 'B', 'C', 'D') to accommodate various biomechanical structures.

    createortbase_4points():
        Creates an orthonormal base using four points. Primarily designed for calculating the trunk and pelvis bases using 3D motion capture data from anatomical landmarks.

    calcmatrot():
        Computes the rotation matrix between two orthonormal bases, defaulting to the canonical basis if only one is provided.

    rotmat2euler():
        Converts a 3x3 rotation matrix into Euler angles, expressed in degrees, useful for analyzing rotational motion in biomechanics.

    rotmat2quat():
        Converts a 3x3 rotation matrix into quaternions, commonly used in 3D graphics and kinematic analysis for representing rotations compactly.

    rotdata():
        Rotates a dataset by applying rotations around the specified axes (x, y, z), allowing the user to define the order of rotation.

Usage Notes:

    - The functions in this script are particularly useful for biomechanical analysis, especially when dealing with motion capture data where body segment orientations are calculated.
    - The `createortbase` and `createortbase_4points` functions are configured to handle anatomical points, making this toolkit essential for analyzing kinematic data from motion capture systems.

Changelog for Version 1.0:

    - Initial release with full support for orthonormal base creation, rotation matrix calculation, and conversion to Euler angles and quaternions.
    - Added flexible data rotation using configurable Euler angles and axes order.

License:

This script is distributed under the GPL3 License.
================================================================================
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation as R


# Define default configuration in def createortbase type 'C'
def createortbase(p1, p2, p3, configuration="C"):
    """
    Create orthonormal bases from arrays of points in space based on the specified configuration.

    Parameters:
    p1, p2, p3 (np.ndarray): Arrays representing the coordinates of points over time.
    configuration (str): The configuration to use for the basis ('A', 'B', 'C', 'D').

         (A)               (B)

         * P2              * P3
        / |                | \
    P3 *  |                |  * P2
        \ |                | /
        * P1               * P1


         (C)              (D)

    P3 *-----* P2          * P2
        \   /            /   \
        * P1         P3 *-----* P1

    Returns:
    np.ndarray: An array containing the orthonormal basis vectors for each time step.
    """

    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    if configuration == "A":
        v1 = (p1 - p3) / np.linalg.norm(p3 - p2, axis=1, keepdims=True)
        v2 = (p2 - p3) / np.linalg.norm(p3 - p2, axis=1, keepdims=True)
        v3_up = (p2 - p1) / np.linalg.norm(p2 - p1, axis=1, keepdims=True)
        v3_up /= np.linalg.norm(v3_up, axis=1, keepdims=True)
        v4_ap = np.cross(v2, v1)
        v4_ap /= np.linalg.norm(v4_ap, axis=1, keepdims=True)
        v5_ml = np.cross(v4_ap, v3_up)
        v5_ml /= np.linalg.norm(v5_ml, axis=1, keepdims=True)
        z_axis = np.cross(v5_ml, v4_ap)
        z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
        y_axis = np.cross(z_axis, v5_ml)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    elif configuration == "B":
        v1 = (p3 - p2) / np.linalg.norm(p3 - p2, axis=1, keepdims=True)
        v2 = (p1 - p2) / np.linalg.norm(p1 - p2, axis=1, keepdims=True)
        v3_up = (p3 - p1) / np.linalg.norm(p3 - p1, axis=1, keepdims=True)
        v3_up /= np.linalg.norm(v3_up, axis=1, keepdims=True)
        v4_ap = np.cross(v2, v1)
        v4_ap /= np.linalg.norm(v4_ap, axis=1, keepdims=True)
        v5_ml = np.cross(v4_ap, v3_up)
        v5_ml /= np.linalg.norm(v5_ml, axis=1, keepdims=True)
        z_axis = np.cross(v5_ml, v4_ap)
        z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
        y_axis = np.cross(z_axis, v5_ml)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    elif configuration == "C":
        v1 = (p2 - p1) / np.linalg.norm(p2 - p1, axis=1, keepdims=True)
        v2 = (p3 - p1) / np.linalg.norm(p3 - p1, axis=1, keepdims=True)
        v3_ml = (p2 - p3) / np.linalg.norm(p2 - p3, axis=1, keepdims=True)
        v4_ap = np.cross(v2, v1)
        v4_ap /= np.linalg.norm(v4_ap, axis=1, keepdims=True)
        z_axis = np.cross(v3_ml, v4_ap)
        z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
        x_axis = np.cross(v4_ap, z_axis)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
    elif configuration == "D":
        v1 = (p1 - p2) / np.linalg.norm(p1 - p2, axis=1, keepdims=True)
        v2 = (p3 - p2) / np.linalg.norm(p3 - p2, axis=1, keepdims=True)
        v3_ml = (p1 - p3) / np.linalg.norm(p1 - p3, axis=1, keepdims=True)
        v4_ap = np.cross(v2, v1)
        v4_ap /= np.linalg.norm(v4_ap, axis=1, keepdims=True)
        z_axis = np.cross(v3_ml, v4_ap)
        z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
        x_axis = np.cross(v4_ap, z_axis)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)

    pm = (p1 + p2 + p3) / 3
    localbase = np.stack((x_axis, y_axis, z_axis), axis=1)

    return localbase, pm


def createortbase_4points(p1, p2, p3, p4, configuration="y"):
    """
    Create an orthonormal base using four 3D points.

    Args:
        For Trunk:
        p1 (numpy.ndarray): STRN 3D coordinates of the first point. Shape (n, 3).
        p2 (numpy.ndarray): CLAV 3D coordinates of the second point. Shape (n, 3).
        p3 (numpy.ndarray): C7 3D coordinates of the third point. Shape (n, 3).
        p4 (numpy.ndarray): T10 3D coordinates of the fourth point. Shape (n, 3).
        configuration (str, optional): Configuration mode. Default is 'x'.

        For Pelvis:
        p1 (numpy.ndarray): RASI 3D coordinates of the first point. Shape (n, 3).
        p2 (numpy.ndarray): LASI 3D coordinates of the second point. Shape (n, 3).
        p3 (numpy.ndarray): RPSI 3D coordinates of the third point. Shape (n, 3).
        p4 (numpy.ndarray): LPSI 3D coordinates of the fourth point. Shape (n, 3).
        configuration (str, optional): Configuration mode. Default is 'z'.

    Returns:
        numpy.ndarray: Orthonormal base matrix and the mean point. Shape (n, 3, 3) and (n, 3).
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
 
    # Calculate the mean point
    pm = (p1 + p2 + p3 + p4) / 4

    if configuration == "x":
        # Trunk configuration
        v1 = (p2 - pm) / np.linalg.norm(
            p2 - pm, axis=1, keepdims=True
        )  # CLAV - PM normalized
        v2 = (p1 - pm) / np.linalg.norm(
            p1 - pm, axis=1, keepdims=True
        )  # STRN - PM normalized
        v3_ml = np.cross(v2, v1)  # Right ML direction
        v3_ml /= np.linalg.norm(v3_ml, axis=1, keepdims=True)
        pm_tprox = (p2 + p3) / 2
        v4_up = (pm_tprox - pm) / np.linalg.norm(
            pm_tprox - pm, axis=1, keepdims=True
        )  # UP direction
        y_axis = np.cross(v4_up, v3_ml)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
        z_axis = np.cross(v3_ml, y_axis)
        z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    elif configuration == "z":
        # Pelvis configuration
        v1 = (p2 - pm) / np.linalg.norm(
            p2 - pm, axis=1, keepdims=True
        )  # LASI - PM normalized
        v2 = (p1 - pm) / np.linalg.norm(
            p1 - pm, axis=1, keepdims=True
        )  # RASI - PM normalized
        v4_up = np.cross(v2, v1)  # UP direction
        v4_up /= np.linalg.norm(v4_up, axis=1, keepdims=True)
        pm_ant = (p1 + p2) / 2
        v5_ap = (pm_ant - pm) / np.linalg.norm(
            pm_ant - pm, axis=1, keepdims=True
        )  # AP direction
        x_axis = np.cross(v5_ap, v4_up)
        x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
        y_axis = np.cross(v4_up, x_axis)
        y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
    else:
        raise ValueError("Error: Configuration not implemented yet.")

    localbase = np.stack((x_axis, y_axis, z_axis), axis=1)
    return localbase, pm


def calcmatrot(base1, base2=None):
    """
    Calculate the rotation matrix from the base1 to the base2 (default is the canonical basis).

    Parameters:
    base1 (np.ndarray): An array containing the local base vectors. Shape can be (3, 3) for a single time step or (n, 3, 3) for multiple time steps.
    base2 (np.ndarray, optional): An array containing the target base vectors. Default is the canonical basis (3, 3).

    Returns:
    np.ndarray: An array containing the rotation matrices for each time step. Shape will be (3, 3) for a single time step or (n, 3, 3) for multiple time steps.
    """
    
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
 
    if base2 is None:
        base2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if base1.ndim == 2:  # Single time step case
        matrot = np.einsum("ij,jk->ik", base2, base1.T)
    elif base1.ndim == 3:  # Multiple time steps case
        matrot = np.einsum("nij,jk->nik", base1, base2.T)
    else:
        raise ValueError("base1 must be a 2D or 3D array")

    return matrot


def rotmat2euler(matrot):
    """
    Convert a rotation matrix to Euler angles in degrees.

    Parameters:
    matrot (np.ndarray): The rotation matrix (3x3).

    Returns:
    np.ndarray: The Euler angles (phi, theta, psi) in degrees.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
 
    rotation_object = R.from_matrix(matrot)
    euler_angles = rotation_object.as_euler("xyz", degrees=False)
    euler_angles_degrees = np.degrees(euler_angles)
    return euler_angles_degrees


def rotmat2quat(matrot):
    """
    Convert a rotation matrix to quaternions.

    Parameters:
    matrot (np.ndarray): The rotation matrix (3x3).

    Returns:
    np.ndarray: The quaternions (w, x, y, z).
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
 
    rotation_object = R.from_matrix(matrot)
    quaternions = rotation_object.as_quat()
    return quaternions


def rotdata(data, xth=0, yth=0, zth=0, ordem="xyz"):
    """
    Rotate the data based on the specified angles and order using scipy.spatial.transform.Rotation.

    Parameters:
    data (np.ndarray): The input data to be rotated, shape (3, n).
    xth, yth, zth (float): Rotation angles in degrees for x, y, and z axes.
    ordem (str): The order of rotation. Options are 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'.

    Returns:
    np.ndarray: The rotated data.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
 
    # Create the rotation object using Euler angles
    rotation_object = R.from_euler(ordem, [xth, yth, zth], degrees=True)

    # Apply the rotation to the data
    datrot = rotation_object.apply(data.T).T

    return datrot

