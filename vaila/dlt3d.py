"""
Script Name: dlt3d.py
Version: v0.01
Date and Time: 2024-08-13
Author: Prof. PhD. Paulo Santiago
Email: vailamultimodaltoolbox@gmail.com
Description: This script performs 3D calibration using the Direct Linear 
             Transformation (DLT) method. It calibrates based on correspondences 
             between 2D coordinates in multiple views and their corresponding 
             3D coordinates.

Dependencies:
    - Python 3.11.8
    - numpy
    - pandas
    - scipy
    - (add other relevant libraries)

Instructions:
    1. Ensure that the required libraries are installed in your Anaconda environment.
    2. Modify the input variables as needed to include the 2D and 3D coordinates.
    3. Run the script to obtain the 3D calibration parameters.
    
    Example usage:
        $ python dlt3d.py
"""


import os
import numpy as np
import pandas as pd
import csv
from numpy.linalg import inv
from tkinter import filedialog, Tk, messagebox
from rich import print

