import numpy as np

from vaila.dlt2d import dlt2d
from vaila.dlt3d import calculate_dlt3d_params
from vaila.rec2d_one_dlt2d import rec2d
from vaila.rec3d import rec3d_multicam


def test_dlt2d_basic():
    # Define 4 real-world points (square)
    F = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    # Define corresponding pixel coordinates
    L = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
    params = dlt2d(F, L)
    assert len(params) == 8
    assert not np.isnan(params).any()


def test_rec2d_basic():
    # Simple DLT2D parameters (identity-like mapping for testing)
    # A = [L1, L2, L3, L4, L5, L6, L7, L8]
    # Simple case: x_real = (pixels_x - 100)/100, y_real = (pixels_y - 100)/100
    # u = (L1*X + L2*Y + L3) / (L7*X + L8*Y + 1)
    # v = (L4*X + L5*Y + L6) / (L7*X + L8*Y + 1)
    # If L7=0, L8=0:
    # u = L1*X + L2*Y + L3
    # v = L4*X + L5*Y + L6
    params = np.array([100, 0, 100, 0, 100, 100, 0, 0])
    pixel_coords = np.array([[150, 150]])
    real_coords = rec2d(params, pixel_coords)
    # 150 = 100*X + 0*Y + 100 => X = 0.5
    # 150 = 0*X + 100*Y + 100 => Y = 0.5
    assert np.allclose(real_coords, [[0.5, 0.5]])


def test_dlt3d_basic():
    # Define 6 points in 3D (minimum for 11 params is 6 non-coplanar points)
    ref_coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]])
    # Define arbitrary pixel coordinates
    pixel_coords = np.array(
        [[100, 100], [200, 100], [100, 200], [150, 150], [250, 250], [300, 300]]
    )
    params = calculate_dlt3d_params(pixel_coords, ref_coords)
    assert len(params) == 11
    assert not np.isnan(params).any()


def test_rec3d_multicam_basic():
    # Define parameters for two "cameras" (identity-like)
    # u = (L1*X + L2*Y + L3*Z + L4) / (L9*X + L10*Y + L11*Z + 1)
    # Cam 1: look from Z (X,Y plane)
    dlt1 = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    # Cam 2: look from X (Y,Z plane)
    dlt2 = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    dlt_list = [dlt1, dlt2]
    # Point at (5, 10, 15)
    # Cam 1 sees (5, 10)
    # Cam 2 sees (10, 15)
    pixel_list = [(5, 10), (10, 15)]

    point3d = rec3d_multicam(dlt_list, pixel_list)
    assert np.allclose(point3d, [5, 10, 15], atol=1e-5)
