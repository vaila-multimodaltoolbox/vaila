import numpy as np

from vaila.tugturn import (
    _parse_pose_connections,
    calculate_absolute_inclination_3d,
    calculate_angle_3d,
    canonical_phase_name,
    get_connection_color,
    phase_display_name,
    sample_frames,
)


def test_canonical_phase_name():
    assert canonical_phase_name("turn180") == "turn180"
    assert canonical_phase_name("first_gait") == "first_gait"
    assert canonical_phase_name(123) == "123"


def test_phase_display_name():
    assert phase_display_name("turn180") == "Turn180"
    assert phase_display_name("first_gait") == "First Gait"
    assert phase_display_name("stop_5s") == "Stop 5S"


def test_sample_frames():
    # Empty frames
    assert sample_frames([]) == []

    # Less than max_frames
    frames = [1, 2, 3, 4, 5]
    assert sample_frames(frames, max_frames=10) == [1, 2, 3, 4, 5]

    # More than max_frames
    frames = list(range(100))
    sampled = sample_frames(frames, max_frames=5)
    assert len(sampled) == 5
    assert sampled == [0, 24, 49, 74, 99]


def test_get_connection_color():
    # Right Points {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
    # Left Points {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
    assert get_connection_color(12, 14) == "red"
    assert get_connection_color(11, 13) == "blue"
    assert get_connection_color(0, 1) == "black"
    assert get_connection_color(11, 12) == "black"  # Cross connection


def test_calculate_angle_3d():
    # Simple 2D equivalent test on Y-axis
    # Hip at origin (0,0), Knee at (1,0), Ankle at (1, 1) -> 90 degrees
    # P1 (Hip), P2 (Knee - Vertex), P3 (Ankle)
    p1 = np.array([[0, 0, 0]])
    p2 = np.array([[1, 0, 0]])
    p3 = np.array([[1, 1, 0]])

    angle = calculate_angle_3d(p1, p2, p3)
    assert np.isclose(angle[0], 90.0)

    # Collinear -> 180 degrees
    p1 = np.array([[-1, 0, 0]])
    p2 = np.array([[0, 0, 0]])
    p3 = np.array([[1, 0, 0]])
    angle = calculate_angle_3d(p1, p2, p3)
    assert np.isclose(angle[0], 180.0)

    # 0 degrees (overlapping)
    p1 = np.array([[1, 0, 0]])
    p2 = np.array([[0, 0, 0]])
    p3 = np.array([[1, 0, 0]])
    angle = calculate_angle_3d(p1, p2, p3)
    assert np.isclose(angle[0], 0.0)


def test_calculate_absolute_inclination_3d():
    # Vertical segment -> 0 degrees
    p_top = np.array([[0, 0, 2]])
    p_bottom = np.array([[0, 0, 1]])
    vertical_vector = np.array([0, 0, 1])
    incl = calculate_absolute_inclination_3d(p_top, p_bottom, vertical_vector)
    assert np.isclose(incl[0], 0.0)

    # Horizontal segment -> 90 degrees
    p_top = np.array([[1, 0, 1]])
    p_bottom = np.array([[0, 0, 1]])
    incl = calculate_absolute_inclination_3d(p_top, p_bottom, vertical_vector)
    assert np.isclose(incl[0], 90.0)

    # Backward segment -> 180 degrees
    p_top = np.array([[0, 0, 0]])
    p_bottom = np.array([[0, 0, 1]])
    incl = calculate_absolute_inclination_3d(p_top, p_bottom, vertical_vector)
    assert np.isclose(incl[0], 180.0)


def test_parse_pose_connections():
    conns = [
        ["p1", "p2"],  # valid (0, 1)
        ["p12", "p13"],  # valid (11, 12)
        ["nose", "p3"],  # invalid pattern
        ["p34", "p1"],  # out of range (>32)
    ]
    parsed = _parse_pose_connections(conns)
    assert parsed == [(0, 1), (11, 12)]
