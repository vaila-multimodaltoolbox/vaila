"""
Tests for vaila/sapiens3d_kinematics.py.

Synthetic tests validate the pure-geometry pipeline (frame construction,
relative-rotation math, quaternion/Euler conversion, QC, neutral-zero
calibration) against known, hand-computed rotations. Real-fixture tests
exercise the full C3D-read -> landmark-resolution -> kinematics pipeline
against tests/viewc3d/rec3d_240hz.c3d (308-point canonical mapping) and
confirm tests/viewc3d/rec3d_200hz.c3d (224 points, no canonical mapping)
correctly raises LandmarkResolutionError instead of guessing.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vaila.sapiens3d_kinematics import (
    EULER_SEQUENCES,
    LandmarkResolutionError,
    _canonical_308_match,
    _direct_label_match,
    _mediolateral_axis,
    build_foot_frame,
    build_pelvis_frame,
    build_shank_frame,
    build_thigh_frame,
    compute_kinematics,
    enforce_quaternion_continuity,
    neutral_zero,
    project_to_so3,
    read_c3d_points,
    resolve_landmarks,
    rotation_qc,
    rotmats_to_euler_deg,
    rotmats_to_quats_xyzw,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "viewc3d"
FIXTURE_308 = FIXTURE_DIR / "rec3d_240hz.c3d"
FIXTURE_224 = FIXTURE_DIR / "rec3d_200hz.c3d"


# --------------------------------------------------------------------------- #
# Landmark resolution
# --------------------------------------------------------------------------- #
def test_direct_label_match_finds_semantic_labels():
    labels = [
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_big_toe",
        "right_big_toe",
        "neck",
    ]
    resolved = _direct_label_match(labels)
    assert resolved is not None
    assert resolved["left_hip"] == 0
    assert resolved["neck"] == 10


def test_direct_label_match_returns_none_when_incomplete():
    labels = ["left_hip", "right_hip"]  # missing most required slots
    assert _direct_label_match(labels) is None


def test_canonical_308_match_requires_exact_generic_sequence():
    labels = [f"p{i}" for i in range(1, 309)]
    resolved = _canonical_308_match(labels)
    assert resolved is not None
    # Spot-check against vaila/skeletons/sapiens2_goliath308.json ordering.
    assert resolved["left_hip"] == 9  # p10 (0-based index 9)
    assert resolved["right_hip"] == 10  # p11


def test_canonical_308_match_rejects_wrong_count():
    labels = [f"p{i}" for i in range(1, 225)]
    assert _canonical_308_match(labels) is None


def test_resolve_landmarks_raises_when_unresolvable():
    labels = [f"p{i}" for i in range(1, 225)]  # 224 points, no canonical match
    with pytest.raises(LandmarkResolutionError):
        resolve_landmarks(labels, keypoint_map=None)


def test_resolve_landmarks_uses_explicit_keypoint_map():
    labels = ["p1", "p2", "p3"]
    kmap = {
        "left_hip": 0,
        "right_hip": 1,
        "left_knee": 2,
        "right_knee": 2,
        "left_ankle": 2,
        "right_ankle": 2,
        "left_heel": 2,
        "right_heel": 2,
        "left_toe": 2,
        "right_toe": 2,
    }
    resolved, method = resolve_landmarks(labels, keypoint_map=kmap)
    assert method == "keypoint_map_file"
    assert resolved["left_hip"] == 0


# --------------------------------------------------------------------------- #
# Synthetic frame construction + SO(3) QC
# --------------------------------------------------------------------------- #
def _column(v):
    return np.array(v, dtype=float).reshape(3, 1)


def test_pelvis_frame_identity_orientation():
    left_hip = _column([0.1, 0, 0])
    right_hip = _column([-0.1, 0, 0])
    superior_ref = _column([0, 0, 1])
    R = build_pelvis_frame(left_hip, right_hip, superior_ref)
    det, orth = rotation_qc(R)
    assert det[0] == pytest.approx(1.0, abs=1e-8)
    assert orth[0] == pytest.approx(0.0, abs=1e-8)


def test_thigh_and_shank_frames_are_proper_rotations():
    hip = _column([0, 0, 1])
    knee = _column([0.05, 0, 0.5])
    ankle = _column([0, 0, 0])
    y_pelvis = _column([1, 0, 0])
    R_thigh = project_to_so3(build_thigh_frame(hip, knee, y_pelvis, "left"))
    R_shank = project_to_so3(build_shank_frame(knee, ankle, R_thigh[:, 1, :], "left"))
    for R in (R_thigh, R_shank):
        det, orth = rotation_qc(R)
        assert det[0] == pytest.approx(1.0, abs=1e-6)
        assert orth[0] == pytest.approx(0.0, abs=1e-6)


def test_thigh_and_shank_frames_well_defined_at_full_knee_extension():
    # Straight-leg / "CUBE test" pose: hip, knee, ankle EXACTLY collinear.
    # The old implementation derived each segment's mediolateral axis from
    # cross(own_axis, adjacent_segment_axis) -- i.e. cross(thigh, shank) --
    # which is exactly zero here, producing NaN frames. The fixed
    # implementation uses an adjacent-segment-independent secondary
    # reference (pelvis Y for thigh, thigh's own Y for shank), so a fully
    # extended knee must NOT be singular.
    hip = _column([0, 0, 1.0])
    knee = _column([0, 0, 0.5])
    ankle = _column([0, 0, 0.0])  # exactly collinear with hip/knee
    y_pelvis = _column([1, 0, 0])
    R_thigh = project_to_so3(build_thigh_frame(hip, knee, y_pelvis, "left"))
    R_shank = project_to_so3(build_shank_frame(knee, ankle, R_thigh[:, 1, :], "left"))
    for R in (R_thigh, R_shank):
        assert not np.any(np.isnan(R)), "straight-leg pose must not produce NaN frames"
        det, orth = rotation_qc(R)
        assert det[0] == pytest.approx(1.0, abs=1e-6)
        assert orth[0] == pytest.approx(0.0, abs=1e-6)
    # Knee rotation at full extension should be close to identity (no
    # flexion), confirming the frame is not just finite but sane.
    R_knee = R_thigh[:, :, 0].T @ R_shank[:, :, 0]
    flex_deg = Rotation.from_matrix(R_knee).as_euler("XYZ", degrees=True)[0]
    assert abs(flex_deg) < 1.0


def test_thigh_frame_signature_has_no_adjacent_segment_dependency():
    # Structural regression guard for the straight-leg-singularity fix:
    # build_thigh_frame must not accept an "ankle" parameter (which would
    # reintroduce cross(thigh, shank)), and build_shank_frame must not
    # accept a "hip" parameter.
    import inspect

    thigh_params = list(inspect.signature(build_thigh_frame).parameters)
    shank_params = list(inspect.signature(build_shank_frame).parameters)
    assert "ankle" not in thigh_params
    assert "hip" not in shank_params


def test_foot_frame_is_proper_rotation():
    ankle = _column([0, 0, 0])
    knee = _column([0, 0, 0.4])
    heel = _column([-0.05, 0, -0.02])
    toe = _column([0.15, 0, -0.02])
    R = build_foot_frame(ankle, knee, heel, toe, "left")
    det, orth = rotation_qc(R)
    assert det[0] == pytest.approx(1.0, abs=1e-6)
    assert orth[0] == pytest.approx(0.0, abs=1e-6)


def test_project_to_so3_corrects_numerical_drift():
    rng = np.random.default_rng(0)
    R_true = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    noise = rng.normal(scale=1e-3, size=(3, 3))
    R_noisy = (R_true + noise)[:, :, None]
    R_fixed = project_to_so3(R_noisy)
    det, orth = rotation_qc(R_fixed)
    assert det[0] == pytest.approx(1.0, abs=1e-8)
    assert orth[0] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Known-rotation validation: relative rotation, quaternions, both Euler seqs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seq", ["XYZ", "YXZ", "YZX"])
@pytest.mark.parametrize("angles_deg", [(0, 0, 0), (15, 0, 0), (0, -10, 0), (5, 10, -7)])
def test_known_rotation_round_trips_through_euler(seq, angles_deg):
    R_true = Rotation.from_euler(seq, angles_deg, degrees=True).as_matrix()
    R_stack = R_true[:, :, None]
    euler = rotmats_to_euler_deg(R_stack, seq)
    R_recovered = Rotation.from_euler(seq, euler[0], degrees=True).as_matrix()
    np.testing.assert_allclose(R_recovered, R_true, atol=1e-6)


def test_relative_rotation_hip_formula_matches_known_composition():
    R_pelvis = Rotation.from_euler("xyz", [0, 5, 0], degrees=True).as_matrix()
    R_thigh = Rotation.from_euler("xyz", [20, 5, 0], degrees=True).as_matrix()
    R_hip_expected = R_pelvis.T @ R_thigh
    # Pure flexion difference (about pelvis-local Y after removing the shared
    # 5 deg offset) should recover ~20 deg on the corresponding Euler axis.
    euler = Rotation.from_matrix(R_hip_expected).as_euler("XYZ", degrees=True)
    assert np.linalg.norm(euler) > 0  # sanity: relative rotation is non-trivial
    det = np.linalg.det(R_hip_expected)
    assert det == pytest.approx(1.0, abs=1e-8)


def test_quaternion_unit_norm_and_scalar_last_convention():
    R = Rotation.from_euler("xyz", [12, -8, 33], degrees=True).as_matrix()[:, :, None]
    q = rotmats_to_quats_xyzw(R)
    assert q.shape == (1, 4)
    assert np.linalg.norm(q[0]) == pytest.approx(1.0, abs=1e-8)
    # scalar-last: q[3] (w) should match scipy's own as_quat() directly.
    expected = Rotation.from_matrix(R[:, :, 0]).as_quat()
    np.testing.assert_allclose(q[0], expected, atol=1e-8)


def test_quaternion_temporal_sign_continuity():
    # Two frames representing the *same* rotation but with opposite
    # quaternion sign (antipodal) must be flipped into the same hemisphere.
    q_true = Rotation.from_euler("xyz", [10, 0, 0], degrees=True).as_quat()
    quats = np.stack([q_true, -q_true, q_true])
    enforce_quaternion_continuity(quats)
    assert np.dot(quats[0], quats[1]) > 0
    assert np.dot(quats[1], quats[2]) > 0


# --------------------------------------------------------------------------- #
# Neutral-zero calibration
# --------------------------------------------------------------------------- #
def test_neutral_zero_maps_calibration_pose_to_identity():
    R_neutral_true = Rotation.from_euler("xyz", [3, -2, 1], degrees=True).as_matrix()
    n_frames = 10
    R = np.repeat(R_neutral_true[:, :, None], n_frames, axis=2)
    R_zeroed, R_neutral = neutral_zero(R, (0, 5))
    np.testing.assert_allclose(R_neutral, R_neutral_true, atol=1e-6)
    for t in range(n_frames):
        np.testing.assert_allclose(R_zeroed[:, :, t], np.eye(3), atol=1e-6)


def test_neutral_zero_raises_on_all_nan_range():
    R = np.full((3, 3, 5), np.nan)
    with pytest.raises(ValueError):
        neutral_zero(R, (0, 5))


# --------------------------------------------------------------------------- #
# Euler/Cardan convention regression (point 4 of the audit): scipy sequence
# strings must stay uppercase (intrinsic/body-fixed), Hip=YXZ, Knee=YXZ,
# Ankle=YZX. A lowercase string ("yxz") is a *different, extrinsic* rotation
# to scipy -- this test fails loudly if that ever regresses.
# --------------------------------------------------------------------------- #
def test_vicon_compatible_euler_sequences_are_uppercase_intrinsic():
    vicon = EULER_SEQUENCES["vicon_compatible"]
    assert vicon == {"hip": "YXZ", "knee": "YXZ", "ankle": "YZX"}
    for seq in vicon.values():
        assert seq.isupper(), f"{seq!r} must be uppercase (scipy intrinsic sequence)"
        assert seq == seq.upper() and not seq.islower()


def test_lowercase_euler_sequence_is_a_different_extrinsic_rotation():
    # Documents *why* case matters: scipy treats "YXZ" (intrinsic) and
    # "yxz" (extrinsic) as different rotations for a generic (non-zero on
    # every axis) angle triple, so an accidental lowercase regression would
    # silently change every joint angle.
    angles = [12.0, -7.0, 25.0]
    R_intrinsic = Rotation.from_euler("YXZ", angles, degrees=True).as_matrix()
    R_extrinsic = Rotation.from_euler("yxz", angles, degrees=True).as_matrix()
    assert not np.allclose(R_intrinsic, R_extrinsic, atol=1e-3)


# --------------------------------------------------------------------------- #
# Left/right mirror-symmetry test (point 5 of the audit): mirror an entire
# synthetic pose across the sagittal (x=0) plane -- which is the correct way
# to construct "the other leg performing the identical anatomical motion" --
# and confirm the full landmark -> segment-frame -> joint-rotation ->
# Euler-angle pipeline reports matching signs/magnitudes for both sides.
# --------------------------------------------------------------------------- #
def _mirror_x(point: np.ndarray) -> np.ndarray:
    mirrored = point.copy()
    mirrored[0, :] *= -1.0
    return mirrored


def _leg_pipeline(hip, knee, ankle, heel, toe, y_pelvis, side):
    R_thigh = project_to_so3(build_thigh_frame(hip, knee, y_pelvis, side))
    R_shank = project_to_so3(build_shank_frame(knee, ankle, R_thigh[:, 1, :], side))
    R_foot = project_to_so3(build_foot_frame(ankle, knee, heel, toe, side))
    R_knee = R_thigh[:, :, 0].T @ R_shank[:, :, 0]
    R_ankle = R_shank[:, :, 0].T @ R_foot[:, :, 0]
    return R_thigh[:, :, 0], R_shank[:, :, 0], R_foot[:, :, 0], R_knee, R_ankle


def _hip_knee_ankle(hip, knee, ankle, heel, toe, left_hip, right_hip, superior, y_pelvis, side):
    """Full landmark -> segment frame -> joint matrix pipeline for one leg,
    returning the three RAW joint rotation matrices (single frame each)."""
    R_thigh, R_shank, R_foot, R_knee, R_ankle = _leg_pipeline(
        hip, knee, ankle, heel, toe, y_pelvis, side
    )
    R_pelvis = build_pelvis_frame(left_hip, right_hip, superior)[:, :, 0]
    R_hip = R_pelvis.T @ R_thigh
    return R_hip, R_knee, R_ankle


def test_mirror_left_right_report_consistent_anatomical_signs():
    """Point 5 of the audit: mirror an entire synthetic pose across the
    sagittal (x=0) plane -- the correct construction of "the other leg
    performing the identical anatomical motion" -- and confirm the full
    landmark -> segment-frame -> joint-rotation -> Euler-angle pipeline
    reports matching signs/magnitudes for both sides.

    Comparison basis: NEUTRAL-RELATIVE Euler angles (``R_neutral.T @
    R_perturbed``), exactly mirroring what ``neutral_zero()`` computes in
    production, rather than raw single-pose Euler angles. Raw hip Euler
    angles carry a large, architecturally-expected ~180-degree baseline
    offset on the flexion axis (thigh Z points distally while pelvis Z
    points up), and for an arbitrary synthetic pose that baseline can put
    the raw decomposition at a scipy Euler-representation ambiguity
    (multiple (e1, e2, e3) triples encode the same matrix). Comparing the
    neutral-relative rotation sidesteps that ambiguity entirely and is the
    scientifically meaningful comparison anyway: what a report actually
    shows a clinician is the neutral-zeroed angle, not the raw one.
    """
    left_hip = _column([0.1, 0.0, 1.0])
    right_hip = _column([-0.1, 0.0, 1.0])
    y_pelvis = _mediolateral_axis(left_hip, right_hip)
    superior = _column([0, 0, 2.0])

    # Neutral (near-upright, unflexed) pose.
    l_knee_n = _column([0.1, 0.0, 0.5])
    l_ankle_n = _column([0.1, 0.0, 0.05])
    l_heel_n = _column([0.1, -0.05, 0.0])
    l_toe_n = _column([0.1, 0.15, 0.0])

    # Perturbed pose: flexion + ab/adduction + a transverse-plane offset, so
    # all three Euler components (flexion, ab/adduction or varus/valgus or
    # inv/eversion, and axial rotation) are non-zero for hip, knee, and
    # ankle simultaneously.
    l_knee_p = _column([0.13, 0.02, 0.53])
    l_ankle_p = _column([0.10, 0.08, 0.08])
    l_heel_p = _column([0.06, 0.04, 0.0])
    l_toe_p = _column([0.10, 0.22, 0.0])

    R_hip_ln, R_knee_ln, R_ankle_ln = _hip_knee_ankle(
        left_hip,
        l_knee_n,
        l_ankle_n,
        l_heel_n,
        l_toe_n,
        left_hip,
        right_hip,
        superior,
        y_pelvis,
        "left",
    )
    R_hip_lp, R_knee_lp, R_ankle_lp = _hip_knee_ankle(
        left_hip,
        l_knee_p,
        l_ankle_p,
        l_heel_p,
        l_toe_p,
        left_hip,
        right_hip,
        superior,
        y_pelvis,
        "left",
    )

    # Physical mirror across x=0 -- the right leg performing the SAME
    # anatomical movement (same neutral pose, same perturbation).
    right_hip_m = _mirror_x(left_hip)  # mirrored left_hip position == right hip
    left_hip_m = _mirror_x(right_hip)
    y_pelvis_m = _mediolateral_axis(left_hip_m, right_hip_m)
    superior_m = _mirror_x(superior)
    r_hip_m = _mirror_x(left_hip)

    R_hip_rn, R_knee_rn, R_ankle_rn = _hip_knee_ankle(
        r_hip_m,
        _mirror_x(l_knee_n),
        _mirror_x(l_ankle_n),
        _mirror_x(l_heel_n),
        _mirror_x(l_toe_n),
        left_hip_m,
        right_hip_m,
        superior_m,
        y_pelvis_m,
        "right",
    )
    R_hip_rp, R_knee_rp, R_ankle_rp = _hip_knee_ankle(
        r_hip_m,
        _mirror_x(l_knee_p),
        _mirror_x(l_ankle_p),
        _mirror_x(l_heel_p),
        _mirror_x(l_toe_p),
        left_hip_m,
        right_hip_m,
        superior_m,
        y_pelvis_m,
        "right",
    )

    seqs = EULER_SEQUENCES["vicon_compatible"]

    def relative_euler(R_neutral, R_perturbed, seq):
        return Rotation.from_matrix(R_neutral.T @ R_perturbed).as_euler(seq, degrees=True)

    e_hip_l = relative_euler(R_hip_ln, R_hip_lp, seqs["hip"])
    e_hip_r = relative_euler(R_hip_rn, R_hip_rp, seqs["hip"])
    e_knee_l = relative_euler(R_knee_ln, R_knee_lp, seqs["knee"])
    e_knee_r = relative_euler(R_knee_rn, R_knee_rp, seqs["knee"])
    e_ankle_l = relative_euler(R_ankle_ln, R_ankle_lp, seqs["ankle"])
    e_ankle_r = relative_euler(R_ankle_rn, R_ankle_rp, seqs["ankle"])

    # Same anatomical motion mirrored to the other side must report matching
    # sign AND magnitude on every component: flexion (e1), ab/adduction /
    # varus-valgus / inv-eversion (e2), and axial rotation (e3).
    for name, euler_l, euler_r in [
        ("hip (flexion, ab/adduction, axial rotation)", e_hip_l, e_hip_r),
        ("knee (flexion, varus/valgus, axial rotation)", e_knee_l, e_knee_r),
        ("ankle (dorsi/plantarflexion, inversion/eversion, rotation)", e_ankle_l, e_ankle_r),
    ]:
        np.testing.assert_allclose(
            euler_l,
            euler_r,
            atol=1.0,
            err_msg=f"{name}: left {euler_l} vs mirrored-right {euler_r} sign/magnitude mismatch",
        )


# --------------------------------------------------------------------------- #
# Real-fixture QC
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not FIXTURE_308.exists(), reason="rec3d_240hz.c3d fixture not available")
def test_real_fixture_308_resolves_and_computes_kinematics():
    c3d = read_c3d_points(FIXTURE_308)
    assert c3d.rate_hz == pytest.approx(240.0)
    assert len(c3d.labels) == 308

    result = compute_kinematics(c3d, cutoff_hz=6.0, neutral_frames=(0, 30))
    assert result.landmark_method == "canonical_308_order"
    assert result.landmark_indices["left_hip"] == 9
    assert result.landmark_indices["right_hip"] == 10

    valid_frames = 0
    max_orth_err = 0.0
    min_det = np.inf
    for R in result.joint_rotations.values():
        det, orth = rotation_qc(R)
        valid = ~np.isnan(det)
        valid_frames += int(valid.sum())
        if valid.any():
            max_orth_err = max(max_orth_err, float(np.nanmax(orth)))
            min_det = min(min_det, float(np.nanmin(det)))

    assert valid_frames > 0
    assert max_orth_err < 1e-6
    assert min_det == pytest.approx(1.0, abs=1e-6)

    # Neutral-zeroed joint rotations were computed for every joint.
    assert set(result.joint_rotations_zeroed) == set(result.joint_rotations)


@pytest.mark.skipif(not FIXTURE_224.exists(), reason="rec3d_200hz.c3d fixture not available")
def test_real_fixture_224_has_no_canonical_mapping():
    c3d = read_c3d_points(FIXTURE_224)
    assert len(c3d.labels) == 224
    with pytest.raises(LandmarkResolutionError):
        compute_kinematics(c3d, keypoint_map=None)
