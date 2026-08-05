"""Real-data regression test for rec3d_one_dlt3d.py's mesh-for-Blender export.

Runs the full run_reconstruction() pipeline (DLT triangulation + Umeyama mesh
alignment) against the real fixture at
/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo/ — a 3-camera COD
(change-of-direction) trial with an existing SAM3+DINOv3 run, one selected
person per camera already produced by sam3dinov3_visualize.py.

Skipped entirely (not failed) when the fixture is absent, e.g. on CI or any
machine other than the one this dataset lives on — this tier is the
real-data regression floor, not something every environment is expected to
reproduce. The synthetic tier in test_rec3d_mesh_alignment.py always runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from vaila.mesh_alignment import read_obj_vertices
    from vaila.rec3d_one_dlt3d import find_markers_csv_in_dir, run_reconstruction
except ImportError:
    from mesh_alignment import read_obj_vertices  # ty: ignore[unresolved-import]
    from rec3d_one_dlt3d import (  # ty: ignore[unresolved-import]
        find_markers_csv_in_dir,
        run_reconstruction,
    )

FIXTURE_ROOT = Path("/home/preto/data/sep_runcod_01072026/REC3D_COD/rec3d_todo")
VISUALIZED_ROOT = FIXTURE_ROOT / "sam3dinov3_one_person_visualized"

DLT_FILES = [
    FIXTURE_ROOT / "c1_cod_markers_1_line.dlt3d",
    FIXTURE_ROOT / "c2_cod_markers_1_line.dlt3d",
    FIXTURE_ROOT / "c3_cod_markers_1_line.dlt3d",
]
MESH_SOURCE_DIRS = [
    VISUALIZED_ROOT / "c1_cod_sam3dinov3_visualized_id_04",
    VISUALIZED_ROOT / "c2_cod_sam3dinov3_visualized_id_08",
    VISUALIZED_ROOT / "c3_cod_sam3dinov3_visualized_id_03",
]
PIXEL_FILES = [
    MESH_SOURCE_DIRS[0] / "c1_cod_id_04_markers.csv",
    MESH_SOURCE_DIRS[1] / "c2_cod_id_08_markers.csv",
    MESH_SOURCE_DIRS[2] / "c3_cod_id_03_markers.csv",
]

_FIXTURE_AVAILABLE = (
    all(p.is_file() for p in DLT_FILES)
    and all(p.is_file() for p in PIXEL_FILES)
    and all(d.is_dir() for d in MESH_SOURCE_DIRS)
)

pytestmark = pytest.mark.skipif(
    not _FIXTURE_AVAILABLE,
    reason=f"real rec3d_todo fixture not available at {FIXTURE_ROOT}",
)

# Segment-length centers independently validated on this exact dataset
# (project memory, 2026-08-01: thigh 0.387+/-0.017 m, shank 0.371+/-0.012 m,
# shoulder width ~0.360 m), reused here as a regression floor rather than
# recomputed. That prior validation measured each camera's own monocular
# mesh reprojection; this test measures the multi-camera DLT-triangulated
# skeleton instead — a different, arguably more direct pipeline — so a wider
# +/-25% plausibility band is used instead of the original tight sigma (a
# real run measured shank at 0.409 m, ~10% above the documented 0.371 m
# center, which is a reasonable cross-pipeline difference, not a defect).
THIGH_RANGE_M = (0.387 * 0.75, 0.387 * 1.25)
SHANK_RANGE_M = (0.371 * 0.75, 0.371 * 1.25)
SHOULDER_WIDTH_RANGE_M = (0.360 * 0.75, 0.360 * 1.25)

# MHR70 marker indices (1-based, p{i}) used below — see mesh_alignment.py's
# ALIGNMENT_MARKER_SPEC for the full name mapping.
P_LEFT_SHOULDER, P_RIGHT_SHOULDER = 6, 7
P_LEFT_HIP, P_RIGHT_HIP = 10, 11
P_LEFT_KNEE, P_RIGHT_KNEE = 12, 13
P_LEFT_ANKLE, P_RIGHT_ANKLE = 14, 15


def _segment_length(df, p_a, p_b):
    dx = df[f"p{p_a}_x"] - df[f"p{p_b}_x"]
    dy = df[f"p{p_a}_y"] - df[f"p{p_b}_y"]
    dz = df[f"p{p_a}_z"] - df[f"p{p_b}_z"]
    return np.sqrt(dx**2 + dy**2 + dz**2)


def test_find_markers_csv_in_dir_derives_pixel_files_from_mesh_source_dirs():
    """The simplified CLI/GUI path (2026-08-04): --pixels can be omitted when
    --mesh-source-dir is given, since each Visualize-ID run directory already
    contains its own *_markers.csv. Locks in that each of the three real
    fixture directories resolves to exactly the expected file."""
    expected_names = [
        "c1_cod_id_04_markers.csv",
        "c2_cod_id_08_markers.csv",
        "c3_cod_id_03_markers.csv",
    ]
    for mesh_dir, expected_name in zip(MESH_SOURCE_DIRS, expected_names, strict=True):
        found = find_markers_csv_in_dir(mesh_dir)
        assert found is not None
        assert found.name == expected_name
        assert found == PIXEL_FILES[MESH_SOURCE_DIRS.index(mesh_dir)]


def test_find_markers_csv_in_dir_returns_none_for_directory_without_markers(tmp_path):
    assert find_markers_csv_in_dir(tmp_path) is None


@pytest.fixture(scope="module")
def mesh_run_result(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("rec3d_mesh_export")
    result = run_reconstruction(
        [str(p) for p in DLT_FILES],
        [str(p) for p in PIXEL_FILES],
        str(output_dir),
        point_rate=119.88012001,
        gui=False,
        swap_yz=True,
        skeleton_json_path=None,
        mesh_source_dirs=[str(d) for d in MESH_SOURCE_DIRS],
        export_mesh="obj",
    )
    assert result is not None, "run_reconstruction returned None on the real fixture"
    new_dir, file_base = result
    return Path(new_dir), file_base


@pytest.fixture(scope="module")
def rec3d_df(mesh_run_result):
    new_dir, file_base = mesh_run_result
    return pd.read_csv(new_dir / f"{file_base}.csv")


@pytest.fixture(scope="module")
def manifest_df(mesh_run_result):
    new_dir, file_base = mesh_run_result
    manifest_path = new_dir / f"{file_base}_mesh_alignment.csv"
    assert manifest_path.is_file(), f"mesh alignment manifest not written: {manifest_path}"
    return pd.read_csv(manifest_path)


def test_triangulated_thigh_length_within_validated_range(rec3d_df):
    left = _segment_length(rec3d_df, P_LEFT_HIP, P_LEFT_KNEE).dropna()
    right = _segment_length(rec3d_df, P_RIGHT_HIP, P_RIGHT_KNEE).dropna()
    assert len(left) > 100 and len(right) > 100
    mean_thigh = pd.concat([left, right]).mean()
    assert THIGH_RANGE_M[0] <= mean_thigh <= THIGH_RANGE_M[1], (
        f"mean thigh length {mean_thigh:.4f} m outside validated range {THIGH_RANGE_M}"
    )


def test_triangulated_shank_length_within_validated_range(rec3d_df):
    left = _segment_length(rec3d_df, P_LEFT_KNEE, P_LEFT_ANKLE).dropna()
    right = _segment_length(rec3d_df, P_RIGHT_KNEE, P_RIGHT_ANKLE).dropna()
    assert len(left) > 100 and len(right) > 100
    mean_shank = pd.concat([left, right]).mean()
    assert SHANK_RANGE_M[0] <= mean_shank <= SHANK_RANGE_M[1], (
        f"mean shank length {mean_shank:.4f} m outside validated range {SHANK_RANGE_M}"
    )


def test_triangulated_shoulder_width_within_validated_range(rec3d_df):
    width = _segment_length(rec3d_df, P_LEFT_SHOULDER, P_RIGHT_SHOULDER).dropna()
    assert len(width) > 100
    mean_width = width.mean()
    assert SHOULDER_WIDTH_RANGE_M[0] <= mean_width <= SHOULDER_WIDTH_RANGE_M[1], (
        f"mean shoulder width {mean_width:.4f} m outside plausible range {SHOULDER_WIDTH_RANGE_M}"
    )


def test_mesh_alignment_manifest_has_most_frames(manifest_df, rec3d_df):
    total_frames = len(rec3d_df)
    written = len(manifest_df)
    assert written > 0.7 * total_frames, (
        f"only {written}/{total_frames} frames produced an aligned mesh "
        f"(expected >70% for a 3-camera trial with minimal occlusion)"
    )


def test_mesh_alignment_residuals_are_recorded_and_finite(manifest_df):
    residuals = manifest_df["mean_residual_m"].to_numpy()
    assert np.isfinite(residuals).all()
    assert (residuals >= 0).all()
    median = float(np.median(residuals))
    p95 = float(np.percentile(residuals, 95))
    print(
        f"\nresidual (m): min={residuals.min():.4f} median={median:.4f} "
        f"mean={residuals.mean():.4f} max={residuals.max():.4f} p95={p95:.4f}"
    )
    # Frozen 2026-08-04 from a real run on this fixture: min=0.0041 m,
    # median=0.0141 m, mean=0.0147 m, max=0.0332 m, p95=0.0239 m (631/631
    # frames, 79 camera switches). Thresholds below carry ~2x headroom over
    # those observed values — tight enough to catch a broken transform
    # (which would land in the tens of centimeters, not millimeters), loose
    # enough to tolerate normal fixture-to-fixture variation. Only widen
    # these with a new observed distribution + explicit sign-off, per
    # rec3d-mesh-blender-loop.md's human-approval gate.
    assert median < 0.03, f"median residual {median:.4f} m exceeds frozen threshold 0.03 m"
    assert p95 < 0.05, f"p95 residual {p95:.4f} m exceeds frozen threshold 0.05 m"
    assert residuals.max() < 0.08, (
        f"max residual {residuals.max():.4f} m exceeds frozen threshold 0.08 m"
    )


def test_exported_mesh_vertex_and_face_counts_match_source(mesh_run_result, manifest_df):
    new_dir, _file_base = mesh_run_result
    sample = manifest_df.iloc[len(manifest_df) // 2]
    frame = int(sample["frame"])
    camera_index = int(sample["camera_index"])

    source_dir = MESH_SOURCE_DIRS[camera_index]
    source_obj = source_dir / "meshes_obj" / f"frame_{frame:06d}.obj"
    output_obj = new_dir / "meshes_obj" / f"frame_{frame:06d}.obj"
    assert source_obj.is_file()
    assert output_obj.is_file()

    source_vertices = read_obj_vertices(source_obj)
    output_vertices = read_obj_vertices(output_obj)
    assert output_vertices.shape == source_vertices.shape

    source_faces = source_obj.read_text().count("\nf ")
    output_faces = output_obj.read_text().count("\nf ")
    assert output_faces == source_faces
    assert output_faces == 36874  # fixed MHR mesh topology for this dataset


def test_mesh_vertices_stay_raw_regardless_of_swap_yz(mesh_run_result, manifest_df):
    """Regression test for a real bug found TWICE on this fixture, each time
    fixed the opposite way:

    2026-08-04: reconstruct_mesh_sequence() ignored swap_yz entirely and
    always wrote mesh vertices in the raw (unswapped) DLT frame, while
    save_rec3d_as_bvh() swapped Y/Z for Blender's Z-up convention -- so the
    BVH and the OBJ mesh ended up in two different axis conventions.

    2026-08-05: the first fix (mesh follows swap_yz, matching the BVH file's
    OWN raw convention byte for byte) was itself wrong for how meshes are
    actually viewed in Blender. Blender's own BVH importer DOES apply an
    axis conversion by default, but the bundled "Stop Motion OBJ" family of
    mesh-sequence add-ons does not (confirmed from its source:
    stop_motion_obj2/core.py's parse_obj()/apply_to_mesh() assign "v x y z"
    straight to mesh.vertices, no conversion, no axis parameter to override).
    So a swapped mesh file landed correctly only through Blender's native
    `wm.obj_import` (which also converts) and was reported as "the mesh has
    Z where Y should be" through the add-on almost everyone actually uses
    for playback.

    Mesh vertices must therefore stay in the RAW (unswapped) DLT/world frame
    UNCONDITIONALLY -- the same convention as the triangulated skeleton CSV,
    which is also where the BVH ends up once Blender's BVH importer converts
    it. Column 2 (DLT Z/height, per this fixture's calibration volume) must
    stay within a plausible human-height band regardless of swap_yz, while
    column 1 (DLT Y/horizontal) ranges far wider across the run.
    """
    new_dir, _file_base = mesh_run_result
    sample_frames = sorted(manifest_df["frame"].to_numpy())[::50]
    assert len(sample_frames) > 5

    col1_means = []
    col2_means = []
    for frame in sample_frames:
        obj_path = new_dir / "meshes_obj" / f"frame_{int(frame):06d}.obj"
        vertices = read_obj_vertices(obj_path)
        col1_means.append(vertices[:, 1].mean())
        col2_means.append(vertices[:, 2].mean())

    col1_means = np.array(col1_means)
    col2_means = np.array(col2_means)

    assert col2_means.min() > -1.0 and col2_means.max() < 3.0, (
        f"mesh column 2 (DLT Z/height) out of plausible range "
        f"{col2_means.min():.3f} to {col2_means.max():.3f} m -- looks swapped"
    )
    assert col1_means.max() - col1_means.min() > 0.3, (
        "mesh column 1 (DLT Y/horizontal) varies too little across the run -- looks swapped"
    )


def test_mesh_centroid_never_resets_to_origin(mesh_run_result, manifest_df):
    new_dir, _file_base = mesh_run_result
    centroids = []
    for frame in sorted(manifest_df["frame"].to_numpy())[:200]:
        obj_path = new_dir / "meshes_obj" / f"frame_{int(frame):06d}.obj"
        if not obj_path.is_file():
            continue
        vertices = read_obj_vertices(obj_path)
        centroids.append(vertices.mean(axis=0))
    centroids = np.asarray(centroids)
    assert len(centroids) > 50

    # Regression guard for the origin-reset bug class fixed once in
    # sam3dinov3_visualize.py: no frame's centroid should sit at/near (0,0,0)
    # while neighboring frames are far from it.
    distances_from_origin = np.linalg.norm(centroids, axis=1)
    near_origin = distances_from_origin < 0.05
    assert not near_origin.any(), (
        f"{near_origin.sum()} frame(s) have a mesh centroid within 5cm of the "
        f"world origin — looks like an unapplied translation, not real motion"
    )

    # World frame here is the calibration volume from c1c2c3_cod.ref3d
    # (X in [0, 2.5], Y in [-5, 5], Z/height in [0.07, 1.6] m). Mesh vertices
    # are always written raw (see test_mesh_vertices_stay_raw_regardless_of_swap_yz),
    # so column 0 = DLT X, column 1 = DLT Y/horizontal, column 2 = DLT
    # Z/height — a human centroid should sit within a generous margin of
    # that box, not run away to numerically implausible coordinates.
    assert np.all(np.abs(centroids[:, 0]) < 10.0)
    assert np.all((centroids[:, 1] > -10.0) & (centroids[:, 1] < 10.0))
    assert np.all((centroids[:, 2] > -1.0) & (centroids[:, 2] < 3.0))
