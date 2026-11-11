"""
================================================================================
Script: freekiki_blender.py - Blender 3D scene for a FreeKiki penalty / kick run
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.16
Created: 09 November 2026
Update Date: 11 November 2026

Description:
    Exports a Blender-ready package from a finished FreeKiki 3D camera folder
    (``person_*.c3d`` / ``*_c3d.csv``, ``ball_c3d.csv``, ``player_map.csv``,
    ``data/penalty_metrics.json``):

    * BVH skeletons for the keeper and the kicker (named markers, Z-up pitch →
      BVH Y-up);
    * a JSON ball trajectory + goal geometry;
    * a companion ``*_blender_penalty_viz.py`` that imports the BVHs, draws the
      goal mouth, keyframes the ball, and marks kick / response / save on the
      timeline;
    * optional headless EEVEE MP4 (``animacao_3d.mp4``) when Blender is found.

    CLI::

        uv run python -m vaila.freekiki_blender -i /path/to/c1_main_3d_… [--render]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from .blender_viz import find_blender_executable
except ImportError:
    from blender_viz import find_blender_executable  # ty: ignore[unresolved-import]

GOAL_WIDTH_M = 7.32
GOAL_HEIGHT_M = 2.44
BALL_RADIUS_M = 0.11

# Body bones as short anatomical names (MHR70 body subset).
_BODY_BONES: list[tuple[str, str]] = [
    ("nose", "left-eye"),
    ("nose", "right-eye"),
    ("left-eye", "left-ear"),
    ("right-eye", "right-ear"),
    ("left-shoulder", "right-shoulder"),
    ("left-shoulder", "left-elbow"),
    ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"),
    ("right-elbow", "right-wrist"),
    ("left-shoulder", "left-hip"),
    ("right-shoulder", "right-hip"),
    ("left-hip", "right-hip"),
    ("left-hip", "left-knee"),
    ("left-knee", "left-ankle"),
    ("right-hip", "right-knee"),
    ("right-knee", "right-ankle"),
    ("left-ankle", "left-heel"),
    ("right-ankle", "right-heel"),
    ("neck", "nose"),
    ("neck", "left-shoulder"),
    ("neck", "right-shoulder"),
]


def _read_player_map(path: Path) -> dict[str, int]:
    """``role -> sam_id`` for keeper / kicker (and others ignored)."""
    out: dict[str, int] = {}
    if not path.is_file():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            role = (row.get("role") or "").strip().lower()
            if role not in ("keeper", "kicker"):
                continue
            try:
                out[role] = int(row.get("sam_id") or row.get("person_id") or "")
            except ValueError:
                continue
    return out


def _load_marker_csv(path: Path) -> tuple[list[str], np.ndarray, float]:
    """Return ``(short_labels, xyz[n,m,3], fps)`` from a FreeKiki ``*_c3d.csv``."""
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no frames")
    x_cols = [c for c in fieldnames if c.endswith("_X")]
    labels: list[str] = []
    for col in x_cols:
        raw = col[:-2]  # drop _X
        # P39_nose → nose; BALL → ball
        if "_" in raw:
            labels.append(raw.split("_", 1)[1])
        else:
            labels.append(raw.lower())
    n = len(rows)
    m = len(labels)
    xyz = np.full((n, m, 3), np.nan, dtype=float)
    times: list[float] = []
    for i, row in enumerate(rows):
        times.append(float(row.get("time_s") or i))
        for j, col in enumerate(x_cols):
            base = col[:-2]
            try:
                xyz[i, j, 0] = float(row[f"{base}_X"])
                xyz[i, j, 1] = float(row[f"{base}_Y"])
                xyz[i, j, 2] = float(row[f"{base}_Z"])
            except (KeyError, TypeError, ValueError):
                continue
    fps = 30.0
    if n > 1 and times[-1] > times[0]:
        fps = (n - 1) / (times[-1] - times[0])
    return labels, xyz, float(fps)


def _gap_fill(xyz: np.ndarray) -> np.ndarray:
    """Linear fill along time for each marker/axis (BVH has no NaN)."""
    out = xyz.copy()
    n, m, _ = out.shape
    for j in range(m):
        for c in range(3):
            series = out[:, j, c]
            good = np.isfinite(series)
            if not good.any():
                out[:, j, c] = 0.0
                continue
            if good.all():
                continue
            idx = np.arange(n)
            out[:, j, c] = np.interp(idx, idx[good], series[good])
    return out


def write_bvh(
    path: Path,
    labels: list[str],
    xyz: np.ndarray,
    *,
    fps: float,
    swap_yz: bool = True,
) -> Path:
    """Write one BVH with a ROOT per marker (same convention as :func:`rec3d.save_rec3d_as_bvh`)."""
    filled = _gap_fill(xyz)
    n = filled.shape[0]
    frame_time = 1.0 / fps if fps > 0 else 0.01
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("HIERARCHY\n")
        for label in labels:
            safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in label)
            fh.write(f"ROOT {safe}\n{{\n")
            fh.write("\tOFFSET 0.000000 0.000000 0.000000\n")
            fh.write("\tCHANNELS 3 Xposition Yposition Zposition\n")
            fh.write("\tEnd Site\n\t{\n\t\tOFFSET 0.000000 0.000000 0.000000\n\t}\n}\n")
        fh.write("MOTION\n")
        fh.write(f"Frames: {n}\n")
        fh.write(f"Frame Time: {frame_time:.9f}\n")
        for i in range(n):
            cells: list[str] = []
            for j in range(len(labels)):
                x, y, z = filled[i, j]
                if swap_yz:
                    x, y, z = float(x), float(z), float(-y)
                cells += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]
            fh.write(" ".join(cells) + "\n")
    return path


def _ball_series(cam3d: Path) -> tuple[np.ndarray, float]:
    """``(n, 3)`` ball centres and fps from ``ball_c3d.csv`` or ``ball_3d_on_field.csv``."""
    for name in ("ball_c3d.csv", "ball_3d_on_field.csv", "ball_3d.csv"):
        path = cam3d / name
        if not path.is_file():
            continue
        if name == "ball_c3d.csv":
            _labels, xyz, fps = _load_marker_csv(path)
            return xyz[:, 0, :], fps
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            continue
        xyz = np.full((len(rows), 3), np.nan)
        for i, row in enumerate(rows):
            try:
                xyz[i] = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
            except (KeyError, ValueError):
                continue
        return xyz, 30.0
    raise FileNotFoundError(f"no ball CSV under {cam3d}")


def _penalty_events(cam3d: Path) -> dict:
    path = cam3d / "data" / "penalty_metrics.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _goal_corners(goal_x: float) -> list[list[float]]:
    """Four goal-mouth corners in pitch metres (Z-up), bottom-left → … → bottom-right."""
    half = GOAL_WIDTH_M / 2.0
    return [
        [goal_x, -half, 0.0],
        [goal_x, -half, GOAL_HEIGHT_M],
        [goal_x, half, GOAL_HEIGHT_M],
        [goal_x, half, 0.0],
    ]


def _companion_script(
    out_dir: Path,
    *,
    keeper_bvh: str,
    kicker_bvh: str,
    ball_json: str,
    fps: float,
    n_frames: int,
    events: dict,
    goal_x: float,
) -> str:
    """Python source run inside Blender (imports BVH, draws goal + ball, optional render)."""
    fps_i = max(1, int(round(fps)))
    m = events.get("metrics") or {}
    kick = int(events.get("kick_frame") or m.get("kick_frame") or 0)
    onset = (
        events.get("gk_move_frame")
        if events.get("gk_move_frame") is not None
        else m.get("gk_move_frame")
    )
    save = (
        events.get("save_frame")
        or (events.get("interception") or {}).get("frame")
        or m.get("goal_frame")
    )
    interception_xyz = (events.get("interception") or {}).get("xyz")
    cf = events.get("counterfactual") or {}
    cf_xyz = (
        [float(goal_x), float(cf["y_m"]), float(cf["z_m"])]
        if cf.get("y_m") is not None and cf.get("z_m") is not None
        else None
    )
    # Blender's import_anim.bvh(axis_forward="-Z", axis_up="Y") converts the
    # Y-up BVH back into Blender's native Z-up world frame (x_pitch, y_pitch, z_pitch).
    corners = _goal_corners(goal_x)
    bones = json.dumps(_BODY_BONES)
    return f'''# Auto-generated by vailá freekiki_blender — open in Blender Text Editor or:
#   blender --background --python this_file.py -- --render-mp4 /path/out.mp4
import bpy
import json
import math
import os
import sys
from mathutils import Vector

OUT_DIR = r"{out_dir.as_posix()}"
KEEPER_BVH = os.path.join(OUT_DIR, "{keeper_bvh}")
KICKER_BVH = os.path.join(OUT_DIR, "{kicker_bvh}")
BALL_JSON = os.path.join(OUT_DIR, "{ball_json}")
FPS = {fps_i}
FRAME_END = {max(1, n_frames)}
KICK_F = {kick}
ONSET_F = {onset if onset is not None else "None"}
SAVE_F = {save if save is not None else "None"}
INTERCEPTION_XYZ = {interception_xyz!r}
COUNTERFACTUAL_XYZ = {cf_xyz!r}
GOAL_CORNERS = {corners!r}
BONES = {bones}

def _clear():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.actions):
        bpy.data.actions.remove(block)

def _mat(name, rgba, emission=0.0, roughness=0.35):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if emission > 0.0:
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = rgba
            elif "Emission" in bsdf.inputs:
                bsdf.inputs["Emission"].default_value = rgba
            if "Emission Strength" in bsdf.inputs:
                bsdf.inputs["Emission Strength"].default_value = emission
    return master_mat if (master_mat := mat) else mat

def _poly_curve(name, pts, bevel, mat, cyclic=False):
    curve = bpy.data.curves.new(name=name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    if mat is not None:
        obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj

def _import_bvh(path, prefix):
    if not os.path.isfile(path):
        print("missing BVH", path)
        return None
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(
        filepath=path,
        axis_forward="-Z",
        axis_up="Y",
        global_scale=1.0,
        frame_start=1,
        update_scene_fps=True,
        update_scene_duration=True,
    )
    after = [n for n in bpy.data.objects.keys() if n not in before]
    arm = None
    for name in after:
        obj = bpy.data.objects[name]
        obj.name = f"{{prefix}}_{{obj.name}}"
        if obj.type == "ARMATURE" and arm is None:
            arm = obj
    return arm

def _draw_bones(prefix, arm):
    """Animated 3D bone cylinders and joint spheres across 1..FRAME_END in Z-up world space."""
    if arm is None:
        return
    scene = bpy.context.scene
    if prefix == "keeper":
        bone_mat = _mat("keeper_bone_mat", (0.06, 0.82, 0.52, 1.0), emission=0.35)
        joint_mat = _mat("keeper_joint_mat", (0.98, 0.80, 0.15, 1.0), emission=0.45)
    else:
        bone_mat = _mat("kicker_bone_mat", (0.98, 0.42, 0.12, 1.0), emission=0.35)
        joint_mat = _mat("kicker_joint_mat", (1.0, 0.88, 0.30, 1.0), emission=0.40)

    valid_pairs = []
    joint_names = set()
    for a, b in BONES:
        if arm.pose.bones.get(a) is not None and arm.pose.bones.get(b) is not None:
            valid_pairs.append((a, b))
            joint_names.add(a)
            joint_names.add(b)

    cylinders = {{}}
    for a, b in valid_pairs:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.036, depth=1.0, vertices=12, location=(0, 0, 0))
        cyl = bpy.context.active_object
        cyl.name = f"bone_{{prefix}}_{{a}}_{{b}}"
        cyl.data.materials.append(bone_mat)
        cylinders[(a, b)] = cyl

    spheres = {{}}
    for jname in sorted(joint_names):
        rad = 0.085 if jname == "nose" else (0.058 if "wrist" in jname else 0.044)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=rad, segments=12, ring_count=8, location=(0, 0, 0))
        sp = bpy.context.active_object
        sp.name = f"joint_{{prefix}}_{{jname}}"
        sp.data.materials.append(joint_mat if ("wrist" in jname or jname == "nose") else bone_mat)
        spheres[jname] = sp

    for f in range(1, FRAME_END + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()
        pos = {{}}
        for jname in joint_names:
            pb = arm.pose.bones.get(jname)
            if pb is not None:
                pos[jname] = (arm.matrix_world @ pb.matrix).to_translation()
        for jname, sp in spheres.items():
            p = pos.get(jname)
            if p is None:
                continue
            sp.location = p
            sp.keyframe_insert(data_path="location", frame=f)
        for (a, b), cyl in cylinders.items():
            pa, pb = pos.get(a), pos.get(b)
            if pa is None or pb is None:
                continue
            diff = pb - pa
            dist = diff.length
            if dist < 1e-4:
                continue
            cyl.location = (pa + pb) * 0.5
            cyl.rotation_euler = diff.to_track_quat("Z", "X").to_euler()
            cyl.scale = (1.0, 1.0, dist)
            cyl.keyframe_insert(data_path="location", frame=f)
            cyl.keyframe_insert(data_path="rotation_euler", frame=f)
            cyl.keyframe_insert(data_path="scale", frame=f)

def _goal():
    gx = float(GOAL_CORNERS[0][0])
    white_mat = _mat("goal_white", (0.97, 0.98, 1.0, 1.0), emission=0.45, roughness=0.2)
    net_mat = _mat("goal_net", (0.75, 0.82, 0.90, 1.0), emission=0.15, roughness=0.5)
    line_mat = _mat("pitch_line", (0.92, 0.95, 0.92, 1.0), emission=0.25, roughness=0.4)

    # 3D goalposts + crossbar (goal_mouth)
    _poly_curve("goal_mouth", GOAL_CORNERS, bevel=0.06, mat=white_mat, cyclic=False)
    # Back frame of goal net (depth 1.5 m)
    depth = 1.5
    c0, c1, c2, c3 = GOAL_CORNERS
    _poly_curve("goal_back_left", [c0, [gx + depth, c0[1], 0.0], [gx + depth, c1[1], c1[2]], c1], bevel=0.03, mat=net_mat)
    _poly_curve("goal_back_right", [c3, [gx + depth, c3[1], 0.0], [gx + depth, c2[1], c2[2]], c2], bevel=0.03, mat=net_mat)
    _poly_curve("goal_back_top", [[gx + depth, c1[1], c1[2]], [gx + depth, c2[1], c2[2]]], bevel=0.03, mat=net_mat)
    _poly_curve("goal_back_bottom", [[gx + depth, c0[1], 0.0], [gx + depth, c3[1], 0.0]], bevel=0.03, mat=net_mat)

    # Pitch plane under the goal (horizontal Z = -0.01 in Blender Z-up).
    bpy.ops.mesh.primitive_plane_add(size=32.0, location=(gx - 7.0, 0.0, -0.01))
    plane = bpy.context.active_object
    plane.name = "pitch"
    mat = _mat("pitch_green", (0.10, 0.36, 0.14, 1.0), emission=0.05, roughness=0.85)
    plane.data.materials.append(mat)

    # Pitch markings: goal line, 6-yard box, penalty box, penalty spot
    _poly_curve("line_goal", [[gx, -20.16, 0.005], [gx, 20.16, 0.005]], bevel=0.035, mat=line_mat)
    _poly_curve(
        "line_goal_area",
        [[gx, -9.16, 0.005], [gx - 5.5, -9.16, 0.005], [gx - 5.5, 9.16, 0.005], [gx, 9.16, 0.005]],
        bevel=0.03,
        mat=line_mat,
    )
    _poly_curve(
        "line_penalty_area",
        [[gx, -20.16, 0.005], [gx - 16.5, -20.16, 0.005], [gx - 16.5, 20.16, 0.005], [gx, 20.16, 0.005]],
        bevel=0.03,
        mat=line_mat,
    )
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=0.01, location=(gx - 11.0, 0.0, 0.005))
    spot = bpy.context.active_object
    spot.name = "penalty_spot"
    spot.data.materials.append(white_mat)

def _ball():
    data = json.load(open(BALL_JSON, encoding="utf-8"))
    pts = data["xyz"]  # pitch Z-up (x, y, z)
    ball_mat = _mat("ball_mat", (1.0, 0.95, 0.25, 1.0), emission=0.65, roughness=0.2)
    path_mat = _mat("ball_path_mat", (0.20, 0.85, 1.0, 1.0), emission=0.55, roughness=0.2)
    bpy.ops.mesh.primitive_uv_sphere_add(radius={BALL_RADIUS_M * 1.25:.4f}, location=(0, 0, 0))
    ball = bpy.context.active_object
    ball.name = "ball"
    ball.data.materials.append(ball_mat)

    # Curve of the flight in Z-up world coordinates (x, y, z).
    finite = [p for p in pts if p and all(c is not None and math.isfinite(c) for c in p)]
    if finite:
        _poly_curve("ball_path", finite, bevel=0.025, mat=path_mat)
    if INTERCEPTION_XYZ and len(INTERCEPTION_XYZ) == 3:
        save_mat = _mat("save_marker_mat", (1.0, 0.25, 0.25, 1.0), emission=0.75)
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.09,
            location=(float(INTERCEPTION_XYZ[0]), float(INTERCEPTION_XYZ[1]), float(INTERCEPTION_XYZ[2])),
        )
        sv = bpy.context.active_object
        sv.name = "save_interception_point"
        sv.data.materials.append(save_mat)
        if COUNTERFACTUAL_XYZ and len(COUNTERFACTUAL_XYZ) == 3:
            _poly_curve("ball_counterfactual_ext", [INTERCEPTION_XYZ, COUNTERFACTUAL_XYZ], bevel=0.018, mat=save_mat)

    for i, p in enumerate(pts):
        if not p or not all(c is not None and math.isfinite(c) for c in p):
            continue
        ball.location = (float(p[0]), float(p[1]), float(p[2]))
        ball.keyframe_insert(data_path="location", frame=i + 1)
    return ball

def _markers():
    scene = bpy.context.scene
    for frame, name in ((KICK_F, "kick"), (ONSET_F, "onset"), (SAVE_F, "save")):
        if frame is None:
            continue
        scene.timeline_markers.new(name, frame=int(frame) + 1)

def _camera(ball):
    gx = float(GOAL_CORNERS[0][0])
    # Elevated 3/4 broadcast view behind the penalty taker looking toward the goal mouth (Z-up).
    bpy.ops.object.camera_add(location=(gx - 17.5, -4.2, 3.6))
    cam = bpy.context.active_object
    cam.name = "penalty_cam"
    cam.data.lens = 28.0
    target = Vector((gx - 3.2, 0.2, 1.15))
    direction = target - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    # Dark stadium world background for high contrast
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.04, 0.07, 0.12, 1.0)
        bg.inputs[1].default_value = 0.9

    sun = bpy.data.lights.new("sun", type="SUN")
    sun.energy = 4.5
    sun_obj = bpy.data.objects.new("sun", sun)
    sun_obj.rotation_euler = (math.radians(42), math.radians(-20), math.radians(35))
    bpy.context.collection.objects.link(sun_obj)

def build():
    _clear()
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    keeper = _import_bvh(KEEPER_BVH, "keeper")
    kicker = _import_bvh(KICKER_BVH, "kicker")
    _draw_bones("keeper", keeper)
    _draw_bones("kicker", kicker)
    if keeper is None and kicker is None:
        print("freekiki_blender: warning — no armatures imported")
    _goal()
    ball = _ball()
    _markers()
    _camera(ball)
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    scene.frame_set(max(1, min(FRAME_END, int(SAVE_F or KICK_F or 1) + 1)))
    print("freekiki_blender: scene ready", FRAME_END, "frames @", FPS, "fps")

def render_mp4(out_path):
    import shutil
    import subprocess
    import tempfile
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if hasattr(scene, "eevee"):
        for attr, val in (("taa_render_samples", 8), ("taa_samples", 8), ("use_raytracing", False)):
            if hasattr(scene.eevee, attr):
                try:
                    setattr(scene.eevee, attr, val)
                except Exception:
                    pass
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = FPS
    step = max(1, FRAME_END // 46)
    scene.frame_step = step
    scene.render.image_settings.file_format = "PNG"
    tmp = tempfile.mkdtemp(prefix="freekiki_blender_")
    scene.render.filepath = os.path.join(tmp, "frame_")
    bpy.ops.render.render(animation=True)
    # Blender 5.x dropped direct FFMPEG file_format; stitch PNGs with ffmpeg.
    # frame_step leaves gaps in frame numbers — always renumber sequentially.
    pngs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
    if not pngs:
        raise RuntimeError("Blender wrote no PNG frames")
    for i, name in enumerate(pngs, start=1):
        os.rename(os.path.join(tmp, name), os.path.join(tmp, f"seq_{{i:04d}}.png"))
    pattern = os.path.join(tmp, "seq_%04d.png")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-framerate", str(max(1, FPS // max(1, step))), "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path,
    ]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("freekiki_blender: wrote", out_path, "engine=", scene.render.engine, "frames=", len(pngs))

if __name__ == "__main__":
    try:
        build()
        blend = os.path.join(OUT_DIR, "penalty_scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend)
        print("freekiki_blender: saved", blend)
        argv = sys.argv
        if "--" in argv:
            argv = argv[argv.index("--") + 1 :]
        else:
            argv = []
        if "--render-mp4" in argv:
            i = argv.index("--render-mp4")
            out = argv[i + 1] if i + 1 < len(argv) else os.path.join(OUT_DIR, "animacao_3d.mp4")
            render_mp4(out)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("freekiki_blender: FAILED:", exc)
        sys.exit(1)
'''


def export_penalty_blender(
    cam3d: Path | str,
    *,
    out_dir: Path | str | None = None,
    render: bool = False,
    blender: str | None = None,
) -> dict[str, Path]:
    """Write BVHs + companion script (+ optional MP4) under ``cam3d/blender/``."""
    cam3d = Path(cam3d)
    out = Path(out_dir) if out_dir else cam3d / "blender"
    out.mkdir(parents=True, exist_ok=True)
    roles = _read_player_map(cam3d / "player_map.csv")
    if "keeper" not in roles or "kicker" not in roles:
        raise FileNotFoundError(f"{cam3d}/player_map.csv needs keeper and kicker sam_ids")
    written: dict[str, Path] = {"dir": out}

    def person_csv(sam_id: int) -> Path:
        path = cam3d / f"person_{sam_id:03d}_c3d.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    for role in ("keeper", "kicker"):
        labels, xyz, fps = _load_marker_csv(person_csv(roles[role]))
        bvh = write_bvh(out / f"{role}.bvh", labels, xyz, fps=fps)
        written[f"{role}_bvh"] = bvh

    ball_xyz, ball_fps = _ball_series(cam3d)
    fps = ball_fps
    events = _penalty_events(cam3d)
    goal_x = float(events.get("goal_x_m") or 52.5)
    # Prefer flight contact for frame count alignment with the video.
    n_frames = int(ball_xyz.shape[0])
    ball_json = out / "ball_path.json"
    ball_json.write_text(
        json.dumps(
            {
                "fps": fps,
                "xyz": [
                    [None if not np.isfinite(v) else round(float(v), 5) for v in row]
                    for row in ball_xyz
                ],
                "events": {
                    "kick_frame": events.get("kick_frame"),
                    "gk_move_frame": events.get("gk_move_frame"),
                    "save_frame": events.get("save_frame")
                    or (events.get("interception") or {}).get("frame"),
                },
                "goal_x_m": goal_x,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written["ball_json"] = ball_json

    script = out / "penalty_blender_viz.py"
    script.write_text(
        _companion_script(
            out,
            keeper_bvh="keeper.bvh",
            kicker_bvh="kicker.bvh",
            ball_json="ball_path.json",
            fps=fps,
            n_frames=n_frames,
            events=events,
            goal_x=goal_x,
        ),
        encoding="utf-8",
    )
    written["script"] = script

    exe = find_blender_executable(gui=False, explicit=blender) if render else None
    if render and exe:
        mp4 = out / "animacao_3d.mp4"
        cmd = [
            exe,
            "--background",
            "--python",
            str(script),
            "--",
            "--render-mp4",
            str(mp4),
        ]
        print(">> vaila/freekiki_blender:", " ".join(cmd))
        proc = subprocess.run(cmd, check=False)
        if proc.returncode == 0 and mp4.is_file():
            written["mp4"] = mp4
            for dest in (
                cam3d / "report_tv" / "pt",
                cam3d / "report_tv" / "en",
                cam3d / "report_tv" / "penalty" / "media",
            ):
                if dest.is_dir():
                    shutil.copy2(mp4, dest / "animacao_3d.mp4")
        else:
            print(
                f">> vaila/freekiki_blender: Blender render exited {proc.returncode}; "
                "companion script and BVHs are still available"
            )
    elif render and not exe:
        print(">> vaila/freekiki_blender: Blender not found; wrote BVH + companion only")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a FreeKiki run to Blender (BVH + scene).")
    ap.add_argument("-i", "--input", required=True, help="Camera 3D folder (person_*.c3d, …)")
    ap.add_argument("-o", "--output", default=None, help="Output folder (default: <input>/blender)")
    ap.add_argument(
        "--render", action="store_true", help="Headless EEVEE MP4 when Blender is found"
    )
    ap.add_argument("--blender", default=None, help="Blender executable path")
    args = ap.parse_args(argv)
    try:
        written = export_penalty_blender(
            args.input, out_dir=args.output, render=args.render, blender=args.blender
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"freekiki_blender: {exc}", file=sys.stderr)
        return 1
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
