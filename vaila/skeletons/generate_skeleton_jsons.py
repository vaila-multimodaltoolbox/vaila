"""
================================================================================
Script: generate_skeleton_jsons.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.3.112
Created: 04 August 2026
Last Updated: 24 August 2026

Description:
    Maintenance/dev tool that regenerates all standard skeleton-connection
    JSON presets in ``vaila/skeletons/`` and ``tests/skeleton_templates/``:

      - mediapipe_pose33.json    <- MediaPipe BlazePose (33 keypoints)
      - yolo_coco17.json         <- YOLO / COCO-17 (17 keypoints)
      - sam3dinov3_mhr70.json    <- SAM3+DINOv3 (SAM 3D Body) MHR70 (70 keypoints)
      - sapiens2_goliath308.json <- Sapiens2 Sociopticon/Goliath (308 keypoints)
      - fifa_body15.json         <- FIFA Skeletal Challenge Body-15 (15 keypoints)
      - openpose_body25.json     <- OpenPose Body-25 (25 keypoints)
      - mediapipe_hand21.json    <- MediaPipe Hand (21 keypoints)
      - mediapipe_hands42.json   <- MediaPipe Both Hands (42 keypoints)
      - mediapipe_holistic75.json<- MediaPipe Holistic Body+Hands (75 keypoints)
      - halpe26.json             <- Halpe 26 Body+Feet (26 keypoints)
      - coco_wholebody133.json   <- COCO WholeBody / Sapiens-133 (133 keypoints)
      - soccerfield_pitch32.json <- Soccer Field 32 Pitch Keypoints (32 points)
      - soccerfield_calib29.json <- Soccer Field 29 FIFA Calib Keypoints (29 points)

    "pN" in every emitted connection is the 1-based column index of that
    keypoint in the wide CSV rec3d_one_dlt3d.py/rec3d.py write (frame,
    p1_x,p1_y,p1_z,...), i.e. keypoint index 0 (0-based) in the source
    definition becomes p1, index 1 becomes p2, etc.

Usage:
    uv run python vaila/skeletons/generate_skeleton_jsons.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKEL_DIR = Path(__file__).resolve().parent
TEST_SKEL_DIR = REPO_ROOT / "tests" / "skeleton_templates"
TEST_REC3D_DIR = REPO_ROOT / "tests" / "rec3d_one_dlt3d"


def _generate_mediapipe_pose33() -> dict:
    names = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]
    connections = [
        # Torso & hips
        ["p12", "p13"],
        ["p24", "p25"],
        ["p12", "p24"],
        ["p13", "p25"],
        ["p12", "p25"],
        ["p13", "p24"],
        # Face
        ["p1", "p3"],
        ["p1", "p6"],
        ["p3", "p6"],
        ["p3", "p8"],
        ["p6", "p9"],
        ["p10", "p11"],
        # Left upper limb
        ["p12", "p14"],
        ["p14", "p16"],
        ["p16", "p18"],
        ["p16", "p20"],
        ["p16", "p22"],
        # Right upper limb
        ["p13", "p15"],
        ["p15", "p17"],
        ["p17", "p19"],
        ["p17", "p21"],
        ["p17", "p23"],
        # Left lower limb
        ["p24", "p26"],
        ["p26", "p28"],
        ["p28", "p30"],
        ["p30", "p32"],
        ["p28", "p32"],
        # Right lower limb
        ["p25", "p27"],
        ["p27", "p29"],
        ["p29", "p31"],
        ["p31", "p33"],
        ["p29", "p33"],
    ]
    return {
        "schema": "mediapipe_pose_33_pn",
        "num_keypoints": 33,
        "note": (
            "pN is 1-based index mapping to MediaPipe: p1=nose(0), "
            "p12=left_shoulder(11), p13=right_shoulder(12), "
            "p24=left_hip(23), p25=right_hip(24), ..., p33=right_foot_index(32)."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_yolo_coco17() -> dict:
    names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    connections = [
        ["p6", "p7"],
        ["p12", "p13"],
        ["p6", "p8"],
        ["p8", "p10"],
        ["p7", "p9"],
        ["p9", "p11"],
        ["p12", "p14"],
        ["p14", "p16"],
        ["p13", "p15"],
        ["p15", "p17"],
        ["p1", "p2"],
        ["p1", "p3"],
        ["p2", "p4"],
        ["p3", "p5"],
        ["p6", "p12"],
        ["p7", "p13"],
    ]
    return {
        "schema": "coco_17_keypoints",
        "num_keypoints": 17,
        "note": (
            "pN is 1-based index into standard COCO-17 order (p1=nose, p2=left_eye, "
            "p3=right_eye, p4=left_ear, p5=right_ear, p6=left_shoulder, p7=right_shoulder, "
            "p8=left_elbow, p9=right_elbow, p10=left_wrist, p11=right_wrist, p12=left_hip, "
            "p13=right_hip, p14=left_knee, p15=right_knee, p16=left_ankle, p17=right_ankle)."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_dinov3_mhr70() -> dict:
    src = (REPO_ROOT / "vaila" / "sam3dinov3.py").read_text(encoding="utf-8")

    names_match = re.search(r"MHR70_NAMES: tuple\[str, \.\.\.\] = \((.*?)\)\n", src, re.S)
    if not names_match:
        raise RuntimeError("MHR70_NAMES not found in sam3dinov3.py")
    names = re.findall(r'"([^"]+)"', names_match.group(1))
    if len(names) != 70:
        raise RuntimeError(f"expected 70 MHR70 names, found {len(names)}")

    edges_match = re.search(
        r"SKELETON_EDGE_NAMES: tuple\[tuple\[str, str\], \.\.\.\] = \((.*?)\n\)\n", src, re.S
    )
    if not edges_match:
        raise RuntimeError("SKELETON_EDGE_NAMES not found in sam3dinov3.py")
    edge_pairs = re.findall(r'\("([^"]+)",\s*"([^"]+)"\)', edges_match.group(1))

    name_to_pidx = {n: i + 1 for i, n in enumerate(names)}
    connections = []
    for a, b in edge_pairs:
        if a not in name_to_pidx or b not in name_to_pidx:
            raise RuntimeError(f"skeleton edge references unknown MHR70 name: {a!r}/{b!r}")
        connections.append([f"p{name_to_pidx[a]}", f"p{name_to_pidx[b]}"])

    # Foot triangle extra stabilization
    extra_foot_pairs = [
        ["p18", "p16"],  # left_heel - left_big_toe
        ["p18", "p17"],  # left_heel - left_small_toe
        ["p16", "p17"],  # left_big_toe - left_small_toe
        ["p21", "p19"],  # right_heel - right_big_toe
        ["p21", "p20"],  # right_heel - right_small_toe
        ["p19", "p20"],  # right_big_toe - right_small_toe
    ]
    for pair in extra_foot_pairs:
        if pair not in connections and [pair[1], pair[0]] not in connections:
            connections.append(pair)

    return {
        "schema": "sam3dinov3_mhr70",
        "num_keypoints": 70,
        "note": (
            "pN is 1-based index into the 70-keypoint MHR order used by "
            "sam3dinov3.py/sam3dinov3_visualize.py (p1=nose, p6=left-shoulder, "
            "p7=right-shoulder, p10=left-hip, p11=right-hip, ..., p70=neck). "
            "Derived programmatically from vaila/sam3dinov3.py."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_sapiens2_goliath308() -> dict:
    config_path = (
        REPO_ROOT
        / ".local"
        / "third_party"
        / "sapiens2"
        / "sapiens"
        / "pose"
        / "configs"
        / "_base_"
        / "keypoints308.py"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found — run bash bin/setup_sapiens2.sh first.")

    ns: dict = {}
    code = config_path.read_text(encoding="utf-8")
    exec(compile(code, str(config_path), "exec"), ns)  # noqa: S102

    dataset_info = ns["dataset_info"]
    kp_info = dataset_info["keypoint_info"]
    skel_info = dataset_info["skeleton_info"]
    if len(kp_info) != 308:
        raise RuntimeError(f"expected 308 Sapiens2 keypoints, found {len(kp_info)}")

    names = [kp_info[i]["name"] for i in sorted(kp_info.keys())]
    name_to_pidx = {v["name"]: idx + 1 for idx, v in kp_info.items()}
    connections = []
    for i in sorted(skel_info.keys()):
        a, b = skel_info[i]["link"]
        if a not in name_to_pidx or b not in name_to_pidx:
            raise RuntimeError(f"skeleton_info link references unknown keypoint: {a!r}/{b!r}")
        connections.append([f"p{name_to_pidx[a]}", f"p{name_to_pidx[b]}"])

    extra_foot_pairs = [
        ["p18", "p16"],  # left_heel - left_big_toe
        ["p18", "p17"],  # left_heel - left_small_toe
        ["p16", "p17"],  # left_big_toe - left_small_toe
        ["p21", "p19"],  # right_heel - right_big_toe
        ["p21", "p20"],  # right_heel - right_small_toe
        ["p19", "p20"],  # right_big_toe - right_small_toe
    ]
    for pair in extra_foot_pairs:
        if pair not in connections and [pair[1], pair[0]] not in connections:
            connections.append(pair)

    return {
        "schema": "sapiens2_goliath_308",
        "num_keypoints": 308,
        "note": (
            "pN is 1-based index into the 308-keypoint Sociopticon/Goliath "
            "order used by vaila_sapiens.py/sam3sapiens2.py (p1=nose, "
            "p6=left_shoulder, p7=right_shoulder, p10=left_hip, "
            "p11=right_hip, ..., p308=last face point). Derived from "
            "sapiens/pose/configs/_base_/keypoints308.py."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_fifa_body15() -> dict:
    names = [
        "nose",
        "right_shoulder",
        "left_shoulder",
        "right_elbow",
        "left_elbow",
        "right_wrist",
        "left_wrist",
        "right_hip",
        "left_hip",
        "right_knee",
        "left_knee",
        "right_ankle",
        "left_ankle",
        "right_toe",
        "left_toe",
    ]
    connections = [
        # Head & Shoulders
        ["p1", "p2"],
        ["p1", "p3"],
        ["p2", "p3"],
        # Right arm
        ["p2", "p4"],
        ["p4", "p6"],
        # Left arm
        ["p3", "p5"],
        ["p5", "p7"],
        # Torso & pelvis
        ["p2", "p8"],
        ["p3", "p9"],
        ["p8", "p9"],
        # Right leg & foot
        ["p8", "p10"],
        ["p10", "p12"],
        ["p12", "p14"],
        # Left leg & foot
        ["p9", "p11"],
        ["p11", "p13"],
        ["p13", "p15"],
    ]
    return {
        "schema": "fifa_body_15",
        "num_keypoints": 15,
        "note": (
            "pN is 1-based index for FIFA Challenge 2026 Body-15 keypoint set: "
            "p1=nose, p2=right_shoulder, p3=left_shoulder, p4=right_elbow, "
            "p5=left_elbow, p6=right_wrist, p7=left_wrist, p8=right_hip, "
            "p9=left_hip, p10=right_knee, p11=left_knee, p12=right_ankle, "
            "p13=left_ankle, p14=right_toe, p15=left_toe."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_openpose_body25() -> dict:
    names = [
        "nose",
        "neck",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "mid_hip",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_eye",
        "left_eye",
        "right_ear",
        "left_ear",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    ]
    connections = [
        # Spine & Trunk
        ["p1", "p2"],
        ["p2", "p9"],
        # Right arm
        ["p2", "p3"],
        ["p3", "p4"],
        ["p4", "p5"],
        # Left arm
        ["p2", "p6"],
        ["p6", "p7"],
        ["p7", "p8"],
        # Right leg & foot
        ["p9", "p10"],
        ["p10", "p11"],
        ["p11", "p12"],
        ["p12", "p23"],
        ["p23", "p24"],
        ["p12", "p25"],
        # Left leg & foot
        ["p9", "p13"],
        ["p13", "p14"],
        ["p14", "p15"],
        ["p15", "p20"],
        ["p20", "p21"],
        ["p15", "p22"],
        # Face
        ["p1", "p16"],
        ["p16", "p18"],
        ["p1", "p17"],
        ["p17", "p19"],
    ]
    return {
        "schema": "openpose_body_25",
        "num_keypoints": 25,
        "note": (
            "pN is 1-based index for standard OpenPose Body-25 format: "
            "p1=nose, p2=neck, p3=right_shoulder, p4=right_elbow, p5=right_wrist, "
            "p6=left_shoulder, p7=left_elbow, p8=left_wrist, p9=mid_hip, "
            "p10=right_hip, p11=right_knee, p12=right_ankle, p13=left_hip, "
            "p14=left_knee, p15=left_ankle, p16=right_eye, p17=left_eye, "
            "p18=right_ear, p19=left_ear, p20=left_big_toe, p21=left_small_toe, "
            "p22=left_heel, p23=right_big_toe, p24=right_small_toe, p25=right_heel."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_mediapipe_hand21() -> dict:
    names = [
        "wrist",
        "thumb_cmc",
        "thumb_mcp",
        "thumb_ip",
        "thumb_tip",
        "index_mcp",
        "index_pip",
        "index_dip",
        "index_tip",
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "middle_tip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "ring_tip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
        "pinky_tip",
    ]
    connections = [
        # Palm
        ["p1", "p2"],
        ["p1", "p6"],
        ["p6", "p10"],
        ["p10", "p14"],
        ["p14", "p18"],
        ["p1", "p18"],
        # Thumb
        ["p2", "p3"],
        ["p3", "p4"],
        ["p4", "p5"],
        # Index
        ["p6", "p7"],
        ["p7", "p8"],
        ["p8", "p9"],
        # Middle
        ["p10", "p11"],
        ["p11", "p12"],
        ["p12", "p13"],
        # Ring
        ["p14", "p15"],
        ["p15", "p16"],
        ["p16", "p17"],
        # Pinky
        ["p18", "p19"],
        ["p19", "p20"],
        ["p20", "p21"],
    ]
    return {
        "schema": "mediapipe_hand_21",
        "num_keypoints": 21,
        "note": (
            "pN is 1-based index for single hand MediaPipe landmarks: "
            "p1=wrist, p2..p5=thumb, p6..p9=index, p10..p13=middle, "
            "p14..p17=ring, p18..p21=pinky."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_mediapipe_hands42() -> dict:
    hand_names = [
        "wrist",
        "thumb_cmc",
        "thumb_mcp",
        "thumb_ip",
        "thumb_tip",
        "index_mcp",
        "index_pip",
        "index_dip",
        "index_tip",
        "middle_mcp",
        "middle_pip",
        "middle_dip",
        "middle_tip",
        "ring_mcp",
        "ring_pip",
        "ring_dip",
        "ring_tip",
        "pinky_mcp",
        "pinky_pip",
        "pinky_dip",
        "pinky_tip",
    ]
    names = [f"left_{n}" for n in hand_names] + [f"right_{n}" for n in hand_names]
    h21_conns = _generate_mediapipe_hand21()["connections"]

    connections = []
    # Left hand p1..p21
    for a, b in h21_conns:
        connections.append([a, b])
    # Right hand p22..p42
    for a, b in h21_conns:
        ia = int(a[1:]) + 21
        ib = int(b[1:]) + 21
        connections.append([f"p{ia}", f"p{ib}"])

    return {
        "schema": "mediapipe_hands_42",
        "num_keypoints": 42,
        "note": (
            "pN is 1-based index for two-hand MediaPipe landmarks: "
            "p1..p21=left hand (wrist..pinky), p22..p42=right hand (wrist..pinky)."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_mediapipe_holistic75() -> dict:
    body = _generate_mediapipe_pose33()
    hand21 = _generate_mediapipe_hand21()

    names = list(body["keypoints"])
    names += [f"left_hand_{n}" for n in hand21["keypoints"]]
    names += [f"right_hand_{n}" for n in hand21["keypoints"]]

    connections = list(body["connections"])
    # Left hand p34..p54
    for a, b in hand21["connections"]:
        ia = int(a[1:]) + 33
        ib = int(b[1:]) + 33
        connections.append([f"p{ia}", f"p{ib}"])
    # Link body left wrist (p16) to left hand wrist (p34)
    connections.append(["p16", "p34"])

    # Right hand p55..p75
    for a, b in hand21["connections"]:
        ia = int(a[1:]) + 54
        ib = int(b[1:]) + 54
        connections.append([f"p{ia}", f"p{ib}"])
    # Link body right wrist (p17) to right hand wrist (p55)
    connections.append(["p17", "p55"])

    return {
        "schema": "mediapipe_holistic_75",
        "num_keypoints": 75,
        "note": (
            "pN is 1-based index for MediaPipe Holistic Body (33) + Left Hand (21) "
            "+ Right Hand (21): p1..p33=body pose, p34..p54=left hand, p55..p75=right hand."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_halpe26() -> dict:
    names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "head",
        "neck",
        "hip",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    ]
    connections = [
        # Spine & Head
        ["p18", "p1"],
        ["p1", "p19"],
        ["p19", "p20"],
        ["p19", "p6"],
        ["p19", "p7"],
        # Upper limbs
        ["p6", "p8"],
        ["p8", "p10"],
        ["p7", "p9"],
        ["p9", "p11"],
        # Trunk
        ["p6", "p12"],
        ["p7", "p13"],
        ["p12", "p20"],
        ["p13", "p20"],
        # Face
        ["p1", "p2"],
        ["p1", "p3"],
        ["p2", "p4"],
        ["p3", "p5"],
        # Left leg & foot
        ["p12", "p14"],
        ["p14", "p16"],
        ["p16", "p21"],
        ["p16", "p22"],
        ["p16", "p23"],
        ["p21", "p22"],
        # Right leg & foot
        ["p13", "p15"],
        ["p15", "p17"],
        ["p17", "p24"],
        ["p17", "p25"],
        ["p17", "p26"],
        ["p24", "p25"],
    ]
    return {
        "schema": "halpe_26",
        "num_keypoints": 26,
        "note": (
            "pN is 1-based index for Halpe 26 (AlphaPose / YOLO-Pose Body+Feet): "
            "p1..p17=COCO-17, p18=head, p19=neck, p20=hip, "
            "p21..p23=left foot (big_toe, small_toe, heel), "
            "p24..p26=right foot (big_toe, small_toe, heel)."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_coco_wholebody133() -> dict:
    config_path = (
        REPO_ROOT
        / ".local"
        / "third_party"
        / "sapiens2"
        / "sapiens"
        / "pose"
        / "configs"
        / "_base_"
        / "keypoints308.py"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} not found — run bash bin/setup_sapiens2.sh first.")

    ns: dict = {}
    code = config_path.read_text(encoding="utf-8")
    exec(compile(code, str(config_path), "exec"), ns)  # noqa: S102

    wb = ns["coco_wholebody_info"]
    kp_info = wb["keypoint_info"]
    skel_info = wb["skeleton_info"]
    if len(kp_info) != 133:
        raise RuntimeError(f"expected 133 WholeBody keypoints, found {len(kp_info)}")

    names = [kp_info[i]["name"] for i in sorted(kp_info.keys())]
    name_to_pidx = {v["name"]: idx + 1 for idx, v in kp_info.items()}
    connections = []
    for i in sorted(skel_info.keys()):
        a, b = skel_info[i]["link"]
        if a not in name_to_pidx or b not in name_to_pidx:
            continue
        connections.append([f"p{name_to_pidx[a]}", f"p{name_to_pidx[b]}"])

    # Foot triangle extra stabilization
    extra_foot_pairs = [
        ["p20", "p18"],  # left_heel - left_big_toe
        ["p20", "p19"],  # left_heel - left_small_toe
        ["p18", "p19"],  # left_big_toe - left_small_toe
        ["p23", "p21"],  # right_heel - right_big_toe
        ["p23", "p22"],  # right_heel - right_small_toe
        ["p21", "p22"],  # right_big_toe - right_small_toe
    ]
    for pair in extra_foot_pairs:
        if pair not in connections and [pair[1], pair[0]] not in connections:
            connections.append(pair)

    return {
        "schema": "coco_wholebody_133",
        "num_keypoints": 133,
        "note": (
            "pN is 1-based index for COCO WholeBody / Sapiens-133: "
            "p1..p17=body, p18..p23=feet, p24..p91=face (68), p92..p112=left hand (21), "
            "p113..p133=right hand (21)."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_soccerfield_pitch32() -> dict:
    try:
        from vaila.fifa_dataset_builder import CANONICAL_KP_NAMES_32

        names = list(CANONICAL_KP_NAMES_32)
    except Exception:
        names = [
            "top_left_corner",
            "left_pen_box_top_outer",
            "left_goal_area_top_outer",
            "left_goal_area_bottom_outer",
            "left_pen_box_bottom_outer",
            "bottom_left_corner",
            "left_goal_area_top_inner",
            "left_goal_area_bottom_inner",
            "left_penalty_spot",
            "left_pen_box_top_inner",
            "left_pen_box_inner_top_at_goal_y",
            "left_pen_box_inner_bottom_at_goal_y",
            "left_pen_box_bottom_inner",
            "midfield_top",
            "center_circle_top",
            "center_circle_bottom",
            "midfield_bottom",
            "right_pen_box_top_inner",
            "right_pen_box_inner_top_at_goal_y",
            "right_pen_box_inner_bottom_at_goal_y",
            "right_pen_box_bottom_inner",
            "right_penalty_spot",
            "right_goal_area_top_inner",
            "right_goal_area_bottom_inner",
            "top_right_corner",
            "right_pen_box_top_outer",
            "right_goal_area_top_outer",
            "right_goal_area_bottom_outer",
            "right_pen_box_bottom_outer",
            "bottom_right_corner",
            "center_circle_left",
            "center_circle_right",
        ]

    connections = [
        # Top touchline
        ["p1", "p14"],
        ["p14", "p25"],
        # Bottom touchline
        ["p6", "p17"],
        ["p17", "p30"],
        # Left goal line
        ["p1", "p2"],
        ["p2", "p3"],
        ["p3", "p4"],
        ["p4", "p5"],
        ["p5", "p6"],
        # Right goal line
        ["p25", "p26"],
        ["p26", "p27"],
        ["p27", "p28"],
        ["p28", "p29"],
        ["p29", "p30"],
        # Midfield line
        ["p14", "p15"],
        ["p15", "p16"],
        ["p16", "p17"],
        # Left penalty box
        ["p2", "p10"],
        ["p10", "p11"],
        ["p11", "p12"],
        ["p12", "p13"],
        ["p13", "p5"],
        # Left goal area
        ["p3", "p7"],
        ["p7", "p8"],
        ["p8", "p4"],
        # Right penalty box
        ["p26", "p18"],
        ["p18", "p19"],
        ["p19", "p20"],
        ["p20", "p21"],
        ["p21", "p29"],
        # Right goal area
        ["p27", "p23"],
        ["p23", "p24"],
        ["p24", "p28"],
        # Center circle
        ["p15", "p31"],
        ["p31", "p16"],
        ["p16", "p32"],
        ["p32", "p15"],
    ]
    return {
        "schema": "soccerfield_pitch_32",
        "num_keypoints": 32,
        "note": (
            "pN is 1-based index into FIFA dataset builder 32-point canonical pitch layout: "
            "p1..p6=left touchline/goal line, p14..p17=midfield, p25..p30=right touchline/goal line, "
            "p7..p13=left boxes, p18..p24=right boxes, p31..p32=center circle."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _generate_soccerfield_calib29() -> dict:
    names = [
        "bottom_left_corner",
        "top_left_corner",
        "bottom_right_corner",
        "top_right_corner",
        "midfield_left",
        "midfield_right",
        "center_field",
        "center_circle_top_intersection",
        "center_circle_bottom_intersection",
        "left_goal_bottom_post",
        "left_goal_top_post",
        "right_goal_bottom_post",
        "right_goal_top_post",
        "left_penalty_area_top_left",
        "left_penalty_area_top_right",
        "left_penalty_area_bottom_left",
        "left_penalty_area_bottom_right",
        "left_goal_area_top_left",
        "left_goal_area_top_right",
        "left_goal_area_bottom_left",
        "left_goal_area_bottom_right",
        "left_penalty_spot",
        "left_penalty_arc_top",
        "left_penalty_arc_left_intersection",
        "left_penalty_arc_right_intersection",
        "right_penalty_area_top_left",
        "right_penalty_area_top_right",
        "right_penalty_area_bottom_left",
        "right_penalty_area_bottom_right",
    ]
    connections = [
        # Outer boundary
        ["p1", "p2"],
        ["p2", "p6"],
        ["p6", "p4"],
        ["p4", "p3"],
        ["p3", "p5"],
        ["p5", "p1"],
        # Midfield line
        ["p5", "p9"],
        ["p9", "p7"],
        ["p7", "p8"],
        ["p8", "p6"],
        # Left penalty area
        ["p16", "p14"],
        ["p14", "p15"],
        ["p15", "p17"],
        # Left goal area
        ["p20", "p18"],
        ["p18", "p19"],
        ["p19", "p21"],
        # Left goal posts
        ["p10", "p11"],
        # Left penalty arc
        ["p24", "p23"],
        ["p23", "p25"],
        # Right penalty area
        ["p28", "p26"],
        ["p26", "p27"],
        ["p27", "p29"],
        # Right goal posts
        ["p12", "p13"],
    ]
    return {
        "schema": "soccerfield_calib_29",
        "num_keypoints": 29,
        "note": (
            "pN is 1-based index into FIFA 29-point soccer-field calibration reference "
            "(models/soccerfield_ref3d_fifa.csv)."
        ),
        "keypoints": names,
        "connections": connections,
    }


def _write_compact(payload: dict, out_path: Path) -> None:
    """Write with clean formatted JSON."""
    lines = ["{"]
    lines.append(f'  "schema": {json.dumps(payload["schema"])},')
    lines.append(f'  "num_keypoints": {json.dumps(payload.get("num_keypoints", 0))},')
    lines.append(f'  "note": {json.dumps(payload["note"])},')
    if "keypoints" in payload:
        lines.append('  "keypoints": [')
        kp_lines = [f"    {json.dumps(name)}" for name in payload["keypoints"]]
        lines.append(",\n".join(kp_lines))
        lines.append("  ],")
    lines.append('  "connections": [')
    conn_lines = [f"    {json.dumps(pair)}" for pair in payload["connections"]]
    lines.append(",\n".join(conn_lines))
    lines.append("  ]")
    lines.append("}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all_presets() -> dict[str, dict]:
    """Generate all skeleton dictionary specifications."""
    presets = {
        "mediapipe_pose33.json": _generate_mediapipe_pose33(),
        "yolo_coco17.json": _generate_yolo_coco17(),
        "sam3dinov3_mhr70.json": _generate_dinov3_mhr70(),
        "fifa_body15.json": _generate_fifa_body15(),
        "openpose_body25.json": _generate_openpose_body25(),
        "mediapipe_hand21.json": _generate_mediapipe_hand21(),
        "mediapipe_hands42.json": _generate_mediapipe_hands42(),
        "mediapipe_holistic75.json": _generate_mediapipe_holistic75(),
        "halpe26.json": _generate_halpe26(),
        "soccerfield_pitch32.json": _generate_soccerfield_pitch32(),
        "soccerfield_calib29.json": _generate_soccerfield_calib29(),
    }

    try:
        presets["sapiens2_goliath308.json"] = _generate_sapiens2_goliath308()
    except FileNotFoundError as e:
        print(f"skipped sapiens2_goliath308.json: {e}")

    try:
        presets["coco_wholebody133.json"] = _generate_coco_wholebody133()
    except FileNotFoundError as e:
        print(f"skipped coco_wholebody133.json: {e}")

    return presets


def write_all(presets: dict[str, dict]) -> None:
    """Write all presets to vaila/skeletons/ and tests/skeleton_templates/."""
    # 1. Target: vaila/skeletons/
    for filename, data in presets.items():
        out_path = SKEL_DIR / filename
        _write_compact(data, out_path)
        print(f"wrote {out_path} ({len(data['connections'])} connections)")

    # 2. Target: tests/skeleton_templates/
    TEST_SKEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename, data in presets.items():
        out_path = TEST_SKEL_DIR / filename
        _write_compact(data, out_path)

    # 3. Compatibility aliases in tests/skeleton_templates/
    aliases = {
        "skeleton_pose_mediapipe.json": "mediapipe_pose33.json",
        "skeleton_pose_yolo.json": "yolo_coco17.json",
        "skeleton_pose_sam3dinov3.json": "sam3dinov3_mhr70.json",
        "skeleton_pose_sapiens2.json": "sapiens2_goliath308.json",
        "skeleton_pose_fifa.json": "fifa_body15.json",
        "skeleton_pose_openpose25.json": "openpose_body25.json",
        "skeleton_pose_hand21.json": "mediapipe_hand21.json",
    }
    for alias_name, target in aliases.items():
        if target in presets:
            out_path = TEST_SKEL_DIR / alias_name
            _write_compact(presets[target], out_path)

    # 4. Compatibility files in tests/rec3d_one_dlt3d/
    TEST_REC3D_DIR.mkdir(parents=True, exist_ok=True)
    rec3d_fixtures = {
        "skeleton_pose_mediapipe.json": "mediapipe_pose33.json",
        "skeleton_pose_yolo.json": "yolo_coco17.json",
        "skeleton_pose_sam3dinov3.json": "sam3dinov3_mhr70.json",
        "skeleton_pose_sapiens2.json": "sapiens2_goliath308.json",
        "skeleton_pose_fifa.json": "fifa_body15.json",
    }
    for fname, target in rec3d_fixtures.items():
        if target in presets:
            out_path = TEST_REC3D_DIR / fname
            _write_compact(presets[target], out_path)


def main() -> None:
    presets = generate_all_presets()
    write_all(presets)
    print(f"\nDone! Generated {len(presets)} skeleton presets.")


if __name__ == "__main__":
    main()
