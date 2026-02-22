"""
===============================================================================
vaila_tugtun.py
===============================================================================
Author: Paulo R. P. Santiago
Created: 20 February 2026
Updated: 22 February 2026
Version: 0.0.3
Python Version: 3.12.12

Description:
------------
This script provides functionality for Timed Up and Go (TUG) instrumented
analysis with 3D kinematics.

How to use:
-----------
python vaila_tugtun.py -i <input_path> -c <config_path> -o <output_path>

or

uv run vaila_tugtun.py -i <input_path> -c <config_path> -o <output_path>

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
===============================================================================
"""

import matplotlib.pyplot as plt
import argparse
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import json
import importlib
try:
    import tomllib
except ModuleNotFoundError:
    _tomli = importlib.util.find_spec("tomli")
    tomllib = importlib.import_module("tomli") if _tomli is not None else None

import numpy as np
import pandas as pd

DEFAULT_SKELETON_JSON = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "vaila_tugturn"
    / "skeleton_pose_mediapipe.json"
)
# ── TUG Spatial Protocol Constants (LaBioCoM) ────────────────────────────────
# These can be overridden per-subject via the .toml config file:
#   [spatial]
#   y_chair = 1.125   # metres
#   y_turn  = 4.5     # metres
Y_CHAIR_THRESHOLD = 1.125   # m: chair boundary; end of STS / start of sit-to-sit
Y_TURN_APPROX    = 4.5      # m: approximate centre of the turn / pause zone
Y_TURN_TOLERANCE = 0.5      # m: ± search window around the turn zone
# ─────────────────────────────────────────────────────────────────────────────


FALLBACK_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32),
]

RIGHT_BODY_POINTS = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
LEFT_BODY_POINTS = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
PHASE_PLOT_ORDER = [
    "stand",
    "gait_forward",
    "stop_5s",
    "turn180",
    "gait_back",
    "sit",
]

# MediaPipe Pose landmark names in 0-based index order.
MEDIAPIPE_LANDMARK_NAMES = [
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


def load_mediapipe_pose_connections(json_path: Path = DEFAULT_SKELETON_JSON):
    """Load mediapipe_pose_33_pn connections from JSON, fallback to defaults on any failure."""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        conns = data.get("connections", [])
        parsed = []
        for pair in conns:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = str(pair[0]).strip(), str(pair[1]).strip()
            if not (a.lower().startswith("p") and b.lower().startswith("p")):
                continue
            if not (a[1:].isdigit() and b[1:].isdigit()):
                continue
            ia = int(a[1:]) - 1
            ib = int(b[1:]) - 1
            if 0 <= ia < 33 and 0 <= ib < 33:
                parsed.append((ia, ib))
        if parsed:
            return parsed
    except Exception as e:
        print(f"Warning: could not load skeleton JSON '{json_path}': {e}. Using fallback.")
    return FALLBACK_SKELETON_CONNECTIONS


def _agent_debug_log(run_id, hypothesis_id, location, message, data=None):
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data or {},
        "timestamp": int(datetime.datetime.now().timestamp() * 1000),
    }
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as dbg:
            dbg.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


def get_connection_color(a_idx: int, b_idx: int):
    if a_idx in RIGHT_BODY_POINTS and b_idx in RIGHT_BODY_POINTS:
        return "red"
    if a_idx in LEFT_BODY_POINTS and b_idx in LEFT_BODY_POINTS:
        return "blue"
    return "black"


def sample_frames(frames, max_frames=12):
    if not frames:
        return []
    ordered = sorted({int(f) for f in frames})
    if len(ordered) <= max_frames:
        return ordered
    idx = np.linspace(0, len(ordered) - 1, max_frames).astype(int)
    return [ordered[i] for i in idx]


def ordered_phase_ranges(phases_dict):
    for phase_name in PHASE_PLOT_ORDER:
        val = phases_dict.get(phase_name)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            yield phase_name, val


def write_single_row_csv(filepath: Path, data: dict):
    """Write one-row CSV (overwrite mode)."""
    pd.DataFrame([data]).to_csv(filepath, index=False)


def _format_html_value(value):
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "" if value is None else str(value)


def build_side_by_side_rows(left_dict: dict, right_dict: dict):
    """Build HTML rows for Variable | Left | Right comparative table."""
    keys = sorted(set(left_dict.keys()) | set(right_dict.keys()))
    rows = []
    for key in keys:
        if key == 'per_step':
            continue
        label = key.replace("_", " ")
        left_val = _format_html_value(left_dict.get(key))
        right_val = _format_html_value(right_dict.get(key))
        rows.append(
            f"<tr><th>{label}</th><td class='col-left'>{left_val}</td><td class='col-right'>{right_val}</td></tr>"
        )
    return "".join(rows)

def calculate_angle_3d(p1, p2, p3):
    """
    Calculates the 3D angle between three points (e.g., Hip, Knee, Ankle).
    p2 is the vertex of the angle (joint center).
    Parameters: Numpy arrays of shape (N_frames, 3) containing [X, Y, Z].
    Returns: Array (N_frames,) with the angle in degrees.
    """
    # Create vectors from vertex (p2)
    v1 = p1 - p2  # Vector Knee to Hip
    v2 = p3 - p2  # Vector Knee to Ankle
    
    # Normalize vectors (magnitude = 1)
    # axis=1 calculates the norm for each frame independently
    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
    
    # Add 1e-8 to avoid division by zero
    v1_unit = v1 / (v1_norm + 1e-8)
    v2_unit = v2 / (v2_norm + 1e-8)
    
    # Dot Product along X, Y, Z
    dot_product = np.sum(v1_unit * v2_unit, axis=1)
    
    # Prevent floating point precision errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate angle in radians and convert to degrees
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def calculate_absolute_inclination_3d(p_top, p_bottom, vertical_vector=np.array([0, 0, 1])):
    """
    Calculates the inclination of a segment (e.g., Trunk) relative to the global vertical.
    Assumes vertical_vector points upwards in your coordinate system (e.g., Z = 1).
    """
    # Segment vector (e.g., Pelvis to Thorax)
    v_seg = p_top - p_bottom
    v_seg_norm = np.linalg.norm(v_seg, axis=1, keepdims=True)
    v_seg_unit = v_seg / (v_seg_norm + 1e-8)
    
    # The vertical_vector is broadcasted to shape (N_frames, 3)
    dot_product = np.sum(v_seg_unit * vertical_vector, axis=1)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

class TUGAnalyzer:
    def __init__(self, df_3d: pd.DataFrame, fs: float):
        """
        Initializes the TUG analyzer with 3D kinematic data and sampling rate.
        :param df_3d: DataFrame containing X, Y, Z coordinates of MediaPipe markers.
        :param fs: Sampling frequency (frames per second).
        """
        self.df = df_3d.copy()
        self.fs = fs
        self.dt = 1.0 / fs if fs > 0 else 0.0

    def _get_point_3d(self, index: int) -> np.ndarray:
        """
        Extracts the time series (N, 3) for a specific MediaPipe point.
        Attempts different common column name patterns (e.g., p0_x, 0_X, p1_x for 1-indexed).
        Supports both XYZ and XY-only inputs (XY-only gets Z=0 for visualization compatibility).
        """
        if not 0 <= index < len(MEDIAPIPE_LANDMARK_NAMES):
            raise ValueError(f"MediaPipe point index out of range: {index}")

        def _resolve_columns(x_col: str, y_col: str, z_col: str | None = None) -> np.ndarray | None:
            if x_col not in self.df.columns or y_col not in self.df.columns:
                return None
            if z_col and z_col in self.df.columns:
                return self.df[[x_col, y_col, z_col]].to_numpy()
            xy = self.df[[x_col, y_col]].to_numpy()
            z = np.zeros((len(xy), 1), dtype=float)
            return np.hstack([xy, z])

        # Numeric index patterns commonly seen in legacy and vaila exports.
        # IMPORTANT: when files are 1-based (p1..p33), querying p{index} first would
        # shift all points by one (e.g., point 11 would read p11 instead of p12).
        has_p0 = any(k in self.df.columns for k in ("p0_x", "p0_X", "p0_y", "p0_Y"))
        has_p1 = any(k in self.df.columns for k in ("p1_x", "p1_X", "p1_y", "p1_Y"))
        has_p33 = any(k in self.df.columns for k in ("p33_x", "p33_X", "p33_y", "p33_Y"))

        if has_p1 and has_p33 and not has_p0:
            # Pure 1-based schema: p1..p33
            p_indices = [index + 1]
        elif has_p0 and not has_p33:
            # Pure 0-based schema: p0..p32
            p_indices = [index]
        else:
            # Ambiguous/mixed: keep both, but prefer 0-based first.
            p_indices = [index, index + 1]

        patterns = []
        for p_idx in p_indices:
            patterns.extend(
                [
                    (f"p{p_idx}_x", f"p{p_idx}_y", f"p{p_idx}_z"),
                    (f"p{p_idx}_X", f"p{p_idx}_Y", f"p{p_idx}_Z"),
                ]
            )
        patterns.extend(
            [
                (f"x_{index}", f"y_{index}", f"z_{index}"),
                (f"X_{index}", f"Y_{index}", f"Z_{index}"),
                (f"{index}_x", f"{index}_y", f"{index}_z"),
                (f"{index}_X", f"{index}_Y", f"{index}_Z"),
            ]
        )
        for px, py, pz in patterns:
            resolved = _resolve_columns(px, py, pz)
            if resolved is not None:
                return resolved

        # Landmark-name patterns (e.g., left_ankle_x, left_ankle_y, left_ankle_z).
        name = MEDIAPIPE_LANDMARK_NAMES[index]
        name_patterns = [
            (f"{name}_x", f"{name}_y", f"{name}_z"),
            (f"{name}_X", f"{name}_Y", f"{name}_Z"),
        ]
        for px, py, pz in name_patterns:
            resolved = _resolve_columns(px, py, pz)
            if resolved is not None:
                return resolved

        raise ValueError(f"Could not find X, Y, Z columns for MediaPipe point {index}.")

    def calculate_com_3d(self):
        """
        Calculates the 3D Center of Mass (CoM) using a simplified model,
        explicitly excluding hand and wrist markers:
        (MediaPipe 0-based: 15 to 22).
        This exclusion is used only for CoM estimation; skeleton drawing still uses
        the complete 33-landmark MediaPipe model and full connection set.
        
        Uses simplified Dempster-based weights:
        - Head (0): 0.081
        - Trunk (means 11,12,23,24): 0.497
        - Upper Arms (means 11,13 and 12,14): 0.056
        - Forearms (13,14 - wrists excluded): 0.044
        - Thighs (means 23,25 and 24,26): 0.200
        - Shanks (means 25,27 and 26,28): 0.093
        - Feet (means 27,29,31 and 28,30,32): 0.029
        Total: 1.0
        """
        # Extract necessary points
        p = {i: self._get_point_3d(i) for i in [0, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]}
        
        # Segments and their approximated CoMs
        head_com = p[0]
        trunk_com = (p[11] + p[12] + p[23] + p[24]) / 4.0
        upper_arm_l = (p[11] + p[13]) / 2.0
        upper_arm_r = (p[12] + p[14]) / 2.0
        forearm_l = p[13]  # Wrist excluded, using elbow as approx
        forearm_r = p[14]
        thigh_l = (p[23] + p[25]) / 2.0
        thigh_r = (p[24] + p[26]) / 2.0
        shank_l = (p[25] + p[27]) / 2.0
        shank_r = (p[26] + p[28]) / 2.0
        foot_l = (p[27] + p[29] + p[31]) / 3.0
        foot_r = (p[28] + p[30] + p[32]) / 3.0
        
        # Weights (Adapted Dempster)
        w_head = 0.081
        w_trunk = 0.497
        w_uarm = 0.028
        w_farm = 0.022
        w_thigh = 0.100
        w_shank = 0.0465
        w_foot = 0.0145
        
        com_3d = (
            head_com * w_head +
            trunk_com * w_trunk +
            upper_arm_l * w_uarm + upper_arm_r * w_uarm +
            forearm_l * w_farm + forearm_r * w_farm +
            thigh_l * w_thigh + thigh_r * w_thigh +
            shank_l * w_shank + shank_r * w_shank +
            foot_l * w_foot + foot_r * w_foot
        )
        
        self.df['CoM_x'] = com_3d[:, 0]
        self.df['CoM_y'] = com_3d[:, 1]
        self.df['CoM_z'] = com_3d[:, 2]
        
        return com_3d

    def extract_kinematics(self, vertical_vector=np.array([0, 0, 1])):
        """
        Calculates 3D joint angles and trunk inclination using dot product.
        Also calculates derivatives (velocity and acceleration) for CoM, heels and trunk.
        Assumes vertical_vector points upwards.
        """
        # Extract Numpy arrays (N, 3)
        # Right
        hip_r = self._get_point_3d(24)
        knee_r = self._get_point_3d(26)
        ankle_r = self._get_point_3d(28)
        # Left
        hip_l = self._get_point_3d(23)
        knee_l = self._get_point_3d(25)
        ankle_l = self._get_point_3d(27)
        # Trunk
        shoulder_l = self._get_point_3d(11)
        shoulder_r = self._get_point_3d(12)
        
        # Knee Angles (anatomical: 0° extended, positive flexed)
        knee_angle_r_raw = calculate_angle_3d(hip_r, knee_r, ankle_r)
        self.df['Knee_Angle_R'] = 180.0 - knee_angle_r_raw
        
        knee_angle_l_raw = calculate_angle_3d(hip_l, knee_l, ankle_l)
        self.df['Knee_Angle_L'] = 180.0 - knee_angle_l_raw
        
        # Optional: Hip and Ankle angles can be calculated similarly if we have 
        # additional points, e.g., shoulder->hip->knee
        self.df['Hip_Angle_R'] = 180.0 - calculate_angle_3d(shoulder_r, hip_r, knee_r)
        self.df['Hip_Angle_L'] = 180.0 - calculate_angle_3d(shoulder_l, hip_l, knee_l)
        
        foot_index_r = self._get_point_3d(32)
        foot_index_l = self._get_point_3d(31)
        self.df['Ankle_Angle_R'] = 180.0 - calculate_angle_3d(knee_r, ankle_r, foot_index_r)
        self.df['Ankle_Angle_L'] = 180.0 - calculate_angle_3d(knee_l, ankle_l, foot_index_l)

        # Trunk Inclination
        mid_shoulder = (shoulder_l + shoulder_r) / 2.0
        mid_hip = (hip_l + hip_r) / 2.0
        self.df['Trunk_Inclination'] = calculate_absolute_inclination_3d(mid_shoulder, mid_hip, vertical_vector)
        
        # Velocities and accelerations via np.gradient
        if not hasattr(self, 'dt') or self.dt == 0.0:
            return
            
        # Vector Coding (Coupling Angle: Hip vs Knee Sagittal)
        for side in ['R', 'L']:
            d_hip = np.gradient(self.df[f'Hip_Angle_{side}'], self.dt)
            d_knee = np.gradient(self.df[f'Knee_Angle_{side}'], self.dt)
            # Coupling Angle in degrees [0, 360)
            self.df[f'Coupling_Angle_Hip_Knee_{side}'] = np.degrees(np.arctan2(d_knee, d_hip)) % 360
            
        def calc_derivatives(series_3d, prefix):
            vel = np.gradient(series_3d, self.dt, axis=0)
            acc = np.gradient(vel, self.dt, axis=0)
            for i, axis in enumerate(['x', 'y', 'z']):
                self.df[f'{prefix}_vel_{axis}'] = vel[:, i]
                self.df[f'{prefix}_acc_{axis}'] = acc[:, i]
                
        if 'CoM_x' in self.df.columns:
            com_3d = self.df[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy()
            calc_derivatives(com_3d, 'CoM')
            
            # Extrapolated Center of Mass (XcoM) 
            omega_l = np.mean(self.df['CoM_z']) # Pendulum length approximated by CoM Z height
            omega_0 = np.sqrt(9.81 / omega_l) if omega_l > 0 else 1.0
            self.df['XcoM_x'] = self.df['CoM_x'] + self.df['CoM_vel_x'] / omega_0
            self.df['XcoM_y'] = self.df['CoM_y'] + self.df['CoM_vel_y'] / omega_0
            
        heel_l = self._get_point_3d(29)
        heel_r = self._get_point_3d(30)
        calc_derivatives(heel_l, 'Heel_L')
        calc_derivatives(heel_r, 'Heel_R')
        calc_derivatives(mid_shoulder, 'Trunk')

        # Mid Trunk (Shoulders + Hips: 11, 12, 23, 24)
        mid_trunk = (shoulder_l + shoulder_r + hip_l + hip_r) / 4.0
        self.df['Mid_Trunk_x'] = mid_trunk[:, 0]
        self.df['Mid_Trunk_y'] = mid_trunk[:, 1]
        self.df['Mid_Trunk_z'] = mid_trunk[:, 2]
        calc_derivatives(mid_trunk, 'Mid_Trunk')

        # Med Foot Right (Average of ankle=28, heel=30, toe=32)
        ankle_r = self._get_point_3d(28)
        toe_r = self._get_point_3d(32)
        med_foot_right = np.mean([ankle_r, heel_r, toe_r], axis=0)
        self.df['Med_Foot_Right_x'] = med_foot_right[:, 0]
        self.df['Med_Foot_Right_y'] = med_foot_right[:, 1]
        self.df['Med_Foot_Right_z'] = med_foot_right[:, 2]
        calc_derivatives(med_foot_right, 'Med_Foot_Right')

        # Med Foot Left (Average of ankle=27, heel=29, toe=31)
        ankle_l = self._get_point_3d(27)
        toe_l = self._get_point_3d(31)
        med_foot_left = np.mean([ankle_l, heel_l, toe_l], axis=0)
        self.df['Med_Foot_Left_x'] = med_foot_left[:, 0]
        self.df['Med_Foot_Left_y'] = med_foot_left[:, 1]
        self.df['Med_Foot_Left_z'] = med_foot_left[:, 2]
        calc_derivatives(med_foot_left, 'Med_Foot_Left')

    def detect_gait_events(self) -> dict:
        """
        Detecção de Eventos da Marcha Híbrida (Baseada em Máscara de Fase).
        - HS (Heel Strike): Mínimo local da altura vertical (Eixo Z) do calcanhar.
        - TO (Toe Off): Pico negativo da projeção do dedão em relação à pelve.
        - Aplica uma 'Walking Mask' para ignorar movimentos dos pés durante o STS, Turn e Sit.
        """
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d

        try:
            pelvis_r = self._get_point_3d(24)
            pelvis_l = self._get_point_3d(23)
            heel_r = self._get_point_3d(30)
            heel_l = self._get_point_3d(29)
            toe_r = self._get_point_3d(32)
            toe_l = self._get_point_3d(31)
        except ValueError as e:
            print(f"Erro ao obter marcadores: {e}")
            return {}

        # 1. Criar a "Walking Mask" (O indivíduo está se deslocando no espaço?)
        pelvis_center = (pelvis_r + pelvis_l) / 2.0
        pelvis_xy = pelvis_center[:, [0, 1]]
        pelvis_z = pelvis_center[:, 2]
        
        # Filtro de postura (apenas detecta passos se estiver de pé)
        z_min = np.percentile(pelvis_z, 5)
        z_max = np.percentile(pelvis_z, 95)
        is_standing_mask = pelvis_z > (z_min + 0.4 * (z_max - z_min))
        
        # Velocidade Macro da Pelve no plano XY
        pelvis_vel_xy = np.linalg.norm(
            np.gradient(gaussian_filter1d(pelvis_xy, sigma=self.fs*0.5, axis=0), self.dt, axis=0), 
            axis=1
        )
        
        # Threshold de marcha restaurado: ignorar passos minúsculos parados e no giro
        is_walking_mask = is_standing_mask & (pelvis_vel_xy > 0.25)

        # 2. Vetor de Progressão Direcional para o Toe Off
        pelvis_smooth = gaussian_filter1d(pelvis_center, sigma=self.fs * 0.5, axis=0)
        direction_vector = np.gradient(pelvis_smooth, axis=0) / (self.dt if self.dt > 0 else 1/30.0)
        dir_xy = direction_vector[:, [0, 1]]
        dir_unit = dir_xy / (np.linalg.norm(dir_xy, axis=1, keepdims=True) + 1e-8)

        def find_events(heel, toe):
            # -----------------------------------------------------------------
            # HEEL STRIKE (Mínimo da Trajetória Z - Calcanhar encosta no chão)
            # -----------------------------------------------------------------
            z_heel = gaussian_filter1d(heel[:, 2], sigma=self.fs * 0.05)
            inv_z_heel = -z_heel # Invertemos para usar find_peaks (que acha máximos)
            
            # Zerar o sinal se não estiver andando (evita detectar passos sentado/girando)
            inv_z_masked = np.where(is_walking_mask, inv_z_heel, np.min(inv_z_heel))
            
            min_dist = int(self.fs * 0.4) # Passos não ocorrem em menos de 400ms
            hs_indices, _ = find_peaks(inv_z_masked, distance=min_dist, prominence=0.04)

            # -----------------------------------------------------------------
            # TOE OFF (Zeni Modificado - Dedão mais esticado atrás da pelve)
            # -----------------------------------------------------------------
            rel_toe_xy = toe[:, [0, 1]] - pelvis_center[:, [0, 1]]
            proj_toe = np.sum(rel_toe_xy * dir_unit, axis=1)
            proj_toe_smooth = gaussian_filter1d(proj_toe, sigma=self.fs * 0.05)
            
            # Inverter a projeção (queremos o pico negativo, quando o pé tá mais atrás)
            inv_proj_toe = -proj_toe_smooth
            inv_proj_masked = np.where(is_walking_mask, inv_proj_toe, np.min(inv_proj_toe))
            
            to_indices, _ = find_peaks(inv_proj_masked, distance=min_dist, prominence=0.04)

            return hs_indices, to_indices

        # Extração das duas pernas
        hs_r, to_r = find_events(heel_r, toe_r)
        hs_l, to_l = find_events(heel_l, toe_l)

        # Atualiza a estrutura da classe
        self.gait_events = {
            'Right': {'HS': hs_r, 'TO': to_r},
            'Left':  {'HS': hs_l, 'TO': to_l}
        }
        
        # Injetar no DataFrame de saída para depuração/plotagem
        self.df['Right_HS'] = 0
        self.df['Right_TO'] = 0
        self.df['Left_HS'] = 0
        self.df['Left_TO'] = 0
        if len(hs_r) > 0: self.df.loc[hs_r, 'Right_HS'] = 1
        if len(to_r) > 0: self.df.loc[to_r, 'Right_TO'] = 1
        if len(hs_l) > 0: self.df.loc[hs_l, 'Left_HS'] = 1
        if len(to_l) > 0: self.df.loc[to_l, 'Left_TO'] = 1
        
        return self.gait_events

    def calculate_spatiotemporal_params(self, fps: float = 30.0, phases: dict = None) -> dict:
        """
        Phase 3: 3D Spatiotemporal Parameters.
        Calculates mean and SD metrics for both sides using detected gait events.
        Optionally uses phases to calculate XcoM trajectory deviations.
        Returns a dictionary with the results.
        """
        if not hasattr(self, 'gait_events'):
            self.detect_gait_events()
            
        try:
            heel_r = self._get_point_3d(30)[:, [0, 1]]  # XY plane
            heel_l = self._get_point_3d(29)[:, [0, 1]]
        except ValueError:
            return {}

        stats = {'Right': {}, 'Left': {}}
        dt = 1.0 / fps

        for side, opp_side, heel, opp_heel in [('Right', 'Left', heel_r, heel_l), ('Left', 'Right', heel_l, heel_r)]:
            hs_idx = self.gait_events[side]['HS']
            to_idx = self.gait_events[side]['TO']
            opp_hs_idx = self.gait_events[opp_side]['HS']
            
            step_metrics = {str(int(hs)): {} for hs in hs_idx}
            
            # Stride Length (same foot successive)
            stride_lengths = []
            for i in range(len(hs_idx) - 1):
                p1 = heel[hs_idx[i]]
                p2 = heel[hs_idx[i+1]]
                val = float(np.linalg.norm(p2 - p1))
                stride_lengths.append(val)
                step_metrics[str(int(hs_idx[i]))]['Stride_Length_m'] = val
                
            # Step Length (opposite foot to this foot)
            step_lengths = []
            for hs in hs_idx:
                # Finds the last HS of the opposite foot before this HS
                prev_opp = opp_hs_idx[opp_hs_idx < hs]
                if len(prev_opp) > 0:
                    p1 = opp_heel[prev_opp[-1]]
                    p2 = heel[hs]
                val = float(np.linalg.norm(p2 - p1))
                step_lengths.append(val)
                step_metrics[str(int(hs))]['Step_Length_m'] = val

            # Step Width (lateral distance during double support - at HS)
            # Approximation by 2D distance to opposite heel at current HS
            step_widths = []
            for hs in hs_idx:
                p1 = heel[hs]
                p2 = opp_heel[hs]
                val = float(np.linalg.norm(p2 - p1))
                step_widths.append(val)
                step_metrics[str(int(hs))]['Step_Width_m'] = val
                
            # Times (Stance and Swing)
            stance_times = []
            swing_times = []
            
            # Stance: HS to next TO of the same foot
            for hs in hs_idx:
                next_to = to_idx[to_idx > hs]
                if len(next_to) > 0:
                    val = float((next_to[0] - hs) * dt)
                    stance_times.append(val)
                    step_metrics[str(int(hs))]['Stance_Time_s'] = val
                    
            # Swing: TO to next HS of the same foot
            for to in to_idx:
                next_hs = hs_idx[hs_idx > to]
                if len(next_hs) > 0:
                    val = float((next_hs[0] - to) * dt)
                    swing_times.append(val)
                    # Assign swing time to the HS that preceded this TO (the start of this gait cycle)
                    prev_hs = hs_idx[hs_idx < to]
                    if len(prev_hs) > 0:
                        step_metrics[str(int(prev_hs[-1]))]['Swing_Time_s'] = val

            # Cadence (steps per minute) - approximation using all steps
            # We will calculate globally later

            stats[side] = {
                'Stride_Length_m': np.mean(stride_lengths) if stride_lengths else 0,
                'Stride_Length_sd': np.std(stride_lengths) if stride_lengths else 0,
                'Step_Length_m': np.mean(step_lengths) if step_lengths else 0,
                'Step_Length_sd': np.std(step_lengths) if step_lengths else 0,
                'Step_Width_m': np.mean(step_widths) if step_widths else 0,
                'Step_Width_sd': np.std(step_widths) if step_widths else 0,
                'Stance_Time_s': np.mean(stance_times) if stance_times else 0,
                'Stance_Time_sd': np.std(stance_times) if stance_times else 0,
                'Swing_Time_s': np.mean(swing_times) if swing_times else 0,
                'Swing_Time_sd': np.std(swing_times) if swing_times else 0,
                'per_step': step_metrics
            }

        # Global Cadence
        all_hs = np.sort(np.concatenate([self.gait_events['Right']['HS'], self.gait_events['Left']['HS']]))
        if len(all_hs) > 1:
            total_time_min = (all_hs[-1] - all_hs[0]) * dt / 60.0
            cadence = len(all_hs) / total_time_min if total_time_min > 0 else 0
        else:
            cadence = 0
            
        # Average Velocity
        if 'CoM_x' in self.df.columns and len(all_hs) > 1:
            com = self.df[['CoM_x', 'CoM_y']].to_numpy() # XY plane
            total_dist = 0
            for i in range(all_hs[0], all_hs[-1]):
                total_dist += np.linalg.norm(com[i+1] - com[i])
            velocity = total_dist / (total_time_min * 60) if total_time_min > 0 else 0
        else:
            velocity = 0

        # Balance Metrics (XcoM Path Deviation)
        xcom_dev_fwd = 0.0
        xcom_dev_bwd = 0.0
        if phases and 'XcoM_x' in self.df.columns and 'XcoM_y' in self.df.columns:
            def calc_dev(start_s, end_s):
                if end_s <= start_s: return 0.0
                s_idx, e_idx = int(start_s * fps), int(end_s * fps)
                if s_idx >= len(self.df) or e_idx >= len(self.df) or s_idx == e_idx: return 0.0
                path = self.df[['XcoM_x', 'XcoM_y']].iloc[s_idx:e_idx].to_numpy()
                if len(path) < 2: return 0.0
                p1, p2 = path[0], path[-1]
                line_vec = p2 - p1
                line_len = np.linalg.norm(line_vec)
                if line_len == 0: return 0.0
                cross_prod = np.abs(np.cross(line_vec, p1 - path))
                return float(np.mean(cross_prod / line_len))
                
            xcom_dev_fwd = calc_dev(*phases.get('gait_forward', (0,0)))
            xcom_dev_bwd = calc_dev(*phases.get('gait_back', (0,0)))

        stats['Global'] = {
            'Cadence_steps_per_min': cadence,
            'Velocity_m_s': velocity,
            'XcoM_Deviation_Fwd_m': xcom_dev_fwd,
            'XcoM_Deviation_Bwd_m': xcom_dev_bwd
        }
        
        self.spatiotemporal_params = stats
        return stats

    def calculate_anatomical_frames(self):
        """
        Phase 7: Anatomical Coordinate System.
        Z = norm(mid_shoulder - CoM)
        X_temp = (p13 - p12)
        Y = norm(cross(Z, X_temp))
        X = norm(cross(Y, Z))
        """
        try:
            sh_r = self._get_point_3d(12)
            sh_l = self._get_point_3d(13)
            com = self.df[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy()
        except ValueError:
            return None, None, None
            
        mid_shoulder = (sh_r + sh_l) / 2.0
        
        # Z (Body vertical)
        Z = mid_shoulder - com
        Z_norm = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        Z = Z / Z_norm
        
        # X_temp (Temporary Medio-Lateral: L - R = points to the Left)
        X_temp = sh_l - sh_r
        X_temp_norm = np.linalg.norm(X_temp, axis=1, keepdims=True) + 1e-8
        X_temp = X_temp / X_temp_norm
        
        # Y (Anteroposterior)
        Y = np.cross(Z, X_temp, axis=1)
        Y_norm = np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8
        Y = Y / Y_norm
        
        # X (Final Orthogonal Medio-Lateral)
        X = np.cross(Y, Z, axis=1)
        X_norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X = X / X_norm
        
        self.anatomical_X = X
        self.anatomical_Y = Y
        self.anatomical_Z = Z
        return X, Y, Z

    def segment_tug_phases(self) -> dict:
        """
        Phase 5: TUG Segmentation (Sit-to-Stand, Walk, Turn, Stand-to-Sit).
        Based on the functional amplitude (percentiles) of the CoM Z trajectory
        and the velocity orientation in the XY plane.
        """
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks
        
        if 'CoM_y' not in self.df.columns or 'CoM_x' not in self.df.columns or 'CoM_z' not in self.df.columns:
            return {}
            
        com_y = self.df['CoM_y'].to_numpy()
        com_x = self.df['CoM_x'].to_numpy()
        com_z = self.df['CoM_z'].to_numpy()

        N = len(com_z)
        dt = self.dt if hasattr(self, 'dt') and self.dt > 0 else 1 / 30.0
        fs = self.fs if hasattr(self, 'fs') and self.fs > 0 else 30.0

        # Allow TOML metadata to override the spatial protocol constants
        y_chair = float(getattr(self, '_meta_y_chair', Y_CHAIR_THRESHOLD))
        y_turn  = float(getattr(self, '_meta_y_turn',  Y_TURN_APPROX))
        y_tol   = float(getattr(self, '_meta_y_tol',   Y_TURN_TOLERANCE))

        # ---------------------------------------------------------
        # 1. Vertical Phase: Sit-to-Stand and Stand-to-Sit (CoM-Z)
        # ---------------------------------------------------------
        com_z_smooth = gaussian_filter1d(com_z, sigma=fs * 0.2)

        sitting_z  = np.percentile(com_z_smooth, 5)    # lowest stable height
        standing_z = np.percentile(com_z_smooth, 95)   # highest stable height
        amplitude_z = standing_z - sitting_z

        thresh_start = sitting_z + 0.10 * amplitude_z
        thresh_end   = sitting_z + 0.90 * amplitude_z

        is_standing   = com_z_smooth > (sitting_z + 0.50 * amplitude_z)
        stand_indices = np.where(is_standing)[0]

        if len(stand_indices) < int(fs):  # must stand for ≥ 1 s
            print("Warning: Standing phase not clearly detected or too short.")
            sts_start = sts_end = 0
            sit_start = sit_end = N - 1
        else:
            first_stand_idx = stand_indices[0]
            last_stand_idx  = stand_indices[-1]

            sts_start = first_stand_idx
            while sts_start > 0 and com_z_smooth[sts_start] > thresh_start:
                sts_start -= 1

            sts_end = first_stand_idx
            while sts_end < N and com_z_smooth[sts_end] < thresh_end:
                sts_end += 1

            sit_start = last_stand_idx
            while sit_start > 0 and com_z_smooth[sit_start] < thresh_end:
                sit_start -= 1

            sit_end = last_stand_idx
            while sit_end < N - 1 and com_z_smooth[sit_end] > thresh_start:
                sit_end += 1

        # ---------------------------------------------------------
        # 2. Horizontal Phases — primary: Y-axis spatial thresholds
        #    (LaBioCoM protocol: chair @ y_chair m, turn zone @ y_turn ± y_tol m)
        # ---------------------------------------------------------
        com_y_smooth = gaussian_filter1d(com_y, sigma=fs * 0.3)

        # ── Phase-window to search (between end of STS and start of SitToSit) ──
        search_start = sts_end
        search_end   = sit_start if sit_start > sts_end else N - 1

        com_y_window = com_y_smooth[search_start:search_end]
        t_window     = np.arange(len(com_y_window))

        # ── 2a. Forward gait start: first frame where Y crosses y_chair ─────────
        fwd_start_local = 0
        for i, y_val in enumerate(com_y_window):
            if y_val >= y_chair:
                fwd_start_local = i
                break
        fwd_start = search_start + fwd_start_local

        # ── 2b. Turn zone: frame of maximum Y (peak AP progression) ─────────────
        y_peak_local = int(np.argmax(com_y_window))
        y_peak_abs   = search_start + y_peak_local
        y_peak_val   = com_y_smooth[y_peak_abs]

        # Validate: peak must be inside the expected turn zone; fallback to
        # closest-to-y_turn frame if not.
        if abs(y_peak_val - y_turn) > y_tol * 2.0:
            print(f"Warning: Y peak ({y_peak_val:.2f} m) far from expected turn zone "
                  f"({y_turn} ± {y_tol} m). Using closest frame instead.")
            dist_to_turn = np.abs(com_y_window - y_turn)
            y_peak_local = int(np.argmin(dist_to_turn))
            y_peak_abs   = search_start + y_peak_local

        # ── 2c. Stop_5s anchor: flattest speed window near the Y peak ───────────
        # Use trunk speed (CoM XY) to find the speed plateau — same method as vaila_tugtunb.py
        from scipy.ndimage import uniform_filter1d, gaussian_filter1d as _gf1d
        com_vx_s = np.gradient(_gf1d(com_x, sigma=fs * 0.5)) / dt
        com_vy_s = np.gradient(_gf1d(com_y, sigma=fs * 0.5)) / dt

        if 'Mid_Trunk_vel_y' in self.df.columns and 'Mid_Trunk_vel_x' in self.df.columns:
            mid_trunk_vy = _gf1d(self.df['Mid_Trunk_vel_y'].to_numpy(), sigma=fs * 0.5)
            mid_trunk_vx = _gf1d(self.df['Mid_Trunk_vel_x'].to_numpy(), sigma=fs * 0.5)
            trunk_speed_xy = np.sqrt(mid_trunk_vx**2 + mid_trunk_vy**2)
        else:
            trunk_speed_xy = np.sqrt(com_vx_s**2 + com_vy_s**2)
            mid_trunk_vy = com_vy_s

        # Search for the speed plateau within the Y-spatial search window
        spd_window = trunk_speed_xy[search_start:search_end]
        win_len = min(int(fs * 3.5), len(spd_window))
        if win_len > 0:
            spd_smoothed   = uniform_filter1d(spd_window, size=win_len)
            anchor_local   = int(np.argmin(spd_smoothed))
            anchor_idx     = search_start + anchor_local
        else:
            anchor_idx = y_peak_abs  # fallback

        speed_thresh = 0.15

        # fwd_stop: walk backwards from anchor until speed > thresh (subject still moving)
        stop_search = anchor_idx
        while stop_search > sts_end and (
            mid_trunk_vy[stop_search] < speed_thresh
            and trunk_speed_xy[stop_search] < speed_thresh
        ):
            stop_search -= 1
        fwd_stop = max(sts_end, stop_search)

        # ── 2d. Turn180 boundaries — shoulder yaw within Y-spatial window ─────────
        try:
            shoulder_l = self._get_point_3d(11)
            shoulder_r = self._get_point_3d(12)
            sh_vec_xy  = shoulder_r[:, [0, 1]] - shoulder_l[:, [0, 1]]
            sh_angle   = np.unwrap(np.arctan2(sh_vec_xy[:, 1], sh_vec_xy[:, 0]))
            sh_angle_s = gaussian_filter1d(sh_angle, sigma=fs * 0.2)
            sh_yaw_rate = np.abs(np.gradient(sh_angle_s) / dt)
        except ValueError:
            sh_yaw_rate = np.zeros(N)

        # Search for the yaw peak *after* the speed anchor (the pause comes first, then the turn)
        turn_search_start = anchor_idx
        turn_search_end   = min(anchor_idx + int(fs * 15), search_end)

        # Narrow the search window: stop once CoM_Y velocity is strongly negative (walking back)
        for i in range(anchor_idx, turn_search_end):
            if mid_trunk_vy[i] < -0.2:
                turn_search_end = min(turn_search_end, i + int(fs * 1.5))
                break

        if turn_search_end > turn_search_start:
            zone_yaw   = sh_yaw_rate[turn_search_start:turn_search_end]
            max_yaw    = np.max(zone_yaw) if len(zone_yaw) > 0 else 0.0
            yaw_thresh = max(0.15, max_yaw * 0.15)
            peak_turn_idx = turn_search_start + int(np.argmax(zone_yaw))
        else:
            peak_turn_idx = y_peak_abs
            yaw_thresh    = 0.15
            max_yaw       = 0.0

        # turn_start: walk backwards from yaw peak until rate < threshold
        turn_start = peak_turn_idx
        while turn_start > fwd_stop and sh_yaw_rate[turn_start] > yaw_thresh:
            turn_start -= 1

        # turn_end: walk forwards from yaw peak until rate < threshold
        turn_end = peak_turn_idx
        while turn_end < search_end and sh_yaw_rate[turn_end] > yaw_thresh:
            turn_end += 1

        # ── 2e. Back gait: start where CoM_Y velocity is consistently negative ────
        bck_start = turn_end
        while bck_start < search_end and mid_trunk_vy[bck_start] > -speed_thresh:
            bck_start += 1

        if bck_start >= search_end:
            bck_start = turn_end
        turn_end = bck_start  # tie turn_end to back-gait start (contiguous)

        # Back gait ends where CoM-Y crosses back below y_chair
        bck_end = sit_start
        for i in range(bck_start, search_end):
            if com_y_smooth[i] <= y_chair:
                bck_end = i
                break

        sit_start = max(bck_end, sit_start)

        # ── Sanity: enforce contiguous, non-negative durations ───────────────────
        fwd_stop   = max(fwd_stop,  sts_end)
        turn_start = max(turn_start, fwd_stop)
        turn_end   = max(turn_end,   turn_start)
        bck_start  = max(bck_start,  turn_end)
        sit_start  = max(sit_start,  bck_start)
        sit_end    = max(sit_end,    sit_start)

        stop_start = fwd_stop
        stop_end   = turn_start


        # ---------------------------------------------------------
        # 3. Packaging Results
        # ---------------------------------------------------------
        phases = {
            'stand':        (sts_start * dt,  sts_end * dt),
            'gait_forward': (sts_end * dt,    fwd_stop * dt),
            'stop_5s':      (fwd_stop * dt,   turn_start * dt) if turn_start > fwd_stop else None,
            'turn180':      (turn_start * dt,  turn_end * dt),
            'gait_back':    (bck_start * dt,   sit_start * dt),
            'sit':          (sit_start * dt,   sit_end * dt),
            'Total_TUG_Time': N * dt,
        }

        if phases['stop_5s'] is None:
            phases.pop('stop_5s')

        # Turn direction (cross-product of anatomical Y-axis frames)
        X_frame, Y_frame, Z_frame = self.calculate_anatomical_frames()
        if X_frame is not None and turn_start < turn_end:
            y_s = Y_frame[turn_start, [0, 1]]
            y_e = Y_frame[turn_end,   [0, 1]]
            cross_val = y_s[0] * y_e[1] - y_s[1] * y_e[0]
            phases['Turn_Direction'] = "Right" if cross_val < 0 else "Left"

        self.tug_phases = phases
        return phases


def generate_plotly_report(analyzer: TUGAnalyzer, out_dir: Path, name: str, fps: float, report_data: dict):
    """
    Generates a dynamic visual report with interactive Plotly plots,
    including XYZ overlay with gait events and sagittal stick figures.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    com = analyzer.df[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy()
    t = np.arange(len(com)) / fps
    skeleton_connections = load_mediapipe_pose_connections()
    required_points = sorted({idx for a, b in skeleton_connections for idx in (a, b)})

    skeleton_series = {}
    for point_idx in required_points:
        try:
            skeleton_series[point_idx] = analyzer._get_point_3d(point_idx)
        except ValueError:
            continue
    
    plots_html = []
    
    def add_gait_events(fig, row=None, col=None):
        for leg, color in [('Right', 'red'), ('Left', 'green')]:
            for ev, ls in [('HS', 'dash'), ('TO', 'dot')]:
                frames = analyzer.gait_events.get(leg, {}).get(ev, [])
                for f in frames:
                    time = f / fps
                    if row is not None and col is not None:
                        fig.add_vline(x=time, line_dash=ls, line_color=color, opacity=0.6, row=row, col=col)
                    else:
                        fig.add_vline(x=time, line_dash=ls, line_color=color, opacity=0.6)

    # --- Plot 1: CoM XYZ vs Time + Gait Events ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=com[:, 0], mode='lines', name='X (Mediolateral)', line=dict(color='blue', width=2)))
    fig1.add_trace(go.Scatter(x=t, y=com[:, 1], mode='lines', name='Y (Anteroposterior)', line=dict(color='orange', width=2)))
    fig1.add_trace(go.Scatter(x=t, y=com[:, 2], mode='lines', name='Z (Vertical)', line=dict(color='purple', width=2)))
    add_gait_events(fig1)
    
    fig1.update_layout(title='CoM Trajectory (X, Y, Z) vs Time + Gait Events (Red=Right, Green=Left | Dashed=HS, Dotted=TO)', 
                       xaxis_title='Time (s)', yaxis_title='Position (m)', height=500, template='plotly_white')
    plots_html.append(fig1.to_html(full_html=False, include_plotlyjs=False))
    
    # --- Plot 2: Split XYZ ---
    fig2 = make_subplots(rows=1, cols=3, subplot_titles=('CoM X (Mediolateral)', 'CoM Y (Anteroposterior)', 'CoM Z (Vertical)'))
    fig2.add_trace(go.Scatter(x=t, y=com[:, 0], mode='lines', line=dict(color='blue'), showlegend=False), row=1, col=1)
    fig2.add_trace(go.Scatter(x=t, y=com[:, 1], mode='lines', line=dict(color='orange'), showlegend=False), row=1, col=2)
    fig2.add_trace(go.Scatter(x=t, y=com[:, 2], mode='lines', line=dict(color='purple'), showlegend=False), row=1, col=3)
    fig2.update_layout(height=400, template='plotly_white')
    plots_html.append(fig2.to_html(full_html=False, include_plotlyjs=False))
    
    # --- Plot 3: Sagittal Plane Stick Figures ---
    def add_stick_figures(fig, frames, row, col, max_frames=12):
        for f in sample_frames(frames, max_frames=max_frames):
            for a_idx, b_idx in skeleton_connections:
                if a_idx not in skeleton_series or b_idx not in skeleton_series:
                    continue
                pa = skeleton_series[a_idx][f]
                pb = skeleton_series[b_idx][f]
                if np.isnan(pa).any() or np.isnan(pb).any():
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[pa[1], pb[1]],
                        y=[pa[2], pb[2]],
                        mode='lines',
                        line=dict(color=get_connection_color(a_idx, b_idx), width=2),
                        opacity=0.45,
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

    events = []
    for leg in ['Right', 'Left']:
        for ev in ['HS', 'TO']:
            for f in analyzer.gait_events.get(leg, {}).get(ev, []):
                events.append(int(round(f)))
    events.sort()
    
    phases = report_data.get('Phases_Seconds', {})
    wf_s = phases.get('gait_forward', (0,0))[0]
    wf_e = phases.get('gait_forward', (0,0))[1]
    wb_s = phases.get('gait_back', (0,0))[0]
    wb_e = phases.get('gait_back', (0,0))[1]
    
    fig3 = make_subplots(rows=1, cols=2, subplot_titles=('Walk Forward Sagittal Events (Y-Z)', 'Walk Back Sagittal Events (Y-Z)'))
    add_stick_figures(fig3, [f for f in events if wf_s*fps <= f <= wf_e*fps], 1, 1)
    add_stick_figures(fig3, [f for f in events if wb_s*fps <= f <= wb_e*fps], 1, 2)
    fig3.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig3.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
    fig3.update_layout(height=500, template='plotly_white')
    plots_html.append(fig3.to_html(full_html=False, include_plotlyjs=False))
    
    # --- Plot 4: 3D Equal Aspect Plot ---
    fig4 = go.Figure()
    phase_colors = {
        'stand': 'gray', 'gait_forward': 'blue', 'stop_5s': 'brown',
        'turn180': 'orange', 'gait_back': 'green', 'sit': 'purple'
    }
    
    last_idx = 0
    legend_added = set()
    for phase_name, val in ordered_phase_ranges(phases):
        start_s, end_s = val
        s_idx, e_idx = int(start_s * fps), int(end_s * fps)
        color = phase_colors.get(phase_name, 'darkred')
        show_leg = phase_name not in legend_added
        legend_added.add(phase_name)
        
        fig4.add_trace(go.Scatter3d(
            x=com[s_idx:e_idx+1, 0], y=com[s_idx:e_idx+1, 1], z=com[s_idx:e_idx+1, 2],
            mode='lines', line=dict(color=color, width=4), name=phase_name.replace('_', ' '), showlegend=show_leg
        ))
        last_idx = max(last_idx, e_idx)

    event_frames_3d = sample_frames(events, max_frames=10)
    for f in event_frames_3d:
        for a_idx, b_idx in skeleton_connections:
            if a_idx not in skeleton_series or b_idx not in skeleton_series:
                continue
            pa = skeleton_series[a_idx][f]
            pb = skeleton_series[b_idx][f]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            fig4.add_trace(
                go.Scatter3d(
                    x=[pa[0], pb[0]],
                    y=[pa[1], pb[1]],
                    z=[pa[2], pb[2]],
                    mode='lines',
                    line=dict(color=get_connection_color(a_idx, b_idx), width=2),
                    opacity=0.35,
                    showlegend=False,
                )
            )

    if last_idx < len(com) - 1:
        fig4.add_trace(go.Scatter3d(
            x=com[last_idx:, 0], y=com[last_idx:, 1], z=com[last_idx:, 2],
            mode='lines', line=dict(color='black', width=2, dash='dot'), name='Remainder'
        ))

    fig4.add_trace(go.Scatter3d(x=[com[0, 0]], y=[com[0, 1]], z=[com[0, 2]], mode='markers', marker=dict(size=8, color='green'), name='Start'))
    fig4.add_trace(go.Scatter3d(x=[com[-1, 0]], y=[com[-1, 1]], z=[com[-1, 2]], mode='markers', marker=dict(size=8, color='red'), name='End'))
    max_range = np.array([com[:,0].max()-com[:,0].min(), com[:,1].max()-com[:,1].min(), com[:,2].max()-com[:,2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = (com[:,0].max()+com[:,0].min())*0.5, (com[:,1].max()+com[:,1].min())*0.5, (com[:,2].max()+com[:,2].min())*0.5
    fig4.update_layout(
        title='3D CoM Trajectory (Colored by Phase)',
        scene=dict(
            xaxis_title='X (ML)', yaxis_title='Y (AP)', zaxis_title='Z (Vert)',
            xaxis=dict(range=[mid_x - max_range, mid_x + max_range]),
            yaxis=dict(range=[mid_y - max_range, mid_y + max_range]),
            zaxis=dict(range=[mid_z - max_range, mid_z + max_range]),
            aspectmode='cube'
        ),
        height=800, template='plotly_white'
    )
    plots_html.append(fig4.to_html(full_html=False, include_plotlyjs=False))

    # --- Plot 5: Mean Foot Sagittal View (YZ) - Progressive vs Regressive ---
    fig5 = go.Figure()
    
    for side, prefix, c_path, c_fwd, c_bwd, marker_hs, marker_to in [
        ('Right', 'Med_Foot_Right', 'lightsalmon', 'darkred', 'lightcoral', 'circle', 'square'),
        ('Left', 'Med_Foot_Left', 'lightgreen', 'darkgreen', 'palegreen', 'circle', 'square')
    ]:
        y = analyzer.df[f'{prefix}_y'].to_numpy()
        z = analyzer.df[f'{prefix}_z'].to_numpy()
        vy = analyzer.df[f'{prefix}_vel_y'].to_numpy()
        
        fig5.add_trace(go.Scatter(x=y, y=z, mode='lines', line=dict(color=c_path), opacity=0.4, name=f'{side} Path'))
        
        fwd_mask = vy > 0
        bwd_mask = vy <= 0
        fig5.add_trace(go.Scatter(x=y[fwd_mask], y=z[fwd_mask], mode='markers', marker=dict(color=c_fwd, size=4), name=f'{side} Fwd (Vy > 0)'))
        fig5.add_trace(go.Scatter(x=y[bwd_mask], y=z[bwd_mask], mode='markers', marker=dict(color=c_bwd, size=4, symbol='x'), name=f'{side} Bwd (Vy <= 0)'))
        
        # Add Gait Events in 5
        hs_idx = analyzer.gait_events.get(side, {}).get('HS', [])
        to_idx = analyzer.gait_events.get(side, {}).get('TO', [])
        if len(hs_idx) > 0:
            fig5.add_trace(go.Scatter(x=y[hs_idx], y=z[hs_idx], mode='markers', marker=dict(color=c_fwd, size=10, symbol=marker_hs, line=dict(width=2, color='black')), name=f'{side} HS'))
        if len(to_idx) > 0:
            fig5.add_trace(go.Scatter(x=y[to_idx], y=z[to_idx], mode='markers', marker=dict(color=c_bwd, size=10, symbol=marker_to, line=dict(width=2, color='black')), name=f'{side} TO'))

    fig5.update_layout(title='Mean Foot Sagittal View (YZ) - Progressive vs Regressive w/ Gait Events', xaxis_title='Y (AP)', yaxis_title='Z (Vert)', height=600, template='plotly_white')
    plots_html.append(fig5.to_html(full_html=False, include_plotlyjs=False))
    
    # --- Plot 6: Mean Foot Kinematics (Split Y and Z Position/Velocity/Acceleration) vs Time ---
    fig6 = make_subplots(rows=5, cols=2, shared_xaxes=True,
                         subplot_titles=[
                             'Right Position Y', 'Left Position Y',
                             'Right Position Z', 'Left Position Z',
                             'Right Velocity Y & Z', 'Left Velocity Y & Z',
                             'Right Acceleration Y & Z', 'Left Acceleration Y & Z',
                             'Right 2D Kin Magnitude', 'Left 2D Kin Magnitude'
                         ], vertical_spacing=0.03)
    
    for side, prefix, col in [('Right', 'Med_Foot_Right', 1), ('Left', 'Med_Foot_Left', 2)]:
        y = analyzer.df[f'{prefix}_y'].to_numpy()
        z = analyzer.df[f'{prefix}_z'].to_numpy()
        vy = analyzer.df[f'{prefix}_vel_y'].to_numpy()
        vz = analyzer.df[f'{prefix}_vel_z'].to_numpy()
        ay = analyzer.df[f'{prefix}_acc_y'].to_numpy()
        az = analyzer.df[f'{prefix}_acc_z'].to_numpy()
        
        v_yz = np.sqrt(vy**2 + vz**2)
        a_yz = np.sqrt(ay**2 + az**2)
        
        # Pos Y & Z Separate Traces
        fig6.add_trace(go.Scatter(x=t, y=y, name=f'{side} Y', line=dict(color='orange')), row=1, col=col)
        fig6.add_trace(go.Scatter(x=t, y=z, name=f'{side} Z', line=dict(color='purple')), row=2, col=col)
        
        fig6.add_trace(go.Scatter(x=t, y=vy, name=f'{side} Vy', line=dict(color='orange')), row=3, col=col)
        fig6.add_trace(go.Scatter(x=t, y=vz, name=f'{side} Vz', line=dict(color='purple')), row=3, col=col)
        fig6.add_trace(go.Scatter(x=[t[0], t[-1]], y=[0, 0], mode='lines', line=dict(color='black', dash='dash'), showlegend=False), row=3, col=col)
        
        fig6.add_trace(go.Scatter(x=t, y=ay, name=f'{side} Ay', line=dict(color='orange')), row=4, col=col)
        fig6.add_trace(go.Scatter(x=t, y=az, name=f'{side} Az', line=dict(color='purple')), row=4, col=col)
        fig6.add_trace(go.Scatter(x=[t[0], t[-1]], y=[0, 0], mode='lines', line=dict(color='black', dash='dash'), showlegend=False), row=4, col=col)
        
        fig6.add_trace(go.Scatter(x=t, y=v_yz, name=f'{side} |V_yz|', line=dict(color='blue')), row=5, col=col)
        fig6.add_trace(go.Scatter(x=t, y=a_yz, name=f'{side} |A_yz|', line=dict(color='red', dash='dot')), row=5, col=col)
        
        # Add gait events
        for ev, ls in [('HS', 'dash'), ('TO', 'dot')]:
            frames = analyzer.gait_events.get(side, {}).get(ev, [])
            for f in frames:
                time = f / fps
                fig6.add_vline(x=time, line_dash=ls, line_color='black', opacity=0.5, row=1, col=col)
                fig6.add_vline(x=time, line_dash=ls, line_color='black', opacity=0.5, row=2, col=col)

    fig6.update_layout(title='Mean Foot Kinematics Separated', height=1200, template='plotly_white')
    plots_html.append(fig6.to_html(full_html=False, include_plotlyjs=False))

    # --- Plot 7: Mid Trunk Sagittal View (YZ) ---
    fig7 = go.Figure()
    trunk_y = analyzer.df['Mid_Trunk_y'].to_numpy()
    trunk_z = analyzer.df['Mid_Trunk_z'].to_numpy()
    trunk_vy = analyzer.df['Mid_Trunk_vel_y'].to_numpy()
    
    fig7.add_trace(go.Scatter(x=trunk_y, y=trunk_z, mode='lines', line=dict(color='purple'), opacity=0.4, name='Mid Trunk Path'))
    fig7.add_trace(go.Scatter(x=trunk_y[trunk_vy > 0], y=trunk_z[trunk_vy > 0], mode='markers', marker=dict(color='darkmagenta', size=4), name='Fwd (Vy > 0)'))
    fig7.add_trace(go.Scatter(x=trunk_y[trunk_vy <= 0], y=trunk_z[trunk_vy <= 0], mode='markers', marker=dict(color='violet', size=4, symbol='x'), name='Bwd (Vy <= 0)'))
    
    for leg, color_hs, color_to in [('Right', 'darkred', 'lightcoral'), ('Left', 'darkgreen', 'palegreen')]:
        hs_idx = analyzer.gait_events.get(leg, {}).get('HS', [])
        to_idx = analyzer.gait_events.get(leg, {}).get('TO', [])
        if len(hs_idx) > 0:
            fig7.add_trace(go.Scatter(x=trunk_y[hs_idx], y=trunk_z[hs_idx], mode='markers', marker=dict(color=color_hs, size=10, symbol='circle', line=dict(width=2, color='black')), name=f'{leg} HS (Trunk)'))
        if len(to_idx) > 0:
            fig7.add_trace(go.Scatter(x=trunk_y[to_idx], y=trunk_z[to_idx], mode='markers', marker=dict(color=color_to, size=10, symbol='square', line=dict(width=2, color='black')), name=f'{leg} TO (Trunk)'))

    fig7.update_layout(title='Mid Trunk Sagittal View (YZ) w/ Gait Events', xaxis_title='Y (AP)', yaxis_title='Z (Vert)', height=600, template='plotly_white')
    plots_html.append(fig7.to_html(full_html=False, include_plotlyjs=False))
    
    # --- Plot 8: Mid Trunk Kinematics Separated ---
    fig8 = make_subplots(rows=5, cols=1, shared_xaxes=True,
                         subplot_titles=[
                             'Trunk Position Y', 'Trunk Position Z', 'Trunk Velocity Y & Z',
                             'Trunk Acceleration Y & Z', 'Trunk 2D Kin Magnitude'
                         ], vertical_spacing=0.03)
    
    vy_trunk = analyzer.df['Mid_Trunk_vel_y'].to_numpy()
    vz_trunk = analyzer.df['Mid_Trunk_vel_z'].to_numpy()
    ay_trunk = analyzer.df['Mid_Trunk_acc_y'].to_numpy()
    az_trunk = analyzer.df['Mid_Trunk_acc_z'].to_numpy()
    v_yz_trunk = np.sqrt(vy_trunk**2 + vz_trunk**2)
    a_yz_trunk = np.sqrt(ay_trunk**2 + az_trunk**2)
    
    fig8.add_trace(go.Scatter(x=t, y=trunk_y, name='Y (AP)', line=dict(color='orange')), row=1, col=1)
    fig8.add_trace(go.Scatter(x=t, y=trunk_z, name='Z (Vert)', line=dict(color='purple')), row=2, col=1)
    
    fig8.add_trace(go.Scatter(x=t, y=vy_trunk, name='Vy', line=dict(color='orange')), row=3, col=1)
    fig8.add_trace(go.Scatter(x=t, y=vz_trunk, name='Vz', line=dict(color='purple')), row=3, col=1)
    fig8.add_trace(go.Scatter(x=[t[0], t[-1]], y=[0, 0], mode='lines', line=dict(color='black', dash='dash'), showlegend=False), row=3, col=1)
    
    fig8.add_trace(go.Scatter(x=t, y=ay_trunk, name='Ay', line=dict(color='orange')), row=4, col=1)
    fig8.add_trace(go.Scatter(x=t, y=az_trunk, name='Az', line=dict(color='purple')), row=4, col=1)
    fig8.add_trace(go.Scatter(x=[t[0], t[-1]], y=[0, 0], mode='lines', line=dict(color='black', dash='dash'), showlegend=False), row=4, col=1)
    
    fig8.add_trace(go.Scatter(x=t, y=v_yz_trunk, name='|V_yz|', line=dict(color='blue')), row=5, col=1)
    fig8.add_trace(go.Scatter(x=t, y=a_yz_trunk, name='|A_yz|', line=dict(color='red', dash='dot')), row=5, col=1)
    
    add_gait_events(fig8, row=1, col=1)
    add_gait_events(fig8, row=2, col=1)
    
    fig8.update_layout(title='Mid Trunk Kinematics Separated', height=1200, template='plotly_white')
    plots_html.append(fig8.to_html(full_html=False, include_plotlyjs=False))

    # ── Vector Coding: Timelines and Donut Charts ────────────────────────────
    vc_summary = report_data.get('VC_Summary', {})
    
    # Colour map for the 4 patterns
    _vc_colours = {
        'In_Phase':        'rgba(90, 110, 250, 0.85)',   # Per screenshot: In-Phase is blue
        'Anti_Phase':      'rgba(0, 200, 150, 0.85)',    # Anti-Phase is green
        'Proximal_Phase':  'rgba(240, 80, 60, 0.85)',    # Proximal is red
        'Distal_Phase':    'rgba(170, 80, 250, 0.85)',   # Distal is purple
    }
    _vc_labels = {
        'In_Phase':       'In_Phase',
        'Anti_Phase':     'Anti_Phase',
        'Proximal_Phase': 'Proximal_Dominance',
        'Distal_Phase':   'Distal_Dominance',
    }

    vc_titles = {
        'Axial_Turn': 'Trunk-Pelvis Vector Coding (Turn Phase)',
        'Axial_Stand': 'Trunk-Pelvis Vector Coding (Sit-to-Stand Phase)',
        'Limb_R_GaitFwd': 'Arm-Leg Vector Coding (Walk Forward Analysis)',
        'Limb_R_GaitBack': 'Arm-Leg Vector Coding (Walk Back Analysis)',
    }

    for phase_key, title_suffix in vc_titles.items():
        res = vc_summary.get(phase_key, {})
        if not res.get('gamma_deg'):
            continue
            
        vc_pct     = res['Movement_Percent']
        vc_gamma   = res['gamma_deg']
        vc_pattern = res['Coordination_Pattern']
        marker_colours = [_vc_colours.get(p, 'grey') for p in vc_pattern]

        # ── Timeline Plot ──
        fig_time = go.Figure()

        # Background zone bands
        for lo, hi, col, lbl in [
            (0,   22.5, 'rgba(240, 80, 60, 0.08)',  'Proximal'),
            (22.5, 67.5, 'rgba(90, 110, 250, 0.08)', 'In-Phase'),
            (67.5,112.5, 'rgba(170, 80, 250, 0.08)', 'Distal'),
            (112.5,157.5,'rgba(0, 200, 150, 0.08)',  'Anti-Phase'),
            (157.5,202.5,'rgba(240, 80, 60, 0.08)',  'Proximal'),
            (202.5,247.5,'rgba(90, 110, 250, 0.08)', 'In-Phase'),
            (247.5,292.5,'rgba(170, 80, 250, 0.08)', 'Distal'),
            (292.5,337.5,'rgba(0, 200, 150, 0.08)',  'Anti-Phase'),
            (337.5,360,  'rgba(240, 80, 60, 0.08)',  'Proximal'),
        ]:
            fig_time.add_hrect(y0=lo, y1=hi, fillcolor=col, line_width=0, annotation_text=lbl if lo < 180 else '', annotation_position='right', annotation_font_size=9)

        fig_time.add_trace(go.Scatter(
            x=vc_pct, y=vc_gamma,
            mode='markers+lines',
            marker=dict(color=marker_colours, size=7),
            line=dict(color='rgba(100,100,100,0.4)', width=1),
            name='\u03b3 Coupling Angle',
            hovertemplate='%{x:.0f}%<br>\u03b3 = %{y:.1f}\u00b0<extra></extra>',
        ))
        fig_time.update_layout(
            title=f'Coupling Angle Timeline \u2014 {title_suffix}',
            xaxis_title='Movement Progress (%)',
            yaxis_title='Coupling Angle \u03b3 (\u00b0)',
            yaxis=dict(range=[0, 360], dtick=45),
            height=420,
            template='plotly_white',
            showlegend=False,
        )
        plots_html.append(fig_time.to_html(full_html=False, include_plotlyjs=False))

        # ── Donut Chart ──
        labels = []
        values = []
        colors = []
        for pat in ('In_Phase', 'Anti_Phase', 'Proximal_Phase', 'Distal_Phase'):
            pct_val = res.get(f'{pat}_pct', 0)
            if pct_val > 0:
                labels.append(_vc_labels[pat])
                values.append(pct_val)
                colors.append(_vc_colours[pat])
                
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.4,
            marker=dict(colors=colors),
            textinfo='label+percent',
            sort=False
        )])
        fig_donut.update_layout(
            title=f'Coordination Pattern Distribution \u2014 {title_suffix}',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        plots_html.append(fig_donut.to_html(full_html=False, include_plotlyjs=False))

    # --- HTML Formatting ---

    plotly_js = '<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>'
    images_html = '\n'.join([f'<div style="margin-bottom: 40px; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">{html}</div>' for html in plots_html])
    
    meta_html = ''.join([f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in report_data.get('Metadata', {}).items()])
    spat_global = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{v:.3f}</td></tr>' if isinstance(v, (int, float)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Spatiotemporal', {}).get('Global', {}).items()])
    spat_left_right = build_side_by_side_rows(
        report_data.get('Spatiotemporal', {}).get('Left', {}),
        report_data.get('Spatiotemporal', {}).get('Right', {}),
    )
    phases_html = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{round(v[0],2)}s to {round(v[1],2)}s (Dur: {round(v[1]-v[0],2)}s)</td></tr>' if isinstance(v, (list, tuple)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Phases_Seconds', {}).items()])
    
    steps_list = report_data.get('Steps_Timeseries', [])
    steps_html = ''.join([
        f"<tr><td>{i+1}</td><td style='color: {'red' if s['Side'] == 'Right' else 'green'}; font-weight: bold;'>{s['Side']}</td>"
        f"<td>{s['Time_s']:.3f}</td><td>{s['Phase']}</td>"
        f"<td>{s.get('Stance_Time_s', '')}</td><td>{s.get('Step_Length_m', '')}</td>"
        f"<td>{s.get('Step_Width_m', '')}</td><td>{s.get('Stride_Length_m', '')}</td>"
        f"<td>{s.get('Swing_Time_s', '')}</td></tr>" 
        for i, s in enumerate(steps_list)
    ])

    phase_videos = report_data.get('Phase_Videos', [])
    videos_html = ""
    if phase_videos:
        videos_html = '<h3 class="section-title">Phase Recordings</h3>\n<div class="video-grid">'
        for pv in phase_videos:
            vpath = pv['Video_Path']
            path_url = Path(vpath).as_uri()
            if vpath.lower().endswith('.gif'):
                videos_html += f"""
            <div class="video-card">
                <img src="{path_url}" alt="{pv['Phase']}">
                <h4>{pv['Phase'].replace('_', ' ')}</h4>
            </div>"""
            else:
                videos_html += f"""
            <div class="video-card">
                <video autoplay loop muted playsinline>
                    <source src="{path_url}" type="video/mp4">
                </video>
                <h4>{pv['Phase'].replace('_', ' ')}</h4>
            </div>"""
        videos_html += '</div>\n'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>TUG Report: {name}</title>
        {plotly_js}
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .table-container {{ border: 1px solid #ddd; border-radius: 6px; margin-top: 20px; overflow-x: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            thead th {{ background-color: #f2f2f2; font-weight: bold; }}
            tbody tr:nth-child(even) {{ background-color: #fafafa; }}
            tbody tr:hover {{ background-color: #f3f7ff; }}
            .comparative-table th:first-child, .comparative-table td:first-child {{ width: 50%; }}
            .comparative-table th:not(:first-child), .comparative-table td:not(:first-child) {{ width: 25%; text-align: right; }}
            .col-left, .col-right {{ font-variant-numeric: tabular-nums; }}
            .video-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; margin-top: 20px; }}
            .video-card {{ background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #ddd; text-align: center; }}
            .video-card video, .video-card img {{ width: 100%; border-radius: 4px; }}
            .video-card h4 {{ margin: 10px 0 5px 0; color: #2c3e50; text-transform: capitalize; }}
            .section-title {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; color: #2c3e50; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TUG Analysis Interactive Report</h1>
            <h2>Subject: {name}</h2>
            
            {videos_html}
            
            <h3 class="section-title">Visual Charts</h3>
            {images_html}
            
            <h3 class="section-title">Metadata</h3>
            <div class="table-container">
                <table>
                    <thead><tr><th>Field</th><th>Value</th></tr></thead>
                    <tbody>{meta_html}</tbody>
                </table>
            </div>
            
            <h3 class="section-title">Spatiotemporal Parameters</h3>
            <h4>Global</h4>
            <div class="table-container">
                <table>
                    <thead><tr><th>Variable</th><th>Value</th></tr></thead>
                    <tbody>{spat_global}</tbody>
                </table>
            </div>
            <h4>Left vs Right</h4>
            <div class="table-container">
                <table class="comparative-table">
                    <thead><tr><th>Variable</th><th>Left</th><th>Right</th></tr></thead>
                    <tbody>{spat_left_right}</tbody>
                </table>
            </div>
            
            <h3 class="section-title">Step-by-Step Timeseries (Gait Events / HS)</h3>
            <div class="table-container">
                <table style="width: 100%;">
                    <thead><tr><th>Step Index</th><th>Side</th><th>Time (s)</th><th>Phase</th><th>Stance Time (s)</th><th>Step Length (m)</th><th>Step Width (m)</th><th>Stride Length (m)</th><th>Swing Time (s)</th></tr></thead>
                    <tbody>{steps_html}</tbody>
                </table>
            </div>
            
            <h3 class="section-title">TUG Phases (Seconds)</h3>
            <div class="table-container">
                <table>
                    <thead><tr><th>Phase</th><th>Range</th></tr></thead>
                    <tbody>{phases_html}</tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    report_file = out_dir / f"{name}_tug_report_interactive.html"
    report_file.write_text(html_content, encoding='utf-8')
    return report_file


def generate_phase_skeleton_gifs(analyzer: TUGAnalyzer, fps: float, report_data: dict, out_dir: Path, name: str):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    skeleton_connections = load_mediapipe_pose_connections()
    required_points = sorted({idx for a, b in skeleton_connections for idx in (a, b)})
    skeleton_series = {}
    for point_idx in required_points:
        try:
            skeleton_series[point_idx] = analyzer._get_point_3d(point_idx)
        except ValueError:
            continue
            
    phases = report_data.get('Phases_Seconds', {})
    
    all_pts = []
    for s in skeleton_series.values():
        all_pts.append(s)
    if all_pts:
        all_pts = np.vstack(all_pts)
        valid = ~np.isnan(all_pts)
        if valid.any():
            x_min, x_max = np.nanmin(all_pts[:,0]), np.nanmax(all_pts[:,0])
            y_min, y_max = np.nanmin(all_pts[:,1]), np.nanmax(all_pts[:,1])
            z_min, z_max = np.nanmin(all_pts[:,2]), np.nanmax(all_pts[:,2])
        else:
            x_min, x_max, y_min, y_max, z_min, z_max = 0,1,0,1,0,1

    phase_videos = []
    for phase_name, val in phases.items():
        if phase_name == "Global": continue
        if not isinstance(val, (list, tuple)) or len(val) != 2: continue
        
        start_s, end_s = val
        if end_s <= start_s: continue
        
        start_idx = int(start_s * fps)
        end_idx = int(end_s * fps)
        
        step = max(1, int(fps / 15))
        frames = list(range(start_idx, end_idx, step))
        if not frames: continue
        
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        lines = []
        for a_idx, b_idx in skeleton_connections:
            line, = ax.plot([], [], [], color=get_connection_color(a_idx, b_idx), lw=2)
            lines.append((line, a_idx, b_idx))
            
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        # Set consistent 1:1:1 aspect ratio based on limits
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        max_range = max(x_range, y_range, z_range)
        if max_range > 0:
            ax.set_box_aspect([x_range/max_range, y_range/max_range, z_range/max_range])
            
        ax.set_title(phase_name.replace('_', ' ').title(), pad=0)
        ax.axis('off')
        
        def update(f):
            for line, a_idx, b_idx in lines:
                if a_idx in skeleton_series and b_idx in skeleton_series:
                    pa = skeleton_series[a_idx][f]
                    pb = skeleton_series[b_idx][f]
                    if not (np.isnan(pa).any() or np.isnan(pb).any()):
                        line.set_data(np.array([pa[0], pb[0]]), np.array([pa[1], pb[1]]))
                        line.set_3d_properties(np.array([pa[2], pb[2]]))
                    else:
                        line.set_data([], [])
                        line.set_3d_properties([])
            return [l[0] for l in lines]
            
        anim = animation.FuncAnimation(fig, update, frames=frames, blit=False)
        gif_name = f"{name}_{phase_name}.gif"
        gif_path = out_dir / gif_name
        anim.save(str(gif_path), writer=animation.PillowWriter(fps=15))
        plt.close(fig)
        
        phase_videos.append({
            'Phase': phase_name,
            'Video_Path': str(gif_path.resolve())
        })
        
    return phase_videos

def generate_matplotlib_report(analyzer: TUGAnalyzer, out_dir: Path, name: str, fps: float, report_data: dict):
    """
    Generates a dynamic visual report with multiple plots,
    including XYZ overlay with gait events and sagittal stick figures.
    """
    import base64
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D
    from io import BytesIO
    
    plt.switch_backend('Agg') # Avoid Tkinter freezing in batch
    
    com = analyzer.df[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy()
    t = np.arange(len(com)) / fps
    skeleton_connections = load_mediapipe_pose_connections()
    # Keep the stick figure independent from CoM simplifications:
    # draw with all available points required by the full skeleton connections.
    required_points = sorted({idx for a, b in skeleton_connections for idx in (a, b)})
    skeleton_series = {}
    for point_idx in required_points:
        try:
            skeleton_series[point_idx] = analyzer._get_point_3d(point_idx)
        except ValueError:
            continue
    
    def get_base64_image(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
        
    images_b64 = []
    
    # --- Plot 1: CoM XYZ vs Time + Gait Events ---
    fig1 = plt.figure(figsize=(16, 6))
    ax1 = fig1.add_subplot(111)
    ax1.plot(t, com[:, 0], label='X (Mediolateral)', color='blue', linewidth=2)
    ax1.plot(t, com[:, 1], label='Y (Anteroposterior)', color='orange', linewidth=2)
    ax1.plot(t, com[:, 2], label='Z (Vertical)', color='purple', linewidth=2)
    
    for leg, color in [('Right', 'red'), ('Left', 'green')]:
        for ev, ls in [('HS', '--'), ('TO', ':')]:
            frames = analyzer.gait_events.get(leg, {}).get(ev, [])
            for i, f in enumerate(frames):
                label = f"{leg} {ev}" if i == 0 else ""
                ax1.axvline(x=f/fps, color=color, linestyle=ls, alpha=0.6, label=label)
    
    ax1.set_title('CoM Trajectory (X, Y, Z) vs Time + Gait Events (Red=Right, Green=Left | Dashed=HS, Dotted=TO)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.grid(True)
    images_b64.append(get_base64_image(fig1))
    
    # --- Plot 2: Split XYZ ---
    fig2 = plt.figure(figsize=(16, 4))
    gs2 = gridspec.GridSpec(1, 3, figure=fig2)
    ax_x = fig2.add_subplot(gs2[0, 0])
    ax_x.plot(t, com[:, 0], color='blue')
    ax_x.set_title('CoM X (Mediolateral)')
    ax_x.grid(True)
    ax_y = fig2.add_subplot(gs2[0, 1])
    ax_y.plot(t, com[:, 1], color='orange')
    ax_y.set_title('CoM Y (Anteroposterior)')
    ax_y.grid(True)
    ax_z = fig2.add_subplot(gs2[0, 2])
    ax_z.plot(t, com[:, 2], color='purple')
    ax_z.set_title('CoM Z (Vertical)')
    ax_z.grid(True)
    plt.tight_layout()
    images_b64.append(get_base64_image(fig2))
    
    # --- Plot 3: Sagittal Plane Stick Figures ---
    def draw_stick_figures(ax, frames, title, max_frames=12):
        for f in sample_frames(frames, max_frames=max_frames):
            for a_idx, b_idx in skeleton_connections:
                if a_idx not in skeleton_series or b_idx not in skeleton_series:
                    continue
                pa = skeleton_series[a_idx][f]
                pb = skeleton_series[b_idx][f]
                if np.isnan(pa).any() or np.isnan(pb).any():
                    continue
                ax.plot(
                    [pa[1], pb[1]],
                    [pa[2], pb[2]],
                    color=get_connection_color(a_idx, b_idx),
                    linewidth=2,
                    alpha=0.45,
                )
        ax.set_title(title)
        ax.set_xlabel('Y (Anteroposterior) [m]')
        ax.set_ylabel('Z (Vertical) [m]')
        ax.axis('equal')
        ax.grid(True)
        ax.legend([Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='blue', lw=2)], ['Right Side', 'Left Side'], loc='upper right')

    events = []
    for leg in ['Right', 'Left']:
        for ev in ['HS', 'TO']:
            for f in analyzer.gait_events.get(leg, {}).get(ev, []):
                events.append(int(round(f)))
    events.sort()
    
    phases = report_data.get('Phases_Seconds', {})
    wf_s = phases.get('gait_forward', (0,0))[0]
    wf_e = phases.get('gait_forward', (0,0))[1]
    wb_s = phases.get('gait_back', (0,0))[0]
    wb_e = phases.get('gait_back', (0,0))[1]
    
    fig3 = plt.figure(figsize=(16, 5))
    draw_stick_figures(fig3.add_subplot(1, 2, 1), [f for f in events if wf_s*fps <= f <= wf_e*fps], "Walk Forward Sagittal Events (Y-Z)")
    draw_stick_figures(fig3.add_subplot(1, 2, 2), [f for f in events if wb_s*fps <= f <= wb_e*fps], "Walk Back Sagittal Events (Y-Z)")
    plt.tight_layout()
    images_b64.append(get_base64_image(fig3))
    
    # --- Plot 4: 3D Equal Aspect Plot ---
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    phase_colors_mpl = {
        'stand': 'gray', 'gait_forward': 'blue', 'stop_5s': 'brown',
        'turn180': 'orange', 'gait_back': 'green', 'sit': 'purple'
    }
    phases = report_data.get('Phases_Seconds', {})
    legend_added = set()
    last_idx = 0
    for phase_name, val in ordered_phase_ranges(phases):
        start_s, end_s = val
        s_idx, e_idx = int(start_s * fps), int(end_s * fps)
        color = phase_colors_mpl.get(phase_name, 'darkred')
        show_leg = phase_name not in legend_added
        legend_added.add(phase_name)
        if show_leg:
            ax4.plot(com[s_idx:e_idx+1, 0], com[s_idx:e_idx+1, 1], com[s_idx:e_idx+1, 2], color=color, linewidth=2, label=phase_name.replace('_', ' '))
        else:
            ax4.plot(com[s_idx:e_idx+1, 0], com[s_idx:e_idx+1, 1], com[s_idx:e_idx+1, 2], color=color, linewidth=2)
        last_idx = max(last_idx, e_idx)

    for f in sample_frames(events, max_frames=10):
        for a_idx, b_idx in skeleton_connections:
            if a_idx not in skeleton_series or b_idx not in skeleton_series:
                continue
            pa = skeleton_series[a_idx][f]
            pb = skeleton_series[b_idx][f]
            if np.isnan(pa).any() or np.isnan(pb).any():
                continue
            ax4.plot(
                [pa[0], pb[0]],
                [pa[1], pb[1]],
                [pa[2], pb[2]],
                color=get_connection_color(a_idx, b_idx),
                linewidth=1.3,
                alpha=0.28,
            )
    
    if last_idx < len(com) - 1:
        ax4.plot(com[last_idx:, 0], com[last_idx:, 1], com[last_idx:, 2], color='black', linestyle=':', linewidth=2, label='Remainder')

    ax4.scatter(com[0, 0], com[0, 1], com[0, 2], color='green', s=100, label='Start')
    ax4.scatter(com[-1, 0], com[-1, 1], com[-1, 2], color='red', s=100, label='End')
    max_range = np.array([com[:,0].max()-com[:,0].min(), com[:,1].max()-com[:,1].min(), com[:,2].max()-com[:,2].min()]).max() / 2.0
    mid_x, mid_y, mid_z = (com[:,0].max()+com[:,0].min())*0.5, (com[:,1].max()+com[:,1].min())*0.5, (com[:,2].max()+com[:,2].min())*0.5
    ax4.set_xlim(mid_x - max_range, mid_x + max_range)
    ax4.set_ylim(mid_y - max_range, mid_y + max_range)
    ax4.set_zlim(mid_z - max_range, mid_z + max_range)
    ax4.set_box_aspect([1, 1, 1])
    ax4.set_title('3D CoM Trajectory (Equal Aspect)')
    ax4.set_xlabel('X (ML)')
    ax4.set_ylabel('Y (AP)')
    ax4.set_zlabel('Z (Vert)')
    ax4.legend()
    images_b64.append(get_base64_image(fig4))

    # --- Plot 5: Mean Foot Sagittal View (YZ) - Progressive vs Regressive ---
    fig5 = plt.figure(figsize=(14, 8))
    ax5 = fig5.add_subplot(111)
    
    med_r_y = analyzer.df['Med_Foot_Right_y'].to_numpy()
    med_r_z = analyzer.df['Med_Foot_Right_z'].to_numpy()
    med_r_vy = analyzer.df['Med_Foot_Right_vel_y'].to_numpy()
    
    med_l_y = analyzer.df['Med_Foot_Left_y'].to_numpy()
    med_l_z = analyzer.df['Med_Foot_Left_z'].to_numpy()
    med_l_vy = analyzer.df['Med_Foot_Left_vel_y'].to_numpy()
    
    # Right Foot
    ax5.plot(med_r_y, med_r_z, color='red', alpha=0.3, label='Right Path')
    ax5.scatter(med_r_y[med_r_vy > 0], med_r_z[med_r_vy > 0], color='darkred', s=15, label='Right Fwd (Vy > 0)')
    ax5.scatter(med_r_y[med_r_vy <= 0], med_r_z[med_r_vy <= 0], color='lightcoral', marker='x', s=15, label='Right Bwd (Vy <= 0)')
    
    hs_r_idx = analyzer.gait_events.get('Right', {}).get('HS', [])
    to_r_idx = analyzer.gait_events.get('Right', {}).get('TO', [])
    if len(hs_r_idx) > 0: ax5.scatter(med_r_y[hs_r_idx], med_r_z[hs_r_idx], color='darkred', marker='o', edgecolors='black', s=50, label='Right HS')
    if len(to_r_idx) > 0: ax5.scatter(med_r_y[to_r_idx], med_r_z[to_r_idx], color='lightcoral', marker='s', edgecolors='black', s=50, label='Right TO')

    # Left Foot
    ax5.plot(med_l_y, med_l_z, color='green', alpha=0.3, label='Left Path')
    ax5.scatter(med_l_y[med_l_vy > 0], med_l_z[med_l_vy > 0], color='darkgreen', s=15, label='Left Fwd (Vy > 0)')
    ax5.scatter(med_l_y[med_l_vy <= 0], med_l_z[med_l_vy <= 0], color='lightgreen', marker='x', s=15, label='Left Bwd (Vy <= 0)')
    
    hs_l_idx = analyzer.gait_events.get('Left', {}).get('HS', [])
    to_l_idx = analyzer.gait_events.get('Left', {}).get('TO', [])
    if len(hs_l_idx) > 0: ax5.scatter(med_l_y[hs_l_idx], med_l_z[hs_l_idx], color='darkgreen', marker='o', edgecolors='black', s=50, label='Left HS')
    if len(to_l_idx) > 0: ax5.scatter(med_l_y[to_l_idx], med_l_z[to_l_idx], color='lightgreen', marker='s', edgecolors='black', s=50, label='Left TO')
    
    ax5.set_title('Sagittal View (YZ) - Progressive (Vy>0) vs Regressive (Vy<=0)')
    ax5.set_xlabel('Y (Anteroposterior) [m]')
    ax5.set_ylabel('Z (Vertical) [m]')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    images_b64.append(get_base64_image(fig5))
    
    # --- Plot 6: Mean Foot YZ Kinematics (Position, Velocity, Acceleration, 2D Magnitude) vs Time ---
    fig6 = plt.figure(figsize=(16, 25))
    gs6 = gridspec.GridSpec(5, 2, figure=fig6)
    
    for side, prefix, color in [('Right', 'Med_Foot_Right', 'red'), ('Left', 'Med_Foot_Left', 'green')]:
        col = 0 if side == 'Right' else 1
        
        y = analyzer.df[f'{prefix}_y'].to_numpy()
        z = analyzer.df[f'{prefix}_z'].to_numpy()
        vy = analyzer.df[f'{prefix}_vel_y'].to_numpy()
        vz = analyzer.df[f'{prefix}_vel_z'].to_numpy()
        ay = analyzer.df[f'{prefix}_acc_y'].to_numpy()
        az = analyzer.df[f'{prefix}_acc_z'].to_numpy()
        
        v_yz = np.sqrt(vy**2 + vz**2)
        a_yz = np.sqrt(ay**2 + az**2)
        
        # Position Y
        ax_pos_y = fig6.add_subplot(gs6[0, col])
        ax_pos_y.plot(t, y, label='Y (AP)', color='orange')
        ax_pos_y.set_title(f'{side} Mean Foot Position Y')
        ax_pos_y.set_ylabel('Position (m)')
        ax_pos_y.grid(True); ax_pos_y.legend()

        # Position Z
        ax_pos_z = fig6.add_subplot(gs6[1, col])
        ax_pos_z.plot(t, z, label='Z (Vert)', color='purple')
        ax_pos_z.set_title(f'{side} Mean Foot Position Z')
        ax_pos_z.set_ylabel('Position (m)')
        ax_pos_z.grid(True); ax_pos_z.legend()
        
        # Add gait events to Pos Y & Z plots
        for ev, ls in [('HS', '--'), ('TO', ':')]:
            frames = analyzer.gait_events.get(side, {}).get(ev, [])
            for f in frames:
                ax_pos_y.axvline(x=f/fps, color='black', linestyle=ls, alpha=0.5)
                ax_pos_z.axvline(x=f/fps, color='black', linestyle=ls, alpha=0.5)
        
        # Velocity
        ax_vel = fig6.add_subplot(gs6[2, col])
        ax_vel.plot(t, vy, label='Vy', color='orange')
        ax_vel.plot(t, vz, label='Vz', color='purple')
        ax_vel.axhline(0, color='black', linewidth=1, linestyle='--')
        ax_vel.set_title(f'{side} Mean Foot Velocity (Vy, Vz)')
        ax_vel.set_ylabel('Velocity (m/s)')
        ax_vel.grid(True); ax_vel.legend()
        
        # Acceleration
        ax_acc = fig6.add_subplot(gs6[3, col])
        ax_acc.plot(t, ay, label='Ay', color='orange')
        ax_acc.plot(t, az, label='Az', color='purple')
        ax_acc.axhline(0, color='black', linewidth=1, linestyle='--')
        ax_acc.set_title(f'{side} Mean Foot Acceleration (Ay, Az)')
        ax_acc.set_ylabel('Acceleration (m/s²)')
        ax_acc.grid(True); ax_acc.legend()
        
        # 2D Magnitude
        ax_mag = fig6.add_subplot(gs6[4, col])
        ax_mag.plot(t, v_yz, label='|V_yz|', color='blue')
        ax_mag.plot(t, a_yz, label='|A_yz|', color='red', alpha=0.6)
        ax_mag.set_title(f'{side} Mean Foot 2D Kinematics Magnitude (YZ)')
        ax_mag.set_ylabel('Magnitude')
        ax_mag.set_xlabel('Time (s)')
        ax_mag.grid(True); ax_mag.legend()
        
    plt.tight_layout()
    images_b64.append(get_base64_image(fig6))

    # --- Plot 7: Mid Trunk Sagittal View (YZ) - Progressive vs Regressive ---
    fig7 = plt.figure(figsize=(14, 8))
    ax7 = fig7.add_subplot(111)
    
    trunk_y = analyzer.df['Mid_Trunk_y'].to_numpy()
    trunk_z = analyzer.df['Mid_Trunk_z'].to_numpy()
    trunk_vy = analyzer.df['Mid_Trunk_vel_y'].to_numpy()
    
    ax7.plot(trunk_y, trunk_z, color='purple', alpha=0.3, label='Mid Trunk Path')
    ax7.scatter(trunk_y[trunk_vy > 0], trunk_z[trunk_vy > 0], color='darkmagenta', s=15, label='Fwd (Vy > 0)')
    ax7.scatter(trunk_y[trunk_vy <= 0], trunk_z[trunk_vy <= 0], color='violet', marker='x', s=15, label='Bwd (Vy <= 0)')
    
    for leg, color_hs, color_to in [('Right', 'darkred', 'lightcoral'), ('Left', 'darkgreen', 'lightgreen')]:
        hs_idx = analyzer.gait_events.get(leg, {}).get('HS', [])
        to_idx = analyzer.gait_events.get(leg, {}).get('TO', [])
        if len(hs_idx) > 0: ax7.scatter(trunk_y[hs_idx], trunk_z[hs_idx], color=color_hs, marker='o', edgecolors='black', s=50, label=f'{leg} HS (Trunk)')
        if len(to_idx) > 0: ax7.scatter(trunk_y[to_idx], trunk_z[to_idx], color=color_to, marker='s', edgecolors='black', s=50, label=f'{leg} TO (Trunk)')
    
    ax7.set_title('Mid Trunk Sagittal View (YZ) - Progressive (Vy>0) vs Regressive (Vy<=0)')
    ax7.set_xlabel('Y (Anteroposterior) [m]')
    ax7.set_ylabel('Z (Vertical) [m]')
    ax7.legend()
    ax7.grid(True)
    
    plt.tight_layout()
    images_b64.append(get_base64_image(fig7))
    
    # --- Plot 8: Mid Trunk Kinematics (Position, Velocity, Acceleration, 2D Magnitude) vs Time ---
    fig8 = plt.figure(figsize=(12, 25))
    gs8 = gridspec.GridSpec(5, 1, figure=fig8)
    
    vy_trunk = analyzer.df['Mid_Trunk_vel_y'].to_numpy()
    vz_trunk = analyzer.df['Mid_Trunk_vel_z'].to_numpy()
    ay_trunk = analyzer.df['Mid_Trunk_acc_y'].to_numpy()
    az_trunk = analyzer.df['Mid_Trunk_acc_z'].to_numpy()
    
    v_yz_trunk = np.sqrt(vy_trunk**2 + vz_trunk**2)
    a_yz_trunk = np.sqrt(ay_trunk**2 + az_trunk**2)
    
    # Position Y
    ax_pos_trunk_y = fig8.add_subplot(gs8[0, 0])
    ax_pos_trunk_y.plot(t, trunk_y, label='Y (AP)', color='orange')
    ax_pos_trunk_y.set_title('Mid Trunk Position Y')
    ax_pos_trunk_y.set_ylabel('Position (m)')
    ax_pos_trunk_y.grid(True); ax_pos_trunk_y.legend()

    # Position Z
    ax_pos_trunk_z = fig8.add_subplot(gs8[1, 0])
    ax_pos_trunk_z.plot(t, trunk_z, label='Z (Vert)', color='purple')
    ax_pos_trunk_z.set_title('Mid Trunk Position Z')
    ax_pos_trunk_z.set_ylabel('Position (m)')
    ax_pos_trunk_z.grid(True); ax_pos_trunk_z.legend()
    
    for leg, color in [('Right', 'red'), ('Left', 'green')]:
        for ev, ls in [('HS', '--'), ('TO', ':')]:
            frames = analyzer.gait_events.get(leg, {}).get(ev, [])
            for f in frames:
                ax_pos_trunk_y.axvline(x=f/fps, color='black', linestyle=ls, alpha=0.5)
                ax_pos_trunk_z.axvline(x=f/fps, color='black', linestyle=ls, alpha=0.5)
    
    # Velocity
    ax_vel_trunk = fig8.add_subplot(gs8[2, 0])
    ax_vel_trunk.plot(t, vy_trunk, label='Vy', color='orange')
    ax_vel_trunk.plot(t, vz_trunk, label='Vz', color='purple')
    ax_vel_trunk.axhline(0, color='black', linewidth=1, linestyle='--')
    ax_vel_trunk.set_title('Mid Trunk Velocity (Vy, Vz)')
    ax_vel_trunk.set_ylabel('Velocity (m/s)')
    ax_vel_trunk.grid(True); ax_vel_trunk.legend()
    
    # Acceleration
    ax_acc_trunk = fig8.add_subplot(gs8[3, 0])
    ax_acc_trunk.plot(t, ay_trunk, label='Ay', color='orange')
    ax_acc_trunk.plot(t, az_trunk, label='Az', color='purple')
    ax_acc_trunk.axhline(0, color='black', linewidth=1, linestyle='--')
    ax_acc_trunk.set_title('Mid Trunk Acceleration (Ay, Az)')
    ax_acc_trunk.set_ylabel('Acceleration (m/s²)')
    ax_acc_trunk.grid(True); ax_acc_trunk.legend()
    
    # 2D Magnitude
    ax_mag_trunk = fig8.add_subplot(gs8[4, 0])
    ax_mag_trunk.plot(t, v_yz_trunk, label='|V_yz|', color='blue')
    ax_mag_trunk.plot(t, a_yz_trunk, label='|A_yz|', color='red', alpha=0.6)
    ax_mag_trunk.set_title('Mid Trunk 2D Kinematics Magnitude (YZ)')
    ax_mag_trunk.set_ylabel('Magnitude')
    ax_mag_trunk.set_xlabel('Time (s)')
    ax_mag_trunk.grid(True); ax_mag_trunk.legend()
    
    plt.tight_layout()
    images_b64.append(get_base64_image(fig8))

    # --- HTML Formatting ---
    images_html = '\n'.join([f'<img src="data:image/png;base64,{b64}" alt="TUG Chart" />' for b64 in images_b64])
    
    meta_html = ''.join([f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in report_data.get('Metadata', {}).items()])
    spat_global = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{v:.3f}</td></tr>' if isinstance(v, (int, float)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Spatiotemporal', {}).get('Global', {}).items()])
    spat_left_right = build_side_by_side_rows(
        report_data.get('Spatiotemporal', {}).get('Left', {}),
        report_data.get('Spatiotemporal', {}).get('Right', {}),
    )
    phases_html = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{round(v[0],2)}s to {round(v[1],2)}s (Dur: {round(v[1]-v[0],2)}s)</td></tr>' if isinstance(v, (list, tuple)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Phases_Seconds', {}).items()])
    
    steps_list = report_data.get('Steps_Timeseries', [])
    steps_html = ''.join([
        f"<tr><td>{i+1}</td><td style='color: {'red' if s['Side'] == 'Right' else 'green'}; font-weight: bold;'>{s['Side']}</td>"
        f"<td>{s['Time_s']:.3f}</td><td>{s['Phase']}</td>"
        f"<td>{s.get('Stance_Time_s', '')}</td><td>{s.get('Step_Length_m', '')}</td>"
        f"<td>{s.get('Step_Width_m', '')}</td><td>{s.get('Stride_Length_m', '')}</td>"
        f"<td>{s.get('Swing_Time_s', '')}</td></tr>" 
        for i, s in enumerate(steps_list)
    ])
    
    phase_videos = report_data.get('Phase_Videos', [])
    videos_html = ""
    if phase_videos:
        videos_html = '<h3 class="section-title">Phase Recordings</h3>\n<div class="video-grid">'
        for pv in phase_videos:
            vpath = pv['Video_Path']
            path_url = Path(vpath).as_uri()
            if vpath.lower().endswith('.gif'):
                videos_html += f"""
            <div class="video-card">
                <img src="{path_url}" alt="{pv['Phase']}">
                <h4>{pv['Phase'].replace('_', ' ')}</h4>
            </div>"""
            else:
                videos_html += f"""
            <div class="video-card">
                <video autoplay loop muted playsinline>
                    <source src="{path_url}" type="video/mp4">
                </video>
                <h4>{pv['Phase'].replace('_', ' ')}</h4>
            </div>"""
        videos_html += '</div>\n'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>TUG Report: {name}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f9f9f9; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; display: block; margin: 20px auto; }}
            .table-container {{ border: 1px solid #ddd; border-radius: 6px; margin-top: 20px; overflow-x: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            thead th {{ background-color: #f2f2f2; font-weight: bold; }}
            tbody tr:nth-child(even) {{ background-color: #fafafa; }}
            tbody tr:hover {{ background-color: #f3f7ff; }}
            .comparative-table th:first-child, .comparative-table td:first-child {{ width: 50%; }}
            .comparative-table th:not(:first-child), .comparative-table td:not(:first-child) {{ width: 25%; text-align: right; }}
            .col-left, .col-right {{ font-variant-numeric: tabular-nums; }}
            .video-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 15px; margin-top: 20px; }}
            .video-card {{ background: #f8f9fa; padding: 10px; border-radius: 6px; border: 1px solid #ddd; text-align: center; }}
            .video-card video {{ width: 100%; border-radius: 4px; }}
            .video-card h4 {{ margin: 10px 0 5px 0; color: #2c3e50; text-transform: capitalize; }}
            .section-title {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; color: #2c3e50; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TUG Analysis Report</h1>
            <h2>Subject: {name}</h2>
            
            {videos_html}
            
            <h3 class="section-title">Visual Charts</h3>
            {images_html}
            
            <h3 class="section-title">Metadata</h3>
            <div class="table-container">
                <table>
                    <thead><tr><th>Field</th><th>Value</th></tr></thead>
                    <tbody>{meta_html}</tbody>
                </table>
            </div>
            
            <h3 class="section-title">Spatiotemporal Parameters</h3>
            <h4>Global</h4>
            <div class="table-container">
                <table>
                    <thead><tr><th>Variable</th><th>Value</th></tr></thead>
                    <tbody>{spat_global}</tbody>
                </table>
            </div>
            <h4>Left vs Right</h4>
            <div class="table-container">
                <table class="comparative-table">
                    <thead><tr><th>Variable</th><th>Left</th><th>Right</th></tr></thead>
                    <tbody>{spat_left_right}</tbody>
                </table>
            </div>
            
            <h3 class="section-title">Step-by-Step Timeseries (Gait Events / HS)</h3>
            <div class="table-container">
                <table style="width: 100%;">
                    <thead><tr><th>Step Index</th><th>Side</th><th>Time (s)</th><th>Phase</th><th>Stance Time (s)</th><th>Step Length (m)</th><th>Step Width (m)</th><th>Stride Length (m)</th><th>Swing Time (s)</th></tr></thead>
                    <tbody>{steps_html}</tbody>
                </table>
            </div>
            
            <h3 class="section-title">TUG Phases (Seconds)</h3>
            <div class="table-container">
                <table>
                    <thead><tr><th>Phase</th><th>Range</th></tr></thead>
                    <tbody>{phases_html}</tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    report_file = out_dir / f"{name}_tug_report.html"
    report_file.write_text(html_content, encoding='utf-8')
    return report_file


def calculate_axial_vector_coding(analyzer: 'TUGAnalyzer', fps: float, start_s: float, end_s: float) -> dict:
    """
    Calculates the Vector Coding (Coupling Angle) of axial coordination between
    Trunk and Pelvis during a specific phase of the TUG test.

    Method: Chang et al. (2008) — 4-bin classification on [0°, 360°].

    Trunk yaw  : angle of the right-shoulder → left-shoulder vector in the XY plane
    Pelvis yaw : angle of the right-hip       → left-hip        vector in the XY plane

    Coupling angle γ = atan2(Δpelvis, Δtrunk)  →  normalised to [0°, 360°]

    Coordination patterns (bins):
        Proximal-Phase  (Trunk dominant)  : 0±22.5° and 180±22.5°
        In-Phase        (en-bloc / both)  : 45±22.5° and 225±22.5°
        Distal-Phase    (Pelvis dominant) : 90±22.5° and 270±22.5°
        Anti-Phase      (opposite dirs)   : 135±22.5° and 315±22.5°
    """
    result: dict = {
        'gamma_deg': [],
        'Movement_Percent': [],
        'Coordination_Pattern': [],
        'In_Phase_pct': 0.0,
        'Anti_Phase_pct': 0.0,
        'Proximal_Phase_pct': 0.0,
        'Distal_Phase_pct': 0.0,
        'CAV_deg': 0.0,          # Coupling Angle Variability (circular SD)
        'Dominant_Pattern': 'N/A',
    }

    # ── Extract yaw signals ─────────────────────────────────────────────────
    try:
        shoulder_l = analyzer._get_point_3d(11)
        shoulder_r = analyzer._get_point_3d(12)
        hip_l      = analyzer._get_point_3d(23)
        hip_r      = analyzer._get_point_3d(24)
    except ValueError as e:
        print(f"  Vector Coding: could not extract landmark data — {e}")
        return result

    trunk_vec  = shoulder_r[:, :2] - shoulder_l[:, :2]   # XY only
    pelvis_vec = hip_r[:, :2]      - hip_l[:, :2]

    trunk_yaw  = np.degrees(np.arctan2(trunk_vec[:, 1],  trunk_vec[:, 0]))
    pelvis_yaw = np.degrees(np.arctan2(pelvis_vec[:, 1], pelvis_vec[:, 0]))

    # ── Slice the phase ────────────────────────────────────────────────
    s_idx = int(start_s * fps)
    e_idx = int(end_s   * fps)
    N     = len(trunk_yaw)

    if e_idx <= s_idx or s_idx >= N:
        print("  Vector Coding: phase too short or out of range.")
        return result

    e_idx = min(e_idx, N)
    trunk_turn  = trunk_yaw[s_idx:e_idx]
    pelvis_turn = pelvis_yaw[s_idx:e_idx]

    if len(trunk_turn) < 3:
        print("  Vector Coding: too few frames in phase.")
        return result

    # ── Interpolate to 101 points (0 → 100% of movement) ───────────────────
    from scipy.ndimage import gaussian_filter1d
    pct = np.linspace(0, 100, 101)
    old_pct = np.linspace(0, 100, len(trunk_turn))

    trunk_interp  = np.interp(pct, old_pct, trunk_turn)
    pelvis_interp = np.interp(pct, old_pct, pelvis_turn)

    # ── Coupling angle γ = atan2(Δpelvis, Δtrunk) ───────────────────────────
    d_trunk  = np.diff(trunk_interp)
    d_pelvis = np.diff(pelvis_interp)

    gamma_rad = np.arctan2(d_pelvis, d_trunk)
    gamma_deg = np.degrees(gamma_rad) % 360.0   # map to [0°, 360°]

    # ── 4-bin classification (Chang et al. 2008) ────────────────────────────
    def _classify(g: float) -> str:
        g = g % 360.0
        if (  0  <= g <  22.5) or (157.5 <= g < 202.5) or (337.5 <= g <= 360):
            return 'Proximal_Phase'   # trunk dominates
        elif (22.5 <= g <  67.5) or (202.5 <= g < 247.5):
            return 'In_Phase'         # en-bloc
        elif (67.5 <= g < 112.5) or (247.5 <= g < 292.5):
            return 'Distal_Phase'     # pelvis dominates
        else:
            return 'Anti_Phase'       # opposite directions

    patterns = [_classify(g) for g in gamma_deg]
    n = len(patterns)

    counts = {p: patterns.count(p) for p in ('In_Phase', 'Anti_Phase', 'Proximal_Phase', 'Distal_Phase')}
    dominant = max(counts, key=counts.get)

    # ── Circular SD (CAV) ────────────────────────────────────────────────────
    gamma_rad_arr = np.radians(gamma_deg)
    R = np.sqrt(np.mean(np.cos(gamma_rad_arr))**2 + np.mean(np.sin(gamma_rad_arr))**2)
    cav_deg = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))))

    result.update({
        'gamma_deg': gamma_deg.tolist(),
        'Movement_Percent': pct[:-1].tolist(),    # 100 points (diff reduces by 1)
        'Coordination_Pattern': patterns,
        'In_Phase_pct':       round(counts['In_Phase']       / n * 100, 2),
        'Anti_Phase_pct':     round(counts['Anti_Phase']     / n * 100, 2),
        'Proximal_Phase_pct': round(counts['Proximal_Phase'] / n * 100, 2),
        'Distal_Phase_pct':   round(counts['Distal_Phase']   / n * 100, 2),
        'CAV_deg':            round(cav_deg, 2),
        'Dominant_Pattern':   dominant,
    })
    return result


def calculate_limb_vector_coding_y(analyzer: 'TUGAnalyzer', fps: float, start_s: float, end_s: float, joint_a_idx: int, joint_b_idx: int) -> dict:
    """
    Calculates the Y-axis (Anteroposterior) Vector Coding between two joints 
    (e.g., Elbow vs Knee) relative to the mid-trunk during a specific phase.
    """
    result = {
        'gamma_deg': [],
        'Movement_Percent': [],
        'Coordination_Pattern': [],
        'In_Phase_pct': 0.0,
        'Anti_Phase_pct': 0.0,
        'Proximal_Phase_pct': 0.0,
        'Distal_Phase_pct': 0.0,
        'CAV_deg': 0.0,
        'Dominant_Pattern': 'N/A',
    }
    
    try:
        pt_a = analyzer._get_point_3d(joint_a_idx)[:, 1]  # Y only
        pt_b = analyzer._get_point_3d(joint_b_idx)[:, 1]  # Y only
        mt_y = analyzer.df['Mid_Trunk_y'].to_numpy()
    except ValueError as e:
        print(f"  Limb VC: could not extract landmark data — {e}")
        return result
        
    rel_a = pt_a - mt_y
    rel_b = pt_b - mt_y
    
    s_idx = int(start_s * fps)
    e_idx = int(end_s * fps)
    N     = len(rel_a)
    
    if e_idx <= s_idx or s_idx >= N:
        return result
        
    e_idx = min(e_idx, N)
    a_phase = rel_a[s_idx:e_idx]
    b_phase = rel_b[s_idx:e_idx]
    
    if len(a_phase) < 3:
        return result
        
    pct = np.linspace(0, 100, 101)
    old_pct = np.linspace(0, 100, len(a_phase))
    
    a_interp = np.interp(pct, old_pct, a_phase)
    b_interp = np.interp(pct, old_pct, b_phase)
    
    d_a = np.diff(a_interp)
    d_b = np.diff(b_interp)
    
    gamma_rad = np.arctan2(d_b, d_a)
    gamma_deg = np.degrees(gamma_rad) % 360.0
    
    def _classify(g: float) -> str:
        g = g % 360.0
        if (  0  <= g <  22.5) or (157.5 <= g < 202.5) or (337.5 <= g <= 360):
            return 'Proximal_Phase'   # Joint A dominates
        elif (22.5 <= g <  67.5) or (202.5 <= g < 247.5):
            return 'In_Phase'         # Both move same direction
        elif (67.5 <= g < 112.5) or (247.5 <= g < 292.5):
            return 'Distal_Phase'     # Joint B dominates
        else:
            return 'Anti_Phase'       # Opposite directions

    patterns = [_classify(g) for g in gamma_deg]
    n = len(patterns)

    counts = {p: patterns.count(p) for p in ('In_Phase', 'Anti_Phase', 'Proximal_Phase', 'Distal_Phase')}
    dominant = max(counts, key=counts.get)

    gamma_rad_arr = np.radians(gamma_deg)
    R = np.sqrt(np.mean(np.cos(gamma_rad_arr))**2 + np.mean(np.sin(gamma_rad_arr))**2)
    cav_deg = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))))

    result.update({
        'gamma_deg': gamma_deg.tolist(),
        'Movement_Percent': pct[:-1].tolist(),
        'Coordination_Pattern': patterns,
        'In_Phase_pct':       round(counts['In_Phase']       / n * 100, 2),
        'Anti_Phase_pct':     round(counts['Anti_Phase']     / n * 100, 2),
        'Proximal_Phase_pct': round(counts['Proximal_Phase'] / n * 100, 2),
        'Distal_Phase_pct':   round(counts['Distal_Phase']   / n * 100, 2),
        'CAV_deg':            round(cav_deg, 2),
        'Dominant_Pattern':   dominant,
    })
    return result


def process_tug_file(csv_path: Path, out_dir: Path, config_file: Path = None):

    print(f"\nProcessing {csv_path.name}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load {csv_path.name}: {e}")
        return
        
    # Metadata config
    toml_path = config_file if config_file and config_file.exists() else csv_path.with_suffix('.toml')
    metadata = {}
    if toml_path.exists() and tomllib is not None:
        try:
            with open(toml_path, "rb") as f:
                metadata = tomllib.load(f)
                print(f"Loaded metadata from {toml_path.name}: {metadata}")
        except Exception as e:
            print(f"Warning: Failed to load TOML metadata {toml_path.name} - {e}")
    
    raw_fps = metadata.get('FPS', 30.0)
    if isinstance(raw_fps, str) and '/' in raw_fps:
        try:
            n, d = map(float, raw_fps.split('/'))
            fps = n / d if d != 0 else 30.0
        except ValueError:
            fps = 30.0
    else:
        try:
            fps = float(raw_fps)
        except (ValueError, TypeError):
            fps = 30.0
            
    analyzer = TUGAnalyzer(df, fs=fps)
    analyzer.calculate_com_3d()
    analyzer.extract_kinematics()

    # Inject optional per-subject spatial overrides from TOML [spatial] block
    spatial_cfg = metadata.get('spatial', {})
    if spatial_cfg:
        if 'y_chair' in spatial_cfg:
            analyzer._meta_y_chair = float(spatial_cfg['y_chair'])
            print(f"  Spatial override: y_chair = {analyzer._meta_y_chair} m")
        if 'y_turn' in spatial_cfg:
            analyzer._meta_y_turn = float(spatial_cfg['y_turn'])
            print(f"  Spatial override: y_turn  = {analyzer._meta_y_turn} m")
        if 'y_tol' in spatial_cfg:
            analyzer._meta_y_tol = float(spatial_cfg['y_tol'])
            print(f"  Spatial override: y_tol   = {analyzer._meta_y_tol} m")

    analyzer.detect_gait_events()
    phases = analyzer.segment_tug_phases()

    
    # Strictly filter out any gait events that fall outside the Walk Forward and Walk Back phases
    wf_s, wf_e = phases.get('gait_forward', (0,0))
    wb_s, wb_e = phases.get('gait_back', (0,0))
    
    for leg in ['Right', 'Left']:
        for ev in ['HS', 'TO']:
            frames = analyzer.gait_events.get(leg, {}).get(ev, [])
            valid_frames = []
            for f in frames:
                t = f / fps
                if (wf_s <= t < wf_e) or (wb_s <= t < wb_e):
                    valid_frames.append(f)
                else:
                    print(f"Notice: Removed fictitious {leg} {ev} step at time {t:.2f}s (outside gait phases).")
            if len(frames) > 0:
                analyzer.gait_events[leg][ev] = np.array(valid_frames, dtype=int)
    
    stats = analyzer.calculate_spatiotemporal_params(fps=fps, phases=phases)
    
    # Calculate individual step times
    steps_list = []
    
    sts_s, sts_e = phases.get('stand', (0,0))
    wf_s, wf_e = phases.get('gait_forward', (0,0))
    turn_s, turn_e = phases.get('turn180', (0,0))
    wb_s, wb_e = phases.get('gait_back', (0,0))
    sit_s, sit_e = phases.get('sit', (0,0))
    
    pause_s, pause_e = phases.get('stop_5s', (wf_e, turn_s))
    
    wf_steps = 0
    wb_steps = 0
    for leg in ['Right', 'Left']:
        for f in analyzer.gait_events.get(leg, {}).get('HS', []):
            t = f / fps
            phase_label = "Other"
            
            if sts_s <= t < sts_e:
                continue
            elif wf_s <= t < wf_e:
                wf_steps += 1
                phase_label = "gait_forward"
            elif pause_s <= t < pause_e and pause_s < pause_e:
                continue
            elif turn_s <= t < turn_e:
                continue
            elif wb_s <= t < wb_e:
                wb_steps += 1
                phase_label = "gait_back"
            elif sit_s <= t <= sit_e:
                continue
            else:
                continue
            
            metrics = stats.get(leg, {}).get('per_step', {}).get(str(int(f)), {})
            step_data = {
                'Time_s': t,
                'Side': leg,
                'Phase': phase_label
            }
            for k in ['Stance_Time_s', 'Swing_Time_s', 'Step_Length_m', 'Stride_Length_m', 'Step_Width_m']:
                step_data[k] = round(metrics[k], 4) if k in metrics else ""
            
            steps_list.append(step_data)
            
    # Sort steps chronologically
    steps_list = sorted(steps_list, key=lambda x: x['Time_s'])
    
    print("Generating animated GIFs for each phase (this may take a moment)...")
    phase_videos = generate_phase_skeleton_gifs(analyzer, fps, {'Phases_Seconds': phases}, out_dir, csv_path.stem)

    # ── Vector Coding ────────────────────────────────────────────────────────
    vc_ts_data = []

    def _append_vc_ts(res, metric_name):
        if res.get('gamma_deg'):
            for pct, g, pat in zip(res['Movement_Percent'], res['gamma_deg'], res['Coordination_Pattern']):
                vc_ts_data.append({
                    'File_ID': csv_path.stem,
                    'Phase_Metric': metric_name,
                    'Movement_Percent': pct,
                    'Coupling_Angle_deg': round(g, 4),
                    'Coordination_Pattern': pat,
                })

    # 1. Axial (Trunk vs Pelvis) during Turn
    turn_s, turn_e = phases.get('turn180', (0, 0))
    vc_turn = calculate_axial_vector_coding(analyzer, fps, turn_s, turn_e)
    _append_vc_ts(vc_turn, 'Axial_Turn')

    # 2. Axial (Trunk vs Pelvis) during Stand (Sit-to-Stand)
    sts_s, sts_e = phases.get('stand', (0, 0))
    vc_stand = calculate_axial_vector_coding(analyzer, fps, sts_s, sts_e)
    _append_vc_ts(vc_stand, 'Axial_Stand')

    # 3. Limb Y-axis (Right Elbow 14 vs Right Knee 26) during Gait Forward
    wf_s, wf_e = phases.get('gait_forward', (0, 0))
    vc_wf = calculate_limb_vector_coding_y(analyzer, fps, wf_s, wf_e, 14, 26)
    _append_vc_ts(vc_wf, 'Limb_R_GaitFwd')

    # 4. Limb Y-axis (Right Elbow 14 vs Right Knee 26) during Gait Back
    wb_s, wb_e = phases.get('gait_back', (0, 0))
    vc_wb = calculate_limb_vector_coding_y(analyzer, fps, wb_s, wb_e, 14, 26)
    _append_vc_ts(vc_wb, 'Limb_R_GaitBack')

    if vc_ts_data:
        vc_df = pd.DataFrame(vc_ts_data)
        vc_df.to_csv(out_dir / f"{csv_path.stem}_bd_vector_coding.csv", index=False)
        print("  Vector Coding calculations complete.")

    # Render steps HTML
    report_data = {
        'Metadata': metadata,
        'Spatiotemporal': stats,
        'Phases_Seconds': phases,
        'Steps_Timeseries': steps_list,
        'Phase_Videos': phase_videos,
        'Vector_Coding': vc_ts_data,
        'VC_Summary': {
            'Axial_Turn': vc_turn,
            'Axial_Stand': vc_stand,
            'Limb_R_GaitFwd': vc_wf,
            'Limb_R_GaitBack': vc_wb,
        }
    }
    
    if 'Global' not in stats:
        stats['Global'] = {}
    stats['Global']['Steps_Walk_Forward'] = wf_steps
    stats['Global']['Steps_Walk_Back'] = wb_steps
    
    # Export individual steps to CSV file (overwrite)
    if steps_list:
        steps_df = pd.DataFrame(steps_list)
        steps_df.insert(0, 'File_ID', csv_path.stem)
        steps_csv_path = out_dir / f"{csv_path.stem}_bd_steps.csv"
        steps_df.to_csv(steps_csv_path, index=False)
    
    # 1. JSON Report
    report = report_data
    json_file = out_dir / f"{csv_path.stem}_tug_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    # 2. Database Export (CSV per input file, overwrite)
    # Participants
    participant_data = {
        'File_ID': csv_path.stem,
        'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    participant_data.update(metadata)
    write_single_row_csv(out_dir / f"{csv_path.stem}_bd_participants.csv", participant_data)
    
    # Results
    results_data = {'File_ID': csv_path.stem}
    if phases:
        pause_dur = 0
        if 'stop_5s' in phases:
            pause_dur = phases['stop_5s'][1] - phases['stop_5s'][0]
            
        results_data.update({
            'STS_Time_s': phases.get('stand', [0,0])[1] - phases.get('stand', [0,0])[0],
            'Walk_Fwd_Time_s': phases.get('gait_forward', [0,0])[1] - phases.get('gait_forward', [0,0])[0],
            'Freeze_Before_Turn_s': pause_dur,
            'Turn_Time_s': phases.get('turn180', [0,0])[1] - phases.get('turn180', [0,0])[0],
            'Walk_Back_Time_s': phases.get('gait_back', [0,0])[1] - phases.get('gait_back', [0,0])[0],
            'Stand_to_Sit_Time_s': phases.get('sit', [0,0])[1] - phases.get('sit', [0,0])[0],
            'Total_Time_s': phases.get('Total_TUG_Time', 0),
            'Turn_Direction': phases.get('Turn_Direction', 'Unknown')
        })
    if stats:
        results_data.update({
            'Velocity_m_s': stats.get('Global', {}).get('Velocity_m_s', 0),
            'Cadence': stats.get('Global', {}).get('Cadence_steps_per_min', 0),
            'R_Step_Length': stats.get('Right', {}).get('Step_Length_m', 0),
            'L_Step_Length': stats.get('Left', {}).get('Step_Length_m', 0),
            'Steps_Walk_Forward': stats.get('Global', {}).get('Steps_Walk_Forward', 0),
            'Steps_Walk_Back': stats.get('Global', {}).get('Steps_Walk_Back', 0),
            'XcoM_Dev_Fwd_m': stats.get('Global', {}).get('XcoM_Deviation_Fwd_m', 0),
            'XcoM_Dev_Bwd_m': stats.get('Global', {}).get('XcoM_Deviation_Bwd_m', 0),
        })
        for prefix, res in [('VCTurn', vc_turn), ('VCStand', vc_stand), ('VCWkFwd', vc_wf), ('VCWkBck', vc_wb)]:
            results_data.update({
                f'{prefix}_Dominant':   res.get('Dominant_Pattern', 'N/A'),
                f'{prefix}_InPhase_%':  res.get('In_Phase_pct', 0),
                f'{prefix}_AntiPh_%':   res.get('Anti_Phase_pct', 0),
                f'{prefix}_ProxPh_%':   res.get('Proximal_Phase_pct', 0),
                f'{prefix}_DistPh_%':   res.get('Distal_Phase_pct', 0),
                f'{prefix}_CAV_deg':    res.get('CAV_deg', 0),
            })
    write_single_row_csv(out_dir / f"{csv_path.stem}_bd_results.csv", results_data)
    
    # Kinematics summary (Max/Mean ranges)
    kin_data = {'File_ID': csv_path.stem}
    kin_data['Knee_R_Max'] = analyzer.df['Knee_Angle_R'].max() if 'Knee_Angle_R' in analyzer.df else 0
    kin_data['Knee_L_Max'] = analyzer.df['Knee_Angle_L'].max() if 'Knee_Angle_L' in analyzer.df else 0
    kin_data['Trunk_Inc_Mean'] = analyzer.df['Trunk_Inclination'].mean() if 'Trunk_Inclination' in analyzer.df else 0
    
    kin_data['Coupling_Hip_Knee_R_SD'] = analyzer.df['Coupling_Angle_Hip_Knee_R'].std() if 'Coupling_Angle_Hip_Knee_R' in analyzer.df else 0
    kin_data['Coupling_Hip_Knee_L_SD'] = analyzer.df['Coupling_Angle_Hip_Knee_L'].std() if 'Coupling_Angle_Hip_Knee_L' in analyzer.df else 0
    
    # Calculate Phase-specific kinematics (e.g. Max Trunk Velocity) if needed...
    write_single_row_csv(out_dir / f"{csv_path.stem}_bd_kinematics.csv", kin_data)
    
    # 3. Visual Report (HTML)
    html_file_matplotlib = generate_matplotlib_report(analyzer, out_dir, csv_path.stem, fps, report)
    html_file_plotly = generate_plotly_report(analyzer, out_dir, csv_path.stem, fps, report)
    
    print(f"Exported JSON, DB entries, and visual report HTMLs:\n  - {html_file_matplotlib.name}\n  - {html_file_plotly.name}")
    return report
    
def main():
    parser = argparse.ArgumentParser(
        description="TUG and TURN 3D Analysis Batch Processor\n"
                    "Extracts spatiotemporal and kinematic parameters from TUG test CSV files (MediaPipe 3D).\n"
                    "Uses a 5-second pause as a temporal anchor to robustly segment Walk, Turn, and STS phases.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input', 
                        help="Input CSV file or directory containing CSVs.\n"
                             "If not provided, a GUI file dialog will open to select the input.")
    parser.add_argument('-o', '--output', 
                        help="Output directory to save results.\n"
                             "If not provided, a GUI folder dialog will prompt you to choose one.\n"
                             "If you cancel the prompt, it defaults to a 'tug_results' folder next to the input.")
    parser.add_argument('-c', '--config', 
                        help="Path to a specific TOML config file (overrides automatic matching).\n"
                             "If not provided via CLI, the GUI will ask if you want to select one manually.")
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    config_path = Path(args.config) if args.config else None
    
    if not input_path:
        print("No input specified. Opening GUI for selection...")
        root = tk.Tk()
        root.withdraw()
        
        input_path = filedialog.askopenfilename(
            title="Select the TUG CSV file to analyze",
            filetypes=[("CSV files", "*.csv")]
        )
        if not input_path:
            input_path = filedialog.askdirectory(title="Or select a folder with CSV files for Batch processing")
            if not input_path:
                print("No file or directory selected. Exiting.")
                return
                
        # Prompt for explicit TOML selection if running via GUI
        input_path_obj = Path(input_path)
        if input_path_obj.is_file():
            ans = messagebox.askyesno(
                "Select Configuration",
                "Do you want to manually select a specific TOML config file?\n\n"
                "(If 'No', the script will automatically look for a .toml file with the same name as the CSV in the same folder)"
            )
            if ans:
                cfg = filedialog.askopenfilename(
                    title="Select the TOML configuration file",
                    filetypes=[("TOML files", "*.toml"), ("All files", "*.*")]
                )
                if cfg:
                    config_path = Path(cfg)
        else:
            ans = messagebox.askyesno(
                "Select Configuration",
                "Do you want to manually select a single TOML config file to apply to ALL CSVs in this folder?\n\n"
                "(If 'No', the script will automatically look for matching .toml files for each CSV)"
            )
            if ans:
                cfg = filedialog.askopenfilename(
                    title="Select the TOML configuration file",
                    filetypes=[("TOML files", "*.toml"), ("All files", "*.*")]
                )
                if cfg:
                    config_path = Path(cfg)
                
    input_path = Path(input_path)
    
    # Prompt for explicit Output selection if running via GUI and not passed in CLI
    
    if not output_path:
        # Ensure a root Tk instance exists and lies dormant in background to prevent ugly windows
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError:
            pass # CLI only environment 
            
        ans = messagebox.askyesno(
            "Select Output Directory",
            "Do you want to manually select an Output folder for the reports and CSVs?\n\n"
            "(If 'No', the script will automatically create a 'tug_results' folder next to your input)"
        )
        if ans:
            out_folder = filedialog.askdirectory(title="Select the Output directory for TUG results")
            if out_folder:
                output_path = out_folder

    if not output_path:
        if input_path.is_file():
            output_path = input_path.parent / "tug_results"
        else:
            output_path = input_path / "tug_results"
            
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        files_to_process = [input_path]
    else:
        files_to_process = list(input_path.glob("*.csv"))
        
    if not files_to_process:
        print(f"No CSV files found in {input_path}")
        return
        
    print(f"Starting batch processing of {len(files_to_process)} files...")
    for f in files_to_process:
        process_tug_file(f, output_path, config_path)
        
    print(f"\nAll processing complete. Results saved in {output_path}")

if __name__ == "__main__":
    main()
