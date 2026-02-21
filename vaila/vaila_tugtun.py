"""
===============================================================================
vaila_tugtun.py
===============================================================================
Author: Paulo R. P. Santiago
Created: 20 February 2026
Updated: 20 February 2026
Version: 0.0.1
Python Version: 3.12.12

Description:
------------
This script provides functionality for Timed Up and Go (TUG) instrumented
analysis with 3D kinematics.

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/agpl-3.0.html
===============================================================================
"""

import matplotlib.pyplot as plt
import argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

import numpy as np
import pandas as pd

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
        """
        patterns = [
            (f'p{index}_x', f'p{index}_y', f'p{index}_z'),
            (f'p{index}_X', f'p{index}_Y', f'p{index}_Z'),
            (f'p{index+1}_x', f'p{index+1}_y', f'p{index+1}_z'),
            (f'p{index+1}_X', f'p{index+1}_Y', f'p{index+1}_Z'),
            (f'x_{index}', f'y_{index}', f'z_{index}'),
            (f'X_{index}', f'Y_{index}', f'Z_{index}'),
            (f'{index}_x', f'{index}_y', f'{index}_z'),
            (f'{index}_X', f'{index}_Y', f'{index}_Z')
        ]
        for px, py, pz in patterns:
            if px in self.df.columns and py in self.df.columns and pz in self.df.columns:
                return self.df[[px, py, pz]].to_numpy()
        raise ValueError(f"Could not find X, Y, Z columns for MediaPipe point {index}.")

    def calculate_com_3d(self):
        """
        Calculates the 3D Center of Mass (CoM) using a simplified model,
        explicitly excluding hand and wrist markers:
        (MediaPipe 0-based: 15 to 22).
        
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
        
        if 32 in [32] and 31 in [31]: # Placeholder to check limits
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
            
        def calc_derivatives(series_3d, prefix):
            vel = np.gradient(series_3d, self.dt, axis=0)
            acc = np.gradient(vel, self.dt, axis=0)
            for i, axis in enumerate(['x', 'y', 'z']):
                self.df[f'{prefix}_vel_{axis}'] = vel[:, i]
                self.df[f'{prefix}_acc_{axis}'] = acc[:, i]
                
        if 'CoM_x' in self.df.columns:
            com_3d = self.df[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy()
            calc_derivatives(com_3d, 'CoM')
            
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

        # Med Foot Right (ankle=28, heel=30, toe=32)
        ankle_r = self._get_point_3d(28)
        toe_r = self._get_point_3d(32)
        med_foot_right = (ankle_r + heel_r + toe_r) / 3.0
        self.df['Med_Foot_Right_x'] = med_foot_right[:, 0]
        self.df['Med_Foot_Right_y'] = med_foot_right[:, 1]
        self.df['Med_Foot_Right_z'] = med_foot_right[:, 2]
        calc_derivatives(med_foot_right, 'Med_Foot_Right')

        # Med Foot Left (ankle=27, heel=29, toe=31)
        ankle_l = self._get_point_3d(27)
        toe_l = self._get_point_3d(31)
        med_foot_left = (ankle_l + heel_l + toe_l) / 3.0
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
            hs_indices, _ = find_peaks(inv_z_masked, distance=min_dist, prominence=0.01)

            # -----------------------------------------------------------------
            # TOE OFF (Zeni Modificado - Dedão mais esticado atrás da pelve)
            # -----------------------------------------------------------------
            rel_toe_xy = toe[:, [0, 1]] - pelvis_center[:, [0, 1]]
            proj_toe = np.sum(rel_toe_xy * dir_unit, axis=1)
            proj_toe_smooth = gaussian_filter1d(proj_toe, sigma=self.fs * 0.05)
            
            # Inverter a projeção (queremos o pico negativo, quando o pé tá mais atrás)
            inv_proj_toe = -proj_toe_smooth
            inv_proj_masked = np.where(is_walking_mask, inv_proj_toe, np.min(inv_proj_toe))
            
            to_indices, _ = find_peaks(inv_proj_masked, distance=min_dist, prominence=0.015)

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

    def calculate_spatiotemporal_params(self, fps: float = 30.0) -> dict:
        """
        Phase 3: 3D Spatiotemporal Parameters.
        Calculates mean and SD metrics for both sides using detected gait events.
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
            
            # Stride Length (same foot successive)
            stride_lengths = []
            for i in range(len(hs_idx) - 1):
                p1 = heel[hs_idx[i]]
                p2 = heel[hs_idx[i+1]]
                stride_lengths.append(np.linalg.norm(p2 - p1))
                
            # Step Length (opposite foot to this foot)
            step_lengths = []
            for hs in hs_idx:
                # Finds the last HS of the opposite foot before this HS
                prev_opp = opp_hs_idx[opp_hs_idx < hs]
                if len(prev_opp) > 0:
                    p1 = opp_heel[prev_opp[-1]]
                    p2 = heel[hs]
                    step_lengths.append(np.linalg.norm(p2 - p1))

            # Step Width (lateral distance during double support - at HS)
            # Approximation by 2D distance to opposite heel at current HS
            step_widths = []
            for hs in hs_idx:
                p1 = heel[hs]
                p2 = opp_heel[hs]
                step_widths.append(np.linalg.norm(p2 - p1))
                
            # Times (Stance and Swing)
            stance_times = []
            swing_times = []
            
            # Stance: HS to next TO of the same foot
            for hs in hs_idx:
                next_to = to_idx[to_idx > hs]
                if len(next_to) > 0:
                    stance_times.append((next_to[0] - hs) * dt)
                    
            # Swing: TO to next HS of the same foot
            for to in to_idx:
                next_hs = hs_idx[hs_idx > to]
                if len(next_hs) > 0:
                    swing_times.append((next_hs[0] - to) * dt)

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

        stats['Global'] = {
            'Cadence_steps_per_min': cadence,
            'Velocity_m_s': velocity
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
        dt = self.dt if hasattr(self, 'dt') and self.dt > 0 else 1/30.0
        fs = self.fs if hasattr(self, 'fs') and self.fs > 0 else 30.0
        
        # ---------------------------------------------------------
        # 1. Vertical Phase: Sit-to-Stand and Stand-to-Sit
        # ---------------------------------------------------------
        # Moderate filter on Z axis to remove vibration
        com_z_smooth = gaussian_filter1d(com_z, sigma=fs * 0.2)
        
        # Robust percentiles (immune to outliers)
        sitting_z = np.percentile(com_z_smooth, 5)   # Basal height (sitting)
        standing_z = np.percentile(com_z_smooth, 95) # Max stable height (standing)
        amplitude_z = standing_z - sitting_z
        
        # Biomechanical thresholds: 10% (movement start) and 90% (STS end)
        thresh_start = sitting_z + 0.10 * amplitude_z
        thresh_end = sitting_z + 0.90 * amplitude_z
        
        # Macroscopically identify when the subject is standing (> 50% height)
        is_standing = com_z_smooth > (sitting_z + 0.50 * amplitude_z)
        stand_indices = np.where(is_standing)[0]
        
        if len(stand_indices) < int(fs): # Needs to be standing for at least 1 second
            print("Warning: Standing phase not clearly detected or too short.")
            sts_start, sts_end, sit_start, sit_end = 0, 0, N-1, N-1
        else:
            first_stand_idx = stand_indices[0]
            last_stand_idx = stand_indices[-1]
            
            # Refine STS (Walk backwards to find chair exit)
            sts_start = first_stand_idx
            while sts_start > 0 and com_z_smooth[sts_start] > thresh_start:
                sts_start -= 1
                
            # Refine STS (Walk forwards to find standing stabilization)
            sts_end = first_stand_idx
            while sts_end < N and com_z_smooth[sts_end] < thresh_end:
                sts_end += 1

            # Refine Stand-to-Sit (Walk backwards to find descent start)
            sit_start = last_stand_idx
            while sit_start > 0 and com_z_smooth[sit_start] < thresh_end:
                sit_start -= 1
                
            # Refine Stand-to-Sit (Walk forwards until back in the chair)
            sit_end = last_stand_idx
            while sit_end < N - 1 and com_z_smooth[sit_end] > thresh_start:
                sit_end += 1

        # ---------------------------------------------------------
        # 2. Horizontal Phase: Using Mid Trunk Y-Velocity
        # ---------------------------------------------------------
        # Instead of yaw rate which was mixing start/stop/turn, we use the Mid Trunk
        # velocity in the Anteroposterior (Y) direction to clearly distinguish
        # Walk Forward (Vy > 0) from Walk Back (Vy < 0) and the Turn in between.
        
        if 'Mid_Trunk_vel_y' not in self.df.columns:
            # Fallback if Mid Trunk is not calculated yet
            vel_x = np.gradient(gaussian_filter1d(com_x, sigma=fs * 0.5))
            vel_y = np.gradient(gaussian_filter1d(com_y, sigma=fs * 0.5))
            movement_angle = np.unwrap(np.arctan2(vel_y, vel_x))
            yaw_rate = np.abs(np.gradient(movement_angle))
            turn_start, turn_end = sts_end, sit_start
            
            if sit_start > sts_end:
                search_window = yaw_rate[sts_end:sit_start]
                peaks, _ = find_peaks(search_window, distance=int(fs*1.5), prominence=np.max(search_window)*0.3)
                if len(peaks) > 0:
                    global_peaks = peaks + sts_end
                    center_peak = global_peaks[np.argmin(np.abs(global_peaks - (sts_end + sit_start)//2))]
                    threshold_yaw = np.max(search_window) * 0.15
                    t_s = center_peak
                    while t_s > sts_end and yaw_rate[t_s] > threshold_yaw: t_s -= 1
                    t_e = center_peak
                    while t_e < sit_start and yaw_rate[t_e] > threshold_yaw: t_e += 1
                    turn_start, turn_end = t_s, t_e
        else:
            mid_trunk_vy = gaussian_filter1d(self.df['Mid_Trunk_vel_y'].to_numpy(), sigma=fs * 0.5)
            
            # Find the peak forward velocity (Walk Forward)
            search_fwd = mid_trunk_vy[sts_end:(sts_end + sit_start)//2]
            peak_fwd = np.argmax(search_fwd) + sts_end if len(search_fwd) > 0 else sts_end
            
            # Find the peak backward velocity (Walk Back)
            search_bck = mid_trunk_vy[(sts_end + sit_start)//2:sit_start]
            peak_bck = np.argmin(search_bck) + (sts_end + sit_start)//2 if len(search_bck) > 0 else sit_start
            
            # The forward walk ends when forward velocity drops near zero
            threshold_vy = 0.1  # Limit to define "Quiet / Paused / Turning"
            
            fwd_stop = peak_fwd
            while fwd_stop < peak_bck and mid_trunk_vy[fwd_stop] > threshold_vy:
                fwd_stop += 1
                
            # The backward walk starts when velocity goes negative backwards
            bck_start = peak_bck
            while bck_start > fwd_stop and mid_trunk_vy[bck_start] < -threshold_vy:
                bck_start -= 1
                
            # Fallback if logic fails to keep boundaries sane
            fwd_stop = min(fwd_stop, sit_start)
            bck_start = max(fwd_stop, min(bck_start, sit_start))
            
            # Calculate rotation of the trunk (shoulders) to find actual Turn start
            try:
                shoulder_l = self._get_point_3d(11)
                shoulder_r = self._get_point_3d(12)
                shoulder_vec_xy = shoulder_r[:, [0, 1]] - shoulder_l[:, [0, 1]]
                shoulder_angle = np.unwrap(np.arctan2(shoulder_vec_xy[:, 1], shoulder_vec_xy[:, 0]))
                shoulder_yaw_rate = np.abs(np.gradient(gaussian_filter1d(shoulder_angle, sigma=fs*0.2)))
                
                # Search for rotation start inside the Paused Zone [fwd_stop, bck_start]
                if bck_start > fwd_stop:
                    zone_yaw = shoulder_yaw_rate[fwd_stop:bck_start]
                    max_yaw = np.max(zone_yaw) if len(zone_yaw) > 0 else 0
                    if max_yaw > 0.05: # Threshold for significant rotation
                        rot_thresh = max_yaw * 0.15
                        # Find first frame where rotation exceeds threshold
                        rot_idx = np.where(zone_yaw > rot_thresh)[0]
                        turn_start = fwd_stop + rot_idx[0] if len(rot_idx) > 0 else fwd_stop
                    else:
                        turn_start = fwd_stop
                else:
                    turn_start = fwd_stop
            except ValueError:
                turn_start = fwd_stop
                
            turn_end = bck_start

        # ---------------------------------------------------------
        # 3. Packaging Results
        # ---------------------------------------------------------
        phases = {
            'Sit_to_Stand': (sts_start * dt, sts_end * dt),
            'Walk_Forward': (sts_end * dt, fwd_stop * dt),
            'Pause_Before_Turn': (fwd_stop * dt, turn_start * dt) if turn_start > fwd_stop else None,
            'Turn': (turn_start * dt, turn_end * dt),
            'Walk_Back': (turn_end * dt, sit_start * dt),
            'Stand_to_Sit': (sit_start * dt, sit_end * dt),
            'Total_TUG_Time': N * dt
        }
        
        # Remove empty Pause phase from dictionary to not break HTML unpacking
        if phases['Pause_Before_Turn'] is None:
            phases.pop('Pause_Before_Turn')
        
        # Optional: Turn Direction
        X, Y, Z = self.calculate_anatomical_frames()
        turn_direction = "Unknown"
        if X is not None and turn_start < turn_end:
            y_start = Y[turn_start, [0, 1]] 
            y_end = Y[turn_end, [0, 1]]
            cross_val = y_start[0]*y_end[1] - y_start[1]*y_end[0]
            turn_direction = "Right" if cross_val < 0 else "Left"  # Inverted logic to match AP clockwise mapping
            phases['Turn_Direction'] = turn_direction
        
        self.tug_phases = phases
        return phases


def append_to_csv_db(filepath: Path, data: dict):
    """
    Appends a flat dictionary as a new row in a CSV file.
    If the file exists, it reads it, concatenates (aligning by columns), and saves to preserve structure.
    """
    df_new = pd.DataFrame([data])
    if filepath.exists():
        try:
            df_existing = pd.read_csv(filepath)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(filepath, index=False)
        except Exception as e:
            print(f"Warn: Could not properly append to existing DB, rewriting. Error: {e}")
            df_new.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df_new.to_csv(filepath, index=False)

def generate_visual_report(analyzer: TUGAnalyzer, out_dir: Path, name: str, fps: float, report_data: dict):
    """
    Generates a dynamic visual report with multiple plots,
    including XYZ overlay with gait events and sagittal stick figures.
    """
    import base64
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.lines import Line2D
    from io import BytesIO
    
    plt.switch_backend('Agg') # Avoid Tkinter freezing in batch
    
    com = analyzer.df[['CoM_x', 'CoM_y', 'CoM_z']].to_numpy()
    t = np.arange(len(com)) / fps
    
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
        if len(frames) > max_frames:
            idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in idx]
            
        for f in frames:
            pts = {}
            for p_idx in range(11, 33):
                try:
                    pts[p_idx] = analyzer._get_point_3d(p_idx)[f]
                except ValueError:
                    continue
            
            lines = [
                ([12, 14], 'red'), ([14, 16], 'red'), # Arm R
                ([12, 24], 'red'), ([24, 26], 'red'), ([26, 28], 'red'), ([28, 30], 'red'), ([30, 32], 'red'), ([28, 32], 'red'), # Leg R
                ([11, 13], 'blue'), ([13, 15], 'blue'), # Arm L
                ([11, 23], 'blue'), ([23, 25], 'blue'), ([25, 27], 'blue'), ([27, 29], 'blue'), ([29, 31], 'blue'), ([27, 31], 'blue'), # Leg L
                ([11, 12], 'black'), ([23, 24], 'black') # Torso/Hips
            ]
            for pair, color in lines:
                if pair[0] in pts and pair[1] in pts:
                    ax.plot([pts[pair[0]][1], pts[pair[1]][1]], [pts[pair[0]][2], pts[pair[1]][2]], color=color, linewidth=2, alpha=0.5)
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
    wf_s = phases.get('Walk_Forward', (0,0))[0]
    wf_e = phases.get('Walk_Forward', (0,0))[1]
    wb_s = phases.get('Walk_Back', (0,0))[0]
    wb_e = phases.get('Walk_Back', (0,0))[1]
    
    fig3 = plt.figure(figsize=(16, 5))
    draw_stick_figures(fig3.add_subplot(1, 2, 1), [f for f in events if wf_s*fps <= f <= wf_e*fps], "Walk Forward Sagittal Events (Y-Z)")
    draw_stick_figures(fig3.add_subplot(1, 2, 2), [f for f in events if wb_s*fps <= f <= wb_e*fps], "Walk Back Sagittal Events (Y-Z)")
    plt.tight_layout()
    images_b64.append(get_base64_image(fig3))
    
    # --- Plot 4: 3D Equal Aspect Plot ---
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.plot(com[:, 0], com[:, 1], com[:, 2], color='darkred', alpha=0.8, linewidth=2, label='CoM Trajectory')
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

    # --- Plot 5: Median Foot Sagittal View (YZ) - Progressive vs Regressive ---
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
    
    # Left Foot
    ax5.plot(med_l_y, med_l_z, color='green', alpha=0.3, label='Left Path')
    ax5.scatter(med_l_y[med_l_vy > 0], med_l_z[med_l_vy > 0], color='darkgreen', s=15, label='Left Fwd (Vy > 0)')
    ax5.scatter(med_l_y[med_l_vy <= 0], med_l_z[med_l_vy <= 0], color='lightgreen', marker='x', s=15, label='Left Bwd (Vy <= 0)')
    
    ax5.set_title('Sagittal View (YZ) - Progressive (Vy>0) vs Regressive (Vy<=0)')
    ax5.set_xlabel('Y (Anteroposterior) [m]')
    ax5.set_ylabel('Z (Vertical) [m]')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    images_b64.append(get_base64_image(fig5))
    
    # --- Plot 6: Med Foot YZ Kinematics (Position, Velocity, Acceleration, 2D Magnitude) vs Time ---
    fig6 = plt.figure(figsize=(16, 20))
    gs6 = gridspec.GridSpec(4, 2, figure=fig6)
    
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
        
        # Position
        ax_pos = fig6.add_subplot(gs6[0, col])
        ax_pos.plot(t, y, label='Y (AP)', color='orange')
        ax_pos.plot(t, z, label='Z (Vert)', color='purple')
        ax_pos.set_title(f'{side} Med Foot Position (Y, Z)')
        ax_pos.set_ylabel('Position (m)')
        ax_pos.grid(True); ax_pos.legend()
        
        # Velocity
        ax_vel = fig6.add_subplot(gs6[1, col])
        ax_vel.plot(t, vy, label='Vy', color='orange')
        ax_vel.plot(t, vz, label='Vz', color='purple')
        ax_vel.axhline(0, color='black', linewidth=1, linestyle='--')
        ax_vel.set_title(f'{side} Med Foot Velocity (Vy, Vz)')
        ax_vel.set_ylabel('Velocity (m/s)')
        ax_vel.grid(True); ax_vel.legend()
        
        # Acceleration
        ax_acc = fig6.add_subplot(gs6[2, col])
        ax_acc.plot(t, ay, label='Ay', color='orange')
        ax_acc.plot(t, az, label='Az', color='purple')
        ax_acc.axhline(0, color='black', linewidth=1, linestyle='--')
        ax_acc.set_title(f'{side} Med Foot Acceleration (Ay, Az)')
        ax_acc.set_ylabel('Acceleration (m/s²)')
        ax_acc.grid(True); ax_acc.legend()
        
        # 2D Magnitude
        ax_mag = fig6.add_subplot(gs6[3, col])
        ax_mag.plot(t, v_yz, label='|V_yz|', color='blue')
        ax_mag.plot(t, a_yz, label='|A_yz|', color='red', alpha=0.6)
        ax_mag.set_title(f'{side} Med Foot 2D Kinematics Magnitude (YZ)')
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
    
    ax7.set_title('Mid Trunk Sagittal View (YZ) - Progressive (Vy>0) vs Regressive (Vy<=0)')
    ax7.set_xlabel('Y (Anteroposterior) [m]')
    ax7.set_ylabel('Z (Vertical) [m]')
    ax7.legend()
    ax7.grid(True)
    
    plt.tight_layout()
    images_b64.append(get_base64_image(fig7))
    
    # --- Plot 8: Mid Trunk Kinematics (Position, Velocity, Acceleration, 2D Magnitude) vs Time ---
    fig8 = plt.figure(figsize=(12, 20))
    gs8 = gridspec.GridSpec(4, 1, figure=fig8)
    
    vy_trunk = analyzer.df['Mid_Trunk_vel_y'].to_numpy()
    vz_trunk = analyzer.df['Mid_Trunk_vel_z'].to_numpy()
    ay_trunk = analyzer.df['Mid_Trunk_acc_y'].to_numpy()
    az_trunk = analyzer.df['Mid_Trunk_acc_z'].to_numpy()
    
    v_yz_trunk = np.sqrt(vy_trunk**2 + vz_trunk**2)
    a_yz_trunk = np.sqrt(ay_trunk**2 + az_trunk**2)
    
    # Position
    ax_pos_trunk = fig8.add_subplot(gs8[0, 0])
    ax_pos_trunk.plot(t, trunk_y, label='Y (AP)', color='orange')
    ax_pos_trunk.plot(t, trunk_z, label='Z (Vert)', color='purple')
    ax_pos_trunk.set_title('Mid Trunk Position (Y, Z)')
    ax_pos_trunk.set_ylabel('Position (m)')
    ax_pos_trunk.grid(True); ax_pos_trunk.legend()
    
    # Velocity
    ax_vel_trunk = fig8.add_subplot(gs8[1, 0])
    ax_vel_trunk.plot(t, vy_trunk, label='Vy', color='orange')
    ax_vel_trunk.plot(t, vz_trunk, label='Vz', color='purple')
    ax_vel_trunk.axhline(0, color='black', linewidth=1, linestyle='--')
    ax_vel_trunk.set_title('Mid Trunk Velocity (Vy, Vz)')
    ax_vel_trunk.set_ylabel('Velocity (m/s)')
    ax_vel_trunk.grid(True); ax_vel_trunk.legend()
    
    # Acceleration
    ax_acc_trunk = fig8.add_subplot(gs8[2, 0])
    ax_acc_trunk.plot(t, ay_trunk, label='Ay', color='orange')
    ax_acc_trunk.plot(t, az_trunk, label='Az', color='purple')
    ax_acc_trunk.axhline(0, color='black', linewidth=1, linestyle='--')
    ax_acc_trunk.set_title('Mid Trunk Acceleration (Ay, Az)')
    ax_acc_trunk.set_ylabel('Acceleration (m/s²)')
    ax_acc_trunk.grid(True); ax_acc_trunk.legend()
    
    # 2D Magnitude
    ax_mag_trunk = fig8.add_subplot(gs8[3, 0])
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
    spat_right = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{v:.3f}</td></tr>' if isinstance(v, (int, float)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Spatiotemporal', {}).get('Right', {}).items()])
    spat_left = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{v:.3f}</td></tr>' if isinstance(v, (int, float)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Spatiotemporal', {}).get('Left', {}).items()])
    phases_html = ''.join([f'<tr><th>{k.replace("_", " ")}</th><td>{round(v[0],2)}s to {round(v[1],2)}s (Dur: {round(v[1]-v[0],2)}s)</td></tr>' if isinstance(v, (list, tuple)) else f'<tr><th>{k.replace("_", " ")}</th><td>{v}</td></tr>' for k,v in report_data.get('Phases_Seconds', {}).items()])
    
    steps_list = report_data.get('Steps_Timeseries', [])
    steps_html = ''.join([f"<tr><td>{i+1}</td><td style='color: {'red' if s['Side'] == 'Right' else 'green'}; font-weight: bold;'>{s['Side']}</td><td>{s['Time_s']:.3f}</td><td>{s['Phase']}</td></tr>" for i, s in enumerate(steps_list)])

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
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; font-weight: bold; width: 40%; }}
            .section-title {{ border-bottom: 2px solid #3498db; padding-bottom: 5px; color: #2c3e50; margin-top: 40px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>TUG Analysis Report</h1>
            <h2>Subject: {name}</h2>
            
            <h3 class="section-title">Visual Charts</h3>
            {images_html}
            
            <h3 class="section-title">Metadata</h3>
            <table>{meta_html}</table>
            
            <h3 class="section-title">Spatiotemporal Parameters</h3>
            <h4>Global</h4><table>{spat_global}</table>
            <h4>Right Leg</h4><table>{spat_right}</table>
            <h4>Left Leg</h4><table>{spat_left}</table>
            
            <h3 class="section-title">Step-by-Step Timeseries (Gait Events / HS)</h3>
            <table style="width: 50%;">
                <tr><th>Step Index</th><th>Side</th><th>Time (s)</th><th>Phase</th></tr>
                {steps_html}
            </table>
            
            <h3 class="section-title">TUG Phases (Seconds)</h3>
            <table>{phases_html}</table>
        </div>
    </body>
    </html>
    """
    
    report_file = out_dir / f"{name}_tug_report.html"
    report_file.write_text(html_content, encoding='utf-8')
    return report_file


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
    
    fps = metadata.get('FPS', 30.0)
    analyzer = TUGAnalyzer(df, fs=fps)
    analyzer.calculate_com_3d()
    analyzer.extract_kinematics()
    
    stats = analyzer.calculate_spatiotemporal_params(fps=fps)
    phases = analyzer.segment_tug_phases()
    
    # Calculate individual step times
    steps_list = []
    
    sts_s, sts_e = phases.get('Sit_to_Stand', (0,0))
    wf_s, wf_e = phases.get('Walk_Forward', (0,0))
    turn_s, turn_e = phases.get('Turn', (0,0))
    wb_s, wb_e = phases.get('Walk_Back', (0,0))
    sit_s, sit_e = phases.get('Stand_to_Sit', (0,0))
    
    pause_s, pause_e = phases.get('Pause_Before_Turn', (wf_e, turn_s))
    
    wf_steps = 0
    wb_steps = 0
    for leg in ['Right', 'Left']:
        for f in analyzer.gait_events.get(leg, {}).get('HS', []):
            t = f / fps
            phase_label = "Other"
            
            if sts_s <= t < sts_e:
                phase_label = "Sit_to_Stand"
            elif wf_s <= t < wf_e:
                wf_steps += 1
                phase_label = "Walk_Forward"
            elif pause_s <= t < pause_e and pause_s < pause_e:
                phase_label = "Pause_Before_Turn"
            elif turn_s <= t < turn_e:
                phase_label = "Turn"
            elif wb_s <= t < wb_e:
                wb_steps += 1
                phase_label = "Walk_Back"
            elif sit_s <= t <= sit_e:
                phase_label = "Stand_to_Sit"
            else:
                if t < sts_s:
                    phase_label = "Sitting (Start)"
                elif t > sit_e:
                    phase_label = "Sitting (End)"
            
            steps_list.append({
                'Time_s': t,
                'Side': leg,
                'Phase': phase_label
            })
            
    # Sort steps chronologically
    steps_list = sorted(steps_list, key=lambda x: x['Time_s'])
    
    # Render steps HTML
    report_data = {
        'Metadata': metadata,
        'Spatiotemporal': stats,
        'Phases_Seconds': phases,
        'Steps_Timeseries': steps_list
    }
    
    if 'Global' not in stats:
        stats['Global'] = {}
    stats['Global']['Steps_Walk_Forward'] = wf_steps
    stats['Global']['Steps_Walk_Back'] = wb_steps
    
    # Export individual steps to CSV Database
    if steps_list:
        steps_df = pd.DataFrame(steps_list)
        steps_df.insert(0, 'File_ID', csv_path.stem)
        steps_csv_path = out_dir / "tb_steps.csv"
        
        if steps_csv_path.exists():
            try:
                df_existing = pd.read_csv(steps_csv_path)
                df_combined = pd.concat([df_existing, steps_df], ignore_index=True)
                df_combined.to_csv(steps_csv_path, index=False)
            except Exception as e:
                print(f"Warn: Could not properly append to existing step DB, rewriting. Error: {e}")
                steps_df.to_csv(steps_csv_path, mode='a', header=False, index=False)
        else:
            steps_df.to_csv(steps_csv_path, index=False)
    
    # 1. JSON Report
    report = report_data
    import json
    json_file = out_dir / f"{csv_path.stem}_tug_data.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    # 2. Database Export (CSV Append)
    # Participants
    participant_data = {
        'File_ID': csv_path.stem,
        'Timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    participant_data.update(metadata)
    append_to_csv_db(out_dir / "tb_participants.csv", participant_data)
    
    # Results
    results_data = {'File_ID': csv_path.stem}
    if phases:
        pause_dur = 0
        if 'Pause_Before_Turn' in phases:
            pause_dur = phases['Pause_Before_Turn'][1] - phases['Pause_Before_Turn'][0]
            
        results_data.update({
            'STS_Time_s': phases.get('Sit_to_Stand', [0,0])[1] - phases.get('Sit_to_Stand', [0,0])[0],
            'Walk_Fwd_Time_s': phases.get('Walk_Forward', [0,0])[1] - phases.get('Walk_Forward', [0,0])[0],
            'Freeze_Before_Turn_s': pause_dur,
            'Turn_Time_s': phases.get('Turn', [0,0])[1] - phases.get('Turn', [0,0])[0],
            'Walk_Back_Time_s': phases.get('Walk_Back', [0,0])[1] - phases.get('Walk_Back', [0,0])[0],
            'Stand_to_Sit_Time_s': phases.get('Stand_to_Sit', [0,0])[1] - phases.get('Stand_to_Sit', [0,0])[0],
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
            'Steps_Walk_Back': stats.get('Global', {}).get('Steps_Walk_Back', 0)
        })
    append_to_csv_db(out_dir / "tb_results.csv", results_data)
    
    # Kinematics summary (Max/Mean ranges)
    kin_data = {'File_ID': csv_path.stem}
    kin_data['Knee_R_Max'] = analyzer.df['Knee_Angle_R'].max() if 'Knee_Angle_R' in analyzer.df else 0
    kin_data['Knee_L_Max'] = analyzer.df['Knee_Angle_L'].max() if 'Knee_Angle_L' in analyzer.df else 0
    kin_data['Trunk_Inc_Mean'] = analyzer.df['Trunk_Inclination'].mean() if 'Trunk_Inclination' in analyzer.df else 0
    
    # Calculate Phase-specific kinematics (e.g. Max Trunk Velocity) if needed...
    append_to_csv_db(out_dir / "tb_kinematics.csv", kin_data)
    
    # 3. Visual Report (HTML)
    html_file = generate_visual_report(analyzer, out_dir, csv_path.stem, fps, report)
    
    print(f"Exported JSON, DB entries, and visual report HTML: {html_file.name}")
    return report
    
def main():
    parser = argparse.ArgumentParser(description="TUG and TURN 3D Analysis Batch Processor")
    parser.add_argument('-i', '--input', help="Input CSV file or directory containing CSVs")
    parser.add_argument('-o', '--output', help="Output directory to save results")
    parser.add_argument('-c', '--config', help="Path to TOML config file (overrides automatic matching)")
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
                
    input_path = Path(input_path)
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
