"""
Project: vailá Multimodal Toolbox
Script: pynalty.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 19 December 2025
Update Date: 19 December 2025
Version: 0.1.0

Example of usage:
First activate the vaila environment:
conda activate vaila
Then run the pynalty.py script:
python pynalty.py -i input_directory -o output_directory -c config.toml

Description:
This script processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports the ball position and
velocity to CSV files. The script also generates a
video with the landmarks overlaid on the original frames.

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the MediaPipe configuration parameters.
- Choose whether to enable video resize for better pose detection
- The script processes each video, generating an output video with overlaid pose
  landmarks, and CSV files containing both normalized and pixel-based landmark
  coordinates in original video dimensions.

Requirements:
- Python 3.12.12
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Tkinter (usually included with Python installations)
- Pillow (if using image manipulation: `pip install Pillow`)
- Pandas (for coordinate conversion: `pip install pandas`)
- psutil (pip install psutil) - for memory monitoring

Output:
- CSV files containing the ball position and velocity in meters and kilometers per hour.

License:
    This project is licensed under the terms of AGPLv3.0.
"""

import os
import sys
import numpy as np
import cv2
import pygame
from numpy import linalg as LA
from numpy.linalg import inv
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import datetime
from datetime import datetime as dt
import shutil
import csv
import math
import datetime

# Tentar importar toml, se não tiver usa json como fallback ou avisa
try:
    import toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        print("Warning: 'toml' library not found. Saving/Loading might fail or require installation.")
        toml = None

# ==============================================================================
# CLASSES E FUNÇÕES AUXILIARES (Lógica Matemática)
# ==============================================================================

def dlt2d(F, L):
    '''
    Create DLT2D
    - F  matrix containing the global coordinates (X,Y)
         of the calibration frame 
    - L  matrix containing 2d coordinates of calibration 
         points seen in camera (same sequence as in F)
    '''
    F = np.matrix(F)
    L = np.matrix(L)
    Lt = L.transpose()
    C = Lt.flatten('F').transpose()
    
    m = np.size(F, 0)
    B = np.zeros((2*m, 8))
    for i in range(m):
        j = i + 1
        B[(2*j-1)-1,0] = F[i,0]
        B[(2*j-1)-1,1] = F[i,1]
        B[(2*j-1)-1,2] = 1
        B[(2*j-1)-1,6] = -F[i,0]*L[i,0]
        B[(2*j-1)-1,7] = -F[i,1]*L[i,0]
        B[(2*j)-1,3] = F[i,0]
        B[(2*j)-1,4] = F[i,1]
        B[(2*j)-1,5] = 1
        B[(2*j)-1,6] = -F[i,0]*L[i,1]
        B[(2*j)-1,7] = -F[i,1]*L[i,1]
    
    A = inv(B) * C
    return np.asarray(A)

def rec2d(A, cc2d):
    nlin = np.size(cc2d, 0)
    H = np.matrix(np.zeros((nlin, 2)))

    for k in range(nlin):
        cc2d1 = []
        cc2d2 = []
        x = cc2d[k, 0]
        y = cc2d[k, 1]
        cc2d1 = np.matrix([[A[0,0] -x * A[6,0], A[1,0] -x * A[7,0]], [A[3,0] -y * A[6,0], A[4,0]-y*A[7,0]]])
        cc2d2 = np.matrix([[x - A[2,0]], [y - A[5,0]]])
        G1 = inv(cc2d1) * cc2d2
        H[k, :] = G1.transpose()

    return np.asarray(H)

def distvelball_penalti(phorz, pvert, nframes, fpsvideo=30):
    ballradius = 0.11 # radius of the ball in meters
    distpenalty = 11 # distance of the penalty or free kick
    midgoal = 3.66 # Middle of the goal
    phorz = phorz - midgoal
    ppenalty = np.asarray([0, 0, ballradius]) # position of ball in penalty

    # Condições para bola rasteira
    if pvert < ballradius:
       pvert = ballradius
       
    pgoal = np.asarray([phorz, distpenalty, pvert]) # location 3D ball in goal
    distball = LA.norm(pgoal - ppenalty) # distance of ball to 3D location in goal
    if nframes > 0:
        velball_ms = distball / nframes * fpsvideo
        velball_kmh = velball_ms * 3.6
    else:
        velball_ms = 0
        velball_kmh = 0
    
    return distball, velball_ms, velball_kmh

# ==============================================================================
# CONFIGURAÇÃO PYGAME
# ==============================================================================

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)

# Estados da Aplicação
# Estados da Aplicação (Refactored to Events)
# Os estados agora são definidos pelo índice do evento atual na lista de eventos

class PynaltyEvent:
    def __init__(self, name, instructions):
        self.name = name
        self.instructions = instructions
        self.frame_idx = -1
        self.points = {} # dict of point_name: (x, y)
        self.is_done = False
        
    def reset(self):
        self.frame_idx = -1
        self.points = {}
        self.is_done = False

class PynaltyApp:
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.width = 800
        self.height = 600
        
        # Workflow Events
        self.events = []
        self._init_events()
        self.current_event_idx = 0
        
        # Calibration (now part of an event logic but stored here for DLT)
        self.calibration_real_coords = np.matrix('0 0; 0 2.44; 7.32 2.44; 7.32 0')
        
        self.current_frame_idx = 0
        self.frame_img = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.is_dragging = False
        self.last_mouse_pos = (0,0)
        
        # GUI Setup
        pygame.init()
        self.font = pygame.font.SysFont('Arial', 16)
        self.font_big = pygame.font.SysFont('Arial', 24)
        self.font_small = pygame.font.SysFont('Arial', 12)
        
        self.show_help = False
        self.start_drag_slider = False
        self.playing = False
        
        # Feedback System
        self.feedback_msg = ""
        self.feedback_timer = 0
        
        if self.video_path:
            self.load_video(self.video_path)
            
    def _init_events(self):
        # Step 1: GK Move Start
        e1 = PynaltyEvent("1. GK Move Start", "Select Frame where GK starts moving. Press ENTER.")
        
        # Step 2: Kick (Touch)
        e2 = PynaltyEvent("2. Kick Event", "Select Kick Frame (ENTER). Mark Ball and GK Center.")
        
        # Step 3: Goal Entry
        e3 = PynaltyEvent("3. Goal Event", "Select Goal Frame (ENTER). Mark Ball and GK Center.")
        
        # Step 4: Calibration
        e4 = PynaltyEvent("4. Calibration", "Mark 4 Goal Corners (Inf-L, Top-L, Top-R, Inf-R).")
        
        self.events = [e1, e2, e3, e4]

    @property
    def current_event(self):
        if 0 <= self.current_event_idx < len(self.events):
            return self.events[self.current_event_idx]
        return None
        
    def load_video(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"Error opening video: {path}")
            return False
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Ajustar janela inicial (Max 1280x720 ou tamanho do video)
        screen_w = 1280
        screen_h = 800
        dis_w = min(self.width, screen_w)
        dis_h = min(self.height, screen_h)
        self.display_size = (dis_w, dis_h)
        
        self.screen = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)
        pygame.display.set_caption(f"vailá - Pynalty Analysis - {os.path.basename(path)}")
        
        self.update_frame()
        return True

    def update_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_img = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")
            else:
                print("Error reading frame")

    def screen_to_image_coords(self, sx, sy):
        # Convert screen mouse pos to image coordinates considering zoom and offset
        # screen_x = (img_x * zoom) + offset_x
        # img_x = (screen_x - offset_x) / zoom
        ix = (sx - self.offset_x) / self.zoom
        iy = (sy - self.offset_y) / self.zoom
        return ix, iy

    def draw_content(self):
        self.screen.fill(BLACK)
        
        if self.frame_img:
            # Scale and blit frame
            w = int(self.frame_img.get_width() * self.zoom)
            h = int(self.frame_img.get_height() * self.zoom)
            scaled_frame = pygame.transform.scale(self.frame_img, (w, h))
            self.screen.blit(scaled_frame, (self.offset_x, self.offset_y))
            
        # Draw Markers based on Events
        
        # Helper to draw points for an event
        for evt in self.events:
            # Kick Event Points
            if evt.name.startswith("2."):
                 if "Ball" in evt.points: 
                     self.draw_marker(evt.points["Ball"], GREEN, "Kick Ball")
                 if "GK" in evt.points:
                     self.draw_marker(evt.points["GK"], (0, 200, 200), "Kick GK")
            
            # Goal Event Points
            elif evt.name.startswith("3."):
                 if "Ball" in evt.points: 
                     self.draw_marker(evt.points["Ball"], RED, "Goal Ball")
                 if "GK" in evt.points:
                     self.draw_marker(evt.points["GK"], (200, 0, 200), "Goal GK")
            
            # Calibration Points
            elif evt.name.startswith("4."):
                for i, p in enumerate(evt.points.get("points", [])):
                    self.draw_marker(p, (100, 200, 255), f"C{i+1}")
                    
        # Draw Goal Axes (if Calib done)
        self.draw_goal_axes()

        # Draw Vectors & Distance Labels
        # Kick Ball -> Goal Ball
        k_ball = self.events[1].points.get("Ball")
        g_ball = self.events[2].points.get("Ball")
        
        if k_ball and g_ball:
             k_sx, k_sy = self.image_to_screen_coords(k_ball[0], k_ball[1])
             g_sx, g_sy = self.image_to_screen_coords(g_ball[0], g_ball[1])
             pygame.draw.line(self.screen, (0, 255, 255), (k_sx, k_sy), (g_sx, g_sy), 2)
             
             # Label: Distance
             if hasattr(self, 'last_results') and self.last_results:
                 dist = self.last_results.get('dist', 0)
                 mid_x = (k_sx + g_sx) // 2
                 mid_y = (k_sy + g_sy) // 2
                 lbl = self.font.render(f"{dist:.2f} m", True, BLACK, (0, 255, 255))
                 self.screen.blit(lbl, (mid_x, mid_y))

        # Kick GK -> Goal GK
        k_gk = self.events[1].points.get("GK")
        g_gk = self.events[2].points.get("GK")
        
        if k_gk and g_gk:
             k_sx, k_sy = self.image_to_screen_coords(k_gk[0], k_gk[1])
             g_sx, g_sy = self.image_to_screen_coords(g_gk[0], g_gk[1])
             pygame.draw.line(self.screen, (255, 0, 255), (k_sx, k_sy), (g_sx, g_sy), 2)
             
             if hasattr(self, 'last_results') and self.last_results:
                 dist = self.last_results.get('gk_dist', 0)
                 mid_x = (k_sx + g_sx) // 2
                 mid_y = (k_sy + g_sy) // 2
                 lbl = self.font.render(f"{dist:.2f} m", True, BLACK, (255, 0, 255))
                 self.screen.blit(lbl, (mid_x, mid_y + 20))
             
        # Draw Event Table
        self.draw_event_table()



        # Draw Overlay UI
        self.draw_overlay()
        
        # Draw Feedback Message
        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            # Draw semi-transparent box
            msg_surf = self.font_big.render(self.feedback_msg, True, YELLOW)
            padding = 20
            w = msg_surf.get_width() + padding*2
            h = msg_surf.get_height() + padding*2
            cx, cy = self.screen.get_width()//2, self.screen.get_height()//2
            
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (cx - w//2, cy - h//2))
            
            self.screen.blit(msg_surf, (cx - msg_surf.get_width()//2, cy - msg_surf.get_height()//2))

        if self.show_help:
            self.draw_help_overlay()
        
        pygame.display.flip()
        
    def draw_marker(self, pos, color, label=""):
        # pos is in image coords
        sx = int(pos[0] * self.zoom + self.offset_x)
        sy = int(pos[1] * self.zoom + self.offset_y)
        
        # Draw Crosshair + Circle for better visibility
        size = 8
        pygame.draw.line(self.screen, color, (sx - size, sy), (sx + size, sy), 2)
        pygame.draw.line(self.screen, color, (sx, sy - size), (sx, sy + size), 2)
        pygame.draw.circle(self.screen, color, (sx, sy), 6, 2)
        
        if label:
            # Draw with outline
            txt_bg = self.font.render(label, True, BLACK)
            txt_fg = self.font.render(label, True, color)
            self.screen.blit(txt_bg, (sx + 10, sy - 10))
            self.screen.blit(txt_fg, (sx + 8, sy - 12))
            
    def image_to_screen_coords(self, ix, iy):
        sx = int(ix * self.zoom + self.offset_x)
        sy = int(iy * self.zoom + self.offset_y)
        return sx, sy

    def draw_event_table(self):
        # Draw table at Top Right
        table_w = 250
        table_h = 160
        x = self.screen.get_width() - table_w - 20
        y = 20
        
        # Background
        pygame.draw.rect(self.screen, (30, 30, 30, 200), (x, y, table_w, table_h))
        pygame.draw.rect(self.screen, WHITE, (x, y, table_w, table_h), 1)
        
        # Header
        header = self.font.render("Events (Tab / Shift+Tab)", True, YELLOW)
        self.screen.blit(header, (x + 10, y + 10))
        
        # List Events
        curr_y = y + 35
        for i, idx in enumerate(range(len(self.events))):
            evt = self.events[idx]
            color = WHITE
            prefix = "  "
            if idx == self.current_event_idx:
                color = GREEN
                prefix = "> "
                # Highlight bar
                pygame.draw.rect(self.screen, (50, 100, 50), (x+5, curr_y-2, table_w-10, 20))
            
            # Status symbol
            status = "[ ]"
            if evt.is_done: status = "[x]"
            if idx == 0 and evt.frame_idx != -1: status = "[x]" # Special case for simple events
            if idx == 3 and len(evt.points.get("points", [])) == 4: status = "[x]"
            if (idx == 1 or idx == 2) and "Ball" in evt.points and "GK" in evt.points and evt.frame_idx != -1: status = "[x]"
            
            txt = f"{prefix}{status} {evt.name}"
            surf = self.font.render(txt, True, color)
            self.screen.blit(surf, (x + 10, curr_y))
            curr_y += 22
            
        # Draw current event feedback on user actions
        evt = self.current_event
        if evt:
            # Show what is collected
            info_y = curr_y + 10
            # Frame
            f_txt = f"Frame: {evt.frame_idx}" if evt.frame_idx != -1 else "Frame: -"
            self.screen.blit(self.font_small.render(f_txt, True, GRAY), (x+10, info_y))
            
            # Points
            pts_txt = ""
            if idx == 3: # Calibration
                 pts_txt = f"Pts: {len(evt.points.get('points', []))}/4"
            elif idx in [1, 2]:
                 got = []
                 if "Ball" in evt.points: got.append("Ball")
                 if "GK" in evt.points: got.append("GK")
                 pts_txt = f"Mk: {', '.join(got)}"
            
            if pts_txt:
                 self.screen.blit(self.font_small.render(pts_txt, True, GRAY), (x+100, info_y))

    def draw_overlay(self):
        w, h = self.screen.get_size()
        
        # Bottom Control Panel (Slider)
        panel_h = 60
        pygame.draw.rect(self.screen, (30, 30, 30), (0, h - panel_h, w, panel_h))
        
        # Slider
        slider_margin = 20
        slider_w = w - (slider_margin * 2)
        slider_y = h - 30
        pygame.draw.rect(self.screen, GRAY, (slider_margin, slider_y, slider_w, 4))
        
        # Slider Handle
        if self.total_frames > 0:
            handle_x = slider_margin + int((self.current_frame_idx / self.total_frames) * slider_w)
            pygame.draw.circle(self.screen, WHITE, (handle_x, slider_y + 2), 8)
        
        # Info Text (Bottom Left)
        info_txt = f"Frame: {self.current_frame_idx}/{self.total_frames} | FPS: {self.fps:.2f} | Zoom: {self.zoom:.1f}x"
        info_surf = self.font.render(info_txt, True, WHITE)
        self.screen.blit(info_surf, (20, h - 55))
        
        # Shortcuts Hint (Bottom Right)
        hint = "H: Help | F: FPS | S: Save All Results"
        hint_surf = self.font.render(hint, True, (200, 200, 200))
        self.screen.blit(hint_surf, (w - hint_surf.get_width() - 20, h - 55))

        # Top Instruction Bar
        pygame.draw.rect(self.screen, (0,0,0,150), (0,0, w, 50))
        
        instr = ""
        st_color = WHITE
        
        evt = self.current_event
        if evt:
            instr = evt.instructions
            st_color = GREEN
            
            # Contextual hints
            if self.current_event_idx == 3: # Calibration
                pts = 4 - len(evt.points.get("points", []))
                instr = f"{evt.instructions} ({pts} remaining)"

        if self.current_event_idx >= 0 and self.all_events_done():
             instr = "All Set! Press S to Save All Results."
             st_color = (100, 255, 100)
            
        txt_surf = self.font_big.render(instr, True, st_color)
        self.screen.blit(txt_surf, (20, 10))
        
        # Results Overlay (Bottom Left above bar)
        if self.all_events_done():
            self.calculate_and_show_results(y_start=150)
            
    def all_events_done(self):
        # Simple check
        e1 = self.events[0].frame_idx != -1
        e2 = self.events[1].frame_idx != -1 and "Ball" in self.events[1].points and "GK" in self.events[1].points
        e3 = self.events[2].frame_idx != -1 and "Ball" in self.events[2].points and "GK" in self.events[2].points
        e4 = len(self.events[3].points.get("points", [])) == 4
        return e1 and e2 and e3 and e4
        
        # Results Overlay (Center/Top)
        if self.state == STATE_RESULTS:
            self.calculate_and_show_results(y_start=70)

    def draw_help_overlay(self):
        s = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        s.fill((0, 0, 0, 200))
        self.screen.blit(s, (0,0))
        
        lines = [
            "PYNALTY HELP",
            "----------------",
            "Workflow Navigation:",
            "  TAB / Shift+Tab  : Next / Prev Step",
            "  1, 2, 3, 4       : Jump to Step",
            "",
            "Video Controls:",
            "  SPACE            : Play / Pause",
            "  Left/Right Arrow : Prev/Next Frame",
            "  Up/Down Arrow    : +/- 10 Frames",
            "  Mouse Wheel      : Zoom In/Out",
            "  Middle Click Drag: Pan Image",
            "",
            "Actions:",
            "  ENTER            : Set Frame (for current step)",
            "  Left Click       : Mark Points (Ball/GK)",
            "  Right Click      : Undo / Remove Point",
            "  S                : Save All Results (TOML + CSV + HTML)",
            "  L                : Load Analysis from TOML",
            "  F                : Change FPS",
            "  H                : Toggle Help",
            "  ESC              : Exit"
        ]
        
        y = 50
        for line in lines:
            c = WHITE
            if "HELP" in line: c = YELLOW
            t = self.font_big.render(line, True, c)
            self.screen.blit(t, (100, y))
            y += 35

    def calculate_and_show_results(self, y_start=100):
        try:
            # Transform 2D coords
            # Gather data from events
            e1 = self.events[0]
            e2 = self.events[1]
            e3 = self.events[2]
            e4 = self.events[3]
            
            kick_frame = e2.frame_idx
            goal_frame = e3.frame_idx
            gk_move_frame = e1.frame_idx
            
            calib_pts = e4.points.get("points", [])
            if len(calib_pts) != 4: return
            
            calib_pix = np.array(calib_pts)
            dlt8 = dlt2d(self.calibration_real_coords, calib_pix)
            
            # Reconstruct Points
            goal_ball_pix = np.matrix(e3.points["Ball"])
            goal_ball_real = rec2d(dlt8, goal_ball_pix)
            
            # GK Points (Projected to goal plane)
            gk_kick_pix = np.matrix(e2.points["GK"])
            gk_kick_real = rec2d(dlt8, gk_kick_pix)
            
            gk_goal_pix = np.matrix(e3.points["GK"])
            gk_goal_real = rec2d(dlt8, gk_goal_pix)
            
            
            # --- BALL METRICS ---
            delta_frames = goal_frame - kick_frame
            dist, vel_ms, vel_kmh = distvelball_penalti(goal_ball_real[0,0], goal_ball_real[0,1], delta_frames, self.fps)
            
            # --- GK METRICS ---
            # 1. Response Time
            # Time between Kick and GK Move start
            gk_response_frames = gk_move_frame - kick_frame 
            gk_response_time = gk_response_frames / self.fps # Seconds
            
            # 2. GK Velocity (2D on goal plane)
            # Distance moved by GK projected on goal plane
            gk_dist = LA.norm(gk_goal_real - gk_kick_real)
            
            # Velocity over the kick duration (Kick -> Goal)
            # Or should it be (GK Move -> Goal)? Usually average velocity during the dive.
            # Let's use Kick -> Goal duration for average velocity during the penalty event?
            # Or (GK Move -> Goal) if we want jump velocity.
            # User request: "velocidade do salto do goleiro". 
            # Duration of jump = Goal Frame - GK Move Frame (assuming GK Move is start of jump)
            delta_jump_frames = goal_frame - gk_move_frame
            if delta_jump_frames > 0:
                gk_vel_ms = (gk_dist / delta_jump_frames) * self.fps
            else:
                gk_vel_ms = 0
            gk_vel_kmh = gk_vel_ms * 3.6
            
            
            self.last_results = {
                "dist": dist, "vel_ms": vel_ms, "vel_kmh": vel_kmh, "delta": delta_frames, 
                "coord_x": goal_ball_real[0,0], "coord_z": goal_ball_real[0,1],
                "gk_response_time": gk_response_time,
                "gk_dist": gk_dist,
                "gk_vel_ms": gk_vel_ms,
                "gk_vel_kmh": gk_vel_kmh
            }

            res_txt = [
                f"BALL Distance: {dist:.2f} m",
                f"BALL Velocity: {vel_kmh:.2f} km/h ({vel_ms:.2f} m/s)",
                f"BALL Delta Frames: {delta_frames}",
                f"GK Response Time: {gk_response_time:.3f} s ({gk_response_frames} frames)",
                f"GK Jump Dist: {gk_dist:.2f} m",
                f"GK Velocity: {gk_vel_kmh:.2f} km/h ({gk_vel_ms:.2f} m/s)",
                f"Ball@Goal (X,Z): ({goal_ball_real[0,0]:.2f}, {goal_ball_real[0,1]:.2f})"
            ]
            
            y_off = y_start
            for line in res_txt:
                t = self.font_big.render(line, True, WHITE)
                 # Add black outline for readability
                outline = self.font_big.render(line, True, BLACK)
                self.screen.blit(outline, (22, y_off+2))
                self.screen.blit(t, (20, y_off))
                y_off += 30
                
        except Exception as e:
            err = self.font.render(f"Error calc: {str(e)}", True, RED)
            self.screen.blit(err, (20, 100))


    def save_csv(self):
        if not hasattr(self, 'last_results') or not self.last_results:
            messagebox.showwarning("Warning", "No results calculated yet.")
            return

        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        root.destroy()
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Variable", "Value", "Unit"])
                    writer.writerow(["Video Path", self.video_path, ""])
                    writer.writerow(["FPS", self.fps, "Hz"])
                    writer.writerow(["BALL Distance", f"{self.last_results['dist']:.4f}", "m"])
                    writer.writerow(["BALL Velocity MS", f"{self.last_results['vel_ms']:.4f}", "m/s"])
                    writer.writerow(["BALL Velocity KMH", f"{self.last_results['vel_kmh']:.4f}", "km/h"])
                    writer.writerow(["BALL Delta Frames", self.last_results['delta'], "frames"])
                    
                    writer.writerow(["GK Response Time", f"{self.last_results['gk_response_time']:.4f}", "s"])
                    writer.writerow(["GK Jump Dist", f"{self.last_results['gk_dist']:.4f}", "m"])
                    writer.writerow(["GK Velocity MS", f"{self.last_results['gk_vel_ms']:.4f}", "m/s"])
                    writer.writerow(["GK Velocity KMH", f"{self.last_results['gk_vel_kmh']:.4f}", "km/h"])

                    writer.writerow(["Goal X", f"{self.last_results['coord_x']:.4f}", "m"])
                    writer.writerow(["Goal Z", f"{self.last_results['coord_z']:.4f}", "m"])
                    
                    # Save raw points too
                    writer.writerow([])
                    writer.writerow(["Raw Data", "Pixel Coords (X,Y)"])
                    e1 = self.events[0]
                    e2 = self.events[1]
                    e3 = self.events[2]
                    e4 = self.events[3]
                    
                    writer.writerow(["GK Move Frame", e1.frame_idx])
                    writer.writerow(["Kick Frame", e2.frame_idx])
                    writer.writerow(["Kick Ball", e2.points.get("Ball", "")])
                    writer.writerow(["Kick GK", e2.points.get("GK", "")])
                    writer.writerow(["Goal Frame", e3.frame_idx])
                    writer.writerow(["Goal Ball", e3.points.get("Ball", "")])
                    writer.writerow(["Goal GK", e3.points.get("GK", "")])
                    for i, cp in enumerate(e4.points.get("points", [])):
                        writer.writerow([f"Calib Point {i+1}", cp])
                        
                messagebox.showinfo("Success", "Results saved to CSV!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save CSV: {e}")

    def save_toml(self):
        if not toml:
            print("TOML library not available.")
            return

        data = {
            "video_path": self.video_path,
            "fps": self.fps,
            # Flatten for backward compat or just save structure? 
            # Let's save new structure structure, maybe add legacy fields if critical.
            "events": [
                {
                    "name": e.name,
                    "frame_idx": e.frame_idx,
                    "points": e.points
                } for e in self.events
            ],
            "results": getattr(self, "last_results", {})
        }
        
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(defaultextension=".toml", filetypes=[("TOML files", "*.toml")])
        if file_path:
            try:
                with open(file_path, "w") as f:
                    toml.dump(data, f)
                messagebox.showinfo("Success", "Analysis saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
        root.destroy()
        
    def load_toml(self):
        if not toml: return
        
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("TOML files", "*.toml")])
        root.destroy()
        
        if file_path:
            try:
                with open(file_path, "r") as f:
                    data = toml.load(f)
                
                # Restore
                events_data = data.get("events", [])
                if events_data:
                    # Load New format
                    for i, edata in enumerate(events_data):
                        if i < len(self.events):
                            self.events[i].frame_idx = edata.get("frame_idx", -1)
                            self.events[i].points = edata.get("points", {})
                            
                    # Trigger state
                    print("Loaded TOML data (New Format).")
                else:
                    # Legacy Fallback
                    print("Attempting Legacy Load...")
                    self.events[0].frame_idx = data.get("gk_move_frame", -1)
                    
                    self.events[1].frame_idx = data.get("kick_frame", -1)
                    self.events[1].points["Ball"] = data.get("kick_ball_pixel")
                    self.events[1].points["GK"] = data.get("kick_gk_pixel")
                    
                    self.events[2].frame_idx = data.get("goal_frame", -1)
                    self.events[2].points["Ball"] = data.get("goal_ball_pixel")
                    self.events[2].points["GK"] = data.get("goal_gk_pixel")
                    
                    self.events[3].points["points"] = data.get("calibration_pixels", [])
                
                # Check completion to maybe jump to results?
                if self.all_events_done():
                    # Just stay at end or start? User decides.
                    pass
                
                self.current_event_idx = 0
                self.update_frame() # update to current event frame?
                
            except Exception as e:
                print(f"Error loading: {e}")


    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Resize
                elif event.type == pygame.VIDEORESIZE:
                    self.display_size = event.size
                    self.screen = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)
                
                # Keyboard
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                         if self.current_frame_idx < self.total_frames - 1:
                             self.current_frame_idx += 1
                             self.update_frame()
                    elif event.key == pygame.K_LEFT:
                         if self.current_frame_idx > 0:
                             self.current_frame_idx -= 1
                             self.update_frame()
                    elif event.key == pygame.K_UP:
                         # Forward 10
                         self.current_frame_idx = min(self.total_frames-1, self.current_frame_idx + 10)
                         self.update_frame()
                    elif event.key == pygame.K_DOWN:
                         # Back 10
                         self.current_frame_idx = max(0, self.current_frame_idx - 10)
                         self.update_frame()
                         
                    # State transitions
                    elif event.key == pygame.K_RETURN:
                         evt = self.current_event
                         if evt:
                             # Confirm Frame selection for events that need it
                             if self.current_event_idx in [0, 1, 2]:
                                 evt.frame_idx = self.current_frame_idx
                                 print(f"Set Frame {self.current_frame_idx} for {evt.name}")
                                 self.feedback_msg = f"Frame {self.current_frame_idx} Set!"
                                 self.feedback_timer = 60 # 1 second at 60fps
                                 
                                 # Auto-advance if logic allows?
                                 # For simplicity, stay on event so user can mark points if needed (steps 2,3)
                                 # Or move to next points marking "sub-state"?
                                 # User request said: "Tab ... to go up and down"
                                 pass

                    elif event.key == pygame.K_TAB:
                        direction = -1 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1
                        self.current_event_idx = (self.current_event_idx + direction) % len(self.events)
                        
                        # When switching, jump to that event's frame if set
                        curr = self.current_event
                        if curr.frame_idx != -1:
                            self.current_frame_idx = curr.frame_idx
                            self.update_frame()
                        elif self.current_event_idx == 3: # Calibration
                             # Maybe stay where we are?
                             pass
                        elif self.current_event_idx == 2: # Goal
                             # If not set, maybe guess forward? No, stay.
                             pass
                             
                    # Shortcuts
                    elif event.key == pygame.K_1: self.current_event_idx = 0
                    elif event.key == pygame.K_2: self.current_event_idx = 1
                    elif event.key == pygame.K_3: self.current_event_idx = 2
                    elif event.key == pygame.K_4: self.current_event_idx = 3
                    
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing

                    # Zoom Keys
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                         self.zoom = min(self.zoom * 1.1, 10.0)
                    elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                         self.zoom = max(self.zoom / 1.1, 0.1)

                    elif event.key == pygame.K_s:
                        ok = self.save_results_package()
                        if ok:
                            self.feedback_msg = "All Results Saved!"
                        else:
                            self.feedback_msg = "Save Failed!"
                        self.feedback_timer = 60
                            
                    elif event.key == pygame.K_l:
                        # Shortcut to load toml?
                        self.load_toml()

                    elif event.key == pygame.K_h:
                        self.show_help = not self.show_help

                    elif event.key == pygame.K_f:
                        # Change FPS manually
                        root = tk.Tk()
                        root.withdraw()
                        new_fps = simpledialog.askfloat("Input", "Enter Video FPS:", initialvalue=self.fps)
                        if new_fps:
                            self.fps = new_fps
                            self.feedback_msg = f"FPS Set to {self.fps}"
                            self.feedback_timer = 60
                        root.destroy()
                        
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                
                # Mouse Interaction
                elif event.type == pygame.MOUSEWHEEL:
                    # Zoom
                    old_zoom = self.zoom
                    if event.y > 0:
                        self.zoom = min(self.zoom * 1.1, 10.0)
                    else:
                        self.zoom = max(self.zoom / 1.1, 0.1)
                    
                    # Optional: Adjust offset to zoom toward mouse (simplified here)
                    
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    w, h = self.screen.get_size()
                    
                    if event.button == 1: # Left Click
                        slider_y_zone = h - 35
                        if my > slider_y_zone:
                            # Slider
                            self.start_drag_slider = True
                            slider_margin = 20
                            slider_w = w - (2 * slider_margin)
                            ratio = (mx - slider_margin) / slider_w
                            ratio = max(0.0, min(1.0, ratio))
                            self.current_frame_idx = int(ratio * (self.total_frames - 1))
                            self.update_frame()
                        else:
                            # Point Marking
                            ix, iy = self.screen_to_image_coords(mx, my)
                            evt = self.current_event
                            
                            if self.current_event_idx == 1: # Kick
                                if "Ball" not in evt.points:
                                    evt.points["Ball"] = [ix, iy]
                                    self.feedback_msg = "Kick Ball Marked!"
                                elif "GK" not in evt.points:
                                    evt.points["GK"] = [ix, iy]
                                    self.feedback_msg = "Kick GK Marked!"
                                else:
                                    evt.points = {}
                                    evt.points["Ball"] = [ix, iy]
                                    self.feedback_msg = "Kick Ball Reset!"
                                self.feedback_timer = 45
                                    
                            elif self.current_event_idx == 2: # Goal
                                if "Ball" not in evt.points:
                                    evt.points["Ball"] = [ix, iy]
                                    self.feedback_msg = "Goal Ball Marked!"
                                elif "GK" not in evt.points:
                                    evt.points["GK"] = [ix, iy]
                                    self.feedback_msg = "Goal GK Marked!"
                                else:
                                    evt.points = {}
                                    evt.points["Ball"] = [ix, iy]
                                    self.feedback_msg = "Goal Ball Reset!"
                                self.feedback_timer = 45
                                    
                            elif self.current_event_idx == 3: # Calibration
                                pts = evt.points.get("points", [])
                                if len(pts) < 4:
                                    pts.append([ix, iy])
                                    evt.points["points"] = pts
                                    self.feedback_msg = f"Calib Point {len(pts)}/4"
                                    self.feedback_timer = 45

                    elif event.button == 3: # Right Click Undo
                         evt = self.current_event
                         if self.current_event_idx == 3:
                             pts = evt.points.get("points", [])
                             if pts: 
                                 pts.pop()
                                 evt.points["points"] = pts
                         elif self.current_event_idx in [1, 2]:
                             if "GK" in evt.points:
                                 del evt.points["GK"]
                             elif "Ball" in evt.points:
                                 del evt.points["Ball"]
                                 
                    elif event.button == 2: # Middle Click Pan
                        self.is_dragging = True
                        self.last_mouse_pos = (mx, my)

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.start_drag_slider = False
                    if event.button == 2:
                        self.is_dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if self.start_drag_slider:
                         w, h = self.screen.get_size()
                         slider_margin = 20
                         slider_w = w - (2 * slider_margin)
                         ratio = (mx - slider_margin) / slider_w
                         ratio = max(0.0, min(1.0, ratio))
                         self.current_frame_idx = int(ratio * (self.total_frames - 1))
                         self.update_frame()
                    
                    elif self.is_dragging:
                        dx = mx - self.last_mouse_pos[0]
                        dy = my - self.last_mouse_pos[1]
                        self.offset_x += dx
                        self.offset_y += dy
                        self.last_mouse_pos = (mx, my)

            if self.playing:
                if self.current_frame_idx < self.total_frames - 1:
                    self.current_frame_idx += 1
                    self.update_frame()
                    pygame.time.delay(int(1000 / self.fps))
            
            self.draw_content()
            clock.tick(60)

        pygame.quit()
        if self.cap:
            self.cap.release()

    def get_results_dir(self):
        if hasattr(self, 'output_dir_override') and self.output_dir_override:
             base_name = os.path.splitext(os.path.basename(self.video_path))[0]
             # If user specified output dir, do we create a subdir there or just use it?
             # User expectation: -o /path/to/docs -> /path/to/docs/video_results or just /path/to/docs?
             # Let's behave safely: create subdir inside override to keep it clean, unless override ENDS in _results?
             # CLI usually expects output dir to BE the destination.
             # But if multiple videos are processed, they need separate folders.
             # Pynalty is single-video interactive.
             # Let's append _results to avoid cluttering generic folders like Documents.
             dir_name = f"{base_name}_results"
             target_dir = os.path.join(self.output_dir_override, dir_name)
        else:
            # Create dir based on video filename
            if not self.video_path: return None
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            dir_name = f"{base_name}_results"
            target_dir = os.path.join(os.path.dirname(self.video_path), dir_name)
        
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        return target_dir

    def save_snapshot(self, frame_idx, filename, out_dir, overlay_func=None):
        if self.cap and frame_idx >= 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                # Apply custom drawing if provided
                if overlay_func:
                    frame = overlay_func(frame)
                
                path = os.path.join(out_dir, filename)
                cv2.imwrite(path, frame)

    def generate_html_report(self, out_dir, toml_path):
        if not hasattr(self, 'last_results') or not self.last_results: return
        
        res = self.last_results
        
        # Load Template (embedded for now)
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pynalty Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #f0f0f0; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background-color: #2d2d2d; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1, h2 { color: #4caf50; border-bottom: 2px solid #4caf50; padding-bottom: 10px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric-card { background-color: #383838; padding: 15px; border-radius: 8px; border-left: 4px solid #2196f3; }
        .metric-label { font-size: 0.9em; color: #aaa; }
        .metric-value { font-size: 1.4em; font-weight: bold; margin-top: 5px; }
        .snapshots { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .snapshot-item { background: #383838; padding: 10px; border-radius: 8px; text-align: center; }
        .snapshot-item img { max-width: 100%; border-radius: 4px; border: 1px solid #555; }
        .footer { margin-top: 40px; font-size: 0.8em; color: #888; text-align: center; }
        em { font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pynalty Analysis Report</h1>
        <p><strong>Video:</strong> {{VIDEO_NAME}} | <strong>Date:</strong> {{DATE}} | <strong>FPS:</strong> {{FPS}}</p>
        
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Ball Velocity</div>
                <div class="metric-value">{{BALL_VEL_KMH}} km/h</div>
                <div class="metric-label">{{BALL_VEL_MS}} m/s</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Ball Distance</div>
                <div class="metric-value">{{BALL_DIST}} m</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">GK Response Time</div>
                <div class="metric-value">{{GK_RESPONSE}} s</div>
            </div>
             <div class="metric-card">
                <div class="metric-label">GK Jump Velocity</div>
                <div class="metric-value">{{GK_VEL_KMH}} km/h</div>
                 <div class="metric-label">{{GK_VEL_MS}} m/s</div>
            </div>
             <div class="metric-card">
                <div class="metric-label">GK Jump Distance</div>
                <div class="metric-value">{{GK_DIST}} m</div>
            </div>
             <div class="metric-card">
                <div class="metric-label">Ball @ Goal (X, Z)</div>
                <div class="metric-value">({{GOAL_X}}, {{GOAL_Z}})</div>
            </div>
        </div>
        
        <h2>Event Snapshots</h2>
        <div class="snapshots">
            <div class="snapshot-item">
                <h3>1. GK Move Start</h3>
                <img src="snapshot_gk_move.png" alt="GK Move Start">
            </div>
            <div class="snapshot-item">
                <h3>2. Kick Event</h3>
                <img src="snapshot_kick.png" alt="Kick Event">
            </div>
            <div class="snapshot-item">
                <h3>3. Goal Event</h3>
                <img src="snapshot_goal.png" alt="Goal Event">
            </div>
             <div class="snapshot-item">
                <h3>4. Calibration</h3>
                <img src="snapshot_calibration.png" alt="Calibration">
            </div>
        </div>
        
        <div class="footer">
            Generated by <em>vailá</em> - Pynalty Module
        </div>
    </div>
</body>
</html>
"""
        # Replace placeholders
        html = html_template.replace("{{VIDEO_NAME}}", os.path.basename(self.video_path))
        html = html.replace("{{DATE}}", dt.now().strftime("%Y-%m-%d %H:%M:%S"))
        html = html.replace("{{FPS}}", str(self.fps))
        
        html = html.replace("{{BALL_VEL_KMH}}", f"{res['vel_kmh']:.2f}")
        html = html.replace("{{BALL_VEL_MS}}", f"{res['vel_ms']:.2f}")
        html = html.replace("{{BALL_DIST}}", f"{res['dist']:.2f}")
        
        html = html.replace("{{GK_RESPONSE}}", f"{res['gk_response_time']:.3f}")
        html = html.replace("{{GK_VEL_KMH}}", f"{res['gk_vel_kmh']:.2f}")
        html = html.replace("{{GK_VEL_MS}}", f"{res['gk_vel_ms']:.2f}")
        html = html.replace("{{GK_DIST}}", f"{res['gk_dist']:.2f}")
        
        html = html.replace("{{GOAL_X}}", f"{res['coord_x']:.2f}")
        html = html.replace("{{GOAL_Z}}", f"{res['coord_z']:.2f}")
        
        report_path = os.path.join(out_dir, "report.html")
        with open(report_path, "w") as f:
            f.write(html)
            
    def draw_goal_axes(self):
        # Only if calibration is done
        e4 = self.events[3]
        pts = e4.points.get("points", [])
        if len(pts) != 4: return
        
        try:
             # Calculate Real -> Pixel Transform
             calib_pix = np.array(pts)
             # To project Real(Object) to Pixel(Image), we swap args compared to Reconstruction
             # dlt2d(ctrl_points, image_points) -> usually Rec maps Image->Ctrl
             # So if we want Real->Pixel, we treat Pixel as "Ctrl" (Target) and Real as "Image" (Source) for rec2d?
             # Let's rely on standard: H maps P1 -> P2. dlt2d(P1, P2).
             # We want H that maps Real -> Pix.
             # So dlt2d(Real, Pix) -> H_real_pix
             # Wait, existing code: dlt8 = dlt2d(real, pix). And rec2d(dlt8, pix) -> real.
             # So dlt8 IS Pix->Real.
             # So we need dlt_inv = dlt2d(pix, real). rec2d(dlt_inv, real) -> pix.
             
             dlt_inv = dlt2d(calib_pix, self.calibration_real_coords)
             
             # Define Axes Points (Real)
             origin = np.matrix("0 0")
             axis_x = np.matrix("1 0") # 1 meter right
             axis_z = np.matrix("0 1") # 1 meter up
             
             # Project to Pixel
             p_origin = rec2d(dlt_inv, origin)
             p_x = rec2d(dlt_inv, axis_x)
             p_z = rec2d(dlt_inv, axis_z)
             
             # Draw Lines
             # Origin -> X (Red )
             sx0, sy0 = self.image_to_screen_coords(p_origin[0,0], p_origin[0,1])
             sxX, syX = self.image_to_screen_coords(p_x[0,0], p_x[0,1])
             pygame.draw.line(self.screen, (255, 0, 0), (sx0, sy0), (sxX, syX), 5)
             self.screen.blit(self.font.render("X", True, (255,0,0)), (sxX, syX))
             
             # Origin -> Z (Blue)
             sxZ, syZ = self.image_to_screen_coords(p_z[0,0], p_z[0,1])
             pygame.draw.line(self.screen, (0, 0, 255), (sx0, sy0), (sxZ, syZ), 5)
             self.screen.blit(self.font.render("Z", True, (0,0,255)), (sxZ, syZ))
             
             # Draw Origin Dot
             pygame.draw.circle(self.screen, YELLOW, (sx0, sy0), 4)
             
        except Exception:
            pass

    def save_results_package(self):
        # Master save method
        out_dir = self.get_results_dir()
        if not out_dir: return False
        
        # 1. Snapshots
        e1 = self.events[0] # GK Move
        e2 = self.events[1] # Kick
        e3 = self.events[2] # Goal
        e4 = self.events[3] # Calib (maybe save frame too?)
        
        def draw_kick_overlay(img):
            # Draw Ball (Green) and GK (Cyan)
            if "Ball" in e2.points:
                p = e2.points["Ball"]
                cv2.circle(img, (int(p[0]), int(p[1])), 8, (0, 255, 0), -1)
                cv2.putText(img, "Kick Ball", (int(p[0])+10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if "GK" in e2.points:
                p = e2.points["GK"]
                cv2.circle(img, (int(p[0]), int(p[1])), 8, (255, 255, 0), -1) # BGR: Cyan is (255, 255, 0)
                cv2.putText(img, "Kick GK", (int(p[0])+10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return img

        def draw_goal_overlay(img):
            # Draw Ball (Red) and GK (Purple-ish)
            k_ball = e2.points.get("Ball")
            k_gk = e2.points.get("GK")
            g_ball = e3.points.get("Ball")
            g_gk = e3.points.get("GK")
            
            # Goal Points
            if g_ball:
                cv2.circle(img, (int(g_ball[0]), int(g_ball[1])), 8, (0, 0, 255), -1)
                cv2.putText(img, "Goal Ball", (int(g_ball[0])+10, int(g_ball[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if g_gk:
                cv2.circle(img, (int(g_gk[0]), int(g_gk[1])), 8, (255, 0, 255), -1) # Magenta
                cv2.putText(img, "Goal GK", (int(g_gk[0])+10, int(g_gk[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Vectors
            if k_ball and g_ball:
                pt1 = (int(k_ball[0]), int(k_ball[1]))
                pt2 = (int(g_ball[0]), int(g_ball[1]))
                cv2.line(img, pt1, pt2, (255, 255, 0), 2) # Cyan line
                
                # Distance Label
                if hasattr(self, 'last_results') and self.last_results:
                     dist = self.last_results.get('dist', 0)
                     mx, my = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
                     cv2.putText(img, f"{dist:.2f} m", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if k_gk and g_gk:
                pt1 = (int(k_gk[0]), int(k_gk[1]))
                pt2 = (int(g_gk[0]), int(g_gk[1]))
                cv2.line(img, pt1, pt2, (255, 0, 255), 2)
            
            return img

        def draw_calib_overlay(img):
            pts = e4.points.get("points", [])
            for i, p in enumerate(pts):
                cv2.circle(img, (int(p[0]), int(p[1])), 6, (255, 200, 100), -1)
                cv2.putText(img, f"C{i+1}", (int(p[0])+10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
            if len(pts) == 4:
                # Draw box
                cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [cnt], True, (255, 200, 100), 2)
            return img

        self.save_snapshot(e1.frame_idx, "snapshot_gk_move.png", out_dir) # Just frame
        self.save_snapshot(e2.frame_idx, "snapshot_kick.png", out_dir, draw_kick_overlay)
        self.save_snapshot(e3.frame_idx, "snapshot_goal.png", out_dir, draw_goal_overlay)
        self.save_snapshot(e3.frame_idx, "snapshot_calibration.png", out_dir, draw_calib_overlay) # Use Goal Frame Context
        
        # 2. TOML
        toml_path = os.path.join(out_dir, "data.toml")
        with open(toml_path, "w") as f:
             # Construct data dict
            data = {
                 "video": self.video_path,
                 "fps": self.fps,
                 "events": []
            }
            for e in self.events:
                data["events"].append({
                    "name": e.name,
                    "frame_idx": e.frame_idx,
                    "points": e.points
                })
            
            # Legacy fallback keys for compatibility if needed (optional)
            toml.dump(data, f)
            
        # 3. CSV
        csv_path = os.path.join(out_dir, "results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Variable", "Value", "Unit"])
            if hasattr(self, 'last_results') and self.last_results:
                r = self.last_results
                writer.writerow(["Ball Distance", f"{r['dist']:.2f}", "m"])
                writer.writerow(["Ball Velocity", f"{r['vel_kmh']:.2f}", "km/h"])
                writer.writerow(["Ball Velocity", f"{r['vel_ms']:.2f}", "m/s"])
                writer.writerow(["GK Response Time", f"{r['gk_response_time']:.3f}", "s"])
                writer.writerow(["GK Jump Dist", f"{r['gk_dist']:.2f}", "m"])
                writer.writerow(["GK Velocity", f"{r['gk_vel_kmh']:.2f}", "km/h"])
                writer.writerow(["Goal X", f"{r['coord_x']:.2f}", "m"])
                writer.writerow(["Goal Z", f"{r['coord_z']:.2f}", "m"])
                # Enhanced CSV Data
                writer.writerow([])
                writer.writerow(["Raw Coordinates", "Pixel X", "Pixel Y"])
                if "Ball" in e2.points: writer.writerow(["Kick Ball", e2.points["Ball"][0], e2.points["Ball"][1]])
                if "GK" in e2.points: writer.writerow(["Kick GK", e2.points["GK"][0], e2.points["GK"][1]])
                if "Ball" in e3.points: writer.writerow(["Goal Ball", e3.points["Ball"][0], e3.points["Ball"][1]])
                if "GK" in e3.points: writer.writerow(["Goal GK", e3.points["GK"][0], e3.points["GK"][1]])
                for i, p in enumerate(e4.points.get("points", [])):
                    writer.writerow([f"Calib Pt {i+1}", p[0], p[1]])
                
        # 4. JSON/HTML Report
        self.generate_html_report(out_dir, toml_path)
        
        return True

def load_video_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.destroy()
    return file_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pynalty - Penalty Knock Analysis")
    parser.add_argument("-i", "--input", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Path to output directory (optional)")
    parser.add_argument("-c", "--config", help="Path to config/toml file (optional)")
    
    args = parser.parse_args()
    
    vid_path = None
    if args.input:
        vid_path = args.input
    elif len(sys.argv) > 1 and not args.input: # Fallback for direct path
         if not sys.argv[1].startswith("-"):
             vid_path = sys.argv[1]
    
    if not vid_path:
        vid_path = load_video_file_dialog()

    if vid_path:
        # Check if file exists
        if not os.path.exists(vid_path):
             print(f"Error: Video file not found: {vid_path}")
             sys.exit(1)
             
        app = PynaltyApp(vid_path)
        
        # Determine output strategy
        # Currently PynaltyApp.get_results_dir() uses video_path dirname
        # If user supplied -o, we should probably set an internal result_dir_override
        if args.output:
            if not os.path.exists(args.output):
                os.makedirs(args.output, exist_ok=True)
            app.output_dir_override = args.output
            
        # If config is passed, try to load it immediately
        if args.config:
            if os.path.exists(args.config):
                print(f"Loading config from {args.config}...")
                try:
                    with open(args.config, "r") as f:
                        data = toml.load(f)
                    app.load_from_data(data)
                except Exception as e:
                    print(f"Failed to load config: {e}")
            else:
                print(f"Config file not found: {args.config}")

        app.run()
    else:
        print("No video selected.")
