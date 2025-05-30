"""
Script: markerless_2D_analysis.py
Author: Prof. Dr. Paulo Santiago
Version: 0.3.0
Last Updated: April 2025

Description:
Versão aprimorada do script markerless_2D_analysis.py que incorpora detecção prévia
com YOLO para melhorar a precisão do MediaPipe na estimativa de pose, especialmente
em casos de oclusão no plano sagital.

Melhorias:
- Detecção prévia de pessoas usando YOLO antes do MediaPipe
- Processamento em regiões de interesse (ROI) para aumentar precisão
- Rastreamento de múltiplas pessoas ao longo dos frames
- Melhoria na detecção em plano sagital e casos de oclusão parcial
- Mantém compatibilidade com o formato de saída original

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the MediaPipe configuration parameters.
- The script processes each video, generating an output video with overlaid pose
  landmarks, and CSV files containing both normalized and pixel-based landmark
  coordinates.

How to Execute:
1. Ensure you have all dependencies installed:
   - Install OpenCV: `pip install opencv-python`
   - Install MediaPipe: `pip install mediapipe`
   - Tkinter is usually bundled with Python installations.
2. Open a terminal and navigate to the directory where `markerless_2D_analysis.py` is located.
3. Run the script using Python:

   python markerless_2D_analysis.py

4. Follow the graphical interface prompts:
   - Select the input directory with videos (.mp4, .avi, .mov).
   - Select the base output directory for processed videos and CSVs.
   - Configure the MediaPipe parameters (or leave them as default for maximum accuracy).
5. The script will process the videos and save the outputs in the specified output directory.

Requirements:
- Python 3.11.9
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Tkinter (usually included with Python installations)
- Pillow (if using image manipulation: `pip install Pillow`)

Output:
The following files are generated for each processed video:
1. Processed Video (`*_mp.mp4`):
   The video with the 2D pose landmarks overlaid on the original frames.
2. Normalized Landmark CSV (`*_mp_norm.csv`):
   A CSV file containing the landmark coordinates normalized to a scale between 0 and 1
   for each frame. These coordinates represent the relative positions of landmarks in the video.
3. Pixel Landmark CSV (`*_mp_pixel.csv`):
   A CSV file containing the landmark coordinates in pixel format. The x and y coordinates
   are scaled to the video's resolution, representing the exact pixel positions of the landmarks.
4. Log File (`log_info.txt`):
   A log file containing video metadata and processing information, such as resolution, frame rate,
   total number of frames, codec used, and the MediaPipe Pose configuration used in the processing.

License:
This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU GPLv3 (General Public License Version 3) along with this program.
If not, see <https://www.gnu.org/licenses/>.
"""

import cv2
import mediapipe as mp
import os
import time
import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import platform
import numpy as np
from ultralytics import YOLO
import torch
from scipy import signal
from scipy.signal import savgol_filter
from pykalman import KalmanFilter
import uuid
from mediapipe.framework.formats import landmark_pb2

landmark_names = [
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


class ConfidenceInputDialog(simpledialog.Dialog):
    def body(self, master):
        tk.Label(master, text="Enter minimum detection confidence (0.0 - 1.0):").grid(
            row=0
        )
        tk.Label(master, text="Enter minimum tracking confidence (0.0 - 1.0):").grid(
            row=1
        )
        tk.Label(master, text="Enter model complexity (0, 1, or 2):").grid(row=2)
        tk.Label(master, text="Enable segmentation? (True/False):").grid(row=3)
        tk.Label(master, text="Smooth segmentation? (True/False):").grid(row=4)
        tk.Label(master, text="Static image mode? (True/False):").grid(row=5)
        tk.Label(master, text="Use YOLO detection? (True/False):").grid(row=6)
        tk.Label(master, text="YOLO confidence threshold (0.0 - 1.0):").grid(row=7)
        tk.Label(master, text="Refine face landmarks? (True/False):").grid(row=8)
        tk.Label(master, text="Apply filter? (none/kalman/savgol):").grid(row=9)
        tk.Label(master, text="Estimate occluded points? (True/False):").grid(row=10)

        self.min_detection_entry = tk.Entry(master)
        self.min_detection_entry.insert(0, "0.1")
        self.min_tracking_entry = tk.Entry(master)
        self.min_tracking_entry.insert(0, "0.1")
        self.model_complexity_entry = tk.Entry(master)
        self.model_complexity_entry.insert(0, "2")
        self.enable_segmentation_entry = tk.Entry(master)
        self.enable_segmentation_entry.insert(0, "False")
        self.smooth_segmentation_entry = tk.Entry(master)
        self.smooth_segmentation_entry.insert(0, "False")
        self.static_image_mode_entry = tk.Entry(master)
        self.static_image_mode_entry.insert(0, "False")
        self.use_yolo_entry = tk.Entry(master)
        self.use_yolo_entry.insert(0, "True")
        self.yolo_conf_entry = tk.Entry(master)
        self.yolo_conf_entry.insert(0, "0.3")
        self.refine_face_entry = tk.Entry(master)
        self.refine_face_entry.insert(0, "True")
        self.filter_type_entry = tk.Entry(master)
        self.filter_type_entry.insert(0, "kalman")
        self.estimate_occluded_entry = tk.Entry(master)
        self.estimate_occluded_entry.insert(0, "True")

        self.min_detection_entry.grid(row=0, column=1)
        self.min_tracking_entry.grid(row=1, column=1)
        self.model_complexity_entry.grid(row=2, column=1)
        self.enable_segmentation_entry.grid(row=3, column=1)
        self.smooth_segmentation_entry.grid(row=4, column=1)
        self.static_image_mode_entry.grid(row=5, column=1)
        self.use_yolo_entry.grid(row=6, column=1)
        self.yolo_conf_entry.grid(row=7, column=1)
        self.refine_face_entry.grid(row=8, column=1)
        self.filter_type_entry.grid(row=9, column=1)
        self.estimate_occluded_entry.grid(row=10, column=1)

        return self.min_detection_entry

    def apply(self):
        self.result = {
            "min_detection_confidence": float(self.min_detection_entry.get()),
            "min_tracking_confidence": float(self.min_tracking_entry.get()),
            "model_complexity": int(self.model_complexity_entry.get()),
            "enable_segmentation": self.enable_segmentation_entry.get().lower()
            == "true",
            "smooth_segmentation": self.smooth_segmentation_entry.get().lower()
            == "true",
            "static_image_mode": self.static_image_mode_entry.get().lower() == "true",
            "use_yolo": self.use_yolo_entry.get().lower() == "true",
            "yolo_conf": float(self.yolo_conf_entry.get()),
            "refine_face_landmarks": self.refine_face_entry.get().lower() == "true",
            "filter_type": self.filter_type_entry.get().lower(),
            "estimate_occluded": self.estimate_occluded_entry.get().lower() == "true",
        }


def get_pose_config():
    root = tk.Tk()
    root.withdraw()
    dialog = ConfidenceInputDialog(root, title="Pose Configuration")
    if dialog.result:
        return dialog.result
    else:
        messagebox.showerror("Error", "No values entered.")
        return None


def download_or_load_yolo_model():
    """Download or load the YOLO model"""
    model_name = "yolo12x.pt"

    # Verificar se o modelo já existe localmente
    script_dir = Path(__file__).parent.resolve()
    models_dir = script_dir / "models"
    model_path = models_dir / model_name

    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        print(f"Found local model at {model_path}")
    else:
        print(f"Downloading YOLO model {model_name}...")
        try:
            # Criar instância temporária do YOLO que baixará os pesos
            model = YOLO(model_name)

            # Obter caminho onde o YOLO baixou o modelo
            source_path = model.ckpt_path

            if os.path.exists(source_path):
                # Copiar o modelo baixado para nossa pasta models
                import shutil

                shutil.copy2(source_path, model_path)
                print(f"Downloaded model saved to {model_path}")
            else:
                print(
                    f"YOLO downloaded the model but couldn't find it at {source_path}"
                )
                return None

        except Exception as e:
            print(f"Error downloading YOLO model: {e}")
            return None

    # Carregar e retornar o modelo
    try:
        model = YOLO(str(model_path))
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None


def detect_persons_with_yolo(frame, model, conf_threshold=0.3):
    """Detect persons in a frame using YOLO"""
    results = model(frame, conf=conf_threshold, classes=0)  # class 0 = person
    persons = []

    if results and len(results) > 0:
        for r in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = r
            if cls == 0:  # person class
                persons.append(
                    {"bbox": [int(x1), int(y1), int(x2), int(y2)], "conf": float(conf)}
                )

    return persons


def process_person_with_mediapipe(frame, bbox, pose, width, height):
    x1, y1, x2, y2 = bbox

    # Usar padding maior (20-25%) para garantir que o corpo completo seja capturado
    pad_x = int((x2 - x1) * 0.25)
    pad_y = int((y2 - y1) * 0.25)

    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(width, x2 + pad_x)
    y2_pad = min(height, y2 + pad_y)

    # Extrair região da pessoa
    person_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]

    if person_crop.size == 0:
        return None, None

    # Processar com MediaPipe
    rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_crop)

    if not results.pose_landmarks:
        return None, None

    # Converter landmarks para coordenadas globais
    landmarks_norm = []
    landmarks_px = []

    crop_height, crop_width = person_crop.shape[:2]

    for landmark in results.pose_landmarks.landmark:
        # Calcular coordenadas absolutas no recorte
        crop_x = landmark.x * crop_width
        crop_y = landmark.y * crop_height

        # Converter para coordenadas na imagem original
        global_x = crop_x + x1_pad
        global_y = crop_y + y1_pad

        # Normalizar para frame completo
        norm_x = global_x / width
        norm_y = global_y / height

        # Incluir visibilidade
        visibility = landmark.visibility if hasattr(landmark, "visibility") else 1.0

        landmarks_norm.append([norm_x, norm_y, landmark.z, visibility])
        landmarks_px.append([int(global_x), int(global_y), landmark.z, visibility])

    return landmarks_norm, landmarks_px


def process_video(video_path, output_dir, pose_config, yolo_model=None):
    """
    Process a video file using YOLO for person detection and MediaPipe for pose estimation.
    """
    print(f"Processing video: {video_path}")
    start_time = time.time()

    # Verificação de caminho longo no Windows
    if platform.system() == "Windows" and platform.version().startswith("10."):
        if len(str(video_path)) > 255 or len(str(output_dir)) > 255:
            messagebox.showerror(
                "Path Too Long",
                "The selected path is too long. Please choose a shorter path for both the input video and output directory.",
            )
            return

    # Abrir o vídeo
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar caminhos de saída
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_path.stem}_mp.mp4"
    output_file_path = output_dir / f"{video_path.stem}_mp_norm.csv"
    output_pixel_file_path = output_dir / f"{video_path.stem}_mp_pixel.csv"

    # Inicializar MediaPipe
    pose = mp.solutions.pose.Pose(
        static_image_mode=pose_config["static_image_mode"],
        min_detection_confidence=pose_config["min_detection_confidence"],
        min_tracking_confidence=pose_config["min_tracking_confidence"],
        model_complexity=pose_config["model_complexity"],
        enable_segmentation=pose_config["enable_segmentation"],
        smooth_segmentation=pose_config["smooth_segmentation"],
        smooth_landmarks=True,
    )

    # Inicializar rastreador de pessoas
    person_tracker = PersonTracker(max_disappeared=30)
    landmarks_history = {}  # {person_id: {landmark_idx: [[x,y,z], ...]}}

    # Preparar cabeçalhos para CSV
    headers = ["frame_index"] + [
        f"{name}_x,{name}_y,{name}_z" for name in landmark_names
    ]

    # Listas para armazenar landmarks
    normalized_landmarks_list = []
    pixel_landmarks_list = []
    frames_with_missing_data = []

    print(f"\nEtapa 1/2: Processando landmarks (total frames: {total_frames})")

    # ETAPA 1: Processar landmarks e gerar CSVs
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Mostrar progresso
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(
                f"\rProcessando frame {frame_count}/{total_frames} ({progress:.1f}%)",
                end="",
            )

        # Processar com YOLO se configurado
        if pose_config["use_yolo"] and yolo_model:
            persons = detect_persons_with_yolo(
                frame, yolo_model, pose_config["yolo_conf"]
            )
            person_bboxes = [p["bbox"] for p in persons]
            tracked_persons = person_tracker.update(person_bboxes)

            frame_landmarks_norm = None
            frame_landmarks_px = None

            for person_id, bbox in tracked_persons.items():
                if person_id not in landmarks_history:
                    landmarks_history[person_id] = {i: [] for i in range(33)}

                landmarks_norm, landmarks_px = process_person_with_mediapipe(
                    frame, bbox, pose, width, height
                )

                if landmarks_norm:
                    # Aplicar filtragem para cada landmark
                    for i, (norm, px) in enumerate(zip(landmarks_norm, landmarks_px)):
                        landmarks_history[person_id][i].append(norm)

                        max_history = 30
                        if len(landmarks_history[person_id][i]) > max_history:
                            landmarks_history[person_id][i] = landmarks_history[
                                person_id
                            ][i][-max_history:]

                        # Aplicar filtro selecionado
                        if pose_config.get("filter_type") == "kalman":
                            norm_xyz = norm[:3]
                            filtered = apply_kalman_filter(
                                [lm[:3] for lm in landmarks_history[person_id][i]],
                                norm_xyz,
                            )

                            visibility = norm[3] if len(norm) > 3 else 1.0
                            landmarks_norm[i] = filtered + [visibility]
                            landmarks_px[i] = [
                                int(filtered[0] * width),
                                int(filtered[1] * height),
                                filtered[2],
                                visibility,
                            ]
                        elif (
                            pose_config.get("filter_type") == "savgol"
                            and len(landmarks_history[person_id][i]) >= 5
                        ):
                            filtered = apply_savgol_filter(
                                landmarks_history[person_id][i], norm
                            )
                            landmarks_norm[i] = filtered
                            landmarks_px[i] = [
                                int(filtered[0] * width),
                                int(filtered[1] * height),
                                filtered[2],
                            ]

                    # Estimar pontos ocultos
                    if pose_config.get("estimate_occluded", False):
                        landmarks_norm = estimate_missing_landmarks(landmarks_norm)
                        for i, norm in enumerate(landmarks_norm):
                            if norm is not None and not np.isnan(norm[0]):
                                landmarks_px[i] = [
                                    int(norm[0] * width),
                                    int(norm[1] * height),
                                    norm[2],
                                ]

                    # Selecionar a pessoa principal
                    if frame_landmarks_norm is None:
                        frame_landmarks_norm = landmarks_norm
                        frame_landmarks_px = landmarks_px

            # Se nenhuma pessoa detectada, usar MediaPipe diretamente
            if frame_landmarks_norm is None:
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    landmarks_norm = []
                    landmarks_px = []

                    for landmark in results.pose_landmarks.landmark:
                        landmarks_norm.append([landmark.x, landmark.y, landmark.z])
                        landmarks_px.append(
                            [
                                int(landmark.x * width),
                                int(landmark.y * height),
                                landmark.z,
                            ]
                        )

                    frame_landmarks_norm = landmarks_norm
                    frame_landmarks_px = landmarks_px
        else:
            # Usar apenas MediaPipe
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                frame_landmarks_norm = []
                frame_landmarks_px = []

                for landmark in results.pose_landmarks.landmark:
                    frame_landmarks_norm.append([landmark.x, landmark.y, landmark.z])
                    frame_landmarks_px.append(
                        [int(landmark.x * width), int(landmark.y * height), landmark.z]
                    )
            else:
                frame_landmarks_norm = None
                frame_landmarks_px = None

        # Armazenar resultados
        if frame_landmarks_norm:
            normalized_landmarks_list.append(frame_landmarks_norm)
            pixel_landmarks_list.append(frame_landmarks_px)
        else:
            num_landmarks = len(landmark_names)
            nan_landmarks = [[np.nan, np.nan, np.nan] for _ in range(num_landmarks)]
            normalized_landmarks_list.append(nan_landmarks)
            pixel_landmarks_list.append(nan_landmarks)
            frames_with_missing_data.append(frame_count)

        frame_count += 1

    # Fechar recursos
    cap.release()
    pose.close()

    # Salvar CSVs
    with open(output_file_path, "w") as f_norm, open(
        output_pixel_file_path, "w"
    ) as f_pixel:
        f_norm.write(",".join(headers) + "\n")
        f_pixel.write(",".join(headers) + "\n")

        for frame_idx in range(len(normalized_landmarks_list)):
            landmarks_norm = normalized_landmarks_list[frame_idx]
            landmarks_pixel = pixel_landmarks_list[frame_idx]

            flat_landmarks_norm = [
                coord for landmark in landmarks_norm for coord in landmark
            ]
            flat_landmarks_pixel = [
                coord for landmark in landmarks_pixel for coord in landmark
            ]

            landmarks_norm_str = ",".join(
                "NaN" if np.isnan(value) else f"{value:.6f}"
                for value in flat_landmarks_norm
            )
            landmarks_pixel_str = ",".join(
                "NaN" if np.isnan(value) else str(value)
                for value in flat_landmarks_pixel
            )

            f_norm.write(f"{frame_idx}," + landmarks_norm_str + "\n")
            f_pixel.write(f"{frame_idx}," + landmarks_pixel_str + "\n")

    print(f"\n\nEtapa 2/2: Criando vídeo com landmarks processados")

    # ETAPA 2: Gerar vídeo a partir dos landmarks processados
    cap = cv2.VideoCapture(str(video_path))
    codec = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Estilos de desenho mais visíveis
    landmark_spec = mp_drawing.DrawingSpec(
        color=(0, 255, 0), thickness=2, circle_radius=2
    )
    connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)

    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % 30 == 0:
            progress = (frame_idx / total_frames) * 100
            print(
                f"\rCriando vídeo {frame_idx}/{total_frames} ({progress:.1f}%)", end=""
            )

        # Recuperar landmarks para este frame
        if frame_idx < len(pixel_landmarks_list):
            landmarks_px = pixel_landmarks_list[frame_idx]

            # Verificar se há landmarks válidos
            if not all(np.isnan(lm[0]) for lm in landmarks_px):
                # Se YOLO estiver ativado, também desenhar os bounding boxes
                if pose_config["use_yolo"] and yolo_model:
                    # Detectar pessoas novamente para mostrar todas as bboxes
                    persons = detect_persons_with_yolo(
                        frame, yolo_model, pose_config["yolo_conf"]
                    )
                    for person in persons:
                        bbox = person["bbox"]
                        cv2.rectangle(
                            frame,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (0, 255, 0),
                            2,
                        )

                # Criar um objeto de landmarks para desenho
                landmark_proto = landmark_pb2.NormalizedLandmarkList()

                for i, lm in enumerate(landmarks_px):
                    if not np.isnan(lm[0]):
                        landmark = landmark_proto.landmark.add()
                        landmark.x = lm[0] / width
                        landmark.y = lm[1] / height
                        landmark.z = lm[2] if not np.isnan(lm[2]) else 0
                        landmark.visibility = 1.0

                # Desenhar landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    landmark_proto,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=landmark_spec,
                    connection_drawing_spec=connection_spec,
                )

        out.write(frame)
        frame_idx += 1

    # Fechar recursos
    cap.release()
    out.release()

    end_time = time.time()
    execution_time = end_time - start_time

    # Criar log
    log_info_path = output_dir / "log_info.txt"
    with open(log_info_path, "w") as log_file:
        log_file.write(f"Video Path: {video_path}\n")
        log_file.write(f"Output Video Path: {output_video_path}\n")
        log_file.write(f"Codec: {codec}\n")
        log_file.write(f"Resolution: {width}x{height}\n")
        log_file.write(f"FPS: {fps}\n")
        log_file.write(f"Total Frames: {frame_count}\n")
        log_file.write(f"Execution Time: {execution_time} seconds\n")
        log_file.write(f"MediaPipe Pose Configuration: {pose_config}\n")
        if frames_with_missing_data:
            log_file.write(
                f"Frames with missing data (NaN inserted): {len(frames_with_missing_data)}\n"
            )
        else:
            log_file.write("No frames with missing data.\n")

    print(f"\nCompleted processing {video_path.name}")
    print(f"Output saved to: {output_dir}")
    print(f"Processing time: {execution_time:.2f} seconds\n")


# Classe para rastreamento de pessoas
class PersonTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}  # {ID: bbox}
        self.disappeared = {}  # {ID: count}
        self.max_disappeared = max_disappeared
        self.history = {}  # {ID: [lista de bboxes anteriores]}

    def register(self, bbox):
        self.objects[self.next_id] = bbox
        self.disappeared[self.next_id] = 0
        self.history[self.next_id] = [bbox]
        self.next_id += 1
        return self.next_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.history[object_id]

    def get_best_match(self, bbox, threshold=0.5):
        # IOU matching entre bbox atual e objetos conhecidos
        best_id = None
        best_iou = 0

        for person_id, person_bbox in self.objects.items():
            iou = compute_iou(bbox, person_bbox)
            if iou > best_iou and iou > threshold:
                best_iou = iou
                best_id = person_id

        return best_id, best_iou

    def update(self, bboxes):
        if len(bboxes) == 0:
            # Incrementar contadores de desaparecimento
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Associar detecções com objetos existentes
        matched_ids = []
        assigned_bboxes = set()

        # Para cada bbox atual, encontrar o melhor match
        for i, bbox in enumerate(bboxes):
            best_id, best_iou = self.get_best_match(bbox)

            if best_id is not None:
                self.objects[best_id] = bbox
                self.history[best_id].append(bbox)
                self.disappeared[best_id] = 0
                matched_ids.append(best_id)
                assigned_bboxes.add(i)

        # Registrar novas detecções
        for i, bbox in enumerate(bboxes):
            if i not in assigned_bboxes:
                new_id = self.register(bbox)
                matched_ids.append(new_id)

        # Atualizar objetos desaparecidos
        for object_id in list(self.disappeared.keys()):
            if object_id not in matched_ids:
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

        return self.objects


def compute_iou(box1, box2):
    """Compute IOU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def apply_kalman_filter(landmarks_history, current_landmarks=None):
    """
    Aplica filtro de Kalman em landmarks históricos
    """
    if not landmarks_history:
        return current_landmarks

    if current_landmarks is None:
        current_landmarks = (
            [np.nan, np.nan, np.nan, 1.0]
            if len(landmarks_history[0]) > 3
            else [np.nan, np.nan, np.nan]
        )

    # Determinar dimensões (3 ou 4)
    n_dim = len(landmarks_history[0])

    # Converter para array numpy
    history_array = np.array(landmarks_history)

    # Lidar com dados faltantes
    if np.isnan(current_landmarks[0]):
        if len(landmarks_history) > 0:
            return landmarks_history[-1]
        return current_landmarks

    # Kalman necessita de pelo menos 2 pontos
    if len(history_array) < 2:
        return current_landmarks

    try:
        # Configurar o filtro de Kalman
        kf = KalmanFilter(initial_state_mean=history_array[0], n_dim_obs=n_dim)

        # Ajustar o filtro e aplicar
        smoothed_state_means, _ = kf.smooth(history_array)

        if not np.isnan(current_landmarks[0]):
            full_data = np.vstack([history_array, current_landmarks])
            smoothed_full, _ = kf.smooth(full_data)
            return smoothed_full[-1].tolist()
        else:
            return smoothed_state_means[-1].tolist()
    except Exception as e:
        print(f"Erro no filtro de Kalman: {e}")
        return current_landmarks


def apply_savgol_filter(
    landmarks_history, current_landmarks=None, window_length=5, poly_order=2
):
    """
    Aplica filtro Savitzky-Golay em landmarks históricos
    """
    if not landmarks_history:
        return current_landmarks

    if current_landmarks is None:
        current_landmarks = [np.nan, np.nan, np.nan]

    # Converter para array numpy
    history_array = np.array(landmarks_history)

    # Lidar com dados faltantes
    if np.isnan(current_landmarks[0]):
        if len(landmarks_history) > 0:
            return landmarks_history[-1]
        return current_landmarks

    # SavGol precisa de pontos suficientes e window_length ímpar
    if len(history_array) < window_length:
        return current_landmarks

    try:
        # Ajustar window_length para ser ímpar
        if window_length % 2 == 0:
            window_length -= 1

        # Aplicar o filtro separadamente em x, y, z
        smoothed = []
        for i in range(3):  # x, y, z
            data = history_array[:, i]
            if not np.isnan(current_landmarks[i]):
                data = np.append(data, current_landmarks[i])

            # Aplicar o filtro
            filtered = savgol_filter(data, window_length, poly_order)
            smoothed.append(filtered[-1])

        return smoothed
    except Exception as e:
        print(f"Erro no filtro Savgol: {e}")
        return current_landmarks


def estimate_missing_landmarks(landmarks, visibility_threshold=0.5):
    """
    Estima landmarks ausentes/ocultos baseado na anatomia
    """
    if landmarks is None:
        return None

    # Definir restrições anatômicas conhecidas
    # Exemplo: distância constante entre ombros/quadris, simetria, etc.

    # Copiar landmarks para não modificar o original
    estimated = []
    for i, lm in enumerate(landmarks):
        if (
            lm is None
            or lm[0] is None
            or np.isnan(lm[0])
            or (len(lm) > 3 and lm[3] < visibility_threshold)
        ):
            # Landmark ausente ou com baixa visibilidade
            # Aplicar regras específicas baseadas em conhecimento anatômico

            # Exemplo: estimar posição de cotovelo com base no ombro e pulso
            if i == 13:  # left_elbow
                if (
                    landmarks[11] and landmarks[15]
                ):  # left_shoulder e left_wrist estão visíveis
                    # Interpolar linearmente entre ombro e pulso
                    shoulder = landmarks[11]
                    wrist = landmarks[15]
                    estimated.append(
                        [
                            (shoulder[0] + wrist[0]) / 2,
                            (shoulder[1] + wrist[1]) / 2,
                            (shoulder[2] + wrist[2]) / 2,
                            0.5,  # confiança média
                        ]
                    )
                else:
                    estimated.append(lm)
            elif i == 14:  # right_elbow
                if (
                    landmarks[12] and landmarks[16]
                ):  # right_shoulder e right_wrist estão visíveis
                    shoulder = landmarks[12]
                    wrist = landmarks[16]
                    estimated.append(
                        [
                            (shoulder[0] + wrist[0]) / 2,
                            (shoulder[1] + wrist[1]) / 2,
                            (shoulder[2] + wrist[2]) / 2,
                            0.5,  # confiança média
                        ]
                    )
                else:
                    estimated.append(lm)

            # Adicionar mais regras para outros pontos anatômicos
            # ...

            else:
                estimated.append(lm)
        else:
            estimated.append(lm)

    return estimated


def apply_anatomical_constraints(landmarks):
    """
    Aplica restrições anatômicas para melhorar a coerência
    """
    if landmarks is None:
        return None

    # Copiar landmarks para não modificar o original
    constrained = []

    # Aplicar restrições anatômicas conhecidas
    # Exemplo 1: Manter proporções corporais consistentes

    # Exemplo 2: Garantir simetria bilateral (se aplicável)

    # Exemplo 3: Limitar ângulos articulares a valores plausíveis
    # (por exemplo, joelhos e cotovelos não dobram para trás)

    return constrained


def landmarks_to_mp_format(landmarks_px, width, height):
    """Converte landmarks em formato de lista para formato MediaPipe"""
    landmark_list = landmark_pb2.NormalizedLandmarkList()

    # Preencher com os landmarks
    for i, lm in enumerate(landmarks_px):
        landmark = landmark_list.landmark.add()
        landmark.x = lm[0] / width  # Normalizar para 0-1
        landmark.y = lm[1] / height  # Normalizar para 0-1
        landmark.z = lm[2] if len(lm) > 2 else 0
        landmark.visibility = lm[3] if len(lm) > 3 else 1.0

    return landmark_list


def process_videos_in_directory():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")

    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(
        title="Select the input directory containing videos"
    )
    if not input_dir:
        messagebox.showerror("Error", "No input directory selected.")
        return

    output_base = filedialog.askdirectory(title="Select the base output directory")
    if not output_base:
        messagebox.showerror("Error", "No output directory selected.")
        return

    pose_config = get_pose_config()
    if not pose_config:
        return

    # Carregar modelo YOLO se necessário
    yolo_model = None
    if pose_config["use_yolo"]:
        yolo_model = download_or_load_yolo_model()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(output_base) / f"mediapipe_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_dir)
    video_files = list(input_dir.glob("*.*"))
    video_files = [
        f for f in video_files if f.suffix.lower() in [".mp4", ".avi", ".mov"]
    ]

    print(f"\nFound {len(video_files)} videos to process")

    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file.name}")
        output_dir = output_base / video_file.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        process_video(video_file, output_dir, pose_config, yolo_model)


if __name__ == "__main__":
    process_videos_in_directory()
