"""
Script: markerless2d_mpyolo.py
Author: Prof. Dr. Paulo Santiago
Version: 0.0.5
Last Updated: January 24, 2025

Description:
This script combines YOLOv11 for person detection/tracking with MediaPipe for pose estimation.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import datetime
import pandas as pd

# Configurações para evitar conflitos
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

def initialize_csv(output_dir, person_id):
    """Inicializa um arquivo CSV para uma pessoa específica."""
    csv_path = os.path.join(output_dir, f'person_{person_id}_landmarks.csv')
    
    # Define os nomes das colunas
    columns = ['frame', 'person_id']
    # Usando os nomes corretos dos landmarks do MediaPipe
    for idx in range(33):  # MediaPipe Pose tem 33 landmarks
        columns.extend([f'landmark_{idx}_x', f'landmark_{idx}_y', 
                       f'landmark_{idx}_z', f'landmark_{idx}_visibility'])
    
    # Cria o DataFrame com as colunas definidas
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, index=False)
    return csv_path

def save_landmarks_to_csv(csv_path, frame_idx, person_id, landmarks):
    """Salva os dados dos landmarks no CSV."""
    row_data = {'frame': frame_idx, 'person_id': person_id}
    
    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            prefix = f'landmark_{idx}'
            row_data[f'{prefix}_x'] = landmark.x
            row_data[f'{prefix}_y'] = landmark.y
            row_data[f'{prefix}_z'] = landmark.z
            row_data[f'{prefix}_visibility'] = landmark.visibility
    else:
        # Preenche com NaN quando não há landmarks
        for idx in range(33):
            prefix = f'landmark_{idx}'
            row_data[f'{prefix}_x'] = np.nan
            row_data[f'{prefix}_y'] = np.nan
            row_data[f'{prefix}_z'] = np.nan
            row_data[f'{prefix}_visibility'] = np.nan
    
    # Append ao CSV existente
    df = pd.DataFrame([row_data])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)

def run_markerless2d_mpyolo():
    root = tk.Tk()
    root.withdraw()

    video_dir = filedialog.askdirectory(title="Select Input Directory")
    if not video_dir:
        print("No input directory selected. Exiting...")
        return

    output_base_dir = filedialog.askdirectory(title="Select Output Directory")
    if not output_base_dir:
        print("No output directory selected. Exiting...")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_base_dir, f"markerless2d_mpyolo_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    # Inicializa YOLOv11 e MediaPipe
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolo11x.pt')
    model = YOLO(model_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        smooth_landmarks=False,
        min_detection_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    mp_drawing = mp.solutions.drawing_utils

    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(video_dir, video_file)
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(main_output_dir, video_name)
            os.makedirs(output_dir, exist_ok=True)

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_video_path = os.path.join(output_dir, f"processed_{video_name}.mp4")
            writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

            person_csv_files = {}
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Tracking com YOLOv11
                results = model.track(frame, persist=True, stream=True, classes=0)
                detected_ids = set()

                for result in results:
                    if result.boxes.id is not None:
                        for box, track_id in zip(result.boxes, result.boxes.id):
                            x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                            person_id = int(track_id.item())
                            detected_ids.add(person_id)

                            # Desenha bounding box e ID
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(frame, f"ID: {person_id}", (x_min, y_min - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Processa cada pessoa individualmente com MediaPipe
                            person_crop = frame[y_min:y_max, x_min:x_max]
                            if person_crop.size > 0:  # Verifica se o crop é válido
                                # Adiciona padding para melhorar a detecção
                                padding = 20
                                padded_crop = cv2.copyMakeBorder(
                                    person_crop,
                                    padding, padding, padding, padding,
                                    cv2.BORDER_CONSTANT,
                                    value=[0, 0, 0]
                                )
                                
                                # Processa com MediaPipe
                                crop_rgb = cv2.cvtColor(padded_crop, cv2.COLOR_BGR2RGB)
                                results_pose = pose.process(crop_rgb)

                                if results_pose.pose_landmarks:
                                    # Ajusta as coordenadas dos landmarks para o frame original
                                    for landmark in results_pose.pose_landmarks.landmark:
                                        # Remove o padding das coordenadas
                                        x_unpadded = (landmark.x * padded_crop.shape[1] - padding) / person_crop.shape[1]
                                        y_unpadded = (landmark.y * padded_crop.shape[0] - padding) / person_crop.shape[0]
                                        
                                        # Converte para coordenadas globais
                                        landmark.x = (x_unpadded * (x_max - x_min) + x_min) / width
                                        landmark.y = (y_unpadded * (y_max - y_min) + y_min) / height

                                    # Desenha landmarks
                                    mp_drawing.draw_landmarks(
                                        frame,
                                        results_pose.pose_landmarks,
                                        mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                                            color=(0, 0, 255),  # Vermelho para pontos
                                            thickness=2,
                                            circle_radius=2
                                        ),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(
                                            color=(255, 255, 255),  # Branco para conexões
                                            thickness=1
                                        )
                                    )

                                    # Inicializa ou atualiza CSV para esta pessoa
                                    if person_id not in person_csv_files:
                                        person_csv_files[person_id] = initialize_csv(output_dir, person_id)
                                    
                                    # Salva dados dos landmarks
                                    save_landmarks_to_csv(person_csv_files[person_id], 
                                                       frame_idx, person_id, 
                                                       results_pose.pose_landmarks)
                                else:
                                    # Se não detectou pose, salva NaN
                                    if person_id not in person_csv_files:
                                        person_csv_files[person_id] = initialize_csv(output_dir, person_id)
                                    save_landmarks_to_csv(person_csv_files[person_id], frame_idx, person_id, None)

                # Adiciona linha com NaN para pessoas não detectadas
                for pid in person_csv_files.keys():
                    if pid not in detected_ids:
                        save_landmarks_to_csv(person_csv_files[pid], frame_idx, pid, None)

                writer.write(frame)
                frame_idx += 1

                if frame_idx % 30 == 0:
                    print(f"\rProcessing frame {frame_idx}/{total_frames} "
                          f"({(frame_idx/total_frames)*100:.1f}%)", end="")

            cap.release()
            writer.release()
            print(f"\nProcessamento concluído para {video_file}.")
            print(f"Resultados salvos em: '{output_dir}'")

    root.destroy()

if __name__ == "__main__":
    run_markerless2d_mpyolo()