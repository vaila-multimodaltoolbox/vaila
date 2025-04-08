import cv2
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import mediapipe as mp
import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox, simpledialog

# Configuration settings
# Camera settings
CAMERA_DEVICE = 0  # Default camera device (0 for built-in webcam)
CAMERA_FPS = 30
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for YOLO detections
BUFFER_SIZE = 100  # Number of frames to keep in the angle buffer

# Output settings
SAVE_DATA = True
OUTPUT_DIR = "output"  # Directory to save output files

# Default engine
ENGINE = "yolo"  # Options: "yolo" or "mediapipe"

# ===== CÓDIGO DO ANGLE_CALCULATOR.PY =====
class AngleCalculator:
    """Classe base para calculadores de ângulos."""
    def calculate_angle(self, p1, p2, p3):
        """Calcula o ângulo entre três pontos."""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # Lidar com erros de precisão numérica
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)    

class YOLOAngleCalculator(AngleCalculator):
    """Calculador de ângulos usando pontos-chave do YOLO."""
    def __init__(self):
        super().__init__()
        # Mapeamento atualizado das articulações YOLO para YOLOv11n-pose
        # Baseado no output que mostra 7 pontos válidos (0-6)
        self.joint_map = {
            'pescoco': [1, 0, 2],  # nariz-pescoço-ombro_esquerdo
            'ombro_direito': [0, 1, 5],  # pescoço-ombro_direito-braco_direito
            'ombro_esquerdo': [0, 2, 6],  # pescoço-ombro_esquerdo-braco_esquerdo
            'braco_direito': [1, 5, 3],  # ombro_direito-braco_direito-cotovelo_direito
            'braco_esquerdo': [2, 6, 4]  # ombro_esquerdo-braco_esquerdo-cotovelo_esquerdo
        }
        
        # Default keypoint structure (YOLOv11n-pose)
        self.num_keypoints = 17
        
    def adapt_to_keypoint_structure(self, num_keypoints):
        """Adapt the joint map to the detected keypoint structure."""
        self.num_keypoints = num_keypoints
        print(f"Adapting to keypoint structure with {num_keypoints} keypoints")
        
        # Para o YOLOv11n-pose, manteremos o mapeamento padrão
        # pois já está otimizado para a estrutura de keypoints deste modelo
        print("Using YOLOv11n-pose keypoint structure")
        
    def process_keypoints(self, keypoints):
        """Processa os pontos-chave do YOLO e calcula os ângulos."""
        angles = {}
        
        # Debug: Print keypoints shape and content
        print(f"Keypoints shape: {keypoints.shape}")
        print(f"Processing keypoints...")
        
        # Verificar se os keypoints têm confiança suficiente (terceira coluna)
        confidence_threshold = 0.3  # Reduzido para 0.3 para capturar mais pontos
        valid_keypoints = keypoints[:, 2] > confidence_threshold
        
        # Criar máscara para keypoints válidos (coordenadas diferentes de zero e confiança alta)
        valid_mask = np.logical_and(
            valid_keypoints,
            np.logical_and(
                keypoints[:, 0] != 0,  # x não é zero
                keypoints[:, 1] != 0   # y não é zero
            )
        )
        
        print(f"Valid keypoints mask: {valid_mask}")
        print(f"Valid keypoints coordinates:")
        for i, valid in enumerate(valid_mask):
            if valid:
                print(f"Point {i}: ({keypoints[i][0]:.1f}, {keypoints[i][1]:.1f}), conf: {keypoints[i][2]:.3f}")
        
        for joint_name, indices in self.joint_map.items():
            p1_idx, p2_idx, p3_idx = indices
            
            # Verificar se todos os pontos necessários são válidos
            if valid_mask[p1_idx] and valid_mask[p2_idx] and valid_mask[p3_idx]:
                p1 = keypoints[p1_idx][:2]  # Usar apenas x,y
                p2 = keypoints[p2_idx][:2]
                p3 = keypoints[p3_idx][:2]
                
                angle = self.calculate_angle(p1, p2, p3)
                angles[joint_name] = angle
                print(f"Calculated {joint_name}: {angle:.1f}°")
                print(f"  Points used: {p1_idx}({p1}), {p2_idx}({p2}), {p3_idx}({p3})")
        
        print(f"Final angles: {angles}")
        return angles

class MediaPipeAngleCalculator(AngleCalculator):
    """Calculador de ângulos usando pontos-chave do MediaPipe."""
    def __init__(self):
        super().__init__()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Mapeamento das articulações MediaPipe
        self.joint_map = {
            'cotovelo_direito': [self.mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                self.mp_pose.PoseLandmark.RIGHT_ELBOW, 
                                self.mp_pose.PoseLandmark.RIGHT_WRIST],
            'cotovelo_esquerdo': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                self.mp_pose.PoseLandmark.LEFT_ELBOW, 
                                self.mp_pose.PoseLandmark.LEFT_WRIST],
            'ombro_direito': [self.mp_pose.PoseLandmark.RIGHT_HIP, 
                            self.mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                            self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            'ombro_esquerdo': [self.mp_pose.PoseLandmark.LEFT_HIP, 
                            self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                            self.mp_pose.PoseLandmark.LEFT_ELBOW],
            'quadril_direito': [self.mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                self.mp_pose.PoseLandmark.RIGHT_HIP, 
                                self.mp_pose.PoseLandmark.RIGHT_KNEE],
            'quadril_esquerdo': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                self.mp_pose.PoseLandmark.LEFT_HIP, 
                                self.mp_pose.PoseLandmark.LEFT_KNEE],
            'joelho_direito': [self.mp_pose.PoseLandmark.RIGHT_HIP, 
                            self.mp_pose.PoseLandmark.RIGHT_KNEE, 
                            self.mp_pose.PoseLandmark.RIGHT_ANKLE],
            'joelho_esquerdo': [self.mp_pose.PoseLandmark.LEFT_HIP, 
                            self.mp_pose.PoseLandmark.LEFT_KNEE, 
                            self.mp_pose.PoseLandmark.LEFT_ANKLE],
            'tornozelo_direito': [self.mp_pose.PoseLandmark.RIGHT_KNEE, 
                                self.mp_pose.PoseLandmark.RIGHT_ANKLE, 
                                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
            'tornozelo_esquerdo': [self.mp_pose.PoseLandmark.LEFT_KNEE, 
                                self.mp_pose.PoseLandmark.LEFT_ANKLE, 
                                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        }

    def process_frame(self, frame):
        """Processa um frame com MediaPipe e calcula os ângulos."""
        # Converter BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        angles = {}
        skeleton_connections = []
        visible_landmarks = []
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calcular ângulos para cada articulação definida
            for joint_name, indices in self.joint_map.items():
                p1_idx, p2_idx, p3_idx = indices
                
                # Verificar se os landmarks necessários são visíveis
                if (landmarks[p1_idx].visibility > 0.5 and 
                    landmarks[p2_idx].visibility > 0.5 and 
                    landmarks[p3_idx].visibility > 0.5):
                    
                    p1 = [landmarks[p1_idx].x * frame.shape[1], 
                          landmarks[p1_idx].y * frame.shape[0]]
                    p2 = [landmarks[p2_idx].x * frame.shape[1], 
                          landmarks[p2_idx].y * frame.shape[0]]
                    p3 = [landmarks[p3_idx].x * frame.shape[1], 
                          landmarks[p3_idx].y * frame.shape[0]]
                    
                    angle = self.calculate_angle(p1, p2, p3)
                    angles[joint_name] = angle
                    
                    # Adicionar conexões para desenhar o esqueleto
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    pt3 = (int(p3[0]), int(p3[1]))
                    
                    skeleton_connections.append((pt1, pt2))
                    skeleton_connections.append((pt2, pt3))
            
            # Obter coordenadas de todos os landmarks visíveis para desenhar
            visible_landmarks = []
            for i, landmark in enumerate(landmarks):
                if landmark.visibility > 0.5:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    visible_landmarks.append((x, y, i))
        
        return angles, skeleton_connections, visible_landmarks
# ===== FIM DO CÓDIGO DO ANGLE_CALCULATOR.PY =====

def download_model(model_name):
    """
    Download a specific YOLO model to the vaila/vaila/models directory.

    Args:
        model_name: Name of the model to download (e.g., "yolov11n.pt")

    Returns:
        Path to the downloaded model
    """
    # Correto caminho para vaila/vaila/models
    script_dir = os.path.dirname(os.path.abspath(__file__))  # vaila/
    vaila_dir = os.path.dirname(script_dir)  # root directory
    models_dir = os.path.join(vaila_dir, "vaila", "models")  # vaila/vaila/models

    # Create the models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    print(f"Models will be downloaded to: {models_dir}")

    model_path = os.path.join(models_dir, model_name)

    # Check if model already exists
    if os.path.exists(model_path):
        print(
            f"Model {model_name} already exists at {model_path}, using existing file."
        )
        return model_path

    print(f"Downloading {model_name} to {model_path}...")
    try:
        # Create a temporary YOLO model instance that will download the weights
        model = YOLO(model_name)

        # Get the path where YOLO downloaded the model
        source_path = model.ckpt_path

        if os.path.exists(source_path):
            # Copy the downloaded model to our models directory
            import shutil

            shutil.copy2(source_path, model_path)
            print(f"Successfully saved {model_name} to {model_path}")
        else:
            print(f"YOLO downloaded the model but couldn't find it at {source_path}")

    except Exception as e:
        print(f"Error downloading {model_name}: {e}")
        print("Trying alternative download method...")

        try:
            # Alternative download method using requests
            import requests
            
            # URL for the model - updated to use the correct model name and version
            if model_name.lower().startswith("yolo11"):
                version_tag = "v11.0.0"
            else:
                version_tag = "v0.0.0"
            url = f"https://github.com/ultralytics/assets/releases/download/{version_tag}/{model_name}"
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Save the file
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print(f"Successfully downloaded {model_name} using requests")
        except Exception as e2:
            print(f"All download methods failed for {model_name}: {e2}")
            print("Trying to find the model in the local directory...")
            
            # Try to find the model in the local directory
            local_model_path = os.path.join(script_dir, model_name)
            if os.path.exists(local_model_path):
                print(f"Found model at {local_model_path}, copying to {model_path}")
                import shutil
                shutil.copy2(local_model_path, model_path)
                return model_path
            else:
                print(f"Could not find model {model_name} locally or download it.")
                print("Please manually download the model and place it in the models directory.")

    return model_path

class MovementAnalyzer:
    def __init__(self, engine='yolo'):
        self.engine = engine.lower()
        
        # Inicializar o modelo YOLO para detecção
        if self.engine == 'yolo':
            # Use o modelo YOLOv11n-pose
            model_name = "yolo11n-pose.pt"  # Usar o modelo YOLOv11n-pose
            self.model = YOLO(self.get_model_path(model_name), verbose=False)
            self.angle_calculator = YOLOAngleCalculator()
            # Set lower confidence threshold for better detection
            self.conf_threshold = 0.3
            
            # Detect YOLO model type and adapt keypoint processing
            self.detect_yolo_model_type()
        elif self.engine == 'mediapipe':
            self.model = None  # Não precisamos do modelo YOLO para MediaPipe
            self.angle_calculator = MediaPipeAngleCalculator()
            # Definir as conexões do esqueleto para MediaPipe
            self.mp_pose = self.angle_calculator.mp_pose
            self.mediapipe_connections = [
                # Braço direito
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
                (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_PINKY),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_INDEX),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_THUMB),
                (self.mp_pose.PoseLandmark.RIGHT_PINKY, self.mp_pose.PoseLandmark.RIGHT_INDEX),
                
                # Braço esquerdo
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
                (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_PINKY),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_INDEX),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_THUMB),
                (self.mp_pose.PoseLandmark.LEFT_PINKY, self.mp_pose.PoseLandmark.LEFT_INDEX),
                
                # Tronco
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
                
                # Perna direita
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
                (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
                (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL),
                (self.mp_pose.PoseLandmark.RIGHT_HEEL, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                
                # Perna esquerda
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
                (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
                (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
                (self.mp_pose.PoseLandmark.LEFT_HEEL, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                
                # Face (opcional, mas adiciona completude ao esqueleto)
                (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_EYE_INNER),
                (self.mp_pose.PoseLandmark.LEFT_EYE_INNER, self.mp_pose.PoseLandmark.LEFT_EYE),
                (self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.LEFT_EYE_OUTER),
                (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_EYE_INNER),
                (self.mp_pose.PoseLandmark.RIGHT_EYE_INNER, self.mp_pose.PoseLandmark.RIGHT_EYE),
                (self.mp_pose.PoseLandmark.RIGHT_EYE, self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
                (self.mp_pose.PoseLandmark.MOUTH_LEFT, self.mp_pose.PoseLandmark.MOUTH_RIGHT),
                #(self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_EAR),
                #(self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_EAR),
                
                # Novas conexões entre olhos e orelhas
                #(self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.LEFT_EAR),
                #(self.mp_pose.PoseLandmark.RIGHT_EYE, self.mp_pose.PoseLandmark.RIGHT_EAR),
                (self.mp_pose.PoseLandmark.LEFT_EYE_OUTER, self.mp_pose.PoseLandmark.LEFT_EAR),
                (self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER, self.mp_pose.PoseLandmark.RIGHT_EAR),
                #(self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_EAR),
                #(self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_EAR),
            ]
        else:
            raise ValueError(f"Engine não reconhecido: {engine}. Use 'yolo' ou 'mediapipe'.")
        
        self.cap = cv2.VideoCapture(CAMERA_DEVICE)
        self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        self.angle_buffer = []
        self.start_time = time.time()

    def get_model_path(self, model_name="yolo11n-pose.pt"):
        """Get the path to the YOLO model, downloading it if necessary."""
        # Get the correct path relative to the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vaila_dir = os.path.dirname(script_dir)
        models_dir = os.path.join(vaila_dir, "vaila", "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, model_name)
        
        # Download if not exists
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, downloading...")
            model_path = download_model(model_name)
        else:
            print(f"Using existing model at: {model_path}")
        
        return model_path

    def detect_yolo_model_type(self):
        """Detect the YOLO model type and adapt keypoint processing accordingly."""
        # Create a small test frame
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Run inference on the test frame
        results = self.model(test_frame, conf=self.conf_threshold, verbose=False)
        
        # Check if keypoints are available
        if len(results) > 0:
            print(f"YOLO results type: {type(results)}")
            print(f"First result attributes: {dir(results[0])}")
            
            # Try to find keypoints in different possible locations
            keypoints = None
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                print("Found keypoints in results[0].keypoints")
                if len(results[0].keypoints.data) > 0:
                    keypoints = results[0].keypoints.data[0].cpu().numpy()
                    print(f"Keypoints from results[0].keypoints: {keypoints.shape}")
            elif hasattr(results[0], 'poses') and results[0].poses is not None:
                print("Found poses in results[0].poses")
                if len(results[0].poses) > 0:
                    keypoints = results[0].poses[0].cpu().numpy()
                    print(f"Keypoints from results[0].poses: {keypoints.shape}")
            
            if keypoints is not None:
                # Get the keypoints shape
                keypoints_shape = keypoints.shape
                print(f"Detected YOLO model with keypoints shape: {keypoints_shape}")
                
                # Adapt the angle calculator based on the keypoints shape
                if len(keypoints_shape) >= 2:
                    num_keypoints = keypoints_shape[1] if len(keypoints_shape) > 1 else keypoints_shape[0]
                    print(f"Number of keypoints: {num_keypoints}")
                    
                    # Update the angle calculator with the detected keypoint structure
                    self.angle_calculator.adapt_to_keypoint_structure(num_keypoints)
            else:
                print("Warning: Could not detect keypoints in the YOLO model. Using default keypoint structure.")
                print("Available attributes in results[0]:", dir(results[0]))
                
                # Try to find pose-related attributes
                for attr in dir(results[0]):
                    if 'pose' in attr.lower() or 'keypoint' in attr.lower() or 'landmark' in attr.lower():
                        print(f"Found potential pose-related attribute: {attr}")
                        try:
                            value = getattr(results[0], attr)
                            print(f"Value type: {type(value)}")
                            if hasattr(value, 'shape'):
                                print(f"Shape: {value.shape}")
                        except Exception as e:
                            print(f"Error accessing attribute {attr}: {e}")
        else:
            print("No results from YOLO model on test frame")

    def process_frame(self, frame):
        """Processa um único quadro e detecta poses."""
        processed_frame = frame.copy()
        
        if self.engine == 'yolo':
            # Usando YOLO para detecção e cálculo de ângulos
            # Use a lower confidence threshold for better detection
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            # Debug: Print the structure of results to understand what's available
            if len(results) > 0:
                print(f"YOLO results type: {type(results)}")
                print(f"First result attributes: {dir(results[0])}")
                
                # Check for pose-related attributes
                pose_attrs = [attr for attr in dir(results[0]) if 'pose' in attr.lower() or 'keypoint' in attr.lower() or 'landmark' in attr.lower()]
                if pose_attrs:
                    print(f"Found pose-related attributes: {pose_attrs}")
                
                # Try to find keypoints in different possible locations
                keypoints = None
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    print("Found keypoints in results[0].keypoints")
                    if len(results[0].keypoints.data) > 0:
                        keypoints = results[0].keypoints.data[0].cpu().numpy()
                        print(f"Keypoints from results[0].keypoints: {keypoints.shape}")
                elif hasattr(results[0], 'poses') and results[0].poses is not None:
                    print("Found poses in results[0].poses")
                    if len(results[0].poses) > 0:
                        keypoints = results[0].poses[0].cpu().numpy()
                        print(f"Keypoints from results[0].poses: {keypoints.shape}")
                elif hasattr(results[0], 'landmarks') and results[0].landmarks is not None:
                    print("Found landmarks in results[0].landmarks")
                    if len(results[0].landmarks) > 0:
                        keypoints = results[0].landmarks[0].cpu().numpy()
                        print(f"Keypoints from results[0].landmarks: {keypoints.shape}")
                
                if keypoints is not None:
                    print(f"Keypoints shape: {keypoints.shape}")
                    print(f"Keypoints content: {keypoints}")
                    
                    # Desenhar caixa delimitadora (bounding box) ao redor da pessoa
                    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        boxes = results[0].boxes.cpu().numpy()
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            # Desenhar retângulo ao redor da pessoa
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            # Adicionar rótulo "Pessoa" com confiança
                            label = f"Pessoa {conf:.2f}"
                            cv2.putText(processed_frame, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Definir conexões do esqueleto (pares de índices dos pontos-chave)
                    # Adaptar para o formato de keypoints do YOLO12n
                    skeleton = [
                        # Corpo superior
                        (5, 7), (7, 9), (6, 8), (8, 10),
                        # Tronco
                        (5, 6), (5, 11), (6, 12), (11, 12),
                        # Corpo inferior
                        (11, 13), (13, 15), (12, 14), (14, 16)
                    ]
                    
                    # Desenhar linhas do esqueleto
                    for connection in skeleton:
                        # Verificar se os índices estão dentro dos limites do array de keypoints
                        if (connection[0] < len(keypoints) and connection[1] < len(keypoints) and
                            all(keypoints[connection[0]] != 0) and all(keypoints[connection[1]] != 0)):
                            pt1 = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
                            pt2 = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
                            cv2.line(processed_frame, pt1, pt2, (0, 255, 0), 2)
                    
                    # Calcular ângulos utilizando o calculador YOLO
                    angles = self.angle_calculator.process_keypoints(keypoints)
                    print(f"Calculated angles: {angles}")
                    
                    # Desenhar pontos-chave (excluindo pontos-chave do rosto)
                    for i, kp in enumerate(keypoints):
                        if i < len(keypoints) and all(kp != 0) and i >= 5:  # Ignorar pontos-chave do rosto (0-4)
                            cv2.circle(processed_frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)
                    
                    # Armazenar ângulos no buffer se disponíveis
                    if angles:
                        self.angle_buffer.append({
                            'timestamp': time.time() - self.start_time,
                            'angles': angles
                        })
                        
                        # Manter apenas as últimas medições BUFFER_SIZE
                        if len(self.angle_buffer) > BUFFER_SIZE:
                            self.angle_buffer.pop(0)
                else:
                    print("No keypoints detected in this frame")
                    # Try to find any pose-related data
                    for attr in dir(results[0]):
                        if 'pose' in attr.lower() or 'keypoint' in attr.lower() or 'landmark' in attr.lower():
                            try:
                                value = getattr(results[0], attr)
                                print(f"Attribute {attr} type: {type(value)}")
                                if hasattr(value, 'shape'):
                                    print(f"Attribute {attr} shape: {value.shape}")
                            except Exception as e:
                                print(f"Error accessing attribute {attr}: {e}")
        
        elif self.engine == 'mediapipe':
            # Usar MediaPipe para detecção e cálculo de ângulos
            angles, _, landmarks = self.angle_calculator.process_frame(frame)
            
            # Processar o frame com MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.angle_calculator.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                
                # Criar caixa delimitadora ao redor da pessoa
                x_coords = []
                y_coords = []
                for landmark in pose_landmarks:
                    if landmark.visibility > 0.5:
                        x_coords.append(landmark.x * frame.shape[1])
                        y_coords.append(landmark.y * frame.shape[0])
                
                if x_coords and y_coords:
                    # Calcular as coordenadas da caixa delimitadora
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    # Adicionar algum padding à caixa
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    # Desenhar retângulo ao redor da pessoa
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Adicionar rótulo "Pessoa"
                    cv2.putText(processed_frame, "Pessoa", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Desenhar as conexões do esqueleto usando as definições personalizadas
                for connection in self.mediapipe_connections:
                    start_idx, end_idx = connection
                    
                    if (pose_landmarks[start_idx].visibility > 0.5 and
                        pose_landmarks[end_idx].visibility > 0.5):
                        
                        start_point = (
                            int(pose_landmarks[start_idx].x * frame.shape[1]),
                            int(pose_landmarks[start_idx].y * frame.shape[0])
                        )
                        
                        end_point = (
                            int(pose_landmarks[end_idx].x * frame.shape[1]),
                            int(pose_landmarks[end_idx].y * frame.shape[0])
                        )
                        
                        cv2.line(processed_frame, start_point, end_point, (0, 255, 0), 2)
                
                # Desenhar landmarks visíveis
                for i, landmark in enumerate(pose_landmarks):
                    if landmark.visibility > 0.5:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(processed_frame, (x, y), 4, (0, 0, 255), -1)
            
            # Armazenar ângulos no buffer se disponíveis
            if angles:
                self.angle_buffer.append({
                    'timestamp': time.time() - self.start_time,
                    'angles': angles
                })
                
                # Manter apenas as últimas medições BUFFER_SIZE
                if len(self.angle_buffer) > BUFFER_SIZE:
                    self.angle_buffer.pop(0)
        
        # Desenhar ângulos no quadro (independente do motor)
        if self.angle_buffer and 'angles' in self.angle_buffer[-1]:
            angles = self.angle_buffer[-1]['angles']
            
            # Configurações de texto
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            font_color = (255, 255, 255)  # Branco
            bg_color = (0, 0, 0)  # Preto
            
            # Calcular tamanho do texto para criar fundo
            max_text_width = 0
            for joint, angle in angles.items():
                text = f"{joint}: {angle:.1f}°"
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                max_text_width = max(max_text_width, text_width)
            
            # Criar fundo semi-transparente
            padding = 10
            rows = (len(angles) + 1) // 2  # Dividir em duas colunas
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, 
                         (10, 10), 
                         (max_text_width + 2*padding + 10, (rows * 35) + 2*padding + 10),
                         bg_color, -1)
            cv2.addWeighted(overlay, 0.5, processed_frame, 0.5, 0, processed_frame)
            
            # Desenhar ângulos em duas colunas
            y_offset = 35
            x_offset = 15
            col = 0
            for joint, angle in angles.items():
                text = f"{joint}: {angle:.1f}°"
                # Desenhar texto com contorno para melhor visibilidade
                cv2.putText(processed_frame, text, (x_offset, y_offset),
                           font, font_scale, (0, 0, 0), font_thickness + 1)  # Contorno
                cv2.putText(processed_frame, text, (x_offset, y_offset),
                           font, font_scale, font_color, font_thickness)  # Texto
                
                col += 1
                if col % 2 == 0:
                    y_offset += 35
                    x_offset = 15
                else:
                    x_offset = max_text_width + 30
        
        # Mostrar qual motor está sendo usado
        cv2.putText(processed_frame, f"Motor: {self.engine.upper()}", 
                   (processed_frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return processed_frame
    
    def save_data(self):
        """Salvar dados coletados em arquivo CSV."""
        if SAVE_DATA and self.angle_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Salvar como CSV
            csv_filename = f"{OUTPUT_DIR}/dados_dos_movimentos_{self.engine}_{timestamp}.csv"
            if self.angle_buffer:
                # Obter todos os nomes de articulações únicos
                joint_names = set()
                for data in self.angle_buffer:
                    joint_names.update(data['angles'].keys())
                joint_names = sorted(list(joint_names))
                
                # Escrever cabeçalho CSV
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp'] + joint_names)
                    
                    # Escrever linhas de dados
                    for data in self.angle_buffer:
                        row = [data['timestamp']]
                        for joint in joint_names:
                            row.append(data['angles'].get(joint, ''))
                        writer.writerow(row)
            
            # Criar gráfico de ângulos
            if self.angle_buffer:
                plt.figure(figsize=(10, 6))
                times = [d['timestamp'] for d in self.angle_buffer]
                
                for joint in joint_names:
                    angles = []
                    for d in self.angle_buffer:
                        if joint in d['angles']:
                            angles.append(d['angles'][joint])
                        else:
                            angles.append(None)  # Usar None para dados ausentes
                    
                    # Filtrar None antes de plotar
                    valid_times = []
                    valid_angles = []
                    for t, a in zip(times, angles):
                        if a is not None:
                            valid_times.append(t)
                            valid_angles.append(a)
                    
                    if valid_times and valid_angles:
                        plt.plot(valid_times, valid_angles, label=joint)
                
                plt.xlabel('Tempo (s)')
                plt.ylabel('Ângulo (graus)')
                plt.title(f'Ângulos das articulações ao longo do tempo - {self.engine.upper()}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{OUTPUT_DIR}/angulos_articulacoes_{self.engine}_{timestamp}.png")
                plt.close()
    
    def run(self):
        """Loop principal para o analisador de movimento."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Processar quadro
                processed_frame = self.process_frame(frame)
                
                # Exibir quadro
                cv2.imshow('Analisador de Movimento', processed_frame)
                # Quebrar loop com 'q' pressionado
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.save_data()
            self.cap.release()
            cv2.destroyAllWindows()

def run_markerless_live():
    """
    Run the markerless live analysis tool.
    
    This function prompts the user to select an engine (YOLO or MediaPipe),
    then initializes and runs the MovementAnalyzer with the selected engine.
    """
    # Create a root window but hide it
    root = tk.Tk()
    root.withdraw()
    
    # Ask user to select the engine
    engine_choice = simpledialog.askstring(
        "Select Engine",
        "Choose the analysis engine:\n\n1: YOLO (Better for multiple people)\n2: MediaPipe (Faster, single person)",
        initialvalue="1"
    )
    
    # Map the choice to the engine name
    if engine_choice == "1":
        selected_engine = "yolo"
    elif engine_choice == "2":
        selected_engine = "mediapipe"
    else:
        messagebox.showerror("Error", "Invalid selection. Using default engine (YOLO).")
        selected_engine = "yolo"
    
    # Initialize and run the analyzer with the selected engine
    analyzer = MovementAnalyzer(engine=selected_engine)
    analyzer.run()

if __name__ == "__main__":
    run_markerless_live() 