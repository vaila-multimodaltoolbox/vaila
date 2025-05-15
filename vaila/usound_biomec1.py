"""
usound_biomec1.py

Module to analyze ultrasound data from images.

Created by: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br 
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 14 May 2025
Updated: 15 May 2025
Version: 0.0.1

Description:
    This module provides a standalone script for ultrasound data analysis from images.
    It can be imported into usound_biomec1.py or run directly.

Usage:
    This module can be imported into usound_biomec1.py or run directly.
    Directly: python usound_biomec1.py
    Import: from vaila import usound_biomec1
    usound_biomec1.run_usound()   

Dependencies:
    - opencv-python
    - numpy
    - pandas
    - math
    - sys
    - pathlib
    - tkinter

License:
    This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.
"""

import cv2
import pandas as pd
import numpy as np
import math
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os

# Função para listar todos os arquivos de imagem no diretório especificado
def listar_imagens(diretorio):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    arquivos = []
    for ext in exts:
        arquivos.extend(Path(diretorio).glob(ext))
    return [str(arq) for arq in sorted(arquivos)]

# Função de pré-processamento da imagem
def preprocessar_imagem(imagem_bgr):
    imagem_gray = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imagem_gray = clahe.apply(imagem_gray)
    imagem_gray = cv2.bilateralFilter(imagem_gray, d=5, sigmaColor=75, sigmaSpace=75)
    bordas = cv2.Canny(imagem_gray, threshold1=50, threshold2=150)
    imagem_processada = cv2.cvtColor(imagem_gray, cv2.COLOR_GRAY2BGR)
    mask = bordas != 0
    imagem_processada[mask] = [0, 0, 255]
    return imagem_processada

# Função para redesenhar a imagem com anotações
def redesenhar_imagem(state):
    img = state["base_img"].copy()
    if len(state["calib_points"]) == 1 and not state["calibrated"]:
        cv2.circle(img, state["calib_points"][0], 5, (255, 255, 0), -1)
    if state["calibrated"] and len(state["calib_points"]) == 2:
        pt1, pt2 = state["calib_points"]
        cv2.circle(img, pt1, 5, (255, 255, 0), -1)
        cv2.circle(img, pt2, 5, (255, 255, 0), -1)
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
        dist_cm = state["scale"] * state["calib_dist_px"]
        mx, my = (pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2
        cv2.putText(img, f"{dist_cm:.2f} cm", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
    for p1, p2, dist_px, dist_cm in state["measurements"]:
        cv2.circle(img, p1, 5, (0, 255, 0), -1)
        cv2.circle(img, p2, 5, (0, 255, 0), -1)
        cv2.line(img, p1, p2, (0, 255, 0), 2)
        mx, my = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        cv2.putText(img, f"{dist_cm:.2f} cm", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(img, f"{dist_cm:.2f} cm", (mx+5, my-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if state["calibrated"] and state["first_point"] is not None:
        cv2.circle(img, state["first_point"], 5, (0, 165, 255), -1)
    # Aplicar zoom se necessário
    if state.get("zoom_factor", 1.0) != 1.0:
        h, w = img.shape[:2]
        # Dimensões da imagem ampliada mantendo proporção
        new_w = int(w * state["zoom_factor"])
        new_h = int(h * state["zoom_factor"])
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if state["zoom_factor"] < 1 else cv2.INTER_LINEAR)
    return img

# Callback de mouse para eventos de clique
def evento_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        if not state["calibrated"]:
            if len(state["calib_points"]) < 2:
                state["calib_points"].append((x, y))
                if len(state["calib_points"]) == 2:
                    pt1, pt2 = state["calib_points"]
                    dx, dy = pt2[0]-pt1[0], pt2[1]-pt1[1]
                    dist_px = math.hypot(dx, dy)
                    state["calib_dist_px"] = dist_px
                    val = input(f"Distância real (cm) entre calibragem em {state['img_name']}: ")
                    try:
                        real_val = float(val.replace(',', '.'))
                    except:
                        real_val = 1.0
                    state["scale"] = real_val / dist_px
                    state["calibrated"] = True
            state["display_img"] = redesenhar_imagem(state)
        else:
            if state["first_point"] is None:
                state["first_point"] = (x, y)
            else:
                p1 = state["first_point"]
                p2 = (x, y)
                dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                dist_px = math.hypot(dx, dy)
                dist_cm = dist_px * state["scale"]
                state["measurements"].append((p1, p2, dist_px, dist_cm))
                state["first_point"] = None
            state["display_img"] = redesenhar_imagem(state)

# Processa todas as imagens no diretório dado e salva resultados
def processar_imagens(input_dir, output_csv, scale=None):
    arquivos = listar_imagens(input_dir)
    if not arquivos:
        print("Nenhuma imagem encontrada.")
        return
    resultados = []
    cv2.namedWindow("Imagem", cv2.WINDOW_NORMAL)
    for caminho in arquivos:
        img = cv2.imread(caminho)
        if img is None:
            print(f"Erro ao carregar {caminho}")
            continue
        img_proc = preprocessar_imagem(img)
        state = {"img_name": Path(caminho).name,
                 "base_img": img_proc,
                 "display_img": img_proc.copy(),
                 "calib_points": [],
                 "calibrated": scale is not None,  # Já calibrado se scale foi fornecido
                 "calib_dist_px": None,
                 "scale": scale,  # Usar escala fornecida se disponível
                 "first_point": None,
                 "measurements": [],
                 "zoom_factor": 1.0}  # Fator de zoom inicial
        cv2.setMouseCallback("Imagem", evento_mouse, state)
        print(f"Processando: {state['img_name']}")
        
        if scale is None:
            print("Clique 2 pontos para calibração, informe valor cm. Depois pares para medir.")
        else:
            print(f"Calibração: {1/scale:.2f} pixels/cm. Clique pares de pontos para medir.")
            
        print("Teclas: u=undo, r=reset, n=próxima, q=quit, +/- para zoom.")
        while True:
            cv2.imshow("Imagem", state["display_img"])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                cv2.destroyWindow("Imagem")
                df = pd.DataFrame(resultados, columns=["Arquivo","Ponto 1","Ponto 2","Distância em Pixels","Distância Real (cm)"])
                df.to_csv(output_csv, index=False)
                print(f"Resultados salvos em {output_csv}")
                return
            if key == ord('n'):
                break
            if key == ord('u'):
                if state["first_point"] is not None:
                    state["first_point"] = None
                elif not state["calibrated"] and len(state["calib_points"]) == 1:
                    state["calib_points"].clear()
                elif state["measurements"]:
                    state["measurements"].pop()
                state["display_img"] = redesenhar_imagem(state)
            if key == ord('r'):
                state.update({"calib_points": [], "calibrated": scale is not None,
                              "calib_dist_px": None, "scale": scale,
                              "first_point": None, "measurements": []})
                state["display_img"] = redesenhar_imagem(state)
            # Controles de zoom
            if key == ord('+') or key == ord('='):  # '=' e '+' geralmente são a mesma tecla
                state["zoom_factor"] = min(5.0, state["zoom_factor"] * 1.1)  # Limitar zoom máximo
                state["display_img"] = redesenhar_imagem(state)
                print(f"Zoom: {state['zoom_factor']:.1f}x")
            if key == ord('-'):
                state["zoom_factor"] = max(0.1, state["zoom_factor"] / 1.1)  # Limitar zoom mínimo
                state["display_img"] = redesenhar_imagem(state)
                print(f"Zoom: {state['zoom_factor']:.1f}x")
        for p1, p2, dist_px, dist_cm in state["measurements"]:
            resultados.append([state["img_name"],
                               f"({p1[0]}, {p1[1]})",
                               f"({p2[0]}, {p2[1]})",
                               round(dist_px,2), round(dist_cm,2)])
    cv2.destroyWindow("Imagem")
    if resultados:
        df = pd.DataFrame(resultados, columns=["Arquivo","Ponto 1","Ponto 2","Distância em Pixels","Distância Real (cm)"])
        df.to_csv(output_csv, index=False)
        print(f"CSV gerado: {output_csv}")
    else:
        print("Nenhuma medição realizada.")

def crop_images_batch(input_dir, output_dir):
    arquivos = listar_imagens(input_dir)
    if not arquivos:
        print("Nenhuma imagem encontrada para crop.")
        return None, None

    primeira_img_path = arquivos[0]
    img = cv2.imread(primeira_img_path)
    if img is None:
        print(f"Erro ao carregar {primeira_img_path}")
        return None, None

    # Variáveis para armazenar os pontos do crop
    crop_points = []

    def mouse_crop(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(crop_points) < 2:
            crop_points.append((x, y))
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            if len(crop_points) == 1:
                cv2.imshow("Selecione os pontos de crop", img_display)
                print("Clique no canto inferior direito para definir o crop.")
            elif len(crop_points) == 2:
                # Desenha o retângulo do crop
                cv2.rectangle(img_display, crop_points[0], crop_points[1], (0, 255, 0), 2)
                cv2.imshow("Selecione os pontos de crop", img_display)

    img_display = img.copy()
    cv2.imshow("Selecione os pontos de crop", img_display)
    cv2.setMouseCallback("Selecione os pontos de crop", mouse_crop)

    print("Clique no canto superior esquerdo para definir o crop.")
    while len(crop_points) < 2:
        key = cv2.waitKey(1)
        if key == 27:  # ESC para cancelar
            cv2.destroyWindow("Selecione os pontos de crop")
            return None, None
    cv2.destroyWindow("Selecione os pontos de crop")

    (x1, y1), (x2, y2) = crop_points
    # Garantir que (x1,y1) seja o canto superior esquerdo e (x2,y2) o canto inferior direito
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Caixa de diálogo para valor em cm (altura vertical)
    root = tk.Tk()
    root.withdraw()
    altura_pixels = y_max - y_min
    calib_cm = tk.simpledialog.askfloat(
        "Calibração",
        f"Informe o valor em cm correspondente à altura do crop ({altura_pixels} px):"
    )
    if not calib_cm:
        print("Calibração não informada. Encerrando.")
        return None, None
    
    # Calcular escala (cm/pixel)
    scale = calib_cm / altura_pixels

    # Cria diretório de saída para crops
    cropped_dir = os.path.join(output_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    # Diretório para imagens side by side e sobreposição
    sidebyside_dir = os.path.join(output_dir, "side_by_side")
    os.makedirs(sidebyside_dir, exist_ok=True)
    superposicao_dir = os.path.join(output_dir, "superposicao")
    os.makedirs(superposicao_dir, exist_ok=True)

    # Lista para armazenar caminhos das imagens cortadas
    cropped_paths = []

    for arq in arquivos:
        img = cv2.imread(arq)
        if img is None:
            print(f"Erro ao carregar {arq}")
            continue
        cropped = img[y_min:y_max, x_min:x_max]
        out_path = os.path.join(cropped_dir, Path(arq).name)
        cv2.imwrite(out_path, cropped)
        cropped_paths.append(out_path)
        
    print(f"Imagens cortadas salvas em: {cropped_dir}")
    
    # Criar imagens side by side e sobreposição
    if len(cropped_paths) >= 2:
        for i in range(len(cropped_paths) - 1):
            for j in range(i + 1, len(cropped_paths)):
                # Nome dos arquivos para referência
                nome_i = Path(cropped_paths[i]).stem
                nome_j = Path(cropped_paths[j]).stem
                
                # Side by side
                side_path = os.path.join(sidebyside_dir, f"{nome_i}_vs_{nome_j}.jpg")
                imagens_side_by_side(cropped_paths[i], cropped_paths[j], side_path)
                
                # Sobreposição
                super_path = os.path.join(superposicao_dir, f"{nome_i}_over_{nome_j}.jpg")
                imagens_superposicao(cropped_paths[i], cropped_paths[j], super_path)
                
        print(f"Imagens side by side salvas em: {sidebyside_dir}")
        print(f"Imagens com sobreposição salvas em: {superposicao_dir}")
    
    return cropped_dir, scale

def run_usound():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal

    input_dir = filedialog.askdirectory(title="Selecione o diretório de imagens para análise")
    if not input_dir:
        print("Nenhum diretório selecionado. Encerrando.")
        return

    output_dir = filedialog.askdirectory(title="Selecione o diretório de saída para imagens cortadas e CSV")
    if not output_dir:
        print("Nenhum diretório de saída selecionado. Encerrando.")
        return

    # 1. Crop em batch com dois pontos e calibração vertical
    cropped_dir, scale = crop_images_batch(input_dir, output_dir)
    if not cropped_dir:
        print("Erro no crop. Encerrando.")
        return

    # 2. Selecionar arquivo de saída CSV
    output_csv = filedialog.asksaveasfilename(
        title="Selecione o arquivo de saída CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialdir=output_dir
    )
    if not output_csv:
        print("Nenhum arquivo de saída selecionado. Encerrando.")
        return

    # 3. Processar imagens já cortadas usando a escala definida
    processar_imagens(cropped_dir, output_csv, scale)

def imagens_side_by_side(img_path1, img_path2, output_path):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    # Redimensiona para mesma altura
    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    side = np.hstack((img1, img2))
    cv2.imwrite(output_path, side)

def imagens_superposicao(img_path1, img_path2, output_path, alpha=0.5):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    # Redimensiona para mesmo tamanho
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))
    superposta = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
    cv2.imwrite(output_path, superposta)

# Entrada principal
if __name__ == "__main__":
    run_usound()
