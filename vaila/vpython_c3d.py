from vpython import canvas, sphere, vector, rate, scene
import ezc3d
import numpy as np
import tkinter as tk
from tkinter import filedialog

def load_c3d_file():
    """
    Abre um diálogo para selecionar um arquivo C3D e carrega os dados dos marcadores.
    Retorna:
        pts: np.ndarray com shape (num_frames, num_markers, 3) – pontos convertidos para metros.
        filepath: caminho do arquivo selecionado.
    """
    # Cria uma janela de seleção de arquivo
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Selecione um arquivo C3D",
                                          filetypes=[("Arquivos C3D", "*.c3d")])
    root.destroy()
    if not filepath:
        print("Nenhum arquivo foi selecionado. Encerrando.")
        exit(0)
    
    # Carrega o arquivo C3D
    c3d = ezc3d.c3d(filepath)
    pts = c3d["data"]["points"]   # shape: (4, num_markers, num_frames)
    pts = pts[:3, :, :]            # pega somente x, y, z
    pts = np.transpose(pts, (2, 1, 0))  # reorganiza para (num_frames, num_markers, 3)
    pts = pts * 0.001              # converte de mm para m
    return pts, filepath

def create_spheres(points_frame, radius=0.01):
    """
    Cria esferas no VPython para cada marcador do frame atual.
    Args:
        points_frame: np.ndarray com shape (num_markers, 3)
        radius: raio das esferas (em metros)
    Retorna:
        lista de objetos sphere
    """
    spheres_list = []
    for pt in points_frame:
        s = sphere(pos=vector(pt[0], pt[1], pt[2]), radius=radius, color=vector(1, 0, 0))
        spheres_list.append(s)
    return spheres_list

def update_spheres(spheres_list, points_frame):
    """
    Atualiza a posição das esferas de acordo com os dados do novo frame.
    """
    for s, pt in zip(spheres_list, points_frame):
        s.pos = vector(pt[0], pt[1], pt[2])

def main():
    # Carrega os dados do arquivo C3D
    points, filepath = load_c3d_file()
    num_frames, num_markers, _ = points.shape
    print(f"Arquivo carregado: {filepath}")
    print(f"Número de frames: {num_frames}, Número de marcadores: {num_markers}")

    # Cria a cena do VPython
    scene.title = f"Visualizador C3D com VPython - Frame 1/{num_frames}"
    scene.width = 800
    scene.height = 600
    scene.background = vector(0.8, 0.8, 0.8)
    
    # Cria as esferas para o primeiro frame
    spheres_list = create_spheres(points[0])
    current_frame = 0

    def keydown(evt):
        nonlocal current_frame, spheres_list
        key = evt.key
        if key == "n":   # Próximo frame
            current_frame = (current_frame + 1) % num_frames
        elif key == "p": # Frame anterior
            current_frame = (current_frame - 1) % num_frames
        else:
            return  # Ignora outras teclas
        
        update_spheres(spheres_list, points[current_frame])
        scene.title = f"Visualizador C3D com VPython - Frame {current_frame+1}/{num_frames}"
        print(f"Frame: {current_frame+1}/{num_frames}")

    # Associa a função de evento de tecla à cena
    scene.bind('keydown', keydown)
    
    # Loop principal para manter a janela atualizada
    while True:
        rate(60)  # 60 atualizações por segundo

if __name__ == "__main__":
    main()
