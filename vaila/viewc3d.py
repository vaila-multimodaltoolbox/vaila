"""
viewc3d.py

Descrição:
-----------
Novo visualizador 3D para arquivos C3D, rápido e eficiente, que respeita as dimensões 
dos dados (convertendo de milímetros para metros). O usuário pode navegar pelos frames 
usando as teclas 'N' (próximo frame) e 'P' (frame anterior), além de pular 10 frames 
com 'F' e 'B' e reproduzir automaticamente com a barra de espaço.

Referências:
- Mokka: https://github.com/Biomechanical-ToolKit/Mokka
- BTKPython: https://github.com/Biomechanical-ToolKit/BTKPython

Requisitos:
-----------
- open3d (pip install open3d)
- ezc3d (pip install ezc3d)
- numpy
- tkinter (para seleção do arquivo)
"""

import open3d as o3d
import ezc3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time

def load_c3d_file():
    """
    Abre um diálogo para selecionar um arquivo C3D e carrega os dados dos marcadores.
    
    Retorna:
        pts: np.ndarray com shape (num_frames, num_markers, 3) – os pontos convertidos (em metros)
        filepath: caminho do arquivo selecionado.
    """
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(title="Selecione um arquivo C3D",
                                          filetypes=[("Arquivos C3D", "*.c3d")])
    root.destroy()
    if not filepath:
        print("Nenhum arquivo foi selecionado. Encerrando.")
        exit(0)
    
    c3d = ezc3d.c3d(filepath)
    pts = c3d["data"]["points"]
    pts = pts[:3, :, :]   # pega somente x, y, z
    pts = np.transpose(pts, (2, 1, 0))  # (num_frames, num_markers, 3)
    pts = pts * 0.001  # Converte de milímetros para metros
    return pts, filepath

def create_coordinate_lines(axis_length=0.5):
    """
    Cria linhas representando os eixos cartesianos:
      - Eixo X em vermelho
      - Eixo Y em verde
      - Eixo Z em azul
    """
    points = np.array([
        [0, 0, 0],                   # Origem
        [axis_length, 0, 0],         # Eixo X
        [0, axis_length, 0],         # Eixo Y
        [0, 0, axis_length]          # Eixo Z
    ])
    lines = np.array([
        [0, 1],  # Linha para o eixo X
        [0, 2],  # Linha para o eixo Y
        [0, 3]   # Linha para o eixo Z
    ])
    colors = np.array([
        [1, 0, 0],    # Vermelho para X
        [0, 1, 0],    # Verde para Y
        [0, 0, 1]     # Azul para Z
    ])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_ground_plane(width=5.0, height=5.0):
    """
    Cria um plano no eixo XY (chão) com dimensões width x height e cor cinza escuro.
    O plano é definido na altura z = 0.
    """
    half_w = width / 2.0
    half_h = height / 2.0
    vertices = [
        [-half_w, -half_h, 0],
        [ half_w, -half_h, 0],
        [ half_w,  half_h, 0],
        [-half_w,  half_h, 0]
    ]
    triangles = [
        [0, 1, 2],
        [0, 2, 3]
    ]
    ground = o3d.geometry.TriangleMesh()
    ground.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    ground.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    # Pintamos o ground de cinza escuro ([0.2, 0.2, 0.2])
    ground.paint_uniform_color([0.2, 0.2, 0.2])
    ground.compute_vertex_normals()
    return ground

def create_x_marker(position, size=0.2):
    """
    Cria um marcador em forma de "X" no plano XY para indicar os limites.
    
    Args:
        position (np.ndarray): Coordenada (x, y, z) onde o "X" será colocado.
        size (float): Comprimento das linhas que formam o "X".
    
    Retorna:
        Um LineSet representando o "X".
    """
    half = size / 2.0
    x, y, z = position
    points = np.array([
        [x - half, y - half, z],
        [x + half, y + half, z],
        [x - half, y + half, z],
        [x + half, y - half, z]
    ])
    lines = np.array([
        [0, 1],
        [2, 3]
    ])
    x_marker = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    x_marker.paint_uniform_color([1, 0, 0])  # Vermelho para destaque
    return x_marker

def main():
    # Carrega os dados do arquivo C3D
    points, filepath = load_c3d_file()
    num_frames, num_markers, _ = points.shape

    # Diminuímos um pouco o tamanho dos markers e os pintamos de azul
    marker_radius = 0.015  # Tamanho reduzido em comparação ao valor anterior de 0.02
    spheres = []
    spheres_bases = []
    for i in range(num_markers):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius, resolution=8)
        base_vertices = np.asarray(sphere.vertices).copy()  # Base centrada na origem
        initial_pos = points[0][i]
        sphere.vertices = o3d.utility.Vector3dVector(base_vertices + initial_pos)
        sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Pintado de azul
        spheres.append(sphere)
        spheres_bases.append(base_vertices)

    # Cria o visualizador
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Visualizador C3D - Frame 1/{num_frames}")

    # Adiciona as esferas (markers) à cena
    for sphere in spheres:
        vis.add_geometry(sphere)
    
    # Adiciona os eixos cartesianos X, Y, Z como antes
    axes = create_coordinate_lines(axis_length=0.5)
    vis.add_geometry(axes)
    
    # --- Atualiza o ground para novos limites ---
    ground = create_ground_plane(width=6.0, height=7.0)
    # Translada o ground para que os cantos fiquem em (-1,-1), (5,-1), (5,6) e (-1,6)
    ground.translate(np.array([2.0, 2.5, 0.0]))
    vis.add_geometry(ground)

    # --- Atualiza os parâmetros da câmera para o novo ground ---
    ctr = vis.get_view_control()
    # O centro do ground agora é ((-1+5)/2, (-1+6)/2) = (2, 2.5, 0)
    bbox_center = np.array([2.0, 2.5, 0.0])
    ctr.set_lookat(bbox_center)
    ctr.set_front(np.array([-1, 0, 0]))  # Vista lateral: câmera posicionada a partir do lado positivo de X
    ctr.set_up(np.array([0, 0, 1]))
    ctr.set_zoom(0.8)  # Ajuste conforme necessário

    # Adiciona os marcadores "X" nos 4 cantos do ground
    corners = [np.array([-1, -1, 0]),
               np.array([5, -1, 0]),
               np.array([5, 6, 0]),
               np.array([-1, 6, 0])]
    for corner in corners:
        x_marker = create_x_marker(corner, size=0.2)
        vis.add_geometry(x_marker)

    # Configura as opções de renderização
    render_option = vis.get_render_option()
    render_option.point_size = 5.0
    render_option.background_color = np.array([0.8, 0.8, 0.8])
    # Desabilita a iluminação para remover o reflexo de ponto de luz
    render_option.light_on = False

    # Função para atualizar o título do console com o número do frame atual
    def update_window_title():
        new_title = f"Visualizador C3D - Frame {current_frame+1}/{num_frames}"
        print(new_title)

    # Callback para imprimir os parâmetros atuais do viewpoint (para feedback)
    def show_camera_params(vis_obj):
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        print("Extrinsics (viewpoint atual):")
        print(cam_params.extrinsic)
        return False

    vis.register_key_callback(ord("O"), show_camera_params)

    # Função auxiliar para atualizar as posições das esferas (markers) no frame atual
    def update_spheres(frame_data):
        for i, sphere in enumerate(spheres):
            new_pos = frame_data[i]
            new_vertices = spheres_bases[i] + new_pos
            sphere.vertices = o3d.utility.Vector3dVector(new_vertices)
            vis.update_geometry(sphere)
        update_window_title()  # Atualiza o título com o frame atual
        vis.poll_events()
        vis.update_renderer()

    # Variáveis de controle dos frames
    current_frame = 0
    is_playing = False

    def next_frame(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame + 1) % num_frames
        update_spheres(points[current_frame])
        return False

    def previous_frame(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame - 1) % num_frames
        update_spheres(points[current_frame])
        return False

    def forward_10_frames(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame + 10) % num_frames
        update_spheres(points[current_frame])
        return False

    def backward_10_frames(vis_obj):
        nonlocal current_frame
        current_frame = (current_frame - 10) % num_frames
        update_spheres(points[current_frame])
        return False

    def toggle_play(vis_obj):
        nonlocal is_playing
        is_playing = not is_playing
        return False

    # Registra os callbacks para as teclas
    vis.register_key_callback(ord("N"), next_frame)
    vis.register_key_callback(ord("P"), previous_frame)
    vis.register_key_callback(ord("F"), forward_10_frames)
    vis.register_key_callback(ord("B"), backward_10_frames)
    vis.register_key_callback(ord(" "), toggle_play)

    # Loop principal para reprodução automática
    while True:
        if not vis.poll_events():
            break
        if is_playing:
            next_frame(vis)
            time.sleep(0.03)
    vis.destroy_window()

if __name__ == "__main__":
    main() 