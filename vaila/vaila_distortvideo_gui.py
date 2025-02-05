"""
===============================================================================
vaila_lensdistortvideo.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 20 December 2024
Version: 0.1.0
Python Version: 3.12.8
===============================================================================

Este script processa vídeos aplicando a correção de distorção da lente com base
nos parâmetros intrínsecos da câmera e coeficientes de distorção. Agora, em
vez de carregar os parâmetros de um arquivo CSV, é possível ajustá-los interativamente
através de uma interface gráfica com sliders e botões. Para isso é extraído o primeiro
frame do vídeo e o resultado (imagem undistorted) é exibido em um preview atualizado em tempo
real.
===============================================================================
"""

import cv2
import numpy as np
import pandas as pd
import os
from rich import print
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.console import Console
from rich import print as rprint
import subprocess
from PIL import Image, ImageTk  # Para converter imagens para exibição com Tkinter
import math
import pygame

def load_distortion_parameters(csv_path):
    """
    Load distortion parameters from a CSV file.
    """
    df = pd.read_csv(csv_path)
    return df.iloc[0].to_dict()

def process_video(input_path, output_path, parameters):
    """Process video applying lens distortion correction."""
    console = Console()
    
    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create camera matrix and distortion coefficients
    camera_matrix = np.array([
        [parameters["fx"], 0, parameters["cx"]],
        [0, parameters["fy"], parameters["cy"]],
        [0, 0, 1]
    ])
    
    dist_coeffs = np.array([
        parameters["k1"],
        parameters["k2"],
        parameters["p1"],
        parameters["p2"],
        parameters["k3"]
    ])
    
    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    
    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            # Add task
            process_task = progress.add_task(
                "[cyan]Processing frames...", 
                total=total_frames
            )
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Undistort frame
                undistorted = cv2.undistort(
                    frame, 
                    camera_matrix, 
                    dist_coeffs, 
                    None, 
                    new_camera_matrix
                )
                
                # Save frame como PNG (lossless)
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_path, undistorted)
                
                frame_count += 1
                progress.update(process_task, advance=1)
                
                # Exibe informações adicionais a cada 100 frames
                if frame_count % 100 == 0:
                    elapsed = progress.tasks[0].elapsed
                    if elapsed:
                        fps_processing = frame_count / elapsed
                        remaining = (total_frames - frame_count) / fps_processing
                        progress.console.print(
                            f"[dim]Processing speed: {fps_processing:.1f} fps | "
                            f"Estimated time remaining: {remaining:.1f}s[/dim]"
                        )
        
        # Cria o vídeo final com FFmpeg
        rprint("\n[yellow]Creating final video with FFmpeg...[/yellow]")
        input_pattern = os.path.join(temp_dir, "frame_%06d.png")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)
        
    finally:
        # Libera a captura de vídeo
        cap.release()
        
        # Remove arquivos temporários
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
    
    rprint(f"\n[green]Video processing complete![/green]")
    rprint(f"[blue]Output saved as: {output_path}[/blue]")

def select_directory(title="Select a directory"):
    """
    Open a dialog to select a directory.
    """
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory

def select_file(title="Select a file", filetypes=(("CSV Files", "*.csv"),)):
    """
    Open a dialog to select a file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path

def distort_video_gui():
    """
    GUI para ajustar interativamente os parâmetros de distorção 
    usando o primeiro frame de um vídeo como exemplo.
    
    Após o ajuste, os parâmetros (fx,fy,cx,cy,k1,k2,k3,p1,p2) são salvos
    em um arquivo CSV para uso posterior.
    """
    # Cria a janela raiz (oculta)
    root = tk.Tk()
    root.withdraw()

    # Seleciona o vídeo e extrai o primeiro frame
    video_path = filedialog.askopenfilename(
        title="Selecione o vídeo para extrair o primeiro frame",
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*"))
    )
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Erro", "Não foi possível ler o frame do vídeo.")
        return None

    original_frame = frame.copy()
    orig_height, orig_width = original_frame.shape[:2]

    # Estima os parâmetros iniciais usando FOV de 90°
    fov = 90
    default_fx = int((orig_width / 2) / math.tan(math.radians(fov / 2)))
    default_fy = default_fx
    default_cx = orig_width // 2
    default_cy = orig_height // 2

    # Cria a janela de preview
    preview_win = tk.Toplevel(root)
    preview_win.title("Preview - Distortion Correction")
    init_width = min(800, orig_width)
    init_height = int(init_width * (orig_height / orig_width))
    preview_win.geometry(f"{init_width}x{init_height}")
    preview_label = tk.Label(preview_win)
    preview_label.pack(expand=True, fill="both")

    # Cria a janela de controles
    control_win = tk.Toplevel(root)
    control_win.title("Controles de Parâmetros")
    control_win.geometry("350x700")
    controls_frame = tk.Frame(control_win)
    controls_frame.pack(expand=True, fill="both", padx=5, pady=5)

    # Define as variáveis para os parâmetros
    fx_var = tk.DoubleVar(value=default_fx)
    fy_var = tk.DoubleVar(value=default_fy)
    cx_var = tk.DoubleVar(value=default_cx)
    cy_var = tk.DoubleVar(value=default_cy)
    k1_var = tk.DoubleVar(value=0.0)
    k2_var = tk.DoubleVar(value=0.0)
    k3_var = tk.DoubleVar(value=0.0)
    p1_var = tk.DoubleVar(value=0.0)
    p2_var = tk.DoubleVar(value=0.0)
    scale_var = tk.DoubleVar(value=1.0)

    def update_preview():
        # Obtém os parâmetros atuais
        fx = fx_var.get()
        fy = fy_var.get()
        cx = cx_var.get()
        cy = cy_var.get()
        k1 = k1_var.get()
        k2 = k2_var.get()
        k3 = k3_var.get()
        p1 = p1_var.get()
        p2 = p2_var.get()

        # Cria a matriz da câmera e os coeficientes de distorção
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        # Calcula a nova matriz da câmera (opcional, porém útil)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (orig_width, orig_height), 1, (orig_width, orig_height)
        )
        undistorted = cv2.undistort(original_frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Redimensiona a imagem para caber na janela de preview
        scale = scale_var.get()
        preview_win.update_idletasks()
        win_w = preview_win.winfo_width()
        win_h = preview_win.winfo_height()
        new_w = int(win_w * scale)
        new_h = int(win_h * scale)
        resized = cv2.resize(undistorted, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Converte para RGB e cria a imagem para Tkinter
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(resized_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        preview_label.configure(image=tk_image)
        preview_label.image = tk_image  # Mantém referência
        preview_win.after(100, update_preview)

    update_preview()

    # Função auxiliar para criar sliders (mantida, mas sem bindings de teclado)
    slider_row = 0
    def add_slider(label_text, var, from_val, to_val, resolution):
        nonlocal slider_row
        # Cria um frame para agrupar o rótulo, o slider e o campo entry
        frame = tk.Frame(controls_frame)
        frame.grid(row=slider_row, column=0, columnspan=2, sticky="we", padx=2, pady=2)
        
        # Rótulo do slider
        lbl = tk.Label(frame, text=label_text)
        lbl.pack(side="left")
        
        # Slider propriamente dito
        slider = tk.Scale(
            frame,
            variable=var,
            from_=from_val,
            to=to_val,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            length=150,
            takefocus=True
        )
        slider.pack(side="left", fill="x", expand=True)
        
        # Campo entry para digitação manual do valor
        entry = tk.Entry(frame, width=8)
        entry.pack(side="left", padx=5)
        entry.insert(0, str(var.get()))
        
        # Atualiza o campo entry quando o slider é movido
        def slider_changed(val):
            try:
                fval = float(val)
            except ValueError:
                fval = 0
            # Se a resolução for menor que 1, usa formatação float; senão, inteiro.
            if resolution < 1:
                entry_value = f"{fval:.3f}"
            else:
                entry_value = f"{int(round(fval))}"
            entry.delete(0, tk.END)
            entry.insert(0, entry_value)
        
        slider.config(command=slider_changed)
        
        # Atualiza o slider quando o usuário digita o valor manualmente
        def entry_changed(event):
            try:
                new_val = float(entry.get())
            except ValueError:
                new_val = slider.get()
            # Garante que o valor esteja dentro dos limites
            if new_val < from_val:
                new_val = from_val
            elif new_val > to_val:
                new_val = to_val
            slider.set(new_val)
        
        entry.bind("<Return>", entry_changed)
        entry.bind("<FocusOut>", entry_changed)
        
        # Configura o scroll do mouse para incrementar/decrementar exatamente 1 unidade de 'resolution'
        def on_mousewheel(event):
            r = float(slider.cget("resolution"))
            if event.delta:
                # Ignora o valor absoluto; usa somente o sinal
                step = 1 if event.delta > 0 else -1
                slider.set(slider.get() + step * r)
            elif hasattr(event, 'num'):
                if event.num == 4:
                    slider.set(slider.get() + r)
                elif event.num == 5:
                    slider.set(slider.get() - r)
            return "break"
        
        slider.bind("<MouseWheel>", on_mousewheel)
        slider.bind("<Button-4>", on_mousewheel)
        slider.bind("<Button-5>", on_mousewheel)
        
        slider_row += 1

    add_slider("fx", fx_var, default_fx * 0.5, default_fx * 1.5, 1)
    add_slider("fy", fy_var, default_fy * 0.5, default_fy * 1.5, 1)
    add_slider("cx", cx_var, 0, orig_width, 1)
    add_slider("cy", cy_var, 0, orig_height, 1)
    add_slider("k1", k1_var, -1.0, 1.0, 0.001)
    add_slider("k2", k2_var, -1.0, 1.0, 0.001)
    add_slider("k3", k3_var, -1.0, 1.0, 0.001)
    add_slider("p1", p1_var, -1.0, 1.0, 0.001)
    add_slider("p2", p2_var, -1.0, 1.0, 0.001)
    add_slider("Scale", scale_var, 0.5, 1.5, 0.001)

    # --- Vinculação global de eventos de teclado para sliders --- #
    def on_key_global(event):
        focused_widget = control_win.focus_get()
        if isinstance(focused_widget, tk.Scale):
            current_val = focused_widget.get()
            # Obtém a resolução configurada para o slider
            resolution = float(focused_widget.cget("resolution"))
            if event.keysym == 'Left':
                focused_widget.set(current_val - resolution)
            elif event.keysym == 'Right':
                focused_widget.set(current_val + resolution)

    control_win.bind_all("<KeyPress-Left>", on_key_global)
    control_win.bind_all("<KeyPress-Right>", on_key_global)
    # ----------------------------------------------------------------- #

    # Classe para manter o estado da confirmação
    class State:
        def __init__(self):
            self.confirmed = False
            self.results = {}

    state = State()

    def confirm():
        state.results = {
            "fx": fx_var.get(),
            "fy": fy_var.get(),
            "cx": cx_var.get(),
            "cy": cy_var.get(),
            "k1": k1_var.get(),
            "k2": k2_var.get(),
            "k3": k3_var.get(),
            "p1": p1_var.get(),
            "p2": p2_var.get()
        }
        state.confirmed = True
        preview_win.destroy()
        control_win.destroy()
        root.quit()

    def cancel():
        preview_win.destroy()
        control_win.destroy()
        root.quit()

    btn_confirm = tk.Button(controls_frame, text="Confirmar", command=confirm)
    btn_confirm.grid(row=slider_row, column=0, columnspan=2, pady=5)
    slider_row += 1
    btn_cancel = tk.Button(controls_frame, text="Cancelar", command=cancel)
    btn_cancel.grid(row=slider_row, column=0, columnspan=2, pady=5)

    root.mainloop()

    if state.confirmed:
        # Seleciona onde salvar o arquivo de parâmetros
        params_file = filedialog.asksaveasfilename(
            title="Salvar parâmetros",
            defaultextension=".csv",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if params_file:
            with open(params_file, "w") as f:
                f.write("fx,fy,cx,cy,k1,k2,k3,p1,p2\n")
                f.write(f"{state.results['fx']:.2f},"
                        f"{state.results['fy']:.2f},"
                        f"{state.results['cx']:.2f},"
                        f"{state.results['cy']:.3f},"
                        f"{state.results['k1']:.17f},"
                        f"{state.results['k2']:.17f},"
                        f"{state.results['k3']:.17f},"
                        f"{state.results['p1']:.17f},"
                        f"{state.results['p2']:.17f}\n")
        return state.results
    else:
        return None

def distort_video_gui_cv2():
    """
    Ajuste dos parâmetros de distorção utilizando a interface do OpenCV.
    
    O usuário seleciona um vídeo (para extrair o primeiro frame) e na janela com trackbars
    pode modificar os parâmetros de correção: [fx, fy, cx, cy, k1, k2, k3, p1, p2] e o fator de escala.
    
    Pressione 'c' para confirmar ou 'q' para cancelar.
    
    Returns:
        dict: Parâmetros confirmados, ou None se cancelados.
    """
    import math
    # Seleciona o vídeo para extração do frame
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Selecione o vídeo para extrair o primeiro frame",
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*"))
    )
    if not video_path:
        return None

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        messagebox.showerror("Erro", "Não foi possível ler o frame do vídeo.")
        return None
    original_frame = frame.copy()
    orig_height, orig_width = original_frame.shape[:2]

    # Estima os parâmetros iniciais considerando um FOV de 90°
    fov = 90
    default_fx = int((orig_width / 2) / math.tan(math.radians(fov / 2)))
    default_fy = default_fx
    default_cx = orig_width // 2
    default_cy = orig_height // 2

    # Cria uma janela para preview e ajuste dos parâmetros
    window_name = "Ajuste de Parâmetros (Pressione 'c' para confirmar, 'q' para cancelar)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Define os trackbars com intervalos e valores padrão
    trackbars = {
        "fx":   {"min": int(default_fx * 0.5), "max": int(default_fx * 1.5), "default": default_fx},
        "fy":   {"min": int(default_fy * 0.5), "max": int(default_fy * 1.5), "default": default_fy},
        "cx":   {"min": 0, "max": orig_width, "default": default_cx},
        "cy":   {"min": 0, "max": orig_height, "default": default_cy},
        "k1":   {"min": -1000, "max": 1000, "default": 0},
        "k2":   {"min": -1000, "max": 1000, "default": 0},
        "k3":   {"min": -1000, "max": 1000, "default": 0},
        "p1":   {"min": -1000, "max": 1000, "default": 0},
        "p2":   {"min": -1000, "max": 1000, "default": 0},
        # Fator de escala para visualização (0.5 a 1.5) multiplicado por 100 para ter resolução de 0.01
        "scale": {"min": 50, "max": 150, "default": 100},
    }

    # Função "dummy" para callback das trackbars
    def nothing(x):
        pass

    # Cria os trackbars na janela
    for name, params in trackbars.items():
        cv2.createTrackbar(name, window_name,
                           params["default"] - params["min"],
                           params["max"] - params["min"],
                           nothing)

    # Função auxiliar para retornar o valor real do trackbar (com offset)
    def get_trackbar_value(trackbar_name):
        params = trackbars[trackbar_name]
        pos = cv2.getTrackbarPos(trackbar_name, window_name)
        return params["min"] + pos

    while True:
        # Verifica se a janela ainda está visível; se não, encerra a função
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return None

        # Obtém os valores atuais dos trackbars
        fx = float(get_trackbar_value("fx"))
        fy = float(get_trackbar_value("fy"))
        cx = float(get_trackbar_value("cx"))
        cy = float(get_trackbar_value("cy"))
        k1 = get_trackbar_value("k1") / 1000.0
        k2 = get_trackbar_value("k2") / 1000.0
        k3 = get_trackbar_value("k3") / 1000.0
        p1 = get_trackbar_value("p1") / 1000.0
        p2 = get_trackbar_value("p2") / 1000.0
        scale = get_trackbar_value("scale") / 100.0

        # Monta as matrizes de câmera e distorção
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0,  0,  1]], dtype=np.float32)
        dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                               dist_coeffs,
                                                               (orig_width, orig_height),
                                                               1,
                                                               (orig_width, orig_height))
        undistorted = cv2.undistort(original_frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # Redimensiona a imagem para visualização de acordo com o fator "scale"
        new_w = int(orig_width * scale)
        new_h = int(orig_height * scale)
        preview = cv2.resize(undistorted, (new_w, new_h))

        cv2.imshow(window_name, preview)
        key = cv2.waitKey(50) & 0xFF
        if key == ord('c'):
            cv2.destroyAllWindows()
            return {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "p1": p1,
                "p2": p2
            }
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return None

def run_distortvideo_gui():
    """Main function to run lens distortion correction using a single video and an OpenCV-based GUI."""
    rprint("[yellow]Running lens distortion correction with OpenCV GUI...[/yellow]")
    
    # Extrai os parâmetros via interface OpenCV (versão anterior que funcionava)
    parameters = distort_video_gui_cv2()
    if parameters is None:
        rprint("[red]A extração dos parâmetros foi cancelada.[/red]")
        return

    # Seleciona o vídeo a ser processado (pode ser o mesmo utilizado para ajuste)
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Selecione o vídeo para processar",
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*"))
    )
    if not video_path:
        rprint("[red]Nenhum vídeo foi selecionado para processamento.[/red]")
        return

    # Salva os parâmetros em um arquivo CSV no mesmo diretório do vídeo selecionado
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    params_csv = os.path.join(os.path.dirname(video_path), f"{base_name}_parameters.csv")
    try:
        with open(params_csv, "w") as f:
            f.write("fx,fy,cx,cy,k1,k2,k3,p1,p2\n")
            f.write(f"{parameters['fx']:.2f},"
                    f"{parameters['fy']:.2f},"
                    f"{parameters['cx']:.2f},"
                    f"{parameters['cy']:.2f},"
                    f"{parameters['k1']:.17f},"
                    f"{parameters['k2']:.17f},"
                    f"{parameters['k3']:.17f},"
                    f"{parameters['p1']:.17f},"
                    f"{parameters['p2']:.17f}\n")
        rprint(f"\n[blue]Parâmetros salvos em: {params_csv}[/blue]")
    except Exception as e:
        rprint(f"[red]Erro ao salvar parâmetros: {e}[/red]")

    # Define o caminho de saída do vídeo processado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(os.path.dirname(video_path), f"{base_name}_undistorted_{timestamp}.mp4")
    
    try:
        rprint(f"\n[cyan]Processando o vídeo: {video_path}[/cyan]")
        process_video(video_path, output_path, parameters)
    except Exception as e:
        rprint(f"[red]Erro no processamento do vídeo: {e}[/red]")
    
    rprint("\n[green]Processamento completo![/green]")
    rprint(f"[blue]Vídeo de saída salvo em: {output_path}[/blue]")

if __name__ == "__main__":
    run_distortvideo_gui()