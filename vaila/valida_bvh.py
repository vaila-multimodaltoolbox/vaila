#!/usr/bin/env python3
"""
================================================================================
Script: valida_bvh.py
Descrição: Validador de integridade para arquivos BVH gerados pelo vailá Toolbox.
           Verifica a coerência entre a hierarquia declarada e os dados de motion.
           Suporta modo CLI (argumentos) e GUI (diálogo de arquivo).
================================================================================
"""

import argparse
import os
from tkinter import Tk, filedialog, messagebox


def validate_vaila_bvh(filepath, gui=False):
    """
    Valida um arquivo BVH e exibe o relatório.

    Args:
        filepath (str): Caminho para o arquivo BVH.
        gui (bool): Se True, exibe resultados em janelas pop-up.
    """
    report = []

    def log(msg):
        print(msg)
        report.append(msg)

    log(f"{'=' * 60}")
    log(f"Validando Arquivo BVH: {os.path.basename(filepath)}")
    log(f"{'=' * 60}")

    if not os.path.exists(filepath):
        msg = f"[FAIL] Erro: Arquivo '{filepath}' não encontrado."
        log(msg)
        if gui:
            messagebox.showerror("Erro", msg)
        return False

    try:
        with open(filepath, encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
    except Exception as e:
        msg = f"[FAIL] Erro ao ler o arquivo: {e}"
        log(msg)
        if gui:
            messagebox.showerror("Erro", msg)
        return False

    roots = []
    frames_declared = 0
    frame_time = 0.0
    data_rows = 0
    expected_channels_per_row = 0

    section = None
    primeira_linha_dados_ok = True

    for i, line in enumerate(lines):
        if not line:
            continue

        if line == "HIERARCHY":
            section = "HIERARCHY"
            continue
        elif line == "MOTION":
            section = "MOTION"
            continue

        if section == "HIERARCHY":
            if line.startswith("ROOT"):
                marker_name = line.split()[1]
                roots.append(marker_name)
                expected_channels_per_row += 3  # X, Y, Z para cada ROOT independente

        elif section == "MOTION":
            if line.startswith("Frames:"):
                try:
                    frames_declared = int(line.split()[1])
                except ValueError:
                    log("[FAIL] Erro: Formato de 'Frames:' inválido.")
            elif line.startswith("Frame Time:"):
                try:
                    frame_time = float(line.split()[2])
                except ValueError:
                    log("[FAIL] Erro: Formato de 'Frame Time:' inválido.")
            else:
                # Se não é o cabeçalho de MOTION, é uma linha de matriz de dados
                data_rows += 1

                # Checagem de sanidade apenas na primeira linha de dados para otimização
                if data_rows == 1:
                    channels_in_row = len(line.split())
                    if channels_in_row != expected_channels_per_row:
                        log(
                            f"[WARNING] AVISO: A linha de dados 1 tem {channels_in_row} canais, "
                            f"mas a hierarquia define {expected_channels_per_row} canais."
                        )
                        primeira_linha_dados_ok = False

    # --- Geração do Relatório ---
    log("\n[ Resumo da Hierarquia ]")
    log(f"Total de Marcadores (ROOTs): {len(roots)}")
    if roots:
        display_roots = (
            roots if len(roots) <= 15 else roots[:15] + [f"... (+ {len(roots) - 15} outros)"]
        )
        log(f"Marcadores: {', '.join(display_roots)}")

    log("\n[ Resumo do Movimento (Cinemática) ]")
    log(f"Frames declarados no cabeçalho: {frames_declared}")
    log(f"Linhas de dados processadas:  {data_rows}")

    freq_hz = (1.0 / frame_time) if frame_time > 0 else 0
    log(f"Frame Time: {frame_time:.6f}s (Aprox. {freq_hz:.1f} Hz)")

    log("\n[ Diagnóstico Final ]")
    erros = 0

    if frames_declared != data_rows:
        log(f"[FAIL] ERRO: Descompasso de frames. Cabeçalho = {frames_declared}, Dados = {data_rows}.")
        erros += 1
    if not primeira_linha_dados_ok:
        log(
            "[FAIL] ERRO: O número de coordenadas na matriz de dados não bate com os marcadores definidos."
        )
        erros += 1
    if frames_declared == 0 or data_rows == 0:
        log("[FAIL] ERRO: Arquivo sem dados de movimento válidos.")
        erros += 1

    if erros == 0:
        log("[OK] SUCESSO: A estrutura do BVH está íntegra e pronta para o Blender!")
        if gui:
            # Mostra apenas as últimas linhas relevantes no popup para não ficar gigante
            summary = "\n".join(report[-6:])
            messagebox.showinfo("Sucesso", f"O arquivo BVH é válido!\n\n{summary}")
        return True
    else:
        log(f"{'=' * 60}\n")
        if gui:
            messagebox.showerror("Erros Encontrados", "\n".join(report))
        return False


def main():
    parser = argparse.ArgumentParser(description="Validador de arquivos BVH do vailá Toolbox.")
    parser.add_argument("file", nargs="?", help="Caminho para o arquivo .bvh a ser validado")
    parser.add_argument(
        "--gui", action="store_true", help="Forçar modo GUI (janelas) mesmo se arquivo for passado"
    )

    args = parser.parse_args()

    # Lógica de decisão GUI vs CLI
    # 1. Se nenhum arquivo foi passado, assume GUI para pedir o arquivo.
    # 2. Se --gui foi passado explicitamente, usa GUI.
    use_gui = args.gui or (not args.file)

    if use_gui:
        root = Tk()
        root.withdraw()  # Esconde a janela principal do Tkinter

        filepath = args.file
        if not filepath:
            filepath = filedialog.askopenfilename(
                title="Selecione o arquivo BVH para validar",
                filetypes=[("Biovision Hierarchy", "*.bvh"), ("All files", "*.*")],
            )

        if not filepath:
            print("Nenhum arquivo selecionado. Encerrando.")
            root.destroy()
            return

        validate_vaila_bvh(filepath, gui=True)
        root.destroy()
    else:
        # Modo CLI puro
        validate_vaila_bvh(args.file, gui=False)


if __name__ == "__main__":
    main()
