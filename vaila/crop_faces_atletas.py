"""
===============================================================================
crop_faces_atletas.py
===============================================================================
Creator: Abel Gonçalves Chinaglia
Project: vailá - Multimodal Toolbox
Creation Date: 2026
Update Date: 01 June 2026
Version: 0.3.46
Python Version: 3.12

Description:
------------
Batch crop athlete face photos into square 5 x 5 cm images at 300 DPI. The
module uses MediaPipe Face Detector to locate the largest face in each image,
applies a configurable square crop around the face, resizes the result to
591 x 591 pixels, and optionally overlays the athlete name from the file name.

The GUI workflow follows the vailá directory-selection pattern:
1. Select the input directory with athlete photos.
2. Select the output directory where cropped JPEG images will be saved.

Supported input formats: .jpg, .jpeg, .png, .webp.

Usage:
------
GUI through vailá:
    Tools -> Video and Image -> Crop Face

Standalone GUI:
    uv run python vaila/crop_faces_atletas.py

CLI:
    uv run python vaila/crop_faces_atletas.py --input /path/photos --output /path/out

License:
--------
This program is licensed under the GNU Affero General Public License v3.0.
===============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp"}

SCRIPT_DIR = Path(__file__).resolve().parent
MODELO_PATH = SCRIPT_DIR / "crop_face" / "models" / "face_detector.task"

# 5 cm em 300 DPI aproximadamente 591 pixels.
TAMANHO_SAIDA_PX = 591

# True para inserir o nome no topo.
INSERIR_NOME = True

# Ajustes do crop. Menor margem = mais zoom no rosto.
MARGEM_LATERAL = 0.55
MARGEM_SUPERIOR = 0.55
MARGEM_INFERIOR = 0.45

# Valor negativo sobe o enquadramento; valor positivo desce.
DESLOCAMENTO_VERTICAL = -0.04


def limitar(valor, minimo, maximo):
    """Limit a numeric value to the inclusive interval [minimo, maximo]."""
    return max(minimo, min(valor, maximo))


def fazer_crop_5x5(
    imagem,
    bbox,
    margem_lateral=0.55,
    margem_superior=0.55,
    margem_inferior=0.45,
    deslocamento_vertical=-0.04,
):
    """
    Faz um crop quadrado mais fechado na face.

    Este ajuste deixa a imagem parecida com foto de cadastro:
    - rosto mais próximo;
    - menos ombro;
    - menos tronco;
    - mantém o crop quadrado para imagem 5 x 5 cm.
    """
    altura_img, largura_img, _ = imagem.shape

    x = bbox.origin_x
    y = bbox.origin_y
    w = bbox.width
    h = bbox.height

    x1 = x - margem_lateral * w
    y1 = y - margem_superior * h
    x2 = x + w + margem_lateral * w
    y2 = y + h + margem_inferior * h

    largura_crop = x2 - x1
    altura_crop = y2 - y1

    lado = max(largura_crop, altura_crop)

    centro_x = (x1 + x2) / 2
    centro_y = (y1 + y2) / 2

    centro_y += deslocamento_vertical * lado

    novo_x1 = centro_x - lado / 2
    novo_x2 = centro_x + lado / 2
    novo_y1 = centro_y - lado / 2
    novo_y2 = centro_y + lado / 2

    novo_x1 = int(limitar(novo_x1, 0, largura_img))
    novo_y1 = int(limitar(novo_y1, 0, altura_img))
    novo_x2 = int(limitar(novo_x2, 0, largura_img))
    novo_y2 = int(limitar(novo_y2, 0, altura_img))

    return imagem[novo_y1:novo_y2, novo_x1:novo_x2]


def adicionar_nome(imagem_bgr, nome):
    """Adiciona o nome do atleta no topo da imagem."""
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)

    draw = ImageDraw.Draw(pil_img)

    largura, altura = pil_img.size

    try:
        fonte = ImageFont.truetype(
            "DejaVuSans.ttf",
            size=int(altura * 0.055),
        )
    except Exception:
        fonte = ImageFont.load_default()

    bbox_texto = draw.textbbox((0, 0), nome, font=fonte)
    texto_w = bbox_texto[2] - bbox_texto[0]
    texto_h = bbox_texto[3] - bbox_texto[1]

    padding_x = int(largura * 0.035)
    padding_y = int(altura * 0.018)

    caixa_w = texto_w + 2 * padding_x
    caixa_h = texto_h + 2 * padding_y

    caixa_x1 = int((largura - caixa_w) / 2)
    caixa_y1 = int(altura * 0.015)
    caixa_x2 = caixa_x1 + caixa_w
    caixa_y2 = caixa_y1 + caixa_h

    raio = int(caixa_h * 0.25)

    draw.rounded_rectangle(
        [caixa_x1, caixa_y1, caixa_x2, caixa_y2],
        radius=raio,
        fill=(255, 255, 255),
    )

    texto_x = caixa_x1 + padding_x
    texto_y = caixa_y1 + padding_y - int(texto_h * 0.12)

    draw.text(
        (texto_x, texto_y),
        nome,
        font=fonte,
        fill=(0, 0, 0),
    )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def detectar_maior_face(imagem_bgr, detector):
    """Detecta a maior face da imagem usando MediaPipe Tasks API."""
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.asarray(imagem_rgb),
    )

    resultado = detector.detect(mp_image)

    if not resultado.detections:
        return None

    maior_deteccao = max(
        resultado.detections,
        key=lambda det: det.bounding_box.width * det.bounding_box.height,
    )

    return maior_deteccao.bounding_box


def salvar_com_dpi_300(imagem_bgr, caminho_saida):
    """
    Salva a imagem com metadado de 300 DPI.

    591 x 591 px em 300 DPI corresponde aproximadamente a 5 x 5 cm.
    """
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imagem_rgb)

    pil_img.save(
        caminho_saida,
        quality=95,
        dpi=(300, 300),
    )


def localizar_modelo_padrao():
    """Return the first available MediaPipe face detector model path."""
    candidatos = [
        MODELO_PATH,
        SCRIPT_DIR / "models" / "face_detector.task",
        SCRIPT_DIR.parent / "models" / "face_detector.task",
    ]

    for caminho in candidatos:
        if caminho.exists():
            return caminho

    return MODELO_PATH


def mensagem_modelo_nao_encontrado(modelo_path):
    """Build the user-facing missing-model message."""
    return (
        f"Modelo não encontrado: {modelo_path}\n\n"
        "Baixe o modelo MediaPipe face_detector.task e selecione o arquivo quando "
        "solicitado, ou salve em vaila/crop_face/models/face_detector.task."
    )


def processar_pasta(
    pasta_entrada,
    pasta_saida,
    modelo_path=None,
    tamanho_saida_px=591,
    inserir_nome=True,
):
    """Process all supported images from an input directory into cropped face JPEGs."""
    pasta_entrada = Path(pasta_entrada)
    pasta_saida = Path(pasta_saida)
    modelo_path = Path(modelo_path) if modelo_path else localizar_modelo_padrao()

    if not pasta_entrada.exists():
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {pasta_entrada}")

    if not modelo_path.exists():
        raise FileNotFoundError(mensagem_modelo_nao_encontrado(modelo_path))

    pasta_saida.mkdir(parents=True, exist_ok=True)

    base_options = mp.tasks.BaseOptions
    face_detector = mp.tasks.vision.FaceDetector
    face_detector_options = mp.tasks.vision.FaceDetectorOptions
    vision_running_mode = mp.tasks.vision.RunningMode

    options = face_detector_options(
        base_options=base_options(model_asset_path=str(modelo_path)),
        running_mode=vision_running_mode.IMAGE,
        min_detection_confidence=0.5,
    )

    imagens = sorted(p for p in pasta_entrada.iterdir() if p.suffix.lower() in EXTENSOES_VALIDAS)

    if not imagens:
        print(f"Nenhuma imagem encontrada em: {pasta_entrada}")
        return {"processed": 0, "skipped": 0, "output_dir": pasta_saida}

    processadas = 0
    ignoradas = 0

    with face_detector.create_from_options(options) as detector:
        for caminho_imagem in imagens:
            imagem = cv2.imread(str(caminho_imagem))

            if imagem is None:
                print(f"Não foi possível abrir: {caminho_imagem.name}")
                ignoradas += 1
                continue

            bbox = detectar_maior_face(imagem, detector)

            if bbox is None:
                print(f"Nenhuma face detectada em: {caminho_imagem.name}")
                ignoradas += 1
                continue

            crop = fazer_crop_5x5(
                imagem=imagem,
                bbox=bbox,
                margem_lateral=MARGEM_LATERAL,
                margem_superior=MARGEM_SUPERIOR,
                margem_inferior=MARGEM_INFERIOR,
                deslocamento_vertical=DESLOCAMENTO_VERTICAL,
            )

            if crop.size == 0:
                print(f"Crop vazio em: {caminho_imagem.name}")
                ignoradas += 1
                continue

            crop_final = cv2.resize(
                crop,
                (tamanho_saida_px, tamanho_saida_px),
                interpolation=cv2.INTER_AREA,
            )

            nome_atleta = caminho_imagem.stem

            if inserir_nome:
                crop_final = adicionar_nome(crop_final, nome_atleta)

            caminho_saida = pasta_saida / f"{nome_atleta}_resultado.jpeg"

            salvar_com_dpi_300(crop_final, caminho_saida)

            print(f"Salvo: {caminho_saida}")
            processadas += 1

    return {"processed": processadas, "skipped": ignoradas, "output_dir": pasta_saida}


def selecionar_modelo(root):
    """Ask the user to select face_detector.task when the default model is absent."""
    modelo_path = localizar_modelo_padrao()
    if modelo_path.exists():
        return modelo_path

    messagebox.showwarning(
        "Crop Face - modelo não encontrado",
        mensagem_modelo_nao_encontrado(modelo_path),
        parent=root,
    )
    selecionado = filedialog.askopenfilename(
        parent=root,
        title="Select MediaPipe face_detector.task",
        filetypes=[("MediaPipe task model", "*.task"), ("All files", "*.*")],
    )

    return Path(selecionado) if selecionado else None


def run_crop_faces_atletas_gui(parent=None):
    """Run the Crop Face workflow with vailá-style directory selection dialogs."""
    root = parent
    root_created = False

    if root is None:
        root = Tk()
        root.withdraw()
        root_created = True

    try:
        pasta_entrada = filedialog.askdirectory(
            parent=root,
            title="Select input directory with athlete photos",
        )
        if not pasta_entrada:
            return None

        pasta_saida = filedialog.askdirectory(
            parent=root,
            title="Select output directory for cropped faces",
        )
        if not pasta_saida:
            return None

        modelo_path = selecionar_modelo(root)
        if not modelo_path:
            return None

        resultado = processar_pasta(
            pasta_entrada=pasta_entrada,
            pasta_saida=pasta_saida,
            modelo_path=modelo_path,
            tamanho_saida_px=TAMANHO_SAIDA_PX,
            inserir_nome=INSERIR_NOME,
        )

        messagebox.showinfo(
            "Crop Face finished",
            "Crop Face processing finished.\n\n"
            f"Processed images: {resultado['processed']}\n"
            f"Skipped images: {resultado['skipped']}\n"
            f"Output directory: {resultado['output_dir']}",
            parent=root,
        )
        return resultado
    except Exception as exc:
        messagebox.showerror("Crop Face error", str(exc), parent=root)
        return None
    finally:
        if root_created:
            root.destroy()


def parse_args(argv=None):
    """Parse CLI arguments for batch face cropping."""
    parser = argparse.ArgumentParser(
        description="Crop athlete face photos into 5 x 5 cm JPEG images at 300 DPI."
    )
    parser.add_argument("-i", "--input", help="Input directory with athlete photos.")
    parser.add_argument("-o", "--output", help="Output directory for cropped JPEG images.")
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Path to MediaPipe face_detector.task. Defaults to vaila/crop_face/models/face_detector.task.",
    )
    parser.add_argument(
        "--no-name",
        action="store_true",
        help="Do not overlay the athlete name on the cropped image.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=TAMANHO_SAIDA_PX,
        help="Output square size in pixels. Default: 591.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """CLI entry point; launches GUI when input/output are not provided."""
    args = parse_args(argv)

    if not args.input or not args.output:
        run_crop_faces_atletas_gui()
        return 0

    processar_pasta(
        pasta_entrada=args.input,
        pasta_saida=args.output,
        modelo_path=args.model,
        tamanho_saida_px=args.size,
        inserir_nome=not args.no_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
