"""
usvideoia.py

Module to provide Video Super-Resolution (VSR) upscale functionality as a standalone script
that can be imported into resize_video.py or run directly.

Created by: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br 
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 May 2025
Updated: 09 May 2025
Version: 0.0.1

Description:
    This module provides a standalone script for Video Super-Resolution (VSR) upscale functionality.
    It can be imported into resize_video.py or run directly.

Usage:
    This module can be imported into resize_video.py or run directly.
    Directly: python usvideoia.py
    Import: from vaila import usvideoia
    usvideoia.run_usvideoia()   

Dependencies:
    - mmagic
    - torch
    - opencv-python
    - tqdm
    - einops

License:
    This project is licensed under the GNU General Public License v3.0. See the LICENSE file for details.

"""

import os
import argparse
import torch
import cv2
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from mmagic.apis import init_model, inference_video
from mmagic.utils import register_all_modules

# ----------------------------
# Helpers de I/O de vídeo
# ----------------------------
def extract_frames(video_path, temp_dir, fps=30):
    os.makedirs(temp_dir, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(f"{temp_dir}/{idx:05d}.png", frame)
        idx += 1
    cap.release()
    return idx

def reconstruct_video(frames_dir, out_path, fps=30):
    files = sorted(Path(frames_dir).glob("*.png"))
    h, w = cv2.imread(str(files[0])).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in files:
        frame = cv2.imread(str(f))
        writer.write(frame)
    writer.release()

# ----------------------------
# Inferência VSR com MMagic
# ----------------------------
def vsr_inference(
    input_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
):
    temp = Path(tempfile.mkdtemp(prefix="vsr_"))
    in_frames = temp / "in"
    sr_frames = temp / "sr"
    in_frames.mkdir()
    sr_frames.mkdir()

    # 1) Extrai frames
    num_frames = extract_frames(input_dir, in_frames)

    # 2) Processa em janelas deslizantes de 5 frames
    idxs = list(range(num_frames))
    pad = 2
    padded = [max(0, i) for i in idxs[:pad]] + idxs + [min(num_frames-1, i) for i in idxs[-pad:]]
    
    for i in tqdm(range(num_frames), desc="VSR Inference"):
        # constrói janelinha
        window = padded[i:i+5]
        # carrega frames
        frames = []
        for j in window:
            img = cv2.imread(f"{in_frames}/{j:05d}.png")
            frames.append(img)
        
        # inferência com MMagic
        with torch.no_grad():
            sr = model.inference(frames)
        
        # salva frame super-resolvido
        sr_img = sr.squeeze(0).cpu().clamp(0,1)
        sr_img = rearrange(sr_img, 'c h w -> h w c').numpy() * 255
        sr_img = cv2.cvtColor(sr_img.astype('uint8'), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{sr_frames}/{i:05d}.png", sr_img)

    # 3) Reconstrói vídeo
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (input_dir.stem + "_upscaled.mp4")
    reconstruct_video(sr_frames, out_path)
    shutil.rmtree(temp)
    print(f"Upscaled salvo em {out_path}")

# ----------------------------
# Entry Point
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  "-i", required=True, type=Path)
    parser.add_argument("--output_dir", "-o", required=True, type=Path)
    parser.add_argument("--config",     "-c", required=True, type=str)
    parser.add_argument("--checkpoint", "-k", required=True, type=str)
    parser.add_argument("--device",     "-d", default="cpu")
    args = parser.parse_args()

    # Registra todos os módulos MMagic
    register_all_modules()
    
    # Inicializa o modelo
    device = torch.device(args.device)
    model = init_model(args.config, args.checkpoint, device=device)

    vsr_inference(args.input_dir, args.output_dir, model, device)

if __name__ == "__main__":
    main()