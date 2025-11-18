import platform

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image


def check_device():
    if torch.cuda.is_available():
        return "cuda"
    elif platform.machine() == "arm64" and platform.system() == "Darwin":
        return "mps"  # Metal Performance Shaders para chips Apple (M1, M2, M3)
    else:
        return "cpu"


def load_pipeline(device):
    # Ajustar o tipo de dado conforme o dispositivo
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Carregar o modelo pré-treinado
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch_dtype
    ).to(device)

    return pipeline


def upscale_image(
    input_image_path,
    output_image_path,
    prompt="Ultra high definition, 4K, very detailed, smooth, vibrant colors",
):
    # Detectar o dispositivo
    device = check_device()
    print(f"Using device: {device}")

    # Carregar o pipeline adequado ao dispositivo
    pipeline = load_pipeline(device)

    # Carregar a imagem de entrada
    image = Image.open(input_image_path).convert("RGB")

    # Realizar o upscale
    upscaled_image = pipeline(prompt=prompt, image=image).images[0]

    # Salvar a imagem upscaled
    upscaled_image.save(output_image_path)
    print(f"Image saved to: {output_image_path}")


# Exemplo de uso
input_image_path = "000000001.png"
output_image_path = "upscaler_000000001.png"
upscale_image(input_image_path, output_image_path)
