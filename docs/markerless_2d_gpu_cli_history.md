# Markerless 2D — histórico da sessão (GPU, bootstrap, CLI)

Notas guardadas para continuar sem depender do histórico do IDE. Para export do chat Cursor: usar **Export / Copy** no painel da conversa.

## Problemas tratados

1. **Pose fraca nos primeiros frames** — `RunningMode.VIDEO` aquece o tracker devagar.
2. **Dois landmarkers na GPU o vídeo inteiro** — segundo modelo (IMAGE) mantido até ao fim consumia VRAM e podia **atrasar** o pipeline.
3. **NVENC por defeito** — `ffmpeg` com `h264_nvenc` usava ~0.5–1 GiB VRAM **sem** acelerar MediaPipe; podia competir com o delegate de pose.

## Alterações em `vaila/markerless_2d_analysis.py` (resumo)

- **Bootstrap IMAGE só nos primeiros N frames**: segundo `PoseLandmarker` em `IMAGE` é criado só para `frame_count < image_bootstrap_num_frames`, depois **`__exit__`** — um só modelo na GPU no restante vídeo.
- **`image_bootstrap_num_frames`** (defeito **45**) em `[mediapipe]` + campo na GUI (*IMAGE refine first N frames*).
- **`use_nvenc_encoder`**: defeito **false**; secção TOML **`[video_encoding] use_nvenc = false`**; checkbox na GUI.
- **`libx264`**: `-threads 0`, `-preset veryfast`, `-crf 20` quando NVENC está off.
- **`psutil.cpu_percent(interval=None)`** — evita bloqueios de 100 ms nos checks de CPU.
- **`CAP_PROP_BUFFERSIZE = 1`** na captura (quando suportado).
- **CLI headless**: subcomando **`batch`** (sem Tk).

## CLI (GPU NVIDIA)

Na raiz do clone:

```bash
uv run python vaila/markerless_2d_analysis.py batch \
  -i /CAMINHO/pasta_com_videos \
  -o /CAMINHO/saida_base \
  -c /CAMINHO/config.toml \
  --device nvidia \
  --no-sleep-between-videos
```

- **`--nvenc`**: força encode GPU no MP4 anotado (opcional).
- **`--libx264-encode`**: força CPU encode mesmo que o TOML peça NVENC.
- **`--help`**: ajuda completa.

Exemplo com vídeo isolado: pasta só com um `.mp4`.

## TOML relevante

```toml
[mediapipe]
image_bootstrap_first_frame = true
image_bootstrap_num_frames = 45   # subir para 60–120 se clip cortado ainda “demora” a ficar bom

[video_encoding]
use_nvenc = false                  # true só se quiseres offload do encode e aceitares VRAM extra
```

## Ficheiros de saída (por vídeo)

Subpasta timestampada sob o output escolhido; por vídeo: `*_mp.mp4`, CSVs MediaPipe, `configuration_used.toml`, etc.

## Data da sessão

2026-05-03 — branch `trainsoccer`, RTX 4090 / Linux, vídeo de teste mencionado sob `AnnaLeonel`.
