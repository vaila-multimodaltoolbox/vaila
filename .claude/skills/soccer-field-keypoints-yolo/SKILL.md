---
name: soccer-field-keypoints-yolo
description: Treino Ultralytics YOLO pose para keypoints do campo de futebol (32 kp), export ONNX, e inferência em vídeo com vaila.soccerfield_keypoints_ai (CSV getpixelvideo + overlay). Use quando o usuário fala em football pitch keypoints, YOLO26 pose, treino Roboflow/HF dataset martinjolif, field_keypoints_getpixelvideo, calibração FIFA, ou soccerfield_keypoints_ai.
---

# Soccer field keypoints — YOLO pose + inferência vailá

Esta skill resume o que foi implementado e como retomar em **outra IDE ou LLM** sem perder o fio: leia este ficheiro, depois [`AGENTS.md`](../../../AGENTS.md) (secção de handoff) e o módulo [`vaila/soccerfield_keypoints_ai.py`](../../../vaila/soccerfield_keypoints_ai.py).

## Handoff rápido (outro agente)

1. Abrir este `SKILL.md`.
2. Abrir `vaila/soccerfield_keypoints_ai.py` — CLI/GUI, backends `roboflow` e `ultralytics`, vídeo frame-a-frame.
3. Dataset YOLO local: `vaila/models/hf_datasets/football-pitch-detection/data/data.yaml` (`kpt_shape: [32, 3]`).
4. Referência 3D vailá (nomes de pontos): `vaila/models/soccerfield_ref3d.csv` — **mapeamento semântico kp_00..31 → nomes** ainda é trabalho futuro se os IDs do dataset não coincidirem com a vailá.

## O que existe no código

- **Módulo**: `vaila/soccerfield_keypoints_ai.py`
  - `--mode frame|video`, `--weights`, `--imgsz`, `--conf`, `--draw-min-conf`, `--device`, `--stride`, `--max-frames`, `--overlay-video`.
  - Saídas numa pasta `processed_field_kps_<timestamp>/` sob `-o`:
    - `field_keypoints_video.csv` (longo: frame, name, x, y, conf)
    - `field_keypoints_getpixelvideo.csv` (wide: frame, p1_x, p1_y, …)
    - `field_keypoints_overlay_markers.csv` (mesmo schema do getpixelvideo; mesmo filtro `draw_min_conf`; comparação no mesmo run)
    - `field_keypoints_overlay.mp4` se `--overlay-video`
  - **Resolver de pesos**: se `--weights` aponta para `.../<run_name>/weights/best.pt` mas o Ultralytics criou `.../<run_name>-N/`, o código procura candidatos por glob no repo e escolhe o mais recente.

- **GUI**: botão em `vaila.py` que chama este módulo (`run_vaila_module("vaila.soccerfield_keypoints_ai", ...)`).

## Treino (Ultralytics YOLO pose) — receita que funcionou melhor

Problema observado com treino “ingênuo” (`yolo26x-pose`, `imgsz=640`, `mosaic=1.0`): o modelo **colapsava** os 32 keypoints num cluster ~20–30 px dentro do bbox (métricas OKS enganosas com bbox enorme).

Receita **mais estável** para campo broadcast:

- `model=yolo26s-pose.pt` (ou `m`; `x` só com muito mais dados/regularização)
- `imgsz=1280`
- `mosaic=0.0`, `mixup=0.0`, `close_mosaic=0`, `erasing=0.0`
- `pose=25.0`, `kobj=2.0`
- `epochs` alto (ex. 800) com `patience` ajustável; se aparecer **Loss NaN/Inf**, o Ultralytics tenta recuperar; estabilizar com `lr0` mais baixo ou `amp=False` se necessário.

Exemplo (ajustar `project`/`name` para não duplicar segmentos `runs/pose` no caminho):

```bash
uv run yolo pose train \
  model=yolo26s-pose.pt \
  data=vaila/models/hf_datasets/football-pitch-detection/data/data.yaml \
  epochs=800 imgsz=1280 batch=8 \
  mosaic=0.0 mixup=0.0 close_mosaic=0 erasing=0.0 \
  pose=25.0 kobj=2.0 \
  project=vaila/models/runs/pose/vaila/models/training \
  name=football_pitch_prod
```

**Atenção ao caminho de saída**: se `project=` já contiver `runs/pose/...`, o Ultralytics pode **prefixar** outro `runs/pose/`, gerando caminhos longos duplicados. Verificar sempre o log `Results saved to ...` e usar esse caminho absoluto nos comandos seguintes.

## Export ONNX

Usar o caminho **absoluto** do `best.pt` que o treino gravou:

```bash
uv run yolo pose export \
  model=/ABS/PARA/.../weights/best.pt \
  format=onnx
```

Pode aparecer um `UserWarning` sobre `aten::index` no export — em geral o ONNX continua utilizável; validar com `yolo predict` ou com o módulo vailá.

## Inferência no vídeo de teste (frame a frame + overlay)

```bash
uv run python -m vaila.soccerfield_keypoints_ai \
  --mode video \
  -i tests/sport_fields/ENG_FRA_224512.mp4 \
  -o tests/sport_fields/ \
  --backend ultralytics \
  --weights /ABS/PARA/.../weights/best.pt \
  --imgsz 1280 --conf 0.3 --draw-min-conf 0.05 \
  --device 0 --start 0 --stride 1 --overlay-video
```

Para teste rápido: `--max-frames 200` ou `--stride 2`.

## Dependências relevantes

- `ultralytics`, `opencv-python`, `pandas`, `numpy`
- Opcional Roboflow: `inference`, `supervision` + `ROBOFLOW_API_KEY` (backend remoto).

## Próximos passos (fora desta skill)

- **Auto-mapping** dos índices YOLO (0..31) para `point_name` em `soccerfield_ref3d.csv` (transfer learning geométrico / homografia).
- Rotular frames FIFA e **fine-tune** a partir do `best.pt` atual.

## Exemplos de prompts que devem acionar esta skill

- “Treinar YOLO pose no dataset football-pitch-detection e exportar ONNX”
- “Rodar inferência de keypoints do campo no MP4 e gerar CSV no formato getpixelvideo”
- “O export diz FileNotFoundError no best.pt”
- “Os keypoints estão todos no mesmo pixel / colapsados”
