# FreeKiki TV: lançamento do Weverton (Grêmio × Galo, 01/11/2026)

## Brief TV (Sérgio Cunha)

Só dois números na matéria: distância alcançada e velocidade média da bola
(ambos excelentes). Curva que engana o zagueiro e biomecânica do corpo do
goleiro ficam para outra oportunidade.

## Números (consenso Paulo / Sérgio, WhatsApp 01–02/11)

| Métrica | Valor |
| --- | --- |
| Distância no chão (linha reta, aprox.) | 64.82 m |
| Tempo de voo | ~2.7666 s |
| Velocidade média | 35.17 m/s = 126.6 km/h |

Recorte de meio de campo `frames 716–781` = 66 quadros @ 23.976 fps ≈ 2.753 s.

## Pacote

Build: `/home/preto/data/Lance_decifrado/gremio_amg01nov/sam3pose/enviar/resultados_lancamento_tv_20261102/`

Para enviar: `.../enviar/para_enviar_lancamento_tv_20261102/` (+ `.zip` irmão)

- `explicacao_tv.mp4` — 60 s, 30 fps, 1920×1080, 1800 frames (ffprobe OK; decode limpo)
- overlays saída/chegada + `lance_voo.mp4`
- **sem top view** (apagadas: projeção no solo errada nos runs FreeKiki)
- `relatorio_lancamento.html` + `.pdf` com frames de conferência
- `data/throw_analysis.json`, roteiro/SRT, hashes, `validacao_entrega.json` (`no_topview_files: true`)

Regenerar:

```bash
UV_NO_SYNC=1 uv run --no-sync python \
  /home/preto/data/Lance_decifrado/gremio_amg01nov/sam3pose/enviar/resultados_lancamento_tv_20261102/build_explicacao_tv.py
```

## Limite

Os fits FreeKiki monocular dos recortes 166–186 e 850–870 **não** são a fonte
dos 64.82 m / 126.6 km/h: a projeção no solo não é contínua entre pans da
broadcast. Números TV = medição da equipe.

## Fora deste pacote

Chute do Carlos Vinícius (~24.3 m, ~105 km/h) está no PDF narrativo
`gol gremio.pdf` — pacote TV à parte se pedido.
