# Lance Decifrado TV: 1º gol do Grêmio — lançamento + finalização (2026-11-02)

## O que estava errado

`sam3pose/enviar/para_enviar_lancamento_tv_20261102/` (build `resultados_lancamento_tv_20261102/build_explicacao_tv.py`)
usava `GOLGREMIO _frame_166_to_186` e `_850_to_870` como saída e chegada do tiro de meta do Weverton.
Esses dois recortes são **o chute do Carlos Vinícius** (2ª/3ª folha de `sam3pose/Lancamento_e_gol.pdf`).
A saída do lançamento é `GOLGREMIO _frame_1631_to_1700.mp4` (quadro 1631 = 1ª folha do PDF, achado por
correlação de imagem e conferido a olho); o voo é `GOLGREMIO _frame_716_to_781.mp4`.

## Pacote novo (um relatório, duas partes)

Build: `sam3pose/enviar/resultados_gol_gremio_tv_20261102/build_tv.py` →
`sam3pose/enviar/para_enviar_gol_gremio_tv_20261102/` (+ `.zip`). O pacote antigo ficou onde estava.

| Parte | Número | Fonte |
| --- | --- | --- |
| Lançamento | 64,82 m · 126,6 km/h (35,17 m/s) · 2,77 s | equipe |
| Finalização | 24,3 m ao gol · 105,4 km/h (29,3 m/s, 0,83 s) | equipe |
| Finalização | posição: 24,3 m da linha, 7,8 m fora da área, alinhado com a trave mais próxima | vailá, reescalado |
| Finalização | chutador → goleiro 14,0 m (goleiro ~11,3 m da linha) | vailá, reescalado |

vailá: FreeKiki 3D do recorte 166–186, quadro 0 (a bola já voa), SAM id 35 = Carlos Vinícius, id 23 = goleiro
(conferidos na imagem). Escala k = 24,3 / 20,35 = 1,194 em torno da trave mais próxima (−52,45; −3,66): o 3D
monocular dá 20,35 m da bola à trave, a medida manual 24,3 m.

`validacao_entrega.json`: vídeo 1920×1080, 60 s, 1800 quadros; PDF com duas partes; a parte do lançamento
só com distância, velocidade e tempo; a da finalização com os quatro itens; sem top view.

## Regenerar

```bash
UV_NO_SYNC=1 uv run --no-sync python \
  "/home/preto/data/Lance_decifrado/gremio_amg01nov/sam3pose/enviar/resultados_gol_gremio_tv_20261102/build_tv.py"
```
