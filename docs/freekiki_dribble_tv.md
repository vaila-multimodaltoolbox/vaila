# Drible para TV: duas câmeras e relógio original

Version: 0.4.14 · Updated: 02 November 2026

O `openplay-report` mantém os relatórios existentes por `--folder` e acrescenta
uma entrada JSON para um driblador e até três defensores, com uma identidade
global por pessoa. Os IDs SAM de cada câmera não são identidades globais.

```bash
.venv/bin/python -m vaila.freekiki openplay-report \
  --manifest manifesto.json --output /dados/resultados_novos \
  --kind dribble --lang pt --animation --prepare-originals --rerun-sam
```

SAM 3D Body exige CUDA e os modelos locais já provisionados pelo setup existente.
Não muda dependências nem baixa modelos por conta própria. O processo de inferência
termina antes da próxima câmera. Para reconstruir apenas os produtos derivados,
omita `--prepare-originals --rerun-sam` e preserve a pasta `cameras/`.

O manifesto contém `schema: "freekiki-dribble-tv-v1"`, `participants` e `cameras`:

```json
{
  "schema": "freekiki-dribble-tv-v1",
  "participants": [
    {"id": 0, "role": "dribbler", "name": "Driblador", "color": "#ffe25c",
     "camera_ids": {"aberta": 41, "principal": 35}},
    {"id": 1, "role": "defender", "name": "Defensor 1", "color": "#5dd6ff",
     "camera_ids": {"aberta": 0, "principal": 23}}
  ],
  "cameras": [
    {"id": "aberta", "main": false, "range_1based": [4, 81],
     "video": "original_aberta.mp4", "seed_video": "equalized_aberta.mp4",
     "frame_map": "reference_frames.csv", "dlt2d": "aberta.dlt2d", "dlt3d": "aberta.dlt3d",
     "ball_markers": "equalized_aberta_markers.csv",
     "field_points": "field_kps_getpixelvideo.csv",
     "seed_camera": "equalized_aberta_sam3dinov3_camera.csv",
     "seed_keypoints": "equalized_aberta_sam3dinov3_keypoints2d.csv"},
    {"id": "principal", "main": true, "range_1based": [1, 78],
     "video": "original_principal.mp4", "seed_video": "equalized_principal.mp4",
     "frame_map": "source_frames.csv", "dlt2d": "principal.dlt2d", "dlt3d": "principal.dlt3d",
     "ball_markers": "equalized_principal_markers.csv",
     "field_points": "principal_field_kps_getpixelvideo.csv",
     "seed_camera": "equalized_principal_sam3dinov3_camera.csv",
     "seed_keypoints": "equalized_principal_sam3dinov3_keypoints2d.csv"}
  ]
}
```

Os caminhos relativos são resolvidos em relação ao manifesto. As faixas são
inclusivas e começam em 1; tabelas internas de poses/DLT usam índice 0. As duas
câmeras devem ter o mesmo FPS original e número de pares. Coloque a aberta
primeiro, a principal depois. O mapa RIFE deve fornecer `source_position` racional,
`kind` e uma linha por quadro equalizado. `seed_keypoints` é preservado como
referência de proveniência; as poses usadas são as novas inferências.

Preparação: registro robusto das imagens, checagem por fluxo óptico de ida/volta
dos pontos aceitos e transferência das matrizes DLT. Interrompe se houver menos
de quatro pontos conferidos ou RMSE de campo acima de 10 px. A escala e a
calibração vertical continuam herdadas. A bola amarela é localizada por componente
de cor próximo à semente na imagem original; ausência/oclusão deixa lacuna.
Este localizador é específico deste material com bola amarela. As caixas SAM
selecionadas são transferidas às imagens originais e alimentam o SAM 3D Body
sem máscaras; não há nova segmentação SAM 3.

A bola é triangulada com os dois DLT3D. O CSV retém resultados rejeitados;
as séries aceitas exigem reprojeção ≤10 px em cada câmera, ângulo de raios ≥1°
e altura entre −0,1 e 4 m (tolerância numérica/geométrica, não altura validada).
Valores negativos são diagnósticos de erro vertical. O DLT2D só coloca os
tornozelos estimados no plano; nunca fixa a altura da bola. O ponto do jogador
usa a câmera principal consistentemente, sem fundir estimativas entre vistas.

Suavização Savitzky–Golay de ordem 2 com janelas de 5/7/9 quadros, por trechos
contíguos. Derivadas não atravessam lacunas. A velocidade em tela é arredondada
e omitida quando a dispersão entre filtros excede 25% da média ou 0,7 m/s,
o maior. Acelerações e curvaturas permanecem séries diagnósticas, com limites
de modelo e calibração explícitos. Distância bola–driblador no plano é distinta
da distância 3D ao ponto de apoio no chão.

Mudanças de direção (opcional): `turns` no manifesto, em pares originais com base 0.

```json
"turns": [
  {"id": "meia_volta", "name": "Meia-volta com a bola", "primary": "body",
   "body_frames": [5, 21], "before": [0, 5], "after": [21, 26]},
  {"id": "curva_final", "name": "Curva antes do desfecho", "primary": "path",
   "before": [20, 34], "after": [53, 65], "break_frame": 53}
]
```

`before`/`after` são trechos retos escolhidos no vídeo; o ângulo é a diferença entre as
direções das cordas, o tempo vai do fim de `before` ao início de `after`, e as velocidades de
entrada e saída são as das cordas (filtros 5/7/9). `body_frames` mede o giro da linha do
quadril do SAM 3D Body no referencial de cada câmera (meia-volta quase parado, quando a
trajetória não tem direção). `primary` escolhe o número do card. Zonas de Monteiro, Santiago
et al. (2025) só como leitura qualitativa. Para converter quadros do vídeo equalizado (RIFE)
em originais, use `source_position` de `source_frames.csv` (base 1) menos 1.

Produtos: `lance_sam3body.mp4` (78 quadros no FPS racional original neste lance),
`explicacao_tv.mp4` (1800 quadros, 30 fps, 60 s), relatório HTML/PDF, figuras,
conferências de início/disputa/fim, roteiro e SRT, hashes SHA-256, observações
brutas e séries CSV/JSON. Câmera lenta repete quadros dos originais; não cria
medidas. O relatório documenta a analogia presa–predador e as quatro referências
indicadas, sem inferir intenção, força muscular ou reação neural.

```bash
.venv/bin/pytest tests/test_freekiki_openplay_manifest.py \
  tests/test_freekiki_openplay_kinematics.py tests/test_freekiki_dribble.py \
  tests/test_freekiki_pass.py tests/test_freekiki_openplay_report.py -q
```

Além dos testes, confira as correspondências e as sobreposições nos originais.
`ffprobe` deve confirmar resolução, FPS, número de quadros e timestamps; uma
decodificação completa com `ffmpeg -v error -i VIDEO -f null -` verifica o arquivo.
O PDF usa o Chrome já adotado pelo FreeKiki; se o ambiente impedir o navegador,
o HTML continua disponível e a ausência de PDF é informada.
