# FreeKiki: super-dataset, select, pseudo labels, detect screen, snap (handoff, 2026-10-15)

Continuation note for the next agent (Antigravity, Cursor, Claude Code). Code of this round is committed
on `main` as `9eee0d7d update freekiki` (pushed). Workspace (not in git): `W=/home/preto/data/FreeKiki`.
Always run `uv run --no-sync` (CUDA machine: a bare `uv run` downgrades torch to CPU).

## Current state

- **Model in use:** slot `l` = run `kiki49_l_ft_v011` (`models/freekiki_l.pt`, sha256 `fda8f62108dc349c...`),
  and `active` points to slot `l`. Older weights stay in `models/` (`kiki49_kiki49_l_ft_v007.pt`, backups
  `freekiki_l_before_*.pt`).
- **v011 numbers.** Val: pose mAP50-95 0.8869, PCK10 0.7835, homography correct 0.870.
  Test (3 622 images, run once on 2026-10-14, `outputs/processed_freekiki_eval_test_20261014_114233`):
  mAP50-95 0.9076, recall 0.9314, precision 0.9186, PCK10 0.8439.
  Test recall with `--geom fill`: p5 0.85 (n 111), p29 0.30 (n 33), p39 0.90 (n 105), p47 0.56 (n 18).
- **Dataset:** train has 37 201 images. The extension sources are roboflow, soccernet_gsr and soccernet_rescue;
  human-reviewed frames have source `freekiki_review`. The pseudo labels were retired with `remove-source`
  (files kept in `datasets/kiki49/_removed/`). On 2026-10-14 the user removed 712 TS-WorldCup train frames that
  overlapped test, 28 frames with label issues and 131 outliers above 200 px (backups
  `datasets/kiki49/manifest_backup_*.csv`).
- **Still weak:** p29 and p47 (the near-touchline right corner and flag). The val split holds only 18-33 labels
  for each of them, many projected rather than seen. Use `evaluate --origin 1` (human-annotated labels only)
  before believing any change on these points.

## Commands added this round (all in `vaila/freekiki.py`)

| command | what it does |
|---|---|
| `extend --source roboflow\|gsr\|soccernet-rescue` / `--src S --commit` | external datasets converted to the 49 points (`vaila/kiki49_build/`), staged, then appended to train only (leak and dup gates) |
| `select --src GAMES --out CLIPS` | finds the moments where p5/p29/p39/p47 show in long games and cuts them (Cut Video ffmpeg). It uses a camera check (rmse <= 5 px, height 8-45 m, >= 6 points) and a MediaPipe pose gate (`freekiki_posegate.py`, pose > 1/10 of the image = close-up) |
| `pseudo --batch DETECT` / `pseudo-queue` | complete labels from network + fitted camera, for review only. Pseudo p29 was wrong in about 90 % of frames: never commit unreviewed pseudo labels |
| `remove-source --source X [--apply]` | retires a source from train (backup, files moved, reversible) |
| `detect` (folder) | **screens each video** first (24 sampled frames): a camera the network was not trained for (zoom, behind the goal, close-up) is skipped and listed in `skipped_videos.csv`. Options: `--no-screen`, `--min-camera-ok 0.3` |
| `snap --run DETECT_RUN_OR_BATCH` | moves accepted points onto the painted lines (`freekiki_snap.py`); writes `<run>/snap/` (CSV, moves, overlay). Network CSVs are untouched |

`evaluate` also writes `per_keypoint_origin_<digits>.csv` (`--origin`, default 1 = human-annotated).

## snap: what it does and why it is shaped like this

- **Problem.** On rare cameras (palxscri clip 17953: high, lateral, near the box) the network draws the penalty box
  20-30 px too large, and the fitted camera inherits the error.
- **Method.** Straight painted lines are refitted to the paint in a 35 px band; a corner moves to the intersection of
  its lines and a point on one line moves to its foot.
- **Guards** (user reports: "more shift and flicker", then "worse on FIFA MOR_POR_184642, goal-area markers oscillate"):
  - a point within 3 px of its refitted line(s) is kept;
  - a refitted line must lie on paint for 60 % of its length;
  - lines crossing at less than 15 degrees give no corner;
  - stabilisation runs per visible stretch and picks the track with the lower *mean* second difference;
  - a stretch whose median correction is below 3 px keeps the network points exactly.
- **Result:**

  | clip | paint distance (network -> snap) | jitter p90 (network -> snap) |
  |---|---|---|
  | FIFA MOR_POR_184642 | identical to network | 1.91 -> 1.91 |
  | palxscri 17953 | 19.1 -> 0.0 px | 10.4 -> 3.4 px |

- The user concluded snap is not needed on cameras the network knows. It stays an optional step: it is **not** in
  `detect` nor in the DLT export.

## Open items

1. **[RESOLVIDO 2026-10-16]** Snap atualizado nos 20 vídeos FIFA em `outputs/fifa_v011_full/processed_freekiki_batch_20261015_140142/*/snap/`
   e nos lotes `palx_v011_screened` e `palx_v011`.
   - Melhorias no `freekiki_snap.py`:
     1. Busca de crista 1D (`fit_line_in_band`) imune ao arco da grande área e a jogadores dentro da faixa de busca;
     2. Projeção de pontos de linha única (cruzamento do arco) na direção das linhas transversais projetivas (mantém coordenada Y na grande área);
     3. Preenchimento de dropouts transitórios em `stabilize` sem oscilações abruptas de 40 px.
   - Testes: 16 unit tests passando em `tests/test_freekiki_snap.py`.
2. **[RESOLVIDO 2026-10-16]** Snap integrado por padrão em `detect` (com opt-out via `--no-snap` e checkbox na GUI).
   - O modo inferior não ajustado (red) foi descartado das visualizações padrão (overlay e snapshots mostram apenas os círculos verdes ajustados nas linhas pintadas).
   - Os arquivos de calibração DLT (`.dlt2d`, `.dlt3d`), a homografia e `field_kps_getpixelvideo.csv` agora utilizam diretamente os pontos ajustados (snapped) e estabilizados temporalmente, melhorando a calibração de câmeras 2D/3D.
   - Os dados brutos da rede continuam preservados em `field_kps_raw.csv` e as correções registradas em `snap_moves.csv`.
3. p29/p47: diagnóstico concluído sobre rotação $C_2$ (SoccerNet com coordenadas fixadas no estádio e 0 rótulos humanos em val); coleta de mais cantos rotulados por humanos via `select` -> `queue --need p5,p29,p39,p47 --montage` -> `getpixelvideo` -> `ingest`.
4. **[RESOLVIDO 2026-10-16]** Model card `v011` criado em `release/MODEL_CARD_freekiki_l_v011.md` com números limpos de teste/val e documentação do diagnóstico dos keypoints raros.
5. A chave Roboflow no `.env` gitignorado (`ROBOFLOW_API_KEY`) pode ser rotacionada pelo usuário quando desejado.

## Rules the user set (keep)

- Do not start long trainings; give the command (the user runs it).
- Ask before changing the dataset (`extend --commit`, `ingest --commit`, `remove-source --apply`).
- Never choose thresholds or hyper-parameters on the test split.
- Do not promote models or change the default slot without asking.
- On every script change, update the header (Update Date, Version) and the README `Last updated`, the help index
  `Generated on` and the module help `Updated` lines.
- Replies to the user in Portuguese.

## Key files

`vaila/freekiki.py`, `vaila/freekiki_snap.py`, `vaila/freekiki_posegate.py`, `vaila/freekiki_geom.py`,
`vaila/kiki49_build/`, tests `tests/test_freekiki_{select,posegate,pseudo,screen,snap,extend}.py`;
help `vaila/help/freekiki.md|html`, `docs/vaila_buttons/freekiki.md`.
