# FreeKiki: entrada observada e revisão da calibração monocular

Data: 02 November 2026. Versão global: 0.4.14. Sem commit.

## Resultado e limite científico

O usuário confirmou o **último quadro, 14 (base 0)**, como o cruzamento da linha. O
run foi recalculado em uma cópia separada. O quadro 13 não foi usado como entrada.
A chegada passou de mão estimada do goleiro para **entrada observada**, com os
cantos e o centro da bola revisados na imagem ampliada do quadro 14.

O resultado permanece **provisório**: RMS da bola **3.405 px**, máximo **10.031 px**,
acima do limite de 3 px. JSON, terminal e relatório exibem a necessidade de revisão.
A entrada foi corrigida, mas a consistência global do voo ainda exige revisar as
marcações dos demais quadros, sobretudo o quadro 4 (maior resíduo). Não considerar
altura, velocidade ou rotação medições independentes de uma câmera.

## Diagnóstico e revisão manual

O código anterior interpretava uma entrada não reconhecida como defesa e criava
uma âncora na média dos punhos do jogador estimado mais próximo do gol. Neste run,
ela era `(-49.983, -4.777, 1.029) m`, 1.117 m além da meia largura da baliza. Seu
resíduo físico era 2.368 m, embora `quality` dissesse `ok`.

Com os marcadores anteriores no quadro 14, a homografia local dava
`y=-4.413, z=1.674 m`; a câmera global dava `y=-2.839, z=1.322 m`: diferença de
**1.613 m**. A projeção de p34 tinha erro de **14.243 px**. Não eram evidências
compatíveis com aceitar a entrada pela aparência da animação.

Revisão visual por Codex, restrita à cópia, sobre pixels do vídeo original ampliado:

| Marcação no quadro 14 | Antes (px) | Revisada (px) |
| --- | --- | --- |
| Centro da bola | (275, 423) | (278, 426) |
| p32, base próxima | (277.71, 476.29) | (277, 478) |
| p33, base distante | (319.41, 389.27) | (318, 390) |
| p34, topo próximo | (279.84, 389.45) | (271, 396) |
| p35, topo distante | (317.05, 316.30) | (314, 317) |

O usuário confirmou o quadro; **as coordenadas acima são uma revisão visual do
agente**, não novas coordenadas fornecidas pelo usuário. A incerteza de marcação
adotada é 2 px; a incerteza física tem piso de 0.2 m e propagação pelo Jacobiano da
homografia, resultando em **0.290 m**. São hipóteses de incerteza, não erros medidos
com referência independente.

Após a revisão e o ajuste da câmera:

- Resíduos de p32/p33/p34/p35: **2.967 / 1.588 / 0.572 / 2.051 px**.
- RMS da câmera sobre os pontos participantes, quadro 14: **3.285 px**.
- Entrada pela homografia local: `(-52.450, -3.022, 1.320) m`.
- Diferença câmera global/plano local: **0.024 m**.
- Bola ajustada no quadro 14: `(-52.574, -3.091, 1.315) m`.
- Resíduo da trajetória na entrada: **0.142 m**; diferença para o plano: **0.124 m**.
- Cruzamento contínuo do ajuste: **13.794 quadros**, que corresponde ao quadro 14.
- Reprojeção da bola ajustada no quadro 14: **4.713 px**.

O ajuste não foi deslocado ou recortado artificialmente para coincidir exatamente
com a baliza. O centro observado e o ponto ajustado continuam separados e auditáveis.
Os demais quadros têm seus próprios resíduos na tabela de calibração.

## Implementação

- `freekiki_arrival.py`: valida e persiste uma chegada observada; homografia dos
  quatro cantos kiki49 do mesmo quadro; propagação da incerteza; ajuste pinhole
  com pontos válidos do campo e os cantos confirmados; auditoria de participação
  e resíduos por ponto/quadro. Nenhuma coordenada física do kiki49 foi alterada.
- O ajuste usa 1 px para os cantos e 3 px para os pontos de campo, normalizados
  para largura 1920. O quadro confirmado exige máximo dos cantos ≤3 px e RMS dos
  participantes ≤8 px. Caso contrário, guarda os resíduos e interrompe antes de
  substituir as DLTs. A câmera anterior fica em `before_goal_refit/` do novo detect.
- `auto_flight`: entrada marcada vira uma âncora física com sua incerteza. A bola
  observada substitui o pixel daquela amostra no ajuste. Quadros posteriores à
  entrada/contato são excluídos do voo livre e ficam ausentes nos exports físicos.
- Sem chegada reconhecida, `arrival_status=undetermined`. Uma âncora de mão exige
  `type=keeper_hand`, XYZ e `confirmed=true`. Marcar entrada nunca cria âncora no goleiro.
- A entrada tem chave própria `goal_entry`: o relatório não a interpreta como mão
  do goleiro. Proveniência manual, cantos, pixel, incerteza e resíduos são persistidos.
- GUI: **Open run and correct → Mark arrival (reference)**. Escolher quadro, revisar
  bola/cantos, clicar no preview ou editar coordenadas, salvar e executar **Re-run
  analysis**. Também disponível em **Observed arrival…** junto ao campo de âncoras.
- CLI: `kick-analysis --resume RUN --arrival-observation FILE.json`; omitir o flag
  reutiliza a observação armazenada; `none` limpa a escolha. O fluxo de Rebuild ball
  aceita `--ball-anchors '{"arrival": {...}}'`, com diagnóstico da câmera existente.
- Cantos diferentes invalidam a câmera e o SAM. Mudar só o pixel da bola refaz a
  bola e derivados, reutilizando detecção e SAM. Backups de marcações ficam em
  `corrections/`; resultados substituídos em `superseded/`.
- Topview agora mantém amostras físicas finitas que cruzam a linha. `on_field=False`
  indica limites do campo; não significa amostra ausente. Isso evita perder a entrada
  visível no CSV e no C3D.
- Metadados/README/índices/ajudas atualizados para 02 November 2026, v0.4.14.
  Banner principal e instalação não mudaram; instaladores não precisaram de alteração.

## Isolamento, resultados e evidências

Original somente lido:

`/home/preto/data/Lance_decifrado/pal_vas01nov/chute/c2_chute_16mm_vailacut_20261101_195320`

Base de evidências local, ignorada pelo Git:

`/home/preto/data/vaila/.local/validation/freekiki_goal_entry_20261101/`

Run separado:

`isolated/kick_analysis_20261101_195640/`

Relatório português:

`isolated/kick_analysis_20261101_195640/c2_chute_16mm_frame_6_to_20_3d_20261101_195640/report_tv/pt/relatorio_bola.html`

Há versão inglesa, PDFs, animações, topview, CSV e C3Ds na mesma pasta de resultados.

Evidências principais na base:

- `original_manifest.json`: inventário dos 673 arquivos originais (SHA-256, tamanho,
  mtime); `acceptance_outputs.json`: nenhum desses arquivos mudou.
- `review_provenance.json`, `contact_sheet.jpg`, `last_frame_raw.png`,
  `review_*.png`: origem da revisão; `entry_reprojection.png`: marcações e reprojeção.
- `before_goal_diagnostic.json`, `goal_calibration.json`,
  `goal_calibration_residuals.csv`: antes/depois, pontos usados e resíduos.
- `real_run.log`: primeira tentativa, CUDA indisponível no sandbox.
  `real_camera_rebuild.log`: recálculo concluído fora do sandbox, com SAM 3D Body.
  `real_run_cuda.log`: execução final, com câmera e SAM reutilizados e bola recalculada.
- `work_before_ball_only.json` e `acceptance_outputs.json`: os **171 arquivos de
  work/** mantiveram SHA-256, tamanho e mtime após o recálculo apenas da bola.
- `arrival_dialog.png`, `gui_check.json`, `main_window_width.json`: GUI real no
  display X11 `:10`, carregamento do quadro 14 e Save. A captura inicial em `:0`
  saiu preta; a captura entregue foi refeita em `:10`.
- `gui_click_check.json`: evento real do Canvas em uma observação de teste separada;
  pixel alterado para `(281.27, 425.72)`, salvo e reaberto sem mudança. Não altera a
  marcação final. O teste invoca callbacks Tk; não simula mouse físico via XTEST.
- `acceptance_outputs.json`: CSV e C3Ds da bola e campo/bola concordam em todos os
  quadros, máximo absoluto **0.000001885 m** (float32 do C3D). Topview lê as mesmas
  15 amostras e o pixel amarelo final está na posição calculada da amostra 14.
- Fontes dos drivers de validação em `*_source.txt`.

## Verificação automatizada

- Testes focados finais: **174 passed em 65.64 s** (`focused_final.log`). Incluem
  entrada próxima à trave, chegada antes do último quadro, exclusão da rede, defesa
  confirmada, chegada indeterminada, validação de JSON, falha de calibração sem
  substituir DLT, persistência e reutilização/invalidação do SAM.
- Ruff check/format, Ty nos módulos editados e `git diff --check`: passaram.
- Primeira suíte completa: **1 failed, 2692 passed, 49 skipped**. A regressão foi a
  largura da janela (1027 px). O botão foi movido para a linha das âncoras; a GUI
  real passou a pedir **897 px**, dentro do limite de 900 px. Logs anteriores
  preservados em `full_suite_layout_failure.log` e `full_suite_trace_layout_failure.log`.
- Execução intermediária após corrigir o layout: **2693 passed, 49 skipped** em
  357.99 s (`full_suite.log`). A regressão adicional de segmento estático inicialmente
  usava um pixel de chegada diferente dos demais pixels; isso criava movimento no
  próprio teste. O cenário foi corrigido para manter a bola realmente estática.
  A tentativa está preservada em `full_suite_fixture_failure.log`.
- Resultado da suíte completa após a última alteração: **2694 passed, 49 skipped, 49 warnings em 322.06 s, exit=0**
  (`full_suite_final.log`).

Comando final (fora do sandbox, sem filtrar testes):

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  .venv/bin/python -u /tmp/freekiki_goal_full_suite.py
```

O driver usa `DISPLAY=:0` e `pytest.main(['tests/', '-q'])`, registra cada teste em
`full_suite_trace.log` e está preservado como `full_suite_source.txt`. Skips e avisos
não equivalem à validação dos cenários pulados.
