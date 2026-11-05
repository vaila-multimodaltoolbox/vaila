# FreeKiki: corrigir calibração e executar novamente

Data: 01 November 2026. Versão: 0.4.14. Sem commit.

## Correção reproduzida e aplicada

O F9 do getpixelvideo oferecia mudar a sessão de revisão para uma pasta de dataset.
`kick_review_session()` reabria o `session.json` que permanecia em `corrections/`, com os
pontos anteriores. O CSV corrigido já existia, mas a reabertura mostrava o rascunho antigo.

A sessão de correção de um kick run agora declara `dataset_folder` como sua própria pasta
em `corrections/<video>/`. O F9 salva ali e não oferece mudar essa sessão de lugar. A mesma
propriedade é preenchida ao abrir uma sessão existente. Os dois casos têm regressão em
`tests/test_freekiki_kick_correct.py::test_f9_keeps_the_session_in_the_run_and_reopens_the_saved_edit`.
As ajudas Markdown e HTML documentam esse comportamento.

Os cabeçalhos dos arquivos Python editados, README e índices de ajuda já estavam na data
de hoje e na versão global 0.4.14. A correção não altera banner, instalação ou lançamento;
não exige mudança nos instaladores.

## Dados e isolamento

Original, somente lido:

`/home/preto/data/Lance_decifrado/pal_vas01nov/chute/c2_chute_16mm_vailacut_20261101_195320/kick_analysis_20261101_195640`

Evidências locais, não versionadas:

`/home/preto/data/vaila/.local/validation/freekiki_correction_20261101/`

Run validado:

`/home/preto/data/vaila/.local/validation/freekiki_correction_20261101/isolated/kick_analysis_20261101_195640`

Foram copiados o run, o MP4 e o CSV da bola. Caminhos textuais dentro da cópia foram
substituídos pelos caminhos isolados. `isolation.json` lista os arquivos ajustados;
`original_manifest.json` registra SHA-256, tamanho e mtime dos 301 arquivos originais.
`original_check.log` registra a comparação de integridade. A cópia usada para reproduzir
a falha ficou em `before_fix/`.

Na comparação final, os 301 arquivos preexistentes continuaram idênticos, sem remoção.
Foram encontrados arquivos adicionais de correção na pasta original, com gravações
às 21:38–21:39, posteriores às interações desta validação isolada. O p0 dessa outra sessão
é `(305.45, 145.45)`, distinto do deslocamento de teste `(293, 158)`. A origem das novas
gravações não foi atribuída; os arquivos foram mantidos e estão listados em
`original_added_files.json`. Portanto, a prova de preservação vale para os 301 arquivos
do inventário inicial, não para afirmar ausência de atividade concorrente na pasta.

## Desktop, resultados e auditoria

O FreeKiki e o getpixelvideo rodaram na sessão X11 real `:10`. A automação forneceu o run
isolado ao seletor de pastas e redirecionou a preferência de workspace para um arquivo de
teste. Os botões, os subprocessos CLI, o editor e o processamento não foram simulados.
Mouse e teclado foram enviados via XTEST. Os fontes dos drivers ficaram nas evidências
como `*_source.txt`.

1. **Open run and correct → Correct field keypoints.** Os rascunhos da primeira abertura
   correspondem ao `field_kps_getpixelvideo.csv` usado pela análise. No quadro 0, p0 foi
   selecionado com botão direito e movido com botão esquerdo de `(287.41, 154.34)` para
   `(293, 158)`. O F9 salvou em modo **Full**: os demais pontos mantêm sua proveniência
   de previsão, sem inventar marcações para os quatro pontos ausentes que impediam Lite.
   O CSV e a sessão foram conferidos; fechar e reabrir pelo FreeKiki mostrou `(293, 158)`.
2. **Re-run analysis após o campo:** terminou com `exit=0`. O novo CSV de pontos aceitos
   contém `(293, 158)`. A DLT mudou; a focal do primeiro quadro passou de aproximadamente
   2933.4192 para 2972.9370 pixels. O SAM 3D Body recebeu o novo `cam_int.npy` e `--fresh`.
   O conteúdo das poses em coordenadas do campo mudou, e o relatório foi refeito.
   A detecção e o SAM anteriores estão em `superseded/20261101_212417/<video>/`.
3. **Correct ball → Save → Quick Save:** no quadro 0, a bola foi movida de `(500, 569)`
   para `(506, 573)`. O arquivo `<video>_markers.csv` e o `ball.csv` recalculado contêm
   `(506, 573)`. O reprocessamento terminou com `exit=0` e refez a bola e os derivados.
   Os 167 arquivos de `work/`, incluindo detecção, intrínsecos e SAM, mantiveram SHA-256,
   tamanho e mtime. As etapas detect/3d mantiveram seus registros. O CSV derivado de poses
   foi reescrito com conteúdo idêntico, sem executar novamente o SAM.
4. **Concorrência e responsividade:** clicar Re-run com o editor ainda aberto mostrou
   “A task is still running.”; nenhum segundo pipeline iniciou. Durante o processamento
   do campo, foram observadas 97 atualizações do loop Tk em aproximadamente 10 segundos
   (`responsiveness_field.json`).
5. **CLI:** os comandos usados pela GUI estão em `gui_desktop.log`, com o prefixo `>>`.
   Uma chamada direta de `kick-analysis --resume RUN` terminou com `exit=0` e reutilizou
   as oito etapas (`cli_resume.log`). Backups dos CSVs de campo e bola estão em
   `corrections/<video>/`.

Provas por arquivo: `session_initial.json`, `session_saved.json`, `session_reopened.json`,
`field_saved.csv`, `ball_saved.csv`, `after_field.json`, `after_ball.json`,
`ball_reuse_check.log` e `post_ball_check.json`.

Capturas na pasta de evidências:

- `01_open_run.png`: diálogo de correção com a pasta isolada.
- `02_concurrent_guard.png`: bloqueio da execução simultânea.
- `03_field_before.png`, `06_reopen_before_fix.png`: comportamento anterior.
- `07_field_saved_fixed.png`, `08_field_reopened_fixed.png`: gravação e reabertura corrigidas.
- `09_rerun_processing.png`: processamento na GUI.
- `10_ball_before.png`, `13_ball_corrected_saved.png`: correção manual da bola e Save.
- `14_rerun_complete.png`: conclusão do processamento na GUI.

Relatório recalculado:

`isolated/kick_analysis_20261101_195640/c2_chute_16mm_frame_6_to_20_3d_20261101_195640/report_tv/en/ball_report.html`

Esse resultado é uma cópia de validação com deslocamentos de teste, não uma revisão
científica do run original. A validação real abrange um run monocular; a seleção da outra
câmera e o fluxo binocular continuam cobertos pelos testes automatizados pertinentes.

## Verificações automatizadas

- Antes da correção: 106 testes focados passaram (kick-correct, kick pipeline, keypoints
  manuais e integração FreeKiki/getpixelvideo).
- Após a correção: 9 testes de kick-correct passaram, incluindo as duas regressões novas.
- Ruff check e format check dos dois arquivos Python: passaram. `git diff --check`: passou.
- A execução completa inicial ficou na inicialização de áudio do MediaPipe/PortAudio
  dentro do sandbox (`full_suite_sandbox_audio_trace.log`). Fora do sandbox, a coleta
  exigiu um display para módulos que usam TkAgg (`full_suite_headless.log`).
- A primeira execução com display saiu com código 139 durante os testes de pênalti,
  enquanto o despejo periódico de pilhas escrevia o diagnóstico. A causa nativa não foi
  determinada. O grupo de pênalti passou isoladamente: **21 passed em 66.06 s**
  (`penalty_check.log`). Nenhum código de física foi alterado nesta validação. A tentativa
  interrompida foi preservada em `full_suite_crash.log` e `full_suite_crash_trace.log`.
- A execução final, sem despejo periódico, usa o desktop secundário `:0`,
  `QT_QPA_PLATFORM=offscreen`, `MPLBACKEND=Agg` e `SDL_VIDEODRIVER=dummy`.
  Log: `full_suite.log`; progresso por teste: `full_suite_trace.log`.
  **Resultado final: 2673 passed, 49 skipped, 49 warnings em 380.56 s (6 min 20 s), exit=0.**
  As regressões novas estão incluídas nessa execução. Os testes pulados não são validação
  de seus cenários. Os avisos estão integralmente registrados no log.

Comando da execução final, a partir da raiz do repositório (fora do sandbox):

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  .venv/bin/python -u /tmp/freekiki_run_full_suite.py
```

O driver define `DISPLAY=:0`, registra o progresso e chama `pytest.main(['tests/', '-q'])`,
sem filtrar testes. Seu fonte final está em `freekiki_run_full_suite_source.txt`.
`acceptance_outputs.json` registra as verificações dos resultados. A comparação final de
integridade confirmou novamente os 301 arquivos originais, incluindo hashes e datas.
