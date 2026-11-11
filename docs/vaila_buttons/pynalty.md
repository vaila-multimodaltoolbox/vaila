# Pynalty

Frame B → **Pynalty** (`B_r5_c5`), `vaila/pynalty.py`.

**PT:** Ao abrir, o FPS detectado automaticamente é aceito (mesma detecção
via ffprobe do `numberframes.py`) — pressione **F** para corrigir.
A interface opera com **4 Macro-Etapas** (Chute, Gol/Calibração, Resposta,
Salvar): confirme o contato pé-bola e marque bola/goleiro (`1`), no frame da
chegada/defesa (em geral o **último**) calibre o gol (ou aceite o TOML),
marque bola + mão do goleiro e o resultado (`2`), marque o início de movimento
do goleiro (`3`), e **Salvar (S)**. Métricas em tempo real aparecem no painel
lateral. Trajetória IA, pose e medidas permanecem opcionais e nunca bloqueiam.
Um **banner grande sobre o vídeo** mostra a ação atual. **PT / EN** ou **F2**
troca o idioma; **C** refaz a calibração; **E** edita um frame.

**EN:** On open, the auto-detected FPS is accepted (same ffprobe-based
detection as `numberframes.py`) — press **F** to correct it.
The interface uses **4 macro steps** (Kick, Goal/Calibration, Response,
Save): confirm foot-ball contact and mark ball/keeper (`1`); on the
arrival/save frame (usually the **last**) calibrate the goal (or accept TOML),
mark ball + keeper hand and the outcome (`2`); mark keeper first movement
(`3`); then **Save (S)**. Live metrics appear in the sidebar. AI trajectory,
pose and body measures stay optional and never block. A **large on-video
banner** shows the current action. **PT / EN** or **F2** changes language;
**C** recalibrates; **E** edits a frame.

`pynalty_calibration.toml` beside the video is previewed for reuse on the
goal step. Confirm matching framing/zoom/camera position. Explicit calibration
takes priority over session calibration, then directory discovery. Calibration
saves separately; failed writes keep it in session and offer another
destination. Current and legacy sessions resume at the first incomplete phase.

```bash
uv run vaila/pynalty.py -i video.mp4 -o out_dir --ui-lang pt
uv run vaila/pynalty.py -i video.mp4 --calibration /path/pynalty_calibration.toml --ui-lang en
uv run vaila/pynalty.py -i video.mp4 -c data.toml --report-only --lang both
```

`--ui-lang` controls the interface; `--lang` controls reports.
`--report-only` requires input and config, opens no dialogs and does not
automatically discover calibration.

Outputs: reloadable `data.toml`, metric and summary CSVs, EN/PT HTML reports,
snapshots, optional trajectory/pose outputs and an appendable penalty database.
Scientific calculations remain in `pynalty_analysis.py`, vision in
`pynalty_vision.py`, reporting in `pynalty_report.py`.

[Full bilingual help](../../vaila/help/pynalty_help.md)

Version: 0.4.16 · Updated: 11 November 2026 · AGPLv3.0
