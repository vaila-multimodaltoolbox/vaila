# Pynalty

Frame B → **Pynalty** (`B_r5_c5`), `vaila/pynalty.py`.

**PT:** Ao abrir, confirme o FPS detectado automaticamente (mesma detecção
via ffprobe do `numberframes.py`) — necessário para velocidade e tempo.
Calibre o gol, confirme os três frames (movimento do goleiro, chute,
chegada), marque bola/goleiro nos frames pausados, escolha o resultado e
avance pelos opcionais até **Salvar resultados**. O botão **PT / EN** troca
o idioma durante a sessão. **C** refaz a calibração; **E** edita um frame;
**F** ajusta o FPS a qualquer momento.
O player usa leitura sequencial durante a reprodução; clique ou arraste a
timeline para pausar e navegar pelo progresso e pelos três eventos.

**EN:** On open, confirm the auto-detected FPS (same ffprobe-based detection
as `numberframes.py`) — required for velocity and time. Calibrate the goal,
confirm all three event frames, click ball/keeper points on paused contact
and arrival frames, select the outcome and skip or complete optional
trajectory, pose and body measures before saving at review. **PT / EN**
changes interface language; **C** recalibrates; **E** edits a frame; **F**
adjusts FPS anytime.
Playback uses sequential decoding; click or drag the timeline to pause and
seek through progress and the three event markers.

`pynalty_calibration.toml` beside the video is previewed for reuse. Confirm
matching framing/zoom/camera position. Explicit calibration takes priority
over session calibration, then directory discovery. Calibration saves
separately; failed writes keep it in session and offer another destination.
Current and legacy sessions resume at the first incomplete phase.

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

Version: 0.3.130 · Updated: 08 September 2026 · AGPLv3.0
