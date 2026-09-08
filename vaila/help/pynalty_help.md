# Pynalty — guia / guide

**Version:** 0.3.130

**Updated:** 08 September 2026

**Projeto / Project:** vailá Multimodal Toolbox

## Português

Pynalty analisa o chute e a reação do goleiro a partir de vídeo. A interface
começa em português; o botão **PT / EN** ou **F2** troca o idioma durante a
sessão sem perder marcações. `--lang` controla somente os relatórios.

### Fluxo guiado

0. **Confirmar o FPS.** Ao abrir a janela, uma caixa mostra o FPS detectado
   automaticamente no vídeo (mesma detecção precisa via ffprobe usada em
   `numberframes.py`, com respaldo em OpenCV). Pressione **Enter** para aceitar
   ou digite o valor correto antes de calibrar — é essa taxa que converte
   frames em tempo e velocidade em todo o restante da análise. Pode ser
   ajustada depois a qualquer momento com **F**.
1. **Calibrar.** Escolha um frame com o gol visível e pressione **Enter**.
   O vídeo fica parado. Clique nos cantos, conforme aparecem na imagem:
   **1 inferior esquerdo → 2 superior esquerdo → 3 superior direito →
   4 inferior direito**. O esquema destaca o próximo canto. O botão direito
   desfaz. Confira a prévia e clique em **Confirmar e salvar**.
2. **Confirmar os três frames**, nesta ordem: primeiro movimento claro de
   preparação/mergulho do goleiro; contato do pé com a bola; bola na linha do
   gol. Navegue com as setas ou a linha do tempo e pressione **Enter** em
   cada evento. Enter pausa e avança; cliques na imagem não marcam nada.
   A chegada deve ser posterior ao chute. O goleiro pode começar antes dele.
3. **Marcar os pontos.** O vídeo retorna ao frame do chute: clique no centro
   da bola, depois no centro do goleiro. Ele avança ao frame da chegada e
   solicita os mesmos dois pontos. A navegação temporal fica travada; zoom e
   deslocamento da imagem continuam disponíveis. **Editar frame / E** volta
   à escolha temporal. Alterar o frame elimina seus pontos e os resultados,
   trajetória e pose dependentes. **Voltar** permite revisar etapas anteriores.
4. **Resultado.** Escolha **Gol / Defesa / Fora / Trave**, ou **G / D / M / W**.
5. **Opcionais.** Trajetória: **A** para YOLO ou cliques manuais por frame.
   Pose: arraste a caixa do cobrador, depois do goleiro, e use **P** para
   MediaPipe. Medidas corporais: **B**. Cada fase oferece **Pular / Continuar**;
   sem medidas, os cálculos usam valores padrão (estatura do goleiro: 1,88 m).
6. **Revisão.** Confira os frames, avisos e medidas, e clique em
   **Salvar resultados**. A interface permanece na revisão após salvar.

### Calibração reutilizável

O módulo procura `pynalty_calibration.toml` na pasta do vídeo. Se encontrar,
mostra os quatro cantos sobre o vídeo: **Usar calibração / Enter** confirma,
**Refazer calibração / C** inicia outra. Confira **enquadramento, zoom e
posição da câmera** antes de reutilizar. A mesma resolução não garante que a
câmera permaneceu na mesma posição. O frame de origem é apenas referência;
não se torna um evento do vídeo atual.

O TOML guarda `format_version = 1`, `width`, `height`, quatro `points` em
pixels na ordem acima, `geometry` (largura, altura, distância do pênalti e
raio da bola em metros), `source_video` e `source_frame` (índice a partir de
zero). São exigidos resolução compatível, pontos finitos dentro da imagem,
quadrilátero convexo e transformação DLT calculável. Um arquivo incompatível
leva à nova calibração com uma explicação na tela.

**C** permite recalibrar durante a sessão. Os cliques ficam em uma cópia
provisória; **Cancelar / Esc** mantém a calibração anterior. Confirmar salva a
substituição e recalcula as medidas, preservando os eventos. A calibração é
salva independentemente dos resultados. Se a gravação falhar, ela permanece
na sessão e um diálogo oferece outro destino.

Precedência: **`--calibration` explícita → calibração da sessão → descoberta
na pasta**. Sessões atuais e legadas, inclusive listas sem `key`, continuam
aceitas e retomam a primeira fase incompleta. `--report-only` exige `-i` e
`-c`, não abre diálogos e não procura calibração automaticamente.

### Controles

| Controle | Ação |
| --- | --- |
| ← / →; ↑ / ↓ | ±1; ±10 frames |
| Espaço; Home / End | Reproduzir/pausar; primeiro/último frame |
| Linha do tempo | Clique/arraste para navegar e pausar; exibe progresso e os três eventos confirmados |
| Roda / + / − | Zoom |
| Arrastar com botão do meio; 0 | Mover imagem; ajustar à janela |
| Clique esquerdo / direito | Próximo ponto / desfazer último ponto |
| Enter | Confirmar a ação indicada na tela |
| Tab / Shift+Tab | Avançar / voltar |
| C / E | Recalibrar / editar frame dos pontos |
| G / D / M / W | Gol / defesa / fora / trave na fase de resultado |
| A / P / B | Trajetória / pose / medidas, nas respectivas fases |
| S / L / H | Salvar na revisão / carregar sessão / ajuda |
| F / F2 | Ajustar fps / trocar PT–EN |
| Esc | Cancelar recalibração ou sair |

## English

Pynalty analyses shot kinematics and goalkeeper reaction from video. The
interface defaults to Portuguese. **PT / EN** or **F2** changes the interface
language without losing marks; `--lang` selects report languages separately.

### Guided workflow

0. **Confirm FPS.** When the window opens, a box shows the video's
   auto-detected FPS (same precise ffprobe-based detection as
   `numberframes.py`, with an OpenCV fallback). Press **Enter** to accept it,
   or type the correct value before calibrating — this rate converts frames
   into time and velocity throughout the rest of the analysis. It can still be
   changed anytime afterwards with **F**.
1. **Calibrate.** Choose a frame with the goal visible and press **Enter** to
   pause. Click **bottom left → top left → top right → bottom right**, as
   seen in the image. The numbered diagram highlights the next corner.
   Right click undoes. Check the preview, then **Confirm and save**.
2. **Confirm all three frames**: first clear keeper preparation/dive movement;
   foot–ball contact; ball reaching the goal line. Navigate and press **Enter**
   for each. Enter pauses and advances; image clicks do nothing in this phase.
   Arrival must follow contact; keeper movement may precede contact.
3. **Click key-frame points.** The video returns to contact: click ball centre,
   then keeper centre. It advances to arrival and asks for both points again.
   Time navigation is locked, while zoom and pan remain available.
   **Edit frame / E** returns to frame selection. Changing a frame clears its
   old points and dependent results, trajectory and pose. **Back** revisits
   earlier phases.
4. **Outcome.** Choose **Goal / Save / Miss / Woodwork**, or **G / D / M / W**.
5. **Optional steps.** Trajectory: **A** runs YOLO, or click the ball frame by
   frame. Pose: drag kicker then keeper boxes and press **P** for MediaPipe.
   Body measurements: **B**. Each step has **Skip / Continue**. Skipped body
   measures use defaults (keeper stature 1.88 m).
6. **Review and save.** Check frames, warnings and measures, then **Save
   results**. Saving leaves the interface at review.

### Reusable calibration

The app looks for `pynalty_calibration.toml` beside the video and previews its
corners. **Use calibration / Enter** accepts; **Recalibrate / C** starts over.
Check framing, zoom and camera position: matching resolution alone does not
establish matching camera geometry. The source frame is provenance only,
never a new event frame.

The versioned TOML stores source resolution, four finite in-image pixel
corners in the specified order, goal geometry in metres and source video/frame
(zero-based). Validation requires matching resolution, a convex quadrilateral
and a computable DLT transform. Invalid files lead to fresh calibration with
an explanation.

**C** starts a temporary replacement during the session. **Cancel / Esc**
retains the previous calibration. Confirming replaces it and recomputes
results while preserving events. Calibration saves independently of the
results package. Write failures retain it in memory and offer another path.

Precedence: **explicit `--calibration` → saved session calibration → directory
discovery**. Current and legacy sessions, including unkeyed event lists in
the original order, resume at the first incomplete phase. `--report-only`
requires `-i` and `-c`, opens no dialogs and performs no automatic discovery.

Controls: arrows ±1/±10 frames; Space play/pause; Home/End first/last; click/drag
the progress and event timeline to seek and pause; wheel or +/− zoom; middle drag pan;
0 fit; left/right click
mark/undo; Enter confirm; Tab/Shift+Tab next/back; C recalibrate; E edit frame;
A/P/B optional trajectory/pose/body; S save at review; L load; H help; F fps;
F2 language; Esc cancel recalibration or quit.

## GUI / CLI

GUI: **Frame B → Pynalty**. Every launch prints a copyable equivalent command
with the `>>` prefix.

```bash
uv run vaila/pynalty.py -i video.mp4 --ui-lang pt
uv run vaila/pynalty.py -i next_video.mp4 --calibration /path/pynalty_calibration.toml --ui-lang en
uv run vaila/pynalty.py -i video.mp4 -c data.toml -o out_dir --report-only --lang both
```

Other existing options: `--database`, `--auto-ball`, `--pose`, `--no-wizard`
(compatibility), `--penalty-distance`, `--goal-width`, `--goal-height`.

## Saídas / Outputs

Saved under `<video_stem>_results/` (under `-o` when supplied): `data.toml`,
`results.csv`, `pynalty_summary.csv`, `pynalty_database.csv`, English/Portuguese
HTML reports, event snapshots, ball-path CSVs/videos and pose CSVs/overlays
when available. The reusable calibration TOML is separate.

DLT describes the goal plane; full-flight speed uses the existing model,
not raw pixel displacement. / A DLT descreve o plano do gol; a velocidade
de voo usa o modelo existente, não diferenças brutas de pixels.

[Help index](index.html) · [Button documentation](../../docs/vaila_buttons/pynalty.md)
