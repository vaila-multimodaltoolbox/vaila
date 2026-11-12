# Pynalty — guia / guide

**Version:** 0.4.17

**Updated:** 12 November 2026

**Projeto / Project:** *vailá* Multimodal Toolbox

## Português

Pynalty analisa o chute e a reação do goleiro a partir de vídeo. A interface
usa **4 Macro-Etapas** (Chute → Gol/Calibração → Resposta → Salvar), um
**banner grande sobre o vídeo** com a ação atual, métricas em tempo real e
salvamento direto sem bloqueio por etapas opcionais.
A interface começa em português; o botão **PT / EN** ou **F2** troca o idioma
durante a sessão sem perder marcações. `--lang` controla somente os relatórios.

### Fluxo guiado em 4 Macro-Etapas

0. **FPS.** Ao abrir, o FPS detectado automaticamente (ffprobe como em
   `numberframes.py`, com respaldo OpenCV) é aceito. Um aviso mostra o valor;
   pressione **F** a qualquer momento para corrigir — essa taxa converte
   frames em tempo e velocidade.
1. **[1. Chute ⚽] Contato e pontos.** Navegue até o contato pé-bola e confirme
   com **Enter** (ou tecla **1**). Clique no centro da **bola** e no centro do
   **goleiro** no mesmo frame.
2. **[2. Gol 🥅] Chegada = calibração.** Em geral use o **último frame** (bola
   na linha ou defesa). Confirme com **Enter** (ou tecla **2**). No **mesmo
   frame**: marque os 4 cantos do gol (**inferior esquerdo → superior esquerdo
   → superior direito → inferior direito**) ou aceite o TOML com **Usar
   calibração / Enter**; clique na **bola** e na **mão do goleiro**; escolha o
   resultado **Gol (G)**, **Defesa (D)**, **Fora (M)** ou **Trave (W)**.
3. **[3. Resposta 🧤] Tempo de reação.** Navegue até o primeiro movimento claro
   de preparação/mergulho do goleiro e confirme com **Enter** (ou tecla **3**).
4. **[4. Salvar 💾]** Com os passos essenciais feitos, o painel mostra km/h,
   tempo de voo e reação, e **Salvar resultados (S)** fica em destaque.
   - *Opcionais (sem bloqueio):* Trajetória IA (**A**), Pose MediaPipe (**P**)
     e Medidas corporais (**B**).

> [!TIP]
> Clique em qualquer etapa no painel esquerdo para revisar. Marcadores
> coloridos na timeline pulam direto ao frame do evento. O banner amarelo
> sobre o vídeo sempre diz o próximo clique.

### Calibração reutilizável

O módulo procura `pynalty_calibration.toml` na pasta do vídeo. Na etapa Gol,
**Usar calibração / Enter** confirma; **Refazer calibração / C** inicia outra.
Confira **enquadramento, zoom e posição da câmera** antes de reutilizar.

O TOML guarda `format_version = 1`, `width`, `height`, quatro `points`,
`geometry`, `source_video` e `source_frame`. Precedência: **`--calibration`
explícita → sessão → pasta**. `--report-only` exige `-i` e `-c`, sem diálogos
nem descoberta automática.

### Controles

| Controle | Ação |
| --- | --- |
| 1 / 2 / 3 | Chute / Gol / Resposta |
| S | Salvar resultados |
| Banner / Enter | Confirma a ação atual |
| Painel lateral | Clique nas 4 macro-etapas |
| Marcadores da timeline | Pular para o frame do evento |
| ← / →; ↑ / ↓ | ±1; ±10 frames |
| Espaço; Home / End | Reproduzir/pausar; primeiro/último |
| C / E | Recalibrar / editar frame |
| G / D / M / W | Gol / defesa / fora / trave |
| A / P / B | Trajetória / pose / medidas (opcionais) |
| L / H | Carregar sessão / ajuda |
| F / F2 | Ajustar fps / trocar PT–EN |
| Esc | Cancelar recalibração ou sair |

## English

Pynalty analyses shot kinematics and goalkeeper reaction from video. The
interface uses **4 macro steps** (Kick → Goal/Calibration → Response → Save),
a **large on-video action banner**, live sidebar metrics, and one-click saving
without mandatory optional steps.
Defaults to Portuguese. **PT / EN** or **F2** changes language; `--lang`
selects report languages separately.

### Guided 4-Step Macro Workflow

0. **FPS.** On open, the auto-detected FPS (ffprobe as in `numberframes.py`,
   OpenCV fallback) is accepted. A flash shows the value; press **F** anytime
   to correct it.
1. **[1. Kick ⚽] Contact & points.** Scrub to foot–ball contact, confirm with
   **Enter** (or **1**), then click **ball** and **keeper** centres.
2. **[2. Goal 🥅] Arrival = calibration.** Usually the **last frame** (ball on
   the line or save). Confirm with **Enter** (or **2**). On the **same frame**:
   mark 4 goal corners (**BL → TL → TR → BR**) or accept TOML; click **ball**
   and keeper **hand**; choose **Goal (G)**, **Save (D)**, **Miss (M)** or
   **Woodwork (W)**.
3. **[3. Response 🧤] Reaction time.** Confirm the keeper's first clear
   preparation/dive movement with **Enter** (or **3**).
4. **[4. Save 💾]** When core steps are done, press **S**. Optional **A** / **P**
   / **B** never block saving.

### Reusable calibration

Looks for `pynalty_calibration.toml` beside the video and previews it on the
Goal step. **C** recalibrates. Precedence: explicit `--calibration` → session
→ directory. `--report-only` needs `-i` and `-c`, no dialogs, no discovery.

## GUI / CLI

GUI: **Frame B → Pynalty**. Every launch prints a copyable equivalent command
with the `>>` prefix.

```bash
uv run vaila/pynalty.py -i video.mp4 --ui-lang pt
uv run vaila/pynalty.py -i next_video.mp4 --calibration /path/pynalty_calibration.toml --ui-lang en
uv run vaila/pynalty.py -i video.mp4 -c data.toml -o out_dir --report-only --lang both
```

## Saídas / Outputs

Saved under `<video_stem>_results/`: `data.toml`, `results.csv`,
`pynalty_summary.csv`, `pynalty_database.csv`, EN/PT HTML reports, snapshots,
optional ball-path and pose artefacts. Calibration TOML is separate.

[Help index](index.html) · [Button documentation](../../docs/vaila_buttons/pynalty.md)
