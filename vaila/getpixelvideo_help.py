"""
================================================================================
Pixel Coordinate Tool - help content (getpixelvideo_help.py)
================================================================================
*vailá* – Multimodal Toolbox
Authors: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
https://github.com/vaila-multimodaltoolbox/vaila
Date: 14 November 2026
Update: 24 November 2026
Version: 0.4.19
Python Version: 3.12.16

Description:
------------
What the **H** help screen of ``getpixelvideo.py`` and the hints on its toolbar
buttons say, in English and Portuguese. Data and pure functions only (no pygame):
the window draws them, the tests read them.

Every entry keeps the key, the mouse action and the toolbar button apart, so the
screen can show "which key" and "which button" in their own columns:

    HelpEntry(section, en, pt, keys=("Ctrl+I",), mouse="", button="VISUAL")

``now_steps(context, lang)`` gives the numbered "what to do now" steps for the mode
the user is in; ``TOOLBAR_HINTS`` gives the one-line hint (and shortcut key) of each
toolbar button. An entry must describe what the event handlers of
``play_video_with_controls`` really do: change both together.
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass

LANGS = ("en", "pt")

# Mouse actions are written in English in the entries and shown translated.
MOUSE_PT = {
    "Left click": "Clique esquerdo",
    "Right click": "Clique direito",
    "Click": "Clique",
    "Drag": "Arrastar",
    "Wheel": "Roda do mouse",
    "Middle drag": "Arrastar com o botão do meio",
    "Shift+Click": "Shift+Clique",
}


@dataclass(frozen=True)
class HelpEntry:
    """One line of the help: what a key, a mouse action or a toolbar button does."""

    section: str
    en: str
    pt: str
    keys: tuple[str, ...] = ()  # alternatives, each a combination such as "Ctrl+I"
    mouse: str = ""  # a key of MOUSE_PT
    button: str = ""  # the label on the toolbar

    def text(self, lang: str) -> str:
        return self.pt if lang == "pt" else self.en

    def mouse_label(self, lang: str) -> str:
        return MOUSE_PT.get(self.mouse, self.mouse) if lang == "pt" else self.mouse


@dataclass(frozen=True)
class HelpSection:
    id: str
    tab_en: str
    tab_pt: str
    en: str
    pt: str

    def title(self, lang: str) -> str:
        return self.pt if lang == "pt" else self.en

    def tab(self, lang: str) -> str:
        return self.tab_pt if lang == "pt" else self.tab_en


@dataclass(frozen=True)
class HelpContext:
    """What the user is doing when the help opens."""

    mode: str = "mark"  # mark | seq | 1-line | labeling | measure | calib | freekiki
    insert: bool = False
    ai_track: bool = False
    n_points: int = 0  # 1-line: points in the row
    next_point: int = 0  # 1-line: number the next click writes
    selected: int = -1  # 1-line: selected point, or -1
    ref3d_total: int = 0  # REF3D: points of the calibration object
    ref3d_next: str = ""  # REF3D: "p5 = pos3 · mark 1 m (0, 5, 1)", empty when all clicked


# Digit keys 1..9, 0 jump to these, in this order.
SECTIONS: tuple[HelpSection, ...] = (
    HelpSection("player", "Video", "Vídeo", "Video and frames", "Vídeo e quadros"),
    HelpSection(
        "edit", "Markers", "Pontos", "Placing and editing markers", "Marcar e editar pontos"
    ),
    HelpSection(
        "modes", "Modes", "Modos", "Marker modes and templates", "Modos de marcação e templates"
    ),
    HelpSection(
        "files", "Open/Save", "Abrir/Salvar", "Open, load and save", "Abrir, carregar e salvar"
    ),
    HelpSection("ai", "AI Track", "AI Track", "AI Track", "AI Track (rastreio automático)"),
    HelpSection(
        "timeline", "Timeline", "Linha", "Marker timeline", "Linha do tempo dos marcadores"
    ),
    HelpSection("measure", "Measure", "Medir", "CALIB and MEASURE", "CALIB e MEASURE (medidas)"),
    HelpSection(
        "labeling", "Boxes", "Caixas", "Bounding-box labeling", "Rotulagem com caixas (bbox)"
    ),
    HelpSection(
        "freekiki",
        "FreeKiki",
        "FreeKiki",
        "FreeKiki review (only in a FreeKiki session)",
        "Revisão FreeKiki (só em sessão FreeKiki)",
    ),
    HelpSection("general", "Help", "Ajuda", "Help", "Ajuda"),
)

_E = HelpEntry

_ENTRIES: tuple[HelpEntry, ...] = (
    # --- player ---------------------------------------------------------------------
    _E("player", "Play / pause the video.", "Reproduz / pausa o vídeo.", keys=("Space",)),
    _E(
        "player",
        "Next / previous frame (video paused). Hold to keep stepping.",
        "Próximo quadro / quadro anterior (vídeo pausado). Segure para continuar.",
        keys=("Right", "Left"),
    ),
    _E(
        "player",
        "Jump to the next / previous frame that has markers.",
        "Vai ao próximo / anterior quadro que tem marcadores.",
        keys=("Shift+Right", "Shift+Left"),
    ),
    _E(
        "player",
        "Jump 60 frames forward / back (video paused). Change the step in Cfg, Nav jump.",
        "Pula 60 quadros para frente / para trás (vídeo pausado). O passo muda em Cfg, Nav jump.",
        keys=("Up", "Down"),
    ),
    _E("player", "Zoom in / zoom out.", "Aproxima / afasta o zoom.", keys=("+", "-")),
    _E(
        "player",
        "Zoom in / out at the mouse pointer.",
        "Zoom no ponto onde está o mouse.",
        mouse="Wheel",
    ),
    _E(
        "player",
        "Move the zoomed image.",
        "Move a imagem ampliada.",
        mouse="Middle drag",
    ),
    _E(
        "player",
        "Double / halve the playback speed (16x down to 1/16x).",
        "Dobra / reduz à metade a velocidade (de 16x até 1/16x).",
        keys=("]", "["),
    ),
    _E(
        "player",
        "On the bar at the bottom: go straight to any frame.",
        "Na barra de baixo: vai direto a qualquer quadro.",
        mouse="Drag",
    ),
    _E(
        "player",
        "Type the video frame rate (Hz). Used for time, velocity and acceleration.",
        "Digita a taxa de quadros do vídeo (Hz). Usada em tempo, velocidade e aceleração.",
        keys=("I",),
        button="FPS … Hz",
    ),
    # --- edit -----------------------------------------------------------------------
    _E(
        "edit",
        "VISUAL = look only (how the window opens). INSERT = edit markers.",
        "VISUAL = só olhar (como a janela abre). INSERT = editar marcadores.",
        keys=("Ctrl+I",),
        button="VISUAL / INSERT",
    ),
    _E(
        "edit",
        "Place the selected marker, or move it, on this frame.",
        "Coloca o marcador selecionado, ou muda a posição dele, neste quadro.",
        mouse="Left click",
    ),
    _E(
        "edit",
        "Remove the selected marker on this frame. In 1-line mode: the selected point, "
        "or the last point when none is selected.",
        "Remove o marcador selecionado neste quadro. No modo 1-line: o ponto selecionado "
        "ou, sem seleção, o último ponto.",
        keys=("R", "Delete"),
        mouse="Right click",
    ),
    _E(
        "edit",
        "Select the next / previous marker.",
        "Seleciona o próximo marcador / o anterior.",
        keys=("TAB", "Shift+TAB"),
    ),
    _E(
        "edit",
        "Select a marker by its number.",
        "Seleciona um marcador pelo número.",
        keys=("Ctrl+G",),
        button="Go KP",
    ),
    _E(
        "edit",
        "Lock the selection on this marker (TAB and AI Track stay on it). Again to unlock.",
        "Trava a seleção neste marcador (TAB e AI Track ficam nele). De novo para destravar.",
        keys=("B",),
        button="Lock (B)",
    ),
    _E(
        "edit",
        "Remove ALL markers of this frame. In 1-line mode: the points clicked in this frame.",
        "Remove TODOS os marcadores deste quadro. No modo 1-line: os pontos clicados neste quadro.",
        keys=("D",),
    ),
    _E(
        "edit",
        "Add an empty marker (a point that is not visible keeps its number).",
        "Adiciona um marcador vazio (um ponto que não aparece mantém o número).",
        keys=("A",),
    ),
    _E(
        "edit",
        "Undo the last edit (up to 50 steps).",
        "Desfaz a última edição (até 50 passos).",
        keys=("Ctrl+Z",),
    ),
    _E(
        "edit",
        "Go back to the last Save, or to how the video opened.",
        "Volta ao último Save, ou a como o vídeo abriu.",
        button="Restore",
    ),
    _E(
        "edit",
        "Delete one marker, a list (0,3,7) or a range (1:10) over a span of frames.",
        "Apaga um marcador, uma lista (0,3,7) ou um intervalo (1:10) em vários quadros.",
        button="Del Range",
    ),
    _E(
        "edit",
        "Swap marker numbers (26 with 27, 1:10 with 11:20) over a span of frames.",
        "Troca números de marcadores (26 com 27, 1:10 com 11:20) em vários quadros.",
        button="Swap Range",
    ),
    _E(
        "edit",
        "Swap rules for the whole file: create, save and load them.",
        "Regras de troca para o arquivo inteiro: criar, salvar e carregar.",
        keys=("W",),
    ),
    _E(
        "edit",
        "Fill the gaps of one marker (Kalman/RTS prediction or straight line).",
        "Preenche as falhas de um marcador (previsão Kalman/RTS ou linha reta).",
        button="Gap Fill",
    ),
    _E(
        "edit",
        "Add the 33 MediaPipe pose points of the person in this frame.",
        "Adiciona os 33 pontos de pose do MediaPipe da pessoa neste quadro.",
        keys=("J",),
    ),
    # --- modes ----------------------------------------------------------------------
    _E(
        "modes",
        "Change the mode: Mark (one marker at a time), Seq, 1-line.",
        "Muda o modo: Mark (um marcador por vez), Seq, 1-line.",
        button="Mode: …",
    ),
    _E(
        "modes",
        "1-line mode on / off: one row of points p0..pN in click order, on any frames "
        "(the pixel file for DLT2D / DLT3D).",
        "Liga / desliga o modo 1-line: uma linha de pontos p0..pN na ordem dos cliques, em "
        "quaisquer quadros (o arquivo de pixels para DLT2D / DLT3D).",
        keys=("C",),
    ),
    _E(
        "modes",
        "Sequential mode on / off: each click moves on to the next marker.",
        "Liga / desliga o modo sequencial: cada clique passa para o próximo marcador.",
        keys=("O", "S"),
    ),
    _E(
        "modes",
        "Mark with the mouse position while the video plays.",
        "Marca com a posição do mouse enquanto o vídeo toca.",
        keys=("M",),
        button="MousePlay",
    ),
    _E(
        "modes",
        "Go to the next frame after each click.",
        "Vai para o próximo quadro depois de cada clique.",
        button="ClickPass",
    ),
    _E(
        "modes",
        "Choose the set of markers: Free, Soccer-Kiki (pitch), or a pose skeleton.",
        "Escolhe o conjunto de marcadores: Free, Soccer-Kiki (campo) ou um esqueleto de pose.",
        button="Tpl: …",
    ),
    _E(
        "modes",
        "Soccer-Kiki: load or edit its TOML. Other templates: the same template list.",
        "Soccer-Kiki: carrega ou edita o TOML. Outros templates: a mesma lista de templates.",
        keys=("K",),
    ),
    _E(
        "modes",
        "Show the drawing of the template over the video (pitch or skeleton).",
        "Mostra o desenho do template sobre o vídeo (campo ou esqueleto).",
        keys=("G",),
        button="Guide",
    ),
    _E(
        "modes",
        "With the Guide on: show / hide the small reference map.",
        "Com o Guide ligado: mostra / esconde o mapa de referência.",
        keys=("V",),
    ),
    _E(
        "modes",
        "Fit a plane to the markers of each frame and fill in the missing ones.",
        "Ajusta um plano aos marcadores de cada quadro e completa os que faltam.",
        button="Geo Homog",
    ),
    _E(
        "modes",
        "Show / hide the Geo Homog wireframe (after a Geo Homog run).",
        "Mostra / esconde o desenho do Geo Homog (depois de rodar o Geo Homog).",
        keys=("Shift+G",),
    ),
    _E(
        "modes",
        "Show the markers of the previous frames as a trail.",
        "Mostra os marcadores dos quadros anteriores como um rastro.",
        keys=("P",),
        button="Persist",
    ),
    _E(
        "modes",
        "With Persist on: fewer / more frames in the trail.",
        "Com o Persist ligado: menos / mais quadros no rastro.",
        keys=("1", "2"),
    ),
    _E(
        "modes",
        "Trail of the whole video, then 10 frames, then off.",
        "Rastro do vídeo inteiro, depois 10 quadros, depois desligado.",
        keys=("3",),
    ),
    # --- files ----------------------------------------------------------------------
    _E(
        "files",
        "Open another video or image without closing the window. With 1-line mode on, "
        "the points go with you and the next click continues the row.",
        "Abre outro vídeo ou imagem sem fechar a janela. Com o modo 1-line ligado, os "
        "pontos vão junto e o próximo clique continua a linha.",
        keys=("F8",),
        button="Open",
    ),
    _E(
        "files",
        "Load saved markers (CSV). A *_markers_1_line.csv opens in 1-line mode.",
        "Carrega marcadores salvos (CSV). Um *_markers_1_line.csv abre no modo 1-line.",
        button="Load",
    ),
    _E(
        "files",
        "Save menu: markers CSV, YOLO datasets, video with the markers drawn.",
        "Menu de salvar: CSV dos marcadores, datasets YOLO, vídeo com os marcadores.",
        keys=("Ctrl+S",),
        button="Save",
    ),
    _E(
        "files",
        "Save the markers and close the window.",
        "Salva os marcadores e fecha a janela.",
        keys=("ESC",),
    ),
    _E(
        "files",
        "Load boxes from a tracker (YOLO, SAM 3) and show them over the video. "
        "The small square beside it shows / hides them.",
        "Carrega caixas de um rastreador (YOLO, SAM 3) e mostra sobre o vídeo. "
        "O quadradinho ao lado mostra / esconde.",
        button="Load Track CSV",
    ),
    _E(
        "files",
        "Turn the loaded boxes into marker CSV files.",
        "Transforma as caixas carregadas em arquivos CSV de marcadores.",
        button="BBox>Coords",
    ),
    _E(
        "files",
        "Choose a dataset folder: the next export is added to it (several videos).",
        "Escolhe uma pasta de dataset: a próxima exportação entra nela (vários vídeos).",
        keys=("F7",),
        button="Dataset",
    ),
    _E(
        "files",
        "Export the markers as a YOLO-pose dataset.",
        "Exporta os marcadores como dataset YOLO-pose.",
        keys=("F9",),
    ),
    _E(
        "files",
        "Save a training dataset: pose or detect, train/val/test split, PNG images.",
        "Salva um dataset de treino: pose ou detect, divisão train/val/test, imagens PNG.",
        keys=("Ctrl+E",),
    ),
    _E(
        "files",
        "Give names to the markers p0, p1… (COCO or your own).",
        "Dá nomes aos marcadores p0, p1… (COCO ou os seus).",
        keys=("Ctrl+N",),
        button="Rename",
    ),
    # --- ai -------------------------------------------------------------------------
    _E(
        "ai",
        "AI Track on / off (green = on): the selected marker follows the image as you "
        "step the frames. Click to correct it; every click teaches the tracker.",
        "Liga / desliga o AI Track (verde = ligado): o marcador selecionado acompanha a "
        "imagem conforme você passa os quadros. Clique para corrigir; cada clique ensina "
        "o rastreador.",
        keys=("T",),
        button="AI Track",
    ),
    _E(
        "ai",
        "Track the whole video, forward and back, and smooth the result.",
        "Rastreia o vídeo inteiro, para frente e para trás, e suaviza o resultado.",
        keys=("Shift+T",),
        mouse="Shift+Click",
        button="AI Track",
    ),
    _E(
        "ai",
        "Tracking settings (search window, thresholds) and the Up/Down frame step.",
        "Ajustes do rastreio (janela de busca, limiares) e o passo das teclas Up/Down.",
        keys=("Ctrl+T",),
        button="Cfg",
    ),
    _E(
        "ai",
        "What is tracked: a point, a circle or a box. With circle or box, drag on the "
        "video to set its size.",
        "O que é rastreado: um ponto, um círculo ou uma caixa. Com círculo ou caixa, "
        "arraste no vídeo para definir o tamanho.",
        button="Shp:…",
    ),
    _E(
        "ai",
        "Use a neural network (ResNet and others) to compare the images. Off = faster.",
        "Usa uma rede neural (ResNet e outras) para comparar as imagens. Desligado = mais rápido.",
        button="Deep",
    ),
    _E(
        "ai",
        "Choose the network weights file (.pt / .pth).",
        "Escolhe o arquivo de pesos da rede (.pt / .pth).",
        keys=("Ctrl+W", "Ctrl+M"),
        button="Model",
    ),
    _E(
        "ai",
        "Net: open the AI Track network window (the folder where the network is kept, "
        "download / upload on Hugging Face, retrain, go back to an earlier version). "
        "The same window: python -m vaila.ai_tracker_hub gui.",
        "Net: abre a janela da rede do AI Track (a pasta onde a rede fica guardada, baixar / "
        "enviar no Hugging Face, retreinar, voltar a uma versão anterior). A mesma janela: "
        "python -m vaila.ai_tracker_hub gui.",
        button="Net",
    ),
    _E(
        "ai",
        "Where the AI Track network lives: the folder you choose (asked once, the first "
        "time it saves), outside vailá, so reinstalling vailá does not lose it. Start with "
        "--ai-models DIR (or VAILA_AI_MODELS_HOME) to use another folder. It keeps one "
        "network per feature space (Deep on or off), the points you placed by hand for "
        "retraining, and earlier versions. It is saved when AI Track goes off, on Save, "
        "when Net opens and when the window closes, even with AI Track still on.",
        "Onde a rede do AI Track fica: na pasta que você escolhe (perguntada uma vez, na "
        "primeira gravação), fora do vailá, então reinstalar o vailá não a perde. Abra com "
        "--ai-models PASTA (ou VAILA_AI_MODELS_HOME) para usar outra pasta. Ela guarda uma "
        "rede por espaço de features (Deep ligado ou desligado), os pontos que você marcou "
        "à mão para retreinar e as versões anteriores. É salva quando o AI Track desliga, "
        "ao salvar, ao abrir o Net e ao fechar a janela, mesmo com o AI Track ligado.",
        button="Net",
    ),
    # --- timeline -------------------------------------------------------------------
    _E(
        "timeline",
        "On the coloured strip above the bar: go to that frame. Green = frames of the "
        "selected marker, blue = other markers, gold line = this frame.",
        "Na faixa colorida acima da barra: vai para aquele quadro. Verde = quadros do "
        "marcador selecionado, azul = outros marcadores, linha dourada = este quadro.",
        mouse="Click",
    ),
    # --- measure --------------------------------------------------------------------
    _E(
        "measure",
        "CALIB on / off: tell the real size (a line, a plane, or a REF3D file) so the "
        "measures come out in real units.",
        "Liga / desliga o CALIB: informa o tamanho real (uma linha, um plano ou um "
        "arquivo REF3D) para as medidas saírem em unidades reais.",
        keys=("Shift+Q",),
        button="CALIB",
    ),
    _E(
        "measure",
        "REF3D: build the 3D points of a DLT3D calibration from the measures of a box, "
        "a pole with marks, or a triad (X, Y, Z arms); click them in order; Save writes "
        ".ref3d, the pixels and the checked .dlt3d. Also option 4 of CALIB. The measures "
        "come from ref3d_template.toml: N writes it to fill in, T (Enter) loads it.",
        "REF3D: monta os pontos 3D de uma calibração DLT3D a partir das medidas de uma "
        "caixa, de um bastão com marcas ou de um triedro (hastes X, Y, Z); clique-os em "
        "ordem; Save grava .ref3d, os pixels e o .dlt3d conferido. Também é a opção 4 do CALIB. "
        "As medidas vêm do ref3d_template.toml: N gera o arquivo para preencher, T (Enter) "
        "carrega.",
        keys=("Shift+R",),
        button="REF3D",
    ),
    _E(
        "measure",
        "MEASURE on / off. The small square beside it shows / hides the values.",
        "Liga / desliga o MEASURE. O quadradinho ao lado mostra / esconde os valores.",
        keys=("Q",),
        button="MEASURE",
    ),
    _E(
        "measure",
        "What to measure: 1 distance, 2 area, 3 angle, 4 velocity, 5 acceleration, 6 level.",
        "O que medir: 1 distância, 2 área, 3 ângulo, 4 velocidade, 5 aceleração, 6 nível.",
        keys=("1 … 6",),
    ),
    _E(
        "measure",
        "Kick Measure (needs --kick-run, see Tpl: K): 7 height of an object, 2 clicks in "
        "the same frame (base on the grass, then top).",
        "Kick Measure (precisa de --kick-run, veja Tpl: K): 7 altura de um objeto, 2 "
        "cliques no mesmo quadro (base no gramado, depois o topo).",
        keys=("7",),
    ),
    _E(
        "measure",
        "Kick Measure: 8 line plane. 2 clicks on the grass (kick, arrival) fix a vertical "
        "plane; then each click on the ball gives the distance along the line and the height. "
        "The bottom of the window shows s, z, speed and acceleration of the current frame.",
        "Kick Measure: 8 plano da linha. 2 cliques no gramado (chute, chegada) fixam um plano "
        "vertical; depois cada clique na bola dá a distância ao longo da linha e a altura. "
        "A base da janela mostra s, z, velocidade e aceleração do quadro atual.",
        keys=("8",),
    ),
    _E(
        "measure",
        "Kick Measure: 9 event. 2 clicks, each in its own frame: duration, average speed "
        "between the two points and the frame rate.",
        "Kick Measure: 9 evento. 2 cliques, cada um no seu quadro: duração, velocidade "
        "média entre os dois pontos e a taxa de quadros.",
        keys=("9",),
    ),
    _E(
        "measure",
        "Kick Measure: 0 spin. Click the same mark of the ball in each frame it shows up: "
        "turns, rps and rpm (Enter closes the result).",
        "Kick Measure: 0 giro. Clique a mesma marca da bola em cada quadro em que ela "
        "aparece: voltas, rps e rpm (Enter encerra o resultado).",
        keys=("0",),
    ),
    _E(
        "measure",
        "Kick Measure: arc / curve. Click points along it: length, arc radius, angle and "
        "rise of the fitted arc (Enter closes the result).",
        "Kick Measure: arco / curva. Clique pontos ao longo dela: comprimento, raio, ângulo "
        "e flecha do arco ajustado (Enter encerra o resultado).",
        keys=(",",),
    ),
    _E(
        "measure",
        "Kick Measure: height line. One click at the wanted height (after the line of 8): "
        "horizontal guides at that height are drawn on the video.",
        "Kick Measure: linha de altura. Um clique na altura desejada (depois da linha do 8): "
        "guias horizontais naquela altura são desenhadas no vídeo.",
        keys=(".",),
    ),
    _E(
        "measure",
        "Kick Measure: open a FreeKiki kick_analysis folder to measure on its video, in "
        "metres of the pitch, in any frame. Same as uv run --no-sync vaila/getpixelvideo.py "
        "--kick-run DIR -f VIDEO. After loading only the pitch lines are drawn (check the "
        "calibration); the Kick panel on the right does the rest, step by step. Hover shows "
        "Real: (x, y) m.",
        "Kick Measure: abre uma pasta kick_analysis do FreeKiki para medir no vídeo dela, em "
        "metros do campo, em qualquer quadro. Igual a uv run --no-sync vaila/getpixelvideo.py "
        "--kick-run PASTA -f VIDEO. Depois de carregar só as linhas do campo aparecem (confira "
        "a calibração); o painel Kick, à direita, faz o resto, por etapas. O mouse mostra "
        "Real: (x, y) m.",
        keys=("Tpl: K",),
    ),
    _E(
        "measure",
        "Per-frame calibration (frame by frame): the SAM 3 output folder of the camera "
        "(processed_sam3dinov3_*/<stem>/) keeps <stem>_frames.dlt3d (the camera of EVERY "
        "frame, 11 DLT3D parameters per row), <stem>_frames.dlt2d (the pitch-plane homography "
        "of every frame) and <stem>_frames_cameras.csv (frame, source: dlt3d / solved / "
        "tracked / approximate). Loaded, the video is calibrated in metres of the pitch in any "
        "frame: keys 1-6 give real distances, 7 heights. Auto-load, only when the session has "
        "no calibration yet: --kick-run; Load Tracking CSV of the SAM 3 folder; opening the "
        "video when <video folder>/kick_analysis_*/work/<stem>/sam3dinov3/... holds the file. "
        "Manual: CALIB (Shift+Q) then L and pick the *_frames.dlt3d / *_frames.dlt2d, or "
        "uv run --no-sync vaila/getpixelvideo.py -f VIDEO --calib-frames FILE. The file must "
        "have one row per video frame. On frames whose camera is approximate (copied from "
        "another frame) a warning is shown: measure with care there.",
        "Calibração por quadro: a pasta de saída do SAM 3 da câmera "
        "(processed_sam3dinov3_*/<stem>/) guarda <stem>_frames.dlt3d (a câmera de TODOS os "
        "quadros, 11 parâmetros DLT3D por linha), <stem>_frames.dlt2d (a homografia do plano "
        "do campo de cada quadro) e <stem>_frames_cameras.csv (quadro, origem: dlt3d / solved "
        "/ tracked / approximate). Carregado, o vídeo fica calibrado em metros do campo em "
        "qualquer quadro: teclas 1-6 dão distâncias reais, 7 alturas. Carga automática, só "
        "quando a sessão ainda não tem calibração: --kick-run; Load Tracking CSV da pasta do "
        "SAM 3; ao abrir o vídeo quando <pasta do vídeo>/kick_analysis_*/work/<stem>/"
        "sam3dinov3/... tem o arquivo. Manual: CALIB (Shift+Q) depois L e escolha o "
        "*_frames.dlt3d / *_frames.dlt2d, ou uv run --no-sync vaila/getpixelvideo.py -f VIDEO "
        "--calib-frames ARQUIVO. O arquivo precisa ter uma linha por quadro do vídeo. Nos "
        "quadros com câmera aproximada (copiada de outro quadro) aparece um aviso: meça com "
        "cautela ali.",
        keys=("Shift+Q, L",),
    ),
    _E(
        "measure",
        "Kick SAM: with the video open, click it and pick the kick_analysis folder (the "
        "one with the SAM 3 run). Same as Tpl: K, without the Tpl dialog.",
        "Kick SAM: com o vídeo aberto, clique e escolha a pasta kick_analysis (a da "
        "rodada do SAM 3). Igual ao Tpl: K, sem passar pelo diálogo do Tpl.",
        button="Kick SAM",
    ),
    _E(
        "measure",
        "Kick panel (right side, after Kick SAM / --kick-run), 4 steps by its tabs. 1 "
        "Calibration: frames per camera source and the pitch lines on the video (green = "
        "DLT3D, yellow = solved, orange = approximate). 2 Layers: boxes, 2D pose, ball and "
        "labels are off until you switch them on; filter people by role or team and tick the "
        "ones to measure; Ball -> p0 copies the ball track into the editable marker p0. 3 "
        "Metrics: pick a metric (speed, distance to the ball or to a person, height of a "
        "body point, jump, point speed) for the ticked person and + Add: its value is shown "
        "above the person; or Measure by hand (Quick Measure modes). 4 Export: mkvis3d "
        "(.c3d), Blender (.bvh + script), metrics CSV or all, into "
        "processed_kickmeasure_* inside the kick_analysis folder; Open mkvis3d / Open "
        "Blender show the last export. The terminal prints the command that repeats it.",
        "Painel Kick (à direita, depois do Kick SAM / --kick-run), 4 etapas pelas abas. 1 "
        "Calibração: quadros por fonte de câmera e as linhas do campo no vídeo (verde = "
        "DLT3D, amarelo = resolvida, laranja = aproximada). 2 Camadas: caixas, pose 2D, "
        "bola e rótulos ficam desligados até você ligar; filtre as pessoas por função ou "
        "time e marque as que quer medir; Bola -> p0 copia a trilha da bola para o "
        "marcador p0 editável. 3 Métricas: escolha uma métrica (velocidade, distância à "
        "bola ou a uma pessoa, altura de um ponto do corpo, salto, velocidade do ponto) "
        "para a pessoa marcada e + Adicionar: o valor aparece acima da pessoa; ou Medir "
        "à mão (modos do Quick Measure). 4 Exportar: mkvis3d (.c3d), Blender (.bvh + "
        "script), CSV das métricas ou tudo, em processed_kickmeasure_* dentro da pasta "
        "kick_analysis; Abrir mkvis3d / Abrir Blender mostram a última exportação. O "
        "terminal imprime o comando que repete a exportação.",
        button="Kick SAM",
    ),
    _E(
        "measure",
        "Kick panel: click a player (box or skeleton) to select it for the metrics, click "
        "again to unselect (steps 2 and 3, with the boxes or pose layer on and no measure "
        "mode). The mouse wheel over the panel scrolls its lists instead of zooming.",
        "Painel Kick: clique num jogador (caixa ou esqueleto) para selecioná-lo nas "
        "métricas, clique de novo para desmarcar (etapas 2 e 3, com a camada de caixas ou "
        "pose ligada e sem modo de medida). A roda do mouse sobre o painel rola as listas "
        "em vez de dar zoom.",
        mouse="Left click",
    ),
    _E(
        "measure",
        "Add a point of the measure.",
        "Adiciona um ponto da medida.",
        mouse="Left click",
    ),
    _E(
        "measure",
        "Undo the last point.",
        "Desfaz o último ponto.",
        mouse="Right click",
    ),
    _E(
        "measure",
        "Close the area, or open the menu to save the measures. In CALIB: confirm the "
        "points and type the real sizes.",
        "Fecha a área, ou abre o menu para salvar as medidas. No CALIB: confirma os "
        "pontos e digita os tamanhos reais.",
        keys=("Enter",),
    ),
    _E(
        "measure",
        "Erase the measures drawn on the image (the calibration stays).",
        "Apaga as medidas desenhadas na imagem (a calibração fica).",
        keys=("Backspace",),
        button="Clear",
    ),
    _E(
        "measure",
        "Undo / redo a measure.",
        "Desfaz / refaz uma medida.",
        keys=("Ctrl+Z", "Ctrl+Y"),
    ),
    # --- labeling -------------------------------------------------------------------
    _E(
        "labeling",
        "Box labeling on / off (datasets for object detection).",
        "Liga / desliga a rotulagem com caixas (datasets para detecção de objetos).",
        keys=("L",),
        button="Labeling",
    ),
    _E(
        "labeling",
        "Draw a box of the current class.",
        "Desenha uma caixa da classe atual.",
        mouse="Drag",
    ),
    _E(
        "labeling",
        "Remove the last box of this frame.",
        "Remove a última caixa deste quadro.",
        keys=("Z",),
    ),
    _E(
        "labeling",
        "Change the name of the class.",
        "Muda o nome da classe.",
        keys=("N",),
    ),
    _E(
        "labeling",
        "Save the labeling project and export the dataset.",
        "Salva o projeto de rotulagem e exporta o dataset.",
        keys=("F5",),
    ),
    _E(
        "labeling",
        "Load a labeling project (JSON).",
        "Carrega um projeto de rotulagem (JSON).",
        keys=("F6",),
    ),
    # --- freekiki -------------------------------------------------------------------
    _E(
        "freekiki",
        "Pick the nearest point, or its ghost (the suggested position).",
        "Escolhe o ponto mais próximo, ou o fantasma dele (a posição sugerida).",
        mouse="Right click",
    ),
    _E(
        "freekiki",
        "Place the point that is selected.",
        "Coloca o ponto que está selecionado.",
        mouse="Left click",
    ),
    _E(
        "freekiki",
        "The selected point is not visible in this frame.",
        "O ponto selecionado não aparece neste quadro.",
        keys=("Delete",),
    ),
    _E(
        "freekiki",
        "Predict the 49 points of this frame with the model.",
        "Prevê os 49 pontos deste quadro com o modelo.",
        keys=("F2",),
    ),
    _E("freekiki", "Accept the ghost.", "Aceita o fantasma.", keys=("F10",)),
    _E(
        "freekiki",
        "This frame is reviewed (every visible point placed or hidden).",
        "Este quadro está revisado (todo ponto visível colocado ou escondido).",
        keys=("F3",),
    ),
    _E(
        "freekiki",
        "Discard this frame / show the discarded frames again.",
        "Descarta este quadro / mostra de novo os quadros descartados.",
        keys=("X", "Shift+X"),
    ),
    _E(
        "freekiki",
        "Next / previous frame still to review.",
        "Próximo / anterior quadro ainda por revisar.",
        keys=("PgDn", "PgUp"),
    ),
    _E(
        "freekiki",
        "Go to a frame by its number (first frame is 0).",
        "Vai a um quadro pelo número (o primeiro é 0).",
        keys=("Ctrl+F",),
    ),
    _E(
        "freekiki",
        "Save the session / save the dataset folder of the reviewed frames.",
        "Salva a sessão / salva a pasta de dataset dos quadros revisados.",
        keys=("F4", "F9"),
    ),
    # --- general --------------------------------------------------------------------
    _E(
        "general",
        "This screen. Inside it: L changes the language, 1…9 and 0 jump to a section.",
        "Esta tela. Dentro dela: L muda o idioma, 1…9 e 0 vão para uma seção.",
        keys=("H",),
        button="Help",
    ),
    _E(
        "general",
        "The complete manual, in the web browser.",
        "O manual completo, no navegador.",
        keys=("F1", "?"),
        button="?",
    ),
)


def help_entries(section: str | None = None) -> list[HelpEntry]:
    """The reference entries, in display order (one section, or all)."""
    return [e for e in _ENTRIES if section is None or e.section == section]


# --- "what to do now" ---------------------------------------------------------------

_MODE_NAMES = {
    "mark": ("Mark", "Mark"),
    "seq": ("Seq (sequential)", "Seq (sequencial)"),
    "1-line": ("1-line", "1-line"),
    "labeling": ("Labeling (boxes)", "Labeling (caixas)"),
    "measure": ("MEASURE", "MEASURE"),
    "calib": ("CALIB", "CALIB"),
    "freekiki": ("FreeKiki review", "Revisão FreeKiki"),
    "ref3d": ("REF3D (DLT3D calibration)", "REF3D (calibração DLT3D)"),
}

_EDIT_MODES = ("mark", "seq", "1-line", "labeling", "freekiki", "ref3d")


def _now(en: str, pt: str, **kw) -> HelpEntry:
    return HelpEntry("now", en, pt, **kw)


def now_title(context: HelpContext, lang: str) -> str:
    """One line saying where the user is: editor state, mode, 1-line point count."""
    name = _MODE_NAMES.get(context.mode, _MODE_NAMES["mark"])[1 if lang == "pt" else 0]
    parts = ["INSERT" if context.insert else "VISUAL", name]
    if context.mode == "1-line":
        n = context.n_points
        parts.append(
            f"{n} ponto{'s' if n != 1 else ''}"
            if lang == "pt"
            else f"{n} point{'s' if n != 1 else ''}"
        )
    if context.ai_track:
        parts.append("AI Track")
    head = "Agora você está em" if lang == "pt" else "You are now in"
    return f"{head}: " + " · ".join(parts)


def now_steps(context: HelpContext) -> list[HelpEntry]:
    """The steps for what the user is doing, most likely next action first."""
    steps: list[HelpEntry] = []
    if context.mode in _EDIT_MODES and not context.insert:
        steps.append(
            _now(
                "FIRST: turn on INSERT. The window opens in VISUAL, where clicks do not edit.",
                "PRIMEIRO: ligue o INSERT. A janela abre em VISUAL, onde os cliques não editam.",
                keys=("Ctrl+I",),
                button="VISUAL",
            )
        )

    if context.mode == "ref3d":
        nxt = context.ref3d_next
        steps += [
            _now(
                f"Click {nxt}. The panel on the right shows it in the 3D sketch."
                if nxt
                else "Every point is clicked: save.",
                f"Clique o {nxt}. O painel à direita mostra o ponto no desenho 3D."
                if nxt
                else "Todos os pontos foram clicados: salve.",
                mouse="Left click",
            ),
            _now(
                "Remove the last point (click again to redo it).",
                "Remove o último ponto (clique de novo para refazer).",
                mouse="Right click",
            ),
            _now(
                "This point is not visible: leave it empty and go to the next one.",
                "Este ponto não aparece: deixa vazio e passa para o próximo.",
                keys=("A",),
            ),
            _now(
                "Re-place a point: TAB, or click its row in the panel.",
                "Recolocar um ponto: TAB, ou clique a linha dele no painel.",
                keys=("TAB",),
            ),
            _now(
                "From 6 points on, the DLT3D is solved at each click: rings show where "
                "every 3D point should be (green = click agrees, red = far). "
                "fx and fy must be close: if not, a measure of the object is wrong.",
                "A partir de 6 pontos o DLT3D é calculado a cada clique: os anéis mostram "
                "onde cada ponto 3D deveria estar (verde = clique confere, vermelho = longe). "
                "fx e fy devem ser parecidos: se não forem, uma medida do objeto está errada.",
            ),
            _now(
                "The object is in another video: open it; the points go with you.",
                "O objeto está em outro vídeo: abra-o; os pontos vão junto.",
                keys=("F8",),
                button="Add video",
            ),
            _now(
                "Save <video>_markers_1_line .csv, .ref3d and .dlt3d (ESC saves and closes). "
                "The .dlt3d is final: no need to run Make DLT3D.",
                "Salva <video>_markers_1_line .csv, .ref3d e .dlt3d (ESC salva e fecha). "
                "O .dlt3d já é o final: não precisa rodar o Make DLT3D.",
                keys=("Ctrl+S", "ESC"),
                button="Save",
            ),
            _now(
                "Illustrated guide in the browser: axes, templates, point order, checks.",
                "Guia ilustrado no navegador: eixos, gabaritos, ordem dos pontos, conferência.",
                keys=("F1",),
                button="Guide",
            ),
            _now(
                "Other measures: edit ref3d_template.toml next to the video, then REF3D off "
                "and on + Enter (T loads it; N writes a new one).",
                "Outras medidas: edite o ref3d_template.toml ao lado do vídeo e depois desligue "
                "e ligue o REF3D + Enter (T carrega; N gera um novo).",
                keys=("Shift+R",),
                button="REF3D",
            ),
            _now(
                "Leave REF3D (the clicks stay as 1-line points).",
                "Sai do REF3D (os cliques ficam como pontos 1-line).",
                keys=("Shift+R",),
                button="REF3D",
            ),
        ]
        return steps

    if context.mode == "1-line":
        n, nxt, sel = context.n_points, context.next_point, context.selected
        if sel >= 0:
            steps.append(
                _now(
                    f"Re-place p{sel}, the selected point (it keeps its number).",
                    f"Recoloca o p{sel}, o ponto selecionado (o número não muda).",
                    mouse="Left click",
                )
            )
            steps.append(_now(f"Remove p{sel}.", f"Remove o p{sel}.", mouse="Right click"))
        else:
            steps.append(
                _now(
                    f"Add the next point: p{nxt}.",
                    f"Adiciona o próximo ponto: p{nxt}.",
                    mouse="Left click",
                )
            )
            steps.append(
                _now(
                    "Remove the last point (click again to redo it).",
                    "Remove o último ponto (clique de novo para refazer).",
                    mouse="Right click",
                )
            )
        steps.append(
            _now(
                "Select a point to re-place or remove. After the last one: back to adding.",
                "Seleciona um ponto para recolocar ou remover. Depois do último: volta a adicionar.",
                keys=("TAB",),
            )
        )
        steps.append(
            _now(
                "Go to another frame. The points stay on screen (cyan ring = another frame).",
                "Vai a outro quadro. Os pontos ficam na tela (anel ciano = outro quadro).",
                keys=("Right", "Left"),
            )
        )
        steps.append(
            _now(
                f"CONTINUE IN ANOTHER VIDEO: open it; the {n} points go with you and the "
                f"next click is p{nxt}."
                if n
                else "Open another video.",
                f"CONTINUAR EM OUTRO VÍDEO: abra o vídeo; os {n} pontos vão junto e o "
                f"próximo clique é o p{nxt}."
                if n
                else "Abre outro vídeo.",
                keys=("F8",),
                button="Add video" if n else "Open",
            )
        )
        steps.append(
            _now(
                "Save <video>_markers_1_line.csv next to the video on screen "
                "(ESC saves and closes the window).",
                "Salva <video>_markers_1_line.csv ao lado do vídeo que está na tela "
                "(ESC salva e fecha a janela).",
                keys=("Ctrl+S", "ESC"),
                button="Save",
            )
        )
        steps.append(
            _now("Leave 1-line mode.", "Sai do modo 1-line.", keys=("C",), button="Mode: 1-line")
        )
        return steps

    if context.mode == "measure":
        return [
            _now(
                "Choose what to measure: 1 distance, 2 area, 3 angle, 4 velocity, "
                "5 acceleration, 6 level.",
                "Escolha o que medir: 1 distância, 2 área, 3 ângulo, 4 velocidade, "
                "5 aceleração, 6 nível.",
                keys=("1 … 6",),
            ),
            _now("Add a point.", "Adiciona um ponto.", mouse="Left click"),
            _now("Undo the last point.", "Desfaz o último ponto.", mouse="Right click"),
            _now(
                "Close the area, or open the menu to save the measures.",
                "Fecha a área, ou abre o menu para salvar as medidas.",
                keys=("Enter",),
            ),
            _now(
                "Measures in real units: calibrate first.",
                "Medidas em unidades reais: calibre antes.",
                keys=("Shift+Q",),
                button="CALIB",
            ),
            _now(
                "Erase what is drawn (the calibration stays).",
                "Apaga o que está desenhado (a calibração fica).",
                keys=("Backspace",),
                button="Clear",
            ),
            _now("Leave MEASURE.", "Sai do MEASURE.", keys=("Q",), button="MEASURE"),
        ]

    if context.mode == "calib":
        return [
            _now(
                "Click the reference points in the order shown on the video.",
                "Clique os pontos de referência na ordem mostrada no vídeo.",
                mouse="Left click",
            ),
            _now("Undo the last point.", "Desfaz o último ponto.", mouse="Right click"),
            _now(
                "Confirm the points and type the real sizes.",
                "Confirma os pontos e digita os tamanhos reais.",
                keys=("Enter",),
            ),
            _now("Leave CALIB.", "Sai do CALIB.", keys=("Shift+Q",), button="CALIB"),
        ]

    if context.mode == "labeling":
        steps += [
            _now(
                "Draw a box around the object.",
                "Desenha uma caixa em volta do objeto.",
                mouse="Drag",
            ),
            _now("Change the class name.", "Muda o nome da classe.", keys=("N",)),
            _now("Remove the last box.", "Remove a última caixa.", keys=("Z",)),
            _now("Next / previous frame.", "Próximo quadro / anterior.", keys=("Right", "Left")),
            _now(
                "Save the project and export the dataset.",
                "Salva o projeto e exporta o dataset.",
                keys=("F5",),
                button="Save",
            ),
            _now(
                "Continue in another video (the dataset stays the same).",
                "Continua em outro vídeo (o dataset continua o mesmo).",
                keys=("F8",),
                button="Open",
            ),
            _now("Leave Labeling.", "Sai do Labeling.", keys=("L",), button="Labeling"),
        ]
        return steps

    if context.mode == "freekiki":
        steps += [
            _now(
                "Pick the nearest point or ghost.",
                "Escolhe o ponto ou fantasma mais próximo.",
                mouse="Right click",
            ),
            _now("Place the selected point.", "Coloca o ponto selecionado.", mouse="Left click"),
            _now(
                "The selected point is not visible.",
                "O ponto selecionado não aparece.",
                keys=("Delete",),
            ),
            _now("Accept the ghost.", "Aceita o fantasma.", keys=("F10",)),
            _now("This frame is reviewed.", "Este quadro está revisado.", keys=("F3",)),
            _now("Next frame still to review.", "Próximo quadro por revisar.", keys=("PgDn",)),
            _now(
                "Save the session / the dataset folder.",
                "Salva a sessão / a pasta de dataset.",
                keys=("F4", "F9"),
            ),
        ]
        return steps

    # mark / seq
    steps.append(
        _now(
            "Choose which marker the next click places.",
            "Escolhe qual marcador o próximo clique coloca.",
            keys=("TAB",),
            button="Go KP",
        )
    )
    if context.mode == "seq":
        steps.append(
            _now(
                "Place the selected marker and move on to the next one.",
                "Coloca o marcador selecionado e passa para o próximo.",
                mouse="Left click",
            )
        )
    else:
        steps.append(
            _now(
                "Place the selected marker on this frame, or move it.",
                "Coloca o marcador selecionado neste quadro, ou muda a posição dele.",
                mouse="Left click",
            )
        )
    steps.append(
        _now(
            "Remove the selected marker on this frame.",
            "Remove o marcador selecionado neste quadro.",
            keys=("R",),
            mouse="Right click",
        )
    )
    steps.append(
        _now(
            "Next / previous frame (video paused). Space plays and pauses.",
            "Próximo quadro / anterior (vídeo pausado). Space toca e pausa.",
            keys=("Right", "Left"),
        )
    )
    if context.ai_track:
        steps.append(
            _now(
                "AI Track is on: the marker follows as you step the frames; click to "
                "correct it. T turns it off, Shift+T tracks the whole video.",
                "O AI Track está ligado: o marcador acompanha conforme você passa os "
                "quadros; clique para corrigir. T desliga, Shift+T rastreia o vídeo inteiro.",
                keys=("T", "Shift+T"),
                button="AI Track",
            )
        )
    else:
        steps.append(
            _now(
                "Let the marker follow the image by itself (AI Track).",
                "Deixa o marcador acompanhar a imagem sozinho (AI Track).",
                keys=("T",),
                button="AI Track",
            )
        )
    steps.append(
        _now(
            "Save <video>_markers.csv (ESC saves and closes the window).",
            "Salva <video>_markers.csv (ESC salva e fecha a janela).",
            keys=("Ctrl+S", "ESC"),
            button="Save",
        )
    )
    steps.append(
        _now(
            "Continue in another video.",
            "Continua em outro vídeo.",
            keys=("F8",),
            button="Open",
        )
    )
    return steps


# --- toolbar hints ------------------------------------------------------------------
# id of the button in getpixelvideo's toolbar -> (shortcut keys, English, Portuguese).

TOOLBAR_HINTS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "editor_mode": (
        ("Ctrl+I",),
        "VISUAL = look only, INSERT = edit markers",
        "VISUAL = só olhar, INSERT = editar marcadores",
    ),
    "template": (
        (),
        "Choose the set of markers (template)",
        "Escolhe o conjunto de marcadores (template)",
    ),
    "marker_mode": (
        ("C",),
        "Change mode: Mark, Seq, 1-line (C = 1-line)",
        "Muda o modo: Mark, Seq, 1-line (C = 1-line)",
    ),
    "persist": (("P",), "Trail of the previous frames", "Rastro dos quadros anteriores"),
    "mouse_play": (
        ("M",),
        "Mark with the mouse while the video plays",
        "Marca com o mouse enquanto o vídeo toca",
    ),
    "click_pass": (
        (),
        "Go to the next frame after each click",
        "Vai ao próximo quadro depois de cada clique",
    ),
    "labeling": (
        ("L",),
        "Draw boxes (object detection datasets)",
        "Desenha caixas (datasets de detecção)",
    ),
    "calib": (
        ("Shift+Q",),
        "Calibrate: pixels to real units",
        "Calibra: pixels para unidades reais",
    ),
    "ref3d": (
        ("Shift+R",),
        "Build the .ref3d from the object's measures, click, get the .dlt3d",
        "Monta o .ref3d com as medidas do objeto, clique, gera o .dlt3d",
    ),
    "ref3d_panel": (
        (),
        "Hide / show the REF3D panel (the rings on the video stay)",
        "Esconde / mostra o painel do REF3D (os anéis no vídeo continuam)",
    ),
    "ref3d_guide": (
        ("F1",),
        "Illustrated calibration guide (axes, templates, point order) in the browser",
        "Guia ilustrado da calibração (eixos, gabaritos, ordem dos pontos) no navegador",
    ),
    "measure": (
        ("Q",),
        "Measure distance, area, angle, velocity",
        "Mede distância, área, ângulo, velocidade",
    ),
    "measure_overlay": (
        (),
        "Show / hide the measured values",
        "Mostra / esconde os valores medidos",
    ),
    "measure_clear": (("Backspace",), "Erase the measures drawn", "Apaga as medidas desenhadas"),
    "fps": (("I",), "Type the video frame rate (Hz)", "Digita a taxa de quadros do vídeo (Hz)"),
    "guide": (("G",), "Template drawing over the video", "Desenho do template sobre o vídeo"),
    "guide_toggle": (("G",), "Guide on / off", "Guide ligado / desligado"),
    "help": (("H",), "Keys and buttons, step by step", "Teclas e botões, passo a passo"),
    "help_web": (("F1",), "Complete manual in the browser", "Manual completo no navegador"),
    "tracking_csv": (
        (),
        "Load boxes from a tracker (YOLO, SAM 3)",
        "Carrega caixas de um rastreador (YOLO, SAM 3)",
    ),
    "tracking_show": ((), "Show / hide the loaded boxes", "Mostra / esconde as caixas carregadas"),
    "bbox_coords": (
        (),
        "Turn the loaded boxes into marker CSVs",
        "Transforma as caixas em CSVs de marcadores",
    ),
    "ai_track": (
        ("T",),
        "Marker follows the image (Shift+T: whole video)",
        "Marcador acompanha a imagem (Shift+T: vídeo inteiro)",
    ),
    "ai_deep": ((), "Neural network features for AI Track", "Rede neural para o AI Track"),
    "ai_net": (
        (),
        "AI Track network: folder, Hugging Face, retrain, versions",
        "Rede do AI Track: pasta, Hugging Face, retreino, versões",
    ),
    "ai_shape": (
        (),
        "Track a point, a circle or a box",
        "Rastreia um ponto, um círculo ou uma caixa",
    ),
    "ai_cfg": (("Ctrl+T",), "AI Track settings", "Ajustes do AI Track"),
    "ai_model": (
        ("Ctrl+W",),
        "Choose the network weights file",
        "Escolhe o arquivo de pesos da rede",
    ),
    "open": (("F8",), "Open another video or image", "Abre outro vídeo ou imagem"),
    "open_next": (
        ("F8",),
        "Add video: the 1-line points go with you",
        "Add video: os pontos do 1-line vão junto",
    ),
    "kick_sam": (
        (),
        "Kick SAM: pick a kick_analysis folder; the Kick panel then guides the measure",
        "Kick SAM: escolhe a pasta kick_analysis; o painel Kick guia a medida",
    ),
    "load": ((), "Load saved markers (CSV)", "Carrega marcadores salvos (CSV)"),
    "save": (
        ("Ctrl+S",),
        "Save menu (ESC = save and close)",
        "Menu de salvar (ESC = salva e fecha)",
    ),
    "restore": ((), "Back to the last Save", "Volta ao último Save"),
    "dataset": (
        ("F7",),
        "Dataset folder for the next export",
        "Pasta de dataset da próxima exportação",
    ),
    "rename": (("Ctrl+N",), "Name the markers p0, p1…", "Dá nomes aos marcadores p0, p1…"),
    "goto_marker": (
        ("Ctrl+G",),
        "Select a marker by its number",
        "Seleciona um marcador pelo número",
    ),
    "lock_marker": (("B",), "Lock the selection on this marker", "Trava a seleção neste marcador"),
    "delete_range": (
        (),
        "Delete markers over a span of frames",
        "Apaga marcadores em vários quadros",
    ),
    "gap_fill": ((), "Fill the gaps of one marker", "Preenche as falhas de um marcador"),
    "swap_range": (
        (),
        "Swap marker numbers over a span of frames",
        "Troca números de marcadores em vários quadros",
    ),
    "geo_homog": (
        (),
        "Fit a plane and fill in missing markers",
        "Ajusta um plano e completa marcadores",
    ),
}


def toolbar_hint(button_id: str, lang: str) -> tuple[tuple[str, ...], str] | None:
    """``(shortcut keys, text)`` for a toolbar button, or None when it has no hint."""
    hint = TOOLBAR_HINTS.get(button_id)
    if hint is None:
        return None
    return hint[0], hint[2] if lang == "pt" else hint[1]


def wrap_text(text: str, max_width: int, measure) -> list[str]:
    """Break ``text`` into lines no wider than ``max_width``; ``measure(str) -> px``."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if line and measure(candidate) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines or [""]
