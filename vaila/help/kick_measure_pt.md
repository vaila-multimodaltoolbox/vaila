# kick_measure (Português)

> [English](kick_measure.md) · Português

## Informações do módulo

| Campo | Valor |
|-------|--------|
| **Category** | Analysis |
| **Arquivo** | `vaila/kick_measure.py` |
| **Versão** | 0.4.19 |
| **Atualizado** | 24 de novembro de 2026 |
| **Autor** | Paulo Roberto Pereira Santiago |
| **GUI** | Sim (a medição acontece no `getpixelvideo.py`, com o painel por passos do `kick_panel.py`; este módulo é a lógica por trás) |
| **CLI** | Sim (`python -m vaila.kick_measure --run PASTA`, `freekiki.py kick-measure PASTA`, `getpixelvideo.py --kick-run PASTA`, `python -m vaila.kick_measure --run PASTA --export-scene SAIDA`, `python -m vaila.kick_measure --run PASTA --write-cameras`, `freekiki.py kick-cameras PASTA`) |

---

## O que é

Este módulo abre uma pasta de análise de chute do FreeKiki e transforma qualquer pixel do vídeo dela em metros do campo, **em qualquer quadro**, não só nos quadros que o pipeline conseguiu calibrar. Com ele, o *vailá* mede no vídeo, ao vivo, e mostra o resultado na hora: distância, ângulo, velocidade, aceleração, altura de um objeto ou da bola, o tempo entre dois eventos, o giro da bola e o comprimento e o raio de uma curva. Tudo se combina com o AI Track do getpixelvideo.

Sistema de coordenadas do campo (o mesmo do FreeKiki): `x` ao longo do campo, `y` na largura, `z` para cima, em metros, origem no centro do campo. Um quadro do vídeo é um passo de tempo de `1 / fps` segundos (índice do quadro a partir de 0).

## Como abrir uma pasta de chute

| Entrada | O que fazer |
|---------|-------------|
| getpixelvideo, botão **Tpl:** | Aperte **Tpl:** e responda **K** (Kick Measure), escolha a pasta `kick_analysis_*` no navegador e aperte **Enter**. Se o vídeo aberto não for o da câmera da pasta, ele abre o certo. |
| getpixelvideo, botão **Kick SAM** | Com o vídeo aberto, clique em **Kick SAM** e escolha a pasta `kick_analysis_*`. |
| CLI do getpixelvideo | `uv run --no-sync vaila/getpixelvideo.py --kick-run PASTA -f VIDEO` (`-f` é opcional: sem ele vale o vídeo de referência da pasta). |
| Janela do FreeKiki | **Frame B → Soccer Tools → FreeKiki**, seção 5, **Open run and correct...**: escolha a pasta `kick_analysis_*` e a **Camera** (Reference ou Other) e aperte **Measure kick** (ao lado de *Correct ball* e *Correct field keypoints*), ou **Measure kick: video + SAM folder**. |
| CLI do FreeKiki | `uv run --no-sync vaila/freekiki.py kick-measure PASTA [--camera reference\|other] [--no-wait]` |
| Módulo Python | `uv run --no-sync python -m vaila.kick_measure --run PASTA [--camera reference\|other] [--info] [--export cameras.json]` imprime as câmeras (calibradas, resolvidas, rastreadas, aproximadas), a bola e as pessoas; `--export` grava tudo num JSON; `--write-cameras` grava os arquivos de calibração quadro a quadro (veja *Arquivos de calibração quadro a quadro*). |

O botão **Load** do getpixelvideo continua com a sua função (pasta do FreeKiki para *corrigir*, `Tpl: L`); o Kick Measure é o `Tpl: K`. Toda entrada pela janela imprime o comando equivalente no terminal (`>> uv run --no-sync vaila/getpixelvideo.py --kick-run ... -f ...`), então a sessão pode ser repetida sem a janela.

`PASTA` pode ser a pasta `kick_analysis_*`, a sua pasta `*_3d_*` ou a execução de detecção dentro de `work/`.

### O que o carregamento faz

O carregamento mostra **só a calibração**; nada mais é desenhado até você pedir no painel por passos (próxima seção).

- A câmera de **cada** quadro vira uma calibração `camera3d` da sessão do Quick Measure (veja abaixo), então as teclas `1`-`6` já dão metros reais. Quando os arquivos quadro a quadro da próxima seção existem ao lado das saídas do SAM 3, eles são lidos (rápido); senão as câmeras são montadas na hora, com o rastreamento.
- As linhas do campo são reprojetadas no vídeo com essa câmera e coloridas pela fonte da câmera do quadro: DLT3D verde, resolvida amarelo, rastreada ciano, aproximada laranja.
- A trilha da bola (`<video>_markers.csv` ao lado do vídeo, senão o `ball.csv` da pasta 3D) **não** é mais gravada no marcador p0: o painel faz isso só quando você aperta **Bola → marcador p0 (editar / AI Track)**. Um p0 que você já clicou nunca é sobrescrito, e o AI Track continua valendo.
- O mouse mostra `Real: (x, y) m`, e `[approx. camera]` quando o quadro não tem calibração própria.
- No modo `8` a base da janela mostra `s`, `z`, velocidade e aceleração do quadro atual.

## Painel por passos: medir por partes e exportar para o mkvis3d e o Blender

Por quê: com 31 caixas, rótulos e a bola em p0 desenhados de uma vez, nada podia ser medido. O painel (`vaila/kick_panel.py`) fica encaixado à direita do vídeo e tem quatro passos; cada camada fica desligada até você ligá-la. Os textos seguem o idioma da ajuda `H` (inglês ou português).

**Como começar.** Abra o vídeo no getpixelvideo, clique em **Kick SAM** (ou FreeKiki → **Measure kick: video + SAM folder**, ou `--kick-run PASTA`) e escolha a pasta `kick_analysis_*`.

| Passo | O que tem |
|-------|-----------|
| **1 Calib** | Quadros contados por fonte de calibração (DLT3D, resolvida, rastreada, aproximada), a fonte **deste** quadro e **Linhas do campo** ligado/desligado. Confira se as linhas do campo ficam sobre o gramado e siga. |
| **2 Camad.** | Liga **Caixas**, **Pose 2D**, **Bola**, **Rótulos**; filtros por papel **todos / batedor / goleiro / time A / time B / árbitro / marcados**; a lista de pessoas com uma caixa de seleção cada (a roda do mouse sobre o painel rola a lista). Um clique num jogador no vídeo o seleciona. **Bola → marcador p0 (editar / AI Track)** põe a trilha da bola no p0, só quando você aperta. |
| **3 Métr.** | Escolha a pessoa entre as marcadas, a métrica e, quando precisar, o ponto do corpo, e aperte **+ Adicionar métrica**. Cada métrica adicionada mostra o valor *agora* / *máx* no painel e acima do jogador no vídeo. Abaixo, os botões **Medir à mão** iniciam os modos do Quick Measure: `1` distância, `3` ângulo, `4` velocidade, `7` altura base+topo, `8` plano da linha, `9` tempo de evento, `0` giro, `,` arco, `.` linha de altura. |
| **4 Export** | **mkvis3d C3D**, **Blender BVH + script**, **Métricas CSV** ou **Tudo**; **Abrir mkvis3d** e **Abrir Blender** abrem os visualizadores sem travar a janela. |

Recolha o painel com ▶ e reabra pela aba na borda direita.

### Métricas

| Métrica | O que é | Unidade |
|---------|---------|---------|
| Velocidade no campo (`speed`) | velocidade do ponto dos pés da pessoa no gramado | m/s |
| Distância até a bola (`distance_ball`) | ponto dos pés até a bola no gramado | m |
| Distância até outra pessoa (`distance_person`) | exige uma segunda pessoa marcada | m |
| Altura de um ponto do corpo (`height_kpt`) | `z` do ponto na pose 3D | m |
| Altura do salto (`jump`) | subida da pelve sobre a linha de base em pé (mediana dos 10 primeiros quadros válidos); o máximo do resumo é o salto | m |
| Velocidade de um ponto do corpo (`kpt_speed`) | velocidade 3D do ponto | m/s |

Pontos do corpo: `pelvis` (média dos dois quadris), `head`, `neck`, `left_wrist`, `right_wrist`, `left_ankle`, `right_ankle`. Quadros sem dado ficam sem valor (nunca um valor inventado); uma métrica que não pode ser calculada (por exemplo uma distância sem segunda pessoa) diz o motivo.

### Exportação

A exportação vai para `<kick_analysis>/processed_kickmeasure_AAAAMMDD_HHMMSS/`. Com **ninguém marcado** ela exporta todas as pessoas que têm pose 3D.

| Arquivo | Conteúdo |
|---------|----------|
| `kick_scene.c3d` | marcadores `P<id>_<kpt>` das pessoas marcadas (poses 3D), `BALL` e o modelo do campo como `FKP_<nome>` estáticos; abra com o mkvis3d |
| `kick_scene_c3d.csv` | os mesmos marcadores em CSV |
| `blender/person_<id>.bvh` | um BVH por pessoa (lacunas são preenchidas com a posição válida mais próxima) |
| `blender/ball_path.json`, `blender/pitch_lines.json` | a trajetória da bola e as linhas do campo para o Blender |
| `blender/kick_scene_blender.py` | script `bpy` simples: importa os BVHs, uma esfera da bola com chaves a partir do JSON, as linhas do campo como curvas; a taxa de quadros e o intervalo da cena são ajustados por último |
| `metrics.csv`, `metrics_summary.json` | valores por quadro (`frame,time_s,<métrica>...`) e máximo / mínimo / média / quadro do máximo de cada métrica |
| `README.txt` | os comandos para abrir os arquivos |

O terminal imprime o comando equivalente, então a exportação pode ser repetida sem a janela:

```bash
uv run --no-sync python -m vaila.kick_measure --run PASTA --export-scene SAIDA \
    --people 21,23 --metric speed:21 --metric height_kpt:23:right_wrist \
    --metric distance_person:21:23 [--no-blender] [--no-mkvis3d]
```

`--metric TIPO:PESSOA[:OUTRA_OU_PONTO]` pode ser repetido: o terceiro campo é a outra pessoa em `distance_person` e o ponto do corpo em `height_kpt` e `kpt_speed` (padrão `pelvis`); o `jump` usa sempre a pelve. `--people` recebe ids do SAM.

O **mkvis3d** é um projeto separado. O *vailá* o encontra pela variável de ambiente `VAILA_MKVIS3D` (caminho do `mkvis3d.py`) ou por uma pasta `mkvis3d/` ao lado da cópia do *vailá*; sem ele o botão **Abrir mkvis3d** avisa e a exportação dos demais arquivos continua funcionando. À mão: `python mkvis3d.py kick_scene.c3d` abre o visualizador 3D e `python mkvis3d.py info kick_scene.c3d` imprime um resumo.

Exemplo real (chute do Vargas, 169 quadros a 30 fps): 31 ids do SAM, 24 com pose 3D; 97 segmentos do campo por quadro; `mkvis3d info` da exportação: 169 quadros, 190 marcadores (2 x 70 pontos do corpo, BALL, 49 FKP_*); velocidade máxima do batedor P21 de 8,7 m/s no gramado (fonte `box_foot`, o pé da caixa SAM visto pela câmera de cada quadro, 60 de 96 quadros). Uma série de velocidade ou distância usa uma só fonte de posição, a que tem mais quadros válidos (`ground_tracks`, o ponto no chão da pose 3D, os tornozelos ou `box_foot`), e o `metrics_summary.json` registra qual. O `box_foot` pula os quadros com câmera aproximada, porque essas câmeras copiam a vizinha e não acompanham a panorâmica (o pé saltava 9 m em um quadro). Por isso os quadros 36-43 e 47-74 do batedor ficam sem velocidade.

### Limites do painel

- **Uma câmera.** As alturas dependem do plano de referência escolhido; o ± dos modos manuais continua valendo.
- **O salto a partir de poses 3D monoculares é ruidoso:** no exemplo real a pelve variou ±0,18 m em torno da linha de base, então leia o salto como ordem de grandeza, não como uma medida de poucos centímetros.
- O BVH preenche lacunas com a posição válida mais próxima, então uma pessoa que sai do quadro fica parada ali.
- Os arquivos de árbitro `referees/P<n>_referee*.c3d` da pasta 3D usam o `player_id` do `player_map.csv`, não o id do SAM; a exportação faz a conversão.

## Teclas (MEASURE ligado: `Q`)

| Tecla | Medida | Cliques | Resultado |
|-------|--------|---------|-----------|
| `1` | Distância | 2 | metros no gramado daquele quadro |
| `2` | Área | 3 ou mais + Enter | m² |
| `3` | Ângulo | 3 (vértice no meio) | graus |
| `4` | Velocidade | 2, em quadros diferentes | m/s (precisa do FPS) |
| `5` | Aceleração | 3, em quadros distintos | m/s² (precisa do FPS) |
| `6` | Nível | 2 | guia H+V e fita |
| `7` | **Altura** | 2 no mesmo quadro: base no gramado e depois o topo | metros, com o plano usado e um ± de 1 px |
| `8` | **Plano da linha** | 2 no gramado (chute, chegada) e depois 1 clique por posição da bola | `s` (distância ao longo da linha) e `z` (altura) de cada clique, mais velocidade e aceleração 3D da sequência |
| `9` | **Evento** | 2, cada um no seu quadro (início, fim) | duração em s (± 1 quadro), velocidade média entre os dois pontos, taxa de quadros |
| `0` | **Giro** | cada clique marca a *mesma* marca da bola em outro quadro | voltas, rps e rpm (± 1 quadro); o resultado atualiza a partir do 2º clique |
| `,` | **Arco / curva** | 3 ou mais ao longo da curva | comprimento da curva e arco ajustado (raio, ângulo, flecha); atualiza a cada clique |
| `.` | **Linha de altura** | 1 clique na altura desejada, depois da linha do `8` | uma guia horizontal naquela altura, desenhada sobre o vídeo em todos os quadros |

`Enter` fecha os resultados abertos (`0`, `,`); `Ctrl+Z` / `Ctrl+Y` desfazem e refazem; **Clear** / Backspace apagam o desenho e mantêm a calibração. `,` e `.` são as teclas livres do getpixelvideo e existem nos teclados US e ABNT2.

Giro: `k` cliques da mesma marca são `k - 1` voltas entre o primeiro e o último quadro, então `rps = (k - 1) / ((último - primeiro) / fps)` e `rpm = 60 rps`. Com `fps / 2` voltas por segundo ou mais a marca é vista menos de duas vezes por volta e a contagem sofre aliasing; o resultado traz uma nota quando isso acontece.

## Calibração `camera3d`

- **Uma câmera por quadro.** Um quadro com DLT3D do FreeKiki o mantém intacto (fonte `dlt3d`). Um quadro sem ele é **resolvido**: rotação e zoom são ajustados, quadro a quadro, aos pontos brutos do campo de `field_kps_raw.csv` (3 ou mais pontos com confiança de pelo menos 0,5, perda robusta), a partir do quadro vizinho (fonte `solved`, com o erro de reprojeção em pixels).
- **Centro óptico fixo.** Só rotação e zoom são livres: o centro é a mediana dos **10 quadros calibrados mais próximos no tempo**, não uma mediana do clipe inteiro. Um único centro global falhou numa lente longa (cerca de 7000 px de distância focal), em que o centro do DLT3D anda metros ao longo do eixo de visada. O centro fixo é uma hipótese de câmera de transmissão (câmera em tripé).
- **Quadros rastreados.** Um quadro com poucos pontos do campo para ser resolvido costumava copiar o melhor quadro vizinho, enquanto a câmera de transmissão gira e dá zoom dezenas de pixels por quadro. Agora a cena estática (arquibancada, placas de publicidade, marcações) é seguida de quadro a quadro (cantos de Shi-Tomasi e Lucas-Kanade, pareamentos a distâncias de 1, 2, 4, 8 e 16 quadros), com as caixas de pessoas do SAM (aumentadas em 10 %), a bola e as sobreposições da transmissão (placar, logos do canal) mascaradas, e a rotação e o zoom de cada um desses quadros são ajustados a esses pareamentos. O campo é levado das câmeras conhecidas **dos dois lados** do intervalo (até 4 âncoras por lado). O quadro vira `tracked`; `rms_px` é a sua incerteza em pixels (o desacordo das âncoras que o levaram mais o desajuste da família de centro fixo, em quadratura) e `from_frame` é a âncora mais próxima.
- **Quadros aproximados.** Um quadro continua `approximate` (cópia do melhor quadro mais próximo) só quando o rastreador falha: menos de 60 pareamentos o tocam, o erro de transferência das características passa de 4 px, a incerteza passa de 60 px, ou não há âncora ao alcance. A ferramenta avisa (etiqueta no mouse, coluna nos resultados, aviso no HUD do getpixelvideo) em vez de inventar uma calibração. Medidas feitas ali são menos confiáveis.
- Um pixel vira um ponto do gramado **do seu próprio quadro**: o movimento da câmera entre quadros já está considerado.

## Arquivos de calibração quadro a quadro

Toda execução de chute guarda a câmera de **cada** quadro do vídeo em quatro arquivos, gravados **sempre** ao lado das saídas do SAM 3 da câmera: `<kick_analysis>/work/<stem>/sam3dinov3/processed_sam3dinov3_<data_hora>/<stem>/` (onde estão o `<stem>_sam3dinov3_predictions.json.gz` e os `<stem>_id_NN_*.csv`). O pipeline os grava na etapa `cameras:<stem>`, logo depois de `sam:<stem>` (também com `--stop-after sam`, para medir o vídeo antes da etapa 3D); `freekiki.py kick-cameras PASTA` e a CLI abaixo os gravam para uma execução existente.

| Arquivo | Conteúdo |
|---------|----------|
| `<stem>_frames.dlt3d` | CSV `frame,L1..L11`, uma linha por quadro do vídeo, todos finitos (`L12 = 1`): o DLT3D de cada quadro, em metros do referencial do campo |
| `<stem>_frames.dlt2d` | CSV `frame,L1..L8`, uma linha por quadro: a homografia do plano do solo (`z = 0`) da mesma câmera (`pixel = H [x, y, 1]`, `H[2,2] = 1`); é o que `QuickMeasureCalibration.from_dlt2d_sequence` lê |
| `<stem>_frames_cameras.csv` | `frame,source,rms_px,n_points,focal_px,from_frame`: de onde vem cada câmera e quão boa é |
| `<stem>_frames_cameras.json` | resumo: vídeo, número de quadros, tamanho, fps, contagem por fonte, centro óptico, quem fez e quando |

Fontes de uma câmera (coluna `source`; a cor é a das linhas do campo e da barra do painel por passos):

| Fonte | Cor | O que é | `rms_px` |
|-------|-----|---------|----------|
| `dlt3d` | verde | o DLT3D do FreeKiki do quadro, intacto | vazio |
| `solved` | amarelo | rotação e zoom ajustados aos pontos brutos do campo do quadro (centro óptico fixo) | erro de reprojeção desses pontos |
| `tracked` | ciano | levada pelo movimento da imagem a partir das câmeras conhecidas dos dois lados de um intervalo | incerteza do transporte (desacordo das âncoras mais desajuste, em quadratura) |
| `approximate` | laranja | cópia do melhor quadro mais próximo; o rastreador falhou | vazio |

O `rms_px` **não** é, portanto, um número único e comparável: em `solved` é um resíduo de ajuste em pontos reais, em `tracked` é o erro esperado da câmera transportada. Leia-o como "cerca de tantos pixels no campo" e compare quadros `tracked` só entre si.

CLI:

```bash
uv run --no-sync python -m vaila.kick_measure --run PASTA_KICK --write-cameras [--cameras-dir PASTA] [--no-track] [--camera reference|other]
uv run --no-sync python -m vaila.kick_measure --run PASTA_KICK --validate-tracking 5,10,20,30
uv run --no-sync vaila/freekiki.py kick-cameras PASTA_KICK [--camera reference|other]
```

- `--write-cameras` grava os quatro arquivos na pasta do SAM 3 e imprime os caminhos e a contagem por fonte; `--cameras-dir PASTA` grava lá.
- `--no-track` não rastreia os quadros aproximados e ignora as câmeras quadro a quadro salvas (o comportamento antigo, para comparar).
- `--validate-tracking K,K,...` é o teste de exclusão (padrão `5,10,20,30`): um quadro de câmera conhecida é tratado como desconhecido, rastreado a partir de um quadro bom `k` quadros adiante e comparado com a câmera real.
- A janela do FreeKiki tem o botão **Calibration per frame** (seção 5, **Open run and correct...**), que roda `kick-cameras` e espelha o comando no terminal.

**Validação (exclusão, clipe Fla x CAP, 169 quadros).** A `k = 10` quadros de uma câmera conhecida o erro mediano é de **4,5 px** nos pontos brutos do campo (a própria referência, a câmera real contra os mesmos pontos ruidosos, fica em cerca de 3,7 px: o piso do teste) e de **0,21 m** no solo (9 pixels da imagem contra a câmera real). Projetadas sobre o vídeo, as linhas do campo dos quadros 56-74 ficam a 1-4 px das linhas brancas. Neste clipe 42 quadros viraram `tracked` e 2 continuaram `approximate` (116-117). Limites da validação: um clipe, uma câmera de transmissão; o erro cresce com a distância às âncoras, e os quadros 100-140 são os mais fracos do clipe, então meça ali tendo em mente o `rms_px` do quadro; um corte de câmera não é detectado (um corte precisa da sua própria execução); um quadro que o rastreador não consegue seguir (poucos pareamentos, veja acima) continua `approximate`.

**Como o getpixelvideo usa os arquivos.** Abrir um vídeo, `--kick-run`, o botão **Kick SAM** e o **Load Tracking CSV** de um arquivo do SAM 3 instalam o `<stem>_frames.dlt3d` como calibração da sessão quando ainda não há uma (também `getpixelvideo.py -f VIDEO --calib-frames ARQUIVO`, ou **CALIB** `L` com o `.dlt3d` / `.dlt2d`). O HUD diz *Calibrated per frame* e avisa nos quadros `approximate`; o painel do chute e as linhas do campo pintam `tracked` de ciano. Veja [getpixelvideo (português)](getpixelvideo_pt.md).

## O que uma altura pressupõe

Uma câmera vê a bola ao longo de um raio, então uma altura sempre depende do plano que corta esse raio. O resultado nomeia o plano:

- **`7` Altura:** um plano vertical pelo ponto da base, perpendicular ao raio da câmera da base. Um poste ou uma pessoa está nele. A base precisa estar no gramado. O ± é a maior mudança da altura quando um dos cliques anda 1 px.
- **`8` Plano da linha:** o plano vertical pela linha que você marcou no gramado. A bola é **tomada nesse plano**, mas uma câmera não sabe o quanto a bola está afastada da linha. O ± de cada ponto é a mudança de `z` se o plano fosse deslocado 2 m para o lado. Com uma câmera de transmissão típica isso dá cerca de 0,1 m de altura por metro de erro lateral (2 m de erro lateral mudaram a altura em no máximo 0,37 m no clipe Vargas x Rossi).
- A altura de uma bola longe do plano escolhido é, portanto, um resultado de **modelo**: escolha a linha que acompanha o chute.

## Limites

- **Uma câmera.** Sem triangulação: o plano escolhido decide a altura.
- **Sem cortes de câmera.** O módulo não detecta um corte da câmera de transmissão; um corte move o centro óptico, então um corte precisa da sua própria execução (um trecho por execução).
- **Quadros com poucos pontos do campo** são rastreados pelo movimento da imagem (a precisão está em *Arquivos de calibração quadro a quadro*); só os quadros que o rastreador não consegue seguir ficam aproximados (cópia do quadro mais próximo).
- Validado em um único clipe real (Fla x CAP, o chute do Vargas x Rossi, câmera de transmissão); outras câmeras e jogos não foram verificados.
- O giro conta o que você clica; ele não detecta a marca sozinho.
- A calibração `dlt3d` do Quick Measure (11 parâmetros a partir de um vídeo) continua reservada; a `camera3d` é a que está em uso aqui.

## Arquivos gravados

O carregamento e a medição não gravam nada na pasta `kick_analysis_*`; só o passo **Export** grava ali (`processed_kickmeasure_<data_hora>/`, veja acima). **S** (salvar) no getpixelvideo grava ao lado do vídeo:

| Arquivo | Conteúdo |
|---------|----------|
| `<video>_markers.csv` | os marcadores do getpixelvideo, incluindo a trilha da bola p0 que você corrigiu |
| `processed_quickmeasure_<data_hora>/` | uma pasta por sessão, atualizada a cada salvamento |
| `<stem>_quickmeasure_points.csv` | cada clique: quadro, pixel e coordenadas reais |
| `<stem>_quickmeasure_calibration.csv` | a calibração (`camera3d`, pasta de origem) |
| `<stem>_quickmeasure_camera3d.csv` | uma linha por quadro do vídeo: `source` (`dlt3d` / `solved` / `tracked` / `approximate`), `approximate`, `rms_px`, `n_points` e a matriz 3x4 `P11`...`P34` |
| `<stem>_quickmeasure_results.csv` e `..._results_<tipo>.csv` | um resultado por linha; tipos `height`, `line_plane`, `plane_point`, `height_line`, `event`, `spin`, `arc` (mais `distance`, `angle`...), com `uncertainty`, `camera_source`, `approximate`, `plane`, `s`, `z`, `speed`, `accel`, `turns`, `rps`, `rpm`, `radius`, `angle_deg`, `sagitta` |
| `<stem>_quickmeasure_lineplane_series.csv` | modo `8`: x, y, z, s, velocidade, aceleração (e as componentes) por quadro, do primeiro ao último clique da bola (precisa do FPS e de 2 cliques em quadros diferentes) |
| `quickmeasure_report.html` | relatório didático: cada medida, o seu plano e o seu ± |
| `README_quickmeasure.txt` | lista dos arquivos e o comando para recalcular |

Os quatro arquivos de calibração quadro a quadro estão em *Arquivos de calibração quadro a quadro* (ficam ao lado das saídas do SAM 3, não aqui). O `--export` da CLI do módulo grava um JSON com o resumo, a câmera de cada quadro, os pixels da bola e o número de pessoas por quadro.

## API Python (congelada nas docstrings de `vaila/kick_measure.py`)

| Nome | Função |
|------|--------|
| `load_kick_measure_run(path, camera="reference")` | abre a pasta: `KickMeasureRun` (vídeo, fps, câmeras, trilha da bola, pessoas) |
| `load_people_info`, `load_pose_2d`, `pose_connections`, `field_segments_px` | pessoas (papel, time, quadros), poses 2D, ossos e as linhas do campo projetadas num quadro, para as camadas do painel |
| `metric_series(run, kind, person, ...)` | `MetricSeries` de uma métrica (`METRIC_KINDS`, `BODY_POINTS`) com `at(frame)` e `summary()` |
| `export_scene(run, out_dir, people, ...)`, `find_mkvis3d`, `mkvis3d_argv` | a exportação da tabela acima |
| `write_frame_calibration`, `read_frame_calibration`, `find_frame_calibration` | gravam, leem de volta e acham os arquivos quadro a quadro (`<stem>_frames.*`) |
| `track_approximate_frames`, `track_features`, `validate_tracking` | rastreamento quadro a quadro dos quadros aproximados e o seu teste de exclusão |
| `ground_homography(P)`, `CameraTrack.homography(frame)` | a homografia `z = 0` de uma câmera (a linha do `.dlt2d`) |
| `CameraTrack` | uma matriz 3x4 por quadro; `from_calibration`, `matrix`, `ground_point`, `height_base_top`, `counts` |
| `ground_point(P, u, v, z=0)` | pixel para o gramado (ou para um plano horizontal na altura `z`) |
| `height_base_top(P, base_uv, top_uv)` | altura de um objeto pelo plano vertical que passa pela sua base |
| `LinePlane` | plano vertical de uma linha no gramado: `locate`, `lateral_sensitivity`, `height_line` |
| `event_time`, `mean_speed`, `spin_rate` | tempo, velocidade média, giro (± 1 quadro) |
| `path_length`, `fit_arc`, `fit_parabola` | comprimento da curva e ajustes |
| `velocity_series`, `acceleration_series`, `speed_series` | cinemática de uma trilha, com suavização opcional |

## Módulos relacionados

| Módulo | Função |
|--------|--------|
| [getpixelvideo (português)](getpixelvideo_pt.md) | hospedeiro: carrega a pasta (`--kick-run`, `Tpl: K`, **Kick SAM**), encaixa o painel por passos, recebe os cliques |
| [quickmeasure](quickmeasure.md) | modos de medida e arquivos da sessão (calibração `camera3d`) |
| [freekiki](freekiki.md) | produz a pasta `kick_analysis_*`; `kick-measure`, `kick-cameras` e os botões **Measure kick** e **Calibration per frame** |
| [soccerfield](soccerfield.md) | o modelo de campo de 49 pontos ao qual as câmeras são ajustadas |

---

Parte do *vailá* - Multimodal Toolbox · [English](kick_measure.md) · [GitHub](https://github.com/vaila-multimodaltoolbox/vaila)
