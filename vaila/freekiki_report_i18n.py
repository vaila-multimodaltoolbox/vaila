"""
================================================================================
Script: freekiki_report_i18n.py - the texts of the press report, in Portuguese and English
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.17
Created: 27 October 2026
Update Date: 12 November 2026

Description:
    Every text the report, its figures, its animation and its interactive editor show outside
    the HTML templates: tier names and warnings, figure titles and legends, the overlay labels
    of the animation, the anchor labels, the editor labels and the number format.
    ``tr(lang, key)`` returns a text; ``fmt_num`` writes a number with the decimal mark of the
    language; ``localize_anchor_label`` translates an anchor label written by the dialog.
"""

from __future__ import annotations

import math

LANGS = ("pt", "en")

_STRINGS: dict[str, dict[str, str]] = {
    "pt": {
        "tier_alta": "alta",
        "tier_média": "média",
        "tier_baixa": "baixa",
        "tier_text_alta": "A trajetória reproduz as marcações e as posições físicas conferem com a imagem.",
        "tier_text_média": "A trajetória reproduz as marcações; algumas posições físicas diferem de 0,3 a 1 m.",
        "tier_text_baixa": (
            "A trajetória reproduz as marcações da bola, mas as posições físicas do pé, da barreira e do "
            "goleiro diferem de 1 m ou mais. Trate altura, curva e velocidade como estimativas."
        ),
        "warn": (
            "<b>Confiança {tier} no ajuste.</b> {tier_text} O maior desvio entre a trajetória e {reference} "
            "é de {worst} m. Veja a seção “O que estes números não dizem”."
        ),
        "kicker_of": "do pé de {name}",
        "kicker_default": "do pé do jogador que bateu a falta",
        "kicker_default_pk": "do pé do cobrador do pênalti",
        "kicker_default_op": "do pé do jogador que chutou",
        "kicker_name_default": "o jogador que bateu a falta",
        "wall_above": "acima",
        "wall_below": "abaixo",
        "spin_vertical": "quase vertical",
        "spin_tilted": "inclinado",
        "date_format": "{d:02d}/{m:02d}/{y}",
        "default_title": "A física de um chute de falta",
        "default_title_pk": "A física de um pênalti",
        "default_title_op": "A física de um chute a gol",
        "default_subtitle": "Reconstrução 3D a partir de transmissão, feita com software livre vailá",
        "fig_top_title": "Vista de cima: {d} m em {t} s",
        "fig_top_x": "x no campo (m)",
        "fig_top_y": "y no campo (m)",
        "fig_top_cbar": "velocidade da bola (km/h)",
        "fig_start": "chute",
        "fig_entry": "entrada",
        "fig_entry_pk": "último quadro",
        "fig_hands_start": "mãos do goleiro: início",
        "fig_hands_end": "mãos do goleiro: fim",
        "fig_profile_title": "De lado: sobe até {z} m",
        "fig_profile_x": "distância horizontal desde o chute (m)",
        "fig_profile_y": "altura da bola (m)",
        "fig_wall_label": "barreira\n(1,8 m, hipótese)",
        "fig_crossbar": "travessão 2,44 m",
        "fig_traj": "trajetória ajustada",
        "fig_time_x": "tempo depois do chute (s)",
        "fig_time_speed": "bola (km/h)",
        "fig_time_cd": "coeficiente de arrasto Cd",
        "fig_time_re": "Reynolds (×10⁵)",
        "fig_time_title": "A bola perde velocidade no ar",
        "fig_curve_line": "linha reta do chute até a entrada",
        "fig_curve_line_pk": "linha reta do chute até o último quadro",
        "fig_curve_x": "distância ao longo da linha reta (m)",
        "fig_curve_y": "afastamento lateral (m)",
        "fig_curve_title": "A curva: até {dev} m para o lado da linha reta",
        "fig_forces_title": "Forças sobre a bola em três instantes",
        "fig_force_y": "aceleração (m/s²)",
        "fig_force_grav": "gravidade",
        "fig_force_drag": "arrasto",
        "fig_force_magnus": "Magnus",
        "fig_when_after": "logo após o chute",
        "fig_when_wall": "ao passar a barreira",
        "fig_when_mid": "no meio do voo",
        "fig_when_end": "perto da entrada",
        "fig_res_title": "Quanto a trajetória se afasta das marcações",
        "fig_res_x": "quadro",
        "fig_res_y": "erro na imagem (px)",
        "fig_res_mean": "média {rms} px",
        "vid_prep": "Preparação do chute",
        "vid_kick": "CHUTE",
        "vid_rising": "Bola subindo",
        "vid_wall": "Bola sobre a barreira",
        "vid_goal": "Entrada no gol",
        "vid_flight": "Bola em voo",
        "vid_last": "Último quadro",
        "vid_onset": "INÍCIO DA RESPOSTA",
        "vid_save": "DEFESA",
        "vid_after": "após o chute",
        "vid_speed": "velocidade (modelo)",
        "vid_height": "altura",
        "vid_re": "Reynolds",
        "vid_cd": "Cd (arrasto)",
        "vid_spin": "rotação (inferida)",
        "vid_caption": "{title}  ·  câmera lenta ≈ {slow}×  ·  reconstrução 3D projetada sobre a transmissão  ·  vailá",
        "vid_minimap": "vista de cima",
        "vid_force_gravity": "gravidade",
        "vid_force_drag": "arrasto",
        "vid_force_magnus": "Magnus",
        "vid_anchor_kicker": "pé do chutador",
        "vid_anchor_shoulder": "ombro",
        "vid_anchor_wall": "barreira",
        "vid_anchor_keeper": "mãos do goleiro",
        "vid_anchor_hands_start": "mãos do goleiro: início",
        "vid_anchor_hands_end": "mãos do goleiro: fim",
        "anchor_start": "início: bola no pé",
        "anchor_middle": "meio: bola no ar",
        "anchor_barrier": "barreira: bola sobre a barreira",
        "anchor_contact": "início: contato do pé",
        "anchor_wall_plane": "barreira: plano a 9,15 m",
        "anchor_entry": "chegada: entrada no gol",
        "anchor_end": "chegada: mãos do goleiro",
        "anchor_goal": "chegada: gol",
        "anchor_kicker": "pé do chutador",
        "anchor_shoulder": "ombro",
        "mean_range": "(faixa de {lo} a {hi} km/h)",
        "time_note": (
            "Tempo de voo: do contato do pé, que cai entre dois quadros (quadro {kick}), até a defesa no "
            "quadro {save}, a {fps} quadros por segundo (1 quadro = {ms} ms): <b>{flight} s</b>. Contando só "
            "quadros inteiros seriam {n} quadros = {counted} s e {counted_kmh} km/h. A velocidade média é a "
            "distância da marca até as mãos dividida por esse tempo, com incerteza de cerca de ±{sigma} km/h "
            "(um quarto de quadro em cada ponta e 0,3 m na distância). A saída do pé (cerca de {exit_kmh} km/h) "
            "vem do modelo de voo e não é medida."
        ),
        "quote_distance": "A bola percorreu {dist} metros em {flight} segundos.",
        "quote_mean": "Na média, a bola andou a cerca de {mean} km/h do pé até a entrada.",
        "quote_height": "A bola subiu até {zmax} metros de altura, e a barreira tem cerca de 1,8 metro.",
        "quote_mean_pk": "Na média, a bola andou a cerca de {mean} km/h do pé até o último quadro analisado.",
        "quote_height_pk": "A bola subiu até {zmax} metros de altura; o travessão fica a 2,44 metros.",
        "penalty_tier_text_baixa": (
            "A trajetória reproduz as marcações da bola, mas as posições físicas do pé e do goleiro "
            "diferem de 1 m ou mais. Trate altura, curva e velocidade como estimativas."
        ),
        "hands_lead": "No último quadro, as mãos do goleiro estavam a <b>{gap} metros</b> da bola.",
        "hands_moved": (
            "O goleiro não é um ponto fixo: as mãos dele (a média dos dois punhos) estavam no ponto marcado como "
            "“início” no momento do chute e no ponto marcado como “fim” no último quadro. A distância entre os "
            "dois pontos é de <b>{travel} metros</b>."
        ),
        "hands_near": (
            "No último quadro, as mãos ficaram a <b>{gap} metros</b> da bola, uma distância da ordem do erro desta "
            "medição: o punho não é a mão, e a posição de ambos vem de uma só câmera. A reconstrução não separa um "
            "toque de uma bola que passou perto."
        ),
        "hands_far": "No último quadro, as mãos ficaram a <b>{gap} metros</b> da bola.",
        "hands_missing": (
            "As mãos do goleiro não puderam ser medidas neste lance (faltam os punhos dele na reconstrução 3D "
            "ou o mapa de jogadores)."
        ),
        "cameras_note_default": "Não há medição com várias câmeras.",
        "cameras_note_manual": (
            "O voo vem de uma câmera principal; só o ponto em que a bola chegou às mãos foi confirmado com a "
            "câmera de trás."
        ),
        "keeper_one_camera": (
            "O goleiro vem de uma câmera só: a posição lateral dele é aproximada. A bola e o ponto da defesa "
            "seguem as duas câmeras."
        ),
        "hands_manual": (
            "O ponto “fim” é a chegada da bola nas mãos, confirmada com duas câmeras (o lado vem da câmera de trás e "
            "a profundidade e a altura, da principal). O ponto “início” é a média dos punhos na pose de uma câmera, "
            "no lugar que a câmera de trás confirma no começo do movimento."
        ),
        "hands_limit": (
            "<b>As mãos do goleiro são os punhos.</b> O modelo de pose não tem um ponto da mão, e a posição 3D de "
            "cada punho vem de uma só câmera, com erro de algumas dezenas de centímetros."
        ),
        "quote_measured": "Tudo isso foi medido a partir de uma única transmissão, usando as linhas do campo como régua.",
        "incoming_lead": "A bola chegava rolando a cerca de <b>{speed} km/h</b>.",
        "incoming_text": (
            "Antes do chute, a bola chegava rolando a cerca de <b>{speed} km/h</b>. O chute mudou a "
            "direção dela em <b>{turn}°</b> (vista de cima) e a lançou a cerca de {launch} km/h."
        ),
        "incoming_unseen": (
            "A bola já voava no primeiro quadro do vídeo: o chute aconteceu nele ou logo antes, e a "
            "bola que chegou não aparece na imagem."
        ),
        "contact_limit_kink": (
            "<b>O contato foi achado na quebra da trajetória da bola rolando</b>: duas retas no chão, "
            "antes e depois do chute. A incerteza é de um quadro."
        ),
        "contact_limit_first_frame": (
            "<b>O chute é o primeiro quadro do vídeo ou acontece logo antes.</b> A saída da bola tem "
            "até um quadro de incerteza, e não há bola antes do contato para medir."
        ),
        "contact_limit_manual": "<b>O quadro do contato foi informado à mão.</b>",
        "contact_limit_rest": "<b>O contato é o primeiro quadro em que a bola deixa o repouso.</b>",
        "quote_curve": "A bola fez uma curva de até {dev} metro para o lado da linha reta do chute.",
        "footer_generated": "Gerado em {date}. Animação em câmera lenta: {animation}.",
        "footer_tier": "Confiança do ajuste: {tier} · erro médio na imagem {rms} px · pior desvio nas referências {worst} m.",
        "top_kicker": "chutador (id {id})",
        "top_wall": "barreira (ids {a} e {b})",
        "top_wall_one": "barreira (id {a})",
        "top_keeper": "goleiro (id {id})",
        "top_note": (
            "Jogadores no chão (média dos tornozelos): chutador id {kicker}, barreira ids {wall}, "
            "goleiro id {keeper}. {how}"
        ),
        "top_note_pk": (
            "Jogadores no chão (média dos tornozelos): cobrador id {kicker} e goleiro id {keeper}. "
            "Pênalti: sem barreira. {how}"
        ),
        "top_note_op": (
            "Jogadores no chão (média dos tornozelos): chutador id {kicker} e goleiro id {keeper}. "
            "Lance de jogo: sem barreira. {how}"
        ),
        "top_how_auto": "Identificados pelo ponto mais próximo; --kicker-id, --wall-ids e --keeper-id trocam.",
        "top_how_manual": "Informados por --kicker-id, --wall-ids e --keeper-id.",
        "top_how_map": (
            "Mesmo ID nas duas câmeras: goleiro 0, cobrador 1, demais pela distância do pé à bola no contato."
        ),
        "top_none": "Sem jogadores identificados no run (falta poses_3d_on_field.csv).",
        "ref_anchors": "um ponto físico do lance",
        "ref_binocular": "a posição triangulada pelas duas câmeras",
        "ref_auto": "um ponto de referência da imagem",
        "method_anchors": (
            "Cada quadro da transmissão tem uma calibração da câmera, obtida pelas linhas pintadas do campo, "
            "de medidas conhecidas. Com ela, a posição da bola em cada quadro vira uma posição real no campo. "
            "A trajetória física (a gravidade, o arrasto e o Magnus) é ajustada de modo a reproduzir essas "
            "posições com um erro médio de <b>{rms} pixel</b> na imagem. Além disso, três pontos físicos do lance "
            "entram como referência: a bola começa no pé do chutador, passa a 1,8 m de altura a 0,3 m do ombro de "
            "um jogador da barreira, e chega às mãos do goleiro. Essas posições não são reproduzidas exatamente: a "
            "diferença está na seção 6."
        ),
        "method_binocular": (
            "Cada quadro da transmissão tem uma calibração da câmera, obtida pelas linhas pintadas do campo. A bola "
            "foi vista por <b>duas câmeras ao mesmo tempo</b>: em cada quadro, a posição é triangulada pelos dois "
            "pontos de vista, e a trajetória física (a gravidade, o arrasto e o Magnus) é ajustada aos pixels das "
            "duas câmeras, com um erro médio de <b>{rms} pixel</b>. Assim a altura da bola é medida, e não suposta. "
            "A linha da barreira é marcada no ponto a 9,15 m do chute, onde a bola está nesse ajuste."
        ),
        "refs_anchors": (
            "<b>Os pontos físicos de referência podem diferir.</b> O pé, a barreira e o goleiro são escolhidos a "
            "partir dos jogadores reconstruídos, e não medidos diretamente no campo. Nesta reconstrução, a diferença "
            "entre o ajuste e os pontos de referência é de {worst} m no pior caso."
        ),
        "refs_binocular": (
            "<b>A posição é triangulada, não medida no campo.</b> Cada ponto vem de duas câmeras calibradas pelas "
            "linhas do campo. A diferença entre o ajuste físico e a triangulação de cada quadro é de {worst} m no "
            "pior caso (percentil 90). Os jogadores não entram como referência."
        ),
        "wall_anchors": "<b>A altura da barreira é uma hipótese</b> de 1,8 metro, e a trajetória sobre ela depende dessa escolha.",
        "wall_binocular": (
            "<b>A altura da barreira é uma hipótese</b> de 1,8 metro. A altura da bola no plano de 9,15 m é medida "
            "pelas duas câmeras, mas isso não diz se ela passou sobre a barreira real."
        ),
        "ed_disabled": "Editor indisponível para a reconstrução com duas câmeras.",
        "ed_disabled_auto": "Editor indisponível: ajuste do contato e da chegada pela GUI/CLI do FreeKiki.",
        "method_auto": (
            "Cada quadro da transmissão tem uma calibração da câmera, obtida pelas linhas pintadas do campo. Com "
            "ela, a bola vira uma posição no campo. Uma só câmera não mede a profundidade, então a altura da bola "
            "vem da física: a trajetória (gravidade, arrasto e Magnus) é ajustada aos pixels, com erro médio de "
            "<b>{rms} pixel</b>. O lance começa no <b>contato do pé</b>, quando a bola sai do repouso; antes disso "
            "ela está parada e não entra no cálculo. Uma entrada marcada usa a homografia dos quatro cantos confirmados da baliza. "
            "A chegada à mão do goleiro exige confirmação explícita; sem observação ou cruzamento "
            "reconhecido, a chegada permanece indeterminada."
        ),
        "refs_auto": (
            "<b>Os pontos de referência vêm da própria imagem.</b> O repouso é a bola parada antes do contato, e a "
            "entrada no gol é medida pela homografia do plano da baliza. A diferença entre o ajuste e os pontos de "
            "referência é de {worst} m no pior caso."
        ),
        "wall_auto": (
            "<b>A altura da barreira é uma hipótese</b> de 1,8 metro. A linha da barreira é marcada no ponto a "
            "9,15 m do chute, onde a bola está nesse ajuste monocular; uma câmera não diz se ela passou sobre a "
            "barreira real."
        ),
        "ed_title": "Ajustar as âncoras e refazer o cálculo",
        "ed_intro": (
            "Mude as posições (metros no campo) ou os quadros dos pontos de referência e clique em "
            "Recalcular. O cálculo roda neste navegador, com o mesmo modelo do vailá. Os números do texto "
            "acima não mudam: eles vêm do cálculo original."
        ),
        "ed_col_label": "ponto",
        "ed_col_frame": "quadro",
        "ed_col_x": "x (m)",
        "ed_col_y": "y (m)",
        "ed_col_z": "z (m)",
        "ed_col_sigma": "σ (m)",
        "ed_col_remove": "remover",
        "ed_add": "+ Adicionar âncora",
        "ed_recalc": "Recalcular",
        "ed_reset": "Restaurar originais",
        "ed_running": "Calculando…",
        "ed_min": "São necessárias ao menos 3 âncoras.",
        "ed_fail": "O ajuste não convergiu. Tente outros quadros ou valores.",
        "ed_done": "Pronto.",
        "ed_out_rms": "Erro médio na imagem",
        "ed_out_max": "Maior erro na imagem",
        "ed_out_anchor": "Desvio em cada âncora",
        "ed_out_kick": "Instante do chute",
        "ed_out_spin": "Rotação inferida",
        "ed_out_peak": "Velocidade máxima",
        "ed_out_dist": "Distância horizontal",
        "ed_out_zmax": "Altura máxima",
        "ed_out_tier": "Confiança",
        "ed_json_title": "Âncoras em JSON (para o botão Rebuild ball)",
        "ed_download": "Baixar JSON",
        "ed_top": "Vista de cima recalculada",
        "ed_side": "Perfil recalculado",
        "ed_note": (
            "Os pontos antigos (sem σ) entram com σ 0,3 m. A distância usa o quadro inicial do chute e o "
            "último quadro com marcação."
        ),
        # --- open-play dribble / pass report ---
        "op_kicker": "Biomecânica no futebol · relatório para imprensa",
        "op_title_dribble": "A física de um drible 1 contra N",
        "op_title_pass": "A física de um passe",
        "op_subtitle": "Cinemática no plano do campo a partir de transmissão, feita com software livre vailá",
        "op_lead_dribble": (
            "O driblador (id {dribbler}) percorreu <b>{path} m</b> com <b>{cod}</b> mudanças de direção "
            "e curvatura média de <b>{curv} rad/m</b>, diante de {n} marcador(es)."
        ),
        "op_lead_pass": (
            "Passe do id {passer} ao id {receiver}: a bola percorreu cerca de <b>{length} m</b> "
            "em <b>{flight} s</b> entre o toque de saída e a recepção (heurística)."
        ),
        "op_card_ball_path": "caminho da bola no plano",
        "op_card_ball_vmax": "velocidade máxima da bola",
        "op_card_ball_accel": "pico de aceleração (m/s²)",
        "op_card_drib_path": "caminho do driblador",
        "op_card_drib_vmax": "velocidade máxima do driblador",
        "op_card_drib_cod": "mudanças de direção (COD)",
        "op_card_drib_curv": "curvatura média (rad/m)",
        "op_card_drib_accel": "pico de aceleração (m/s²)",
        "op_card_drib_decel": "pico de desaceleração (m/s²)",
        "op_card_pass_len": "caminho da bola no passe",
        "op_card_pass_flight": "tempo saída → recepção",
        "op_card_pass_pair": "distância entre jogadores na saída",
        "op_h_top": "Vista de cima",
        "op_h_speed": "Velocidades no tempo",
        "op_h_dist": "Distâncias no tempo",
        "op_h_science": "Leitura científica",
        "op_h_markers": "Marcadores",
        "op_cap_top": "Trajetórias no plano do campo (metros).",
        "op_cap_speed": "Velocidade instantânea (km/h) a partir das posições no plano.",
        "op_cap_dist": "Distâncias interpessoais ou bola–jogador (m).",
        "op_fig_top_title": "Vista de cima: trajetórias no campo",
        "op_fig_x": "x no campo (m)",
        "op_fig_y": "y no campo (m)",
        "op_fig_t": "tempo (s)",
        "op_fig_speed": "velocidade (km/h)",
        "op_fig_speed_title": "Velocidade no plano",
        "op_fig_dist": "distância (m)",
        "op_fig_dist_drib_title": "Distância driblador–marcador",
        "op_fig_dist_pass_title": "Distâncias do passe",
        "op_legend_ball": "bola",
        "op_legend_dribbler": "driblador",
        "op_legend_marker": "marcador {id}",
        "op_legend_passer": "passador",
        "op_legend_receiver": "recebedor",
        "op_legend_passer_ball": "passador–bola",
        "op_legend_receiver_ball": "recebedor–bola",
        "op_legend_passer_receiver": "passador–recebedor",
        "op_th_id": "id",
        "op_th_min_d": "mín. (m)",
        "op_th_rel_v": "v rel. no mín. (m/s)",
        "op_th_cod": "COD",
        "op_th_path": "caminho (m)",
        "op_no_markers": "Nenhum marcador informado.",
        "op_science_dribble": (
            "No enquadramento predador–presa de Wilson e Santiago, o sucesso no um-contra-um depende "
            "mais da capacidade de mudar de direção e manter velocidade em trajetórias curvas do que "
            "do sprint em linha reta. Distância interpessoal e velocidade relativa (Corrêa / Duarte) "
            "descrevem a pressão do marcador sobre o driblador."
        ),
        "op_science_pass": (
            "A linha de Wilson–Santiago sobre passe e controle trata a combinação de velocidade, "
            "precisão e agilidade no passe como preditora de desempenho em jogos reduzidos. Aqui "
            "medimos o clip: caminho da bola, distâncias entre os dois jogadores e a bola, e a "
            "janela heurística entre toque de saída e recepção."
        ),
        "op_limit": (
            "Métricas operacionais no plano do campo a partir de tracks FreeKiki; não substituem "
            "testes de skill fechada em circuito (curvatura 0–1,37 rad/m) nem baterias de passe em "
            "rebound board."
        ),
        "op_ref1": "Wilson et al. (2018). J Sports Sci — dribbling speed along curved paths (PMID 30470166).",
        "op_ref2": "Wilson, Santiago et al. (2026). Int J Sports Sci Coach — dribbling and passing vs 3v3 success.",
        "op_ref3": "Corrêa et al. / Duarte et al. — distância interpessoal e velocidade relativa no 1v1.",
        # --- pass report: timing, receiver run, shot link ---
        "op_lead_pass_main": (
            "Passe do id {passer} para o id {receiver}: a bola saiu a cerca de <b>{v0} km/h</b> e "
            "chegou ao pé do recebedor <b>{flight} s</b> depois, a <b>{dist} m</b> de onde estava "
            "o passador."
        ),
        "op_lead_pass_run_before": (
            " Quando a bola saiu, o id {receiver} já corria a <b>{vr} km/h</b>: a corrida começou "
            "antes do passe, pelo menos <b>{lead} s</b> antes (estimativa)."
        ),
        "op_lead_pass_run_onset": (
            " O id {receiver} começou a correr <b>{lead} s</b> {when} da saída da bola."
        ),
        "op_when_before": "antes",
        "op_when_after": "depois",
        "op_lead_shot": (
            " Em seguida, o id {receiver} finalizou a <b>{goal} m</b> da linha do gol, com a bola "
            "saindo do pé a cerca de <b>{launch} km/h</b>."
        ),
        "op_tag_measured": "medido",
        "op_tag_estimate": "estimativa",
        "op_card_pass_dist": "do passador até o ponto de chegada",
        "op_card_pass_v0": "velocidade da bola na saída",
        "op_card_pass_time": "tempo da saída ao 1º toque",
        "op_card_pass_v1": "velocidade da bola na chegada",
        "op_card_pass_vmean": "velocidade média da bola no passe",
        "op_card_pair_release": "passador–recebedor na saída",
        "op_card_recv_v0": "recebedor quando a bola saiu",
        "op_card_recv_lead": "o recebedor começou a correr antes do passe (pelo menos)",
        "op_card_recv_onset": "início da corrida em relação à saída da bola",
        "op_card_recv_peak": "velocidade máxima do recebedor",
        "op_card_recv_disp": "deslocamento do recebedor durante o passe",
        "op_card_recv_v1": "recebedor no 1º toque",
        "op_card_passer_v0": "passador na hora do passe",
        "op_h_pass_ball": "A bola",
        "op_h_pass_receiver": "O recebedor",
        "op_h_table": "Posição, velocidade e tempo nos instantes-chave",
        "op_table_note": (
            "Posições em metros no sistema do campo (x ao longo do campo, y na largura; o centro "
            "do campo é a origem). t = 0 na saída da bola."
        ),
        "op_th_instant": "instante",
        "op_th_t": "t (s)",
        "op_th_ball_xy": "bola (x, y)",
        "op_th_vball": "bola (km/h)",
        "op_th_player_xy": "id {id} (x, y)",
        "op_th_vplayer": "id {id} (km/h)",
        "op_row_release": "saída da bola",
        "op_row_first": "1º quadro (logo após a saída)",
        "op_science_pass_timed": (
            "A linha de Wilson–Santiago sobre passe e controle trata a combinação de velocidade, "
            "precisão e agilidade no passe como preditora de desempenho em jogos reduzidos. Aqui "
            "medimos o lance: a velocidade da bola na saída e na chegada, a distância do passador "
            "até o ponto de chegada e o deslocamento do recebedor enquanto a bola viaja."
        ),
        "op_row_mid": "meio do passe",
        "op_row_arrival": "1º toque do recebedor",
        "op_row_shot": "chute (outro trecho)",
        "op_h_shot": "Depois do passe: o chute",
        "op_card_shot_launch": "velocidade na saída do pé (arredondada)",
        "op_card_shot_goal": "distância do chute à linha do gol",
        "op_card_shot_from_arrival": "do 1º toque até o ponto do chute",
        "op_card_shot_from_passer": "do passador até o ponto do chute",
        "op_card_shot_mean": "velocidade média do chute",
        "op_card_shot_height": "altura máxima da bola no chute",
        "op_shot_text": (
            "O chute foi filmado em outro trecho do vídeo, por outra câmera: o tempo entre a "
            "recepção e o chute não é medido. Pela distância de {d} m entre o 1º toque e o ponto "
            "do chute, e pelas velocidades da bola e do recebedor no 1º toque, esse tempo fica "
            "entre {g0} e {g1} s (estimativa). Do chute até o último quadro analisado, a bola "
            "percorreu {path} m em {fl} s. Velocidade: {launch} km/h na saída do pé (modelo de voo) "
            "e {mean} km/h em média até o último quadro, pois a bola só perde velocidade depois "
            "do chute; na TV, cerca de {head} km/h."
        ),
        "op_cap_top_pass": (
            "Vista de cima, em metros. Quadrados: posições na saída da bola. Círculos: no 1º toque. "
            "Estrela: ponto de chegada. Tracejado preto: distância do passador até a chegada."
        ),
        "op_cap_top_shot": " Triângulo vermelho: ponto do chute, com a direção da bola.",
        "op_cap_speed_pass": (
            "Velocidade no plano (km/h). t = 0 na saída da bola; linha verde no 1º toque."
        ),
        "op_cap_dist_pass": "Distâncias bola–jogador e entre os jogadores (m); t = 0 na saída da bola.",
        "op_fig_t_rel": "tempo desde a saída da bola (s)",
        "op_legend_release": "saída da bola",
        "op_legend_arrival": "1º toque",
        "op_legend_arrival_pt": "chegada",
        "op_legend_shot": "chute",
        "op_legend_passer_id": "passador (id {id})",
        "op_legend_receiver_id": "recebedor (id {id})",
        "op_fig_already_running": "id {id} já corria a {v} km/h",
        "op_science_pass_run": (
            " O momento em que o recebedor começa a correr, em relação à saída da bola, é uma "
            "leitura de antecipação: quem corre antes do passe ganha metros sobre o marcador, e o "
            "passe precisa ser forte e preciso o bastante para chegar ao espaço para onde ele corre."
        ),
        "op_limit_pass_fps": "Vídeo a {fps} quadros por segundo (um quadro a cada {dt} ms).",
        "op_limit_pass_release_fit": (
            "A bola já estava em movimento no primeiro quadro: o instante da saída ({rel} s antes do "
            "primeiro quadro) vem do ajuste do voo da bola e confere com a extrapolação para trás "
            "({back} s)."
        ),
        "op_limit_pass_release_heur": (
            "O instante da saída é estimado pela menor distância passador–bola seguida do aumento "
            "da velocidade da bola."
        ),
        "op_limit_pass_arrival_user": "O 1º toque do recebedor foi marcado no vídeo (quadro {f}).",
        "op_limit_pass_arrival_auto": (
            "O 1º toque é a menor distância entre o recebedor e a bola (quadro {f}); confira no vídeo."
        ),
        "op_limit_pass_tracks": (
            "A bola é a trajetória no chão (raio da câmera cortado na altura da bola). As posições "
            "dos jogadores são o ponto no chão sob a caixa de detecção, não o pé, com remoção de "
            "pontos fora da curva e filtro passa-baixa de 2 Hz."
        ),
        "op_limit_pass_run_bound": (
            "Como o recebedor já corria no primeiro quadro, o início da corrida não aparece no "
            "vídeo: o valor mostrado é um limite inferior, supondo aceleração máxima de {a} m/s² "
            "(fica entre {l0} s, com {ahi} m/s², e {l1} s, com {alo} m/s²)."
        ),
        "op_limit_pass_shot": (
            "O passe e o chute são trechos diferentes do vídeo; o tempo entre eles não foi medido."
        ),
    },
    "en": {
        "tier_alta": "high",
        "tier_média": "medium",
        "tier_baixa": "low",
        "tier_text_alta": "The trajectory reproduces the marked points, and the physical positions agree with the image.",
        "tier_text_média": "The trajectory reproduces the marked points; some physical positions differ by 0.3 to 1 m.",
        "tier_text_baixa": (
            "The trajectory reproduces the marked points of the ball, but the physical positions of the foot, "
            "the wall and the goalkeeper differ by 1 m or more. Treat height, curve and speed as estimates."
        ),
        "warn": (
            "<b>Fit confidence: {tier}.</b> {tier_text} The largest deviation between the trajectory and "
            "{reference} is {worst} m. See “What these numbers do not say”."
        ),
        "kicker_of": "from the foot of {name}",
        "kicker_default": "from the foot of the kicker",
        "kicker_default_pk": "from the foot of the penalty taker",
        "kicker_default_op": "from the foot of the player who shot",
        "kicker_name_default": "the kicker",
        "wall_above": "above",
        "wall_below": "below",
        "spin_vertical": "nearly vertical",
        "spin_tilted": "tilted",
        "date_format": "{m:02d}/{d:02d}/{y}",
        "default_title": "The physics of a free kick",
        "default_title_pk": "The physics of a penalty kick",
        "default_title_op": "The physics of a shot at goal",
        "default_subtitle": "3D reconstruction from broadcast footage, made with free software vailá",
        "fig_top_title": "Top view: {d} m in {t} s",
        "fig_top_x": "x on the pitch (m)",
        "fig_top_y": "y on the pitch (m)",
        "fig_top_cbar": "ball speed (km/h)",
        "fig_start": "kick",
        "fig_entry": "entry",
        "fig_entry_pk": "last frame",
        "fig_hands_start": "goalkeeper's hands: start",
        "fig_hands_end": "goalkeeper's hands: end",
        "fig_profile_title": "Side view: rises to {z} m",
        "fig_profile_x": "horizontal distance from the kick (m)",
        "fig_profile_y": "ball height (m)",
        "fig_wall_label": "wall\n(1.8 m, hypothesis)",
        "fig_crossbar": "crossbar 2.44 m",
        "fig_traj": "fitted trajectory",
        "fig_time_x": "time after the kick (s)",
        "fig_time_speed": "ball (km/h)",
        "fig_time_cd": "drag coefficient Cd",
        "fig_time_re": "Reynolds (×10⁵)",
        "fig_time_title": "The ball loses speed in the air",
        "fig_curve_line": "straight line from the kick to the entry",
        "fig_curve_line_pk": "straight line from the kick to the last frame",
        "fig_curve_x": "distance along the straight line (m)",
        "fig_curve_y": "sideways offset (m)",
        "fig_curve_title": "The curve: up to {dev} m to the side of the straight line",
        "fig_forces_title": "Forces on the ball at three moments",
        "fig_force_y": "acceleration (m/s²)",
        "fig_force_grav": "gravity",
        "fig_force_drag": "drag",
        "fig_force_magnus": "Magnus",
        "fig_when_after": "right after the kick",
        "fig_when_wall": "passing the wall",
        "fig_when_mid": "mid-flight",
        "fig_when_end": "near the entry",
        "fig_res_title": "How far the trajectory is from the marked points",
        "fig_res_x": "frame",
        "fig_res_y": "error in the image (px)",
        "fig_res_mean": "mean {rms} px",
        "vid_prep": "Preparing the kick",
        "vid_kick": "KICK",
        "vid_rising": "Ball rising",
        "vid_wall": "Ball over the wall",
        "vid_goal": "Ball enters the goal",
        "vid_flight": "Ball in flight",
        "vid_last": "Last frame",
        "vid_onset": "RESPONSE ONSET",
        "vid_save": "SAVE",
        "vid_after": "after the kick",
        "vid_speed": "speed (model)",
        "vid_height": "height",
        "vid_re": "Reynolds",
        "vid_cd": "Cd (drag)",
        "vid_spin": "spin (inferred)",
        "vid_caption": "{title}  ·  slow motion ≈ {slow}×  ·  3D reconstruction projected on the broadcast  ·  vailá",
        "vid_minimap": "top view",
        "vid_force_gravity": "gravity",
        "vid_force_drag": "drag",
        "vid_force_magnus": "Magnus",
        "vid_anchor_kicker": "kicker's foot",
        "vid_anchor_shoulder": "shoulder",
        "vid_anchor_wall": "wall",
        "vid_anchor_keeper": "keeper's hands",
        "vid_anchor_hands_start": "goalkeeper's hands: start",
        "vid_anchor_hands_end": "goalkeeper's hands: end",
        "anchor_start": "start: ball on kick foot",
        "anchor_middle": "middle: ball in the air",
        "anchor_barrier": "barrier: ball over the wall",
        "anchor_contact": "start: foot contact",
        "anchor_wall_plane": "wall: the 9.15 m plane",
        "anchor_entry": "end: goal entry",
        "anchor_end": "end: keeper's hands",
        "anchor_goal": "end: goal",
        "anchor_kicker": "kicker's foot",
        "anchor_shoulder": "shoulder",
        "mean_range": "(range {lo} to {hi} km/h)",
        "time_note": (
            "Flight time: from the foot contact, which falls between two frames (frame {kick}), to the save "
            "at frame {save}, at {fps} frames per second (1 frame = {ms} ms): <b>{flight} s</b>. Counting whole "
            "frames only would give {n} frames = {counted} s and {counted_kmh} km/h. The mean speed is the "
            "distance from the spot to the hands over that time, with an uncertainty of about ±{sigma} km/h "
            "(a quarter of a frame at each end and 0.3 m in the distance). The exit speed (about {exit_kmh} km/h) "
            "comes from the flight model and is not measured."
        ),
        "quote_distance": "The ball travelled {dist} metres in {flight} seconds.",
        "quote_mean": "On average, the ball moved at about {mean} km/h from the foot to the entry.",
        "quote_height": "The ball rose to {zmax} metres, and the wall is about 1.8 metres high.",
        "quote_mean_pk": "On average, the ball moved at about {mean} km/h from the foot to the last frame analysed.",
        "quote_height_pk": "The ball rose to {zmax} metres; the crossbar is 2.44 metres high.",
        "penalty_tier_text_baixa": (
            "The trajectory reproduces the marked points of the ball, but the physical positions of the foot "
            "and the goalkeeper differ by 1 m or more. Treat height, curve and speed as estimates."
        ),
        "hands_lead": "In the last frame, the goalkeeper's hands were <b>{gap} metres</b> from the ball.",
        "hands_moved": (
            "The goalkeeper is not a fixed point: the hands (the mean of the two wrists) were at the point marked "
            "“start” when the ball was kicked and at the point marked “end” in the last frame. The distance between "
            "the two points is <b>{travel} metres</b>."
        ),
        "hands_near": (
            "In the last frame the hands were <b>{gap} metres</b> from the ball, a distance of the order of the "
            "error of this measurement: a wrist is not a hand, and the position of both comes from a single "
            "camera. The reconstruction does not tell a touch from a ball that passed close."
        ),
        "hands_far": "In the last frame the hands were <b>{gap} metres</b> from the ball.",
        "hands_missing": (
            "The goalkeeper's hands could not be measured in this kick (the wrists are missing in the 3D "
            "reconstruction, or the player map is)."
        ),
        "cameras_note_default": "There is no measurement with several cameras.",
        "cameras_note_manual": (
            "The flight comes from one main camera; only the point where the ball reached the hands was "
            "confirmed with the rear camera."
        ),
        "keeper_one_camera": (
            "The goalkeeper comes from a single camera: his lateral position is approximate. The ball and the "
            "point of the save follow both cameras."
        ),
        "hands_manual": (
            "The “end” point is the arrival of the ball at the hands, confirmed with two cameras (the side comes "
            "from the rear camera, the depth and height from the main one). The “start” point is the mean of the "
            "wrists in the one-camera pose, where the rear camera confirms the keeper at the start of the movement."
        ),
        "hands_limit": (
            "<b>The goalkeeper's hands are the wrists.</b> The pose model has no hand point, and the 3D position "
            "of each wrist comes from a single camera, with an error of a few tens of centimetres."
        ),
        "quote_measured": "All of this was measured from a single broadcast, using the pitch lines as a ruler.",
        "incoming_lead": "The ball came in rolling at about <b>{speed} km/h</b>.",
        "incoming_text": (
            "Before the kick, the ball came in rolling at about <b>{speed} km/h</b>. The kick turned it "
            "by <b>{turn}°</b> (seen from above) and launched it at about {launch} km/h."
        ),
        "incoming_unseen": (
            "The ball was already flying in the first frame of the video: the kick happened in it or "
            "just before, and the incoming ball is not in the picture."
        ),
        "contact_limit_kink": (
            "<b>The contact was found at the break of the rolling ball's path</b>: two straight lines on "
            "the ground, before and after the kick. It is uncertain by one frame."
        ),
        "contact_limit_first_frame": (
            "<b>The kick is the first frame of the video or just before it.</b> The launch is uncertain "
            "by up to one frame, and there is no ball before the contact to measure."
        ),
        "contact_limit_manual": "<b>The contact frame was given by hand.</b>",
        "contact_limit_rest": "<b>The contact is the first frame in which the ball leaves its rest.</b>",
        "quote_curve": "The ball curved up to {dev} metre to the side of the straight line of the kick.",
        "footer_generated": "Generated on {date}. Slow-motion animation: {animation}.",
        "footer_tier": "Fit confidence: {tier} · mean error in the image {rms} px · worst deviation from the references {worst} m.",
        "top_kicker": "kicker (id {id})",
        "top_wall": "wall (ids {a} and {b})",
        "top_wall_one": "wall (id {a})",
        "top_keeper": "goalkeeper (id {id})",
        "top_note": (
            "People on the ground (mean of the ankles): kicker id {kicker}, wall ids {wall}, "
            "goalkeeper id {keeper}. {how}"
        ),
        "top_note_pk": (
            "People on the ground (mean of the ankles): penalty taker id {kicker} and goalkeeper id {keeper}. "
            "Penalty kick: no wall. {how}"
        ),
        "top_note_op": (
            "People on the ground (mean of the ankles): kicker id {kicker} and goalkeeper id {keeper}. "
            "Open play: no wall. {how}"
        ),
        "top_how_auto": "Found by the nearest point; --kicker-id, --wall-ids and --keeper-id replace them.",
        "top_how_manual": "Given by --kicker-id, --wall-ids and --keeper-id.",
        "top_how_map": (
            "Same ID in both cameras: goalkeeper 0, kicker 1, the rest by the distance of the foot to the ball at contact."
        ),
        "top_none": "No people identified in the run (poses_3d_on_field.csv is missing).",
        "ref_anchors": "a physical point of the kick",
        "ref_binocular": "the position triangulated by the two cameras",
        "ref_auto": "a reference point of the image",
        "method_anchors": (
            "Each frame of the broadcast has a camera calibration, obtained from the painted lines of the pitch, "
            "whose measures are known. With it, the position of the ball in each frame becomes a real position on "
            "the pitch. The physical trajectory (gravity, drag and Magnus) is fitted so as to reproduce these "
            "positions with a mean error of <b>{rms} pixel</b> in the image. In addition, three physical points of "
            "the kick enter as references: the ball starts on the kicker's foot, passes 1.8 m high 0.3 m from the "
            "shoulder of a player in the wall, and reaches the keeper's hands. These positions are not reproduced "
            "exactly: the difference is discussed in section 6."
        ),
        "method_binocular": (
            "Each frame of the broadcast has a camera calibration, obtained from the painted lines of the pitch. The "
            "ball was seen by <b>two cameras at the same time</b>: in each frame its position is triangulated from "
            "both views, and the physical trajectory (gravity, drag and Magnus) is fitted to the pixels of both "
            "cameras, with a mean error of <b>{rms} pixel</b>. The height of the ball is therefore measured, not "
            "assumed. The wall line is marked at the point 9.15 m from the kick, where the ball is in this fit."
        ),
        "refs_anchors": (
            "<b>The physical reference points may differ.</b> The foot, the wall and the goalkeeper are chosen from "
            "the reconstructed players, and were not measured directly on the pitch. In this reconstruction, the "
            "difference between the fit and the reference points is {worst} m in the worst case."
        ),
        "refs_binocular": (
            "<b>The position is triangulated, not measured on the pitch.</b> Each point comes from two cameras "
            "calibrated by the pitch lines. The difference between the physical fit and the triangulation of each "
            "frame is {worst} m in the worst case (90th percentile). The players are not references."
        ),
        "wall_anchors": "<b>The height of the wall is a hypothesis</b> of 1.8 metres, and the trajectory over it depends on that choice.",
        "wall_binocular": (
            "<b>The height of the wall is a hypothesis</b> of 1.8 metres. The height of the ball on the 9.15 m plane "
            "is measured by the two cameras, but that does not say whether it passed over the real wall."
        ),
        "ed_disabled": "Editor unavailable for the two-camera reconstruction.",
        "ed_disabled_auto": "Editor unavailable: edit contact and arrival through the FreeKiki GUI/CLI.",
        "method_auto": (
            "Each frame of the broadcast has a camera calibration, obtained from the painted lines of the pitch. "
            "With it, the ball becomes a position on the pitch. A single camera does not measure depth, so the "
            "height of the ball comes from the physics: the trajectory (gravity, drag and Magnus) is fitted to the "
            "pixels, with a mean error of <b>{rms} pixel</b>. The kick starts at the <b>foot contact</b>, when the "
            "ball leaves rest; before that the ball is still and is not part of the calculation. A marked entry uses the homography of the four confirmed goal corners. "
            "A keeper-hand arrival requires explicit confirmation; without an observation or "
            "recognized crossing, the arrival remains undetermined."
        ),
        "refs_auto": (
            "<b>The reference points come from the image itself.</b> The rest point is the still ball before the "
            "contact, and the goal entry is measured by the homography of the goal plane. The difference between "
            "the fit and the reference points is {worst} m in the worst case."
        ),
        "wall_auto": (
            "<b>The height of the wall is a hypothesis</b> of 1.8 metres. The wall line is marked at the point "
            "9.15 m from the kick, where the ball is in this monocular fit; one camera does not say whether it "
            "passed over the real wall."
        ),
        "ed_title": "Adjust the anchors and recompute",
        "ed_intro": (
            "Change the positions (metres on the pitch) or the frames of the reference points and click "
            "Recompute. The computation runs in this browser, with the same model as vailá. The numbers in "
            "the text above do not change: they come from the original computation."
        ),
        "ed_col_label": "point",
        "ed_col_frame": "frame",
        "ed_col_x": "x (m)",
        "ed_col_y": "y (m)",
        "ed_col_z": "z (m)",
        "ed_col_sigma": "σ (m)",
        "ed_col_remove": "remove",
        "ed_add": "+ Add anchor",
        "ed_recalc": "Recompute",
        "ed_reset": "Restore originals",
        "ed_running": "Computing…",
        "ed_min": "At least 3 anchors are needed.",
        "ed_fail": "The fit did not converge. Try other frames or values.",
        "ed_done": "Done.",
        "ed_out_rms": "Mean error in the image",
        "ed_out_max": "Largest error in the image",
        "ed_out_anchor": "Deviation at each anchor",
        "ed_out_kick": "Kick time",
        "ed_out_spin": "Inferred spin",
        "ed_out_peak": "Peak speed",
        "ed_out_dist": "Horizontal distance",
        "ed_out_zmax": "Maximum height",
        "ed_out_tier": "Confidence",
        "ed_json_title": "Anchors as JSON (for the Rebuild ball button)",
        "ed_download": "Download JSON",
        "ed_top": "Recomputed top view",
        "ed_side": "Recomputed side view",
        "ed_note": (
            "Older points (without σ) use σ 0.3 m. The distance runs from the kick frame to the last "
            "frame with a marker."
        ),
        # --- open-play dribble / pass report ---
        "op_kicker": "Football biomechanics · press report",
        "op_title_dribble": "The physics of a 1-vs-N dribble",
        "op_title_pass": "The physics of a pass",
        "op_subtitle": "Planar pitch kinematics from broadcast footage, made with free software vailá",
        "op_lead_dribble": (
            "The dribbler (id {dribbler}) covered <b>{path} m</b> with <b>{cod}</b> changes of direction "
            "and mean curvature <b>{curv} rad/m</b>, against {n} marker(s)."
        ),
        "op_lead_pass": (
            "Pass from id {passer} to id {receiver}: the ball covered about <b>{length} m</b> "
            "in <b>{flight} s</b> between release and reception (heuristic)."
        ),
        "op_card_ball_path": "ball path on the plane",
        "op_card_ball_vmax": "ball peak speed",
        "op_card_ball_accel": "peak acceleration (m/s²)",
        "op_card_drib_path": "dribbler path",
        "op_card_drib_vmax": "dribbler peak speed",
        "op_card_drib_cod": "changes of direction (COD)",
        "op_card_drib_curv": "mean curvature (rad/m)",
        "op_card_drib_accel": "peak acceleration (m/s²)",
        "op_card_drib_decel": "peak deceleration (m/s²)",
        "op_card_pass_len": "ball path of the pass",
        "op_card_pass_flight": "release → reception time",
        "op_card_pass_pair": "player–player distance at release",
        "op_h_top": "Top view",
        "op_h_speed": "Speeds over time",
        "op_h_dist": "Distances over time",
        "op_h_science": "Scientific reading",
        "op_h_markers": "Markers",
        "op_cap_top": "Trajectories on the pitch plane (metres).",
        "op_cap_speed": "Instantaneous speed (km/h) from planar positions.",
        "op_cap_dist": "Interpersonal or ball–player distances (m).",
        "op_fig_top_title": "Top view: pitch trajectories",
        "op_fig_x": "x on the pitch (m)",
        "op_fig_y": "y on the pitch (m)",
        "op_fig_t": "time (s)",
        "op_fig_speed": "speed (km/h)",
        "op_fig_speed_title": "Planar speed",
        "op_fig_dist": "distance (m)",
        "op_fig_dist_drib_title": "Dribbler–marker distance",
        "op_fig_dist_pass_title": "Pass distances",
        "op_legend_ball": "ball",
        "op_legend_dribbler": "dribbler",
        "op_legend_marker": "marker {id}",
        "op_legend_passer": "passer",
        "op_legend_receiver": "receiver",
        "op_legend_passer_ball": "passer–ball",
        "op_legend_receiver_ball": "receiver–ball",
        "op_legend_passer_receiver": "passer–receiver",
        "op_th_id": "id",
        "op_th_min_d": "min (m)",
        "op_th_rel_v": "rel. v at min (m/s)",
        "op_th_cod": "COD",
        "op_th_path": "path (m)",
        "op_no_markers": "No markers given.",
        "op_science_dribble": (
            "In the Wilson–Santiago predator–prey framing, one-versus-one success depends more on "
            "change of direction and speed along curved paths than on straight-line sprinting. "
            "Interpersonal distance and relative velocity (Corrêa / Duarte) describe the marker’s "
            "pressure on the dribbler."
        ),
        "op_science_pass": (
            "The Wilson–Santiago passing and control line treats the mix of speed, accuracy and "
            "agility in the pass as a predictor of small-sided success. Here we measure the clip: "
            "ball path, pairwise distances, and the heuristic release→reception window."
        ),
        "op_limit": (
            "Operational planar metrics from FreeKiki tracks; they do not replace closed-skill "
            "curved-path circuits (0–1.37 rad/m) or rebound-board passing batteries."
        ),
        "op_ref1": "Wilson et al. (2018). J Sports Sci — dribbling speed along curved paths (PMID 30470166).",
        "op_ref2": "Wilson, Santiago et al. (2026). Int J Sports Sci Coach — dribbling and passing vs 3v3 success.",
        "op_ref3": "Corrêa et al. / Duarte et al. — interpersonal distance and relative velocity in 1v1.",
        # --- pass report: timing, receiver run, shot link ---
        "op_lead_pass_main": (
            "Pass from id {passer} to id {receiver}: the ball left at about <b>{v0} km/h</b> and "
            "reached the receiver's foot <b>{flight} s</b> later, <b>{dist} m</b> from where the "
            "passer stood."
        ),
        "op_lead_pass_run_before": (
            " When the ball left, id {receiver} was already running at <b>{vr} km/h</b>: the run "
            "began before the pass, at least <b>{lead} s</b> earlier (estimate)."
        ),
        "op_lead_pass_run_onset": (
            " Id {receiver} started to run <b>{lead} s</b> {when} the ball left."
        ),
        "op_when_before": "before",
        "op_when_after": "after",
        "op_lead_shot": (
            " Then id {receiver} shot from <b>{goal} m</b> off the goal line, the ball leaving the "
            "foot at about <b>{launch} km/h</b>."
        ),
        "op_tag_measured": "measured",
        "op_tag_estimate": "estimate",
        "op_card_pass_dist": "from the passer to the arrival point",
        "op_card_pass_v0": "ball speed at release",
        "op_card_pass_time": "release to first touch",
        "op_card_pass_v1": "ball speed at arrival",
        "op_card_pass_vmean": "mean ball speed of the pass",
        "op_card_pair_release": "passer–receiver at release",
        "op_card_recv_v0": "receiver when the ball left",
        "op_card_recv_lead": "the receiver started running before the pass (at least)",
        "op_card_recv_onset": "run onset relative to the release",
        "op_card_recv_peak": "receiver peak speed",
        "op_card_recv_disp": "receiver displacement during the pass",
        "op_card_recv_v1": "receiver at the first touch",
        "op_card_passer_v0": "passer at the moment of the pass",
        "op_h_pass_ball": "The ball",
        "op_h_pass_receiver": "The receiver",
        "op_h_table": "Position, speed and time at the key instants",
        "op_table_note": (
            "Positions in metres in the pitch frame (x along the pitch, y across it; the centre "
            "spot is the origin). t = 0 at the ball release."
        ),
        "op_th_instant": "instant",
        "op_th_t": "t (s)",
        "op_th_ball_xy": "ball (x, y)",
        "op_th_vball": "ball (km/h)",
        "op_th_player_xy": "id {id} (x, y)",
        "op_th_vplayer": "id {id} (km/h)",
        "op_row_release": "ball release",
        "op_row_first": "first frame (just after release)",
        "op_science_pass_timed": (
            "The Wilson–Santiago line on passing and control treats the combination of speed, "
            "accuracy and agility in passing as a predictor of small-sided game performance. Here "
            "we measure the play: ball speed at release and at arrival, the distance from the "
            "passer to the arrival point and the receiver's displacement while the ball travels."
        ),
        "op_row_mid": "mid-pass",
        "op_row_arrival": "receiver's first touch",
        "op_row_shot": "shot (separate clip)",
        "op_h_shot": "After the pass: the shot",
        "op_card_shot_launch": "speed off the foot (rounded)",
        "op_card_shot_goal": "shot distance to the goal line",
        "op_card_shot_from_arrival": "from the first touch to the shot point",
        "op_card_shot_from_passer": "from the passer to the shot point",
        "op_card_shot_mean": "mean speed of the shot",
        "op_card_shot_height": "peak ball height of the shot",
        "op_shot_text": (
            "The shot was filmed in another part of the footage, by another camera: the time "
            "between the reception and the shot is not measured. From the {d} m between the first "
            "touch and the shot point, and the ball and receiver speeds at the first touch, it lies "
            "between {g0} and {g1} s (estimate). From the shot to the last analysed frame, the ball "
            "covered {path} m in {fl} s. Speed: {launch} km/h off the foot (flight fit) and {mean} "
            "km/h on average to the last frame, since the ball only slows after the kick; on air, "
            "about {head} km/h."
        ),
        "op_cap_top_pass": (
            "Top view, in metres. Squares: positions at the ball release. Circles: at the first "
            "touch. Star: arrival point. Black dashes: distance from the passer to the arrival."
        ),
        "op_cap_top_shot": " Red triangle: shot point, with the ball direction.",
        "op_cap_speed_pass": (
            "Planar speed (km/h). t = 0 at the ball release; green line at the first touch."
        ),
        "op_cap_dist_pass": "Ball–player and player–player distances (m); t = 0 at the ball release.",
        "op_fig_t_rel": "time since the ball release (s)",
        "op_legend_release": "ball release",
        "op_legend_arrival": "first touch",
        "op_legend_arrival_pt": "arrival",
        "op_legend_shot": "shot",
        "op_legend_passer_id": "passer (id {id})",
        "op_legend_receiver_id": "receiver (id {id})",
        "op_fig_already_running": "id {id} already at {v} km/h",
        "op_science_pass_run": (
            " When the receiver starts to run, relative to the ball release, is a reading of "
            "anticipation: running before the pass gains metres on the marker, and the pass must be "
            "firm and accurate enough to reach the space he runs into."
        ),
        "op_limit_pass_fps": "Video at {fps} frames per second (one frame every {dt} ms).",
        "op_limit_pass_release_fit": (
            "The ball already moved in the first frame: the release instant ({rel} s before the "
            "first frame) comes from the ball flight fit and agrees with the backward "
            "extrapolation ({back} s)."
        ),
        "op_limit_pass_release_heur": (
            "The release instant is estimated from the closest passer–ball distance followed by a "
            "rise in ball speed."
        ),
        "op_limit_pass_arrival_user": "The receiver's first touch was marked in the video (frame {f}).",
        "op_limit_pass_arrival_auto": (
            "The first touch is the closest receiver–ball approach (frame {f}); check it in the video."
        ),
        "op_limit_pass_tracks": (
            "The ball is the ground track (camera ray cut at ball height). Player positions are the "
            "ground point under the detection box, not the foot, with outlier removal and a 2 Hz "
            "low-pass filter."
        ),
        "op_limit_pass_run_bound": (
            "Because the receiver was already running in the first frame, the run onset is not in "
            "the footage: the value shown is a lower bound assuming a peak acceleration of "
            "{a} m/s² ({l0} s with {ahi} m/s², {l1} s with {alo} m/s²)."
        ),
        "op_limit_pass_shot": (
            "The pass and the shot are separate clips; the time between them was not measured."
        ),
    },
}

# The anchor labels of the first version (Portuguese and English), mapped to the keys above.
_ANCHOR_KEYS = (
    ("contato do pé", "anchor_contact"),
    ("início", "anchor_start"),
    ("start", "anchor_start"),
    ("meio", "anchor_middle"),
    ("middle", "anchor_middle"),
    ("plano a 9,15", "anchor_wall_plane"),
    ("barreira", "anchor_barrier"),
    ("barrier", "anchor_barrier"),
    ("chegada: gol", "anchor_goal"),
    ("end: goal", "anchor_goal"),
    ("entrada no gol", "anchor_entry"),
    ("chegada", "anchor_end"),
    ("end:", "anchor_end"),
    ("keeper", "anchor_end"),
)


def tr(lang: str, key: str, **values) -> str:
    """The text ``key`` in ``lang`` (``pt`` or ``en``), with ``values`` filled in."""
    text = _STRINGS[lang][key]
    return text.format(**values) if values else text


def editor_labels(lang: str) -> dict[str, str]:
    """The texts of the interactive editor (the keys starting with ``ed_``, plus the tiers)."""
    keys = [k for k in _STRINGS[lang] if k.startswith(("ed_", "tier_"))]
    return {k: _STRINGS[lang][k] for k in keys}


def fmt_num(x: float | None, nd: int = 1, lang: str = "pt") -> str:
    """A number with the decimal mark of the language ('.' in English, ',' in Portuguese)."""
    if x is None or not math.isfinite(x):
        return "—"
    text = f"{x:.{nd}f}"
    return text.replace(".", ",") if lang == "pt" else text


def localize_anchor_label(lang: str, label: str) -> str:
    """An anchor label in ``lang``: a label written by the dialog is translated by its keyword."""
    low = label.lower()
    for keyword, key in _ANCHOR_KEYS:
        if keyword in low:
            return tr(lang, key)
    return label
