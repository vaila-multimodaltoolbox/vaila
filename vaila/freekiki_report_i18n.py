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
Version: 0.4.11
Created: 27 October 2026
Update Date: 27 October 2026

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
            "<b>Confiança {tier} no ajuste.</b> {tier_text} O maior desvio entre a trajetória e um ponto "
            "físico do lance é de {worst} m. Veja a seção “O que estes números não dizem”."
        ),
        "kicker_of": "do pé de {name}",
        "kicker_default": "do pé do jogador que bateu a falta",
        "kicker_name_default": "o jogador que bateu a falta",
        "wall_above": "acima",
        "wall_below": "abaixo",
        "spin_vertical": "quase vertical",
        "spin_tilted": "inclinado",
        "date_format": "{d:02d}/{m:02d}/{y}",
        "default_title": "A física de um chute de falta",
        "default_subtitle": "Reconstrução 3D a partir de transmissão, feita com software livre vailá",
        "fig_top_title": "Vista de cima: {d} m em {t} s",
        "fig_top_x": "x no campo (m)",
        "fig_top_y": "y no campo (m)",
        "fig_top_cbar": "velocidade da bola (km/h)",
        "fig_start": "chute",
        "fig_entry": "entrada",
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
        "vid_after": "após o chute",
        "vid_speed": "velocidade",
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
        "anchor_start": "início: bola no pé",
        "anchor_middle": "meio: bola no ar",
        "anchor_barrier": "barreira: bola sobre a barreira",
        "anchor_end": "chegada: mãos do goleiro",
        "anchor_goal": "chegada: gol",
        "anchor_kicker": "pé do chutador",
        "anchor_shoulder": "ombro",
        "quote_distance": "A bola percorreu {dist} metros em {flight} segundos.",
        "quote_mean": "Na média, a bola andou a cerca de {mean} km/h do pé até a entrada.",
        "quote_height": "A bola subiu até {zmax} metros de altura, e a barreira tem cerca de 1,8 metro.",
        "quote_measured": "Tudo isso foi medido a partir de uma única transmissão, usando as linhas do campo como régua.",
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
        "top_how_auto": "Identificados pelo ponto mais próximo; --kicker-id, --wall-ids e --keeper-id trocam.",
        "top_how_manual": "Informados por --kicker-id, --wall-ids e --keeper-id.",
        "top_none": "Sem jogadores identificados no run (falta poses_3d_on_field.csv).",
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
            "<b>Fit confidence: {tier}.</b> {tier_text} The largest deviation between the trajectory and a "
            "physical point of the kick is {worst} m. See “What these numbers do not say”."
        ),
        "kicker_of": "from the foot of {name}",
        "kicker_default": "from the foot of the kicker",
        "kicker_name_default": "the kicker",
        "wall_above": "above",
        "wall_below": "below",
        "spin_vertical": "nearly vertical",
        "spin_tilted": "tilted",
        "date_format": "{m:02d}/{d:02d}/{y}",
        "default_title": "The physics of a free kick",
        "default_subtitle": "3D reconstruction from broadcast footage, made with free software vailá",
        "fig_top_title": "Top view: {d} m in {t} s",
        "fig_top_x": "x on the pitch (m)",
        "fig_top_y": "y on the pitch (m)",
        "fig_top_cbar": "ball speed (km/h)",
        "fig_start": "kick",
        "fig_entry": "entry",
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
        "vid_after": "after the kick",
        "vid_speed": "speed",
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
        "anchor_start": "start: ball on kick foot",
        "anchor_middle": "middle: ball in the air",
        "anchor_barrier": "barrier: ball over the wall",
        "anchor_end": "end: keeper's hands",
        "anchor_goal": "end: goal",
        "anchor_kicker": "kicker's foot",
        "anchor_shoulder": "shoulder",
        "quote_distance": "The ball travelled {dist} metres in {flight} seconds.",
        "quote_mean": "On average, the ball moved at about {mean} km/h from the foot to the entry.",
        "quote_height": "The ball rose to {zmax} metres, and the wall is about 1.8 metres high.",
        "quote_measured": "All of this was measured from a single broadcast, using the pitch lines as a ruler.",
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
        "top_how_auto": "Found by the nearest point; --kicker-id, --wall-ids and --keeper-id replace them.",
        "top_how_manual": "Given by --kicker-id, --wall-ids and --keeper-id.",
        "top_none": "No people identified in the run (poses_3d_on_field.csv is missing).",
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
    },
}

# The anchor labels of the first version (Portuguese and English), mapped to the keys above.
_ANCHOR_KEYS = (
    ("início", "anchor_start"),
    ("start", "anchor_start"),
    ("meio", "anchor_middle"),
    ("middle", "anchor_middle"),
    ("barreira", "anchor_barrier"),
    ("barrier", "anchor_barrier"),
    ("chegada: gol", "anchor_goal"),
    ("end: goal", "anchor_goal"),
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
