"""The report texts of the monocular fit with the automatic start and arrival (0.4.12).

These check labels and switches only; the numbers come from the run files and are covered by
``test_freekiki_report.py``.
"""

from __future__ import annotations

from vaila.freekiki_report import _method_key, editor_payload
from vaila.freekiki_report_i18n import _STRINGS, localize_anchor_label


def test_method_key_separates_the_three_fits() -> None:
    assert _method_key({"method": "binocular"}) == "binocular"
    assert _method_key({"method": "monocular_auto"}) == "auto"
    assert _method_key({"method": "anchors"}) == "anchors"
    assert _method_key({}) == "anchors"  # runs written before the method field


def test_auto_anchor_labels_name_the_kick_points() -> None:
    contact = "start: contato do pé (início do movimento)"
    entry = "chegada: entrada no gol (plano da baliza)"
    wall = "barreira: plano a 9,15 m do chute (regra FIFA)"
    keeper = "chegada: mão do goleiro no último quadro"
    assert localize_anchor_label("pt", contact) == "início: contato do pé"
    assert localize_anchor_label("pt", entry) == "chegada: entrada no gol"
    assert localize_anchor_label("pt", wall) == "barreira: plano a 9,15 m"
    assert localize_anchor_label("pt", keeper) == "chegada: mãos do goleiro"
    assert localize_anchor_label("en", contact) == "start: foot contact"
    assert localize_anchor_label("en", entry) == "end: goal entry"
    assert localize_anchor_label("en", wall) == "wall: the 9.15 m plane"
    assert localize_anchor_label("en", keeper) == "end: keeper's hands"


def test_every_method_text_exists_in_both_languages() -> None:
    for lang in ("pt", "en"):
        for key in ("method_auto", "refs_auto", "wall_auto", "ref_auto", "ed_disabled_auto"):
            assert _STRINGS[lang][key], (lang, key)


def test_auto_editor_is_off_with_its_own_message() -> None:
    payload = editor_payload(None, {"method": "monocular_auto"}, "pt")  # type: ignore[arg-type]
    assert payload["disabled"] is True
    assert payload["labels"]["ed_disabled"] == _STRINGS["pt"]["ed_disabled_auto"]
    payload_en = editor_payload(None, {"method": "monocular_auto"}, "en")  # type: ignore[arg-type]
    assert payload_en["labels"]["ed_disabled"] == _STRINGS["en"]["ed_disabled_auto"]
