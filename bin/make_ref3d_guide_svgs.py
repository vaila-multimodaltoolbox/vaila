"""
================================================================================
make_ref3d_guide_svgs.py - figures of the illustrated REF3D calibration guide
================================================================================
*vailá* – Multimodal Toolbox
Authors: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
Date: 16 November 2026
Update: 16 November 2026
Version: 0.4.19

Writes the schematic drawings used by vaila/help/ref3d_guide.html and
ref3d_guide_pt.html into vaila/help/images/ref3d/. The template drawings come from
vaila.ref3d_builder (same points, same order p0, p1, ... and same right-handed view as
the REF3D panel of getpixelvideo), so they cannot drift from the tool.

Usage:
    uv run --no-sync python bin/make_ref3d_guide_svgs.py           # write missing files
    uv run --no-sync python bin/make_ref3d_guide_svgs.py --force   # rewrite all
    uv run --no-sync python bin/make_ref3d_guide_svgs.py --check   # exit 1 if out of date
================================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vaila import ref3d_builder as rb  # noqa: E402

OUT_DIR = ROOT / "vaila" / "help" / "images" / "ref3d"

INK, MUTED, LINE, PAPER = "#1e293b", "#64748b", "#cbd5e1", "#ffffff"
ORANGE, GREEN, YELLOW, RED, PINK = "#f08c00", "#2f9e44", "#e0a800", "#e03131", "#ae3ec9"
FONT = 'font-family="Segoe UI, system-ui, sans-serif"'


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x, y, text, size=13, fill=INK, weight="400", anchor="start") -> str:
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{_esc(text)}</text>'
    )


def _open(width: int, height: int, label: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" {FONT} role="img" aria-label="{_esc(label)}">',
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
        "</defs>",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" '
        f'fill="{PAPER}" stroke="{LINE}"/>',
    ]


# --- workflow: six steps ----------------------------------------------------------

WORKFLOW = {
    "en": [
        ("1", "Measure", ["Tape the object;", "write it in the", ".toml (REF3D N)"]),
        ("2", "REF3D", ["Ctrl+I (INSERT),", "button REF3D,", "Enter = T loads"]),
        ("3", "Click", ["the point lit", "in the panel,", "in order p0, p1…"]),
        ("4", "Move", ["pole to the next", "corner: Add video", "(F8) if new video"]),
        ("5", "Check", ["rings green,", "fx ≈ fy,", "camera OK"]),
        ("6", "Save", [".ref3d + pixels", "+ .dlt3d →", "dlt3d.py / rec3d"]),
    ],
    "pt": [
        ("1", "Medir", ["Trena no objeto;", "anote no .toml", "(REF3D N)"]),
        ("2", "REF3D", ["Ctrl+I (INSERT),", "botão REF3D,", "Enter = T carrega"]),
        ("3", "Clicar", ["o ponto aceso", "no painel, na", "ordem p0, p1…"]),
        ("4", "Mover", ["o poste ao próximo", "canto: Add video", "(F8) se outro vídeo"]),
        ("5", "Conferir", ["anéis verdes,", "fx ≈ fy,", "camera OK"]),
        ("6", "Salvar", [".ref3d + pixels", "+ .dlt3d →", "dlt3d.py / rec3d"]),
    ],
}


def workflow_svg(lang: str) -> str:
    steps = WORKFLOW[lang]
    box_w, box_h, gap, top = 132, 128, 22, 20
    width = 2 * 16 + len(steps) * box_w + (len(steps) - 1) * gap
    height = top + box_h + 22
    label = "REF3D calibration in six steps" if lang == "en" else "Calibração REF3D em seis passos"
    out = _open(width, height, label)
    for k, (num, name, lines) in enumerate(steps):
        x = 16 + k * (box_w + gap)
        accent = ORANGE if name == "REF3D" else GREEN if num == "5" else "#2563eb"
        out.append(
            f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="9" fill="#f8fafc" '
            f'stroke="{accent}" stroke-width="2"/>'
        )
        out.append(f'<circle cx="{x + 20}" cy="{top + 22}" r="13" fill="{accent}"/>')
        out.append(_text(x + 20, top + 27, num, 14, PAPER, "700", "middle"))
        out.append(_text(x + 40, top + 27, name, 15, INK, "700"))
        for n, line in enumerate(lines):
            out.append(_text(x + 12, top + 62 + n * 19, line, 12.5, INK))
        if k < len(steps) - 1:
            ax = x + box_w + 3
            out.append(
                f'<line x1="{ax}" y1="{top + box_h / 2}" x2="{ax + gap - 6}" '
                f'y2="{top + box_h / 2}" stroke="{MUTED}" stroke-width="2" '
                'marker-end="url(#arr)"/>'
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- screen: the getpixelvideo window with REF3D on ---------------------------------

SCREEN_WORDS = {
    "en": {
        "label": "getpixelvideo window with REF3D on",
        "toolbar": "toolbar",
        "video": "video",
        "panel": "REF3D panel",
        "list": "list: pN · label · (X, Y, Z) · error",
        "sketch": "3D sketch, next point blinks",
        "ring_ok": "click on its ring: < 2 px",
        "ring_far": "far from the DLT: check it",
        "next": "pN? = where the next point should be",
        "tab": "Hide panel → this tab; click it to show",
    },
    "pt": {
        "label": "janela do getpixelvideo com o REF3D ligado",
        "toolbar": "barra",
        "video": "vídeo",
        "panel": "painel REF3D",
        "list": "lista: pN · rótulo · (X, Y, Z) · erro",
        "sketch": "desenho 3D, o próximo ponto pisca",
        "ring_ok": "clique no anel: < 2 px",
        "ring_far": "longe do DLT: confira",
        "next": "pN? = onde o próximo ponto deve cair",
        "tab": "Hide panel → vira esta aba; clique para voltar",
    },
}


def screen_svg(lang: str) -> str:
    w = SCREEN_WORDS[lang]
    width, height = 760, 420
    out = _open(width, height, w["label"])
    # window
    out.append(
        f'<rect x="20" y="20" width="560" height="380" rx="6" fill="#1f2937" stroke="{INK}"/>'
    )
    # video area: a walkway in perspective
    out.append(
        '<polygon points="150,330 470,330 360,150 250,150" fill="#374151" stroke="#6b7280"/>'
    )
    out.append(_text(40, 48, w["video"], 13, "#e5e7eb", "600"))
    # poles at the 4 corners (perspective) with clicked points and rings
    poles = [((150, 330), (150, 250)), ((470, 330), (470, 250)), ((250, 150), (250, 110))]
    for (bx, by), (tx, ty) in poles:
        out.append(
            f'<line x1="{bx}" y1="{by}" x2="{tx}" y2="{ty}" stroke="#fbbf24" stroke-width="4"/>'
        )
    for cx, cy in ((150, 330), (150, 250), (470, 330), (470, 250), (250, 150)):
        out.append(f'<circle cx="{cx}" cy="{cy}" r="3.5" fill="#f8fafc"/>')
        out.append(
            f'<circle cx="{cx}" cy="{cy}" r="8" fill="none" stroke="#40c057" stroke-width="2"/>'
        )
    # one suspicious click
    out.append('<circle cx="262" cy="108" r="3.5" fill="#f8fafc"/>')
    out.append(f'<line x1="262" y1="108" x2="250" y2="110" stroke="{RED}" stroke-width="1.5"/>')
    out.append(f'<circle cx="250" cy="110" r="8" fill="none" stroke="{RED}" stroke-width="2"/>')
    # next point predicted
    out.append(
        '<circle cx="360" cy="150" r="9" fill="none" stroke="#ffd43b" stroke-width="2" '
        'stroke-dasharray="3 2"/>'
    )
    out.append(_text(372, 146, "p6?", 13, "#ffd43b", "700"))
    out.append(f'<circle cx="360" cy="112" r="6" fill="none" stroke="{PINK}" stroke-width="1.5"/>')
    out.append(_text(370, 108, "p7?", 12, PINK, "700"))
    # toolbar with buttons
    out.append('<rect x="20" y="360" width="560" height="40" fill="#111827"/>')
    out.append(_text(30, 352, w["toolbar"], 11, "#9ca3af"))
    for k, name in enumerate(("Open", "Save", "CALIB", "REF3D", "1-line", "Help")):
        bx = 30 + k * 88
        fill = ORANGE if name == "REF3D" else "#374151"
        out.append(f'<rect x="{bx}" y="368" width="78" height="24" rx="4" fill="{fill}"/>')
        out.append(_text(bx + 39, 385, name, 12, PAPER, "600", "middle"))
    # side panel
    out.append(
        '<rect x="430" y="28" width="142" height="322" rx="4" fill="#0f172a" opacity="0.92"/>'
    )
    out.append(_text(438, 45, "REF3D · pole", 11, "#ffd43b", "700"))
    out.append('<rect x="508" y="33" width="58" height="16" rx="3" fill="#475569"/>')
    out.append(_text(537, 45, "Hide panel", 9, PAPER, "600", "middle"))
    out.append(_text(438, 62, "clicked 6 of 8", 10, "#e5e7eb"))
    out.append(_text(438, 76, "fy/fx 1.00", 10, "#69db7c"))
    out.append(_text(438, 90, "camera OK", 10, "#69db7c", "700"))
    out.append('<rect x="438" y="98" width="126" height="90" fill="#1e293b"/>')
    # mini sketch of the pole template
    pts = [
        (450, 175),
        (450, 150),
        (490, 175),
        (490, 150),
        (470, 135),
        (470, 112),
        (515, 135),
        (515, 112),
    ]
    for k in range(0, 8, 2):
        (x1, y1), (x2, y2) = pts[k], pts[k + 1]
        out.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#cbd5e1" stroke-width="2"/>'
        )
    for k, (px, py) in enumerate(pts):
        fill = "#69db7c" if k < 6 else "#ffd43b" if k == 6 else "#94a3b8"
        out.append(f'<circle cx="{px}" cy="{py}" r="{4 if k == 6 else 2.5}" fill="{fill}"/>')
    out.append(
        f'<line x1="530" y1="180" x2="552" y2="180" stroke="{rb.SVG_AXIS_COLOURS["x"]}" stroke-width="2"/>'
    )
    out.append(
        f'<line x1="530" y1="180" x2="542" y2="171" stroke="{rb.SVG_AXIS_COLOURS["y"]}" stroke-width="2"/>'
    )
    out.append(
        f'<line x1="530" y1="180" x2="530" y2="160" stroke="{rb.SVG_AXIS_COLOURS["z"]}" stroke-width="2"/>'
    )
    rows = [
        ("p4 pos3 0 m", "0.8px", "#69db7c"),
        ("p5 pos3 1 m", "1.1px", "#69db7c"),
        ("p6 pos4 0 m", "-", "#ffd43b"),
        ("p7 pos4 1 m", "-", "#cbd5e1"),
    ]
    for k, (name, state, fill) in enumerate(rows):
        ry = 200 + k * 18
        if state == "-" and fill == "#ffd43b":
            out.append(f'<rect x="432" y="{ry}" width="138" height="17" fill="#5c4813"/>')
        out.append(_text(438, ry + 13, name, 10, fill))
        out.append(_text(566, ry + 13, state, 10, fill, anchor="end"))
    # collapsed tab (inset)
    out.append(
        f'<rect x="600" y="300" width="146" height="62" rx="6" fill="#f8fafc" stroke="{LINE}"/>'
    )
    out.append('<rect x="610" y="312" width="126" height="20" rx="3" fill="#2f7a46"/>')
    out.append(_text(673, 326, "REF3D 8/8 · OK · Show", 9.5, PAPER, "700", "middle"))
    for n, line in enumerate(_wrap(w["tab"], 24)):
        out.append(_text(610, 348 + n * 12, line, 9.5, MUTED))
    # numbered badges on the drawing, explained in the legend on the right
    notes = [
        (128, 250, w["ring_ok"], GREEN),
        (282, 104, w["ring_far"], RED),
        (338, 168, w["next"], "#c79100"),
        (548, 112, w["sketch"], MUTED),
        (548, 252, w["list"], MUTED),
        (333, 352, "REF3D = Shift+R", ORANGE),
    ]
    for k, (bx, by, text, colour) in enumerate(notes, start=1):
        out.append(f'<circle cx="{bx}" cy="{by}" r="9" fill="{colour}" stroke="{PAPER}"/>')
        out.append(_text(bx, by + 4, str(k), 11, PAPER, "700", "middle"))
        ly = 44 + (k - 1) * 40
        out.append(f'<circle cx="608" cy="{ly - 4}" r="9" fill="{colour}"/>')
        out.append(_text(608, ly, str(k), 11, PAPER, "700", "middle"))
        for n, line in enumerate(_wrap(text, 21)):
            out.append(_text(624, ly + n * 14, line, 11.5, colour, "600"))
    out.append(_text(438, 344, w["panel"], 10, "#94a3b8"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


# --- check: a wrong measure still fits the clicks ------------------------------------

CHECK_WORDS = {
    "en": {
        "label": "Right and wrong measure: both fit the clicks, only one is a real camera",
        "right": "Measure right: width 1.38 m",
        "wrong": "Measure wrong: width typed 0.55 m",
        "fit": "clicks fit: ~1 px",
        "pixel": "camera pixel",
        "ok": "fx 1349 vs fy 1343 → camera OK",
        "bad": "fx 3359 vs fy 1339 → NOT A REAL CAMERA",
        "why": "The 11 DLT parameters bend to fit any measure; only fx / fy shows the error.",
    },
    "pt": {
        "label": "Medida certa e errada: as duas encaixam os cliques, só uma é câmera real",
        "right": "Medida certa: largura 1,38 m",
        "wrong": "Medida errada: largura digitada 0,55 m",
        "fit": "cliques encaixam: ~1 px",
        "pixel": "pixel da câmera",
        "ok": "fx 1349 vs fy 1343 → camera OK",
        "bad": "fx 3359 vs fy 1339 → NOT A REAL CAMERA",
        "why": "Os 11 parâmetros do DLT se dobram a qualquer medida; só fx / fy mostra o erro.",
    },
}


def check_svg(lang: str) -> str:
    w = CHECK_WORDS[lang]
    width, height = 760, 300
    out = _open(width, height, w["label"])
    for k, (title, verdict, colour, px) in enumerate(
        ((w["right"], w["ok"], GREEN, (1.0, 1.0)), (w["wrong"], w["bad"], RED, (2.4, 0.95)))
    ):
        x0 = 20 + k * 370
        out.append(
            f'<rect x="{x0}" y="20" width="350" height="230" rx="8" fill="#f8fafc" '
            f'stroke="{colour}" stroke-width="2"/>'
        )
        out.append(_text(x0 + 14, 44, title, 14, INK, "700"))
        # a pixel: square for the real camera, stretched for the wrong measure
        sx, sy = 70 * px[0], 70 * px[1]
        out.append(
            f'<rect x="{x0 + 175 - sx / 2:.1f}" y="{80:.1f}" width="{sx:.1f}" height="{sy:.1f}" '
            f'fill="none" stroke="{colour}" stroke-width="3"/>'
        )
        out.append(
            _text(
                x0 + 175,
                80 + sy + 20,
                w["pixel"] + (": fx = fy" if k == 0 else ": fx ≠ fy"),
                12,
                MUTED,
                "600",
                "middle",
            )
        )
        out.append(_text(x0 + 14, 210, w["fit"], 12.5, INK))
        out.append(_text(x0 + 14, 232, verdict, 13, colour, "700"))
    out.append(_text(width / 2, 278, w["why"], 12.5, MUTED, "400", "middle"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- all figures --------------------------------------------------------------------


def figures() -> dict[str, str]:
    """File name -> SVG text of every figure of the guide."""
    walk = rb.parse_positions("rect 1.38 x 5")
    out = {
        "box.svg": rb.template_svg(rb.Ref3dTemplate("box", size=(1.38, 5.0, 1.0))),
        "pole_rect.svg": rb.template_svg(
            rb.Ref3dTemplate("pole", marks=(0.0, 1.0), positions=walk), highlight=6
        ),
        "triad.svg": rb.template_svg(
            rb.Ref3dTemplate(
                "triad", arms_x=(0.5,), arms_y=(0.5, 1.0), arms_z=(0.5, 1.0), positions=((0, 0),)
            )
        ),
    }
    for lang in ("en", "pt"):
        out[f"axes_{lang}.svg"] = rb.axes_svg(lang=lang)
        out[f"workflow_{lang}.svg"] = workflow_svg(lang)
        out[f"screen_{lang}.svg"] = screen_svg(lang)
        out[f"check_{lang}.svg"] = check_svg(lang)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--force", action="store_true", help="rewrite existing files")
    parser.add_argument("--check", action="store_true", help="exit 1 when a file is out of date")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    stale = []
    for name, svg in figures().items():
        path = args.out_dir / name
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if args.check:
            if current != svg:
                stale.append(name)
            continue
        if current is not None and not args.force:
            if current != svg:
                print(f"{path} exists and differs (use --force)")
            continue
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(svg, encoding="utf-8")
        print(f"wrote {path}")
    if stale:
        print("out of date: " + ", ".join(stale) + " (run with --force)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
