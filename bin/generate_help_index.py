#!/usr/bin/env python3
"""Generate vaila/help/index.md and index.html from help pages on disk.

Scans ``vaila/help/*.md`` (excluding index.md / README.md), normalizes Category
metadata into a small set of buckets, and writes a project intro + full catalog
with HTML/MD links for each help topic.

Usage (from repo root)::

    uv run python bin/generate_help_index.py
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HELP_DIR = REPO_ROOT / "vaila" / "help"
VAILA_PY = REPO_ROOT / "vaila.py"

SKIP_NAMES = frozenset({"index.md", "README.md"})

# Canonical category order for the catalog
CATEGORY_ORDER = (
    "Analysis",
    "ML",
    "Processing",
    "Tools",
    "Utils",
    "Visualization",
    "Guides",
)

# Map free-form Category strings from help pages → canonical bucket
_CATEGORY_ALIASES: dict[str, str] = {
    "analysis": "Analysis",
    "ml": "ML",
    "machine learning": "ML",
    "machine learning / computer vision / biomechanics analysis": "ML",
    "markerless 2d / meta (facebook)": "ML",
    "markerless 3d / meta (facebook)": "ML",
    "multimodal analysis / video segmentation": "ML",
    "processing": "Processing",
    "processing / markerless 3d": "Processing",
    "processing / re-id": "Processing",
    "tools": "Tools",
    "tools / brainstorm": "Tools",
    "tools / research": "Tools",
    "tools / video and image": "Tools",
    "tools → video and image": "Tools",
    "data files": "Tools",
    "utils": "Utils",
    "utils / re-id": "Utils",
    "visualization": "Visualization",
    "multimodal analysis / sports field calibration": "Analysis",
    "uncategorized": "Analysis",
}

# Filename stems forced into Guides (reference / how-to pages, not modules)
_GUIDE_STEMS = frozenset(
    {
        "BRAINSTORM_GUIDE",
        "gpu_guide",
        "sports_fields_courts",
        "tennis_court",
        "soccerfield",
        "markerless2d_yolo26",
    }
)

# Heuristic fallback when Category metadata is missing
_STEM_CATEGORY: dict[str, str] = {
    "emg_labiocom": "Analysis",
    "treadmill_lc": "Analysis",
    "tugturn": "Analysis",
    "vek": "Analysis",
    "vaila_and_jump_help": "Analysis",
    "vailasprint_help": "Analysis",
    "vailasprint_br_help": "Analysis",
    "pynalty_help": "Analysis",
    "mp_facemesh_help": "Analysis",
    "soccerfield_vitruvian_dlt3d": "Analysis",
    "getpixelvideo": "Tools",
    "getpixelvideo_pt": "Tools",
    "vaila_stroboscopic": "Tools",
    "scout_vaila_pt": "Tools",
    "sam3dinov3_visualize": "ML",
    "sam3sapiens2_visualize": "ML",
    "help_reid_markers": "Processing",
}


@dataclass(frozen=True)
class HelpEntry:
    stem: str
    title: str
    category: str
    summary: str
    has_html: bool
    has_md: bool


def _read_vaila_version() -> str:
    text = VAILA_PY.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^Version:\s*([0-9.]+)", text, re.MULTILINE)
    return m.group(1) if m else "unknown"


def _extract_category(text: str) -> str | None:
    m = re.search(r"\*\*Category:\*\*\s*(.+)", text)
    if m:
        return m.group(1).strip().rstrip("|").strip()
    m = re.search(r"\|\s*\*\*Category\*\*\s*\|\s*([^|]+)\|", text)
    if m:
        return m.group(1).strip()
    return None


def _normalize_category(raw: str | None, stem: str) -> str:
    if stem in _GUIDE_STEMS:
        return "Guides"
    if stem in _STEM_CATEGORY:
        return _STEM_CATEGORY[stem]
    if not raw:
        return "Guides" if stem.endswith("_guide") else "Tools"
    key = raw.strip().lower()
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    # Prefix fallbacks
    if key.startswith("analysis") or "analysis" in key:
        return "Analysis"
    if key.startswith("ml") or "machine learning" in key or "markerless" in key:
        return "ML"
    if key.startswith("processing"):
        return "Processing"
    if key.startswith("tools") or key.startswith("data files"):
        return "Tools"
    if key.startswith("utils"):
        return "Utils"
    if key.startswith("visualization"):
        return "Visualization"
    return "Tools"


def _extract_title(text: str, stem: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s[2:].strip()
            # Drop leading emoji / decoration for cleaner catalog
            title = re.sub(r"^[^\w]*(?=\w)", "", title).strip() or stem
            return title
    return stem


def _is_noise_line(s: str) -> bool:
    """Skip banners, rules, TOC junk, and meta headers that are not summaries."""
    if not s:
        return True
    # Leading/trailing rule characters left after join
    s = re.sub(r"^[=\-─_~*\s]+", "", s)
    s = re.sub(r"[=\-─_~*\s]+$", "", s).strip()
    if not s:
        return True
    if re.fullmatch(r"[=\-─_~*]{4,}", s):
        return True
    if s.startswith("#") or s.startswith("|") or s.startswith("```"):
        return True
    if s.startswith("- ") or s.startswith("* ") or s.startswith("> "):
        return True
    low = s.lower()
    if low.startswith(
        (
            "script:",
            "script ",
            "project:",
            "author:",
            "version:",
            "python version",
            "update date",
            "update:",
            "date:",
            "email:",
            "github:",
            "http://",
            "https://",
            "file:",
            "category:",
            "lines:",
            "size:",
            "created:",
            "creation date",
            "last updated",
            "overview:",
            "description:",
            "main features:",
            "key features:",
            "features:",
            "usage:",
            "requirements:",
            "generated automatically",
            "📅",
        )
    ):
        return True
    if "generated automatically" in low:
        return True
    if low.startswith("1. ") and ("click" in low or "launch" in low):
        return True
    # Bare module filename as a banner line (underscores may already be stripped)
    if re.fullmatch(r"[\w.-]+\.py", s, flags=re.IGNORECASE):
        return True
    # Title-ish banner without a verb / sentence (e.g. "Cluster Data Analysis Toolkit...")
    if s.endswith(".py") and " " not in s.strip():
        return True
    letters = sum(c.isalpha() for c in s)
    return letters < 12


def _clean_summary(raw: str, *, min_len: int = 24) -> str | None:
    # Keep underscores in module names; only strip emphasis markers
    cleaned = re.sub(r"[*`#]+", "", raw)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = re.sub(r"^[=\-─_~*\s]+", "", cleaned)
    cleaned = re.sub(r"[=\-─_~*\s]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if _is_noise_line(cleaned):
        return None
    if cleaned.lower().startswith("|"):
        return None
    if len(cleaned) < min_len:
        return None
    # Prefer sentences that look like descriptions
    low = cleaned.lower()
    has_signal = any(
        k in low
        for k in (
            " this ",
            "this ",
            " script",
            " module",
            " tool",
            " analyze",
            " analyses",
            " process",
            " provide",
            " perform",
            " detect",
            " convert",
            " export",
            " import",
            " read ",
            " writes",
            " generate",
            " calculate",
            " reconstruct",
            " track",
            " segment",
            " help topic",
            " opens ",
            " fits ",
            " combines",
            " welcome",
            " designed",
            " supports",
            " merged",
            " normalize",
            " coupling",
            " purpose:",
            "este script",
            "este módulo",
        )
    )
    looks_like_prose = cleaned[0].isupper() and (" " in cleaned) and len(cleaned) > 40
    if not has_signal and not looks_like_prose:
        return None
    if len(cleaned) > 160:
        cleaned = cleaned[:157].rstrip() + "..."
    return cleaned


_DOC_BLOCK = re.compile(
    r"(?im)^(?:Description|Overview):\s*\n(?:-+\s*\n)?\s*((?:.+\n)+?)"
    r"(?:\n(?:Key Features|Main Features|Features|Requirements|Author|Date|Version|Usage|Update):|\n={5,}|\Z)"
)


def _extract_summary(text: str, stem: str) -> str:
    """Best-effort one-liner from Description / Overview sections."""
    for m_doc in _DOC_BLOCK.finditer(text):
        lines = []
        for ln in m_doc.group(1).splitlines():
            t = ln.strip()
            if not t or _is_noise_line(re.sub(r"[*`#]+", "", t).strip()):
                continue
            lines.append(t)
        if lines:
            got = _clean_summary(" ".join(lines))
            if got:
                return got

    for heading in (
        r"##\s*📋?\s*Description",
        r"##\s*📖?\s*Description",
        r"##\s*Description",
        r"##\s*Overview",
        r"##\s*What (?:it|this) does",
        r"##\s*About",
    ):
        m = re.search(heading + r"\s*\n+(.*?)(?:\n##|\Z)", text, re.IGNORECASE | re.DOTALL)
        if m:
            body = m.group(1).strip()
            nested = _DOC_BLOCK.search(body)
            if nested:
                joined = " ".join(
                    ln.strip()
                    for ln in nested.group(1).splitlines()
                    if ln.strip() and not _is_noise_line(ln.strip())
                )
                got = _clean_summary(joined)
                if got:
                    return got
            for line in body.splitlines():
                got = _clean_summary(line.strip(), min_len=16)
                if got:
                    return got

    past_title = False
    for line in text.splitlines():
        s = line.strip()
        if not past_title:
            if s.startswith("# "):
                past_title = True
            continue
        got = _clean_summary(s)
        if got:
            return got
    return f"Help topic for `{stem}`."


def collect_entries() -> list[HelpEntry]:
    entries: list[HelpEntry] = []
    for md_path in sorted(HELP_DIR.glob("*.md")):
        if md_path.name in SKIP_NAMES:
            continue
        stem = md_path.stem
        # Skip private / junk
        if stem.startswith("__"):
            continue
        text = md_path.read_text(encoding="utf-8", errors="replace")
        raw_cat = _extract_category(text)
        category = _normalize_category(raw_cat, stem)
        html_path = HELP_DIR / f"{stem}.html"
        entries.append(
            HelpEntry(
                stem=stem,
                title=_extract_title(text, stem),
                category=category,
                summary=_extract_summary(text, stem),
                has_html=html_path.is_file(),
                has_md=True,
            )
        )
    return entries


def _group(entries: list[HelpEntry]) -> dict[str, list[HelpEntry]]:
    grouped: dict[str, list[HelpEntry]] = defaultdict(list)
    for e in entries:
        grouped[e.category].append(e)
    for cat in grouped:
        grouped[cat].sort(key=lambda x: x.stem.lower())
    return grouped


def render_markdown(entries: list[HelpEntry], version: str, generated: str) -> str:
    grouped = _group(entries)
    cats_present = [c for c in CATEGORY_ORDER if grouped.get(c)]
    n = len(entries)
    lines: list[str] = [
        "# *vailá* — Help Index",
        "",
        "**Versatile Anarcho Integrated Liberation Ánalysis** — open-source Python 3.12 "
        "multimodal toolbox for biomechanical and movement analysis (IMU, MoCap, "
        "markerless 2D/3D, EMG, force plates, GNSS/GPS, and more), with a Tkinter desktop GUI.",
        "",
        "- **Frame A** — File manager (rename, import/export, copy/move, tree, find, SSH)",
        "- **Frame B** — Multimodal analysis pipelines",
        "- **Frame C** — Data, video/image, and visualization tools",
        "",
        "[Project documentation](../../docs/index.md) · "
        "[GitHub](https://github.com/vaila-multimodaltoolbox/vaila) · "
        "[README](../../README.md)",
        "",
        f"**Documented topics:** {n} | **Categories:** {len(cats_present)} | "
        f"**Generated on:** {generated} (v{version})",
        "",
        "This page lists every help topic under `vaila/help/` with links to HTML and Markdown.",
        "",
        "---",
        "",
    ]

    for cat in cats_present:
        items = grouped[cat]
        lines.append(f"## {cat} ({len(items)})")
        lines.append("")
        for e in items:
            lines.append(f"- **{e.stem}** — {e.summary}")
            link_bits: list[str] = []
            if e.has_html:
                link_bits.append(f"[HTML]({e.stem}.html)")
            if e.has_md:
                link_bits.append(f"[Markdown]({e.stem}.md)")
            if link_bits:
                lines.append(f"  - {' · '.join(link_bits)}")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "Regenerate this index:",
            "",
            "```bash",
            "uv run python bin/generate_help_index.py",
            "```",
            "",
            "© 2026 *vailá* — Multimodal Toolbox",
            "",
        ]
    )
    return "\n".join(lines)


def _html_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def render_html(entries: list[HelpEntry], version: str, generated: str) -> str:
    grouped = _group(entries)
    cats_present = [c for c in CATEGORY_ORDER if grouped.get(c)]
    n = len(entries)

    sections: list[str] = []
    for cat in cats_present:
        items = grouped[cat]
        cards: list[str] = []
        for e in items:
            links: list[str] = []
            if e.has_html:
                links.append(f'<a href="{_html_escape(e.stem)}.html">HTML</a>')
            if e.has_md:
                links.append(f'<a href="{_html_escape(e.stem)}.md">MD</a>')
            link_html = " | ".join(links) if links else ""
            cards.append(
                f"""                <div class="tool-card" data-name="{_html_escape(e.stem.lower())} {_html_escape(e.title.lower())}">
                    <div class="tool-title">{_html_escape(e.stem)}</div>
                    <p class="muted text-sm mb-sm">{_html_escape(e.summary)}</p>
                    <div class="tool-links">{link_html}</div>
                </div>"""
            )
        sections.append(
            f"""            <h2>{_html_escape(cat)} ({len(items)})</h2>
            <div class="grid-container">
{chr(10).join(cards)}
            </div>"""
        )

    body_sections = "\n\n".join(sections)

    return f"""<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>vailá — Help Index</title>
    <style>
        body {{
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            margin: 0;
            line-height: 1.55;
            background: #f6f7fb;
            font-size: 17px;
            color: #111;
        }}

        .container {{
            max-width: 1100px;
            margin: 28px auto;
            background: #fff;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(0, 0, 0, 0.06);
        }}

        h1 {{
            color: #111;
            text-align: center;
            border-bottom: 1px solid #e8e8e8;
            padding-bottom: 16px;
            margin: 4px 0 12px 0;
        }}

        h2 {{
            color: #111;
            margin-top: 36px;
            border-left: 4px solid #1e3a5f;
            background: #fafafa;
            padding: 10px 12px;
            border-radius: 0 10px 10px 0;
            font-size: 1.15rem;
        }}

        .tagline {{
            color: #333;
            margin: 0 0 14px 0;
        }}

        .frames {{
            margin: 0 0 18px 0;
            padding-left: 1.2em;
        }}

        .frames li {{
            margin-bottom: 0.25em;
        }}

        .links {{
            text-align: center;
            margin-bottom: 18px;
        }}

        .links a {{
            font-weight: 700;
            color: #1e3a5f;
            text-decoration: none;
        }}

        .links a:hover {{
            text-decoration: underline;
        }}

        .sep {{
            opacity: 0.55;
            padding: 0 6px;
        }}

        .topbar {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            margin: 18px 0 10px 0;
            padding: 12px;
            background: #fafafa;
            border: 1px solid #eee;
            border-radius: 12px;
        }}

        .searchbox {{
            flex: 1 1 420px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}

        .searchbox input {{
            width: 100%;
            padding: 10px 12px;
            border: 1px solid #d6d6d6;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
        }}

        .searchbox input:focus {{
            border-color: #7a7a7a;
            box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.06);
        }}

        .pill {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid #e3e3e3;
            background: #fff;
            font-size: 14px;
            color: #222;
        }}

        .banner {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            background: #f7f7f7;
            color: #111;
            padding: 12px 14px;
            border-radius: 12px;
            margin: 14px 0 22px 0;
            overflow-x: auto;
            font-size: 14px;
            border: 1px solid #e6e6e6;
        }}

        .grid-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}

        .tool-card {{
            border: 1px solid #ddd;
            border-radius: 12px;
            padding: 12px;
            background: #fff;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .tool-card:hover {{
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08);
            border-color: #c9c9c9;
        }}

        .tool-title {{
            font-weight: bold;
            color: #111;
            margin-bottom: 6px;
        }}

        .tool-links a {{
            font-size: 0.95em;
            margin-right: 8px;
            text-decoration: none;
            color: #1e3a5f;
        }}

        .tool-links a:hover {{
            text-decoration: underline;
        }}

        .muted {{
            color: #444;
        }}

        .text-sm {{
            font-size: 0.88em;
        }}

        .mb-sm {{
            margin: 0 0 10px 0;
        }}

        .is-hidden {{
            display: none !important;
        }}

        .footer {{
            margin-top: 40px;
            text-align: center;
            color: #555;
            font-size: 0.9em;
        }}

        code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.92em;
            background: #f3f4f6;
            padding: 0.1em 0.35em;
            border-radius: 0.4em;
        }}

        @media (max-width: 640px) {{
            .container {{
                margin: 14px auto;
                padding: 16px;
            }}
        }}
    </style>
</head>

<body>
    <div class="container">
        <h1><i>vailá</i> — Help Index</h1>

        <p class="tagline">
            <strong>Versatile Anarcho Integrated Liberation Ánalysis</strong> — open-source
            Python&nbsp;3.12 multimodal toolbox for biomechanical and movement analysis
            (IMU, MoCap, markerless 2D/3D, EMG, force plates, GNSS/GPS, and more),
            with a Tkinter desktop GUI.
        </p>

        <ul class="frames">
            <li><strong>Frame A</strong> — File manager (rename, import/export, copy/move, tree, find, SSH)</li>
            <li><strong>Frame B</strong> — Multimodal analysis pipelines</li>
            <li><strong>Frame C</strong> — Data, video/image, and visualization tools</li>
        </ul>

        <div class="links">
            <a href="../../docs/index.md">Project documentation</a>
            <span class="sep" aria-hidden="true">|</span>
            <a href="../../docs/help.html">Docs hub (HTML)</a>
            <span class="sep" aria-hidden="true">|</span>
            <a href="https://github.com/vaila-multimodaltoolbox/vaila">GitHub</a>
        </div>

        <div class="topbar">
            <div class="searchbox">
                <span class="pill">Search</span>
                <input id="toolSearch" type="text"
                    placeholder="Type a script name (e.g. imu, SAM, rec3d, tugturn)...">
            </div>
            <span class="pill">{n} topics · {len(cats_present)} categories</span>
        </div>

        <pre class="banner">Generated on {generated} · v{version} · uv run python bin/generate_help_index.py</pre>

{body_sections}

        <div class="footer">
            © 2026 <i>vailá</i> — Multimodal Toolbox
        </div>
    </div>

    <script>
        (function () {{
            const input = document.getElementById("toolSearch");
            if (!input) return;
            const cards = Array.from(document.querySelectorAll(".tool-card"));
            input.addEventListener("input", function () {{
                const q = (input.value || "").trim().toLowerCase();
                cards.forEach(function (card) {{
                    const name = card.getAttribute("data-name") || "";
                    const show = !q || name.indexOf(q) !== -1;
                    card.classList.toggle("is-hidden", !show);
                }});
            }});
        }})();
    </script>
</body>

</html>
"""


def main() -> int:
    if not HELP_DIR.is_dir():
        print(f"Help directory not found: {HELP_DIR}", file=sys.stderr)
        return 1

    entries = collect_entries()
    if not entries:
        print("No help markdown pages found.", file=sys.stderr)
        return 1

    version = _read_vaila_version()
    generated = date.today().strftime("%d/%m/%Y")

    md_path = HELP_DIR / "index.md"
    html_path = HELP_DIR / "index.html"
    md_path.write_text(render_markdown(entries, version, generated), encoding="utf-8")
    html_path.write_text(render_html(entries, version, generated), encoding="utf-8")

    grouped = _group(entries)
    print(f"Wrote {md_path.relative_to(REPO_ROOT)} and {html_path.relative_to(REPO_ROOT)}")
    print(f"Topics: {len(entries)} | version {version} | {generated}")
    for cat in CATEGORY_ORDER:
        if cat in grouped:
            print(f"  {cat}: {len(grouped[cat])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
