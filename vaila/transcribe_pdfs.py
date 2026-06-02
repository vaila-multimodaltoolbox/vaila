#!/usr/bin/env python3
"""
Project: vailá
Script: transcribe_pdfs.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Create: 28 May 2026
Update: 01 June 2026
Version: 0.3.47

Description:
    Batch transcription for scanned, typed, or handwritten PDF exams.
    The module can extract native PDF text or render pages and send them
    to a vision backend through the Gemini CLI. It writes one .txt file
    and one .report.json file per PDF, plus a batch_report.csv summary.

Usage:
    uv run python -m vaila.transcribe_pdfs --gui
    uv run python -m vaila.transcribe_pdfs --input-dir originais --output-dir transcritas_originais

Requirements:
    - Python 3.12
    - Pillow
    - Poppler command-line tools: pdfinfo or qpdf, pdftotext, pdftoppm
    - Gemini CLI for vision transcription mode

License:
    GNU Affero General Public License v3.0
"""

from __future__ import annotations

import argparse
import contextlib
import os
import csv
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


BLANK_MARKER = "[Página aparentemente em branco ou sem respostas manuscritas legíveis]"
LOW_CONFIDENCE_MARKERS = (
    "[ilegível]",
    "[revisar]",
    "não reconhecido",
    "sem respostas manuscritas legíveis",
)
# Default root can be overridden via VAILA_PAE_ROOT environment variable.
# Falls back to ~/vaila_data/pae if not set.
DEFAULT_PAE_ROOT = Path(os.environ.get("VAILA_PAE_ROOT", str(Path.home() / "vaila_data" / "pae")))
DEFAULT_INPUT_DIR = DEFAULT_PAE_ROOT / "originais"
DEFAULT_OUTPUT_DIR = DEFAULT_PAE_ROOT / "transcritas_originais"
DEFAULT_DEBUG_DIR = DEFAULT_PAE_ROOT / "transcritas_originais_debug" / "pages"
ProgressCallback = Callable[[str], None]


@dataclass
class PageResult:
    page: int
    text: str
    mode: str
    low_confidence: bool = False
    blank: bool = False
    error: str | None = None


@dataclass
class PdfReport:
    input_file: str
    output_txt: str
    pages_total: int = 0
    pages_processed: int = 0
    mode_by_page: dict[str, str] = field(default_factory=dict)
    blank_pages: list[int] = field(default_factory=list)
    low_confidence_pages: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"


def run_command(
    cmd: list[str],
    timeout: int | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
        cwd=cwd,
    )


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required command not found: {name}")


def safe_output_path(output_dir: Path, stem: str, suffix: str, overwrite: bool) -> Path:
    candidate = output_dir / f"{stem}{suffix}"
    if overwrite or not candidate.exists():
        return candidate
    version = 2
    while True:
        candidate = output_dir / f"{stem}_v{version}{suffix}"
        if not candidate.exists():
            return candidate
        version += 1


def collect_pdfs(args: argparse.Namespace) -> list[Path]:
    pdfs: list[Path] = []
    if args.input_file:
        input_file = Path(args.input_file).expanduser()
        if input_file.suffix.lower() == ".pdf":
            pdfs.append(input_file)
    if args.input_dir:
        root = Path(args.input_dir).expanduser()
        pattern = "**/*" if args.recursive else "*"
        pdfs.extend(sorted(path for path in root.glob(pattern) if path.suffix.lower() == ".pdf"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for pdf in pdfs:
        if not pdf.exists() or pdf.suffix.lower() != ".pdf":
            continue
        resolved = pdf.resolve()
        if resolved in seen:
            continue
        if args.skip_presence and "presenca" in pdf.name.lower():
            continue
        seen.add(resolved)
        unique.append(pdf)
    return unique


def page_count(pdf: Path) -> int:
    if shutil.which("pdfinfo"):
        result = run_command(["pdfinfo", str(pdf)], timeout=60)
        if result.returncode == 0:
            match = re.search(r"^Pages:\s+(\d+)", result.stdout, flags=re.MULTILINE)
            if match:
                return int(match.group(1))
    if shutil.which("qpdf"):
        result = run_command(["qpdf", "--show-npages", str(pdf)], timeout=60)
        if result.returncode == 0:
            return int(result.stdout.strip())
    raise RuntimeError(f"Could not count pages: {pdf}")


def extract_native_text(pdf: Path) -> str:
    result = run_command(["pdftotext", "-layout", str(pdf), "-"], timeout=120)
    if result.returncode != 0:
        return ""
    return result.stdout.replace("\f", "\n").strip()


def native_text_is_useful(text: str, min_chars: int) -> bool:
    compact = re.sub(r"\s+", "", text)
    alpha_count = sum(ch.isalpha() for ch in compact)
    return len(compact) >= min_chars and alpha_count >= max(20, min_chars // 3)


def split_native_pages(text: str, total_pages: int) -> list[str]:
    pages = text.split("\f")
    if len(pages) == 1:
        pages = re.split(r"\n\s*\n\s*\n+", text)
    if len(pages) < total_pages:
        pages.extend([""] * (total_pages - len(pages)))
    return pages[:total_pages]


def render_pdf_pages(
    pdf: Path, image_dir: Path, dpi: int, max_width: int, jpeg_quality: int
) -> list[Path]:
    require_binary("pdftoppm")
    image_dir.mkdir(parents=True, exist_ok=True)
    prefix = image_dir / pdf.stem
    result = run_command(["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)], timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Could not render PDF: {pdf}")

    pngs = sorted(image_dir.glob(f"{pdf.stem}-*.png"))
    jpgs: list[Path] = []
    for png in pngs:
        jpg = png.with_suffix(".jpg")
        with Image.open(png) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
            img.save(jpg, "JPEG", quality=jpeg_quality, optimize=True)
        png.unlink(missing_ok=True)
        jpgs.append(jpg)
    return jpgs


def build_vision_prompt(pdf: Path, image_paths: list[Path], example_format: str) -> str:
    page_refs = "\n".join(f"Página {i + 1}: @{path.name}" for i, path in enumerate(image_paths))
    return f"""Transcreva fielmente as respostas manuscritas deste PDF de prova.

Arquivo original: {pdf.name}

Regras obrigatórias:
- Preserve português, grafia, nomes, números USP e numeração das questões quando legíveis.
- Não corrija gramática do aluno.
- Não invente conteúdo.
- Se uma palavra estiver duvidosa, use [palavra incerta: ...] ou [revisar].
- Se houver trecho ilegível, use [ilegível].
- Se a página não tiver resposta manuscrita legível, escreva: {BLANK_MARKER}
- Crie uma seção separada para cada página, sempre no formato [Página N].
- Não agrupe páginas, mesmo se páginas consecutivas estiverem em branco.
- Use exatamente o estilo deste exemplo:

{example_format}

Imagens do PDF:
{page_refs}

Retorne apenas o conteúdo final do arquivo .txt, sem cercas Markdown.
"""


def clean_gemini_output(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("[ExtensionManager]"):
            continue
        if "Validation failed: Agent Definition" in line:
            continue
        if line.startswith("tools.") or line.startswith("Ripgrep is not available"):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"^```(?:text)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def call_gemini(
    prompt: str,
    model: str | None,
    retries: int,
    timeout: int,
    cwd: Path,
) -> tuple[str, list[str]]:
    cmd = ["gemini", "-p", prompt, "--skip-trust", "--output-format", "text", "--extensions", ""]
    if model:
        cmd.extend(["--model", model])

    warnings: list[str] = []
    for attempt in range(1, retries + 1):
        result = run_command(cmd, timeout=timeout, cwd=cwd)
        output = clean_gemini_output(
            (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        )
        if result.returncode == 0 and output:
            return output, warnings
        message = output or f"gemini exited with code {result.returncode}"
        warnings.append(f"Gemini attempt {attempt} failed: {message[:500]}")
        if attempt < retries:
            time.sleep(min(20 * attempt, 60))
    raise RuntimeError(warnings[-1] if warnings else "Gemini transcription failed")


def fallback_native_txt(pdf: Path, total_pages: int, text: str) -> str:
    pages = split_native_pages(text, total_pages)
    parts = [f"Transcrição do arquivo: {pdf.name}", ""]
    for index, page_text in enumerate(pages, start=1):
        parts.extend([f"[Página {index}]", "", page_text.strip() or BLANK_MARKER, ""])
    return "\n".join(parts).strip() + "\n"


def analyze_page_text(text: str) -> tuple[bool, bool]:
    lowered = text.lower()
    blank = BLANK_MARKER.lower() in lowered or "aparentemente em branco" in lowered
    low_confidence = blank or any(marker.lower() in lowered for marker in LOW_CONFIDENCE_MARKERS)
    return blank, low_confidence


def extract_page_blocks(transcription: str, total_pages: int) -> list[PageResult]:
    results: list[PageResult] = []
    for page in range(1, total_pages + 1):
        pattern = rf"\[P[áa]gina\s+{page}\](.*?)(?=\n\[P[áa]gina\s+{page + 1}\]|\Z)"
        match = re.search(pattern, transcription, flags=re.IGNORECASE | re.DOTALL)
        text = match.group(1).strip() if match else ""
        blank, low_conf = analyze_page_text(text)
        results.append(
            PageResult(page=page, text=text, mode="vision", blank=blank, low_confidence=low_conf)
        )
    return results


def process_pdf(
    args: argparse.Namespace, pdf: Path, output_dir: Path, debug_root: Path, example_format: str
) -> PdfReport:
    stem = f"transcricao_{pdf.stem}"
    output_txt = safe_output_path(output_dir, stem, ".txt", args.overwrite)
    output_report = output_txt.with_suffix(".report.json")
    report = PdfReport(input_file=str(pdf), output_txt=str(output_txt))

    try:
        total_pages = page_count(pdf)
        report.pages_total = total_pages
        native_text = extract_native_text(pdf)
        use_native = args.mode in {"native", "auto"} and native_text_is_useful(
            native_text, args.min_native_chars
        )

        if args.mode == "native" or use_native:
            content = fallback_native_txt(pdf, total_pages, native_text)
            mode = "native"
        elif args.mode == "ocr":
            raise RuntimeError(
                "Local OCR mode requires Tesseract/pytesseract; use --mode vision for handwritten exams here."
            )
        else:
            image_dir = debug_root / pdf.stem
            images = render_pdf_pages(
                pdf, image_dir, args.dpi, args.max_image_width, args.jpeg_quality
            )
            prompt = build_vision_prompt(pdf, images, example_format)
            content, warnings = call_gemini(
                prompt, args.gemini_model, args.retries, args.timeout, image_dir
            )
            report.warnings.extend(warnings)
            mode = "vision"
            if not args.save_page_images:
                for image in images:
                    image.unlink(missing_ok=True)
                with contextlib.suppress(OSError):
                    image_dir.rmdir()

        output_txt.write_text(content.strip() + "\n", encoding="utf-8")
        page_results = extract_page_blocks(content, report.pages_total)
        report.pages_processed = report.pages_total
        for page_result in page_results:
            report.mode_by_page[str(page_result.page)] = mode
            if page_result.blank:
                report.blank_pages.append(page_result.page)
            if page_result.low_confidence:
                report.low_confidence_pages.append(page_result.page)
        if mode == "vision":
            report.warnings.append(
                "Documento manuscrito transcrito por backend de visão; revisar trechos duvidosos."
            )
    except Exception as exc:  # noqa: BLE001 - batch should continue after per-file failures.
        report.status = "error"
        report.errors.append(str(exc))
        output_txt.write_text(
            f"Transcrição do arquivo: {pdf.name}\n\n[erro de processamento: {exc}]\n",
            encoding="utf-8",
        )

    output_report.write_text(
        json.dumps(report.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def write_batch_report(reports: list[PdfReport], output_dir: Path) -> Path:
    csv_path = output_dir / "batch_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input_file",
                "output_txt",
                "pages_total",
                "pages_processed",
                "blank_pages",
                "low_confidence_pages",
                "status",
                "errors",
            ],
        )
        writer.writeheader()
        for report in reports:
            writer.writerow(
                {
                    "input_file": report.input_file,
                    "output_txt": report.output_txt,
                    "pages_total": report.pages_total,
                    "pages_processed": report.pages_processed,
                    "blank_pages": ";".join(map(str, report.blank_pages)),
                    "low_confidence_pages": ";".join(map(str, report.low_confidence_pages)),
                    "status": report.status,
                    "errors": " | ".join(report.errors),
                }
            )
    return csv_path


def load_example_format(path: Path | None) -> str:
    if path and path.exists():
        return path.read_text(encoding="utf-8").strip()
    return """Transcrição do arquivo: aluno.pdf

Nome: Nome Sobrenome
Número USP: 00000000

[Página 1]

1-) Texto transcrito.

[Página 2]

[Página aparentemente em branco ou sem respostas manuscritas legíveis]"""


def process_pdfs(
    args: argparse.Namespace,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[PdfReport], Path, Path]:
    """Run PDF transcription and return per-file reports plus batch CSV path."""
    if not args.output_dir:
        raise RuntimeError("Output directory is required.")

    output_dir = Path(args.output_dir).expanduser()
    debug_dir = Path(args.debug_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir = debug_dir

    if shutil.which("pdfinfo") is None and shutil.which("qpdf") is None:
        raise RuntimeError("Required command not found: pdfinfo or qpdf")
    require_binary("pdftotext")
    if args.mode in {"auto", "vision"}:
        require_binary("gemini")

    pdfs = collect_pdfs(args)
    if not pdfs:
        raise RuntimeError("No PDFs found.")

    if progress_callback:
        progress_callback(f"Found {len(pdfs)} PDF file(s).")
        progress_callback(f"Output directory: {output_dir}")

    example_format = load_example_format(args.example_format)
    reports: list[PdfReport] = []
    for index, pdf in enumerate(tqdm(pdfs, desc="Transcribing PDFs", unit="pdf"), start=1):
        if progress_callback:
            progress_callback(f"[{index}/{len(pdfs)}] Processing {pdf.name}")
        report = process_pdf(args, pdf, output_dir, debug_dir, example_format)
        reports.append(report)
        if progress_callback:
            status = "OK" if report.status == "ok" else "ERROR"
            progress_callback(f"{status}: {pdf.name} -> {report.output_txt}")
            for error in report.errors:
                progress_callback(f"  Error: {error}")

    csv_path = write_batch_report(reports, output_dir)
    if progress_callback:
        progress_callback(f"Batch report: {csv_path}")
    return reports, csv_path, output_dir


def summarize_reports(reports: list[PdfReport], csv_path: Path, output_dir: Path) -> str:
    ok = sum(1 for report in reports if report.status == "ok")
    errors = len(reports) - ok
    pages = sum(report.pages_processed for report in reports)
    low_conf = sum(1 for report in reports if report.low_confidence_pages)
    return "\n".join(
        [
            f"PDFs processed: {len(reports)}",
            f"TXT files created: {len(reports)}",
            f"Pages processed: {pages}",
            f"Documents with low confidence: {low_conf}",
            f"Errors: {errors}",
            f"Output directory: {output_dir}",
            f"Batch report: {csv_path}",
        ]
    )


class TranscribePDFsGUI:
    """Tkinter launcher for the PDF transcription workflow."""

    def __init__(self, parent: tk.Misc | None = None) -> None:
        self.parent = parent
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("vailá PDF Transcription")
        self.window.geometry("900x720")
        self.worker: threading.Thread | None = None

        default_base = DEFAULT_PAE_ROOT if DEFAULT_PAE_ROOT.exists() else Path.home()
        default_input = DEFAULT_INPUT_DIR if DEFAULT_INPUT_DIR.exists() else default_base
        default_output = (
            DEFAULT_OUTPUT_DIR if DEFAULT_PAE_ROOT.exists() else Path.cwd() / "transcritas_pdfs"
        )
        default_debug = (
            DEFAULT_DEBUG_DIR
            if DEFAULT_PAE_ROOT.exists()
            else Path.cwd() / "transcritas_pdfs_debug" / "pages"
        )

        self.source_mode = tk.StringVar(value="dir")
        self.input_dir_var = tk.StringVar(value=str(default_input))
        self.input_file_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value=str(default_output))
        self.debug_dir_var = tk.StringVar(value=str(default_debug))
        self.mode_var = tk.StringVar(value="auto")
        self.dpi_var = tk.StringVar(value="180")
        self.max_width_var = tk.StringVar(value="1600")
        self.jpeg_quality_var = tk.StringVar(value="85")
        self.min_native_var = tk.StringVar(value="120")
        self.retries_var = tk.StringVar(value="4")
        self.timeout_var = tk.StringVar(value="600")
        self.gemini_model_var = tk.StringVar(value="")
        self.recursive_var = tk.BooleanVar(value=False)
        self.skip_presence_var = tk.BooleanVar(value=True)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.save_images_var = tk.BooleanVar(value=False)
        self.source_feedback_var = tk.StringVar(value="")

        self._build_ui()
        self.source_mode.trace_add("write", lambda *_: self._refresh_source_feedback())
        self.recursive_var.trace_add("write", lambda *_: self._refresh_source_feedback())
        self.skip_presence_var.trace_add("write", lambda *_: self._refresh_source_feedback())
        self._refresh_source_feedback()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.window, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        title = ttk.Label(
            main,
            text="PDF Transcription for Exams and Handwritten Documents",
            font=("Arial", 14, "bold"),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 8))

        source_frame = ttk.LabelFrame(main, text="Source PDFs")
        source_frame.grid(row=1, column=0, sticky="ew", pady=5)
        source_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            source_frame,
            text="Directory",
            variable=self.source_mode,
            value="dir",
        ).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(source_frame, textvariable=self.input_dir_var).grid(
            row=0, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Button(source_frame, text="Browse", command=self._browse_input_dir).grid(
            row=0, column=2, padx=6, pady=4
        )

        ttk.Radiobutton(
            source_frame,
            text="Single PDF",
            variable=self.source_mode,
            value="file",
        ).grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(source_frame, textvariable=self.input_file_var).grid(
            row=1, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Button(source_frame, text="Browse", command=self._browse_input_file).grid(
            row=1, column=2, padx=6, pady=4
        )
        ttk.Label(
            source_frame,
            textvariable=self.source_feedback_var,
            foreground="#1f4e79",
            wraplength=760,
        ).grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))

        output_frame = ttk.LabelFrame(main, text="Outputs")
        output_frame.grid(row=2, column=0, sticky="ew", pady=5)
        output_frame.columnconfigure(1, weight=1)
        ttk.Label(output_frame, text="Text/report directory").grid(
            row=0, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Entry(output_frame, textvariable=self.output_dir_var).grid(
            row=0, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Button(output_frame, text="Browse", command=self._browse_output_dir).grid(
            row=0, column=2, padx=6, pady=4
        )
        ttk.Label(output_frame, text="Debug page images").grid(
            row=1, column=0, sticky="w", padx=6, pady=4
        )
        ttk.Entry(output_frame, textvariable=self.debug_dir_var).grid(
            row=1, column=1, sticky="ew", padx=6, pady=4
        )
        ttk.Button(output_frame, text="Browse", command=self._browse_debug_dir).grid(
            row=1, column=2, padx=6, pady=4
        )

        options_frame = ttk.LabelFrame(main, text="Options")
        options_frame.grid(row=3, column=0, sticky="ew", pady=5)
        for column in range(6):
            options_frame.columnconfigure(column, weight=1)

        ttk.Label(options_frame, text="Mode").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        mode_combo = ttk.Combobox(
            options_frame,
            textvariable=self.mode_var,
            values=("auto", "vision", "native"),
            width=12,
            state="readonly",
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Label(options_frame, text="Gemini model").grid(
            row=0, column=2, sticky="w", padx=6, pady=4
        )
        ttk.Entry(options_frame, textvariable=self.gemini_model_var, width=22).grid(
            row=0, column=3, columnspan=3, sticky="ew", padx=6, pady=4
        )

        fields = [
            ("DPI", self.dpi_var),
            ("Max width", self.max_width_var),
            ("JPEG quality", self.jpeg_quality_var),
            ("Min native chars", self.min_native_var),
            ("Retries", self.retries_var),
            ("Timeout s", self.timeout_var),
        ]
        for index, (label, variable) in enumerate(fields):
            row = 1 + index // 3
            column = (index % 3) * 2
            ttk.Label(options_frame, text=label).grid(
                row=row, column=column, sticky="w", padx=6, pady=4
            )
            ttk.Entry(options_frame, textvariable=variable, width=10).grid(
                row=row, column=column + 1, sticky="w", padx=6, pady=4
            )

        checks_frame = ttk.Frame(options_frame)
        checks_frame.grid(row=3, column=0, columnspan=6, sticky="w", padx=2, pady=4)
        ttk.Checkbutton(checks_frame, text="Recursive", variable=self.recursive_var).pack(
            side="left", padx=6
        )
        ttk.Checkbutton(
            checks_frame, text="Skip presença/lista files", variable=self.skip_presence_var
        ).pack(side="left", padx=6)
        ttk.Checkbutton(checks_frame, text="Overwrite", variable=self.overwrite_var).pack(
            side="left", padx=6
        )
        ttk.Checkbutton(checks_frame, text="Save page images", variable=self.save_images_var).pack(
            side="left", padx=6
        )

        actions = ttk.Frame(main)
        actions.grid(row=4, column=0, sticky="ew", pady=8)
        self.run_button = ttk.Button(
            actions, text="Transcribe PDFs", command=self._start_processing
        )
        self.run_button.pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Help", command=self._open_help).pack(side="left", padx=8)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=8)

        log_frame = ttk.LabelFrame(main, text="Log")
        log_frame.grid(row=5, column=0, sticky="nsew", pady=5)
        main.rowconfigure(5, weight=1)
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=16, wrap=tk.WORD, state="disabled"
        )
        self.log_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _source_pdf_is_included(self, pdf: Path) -> bool:
        if pdf.suffix.lower() != ".pdf":
            return False
        return not (self.skip_presence_var.get() and "presenca" in pdf.name.lower())

    def _selected_source_pdfs(self) -> list[Path]:
        if self.source_mode.get() == "file":
            pdf = Path(self.input_file_var.get()).expanduser()
            if pdf.is_file() and self._source_pdf_is_included(pdf):
                return [pdf]
            return []

        root = Path(self.input_dir_var.get()).expanduser()
        if not root.is_dir():
            return []
        pattern = "**/*" if self.recursive_var.get() else "*"
        return sorted(
            pdf for pdf in root.glob(pattern) if pdf.is_file() and self._source_pdf_is_included(pdf)
        )

    def _source_feedback_message(self) -> str:
        try:
            if self.source_mode.get() == "file":
                raw_path = self.input_file_var.get().strip()
                if not raw_path:
                    return "No PDF file selected."
                pdf = Path(raw_path).expanduser()
                if not pdf.exists():
                    return f"File not found: {pdf}"
                if pdf.suffix.lower() != ".pdf":
                    return f"Selected file is not a PDF: {pdf.name}"
                if self.skip_presence_var.get() and "presenca" in pdf.name.lower():
                    return f"Selected PDF is skipped by presença/lista filter: {pdf.name}"
                return f"Selected file: {pdf.name} (1 PDF ready)."

            root = Path(self.input_dir_var.get().strip() or ".").expanduser()
            if not root.exists():
                return f"Directory not found: {root}"
            if not root.is_dir():
                return f"Selected source is not a directory: {root}"
            count = len(self._selected_source_pdfs())
            scope = "recursive" if self.recursive_var.get() else "top-level"
            if count == 0:
                return f"Selected directory: {root} (no PDFs found, {scope})."
            return f"Selected directory: {root} ({count} PDF(s) ready, {scope})."
        except OSError as exc:
            return f"Could not inspect selected source: {exc}"

    def _refresh_source_feedback(self, log: bool = False) -> None:
        message = self._source_feedback_message()
        self.source_feedback_var.set(message)
        if hasattr(self, "status_var"):
            self.status_var.set(message)
        if log and hasattr(self, "log_text"):
            self._append_log(message)

    def _browse_input_dir(self) -> None:
        directory = filedialog.askdirectory(
            title="Select directory with PDFs",
            initialdir=self.input_dir_var.get() or str(Path.home()),
            parent=self.window,
        )
        if directory:
            self.source_mode.set("dir")
            self.input_dir_var.set(directory)
            self._refresh_source_feedback(log=True)

    def _browse_input_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select PDF file",
            initialdir=self.input_dir_var.get() or str(Path.home()),
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            parent=self.window,
        )
        if file_path:
            self.source_mode.set("file")
            self.input_file_var.set(file_path)
            self._refresh_source_feedback(log=True)

    def _browse_output_dir(self) -> None:
        directory = filedialog.askdirectory(
            title="Select output directory",
            initialdir=self.output_dir_var.get() or str(Path.home()),
            parent=self.window,
        )
        if directory:
            self.output_dir_var.set(directory)
            self.status_var.set(f"Output directory selected: {directory}")
            self._append_log(f"Output directory selected: {directory}")

    def _browse_debug_dir(self) -> None:
        directory = filedialog.askdirectory(
            title="Select debug page-image directory",
            initialdir=self.debug_dir_var.get() or str(Path.home()),
            parent=self.window,
        )
        if directory:
            self.debug_dir_var.set(directory)
            self.status_var.set(f"Debug image directory selected: {directory}")
            self._append_log(f"Debug image directory selected: {directory}")

    def _open_help(self) -> None:
        help_html = Path(__file__).resolve().parent / "help" / "transcribe_pdfs.html"
        if help_html.exists():
            webbrowser.open(help_html.as_uri())
        else:
            messagebox.showwarning(
                "Help",
                "Help file not found. See vaila/help/transcribe_pdfs.md.",
                parent=self.window,
            )

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message.rstrip() + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _log(self, message: str) -> None:
        with contextlib.suppress(tk.TclError):
            self.window.after(0, self._append_log, message)

    def _set_running(self, running: bool) -> None:
        self.run_button.configure(state="disabled" if running else "normal")
        self.status_var.set("Running..." if running else "Ready")

    @staticmethod
    def _read_int(value: str, label: str, minimum: int, maximum: int | None = None) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if parsed < minimum:
            raise ValueError(f"{label} must be >= {minimum}.")
        if maximum is not None and parsed > maximum:
            raise ValueError(f"{label} must be <= {maximum}.")
        return parsed

    def _build_args(self) -> argparse.Namespace:
        input_dir: Path | None = None
        input_file: Path | None = None
        if self.source_mode.get() == "file":
            input_file = Path(self.input_file_var.get()).expanduser()
            if not input_file.is_file():
                raise ValueError("Select a valid PDF file.")
            if input_file.suffix.lower() != ".pdf":
                raise ValueError("Input file must be a PDF.")
        else:
            input_dir = Path(self.input_dir_var.get()).expanduser()
            if not input_dir.is_dir():
                raise ValueError("Select a valid directory containing PDFs.")

        output_dir = Path(self.output_dir_var.get()).expanduser()
        debug_dir = Path(self.debug_dir_var.get()).expanduser()
        model = self.gemini_model_var.get().strip() or None

        return argparse.Namespace(
            input_dir=str(input_dir) if input_dir else None,
            input_file=str(input_file) if input_file else None,
            output_dir=str(output_dir),
            mode=self.mode_var.get(),
            lang="por+eng",
            dpi=self._read_int(self.dpi_var.get(), "DPI", 72, 600),
            max_image_width=self._read_int(self.max_width_var.get(), "Max width", 400, 4000),
            jpeg_quality=self._read_int(self.jpeg_quality_var.get(), "JPEG quality", 30, 100),
            recursive=self.recursive_var.get(),
            save_page_images=self.save_images_var.get(),
            csv_report=True,
            overwrite=self.overwrite_var.get(),
            skip_presence=self.skip_presence_var.get(),
            min_native_chars=self._read_int(self.min_native_var.get(), "Min native chars", 1),
            example_format=None,
            debug_dir=debug_dir,
            gemini_model=model,
            retries=self._read_int(self.retries_var.get(), "Retries", 1, 20),
            timeout=self._read_int(self.timeout_var.get(), "Timeout", 30),
        )

    def _start_processing(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            args = self._build_args()
        except ValueError as exc:
            messagebox.showerror("PDF Transcription", str(exc), parent=self.window)
            return

        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")
        self._refresh_source_feedback(log=True)
        self._set_running(True)
        self.worker = threading.Thread(target=self._worker, args=(args,), daemon=True)
        self.worker.start()

    def _worker(self, args: argparse.Namespace) -> None:
        try:
            reports, csv_path, output_dir = process_pdfs(args, progress_callback=self._log)
            summary = summarize_reports(reports, csv_path, output_dir)
            self._log(summary)
            errors = sum(1 for report in reports if report.status != "ok")

            def finish_success() -> None:
                self._set_running(False)
                title = "Completed with errors" if errors else "Completed"
                messagebox.showinfo(title, summary, parent=self.window)

            self.window.after(0, finish_success)
        except Exception as exc:  # noqa: BLE001 - GUI must surface all backend errors.
            error_message = str(exc)
            self._log(f"ERROR: {error_message}")

            def finish_error() -> None:
                self._set_running(False)
                messagebox.showerror("PDF Transcription Error", error_message, parent=self.window)

            self.window.after(0, finish_error)


def run_gui(parent: tk.Misc | None = None) -> TranscribePDFsGUI:
    """Open the PDF transcription GUI."""
    app = TranscribePDFsGUI(parent)
    if parent is None:
        app.window.mainloop()
    return app


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe scanned or handwritten PDFs in batch.")
    parser.add_argument("--gui", action="store_true", help="Open the Tkinter GUI.")
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--input-dir", help="Directory containing PDFs.")
    source.add_argument("--input-file", help="Single PDF file.")
    parser.add_argument("--output-dir", help="Directory for .txt and reports.")
    parser.add_argument("--mode", choices=["auto", "native", "ocr", "vision"], default="auto")
    parser.add_argument("--lang", default="por+eng", help="Reserved for OCR mode.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-image-width", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-page-images", action="store_true")
    parser.add_argument("--csv-report", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-presence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip files whose names contain 'presenca' (default: true).",
    )
    parser.add_argument("--min-native-chars", type=int, default=120)
    parser.add_argument("--example-format", type=Path)
    parser.add_argument("--debug-dir", type=Path, default=DEFAULT_DEBUG_DIR)
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.gui:
        return args
    if not args.input_dir and not args.input_file:
        parser.error("one of --input-dir or --input-file is required unless --gui is used")
    if not args.output_dir:
        parser.error("--output-dir is required unless --gui is used")
    return args


def main(argv: list[str] | None = None) -> int:
    raw_args = sys.argv[1:] if argv is None else argv
    if not raw_args:
        run_gui()
        return 0

    args = parse_args(raw_args)
    if args.gui:
        run_gui()
        return 0

    try:
        reports, csv_path, output_dir = process_pdfs(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = summarize_reports(reports, csv_path, output_dir)
    print(summary)
    errors = sum(1 for report in reports if report.status != "ok")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
