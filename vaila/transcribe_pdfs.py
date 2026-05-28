#!/usr/bin/env python3
"""Batch transcription for scanned or handwritten PDF exams."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


BLANK_MARKER = "[Página aparentemente em branco ou sem respostas manuscritas legíveis]"
LOW_CONFIDENCE_MARKERS = ("[ilegível]", "[revisar]", "não reconhecido", "sem respostas manuscritas legíveis")


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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
        pdfs.append(Path(args.input_file))
    if args.input_dir:
        root = Path(args.input_dir)
        pattern = "**/*.pdf" if args.recursive else "*.pdf"
        pdfs.extend(sorted(root.glob(pattern)))
    unique = []
    seen = set()
    for pdf in pdfs:
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


def render_pdf_pages(pdf: Path, image_dir: Path, dpi: int, max_width: int, jpeg_quality: int) -> list[Path]:
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
        output = clean_gemini_output((result.stdout or "") + ("\n" + result.stderr if result.stderr else ""))
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
        results.append(PageResult(page=page, text=text, mode="vision", blank=blank, low_confidence=low_conf))
    return results


def process_pdf(args: argparse.Namespace, pdf: Path, output_dir: Path, debug_root: Path, example_format: str) -> PdfReport:
    stem = f"transcricao_{pdf.stem}"
    output_txt = safe_output_path(output_dir, stem, ".txt", args.overwrite)
    output_report = output_txt.with_suffix(".report.json")
    report = PdfReport(input_file=str(pdf), output_txt=str(output_txt))

    try:
        total_pages = page_count(pdf)
        report.pages_total = total_pages
        native_text = extract_native_text(pdf)
        use_native = args.mode in {"native", "auto"} and native_text_is_useful(native_text, args.min_native_chars)

        if args.mode == "native" or use_native:
            content = fallback_native_txt(pdf, total_pages, native_text)
            mode = "native"
        elif args.mode == "ocr":
            raise RuntimeError("Local OCR mode requires Tesseract/pytesseract; use --mode vision for handwritten exams here.")
        else:
            image_dir = debug_root / pdf.stem
            images = render_pdf_pages(pdf, image_dir, args.dpi, args.max_image_width, args.jpeg_quality)
            prompt = build_vision_prompt(pdf, images, example_format)
            content, warnings = call_gemini(prompt, args.gemini_model, args.retries, args.timeout, image_dir)
            report.warnings.extend(warnings)
            mode = "vision"
            if not args.save_page_images:
                for image in images:
                    image.unlink(missing_ok=True)
                try:
                    image_dir.rmdir()
                except OSError:
                    pass

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
            report.warnings.append("Documento manuscrito transcrito por backend de visão; revisar trechos duvidosos.")
    except Exception as exc:  # noqa: BLE001 - batch should continue after per-file failures.
        report.status = "error"
        report.errors.append(str(exc))
        output_txt.write_text(
            f"Transcrição do arquivo: {pdf.name}\n\n[erro de processamento: {exc}]\n",
            encoding="utf-8",
        )

    output_report.write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe scanned or handwritten PDFs in batch.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", help="Directory containing PDFs.")
    source.add_argument("--input-file", help="Single PDF file.")
    parser.add_argument("--output-dir", required=True, help="Directory for .txt and reports.")
    parser.add_argument("--mode", choices=["auto", "native", "ocr", "vision"], default="auto")
    parser.add_argument("--lang", default="por+eng", help="Reserved for OCR mode.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-image-width", type=int, default=1600)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-page-images", action="store_true")
    parser.add_argument("--csv-report", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-presence", action="store_true", default=True)
    parser.add_argument("--min-native-chars", type=int, default=120)
    parser.add_argument("--example-format", type=Path)
    parser.add_argument("--debug-dir", type=Path, default=Path("transcritas_originais_debug/pages"))
    parser.add_argument("--gemini-model", default=None)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.debug_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("pdfinfo") is None and shutil.which("qpdf") is None:
        raise RuntimeError("Required command not found: pdfinfo or qpdf")
    require_binary("pdftotext")
    if args.mode in {"auto", "vision"}:
        require_binary("gemini")

    pdfs = collect_pdfs(args)
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        return 1

    example_format = load_example_format(args.example_format)
    reports: list[PdfReport] = []
    for pdf in tqdm(pdfs, desc="Transcribing PDFs", unit="pdf"):
        reports.append(process_pdf(args, pdf, output_dir, args.debug_dir, example_format))

    csv_path = write_batch_report(reports, output_dir)
    ok = sum(1 for report in reports if report.status == "ok")
    errors = len(reports) - ok
    pages = sum(report.pages_processed for report in reports)
    low_conf = sum(1 for report in reports if report.low_confidence_pages)
    print(f"PDFs processed: {len(reports)}")
    print(f"TXT files created: {len(reports)}")
    print(f"Pages processed: {pages}")
    print(f"Documents with low confidence: {low_conf}")
    print(f"Errors: {errors}")
    print(f"Output directory: {output_dir}")
    print(f"Batch report: {csv_path}")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
