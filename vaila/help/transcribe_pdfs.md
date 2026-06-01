# transcribe_pdfs

## Module Information

- **Category:** Tools / Brainstorm
- **File:** `vaila/transcribe_pdfs.py`
- **Version:** 0.3.47
- **Updated:** 29/05/2026
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** Yes — **Frame B -> Brainstorm -> Transcribe PDFs**
- **CLI Interface:** Yes
- **License:** AGPL-3.0

## Description

`transcribe_pdfs.py` converts PDF exams or handwritten/scanned documents into text files. It supports:

- native PDF text extraction with `pdftotext`;
- vision transcription through the Gemini CLI after rendering pages to images;
- one `.txt` transcription and one `.report.json` per PDF;
- a `batch_report.csv` summary with pages, low-confidence pages, blank pages, and errors.

The Brainstorm button opens a Tkinter launcher with defaults for the PAE Biomec1 workflow:

```text
~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/originais
~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/transcritas_originais
```

## GUI Workflow

1. Open `vaila.py`.
2. Click **Brainstorm** in Frame B.
3. Use the separate **PDF Transcription** section near the top and click **Transcribe PDFs**.
4. Choose a PDF directory or a single PDF.
5. Check the source feedback line for the selected path and PDF count.
6. Confirm the output folder and options.
7. Click **Transcribe PDFs** in the dialog.

## Modes

| Mode | Use |
|---|---|
| `auto` | Try native text first; if not useful, use vision transcription. |
| `vision` | Render every page and send the images to Gemini CLI. Best for handwritten exams. |
| `native` | Use embedded PDF text only. Does not require Gemini. |
| `ocr` | Reserved for future local OCR backend. |

## Requirements

System commands:

- `pdftotext`
- `pdftoppm`
- `pdfinfo` or `qpdf`
- `gemini` for `auto` / `vision` handwritten transcription

Python dependencies already used by vaila:

- `Pillow`
- `tkinter`
- `tqdm` when available (falls back to plain iteration if absent)

## Outputs

For each input PDF named `student.pdf`, the output directory receives:

```text
transcricao_student.txt
transcricao_student.report.json
```

For the whole batch:

```text
batch_report.csv
```

Optional page images are saved under the debug image directory when **Save page images** is enabled.

## CLI Examples

Open the GUI directly:

```bash
uv run python -m vaila.transcribe_pdfs --gui
```

Batch transcribe the default PAE folder:

```bash
uv run python -m vaila.transcribe_pdfs \
  --input-dir ~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/originais \
  --output-dir ~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/transcritas_originais \
  --mode auto
```

Native text only:

```bash
uv run python -m vaila.transcribe_pdfs \
  --input-file prova.pdf \
  --output-dir transcritas \
  --mode native
```

Vision mode with retained page images:

```bash
uv run python -m vaila.transcribe_pdfs \
  --input-dir originais \
  --output-dir transcritas_originais \
  --mode vision \
  --save-page-images
```

## Review Notes

Vision output is meant to accelerate review, not replace it. Check any pages listed in `low_confidence_pages`, and review `[ilegível]`, `[revisar]`, and `[palavra incerta: ...]` markers before using the text for grading or records.

---

Updated: 29/05/2026  
Part of vailá - Multimodal Toolbox
