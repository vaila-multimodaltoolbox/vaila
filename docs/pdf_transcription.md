# PDF Transcription in Brainstorm

Updated: 2026-05-29

`vaila/transcribe_pdfs.py` adds PDF transcription to the Brainstorm tool. It is designed for typed, scanned, and handwritten exam PDFs, including the PAE Biomec1 folder layout.

## Open from the GUI

1. Run `uv run vaila.py`.
2. Click **Brainstorm** in Frame B.
3. Click **Transcribe PDFs**.
4. Select a PDF directory or a single PDF.
5. Confirm output and click **Transcribe PDFs**.

Default folders when present:

```text
~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/originais
~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/transcritas_originais
```

## CLI

```bash
uv run python -m vaila.transcribe_pdfs --gui
```

```bash
uv run python -m vaila.transcribe_pdfs \
  --input-dir ~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/originais \
  --output-dir ~/Preto/USP_RP/Alunos/PAE_USP/PAE_Biomec1/transcritas_originais \
  --mode auto
```

## Outputs

- `transcricao_<pdf_stem>.txt`
- `transcricao_<pdf_stem>.report.json`
- `batch_report.csv`

## Dependencies

- `pdftotext`, `pdftoppm`, and `pdfinfo` or `qpdf`
- Gemini CLI for handwritten/scanned vision mode
- Pillow; optional tqdm progress display

See also: [`vaila/help/transcribe_pdfs.md`](../vaila/help/transcribe_pdfs.md).
