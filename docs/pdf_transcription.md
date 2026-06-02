# PDF Transcription

Updated: 2026-06-01

`vaila/transcribe_pdfs.py` adds PDF transcription to the Brainstorm tool. It is designed for typed, scanned, and handwritten exam PDFs.

## Open from the GUI

1. Run `uv run vaila.py`.
2. Click **Brainstorm** in Frame B.
3. Use the separate **PDF Transcription** section near the top and click **Transcribe PDFs**.
4. Select a PDF directory or a single PDF.
5. Check the source feedback line for the selected path and PDF count.
6. Confirm output and click **Transcribe PDFs**.

The GUI remembers the last used directories. On first launch, point it to
the folder containing your source PDFs and choose an output folder.

## CLI

```bash
uv run python -m vaila.transcribe_pdfs --gui
```

```bash
uv run python -m vaila.transcribe_pdfs \
  --input-dir /path/to/your/pdf_originals \
  --output-dir /path/to/your/pdf_transcriptions \
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
