# Brainstorm - Button B_r5_c2

## Overview

**Button Position:** B_r5_c2  
**Method Name:** `brainstorm`  
**Button Text:** Brainstorm

## Description

Opens the Brainstorm workspace for audio transcription, prompt editing, creative text workflows, music-code generation, image prompt generation, text-to-audio, and PDF transcription.

## New PDF Transcription Action

Inside Brainstorm, use the separate **PDF Transcription** section near the top and click **Transcribe PDFs** to launch `vaila/transcribe_pdfs.py`. The dialog can process a directory of PDFs or a single PDF and writes `.txt`, `.report.json`, and `batch_report.csv` outputs.

Example workflow folders:

```text
/path/to/your/pdf_originals
/path/to/your/pdf_transcriptions
```

## Usage

1. Click **Brainstorm** in the vailá GUI.
2. Create or select a session when using audio/text creative workflows.
3. For PDFs, click **Transcribe PDFs** and choose PDF source/output folders.
4. Review low-confidence pages and markers in the generated reports.

## Related Scripts

- `vaila/brainstorm.py`
- `vaila/transcribe_pdfs.py`
- `vaila/help/brainstorm.md`
- `vaila/help/transcribe_pdfs.md`
- `docs/pdf_transcription.md`

## Requirements

- Audio workflows: `speech_recognition`, `sounddevice`, `soundfile`, optional FFmpeg/MIDI/TTS packages.
- PDF workflows: `pdftotext`, `pdftoppm`, `pdfinfo` or `qpdf`, and Gemini CLI for vision transcription.

---

**Last Updated:** 2026-06-01  
**Part of vailá - Multimodal Toolbox**  
**License:** AGPLv3.0
