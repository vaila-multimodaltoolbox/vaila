# brainstorm

## Module Information

- **Category:** Tools
- **File:** `vaila/brainstorm.py`
- **Version:** 0.3.47
- **Updated:** 01/06/2026
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** Yes — **Frame B -> Brainstorm**
- **License:** AGPL-3.0

## Description

`brainstorm.py` is a Tkinter workspace for creative text workflows inside vailá. It records or loads audio, transcribes speech, edits text prompts, generates music code / ideas / image prompts, and now launches PDF transcription for scanned or handwritten exams.

## Main Workflows

- **Record Audio** — capture voice audio and save WAV/MP3 in the current session.
- **Transcribe** — convert a loaded or recorded audio file to text.
- **Batch Transcribe** — process multiple audio files.
- **Transcribe PDFs** — open `transcribe_pdfs.py` for PDF exam transcription.
- **Load Transcription / Save Text** — edit and reuse text prompts.
- **Generate Music Code / Batch Music Generation** — create Python/MIDI code from text.
- **Generate Image / Creative Ideas / Text to Audio** — produce creative downstream outputs.

## PDF Transcription Button

Use the separate **PDF Transcription** section near the top of Brainstorm and click **Transcribe PDFs** to open the PDF transcription dialog. Default folders are configurable via the `VAILA_PAE_ROOT` environment variable:

```text
/path/to/your/pdf_originals
/path/to/your/pdf_transcriptions
```

Detailed help: [`transcribe_pdfs.md`](transcribe_pdfs.md) / [`transcribe_pdfs.html`](transcribe_pdfs.html).

## Session Structure

```text
brainstorm_YYYYMMDD_HHMMSS/
  audio/      WAV, MP3, generated TTS audio
  text/       transcriptions, prompts, ideas
  scripts/    generated Python/MIDI code
  images/     image prompts and descriptions
```

## Requirements

Core Brainstorm audio features:

- `tkinter`
- `speech_recognition`
- `sounddevice`
- `soundfile`
- optional `openai`, `midiutil`, `music21`, `pyttsx3`, `gtts`, FFmpeg

PDF transcription:

- Poppler command-line tools: `pdftotext`, `pdftoppm`, `pdfinfo` or `qpdf`
- Gemini CLI for handwritten/scanned vision transcription

---

Updated: 01/06/2026  
Part of vailá - Multimodal Toolbox
