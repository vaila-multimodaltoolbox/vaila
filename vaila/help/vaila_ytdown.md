# YouTube Downloader

**Version:** 0.4.6
**Updated:** 2026-09-30

Download video as MP4 or audio as MP3 from one editable list. Open **Video and Image → YouTube Downloader** or run the module without CLI inputs.

## GUI

1. Paste URLs, one per line, or choose **Load TXT…**. Loading only fills the editable list; it never starts downloads. Blank lines and lines beginning with # are ignored. Review the count; **Clear** empties the list.
2. Choose a destination and **Video (MP4)** or **Audio (MP3)**.
3. For MP4, click **Consult qualities**. Select each video in the table and choose its resolution/FPS in the dropdown. Highest FPS, then highest resolution, is preselected: 1080p/60 wins over 4K/30. For each quality the codec OpenCV can decode is chosen: H.264, then VP9, and AV1 only when nothing else exists. AV1 opens in vailá tools but its frames cannot be read. If the downloaded MP4 still cannot be read by OpenCV, it is re-encoded to H.264 in place (same name, every frame, timestamps and audio kept), so it opens directly in getpixelvideo and the other vailá tools. Click **Download**, or press **Ctrl+Enter**, after reviewing. Download before consultation performs only the consultation; click again to download. Editing URLs clears previous choices. Failed consultations are shown per item and skipped with a failure report; use Consult qualities to retry. Inputs lock during work, preventing duplicate runs.
4. Follow the current item, transfer progress, processing/conversion and success/failure counts. Unknown transfer sizes use an indeterminate bar.
5. **Cancel** or Escape requests cooperative cancellation. A progress callback can interrupt transfer; extraction and ffmpeg processing may need to finish first. Pending cancellation is shown explicitly. New items are not started and completed files are retained.
6. **Open folder** opens the latest output directory. **Help** opens local help. Both remain accessible while working. **Show details** expands the log; **Diagnostic details** enables --debug.
7. Closing during work requests cancellation and waits safely. The window is resizable; Tab navigates fields and buttons.

Workers send queued events consumed by Tk after(). Downloads never pump the GUI with root.update() or update widgets from a worker.

## Quality and requirements

- **MP4:** selects highest FPS first, then resolution (shortest dimension, longest dimension as tie-breaker); unknown values rank last. All distinct dimensions/FPS combinations are offered. Equivalent codec/protocol variants use yt-dlp preference. The selected video gets the best available audio unless audio is already embedded. No artificial FPS or resolution increase. Expired/denied media URLs (403/410) trigger one metadata refresh requiring the same quality; quality is never silently reduced.
- **MP3:** preserves bestaudio/best and FFmpegExtractAudio at 192 kbps.
- **ffmpeg** must be on PATH for both outputs. There is no embedded-ffmpeg fallback. The final expected .mp4/.mp3 file is checked after post-processing; changing an extension never counts as conversion.
- Existing yt-dlp retry, JavaScript runtime detection (Deno/Node/QuickJS) and EJS fallback settings are retained. No new dependencies.
- Each input is an individual video URL; playlist expansion is not offered.

A completed transfer can still be merging or converting. Success is counted only after all post-processing and final-file verification.

Fragmented downloads show completed fragments/total. Transfer estimates never display 100% before completion; processing remains a separate phase. Metadata from consultation is reused for download.

## CLI

~~~bash
python -m vaila.vaila_ytdown --url "https://www.youtube.com/watch?v=VIDEO_ID" --list-formats
python -m vaila.vaila_ytdown --file urls.txt --video-format 1=137 --video-format 2=299 --no-gui
python -m vaila.vaila_ytdown --file urls.txt --keep-codec --no-gui   # keep YouTube codec (AV1), no re-encode
python -m vaila.vaila_ytdown --url "https://www.youtube.com/watch?v=VIDEO_ID" --output "/data/my videos" --no-gui
python -m vaila.vaila_ytdown --file "/data/my URLs.txt" --output "/data/my videos" --no-gui
python -m vaila.vaila_ytdown --file "/data/my URLs.txt" --output "/data/my audio" --audio-only --no-gui --debug
python -m vaila.vaila_ytdown --no-gui --audio-only --output "/data/my audio"
~~~

Short options remain: -u, -f, -o, -a. --url and --file are mutually exclusive; either selects CLI mode without a display. --no-gui without input prompts for one URL and respects --audio-only. Ctrl+C interrupts CLI execution.

`--list-formats` consults without downloading or creating output directories. `--video-format INDEX=FORMAT_ID` is repeatable; indices start at 1 in the input list. Use IDs returned by consultation, not the example IDs above. Unspecified items use FPS-first automatic selection. Invalid indices/duplicate selections are argument errors; unavailable formats fail that item. Video quality flags cannot be combined with MP3. GUI equivalent commands include per-video selections.

**Codec: vailá-ready (default) or `--keep-codec`.** Each quality label shows its codec, e.g. `1920x1080 · 60 FPS · H.264`. The codec choice never lowers resolution or FPS; it only picks among variants of the same quality. You choose it in the CLI or in the GUI box **vailá-ready MP4**.

- **vailá-ready (default, box ticked):**
  - takes the H.264 variant, else VP9, since OpenCV reads both;
  - re-encodes a quality that only exists in AV1 (often 4K/8K) to H.264 after download;
  - keeps the file name, every frame, the timestamps and the audio;
  - costs a larger file (about 10× AV1 at 1080p60), a small quality loss (NVENC CQ 18) and encode time. Above 4096 px (8K) NVENC cannot encode H.264 and the CPU fallback is slow.
- **`--keep-codec` (box unticked):**
  - keeps YouTube's own best variant (often AV1, smallest file), with no re-encode;
  - getpixelvideo then offers an H.264 copy (`<name>_h264.mp4`) when you open an AV1 file.

Changing the box clears the consultation; consult again. The equivalent CLI includes `--keep-codec` when the box is unticked.

Exit codes: 0 all succeeded; 1 failure including partial failures; 2 invalid arguments; 130 cancellation without other failures. Failed items do not prevent subsequent batch items from being attempted.

## Outputs and reproducibility

- Single video: vaila_ytdownload_TIMESTAMP/TITLE.mp4 and video_info.txt with title, channel, source URL, downloaded dimensions/FPS, duration and available formats.
- Single audio: vaila_ytaudio_TIMESTAMP/TITLE.mp3.
- Batch video: vaila_batch_TIMESTAMP/001/vaila_ytdownload_TIMESTAMP/001_TITLE.mp4, then 002, etc.
- Batch audio: vaila_audio_TIMESTAMP/001/vaila_ytaudio_TIMESTAMP/001_TITLE.mp3.
- A numerical suffix prevents runs started in the same second from reusing an output directory.
- download_log.txt records progress, successful paths, failures and summary. Batch logs live at the batch root; single-run logs live beside the final media.
- Multiple edited GUI URLs and CLI TXT runs save the exact list as urls.txt in the run directory. The equivalent command references that list; single-URL commands use --url. Destination, audio choice and diagnostics are included with native shell quoting.
- On launch and on Run, the terminal prints a highlighted `>> vaila/vaila_ytdown: Equivalent CLI` banner with the copy-paste command (`uv run --no-sync vaila/vaila_ytdown.py ...`) to repeat the run headlessly, and logs transparently each step, destination directory, and the exact output run folders.
- urls.txt is a replay input, not a sanitized log: treat it as private if it contains private links. Terminal and diagnostic logs redact credentials.
- Interrupted transfers or failed post-processing may leave partial files. Completed outputs are preserved. Inspect the summary before retrying.

## Diagnosis

Messages with **>> vaila/vaila_ytdown:** are always immediate, even when details are collapsed. Technical yt-dlp messages and tracebacks require --debug; secrets are redacted. Project /debug uses simulated downloads; external downloads are not automatic diagnostic steps.

GUI and CLI share YTDownloader.download_urls. The download_video, download_audio, callbacks and compatibility download_from_file entry points remain. Detailed counts are returned by download_urls and stored in last_result. Shared worker/feedback support is in task_feedback.py.
