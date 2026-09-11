# YouTube Downloader

**Version:** 0.3.137
**Updated:** 2026-09-11

Download video as MP4 or audio as MP3 from one editable list. Open **Video and Image → YouTube Downloader** or run the module without CLI inputs.

## GUI

1. Paste URLs, one per line, or choose **Load TXT…**. Loading only fills the editable list; it never starts downloads. Blank lines and lines beginning with # are ignored. Review the count; **Clear** empties the list.
2. Choose a destination and **Video (MP4)** or **Audio (MP3)**.
3. Click **Download**, or press **Ctrl+Enter**. Inputs lock until the run finishes, preventing duplicate runs.
4. Follow the current item, transfer progress, processing/conversion and success/failure counts. Unknown transfer sizes use an indeterminate bar.
5. **Cancel** or Escape requests cooperative cancellation. A progress callback can interrupt transfer; extraction and ffmpeg processing may need to finish first. Pending cancellation is shown explicitly. New items are not started and completed files are retained.
6. **Open folder** opens the latest output directory. **Help** opens local help. Both remain accessible while working. **Show details** expands the log; **Diagnostic details** enables --debug.
7. Closing during work requests cancellation and waits safely. The window is resizable; Tab navigates fields and buttons.

Workers send queued events consumed by Tk after(). Downloads never pump the GUI with root.update() or update widgets from a worker.

## Quality and requirements

- **MP4:** preserves bestvideo+bestaudio/best, merges video/audio and converts to MP4. yt-dlp chooses the best available quality. This does not promise the highest FPS independently of other quality attributes.
- **MP3:** preserves bestaudio/best and FFmpegExtractAudio at 192 kbps.
- **ffmpeg** must be on PATH for both outputs. There is no embedded-ffmpeg fallback. The final expected .mp4/.mp3 file is checked after post-processing; changing an extension never counts as conversion.
- Existing yt-dlp retry, JavaScript runtime detection (Deno/Node/QuickJS) and EJS fallback settings are retained. No new dependencies.
- Each input is an individual video URL; playlist expansion is not offered.

A completed transfer can still be merging or converting. Success is counted only after all post-processing and final-file verification.

## CLI

~~~bash
python -m vaila.vaila_ytdown --url "https://www.youtube.com/watch?v=VIDEO_ID" --output "/data/my videos" --no-gui
python -m vaila.vaila_ytdown --file "/data/my URLs.txt" --output "/data/my videos" --no-gui
python -m vaila.vaila_ytdown --file "/data/my URLs.txt" --output "/data/my audio" --audio-only --no-gui --debug
python -m vaila.vaila_ytdown --no-gui --audio-only --output "/data/my audio"
~~~

Short options remain: -u, -f, -o, -a. --url and --file are mutually exclusive; either selects CLI mode without a display. --no-gui without input prompts for one URL and respects --audio-only. Ctrl+C interrupts CLI execution.

Exit codes: 0 all succeeded; 1 failure including partial failures; 2 invalid arguments; 130 cancellation without other failures. Failed items do not prevent subsequent batch items from being attempted.

## Outputs and reproducibility

- Single video: vaila_ytdownload_TIMESTAMP/TITLE.mp4 and video_info.txt with title, channel, source URL, downloaded dimensions/FPS, duration and available formats.
- Single audio: vaila_ytaudio_TIMESTAMP/TITLE.mp3.
- Batch video: vaila_batch_TIMESTAMP/001/vaila_ytdownload_TIMESTAMP/001_TITLE.mp4, then 002, etc.
- Batch audio: vaila_audio_TIMESTAMP/001/vaila_ytaudio_TIMESTAMP/001_TITLE.mp3.
- A numerical suffix prevents runs started in the same second from reusing an output directory.
- download_log.txt records progress, successful paths, failures and summary. Batch logs live at the batch root; single-run logs live beside the final media.
- Multiple edited GUI URLs and CLI TXT runs save the exact list as urls.txt in the run directory. The equivalent command references that list; single-URL commands use --url. Destination, audio choice and diagnostics are included with native shell quoting.
- urls.txt is a replay input, not a sanitized log: treat it as private if it contains private links. Terminal and diagnostic logs redact credentials.
- Interrupted transfers or failed post-processing may leave partial files. Completed outputs are preserved. Inspect the summary before retrying.

## Diagnosis

Messages with **>> vaila/vaila_ytdown:** are always immediate, even when details are collapsed. Technical yt-dlp messages and tracebacks require --debug; secrets are redacted. Project /debug uses simulated downloads; external downloads are not automatic diagnostic steps.

GUI and CLI share YTDownloader.download_urls. The download_video, download_audio, callbacks and compatibility download_from_file entry points remain. Detailed counts are returned by download_urls and stored in last_result. Shared worker/feedback support is in task_feedback.py.
