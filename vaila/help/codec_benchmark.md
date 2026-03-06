# codec_benchmark

## 📋 Module Information

- **Category:** Tools / Research
- **File:** `vaila/codec_benchmark.py`
- **Version:** 0.1.0
- **GUI Interface:** ❌ No (CLI only)
- **CLI Interface:** ✅ Yes

## 📖 Description

Systematically benchmarks **H.264**, **H.265 (HEVC)**, and **H.266 (VVC)** codecs on a dataset of videos.
It measures performance metrics for each codec to facilitate comparative analysis.

### Key Features

- **Automated comparison**: Runs all three codecs on every video in a directory.
- **Performance metrics**: Records elapsed time, input/output size, and compression ratio.
- **Results storage**: Saves results to `results.json` in the specified output folder.
- **Parallel execution**: Supports processing multiple videos simultaneously.

## 🚀 Usage

### CLI Mode

```bash
# Basic usage (all codecs, 1 worker)
python -m vaila.codec_benchmark --dir /path/to/videos

# High performance mode (multiple workers)
python -m vaila.codec_benchmark --dir /path/to/videos --workers 4 --output experiment_1
```

### CLI Options

| Option      | Default             | Description                        |
| ----------- | ------------------- | ---------------------------------- |
| `--dir`     | (required)          | Directory containing source videos |
| `--workers` | `1`                 | Number of parallel video tasks     |
| `--output`  | `benchmark_results` | Directory to save output and JSON  |

## 📊 Results Output

The script generates a `results.json` file structured as follows:

```json
[
  {
    "video": "sample.mp4",
    "input_size_bytes": 1048576,
    "results": [
      {
        "codec": "H.264",
        "success": true,
        "output_size_bytes": 524288,
        "compression_ratio": 2.0,
        "time_seconds": 15.5
      },
      ...
    ]
  }
]
```

---

📅 **Updated:** 05/03/2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
