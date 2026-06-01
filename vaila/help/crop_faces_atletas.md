# Crop Faces Atletas - Crop Face

- **Script:** `crop_faces_atletas.py`
- **Category:** Tools / Video and Image
- **Creator:** Abel Gonçalves Chinaglia
- **Version:** 0.3.46
- **Updated:** 01 June 2026
- **Python:** 3.12

## Overview

`crop_faces_atletas.py` batch-processes athlete photos and creates square face crops suitable for 5 x 5 cm outputs at 300 DPI. It uses MediaPipe Face Detector to find the largest face in each image, applies a square crop with configurable margins, resizes to 591 x 591 pixels, and saves JPEG files with 300 DPI metadata.

The file name stem is used as the athlete name. By default, the name is drawn in a small white rounded label at the top of the exported image.

## GUI Workflow

In the main vailá GUI, open:

**Tools -> Video and Image -> Crop Face**

The module follows the same directory-selection pattern used by other vailá tools:

1. Select the input directory containing athlete photos.
2. Select the output directory for the cropped face images.
3. If the default face detector model is absent, wait for its automatic download.
4. If automatic download fails, select a compatible MediaPipe `.task` or `.tflite` model file.
5. Wait for the completion dialog with processed and skipped image counts.

## Inputs

Supported image extensions:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

The module processes images directly inside the selected input directory. It does not recurse into subdirectories.

## Outputs

Each successful crop is saved in the selected output directory as:

```text
<athlete_name>_resultado.jpeg
```

Output properties:

- 591 x 591 pixels by default
- JPEG quality 95
- 300 DPI metadata
- Optional athlete-name label from the input file name

## Face Detector Model

The module uses the official MediaPipe BlazeFace short-range detector. On first use, it downloads the versioned model into the local Git-ignored cache:

```text
vaila/models/crop_face/face_detector.task
```

Legacy locations remain auto-detected for compatibility:

```text
vaila/crop_face/models/face_detector.task
vaila/models/face_detector.task
models/face_detector.task
```

To provision the model before opening the GUI:

```bash
uv run python vaila/crop_faces_atletas.py --download-model
```

The downloaded bytes come from Google's official `.tflite` model and are stored locally as `face_detector.task`. If automatic download fails, the GUI prompts for a compatible `.task` or `.tflite` file. In CLI mode, you can pass a custom model explicitly with `--model`.

## CLI Usage

Launch the directory-selection GUI directly:

```bash
uv run python vaila/crop_faces_atletas.py
```

Run with explicit input and output directories:

```bash
uv run python vaila/crop_faces_atletas.py \
  --input /path/to/photos \
  --output /path/to/cropped_faces \
  --model /path/to/face_detector.task
```

Disable the athlete-name label:

```bash
uv run python vaila/crop_faces_atletas.py -i photos -o output --no-name
```

Change output size in pixels:

```bash
uv run python vaila/crop_faces_atletas.py -i photos -o output --size 768
```

## Processing Details

For each image, the module:

1. Reads the image with OpenCV.
2. Converts it to RGB for MediaPipe.
3. Detects faces with MediaPipe Tasks Vision Face Detector.
4. Selects the largest detected face.
5. Builds a square crop using lateral, superior, inferior, and vertical-offset parameters.
6. Clips the crop to the image boundaries.
7. Resizes to the requested square output size.
8. Optionally draws the athlete name.
9. Saves the result as JPEG with 300 DPI metadata.

## Tunable Constants

The crop behavior is controlled near the top of `crop_faces_atletas.py`:

```python
TAMANHO_SAIDA_PX = 591
INSERIR_NOME = True
MARGEM_LATERAL = 0.55
MARGEM_SUPERIOR = 0.55
MARGEM_INFERIOR = 0.45
DESLOCAMENTO_VERTICAL = -0.04
```

Lower margins zoom closer to the face. A negative vertical offset moves the crop upward; a positive value moves it downward.

## Troubleshooting

| Problem | Cause | Action |
|---|---|---|
| `Modelo não encontrado` | Automatic model download failed | Check network access, run `--download-model`, select a model in the GUI, or pass `--model` in CLI |
| No images processed | Unsupported extension or empty folder | Use `.jpg`, `.jpeg`, `.png`, or `.webp` files |
| `Nenhuma face detectada` | Face too small, blurred, occluded, or side-facing | Use clearer frontal photos or adjust input images |
| Name label too large/small | File name length or output size | Rename files or adjust font scaling in `adicionar_nome()` |

## Dependencies

- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- Pillow (`PIL`)
- Tkinter (standard library, GUI dialogs)

## Version History

- **0.3.46 (01 June 2026):** Added automatic download and SHA-256 verification for the official MediaPipe BlazeFace short-range model, plus `--download-model` provisioning. Added vailá GUI directory-selection workflow, CLI arguments, manual model-selection fallback, complete script header, and help documentation. Integrated into **Tools -> Video and Image -> Crop Face**.
