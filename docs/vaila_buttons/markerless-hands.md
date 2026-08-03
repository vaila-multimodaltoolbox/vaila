# Markerless Hands — via Markerless 2D Chooser

## Overview

> **v0.3.98:** this used to be its own button (`B4_r4_c3`, text
> "Markerless Hands"). It is now under the **"Other 2D tools"** section of
> the **Markerless 2D** coringa chooser (`B1_r1_c4`, method
> `markerless_2d_analysis`) — the underlying handler and script
> (`vaila/mphands.py`) are unchanged.

**Method Name:** `markerless_hands`  
**Button Text (in chooser):** Markerless Hands

## Description

Invokes the `vaila/mphands.py` module — MediaPipe hand-landmark tracking.

## Usage

1. Click **Markerless 2D** in the vailá GUI, then **Markerless Hands** in the "Other 2D tools" section of the chooser
2. Follow the prompts in the dialog windows
3. Select input files/directories as requested
4. Configure parameters if needed
5. Review the output files

## Related Scripts

This button launches one or more Python scripts from the `vaila/` directory. For detailed script documentation, see:
- `vaila/help/` - Script-specific help files

## Integration

This button integrates with other vailá modules:
- Check related buttons in the same frame/section
- Output files can be used as input for other modules

## Troubleshooting

### Common Issues

- **Module not found**: Ensure all dependencies are installed
- **File not found**: Check that input files exist in the specified directory
- **Permission errors**: Ensure write permissions for output directory

### Getting Help

- Check the script-specific help in `vaila/help/`
- Review the main documentation in `docs/`
- Open an issue on GitHub if problems persist

---

**Last Updated:** 02 August 2026  
**Part of vailá - Multimodal Toolbox**  
**License:** AGPLv3.0
