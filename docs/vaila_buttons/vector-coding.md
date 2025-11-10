# Select YOLO tracker version to use: - Button B_r2_c1

## Overview

**Button Position:** B_r2_c1  
**Method Name:** `vector_coding`  
**Button Text:** Select YOLO tracker version to use:

## Description

Runs the Vector Coding module.

        This function runs the Vector Coding module, which can be used to calculate
        the coupling angle between two joints from a C3D file containing 3D marker
        positions.

        The user will be prompted to select the file, the axis, and the names of the
        two joints. The module will then calculate the coupling angle between the two
        joints and save the result in a CSV file.

## Usage

1. Click the **Select YOLO tracker version to use:** button in the vailá GUI
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

**Last Updated:** November 2025  
**Part of vailá - Multimodal Toolbox**  
**License:** AGPLv3.0
