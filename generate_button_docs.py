#!/usr/bin/env python3
"""
Script to generate button documentation for vailá GUI buttons.
This script analyzes vaila.py and creates documentation files for each button.
"""

import re
from pathlib import Path


def extract_button_info():
    """Extract all button information from vaila.py"""
    vaila_path = Path("vaila.py")

    with open(vaila_path, encoding="utf-8") as f:
        content = f.read()

    # Find all button methods with their positions
    button_pattern = r"# ([A-Z]\d*_r\d+_c\d+).*?\n\s+def (\w+)\(self\):"
    buttons = re.findall(button_pattern, content)

    # Extract docstrings for each method
    button_info = []
    for pos, method in buttons:
        # Find the method definition and its docstring
        method_pattern = rf'def {method}\(self\):.*?"""(.*?)"""'
        docstring_match = re.search(method_pattern, content, re.DOTALL)
        docstring = (
            docstring_match.group(1).strip() if docstring_match else "No description available."
        )

        # Get button text from GUI creation
        button_text_pattern = rf'# {pos}.*?\n.*?text="([^"]+)"'
        text_match = re.search(button_text_pattern, content, re.DOTALL)
        button_text = text_match.group(1) if text_match else method.replace("_", " ").title()

        button_info.append(
            {
                "position": pos,
                "method": method,
                "button_text": button_text,
                "docstring": docstring,
            }
        )

    return button_info


def create_button_doc(button_info):
    """Create markdown documentation for a button"""
    pos = button_info["position"]
    method = button_info["method"]
    text = button_info["button_text"]
    doc = button_info["docstring"]

    # Create filename from method name
    filename = method.replace("_", "-")

    md_content = f"""# {text} - Button {pos}

## Overview

**Button Position:** {pos}  
**Method Name:** `{method}`  
**Button Text:** {text}

## Description

{doc}

## Usage

1. Click the **{text}** button in the vailá GUI
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
"""

    return filename, md_content


def main():
    """Main function to generate all button documentation"""
    buttons_dir = Path("docs/vaila_buttons")
    buttons_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting button information from vaila.py...")
    button_info_list = extract_button_info()

    print(f"Found {len(button_info_list)} buttons")
    print("\nGenerating documentation files...")

    created_files = []
    for button_info in button_info_list:
        filename, md_content = create_button_doc(button_info)
        md_path = buttons_dir / f"{filename}.md"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        created_files.append((filename, button_info["position"], button_info["button_text"]))
        print(f"  Created: {md_path}")

    # Update README
    readme_path = buttons_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# vailá GUI Buttons Documentation\n\n")
        f.write(
            "This directory contains documentation for all buttons in the vailá GUI (`vaila.py`).\n\n"
        )
        f.write("## Structure\n\n")
        f.write("Each button in the vailá GUI has its own documentation file:\n")
        f.write("- **Markdown format** (`.md`) - For easy editing and version control\n")
        f.write("- **HTML format** (`.html`) - For web viewing (to be generated)\n\n")
        f.write("## Button List\n\n")
        f.write("| Position | Button Text | Method | Documentation |\n")
        f.write("|----------|-------------|--------|---------------|\n")

        for filename, pos, text in sorted(created_files, key=lambda x: x[1]):
            f.write(
                f"| {pos} | {text} | `{filename.replace('-', '_')}` | [{filename}.md]({filename}.md) |\n"
            )

        f.write("\n## Categories\n\n")
        f.write("### File Manager (Frame A)\n")
        f.write("- A_r1_c1 through A_r1_c9\n\n")
        f.write("### Multimodal Analysis (Frame B)\n")
        f.write("- B1_r1_c1 through B5_r5_c5\n\n")
        f.write("### Tools (Frame C)\n")
        f.write("- C_A, C_B, C_C sections\n\n")
        f.write("## Related Documentation\n\n")
        f.write(
            "- Script-specific help: `vaila/help/` - Contains help for individual Python scripts\n"
        )
        f.write("- Module documentation: `docs/` - Contains module-level documentation\n\n")
        f.write("---\n\n**Last Updated:** November 2025\n")

    print(f"\n✓ Generated {len(created_files)} button documentation files")
    print("✓ Updated README.md")
    print("\nNext steps:")
    print(f"  1. Review generated files in {buttons_dir}")
    print("  2. Add HTML versions if needed")
    print("  3. Update docs/index.md and docs/help.html with links")


if __name__ == "__main__":
    main()
