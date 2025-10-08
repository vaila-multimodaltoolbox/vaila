#!/usr/bin/env python3
"""
Script to convert existing HTML documentation to Markdown for Read the Docs
"""

import os
import re
from pathlib import Path
from bs4 import BeautifulSoup


def convert_html_to_markdown(html_file, output_file):
    """Convert HTML file to Markdown format"""

    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Extract main content
    content = soup.find("div", class_="container")
    if not content:
        content = soup.find("body")

    # Convert to markdown
    markdown_content = html_to_markdown(content)

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(markdown_content)


def html_to_markdown(element):
    """Convert HTML element to Markdown"""

    markdown = ""

    for child in element.children:
        if child.name == "h1":
            markdown += f"# {child.get_text()}\n\n"
        elif child.name == "h2":
            markdown += f"## {child.get_text()}\n\n"
        elif child.name == "h3":
            markdown += f"### {child.get_text()}\n\n"
        elif child.name == "p":
            markdown += f"{child.get_text()}\n\n"
        elif child.name == "ul":
            for li in child.find_all("li"):
                markdown += f"- {li.get_text()}\n"
            markdown += "\n"
        elif child.name == "ol":
            for i, li in enumerate(child.find_all("li"), 1):
                markdown += f"{i}. {li.get_text()}\n"
            markdown += "\n"
        elif child.name == "code":
            markdown += f"`{child.get_text()}`"
        elif child.name == "pre":
            markdown += f"```\n{child.get_text()}\n```\n\n"
        elif child.name == "table":
            markdown += convert_table_to_markdown(child)

    return markdown


def convert_table_to_markdown(table):
    """Convert HTML table to Markdown table"""

    markdown = ""
    rows = table.find_all("tr")

    if not rows:
        return markdown

    # Header
    header_cells = rows[0].find_all(["th", "td"])
    markdown += (
        "| " + " | ".join(cell.get_text().strip() for cell in header_cells) + " |\n"
    )
    markdown += "|" + "|".join("---" for _ in header_cells) + "|\n"

    # Data rows
    for row in rows[1:]:
        cells = row.find_all("td")
        markdown += (
            "| " + " | ".join(cell.get_text().strip() for cell in cells) + " |\n"
        )

    return markdown + "\n"


def main():
    """Main conversion script"""

    # Define conversion mappings
    conversions = [
        (
            "vaila/help/markerless_2D_analysis_br_help.html",
            "docs/modules/markerless-analysis/markerless-2d-video.md",
        ),
        (
            "vaila/help/getpixelvideo_help.html",
            "docs/modules/tools/get-pixel-coordinates.md",
        ),
        (
            "vaila/help/dlt3d_and_rec3d.html",
            "docs/modules/tools/dlt3d-reconstruction.md",
        ),
        ("vaila/help/view3d_help.html", "docs/modules/visualization/view3d.md"),
    ]

    for html_file, md_file in conversions:
        if os.path.exists(html_file):
            print(f"Converting {html_file} to {md_file}")
            convert_html_to_markdown(html_file, md_file)
        else:
            print(f"Warning: {html_file} not found")


if __name__ == "__main__":
    main()
