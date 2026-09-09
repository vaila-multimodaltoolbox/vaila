#!/usr/bin/env python3
"""Package a skill directory into a .skill archive file."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def package_skill(skill_dir: Path, output: Path | None = None) -> Path:
    skill_dir = skill_dir.resolve()
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill directory not found: {skill_dir}")
    sk = skill_dir / "SKILL.md"
    if not sk.exists():
        raise ValueError(f"SKILL.md not found in {skill_dir}")

    if output is None:
        output = skill_dir.parent / f"{skill_dir.name}.skill"

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in skill_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(skill_dir.parent)
                zf.write(file, arcname)

    print(f"Packaged {skill_dir.name} -> {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a skill as .skill file")
    parser.add_argument("skill_dir", type=Path, help="Path to skill directory")
    parser.add_argument("--output", "-o", type=Path, help="Output .skill path")
    args = parser.parse_args()
    result = package_skill(args.skill_dir, args.output)
    print(result)


if __name__ == "__main__":
    main()
