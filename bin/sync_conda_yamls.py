#!/usr/bin/env python3
"""
Sync Conda YAMLs with pyproject.toml

This script reads dependencies from pyproject.toml and updates the pip section
in yaml_for_conda_env/vaila_linux.yaml (and potentially others).
It avoids adding packages to pip if they are already defined in the Conda dependencies list.
"""

import re
import sys
from pathlib import Path

# Try to import toml, otherwise use simple parsing
try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False
    print("Warning: 'toml' module not found. Using simple regex parsing.")

def load_toml_dependencies(toml_path):
    """Load dependencies from pyproject.toml"""
    if HAS_TOML:
        try:
            with open(toml_path, encoding="utf-8") as f:
                data = toml.load(f)
            return data.get("project", {}).get("dependencies", [])
        except Exception as e:
            print(f"Error reading TOML with toml module: {e}")
            return []
    else:
        # Simple regex fallback
        deps = []
        in_deps = False
        with open(toml_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("dependencies = ["):
                    in_deps = True
                    continue
                if in_deps:
                    if line.startswith("]"):
                        in_deps = False
                        break
                    # Extract string inside quotes
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        deps.append(match.group(1))
        return deps

def parse_yaml_dependencies(yaml_path):
    """Parse Conda YAML to get existing conda dependencies and the structure"""
    conda_deps = set()
    pip_deps_start_line = -1
    lines = []

    with open(yaml_path, encoding="utf-8") as f:
        lines = f.readlines()

    in_deps = False
    in_pip = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("dependencies:"):
            in_deps = True
            continue

        if in_deps:
            if stripped.startswith("- pip:"):
                in_pip = True
                pip_deps_start_line = i
                continue

            if stripped.startswith("-"):
                # It's a dependency
                dep = stripped.lstrip("- ").split("=")[0].split(">")[0].split("<")[0]
                if in_pip:
                    # We will overwrite pip section, so ignore what's there
                    pass
                else:
                    conda_deps.add(dep.lower())
            else:
                # End of dependencies section or empty line
                pass

    return lines, conda_deps, pip_deps_start_line

def update_yaml(yaml_path, toml_deps):
    """Update the YAML file with dependencies from TOML"""
    lines, conda_deps, pip_start = parse_yaml_dependencies(yaml_path)

    if pip_start == -1:
        print(f"Error: Could not find '- pip:' section in {yaml_path}")
        return

    # Filter TOML dependencies
    new_pip_deps = []
    for dep in toml_deps:
        # Handle environment markers and version specifiers for comparison
        pkg_name = dep.split(";")[0].strip().split("=")[0].split(">")[0].split("<")[0]

        # Exclude if in conda_deps
        if pkg_name.lower() in conda_deps:
            continue

        new_pip_deps.append(dep)

    # Reconstruct file content
    new_lines = lines[:pip_start+1]

    for dep in new_pip_deps:
        new_lines.append(f"      - {dep}\n")

    # Check if there was content after pip section (unlikely in simple env files but possible)
    # For now, we assume pip section is at the end or we just truncate the old pip list
    # Actually, we need to find where the pip section ENDS.
    # Usually it ends at EOF or next root key (no indentation).

    start_scanning = False
    rest_of_file = []
    for i in range(pip_start + 1, len(lines)):
        line = lines[i]
        if not start_scanning:
            if line.strip().startswith("-"): # pip dep
                continue
            else:
                start_scanning = True

        if start_scanning:
            rest_of_file.append(line)

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines + rest_of_file)

    print(f"Updated {yaml_path} with {len(new_pip_deps)} pip dependencies.")

def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    toml_path = project_root / "pyproject.toml"
    yaml_dir = project_root / "yaml_for_conda_env"

    if not toml_path.exists():
        print(f"Error: {toml_path} not found.")
        sys.exit(1)

    print(f"Reading dependencies from {toml_path}...")
    deps = load_toml_dependencies(toml_path)
    print(f"Found {len(deps)} dependencies.")

    if not yaml_dir.exists():
        print(f"Error: {yaml_dir} not found.")
        sys.exit(1)

    # Process all .yaml files in the directory
    for yaml_file in yaml_dir.glob("*.yaml"):
        print(f"Updating {yaml_file.name}...")
        update_yaml(yaml_file, deps)

if __name__ == "__main__":
    main()
