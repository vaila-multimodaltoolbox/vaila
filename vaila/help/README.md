# vailá Script Help Documentation

This directory contains help documentation for all Python scripts in the `vaila/` directory.

## Structure

Each script in `vaila/` has its own help documentation:
- **Markdown format** (`.md`) - For easy editing and version control
- **HTML format** (`.html`) - For web viewing

## Directory Organization

Help pages live **flat** under `vaila/help/` (one `.md` + `.html` pair per topic). Categories are declared inside each page’s Module Information and used by the index generator.

## Current Help Files

The authoritative catalog is the generated index (all topics, with links):

- [`index.html`](index.html) — searchable HTML
- [`index.md`](index.md) — Markdown list for GitHub / editors

Do not maintain a hand-curated subset list here; run the generator after adding pages.

## Script Help Format

Each script help should include:

1. **Module Information**
   - Category
   - File path
   - Lines of code
   - Version
   - Author
   - GUI Interface (Yes/No)

2. **Description**
   - What the script does
   - Key features
   - Use cases

3. **Main Functions**
   - List of main functions
   - Function descriptions

4. **Configuration Parameters**
   - All configurable parameters
   - Default values
   - Parameter ranges

5. **Output Files**
   - File formats
   - File naming conventions
   - Output structure

6. **Usage**
   - GUI mode instructions
   - Programmatic usage examples
   - Command-line usage (if applicable)

7. **Requirements**
   - System requirements
   - Python dependencies
   - Hardware requirements

8. **Performance Characteristics**
   - Processing speed
   - Memory usage
   - Best use cases

9. **Troubleshooting**
   - Common issues
   - Solutions
   - Performance tips

10. **Integration**
    - Compatible modules
    - Data flow
    - Integration examples

## Adding New Script Help

When adding help for a new script:

1. Create both `.md` and `.html` files named after the script (flat under `vaila/help/`)
2. Include **Category** in the Module Information section (Analysis, ML, Processing, Tools, Utils, Visualization, or Guides)
3. Write a clear Description / Overview paragraph (used as the one-liner in the index)
4. Regenerate the catalog indexes:

```bash
uv run python bin/generate_help_index.py
```

That writes `vaila/help/index.md` and `vaila/help/index.html` with the project intro plus a full linked list of every help topic. Do **not** hand-edit those indexes except via the generator.

## Mandatory: keep help synced with code (version/date)

Whenever any `*.py` script is updated, also update the matching help files:

- module help: `vaila/help/<module>.md` + `vaila/help/<module>.html` (Version + Updated)
- regenerate main index: `uv run python bin/generate_help_index.py`
- root `README.md`: `Last updated: YYYY-MM-DD`

Global version source: `vaila.py` header/banner. Reference checklist: `AGENTS.md` (“Mandatory: Update metadata on any script change”).

## Related Documentation

- Project docs hub: `docs/index.md` · `docs/help.html`
- Button documentation: `docs/vaila_buttons/`

---

**Last Updated:** August 2026
