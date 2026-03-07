# /check

Run all quality checks for vailá before committing.

## What this runs
1. `uv run pytest` — full test suite
2. `uv run python -c "import vaila"` — verify package imports
3. Check for any hardcoded paths (`/home/`, `C:\\Users`)
4. Check all `pyproject_*.toml` files have consistent dependencies

## Usage
```
/check
```

Run this before every commit or pull request.
