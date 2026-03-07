# /new-module

Scaffold a new vailá analysis module from scratch.

## Usage
```
/new-module <module_name> <frame> <description>
```

## What this command does

1. Reads `.claude/skills/create-analysis-module.md`
2. Creates `vaila/<module_name>.py` with the full skeleton
3. Creates `tests/test_<module_name>.py` with basic test stubs
4. Shows you exactly where to add the button in `vaila.py`

## Example
```
/new-module sprint_analysis B "Analyze sprint kinematics from CSV data"
```

## Arguments
- `module_name` — snake_case name for the Python file
- `frame` — A, B, or C (which GUI frame the button goes in)
- `description` — one-line description of what the module does
