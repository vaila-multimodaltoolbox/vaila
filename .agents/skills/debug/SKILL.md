---
name: debug
description: Investigate a vailá failure from terminal evidence and a safe reproduction; use for /debug or requests to diagnose broken GUI, CLI, file operations or downloads.
---

# Debug vailá

Start from the symptom, expected result, command or GUI action, and affected module. Read applicable AGENTS.md and task-relevant skills. Preserve unrelated working changes.

1. Collect only relevant context: OS, Python/environment, vailá version, sanitized command or button/fields, input shape, output location, error and repeatability. Ask only for missing information that blocks reproduction. Do not dump environment variables, authentication files, cookies, private URLs or tokens.
2. Trace the entry point to the responsible operation. Separate parameter collection, worker execution, callbacks and outputs. State the failing invariant; distinguish observations from hypotheses.
3. Reproduce with the smallest safe fixture. File Manager: temporary directories and --dry-run first. Downloader: simulated yt-dlp callbacks. Use --debug and optional File Manager --log-file outside selected targets when needed.
4. Never automatically remove real user data, start a transfer or download media as diagnosis. Those actions need explicit task-specific authorization; prepare and show the exact command and effects first. Normal temporary-file tests need no additional approval.
5. Narrow the cause with evidence. Record the reproducer and outcome before editing. If a fix is authorized, change the responsible layer and add a meaningful regression proof. Otherwise provide the concrete proposed fix with evidence.
6. Run focused validation: python -m pytest tests/test_vaila_ytdown.py tests/test_filemanager_operations.py tests/test_filetools_downloader.py. GUI tests need a display and simulated downloads. Set VAILA_GUI_ARTIFACTS to a temporary directory when screenshots are useful.
7. Report cause, affected behavior, changed files, exact checks/outcomes and anything unverified. Stop when the requested failure is resolved or identify concrete missing evidence.

File Manager invariants: fixed confirmed targets; no silent overwrite or output traversal; permanent removal explicitly confirmed; no worker widget calls; terminal launch distinct from transfer completion.
Downloader invariants: TXT loads without starting; one active run; shared audio/video batch behavior; completion after post-processing; cooperative cancellation; nonzero status for partial failures.
