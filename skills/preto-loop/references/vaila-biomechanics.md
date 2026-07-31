# Vailá Biomechanics and Data-Science Reference

Use this reference when the loop changes vailá code, processes biomechanical data, or evaluates a markerless/video pipeline. It is a domain checklist, not a substitute for the repository's current `AGENTS.md`, module help, tests, or user-provided protocol.

## Scientific contract

Freeze these before changing an analysis:

- Population, subject/session/video identity, inclusion/exclusion criteria, and train/validation/test split.
- Input schema, header rows, units, coordinate frame, axis direction, origin, sign convention, and synchronization.
- Sampling rate or video frame rate, time base, interpolation policy, smoothing/filter family and cutoff/order, and treatment of missing/occluded samples.
- Target quantities, equations, event definitions, uncertainty/sensitivity expectations, and reference or synthetic fixtures.
- Output schema, column meanings, units, provenance, and whether values are estimates, detections, or measurements.

Never accept a plot or a plausible trajectory as sole proof. Pair visual review with assertions, dimensional checks, known-value fixtures, held-out data, or an external measurement when available.

## Vailá software contract

For a Python change in the vailá repository, plan for the relevant subset of:

```text
uv run pytest <focused tests> -v
uv run ruff check <changed files>
uv run ruff format --check <changed files>
uv run ty check <changed files or package>
```

Keep Python 3.12 compatibility, lazy imports, a single Tk root, timestamped outputs, and standalone CLI behavior. If a GUI button launches a module, ensure the equivalent command is printed with the `>>` prefix and that the module does not create a hidden or competing event loop. If a module has both GUI and CLI paths, test both at the appropriate scope.

When editing any `.py`, update the module header date/version and the repository-mandated README, help index, and module-help metadata. Treat this metadata as part of the acceptance check, not post-hoc cleanup.

## Data and AI checks

For CSV/marker pipelines, test header detection, column order, frame numbering, NaN/Inf behavior, units, and round-trip read/write. For DLT/reconstruction, test calibration geometry, degeneracy, reprojection error, and fixed versus moving-camera assumptions. For YOLO/SAM3/Sapiens2/tracking, test coordinate conversion, keypoint order, confidence/occlusion rules, track-ID continuity, bbox/contour/mask exports, and downstream schema compatibility.

For model work, report the data split, leakage controls, class/subject balance, sample counts, baseline, metric definition, and reproducible seed/configuration. Do not let a loop edit the evaluation set or the metric implementation it uses as its only verifier.

## Recommended loop shape

1. Snapshot the current code, data contract, and focused check output.
2. Select one worst failure or highest-value missing test.
3. Make one narrowly scoped code/config/data-contract change.
4. Run focused deterministic checks and capture raw output.
5. Run scientific/reference checks and inspect artifacts where needed.
6. Run broader vailá regression checks proportional to risk.
7. Retain only non-regressive changes; persist provenance, assumptions, and evidence.

## Common failure states

Use `blocked` for missing model weights, unavailable CUDA, absent protected data, or an unresolved scientific protocol. Use `no-progress` for repeated iterations that do not improve the declared metric or evidence. Use `exhausted` for the declared budget. Do not report any of these as success.
