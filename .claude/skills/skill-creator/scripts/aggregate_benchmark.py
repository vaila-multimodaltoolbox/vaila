#!/usr/bin/env python3
"""Aggregate grading and timing results into benchmark.json and benchmark.md."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def stddev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def pass_rate(grading: dict) -> float:
    expectations = grading.get("expectations", [])
    if not expectations:
        return 0.0
    return sum(1 for e in expectations if e.get("passed", False)) / len(expectations)


def compute_stats(
    run_dirs: list[Path],
) -> dict:
    pass_rates = []
    times = []
    tokens = []
    eval_names = []

    for run_dir in run_dirs:
        grading_path = run_dir / "grading.json"
        timing_path = run_dir / "timing.json"
        meta_path = run_dir / "eval_metadata.json"

        if grading_path.exists():
            grading = load_json(grading_path)
            pass_rates.append(pass_rate(grading))

        if timing_path.exists():
            timing = load_json(timing_path)
            times.append(timing.get("total_duration_seconds", 0.0))
            tokens.append(timing.get("total_tokens", 0))

        if meta_path.exists():
            meta = load_json(meta_path)
            eval_names.append(meta.get("eval_name", run_dir.name))

    _std_t = stddev(times)
    _std_tok = stddev(tokens)
    return {
        "pass_rate": round(mean(pass_rates), 3) if pass_rates else 0.0,
        "mean_time_seconds": round(mean(times), 1) if times else 0.0,
        "mean_tokens": round(mean(tokens)) if tokens else 0,
        "stddev_time": round(_std_t, 1) if _std_t is not None else None,
        "stddev_tokens": round(_std_tok) if _std_tok is not None else None,
        "n_evals": len(run_dirs),
    }


def compute_delta(a: dict, b: dict) -> dict:
    return {
        "pass_rate": round(a["pass_rate"] - b["pass_rate"], 3),
        "mean_time_seconds": round(a["mean_time_seconds"] - b["mean_time_seconds"], 1),
        "mean_tokens": round(a["mean_tokens"] - b["mean_tokens"]),
        "stddev_time": None,
        "stddev_tokens": None,
    }


def main(workspace: Path, skill_name: str) -> None:
    workspace = Path(workspace)

    it_dirs = sorted(workspace.glob("iteration-*"), key=lambda p: p.name)
    if not it_dirs:
        print("No iteration directories found.")
        sys.exit(1)

    iteration_dir = it_dirs[-1]
    iteration_num = int(iteration_dir.name.split("-")[1])

    with_skill_dirs = sorted(iteration_dir.glob("*/with_skill"))
    without_skill_dirs = sorted(iteration_dir.glob("*/without_skill"))
    old_skill_dirs = sorted(iteration_dir.glob("*/old_skill"))

    baseline_dirs = without_skill_dirs if without_skill_dirs else old_skill_dirs

    if not with_skill_dirs:
        print("No with_skill directories found.")
        sys.exit(1)
    if not baseline_dirs:
        print("No baseline directories found (without_skill or old_skill).")
        sys.exit(1)

    with_skill_stats = compute_stats(with_skill_dirs)
    baseline_stats = compute_stats(baseline_dirs)
    delta = compute_delta(with_skill_stats, baseline_stats)

    eval_stats = []
    for ws_dir, bs_dir in zip(with_skill_dirs, baseline_dirs):
        meta_path = ws_dir / "eval_metadata.json"
        grading_w = load_json(ws_dir / "grading.json") if (ws_dir / "grading.json").exists() else {}
        grading_b = load_json(bs_dir / "grading.json") if (bs_dir / "grading.json").exists() else {}
        timing_w = load_json(ws_dir / "timing.json") if (ws_dir / "timing.json").exists() else {}
        timing_b = load_json(bs_dir / "timing.json") if (bs_dir / "timing.json").exists() else {}

        eval_name = ws_dir.parent.name if ws_dir.parent.name else ws_dir.name
        if meta_path.exists():
            eval_name = load_json(meta_path).get("eval_name", eval_name)

        pr_w = pass_rate(grading_w)
        pr_b = pass_rate(grading_b)
        passed = all(e.get("passed", False) for e in grading_w.get("expectations", []))

        eval_stats.append(
            {
                "eval_name": eval_name,
                "pass_rate": round(pr_w, 3),
                "time_seconds": round(timing_w.get("total_duration_seconds", 0.0), 1),
                "tokens": round(timing_w.get("total_tokens", 0)),
                "passed": passed,
                "baseline_pass_rate": round(pr_b, 3),
            }
        )

    benchmark = {
        "skill_name": skill_name,
        "iteration": iteration_num,
        "configurations": [
            {**with_skill_stats, "name": "with_skill", "evals": eval_stats},
            {**baseline_stats, "name": baseline_dirs[0].name, "evals": []},
            {**delta, "name": "delta", "evals": []},
        ],
        "analyst_notes": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    benchmark_path = iteration_dir / "benchmark.json"
    with open(benchmark_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    md = f"""# Benchmark Results — {skill_name} (Iteration {iteration_num})

## Summary

| Configuration | Pass Rate | Mean Time | Mean Tokens |
|---|---|---|---|
| with_skill | {with_skill_stats["pass_rate"]:.1%} | {with_skill_stats["mean_time_seconds"]:.1f}s | {with_skill_stats["mean_tokens"]:,} |
| {baseline_dirs[0].name} | {baseline_stats["pass_rate"]:.1%} | {baseline_stats["mean_time_seconds"]:.1f}s | {baseline_stats["mean_tokens"]:,} |
| **delta** | **+{delta["pass_rate"]:+.1%}** | **{delta["mean_time_seconds"]:+.1f}s** | **{delta["mean_tokens"]:+,}** |

## Per-Eval Breakdown

| Eval | Pass Rate (skill) | Pass Rate (baseline) | Time | Tokens |
|---|---|---|---|---|
"""
    for es in eval_stats:
        md += f"| {es['eval_name']} | {es['pass_rate']:.0%} | {es['baseline_pass_rate']:.0%} | {es['time_seconds']:.1f}s | {es['tokens']:,} |\n"

    md_path = iteration_dir / "benchmark.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"Written: {benchmark_path}")
    print(f"Written: {md_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <workspace> [--skill-name NAME]")
        sys.exit(1)

    workspace = Path(sys.argv[1])
    skill_name = "my-skill"
    for i, arg in enumerate(sys.argv):
        if arg == "--skill-name" and i + 1 < len(sys.argv):
            skill_name = sys.argv[i + 1]

    main(workspace, skill_name)
