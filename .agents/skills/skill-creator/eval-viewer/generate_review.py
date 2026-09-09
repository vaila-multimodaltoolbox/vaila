#!/usr/bin/env python3
"""Generate an interactive HTML reviewer for skill eval outputs.

Usage:
    python generate_review.py <workspace> --skill-name NAME --benchmark <bench.json>
    python generate_review.py <workspace> --skill-name NAME --benchmark <bench.json> --static <output.html>
    python generate_review.py <workspace> --skill-name NAME --benchmark <bench.json> --previous-workspace <prev>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import socketserver
from http.server import SimpleHTTPRequestHandler

try:
    import markdown

    _HAS_MARKDOWN = True
except ImportError:
    _HAS_MARKDOWN = False


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def walk_runs(workspace: Path) -> list[dict[str, Any]]:
    runs = []
    for it_dir in sorted(workspace.glob("iteration-*"), key=lambda p: p.name):
        iteration = int(it_dir.name.split("-")[1])
        for eval_dir in sorted(it_dir.iterdir()):
            for config_dir in sorted(eval_dir.iterdir()):
                if config_dir.is_dir() and config_dir.name in (
                    "with_skill",
                    "without_skill",
                    "old_skill",
                ):
                    meta = load_json(config_dir / "eval_metadata.json")
                    grading = load_json(config_dir / "grading.json")
                    timing = load_json(config_dir / "timing.json")
                    feedback = load_json(config_dir / "feedback.json")
                    run_id = f"{eval_dir.name}/{config_dir.name}"
                    runs.append(
                        {
                            "iteration": iteration,
                            "eval_name": eval_dir.name,
                            "config": config_dir.name,
                            "run_id": run_id,
                            "meta": meta,
                            "grading": grading,
                            "timing": timing,
                            "feedback": feedback,
                            "output_dir": str(config_dir / "outputs"),
                            "outputs_exist": (config_dir / "outputs").exists(),
                            "iteration_dir": it_dir.name,
                        }
                    )
    return runs


def get_file_tree(root: Path, rel_prefix: str = "") -> list[dict[str, Any]]:
    items = []
    if not root.exists():
        return items
    for p in sorted(root.iterdir()):
        rel = f"{rel_prefix}/{p.name}" if rel_prefix else p.name
        if p.is_file():
            size = p.stat().st_size
            try:
                content = p.read_text(errors="replace")
                if size > 50000:
                    content = (
                        content[:50000] + f"\n... [truncated, {size - 50000:,} bytes remaining]"
                    )
                is_text = True
            except Exception:
                content = None
                is_text = False
            items.append(
                {"path": rel, "is_file": True, "size": size, "content": content, "is_text": is_text}
            )
        else:
            items.append({"path": rel + "/", "is_file": False})
            items.extend(get_file_tree(p, rel))
    return items


def format_time(ms: int) -> str:
    s = ms / 1000
    if s < 1:
        return f"{ms}ms"
    return f"{s:.1f}s"


def render_content(content: str, path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".csv", ".tsv"):
        return render_table(content)
    elif ext in (".json",):
        return render_json(content)
    elif ext in (".md", ".txt", ".log"):
        return f"<pre>{escape_html(content)}</pre>"
    elif ext in (".html",):
        return content
    else:
        return f"<pre>{escape_html(content)}</pre>"


def render_table(csv: str) -> str:
    lines = csv.strip().split("\n")
    if len(lines) < 2:
        return f"<pre>{escape_html(csv)}</pre>"
    rows = [l.split(",") if "," in l else l.split("\t") for l in lines]
    header = rows[0]
    body = rows[1:]
    h = "".join(f"<th>{escape_html(str(c))}</th>" for c in header)
    b = "".join(
        f"<tr>{''.join(f'<td>{escape_html(str(c))}</td>' for c in row)}</tr>" for row in body
    )
    return f"<table class='data-table'><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def render_json(text: str) -> str:
    try:
        obj = json.loads(text)
        pretty = json.dumps(obj, indent=2)
    except Exception:
        pretty = text
    return f"<pre class='json'>{escape_html(pretty)}</pre>"


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def make_html(
    workspace: Path,
    skill_name: str,
    benchmark: dict | None,
    previous_workspace: Path | None,
) -> str:
    runs = walk_runs(workspace)
    prev_runs = walk_runs(previous_workspace) if previous_workspace else []

    config_labels = {
        "with_skill": "With Skill",
        "without_skill": "Baseline",
        "old_skill": "Old Skill",
    }

    run_jsons = []
    for r in runs:
        meta = r["meta"] or {}
        grading = r["grading"] or {}
        timing = r["timing"] or {}
        files = []
        if r["outputs_exist"]:
            outputs = Path(r["output_dir"])
            for fi in get_file_tree(outputs):
                fi_copy = fi.copy()
                if fi["is_text"] and fi["content"] is not None:
                    fi_copy["rendered"] = render_content(fi["content"], fi["path"])
                files.append(fi_copy)

        pr = 0.0
        total = 0
        passed_count = 0
        for e in grading.get("expectations", []):
            total += 1
            if e.get("passed"):
                passed_count += 1
        if total > 0:
            pr = passed_count / total

        prev_fb = None
        for pr_ in prev_runs:
            if pr_["run_id"] == r["run_id"]:
                fb = pr_.get("feedback") or {}
                for item in fb.get("reviews", []):
                    if item.get("run_id") == r["run_id"]:
                        prev_fb = item.get("feedback", "")
                        break
                break

        run_jsons.append(
            {
                "run_id": r["run_id"],
                "iteration": r["iteration"],
                "eval_name": r["eval_name"],
                "config": r["config"],
                "config_label": config_labels.get(r["config"], r["config"]),
                "prompt": meta.get("prompt", ""),
                "expectations": grading.get("expectations", []),
                "pass_rate": pr,
                "passed_count": passed_count,
                "total_count": total,
                "time_ms": timing.get("duration_ms", 0),
                "tokens": timing.get("total_tokens", 0),
                "files": files,
                "feedback": r["feedback"],
                "prev_feedback": prev_fb,
            }
        )

    bench_json = json.dumps(benchmark or {}, indent=2)
    runs_json = json.dumps(run_jsons, indent=2)

    prev_iters = sorted(set(r.get("iteration", 0) for r in runs), reverse=True)
    current_iter = prev_iters[0] if prev_iters else 1

    bm_html = ""
    if benchmark:
        configs = benchmark.get("configurations", [])
        bm_html = (
            f"""
    <div class="benchmark-summary">
        <h3>Summary</h3>
        <table>
            <thead><tr><th>Config</th><th>Pass Rate</th><th>Mean Time</th><th>Mean Tokens</th></tr></thead>
            <tbody>
                {"".join(f"<tr><td>{c['name']}</td><td>{c['pass_rate']:.1%}</td><td>{c['mean_time_seconds']:.1f}s</td><td>{c['mean_tokens']:,}</td></tr>" for c in configs)}
            </tbody>
        </table>
        <h4>Analyst Notes</h4>
        <ul>
            {"".join(f"<li>{escape_html(n)}</li>" for n in benchmark.get("analyst_notes", []))}
        </ul>
    </div>"""
            if configs
            else "<p>No benchmark data available.</p>"
        )

    return _HTML_TEMPLATE.format(
        skill_name=escape_html(skill_name),
        current_iter=current_iter,
        benchmark=bm_html,
        runs_json=runs_json,
        benchmark_json=bench_json,
    )


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Skill Review — {skill_name}</title>
<style>
:root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #242836;
    --border: #2d3348;
    --text: #e4e7ef;
    --text-muted: #8b8fa3;
    --accent: #7c6af7;
    --accent-hover: #6b57f0;
    --pass: #4ade80;
    --fail: #f87171;
    --warn: #fbbf24;
    --tab-bg: #242836;
    --tab-active: #7c6af7;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }}
.header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }}
.header h1 {{ font-size: 1.1rem; font-weight: 600; }}
.header .tag {{ background: var(--accent); color: #fff; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 4px; }}
.tabs {{ background: var(--surface); border-bottom: 1px solid var(--border); display: flex; gap: 0; padding: 0 2rem; }}
.tab {{ padding: 0.8rem 1.5rem; cursor: pointer; color: var(--text-muted); font-size: 0.9rem; border: none; background: none; border-bottom: 2px solid transparent; transition: all 0.15s; }}
.tab:hover {{ color: var(--text); }}
.tab.active {{ color: var(--tab-active); border-bottom-color: var(--tab-active); }}
.content {{ padding: 2rem; max-width: 1200px; margin: 0 auto; width: 100%; }}
.tab-panel {{ display: none; }}
.tab-panel.active {{ display: block; }}
.nav-bar {{ display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem; }}
.nav-btn {{ background: var(--surface2); border: 1px solid var(--border); color: var(--text); padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }}
.nav-btn:hover {{ background: var(--border); }}
.nav-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
.eval-counter {{ color: var(--text-muted); font-size: 0.85rem; }}
.config-badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
.config-badge.with_skill {{ background: rgba(124,106,247,0.2); color: #a89df9; }}
.config-badge.without_skill {{ background: rgba(139,143,163,0.15); color: var(--text-muted); }}
.config-badge.old_skill {{ background: rgba(139,143,163,0.15); color: var(--text-muted); }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem; }}
.card-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; flex-wrap: wrap; }}
.card-title {{ font-size: 1rem; font-weight: 600; }}
.card-subtitle {{ font-size: 0.8rem; color: var(--text-muted); }}
.section-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin: 1rem 0 0.5rem; }}
.prompt {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; font-size: 0.9rem; line-height: 1.5; white-space: pre-wrap; margin-bottom: 1rem; }}
.metrics {{ display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap; }}
.metric {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; text-align: center; min-width: 80px; }}
.metric-value {{ font-size: 1.1rem; font-weight: 700; }}
.metric-label {{ font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; margin-top: 0.2rem; }}
.metric.pass .metric-value {{ color: var(--pass); }}
.metric.fail .metric-value {{ color: var(--fail); }}
.assertions {{ display: flex; flex-direction: column; gap: 0.5rem; margin-bottom: 1rem; }}
.assertion {{ display: flex; align-items: flex-start; gap: 0.75rem; padding: 0.75rem; background: var(--surface2); border-radius: 6px; font-size: 0.85rem; }}
.assertion .icon {{ flex-shrink: 0; width: 18px; height: 18px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; margin-top: 0.1rem; }}
.assertion.pass .icon {{ background: var(--pass); color: #000; }}
.assertion.fail .icon {{ background: var(--fail); color: #000; }}
.assertion-text {{ flex: 1; }}
.assertion-evidence {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }}
.files {{ display: flex; flex-direction: column; gap: 0.5rem; }}
.file-item {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }}
.file-header {{ display: flex; align-items: center; gap: 0.75rem; padding: 0.6rem 1rem; cursor: pointer; font-size: 0.85rem; user-select: none; }}
.file-header:hover {{ background: var(--border); }}
.file-header .arrow {{ transition: transform 0.15s; }}
.file-header.open .arrow {{ transform: rotate(90deg); }}
.file-size {{ color: var(--text-muted); font-size: 0.75rem; }}
.file-content {{ display: none; padding: 0.75rem; border-top: 1px solid var(--border); max-height: 400px; overflow: auto; }}
.file-content.open {{ display: block; }}
.file-content pre {{ font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap; word-break: break-all; }}
.file-content table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
.file-content th {{ background: var(--border); padding: 0.4rem 0.75rem; text-align: left; border: 1px solid var(--border); white-space: nowrap; }}
.file-content td {{ padding: 0.4rem 0.75rem; border: 1px solid var(--border); max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.file-content tr:nth-child(even) td {{ background: rgba(255,255,255,0.02); }}
.feedback-section {{ margin-top: 1rem; }}
.feedback-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 0.5rem; }}
textarea {{ width: 100%; min-height: 80px; background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: 0.75rem; font-family: inherit; font-size: 0.85rem; resize: vertical; }}
textarea:focus {{ outline: none; border-color: var(--accent); }}
.submit-bar {{ position: sticky; bottom: 0; background: var(--surface); border-top: 1px solid var(--border); padding: 1rem 2rem; display: flex; justify-content: flex-end; gap: 1rem; }}
.submit-btn {{ background: var(--accent); color: #fff; border: none; padding: 0.7rem 2rem; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; }}
.submit-btn:hover {{ background: var(--accent-hover); }}
.benchmark-summary table {{ width: 100%; border-collapse: collapse; margin-bottom: 1rem; }}
.benchmark-summary th {{ background: var(--surface2); padding: 0.75rem 1rem; text-align: left; border: 1px solid var(--border); font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }}
.benchmark-summary td {{ padding: 0.75rem 1rem; border: 1px solid var(--border); }}
.benchmark-summary ul {{ list-style: none; display: flex; flex-direction: column; gap: 0.4rem; }}
.benchmark-summary li {{ font-size: 0.85rem; color: var(--text-muted); padding: 0.4rem 0; border-bottom: 1px solid var(--border); }}
.prev-output {{ margin-bottom: 1rem; }}
.prev-badge {{ background: var(--warn); color: #000; font-size: 0.65rem; padding: 0.1rem 0.4rem; border-radius: 3px; font-weight: 700; text-transform: uppercase; margin-right: 0.5rem; }}
</style>
</head>
<body>
<div class="header">
    <h1>{skill_name}</h1>
    <span class="tag">Iteration {current_iter}</span>
</div>
<div class="tabs">
    <button class="tab active" data-tab="outputs">Outputs</button>
    <button class="tab" data-tab="benchmark">Benchmark</button>
</div>
<div class="content">
    <div id="tab-outputs" class="tab-panel active">
        <div class="nav-bar">
            <button class="nav-btn" id="prev-btn">&larr; Previous</button>
            <span class="eval-counter" id="eval-counter">1 / 3</span>
            <button class="nav-btn" id="next-btn">Next &rarr;</button>
        </div>
        <div id="eval-card"></div>
    </div>
    <div id="tab-benchmark" class="tab-panel">
{benchmark}
    </div>
</div>
<div class="submit-bar">
    <button class="submit-btn" id="submit-btn">Submit All Reviews</button>
</div>
<script>
const RUNS = {runs_json};
const BENCHMARK = {benchmark_json};
let currentIndex = 0;
const feedback = {{}};
const prevFeedback = {{}};
RUNS.forEach(r => {{
    feedback[r.run_id] = r.feedback?.reviews?.find(x => x.run_id === r.run_id)?.feedback || '';
    prevFeedback[r.run_id] = r.prev_feedback || '';
}});

function escapeHtml(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function renderRun(run, index) {{
    const prPct = (run.pass_rate * 100).toFixed(0);
    const prClass = run.pass_rate >= 0.8 ? 'pass' : run.pass_rate >= 0.5 ? '' : 'fail';
    const passed = run.expectations.filter(e => e.passed).length;
    let html = `<div class="card">
<div class="card-header">
    <span class="eval-counter">#${index + 1}</span>
    <span class="config-badge ${run.config}">${escapeHtml(run.config_label)}</span>
    <span class="card-title">${escapeHtml(run.eval_name)}</span>
</div>`;

    if (run.prompt) {{
        html += `<div class="section-label">Prompt</div>
<div class="prompt">${{escapeHtml(run.prompt)}}</div>`;
    }}

    html += `<div class="metrics">
<div class="metric ${{prClass}}">
<div class="metric-value">${{prPct}}%</div>
<div class="metric-label">Pass Rate</div>
</div>
<div class="metric">
<div class="metric-value">${{passed}}/${{run.total_count}}</div>
<div class="metric-label">Assertions</div>
</div>
<div class="metric">
<div class="metric-value">${{run.tokens.toLocaleString()}}</div>
<div class="metric-label">Tokens</div>
</div>
<div class="metric">
<div class="metric-value">${{(run.time_ms/1000).toFixed(1)}}s</div>
<div class="metric-label">Time</div>
</div>
</div>`;

    if (run.expectations.length > 0) {{
        html += `<div class="section-label">Assertions</div><div class="assertions">`;
        run.expectations.forEach(e => {{
            const cls = e.passed ? 'pass' : 'fail';
            const icon = e.passed ? '&#10003;' : '&#10007;';
            html += `<div class="assertion ${{cls}}">
<div class="icon">${{icon}}</div>
<div>
<div class="assertion-text">${{escapeHtml(e.text)}}</div>
<div class="assertion-evidence">${{escapeHtml(e.evidence || '')}}</div>
</div>
</div>`;
        }});
        html += `</div>`;
    }}

    if (run.files && run.files.length > 0) {{
        html += `<div class="section-label">Output Files</div><div class="files">`;
        run.files.forEach(f => {{
            if (f.is_file) {{
                const rendered = f.is_text && f.content !== null ? f.rendered : null;
                html += `<div class="file-item">
<div class="file-header" onclick="this.classList.toggle('open'); this.nextElementSibling.classList.toggle('open');">
    <span class="arrow">&#9658;</span>
    <span>${{escapeHtml(f.path)}}</span>
    <span class="file-size">${{(f.size/1024).toFixed(1)}} KB</span>
</div>
<div class="file-content">${{rendered ? rendered : '<pre>Binary file — cannot preview</pre>'}}</div>
</div>`;
            }} else {{
                html += `<div class="file-item"><div class="file-header"><span class="arrow">&#9658;</span><span>${{escapeHtml(f.path)}}</span></div></div>`;
            }}
        }});
        html += `</div>`;
    }}

    const fb = feedback[run.run_id] || '';
    const prevFb = prevFeedback[run.run_id] || '';
    if (prevFb) {{
        html += `<div class="feedback-section"><div class="prev-badge">Previous Feedback</div><div class="card" style="background:var(--surface2);margin-top:0.5rem;"><div class="prompt" style="margin:0">${{escapeHtml(prevFb)}}</div></div></div>`;
    }}
    html += `<div class="feedback-section">
<div class="feedback-label">Your Feedback</div>
<textarea id="fb-${{escapeHtml(run.run_id)}}" placeholder="What went well? What could be improved?">${{escapeHtml(fb)}}</textarea>
</div>`;

    html += `</div>`;
    return html;
}}

function updateCard() {{
    const run = RUNS[currentIndex];
    document.getElementById('eval-card').innerHTML = renderRun(run, currentIndex);
    document.getElementById('eval-counter').textContent = `${{currentIndex + 1}} / ${{RUNS.length}}`;
    document.getElementById('prev-btn').disabled = currentIndex === 0;
    document.getElementById('next-btn').disabled = currentIndex === RUNS.length - 1;
    const ta = document.getElementById(`fb-${{escapeHtml(run.run_id)}}`);
    if (ta) {{
        ta.addEventListener('input', e => {{ feedback[run.run_id] = e.target.value; }});
    }}
}}

document.addEventListener('keydown', e => {{
    if (e.key === 'ArrowLeft' && currentIndex > 0) {{ currentIndex--; updateCard(); }}
    if (e.key === 'ArrowRight' && currentIndex < RUNS.length - 1) {{ currentIndex++; updateCard(); }}
}});

document.getElementById('prev-btn').addEventListener('click', () => {{ if (currentIndex > 0) {{ currentIndex--; updateCard(); }} }});
document.getElementById('next-btn').addEventListener('click', () => {{ if (currentIndex < RUNS.length - 1) {{ currentIndex++; updateCard(); }} }});

document.querySelectorAll('.tab').forEach(t => {{
    t.addEventListener('click', () => {{
        document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(x => x.classList.remove('active'));
        t.classList.add('active');
        document.getElementById('tab-' + t.dataset.tab).classList.add('active');
    }});
}});

document.getElementById('submit-btn').addEventListener('click', () => {{
    const reviews = Object.entries(feedback).map(([run_id, fb]) => ({{
        run_id, feedback: fb, timestamp: new Date().toISOString()
    }}));
    const blob = new Blob([JSON.stringify({{ reviews, status: 'complete', submitted_at: new Date().toISOString() }}, null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'feedback.json';
    a.click();
}});

updateCard();
</script>
</body>
</html>"""


def serve_html(html: str, port: int = 8765) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            else:
                super().do_GET()

        def log_message(self, format, *args):
            pass

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        webbrowser.open(f"http://localhost:{port}")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval review HTML")
    parser.add_argument("workspace", type=Path, help="Workspace directory with iteration results")
    parser.add_argument("--skill-name", default="my-skill", help="Skill name for display")
    parser.add_argument("--benchmark", type=Path, help="Path to benchmark.json")
    parser.add_argument("--previous-workspace", type=Path, help="Previous workspace for comparison")
    parser.add_argument("--static", type=Path, help="Write static HTML to file instead of serving")
    args = parser.parse_args()

    benchmark = None
    if args.benchmark and args.benchmark.exists():
        benchmark = load_json(args.benchmark)

    html = make_html(args.workspace, args.skill_name, benchmark, args.previous_workspace)

    if args.static:
        args.static.parent.mkdir(parents=True, exist_ok=True)
        args.static.write_text(html)
        print(f"Written static HTML to {args.static}")
    else:
        port = 8765
        if "PORT" in os.environ:
            port = int(os.environ["PORT"])
        serve_html(html, port)


if __name__ == "__main__":
    main()
