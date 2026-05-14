"""Self-contained HTML report per run.

Inlines PNGs as base64 and links the WebP replay locally. The file lives
inside the run's viz/ folder so it can be shared without the server.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

_STYLE = """
  body { font-family: ui-monospace, Menlo, monospace; background: #0e1116;
         color: #d8dee9; margin: 0; padding: 24px; }
  h1 { font-size: 16px; letter-spacing: 0.1em; margin: 0 0 16px; }
  h2 { font-size: 13px; color: #8c98a8; margin: 24px 0 8px; }
  .summary { background: #161b22; padding: 12px; border-radius: 4px;
             white-space: pre-wrap; font-size: 11px; color: #d8dee9;
             border: 1px solid #2a323c; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
          margin-top: 14px; }
  img, video { width: 100%; border: 1px solid #2a323c; border-radius: 4px; }
  .full { grid-column: span 2; }
  footer { color: #8c98a8; font-size: 10px; margin-top: 24px; }
"""


def _embed(p: Path) -> str:
    if not p.exists():
        return ""
    mime = {"png": "image/png", "webp": "image/webp",
            "gif": "image/gif", "mp4": "video/mp4"}.get(p.suffix[1:], "")
    b = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b}"


def write_html_report(run_dir: Path) -> Path:
    """Render viz/report.html using the run's existing artifacts."""
    run_dir = Path(run_dir)
    viz = run_dir / "viz"
    out = viz / "report.html"

    results = {}
    rj = run_dir / "results.json"
    if rj.exists():
        try: results = json.loads(rj.read_text())
        except Exception: pass

    summary_lines = []
    for k, v in results.items():
        if not isinstance(v, (list, dict)):
            summary_lines.append(f"{k}: {v}")
    summary = "\n".join(summary_lines) or "(no results recorded)"

    panels: list[str] = []
    def add(title: str, fname: str, full: bool = False):
        p = viz / fname
        if not p.exists():
            return
        src = _embed(p)
        cls = "full" if full else ""
        tag = ("<video controls autoplay loop muted>"
               f"<source src='{src}'></video>") if p.suffix == ".mp4" \
              else f"<img src='{src}'>"
        panels.append(f"<div class='{cls}'><h2>{title}</h2>{tag}</div>")

    add("Replay (greedy episode)", "replay.webp", full=True)
    add("Replay (greedy episode)", "replay.gif",  full=True)
    add("Replay (greedy episode)", "replay.mp4",  full=True)
    add("Training curves",         "curves.png")
    add("Policy heatmap",          "policy.png")
    add("Behavioral rollout",      "rollout.png")
    add("Visitation",              "visitation.png")

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>{run_dir.name}</title><style>{_STYLE}</style></head>
<body><h1>deepMaze — {run_dir.name}</h1>
<pre class='summary'>{summary}</pre>
<div class='grid'>{''.join(panels)}</div>
<footer>self-contained — base64 embedded</footer>
</body></html>"""
    out.write_text(html)
    return out
