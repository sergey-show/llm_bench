from __future__ import annotations

from collections import Counter
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_bench.schemas import BenchRun
from llm_bench.settings import Settings, settings as default_settings

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def render_html_report(run: BenchRun, *, sample_limit: int = 40, cfg: Settings | None = None) -> str:
    cfg = cfg or default_settings
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")
    best = next((s for s in run.summaries if s.score_100 is not None), None)
    if best is None:
        best = next((s for s in run.summaries if s.mae is not None), None)
    with_eff = [s for s in run.summaries if s.score_per_1k_tokens is not None]
    best_efficiency = max(with_eff, key=lambda s: s.score_per_1k_tokens or 0.0) if with_eff else None
    counts = Counter(s.traffic_light or "unknown" for s in run.summaries)
    sample = run.predictions[:sample_limit]
    return template.render(
        run=run,
        best=best,
        best_efficiency=best_efficiency,
        sample=sample,
        scorers=", ".join(s.value for s in run.scorers),
        traffic_counts={
            "green": counts.get("green", 0),
            "yellow": counts.get("yellow", 0),
            "red": counts.get("red", 0),
            "unknown": counts.get("unknown", 0),
        },
        thresholds={
            "green": cfg.traffic_green_min,
            "yellow": cfg.traffic_yellow_min,
        },
    )


def write_html_report(run: BenchRun, cfg: Settings | None = None) -> Path:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    html = render_html_report(run, cfg=cfg)
    path = cfg.runs_dir / f"{run.id}.html"
    path.write_text(html, encoding="utf-8")
    return path
