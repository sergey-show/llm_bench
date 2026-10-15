from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_bench.schemas import BenchRun, RowPrediction
from llm_bench.settings import Settings, settings as default_settings

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _build_question_reviews(predictions: list[RowPrediction]) -> list[dict]:
    """Group predictions by question for side-by-side answer + judge review."""
    by_row: dict[str, list[RowPrediction]] = defaultdict(list)
    for p in predictions:
        by_row[p.row_id].append(p)

    reviews: list[dict] = []
    for row_id, rows in sorted(by_row.items(), key=lambda x: x[0]):
        rows_sorted = sorted(rows, key=lambda r: r.model_name)
        first = rows_sorted[0]
        reviews.append(
            {
                "row_id": row_id,
                "question": first.question,
                "reference_answer": first.reference_answer,
                "gold_score": first.gold_score,
                "tags": first.tags,
                "answers": [
                    {
                        "model_name": r.model_name,
                        "pred_answer": r.pred_answer or r.raw_output or "",
                        "judge_score": r.judge_score
                        if r.judge_score is not None
                        else r.metrics.get("llm_judge"),
                        "judge_rationale": r.judge_rationale,
                        "score_100": r.score_100,
                        "traffic_light": r.traffic_light,
                        "token_f1": r.metrics.get("token_f1"),
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "error": r.error,
                    }
                    for r in rows_sorted
                ],
            }
        )
    return reviews


def render_html_report(
    run: BenchRun,
    *,
    sample_limit: int = 40,
    review_limit: int | None = None,
    cfg: Settings | None = None,
) -> str:
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
    best_efficiency = (
        max(with_eff, key=lambda s: s.score_per_1k_tokens or 0.0) if with_eff else None
    )
    counts = Counter(s.traffic_light or "unknown" for s in run.summaries)
    sample = run.predictions[:sample_limit]
    reviews = _build_question_reviews(run.predictions)
    if review_limit is not None:
        reviews = reviews[:review_limit]
    return template.render(
        run=run,
        best=best,
        best_efficiency=best_efficiency,
        sample=sample,
        reviews=reviews,
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
