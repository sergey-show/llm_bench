from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from llm_bench.schemas import ModelSummary, RowPrediction, TagBreakdown
from llm_bench.settings import Settings, settings as default_settings
from llm_bench.traffic_light import classify_traffic_light, composite_score_100


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _corr(xs: list[float], ys: list[float], method: str) -> float | None:
    if len(xs) < 2:
        return None
    a = np.asarray(xs, dtype=float)
    b = np.asarray(ys, dtype=float)
    if method == "pearson":
        if a.std() < 1e-12 or b.std() < 1e-12:
            return None
        return float(np.corrcoef(a, b)[0, 1])
    # spearman via rank
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(v), dtype=float)
        # average ties simply: skip for MVP uniqueness; OK for scores
        return ranks

    ra, rb = rank(a), rank(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _weighted_mean(values: list[float], weights: list[float]) -> float | None:
    if not values:
        return None
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    if float(w.sum()) <= 0:
        return float(v.mean())
    return float(np.average(v, weights=w))


def summarize_predictions(
    predictions: list[RowPrediction],
    cfg: Settings | None = None,
) -> list[ModelSummary]:
    cfg = cfg or default_settings
    by_model: dict[str, list[RowPrediction]] = defaultdict(list)
    for p in predictions:
        by_model[p.model_id].append(p)

    summaries: list[ModelSummary] = []
    for model_id, rows in by_model.items():
        name = rows[0].model_name
        n_total = len(rows)
        ok = [r for r in rows if not r.error]
        n_ok = len(ok)
        n_error = n_total - n_ok
        error_rate = (n_error / n_total) if n_total else 0.0

        abs_errors, abs_w = [], []
        sq_errors, sq_w = [], []
        pred_scores, gold_scores = [], []
        ems, em_w = [], []
        f1s, f1_w = [], []
        judges, judge_w = [], []
        latencies = []
        prompt_tokens = 0
        completion_tokens = 0

        for r in ok:
            w = r.weight if r.weight > 0 else 0.0
            if "abs_error" in r.metrics:
                abs_errors.append(r.metrics["abs_error"])
                abs_w.append(w)
            if "sq_error" in r.metrics:
                sq_errors.append(r.metrics["sq_error"])
                sq_w.append(w)
            if r.pred_score is not None and r.gold_score is not None:
                pred_scores.append(r.pred_score)
                gold_scores.append(r.gold_score)
            if "exact_match" in r.metrics:
                ems.append(r.metrics["exact_match"])
                em_w.append(w)
            if "token_f1" in r.metrics:
                f1s.append(r.metrics["token_f1"])
                f1_w.append(w)
            if "llm_judge" in r.metrics:
                judges.append(r.metrics["llm_judge"])
                judge_w.append(w)
            if r.latency_ms is not None:
                latencies.append(r.latency_ms)
            prompt_tokens += r.prompt_tokens or 0
            completion_tokens += r.completion_tokens or 0

        mae = _weighted_mean(abs_errors, abs_w)
        mse = _weighted_mean(sq_errors, sq_w)
        rmse = math.sqrt(mse) if mse is not None else None
        exact = _weighted_mean(ems, em_w)
        f1 = _weighted_mean(f1s, f1_w)
        judge = _weighted_mean(judges, judge_w)
        score_100 = composite_score_100(
            mae=mae,
            token_f1=f1,
            exact_match=exact,
            llm_judge=judge,
            error_rate=error_rate,
            cfg=cfg,
        )
        light = classify_traffic_light(score_100, cfg)

        summaries.append(
            ModelSummary(
                model_id=model_id,
                model_name=name,
                n_total=n_total,
                n_ok=n_ok,
                n_error=n_error,
                mae=mae,
                rmse=rmse,
                spearman=_corr(pred_scores, gold_scores, "spearman"),
                pearson=_corr(pred_scores, gold_scores, "pearson"),
                exact_match=exact,
                token_f1=f1,
                llm_judge=judge,
                score_100=score_100,
                traffic_light=light.value,
                latency_p50_ms=_percentile(latencies, 50),
                latency_p95_ms=_percentile(latencies, 95),
                error_rate=error_rate,
                total_prompt_tokens=prompt_tokens,
                total_completion_tokens=completion_tokens,
            )
        )
    return summaries


def summarize_by_tag(
    predictions: list[RowPrediction],
    cfg: Settings | None = None,
) -> list[TagBreakdown]:
    cfg = cfg or default_settings
    buckets: dict[tuple[str, str], list[RowPrediction]] = defaultdict(list)
    for p in predictions:
        tags = p.tags or ["__untagged__"]
        for tag in tags:
            buckets[(tag, p.model_id)].append(p)

    out: list[TagBreakdown] = []
    for (tag, model_id), rows in sorted(buckets.items()):
        ok = [r for r in rows if not r.error]
        abs_errors, abs_w, f1s, f1_w, ems, em_w = [], [], [], [], [], []
        for r in ok:
            w = r.weight if r.weight > 0 else 0.0
            if "abs_error" in r.metrics:
                abs_errors.append(r.metrics["abs_error"])
                abs_w.append(w)
            if "token_f1" in r.metrics:
                f1s.append(r.metrics["token_f1"])
                f1_w.append(w)
            if "exact_match" in r.metrics:
                ems.append(r.metrics["exact_match"])
                em_w.append(w)
        mae = _weighted_mean(abs_errors, abs_w)
        f1 = _weighted_mean(f1s, f1_w)
        exact = _weighted_mean(ems, em_w)
        score_100 = composite_score_100(
            mae=mae,
            token_f1=f1,
            exact_match=exact,
            error_rate=0.0,
            cfg=cfg,
        )
        light = classify_traffic_light(score_100, cfg)
        out.append(
            TagBreakdown(
                tag=tag,
                model_id=model_id,
                model_name=rows[0].model_name,
                n=len(ok),
                mae=mae,
                token_f1=f1,
                exact_match=exact,
                score_100=score_100,
                traffic_light=light.value,
            )
        )
    return out
