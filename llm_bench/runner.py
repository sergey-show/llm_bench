from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from llm_bench.client import chat_complete
from llm_bench.metrics import summarize_by_tag, summarize_predictions
from llm_bench.prompts import build_judge_prompt, build_prompt
from llm_bench.schemas import (
    BenchRun,
    DatasetBundle,
    DatasetRow,
    ModelConfig,
    RowPrediction,
    RunMode,
    RunRequest,
    ScorerKind,
)
from llm_bench.scorers import (
    compute_row_metrics,
    extract_answer_and_score,
    extract_judge_verdict,
    extract_score,
)
from llm_bench.settings import Settings, settings as default_settings
from llm_bench.storage import save_run
from llm_bench.traffic_light import classify_traffic_light, row_score_100


ProgressCallback = Callable[[int, int, str], None]


async def _predict_one(
    *,
    row: DatasetRow,
    model: ModelConfig,
    mode: RunMode,
    scorers: list[ScorerKind],
    judge_model: ModelConfig | None,
    prompt_template: str | None,
    semaphore: asyncio.Semaphore,
    cfg: Settings,
) -> RowPrediction:
    async with semaphore:
        prompt = build_prompt(
            row,
            mode,
            score_min=cfg.score_min,
            score_max=cfg.score_max,
            template=prompt_template,
        )
        try:
            result = await chat_complete(model, prompt, cfg=cfg)
            pred_answer: str | None = None
            pred_score: float | None = None
            if mode == RunMode.score_only:
                pred_score = extract_score(result.text)
            elif mode == RunMode.answer_only:
                pred_answer = result.text
            else:
                pred_answer, pred_score = extract_answer_and_score(result.text)

            judge_score = None
            judge_rationale = None
            judge_raw = None
            if ScorerKind.llm_judge in scorers and judge_model and pred_answer:
                judge_prompt = build_judge_prompt(
                    row,
                    pred_answer,
                    score_min=cfg.score_min,
                    score_max=cfg.score_max,
                )
                judge_res = await chat_complete(judge_model, judge_prompt, cfg=cfg)
                judge_raw = judge_res.text
                judge_rationale, judge_score = extract_judge_verdict(judge_res.text)

            metrics = compute_row_metrics(
                pred_score=pred_score,
                gold_score=row.gold_score,
                pred_answer=pred_answer,
                reference_answer=row.reference_answer,
                scorers=scorers,
                judge_score=judge_score,
            )
            s100 = row_score_100(metrics, cfg=cfg)
            light = classify_traffic_light(s100, cfg)
            return RowPrediction(
                row_id=row.id,
                model_id=model.id,
                model_name=model.name,
                question=row.question,
                raw_output=result.text,
                pred_answer=pred_answer,
                pred_score=pred_score,
                gold_score=row.gold_score,
                reference_answer=row.reference_answer,
                metrics=metrics,
                score_100=s100,
                traffic_light=light.value,
                judge_score=judge_score,
                judge_rationale=judge_rationale,
                judge_raw=judge_raw,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                tags=row.tags,
                weight=row.weight,
            )
        except Exception as exc:  # noqa: BLE001 — capture per-row failures
            return RowPrediction(
                row_id=row.id,
                model_id=model.id,
                model_name=model.name,
                question=row.question,
                gold_score=row.gold_score,
                reference_answer=row.reference_answer,
                error=str(exc),
                traffic_light="red",
                score_100=0.0,
                tags=row.tags,
                weight=row.weight,
            )


async def run_benchmark(
    *,
    dataset: DatasetBundle,
    models: list[ModelConfig],
    request: RunRequest,
    judge_model: ModelConfig | None = None,
    cfg: Settings | None = None,
    on_progress: ProgressCallback | None = None,
) -> BenchRun:
    cfg = cfg or default_settings
    if not dataset.calibration.ready_to_run:
        raise ValueError("Dataset is not calibrated / ready_to_run=false")
    if not models:
        raise ValueError("No models selected")

    rows = dataset.rows
    if request.limit is not None:
        rows = rows[: max(0, request.limit)]

    # Per-model concurrency caps
    global_limit = request.concurrency or cfg.default_concurrency
    semaphores = {
        m.id: asyncio.Semaphore(min(global_limit, max(1, m.concurrency)))
        for m in models
    }

    tasks = []
    for model in models:
        for row in rows:
            tasks.append(
                _predict_one(
                    row=row,
                    model=model,
                    mode=request.mode,
                    scorers=request.scorers,
                    judge_model=judge_model,
                    prompt_template=request.prompt_template,
                    semaphore=semaphores[model.id],
                    cfg=cfg,
                )
            )

    total = len(tasks)
    predictions: list[RowPrediction] = []
    done = 0

    for coro in asyncio.as_completed(tasks):
        pred = await coro
        predictions.append(pred)
        done += 1
        if on_progress:
            on_progress(done, total, f"{pred.model_name}:{pred.row_id}")

    summaries = summarize_predictions(predictions, cfg=cfg)
    tag_breakdown = summarize_by_tag(predictions, cfg=cfg)
    run = BenchRun(
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        mode=request.mode,
        scorers=request.scorers,
        model_ids=[m.id for m in models],
        predictions=predictions,
        summaries=summaries,
        tag_breakdown=tag_breakdown,
        meta={
            "n_rows": len(rows),
            "dataset_hash": dataset.content_hash,
            "calibration_ready": dataset.calibration.ready_to_run,
        },
    )
    save_run(run, cfg)
    return run


def run_benchmark_sync(**kwargs: Any) -> BenchRun:
    return asyncio.run(run_benchmark(**kwargs))
