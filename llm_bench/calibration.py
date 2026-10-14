from __future__ import annotations

from collections import Counter
from math import isfinite

import numpy as np

from llm_bench.schemas import (
    CalibrationIssue,
    CalibrationReport,
    CalibrationSeverity,
    DatasetRow,
)
from llm_bench.settings import Settings, settings as default_settings


def _bin_label(score: float, lo: float, hi: float, n_bins: int = 5) -> str:
    if hi <= lo:
        return f"[{lo:.1f}]"
    width = (hi - lo) / n_bins
    idx = min(int((score - lo) / width), n_bins - 1) if width > 0 else 0
    a = lo + idx * width
    b = lo + (idx + 1) * width
    return f"[{a:.1f}, {b:.1f})"


def calibrate_dataset(
    rows: list[DatasetRow],
    *,
    normalize_weights: bool = True,
    cfg: Settings | None = None,
) -> tuple[list[DatasetRow], CalibrationReport]:
    """Validate and calibrate a labeled dataset before benchmarking.

    Calibration covers schema readiness, score distribution, duplicates,
    weight balance, and actionable readiness gates for MAE-style runs.
    """
    cfg = cfg or default_settings
    issues: list[CalibrationIssue] = []
    suggestions: list[str] = []

    if not rows:
        report = CalibrationReport(
            n_rows=0,
            n_valid=0,
            n_dropped=0,
            score_present=0,
            answer_present=0,
            ready_to_run=False,
            issues=[
                CalibrationIssue(
                    code="empty_dataset",
                    severity=CalibrationSeverity.error,
                    message="Dataset is empty after parsing.",
                )
            ],
            suggestions=["Upload a CSV/JSONL with at least question and gold_score or reference_answer."],
        )
        return [], report

    # Drop rows with empty questions
    valid: list[DatasetRow] = []
    bad_questions: list[str] = []
    for row in rows:
        if not row.question or not str(row.question).strip():
            bad_questions.append(row.id)
            continue
        valid.append(row)

    if bad_questions:
        issues.append(
            CalibrationIssue(
                code="empty_question",
                severity=CalibrationSeverity.error,
                message="Rows with empty question were dropped.",
                count=len(bad_questions),
                examples=bad_questions[:5],
            )
        )

    # Duplicate IDs
    id_counts = Counter(r.id for r in valid)
    dup_ids = [i for i, c in id_counts.items() if c > 1]
    if dup_ids:
        issues.append(
            CalibrationIssue(
                code="duplicate_ids",
                severity=CalibrationSeverity.warning,
                message="Duplicate row ids found; later metrics may collide.",
                count=len(dup_ids),
                examples=dup_ids[:5],
            )
        )
        # Make ids unique
        seen: dict[str, int] = {}
        fixed: list[DatasetRow] = []
        for row in valid:
            n = seen.get(row.id, 0)
            seen[row.id] = n + 1
            if n == 0:
                fixed.append(row)
            else:
                fixed.append(row.model_copy(update={"id": f"{row.id}__{n}"}))
        valid = fixed
        suggestions.append("Prefer unique stable ids in the source file.")

    # Duplicate questions
    q_counts = Counter(r.question.strip().lower() for r in valid)
    dup_q = sum(1 for c in q_counts.values() if c > 1)
    if dup_q:
        issues.append(
            CalibrationIssue(
                code="duplicate_questions",
                severity=CalibrationSeverity.warning,
                message="Duplicate questions detected (case-insensitive).",
                count=dup_q,
            )
        )

    scores = [r.gold_score for r in valid if r.gold_score is not None and isfinite(r.gold_score)]
    answers = [r for r in valid if r.reference_answer and str(r.reference_answer).strip()]

    out_of_range = [
        r.id
        for r in valid
        if r.gold_score is not None
        and isfinite(r.gold_score)
        and not (cfg.score_min <= r.gold_score <= cfg.score_max)
    ]
    if out_of_range:
        issues.append(
            CalibrationIssue(
                code="score_out_of_range",
                severity=CalibrationSeverity.warning,
                message=(
                    f"gold_score outside configured range "
                    f"[{cfg.score_min}, {cfg.score_max}]."
                ),
                count=len(out_of_range),
                examples=out_of_range[:5],
            )
        )
        suggestions.append(
            f"Align gold_score to [{cfg.score_min}, {cfg.score_max}] "
            "or change LLM_BENCH_SCORE_MIN/MAX."
        )

    non_finite = [
        r.id
        for r in valid
        if r.gold_score is not None and not isfinite(r.gold_score)
    ]
    if non_finite:
        issues.append(
            CalibrationIssue(
                code="score_non_finite",
                severity=CalibrationSeverity.error,
                message="Non-finite gold_score values found.",
                count=len(non_finite),
                examples=non_finite[:5],
            )
        )
        valid = [
            r
            for r in valid
            if r.gold_score is None or isfinite(r.gold_score)
        ]
        scores = [r.gold_score for r in valid if r.gold_score is not None]

    if not scores and not answers:
        issues.append(
            CalibrationIssue(
                code="no_labels",
                severity=CalibrationSeverity.error,
                message="Need gold_score and/or reference_answer for evaluation.",
            )
        )
    elif not scores:
        issues.append(
            CalibrationIssue(
                code="no_gold_score",
                severity=CalibrationSeverity.warning,
                message="No gold_score column — MAE/RMSE unavailable; use answer metrics.",
            )
        )
        suggestions.append("Add gold_score for regression metrics (MAE, RMSE, correlations).")
    elif not answers:
        issues.append(
            CalibrationIssue(
                code="no_reference_answer",
                severity=CalibrationSeverity.info,
                message="No reference_answer — answer similarity metrics unavailable.",
            )
        )

    score_bins: dict[str, int] = {}
    score_min = score_max = score_mean = score_std = None
    if scores:
        arr = np.asarray(scores, dtype=float)
        score_min = float(arr.min())
        score_max = float(arr.max())
        score_mean = float(arr.mean())
        score_std = float(arr.std(ddof=0))
        lo, hi = cfg.score_min, cfg.score_max
        if score_min is not None and score_max is not None and score_max > score_min:
            # Prefer configured scale for bins when most scores fit
            in_range = sum(1 for s in scores if lo <= s <= hi)
            if in_range / len(scores) >= 0.8:
                bin_lo, bin_hi = lo, hi
            else:
                bin_lo, bin_hi = score_min, score_max
        else:
            bin_lo, bin_hi = lo, hi
        for s in scores:
            label = _bin_label(s, bin_lo, bin_hi)
            score_bins[label] = score_bins.get(label, 0) + 1

        # Imbalance: one bin > 70%
        if score_bins:
            top = max(score_bins.values())
            if top / len(scores) >= 0.7:
                issues.append(
                    CalibrationIssue(
                        code="score_imbalance",
                        severity=CalibrationSeverity.warning,
                        message="Score distribution is heavily skewed to one bin (>70%).",
                        count=top,
                    )
                )
                suggestions.append(
                    "Balance score bins or report metrics by tag/bin to avoid misleading MAE."
                )

        if score_std is not None and score_std < 1e-6:
            issues.append(
                CalibrationIssue(
                    code="score_constant",
                    severity=CalibrationSeverity.warning,
                    message="All gold_score values are (nearly) constant — MAE will be hard to interpret.",
                )
            )

    tag_counts = Counter(tag for r in valid for tag in r.tags)

    # Weights
    weights = [r.weight if r.weight > 0 else 0.0 for r in valid]
    weight_sum = float(sum(weights))
    normalized = False
    if any(w <= 0 for w in weights):
        issues.append(
            CalibrationIssue(
                code="non_positive_weight",
                severity=CalibrationSeverity.warning,
                message="Some weights are <= 0; they will be treated as 0 and ignored in weighted metrics.",
                count=sum(1 for w in weights if w <= 0),
            )
        )
    if normalize_weights and weight_sum > 0 and abs(weight_sum - len(valid)) > 1e-6:
        scale = len(valid) / weight_sum
        valid = [
            r.model_copy(update={"weight": (r.weight if r.weight > 0 else 0.0) * scale})
            for r in valid
        ]
        weight_sum = float(sum(r.weight for r in valid))
        normalized = True
        suggestions.append("Weights were renormalized to mean 1.0 for stable aggregation.")

    # Small sample
    if len(valid) < 20:
        issues.append(
            CalibrationIssue(
                code="small_sample",
                severity=CalibrationSeverity.warning,
                message="Fewer than 20 valid rows — ranking models may be unstable.",
                count=len(valid),
            )
        )
        suggestions.append("Aim for 50+ labeled rows for more reliable model selection.")

    has_errors = any(i.severity == CalibrationSeverity.error for i in issues)
    ready = (not has_errors) and len(valid) > 0 and (bool(scores) or bool(answers))

    if ready and scores:
        suggestions.append(
            "Ready for score_only / answer_and_score runs with MAE & RMSE."
        )
    if ready and answers:
        suggestions.append(
            "Ready for answer_only runs with exact_match / token_f1."
        )

    report = CalibrationReport(
        n_rows=len(rows),
        n_valid=len(valid),
        n_dropped=len(rows) - len(valid),
        score_present=len(scores),
        answer_present=len(answers),
        score_min=score_min,
        score_max=score_max,
        score_mean=score_mean,
        score_std=score_std,
        score_bins=dict(sorted(score_bins.items())),
        tag_counts=dict(tag_counts.most_common()),
        duplicate_ids=len(dup_ids),
        duplicate_questions=dup_q,
        weight_sum=weight_sum,
        normalized_weights=normalized,
        ready_to_run=ready,
        issues=issues,
        suggestions=suggestions,
    )
    return valid, report
