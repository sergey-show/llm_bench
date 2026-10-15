from __future__ import annotations

import json
import math
import re
from collections import Counter

from llm_bench.schemas import ScorerKind

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_score(text: str) -> float | None:
    if not text:
        return None
    text = text.strip()
    # JSON first
    if "{" in text and "}" in text:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            obj = json.loads(text[start:end])
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception:
            pass
    try:
        return float(text)
    except ValueError:
        pass
    matches = _NUM_RE.findall(text.replace(",", "."))
    if not matches:
        return None
    return float(matches[-1])


def extract_judge_verdict(text: str) -> tuple[str | None, float | None]:
    """Parse judge output into (analysis, score)."""
    if not text:
        return None, None
    text = text.strip()
    if "{" in text and "}" in text:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            obj = json.loads(text[start:end])
            if isinstance(obj, dict):
                analysis = obj.get("analysis") or obj.get("rationale") or obj.get("reasoning")
                score_raw = obj.get("score")
                score = float(score_raw) if score_raw is not None else extract_score(text)
                if analysis is not None:
                    return str(analysis).strip(), score
                return text, score
        except Exception:
            pass
    score = extract_score(text)
    # If model returned only a number, no separate analysis
    if score is not None and text.replace(",", ".").strip() in {
        str(score),
        f"{score:g}",
        f"{score:.1f}",
        f"{score:.0f}",
    }:
        return None, score
    return text, score


def extract_answer_and_score(text: str) -> tuple[str | None, float | None]:
    if not text:
        return None, None
    text = text.strip()
    if "{" in text and "}" in text:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            obj = json.loads(text[start:end])
            if isinstance(obj, dict):
                ans = obj.get("answer")
                score = obj.get("score")
                return (
                    str(ans) if ans is not None else None,
                    float(score) if score is not None else None,
                )
        except Exception:
            pass
    return text, extract_score(text)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def exact_match(pred: str | None, ref: str | None) -> float | None:
    if pred is None or ref is None:
        return None
    a = " ".join(_tokens(pred))
    b = " ".join(_tokens(ref))
    if not a and not b:
        return 1.0
    return 1.0 if a == b else 0.0


def token_f1(pred: str | None, ref: str | None) -> float | None:
    if pred is None or ref is None:
        return None
    pred_t = _tokens(pred)
    ref_t = _tokens(ref)
    if not pred_t and not ref_t:
        return 1.0
    if not pred_t or not ref_t:
        return 0.0
    common = Counter(pred_t) & Counter(ref_t)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_t)
    recall = num_same / len(ref_t)
    return 2 * precision * recall / (precision + recall)


def absolute_error(pred: float | None, gold: float | None) -> float | None:
    if pred is None or gold is None:
        return None
    return abs(pred - gold)


def squared_error(pred: float | None, gold: float | None) -> float | None:
    if pred is None or gold is None:
        return None
    return (pred - gold) ** 2


def compute_row_metrics(
    *,
    pred_score: float | None,
    gold_score: float | None,
    pred_answer: str | None,
    reference_answer: str | None,
    scorers: list[ScorerKind],
    judge_score: float | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    # Prefer candidate pred_score; if absent (answer_only), fall back to judge vs gold.
    score_for_error = pred_score
    if score_for_error is None and judge_score is not None:
        score_for_error = judge_score

    for kind in scorers:
        if kind == ScorerKind.score_mae:
            ae = absolute_error(score_for_error, gold_score)
            if ae is not None:
                out["abs_error"] = ae
        elif kind == ScorerKind.score_rmse:
            se = squared_error(score_for_error, gold_score)
            if se is not None:
                out["sq_error"] = se
        elif kind == ScorerKind.exact_match:
            em = exact_match(pred_answer, reference_answer)
            if em is not None:
                out["exact_match"] = em
        elif kind == ScorerKind.token_f1:
            f1 = token_f1(pred_answer, reference_answer)
            if f1 is not None:
                out["token_f1"] = f1
        elif kind == ScorerKind.llm_judge:
            if judge_score is not None and math.isfinite(judge_score):
                out["llm_judge"] = float(judge_score)
        elif kind == ScorerKind.embedding_cosine:
            # Placeholder for optional embedding backend; skipped if unavailable.
            pass
    return out
