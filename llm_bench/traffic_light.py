from __future__ import annotations

from enum import Enum

from llm_bench.settings import Settings, settings as default_settings


class TrafficLight(str, Enum):
    green = "green"
    yellow = "yellow"
    red = "red"
    unknown = "unknown"

    @property
    def emoji(self) -> str:
        return {
            TrafficLight.green: "🟢",
            TrafficLight.yellow: "🟡",
            TrafficLight.red: "🔴",
            TrafficLight.unknown: "⚪",
        }[self]

    @property
    def label_ru(self) -> str:
        return {
            TrafficLight.green: "зелёный",
            TrafficLight.yellow: "жёлтый",
            TrafficLight.red: "красный",
            TrafficLight.unknown: "н/д",
        }[self]


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def classify_traffic_light(score_100: float | None, cfg: Settings | None = None) -> TrafficLight:
    cfg = cfg or default_settings
    if score_100 is None:
        return TrafficLight.unknown
    if score_100 >= cfg.traffic_green_min:
        return TrafficLight.green
    if score_100 >= cfg.traffic_yellow_min:
        return TrafficLight.yellow
    return TrafficLight.red


def score_from_abs_error(
    abs_error: float | None,
    *,
    score_min: float,
    score_max: float,
) -> float | None:
    """Map absolute error on [score_min, score_max] to 0..100 (100 = perfect)."""
    if abs_error is None:
        return None
    span = score_max - score_min
    if span <= 0:
        return 100.0 if abs_error <= 0 else 0.0
    return clamp(100.0 * (1.0 - abs_error / span))


def score_from_unit_metric(value: float | None) -> float | None:
    """Map 0..1 metric (F1, exact) to 0..100."""
    if value is None:
        return None
    return clamp(100.0 * value)


def score_from_scale(
    value: float | None,
    *,
    score_min: float,
    score_max: float,
) -> float | None:
    """Map a value on [score_min, score_max] to 0..100."""
    if value is None:
        return None
    span = score_max - score_min
    if span <= 0:
        return 100.0 if value >= score_min else 0.0
    return clamp(100.0 * (value - score_min) / span)


def composite_score_100(
    *,
    mae: float | None = None,
    token_f1: float | None = None,
    exact_match: float | None = None,
    llm_judge: float | None = None,
    error_rate: float = 0.0,
    cfg: Settings | None = None,
) -> float | None:
    """Build a single 0..100 quality score from available metrics."""
    cfg = cfg or default_settings
    parts: list[tuple[float, float]] = []  # (score, weight)

    mae_s = score_from_abs_error(mae, score_min=cfg.score_min, score_max=cfg.score_max)
    if mae_s is not None:
        parts.append((mae_s, cfg.traffic_weight_mae))

    f1_s = score_from_unit_metric(token_f1)
    if f1_s is not None:
        parts.append((f1_s, cfg.traffic_weight_f1))

    em_s = score_from_unit_metric(exact_match)
    if em_s is not None:
        parts.append((em_s, cfg.traffic_weight_exact))

    judge_s = score_from_scale(
        llm_judge, score_min=cfg.score_min, score_max=cfg.score_max
    )
    if judge_s is not None:
        parts.append((judge_s, cfg.traffic_weight_judge))

    if not parts:
        return None

    total_w = sum(w for _, w in parts)
    if total_w <= 0:
        return None
    base = sum(s * w for s, w in parts) / total_w
    # Penalize unreliable runs
    penalty = clamp(1.0 - max(0.0, min(1.0, error_rate)), 0.0, 1.0)
    return clamp(base * penalty)


def row_score_100(
    metrics: dict[str, float],
    *,
    cfg: Settings | None = None,
) -> float | None:
    cfg = cfg or default_settings
    if "abs_error" in metrics:
        return score_from_abs_error(
            metrics["abs_error"],
            score_min=cfg.score_min,
            score_max=cfg.score_max,
        )
    if "token_f1" in metrics:
        return score_from_unit_metric(metrics["token_f1"])
    if "exact_match" in metrics:
        return score_from_unit_metric(metrics["exact_match"])
    if "llm_judge" in metrics:
        return score_from_scale(
            metrics["llm_judge"],
            score_min=cfg.score_min,
            score_max=cfg.score_max,
        )
    return None
