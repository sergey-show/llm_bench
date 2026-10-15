from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelConfig(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    base_url: str
    api_key: str
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 4
    enabled: bool = True
    system_prompt: str | None = None
    provider: str | None = None
    source: str | None = None  # "yaml" | "manual" | None
    is_judge: bool = False  # listed under YAML judges
    is_candidate: bool = True  # listed under YAML providers (or manual)

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, v: str) -> str:
        return v.rstrip("/")

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        key = data.get("api_key") or ""
        data["api_key"] = ("*" * max(0, len(key) - 4)) + key[-4:] if key else ""
        return data


class DatasetRow(BaseModel):
    id: str
    question: str
    reference_answer: str | None = None
    gold_score: float | None = None
    rubric: str | None = None
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, v: Any) -> list[str]:
        if v is None or (isinstance(v, float) and str(v) == "nan"):
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        text = str(v).strip()
        if not text:
            return []
        return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]

    @field_validator("weight", mode="before")
    @classmethod
    def default_weight(cls, v: Any) -> float:
        if v is None or v == "" or (isinstance(v, float) and str(v) == "nan"):
            return 1.0
        return float(v)


class CalibrationSeverity(str, Enum):
    info = "info"
    warning = "warning"
    error = "error"


class CalibrationIssue(BaseModel):
    code: str
    severity: CalibrationSeverity
    message: str
    count: int | None = None
    examples: list[str] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    n_rows: int
    n_valid: int
    n_dropped: int
    score_present: int
    answer_present: int
    score_min: float | None = None
    score_max: float | None = None
    score_mean: float | None = None
    score_std: float | None = None
    score_bins: dict[str, int] = Field(default_factory=dict)
    tag_counts: dict[str, int] = Field(default_factory=dict)
    duplicate_ids: int = 0
    duplicate_questions: int = 0
    weight_sum: float = 0.0
    normalized_weights: bool = False
    ready_to_run: bool = False
    issues: list[CalibrationIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == CalibrationSeverity.error)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == CalibrationSeverity.warning)


class DatasetBundle(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str
    source_filename: str
    created_at: datetime = Field(default_factory=utc_now)
    rows: list[DatasetRow]
    calibration: CalibrationReport
    content_hash: str


class ScorerKind(str, Enum):
    score_mae = "score_mae"
    score_rmse = "score_rmse"
    exact_match = "exact_match"
    token_f1 = "token_f1"
    embedding_cosine = "embedding_cosine"
    llm_judge = "llm_judge"


class RunMode(str, Enum):
    """How the candidate model is prompted."""

    score_only = "score_only"  # model returns a numeric score
    answer_only = "answer_only"  # model returns text answer
    answer_and_score = "answer_and_score"  # JSON with answer + score


class RunRequest(BaseModel):
    dataset_id: str
    model_ids: list[str]
    mode: RunMode = RunMode.score_only
    scorers: list[ScorerKind] = Field(
        default_factory=lambda: [ScorerKind.score_mae, ScorerKind.score_rmse]
    )
    concurrency: int | None = None
    limit: int | None = None
    judge_model_id: str | None = None
    prompt_template: str | None = None


class RowPrediction(BaseModel):
    row_id: str
    model_id: str
    model_name: str
    raw_output: str = ""
    pred_answer: str | None = None
    pred_score: float | None = None
    gold_score: float | None = None
    reference_answer: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    score_100: float | None = None
    traffic_light: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    tags: list[str] = Field(default_factory=list)
    weight: float = 1.0


class ModelSummary(BaseModel):
    model_id: str
    model_name: str
    n_total: int
    n_ok: int
    n_error: int
    mae: float | None = None
    rmse: float | None = None
    spearman: float | None = None
    pearson: float | None = None
    exact_match: float | None = None
    token_f1: float | None = None
    embedding_cosine: float | None = None
    llm_judge: float | None = None
    score_100: float | None = None
    traffic_light: str | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    error_rate: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_prompt_tokens: float | None = None
    avg_completion_tokens: float | None = None
    avg_total_tokens: float | None = None
    # Quality efficiency: score_100 points per 1k total tokens (higher = better)
    score_per_1k_tokens: float | None = None


class TagBreakdown(BaseModel):
    tag: str
    model_id: str
    model_name: str
    n: int
    mae: float | None = None
    token_f1: float | None = None
    exact_match: float | None = None
    score_100: float | None = None
    traffic_light: str | None = None
    total_tokens: int = 0
    avg_completion_tokens: float | None = None
    score_per_1k_tokens: float | None = None


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class JobRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    status: JobStatus = JobStatus.queued
    request: RunRequest
    dataset_id: str
    dataset_name: str = ""
    progress_done: int = 0
    progress_total: int = 0
    progress_label: str = ""
    run_id: str | None = None
    error: str | None = None
    html_path: str | None = None


class BenchRun(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = Field(default_factory=utc_now)
    dataset_id: str
    dataset_name: str
    mode: RunMode
    scorers: list[ScorerKind]
    model_ids: list[str]
    predictions: list[RowPrediction] = Field(default_factory=list)
    summaries: list[ModelSummary] = Field(default_factory=list)
    tag_breakdown: list[TagBreakdown] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sort_summaries(self) -> BenchRun:
        # Prefer higher score_100; fall back to lower MAE
        self.summaries = sorted(
            self.summaries,
            key=lambda s: (
                s.score_100 is None,
                -(s.score_100 if s.score_100 is not None else -1.0),
                s.mae is None,
                s.mae if s.mae is not None else 1e9,
            ),
        )
        return self
