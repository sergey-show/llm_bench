from __future__ import annotations

from llm_bench.schemas import DatasetRow, RunMode

SCORE_ONLY_TEMPLATE = """\
You are a careful grader. Rate the following item.
Return ONLY a number between {score_min} and {score_max}.

Question:
{question}

{rubric_block}
Number only:"""

ANSWER_ONLY_TEMPLATE = """\
Answer the question concisely and correctly.

Question:
{question}

Answer:"""

ANSWER_AND_SCORE_TEMPLATE = """\
Answer the question, then provide a quality/relevance score.
Return ONLY valid JSON with keys "answer" (string) and "score" (number between {score_min} and {score_max}).

Question:
{question}

{rubric_block}
JSON:"""

JUDGE_TEMPLATE = """\
You are an impartial judge. Compare the candidate answer to the reference.
Return ONLY a number between {score_min} and {score_max}.

Question:
{question}

Reference answer:
{reference_answer}

Candidate answer:
{candidate_answer}

{rubric_block}
Score only:"""


def _rubric_block(row: DatasetRow) -> str:
    if row.rubric and str(row.rubric).strip():
        return f"Rubric:\n{row.rubric.strip()}\n"
    return ""


def build_prompt(
    row: DatasetRow,
    mode: RunMode,
    *,
    score_min: float = 0.0,
    score_max: float = 5.0,
    template: str | None = None,
) -> str:
    if template:
        return template.format(
            question=row.question,
            reference_answer=row.reference_answer or "",
            rubric=row.rubric or "",
            rubric_block=_rubric_block(row),
            gold_score=row.gold_score if row.gold_score is not None else "",
            score_min=score_min,
            score_max=score_max,
            tags=", ".join(row.tags),
        )

    kwargs = dict(
        question=row.question,
        rubric_block=_rubric_block(row),
        score_min=score_min,
        score_max=score_max,
    )
    if mode == RunMode.score_only:
        return SCORE_ONLY_TEMPLATE.format(**kwargs)
    if mode == RunMode.answer_only:
        return ANSWER_ONLY_TEMPLATE.format(question=row.question)
    return ANSWER_AND_SCORE_TEMPLATE.format(**kwargs)


def build_judge_prompt(
    row: DatasetRow,
    candidate_answer: str,
    *,
    score_min: float = 0.0,
    score_max: float = 5.0,
) -> str:
    return JUDGE_TEMPLATE.format(
        question=row.question,
        reference_answer=row.reference_answer or "",
        candidate_answer=candidate_answer,
        rubric_block=_rubric_block(row),
        score_min=score_min,
        score_max=score_max,
    )
