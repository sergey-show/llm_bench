from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from llm_bench.calibration import calibrate_dataset
from llm_bench.schemas import DatasetBundle, DatasetRow
from llm_bench.settings import Settings, settings as default_settings

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "id": ("id", "uid", "example_id", "row_id"),
    "question": ("question", "query", "prompt", "input", "q"),
    "reference_answer": (
        "reference_answer",
        "answer",
        "gold_answer",
        "expected_answer",
        "target",
        "a",
    ),
    "gold_score": ("gold_score", "score", "label", "rating", "grade", "y"),
    "rubric": ("rubric", "criteria", "grading_notes"),
    "tags": ("tags", "tag", "category", "categories", "split"),
    "weight": ("weight", "w", "sample_weight"),
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c: str(c).strip().lower() for c in df.columns}
    rename: dict[str, str] = {}
    used_targets: set[str] = set()
    for target, aliases in COLUMN_ALIASES.items():
        for col, low in lower_map.items():
            if low in aliases and target not in used_targets:
                rename[col] = target
                used_targets.add(target)
                break
    return df.rename(columns=rename)


def _read_table(path_or_buf: Path | BinaryIO | bytes, filename: str) -> pd.DataFrame:
    from io import BytesIO

    buf: Path | BinaryIO
    if isinstance(path_or_buf, (bytes, bytearray)):
        buf = BytesIO(path_or_buf)
    else:
        buf = path_or_buf
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(buf)
    if name.endswith(".jsonl") or name.endswith(".ndjson"):
        return pd.read_json(buf, lines=True)
    if name.endswith(".json"):
        raw = pd.read_json(buf)
        if isinstance(raw, pd.DataFrame):
            return raw
        raise ValueError("JSON must be an array of objects")
    raise ValueError("Supported formats: .csv, .jsonl, .json")


def _row_from_series(idx: int, row: pd.Series) -> DatasetRow:
    data: dict[str, Any] = row.where(pd.notna(row), None).to_dict()
    rid = data.get("id")
    if rid is None or str(rid).strip() == "":
        rid = f"row_{idx + 1}"
    payload = {
        "id": str(rid),
        "question": data.get("question"),
        "reference_answer": data.get("reference_answer"),
        "gold_score": data.get("gold_score"),
        "rubric": data.get("rubric"),
        "tags": data.get("tags"),
        "weight": data.get("weight"),
    }
    known = set(payload)
    meta = {k: v for k, v in data.items() if k not in known and v is not None}
    payload["meta"] = meta
    return DatasetRow.model_validate(payload)


def parse_dataset_file(
    path_or_buf: Path | BinaryIO | bytes,
    filename: str,
    *,
    name: str | None = None,
    normalize_weights: bool = True,
    cfg: Settings | None = None,
) -> DatasetBundle:
    cfg = cfg or default_settings
    df = _read_table(path_or_buf, filename)
    df = _normalize_columns(df)

    if "question" not in df.columns:
        raise ValueError(
            "Missing required column 'question' "
            f"(aliases: {', '.join(COLUMN_ALIASES['question'])})"
        )

    rows = [_row_from_series(i, row) for i, row in df.iterrows()]
    calibrated, report = calibrate_dataset(rows, normalize_weights=normalize_weights, cfg=cfg)

    # Content hash over calibrated rows
    blob = json.dumps(
        [r.model_dump(mode="json") for r in calibrated],
        ensure_ascii=False,
        sort_keys=True,
    )
    content_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    return DatasetBundle(
        name=name or Path(filename).stem,
        source_filename=filename,
        rows=calibrated,
        calibration=report,
        content_hash=content_hash,
    )


def save_dataset_bundle(bundle: DatasetBundle, cfg: Settings | None = None) -> Path:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    path = cfg.datasets_dir / f"{bundle.id}.json"
    path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_dataset_bundle(dataset_id: str, cfg: Settings | None = None) -> DatasetBundle:
    cfg = cfg or default_settings
    path = cfg.datasets_dir / f"{dataset_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_id}")
    return DatasetBundle.model_validate_json(path.read_text(encoding="utf-8"))


def list_datasets(cfg: Settings | None = None) -> list[DatasetBundle]:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    out: list[DatasetBundle] = []
    for path in sorted(cfg.datasets_dir.glob("*.json")):
        try:
            out.append(DatasetBundle.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out
