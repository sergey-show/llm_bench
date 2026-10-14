from __future__ import annotations

import json
from pathlib import Path

from llm_bench.schemas import BenchRun, JobRecord, ModelConfig
from llm_bench.secrets import decrypt_secret, encrypt_secret, mask_secret
from llm_bench.settings import Settings, settings as default_settings

_yaml_sync_mtime: float | None = None


def resolve_models_yaml(cfg: Settings) -> Path:
    path = cfg.models_yaml
    if path.is_absolute():
        return path
    # Prefer CWD, then package project root
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    root = Path(__file__).resolve().parents[1]
    return (root / path).resolve()


def maybe_sync_models_yaml(cfg: Settings | None = None, *, force: bool = False) -> bool:
    """Sync configs/models.yaml into the encrypted store when file changes."""
    global _yaml_sync_mtime
    cfg = cfg or default_settings
    if not cfg.auto_sync_models_yaml and not force:
        return False
    path = resolve_models_yaml(cfg)
    if not path.exists():
        return False
    mtime = path.stat().st_mtime
    if not force and _yaml_sync_mtime is not None and mtime <= _yaml_sync_mtime:
        return False
    from llm_bench.models_yaml import sync_models_from_yaml

    sync_models_from_yaml(path, cfg=cfg, replace_yaml_sourced=True)
    _yaml_sync_mtime = mtime
    return True


def _load_models_raw(cfg: Settings) -> list[ModelConfig]:
    path = cfg.models_path
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    models: list[ModelConfig] = []
    for item in raw:
        model = ModelConfig.model_validate(item)
        model.api_key = decrypt_secret(model.api_key, cfg)
        models.append(model)
    return models


def _save_models(models: list[ModelConfig], cfg: Settings) -> None:
    cfg.ensure_dirs()
    payload = []
    for m in models:
        data = m.model_dump(mode="json")
        data["api_key"] = encrypt_secret(m.api_key, cfg)
        payload.append(data)
    cfg.models_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def list_models(cfg: Settings | None = None, *, sync_yaml: bool = True) -> list[ModelConfig]:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    if sync_yaml:
        try:
            maybe_sync_models_yaml(cfg)
        except Exception:
            # Keep previously stored models if YAML sync fails (e.g. missing API key)
            pass
    return _load_models_raw(cfg)


def list_models_public(cfg: Settings | None = None) -> list[dict]:
    return [
        {**m.model_dump(), "api_key": mask_secret(m.api_key)}
        for m in list_models(cfg)
    ]


def upsert_model(model: ModelConfig, cfg: Settings | None = None) -> ModelConfig:
    cfg = cfg or default_settings
    models = _load_models_raw(cfg)
    for i, existing in enumerate(models):
        if existing.id == model.id or (
            existing.name == model.name and existing.model_id == model.model_id
        ):
            # Keep previous key if UI sent masked placeholder
            if model.api_key.startswith("*") and existing.api_key:
                model.api_key = existing.api_key
            model.id = existing.id
            models[i] = model
            _save_models(models, cfg)
            return model
    models.append(model)
    _save_models(models, cfg)
    return model


def delete_model(model_id: str, cfg: Settings | None = None) -> bool:
    cfg = cfg or default_settings
    models = _load_models_raw(cfg)
    new_models = [m for m in models if m.id != model_id]
    if len(new_models) == len(models):
        return False
    _save_models(new_models, cfg)
    return True


def get_models_by_ids(model_ids: list[str], cfg: Settings | None = None) -> list[ModelConfig]:
    cfg = cfg or default_settings
    by_id = {m.id: m for m in _load_models_raw(cfg)}
    missing = [mid for mid in model_ids if mid not in by_id]
    if missing:
        raise KeyError(f"Unknown model ids: {', '.join(missing)}")
    return [by_id[mid] for mid in model_ids]


def save_run(run: BenchRun, cfg: Settings | None = None) -> Path:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    path = cfg.runs_dir / f"{run.id}.json"
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run(run_id: str, cfg: Settings | None = None) -> BenchRun:
    cfg = cfg or default_settings
    path = cfg.runs_dir / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return BenchRun.model_validate_json(path.read_text(encoding="utf-8"))


def list_runs(cfg: Settings | None = None) -> list[BenchRun]:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    runs: list[BenchRun] = []
    for path in sorted(cfg.runs_dir.glob("*.json"), reverse=True):
        try:
            runs.append(BenchRun.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return runs


def save_job(job: JobRecord, cfg: Settings | None = None) -> Path:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    path = cfg.jobs_dir / f"{job.id}.json"
    path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_job(job_id: str, cfg: Settings | None = None) -> JobRecord:
    cfg = cfg or default_settings
    path = cfg.jobs_dir / f"{job_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Job not found: {job_id}")
    return JobRecord.model_validate_json(path.read_text(encoding="utf-8"))


def list_jobs(cfg: Settings | None = None) -> list[JobRecord]:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    jobs: list[JobRecord] = []
    for path in sorted(cfg.jobs_dir.glob("*.json"), reverse=True):
        try:
            jobs.append(JobRecord.model_validate_json(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    return jobs
