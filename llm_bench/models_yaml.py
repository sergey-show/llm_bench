from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from llm_bench.schemas import ModelConfig
from llm_bench.settings import Settings, settings as default_settings

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class YamlModelEntry(BaseModel):
    id: str
    name: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    concurrency: int | None = None
    system_prompt: str | None = None
    enabled: bool = True


class YamlProvider(BaseModel):
    name: str
    base_url: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 1024
    concurrency: int = 4
    system_prompt: str | None = None
    models: list[YamlModelEntry | str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class ModelCatalog(BaseModel):
    kind: str = "ModelCatalog"
    providers: list[YamlProvider] = Field(default_factory=list)
    judges: list[YamlProvider] = Field(default_factory=list)
    # Optional default judge model id (OpenAI model string), e.g. nvidia/nemotron-...
    default_judge: str | None = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_models_yaml_path() -> Path:
    return _project_root() / "configs" / "models.yaml"


def _load_dotenv_files() -> None:
    root = _project_root()
    for name in (".env", ".env "):
        path = root / name
        if path.exists():
            load_dotenv(path, override=False)


def expand_env(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        env_val = os.environ.get(key)
        if env_val is None or env_val == "":
            raise ValueError(f"Environment variable {key} is not set (needed by models.yaml)")
        return env_val

    return _ENV_PATTERN.sub(repl, value)


def _normalize_provider(raw: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw)
    if "name" not in data and "provider" in data:
        data["name"] = data.pop("provider")
    if "base_url" not in data and "host" in data:
        data["base_url"] = data.pop("host")
    models = []
    for item in data.get("models") or []:
        if isinstance(item, str):
            models.append({"id": item})
        elif isinstance(item, dict):
            if "id" not in item and "model_id" in item:
                item = {**item, "id": item["model_id"]}
            models.append(item)
        else:
            raise ValueError(f"Invalid model entry: {item!r}")
    data["models"] = models
    return data


def load_model_catalog(path: Path | None = None) -> ModelCatalog:
    _load_dotenv_files()
    path = path or default_models_yaml_path()
    if not path.exists():
        raise FileNotFoundError(f"models.yaml not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict) and "providers" not in raw and "judges" not in raw:
        if "provider" in raw or "name" in raw or "host" in raw or "base_url" in raw:
            raw = {"kind": raw.get("kind", "ModelCatalog"), "providers": [raw]}
    providers = [_normalize_provider(p) for p in (raw.get("providers") or [])]
    judges = [_normalize_provider(p) for p in (raw.get("judges") or [])]
    return ModelCatalog.model_validate(
        {
            "kind": raw.get("kind", "ModelCatalog"),
            "providers": providers,
            "judges": judges,
            "default_judge": raw.get("default_judge"),
        }
    )


def _provider_models_to_configs(
    providers: list[YamlProvider],
    *,
    expand_secrets: bool,
    is_judge: bool,
    is_candidate: bool,
) -> list[ModelConfig]:
    out: list[ModelConfig] = []
    for provider in providers:
        api_key = expand_env(provider.api_key) if expand_secrets else provider.api_key
        base_url = expand_env(provider.base_url) if expand_secrets else provider.base_url
        for entry in provider.models:
            if isinstance(entry, str):
                entry = YamlModelEntry(id=entry)
            display = entry.name or f"{provider.name}/{entry.id}"
            stable = f"{provider.name}:{entry.id}"
            mid = _stable_id(stable)
            out.append(
                ModelConfig(
                    id=mid,
                    name=display,
                    base_url=base_url.rstrip("/"),
                    api_key=api_key,
                    model_id=entry.id,
                    temperature=(
                        entry.temperature if entry.temperature is not None else provider.temperature
                    ),
                    max_tokens=(
                        entry.max_tokens if entry.max_tokens is not None else provider.max_tokens
                    ),
                    concurrency=(
                        entry.concurrency if entry.concurrency is not None else provider.concurrency
                    ),
                    enabled=entry.enabled,
                    system_prompt=entry.system_prompt or provider.system_prompt,
                    provider=provider.name,
                    source="yaml",
                    is_judge=is_judge,
                    is_candidate=is_candidate,
                )
            )
    return out


def catalog_to_model_configs(
    catalog: ModelCatalog,
    *,
    expand_secrets: bool = True,
) -> list[ModelConfig]:
    """Merge providers + judges; same provider:model keeps one id and is_judge=True if listed under judges."""
    by_id: dict[str, ModelConfig] = {}
    for model in _provider_models_to_configs(
        catalog.providers,
        expand_secrets=expand_secrets,
        is_judge=False,
        is_candidate=True,
    ):
        by_id[model.id] = model
    for model in _provider_models_to_configs(
        catalog.judges,
        expand_secrets=expand_secrets,
        is_judge=True,
        is_candidate=False,
    ):
        if model.id in by_id:
            existing = by_id[model.id]
            by_id[model.id] = existing.model_copy(update={"is_judge": True})
        else:
            by_id[model.id] = model  # judge-only
    return list(by_id.values())


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def default_judge_model_id(catalog: ModelCatalog, models: list[ModelConfig]) -> str | None:
    """Resolve default judge: explicit default_judge, else first is_judge model."""
    if catalog.default_judge:
        for m in models:
            if m.model_id == catalog.default_judge and m.is_judge:
                return m.id
        for m in models:
            if m.model_id == catalog.default_judge:
                return m.id
    for m in models:
        if m.is_judge:
            return m.id
    return None


def sync_models_from_yaml(
    path: Path | None = None,
    *,
    cfg: Settings | None = None,
    replace_yaml_sourced: bool = True,
) -> list[ModelConfig]:
    """Load configs/models.yaml and upsert into encrypted models store."""
    from llm_bench.storage import list_models, upsert_model, _save_models

    cfg = cfg or default_settings
    catalog = load_model_catalog(path)
    yaml_models = catalog_to_model_configs(catalog, expand_secrets=True)

    existing = list_models(cfg, sync_yaml=False)
    if replace_yaml_sourced:
        kept = [m for m in existing if getattr(m, "source", None) != "yaml"]
        merged = kept + yaml_models
        _save_models(merged, cfg)
        return merged

    for model in yaml_models:
        upsert_model(model, cfg)
    return list_models(cfg, sync_yaml=False)


def list_providers_from_yaml(path: Path | None = None) -> list[dict[str, Any]]:
    catalog = load_model_catalog(path)
    return [
        {
            "name": p.name,
            "base_url": p.base_url,
            "model_count": len(p.models),
            "models": [(m if isinstance(m, str) else m.id) for m in p.models],
            "kind": "provider",
        }
        for p in catalog.providers
    ]


def list_judges_from_yaml(path: Path | None = None) -> list[dict[str, Any]]:
    catalog = load_model_catalog(path)
    models = catalog_to_model_configs(catalog, expand_secrets=False)
    default_id = default_judge_model_id(catalog, models)
    return [
        {
            "id": m.id,
            "name": m.name,
            "provider": m.provider,
            "model_id": m.model_id,
            "base_url": m.base_url,
            "is_default": m.id == default_id,
        }
        for m in models
        if m.is_judge
    ]


def catalog_summary(path: Path | None = None) -> dict[str, Any]:
    catalog = load_model_catalog(path)
    models = catalog_to_model_configs(catalog, expand_secrets=False)
    return {
        "providers": list_providers_from_yaml(path),
        "judges": list_judges_from_yaml(path),
        "default_judge_model_id": catalog.default_judge,
        "default_judge_id": default_judge_model_id(catalog, models),
    }
