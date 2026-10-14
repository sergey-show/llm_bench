from pathlib import Path

from llm_bench.models_yaml import (
    catalog_to_model_configs,
    default_judge_model_id,
    list_judges_from_yaml,
    load_model_catalog,
)


def test_load_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OR_KEY", "sk-test-key")
    path = tmp_path / "models.yaml"
    path.write_text(
        """
kind: ModelCatalog
providers:
  - name: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${TEST_OR_KEY}
    models:
      - id: google/gemma-test
        name: gemma-test
      - other/model:free
""",
        encoding="utf-8",
    )
    catalog = load_model_catalog(path)
    models = catalog_to_model_configs(catalog)
    assert len(models) == 2
    assert models[0].provider == "openrouter"
    assert models[0].source == "yaml"
    assert models[0].api_key == "sk-test-key"
    assert models[1].model_id == "other/model:free"


def test_judges_section(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_OR_KEY", "sk-judge")
    path = tmp_path / "models.yaml"
    path.write_text(
        """
kind: ModelCatalog
default_judge: judge/model-a
judges:
  - name: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${TEST_OR_KEY}
    models:
      - id: judge/model-a
        name: judge-a
providers:
  - name: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${TEST_OR_KEY}
    models:
      - id: judge/model-a
        name: judge-a
      - id: cand/model-b
        name: cand-b
""",
        encoding="utf-8",
    )
    catalog = load_model_catalog(path)
    models = catalog_to_model_configs(catalog)
    assert len(models) == 2
    by_mid = {m.model_id: m for m in models}
    assert by_mid["judge/model-a"].is_judge is True
    assert by_mid["judge/model-a"].is_candidate is True
    assert by_mid["cand/model-b"].is_judge is False
    assert default_judge_model_id(catalog, models) == by_mid["judge/model-a"].id
    judges = list_judges_from_yaml(path)
    assert len(judges) == 1
    assert judges[0]["is_default"] is True


def test_project_models_yaml_exists():
    path = Path("configs/models.yaml")
    assert path.exists()
    catalog = load_model_catalog(path)
    assert catalog.judges, "expected judges section in project models.yaml"
