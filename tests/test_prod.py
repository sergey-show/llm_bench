from llm_bench.metrics import summarize_by_tag
from llm_bench.schemas import RowPrediction
from llm_bench.secrets import decrypt_secret, encrypt_secret
from llm_bench.settings import Settings
from llm_bench.storage import list_models_public, upsert_model
from llm_bench.schemas import ModelConfig


def test_encrypt_roundtrip(tmp_path):
    cfg = Settings(data_dir=tmp_path, secret_key="unit-test-secret")
    token = encrypt_secret("sk-secret-key", cfg)
    assert token.startswith("enc:v1:")
    assert decrypt_secret(token, cfg) == "sk-secret-key"


def test_model_keys_encrypted_at_rest(tmp_path):
    cfg = Settings(data_dir=tmp_path, secret_key="unit-test-secret-2")
    cfg.ensure_dirs()
    upsert_model(
        ModelConfig(
            name="m1",
            base_url="http://localhost/v1",
            api_key="sk-abc12345",
            model_id="model-x",
        ),
        cfg,
    )
    raw = cfg.models_path.read_text(encoding="utf-8")
    assert "sk-abc12345" not in raw
    assert "enc:v1:" in raw
    public = list_models_public(cfg)
    assert public[0]["api_key"].endswith("2345")
    assert "*" in public[0]["api_key"]


def test_tag_breakdown():
    preds = [
        RowPrediction(
            row_id="1",
            model_id="m",
            model_name="M",
            metrics={"abs_error": 1.0},
            tags=["math"],
            weight=1.0,
        ),
        RowPrediction(
            row_id="2",
            model_id="m",
            model_name="M",
            metrics={"abs_error": 3.0},
            tags=["math", "hard"],
            weight=1.0,
        ),
    ]
    rows = summarize_by_tag(preds)
    by_tag = {r.tag: r.mae for r in rows}
    assert by_tag["math"] == 2.0
    assert by_tag["hard"] == 3.0
