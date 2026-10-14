from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_BENCH_",
        extra="ignore",
    )

    data_dir: Path = Field(default=Path("./data"))
    default_concurrency: int = Field(default=8, ge=1, le=64)
    request_timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    score_min: float = 0.0
    score_max: float = 5.0

    # Traffic light (100-point score)
    traffic_green_min: float = Field(default=80.0, ge=0, le=100)
    traffic_yellow_min: float = Field(default=60.0, ge=0, le=100)
    traffic_weight_mae: float = Field(default=0.50, ge=0)
    traffic_weight_f1: float = Field(default=0.25, ge=0)
    traffic_weight_exact: float = Field(default=0.15, ge=0)
    traffic_weight_judge: float = Field(default=0.10, ge=0)

    # Prod
    secret_key: str = Field(
        default="dev-only-change-me-llm-bench-secret",
        description="Used to encrypt API keys at rest",
    )
    api_token: str | None = Field(
        default=None,
        description="If set, require Bearer token for API mutating/read endpoints except /health",
    )
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str | None = Field(
        default=None,
        description="UI uses this to talk to API; if empty UI uses in-process services",
    )
    max_parallel_jobs: int = Field(default=2, ge=1, le=16)
    cors_origins: str = Field(
        default="*",
        description="Comma-separated CORS origins for API",
    )
    models_yaml: Path = Field(
        default=Path("./configs/models.yaml"),
        description="Path to models catalog YAML used by UI/API sync",
    )
    auto_sync_models_yaml: bool = Field(
        default=True,
        description="If true, sync models.yaml into store on API/UI startup listing",
    )

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_path(self) -> Path:
        return self.data_dir / "models.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def ensure_dirs(self) -> None:
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
