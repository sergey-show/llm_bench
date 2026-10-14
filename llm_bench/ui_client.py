from __future__ import annotations

from typing import Any

import httpx

from llm_bench.settings import settings


class BenchApiClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (base_url or settings.api_url or "http://127.0.0.1:8000").rstrip("/")
        self.token = token if token is not None else settings.api_token

    def _headers(self) -> dict[str, str]:
        if self.token:
            return {"Authorization": f"Bearer {self.token}"}
        return {}

    def health(self) -> dict[str, Any]:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(f"{self.base_url}/health")
            r.raise_for_status()
            return r.json()

    def list_datasets(self) -> list[dict]:
        return self._get("/v1/datasets")

    def upload_dataset(self, filename: str, content: bytes, name: str | None = None) -> dict:
        files = {"file": (filename, content)}
        data = {"normalize_weights": "true"}
        if name:
            data["name"] = name
        with httpx.Client(timeout=120.0) as client:
            r = client.post(
                f"{self.base_url}/v1/datasets/upload",
                headers=self._headers(),
                files=files,
                data=data,
            )
            r.raise_for_status()
            return r.json()

    def list_models(self) -> list[dict]:
        return self._get("/v1/models")

    def create_model(self, payload: dict) -> dict:
        return self._post("/v1/models", payload)

    def sync_models_yaml(self) -> dict:
        return self._post("/v1/models/sync-yaml", {})

    def list_providers(self) -> list[dict]:
        return self._get("/v1/providers")

    def list_judges(self) -> dict:
        return self._get("/v1/judges")

    def delete_model(self, model_id: str) -> dict:
        with httpx.Client(timeout=30.0) as client:
            r = client.delete(f"{self.base_url}/v1/models/{model_id}", headers=self._headers())
            r.raise_for_status()
            return r.json()

    def create_job(self, payload: dict) -> dict:
        return self._post("/v1/jobs", payload)

    def get_job(self, job_id: str) -> dict:
        return self._get(f"/v1/jobs/{job_id}")

    def list_jobs(self) -> list[dict]:
        return self._get("/v1/jobs")

    def list_runs(self) -> list[dict]:
        return self._get("/v1/runs")

    def get_run(self, run_id: str) -> dict:
        return self._get(f"/v1/runs/{run_id}")

    def get_report_html(self, run_id: str) -> bytes:
        with httpx.Client(timeout=60.0) as client:
            r = client.get(
                f"{self.base_url}/v1/runs/{run_id}/report.html",
                headers=self._headers(),
            )
            r.raise_for_status()
            return r.content

    def _get(self, path: str) -> Any:
        with httpx.Client(timeout=60.0) as client:
            r = client.get(f"{self.base_url}{path}", headers=self._headers())
            r.raise_for_status()
            return r.json()

    def _post(self, path: str, payload: dict) -> Any:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            r.raise_for_status()
            return r.json()
