from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from llm_bench import __version__
from llm_bench.dataset_io import list_datasets, load_dataset_bundle, parse_dataset_file, save_dataset_bundle
from llm_bench.jobs import get_job_manager
from llm_bench.reports.html_report import render_html_report
from llm_bench.schemas import ModelConfig, RunRequest
from llm_bench.settings import settings
from llm_bench.models_yaml import catalog_summary, list_providers_from_yaml
from llm_bench.storage import (
    delete_model,
    list_jobs,
    list_models_public,
    list_runs,
    load_run,
    maybe_sync_models_yaml,
    resolve_models_yaml,
    upsert_model,
)

settings.ensure_dirs()

app = FastAPI(
    title="LLM Bench API",
    version=__version__,
    description="Production API for calibrated LLM benchmarking",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_auth(authorization: Annotated[str | None, Header()] = None) -> None:
    token = settings.api_token
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if authorization.removeprefix("Bearer ").strip() != token:
        raise HTTPException(status_code=403, detail="Invalid token")


AuthDep = Depends(require_auth)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/datasets", dependencies=[AuthDep])
def get_datasets() -> list[dict]:
    return [
        {
            "id": d.id,
            "name": d.name,
            "n_valid": d.calibration.n_valid,
            "ready_to_run": d.calibration.ready_to_run,
            "content_hash": d.content_hash,
            "calibration": d.calibration.model_dump(mode="json"),
            "created_at": d.created_at.isoformat(),
        }
        for d in list_datasets()
    ]


@app.get("/v1/datasets/{dataset_id}", dependencies=[AuthDep])
def get_dataset(dataset_id: str) -> dict:
    try:
        d = load_dataset_bundle(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return d.model_dump(mode="json")


@app.post("/v1/datasets/upload", dependencies=[AuthDep])
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    normalize_weights: bool = Form(default=True),
) -> dict:
    raw = await file.read()
    try:
        bundle = parse_dataset_file(
            raw,
            file.filename or "upload.csv",
            name=name or None,
            normalize_weights=normalize_weights,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    save_dataset_bundle(bundle)
    return {
        "id": bundle.id,
        "name": bundle.name,
        "ready_to_run": bundle.calibration.ready_to_run,
        "calibration": bundle.calibration.model_dump(mode="json"),
        "content_hash": bundle.content_hash,
    }


@app.get("/v1/models", dependencies=[AuthDep])
def get_models() -> list[dict]:
    return list_models_public()


@app.get("/v1/providers", dependencies=[AuthDep])
def get_providers() -> list[dict]:
    try:
        return list_providers_from_yaml(resolve_models_yaml(settings))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/v1/judges", dependencies=[AuthDep])
def get_judges() -> dict:
    try:
        path = resolve_models_yaml(settings)
        return catalog_summary(path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/v1/models/sync-yaml", dependencies=[AuthDep])
def sync_models_yaml() -> dict:
    try:
        path = resolve_models_yaml(settings)
        synced = maybe_sync_models_yaml(settings, force=True)
        models = list_models_public()
        summary = catalog_summary(path)
        return {
            "synced": synced,
            "path": str(path),
            "count": len(models),
            "models": models,
            "judges": summary["judges"],
            "default_judge_id": summary["default_judge_id"],
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/models", dependencies=[AuthDep])
def create_model(payload: ModelConfig) -> dict:
    if not payload.source:
        payload.source = "manual"
    model = upsert_model(payload)
    public = model.model_dump()
    public["api_key"] = "***" + model.api_key[-4:] if model.api_key else ""
    return public


@app.delete("/v1/models/{model_id}", dependencies=[AuthDep])
def remove_model(model_id: str) -> dict:
    ok = delete_model(model_id)
    if not ok:
        raise HTTPException(404, "Model not found")
    return {"deleted": True, "id": model_id}


@app.post("/v1/jobs", dependencies=[AuthDep])
async def create_job(request: RunRequest) -> dict:
    mgr = get_job_manager()
    try:
        job = await mgr.enqueue(request)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return job.model_dump(mode="json")


@app.get("/v1/jobs", dependencies=[AuthDep])
def get_jobs() -> list[dict]:
    return [j.model_dump(mode="json") for j in list_jobs()]


@app.get("/v1/jobs/{job_id}", dependencies=[AuthDep])
def get_job(job_id: str) -> dict:
    try:
        return load_job_safe(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


def load_job_safe(job_id: str) -> dict:
    from llm_bench.storage import load_job

    return load_job(job_id).model_dump(mode="json")


@app.get("/v1/runs", dependencies=[AuthDep])
def get_runs() -> list[dict]:
    # Lightweight listing without huge prediction payloads
    out = []
    for r in list_runs():
        out.append(
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "dataset_id": r.dataset_id,
                "dataset_name": r.dataset_name,
                "mode": r.mode.value,
                "model_ids": r.model_ids,
                "summaries": [s.model_dump(mode="json") for s in r.summaries],
                "n_predictions": len(r.predictions),
            }
        )
    return out


@app.get("/v1/runs/{run_id}", dependencies=[AuthDep])
def get_run(run_id: str) -> dict:
    try:
        return load_run(run_id).model_dump(mode="json")
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/v1/runs/{run_id}/report.html", dependencies=[AuthDep])
def get_run_html(run_id: str) -> HTMLResponse:
    try:
        run = load_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return HTMLResponse(render_html_report(run))


