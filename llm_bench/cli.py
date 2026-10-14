from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_bench.dataset_io import parse_dataset_file, save_dataset_bundle
from llm_bench.reports.html_report import write_html_report
from llm_bench.runner import run_benchmark_sync
from llm_bench.schemas import ModelConfig, RunMode, RunRequest, ScorerKind
from llm_bench.settings import settings
from llm_bench.storage import upsert_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="llm-bench", description="LLM comparison bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser("calibrate", help="Parse + calibrate a dataset file")
    p_cal.add_argument("path", type=Path)
    p_cal.add_argument("--name", default=None)
    p_cal.add_argument("--no-normalize-weights", action="store_true")

    p_model = sub.add_parser("add-model", help="Register an OpenAI-compatible model")
    p_model.add_argument("--name", required=True)
    p_model.add_argument("--base-url", required=True)
    p_model.add_argument("--api-key", required=True)
    p_model.add_argument("--model-id", required=True)
    p_model.add_argument("--temperature", type=float, default=0.0)
    p_model.add_argument("--concurrency", type=int, default=4)

    p_run = sub.add_parser("run", help="Run benchmark")
    p_run.add_argument("--dataset-id", required=True)
    p_run.add_argument("--model-id", action="append", dest="model_ids", required=True)
    p_run.add_argument("--mode", default=RunMode.score_only.value, choices=[m.value for m in RunMode])
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--concurrency", type=int, default=None)

    sub.add_parser("ui", help="Launch Streamlit UI")
    sub.add_parser("sync-models", help="Sync models from configs/models.yaml")

    p_serve = sub.add_parser("serve", help="Launch production FastAPI")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)

    args = parser.parse_args()
    settings.ensure_dirs()

    if args.cmd == "calibrate":
        bundle = parse_dataset_file(
            args.path,
            args.path.name,
            name=args.name,
            normalize_weights=not args.no_normalize_weights,
        )
        path = save_dataset_bundle(bundle)
        print(json.dumps({
            "dataset_id": bundle.id,
            "path": str(path),
            "ready_to_run": bundle.calibration.ready_to_run,
            "n_valid": bundle.calibration.n_valid,
            "issues": [i.model_dump() for i in bundle.calibration.issues],
            "suggestions": bundle.calibration.suggestions,
        }, ensure_ascii=False, indent=2))
        return

    if args.cmd == "add-model":
        model = ModelConfig(
            name=args.name,
            base_url=args.base_url,
            api_key=args.api_key,
            model_id=args.model_id,
            temperature=args.temperature,
            concurrency=args.concurrency,
        )
        upsert_model(model)
        print(json.dumps(model.public_dict(), ensure_ascii=False, indent=2))
        return

    if args.cmd == "run":
        from llm_bench.dataset_io import load_dataset_bundle
        from llm_bench.storage import get_models_by_ids

        dataset = load_dataset_bundle(args.dataset_id)
        models = get_models_by_ids(args.model_ids)
        req = RunRequest(
            dataset_id=dataset.id,
            model_ids=args.model_ids,
            mode=RunMode(args.mode),
            scorers=[ScorerKind.score_mae, ScorerKind.score_rmse, ScorerKind.token_f1, ScorerKind.exact_match],
            concurrency=args.concurrency,
            limit=args.limit,
        )
        run = run_benchmark_sync(dataset=dataset, models=models, request=req)
        html = write_html_report(run)
        print(json.dumps({
            "run_id": run.id,
            "html": str(html),
            "summaries": [s.model_dump() for s in run.summaries],
        }, ensure_ascii=False, indent=2))
        return

    if args.cmd == "sync-models":
        from llm_bench.storage import list_models_public, maybe_sync_models_yaml, resolve_models_yaml

        path = resolve_models_yaml(settings)
        maybe_sync_models_yaml(settings, force=True)
        models = list_models_public()
        print(json.dumps({
            "path": str(path),
            "count": len(models),
            "models": [{"id": m["id"], "name": m["name"], "provider": m.get("provider"), "model_id": m["model_id"]} for m in models],
        }, ensure_ascii=False, indent=2))
        return

    if args.cmd == "ui":
        import subprocess
        import sys

        app = Path(__file__).with_name("ui_app.py")
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run(
            "llm_bench.api.app:app",
            host=args.host or settings.api_host,
            port=args.port or settings.api_port,
            reload=False,
        )


if __name__ == "__main__":
    main()
