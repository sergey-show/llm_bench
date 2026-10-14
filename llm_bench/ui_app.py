from __future__ import annotations

import time
from io import BytesIO

import pandas as pd
import streamlit as st

from llm_bench.dataset_io import list_datasets, load_dataset_bundle, parse_dataset_file, save_dataset_bundle
from llm_bench.reports.html_report import render_html_report, write_html_report
from llm_bench.schemas import (
    CalibrationSeverity,
    ModelConfig,
    RunMode,
    RunRequest,
    ScorerKind,
)
from llm_bench.settings import settings
from llm_bench.storage import (
    delete_model,
    list_jobs,
    list_models,
    list_runs,
    load_job,
    load_run,
    upsert_model,
)
from llm_bench.traffic_light import TrafficLight
from llm_bench.ui_client import BenchApiClient

st.set_page_config(page_title="LLM Bench", page_icon="⚖️", layout="wide")
settings.ensure_dirs()


def use_api() -> bool:
    return bool(settings.api_url)


def traffic_label(light: str | None, score: float | None = None) -> str:
    try:
        tl = TrafficLight(light) if light else TrafficLight.unknown
    except ValueError:
        tl = TrafficLight.unknown
    score_part = f" {score:.1f}" if score is not None else ""
    return f"{tl.emoji}{score_part}"


def enrich_summaries_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "traffic_light" in df.columns or "score_100" in df.columns:
        df.insert(
            0,
            "светофор",
            [
                traffic_label(r.get("traffic_light"), r.get("score_100"))
                for r in rows
            ],
        )
    return df


def api() -> BenchApiClient:
    return BenchApiClient()


def _sev_color(sev: str | CalibrationSeverity) -> str:
    value = sev.value if isinstance(sev, CalibrationSeverity) else sev
    return {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(value, "⚪")


def page_dataset() -> None:
    st.header("Dataset")
    st.caption("Upload labeled data. Calibration runs automatically; runs require ready_to_run=true.")

    uploaded = st.file_uploader("CSV / JSONL / JSON", type=["csv", "jsonl", "json", "ndjson"])
    name = st.text_input("Dataset name", value="")
    normalize_weights = st.checkbox("Normalize weights to mean 1.0", value=True)

    if uploaded is not None and st.button("Upload & calibrate", type="primary"):
        try:
            raw = uploaded.getvalue()
            if use_api():
                result = api().upload_dataset(uploaded.name, raw, name=name.strip() or None)
                st.session_state["last_dataset_id"] = result["id"]
                st.session_state["calibration_view"] = result
                st.success(
                    f"Saved `{result['name']}` ({result['id']}) · ready={result['ready_to_run']}"
                )
            else:
                bundle = parse_dataset_file(
                    BytesIO(raw),
                    uploaded.name,
                    name=name.strip() or None,
                    normalize_weights=normalize_weights,
                )
                save_dataset_bundle(bundle)
                st.session_state["last_dataset_id"] = bundle.id
                st.session_state["calibration_view"] = {
                    "id": bundle.id,
                    "name": bundle.name,
                    "ready_to_run": bundle.calibration.ready_to_run,
                    "calibration": bundle.calibration.model_dump(),
                    "content_hash": bundle.content_hash,
                }
                st.success(f"Saved `{bundle.name}` ({bundle.id})")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

    # Listing
    if use_api():
        datasets = api().list_datasets()
        options = {
            f"{d['name']} ({d['id']}) · ready={d['ready_to_run']}": d for d in datasets
        }
    else:
        local = list_datasets()
        options = {
            f"{d.name} ({d.id}) · ready={d.calibration.ready_to_run}": {
                "id": d.id,
                "name": d.name,
                "ready_to_run": d.calibration.ready_to_run,
                "calibration": d.calibration.model_dump(),
                "content_hash": d.content_hash,
                "_bundle": d,
            }
            for d in local
        }

    if not options and "calibration_view" not in st.session_state:
        st.info(
            "No dataset yet. Columns: `question`, optional `reference_answer`, `gold_score`, `tags`, `weight`."
        )
        return

    selected = None
    if options:
        label = st.selectbox("Saved datasets", list(options.keys()))
        selected = options[label]
        st.session_state["last_dataset_id"] = selected["id"]
    view = selected or st.session_state.get("calibration_view")
    if not view:
        return

    cal = view["calibration"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", cal.get("n_valid"))
    c2.metric("Dropped", cal.get("n_dropped"))
    c3.metric("With score", cal.get("score_present"))
    c4.metric("With answer", cal.get("answer_present"))
    c5.metric("Ready", "✅" if cal.get("ready_to_run") else "❌")

    if cal.get("score_mean") is not None:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Score min", f"{cal['score_min']:.3f}")
        s2.metric("Score max", f"{cal['score_max']:.3f}")
        s3.metric("Score mean", f"{cal['score_mean']:.3f}")
        s4.metric("Score std", f"{cal['score_std']:.3f}")

    st.subheader("Calibration issues")
    issues = cal.get("issues") or []
    if not issues:
        st.write("No issues.")
    else:
        for issue in issues:
            st.write(
                f"{_sev_color(issue['severity'])} **{issue['code']}** — {issue['message']}"
                + (f" (n={issue['count']})" if issue.get("count") is not None else "")
            )

    if cal.get("suggestions"):
        st.subheader("Suggestions")
        for tip in cal["suggestions"]:
            st.write(f"- {tip}")

    if cal.get("score_bins"):
        st.subheader("Score bins")
        st.bar_chart(pd.Series(cal["score_bins"], name="count"))

    if not use_api() and selected and selected.get("_bundle"):
        st.subheader("Preview")
        st.dataframe(
            pd.DataFrame([r.model_dump() for r in selected["_bundle"].rows[:50]]),
            use_container_width=True,
        )


def page_models() -> None:
    st.header("Models")
    st.caption(
        f"Каталог: `{settings.models_yaml}` · ключи из `${{ENV}}` · encrypted at rest"
    )

    c1, c2 = st.columns(2)
    if c1.button("Sync from models.yaml", type="primary"):
        try:
            if use_api():
                result = api().sync_models_yaml()
                st.success(f"Synced {result.get('count')} models from `{result.get('path')}`")
            else:
                from llm_bench.storage import maybe_sync_models_yaml, resolve_models_yaml

                maybe_sync_models_yaml(settings, force=True)
                st.success(f"Synced from `{resolve_models_yaml(settings)}`")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    # Providers + judges from YAML
    try:
        if use_api():
            summary = api().list_judges()
            providers = summary.get("providers") or api().list_providers()
            judges = summary.get("judges") or []
        else:
            from llm_bench.models_yaml import catalog_summary
            from llm_bench.storage import resolve_models_yaml

            summary = catalog_summary(resolve_models_yaml(settings))
            providers = summary["providers"]
            judges = summary["judges"]
        st.subheader("Providers (from YAML)")
        for p in providers:
            with st.expander(f"**{p['name']}** · {p['model_count']} models · `{p['base_url']}`", expanded=True):
                st.write(", ".join(f"`{m}`" for m in p["models"]))
        st.subheader("Judges (from YAML)")
        if not judges:
            st.info("Секция `judges:` в models.yaml пуста — LLM-as-judge недоступен.")
        else:
            for j in judges:
                mark = " ← default" if j.get("is_default") else ""
                st.write(f"⚖️ **{j['name']}** `{j['model_id']}` · {j.get('provider')}{mark}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"YAML catalog unavailable: {exc}")

    with st.expander("Add manual model", expanded=False):
        with st.form("add_model"):
            name = st.text_input("Display name", placeholder="gpt-local")
            provider = st.text_input("Provider", placeholder="custom")
            base_url = st.text_input("Base URL", placeholder="http://localhost:8000/v1")
            api_key = st.text_input("API key", type="password", placeholder="sk-...")
            model_id = st.text_input("Model id", placeholder="Qwen2.5-7B-Instruct")
            c1, c2, c3 = st.columns(3)
            temperature = c1.number_input("Temperature", 0.0, 2.0, 0.0, 0.1)
            max_tokens = c2.number_input("Max tokens", 64, 8192, 1024, 64)
            concurrency = c3.number_input("Concurrency", 1, 32, 4, 1)
            system_prompt = st.text_area("System prompt (optional)", value="")
            submitted = st.form_submit_button("Save model", type="primary")
            if submitted:
                if not (name and base_url and api_key and model_id):
                    st.error("name, base_url, api_key, model_id are required")
                else:
                    payload = {
                        "name": name.strip(),
                        "base_url": base_url.strip(),
                        "api_key": api_key.strip(),
                        "model_id": model_id.strip(),
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                        "concurrency": int(concurrency),
                        "system_prompt": system_prompt.strip() or None,
                        "provider": provider.strip() or "manual",
                        "source": "manual",
                    }
                    try:
                        if use_api():
                            model = api().create_model(payload)
                        else:
                            model = upsert_model(ModelConfig.model_validate(payload)).public_dict()
                        st.success(f"Saved `{model['name']}` ({model['id']})")
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

    models = api().list_models() if use_api() else [m.public_dict() for m in list_models()]
    if not models:
        st.info("No models yet. Put providers into configs/models.yaml and click Sync.")
        return

    st.subheader("Available models")
    by_provider: dict[str, list[dict]] = {}
    for m in models:
        by_provider.setdefault(m.get("provider") or "unknown", []).append(m)
    for prov, items in by_provider.items():
        st.markdown(f"#### {prov}")
        for m in items:
            cols = st.columns([3, 3, 2, 1])
            src = m.get("source") or "—"
            roles = []
            if m.get("is_candidate", True):
                roles.append("candidate")
            if m.get("is_judge"):
                roles.append("judge")
            role = "+".join(roles) or "—"
            cols[0].write(f"**{m['name']}** `{m['model_id']}`")
            cols[1].write(m["base_url"])
            cols[2].write(f"{src} · {role} · key={m.get('api_key', '')}")
            if cols[3].button("Delete", key=f"del_{m['id']}"):
                if use_api():
                    api().delete_model(m["id"])
                else:
                    delete_model(m["id"])
                st.rerun()


def page_run() -> None:
    st.header("Run")
    if use_api():
        datasets = api().list_datasets()
        models = api().list_models()
    else:
        datasets = [
            {
                "id": d.id,
                "name": d.name,
                "ready_to_run": d.calibration.ready_to_run,
            }
            for d in list_datasets()
        ]
        models = [m.public_dict() for m in list_models()]

    if not datasets:
        st.warning("Upload a calibrated dataset first.")
        return
    if not models:
        st.warning("Add at least one model first.")
        return

    ds_map = {f"{d['name']} ({d['id']}) · ready={d['ready_to_run']}": d for d in datasets}
    ds = ds_map[st.selectbox("Dataset", list(ds_map.keys()))]
    if not ds["ready_to_run"]:
        st.error("Dataset failed calibration (ready_to_run=false).")
        return

    candidates = [m for m in models if m.get("is_candidate", True)]
    judge_models = [m for m in models if m.get("is_judge")]
    if not judge_models:
        # fallback: YAML judges summary
        try:
            if use_api():
                judge_models = api().list_judges().get("judges") or []
            else:
                from llm_bench.models_yaml import list_judges_from_yaml
                from llm_bench.storage import resolve_models_yaml

                judge_models = list_judges_from_yaml(resolve_models_yaml(settings))
        except Exception:
            judge_models = []

    st.write("Select candidate models (YAML `providers`):")
    selected_ids: list[str] = []
    by_provider: dict[str, list[dict]] = {}
    for m in candidates:
        by_provider.setdefault(m.get("provider") or "unknown", []).append(m)
    for prov, items in by_provider.items():
        st.markdown(f"**{prov}**")
        for m in items:
            label = f"{m['name']} (`{m['model_id']}`)"
            if st.checkbox(label, value=m.get("enabled", True), key=f"run_m_{m['id']}"):
                selected_ids.append(m["id"])

    mode = st.selectbox(
        "Run mode",
        options=[m.value for m in RunMode],
    )
    scorer_values = [s.value for s in ScorerKind if s != ScorerKind.embedding_cosine]
    default = ["score_mae", "score_rmse"]
    if mode != RunMode.score_only.value:
        default = default + ["token_f1", "exact_match"]
    if judge_models and "llm_judge" not in default:
        default = default + ["llm_judge"]
    scorers = st.multiselect("Scorers", options=scorer_values, default=default)

    judge_id = None
    st.subheader("Judge (YAML `judges`)")
    if not judge_models:
        st.caption("Добавьте секцию `judges:` в configs/models.yaml и нажмите Sync.")
    else:
        use_judge = st.checkbox(
            "Use LLM-as-judge",
            value="llm_judge" in scorers,
            help="Оценка ответов моделью-судьёй из YAML",
        )
        if use_judge:
            if "llm_judge" not in scorers:
                scorers = list(scorers) + ["llm_judge"]
            default_judge = next(
                (j for j in judge_models if j.get("is_default") or j.get("is_judge")),
                judge_models[0],
            )
            judge_labels = {
                f"{'⭐ ' if (j.get('is_default') or j.get('id') == default_judge.get('id')) else ''}"
                f"{j['name']} (`{j['model_id']}`)": j["id"]
                for j in judge_models
            }
            # put default first
            ordered = sorted(
                judge_labels.keys(),
                key=lambda k: (0 if judge_labels[k] == default_judge.get("id") else 1, k),
            )
            chosen = st.selectbox("Judge model", ordered)
            judge_id = judge_labels[chosen]
        elif "llm_judge" in scorers:
            scorers = [s for s in scorers if s != "llm_judge"]

    c1, c2 = st.columns(2)
    concurrency = c1.number_input("Global concurrency cap", 1, 64, settings.default_concurrency)
    limit = c2.number_input("Limit rows (0 = all)", 0, 100000, 0)

    if st.button("Start benchmark", type="primary", disabled=not selected_ids):
        req = {
            "dataset_id": ds["id"],
            "model_ids": selected_ids,
            "mode": mode,
            "scorers": scorers or ["score_mae"],
            "concurrency": int(concurrency),
            "limit": int(limit) or None,
            "judge_model_id": judge_id,
        }
        try:
            if use_api():
                job = api().create_job(req)
                st.session_state["active_job_id"] = job["id"]
                st.success(f"Job `{job['id']}` queued")
                st.rerun()
            else:
                from llm_bench.runner import run_benchmark_sync
                from llm_bench.schemas import JobRecord, JobStatus
                from llm_bench.storage import get_models_by_ids, save_job

                request = RunRequest.model_validate(req)
                dataset = load_dataset_bundle(request.dataset_id)
                models = get_models_by_ids(request.model_ids)
                judge = (
                    get_models_by_ids([request.judge_model_id])[0]
                    if request.judge_model_id
                    else None
                )
                with st.spinner("Running benchmark…"):
                    run = run_benchmark_sync(
                        dataset=dataset,
                        models=models,
                        request=request,
                        judge_model=judge,
                    )
                    html_path = write_html_report(run)
                job_obj = JobRecord(
                    request=request,
                    dataset_id=dataset.id,
                    dataset_name=dataset.name,
                    status=JobStatus.succeeded,
                    progress_done=len(run.predictions),
                    progress_total=len(run.predictions),
                    run_id=run.id,
                    html_path=str(html_path),
                )
                save_job(job_obj)
                job = job_obj.model_dump(mode="json")
                st.session_state["active_job_id"] = job["id"]
                st.session_state["finished_job"] = job
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return

    job_id = st.session_state.get("active_job_id")
    if not job_id:
        return

    if use_api():
        job = api().get_job(job_id)
        total = max(job.get("progress_total") or 1, 1)
        done = job.get("progress_done") or 0
        st.progress(min(done / total, 1.0), text=f"{done}/{total} · {job.get('progress_label', '')}")
        st.info(f"Status: **{job['status']}**")
        if job["status"] not in {"succeeded", "failed", "cancelled"}:
            time.sleep(1.0)
            st.rerun()
            return
    else:
        job = st.session_state.get("finished_job") or load_job(job_id).model_dump(mode="json")

    if job["status"] == "failed":
        st.error(job.get("error") or "Job failed")
        return

    if job["status"] != "succeeded":
        return

    st.success(f"Done. run_id=`{job.get('run_id')}`")
    if job.get("run_id"):
        run = api().get_run(job["run_id"]) if use_api() else load_run(job["run_id"]).model_dump(mode="json")
        summaries = run.get("summaries") or []
        if summaries:
            top = summaries[0]
            st.metric(
                "Лучший score",
                traffic_label(top.get("traffic_light"), top.get("score_100")),
                help=top.get("model_name"),
            )
        st.dataframe(enrich_summaries_df(summaries), use_container_width=True)
        if run.get("tag_breakdown"):
            st.subheader("By tag")
            st.dataframe(enrich_summaries_df(run["tag_breakdown"]), use_container_width=True)


def page_results() -> None:
    st.header("Results")
    runs = api().list_runs() if use_api() else [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat(),
            "dataset_name": r.dataset_name,
            "summaries": [s.model_dump() for s in r.summaries],
            "n_predictions": len(r.predictions),
        }
        for r in list_runs()
    ]
    jobs = api().list_jobs() if use_api() else [j.model_dump(mode="json") for j in list_jobs()]

    if jobs:
        with st.expander("Jobs", expanded=False):
            st.dataframe(pd.DataFrame(jobs), use_container_width=True)

    if not runs:
        st.info("No runs yet.")
        return

    labels = {
        f"{r['created_at']} · {r['dataset_name']} · {r['id']}": r["id"] for r in runs
    }
    run_id = labels[st.selectbox("Run", list(labels.keys()))]
    run = api().get_run(run_id) if use_api() else load_run(run_id).model_dump(mode="json")

    summaries = run.get("summaries") or []
    if summaries:
        cols = st.columns(min(3, len(summaries)))
        for col, s in zip(cols, summaries[:3]):
            col.metric(
                s.get("model_name", "model"),
                traffic_label(s.get("traffic_light"), s.get("score_100")),
            )
    st.subheader("Leaderboard")
    st.dataframe(enrich_summaries_df(summaries), use_container_width=True)
    if run.get("tag_breakdown"):
        st.subheader("By tag")
        st.dataframe(enrich_summaries_df(run["tag_breakdown"]), use_container_width=True)

    st.subheader("Predictions")
    preds = run.get("predictions") or []
    st.dataframe(enrich_summaries_df(preds), use_container_width=True)

    if use_api():
        html = api().get_report_html(run_id)
    else:
        from llm_bench.schemas import BenchRun

        html = render_html_report(BenchRun.model_validate(run)).encode("utf-8")
        write_html_report(BenchRun.model_validate(run))
    st.download_button(
        "Download HTML report",
        data=html,
        file_name=f"llm_bench_{run_id}.html",
        mime="text/html",
    )


def main() -> None:
    st.title("LLM Bench")
    mode = "API" if use_api() else "local"
    st.caption(f"Calibrate · compare · светофор 0–100 · mode=`{mode}`")
    st.sidebar.markdown(
        f"**Светофор**  \n"
        f"🟢 ≥ {settings.traffic_green_min:.0f}  \n"
        f"🟡 ≥ {settings.traffic_yellow_min:.0f}  \n"
        f"🔴 &lt; {settings.traffic_yellow_min:.0f}"
    )
    if use_api():
        try:
            h = api().health()
            st.sidebar.success(f"API ok · v{h.get('version')}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"API unreachable: {exc}")

    page = st.sidebar.radio("Pages", ["Dataset", "Models", "Run", "Results"], index=0)
    if page == "Dataset":
        page_dataset()
    elif page == "Models":
        page_models()
    elif page == "Run":
        page_run()
    else:
        page_results()


if __name__ == "__main__":
    main()
