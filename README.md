# LLM Bench

Compare OpenAI-compatible LLMs on a labeled dataset: `question → answer → score`.


## Features

- CSV/JSONL upload with **auto-calibration** (`ready_to_run` gate)
- Models via `base_url` + API key + `model_id` (keys **encrypted at rest**)
- Parallel runs + **background jobs** with progress
- Metrics: MAE, RMSE, Spearman/Pearson, exact match, token F1, LLM-as-judge
- **Traffic light on a 0–100 score** (🟢 ≥80 / 🟡 ≥60 / 🔴 &lt;60; thresholds configurable)
- Per-tag MAE / score breakdown
- **Token usage** in reports: prompt / completion / total, averages, and **score per 1k tokens**
- HTML report + REST API

## Production (Docker)

```bash
cp .env.example .env
# set LLM_BENCH_SECRET_KEY and optionally LLM_BENCH_API_TOKEN

docker compose up --build
```

- API: http://localhost:8000/docs  
- UI: http://localhost:8501  

The UI talks to the API via `LLM_BENCH_API_URL=http://api:8000`.

## Model catalog (`configs/models.yaml`)

The UI and API load **providers** (candidates) and **judges** (LLM-as-judge) from YAML:

```yaml
kind: ModelCatalog
default_judge: nvidia/nemotron-3-ultra-550b-a55b:free
judges:
  - name: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    models:
      - id: nvidia/nemotron-3-ultra-550b-a55b:free
        name: nemotron-3-ultra-free
providers:
  - name: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}
    models:
      - id: google/gemma-4-26b-a4b-it:free
        name: gemma-4-26b-a4b-it:free
```

In **UI → Run**, the **Judge** block comes from `judges:` (default from `default_judge`).  
API: `GET /v1/judges`, sync: `POST /v1/models/sync-yaml`.

```bash
llm-bench sync-models
```

Auto-sync on model listing: `LLM_BENCH_AUTO_SYNC_MODELS_YAML=true`.

## Local development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# API
llm-bench serve

# UI (another terminal, against API)
export LLM_BENCH_API_URL=http://127.0.0.1:8000
streamlit run llm_bench/ui_app.py

# or UI without API (in-process)
unset LLM_BENCH_API_URL
streamlit run llm_bench/ui_app.py
```

Sample domain dataset (15 questions): `data/datasets/domains_15.csv` (medicine / industry / programming).

## API (overview)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/v1/datasets/upload` | Upload + calibrate |
| GET | `/v1/datasets` | List datasets |
| POST | `/v1/models` | Add model |
| GET | `/v1/models` | List models (keys masked) |
| GET | `/v1/judges` | Judges from YAML |
| POST | `/v1/models/sync-yaml` | Sync `models.yaml` |
| POST | `/v1/jobs` | Start benchmark |
| GET | `/v1/jobs/{id}` | Status / progress |
| GET | `/v1/runs/{id}` | Full result |
| GET | `/v1/runs/{id}/report.html` | HTML report |

If `LLM_BENCH_API_TOKEN` is set, send `Authorization: Bearer <token>`.

## Dataset format

| Column | Required | Description |
|--------|----------|-------------|
| `question` | yes | Question / prompt |
| `gold_score` | for MAE | Reference numeric score |
| `reference_answer` | for text metrics | Reference answer |
| `tags` | no | `,` / `;` separated |
| `weight` | no | Normalized on upload |
| `rubric` | no | Grading criteria |

Example: `data/datasets/sample_qa_scores.csv`.

## Calibration

On upload the system validates schema, duplicates, score distribution, and weights, then sets `ready_to_run`. Runs are blocked when it is `false`.

## How MAE is computed

Per row:

`abs_error = |pred_score − gold_score|`

Model MAE is the weighted mean of `abs_error` values.

- `score_only` / `answer_and_score` — `pred_score` comes from the candidate model  
- `answer_only` + LLM judge — if there is no `pred_score`, MAE uses `|judge_score − gold_score|`

## Traffic light (0–100)

Composite `score_100` is built from available metrics (weights via env):

- from MAE: `100 * (1 - mae / (score_max - score_min))`
- from F1 / exact: `× 100`
- from LLM-judge: scaled to 0–100
- penalty for `error_rate`

| Color | Default rule |
|-------|----------------|
| 🟢 green | `score_100 ≥ 80` |
| 🟡 yellow | `score_100 ≥ 60` |
| 🔴 red | otherwise |

Thresholds: `LLM_BENCH_TRAFFIC_GREEN_MIN`, `LLM_BENCH_TRAFFIC_YELLOW_MIN`.

## Run modes

- `score_only` — numeric output → MAE / RMSE  
- `answer_only` — text → exact / F1 (optional judge + MAE vs gold)  
- `answer_and_score` — JSON `{answer, score}`

## Environment

See `.env.example`. Critical for production:

- `LLM_BENCH_SECRET_KEY` — encrypts API keys at rest  
- `LLM_BENCH_API_TOKEN` — optional Bearer auth  
- `LLM_BENCH_MAX_PARALLEL_JOBS` — concurrent benchmark jobs  

## Layout

```
llm_bench/
  api/app.py       # FastAPI
  jobs.py          # background jobs
  secrets.py       # Fernet encryption
  calibration.py
  runner.py
  ui_app.py        # Streamlit (API or local)
  reports/
configs/models.yaml
docker-compose.yml
```
