FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LLM_BENCH_DATA_DIR=/data

COPY pyproject.toml README.md ./
COPY llm_bench ./llm_bench
COPY data/datasets/sample_qa_scores.csv ./data/datasets/sample_qa_scores.csv

RUN pip install --no-cache-dir -e . \
    && mkdir -p /data/datasets /data/runs /data/jobs

VOLUME ["/data"]
EXPOSE 8000 8501

CMD ["llm-bench", "serve", "--host", "0.0.0.0", "--port", "8000"]
