from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from llm_bench.dataset_io import load_dataset_bundle
from llm_bench.reports.html_report import write_html_report
from llm_bench.runner import run_benchmark
from llm_bench.schemas import JobRecord, JobStatus, RunRequest
from llm_bench.settings import Settings, settings as default_settings
from llm_bench.storage import get_models_by_ids, load_job, list_jobs, save_job


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    """In-process async job queue for production UI/API."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or default_settings
        self.cfg.ensure_dirs()
        self._sem = asyncio.Semaphore(self.cfg.max_parallel_jobs)
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def create(self, request: RunRequest) -> JobRecord:
        dataset = load_dataset_bundle(request.dataset_id, self.cfg)
        if not dataset.calibration.ready_to_run:
            raise ValueError("Dataset is not ready_to_run after calibration")
        job = JobRecord(
            request=request,
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            status=JobStatus.queued,
        )
        n_models = len(request.model_ids)
        n_rows = len(dataset.rows)
        if request.limit:
            n_rows = min(n_rows, request.limit)
        job.progress_total = n_models * n_rows
        save_job(job, self.cfg)
        return job

    def get(self, job_id: str) -> JobRecord:
        return load_job(job_id, self.cfg)

    def list(self) -> list[JobRecord]:
        return list_jobs(self.cfg)

    async def enqueue(self, request: RunRequest) -> JobRecord:
        job = self.create(request)
        task = asyncio.create_task(self._run(job.id), name=f"job-{job.id}")
        async with self._lock:
            self._tasks[job.id] = task
        return job

    async def run_to_completion(self, request: RunRequest) -> JobRecord:
        """Enqueue and await (for CLI / local Streamlit without long-lived loop)."""
        job = await self.enqueue(request)
        async with self._lock:
            task = self._tasks.get(job.id)
        if task is not None:
            await task
        return load_job(job.id, self.cfg)

    async def _run(self, job_id: str) -> None:
        async with self._sem:
            job = load_job(job_id, self.cfg)
            job.status = JobStatus.running
            job.updated_at = _now()
            save_job(job, self.cfg)
            try:
                dataset = load_dataset_bundle(job.dataset_id, self.cfg)
                models = get_models_by_ids(job.request.model_ids, self.cfg)
                judge = None
                if job.request.judge_model_id:
                    judge = get_models_by_ids([job.request.judge_model_id], self.cfg)[0]

                def on_progress(done: int, total: int, label: str) -> None:
                    current = load_job(job_id, self.cfg)
                    current.progress_done = done
                    current.progress_total = total
                    current.progress_label = label
                    current.updated_at = _now()
                    save_job(current, self.cfg)

                run = await run_benchmark(
                    dataset=dataset,
                    models=models,
                    request=job.request,
                    judge_model=judge,
                    cfg=self.cfg,
                    on_progress=on_progress,
                )
                html_path = write_html_report(run, self.cfg)
                job = load_job(job_id, self.cfg)
                job.status = JobStatus.succeeded
                job.run_id = run.id
                job.html_path = str(html_path)
                job.progress_done = job.progress_total
                job.updated_at = _now()
                save_job(job, self.cfg)
            except Exception as exc:  # noqa: BLE001
                job = load_job(job_id, self.cfg)
                job.status = JobStatus.failed
                job.error = str(exc)
                job.updated_at = _now()
                save_job(job, self.cfg)
            finally:
                async with self._lock:
                    self._tasks.pop(job_id, None)


_manager: JobManager | None = None


def get_job_manager(cfg: Settings | None = None) -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager(cfg)
    return _manager
