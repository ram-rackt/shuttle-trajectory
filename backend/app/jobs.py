"""In-memory analysis jobs.

Good enough for a single-process prototype: jobs are lost on restart and are
not shared between workers. Swap for a persistent queue before production.
"""

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Literal

from .analysis import AnalysisError, AnalysisProvider, AnalysisRequest, RallyAnalysis

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "processing", "ready", "failed"]


@dataclass
class Job:
    id: str
    request: AnalysisRequest
    status: JobStatus = "queued"
    progress: float = 0.0
    current_stage: int | None = None
    error: str | None = None
    error_code: str | None = None
    result: RallyAnalysis | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes: object) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)


def run_job(store: JobStore, provider: AnalysisProvider, job_id: str) -> None:
    """Runs one job to completion. Called from a worker thread."""
    job = store.get(job_id)
    if job is None:
        return
    store.update(job_id, status="processing", current_stage=0)

    def on_progress(stage_index: int, progress: float) -> None:
        store.update(job_id, current_stage=stage_index, progress=min(max(progress, 0.0), 1.0))

    try:
        result = provider.analyze(job.request, on_progress)
    except AnalysisError as e:
        logger.info("Analysis job %s failed: %s (%s)", job_id, e.message, e.code)
        store.update(job_id, status="failed", error=e.message, error_code=e.code)
        return
    except Exception:
        logger.exception("Analysis job %s crashed", job_id)
        store.update(
            job_id,
            status="failed",
            error="Analysis failed unexpectedly. Please try again.",
            error_code="internal_error",
        )
        return
    store.update(job_id, status="ready", progress=1.0, current_stage=None, result=result)
