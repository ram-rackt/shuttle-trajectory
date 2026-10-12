"""RallyReview API entry point.

Request handlers depend only on the `AnalysisProvider` interface and the
analysis contract (see docs/analysis-contract.md). The provider is the CV
pipeline by default; RALLYREVIEW_ANALYZER=mock selects simulated data.
Videos, jobs and results live in local storage (see storage.py).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from . import youtube
from .analysis import AnalysisProvider
from .analysis.mock import MockAnalysisProvider
from .api.analyses import router as analyses_router
from .api.media import router as media_router
from .config import Settings
from .jobs import JobRunner, JobStore
from .storage import Storage

logger = logging.getLogger(__name__)

try:
    __version__ = version("rallyreview-backend")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0"


def create_app(
    settings: Settings | None = None,
    provider: AnalysisProvider | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    provider = provider or _default_provider(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        storage = Storage(settings.storage_dir)
        interrupted = storage.fail_interrupted_jobs()
        if interrupted:
            logger.info("Marked %d job(s) interrupted by a restart as failed.", interrupted)
        jobs = JobStore(storage)
        app.state.storage = storage
        app.state.jobs = jobs
        app.state.runner = JobRunner(jobs, storage, provider, settings)
        try:
            yield
        finally:
            storage.close()

    app = FastAPI(title="RallyReview API", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.provider = provider

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/config")
    def config() -> dict[str, object]:
        """Limits the UI should enforce before sending anything."""
        return {
            "maxUploadBytes": settings.max_upload_bytes,
            "maxVideoSeconds": settings.max_video_seconds,
            "youtube": {
                "enabled": settings.youtube_enabled and youtube.available(),
                "maxSourceSeconds": settings.youtube_max_source_seconds,
            },
        }

    app.include_router(analyses_router)
    app.include_router(media_router)
    return app


def _default_provider(settings: Settings) -> AnalysisProvider:
    if settings.analyzer == "mock":
        return MockAnalysisProvider(settings.mock_stage_seconds)
    from .analysis.cv_provider import CVAnalysisProvider

    return CVAnalysisProvider(
        settings.models_dir, settings.person_model, settings.device, settings.max_video_seconds
    )


app = create_app()
