"""RallyReview API entry point.

Request handlers depend only on the `AnalysisProvider` interface and the
analysis contract (see docs/analysis-contract.md). The provider is the CV
pipeline by default; RALLYREVIEW_ANALYZER=mock selects simulated data.
"""

from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from .analysis import AnalysisProvider
from .analysis.mock import MockAnalysisProvider
from .api.analyses import router as analyses_router
from .config import Settings
from .jobs import JobStore

try:
    __version__ = version("rallyreview-backend")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0"


def create_app(
    settings: Settings | None = None,
    provider: AnalysisProvider | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="RallyReview API", version=__version__)
    app.state.settings = settings
    app.state.jobs = JobStore()
    app.state.provider = provider or _default_provider(settings)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(analyses_router)
    return app


def _default_provider(settings: Settings) -> AnalysisProvider:
    if settings.analyzer == "mock":
        return MockAnalysisProvider(settings.mock_stage_seconds)
    from .analysis.cv_provider import CVAnalysisProvider

    return CVAnalysisProvider(settings.models_dir)


app = create_app()
