import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

Analyzer = Literal["cv", "mock"]

DEFAULT_PERSON_MODEL = "PekingU/rtdetr_v2_r18vd"


@dataclass(frozen=True)
class Settings:
    storage_dir: Path
    """Everything the app stores: database, videos, caches. Git-ignored."""
    max_upload_bytes: int
    analyzer: Analyzer = "cv"
    """'cv' runs the computer-vision pipeline; 'mock' returns simulated data (UI work)."""
    models_dir: Path = BACKEND_ROOT / "data" / "models"
    """Model weights (RT-DETRv2 cache, TrackNet). See scripts/download_models.py."""
    person_model: str = DEFAULT_PERSON_MODEL
    """Hugging Face model id (or local path) of the RT-DETRv2 person detector."""
    device: str | None = None
    """Torch device ('cuda', 'mps', 'cpu'); None picks the best available."""
    mock_stage_seconds: float = 1.2
    """Simulated duration of each mock pipeline stage."""
    max_video_seconds: float = 660.0
    """Longest stretch of video analysed in one job (11 minutes)."""
    youtube_enabled: bool = True
    """Accept YouTube links (needs the optional yt-dlp dependency)."""
    youtube_max_source_seconds: float = 3 * 60 * 60
    """Longest YouTube video that may be downloaded at all (a part of it, up to
    max_video_seconds, is analysed)."""
    youtube_max_concurrent_downloads: int = 1

    @classmethod
    def from_env(cls) -> "Settings":
        # Defaults to the real pipeline; mock data only when explicitly requested.
        analyzer = os.environ.get("RALLYREVIEW_ANALYZER", "cv").lower()
        if analyzer not in ("cv", "mock"):
            raise ValueError("RALLYREVIEW_ANALYZER must be 'cv' or 'mock'.")
        env = os.environ.get
        return cls(
            storage_dir=Path(env("RALLYREVIEW_STORAGE_DIR", PROJECT_ROOT / "storage")),
            max_upload_bytes=int(env("RALLYREVIEW_MAX_UPLOAD_BYTES", 2 * 1024**3)),
            analyzer=analyzer,  # type: ignore[arg-type]
            models_dir=Path(env("RALLYREVIEW_MODELS_DIR", BACKEND_ROOT / "data" / "models")),
            person_model=env("RALLYREVIEW_PERSON_MODEL", DEFAULT_PERSON_MODEL),
            device=env("RALLYREVIEW_DEVICE") or None,
            mock_stage_seconds=float(env("RALLYREVIEW_MOCK_STAGE_SECONDS", 1.2)),
            max_video_seconds=float(env("RALLYREVIEW_MAX_VIDEO_SECONDS", 660)),
            youtube_enabled=env("RALLYREVIEW_YOUTUBE", "on").lower() not in ("0", "off", "false"),
            youtube_max_source_seconds=float(
                env("RALLYREVIEW_YOUTUBE_MAX_SOURCE_SECONDS", 3 * 60 * 60)
            ),
        )
