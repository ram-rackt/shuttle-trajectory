import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BACKEND_ROOT = Path(__file__).resolve().parent.parent

Analyzer = Literal["cv", "mock"]


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    """Where uploaded videos are stored."""
    max_upload_bytes: int
    analyzer: Analyzer = "cv"
    """'cv' runs the computer-vision pipeline; 'mock' returns simulated data (UI work)."""
    models_dir: Path = BACKEND_ROOT / "data" / "models"
    """Model weights (YOLO, TrackNet). See scripts/download_models.py."""
    mock_stage_seconds: float = 1.2
    """Simulated duration of each mock pipeline stage."""

    @classmethod
    def from_env(cls) -> "Settings":
        analyzer = os.environ.get("RALLYREVIEW_ANALYZER", "cv").lower()
        if analyzer not in ("cv", "mock"):
            raise ValueError("RALLYREVIEW_ANALYZER must be 'cv' or 'mock'.")
        return cls(
            data_dir=Path(os.environ.get("RALLYREVIEW_DATA_DIR", BACKEND_ROOT / "data")),
            max_upload_bytes=int(os.environ.get("RALLYREVIEW_MAX_UPLOAD_BYTES", 2 * 1024**3)),
            analyzer=analyzer,  # type: ignore[arg-type]
            models_dir=Path(
                os.environ.get("RALLYREVIEW_MODELS_DIR", BACKEND_ROOT / "data" / "models")
            ),
            mock_stage_seconds=float(os.environ.get("RALLYREVIEW_MOCK_STAGE_SECONDS", 1.2)),
        )
