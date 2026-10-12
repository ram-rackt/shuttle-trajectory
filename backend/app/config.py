import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

BACKEND_ROOT = Path(__file__).resolve().parent.parent

Analyzer = Literal["cv", "mock"]

DEFAULT_PERSON_MODEL = "PekingU/rtdetr_v2_r18vd"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    """Where uploaded videos are stored."""
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

    @classmethod
    def from_env(cls) -> "Settings":
        # Defaults to the real pipeline; mock data only when explicitly requested.
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
            person_model=os.environ.get("RALLYREVIEW_PERSON_MODEL", DEFAULT_PERSON_MODEL),
            device=os.environ.get("RALLYREVIEW_DEVICE") or None,
            mock_stage_seconds=float(os.environ.get("RALLYREVIEW_MOCK_STAGE_SECONDS", 1.2)),
        )
