from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .contract import RallyAnalysis

ProgressCallback = Callable[[int, float], None]
"""Called as on_progress(stage_index, overall_progress_0_to_1)."""


@dataclass(frozen=True)
class AnalysisRequest:
    video_path: Path
    duration_sec: float | None = None
    """Duration reported by the browser; the CV pipeline reads its own."""
    court_corners: tuple[tuple[float, float], ...] | None = None
    """Optional manual calibration: the 4 outer court corners in normalized image
    coordinates (0-1), ordered near-left, near-right, far-right, far-left."""
    start_sec: float | None = None
    end_sec: float | None = None
    """Analyse only this part of the video; result times stay relative to its start."""
    cache_dir: Path | None = None
    """Where this video's intermediate results may be cached (None: no caching)."""


class AnalysisError(Exception):
    """A failure the user can understand and possibly fix (e.g. by marking the court)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AnalysisProvider(Protocol):
    """Anything that turns a rally video into a RallyAnalysis.

    `analyze` is synchronous and may be slow/CPU-bound; callers run it off the
    event loop. Implementations: MockAnalysisProvider and the CV pipeline.
    """

    @property
    def stages(self) -> Sequence[str]:
        """Human-readable pipeline stage labels, in order."""
        ...

    # Optionally, a `version` attribute identifying what produces the results;
    # cached results from a different version are not reused.
    #
    # Optionally, `warm_up(request, should_stop)`: precompute what doesn't need
    # the court (into request.cache_dir) after the court wasn't found, while the
    # user marks its corners. It must return soon after should_stop() is True.

    def analyze(self, request: AnalysisRequest, on_progress: ProgressCallback) -> RallyAnalysis: ...
