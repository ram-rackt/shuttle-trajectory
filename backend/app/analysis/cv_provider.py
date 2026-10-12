"""Adapter: runs the `rallycv` pipeline and converts its output to the contract."""

import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from .contract import (
    CameraCalibration,
    CourtPoint,
    ImagePoint,
    PlayerSample,
    PlayerTrack,
    RallyAnalysis,
    Shot,
    ShuttleSample,
)
from .provider import AnalysisError, AnalysisRequest, ProgressCallback

logger = logging.getLogger(__name__)


class CVAnalysisProvider:
    """Real analysis: court detection, player and shuttle tracking, shot detection."""

    def __init__(
        self,
        models_dir: Path,
        person_model: str,
        device: str | None = None,
        max_duration_sec: float = 660.0,
    ) -> None:
        self._models_dir = models_dir
        self._max_duration_sec = max_duration_sec
        self._person_model = person_model
        self._device = device
        self._detector = None
        self._lock = threading.Lock()  # one analysis at a time: models are not thread-safe

    @property
    def version(self) -> str:
        """Cached results are reused only from the same pipeline and models."""
        from rallycv.pipeline import PIPELINE_VERSION

        return f"cv-{PIPELINE_VERSION}-{self._person_model}"

    @property
    def stages(self) -> Sequence[str]:
        from rallycv.pipeline import STAGES

        return STAGES

    def _person_detector(self):  # noqa: ANN202 - lazy heavy import
        if self._detector is None:
            from rallycv.players import RtDetrPersonDetector

            self._detector = RtDetrPersonDetector(
                self._person_model, device=self._device, cache_dir=self._models_dir / "hf"
            )
        return self._detector

    def _config(self, request: AnalysisRequest):  # noqa: ANN202 - rallycv imported lazily
        from rallycv import pipeline

        return pipeline.PipelineConfig(
            tracknet_weights=self._models_dir / "ckpts" / "TrackNet_best.pt",
            device=self._device,
            max_duration_sec=self._max_duration_sec,
            start_sec=request.start_sec,
            end_sec=request.end_sec,
        )

    def warm_up(self, request: AnalysisRequest, should_stop: Callable[[], bool]) -> None:
        """Caches the shuttle detections for this video while nothing else needs
        the models (see rallycv.pipeline.warm_scan). Returns early once
        `should_stop()` is True."""
        from rallycv import pipeline
        from rallycv.cache import StageCache

        if request.cache_dir is None:
            return
        with self._lock:
            if should_stop():
                return
            started = time.monotonic()
            done = pipeline.warm_scan(
                request.video_path,
                self._config(request),
                StageCache(request.cache_dir),
                should_stop,
            )
            logger.info(
                "Warm-up of %s %s after %.0f s",
                request.video_path.name,
                "finished" if done else "stopped",
                time.monotonic() - started,
            )

    def analyze(self, request: AnalysisRequest, on_progress: ProgressCallback) -> RallyAnalysis:
        from rallycv import pipeline
        from rallycv.cache import StageCache

        config = self._config(request)
        cache = StageCache(request.cache_dir) if request.cache_dir is not None else None
        with self._lock:
            try:
                result = pipeline.run(
                    request.video_path,
                    self._person_detector(),
                    on_progress=on_progress,
                    court_corners=request.court_corners,
                    config=config,
                    cache=cache,
                )
            except pipeline.PipelineError as e:
                raise AnalysisError(e.code, e.message) from e
        logger.info("CV analysis stats: %s", result.stats)
        return to_contract(result)


def _court_point(xy: np.ndarray) -> CourtPoint:
    from rallycv import court_model as cm

    return CourtPoint(x=round(float(xy[0]) / cm.WIDTH, 4), y=round(float(xy[1]) / cm.LENGTH, 4))


def to_contract(result) -> RallyAnalysis:  # noqa: ANN001 - rallycv.pipeline.RallyResult
    from rallycv import court_model as cm

    w, h = result.image_size

    shuttle: list[ShuttleSample] = []
    for t, g, im, z in zip(
        result.shuttle_t,
        result.shuttle_ground,
        result.shuttle_image,
        result.shuttle_height,
        strict=True,
    ):
        shuttle.append(
            ShuttleSample(
                t=round(float(t), 3),
                position=_court_point(g) if np.isfinite(g).all() else None,
                image=(
                    ImagePoint(x=round(float(im[0]) / w, 4), y=round(float(im[1]) / h, 4))
                    if np.isfinite(im).all()
                    else None
                ),
                height=round(float(z), 2) if np.isfinite(z) else None,
            )
        )

    players = [
        PlayerTrack(
            id=pid,  # type: ignore[arg-type]
            label=f"Player {pid}",
            samples=[
                PlayerSample(t=round(float(t), 3), position=_court_point(p))
                for t, p in zip(times, pos, strict=True)
            ],
        )
        for pid, (times, pos) in sorted(result.players.items())
    ]

    shots = [
        Shot(
            index=s.index,
            hitter=s.hitter,  # type: ignore[arg-type]
            type=s.type,  # type: ignore[arg-type]
            start_time=round(s.start_time, 3),
            end_time=round(s.end_time, 3),
            outcome=s.outcome,  # type: ignore[arg-type]
        )
        for s in result.shots
    ]

    # Homography for the UI: normalized court (x, y in 0-1) -> normalized image.
    to_metres = np.diag([cm.WIDTH, cm.LENGTH, 1.0])
    to_unit_image = np.diag([1 / w, 1 / h, 1.0])
    H = to_unit_image @ result.calibration.court_to_image @ to_metres
    H = H / H[2, 2]

    return RallyAnalysis(
        source="cv",
        duration_sec=round(result.duration, 3),
        players=players,
        shuttle=shuttle,
        shots=shots,
        calibration=CameraCalibration(
            court_to_image=[round(float(v), 8) for v in H.ravel()],
            confidence=round(result.calibration.confidence, 3),
            manual=result.calibration.manual,
        ),
    )
