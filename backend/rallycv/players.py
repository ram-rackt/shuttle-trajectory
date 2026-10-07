"""Player detection and tracking.

People are detected per frame (YOLO by default). Each detection's feet point
(bottom-centre of the box) is mapped to the floor with the court homography;
detections standing outside the court area (umpires, line judges, coaches)
are dropped, and the remaining ones are assigned to the near half (Player A)
or the far half (Player B). Singles only: one player per side.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import court_model as cm
from .court import CourtCalibration

logger = logging.getLogger(__name__)

COURT_MARGIN_X = 1.2
COURT_MARGIN_Y = 2.0
"""How far outside the doubles lines (m) a player's feet may be and still count."""

MAX_GAP_SEC = 1.0
"""Gaps in a player's track up to this long are interpolated."""


class PersonDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns (N, 5) boxes: x1, y1, x2, y2, confidence, in pixels."""
        ...


class YoloPersonDetector:
    """Ultralytics YOLO person detector. Weights download on first use."""

    def __init__(self, weights: Path, device: str | None = None, conf: float = 0.25) -> None:
        from ultralytics import YOLO  # heavy import, keep lazy

        weights.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(weights))
        self._device = device or _best_device()
        self._conf = conf

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        result = self._model.predict(
            frame_bgr,
            classes=[0],
            conf=self._conf,
            imgsz=960,
            device=self._device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 5))
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()[:, None]
        return np.hstack([xyxy, conf])


def _best_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch missing or broken
        pass
    return "cpu"


@dataclass
class PlayerObservation:
    frame: int
    feet_court: np.ndarray  # (X, Y) metres
    box: np.ndarray  # x1, y1, x2, y2 pixels
    confidence: float


@dataclass
class PlayerTracks:
    """Per-player observations, keyed by contract id ('A' near, 'B' far)."""

    observations: dict[str, list[PlayerObservation]] = field(
        default_factory=lambda: {"A": [], "B": []}
    )

    def box_at(self, player: str, frame: int, max_frame_gap: int) -> np.ndarray | None:
        """Nearest observed box within max_frame_gap frames."""
        obs = self.observations[player]
        if not obs:
            return None
        frames = np.array([o.frame for o in obs])
        i = int(np.argmin(np.abs(frames - frame)))
        return obs[i].box if abs(frames[i] - frame) <= max_frame_gap else None

    def position_at(self, player: str, t_frame: float) -> np.ndarray | None:
        """Interpolated floor position (metres) at a (fractional) frame index."""
        obs = self.observations[player]
        if not obs:
            return None
        frames = np.array([o.frame for o in obs], dtype=np.float64)
        xs = np.array([o.feet_court[0] for o in obs])
        ys = np.array([o.feet_court[1] for o in obs])
        return np.array([np.interp(t_frame, frames, xs), np.interp(t_frame, frames, ys)])


class PlayerTracker:
    """Assigns per-frame detections to Player A (near half) and B (far half)."""

    def __init__(self, calibration: CourtCalibration) -> None:
        self._cal = calibration
        self.tracks = PlayerTracks()
        self._last: dict[str, np.ndarray | None] = {"A": None, "B": None}

    def update(self, frame_index: int, detections: np.ndarray) -> None:
        if len(detections) == 0:
            return
        feet_px = np.column_stack([(detections[:, 0] + detections[:, 2]) / 2, detections[:, 3]])
        feet = self._cal.to_court(feet_px)
        on_court = (
            (feet[:, 0] > -COURT_MARGIN_X)
            & (feet[:, 0] < cm.WIDTH + COURT_MARGIN_X)
            & (feet[:, 1] > -COURT_MARGIN_Y)
            & (feet[:, 1] < cm.LENGTH + COURT_MARGIN_Y)
        )
        for player, side in (("A", feet[:, 1] < cm.NET_Y), ("B", feet[:, 1] >= cm.NET_Y)):
            idx = np.nonzero(on_court & side)[0]
            if len(idx) == 0:
                continue
            last = self._last[player]
            if last is None:
                # Prefer confident, large (close-to-camera-scale) boxes.
                heights = detections[idx, 3] - detections[idx, 1]
                score = detections[idx, 4] * heights
            else:
                dist = np.linalg.norm(feet[idx] - last, axis=1)
                score = detections[idx, 4] - 0.25 * dist
            k = idx[int(np.argmax(score))]
            self._last[player] = feet[k]
            self.tracks.observations[player].append(
                PlayerObservation(
                    frame=frame_index,
                    feet_court=feet[k],
                    box=detections[k, :4].copy(),
                    confidence=float(detections[k, 4]),
                )
            )


def smooth_track(
    frames: np.ndarray, positions: np.ndarray, fps: float, window: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Median-filters a (N, 2) track to suppress detection jitter and outliers."""
    if len(positions) < window:
        return frames, positions
    half = window // 2
    padded = np.pad(positions, ((half, half), (0, 0)), mode="edge")
    smoothed = np.stack([np.median(padded[i : i + window], axis=0) for i in range(len(positions))])
    return frames, smoothed
