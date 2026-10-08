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


class RtDetrPersonDetector:
    """RT-DETRv2 person detector via Hugging Face Transformers."""

    def __init__(self, model_id: str, device: str | None = None, conf: float = 0.25) -> None:
        from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
        import torch

        self._device = device or _best_device()
        self._conf = conf
        
        self._processor = RTDetrImageProcessor.from_pretrained(model_id)
        self._model = RTDetrV2ForObjectDetection.from_pretrained(model_id).to(self._device)
        self._model.eval()

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch
        import cv2
        from PIL import Image

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)]).to(self._device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self._conf
        )[0]

        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()

        person_id = None
        for k, v in self._model.config.id2label.items():
            if v.lower() == 'person':
                person_id = k
                break
        
        if person_id is None:
            person_id = 1  # Fallback to COCO 'person' label (usually 1, sometimes 0)

        mask = labels == person_id
        if not np.any(mask):
            return np.zeros((0, 5))

        person_boxes = boxes[mask]
        person_scores = scores[mask][:, None]
        
        return np.hstack([person_boxes, person_scores])


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
