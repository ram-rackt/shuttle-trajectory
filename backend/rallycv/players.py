"""Player detection and tracking.

People are detected per frame (RT-DETRv2 by default). Each detection's feet point
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


DETECT_LONG_SIDE = 640
"""Network input size along the image's long side (RT-DETR is trained at 640)."""


def input_size(height: int, width: int, long_side: int = DETECT_LONG_SIDE) -> tuple[int, int]:
    """Network input (height, width): the image's own aspect ratio, long side
    `long_side`, both multiples of 32. Squashing a wide frame into a square
    wastes pixels on the short axis and distorts people."""
    scale = long_side / max(height, width)
    return (
        max(32, round(height * scale / 32) * 32),
        max(32, round(width * scale / 32) * 32),
    )


class RtDetrPersonDetector:
    """RT-DETRv2 person detector via Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str,
        device: str | None = None,
        conf: float = 0.25,
        cache_dir: Path | None = None,
    ) -> None:
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        self._device = device or _best_device()
        self._conf = conf

        def load(cls):  # noqa: ANN001, ANN202
            # Prefer the local cache (filled by scripts/download_models.py) so
            # startup needs no network; download only if it isn't there.
            try:
                return cls.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=True)
            except OSError:
                return cls.from_pretrained(model_id, cache_dir=cache_dir)

        self._processor = load(RTDetrImageProcessor)
        self._model = load(RTDetrV2ForObjectDetection).to(self._device)
        self._model.eval()
        self._person_label = person_label_id(self._model.config.id2label, model_id)

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import torch
        from PIL import Image

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        h, w = input_size(image.height, image.width)

        inputs = self._processor(
            images=image, size={"height": h, "width": w}, return_tensors="pt"
        ).to(self._device)

        with torch.inference_mode():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([(image.height, image.width)]).to(self._device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self._conf
        )[0]

        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()

        mask = labels == self._person_label
        if not np.any(mask):
            return np.zeros((0, 5))
        return np.hstack([boxes[mask], scores[mask][:, None]])


def person_label_id(id2label: dict[int, str], model_id: str = "model") -> int:
    """The class id a detector uses for people.

    Looked up by name: ids differ between label maps (0 in Hugging Face COCO
    models, where 1 is "bicycle"), so guessing a number would silently detect
    the wrong thing.
    """
    for k, v in id2label.items():
        if v.lower() == "person":
            return int(k)
    raise ValueError(f"{model_id} has no 'person' class; use a COCO-trained detector.")


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

    def box_at(self, player: str, frame: float, max_frame_gap: int) -> np.ndarray | None:
        """The player's box at `frame`: interpolated between the observations on
        either side when both are within max_frame_gap, else the nearest one
        within max_frame_gap (None if there is none)."""
        obs = self.observations[player]
        if not obs:
            return None
        frames = np.array([o.frame for o in obs], dtype=np.float64)
        j = int(np.searchsorted(frames, frame))
        if 0 < j < len(obs):
            f0, f1 = frames[j - 1], frames[j]
            if frame - f0 <= max_frame_gap and f1 - frame <= max_frame_gap and f1 > f0:
                u = (frame - f0) / (f1 - f0)
                return obs[j - 1].box * (1 - u) + obs[j].box * u
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
