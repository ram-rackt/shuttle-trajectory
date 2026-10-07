"""Video access helpers (OpenCV / FFmpeg backend)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoError(Exception):
    """Raised when a video can't be opened or decoded."""


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError("Could not open the video.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 1 or w == 0 or h == 0:
            raise VideoError("The video has no readable frames.")
        if count <= 0:  # some containers (e.g. WebM) don't store a frame count
            count = _count_frames(cap)
        return VideoInfo(fps=float(fps), frame_count=count, width=w, height=h)
    finally:
        cap.release()


def _count_frames(cap: cv2.VideoCapture) -> int:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n = 0
    while cap.grab():
        n += 1
    return n


def iter_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (frame_index, BGR frame) for every decodable frame, in order."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("Could not open the video.")
    try:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


def sample_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    """Reads specific frames (sorted indices) by seeking."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError("Could not open the video.")
    frames = []
    try:
        for idx in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    finally:
        cap.release()
    return frames


def median_background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of frames: removes moving players and the shuttle."""
    if not frames:
        raise VideoError("No frames to build a background from.")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)
