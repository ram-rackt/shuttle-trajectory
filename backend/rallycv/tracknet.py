"""Learned shuttle detector: TrackNetV3 (Chen & Wang, 2023).

The network below is vendored from https://github.com/qaz812345/TrackNetV3
(MIT License, Copyright (c) 2024 qaz812345); see THIRD_PARTY_NOTICES.md.
Only the inference path is reproduced. Pretrained weights are downloaded
separately (see `scripts/download_models.py`).

Input: the median background plus 8 consecutive frames, RGB, resized to
512x288 and scaled to [0, 1], stacked as 27 channels. Output: one heatmap per
frame; the shuttle is the centroid of the largest region above 0.5.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HEIGHT, WIDTH = 288, 512
SEQ_LEN = 8
HEATMAP_THRESHOLD = 0.5
MAX_JUMP_PX = 120.0
"""Between consecutive distinct frames at 1080p, a jump larger than this means
one of the detections is on something other than the shuttle."""


def _build_model(in_dim: int, out_dim: int):  # noqa: ANN202 - torch types are lazy
    import torch
    from torch import nn

    class Conv2DBlock(nn.Module):
        def __init__(self, i: int, o: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(i, o, kernel_size=3, padding="same", bias=False)
            self.bn = nn.BatchNorm2d(o)
            self.relu = nn.ReLU()

        def forward(self, x):  # noqa: ANN001, ANN202
            return self.relu(self.bn(self.conv(x)))

    class Double2DConv(nn.Module):
        def __init__(self, i: int, o: int) -> None:
            super().__init__()
            self.conv_1, self.conv_2 = Conv2DBlock(i, o), Conv2DBlock(o, o)

        def forward(self, x):  # noqa: ANN001, ANN202
            return self.conv_2(self.conv_1(x))

    class Triple2DConv(nn.Module):
        def __init__(self, i: int, o: int) -> None:
            super().__init__()
            self.conv_1, self.conv_2, self.conv_3 = (
                Conv2DBlock(i, o),
                Conv2DBlock(o, o),
                Conv2DBlock(o, o),
            )

        def forward(self, x):  # noqa: ANN001, ANN202
            return self.conv_3(self.conv_2(self.conv_1(x)))

    class TrackNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_block_1 = Double2DConv(in_dim, 64)
            self.down_block_2 = Double2DConv(64, 128)
            self.down_block_3 = Triple2DConv(128, 256)
            self.bottleneck = Triple2DConv(256, 512)
            self.up_block_1 = Triple2DConv(768, 256)
            self.up_block_2 = Double2DConv(384, 128)
            self.up_block_3 = Double2DConv(192, 64)
            self.predictor = nn.Conv2d(64, out_dim, (1, 1))
            self.sigmoid = nn.Sigmoid()
            self.pool = nn.MaxPool2d((2, 2), stride=(2, 2))
            self.up = nn.Upsample(scale_factor=2)

        def forward(self, x):  # noqa: ANN001, ANN202
            x1 = self.down_block_1(x)
            x2 = self.down_block_2(self.pool(x1))
            x3 = self.down_block_3(self.pool(x2))
            x = self.bottleneck(self.pool(x3))
            x = self.up_block_1(torch.cat([self.up(x), x3], dim=1))
            x = self.up_block_2(torch.cat([self.up(x), x2], dim=1))
            x = self.up_block_3(torch.cat([self.up(x), x1], dim=1))
            return self.sigmoid(self.predictor(x))

    return TrackNet()


@dataclass
class Detection:
    frame: int
    x: float  # full-resolution pixels
    y: float
    confidence: float  # peak heatmap value


Overlap = Literal["full", "adaptive", "none"]


class TrackNetDetector:
    """Runs TrackNet over a stream of frames (distinct frames only).

    The network sees 8 frames at a time. Frames are processed in consecutive
    8-frame windows; an extra window straddling each boundary (the last 4
    frames of one window and the first 4 of the next) can be run too, and each
    frame's heatmap is then the average of the two windows that saw it. That
    averaging recovers the shuttle in frames where one pass misses it, which
    are typically the motion-blurred frames right at a hit.

    `overlap` chooses when the straddling window runs:
      - "full": always (every frame processed twice; the original TrackNet scheme)
      - "adaptive": only where the first pass looks unreliable: a frame with
        no shuttle, or a detection that jumps implausibly far from the frame
        before it (locked onto a racket or shirt). Roughly half the windows
      - "none": never (fastest; misses some detections at hits)
    """

    def __init__(
        self,
        weights: Path,
        background_bgr: np.ndarray,
        device: str | None = None,
        overlap: Overlap = "adaptive",
    ) -> None:
        import torch

        self._torch = torch
        self._overlap = overlap
        self._device = torch.device(device or _best_device())
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        params = ckpt.get("param_dict", {})
        if params.get("seq_len", SEQ_LEN) != SEQ_LEN or params.get("bg_mode") != "concat":
            raise ValueError(
                "Unsupported TrackNet checkpoint (expected seq_len=8, bg_mode=concat)."
            )
        model = _build_model((SEQ_LEN + 1) * 3, SEQ_LEN)
        model.load_state_dict(ckpt["model"])
        self._model = model.to(self._device, memory_format=torch.channels_last).eval()

        h, w = background_bgr.shape[:2]
        self._scale = (w / WIDTH, h / HEIGHT)
        self._background = self._prep(background_bgr)
        self._new: list[tuple[int, np.ndarray]] = []
        """Frames waiting for their window."""
        self._tail: list[tuple[int, np.ndarray]] = []
        """Last half of the previous window, for the straddling window."""
        self._heats: dict[int, list[np.ndarray]] = {}
        self.detections: dict[int, Detection] = {}
        self.windows_run = 0

    @staticmethod
    def _prep(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(
            cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        return np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0

    def push(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        self._new.append((frame_index, self._prep(frame_bgr)))
        if len(self._new) == SEQ_LEN:
            self._step(self._new)
            self._new = []

    def flush(self) -> None:
        """Processes the remaining frames (padding the last window) and finalises."""
        if self._new:
            self._step(self._new)
            self._new = []
        for idx in [i for i, _ in self._tail]:
            self._finalise_frame(idx)
        self._tail = []
        for idx in list(self._heats):
            self._finalise_frame(idx)

    def _step(self, window: list[tuple[int, np.ndarray]]) -> None:
        """Runs one window of new frames, then the straddling window if wanted,
        and finalises the frames no later window can see."""
        self._run(window)
        half = SEQ_LEN // 2
        head = window[:half]
        straddle = self._tail + head
        if self._tail and self._wants_straddle(straddle):
            self._run(straddle)
        for idx, _ in self._tail + head:
            self._finalise_frame(idx)
        self._tail = window[half:]

    def _wants_straddle(self, frames: list[tuple[int, np.ndarray]]) -> bool:
        if self._overlap == "full":
            return True
        if self._overlap == "none":
            return False
        image_width = WIDTH * self._scale[0]
        max_jump = MAX_JUMP_PX * image_width / 1920
        prev: np.ndarray | None = None
        for idx, _ in frames:
            heat = self._heats[idx][0]
            if float(heat.max()) < HEATMAP_THRESHOLD:
                return True  # no shuttle in the first pass
            y, x = np.unravel_index(int(np.argmax(heat)), heat.shape)
            pos = np.array([x * self._scale[0], y * self._scale[1]])
            if prev is not None and np.hypot(*(pos - prev)) > max_jump:
                return True  # implausible jump: one of these is not the shuttle
            prev = pos
        return False

    def _run(self, frames: list[tuple[int, np.ndarray]]) -> None:
        """One network pass. Short windows are padded by repeating the last
        frame; only real frames receive a heatmap."""
        torch = self._torch
        arrays = [a for _, a in frames]
        arrays += [arrays[-1]] * (SEQ_LEN - len(arrays))
        x = np.concatenate([self._background, *arrays], axis=0)[None]
        with torch.inference_mode():
            inp = torch.from_numpy(x).to(self._device, memory_format=torch.channels_last)
            heat = self._model(inp).cpu().numpy()[0]
        self.windows_run += 1
        for k, (idx, _) in enumerate(frames):
            self._heats.setdefault(idx, []).append(heat[k])

    def _finalise_frame(self, idx: int) -> None:
        maps = self._heats.pop(idx, None)
        if maps:
            self._finalise(idx, maps)

    def _finalise(self, idx: int, maps: list[np.ndarray]) -> None:
        heat = np.mean(maps, axis=0)
        peak = float(heat.max())
        if peak < HEATMAP_THRESHOLD:
            return
        mask = (heat > HEATMAP_THRESHOLD).astype(np.uint8)
        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx, cy = centroids[k]
        self.detections[idx] = Detection(
            frame=idx, x=float(cx * self._scale[0]), y=float(cy * self._scale[1]), confidence=peak
        )


def _best_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
