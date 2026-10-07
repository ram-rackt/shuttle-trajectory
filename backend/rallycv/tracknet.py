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

import cv2
import numpy as np

logger = logging.getLogger(__name__)

HEIGHT, WIDTH = 288, 512
SEQ_LEN = 8
HEATMAP_THRESHOLD = 0.5


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


class TrackNetDetector:
    """Runs TrackNet over a stream of frames (distinct frames only).

    Windows of 8 frames overlap by half; each frame's heatmap is the average
    of the two windows that contain it, which smooths window-edge errors.
    """

    def __init__(
        self, weights: Path, background_bgr: np.ndarray, device: str | None = None
    ) -> None:
        import torch

        self._torch = torch
        self._device = torch.device(device or _best_device())
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        params = ckpt.get("param_dict", {})
        if params.get("seq_len", SEQ_LEN) != SEQ_LEN or params.get("bg_mode") != "concat":
            raise ValueError(
                "Unsupported TrackNet checkpoint (expected seq_len=8, bg_mode=concat)."
            )
        model = _build_model((SEQ_LEN + 1) * 3, SEQ_LEN)
        model.load_state_dict(ckpt["model"])
        self._model = model.to(self._device).eval()

        h, w = background_bgr.shape[:2]
        self._scale = (w / WIDTH, h / HEIGHT)
        self._background = self._prep(background_bgr)
        self._frames: list[np.ndarray] = []
        self._indices: list[int] = []
        self._pending: dict[int, list[np.ndarray]] = {}
        self.detections: dict[int, Detection] = {}

    @staticmethod
    def _prep(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(
            cv2.resize(frame_bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
        )
        return np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0

    def push(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        self._frames.append(self._prep(frame_bgr))
        self._indices.append(frame_index)
        if len(self._frames) == SEQ_LEN:
            self._run_window()
            half = SEQ_LEN // 2
            self._frames = self._frames[half:]
            self._indices = self._indices[half:]

    def flush(self) -> None:
        """Processes the remaining frames (padding the last window)."""
        if self._frames:
            n = len(self._frames)
            while len(self._frames) < SEQ_LEN:
                self._frames.append(self._frames[-1])
                self._indices.append(self._indices[-1])
            self._run_window(valid=n)
            self._frames, self._indices = [], []
        for idx, maps in list(self._pending.items()):
            self._finalise(idx, maps)
        self._pending.clear()

    def _run_window(self, valid: int = SEQ_LEN) -> None:
        torch = self._torch
        x = np.concatenate([self._background, *self._frames], axis=0)[None]
        with torch.no_grad():
            heat = self._model(torch.from_numpy(x).to(self._device)).cpu().numpy()[0]
        for k in range(valid):
            idx = self._indices[k]
            self._pending.setdefault(idx, []).append(heat[k])
        # Frames that can't appear in any later window are complete.
        half = SEQ_LEN // 2
        done = self._indices[:half] if valid == SEQ_LEN else self._indices[:valid]
        for idx in done:
            if idx in self._pending:
                self._finalise(idx, self._pending.pop(idx))

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
