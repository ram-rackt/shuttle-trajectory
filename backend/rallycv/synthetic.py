"""Synthetic broadcast frames with exact ground truth, for tests."""

import cv2
import numpy as np

from . import court_model as cm
from .camera import Camera


def broadcast_camera(
    width: int = 1920,
    height: int = 1080,
    focal: float = 2600.0,
    position: tuple[float, float, float] = (3.05, -14.0, 7.0),
    look_at: tuple[float, float, float] = (3.05, 7.5, 0.0),
) -> Camera:
    """A camera behind and above the near baseline, looking down the court."""
    c = np.array(position, dtype=np.float64)
    forward = np.array(look_at, dtype=np.float64) - c
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])  # rows: camera x (right), y (down), z (forward)
    t = -R @ c
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    return Camera(K=K, R=R, t=t)


def court_homography(camera: Camera) -> np.ndarray:
    """Floor homography (court metres -> pixels) of a camera."""
    H = camera.K @ np.column_stack([camera.R[:, 0], camera.R[:, 1], camera.t])
    return H / H[2, 2]


def render_court(
    camera: Camera,
    size: tuple[int, int] = (1920, 1080),
    *,
    distractors: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Renders a green court with white lines, a darker surround, and (optionally)
    bright advertising-board clutter above the far end."""
    w, h = size
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (70, 60, 90), np.uint8)  # arena floor (BGR)
    img[: h // 3] = (120, 60, 30)  # back wall

    floor = np.array(
        [
            [-2.5, -3.0],
            [cm.WIDTH + 2.5, -3.0],
            [cm.WIDTH + 2.5, cm.LENGTH + 3.0],
            [-2.5, cm.LENGTH + 3.0],
        ]
    )
    poly = camera.project(np.c_[floor, np.zeros(4)]).astype(np.int32)
    cv2.fillConvexPoly(img, poly, (110, 150, 40))  # green mat

    if distractors:
        for _ in range(12):
            x = int(rng.uniform(0, w - 200))
            y = int(rng.uniform(h * 0.1, h * 0.32))
            cv2.rectangle(img, (x, y), (x + int(rng.uniform(80, 300)), y + 30), (240, 240, 240), 2)
            cv2.putText(
                img, "VICTOR", (x + 5, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 250, 250), 2
            )

    half = 0.02  # line half-width in metres (40 mm lines)
    for seg in cm.court_segments():
        (x0, y0), (x1, y1) = seg.start, seg.end
        d = np.array([x1 - x0, y1 - y0])
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]]) * half
        quad = np.array([[x0, y0] + n, [x1, y1] + n, [x1, y1] - n, [x0, y0] - n])
        pts = camera.project(np.c_[quad, np.zeros(4)])
        cv2.fillConvexPoly(
            img, np.round(pts * 16).astype(np.int32), (245, 245, 245), lineType=cv2.LINE_AA, shift=4
        )

    noise = rng.normal(0, 3, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
