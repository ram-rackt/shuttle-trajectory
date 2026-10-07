"""Small projective-geometry helpers shared by the pipeline."""

import numpy as np


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Maps (N, 2) points through a 3x3 homography. Points at infinity become nan."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    w = homog[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = homog[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def line_through(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Homogeneous line through two image points."""
    return np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])


def intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    """Intersection of two homogeneous lines, or None if (nearly) parallel."""
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def scale_matrix(sx: float, sy: float | None = None) -> np.ndarray:
    sy = sx if sy is None else sy
    return np.diag([sx, sy, 1.0])


def is_convex_quad(quad: np.ndarray) -> bool:
    """True if the 4 points form a convex, non-degenerate quadrilateral (in order)."""
    signs = []
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    return bool(all(s == signs[0] and s != 0 for s in signs))


def polygon_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
