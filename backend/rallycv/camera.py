"""Full camera recovery from the floor homography.

A floor homography H (court metres -> pixels) equals K [r1 r2 t] up to scale.
Assuming square pixels, zero skew and the principal point at the image centre
(true for broadcast cameras to within a few percent), the focal length follows
from the orthonormality of r1 and r2 (Zhang, 2000). With K known we get the
full pose and can cast a ray through any pixel, which is what lets us place an
airborne shuttle in 3D.
"""

from dataclasses import dataclass

import numpy as np


class CameraError(Exception):
    """Raised when the homography does not correspond to a plausible camera."""


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Camera position in court metres (X, Y, Z)."""
        return -self.R.T @ self.t

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    def project(self, pts3d: np.ndarray) -> np.ndarray:
        """(N, 3) court-space points -> (N, 2) pixels."""
        pts3d = np.asarray(pts3d, dtype=np.float64).reshape(-1, 3)
        h = np.hstack([pts3d, np.ones((len(pts3d), 1))]) @ self.P.T
        return h[:, :2] / h[:, 2:3]

    def rays(self, pixels: np.ndarray) -> np.ndarray:
        """Unit ray directions (N, 3) in court space through the given pixels."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        h = np.hstack([pixels, np.ones((len(pixels), 1))])
        d = (self.R.T @ np.linalg.inv(self.K) @ h.T).T
        return d / np.linalg.norm(d, axis=1, keepdims=True)


def camera_from_homography(H: np.ndarray, image_size: tuple[int, int]) -> Camera:
    """Recovers K, R, t from a court-to-image homography."""
    w, h = image_size
    cx, cy = w / 2, h / 2
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    Hc = T @ H  # principal point moved to the origin
    h1, h2 = Hc[:, 0], Hc[:, 1]

    # The closed-form solutions for f divide by h1[2]*h2[2] or h1[2]^2 - h2[2]^2,
    # which are near zero for a camera looking straight down the court (the
    # court's X axis is almost parallel to the image plane). Instead, search
    # for the f that makes K^-1 h1 and K^-1 h2 closest to orthonormal.
    def residual(f: float) -> float:
        k = np.array([1 / f, 1 / f, 1.0])
        r1, r2 = h1 * k, h2 * k
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        cos = (r1 @ r2) / (n1 * n2)
        ratio = (n1 - n2) / (n1 + n2)
        return float(cos**2 + ratio**2)

    candidates = np.geomspace(0.3 * w, 20 * w, 400)
    errs = np.array([residual(f) for f in candidates])
    i = int(np.argmin(errs))
    lo, hi = candidates[max(i - 1, 0)], candidates[min(i + 1, len(candidates) - 1)]
    for _ in range(60):  # golden-section refinement
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        if residual(m1) < residual(m2):
            hi = m2
        else:
            lo = m1
    f = float((lo + hi) / 2)
    if residual(f) > 0.01 or not (0.3 * w < f < 20 * w):
        raise CameraError(f"No consistent camera for this court fit (f={f:.0f}px).")

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Kinv = np.linalg.inv(K)
    r1 = Kinv @ H[:, 0]
    r2 = Kinv @ H[:, 1]
    t = Kinv @ H[:, 2]
    scale = 2.0 / (np.linalg.norm(r1) + np.linalg.norm(r2))
    r1, r2, t = r1 * scale, r2 * scale, t * scale
    if t[2] < 0:  # court must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    U, _, Vt = np.linalg.svd(R)  # nearest true rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1

    cam = Camera(K=K, R=R, t=t)
    if cam.center[2] <= 0:
        # r3 = r1 x r2 points up for our right-handed court frame; if the camera
        # ends up below the floor the handedness is flipped.
        R = R @ np.diag([1, 1, -1.0])
        cam = Camera(K=K, R=R, t=t)
    if cam.center[2] <= 0:
        raise CameraError("Recovered camera is below the court.")
    return cam


def intersect_vertical_plane(
    camera: Camera, pixels: np.ndarray, p0: np.ndarray, p1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Intersects pixel rays with the vertical plane through floor points p0, p1.

    Returns (points (N, 3), conditioning (N,)) where conditioning is |cos| of
    the angle between ray and plane normal: near 0 means the ray grazes the
    plane and the depth along it is unreliable.
    """
    direction = np.array([p1[0] - p0[0], p1[1] - p0[1], 0.0])
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        raise ValueError("Plane endpoints coincide.")
    normal = np.array([-direction[1], direction[0], 0.0]) / norm
    origin = np.array([p0[0], p0[1], 0.0])
    c = camera.center
    rays = camera.rays(pixels)
    denom = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        s = ((origin - c) @ normal) / denom
    pts = c + rays * s[:, None]
    return pts, np.abs(denom)
