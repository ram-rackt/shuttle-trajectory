"""Badminton court model in meters (BWF Laws of Badminton, Appendix 1).

World frame used throughout the pipeline:
  X across the court, 0 = left doubles sideline as seen from the camera, 6.10 = right.
  Y along the court, 0 = near baseline (closest to the camera), 13.40 = far baseline.
  Z up, 0 = floor.

The near player is "Player A" in the analysis contract, so normalized contract
coordinates are simply x = X / WIDTH, y = Y / LENGTH.
"""

from dataclasses import dataclass

import numpy as np

LENGTH = 13.40
WIDTH = 6.10
NET_Y = LENGTH / 2
NET_HEIGHT_CENTER = 1.524
NET_HEIGHT_POSTS = 1.55

SINGLES_INSET = 0.46
SHORT_SERVICE_FROM_NET = 1.98
DOUBLES_LONG_SERVICE_FROM_BASELINE = 0.76

# Painted lines running across the court (constant Y), near to far.
CROSS_LINES_Y: tuple[float, ...] = (
    0.0,
    DOUBLES_LONG_SERVICE_FROM_BASELINE,
    NET_Y - SHORT_SERVICE_FROM_NET,
    NET_Y + SHORT_SERVICE_FROM_NET,
    LENGTH - DOUBLES_LONG_SERVICE_FROM_BASELINE,
    LENGTH,
)

# Painted lines running along the court (constant X), left to right.
# The centre line only runs from each short service line to the baseline.
LONG_LINES_X: tuple[float, ...] = (0.0, SINGLES_INSET, WIDTH / 2, WIDTH - SINGLES_INSET, WIDTH)


@dataclass(frozen=True)
class Segment:
    start: tuple[float, float]
    end: tuple[float, float]


def court_segments() -> list[Segment]:
    """Every painted line segment of the court, in meters on the floor plane."""
    segs = [Segment((0.0, y), (WIDTH, y)) for y in CROSS_LINES_Y]
    for x in LONG_LINES_X:
        if x == WIDTH / 2:
            segs.append(Segment((x, 0.0), (x, CROSS_LINES_Y[2])))
            segs.append(Segment((x, CROSS_LINES_Y[3]), (x, LENGTH)))
        else:
            segs.append(Segment((x, 0.0), (x, LENGTH)))
    return segs


def sample_court_points(spacing: float = 0.25) -> np.ndarray:
    """Points along every painted line, shape (N, 2), for scoring a projection."""
    pts: list[tuple[float, float]] = []
    for seg in court_segments():
        (x0, y0), (x1, y1) = seg.start, seg.end
        n = max(int(np.hypot(x1 - x0, y1 - y0) / spacing), 1)
        for i in range(n + 1):
            u = i / n
            pts.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u))
    return np.asarray(pts, dtype=np.float64)


# Outer doubles corners in the order used for manual calibration:
# near-left, near-right, far-right, far-left.
CORNERS: np.ndarray = np.array(
    [[0.0, 0.0], [WIDTH, 0.0], [WIDTH, LENGTH], [0.0, LENGTH]], dtype=np.float64
)
