"""Automatic court detection: find the homography from court meters to image pixels.

Approach (after Farin et al., "Robust camera calibration for sport videos using
court models"), adapted to badminton:

1. Line-pixel mask: thin bright structures (brighter than pixels `tau` away on
   both sides) with low colour saturation, i.e. white paint.
2. Hough segments, merged into infinite candidate lines and split into two
   families: lines running across the image (court cross lines) and lines
   running into the image (sidelines / centre line).
3. Exhaustive hypothesis search: two lines from each family are matched to two
   model lines each, giving four point correspondences and a homography. Each
   hypothesis is scored by how many projected model-line points land on paint.
4. The best hypothesis is refined by re-fitting every model line to nearby
   paint pixels and solving a least-squares homography from all intersections.
"""

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np

from . import court_model as cm
from .geometry import apply_homography, intersect, is_convex_quad, polygon_area, scale_matrix

WORK_WIDTH = 1280
"""Detection runs at this width; the result is scaled back to full resolution."""

MIN_CONFIDENCE = 0.45
"""Below this fraction of model-line points on paint, the fit is rejected."""

MIN_HYPOTHESIS_FRACTION = 0.25
"""Hypotheses explaining less than this fraction of their own lines are ignored."""

MAX_CROSS_CANDIDATES = 10
MAX_LONG_CANDIDATES = 8
CROSS_LINE_MAX_TILT_DEG = 20.0
MIN_SIDELINE_TILT_DEG = 30.0


class CourtNotFoundError(Exception):
    """Raised when no court model fits the image well enough."""


@dataclass(frozen=True)
class CourtCalibration:
    court_to_image: np.ndarray
    """3x3 homography: court meters (X, Y) on the floor -> image pixels (full resolution)."""
    image_size: tuple[int, int]
    """(width, height) in pixels."""
    confidence: float
    """Fraction of model line points that land on painted lines (0-1)."""
    manual: bool = False

    @property
    def image_to_court(self) -> np.ndarray:
        return np.linalg.inv(self.court_to_image)

    def to_image(self, pts: np.ndarray) -> np.ndarray:
        return apply_homography(self.court_to_image, pts)

    def to_court(self, pts: np.ndarray) -> np.ndarray:
        return apply_homography(self.image_to_court, pts)


@dataclass
class _Line:
    coef: np.ndarray  # homogeneous line a*x + b*y + c = 0, normalised so a^2 + b^2 = 1
    support: float  # total length of Hough segments merged into this line
    angle: float  # direction in degrees, [0, 180)

    def y_at(self, x: float) -> float:
        a, b, c = self.coef
        return -(a * x + c) / b if abs(b) > 1e-9 else np.inf

    def x_at(self, y: float) -> float:
        a, b, c = self.coef
        return -(b * y + c) / a if abs(a) > 1e-9 else np.inf


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def detect_court(image_bgr: np.ndarray) -> CourtCalibration:
    """Detects the court in a (preferably player-free) frame.

    Raises CourtNotFoundError if no plausible court is found.
    """
    h_full, w_full = image_bgr.shape[:2]
    scale = WORK_WIDTH / w_full
    img = cv2.resize(image_bgr, (WORK_WIDTH, round(h_full * scale)), interpolation=cv2.INTER_AREA)

    mask = line_pixel_mask(img) & floor_mask(img)
    lines = _candidate_lines(mask)
    cross = [ln for ln in lines if _is_cross(ln)]
    long_ = [ln for ln in lines if not _is_cross(ln)]
    if len(cross) < 2 or len(long_) < 2:
        raise CourtNotFoundError("Not enough court lines visible.")

    score_map = _score_map(mask)
    H, support = _search(cross, long_, score_map, img.shape[1], img.shape[0])
    if H is None:
        raise CourtNotFoundError("No court model fits the detected lines.")

    for _ in range(3):
        refined = _refine(H, mask)
        if refined is None or not _plausible(refined[None], img.shape[1], img.shape[0])[0]:
            break
        refined_support = _evaluate(refined[None], score_map)[0][0]
        if refined_support + 1e-6 < support:
            break
        H, support = refined, refined_support
    score = float(_score(H[None], score_map)[0])

    if score < MIN_CONFIDENCE:
        raise CourtNotFoundError(f"Court fit too weak (confidence {score:.2f}).")

    to_full = scale_matrix(1 / scale)
    return CourtCalibration(
        court_to_image=to_full @ H, image_size=(w_full, h_full), confidence=float(score)
    )


def calibration_from_corners(
    corners_px: np.ndarray, image_size: tuple[int, int], image_bgr: np.ndarray | None = None
) -> CourtCalibration:
    """Builds a calibration from 4 user-marked outer corners (near-left, near-right,
    far-right, far-left), in full-resolution pixels. If a frame is given, the fit
    is refined against the painted lines and its confidence measured."""
    corners_px = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    if not is_convex_quad(corners_px):
        raise ValueError("The four corners must form a convex quadrilateral, in order.")
    H = cv2.getPerspectiveTransform(cm.CORNERS.astype(np.float32), corners_px.astype(np.float32))
    H = H.astype(np.float64)
    confidence = 1.0
    if image_bgr is not None:
        w_full = image_bgr.shape[1]
        scale = WORK_WIDTH / w_full
        img = cv2.resize(
            image_bgr, (WORK_WIDTH, round(image_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA
        )
        mask = line_pixel_mask(img) & floor_mask(img)
        score_map = _score_map(mask)
        to_work = scale_matrix(scale)
        Hw = to_work @ H
        score = _score(Hw[None], score_map)[0]
        refined = _refine(Hw, mask)
        if refined is not None:
            refined_score = _score(refined[None], score_map)[0]
            if refined_score > score:
                Hw, score = refined, refined_score
        H = np.linalg.inv(to_work) @ Hw
        confidence = float(score)
    return CourtCalibration(
        court_to_image=H, image_size=image_size, confidence=confidence, manual=True
    )


def line_pixel_mask(img_bgr: np.ndarray, tau: int = 4) -> np.ndarray:
    """Binary mask of thin, bright, unsaturated pixels (painted court lines)."""
    # Luminance, not HSV value: a bright green floor has a high V but a modest
    # luminance, so white paint stands out far more clearly in luminance.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    d = 18

    def brighter_than_neighbours(axis: int) -> np.ndarray:
        before = np.roll(gray, tau, axis=axis)
        after = np.roll(gray, -tau, axis=axis)
        return ((gray - before) > d) & ((gray - after) > d)

    thin = brighter_than_neighbours(0) | brighter_than_neighbours(1)
    mask = thin & (gray > 150) & (sat < 110)
    mask[:tau, :] = mask[-tau:, :] = False
    mask[:, :tau] = mask[:, -tau:] = False
    return mask.astype(np.uint8) * 255


def floor_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Region of the playing floor, found from its dominant colour.

    Court lines are painted on the floor, so restricting line pixels to this
    region removes advertising boards, scoreboards and the crowd, which
    otherwise contribute many false straight lines. Falls back to the whole
    image when no dominant floor colour is found.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # The court almost always fills the bottom-centre of a broadcast frame.
    sample = hsv[int(h * 0.55) : int(h * 0.95), int(w * 0.25) : int(w * 0.75)]
    coloured = sample[(sample[..., 1] > 50) & (sample[..., 2] > 50)]
    if len(coloured) < 0.2 * sample.shape[0] * sample.shape[1]:
        return np.full((h, w), 255, np.uint8)
    hist = np.bincount(coloured[:, 0], minlength=180).astype(np.float64)
    hist = np.convolve(np.r_[hist[-5:], hist, hist[:5]], np.ones(5), "same")[5:-5]
    hue = int(np.argmax(hist))

    diff = np.abs(hsv[..., 0].astype(np.int16) - hue)
    diff = np.minimum(diff, 180 - diff)
    floor = ((diff < 12) & (hsv[..., 1] > 40) & (hsv[..., 2] > 40)).astype(np.uint8) * 255
    # Close gaps left by lines, players and logos, then keep the largest region.
    floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(floor)
    if n <= 1:
        return np.full((h, w), 255, np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 0.1 * h * w:
        return np.full((h, w), 255, np.uint8)
    region = (labels == largest).astype(np.uint8) * 255
    # Fill holes (logos, players) and grow slightly so outer lines are included.
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return cv2.dilate(filled, np.ones((15, 15), np.uint8))


# --------------------------------------------------------------------------- #
# Candidate lines
# --------------------------------------------------------------------------- #


def _candidate_lines(mask: np.ndarray) -> list[_Line]:
    segs = cv2.HoughLinesP(
        mask, rho=1, theta=np.pi / 360, threshold=50, minLineLength=40, maxLineGap=8
    )
    if segs is None:
        return []
    segs = np.asarray(segs).reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    order = np.argsort(-lengths)

    clusters: list[dict] = []
    for i in order:
        x1, y1, x2, y2 = segs[i]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        for c in clusters:
            da = abs(angle - c["angle"])
            da = min(da, 180 - da)
            if da < 2.0 and _point_line_dist(mid, c["coef"]) < 4.0:
                c["pts"].extend([(x1, y1), (x2, y2)])
                c["support"] += lengths[i]
                break
        else:
            coef = _normalise(np.cross([x1, y1, 1.0], [x2, y2, 1.0]))
            clusters.append(
                {"coef": coef, "angle": angle, "pts": [(x1, y1), (x2, y2)], "support": lengths[i]}
            )

    ys, xs = np.nonzero(mask)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    lines: list[_Line] = []
    for c in clusters:
        near = pixels[np.abs(pixels @ c["coef"][:2] + c["coef"][2]) < 2.5]
        if len(near) >= 20:
            vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
            coef = _normalise(np.cross([x0[0], y0[0], 1.0], [x0[0] + vx[0], y0[0] + vy[0], 1.0]))
            angle = float(np.degrees(np.arctan2(vy[0], vx[0])) % 180)
        else:
            coef, angle = c["coef"], c["angle"]
        lines.append(_Line(coef=coef, support=float(c["support"]), angle=angle))

    # Fitting can make two clusters converge; drop near-duplicates.
    lines.sort(key=lambda ln: -ln.support)
    unique: list[_Line] = []
    for ln in lines:
        if not any(_same_line(ln, u) for u in unique):
            unique.append(ln)
    return unique


def _is_cross(line: _Line) -> bool:
    tilt = min(line.angle, 180 - line.angle)
    return tilt < CROSS_LINE_MAX_TILT_DEG


def _same_line(a: _Line, b: _Line) -> bool:
    da = abs(a.angle - b.angle)
    da = min(da, 180 - da)
    if da > 2.0:
        return False
    # Compare at a point on line a.
    p = _foot_of_origin(a.coef)
    return _point_line_dist(p, b.coef) < 3.0


def _foot_of_origin(coef: np.ndarray) -> np.ndarray:
    a, b, c = coef
    return np.array([-a * c, -b * c])


def _normalise(coef: np.ndarray) -> np.ndarray:
    return coef / np.hypot(coef[0], coef[1])


def _point_line_dist(p: np.ndarray, coef: np.ndarray) -> float:
    return float(abs(coef[0] * p[0] + coef[1] * p[1] + coef[2]))


# --------------------------------------------------------------------------- #
# Hypothesis search and scoring
# --------------------------------------------------------------------------- #


def _model_samples(spacing: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points along each painted line plus index pairs of neighbouring points.

    The pairs let us measure each point's share of projected line length in
    pixels, so the score counts *how much* paint a fit explains, not just the
    fraction. (A fraction alone lets a tiny "court" on a cluttered logo win.)
    """
    pts: list[tuple[float, float]] = []
    a_idx: list[int] = []
    b_idx: list[int] = []
    for seg in cm.court_segments():
        (x0, y0), (x1, y1) = seg.start, seg.end
        n = max(int(np.hypot(x1 - x0, y1 - y0) / spacing), 1)
        base = len(pts)
        for i in range(n + 1):
            u = i / n
            pts.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u))
        a_idx.extend(range(base, base + n))
        b_idx.extend(range(base + 1, base + n + 1))
    homog = np.hstack([np.asarray(pts), np.ones((len(pts), 1))]).T  # (3, N)
    return homog, np.asarray(a_idx), np.asarray(b_idx)


_MODEL_POINTS_H, _PAIR_A, _PAIR_B = _model_samples()


def _score_map(mask: np.ndarray) -> np.ndarray:
    """Soft version of the mask: 1 on paint, decaying over ~2 px."""
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    return np.exp(-(dist**2) / (2 * 2.0**2)).astype(np.float32)


def _evaluate(Hs: np.ndarray, score_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a batch of homographies (M, 3, 3) returns:

    - support: pixels of projected court line that lie on paint (higher is better)
    - fraction: support / total projected line length (the confidence, 0-1)

    Line outside the image counts towards the total but never as support, so
    fits that push the court out of frame are penalised.
    """
    h, w = score_map.shape
    proj = Hs @ _MODEL_POINTS_H  # (M, 3, N)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
    inside = (proj[:, 2] > 0) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    ui = np.where(inside, u, 0).astype(np.int32)
    vi = np.where(inside, v, 0).astype(np.int32)
    vals = np.where(inside, score_map[vi, ui], 0.0)

    seg_len = np.hypot(u[:, _PAIR_B] - u[:, _PAIR_A], v[:, _PAIR_B] - v[:, _PAIR_A])
    seg_len = np.nan_to_num(seg_len, nan=0.0, posinf=0.0)
    seg_val = (vals[:, _PAIR_A] + vals[:, _PAIR_B]) / 2
    support = (seg_len * seg_val).sum(axis=1)
    total = seg_len.sum(axis=1)
    fraction = np.where(total > 0, support / np.maximum(total, 1e-9), 0.0)
    return support, fraction


def _score(Hs: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    """Confidence (fraction of projected court line on paint) for a batch of homographies."""
    return _evaluate(Hs, score_map)[1]


def _search(
    cross: list[_Line], long_: list[_Line], score_map: np.ndarray, w: int, h: int
) -> tuple[np.ndarray | None, float]:
    cross = sorted(cross, key=lambda ln: -ln.support)[:MAX_CROSS_CANDIDATES]
    long_ = sorted(long_, key=lambda ln: -ln.support)[:MAX_LONG_CANDIDATES]
    # Far (top of image) first; left first.
    cross.sort(key=lambda ln: ln.y_at(w / 2))
    long_.sort(key=lambda ln: ln.x_at(h * 0.75))

    model_cross_pairs = [
        (far, near) for near, far in combinations(cm.CROSS_LINES_Y, 2)
    ]  # far > near in metres
    model_long_pairs = list(combinations(cm.LONG_LINES_X, 2))  # left < right

    src_batches: list[np.ndarray] = []
    dst_batches: list[np.ndarray] = []
    for i, j in combinations(range(len(cross)), 2):
        top, bottom = cross[i], cross[j]
        for k, m in combinations(range(len(long_)), 2):
            left, right = long_[k], long_[m]
            quad = [
                intersect(top.coef, left.coef),
                intersect(top.coef, right.coef),
                intersect(bottom.coef, right.coef),
                intersect(bottom.coef, left.coef),
            ]
            if any(p is None for p in quad):
                continue
            quad_arr = np.array(quad)
            if not is_convex_quad(quad_arr) or polygon_area(quad_arr) < 0.002 * w * h:
                continue
            if np.abs(quad_arr).max() > 4 * max(w, h):
                continue
            for y_far, y_near in model_cross_pairs:
                for x_left, x_right in model_long_pairs:
                    src_batches.append(
                        np.array(
                            [[x_left, y_far], [x_right, y_far], [x_right, y_near], [x_left, y_near]]
                        )
                    )
                    dst_batches.append(quad_arr)

    if not src_batches:
        return None, 0.0

    src = np.array(src_batches)
    dst = np.array(dst_batches)
    Hs = _batch_homographies(src, dst)
    valid = _plausible(Hs, w, h)
    Hs = Hs[valid]
    if len(Hs) == 0:
        return None, 0.0

    best_support, best_H = -1.0, None
    for start in range(0, len(Hs), 4000):
        batch = Hs[start : start + 4000]
        support, fraction = _evaluate(batch, score_map)
        support = np.where(fraction >= MIN_HYPOTHESIS_FRACTION, support, -1.0)
        k = int(np.argmax(support))
        if support[k] > best_support:
            best_support, best_H = float(support[k]), batch[k]
    if best_H is None or best_support <= 0:
        return None, 0.0
    return best_H, best_support


def _batch_homographies(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solves M four-point homographies at once (DLT with h33 = 1)."""
    M = len(src)
    A = np.zeros((M, 8, 8))
    b = np.zeros((M, 8))
    for i in range(4):
        X, Y = src[:, i, 0], src[:, i, 1]
        u, v = dst[:, i, 0], dst[:, i, 1]
        A[:, 2 * i, 0], A[:, 2 * i, 1], A[:, 2 * i, 2] = X, Y, 1
        A[:, 2 * i, 6], A[:, 2 * i, 7] = -u * X, -u * Y
        A[:, 2 * i + 1, 3], A[:, 2 * i + 1, 4], A[:, 2 * i + 1, 5] = X, Y, 1
        A[:, 2 * i + 1, 6], A[:, 2 * i + 1, 7] = -v * X, -v * Y
        b[:, 2 * i], b[:, 2 * i + 1] = u, v
    with np.errstate(all="ignore"):
        det = np.linalg.det(A)
        ok = np.abs(det) > 1e-9
        h = np.full((M, 8), np.nan)
        h[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
    H = np.concatenate([h, np.ones((M, 1))], axis=1).reshape(M, 3, 3)
    return H


def _plausible(Hs: np.ndarray, w: int, h: int) -> np.ndarray:
    """Rejects homographies that can't come from a camera behind the near baseline."""
    finite = np.isfinite(Hs).all(axis=(1, 2))
    corners = np.hstack([cm.CORNERS, np.ones((4, 1))]).T  # (3, 4)
    proj = np.einsum("mij,jk->mik", np.nan_to_num(Hs), corners)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
    in_front = (proj[:, 2] > 0).all(axis=1)
    nl, nr, fr, fl = 0, 1, 2, 3
    near_below_far = (v[:, nl] > v[:, fl]) & (v[:, nr] > v[:, fr])
    left_of_right = (u[:, nl] < u[:, nr]) & (u[:, fl] < u[:, fr])
    near_wider = (u[:, nr] - u[:, nl]) > (u[:, fr] - u[:, fl])

    # Sidelines run into the image (steep); baselines run across it (shallow).
    def tilt(a: int, b: int) -> np.ndarray:
        return np.degrees(np.arctan2(np.abs(v[:, b] - v[:, a]), np.abs(u[:, b] - u[:, a])))

    steep_sides = (tilt(nl, fl) > MIN_SIDELINE_TILT_DEG) & (tilt(nr, fr) > MIN_SIDELINE_TILT_DEG)
    flat_ends = (tilt(nl, nr) < CROSS_LINE_MAX_TILT_DEG) & (tilt(fl, fr) < CROSS_LINE_MAX_TILT_DEG)

    def length(a: int, b: int) -> np.ndarray:
        return np.hypot(u[:, b] - u[:, a], v[:, b] - v[:, a])

    # Degenerate fits collapse corners together; every side must have real length.
    sides_long_enough = (
        (length(nl, fl) > 0.15 * h)
        & (length(nr, fr) > 0.15 * h)
        & (length(nl, nr) > 0.15 * w)
        & (length(fl, fr) > 0.08 * w)
    )
    span_u = np.nanmax(u, axis=1) - np.nanmin(u, axis=1)
    span_v = np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
    big_enough = (span_u > 0.25 * w) & (span_v > 0.15 * h)
    not_huge = (np.abs(u) < 3 * w).all(axis=1) & (np.abs(v) < 3 * h).all(axis=1)
    return (
        finite
        & in_front
        & near_below_far
        & left_of_right
        & near_wider
        & steep_sides
        & flat_ends
        & sides_long_enough
        & big_enough
        & not_huge
    )


# --------------------------------------------------------------------------- #
# Refinement
# --------------------------------------------------------------------------- #


def _refine(H: np.ndarray, mask: np.ndarray, band: float = 4.0) -> np.ndarray | None:
    """Re-fits each model line to nearby paint and solves a least-squares homography."""
    ys, xs = np.nonzero(mask)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    h, w = mask.shape

    fitted: dict[tuple[str, float], np.ndarray] = {}
    for kind, values in (("y", cm.CROSS_LINES_Y), ("x", cm.LONG_LINES_X)):
        for val in values:
            if kind == "y":
                ends = np.array([[0.0, val], [cm.WIDTH, val]])
            else:
                ends = np.array([[val, 0.0], [val, cm.LENGTH]])
            p, q = apply_homography(H, ends)
            if not (np.isfinite(p).all() and np.isfinite(q).all()):
                continue
            coef = _normalise(np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0]))
            dist = np.abs(pixels @ coef[:2] + coef[2])
            # Keep pixels near the line and between its projected endpoints.
            d = q - p
            t = ((pixels - p) @ d) / (d @ d)
            near = pixels[(dist < band) & (t > -0.02) & (t < 1.02)]
            if kind == "x" and val == cm.WIDTH / 2:
                continue  # centre line is split in two; skip to keep fitting simple
            if len(near) < 25:
                continue
            vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
            fitted[(kind, val)] = np.cross([x0[0], y0[0], 1.0], [x0[0] + vx[0], y0[0] + vy[0], 1.0])

    src, dst = [], []
    for (k1, y), l1 in fitted.items():
        if k1 != "y":
            continue
        for (k2, x), l2 in fitted.items():
            if k2 != "x":
                continue
            p = intersect(l1, l2)
            if p is None or not (-w < p[0] < 2 * w and -h < p[1] < 2 * h):
                continue
            src.append([x, y])
            dst.append(p)
    if len(src) < 6:
        return None
    H_new, _ = cv2.findHomography(np.array(src), np.array(dst), 0)
    if H_new is None:
        return None
    return H_new / H_new[2, 2]
