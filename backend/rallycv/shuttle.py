"""Shuttle detection and tracking in image space.

Detection (per frame t, streaming):
  A pixel is a shuttle candidate if it is bright and *brighter* than the same
  pixel in both frame t-1 and frame t+1 (three-frame differencing). A white
  shuttle passing over a darker background satisfies this; static scenery,
  overlays and slowly moving players mostly don't. Connected components of
  this mask with a plausible size become candidates, restricted to the image
  of the 3D volume above the court and away from players' bodies.

Tracking:
  Candidates are linked frame-to-frame into short tracklets with a
  constant-velocity gate, then the best set of compatible tracklets is chosen
  by dynamic programming (longest consistent path through time). Short gaps
  are interpolated.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import court_model as cm
from .camera import Camera
from .court import CourtCalibration

WORK_WIDTH = 1280

MIN_BRIGHTNESS = 150
MIN_CONTRAST = 22
MIN_AREA = 1
MAX_AREA = 160
MAX_EXTENT = 36
"""Max bbox side in work pixels (a motion-blurred streak can be long)."""

MAX_CANDIDATES_PER_FRAME = 8
VOLUME_HEIGHT = 9.0
"""Shuttle search volume height above the floor, metres."""


@dataclass
class Candidate:
    frame: int
    x: float  # full-resolution pixels
    y: float
    score: float
    near_player: bool = False
    """Close to a player's box: plausible (rackets, hits) but often a false positive."""


@dataclass
class Tracklet:
    points: list[Candidate] = field(default_factory=list)
    misses: int = 0

    @property
    def start(self) -> int:
        return self.points[0].frame

    @property
    def end(self) -> int:
        return self.points[-1].frame

    def predict(self, frame: int) -> np.ndarray:
        last = self.points[-1]
        if len(self.points) < 2:
            return np.array([last.x, last.y])
        prev = self.points[-2]
        dt = max(last.frame - prev.frame, 1)
        v = np.array([last.x - prev.x, last.y - prev.y]) / dt
        return np.array([last.x, last.y]) + v * (frame - last.frame)

    def speed(self) -> float:
        if len(self.points) < 2:
            return 0.0
        a, b = self.points[-2], self.points[-1]
        return float(np.hypot(b.x - a.x, b.y - a.y) / max(b.frame - a.frame, 1))


class ShuttleDetector:
    """Streaming three-frame-difference detector. Feed every frame in order."""

    def __init__(self, calibration: CourtCalibration, camera: Camera | None) -> None:
        w, h = calibration.image_size
        self._scale = WORK_WIDTH / w
        self._size = (WORK_WIDTH, round(h * self._scale))
        self._roi = _search_region(calibration, camera, self._size, self._scale)
        self._window: list[tuple[int, np.ndarray]] = []
        self.candidates: dict[int, list[Candidate]] = {}
        self.distinct_frames: list[int] = []
        self.repeated_frames = 0

    @property
    def roi(self) -> np.ndarray:
        return self._roi

    def push(self, frame_index: int, frame_bgr: np.ndarray, player_boxes: list[np.ndarray]) -> None:
        small = cv2.resize(frame_bgr, self._size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # Re-encoded footage often repeats frames (e.g. 30 fps content at 60 fps).
        # Differencing a frame against its own copy shows no motion, so only
        # distinct frames enter the window; repeats are filled in later.
        if self._window and is_repeat(gray, self._window[-1][1]):
            self.repeated_frames += 1
            return
        self.distinct_frames.append(frame_index)
        self._window.append((frame_index, gray))
        if len(self._window) > 3:
            self._window.pop(0)
        if len(self._window) == 3:
            (_, prev), (idx, cur), (_, nxt) = self._window
            self.candidates[idx] = self._detect(idx, prev, cur, nxt, player_boxes)

    def _detect(
        self,
        idx: int,
        prev: np.ndarray,
        cur: np.ndarray,
        nxt: np.ndarray,
        player_boxes: list[np.ndarray],
    ) -> list[Candidate]:
        cur16 = cur.astype(np.int16)
        contrast = np.minimum(cur16 - prev, cur16 - nxt)
        mask = (contrast > MIN_CONTRAST) & (cur > MIN_BRIGHTNESS) & (self._roi > 0)
        mask = mask.astype(np.uint8)
        # Players' moving limbs, shirts and rackets are the main source of false
        # candidates, so blank each player's box (plus a small margin). The
        # shuttle is briefly lost at contact; hits are inferred from the flight
        # on either side.
        boxes_small = [box * self._scale for box in player_boxes]
        for x1, y1, x2, y2 in boxes_small:
            bw = x2 - x1
            mask[
                max(int(y1 - 0.05 * (y2 - y1)), 0) : max(int(y2 + 4), 0),
                max(int(x1 - 0.1 * bw), 0) : max(int(x2 + 0.1 * bw), 0),
            ] = 0

        n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out: list[Candidate] = []
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            if not (MIN_AREA <= area <= MAX_AREA) or max(bw, bh) > MAX_EXTENT:
                continue
            cx, cy = centroids[k]
            peak = float(contrast[int(cy), int(cx)]) if mask[int(cy), int(cx)] else MIN_CONTRAST
            out.append(
                Candidate(
                    frame=idx,
                    x=cx / self._scale,
                    y=cy / self._scale,
                    score=peak + area,
                    near_player=_near_any(cx, cy, boxes_small),
                )
            )
        out.sort(key=lambda c: -c.score)
        return out[:MAX_CANDIDATES_PER_FRAME]


def _near_any(x: float, y: float, boxes: list[np.ndarray]) -> bool:
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        if x1 - 0.6 * bw <= x <= x2 + 0.6 * bw and y1 - 0.5 * bh <= y <= y2:
            return True
    return False


def is_repeat(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two frames are (near-)identical copies.

    Counts changed pixels rather than the mean difference: with small players
    in a static broadcast shot, genuinely new frames differ in only a few
    hundred pixels, so a mean is near zero for both repeats and real frames.
    """
    changed = cv2.absdiff(a[::2, ::2], b[::2, ::2]) > 12
    return int(changed.sum()) < 40


def _search_region(
    cal: CourtCalibration, camera: Camera | None, size: tuple[int, int], scale: float
) -> np.ndarray:
    """Image region where the shuttle can be: the volume above the court (+margin)."""
    w, h = size
    mx, my = 1.5, 2.0
    floor = np.array(
        [[-mx, -my], [cm.WIDTH + mx, -my], [cm.WIDTH + mx, cm.LENGTH + my], [-mx, cm.LENGTH + my]]
    )
    pts = list(cal.to_image(floor) * scale)
    if camera is not None:
        top = np.c_[floor, np.full(4, VOLUME_HEIGHT)]
        pts += list(camera.project(top) * scale)
    else:  # no 3D camera: extend the floor polygon to the top of the image
        pts += [[p[0], 0] for p in pts]
    hull = cv2.convexHull(np.array(pts, dtype=np.float32)).astype(np.int32)
    roi = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(roi, hull, 255)
    return roi


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #


@dataclass
class ShuttleTrack:
    """Image-space shuttle track: frame -> (x, y) full-res pixels, NaN where unseen."""

    frames: np.ndarray
    xy: np.ndarray  # (N, 2), NaN for missing

    def visible(self) -> np.ndarray:
        return ~np.isnan(self.xy[:, 0])


def build_tracklets(
    candidates: dict[int, list[Candidate]],
    fps: float,
    image_width: int,
    *,
    min_points: int = 4,
    require_motion: bool = True,
) -> list[Tracklet]:
    """Greedy frame-to-frame linking with a constant-velocity gate."""
    px_scale = image_width / 1920
    base_gate = 25 * px_scale * (60 / fps) ** 0.5
    max_misses = max(2, round(fps * 0.12))
    active: list[Tracklet] = []
    finished: list[Tracklet] = []

    for frame in sorted(candidates):
        cands = list(candidates[frame])
        # Associate best-predicted tracklets first.
        active.sort(key=lambda t: -len(t.points))
        for tr in active:
            if not cands:
                break
            pred = tr.predict(frame)
            gate = base_gate + 0.6 * tr.speed() * (frame - tr.end)
            d = [np.hypot(c.x - pred[0], c.y - pred[1]) for c in cands]
            k = int(np.argmin(d))
            if d[k] <= gate:
                tr.points.append(cands.pop(k))
                tr.misses = 0
            else:
                tr.misses += 1
        still: list[Tracklet] = []
        for tr in active:
            if tr.end < frame:
                tr.misses = frame - tr.end
            (finished if tr.misses > max_misses else still).append(tr)
        active = still
        active.extend(Tracklet(points=[c]) for c in cands)

    finished.extend(active)
    return [
        t
        for t in finished
        if len(t.points) >= min_points and (not require_motion or _is_moving_tracklet(t, px_scale))
    ]


def _is_moving_tracklet(t: Tracklet, px_scale: float) -> bool:
    if len(t.points) < 4:
        return False
    xy = np.array([[p.x, p.y] for p in t.points])
    path = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
    span = np.linalg.norm(xy[-1] - xy[0])
    # Must actually travel (a flying shuttle covers well over 60 px in a few
    # frames; crowd and banner motion mostly doesn't), and travel reasonably
    # straight rather than jitter in place.
    return span > 60 * px_scale and span > 0.35 * path


def select_tracklets(tracklets: list[Tracklet], fps: float, image_width: int) -> list[Tracklet]:
    """Chooses non-overlapping tracklets maximising (weighted) covered frames.

    There is only one shuttle, so selected tracklets may not overlap in time.
    Across short gaps the jump between consecutive tracklets must also be
    physically plausible; across longer gaps (the shuttle hidden by a player
    or lost against a busy background) any continuation is allowed.
    """
    if not tracklets:
        return []
    px_scale = image_width / 1920
    short_gap = max(2, int(fps * 0.25))
    max_speed = 70 * px_scale * (60 / fps)  # px per frame
    ts = sorted(tracklets, key=lambda t: t.start)
    n = len(ts)
    weight = [_tracklet_weight(t) for t in ts]
    best = list(weight)
    parent = [-1] * n
    for j in range(n):
        for i in range(n):
            a, b = ts[i], ts[j]
            gap = b.start - a.end
            if gap <= 0:
                continue
            if gap <= short_gap:
                dist = np.hypot(b.points[0].x - a.points[-1].x, b.points[0].y - a.points[-1].y)
                if dist > max_speed * gap + 30 * px_scale:
                    continue
            if best[i] + weight[j] > best[j]:
                best[j] = best[i] + weight[j]
                parent[j] = i
    j = int(np.argmax(best))
    chain = []
    while j != -1:
        chain.append(ts[j])
        j = parent[j]
    return chain[::-1]


def _tracklet_weight(t: Tracklet) -> float:
    """Frames covered, discounting points that hug a player (likely limbs/rackets)."""
    return float(sum(0.2 if p.near_player else 1.0 for p in t.points))


def track_shuttle(
    candidates: dict[int, list[Candidate]],
    frame_count: int,
    fps: float,
    image_width: int,
    *,
    learned: bool = False,
) -> ShuttleTrack:
    """Builds the rally's shuttle track from per-frame candidates.

    `learned=True` is for TrackNet detections: one confident detection per
    frame, so short tracklets are kept and a resting shuttle (landing) isn't
    discarded as "not moving".
    """
    tracklets = build_tracklets(
        candidates,
        fps,
        image_width,
        min_points=3 if learned else 4,
        require_motion=not learned,
    )
    chain = select_tracklets(tracklets, fps, image_width)
    xy = np.full((frame_count, 2), np.nan)
    for tr in chain:
        for p in tr.points:
            if 0 <= p.frame < frame_count:
                xy[p.frame] = (p.x, p.y)
    _interpolate_short_gaps(xy, max_gap=max(2, round(fps * 0.15)))
    return ShuttleTrack(frames=np.arange(frame_count), xy=xy)


def _interpolate_short_gaps(xy: np.ndarray, max_gap: int) -> None:
    seen = np.nonzero(~np.isnan(xy[:, 0]))[0]
    for a, b in zip(seen, seen[1:], strict=False):
        if 1 < b - a <= max_gap + 1:
            for k in range(a + 1, b):
                u = (k - a) / (b - a)
                xy[k] = xy[a] + (xy[b] - xy[a]) * u
