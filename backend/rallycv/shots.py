"""Hit, landing and rally detection from the image-space shuttle track.

With a dense track (TrackNet), a hit shows up as a cusp: the shuttle arrives
at a player and leaves in a clearly different direction (or, for an overhead
taken at the top of the flight, sharply changes speed). A landing is the
shuttle coming to rest on the floor; it ends the rally. Hard cuts in the
video (highlight reels splice rallies together) also end a rally.

Who hit is decided per rally, not per hit: in singles the hitters strictly
alternate, so the A/B sequence is chosen jointly (dynamic programming) to
best agree with which player could reach the shuttle at each hit, allowing
for an occasional missed hit (long gap) or spurious one (dropped).

A rally's first hit is only a serve when the rally was seen to start: the
shuttle was not already in flight before it. A clip that begins mid-rally
does not get a fake serve.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from . import court_model as cm
from .court import CourtCalibration
from .players import PlayerTracks
from .shuttle import ShuttleTrack

MIN_SHOT_SEC = 0.3
"""Two contacts closer than this are the same hit."""
DEAD_TIME_SEC = 1.2
"""After a landing, shuttle movement this soon is players handing the shuttle back."""
RALLY_BREAK_SEC = 3.0
"""No hit for this long ends a rally even without a detected landing."""
BOUNCE_WINDOW_SEC = 0.25
"""Direction changes this close to a landing are the shuttle bouncing, not a hit."""
MAX_REACH = 0.8
"""How far outside a player's reach region (in box units) a hit may be."""
MIN_CUSP_ANGLE = 55.0

MISSED_HIT_MIN_SEC = 0.8
"""Two consecutive hits by the same player this far apart imply the opponent's
hit in between was missed; closer than this, one of them is spurious."""
MISSED_HIT_COST = 0.8
UNREACHABLE_COST = 2.0
"""Attribution cost when a player's box is unknown or far from the shuttle."""
SERVE_LOOKBACK_SEC = 0.8
"""A serve needs this long (or the whole rally window) without the shuttle in flight."""


@dataclass
class Hit:
    frame: float
    player: str  # 'A' (near) or 'B' (far)
    strength: float


@dataclass
class Rally:
    hits: list[Hit]
    end_frame: float
    """When the final shot ended: the landing, or the last time the shuttle was seen."""
    landed: bool
    served: bool = True
    """Whether the first hit is a serve (False when the clip starts mid-rally)."""


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def reach_distance(xy: np.ndarray, box: np.ndarray) -> float:
    """0 inside a player's reach region, growing with distance (in box units)."""
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    rx1, rx2 = x1 - 0.9 * bw, x2 + 0.9 * bw
    ry1, ry2 = y1 - 0.6 * bh, y2
    dx = max(rx1 - xy[0], 0.0, xy[0] - rx2) / bw
    dy = max(ry1 - xy[1], 0.0, xy[1] - ry2) / bh
    return float(np.hypot(dx, dy))


def _reach_costs(
    xy: np.ndarray, frame: int, players: PlayerTracks, max_gap: int
) -> dict[str, float]:
    """Per player: how implausible it is that they struck the shuttle at `xy`.

    The reach distance, plus a small tie-breaker for distance to the box centre
    (when both players are within reach, e.g. at the net).
    """
    costs: dict[str, float] = {}
    for p in ("A", "B"):
        box = players.box_at(p, frame, max_gap)
        if box is None:
            costs[p] = UNREACHABLE_COST
            continue
        d = reach_distance(xy, box)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        centre = np.hypot(xy[0] - cx, xy[1] - cy) / max(box[3] - box[1], 1.0)
        costs[p] = min(d + 0.1 * min(centre, 2.0), UNREACHABLE_COST)
    return costs


def _runs(visible: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(visible):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(visible) - 1))
    return runs


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #


@dataclass
class _Event:
    frame: float
    kind: str  # 'hit' or 'landing'
    player: str | None
    """Most likely hitter; the final choice is made per rally (see _assign_hitters)."""
    strength: float
    costs: dict[str, float] = field(default_factory=dict)
    """Per-player attribution cost (see _reach_costs)."""


def _hit_event(frame: float, strength: float, costs: dict[str, float]) -> _Event:
    player = min(costs, key=costs.__getitem__) if costs else None
    return _Event(frame, "hit", player, strength, costs)


def _cusp_events(
    track: ShuttleTrack, players: PlayerTracks, fps: float, px_scale: float
) -> list[_Event]:
    xy = track.xy
    w = max(2, round(0.08 * fps))
    min_speed = 1.5 * px_scale * (60 / fps)
    max_gap = int(fps * 0.25)
    events: list[_Event] = []
    for s, e in _runs(track.visible()):
        if e - s < 2 * w + 1:
            continue
        seg = xy[s : e + 1]
        for i in range(w, len(seg) - w):
            v_in = (seg[i] - seg[i - w]) / w
            v_out = (seg[i + w] - seg[i]) / w
            n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
            if max(n_in, n_out) < min_speed:
                continue
            ratio = max(n_in, n_out) / max(min(n_in, n_out), 1e-6)
            # An overhead taken near the top of the flight, or while the shuttle
            # falls, barely changes its image direction but kicks its speed up
            # several-fold at once, after which drag slows it. (Falling after
            # the apex speeds up gradually and keeps speeding up.)
            later = seg[min(i + 2 * w, len(seg) - 1)] - seg[i + w]
            kicked = (
                n_out > 8 * min_speed
                and n_out > 2.5 * n_in
                and np.linalg.norm(later) / w < 1.1 * n_out
            )
            if min(n_in, n_out) < min_speed:
                if not kicked:
                    continue
                angle = 45.0
            else:
                cos = float(np.clip(v_in @ v_out / (n_in * n_out), -1, 1))
                angle = float(np.degrees(np.arccos(cos)))
                if angle < MIN_CUSP_ANGLE and not (angle > 25 and ratio > 2.5) and not kicked:
                    continue
            frame = s + i
            costs = _reach_costs(seg[i], frame, players, max_gap)
            reach = min(costs.values())
            if reach > MAX_REACH:
                continue
            strength = angle / 180 + min(ratio, 4) / 8 + (MAX_REACH - reach)
            events.append(_hit_event(float(frame), strength, costs))
    return events


def _gap_events(
    track: ShuttleTrack, players: PlayerTracks, fps: float, px_scale: float = 1.0
) -> list[_Event]:
    """A run that starts next to a player after a gap: the hit happened in the gap.

    The shuttle must leave quickly; one lying or rolling on the floor (or being
    picked up after the point) is not a hit.
    """
    events: list[_Event] = []
    max_gap = int(fps * 0.25)
    w = max(2, round(0.08 * fps))
    min_flight_speed = 6.0 * px_scale * (60 / fps)
    runs = [r for r in _runs(track.visible()) if r[1] - r[0] >= 3]
    for k, (s, e) in enumerate(runs):
        gap = s - runs[k - 1][1] if k > 0 else s
        if gap < 0.12 * fps:
            continue
        n = min(w, e - s)
        if np.linalg.norm(track.xy[s + n] - track.xy[s]) / n < min_flight_speed:
            continue
        costs = _reach_costs(track.xy[s], s, players, max_gap)
        reach = min(costs.values())
        if reach > 0.4:
            continue
        lead = min(gap / 2, 0.1 * fps)
        events.append(_hit_event(float(s - lead), 0.6 - reach, costs))
    return events


def _landing_events(
    track: ShuttleTrack, calibration: CourtCalibration, fps: float, px_scale: float
) -> list[_Event]:
    """The shuttle at rest on the floor (on or near the court) for a moment.

    A shuttle that lands often bounces a little, and detections jitter by a few
    pixels, so "at rest" allows ~8 px of movement and "came down" checks that
    it fell from well above within the previous ~0.6 s rather than just before.
    """
    events: list[_Event] = []
    still_frames = max(3, round(0.15 * fps))
    still_px = 8 * px_scale
    drop_px = 40 * px_scale
    lookback = round(0.6 * fps)
    xy = track.xy
    i = 0
    n = len(xy)
    while i + still_frames <= n:
        window = xy[i : i + still_frames]
        if (
            np.isnan(window).any()
            or np.ptp(window[:, 0]) > still_px
            or np.ptp(window[:, 1]) > still_px
        ):
            i += 1
            continue
        before = xy[max(0, i - lookback) : i]
        before = before[~np.isnan(before[:, 1])]
        came_down = len(before) > 0 and window[0, 1] - before[:, 1].min() > drop_px
        floor = calibration.to_court(window.mean(axis=0)[None])[0]
        on_floor = bool(np.isfinite(floor).all()) and (
            -1.5 < floor[0] < cm.WIDTH + 1.5 and -2.0 < floor[1] < cm.LENGTH + 2.0
        )
        if came_down and on_floor:
            # The landing is the first contact: the lowest point just before rest.
            seg = xy[max(0, i - lookback) : i + 1]
            ys = np.where(np.isnan(seg[:, 1]), -np.inf, seg[:, 1])
            first_contact = max(0, i - lookback) + int(np.argmax(ys))
            events.append(_Event(float(first_contact), "landing", None, 1.0))
            # Skip past this resting stretch.
            j = i + still_frames
            while j < n and not np.isnan(xy[j, 1]) and abs(xy[j, 1] - window[-1, 1]) < still_px:
                j += 1
            i = j
            continue
        i += 1
    return events


# --------------------------------------------------------------------------- #
# Rally assembly
# --------------------------------------------------------------------------- #


def detect_rallies(
    track: ShuttleTrack,
    players: PlayerTracks,
    calibration: CourtCalibration,
    fps: float,
    image_width: int,
    cuts: Sequence[float] = (),
) -> list[Rally]:
    """Rallies in the clip. `cuts` are frames where the video jumps (edited
    footage); play never continues across a cut."""
    px_scale = image_width / 1920
    visible = track.visible()
    if visible.sum() < 3:
        return []

    events = (
        _cusp_events(track, players, fps, px_scale)
        + _gap_events(track, players, fps, px_scale)
        + _landing_events(track, calibration, fps, px_scale)
    )
    landings = sorted(e.frame for e in events if e.kind == "landing")
    # The bounce of a landing changes direction too; it is not a hit.
    bounce = BOUNCE_WINDOW_SEC * fps
    hit_events = [
        e for e in events if e.kind == "hit" and all(abs(e.frame - f) > bounce for f in landings)
    ]
    hits = _suppress(hit_events, MIN_SHOT_SEC * fps)

    rallies: list[Rally] = []
    bounds = [0.0, *sorted(c for c in cuts if 0 < c < len(track.xy)), float(len(track.xy))]
    for start, end in zip(bounds, bounds[1:], strict=False):
        seg_hits = [h for h in hits if start <= h.frame < end]
        seg_landings = [f for f in landings if start <= f < end]
        rallies += _assemble_rallies(seg_hits, seg_landings, track, fps, start, end, px_scale)
    return rallies


def _assemble_rallies(
    hits: list[_Event],
    landings: list[float],
    track: ShuttleTrack,
    fps: float,
    seg_start: float = 0.0,
    seg_end: float | None = None,
    px_scale: float = 1.0,
) -> list[Rally]:
    """Groups time-ordered hits of one continuous stretch of video into rallies,
    closed by landings, long pauses or the end of the stretch."""
    seg_end = float(len(track.xy)) if seg_end is None else seg_end
    groups: list[tuple[list[_Event], float, float | None]] = []  # hits, window start, landing
    current: list[_Event] = []
    window_start = seg_start
    blocked_until = -np.inf
    li = 0
    for ev in hits:
        # Close the rally at any landing that happens before this hit.
        while li < len(landings) and landings[li] < ev.frame:
            if current:
                groups.append((current, window_start, landings[li]))
                current = []
                blocked_until = landings[li] + DEAD_TIME_SEC * fps
                window_start = blocked_until
            li += 1
        if ev.frame < blocked_until:
            continue
        if current and ev.frame - current[-1].frame > RALLY_BREAK_SEC * fps:
            groups.append((current, window_start, None))
            window_start = current[-1].frame + RALLY_BREAK_SEC * fps / 2
            current = []
        current.append(ev)
    if current:
        groups.append((current, window_start, landings[li] if li < len(landings) else None))

    rallies: list[Rally] = []
    for group, w_start, landing in groups:
        seq = _assign_hitters(group, fps)
        if landing is not None:
            end, landed = landing, True
        else:
            end, landed = _last_seen_after(track, seq[-1].frame, seg_end), False
        # A rally must end after its last hit. Trim inconsistent trailing hits
        # rather than discarding the whole rally.
        seq = [h for h in seq if h.frame < end]
        if seq:
            served = _starts_with_serve(track, seq[0].frame, w_start, fps, px_scale)
            rallies.append(Rally(seq, end, landed, served))
    return rallies


def _last_seen_after(track: ShuttleTrack, frame: float, limit: float | None = None) -> float:
    vis = np.nonzero(track.visible())[0]
    after = vis[(vis > frame) & (vis < (limit if limit is not None else np.inf))]
    if len(after) == 0:
        return frame + 1
    # End of the first visible run after the hit.
    end = after[0]
    for f in after[1:]:
        if f - end > 3:
            break
        end = f
    return float(end)


def _starts_with_serve(
    track: ShuttleTrack, first_hit: float, window_start: float, fps: float, px_scale: float
) -> bool:
    """Whether the rally was seen to start: no shuttle flying before its first hit.

    Looks back up to SERVE_LOOKBACK_SEC (not before the rally window: the start
    of the clip, a cut or the end of the previous rally). Before a serve the
    shuttle is in the server's hand, so it is either unseen or barely moving.
    """
    lo = int(max(window_start, first_hit - SERVE_LOOKBACK_SEC * fps))
    hi = int(first_hit - 0.1 * fps)
    if hi - lo < 0.25 * fps and window_start <= 0:
        return False  # the clip starts right at this hit: can't tell
    xy = track.xy[max(lo, 0) : max(hi, 0)]
    if len(xy) < 2:
        return True
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    flying = step[np.isfinite(step)] > 4.0 * px_scale * (60 / fps)
    return int(flying.sum()) < max(3, round(0.1 * fps))


def _suppress(events: list[_Event], window: float) -> list[_Event]:
    """Non-maximum suppression in time: keep the strongest event in each window."""
    out: list[_Event] = []
    for ev in sorted(events, key=lambda e: -e.strength):
        if all(abs(ev.frame - o.frame) >= window for o in out):
            out.append(ev)
    return sorted(out, key=lambda e: e.frame)


def _assign_hitters(events: list[_Event], fps: float) -> list[Hit]:
    """Chooses who hit each detected contact, jointly for the rally.

    Hitters alternate. Each detected hit is either kept (as A or B, costing its
    attribution cost) or dropped as spurious (costing more the stronger it
    was). Two consecutive kept hits by the same player are allowed only far
    enough apart that the opponent's hit in between could have been missed;
    that hit is then inserted. Minimum total cost wins.
    """
    n = len(events)
    if n == 0:
        return []
    other = {"A": "B", "B": "A"}
    drop = [1.0 + max(e.strength, 0.0) for e in events]
    drop_prefix = np.concatenate([[0.0], np.cumsum(drop)])  # cost of dropping events[a:b]

    def emit(i: int, p: str) -> float:
        return events[i].costs.get(p, 0.0 if events[i].player == p else UNREACHABLE_COST)

    def trans(j: int, q: str, i: int, p: str) -> float:
        if p != q:
            return 0.0
        gap = events[i].frame - events[j].frame
        return MISSED_HIT_COST if gap >= MISSED_HIT_MIN_SEC * fps else np.inf

    INF = np.inf
    best = {(i, p): INF for i in range(n) for p in "AB"}
    parent: dict[tuple[int, str], tuple[int, str] | None] = {}
    for i in range(n):
        for p in "AB":
            c, par = drop_prefix[i], None  # first kept hit
            for j in range(i):
                for q in "AB":
                    if best[(j, q)] == INF:
                        continue
                    t = trans(j, q, i, p)
                    if t == INF:
                        continue
                    cand = best[(j, q)] + t + drop_prefix[i] - drop_prefix[j + 1]
                    if cand < c:
                        c, par = cand, (j, q)
            best[(i, p)] = c + emit(i, p)
            parent[(i, p)] = par

    end = min(best, key=lambda k: best[k] + drop_prefix[n] - drop_prefix[k[0] + 1])
    chain: list[tuple[int, str]] = []
    node: tuple[int, str] | None = end
    while node is not None:
        chain.append(node)
        node = parent[node]
    chain.reverse()

    out: list[Hit] = []
    for i, p in chain:
        ev = events[i]
        if out and out[-1].player == p:
            # The opponent's hit was missed; place it halfway.
            out.append(Hit((out[-1].frame + ev.frame) / 2, other[p], 0.0))
        out.append(Hit(ev.frame, p, ev.strength))
    return out


# --------------------------------------------------------------------------- #
# Shot classification
# --------------------------------------------------------------------------- #


@dataclass
class ShotFeatures:
    """What is actually measured about one shot (see mapping.py)."""

    serve: bool
    """First hit of a rally that was seen to start."""
    start: np.ndarray
    """Hitter's floor position at contact, metres."""
    end: np.ndarray
    """Receiver's floor position at their contact, or where the shuttle landed."""
    duration: float
    """Seconds from this contact to the next contact (or the landing)."""
    rise: float | None
    """How far the shuttle climbed in the image above both ends of the flight,
    as a fraction of the image height (None if barely seen). A high, looping
    flight (clear, lift) rises a lot; a flat or downward one (drive, smash)
    hardly at all."""
    overhead: bool | None
    """Contact above the hitter's head (None if unknown)."""
    received_high: bool | None = None
    """The receiver had to play it above their head (None if unknown or it landed)."""


HIGH_RISE = 0.12
FLAT_RISE = 0.05


def depth(p: np.ndarray) -> float:
    """Distance from the net within its half: 0 at the net, 1 at the baseline."""
    return float(abs(p[1] - cm.NET_Y) / cm.NET_Y)


def classify_shot(f: ShotFeatures) -> str:
    """Shot type from where it was hit from and to, flight time, arc and stroke.

    Reference (singles): a smash is a fast overhead from mid/rear court; a drop
    an overhead to the opponent's front court; a clear a high shot rear to
    rear; a lift a high shot from the front/mid court to the rear; a net shot
    front to front; a drive a flat, fast shot through the middle.
    """
    if f.serve:
        return "serve"
    d0, d1, t = depth(f.start), depth(f.end), f.duration
    rise = f.rise
    # A slow shot the receiver must play overhead was lofted, even if the
    # image arc looks flat (an early interception high up hides the loop).
    high = (rise is not None and rise > HIGH_RISE) or (bool(f.received_high) and t > 0.75)
    flat = rise is not None and rise < FLAT_RISE and not high
    overhead = bool(f.overhead)

    def lofted() -> str:
        # Overhead from the back court is a clear; an underarm one is a lift.
        if f.overhead is None:
            return "clear" if d0 >= 0.55 else "lift"
        return "clear" if overhead and d0 >= 0.45 else "lift"

    def to_front() -> str:
        return "net" if d0 < 0.5 and not overhead else "drop"

    if d0 < 0.4 and d1 < 0.45 and not high and t < 1.3:
        return "net"
    if high or (rise is None and t > 1.2):
        return lofted()
    if overhead and d0 >= 0.3 and t < 0.75:
        return "smash"
    if d1 < 0.5 and (overhead or d0 >= 0.5):
        return "drop"
    if flat or t < 0.8:
        return "drive" if d1 >= 0.45 else to_front()
    if d1 < 0.5:
        return to_front()
    return lofted()
