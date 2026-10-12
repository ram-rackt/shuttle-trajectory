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
from .camera import Camera, intersect_vertical_plane
from .court import CourtCalibration
from .players import PlayerTracks
from .shuttle import ShuttleTrack

MIN_SHOT_SEC = 0.3
"""Two contacts closer than this are the same hit."""
DEAD_TIME_SEC = 1.2
"""After a landing, shuttle movement this soon is players handing the shuttle back."""
RALLY_BREAK_SEC = 3.0
"""No hit for this long ends a rally even without a detected landing. It also
bounds the rally's last shot: a landing (or net contact) longer than this after
the last hit is dead time, e.g. the shuttle tossed back after an unseen landing."""
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
MIN_SERVE_GAP_SEC = 0.5
"""A serve comes at least this long after the previous rally ended (1.3 s or more
in real footage, even in highlight reels). Play that seems to restart sooner is
play carrying on, e.g. a replay cutting in mid-rally, so it isn't a serve."""
NET_STALL_MAX_SPEED = 6.0
"""px per 1/60 s at 1080p: after striking the net the shuttle all but stops."""
NET_ARRIVAL_MIN_SPEED = 4.0
"""px per 1/60 s at 1080p: it was in flight when it got there (a soft shot into
the net arrives slowly, but a shuttle lying on the floor does not move at all)."""
NET_LEAD_SEC = 0.1
"""No racket contact this close before the shuttle meets the net."""
NET_SETTLE_SEC = 0.25
"""After striking the net the shuttle stays slow at least this long (it hangs in
the net or drops from tape height). A shot that only seems to pause for a frame
or two (a jittery or repeated detection) flies on at full speed straight after."""


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
    net_point: np.ndarray | None = None
    """Where the last shot struck the net, floor (X, Y) metres, if it ended there."""


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
    rest_frame: float | None = None
    """Landings: when the shuttle was first seen at rest (`frame` is the earlier
    moment it is judged to have touched down)."""


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
    track: ShuttleTrack,
    players: PlayerTracks,
    fps: float,
    px_scale: float = 1.0,
    start_frame: float = 0.0,
) -> list[_Event]:
    """A run that starts next to a player after a gap: the hit happened in the gap.

    The shuttle must leave quickly; one lying or rolling on the floor (or being
    picked up after the point) is not a hit. The first run's gap is counted from
    `start_frame`, where the analysed video begins.
    """
    events: list[_Event] = []
    max_gap = int(fps * 0.25)
    w = max(2, round(0.08 * fps))
    min_flight_speed = 6.0 * px_scale * (60 / fps)
    runs = [r for r in _runs(track.visible()) if r[1] - r[0] >= 3]
    for k, (s, e) in enumerate(runs):
        gap = s - (runs[k - 1][1] if k > 0 else start_frame)
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


def _net_contacts(
    track: ShuttleTrack, camera: Camera | None, fps: float, px_scale: float
) -> list[tuple[float, np.ndarray]]:
    """The shuttle striking the net: (frame, floor point on the net line).

    In flight, on the plane of the net and below its tape, the shuttle all but
    stops and then drops. A racket does the opposite: the shuttle leaves fast.
    Needs the 3D camera (to place the shuttle on the net plane); without one,
    nothing is reported.
    """
    if camera is None:
        return []
    xy = track.xy
    w = max(2, round(0.08 * fps))
    settle = max(w + 1, round(NET_SETTLE_SEC * fps))
    per_frame = px_scale * (60 / fps)
    found: list[tuple[float, np.ndarray]] = []
    last = -np.inf
    # Nothing before the first detection can qualify (e.g. a window late in a long video).
    for i in range(_first_seen(track) + w, len(xy) - w):
        before, at, after = xy[i - w], xy[i], xy[i + w]
        if not (np.isfinite(before).all() and np.isfinite(at).all() and np.isfinite(after).all()):
            continue
        v_in = np.linalg.norm(at - before) / w
        v_out = np.linalg.norm(after - at) / w
        if v_out >= NET_STALL_MAX_SPEED * per_frame or v_in < NET_ARRIVAL_MIN_SPEED * per_frame:
            continue
        if v_in < 2.5 * max(v_out, 0.5 * per_frame):
            continue
        later = xy[min(i + 3 * w, len(xy) - 1)]
        if not np.isfinite(later).all():
            later = after
        dx, dy = later - at
        if not (dy > 0 and dy >= abs(dx)):
            continue  # it must drop, not fly on
        if not _stays_slow(xy, i, settle, NET_STALL_MAX_SPEED * per_frame):
            continue
        pt, _ = intersect_vertical_plane(
            camera, at[None], np.array([0.0, cm.NET_Y]), np.array([cm.WIDTH, cm.NET_Y])
        )
        x, _, z = pt[0]
        if not (-0.3 <= x <= cm.WIDTH + 0.3 and 0 <= z <= cm.NET_HEIGHT_POSTS + 0.15):
            continue
        if i - last > 0.5 * fps:  # first frame of each contact
            found.append((float(i), np.array([float(np.clip(x, 0, cm.WIDTH)), cm.NET_Y])))
        last = i
    return found


def _first_seen(track: ShuttleTrack) -> int:
    """First frame with a detection (the track's length if there is none)."""
    seen = np.flatnonzero(track.visible())
    return int(seen[0]) if len(seen) else len(track.xy)


def _stays_slow(xy: np.ndarray, i: int, frames: int, max_speed: float) -> bool:
    """Whether the shuttle, from frame i, covers less than max_speed px per frame
    on average over the next `frames` frames (judged from the last detection in
    that span; with no later detection there is nothing against it)."""
    seg = xy[i + 1 : i + frames + 1]
    seen = np.flatnonzero(np.isfinite(seg).all(axis=1))
    if len(seen) == 0:
        return True
    j = int(seen[-1])
    return float(np.linalg.norm(seg[j] - xy[i])) / (j + 1) < max_speed


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
    i = _first_seen(track)
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
            events.append(_Event(float(first_contact), "landing", None, 1.0, rest_frame=float(i)))
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
    camera: Camera | None = None,
    start_frame: float = 0.0,
) -> list[Rally]:
    """Rallies in the clip. `cuts` are frames where the video jumps (edited
    footage); play never continues across a cut. With a 3D `camera`, a shot
    into the net is recognised and ends the rally. `start_frame` is where the
    analysed part of the video begins (nothing before it was looked at)."""
    px_scale = image_width / 1920
    visible = track.visible()
    if visible.sum() < 3:
        return []

    events = (
        _cusp_events(track, players, fps, px_scale)
        + _gap_events(track, players, fps, px_scale, start_frame)
        + _landing_events(track, calibration, fps, px_scale)
    )
    nets = _net_contacts(track, camera, fps, px_scale)
    # The shuttle stalling against the net, then lying under it, looks "at rest"
    # too: that is part of the net contact, not a separate landing. Judge by when
    # it came to rest (the touch-down frame is traced back and can land on the hit).
    landings = sorted(
        e.frame
        for e in events
        if e.kind == "landing"
        and not any(
            f - NET_LEAD_SEC * fps <= (e.rest_frame or e.frame) <= f + DEAD_TIME_SEC * fps
            for f, _ in nets
        )
    )
    # The bounce of a landing changes direction too; it is not a hit. Nor is
    # the shuttle stopping in the net, or anything until play restarts.
    bounce = BOUNCE_WINDOW_SEC * fps
    hit_events = [
        e
        for e in events
        if e.kind == "hit"
        and all(abs(e.frame - f) > bounce for f in landings)
        and not any(f - NET_LEAD_SEC * fps <= e.frame <= f + DEAD_TIME_SEC * fps for f, _ in nets)
    ]
    hits = _suppress(hit_events, MIN_SHOT_SEC * fps)

    rallies: list[Rally] = []
    # Stretches of continuous video: from where analysis starts, split at each cut.
    n = float(len(track.xy))
    bounds = [float(start_frame), *sorted(c for c in cuts if start_frame < c < n), n]
    for start, end in zip(bounds, bounds[1:], strict=False):
        seg_hits = [h for h in hits if start <= h.frame < end]
        seg_landings = [f for f in landings if start <= f < end]
        seg_nets = {f: p for f, p in nets if start <= f < end}
        rallies += _assemble_rallies(
            seg_hits,
            seg_landings,
            track,
            fps,
            start,
            end,
            px_scale,
            seg_nets,
            starts_video=start == bounds[0],
        )
    _require_serve_gap(rallies, fps)
    return rallies


def _require_serve_gap(rallies: list[Rally], fps: float) -> None:
    """Unmarks serves that come less than MIN_SERVE_GAP_SEC after the previous
    rally ended (rallies in time order, across cuts too)."""
    for prev, rally in zip(rallies, rallies[1:], strict=False):
        if rally.served and rally.hits[0].frame - prev.end_frame < MIN_SERVE_GAP_SEC * fps:
            rally.served = False


def _assemble_rallies(
    hits: list[_Event],
    landings: list[float],
    track: ShuttleTrack,
    fps: float,
    seg_start: float = 0.0,
    seg_end: float | None = None,
    px_scale: float = 1.0,
    net_contacts: dict[float, np.ndarray] | None = None,
    starts_video: bool = True,
) -> list[Rally]:
    """Groups time-ordered hits of one continuous stretch of video into rallies,
    closed by landings, shots into the net, long pauses or the end of the stretch.
    `starts_video`: the stretch begins where the analysed video does (rather
    than at a cut)."""
    seg_end = float(len(track.xy)) if seg_end is None else seg_end
    net_contacts = net_contacts or {}
    max_flight = RALLY_BREAK_SEC * fps
    # Anything that ends a rally: the shuttle on the floor or in the net.
    landings = sorted([*landings, *net_contacts])

    def ending(last_hit: _Event, landing: float | None) -> float | None:
        """The landing, if it can be how the rally ended (not dead time long after)."""
        return landing if landing is not None and landing - last_hit.frame <= max_flight else None

    groups: list[tuple[list[_Event], float, float | None]] = []  # hits, window start, landing
    current: list[_Event] = []
    window_start = seg_start
    blocked_until = -np.inf
    li = 0
    for ev in hits:
        # Close the rally at any landing that happens before this hit.
        while li < len(landings) and landings[li] < ev.frame:
            if current:
                groups.append((current, window_start, ending(current[-1], landings[li])))
                current = []
                blocked_until = landings[li] + DEAD_TIME_SEC * fps
                window_start = blocked_until
            li += 1
        if ev.frame < blocked_until:
            continue
        if current and ev.frame - current[-1].frame > max_flight:
            groups.append((current, window_start, None))
            window_start = current[-1].frame + max_flight / 2
            current = []
        current.append(ev)
    if current:
        landing = landings[li] if li < len(landings) else None
        groups.append((current, window_start, ending(current[-1], landing)))

    rallies: list[Rally] = []
    for group, w_start, landing in groups:
        seq = _assign_hitters(group, fps)
        net_point = net_contacts.get(landing) if landing is not None else None
        if landing is not None:
            end, landed = landing, net_point is None
        else:
            # Not seen to land: the shuttle is last seen in flight, which can't
            # last longer than a shot (it may be carried around in dead time).
            last_seen = _last_seen_after(track, seq[-1].frame, seg_end)
            end, landed = min(last_seen, seq[-1].frame + max_flight), False
        # A rally must end after its last hit. Trim inconsistent trailing hits
        # rather than discarding the whole rally.
        seq = [h for h in seq if h.frame < end]
        if seq:
            at_video_start = starts_video and w_start <= seg_start
            served = _starts_with_serve(
                track, seq[0].frame, w_start, fps, px_scale, at_video_start=at_video_start
            )
            rallies.append(Rally(seq, end, landed, served, net_point))
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
    track: ShuttleTrack,
    first_hit: float,
    window_start: float,
    fps: float,
    px_scale: float,
    *,
    at_video_start: bool,
) -> bool:
    """Whether the rally was seen to start: no shuttle flying before its first hit.

    Looks back up to SERVE_LOOKBACK_SEC (not before the rally window: the start
    of the analysed video, a cut or the end of the previous rally). Before a
    serve the shuttle is in the server's hand, so it is either unseen or barely
    moving. `at_video_start`: the window opens where the analysed video (clip or
    time window) begins, so a hit right there can't be called a serve. Not so
    after a cut: highlight reels cut in just before the serve (0.2-0.35 s
    before it in real footage), so there an unseen lead-in still counts.
    """
    lo = int(max(window_start, first_hit - SERVE_LOOKBACK_SEC * fps))
    hi = int(first_hit - 0.1 * fps)
    if hi - lo < 0.25 * fps and at_video_start:
        return False  # the analysed video starts right at this hit: can't tell
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
