"""Hit, landing and rally detection from the image-space shuttle track.

With a dense track (TrackNet), a hit shows up as a cusp: the shuttle arrives
at a player and leaves in a clearly different direction. Each cusp is
attributed to the player whose "reach" region (their box, widened and
extended above the head) contains the shuttle. A landing is the shuttle
coming to rest on the floor; it ends the rally. Play that resumes after a
landing starts a new rally with a serve. Hitters must alternate within a
rally, which is used to clean up the sequence.
"""

from dataclasses import dataclass

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


def _hitter(
    xy: np.ndarray, frame: int, players: PlayerTracks, max_gap: int
) -> tuple[str | None, float]:
    """The player within reach of the shuttle, preferring the closer box centre."""
    best, best_key = None, (np.inf, np.inf)
    for p in ("A", "B"):
        box = players.box_at(p, frame, max_gap)
        if box is None:
            continue
        d = reach_distance(xy, box)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        centre = np.hypot(xy[0] - cx, xy[1] - cy) / max(box[3] - box[1], 1.0)
        key = (round(d, 2), centre)
        if key < best_key:
            best, best_key = p, key
    return best, best_key[0]


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
    strength: float


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
            if n_in < min_speed or n_out < min_speed:
                continue
            cos = float(np.clip(v_in @ v_out / (n_in * n_out), -1, 1))
            angle = float(np.degrees(np.arccos(cos)))
            ratio = max(n_in, n_out) / min(n_in, n_out)
            if angle < MIN_CUSP_ANGLE and not (angle > 25 and ratio > 2.5):
                continue
            frame = s + i
            player, dist = _hitter(seg[i], frame, players, max_gap)
            if player is None or dist > MAX_REACH:
                continue
            strength = angle / 180 + min(ratio, 4) / 8 + (MAX_REACH - dist)
            events.append(_Event(float(frame), "hit", player, strength))
    return events


def _gap_events(track: ShuttleTrack, players: PlayerTracks, fps: float) -> list[_Event]:
    """A run that starts next to a player after a gap: the hit happened in the gap."""
    events: list[_Event] = []
    max_gap = int(fps * 0.25)
    runs = [r for r in _runs(track.visible()) if r[1] - r[0] >= 3]
    for k, (s, _e) in enumerate(runs):
        gap = s - runs[k - 1][1] if k > 0 else s
        if gap < 0.12 * fps:
            continue
        player, dist = _hitter(track.xy[s], s, players, max_gap)
        if player is None or dist > 0.4:
            continue
        lead = min(gap / 2, 0.1 * fps)
        events.append(_Event(float(s - lead), "hit", player, 0.6 - dist))
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
) -> list[Rally]:
    px_scale = image_width / 1920
    visible = track.visible()
    if visible.sum() < 3:
        return []

    events = (
        _cusp_events(track, players, fps, px_scale)
        + _gap_events(track, players, fps)
        + _landing_events(track, calibration, fps, px_scale)
    )
    landings = sorted(e.frame for e in events if e.kind == "landing")
    # The bounce of a landing changes direction too; it is not a hit.
    bounce = BOUNCE_WINDOW_SEC * fps
    hit_events = [
        e for e in events if e.kind == "hit" and all(abs(e.frame - f) > bounce for f in landings)
    ]
    hits = _suppress(hit_events, MIN_SHOT_SEC * fps)
    last_seen = float(np.nonzero(visible)[0][-1])
    return _assemble_rallies(hits, landings, last_seen, track, fps)


def _assemble_rallies(
    hits: list[_Event],
    landings: list[float],
    last_seen: float,
    track: ShuttleTrack,
    fps: float,
) -> list[Rally]:
    """Groups time-ordered hits into rallies, closed by landings or long pauses."""
    rallies: list[Rally] = []
    current: list[Hit] = []
    blocked_until = -np.inf
    li = 0
    for ev in hits:
        # Close the rally at any landing that happens before this hit.
        while li < len(landings) and landings[li] < ev.frame:
            if current:
                rallies.append(Rally(_alternate(current, fps), landings[li], landed=True))
                current = []
                blocked_until = landings[li] + DEAD_TIME_SEC * fps
            li += 1
        if ev.frame < blocked_until:
            continue
        if current and ev.frame - current[-1].frame > RALLY_BREAK_SEC * fps:
            rallies.append(
                Rally(_alternate(current, fps), _last_seen_after(track, current[-1].frame), False)
            )
            current = []
        current.append(Hit(ev.frame, ev.player or "A", ev.strength))
    if current:
        end = landings[li] if li < len(landings) else last_seen
        rallies.append(Rally(_alternate(current, fps), end, landed=li < len(landings)))

    # A rally must end after its last hit. Trim inconsistent trailing hits
    # rather than discarding the whole rally.
    out = []
    for r in rallies:
        hits_before_end = [h for h in r.hits if h.frame < r.end_frame]
        if hits_before_end:
            out.append(Rally(hits_before_end, r.end_frame, r.landed))
    return out


def _last_seen_after(track: ShuttleTrack, frame: float) -> float:
    vis = np.nonzero(track.visible())[0]
    after = vis[vis > frame]
    if len(after) == 0:
        return frame + 1
    # End of the first visible run after the hit.
    end = after[0]
    for f in after[1:]:
        if f - end > 3:
            break
        end = f
    return float(end)


def _suppress(events: list[_Event], window: float) -> list[_Event]:
    """Non-maximum suppression in time: keep the strongest event in each window."""
    out: list[_Event] = []
    for ev in sorted(events, key=lambda e: -e.strength):
        if all(abs(ev.frame - o.frame) >= window for o in out):
            out.append(ev)
    return sorted(out, key=lambda e: e.frame)


def _alternate(hits: list[Hit], fps: float) -> list[Hit]:
    """Hitters alternate. Resolve repeats: a long gap means the opponent's hit was
    missed (insert it); a short one means a duplicate (keep the stronger)."""
    out: list[Hit] = []
    for h in hits:
        if out and out[-1].player == h.player:
            gap = h.frame - out[-1].frame
            if gap > 1.4 * fps:
                out.append(Hit(out[-1].frame + gap / 2, "B" if h.player == "A" else "A", 0.0))
                out.append(h)
            elif h.strength > out[-1].strength:
                out[-1] = h
            continue
        out.append(h)
    return out


# --------------------------------------------------------------------------- #
# Shot classification
# --------------------------------------------------------------------------- #


def classify_shot(
    index_in_rally: int,
    start: np.ndarray,
    end: np.ndarray,
    duration: float,
    apex_height: float | None,
) -> str:
    """Heuristic shot type from where it was hit, where it went, speed and height.

    Depth is measured from the net in each half: 0 at the net, 1 at the baseline.
    """
    if index_in_rally == 0:
        return "serve"
    d0 = abs(start[1] - cm.NET_Y) / cm.NET_Y
    d1 = abs(end[1] - cm.NET_Y) / cm.NET_Y
    speed = float(np.linalg.norm(end - start) / max(duration, 1e-3))
    high = apex_height is not None and apex_height > 4.0
    flat = apex_height is not None and apex_height < 2.6

    if d0 > 0.45 and speed > 9.0 and not high:
        return "smash"
    if d0 < 0.35 and d1 < 0.4:
        return "net"
    if d0 < 0.5 and d1 > 0.6:
        return "lift"
    if d0 >= 0.45 and d1 > 0.62:
        return "clear"
    if d0 >= 0.4 and d1 < 0.45:
        return "drop"
    if speed > 6.0 and (flat or apex_height is None):
        return "drive"
    return "unknown"
