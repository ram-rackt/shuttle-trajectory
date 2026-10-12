"""Image track + hits -> court-space shuttle trajectory.

A single camera can't tell how high an airborne shuttle is, so its floor
position is ambiguous. We use the fact that a shuttle flies (almost) in the
vertical plane through the hitter and the receiver: intersecting each pixel's
camera ray with that plane gives a 3D point, hence a floor position and a
height. Where the ray grazes the plane (shots straight along the camera axis)
that intersection is unstable, so progress along the shot is interpolated in
time from the reliable points, with a drag-like profile as the fallback.
"""

from dataclasses import dataclass

import numpy as np

from . import court_model as cm
from .camera import Camera, intersect_vertical_plane
from .court import CourtCalibration
from .players import PlayerTracks
from .shots import Rally, ShotFeatures
from .shuttle import ShuttleTrack

MIN_CONDITIONING = 0.12
MAX_HEIGHT = 12.0
DRAG_K = 1.6
"""Shape of the fallback progress curve: shuttles decelerate strongly."""


@dataclass
class ShotFlight:
    index_in_rally: int
    start_frame: float
    end_frame: float
    hitter: str
    start: np.ndarray  # floor XY, metres
    end: np.ndarray
    apex_height: float | None
    features: ShotFeatures


@dataclass
class CourtTrajectory:
    frames: np.ndarray  # frames covered by the rally (inclusive range)
    ground: np.ndarray  # (N, 2) metres, NaN outside shots
    height: np.ndarray  # (N,) metres, NaN where unknown
    flights: list[ShotFlight]


def _drag_progress(u: np.ndarray) -> np.ndarray:
    return (1 - np.exp(-DRAG_K * u)) / (1 - np.exp(-DRAG_K))


def map_rallies(
    rallies: list[Rally],
    track: ShuttleTrack,
    players: PlayerTracks,
    calibration: CourtCalibration,
    camera: Camera | None,
    fps: float,
) -> CourtTrajectory:
    n_frames = len(track.xy)
    ground = np.full((n_frames, 2), np.nan)
    height = np.full(n_frames, np.nan)
    flights: list[ShotFlight] = []
    for rally in rallies:
        _map_one(rally, track, players, calibration, camera, fps, ground, height, flights)

    covered = np.nonzero(~np.isnan(ground[:, 0]))[0]
    rng = np.arange(covered[0], covered[-1] + 1) if len(covered) else np.zeros(0, int)
    return CourtTrajectory(frames=rng, ground=ground, height=height, flights=flights)


def _map_one(
    rally: Rally,
    track: ShuttleTrack,
    players: PlayerTracks,
    calibration: CourtCalibration,
    camera: Camera | None,
    fps: float,
    ground: np.ndarray,
    height: np.ndarray,
    flights: list[ShotFlight],
) -> None:
    n_frames = len(track.xy)
    hits = rally.hits
    for i, hit in enumerate(hits):
        f0 = hit.frame
        last = i == len(hits) - 1
        f1 = rally.end_frame if last else hits[i + 1].frame
        if f1 <= f0:
            continue
        receiver = "B" if hit.player == "A" else "A"
        p0 = players.position_at(hit.player, f0)
        if p0 is None:
            p0 = np.array([cm.WIDTH / 2, cm.NET_Y / 2 if hit.player == "A" else cm.NET_Y * 1.5])
        p1 = _landing_point(track, int(f1), calibration) if last and rally.landed else None
        if p1 is None:
            p1 = players.position_at(receiver, f1)
        if p1 is None:
            p1 = np.array([cm.WIDTH / 2, cm.NET_Y * 1.5 if hit.player == "A" else cm.NET_Y / 2])

        frames = np.arange(int(np.ceil(f0)), int(np.floor(f1)) + 1)
        frames = frames[(frames >= 0) & (frames < n_frames)]
        u_time = (frames - f0) / (f1 - f0)
        s = np.full(len(frames), np.nan)
        z = np.full(len(frames), np.nan)

        seen = ~np.isnan(track.xy[frames, 0]) if len(frames) else np.zeros(0, bool)
        if camera is not None and seen.any() and np.linalg.norm(p1 - p0) > 0.3:
            pts, cond = intersect_vertical_plane(camera, track.xy[frames[seen]], p0, p1)
            d = p1 - p0
            s_seen = ((pts[:, :2] - p0) @ d) / (d @ d)
            ok = (
                (cond > MIN_CONDITIONING)
                & (pts[:, 2] > -0.3)
                & (pts[:, 2] < MAX_HEIGHT)
                & (s_seen > -0.2)
                & (s_seen < 1.2)
            )
            idx = np.nonzero(seen)[0]
            s[idx[ok]] = np.clip(s_seen[ok], 0, 1)
            z[idx[ok]] = np.clip(pts[ok, 2], 0, None)

        s = _fill_progress(u_time, s)
        pos = p0 + np.outer(s, p1 - p0)
        ground[frames] = pos
        height[frames] = z
        apex = float(np.nanmax(z)) if np.isfinite(z).any() else None
        rise, overhead, received_high = _image_shape(
            track, players, hit.player, receiver, f0, f1, calibration, ends_in_hit=not last
        )
        flights.append(
            ShotFlight(
                index_in_rally=i,
                start_frame=f0,
                end_frame=f1,
                hitter=hit.player,
                start=p0,
                end=p1,
                apex_height=apex,
                features=ShotFeatures(
                    serve=i == 0 and rally.served,
                    start=p0,
                    end=p1,
                    duration=float(f1 - f0) / fps,
                    rise=rise,
                    overhead=overhead,
                    received_high=received_high,
                ),
            )
        )


def _image_shape(
    track: ShuttleTrack,
    players: PlayerTracks,
    hitter: str,
    receiver: str,
    f0: float,
    f1: float,
    calibration: CourtCalibration,
    *,
    ends_in_hit: bool,
) -> tuple[float | None, bool | None, bool | None]:
    """How high the flight loops in the image, whether the hitter struck it
    overhead, and whether the receiver had to take it above their head.

    All come straight from the detected image track and player boxes, so they
    don't depend on the (ambiguous) single-camera height reconstruction.
    """
    h = calibration.image_size[1]
    lo, hi = int(np.floor(f0)), int(np.ceil(f1)) + 1
    ys = track.xy[max(lo, 0) : min(hi, len(track.xy)), 1]
    seen = np.nonzero(np.isfinite(ys))[0]
    if len(seen) < 3:
        return None, None, None
    y_start, y_end = ys[seen[0]], ys[seen[-1]]
    top = float(np.min(ys[seen]))
    rise = (min(y_start, y_end) - top) / h
    # A shuttle that leaves through the top of the frame looped very high.
    if top < 0.04 * h:
        rise = max(rise, 0.3)

    near = 0.15 * (hi - lo) + 2
    overhead = None
    box = players.box_at(hitter, int(round(f0)), max_frame_gap=10)
    if box is not None and seen[0] <= near:
        overhead = bool(y_start < box[1] + 0.05 * (box[3] - box[1]))
    received_high = None
    box = players.box_at(receiver, int(round(f1)), max_frame_gap=10)
    if ends_in_hit and box is not None and (hi - lo - 1) - seen[-1] <= near:
        received_high = bool(y_end < box[1])
    return float(rise), overhead, received_high


def _fill_progress(u: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Monotone progress in [0, 1] anchored at both hits, through reliable points."""
    if len(u) == 0:
        return s
    known_u = [0.0]
    known_s = [0.0]
    for ui, si in zip(u, s, strict=False):
        if np.isfinite(si) and 0 < ui < 1:
            known_u.append(float(ui))
            known_s.append(float(si))
    known_u.append(1.0)
    known_s.append(1.0)
    ks = np.maximum.accumulate(np.array(known_s))  # progress never goes backwards
    if len(ks) == 2:
        return _drag_progress(np.clip(u, 0, 1))
    return np.interp(u, np.array(known_u), ks)


def _landing_point(
    track: ShuttleTrack, end_frame: int, calibration: CourtCalibration
) -> np.ndarray | None:
    """Where the final shot came down: last tracked pixel mapped onto the floor."""
    visible = np.nonzero(~np.isnan(track.xy[: end_frame + 1, 0]))[0]
    if len(visible) == 0:
        return None
    p = calibration.to_court(track.xy[visible[-1]][None])[0]
    if not np.isfinite(p).all():
        return None
    if -1.0 < p[0] < cm.WIDTH + 1.0 and -1.5 < p[1] < cm.LENGTH + 1.5:
        return p
    return None
