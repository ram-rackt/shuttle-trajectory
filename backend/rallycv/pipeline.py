"""End-to-end rally analysis: video in, court-space trajectories out."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import video
from .cache import StageCache, stage_key
from .camera import Camera, CameraError, camera_from_homography
from .court import CourtCalibration, CourtNotFoundError, calibration_from_corners, detect_court
from .mapping import CourtTrajectory, map_rallies
from .players import (
    PersonDetector,
    PlayerTracker,
    PlayerTracks,
    smooth_track,
)
from .shots import classify_shot, detect_rallies
from .shuttle import Candidate, ShuttleDetector, ShuttleTrack, is_repeat, track_shuttle

if TYPE_CHECKING:
    from .tracknet import TrackNetDetector  # imported lazily at run time (torch is heavy)

logger = logging.getLogger(__name__)

STAGES: tuple[str, ...] = (
    "Detecting court",
    "Tracking players and shuttle",
    "Detecting shots",
    "Mapping to court",
)

BACKGROUND_SAMPLES = 25
OFF_VIEW_THRESHOLD = 28.0
"""Mean grey-level difference from the court background above which a frame is
treated as a different shot (replay, close-up) and ignored."""
CUT_RATIO = 3.5
CUT_MIN_FRACTION = 0.015
"""A hard cut to the same camera view (highlight reels splice rallies) changes
far more pixels between consecutive frames than any motion in play: more
than CUT_RATIO x the clip's median change and CUT_MIN_FRACTION of the frame."""

ProgressFn = Callable[[int, float], None]

PIPELINE_VERSION = "2026.10.12.4"
"""Bump whenever results would change (detection, shot logic, output format):
cached results from older versions are then recomputed instead of reused."""
TRACKNET_OVERLAP = "adaptive"


class PipelineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PipelineConfig:
    tracknet_weights: Path | None = None
    """TrackNetV3 checkpoint; without it the classical motion detector is used."""
    device: str | None = None
    """Torch device for TrackNet; None picks the best available."""
    player_detect_hz: float = 7.5
    """Person detections per second of video. Boxes are interpolated between
    detections, so this can be well below the frame rate."""
    person_region: bool = True
    """Run the person detector only on the part of the frame where players can
    be (the court plus margins, up to head height), at higher resolution."""
    output_hz: float = 30.0
    player_output_hz: float = 15.0
    max_duration_sec: float = 660.0
    """Longest stretch of video analysed at once (11 minutes)."""
    start_sec: float | None = None
    end_sec: float | None = None
    """Analyse only this part of the video (e.g. one rally of a long YouTube video).
    Times in the result stay relative to the start of the whole video."""


@dataclass
class ShotResult:
    index: int
    hitter: str
    type: str
    start_time: float
    end_time: float
    outcome: str | None = None


@dataclass
class RallyResult:
    fps: float
    duration: float
    image_size: tuple[int, int]
    calibration: CourtCalibration
    camera: Camera | None
    shots: list[ShotResult]
    shuttle_t: np.ndarray
    shuttle_ground: np.ndarray  # (N, 2) metres, NaN when unknown
    shuttle_image: np.ndarray  # (N, 2) pixels, NaN when not detected
    shuttle_height: np.ndarray  # (N,) metres, NaN when unknown
    players: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    """player id -> (times (N,), floor positions (N, 2) metres)"""
    stats: dict[str, float] = field(default_factory=dict)


def run(
    video_path: Path,
    detector: PersonDetector,
    on_progress: ProgressFn = lambda stage, p: None,
    court_corners: tuple[tuple[float, float], ...] | None = None,
    config: PipelineConfig | None = None,
    cache: StageCache | None = None,
) -> RallyResult:
    """Analyses the video, or the window [config.start_sec, config.end_sec) of it.

    With a `cache`, the expensive stages (shuttle and person detection) are
    loaded when they were already computed for this video with the same
    settings, and saved otherwise.
    """
    config = config or PipelineConfig()
    try:
        info = video.probe(video_path)
    except video.VideoError as e:
        raise PipelineError("video_unreadable", str(e)) from e
    first, last = _window_frames(info, config)
    span = (last - first) / info.fps
    if span > config.max_duration_sec + 0.5:
        raise PipelineError(
            "video_too_long",
            f"That's {span / 60:.1f} minutes of video. The limit is "
            f"{config.max_duration_sec / 60:.0f} minutes; choose a shorter part.",
        )
    if last - first < info.fps:
        raise PipelineError("video_too_short", "Choose at least one second of video.")

    # ---- Stage 0: court ------------------------------------------------------
    on_progress(0, 0.01)
    background = _background(video_path, first, last)
    calibration = _calibrate(background, info, court_corners)
    try:
        camera: Camera | None = camera_from_homography(
            calibration.court_to_image, calibration.image_size
        )
    except CameraError as e:
        logger.warning("No 3D camera; heights unavailable: %s", e)
        camera = None
    on_progress(0, 0.08)

    # ---- Stage 1: players + shuttle (single streaming pass) -----------------
    tracker = PlayerTracker(calibration)
    region = (
        player_region(calibration, camera)
        if config.person_region
        else (0, 0, info.width, info.height)
    )
    x0, y0, x1, y1 = region
    learned = config.tracknet_weights is not None and config.tracknet_weights.exists()

    window = (first, last)
    scan_key = players_key = None
    scan = players_cached = None
    if cache is not None and learned:
        scan_key = _scan_key(window, config.tracknet_weights)  # type: ignore[arg-type]
        scan = cache.load(scan_key)
    detector_id = getattr(detector, "cache_id", None)
    if cache is not None and detector_id is not None:
        # The image region isn't part of the key: a re-run with a slightly
        # different court (e.g. corners marked by hand) can reuse detections
        # made on a region that covers nearly all of the new one.
        players_key = stage_key(
            "players", window=window, detector=detector_id, hz=config.player_detect_hz
        )
        players_cached = cache.load(players_key)
        if players_cached is not None and not region_covers(players_cached.get("region"), region):
            players_cached = None

    if learned and scan is None:
        from .tracknet import TrackNetDetector  # torch import is heavy; keep lazy

        tracknet = TrackNetDetector(
            config.tracknet_weights,  # type: ignore[arg-type]
            background,
            device=config.device,
            overlap=TRACKNET_OVERLAP,
        )
        classical = None
    elif learned:
        tracknet = classical = None  # shuttle detections come from the cache
    else:
        logger.warning("TrackNet weights not found; using the classical shuttle detector.")
        tracknet = None
        classical = ShuttleDetector(calibration, camera)

    person_boxes: dict[int, np.ndarray] = (
        _unpack_boxes(players_cached) if players_cached is not None else {}
    )
    detect_players = players_cached is None

    frames = _FrameScan.from_cache(scan) if scan is not None else _FrameScan(background)

    need_frames = scan is None or detect_players or classical is not None
    if need_frames:
        distinct_set = set(frames.distinct) if scan is not None else None
        boxes: list[np.ndarray] = []
        last_detect_t = -np.inf
        last_report = first
        for i, frame in video.iter_frames(video_path, first, last):
            if distinct_set is not None:
                if i not in distinct_set:
                    continue
            else:
                kind = frames.check(i, frame)
                if kind == "off_view":
                    boxes = []  # replay / camera cut: nothing here is on our court
                if kind != "new":
                    continue

            t = i / info.fps
            if t - last_detect_t >= 1 / config.player_detect_hz - 1e-6:
                last_detect_t = t
                if detect_players:
                    found = detector.detect(frame[y0:y1, x0:x1])
                    if len(found):
                        found = found.copy()
                        found[:, [0, 2]] += x0
                        found[:, [1, 3]] += y0
                    person_boxes[i] = found
                tracker.update(i, person_boxes.get(i, np.zeros((0, 5))))
                boxes = [
                    obs[-1].box
                    for obs in tracker.tracks.observations.values()
                    if obs and i - obs[-1].frame <= info.fps * 0.3
                ]
            if tracknet is not None:
                tracknet.push(i, frame)
            elif classical is not None:
                classical.push(i, frame, boxes)
            if i - last_report >= 15:
                last_report = i
                on_progress(1, 0.08 + 0.8 * min((i - first) / max(last - first, 1), 1.0))
        if tracknet is not None:
            tracknet.flush()
    else:
        # Everything this stage produces is cached: replay the player detections.
        for i in sorted(person_boxes):
            tracker.update(i, person_boxes[i])
    on_progress(1, 0.88)

    if tracknet is not None:
        detections = _detections(tracknet)
        if cache is not None and scan_key is not None:
            cache.save(scan_key, **frames.arrays(), detections=detections)
    elif scan is not None:
        detections = scan["detections"]
    if cache is not None and players_key is not None and detect_players:
        cache.save(
            players_key, region=np.array(region, dtype=np.int64), **_pack_boxes(person_boxes)
        )

    total = max(last - first, 1)
    if frames.off_view > 0.6 * total:
        raise PipelineError(
            "court_not_visible",
            "The court isn't visible for most of the video. Use footage from a fixed camera "
            "behind the baseline.",
        )
    if not tracker.tracks.observations["A"] and not tracker.tracks.observations["B"]:
        raise PipelineError("players_not_found", "No players were found on the court.")

    # ---- Stage 2: shuttle track + hits ---------------------------------------
    if learned:
        candidates = {
            int(f): [Candidate(frame=int(f), x=float(x), y=float(y), score=float(c))]
            for f, x, y, c in detections
        }
    else:
        candidates = classical.candidates  # type: ignore[union-attr]
    n_frames = max(last, (max(candidates) + 2) if candidates else 0)
    cuts = sorted(set(find_cuts(frames.changes)) | set(frames.returned))
    track = track_shuttle(candidates, n_frames, info.fps, info.width, learned=learned, cuts=cuts)
    rallies = detect_rallies(
        track,
        tracker.tracks,
        calibration,
        info.fps,
        info.width,
        cuts=cuts,
        camera=camera,
        start_frame=first,
    )
    on_progress(2, 0.93)
    if not rallies:
        raise PipelineError(
            "shuttle_not_found",
            "No rally could be found: the shuttle couldn't be tracked between the players. "
            "Try a clearer or higher-resolution clip.",
        )

    # ---- Stage 3: court mapping ---------------------------------------------
    traj = map_rallies(rallies, track, tracker.tracks, calibration, camera, info.fps)
    result = _assemble(info, calibration, camera, track, tracker.tracks, traj, config, window)
    result.stats.update(
        {
            "court_confidence": calibration.confidence,
            "shuttle_visible_fraction": (
                float(track.visible()[first:last].mean()) if last > first else 0.0
            ),
            "repeated_frames": float(frames.repeated),
            "off_view_frames": float(frames.off_view),
            "rallies": float(len(rallies)),
            "cuts": float(len(cuts)),
            "learned_shuttle_detector": float(learned),
            "cached_scan": float(scan is not None),
            "cached_players": float(players_cached is not None),
        }
    )
    on_progress(3, 1.0)
    return result


def warm_scan(
    video_path: Path,
    config: PipelineConfig,
    cache: StageCache,
    should_stop: Callable[[], bool] = lambda: False,
) -> bool:
    """Computes and caches the "scan" stage on its own: the TrackNet shuttle
    detections and the frame bookkeeping (off-view and repeated frames, cuts).

    None of it depends on the court, so it can run while the user marks the
    court corners after automatic detection failed; the analysis that follows
    then only has to find the players. Gives up, saving nothing, as soon as
    `should_stop()` returns True. Returns whether the stage is now cached.
    """
    weights = config.tracknet_weights
    if weights is None or not weights.exists():
        return False
    try:
        info = video.probe(video_path)
    except video.VideoError:
        return False
    first, last = _window_frames(info, config)
    if last - first < info.fps or (last - first) / info.fps > config.max_duration_sec + 0.5:
        return False  # run() refuses this window anyway
    key = _scan_key((first, last), weights)
    if cache.has(key):
        return True

    from .tracknet import TrackNetDetector  # torch import is heavy; keep lazy

    background = _background(video_path, first, last)
    tracknet = TrackNetDetector(weights, background, device=config.device, overlap=TRACKNET_OVERLAP)
    frames = _FrameScan(background)
    for n, (i, frame) in enumerate(video.iter_frames(video_path, first, last)):
        if n % 15 == 0 and should_stop():
            return False
        if frames.check(i, frame) == "new":
            tracknet.push(i, frame)
    tracknet.flush()
    cache.save(key, **frames.arrays(), detections=_detections(tracknet))
    return True


def _background(video_path: Path, first: int, last: int) -> np.ndarray:
    """Median of frames spread over [first, last): the court without players."""
    idx = np.linspace(first, last - 1, BACKGROUND_SAMPLES).astype(int).tolist()
    return video.median_background(video.sample_frames(video_path, idx))


def _scan_key(window: tuple[int, int], weights: Path) -> str:
    return stage_key(
        "scan",
        window=window,
        tracknet=(weights.name, weights.stat().st_size),
        overlap=TRACKNET_OVERLAP,
    )


def _detections(tracknet: "TrackNetDetector") -> np.ndarray:
    """TrackNet's detections as (N, 4) rows: frame, x, y, confidence."""
    return np.array(
        [(f, d.x, d.y, d.confidence) for f, d in sorted(tracknet.detections.items())],
        dtype=np.float64,
    ).reshape(-1, 4)


class _FrameScan:
    """Sorts the frames of the pass: off-view (a replay or close-up, judged
    against the court background), a repeat of the previous frame, or new. For
    new frames it records how much changed, which is how cuts are found."""

    def __init__(self, background: np.ndarray | None) -> None:
        self.distinct: list[int] = []
        self.changes: list[tuple[int, int]] = []
        self.returned: list[int] = []
        """Frames where the view returns to the court after off-view frames."""
        self.off_view = 0
        self.repeated = 0
        self._bg_small = None if background is None else _small_gray(background)
        self._prev_gray: np.ndarray | None = None
        self._was_off_view = False

    @classmethod
    def from_cache(cls, scan: dict[str, np.ndarray]) -> "_FrameScan":
        frames = cls(None)
        frames.distinct = [int(f) for f in scan["distinct"]]
        frames.changes = [(int(f), int(c)) for f, c in scan["changes"]]
        frames.returned = [int(f) for f in scan["returned"]]
        frames.off_view, frames.repeated = (int(v) for v in scan["counts"])
        return frames

    def check(self, i: int, frame: np.ndarray) -> str:
        """'off_view', 'repeat' or 'new' for frame i (frames in order)."""
        if float(np.abs(_small_gray(frame) - self._bg_small).mean()) > OFF_VIEW_THRESHOLD:
            self.off_view += 1
            self._was_off_view = True
            return "off_view"
        if self._was_off_view:
            self.returned.append(i)
            self._was_off_view = False
        # Re-encoded footage often repeats frames (30 fps content in a 60 fps
        # file). Repeats carry no new information and confuse motion cues.
        gray = cv2.cvtColor(cv2.resize(frame, (1280, 720)), cv2.COLOR_BGR2GRAY)
        if self._prev_gray is not None and is_repeat(gray, self._prev_gray):
            self.repeated += 1
            return "repeat"
        if self._prev_gray is not None:
            self.changes.append((i, changed_pixels(gray, self._prev_gray)))
        self._prev_gray = gray
        self.distinct.append(i)
        return "new"

    def arrays(self) -> dict[str, np.ndarray]:
        """For the stage cache (TrackNet's detections are added alongside)."""
        return {
            "distinct": np.array(self.distinct, dtype=np.int64),
            "changes": np.array(self.changes, dtype=np.int64).reshape(-1, 2),
            "returned": np.array(self.returned, dtype=np.int64),
            "counts": np.array([self.off_view, self.repeated], dtype=np.int64),
        }


def _small_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)


REGION_REUSE_MIN_COVER = 0.95
"""Cached person detections are reused when the image region they were made on
covers at least this much of the region a new run wants: e.g. after the court
corners were marked by hand close to where they were detected. People in the
small uncovered strip would be at the very edge of where players can be."""


def region_covers(cached: np.ndarray | None, wanted: tuple[int, int, int, int]) -> bool:
    """Whether detections made on the `cached` region can stand in for `wanted`."""
    if cached is None or len(cached) != 4:
        return False
    cx0, cy0, cx1, cy1 = (float(v) for v in cached)
    wx0, wy0, wx1, wy1 = wanted
    overlap = max(0.0, min(cx1, wx1) - max(cx0, wx0)) * max(0.0, min(cy1, wy1) - max(cy0, wy0))
    area = max((wx1 - wx0) * (wy1 - wy0), 1)
    return overlap / area >= REGION_REUSE_MIN_COVER


def _window_frames(info: video.VideoInfo, config: "PipelineConfig") -> tuple[int, int]:
    """[first, last) frame range to analyse."""
    first = 0 if config.start_sec is None else int(round(config.start_sec * info.fps))
    last = info.frame_count if config.end_sec is None else int(round(config.end_sec * info.fps))
    first = min(max(first, 0), info.frame_count)
    last = min(max(last, first), info.frame_count)
    return first, last


def _pack_boxes(boxes: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-frame detections as flat arrays (npz can't hold ragged lists)."""
    frames = sorted(boxes)
    rows = [boxes[f] for f in frames]
    return {
        "frames": np.array(frames, dtype=np.int64),
        "counts": np.array([len(r) for r in rows], dtype=np.int64),
        "boxes": (np.vstack(rows) if rows else np.zeros((0, 5))).astype(np.float64).reshape(-1, 5),
    }


def _unpack_boxes(data: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    k = 0
    for f, n in zip(data["frames"], data["counts"], strict=True):
        out[int(f)] = data["boxes"][k : k + int(n)]
        k += int(n)
    return out


PLAYER_HEIGHT_M = 2.6
"""Head-and-racket height used to bound where players appear in the image."""
REGION_MARGIN_SIDE_M = 0.8
REGION_MARGIN_NEAR_M = 1.0
REGION_MARGIN_FAR_M = 1.5
"""How far outside the court lines (m) a player's feet are looked for."""


def player_region(
    calibration: CourtCalibration, camera: Camera | None
) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) that contains every player on or near the court.

    The court plus the margins players may stand in, from the floor up to head
    height. Detecting on this crop instead of the whole broadcast frame skips
    crowd and banners and gives players more pixels at the same network size.
    """
    w, h = calibration.image_size
    from . import court_model as cm

    xs = (-REGION_MARGIN_SIDE_M, cm.WIDTH + REGION_MARGIN_SIDE_M)
    ys = (-REGION_MARGIN_NEAR_M, cm.LENGTH + REGION_MARGIN_FAR_M)
    floor = np.array([(x, y) for x in xs for y in ys], dtype=np.float64)
    pts = [calibration.to_image(floor)]
    if camera is not None:
        heads = np.column_stack([floor, np.full(len(floor), PLAYER_HEIGHT_M)])
        pts.append(camera.project(heads))
    pts_all = np.vstack(pts)
    if not np.isfinite(pts_all).all():
        return 0, 0, w, h
    lo, hi = pts_all.min(axis=0), pts_all.max(axis=0)
    if camera is None:
        lo[1] -= 0.4 * (hi[1] - lo[1])  # no 3D camera: allow generously for heads
    pad = 0.03 * np.array([w, h])
    x0, y0 = np.maximum(lo - pad, 0).astype(int)
    x1, y1 = np.minimum(hi + pad, [w, h]).astype(int)
    if x1 - x0 < 0.2 * w or y1 - y0 < 0.2 * h:
        return 0, 0, w, h  # implausible geometry: fall back to the whole frame
    return int(x0), int(y0), int(x1), int(y1)


def nearest_gap(times: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """For each query, the distance to the nearest of `times` (sorted ascending).

    A sorted lookup: memory stays proportional to the inputs, where comparing
    every query with every time would need an 11-minute window's ~10k x 5k matrix.
    """
    j = np.searchsorted(times, queries)
    before = times[np.clip(j - 1, 0, len(times) - 1)]
    after = times[np.clip(j, 0, len(times) - 1)]
    return np.minimum(np.abs(queries - before), np.abs(after - queries))


def changed_pixels(gray: np.ndarray, prev_gray: np.ndarray) -> int:
    """Pixels (at half the working resolution) that changed noticeably."""
    return int((cv2.absdiff(gray[::2, ::2], prev_gray[::2, ::2]) > 25).sum())


def find_cuts(changes: list[tuple[int, int]]) -> list[int]:
    """Frames that start a new shot of the same camera view (hard edits)."""
    if len(changes) < 10:
        return []
    counts = np.array([c for _, c in changes], dtype=np.float64)
    pixels = (720 // 2) * (1280 // 2)
    threshold = max(CUT_RATIO * float(np.median(counts)), CUT_MIN_FRACTION * pixels)
    return [f for f, c in changes if c > threshold]


def _calibrate(
    background: np.ndarray,
    info: video.VideoInfo,
    court_corners: tuple[tuple[float, float], ...] | None,
) -> CourtCalibration:
    size = (info.width, info.height)
    if court_corners is not None:
        corners_px = np.array(court_corners, dtype=np.float64) * np.array(size, dtype=np.float64)
        try:
            return calibration_from_corners(corners_px, size, background)
        except ValueError as e:
            raise PipelineError("invalid_court_corners", str(e)) from e
    try:
        return detect_court(background)
    except CourtNotFoundError as e:
        raise PipelineError(
            "court_not_found",
            "The court lines couldn't be found automatically. Mark the four court corners "
            "to continue.",
        ) from e


def _assemble(
    info: video.VideoInfo,
    calibration: CourtCalibration,
    camera: Camera | None,
    track: ShuttleTrack,
    players: PlayerTracks,
    traj: CourtTrajectory,
    config: PipelineConfig,
    window: tuple[int, int] | None = None,
) -> RallyResult:
    fps = info.fps
    first, last = window or (0, info.frame_count)
    shots: list[ShotResult] = []
    for i, fl in enumerate(traj.flights):
        shots.append(
            ShotResult(
                index=i,
                hitter=fl.hitter,
                type=classify_shot(fl.features),
                start_time=fl.start_frame / fps,
                end_time=fl.end_frame / fps,
                outcome=fl.outcome,
            )
        )

    # Shuttle samples at output rate over the rally.
    step = max(1, round(fps / config.output_hz))
    frames = traj.frames[::step] if len(traj.frames) else np.zeros(0, int)
    shuttle_t = frames / fps
    shuttle_ground = traj.ground[frames] if len(frames) else np.zeros((0, 2))
    shuttle_height = traj.height[frames] if len(frames) else np.zeros(0)
    shuttle_image = track.xy[frames] if len(frames) else np.zeros((0, 2))

    out_players: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for pid, obs in players.observations.items():
        if not obs:
            continue
        f = np.array([o.frame for o in obs], dtype=np.float64)
        xy = np.array([o.feet_court for o in obs])
        f, xy = smooth_track(f, xy, fps)
        t_out = np.arange(first / fps, last / fps, 1 / config.player_output_hz)
        # Only report times close to a real observation.
        t_out = t_out[nearest_gap(f / fps, t_out) <= 1.0]
        pos = np.column_stack(
            [np.interp(t_out * fps, f, xy[:, 0]), np.interp(t_out * fps, f, xy[:, 1])]
        )
        out_players[pid] = (t_out, pos)

    return RallyResult(
        fps=fps,
        duration=info.duration,
        image_size=(info.width, info.height),
        calibration=calibration,
        camera=camera,
        shots=shots,
        shuttle_t=shuttle_t,
        shuttle_ground=shuttle_ground,
        shuttle_image=shuttle_image,
        shuttle_height=shuttle_height,
        players=out_players,
    )
