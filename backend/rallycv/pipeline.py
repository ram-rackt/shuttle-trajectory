"""End-to-end rally analysis: video in, court-space trajectories out."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import video
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
    max_duration_sec: float = 300.0


@dataclass
class ShotResult:
    index: int
    hitter: str
    type: str
    start_time: float
    end_time: float


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
) -> RallyResult:
    config = config or PipelineConfig()
    try:
        info = video.probe(video_path)
    except video.VideoError as e:
        raise PipelineError("video_unreadable", str(e)) from e
    if info.duration > config.max_duration_sec:
        raise PipelineError(
            "video_too_long",
            f"The video is {info.duration / 60:.1f} minutes long. "
            f"Upload a single rally (under {config.max_duration_sec / 60:.0f} minutes).",
        )

    # ---- Stage 0: court ------------------------------------------------------
    on_progress(0, 0.01)
    idx = np.linspace(0, info.frame_count - 1, BACKGROUND_SAMPLES).astype(int).tolist()
    background = video.median_background(video.sample_frames(video_path, idx))
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
    x0, y0, x1, y1 = (
        player_region(calibration, camera)
        if config.person_region
        else (0, 0, info.width, info.height)
    )
    learned = config.tracknet_weights is not None and config.tracknet_weights.exists()
    if learned:
        from .tracknet import TrackNetDetector  # torch import is heavy; keep lazy

        tracknet = TrackNetDetector(config.tracknet_weights, background, device=config.device)  # type: ignore[arg-type]
        classical = None
    else:
        logger.warning("TrackNet weights not found; using the classical shuttle detector.")
        tracknet = None
        classical = ShuttleDetector(calibration, camera)

    bg_small = cv2.cvtColor(cv2.resize(background, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
    boxes: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None
    last_detect_t = -np.inf
    off_view = repeated = 0
    last_report = 0
    changes: list[tuple[int, int]] = []  # (frame, changed pixels vs previous distinct frame)
    returned_from_off_view: list[int] = []
    was_off_view = False
    for i, frame in video.iter_frames(video_path):
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)
        if float(np.abs(small - bg_small).mean()) > OFF_VIEW_THRESHOLD:
            off_view += 1  # replay / camera cut: nothing here is on our court
            boxes = []
            was_off_view = True
            continue
        if was_off_view:
            returned_from_off_view.append(i)
            was_off_view = False
        # Re-encoded footage often repeats frames (30 fps content in a 60 fps
        # file). Repeats carry no new information and confuse motion cues.
        gray = cv2.cvtColor(cv2.resize(frame, (1280, 720)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and is_repeat(gray, prev_gray):
            repeated += 1
            continue
        if prev_gray is not None:
            changes.append((i, changed_pixels(gray, prev_gray)))
        prev_gray = gray

        t = i / info.fps
        if t - last_detect_t >= 1 / config.player_detect_hz - 1e-6:
            last_detect_t = t
            found = detector.detect(frame[y0:y1, x0:x1])
            if len(found):
                found = found.copy()
                found[:, [0, 2]] += x0
                found[:, [1, 3]] += y0
            tracker.update(i, found)
            boxes = [
                obs[-1].box
                for obs in tracker.tracks.observations.values()
                if obs and i - obs[-1].frame <= info.fps * 0.3
            ]
        if tracknet is not None:
            tracknet.push(i, frame)
        else:
            classical.push(i, frame, boxes)  # type: ignore[union-attr]
        if i - last_report >= 15:
            last_report = i
            on_progress(1, 0.08 + 0.8 * min(i / max(info.frame_count, 1), 1.0))
    if tracknet is not None:
        tracknet.flush()
    on_progress(1, 0.88)

    total = max(info.frame_count, 1)
    if off_view > 0.6 * total:
        raise PipelineError(
            "court_not_visible",
            "The court isn't visible for most of the video. Upload footage from a fixed camera "
            "behind the baseline.",
        )
    if not tracker.tracks.observations["A"] and not tracker.tracks.observations["B"]:
        raise PipelineError("players_not_found", "No players were found on the court.")

    # ---- Stage 2: shuttle track + hits ---------------------------------------
    if tracknet is not None:
        candidates = {
            f: [Candidate(frame=f, x=d.x, y=d.y, score=d.confidence)]
            for f, d in tracknet.detections.items()
        }
    else:
        candidates = classical.candidates  # type: ignore[union-attr]
    n_frames = max(total, (max(candidates) + 2) if candidates else 0)
    cuts = sorted(set(find_cuts(changes)) | set(returned_from_off_view))
    track = track_shuttle(candidates, n_frames, info.fps, info.width, learned=learned, cuts=cuts)
    rallies = detect_rallies(track, tracker.tracks, calibration, info.fps, info.width, cuts=cuts)
    on_progress(2, 0.93)
    if not rallies:
        raise PipelineError(
            "shuttle_not_found",
            "No rally could be found: the shuttle couldn't be tracked between the players. "
            "Try a clearer or higher-resolution clip.",
        )

    # ---- Stage 3: court mapping ---------------------------------------------
    traj = map_rallies(rallies, track, tracker.tracks, calibration, camera, info.fps)
    result = _assemble(info, calibration, camera, track, tracker.tracks, traj, config)
    result.stats.update(
        {
            "court_confidence": calibration.confidence,
            "shuttle_visible_fraction": float(track.visible().mean()) if len(track.xy) else 0.0,
            "repeated_frames": float(repeated),
            "off_view_frames": float(off_view),
            "rallies": float(len(rallies)),
            "cuts": float(len(cuts)),
            "learned_shuttle_detector": float(learned),
        }
    )
    on_progress(3, 1.0)
    return result


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
) -> RallyResult:
    fps = info.fps
    shots: list[ShotResult] = []
    for i, fl in enumerate(traj.flights):
        shots.append(
            ShotResult(
                index=i,
                hitter=fl.hitter,
                type=classify_shot(fl.features),
                start_time=fl.start_frame / fps,
                end_time=fl.end_frame / fps,
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
        t_out = np.arange(0, info.duration, 1 / config.player_output_hz)
        # Only report times close to a real observation.
        nearest = np.min(np.abs(f[None, :] / fps - t_out[:, None]), axis=1)
        t_out = t_out[nearest <= 1.0]
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
