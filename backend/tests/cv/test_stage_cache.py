"""The stage cache across re-runs of the same video: person detections reused
when the court (hence the image region searched for players) barely changes,
and the scan stage computed ahead of time, without a court, by warm_scan."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from rallycv import court_model as cm
from rallycv import pipeline, tracknet
from rallycv.cache import StageCache
from rallycv.pipeline import PipelineConfig, PipelineError, region_covers
from rallycv.synthetic import broadcast_camera, render_court

SIZE = (960, 540)
FPS = 30.0
CAM = broadcast_camera(width=SIZE[0], height=SIZE[1], focal=1300.0)
PLAYERS = [(3.0, 2.5), (3.2, 10.8)]  # floor XY (m) of the near and far player


def _corners_px() -> np.ndarray:
    return CAM.project(np.c_[cm.CORNERS, np.zeros(4)])


def corners(scale: float = 1.0, nudge_px: float = 0.0) -> tuple[tuple[float, float], ...]:
    """The outer court corners, normalized: exact, nudged a few pixels (a hand
    mark close to the detected court) or scaled about the centre."""
    px = _corners_px()
    centre = px.mean(axis=0)
    px = centre + (px - centre) * scale + nudge_px
    return tuple((float(x / SIZE[0]), float(y / SIZE[1])) for x, y in px)


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """1.5 s of a broadcast view with two players drawn as solid red boxes."""
    frame = render_court(CAM, SIZE, distractors=False)
    for x, y in PLAYERS:
        feet = CAM.project(np.array([[x, y, 0.0]]))[0]
        head = CAM.project(np.array([[x, y, 1.8]]))[0]
        half = 0.22 * (feet[1] - head[1])
        top_left = (int(feet[0] - half), int(head[1]))
        bottom_right = (int(feet[0] + half), int(feet[1]))
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), -1)
    path = tmp_path_factory.mktemp("clip") / "rally.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for _ in range(int(1.5 * FPS)):
        writer.write(frame)
    writer.release()
    return path


class RedPeople:
    """Person detector stand-in: the red boxes, in the crop it is given."""

    cache_id = "red-people"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        self.calls += 1
        b, g, r = (frame_bgr[..., k].astype(int) for k in range(3))
        mask = ((r > 180) & (g < 80) & (b < 80)).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        return np.array(
            [[x, y, x + w, y + h, 0.9] for x, y, w, h, area in stats[1:] if area > 20],
            dtype=np.float64,
        ).reshape(-1, 5)


def analyse(clip: Path, cache: StageCache, detector: RedPeople, court) -> None:  # noqa: ANN001
    try:
        pipeline.run(clip, detector, court_corners=court, config=PipelineConfig(), cache=cache)
    except PipelineError as e:
        assert e.code in ("shuttle_not_found", "players_not_found")  # no shuttle in this clip


def test_region_covers() -> None:
    assert region_covers(np.array([0, 0, 100, 100]), (0, 0, 100, 100))
    assert region_covers(np.array([0, 0, 100, 100]), (2, 0, 102, 100))  # 98% covered
    assert not region_covers(np.array([0, 0, 100, 100]), (0, 0, 130, 100))
    assert not region_covers(None, (0, 0, 100, 100))


def test_person_detections_are_reused_when_the_court_barely_moves(
    clip: Path, tmp_path: Path
) -> None:
    cache = StageCache(tmp_path)
    detector = RedPeople()
    analyse(clip, cache, detector, corners())
    first_run = detector.calls
    assert first_run > 0

    # Corners marked by hand, a few pixels off: the cached detections serve.
    analyse(clip, cache, detector, corners(nudge_px=3))
    assert detector.calls == first_run

    # A court placed much larger needs a larger region: detect again.
    analyse(clip, cache, detector, corners(scale=1.3))
    assert detector.calls > first_run


class FakeTrackNet:
    """Stands in for TrackNetDetector (no weights needed); counts how often
    the network would have been set up."""

    created = 0

    def __init__(self, weights, background, device=None, overlap="adaptive") -> None:  # noqa: ANN001
        FakeTrackNet.created += 1
        self.detections: dict[int, tracknet.Detection] = {}

    def push(self, frame_index: int, frame_bgr: np.ndarray) -> None:
        self.detections[frame_index] = tracknet.Detection(frame_index, 480.0, 200.0, 0.9)

    def flush(self) -> None:
        pass


def test_warm_scan_computes_what_the_run_then_reuses(
    clip: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tracknet, "TrackNetDetector", FakeTrackNet)
    FakeTrackNet.created = 0
    weights = tmp_path / "TrackNet_best.pt"
    weights.write_bytes(b"weights")
    config = PipelineConfig(tracknet_weights=weights)
    cache = StageCache(tmp_path / "cache")

    # Told to stop (another video is waiting): nothing is saved.
    assert not pipeline.warm_scan(clip, config, cache, should_stop=lambda: True)
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())

    assert pipeline.warm_scan(clip, config, cache)
    assert FakeTrackNet.created == 2  # set up both times, ran to the end once
    assert pipeline.warm_scan(clip, config, cache)  # already cached: nothing to do
    assert FakeTrackNet.created == 2

    # The analysis with the user's corners doesn't run TrackNet again.
    try:
        pipeline.run(clip, RedPeople(), court_corners=corners(), config=config, cache=cache)
    except PipelineError:
        pass  # this clip has no rally; only the stages up to shots matter here
    assert FakeTrackNet.created == 2
