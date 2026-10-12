"""The speed optimisations: TrackNet window scheduling, player box interpolation,
person-detector input size and the region players are searched in."""

import numpy as np
import pytest

from rallycv import tracknet
from rallycv.court import CourtCalibration
from rallycv.pipeline import player_region
from rallycv.players import PlayerObservation, PlayerTracks, input_size
from rallycv.synthetic import broadcast_camera, court_homography


class FakeTrackNet(tracknet.TrackNetDetector):
    """TrackNetDetector with the network replaced: each frame's heatmap peak is
    scripted, so the window scheduling can be tested without weights."""

    def __init__(self, overlap: str, peaks: dict[int, tuple[float, int]]) -> None:
        self._overlap = overlap
        self._scale = (1920 / tracknet.WIDTH, 1080 / tracknet.HEIGHT)
        self._new, self._tail, self._heats = [], [], {}
        self.detections = {}
        self.windows_run = 0
        self.windows: list[list[int]] = []
        self._peaks = peaks  # frame -> (peak value, x position in heatmap pixels)

    def _run(self, frames):  # noqa: ANN001
        self.windows.append([i for i, _ in frames])
        self.windows_run += 1
        for idx, _ in frames:
            heat = np.zeros((tracknet.HEIGHT, tracknet.WIDTH), np.float32)
            value, x = self._peaks.get(idx, (0.9, 100 + idx))
            heat[100:103, x : x + 3] = value
            self._heats.setdefault(idx, []).append(heat)


def run(overlap: str, n: int = 24, peaks: dict | None = None) -> FakeTrackNet:
    det = FakeTrackNet(overlap, peaks or {})
    for i in range(n):
        det._new.append((i, None))
        if len(det._new) == tracknet.SEQ_LEN:
            det._step(det._new)
            det._new = []
    det.flush()
    return det


def test_full_overlap_runs_every_straddling_window() -> None:
    det = run("full")
    assert det.windows == [
        list(range(0, 8)),
        list(range(8, 16)),
        list(range(4, 12)),
        list(range(16, 24)),
        list(range(12, 20)),
    ]


def test_no_overlap_runs_each_frame_once() -> None:
    assert run("none").windows == [list(range(0, 8)), list(range(8, 16)), list(range(16, 24))]


def test_adaptive_overlap_only_where_the_first_pass_is_unsure() -> None:
    # Clean track: one pass is enough.
    assert len(run("adaptive").windows) == 3
    # A frame with no shuttle (frame 10) triggers the window straddling it.
    det = run("adaptive", peaks={10: (0.2, 110)})
    assert list(range(4, 12)) in det.windows and list(range(12, 20)) not in det.windows
    # A detection that jumps far off the path (frame 14) does too.
    det = run("adaptive", peaks={14: (0.9, 400)})
    assert list(range(12, 20)) in det.windows and list(range(4, 12)) not in det.windows


def test_every_frame_is_finalised_once() -> None:
    det = run("adaptive", n=21, peaks={10: (0.2, 110)})
    assert sorted(det.detections) == [i for i in range(21) if i != 10]


def test_player_boxes_are_interpolated_between_detections() -> None:
    tracks = PlayerTracks()
    tracks.observations["A"] = [
        PlayerObservation(0, np.zeros(2), np.array([0.0, 0, 10, 20]), 0.9),
        PlayerObservation(8, np.zeros(2), np.array([8.0, 0, 18, 20]), 0.9),
    ]
    assert tracks.box_at("A", 2, max_frame_gap=8) == pytest.approx([2, 0, 12, 20])
    assert tracks.box_at("A", 20, max_frame_gap=8) is None  # too far from any detection
    assert tracks.box_at("B", 2, max_frame_gap=8) is None


def test_detector_input_keeps_the_aspect_ratio() -> None:
    assert input_size(928, 1711) == (352, 640)
    assert input_size(1080, 1920) == (352, 640)
    h, w = input_size(1920, 1080)  # portrait
    assert (h, w) == (640, 352)


def test_player_region_covers_the_court_and_players() -> None:
    cam = broadcast_camera()
    cal = CourtCalibration(court_homography(cam), image_size=(1920, 1080), confidence=1)
    x0, y0, x1, y1 = player_region(cal, cam)
    corners = cal.to_image(np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]]))
    inside = corners[(corners[:, 0] >= 0) & (corners[:, 0] <= 1920) & (corners[:, 1] <= 1080)]
    assert np.all(inside[:, 0] >= x0) and np.all(inside[:, 0] <= x1)
    assert np.all(inside[:, 1] >= y0) and np.all(inside[:, 1] <= y1)
    # The far player's head (1.8 m above the far baseline) is inside too.
    head = cam.project(np.array([[3.05, 13.4, 1.8]]))[0]
    assert x0 <= head[0] <= x1 and y0 <= head[1] <= y1
    # Without a 3D camera the region still holds the court, with room above it.
    assert player_region(cal, None)[1] <= y0
