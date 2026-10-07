"""Hit/landing detection and court mapping on a synthetic rally with known truth."""

from dataclasses import dataclass

import numpy as np
import pytest

from rallycv import court_model as cm
from rallycv.court import CourtCalibration
from rallycv.mapping import map_rallies
from rallycv.players import PlayerObservation, PlayerTracks
from rallycv.shots import classify_shot, detect_rallies
from rallycv.shuttle import ShuttleTrack
from rallycv.synthetic import broadcast_camera, court_homography

FPS = 30.0
CAM = broadcast_camera()
CAL = CourtCalibration(court_to_image=court_homography(CAM), image_size=(1920, 1080), confidence=1)


@dataclass
class Flight:
    hitter: str
    start: tuple[float, float]  # floor XY of the hitter
    end: tuple[float, float]  # floor XY where it is struck next (or lands)
    duration: float
    apex: float
    contact_height: float = 2.2


# A short rally: A serves, B clears, A drops, B lifts... the last shot lands.
FLIGHTS = [
    Flight("A", (2.2, 4.0), (3.8, 11.5), 1.3, 4.5),
    Flight("B", (3.8, 11.5), (1.2, 1.0), 1.5, 6.0),
    Flight("A", (1.2, 1.0), (4.5, 8.0), 1.1, 3.0),
    Flight("B", (4.5, 8.0), (1.5, 2.0), 1.4, 5.5),
    Flight("A", (1.5, 2.0), (5.0, 11.0), 1.5, 5.0),  # lands in B's back corner
]
LEAD_IN = 0.8  # seconds before the serve
REST = 0.8  # shuttle lying on the floor after landing


def synth_rally() -> tuple[ShuttleTrack, PlayerTracks, list[float]]:
    """Image-space shuttle track, player boxes, and true hit times (seconds)."""
    total = LEAD_IN + sum(f.duration for f in FLIGHTS) + REST
    n = int(total * FPS)
    xy = np.full((n, 2), np.nan)
    hit_times: list[float] = []
    pos = {"A": np.array(FLIGHTS[0].start), "B": np.array(FLIGHTS[0].end)}
    player_pos_at = np.zeros((n, 2, 2))  # frame, player(A,B), XY

    t = LEAD_IN
    for k, fl in enumerate(FLIGHTS):
        last = k == len(FLIGHTS) - 1
        hit_times.append(t)
        f0, f1 = int(round(t * FPS)), int(round((t + fl.duration) * FPS))
        p0, p1 = np.array(fl.start), np.array(fl.end)
        h0, h1 = fl.contact_height, (0.0 if last else 2.0)
        for f in range(f0, min(f1, n)):
            u = (f - f0) / (f1 - f0)
            s = (1 - np.exp(-1.6 * u)) / (1 - np.exp(-1.6))  # drag-like progress
            ground = p0 + (p1 - p0) * s
            z = h0 + (h1 - h0) * u + 4 * (fl.apex - max(h0, h1)) * u * (1 - u)
            xy[f] = CAM.project(np.array([[ground[0], ground[1], max(z, 0.0)]]))[0]
        t += fl.duration
    landing = np.array(FLIGHTS[-1].end)
    f_land = int(round(t * FPS))
    xy[f_land:] = CAM.project(np.array([[landing[0], landing[1], 0.0]]))[0]

    # Players: hitter stands at the contact point; receiver moves to the next one.
    for f in range(n):
        tt = f / FPS
        for k, fl in enumerate(FLIGHTS):
            if tt >= hit_times[k]:
                pos[fl.hitter] = np.array(fl.start)
                other = "B" if fl.hitter == "A" else "A"
                u = min((tt - hit_times[k]) / fl.duration, 1.0)
                pos[other] = (
                    np.array(fl.start if k == 0 and other == "A" else pos[other]) * (1 - u)
                    + np.array(fl.end) * u
                )
        player_pos_at[f, 0], player_pos_at[f, 1] = pos["A"], pos["B"]

    tracks = PlayerTracks()
    for f in range(0, n, 2):
        for j, pid in enumerate(("A", "B")):
            X, Y = player_pos_at[f, j]
            feet = CAM.project(np.array([[X, Y, 0.0]]))[0]
            head = CAM.project(np.array([[X, Y, 1.75]]))[0]
            hgt = feet[1] - head[1]
            box = np.array([feet[0] - 0.22 * hgt, head[1], feet[0] + 0.22 * hgt, feet[1]])
            tracks.observations[pid].append(
                PlayerObservation(frame=f, feet_court=np.array([X, Y]), box=box, confidence=0.9)
            )
    return ShuttleTrack(frames=np.arange(n), xy=xy), tracks, hit_times


@pytest.fixture(scope="module")
def rally_data():
    track, players, truth = synth_rally()
    rallies = detect_rallies(track, players, CAL, FPS, 1920)
    return track, players, truth, rallies


def test_finds_one_rally_ending_in_a_landing(rally_data) -> None:
    _, _, truth, rallies = rally_data
    assert len(rallies) == 1
    rally = rallies[0]
    assert rally.landed
    land_time = LEAD_IN + sum(f.duration for f in FLIGHTS)
    assert rally.end_frame / FPS == pytest.approx(land_time, abs=0.15)


def test_hits_match_truth_in_time_and_player(rally_data) -> None:
    _, _, truth, rallies = rally_data
    hits = rallies[0].hits
    assert [h.player for h in hits] == [f.hitter for f in FLIGHTS]
    for h, t in zip(hits, truth, strict=True):
        assert h.frame / FPS == pytest.approx(t, abs=0.15)


def test_court_mapping_places_hits_and_landing(rally_data) -> None:
    track, players, _, rallies = rally_data
    traj = map_rallies(rallies, track, players, CAL, CAM)
    assert len(traj.flights) == len(FLIGHTS)
    for got, want in zip(traj.flights, FLIGHTS, strict=True):
        assert np.linalg.norm(got.start - np.array(want.start)) < 0.5
    # Landing point from the shuttle resting on the floor.
    assert np.linalg.norm(traj.flights[-1].end - np.array(FLIGHTS[-1].end)) < 0.3
    # Ground track stays on the court during the rally.
    covered = traj.ground[traj.frames]
    assert np.all((covered[:, 0] > -1) & (covered[:, 0] < cm.WIDTH + 1))
    assert np.all((covered[:, 1] > -1) & (covered[:, 1] < cm.LENGTH + 1))


def test_no_rally_without_a_tracked_shuttle() -> None:
    _, players, _ = synth_rally()
    empty = ShuttleTrack(frames=np.arange(10), xy=np.full((10, 2), np.nan))
    assert detect_rallies(empty, players, CAL, FPS, 1920) == []


@pytest.mark.parametrize(
    ("start", "end", "duration", "apex", "expected"),
    [
        ((3.0, 1.0), (3.0, 12.5), 1.6, 7.0, "clear"),  # deep to deep, high
        ((3.0, 1.5), (3.0, 8.0), 0.45, 2.2, "smash"),  # deep, fast, flat
        ((3.0, 5.5), (3.0, 7.6), 0.8, 1.8, "net"),  # front to front
        ((3.0, 5.5), (3.0, 12.4), 1.4, 5.0, "lift"),  # front to deep
        ((3.0, 1.5), (3.0, 7.8), 1.0, 3.5, "drop"),  # deep to front
    ],
)
def test_shot_classification(start, end, duration, apex, expected) -> None:
    got = classify_shot(3, np.array(start), np.array(end), duration, apex)
    assert got == expected


def test_first_shot_of_a_rally_is_a_serve() -> None:
    assert classify_shot(0, np.array([3, 4.0]), np.array([3, 11.0]), 1.0, 3.0) == "serve"


def test_hit_at_the_landing_frame_does_not_drop_the_rally() -> None:
    """Regression: a cusp detected at the landing bounce used to equal the rally's
    end frame, and the whole rally was discarded."""
    from rallycv.shots import _assemble_rallies, _Event

    track = ShuttleTrack(frames=np.arange(400), xy=np.zeros((400, 2)))
    hits = [
        _Event(30.0, "hit", "A", 1.0),
        _Event(70.0, "hit", "B", 1.0),
        _Event(110.0, "hit", "A", 1.0),
        _Event(150.0, "hit", "B", 1.0),  # same frame as the landing
    ]
    rallies = _assemble_rallies(hits, [150.0], 399.0, track, FPS)
    assert len(rallies) == 1
    assert [h.frame for h in rallies[0].hits] == [30.0, 70.0, 110.0]
    assert rallies[0].landed and rallies[0].end_frame == 150.0
