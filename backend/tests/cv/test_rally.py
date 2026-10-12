"""Hit/landing detection and court mapping on a synthetic rally with known truth."""

from dataclasses import dataclass

import numpy as np
import pytest

from rallycv import court_model as cm
from rallycv.court import CourtCalibration
from rallycv.mapping import map_rallies
from rallycv.players import PlayerObservation, PlayerTracks
from rallycv.shots import ShotFeatures, classify_shot, detect_rallies
from rallycv.shuttle import Candidate, ShuttleTrack, track_shuttle
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
    rallies = detect_rallies(track, players, CAL, FPS, 1920, camera=CAM)
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
    traj = map_rallies(rallies, track, players, CAL, CAM, FPS)
    assert len(traj.flights) == len(FLIGHTS)
    for got, want in zip(traj.flights, FLIGHTS, strict=True):
        assert np.linalg.norm(got.start - np.array(want.start)) < 0.5
    # Landing point from the shuttle resting on the floor.
    assert np.linalg.norm(traj.flights[-1].end - np.array(FLIGHTS[-1].end)) < 0.3
    # Ground track stays on the court during the rally.
    covered = traj.ground[traj.frames]
    assert np.all((covered[:, 0] > -1) & (covered[:, 0] < cm.WIDTH + 1))
    assert np.all((covered[:, 1] > -1) & (covered[:, 1] < cm.LENGTH + 1))


def test_rally_seen_from_its_start_is_served(rally_data) -> None:
    _, _, _, rallies = rally_data
    assert rallies[0].served


def test_no_rally_without_a_tracked_shuttle() -> None:
    _, players, _ = synth_rally()
    empty = ShuttleTrack(frames=np.arange(10), xy=np.full((10, 2), np.nan))
    assert detect_rallies(empty, players, CAL, FPS, 1920) == []


def _shift(players: PlayerTracks, offset: int, start: int = 0) -> list[tuple[str, list]]:
    return [
        (
            pid,
            [
                PlayerObservation(o.frame + offset, o.feet_court, o.box, o.confidence)
                for o in obs
                if o.frame >= start
            ],
        )
        for pid, obs in players.observations.items()
    ]


def test_clip_starting_mid_rally_has_no_serve() -> None:
    """Drop the lead-in and the serve: the shuttle is already flying at frame 0."""
    track, players, truth = synth_rally()
    skip = int(round((truth[0] + 0.5) * FPS))
    clipped = ShuttleTrack(frames=np.arange(len(track.xy) - skip), xy=track.xy[skip:])
    shifted = PlayerTracks()
    for pid, obs in _shift(players, -skip, start=skip):
        shifted.observations[pid] = obs
    rallies = detect_rallies(clipped, shifted, CAL, FPS, 1920)
    assert len(rallies) == 1
    assert not rallies[0].served
    assert [h.player for h in rallies[0].hits] == [f.hitter for f in FLIGHTS[1:]]


def test_rallies_never_span_a_hard_cut() -> None:
    """Two rallies spliced together (a highlight reel): each is its own rally,
    each served, and the hitter sequence restarts after the cut."""
    track, players, truth = synth_rally()
    n = len(track.xy)
    joined = PlayerTracks()
    for (pid, first), (_, second) in zip(_shift(players, 0), _shift(players, n), strict=True):
        joined.observations[pid] = first + second
    two = ShuttleTrack(frames=np.arange(2 * n), xy=np.vstack([track.xy, track.xy]))
    rallies = detect_rallies(two, joined, CAL, FPS, 1920, cuts=[n])
    assert len(rallies) == 2
    for r in rallies:
        assert r.served
        assert [h.player for h in r.hits] == [f.hitter for f in FLIGHTS]
    assert rallies[1].hits[0].frame / FPS == pytest.approx(truth[0] + n / FPS, abs=0.15)


def test_window_starting_right_at_a_hit_has_no_serve() -> None:
    """A window of a longer video (e.g. part of a YouTube match) that starts
    mid-rally, just before B's clear. The frames before it were never analysed,
    so the hit can't be a serve, any more than at the start of a clip."""
    track, players, truth = synth_rally()
    start = int(round(truth[1] * FPS)) - 3
    xy = track.xy.copy()
    xy[:start] = np.nan
    windowed = PlayerTracks()
    for pid, obs in _shift(players, 0, start=start):
        windowed.observations[pid] = obs
    rallies = detect_rallies(
        ShuttleTrack(frames=track.frames, xy=xy), windowed, CAL, FPS, 1920, start_frame=start
    )
    assert len(rallies) == 1
    assert not rallies[0].served
    assert [h.player for h in rallies[0].hits] == [f.hitter for f in FLIGHTS[1:]]


def test_serve_just_after_a_cut_is_kept_but_not_at_the_video_start() -> None:
    """Highlight reels cut in just before the serve (0.23-0.33 s before it in
    real footage): that is still a serve. At the start of the analysed video,
    a hit that soon can't be told from one in mid-rally."""
    from rallycv.shots import _assemble_rallies, _Event

    track = ShuttleTrack(frames=np.arange(400), xy=np.full((400, 2), np.nan))
    hits = [_Event(210.0, "hit", "A", 1.0), _Event(250.0, "hit", "B", 1.0)]  # 0.33 s in
    (after_cut,) = _assemble_rallies(hits, [], track, FPS, seg_start=200.0, starts_video=False)
    assert after_cut.served
    (at_start,) = _assemble_rallies(hits, [], track, FPS, seg_start=200.0)
    assert not at_start.served


def test_a_serve_needs_half_a_second_after_the_previous_rally() -> None:
    """Play restarting sooner than that after a rally ended (e.g. a replay
    cutting in mid-rally) is play carrying on, not a serve."""
    from rallycv.shots import MIN_SERVE_GAP_SEC, Hit, Rally, _require_serve_gap

    def rally(first_hit: float, end: float) -> Rally:
        return Rally([Hit(first_hit, "A", 1.0), Hit(first_hit + 30, "B", 1.0)], end, True)

    gap = MIN_SERVE_GAP_SEC * FPS
    first = rally(0.0, 100.0)
    too_soon = rally(100.0 + gap - 3, 200.0)  # just under 0.5 s after the first ended
    in_time = rally(200.0 + gap + 3, 300.0)  # just over 0.5 s after the second ended
    rallies = [first, too_soon, in_time]
    _require_serve_gap(rallies, FPS)
    assert [r.served for r in rallies] == [True, False, True]


def test_a_landing_long_after_the_last_hit_does_not_end_that_rally() -> None:
    """The rally's real landing was missed; 12 s later the shuttle, tossed back
    in dead time, settles on the floor. That isn't how the rally ended: its
    last shot must not stretch over the dead time."""
    from rallycv.shots import RALLY_BREAK_SEC, _assemble_rallies, _Event

    # The shuttle stays in view throughout (carried around between rallies).
    track = ShuttleTrack(frames=np.arange(900), xy=np.zeros((900, 2)))
    rally = [_Event(30.0, "hit", "A", 1.0), _Event(70.0, "hit", "B", 1.0)]
    rally.append(_Event(110.0, "hit", "A", 1.0))
    toss = 110.0 + 12 * FPS
    longest = 110.0 + RALLY_BREAK_SEC * FPS

    # Closed by the far landing before the next rally's hit...
    first, second = _assemble_rallies([*rally, _Event(800.0, "hit", "B", 1.0)], [toss], track, FPS)
    assert [h.frame for h in first.hits] == [30.0, 70.0, 110.0]
    assert not first.landed and first.end_frame <= longest
    assert second.hits[0].frame == 800.0
    # ...or as the clip's last rally.
    (only,) = _assemble_rallies(rally, [toss], track, FPS)
    assert not only.landed and only.end_frame <= longest


def test_ambiguous_hit_is_resolved_by_alternation() -> None:
    """At the net both players are within reach; alternation decides."""
    from rallycv.shots import _assign_hitters, _Event

    events = [
        _Event(0.0, "hit", "A", 1.5, {"A": 0.0, "B": 2.0}),
        _Event(20.0, "hit", "A", 1.5, {"A": 0.05, "B": 0.1}),  # both close: really B
        _Event(40.0, "hit", "A", 1.5, {"A": 0.0, "B": 2.0}),
    ]
    assert [h.player for h in _assign_hitters(events, FPS)] == ["A", "B", "A"]


def test_spurious_hit_is_dropped_and_missed_hit_inserted() -> None:
    from rallycv.shots import _assign_hitters, _Event

    events = [
        _Event(0.0, "hit", "A", 1.5, {"A": 0.0, "B": 2.0}),
        _Event(12.0, "hit", "A", 0.2, {"A": 0.3, "B": 2.0}),  # 0.4 s later: spurious
        _Event(30.0, "hit", "B", 1.5, {"A": 2.0, "B": 0.0}),
        _Event(90.0, "hit", "B", 1.5, {"A": 2.0, "B": 0.0}),  # 2 s later: A's hit missed
    ]
    hits = _assign_hitters(events, FPS)
    assert [(h.frame, h.player) for h in hits] == [
        (0.0, "A"),
        (30.0, "B"),
        (60.0, "A"),
        (90.0, "B"),
    ]


def test_fast_shuttle_after_a_hit_stays_in_the_track() -> None:
    """A learned detector's detections right after a hit jump far and change
    direction. They are the evidence for the hit and must stay linked."""
    pts = [(1000 - 5 * k, 600 - 2 * k) for k in range(10)]  # arriving slowly
    pts += [(pts[-1][0] + 60 * k, pts[-1][1] - 50 * k) for k in range(1, 8)]  # struck
    cands = {f: [Candidate(f, x, y, 0.8)] for f, (x, y) in enumerate(pts)}
    track = track_shuttle(cands, len(pts), 60.0, 1920, learned=True)
    assert track.visible().all()


def _shot(start, end, duration, *, serve=False, rise=None, overhead=None, received_high=None):
    return ShotFeatures(
        serve=serve,
        start=np.array(start, dtype=float),
        end=np.array(end, dtype=float),
        duration=duration,
        rise=rise,
        overhead=overhead,
        received_high=received_high,
    )


@pytest.mark.parametrize(
    ("shot", "expected"),
    [
        # Rear-court overhead to the opponent's rear court, high loop.
        (_shot((3, 1.0), (3, 12.5), 1.6, rise=0.3, overhead=True), "clear"),
        # Rear-court overhead, fast and flat.
        (_shot((3, 1.5), (3, 8.5), 0.45, rise=0.0, overhead=True), "smash"),
        # Front to front.
        (_shot((3, 5.5), (3, 7.6), 0.8, rise=0.04, overhead=False), "net"),
        # Front-court underarm to the rear, high loop.
        (_shot((3, 5.5), (3, 12.4), 1.4, rise=0.25, overhead=False), "lift"),
        # Rear-court overhead to the front, slower than a smash.
        (_shot((3, 1.5), (3, 7.8), 1.0, rise=0.02, overhead=True), "drop"),
        # Mid court, flat and fast, to mid court.
        (_shot((3, 3.5), (3, 10.5), 0.6, rise=0.01, overhead=False), "drive"),
        # A smash pushed back flat to mid court (used to be labelled a lift).
        (_shot((3, 9.9), (3, 2.2), 0.9, rise=0.01, overhead=True), "drive"),
        # Looks flat in the image, but the receiver had to take it overhead: lofted.
        (_shot((3, 2.3), (3, 11.4), 0.9, rise=0.03, overhead=False, received_high=True), "lift"),
    ],
)
def test_shot_classification(shot, expected) -> None:
    assert classify_shot(shot) == expected


def test_serve_is_labelled_only_when_seen() -> None:
    assert classify_shot(_shot((3, 4.0), (3, 11.0), 1.0, serve=True, rise=0.3)) == "serve"
    assert classify_shot(_shot((3, 4.0), (3, 11.0), 1.0, rise=0.3, overhead=False)) == "lift"


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
    rallies = _assemble_rallies(hits, [150.0], track, FPS)
    assert len(rallies) == 1
    assert [h.frame for h in rallies[0].hits] == [30.0, 70.0, 110.0]
    assert rallies[0].landed and rallies[0].end_frame == 150.0


def test_hard_cuts_found_from_frame_changes() -> None:
    from rallycv.pipeline import find_cuts

    changes = [(f, 1200 + (f % 7) * 150) for f in range(0, 600, 2)]  # ordinary play
    changes[40] = (80, 8000)  # the whole picture changes at once
    changes[200] = (400, 2300)  # a fast player near the camera: not a cut
    assert find_cuts(changes) == [80]


def test_a_normal_landing_is_not_mistaken_for_the_net(rally_data) -> None:
    _, _, _, rallies = rally_data
    assert rallies[0].net_point is None and rallies[0].landed


def test_shot_into_the_net_ends_the_rally_without_a_phantom_hit() -> None:
    """A's last shot strikes the net, stops and drops. Before net detection the
    stop-and-drop read as a direction change near B: a 'B drive' that never was."""
    track, players, truth = synth_rally()
    xy = track.xy.copy()
    f_hit = int(round(truth[-1] * FPS))
    contact = np.array([1.5, 2.0, 2.2])  # A smashes from the back of the court
    at_net = np.array([2.0, cm.NET_Y, 1.2])  # ...into the net, below the tape
    on_floor = np.array([2.0, cm.NET_Y - 0.15, 0.0])
    n_fly, n_drop = int(0.35 * FPS), int(0.5 * FPS)
    xy[f_hit:] = np.nan
    for k in range(n_fly + 1):
        u = k / n_fly
        xy[f_hit + k] = CAM.project((contact + (at_net - contact) * u)[None])[0]
    for k in range(1, n_drop + 1):
        u = (k / n_drop) ** 2  # falls, accelerating from rest
        xy[f_hit + n_fly + k] = CAM.project((at_net + (on_floor - at_net) * u)[None])[0]
    rest = f_hit + n_fly + n_drop
    xy[rest : rest + int(0.6 * FPS)] = xy[rest]
    into_net = ShuttleTrack(frames=track.frames, xy=xy)

    rallies = detect_rallies(into_net, players, CAL, FPS, 1920, camera=CAM)
    assert len(rallies) == 1
    rally = rallies[0]
    assert [h.player for h in rally.hits] == [f.hitter for f in FLIGHTS]  # no extra hit
    assert rally.net_point is not None and not rally.landed
    assert rally.net_point == pytest.approx([2.0, cm.NET_Y], abs=0.3)
    assert rally.end_frame / FPS == pytest.approx(truth[-1] + 0.35, abs=0.15)

    # The court view ends the shot at the net, and says so.
    traj = map_rallies(rallies, into_net, players, CAL, CAM, FPS)
    assert traj.flights[-1].outcome == "net"
    assert traj.flights[-1].end == pytest.approx([2.0, cm.NET_Y], abs=0.3)
    assert all(fl.outcome is None for fl in traj.flights[:-1])


def test_a_momentary_pause_near_the_net_is_not_a_net_contact(monkeypatch) -> None:
    """A shot that seems to stop for a frame or two near the net (a jittery or
    repeated detection) but then flies on at full speed did not strike the net.
    Seen on a re-encoded copy of a real clip: the false contact ended the rally
    after B's net shot and turned the rest into a new rally with a 'serve'."""
    from rallycv import shots

    start = np.array([3.0, cm.NET_Y + 2.0, 1.6])  # B plays it towards A...
    end = np.array([2.8, cm.NET_Y - 2.0, 0.0])
    n, k_net, pause = 18, 9, 2
    pts = [start + (end - start) * k / n for k in range(n + 1)]
    pts = pts[: k_net + 1] + [pts[k_net]] * pause + pts[k_net + 1 :]  # ...stalling at the net
    xy = np.full((60, 2), np.nan)
    xy[10 : 10 + len(pts)] = CAM.project(np.array(pts))
    track = ShuttleTrack(frames=np.arange(60), xy=xy)

    # Every other sign of a net contact is there: it arrives fast, stops, drops...
    monkeypatch.setattr(shots, "_stays_slow", lambda *args: True)
    assert len(shots._net_contacts(track, CAM, FPS, 1.0)) == 1
    # ...but it doesn't stay slow.
    monkeypatch.undo()
    assert shots._net_contacts(track, CAM, FPS, 1.0) == []


def test_net_contacts_need_the_3d_camera() -> None:
    from rallycv.shots import _net_contacts

    track, _, _ = synth_rally()
    assert _net_contacts(track, None, FPS, 1.0) == []
