import pytest

from app.analysis.contract import RallyAnalysis
from app.analysis.mock import generate_mock_rally

DURATIONS = [0.5, 2.0, 8.0, 20.0, 90.0]
MARGIN = 0.06  # positions may sit slightly outside the lines


def assert_valid_rally(rally: RallyAnalysis, duration: float) -> None:
    assert rally.schema_version == 1
    assert rally.duration_sec == pytest.approx(duration, abs=1e-3)

    # Shots: at least one, ordered, contiguous, inside the video, alternating hitters.
    assert rally.shots, "expected at least one shot"
    assert rally.shots[0].type == "serve"
    assert rally.shots[0].hitter == "A"
    for i, shot in enumerate(rally.shots):
        assert shot.index == i
        assert 0 <= shot.start_time < shot.end_time <= duration
        if i > 0:
            prev = rally.shots[i - 1]
            assert shot.start_time == pytest.approx(prev.end_time)
            assert shot.hitter != prev.hitter

    # Shuttle: time-ordered, inside the video, positions near the court.
    times = [s.t for s in rally.shuttle]
    assert times == sorted(times)
    assert all(0 <= t <= duration for t in times)
    for s in rally.shuttle:
        if s.position:
            assert -MARGIN <= s.position.x <= 1 + MARGIN
            assert -MARGIN <= s.position.y <= 1 + MARGIN

    # Players: A and B, each staying in their own half.
    assert [p.id for p in rally.players] == ["A", "B"]
    for track in rally.players:
        assert track.samples
        assert track.samples[-1].t <= duration
        for sample in track.samples:
            if track.id == "A":
                assert sample.position.y <= 0.5
            else:
                assert sample.position.y >= 0.5


@pytest.mark.parametrize("duration", DURATIONS)
def test_mock_rally_is_valid(duration: float) -> None:
    assert_valid_rally(generate_mock_rally(duration, seed=7), duration)


@pytest.mark.parametrize("seed", range(25))
def test_mock_rally_is_valid_across_seeds(seed: int) -> None:
    assert_valid_rally(generate_mock_rally(15.0, seed=seed), 15.0)


def test_mock_rally_is_deterministic() -> None:
    assert generate_mock_rally(12.0, seed=1) == generate_mock_rally(12.0, seed=1)
    assert generate_mock_rally(12.0, seed=1) != generate_mock_rally(12.0, seed=2)


def test_long_rally_has_many_shots_and_tracking_gaps() -> None:
    rally = generate_mock_rally(30.0, seed=3)
    assert len(rally.shots) >= 10
    assert any(s.position is None for s in rally.shuttle)


def test_serialises_with_camel_case_keys() -> None:
    data = generate_mock_rally(5.0, seed=1).model_dump(by_alias=True)
    assert {
        "schemaVersion",
        "source",
        "durationSec",
        "players",
        "shuttle",
        "shots",
        "calibration",
    } == set(data)
    assert data["source"] == "mock"
    assert {"index", "hitter", "type", "startTime", "endTime"} == set(data["shots"][0])
