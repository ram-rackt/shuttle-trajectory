"""End-to-end run of the real pipeline on broadcast footage (slow, needs models).

Run with `make test-cv`. Skipped when the sample clip or weights are missing;
see THIRD_PARTY_NOTICES.md for the source of the clip.
"""

from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2]
SAMPLE = BACKEND / "data" / "samples" / "rally_china1.mp4"
MODELS = BACKEND / "data" / "models"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def result():
    if not SAMPLE.exists() or not (MODELS / "ckpts" / "TrackNet_best.pt").exists():
        pytest.skip("sample clip or model weights not available")
    from rallycv import pipeline
    from rallycv.players import YoloPersonDetector

    return pipeline.run(
        SAMPLE,
        YoloPersonDetector(MODELS / "yolo11n.pt"),
        config=pipeline.PipelineConfig(tracknet_weights=MODELS / "ckpts" / "TrackNet_best.pt"),
    )


def test_court_is_found(result) -> None:
    assert result.calibration.confidence > 0.5
    assert result.camera is not None


def test_rally_structure(result) -> None:
    # Hand-checked: the clip starts mid-rally; that point ends with a landing at
    # ~2.9 s. The next rally is served by the near player at ~4.4 s.
    shots = result.shots
    serves = [s for s in shots if s.type == "serve"]
    main = next(s for s in serves if 4.0 < s.start_time < 5.0)
    assert main.hitter == "A"
    rally = [s for s in shots if s.start_time >= main.start_time]
    assert len(rally) >= 10
    assert all(a.hitter != b.hitter for a, b in zip(rally, rally[1:], strict=False))


def test_players_and_shuttle_tracked(result) -> None:
    assert result.stats["shuttle_visible_fraction"] > 0.5
    for pid in ("A", "B"):
        times, _ = result.players[pid]
        assert len(times) > 0.8 * result.duration * 15


CLEAN = BACKEND / "data" / "samples" / "rally_clean.mp4"


def test_clean_single_rally_clip() -> None:
    """The recommended sample: exactly one rally, serve to landing."""
    if not CLEAN.exists() or not (MODELS / "ckpts" / "TrackNet_best.pt").exists():
        pytest.skip("rally_clean.mp4 or model weights not available")
    from rallycv import pipeline
    from rallycv.players import YoloPersonDetector

    res = pipeline.run(
        CLEAN,
        YoloPersonDetector(MODELS / "yolo11n.pt"),
        config=pipeline.PipelineConfig(tracknet_weights=MODELS / "ckpts" / "TrackNet_best.pt"),
    )
    assert res.stats["rallies"] == 1
    assert len(res.shots) >= 12
    assert res.shots[0].type == "serve" and res.shots[0].start_time < 1.5
    assert res.shots[-1].end_time == pytest.approx(20.07, abs=0.5)  # landing
