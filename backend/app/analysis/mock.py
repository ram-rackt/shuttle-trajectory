"""Mock analysis provider.

Generates a plausible singles rally so the frontend can be built end-to-end
before any computer vision exists. Output is deterministic for a given seed.

Geometry reminder (normalized court, see docs/analysis-contract.md):
player A defends y in [0, 0.5), player B defends y in (0.5, 1], net at y = 0.5.
"""

import math
import random
import time
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

from .contract import (
    CourtPoint,
    PlayerId,
    PlayerSample,
    PlayerTrack,
    RallyAnalysis,
    Shot,
    ShotType,
    ShuttleSample,
)
from .provider import AnalysisRequest, ProgressCallback

SHUTTLE_HZ = 30
PLAYER_HZ = 15
DEFAULT_DURATION_SEC = 20.0

MOCK_STAGES: tuple[str, ...] = (
    "Detecting court",
    "Tracking players",
    "Tracking shuttle",
    "Detecting shots",
    "Mapping to court",
)


@dataclass(frozen=True)
class ShotProfile:
    duration: tuple[float, float]
    """Flight time range in seconds."""
    depth: tuple[float, float]
    """Landing depth range: 0 = at the net, 1 = on the receiver's baseline."""
    ease: float
    """>1 decelerates through the flight (high, floaty shots); 1 = constant speed."""


PROFILES: dict[ShotType, ShotProfile] = {
    "serve": ShotProfile((1.0, 1.3), (0.28, 0.4), 1.3),
    "clear": ShotProfile((1.3, 1.7), (0.85, 0.97), 1.8),
    "lift": ShotProfile((1.2, 1.6), (0.8, 0.95), 1.6),
    "drop": ShotProfile((0.9, 1.2), (0.12, 0.3), 1.2),
    "smash": ShotProfile((0.45, 0.65), (0.35, 0.75), 1.0),
    "drive": ShotProfile((0.6, 0.8), (0.4, 0.65), 1.0),
    "net": ShotProfile((0.7, 0.95), (0.05, 0.18), 1.0),
}

# Which shots are realistic from where the hitter is standing (depth in own half).
NEXT_SHOT_WEIGHTS: dict[str, dict[ShotType, float]] = {
    "deep": {"clear": 4, "drop": 3, "smash": 3},
    "mid": {"drive": 3, "smash": 2, "lift": 2, "drop": 2},
    "front": {"net": 4, "lift": 4, "drive": 1},
}

BASE_DEPTH = 0.45
"""Players recover to roughly the middle of their half between shots."""


def _y_for(player: PlayerId, depth: float) -> float:
    return 0.5 - depth * 0.5 if player == "A" else 0.5 + depth * 0.5


def _depth_of(player: PlayerId, y: float) -> float:
    return (0.5 - y) * 2 if player == "A" else (y - 0.5) * 2


def _other(player: PlayerId) -> PlayerId:
    return "B" if player == "A" else "A"


def _zone(depth: float) -> str:
    return "deep" if depth > 0.7 else "front" if depth < 0.32 else "mid"


def _pick(rng: random.Random, weights: dict[ShotType, float]) -> ShotType:
    types = list(weights)
    return rng.choices(types, weights=[weights[t] for t in types])[0]


def _smoothstep(u: float) -> float:
    return u * u * (3 - 2 * u)


def _interpolate(keyframes: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    """Smooth interpolation between time-sorted (t, x, y) keyframes."""
    if t <= keyframes[0][0]:
        return keyframes[0][1], keyframes[0][2]
    for (t0, x0, y0), (t1, x1, y1) in zip(keyframes, keyframes[1:], strict=False):
        if t <= t1:
            u = _smoothstep((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return keyframes[-1][1], keyframes[-1][2]


def _point(x: float, y: float) -> CourtPoint:
    return CourtPoint(x=round(x, 4), y=round(y, 4))


def generate_mock_rally(duration_sec: float, seed: int) -> RallyAnalysis:
    """Builds a deterministic, physically plausible rally that fits in `duration_sec`."""
    rng = random.Random(seed)
    duration = max(duration_sec, 0.1)
    rally_start = min(1.0, duration * 0.1)
    rally_end_limit = duration - min(0.3, duration * 0.05)

    positions: dict[PlayerId, tuple[float, float]] = {
        "A": (rng.uniform(0.38, 0.48), _y_for("A", 0.32)),
        "B": (rng.uniform(0.52, 0.62), _y_for("B", 0.4)),
    }
    keyframes: dict[PlayerId, list[tuple[float, float, float]]] = {
        p: [(0.0, *positions[p])] for p in ("A", "B")
    }

    shots: list[Shot] = []
    flights: list[tuple[float, float, tuple[float, float], tuple[float, float], float]] = []
    hitter: PlayerId = "A"
    t = rally_start

    while True:
        receiver = _other(hitter)
        if not shots:
            shot_type: ShotType = "serve"
        else:
            hitter_depth = _depth_of(hitter, positions[hitter][1])
            shot_type = _pick(rng, NEXT_SHOT_WEIGHTS[_zone(hitter_depth)])
        profile = PROFILES[shot_type]
        flight = rng.uniform(*profile.duration)

        if t + flight > rally_end_limit:
            if shots:
                break
            flight = max(rally_end_limit - t, 0.05)  # very short clip: one shortened shot

        start = positions[hitter]
        target = (rng.uniform(0.12, 0.88), _y_for(receiver, rng.uniform(*profile.depth)))
        end_t = t + flight
        flights.append((t, end_t, start, target, profile.ease))
        shots.append(
            Shot(
                index=len(shots),
                hitter=hitter,
                type=shot_type,
                start_time=round(t, 3),
                end_time=round(end_t, 3),
            )
        )

        # Hitter drifts back toward base; receiver moves to meet the shuttle.
        base = (0.5 + rng.uniform(-0.06, 0.06), _y_for(hitter, BASE_DEPTH))
        keyframes[hitter].append((t + flight * 0.6, *base))
        positions[hitter] = base
        keyframes[receiver].append((end_t, *target))
        positions[receiver] = target

        hitter = receiver
        t = end_t

    # The final shot is the winner: the receiver only gets most of the way there.
    last_receiver: PlayerId = _other(shots[-1].hitter)
    lt, lx, ly = keyframes[last_receiver][-1]
    px, py = keyframes[last_receiver][-2][1:]
    keyframes[last_receiver][-1] = (lt, px + (lx - px) * 0.75, py + (ly - py) * 0.75)

    shuttle = _shuttle_samples(rng, flights, duration)
    players = [
        PlayerTrack(
            id=pid,
            label=f"Player {pid}",
            samples=[
                PlayerSample(t=round(st, 3), position=_point(*_interpolate(keyframes[pid], st)))
                for st in _sample_times(0.0, duration, PLAYER_HZ)
            ],
        )
        for pid in ("A", "B")
    ]
    return RallyAnalysis(
        duration_sec=round(duration_sec, 3), players=players, shuttle=shuttle, shots=shots
    )


def _sample_times(start: float, end: float, hz: int) -> list[float]:
    n = int(math.floor((end - start) * hz + 1e-9))
    return [start + i / hz for i in range(n + 1)]


def _shuttle_samples(
    rng: random.Random,
    flights: list[tuple[float, float, tuple[float, float], tuple[float, float], float]],
    duration: float,
) -> list[ShuttleSample]:
    rally_start = flights[0][0]
    landing_t = flights[-1][1]
    landing = flights[-1][3]

    # A few short detection gaps, as a real tracker would have.
    gaps: list[tuple[float, float]] = []
    if landing_t - rally_start > 3:
        for _ in range(rng.randint(1, 3)):
            g0 = rng.uniform(rally_start + 0.5, landing_t - 0.5)
            gaps.append((g0, g0 + rng.uniform(0.1, 0.25)))

    samples: list[ShuttleSample] = []
    flight_i = 0
    for t in _sample_times(rally_start, duration, SHUTTLE_HZ):
        while flight_i < len(flights) - 1 and t > flights[flight_i][1]:
            flight_i += 1
        t0, t1, (x0, y0), (x1, y1), ease = flights[flight_i]

        if any(g0 <= t <= g1 for g0, g1 in gaps):
            position = None
        elif t >= landing_t:
            position = _point(*landing)
        else:
            u = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
            s = 1 - (1 - u) ** ease
            # Slight sideways bow so ground tracks don't look ruler-straight.
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy) or 1.0
            bow = math.sin(math.pi * u) * 0.02
            position = _point(
                x0 + dx * s - dy / length * bow,
                y0 + dy * s + dx / length * bow,
            )
        samples.append(ShuttleSample(t=round(t, 3), position=position))
    return samples


class MockAnalysisProvider:
    """Simulates pipeline stages with delays, then returns generated data."""

    def __init__(self, stage_seconds: float = 1.2) -> None:
        self._stage_seconds = stage_seconds

    @property
    def stages(self) -> Sequence[str]:
        return MOCK_STAGES

    def analyze(self, request: AnalysisRequest, on_progress: ProgressCallback) -> RallyAnalysis:
        steps_per_stage = 10
        total = len(MOCK_STAGES) * steps_per_stage
        for stage_index in range(len(MOCK_STAGES)):
            for step in range(steps_per_stage):
                if self._stage_seconds > 0:
                    time.sleep(self._stage_seconds / steps_per_stage)
                on_progress(stage_index, (stage_index * steps_per_stage + step + 1) / total)

        duration = request.duration_sec or DEFAULT_DURATION_SEC
        # Same file + duration gives the same rally, which makes manual testing repeatable.
        seed = zlib.crc32(f"{request.video_path.stat().st_size}:{duration:.3f}".encode())
        return generate_mock_rally(duration, seed)
