"""Pydantic form of the analysis contract (schema v1).

Mirrors frontend/src/types/analysis.ts; see docs/analysis-contract.md.
Field names are snake_case in Python and camelCase on the wire.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

ANALYSIS_SCHEMA_VERSION = 1

PlayerId = Literal["A", "B"]
ShotType = Literal["serve", "clear", "drop", "smash", "drive", "net", "lift", "unknown"]


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)


class CourtPoint(CamelModel):
    """Normalized court position. x across the width, y along the length (net at 0.5)."""

    x: float
    y: float


class PlayerSample(CamelModel):
    t: float
    position: CourtPoint


class PlayerTrack(CamelModel):
    id: PlayerId
    label: str
    samples: list[PlayerSample]


class ImagePoint(CamelModel):
    """Normalized video-frame position: x, y in [0, 1] from the top-left corner."""

    x: float
    y: float


class ShuttleSample(CamelModel):
    t: float
    position: CourtPoint | None
    """Floor (top-down) position; None when unknown at this time."""
    image: ImagePoint | None = None
    """Where the shuttle was detected in the video frame; None when not seen."""
    height: float | None = None
    """Estimated height above the floor in metres, when it could be reconstructed."""


class Shot(CamelModel):
    index: int
    hitter: PlayerId
    type: ShotType
    start_time: float
    end_time: float


class CameraCalibration(CamelModel):
    court_to_image: list[float]
    """Row-major 3x3 homography from normalized court coordinates (x, y in 0-1)
    to normalized image coordinates (0-1). Lets the UI draw the court on the video."""
    confidence: float
    """Fraction of the court's painted lines confirmed in the image (0-1)."""
    manual: bool = False
    """True if the court was marked by the user rather than detected."""


class RallyAnalysis(CamelModel):
    schema_version: Literal[1] = ANALYSIS_SCHEMA_VERSION
    source: Literal["mock", "cv"] = "mock"
    duration_sec: float
    players: list[PlayerTrack]
    shuttle: list[ShuttleSample]
    shots: list[Shot]
    calibration: CameraCalibration | None = None
