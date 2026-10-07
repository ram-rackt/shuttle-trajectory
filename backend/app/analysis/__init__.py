"""Analysis layer: the data contract and the providers that produce it.

The API only depends on `AnalysisProvider` and `RallyAnalysis`. Providers:
`MockAnalysisProvider` (simulated data, for UI work) and `CVAnalysisProvider`
(the computer-vision pipeline in the `rallycv` package).
"""

from .contract import RallyAnalysis
from .provider import AnalysisError, AnalysisProvider, AnalysisRequest, ProgressCallback

__all__ = [
    "AnalysisError",
    "AnalysisProvider",
    "AnalysisRequest",
    "ProgressCallback",
    "RallyAnalysis",
]
