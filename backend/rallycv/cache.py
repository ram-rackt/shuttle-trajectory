"""On-disk cache for the expensive per-video pipeline stages.

The neural networks (TrackNet, the person detector) are ~90% of the cost of
an analysis; everything after them takes milliseconds. Their outputs are
stored per video, keyed by everything that affects them, so a re-run (manual
court marking, a newer version of the shot logic) skips them.
"""

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CACHE_FORMAT = 1
"""Bump to invalidate every cached stage (e.g. when what is stored changes)."""


def stage_key(stage: str, **params: object) -> str:
    """A file-safe key for a stage run with these parameters."""
    blob = json.dumps({"format": CACHE_FORMAT, **params}, sort_keys=True, default=str)
    return f"{stage}-{hashlib.sha256(blob.encode()).hexdigest()[:24]}"


class StageCache:
    """Named sets of numpy arrays in one directory (one directory per video)."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def has(self, key: str) -> bool:
        return (self._dir / f"{key}.npz").exists()

    def load(self, key: str) -> dict[str, np.ndarray] | None:
        path = self._dir / f"{key}.npz"
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                return {name: data[name] for name in data.files}
        except (OSError, ValueError) as e:  # truncated or corrupt: recompute
            logger.warning("Ignoring unreadable cache entry %s: %s", path.name, e)
            return None

    def save(self, key: str, **arrays: np.ndarray) -> None:
        """Writes atomically, so a crash never leaves a half-written entry."""
        self._dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez_compressed(f, **arrays)
            os.replace(tmp, self._dir / f"{key}.npz")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
