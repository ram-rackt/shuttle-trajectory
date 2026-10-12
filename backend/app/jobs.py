"""Analysis jobs, persisted in SQLite (see storage.py), and the job runner.

A job analyses one video: an upload (already stored when the job is created)
or a YouTube video (downloaded by the job's first stage). Finished results
are stored under a key built from the video's content hash, the analyser's
version and the request settings, so the same analysis is never run twice.

Jobs run in worker threads of this process. That is fine for one machine;
a hosted deployment would use a queue and separate GPU workers.
"""

import json
import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

from .analysis import AnalysisError, AnalysisProvider, AnalysisRequest, RallyAnalysis
from .config import Settings
from .storage import Storage, VideoRecord
from .youtube import FORMAT_VERSION, DownloadCancelled, YouTubeError

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "processing", "ready", "failed"]
JobKind = Literal["upload", "youtube"]

DOWNLOAD_STAGE = "Downloading video"
DOWNLOAD_SHARE = 0.15
"""Fraction of a YouTube job's progress bar given to the download."""

JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
"""Job ids: 32 lowercase hex digits. Clients may choose the id of a job they
create, so they can cancel it before the create request has even returned."""
_JOB_ID = re.compile(JOB_ID_PATTERN)
EARLY_CANCEL_TTL_SEC = 600.0
"""How long a cancel for a job that doesn't exist yet is remembered."""
EARLY_CANCEL_MAX = 1000


@dataclass(frozen=True)
class JobRequest:
    """What the analysis was asked for. Everything here is part of the result key."""

    duration_sec: float | None = None
    """Duration reported by the browser (the mock analyser uses it)."""
    court_corners: tuple[tuple[float, float], ...] | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    """Analyse only this part of the video."""
    youtube_id: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "JobRequest":
        data = json.loads(text)
        corners = data.get("court_corners")
        data["court_corners"] = tuple(tuple(c) for c in corners) if corners else None
        return cls(**data)

    def cache_fields(self) -> dict:
        return {
            "duration_sec": None if self.duration_sec is None else round(self.duration_sec, 3),
            "court_corners": self.court_corners,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }


@dataclass
class Job:
    id: str
    kind: JobKind
    request: JobRequest
    stages: list[str]
    video_id: str | None = None
    status: JobStatus = "queued"
    progress: float = 0.0
    current_stage: int | None = None
    error: str | None = None
    error_code: str | None = None
    result_key: str | None = None
    cached: bool = False
    cancel: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)


class DuplicateJobId(Exception):
    pass


class JobStore:
    """Jobs in the database, plus cancellation flags for jobs running here.

    A cancel can arrive before its job exists: the client chose the id and
    gave up while the create request was still being handled (e.g. while a
    YouTube link was being checked). It is remembered, and the job is then
    created already cancelled instead of running for nobody.
    """

    def __init__(self, storage: Storage) -> None:
        self._storage = storage
        self._cancel: dict[str, threading.Event] = {}
        self._early_cancels: dict[str, float] = {}  # job id -> when (monotonic)
        # Re-entrant: cancel() reads the job while holding it. Held across
        # "check for an early cancel + insert" so a cancel can't slip in between.
        self._lock = threading.RLock()

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def valid_id(job_id: str) -> bool:
        return bool(_JOB_ID.match(job_id))

    def add(self, job: Job) -> None:
        """Stores a new job. If it was cancelled before it existed, it is stored
        as cancelled (check job.status afterwards before running it)."""
        with self._lock:
            if self._storage.job_row(job.id) is not None:
                raise DuplicateJobId(job.id)
            if self._early_cancels.pop(job.id, None) is not None and job.status == "queued":
                job.status, job.error, job.error_code = "failed", "Cancelled.", "cancelled"
            if job.status == "queued":
                self._cancel[job.id] = job.cancel
            self._storage.insert_job(
                {
                    "id": job.id,
                    "kind": job.kind,
                    "video_id": job.video_id,
                    "status": job.status,
                    "progress": job.progress,
                    "current_stage": job.current_stage,
                    "stages": json.dumps(job.stages),
                    "error": job.error,
                    "error_code": job.error_code,
                    "request": job.request.to_json(),
                    "result_key": job.result_key,
                    "cached": int(job.cached),
                }
            )

    def cancel(self, job_id: str) -> Job | None:
        """Stops a queued or running job and returns it (None if there is no
        such job yet; a well-formed id is then remembered, see the class doc)."""
        with self._lock:
            job = self.get(job_id)
            if job is None:
                if self.valid_id(job_id):
                    self._remember_early_cancel(job_id)
                return None
            if job.status in ("queued", "processing") and not self.request_cancel(job_id):
                # Not running in this process (e.g. queued before a restart): just mark it.
                self.update(job_id, status="failed", error="Cancelled.", error_code="cancelled")
            return self.get(job_id)

    def _remember_early_cancel(self, job_id: str) -> None:
        now = time.monotonic()
        cutoff = now - EARLY_CANCEL_TTL_SEC
        for old in [k for k, t in self._early_cancels.items() if t < cutoff]:
            del self._early_cancels[old]
        while len(self._early_cancels) >= EARLY_CANCEL_MAX:
            del self._early_cancels[next(iter(self._early_cancels))]  # oldest first
        self._early_cancels[job_id] = now

    def get(self, job_id: str) -> Job | None:
        row = self._storage.job_row(job_id)
        if row is None:
            return None
        with self._lock:
            cancel = self._cancel.get(job_id) or threading.Event()
        return Job(
            id=row["id"],
            kind=row["kind"],
            request=JobRequest.from_json(row["request"]),
            stages=json.loads(row["stages"]),
            video_id=row["video_id"],
            status=row["status"],
            progress=row["progress"],
            current_stage=row["current_stage"],
            error=row["error"],
            error_code=row["error_code"],
            result_key=row["result_key"],
            cached=bool(row["cached"]),
            cancel=cancel,
        )

    def update(self, job_id: str, **changes: object) -> None:
        if "cached" in changes:
            changes["cached"] = int(bool(changes["cached"]))
        self._storage.update_job(job_id, **changes)

    def request_cancel(self, job_id: str) -> bool:
        """Asks a running job to stop. False if it isn't running in this process."""
        with self._lock:
            event = self._cancel.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def finish(self, job_id: str) -> None:
        with self._lock:
            self._cancel.pop(job_id, None)

    def result(self, job: Job) -> RallyAnalysis | None:
        if job.result_key is None:
            return None
        text = self._storage.result(job.result_key)
        return RallyAnalysis.model_validate_json(text) if text else None


def analyzer_version(provider: AnalysisProvider) -> str:
    """Identifies what produced a result, for the result cache."""
    return str(getattr(provider, "version", type(provider).__name__))


def result_key(
    storage: Storage, provider: AnalysisProvider, video_id: str, request: JobRequest
) -> str:
    return storage.result_key(video_id, analyzer_version(provider), request.cache_fields())


def youtube_source_key(youtube_id: str) -> str:
    return f"youtube:{youtube_id}:{FORMAT_VERSION}"


class JobRunner:
    """Runs jobs: downloads (YouTube), then analysis, with results cached."""

    def __init__(
        self, store: JobStore, storage: Storage, provider: AnalysisProvider, settings: Settings
    ) -> None:
        self.store = store
        self.storage = storage
        self.provider = provider
        self.settings = settings
        self._downloads = threading.BoundedSemaphore(settings.youtube_max_concurrent_downloads)
        self._active: dict[str, str | None] = {}
        """Jobs in run() (queued, running or waiting for the analyser) -> their
        video id (None while a YouTube video is still unknown/downloading)."""
        self._active_lock = threading.Lock()

    def run(self, job_id: str) -> None:
        """Runs one job to completion. Called from a worker thread."""
        job = self.store.get(job_id)
        if job is None or job.status != "queued":
            return
        with self._active_lock:
            self._active[job_id] = job.video_id
        court_not_found = False
        try:
            self._run(job)
        except (AnalysisError, YouTubeError) as e:
            logger.info("Job %s failed: %s (%s)", job_id, e.message, e.code)
            self._fail(job_id, e.code, e.message)
            court_not_found = e.code == "court_not_found"
        except DownloadCancelled:
            self._fail(job_id, "cancelled", "The download was cancelled.")
        except Exception:
            logger.exception("Job %s crashed", job_id)
            self._fail(job_id, "internal_error", "Analysis failed unexpectedly. Please try again.")
        finally:
            self.store.finish(job_id)
            with self._active_lock:
                self._active.pop(job_id, None)
        if court_not_found:
            self._warm_up(job)

    def _warm_up(self, job: Job) -> None:
        """The user is now marking the court corners: meanwhile, precompute what
        the re-run will need that doesn't depend on the court, so it is quicker.
        Gives way as soon as a job for another video is waiting."""
        warm_up = getattr(self.provider, "warm_up", None)
        video = self.storage.video(job.video_id) if job.video_id else None
        if warm_up is None or video is None:
            return

        def should_stop() -> bool:
            with self._active_lock:
                return any(v != video.id for v in self._active.values())

        try:
            warm_up(self._request(job, video), should_stop)
        except Exception:
            logger.exception("Warming up the cache for video %s failed", video.id)

    def _request(self, job: Job, video: VideoRecord) -> AnalysisRequest:
        return AnalysisRequest(
            video_path=self.storage.video_path(video),
            duration_sec=job.request.duration_sec,
            court_corners=job.request.court_corners,
            start_sec=job.request.start_sec,
            end_sec=job.request.end_sec,
            cache_dir=self.storage.cache_dir(video.id),
        )

    def _fail(self, job_id: str, code: str, message: str) -> None:
        self.store.update(
            job_id, status="failed", current_stage=None, error=message, error_code=code
        )

    def _run(self, job: Job) -> None:
        self.store.update(job.id, status="processing", current_stage=0, progress=0.0)
        offset, share = 0, 0.0
        if job.kind == "youtube":
            offset, share = 1, DOWNLOAD_SHARE
            if job.video_id is None:
                video = self._download(job)
                job.video_id = video.id
                self.store.update(job.id, video_id=video.id)
                with self._active_lock:
                    self._active[job.id] = video.id
            self.store.update(job.id, current_stage=1, progress=share)

        video = self.storage.video(job.video_id or "")
        if video is None or not self.storage.video_path(video).exists():
            raise AnalysisError("video_missing", "The video for this analysis is no longer stored.")
        self.storage.touch_video(video.id)

        key = result_key(self.storage, self.provider, video.id, job.request)
        if self.storage.result(key) is not None:
            self._ready(job.id, key, cached=True)
            return

        def on_progress(stage_index: int, progress: float) -> None:
            # Progress reports double as cancellation points for the analysis.
            if job.cancel.is_set():
                raise AnalysisError("cancelled", "The analysis was cancelled.")
            p = min(max(progress, 0.0), 1.0)
            self.store.update(
                job.id, current_stage=stage_index + offset, progress=share + (1 - share) * p
            )

        result = self.provider.analyze(self._request(job, video), on_progress)
        self.storage.put_result(key, video.id, result.model_dump_json(by_alias=True))
        self._ready(job.id, key, cached=False)

    def _ready(self, job_id: str, key: str, cached: bool) -> None:
        self.store.update(
            job_id, status="ready", progress=1.0, current_stage=None, result_key=key, cached=cached
        )

    def _download(self, job: Job) -> VideoRecord:
        """Fetches the YouTube video into the store (or reuses an earlier download)."""
        from . import youtube

        youtube_id = job.request.youtube_id or ""
        source_key = youtube_source_key(youtube_id)
        existing = self.storage.downloaded(source_key)
        if existing is not None:
            return existing

        info = youtube.fetch_info(youtube_id, self.settings.youtube_max_source_seconds)
        work_dir = self.storage.new_tmp_path()
        reported = [0.0]

        def on_download(fraction: float) -> None:
            # yt-dlp reports many times a second; record whole percents only.
            if fraction - reported[0] >= 0.01 or fraction >= 1.0:
                reported[0] = fraction
                self.store.update(job.id, progress=DOWNLOAD_SHARE * fraction)

        try:
            with self._downloads:  # at most N downloads at once
                if job.cancel.is_set():
                    raise DownloadCancelled
                path = youtube.download(
                    youtube_id,
                    work_dir,
                    self.settings.max_upload_bytes,
                    on_progress=on_download,
                    cancelled=job.cancel,
                )
            sha = self.storage.hash_file(path)
            video = self.storage.add_video(
                path,
                sha,
                path.suffix.lower(),
                name=info.title,
                source="youtube",
                youtube_id=youtube_id,
                channel=info.channel,
            )
            self.storage.record_download(source_key, video.id)
            return video
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
