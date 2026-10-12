"""Analysis job endpoints.

POST /api/analyses                    upload a video, start a job          -> 202 + status
POST /api/analyses/youtube            analyse a YouTube video (by link)    -> 202 + status
GET  /api/analyses/{id}               poll job status and progress
GET  /api/analyses/{id}/result        the RallyAnalysis once status is "ready"
POST /api/analyses/{id}/calibration   re-run on the same video with manually
                                      marked court corners                  -> 202 + new job
POST /api/analyses/{id}/cancel        stop a queued or running job

A video already analysed with the same settings is answered from the result
cache: the job is created already "ready" (cached: true).

The create endpoints accept an optional client-chosen job id (32 hex digits),
so a client that gives up while its create request is still in flight can
cancel the job by that id; see JobStore.
"""

import json
import math
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field

from ..analysis import AnalysisProvider, RallyAnalysis
from ..analysis.contract import CamelModel, ImagePoint
from ..config import Settings
from ..jobs import (
    DOWNLOAD_STAGE,
    JOB_ID_PATTERN,
    DuplicateJobId,
    Job,
    JobRequest,
    JobRunner,
    JobStatus,
    JobStore,
    result_key,
    youtube_source_key,
)
from ..storage import Storage
from .uploads import receive_upload

router = APIRouter(prefix="/api/analyses", tags=["analyses"])

ALLOWED_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm"}
MIN_WINDOW_SEC = 2.0

UPLOAD_FORM = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["video"],
                    "properties": {
                        "video": {
                            "type": "string",
                            "format": "binary",
                            "description": "Rally video (MP4, MOV or WebM)",
                        },
                        "duration_sec": {"type": "number", "exclusiveMinimum": 0},
                        "court_corners": {
                            "type": "string",
                            "description": "Optional JSON list of 4 {x, y} corners",
                        },
                        "job_id": {"type": "string", "pattern": JOB_ID_PATTERN},
                    },
                }
            }
        },
    }
}
"""The upload form, for the API docs (the body is parsed by hand, streamed)."""


class VideoInfo(CamelModel):
    id: str
    url: str
    """Where the browser can play the stored video."""
    name: str
    size_bytes: int
    source: str
    """'upload' or 'youtube'."""
    youtube_url: str | None = None
    channel: str | None = None


class TimeWindow(CamelModel):
    start_sec: float
    end_sec: float


class AnalysisJobStatus(CamelModel):
    id: str
    status: JobStatus
    progress: float
    """Overall progress, 0 to 1."""
    stages: list[str]
    current_stage: int | None
    """Index into `stages` while processing, otherwise null."""
    error: str | None
    error_code: str | None = None
    """Machine-readable failure reason, e.g. 'court_not_found' (the UI then offers
    manual court calibration)."""
    cached: bool = False
    """The result was reused from an earlier identical analysis."""
    video: VideoInfo | None = None
    """The stored video (for YouTube jobs, once it has been downloaded)."""
    window: TimeWindow | None = None
    """The part of the video analysed, when not all of it."""


class CourtCalibrationRequest(CamelModel):
    corners: list[ImagePoint] = Field(min_length=4, max_length=4)
    """Outer court corners in normalized image coordinates, ordered near-left,
    near-right, far-right, far-left (near = closest to the camera)."""
    job_id: str | None = Field(default=None, pattern=JOB_ID_PATTERN)
    """Optional id for the new job, chosen by the client."""


class YouTubeAnalysisRequest(CamelModel):
    url: str = Field(max_length=2048)
    start_sec: float | None = Field(default=None, ge=0)
    end_sec: float | None = Field(default=None, gt=0)
    job_id: str | None = Field(default=None, pattern=JOB_ID_PATTERN)
    """Optional id for the new job, chosen by the client."""


def _deps(request: Request) -> tuple[Settings, Storage, JobStore, JobRunner, AnalysisProvider]:
    s = request.app.state
    return s.settings, s.storage, s.jobs, s.runner, s.provider


def _video_info(storage: Storage, video_id: str | None) -> VideoInfo | None:
    video = storage.video(video_id) if video_id else None
    if video is None:
        return None
    return VideoInfo(
        id=video.id,
        url=f"/api/videos/{video.id}/file",
        name=video.name,
        size_bytes=video.size_bytes,
        source=video.source,
        youtube_url=(
            f"https://www.youtube.com/watch?v={video.youtube_id}" if video.youtube_id else None
        ),
        channel=video.channel,
    )


def _status(job: Job, storage: Storage) -> AnalysisJobStatus:
    r = job.request
    window = (
        TimeWindow(start_sec=r.start_sec or 0.0, end_sec=r.end_sec)
        if r.end_sec is not None
        else None
    )
    return AnalysisJobStatus(
        id=job.id,
        status=job.status,
        progress=round(job.progress, 3),
        stages=job.stages,
        current_stage=job.current_stage,
        error=job.error,
        error_code=job.error_code,
        cached=job.cached,
        video=_video_info(storage, job.video_id),
        window=window,
    )


def _error(status: int, code: str, message: str) -> JSONResponse:
    """Errors the UI can act on carry a machine-readable code next to the message."""
    return JSONResponse(status_code=status, content={"detail": message, "code": code})


def _parse_corners(raw: str | None) -> tuple[tuple[float, float], ...] | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw)
        model = CourtCalibrationRequest.model_validate({"corners": data})
    except (ValueError, TypeError) as e:
        raise HTTPException(422, "court_corners must be a JSON list of 4 {x, y} points.") from e
    return _corners_tuple(model)


def _corners_tuple(model: CourtCalibrationRequest) -> tuple[tuple[float, float], ...]:
    pts = tuple((p.x, p.y) for p in model.corners)
    if not all(0 <= v <= 1 for p in pts for v in p):
        raise HTTPException(422, "Corner coordinates must be normalized to 0-1.")
    return pts


def _start(
    job: Job,
    storage: Storage,
    jobs: JobStore,
    runner: JobRunner,
    provider: AnalysisProvider,
    background_tasks: BackgroundTasks,
) -> AnalysisJobStatus:
    """Answers from the result cache when possible, otherwise queues the job."""
    if job.video_id is not None:
        key = result_key(storage, provider, job.video_id, job.request)
        if storage.result(key) is not None:
            job.status, job.progress, job.result_key, job.cached = "ready", 1.0, key, True
            storage.touch_video(job.video_id)
    try:
        jobs.add(job)  # comes back "failed" if the client already cancelled it
    except DuplicateJobId:
        raise HTTPException(409, "An analysis with this id already exists.") from None
    if job.status == "queued":
        # Sync task -> Starlette runs it in a worker thread after the response is sent.
        background_tasks.add_task(runner.run, job.id)
    return _status(job, storage)


def _parse_duration(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise HTTPException(422, "duration_sec must be a positive number of seconds.")
    return value


def _check_duration(duration_sec: float | None, settings: Settings) -> None:
    if duration_sec is not None and duration_sec > settings.max_video_seconds + 1:
        minutes = settings.max_video_seconds / 60
        raise HTTPException(
            422,
            f"The video is {duration_sec / 60:.1f} minutes long; "
            f"the limit is {minutes:.0f} minutes.",
        )


def _job_id(requested: str | None, jobs: JobStore) -> str:
    if requested is None or requested == "":
        return jobs.new_id()
    if not jobs.valid_id(requested):
        raise HTTPException(422, "job_id must be 32 lowercase hex digits.")
    return requested


@router.post("", status_code=202, response_model=AnalysisJobStatus, openapi_extra=UPLOAD_FORM)
async def create_analysis(request: Request, background_tasks: BackgroundTasks) -> AnalysisJobStatus:
    """Upload a video (multipart form: `video`, optional `duration_sec`,
    `court_corners`, `job_id`) and start a job on it.

    The body is streamed into storage/tmp/ and hashed on the way, so a re-upload
    of the same video is recognised without holding it in memory, and an
    upload over the size limit is stopped as soon as it gets there. Fields
    sent before the video (the web app does) are checked before any of it is
    stored.
    """
    settings, storage, jobs, runner, provider = _deps(request)

    def check_file(filename: str, fields: dict[str, str]) -> None:
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, "Unsupported video format. Use MP4, MOV or WebM.")
        _check_duration(_parse_duration(fields.get("duration_sec")), settings)

    tmp_path = storage.new_tmp_path()
    try:
        upload = await receive_upload(
            request, tmp_path, settings.max_upload_bytes, "video", check_file
        )
        if upload.filename is None:
            raise HTTPException(422, "The form has no video.")
        if upload.size == 0:
            raise HTTPException(400, "Uploaded video is empty.")
        duration_sec = _parse_duration(upload.fields.get("duration_sec"))
        _check_duration(duration_sec, settings)
        corners = _parse_corners(upload.fields.get("court_corners"))
        job_id = _job_id(upload.fields.get("job_id"), jobs)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    stored = storage.add_video(
        tmp_path,
        upload.sha256,
        Path(upload.filename).suffix.lower(),
        name=upload.filename or "video",
        source="upload",
    )
    job = Job(
        id=job_id,
        kind="upload",
        request=JobRequest(duration_sec=duration_sec, court_corners=corners),
        stages=list(provider.stages),
        video_id=stored.id,
    )
    return _start(job, storage, jobs, runner, provider, background_tasks)


@router.post("/youtube", status_code=202, response_model=AnalysisJobStatus)
def create_youtube_analysis(
    request: Request, background_tasks: BackgroundTasks, body: YouTubeAnalysisRequest
):
    settings, storage, jobs, runner, provider = _deps(request)
    from .. import youtube

    if not settings.youtube_enabled or not youtube.available():
        return _error(503, "youtube_unavailable", "YouTube links aren't enabled on this server.")
    try:
        video_id = youtube.video_id_from_url(body.url)
    except youtube.YouTubeError as e:
        return _error(e.status, e.code, e.message)

    # Checked before queueing, so problems (private, too long...) are reported now.
    downloaded = storage.downloaded(youtube_source_key(video_id))
    if downloaded is not None:
        duration = None  # already stored and checked; the pipeline enforces the window
    else:
        try:
            info = youtube.fetch_info(video_id, settings.youtube_max_source_seconds)
        except youtube.YouTubeError as e:
            return _error(e.status, e.code, e.message)
        duration = info.duration_sec
        if info.filesize and info.filesize > settings.max_upload_bytes:
            return _error(413, "too_large", "The video is too large to download.")

    problem = _check_window(body.start_sec, body.end_sec, duration, settings.max_video_seconds)
    if problem:
        return _error(422, *problem)

    job = Job(
        id=body.job_id or jobs.new_id(),
        kind="youtube",
        request=JobRequest(
            start_sec=body.start_sec if body.end_sec is not None else None,
            end_sec=body.end_sec,
            youtube_id=video_id,
        ),
        stages=[DOWNLOAD_STAGE, *provider.stages],
        video_id=downloaded.id if downloaded else None,
    )
    return _start(job, storage, jobs, runner, provider, background_tasks)


def _check_window(
    start: float | None, end: float | None, duration: float | None, max_seconds: float
) -> tuple[str, str] | None:
    """(code, message) if the requested part of the video can't be analysed."""
    minutes = max_seconds / 60
    if end is None:
        if start is not None:
            return "invalid_range", "Give an end time as well as a start time."
        if duration is not None and duration > max_seconds + 1:
            return (
                "range_required",
                f"The video is longer than {minutes:.0f} minutes. Choose the part to analyse "
                f"(up to {minutes:.0f} minutes).",
            )
        return None
    start = start or 0.0
    if end <= start:
        return "invalid_range", "The end time must be after the start time."
    if end - start < MIN_WINDOW_SEC:
        return "invalid_range", f"Choose at least {MIN_WINDOW_SEC:.0f} seconds."
    if end - start > max_seconds + 1:
        return "range_too_long", f"Choose at most {minutes:.0f} minutes."
    if duration is not None and end > duration + 1:
        return "invalid_range", "The end time is after the end of the video."
    return None


@router.post("/{job_id}/calibration", status_code=202, response_model=AnalysisJobStatus)
def recalibrate(
    request: Request,
    background_tasks: BackgroundTasks,
    job_id: str,
    body: CourtCalibrationRequest,
) -> AnalysisJobStatus:
    _, storage, jobs, runner, provider = _deps(request)
    original = jobs.get(job_id)
    if original is None:
        raise HTTPException(404, "Analysis not found.")
    video = storage.video(original.video_id) if original.video_id else None
    if video is None or not storage.video_path(video).exists():
        raise HTTPException(410, "The video for this analysis is no longer available.")
    r = original.request
    job = Job(
        id=body.job_id or jobs.new_id(),
        kind=original.kind,
        request=JobRequest(
            duration_sec=r.duration_sec,
            court_corners=_corners_tuple(body),
            start_sec=r.start_sec,
            end_sec=r.end_sec,
            youtube_id=r.youtube_id,
        ),
        stages=original.stages,
        video_id=video.id,
    )
    return _start(job, storage, jobs, runner, provider, background_tasks)


@router.post("/{job_id}/cancel", response_model=AnalysisJobStatus)
def cancel_analysis(request: Request, job_id: str) -> AnalysisJobStatus:
    """Stops a queued or running job. 404 if it doesn't exist (yet): a client-
    chosen id whose create request is still in flight is remembered, and that
    job is created already cancelled."""
    _, storage, jobs, _, _ = _deps(request)
    job = jobs.cancel(job_id)
    if job is None:
        raise HTTPException(404, "Analysis not found.")
    return _status(job, storage)


@router.get("/{job_id}", response_model=AnalysisJobStatus)
def get_analysis(request: Request, job_id: str) -> AnalysisJobStatus:
    _, storage, jobs, _, _ = _deps(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Analysis not found.")
    return _status(job, storage)


@router.get("/{job_id}/result", response_model=RallyAnalysis, response_model_exclude_none=False)
def get_analysis_result(request: Request, job_id: str) -> RallyAnalysis:
    _, _, jobs, _, _ = _deps(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Analysis not found.")
    result = jobs.result(job) if job.status == "ready" else None
    if result is None:
        raise HTTPException(409, f"Analysis is not ready (status: {job.status}).")
    return result
