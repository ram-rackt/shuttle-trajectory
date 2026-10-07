"""Analysis job endpoints.

POST /api/analyses                    upload a video, start a job          -> 202 + status
GET  /api/analyses/{id}               poll job status and progress
GET  /api/analyses/{id}/result        the RallyAnalysis once status is "ready"
POST /api/analyses/{id}/calibration   re-run on the same video with manually
                                      marked court corners                  -> 202 + new job
"""

import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from pydantic import Field

from ..analysis import AnalysisProvider, AnalysisRequest, RallyAnalysis
from ..analysis.contract import CamelModel, ImagePoint
from ..config import Settings
from ..jobs import Job, JobStatus, JobStore, run_job

router = APIRouter(prefix="/api/analyses", tags=["analyses"])

ALLOWED_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm"}
CHUNK_BYTES = 1024 * 1024
MAX_DURATION_SEC = 3 * 60 * 60


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


class CourtCalibrationRequest(CamelModel):
    corners: list[ImagePoint] = Field(min_length=4, max_length=4)
    """Outer court corners in normalized image coordinates, ordered near-left,
    near-right, far-right, far-left (near = closest to the camera)."""


def _status(job: Job, provider: AnalysisProvider) -> AnalysisJobStatus:
    return AnalysisJobStatus(
        id=job.id,
        status=job.status,
        progress=round(job.progress, 3),
        stages=list(provider.stages),
        current_stage=job.current_stage,
        error=job.error,
        error_code=job.error_code,
    )


def _deps(request: Request) -> tuple[Settings, JobStore, AnalysisProvider]:
    state = request.app.state
    return state.settings, state.jobs, state.provider


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


@router.post("", status_code=202, response_model=AnalysisJobStatus)
async def create_analysis(
    request: Request,
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(description="Rally video (MP4, MOV or WebM)")],
    duration_sec: Annotated[float | None, Form(gt=0, le=MAX_DURATION_SEC)] = None,
    court_corners: Annotated[str | None, Form(description="Optional JSON corner list")] = None,
) -> AnalysisJobStatus:
    settings, jobs, provider = _deps(request)

    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, "Unsupported video format. Use MP4, MOV or WebM.")
    corners = _parse_corners(court_corners)

    job_id = jobs.new_id()
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{job_id}{suffix}"

    written = 0
    try:
        with video_path.open("wb") as out:
            while chunk := await video.read(CHUNK_BYTES):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(413, "Video is too large.")
                out.write(chunk)
    except BaseException:
        video_path.unlink(missing_ok=True)
        raise
    if written == 0:
        video_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded video is empty.")

    job = Job(
        id=job_id,
        request=AnalysisRequest(
            video_path=video_path, duration_sec=duration_sec, court_corners=corners
        ),
    )
    jobs.add(job)
    # Sync task -> Starlette runs it in a worker thread after the response is sent.
    background_tasks.add_task(run_job, jobs, provider, job_id)
    return _status(job, provider)


@router.post("/{job_id}/calibration", status_code=202, response_model=AnalysisJobStatus)
def recalibrate(
    request: Request,
    background_tasks: BackgroundTasks,
    job_id: str,
    body: CourtCalibrationRequest,
) -> AnalysisJobStatus:
    _, jobs, provider = _deps(request)
    original = jobs.get(job_id)
    if original is None:
        raise HTTPException(404, "Analysis not found.")
    if not original.request.video_path.exists():
        raise HTTPException(410, "The video for this analysis is no longer available.")
    job = Job(
        id=jobs.new_id(),
        request=AnalysisRequest(
            video_path=original.request.video_path,
            duration_sec=original.request.duration_sec,
            court_corners=_corners_tuple(body),
        ),
    )
    jobs.add(job)
    background_tasks.add_task(run_job, jobs, provider, job.id)
    return _status(job, provider)


@router.get("/{job_id}", response_model=AnalysisJobStatus)
def get_analysis(request: Request, job_id: str) -> AnalysisJobStatus:
    _, jobs, provider = _deps(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Analysis not found.")
    return _status(job, provider)


@router.get("/{job_id}/result", response_model=RallyAnalysis, response_model_exclude_none=False)
def get_analysis_result(request: Request, job_id: str) -> RallyAnalysis:
    _, jobs, _ = _deps(request)
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Analysis not found.")
    if job.status != "ready" or job.result is None:
        raise HTTPException(409, f"Analysis is not ready (status: {job.status}).")
    return job.result
