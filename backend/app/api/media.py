"""Video files and YouTube link checks.

GET /api/videos/{id}/file   the stored video (supports HTTP range requests, so
                            the browser can seek without downloading it all)
GET /api/youtube/info?url=  checks a YouTube link and returns what the UI shows
                            before analysing (title, length, thumbnail)
"""

import re

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse

from ..analysis.contract import CamelModel

router = APIRouter(prefix="/api", tags=["media"])

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
}


@router.get("/videos/{video_id}/file")
def video_file(request: Request, video_id: str) -> FileResponse:
    if not _SHA256.match(video_id):  # ids are content hashes; nothing else is a path
        raise HTTPException(404, "Video not found.")
    storage = request.app.state.storage
    video = storage.video(video_id)
    path = storage.video_path(video) if video else None
    if path is None or not path.exists():
        raise HTTPException(404, "Video not found.")
    return FileResponse(
        path,
        media_type=_MEDIA_TYPES.get(video.ext, "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=86400"},
    )


class YouTubeInfo(CamelModel):
    id: str
    url: str
    title: str
    channel: str | None
    duration_sec: float
    thumbnail_url: str | None
    height: int | None
    max_video_seconds: float
    """Longest part analysed at once; longer videos need a start and end time."""
    needs_range: bool


@router.get("/youtube/info", response_model=YouTubeInfo)
def youtube_info(request: Request, url: str = Query(max_length=2048)):
    from .. import youtube

    settings = request.app.state.settings
    if not settings.youtube_enabled or not youtube.available():
        return JSONResponse(
            status_code=503,
            content={
                "detail": "YouTube links aren't enabled on this server.",
                "code": "youtube_unavailable",
            },
        )
    try:
        info = youtube.fetch_info(
            youtube.video_id_from_url(url), settings.youtube_max_source_seconds
        )
    except youtube.YouTubeError as e:
        return JSONResponse(status_code=e.status, content={"detail": e.message, "code": e.code})
    if info.filesize and info.filesize > settings.max_upload_bytes:
        return JSONResponse(
            status_code=413,
            content={"detail": "The video is too large to download.", "code": "too_large"},
        )
    return YouTubeInfo(
        id=info.id,
        url=info.url,
        title=info.title,
        channel=info.channel,
        duration_sec=info.duration_sec,
        thumbnail_url=info.thumbnail_url,
        height=info.height,
        max_video_seconds=settings.max_video_seconds,
        needs_range=info.duration_sec > settings.max_video_seconds + 1,
    )
