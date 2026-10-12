"""YouTube links: validation, metadata and download (via the optional yt-dlp).

Only the video stream is downloaded (no audio: the analysis doesn't need it,
and a single stream needs no ffmpeg to merge), as H.264 MP4 at up to 1080p,
which both OpenCV and every browser can play. A long video is downloaded
whole; the part to analyse is chosen as a time window, so no ffmpeg is needed
to cut it either, and analysis times match YouTube's own timestamps.

Guardrails: YouTube hosts only, one video per link (no playlists), no live
streams, no private / members-only / age-restricted videos, limits on length
and file size, socket timeouts, and cancellation.
"""

import logging
import re
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

FORMAT = (
    "bestvideo[height<=1080][vcodec^=avc1][ext=mp4][protocol^=http]"
    "/bestvideo[height<=1080][ext=mp4][protocol^=http]"
    "/best[height<=1080][ext=mp4][protocol^=http]"
)
FORMAT_VERSION = "v1"
"""Part of the download cache key; bump when FORMAT changes."""
MIN_HEIGHT = 360
"""Below this the shuttle is a pixel or two: too small to track."""

_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}


class YouTubeError(Exception):
    """A failure the user can understand. `code` is machine-readable."""

    def __init__(self, code: str, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class DownloadCancelled(Exception):
    pass


@dataclass(frozen=True)
class VideoInfo:
    id: str
    title: str
    channel: str | None
    duration_sec: float
    thumbnail_url: str | None
    height: int | None
    """Height of the stream that will be downloaded."""
    filesize: int | None
    """Its size in bytes, when YouTube reports it."""

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"


def available() -> bool:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return False
    return True


def video_id_from_url(url: str) -> str:
    """The 11-character video id of a YouTube link, or YouTubeError('invalid_url').

    Accepts watch, youtu.be, shorts, embed and live links. Anything else
    (other sites, playlists without a video, channels) is rejected here, so the
    downloader only ever sees a canonical single-video URL.
    """
    text = url.strip()
    if not text:
        raise YouTubeError("invalid_url", "Paste a YouTube link.")
    if "://" not in text:
        text = "https://" + text
    try:
        parsed = urlparse(text)
    except ValueError as e:
        raise YouTubeError("invalid_url", "That doesn't look like a link.") from e
    host = (parsed.hostname or "").lower()
    candidate = None
    if parsed.scheme not in ("http", "https"):
        raise YouTubeError("unsupported_site", "Only YouTube links are supported.")
    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host in _HOSTS:
        if parsed.path == "/watch":
            candidate = (parse_qs(parsed.query).get("v") or [""])[0]
        else:
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                candidate = parts[1]
    else:
        raise YouTubeError("unsupported_site", "Only YouTube links are supported.")
    if not candidate or not _ID.match(candidate):
        raise YouTubeError(
            "invalid_url",
            "That isn't a link to a single YouTube video (playlists aren't supported).",
        )
    return candidate


def _js_runtimes() -> dict:
    """YouTube needs a JavaScript runtime; yt-dlp only enables Deno by default."""
    runtimes = {}
    for name in ("deno", "node", "bun"):
        path = shutil.which(name)
        if path:
            runtimes[name] = {"path": path}
    return runtimes


def _base_options() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "js_runtimes": _js_runtimes(),
        "logger": _QuietLogger(),
    }


class _QuietLogger:
    """Routes yt-dlp's output to our log instead of stdout/stderr."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        logger.debug("yt-dlp: %s", msg)

    def error(self, msg: str) -> None:
        logger.info("yt-dlp: %s", msg)


def fetch_info(video_id: str, max_source_seconds: float) -> VideoInfo:
    """Metadata, after checking the video can be downloaded and analysed."""
    import yt_dlp

    opts = {**_base_options(), "skip_download": True, "format": FORMAT}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except yt_dlp.utils.DownloadError as e:
        raise _translate(e) from e
    if info is None:
        raise YouTubeError("video_unavailable", "That video isn't available.")

    live = info.get("live_status")
    if info.get("is_live") or live in ("is_live", "is_upcoming", "post_live"):
        raise YouTubeError(
            "live_video",
            "Live streams and premieres can't be analysed. Try again once the video is processed.",
        )
    if (info.get("age_limit") or 0) >= 18:
        raise YouTubeError("age_restricted", "Age-restricted videos can't be downloaded.")
    if info.get("availability") not in (None, "public", "unlisted"):
        raise YouTubeError(
            "video_restricted", "This video isn't public (private, members-only or paid)."
        )
    duration = info.get("duration")
    if not duration:
        raise YouTubeError("unknown_duration", "YouTube didn't report this video's length.")
    if duration > max_source_seconds:
        raise YouTubeError(
            "source_too_long",
            f"The video is {_hm(duration)} long; videos up to {_hm(max_source_seconds)} "
            "are supported.",
        )
    chosen = (info.get("requested_downloads") or [info])[0]
    height = chosen.get("height")
    if height is not None and height < MIN_HEIGHT:
        raise YouTubeError(
            "low_resolution",
            f"This video is only available at {height}p; at least {MIN_HEIGHT}p is needed "
            "to track the shuttle.",
        )
    return VideoInfo(
        id=video_id,
        title=info.get("title") or video_id,
        channel=info.get("channel") or info.get("uploader"),
        duration_sec=float(duration),
        thumbnail_url=info.get("thumbnail"),
        height=height,
        filesize=chosen.get("filesize") or chosen.get("filesize_approx"),
    )


def download(
    video_id: str,
    dest_dir: Path,
    max_bytes: int,
    on_progress: Callable[[float], None] = lambda f: None,
    cancelled: threading.Event | None = None,
) -> Path:
    """Downloads the video stream into dest_dir; returns the file path."""
    import yt_dlp

    def hook(d: dict) -> None:
        if cancelled is not None and cancelled.is_set():
            raise DownloadCancelled
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                on_progress(min(d.get("downloaded_bytes", 0) / total, 1.0))

    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        **_base_options(),
        "format": FORMAT,
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "max_filesize": max_bytes,
        "progress_hooks": [hook],
        "overwrites": True,
        "continuedl": False,
        "nopart": False,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
    except DownloadCancelled:
        raise
    except yt_dlp.utils.DownloadError as e:
        if isinstance(getattr(e, "exc_info", (None, None))[1], DownloadCancelled):
            raise DownloadCancelled from e
        raise _translate(e) from e
    files = [d.get("filepath") for d in (info or {}).get("requested_downloads") or []]
    path = Path(files[0]) if files and files[0] else None
    if path is None or not path.exists():
        # yt-dlp skips files over max_filesize without raising.
        raise YouTubeError("too_large", "The video is too large to download.", status=413)
    return path


def _translate(e: Exception) -> YouTubeError:
    """yt-dlp's errors are free text; map the common ones to clear messages."""
    text = str(e)
    low = text.lower()
    rules: list[tuple[tuple[str, ...], str, str, int]] = [
        # Before the bot check: YouTube's age gate also starts "Sign in to confirm".
        (
            ("confirm your age", "age-restricted", "inappropriate for some users"),
            "age_restricted",
            "Age-restricted videos can't be downloaded.",
            422,
        ),
        (
            ("not a bot", "confirm you're not", "confirm you’re not"),
            "youtube_blocked",
            "YouTube refused the download from this server. Try again later, or download the "
            "video yourself and upload the file.",
            502,
        ),
        (("private video",), "video_private", "This video is private.", 422),
        (
            ("members-only", "join this channel", "requires payment", "premium"),
            "video_restricted",
            "This video isn't public (members-only or paid).",
            422,
        ),
        (
            ("not available in your country", "blocked it in your country", "geo restrict"),
            "geo_blocked",
            "This video isn't available in this server's country.",
            422,
        ),
        (
            ("unavailable", "has been removed", "is not available", "does not exist"),
            "video_unavailable",
            "That video isn't available (removed, or the link is wrong).",
            422,
        ),
        (
            ("http error 429", "too many requests"),
            "rate_limited",
            "YouTube is limiting requests from this server. Wait a few minutes and try again.",
            503,
        ),
        (
            ("timed out", "timeout", "connection", "network", "getaddrinfo", "unreachable"),
            "network_error",
            "Couldn't reach YouTube. Check the server's internet connection and try again.",
            502,
        ),
        (
            ("requested format is not available", "no video formats"),
            "no_suitable_format",
            "YouTube doesn't offer this video in a format that can be analysed.",
            422,
        ),
    ]
    for needles, code, message, status in rules:
        if any(n in low for n in needles):
            return YouTubeError(code, message, status)
    logger.warning("Unrecognised yt-dlp error: %s", text[:500])
    return YouTubeError(
        "download_failed", "The video couldn't be fetched from YouTube. Please try again.", 502
    )


def _hm(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
