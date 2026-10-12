"""Local storage (dedupe, result cache, persistence), video serving, cancellation
and YouTube links. yt-dlp is replaced by fakes: no network in tests."""

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import youtube
from app.config import Settings
from app.jobs import Job, JobRequest, JobRunner, JobStore
from app.main import create_app
from app.storage import Storage

VIDEO = ("rally.mp4", b"\x00\x00\x00\x18ftypmp42 fake video", "video/mp4")
SHA = hashlib.sha256(VIDEO[1]).hexdigest()


def upload(client: TestClient, name: str = "rally.mp4", data: dict | None = None):
    return client.post(
        "/api/analyses", files={"video": (name, VIDEO[1], VIDEO[2])}, data=data or {}
    )


# ---- Storage and the result cache ----------------------------------------------


def test_same_video_twice_is_stored_once_and_answered_from_cache(
    client: TestClient, settings: Settings
) -> None:
    first = upload(client).json()
    second = upload(client, name="renamed copy.mp4").json()

    assert second["status"] == "ready" and second["cached"]  # no job ran
    assert second["progress"] == 1
    assert not client.get(f"/api/analyses/{first['id']}").json()["cached"]
    a = client.get(f"/api/analyses/{first['id']}/result").json()
    b = client.get(f"/api/analyses/{second['id']}/result").json()
    assert a == b
    assert [p.name for p in (settings.storage_dir / "videos").iterdir()] == [f"{SHA}.mp4"]


def test_different_settings_are_not_served_from_cache(client: TestClient) -> None:
    upload(client, data={"duration_sec": "12"})
    other = upload(client, data={"duration_sec": "15"}).json()
    assert not other["cached"]


def test_jobs_and_results_survive_a_restart(settings: Settings) -> None:
    with TestClient(create_app(settings)) as c:
        job_id = upload(c).json()["id"]
        before = c.get(f"/api/analyses/{job_id}/result").json()
    with TestClient(create_app(settings)) as c:
        assert c.get(f"/api/analyses/{job_id}").json()["status"] == "ready"
        assert c.get(f"/api/analyses/{job_id}/result").json() == before


def test_jobs_interrupted_by_a_restart_are_marked_failed(settings: Settings) -> None:
    storage = Storage(settings.storage_dir)
    JobStore(storage).add(
        Job(id="j1", kind="upload", request=JobRequest(), stages=["x"], status="processing")
    )
    storage.close()
    with TestClient(create_app(settings)) as c:
        status = c.get("/api/analyses/j1").json()
    assert status["status"] == "failed"
    assert status["errorCode"] == "interrupted"


def test_leftover_temp_files_are_cleaned_on_start(settings: Settings) -> None:
    tmp = settings.storage_dir / "tmp"
    tmp.mkdir(parents=True)
    (tmp / "half-uploaded").write_bytes(b"x")
    Storage(settings.storage_dir).close()
    assert not any(tmp.iterdir())


def test_config_reports_limits(client: TestClient, settings: Settings) -> None:
    cfg = client.get("/api/config").json()
    assert cfg["maxUploadBytes"] == settings.max_upload_bytes
    assert cfg["maxVideoSeconds"] == 660
    assert "enabled" in cfg["youtube"]


# ---- Video files ---------------------------------------------------------------


def test_stored_video_is_served_with_range_support(client: TestClient) -> None:
    status = upload(client).json()
    url = client.get(f"/api/analyses/{status['id']}").json()["video"]["url"]
    assert url == f"/api/videos/{SHA}/file"
    full = client.get(url)
    assert full.status_code == 200 and full.content == VIDEO[1]
    assert full.headers["content-type"] == "video/mp4"
    part = client.get(url, headers={"Range": "bytes=4-7"})
    assert part.status_code == 206 and part.content == VIDEO[1][4:8]


@pytest.mark.parametrize("bad", ["nope", "../../etc/passwd", "a" * 64])
def test_unknown_or_malformed_video_ids_are_404(client: TestClient, bad: str) -> None:
    assert client.get(f"/api/videos/{bad}/file").status_code == 404


# ---- Cancellation --------------------------------------------------------------


def test_cancelling_stops_a_running_analysis(settings: Settings) -> None:
    from app.analysis.mock import MockAnalysisProvider

    storage = Storage(settings.storage_dir)
    tmp = storage.new_tmp_path()
    tmp.write_bytes(VIDEO[1])
    video = storage.add_video(tmp, SHA, ".mp4", name="rally.mp4", source="upload")
    jobs = JobStore(storage)
    job = Job(id="j1", kind="upload", request=JobRequest(), stages=["a"], video_id=video.id)
    jobs.add(job)
    assert jobs.request_cancel("j1")
    JobRunner(jobs, storage, MockAnalysisProvider(0), settings).run("j1")
    status = jobs.get("j1")
    assert status.status == "failed" and status.error_code == "cancelled"
    storage.close()


def test_cancel_endpoint(client: TestClient) -> None:
    job_id = upload(client).json()["id"]  # already finished: cancel is a no-op
    assert client.post(f"/api/analyses/{job_id}/cancel").json()["status"] == "ready"
    assert client.post("/api/analyses/nope/cancel").status_code == 404


JOB_ID = "0123456789abcdef0123456789abcdef"


def test_clients_may_choose_the_job_id(client: TestClient) -> None:
    res = upload(client, data={"job_id": JOB_ID})
    assert res.status_code == 202 and res.json()["id"] == JOB_ID
    assert client.get(f"/api/analyses/{JOB_ID}").json()["status"] == "ready"
    assert upload(client, data={"job_id": JOB_ID}).status_code == 409  # taken
    assert upload(client, data={"job_id": "not-an-id"}).status_code == 422


def test_a_cancel_sent_before_the_upload_finished_is_honoured(client: TestClient) -> None:
    """The client gave up while its upload was still being received: the
    cancel arrives first, and the job it then creates must not run."""
    assert client.post(f"/api/analyses/{JOB_ID}/cancel").status_code == 404  # not there yet
    res = upload(client, data={"job_id": JOB_ID, "duration_sec": "9"})
    assert res.status_code == 202
    status = client.get(f"/api/analyses/{JOB_ID}").json()
    assert status["status"] == "failed" and status["errorCode"] == "cancelled"
    assert client.get(f"/api/analyses/{JOB_ID}/result").status_code == 409  # nothing ran


def test_cancelling_while_a_youtube_link_is_checked_stops_the_job(
    client: TestClient, fake_youtube, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checking a link takes seconds. A user who cancels meanwhile used to leave
    a job behind that downloaded and analysed the video for nobody."""
    calls, _ = fake_youtube
    checking = youtube.fetch_info

    def fetch_info_then_cancel(video_id: str, max_source_seconds: float):
        info = checking(video_id, max_source_seconds)
        client.app.state.jobs.cancel(JOB_ID)  # the user's cancel, mid-request
        return info

    monkeypatch.setattr(youtube, "fetch_info", fetch_info_then_cancel)
    res = client.post(
        "/api/analyses/youtube", json={"url": "https://youtu.be/E3V9GWOr-_o", "jobId": JOB_ID}
    )
    assert res.status_code == 202
    assert res.json()["status"] == "failed" and res.json()["errorCode"] == "cancelled"
    assert calls["download"] == 0


def test_early_cancels_are_only_kept_for_well_formed_ids(settings: Settings) -> None:
    storage = Storage(settings.storage_dir)
    jobs = JobStore(storage)
    assert jobs.cancel("../not-a-job") is None
    assert jobs._early_cancels == {}
    assert jobs.cancel(JOB_ID) is None
    assert list(jobs._early_cancels) == [JOB_ID]
    storage.close()


# ---- Streamed uploads ----------------------------------------------------------


def test_uploads_are_streamed_into_storage_not_spooled(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Starlette's form parser would copy the whole body to the OS temp folder
    first (outside storage/, and before the size limit could be checked)."""
    import starlette.formparsers

    def no_spooling(*args, **kwargs):
        raise AssertionError("the upload was spooled by Starlette")

    monkeypatch.setattr(starlette.formparsers, "SpooledTemporaryFile", no_spooling)
    assert upload(client).status_code == 202


def test_an_upload_declared_too_large_is_refused_before_it_is_read() -> None:
    import asyncio

    from fastapi import HTTPException
    from starlette.requests import Request

    from app.api.uploads import receive_upload

    async def receive():
        raise AssertionError("the body was read")

    headers = [
        (b"content-type", b"multipart/form-data; boundary=x"),
        (b"content-length", str(10 * 1024**3).encode()),
    ]
    request = Request({"type": "http", "method": "POST", "headers": headers}, receive)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(receive_upload(request, Path("unused"), 1024**3, "video", lambda *a: None))
    assert refused.value.status_code == 413


def test_fields_sent_before_the_video_are_checked_before_storing_it(
    client: TestClient, settings: Settings
) -> None:
    too_long = str(settings.max_video_seconds + 60)
    assert upload(client, data={"duration_sec": too_long}).status_code == 422
    assert upload(client, name="notes.pdf").status_code == 415
    assert upload(client, data={"duration_sec": "abc"}).status_code == 422
    assert not any((settings.storage_dir / "tmp").iterdir())
    assert not any((settings.storage_dir / "videos").iterdir())


# ---- YouTube links -------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=E3V9GWOr-_o", "E3V9GWOr-_o"),
        ("https://youtu.be/E3V9GWOr-_o?t=42", "E3V9GWOr-_o"),
        ("youtube.com/watch?v=E3V9GWOr-_o&list=PL123&index=2", "E3V9GWOr-_o"),
        ("https://m.youtube.com/watch?v=E3V9GWOr-_o", "E3V9GWOr-_o"),
        ("https://www.youtube.com/shorts/E3V9GWOr-_o", "E3V9GWOr-_o"),
        ("https://www.youtube.com/embed/E3V9GWOr-_o", "E3V9GWOr-_o"),
        ("  https://www.youtube.com/live/E3V9GWOr-_o  ", "E3V9GWOr-_o"),
    ],
)
def test_youtube_links_are_parsed(url: str, expected: str) -> None:
    assert youtube.video_id_from_url(url) == expected


@pytest.mark.parametrize(
    ("url", "code"),
    [
        ("", "invalid_url"),
        ("https://vimeo.com/123456", "unsupported_site"),
        ("https://www.youtube.com.evil.example/watch?v=E3V9GWOr-_o", "unsupported_site"),
        ("https://www.youtube.com/playlist?list=PL123", "invalid_url"),
        ("https://www.youtube.com/@BWF", "invalid_url"),
        ("https://www.youtube.com/watch?v=short", "invalid_url"),
        ("file:///etc/passwd", "unsupported_site"),
    ],
)
def test_bad_links_are_rejected(url: str, code: str) -> None:
    with pytest.raises(youtube.YouTubeError) as e:
        youtube.video_id_from_url(url)
    assert e.value.code == code


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("ERROR: [youtube] x: Sign in to confirm you're not a bot", "youtube_blocked"),
        (
            "ERROR: [youtube] x: Private video. Sign in if you've been granted access",
            "video_private",
        ),
        ("ERROR: [youtube] x: Video unavailable", "video_unavailable"),
        ("ERROR: [youtube] aaaaaaaaaaa: This video is unavailable", "video_unavailable"),
        ("ERROR: Sign in to confirm your age. This video may be inappropriate", "age_restricted"),
        ("ERROR: [youtube] x: Sign in to confirm you’re not a bot.", "youtube_blocked"),
        ("ERROR: HTTP Error 429: Too Many Requests", "rate_limited"),
        ("ERROR: Unable to download webpage: timed out", "network_error"),
        ("ERROR: something new and strange", "download_failed"),
    ],
)
def test_ytdlp_errors_get_clear_codes(message: str, code: str) -> None:
    assert youtube._translate(Exception(message)).code == code


INFO = youtube.VideoInfo(
    id="E3V9GWOr-_o",
    title="Play of the Day",
    channel="BWF TV",
    duration_sec=128.0,
    thumbnail_url="https://i.ytimg.com/vi/E3V9GWOr-_o/hqdefault.jpg",
    height=1080,
    filesize=24_000,
)


@pytest.fixture
def fake_youtube(monkeypatch: pytest.MonkeyPatch):
    """yt-dlp replaced: info from INFO, downloads write VIDEO's bytes."""
    calls = {"info": 0, "download": 0}
    state = {"info": INFO}

    def fetch_info(video_id: str, max_source_seconds: float) -> youtube.VideoInfo:
        calls["info"] += 1
        return replace(state["info"], id=video_id)

    def download(video_id, dest_dir: Path, max_bytes, on_progress=None, cancelled=None):
        calls["download"] += 1
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / f"{video_id}.mp4"
        path.write_bytes(VIDEO[1] + video_id.encode())
        if on_progress:
            on_progress(1.0)
        return path

    monkeypatch.setattr(youtube, "available", lambda: True)
    monkeypatch.setattr(youtube, "fetch_info", fetch_info)
    monkeypatch.setattr(youtube, "download", download)
    return calls, state


def test_youtube_info(client: TestClient, fake_youtube) -> None:
    res = client.get("/api/youtube/info", params={"url": "https://youtu.be/E3V9GWOr-_o"})
    assert res.status_code == 200
    body = res.json()
    assert body["title"] == "Play of the Day" and body["durationSec"] == 128
    assert body["url"] == "https://www.youtube.com/watch?v=E3V9GWOr-_o"
    assert body["needsRange"] is False and body["maxVideoSeconds"] == 660


def test_youtube_info_errors_carry_codes(client: TestClient, fake_youtube, monkeypatch) -> None:
    bad = client.get("/api/youtube/info", params={"url": "https://vimeo.com/1"})
    assert bad.status_code == 422 and bad.json()["code"] == "unsupported_site"

    def blocked(*a, **k):
        raise youtube.YouTubeError("youtube_blocked", "YouTube refused.", 502)

    monkeypatch.setattr(youtube, "fetch_info", blocked)
    res = client.get("/api/youtube/info", params={"url": "https://youtu.be/E3V9GWOr-_o"})
    assert res.status_code == 502 and res.json()["code"] == "youtube_blocked"


def test_youtube_disabled(settings: Settings) -> None:
    with TestClient(create_app(replace(settings, youtube_enabled=False))) as c:
        res = c.post("/api/analyses/youtube", json={"url": "https://youtu.be/E3V9GWOr-_o"})
        assert res.status_code == 503 and res.json()["code"] == "youtube_unavailable"


def test_youtube_video_is_downloaded_analysed_and_reused(
    client: TestClient, fake_youtube, settings: Settings
) -> None:
    calls, _ = fake_youtube
    res = client.post("/api/analyses/youtube", json={"url": "https://youtu.be/E3V9GWOr-_o"})
    assert res.status_code == 202
    created = res.json()
    assert created["stages"][0] == "Downloading video"

    status = client.get(f"/api/analyses/{created['id']}").json()
    assert status["status"] == "ready"
    assert status["video"]["source"] == "youtube"
    assert status["video"]["name"] == "Play of the Day"
    assert status["video"]["youtubeUrl"] == "https://www.youtube.com/watch?v=E3V9GWOr-_o"
    assert client.get(status["video"]["url"]).status_code == 200
    assert client.get(f"/api/analyses/{created['id']}/result").status_code == 200
    assert calls["download"] == 1

    # The same link again: no download, no analysis.
    again = client.post("/api/analyses/youtube", json={"url": "youtube.com/watch?v=E3V9GWOr-_o"})
    assert again.json()["status"] == "ready" and again.json()["cached"]
    assert calls["download"] == 1
    assert not any((settings.storage_dir / "tmp").iterdir())


def test_long_youtube_videos_need_a_part_chosen(client: TestClient, fake_youtube) -> None:
    _, state = fake_youtube
    state["info"] = replace(INFO, duration_sec=40 * 60)
    url = "https://youtu.be/E3V9GWOr-_o"

    res = client.post("/api/analyses/youtube", json={"url": url})
    assert res.status_code == 422 and res.json()["code"] == "range_required"
    res = client.post("/api/analyses/youtube", json={"url": url, "startSec": 60, "endSec": 900})
    assert res.status_code == 422 and res.json()["code"] == "range_too_long"
    res = client.post("/api/analyses/youtube", json={"url": url, "startSec": 90, "endSec": 30})
    assert res.status_code == 422 and res.json()["code"] == "invalid_range"
    res = client.post("/api/analyses/youtube", json={"url": url, "startSec": 2000, "endSec": 2500})
    assert res.status_code == 422 and res.json()["code"] == "invalid_range"  # past the end

    ok = client.post("/api/analyses/youtube", json={"url": url, "startSec": 600, "endSec": 1200})
    assert ok.status_code == 202
    status = client.get(f"/api/analyses/{ok.json()['id']}").json()
    assert status["window"] == {"startSec": 600, "endSec": 1200}
    assert status["status"] == "ready"


def test_youtube_download_failure_fails_the_job(client: TestClient, fake_youtube, monkeypatch):
    def fail(*a, **k):
        raise youtube.YouTubeError("network_error", "Couldn't reach YouTube.", 502)

    monkeypatch.setattr(youtube, "download", fail)
    job_id = client.post(
        "/api/analyses/youtube", json={"url": "https://youtu.be/E3V9GWOr-_o"}
    ).json()["id"]
    status = client.get(f"/api/analyses/{job_id}").json()
    assert status["status"] == "failed" and status["errorCode"] == "network_error"
    assert status["video"] is None


def test_storage_layout_is_self_contained(client: TestClient, settings: Settings) -> None:
    upload(client)
    root = settings.storage_dir
    assert (root / "rallyreview.sqlite3").exists()
    assert {p.name for p in root.iterdir()} >= {"videos", "cache", "tmp", "rallyreview.sqlite3"}
    rows = json.loads(json.dumps([p.name for p in (root / "videos").iterdir()]))
    assert rows == [f"{SHA}.mp4"]
    shutil.rmtree(root / "cache", ignore_errors=True)  # cache is disposable
