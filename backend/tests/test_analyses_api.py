import hashlib
from collections.abc import Sequence

from fastapi.testclient import TestClient

from app.analysis import AnalysisError, AnalysisRequest, ProgressCallback, RallyAnalysis
from app.config import Settings
from app.main import create_app

VIDEO = ("rally.mp4", b"\x00\x00\x00\x18ftypmp42 fake video", "video/mp4")


def upload(client: TestClient, files=None, data=None):
    return client.post("/api/analyses", files={"video": files or VIDEO}, data=data or {})


def test_upload_runs_mock_analysis_to_ready(client: TestClient, settings: Settings) -> None:
    res = upload(client, data={"duration_sec": "12.5"})
    assert res.status_code == 202
    created = res.json()
    assert created["status"] == "queued"
    assert created["stages"][0] == "Detecting court"

    # With zero stage delay the background task has finished by now.
    status = client.get(f"/api/analyses/{created['id']}").json()
    assert status["status"] == "ready"
    assert status["progress"] == 1
    assert status["currentStage"] is None

    result = client.get(f"/api/analyses/{created['id']}/result").json()
    assert result["schemaVersion"] == 1
    assert result["durationSec"] == 12.5
    assert result["shots"][-1]["endTime"] <= 12.5

    # Stored once, named by its content.
    stored = list((settings.storage_dir / "videos").iterdir())
    assert [p.name for p in stored] == [f"{hashlib.sha256(VIDEO[1]).hexdigest()}.mp4"]
    assert status["video"]["source"] == "upload"
    assert status["video"]["name"] == "rally.mp4"
    assert not status["cached"]


def test_upload_without_duration_uses_default(client: TestClient) -> None:
    job_id = upload(client).json()["id"]
    assert client.get(f"/api/analyses/{job_id}/result").json()["durationSec"] == 20.0


def test_rejects_unsupported_extension(client: TestClient) -> None:
    res = upload(client, files=("notes.pdf", b"%PDF", "application/pdf"))
    assert res.status_code == 415


def test_rejects_empty_upload(client: TestClient, settings: Settings) -> None:
    res = upload(client, files=("rally.mp4", b"", "video/mp4"))
    assert res.status_code == 400
    assert not any((settings.storage_dir / "videos").iterdir())
    assert not any((settings.storage_dir / "tmp").iterdir())


def test_rejects_oversized_upload(client: TestClient, settings: Settings) -> None:
    big = b"x" * (settings.max_upload_bytes + 1)
    res = upload(client, files=("rally.mp4", big, "video/mp4"))
    assert res.status_code == 413
    assert not any((settings.storage_dir / "videos").iterdir())
    assert not any((settings.storage_dir / "tmp").iterdir())


def test_rejects_invalid_duration(client: TestClient) -> None:
    assert upload(client, data={"duration_sec": "-1"}).status_code == 422


def test_rejects_videos_over_the_length_limit(client: TestClient, settings: Settings) -> None:
    too_long = str(settings.max_video_seconds + 60)
    res = upload(client, data={"duration_sec": too_long})
    assert res.status_code == 422
    assert "limit is 11 minutes" in res.json()["detail"]


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/api/analyses/nope").status_code == 404
    assert client.get("/api/analyses/nope/result").status_code == 404


class FailingProvider:
    @property
    def stages(self) -> Sequence[str]:
        return ["Exploding"]

    def analyze(self, request: AnalysisRequest, on_progress: ProgressCallback) -> RallyAnalysis:
        on_progress(0, 0.5)
        raise RuntimeError("boom")


def test_provider_failure_marks_job_failed(settings: Settings) -> None:
    with TestClient(create_app(settings, provider=FailingProvider())) as client:
        job_id = upload(client).json()["id"]
        status = client.get(f"/api/analyses/{job_id}").json()
        assert status["status"] == "failed"
        assert status["errorCode"] == "internal_error"
        assert "unexpectedly" in status["error"]
        assert client.get(f"/api/analyses/{job_id}/result").status_code == 409


class CourtNotFoundProvider:
    """Fails like the CV pipeline does when it can't find the court, unless the
    request carries manual corners."""

    def __init__(self) -> None:
        self.requests: list[AnalysisRequest] = []

    @property
    def stages(self) -> Sequence[str]:
        return ["Detecting court"]

    def analyze(self, request: AnalysisRequest, on_progress: ProgressCallback) -> RallyAnalysis:
        self.requests.append(request)
        if request.court_corners is None:
            raise AnalysisError("court_not_found", "Mark the court corners.")
        from app.analysis.mock import generate_mock_rally

        return generate_mock_rally(5.0, seed=1)


CORNERS = [{"x": 0.2, "y": 0.9}, {"x": 0.8, "y": 0.9}, {"x": 0.7, "y": 0.5}, {"x": 0.3, "y": 0.5}]


def test_failure_reports_error_code(settings: Settings) -> None:
    with TestClient(create_app(settings, provider=CourtNotFoundProvider())) as client:
        job_id = upload(client).json()["id"]
        status = client.get(f"/api/analyses/{job_id}").json()
        assert status["status"] == "failed"
        assert status["errorCode"] == "court_not_found"


def test_recalibration_reruns_same_video_with_corners(settings: Settings) -> None:
    provider = CourtNotFoundProvider()
    with TestClient(create_app(settings, provider=provider)) as client:
        job_id = upload(client).json()["id"]
        res = client.post(f"/api/analyses/{job_id}/calibration", json={"corners": CORNERS})
        assert res.status_code == 202
        new_id = res.json()["id"]
        assert new_id != job_id
        assert client.get(f"/api/analyses/{new_id}").json()["status"] == "ready"
        first, second = provider.requests
        assert second.video_path == first.video_path
        assert second.court_corners == ((0.2, 0.9), (0.8, 0.9), (0.7, 0.5), (0.3, 0.5))


class WarmingProvider(CourtNotFoundProvider):
    """Also offers warm_up, recording what it was asked to precompute."""

    def __init__(self) -> None:
        super().__init__()
        self.warmed: list[tuple[AnalysisRequest, object]] = []

    def warm_up(self, request: AnalysisRequest, should_stop) -> None:  # noqa: ANN001
        self.warmed.append((request, should_stop))


def test_court_not_found_starts_a_warm_up_for_the_re_run(settings: Settings) -> None:
    """While the user marks the corners, the shuttle detections are computed so
    the re-run with the corners is quicker. Another video's job stops it."""
    provider = WarmingProvider()
    with TestClient(create_app(settings, provider=provider)) as client:
        job_id = upload(client).json()["id"]
        assert client.get(f"/api/analyses/{job_id}").json()["errorCode"] == "court_not_found"
        ((request, should_stop),) = provider.warmed
        assert request.video_path == provider.requests[0].video_path
        assert request.cache_dir == provider.requests[0].cache_dir is not None
        assert not should_stop()

        runner = client.app.state.runner
        runner._active["other-job"] = "a-different-video"
        assert should_stop()
        runner._active["other-job"] = request.video_path.stem  # the re-run, same video
        assert not should_stop()
        del runner._active["other-job"]

        # A successful analysis doesn't warm anything up.
        client.post(f"/api/analyses/{job_id}/calibration", json={"corners": CORNERS})
        assert len(provider.warmed) == 1


def test_recalibration_validates_corners(client: TestClient) -> None:
    job_id = upload(client).json()["id"]
    bad = client.post(f"/api/analyses/{job_id}/calibration", json={"corners": CORNERS[:3]})
    assert bad.status_code == 422
    out_of_range = [{"x": 1.5, "y": 0.5}] + CORNERS[1:]
    res = client.post(f"/api/analyses/{job_id}/calibration", json={"corners": out_of_range})
    assert res.status_code == 422
    assert (
        client.post("/api/analyses/nope/calibration", json={"corners": CORNERS}).status_code == 404
    )


def test_upload_accepts_court_corners(settings: Settings) -> None:
    import json

    provider = CourtNotFoundProvider()
    with TestClient(create_app(settings, provider=provider)) as client:
        res = upload(client, data={"court_corners": json.dumps(CORNERS)})
        assert res.status_code == 202
        assert client.get(f"/api/analyses/{res.json()['id']}").json()["status"] == "ready"
