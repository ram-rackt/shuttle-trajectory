import numpy as np
import pytest

from rallycv import court_model as cm
from rallycv.camera import camera_from_homography
from rallycv.court import CourtNotFoundError, calibration_from_corners, detect_court
from rallycv.geometry import apply_homography
from rallycv.synthetic import broadcast_camera, court_homography, render_court

CAMERAS = {
    "centred": broadcast_camera(),
    "offset_left": broadcast_camera(position=(1.5, -15.0, 6.5), look_at=(3.3, 7.0, 0.0)),
    "long_lens": broadcast_camera(focal=3400, position=(3.05, -20.0, 7.5)),
}


def corner_error(H_est: np.ndarray, H_true: np.ndarray) -> float:
    pts = np.vstack([cm.CORNERS, [[cm.WIDTH / 2, cm.NET_Y]]])
    return float(np.abs(apply_homography(H_est, pts) - apply_homography(H_true, pts)).max())


@pytest.mark.parametrize("name", CAMERAS)
def test_detects_court_on_synthetic_broadcast_frame(name: str) -> None:
    cam = CAMERAS[name]
    img = render_court(cam)
    cal = detect_court(img)
    assert cal.confidence > 0.6
    assert corner_error(cal.court_to_image, court_homography(cam)) < 4.0  # pixels at 1080p


def test_recovers_camera_from_detected_court() -> None:
    cam = CAMERAS["centred"]
    cal = detect_court(render_court(cam))
    est = camera_from_homography(cal.court_to_image, cal.image_size)
    assert est.K[0, 0] == pytest.approx(cam.K[0, 0], rel=0.05)
    assert np.linalg.norm(est.center - cam.center) < 0.6  # metres
    # Heights come out right: the top of a net post (1.55 m).
    post = np.array([[0.0, cm.NET_Y, cm.NET_HEIGHT_POSTS]])
    assert np.abs(est.project(post) - cam.project(post)).max() < 6.0


def test_rejects_frame_without_court() -> None:
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    with pytest.raises(CourtNotFoundError):
        detect_court(noise)


def test_manual_corners_are_refined_against_painted_lines() -> None:
    cam = CAMERAS["centred"]
    img = render_court(cam)
    H_true = court_homography(cam)
    rough = apply_homography(H_true, cm.CORNERS) + np.array(
        [[6, -5], [-7, 4], [5, 6], [-6, -4]]
    )  # a user's clicks are a few pixels off
    cal = calibration_from_corners(rough, (1920, 1080), img)
    assert cal.manual
    assert corner_error(cal.court_to_image, H_true) < 4.0


def test_manual_corners_must_be_a_convex_quad() -> None:
    bowtie = np.array([[100, 900], [1800, 100], [1800, 900], [100, 100]], dtype=float)
    with pytest.raises(ValueError):
        calibration_from_corners(bowtie, (1920, 1080))
