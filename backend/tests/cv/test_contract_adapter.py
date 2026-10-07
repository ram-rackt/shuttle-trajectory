import numpy as np

from app.analysis.cv_provider import to_contract
from rallycv import court_model as cm
from rallycv.court import CourtCalibration
from rallycv.geometry import apply_homography
from rallycv.pipeline import RallyResult, ShotResult
from rallycv.synthetic import broadcast_camera, court_homography

CAM = broadcast_camera()


def make_result() -> RallyResult:
    nan = np.nan
    return RallyResult(
        fps=30.0,
        duration=4.0,
        image_size=(1920, 1080),
        calibration=CourtCalibration(court_homography(CAM), (1920, 1080), 0.8),
        camera=CAM,
        shots=[ShotResult(0, "A", "serve", 1.0, 2.0), ShotResult(1, "B", "clear", 2.0, 3.5)],
        shuttle_t=np.array([1.0, 1.5, 2.0]),
        shuttle_ground=np.array([[3.05, 2.0], [3.05, 6.7], [nan, nan]]),
        shuttle_image=np.array([[960.0, 900.0], [nan, nan], [960.0, 400.0]]),
        shuttle_height=np.array([2.0, nan, 3.0]),
        players={"A": (np.array([0.0, 1.0]), np.array([[3.05, 2.0], [3.0, 2.1]]))},
    )


def test_converts_metres_and_pixels_to_normalized_coordinates() -> None:
    out = to_contract(make_result())
    assert out.source == "cv"
    s0, s1, s2 = out.shuttle
    assert (s0.position.x, s0.position.y) == (0.5, round(2.0 / cm.LENGTH, 4))
    assert (s0.image.x, s0.image.y) == (0.5, round(900 / 1080, 4))
    assert s0.height == 2.0
    assert s1.image is None and s1.height is None
    assert s2.position is None
    assert [s.type for s in out.shots] == ["serve", "clear"]
    assert out.players[0].id == "A" and len(out.players[0].samples) == 2


def test_calibration_maps_normalized_court_to_normalized_image() -> None:
    out = to_contract(make_result())
    H = np.array(out.calibration.court_to_image).reshape(3, 3)
    corners_norm = cm.CORNERS / np.array([cm.WIDTH, cm.LENGTH])
    expected = apply_homography(court_homography(CAM), cm.CORNERS) / np.array([1920, 1080])
    assert np.allclose(apply_homography(H, corners_norm), expected, atol=1e-5)
    assert out.calibration.confidence == 0.8
