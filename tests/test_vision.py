import numpy as np
import pytest

from nerva.vision import VisionPerception, find_blobs, masks, pixel_ray, rgb_to_hsv


def _scene():
    """Synthetic 120x160 frame: grey floor, an orange ball, a blue 'person', a thin dark grid line."""
    rgb = np.full((120, 160, 3), 120, np.uint8)
    rgb[60, :] = 30  # grid line: dark but grey, must not be detected
    rgb[70:80, 30:40] = (242, 115, 25)  # ball
    rgb[20:100, 100:120] = (64, 102, 166)  # clothing
    depth = np.full((120, 160), 3.0, np.float32)
    depth[70:80, 30:40] = 1.0
    depth[20:100, 100:120] = 2.0
    return rgb, depth


def test_hsv_of_primary_colours():
    hsv = rgb_to_hsv(np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8))
    np.testing.assert_allclose(hsv[0, :, 0], [0, 120, 240])


def test_colour_masks_and_blobs():
    rgb, depth = _scene()
    m = masks(rgb)
    assert m["ball"][75, 35] and m["person"][50, 110] and not m["person"][60, 5]
    blobs = {b.kind: b for b in find_blobs(rgb, depth)}
    assert set(blobs) == {"ball", "person"}
    assert blobs["ball"].depth_m == pytest.approx(1.0) and blobs["person"].top_row == 20
    assert blobs["person"].appearance.sum() == pytest.approx(1.0)


def test_detection_geometry_from_depth():
    rgb, depth = _scene()
    xmat = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)  # looking along world +x
    st = VisionPerception().detect_frame(0.0, rgb, depth, np.zeros(3), xmat, 70.0, 0.0, 0.1)
    tr = {t.kind: t for t in st.tracks}
    assert tr["ball"].bearing > 0 > tr["person"].bearing  # ball left of centre, person right
    assert tr["person"].distance == pytest.approx(2.0, rel=0.1)
    assert {e.kind for e in st.events} == {"ball_appeared", "person_appeared"}


def test_pixel_ray_centre_looks_forward():
    ray = pixel_ray(59.5, 79.5, 120, 160, 70.0)
    np.testing.assert_allclose(ray, [0, 0, -1], atol=1e-9)
