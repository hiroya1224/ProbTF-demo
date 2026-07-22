import numpy as np

from prob_artag_renderer.tag_texture import generate_tag_texture


def _detect(gray):
    import cv2
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    if hasattr(aruco, "ArucoDetector"):
        return aruco.ArucoDetector(dictionary).detectMarkers(gray)
    return aruco.detectMarkers(gray, dictionary)


def test_generated_texture_has_white_margin_and_detectable_ordered_corners():
    texture = generate_tag_texture(17, marker_pixels=256, margin_pixels=32)
    assert texture.rgb.shape == (320, 320, 3)
    assert np.all(texture.rgb[:32] == 255)
    corners, ids, _ = _detect(texture.rgb[:, :, 0])
    assert ids is not None and ids.reshape(-1).tolist() == [17]
    expected = np.array([[32, 32], [287, 32], [287, 287], [32, 287]], dtype=float)
    np.testing.assert_allclose(corners[0].reshape(4, 2), expected, atol=2.0)


def test_texture_is_deterministic():
    first = generate_tag_texture(4, marker_pixels=128, margin_pixels=16)
    second = generate_tag_texture(4, marker_pixels=128, margin_pixels=16)
    assert np.array_equal(first.rgb, second.rgb)
