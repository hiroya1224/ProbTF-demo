import json

import numpy as np

from prob_artag_renderer.annotations import annotate_tags, update_visibility
from prob_artag_renderer.camera_model import CameraModel
from prob_artag_renderer.coordinates import make_transform
from prob_artag_renderer.dataset_writer import DatasetWriter
from prob_artag_renderer.renderer import RenderedFrame
from prob_artag_renderer.scene_sampler import TagSpec


def test_dataset_layout_and_metadata(tmp_path):
    camera = CameraModel(width=80, height=60, fx=70, fy=70, cx=40, cy=30)
    tag = TagSpec("DICT_APRILTAG_36h11", 3, 1, 0.2,
                  make_transform(np.diag([1.0, -1.0, -1.0]), [0, 0, 1]))
    annotation = annotate_tags(7, "frontal", 9, camera, np.eye(4), [tag], {})
    instances = np.zeros((60, 80), dtype=np.uint16)
    corners = np.rint(annotation.tags[0].corners_px).astype(int)
    import cv2
    cv2.fillConvexPoly(instances, corners, 1)
    annotation = update_visibility(annotation, instances)
    rendered = RenderedFrame(
        np.full((60, 80, 3), 127, dtype=np.uint8),
        np.full((60, 80), 1.0, dtype=np.float32), instances, annotation,
    )
    path = DatasetWriter(tmp_path).write(rendered)
    assert sorted(item.name for item in path.iterdir()) == [
        "depth.npy", "instance_id.png", "metadata.json", "rgb.png"
    ]
    np.testing.assert_array_equal(np.load(str(path / "depth.npy")), rendered.depth_m)
    with (path / "metadata.json").open() as stream:
        metadata = json.load(stream)
    assert metadata["frame_id"] == 7
    assert metadata["camera"]["T_W_C"] == np.eye(4).tolist()
    assert metadata["tags"][0]["id"] == 3
    assert metadata["tags"][0]["corner_order"] == "IPPE_SQUARE_TL_TR_BR_BL"
    assert metadata["tags"][0]["visible_fraction"] > 0.9
