"""Writer for RGB, metric depth, instance IDs, and JSON ground truth."""

import json
from pathlib import Path
from typing import Union

import numpy as np

from .renderer import RenderedFrame


class DatasetWriter:
    def __init__(self, root: Union[str, Path], overwrite: bool = False):
        self.root = Path(root)
        self.overwrite = bool(overwrite)

    @staticmethod
    def _write_png(path: Path, image: np.ndarray, rgb: bool = False) -> None:
        try:
            import cv2
            output = image[:, :, ::-1] if rgb else image
            if not cv2.imwrite(str(path), output):
                raise IOError("OpenCV failed to write {}".format(path))
        except ImportError:
            if image.dtype != np.uint8:
                raise RuntimeError("OpenCV is required to write uint16 instance PNGs")
            from PIL import Image
            Image.fromarray(image, mode="RGB" if rgb else None).save(str(path))

    def write(self, frame: RenderedFrame) -> Path:
        frame_dir = self.root / "frames" / "{:06d}".format(frame.annotation.frame_id)
        if frame_dir.exists() and not self.overwrite:
            raise FileExistsError("frame already exists: {}".format(frame_dir))
        frame_dir.mkdir(parents=True, exist_ok=True)
        self._write_png(frame_dir / "rgb.png", frame.rgb, rgb=True)
        np.save(str(frame_dir / "depth.npy"), frame.depth_m, allow_pickle=False)
        self._write_png(frame_dir / "instance_id.png", frame.instance_id)
        with (frame_dir / "metadata.json").open("w", encoding="utf-8") as stream:
            json.dump(frame.annotation.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
        return frame_dir
