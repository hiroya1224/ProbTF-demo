"""The unchanged physical ray/plane pixel test without NumPy temporaries."""

import ctypes as ct
from functools import lru_cache
import os
from pathlib import Path

import numpy as np


_DOUBLE = ct.POINTER(ct.c_double)
_BYTE = ct.POINTER(ct.c_uint8)
_NAME = 'libprobtf_physical_camera.so'


@lru_cache(maxsize=1)
def _library():
    explicit = os.environ.get('PROBTF_PHYSICAL_CAMERA_LIBRARY')
    candidates = [Path(explicit).expanduser()] if explicit else []
    if not explicit:
        for variable in ('LD_LIBRARY_PATH', 'CMAKE_PREFIX_PATH'):
            for item in os.environ.get(variable, '').split(os.pathsep):
                if item:
                    candidates.append((Path(item) if variable == 'LD_LIBRARY_PATH'
                                       else Path(item)/'lib')/_NAME)
        for ancestor in Path(__file__).resolve().parents:
            if (ancestor/'devel').is_dir() and (ancestor/'src').is_dir():
                candidates.extend((ancestor/'devel/lib'/_NAME,
                    ancestor/'devel/.private/probtf_observation_fusion_demo/lib'/_NAME))
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise RuntimeError('Native physical camera arithmetic is not built. Run catkin build '
                           'probtf_observation_fusion_demo and source the workspace.')
    library = ct.CDLL(str(path.resolve()))
    library.probtf_backend_path = str(path.resolve())
    library.probtf_physical_camera_abi_version.restype = ct.c_int
    if library.probtf_physical_camera_abi_version() != 1:
        raise RuntimeError('Native physical camera C ABI is incompatible.')
    library.probtf_physical_camera_occlusion_mask.argtypes = (
        [ct.c_int]*6+[_DOUBLE]*5+[ct.c_double]*6+
        [_BYTE, _BYTE, ct.c_char_p, ct.c_size_t])
    library.probtf_physical_camera_occlusion_mask.restype = ct.c_int
    return library


def occlusion_mask(rays, rotation, origin, center, lower, upper, near, board, bounds):
    """Same inclusive rectangle/depth tests on the same cached pinhole rays.

    Board rasterization, near-plane clipping and the image's exposure/noise
    law stay in the existing renderer. No runtime compilation or fallback.
    """
    board = np.ascontiguousarray(board, dtype=np.uint8)
    if board.ndim != 2 or len(rays) != 3:
        raise ValueError('Physical mask needs a board image and three pinhole ray arrays.')
    height, width = board.shape
    arrays = [np.ascontiguousarray(value, dtype=np.float64)
              for value in (*rays, rotation, origin)]
    if (any(value.shape != board.shape for value in arrays[:3])
            or arrays[3].shape != (3, 3) or arrays[4].shape != (3,)):
        raise ValueError('Physical mask geometry and pixel ray dimensions disagree.')
    parameters = np.asarray([center[2], lower[0], lower[1], upper[0], upper[1], near], float)
    if (not np.all(np.isfinite(parameters))
            or not np.all(np.isfinite(arrays[3])) or not np.all(np.isfinite(arrays[4]))):
        raise ValueError('Physical ray/plane parameters must be finite.')
    x0, y0, x1, y1 = map(int, bounds)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Physical mask rectangle must lie inside the image.')
    output = np.empty((y1-y0, x1-x0), dtype=np.bool_)
    error = ct.create_string_buffer(512)
    code = _library().probtf_physical_camera_occlusion_mask(
        width, height, x0, y0, x1, y1,
        *(array.ctypes.data_as(_DOUBLE) for array in arrays),
        *parameters, board.ctypes.data_as(_BYTE), output.ctypes.data_as(_BYTE),
        error, len(error))
    if code:
        raise ValueError(error.value.decode(errors='replace'))
    return output
