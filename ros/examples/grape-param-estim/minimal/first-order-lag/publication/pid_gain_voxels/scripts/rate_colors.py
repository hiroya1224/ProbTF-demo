"""One continuous 0–100% rate palette for saved figures and the offline viewer."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pid-focus-matplotlib')
import numpy as np
from matplotlib import colormaps, colors


def rate_colorscale(settings):
    positions = np.linspace(0, 1, 256)
    rgb = colormaps[settings['rate_colormap']](positions)[:, :3]
    white = float(settings['white_mix'])
    if not 0 <= white < 1:
        raise ValueError('white_mix must lie in [0, 1)')
    rgb = white + (1 - white) * rgb
    return [[float(q), colors.to_hex(color)] for q, color in zip(positions, rgb)]


def rate_colormap(settings):
    return colors.LinearSegmentedColormap.from_list('continuous_rate', rate_colorscale(settings), N=4096)
