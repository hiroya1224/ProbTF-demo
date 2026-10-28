"""Physical PID search boxes, independent of term-clamp normalization scales."""
from pathlib import Path
import json
import numpy as np

GROUPS = ('xy', 'z', 'roll_pitch', 'yaw')

def validate_box(bounds, anchors, minimum_relative_clearance=0.125):
    box = np.asarray(bounds, dtype=float)
    triples = np.asarray(anchors, dtype=float)
    if box.shape != (2, 3) or not np.all(np.isfinite(box)):
        raise ValueError('bounds must be two finite P/I/D triples')
    lower, upper = box
    if np.any(lower < 0) or np.any(upper <= lower):
        raise ValueError('exploration bounds must be nonnegative and strictly increasing')
    if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples) or not np.all(np.isfinite(triples)):
        raise ValueError('recorded anchors must be nonempty finite P/I/D triples')
    if np.any(triples <= lower) or np.any(triples >= upper):
        raise ValueError('all recorded PID gains must be strictly inside the exploration box')
    clearance = float(minimum_relative_clearance)
    if not np.isfinite(clearance) or not 0 <= clearance < .5:
        raise ValueError('minimum relative clearance must lie in [0, 0.5)')
    relative = np.minimum(triples - lower, upper - triples) / (upper - lower)
    if np.any(relative < clearance):
        raise ValueError('recorded PID gains are too close to an exploration boundary')
    return relative

def axes_for_box(size, bounds, anchors, minimum_relative_clearance=0.125):
    if int(size) != size or size < 2:
        raise ValueError('grid size must be an integer >= 2')
    validate_box(bounds, anchors, minimum_relative_clearance)
    lower, upper = np.asarray(bounds, dtype=float)
    triples = np.asarray(anchors, dtype=float)
    return tuple(np.unique(np.r_[np.linspace(lower[k], upper[k], int(size)), triples[:, k]]) for k in range(3))

def load_configuration(path, code_dir, load_estimate, recorded_gains, extra_estimates=()):
    """Construct the box and its sole anchor from this failure record only.

    The scale factors are declared before evaluating the grid. No successful
    flight gain file is loaded, even if it exists next to the failure inputs.
    """
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())
    if config.get('schema') != 'grape-param-estim/pid-voxel-failure-domains/v2':
        raise ValueError('unsupported PID domain configuration; success-anchored domains are retired')
    if len(extra_estimates) != 1:
        raise ValueError('exactly one recorded failure estimate is required for the domain')
    source = Path(extra_estimates[0]).expanduser()
    source = (Path(code_dir) / source).resolve() if not source.is_absolute() else source.resolve()
    gains = np.asarray(recorded_gains(load_estimate(source)), dtype=float)
    if gains.shape != (4, 3) or np.any(~np.isfinite(gains)) or np.any(gains <= 0):
        raise ValueError('failure-domain gains must be finite positive 4-by-3 values')
    lower = float(config['lower_factor'])
    upper = float(config['upper_factor'])
    if not 0 <= lower < 1 < upper or not np.isfinite(upper):
        raise ValueError('domain factors must satisfy 0 <= lower < 1 < upper')
    clearance = float(config['minimum_relative_clearance'])
    anchors = {name: gains[j:j+1].copy() for j, name in enumerate(GROUPS)}
    bounds = {name: np.stack((lower*gains[j], upper*gains[j]))
              for j, name in enumerate(GROUPS)}
    report = {name: {'lower': bounds[name][0].tolist(), 'upper': bounds[name][1].tolist(),
                     'recorded_anchors': anchors[name].tolist(),
                     'basis': 'this recorded failure controller',
                     'lower_factor': lower, 'upper_factor': upper,
                     'minimum_relative_clearance': float(validate_box(bounds[name], anchors[name], clearance).min())}
              for name in GROUPS}
    return bounds, anchors, report, [source], clearance
