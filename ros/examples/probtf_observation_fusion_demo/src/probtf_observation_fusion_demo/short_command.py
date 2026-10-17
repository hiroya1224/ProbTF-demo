"""Time reference for the position selected by one IK update."""

import numpy as np


def command_reference(diagnostics, stamp):
    """Interpolate only the chosen next command; never extrapolate its velocity.

    This does not run IK, propagate a law, or predict further commands. The
    caller applies the existing actuator limits to the reference position.
    """
    try:
        source_stamp = float(diagnostics['source_stamp'])
        stamp = float(stamp)
        duration = float(diagnostics['duration'])
        source = np.asarray(diagnostics['source_command'], dtype=float)
        target = np.asarray(diagnostics['next_command'], dtype=float)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError('A position command requires finite times and matching joint vectors.') from error
    if (not np.all(np.isfinite([source_stamp, stamp, duration])) or duration <= 0.
            or source.ndim != 1 or source.size == 0 or target.shape != source.shape
            or not np.all(np.isfinite(source)) or not np.all(np.isfinite(target))):
        raise ValueError('A position command requires finite times and matching joint vectors.')
    nominal_end = source_stamp+duration
    if not np.isfinite(nominal_end) or nominal_end <= source_stamp:
        raise ValueError('The position command nominal end must be finite and later than its source.')
    past_duration, before_start = stamp >= nominal_end, stamp < source_stamp
    fraction = float(np.clip((stamp-source_stamp)/duration, 0., 1.))
    position = target if past_duration else source if before_start else source+fraction*(target-source)
    velocity = np.zeros_like(source) if past_duration or before_start else (target-source)/duration
    return dict(reference_command=position.tolist(), reference_velocity=velocity.tolist(),
        source_stamp=source_stamp, reference_stamp=stamp, duration=duration,
        nominal_end_stamp=nominal_end, past_selected_duration=bool(past_duration), before_start=bool(before_start),
        next_command=target.tolist(), future_IK_updates=0)
