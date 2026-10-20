"""Estimate stationary raw-vector noise and its exact isotropic Bingham scale.

For y = R.T v + e, e ~ N(0, sigma^2 I), the orientation-dependent
log likelihood is v.T R y / sigma^2.  Normalizing the two directions
therefore gives concentration ||v|| ||y|| / sigma^2, without a small-noise
approximation.  Calibration means are not subtracted from measurements.
"""
import numpy as np


def estimate_vector_noise(samples):
    """Measure the full covariance of one stationary three-axis reading."""
    values = np.asarray(samples, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) < 2
            or not np.isfinite(values).all()):
        raise ValueError('At least two finite three-dimensional stationary measurements are required.')
    mean = values.mean(axis=0)
    centered = values-mean
    covariance = centered.T@centered/(len(values)-1)
    correlations = []
    for component in range(3):
        previous = values[:-1, component]-values[:-1, component].mean()
        following = values[1:, component]-values[1:, component].mean()
        denominator = np.linalg.norm(previous)*np.linalg.norm(following)
        correlations.append(float(previous@following/denominator) if denominator else None)
    return dict(count=len(values), mean=mean.tolist(), sample_covariance=covariance.tolist(),
        covariance_divisor=len(values)-1, lag_one_autocorrelation=correlations,
        covariance_of_individual_measurements=True, covariance_divided_by_sample_count=False)


def estimate_isotropic_sensor_noise(samples):
    """Keep the measured covariance and use its trace for isotropic noise.

    A zero stationary mean is valid for angular velocity. Reference magnitude
    requirements belong to the directional likelihood, not this noise model.
    """
    result = estimate_vector_noise(samples)
    covariance = np.asarray(result['sample_covariance'])
    variance = float(np.trace(covariance)/3)
    return dict(result, isotropic_variance=variance,
        isotropic_covariance=(variance*np.eye(3)).tolist())


def estimate_isotropic_vector_noise(samples):
    result = estimate_isotropic_sensor_noise(samples)
    variance = result['isotropic_variance']
    magnitude = float(np.linalg.norm(result['mean']))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError('A positive finite measured noise variance is required; no variance floor is applied.')
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise ValueError('The stationary mean must define a nonzero reference magnitude.')
    return dict(result, reference_magnitude=magnitude)


def concentration_from_vector_noise(vectors, noise):
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Finite raw measurement vectors with shape (N,3) are required.')
    magnitudes = np.linalg.norm(values, axis=1)
    variance = float(noise['isotropic_variance'])
    reference_magnitude = float(noise['reference_magnitude'])
    if (not np.isfinite(variance) or variance <= 0 or not np.isfinite(reference_magnitude)
            or reference_magnitude <= 0 or np.any(magnitudes <= 0)):
        raise ValueError('Positive raw/reference magnitudes and noise variance are required.')
    return reference_magnitude*magnitudes/variance
