"""Analytic acceptance checks for the frozen-distribution fusion contract."""

import math
from dataclasses import replace

import numpy as np
import pytest

from probtf.spherical_law.fusion import (
    FrozenGaussianQuery, Gaussian, LabelledPrior, PosteriorComponent,
    SensorPosterior, condition_gaussian, fuse_labelled_sensor_posteriors,
    fuse_sensor_posteriors, gaussian_reference_product, lift_sensor_posterior,
)
from probtf.spherical_law.support import covariance_support


def query(prior, h, mean=None, component_id="body", query_id="target"):
    h = np.atleast_2d(np.asarray(h, dtype=float))
    return FrozenGaussianQuery(
        prior, Gaussian(h @ prior.mean if mean is None else mean,
                        h @ prior.covariance @ h.T),
        h, query_id, 7, 123.25, (0.2, -0.1), component_id, "root")


def posterior(q, observed, noise, sensor_id, observation_id):
    estimate, _ = condition_gaussian(q.distribution, observed, noise)
    return SensorPosterior(q, sensor_id, observation_id,
                           (PosteriorComponent("single", estimate),))


def test_support_keeps_tangent_normal_exact_and_rejects_incompatible_ratios():
    prior = Gaussian([0.0, 0.0, 1.0], np.diag([0.04, 0.09, 0.0]))
    estimated, _ = condition_gaussian(prior, [0.1, -0.2, 1.0],
                                     np.diag([0.02, 0.03, 0.0]))
    assert covariance_support(estimated.covariance).rank == 2
    np.testing.assert_array_equal(estimated.covariance[2], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="outside"):
        condition_gaussian(prior, [0.1, -0.2, 1.1], np.diag([0.02, 0.03, 0.0]))
    with pytest.raises(ValueError, match="different covariance support"):
        gaussian_reference_product(prior, (
            Gaussian(prior.mean, np.diag([0.04, 0.0, 0.09])), prior))
    with pytest.raises(ValueError, match="negative"):
        covariance_support(np.diag([0.1, -0.01, 0.0]))


def test_lift_preserves_frozen_proxy_mean_and_both_total_covariance_terms():
    prior = Gaussian([0.2, -0.1, 0.4],
                     [[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    # The proxy correction is intentionally distinct from H mu.
    q = query(prior, [[1.0, 1.0, 0.0]], mean=[0.07])
    sensor = posterior(q, [0.3], [[0.2]], "sensor", "obs")
    lifted = lift_sensor_posterior(sensor)[0]
    np.testing.assert_allclose(lifted.within + lifted.between,
                               lifted.distribution.covariance, atol=1e-14)
    assert np.linalg.norm(lifted.within) > 0
    assert np.linalg.norm(lifted.between) > 0
    local = sensor.components[0].distribution
    np.testing.assert_allclose(q.intercept + q.jacobian @ lifted.distribution.mean,
                               local.mean)
    np.testing.assert_allclose(q.jacobian @ lifted.distribution.covariance @ q.jacobian.T,
                               local.covariance)
    assert not np.allclose(lifted.distribution.covariance,
                           np.diag(np.diag(lifted.distribution.covariance)))


def test_reference_correction_matches_information_differences_and_associativity():
    prior = Gaussian([0.2, -0.3], [[0.8, 0.2], [0.2, 0.6]])
    qa = query(prior, [[1.0, 0.3]], mean=[0.05])
    qb = query(prior, [[0.2, 1.0]], mean=[-0.2], query_id="second")
    a = posterior(qa, [0.3], [[0.12]], "a", "image-a")
    b = posterior(qb, [-0.1], [[0.09]], "b", "image-b")
    fused = fuse_sensor_posteriors(prior, (a, b))
    result = fused.components[0].distribution
    precision = np.linalg.inv(prior.covariance)
    eta = precision @ prior.mean
    for sensor in (a, b):
        local_prior = sensor.query.distribution
        local_posterior = sensor.components[0].distribution
        delta_precision = np.linalg.inv(local_posterior.covariance) - np.linalg.inv(
            local_prior.covariance)
        delta_eta = (np.linalg.solve(local_posterior.covariance, local_posterior.mean)
                     - np.linalg.solve(local_prior.covariance, local_prior.mean))
        h, c = sensor.query.jacobian, sensor.query.intercept
        precision += h.T @ delta_precision @ h
        eta += h.T @ (delta_eta - delta_precision @ c)
    expected_covariance = np.linalg.inv(precision)
    np.testing.assert_allclose(result.covariance, expected_covariance, atol=1e-13)
    np.testing.assert_allclose(result.mean, expected_covariance @ eta, atol=1e-13)
    reverse = fuse_sensor_posteriors(prior, (b, a)).components[0].distribution
    np.testing.assert_array_equal(reverse.mean, result.mean)
    np.testing.assert_array_equal(reverse.covariance, result.covariance)
    lifted_a = lift_sensor_posterior(a)[0].distribution
    lifted_b = lift_sensor_posterior(b)[0].distribution
    left, _ = gaussian_reference_product(prior, (lifted_a, prior))
    bracketed, _ = gaussian_reference_product(prior, (left, lifted_b))
    np.testing.assert_allclose(bracketed.covariance, result.covariance, atol=1e-13)
    np.testing.assert_allclose(bracketed.mean, result.mean, atol=1e-13)


def test_no_observation_identity_distribution_ratio_and_retransmission():
    prior = Gaussian([0.0, 0.0], [[0.04, 0.01], [0.01, 0.08]])
    empty = fuse_sensor_posteriors(prior, ())
    assert empty.components[0].distribution is prior
    q = query(prior, [[1.0, 0.0]])
    identity = SensorPosterior(q, "identity", "no-new-information",
                               (PosteriorComponent("same", q.distribution),))
    unchanged = fuse_sensor_posteriors(prior, (identity,)).components[0].distribution
    assert unchanged is prior
    actual = posterior(q, [0.1], [[0.02]], "sensor", "actual-observation")
    once = fuse_sensor_posteriors(prior, (actual,))
    repeated = fuse_sensor_posteriors(prior, (actual, actual, identity))
    np.testing.assert_allclose(repeated.components[0].distribution.mean,
                               once.components[0].distribution.mean, atol=1e-14)
    np.testing.assert_allclose(repeated.components[0].distribution.covariance,
                               once.components[0].distribution.covariance, atol=1e-14)
    assert repeated.observation_ids == ("actual-observation", "no-new-information")


def scalar_log_density(value, mean, variance):
    return -0.5 * (math.log(2 * math.pi * variance) + (value - mean) ** 2 / variance)


def test_mixture_combination_weights_include_density_product_normalizers():
    prior = Gaussian([0.0], [[1.0]])
    q = query(prior, [[1.0]])
    a = SensorPosterior(q, "a", "a-observation", (
        PosteriorComponent("negative", Gaussian([-0.6], [[0.4]]), 0.5),
        PosteriorComponent("positive", Gaussian([0.6], [[0.4]]), 0.5)))
    b = SensorPosterior(q, "b", "b-observation", (
        PosteriorComponent("positive", Gaussian([0.6], [[0.4]])),))
    fused = fuse_sensor_posteriors(prior, (a, b))
    assert len(fused.components) == 2
    # Independently evaluate C from the density identity at z=0.
    unnormalized = []
    for first, component in zip(a.components, fused.components):
        value = component.distribution
        log_c = (scalar_log_density(0, first.distribution.mean[0], 0.4)
                 + scalar_log_density(0, 0.6, 0.4)
                 - scalar_log_density(0, 0, 1)
                 - scalar_log_density(0, value.mean[0], value.covariance[0, 0]))
        unnormalized.append(0.5 * math.exp(log_c))
    expected = np.array(unnormalized) / sum(unnormalized)
    np.testing.assert_allclose([component.weight for component in fused.components], expected)
    assert expected[1] > expected[0]
    assert all(len(component.lineage) == 2 for component in fused.components)


def test_shared_prior_hypothesis_masses_are_not_counted_twice():
    priors = (LabelledPrior("h0", Gaussian([0.0], [[0.2]]), 0.3),
              LabelledPrior("h1", Gaussian([1.0], [[0.3]]), 0.7))
    sensors = []
    for sensor_id in ("a", "b"):
        for prior in priors:
            q = query(prior.distribution, [[1.0]], component_id=prior.label)
            sensors.append(SensorPosterior(q, sensor_id, sensor_id + "-obs", (
                PosteriorComponent("identity", q.distribution, prior.weight),)))
    result = fuse_labelled_sensor_posteriors(priors, sensors)
    np.testing.assert_allclose([component.weight for component in result.components], [0.3, 0.7])
    for component, prior in zip(result.components, priors):
        np.testing.assert_allclose(component.distribution.mean, prior.distribution.mean)
        np.testing.assert_allclose(component.distribution.covariance, prior.distribution.covariance)
    with pytest.raises(ValueError, match="correspondence"):
        fuse_labelled_sensor_posteriors(priors, sensors[:-1])


def test_frozen_version_mismatch_and_observation_id_collision_are_errors():
    prior = Gaussian([0.0], [[0.1]])
    q = query(prior, [[1.0]])
    a = posterior(q, [0.1], [[0.02]], "a", "obs")
    b = posterior(q, [0.2], [[0.02]], "a", "obs")
    with pytest.raises(ValueError, match="inconsistent"):
        fuse_sensor_posteriors(prior, (a, b))
    q2 = FrozenGaussianQuery(q.body_prior, q.distribution, q.jacobian,
                             q.query_id, 8, q.stamp, q.command, q.component_id)
    b = posterior(q2, [0.2], [[0.02]], "b", "other")
    with pytest.raises(ValueError, match="frozen version"):
        fuse_sensor_posteriors(prior, (a, b))
    changed_parameters = posterior(replace(q, state_id="different-transform-parameters"),
                                   [0.2], [[0.02]], "b", "other")
    with pytest.raises(ValueError, match="parameter identities"):
        fuse_sensor_posteriors(prior, (a, changed_parameters))


def test_propagation_lifting_and_fusion_do_not_use_randomness(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("Randomness is prohibited in the analytic calculation.")
    monkeypatch.setattr(np.random, "default_rng", prohibited)
    monkeypatch.setattr(np.random, "normal", prohibited)
    monkeypatch.setattr(np.random, "multivariate_normal", prohibited)
    prior = Gaussian([0.0, 0.0], [[0.1, 0.02], [0.02, 0.2]])
    q = query(prior, [[1.0, 1.0]])
    sensors = (posterior(q, [0.1], [[0.03]], "a", "a-obs"),
               posterior(q, [0.2], [[0.04]], "b", "b-obs"))
    assert len(fuse_sensor_posteriors(prior, sensors).components) == 1
