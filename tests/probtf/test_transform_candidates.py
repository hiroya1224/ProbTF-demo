"""Configured transform laws stay distinct from their derived joint draws."""
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest
import yaml

from probtf.candidates import JointFiniteTransformLaw, IndependentTransformLaw, load_initial_transform_law
from probtf.distributions import (
    BinghamOrientation, ConditionalGaussianTranslation, TransformComponent,
    TransformDistribution, TransformDistributionStamped,
)
from probtf.geometry import quat_to_rotmat, rotation_vector_from_quaternion


def component(name, x, weight=1., orientation=None, coupling=None):
    return TransformComponent(name, weight, BinghamOrientation.dirac([1., 0., 0., 0.])
        if orientation is None else orientation,
        ConditionalGaussianTranslation(np.array([x, 0., 0.]), np.zeros((3, 3)),
            np.zeros((3, 9)) if coupling is None else coupling))


def record(parent, child, components):
    return TransformDistributionStamped(parent, child, 0., parent+"_"+child, "test",
                                        TransformDistribution(tuple(components)))


def test_finite_joint_tuples_preserve_correlation_and_immutable_source():
    initial = np.broadcast_to(np.eye(4), (2, 2, 4, 4)).copy()
    initial[0, :, 0, 3], initial[1, :, 0, 3] = [1., 10.], [2., 20.]
    names = ["first", "second"]
    source = JointFiniteTransformLaw(initial, [1., 3.], names, ["a", "b"])
    initial[:] = 0.
    names[0] = "changed"
    rows, weights = source.quadrature()
    np.testing.assert_array_equal(rows[:, :, 0, 3], [[1., 10.], [2., 20.]])
    np.testing.assert_array_equal(weights, [.25, .75])
    assert source.node_ids == ("first", "second")
    assert source.is_finite
    assert source.query_metadata()["integration_method"] == "finite_sum"
    assert source.deterministic_local_indices == ()
    assert len(source.descriptor()["initial_local_transforms"]) == 2
    with pytest.raises(ValueError):
        rows[0, 0, 0, 0] = 2.
    with pytest.raises(FrozenInstanceError):
        source.revision = 2


def test_explicit_independence_enumerates_full_native_discrete_product_exactly():
    first = record("base", "a", [component("l", 1., .2), component("r", 2., .8)])
    second = record("a", "b", [component("x", 10., .1), component("y", 20., .3), component("z", 30., .6)])
    source = IndependentTransformLaw((first, second), independence=True)
    rows, weights = source.quadrature()
    assert source.edge_records == (first, second)
    np.testing.assert_array_equal(rows[:, :, 0, 3], [[1., 10.], [1., 20.], [1., 30.],
                                                   [2., 10.], [2., 20.], [2., 30.]])
    np.testing.assert_allclose(weights, [.02, .06, .12, .08, .24, .48], atol=1e-16)
    np.testing.assert_allclose(weights @ rows[:, :, 0, 3], [1.8, 25.], atol=1e-14)
    assert source.is_finite and len(source.node_ids) == 6
    assert source.quadrature(999)[0] is rows
    assert source.descriptor()["independence"] is True
    assert source.query_metadata()["result_kind"] == "joint_finite_pushforward"


def test_native_bingham_and_coupling_remain_source_while_draws_are_reused():
    orientation = BinghamOrientation.from_parameter_matrix(np.diag([0., -2., -3., -4.]))
    coupling = np.arange(27).reshape(3, 9)*.003
    first = record("base", "a", [component("continuous", .1, orientation=orientation, coupling=coupling)])
    second = record("a", "b", [component("fixed", .4)])
    source = IndependentTransformLaw((first, second), True, sample_count=64, seed=7)
    rows, weights = source.quadrature()
    assert source.edge_records[0] is first
    assert first.distribution.components[0].orientation is orientation
    assert not source.is_finite
    assert source.quadrature(3)[0] is rows
    assert source.node_ids[0] == "sample_000000"
    np.testing.assert_array_equal(weights, np.full(64, 1./64))
    rho = rows[:, 0, :3, :3].transpose(0, 2, 1).reshape(-1, 9)
    reference = rotation_vector_from_quaternion(orientation.reference_quaternion_wxyz)
    expected = np.array([.1, 0., 0.]) + (rho-reference) @ coupling.T
    np.testing.assert_allclose(rows[:, 0, :3, 3], expected, atol=1e-16)
    assert source.deterministic_local_indices == (1,)
    metadata = source.query_metadata()
    assert metadata["integration_method"] == "monte_carlo"
    assert metadata["result_kind"] == "joint_transform_samples"
    assert metadata["sample_count"] == 64 and metadata["seed"] == 7
    descriptor = source.descriptor()
    assert "initial_local_transforms" not in descriptor
    assert descriptor["edges"][0]["components"][0]["orientation"]["kind"] == "finite_bingham"
    json.dumps(descriptor, allow_nan=False)
    same = IndependentTransformLaw((first, second), True, sample_count=64, seed=7)
    different = IndependentTransformLaw((first, second), True, sample_count=64, seed=8)
    np.testing.assert_array_equal(same.quadrature()[0], rows)
    assert not np.array_equal(different.quadrature()[0], rows)


def test_dirac_with_native_nonzero_coupling_is_still_exact_on_its_support():
    native = component("fixed", .2, coupling=np.ones((3, 9)))
    source = IndependentTransformLaw((record("base", "a", [native]),), True)
    assert source.is_finite and source.deterministic_local_indices == (0,)
    assert source.quadrature()[0].shape == (1, 1, 4, 4)
    np.testing.assert_array_equal(native.translation.rotation_coupling, np.ones((3, 9)))


def test_uniform_native_source_is_not_an_empirical_finite_law():
    source = IndependentTransformLaw((record("base", "a", [component("uniform", 0.,
        orientation=BinghamOrientation.uniform())]),), True, sample_count=16, seed=3)
    assert not source.is_finite
    assert source.query_metadata()["sample_count"] == 16
    json.dumps(source.descriptor(), allow_nan=False)


@pytest.mark.parametrize("declaration", [None, False, "true", 1])
def test_undeclared_or_unresolved_dependence_is_rejected(declaration):
    edge = record("base", "a", [component("a", 0.)])
    with pytest.raises(ValueError, match="explicit independence"):
        IndependentTransformLaw((edge,), declaration)


def test_yaml_loader_builds_native_finite_joint_and_continuous_sources(tmp_path):
    pose = dict(translation=[.1, .2, .3], quaternion_wxyz=[1., 0., 0., 0.])
    path = tmp_path / "law.yaml"
    path.write_text(yaml.safe_dump(dict(kind="joint_finite", law_id="joint", revision=2,
        frame_names=["a"], reference_command=[.1], scenarios=[dict(id="scenario", weight=1., local_transforms=[pose])])) )
    finite = load_initial_transform_law(path)
    assert finite.reference_command == (.1,) and finite.revision == 2
    np.testing.assert_array_equal(finite.quadrature()[0][0, 0, :3, 3], [.1, .2, .3])
    config = dict(kind="independent", independence=True, law_id="native", sampling=dict(count=8, seed=2),
        edges=[dict(parent="base", child="a", components=[dict(id="bingham", weight=1.,
            orientation=dict(kind="bingham", parameter_matrix=np.diag([0., -1., -2., -3.]).tolist()),
            translation=dict(mean=[0., 0., 0.]))])])
    path.write_text(yaml.safe_dump(config))
    continuous = load_initial_transform_law(path)
    assert not continuous.is_finite
    assert continuous.query_metadata()["sample_count"] == 8
    del config["independence"]
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="explicit independence"):
        load_initial_transform_law(path)


def test_non_se3_and_negative_weights_are_rejected():
    rows = np.eye(4)[None, None].copy()
    rows[0, 0, 0, 0] = 2.
    with pytest.raises(ValueError, match=r"SE\(3\)"):
        JointFiniteTransformLaw(rows, [1.], ["one"], ["a"])
    with pytest.raises(ValueError, match="nonnegative"):
        IndependentTransformLaw((record("base", "a", [component("bad", 0., -1.), component("good", 1.)]),), True)
