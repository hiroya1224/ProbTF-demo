"""Check the faster equilibrium path against physical and original-solver results."""
from pathlib import Path

import numpy as np
import pytest

from deflecomp_core.model.equilibrium import EquilibriumConfig, EquilibriumSolver
from deflecomp_core.model.spring import LinearSpringModel
from deflecomp_core.robot.pinocchio_robot import RobotArm
from probtf_switch_press_demo.body import EquilibriumBody
from probtf_switch_press_demo.scene import load_config, write_robot


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ros/examples/probtf_switch_press_demo"


@pytest.fixture(scope="module")
def robot_scene(tmp_path_factory):
    config = load_config(PACKAGE / "config/default.yaml")
    urdf = write_robot(ROOT / "ros/examples/symaware_grasp/urdf/simple_six_dof_arm.urdf",
                       tmp_path_factory.mktemp("equilibrium_branch") / "rod.urdf", config)
    return urdf, np.asarray(config["initial_command"], dtype=float)


def original_solver(urdf):
    robot = RobotArm(str(urdf), tip_link="tool0", base_link="base_link")
    return EquilibriumSolver(robot, LinearSpringModel(), EquilibriumConfig(
        n_lambda=1, maxiter=100, ftol=1e-14, refine=True, refine_tol=1e-14))


def assert_stable_bounded_equilibrium(body, pose, command):
    assert np.all(np.isfinite(pose))
    assert np.all(pose >= body.lower) and np.all(pose <= body.upper)
    gradient = body.robot.tau_gravity(pose) + body.spring.torque(pose, command, body.stiffness)
    lower = pose <= body.lower + 1e-8
    upper = pose >= body.upper - 1e-8
    assert np.all(gradient[lower] >= -1e-8)
    assert np.all(gradient[upper] <= 1e-8)
    np.testing.assert_allclose(gradient[~(lower | upper)], 0., atol=1e-8, rtol=0.)
    assert np.linalg.eigvalsh(body.robot.d_tau_gravity(pose) + np.diag(body.stiffness))[0] > 0.
    return gradient


@pytest.mark.parametrize("stiffness", [
    [200., 65., 50., 35., 20., 20.],
    [350., 125., 95., 60., 40., 30.],
    [700., 260., 210., 130., 90., 65.],
])
def test_warm_started_equilibria_match_independent_potential_minimization(robot_scene, stiffness):
    urdf, initial = robot_scene
    body = EquilibriumBody(urdf, stiffness)
    reference = original_solver(urdf)
    assert body.robot.data is not reference.robot.data
    warm = None
    for increment in (np.zeros(6), [.03, -.04, .025, -.01, .02, .01],
                      [.08, .06, -.07, .01, -.05, -.02]):
        command = initial + increment
        result = body.equilibrium(command, warm)
        expected = reference.solve(command, np.asarray(stiffness),
            theta_init=command if warm is None else warm, lambdas=np.array([0.]))
        assert_stable_bounded_equilibrium(body, result, command)
        # The reference optimizer's stopping tolerance leaves up to a few
        # 1e-8 radians of error; the torque-balance check above is tighter.
        np.testing.assert_allclose(result, expected, atol=5e-8, rtol=0.)
        warm = result.copy()


def test_joint_limit_equilibrium_preserves_reaction_torque_and_box_kkt(robot_scene):
    urdf, initial = robot_scene
    body = EquilibriumBody(urdf, [200., 65., 50., 35., 20., 20.])
    command = initial.copy()
    command[1] = body.lower[1]
    result = body.equilibrium(command)
    gradient = assert_stable_bounded_equilibrium(body, result, command)
    assert result[1] == pytest.approx(body.lower[1], abs=1e-12)
    # A joint stop balances this outward torque. Requiring zero raw torque
    # would incorrectly reject this valid constrained equilibrium.
    assert gradient[1] > .5
    reference = original_solver(urdf)
    expected = reference.solve(command, body.stiffness, theta_init=command, lambdas=np.array([0.]))
    np.testing.assert_allclose(result, expected, atol=1e-8, rtol=0.)


@pytest.mark.parametrize("use_warm", [False, True])
def test_failed_fast_path_falls_back_to_original_bounded_solver(robot_scene, monkeypatch, use_warm):
    urdf, initial = robot_scene
    body = EquilibriumBody(urdf, [350., 125., 95., 60., 40., 30.])
    warm = body.equilibrium(initial) if use_warm else None
    command = initial + [.02, -.03, .01, .005, -.02, .01]
    reference = original_solver(urdf)
    expected = reference.solve(command, body.stiffness,
        theta_init=command if warm is None else warm, lambdas=np.array([0.]))
    calls = []
    original = body.solver.solve

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(body, "_newton_equilibrium", lambda _command, _warm: None)
    monkeypatch.setattr(body.solver, "solve", fallback)
    result = body.equilibrium(command, warm)
    assert len(calls) == 1
    arguments, keywords = calls[0]
    np.testing.assert_array_equal(arguments[0], command)
    np.testing.assert_array_equal(arguments[1], body.stiffness)
    np.testing.assert_array_equal(keywords["theta_init"], command if warm is None else warm)
    np.testing.assert_array_equal(keywords["lambdas"], [0.])
    np.testing.assert_array_equal(result, expected)
    # Fallback retains the original optimizer's physical tolerance rather
    # than requiring the tighter Newton residual used in the other tests.
    residual = body.robot.tau_gravity(result) + body.spring.torque(result, command, body.stiffness)
    assert np.max(np.abs(residual)) < 1e-4
    assert np.linalg.eigvalsh(body.robot.d_tau_gravity(result) + np.diag(body.stiffness))[0] > 0.
