import inspect
import json
from dataclasses import replace
import unittest

import numpy as np

from grape_param_estim.alternative_backends import (
    EXACT_ORACLE_PROTOCOL,
    PC_EXACT_ORACLE_CAPABILITIES,
    ExactOracleConformanceFixture,
    ExactOracleFixtureProvenance,
    ExactOracleIdentity,
    ExactOracleReplayOutput,
    evaluate_exact_oracle_conformance,
)
from grape_param_estim.controller import (
    ControllerBackendIdentity,
    ControllerCommand,
    ControllerCoreInput,
    ControllerCoreOutput,
    ControllerCoreState,
    ControllerSnapshot,
    ControllerStaticOptions,
    ControllerTask,
    FIDELITY_ACTUATOR_CALIBRATED,
    FIDELITY_PC_EXACT,
    FIDELITY_PLANT_CLOSED_LOOP,
    PythonSurrogateControllerBackend,
    check_capabilities,
    evaluate_exact_closed_loop_gate,
    run_factual_controller_replay,
)
from grape_param_estim.controller.contracts import (
    ACTUATOR_CALIBRATED_REQUIRED_CAPABILITIES,
    CAPABILITY_PLANT_CLOSED_LOOP,
    CAPABILITY_PLANT_STATE,
    PC_EXACT_REQUIRED_CAPABILITIES,
)


def static_options():
    return ControllerStaticOptions(
        gimbal_dof=1,
        gimbal_calc_in_fc=False,
        hovering_approximate=False,
        underactuate=False,
        need_yaw_d_control=True,
        integration_start_height=0.01,
        force_landing_descending_rate=-0.5,
        estimate_mode=0,
    )


def snapshot(p_gain=1.0):
    return ControllerSnapshot(
        backend_id="gimbalrotor_controller_cpp/v2",
        source_commit="2786cc3e",
        artifact_sha256="a" * 64,
        nominal_model_sha256="b" * 64,
        parameter_dump_sha256="c" * 64,
        controller_rate_hz=200.0,
        gains={
            "p_gain": [p_gain, 0.0, 0.0, 0.0, 0.0, 0.0],
            "i_gain": [0.0] * 6,
            "d_gain": [0.0] * 6,
        },
        limits={
            "limit_sum": [100.0] * 6,
            "limit_p": [100.0] * 6,
            "limit_i": [100.0] * 6,
            "limit_d": [100.0] * 6,
            "limit_err_p": [100.0] * 6,
            "limit_err_i": [100.0] * 6,
            "limit_err_d": [100.0] * 6,
        },
        static_options=static_options(),
        nominal_mass=2.0,
        nominal_cog=[0.0, 0.0, 0.0],
        nominal_inertia=np.diag([1.0, 1.1, 1.2]),
        nominal_geometry={"rotor_count": 4, "motor_order": ["1", "2", "3", "4"]},
    )


def core_input():
    identity = np.eye(3)
    return ControllerCoreInput(
        stamp=10.0,
        dt=0.005,
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        orientation=identity,
        angular_velocity=[0.0, 0.0, 0.0],
        target_position=[1.0, 0.0, 0.0],
        target_velocity=[0.0, 0.0, 0.0],
        target_acceleration=[0.0, 0.0, 0.0],
        target_orientation=identity,
        target_angular_velocity=[0.0, 0.0, 0.0],
        target_angular_acceleration=[0.0, 0.0, 0.0],
        control_mode=[0] * 6,
        integration_enabled=[False] * 6,
        flight_state=2,
        force_landing=False,
        joint_positions=[0.0] * 4,
    )


class ControllerContractTests(unittest.TestCase):
    def test_contracts_are_immutable_serializable_and_hashable(self):
        mutable_gains = {
            "p_gain": np.ones(6),
            "i_gain": np.zeros(6),
            "d_gain": np.zeros(6),
        }
        value = snapshot()
        value_from_mutable = value.with_updates(gains=mutable_gains)
        frozen_hash = value_from_mutable.content_sha256
        mutable_gains["p_gain"][0] = 999.0
        self.assertEqual(value_from_mutable.content_sha256, frozen_hash)
        self.assertIsInstance(hash(value_from_mutable), int)
        self.assertEqual(
            ControllerSnapshot.from_mapping(
                json.loads(value_from_mutable.to_json())
            ),
            value_from_mutable,
        )

        item = core_input()
        state = ControllerCoreState(pending_events=1)
        command = ControllerCommand(
            stamp=item.stamp,
            base_thrust=[1.0, 2.0],
            gimbal_angle=[0.1, -0.1],
            generalized_wrench=[0.0] * 6,
            events=[1, 3],
            saturated=True,
        )
        output = ControllerCoreOutput(
            pid_result=[0.0] * 6,
            pid_p_term=[0.0] * 6,
            pid_i_term=[0.0] * 6,
            pid_d_term=[0.0] * 6,
            target_vectoring_force=[0.0] * 8,
            base_thrust=command.base_thrust,
            gimbal_angle=command.gimbal_angle,
            torque_allocation_matrix_inverse=np.zeros((8, 3)),
            target_roll=0.0,
            target_pitch=0.0,
            candidate_yaw_term=0.0,
            events=command.events,
            stamp=item.stamp,
            saturated=command.saturated,
            generalized_wrench=command.generalized_wrench,
        )
        for contract in (item, state, command, output):
            self.assertIsInstance(hash(contract), int)
            self.assertEqual(
                len(contract.content_sha256),
                64,
            )
            json.loads(contract.to_json())
        self.assertEqual(
            ControllerCoreState.from_mapping(
                json.loads(state.to_json())
            ).pending_events,
            1,
        )
        self.assertEqual(output.command.base_thrust, command.base_thrust)

    def test_snapshot_requires_all_plan_static_options(self):
        values = static_options().to_mapping()
        del values["estimate_mode"]
        with self.assertRaisesRegex(ValueError, "estimate_mode"):
            snapshot().with_updates(static_options=values)

    def test_serialized_flags_reject_truthy_non_booleans(self):
        values = static_options().to_mapping()
        values["underactuate"] = "false"
        with self.assertRaisesRegex(TypeError, "underactuate"):
            snapshot().with_updates(static_options=values)

        values = core_input().to_mapping()
        values["force_landing"] = 1
        with self.assertRaisesRegex(TypeError, "force_landing"):
            ControllerCoreInput.from_mapping(values)
        with self.assertRaisesRegex(ValueError, "pending_events"):
            ControllerCoreState(pending_events=-1)
        with self.assertRaisesRegex(ValueError, "pending_events"):
            ControllerCoreState(pending_events=1.0)


class ControllerArchitectureTests(unittest.TestCase):
    def test_plant_parameters_cannot_enter_factual_controller_replay(self):
        parameters = tuple(
            inspect.signature(run_factual_controller_replay).parameters
        )
        self.assertNotIn("plant_parameters", parameters)
        self.assertNotIn("plant_hypothesis", parameters)

        item = core_input()
        fixed_snapshot = snapshot(p_gain=1.0)
        first = run_factual_controller_replay(
            PythonSurrogateControllerBackend(),
            fixed_snapshot,
            ControllerCoreState(),
            [item],
        )
        # Different plant hypotheses exist outside this controller-only call.
        plant_a = {"mass": 1.0, "thrust_scale": 0.5}
        plant_b = {"mass": 20.0, "thrust_scale": 4.0}
        self.assertNotEqual(plant_a, plant_b)
        second = run_factual_controller_replay(
            PythonSurrogateControllerBackend(),
            fixed_snapshot,
            ControllerCoreState(),
            [item],
        )
        self.assertEqual(first.outputs, second.outputs)

    def test_changing_controller_snapshot_changes_surrogate_output(self):
        item = core_input()
        low = run_factual_controller_replay(
            PythonSurrogateControllerBackend(),
            snapshot(p_gain=1.0),
            ControllerCoreState(),
            [item],
        )
        high = run_factual_controller_replay(
            PythonSurrogateControllerBackend(),
            snapshot(p_gain=2.0),
            ControllerCoreState(),
            [item],
        )
        self.assertAlmostEqual(low.outputs[0].pid_result[0], 1.0)
        self.assertAlmostEqual(high.outputs[0].pid_result[0], 2.0)
        self.assertNotEqual(low.outputs, high.outputs)

    def test_surrogate_never_passes_exact_closed_loop_gate(self):
        report = evaluate_exact_closed_loop_gate(
            PythonSurrogateControllerBackend(), True
        )
        self.assertFalse(report.passed)
        self.assertIn("surrogate", " ".join(report.reasons).lower())


class FidelityCapabilityTests(unittest.TestCase):
    def test_explicit_fidelity_and_task_capability_checks(self):
        identity = ControllerBackendIdentity(
            backend_id="gimbalrotor_pc_cpp/v2",
            fidelity=FIDELITY_PC_EXACT,
            is_exact=True,
            capabilities=PC_EXACT_REQUIRED_CAPABILITIES,
            implementation_language="C++",
            source_commit="2786cc3e",
            artifact_sha256="d" * 64,
        )
        self.assertTrue(
            check_capabilities(identity, fidelity=FIDELITY_PC_EXACT).passed
        )
        actuator = check_capabilities(
            ACTUATOR_CALIBRATED_REQUIRED_CAPABILITIES,
            fidelity=FIDELITY_ACTUATOR_CALIBRATED,
        )
        self.assertTrue(actuator.passed)
        plant = check_capabilities(
            (CAPABILITY_PLANT_CLOSED_LOOP, CAPABILITY_PLANT_STATE),
            fidelity=FIDELITY_PLANT_CLOSED_LOOP,
        )
        self.assertTrue(plant.passed)
        closed_loop = check_capabilities(
            (
                *PC_EXACT_REQUIRED_CAPABILITIES,
                CAPABILITY_PLANT_CLOSED_LOOP,
                CAPABILITY_PLANT_STATE,
            ),
            task=ControllerTask.CLOSED_LOOP_PLANT_IDENTIFICATION,
        )
        self.assertTrue(closed_loop.passed)

    def test_pc_exact_requires_gimbal_but_not_pwm(self):
        no_pwm = tuple(PC_EXACT_REQUIRED_CAPABILITIES)
        self.assertNotIn("pwm", no_pwm)
        self.assertTrue(
            check_capabilities(no_pwm, fidelity=FIDELITY_PC_EXACT).passed
        )
        missing_gimbal = tuple(
            item for item in no_pwm if item != "gimbal_command"
        )
        report = check_capabilities(
            missing_gimbal, fidelity=FIDELITY_PC_EXACT
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.missing, ("gimbal_command",))

    def test_pc_exact_oracle_conformance_does_not_require_pwm(self):
        samples = 8
        base = np.linspace(0.0, 1.0, samples)[:, None]
        continuous = {
            "command_timestamp": np.linspace(
                1.0, 1.07, samples
            )[:, None],
            "pid_terms": np.hstack((base, base, base)),
            "four_axis_command": np.hstack((base, base, base, base)),
            "vectoring_force": np.hstack((base, base)),
            "gimbal_command": np.hstack((base, -base)),
            "allocation_internal": np.hstack(
                (base, -base, base**2)
            ),
            "torque_allocation_matrix_inverse": np.hstack(
                (base, -base, base**2, -base**2)
            ),
        }
        events = np.arange(samples) % 2
        payload = {"fixture": "pc-only"}
        provenance = ExactOracleFixtureProvenance.create(
            source_bag_sha256="1" * 64,
            source_topics=(
                "/debug/pose/pid",
                "/four_axes/command",
                "/target_vectoring_force",
                "/gimbals_ctrl",
            ),
            interval_start_time_ns=1,
            interval_end_time_ns=10,
            frame_conventions={"controller": "body_flu"},
            unit_conventions={"angle": "rad"},
            motor_order=("1", "2", "3", "4"),
            request_payload=payload,
            continuous=continuous,
            events=events,
            extraction_config_sha256="2" * 64,
            source_commit="fixture-source",
        )
        fixture = ExactOracleConformanceFixture(
            continuous=continuous,
            events=events,
            provenance=provenance,
            fidelity=FIDELITY_PC_EXACT,
        )
        identity = ExactOracleIdentity(
            protocol=EXACT_ORACLE_PROTOCOL,
            backend_id="gimbalrotor_pc_cpp/v2",
            implementation_language="C++",
            source_commit="2786cc3e",
            artifact_sha256="3" * 64,
            capabilities=PC_EXACT_ORACLE_CAPABILITIES,
            fidelity=FIDELITY_PC_EXACT,
        )

        class Oracle:
            is_exact = True

            def __init__(self):
                self.identity = identity

            def replay(self, request):
                self.request = request
                return ExactOracleReplayOutput(
                    identity=self.identity,
                    continuous=continuous,
                    events=events,
                )

        report = evaluate_exact_oracle_conformance(
            Oracle(), payload, fixture
        )
        self.assertTrue(report.passed)
        self.assertNotIn("pwm", report.channel_metrics)
        self.assertIn("gimbal_command", report.channel_metrics)
        self.assertIn("command_timestamp", report.channel_metrics)

        controller_identity = ControllerBackendIdentity(
            backend_id=identity.backend_id,
            fidelity=identity.fidelity,
            is_exact=True,
            capabilities=identity.capabilities,
            implementation_language=identity.implementation_language,
            source_commit=identity.source_commit,
            artifact_sha256=identity.artifact_sha256,
            protocol=identity.protocol,
        )
        gate = evaluate_exact_closed_loop_gate(
            controller_identity, report
        )
        self.assertTrue(gate.passed)
        self.assertEqual(
            gate.factual_evidence_sha256, report.evidence_sha256
        )
        with self.assertRaises(TypeError):
            report.channel_metrics["forged"] = next(
                iter(report.channel_metrics.values())
            )
        with self.assertRaises(ValueError):
            report.channel_metrics["command_timestamp"].normalized_rmse[
                0
            ] = 1.0
        loaded_report = type(report).from_mapping(
            report.to_mapping()
        )
        self.assertEqual(
            loaded_report.to_mapping(), report.to_mapping()
        )
        loaded_gate = type(gate).from_mapping(gate.to_mapping())
        self.assertEqual(
            loaded_gate.to_mapping(), gate.to_mapping()
        )

        bare = evaluate_exact_closed_loop_gate(
            controller_identity, True
        )
        duck = evaluate_exact_closed_loop_gate(
            controller_identity,
            type("Duck", (), {"passed": True})(),
        )
        self.assertFalse(bare.passed)
        self.assertFalse(duck.passed)
        self.assertIn("ExactOracleConformanceReport", bare.reasons[0])

        mismatched_oracle_identity = replace(
            identity, artifact_sha256="4" * 64
        )
        mismatched_report = replace(
            report,
            identity=mismatched_oracle_identity,
            evidence_sha256="",
        )
        mismatched_gate = evaluate_exact_closed_loop_gate(
            controller_identity, mismatched_report
        )
        self.assertFalse(mismatched_gate.passed)
        self.assertIn(
            "identity/artifact/source",
            " ".join(mismatched_gate.reasons),
        )
        source_mismatch = replace(
            report,
            identity=replace(identity, source_commit="different-source"),
            evidence_sha256="",
        )
        self.assertFalse(
            evaluate_exact_closed_loop_gate(
                controller_identity, source_mismatch
            ).passed
        )
        with self.assertRaisesRegex(ValueError, "fidelity"):
            replace(
                report,
                fidelity="pc_mcu_exact",
                evidence_sha256="",
            )

        tampered = dict(report.to_mapping())
        tampered["request_payload_sha256"] = "5" * 64
        with self.assertRaisesRegex(ValueError, "request|evidence"):
            type(report).from_mapping(tampered)

        class TimestampMismatchOracle(Oracle):
            def replay(self, request):
                changed = dict(continuous)
                changed["command_timestamp"] = np.array(
                    continuous["command_timestamp"], copy=True
                )
                changed["command_timestamp"][0, 0] += 1.0e-12
                return ExactOracleReplayOutput(
                    identity=self.identity,
                    continuous=changed,
                    events=events,
                )

        timestamp_report = evaluate_exact_oracle_conformance(
            TimestampMismatchOracle(), payload, fixture
        )
        self.assertFalse(timestamp_report.passed)
        self.assertIn(
            "command_timestamp",
            " ".join(timestamp_report.reasons),
        )


if __name__ == "__main__":
    unittest.main()
