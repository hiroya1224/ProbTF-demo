import importlib.util
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from grape_param_estim.data import (
    AUDIT_AVAILABLE,
    AUDIT_DERIVABLE,
    AUDIT_MISSING,
    BagTopicInventory,
    ControllerReplayFixture,
    EventGrid,
    EventScheduler,
    ReplayGrids,
    audit_controller_replay_inventory,
    build_replay_audit_bundle,
    write_replay_audit_bundle,
)


REPOSITORY = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPOSITORY
    / "ros/examples/grape-param-estim/scripts/audit_grape_controller_replay.py"
)


def _entry(message_type="std_msgs/Float32", count=1):
    return {"message_type": message_type, "message_count": count}


def _inventory(topics):
    return BagTopicInventory.from_mapping(
        bag_path="/immutable/episode.bag",
        source_bag_sha256="a" * 64,
        start_record_time=100.0,
        end_record_time=200.0,
        topics=topics,
    )


def _real_like_topics(include_nav=True):
    topics = {
        "/gimbalrotor/debug/pose/pid": _entry(
            "aerial_robot_msgs/PoseControlPid", 100
        ),
        "/gimbalrotor/debug/target_vectoring_force": _entry(
            "std_msgs/Float32MultiArray", 100
        ),
        "/gimbalrotor/four_axes/command": _entry(
            "spinal/FourAxisCommand", 100
        ),
        "/gimbalrotor/gimbals_ctrl": _entry("sensor_msgs/JointState", 100),
        "/gimbalrotor/joint_states": _entry("sensor_msgs/JointState", 50),
        "/gimbalrotor/uav/full_state": _entry(
            "aerial_robot_msgs/States", 50
        ),
        "/gimbalrotor/kf/imu1/data": _entry(
            "aerial_robot_msgs/States", 100
        ),
        "/gimbalrotor/flight_config_cmd": _entry(
            "spinal/FlightConfigCmd", 3
        ),
        "/gimbalrotor/flight_state": _entry("std_msgs/UInt8", 100),
        "/gimbalrotor/motor_pwms": _entry("spinal/Pwms", 20),
        "/gimbalrotor/rpy/pid": _entry("spinal/RollPitchYawTerms", 20),
        "/gimbalrotor/servo/target_states": _entry(
            "spinal/ServoControlCmd", 100
        ),
        "/tf_static": _entry("tf2_msgs/TFMessage", 2),
        "/gimbalrotor/uav_info": _entry("spinal/UavInfo", 1),
        "/gimbalrotor/joint_profiles": _entry("spinal/JointProfiles", 1),
    }
    for group in ("xy", "z", "roll_pitch", "yaw"):
        topics[
            "/gimbalrotor/controller/{}/parameter_updates".format(group)
        ] = _entry("dynamic_reconfigure/Config", 1)
        topics[
            "/gimbalrotor/controller/{}/parameter_descriptions".format(group)
        ] = _entry("dynamic_reconfigure/ConfigDescription", 1)
    if include_nav:
        topics["/gimbalrotor/uav/nav"] = _entry(
            "aerial_robot_msgs/FlightNav", 4
        )
    return topics


class EventSchedulerTests(unittest.TestCase):
    def test_grid_normalization_and_hash_are_deterministic(self):
        first = EventGrid.from_times(
            "controller_tick", [0.2, 0.1, 0.2, 0.3], sort_and_deduplicate=True
        )
        second = EventGrid("controller_tick", (0.1, 0.2, 0.3))
        self.assertEqual(first, second)
        self.assertEqual(first.content_sha256, second.content_sha256)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            EventGrid("bad", (0.1, 0.1))

    def test_scheduler_uses_explicit_priority_for_equal_times(self):
        controller = EventGrid("controller_tick", (0.0, 0.1))
        plant = EventGrid("plant_integration", (0.0, 0.05, 0.1))
        scheduler = EventScheduler(
            (plant, controller),
            priority=("controller_tick", "plant_integration"),
        )
        events = list(scheduler)
        self.assertEqual(
            [(item.time, item.grid_name) for item in events[:2]],
            [(0.0, "controller_tick"), (0.0, "plant_integration")],
        )
        self.assertEqual(
            scheduler.events_between(0.05, 0.1, include_end=False)[0].grid_name,
            "plant_integration",
        )


class ControllerReplayFixtureTests(unittest.TestCase):
    def grids(self):
        return ReplayGrids(
            controller_tick_grid=EventGrid(
                "controller_tick", (10.0, 11.0, 12.0, 13.0)
            ),
            plant_integration_grid=EventGrid(
                "plant_integration", (10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0)
            ),
            observation_grid=EventGrid(
                "observation", (10.5, 11.5, 12.5)
            ),
            likelihood_grid=EventGrid("likelihood", (12.0, 12.5, 13.0)),
            report_grid=EventGrid("report", (12.0, 13.0)),
        )

    def test_fixture_separates_preroll_scoring_and_all_five_grids(self):
        @dataclass(frozen=True)
        class Input:
            stamp: float
            value: float

        metadata = {
            "controller_rate_hz": 100.0,
            "topic_roles": ["/feedback"],
        }
        fixture = ControllerReplayFixture(
            episode_id="20260612-07",
            source_bag_sha256="a" * 64,
            topic_inventory_sha256="b" * 64,
            replay_start_offset_s=10.0,
            score_start_offset_s=12.0,
            score_end_offset_s=13.0,
            grids=self.grids(),
            controller_inputs=tuple(
                Input(stamp, float(index))
                for index, stamp in enumerate(
                    self.grids().controller_tick_grid.timestamps
                )
            ),
            metadata=metadata,
        )
        original_hash = fixture.fixture_sha256
        metadata["topic_roles"].append("/future-mutation")
        payload = fixture.to_dict()
        self.assertEqual(payload["fixture_sha256"], fixture.fixture_sha256)
        self.assertEqual(fixture.fixture_sha256, original_hash)
        self.assertEqual(payload["metadata"]["topic_roles"], ["/feedback"])
        self.assertEqual(len(payload["grids"]), 5)
        self.assertEqual(len(payload["controller_inputs"]), 4)
        self.assertEqual(
            fixture.scheduler().priority,
            (
                "controller_tick",
                "plant_integration",
                "observation",
                "likelihood",
                "report",
            ),
        )

    def test_irregular_controller_ticks_are_required_in_integration_grid(self):
        with self.assertRaisesRegex(
            ValueError, "every controller tick.*plant-integration"
        ):
            ReplayGrids(
                controller_tick_grid=EventGrid(
                    "controller_tick", (10.0, 10.037, 10.1)
                ),
                plant_integration_grid=EventGrid(
                    "plant_integration", (10.0, 10.05, 10.1)
                ),
                observation_grid=EventGrid(
                    "observation", (10.0, 10.1)
                ),
                likelihood_grid=EventGrid(
                    "likelihood", (10.0, 10.1)
                ),
                report_grid=EventGrid("report", (10.0, 10.1)),
            )
        grids = ReplayGrids(
            controller_tick_grid=EventGrid(
                "controller_tick", (10.0, 10.037, 10.1)
            ),
            plant_integration_grid=EventGrid(
                "plant_integration", (10.0, 10.037, 10.05, 10.1)
            ),
            observation_grid=EventGrid("observation", (10.0, 10.1)),
            likelihood_grid=EventGrid("likelihood", (10.0, 10.1)),
            report_grid=EventGrid("report", (10.0, 10.1)),
        )
        self.assertEqual(
            tuple(
                event.grid_name
                for event in grids.scheduler()
                if event.time == 10.037
            ),
            ("controller_tick", "plant_integration"),
        )

    def test_fixture_rejects_missing_preroll_or_report_outside_score(self):
        with self.assertRaisesRegex(ValueError, "pre-roll"):
            ControllerReplayFixture(
                episode_id="episode",
                source_bag_sha256="a" * 64,
                topic_inventory_sha256="b" * 64,
                replay_start_offset_s=12.0,
                score_start_offset_s=12.0,
                score_end_offset_s=13.0,
                grids=self.grids(),
            )
        invalid = ReplayGrids(
            controller_tick_grid=self.grids().controller_tick_grid,
            plant_integration_grid=self.grids().plant_integration_grid,
            observation_grid=self.grids().observation_grid,
            likelihood_grid=self.grids().likelihood_grid,
            report_grid=EventGrid("report", (11.5, 13.0)),
        )
        with self.assertRaisesRegex(ValueError, "report grid"):
            ControllerReplayFixture(
                episode_id="episode",
                source_bag_sha256="a" * 64,
                topic_inventory_sha256="b" * 64,
                replay_start_offset_s=10.0,
                score_start_offset_s=12.0,
                score_end_offset_s=13.0,
                grids=invalid,
            )


class ReplayAuditTests(unittest.TestCase):
    def test_real_like_inventory_is_conservative_and_covers_every_plan_field(self):
        audit = audit_controller_replay_inventory(
            _inventory(_real_like_topics()), episode_id="20260612-07"
        )
        by_field = {item.field: item for item in audit.fields}
        self.assertEqual(len(by_field), 15)
        self.assertEqual(by_field["controller_tick"].status, AUDIT_DERIVABLE)
        self.assertEqual(by_field["navigator_target"].status, AUDIT_AVAILABLE)
        self.assertEqual(by_field["control_mode"].status, AUDIT_DERIVABLE)
        self.assertEqual(
            by_field["nominal_model_geometry_snapshot"].status, AUDIT_MISSING
        )
        self.assertEqual(
            by_field["torque_allocation_matrix"].status, AUDIT_MISSING
        )
        self.assertFalse(audit.fixture_inputs_resolvable)
        self.assertFalse(audit.exact_replay_ready)
        self.assertEqual(audit.decision, "BLOCKED_MISSING_INPUTS")

    def test_bag_without_nav_fails_target_and_control_mode_closed(self):
        audit = audit_controller_replay_inventory(
            _inventory(_real_like_topics(include_nav=False)),
            episode_id="20260612-04",
        )
        by_field = {item.field: item for item in audit.fields}
        self.assertEqual(by_field["navigator_target"].status, AUDIT_MISSING)
        self.assertEqual(by_field["control_mode"].status, AUDIT_MISSING)
        self.assertIn("/gimbalrotor/debug/pose/pid", by_field["navigator_target"].observed_topics)

    def test_replay_frame_and_metadata_make_pc_fixture_ready(self):
        topics = _real_like_topics()
        topics[
            "/gimbalrotor/controller_replay/frame"
        ] = _entry("grape_param_estim/GimbalrotorControllerReplayFrame", 100)
        topics[
            "/gimbalrotor/controller_replay/metadata"
        ] = _entry("grape_param_estim/GimbalrotorControllerReplayMetadata", 1)
        audit = audit_controller_replay_inventory(
            _inventory(topics), episode_id="future"
        )
        self.assertTrue(audit.exact_replay_ready)
        self.assertEqual(
            {item.status for item in audit.fields}, {AUDIT_AVAILABLE}
        )
        self.assertEqual(audit.decision, "READY")

    def test_bundle_is_hash_bound_atomic_and_non_overwriting(self):
        audit = audit_controller_replay_inventory(
            _inventory(_real_like_topics()), episode_id="20260612-07"
        )
        bundle = build_replay_audit_bundle((audit,))
        with tempfile.TemporaryDirectory(prefix="grape-replay-audit-") as directory:
            destination = Path(directory) / "controller_replay_audit.json"
            write_replay_audit_bundle(bundle, destination)
            loaded = json.loads(destination.read_text())
            self.assertEqual(loaded["bundle_sha256"], bundle["bundle_sha256"])
            with self.assertRaises(FileExistsError):
                write_replay_audit_bundle(bundle, destination)
            tampered = dict(bundle)
            tampered["overall_exact_replay_ready"] = True
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                write_replay_audit_bundle(
                    tampered, Path(directory) / "tampered.json"
                )

    def test_cli_resolves_default_bag_4_7_8_names(self):
        spec = importlib.util.spec_from_file_location("replay_audit_cli", SCRIPT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory(prefix="grape-replay-bags-") as directory:
            root = Path(directory)
            for number in (4, 7, 8):
                (root / "20260612_grape_hovering_{}_unit.bag".format(number)).touch()
            arguments = module._arguments(["--bag-root", str(root)])
            paths = module._resolve_bags(arguments)
            self.assertEqual([module._episode_id(path) for path in paths], [
                "20260612-04",
                "20260612-07",
                "20260612-08",
            ])


if __name__ == "__main__":
    unittest.main()
