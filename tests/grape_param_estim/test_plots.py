import importlib.util
import unittest

from test_data import make_recording
from test_model import make_analysis, nominal_parameters


@unittest.skipUnless(
    importlib.util.find_spec("plotly") is not None,
    "Plotly is an optional GUI dependency",
)
class PlotTest(unittest.TestCase):
    def test_phase_zero_figures_contain_selected_data(self):
        from grape_param_estim.plots import (
            make_bag_overview_figure,
            make_command_figure,
            make_state_figure,
            make_trajectory_figure,
        )

        recording = make_recording()
        data = recording.select_interval(0.2, 1.8, 0.5)

        overview = make_bag_overview_figure(recording, (0.2, 1.8))
        trajectory = make_trajectory_figure(data)
        state = make_state_figure(data)
        command = make_command_figure(data)

        self.assertEqual(len(overview.data), 13)
        self.assertEqual(len(trajectory.data), data.segment_count)
        self.assertEqual(len(state.data), 13)
        self.assertEqual(len(command.data), 12)

    def test_phase_one_figures_show_replay_and_residuals(self):
        from grape_param_estim.model import (
            GrapeRigidBodyModel,
            replay_segments,
        )
        from grape_param_estim.plots import (
            make_correction_figure,
            make_replay_pose_figure,
            make_replay_trajectory_figure,
            make_segment_residual_figure,
        )

        data = make_analysis(two_segments=True)
        replay = replay_segments(
            data, GrapeRigidBodyModel(), nominal_parameters()
        )

        trajectory = make_replay_trajectory_figure(data, replay)
        pose = make_replay_pose_figure(data, replay)
        correction = make_correction_figure(data, replay)
        residual = make_segment_residual_figure(data, replay)

        self.assertEqual(len(trajectory.data), 2 * data.segment_count)
        self.assertEqual(
            tuple(trajectory.layout.scene.xaxis.range), (-0.12, 0.12)
        )
        self.assertEqual(len(pose.data), 12)
        self.assertEqual(len(correction.data), 6)
        self.assertEqual(len(residual.data), 2 * data.segment_count)

    def test_phase_two_figures_show_ridges_and_posterior_pushforward(self):
        import numpy as np

        from grape_param_estim.model import (
            GrapeRigidBodyModel,
            replay_segments,
        )
        from grape_param_estim.plots import (
            make_body_frame_particle_figure,
            make_estimated_pose_figure,
            make_parameter_ridge_figure,
            make_posterior_trajectory_figure,
            make_transform_particle_figure,
            make_uncertain_transform_time_figure,
        )

        data = make_analysis(two_segments=True)
        replay = replay_segments(
            data, GrapeRigidBodyModel(), nominal_parameters()
        )
        particles = np.asarray(
            (
                (0.7, 0.6, 0.8, 0.4),
                (0.9, 0.8, 1.0, 0.5),
                (1.1, 1.0, 1.2, 0.6),
                (1.3, 1.2, 1.4, 0.7),
                (1.5, 1.4, 1.6, 0.8),
            )
        )
        weights = np.asarray((0.05, 0.15, 0.50, 0.20, 0.10))
        particle_count = particles.shape[0]
        sample_count = data.times.size
        offsets = np.linspace(-0.02, 0.02, particle_count)[:, None, None]
        posterior_position = (
            replay.position[None, :, :]
            + offsets
            * np.ones((particle_count, sample_count, 3))
        )
        posterior_orientation = np.repeat(
            replay.orientation_xyzw[None, :, :],
            particle_count,
            axis=0,
        )
        delta_translation = posterior_position - replay.position[None]
        delta_rotation = np.zeros_like(delta_translation)
        delta_rotation[:, :, 2] = offsets[:, :, 0]

        ridge = make_parameter_ridge_figure(particles, weights)
        trajectory = make_posterior_trajectory_figure(
            data.times,
            data.segment_id,
            data.position,
            replay.position,
            posterior_position,
            weights,
            2,
        )
        pose = make_estimated_pose_figure(
            data.times,
            data.segment_id,
            data.position,
            data.orientation_xyzw,
            replay.position,
            replay.orientation_xyzw,
            posterior_position[2],
            posterior_orientation[2],
        )
        transform_time = make_uncertain_transform_time_figure(
            data.times,
            data.segment_id,
            delta_translation,
            delta_rotation,
            weights,
        )
        transform_particles = make_transform_particle_figure(
            delta_translation[:, -1],
            delta_rotation[:, -1],
            weights,
        )
        frames = make_body_frame_particle_figure(
            data.position[-1],
            data.orientation_xyzw[-1],
            replay.position[-1],
            replay.orientation_xyzw[-1],
            posterior_position[2, -1],
            posterior_orientation[2, -1],
            posterior_position[:, -1],
            posterior_orientation[:, -1],
            weights,
            data.position,
        )

        self.assertEqual(len(ridge.data), 4)
        self.assertGreaterEqual(len(trajectory.data), 4)
        self.assertLessEqual(len(trajectory.data), particle_count + 3)
        trajectory_names = {trace.name for trace in trajectory.data}
        self.assertIn("pre-fit baseline", trajectory_names)
        self.assertIn(
            "estimated nominal (maximum-weight particle)",
            trajectory_names,
        )
        self.assertNotIn("posterior median", trajectory_names)
        self.assertEqual(len(transform_time.data), 10)
        self.assertEqual(len(transform_particles.data), 4)
        self.assertGreater(len(frames.data), 10)
        self.assertEqual(len(pose.data), 18)
        for figure in (
            ridge,
            trajectory,
            pose,
            transform_time,
            transform_particles,
            frames,
        ):
            self.assertTrue(figure.to_json())


if __name__ == "__main__":
    unittest.main()
