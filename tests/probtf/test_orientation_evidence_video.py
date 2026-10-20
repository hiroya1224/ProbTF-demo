"""Check the single-direction densities displayed beside the six filters."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'ros/examples/probtf_orientation_demo/scripts/render_feedback_video.py'
SPEC = importlib.util.spec_from_file_location('orientation_evidence_video', SCRIPT)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def quaternion_wxyz(rotation):
    return rotation.as_quat()[[3, 0, 1, 2]]


class ObservationEvidenceVideoTest(unittest.TestCase):
    def setUp(self):
        self.arrays = dict(
            acceleration_m_s2=np.array([[0.3, -0.2, 9.8], [1.2, 3.0, 8.7], [-2.0, 0.1, 9.0]]),
            magnetic_raw=np.array([[12.0, 3.0, 34.0], [-10.0, 7.0, 21.0], [0.1, -8.0, 12.0]]))
        reference = np.array([0.4, -0.1, 0.7])
        self.summary = dict(
            reference_magnetic_world_unit=(reference / np.linalg.norm(reference)).tolist(),
            acceleration_direction_concentration=20.0,
            magnetic_direction_concentration=20.0)
        self.rotations = [Rotation.identity(), Rotation.from_rotvec([0.3, -0.6, 0.1]),
                          Rotation.from_euler('xyz', [1.0, -0.7, 2.4]),
                          Rotation.from_rotvec([-1.0, 0.0, 1.5])]

    def directions(self):
        return (np.asarray(renderer.comparison.original.SPECIFIC_FORCE_REFERENCE_WORLD),
                np.asarray(self.summary['reference_magnetic_world_unit']))

    def test_quaternion_quadratic_form_is_the_direction_likelihood(self):
        parameters = renderer.single_observation_parameters(self.arrays, self.summary)
        self.assertEqual(parameters.shape, (3, 2, 4, 4))
        for index in range(3):
            for channel, key in enumerate(('acceleration_m_s2', 'magnetic_raw')):
                reference = self.directions()[channel]
                observed = self.arrays[key][index]
                observed = observed / np.linalg.norm(observed)
                for rotation in self.rotations:
                    q = quaternion_wxyz(rotation)
                    expected = 20.0 * (reference @ rotation.apply(observed) - 1.0)
                    self.assertAlmostEqual(float(q @ parameters[index, channel] @ q),
                                           float(expected), places=11)

    def test_one_direction_leaves_a_circle_of_modes(self):
        parameters = renderer.single_observation_parameters(self.arrays, self.summary)
        np.testing.assert_allclose(np.linalg.eigvalsh(parameters),
                                   np.broadcast_to([-40.0, -40.0, 0.0, 0.0], (3, 2, 4)),
                                   rtol=0.0, atol=2e-13)
        for index in range(3):
            for channel, reference in enumerate(self.directions()):
                for rotation in self.rotations:
                    q = quaternion_wxyz(rotation)
                    baseline = q @ parameters[index, channel] @ q
                    for angle in (-np.pi, -0.7, 0.0, 0.4, np.pi):
                        about_reference = Rotation.from_rotvec(angle * reference)
                        transformed = quaternion_wxyz(about_reference * rotation)
                        actual = transformed @ parameters[index, channel] @ transformed
                        self.assertAlmostEqual(float(actual), float(baseline), places=11)

    def test_saved_concentrations_are_used_for_the_two_independent_channels(self):
        original = renderer.single_observation_parameters(self.arrays, self.summary)
        changed_summary = dict(self.summary, acceleration_direction_concentration=7.0,
                               magnetic_direction_concentration=13.0)
        changed = renderer.single_observation_parameters(self.arrays, changed_summary)
        np.testing.assert_allclose(changed[:, 0], original[:, 0] * 7.0 / 20.0,
                                   rtol=0.0, atol=2e-13)
        np.testing.assert_allclose(changed[:, 1], original[:, 1] * 13.0 / 20.0,
                                   rtol=0.0, atol=2e-13)

    def test_magnetic_reference_changes_only_the_magnetic_observation(self):
        original = renderer.single_observation_parameters(self.arrays, self.summary)
        changed = renderer.single_observation_parameters(
            self.arrays, dict(self.summary, reference_magnetic_world_unit=[0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(changed[:, 0], original[:, 0])
        self.assertGreater(np.linalg.norm(changed[:, 1] - original[:, 1]), 1.0)

    def test_history_gyro_prior_and_filter_output_are_not_read(self):
        class OnlyRawDirections(dict):
            def __getitem__(self, key):
                if key not in ('acceleration_m_s2', 'magnetic_raw'):
                    raise AssertionError('Single-observation density read ' + str(key))
                return super().__getitem__(key)

            def get(self, key, default=None):
                return self[key]

        source = OnlyRawDirections(self.arrays)
        for key in ('gyro_rad_s', 'display_parameter', 'carried_parameter',
                    'mode_wxyz', 'prior_parameter', 'history'):
            source[key] = object()
        actual = renderer.single_observation_parameters(source, self.summary)
        expected = renderer.single_observation_parameters(self.arrays, self.summary)
        np.testing.assert_array_equal(actual, expected)

    def test_other_times_do_not_accumulate_into_the_current_observation(self):
        original = renderer.single_observation_parameters(self.arrays, self.summary)
        changed_arrays = {key: value.copy() for key, value in self.arrays.items()}
        changed_arrays['acceleration_m_s2'][:-1] = [9.8, 0.0, 0.0]
        changed_arrays['magnetic_raw'][:-1] = [0.0, -30.0, 1.0]
        changed = renderer.single_observation_parameters(changed_arrays, self.summary)
        np.testing.assert_array_equal(changed[-1], original[-1])
        self.assertGreater(np.linalg.norm(changed[:-1] - original[:-1]), 1.0)
        isolated = renderer.single_observation_parameters(
            {key: value[-1:] for key, value in self.arrays.items()}, self.summary)
        np.testing.assert_array_equal(isolated[0], original[-1])

    def test_only_vector_direction_is_used_and_raw_values_are_preserved(self):
        before = {key: value.copy() for key, value in self.arrays.items()}
        original = renderer.single_observation_parameters(self.arrays, self.summary)
        scaled = renderer.single_observation_parameters(
            {key: value * 5.0 for key, value in self.arrays.items()}, self.summary)
        np.testing.assert_allclose(scaled, original, rtol=0.0, atol=2e-13)
        for key in self.arrays:
            np.testing.assert_array_equal(self.arrays[key], before[key])

    def test_calibrated_video_uses_the_same_raw_gaussian_factors_as_inference(self):
        noise = renderer.static_vector_noise
        calibration = dict(
            acceleration=noise.estimate_isotropic_vector_noise(self.arrays['acceleration_m_s2']),
            magnetic=noise.estimate_isotropic_vector_noise(self.arrays['magnetic_raw']))
        summary = dict(self.summary, vector_noise_calibration=calibration)
        summary.pop('acceleration_direction_concentration')
        summary.pop('magnetic_direction_concentration')
        arrays = dict(self.arrays)
        arrays['acceleration_direction_concentration'] = noise.concentration_from_vector_noise(
            arrays['acceleration_m_s2'], calibration['acceleration'])
        arrays['magnetic_direction_concentration'] = noise.concentration_from_vector_noise(
            arrays['magnetic_raw'], calibration['magnetic'])
        expected = renderer.comparison.direction_factors(
            arrays['acceleration_m_s2'], arrays['magnetic_raw'],
            summary['reference_magnetic_world_unit'],
            arrays['acceleration_direction_concentration'], arrays['magnetic_direction_concentration'])
        np.testing.assert_array_equal(renderer.single_observation_parameters(arrays, summary),
                                      np.stack(expected, axis=1))
        arrays['acceleration_direction_concentration'] = arrays['acceleration_direction_concentration']*2
        with self.assertRaisesRegex(ValueError, 'differs from its noise calibration'):
            renderer.single_observation_parameters(arrays, summary)

    def test_synthetic_video_uses_assumed_noise_without_reading_generated_noise(self):
        noise = renderer.static_vector_noise
        assumed = dict(
            acceleration=noise.estimate_isotropic_vector_noise(self.arrays['acceleration_m_s2']),
            magnetic=noise.estimate_isotropic_vector_noise(self.arrays['magnetic_raw']))
        expected = renderer.single_observation_parameters(self.arrays,
            dict(self.summary, vector_noise_calibration=assumed))
        actual = renderer.single_observation_parameters(self.arrays,
            dict(self.summary, assumed_sensor_noise=assumed, generation_sensor_noise=object(),
                 vector_noise_calibration=object()))
        np.testing.assert_array_equal(actual, expected)

    def test_saved_sensor_vectors_are_measured_unit_vectors_without_attitude_truth(self):
        actual = renderer.observed_unit_vectors(self.arrays)
        for channel, key in enumerate(('acceleration_m_s2', 'magnetic_raw')):
            values = self.arrays[key]
            np.testing.assert_allclose(actual[:, channel], values/np.linalg.norm(values, axis=1)[:, None])
        changed = dict(self.arrays, mocap_wxyz=object(), truth_wxyz=object(), mode_wxyz=object())
        np.testing.assert_array_equal(renderer.observed_unit_vectors(changed), actual)

    def test_native_observation_modes_map_measured_vectors_to_fixed_world_references(self):
        parameters = renderer.single_observation_parameters(self.arrays, self.summary)
        vectors = renderer.observed_unit_vectors(self.arrays)
        references = renderer.observation_reference_vectors_world(self.summary)
        for index in range(3):
            for channel, reference in enumerate(self.directions()):
                rotation, _ = Rotation.align_vectors(reference[None], vectors[index, channel][None])
                q = quaternion_wxyz(rotation)
                self.assertAlmostEqual(float(q @ parameters[index, channel] @ q), 0.0, places=11)
                np.testing.assert_allclose(rotation.apply(vectors[index, channel]), references[channel], atol=1e-14)
        np.testing.assert_allclose(references, self.directions(), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
