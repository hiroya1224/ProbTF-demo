import unittest

import numpy as np

from grape_param_estim.posterior.representatives import (
    POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT,
    RIDGE_QUANTILES,
    ROLE_RECORD_KEYS,
    select_posterior_representatives,
)


class PosteriorRepresentativeSelectionTests(unittest.TestCase):
    @staticmethod
    def _inputs(count=10):
        coordinate = np.zeros((count, 18))
        coordinate[:, 0] = np.arange(count, dtype=float)
        coordinate[:, 1] = np.linspace(-1.0, 1.0, count)
        return {
            "sample_id": tuple("sample-{:02d}".format(index) for index in range(count)),
            "chain_id": tuple(
                "chain-{}".format(index % 2) for index in range(count)
            ),
            "draw_index": np.arange(count, dtype=np.int64) // 2,
            "static_coordinate": coordinate,
            "delay": np.linspace(0.002, 0.018, count),
            "log_posterior": -np.arange(count, dtype=float),
            "source_mode_id": tuple("mode-a" for _index in range(count)),
            "prior_mean_coordinate": np.zeros(18),
            "prior_covariance": np.eye(18),
            "delay_bounds_seconds": (0.0, 0.02),
            "delay_scale_seconds": 0.001,
            "exact_ridge_direction": -np.eye(18)[0],
        }

    def test_primary_mode_and_ridge_roles_have_canonical_ties_and_sign(self):
        inputs = self._inputs()
        inputs["log_posterior"][[2, 7]] = 10.0
        inputs["source_mode_id"] = tuple(
            "mode-a" if index < 6 else "mode-b" for index in range(10)
        )
        result = select_posterior_representatives(**inputs)

        self.assertEqual(result.selected_sample_ids[0], "sample-02")
        by_id = {role.role_id: role for role in result.role_records}
        self.assertEqual(by_id["primary"].sample_id, "sample-02")
        self.assertEqual(by_id["mode:mode-a"].sample_id, "sample-02")
        self.assertEqual(by_id["mode:mode-b"].sample_id, "sample-07")
        expected = {
            (1, 10): "sample-01",
            (1, 2): "sample-05",
            (9, 10): "sample-08",
        }
        for quantile in RIDGE_QUANTILES:
            role = by_id[
                "ridge_quantile:{}/{}".format(*quantile)
            ]
            self.assertEqual(role.sample_id, expected[quantile])
        self.assertLessEqual(
            len(result.selected_sample_ids),
            POSTERIOR_REPRESENTATIVE_MAXIMUM_SAMPLE_COUNT,
        )

    def test_one_draw_retains_all_overlapping_roles_once_in_union(self):
        result = select_posterior_representatives(**self._inputs(count=1))

        self.assertEqual(result.selected_sample_ids, ("sample-00",))
        self.assertEqual(len(result.role_records), 6)
        self.assertTrue(
            all(role.sample_id == "sample-00" for role in result.role_records)
        )
        payload = result.manifest_payload(("bag-a", "bag-b"))
        self.assertEqual(payload["maximum_sample_count"], 8)
        self.assertEqual(payload["selected_bag_ids"], ["bag-a", "bag-b"])
        self.assertTrue(
            all(set(role) == set(ROLE_RECORD_KEYS) for role in payload["role_records"])
        )

    def test_pid_medoids_and_role_union_are_invariant_to_input_order(self):
        inputs = self._inputs(count=20)
        first = select_posterior_representatives(**inputs)
        permutation = np.asarray(
            (11, 2, 17, 0, 9, 4, 19, 5, 12, 1, 15, 6, 18, 3, 8, 14, 7, 13, 10, 16)
        )
        reordered = dict(inputs)
        for name in (
            "sample_id",
            "chain_id",
            "draw_index",
            "static_coordinate",
            "delay",
            "log_posterior",
            "source_mode_id",
        ):
            value = np.asarray(inputs[name])
            reordered[name] = value[permutation]
        second = select_posterior_representatives(**reordered)

        self.assertEqual(first, second)
        pid_roles = tuple(
            role
            for role in first.role_records
            if role.role_class == "pid_stratified_medoid"
        )
        self.assertLessEqual(len(pid_roles), 4)
        self.assertLessEqual(len(first.selected_sample_ids), 8)

    def test_too_many_mandatory_mode_representatives_fail_closed(self):
        inputs = self._inputs(count=9)
        inputs["source_mode_id"] = tuple(
            "mode-{}".format(index) for index in range(9)
        )
        with self.assertRaisesRegex(ValueError, "mandatory roles"):
            select_posterior_representatives(**inputs)


if __name__ == "__main__":
    unittest.main()
