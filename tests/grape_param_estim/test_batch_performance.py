import unittest

import numpy as np

from grape_param_estim.batch_performance import measure_final_solution_bags
from grape_param_estim.estimation import solve_fixed_graph_laplace
from tests.grape_param_estim.test_batch_graph_builder import (
    BatchGraphBuilderTests,
)


class BatchPerformanceTests(unittest.TestCase):
    def test_measures_final_sparse_work_without_dense_full_inverse(self):
        helper = BatchGraphBuilderTests()
        prepared = helper._prepared()

        def factory(q, delay, static):
            from dataclasses import replace

            return replace(
                prepared,
                dynamics=replace(prepared.dynamics, q=q),
                fixed_delay=delay,
                initial_parameter_coordinates=static,
            )

        solution = solve_fixed_graph_laplace(
            factory,
            prepared.dynamics.q,
            prepared.fixed_delay,
            prepared.initial_parameter_coordinates,
        )
        measured = measure_final_solution_bags(solution)
        self.assertEqual(len(measured), 1)
        bag = measured[0]
        self.assertEqual(bag.bag_id, "bag-a")
        self.assertEqual(bag.knot_count, 2)
        self.assertEqual(
            bag.factor_count,
            sum(
                value.local_bag_id == "bag-a"
                for value in solution.final_linearization.sparse.factor_provenance
            ),
        )
        self.assertEqual(
            bag.residual_dimension,
            sum(
                factor.residual.size
                for factor, provenance in zip(
                    solution.final_linearization.factors,
                    solution.final_linearization.sparse.factor_provenance,
                )
                if provenance.local_bag_id == "bag-a"
            ),
        )
        self.assertGreater(bag.jacobian_nnz, 0)
        self.assertTrue(
            np.all(
                np.asarray(
                    (
                        bag.assembly_seconds,
                        bag.factorization_seconds,
                        bag.schur_solve_seconds,
                    )
                )
                >= 0.0
            )
        )


if __name__ == "__main__":
    unittest.main()
