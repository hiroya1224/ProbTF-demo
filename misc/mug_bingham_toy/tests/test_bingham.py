"""Deterministic checks; run with ``python -m unittest discover -s tests``."""

import importlib.util
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bingham import (BinghamNLL, integration_nodes, matrix_to_parameters,
                     modal_quaternion, parameters_10_to_9, symmetric_matrix)


def load_core_numpy_integrator():
    """Read the existing integrator only for the independent backend check."""
    path = ROOT.parents[1] / "ros/core/probtf_core/src/probtf/_vendor/bingham_normconst.py"
    spec = importlib.util.spec_from_file_location("reference_bingham_normconst", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hopf_integral(eigenvalues, order=48):
    """Independent S^3 surface quadrature (Gauss--Legendre x two circles)."""
    # q=(sqrt(t)cos u,sqrt(t)sin u,sqrt(1-t)cos v,sqrt(1-t)sin v),
    # surface measure = (1/2) dt du dv, t in [0,1].
    points, weights = np.polynomial.legendre.leggauss(order)
    t = (points + 1) / 2
    angles = np.arange(4 * order) * 2 * np.pi / (4 * order)
    squares = np.stack((np.cos(angles) ** 2, np.sin(angles) ** 2), axis=-1)
    first = squares @ np.asarray(eigenvalues[:2])
    second = squares @ np.asarray(eigenvalues[2:])
    values = np.exp(t[:, None, None] * first[None, :, None]
                    + (1 - t[:, None, None]) * second[None, None, :])
    return 2 * np.pi ** 2 * np.einsum("t,tuv->", weights / 2, values) / angles.size ** 2


class BinghamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loss = BinghamNLL()

    def test_nine_parameter_order_and_trace(self):
        p = torch.arange(1, 10, dtype=torch.float64)
        matrix = symmetric_matrix(p)
        expected = torch.tensor([[1, 2, 3, 4], [2, 5, 6, 7],
                                 [3, 6, 8, 9], [4, 7, 9, -14]], dtype=torch.float64)
        torch.testing.assert_close(matrix, expected)
        torch.testing.assert_close(matrix.trace(), torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(matrix_to_parameters(matrix), p)

    def test_uniform_nll_moment_and_first_gradient(self):
        p = torch.zeros((2, 9), dtype=torch.float64, requires_grad=True)
        q = torch.tensor([[1., 0., 0., 0.], [.5, -.5, .5, -.5]], dtype=torch.float64)
        actual = self.loss(p, q, reduce=False)
        torch.testing.assert_close(actual, torch.full((2,), math.log(2 * math.pi ** 2),
                                                     dtype=torch.float64), atol=5e-9, rtol=0)
        actual.sum().backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
        moment = self.loss.second_moment(p)
        torch.testing.assert_close(moment, torch.eye(4, dtype=torch.float64).expand(2, 4, 4) / 4,
                                   atol=2e-9, rtol=0)
        # The analytic parameter gradient at A=0 has diagonal entries
        # q_3^2-q_i^2 and off-diagonals -2 q_i q_j.
        torch.testing.assert_close(p.grad[0], torch.tensor([-1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                                           dtype=torch.float64), atol=2e-12, rtol=0)
        # Refining numerical integration alone recovers the closed form to
        # float64 precision. At the training default N=200 its log error is
        # 3.57e-9; this finite error is not hidden by replacing the result.
        refined = BinghamNLL(nodes=800)(p, q)
        self.assertAlmostEqual(float(refined), math.log(2 * math.pi ** 2), delta=2e-14)

    def test_gradcheck_distinct_and_repeated_spectra(self):
        q = torch.tensor([[.7, -.1, .2, .4]], dtype=torch.float64)
        matrices = [torch.zeros((4, 4), dtype=torch.float64),
                    torch.diag(torch.tensor([-2., -2., 2., 2.], dtype=torch.float64)),
                    torch.tensor([[-3., .3, -.1, .2], [.3, -1., .2, -.4],
                                  [-.1, .2, 1., .1], [.2, -.4, .1, 3.]], dtype=torch.float64)]
        for matrix in matrices:
            with self.subTest(matrix=matrix):
                p = matrix_to_parameters(matrix)[None].requires_grad_()
                self.assertTrue(torch.autograd.gradcheck(lambda x: self.loss(x, q), (p,),
                                                        eps=1e-6, atol=2e-7, rtol=2e-6))

    def test_antipodal_target_and_scalar_identity_shift_invariance(self):
        p10 = torch.tensor([[1., .2, -.3, .1, -2., .4, -.2, .5, .3, 4.]], dtype=torch.float64)
        shifted = p10.clone()
        shifted[:, [0, 4, 7, 9]] += 17.25
        p = parameters_10_to_9(p10)
        q = torch.tensor([[.7, .2, -.4, .1]], dtype=torch.float64)
        torch.testing.assert_close(p, parameters_10_to_9(shifted), atol=2e-14, rtol=0)
        torch.testing.assert_close(self.loss(p, q), self.loss(p, -q), atol=0, rtol=0)
        matrix = symmetric_matrix(p)
        shifted_matrix = matrix + 17.25 * torch.eye(4, dtype=torch.float64)
        torch.testing.assert_close(self.loss.log_normalizer(shifted_matrix)
                                   - self.loss.log_normalizer(matrix),
                                   torch.tensor([17.25], dtype=torch.float64), atol=2e-13, rtol=0)

    def test_normalizer_against_numpy_core_and_independent_surface_integral(self):
        reference = load_core_numpy_integrator()
        for eigenvalues in ([0., 0., 0., 0.], [-1., -.5, -.2, 0.],
                            [-12., -5., -1., 0.], [-100., -10., -2., 0.]):
            with self.subTest(eigenvalues=eigenvalues):
                matrix = torch.diag(torch.tensor(eigenvalues, dtype=torch.float64))[None]
                actual = float(self.loss.log_normalizer(matrix))
                core_logz = math.log(float(reference.calc_constant(np.asarray(eigenvalues), N=200).item()))
                self.assertAlmostEqual(actual, core_logz, delta=2e-12)
                self.assertAlmostEqual(actual, math.log(hopf_integral(eigenvalues)), delta=2e-8)
                p = matrix_to_parameters(matrix)
                core_diagonal = np.asarray(reference.calc_Dconstant(np.asarray(eigenvalues), N=200))
                core_diagonal = core_diagonal.reshape(4) / math.exp(core_logz)
                np.testing.assert_allclose(self.loss.second_moment(p)[0].diagonal().numpy(),
                                           core_diagonal, atol=2e-12, rtol=2e-12)

    def test_rotated_repeated_eigenspace_has_correct_matrix_gradient(self):
        rotation, _ = torch.linalg.qr(torch.tensor([[1., 2., 0., 1.], [-2., 1., 1., 0.],
                                                  [0., 1., -2., 1.], [1., 0., 1., 2.]],
                                                 dtype=torch.float64))
        matrix = (rotation @ torch.diag(torch.tensor([-4., -4., 4., 4.], dtype=torch.float64))
                  @ rotation.T)[None].requires_grad_()
        self.loss.log_normalizer(matrix).sum().backward()
        p = matrix_to_parameters(matrix.detach())
        torch.testing.assert_close(matrix.grad, self.loss.second_moment(p), atol=2e-13, rtol=2e-13)
        self.assertTrue(bool(torch.isfinite(matrix.grad).all()))

    def test_float32_head_receives_finite_float32_gradient_and_reductions(self):
        p = torch.zeros((3, 9), dtype=torch.float32, requires_grad=True)
        q = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [.5, .5, .5, .5]])
        loss = self.loss(p, q)
        self.assertEqual(loss.dtype, torch.float64)
        loss.backward()
        self.assertEqual(p.grad.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(p.grad).all()))
        torch.testing.assert_close(BinghamNLL(reduction="sum")(p, q),
                                   BinghamNLL(reduction="none")(p, q).sum())

    def test_modal_quaternion(self):
        q = torch.tensor([.8, -.4, .2, .4], dtype=torch.float64)
        q = q / q.norm()
        matrix = -12 * (torch.eye(4, dtype=torch.float64) - q[:, None] * q[None, :])
        mode = modal_quaternion(matrix_to_parameters(matrix))
        self.assertAlmostEqual(float((mode @ q).abs()), 1, delta=2e-14)
        self.assertGreaterEqual(float(mode[mode.abs().argmax()]), 0)

    def test_invalid_shape_and_zero_quaternion_raise(self):
        with self.assertRaisesRegex(ValueError, "dimension 9"):
            symmetric_matrix(torch.zeros(2, 10))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            self.loss(torch.zeros(1, 9), torch.zeros(1, 4))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.loss(torch.zeros(1, 9), torch.zeros(2, 4))
        with self.assertRaisesRegex(ValueError, "32"):
            integration_nodes(16)


if __name__ == "__main__":
    unittest.main()
