"""Nine-parameter quaternion Bingham NLL for the isolated mug experiment.

The network emits ``[A00,A01,A02,A03,A11,A12,A13,A22,A23]`` and
``A33 = -A00-A11-A22``. Quaternions are scalar-first ``[w,x,y,z]``.
The density is ``exp(q.T @ A @ q) / Z(A)`` with respect to surface area
on S^3; consequently the uniform NLL is log(2*pi**2).

Normalizer provenance: the Torch continuous-Euler integration formula is
adapted from /home/leus/posecnn_ws/experiments/mug_finetune/mugft/loss.py,
SHA256 8c70ad42389c84489026d1c535648d7ec9e185deeede6630951b24d6cef479dc.
That implementation follows Chen--Tanaka, arXiv:2004.14660 and Toyota's
BinghamNLL reference implementation. This file is self-contained and does
not import or modify that experiment or ProbTF core. The original formula
is retained; the head uses a trace-zero nine-dimensional gauge and the
backward pass supplies d log Z / d A = E[q q.T] directly. No derivative of
an eigenvector is required, including at the uniform initialization.

All normalization and likelihood arithmetic is float64 on the input device.
This is deterministic finite numerical quadrature, not an exact normalizer.
There is no Monte Carlo integration, parameter bound, covariance floor, or
replacement of an invalid normalizer by a clipped value. Only first-order
parameter gradients are implemented.
"""

import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable


PARAMETER_ORDER = ("A00", "A01", "A02", "A03", "A11", "A12", "A13", "A22", "A23")


def symmetric_matrix(parameters):
    """Map tensors with trailing dimension 9 to symmetric trace-zero 4x4 A."""
    if parameters.ndim < 1 or parameters.shape[-1] != 9:
        raise ValueError("Bingham parameters must have trailing dimension 9")
    a, b, c, d, e, f, g, h, i = parameters.unbind(-1)
    return torch.stack((a, b, c, d, b, e, f, g, c, f, h, i,
                        d, g, i, -a - e - h), dim=-1).reshape(*parameters.shape[:-1], 4, 4)


def matrix_to_parameters(matrix):
    """Convert a symmetric 4x4 A into its equivalent trace-zero 9D gauge.

    Subtracting trace(A)/4 times I preserves the Bingham density. The input
    is expected to be symmetric; asymmetry is an error rather than a new law.
    """
    if matrix.ndim < 2 or matrix.shape[-2:] != (4, 4):
        raise ValueError("Bingham matrices must have trailing shape (4, 4)")
    if not torch.allclose(matrix, matrix.transpose(-2, -1)):
        raise ValueError("Bingham matrices must be symmetric")
    eye = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
    centered = matrix - matrix.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
    indices = torch.triu_indices(4, 4, device=matrix.device)
    return centered[..., indices[0][:-1], indices[1][:-1]]


def parameters_10_to_9(parameters):
    """Adapt the original row-major upper-triangular 10D head without loss."""
    if parameters.ndim < 1 or parameters.shape[-1] != 10:
        raise ValueError("Original Bingham parameters must have trailing dimension 10")
    indices = torch.triu_indices(4, 4, device=parameters.device)
    matrix = parameters.new_zeros((*parameters.shape[:-1], 4, 4))
    matrix[..., indices[0], indices[1]] = parameters
    matrix[..., indices[1], indices[0]] = parameters
    return matrix_to_parameters(matrix)


def integration_nodes(count=200, device="cpu"):
    """Return the original Chen--Tanaka continuous-Euler nodes in float64."""
    if not isinstance(count, int) or count < 32:
        raise ValueError("At least 32 positive quadrature nodes are required")
    r, omega_d = 2.5, 0.5
    c = 15 * math.pi / (r * r * (1 + r) * omega_d)
    h = math.sqrt(math.pi * c * (1 + r) / (omega_d * count))
    p = math.sqrt(count * h / omega_d)
    q = math.sqrt(omega_d * count * h / 4)
    frequencies = torch.arange(-count - 1, count + 1, dtype=torch.float64, device=device) * h
    taper = 0.5 * torch.special.erfc(frequencies.abs() / p - q)
    prefactor = math.pi * math.exp(c) * h
    return torch.stack((frequencies, prefactor * taper * frequencies.cos(),
                        prefactor * taper * frequencies.sin()), dim=1).contiguous(), c


def torch_normalizer_and_moments(shifted_eigenvalues, nodes, c):
    """Return log Z and its four diagonal moments for max-shifted spectra.

    Shape is [batch,4]. The returned moments are quadrature derivatives
    divided by the same normalizer, with no clipping or moment adjustment.
    """
    lambdas = shifted_eigenvalues.to(dtype=torch.float64)
    if lambdas.ndim != 2 or lambdas.shape[-1] != 4:
        raise ValueError("Eigenvalues must have shape [batch,4]")
    nodes = nodes.to(device=lambdas.device, dtype=torch.float64)
    s = c - lambdas[..., None] + 1j * nodes[:, 0]
    integrand = s.rsqrt().prod(dim=1)
    weight = torch.complex(nodes[:, 1], nodes[:, 2])
    value = (integrand * weight).real.sum(dim=-1)
    derivative = ((0.5 / s) * integrand[:, None, :] * weight).real.sum(dim=-1)
    if not bool(torch.isfinite(value).all() & (value > 0).all()
                & torch.isfinite(derivative).all()):
        raise FloatingPointError("Chen--Tanaka Bingham normalizer is nonpositive or nonfinite")
    return value.log(), derivative / value[:, None]


def _matrix_statistics(matrix, nodes, c):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    maximum = eigenvalues[:, -1]
    logz, diagonal_moments = torch_normalizer_and_moments(
        eigenvalues - maximum[:, None], nodes, c)
    moment = (eigenvectors * diagonal_moments[:, None, :]) @ eigenvectors.transpose(-2, -1)
    return logz + maximum, moment


class _MatrixLogNormalizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, nodes, c):
        logz, moment = _matrix_statistics(matrix, nodes, c)
        ctx.save_for_backward(moment)
        return logz

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        # At an eigenvalue multiplicity the moments within its eigenspace
        # coincide, so this matrix gradient does not depend on its basis.
        return gradient[:, None, None] * ctx.saved_tensors[0], None, None


@torch.no_grad()
def modal_quaternion(parameters):
    """Return a unit modal wxyz quaternion; a repeated maximum has no unique mode.

    Pick the sign whose largest-magnitude component is nonnegative. Either
    sign describes the same rotation and has exactly the same likelihood.
    """
    matrix = symmetric_matrix(parameters.to(dtype=torch.float64))
    _, vectors = torch.linalg.eigh(matrix)
    mode = vectors[..., :, -1]
    pivot = mode.abs().argmax(dim=-1, keepdim=True)
    sign = torch.where(mode.gather(-1, pivot) < 0, -1.0, 1.0)
    return mode * sign


class BinghamNLL(nn.Module):
    """Bingham negative log likelihood for a nine-output network.

    ``forward(parameters[B,9], targets[B,4], reduce=True)`` returns a
    float64 scalar mean. ``reduce=False`` returns per-image NLLs. Set the
    constructor's ``reduction`` to ``'sum'`` or ``'none'`` if preferred.
    Targets are normalized without an epsilon floor; a zero quaternion is
    invalid. Labels and modes must use the same wxyz camera/object convention.
    """

    def __init__(self, device="cpu", backend="torch", nodes=200, reduction="mean"):
        super().__init__()
        if backend != "torch":
            raise ValueError("This isolated loss supports only the Torch backend")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        weights, self.c = integration_nodes(nodes, device)
        self.register_buffer("nodes", weights)
        self.reduction = reduction

    def log_normalizer(self, matrix):
        """Differentiable log Z(A) for arbitrary symmetric [batch,4,4] A."""
        if matrix.ndim != 3 or matrix.shape[1:] != (4, 4):
            raise ValueError("Bingham matrices must have shape [batch,4,4]")
        return _MatrixLogNormalizer.apply(matrix.to(dtype=torch.float64), self.nodes, self.c)

    @torch.no_grad()
    def second_moment(self, parameters):
        """Return E[q q.T] from the same finite quadrature used by the loss."""
        matrix = symmetric_matrix(parameters.to(dtype=torch.float64))
        shape = matrix.shape
        _, moment = _matrix_statistics(matrix.reshape(-1, 4, 4), self.nodes, self.c)
        return moment.reshape(shape)

    def forward(self, parameters, targets, reduce=None):
        if parameters.ndim != 2 or parameters.shape[1] != 9:
            raise ValueError("Bingham parameters must have shape [batch,9]")
        if targets.ndim != 2 or targets.shape != (parameters.shape[0], 4):
            raise ValueError("Target quaternions must have shape [batch,4]")
        target = targets.to(device=parameters.device, dtype=torch.float64)
        norm = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
        if not bool(torch.isfinite(norm).all() & (norm > 0).all()):
            raise ValueError("Target quaternions must be finite and nonzero")
        target = target / norm
        matrix = symmetric_matrix(parameters.to(dtype=torch.float64))
        loss = self.log_normalizer(matrix) - torch.einsum("bi,bij,bj->b", target, matrix, target)
        reduction = self.reduction if reduce is None else ("mean" if reduce else "none")
        if reduction == "none":
            return loss
        return loss.sum() if reduction == "sum" else loss.mean()
