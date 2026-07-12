from dataclasses import dataclass
from typing import Tuple

from probtf.graph import PathExpression
from probtf.kernels.base import TransformKernelExpression


@dataclass(frozen=True)
class IdentityTransformKernel(TransformKernelExpression):
    frame_id: str


@dataclass(frozen=True)
class ComposedTransformKernel(TransformKernelExpression):
    """Kernels in source-to-target application order."""

    kernels: Tuple[TransformKernelExpression, ...]
    path: PathExpression

    def __post_init__(self):
        kernels = tuple(self.kernels)
        if any(not isinstance(kernel, TransformKernelExpression) for kernel in kernels):
            raise TypeError("kernels must contain TransformKernelExpression objects.")
        if not isinstance(self.path, PathExpression):
            raise TypeError("path must be a PathExpression.")
        if len(kernels) != len(self.path.edge_views):
            raise ValueError("A composed kernel must match its path length.")
        object.__setattr__(self, "kernels", kernels)

    def latent_dependency_ids(self):
        identifiers = set()
        for kernel in self.kernels:
            identifiers.update(kernel.latent_dependency_ids())
        return frozenset(identifiers)

    def repeated_dependency_ids(self):
        seen = set()
        repeated = []
        for kernel in self.kernels:
            for identifier in sorted(kernel.latent_dependency_ids()):
                if identifier in seen and identifier not in repeated:
                    repeated.append(identifier)
                seen.add(identifier)
        return tuple(repeated)

