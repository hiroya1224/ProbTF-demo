"""Lazy compatibility view of established controller-support diagnostics."""

from typing import Any


_NAMES = frozenset(
    (
        "EXTRAPOLATIVE",
        "SUPPORTED",
        "UNSUPPORTED",
        "SupportDiagnostics",
        "SupportReference",
        "classify_support",
    )
)


def __getattr__(name: str) -> Any:
    if name not in _NAMES:
        raise AttributeError(name)
    from grape_param_estim import counterfactual

    return getattr(counterfactual, name)


def __dir__():
    return sorted(set(globals()) | _NAMES)


__all__ = sorted(_NAMES)
