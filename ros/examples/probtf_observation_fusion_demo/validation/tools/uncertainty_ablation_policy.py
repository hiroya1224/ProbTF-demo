"""Current execution policy, separate from retained historical evidence.

The 2026-10-18 truth-assisted experiments are historical. The user's
2026-10-19 instruction revoked further truth input to estimation. A baseline
run and physical-side rendering/posthoc scoring remain available; no oracle
can be enabled by an argument, environment variable, or direct helper call.
"""


class TruthInjectionRevoked(PermissionError):
    """An operation would execute a withdrawn truth-assisted experiment."""


def require_baseline(intervention):
    """Reject before workers, directories, or inference patches are created."""
    if intervention != 'baseline':
        reject_truth_injection(str(intervention))


def reject_truth_injection(operation):
    raise TruthInjectionRevoked(
        'Truth injection authorization was revoked by the user on 2026-10-19: '
        + str(operation) + '. Only baseline execution and physical-side '
        'rendering or posthoc scoring are permitted; historical evidence is retained.')
