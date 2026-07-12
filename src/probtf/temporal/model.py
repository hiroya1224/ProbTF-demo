from dataclasses import dataclass

from probtf.distributions import TransformDistributionStamped
from probtf.temporal.policy import TemporalPolicy


@dataclass(frozen=True)
class ResolvedEdgeRecord:
    record: TransformDistributionStamped
    requested_stamp: float
    sample_stamp: float
    policy: TemporalPolicy
    diagnostic: str = ""

    @property
    def time_offset(self):
        return self.sample_stamp - self.requested_stamp

