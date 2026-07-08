from dataclasses import dataclass
from typing import Optional


@dataclass
class DelayEstimate:
    delay_s: float
    sample_count: int


class CommandDelayRLS:
    def __init__(self, initial_delay_s: float = 0.0) -> None:
        self.delay_s = float(initial_delay_s)
        self.sample_count = 0

    def update(self, reference_stamp: Optional[float], measurement_stamp: Optional[float]) -> DelayEstimate:
        if reference_stamp is not None and measurement_stamp is not None:
            measured_delay = max(0.0, float(measurement_stamp) - float(reference_stamp))
            alpha = 1.0 / float(self.sample_count + 1)
            self.delay_s = (1.0 - alpha) * self.delay_s + alpha * measured_delay
            self.sample_count += 1
        return DelayEstimate(delay_s=self.delay_s, sample_count=self.sample_count)
