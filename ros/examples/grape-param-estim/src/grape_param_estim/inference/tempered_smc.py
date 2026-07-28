"""Batch static-parameter SMC driver for multiple episode rollouts."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any, Callable, Optional, Tuple

import numpy as np

from grape_param_estim._legacy_inference import (
    SmcPosterior,
    SmcStage,
    TemperedResampleMoveSmc,
    TemperedSmcConfig,
)
from grape_param_estim.inference.posterior import PlantPosterior
from grape_param_estim.plant.parameters import PlantHypothesis


class BatchPlantInference:
    """Connect an episode-wide likelihood to the standard tempered SMC."""

    def __init__(
        self,
        prior: Any,
        transform: Any,
        hypothesis_builder: Callable[[np.ndarray], PlantHypothesis],
        log_likelihood: Callable[[np.ndarray], np.ndarray],
        config: TemperedSmcConfig = TemperedSmcConfig(),
        chain_count: int = 1,
        chain_worker_count: int = 1,
    ) -> None:
        if not callable(hypothesis_builder) or not callable(log_likelihood):
            raise TypeError("hypothesis_builder and log_likelihood must be callable")
        chains = int(chain_count)
        if chains < 1:
            raise ValueError("chain_count must be positive")
        workers = int(chain_worker_count)
        if workers < 1:
            raise ValueError("chain_worker_count must be positive")
        self.prior = prior
        self.transform = transform
        self.hypothesis_builder = hypothesis_builder
        self.log_likelihood = log_likelihood
        self.config = config
        self.chain_count = chains
        self.chain_worker_count = workers

    def _run_chain(self, chain: int) -> SmcPosterior:
        config = replace(
            self.config, seed=int(self.config.seed) + int(chain)
        )
        smc = TemperedResampleMoveSmc(
            self.prior, self.transform, config
        )
        return smc.run(self.log_likelihood)

    def run_chains(self) -> Tuple[SmcPosterior, ...]:
        # A persistent/batch C++ controller is preferred by the architecture.
        # Threads keep closure-backed likelihoods and the shared persistent
        # oracle in one process. executor.map preserves seed/chain ordering.
        worker_count = min(
            self.chain_count, self.chain_worker_count
        )
        if worker_count == 1:
            return tuple(
                self._run_chain(chain)
                for chain in range(self.chain_count)
            )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="grape-smc-chain",
        ) as executor:
            return tuple(
                executor.map(
                    self._run_chain, range(self.chain_count)
                )
            )

    def run(
        self,
        *,
        model_id: str,
        prior_id: str,
        likelihood_id: str,
        controller_snapshot_id: str,
        credible_probability: float = 0.95,
        provenance: Optional[dict] = None
    ) -> PlantPosterior:
        chains = self.run_chains()
        # Preserve each chain's empirical law.  Combining normalized chains
        # with equal chain mass is deterministic and retains multimodality.
        hypotheses = []
        weights = []
        likelihood = []
        chain_mass = 1.0 / len(chains)
        for chain in chains:
            for vector in chain.particles:
                hypotheses.append(self.hypothesis_builder(vector))
            weights.append(chain.weights * chain_mass)
            likelihood.append(chain.log_likelihood)
        normalized = np.concatenate(weights)
        normalized /= np.sum(normalized)
        return PlantPosterior.from_arrays(
            particles=hypotheses,
            weights=normalized,
            log_likelihood=np.concatenate(likelihood),
            model_id=model_id,
            prior_id=prior_id,
            likelihood_id=likelihood_id,
            controller_snapshot_id=controller_snapshot_id,
            credible_probability=credible_probability,
            provenance={} if provenance is None else provenance,
        )


__all__ = [
    "BatchPlantInference",
    "SmcPosterior",
    "SmcStage",
    "TemperedResampleMoveSmc",
    "TemperedSmcConfig",
]
