"""Compatibility location for deterministic forward-rollout caching.

The cache implementation is shared with inference so existing imports from
the redesign's target ``forward.cache`` location and the inference package
refer to the same key type and cache entries.
"""

from grape_param_estim.inference.cache import RolloutCache, RolloutCacheKey

__all__ = ["RolloutCache", "RolloutCacheKey"]
