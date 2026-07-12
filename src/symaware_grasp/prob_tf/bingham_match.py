"""Compatibility exports for the shared ProbTF Bingham implementation."""

from probtf.bingham import (
    match_bingham_to_second_moment,
    quaternion_product_second_moment,
)

__all__ = [
    "match_bingham_to_second_moment",
    "quaternion_product_second_moment",
]
