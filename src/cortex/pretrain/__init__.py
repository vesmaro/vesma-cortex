"""Pretrain package — corruption pairs for candidate N (A3b)."""

from cortex.pretrain.corruption import (
    HARD_NEGATIVE_TRANSFORMS,
    WEAK_POSITIVE_TRANSFORMS,
    PretrainPair,
    Transformation,
    generate_pretrain_pairs,
)

__all__ = [
    "HARD_NEGATIVE_TRANSFORMS",
    "WEAK_POSITIVE_TRANSFORMS",
    "PretrainPair",
    "Transformation",
    "generate_pretrain_pairs",
]
