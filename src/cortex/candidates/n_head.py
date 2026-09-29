"""Candidate N — small MLP head over frozen mnema-embed-v1 vectors.

ADR 0001 V1: head ≤ ~0.5M parameters over [a, b, |a−b|, a⊙b] (4×384)
plus the scalar feature block; corruption pretrain (cortex.pretrain),
NO own encoder, NO external pretrain weights, cosine distillation
forbidden. Prod wiring is GATED on the canon ADR 0004 addendum (vectors
in CanonState); the eval runner (A5) reads vectors itself and is not
blocked. torch — train extra only (CPU; XPU walkthrough per charter §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from cortex.features.pair import FEATURE_NAMES, FeatureVector

__all__ = ["NGridConfig", "GRID_N", "NHeadModel", "MAX_HEAD_PARAMS"]

#: Parameter budget (ADR 0001 V1: "голова ≤ ~0.5M параметров").
MAX_HEAD_PARAMS: Final[int] = 500_000

#: Vector block layout fed to the head: [a, b, |a−b|, a⊙b] → 4 × 384.
VECTOR_BLOCK_ROWS: Final[int] = 4
VECTOR_DIM: Final[int] = 384


@dataclass(frozen=True)
class NGridConfig:
    """One frozen grid point. Hidden layers 1–2, widths 32–128."""

    name: str
    hidden_dims: tuple[int, ...]
    dropout: float
    learning_rate: float
    pretrain: bool  # ablation axis: with/without corruption pretrain
    seed: int


#: Frozen grid, ≤ 8 configurations (ADR 0001 V1). Exact values land with
#: A3b; the cardinality cap and the param budget are contract now.
GRID_N: Final[tuple[NGridConfig, ...]] = ()


class NHeadModel:
    """MLP head over frozen store vectors + scalar features.

    Shares the three-method candidate surface with DBoostModel. Vectors
    enter as the (4, 384) block; scalars as FEATURE_NAMES block — the
    exported graph takes BOTH inputs (inference-v1.md §4, variant N).
    """

    feature_names: tuple[str, ...] = FEATURE_NAMES

    def train(
        self,
        vectors: Sequence[FeatureVector],
        labels: Sequence[int],
        config: NGridConfig,
        *,
        vector_blocks: Sequence[np.ndarray],
    ) -> None:
        """Supervised fit; MUST override any pretrain initialization when
        pretrain weights are present (ADR 0001 V1: owner labels win)."""
        raise NotImplementedError("A3b: N supervised training")

    def pretrain(
        self,
        pairs: Sequence[tuple[FeatureVector, np.ndarray]],
        config: NGridConfig,
    ) -> None:
        """Corruption-based pretrain (cortex.pretrain.corruption pairs).
        Distillation of the cosine is FORBIDDEN (ADR 0001 V1)."""
        raise NotImplementedError("A3b: N corruption pretrain")

    def predict_proba(self, vectors: Sequence[FeatureVector], *, vector_blocks: Sequence[np.ndarray]) -> np.ndarray:
        """P(duplicate) per pair, shape (n,), values in [0, 1]."""
        raise NotImplementedError("A3b: N inference")

    def export_onnx(self, path: Path) -> Path:
        """torch.onnx export: inputs scalars float32[K] + vectors
        float32[4, 384] → probability[1]; raises RuntimeError when the
        param budget (MAX_HEAD_PARAMS) or the artifact size gate is
        exceeded."""
        raise NotImplementedError("A3b: N ONNX export")
