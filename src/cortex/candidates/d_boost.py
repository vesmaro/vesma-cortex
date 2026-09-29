"""Candidate D — gradient boosting over the frozen pair features.

ADR 0001 V1: D is an EQUAL ADOPT-candidate and the internal bar, not a
control afterthought. LightGBM here (train env only — never a runtime dep
of the engine, addendum P2). Grids ≤ 8 configurations, frozen in code at
A3 (this file); selection protocol lives in cortex.select.cv.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from cortex.features.pair import FEATURE_NAMES, FeatureVector

__all__ = ["DGridConfig", "GRID_D", "DBoostModel"]


@dataclass(frozen=True)
class DGridConfig:
    """One frozen grid point (hyperparameters + feature variant)."""

    name: str
    num_leaves: int
    learning_rate: float
    min_data_in_leaf: int
    field_cosines: bool  # ablation axis: with/without field cosines
    seed: int


#: Frozen grid, ≤ 8 configurations (ADR 0001 V1: "гриды ≤ 8 конфигураций,
#: замораживаются в коде A3"). Exact values land with A3b training slice;
#: the CARDINALITY gate (≤ 8) is contract now (tests/test_skeleton.py).
GRID_D: Final[tuple[DGridConfig, ...]] = ()


class DBoostModel:
    """LightGBM classifier over FEATURE_NAMES — train/predict/export API.

    The three-method surface is the candidate contract shared with
    NHeadModel: cv_select drives both through it, export-artifact writes
    the winner as a single ONNX (≤ 5 MB, metadata_props per
    cortex.artifacts).
    """

    feature_names: tuple[str, ...] = FEATURE_NAMES

    def train(self, vectors: Sequence[FeatureVector], labels: Sequence[int], config: DGridConfig) -> None:
        """Fit on labeled train pairs (labels ∈ {0, 1}); deterministic
        under config.seed. Train-env only."""
        raise NotImplementedError("A3b: D training")

    def predict_proba(self, vectors: Sequence[FeatureVector]) -> np.ndarray:
        """P(duplicate) per pair, float64 array shape (n,), values in [0, 1]."""
        raise NotImplementedError("A3b: D inference")

    def export_onnx(self, path: Path) -> Path:
        """Export the fitted model as a single ONNX (skl2onnx or native
        LightGBM export), opset 15, input float32[K] → probability[1].
        Raises RuntimeError if unfitted or the file would exceed
        cortex.artifacts.MAX_ARTIFACT_BYTES."""
        raise NotImplementedError("A3b: D ONNX export")
