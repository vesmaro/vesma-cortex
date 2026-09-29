"""Internal selection — stratified 5-fold × 20 fixed seeds (ADR 0001 V1).

Metrics: balanced accuracy + Brier, mean±std across seeds. Selection rule:
N beats D only when its mean exceeds D's by MORE than 1 std of the
repeat-spread; otherwise D wins (at doubt → D). Hyperparameter search and
early stopping touch TRAIN ONLY (internal validation); holdout is absent
from this module's reach by construction (cortex.data.holdout owns it).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = ["CV_FOLDS", "CV_SEEDS", "CvReport", "SelectionVerdict", "run_cv", "select_candidate"]

#: Frozen protocol constants (ADR 0001 V1: stratified 5-fold × 20 seeds).
CV_FOLDS: Final[int] = 5
CV_SEEDS: Final[tuple[int, ...]] = tuple(range(1, 21))  # 1..20, fixed

#: Selection margin: N must beat D by > 1 std of the repeat spread.
SELECTION_MARGIN_STD: Final[float] = 1.0


@dataclass(frozen=True)
class CvReport:
    """Per-candidate CV outcome over the frozen seeds."""

    candidate: str  # "d-boost" | "n-head"
    config_name: str
    balanced_accuracy_mean: float
    balanced_accuracy_std: float
    brier_mean: float
    brier_std: float


@dataclass(frozen=True)
class SelectionVerdict:
    """The internal pick that rides to A5 — at doubt, D (ADR 0001 V1)."""

    winner: str  # "d-boost" | "n-head"
    margin_stds: float
    d_report: CvReport
    n_report: CvReport | None  # None when N was not runnable (pre-addendum)


def run_cv(
    train_fn: Callable[[np.ndarray, np.ndarray, int], Callable[[np.ndarray], np.ndarray]],
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, float]:
    """Run the frozen protocol: 5-fold stratified × CV_SEEDS.

    ``train_fn(features_train, labels_train, seed) → predict_fn`` — the
    candidate injects its frozen grid point; this function owns folds,
    seeds and metrics. Returns (balanced_accuracy_mean, …_std,
    brier_mean, …_std).
    """
    raise NotImplementedError("A3b: CV loop")


def select_candidate(d_report: CvReport, n_report: CvReport | None) -> SelectionVerdict:
    """Apply the frozen rule: N wins only if
    n.balanced_accuracy_mean − d.balanced_accuracy_mean > SELECTION_MARGIN_STD ×
    repeat-spread std; otherwise D."""
    raise NotImplementedError("A3b: selection rule (one comparison, lands with CV)")
