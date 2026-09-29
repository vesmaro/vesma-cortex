"""Internal selection — stratified 5-fold × 20 fixed seeds (ADR 0001 V1).

Metrics: balanced accuracy + Brier, mean±std across seeds. Selection rule:
N beats D only when its mean exceeds D's by MORE than 1 std of the
repeat-spread; otherwise D wins (at doubt → D). Hyperparameter search and
early stopping touch TRAIN ONLY (internal validation); holdout is absent
from this module's reach by construction (cortex.data.holdout owns it).

A3b frozen semantics of "превосходство > 1 std repeat-spread"
(ADR 0001 V1, task-level clarification):

- One repeat = one protocol seed. Per seed, every sample is predicted
  exactly once (out-of-fold across the 5 stratified folds) and the two
  metrics are computed over the full train set — so the per-seed values
  are comparable repeats and their std IS the repeat-spread.
- The repeat-spread of a COMPARISON (no per-seed pairing is available at
  CvReport granularity) is bounded below by the larger of the two
  candidates' stds; :func:`select_candidate` uses that conservative lower
  bound ``max(std_D, std_N)`` per metric.
- N wins ONLY IF it is superior on BOTH metrics simultaneously:
  ``n.ba_mean − d.ba_mean > 1 × spread_ba`` AND
  ``d.brier_mean − n.brier_mean > 1 × spread_brier`` (Brier: lower is
  better). Anything else — including a tie, a win on one metric only, or
  a missing N report — selects D (at doubt → D, ADR 0001 V1).
- Zero spread (both stds 0.0) degenerates the margin to "any strict
  improvement"; ``margin_stds`` is then ±inf as appropriate.
"""

from __future__ import annotations

import math
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
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss
    from sklearn.model_selection import StratifiedKFold

    matrix = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != y.shape[0] or y.ndim != 1:
        raise ValueError(f"features/labels shape mismatch: {matrix.shape} vs {y.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary {0, 1}")
    counts = np.bincount(y)
    if counts.size < 2 or int(counts.min()) < CV_FOLDS:
        raise ValueError(
            f"stratified {CV_FOLDS}-fold needs each class to appear ≥ {CV_FOLDS} times "
            f"(observed counts: {counts.tolist()})"
        )

    per_seed_ba: list[float] = []
    per_seed_brier: list[float] = []
    for seed in CV_SEEDS:
        splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        probabilities = np.zeros(len(y), dtype=np.float64)
        truths = np.zeros(len(y), dtype=np.int64)
        for train_idx, valid_idx in splitter.split(matrix, y):
            predict_fn = train_fn(matrix[train_idx], y[train_idx], seed)
            fold_probs = np.asarray(predict_fn(matrix[valid_idx]), dtype=np.float64)
            if fold_probs.shape != (len(valid_idx),):
                raise ValueError(
                    f"predict_fn returned shape {fold_probs.shape}, expected ({len(valid_idx)},)"
                )
            if not np.isfinite(fold_probs).all() or (fold_probs < 0).any() or (fold_probs > 1).any():
                raise ValueError("predict_fn returned probabilities outside [0, 1]")
            probabilities[valid_idx] = fold_probs
            truths[valid_idx] = y[valid_idx]
        per_seed_ba.append(balanced_accuracy_score(truths, probabilities >= 0.5))
        per_seed_brier.append(brier_score_loss(truths, probabilities))

    ba = np.asarray(per_seed_ba, dtype=np.float64)
    brier = np.asarray(per_seed_brier, dtype=np.float64)
    return (
        float(ba.mean()),
        float(ba.std()),
        float(brier.mean()),
        float(brier.std()),
    )


def _margin_in_stds(margin: float, spread: float) -> float:
    """Margin expressed in repeat-spread units; zero-spread → strict
    improvement is unbounded superiority, zero margin → 0.0."""
    if spread > 0.0:
        return margin / spread
    if margin > 0.0:
        return math.inf
    if margin < 0.0:
        return -math.inf
    return 0.0


def select_candidate(d_report: CvReport, n_report: CvReport | None) -> SelectionVerdict:
    """Apply the frozen rule: N wins only if
    n.balanced_accuracy_mean − d.balanced_accuracy_mean > SELECTION_MARGIN_STD ×
    repeat-spread std; otherwise D."""
    if n_report is None:
        return SelectionVerdict(
            winner="d-boost", margin_stds=math.inf, d_report=d_report, n_report=None
        )

    ba_spread = max(d_report.balanced_accuracy_std, n_report.balanced_accuracy_std)
    brier_spread = max(d_report.brier_std, n_report.brier_std)
    ba_margin = n_report.balanced_accuracy_mean - d_report.balanced_accuracy_mean
    brier_margin = d_report.brier_mean - n_report.brier_mean  # lower Brier is better

    ba_margin_stds = _margin_in_stds(ba_margin, ba_spread)
    brier_margin_stds = _margin_in_stds(brier_margin, brier_spread)
    n_wins = ba_margin > SELECTION_MARGIN_STD * ba_spread and brier_margin > SELECTION_MARGIN_STD * brier_spread

    return SelectionVerdict(
        winner="n-head" if n_wins else "d-boost",
        margin_stds=min(ba_margin_stds, brier_margin_stds),
        d_report=d_report,
        n_report=n_report,
    )
