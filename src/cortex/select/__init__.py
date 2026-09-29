"""Selection package — frozen CV protocol (stratified 5-fold × 20 seeds)."""

from cortex.select.cv import (
    CV_FOLDS,
    CV_SEEDS,
    CvReport,
    SelectionVerdict,
    run_cv,
    select_candidate,
)

__all__ = [
    "CV_FOLDS",
    "CV_SEEDS",
    "CvReport",
    "SelectionVerdict",
    "run_cv",
    "select_candidate",
]
