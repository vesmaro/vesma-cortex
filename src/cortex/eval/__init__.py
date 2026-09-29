"""Eval package — the single-shot runner (A5) and run-log discipline."""

from cortex.eval.runner import (
    BASELINE_THRESHOLD,
    EvalReport,
    RunLog,
    RunLogEntry,
    RunLogRefusalError,
    run_baseline,
    run_single_shot,
)

__all__ = [
    "BASELINE_THRESHOLD",
    "EvalReport",
    "RunLog",
    "RunLogEntry",
    "RunLogRefusalError",
    "run_baseline",
    "run_single_shot",
]
