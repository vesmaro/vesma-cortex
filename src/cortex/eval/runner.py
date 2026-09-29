"""Single-shot eval runner — the ONLY instrument that touches holdout.

Prereg v2 W5c (frozen): holdout is evaluated ONCE per trained artifact;
append-only run-log (timestamp + weights sha256); a repeated run over the
same corpus fingerprint is REFUSED, not warned. The baseline (deterministic
cosine 0.92) runs through THIS runner over the SAME holdout — like-for-like
by construction (engine precedent: run_streaming_baseline docstring).

This module reads vectors itself (read-only store access) — the A5
evaluation does NOT wait for the canon CanonState addendum (ADR 0001 V1
"Зависимость").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, Sequence

__all__ = [
    "BASELINE_THRESHOLD",
    "EvalReport",
    "RunLogEntry",
    "RunLog",
    "run_single_shot",
    "run_baseline",
    "DUPLICATE_THRESHOLD_PROBABILITY",
]

#: The engine's near-duplicate threshold (graph_minting
#: AUTO_DEDUPE_SIMILARITY_THRESHOLD = 0.92) — mirrored here because the
#: runner must not import the engine. Kept in sync manually.
BASELINE_THRESHOLD: Final[float] = 0.92

#: Probability cutoff the REPORT uses to binarize model output for metric
#: 2/3/5 (sensitivity/specificity/balanced accuracy). The APPLICATION
#: threshold is W5d product policy (inference-v1.md §8) — this one is the
#: prereg's fixed scoring cut, pinned to 0.5 (maximum-likelihood readout
#: of a calibrated probability).
DUPLICATE_THRESHOLD_PROBABILITY: Final[float] = 0.5


@dataclass(frozen=True)
class EvalReport:
    """The 6 prereg metrics over holdout + verdict inputs."""

    typified_completeness: float  # metric 1: Noul on every pair, target 1.0
    sensitivity: float  # metric 2, adopt ≥ 0.90
    specificity: float  # metric 3, adopt ≥ 0.90
    brier: float  # metric 4, adopt ≤ 0.25
    balanced_accuracy: float  # metric 5: strictly > baseline on same holdout
    record_quality_completeness: float  # metric 6: Score on every record, 1.0
    baseline_balanced_accuracy: float
    weights_sha256: str
    corpus_fingerprint: str
    run_at: str


@dataclass(frozen=True)
class RunLogEntry:
    """One append-only run-log line (committed: artifacts/runs/run_log.jsonl)."""

    run_at: str  # ISO-8601 UTC
    weights_sha256: str
    corpus_fingerprint: str
    label_fingerprint: str


class ScoredModel(Protocol):
    """What the runner accepts: probability per pair, Noul/Score shaped."""

    def predict_proba(self, pairs: Sequence[object]) -> Sequence[float]: ...


class RunLog:
    """Append-only run log; refuses a repeat of the same corpus fingerprint.

    Prereg discipline: holdout single-shot — the refusal is the mechanism.
    The log file is metadata-only (no store content) and is COMMITTED for
    tamper-evidence (data-contract.md).
    """

    def __init__(self, path: Path) -> None: ...

    def append(self, entry: RunLogEntry) -> None:
        """Append one entry; raise RunLogRefusalError if an entry with the
        same corpus_fingerprint already exists."""
        raise NotImplementedError("A3b: lands with the runner")

    def contains_corpus(self, corpus_fingerprint: str) -> bool:
        raise NotImplementedError("A3b: lands with the runner")


class RunLogRefusalError(RuntimeError):
    """Single-shot violation: this corpus was already evaluated."""


def run_single_shot(
    model: ScoredModel,
    holdout_pairs: Sequence[object],
    *,
    run_log: RunLog,
    weights_sha256: str,
) -> EvalReport:
    """Evaluate holdout ONCE; every prerequisite fingerprint re-verified
    (corpus + labels) before the first inference; refuses on run-log match.
    Offline by construction — the runner performs no network I/O (asserted
    by the repo-wide isolation test)."""
    raise NotImplementedError("A3b+: A5 wave")


def run_baseline(holdout_pairs: Sequence[object]) -> float:
    """Baseline balanced accuracy: cosine ≥ 0.92 step verdict, scored by
    THE SAME runner code path as the model (prereg W5c: baseline in the
    same run, same holdout)."""
    raise NotImplementedError("A3b+: A5 wave")
