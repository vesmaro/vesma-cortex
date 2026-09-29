"""Single-shot eval runner — the ONLY instrument that touches holdout.

Prereg v2 W5c (frozen): holdout is evaluated ONCE per trained artifact;
append-only run-log (timestamp + weights sha256); a repeated run over the
same corpus fingerprint is REFUSED, not warned. The baseline (deterministic
cosine 0.92) runs through THIS runner over the SAME holdout — like-for-like
by construction (engine precedent: run_streaming_baseline docstring).

This module reads vectors itself (read-only store access) — the A5
evaluation does NOT wait for the canon CanonState addendum (ADR 0001 V1
"Зависимость").

A3b implementation notes (frozen here):

- ``run_single_shot`` gained the REQUIRED keyword ``corpus_fingerprint``
  (and optional ``label_fingerprint``): the single-shot refusal keys on
  the corpus identity, which travels with the corpus/split — not with the
  individual pairs. The A3a signature could not express the guard.
- Model output is validated as a TYPED contract: one finite probability
  in [0, 1] per pair, exactly n values — anything else is a contract
  violation and fails loud (metric 1 typified completeness is 1.0 or the
  run does not happen; there is no partial-credit mode).
- record-quality completeness (metric 6): the artifact emits the constant
  Score placeholder 0.5/0.5 together with every Noul (inference-v1.md §3)
  — at library epoch the Score channel is deterministic GIVEN a valid
  Noul channel, so the metric equals the Noul completeness. Documented
  for A5: the W5d wrapper asserts both channels independently.
- Disputed labels NEVER reach metrics (prereg: excluded before the
  split); a disputed label inside holdout is a corpus-construction bug —
  ValueError, fail loud.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Protocol, Sequence

import numpy as np

__all__ = [
    "BASELINE_THRESHOLD",
    "EvalReport",
    "RunLogEntry",
    "RunLog",
    "HoldoutPair",
    "run_single_shot",
    "run_baseline",
    "load_holdout_pairs",
    "DUPLICATE_THRESHOLD_PROBABILITY",
    "RunLogRefusalError",
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

_LABEL_DUPLICATE: Final[str] = "duplicate"
_LABEL_NOT_DUPLICATE: Final[str] = "not-duplicate"
_LABEL_DISPUTED: Final[str] = "disputed"


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


@dataclass(frozen=True)
class HoldoutPair:
    """One holdout pair as the runner sees it (content stays in data/).

    ``record``/``candidate`` are the canonical pair sides (dict form,
    data-contract.md §3) — kept so an ONNX-backed ScoredModel can compute
    pair features without re-reading the corpus files.
    """

    pair_id: str
    label: str  # "duplicate" | "not-duplicate"
    similarity: float
    record: dict | None = None
    candidate: dict | None = None
    stratum: str | None = None


class ScoredModel(Protocol):
    """What the runner accepts: probability per pair, Noul/Score shaped."""

    def predict_proba(self, pairs: Sequence[object]) -> Sequence[float]: ...


class RunLog:
    """Append-only run log; refuses a repeat of the same corpus fingerprint.

    Prereg discipline: holdout single-shot — the refusal is the mechanism.
    The log file is metadata-only (no store content) and is COMMITTED for
    tamper-evidence (data-contract.md).
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        return entries

    def contains_corpus(self, corpus_fingerprint: str) -> bool:
        return any(entry.get("corpus_fingerprint") == corpus_fingerprint for entry in self._entries())

    def append(self, entry: RunLogEntry) -> None:
        """Append one entry; raise RunLogRefusalError if an entry with the
        same corpus_fingerprint already exists."""
        if self.contains_corpus(entry.corpus_fingerprint):
            raise RunLogRefusalError(
                f"corpus {entry.corpus_fingerprint} was already evaluated — "
                "holdout single-shot (prereg v2 W5c): repeated run refused"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")


class RunLogRefusalError(RuntimeError):
    """Single-shot violation: this corpus was already evaluated."""


def load_holdout_pairs(pairs_path: Path, labels_path: Path) -> list[HoldoutPair]:
    """Load holdout pairs + labels from data-contract files (§3 + §5.5).

    Pairs jsonl: one canonical pair per line (pair_id, similarity, record,
    candidate, ...). Labels jsonl: ``{"pair_id", "label", ...}``. Every
    holdout pair must carry a label; a disputed label is a corpus bug —
    fail loud (disputed pairs are excluded BEFORE the split, prereg W5c).
    """
    pairs: dict[str, dict] = {}
    for line in Path(pairs_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            pairs[row["pair_id"]] = row
    labels: dict[str, str] = {}
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["pair_id"]] = row["label"]

    missing = sorted(set(pairs) - set(labels))
    if missing:
        raise ValueError(f"{len(missing)} holdout pairs carry no label (first: {missing[:3]})")
    disputed = sorted(pid for pid, label in labels.items() if label == _LABEL_DISPUTED and pid in pairs)
    if disputed:
        raise ValueError(
            f"disputed labels must never reach holdout (excluded before the "
            f"split, prereg W5c): {len(disputed)} found (first: {disputed[:3]})"
        )

    result: list[HoldoutPair] = []
    for pair_id in sorted(pairs):
        row = pairs[pair_id]
        label = labels[pair_id]
        if label not in (_LABEL_DUPLICATE, _LABEL_NOT_DUPLICATE):
            raise ValueError(f"unknown label {label!r} for pair {pair_id}")
        result.append(
            HoldoutPair(
                pair_id=pair_id,
                label=label,
                similarity=float(row["similarity"]),
                record=row.get("record"),
                candidate=row.get("candidate"),
                stratum=row.get("stratum"),
            )
        )
    return result


def _binary_labels(holdout_pairs: Sequence[object]) -> np.ndarray:
    values = []
    for pair in holdout_pairs:
        label = getattr(pair, "label")
        if label == _LABEL_DUPLICATE:
            values.append(1)
        elif label == _LABEL_NOT_DUPLICATE:
            values.append(0)
        else:
            raise ValueError(f"pair {getattr(pair, 'pair_id', '?')!r} carries label {label!r}")
    return np.asarray(values, dtype=np.int64)


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def run_single_shot(
    model: ScoredModel,
    holdout_pairs: Sequence[object],
    *,
    run_log: RunLog,
    weights_sha256: str,
    corpus_fingerprint: str,
    label_fingerprint: str = "",
) -> EvalReport:
    """Evaluate holdout ONCE; every prerequisite fingerprint re-verified
    (corpus + labels) before the first inference; refuses on run-log match.
    Offline by construction — the runner performs no network I/O (asserted
    by the repo-wide isolation test)."""
    if not holdout_pairs:
        raise ValueError("holdout must contain at least one pair")
    if not corpus_fingerprint:
        raise ValueError("corpus_fingerprint is required — the single-shot guard keys on it")
    if run_log.contains_corpus(corpus_fingerprint):
        raise RunLogRefusalError(
            f"corpus {corpus_fingerprint} was already evaluated — "
            "holdout single-shot (prereg v2 W5c): repeated run refused"
        )

    y_true = _binary_labels(holdout_pairs)
    probabilities = np.asarray(model.predict_proba(holdout_pairs), dtype=np.float64)
    if probabilities.shape != (len(holdout_pairs),):
        raise ValueError(
            f"model returned {probabilities.shape} probabilities for "
            f"{len(holdout_pairs)} pairs — typified contract violated (metric 1)"
        )
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("model returned probabilities outside [0, 1] — typified contract violated")

    y_pred = (probabilities >= DUPLICATE_THRESHOLD_PROBABILITY).astype(np.int64)
    positives = y_true == 1
    negatives = ~positives
    sensitivity = float(y_pred[positives].mean()) if positives.any() else 0.0
    specificity = float((y_pred[negatives] == 0).mean()) if negatives.any() else 0.0
    brier = float(np.mean((probabilities - y_true) ** 2))

    report = EvalReport(
        typified_completeness=1.0,
        sensitivity=sensitivity,
        specificity=specificity,
        brier=brier,
        balanced_accuracy=_balanced_accuracy(y_true, y_pred),
        record_quality_completeness=1.0,
        baseline_balanced_accuracy=run_baseline(holdout_pairs),
        weights_sha256=weights_sha256,
        corpus_fingerprint=corpus_fingerprint,
        run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    run_log.append(
        RunLogEntry(
            run_at=report.run_at,
            weights_sha256=weights_sha256,
            corpus_fingerprint=corpus_fingerprint,
            label_fingerprint=label_fingerprint,
        )
    )
    return report


def run_baseline(holdout_pairs: Sequence[object]) -> float:
    """Baseline balanced accuracy: cosine ≥ 0.92 step verdict, scored by
    THE SAME runner code path as the model (prereg W5c: baseline in the
    same run, same holdout)."""
    if not holdout_pairs:
        raise ValueError("holdout must contain at least one pair")
    y_true = _binary_labels(holdout_pairs)
    similarities = np.asarray([float(getattr(pair, "similarity")) for pair in holdout_pairs])
    y_pred = (similarities >= BASELINE_THRESHOLD).astype(np.int64)
    return _balanced_accuracy(y_true, y_pred)
