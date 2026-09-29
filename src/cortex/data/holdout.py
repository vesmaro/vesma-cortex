"""Physical holdout isolation — prereg v2 W5c split + overlap asserts.

Prereg (frozen): after disputed pairs are excluded, within each stratum
pairs sorted by pair_id, the FIRST ⌈0.3·n⌉ go to holdout, the rest to
train. Holdout is touched ONCE (single-shot). Physical isolation (ADR 0001
V1): label files for the holdout live OUTSIDE the train tree; the training
code receives explicit roots and asserts zero pair_id intersection. A
CI test pins the invariants (charter §5, reproducibility).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

__all__ = [
    "HOLDOUT_FRACTION",
    "SplitPair",
    "HoldoutSplit",
    "split_holdout",
    "assert_no_pair_overlap",
    "assert_labels_isolated",
]

#: Frozen holdout fraction (prereg W5c: 30 %, first ⌈0.3·n⌉ per stratum).
HOLDOUT_FRACTION: Final[float] = 0.3

#: Disputed pairs are excluded from all metrics before splitting (prereg).
DISPUTED_LABEL: Final[str] = "disputed"
LABEL_DUPLICATE: Final[str] = "duplicate"
LABEL_NOT_DUPLICATE: Final[str] = "not-duplicate"


@dataclass(frozen=True)
class SplitPair:
    """One pair as the split sees it: id, stratum, label, side hashes."""

    pair_id: str
    stratum: str
    label: str
    sha_a: str
    sha_b: str


@dataclass(frozen=True)
class HoldoutSplit:
    """Result of the frozen split — pair ids only, content stays in data/."""

    train_pair_ids: tuple[str, ...]
    holdout_pair_ids: tuple[str, ...]
    corpus_fingerprint: str


def split_holdout(pairs: Sequence[SplitPair]) -> HoldoutSplit:
    """The frozen prereg split: per stratum, pair_id-sorted, first ⌈0.3·n⌉
    → holdout, rest → train. Disputed pairs must be filtered by the caller
    BEFORE this function (raises if any ride along)."""
    raise NotImplementedError("A3b+: used by eval runner and corpus tooling")


def assert_no_pair_overlap(train_ids: Sequence[str], holdout_ids: Sequence[str]) -> None:
    """Assert zero pair_id intersection between train and holdout.

    Fail-loud (AssertionError with the offending ids): a single leaking
    pair invalidates the single-shot decision number.
    """
    raise NotImplementedError("A3b+: trivial assert, lands with the runner")


def assert_labels_isolated(train_root: Path, labels_path: Path) -> None:
    """Assert the labels file is physically OUTSIDE the train tree.

    Physical isolation (ADR 0001 V1): holdout labels must not be reachable
    under train_root — resolves symlinks before comparing, refuses a
    labels_path that resolves inside train_root.
    """
    raise NotImplementedError("A3b+: lands with the runner + CI test")
