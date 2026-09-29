"""Physical holdout isolation — prereg v2 W5c split + overlap asserts.

Prereg (frozen): after disputed pairs are excluded, within each stratum
pairs sorted by pair_id, the FIRST ⌈0.3·n⌉ go to holdout, the rest to
train. Holdout is touched ONCE (single-shot). Physical isolation (ADR 0001
V1): label files for the holdout live OUTSIDE the train tree; the training
code receives explicit roots and asserts zero pair_id intersection. A
CI test pins the invariants (charter §5, reproducibility).

A3b note: ``SplitPair`` gained the ``pair_sha256`` field — the frozen
manifest scheme (data-contract.md §5.3) fingerprints pairs as
``pair_id <pair_sha256>``, so ``HoldoutSplit.corpus_fingerprint`` cannot be
computed from side hashes alone. Additive, keyword-constructors unaffected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

from cortex.data.fingerprints import corpus_fingerprint as _blake2b_of_manifest
from cortex.data.fingerprints import manifest_bytes

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
    #: sha256 of the canonical pair object — the manifest-line digest
    #: (data-contract.md §5.2–5.3); required so the split can carry an
    #: honest corpus fingerprint.
    pair_sha256: str


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
    items = list(pairs)
    disputed = [p.pair_id for p in items if p.label == DISPUTED_LABEL]
    if disputed:
        raise ValueError(
            f"disputed pairs must be excluded BEFORE the split (prereg W5c): {len(disputed)} found"
        )
    for pair in items:
        if not pair.pair_sha256:
            raise ValueError(
                f"pair {pair.pair_id!r} carries no pair_sha256 — the split "
                "fingerprint would be wrong (data-contract.md §5.3)"
            )

    strata: dict[str, list[SplitPair]] = {}
    for pair in items:
        strata.setdefault(pair.stratum, []).append(pair)

    train_ids: list[str] = []
    holdout_ids: list[str] = []
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda p: p.pair_id)
        n_holdout = math.ceil(HOLDOUT_FRACTION * len(members))
        holdout_ids.extend(p.pair_id for p in members[:n_holdout])
        train_ids.extend(p.pair_id for p in members[n_holdout:])

    fingerprint = _blake2b_of_manifest(
        manifest_bytes((p.pair_id, p.pair_sha256) for p in items)
    )
    return HoldoutSplit(
        train_pair_ids=tuple(train_ids),
        holdout_pair_ids=tuple(holdout_ids),
        corpus_fingerprint=fingerprint,
    )


def assert_no_pair_overlap(train_ids: Sequence[str], holdout_ids: Sequence[str]) -> None:
    """Assert zero pair_id intersection between train and holdout.

    Fail-loud (AssertionError with the offending ids): a single leaking
    pair invalidates the single-shot decision number.
    """
    leaking = sorted(set(train_ids) & set(holdout_ids))
    if leaking:
        raise AssertionError(
            f"train/holdout pair_id intersection is non-empty ({len(leaking)} ids, "
            f"first: {leaking[:5]}) — single-shot holdout is compromised"
        )


def assert_labels_isolated(train_root: Path, labels_path: Path) -> None:
    """Assert the labels file is physically OUTSIDE the train tree.

    Physical isolation (ADR 0001 V1): holdout labels must not be reachable
    under train_root — resolves symlinks before comparing, refuses a
    labels_path that resolves inside train_root.
    """
    train_resolved = Path(train_root).resolve()
    labels_resolved = Path(labels_path).resolve()
    if labels_resolved == train_resolved or train_resolved in labels_resolved.parents:
        raise AssertionError(
            f"holdout labels {labels_path} resolve INSIDE the train tree {train_root} "
            "(physical isolation violated, ADR 0001 V1)"
        )
