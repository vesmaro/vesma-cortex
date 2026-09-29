"""Corruption pretrain — weak positives / hard negatives (candidate N only).

ADR 0001 V1: weak positives = meaning-preserving transformations of ONE
record; hard negatives = same text with a broken field (the W4c twin
lesson, weaponized as training signal). Distillation of the cosine is
FORBIDDEN. Pretrain pairs carry NO owner labels and never extend the
supervised train definition (ADR 0001 V2: unlabeled pairs are pretrain
only; the prereg train definition is untouchable without an addendum).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final, Protocol

from cortex.features.pair import PairRecord

__all__ = [
    "Transformation",
    "WEAK_POSITIVE_TRANSFORMS",
    "HARD_NEGATIVE_TRANSFORMS",
    "PretrainPair",
    "generate_pretrain_pairs",
]


class Transformation(Protocol):
    """One corruption transformation over a record's field.

    Contract: pure (no I/O), deterministic under the passed RNG, RU+EN
    aware. ``field`` names which CanonRecordView field it corrupts — the
    hard-negative generators rely on "same text, broken field" semantics.
    """

    name: str
    field: str  # "title" | "body" | "tags"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord: ...


@dataclass(frozen=True)
class PretrainPair:
    """One unlabeled corruption pair (positive=True → weak positive)."""

    pair_id: str
    record_a: PairRecord
    record_b: PairRecord
    positive: bool
    transform_name: str


#: Meaning-preserving transformations (weak positives). Registry of names
#: frozen at A3b; interface frozen HERE.
WEAK_POSITIVE_TRANSFORMS: Final[tuple[str, ...]] = ()

#: Field-breaking transformations (hard negatives: same text, broken field).
HARD_NEGATIVE_TRANSFORMS: Final[tuple[str, ...]] = ()


def generate_pretrain_pairs(
    records,
    *,
    seed: int,
    max_pairs: int | None = None,
) -> "list[PretrainPair]":
    """Generate unlabeled corruption pairs from store records (A2 corpus).

    Deterministic under ``seed``; ``records`` are store-exported PairRecord
    rows (data-contract.md pretrain manifest, labels absent by design).
    Raises ValueError if asked to emit cosine-distillation targets — that
    signal is forbidden in this package.
    """
    raise NotImplementedError("A3b: corruption generator (A2 corpus feeds it)")
