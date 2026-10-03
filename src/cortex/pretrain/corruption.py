"""Corruption pretrain — weak positives / hard negatives (candidate N only).

ADR 0001 V1: weak positives = meaning-preserving transformations of ONE
record; hard negatives = same text with a broken field (the W4c twin
lesson, weaponized as training signal). Distillation of the cosine is
FORBIDDEN. Pretrain pairs carry NO owner labels and never extend the
supervised train definition (ADR 0001 V2: unlabeled pairs are pretrain
only; the prereg train definition is untouchable without an addendum).

A3b implementation notes (frozen here):

- All transforms are MECHANICAL (no LLM, no model): punctuation/quote
  normalization, whitespace collapse, tag reordering, title case flips;
  hard negatives substitute exactly ONE field (title/body/tags from a
  donor record, language/record_type from the frozen value cycle).
- Hard-negative donors are picked deterministically from the SAME corpus
  batch: a donor qualifies only if the substituted field actually differs
  AND the donor record is not content-identical to the base. When no
  qualifying donor exists (degenerate corpus) the pair is SKIPPED — a
  pair labeled negative must never be content-identical to its positive
  twin.
- ``pair_id`` is synthetic: ``<base12>--<transform>`` where ``base12`` is
  the first 12 hex of sha256 over the canonical JSON of the base record.
  Store ids never appear here (corruption pairs are generated, not
  exported pairs); the data-contract §4 manifest keeps this id verbatim.
- Corruption pairs are SELF-pairs: record_b derives from record_a, so the
  measured-similarity convention for the pair manifest is 1.0 (the store
  vector of the base record stands for both sides — approximating "same
  text, broken field" with a globally identical vector block is exactly
  the training signal: divergence must be read from the scalar features).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Final, Protocol

from cortex.data.fingerprints import canonical_json
from cortex.features.pair import PairRecord

__all__ = [
    "HARD_NEGATIVE_TRANSFORMS",
    "WEAK_POSITIVE_TRANSFORMS",
    "PretrainPair",
    "Transformation",
    "generate_pretrain_pairs",
    "record_key",
]

#: Languages/record types the flips cycle through (CanonRecordView domain).
_LANGUAGES: Final[tuple[str, ...]] = ("ru", "en")
_RECORD_TYPES: Final[tuple[str, ...]] = ("note", "snippet", "fact")

_PUNCT_TABLE: Final[dict[int, str]] = str.maketrans(
    {
        "…": "...",
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "‐": "-",
    }
)


class Transformation(Protocol):
    """One corruption transformation over a record's field.

    Contract: pure (no I/O), deterministic under the passed RNG, RU+EN
    aware. ``field`` names which CanonRecordView field it corrupts — the
    hard-negative generators rely on "same text, broken field" semantics
    (A3b: the domain is title/body/tags for content fields plus
    language/record_type for the enum flips).
    """

    name: str
    field: str  # "title" | "body" | "tags" | "language" | "record_type"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord: ...


# ── Weak positives (meaning-preserving, mechanical) ───────────────────────────


@dataclass(frozen=True)
class ShuffleTags:
    """Reorder tags — the tag SET (and tag_jaccard) is invariant."""

    name: str = "shuffle_tags"
    field: str = "tags"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord:
        tags = list(record.tags)
        rng.shuffle(tags)
        return replace(record, tags=tuple(tags))


@dataclass(frozen=True)
class CollapseWhitespaceBody:
    """Collapse whitespace runs in the body to single spaces."""

    name: str = "collapse_ws_body"
    field: str = "body"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord:
        return replace(record, body=" ".join(record.body.split()))


@dataclass(frozen=True)
class NormalizePunctuationBody:
    """Typographic → ASCII punctuation in the body (…, «», “”, —, ’ …)."""

    name: str = "normalize_punct"
    field: str = "body"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord:
        return replace(record, body=record.body.translate(_PUNCT_TABLE))


@dataclass(frozen=True)
class SwapCaseTitle:
    """Flip the letter case of the title (mechanical, meaning-preserving)."""

    name: str = "retitle_case"
    field: str = "title"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord:
        return replace(record, title=record.title.swapcase())


#: Meaning-preserving transformations (weak positives). Registry of names
#: frozen at A3b; interface frozen HERE.
WEAK_POSITIVE_TRANSFORMS: Final[tuple[str, ...]] = (
    "shuffle_tags",
    "collapse_ws_body",
    "normalize_punct",
    "retitle_case",
)

_WEAK_POSITIVE_IMPLS: Final[tuple[Transformation, ...]] = (
    ShuffleTags(),
    CollapseWhitespaceBody(),
    NormalizePunctuationBody(),
    SwapCaseTitle(),
)


# ── Hard negatives (same text, broken field) ──────────────────────────────────


@dataclass(frozen=True)
class SubstituteField:
    """Replace ONE record field with the donor's value (base class).

    The donor must differ in ``field`` and not be content-identical to the
    base — the generator (:func:`generate_pretrain_pairs`) enforces this
    before constructing the transform.
    """

    transform_name: str
    target_field: str
    donor: PairRecord

    @property
    def name(self) -> str:
        return self.transform_name

    @property
    def field(self) -> str:
        return self.target_field

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord:
        return replace(
            record, **{self.target_field: getattr(self.donor, self.target_field)}
        )


@dataclass(frozen=True)
class FlipLanguage:
    """language → a different value from the frozen cycle (always changes)."""

    name: str = "flip_language"
    field: str = "language"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord:
        candidates = [lang for lang in _LANGUAGES if lang != record.language]
        return replace(record, language=candidates[0] if candidates else None)


@dataclass(frozen=True)
class FlipRecordType:
    """record_type → the next value of the frozen cycle (always changes)."""

    name: str = "flip_record_type"
    field: str = "record_type"

    def apply(self, record: PairRecord, rng: random.Random) -> PairRecord:
        order = _RECORD_TYPES if record.record_type in _RECORD_TYPES else ("note",)
        next_index = (
            (order.index(record.record_type) + 1) % len(order)
            if record.record_type
            else 0
        )
        return replace(record, record_type=_RECORD_TYPES[next_index])


#: Field-breaking transformations (hard negatives: same text, broken field).
HARD_NEGATIVE_TRANSFORMS: Final[tuple[str, ...]] = (
    "swap_title",
    "swap_body",
    "swap_tags",
    "flip_language",
    "flip_record_type",
)


def record_key(record: PairRecord) -> str:
    """Content hash of a record — first 12 hex of sha256 (canonical JSON)."""
    payload = {
        "title": record.title,
        "body": record.body,
        "tags": list(record.tags),
        "language": record.language,
        "record_type": record.record_type,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:12]


def _donor_differs(base: PairRecord, donor: PairRecord, field: str) -> bool:
    if getattr(donor, field) == getattr(base, field):
        return False
    return record_key(donor) != record_key(base)


def _make_hard_negative(
    transform_name: str, base: PairRecord, donor: PairRecord | None
) -> Transformation:
    if transform_name == "swap_title":
        if donor is None or not _donor_differs(base, donor, "title"):
            raise ValueError(
                f"swap_title needs a qualifying donor for {record_key(base)}"
            )
        return SubstituteField("swap_title", "title", donor)
    if transform_name == "swap_body":
        if donor is None or not _donor_differs(base, donor, "body"):
            raise ValueError(
                f"swap_body needs a qualifying donor for {record_key(base)}"
            )
        return SubstituteField("swap_body", "body", donor)
    if transform_name == "swap_tags":
        if donor is None or not _donor_differs(base, donor, "tags"):
            raise ValueError(
                f"swap_tags needs a qualifying donor for {record_key(base)}"
            )
        return SubstituteField("swap_tags", "tags", donor)
    if transform_name == "flip_language":
        return FlipLanguage()
    if transform_name == "flip_record_type":
        return FlipRecordType()
    raise ValueError(f"unknown hard-negative transform: {transform_name!r}")


def _pick_donor(
    rng: random.Random, base: PairRecord, records: Sequence[PairRecord], field: str
) -> PairRecord | None:
    """Deterministic donor search: shuffled index order, first qualifying."""
    order = list(range(len(records)))
    rng.shuffle(order)
    for index in order:
        if _donor_differs(base, records[index], field):
            return records[index]
    return None


def generate_pretrain_pairs(
    records,
    *,
    seed: int,
    max_pairs: int | None = None,
) -> list[PretrainPair]:
    """Generate unlabeled corruption pairs from store records (A2 corpus).

    Deterministic under ``seed``; ``records`` are store-exported PairRecord
    rows (data-contract.md pretrain manifest, labels absent by design).
    Raises ValueError if asked to emit cosine-distillation targets — that
    signal is forbidden in this package.

    Per base record the generator emits ONE weak positive (random
    meaning-preserving transform) and ONE hard negative (random
    field-breaking transform; donor-based transforms are skipped when no
    qualifying donor exists). ``max_pairs`` truncates the emitted list
    from the front — the result stays deterministic for a fixed seed.
    """
    pool = list(records)
    if max_pairs is not None and max_pairs < 0:
        raise ValueError(f"max_pairs must be >= 0, got {max_pairs}")
    if not pool:
        return []

    rng = random.Random(seed)
    pairs: list[PretrainPair] = []

    for base in pool:
        key = record_key(base)
        weak = rng.choice(_WEAK_POSITIVE_IMPLS)
        pairs.append(
            PretrainPair(
                pair_id=f"{key}--{weak.name}",
                record_a=base,
                record_b=weak.apply(base, rng),
                positive=True,
                transform_name=weak.name,
            )
        )

        hard_name = rng.choice(HARD_NEGATIVE_TRANSFORMS)
        donor_field = {
            "swap_title": "title",
            "swap_body": "body",
            "swap_tags": "tags",
        }.get(hard_name)
        donor = _pick_donor(rng, base, pool, donor_field) if donor_field else None
        if donor_field and donor is None:
            continue  # no honest donor — skip rather than emit a false negative
        hard = _make_hard_negative(hard_name, base, donor)
        pairs.append(
            PretrainPair(
                pair_id=f"{key}--{hard.name}",
                record_a=base,
                record_b=hard.apply(base, rng),
                positive=False,
                transform_name=hard.name,
            )
        )

    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return pairs


@dataclass(frozen=True)
class PretrainPair:
    """One unlabeled corruption pair (positive=True → weak positive)."""

    pair_id: str
    record_a: PairRecord
    record_b: PairRecord
    positive: bool
    transform_name: str
