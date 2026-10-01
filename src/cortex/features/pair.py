"""Pair features — the frozen feature contract of the D/N ladder.

This module FROZES the ordered feature-name list (ArchCom A1 contract,
``pair_features``; ADR 0001 V1). The list is the artifact-side contract too:
inference-v1.md §4 pins it into ONNX metadata_props, and the W5d wrapper
computes the same features BEFORE the graph. Order is part of the contract —
append-only evolution, never reorder (a reorder silently permutes the graph
input and is caught only by feature_set_sha256).

Feature families (ADR 0001 V1):

- target cosine — the MEASURED store/vector-leg cosine (CanonState
  similarity; providers consume it, never re-measure);
- char 3–5-gram Jaccard/containment, RU+EN — local divergence under global
  closeness (the W4c lesson: template twins with a broken field);
- tag Jaccard;
- metadata deltas — lengths, type, language. NO time delta: CanonRecordView
  carries no timestamp, see inference-v1.md §10 OQ-2 (would require a canon
  addendum first);
- field cosines (title/body/tags) — OPTIONAL, behind ``field_cosines=True``;
  precomputed by the bundle embedder on CPU (A2 docompute), ablation
  "with/without field cosines" per ADR 0001.

A3b implementation notes (contract decisions, frozen here):

- Char n-grams are computed over ``title + "\\n" + body`` after
  letter-normalization: unicode ``str.lower()`` plus whitespace collapsing
  (``" ".join(text.split())``). Punctuation is NOT stripped on purpose —
  punctuation-only edits are weak-positive corruption transforms
  (cortex.pretrain.corruption) and must remain visible to the features.
  Tags are NOT part of the n-gram text (they have their own feature).
- ``*_delta`` features are ABSOLUTE differences (|len(a) − len(b)|), so the
  core vector is symmetric under (a, b) swap — pinned by tests.
- Field cosines are GATED: this library carries no embedder, so
  ``features(..., field_cosines=True)`` refuses loudly and the three cosine
  values are attached later from the A2 sidecar via
  :func:`attach_field_cosines` (single-source feature pass preserved).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, NamedTuple

__all__ = [
    "PairRecord",
    "RecordLike",
    "FeatureVector",
    "FEATURE_NAMES",
    "FIELD_COSINE_FEATURES",
    "CHAR_NGRAM_ORDERS",
    "features",
    "attach_field_cosines",
]


@dataclass(frozen=True)
class RecordLike:
    """Mirror of the engine ``CanonRecordView`` (ADR 0004 rule 2 surface).

    Cross-repo mirror: field-for-field equal to ``CanonRecordView`` in the
    engine's ``src/vesmaro/decision_provider.py`` (title, body, tags,
    language, record_type). Kept in sync manually — the drift is pinned by
    tests (a field change there must be mirrored here before any corpus
    export). ``created_at`` deliberately NOT here: the engine view does not
    carry it (OQ-2).
    """

    title: str
    body: str
    tags: tuple[str, ...] = ()
    language: str | None = None
    record_type: str | None = None


#: Backwards-friendly alias — records enter as PairRecord semantics.
PairRecord = RecordLike


class FeatureVector(NamedTuple):
    """An ordered feature vector + the names it was computed against.

    Invariant: ``len(values) == len(names)``; ``names`` is either
    FEATURE_NAMES (core) or FEATURE_NAMES + FIELD_COSINE_FEATURES (ablation
    variant). The selected variant's ``names`` is what export pins into
    artifact metadata (cortex.artifacts.build_metadata_props).
    """

    names: tuple[str, ...]
    values: tuple[float, ...]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.values)


#: Core frozen ordered feature names (13). DO NOT REORDER — see module
#: docstring. Char n-grams are over the RU+EN letter-normalized text
#: (lowercase, whitespace-collapsed); jaccard = |A∩B|/|A∪B|, containment =
#: |A∩B|/min(|A|,|B|) over n-gram sets; *_delta are |len(a)−len(b)|;
#: *_match are 1.0 when both sides carry the same non-null value, 0.0 on
#: mismatch or double-null.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    # measured target cosine (input `similarity`, CanonState semantics)
    "cos_target",
    # char n-gram divergence, RU+EN
    "char3_jaccard",
    "char4_jaccard",
    "char5_jaccard",
    "char3_containment",
    "char4_containment",
    "char5_containment",
    # tags
    "tag_jaccard",
    # metadata deltas
    "title_len_delta",
    "body_len_delta",
    "tag_count_delta",
    "type_match",
    "lang_match",
)

#: Optional field cosines (ablation variant; appended AFTER the core block
#: in this fixed order). Computed by the bundle embedder (vesma-embed-v1)
#: over single fields — docomputed on CPU in A2, short fields are a weak
#: signal by ADR 0001 (accepted risk).
FIELD_COSINE_FEATURES: Final[tuple[str, ...]] = (
    "cos_title",
    "cos_body",
    "cos_tags",
)

#: N-gram orders of the char-divergence family (contract: 3–5).
CHAR_NGRAM_ORDERS: Final[tuple[int, ...]] = (3, 4, 5)


def _normalized_text(record: PairRecord) -> str:
    """Letter-normalized text of one side: title + body, lower, ws-collapsed."""
    return " ".join(f"{record.title}\n{record.body}".split()).lower()


def _ngrams(text: str, n: int) -> frozenset[str]:
    """Character n-gram SET (unicode chars, no tokenizer) of order ``n``."""
    if len(text) < n:
        return frozenset()
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|A∩B|/|A∪B|; two empty sets are identical → 1.0."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    """|A∩B|/min(|A|,|B|); both empty → 1.0, one empty → 0.0."""
    if not a and not b:
        return 1.0
    denominator = min(len(a), len(b))
    if denominator == 0:
        return 0.0
    return len(a & b) / denominator


def _match(value_a: str | None, value_b: str | None) -> float:
    """1.0 iff both sides carry the SAME non-null value, else 0.0."""
    if value_a is None or value_b is None:
        return 0.0
    return 1.0 if value_a == value_b else 0.0


def _validate_similarity(similarity: float) -> float:
    try:
        value = float(similarity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"similarity must be a real number, got {similarity!r}") from exc
    if math.isnan(value) or value < -1.0 or value > 1.0:
        raise ValueError(f"similarity must be within [-1, 1] and not NaN, got {value!r}")
    return value


def _validate_vectors(vec_a: tuple[float, ...] | None, vec_b: tuple[float, ...] | None) -> None:
    """Store vectors are optional; when given they must be non-empty twins."""
    if vec_a is None and vec_b is None:
        return
    if vec_a is None or vec_b is None:
        raise ValueError("vec_a and vec_b must be provided together")
    if len(vec_a) == 0 or len(vec_a) != len(vec_b):
        raise ValueError(
            f"vec_a/vec_b must be non-empty and equal-length, got {len(vec_a)} vs {len(vec_b)}"
        )


def features(
    record_a: PairRecord,
    record_b: PairRecord,
    similarity: float,
    vec_a: tuple[float, ...] | None = None,
    vec_b: tuple[float, ...] | None = None,
    *,
    field_cosines: bool = False,
) -> FeatureVector:
    """Compute the frozen pair features — pure, deterministic, no network.

    Args:
        record_a: earlier record (canonical pair JSON ``record``).
        record_b: candidate record (canonical pair JSON ``candidate``).
        similarity: MEASURED vector-leg cosine over the pair (never
            recomputed here — CanonState "consume, never re-measure").
        vec_a/vec_b: store embeddings (384-dim float32, vesma-embed-v1);
            required input for candidate N's vector block — accepted here
            so the pair-feature pass is single-source (asserted non-empty
            and equal-length when given).
        field_cosines: include the three field-cosine features (requires
            vec_a/vec_b — the bundle embedder docomputes them in A2).

    Returns:
        FeatureVector with names = FEATURE_NAMES (+ FIELD_COSINE_FEATURES
        when ``field_cosines``), values in the same order.

    Raises:
        ValueError: on contract violations — similarity outside [-1, 1] or
            NaN, mismatched/empty vector twins, or ``field_cosines=True``
            (the library carries no embedder; build the ablation variant
            via :func:`attach_field_cosines` instead).
    """
    cos_target = _validate_similarity(similarity)
    _validate_vectors(vec_a, vec_b)
    if field_cosines:
        raise ValueError(
            "field cosines are docomputed by the bundle embedder (A2 sidecar) — "
            "this library has no embedder by contract; compute the core vector "
            "with field_cosines=False and attach the sidecar values via "
            "cortex.features.pair.attach_field_cosines"
        )

    text_a = _normalized_text(record_a)
    text_b = _normalized_text(record_b)
    grams_a = {n: _ngrams(text_a, n) for n in CHAR_NGRAM_ORDERS}
    grams_b = {n: _ngrams(text_b, n) for n in CHAR_NGRAM_ORDERS}

    values: list[float] = [cos_target]
    for n in CHAR_NGRAM_ORDERS:
        values.append(_jaccard(grams_a[n], grams_b[n]))
    for n in CHAR_NGRAM_ORDERS:
        values.append(_containment(grams_a[n], grams_b[n]))
    values.append(_jaccard(frozenset(record_a.tags), frozenset(record_b.tags)))
    values.append(abs(len(record_a.title) - len(record_b.title)))
    values.append(abs(len(record_a.body) - len(record_b.body)))
    values.append(abs(len(record_a.tags) - len(record_b.tags)))
    values.append(_match(record_a.record_type, record_b.record_type))
    values.append(_match(record_a.language, record_b.language))

    return FeatureVector(names=FEATURE_NAMES, values=tuple(float(v) for v in values))


def attach_field_cosines(
    vector: FeatureVector,
    *,
    cos_title: float,
    cos_body: float,
    cos_tags: float,
) -> FeatureVector:
    """Attach the A2-docomputed field cosines to a core feature vector.

    The ablation variant (ADR 0001 V1: "с/без полевых косинусов") is built
    here, NOT inside :func:`features` — the library carries no embedder.
    Field-cosine values must each lie in [-1, 1]; the input vector must be
    the core variant (attaching twice is a contract violation).
    """
    if vector.names != FEATURE_NAMES:
        raise ValueError(
            "field cosines attach to the CORE vector only "
            f"(names == FEATURE_NAMES expected, got {len(vector.names)} names)"
        )
    cosines = []
    for name, value in (
        ("cos_title", cos_title),
        ("cos_body", cos_body),
        ("cos_tags", cos_tags),
    ):
        try:
            v = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if math.isnan(v) or v < -1.0 or v > 1.0:
            raise ValueError(f"{name} must be within [-1, 1] and not NaN, got {v!r}")
        cosines.append(v)
    return FeatureVector(
        names=FEATURE_NAMES + FIELD_COSINE_FEATURES,
        values=vector.values + tuple(cosines),
    )
