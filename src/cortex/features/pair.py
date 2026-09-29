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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, NamedTuple

__all__ = [
    "PairRecord",
    "FeatureVector",
    "FEATURE_NAMES",
    "FIELD_COSINE_FEATURES",
    "features",
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
#: in this fixed order). Computed by the bundle embedder (mnema-embed-v1)
#: over single fields — docomputed on CPU in A2, short fields are a weak
#: signal by ADR 0001 (accepted risk).
FIELD_COSINE_FEATURES: Final[tuple[str, ...]] = (
    "cos_title",
    "cos_body",
    "cos_tags",
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
        vec_a/vec_b: store embeddings (384-dim float32, mnema-embed-v1);
            required input for candidate N's vector block — accepted here
            so the pair-feature pass is single-source (asserted non-empty
            and equal-length when given).
        field_cosines: include the three field-cosine features (requires
            vec_a/vec_b — the bundle embedder docomputes them in A2).

    Returns:
        FeatureVector with names = FEATURE_NAMES (+ FIELD_COSINE_FEATURES
        when ``field_cosines``), values in the same order.

    Raises:
        ValueError: on contract violations (similarity outside [-1, 1],
            mismatched vector dimensions, field_cosines without vectors).
    """
    raise NotImplementedError("A3b: pair feature computation (frozen here, implemented next slice)")
