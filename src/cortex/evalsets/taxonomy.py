"""Probe-class taxonomy — eval-methodology.md §7 as code constants.

The eight canonical blind-spot classes (the §7 table) plus the LLM-batch
slot classes. Every JSONL probe row of a Layer A eval set carries one of
these names in its ``class`` field; the loader refuses anything else
(unknown class = set-integrity failure, fail loud).

Gate kinds mirror the gate table (eval-methodology §4) AND the frozen
``gate_contract.json`` — the runner evaluates them mechanically:

- ``invariant`` — deterministic, blocks MERGE (self-pair floor, ladder
  monotonicity; symmetry is the §7 tail pinned by this wave);
- ``corridor`` — blocks RELEASE (near-boundary floor, far-negative
  ceiling);
- ``breakdown`` — reported per class, feeds BA, carries NO standalone
  threshold (gate_contract.json is frozen and this wave does not touch
  it — only its CONSUMERS grow).

Coverage-matrix discipline (§8): every ADOPT report opens with the
coverage table built from :data:`COVERAGE_MATRIX`; numbers without the
table are not quotable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

__all__ = [
    "CANONICAL_PROBE_CLASSES",
    "ALL_PROBE_CLASSES",
    "LLM_SLOT_PROBE_CLASSES",
    "GATE_BREAKDOWN",
    "GATE_CORRIDOR",
    "GATE_INVARIANT",
    "PROBE_CLASS_DEGENERATE",
    "PROBE_CLASS_FAR_NEGATIVE",
    "PROBE_CLASS_IDENTITY_SELF",
    "PROBE_CLASS_LLM_NEAR_TOPIC",
    "PROBE_CLASS_LLM_PARAPHRASE",
    "PROBE_CLASS_MONOTONICITY",
    "PROBE_CLASS_NEAR_IDENTITY",
    "PROBE_CLASS_PAIR_SYMMETRY",
    "PROBE_CLASS_TRANSLATION_TWINS",
    "PROBE_CLASS_TYPE_LANG_MISMATCH",
    "SOURCE_LLM_LA2",
    "SOURCE_PROCEDURAL",
    "ProbeClassSpec",
    "COVERAGE_MATRIX",
]

# ── the eight canonical classes (eval-methodology §7, in table order) ────────

#: (a) record vs itself at cos=1.0 — the #480 inversion class.
PROBE_CLASS_IDENTITY_SELF: Final[str] = "identity-self"

#: (b) light mechanical perturbation at cos≈0.99 — threshold-edge collapses.
PROBE_CLASS_NEAR_IDENTITY: Final[str] = "near-identity-twin"

#: P(x,y) = P(y,x) — feature-core symmetry, §7 tail closed by this wave.
PROBE_CLASS_PAIR_SYMMETRY: Final[str] = "pair-symmetry"

#: the fixed pair re-scored along COS_LADDER — non-monotone response.
PROBE_CLASS_MONOTONICITY: Final[str] = "monotonicity-ladder"

#: different-topic pairs at low cosine — false positives on far pairs.
PROBE_CLASS_FAR_NEGATIVE: Final[str] = "far-negative"

#: RU memory vs its EN twin — failure on the language double. The pairs
#: are LLM-generated (a static RU/EN topic pair would be a weaker twin);
#: until the LA-2 batch lands the class stays an honest empty slot.
PROBE_CLASS_TRANSLATION_TWINS: Final[str] = "translation-twins"

#: same text, one metadata field broken (note↔fact, ru↔en) — wrong weight
#: of type_match/lang_match; the features exist (frozen 13), the probes
#: appear with this wave.
PROBE_CLASS_TYPE_LANG_MISMATCH: Final[str] = "type-lang-mismatch"

#: empty body / single token / very long body — degenerate-input robustness.
PROBE_CLASS_DEGENERATE: Final[str] = "degenerate"

CANONICAL_PROBE_CLASSES: Final[tuple[str, ...]] = (
    PROBE_CLASS_IDENTITY_SELF,
    PROBE_CLASS_NEAR_IDENTITY,
    PROBE_CLASS_PAIR_SYMMETRY,
    PROBE_CLASS_MONOTONICITY,
    PROBE_CLASS_FAR_NEGATIVE,
    PROBE_CLASS_TRANSLATION_TWINS,
    PROBE_CLASS_TYPE_LANG_MISMATCH,
    PROBE_CLASS_DEGENERATE,
)

# ── LLM-batch slot classes (beyond the §8 table; phase LA-2) ─────────────────

#: LLM paraphrase pairs (synth strategy ``paraphrase`` family) — duplicate
#: surface beyond the mechanical weak positives.
PROBE_CLASS_LLM_PARAPHRASE: Final[str] = "llm-paraphrase"

#: LLM near-topic pairs (synth strategy ``near-topic`` family) — the W4c
#: hard zone (globally close, semantically another memory).
PROBE_CLASS_LLM_NEAR_TOPIC: Final[str] = "llm-near-topic"

#: Classes whose probes arrive ONLY from the LA-2 LLM batch; the generator
#: works without them (honest empty slots) and picks the rows up from the
#: slots JSONL once filled.
LLM_SLOT_PROBE_CLASSES: Final[tuple[str, ...]] = (
    PROBE_CLASS_TRANSLATION_TWINS,
    PROBE_CLASS_LLM_PARAPHRASE,
    PROBE_CLASS_LLM_NEAR_TOPIC,
)

#: Every class name a JSONL row may carry — the loader's allow-list.
ALL_PROBE_CLASSES: Final[tuple[str, ...]] = (
    CANONICAL_PROBE_CLASSES + LLM_SLOT_PROBE_CLASSES
)

# ── gate kinds (eval-methodology §4; values frozen in gate_contract.json) ────

GATE_INVARIANT: Final[str] = "invariant"
GATE_CORRIDOR: Final[str] = "corridor"
GATE_BREAKDOWN: Final[str] = "breakdown"

# ── probe sources ─────────────────────────────────────────────────────────────

SOURCE_PROCEDURAL: Final[str] = "procedural"

#: The ``source`` value every LA-2 LLM-batch row MUST carry in its JSONL.
SOURCE_LLM_LA2: Final[str] = "llm-batch:la2"


@dataclass(frozen=True)
class ProbeClassSpec:
    """One row of the §7 coverage matrix (class → catcher → source → gate)."""

    name: str
    catches: str
    gate_kind: str
    source: str  # SOURCE_* — "число проб и источник" live in the set meta


#: The coverage matrix in §7 table order. Translation twins carry the
#: LA-2 marker; the two extra LLM slots are appended after the canonical
#: block (they are wave additions, not §7 rows).
COVERAGE_MATRIX: Final[tuple[ProbeClassSpec, ...]] = (
    ProbeClassSpec(
        name=PROBE_CLASS_IDENTITY_SELF,
        catches="self-pair inversion (#480)",
        gate_kind=GATE_INVARIANT,
        source=SOURCE_PROCEDURAL,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_NEAR_IDENTITY,
        catches="collapse at the threshold edge (cos≈0.99)",
        gate_kind=GATE_CORRIDOR,
        source=SOURCE_PROCEDURAL,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_PAIR_SYMMETRY,
        catches="asymmetry of the pair feature core / scoring path",
        gate_kind=GATE_INVARIANT,
        source=SOURCE_PROCEDURAL,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_MONOTONICITY,
        catches="non-monotone P(cos) response",
        gate_kind=GATE_INVARIANT,
        source=SOURCE_PROCEDURAL,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_FAR_NEGATIVE,
        catches="false duplicates on far pairs",
        gate_kind=GATE_CORRIDOR,
        source=SOURCE_PROCEDURAL,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_TRANSLATION_TWINS,
        catches="failure on the language twin",
        gate_kind=GATE_BREAKDOWN,
        source=SOURCE_LLM_LA2,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_TYPE_LANG_MISMATCH,
        catches="wrong type_match / lang_match weighting",
        gate_kind=GATE_BREAKDOWN,
        source=SOURCE_PROCEDURAL,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_DEGENERATE,
        catches="degenerate inputs (empty / 1-token / very long)",
        gate_kind=GATE_BREAKDOWN,
        source=SOURCE_PROCEDURAL,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_LLM_PARAPHRASE,
        catches="paraphrase duplicate surface beyond mechanical twins",
        gate_kind=GATE_BREAKDOWN,
        source=SOURCE_LLM_LA2,
    ),
    ProbeClassSpec(
        name=PROBE_CLASS_LLM_NEAR_TOPIC,
        catches="the W4c hard zone: topically close, another memory",
        gate_kind=GATE_BREAKDOWN,
        source=SOURCE_LLM_LA2,
    ),
)
