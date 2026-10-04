"""Layer A eval-set generators — deterministic procedural probes.

Builds the two frozen sets (eval-methodology §3): ``merge`` ~60 pairs
(quick gate, every PR) and ``release`` ≥ 200 pairs (stratified, at the
release gate). Design contracts:

- DETERMINISTIC: no wall clock, no RNG drift — the same recipe + seed
  produce BYTE-IDENTICAL JSONL (pinned by a double-run test). The seed
  only rotates weak-perturbation variants; pairing scans are structural
  (index arithmetic), so determinism does not hinge on the RNG.
- ZERO STORE LINES: probes are built from the frozen sanity anchors and
  the eval topics module (all synthetic, ADR 0003 R2).
- FEATURES AT RUN TIME: the JSONL carries text pairs + assigned cosine +
  label — NEVER vectors; every scoring pass computes features through
  ``cortex.features.pair.features`` (the #480 rule, eval-methodology §3).
- FINGERPRINT: ``eval_set_sha256`` follows data-contract §5 — sha256 of
  canonical JSON per pair over ``{record, candidate, similarity, label}``
  (the label and the assigned cosine are probe SEMANTICS here, hence
  part of the fingerprinted object — an eval-set-specific superset of
  the §3 object, documented in artifacts/manifests/layer-a-evalsets.md),
  manifest lines ``pair_id <sha>`` sorted by pair_id, BLAKE2b-256 over
  the manifest bytes.
- LLM SLOTS: translation twins / paraphrase / near-topic arrive from the
  LA-2 LLM batch as a JSONL file; the generator works without it (the
  slots are honest zero-count classes in v1) and picks rows up once the
  file exists. A frozen set never changes in place — filling slots
  produces a NEW set version with a NEW fingerprint and a manifest
  addendum.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cortex.data.fingerprints import (
    canonical_json,
    corpus_fingerprint,
    manifest_bytes,
    pair_sha256,
)
from cortex.eval.sanity import COS_LADDER, NEAR_BOUNDARY_COS, UNRELATED_COS
from cortex.evalsets.taxonomy import (
    ALL_PROBE_CLASSES,
    LLM_SLOT_PROBE_CLASSES,
    PROBE_CLASS_DEGENERATE,
    PROBE_CLASS_FAR_NEGATIVE,
    PROBE_CLASS_IDENTITY_SELF,
    PROBE_CLASS_MONOTONICITY,
    PROBE_CLASS_NEAR_IDENTITY,
    PROBE_CLASS_PAIR_SYMMETRY,
    PROBE_CLASS_TYPE_LANG_MISMATCH,
    SOURCE_LLM_LA2,
    SOURCE_PROCEDURAL,
)
from cortex.evalsets.topics import anchor_keys, anchor_records
from cortex.features.pair import PairRecord
from cortex.pretrain.corruption import (
    CollapseWhitespaceBody,
    FlipLanguage,
    FlipRecordType,
    NormalizePunctuationBody,
    ShuffleTags,
    SwapCaseTitle,
)

__all__ = [
    "DEFAULT_SEED",
    "DUPLICATE_THRESHOLD_PROBABILITY",
    "EVAL_SET_SCHEMA_VERSION",
    "FAR_COSINES",
    "LABEL_DUPLICATE",
    "LABEL_NOT_DUPLICATE",
    "MERGE_V1",
    "RELEASE_V1",
    "EvalProbe",
    "EvalSet",
    "EvalSetError",
    "EvalSetRecipe",
    "build_eval_set",
    "eval_set_jsonl",
    "eval_set_manifest_bytes",
    "eval_set_sha256",
    "load_eval_set",
    "load_llm_slots",
    "per_class_counts",
    "probe_row",
]

#: Master seed of the procedural generators (fixed — wave LA-1 pin).
DEFAULT_SEED: Final[int] = 20261004

#: JSON schema version stamped into set meta files.
EVAL_SET_SCHEMA_VERSION: Final[int] = 1

LABEL_DUPLICATE: Final[str] = "duplicate"
LABEL_NOT_DUPLICATE: Final[str] = "not-duplicate"

#: Cosine rotation for far-negative probes: anchored on the #480 unrelated
#: point (0.578) with spread around the runner's ceiling side.
FAR_COSINES: Final[tuple[float, ...]] = (0.578, 0.60, 0.55, 0.52)

#: The report's binarization cut — THE SAME constant the holdout runner
#: uses (single scoring-cut convention across all Layer A instruments).
DUPLICATE_THRESHOLD_PROBABILITY: Final[float] = 0.5

_LADDER_DUPLICATE_MIN_COS: Final[float] = 0.95

#: Ladder labels follow the tests/test_eval_sanity.py convention:
#: cos ≥ 0.95 → duplicate, below → not-duplicate (labels by construction).


class EvalSetError(ValueError):
    """Eval-set integrity failure: bad schema, unknown class, fingerprint
    mismatch — the run is VOID, never warned through."""


# ── probe row ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalProbe:
    """One probe pair: content + assigned cosine + label by construction."""

    pair_id: str
    probe_class: str
    record: PairRecord
    candidate: PairRecord
    similarity: float  # the assigned measured-cosine input ([-1, 1])
    label: str  # LABEL_DUPLICATE | LABEL_NOT_DUPLICATE
    source: str  # SOURCE_PROCEDURAL | SOURCE_LLM_LA2
    group: str | None = None  # ladder/symmetry groups for gate evaluation


def _side_dict(record: PairRecord) -> dict:
    return {
        "title": record.title,
        "body": record.body,
        "tags": list(record.tags),
        "language": record.language,
        "record_type": record.record_type,
    }


def _record_like(side: dict) -> PairRecord:
    return PairRecord(
        title=str(side["title"]),
        body=str(side["body"]),
        tags=tuple(str(tag) for tag in side.get("tags", ())),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def fingerprint_object(probe: EvalProbe) -> dict:
    """The §5-fingerprinted object: probe semantics = content + cosine + label."""
    return {
        "record": _side_dict(probe.record),
        "candidate": _side_dict(probe.candidate),
        "similarity": probe.similarity,
        "label": probe.label,
    }


def probe_row(probe: EvalProbe) -> dict:
    """The canonical JSONL row (assignment fields ride OUTSIDE the fingerprint)."""
    row = {
        "pair_id": probe.pair_id,
        "class": probe.probe_class,
        "source": probe.source,
        "similarity": probe.similarity,
        "label": probe.label,
        "record": _side_dict(probe.record),
        "candidate": _side_dict(probe.candidate),
    }
    if probe.group is not None:
        row["group"] = probe.group
    return row


def eval_set_manifest_bytes(probes: Sequence[EvalProbe]) -> bytes:
    """``pair_id <pair_sha256>`` manifest, sorted by pair_id (§5.3)."""
    entries = [
        (probe.pair_id, pair_sha256(fingerprint_object(probe))) for probe in probes
    ]
    return manifest_bytes(entries)


def eval_set_sha256(probes: Sequence[EvalProbe]) -> str:
    """BLAKE2b-256 over the manifest bytes — the frozen-set fingerprint."""
    return corpus_fingerprint(eval_set_manifest_bytes(probes))


def eval_set_jsonl(probes: Sequence[EvalProbe]) -> str:
    """Canonical JSONL text (one probe per line, canonical JSON per row)."""
    return "".join(canonical_json(probe_row(p)) + "\n" for p in probes)


def per_class_counts(probes: Sequence[EvalProbe]) -> dict[str, int]:
    counts = {name: 0 for name in ALL_PROBE_CLASSES}
    for probe in probes:
        counts[probe.probe_class] += 1
    return counts


# ── deterministic procedural generators (per class) ───────────────────────────

#: Weak perturbations for near-identity twins — the W4c-positive family,
#: the same mechanics sanity.py uses (title + ``!``) plus the corruption
#: registry's meaning-preserving transforms. Index-rotated, not sampled.
_WEAK_VARIANTS: Final[tuple] = (
    None,  # the sanity idiom: title + "!"
    ShuffleTags(),
    CollapseWhitespaceBody(),
    NormalizePunctuationBody(),
    SwapCaseTitle(),
)


def _weak_twin(record: PairRecord, variant: int, rng: random.Random) -> PairRecord:
    """Mechanical meaning-preserving twin (near-identity side)."""
    impl = _WEAK_VARIANTS[variant % len(_WEAK_VARIANTS)]
    if impl is None:  # the sanity._perturbed idiom
        from dataclasses import replace

        return replace(record, title=f"{record.title}!")
    return impl.apply(record, rng)


def _identity_self_probes(anchors: Sequence[PairRecord], count: int) -> list[EvalProbe]:
    """Record vs itself, cos=1.0, duplicate — the #480 inversion class."""
    return [
        EvalProbe(
            pair_id=f"self-{i:04d}",
            probe_class=PROBE_CLASS_IDENTITY_SELF,
            record=anchor,
            candidate=anchor,
            similarity=1.0,
            label=LABEL_DUPLICATE,
            source=SOURCE_PROCEDURAL,
        )
        for i, anchor in enumerate(anchors[:count])
    ]


def _near_identity_probes(
    anchors: Sequence[PairRecord], count: int, seed: int
) -> list[EvalProbe]:
    """Anchor vs its weak twin at NEAR_BOUNDARY_COS — duplicate by construction."""
    rng = random.Random(seed)
    probes = []
    for i in range(count):
        anchor = anchors[i % len(anchors)]
        twin = _weak_twin(anchor, i + seed, rng)
        probes.append(
            EvalProbe(
                pair_id=f"near-{i:04d}",
                probe_class=PROBE_CLASS_NEAR_IDENTITY,
                record=anchor,
                candidate=twin,
                similarity=NEAR_BOUNDARY_COS,
                label=LABEL_DUPLICATE,
                source=SOURCE_PROCEDURAL,
            )
        )
    return probes


def _far_pairs(
    anchors: Sequence[PairRecord], keys: Sequence[str]
) -> list[tuple[int, int]]:
    """Deterministic enumeration of different-topic anchor pairs (i < j).

    Same-key partners are REFUSED (translation-twin guard) — a RU anchor
    and its EN half are the same memory and must never pair as negative.
    """
    pairs = []
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            if keys[i] != keys[j]:
                pairs.append((i, j))
    return pairs


def _symmetry_probes(
    anchors: Sequence[PairRecord],
    keys: Sequence[str],
    base_pairs: int,
    seed: int,
) -> list[EvalProbe]:
    """Every base pair in BOTH directions — P(x,y) must equal P(y,x).

    Base pairs alternate weak-twin duplicates (cos 0.99) and cross-topic
    negatives (the #480 unrelated cosine) so the symmetry pin holds on
    both label surfaces.
    """
    rng = random.Random(seed + 1)
    far = _far_pairs(anchors, keys)
    probes: list[EvalProbe] = []
    for k in range(base_pairs):
        if k % 2 == 0:
            anchor = anchors[k % len(anchors)]
            side_b = _weak_twin(anchor, k + seed, rng)
            similarity, label = NEAR_BOUNDARY_COS, LABEL_DUPLICATE
        else:
            i, j = far[(k // 2) % len(far)]
            anchor, side_b = anchors[i], anchors[j]
            similarity, label = UNRELATED_COS, LABEL_NOT_DUPLICATE
        group = f"sym-{k:04d}"
        for direction, (a, b) in enumerate(((anchor, side_b), (side_b, anchor))):
            probes.append(
                EvalProbe(
                    pair_id=f"sym-{k:04d}-{'fwd' if direction == 0 else 'rev'}",
                    probe_class=PROBE_CLASS_PAIR_SYMMETRY,
                    record=a,
                    candidate=b,
                    similarity=similarity,
                    label=label,
                    source=SOURCE_PROCEDURAL,
                    group=group,
                )
            )
    return probes


def _ladder_probes(
    anchors: Sequence[PairRecord], base_pairs: int, seed: int
) -> list[EvalProbe]:
    """Fixed pairs re-scored along COS_LADDER — the monotonicity surface."""
    rng = random.Random(seed + 2)
    probes: list[EvalProbe] = []
    for k in range(base_pairs):
        anchor = anchors[(k * 5) % len(anchors)]
        twin = _weak_twin(anchor, k + seed, rng)
        group = f"lad-{k:04d}"
        for i, cos in enumerate(COS_LADDER):
            label = (
                LABEL_DUPLICATE
                if cos >= _LADDER_DUPLICATE_MIN_COS
                else (LABEL_NOT_DUPLICATE)
            )
            probes.append(
                EvalProbe(
                    pair_id=f"lad-{k:04d}-p{i}",
                    probe_class=PROBE_CLASS_MONOTONICITY,
                    record=anchor,
                    candidate=twin,
                    similarity=float(cos),
                    label=label,
                    source=SOURCE_PROCEDURAL,
                    group=group,
                )
            )
    return probes


def _far_negative_probes(
    anchors: Sequence[PairRecord], keys: Sequence[str], count: int
) -> list[EvalProbe]:
    """Different-topic pairs at low cosine — duplicate must stay OFF."""
    far = _far_pairs(anchors, keys)
    if count > len(far):
        raise EvalSetError(
            f"far-negative quota {count} exceeds the distinct cross-topic "
            f"pair pool ({len(far)}) — lower the quota or extend the topics"
        )
    return [
        EvalProbe(
            pair_id=f"far-{n:04d}",
            probe_class=PROBE_CLASS_FAR_NEGATIVE,
            record=anchors[i],
            candidate=anchors[j],
            similarity=FAR_COSINES[n % len(FAR_COSINES)],
            label=LABEL_NOT_DUPLICATE,
            source=SOURCE_PROCEDURAL,
        )
        for n, (i, j) in enumerate(far[:count])
    ]


def _type_lang_mismatch_probes(
    anchors: Sequence[PairRecord], count: int, seed: int
) -> list[EvalProbe]:
    """Same text, one metadata field broken — the type/lang tail of §7.

    Cosine 1.0 (self-pair convention: the text is identical, the vector
    leg measures it as such — corruption.py docstring); the label is
    not-duplicate: a broken field is the hard-negative signal (W4c).
    """
    rng = random.Random(seed + 3)
    probes = []
    for i in range(count):
        anchor = anchors[i % len(anchors)]
        transform = FlipRecordType() if i % 2 == 0 else FlipLanguage()
        broken = transform.apply(anchor, rng)
        probes.append(
            EvalProbe(
                pair_id=f"tml-{i:04d}",
                probe_class=PROBE_CLASS_TYPE_LANG_MISMATCH,
                record=anchor,
                candidate=broken,
                similarity=1.0,
                label=LABEL_NOT_DUPLICATE,
                source=SOURCE_PROCEDURAL,
            )
        )
    return probes


_DEGENERATE_TITLES: Final[tuple[str, str, str]] = (
    "Вырожденная запись: пустое тело",
    "Вырожденная запись: один токен",
    "Вырожденная запись: очень длинное тело",
)


def _degenerate_records() -> tuple[PairRecord, PairRecord, PairRecord]:
    """The three degenerate variants (eval-methodology §7: не покрыто → хвост)."""
    return (
        PairRecord(
            title=_DEGENERATE_TITLES[0],
            body="",
            tags=("degenerate",),
            language="ru",
            record_type="note",
        ),
        PairRecord(
            title=_DEGENERATE_TITLES[1],
            body="заметка",
            tags=("degenerate",),
            language="ru",
            record_type="note",
        ),
        PairRecord(
            title=_DEGENERATE_TITLES[2],
            body="фрагмент длинного текста про наблюдения " * 1000,
            tags=("degenerate",),
            language="ru",
            record_type="note",
        ),
    )


def _degenerate_probes(anchors: Sequence[PairRecord]) -> list[EvalProbe]:
    """Three degenerate variants × two shapes: vs itself (identity must
    hold even on degenerate content) and vs a normal far anchor."""
    unrelated = anchors[1]
    probes: list[EvalProbe] = []
    for v, degenerate in enumerate(_degenerate_records()):
        probes.append(
            EvalProbe(
                pair_id=f"degen-{v:04d}-self",
                probe_class=PROBE_CLASS_DEGENERATE,
                record=degenerate,
                candidate=degenerate,
                similarity=1.0,
                label=LABEL_DUPLICATE,
                source=SOURCE_PROCEDURAL,
            )
        )
        probes.append(
            EvalProbe(
                pair_id=f"degen-{v:04d}-far",
                probe_class=PROBE_CLASS_DEGENERATE,
                record=degenerate,
                candidate=unrelated,
                similarity=0.5,
                label=LABEL_NOT_DUPLICATE,
                source=SOURCE_PROCEDURAL,
            )
        )
    return probes


# ── recipes: the two frozen sets (eval-methodology §3) ────────────────────────


@dataclass(frozen=True)
class EvalSetRecipe:
    """Per-class quotas of one frozen set (the degenerate block is fixed 6)."""

    set_id: str
    role: str  # "merge" | "release"
    identity_self: int
    near_identity: int
    symmetry_bases: int  # each base emits fwd + rev
    ladder_bases: int  # each base emits len(COS_LADDER) rows
    far_negative: int
    type_lang: int

    @property
    def expected_pairs(self) -> int:
        return (
            self.identity_self
            + self.near_identity
            + self.symmetry_bases * 2
            + self.ladder_bases * len(COS_LADDER)
            + self.far_negative
            + self.type_lang
            + 6
        )


#: eval-merge ~60 pairs — quick gate on merge (eval-methodology §3).
MERGE_V1: Final[EvalSetRecipe] = EvalSetRecipe(
    set_id="merge-v1",
    role="merge",
    identity_self=8,
    near_identity=8,
    symmetry_bases=5,
    ladder_bases=2,
    far_negative=10,
    type_lang=8,
)  # 8 + 8 + 10 + 10 + 10 + 8 + 6 = 60

#: eval-release ≥ 200 pairs — stratified gate at release (BA corridor).
RELEASE_V1: Final[EvalSetRecipe] = EvalSetRecipe(
    set_id="release-v1",
    role="release",
    identity_self=18,
    near_identity=18,
    symmetry_bases=14,
    ladder_bases=7,
    far_negative=60,
    type_lang=36,
)  # 18 + 18 + 28 + 35 + 60 + 36 + 6 = 201


# ── set assembly ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvalSet:
    """A frozen eval set: probes + the derived fingerprint + role."""

    set_id: str
    role: str
    probes: tuple[EvalProbe, ...]
    eval_set_sha256: str

    @property
    def per_class(self) -> dict[str, int]:
        return per_class_counts(self.probes)


def _llm_slot_probes(rows: Sequence[dict]) -> list[EvalProbe]:
    """Validate + convert LA-2 LLM-batch rows (schema identical to probes)."""
    probes: list[EvalProbe] = []
    for n, row in enumerate(rows):
        probe = _row_to_probe(row, origin=f"llm-slots[{n}]")
        if probe.probe_class not in LLM_SLOT_PROBE_CLASSES:
            raise EvalSetError(
                f"llm-slots[{n}]: class {probe.probe_class!r} is not an LLM-slot "
                f"class ({list(LLM_SLOT_PROBE_CLASSES)}) — procedural classes are "
                "generated, never accepted from outside"
            )
        if probe.source != SOURCE_LLM_LA2:
            raise EvalSetError(
                f"llm-slots[{n}]: source must be {SOURCE_LLM_LA2!r}, "
                f"got {probe.source!r}"
            )
        probes.append(probe)
    return probes


def _row_to_probe(row: dict, *, origin: str) -> EvalProbe:
    """Validate one JSONL row against the probe schema — fail loud, no repair."""
    for key in (
        "pair_id",
        "class",
        "source",
        "similarity",
        "label",
        "record",
        "candidate",
    ):
        if key not in row:
            raise EvalSetError(f"{origin}: missing required key {key!r}")
    if row["class"] not in ALL_PROBE_CLASSES:
        raise EvalSetError(
            f"{origin}: unknown probe class {row['class']!r} — "
            f"allowed: {list(ALL_PROBE_CLASSES)}"
        )
    if row["label"] not in (LABEL_DUPLICATE, LABEL_NOT_DUPLICATE):
        raise EvalSetError(f"{origin}: bad label {row['label']!r}")
    similarity = row["similarity"]
    if not isinstance(similarity, (int, float)) or isinstance(similarity, bool):
        raise EvalSetError(f"{origin}: similarity must be a number")
    if not -1.0 <= float(similarity) <= 1.0:
        raise EvalSetError(f"{origin}: similarity {similarity} outside [-1, 1]")
    for side_key in ("record", "candidate"):
        side = row[side_key]
        if not isinstance(side, dict) or "title" not in side or "body" not in side:
            raise EvalSetError(f"{origin}: {side_key} must carry title and body")
    group = row.get("group")
    return EvalProbe(
        pair_id=str(row["pair_id"]),
        probe_class=str(row["class"]),
        record=_record_like(row["record"]),
        candidate=_record_like(row["candidate"]),
        similarity=float(similarity),
        label=str(row["label"]),
        source=str(row["source"]),
        group=str(group) if group is not None else None,
    )


def build_eval_set(
    recipe: EvalSetRecipe,
    *,
    seed: int = DEFAULT_SEED,
    llm_slots: Sequence[dict] | None = None,
) -> EvalSet:
    """Assemble one eval set from its recipe — pure, deterministic.

    ``llm_slots``: parsed JSONL rows of the LA-2 LLM batch (translation
    twins / paraphrase / near-topic). None or empty → the v1 shape with
    honest zero-count slot classes.
    """
    anchors = anchor_records()
    keys = anchor_keys()
    probes: list[EvalProbe] = [
        *_identity_self_probes(anchors, recipe.identity_self),
        *_near_identity_probes(anchors, recipe.near_identity, seed),
        *_symmetry_probes(anchors, keys, recipe.symmetry_bases, seed),
        *_ladder_probes(anchors, recipe.ladder_bases, seed),
        *_far_negative_probes(anchors, keys, recipe.far_negative),
        *_type_lang_mismatch_probes(anchors, recipe.type_lang, seed),
        *_degenerate_probes(anchors),
    ]
    if llm_slots:
        probes.extend(_llm_slot_probes(llm_slots))

    pair_ids = [probe.pair_id for probe in probes]
    if len(set(pair_ids)) != len(pair_ids):
        duplicates = sorted({pid for pid in pair_ids if pair_ids.count(pid) > 1})
        raise EvalSetError(f"duplicate pair ids in assembled set: {duplicates[:5]}")
    if recipe.role == "release" and len(probes) < 200:
        raise EvalSetError(
            f"release recipe {recipe.set_id!r} yields {len(probes)} pairs — "
            "the release gate requires ≥ 200 (eval-methodology §3)"
        )
    return EvalSet(
        set_id=recipe.set_id,
        role=recipe.role,
        probes=tuple(probes),
        eval_set_sha256=eval_set_sha256(probes),
    )


# ── frozen-set loading (runner side; re-verifies the fingerprint) ─────────────

_SET_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def load_llm_slots(path: Path) -> list[dict]:
    """Parse the LA-2 slots JSONL; missing file → empty list (honest v1)."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise EvalSetError(
                f"llm slots {path}[{n}]: not valid JSON — {exc}"
            ) from exc
    return rows


def load_eval_set(
    jsonl_path: Path,
    *,
    meta_path: Path | None = None,
) -> EvalSet:
    """Load a FROZEN eval set and re-verify its fingerprint (§5.6 discipline:
    a mismatch voids the run — never a warning).

    ``meta_path`` defaults to the sibling ``<stem>.meta.json``; it carries
    set_id/role and the pinned ``eval_set_sha256``. A set-id must be a
    machine string; role must be merge|release.
    """
    jsonl_path = Path(jsonl_path)
    if meta_path is None:
        # merge-v1.jsonl → merge-v1.meta.json (sibling, stem-suffixed)
        meta_path = jsonl_path.parent / (jsonl_path.stem + ".meta.json")
    if not meta_path.is_file():
        raise EvalSetError(f"set meta not found: {meta_path}")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalSetError(f"set meta is not valid JSON: {meta_path} — {exc}") from exc
    for key in ("set_id", "role", "eval_set_sha256"):
        if key not in meta:
            raise EvalSetError(f"set meta missing {key!r}: {meta_path}")
    role = str(meta["role"])
    if role not in ("merge", "release"):
        raise EvalSetError(f"set meta role must be merge|release, got {role!r}")
    set_id = str(meta["set_id"])
    if not _SET_ID_RE.fullmatch(set_id):
        raise EvalSetError(f"set id must be a machine string, got {set_id!r}")

    probes: list[EvalProbe] = []
    for n, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalSetError(f"{jsonl_path}[{n}]: not valid JSON — {exc}") from exc
        probes.append(_row_to_probe(row, origin=f"{jsonl_path.name}[{n}]"))
    if not probes:
        raise EvalSetError(f"{jsonl_path}: the frozen set is empty")

    actual = eval_set_sha256(probes)
    if actual != str(meta["eval_set_sha256"]):
        raise EvalSetError(
            f"{jsonl_path}: eval_set_sha256 mismatch — recomputed {actual} vs "
            f"pinned {meta['eval_set_sha256']} (the set was edited after "
            "freezing: data-contract §5.6 — the run is VOID)"
        )
    seen: set[str] = set()
    for probe in probes:
        if probe.pair_id in seen:
            raise EvalSetError(f"{jsonl_path}: duplicate pair_id {probe.pair_id!r}")
        seen.add(probe.pair_id)
    return EvalSet(
        set_id=set_id,
        role=role,
        probes=tuple(probes),
        eval_set_sha256=actual,
    )
