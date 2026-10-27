#!/usr/bin/env python3
"""Shared feature-matrix library for the B2 rev 1.n slice (prereg v3 §3).

The composite contract of revision 1.n (ADR 0002, gated graph block):
``FEATURE_NAMES + GRAPH_FEATURES`` — 13 frozen pair features + 5 graph
features. Digest: ``cortex.artifacts.features_digest`` over the joined
names → the ``feature_set_sha256`` the artifact pins.

Evidence sourcing per row (frozen by the task charter, three disjoint
categories — every row lands in EXACTLY one):

- ``pool``     — the graph strata (edge-direct / edge-common / mint-top /
                 control): the FROZEN pool columns ride verbatim
                 (``edge`` → edge_ab + supersedes kind, ``common_neighbors``,
                 ``neighbor_jaccard``). ``cos_to_common_max`` is NOT among
                 the frozen columns → 0.0 (the contract's omitted-value
                 fallback). The live store is never consulted for these —
                 the store serves a live engine and would break determinism.
- ``zero``     — the T0/T1 injection (``generator`` column present): no
                 evidence BY CONSTRUCTION (synthetic sides carry no store
                 identity; the sidecar's self-pair guard would zero them
                 anyway — asserted, not trusted).
- ``sidecar``  — every other row: graph evidence through the frozen
                 sidecar snapshot built by ``scripts/b2_graph_sidecar.py``
                 over the read-only store (store-backed rows get real
                 evidence; content-external sides get legitimate zeros
                 inside the sidecar itself).

No secrets, no network: the sidecar is a file; the store is touched only
by the sidecar script, once, before this module ever runs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.features.graph import (  # noqa: E402
    GRAPH_FEATURES,
    GraphEvidence,
    GraphSidecar,
    GraphSidecarError,
    attach_graph_features,
)
from cortex.features.pair import (  # noqa: E402
    FEATURE_NAMES,
    FeatureVector,
    RecordLike,
    features,
)

__all__ = [
    "ABLATED_NAMES",
    "EvidenceKind",
    "GATED_NAMES",
    "GRAPH_STRATA",
    "build_dataset",
    "features_digest_of",
    "load_jsonl",
    "row_evidence",
]

#: The gated composite of revision 1.n (13 + 5) and the ablation twin.
GATED_NAMES: tuple[str, ...] = FEATURE_NAMES + GRAPH_FEATURES
ABLATED_NAMES: tuple[str, ...] = FEATURE_NAMES

#: Pool graph strata (prereg v3 §2 quotas) — frozen columns live in-row.
GRAPH_STRATA: frozenset[str] = frozenset(
    {"edge-direct", "edge-common", "mint-top", "control"}
)


class EvidenceKind:
    """Disjoint evidence-sourcing categories (see module docstring)."""

    POOL = "pool"
    ZERO = "zero"
    SIDECAR = "sidecar"


def load_jsonl(path: str | Path) -> list[dict]:
    """Read a jsonl file strictly (every line a JSON object)."""
    rows: list[dict] = []
    for line_no, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: not a JSON object")
        rows.append(row)
    return rows


def features_digest_of(names: tuple[str, ...]) -> str:
    """feature_set_sha256 over the joined feature names."""
    from cortex.artifacts import features_digest

    return features_digest(names)


def _record_like(side: dict) -> RecordLike:
    """Mirror RecordLike from a corpus row side (exact fields only)."""
    return RecordLike(
        title=str(side.get("title", "")),
        body=str(side.get("body", "")),
        tags=tuple(str(t) for t in side.get("tags") or []),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def evidence_kind(row: dict) -> str:
    """The disjoint category of one corpus row (first match wins, and the
    categories are mutually exclusive by corpus construction)."""
    stratum = str(row.get("stratum", ""))
    if stratum in GRAPH_STRATA:
        return EvidenceKind.POOL
    if "generator" in row:
        return EvidenceKind.ZERO
    return EvidenceKind.SIDECAR


def _pool_evidence(row: dict) -> GraphEvidence:
    """Frozen pool columns → evidence (cos_to_common_max not frozen → None)."""
    edge = row.get("edge")
    if edge is not None and not isinstance(edge, dict):
        raise ValueError(f"pool row {row.get('pair_id')!r}: edge must be dict|null")
    kinds = [str(edge["kind"])] if isinstance(edge, dict) and edge.get("kind") else []
    return GraphEvidence(
        edge_ab=bool(kinds) or edge is not None,
        edge_kind_supersedes="supersedes" in kinds,
        common_neighbors=int(row.get("common_neighbors", 0)),
        neighbor_jaccard=float(row.get("neighbor_jaccard", 0.0)),
        cos_to_common_max=None,
    )


def row_evidence(row: dict, sidecar: GraphSidecar | None) -> tuple[GraphEvidence, str]:
    """(evidence, kind) for one corpus row — fail-loud on coverage gaps."""
    kind = evidence_kind(row)
    if kind == EvidenceKind.POOL:
        return _pool_evidence(row), kind
    if kind == EvidenceKind.ZERO:
        return GraphEvidence.zero(), kind
    if sidecar is None:
        raise GraphSidecarError("sidecar rows present but no sidecar supplied")
    evidence = sidecar.evidence_for(str(row["pair_id"]))
    if evidence is None:
        raise GraphSidecarError(
            f"pair {row['pair_id']!r} is not covered by the graph sidecar"
        )
    return evidence, kind


def build_dataset(
    rows: list[dict],
    sidecar: GraphSidecar | None,
    names: tuple[str, ...],
    *,
    zero_cross_check: bool = True,
) -> tuple[list[FeatureVector], list[int], list[dict]]:
    """Feature matrix + binary labels + per-row provenance for a corpus.

    ``names`` is GATED_NAMES (full model) or ABLATED_NAMES (ablation) —
    the 13-name ablation simply never attaches the graph block.

    ``zero_cross_check``: every T0/T1 row the sidecar HAPPENS to cover must
    carry no evidence (the self-pair/content-external guards) — the
    by-construction claim is verified, not assumed.
    """
    gated = names == GATED_NAMES
    if not gated and names != ABLATED_NAMES:
        raise ValueError(f"unknown feature contract: {names!r}")
    vectors: list[FeatureVector] = []
    labels: list[int] = []
    provenance: list[dict] = []
    for row in rows:
        similarity = float(row["similarity"])
        base = features(
            _record_like(row["record"]),
            _record_like(row["candidate"]),
            similarity,
        )
        evidence, kind = row_evidence(row, sidecar)
        if (
            zero_cross_check
            and kind == EvidenceKind.ZERO
            and evidence.has_graph_evidence
        ):
            raise ValueError(
                f"injection row {row['pair_id']!r} carries graph evidence "
                "— violates the T0/T1 no-evidence-by-construction rule"
            )
        vector = attach_graph_features(base, evidence) if gated else base
        if vector.names != names:
            raise ValueError(
                f"row {row.get('pair_id')!r}: built {len(vector.names)} names, "
                f"contract demands {len(names)}"
            )
        label = 1 if str(row["label"]) == "duplicate" else 0
        vectors.append(vector)
        labels.append(label)
        provenance.append(
            {
                "pair_id": str(row["pair_id"]),
                "stratum": str(row.get("stratum", "")),
                "label": str(row["label"]),
                "evidence_kind": kind,
                "has_graph_evidence": evidence.has_graph_evidence,
            }
        )
    return vectors, labels, provenance


@dataclass(frozen=True)
class CorpusAsserts:
    """Re-asserted corpus identity numbers (fail-loud, prereg §1)."""

    rows: int
    dups: int
    not_dups: int
    train_fingerprint: str
    labels_fingerprint: str


def assert_corpus_identity(
    train_path: Path,
    labels_train_path: Path,
    manifest_path: Path,
    expected_train_fp: str,
    expected_labels_fp: str,
) -> CorpusAsserts:
    """Re-derive the corpus fingerprints from disk and compare to the
    sealed manifest — a mismatch voids the run (prereg: no post-hoc edits)."""
    from cortex.data.fingerprints import (
        corpus_fingerprint,
        labels_fingerprint,
        manifest_bytes,
        pair_sha256,
    )

    rows = load_jsonl(train_path)
    labels_rows = load_jsonl(labels_train_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = (
        (
            row["pair_id"],
            pair_sha256(
                {
                    "record": row["record"],
                    "candidate": row["candidate"],
                    "similarity": row["similarity"],
                }
            ),
        )
        for row in rows
    )
    train_fp = corpus_fingerprint(manifest_bytes(entries))
    labels_fp = labels_fingerprint({r["pair_id"]: str(r["label"]) for r in labels_rows})
    if train_fp != expected_train_fp:
        raise ValueError(
            f"train fingerprint mismatch: disk {train_fp} vs manifest {expected_train_fp}"
        )
    if labels_fp != expected_labels_fp:
        raise ValueError(
            f"labels fingerprint mismatch: disk {labels_fp} vs manifest {expected_labels_fp}"
        )
    counts = manifest["counts"]
    dups = sum(1 for r in rows if str(r["label"]) == "duplicate")
    if len(rows) != int(counts["train_rows"]) or dups != int(
        counts["train_labels"]["duplicate"]
    ):
        raise ValueError("train counts diverge from the sealed manifest")
    return CorpusAsserts(
        rows=len(rows),
        dups=dups,
        not_dups=len(rows) - dups,
        train_fingerprint=train_fp,
        labels_fingerprint=labels_fp,
    )
