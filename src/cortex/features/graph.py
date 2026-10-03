"""Graph-evidence sidecar (B2) — reader and the gated graph feature block.

Wave B2 of the v2 duty program: is-duplicate revision 1.n gains a GRAPH
feature block (ADR 0002 — the block is ADDITIVE; the precedent is the A2
field-cosines sidecar of :mod:`cortex.features.pair`). The evidence itself
is docomputed OUTSIDE this library by ``scripts/b2_graph_sidecar.py``
(read-only engine store: memory_edges + vectors.db) — the library carries
no store access, exactly like it carries no embedder for the field
cosines. The sidecar artifact (``data/stage2/b2/graph-sidecar.jsonl``,
gitignored) holds one row per dataset pair:

``{pair_id, stratum, label, a_store_id, b_store_id, edge_ab, edge_kinds,
common_neighbors, neighbor_jaccard, cos_to_common_max?, cos_to_common_mean?,
has_graph_evidence}``

Gating discipline (inference-v1.md §2 variant-N precedent, §4 metadata):
the DEFAULT frozen contract is untouched — :data:`FEATURE_NAMES` stays 13
names, and every consumer that does not ask for graph evidence sees byte
identical behavior. The five graph features are appended ONLY when the
trainer/exporter opts in, via :func:`attach_graph_features`; the gated
composition is ``FEATURE_NAMES [+ FIELD_COSINE_FEATURES] + GRAPH_FEATURES``
(field cosines first, graph last — a single composition order), and
``cortex.artifacts.features_digest`` over the joined names yields the new
``feature_set_sha256`` the artifact pins into metadata_props.

Evidence-absence semantics (deliberately different from the text-Jaccard
convention of pair.py): zero common neighborhoods / no edge mean NO graph
evidence, not identity — graph-empty pairs (sides outside the store, edge
-less records) are legitimate and carry zeros plus
``has_graph_evidence=false``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cortex.features.pair import FEATURE_NAMES, FIELD_COSINE_FEATURES, FeatureVector

__all__ = [
    "GRAPH_FEATURES",
    "GraphEvidence",
    "GraphSidecar",
    "GraphSidecarError",
    "attach_graph_features",
    "attach_to_features",
]

#: Gated graph block (appended AFTER the core/field blocks, in this order).
#: - ``edge_ab``              — 1.0 when a direct memory_edges row connects
#:                              the two sides in ANY direction;
#: - ``edge_kind_supersedes`` — 1.0 when that direct edge is (or includes)
#:                              ``supersedes`` (0.0 for relates_to / no edge);
#: - ``common_neighbors``     — |N(a) ∩ N(b)|, 1-hop undirected;
#: - ``neighbor_jaccard``     — |∩|/|∪| of the neighborhoods, 0.0 when empty;
#: - ``cos_to_common_max``    — max cosine(side vector, common-neighbor
#:                              vector), 0.0 when no common neighbors or no
#:                              store vectors.
GRAPH_FEATURES: Final[tuple[str, ...]] = (
    "edge_ab",
    "edge_kind_supersedes",
    "common_neighbors",
    "neighbor_jaccard",
    "cos_to_common_max",
)


class GraphSidecarError(ValueError):
    """The graph sidecar is missing, malformed, or lacks a requested pair."""


@dataclass(frozen=True)
class GraphEvidence:
    """Validated graph evidence for ONE dataset pair (sidecar row)."""

    edge_ab: bool = False
    edge_kind_supersedes: bool = False
    common_neighbors: int = 0
    neighbor_jaccard: float = 0.0
    cos_to_common_max: float | None = None
    cos_to_common_mean: float | None = None

    @property
    def has_graph_evidence(self) -> bool:
        """True when the pair carries ANY positive graph signal."""
        return self.edge_ab or self.common_neighbors > 0

    @classmethod
    def zero(cls) -> GraphEvidence:
        """The legitimate graph-empty evidence (no store identity / no edges)."""
        return cls()

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> GraphEvidence:
        """Validate one sidecar JSON row into evidence.

        Raises:
            GraphSidecarError: missing/mistyped required fields, values out
                of range, or an inconsistent ``has_graph_evidence`` flag
                (a corrupted row refuses the whole read — fail-loud).
        """
        for key in ("edge_ab", "common_neighbors", "neighbor_jaccard"):
            if key not in row:
                raise GraphSidecarError(f"sidecar row misses required key {key!r}")

        edge_ab = row["edge_ab"]
        if not isinstance(edge_ab, bool):
            raise GraphSidecarError(f"edge_ab must be boolean, got {edge_ab!r}")

        kinds = row.get("edge_kinds", [])
        if not isinstance(kinds, list) or any(not isinstance(k, str) for k in kinds):
            raise GraphSidecarError(
                f"edge_kinds must be a list of strings, got {kinds!r}"
            )
        if bool(kinds) != edge_ab:
            raise GraphSidecarError(
                f"edge_ab={edge_ab} contradicts edge_kinds={kinds!r}"
            )
        edge_kind_supersedes = "supersedes" in kinds

        common = row["common_neighbors"]
        if isinstance(common, bool) or not isinstance(common, int) or common < 0:
            raise GraphSidecarError(
                f"common_neighbors must be a non-negative int, got {common!r}"
            )

        jaccard = _bounded_float(row, "neighbor_jaccard", 0.0, 1.0)
        cos_max = _optional_cosine(row, "cos_to_common_max")
        cos_mean = _optional_cosine(row, "cos_to_common_mean")

        evidence = cls(
            edge_ab=edge_ab,
            edge_kind_supersedes=edge_kind_supersedes,
            common_neighbors=common,
            neighbor_jaccard=jaccard,
            cos_to_common_max=cos_max,
            cos_to_common_mean=cos_mean,
        )
        declared = row.get("has_graph_evidence")
        if declared is not None and bool(declared) != evidence.has_graph_evidence:
            raise GraphSidecarError(
                f"has_graph_evidence={declared!r} contradicts the row payload"
            )
        return evidence


def _bounded_float(
    row: Mapping[str, object], key: str, low: float, high: float
) -> float:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraphSidecarError(f"{key} must be a real number, got {value!r}")
    result = float(value)
    if math.isnan(result) or result < low or result > high:
        raise GraphSidecarError(f"{key} must be within [{low}, {high}], got {result!r}")
    return result


def _optional_cosine(row: Mapping[str, object], key: str) -> float | None:
    if key not in row or row[key] is None:
        return None
    return _bounded_float(row, key, -1.0, 1.0)


class GraphSidecar:
    """Reader over the B2 graph-evidence sidecar jsonl (pair_id → evidence)."""

    def __init__(self, path: str | Path) -> None:
        sidecar_path = Path(path)
        if not sidecar_path.is_file():
            raise GraphSidecarError(f"graph sidecar not found: {sidecar_path}")
        self._index: dict[str, GraphEvidence] = {}
        for line_no, line in enumerate(
            sidecar_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GraphSidecarError(
                    f"graph sidecar {sidecar_path} line {line_no}: bad JSON ({exc})"
                ) from exc
            if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str):
                raise GraphSidecarError(
                    f"graph sidecar {sidecar_path} line {line_no}: no string pair_id"
                )
            pair_id = row["pair_id"]
            if pair_id in self._index:
                raise GraphSidecarError(
                    f"graph sidecar {sidecar_path}: duplicate pair_id {pair_id!r}"
                )
            try:
                self._index[pair_id] = GraphEvidence.from_row(row)
            except GraphSidecarError as exc:
                raise GraphSidecarError(
                    f"graph sidecar {sidecar_path} line {line_no}: {exc}"
                ) from exc

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, pair_id: str) -> bool:
        return pair_id in self._index

    def evidence_for(self, pair_id: str) -> GraphEvidence | None:
        """Evidence for one pair, or None when the sidecar does not cover it.

        A missing pair is NOT auto-zeroed here: coverage gaps are a manifest
        mismatch and the caller decides loudly (see
        :func:`attach_to_features`).
        """
        return self._index.get(pair_id)


def _attach_names(vector: FeatureVector) -> tuple[str, ...]:
    """Allowed prefixes for the graph block: core or core+field-cosines."""
    if vector.names == FEATURE_NAMES:
        return FEATURE_NAMES
    if vector.names == FEATURE_NAMES + FIELD_COSINE_FEATURES:
        return vector.names
    if (
        len(vector.names) >= len(GRAPH_FEATURES)
        and vector.names[-len(GRAPH_FEATURES) :] == GRAPH_FEATURES
    ):
        raise ValueError(
            "graph features are already attached to this vector (attach once)"
        )
    raise ValueError(
        "graph features attach to the CORE or core+field-cosines vector only "
        f"(got {len(vector.names)} names: {vector.names[:3]}…)"
    )


def attach_graph_features(
    vector: FeatureVector, evidence: GraphEvidence
) -> FeatureVector:
    """Append the gated graph block to a core / core+field feature vector.

    Values (contract of :data:`GRAPH_FEATURES`): ``edge_ab`` and
    ``edge_kind_supersedes`` map to 1.0/0.0 (a supersedes flag without an
    edge is a contract violation), ``cos_to_common_max`` falls back to 0.0
    when the sidecar omitted it (no common-neighbor vectors). The result's
    ``names`` is what the gated artifact pins into ``features`` /
    ``feature_set_sha256`` metadata (cortex.artifacts.features_digest).
    """
    base_names = _attach_names(vector)
    if evidence.edge_kind_supersedes and not evidence.edge_ab:
        raise ValueError(
            "edge_kind_supersedes without edge_ab is inconsistent evidence"
        )
    cos_max = evidence.cos_to_common_max
    if cos_max is not None and (math.isnan(cos_max) or not -1.0 <= cos_max <= 1.0):
        raise ValueError(f"cos_to_common_max must be within [-1, 1], got {cos_max!r}")
    jaccard = evidence.neighbor_jaccard
    if math.isnan(jaccard) or not 0.0 <= jaccard <= 1.0:
        raise ValueError(f"neighbor_jaccard must be within [0, 1], got {jaccard!r}")
    if evidence.common_neighbors < 0:
        raise ValueError(
            f"common_neighbors must be non-negative, got {evidence.common_neighbors!r}"
        )

    values = (
        1.0 if evidence.edge_ab else 0.0,
        1.0 if evidence.edge_kind_supersedes else 0.0,
        float(evidence.common_neighbors),
        jaccard,
        0.0 if cos_max is None else cos_max,
    )
    return FeatureVector(
        names=base_names + GRAPH_FEATURES,
        values=vector.values + values,
    )


def attach_to_features(
    vector: FeatureVector,
    sidecar: GraphSidecar,
    pair_id: str,
    *,
    on_missing: str = "raise",
) -> FeatureVector:
    """Attach the sidecar evidence of ``pair_id`` to a feature vector.

    ``on_missing="raise"`` (default) refuses pairs the sidecar does not
    cover — a train-manifest/sidecar mismatch must be loud. With
    ``on_missing="zero"`` uncovered pairs attach the legitimate graph-empty
    evidence (:meth:`GraphEvidence.zero`) — for corpora where graph-empty
    rows are expected by construction.
    """
    evidence = sidecar.evidence_for(pair_id)
    if evidence is None:
        if on_missing == "zero":
            evidence = GraphEvidence.zero()
        elif on_missing == "raise":
            raise GraphSidecarError(
                f"pair {pair_id!r} is not covered by the graph sidecar"
            )
        else:
            raise ValueError(
                f"on_missing must be 'raise' or 'zero', got {on_missing!r}"
            )
    return attach_graph_features(vector, evidence)
