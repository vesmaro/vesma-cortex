"""B2 graph-feature tests: sidecar reader, gated attach, freeze guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex.artifacts import features_digest
from cortex.features import (
    FEATURE_NAMES,
    FIELD_COSINE_FEATURES,
    GRAPH_FEATURES,
    GraphEvidence,
    GraphSidecar,
    GraphSidecarError,
    attach_graph_features,
)
from cortex.features.graph import attach_to_features
from cortex.features.pair import (
    FeatureVector,
    PairRecord,
    attach_field_cosines,
    features,
)

#: The frozen 13-name core digest (A3a contract). B2 must not touch it:
#: the graph block is append-only and gated — this literal is the tripwire.
CORE_FEATURE_SET_SHA256: str = (
    "dd86228f8c634f28d8a15b2d8279da1b99735d68e309be02d30f1c24698fa6af"
)

R_A = PairRecord("Заметка", "Тело записи", ("t",), "ru", "note")


def _core_vector() -> FeatureVector:
    return features(R_A, R_A, 0.5)


def _write_sidecar(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "graph-sidecar.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


EVIDENT_ROW: dict[str, object] = {
    "pair_id": "DV-N-001",
    "stratum": "template-twin",
    "label": "not-duplicate",
    "a_store_id": "id-a",
    "b_store_id": "id-b",
    "edge_ab": True,
    "edge_kinds": ["relates_to"],
    "common_neighbors": 3,
    "neighbor_jaccard": 0.25,
    "cos_to_common_max": 0.9,
    "cos_to_common_mean": 0.8,
    "has_graph_evidence": True,
}
EMPTY_ROW: dict[str, object] = {
    "pair_id": "DV-P-001",
    "stratum": "light",
    "label": "duplicate",
    "a_store_id": "id-a",
    "b_store_id": None,
    "edge_ab": False,
    "edge_kinds": [],
    "common_neighbors": 0,
    "neighbor_jaccard": 0.0,
    "has_graph_evidence": False,
}


# ── freeze guards: the default contract is untouched ─────────────────────────


def test_default_feature_list_is_unchanged() -> None:
    assert len(FEATURE_NAMES) == 13
    assert all(name not in FEATURE_NAMES for name in GRAPH_FEATURES)


def test_core_feature_set_digest_pinned() -> None:
    assert features_digest(FEATURE_NAMES) == CORE_FEATURE_SET_SHA256


def test_gated_digests_differ_from_core() -> None:
    core = features_digest(FEATURE_NAMES)
    with_graph = features_digest(FEATURE_NAMES + GRAPH_FEATURES)
    with_field_graph = features_digest(
        FEATURE_NAMES + FIELD_COSINE_FEATURES + GRAPH_FEATURES
    )
    assert with_graph != core
    assert with_field_graph != core
    assert with_field_graph != with_graph


# ── sidecar reader: valid / broken / evidence-absent ─────────────────────────


def test_reader_valid_rows(tmp_path: Path) -> None:
    path = _write_sidecar(tmp_path, [EVIDENT_ROW, EMPTY_ROW])
    sidecar = GraphSidecar(path)
    assert len(sidecar) == 2

    evident = sidecar.evidence_for("DV-N-001")
    assert evident is not None
    assert evident.edge_ab is True
    assert evident.edge_kind_supersedes is False  # relates_to only
    assert evident.common_neighbors == 3
    assert evident.neighbor_jaccard == 0.25
    assert evident.cos_to_common_max == 0.9
    assert evident.has_graph_evidence is True

    empty = sidecar.evidence_for("DV-P-001")
    assert empty is not None
    assert empty.has_graph_evidence is False
    assert empty == GraphEvidence.zero()

    assert sidecar.evidence_for("DV-N-404") is None


def test_reader_supersedes_kind_derived(tmp_path: Path) -> None:
    row = dict(EVIDENT_ROW, pair_id="X-1", edge_kinds=["supersedes", "relates_to"])
    sidecar = GraphSidecar(_write_sidecar(tmp_path, [row]))
    evident = sidecar.evidence_for("X-1")
    assert evident is not None and evident.edge_kind_supersedes is True


def test_reader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GraphSidecarError, match="not found"):
        GraphSidecar(tmp_path / "absent.jsonl")


@pytest.mark.parametrize(
    "mutation",
    [
        {"edge_ab": "yes"},  # non-bool
        {"edge_ab": True, "edge_kinds": []},  # kinds contradict the flag
        {"edge_ab": False, "edge_kinds": ["relates_to"]},
        {"common_neighbors": -1},
        {"common_neighbors": 2.0},  # float where int is required
        {"common_neighbors": True},  # bool is not an int here
        {"neighbor_jaccard": 1.5},
        {"neighbor_jaccard": "0.5"},
        {"cos_to_common_max": 1.0001},
        {"has_graph_evidence": True},  # on an evidence-empty payload
        {"neighbor_jaccard": None},  # required, not optional
    ],
)
def test_reader_corrupt_rows_refuse(tmp_path: Path, mutation: dict[str, object]) -> None:
    row = dict(EMPTY_ROW, pair_id="DV-N-002", **mutation)
    path = _write_sidecar(tmp_path, [row])
    with pytest.raises(GraphSidecarError):
        GraphSidecar(path)


def test_reader_missing_required_key(tmp_path: Path) -> None:
    row = {key: value for key, value in EMPTY_ROW.items() if key != "neighbor_jaccard"}
    with pytest.raises(GraphSidecarError, match="neighbor_jaccard"):
        GraphSidecar(_write_sidecar(tmp_path, [row]))


def test_reader_duplicate_pair_id(tmp_path: Path) -> None:
    row = dict(EMPTY_ROW, pair_id="DV-P-001")
    twin = dict(EVIDENT_ROW, pair_id="DV-P-001")
    with pytest.raises(GraphSidecarError, match="duplicate pair_id"):
        GraphSidecar(_write_sidecar(tmp_path, [row, twin]))


def test_reader_bad_json_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(GraphSidecarError, match="bad JSON"):
        GraphSidecar(path)


def test_reader_row_without_pair_id(tmp_path: Path) -> None:
    row = {key: value for key, value in EMPTY_ROW.items() if key != "pair_id"}
    with pytest.raises(GraphSidecarError, match="pair_id"):
        GraphSidecar(_write_sidecar(tmp_path, [row]))


# ── gated attach: values, composition order, refusals ────────────────────────


def test_attach_values_exact() -> None:
    evidence = GraphEvidence.from_row(EVIDENT_ROW)
    vector = attach_graph_features(_core_vector(), evidence)
    assert vector.names == FEATURE_NAMES + GRAPH_FEATURES
    values = dict(zip(vector.names, vector.values))
    assert values["edge_ab"] == 1.0
    assert values["edge_kind_supersedes"] == 0.0
    assert values["common_neighbors"] == 3.0
    assert values["neighbor_jaccard"] == 0.25
    assert values["cos_to_common_max"] == 0.9


def test_attach_zero_evidence_maps_absent_cosine_to_zero() -> None:
    vector = attach_graph_features(_core_vector(), GraphEvidence.zero())
    assert vector.names == FEATURE_NAMES + GRAPH_FEATURES
    assert vector.values[-5:] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_attach_composes_after_field_cosines() -> None:
    with_fields = attach_field_cosines(
        _core_vector(), cos_title=0.1, cos_body=0.2, cos_tags=0.3
    )
    vector = attach_graph_features(
        with_fields,
        GraphEvidence(
            edge_ab=True, edge_kind_supersedes=True, common_neighbors=1,
            neighbor_jaccard=1.0, cos_to_common_max=0.5,
        ),
    )
    assert vector.names == FEATURE_NAMES + FIELD_COSINE_FEATURES + GRAPH_FEATURES
    assert dict(zip(vector.names, vector.values))["edge_kind_supersedes"] == 1.0


def test_attach_twice_refuses() -> None:
    once = attach_graph_features(_core_vector(), GraphEvidence.zero())
    with pytest.raises(ValueError, match="already attached"):
        attach_graph_features(once, GraphEvidence.zero())


def test_attach_inconsistent_supersedes_refuses() -> None:
    bad = GraphEvidence(edge_ab=False, edge_kind_supersedes=True)
    with pytest.raises(ValueError, match="inconsistent"):
        attach_graph_features(_core_vector(), bad)


def test_attach_out_of_range_refuses() -> None:
    with pytest.raises(ValueError, match="neighbor_jaccard"):
        attach_graph_features(_core_vector(), GraphEvidence(neighbor_jaccard=1.5))
    with pytest.raises(ValueError, match="cos_to_common_max"):
        attach_graph_features(_core_vector(), GraphEvidence(cos_to_common_max=2.0))
    with pytest.raises(ValueError, match="common_neighbors"):
        attach_graph_features(_core_vector(), GraphEvidence(common_neighbors=-1))


def test_attach_to_wrong_base_refuses() -> None:
    bogus = FeatureVector(("x", "y"), (0.0, 0.0))
    with pytest.raises(ValueError, match="CORE or core\\+field-cosines"):
        attach_graph_features(bogus, GraphEvidence.zero())


# ── train-manifest compatibility ─────────────────────────────────────────────


def test_attach_to_features_missing_pair_modes(tmp_path: Path) -> None:
    sidecar = GraphSidecar(_write_sidecar(tmp_path, [EVIDENT_ROW]))
    covered = attach_to_features(_core_vector(), sidecar, "DV-N-001")
    assert covered.names == FEATURE_NAMES + GRAPH_FEATURES

    with pytest.raises(GraphSidecarError, match="not covered"):
        attach_to_features(_core_vector(), sidecar, "DV-N-404")

    graceful = attach_to_features(
        _core_vector(), sidecar, "DV-N-404", on_missing="zero"
    )
    assert graceful.values[-5:] == (0.0, 0.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="on_missing"):
        attach_to_features(_core_vector(), sidecar, "DV-N-404", on_missing="guess")  # type: ignore[arg-type]


def test_manifest_rows_join_by_pair_id(tmp_path: Path) -> None:
    """The assembled train manifest joins the sidecar by pair_id only."""
    sidecar = GraphSidecar(
        _write_sidecar(tmp_path, [EVIDENT_ROW, EMPTY_ROW])
    )
    manifest_pair_ids = ["DV-N-001", "DV-P-001", "DV-N-002"]
    vectors = []
    for pair_id in manifest_pair_ids:
        base = features(R_A, R_A, 0.5)
        if pair_id in sidecar:
            vectors.append(attach_to_features(base, sidecar, pair_id))
        else:
            vectors.append(base)
    assert [len(v.names) for v in vectors] == [18, 18, 13]
