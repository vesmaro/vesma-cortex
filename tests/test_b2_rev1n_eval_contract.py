"""B2 rev 1.n gated-contract pins (prereg v3, slice-3).

The gated graph block is ADDITIVE (ADR 0002): the legacy 13-feature
behavior must stay byte-identical, and the composite
``FEATURE_NAMES + GRAPH_FEATURES`` must train/predict/grade cleanly —
including the sanity suite's feature-contract check admitting exactly
TWO orders (core, gated composite) and nothing else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cortex.artifacts import features_digest  # noqa: E402
from cortex.candidates.d_boost import DGridConfig, DBoostModel  # noqa: E402
from cortex.eval.sanity import (  # noqa: E402
    COSMETIC_TWIN_MIN,
    FACT_EDIT_TWIN_MAX,
    SELF_PAIR_MIN,
    UNRELATED_MAX,
    _adversarial_checks,
    _contract_checks,
)
from cortex.features.graph import (  # noqa: E402
    GRAPH_FEATURES,
    GraphEvidence,
    attach_graph_features,
)
from cortex.features.pair import FEATURE_NAMES, FeatureVector, features  # noqa: E402

GATED_NAMES = FEATURE_NAMES + GRAPH_FEATURES


def _synthetic_vectors(names: tuple[str, ...], n: int = 40, seed: int = 7):
    """Labeled toy vectors: a linear rule over the first two columns."""
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(n, len(names)))
    labels = (xs[:, 0] + 0.5 * xs[:, 1] > 0).astype(np.int64)
    return [
        FeatureVector(names=names, values=tuple(map(float, row))) for row in xs
    ], labels.tolist()


def _grid(names: tuple[str, ...]) -> DGridConfig:
    return DGridConfig(
        "rev1n-test",
        num_leaves=7,
        learning_rate=0.05,
        min_data_in_leaf=5,
        field_cosines=False,
        seed=1,
        feature_contract=names,
    )


def test_d_boost_legacy_contract_unchanged() -> None:
    vectors, labels = _synthetic_vectors(FEATURE_NAMES)
    model = DBoostModel()
    model.train(vectors, labels, DGridConfig("legacy", 7, 0.05, 5, False, 1))
    assert model.feature_names == FEATURE_NAMES
    probs = model.predict_proba(vectors[:5])
    assert probs.shape == (5,) and bool(((probs >= 0) & (probs <= 1)).all())


def test_d_boost_rejects_mismatched_contract() -> None:
    vectors, labels = _synthetic_vectors(FEATURE_NAMES)
    model = DBoostModel()
    with pytest.raises(ValueError, match="gated contract"):
        model.train(vectors, labels, _grid(GATED_NAMES))


def test_d_boost_trains_the_gated_composite() -> None:
    vectors, labels = _synthetic_vectors(GATED_NAMES)
    model = DBoostModel()
    model.train(vectors, labels, _grid(GATED_NAMES))
    assert model.feature_names == GATED_NAMES
    probs = model.predict_proba(vectors)
    assert probs.shape == (len(vectors),)
    assert bool(((probs >= 0) & (probs <= 1)).all())
    # Round-trip persistence keeps the gated contract.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = model.save(Path(tmp))
        loaded = DBoostModel.load(out)
    assert loaded.feature_names == GATED_NAMES
    assert np.allclose(
        loaded.predict_proba(vectors[:3]), model.predict_proba(vectors[:3])
    )


def test_attach_graph_features_zero_evidence_is_pure_append() -> None:
    base = features(
        __import__("cortex.eval.sanity", fromlist=["PROBE_RECORD"]).PROBE_RECORD,
        __import__("cortex.eval.sanity", fromlist=["PROBE_RECORD"]).PROBE_RECORD,
        1.0,
    )
    gated = attach_graph_features(base, GraphEvidence.zero())
    assert gated.names == GATED_NAMES
    assert list(gated.values[: len(FEATURE_NAMES)]) == list(base.values)
    assert list(gated.values[len(FEATURE_NAMES) :]) == [0.0] * len(GRAPH_FEATURES)


def _meta_of(names: tuple[str, ...]) -> dict[str, str]:
    return {"features": "\n".join(names)}


def _manifest_of(names: tuple[str, ...]) -> dict:
    return {
        "sha256": "0" * 64,
        "features": list(names),
        "candidate": "d-boost",
        "feature_set_sha256": features_digest(names),
    }


def test_feature_contract_admits_core_and_gated_orders_only() -> None:
    for names in (FEATURE_NAMES, GATED_NAMES):
        checks = {
            c.name: c
            for c in _contract_checks(_manifest_of(names), _meta_of(names), "0" * 64)
        }
        assert checks["feature_contract"].passed, names
    # The sha256 mismatch (weights != manifest) is a SEPARATE check —
    # feature_contract must still pass so the two defects stay diagnosable.
    # (bundle_integrity is asserted red here to keep the isolation honest.)
    checks = {
        c.name: c
        for c in _contract_checks(
            _manifest_of(GATED_NAMES), _meta_of(GATED_NAMES), "1" * 64
        )
    }
    assert checks["feature_contract"].passed
    assert not checks["bundle_integrity"].passed


def test_feature_contract_refuses_permutations_and_truncations() -> None:
    bad_orders = [
        GATED_NAMES[-5:] + GATED_NAMES[:-5],  # graph block first
        FEATURE_NAMES[1:] + FEATURE_NAMES[:1],  # core permutation
        GATED_NAMES[:-1],  # truncated composite
        FEATURE_NAMES + (" rogue",),  # unknown tail
    ]
    for names in bad_orders:
        checks = {
            c.name: c
            for c in _contract_checks(_manifest_of(names), _meta_of(names), "0" * 64)
        }
        assert not checks["feature_contract"].passed, names
        assert (
            "#480" in checks["feature_contract"].detail
            or "frozen" in checks["feature_contract"].detail
        )


def test_adversarial_checks_grade_the_gated_graph_on_probes() -> None:
    """The gated session path: probes must carry the zero graph block —
    exercised through a tiny stub session recording the input width."""

    class _StubSession:
        @staticmethod
        def get_inputs():
            class _In:
                name = "features"
                shape = [None, len(GATED_NAMES)]

            return [_In()]

        @staticmethod
        def run(_, feed):
            row = feed["features"]
            # THE pin: every probe is graded 18-wide, graph block == zeros.
            # (_score_pair reshapes to (1, K) for the 2-D input contract.)
            assert row.shape == (1, len(GATED_NAMES)), row.shape
            assert np.allclose(row[0, len(FEATURE_NAMES) :], 0.0), row
            return [np.asarray([0.5], dtype=np.float32)]

    checks = _adversarial_checks(_StubSession(), graph_gated=True)
    names = [c.name for c in checks]
    assert names == [
        "self_pair",
        "cosmetic_twin",
        "fact_edit_twin",
        "envelope_variant",
        "unrelated",
        "monotonicity",
    ]


def test_sanity_thresholds_unchanged() -> None:
    """Prereg v3 does not touch the sanity floors — pinned against drift."""
    assert SELF_PAIR_MIN == 0.9
    assert COSMETIC_TWIN_MIN == 0.5
    assert FACT_EDIT_TWIN_MAX == 0.5
    assert UNRELATED_MAX == 0.5


def test_grid_config_json_round_trip_keeps_gated_contract() -> None:
    config = _grid(GATED_NAMES)
    payload = json.loads(json.dumps(config.__dict__))
    restored = DGridConfig(**payload)
    # json yields a list — the model code normalizes via _contract_of().
    assert tuple(restored.feature_contract or ()) == GATED_NAMES
    from cortex.candidates.d_boost import _contract_of  # noqa: SLF001 — pin

    assert _contract_of(restored) == GATED_NAMES
