"""Adversarial sanity suite tests (#480 gate).

Proves three things end-to-end against real exported ONNX graphs:

1. a healthy artifact — trained WITH self-pairs and the monotone cosine
   ladder in its training surface — passes every check;
2. a column-desynced artifact (trained on permuted feature columns while
   the metadata still claims the canonical order — the #480 defect
   signature) is CAUGHT, failing the self-pair check;
3. the bundle-contract preconditions refuse a lying manifest (feature
   order, n-head candidate) and unloadable bundles fail loud.

Training corpora here are programmatic (no store, no network), following
the tests/synth.py conventions; models are tiny (GRID_D point 1) so the
file stays fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cortex.artifacts import (
    CANDIDATE_D,
    build_metadata_props,
    features_digest,
    sha256_file,
)
from cortex.candidates.d_boost import GRID_D, DBoostModel
from cortex.eval.sanity import (
    COS_LADDER,
    PROBE_RECORD,
    UNRELATED_RECORD,
    SanityLoadError,
    _perturbed,
    run_sanity_suite,
)
from cortex.features.pair import FEATURE_NAMES, FeatureVector, features

from synth import make_pair_rows, make_records, rows_labels, rows_to_vectors


# ── probe-vector invariants (no model needed) ────────────────────────────────


def test_self_pair_vector_is_constant_all_ones_and_matches() -> None:
    """Record vs itself pins every feature: n-grams identical, deltas 0,
    type/lang match, cos 1.0 — the number is content-independent, which is
    what makes the #480 reproduction (0.0108) cross-comparable."""
    vector = features(PROBE_RECORD, PROBE_RECORD, 1.0)
    assert vector.names == FEATURE_NAMES
    assert vector.values == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0)


def test_probe_records_are_topically_disjoint() -> None:
    """The unrelated side shares no tag and no topical anchor with the probe."""
    probe_tags = set(PROBE_RECORD.tags)
    assert probe_tags.isdisjoint(UNRELATED_RECORD.tags)
    assert "кэш" not in UNRELATED_RECORD.body.lower()


# ── training corpora (programmatic, #480-shaped) ─────────────────────────────


def _side(record) -> dict:
    return {
        "title": record.title,
        "body": record.body,
        "tags": list(record.tags),
        "language": record.language,
        "record_type": record.record_type,
    }


def _row(pid: str, a, b, similarity: float, label: str, stratum: str) -> dict:
    return {
        "pair_id": pid,
        "record": _side(a),
        "candidate": _side(b),
        "similarity": similarity,
        "label": label,
        "stratum": stratum,
    }


def _ladder_rows(base_records, prefix: str, start: int = 0) -> list[dict]:
    """The monotone cosine ladder: a fixed pair re-scored at COS_LADDER —
    exactly the surface B2's eval lacked (no exact self point)."""
    rows: list[dict] = []
    for offset, record in enumerate(base_records):
        twin = _perturbed(record)
        for i, cos in enumerate(COS_LADDER):
            label = "duplicate" if cos >= 0.95 else "not-duplicate"
            rows.append(
                _row(
                    f"{prefix}-{start + offset:02d}-ladder{i}",
                    record,
                    twin,
                    float(cos),
                    label,
                    "SANITY-LADDER",
                )
            )
    return rows


def _self_rows(base_records, prefix: str) -> list[dict]:
    return [
        _row(f"{prefix}-self-{i:02d}", record, record, 1.0, "duplicate", "SANITY-SELF")
        for i, record in enumerate(base_records)
    ]


def _unrelated_rows(base_records, prefix: str) -> list[dict]:
    """Different-topic pairs at low cosine (label not-duplicate by
    construction): base_records[i] vs base_records[i + 3] — the synthetic
    topics cycle with period 6, so i and i+3 never share an anchor."""
    cosines = (0.578, 0.60, 0.55)
    rows: list[dict] = []
    for i in range(min(3 * len(cosines), len(base_records) - 3)):
        rows.append(
            _row(
                f"{prefix}-unrel-{i:02d}",
                base_records[i],
                base_records[i + 3],
                float(cosines[i % len(cosines)]),
                "not-duplicate",
                "SANITY-UNRELATED",
            )
        )
    return rows


def _healthy_training_rows() -> list[dict]:
    rows = make_pair_rows(120)  # the W4c hard-zone filler (0.96/0.93)
    extra_records = list(make_records(12, seed=3)) + [PROBE_RECORD]
    rows += _self_rows(extra_records, "sanity")
    rows += _ladder_rows(list(make_records(6, seed=5)) + [PROBE_RECORD], "sanity")
    rows += _unrelated_rows(list(make_records(12, seed=7)) + [UNRELATED_RECORD], "sanity")
    return rows


def _trained_model(vectors, rows) -> DBoostModel:
    model = DBoostModel()
    model.train(vectors, rows_labels(rows), GRID_D[1], calibrate=True)
    return model


def _write_bundle(tmp_path: Path, model: DBoostModel, name: str = "model") -> Path:
    """Export a bundle exactly the way `cortex export-artifact` does
    (metadata + sibling <stem>.manifest.json)."""
    out = tmp_path / f"{name}.onnx"
    props = build_metadata_props(
        embedder_pin="nano:sha256:" + "ab" * 32,
        corpus_fingerprint="cd" * 32,
        trained_at="2026-10-03T00:00:00+00:00",
        candidate=CANDIDATE_D,
        feature_names=FEATURE_NAMES,
    )
    model.export_onnx(out, metadata_props=props)
    manifest = {
        "name": "vesma-cortex-v1",
        "version": "1",
        "sha256": sha256_file(out),
        "embedder_pin": props["embedder_pin"],
        "corpus_fingerprint": props["corpus_fingerprint"],
        "trained_at": props["trained_at"],
        "candidate": CANDIDATE_D,
        "features": list(FEATURE_NAMES),
        "feature_set_sha256": props["feature_set_sha256"],
        "size_bytes": out.stat().st_size,
        "weights_path": str(out),
    }
    manifest_path = tmp_path / f"{name}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return tmp_path


def _permute_columns(vectors: list[FeatureVector], swap: tuple[int, int] = (0, 9)):
    """Feature-column desync simulator (#480): the graph is TRAINED with
    columns permuted (cos_target ↔ body_len_delta) while the exported
    metadata keeps claiming the canonical order."""
    i, j = swap
    out = []
    for vector in vectors:
        values = list(vector.values)
        values[i], values[j] = values[j], values[i]
        out.append(FeatureVector(vector.names, tuple(values)))
    return out


@pytest.fixture(scope="module")
def healthy_bundle(tmp_path_factory) -> Path:
    rows = _healthy_training_rows()
    model = _trained_model(rows_to_vectors(rows), rows)
    return _write_bundle(tmp_path_factory.mktemp("healthy"), model)


@pytest.fixture(scope="module")
def desynced_bundle(tmp_path_factory) -> Path:
    rows = _healthy_training_rows()
    vectors = _permute_columns(rows_to_vectors(rows))
    model = _trained_model(vectors, rows)
    return _write_bundle(tmp_path_factory.mktemp("desynced"), model)


# ── the gate behaviour ───────────────────────────────────────────────────────


def test_healthy_bundle_passes_every_check(healthy_bundle: Path) -> None:
    report = run_sanity_suite(healthy_bundle)
    assert report.passed, "\n".join(
        f"{check.name}: {check.detail}" for check in report.failed
    )
    assert [check.name for check in report.checks] == [
        "bundle_integrity",
        "feature_contract",
        "candidate_supported",
        "self_pair",
        "near_boundary",
        "unrelated",
        "monotonicity",
    ]


def test_column_desync_is_caught_by_self_pair(desynced_bundle: Path) -> None:
    """The #480 regression pin: a graph whose column order diverges from
    the metadata-claimed contract must FAIL the suite — at minimum on the
    self-pair check (inversion), like the shipped B2 bundle did."""
    report = run_sanity_suite(desynced_bundle)
    assert not report.passed
    failed_names = {check.name for check in report.failed}
    assert "self_pair" in failed_names, "\n".join(
        f"{check.name}: {check.detail}" for check in report.checks
    )


def test_monotonicity_reports_the_full_ladder(healthy_bundle: Path) -> None:
    report = run_sanity_suite(healthy_bundle)
    monotonicity = next(check for check in report.checks if check.name == "monotonicity")
    for cos in COS_LADDER:
        assert f"cos={cos:.2f}" in monotonicity.detail


# ── bundle-contract preconditions ────────────────────────────────────────────


def test_manifest_feature_order_mismatch_is_caught(healthy_bundle: Path) -> None:
    manifest_path = healthy_bundle / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rotated = manifest["features"][1:] + manifest["features"][:1]
    manifest["features"] = rotated
    manifest["feature_set_sha256"] = features_digest(tuple(rotated))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = run_sanity_suite(healthy_bundle)
    feature_contract = next(check for check in report.checks if check.name == "feature_contract")
    assert not feature_contract.passed
    assert "#480 desync signature" in feature_contract.detail


def test_n_head_candidate_is_refused(healthy_bundle: Path) -> None:
    manifest_path = healthy_bundle / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate"] = "n-head"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = run_sanity_suite(healthy_bundle)
    candidate_check = next(check for check in report.checks if check.name == "candidate_supported")
    assert not candidate_check.passed
    assert "vector sidecars" in candidate_check.detail


def test_missing_manifest_fails_loud(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"not a real onnx")
    with pytest.raises(SanityLoadError, match="manifest not found"):
        run_sanity_suite(tmp_path)


def test_missing_bundle_path_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(SanityLoadError, match="does not exist"):
        run_sanity_suite(tmp_path / "absent")
