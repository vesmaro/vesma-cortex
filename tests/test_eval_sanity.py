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
from dataclasses import replace
from pathlib import Path

import pytest
from synth import make_pair_rows, make_records, rows_labels, rows_to_vectors

from cortex.artifacts import (
    CANDIDATE_D,
    build_metadata_props,
    features_digest,
    sha256_file,
)
from cortex.candidates.d_boost import GRID_D, DBoostModel
from cortex.eval.sanity import (
    COS_LADDER,
    COSMETIC_TWIN_MIN,
    ENVELOPE_DATES,
    ENVELOPE_VARIANT_COS,
    ENVELOPE_VARIANT_MIN,
    FACT_EDIT_ANCHOR,
    FACT_EDIT_COS,
    FACT_EDIT_REPLACEMENT,
    FACT_EDIT_TWIN_MAX,
    MONOTONICITY_TOLERANCE,
    PROBE_RECORD,
    RAZOR_ZONE_COS_LOW,
    UNRELATED_RECORD,
    SanityLoadError,
    _envelope_variant,
    _fact_edited,
    _perturbed,
    run_sanity_suite,
)
from cortex.features.pair import FEATURE_NAMES, FeatureVector, PairRecord, features

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── probe-vector invariants (no model needed) ────────────────────────────────


def test_self_pair_vector_is_constant_all_ones_and_matches() -> None:
    """Record vs itself pins every feature: n-grams identical, deltas 0,
    type/lang match, cos 1.0 — the number is content-independent, which is
    what makes the #480 reproduction (0.0108) cross-comparable."""
    vector = features(PROBE_RECORD, PROBE_RECORD, 1.0)
    assert vector.names == FEATURE_NAMES
    assert vector.values == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    )


def test_probe_records_are_topically_disjoint() -> None:
    """The unrelated side shares no tag and no topical anchor with the probe."""
    probe_tags = set(PROBE_RECORD.tags)
    assert probe_tags.isdisjoint(UNRELATED_RECORD.tags)
    assert "кэш" not in UNRELATED_RECORD.body.lower()


# ── sanity v2 probe mechanics (no model needed; eval-methodology §10) ────────


def test_fact_edit_twin_changes_exactly_one_key_fact_token() -> None:
    """The razor-side twin: ONE key-fact token in the BODY (the cache-key
    name — policy v1.1 §8.2 'names' class = NOT dup), same length by design
    (razor-zone shaped: zero len-delta, minimal char signature)."""
    fact_twin = _fact_edited(PROBE_RECORD)
    assert fact_twin.title == PROBE_RECORD.title
    assert fact_twin.tags == PROBE_RECORD.tags
    assert fact_twin.body == PROBE_RECORD.body.replace(
        FACT_EDIT_ANCHOR, FACT_EDIT_REPLACEMENT
    )
    assert len(fact_twin.body) == len(PROBE_RECORD.body)
    vector = features(PROBE_RECORD, fact_twin, FACT_EDIT_COS)
    values = dict(zip(vector.names, vector.values))
    assert values["title_len_delta"] == 0.0
    assert values["body_len_delta"] == 0.0
    assert values["char5_jaccard"] < 1.0
    assert values["cos_target"] == FACT_EDIT_COS


def test_envelope_variant_is_feature_degenerate_by_contract() -> None:
    """The whitelist-side twin rides OUTSIDE the 13-feature surface
    (RecordLike carries no timestamp — OQ-2), so its vector equals the
    self-pair vector BY CONTRACT. The documented envelope delta is real at
    the data level (ENVELOPE_DATES differ) but invisible to the features:
    if the feature surface ever grows envelope fields, this pin fails and
    forces the probe to grow the real delta (see sanity._envelope_variant)."""
    assert ENVELOPE_DATES[0] != ENVELOPE_DATES[1]
    env_vector = features(
        _envelope_variant(PROBE_RECORD), PROBE_RECORD, ENVELOPE_VARIANT_COS
    )
    self_vector = features(PROBE_RECORD, PROBE_RECORD, ENVELOPE_VARIANT_COS)
    assert env_vector.values == self_vector.values


def test_sanity_v2_constants_match_the_gate_contract() -> None:
    """The coherence pin (mirror of the corner_qa one): the committed
    sanity_v2 section of gate_contract.json and the frozen v2 constants
    must be EQUAL — an exam-threshold change that touches only one side
    fails here (eval-methodology §6)."""
    contract = json.loads(
        (REPO_ROOT / "gate_contract.json").read_text(encoding="utf-8")
    )
    section = contract["sanity_v2"]
    assert COSMETIC_TWIN_MIN == section["cosmetic_twin_min"]
    assert FACT_EDIT_TWIN_MAX == section["fact_edit_twin_max"]
    assert ENVELOPE_VARIANT_MIN == section["envelope_variant_min"]
    mono = section["monotonicity_v2"]
    assert list(COS_LADDER) == mono["ladder_cos"]
    assert MONOTONICITY_TOLERANCE == mono["tolerance"]
    # the operational zoned rule: razor zone is the OPEN interval
    # (0.95; 1.0) — steps starting above 0.95 are unchecked
    assert RAZOR_ZONE_COS_LOW == 0.95


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


def _razor_base_records(n: int = 6) -> list[PairRecord]:
    """Long-body records for the razor band (the corpus T3 geometry: one
    token of a ~160-char body edited — a minimal char signature)."""
    return [
        PairRecord(
            title=f"рабочая заметка {i}",
            body=(
                f"Рабочая заметка номер {i}: фиксировали состояние проекта, "
                "обсудили сроки следующего этапа, распределили задачи между "
                "участниками и договорились синхронизироваться после релиза "
                "в течение недели."
            ),
            tags=("work", "note"),
            language="ru",
            record_type="note",
        )
        for i in range(n)
    ]


def _generic_fact_edit(record: PairRecord) -> PairRecord:
    """One key-fact token (the record's number) changed in the body — the
    corpus T3 'число' class, the generic analogue of sanity._fact_edited."""
    digits = "0123456789"
    idx = max(record.body.rfind(d) for d in digits)
    assert idx >= 0, "generic fact edit needs a digit in the body"
    swapped = digits[(digits.index(record.body[idx]) + 3) % 10]
    return replace(record, body=record.body[:idx] + swapped + record.body[idx + 1 :])


def _razor_rows(base_records, prefix: str, sides: str = "both") -> list[dict]:
    """The razor band at cos 0.99 (policy v1.1 two-valuedness): cosmetic
    twins = dup (title punctuation), fact-edit twins = NOT dup.
    ``sides`` selects the taught side — ``fact`` builds the negative-dominant
    surface (the B2-prime shape) and ``cosmetic`` the positive-only one for
    the failing fixtures."""
    rows: list[dict] = []
    for i, record in enumerate(base_records):
        if sides in ("both", "cosmetic"):
            rows.append(
                _row(
                    f"{prefix}-razor-cosm-{i:02d}",
                    record,
                    _perturbed(record),
                    0.99,
                    "duplicate",
                    "SANITY-RAZOR",
                )
            )
        if sides in ("both", "fact"):
            rows.append(
                _row(
                    f"{prefix}-razor-fact-{i:02d}",
                    record,
                    _generic_fact_edit(record),
                    FACT_EDIT_COS,
                    "not-duplicate",
                    "SANITY-RAZOR",
                )
            )
    return rows


def _probe_razor_fact_row() -> dict:
    """The probe's own fact-edit negative (the exact anchor the exam probes)."""
    return _row(
        "sanity-razor-fact-probe",
        PROBE_RECORD,
        _fact_edited(PROBE_RECORD),
        FACT_EDIT_COS,
        "not-duplicate",
        "SANITY-RAZOR",
    )


def _healthy_training_rows() -> list[dict]:
    rows = make_pair_rows(120)  # the W4c hard-zone filler (0.96/0.93)
    extra_records = list(make_records(12, seed=3)) + [PROBE_RECORD]
    rows += _self_rows(extra_records, "sanity")
    rows += _ladder_rows(list(make_records(6, seed=5)) + [PROBE_RECORD], "sanity")
    # policy v1.1: BOTH sides of the razor band, incl. the probe's own twin
    rows += _razor_rows(_razor_base_records(), "sanity")
    rows += [_probe_razor_fact_row()]
    rows += _unrelated_rows(
        list(make_records(12, seed=7)) + [UNRELATED_RECORD], "sanity"
    )
    return rows


def _trained_model(vectors, rows) -> DBoostModel:
    model = DBoostModel()
    model.train(vectors, rows_labels(rows), GRID_D[1], calibrate=True)
    return model


def _write_bundle(
    tmp_path: Path,
    model: DBoostModel,
    name: str = "model",
    sanity_exam: str | None = "2",
) -> Path:
    """Export a bundle exactly the way `cortex export-artifact` does
    (metadata + sibling <stem>.manifest.json). The manifest stamp defaults
    to the v2 cohort — `export-artifact` writes `sanity_exam: "2"` since
    the policy-aware exam; pass None for the unstamped (v1-history)
    convention the registry B1 artifact carries."""
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
    if sanity_exam is not None:
        manifest["sanity_exam"] = sanity_exam
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
def v1_bundle(tmp_path_factory) -> Path:
    """A v1-exam bundle: trained on the W4c hard zone with NO razor-band
    two-sidedness (the pre-policy corpus shape — e.g. the registry B1
    weights); the historical exam is the one it was certified under."""
    rows = make_pair_rows(120)
    extra_records = list(make_records(12, seed=3)) + [PROBE_RECORD]
    rows += _self_rows(extra_records, "sanity")
    rows += _ladder_rows(list(make_records(6, seed=5)) + [PROBE_RECORD], "sanity")
    rows += _unrelated_rows(
        list(make_records(12, seed=7)) + [UNRELATED_RECORD], "sanity"
    )
    model = _trained_model(rows_to_vectors(rows), rows)
    return _write_bundle(tmp_path_factory.mktemp("v1-exam"), model, sanity_exam=None)


@pytest.fixture(scope="module")
def desynced_bundle(tmp_path_factory) -> Path:
    rows = _healthy_training_rows()
    vectors = _permute_columns(rows_to_vectors(rows))
    model = _trained_model(vectors, rows)
    return _write_bundle(tmp_path_factory.mktemp("desynced"), model)


@pytest.fixture(scope="module")
def razor_dominant_bundle(tmp_path_factory) -> Path:
    """The B2-prime shape (the wave's root cause, pinned): the razor band
    taught NEGATIVE-dominated — fact edits only, the only positives being
    text-IDENTICAL self-pairs (char5 = 1.0, the v4.0 T0/T1 geometry) — so
    «slightly different text at cos 0.99 = not dup» becomes the zone's
    law and the punctuation twin (char5 < 1) grades below the cut, exactly
    like B2-prime's near_boundary 0.0261."""
    rows = _self_rows([PROBE_RECORD], "sanity")
    rows += _razor_rows(_razor_base_records(8), "sanity", sides="fact")
    rows += [_probe_razor_fact_row()]
    rows += _unrelated_rows(list(make_records(12, seed=7)), "sanity")
    model = _trained_model(rows_to_vectors(rows), rows)
    return _write_bundle(tmp_path_factory.mktemp("razor-dominant"), model)


@pytest.fixture(scope="module")
def cosmetic_dominant_bundle(tmp_path_factory) -> Path:
    """The inverse dominance: the razor band taught POSITIVE-only —
    cosmetic twins at 0.99, zero fact-edit negatives — the model answers
    «dup» to anything close and fails the fact-edit twin."""
    rows = make_pair_rows(120)
    rows += _self_rows([PROBE_RECORD], "sanity")
    rows += _ladder_rows(list(make_records(6, seed=5)) + [PROBE_RECORD], "sanity")
    rows += _razor_rows(_razor_base_records(8), "sanity", sides="cosmetic")
    rows += _unrelated_rows(list(make_records(12, seed=7)), "sanity")
    model = _trained_model(rows_to_vectors(rows), rows)
    return _write_bundle(tmp_path_factory.mktemp("cosmetic-dominant"), model)


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
        "cosmetic_twin",
        "fact_edit_twin",
        "envelope_variant",
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


def test_negative_dominant_razor_band_fails_cosmetic_twin(
    razor_dominant_bundle: Path,
) -> None:
    """The B2-prime regression pin (the wave D4-3 root cause): a corpus
    that teaches the razor band negative-dominated must FAIL the v2 exam —
    the cosmetic twin grades below the cut exactly like B2-prime's
    near_boundary 0.0261."""
    report = run_sanity_suite(razor_dominant_bundle)
    assert not report.passed
    assert "cosmetic_twin" in {c.name for c in report.failed}, "\n".join(
        f"{check.name}: {check.detail}" for check in report.checks
    )


def test_positive_only_razor_band_fails_fact_edit_twin(
    cosmetic_dominant_bundle: Path,
) -> None:
    """The inverse dominance: a razor band taught positive-only answers
    «dup» to a fact edit — the anti-dominance probe catches it (the
    two-sided exam of eval-methodology §10.1)."""
    report = run_sanity_suite(cosmetic_dominant_bundle)
    assert not report.passed
    assert "fact_edit_twin" in {c.name for c in report.failed}, "\n".join(
        f"{check.name}: {check.detail}" for check in report.checks
    )


def test_monotonicity_reports_the_full_ladder(healthy_bundle: Path) -> None:
    report = run_sanity_suite(healthy_bundle)
    monotonicity = next(
        check for check in report.checks if check.name == "monotonicity"
    )
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
    feature_contract = next(
        check for check in report.checks if check.name == "feature_contract"
    )
    assert not feature_contract.passed
    assert "#480 desync signature" in feature_contract.detail


def test_n_head_candidate_is_refused(healthy_bundle: Path) -> None:
    manifest_path = healthy_bundle / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate"] = "n-head"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = run_sanity_suite(healthy_bundle)
    candidate_check = next(
        check for check in report.checks if check.name == "candidate_supported"
    )
    assert not candidate_check.passed
    assert "vector sidecars" in candidate_check.detail


def test_missing_manifest_fails_loud(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"not a real onnx")
    with pytest.raises(SanityLoadError, match="manifest not found"):
        run_sanity_suite(tmp_path)


def test_missing_bundle_path_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(SanityLoadError, match="does not exist"):
        run_sanity_suite(tmp_path / "absent")


# ── exam cohort (manifest stamp; eval-methodology §10 change-control) ────────


def test_v1_stamp_runs_the_historical_exam(v1_bundle: Path) -> None:
    """An ABSENT stamp = v1 cohort: the historical one-sided probes run
    (near_boundary positive side, full-ladder monotonicity) — the registry
    B1 weights stay certified under the exam their ADOPT was graded by."""
    report = run_sanity_suite(v1_bundle)
    assert report.exam_version == "1"
    names = [check.name for check in report.checks]
    assert "near_boundary" in names and "cosmetic_twin" not in names


def test_v2_stamp_runs_the_policy_aware_exam(healthy_bundle: Path) -> None:
    """A v2 stamp = the two-sided exam: both razor sides probed — the
    export-artifact default for all new artifacts (wave D4-3)."""
    manifest_path = healthy_bundle / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sanity_exam"] = "2"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = run_sanity_suite(healthy_bundle)
    assert report.exam_version == "2"
    names = [check.name for check in report.checks]
    assert "cosmetic_twin" in names and "fact_edit_twin" in names
    assert "near_boundary" not in names


def test_v1_certified_bundle_keeps_passing(v1_bundle: Path) -> None:
    """The CI anachronism pinned: a v1-certified bundle (e.g. registry B1)
    is graded under the SAME exam its certification rests on, not
    retroactively under the ratified-but-newer v2 exam the bundle was
    never trained to satisfy."""
    report = run_sanity_suite(v1_bundle)
    assert report.passed, "\n".join(
        f"{check.name}: {check.detail}" for check in report.failed
    )


def test_unknown_stamp_fails_loud(healthy_bundle: Path) -> None:
    """A mistyped stamp must NOT silently degrade the exam to either
    generation (fail-loud boundary)."""
    manifest_path = healthy_bundle / "model.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sanity_exam"] = "2.1"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(SanityLoadError, match="unknown sanity_exam stamp"):
        run_sanity_suite(healthy_bundle)
