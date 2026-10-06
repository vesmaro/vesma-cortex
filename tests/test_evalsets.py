"""Layer A evalsets tests (wave LA-1).

Pins, in order of the wave invariants:

1. the §7 taxonomy: 8 canonical classes + LLM slots, coverage matrix;
2. probe-topic DISJOINTNESS from the train seed list (TOPICS) — a test,
   not luck;
3. generator BYTE-determinism (double run → identical JSONL + sha);
4. frozen-set loading: schema validation, fingerprint re-verification,
   tamper rejection (data-contract §5.6);
5. the COMMITTED datasets match their pinned meta/manifest bytes;
6. the LLM-slot mechanics: valid slots ride in, out-of-contract slots
   are refused;
7. the runner end-to-end on a FIXTURE bundle (never prod weights):
   healthy bundle passes every gate; the #480 column-desync bundle is
   caught.

The fixture model is trained ON the eval probes themselves (plus the W4c
filler) — this file verifies RUNNER PLUMBING, not model quality; the
prod-bundle numbers live in artifacts/manifests/layer-a-evalsets.md.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from synth import make_pair_rows, rows_labels, rows_to_vectors

from cortex.artifacts import (
    CANDIDATE_D,
    build_metadata_props,
    sha256_file,
)
from cortex.candidates.d_boost import GRID_D, DBoostModel
from cortex.eval.sanity import COS_LADDER, PROBE_RECORD, UNRELATED_RECORD
from cortex.evalsets import (
    ALL_PROBE_CLASSES,
    CANONICAL_PROBE_CLASSES,
    COVERAGE_MATRIX,
    DEFAULT_SEED,
    EvalProbe,
    EvalSetError,
    MERGE_V1,
    RELEASE_V1,
    build_eval_set,
    eval_set_jsonl,
    eval_set_manifest_bytes,
    eval_set_sha256,
    evaluate_gates,
    load_eval_set,
    load_llm_slots,
    probe_row,
    run_layer_a,
)
from cortex.evalsets.taxonomy import (
    GATE_CORRIDOR,
    GATE_INVARIANT,
    PROBE_CLASS_DEGENERATE,
    PROBE_CLASS_FAR_NEGATIVE,
    PROBE_CLASS_IDENTITY_SELF,
    PROBE_CLASS_LLM_NEAR_TOPIC,
    PROBE_CLASS_LLM_PARAPHRASE,
    PROBE_CLASS_MONOTONICITY,
    PROBE_CLASS_NEAR_IDENTITY,
    PROBE_CLASS_PAIR_SYMMETRY,
    PROBE_CLASS_TRANSLATION_TWINS,
    PROBE_CLASS_TYPE_LANG_MISMATCH,
)
from cortex.evalsets.topics import EVAL_TOPICS, anchor_keys, anchor_records
from cortex.features.pair import FEATURE_NAMES, features
from cortex.pretrain.corruption import record_key
from cortex.synth.generate import TOPICS

REPO_ROOT = Path(__file__).resolve().parent.parent
EVALSETS_DIR = REPO_ROOT / "datasets" / "evalsets"


# ── 1. the taxonomy ───────────────────────────────────────────────────────────


def test_coverage_matrix_pins_the_eight_canonical_classes() -> None:
    assert len(CANONICAL_PROBE_CLASSES) == 8
    assert [spec.name for spec in COVERAGE_MATRIX][:8] == list(CANONICAL_PROBE_CLASSES)
    # §7 tails closed/kept visible: symmetry + degenerate are procedural
    # now, translation twins carry the LA-2 LLM marker.
    by_name = {spec.name: spec for spec in COVERAGE_MATRIX}
    assert by_name[PROBE_CLASS_TRANSLATION_TWINS].source.startswith("llm-batch")
    assert by_name[PROBE_CLASS_PAIR_SYMMETRY].gate_kind == GATE_INVARIANT
    assert by_name[PROBE_CLASS_DEGENERATE].gate_kind == "breakdown"
    assert by_name[PROBE_CLASS_FAR_NEGATIVE].gate_kind == GATE_CORRIDOR
    # gate kinds: exactly two invariants + two corridors among canonical
    gated = [
        spec.gate_kind
        for spec in COVERAGE_MATRIX
        if spec.gate_kind in (GATE_INVARIANT, GATE_CORRIDOR)
    ]
    assert gated.count(GATE_INVARIANT) == 3  # self, symmetry, ladder
    assert gated.count(GATE_CORRIDOR) == 2  # near, far


# ── 2. topic disjointness (eval surface vs train surface) ─────────────────────


def test_eval_topics_disjoint_from_train_topics() -> None:
    """Eval probes must never train-topic themselves: keys, titles and
    content hashes of the eval anchors are disjoint from TOPICS."""
    train_keys = {topic.key for topic in TOPICS}
    train_titles = {topic.title for topic in TOPICS}
    train_record_keys = {record_key(topic.as_record()) for topic in TOPICS}

    eval_keys = {topic.key for topic in EVAL_TOPICS}
    eval_titles = {topic.title for topic in EVAL_TOPICS}
    # RU/EN halves SHARE a key by design (translation twins) — 16 topics,
    # 8 unique keys, every key twice
    assert len(eval_keys) * 2 == len(EVAL_TOPICS)
    assert eval_keys.isdisjoint(train_keys)
    assert eval_titles.isdisjoint(train_titles)
    assert train_record_keys.isdisjoint(
        {record_key(topic.as_record()) for topic in EVAL_TOPICS}
    )

    # the two sanity anchors ride along as probe anchors — pin them too
    anchors = anchor_records()
    assert PROBE_RECORD in anchors and UNRELATED_RECORD in anchors
    assert {record_key(record) for record in anchors}.isdisjoint(train_record_keys)
    assert set(anchor_keys()).isdisjoint(train_keys)


# ── 3. byte determinism ───────────────────────────────────────────────────────


@pytest.mark.parametrize("recipe", [MERGE_V1, RELEASE_V1], ids=["merge", "release"])
def test_generator_double_run_is_byte_identical(recipe) -> None:
    first = build_eval_set(recipe, seed=DEFAULT_SEED)
    second = build_eval_set(recipe, seed=DEFAULT_SEED)
    assert eval_set_jsonl(first.probes) == eval_set_jsonl(second.probes)
    assert first.eval_set_sha256 == second.eval_set_sha256
    assert eval_set_manifest_bytes(first.probes) == eval_set_manifest_bytes(
        second.probes
    )


def test_fingerprint_moves_when_probe_semantics_move() -> None:
    """The fingerprinted object carries {record, candidate, similarity,
    label} — a flipped label is a DIFFERENT set (new fingerprint)."""
    probes = list(build_eval_set(MERGE_V1, seed=DEFAULT_SEED).probes)
    flipped = [
        dataclasses.replace(
            probes[0],
            label="not-duplicate" if probes[0].label == "duplicate" else "duplicate",
        )
    ] + probes[1:]
    assert eval_set_sha256(flipped) != eval_set_sha256(probes)
    nudged = [
        dataclasses.replace(probes[0], similarity=probes[0].similarity + 0.01)
    ] + probes[1:]
    assert eval_set_sha256(nudged) != eval_set_sha256(probes)


# ── 4. recipes and loading ────────────────────────────────────────────────────


def test_merge_recipe_shape() -> None:
    eval_set = build_eval_set(MERGE_V1, seed=DEFAULT_SEED)
    assert eval_set.role == "merge"
    assert len(eval_set.probes) == 60
    expected = {
        PROBE_CLASS_IDENTITY_SELF: 8,
        PROBE_CLASS_NEAR_IDENTITY: 8,
        PROBE_CLASS_PAIR_SYMMETRY: 10,
        PROBE_CLASS_MONOTONICITY: 10,
        PROBE_CLASS_FAR_NEGATIVE: 10,
        PROBE_CLASS_TYPE_LANG_MISMATCH: 8,
        PROBE_CLASS_DEGENERATE: 6,
        PROBE_CLASS_TRANSLATION_TWINS: 0,  # honest LA-2 slot
        PROBE_CLASS_LLM_PARAPHRASE: 0,
        PROBE_CLASS_LLM_NEAR_TOPIC: 0,
    }
    assert eval_set.per_class == expected
    labels = {probe.label for probe in eval_set.probes}
    assert labels == {"duplicate", "not-duplicate"}
    for probe in eval_set.probes:
        assert -1.0 <= probe.similarity <= 1.0
        assert probe.source == "procedural"
    # ladder rows keep the COS_LADDER cosines inside their group
    ladder_groups: dict[str, list[float]] = {}
    for probe in eval_set.probes:
        if probe.probe_class == PROBE_CLASS_MONOTONICITY:
            ladder_groups.setdefault(probe.group, []).append(probe.similarity)
    assert ladder_groups
    for cosines in ladder_groups.values():
        assert sorted(cosines) == sorted(float(c) for c in COS_LADDER)


def test_release_recipe_min_200() -> None:
    eval_set = build_eval_set(RELEASE_V1, seed=DEFAULT_SEED)
    assert eval_set.role == "release"
    assert len(eval_set.probes) >= 200
    assert eval_set.per_class[PROBE_CLASS_FAR_NEGATIVE] >= 60
    # every canonical procedural class is present with a real count
    for name in (
        PROBE_CLASS_IDENTITY_SELF,
        PROBE_CLASS_NEAR_IDENTITY,
        PROBE_CLASS_PAIR_SYMMETRY,
        PROBE_CLASS_MONOTONICITY,
        PROBE_CLASS_FAR_NEGATIVE,
        PROBE_CLASS_TYPE_LANG_MISMATCH,
        PROBE_CLASS_DEGENERATE,
    ):
        assert eval_set.per_class[name] > 0


def _write_set(tmp_path: Path, eval_set) -> Path:
    jsonl_path = tmp_path / f"{eval_set.set_id}.jsonl"
    jsonl_path.write_text(eval_set_jsonl(eval_set.probes), encoding="utf-8")
    (tmp_path / f"{eval_set.set_id}.meta.json").write_text(
        json.dumps(
            {
                "set_id": eval_set.set_id,
                "role": eval_set.role,
                "eval_set_sha256": eval_set.eval_set_sha256,
            }
        ),
        encoding="utf-8",
    )
    return jsonl_path


def test_load_reverifies_fingerprint_and_rejects_tamper(tmp_path: Path) -> None:
    eval_set = build_eval_set(MERGE_V1, seed=DEFAULT_SEED)
    jsonl_path = _write_set(tmp_path, eval_set)
    assert load_eval_set(jsonl_path).eval_set_sha256 == eval_set.eval_set_sha256

    # tamper ONE label in the frozen file → the run is VOID
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["label"] = "not-duplicate" if row["label"] == "duplicate" else "duplicate"
    lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(EvalSetError, match="mismatch"):
        load_eval_set(jsonl_path)


def test_load_rejects_unknown_class_bad_label_bad_meta(tmp_path: Path) -> None:
    eval_set = build_eval_set(MERGE_V1, seed=DEFAULT_SEED)
    jsonl_path = _write_set(tmp_path, eval_set)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    def _rewrite(first_line: dict) -> None:
        lines[0] = json.dumps(first_line, ensure_ascii=False, sort_keys=True)
        jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    row = json.loads(lines[0])
    good_label = row["label"]

    row["class"] = "totally-unknown-class"
    _rewrite(row)
    with pytest.raises(EvalSetError, match="unknown probe class"):
        load_eval_set(jsonl_path)

    row["class"] = PROBE_CLASS_IDENTITY_SELF
    row["label"] = "maybe"
    _rewrite(row)
    with pytest.raises(EvalSetError, match="bad label"):
        load_eval_set(jsonl_path)

    row["label"] = good_label
    row["similarity"] = 7.5
    _rewrite(row)
    with pytest.raises(EvalSetError, match="outside"):
        load_eval_set(jsonl_path)

    # meta missing the pinned sha → load refused
    row["similarity"] = 1.0
    _rewrite(row)
    (tmp_path / "merge-v1.meta.json").write_text(
        json.dumps({"set_id": "merge-v1", "role": "merge"}), encoding="utf-8"
    )
    with pytest.raises(EvalSetError, match="missing 'eval_set_sha256'"):
        load_eval_set(jsonl_path)


def test_load_rejects_duplicate_pair_ids(tmp_path: Path) -> None:
    eval_set = build_eval_set(MERGE_V1, seed=DEFAULT_SEED)
    jsonl_path = tmp_path / "merge-v1.jsonl"
    lines = eval_set_jsonl(eval_set.probes).splitlines()
    lines[1] = lines[0]  # duplicate the first row wholesale
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # recompute a matching meta so the fingerprint check passes and the
    # dedicated pair-id guard fires
    probes = [
        EvalProbe(
            pair_id=json.loads(line)["pair_id"],
            probe_class=json.loads(line)["class"],
            record=_side(json.loads(line)["record"]),
            candidate=_side(json.loads(line)["candidate"]),
            similarity=json.loads(line)["similarity"],
            label=json.loads(line)["label"],
            source=json.loads(line)["source"],
            group=json.loads(line).get("group"),
        )
        for line in lines
        if line
    ]
    (tmp_path / "merge-v1.meta.json").write_text(
        json.dumps(
            {
                "set_id": "merge-v1",
                "role": "merge",
                "eval_set_sha256": eval_set_sha256(probes),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvalSetError, match="duplicate pair_id"):
        load_eval_set(jsonl_path)


def _side(side: dict):
    from cortex.features.pair import PairRecord

    return PairRecord(
        title=side["title"],
        body=side["body"],
        tags=tuple(side.get("tags", ())),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


# ── 5. the committed frozen sets ──────────────────────────────────────────────


@pytest.mark.parametrize("set_id", ["merge-v1", "release-v1"])
def test_committed_sets_match_their_pins(set_id: str) -> None:
    """The freeze: committed JSONL == deterministic rebuild, sha == meta,
    manifest.txt == §5 bytes, counts == meta."""
    recipe = MERGE_V1 if set_id == "merge-v1" else RELEASE_V1
    rebuilt = build_eval_set(recipe, seed=DEFAULT_SEED)

    jsonl_path = EVALSETS_DIR / f"{set_id}.jsonl"
    assert jsonl_path.is_file(), "the frozen set must be committed"
    assert jsonl_path.read_text(encoding="utf-8") == eval_set_jsonl(rebuilt.probes)

    meta = json.loads((EVALSETS_DIR / f"{set_id}.meta.json").read_text("utf-8"))
    assert meta["eval_set_sha256"] == rebuilt.eval_set_sha256
    assert meta["n_pairs"] == len(rebuilt.probes)
    assert meta["per_class"] == rebuilt.per_class

    manifest_path = EVALSETS_DIR / f"{set_id}.manifest.txt"
    assert manifest_path.read_bytes() == eval_set_manifest_bytes(rebuilt.probes)

    loaded = load_eval_set(jsonl_path)
    assert loaded.eval_set_sha256 == meta["eval_set_sha256"]
    assert loaded.role == meta["role"]


# ── 6. LLM slots (LA-2 mechanics) ─────────────────────────────────────────────


def _slot_row(pair_id: str, probe_class: str, source: str) -> dict:
    ru = EVAL_TOPICS[0].as_record()  # backup-rotation RU
    en = EVAL_TOPICS[8].as_record()  # backup-rotation EN twin
    return {
        "pair_id": pair_id,
        "class": probe_class,
        "source": source,
        "similarity": 0.8,
        "label": "duplicate",
        "record": {
            "title": ru.title,
            "body": ru.body,
            "tags": list(ru.tags),
            "language": ru.language,
            "record_type": ru.record_type,
        },
        "candidate": {
            "title": en.title,
            "body": en.body,
            "tags": list(en.tags),
            "language": en.language,
            "record_type": en.record_type,
        },
    }


def test_llm_slots_ride_in_when_valid(tmp_path: Path) -> None:
    slots_path = tmp_path / "llm-slots-la2.jsonl"
    slots_path.write_text(
        json.dumps(
            _slot_row("tt-0000", PROBE_CLASS_TRANSLATION_TWINS, "llm-batch:la2"),
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps(
            _slot_row("lp-0000", PROBE_CLASS_LLM_PARAPHRASE, "llm-batch:la2"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    slots = load_llm_slots(slots_path)
    assert len(slots) == 2
    enriched = build_eval_set(MERGE_V1, seed=DEFAULT_SEED, llm_slots=slots)
    assert enriched.per_class[PROBE_CLASS_TRANSLATION_TWINS] == 1
    assert enriched.per_class[PROBE_CLASS_LLM_PARAPHRASE] == 1
    assert len(enriched.probes) == 62
    assert (
        enriched.eval_set_sha256
        != build_eval_set(MERGE_V1, seed=DEFAULT_SEED).eval_set_sha256
    )
    # empty/absent slots → the honest v1 shape (zero-count slot classes)
    assert load_llm_slots(tmp_path / "absent.jsonl") == []


def test_llm_slots_reject_out_of_contract_rows(tmp_path: Path) -> None:
    slots_path = tmp_path / "llm-slots-la2.jsonl"
    # a PROCEDURAL class can never come from outside
    slots_path.write_text(
        json.dumps(
            _slot_row("x-0000", PROBE_CLASS_IDENTITY_SELF, "llm-batch:la2"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalSetError, match="not an LLM-slot class"):
        build_eval_set(
            MERGE_V1, seed=DEFAULT_SEED, llm_slots=load_llm_slots(slots_path)
        )

    slots_path.write_text(
        json.dumps(
            _slot_row("x-0001", PROBE_CLASS_TRANSLATION_TWINS, "hand-made"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(EvalSetError, match="source must be"):
        build_eval_set(
            MERGE_V1, seed=DEFAULT_SEED, llm_slots=load_llm_slots(slots_path)
        )


# ── 7. the runner on a fixture bundle (never prod weights) ────────────────────


def _eval_set_rows(eval_set):
    """Eval probes → train-manifest-shaped rows (the fixture model's diet)."""
    rows = []
    for probe in eval_set.probes:
        rows.append(
            {
                "pair_id": probe.pair_id,
                "record": probe_row(probe)["record"],
                "candidate": probe_row(probe)["candidate"],
                "similarity": probe.similarity,
                "label": probe.label,
            }
        )
    return rows


def _write_bundle(tmp_path: Path, model: DBoostModel, name: str) -> Path:
    out = tmp_path / f"{name}.onnx"
    props = build_metadata_props(
        embedder_pin="nano:sha256:" + "ab" * 32,
        corpus_fingerprint="cd" * 32,
        trained_at="2026-10-04T00:00:00+00:00",
        candidate=CANDIDATE_D,
        feature_names=FEATURE_NAMES,
    )
    model.export_onnx(out, metadata_props=props)
    manifest = {
        "name": "vesma-cortex-v1",
        "version": "1",
        "sha256": sha256_file(out),
        "candidate": CANDIDATE_D,
        "features": list(FEATURE_NAMES),
        "feature_set_sha256": props["feature_set_sha256"],
    }
    (tmp_path / f"{name}.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return tmp_path


def _permute_columns(vectors, swap: tuple[int, int] = (0, 9)):
    """Feature-column desync simulator (#480) — the fixture negative."""
    i, j = swap
    out = []
    for vector in vectors:
        values = list(vector.values)
        values[i], values[j] = values[j], values[i]
        out.append(vector.__class__(vector.names, tuple(values)))
    return out


@pytest.fixture(scope="module")
def fixture_bundles(tmp_path_factory) -> tuple[Path, Path]:
    """(healthy, desynced) bundles trained ON the merge probes + filler.

    The diet EXCLUDES the type/lang rows on purpose: they are cos=1.0
    negatives, and a tiny model trained on them resolves the cos=1.0
    boundary a hair below its cos=0.99 response — a monotonicity wiggle
    of the fixture artifact, not of the runner (prod B1 passes the same
    gate; the fixture exists to verify plumbing, not model quality).
    """
    eval_set = build_eval_set(MERGE_V1, seed=DEFAULT_SEED)
    rows = [
        row for row in _eval_set_rows(eval_set) if not row["pair_id"].startswith("tml-")
    ] + make_pair_rows(60)
    model = DBoostModel()
    model.train(rows_to_vectors(rows), rows_labels(rows), GRID_D[1], calibrate=True)
    healthy = _write_bundle(tmp_path_factory.mktemp("healthy"), model, "model")

    desynced = DBoostModel()
    desynced.train(
        _permute_columns(rows_to_vectors(rows)),
        rows_labels(rows),
        GRID_D[1],
        calibrate=True,
    )
    unhealthy = _write_bundle(tmp_path_factory.mktemp("desynced"), desynced, "model")
    return healthy, unhealthy


def test_runner_passes_healthy_fixture_and_reports_contract(
    fixture_bundles: tuple[Path, Path],
) -> None:
    healthy, _ = fixture_bundles
    report = run_layer_a(healthy, load_eval_set(EVALSETS_DIR / "merge-v1.jsonl"))
    assert report.passed, "\n".join(
        f"{g.name}: {g.detail}" for g in report.failed_gates
    )
    assert report.set_id == "merge-v1" and report.role == "merge"
    assert report.n_pairs == 60
    assert report.eval_set_sha256 == eval_set_sha256(
        load_eval_set(EVALSETS_DIR / "merge-v1.jsonl").probes
    )
    assert len(report.weights_sha256) == 64
    assert 0.0 <= report.balanced_accuracy <= 1.0
    assert {b.probe_class for b in report.per_class} <= set(ALL_PROBE_CLASSES)
    names = [check.name for check in report.bundle_contract]
    assert names == ["bundle_integrity", "feature_contract", "candidate_supported"]
    gate_names = [gate.name for gate in report.gate_evaluations]
    assert gate_names == [
        "identity-self_floor",
        "monotonicity",
        "pair_symmetry",
        "near-identity-twin_floor",
        "far-negative_ceiling",
    ]


def test_runner_catches_column_desync_on_self_pair(
    fixture_bundles: tuple[Path, Path],
) -> None:
    """The #480 regression pin through the NEW instrument: a graph whose
    column order diverges from the claimed contract must fail the gate."""
    _, desynced = fixture_bundles
    report = run_layer_a(desynced, load_eval_set(EVALSETS_DIR / "merge-v1.jsonl"))
    assert not report.passed
    failed = {gate.name for gate in report.failed_gates}
    # behavioral catch (the #480 shape): the desynced graph inverts the
    # self-pair. The bundle_contract block stays GREEN here on purpose —
    # the lying manifest is internally consistent (manifest == metadata);
    # the desync is graph-vs-contract and only the probes can see it.
    assert "identity-self_floor" in failed


def test_evaluate_gates_pure_function_pins() -> None:
    """The gate math without ONNX: a healthy response passes, then
    monotonicity violation, symmetry mismatch, and structural blindness
    (empty gated class) all fail."""
    probes = list(build_eval_set(MERGE_V1, seed=DEFAULT_SEED).probes)

    # healthy response: high P on duplicate-side classes, P(cos) itself on
    # the ladder (monotone by construction), low P on far negatives
    healthy = {
        PROBE_CLASS_IDENTITY_SELF: 1.0,
        PROBE_CLASS_NEAR_IDENTITY: 0.99,
        PROBE_CLASS_PAIR_SYMMETRY: 0.99,
        PROBE_CLASS_MONOTONICITY: None,  # filled per row below
        PROBE_CLASS_FAR_NEGATIVE: 0.0,
        PROBE_CLASS_TYPE_LANG_MISMATCH: 0.9,
        PROBE_CLASS_DEGENERATE: 0.9,
    }
    probs = []
    for probe in probes:
        value = healthy[probe.probe_class]
        probs.append(probe.similarity if value is None else value)
    assert all(gate.passed for gate in evaluate_gates(probes, probs))

    # monotonicity violation inside one ladder group: bump step 0.5 above
    # its 0.8 upper step — an OUTSIDE-the-razor-zone increase, still
    # asserted under the zoned v2 rule (eval-methodology §10.2). The old
    # v1 bump (a rise across the 0.99 → 0.95 zone crossing) is NOT a
    # violation anymore under v2 — the la4 change.
    probs = [0.99] * len(probes)
    for i, probe in enumerate(probes):
        if (
            probe.probe_class == PROBE_CLASS_MONOTONICITY
            and probe.group == "lad-0000"
            and probe.pair_id.endswith("-p4")
        ):
            probs[i] = 1.0
    gates = {g.name: g for g in evaluate_gates(probes, probs)}
    assert not gates["monotonicity"].passed
    assert "lad-0000" in gates["monotonicity"].detail

    # v2 zoning: a rise ACROSS the zone crossing (0.99 → 0.95, steps whose
    # upper point sits strictly above cos 0.95) is no longer a violation —
    # the razor band is two-valued by policy v1.1 §8 (the la4 finding:
    # B2-v42 grew +5.5e-05…+1.38e-04 exactly there and the v1 rule read
    # the ratified zone crossing as a defect).
    probs = [0.0] * len(probes)
    for i, probe in enumerate(probes):
        if probe.probe_class == PROBE_CLASS_MONOTONICITY and probe.group == "lad-0000":
            step = float(probe.similarity)
            probs[i] = 0.99 if step in (1.0, 0.99) else (0.995 if step == 0.95 else 0.0)
    gates = {g.name: g for g in evaluate_gates(probes, probs)}
    assert gates["monotonicity"].passed, gates["monotonicity"].detail
    # …but a rise WHOSE upper step lies at/below the zone edge (0.8 → 0.5)
    # still fails — outside the zone the non-increase holds, tol unchanged:
    probs = [0.0] * len(probes)
    for i, probe in enumerate(probes):
        if probe.probe_class == PROBE_CLASS_MONOTONICITY and probe.group == "lad-0000":
            step = float(probe.similarity)
            probs[i] = (
                0.99 if step in (1.0, 0.99, 0.95) else (0.5 if step == 0.5 else 0.0)
            )
    gates = {g.name: g for g in evaluate_gates(probes, probs)}
    assert not gates["monotonicity"].passed
    assert "lad-0000" in gates["monotonicity"].detail

    # symmetry mismatch on one group
    probs = [0.99] * len(probes)
    for i, probe in enumerate(probes):
        if (
            probe.probe_class == PROBE_CLASS_PAIR_SYMMETRY
            and probe.group == "sym-0000"
            and probe.pair_id.endswith("-rev")
        ):
            probs[i] = 0.5
    gates = {g.name: g for g in evaluate_gates(probes, probs)}
    assert not gates["pair_symmetry"].passed

    # structural blindness: a gated class missing from the set → FAIL
    blind = [p for p in probes if p.probe_class != PROBE_CLASS_IDENTITY_SELF]
    gates = {g.name: g for g in evaluate_gates(blind, [0.99] * len(blind))}
    assert not gates["identity-self_floor"].passed
    assert "structural blindness" in gates["identity-self_floor"].detail

    # corridor semantics: far-negative ceiling at exactly 0.5 is a FAIL
    probs = [0.99] * len(probes)
    for i, probe in enumerate(probes):
        if probe.probe_class == PROBE_CLASS_FAR_NEGATIVE:
            probs[i] = 0.5
    gates = {g.name: g for g in evaluate_gates(probes, probs)}
    assert not gates["far-negative_ceiling"].passed


# ── 8. the LA-2 LLM batch and the v2 sets ─────────────────────────────────────

BATCH_PATH = EVALSETS_DIR / "llm-batches" / "la2-llm-batch-1.jsonl"
MERGE_V2_ID = "merge-v2"
RELEASE_V2_ID = "release-v2"
MERGE_SLOTS_PER_CLASS = 5

SLOT_CLASSES = {
    PROBE_CLASS_TRANSLATION_TWINS: ("tt", "duplicate"),
    PROBE_CLASS_LLM_PARAPHRASE: ("lp", "duplicate"),
    PROBE_CLASS_LLM_NEAR_TOPIC: ("ln", "not-duplicate"),
}


@pytest.fixture(scope="module")
def la2_batch() -> list[dict]:
    rows = load_llm_slots(BATCH_PATH)
    assert rows, "the LA-2 batch must be committed"
    return rows


def _side_as_record(side: dict):
    from cortex.features.pair import PairRecord

    return PairRecord(
        title=side["title"],
        body=side["body"],
        tags=tuple(side.get("tags", ())),
        language=side.get("language"),
        record_type=side.get("record_type"),
    )


def _select_merge_slots(rows: list[dict]) -> list[dict]:
    """The documented merge rule: first MERGE_SLOTS_PER_CLASS rows per class."""
    counts = dict.fromkeys(SLOT_CLASSES, 0)
    selected = []
    for row in rows:
        name = row["class"]
        if counts[name] < MERGE_SLOTS_PER_CLASS:
            selected.append(row)
            counts[name] += 1
    return selected


def test_llm_batch_fills_all_three_slots(la2_batch: list[dict]) -> None:
    assert len(la2_batch) == 36
    counts = dict.fromkeys(SLOT_CLASSES, 0)
    for row in la2_batch:
        name = row["class"]
        assert name in SLOT_CLASSES
        counts[name] += 1
        prefix, label = SLOT_CLASSES[name]
        assert row["label"] == label
        assert row["source"] == "llm-batch:la2"
        assert row["pair_id"].startswith(prefix + "-")
    assert counts == dict.fromkeys(SLOT_CLASSES, 12)


def test_llm_batch_language_and_type_mix(la2_batch: list[dict]) -> None:
    types: set[str] = set()
    for row in la2_batch:
        rec, cand = row["record"], row["candidate"]
        types.add(rec["record_type"])
        types.add(cand["record_type"])
        if row["class"] == PROBE_CLASS_TRANSLATION_TWINS:
            assert rec["language"] != cand["language"]
            assert {rec["language"], cand["language"]} == {"ru", "en"}
        else:
            assert rec["language"] == cand["language"]
    assert types == {"note", "fact", "decision"}
    for name in (PROBE_CLASS_LLM_PARAPHRASE, PROBE_CLASS_LLM_NEAR_TOPIC):
        langs = [row["record"]["language"] for row in la2_batch if row["class"] == name]
        assert langs.count("ru") == 6 and langs.count("en") == 6


def test_llm_batch_texts_deduped_and_disjoint(la2_batch: list[dict]) -> None:
    """No two identical texts anywhere: within the batch, against the v1
    sets, and against the train/eval topic surfaces."""
    seen_keys: set[str] = set()
    seen_titles: set[str] = set()
    for set_id in ("merge-v1", "release-v1"):
        for probe in load_eval_set(EVALSETS_DIR / f"{set_id}.jsonl").probes:
            for side in (probe.record, probe.candidate):
                seen_keys.add(record_key(side))
                seen_titles.add(side.title)
    for topic in TOPICS:
        seen_keys.add(record_key(topic.as_record()))
        seen_titles.add(topic.title)
    for topic in EVAL_TOPICS:
        seen_keys.add(record_key(topic.as_record()))
        seen_titles.add(topic.title)
    for anchor in (PROBE_RECORD, UNRELATED_RECORD):
        seen_keys.add(record_key(anchor))
        seen_titles.add(anchor.title)

    for row in la2_batch:
        for side_raw in (row["record"], row["candidate"]):
            side = _side_as_record(side_raw)
            key = record_key(side)
            assert key not in seen_keys, f"duplicate text in batch: {side.title!r}"
            assert side.title not in seen_titles
            seen_keys.add(key)
            seen_titles.add(side.title)


def test_llm_batch_pairs_pass_through_pair_features(la2_batch: list[dict]) -> None:
    """The #480 rule on the batch itself: every pair scores through
    cortex.features.pair.features with the frozen 13-name contract."""
    import math

    for row in la2_batch:
        vector = features(
            _side_as_record(row["record"]),
            _side_as_record(row["candidate"]),
            float(row["similarity"]),
        )
        assert vector.names == FEATURE_NAMES
        assert len(vector.values) == len(FEATURE_NAMES)
        assert all(math.isfinite(value) for value in vector.values)


@pytest.mark.parametrize("set_id", [MERGE_V2_ID, RELEASE_V2_ID])
def test_v2_committed_sets_match_their_pins(set_id: str, la2_batch: list[dict]) -> None:
    """The v2 freeze: committed JSONL == deterministic rebuild over the
    v1 recipe + the batch (merge: first-5-per-class subset; release: full),
    sha == meta, manifest.txt == §5 bytes, counts == meta."""
    recipe = dataclasses.replace(
        MERGE_V1 if set_id == MERGE_V2_ID else RELEASE_V1, set_id=set_id
    )
    slots = _select_merge_slots(la2_batch) if set_id == MERGE_V2_ID else la2_batch
    rebuilt = build_eval_set(recipe, seed=DEFAULT_SEED, llm_slots=slots)

    jsonl_path = EVALSETS_DIR / f"{set_id}.jsonl"
    assert jsonl_path.is_file(), "the frozen v2 set must be committed"
    assert jsonl_path.read_text(encoding="utf-8") == eval_set_jsonl(rebuilt.probes)

    meta = json.loads((EVALSETS_DIR / f"{set_id}.meta.json").read_text("utf-8"))
    assert meta["set_id"] == set_id
    assert meta["eval_set_sha256"] == rebuilt.eval_set_sha256
    assert meta["n_pairs"] == len(rebuilt.probes)
    assert meta["per_class"] == rebuilt.per_class
    assert meta["llm_batch_rows"] == len(slots)
    assert meta["llm_generator"] == "la2-llm-batch-1"

    manifest_path = EVALSETS_DIR / f"{set_id}.manifest.txt"
    assert manifest_path.read_bytes() == eval_set_manifest_bytes(rebuilt.probes)

    loaded = load_eval_set(jsonl_path)
    assert loaded.eval_set_sha256 == meta["eval_set_sha256"]
    assert loaded.role == meta["role"]


@pytest.mark.parametrize(
    "set_id,base_id",
    [(MERGE_V2_ID, "merge-v1"), (RELEASE_V2_ID, "release-v1")],
)
def test_v2_double_run_byte_identical_and_fingerprint_moves(
    set_id: str, base_id: str, la2_batch: list[dict]
) -> None:
    recipe = dataclasses.replace(
        MERGE_V1 if base_id == "merge-v1" else RELEASE_V1, set_id=set_id
    )
    slots = _select_merge_slots(la2_batch) if set_id == MERGE_V2_ID else la2_batch
    first = build_eval_set(recipe, seed=DEFAULT_SEED, llm_slots=slots)
    second = build_eval_set(recipe, seed=DEFAULT_SEED, llm_slots=slots)
    assert eval_set_jsonl(first.probes) == eval_set_jsonl(second.probes)
    assert first.eval_set_sha256 == second.eval_set_sha256

    v1 = load_eval_set(EVALSETS_DIR / f"{base_id}.jsonl")
    assert first.eval_set_sha256 != v1.eval_set_sha256
    # the slots moved from honest zeros to the batch counts
    for name in SLOT_CLASSES:
        assert v1.per_class[name] == 0
        assert first.per_class[name] > 0
    # the procedural base stays byte-stable across the version bump: the
    # v2 probe rows minus the slot rows == the v1 rows, ids and all
    v1_rows = eval_set_jsonl(v1.probes).splitlines()
    v2_rows = eval_set_jsonl(first.probes).splitlines()
    assert v2_rows[: len(v1_rows)] == v1_rows


def test_v2_merge_slots_are_the_documented_subset(
    la2_batch: list[dict],
) -> None:
    """merge-v2 carries the first 5 rows per class, batch order; release-v2
    carries all 36; the merge slot block is a prefix-preserving subset."""
    merge_set = load_eval_set(EVALSETS_DIR / f"{MERGE_V2_ID}.jsonl")
    release_set = load_eval_set(EVALSETS_DIR / f"{RELEASE_V2_ID}.jsonl")
    assert len(merge_set.probes) == 60 + 3 * MERGE_SLOTS_PER_CLASS
    assert len(release_set.probes) == 201 + len(la2_batch)

    expected_ids = [row["pair_id"] for row in _select_merge_slots(la2_batch)]
    merge_slot_ids = [
        probe.pair_id for probe in merge_set.probes if probe.source == "llm-batch:la2"
    ]
    release_slot_ids = [
        probe.pair_id for probe in release_set.probes if probe.source == "llm-batch:la2"
    ]
    assert merge_slot_ids == expected_ids
    assert release_slot_ids == [row["pair_id"] for row in la2_batch]
    assert set(merge_slot_ids) < set(release_slot_ids)
    # the procedural part of merge-v2 is exactly merge-v1 (same pair ids)
    merge_v1_ids = [
        probe.pair_id for probe in load_eval_set(EVALSETS_DIR / "merge-v1.jsonl").probes
    ]
    assert [p.pair_id for p in merge_set.probes][: len(merge_v1_ids)] == merge_v1_ids
