"""Stage-2 orchestrator tests — the final-sprint glue (scripts/stage2_run.py).

Covers the prereg v2 gates the orchestrator enforces: label floors are an
authoritative NO-DATA stop (exit 5), disputed pairs are excluded BEFORE the
frozen split, ⌈0.3·n⌉ per stratum with stratum preservation, physical
holdout isolation (no pair_id overlap, no labels in the train manifest or
the holdout pairs file), labels-fingerprint reproducibility, the frozen
corpus-fingerprint check, the eval single-shot guard with its own run-log
placement, and the dry-run smoke (stages 1–5 through the real code path on
a mini synthetic stand-in — canon-data never touched).

The chain tests run the REAL cortex CLI in-process on tmp fixtures —
no network, no canon-data reads.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from cortex.data.fingerprints import corpus_fingerprint, manifest_bytes

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "stage2_run.py"
_spec = importlib.util.spec_from_file_location("stage2_run", _SCRIPT)
stage2 = importlib.util.module_from_spec(_spec)
sys.modules["stage2_run"] = stage2  # dataclasses resolve types via sys.modules
_spec.loader.exec_module(stage2)


# ── fixture builders ──────────────────────────────────────────────────────────


def _side(seed: str, tags: tuple[str, ...] = ("t",)) -> dict:
    return {
        "title": f"Заголовок {seed}",
        "body": f"Тело записи {seed}: локальная память, без сети, для тестового корпуса.",
        "tags": list(tags),
        "created_at": "2026-08-01T10:00:00Z",
    }


def _write_canon_fixture(
    tmp_path: Path,
    strata_counts: dict[str, int],
    label_of: dict[str, str] | None = None,
    disputed_ids: set[str] | None = None,
) -> tuple[Path, Path]:
    """Mini canon-shaped corpus (pairs.jsonl + manifest.txt + strata.csv +
    cosines.csv) + labels.csv. ``label_of`` maps stratum → label; disputed
    ids get the disputed label + a one-line reason."""
    disputed_ids = disputed_ids or set()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True)
    rows: list[dict] = []
    cosine_lines = ["pair_id,cosine"]
    strata_lines = ["pair_id,stratum,band,fallback"]
    label_lines = ["pair_id,label,note"]
    for stratum, count in strata_counts.items():
        for i in range(1, count + 1):
            pid = f"{stratum}-{i:03d}"
            rows.append({"pair_id": pid, "a": _side(f"{pid}-a"), "b": _side(f"{pid}-b")})
            cosine_lines.append(f"{pid},{0.5 + i * 0.001:.4f}")
            strata_lines.append(f"{pid},{stratum},,no")
            if pid in disputed_ids:
                label_lines.append(f"{pid},disputed,owner cannot decide")
            else:
                label = (label_of or {}).get(stratum, "duplicate")
                label_lines.append(f"{pid},{label},")
    with (corpus_dir / "pairs.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digests = {row["pair_id"]: stage2.canon_pair_digest(row) for row in rows}
    (corpus_dir / "manifest.txt").write_bytes(manifest_bytes(digests.items()))
    (corpus_dir / "strata.csv").write_text("\n".join(strata_lines) + "\n", encoding="utf-8")
    (corpus_dir / "cosines.csv").write_text("\n".join(cosine_lines) + "\n", encoding="utf-8")
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return corpus_dir, labels_csv


def _corpus_ids(corpus_dir: Path) -> set[str]:
    return {
        json.loads(line)["pair_id"]
        for line in (corpus_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _corpus_rows(corpus_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (corpus_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mini_fingerprint(corpus_dir: Path) -> str:
    digests = {row["pair_id"]: stage2.canon_pair_digest(row) for row in _corpus_rows(corpus_dir)}
    return corpus_fingerprint(manifest_bytes(digests.items()))


def _label_map(labels_csv: Path) -> dict[str, str]:
    """Parse the fixture labels.csv into the normalized label map — bypasses
    the prereg floors (small fixtures); the floors are tested separately."""
    labels: dict[str, str] = {}
    for line in labels_csv.read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        pid, raw, _note = (line.split(",") + [""])[:3]
        raw = raw.strip()
        if not raw:
            labels[pid.strip()] = "disputed"  # blank label treated as disputed in split tests
            continue
        normalized = stage2.normalize_label(raw)
        assert normalized is not None, raw
        labels[pid.strip()] = normalized
    return labels


def _ingest(tmp_path: Path, strata_counts: dict[str, int], label_of: dict[str, str], disputed_ids: set[str] | None = None):
    corpus_dir, labels_csv = _write_canon_fixture(tmp_path, strata_counts, label_of, disputed_ids)
    return stage2.ingest_labels(labels_csv, _corpus_ids(corpus_dir))


def _join(
    tmp_path: Path,
    strata_counts: dict[str, int],
    label_of: dict[str, str],
    disputed_ids: set[str] | None = None,
):
    """Corpus + normalized labels for join_and_split (floors bypassed)."""
    corpus_dir, labels_csv = _write_canon_fixture(tmp_path, strata_counts, label_of, disputed_ids)
    labels = _label_map(labels_csv)
    return _corpus_rows(corpus_dir), labels, corpus_dir


def _split(tmp_path: Path, strata_counts: dict[str, int], label_of: dict[str, str], disputed_ids: set[str] | None = None):
    rows, labels, corpus_dir = _join(tmp_path, strata_counts, label_of, disputed_ids)
    digests = {row["pair_id"]: stage2.canon_pair_digest(row) for row in rows}
    return stage2.join_and_split(
        corpus_rows=rows,
        digests=digests,
        strata=stage2.load_strata(corpus_dir, rows),
        cosines=stage2.load_cosines(corpus_dir, rows),
        labels=labels,
        vectors=None,
        out_root=tmp_path / "out",
    )


def _read_ids(path: Path) -> set[str]:
    return {json.loads(line)["pair_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _mini_synth(tmp_path: Path) -> Path:
    """Mini synthetic stand-in (pairs.sim shape, no vectors) for the dry-run
    smoke: label floors and the holdout class floor pass by construction."""
    counts = {"paraphrase": 66, "near-topic": 66, "broken-field": 20, "trivial-negative": 13}
    path = tmp_path / "mini-pairs.sim.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for strategy, count in counts.items():
            label = "duplicate" if strategy == "paraphrase" else "not-duplicate"
            for i in range(count):
                pid = f"{strategy[:2]}{i:04d}--zz{i:04d}"
                row = {
                    "pair_id": pid,
                    "record": _side(f"{pid}-a", ("synth",)),
                    "candidate": _side(f"{pid}-b", ("synth",)),
                    "label": label,
                    "strategy": strategy,
                    "similarity": 0.91,
                    "seed": i,
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


# ── stage 1: label floors are an authoritative stop ───────────────────────────


def test_duplicate_floor_violation_is_no_data(tmp_path: Path) -> None:
    # 39 duplicates (< 40) — NO-DATA, the run is forbidden (prereg W5b)
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        _ingest(tmp_path, {"P1": 39, "N1": 100}, {"P1": "duplicate", "N1": "not-duplicate"})
    assert excinfo.value.kind == "NO-DATA"
    assert excinfo.value.exit_code == stage2.EXIT_NO_DATA
    assert "duplicate 39/40" in str(excinfo.value)


def test_not_duplicate_floor_violation_is_no_data(tmp_path: Path) -> None:
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        _ingest(tmp_path, {"P1": 100, "N1": 39}, {"P1": "duplicate", "N1": "not-duplicate"})
    assert excinfo.value.kind == "NO-DATA"
    assert "not-duplicate 39/40" in str(excinfo.value)


def test_disputed_floor_violation_is_no_data(tmp_path: Path) -> None:
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        _ingest(
            tmp_path,
            {"P1": 100, "N1": 100, "N2": 41},
            {"P1": "duplicate", "N1": "not-duplicate", "N2": "disputed"},
            disputed_ids={f"N2-{i:03d}" for i in range(1, 42)},
        )
    assert excinfo.value.kind == "NO-DATA"
    assert "disputed 41/40" in str(excinfo.value)


def test_empty_label_is_an_authoritative_stop(tmp_path: Path) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(
        tmp_path, {"P1": 50, "N1": 50}, {"P1": "duplicate", "N1": "not-duplicate"}
    )
    text = labels_csv.read_text(encoding="utf-8").replace("N1-007,not-duplicate,", "N1-007,,")
    labels_csv.write_text(text, encoding="utf-8")
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        stage2.ingest_labels(labels_csv, _corpus_ids(corpus_dir))
    assert excinfo.value.kind == "LABELS-INCOMPLETE"
    assert excinfo.value.exit_code == stage2.EXIT_NO_DATA
    assert "labeling (W5b) is unfinished" in str(excinfo.value)


def test_disputed_requires_note_and_unknown_label_refused(tmp_path: Path) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(
        tmp_path, {"P1": 50, "N1": 50}, {"P1": "duplicate", "N1": "not-duplicate"}
    )
    ids = _corpus_ids(corpus_dir)
    original = labels_csv.read_text(encoding="utf-8")

    # disputed without a note — codebook violation
    labels_csv.write_text(original.replace("P1-001,duplicate,", "P1-001,disputed,"), encoding="utf-8")
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        stage2.ingest_labels(labels_csv, ids)
    assert "disputed requires a one-line reason" in str(excinfo.value)

    # unknown label — written from the FRESH original, not the mutated one
    labels_csv.write_text(original.replace("P1-002,duplicate,", "P1-002,maybe,"), encoding="utf-8")
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        stage2.ingest_labels(labels_csv, ids)
    assert "unknown label 'maybe'" in str(excinfo.value)


def test_label_alias_normalization_and_fingerprint_reproducibility(tmp_path: Path) -> None:
    # the canon codebook spells the negative class not_duplicate — normalized
    # to the internal not-duplicate. The CORTEX scheme fingerprints the
    # normalized dict (spelling-independent); the CANON scheme fingerprints
    # the raw csv values, so different spellings yield different canon
    # fingerprints by design (they protect the file as written).
    ingest_a = _ingest(tmp_path / "a", {"P1": 50, "N1": 50}, {"P1": "duplicate", "N1": "not_duplicate"})
    ingest_b = _ingest(tmp_path / "b", {"P1": 50, "N1": 50}, {"P1": "duplicate", "N1": "not-duplicate"})
    assert set(ingest_a.labels.values()) == {"duplicate", "not-duplicate"}
    assert ingest_a.fingerprint_cortex == ingest_b.fingerprint_cortex
    assert ingest_a.fingerprint_cortex == stage2.labels_fingerprint(ingest_a.labels)
    assert ingest_a.fingerprint_canon != ingest_b.fingerprint_canon
    # same spelling → identical canon fingerprint (order-independent)
    ingest_c = _ingest(tmp_path / "c", {"P1": 50, "N1": 50}, {"P1": "duplicate", "N1": "not-duplicate"})
    assert ingest_c.fingerprint_canon == ingest_b.fingerprint_canon


def test_labels_corpus_mismatch_is_a_stop(tmp_path: Path) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(
        tmp_path, {"P1": 50, "N1": 50}, {"P1": "duplicate", "N1": "not-duplicate"}
    )
    lines = labels_csv.read_text(encoding="utf-8").splitlines()
    lines.append("P9-999,duplicate,")  # a pair absent from the corpus
    labels_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        stage2.ingest_labels(labels_csv, _corpus_ids(corpus_dir))
    assert excinfo.value.kind == "LABELS-MISMATCH"


# ── stage 2: the frozen split, floors, isolation ──────────────────────────────


def test_split_fraction_strata_preservation(tmp_path: Path) -> None:
    # two strata of 67 → ceil(0.3·67) = 21 holdout each; the pair_id-FIRST
    # members go to holdout (prereg W5c). Sizes also clear the holdout
    # class floor (21 ≥ 20 per class).
    outcome = _split(tmp_path, {"P1": 67, "N1": 67}, {"P1": "duplicate", "N1": "not-duplicate"})
    assert outcome.disputed_excluded == 0
    assert outcome.train_rows == 92 and outcome.holdout_rows == 42
    assert outcome.per_stratum == {"N1": {"train": 46, "holdout": 21}, "P1": {"train": 46, "holdout": 21}}
    holdout_ids = _read_ids(outcome.holdout_dir / "pairs.jsonl")
    all_ids = _read_ids(outcome.train_manifest) | holdout_ids
    for stratum in ("N1", "P1"):
        members = sorted(pid for pid in all_ids if pid.startswith(f"{stratum}-"))
        assert set(members[:21]) == {pid for pid in members if pid in holdout_ids}
    # stratum preservation: manifest rows carry the corpus stratum
    train_rows = [json.loads(line) for line in outcome.train_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert all(row["stratum"] == row["pair_id"].split("-", 1)[0] for row in train_rows)


def test_disputed_pairs_excluded_before_split(tmp_path: Path) -> None:
    outcome = _split(
        tmp_path,
        {"P1": 67, "N1": 67},
        {"P1": "duplicate", "N1": "not-duplicate"},
        disputed_ids={"P1-001", "N1-004"},
    )
    assert outcome.disputed_excluded == 2
    train_ids = _read_ids(outcome.train_manifest)
    holdout_ids = _read_ids(outcome.holdout_dir / "pairs.jsonl")
    assert "P1-001" not in train_ids | holdout_ids
    assert "N1-004" not in train_ids | holdout_ids
    # 66 non-disputed per stratum → 20 holdout + 46 train each
    assert outcome.train_rows == 92 and outcome.holdout_rows == 40
    # disputed never reach the holdout labels either
    holdout_labels = _read_ids(outcome.holdout_dir / "labels.jsonl")
    assert "P1-001" not in holdout_labels and "N1-004" not in holdout_labels


def test_holdout_class_floor_is_no_data(tmp_path: Path) -> None:
    # 90 duplicates vs 10 not-duplicates → the holdout holds 3
    # not-duplicates < 20 — NO-DATA, the run is forbidden (prereg W5c)
    with pytest.raises(stage2.Stage2Stop) as excinfo:
        _split(tmp_path, {"P1": 90, "N1": 10}, {"P1": "duplicate", "N1": "not-duplicate"})
    assert excinfo.value.kind == "NO-DATA"
    assert "holdout class floor" in str(excinfo.value)


def test_holdout_physical_isolation_and_label_free_pairs(tmp_path: Path) -> None:
    outcome = _split(tmp_path, {"P1": 67, "N1": 67}, {"P1": "duplicate", "N1": "not-duplicate"})
    train_ids = _read_ids(outcome.train_manifest)
    holdout_pairs = [
        json.loads(line)
        for line in (outcome.holdout_dir / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    holdout_ids = {row["pair_id"] for row in holdout_pairs}
    # the load-bearing invariant: zero pair_id intersection
    stage2.assert_no_pair_overlap(sorted(train_ids), sorted(holdout_ids))
    assert train_ids & holdout_ids == set()
    # the holdout pairs file carries NO labels — labels live only in
    # holdout/labels.jsonl, outside the train manifest's path
    assert all("label" not in row for row in holdout_pairs)
    assert _read_ids(outcome.holdout_dir / "labels.jsonl") == holdout_ids
    stage2.assert_labels_isolated(outcome.train_manifest, outcome.holdout_dir / "labels.jsonl")
    # the holdout id manifest exists and is pair_id-sorted
    manifest_lines = (outcome.holdout_dir / "manifest.txt").read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 1)[0] for line in manifest_lines] == sorted(holdout_ids)


# ── the frozen corpus fingerprint check + eval guard (CLI level) ─────────────


def test_corpus_fingerprint_mismatch_voids_the_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(
        tmp_path, {"P1": 50, "N1": 50}, {"P1": "duplicate", "N1": "not-duplicate"}
    )
    code = stage2.main(
        [
            "run",
            "--corpus",
            str(corpus_dir),
            "--labels",
            str(labels_csv),
            "--root",
            str(tmp_path / "out"),
            # no --expect-corpus-fingerprint → the frozen bcdc6e31… check fires
        ]
    )
    assert code == stage2.EXIT_CONTRACT
    assert "not the W5b corpus" in capsys.readouterr().err


def test_eval_without_acknowledgment_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    code = stage2.main(["eval", "--root", str(tmp_path)])
    assert code == stage2.EXIT_CONTRACT
    assert "--i-know-this-is-single-shot" in capsys.readouterr().err


# ── the full chain on tmp fixtures (real code path, no canon-data) ────────────


def _run_mini_chain(root: Path, corpus_dir: Path, labels_csv: Path, extra: list[str]) -> int:
    return stage2.main(
        [
            "run",
            "--corpus",
            str(corpus_dir),
            "--labels",
            str(labels_csv),
            "--root",
            str(root),
            "--expect-corpus-fingerprint",
            _mini_fingerprint(corpus_dir),
            "--candidate",
            "d",
            *extra,
        ]
    )


def test_full_chain_real_mode_and_skip_flags(tmp_path: Path) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(tmp_path, {"P1": 66, "N1": 66}, {"P1": "duplicate", "N1": "not-duplicate"})
    root = tmp_path / "out"
    assert _run_mini_chain(root, corpus_dir, labels_csv, []) == 0
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["corpus"]["fingerprint_match"] is True
    assert report["labels"]["counts"] == {"duplicate": 66, "not-duplicate": 66}
    assert report["split"]["holdout_class_counts"] == {"duplicate": 20, "not-duplicate": 20}
    assert report["split"]["train_rows"] == 92
    assert report["select"]["winner"] == "d-boost"
    assert report["stages"] == {"ingest": "ran", "split": "ran", "train": "ran", "select": "ran", "export": "ran"}
    artifact = Path(report["artifact"]["artifact"])
    assert artifact.exists() and artifact.stat().st_size <= 5 * 1024 * 1024

    # re-run with every stage skipped — the glue supports re-runs
    assert _run_mini_chain(root, corpus_dir, labels_csv, ["--skip-ingest", "--skip-split", "--skip-train", "--skip-select", "--skip-export"]) == 0
    report2 = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report2["stages"] == {
        "ingest": "skipped",
        "split": "skipped",
        "train": "skipped",
        "select": "skipped",
        "export": "skipped",
    }
    assert report2["select"]["winner"] == report["select"]["winner"]
    assert report2["artifact"]["sha256"] == report["artifact"]["sha256"]


def test_dry_run_smoke_never_touches_canon_data(tmp_path: Path) -> None:
    synth = _mini_synth(tmp_path)
    root = tmp_path / "dry"
    assert stage2.main(["run", "--dry-run", "--synth-source", str(synth), "--root", str(root)]) == 0
    # the stand-in is canon-shaped and lives under the dry root
    assert (root / "corpus" / "pairs.jsonl").exists()
    assert (root / "labeling" / "labels.csv").exists()
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["labels"]["counts"] == {"duplicate": 66, "not-duplicate": 99}
    # ceil(0.3·66) + ceil(0.3·66) + ceil(0.3·20) + ceil(0.3·13) = 20+20+6+4
    assert report["split"]["holdout_rows"] == 50
    assert report["select"]["winner"] == "d-boost"
    assert Path(report["artifact"]["artifact"]).exists()


def test_single_shot_eval_then_refusal_on_repeat(tmp_path: Path) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(tmp_path, {"P1": 66, "N1": 66}, {"P1": "duplicate", "N1": "not-duplicate"})
    root = tmp_path / "out"
    assert _run_mini_chain(root, corpus_dir, labels_csv, []) == 0
    eval_argv = ["eval", "--root", str(root), "--i-know-this-is-single-shot"]
    assert stage2.main(eval_argv) == 0
    # the append-only run log refuses the second single shot over the same corpus
    assert stage2.main(eval_argv) == stage2.EXIT_REFUSED
    entries = [json.loads(line) for line in (root / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["corpus_fingerprint"] == _mini_fingerprint(corpus_dir)


def test_skip_ingest_requires_skip_split(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(
        tmp_path, {"P1": 66, "N1": 66}, {"P1": "duplicate", "N1": "not-duplicate"}
    )
    code = stage2.main(
        [
            "run",
            "--corpus",
            str(corpus_dir),
            "--labels",
            str(labels_csv),
            "--root",
            str(tmp_path / "out"),
            "--skip-ingest",  # without --skip-split: the split re-derives from labels
        ]
    )
    assert code == stage2.EXIT_CONTRACT
    assert "--skip-ingest requires --skip-split" in capsys.readouterr().err


def test_vectors_needed_stop_for_explicit_n(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    corpus_dir, labels_csv = _write_canon_fixture(tmp_path, {"P1": 66, "N1": 66}, {"P1": "duplicate", "N1": "not-duplicate"})
    code = stage2.main(
        [
            "run",
            "--corpus",
            str(corpus_dir),
            "--labels",
            str(labels_csv),
            "--root",
            str(tmp_path / "out"),
            "--expect-corpus-fingerprint",
            _mini_fingerprint(corpus_dir),
            "--candidate",
            "both",
        ]
    )
    assert code == stage2.EXIT_CONTRACT
    captured = capsys.readouterr().err
    assert "vec_a/vec_b" in captured
    assert "s1_synth_similarity.py pattern" in captured
