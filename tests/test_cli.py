"""CLI command contracts (A3b): the implemented pipeline commands run for
real on programmatic synthetic data — pretrain, train, select,
export-artifact, eval — with the documented exit codes (0/2/3/4). No
store, no network; everything under tmp_path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synth import make_pair_rows, make_records, write_manifest

from cortex.cli.main import RUN_REFUSED_EXIT, main
from cortex.data.fingerprints import corpus_fingerprint, manifest_bytes, pair_sha256

EMBEDDER_PIN = "nano:sha256:" + "ab" * 32


@pytest.fixture()
def records_path(tmp_path: Path) -> Path:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "title": r.title,
                    "body": r.body,
                    "tags": list(r.tags),
                    "language": r.language,
                    "record_type": r.record_type,
                },
                ensure_ascii=False,
            )
            for r in make_records(12)
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def train_manifest(tmp_path: Path) -> Path:
    return write_manifest(tmp_path / "train.jsonl", make_pair_rows(60))


@pytest.fixture()
def holdout_dir(tmp_path: Path) -> Path:
    """Holdout files OUTSIDE any train tree (physical isolation, §7)."""
    directory = tmp_path / "labels-holdout" / "h1"
    directory.mkdir(parents=True)
    rows = make_pair_rows(30)
    write_manifest(directory / "pairs.jsonl", rows)
    (directory / "labels.jsonl").write_text(
        "\n".join(
            json.dumps({"pair_id": r["pair_id"], "label": r["label"]}) for r in rows
        ),
        encoding="utf-8",
    )
    return directory


def _train_fingerprint(manifest: Path) -> str:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
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
    return corpus_fingerprint(manifest_bytes(entries))


# ── pretrain ──────────────────────────────────────────────────────────────────


def test_pretrain_writes_data_contract_shape(
    records_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "pretrain" / "pairs.jsonl"
    code = main(
        ["pretrain", "--corpus", str(records_path), "--seed", "7", "--out", str(out)]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["seed"] == 7
    assert summary["weak_positive"] > 0 and summary["hard_negative"] > 0

    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        assert "label" not in row  # pretrain manifest carries NO labels (§4)
        assert row["source"] in {"weak-positive", "hard-negative"}
        assert row["transform"]
        assert row["similarity"] == 1.0  # self-pair convention
        for side in ("record", "candidate"):
            assert set(row[side]) == {
                "title",
                "body",
                "tags",
                "language",
                "record_type",
            }
    # manifest.txt + fingerprint follow the frozen scheme
    manifest = (out.parent / "manifest.txt").read_bytes()
    assert manifest == manifest_bytes(
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
    assert summary["corpus_fingerprint"] == corpus_fingerprint(manifest)


def test_pretrain_seed_determinism(records_path: Path, tmp_path: Path) -> None:
    outs = []
    for run in (1, 2):
        out = tmp_path / f"p{run}" / "pairs.jsonl"
        assert (
            main(
                [
                    "pretrain",
                    "--corpus",
                    str(records_path),
                    "--seed",
                    "5",
                    "--out",
                    str(out),
                ]
            )
            == 0
        )
        outs.append(out.read_text(encoding="utf-8"))
    assert outs[0] == outs[1]


# ── train ─────────────────────────────────────────────────────────────────────


def test_train_d_records_provenance(train_manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "models"
    code = main(
        [
            "train",
            "--train-manifest",
            str(train_manifest),
            "--candidate",
            "d",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    meta = json.loads((out / "d-boost" / "meta.json").read_text(encoding="utf-8"))
    assert meta["candidate"] == "d-boost"
    assert meta["corpus_fingerprint"] == _train_fingerprint(train_manifest)
    assert meta["n_pairs"] == 60


def test_train_n_needs_train_env_or_vectors(
    train_manifest: Path, tmp_path: Path
) -> None:
    """No torch and/or no vec_a/vec_b in this manifest → exit 3 (either way
    the n candidate is honestly unrunnable, never silently skipped)."""
    code = main(
        [
            "train",
            "--train-manifest",
            str(train_manifest),
            "--candidate",
            "n",
            "--out",
            str(tmp_path / "m"),
        ]
    )
    assert code == 3


def test_train_unknown_config_refused(train_manifest: Path, tmp_path: Path) -> None:
    code = main(
        [
            "train",
            "--train-manifest",
            str(train_manifest),
            "--candidate",
            "d",
            "--config",
            "no-such-point",
            "--out",
            str(tmp_path / "m"),
        ]
    )
    assert code == 2


def test_train_disputed_rows_skipped_with_counter(tmp_path: Path) -> None:
    rows = make_pair_rows(10)
    rows[0]["label"] = "disputed"
    manifest = write_manifest(tmp_path / "train.jsonl", rows)
    code = main(
        [
            "train",
            "--train-manifest",
            str(manifest),
            "--candidate",
            "d",
            "--out",
            str(tmp_path / "m"),
        ]
    )
    assert code == 0  # 9 usable rows remain; a disputed minority never blocks


# ── select ────────────────────────────────────────────────────────────────────


def test_select_reports_verdict_and_skips_field_cosines(
    train_manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    out = tmp_path / "select.json"
    code = main(
        [
            "select",
            "--train-manifest",
            str(train_manifest),
            "--candidate",
            "d",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "field-cosines grid points skipped" in captured.err

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"]["winner"] == "d-boost"
    d_configs = {r["config_name"] for r in payload["reports"]}
    assert "d-l7-lr005" in d_configs and "d-l15-lr005" in d_configs
    for report in payload["reports"]:
        assert report["candidate"] == "d-boost"
        assert 0.0 <= report["balanced_accuracy_mean"] <= 1.0
        assert 0.0 <= report["brier_mean"] <= 1.0


def test_select_torch_absent_still_returns_d_verdict(
    train_manifest: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    code = main(
        [
            "select",
            "--train-manifest",
            str(train_manifest),
            "--candidate",
            "both",
            "--out",
            str(tmp_path / "s.json"),
        ]
    )
    assert code == 0
    err = capsys.readouterr().err
    # this manifest carries no vec_a/vec_b: N is skipped loudly (torch
    # absent OR vectors absent — both are honest env gaps), D still rides
    assert "torch absent" in err or "vec_a/vec_b" in err
    payload = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert payload["verdict"]["winner"] == "d-boost"


# ── export-artifact ───────────────────────────────────────────────────────────


@pytest.fixture()
def trained_d(train_manifest: Path, tmp_path: Path) -> Path:
    out = tmp_path / "models"
    assert (
        main(
            [
                "train",
                "--train-manifest",
                str(train_manifest),
                "--candidate",
                "d",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    return out / "d-boost"


def test_export_artifact_writes_onnx_and_manifest(
    trained_d: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    from cortex.artifacts import MODEL_NAME, sha256_file

    out = tmp_path / "model.onnx"
    code = main(
        [
            "export-artifact",
            "--model",
            str(trained_d),
            "--out",
            str(out),
            "--embedder-pin",
            EMBEDDER_PIN,
        ]
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["sha256"] == sha256_file(out)

    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["name"] == MODEL_NAME
    assert meta["embedder_pin"] == EMBEDDER_PIN
    assert meta["corpus_fingerprint"] == _train_fingerprint(
        trained_d.parent.parent / "train.jsonl"
    )

    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["sha256"] == sha256_file(out)
    assert manifest["embedder_pin"] == EMBEDDER_PIN
    assert manifest["size_bytes"] == out.stat().st_size


def test_export_artifact_requires_pin(trained_d: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "export-artifact",
                "--model",
                str(trained_d),
                "--out",
                str(tmp_path / "m.onnx"),
            ]
        )
    assert excinfo.value.code == 2  # argparse: --embedder-pin is required


def test_export_artifact_refuses_fingerprintless_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "bare"
    model_dir.mkdir()
    (model_dir / "meta.json").write_text(
        json.dumps({"candidate": "d-boost"}), encoding="utf-8"
    )
    code = main(
        [
            "export-artifact",
            "--model",
            str(model_dir),
            "--out",
            str(tmp_path / "m.onnx"),
            "--embedder-pin",
            EMBEDDER_PIN,
        ]
    )
    assert code == 2  # fingerprint-less export is refused, not silently zeroed


# ── eval: single-shot + refusal ───────────────────────────────────────────────


def test_eval_single_shot_then_refused(
    trained_d: Path, holdout_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    artifact = tmp_path / "model.onnx"
    assert (
        main(
            [
                "export-artifact",
                "--model",
                str(trained_d),
                "--out",
                str(artifact),
                "--embedder-pin",
                EMBEDDER_PIN,
            ]
        )
        == 0
    )
    capsys.readouterr()
    run_log = tmp_path / "run_log.jsonl"

    code = main(
        [
            "eval",
            "--artifact",
            str(artifact),
            "--holdout",
            str(holdout_dir),
            "--run-log",
            str(run_log),
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    for key in (
        "typified_completeness",
        "sensitivity",
        "specificity",
        "brier",
        "balanced_accuracy",
        "record_quality_completeness",
        "baseline_balanced_accuracy",
    ):
        assert key in report

    code = main(
        [
            "eval",
            "--artifact",
            str(artifact),
            "--holdout",
            str(holdout_dir),
            "--run-log",
            str(run_log),
        ]
    )
    assert code == RUN_REFUSED_EXIT
    assert "refused" in capsys.readouterr().err

    # the run log is append-only metadata: exactly one committed line
    lines = run_log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["corpus_fingerprint"] and entry["weights_sha256"]


def test_eval_rejects_foreign_artifact(
    trained_d: Path, holdout_dir: Path, tmp_path: Path
) -> None:
    """An ONNX file without the vesma-cortex identity is refused loudly."""
    from cortex.candidates.d_boost import DBoostModel

    model = DBoostModel.load(trained_d)
    bare = tmp_path / "bare.onnx"
    model.export_onnx(bare)  # no metadata_props → no vesma-cortex identity
    code = main(
        [
            "eval",
            "--artifact",
            str(bare),
            "--holdout",
            str(holdout_dir),
            "--run-log",
            str(tmp_path / "rl.jsonl"),
        ]
    )
    assert code == 2


# ── export-corpus: implemented in A2 (store export, prereg hygiene) ──────────


def test_export_corpus_missing_store_fails_loud(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A2 landed the real exporter: a missing store is a loud usage error
    naming the problem — never fake success, never a stub claim."""
    code = main(
        ["export-corpus", "--store-uri", "file:x?mode=ro", "--out", str(tmp_path / "c")]
    )
    assert code == 2
    stderr = capsys.readouterr().err
    assert stderr.strip()
    assert "not implemented" not in stderr.lower()


def test_export_corpus_requires_single_store_source(tmp_path: Path) -> None:
    """--store-path and --store-uri are mutually exclusive (read-only both)."""
    code = main(
        [
            "export-corpus",
            "--store-path",
            str(tmp_path),
            "--store-uri",
            "file:x?mode=ro",
            "--out",
            str(tmp_path / "c"),
        ]
    )
    assert code == 2
