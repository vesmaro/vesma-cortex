"""Eval-runner contract tests (A3b): append-only run log with single-shot
refusal, typed model-output validation, hand-computed prereg metrics,
baseline through the same code path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cortex.eval.runner import (
    BASELINE_THRESHOLD,
    HoldoutPair,
    RunLog,
    RunLogEntry,
    RunLogRefusalError,
    load_holdout_pairs,
    run_baseline,
    run_single_shot,
)


def _holdout() -> list[HoldoutPair]:
    """Four pairs, two per class — every metric below is hand-computed."""
    return [
        HoldoutPair("p1", "duplicate", 0.95),
        HoldoutPair("p2", "not-duplicate", 0.50),
        HoldoutPair("p3", "duplicate", 0.95),
        HoldoutPair("p4", "not-duplicate", 0.50),
    ]


class _FixedModel:
    """predict_proba → [0.9, 0.2, 0.4, 0.6] (hand-computed case)."""

    def predict_proba(self, pairs):
        return [0.9, 0.2, 0.4, 0.6]


# ── run log: append-only + single-shot refusal ────────────────────────────────


def _entry(corpus: str) -> RunLogEntry:
    return RunLogEntry(
        run_at="2026-09-29T00:00:00+00:00",
        weights_sha256="w" * 64,
        corpus_fingerprint=corpus,
        label_fingerprint="l" * 64,
    )


def test_run_log_append_and_refusal(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "runs" / "run_log.jsonl")
    log.append(_entry("c1"))
    assert log.contains_corpus("c1")
    with pytest.raises(RunLogRefusalError, match="single-shot"):
        log.append(_entry("c1"))
    log.append(_entry("c2"))  # a different corpus is a legitimate next run
    lines = (
        (tmp_path / "runs" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(lines) == 2
    assert set(json.loads(lines[0])) == {
        "run_at",
        "weights_sha256",
        "corpus_fingerprint",
        "label_fingerprint",
    }


def test_run_log_refusal_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "run_log.jsonl"
    RunLog(path).append(_entry("c1"))
    with pytest.raises(RunLogRefusalError):
        RunLog(path).append(_entry("c1"))


# ── run_single_shot: hand-computed metrics ────────────────────────────────────


def test_metrics_hand_computed(tmp_path: Path) -> None:
    # preds at 0.5 cut: [1, 0, 0, 1] vs truth [1, 0, 1, 0]
    # sensitivity 1/2, specificity 1/2, balanced acc 1/2
    # brier = mean((.9-1)^2, .2^2, (.4-1)^2, .6^2) = (0.01+0.04+0.36+0.36)/4
    report = run_single_shot(
        _FixedModel(),
        _holdout(),
        run_log=RunLog(tmp_path / "rl.jsonl"),
        weights_sha256="w" * 64,
        corpus_fingerprint="c" * 64,
    )
    assert report.typified_completeness == 1.0
    assert report.sensitivity == pytest.approx(0.5)
    assert report.specificity == pytest.approx(0.5)
    assert report.balanced_accuracy == pytest.approx(0.5)
    assert report.brier == pytest.approx(0.1925)
    # baseline: sims [0.95, 0.5, 0.95, 0.5] → preds == truth → perfect
    assert report.baseline_balanced_accuracy == pytest.approx(1.0)
    assert report.record_quality_completeness == 1.0
    assert report.weights_sha256 == "w" * 64


def test_baseline_wrong_on_both_classes_scores_zero() -> None:
    """Baseline misses both ways: a high-cosine non-duplicate (FP) and a
    low-similarity duplicate (FN) — one wrong verdict per class → 0.0."""
    pairs = [
        HoldoutPair("p1", "not-duplicate", 0.99),  # FP
        HoldoutPair("p2", "duplicate", 0.10),  # FN
    ]
    assert run_baseline(pairs) == pytest.approx(0.0)


def test_single_shot_appends_and_refuses_second_run(tmp_path: Path) -> None:
    log_path = tmp_path / "rl.jsonl"
    kwargs = dict(
        run_log=RunLog(log_path), weights_sha256="w" * 64, corpus_fingerprint="c" * 64
    )
    run_single_shot(_FixedModel(), _holdout(), **kwargs)
    with pytest.raises(RunLogRefusalError):
        run_single_shot(_FixedModel(), _holdout(), **kwargs)


def test_single_shot_requires_corpus_fingerprint(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="corpus_fingerprint"):
        run_single_shot(
            _FixedModel(),
            _holdout(),
            run_log=RunLog(tmp_path / "rl.jsonl"),
            weights_sha256="w" * 64,
            corpus_fingerprint="",
        )


def test_model_output_contract_enforced(tmp_path: Path) -> None:
    base = dict(
        run_log=RunLog(tmp_path / "rl.jsonl"),
        weights_sha256="w" * 64,
        corpus_fingerprint="c" * 64,
    )

    class _Short:
        def predict_proba(self, pairs):
            return [0.5]

    with pytest.raises(ValueError, match="typified"):
        run_single_shot(_Short(), _holdout(), **base)

    class _OutOfRange:
        def predict_proba(self, pairs):
            return [1.5, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        run_single_shot(_OutOfRange(), _holdout(), **base)

    class _NaN:
        def predict_proba(self, pairs):
            return [float("nan"), 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        run_single_shot(_NaN(), _holdout(), **base)


def test_bad_labels_fail_loud(tmp_path: Path) -> None:
    base = dict(
        run_log=RunLog(tmp_path / "rl.jsonl"),
        weights_sha256="w" * 64,
        corpus_fingerprint="c" * 64,
    )
    with pytest.raises(ValueError, match="disputed"):
        run_single_shot(
            _FixedModel(),
            [HoldoutPair("p1", "disputed", 0.9)],
            **base,
        )
    with pytest.raises(ValueError, match="at least one"):
        run_single_shot(_FixedModel(), [], **base)


# ── holdout file loading (data-contract §3/§5.5) ──────────────────────────────


def _write_holdout(tmp_path: Path, labels: dict[str, str]) -> Path:
    directory = tmp_path / "labels-holdout" / "h"
    directory.mkdir(parents=True)
    pairs = [
        {
            "pair_id": "p1",
            "similarity": 0.95,
            "stratum": "P1",
            "record": {},
            "candidate": {},
        },
        {
            "pair_id": "p2",
            "similarity": 0.50,
            "stratum": "N1",
            "record": {},
            "candidate": {},
        },
    ]
    (directory / "pairs.jsonl").write_text(
        "\n".join(json.dumps(r) for r in pairs), encoding="utf-8"
    )
    (directory / "labels.jsonl").write_text(
        "\n".join(
            json.dumps({"pair_id": pid, "label": label})
            for pid, label in labels.items()
        ),
        encoding="utf-8",
    )
    return directory


def test_load_holdout_pairs_happy(tmp_path: Path) -> None:
    directory = _write_holdout(tmp_path, {"p1": "duplicate", "p2": "not-duplicate"})
    pairs = load_holdout_pairs(directory / "pairs.jsonl", directory / "labels.jsonl")
    assert [p.pair_id for p in pairs] == ["p1", "p2"]
    assert pairs[0].label == "duplicate" and pairs[0].similarity == pytest.approx(0.95)


def test_load_holdout_pairs_missing_label(tmp_path: Path) -> None:
    directory = _write_holdout(tmp_path, {"p1": "duplicate"})
    with pytest.raises(ValueError, match="no label"):
        load_holdout_pairs(directory / "pairs.jsonl", directory / "labels.jsonl")


def test_load_holdout_pairs_disputed_refused(tmp_path: Path) -> None:
    directory = _write_holdout(tmp_path, {"p1": "disputed", "p2": "not-duplicate"})
    with pytest.raises(ValueError, match="disputed"):
        load_holdout_pairs(directory / "pairs.jsonl", directory / "labels.jsonl")


def test_baseline_threshold_constant() -> None:
    assert BASELINE_THRESHOLD == 0.92
