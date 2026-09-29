"""Selection-protocol tests (A3b): the frozen CV loop and the honest
semantics of "N rides only on > 1 std repeat-spread superiority"."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cortex.select.cv import (
    CV_FOLDS,
    CV_SEEDS,
    CvReport,
    run_cv,
    select_candidate,
)

from synth import make_vectors


def _synth_matrix(n: int = 60):
    vectors, labels, _ = make_vectors(n)
    return np.asarray([v.values for v in vectors]), np.asarray(labels, dtype=np.int64)


def _identity_fn(features_train, labels_train, seed):
    """A 'model' that ignores training: reads the first feature (cos_target
    passthrough) — deterministic, seed-independent, fast."""
    return lambda x: np.clip(x[:, 0], 0.0, 1.0)


def _constant_fn(value: float):
    def train_fn(features_train, labels_train, seed):
        return lambda x: np.full(len(x), value)

    return train_fn


# ── frozen protocol constants ─────────────────────────────────────────────────


def test_protocol_constants() -> None:
    assert CV_FOLDS == 5
    assert CV_SEEDS == tuple(range(1, 21))


# ── run_cv ────────────────────────────────────────────────────────────────────


def test_run_cv_returns_four_numbers() -> None:
    features, labels = _synth_matrix()
    result = run_cv(_constant_fn(0.5), features, labels)
    assert len(result) == 4
    ba_mean, ba_std, brier_mean, brier_std = result
    assert 0.0 <= ba_mean <= 1.0 and ba_std >= 0.0
    assert 0.0 <= brier_mean <= 1.0 and brier_std >= 0.0


def test_run_cv_constant_predictor_scores_chance() -> None:
    features, labels = _synth_matrix()
    ba_mean, _, brier_mean, _ = run_cv(_constant_fn(0.5), features, labels)
    assert ba_mean == pytest.approx(0.5)
    assert brier_mean == pytest.approx(0.25)


def test_run_cv_reference_learner_scores_well() -> None:
    """A nearest-centroid reference learner (deterministic numpy, fit on
    the train folds ONLY) clears the synthetic corpus comfortably — this
    proves run_cv wires features/labels/predictions through the right
    slots; it is a wiring check, not a power benchmark."""

    def centroid_fn(f_train, y_train, seed):
        centroids = {c: f_train[y_train == c].mean(axis=0) for c in (0, 1)}
        scale = f_train.std(axis=0)
        scale[scale == 0] = 1.0

        def predict(x):
            distances = {
                c: (((x - centroids[c]) / scale) ** 2).sum(axis=1) for c in (0, 1)
            }
            return (distances[0] > distances[1]).astype(np.float64)

        return predict

    features, labels = _synth_matrix()
    ba_mean, ba_std, brier_mean, brier_std = run_cv(centroid_fn, features, labels)
    assert ba_mean >= 0.9
    assert brier_mean <= 0.15
    assert ba_std >= 0.0 and brier_std >= 0.0


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        (np.array([0, 1, 2, 0] * 15), "binary"),
        (np.array([1] * 60), "stratified"),
    ],
)
def test_run_cv_input_validation(labels, match) -> None:
    features = np.random.default_rng(0).normal(size=(60, 13))
    with pytest.raises(ValueError, match=match):
        run_cv(_identity_fn, features, labels)


def test_run_cv_predict_fn_contract_enforced() -> None:
    features, labels = _synth_matrix()

    def bad_shape_fn(f_train, y_train, seed):
        return lambda x: np.zeros(len(x) + 1)

    with pytest.raises(ValueError, match="shape"):
        run_cv(bad_shape_fn, features, labels)

    def bad_range_fn(f_train, y_train, seed):
        return lambda x: np.full(len(x), 1.5)

    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        run_cv(bad_range_fn, features, labels)


# ── select_candidate: the frozen rule ─────────────────────────────────────────


def _report(candidate: str, ba_mean, ba_std, brier_mean, brier_std) -> CvReport:
    return CvReport(candidate, f"{candidate}-cfg", ba_mean, ba_std, brier_mean, brier_std)


D = _report("d-boost", 0.80, 0.02, 0.30, 0.02)


def test_missing_n_selects_d() -> None:
    verdict = select_candidate(D, None)
    assert verdict.winner == "d-boost"
    assert verdict.margin_stds == math.inf


def test_n_wins_only_beyond_one_std_on_both_metrics() -> None:
    # +0.05 mean over 0.02 spread = 2.5 std on BOTH metrics → N
    n = _report("n-head", 0.85, 0.02, 0.25, 0.02)
    verdict = select_candidate(D, n)
    assert verdict.winner == "n-head"
    assert verdict.margin_stds == pytest.approx(2.5)


def test_n_within_margin_selects_d() -> None:
    # +0.03 mean over max(0.02, 0.03) spread = 1.0 std — NOT > 1 std → D
    n = _report("n-head", 0.83, 0.03, 0.27, 0.01)
    assert select_candidate(D, n).winner == "d-boost"


def test_n_wins_one_metric_only_selects_d() -> None:
    n = _report("n-head", 0.85, 0.01, 0.29, 0.02)  # BA 5 std better, Brier worse
    assert select_candidate(D, n).winner == "d-boost"
    n2 = _report("n-head", 0.81, 0.02, 0.24, 0.02)  # Brier better, BA within margin
    assert select_candidate(D, n2).winner == "d-boost"


def test_exact_tie_selects_d() -> None:
    n = _report("n-head", 0.80, 0.02, 0.30, 0.02)
    verdict = select_candidate(D, n)
    assert verdict.winner == "d-boost"
    assert verdict.margin_stds == 0.0


def test_zero_spread_strict_improvement_is_decisive() -> None:
    d_zero = _report("d-boost", 0.80, 0.0, 0.30, 0.0)
    n = _report("n-head", 0.85, 0.0, 0.25, 0.0)
    verdict = select_candidate(d_zero, n)
    assert verdict.winner == "n-head"
    assert verdict.margin_stds == math.inf


def test_margin_uses_conservative_max_spread() -> None:
    # spread = max(0.02, 0.01) = 0.02; margin 0.04 = exactly 2.0 stds
    n = _report("n-head", 0.84, 0.01, 0.26, 0.02)
    verdict = select_candidate(D, n)
    assert verdict.winner == "n-head"
    assert verdict.margin_stds == pytest.approx(2.0)
