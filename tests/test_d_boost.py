"""Candidate D contract tests (A3b): training on programmatic synthetic
data (no store), determinism, validation, calibration, save/load and the
ONNX export gate (onnxruntime round-trip)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synth import make_vectors

from cortex.artifacts import (
    CANDIDATE_D,
    MODEL_NAME,
    build_metadata_props,
    sha256_file,
)
from cortex.candidates.d_boost import GRID_D, DBoostModel
from cortex.features.pair import FEATURE_NAMES

VECTORS, LABELS, _ROWS = make_vectors(60)


def _fitted(config_idx: int = 1, calibrate: bool = True) -> DBoostModel:
    model = DBoostModel()
    model.train(VECTORS, LABELS, GRID_D[config_idx], calibrate=calibrate)
    return model


# ── frozen grid ───────────────────────────────────────────────────────────────


def test_grid_frozen_and_bounded() -> None:
    assert 1 <= len(GRID_D) <= 8
    assert len({c.name for c in GRID_D}) == len(GRID_D)
    core = [c for c in GRID_D if not c.field_cosines]
    ablation = [c for c in GRID_D if c.field_cosines]
    assert core and ablation, (
        "the ADR-mandated with/without-field-cosines axis must exist"
    )


# ── training and inference ────────────────────────────────────────────────────


def test_train_predict_contract() -> None:
    model = _fitted()
    probs = model.predict_proba(VECTORS)
    assert probs.shape == (len(VECTORS),)
    assert probs.dtype == np.float64
    assert ((0.0 <= probs) & (probs <= 1.0)).all()


def test_training_separates_synthetic_classes() -> None:
    """The synthetic corpus is separable by construction (char-divergence
    under high cosine) — a sane D fit must clear it almost perfectly."""
    probs = _fitted().predict_proba(VECTORS)
    acc = ((probs >= 0.5).astype(int) == np.asarray(LABELS)).mean()
    assert acc >= 0.95


def test_fit_is_deterministic() -> None:
    first = _fitted().predict_proba(VECTORS)
    second = _fitted().predict_proba(VECTORS)
    assert np.array_equal(first, second)


def test_calibration_moves_probabilities_monotonically_only() -> None:
    """Platt is a monotone transform of the margin: calibrated and identity
    readouts must agree in ranking exactly."""
    calibrated = _fitted(calibrate=True).predict_proba(VECTORS)
    identity = _fitted(calibrate=False).predict_proba(VECTORS)
    assert np.array_equal(np.argsort(calibrated), np.argsort(identity))


@pytest.mark.parametrize(
    ("vectors", "labels", "config", "match"),
    [
        ([], [1], GRID_D[0], "at least one"),
        (VECTORS, [1] * len(VECTORS), GRID_D[0], "both classes"),
        (VECTORS, LABELS[:-1], GRID_D[0], "labels length"),
        (VECTORS, [2] + LABELS[1:], GRID_D[0], "binary"),
    ],
)
def test_train_validation(vectors, labels, config, match) -> None:
    with pytest.raises(ValueError, match=match):
        DBoostModel().train(vectors, labels, config)


def test_unfitted_failures() -> None:
    model = DBoostModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(VECTORS)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.export_onnx(Path("unused.onnx"))


def test_predict_rejects_wrong_feature_contract() -> None:
    from cortex.features.pair import FeatureVector

    model = _fitted()
    bad = [FeatureVector(v.names[:-1], v.values[:-1]) for v in VECTORS[:2]]
    with pytest.raises(ValueError, match="features"):
        model.predict_proba(bad)


# ── save/load roundtrip (dev format, no pickle) ──────────────────────────────


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = _fitted()
    model.save(tmp_path)
    loaded = DBoostModel.load(tmp_path)
    assert np.allclose(loaded.predict_proba(VECTORS), model.predict_proba(VECTORS))
    assert (tmp_path / "booster.txt").exists() and (tmp_path / "meta.json").exists()


# ── ONNX export (inference-v1.md §4, variant D) ───────────────────────────────


def _props() -> dict[str, str]:
    return build_metadata_props(
        embedder_pin="nano:sha256:" + "ab" * 32,
        corpus_fingerprint="cd" * 32,
        trained_at="2026-09-29T00:00:00+00:00",
        candidate=CANDIDATE_D,
        feature_names=FEATURE_NAMES,
    )


def test_export_onnx_roundtrip(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")

    model = _fitted()
    out = tmp_path / "model.onnx"
    model.export_onnx(out, metadata_props=_props())
    assert out.stat().st_size <= 5 * 1024 * 1024

    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["name"] == MODEL_NAME
    assert meta["candidate"] == CANDIDATE_D
    assert tuple(meta["features"].split("\n")) == FEATURE_NAMES

    # graph output: float32[1] per single pair, equal to library probability
    for index in (0, 1, len(VECTORS) - 1):
        row = np.asarray(VECTORS[index].values, dtype=np.float32)
        (probability,) = session.run(None, {"features": row})
        assert probability.shape == (1,)
        assert probability.dtype == np.float32
        assert float(probability[0]) == pytest.approx(
            float(model.predict_proba([VECTORS[index]])[0]), abs=1e-5
        )


def test_export_without_metadata_yields_bare_graph(tmp_path: Path) -> None:
    onnxruntime = pytest.importorskip("onnxruntime")

    out = tmp_path / "bare.onnx"
    _fitted().export_onnx(out)  # no metadata_props — bare graph still valid
    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_modelmeta().custom_metadata_map == {}
    assert len(sha256_file(out)) == 64


def test_grid_point_with_field_cosines_needs_extended_vectors() -> None:
    fc_config = next(c for c in GRID_D if c.field_cosines)
    with pytest.raises(ValueError, match="field cosines"):
        DBoostModel().train(VECTORS, LABELS, fc_config)
